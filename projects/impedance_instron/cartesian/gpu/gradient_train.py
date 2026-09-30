# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Train a reference-conditioned coefficient model on the declared 100 stance set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .adjoint_batch import BatchAdjoint
from .backprop_report import render_report
from .benchmark import _plain
from .conditioned_controller import Controller
from .provenance import source_snapshot
from .shared_train import _load_dataset, _load_fit


def _write(path, value):
    """Write strict JSON experiment evidence."""
    path.write_text(json.dumps(_plain(value), indent=2, allow_nan=False) + "\n")


def _summarize(records):
    """Summarize complete stances without awarding truncated rollouts a fit score."""
    valid = [record for record in records if record["complete"]]
    return {
        "count": len(records),
        "complete_count": len(valid),
        "mean_loss": float(np.mean([record["loss"] for record in valid])) if len(valid) == len(records) else None,
        "native_all6_pass_count": sum(record["native_all6_pass"] for record in records),
        "mean_rmse": np.mean([record["nativechannelrmse"] for record in valid], axis=0) if valid else None,
    }


def _score(batch, controller, members, thresholds):
    """Evaluate held-fixed coefficients on each native reference using the reusable graph."""
    records = []
    for offset in range(0, len(members), batch.capacity):
        selected = members[offset : offset + batch.capacity]
        references = [member["reference_data"] for member in selected]
        coefficients, _ = controller.predict(references)
        batch.set_batch(references, coefficients)
        result = batch.forward_only()
        for i, member in enumerate(selected):
            valid = bool(result["valid"][i])
            rmse = np.asarray(result["rmse"][i], dtype=np.float64)
            records.append(
                {
                    "stance_id": member["id"],
                    "trial": member["trial"],
                    "split": member["split"],
                    "complete": valid,
                    "loss": float(result["losses"][i]) if valid else None,
                    "nativechannelrmse": rmse if valid else None,
                    "hip_rmse_m": rmse[:2] if valid else None,
                    "joint_rmse_rad": rmse[2:4] if valid else None,
                    "force_rmse_n": rmse[4:] if valid else None,
                    "native_all6_pass": valid and bool(np.all(rmse <= thresholds)),
                }
            )
    return records


def train(
    dataset: Path,
    fit_directory: Path,
    output: Path,
    *,
    qualification: Path,
    epochs: int = 10,
    capacity: int = 4,
    learning_rate: float = 0.003,
    seed: int = 17,
):
    """Train on balanced batches; select checkpoints using training stances only."""
    if output.exists():
        raise FileExistsError(output)
    if capacity != 4 or epochs < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Use positive epochs/rate and the qualified four-world capacity")
    audit = json.loads(qualification.read_text())
    if not audit.get("qualified", False):
        raise ValueError("A passing heterogeneous physics/predictor experiment is required")
    if audit.get("source_sha256") != source_snapshot():
        raise ValueError("Physics changed after batch qualification")
    if audit.get("dataset_manifest_sha256") != hashlib.sha256(
        (dataset / "manifest.json").read_bytes()
    ).hexdigest() or audit.get("fit_directory") != str(fit_directory.resolve()):
        raise ValueError("The gradient experiment uses a different dataset or physics template")
    if (
        audit.get("controller_source_sha256")
        != hashlib.sha256(Path(__file__).with_name("conditioned_controller.py").read_bytes()).hexdigest()
    ):
        raise ValueError("The coefficient predictor changed after gradient qualification")
    for name, digest in audit["adjoint_sources_sha256"].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"The gradient backend changed after qualification: {name}")
    _, profile, _, config, settings, shoe, _ = _load_fit(fit_directory)
    manifest, _, _, all_members = _load_dataset(dataset, None, None)
    training = [member for member in all_members if member["split"] == "train"]
    evaluation = [member for member in all_members if member["split"] == "eval"]
    if len(training) != 100 or len(evaluation) != 10:
        raise ValueError("This experiment requires all 100 training and 10 evaluation stances")
    trials = {trial: [member for member in training if member["trial"] == trial] for trial in ("FR3_1", "FR3_2")}
    if any(len(members) != 50 for members in trials.values()):
        raise ValueError("Training must contain fifty stances from each trial")
    output.mkdir(parents=True)
    started = perf_counter()
    references = [member["reference_data"] for member in training]
    controller = Controller(references, profile, settings.parameter_scale, seed=seed)
    thresholds = np.asarray(
        [settings.hip_tolerance_m] * 2 + [settings.joint_tolerance_rad] * 2 + [settings.force_tolerance_n] * 2
    )
    batch = BatchAdjoint(fit_directory, dataset, capacity=capacity)
    warm_references = references[:capacity]
    warm_coefficients, _ = controller.predict(warm_references)
    batch.set_batch(warm_references, warm_coefficients)
    batch.capture()
    initial = _score(batch, controller, training + evaluation, thresholds)
    initial_train = _summarize([record for record in initial if record["split"] == "train"])
    initial_eval = _summarize([record for record in initial if record["split"] == "eval"])
    if initial_train["complete_count"] != 100 or initial_eval["complete_count"] != 10:
        raise RuntimeError("The nominal initial controller must complete every declared stance")
    setup_s = perf_counter() - started
    metadata = {
        "dataset": str(dataset.resolve()),
        "dataset_manifest_sha256": hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest(),
        "fit_directory": str(fit_directory.resolve()),
        "training_ids": [member["id"] for member in training],
        "evaluation_ids": [member["id"] for member in evaluation],
        "feature_statistics_training_only": True,
        "features": "nominal spline72, initial q/v10, duration, geometry, desired native GRF at12 phases24; train-only standardization/PCA16",
        "controller": "PCA16 -> tanh32 ->72 normalized stance-specific corrections; exact convex spline projection",
        "initialization": "zero decoder; no fitted controller weights or training labels",
        "simulation_config": vars(config),
        "fit_config": vars(settings),
        "shoe_sha256": shoe["sha256"],
        "seed": seed,
    }
    controller.save(output / "initial_controller.npz", metadata)
    _write(output / "initial_metrics.json", initial)
    best = controller.snapshot()
    best_loss = initial_train["mean_loss"]
    best_update = 0
    history = [{"iteration": 0, "loss": best_loss}]
    batches = []
    rng = np.random.default_rng(seed)
    selected_count = {member["id"]: 0 for member in training}
    active_per_trial = capacity // 2
    search_started = perf_counter()
    iteration = 0
    for epoch in range(epochs):
        order = {trial: rng.permutation(50) for trial in trials}
        for offset in range(0, 50, active_per_trial):
            iteration += 1
            selected = [
                members[index]
                for trial, members in trials.items()
                for index in order[trial][offset : offset + active_per_trial]
            ]
            for member in selected:
                selected_count[member["id"]] += 1
            references = [member["reference_data"] for member in selected]
            prediction_started = perf_counter()
            coefficients, cache = controller.predict(references)
            predictor_s = perf_counter() - prediction_started
            batch.set_batch(references, coefficients)
            result = batch.value_and_grad()
            recovered = False
            if not np.all(result["valid"]) or not np.isfinite(result["gradients"]).all():
                controller.restore(best)
                coefficients, cache = controller.predict(references)
                batch.set_batch(references, coefficients)
                result = batch.value_and_grad()
                recovered = True
                if not np.all(result["valid"]) or not np.isfinite(result["gradients"]).all():
                    raise RuntimeError("The globally validated checkpoint has an invalid rollout/gradient")
            mean_before = float(np.mean(result["losses"]))
            gradients = controller.backward(result["gradients"], cache)
            snapshot = controller.snapshot()
            trials_log = []
            accepted = False
            rate = learning_rate
            trial_snapshot = snapshot
            momentum_reset = False
            for _ in range(8):
                controller.restore(trial_snapshot)
                gradient_record = controller.apply_adam(gradients, learning_rate=rate)
                slope = float(
                    sum(
                        np.sum(gradients[name] * (value - snapshot["parameters"][name]))
                        for name, value in controller.parameters.items()
                    )
                )
                if slope >= 0 and not momentum_reset:
                    controller.restore(snapshot)
                    controller.first = {name: np.zeros_like(value) for name, value in controller.first.items()}
                    controller.second = {name: np.zeros_like(value) for name, value in controller.second.items()}
                    controller.updates = 0
                    trial_snapshot = controller.snapshot()
                    controller.apply_adam(gradients, learning_rate=rate)
                    momentum_reset = True
                candidate, _ = controller.predict(references)
                batch.set_batch(references, candidate)
                after = batch.forward_only()
                valid = bool(np.all(after["valid"]))
                mean_after = float(np.mean(after["losses"])) if valid else None
                accepted = valid and np.isfinite(mean_after) and mean_after <= mean_before
                trials_log.append(
                    {
                        "rate": rate,
                        "loss": mean_after,
                        "valid": valid,
                        "accepted": accepted,
                        "momentum_reset": momentum_reset,
                    }
                )
                if accepted:
                    break
                rate *= 0.5
            if not accepted:
                controller.restore(snapshot)
                mean_after = mean_before
            batches.append(
                {
                    "iteration": iteration,
                    "epoch": epoch + 1,
                    "stance_ids": [member["id"] for member in selected],
                    "loss_before": mean_before,
                    "loss_after": mean_after,
                    "accepted": accepted,
                    "restored_global_checkpoint": recovered,
                    "trials": trials_log,
                    "gradient": gradient_record,
                    "projection": cache["projection"],
                    "predictor_wall_s": predictor_s,
                    "physics_timings": result.get("timings"),
                    "search_wall_s": perf_counter() - search_started,
                }
            )
            _write(output / "progress.json", batches)
            if iteration % 10 == 0:
                print(
                    json.dumps(
                        {
                            "iteration": iteration,
                            "epoch": epoch + 1,
                            "batch_loss": mean_after,
                            "accepted": accepted,
                            "search_s": perf_counter() - search_started,
                        }
                    ),
                    flush=True,
                )
        if (epoch + 1) % 2 == 0 or epoch + 1 == epochs:
            records = _score(batch, controller, training, thresholds)
            score = _summarize(records)
            history.append({"iteration": iteration, "loss": score["mean_loss"]})
            if score["complete_count"] == 100 and score["mean_loss"] < best_loss:
                best, best_loss, best_update = controller.snapshot(), score["mean_loss"], iteration
            controller.save(output / f"checkpoint_{iteration:04d}.npz", {**metadata, "iteration": iteration})
            _write(output / f"training_metrics_{iteration:04d}.json", records)
            print(json.dumps(_plain({"epoch": epoch + 1, "training": score, "best_loss": best_loss})), flush=True)
    search_s = perf_counter() - search_started
    controller.restore(best)
    controller.save(output / "shared_controller.npz", {**metadata, "selected_iteration": best_update})
    scoring_started = perf_counter()
    final = _score(batch, controller, training + evaluation, thresholds)
    scoring_s = perf_counter() - scoring_started
    initial_by_id = {record["stance_id"]: record for record in initial}
    for record in final:
        baseline = initial_by_id[record["stance_id"]]
        record["initial_loss"] = baseline["loss"]
        record["initial_nativechannelrmse"] = baseline["nativechannelrmse"]
    train_final = _summarize([record for record in final if record["split"] == "train"])
    eval_final = _summarize([record for record in final if record["split"] == "eval"])
    report = {
        "schema": "cartesian_conditioned_physics_gradient_training_1",
        "title": "100-stance physics-gradient shared controller",
        "train_count": 100,
        "eval_count": 10,
        "initial_loss": initial_train["mean_loss"],
        "final_loss": train_final["mean_loss"],
        "train_mean_loss": train_final["mean_loss"],
        "eval_mean_loss": eval_final["mean_loss"],
        "initial_training": initial_train,
        "initial_evaluation": initial_eval,
        "final_training": train_final,
        "final_evaluation": eval_final,
        "epochs": epochs,
        "iterations": iteration,
        "accepted_updates": sum(record["accepted"] for record in batches),
        "selected_iteration": best_update,
        "selection": "lowest complete all100-training meanloss; evaluation never selects weights or hyperparameters",
        "coverage": selected_count,
        "eval_excluded_from_updates": True,
        "history": history,
        "batch_history": batches,
        "metrics": final,
        "setup_wall_s": setup_s,
        "search_wall_s": search_s,
        "final_scoring_wall_s": scoring_s,
        "total_wall_s": perf_counter() - started,
        "thresholds": thresholds,
        "acceptance": {
            "native_all6_pass_count": train_final["native_all6_pass_count"] + eval_final["native_all6_pass_count"],
            "native_all6_total": 110,
        },
        "qualification": "Native measured scoring across all110; no all-stance half-timestep qualification yet.",
        "provenance": {
            **metadata,
            "source_sha256": source_snapshot(),
            "batch_qualification": str(qualification.resolve()),
            "batch_qualification_sha256": hashlib.sha256(qualification.read_bytes()).hexdigest(),
        },
        "dataset_policy": manifest["policy"],
    }
    _write(output / "report.json", report)
    _write(output / "stance_metrics.json", final)
    (output / "report.html").write_text(render_report(_plain(report)))
    print(
        json.dumps(
            _plain(
                {
                    "training": train_final,
                    "evaluation": eval_final,
                    "search_s": search_s,
                    "html": str(output / "report.html"),
                }
            )
        ),
        flush=True,
    )
    return report


def main():
    """Run the complete 100-train/10-eval shared-model experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fit-directory", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--capacity", type=int, choices=(4,), default=4)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    train(
        args.dataset,
        args.fit_directory,
        args.output,
        qualification=args.qualification,
        epochs=args.epochs,
        capacity=args.capacity,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
