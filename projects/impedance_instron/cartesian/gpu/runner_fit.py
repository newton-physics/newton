# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Identify one runner equilibrium curve with a reference-independent phase clock."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from itertools import pairwise
from pathlib import Path
from time import perf_counter

import numpy as np

from ..data import validate
from .adjoint_batch import BatchAdjoint
from .benchmark import _plain
from .engine import Engine
from .gradient_train import _summarize, _write
from .provenance import source_snapshot
from .runner_controller import RunnerController
from .shared_train import _load_dataset, _load_fit


def sources():
    """Pin the physics and runner experiment sources used by a qualification."""
    names = (
        "runner_controller.py",
        "runner_fit.py",
        "adjoint_batch.py",
        "adjoint.py",
        "adjoint_contact.py",
        "adjoint_objective.py",
    )
    return {
        **source_snapshot(),
        **{name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names},
    }


def supported_reference(reference, controller):
    """Crop scoring support without changing the human controller's phase clock."""
    result = deepcopy(reference)
    end = min(float(reference["time_s"][-1]), controller.duration)
    original_time = reference["time_s"]
    time = np.r_[original_time[original_time < end], end]
    for name in (
        "state",
        "velocity",
        "hip_target_m",
        "joint_target_rad",
        "foot_ground_target_rad",
        "foot_marker_target_m",
    ):
        if name in reference:
            values = reference[name]
            flat = values.reshape(len(original_time), -1)
            interpolated = np.column_stack([np.interp(time, original_time, column) for column in flat.T])
            result[name] = interpolated.reshape((len(time), *values.shape[1:]))
    result["time_s"] = time
    result["lengths_m"] = controller.mean_geometry.copy()
    validate(result)
    return result


def load_problem(dataset, fit):
    """Initialize a fresh shared curve from training-only reference summaries."""
    reference, profile, _, config, settings, shoe, _ = _load_fit(fit)
    manifest, _, _, members = _load_dataset(dataset, None, None)
    training = [m["reference_data"] for m in members if m["split"] == "train"]
    if len(training) != 100 or len(members) != 110:
        raise ValueError("Use the declared 100-training/10-evaluation dataset")
    controller = RunnerController(training, profile, settings.parameter_scale, mode="fixed")
    for member in members:
        member["original_duration_s"] = member["duration_s"]
        member["reference_data"] = supported_reference(member["reference_data"], controller)
        member["duration_s"] = float(member["reference_data"]["time_s"][-1])
    return manifest, members, controller, reference, profile, config, settings, shoe


def predict(controller, members):
    """Read only initial states to generate the same translated equilibrium curve."""
    return controller.predict_initial(
        np.asarray([m["reference_data"]["state"][0] for m in members]),
        np.asarray([m["reference_data"]["velocity"][0] for m in members]),
    )


def score(batch, controller, members, thresholds):
    """Score native measurements within the shared curve's fixed support."""
    rows = []
    for offset in range(0, len(members), 4):
        selected = members[offset : offset + 4]
        coefficients, _ = predict(controller, selected)
        batch.set_batch([m["reference_data"] for m in selected], coefficients)
        result = batch.forward_only()
        for i, member in enumerate(selected):
            valid = bool(result["valid"][i])
            rmse = result["rmse"][i]
            rows.append(
                {
                    "stance_id": member["id"],
                    "trial": member["trial"],
                    "split": member["split"],
                    "complete": valid,
                    "loss": float(result["losses"][i]) if valid else None,
                    "nativechannelrmse": rmse if valid else None,
                    "native_all6_pass": valid and bool(np.all(rmse <= thresholds)),
                    "original_duration_s": member["original_duration_s"],
                    "scoring_duration_s": member["duration_s"],
                    "scoring_time_coverage": member["duration_s"] / member["original_duration_s"],
                }
            )
    return rows


def qualify(dataset, fit, output):
    """Compare fixed-clock batch derivatives and target-free production rollouts."""
    if output.exists():
        raise FileExistsError(output)
    _, members, controller, _, profile, config, settings, shoe = load_problem(dataset, fit)
    selected = []
    for trial in ("FR3_1", "FR3_2"):
        candidates = [m for m in members if m["trial"] == trial and m["split"] == "train"]
        selected += [
            min(candidates, key=lambda m: m["original_duration_s"]),
            max(candidates, key=lambda m: m["original_duration_s"]),
        ]
    coefficients, cache = predict(controller, selected)
    batch = BatchAdjoint(
        fit, dataset, controller_duration_s=controller.duration, human_lengths_m=controller.mean_geometry
    )
    refs = [m["reference_data"] for m in selected]
    batch.set_batch(refs, coefficients)
    batch.capture()
    baseline = batch.value_and_grad()
    if not np.all(baseline["valid"]):
        raise RuntimeError("The shared seed fails one of the physics qualification rollouts")
    parity = []
    for w, member in enumerate(selected):
        r = member["reference_data"]
        native = Engine(
            r,
            profile,
            shoe["path"],
            shoe["mount_m"],
            shoe["static_pitch_rad"],
            config=config,
            settings=settings,
            world_count=1,
            friction_model=shoe["friction_model"],
            controller_duration_s=controller.duration,
        )
        measured = native.evaluate(coefficients[w : w + 1])
        target_free = Engine.from_initial(
            r["state"][0],
            r["velocity"][0],
            controller.mean_geometry,
            r["endpoint_local_m"],
            profile,
            shoe["path"],
            shoe["mount_m"],
            shoe["static_pitch_rad"],
            duration_s=member["duration_s"],
            controller_duration_s=controller.duration,
            config=config,
            settings=settings,
            friction_model=shoe["friction_model"],
        )
        free = target_free.rollout(coefficients[w : w + 1])
        q = batch._q_storage.numpy()[: native.steps + 1, 0, w]
        force = batch._measured_force_storage.numpy()[: native.steps, 0, w]
        parity.append(
            {
                "stance_id": member["id"],
                "batch_state_exact": bool(np.array_equal(q, native.states.numpy()[:, 0])),
                "batch_force_exact": bool(np.array_equal(force, native.forces.numpy()[:, 0])),
                "loss_absolute_difference": abs(float(measured["loss"][0]) - baseline["losses"][w]),
                "target_free_state_exact": bool(np.array_equal(native.states.numpy(), target_free.states.numpy())),
                "target_free_force_exact": bool(np.array_equal(native.forces.numpy(), target_free.forces.numpy())),
                "target_free_complete": bool(free["completed"][0]),
            }
        )
    gradient = controller.backward(baseline["gradients"], cache)["template"]
    seed = controller.parameters["template"].copy()
    rng = np.random.default_rng(29)
    directions = []
    face = cache["faces"][0]
    for _ in range(3):
        direction = rng.normal(size=72)
        direction -= face.T @ (face @ direction)
        direction /= np.linalg.norm(direction)
        predicted = float(gradient @ direction)
        curve = []
        for epsilon in (0.03, 0.01, 0.003, 0.001, 0.0003):
            losses = []
            for sign in (1, -1):
                controller.parameters["template"] = seed + sign * epsilon * direction
                candidate, _ = predict(controller, selected)
                batch.set_batch(refs, candidate)
                result = batch.forward_only()
                losses.append(float(np.mean(result["losses"])) if np.all(result["valid"]) else np.nan)
            fd = (losses[0] - losses[1]) / (2 * epsilon)
            error = abs(fd - predicted)
            curve.append(
                {
                    "epsilon": epsilon,
                    "autodiff": predicted,
                    "finite_difference": fd,
                    "absolute_error": error,
                    "passed": bool(np.isfinite(error) and error <= 1e-5 + 0.01 * max(abs(fd), abs(predicted))),
                }
            )
        directions.append({"passed": any(a["passed"] and b["passed"] for a, b in pairwise(curve)), "curve": curve})
    passed = all(
        r["batch_state_exact"]
        and r["batch_force_exact"]
        and r["loss_absolute_difference"] < 1e-9
        and r["target_free_state_exact"]
        and r["target_free_force_exact"]
        and r["target_free_complete"]
        for r in parity
    ) and all(r["passed"] for r in directions)
    report = {
        "qualified": passed,
        "sources": sources(),
        "dataset_sha256": hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest(),
        "fit_directory": str(fit.resolve()),
        "duration_s": controller.duration,
        "geometry_m": controller.mean_geometry,
        "scope": "Four training stances from both trials; fixed clock and geometry; target-free production parity; three shared-coefficient directions tangent to active spline faces. Contact nonsmoothness remains.",
        "parity": parity,
        "directions": directions,
        "baseline_losses": baseline["losses"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write(output, report)
    print(json.dumps({"qualified": passed, "output": str(output)}), flush=True)
    if not passed:
        raise RuntimeError("The fixed runner physics experiment did not qualify")


def fit_runner(dataset, fit, qualification, output, *, epochs=10, learning_rate=0.003):
    """Fit one shared curve from scratch across all training initial conditions."""
    if output.exists():
        raise FileExistsError(output)
    audit = json.loads(qualification.read_text())
    if (
        not audit["qualified"]
        or audit["sources"] != sources()
        or audit["dataset_sha256"] != hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest()
        or audit["fit_directory"] != str(fit.resolve())
    ):
        raise ValueError("A passing experiment with identical sources and inputs is required")
    started = perf_counter()
    manifest, members, controller, reference, profile, config, settings, shoe = load_problem(dataset, fit)
    training = [m for m in members if m["split"] == "train"]
    evaluation = [m for m in members if m["split"] == "eval"]
    thresholds = np.asarray(
        [settings.hip_tolerance_m] * 2 + [settings.joint_tolerance_rad] * 2 + [settings.force_tolerance_n] * 2
    )
    batch = BatchAdjoint(
        fit, dataset, controller_duration_s=controller.duration, human_lengths_m=controller.mean_geometry
    )
    initial_coeff, _ = predict(controller, training[:4])
    batch.set_batch([m["reference_data"] for m in training[:4]], initial_coeff)
    batch.capture()
    initial = score(batch, controller, training + evaluation, thresholds)
    initial_train = _summarize(initial[:100])
    if initial_train["complete_count"] != 100:
        raise RuntimeError("Fresh shared seed must complete all training stances before fitting")
    output.mkdir(parents=True)
    metadata = {
        "schema": "runner_shared_curve_1",
        "profile": profile,
        "simulation_config": vars(config),
        "fit_config": vars(settings),
        "shoe": shoe,
        "endpoint_local_m": reference["endpoint_local_m"].tolist(),
        "dataset": str(dataset.resolve()),
        "fit_directory": str(fit.resolve()),
        "initialization": "Unfitted training-only aligned mean nominal; no fitted controller checkpoint",
        "controller": "One shared72-coefficient curve; horizontal translation only",
        "runtime_inputs": "initial_state and initial_velocity only; material belongs to the shoe",
        "phase_clock": "Fixed train-only median period; no per-reference time stretching",
        "training_ids": [m["id"] for m in training],
        "evaluation_ids": [m["id"] for m in evaluation],
    }
    controller.save(output / "initial_controller.npz", _plain(metadata))
    _write(output / "initial_metrics.json", initial)
    _write(output / "profile.json", profile)
    setup_s = perf_counter() - started
    best, best_loss, best_update = controller.snapshot(), initial_train["mean_loss"], 0
    history = [{"iteration": 0, "loss": best_loss}]
    coverage = {m["id"]: 0 for m in training}
    groups = [[m for m in training if m["trial"] == trial] for trial in ("FR3_1", "FR3_2")]
    rng = np.random.default_rng(17)
    updates = []
    search_started = perf_counter()
    for epoch in range(epochs):
        orders = [rng.permutation(50) for _ in groups]
        for offset in range(0, 50, 2):
            selected = [
                group[i] for group, order in zip(groups, orders, strict=True) for i in order[offset : offset + 2]
            ]
            for member in selected:
                coverage[member["id"]] += 1
            coefficients, cache = predict(controller, selected)
            refs = [m["reference_data"] for m in selected]
            batch.set_batch(refs, coefficients)
            result = batch.value_and_grad()
            recovered = False
            if not np.all(result["valid"]):
                controller.restore(best)
                coefficients, cache = predict(controller, selected)
                batch.set_batch(refs, coefficients)
                result = batch.value_and_grad()
                recovered = True
            if not np.all(result["valid"]) or not np.isfinite(result["gradients"]).all():
                raise RuntimeError("Globally valid shared checkpoint has an invalid batch/gradient")
            gradients = controller.backward(result["gradients"], cache)
            before = float(np.mean(result["losses"]))
            snapshot = controller.snapshot()
            attempts = []
            accepted = False
            for attempt in range(8):
                controller.restore(snapshot)
                if attempt >= 4:
                    controller.first = {k: np.zeros_like(v) for k, v in controller.first.items()}
                    controller.second = {k: np.zeros_like(v) for k, v in controller.second.items()}
                    controller.updates = 0
                rate = learning_rate * 0.5 ** (attempt if attempt < 4 else attempt - 4)
                controller.apply_adam(gradients, learning_rate=rate)
                candidate, _ = predict(controller, selected)
                batch.set_batch(refs, candidate)
                after = batch.forward_only()
                valid = bool(np.all(after["valid"]))
                loss = float(np.mean(after["losses"])) if valid else None
                accepted = valid and np.isfinite(loss) and loss <= before
                attempts.append({"rate": rate, "loss": loss, "valid": valid, "accepted": accepted})
                if accepted:
                    break
            if not accepted:
                controller.restore(snapshot)
                loss = before
            iteration = len(updates) + 1
            updates.append(
                {
                    "iteration": iteration,
                    "epoch": epoch + 1,
                    "loss_before": before,
                    "loss_after": loss,
                    "accepted": accepted,
                    "restored_checkpoint": recovered,
                    "stance_ids": [m["id"] for m in selected],
                    "attempts": attempts,
                    "search_s": perf_counter() - search_started,
                }
            )
            _write(output / "progress.json", updates)
            if iteration % 10 == 0:
                print(
                    json.dumps(
                        {
                            "iteration": iteration,
                            "epoch": epoch + 1,
                            "loss": loss,
                            "accepted": accepted,
                            "search_s": perf_counter() - search_started,
                        }
                    ),
                    flush=True,
                )
        if (epoch + 1) % 2 == 0 or epoch + 1 == epochs:
            records = score(batch, controller, training, thresholds)
            summary = _summarize(records)
            history.append(
                {"iteration": len(updates), "loss": summary["mean_loss"], "search_s": perf_counter() - search_started}
            )
            controller.save(output / f"checkpoint_{len(updates):04d}.npz", _plain(metadata))
            _write(output / f"training_metrics_{len(updates):04d}.json", records)
            if summary["complete_count"] == 100 and summary["mean_loss"] < best_loss:
                best, best_loss, best_update = controller.snapshot(), summary["mean_loss"], len(updates)
            print(json.dumps(_plain({"epoch": epoch + 1, "training": summary, "best_loss": best_loss})), flush=True)
    search_s = perf_counter() - search_started
    controller.restore(best)
    controller.save(output / "runner_controller.npz", _plain({**metadata, "selected_iteration": best_update}))
    final = score(batch, controller, training + evaluation, thresholds)
    for current, previous in zip(final, initial, strict=True):
        current.update(initial_loss=previous["loss"], initial_nativechannelrmse=previous["nativechannelrmse"])
    report = {
        "schema": "runner_shared_curve_fit_1",
        "title": "One runner controller ·100 training stances",
        "train_count": 100,
        "eval_count": 10,
        "initial_training": initial_train,
        "initial_evaluation": _summarize(initial[100:]),
        "final_training": _summarize(final[:100]),
        "final_evaluation": _summarize(final[100:]),
        "initial_loss": initial_train["mean_loss"],
        "final_loss": best_loss,
        "train_mean_loss": best_loss,
        "eval_mean_loss": _summarize(final[100:])["mean_loss"],
        "epochs": epochs,
        "iterations": len(updates),
        "accepted_updates": sum(r["accepted"] for r in updates),
        "selected_iteration": best_update,
        "coverage": coverage,
        "history": history,
        "batch_history": updates,
        "metrics": final,
        "thresholds": thresholds,
        "setup_wall_s": setup_s,
        "search_wall_s": search_s,
        "total_wall_s": perf_counter() - started,
        "duration_s": controller.duration,
        "geometry_m": controller.mean_geometry,
        "eval_excluded_from_updates": True,
        "qualification": "Fixed-clock four-stance derivative/target-free qualification passed; frozen final qualification pending.",
        "provenance": metadata,
        "sources": sources(),
        "dataset_policy": manifest["policy"],
        "selection": "Lowest complete all100-training mean loss; evaluation excluded from updates and selection",
    }
    _write(output / "report.json", report)
    print(
        json.dumps(
            _plain(
                {"training": report["final_training"], "evaluation": report["final_evaluation"], "search_s": search_s}
            )
        ),
        flush=True,
    )


def main():
    """Run the fixed runner qualification or from-scratch identification experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("qualify", "fit"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fit-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    args = parser.parse_args()
    if args.stage == "qualify":
        qualify(args.dataset, args.fit_directory, args.output)
    else:
        if (
            args.qualification is None
            or args.epochs < 1
            or not np.isfinite(args.learning_rate)
            or args.learning_rate <= 0
        ):
            parser.error("Fit requires qualification and positive epochs/learning rate")
        fit_runner(
            args.dataset,
            args.fit_directory,
            args.qualification,
            args.output,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )


if __name__ == "__main__":
    main()
