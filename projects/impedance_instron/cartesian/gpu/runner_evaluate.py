# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify a frozen shared runner fit and compare materials without human refitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from ..fit import FitConfig
from ..run import Config
from .engine import Engine
from .gradient_train import _write
from .runner_fit import supported_reference
from .runner_rollout import load_runner, material_variant, simulate
from .shared_train import _load_dataset


def evaluate(run, output):
    """Score independent native/fine physics and target-free material responses."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = perf_counter()
    checkpoint = run / "runner_controller.npz"
    controller, metadata = load_runner(checkpoint)
    runtime_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in ("dataset", "fit_directory", "training_ids", "evaluation_ids")
    }
    fit_report = json.loads((run / "report.json").read_text())
    _, _, _, members = _load_dataset(Path(metadata["dataset"]), None, None)
    expected = {row["stance_id"]: row for row in fit_report["metrics"]}
    thresholds = np.asarray(fit_report["thresholds"])
    config = Config(**metadata["simulation_config"])
    settings = FitConfig(**metadata["fit_config"])
    shoe = metadata["shoe"]
    rows = []
    geometry_errors = []
    for member in members:
        raw = member["reference_data"]
        reference = supported_reference(raw, controller)
        q0, v0 = reference["state"][0], reference["velocity"][0]
        coeff, _ = controller.predict_initial(q0[None, :], v0[None, :])
        native = Engine(
            reference,
            metadata["profile"],
            shoe["path"],
            shoe["mount_m"],
            shoe["static_pitch_rad"],
            config=config,
            settings=settings,
            world_count=1,
            friction_model=shoe["friction_model"],
            controller_duration_s=controller.duration,
        )
        result = native.evaluate(coeff)
        complete = bool(result["failure_code"][0] == 0 and result["integrated_steps"][0] == native.steps)
        prior = expected[member["id"]]
        row = {
            "stance_id": member["id"],
            "split": member["split"],
            "trial": member["trial"],
            "complete": complete,
            "batch_loss_difference": abs(float(result["loss"][0]) - prior["loss"])
            if complete and prior["complete"]
            else None,
            "native_loss": float(result["loss"][0]) if complete else None,
            "native_rmse": result["rmse"][0] if complete else None,
            "native_all6_pass": complete and bool(np.all(result["rmse"][0] <= thresholds)),
        }
        full_engine = Engine.from_initial(
            q0,
            v0,
            controller.mean_geometry,
            metadata["endpoint_local_m"],
            metadata["profile"],
            shoe["path"],
            shoe["mount_m"],
            shoe["static_pitch_rad"],
            duration_s=controller.duration,
            controller_duration_s=controller.duration,
            config=config,
            settings=settings,
            friction_model=shoe["friction_model"],
        )
        full_outcome = full_engine.rollout(coeff)
        row["full_period_target_free_complete"] = bool(full_outcome["completed"][0])
        row["full_period_failure_code"] = int(full_outcome["failure_code"][0])
        if member["split"] == "eval":
            fine = Engine(
                reference,
                metadata["profile"],
                shoe["path"],
                shoe["mount_m"],
                shoe["static_pitch_rad"],
                config=replace(config, dt_s=config.dt_s / 2),
                settings=settings,
                world_count=1,
                friction_model=shoe["friction_model"],
                controller_duration_s=controller.duration,
            )
            fine_result = fine.evaluate(coeff)
            fine_complete = bool(
                fine_result["failure_code"][0] == 0 and fine_result["integrated_steps"][0] == fine.steps
            )
            summary, trace, free_coeff = simulate(
                controller, runtime_metadata, q0, v0, duration_s=float(reference["time_s"][-1])
            )
            row.update(
                fine_complete=fine_complete,
                fine_loss=float(fine_result["loss"][0]) if fine_complete else None,
                fine_rmse=fine_result["rmse"][0] if fine_complete else None,
                fine_all6_pass=fine_complete and bool(np.all(fine_result["rmse"][0] <= thresholds)),
                target_free_complete=summary["complete"],
                target_free_coefficients_exact=bool(np.array_equal(coeff, free_coeff)),
                target_free_state_exact=bool(
                    np.array_equal(trace["state"], native.states.numpy()[: len(trace["state"]), 0])
                ),
                target_free_force_exact=bool(
                    np.array_equal(trace["grf_n"], native.forces.numpy()[: len(trace["grf_n"]), 0])
                ),
            )
        rows.append(row)
        geometry_errors.append(np.max(np.abs(reference["lengths_m"] - controller.mean_geometry)))
        if len(rows) % 10 == 0:
            print(json.dumps({"native_scored": len(rows), "wall_s": perf_counter() - started}), flush=True)
    evaluations = [r for r in rows if r["split"] == "eval"]
    parity = all(
        r["complete"] and r["batch_loss_difference"] is not None and r["batch_loss_difference"] < 1e-9 for r in rows
    )
    target_free = all(
        r["target_free_complete"]
        and r["target_free_state_exact"]
        and r["target_free_force_exact"]
        and r["target_free_coefficients_exact"]
        for r in evaluations
    )
    baseline_artifact = Path(shoe["path"])
    artifacts = {1.0: baseline_artifact}
    for scale in (0.8, 1.2):
        artifacts[scale] = material_variant(baseline_artifact, output / f"synthetic_modulus_{scale:.1f}.json", scale)
    material_rows = []
    exemplar = next(m["id"] for m in members if m["split"] == "eval")
    for member in [m for m in members if m["split"] == "eval"]:
        r = member["reference_data"]
        q0, v0 = r["state"][0], r["velocity"][0]
        hashes = []
        for scale, artifact in sorted(artifacts.items()):
            summary, trace, coefficients = simulate(controller, metadata, q0, v0, artifact=artifact)
            hashes.append(summary["coefficients_sha256"])
            material_rows.append(
                {"stance_id": member["id"], "trial": member["trial"], "modulus_scale": scale, **summary}
            )
            if member["id"] == exemplar:
                np.savez_compressed(output / f"example_{scale:.1f}.npz", **trace, coefficients=coefficients)
                fine_summary, _, fine_coeff = simulate(controller, metadata, q0, v0, artifact=artifact, half_step=True)
                material_rows[-1].update(
                    fine_complete=fine_summary["complete"],
                    fine_peak_grf_n=fine_summary["peak_grf_n"],
                    fine_grf_impulse_ns=fine_summary["grf_impulse_ns"],
                    fine_coefficients_exact=bool(np.array_equal(coefficients, fine_coeff)),
                )
        if len(set(hashes)) != 1:
            raise RuntimeError("Changing material changed the human controller coefficients")
        print(json.dumps({"material_initial_state": member["id"], "wall_s": perf_counter() - started}), flush=True)
    record = {
        "schema": "frozen_runner_experiment_1",
        "native_batch_parity_passed": parity,
        "native_complete_count": sum(r["complete"] for r in rows),
        "full_period_target_free_complete_count": sum(r["full_period_target_free_complete"] for r in rows),
        "maximum_native_loss_difference": max((r["batch_loss_difference"] or 0) for r in rows),
        "target_free_eval_parity_passed": target_free,
        "runtime_dataset_and_reference_paths_removed": True,
        "fine_eval_complete_count": sum(r["fine_complete"] for r in evaluations),
        "fine_eval_all6_pass_count": sum(r["fine_all6_pass"] for r in evaluations),
        "fine_eval_mean_loss": float(np.mean([r["fine_loss"] for r in evaluations]))
        if all(r["fine_complete"] for r in evaluations)
        else None,
        "human_geometry_exact": bool(max(geometry_errors) == 0),
        "controller_same_across_materials": True,
        "materials_scope": "Synthetic0.8/1.0/1.2 modulus sensitivities, unchanged geometry/friction/relaxation, resetshoehistory, fullstoredperiod from10 held-out initial states. No human/material reoptimization. Not calibrated alternative materials.",
        "material_complete_count": sum(r["complete"] for r in material_rows),
        "material_rollout_count": len(material_rows),
        "material_exemplar": exemplar,
        "material_rows": material_rows,
        "rows": rows,
        "wall_s": perf_counter() - started,
        "controller_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    _write(output / "report.json", record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("rows", "material_rows")}), flush=True)
    if not parity or not target_free:
        raise RuntimeError("Frozen runner experiment failed native/target-free parity")


def main():
    """Run frozen human, timestep and synthetic material experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.run, args.output)


if __name__ == "__main__":
    main()
