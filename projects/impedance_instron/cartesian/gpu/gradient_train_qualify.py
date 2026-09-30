# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independently score a frozen shared controller and refine held-out rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .benchmark import _plain
from .conditioned_controller import Controller
from .engine import Engine
from .shared_train import _load_dataset, _load_fit


def qualify(run: Path, output: Path):
    """Check all native scores against the independent engine and refine evaluation stances."""
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((run / "report.json").read_text())
    provenance = report["provenance"]
    _, profile, _, config, settings, shoe, _ = _load_fit(Path(provenance["fit_directory"]))
    _, _, _, members = _load_dataset(Path(provenance["dataset"]), None, None)
    checkpoint = run / "shared_controller.npz"
    controller, _ = Controller.load(checkpoint, profile)
    recorded = {row["stance_id"]: row for row in report["metrics"]}
    thresholds = np.asarray(report["thresholds"])
    started = perf_counter()
    rows = []
    for member in members:
        coefficients, _ = controller.predict([member["reference_data"]])
        engine = Engine(
            member["reference_data"],
            profile,
            shoe["path"],
            shoe["mount_m"],
            shoe["static_pitch_rad"],
            config=config,
            settings=settings,
            world_count=1,
            friction_model=shoe["friction_model"],
        )
        native = engine.evaluate(coefficients)
        valid = bool(
            native["failure_code"][0] == 0
            and native["integrated_steps"][0] == engine.steps
            and np.isfinite(native["loss"][0])
        )
        loss = float(native["loss"][0])
        expected = recorded[member["id"]]
        row = {
            "stance_id": member["id"],
            "split": member["split"],
            "trial": member["trial"],
            "native_complete": valid,
            "native_loss": loss if valid else None,
            "batch_loss_absolute_difference": abs(loss - expected["loss"]) if valid and expected["complete"] else None,
            "native_rmse": native["rmse"][0] if valid else None,
            "native_all6_pass": valid and bool(np.all(native["rmse"][0] <= thresholds)),
        }
        if member["split"] == "eval":
            fine_engine = Engine(
                member["reference_data"],
                profile,
                shoe["path"],
                shoe["mount_m"],
                shoe["static_pitch_rad"],
                config=replace(config, dt_s=config.dt_s / 2),
                settings=settings,
                world_count=1,
                friction_model=shoe["friction_model"],
            )
            fine = fine_engine.evaluate(coefficients)
            fine_valid = bool(
                fine["failure_code"][0] == 0
                and fine["integrated_steps"][0] == fine_engine.steps
                and np.isfinite(fine["loss"][0])
            )
            row.update(
                fine_complete=fine_valid,
                fine_loss=float(fine["loss"][0]) if fine_valid else None,
                fine_rmse=fine["rmse"][0] if fine_valid else None,
                fine_all6_pass=fine_valid and bool(np.all(fine["rmse"][0] <= thresholds)),
            )
        rows.append(row)
        if len(rows) % 10 == 0:
            print(json.dumps({"scored": len(rows), "wall_s": perf_counter() - started}), flush=True)
    parity = all(row["native_complete"] and row["batch_loss_absolute_difference"] <= 1e-9 for row in rows)
    evaluations = [row for row in rows if row["split"] == "eval"]
    result = {
        "schema": "frozen_shared_controller_independent_qualification_1",
        "native_batch_parity_passed": parity,
        "native_complete_count": sum(row["native_complete"] for row in rows),
        "fine_eval_complete_count": sum(row["fine_complete"] for row in evaluations),
        "fine_eval_all6_pass_count": sum(row["fine_all6_pass"] for row in evaluations),
        "fine_eval_mean_loss": float(np.mean([row["fine_loss"] for row in evaluations]))
        if all(row["fine_complete"] for row in evaluations)
        else None,
        "maximum_native_loss_difference": max(row["batch_loss_absolute_difference"] or 0 for row in rows),
        "scope": "All110 independent native rollouts; half timestep only on10 held-out stances. No weights or settings are adapted.",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "wall_s": perf_counter() - started,
        "rows": rows,
    }
    output.write_text(json.dumps(_plain(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}), flush=True)
    if not parity:
        raise RuntimeError("Independent production scores differ from the batch report")


def main():
    """Run frozen native and half-step evaluation experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qualify(args.run, args.output)


if __name__ == "__main__":
    main()
