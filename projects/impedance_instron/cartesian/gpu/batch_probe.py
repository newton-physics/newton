# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Qualify heterogeneous graph reuse against independently simulated stances."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

from .adjoint import EngineAdjoint
from .adjoint_batch import BatchAdjoint
from .benchmark import _plain
from .engine import Engine
from .provenance import source_snapshot
from .shared_controller import nominal_six_channel
from .shared_train import _load_dataset, _load_fit


def probe(dataset: Path, fit_directory: Path, output: Path):
    """Check native forward parity, per-world VJPs, padding and repeated graph uploads."""
    if output.exists():
        raise FileExistsError(output)
    _, profile, _, config, settings, shoe, _ = _load_fit(fit_directory)
    _, _, _, members = _load_dataset(dataset, None, None)
    selected = [
        member
        for trial in ("FR3_1", "FR3_2")
        for member in [m for m in members if m["split"] == "train" and m["trial"] == trial][:2]
    ]
    references = [member["reference_data"] for member in selected]
    coefficients = np.asarray([nominal_six_channel(reference, profile) for reference in references])
    batch = BatchAdjoint(fit_directory, dataset)
    batch.set_batch(references, coefficients)
    first = batch.value_and_grad()
    rows = []
    for w, (member, reference) in enumerate(zip(selected, references, strict=True)):
        engine = Engine(
            reference,
            profile,
            shoe["path"],
            shoe["mount_m"],
            shoe["static_pitch_rad"],
            config=config,
            settings=settings,
            world_count=1,
            friction_model=shoe["friction_model"],
        )
        native = engine.evaluate(coefficients[w : w + 1])
        q = batch._q_storage.numpy()[: engine.steps + 1, 0, w]
        force = batch._measured_force_storage.numpy()[: engine.steps, 0, w]
        row = {
            "stance_id": member["id"],
            "steps": engine.steps,
            "loss_exact": first["losses"][w] == native["loss"][0],
            "state_exact": bool(np.array_equal(q, engine.states.numpy()[:, 0])),
            "force_exact": bool(np.array_equal(force, engine.forces.numpy()[:, 0])),
        }
        if w in (0, 2):
            engine._reset()
            adjoint = EngineAdjoint(engine, engine.steps, objective="measured")
            single = adjoint.value_and_grad()
            error = np.linalg.norm(first["gradients"][w] - single["gradient"][0])
            row["gradient_relative_error"] = float(error / max(np.linalg.norm(single["gradient"]), 1e-30))
            row["gradient_passed"] = bool(single["valid"] and row["gradient_relative_error"] <= 1e-4)
        rows.append(row)
    capture_started = perf_counter()
    batch.capture()
    capture_s = perf_counter() - capture_started
    batch.set_batch(references, coefficients)
    a = batch.value_and_grad()
    subset = [0, 2]
    batch.set_batch([references[i] for i in subset], coefficients[subset])
    partial = batch.value_and_grad()
    partial_exact = bool(np.array_equal(partial["losses"], a["losses"][subset]))
    batch.set_batch(references, coefficients)
    started = perf_counter()
    again = batch.value_and_grad()
    replay_s = perf_counter() - started
    repeated_exact = bool(np.array_equal(again["losses"], a["losses"]))
    repeat_gradient_error = float(
        np.linalg.norm(again["gradients"] - a["gradients"]) / max(np.linalg.norm(a["gradients"]), 1e-30)
    )
    passed = all(
        row["loss_exact"] and row["state_exact"] and row["force_exact"] and row.get("gradient_passed", True)
        for row in rows
    )
    passed = (
        passed and partial_exact and repeated_exact and repeat_gradient_error <= 1e-4 and bool(np.all(again["valid"]))
    )
    report = {
        "schema": "heterogeneous_batch_implementation_qualification_1",
        "qualified": passed,
        "scope": "Parity with existing individual physics/adjoints, inactive padding and graph reuse. Contact finite-difference limitations remain; predictor-to-physics directional qualification is separate.",
        "rows": rows,
        "partial_loss_exact": partial_exact,
        "repeated_loss_exact": repeated_exact,
        "repeated_gradient_relative_error": repeat_gradient_error,
        "capture_wall_s": capture_s,
        "four_world_value_gradient_wall_s": replay_s,
        "warp_pool_gib": wp.get_mempool_used_mem_current(batch.device) / 2**30,
        "source_sha256": source_snapshot(),
        "batch_source_sha256": hashlib.sha256(Path(__file__).with_name("adjoint_batch.py").read_bytes()).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_plain(report), indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({"qualified": passed, "four_world_value_gradient_wall_s": replay_s, "output": str(output)}),
        flush=True,
    )
    if not passed:
        raise RuntimeError("The reusable heterogeneous batch experiment failed")
    return report


def main():
    """Run the reusable heterogeneous batch experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fit-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe(args.dataset, args.fit_directory, args.output)


if __name__ == "__main__":
    main()
