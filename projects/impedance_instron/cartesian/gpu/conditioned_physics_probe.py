# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check predictor-to-physics gradients before fitting the 100 stance controller."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from .adjoint_batch import BatchAdjoint
from .benchmark import _plain
from .conditioned_controller import Controller
from .provenance import source_snapshot
from .shared_train import _load_dataset, _load_fit


def probe(dataset: Path, fit_directory: Path, batch_qualification: Path, output: Path):
    """Compare shared parameter directional derivatives through projection and full physics."""
    if output.exists():
        raise FileExistsError(output)
    qualified = json.loads(batch_qualification.read_text())
    if not qualified.get("qualified") or qualified.get("source_sha256") != source_snapshot():
        raise ValueError("The reusable heterogeneous batch must pass qualification first")
    _, profile, _, _, settings, _, _ = _load_fit(fit_directory)
    _, _, _, members = _load_dataset(dataset, None, None)
    training = [member for member in members if member["split"] == "train"]
    selected = []
    for trial in ("FR3_1", "FR3_2"):
        candidates = [member for member in training if member["trial"] == trial]
        selected.extend(
            (
                min(candidates, key=lambda member: member["duration_s"]),
                max(candidates, key=lambda member: member["duration_s"]),
            )
        )
    references = [member["reference_data"] for member in selected]
    controller = Controller([member["reference_data"] for member in training], profile, settings.parameter_scale)
    rng = np.random.default_rng(19)
    controller.parameters["decoder_weight"][:] = rng.normal(size=controller.parameters["decoder_weight"].shape) * 0.003
    controller.parameters["decoder_bias"][:] = rng.normal(size=72) * 0.005
    coefficients, cache = controller.predict(references)
    batch = BatchAdjoint(fit_directory, dataset, capacity=4)
    batch.set_batch(references, coefficients)
    result = batch.value_and_grad()
    if not np.all(result["valid"]):
        raise RuntimeError("The predictor experiment baseline did not complete")
    batch.capture()
    gradients = controller.backward(result["gradients"], cache)
    baseline = {name: value.copy() for name, value in controller.parameters.items()}
    records = []
    for _ in range(3):
        direction = {name: rng.normal(size=value.shape) for name, value in baseline.items()}
        norm = np.sqrt(sum(np.sum(value * value) for value in direction.values()))
        direction = {name: value / norm for name, value in direction.items()}
        predicted = float(sum(np.sum(gradients[name] * value) for name, value in direction.items()))
        curve = []
        for epsilon in (0.03, 0.01, 0.003, 0.001, 0.0003):
            losses, valid = [], []
            for sign in (1.0, -1.0):
                controller.parameters = {
                    name: value + sign * epsilon * direction[name] for name, value in baseline.items()
                }
                candidate, _ = controller.predict(references)
                batch.set_batch(references, candidate)
                after = batch.forward_only()
                complete = bool(np.all(after["valid"]))
                valid.append(complete)
                losses.append(float(np.mean(after["losses"])) if complete else np.nan)
            fd = (losses[0] - losses[1]) / (2 * epsilon)
            error = abs(fd - predicted)
            passed = all(valid) and np.isfinite(error) and error <= 1e-5 + 0.01 * max(abs(fd), abs(predicted))
            curve.append(
                {
                    "epsilon": epsilon,
                    "autodiff": predicted,
                    "finite_difference": fd,
                    "absolute_error": error,
                    "complete": valid,
                    "passed": passed,
                }
            )
        records.append({"passed": any(a["passed"] and b["passed"] for a, b in pairwise(curve)), "curve": curve})
    report = {
        "schema": "conditioned_controller_full_physics_gradient_experiment_1",
        "qualified": all(record["passed"] for record in records),
        "scope": "Four complete training stances; perturbed decoder away from projection transitions; no guarantee at every subsequent contact branch.",
        "source_sha256": source_snapshot(),
        "dataset_manifest_sha256": hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest(),
        "fit_directory": str(fit_directory.resolve()),
        "adjoint_sources_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("adjoint_batch.py", "adjoint_contact.py", "adjoint_objective.py", "adjoint.py")
        },
        "batch_qualification": str(batch_qualification.resolve()),
        "batch_qualification_sha256": hashlib.sha256(batch_qualification.read_bytes()).hexdigest(),
        "controller_source_sha256": hashlib.sha256(
            Path(__file__).with_name("conditioned_controller.py").read_bytes()
        ).hexdigest(),
        "training_only_statistics": True,
        "stance_ids": [member["id"] for member in selected],
        "baseline_losses": result["losses"],
        "timings": result.get("timings"),
        "directions": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_plain(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"qualified": report["qualified"], "output": str(output)}), flush=True)
    if not report["qualified"]:
        raise RuntimeError("The predictor-to-physics gradient experiment failed")
    return report


def main():
    """Run the complete predictor and heterogeneous-physics experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fit-directory", type=Path, required=True)
    parser.add_argument("--batch-qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe(args.dataset, args.fit_directory, args.batch_qualification, args.output)


if __name__ == "__main__":
    main()
