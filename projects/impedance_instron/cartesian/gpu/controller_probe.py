# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimentally check network and active-face projection derivatives."""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from .benchmark import _plain
from .conditioned_controller import Controller
from .shared_train import _load_dataset, _load_fit


def probe(dataset: Path, fit_directory: Path, output: Path):
    """Compare coefficient-predictor directional derivatives away from face transitions."""
    if output.exists():
        raise FileExistsError(output)
    _, profile, _, _, settings, _, _ = _load_fit(fit_directory)
    _, _, _, members = _load_dataset(dataset, None, None)
    training = [member["reference_data"] for member in members if member["split"] == "train"]
    references = [
        member["reference_data"]
        for trial in ("FR3_1", "FR3_2")
        for member in [next(m for m in members if m["split"] == "train" and m["trial"] == trial)]
    ]
    controller = Controller(training, profile, settings.parameter_scale)
    rng = np.random.default_rng(19)
    controller.parameters["decoder_weight"][:] = rng.normal(size=controller.parameters["decoder_weight"].shape) * 0.003
    controller.parameters["decoder_bias"][:] = rng.normal(size=72) * 0.005
    coefficients, cache = controller.predict(references)
    seed = rng.normal(size=coefficients.shape)
    gradients = controller.backward(seed, cache)
    baseline = {name: value.copy() for name, value in controller.parameters.items()}
    records = []
    for _ in range(3):
        direction = {name: rng.normal(size=value.shape) for name, value in baseline.items()}
        norm = np.sqrt(sum(np.sum(value * value) for value in direction.values()))
        direction = {name: value / norm for name, value in direction.items()}
        predicted = float(sum(np.sum(gradients[name] * value) for name, value in direction.items()))
        curve = []
        for epsilon in (0.003, 0.001, 0.0003, 0.0001, 0.00003):
            values, ranks = [], []
            for sign in (1.0, -1.0):
                controller.parameters = {
                    name: value + sign * epsilon * direction[name] for name, value in baseline.items()
                }
                perturbed, retained = controller.predict(references)
                values.append(float(np.sum(perturbed * seed) / len(references)))
                ranks.append([len(face) for face in retained["faces"]])
            fd = (values[0] - values[1]) / (2 * epsilon)
            error = abs(fd - predicted)
            curve.append(
                {
                    "epsilon": epsilon,
                    "autodiff": predicted,
                    "finite_difference": fd,
                    "absolute_error": error,
                    "active_ranks": ranks,
                    "passed": error <= 1e-6 + 0.001 * max(abs(fd), abs(predicted)),
                }
            )
        acceptable = [item["passed"] for item in curve]
        records.append({"passed": any(a and b for a, b in pairwise(acceptable)), "curve": curve})
    report = {
        "schema": "conditioned_controller_projection_gradient_experiment_1",
        "scope": "Network and convex projection only; coefficient objective is linear, no physics gradient qualification.",
        "training_count": len(training),
        "reference_count": len(references),
        "feature_statistics_training_only": True,
        "gradient_passed": all(item["passed"] for item in records),
        "directions": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_plain(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"gradient_passed": report["gradient_passed"], "output": str(output)}), flush=True)
    if not report["gradient_passed"]:
        raise RuntimeError("The predictor/projection gradient experiment failed")
    return report


def main():
    """Run the predictor gradient experiment on the declared stance dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fit-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe(args.dataset, args.fit_directory, args.output)


if __name__ == "__main__":
    main()
