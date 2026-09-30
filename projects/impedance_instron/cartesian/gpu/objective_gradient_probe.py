# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Isolate measured-objective derivatives from physics and contact branch changes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp

from ..fit import FitConfig
from ..run import Config
from .adjoint_audit import _load_saved_fit
from .adjoint_objective import ObjectiveAdjoint
from .benchmark import _plain
from .engine import Engine
from .provenance import source_snapshot


def probe(directory: Path, output: Path):
    """Compare frozen-trajectory state and force VJPs against central differences."""
    if output.exists():
        raise FileExistsError(output)
    reference, profile, spline, summary, shoe_path = _load_saved_fit(directory)
    shoe = summary["shoe"]
    engine = Engine(
        reference,
        profile,
        shoe_path,
        shoe["mount_m"],
        shoe["static_pitch_rad"],
        config=Config(**summary["simulation_config"]),
        settings=FitConfig(**summary["fit_config"]),
        world_count=1,
        friction_model=shoe["friction_model"],
    )
    scores = engine.evaluate(spline.coefficients[None])
    if scores["failure_code"][0] != 0 or scores["integrated_steps"][0] != engine.steps:
        raise RuntimeError("The production stance did not complete")
    states = wp.clone(engine.states, requires_grad=True)
    forces = wp.clone(engine.forces, requires_grad=True)
    objective = ObjectiveAdjoint(engine.objective)
    loss = wp.zeros(1, dtype=wp.float64, device=engine.device, requires_grad=True)
    with wp.Tape() as tape:
        objective.launch(states, forces, loss)
    value = float(loss.numpy()[0])
    tape.backward(loss)
    q, f = states.numpy(), forces.numpy()
    gq, gf = states.grad.numpy(), forces.grad.numpy()
    rng = np.random.default_rng(37)
    checks = []
    for field, baseline, gradient, array, scale in (
        ("state", q, gq, states, np.asarray([0.02, 0.02, 0.05, 0.05, 0.05])),
        ("force", f, gf, forces, np.asarray([100.0, 100.0])),
    ):
        for _ in range(3):
            direction = rng.normal(size=baseline.shape)
            direction *= scale
            direction /= np.sqrt(engine.steps)
            predicted = float(np.sum(gradient * direction))
            curve = []
            for epsilon in (0.03, 0.01, 0.003):
                values = []
                for sign in (1.0, -1.0):
                    array.assign(baseline + sign * epsilon * direction)
                    loss.zero_()
                    objective.launch(states, forces, loss)
                    values.append(float(loss.numpy()[0]))
                fd = (values[0] - values[1]) / (2 * epsilon)
                error = abs(predicted - fd)
                curve.append(
                    {
                        "epsilon": epsilon,
                        "autodiff": predicted,
                        "finite_difference": fd,
                        "absolute_error": error,
                        "passed": bool(error <= 1e-10 + 1e-6 * max(abs(predicted), abs(fd))),
                    }
                )
            array.assign(baseline)
            checks.append({"field": field, "passed": all(item["passed"] for item in curve), "curve": curve})
    report = {
        "schema": "cartesian_frozen_trajectory_objective_gradient_probe_1",
        "scope": "Measured objective only; trajectory samples are independent inputs. No coefficient gradient or optimizer qualification.",
        "baseline": str(directory.resolve()),
        "steps": engine.steps,
        "loss": value,
        "production_loss": float(scores["loss"][0]),
        "forward_exact": value == float(scores["loss"][0]),
        "gradient_passed": all(item["passed"] for item in checks),
        "directions": checks,
        "tolerance": "1e-10 absolute plus 1e-6 relative; double-precision quadratic objective on frozen samples.",
        "source_sha256": source_snapshot(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "warp_version": wp.__version__,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_plain(report), indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "forward_exact": report["forward_exact"],
                "gradient_passed": report["gradient_passed"],
                "output": str(output),
            }
        ),
        flush=True,
    )
    if not report["forward_exact"] or not report["gradient_passed"]:
        raise RuntimeError("Frozen-trajectory measured objective failed its gradient experiment")
    return report


def main():
    """Run the frozen-trajectory objective experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe(args.baseline, args.output)


if __name__ == "__main__":
    main()
