# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Audit coupled leg/shoe window derivatives without optimizing or scoring a truncated fit."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

from ..fit import FitConfig
from ..run import Config
from ..trajectory import Spline
from .adjoint import EngineAdjoint
from .benchmark import _plain
from .engine import Engine
from .profile_search import _load_bundle
from .provenance import source_snapshot


def audit(
    directory: Path,
    output: Path,
    *,
    steps: int = 32,
    start_step: int = 0,
    directions: int = 3,
    objective: str = "diagnostic",
    capture_gradient: bool = False,
):
    """Check a complete coupled window against reference physics and central differences."""
    if output.exists():
        raise FileExistsError(output)
    for name, value in (("steps", steps), ("directions", directions)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    started = perf_counter()
    reference, profile, initial, baseline, _manifest = _load_bundle(directory)
    shoe = baseline["shoe"]
    engine = Engine(
        reference,
        profile,
        directory / "digital_shoe.json",
        shoe["mount_m"],
        shoe["static_pitch_rad"],
        config=Config(**baseline["simulation_config"]),
        settings=FitConfig(**baseline["fit_config"]),
        world_count=1,
        friction_model=shoe["friction_model"],
    )
    if start_step < 0 or start_step + steps > engine.steps:
        raise ValueError("The window must lie within the original stance")
    values = initial.coefficients[None].copy()
    engine.capture(values)
    engine._reset()
    for _ in range(start_step // engine.chunk_steps):
        wp.capture_launch(engine.graph)
    for _ in range(start_step % engine.chunk_steps):
        engine._step()
    adjoint = EngineAdjoint(engine, steps, start_step=start_step, objective=objective)
    result = adjoint.value_and_grad()
    if not result["valid"]:
        raise RuntimeError(f"The baseline window gradient failed: {result['reason']}")
    timing = {"eager": result["timings"], "warp_pool_used_gib": wp.get_mempool_used_mem_current(engine.device) / 2**30}
    if capture_gradient:
        capture_started = perf_counter()
        adjoint.capture()
        timing["capture_wall_s"] = perf_counter() - capture_started
        warmup = adjoint.value_and_grad()
        if not warmup["valid"]:
            raise RuntimeError("The captured gradient warmup failed")
        timing["first_graph_replay"] = warmup["timings"]
        timing["graph_repeats"] = []
        for _ in range(3):
            repeated = adjoint.value_and_grad()
            if not repeated["valid"]:
                raise RuntimeError("The captured gradient replay failed")
            timing["graph_repeats"].append(
                {
                    **repeated["timings"],
                    "loss_exact": repeated["loss"] == result["loss"],
                    "gradient_max_absolute_difference": float(
                        np.max(np.abs(repeated["gradient"] - result["gradient"]))
                    ),
                }
            )
        print(json.dumps({"gradient_timing": timing}), flush=True)
    primal = {}
    for name, arrays in (("states", adjoint.q), ("velocities", adjoint.v)):
        primal[name] = np.concatenate([array.numpy() for array in arrays], axis=0)
    for name in ("equilibrium", "actuator", "forces", "moments", "fractions", "caps"):
        selected = adjoint.trace if name in ("equilibrium", "actuator") else adjoint.trace[:-1]
        primal[name] = np.concatenate([getattr(step, name).numpy() for step in selected], axis=0)
    if start_step == 0 and steps == engine.steps:
        engine.evaluate_device()
    else:
        for _ in range(steps):
            engine._step()
        engine._prepare()
    errors = {}
    for name, actual in primal.items():
        expected = getattr(engine, name).numpy()[start_step : start_step + len(actual)]
        errors[name] = {
            "exact": bool(np.array_equal(actual, expected)),
            "max_absolute_error": float(np.max(np.abs(actual - expected))),
        }
    contact_exact = {}
    for name in adjoint.contact.states[-1].FIELDS:
        contact_exact[name] = bool(
            np.array_equal(getattr(adjoint.contact.states[-1], name).numpy(), getattr(engine.foundation, name).numpy())
        )
    forward_passed = all(item["exact"] for item in errors.values()) and all(contact_exact.values())
    reference_loss = None
    if objective == "measured":
        reference_loss = float(engine.objective.loss.numpy()[0])
        forward_passed = forward_passed and result["loss"] == reference_loss
    gradient = result["gradient"]
    rng = np.random.default_rng(31)
    scale = np.asarray(engine.settings.parameter_scale)
    bounds = [
        profile[name]
        for name in (
            "equilibrium_lower",
            "equilibrium_upper",
            "equilibrium_rate_limit",
            "equilibrium_acceleration_limit",
        )
    ]
    forward_graph = adjoint.forward_graph
    if forward_graph is None:
        with wp.ScopedCapture(device=adjoint.device) as capture:
            adjoint.reset_diagnostics()
            adjoint.forward()
        forward_graph = capture.graph
    checks = []
    for _ in range(directions):
        direction = rng.normal(size=values.shape)
        direction /= np.linalg.norm(direction)
        direction *= scale
        predicted = float(np.sum(gradient * direction))
        curve = []
        epsilons = (3e-2, 1e-2, 3e-3, 1e-3, 3e-4) if objective == "measured" else (3e-3, 1e-3, 3e-4, 1e-4, 3e-5)
        for epsilon in epsilons:
            losses = []
            feasible = []
            complete = []
            for sign in (1.0, -1.0):
                candidate = values + sign * epsilon * direction
                feasible.append(Spline(initial.duration_s, candidate[0]).bounds(*bounds))
                adjoint.coefficients.assign(candidate)
                wp.capture_launch(forward_graph)
                valid = adjoint.complete()
                complete.append(valid)
                losses.append(float(adjoint.loss.numpy()[0]) if valid else np.nan)
            fd = (losses[0] - losses[1]) / (2 * epsilon)
            absolute_error = abs(fd - predicted)
            relative_error = absolute_error / max(abs(fd), abs(predicted), 1e-10)
            curve.append(
                {
                    "epsilon": epsilon,
                    "autodiff": predicted,
                    "finite_difference": fd,
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                    "complete": complete,
                    "spline_bounds": feasible,
                }
            )
        acceptable = [
            all(item["complete"])
            and np.isfinite(item["absolute_error"])
            and item["absolute_error"] <= 1e-5 + 0.01 * max(abs(item["autodiff"]), abs(item["finite_difference"]))
            for item in curve
        ]
        checks.append({"passed": any(a and b for a, b in pairwise(acceptable)), "curve": curve})
    report = {
        "schema": "cartesian_coupled_window_gradient_audit_1",
        "steps": steps,
        "start_step": start_step,
        "objective": objective,
        "loss": result["loss"],
        "reference_loss": reference_loss,
        "timing": timing,
        "gradient": gradient.tolist(),
        "forward_passed": forward_passed,
        "forward_errors": errors,
        "final_contact_state_exact": contact_exact,
        "directions": checks,
        "gradient_passed": all(item["passed"] for item in checks),
        "scope": (
            "Full-stance measured-objective VJP; no optimizer or acceptance claim."
            if objective == "measured"
            else "Conditional fixed-checkpoint window VJP with a diagnostic scalar; no optimizer or partial measured-fit score."
        ),
        "finite_difference_scope": "Derivative of the unconstrained simulation map; perturbation bound validity is reported, not treated as optimization permission.",
        "tolerance": "Adjacent epsilon agreement within 1% relative plus 1e-5 absolute; no acceptance limits changed.",
        "source_sha256": source_snapshot(),
        "adjoint_sources_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("adjoint.py", "adjoint_contact.py", "adjoint_objective.py", "adjoint_audit.py")
        },
        "warp_version": wp.__version__,
        "wall_s": perf_counter() - started,
        "initial_checkpoint_depends_on_coefficients": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_plain(report), indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "forward_passed": report["forward_passed"],
                "gradient_passed": report["gradient_passed"],
                "output": str(output),
                "wall_s": report["wall_s"],
            }
        ),
        flush=True,
    )
    if not report["forward_passed"] or not report["gradient_passed"]:
        raise RuntimeError("The coupled window audit failed; inspect saved errors and epsilon curves")
    return report


def main(argv: list[str] | None = None):
    """Run a reproducible coupled-window gradient audit on the saved controller."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("outputs/impedance_instron/baseline12_maxwell"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--directions", type=int, default=3)
    parser.add_argument("--objective", choices=("diagnostic", "measured"), default="diagnostic")
    parser.add_argument("--capture-gradient", action="store_true")
    args = parser.parse_args(argv)
    audit(
        args.baseline,
        args.output,
        steps=args.steps,
        start_step=args.start_step,
        directions=args.directions,
        objective=args.objective,
        capture_gradient=args.capture_gradient,
    )


if __name__ == "__main__":
    main()
