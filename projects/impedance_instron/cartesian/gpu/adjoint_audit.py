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

from ..data import load as load_reference
from ..fit import FitConfig
from ..profile import load as load_profile
from ..run import Config
from ..trajectory import Spline
from .adjoint import EngineAdjoint
from .benchmark import _plain
from .engine import Engine
from .profile_search import _load_bundle
from .provenance import source_snapshot


def _load_saved_fit(directory: Path):
    """Read a fitted controller with its frozen input and shoe identities."""
    metadata = json.loads((directory / "optimization_inputs.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if metadata["simulation_config"] != summary["simulation_config"] or metadata["fit_config"] != summary["fit_config"]:
        raise ValueError("Saved fit input and result configurations differ")
    shoe = metadata["shoe"]
    shoe_path = Path(shoe["path"])
    if hashlib.sha256(shoe_path.read_bytes()).hexdigest() != shoe["sha256"]:
        raise ValueError("Saved shoe asset differs from the fit input")
    reference = load_reference(directory / "reference.npz")
    profile = load_profile(directory / "profile.json")
    with np.load(directory / "equilibrium.npz", allow_pickle=False) as archive:
        initial = Spline(float(archive["duration_s"]), archive["coefficients"].copy())
        identity = json.loads(str(archive["identity_json"]))
    if initial.coefficients.shape != (12, 6) or not np.isclose(
        initial.duration_s, reference["time_s"][-1], rtol=0, atol=1e-12
    ):
        raise ValueError("Saved controller does not match the six-channel reference")
    if identity["reference_sha256"] != hashlib.sha256((directory / "reference.npz").read_bytes()).hexdigest():
        raise ValueError("Saved controller reference identity differs")
    if identity["profile_sha256"] != hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest():
        raise ValueError("Saved controller profile identity differs")
    if identity["shoe"]["artifact_sha256"] != shoe["sha256"]:
        raise ValueError("Saved controller shoe identity differs")
    return reference, profile, initial, summary, shoe_path


def audit(
    directory: Path,
    output: Path,
    *,
    steps: int = 32,
    start_step: int = 0,
    directions: int = 3,
    objective: str = "diagnostic",
    capture_gradient: bool = False,
    controller_start: str = "fitted",
):
    """Check a complete coupled window against reference physics and central differences."""
    if output.exists():
        raise FileExistsError(output)
    for name, value in (("steps", steps), ("directions", directions)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    started = perf_counter()
    if (directory / "optimization_inputs.json").is_file():
        reference, profile, initial, baseline, shoe_path = _load_saved_fit(directory)
    else:
        reference, profile, initial, baseline, _manifest = _load_bundle(directory)
        shoe_path = directory / "digital_shoe.json"
    if controller_start == "unfitted":
        metadata = json.loads((directory / "optimization_inputs.json").read_text())
        initial = Spline(initial.duration_s, np.asarray(metadata["starting_coefficients"], dtype=np.float64))
    elif controller_start != "fitted":
        raise ValueError("Controller start must be fitted or unfitted")
    shoe = baseline["shoe"]
    engine = Engine(
        reference,
        profile,
        shoe_path,
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
    for name in ("equilibrium", "actuator", "ankle_force", "forces", "moments", "fractions", "caps"):
        selected = adjoint.trace if name in ("equilibrium", "actuator", "ankle_force") else adjoint.trace[:-1]
        primal[name] = np.concatenate([getattr(step, name).numpy() for step in selected], axis=0)
    primal["measured_force"] = np.concatenate([step.numpy() for step in adjoint.measured_force], axis=0)
    carrier_exact = {
        name: all(
            np.array_equal(getattr(step, f"carrier_{name}").numpy(), getattr(step, f"body_{name}").numpy())
            for step in adjoint.trace[:-1]
        )
        for name in ("q", "qd")
    }
    if start_step == 0 and steps == engine.steps:
        engine.evaluate_device()
    else:
        for _ in range(steps):
            engine._step()
        engine._prepare()
    errors = {}
    for name, actual in primal.items():
        expected_name = "forces" if name == "measured_force" else name
        expected = getattr(engine, expected_name).numpy()[start_step : start_step + len(actual)]
        errors[name] = {
            "exact": bool(np.array_equal(actual, expected)),
            "max_absolute_error": float(np.max(np.abs(actual - expected))),
        }
    contact_exact = {}
    for name in adjoint.contact.states[-1].FIELDS:
        contact_exact[name] = bool(
            np.array_equal(getattr(adjoint.contact.states[-1], name).numpy(), getattr(engine.foundation, name).numpy())
        )
    forward_passed = (
        all(item["exact"] for item in errors.values()) and all(contact_exact.values()) and all(carrier_exact.values())
    )
    reference_loss = None
    if objective == "measured":
        reference_loss = float(engine.objective.loss.numpy()[0])
        forward_passed = forward_passed and result["loss"] == reference_loss
    gradient = result["gradient"]
    contact_branches = None
    if objective.startswith("measured"):
        contact_branches = {
            "stuck": adjoint.contact._state_storage["tangent_stuck"].numpy(),
            "active": adjoint.contact._step_storage["compression"].numpy() > 0.0,
        }
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
        epsilons = (
            (3e-2, 1e-2, 3e-3, 1e-3, 3e-4)
            if objective.startswith("measured")
            else (3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5)
        )
        for epsilon in epsilons:
            losses = []
            feasible = []
            complete = []
            branch_changes = []
            for sign in (1.0, -1.0):
                candidate = values + sign * epsilon * direction
                feasible.append(Spline(initial.duration_s, candidate[0]).bounds(*bounds))
                adjoint.coefficients.assign(candidate)
                wp.capture_launch(forward_graph)
                valid = adjoint.complete()
                complete.append(valid)
                losses.append(float(adjoint.loss.numpy()[0]) if valid else np.nan)
                if contact_branches is not None:
                    branch_changes.append(
                        {
                            "stuck": int(
                                np.count_nonzero(
                                    adjoint.contact._state_storage["tangent_stuck"].numpy() != contact_branches["stuck"]
                                )
                            ),
                            "active": int(
                                np.count_nonzero(
                                    (adjoint.contact._step_storage["compression"].numpy() > 0.0)
                                    != contact_branches["active"]
                                )
                            ),
                        }
                    )
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
                    "contact_branch_changes": branch_changes,
                }
            )
        acceptable = [
            all(item["complete"])
            and np.isfinite(item["absolute_error"])
            and item["absolute_error"] <= 1e-5 + 0.01 * max(abs(item["autodiff"]), abs(item["finite_difference"]))
            for item in curve
        ]
        checks.append({"passed": any(a and b for a, b in pairwise(acceptable)), "curve": curve})
    current_sources = source_snapshot()
    input_names = ("reference.npz", "profile.json", "equilibrium.npz", "summary.json")
    if (directory / "optimization_inputs.json").is_file():
        input_names += ("optimization_inputs.json",)
    else:
        input_names += ("baseline.json", "digital_shoe.json")
    minimum_directions_met = objective != "measured" or directions >= 3
    direction_checks_passed = all(item["passed"] for item in checks)
    gradient_passed = minimum_directions_met and direction_checks_passed
    report = {
        "schema": "cartesian_coupled_window_gradient_audit_1",
        "baseline": str(directory.resolve()),
        "controller_start": controller_start,
        "coefficient_sha256": hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
        "input_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in input_names},
        "shoe_sha256": hashlib.sha256(shoe_path.read_bytes()).hexdigest(),
        "steps": steps,
        "start_step": start_step,
        "objective": objective,
        "loss": result["loss"],
        "reference_loss": reference_loss,
        "timing": timing,
        "gradient": gradient.tolist(),
        "forward_passed": forward_passed,
        "forward_errors": errors,
        "carrier_exact": carrier_exact,
        "final_contact_state_exact": contact_exact,
        "directions": checks,
        "minimum_directions_met": minimum_directions_met,
        "direction_checks_passed": direction_checks_passed,
        "gradient_passed": gradient_passed,
        "optimizer_qualified": objective == "measured" and forward_passed and gradient_passed,
        "scope": (
            "Full-stance measured-objective VJP; no optimizer or acceptance claim."
            if objective.startswith("measured")
            else "Conditional fixed-checkpoint window VJP with a diagnostic scalar; no optimizer or partial measured-fit score."
        ),
        "finite_difference_scope": "Derivative of the unconstrained simulation map; perturbation bound validity is reported, not treated as optimization permission.",
        "tolerance": "Adjacent epsilon agreement within 1% relative plus 1e-5 absolute; no acceptance limits changed.",
        "source_sha256": current_sources,
        "saved_source_changes": [
            name for name, digest in baseline.get("source_sha256", {}).items() if current_sources.get(name) != digest
        ],
        "saved_loss": baseline.get("loss"),
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
    parser.add_argument(
        "--objective",
        choices=(
            "diagnostic",
            "diagnostic_force",
            "diagnostic_terminal",
            "measured",
            "measured_motion",
            "measured_force",
        ),
        default="diagnostic",
    )
    parser.add_argument("--capture-gradient", action="store_true")
    parser.add_argument("--controller-start", choices=("fitted", "unfitted"), default="fitted")
    args = parser.parse_args(argv)
    audit(
        args.baseline,
        args.output,
        steps=args.steps,
        start_step=args.start_step,
        directions=args.directions,
        objective=args.objective,
        capture_gradient=args.capture_gradient,
        controller_start=args.controller_start,
    )


if __name__ == "__main__":
    main()
