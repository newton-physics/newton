# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a safeguarded gradient fitting experiment after full-stance qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ..fit import FitConfig
from ..run import Config
from ..spline import _derivative_control_polygons
from ..trajectory import Spline
from .adjoint import EngineAdjoint
from .adjoint_audit import _load_saved_fit
from .benchmark import _plain
from .engine import Engine
from .provenance import source_snapshot


def _direction(gradient, history):
    """Apply the L-BFGS inverse Hessian in normalized coefficient coordinates."""
    result = gradient.copy()
    alphas = []
    for displacement, change in reversed(history):
        alpha = float(displacement @ result / (change @ displacement))
        alphas.append(alpha)
        result -= alpha * change
    if history:
        displacement, change = history[-1]
        result *= float(displacement @ change / (change @ change))
    for (displacement, change), alpha in zip(history, reversed(alphas), strict=True):
        beta = float(change @ result / (change @ displacement))
        result += displacement * (alpha - beta)
    return -result


def _constraints(profile, duration, scale):
    """Express the existing spline control-polygon bounds as normalized halfspaces."""
    channels = len(profile["equilibrium_lower"])
    first, second = _derivative_control_polygons(np.eye(12))
    rows, limits = [], []
    for channel in range(channels):
        for matrix, lower, upper in (
            (np.eye(12), profile["equilibrium_lower"][channel], profile["equilibrium_upper"][channel]),
            (
                first,
                -profile["equilibrium_rate_limit"][channel] * duration,
                profile["equilibrium_rate_limit"][channel] * duration,
            ),
            (
                second,
                -profile["equilibrium_acceleration_limit"][channel] * duration**2,
                profile["equilibrium_acceleration_limit"][channel] * duration**2,
            ),
        ):
            for coefficients in matrix:
                row = np.zeros(12 * channels)
                row[channel::channels] = coefficients
                row *= scale
                norm = np.linalg.norm(row)
                rows.extend((row / norm, -row / norm))
                limits.extend((upper / norm, -lower / norm))
    return np.asarray(rows), np.asarray(limits)


def _project_proposal(x, direction, matrix, limits):
    """Project the proposal onto coupled spline bounds using Dykstra halfspace updates."""
    proposal = x + direction
    corrections = np.zeros(len(limits))
    target = limits - 1e-10
    cycles = 0
    for cycle in range(1, 2001):
        cycles = cycle
        before = proposal.copy()
        for i, row in enumerate(matrix):
            proposal += corrections[i] * row
            corrections[i] = max(0.0, float(row @ proposal - target[i]))
            proposal -= corrections[i] * row
        violation = float(np.max(matrix @ proposal - target))
        if violation <= 1e-11 and np.linalg.norm(proposal - before) <= 1e-10:
            break
    return proposal - x, {"cycles": cycles, "maximum_violation": float(np.max(matrix @ proposal - limits))}


def _qualification(path, directory, steps, start):
    """Require passing measured gradients for the exact current saved experiment."""
    report = json.loads(path.read_text())
    if not (
        report.get("forward_passed") is True
        and report.get("gradient_passed") is True
        and report.get("objective") == "measured"
        and report.get("start_step") == 0
        and report.get("steps") == steps
        and report.get("controller_start") == start
        and report.get("optimizer_qualified") is True
        and len(report.get("directions", [])) >= 3
        and all(item.get("passed") is True for item in report.get("directions", []))
    ):
        raise ValueError("A passing complete measured-objective gradient audit is required")
    if report.get("source_sha256") != source_snapshot():
        raise ValueError("Physics sources differ from the gradient audit")
    for name, digest in report["adjoint_sources_sha256"].items():
        if hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Gradient source changed after qualification: {name}")
    if Path(report.get("baseline", "")).resolve() != directory.resolve():
        raise ValueError("The gradient audit belongs to a different baseline")
    expected = report.get("input_sha256", {})
    required = {"reference.npz", "profile.json", "equilibrium.npz", "optimization_inputs.json", "summary.json"}
    if not required <= expected.keys():
        raise ValueError("The gradient audit must record all saved input hashes")
    for name in required:
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected[name]:
            raise ValueError(f"Saved input differs from gradient qualification: {name}")
    return report


def experiment(
    directory: Path,
    qualification: Path,
    output: Path,
    *,
    iterations: int = 5,
    wall_seconds: float = 300.0,
    start: str = "fitted",
    target_loss: float | None = None,
):
    """Fit the original measured objective using bounded L-BFGS and production scores."""
    if output.exists():
        raise FileExistsError(output)
    if iterations < 1 or not np.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError("Iterations and wall budget must be positive")
    if target_loss is not None and (not np.isfinite(target_loss) or target_loss <= 0):
        raise ValueError("Target loss must be finite and positive")
    started = perf_counter()
    reference, profile, spline, summary, shoe_path = _load_saved_fit(directory)
    shoe = summary["shoe"]
    settings = FitConfig(**summary["fit_config"])
    engine = Engine(
        reference,
        profile,
        shoe_path,
        shoe["mount_m"],
        shoe["static_pitch_rad"],
        config=Config(**summary["simulation_config"]),
        settings=settings,
        world_count=1,
        friction_model=shoe["friction_model"],
    )
    audited = _qualification(qualification, directory, engine.steps, start)
    coefficients = spline.coefficients.copy()
    if start == "unfitted":
        metadata = json.loads((directory / "optimization_inputs.json").read_text())
        coefficients = np.asarray(metadata["starting_coefficients"], dtype=np.float64)
    elif start != "fitted":
        raise ValueError("Start must be fitted or unfitted")
    if hashlib.sha256(np.ascontiguousarray(coefficients[None]).tobytes()).hexdigest() != audited.get(
        "coefficient_sha256"
    ):
        raise ValueError("Starting coefficients differ from the gradient audit")
    engine.capture(coefficients[None])
    engine._reset()
    adjoint = EngineAdjoint(engine, engine.steps, objective="measured")
    adjoint.coefficients.assign(coefficients[None])
    adjoint.capture()
    scores = engine.evaluate(coefficients[None])
    initial_scores = _plain(scores)
    result = adjoint.value_and_grad()
    if not result["valid"] or result["loss"] != float(scores["loss"][0]):
        raise RuntimeError("Initial gradient rollout does not match production loss")
    setup_s = perf_counter() - started
    output.mkdir(parents=True)
    bounds = [
        profile[name]
        for name in (
            "equilibrium_lower",
            "equilibrium_upper",
            "equilibrium_rate_limit",
            "equilibrium_acceleration_limit",
        )
    ]
    scale = np.broadcast_to(settings.parameter_scale, coefficients.shape).ravel()
    constraint_matrix, constraint_limits = _constraints(profile, spline.duration_s, scale)
    x = coefficients.ravel() / scale
    gradient = result["gradient"].ravel() * scale
    loss = result["loss"]
    history = []
    progress = []
    search_started = perf_counter()
    status = "iteration_limit"
    tolerances = np.asarray(
        [settings.hip_tolerance_m] * 2 + [settings.joint_tolerance_rad] * 2 + [settings.force_tolerance_n] * 2
    )
    evaluations = 1
    gradient_evaluations = 1
    for iteration in range(iterations):
        if target_loss is not None and loss <= target_loss and np.all(scores["rmse"][0] <= tolerances):
            status = "target_reached"
            break
        if perf_counter() - search_started >= wall_seconds:
            status = "wall_limit"
            break
        direction = _direction(gradient, history)
        if not np.isfinite(direction).all() or float(gradient @ direction) >= 0:
            history.clear()
            direction = -gradient
        magnitude = float(np.max(np.abs(direction)))
        if magnitude == 0:
            status = "zero_gradient"
            break
        direction *= min(1.0, settings.initial_step_fraction / magnitude)
        projection_started = perf_counter()
        direction, projection = _project_proposal(x, direction, constraint_matrix, constraint_limits)
        projection["wall_s"] = perf_counter() - projection_started
        slope = float(gradient @ direction)
        if slope >= 0:
            history.clear()
            direction = -gradient
            direction *= settings.initial_step_fraction / float(np.max(np.abs(direction)))
            direction, projection = _project_proposal(x, direction, constraint_matrix, constraint_limits)
            slope = float(gradient @ direction)
        if slope >= 0 or not np.isfinite(direction).all():
            status = "projected_direction_stalled"
            break
        trials = []
        accepted = None
        alpha = 1.0
        for _ in range(24):
            if perf_counter() - search_started >= wall_seconds:
                break
            candidate_x = x + alpha * direction
            candidate = (candidate_x * scale).reshape(coefficients.shape)
            if not Spline(spline.duration_s, candidate).bounds(*bounds):
                trials.append({"alpha": alpha, "reason": "spline_bounds"})
                alpha *= 0.5
                continue
            trial_scores = engine.evaluate(candidate[None])
            evaluations += 1
            valid = bool(trial_scores["failure_code"][0] == 0 and trial_scores["integrated_steps"][0] == engine.steps)
            trial_loss = float(trial_scores["loss"][0])
            valid = valid and np.isfinite(trial_loss)
            armijo = valid and trial_loss < loss and trial_loss <= loss + 1e-4 * alpha * slope
            trials.append({"alpha": alpha, "loss": trial_loss, "valid": valid, "accepted": armijo})
            if armijo:
                adjoint.coefficients.assign(candidate[None])
                candidate_result = adjoint.value_and_grad()
                gradient_evaluations += 1
                if not candidate_result["valid"] or candidate_result["loss"] != trial_loss:
                    raise RuntimeError("Candidate gradient rollout does not match production loss")
                accepted = (candidate_x, candidate, candidate_result, trial_scores)
                break
            alpha *= 0.5
        row = {"iteration": iteration + 1, "loss_before": loss, "projection": projection, "trials": trials}
        if accepted is None:
            status = "wall_limit" if perf_counter() - search_started >= wall_seconds else "line_search_stalled"
            row.update(accepted=False, loss=loss)
            progress.append(row)
            break
        next_x, coefficients, result, scores = accepted
        next_gradient = result["gradient"].ravel() * scale
        displacement, change = next_x - x, next_gradient - gradient
        curvature = float(displacement @ change)
        if curvature > 1e-12 * np.linalg.norm(displacement) * np.linalg.norm(change) and curvature > 0:
            history.append((displacement, change))
            history = history[-10:]
        x, gradient, loss = next_x, next_gradient, result["loss"]
        row.update(
            accepted=True,
            loss=loss,
            rmse=scores["rmse"][0],
            gradient_timing=result["timings"],
            search_wall_s=perf_counter() - search_started,
        )
        progress.append(row)
        (output / "progress.json").write_text(json.dumps(_plain(progress), indent=2, allow_nan=False) + "\n")
        np.savez(output / "gradient_controller.npz", coefficients=coefficients, duration_s=spline.duration_s)
        print(json.dumps(_plain(row), allow_nan=False), flush=True)
    search_s = perf_counter() - search_started
    final_scores = engine.evaluate(coefficients[None])
    trace, run = engine.trace()
    np.savez_compressed(output / "trace.npz", **trace)
    np.savez(output / "gradient_controller.npz", coefficients=coefficients, duration_s=spline.duration_s)
    report = {
        "schema": "cartesian_gradient_fitting_experiment_1",
        "qualification": "Experimental full-stance gradient optimization; no half-step or measured-fit acceptance qualification.",
        "optimizer": "normalized L-BFGS with Dykstra proposal projection, strict spline bounds and production Armijo backtracking",
        "baseline": str(directory.resolve()),
        "gradient_audit": str(qualification.resolve()),
        "gradient_audit_sha256": hashlib.sha256(qualification.read_bytes()).hexdigest(),
        "start": start,
        "status": status,
        "target_loss": target_loss,
        "setup_wall_s": setup_s,
        "search_wall_s": search_s,
        "total_wall_s": perf_counter() - started,
        "initial_scores": initial_scores,
        "final_scores": final_scores,
        "production_run": run,
        "iterations_completed": sum(item["accepted"] for item in progress),
        "production_evaluations": evaluations,
        "gradient_evaluations": gradient_evaluations,
        "progress": progress,
        "source_sha256": source_snapshot(),
        "adjoint_sources_sha256": audited["adjoint_sources_sha256"],
        "optimizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "report.json").write_text(json.dumps(_plain(report), indent=2, allow_nan=False) + "\n")
    (output / "gradient_fit_source.py").write_text(Path(__file__).read_text())
    return report


def main():
    """Run the qualified single-stance optimizer experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--wall-seconds", type=float, default=300.0)
    parser.add_argument("--start", choices=("fitted", "unfitted"), default="fitted")
    parser.add_argument("--target-loss", type=float)
    args = parser.parse_args()
    experiment(
        args.baseline,
        args.qualification,
        args.output,
        iterations=args.iterations,
        wall_seconds=args.wall_seconds,
        start=args.start,
        target_loss=args.target_loss,
    )


if __name__ == "__main__":
    main()
