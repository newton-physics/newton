# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure distinct GPU candidates and verify world isolation and history reset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..fit import FitConfig
from ..run import Config
from ..trajectory import Spline
from .benchmark import _plain, execution_identity, load_frozen
from .engine import Engine
from .provenance import source_snapshot


def benchmark(
    directory: Path,
    output: Path,
    failure_equilibrium: Path | None = None,
    *,
    worlds: int | None = None,
    controls: int | None = None,
):
    """Compare real mixed candidates under permutation, repetition, and isolation."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    reference, profile, coefficients, baseline = load_frozen(directory)
    config, settings = Config(**baseline["simulation_config"]), FitConfig(**baseline["fit_config"])
    sources = source_snapshot()

    if controls not in (None, 12):
        raise ValueError("Control-count conversion is not supported; the frozen controller must use 12 points")
    control_count = len(coefficients)
    if coefficients.shape != (12, 4) or settings.control_count != 12:
        raise ValueError("The frozen controller and FitConfig must use exactly 12 control points")

    poll_count = 1 + 2 * coefficients.size
    if worlds not in (None, 128):
        raise ValueError("The mixed-world benchmark requires exactly 128 worlds")
    count = 128
    candidates = np.repeat(coefficients[None], count, axis=0)
    bounds = [
        profile[key]
        for key in (
            "equilibrium_lower",
            "equilibrium_upper",
            "equilibrium_rate_limit",
            "equilibrium_acceleration_limit",
        )
    ]
    valid = np.ones(count, dtype=bool)
    for coordinate in range(coefficients.size):
        for index, sign in enumerate((1, -1)):
            w = 1 + 2 * coordinate + index
            candidates[w].flat[coordinate] += sign * 0.05 * settings.parameter_scale[coordinate % 4]
            if not Spline(float(reference["time_s"][-1]), candidates[w]).bounds(*bounds):
                valid[w] = False
                candidates[w] = coefficients
    # Qualify larger batches with distinct bounded controllers, not repeated baselines.
    rng = np.random.default_rng(20260916)
    for world in range(poll_count, count):
        delta = rng.uniform(-0.05, 0.05, coefficients.shape) * np.asarray(settings.parameter_scale)
        for _ in range(40):
            candidate = coefficients + delta
            if Spline(float(reference["time_s"][-1]), candidate).bounds(*bounds):
                candidates[world] = candidate
                break
            delta *= 0.5
        else:
            valid[world] = False
    if failure_equilibrium is not None:
        with np.load(failure_equilibrium, allow_pickle=False) as archive:
            failure_candidate = archive["coefficients"].copy()
        if failure_candidate.shape != coefficients.shape or not Spline(
            float(reference["time_s"][-1]), failure_candidate
        ).bounds(*bounds):
            raise ValueError("The failure fixture must use the same bounded coefficient shape")
        candidates[-1] = failure_candidate
    shoe = baseline["shoe"]
    engine = Engine(
        reference,
        profile,
        shoe["path"],
        shoe["mount_m"],
        shoe["static_pitch_rad"],
        config=config,
        settings=settings,
        world_count=count,
        friction_model=shoe["friction_model"],
    )
    engine.capture(candidates)
    first = engine.evaluate(candidates)
    walls = [engine.last_wall_s]
    first_q, first_v, first_f = engine.states.numpy(), engine.velocities.numpy(), engine.forces.numpy()
    recorded = engine.recorded.numpy()
    permutation = np.roll(np.arange(count)[::-1], 7)
    second = engine.evaluate(candidates[permutation])
    walls.append(engine.last_wall_s)
    second_q, second_v, second_f = engine.states.numpy(), engine.velocities.numpy(), engine.forces.numpy()
    inverse = np.argsort(permutation)
    permutation_exact = True
    for world in range(count):
        other = int(inverse[world])
        integrated = int(first["integrated_steps"][world])
        permutation_exact = permutation_exact and all(
            (
                first["failure_code"][world] == second["failure_code"][other],
                integrated == int(second["integrated_steps"][other]),
                np.array_equal(first_q[: integrated + 1, world], second_q[: integrated + 1, other]),
                np.array_equal(first_v[: integrated + 1, world], second_v[: integrated + 1, other]),
                np.array_equal(first_f[: recorded[world], world], second_f[: recorded[world], other]),
                np.array_equal(first["residual"][:, world], second["residual"][:, other]),
            )
        )
    third = engine.evaluate(candidates)
    walls.append(engine.last_wall_s)
    third_q, third_v, third_f = engine.states.numpy(), engine.velocities.numpy(), engine.forces.numpy()
    reset_exact = True
    for world in range(count):
        integrated = int(first["integrated_steps"][world])
        reset_exact = reset_exact and all(
            (
                first["failure_code"][world] == third["failure_code"][world],
                integrated == int(third["integrated_steps"][world]),
                np.array_equal(first_q[: integrated + 1, world], third_q[: integrated + 1, world]),
                np.array_equal(first_v[: integrated + 1, world], third_v[: integrated + 1, world]),
                np.array_equal(first_f[: recorded[world], world], third_f[: recorded[world], world]),
                np.array_equal(first["residual"][:, world], third["residual"][:, world]),
            )
        )
    single = Engine(
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
    single.capture(coefficients[None])
    isolated = []
    # Include a changed controller and the optional known-failing controller.
    for world in sorted({0, 1, count // 2, count - 1}):
        score = single.evaluate(candidates[world : world + 1])
        integrated = int(first["integrated_steps"][world])
        q, v, f = single.states.numpy(), single.velocities.numpy(), single.forces.numpy()
        exact = all(
            (
                score["failure_code"][0] == first["failure_code"][world],
                int(score["integrated_steps"][0]) == integrated,
                np.array_equal(q[: integrated + 1, 0], first_q[: integrated + 1, world]),
                np.array_equal(v[: integrated + 1, 0], first_v[: integrated + 1, world]),
                np.array_equal(f[: recorded[world], 0], first_f[: recorded[world], world]),
                np.array_equal(score["residual"][:, 0], first["residual"][:, world]),
            )
        )
        isolated.append(
            {
                "world": world,
                "exact": bool(exact),
                "warm_wall_s": single.last_wall_s,
                "failure_code": int(score["failure_code"][0]),
            }
        )
    source_unchanged = sources == source_snapshot()
    if not source_unchanged:
        raise ValueError("GPU runtime sources changed during the batch benchmark")
    failure_exercised = bool(failure_equilibrium is None or first["failure_code"][-1] != 0)
    report = {
        "schema": "cartesian_gpu_mixed_world_benchmark_1",
        "batch_size": count,
        "control_count": control_count,
        "device": str(engine.device),
        "gpu_name": engine.device.name,
        "execution_identity": execution_identity(),
        "sources": sources,
        "sources_unchanged_during_run": True,
        "frozen_artifacts_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("reference.npz", "profile.json", "equilibrium.npz", "trace.npz", "summary.json")
        },
        "warm_batch_wall_s": walls,
        "unique_candidates": len({row.tobytes() for row in candidates}),
        "warm_seconds_per_physical_candidate": float(np.median(walls)) / count,
        "historical_cpu_throughput_speedup": 18.835 * count / float(np.median(walls)),
        "timing_scope": "full resident stance and GPU objective including batch-boundary coefficient and score copies; validation history copies are excluded",
        "permutation_full_history_exact": bool(permutation_exact),
        "reset_full_history_exact": bool(reset_exact),
        "isolated_worlds": isolated,
        "failure_fixture_exercised": failure_exercised,
        "failure_equilibrium": str(failure_equilibrium) if failure_equilibrium else None,
        "failure_codes": first["failure_code"].tolist(),
        "losses": first["loss"].tolist(),
        "passed": bool(
            permutation_exact and reset_exact and all(row["exact"] for row in isolated) and failure_exercised
        ),
    }
    np.savez_compressed(output / "candidates_and_scores.npz", coefficients=candidates, **first)
    (output / "benchmark.json").write_text(json.dumps(_plain(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps(_plain(report), indent=2, allow_nan=False), flush=True)
    return report


def main(argv: list[str] | None = None):
    """Run the reproducible mixed-controller batch benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure-equilibrium", type=Path)
    parser.add_argument("--worlds", type=int, default=128, help="Qualify exactly 128 distinct-controller worlds.")
    args = parser.parse_args(argv)
    if not benchmark(
        args.directory,
        args.output,
        args.failure_equilibrium,
        worlds=args.worlds,
    )["passed"]:
        raise SystemExit("Mixed GPU batch did not pass isolation/reset validation")


if __name__ == "__main__":
    main()
