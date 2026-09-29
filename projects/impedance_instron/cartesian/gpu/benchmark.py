# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark full GPU stances against an unchanged frozen CPU result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

from projects.digital_shoe import runtime as shoe_runtime

from ..fit import FitConfig
from ..run import Config
from . import engine as engine_module
from . import foundation as foundation_module
from . import objective as objective_module
from .engine import Engine
from .provenance import source_snapshot, validate_sources


def execution_identity():
    """Record the actual precision policy and native runtime versions."""
    return {
        "warp": wp.__version__,
        "numpy": np.__version__,
        "gpu_name": wp.get_device("cuda:0").name,
        "modules": {
            module.__name__: {
                key: wp.get_module_options(module).get(key)
                for key in ("fuse_fp", "fast_math", "mode", "optimization_level")
            }
            for module in (engine_module, objective_module, foundation_module, shoe_runtime)
        },
    }


def _plain(value):
    """Keep failed numerical diagnostics representable in strict JSON."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def load_frozen(directory: Path):
    """Verify saved data/configuration identities before loading an experiment."""
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    profile = json.loads((directory / "profile.json").read_text())
    with np.load(directory / "reference.npz", allow_pickle=False) as archive:
        reference = dict(archive)
    with np.load(directory / "equilibrium.npz", allow_pickle=False) as archive:
        coefficients = archive["coefficients"].copy()
        duration = float(archive["duration_s"])
        identity = json.loads(str(archive["identity_json"]))
    if not np.isclose(duration, float(reference["time_s"][-1]), rtol=0, atol=1e-12):
        raise ValueError("Frozen equilibrium duration differs from recorded input")
    validate_sources(summary)
    if identity["reference_sha256"] != hashlib.sha256((directory / "reference.npz").read_bytes()).hexdigest():
        raise ValueError("Frozen reference identity changed")
    if identity["profile_sha256"] != hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest():
        raise ValueError("Frozen profile identity changed")
    if identity["simulation_config"] != summary["simulation_config"]:
        raise ValueError("Frozen simulation configuration differs from its summary")
    shoe = summary["shoe"]
    if shoe["sha256"] != hashlib.sha256(Path(shoe["path"]).read_bytes()).hexdigest():
        raise ValueError("Frozen shoe artifact identity changed")
    for key, saved_key in (
        ("sha256", "artifact_sha256"),
        ("mount_m", "mount_m"),
        ("static_pitch_rad", "static_pitch_rad"),
        ("friction", "friction"),
    ):
        if identity["shoe"][saved_key] != shoe[key]:
            raise ValueError(f"Frozen shoe {key} differs from its summary")
    return reference, profile, coefficients, summary


def benchmark(
    directory: Path,
    output: Path,
    *,
    batch: int = 1,
    repeats: int = 2,
    controls: int | None = None,
):
    """Measure actual full GPU rollouts, not isolated kernel throughput."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    reference, profile, coefficients, baseline = load_frozen(directory)
    source_before = source_snapshot()
    source_audit = validate_sources(baseline)

    frozen_artifacts = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("reference.npz", "profile.json", "equilibrium.npz", "trace.npz", "summary.json")
    }
    with np.load(directory / "trace.npz", allow_pickle=False) as archive:
        cpu_trace = dict(archive)
    config, settings = Config(**baseline["simulation_config"]), FitConfig(**baseline["fit_config"])

    if controls not in (None, 12):
        raise ValueError("Control-count conversion is not supported; the frozen controller must use 12 points")
    control_count = len(coefficients)
    channels = len(profile["equilibrium_lower"])
    if coefficients.shape != (12, channels) or channels not in (4, 6) or settings.control_count != 12:
        raise ValueError("The frozen controller and FitConfig must use 12 controls and four or six channels")

    shoe = baseline["shoe"]
    started = perf_counter()
    engine = Engine(
        reference,
        profile,
        shoe["path"],
        shoe["mount_m"],
        shoe["static_pitch_rad"],
        config=config,
        settings=settings,
        world_count=batch,
        friction_model=shoe["friction_model"],
    )
    if engine.shoe.metadata != shoe:
        raise ValueError("GPU shoe metadata differs from frozen CPU registration/material metadata")
    values = np.repeat(coefficients[None], batch, axis=0)
    engine.capture(values)
    cold = perf_counter() - started
    wall = []
    scores = None
    first_loss = None
    for _ in range(repeats):
        scores = engine.evaluate(values)
        wall.append(engine.last_wall_s)
        if first_loss is None:
            first_loss = scores["loss"].copy()
        elif not np.array_equal(first_loss, scores["loss"]):
            raise ValueError("Repeated GPU batch changed its scores after resetting contact histories")
        print(
            json.dumps(
                {
                    "batch": batch,
                    "warm_wall_s": wall[-1],
                    "loss": scores["loss"].tolist(),
                    "failure_code": scores["failure_code"].tolist(),
                }
            ),
            flush=True,
        )
    trace, run = engine.trace(0)
    np.savez_compressed(output / "trace.npz", **trace)
    np.savez_compressed(output / "scores.npz", **scores)
    same_rows = len(trace["time_s"]) == len(cpu_trace["time_s"])
    error = (
        {
            key: float(np.max(np.abs(trace[key] - cpu_trace[key])))
            for key in (
                "state",
                "velocity",
                "grf_n",
                "ankle_contact_moment_nm",
                "equilibrium",
                "compression_fraction",
                "driven_compression_fraction",
                "passive_compression_fraction",
            )
        }
        if same_rows
        else {}
    )
    if same_rows:
        error["hip_m"] = float(np.max(np.linalg.norm(trace["state"][:, :2] - cpu_trace["state"][:, :2], axis=1)))
        error["angle_rad"] = float(np.max(np.abs(trace["state"][:, 2:] - cpu_trace["state"][:, 2:])))
        error["force_vector_n"] = float(np.max(np.linalg.norm(trace["grf_n"] - cpu_trace["grf_n"], axis=1)))
        error["terminal_state"] = float(
            np.max(np.abs(np.array(run["terminal_state"]) - baseline["run"]["terminal_state"]))
        )
        error["terminal_velocity"] = float(
            np.max(np.abs(np.array(run["terminal_velocity"]) - baseline["run"]["terminal_velocity"]))
        )
        error["loss"] = abs(float(scores["loss"][0]) - baseline["loss"])
    limits = {
        "hip_m": 1e-5,
        "angle_rad": 1e-5,
        "force_vector_n": 1.0,
        "ankle_contact_moment_nm": 0.05,
        "compression_fraction": 1e-4,
        "terminal_state": 1e-5,
        "velocity": 1e-3,
        "terminal_velocity": 1e-4,
        "loss": 1e-4,
    }
    equal_clock = bool(same_rows and np.array_equal(trace["time_s"], cpu_trace["time_s"]))
    equal_diagnostics = bool(
        run["passive_cap_steps"] == baseline["run"]["passive_cap_steps"]
        and run["joint_exceedances"] == baseline["run"]["joint_exceedances"]
    )
    parity = bool(
        equal_clock and equal_diagnostics and run["failure"] is None and all(error[k] <= v for k, v in limits.items())
    )
    source_after = source_snapshot()
    if source_before != source_after:
        raise ValueError("A GPU runtime source changed while the benchmark was running")
    isolation = bool(np.all(scores["loss"] == scores["loss"][0]) and np.all(scores["failure_code"] == 0))

    report = {
        "schema": "cartesian_gpu_full_rollout_benchmark_1",
        "baseline": str(directory.resolve()),
        "batch_size": batch,
        "control_count": control_count,
        "device": str(engine.device),
        "gpu_name": engine.device.name,
        "execution_identity": execution_identity(),
        "setup_wall_s": engine.setup_wall_s,
        "capture_wall_s": engine.capture_wall_s,
        "cold_initialization_wall_s": cold,
        "warm_batch_wall_s": wall,
        "warm_seconds_per_candidate": float(np.median(wall)) / batch,
        "cpu_historical_mean_seconds_per_candidate": 18.835,
        "historical_cpu_throughput_speedup": 18.835 * batch / float(np.median(wall)),
        "timing_scope": "coefficients upload, history reset, full resident stance, GPU objective, all batch score/residual readbacks; excludes plotting trace extraction",
        "cpu_fit_started": False,
        "run": run,
        "same_timestep_limits": limits,
        "maximum_absolute_error": error,
        "same_timestep_parity_passed": parity,
        "identical_world_scores_equal": isolation,
        "repeat_scores_exact": True,
        "gpu_loss": float(scores["loss"][0]),
        "cpu_loss": baseline["loss"],
        "sources": source_before,
        "source_audit": source_audit,
        "sources_unchanged_during_run": True,
        "frozen_artifacts_sha256": frozen_artifacts,
        "equal_time_grid": equal_clock,
        "equal_passive_cap_and_joint_diagnostics": equal_diagnostics,
    }
    (output / "benchmark.json").write_text(json.dumps(_plain(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps(_plain(report), indent=2, allow_nan=False), flush=True)
    return report


def main(argv=None):
    """Run the reusable saved-CPU versus resident-GPU stance benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args(argv)
    if args.batch < 1 or args.repeats < 1:
        parser.error("batch and repeats must be positive")
    report = benchmark(args.directory, args.output, batch=args.batch, repeats=args.repeats)
    if not report["same_timestep_parity_passed"] or not report["identical_world_scores_equal"]:
        raise SystemExit("GPU benchmark did not pass full-rollout parity")


if __name__ == "__main__":
    main()
