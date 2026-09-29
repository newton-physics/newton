# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure complete search iterations without claiming numerical qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

from ..fit import FitConfig
from ..run import Config
from ..trajectory import Spline
from . import resident
from .benchmark import _plain, execution_identity
from .engine import Engine
from .provenance import source_snapshot


def _positive(value: str) -> int:
    """Parse a strictly positive iteration or repetition count."""
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _load_bundle(directory: Path):
    """Verify the saved bundle's bytes without reusing its historical qualification."""
    manifest = json.loads((directory / "baseline.json").read_text())
    hashes = manifest["files_sha256"]
    required = {"reference.npz", "profile.json", "equilibrium.npz", "summary.json", "digital_shoe.json"}
    if not required <= hashes.keys():
        raise ValueError("The baseline manifest does not cover all required inputs")
    for name, digest in hashes.items():
        if Path(name).name != name or hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Saved baseline file changed: {name}")
    summary = json.loads((directory / "summary.json").read_text())
    profile = json.loads((directory / "profile.json").read_text())
    with np.load(directory / "reference.npz", allow_pickle=False) as archive:
        reference = dict(archive)
    with np.load(directory / "equilibrium.npz", allow_pickle=False) as archive:
        initial = Spline(float(archive["duration_s"]), archive["coefficients"])
    if initial.coefficients.shape != (12, 4):
        raise ValueError("The saved controller must contain twelve controls per channel")
    return reference, profile, initial, summary, manifest


def profile_search(directory: Path, output: Path, *, iterations: int = 10, repeats: int = 3, seed: int = 17) -> dict:
    """Benchmark real poll-plus-trial searches from a hash-verified saved controller.

    Historical runtime hashes are not numerical permission for changed code.
    This timing-only command writes no qualified controller or acceptance report.
    Use the complete pipeline to qualify any changed runtime and its winner.
    """
    for name, value in (("iterations", iterations), ("repeats", repeats)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if output.exists():
        raise FileExistsError(output)
    started = perf_counter()
    reference, profile, initial, baseline, manifest = _load_bundle(directory)
    sources = source_snapshot()
    sources["resident.py"] = hashlib.sha256(Path(resident.__file__).read_bytes()).hexdigest()
    sources["profile_search.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.mkdir(parents=True)
    shoe = baseline["shoe"]
    engine = Engine(
        reference,
        profile,
        directory / "digital_shoe.json",
        shoe["mount_m"],
        shoe["static_pitch_rad"],
        config=Config(**baseline["simulation_config"]),
        settings=FitConfig(**baseline["fit_config"]),
        world_count=128,
        friction_model=shoe["friction_model"],
    )
    engine.capture(np.repeat(initial.coefficients[None], 128, axis=0))
    wp.load_module(module=resident, device=engine.device)
    # Warm graph replay and GPU clocks outside measured searches.
    for _ in range(3):
        engine.evaluate_device()
        wp.synchronize_device(engine.device)
    setup_wall_s = perf_counter() - started
    runs = []
    for repeat in range(repeats):
        _, _, _, summary = resident.fit_resident(
            engine,
            initial,
            max_iterations=iterations,
            max_wall_s=1.0e9,
            seed=seed,
            plateau_patience=None,
        )
        if summary["iterations_completed"] != iterations or not summary["complete"]:
            raise RuntimeError("The benchmark did not finish every requested iteration")
        runs.append(summary)
        (output / f"search_{repeat:02d}.json").write_text(json.dumps(_plain(summary), indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "repeat": repeat,
                    "seconds_per_iteration": summary["wall_s"] / iterations,
                    "completed_candidates_per_second": summary["completed_candidates_per_second"],
                    "initial_loss": summary["initial_loss"],
                    "final_loss": summary["loss"],
                }
            ),
            flush=True,
        )
    current_sources = source_snapshot()
    current_sources["resident.py"] = hashlib.sha256(Path(resident.__file__).read_bytes()).hexdigest()
    current_sources["profile_search.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if sources != current_sources:
        raise RuntimeError("Runtime source changed during the benchmark")
    seconds = [run["wall_s"] / iterations for run in runs]
    total_search_s = sum(run["wall_s"] for run in runs)
    completed = sum(run["counts"]["completed_real_candidates"] for run in runs)
    report = {
        "schema": "cartesian_search_profile_1",
        "qualification": "Timing only; not numerical qualification, convergence, or measured-fit acceptance.",
        "numerically_qualified": False,
        "baseline": str(directory.resolve()),
        "baseline_manifest": manifest,
        "source_sha256": sources,
        "execution_identity": execution_identity(),
        "simulation_config": baseline["simulation_config"],
        "worlds": 128,
        "steps_per_rollout": engine.steps,
        "shoe_columns": engine.foundation.column_count,
        "iterations_per_repeat": iterations,
        "repeats": repeats,
        "seed": seed,
        "seconds_per_iteration": seconds,
        "median_seconds_per_iteration": float(np.median(seconds)),
        "completed_candidates_per_second": completed / total_search_s,
        "completed_real_candidates": completed,
        "search_wall_s": total_search_s,
        "setup_and_warmup_wall_s": setup_wall_s,
        "total_wall_s": perf_counter() - started,
        "losses": [{"initial": run["initial_loss"], "final": run["loss"]} for run in runs],
        "counts": [run["counts"] for run in runs],
        "iteration_definition": "One complete 128-world poll plus one complete 128-world solve-and-trial batch",
    }
    (output / "profile.json").write_text(json.dumps(_plain(report), indent=2, allow_nan=False) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    """Run the reusable timing-only search benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("outputs/impedance_instron/baseline12_maxwell"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=_positive, default=10)
    parser.add_argument("--repeats", type=_positive, default=3)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    profile_search(args.baseline, args.output, iterations=args.iterations, repeats=args.repeats, seed=args.seed)


if __name__ == "__main__":
    main()
