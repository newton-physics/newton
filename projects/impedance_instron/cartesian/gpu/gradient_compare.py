# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare a gradient experiment with the original 192-world fitter from identical inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ..fit import FitConfig
from ..run import Config
from ..trajectory import Spline
from .adjoint_audit import _load_saved_fit
from .benchmark import _plain
from .engine import Engine
from .provenance import source_snapshot
from .resident import fit_resident


def compare(directory: Path, gradient_report: Path, output: Path):
    """Run the original fitter with the same starting controller and iteration budget."""
    if output.exists():
        raise FileExistsError(output)
    gradient = json.loads(gradient_report.read_text())
    if Path(gradient["baseline"]).resolve() != directory.resolve() or gradient["source_sha256"] != source_snapshot():
        raise ValueError("The gradient experiment uses different baseline or physics sources")
    reference, profile, initial, summary, shoe_path = _load_saved_fit(directory)
    if gradient["start"] == "unfitted":
        metadata = json.loads((directory / "optimization_inputs.json").read_text())
        initial = Spline(initial.duration_s, np.asarray(metadata["starting_coefficients"], dtype=np.float64))
    shoe = summary["shoe"]
    started = perf_counter()
    engine = Engine(
        reference,
        profile,
        shoe_path,
        shoe["mount_m"],
        shoe["static_pitch_rad"],
        config=Config(**summary["simulation_config"]),
        settings=FitConfig(**summary["fit_config"]),
        world_count=192,
        friction_model=shoe["friction_model"],
    )
    engine.capture(np.repeat(initial.coefficients[None], 192, axis=0))
    engine.evaluate(initial.coefficients[None].repeat(192, axis=0))
    setup_s = perf_counter() - started
    winner, trace, run, resident = fit_resident(
        engine,
        initial,
        max_iterations=len(gradient["progress"]),
        max_wall_s=1e9,
        seed=17,
        plateau_patience=None,
    )
    if resident["initial_loss"] != gradient["initial_scores"]["loss"][0]:
        raise RuntimeError("The comparison did not begin with identical measured loss")
    output.mkdir(parents=True)
    np.savez_compressed(output / "resident_trace.npz", **trace)
    np.savez(output / "resident_controller.npz", coefficients=winner.coefficients, duration_s=winner.duration_s)
    report = {
        "schema": "cartesian_gradient_resident_comparison_1",
        "scope": "Matched short experiments; iteration count does not establish time to equal quality or half-step acceptance.",
        "baseline": str(directory.resolve()),
        "gradient_report": str(gradient_report.resolve()),
        "gradient_report_sha256": hashlib.sha256(gradient_report.read_bytes()).hexdigest(),
        "start": gradient["start"],
        "gradient": {
            "iterations": gradient["iterations_completed"],
            "initial_loss": gradient["initial_scores"]["loss"][0],
            "final_loss": gradient["final_scores"]["loss"][0],
            "search_wall_s": gradient["search_wall_s"],
            "setup_wall_s": gradient["setup_wall_s"],
            "rmse": gradient["final_scores"]["rmse"][0],
        },
        "resident": resident,
        "resident_setup_wall_s": setup_s,
        "resident_run": run,
        "source_sha256": source_snapshot(),
        "comparison_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "comparison.json").write_text(json.dumps(_plain(report), indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            _plain(
                {
                    "gradient": report["gradient"],
                    "resident_loss": resident["loss"],
                    "resident_search_s": resident["wall_s"],
                }
            )
        ),
        flush=True,
    )
    return report


def main():
    """Run the matched original-fitter experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--gradient-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.baseline, args.gradient_report, args.output)


if __name__ == "__main__":
    main()
