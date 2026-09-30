# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Apply the existing measured and half-timestep checks to an experimental controller."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..fit import FitConfig, _refinement
from ..run import Config
from .adjoint_audit import _load_saved_fit
from .benchmark import _plain
from .engine import Engine
from .provenance import source_snapshot


def qualify(fit_directory: Path, output: Path):
    """Score the saved winner independently at the original and half timesteps."""
    if output.exists():
        raise FileExistsError(output)
    fitted = json.loads((fit_directory / "report.json").read_text())
    if fitted["source_sha256"] != source_snapshot():
        raise ValueError("Physics sources changed after the fitting experiment")
    baseline = Path(fitted["baseline"])
    reference, profile, initial, summary, shoe_path = _load_saved_fit(baseline)
    with np.load(fit_directory / "gradient_controller.npz", allow_pickle=False) as archive:
        coefficients = archive["coefficients"].copy()
        if float(archive["duration_s"]) != initial.duration_s:
            raise ValueError("Controller duration differs from its original reference")
    shoe = summary["shoe"]
    settings = FitConfig(**summary["fit_config"])
    config = Config(**summary["simulation_config"])

    def run(configuration):
        engine = Engine(
            reference,
            profile,
            shoe_path,
            shoe["mount_m"],
            shoe["static_pitch_rad"],
            config=configuration,
            settings=settings,
            world_count=1,
            friction_model=shoe["friction_model"],
        )
        scores = engine.evaluate(coefficients[None])
        trace, result = engine.trace()
        return scores, trace, result

    native, trace, native_run = run(config)
    fine, fine_trace, fine_run = run(replace(config, dt_s=config.dt_s / 2))
    refinement = _refinement(trace, native_run, fine_trace, fine_run, initial.duration_s, settings)
    tolerances = np.asarray(
        [settings.hip_tolerance_m] * 2 + [settings.joint_tolerance_rad] * 2 + [settings.force_tolerance_n] * 2
    )
    native_passed = bool(native["failure_code"][0] == 0 and np.all(native["rmse"][0] <= tolerances))
    fine_passed = bool(fine["failure_code"][0] == 0 and np.all(fine["rmse"][0] <= tolerances))
    report = {
        "schema": "cartesian_gradient_controller_numerical_qualification_1",
        "fit_directory": str(fit_directory.resolve()),
        "controller_sha256": hashlib.sha256((fit_directory / "gradient_controller.npz").read_bytes()).hexdigest(),
        "accepted": native_passed and fine_passed and bool(refinement["passed"]),
        "native_within_measured_tolerances": native_passed,
        "fine_within_measured_tolerances": fine_passed,
        "native_scores": native,
        "fine_scores": fine,
        "refinement": refinement,
        "source_sha256": source_snapshot(),
        "qualification_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.mkdir(parents=True)
    np.savez_compressed(output / "native_trace.npz", **trace)
    np.savez_compressed(output / "refined_trace.npz", **fine_trace)
    (output / "qualification.json").write_text(json.dumps(_plain(report), indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(_plain({"accepted": report["accepted"], "native_loss": native["loss"], "refinement": refinement})),
        flush=True,
    )
    return report


def main():
    """Qualify one gradient experiment's frozen winner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qualify(args.fit_directory, args.output)


if __name__ == "__main__":
    main()
