# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct one saved local probe and qualify it without another search."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

from ..fit import FitConfig, _refinement
from ..run import Config
from ..springs import load_springs
from ..trajectory import Spline
from . import local_probe
from .__main__ import _validation, _write_final_report, _write_json, _write_npz
from .benchmark import execution_identity, load_frozen
from .engine import Engine
from .provenance import source_snapshot, validate_sources


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _within(rmse, settings: FitConfig) -> bool:
    values = np.asarray(rmse, dtype=np.float64)
    limits = np.repeat([settings.hip_tolerance_m, settings.joint_tolerance_rad, settings.force_tolerance_n], 2)
    return bool(values.shape == (6,) and np.isfinite(values).all() and np.all(values >= 0) and np.all(values <= limits))


def replay_probe(probe_directory: Path, output: Path, *, probe_index: int | None = None) -> dict:
    """Apply the existing half-step and spring checks to a saved bounded probe.

    The default selects the lowest-loss probe that met all native-step RMS
    limits, not necessarily the lowest-loss probe overall. No search, force law,
    gain, target, bound, or acceptance threshold is changed.

    Args:
        probe_directory: Completed local-probe output directory.
        output: New replay/report directory.
        probe_index: Explicit saved probe index, or select by native RMS limits.
    """
    started = perf_counter()
    probe_directory, output = Path(probe_directory), Path(output)
    if output.exists():
        raise FileExistsError(output)
    reference, profile, _, saved = load_frozen(probe_directory)
    inputs = json.loads((probe_directory / "optimization_inputs.json").read_text())
    if inputs["probe_source_sha256"] != _sha(Path(local_probe.__file__)):
        raise ValueError("Probe generator changed; refusing to regenerate saved candidates")
    if not saved.get("complete") or saved.get("status") != "completed":
        raise ValueError("Local probe did not complete")
    baseline_directory = Path(inputs["baseline_directory"])
    single = Path(inputs["single_validation"])
    mixed = Path(inputs["mixed_validation"])
    contact = Path(inputs["contact_rounding_evidence"])
    for path, key in (
        (single, "single_validation_sha256"),
        (mixed, "mixed_validation_sha256"),
        (contact, "contact_rounding_evidence_sha256"),
    ):
        if _sha(path) != inputs[key]:
            raise ValueError("Saved numerical qualification changed")
    _validation(single, baseline_directory, mixed=False, expected_controls=12, contact_evidence=contact)
    _validation(mixed, baseline_directory, mixed=True, expected_controls=12)
    starting = Path(inputs["starting_controller"])
    if _sha(starting) != inputs["starting_controller_sha256"]:
        raise ValueError("Original fitted controller changed")
    baseline_summary = json.loads((baseline_directory / "summary.json").read_text())
    fitted_summary = json.loads((starting.parent / "summary.json").read_text())
    for key in ("simulation_config", "fit_config"):
        if saved[key] != inputs[key] or saved[key] != fitted_summary[key]:
            raise ValueError("Probe settings differ from the original fitted experiment")
        # Fit and qualification budgets can differ, but their physics and tolerances cannot.
        ignored = {"max_evaluations"} if key == "fit_config" else set()
        compared = {name: value for name, value in saved[key].items() if name not in ignored}
        baseline_settings = {name: value for name, value in baseline_summary[key].items() if name not in ignored}
        if compared != baseline_settings:
            raise ValueError("Probe settings differ from the qualified baseline")
    config = Config(**saved["simulation_config"])
    settings = FitConfig(**saved["fit_config"])
    probe_records_file = json.loads((probe_directory / "probes.json").read_text())
    records = probe_records_file["probe_records"]
    seed = probe_records_file["seed"]
    count = inputs["probe_plan"]["random_direction_count"]
    if (
        count != saved["probe_plan"]["random_direction_count"]
        or count != probe_records_file["random_direction_library"]["count"]
    ):
        raise ValueError("Saved direction counts disagree")
    expected_specs = local_probe._probe_specs(count)
    if len(records) != len(expected_specs):
        raise ValueError("Saved probe count differs from the qualified generator")
    for record, spec in zip(records, expected_specs, strict=True):
        if any(record.get(key) != value for key, value in spec.items()):
            raise ValueError("Saved proposal metadata differs from the qualified generator")
    eligible = [r for r in records if r["bounded"] and r["completed"] and _within(r["rmse"], settings)]
    if probe_index is None:
        if not eligible:
            raise ValueError("No saved probe meets all native-step RMS limits")
        chosen = min(eligible, key=lambda record: record["loss"])
    else:
        if isinstance(probe_index, bool) or not isinstance(probe_index, int) or not 0 <= probe_index < len(records):
            raise ValueError("probe_index must identify one saved probe")
        chosen = records[probe_index]
    if not chosen["bounded"] or not chosen["completed"]:
        raise ValueError("Only complete bounded probes can be replayed")
    with np.load(probe_directory / "probe_arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    with np.load(starting, allow_pickle=False) as archive:
        if not np.array_equal(arrays["baseline_coefficients"], archive["coefficients"]):
            raise ValueError("Probe baseline differs from the original fitted coefficients")
        identity = json.loads(str(archive["identity_json"]))
    local_probe._verify_starting_controller_identity(starting, baseline_directory, baseline_summary, profile)
    if identity != saved["identity"] or int(arrays["random_seed"]) != seed:
        raise ValueError("Saved controller identity or direction seed changed")
    if len(records) != len(arrays["kind_code"]):
        raise ValueError("Saved probe records and proposal arrays disagree")
    for record in records:
        i = record["probe_index"]
        for key in ("kind_code", "coord_index", "sign", "scale_fraction"):
            if arrays[key][i] != record[key]:
                raise ValueError("Saved probe metadata and proposal arrays disagree")
        expected_random = -1 if record["random_direction_index"] is None else record["random_direction_index"]
        if arrays["random_direction_index"][i] != expected_random:
            raise ValueError("Saved smooth-direction index changed")
    sources, caller_hash = source_snapshot(), _sha(Path(__file__))
    output.mkdir(parents=True)
    for name in ("reference.npz", "profile.json"):
        (output / name).write_bytes((probe_directory / name).read_bytes())
    _write_json(
        output / "replay_inputs.json",
        {
            "command": sys.argv,
            "probe_directory": str(probe_directory.resolve()),
            "probe_index": chosen["probe_index"],
            "probe_source_sha256": inputs["probe_source_sha256"],
            "replay_source_sha256": caller_hash,
            "artifacts_sha256": {
                name: _sha(probe_directory / name)
                for name in ("summary.json", "probes.json", "probe_arrays.npz", "optimization_inputs.json")
            },
            "selection": "explicit probe index"
            if probe_index is not None
            else "lowest loss among probes meeting all native-step RMS limits",
            "source_sha256": sources,
        },
    )
    device = wp.get_device("cuda:0")
    directions = wp.empty((count, 12, 4), dtype=wp.float64, device=device)
    wp.launch(local_probe._random_relative_directions, dim=count, inputs=[seed, directions], device=device)
    proposal = wp.empty((1, 12, 4), dtype=wp.float64, device=device)
    wp.launch(
        local_probe._populate_probe_batch,
        dim=1,
        inputs=[
            chosen["probe_index"],
            len(records),
            wp.array(arrays["baseline_coefficients"], dtype=wp.float64, device=device),
            wp.array(
                np.asarray(profile["equilibrium_upper"]) - profile["equilibrium_lower"], dtype=wp.float64, device=device
            ),
            wp.array(arrays["kind_code"], dtype=int, device=device),
            wp.array(arrays["coord_index"], dtype=int, device=device),
            wp.array(arrays["random_direction_index"], dtype=int, device=device),
            wp.array(arrays["sign"], dtype=int, device=device),
            wp.array(arrays["scale_fraction"], dtype=wp.float64, device=device),
            directions,
            proposal,
        ],
        device=device,
    )
    coefficients = proposal.numpy()[0]
    spline = Spline(float(reference["time_s"][-1]), coefficients)
    if not spline.bounds(
        profile["equilibrium_lower"],
        profile["equilibrium_upper"],
        profile["equilibrium_rate_limit"],
        profile["equilibrium_acceleration_limit"],
    ):
        raise ValueError("Regenerated probe violates canonical bounds")
    shoe = saved["shoe"]
    engine = Engine(
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
    if engine.shoe.metadata != shoe:
        raise ValueError("Shoe metadata differs from the qualified probe")
    engine.capture(coefficients[None])
    warmup_steps = int(engine.integrated.numpy().sum())
    coarse = engine.evaluate(coefficients[None])
    trace, run = engine.trace()
    if run["status"] != "completed" or not np.isclose(float(coarse["loss"][0]), chosen["loss"], rtol=0, atol=1e-12):
        raise ValueError("Regenerated probe does not reproduce its saved complete loss")
    fine_engine = Engine(
        reference,
        profile,
        shoe["path"],
        shoe["mount_m"],
        shoe["static_pitch_rad"],
        config=replace(config, dt_s=config.dt_s / 2),
        settings=settings,
        world_count=1,
        friction_model=shoe["friction_model"],
    )
    fine_engine.capture(coefficients[None])
    warmup_steps += int(fine_engine.integrated.numpy().sum())
    fine = fine_engine.evaluate(coefficients[None])
    fine_trace, fine_run = fine_engine.trace()
    refinement = _refinement(trace, run, fine_trace, fine_run, engine.duration, settings)
    names = ("hip_rmse_m", "joint_rmse_rad", "force_rmse_n")
    metrics = {name: coarse["rmse"][0, 2 * i : 2 * i + 2].tolist() for i, name in enumerate(names)}
    refinement["metrics"] = {name: fine["rmse"][0, 2 * i : 2 * i + 2].tolist() for i, name in enumerate(names)}
    refinement["within_measured_tolerances"] = bool(refinement["complete"] and _within(fine["rmse"][0], settings))
    within = _within(coarse["rmse"][0], settings)
    summary = {
        "schema": "cartesian_gpu_probe_replay_1",
        "status": "completed",
        "complete": True,
        "accepted": False,
        "within_measured_tolerances": within,
        "loss": float(coarse["loss"][0]),
        "metrics": metrics,
        "run": run,
        "refinement": refinement,
        "simulation_config": asdict(config),
        "fit_config": asdict(settings),
        "control_count": 12,
        "shoe": shoe,
        "identity": identity,
        "source_sha256": sources,
        "original_source_sha256": saved.get("original_source_sha256", sources),
        "source_audit": validate_sources(saved),
        "execution_identity": execution_identity(),
        "initialization": {
            "kind": "saved_local_probe",
            "used_previous_controller_coefficients": True,
            "used_optimizer_history": False,
        },
        "search_performed": False,
        "global_optimality_claimed": False,
        "local_optimality_claimed": False,
        "selected_probe": chosen,
        "probe_directory": str(probe_directory.resolve()),
        "native_tolerance_probe_count": len(eligible),
        "counts": {
            "regenerated_controller_slots": 1,
            "coarse_rollout_steps": int(engine.integrated.numpy().sum()),
            "fine_rollout_steps": int(fine_engine.integrated.numpy().sum()),
            "capture_warmup_world_steps": warmup_steps,
            "spring_replay_leg_steps": 0,
            "spring_contact_replay_rows": len(trace["time_s"]),
        },
        "qualification": "Numerical measured-fit, half-step, and saved-contact replay checks only; not physical or physiological validation.",
    }
    _write_npz(output / "trace.npz", **trace)
    _write_npz(output / "refined_trace.npz", **fine_trace)
    _write_npz(
        output / "equilibrium.npz",
        duration_s=spline.duration_s,
        coefficients=coefficients,
        identity_json=json.dumps(identity, sort_keys=True),
    )
    spring = load_springs(output, reference, trace, summary, profile)
    spring_passed = bool(spring.get("available") and spring.get("validation", {}).get("passed"))
    summary["spring_replay"] = {"passed": spring_passed, "audit": str((output / "spring_view.json").resolve())}
    summary["accepted"] = bool(
        within and refinement["passed"] and refinement["within_measured_tolerances"] and spring_passed
    )
    if (
        source_snapshot() != sources
        or _sha(Path(__file__)) != caller_hash
        or _sha(Path(local_probe.__file__)) != inputs["probe_source_sha256"]
    ):
        raise ValueError("Replay or physics sources changed during qualification")
    summary["total_wall_s"] = perf_counter() - started
    _write_json(output / "summary.json", summary)
    _write_final_report(output, reference, trace, summary, profile)
    print(
        json.dumps(
            {key: summary[key] for key in ("accepted", "loss", "metrics", "refinement", "spring_replay")}, indent=2
        )
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run the saved-probe replay command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-index", type=int)
    args = parser.parse_args(argv)
    replay_probe(args.probe_directory, args.output, probe_index=args.probe_index)


if __name__ == "__main__":
    main()
