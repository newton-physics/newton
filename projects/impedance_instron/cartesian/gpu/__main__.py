# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fit a frozen Cartesian experiment with qualified resident GPU batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from ..fit import FitConfig, _refinement
from ..run import Config
from ..trajectory import Spline
from .benchmark import _plain, execution_identity, load_frozen
from .engine import Engine
from .provenance import source_snapshot, validate_sources


def _write_json(path, value):
    """Atomically replace one completed JSON checkpoint."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_plain(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_npz(path, **values):
    """Atomically replace one completed array checkpoint."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def _write_final_report(directory, reference, trace, summary, profile, *, mesh_only: bool = False):
    """Write the default spring-enabled report after saving the numerical result."""
    from ..report import write_report  # noqa: PLC0415

    started = perf_counter()
    path = write_report(directory, reference, trace, summary, profile=profile, include_springs=not mesh_only)
    print(json.dumps({"report": str(path), "report_wall_s": perf_counter() - started}), flush=True)
    return path


def _validation(
    path: Path,
    directory: Path,
    *,
    mixed: bool,
    expected_controls: int = 12,
    contact_evidence: Path | None = None,
):
    """Require current source/input identities and successful numerical evidence."""
    if expected_controls != 12:
        raise ValueError(f"GPU validation requires exactly 12 controls, got {expected_controls!r}")

    report = json.loads(path.read_text())
    passed = report.get("passed") if mixed else report.get("same_timestep_parity_passed")
    if not passed and (mixed or contact_evidence is None):
        raise ValueError(
            f"GPU validation has not passed: {path}; common-pose evidence is required for a rounding warning"
        )
    if report.get("execution_identity") != execution_identity():
        raise ValueError(f"GPU runtime/compiler policy changed since validation: {path}")

    if report.get("control_count") != expected_controls:
        raise ValueError(f"Evidence report {path} must use exactly {expected_controls} controls")

    if not passed:
        # Keep the failed propagated-moment check visible; do not relabel it as a pass.
        limits, errors = report["same_timestep_limits"], report["maximum_absolute_error"]
        other_limits_pass = all(
            errors.get(key) is not None and errors[key] <= bound
            for key, bound in limits.items()
            if key != "ankle_contact_moment_nm"
        )
        if not (
            other_limits_pass
            and report["equal_time_grid"]
            and report["equal_passive_cap_and_joint_diagnostics"]
            and report["run"]["failure"] is None
        ):
            raise ValueError("Full-rollout differences extend beyond the disclosed moment-rounding warning")
        contact = json.loads(contact_evidence.read_text())
        errors_at_pose = contact["exact_saved_cpu_pose_contact_error"]
        for key in ("force_vector_n", "ankle_contact_moment_nm", "compression_fraction"):
            if errors_at_pose[key]["max_absolute_error"] > limits[key]:
                raise ValueError(f"Common-pose contact failed the unchanged {key} limit")
        for values in contact["engine_prepare_at_saved_cpu_states_vs_exact_cpu_float32"].values():
            if values["max_absolute_error"] != 0:
                raise ValueError("GPU carrier staging differs at the same saved CPU states")
        if (
            contact["leg_integration_executed"]
            or contact["saved_rows"] != report["run"]["integrated_steps"]
            or errors_at_pose["passive_cap_column_count"]["max_absolute_error"] != 0
        ):
            raise ValueError("Common-pose contact evidence has different coverage or cap behavior")
        for name in ("reference.npz", "profile.json", "trace.npz", "summary.json"):
            baseline_path = (directory / name).resolve()
            if (
                contact["inputs_sha256"].get(str(baseline_path))
                != hashlib.sha256(baseline_path.read_bytes()).hexdigest()
            ):
                raise ValueError("Common-pose contact evidence used different frozen inputs")
        for source, digest in contact["sources_sha256"].items():
            if source in source_snapshot() and source_snapshot()[source] != digest:
                raise ValueError(f"Common-pose contact source changed: {source}")
        if contact["engine_source_sha256"] != source_snapshot()["projects/impedance_instron/cartesian/gpu/engine.py"]:
            raise ValueError("GPU engine changed since common-pose contact validation")
        policy = execution_identity()
        if (
            contact["warp_version"] != policy["warp"]
            or contact["gpu_name"] != policy["gpu_name"]
            or contact["runtime_fuse_fp"] != policy["modules"]["projects.digital_shoe.runtime"]["fuse_fp"]
        ):
            raise ValueError("Common-pose contact runtime precision policy differs")
        report["rounding_warning_qualification"] = {
            "strict_full_rollout_parity_passed": False,
            "other_full_rollout_limits_passed": True,
            "common_pose_contact_limits_passed": True,
            "moment_limit_nm": limits["ankle_contact_moment_nm"],
            "full_rollout_moment_error_nm": errors["ankle_contact_moment_nm"],
            "common_pose_moment_error_nm": errors_at_pose["ankle_contact_moment_nm"]["max_absolute_error"],
            "evidence": str(contact_evidence),
            "evidence_sha256": hashlib.sha256(contact_evidence.read_bytes()).hexdigest(),
            "decision": "Allow a bounded numerical GPU search with the propagated float32-contact rounding warning retained. "
            "This does not relax compression or timestep-refinement screens or validate the physical shoe interface.",
        }
    if report.get("sources") != source_snapshot():
        raise ValueError(f"GPU runtime changed since validation: {path}")
    for name in ("reference.npz", "profile.json", "equilibrium.npz", "trace.npz", "summary.json"):
        expected = report.get("frozen_artifacts_sha256", {}).get(name)
        if expected != hashlib.sha256((directory / name).read_bytes()).hexdigest():
            raise ValueError(f"GPU validation used different frozen {name}: {path}")
    if mixed and (not report.get("permutation_full_history_exact") or not report.get("reset_full_history_exact")):
        raise ValueError("GPU mixed-world reset/isolation evidence is incomplete")
    return report


def optimize(args):
    """Run a bounded GPU search and separately qualify its frozen best controller."""
    if args.output.exists():
        raise FileExistsError(args.output)
    reference, profile, coefficients, baseline = load_frozen(args.directory)
    config, settings = Config(**baseline["simulation_config"]), FitConfig(**baseline["fit_config"])
    starting_controller = args.initial_equilibrium or args.directory / "equilibrium.npz"
    requested_controls = getattr(args, "controls", None)
    if requested_controls not in (None, 12):
        raise ValueError("Control-count conversion is not supported; use the saved 12-point controller")

    if args.initial_equilibrium is not None:
        with np.load(starting_controller, allow_pickle=False) as archive:
            candidate = archive["coefficients"].copy()
            candidate_duration = float(archive["duration_s"])
            frozen = json.loads(str(archive["identity_json"]))
        expected_reference = hashlib.sha256((args.directory / "reference.npz").read_bytes()).hexdigest()
        expected_profile = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
        if (
            frozen.get("reference_sha256") != expected_reference
            or frozen.get("profile_sha256") != expected_profile
            or frozen.get("simulation_config") != asdict(config)
        ):
            raise ValueError("Warm-start controller uses different frozen inputs or numerical screens")
        for identity_key, metadata_key in (
            ("artifact_sha256", "sha256"),
            ("mount_m", "mount_m"),
            ("static_pitch_rad", "static_pitch_rad"),
            ("friction", "friction"),
        ):
            if frozen.get("shoe", {}).get(identity_key) != baseline["shoe"][metadata_key]:
                raise ValueError("Warm-start controller uses a different shoe or mounting")
        if not np.isclose(candidate_duration, float(reference["time_s"][-1]), rtol=0, atol=1e-12):
            raise ValueError("Warm-start controller must retain the validated spline duration")
        coefficients = candidate

    if coefficients.shape != (12, 4):
        raise ValueError("The starting controller must have shape (12, 4)")
    active_controls = 12
    single_evidence = _validation(
        args.single_validation,
        args.directory,
        mixed=False,
        expected_controls=active_controls,
        contact_evidence=args.contact_rounding_evidence,
    )
    mixed_evidence = _validation(
        args.batch_validation,
        args.directory,
        mixed=True,
        expected_controls=active_controls,
    )

    plateau_patience = getattr(args, "plateau_patience", 20)
    plateau_rtol = getattr(args, "plateau_rtol", 1.0e-4)
    world_count = 128
    search_mode = "shared_controller"
    settings = replace(
        settings,
        control_count=active_controls,
        max_evaluations=2 * args.iterations * world_count,
        initial_step_fraction=args.step_fraction,
        minimum_step_fraction=args.minimum_step_fraction,
    )
    if settings.control_count != 12:
        raise ValueError("FitConfig must use exactly 12 controls")
    if mixed_evidence["batch_size"] != world_count:
        raise ValueError("The mixed-world validation must use exactly 128 worlds")

    args.output.mkdir(parents=True)
    # Preserve byte identities rather than reserializing the measured input snapshots.
    for name in ("reference.npz", "profile.json"):
        (args.output / name).write_bytes((args.directory / name).read_bytes())
    shoe = baseline["shoe"]
    identity = {
        "reference_sha256": hashlib.sha256((args.output / "reference.npz").read_bytes()).hexdigest(),
        "profile_sha256": hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest(),
        "simulation_config": asdict(config),
        "shoe": {
            "artifact_sha256": shoe["sha256"],
            "mount_m": shoe["mount_m"],
            "static_pitch_rad": shoe["static_pitch_rad"],
            "friction": shoe["friction"],
            "device": "cuda:0",
        },
    }
    sources = source_snapshot()
    metadata = {
        "simulation_config": asdict(config),
        "fit_config": asdict(settings),
        "control_count": active_controls,
        "shoe": shoe,
        "source_sha256": sources,
        "original_source_sha256": baseline.get("original_source_sha256", baseline["source_sha256"]),
        "source_audit": validate_sources(baseline),
        "execution_identity": execution_identity(),
        "rounding_warning_qualification": single_evidence.get("rounding_warning_qualification"),
        "optimizer_source_sha256": {
            str(Path(__file__).parent / name): hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in (
                "__main__.py",
                "resident.py",
                "benchmark.py",
                "batch_benchmark.py",
            )
        },
        "command": sys.argv,
        "baseline": str(args.directory.resolve()),
        "starting_controller": str(starting_controller.resolve()),
        "starting_controller_sha256": hashlib.sha256(starting_controller.read_bytes()).hexdigest(),
        "starting_coefficients": coefficients.tolist(),
        "initialization": (
            baseline.get("provenance", {}).get("initialization", {"kind": "saved_controller"})
            if args.initial_equilibrium is None
            else {"kind": "saved_controller", "used_previous_controller_coefficients": True}
        ),
        "cpu_fit_started": False,
        "gpu_validation": {
            "single": str(args.single_validation),
            "mixed": str(args.batch_validation),
            "single_sha256": hashlib.sha256(args.single_validation.read_bytes()).hexdigest(),
            "mixed_sha256": hashlib.sha256(args.batch_validation.read_bytes()).hexdigest(),
        },
        "validation_timings_s": {
            "single": single_evidence["warm_batch_wall_s"],
            "mixed": mixed_evidence["warm_batch_wall_s"],
        },
    }
    _write_json(
        args.output / "optimization_inputs.json",
        {
            **metadata,
            "identity": identity,
            "optimizer_budget": {
                "iterations": args.iterations,
                "search_wall_s": args.wall_seconds,
                "optimizer": "resident",
                "search_mode": search_mode,
                "batch_size_policy": "fixed",
                "plateau_patience": plateau_patience,
                "plateau_rtol": plateau_rtol,
                "independent_controllers": 1,
                "worlds": world_count,
                "refinement": "separate frozen GPU half-step after search",
            },
        },
    )
    started = perf_counter()
    engine = Engine(
        reference,
        profile,
        shoe["path"],
        shoe["mount_m"],
        shoe["static_pitch_rad"],
        config=config,
        settings=settings,
        world_count=world_count,
        friction_model=shoe["friction_model"],
    )
    if engine.shoe.metadata != shoe:
        raise ValueError("GPU shoe metadata changed from the frozen model")
    engine.capture(np.repeat(coefficients[None], world_count, axis=0))

    from .resident import fit_resident  # noqa: PLC0415

    print(
        json.dumps(
            {
                "optimizer": "resident",
                "search_mode": search_mode,
                "independent_controllers": 1,
                "worlds": world_count,
                "search_wall_budget_s": args.wall_seconds,
                "progress": "GPU-resident search; result arrays are copied only after search ends",
            }
        ),
        flush=True,
    )
    spline, trace, run, summary = fit_resident(
        engine,
        Spline(engine.duration, coefficients),
        max_iterations=args.iterations,
        max_wall_s=args.wall_seconds,
        initial_step_fraction=args.step_fraction,
        minimum_step_fraction=args.minimum_step_fraction,
        seed=getattr(args, "seed", 17),
        plateau_patience=plateau_patience,
        plateau_rtol=plateau_rtol,
    )
    _write_json(args.output / "progress.json", summary)
    summary.update(metadata)
    summary.update(
        run=run,
        accepted=False,
        within_measured_tolerances=False,
        initialization_and_capture_wall_s=engine.setup_wall_s + engine.capture_wall_s,
        refinement={"performed": False, "passed": False, "reason": "coarse_run_incomplete"},
    )
    if summary["complete"]:
        metrics = summary["metrics"]
        criteria = (
            ("hip_rmse_m", settings.hip_tolerance_m),
            ("joint_rmse_rad", settings.joint_tolerance_rad),
            ("force_rmse_n", settings.force_tolerance_n),
        )
        summary["within_measured_tolerances"] = all(max(metrics[name]) <= bound for name, bound in criteria)
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
        fine_score = fine_engine.evaluate(spline.coefficients[None])
        fine_trace, fine_run = fine_engine.trace()
        qualification = _refinement(trace, run, fine_trace, fine_run, engine.duration, settings)
        qualification["metrics"] = {
            name: fine_score["rmse"][0, 2 * i : 2 * i + 2].tolist() for i, (name, _) in enumerate(criteria)
        }
        qualification["within_measured_tolerances"] = bool(
            qualification["complete"] and all(max(qualification["metrics"][name]) <= bound for name, bound in criteria)
        )
        qualification["gpu_wall_s"] = fine_engine.last_wall_s
        summary["refinement"] = qualification
        summary["accepted"] = bool(
            summary["within_measured_tolerances"]
            and qualification["passed"]
            and qualification["within_measured_tolerances"]
        )
        _write_npz(args.output / "refined_trace.npz", **fine_trace)
    if source_snapshot() != sources:
        raise ValueError("GPU runtime sources changed during optimization/qualification")
    if any(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest
        for path, digest in metadata["optimizer_source_sha256"].items()
    ):
        raise ValueError("GPU optimizer/caller sources changed during optimization/qualification")
    summary["total_wall_s"] = perf_counter() - started
    summary["qualification"] = (
        "Measured-fit/refinement acceptance is numerical only, not validation of the inherited last/foam interface."
    )
    _write_json(args.output / "summary.json", summary)
    _write_npz(args.output / "trace.npz", **trace)
    _write_npz(
        args.output / "equilibrium.npz",
        duration_s=spline.duration_s,
        coefficients=spline.coefficients,
        identity_json=json.dumps(identity, sort_keys=True),
    )
    print(
        json.dumps(
            _plain(
                {
                    key: summary[key]
                    for key in (
                        "loss",
                        "initial_loss",
                        "metrics",
                        "accepted",
                        "counts",
                        "wall_s",
                        "total_wall_s",
                        "refinement",
                    )
                }
            ),
            indent=2,
            allow_nan=False,
        )
    )
    _write_final_report(args.output, reference, trace, summary, profile, mesh_only=getattr(args, "mesh_only", False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for Cartesian GPU optimization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mesh-only",
        action="store_true",
        help="Write a mesh-only report instead of the default verified spring and deformation views.",
    )
    parser.add_argument(
        "--initial-equilibrium",
        type=Path,
        help="Warm-start from a saved controller with the same frozen model and input identities.",
    )
    parser.add_argument("--single-validation", type=Path, required=True)
    parser.add_argument("--batch-validation", type=Path, required=True)
    parser.add_argument(
        "--contact-rounding-evidence",
        type=Path,
        help="Keep a failed propagated-moment check as an explicit warning only with same-pose contact evidence.",
    )
    parser.add_argument("--seed", type=int, default=17, help="Deterministic local-proposal seed.")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--plateau-patience", type=int, default=20)
    parser.add_argument("--plateau-rtol", type=float, default=1.0e-4)
    parser.add_argument("--wall-seconds", type=float, default=3600.0)
    parser.add_argument("--step-fraction", type=float, default=0.05)
    parser.add_argument("--minimum-step-fraction", type=float, default=0.005)
    return parser


def main(argv: list[str] | None = None):
    """Expose a GPU-only fitted-stance experiment from a frozen starting controller."""
    parser = build_parser()
    optimize(parser.parse_args(argv))


if __name__ == "__main__":
    main()
