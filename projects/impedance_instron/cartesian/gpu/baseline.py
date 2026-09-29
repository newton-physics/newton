# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Qualify the saved twelve-point controller with a new CPU reference rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..fit import FitConfig, _Objective
from ..profile import load as load_profile
from ..profile import validate as validate_profile
from ..report import _plain
from ..run import Config, simulate
from ..shoe import Shoe
from ..trajectory import Spline
from .provenance import source_snapshot


def build_failure_equilibrium(
    reference: dict[str, Any],
    profile: dict[str, Any],
    controls: int,
    config: Config,
) -> tuple[Spline, dict[str, Any]]:
    """Generate an independent bounded failure equilibrium using constant upper bounds.

    The candidate is constructed entirely from the profile's declared upper bounds,
    guaranteeing it stays within position, rate, and acceleration bounds while
    failing the step-zero hip force screen under the initial kinematic conditions.
    """
    duration = float(reference["time_s"][-1])
    lower = np.asarray(profile["equilibrium_lower"], dtype=np.float64)
    upper = np.asarray(profile["equilibrium_upper"], dtype=np.float64)
    rate = np.asarray(profile["equilibrium_rate_limit"], dtype=np.float64)
    acceleration = np.asarray(profile["equilibrium_acceleration_limit"], dtype=np.float64)

    coefficients = np.tile(upper, (controls, 1))
    spline = Spline(duration, coefficients)
    if not spline.bounds(lower, upper, rate, acceleration):
        raise ValueError("Constant upper equilibrium violates declared spline bounds")

    stiffness_hip = np.asarray(profile["hip_stiffness_n_m"], dtype=np.float64)
    damping_hip = np.asarray(profile["hip_damping_ns_m"], dtype=np.float64)
    state0 = np.asarray(reference["state"])[0]
    velocity0 = np.asarray(reference["velocity"])[0]

    hip_force0 = stiffness_hip * (upper[:2] - state0[:2]) - damping_hip * velocity0[:2]
    hip_force0_norm = float(np.linalg.norm(hip_force0))

    if hip_force0_norm <= config.maximum_force_n:
        raise ValueError(
            f"Analytic initial hip force norm ({hip_force0_norm:.3f} N) does not exceed "
            f"the maximum force screen ({config.maximum_force_n:.3f} N)"
        )

    failure_info = {
        "time_s": 0.0,
        "reasons": ["Hip force screen exceeded"],
        "initial_hip_force_norm_n": hip_force0_norm,
        "screen_threshold_n": float(config.maximum_force_n),
        "mechanism": "constant_equilibrium_upper_analytic_initial_hip_force_screen",
    }
    return spline, failure_info


def _fresh_equilibrium(reference: dict, profile: dict) -> tuple[Spline, dict]:
    """Seed twelve controls from measured kinematics and fixed PD gains, without fitting."""
    from ..spline import _derivative_control_polygons, _uniform_knots  # noqa: PLC0415

    time = np.asarray(reference["time_s"], dtype=np.float64)
    duration = float(time[-1])
    channels = [0, 1, 3, 4]
    stiffness = np.asarray([*profile["hip_stiffness_n_m"], *profile["joint_stiffness_nm_rad"]])
    damping = np.asarray([*profile["hip_damping_ns_m"], *profile["joint_damping_nms_rad"]])
    neutral = (
        np.asarray(reference["state"])[:, channels]
        + damping / stiffness * np.asarray(reference["velocity"])[:, channels]
    )
    knots = _uniform_knots(12, 3)
    sample_times = duration * np.asarray([np.mean(knots[i + 1 : i + 4]) for i in range(12)])
    raw = np.column_stack([np.interp(sample_times, time, neutral[:, c]) for c in range(4)])
    lower = np.asarray(profile["equilibrium_lower"])
    upper = np.asarray(profile["equilibrium_upper"])
    rate = np.asarray(profile["equilibrium_rate_limit"])
    acceleration = np.asarray(profile["equilibrium_acceleration_limit"])
    anchor = np.clip(neutral[0], lower, upper)
    delta = raw - anchor
    first, second = _derivative_control_polygons(delta)
    scale = np.ones(4)
    for c in range(4):
        positive = delta[:, c] > 0
        negative = delta[:, c] < 0
        if np.any(positive):
            scale[c] = min(scale[c], float(np.min((upper[c] - anchor[c]) / delta[positive, c])))
        if np.any(negative):
            scale[c] = min(scale[c], float(np.min((lower[c] - anchor[c]) / delta[negative, c])))
        for values, bound in ((first, rate[c] * duration), (second, acceleration[c] * duration**2)):
            peak = np.max(np.abs(values[:, c]))
            if peak > 0:
                scale[c] = min(scale[c], float(bound / peak))
    scale = np.where(scale < 1.0, scale * (1.0 - 1.0e-12), scale)
    for _ in range(32):
        seed = Spline(duration, anchor + delta * scale)
        if seed.bounds(lower, upper, rate, acceleration):
            break
        # Canonical bounds remain strict even at a rounded derivative boundary.
        scale *= 0.5
    else:
        raise ValueError("Could not construct a bounded fresh controller")
    return seed, {
        "kind": "from_scratch_kinematic_pd_seed",
        "used_previous_controller_coefficients": False,
        "used_optimizer_history": False,
        "simulation_or_loss_evaluations_for_initialization": 0,
        "formula": "q_reference + (D/K) * velocity_reference, channels hip-x/hip-z/knee/ankle",
        "sampling": "linear interpolation at twelve cubic-spline Greville abscissae",
        "bounds_handling": "contract each channel toward its initial neutral point; canonical strict bounds",
        "channel_contraction": scale.tolist(),
        "neutral_anchor": anchor.tolist(),
        "sample_times_s": sample_times.tolist(),
        "measured_grf_used_for_initialization": False,
        "scope": "Fresh controller only; measured data, calibrated shoe, gains, and physical model remain fixed.",
    }


def build(source: Path, output: Path, *, from_scratch: bool = False) -> dict[str, Any]:
    """Qualify a saved controller or a newly generated, unfitted controller.

    ``from_scratch`` uses the bundle only for measured data, fixed model and
    configuration. It never uses the bundle's controller coefficients.
    Original bundle bytes remain unchanged in either mode.
    """
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Baseline output directory already exists: {output}")
    manifest = json.loads((source / "baseline.json").read_text())
    friction_model = manifest.get("friction_model", {}).get("model")
    if friction_model not in ("legacy", "maxwell", "column_maxwell", "elastic_coulomb"):
        raise ValueError("Saved baseline must declare a supported friction model")
    for name, expected in manifest["files_sha256"].items():
        if Path(name).name != name or hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Saved baseline file changed: {name}")
    original = json.loads((source / "summary.json").read_text())
    reference_path, profile_path = source / "reference.npz", source / "profile.json"
    artifact_path = source / "digital_shoe.json"
    controls = 12
    fit_config = FitConfig(**original["fit_config"])
    if fit_config.control_count != controls:
        raise ValueError("Saved configuration must use twelve controls")
    config = Config(**original["simulation_config"])
    equilibrium = None
    initialization = {"kind": "saved_controller", "used_previous_controller_coefficients": True}
    if not from_scratch:
        with np.load(source / "equilibrium.npz", allow_pickle=False) as archive:
            equilibrium = Spline(float(archive["duration_s"]), archive["coefficients"])
            frozen_identity = json.loads(str(archive["identity_json"]))
        if equilibrium.coefficients.shape[0] != 12 or equilibrium.coefficients.shape[1] not in (4, 6):
            raise ValueError("The baseline must have twelve controls and four or six channels")
        expected_identity = {
            "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "profile_sha256": hashlib.sha256(
                json.dumps(json.loads(profile_path.read_text()), sort_keys=True).encode()
            ).hexdigest(),
            "simulation_config": original["simulation_config"],
            "shoe": {
                "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                "mount_m": original["shoe"]["mount_m"],
                "static_pitch_rad": original["shoe"]["static_pitch_rad"],
                "friction": original["shoe"]["friction"],
                "device": original.get("run", original).get("shoe_device", "cuda:0"),
            },
        }
        if frozen_identity != expected_identity:
            raise ValueError("Saved controller identity differs from its inputs")
    return build_inputs(
        reference_path,
        profile_path,
        artifact_path,
        output,
        mount_m=original["shoe"]["mount_m"],
        pitch_rad=original["shoe"]["static_pitch_rad"],
        config=config,
        fit_config=fit_config,
        equilibrium=equilibrium,
        initialization=None if from_scratch else initialization,
        input_provenance={"selected_baseline": manifest},
        friction_model=friction_model,
    )


def build_inputs(
    reference_path: Path,
    profile_path: Path,
    artifact_path: Path,
    output: Path,
    *,
    mount_m,
    pitch_rad: float,
    config: Config,
    fit_config: FitConfig,
    equilibrium: Spline | None = None,
    initialization: dict | None = None,
    input_provenance: dict | None = None,
    friction_model: str = "elastic_coulomb",
) -> dict[str, Any]:
    """Create new numerical evidence from independently frozen measured inputs.

    The caller supplies the subject-specific reference/profile and fixed shoe
    placement. An omitted controller is generated without fitting. Existing
    bundles and their identities are never rewritten.
    """
    reference_path, profile_path, artifact_path, output = map(
        Path, (reference_path, profile_path, artifact_path, output)
    )
    if output.exists():
        raise FileExistsError(output)
    controls = 12
    if fit_config.control_count != controls:
        raise ValueError("FitConfig must use twelve controls")
    from_scratch = equilibrium is None
    sources_before = source_snapshot()
    builder_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    reference_bytes = reference_path.read_bytes()
    profile_bytes = profile_path.read_bytes()

    with np.load(reference_path, allow_pickle=False) as archive:
        reference = dict(archive)
    profile = load_profile(profile_path)
    validate_profile(profile)
    if from_scratch:
        equilibrium, initialization = _fresh_equilibrium(reference, profile)
    if equilibrium.coefficients.shape != (12, len(profile["equilibrium_lower"])) or not np.isclose(
        equilibrium.duration_s, float(reference["time_s"][-1]), rtol=0, atol=1e-12
    ):
        raise ValueError("Initial controller must match the profile channels and twelve-point reference duration")
    if not equilibrium.bounds(
        *(
            profile[k]
            for k in (
                "equilibrium_lower",
                "equilibrium_upper",
                "equilibrium_rate_limit",
                "equilibrium_acceleration_limit",
            )
        )
    ):
        raise ValueError("Initial controller violates canonical bounds")
    initialization = initialization or {"kind": "explicit_controller", "used_previous_controller_coefficients": True}
    if friction_model not in ("legacy", "maxwell", "column_maxwell", "elastic_coulomb"):
        raise ValueError("Baseline inputs must select a supported friction model")
    shoe = Shoe(artifact_path, mount_m, pitch_rad, device="cpu", friction_model=friction_model)
    shoe_identity = {
        "artifact_sha256": shoe.metadata["sha256"],
        "mount_m": shoe.metadata["mount_m"],
        "static_pitch_rad": shoe.metadata["static_pitch_rad"],
        "device": str(shoe.device),
        "friction": shoe.metadata["friction"],
    }

    trace, run_summary = simulate(reference, profile, equilibrium, shoe, config=config)

    sources_after = source_snapshot()
    if sources_after != sources_before:
        raise RuntimeError("GPU runtime sources changed during baseline simulation")

    objective = _Objective(reference, fit_config)
    residual, metrics, costs = objective.evaluate(trace, run_summary)

    if run_summary.get("status") != "completed" or residual is None:
        failure_reason = run_summary.get("failure", "Objective evaluation failed or run incomplete")
        raise RuntimeError(
            f"Saved baseline simulation did not complete successfully for {controls} controls: {failure_reason}"
        )

    loss = float(residual @ residual)

    failure_spline, failure_info = _failure_fixture(
        reference, profile, artifact_path, mount_m, pitch_rad, fit_config, config, friction_model
    )
    output.mkdir(parents=True)
    (output / "reference.npz").write_bytes(reference_bytes)
    (output / "profile.json").write_bytes(profile_bytes)

    profile_for_identity = json.loads(profile_bytes.decode("utf-8"))
    identity = {
        "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "profile_sha256": hashlib.sha256(json.dumps(profile_for_identity, sort_keys=True).encode()).hexdigest(),
        "simulation_config": asdict(config),
        "shoe": shoe_identity,
    }

    np.savez_compressed(
        output / "equilibrium.npz",
        duration_s=equilibrium.duration_s,
        coefficients=equilibrium.coefficients,
        identity_json=json.dumps(identity, sort_keys=True),
    )
    np.savez_compressed(output / "trace.npz", **trace)

    np.savez_compressed(
        output / "failure_equilibrium.npz",
        duration_s=failure_spline.duration_s,
        coefficients=failure_spline.coefficients,
        failure_reason=failure_info["reasons"][0],
        failure_time_s=failure_info["time_s"],
        initial_hip_force_norm_n=failure_info["initial_hip_force_norm_n"],
    )

    summary: dict[str, Any] = {
        "schema": "cartesian_twelve_point_baseline_1",
        "fit_config": asdict(fit_config),
        "simulation_config": asdict(config),
        "source_sha256": sources_before,
        "builder_sha256": builder_hash,
        "shoe": shoe.metadata,
        "status": "completed",
        "complete": True,
        "accepted": False,
        "refinement": {
            "performed": False,
            "passed": False,
            "reason": "saved controller CPU qualification without optimization or post-fit refinement",
        },
        "run": run_summary,
        "loss": loss,
        "metrics": metrics,
        "components": costs,
        "objective_components": costs,
        "objective": objective.description,
        "provenance": {
            "generation": (
                "saved_twelve_point_controller"
                if initialization.get("used_previous_controller_coefficients", True)
                else "fresh_twelve_point_controller"
            ),
            "initialization": initialization,
            "control_count": controls,
            "optimization": "none",
            **(input_provenance or {}),
            "builder_sha256": builder_hash,
            "raw_reference": str(reference_path.resolve()),
            "raw_profile": str(profile_path.resolve()),
            "raw_artifact": str(artifact_path.resolve()),
        },
        "model_scope": (
            "3 leg bodies, 1 shoe, 4 equilibrium channels; external Cartesian hip impedance, "
            "no hip torque/upper-body mass"
        ),
        "identity": identity,
    }

    (output / "summary.json").write_text(json.dumps(_plain(summary), indent=2, allow_nan=False) + "\n")
    return summary


def _failure_fixture(reference, profile, artifact_path, mount, pitch, settings, config, friction_model):
    """Find and confirm a bounded failure without changing the numerical screens."""
    q0, v0 = reference["state"][0], reference["velocity"][0]
    upper = np.asarray(profile["equilibrium_upper"])
    force = (
        np.asarray(profile["hip_stiffness_n_m"]) * (upper[:2] - q0[:2])
        - np.asarray(profile["hip_damping_ns_m"]) * v0[:2]
    )
    if np.linalg.norm(force) > config.maximum_force_n:
        return build_failure_equilibrium(reference, profile, 12, config)
    # Smaller gains may have no step-zero force failure anywhere inside the box.
    # Search fixed box corners, then confirm an actual failure on the CPU.
    from itertools import product  # noqa: PLC0415

    from .engine import Engine  # noqa: PLC0415

    lower = np.asarray(profile["equilibrium_lower"])
    corners = np.asarray([np.where(bits, upper, lower) for bits in product((False, True), repeat=4)])
    candidates = np.repeat(corners[:, None, :], 12, axis=1)
    engine = Engine(
        reference,
        profile,
        artifact_path,
        mount,
        pitch,
        config=config,
        settings=settings,
        world_count=len(candidates),
        friction_model=friction_model,
    )
    scores = engine.evaluate(candidates)
    for index in np.flatnonzero(scores["failure_code"]):
        spline = Spline(float(reference["time_s"][-1]), candidates[index])
        shoe = Shoe(artifact_path, mount, pitch, device="cpu", friction_model=friction_model)
        _, run = simulate(reference, profile, spline, shoe, config=config)
        if run["failure"] is not None:
            initial_force = (
                np.asarray(profile["hip_stiffness_n_m"]) * (corners[index, :2] - q0[:2])
                - np.asarray(profile["hip_damping_ns_m"]) * v0[:2]
            )
            return spline, {
                "time_s": float(run["integrated_duration_s"]),
                "reasons": [str(run["failure"])],
                "initial_hip_force_norm_n": float(np.linalg.norm(initial_force)),
                "mechanism": "bounded_constant_corner_with_cpu_confirmed_failure",
            }
    raise ValueError("No bounded CPU-confirmed failure fixture found; qualification cannot be skipped")


def main(argv: list[str] | None = None) -> None:
    """Create new numerical evidence from a saved twelve-point bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--from-scratch", action="store_true", help="Generate unfitted coefficients from measured kinematics."
    )
    args = parser.parse_args(argv)
    summary = build(args.source, args.output, from_scratch=args.from_scratch)
    print(f"Qualified twelve-point CPU baseline at {args.output}; loss={summary['loss']:.7f}")


if __name__ == "__main__":
    main()
