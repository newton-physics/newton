# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Roll out a frozen human controller from initial conditions and a shoe material."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..fit import FitConfig
from ..run import Config
from .benchmark import _plain
from .engine import Engine
from .runner_controller import RunnerController


def load_runner(checkpoint):
    """Restore human parameters without loading the training dataset or future targets."""
    with np.load(checkpoint, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
    controller, metadata = RunnerController.load(checkpoint, metadata["profile"])
    return controller, metadata


def material_variant(source, output, modulus_scale):
    """Write a synthetic modulus sensitivity variant with unchanged shoe geometry."""
    if not np.isfinite(modulus_scale) or modulus_scale <= 0:
        raise ValueError("Modulus scale must be positive and finite")
    original = json.loads(source.read_text())
    variant = deepcopy(original)
    params = variant["constitutive_model"]["parameters"]
    for name in ("instantaneous_shear_modulus_pa", "instantaneous_shear_modulus_2_pa", "pasternak_n_per_m"):
        params[name] *= modulus_scale
    derived = variant["constitutive_model"].get("derived_quantities", {})
    for name, value in derived.items():
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (name.endswith("_pa") or name.startswith("pasternak_n_per_m_"))
        ):
            derived[name] = value * modulus_scale
    variant["validation"] = {
        "status": "Synthetic modulus sensitivity variant; original Instron validation does not apply"
    }
    variant["provenance"]["runner_material_sensitivity"] = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "modulus_scale": modulus_scale,
        "geometry_unchanged": True,
        "relaxation_time_and_friction_unchanged": True,
    }
    output.write_text(json.dumps(variant, indent=2, allow_nan=False) + "\n")
    return output


def simulate(controller, metadata, initial_state, initial_velocity, *, artifact=None, half_step=False, duration_s=None):
    """Advance frozen human control through shoe physics with no measured reference."""
    q0, v0 = np.asarray(initial_state, dtype=float), np.asarray(initial_velocity, dtype=float)
    coefficients, _ = controller.predict_initial(q0[None, :], v0[None, :])
    config = Config(**metadata["simulation_config"])
    if half_step:
        config = replace(config, dt_s=config.dt_s / 2)
    shoe = metadata["shoe"]
    engine = Engine.from_initial(
        q0,
        v0,
        controller.mean_geometry,
        metadata["endpoint_local_m"],
        metadata["profile"],
        artifact or shoe["path"],
        shoe["mount_m"],
        shoe["static_pitch_rad"],
        duration_s=controller.duration if duration_s is None else duration_s,
        controller_duration_s=controller.duration,
        config=config,
        settings=FitConfig(**metadata["fit_config"]),
        friction_model=shoe["friction_model"],
    )
    outcome = engine.rollout(coefficients)
    trace, diagnostics = engine.trace()
    force = trace["grf_n"]
    contact = force[:, 1] > 5 if len(force) else np.array([], dtype=bool)
    indices = np.flatnonzero(contact)
    powers = np.column_stack(
        (
            trace["hip_force_n"] * trace["velocity"][:, :2],
            trace["joint_torque_nm"] * trace["velocity"][:, 3:5],
            trace["ankle_position_force_n"] * trace["ankle_position_velocity_m_s"],
        )
    )
    summary = {
        "complete": bool(outcome["completed"][0]),
        "failure_code": int(outcome["failure_code"][0]),
        "duration_s": engine.duration,
        "rollout_wall_s": outcome["wall_s"],
        "engine_setup_wall_s": engine.setup_wall_s,
        "engine_capture_wall_s": engine.capture_wall_s,
        "controller_period_s": controller.duration,
        "dt_s": engine.dt,
        "coefficients_sha256": hashlib.sha256(coefficients.tobytes()).hexdigest(),
        "peak_grf_n": np.max(force, axis=0) if len(force) else None,
        "grf_impulse_ns": force.sum(axis=0) * engine.dt,
        "contact_duration_s": np.count_nonzero(contact) * engine.dt,
        "touchdown_s": float(trace["time_s"][indices[0]]) if len(indices) else None,
        "toeoff_s": float(trace["time_s"][indices[-1]]) if len(indices) else None,
        "net_actuator_work_j": float(powers.sum() * engine.dt),
        "positive_actuator_work_j": float(np.maximum(powers, 0).sum() * engine.dt),
        "negative_actuator_work_j": float(np.minimum(powers, 0).sum() * engine.dt),
        "final_state": engine.states.numpy()[int(outcome["integrated_steps"][0]), 0],
        "final_velocity": engine.velocities.numpy()[int(outcome["integrated_steps"][0]), 0],
        "peak_compression_fraction": float(np.max(trace["compression_fraction"])) if len(force) else None,
        "diagnostics": diagnostics,
    }
    return summary, trace, coefficients


def main():
    """Save an initial-condition-only rollout of the frozen runner controller."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--initial-state", type=float, nargs=5)
    parser.add_argument("--initial-velocity", type=float, nargs=5)
    parser.add_argument("--initial-conditions", type=Path, help="JSON with initial_state and initial_velocity vectors")
    parser.add_argument("--shoe-artifact", type=Path)
    parser.add_argument("--modulus-scale", type=float, default=1)
    parser.add_argument("--half-step", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.initial_conditions:
        if args.initial_state is not None or args.initial_velocity is not None:
            parser.error("Use either the initial-conditions file or both initial vectors")
        inputs = json.loads(args.initial_conditions.read_text())
        args.initial_state = inputs["initial_state"]
        args.initial_velocity = inputs["initial_velocity"]
    elif args.initial_state is None or args.initial_velocity is None:
        parser.error("Supply --initial-conditions or both --initial-state and --initial-velocity")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    controller, metadata = load_runner(args.controller)
    artifact = args.shoe_artifact or Path(metadata["shoe"]["path"])
    if args.modulus_scale != 1:
        artifact = material_variant(artifact, args.output / "synthetic_material.json", args.modulus_scale)
    summary, trace, coefficients = simulate(
        controller, metadata, args.initial_state, args.initial_velocity, artifact=artifact, half_step=args.half_step
    )
    summary.update(
        controller_sha256=hashlib.sha256(args.controller.read_bytes()).hexdigest(),
        shoe_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        initial_state=args.initial_state,
        initial_velocity=args.initial_velocity,
        modulus_scale=args.modulus_scale,
        scope="Target-free one-cycle rollout with frozen human controller; material scaling is a synthetic sensitivity experiment",
    )
    (args.output / "report.json").write_text(json.dumps(_plain(summary), indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.output / "trace.npz", **trace, coefficients=coefficients)
    print(json.dumps(_plain(summary)), flush=True)


if __name__ == "__main__":
    main()
