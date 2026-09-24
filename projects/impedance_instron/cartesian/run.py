# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run one CPU single-leg rollout with a Cartesian hip actuator and one shoe."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .mechanics import Body


@dataclass(frozen=True)
class Config:
    """Declare numerical screens, not physiological acceptance limits.

    ``maximum_speed`` bounds hip speed [m/s] and each angular coordinate speed
    [rad/s]. ``maximum_force_n`` bounds both the hip force and ground resultant.
    Joint range limits are diagnostic by default for this exploratory model.
    """

    dt_s: float = 0.000125
    gravity_m_s2: float = 9.81
    compression_limit: float = 0.9
    maximum_force_n: float = 6000.0
    minimum_hip_height_m: float = 0.2
    maximum_speed: float = 100.0
    joint_limits_diagnostic: bool = True

    def __post_init__(self):
        """Reject invalid integration settings before advancing contact."""
        for name, value in vars(self).items():
            if name == "joint_limits_diagnostic":
                if not isinstance(value, bool):
                    raise ValueError("joint_limits_diagnostic must be a boolean")
            elif not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.compression_limit >= 1:
            raise ValueError("compression_limit must be below one")


def simulate(reference: dict, profile: dict, spline, shoe, *, config: Config | None = None):
    """Integrate one free leg from measured initial position and velocity.

    The only actuator loads are ``Khip*(p_eq-p)-Dhip*v`` at the hip point
    and ``Kjoint*(theta_eq-theta)-Djoint*theta_dot`` at knee and ankle.
    Equilibrium velocity is not a damping target. Later measured motion and
    measured forces never drive the equations of motion.

    Args:
        reference: Single-leg reference arrays, with positions in metres and
            angles in radians. Only the first state and velocity initialize motion.
        profile: Explicit three-segment inertias, Cartesian hip impedance,
            knee/ankle impedance, and mixed-unit equilibrium limits.
        spline: Four-channel equilibrium spline, ordered hip x/z [m] then
            knee/ankle [rad], covering the reference duration.
        shoe: One shared-artifact Shoe instance. Its material and friction
            histories reset once per rollout. Its kinematic carrier adds no
            integrated body mass; the declared foot mass belongs to the leg.
        config: Integration step and failure screens.

    Returns:
        Full-resolution preintegration trace and summary. Contact advances
        exactly once per attempted integration step, never at the terminal
        sample. The terminal state and velocity are saved in the summary.
        Separate passive-cap diagnostics do not imply physical acceptance.
    """
    from .data import validate as validate_reference  # noqa: PLC0415
    from .profile import validate as validate_profile  # noqa: PLC0415

    validate_reference(reference)
    validate_profile(profile)
    if "shoe_static_pitch_rad" in reference and not np.isclose(
        float(reference["shoe_static_pitch_rad"]), shoe.static_pitch_rad, rtol=0, atol=1e-12
    ):
        raise ValueError("Ground-angle reference and shoe use different fixed pitch frames")
    cfg = config or Config()
    duration = float(reference["time_s"][-1])
    if duration <= 0:
        raise ValueError("The reference duration must be positive")
    if not np.isclose(spline.duration_s, duration, rtol=0, atol=1e-12):
        raise ValueError("Equilibrium must cover exactly the reference window")
    if not spline.bounds(
        profile["equilibrium_lower"],
        profile["equilibrium_upper"],
        profile["equilibrium_rate_limit"],
        profile["equilibrium_acceleration_limit"],
    ):
        raise ValueError("Equilibrium exceeds position, rate, or acceleration limits")
    body = Body(
        reference["lengths_m"],
        reference["endpoint_local_m"],
        profile["masses_kg"],
        profile["com_local_m"],
        profile["inertias_kg_m2"],
    )
    stiffness_hip = np.asarray(profile["hip_stiffness_n_m"], dtype=float)
    damping_hip = np.asarray(profile["hip_damping_ns_m"], dtype=float)
    stiffness_joint = np.asarray(profile["joint_stiffness_nm_rad"], dtype=float)
    damping_joint = np.asarray(profile["joint_damping_nms_rad"], dtype=float)
    lower_joint = np.asarray(profile["joint_lower_rad"], dtype=float)
    upper_joint = np.asarray(profile["joint_upper_rad"], dtype=float)
    rest_length = np.asarray(shoe.shoe.column_bed.rest_length_m, dtype=float)
    driven = shoe.foundation.driven.numpy().astype(bool)
    passive = ~driven
    passive_cap = float(shoe.foundation.surround.max_strain) if np.any(passive) else None
    shoe.foundation.reset()
    state = np.array(reference["state"][0], dtype=float, copy=True)
    velocity = np.array(reference["velocity"][0], dtype=float, copy=True)
    steps = math.ceil(duration / cfg.dt_s)
    dt = duration / steps
    times = np.linspace(0.0, duration, steps + 1)
    equilibrium, _, _ = spline.sample(times)
    equilibrium = np.asarray(equilibrium, dtype=float)
    if equilibrium.shape != (steps + 1, 4) or not np.isfinite(equilibrium).all():
        raise ValueError("Equilibrium samples must be finite with shape (step_count + 1, 4)")
    shapes = {
        "state": (5,),
        "velocity": (5,),
        "joints_m": (4, 2),
        "equilibrium": (4,),
        "hip_force_n": (2,),
        "joint_torque_nm": (2,),
        "grf_n": (2,),
        "ankle_contact_moment_nm": (),
        "compression_fraction": (),
        "driven_compression_fraction": (),
        "passive_compression_fraction": (),
    }
    trace = {key: np.empty((steps, *shape)) for key, shape in shapes.items()}
    trace["passive_cap_column_count"] = np.empty(steps, dtype=np.int32)
    angular_jacobian = body.angular_jacobian(2)
    ankle_local = np.zeros(2)
    integrated = 0
    recorded = 0
    failure = None
    joint_exceedances = {}
    for index, time in enumerate(times):
        try:
            if not np.isfinite(state).all() or not np.isfinite(velocity).all():
                raise FloatingPointError("Nonfinite leg state or velocity")
            hip_force = stiffness_hip * (equilibrium[index, :2] - state[:2]) - damping_hip * velocity[:2]
            torque = stiffness_joint * (equilibrium[index, 2:] - state[3:]) - damping_joint * velocity[3:]
            if not np.isfinite(hip_force).all() or not np.isfinite(torque).all():
                raise FloatingPointError("Nonfinite actuator load")
            reasons = []
            outside = (state[3:] < lower_joint) | (state[3:] > upper_joint)
            if np.any(outside):
                joint_exceedances.setdefault(
                    "joint range", {"first_time_s": float(time), "joints": np.flatnonzero(outside).tolist()}
                )
                if not cfg.joint_limits_diagnostic:
                    reasons.append(f"Joint range exceeded at knee/ankle indices {np.flatnonzero(outside).tolist()}")
            if state[1] < cfg.minimum_hip_height_m:
                reasons.append("Hip height screen exceeded; leg-ground collision is not modeled")
            if np.linalg.norm(velocity[:2]) > cfg.maximum_speed or np.max(np.abs(velocity[2:])) > cfg.maximum_speed:
                reasons.append("Numerical speed screen exceeded")
            if np.linalg.norm(hip_force) > cfg.maximum_force_n:
                reasons.append("Hip force screen exceeded")
            if reasons:
                failure = {"time_s": float(time), "reasons": reasons}
                break
            if index == steps:
                break
            ankle, jacobian, _ = body.point(state, 2, ankle_local)
            wrench, _ = shoe.apply(
                ankle,
                jacobian @ velocity,
                body.angle(state, 2),
                float(angular_jacobian @ velocity),
                dt,
            )
            wrench = np.asarray(wrench, dtype=float)
            if wrench.shape != (3,) or not np.isfinite(wrench).all():
                raise FloatingPointError("Nonfinite or invalid shoe wrench")
            # Shoe.apply already converts Newton +Y torque to mathematical X/Z moment.
            generalized = np.array([hip_force[0], hip_force[1], 0.0, torque[0], torque[1]])
            generalized += jacobian.T @ wrench[:2] + angular_jacobian * wrench[2]
            compression = shoe.foundation.compression.numpy() / rest_length
            if not np.isfinite(compression).all():
                raise FloatingPointError("Nonfinite shoe compression")
            driven_max = float(np.max(compression[driven]))
            passive_max = float(np.max(compression[passive])) if np.any(passive) else 0.0
            cap_count = int(np.count_nonzero(compression[passive] >= passive_cap - 1e-6)) if passive_cap else 0
            trace["state"][index] = state
            trace["velocity"][index] = velocity
            trace["joints_m"][index] = body.kinematics(state)
            trace["equilibrium"][index] = equilibrium[index]
            trace["hip_force_n"][index] = hip_force
            trace["joint_torque_nm"][index] = torque
            trace["grf_n"][index] = wrench[:2]
            trace["ankle_contact_moment_nm"][index] = wrench[2]
            trace["compression_fraction"][index] = max(driven_max, passive_max)
            trace["driven_compression_fraction"][index] = driven_max
            trace["passive_compression_fraction"][index] = passive_max
            trace["passive_cap_column_count"][index] = cap_count
            recorded += 1
            if driven_max > cfg.compression_limit + 1e-6:
                reasons.append("Driven shoe compression screen exceeded")
            if wrench[1] < -1e-6:
                reasons.append("Shoe supplied tensile ground normal force")
            if np.linalg.norm(wrench[:2]) > cfg.maximum_force_n:
                reasons.append("Ground force screen exceeded")
            if reasons:
                failure = {"time_s": float(time), "reasons": reasons}
                break
            mass, bias = body.dynamics(state, velocity, cfg.gravity_m_s2)
            acceleration = np.linalg.solve(mass, generalized - bias)
            next_velocity = velocity + dt * acceleration
            next_state = state + dt * next_velocity
            if not np.isfinite(next_state).all() or not np.isfinite(next_velocity).all():
                raise FloatingPointError("Nonfinite integrated state or velocity")
            state, velocity = next_state, next_velocity
            integrated += 1
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            failure = {"time_s": float(time), "reasons": [str(error)]}
            break
    trace = {key: value[:recorded].copy() for key, value in trace.items()}
    trace["time_s"] = times[:recorded].copy()
    cap_rows = np.flatnonzero(trace["passive_cap_column_count"] > 0)
    summary = {
        "status": "failed" if failure else "completed",
        "failure": failure,
        "model": "cartesian_single_leg",
        "body_count": 3,
        "shoe_count": 1,
        "actuated_channels": 4,
        "mechanics_backend": "NumPy CPU; not a GPU-vectorized limb solver",
        "shoe_device": str(shoe.device),
        "actual_dt_s": dt,
        "integrated_steps": integrated,
        "integrated_duration_s": integrated * dt,
        "requested_duration_s": duration,
        "terminal_state": state.tolist(),
        "terminal_velocity": velocity.tolist(),
        "leg_mass_kg": float(np.sum(body.masses_kg)),
        "controller": "Fhip = Khip*(p_eq-p)-Dhip*v; tau = Kjoint*(theta_eq-theta)-Djoint*theta_dot",
        "external_loads": "hip point force, gravity on three leg masses, and one shoe-ground wrench only",
        "initial_contact_state": "zero material/friction histories; not a settled or periodic contact state",
        "trace_sampling": "preintegration, one contact update per row; terminal state/velocity in summary",
        "joint_limits_diagnostic": cfg.joint_limits_diagnostic,
        "joint_exceedances": joint_exceedances,
        "limit_handling": "no force/torque clipping, joint stops, prescribed motion, or state projection",
        "driven_compression_limit": cfg.compression_limit,
        "maximum_driven_compression_fraction": float(np.max(trace["driven_compression_fraction"]))
        if recorded
        else None,
        "maximum_passive_compression_fraction": float(np.max(trace["passive_compression_fraction"]))
        if recorded
        else None,
        "passive_cap_fraction": passive_cap,
        "passive_cap_hit": bool(len(cap_rows)),
        "passive_cap_steps": int(len(cap_rows)),
        "passive_cap_first_time_s": float(times[cap_rows[0]]) if len(cap_rows) else None,
        "passive_cap_max_columns": int(np.max(trace["passive_cap_column_count"])) if recorded else 0,
        "qualification": (
            "Exploratory numerical screens only; not physical acceptance or physiological validation. "
            "The shoe's existing passive compression cap is retained and reported, not treated as validated contact. "
            "No trunk, opposite leg, upper-body weight, hip torque, or measured-force input is present."
        ),
    }
    return trace, summary
