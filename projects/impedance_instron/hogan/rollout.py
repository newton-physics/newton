# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Roll out reference-tracking impedance through the pelvis-leg chain and one shoe."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .control import Impedance
from .mechanics import COORDINATE_NAMES, Chain
from .plan import CONTACT_THRESHOLD_N, Plan

PHASE_MODES = ("touchdown", "time")
CONTROL_LAW = "tau = tau_ff(phi) + K(phi) (q_ref(phi) - q) + D(phi) (v_ref(phi) - v)"


@dataclass(frozen=True)
class Config:
    """Rollout options and numerical screens, not physiological acceptance limits.

    ``phase="touchdown"`` shifts the plan clock at simulated touchdown so it
    matches the reference touchdown; this lets a different shoe change contact
    timing. ``"time"`` uses the clock directly. Without ``residual_feedforward``
    the pelvis channels receive impedance feedback only.
    """

    phase: str = "touchdown"
    residual_feedforward: bool = True
    gravity_m_s2: float = 9.81
    contact_threshold_n: float = CONTACT_THRESHOLD_N
    compression_limit: float = 0.9
    maximum_force_n: float = 6000.0
    minimum_hip_height_m: float = 0.2
    maximum_speed: float = 100.0

    def __post_init__(self):
        if self.phase not in PHASE_MODES:
            raise ValueError(f"phase must be one of {PHASE_MODES}")


def _rms(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values), axis=0))


def _metrics(trace: dict, plan: Plan, cfg: Config) -> dict:
    """Summarize tracking, contact, residual, and work outcomes of a recorded trace."""
    n = len(trace["time_s"])
    if n == 0:
        return {}
    dt = plan.dt_s
    grf = trace["grf_n"]
    contact = np.flatnonzero(grf[:, 1] > cfg.contact_threshold_n)
    feedback = trace["load"] - trace["feedforward"]
    power = trace["load"][:, 3:] * trace["velocity"][:, 3:]
    joints = ("hip", "knee", "ankle")
    return {
        "tracking_rmse": dict(zip(COORDINATE_NAMES, _rms(trace["state"] - plan.q[:n]).tolist(), strict=True)),
        "grf_rmse_n": _rms(grf - plan.grf_n[:n]).tolist(),
        "reference_grf_source": plan.contact_source,
        "peak_grf_n": np.max(grf, axis=0).tolist(),
        "grf_impulse_ns": (grf.sum(axis=0) * dt).tolist(),
        "touchdown_s": float(trace["time_s"][contact[0]]) if len(contact) else None,
        "toeoff_s": float(trace["time_s"][contact[-1]]) if len(contact) else None,
        "contact_duration_s": float(len(contact) * dt),
        "reference_touchdown_s": plan.touchdown_s,
        "residual_load_rms": _rms(trace["load"][:, :3]).tolist(),
        "feedback_rms": dict(zip(COORDINATE_NAMES, _rms(feedback).tolist(), strict=True)),
        "joint_positive_work_j": dict(zip(joints, (np.maximum(power, 0).sum(axis=0) * dt).tolist(), strict=True)),
        "joint_negative_work_j": dict(zip(joints, (np.minimum(power, 0).sum(axis=0) * dt).tolist(), strict=True)),
        "maximum_compression_fraction": float(np.max(trace["compression_fraction"])),
    }


def simulate(plan: Plan, chain: Chain, impedance: Impedance, shoe, *, config: Config | None = None):
    """Integrate the chain from the plan's initial state under impedance and shoe contact.

    Measured motion and force enter only through the plan's reference and
    feedforward; they never prescribe the state.

    Args:
        plan: Reference motion and feedforward on the integration clock.
        chain: Pelvis-leg mechanics.
        impedance: Stiffness and damping schedule, already scaled for the shoe.
        shoe: :class:`..cartesian.shoe.Shoe`. Its history is reset first.
        config: Phase policy and failure screens.

    Returns:
        Per-step trace and summary dictionaries.
    """
    cfg = config or Config()
    shoe.foundation.reset()
    driven = shoe.foundation.driven.numpy().astype(bool)
    rest_length = np.asarray(shoe.shoe.column_bed.rest_length_m, dtype=float)[driven]
    dt = plan.dt_s
    steps = len(plan.time_s) - 1
    shapes = {
        "phase_s": (),
        "state": (6,),
        "velocity": (6,),
        "reference_state": (6,),
        "load": (6,),
        "feedforward": (6,),
        "grf_n": (2,),
        "ankle_contact_moment_nm": (),
        "compression_fraction": (),
    }
    trace = {name: np.empty((steps, *shape)) for name, shape in shapes.items()}
    state, velocity = plan.q[0].copy(), plan.v[0].copy()
    foot_angular = chain.angular_jacobian(3)
    touchdown = None
    failure = None
    recorded = 0
    for k in range(steps):
        time = float(plan.time_s[k])
        phase = time if cfg.phase == "time" or touchdown is None else time - touchdown + plan.touchdown_s
        reference, reference_velocity, feedforward = plan.sample(phase)
        if not cfg.residual_feedforward:
            feedforward[:3] = 0.0
        stiffness, damping = impedance.gains(phase)
        load = feedforward + stiffness * (reference - state) + damping * (reference_velocity - velocity)
        ankle, jacobian, _ = chain.point(state, 3, np.zeros(2))
        wrench, _ = shoe.apply(ankle, jacobian @ velocity, chain.angle(state, 3), float(foot_angular @ velocity), dt)
        wrench = np.asarray(wrench, dtype=float)
        compression = float(np.max(shoe.foundation.compression.numpy()[driven] / rest_length))
        for name, value in (
            ("phase_s", phase),
            ("state", state),
            ("velocity", velocity),
            ("reference_state", reference),
            ("load", load),
            ("feedforward", feedforward),
            ("grf_n", wrench[:2]),
            ("ankle_contact_moment_nm", wrench[2]),
            ("compression_fraction", compression),
        ):
            trace[name][k] = value
        recorded += 1
        reasons = []
        if not np.isfinite(load).all() or not np.isfinite(wrench).all():
            reasons.append("Nonfinite actuator load or contact wrench")
        if state[1] < cfg.minimum_hip_height_m:
            reasons.append("Hip height screen exceeded")
        if np.linalg.norm(velocity[:2]) > cfg.maximum_speed or np.max(np.abs(velocity[2:])) > cfg.maximum_speed:
            reasons.append("Numerical speed screen exceeded")
        if np.linalg.norm(wrench[:2]) > cfg.maximum_force_n:
            reasons.append("Ground force screen exceeded")
        if compression > cfg.compression_limit + 1e-6:
            reasons.append("Driven shoe compression screen exceeded")
        if wrench[1] < -1e-6:
            reasons.append("Shoe supplied tensile ground normal force")
        if reasons:
            failure = {"time_s": time, "reasons": reasons}
            break
        if touchdown is None and wrench[1] > cfg.contact_threshold_n:
            touchdown = time
        mass, bias = chain.dynamics(state, velocity, cfg.gravity_m_s2)
        generalized = load + jacobian.T @ wrench[:2] + foot_angular * wrench[2]
        try:
            acceleration = np.linalg.solve(mass, generalized - bias)
        except np.linalg.LinAlgError as error:
            failure = {"time_s": time, "reasons": [str(error)]}
            break
        velocity = velocity + dt * acceleration
        state = state + dt * velocity
        if not np.isfinite(state).all() or not np.isfinite(velocity).all():
            failure = {"time_s": time, "reasons": ["Nonfinite integrated state or velocity"]}
            break
    shoe.foundation.reset()
    trace = {name: value[:recorded].copy() for name, value in trace.items()}
    trace["time_s"] = plan.time_s[:recorded].copy()
    summary = {
        "status": "failed" if failure else "completed",
        "failure": failure,
        "controller": CONTROL_LAW,
        "phase": cfg.phase,
        "residual_feedforward": cfg.residual_feedforward,
        "dt_s": dt,
        "integrated_duration_s": recorded * dt if failure is None else (recorded - 1) * dt,
        "requested_duration_s": plan.duration_s,
        "terminal_state": state.tolist(),
        "terminal_velocity": velocity.tolist(),
        **_metrics(trace, plan, cfg),
    }
    return trace, summary
