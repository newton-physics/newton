# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reference motion and inverse-dynamics feedforward for the pelvis-leg chain."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..cartesian.data import validate as validate_reference
from .mechanics import UPRIGHT_PELVIS_RAD, Chain, from_leg_coordinates

# Matches the dataset's contact-assignment threshold.
CONTACT_THRESHOLD_N = 50.0
CONTACT_SOURCES = ("measured", "shoe_replay")
HIP_SOURCES = ("markers", "grf_com")


@dataclass(frozen=True)
class Plan:
    """Reference motion and feedforward loads on a uniform simulation clock.

    Attributes:
        time_s: Clock [s], shape [N].
        q: Reference coordinates [m, m, rad, rad, rad, rad], shape [N, 6].
        v: Reference velocities [m/s, m/s, rad/s, rad/s, rad/s, rad/s], shape [N, 6].
        a: Reference accelerations [m/s^2, m/s^2, rad/s^2, rad/s^2, rad/s^2, rad/s^2], shape [N, 6].
        feedforward: Inverse-dynamics loads [N, N, N m, N m, N m, N m], shape [N, 6].
            The pelvis channels are residuals that an exact rest-of-body model would not need.
        grf_n: Ground force used by inverse dynamics [N], shape [N, 2].
        touchdown_s: First time vertical force exceeds the contact threshold [s].
        contact_source: ``"measured"`` force and COP, or ``"shoe_replay"`` of the reference foot.
        diagnostics: Residual-load and contact summaries.
    """

    time_s: np.ndarray
    q: np.ndarray
    v: np.ndarray
    a: np.ndarray
    feedforward: np.ndarray
    grf_n: np.ndarray
    touchdown_s: float
    contact_source: str
    diagnostics: dict = field(default_factory=dict)

    @property
    def dt_s(self) -> float:
        """Return the uniform clock step [s]."""
        return float(self.time_s[1] - self.time_s[0])

    @property
    def duration_s(self) -> float:
        """Return the plan duration [s]."""
        return float(self.time_s[-1])

    def sample(self, phase_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate reference coordinates, velocities, and feedforward, holding the end values."""
        position = min(max(phase_s / self.dt_s, 0.0), len(self.time_s) - 1.0)
        i = min(int(position), len(self.time_s) - 2)
        w = position - i
        return (
            (1.0 - w) * self.q[i] + w * self.q[i + 1],
            (1.0 - w) * self.v[i] + w * self.v[i + 1],
            (1.0 - w) * self.feedforward[i] + w * self.feedforward[i + 1],
        )


def _resample(clock: np.ndarray, source: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Linearly resample each column of ``values`` onto ``clock``."""
    return np.column_stack([np.interp(clock, source, column) for column in np.asarray(values, dtype=float).T])


def lowpass(values: np.ndarray, sample_rate_hz: float, cutoff_hz: float) -> np.ndarray:
    """Apply a zero-phase second-order Butterworth low-pass filter along the first axis.

    Forward-backward filtering doubles the order to four with no phase lag. Odd
    reflection padding limits end transients.
    """
    if not 0 < cutoff_hz < 0.5 * sample_rate_hz:
        raise ValueError("cutoff_hz must lie between zero and the Nyquist frequency")
    k = math.tan(math.pi * cutoff_hz / sample_rate_hz)
    norm = 1.0 / (1.0 + math.sqrt(2.0) * k + k * k)
    b0 = k * k * norm
    a1 = 2.0 * (k * k - 1.0) * norm
    a2 = (1.0 - math.sqrt(2.0) * k + k * k) * norm

    def run(x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        x1 = x2 = y1 = y2 = x[0]
        for i, xi in enumerate(x):
            y[i] = b0 * (xi + 2.0 * x1 + x2) - a1 * y1 - a2 * y2
            x2, x1, y2, y1 = x1, xi, y1, y[i]
        return y

    values = np.asarray(values, dtype=float)
    pad = min(len(values) - 1, int(3 * sample_rate_hz / cutoff_hz))
    padded = np.concatenate(
        (2 * values[0] - values[pad:0:-1], values, 2 * values[-1] - values[-2 : -pad - 2 : -1]), axis=0
    )
    return run(run(padded)[::-1])[::-1][pad : pad + len(values)]


def build_plan(
    reference: dict,
    chain: Chain,
    *,
    dt_s: float,
    contact: str = "measured",
    shoe=None,
    gravity_m_s2: float = 9.81,
    contact_threshold_n: float = CONTACT_THRESHOLD_N,
    height_offset_m: float = 0.0,
    cutoff_hz: float | None = 12.0,
    hip_source: str | None = None,
    pelvis_cutoff_hz: float | None = 4.0,
    leg_ik: bool = True,
) -> Plan:
    """Compute reference motion and inverse-dynamics loads on a uniform clock.

    ``measured`` applies the measured GRF at the measured COP (``cop_target_m``),
    so the feedforward does not depend on any shoe model. ``shoe_replay`` drives
    the reference foot through ``shoe`` and uses its wrench: the loads that keep
    the measured motion in that shoe. Pelvis motion uses ``pelvis_target_rad``
    when present, otherwise a constant upright pelvis.

    Args:
        reference: Validated Cartesian single-leg reference arrays.
        chain: Pelvis-leg mechanics.
        dt_s: Requested clock step [s]; the actual step divides the duration evenly.
        contact: ``"measured"`` or ``"shoe_replay"``.
        shoe: :class:`..cartesian.shoe.Shoe` for ``shoe_replay``. Its history is reset.
        gravity_m_s2: Gravitational acceleration magnitude [m/s^2].
        contact_threshold_n: Vertical force that marks contact [N].
        height_offset_m: Vertical shift of the measured motion [m] that registers it to the
            shoe geometry; see :func:`.registration.register_static_height`.
        cutoff_hz: Kinematic low-pass cutoff before differentiation [Hz]; ``None`` skips
            filtering. Measured GRF is not filtered.
        hip_source: ``"markers"`` uses the measured hip center. ``"grf_com"`` keeps the measured
            angles but moves the hip so the whole-chain COM follows the double-integrated
            measured GRF, with initial position and velocity fitted to the markers. The lumped
            pelvis then needs no residual force. ``None`` selects ``"grf_com"`` for measured
            contact and ``"markers"`` for shoe replay.
        pelvis_cutoff_hz: Low-pass cutoff for ``pelvis_target_rad`` [Hz]. The lumped pelvis
            carries the trunk, which does not follow the fast pelvis tilt during impact; ``None``
            keeps the full measured tilt.
        leg_ik: Re-solve hip and knee angles with the chain's fixed segment lengths so the
            ankle follows ``ankle_target_m`` when present. Fixed lengths and 3D knee angles
            otherwise misplace the ankle, and with it the shoe, by centimeters in stance.
    """
    validate_reference(reference)
    if contact not in CONTACT_SOURCES:
        raise ValueError(f"contact must be one of {CONTACT_SOURCES}")
    if contact == "measured" and "cop_target_m" not in reference:
        raise ValueError(
            "Measured inverse dynamics needs cop_target_m; re-prepare the reference or use contact='shoe_replay'"
        )
    if contact == "shoe_replay" and shoe is None:
        raise ValueError("shoe_replay requires a shoe")
    if hip_source is None:
        hip_source = "grf_com" if contact == "measured" else "markers"
    if hip_source not in HIP_SOURCES:
        raise ValueError(f"hip_source must be one of {HIP_SOURCES}")
    if hip_source == "grf_com" and contact != "measured":
        raise ValueError("hip_source='grf_com' integrates the measured GRF and requires contact='measured'")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    time = reference["time_s"]
    duration = float(time[-1])
    steps = math.ceil(duration / dt_s)
    clock = np.linspace(0.0, duration, steps + 1)
    dt = duration / steps
    sample_rate = 1.0 / float(np.mean(np.diff(time)))
    if "pelvis_target_rad" in reference:
        pelvis = np.unwrap(reference["pelvis_target_rad"])
        if pelvis_cutoff_hz is not None:
            pelvis = lowpass(pelvis, sample_rate, pelvis_cutoff_hz)
        rate = np.gradient(pelvis, time, edge_order=2)
    else:
        pelvis = np.full(len(time), UPRIGHT_PELVIS_RAD)
        rate = np.zeros(len(time))
    q_motion, _ = from_leg_coordinates(reference["state"], reference["velocity"], pelvis, rate)
    q_motion[:, 1] += height_offset_m

    def smooth(values: np.ndarray) -> np.ndarray:
        return values if cutoff_hz is None else lowpass(values, sample_rate, cutoff_hz)

    q_motion = smooth(q_motion)
    measured_hip = q_motion[:, :2].copy()
    ankle_target = None
    if leg_ik and "ankle_target_m" in reference:
        ankle_target = smooth(reference["ankle_target_m"] + np.array([0.0, height_offset_m]))
        q_motion = chain.reach(q_motion, ankle_target)

    if hip_source == "grf_com":
        com_a = _resample(clock, reference["grf_time_s"], reference["grf_target_n"]) / chain.total_mass_kg
        com_a[:, 1] -= gravity_m_s2
        com_v = np.vstack((np.zeros(2), np.cumsum(0.5 * (com_a[1:] + com_a[:-1]) * dt, axis=0)))
        com_p = np.vstack((np.zeros(2), np.cumsum(0.5 * (com_v[1:] + com_v[:-1]) * dt, axis=0)))
        integrated = _resample(time, clock, com_p)
        basis = np.column_stack((np.ones(len(time)), time))
        # The COM offset depends on the leg angles, which depend on the hip when the ankle is pinned.
        for _ in range(20):
            offset = np.array([chain.com(row) - row[:2] for row in q_motion])
            initial, *_ = np.linalg.lstsq(basis, measured_hip + offset - integrated, rcond=None)
            hip = integrated + basis @ initial - offset
            change = float(np.max(np.abs(hip - q_motion[:, :2])))
            q_motion[:, :2] = hip
            if ankle_target is not None:
                q_motion = chain.reach(q_motion, ankle_target)
            if change < 1e-7:
                break
        offset = np.array([chain.com(row) - row[:2] for row in q_motion])

    v_motion = np.gradient(q_motion, time, axis=0, edge_order=2)
    a_motion = np.gradient(v_motion, time, axis=0, edge_order=2)
    q, v, a = (_resample(clock, time, values) for values in (q_motion, v_motion, a_motion))

    if hip_source == "grf_com":
        offset_v = np.gradient(offset, time, axis=0, edge_order=2)
        offset_a = np.gradient(offset_v, time, axis=0, edge_order=2)
        r, rv, ra = (_resample(clock, time, values) for values in (offset, offset_v, offset_a))
        fine_basis = np.column_stack((np.ones(len(clock)), clock))
        q[:, :2] = com_p + fine_basis @ initial - r
        v[:, :2] = com_v + initial[1] - rv
        a[:, :2] = com_a - ra
    adjustment = q[:, :2] - _resample(clock, time, measured_hip)
    ankle_error = np.zeros((len(clock), 2))
    if ankle_target is not None:
        ankle_error = np.array([chain.kinematics(row)[2] for row in q]) - _resample(clock, time, ankle_target)

    loads = np.zeros((len(clock), 6))
    grf = np.zeros((len(clock), 2))
    foot_angular = chain.angular_jacobian(3)
    cop_local_x = np.full(len(clock), np.nan)
    if contact == "measured":
        grf = _resample(clock, reference["grf_time_s"], reference["grf_target_n"])
        cop = np.interp(clock, reference["grf_time_s"], reference["cop_target_m"])
        for k in range(len(clock)):
            ankle, _, _ = chain.point(q[k], 3, np.zeros(2))
            angle = chain.angle(q[k], 3)
            offset = np.array([cop[k], 0.0]) - ankle
            local = np.array(
                [
                    math.cos(angle) * offset[0] + math.sin(angle) * offset[1],
                    -math.sin(angle) * offset[0] + math.cos(angle) * offset[1],
                ]
            )
            _, jacobian, _ = chain.point(q[k], 3, local)
            loads[k] = jacobian.T @ grf[k]
            cop_local_x[k] = local[0]
    else:
        shoe.foundation.reset()
        for k in range(len(clock)):
            ankle, jacobian, _ = chain.point(q[k], 3, np.zeros(2))
            wrench, _ = shoe.apply(ankle, jacobian @ v[k], chain.angle(q[k], 3), float(foot_angular @ v[k]), dt)
            grf[k] = wrench[:2]
            loads[k] = jacobian.T @ wrench[:2] + foot_angular * wrench[2]
        shoe.foundation.reset()

    feedforward = np.empty_like(loads)
    for k in range(len(clock)):
        mass, bias = chain.dynamics(q[k], v[k], gravity_m_s2)
        feedforward[k] = mass @ a[k] + bias - loads[k]
    in_contact = grf[:, 1] > contact_threshold_n
    if not np.any(in_contact):
        raise ValueError("Reference contains no contact above the threshold")
    touchdown = float(clock[np.argmax(in_contact)])
    weight = chain.total_mass_kg * gravity_m_s2
    residual_rms = np.sqrt(np.mean(np.square(feedforward[:, :3]), axis=0))
    diagnostics = {
        "contact_source": contact,
        "height_offset_m": height_offset_m,
        "kinematic_cutoff_hz": cutoff_hz,
        "hip_source": hip_source,
        "hip_adjustment_rms_m": np.sqrt(np.mean(np.square(adjustment), axis=0)).tolist(),
        "hip_adjustment_peak_m": np.max(np.abs(adjustment), axis=0).tolist(),
        "ankle_source": "ankle_target_m" if ankle_target is not None else "forward kinematics",
        "ankle_error_peak_m": np.max(np.abs(ankle_error), axis=0).tolist(),
        "pelvis_source": "pelvis_target_rad" if "pelvis_target_rad" in reference else "constant upright pelvis",
        "pelvis_cutoff_hz": pelvis_cutoff_hz,
        "touchdown_s": touchdown,
        "contact_duration_s": float(np.count_nonzero(in_contact) * dt),
        "residual_force_rms_n": residual_rms[:2].tolist(),
        "residual_force_rms_body_weight": (residual_rms[:2] / weight).tolist(),
        "residual_force_peak_n": np.max(np.abs(feedforward[:, :2]), axis=0).tolist(),
        "residual_moment_rms_nm": float(residual_rms[2]),
        "residual_moment_peak_nm": float(np.max(np.abs(feedforward[:, 2]))),
        "joint_torque_peak_nm": dict(
            zip(("hip", "knee", "ankle"), np.max(np.abs(feedforward[:, 3:]), axis=0).tolist(), strict=True)
        ),
        "peak_grf_n": np.max(grf, axis=0).tolist(),
    }
    if contact == "measured":
        diagnostics["cop_local_x_range_m"] = [
            float(np.min(cop_local_x[in_contact])),
            float(np.max(cop_local_x[in_contact])),
        ]
    return Plan(clock, q, v, a, feedforward, grf, touchdown, contact, diagnostics)
