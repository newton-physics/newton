# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared residual controllers and bound projection for stance-conditioned fitting."""

from __future__ import annotations

import numpy as np

from ..spline import _derivative_control_polygons, _uniform_knots
from ..trajectory import Spline


def nominal_six_channel(reference: dict, profile: dict, controls: int = 12) -> np.ndarray:
    """Build a six-channel PD-compensated nominal spline from one stance.

    Channels are hip x/z, knee/ankle angles, then ankle x/z. Ankle velocity
    comes from the two segment Jacobian terms evaluated along reference motion.
    """
    if controls != 12:
        raise ValueError("The shared controller requires exactly twelve controls")
    time = np.asarray(reference["time_s"], dtype=np.float64)
    state = np.asarray(reference["state"], dtype=np.float64)
    velocity = np.asarray(reference["velocity"], dtype=np.float64)
    lengths = np.asarray(reference["lengths_m"], dtype=np.float64)
    if time.ndim != 1 or state.shape != (len(time), 5) or velocity.shape != state.shape:
        raise ValueError("Reference state and velocity must match its time grid")
    duration = float(time[-1])
    if duration <= 0 or len(profile["equilibrium_lower"]) != 6:
        raise ValueError("A positive-duration reference and six-channel profile are required")
    stiffness = np.asarray(
        [*profile["hip_stiffness_n_m"], *profile["joint_stiffness_nm_rad"], *profile["ankle_stiffness_n_m"]],
        dtype=np.float64,
    )
    damping = np.asarray(
        [*profile["hip_damping_ns_m"], *profile["joint_damping_nms_rad"], *profile["ankle_damping_ns_m"]],
        dtype=np.float64,
    )
    if (
        stiffness.shape != (6,)
        or damping.shape != (6,)
        or not np.isfinite(stiffness).all()
        or not np.isfinite(damping).all()
        or np.any(stiffness <= 0)
    ):
        raise ValueError("Profile gains must provide six positive stiffness and finite damping values")
    neutral = np.empty((len(time), 6), dtype=np.float64)
    neutral[:, :4] = state[:, [0, 1, 3, 4]] + velocity[:, [0, 1, 3, 4]] * damping[:4] / stiffness[:4]
    a0, a1 = state[:, 2], state[:, 2] + state[:, 3]
    dx = lengths[0] * np.cos(a0) + lengths[1] * np.cos(a1)
    dz = lengths[0] * np.sin(a0) + lengths[1] * np.sin(a1)
    vx = (
        velocity[:, 0]
        - lengths[0] * np.sin(a0) * velocity[:, 2]
        - lengths[1] * np.sin(a1) * (velocity[:, 2] + velocity[:, 3])
    )
    vz = (
        velocity[:, 1]
        + lengths[0] * np.cos(a0) * velocity[:, 2]
        + lengths[1] * np.cos(a1) * (velocity[:, 2] + velocity[:, 3])
    )
    neutral[:, 4] = state[:, 0] + dx + damping[4] / stiffness[4] * vx
    neutral[:, 5] = state[:, 1] + dz + damping[5] / stiffness[5] * vz
    knots = _uniform_knots(controls, 3)
    sample_times = duration * np.asarray([np.mean(knots[i + 1 : i + 4]) for i in range(controls)])
    raw = np.column_stack([np.interp(sample_times, time, neutral[:, channel]) for channel in range(6)])
    lower, upper = np.asarray(profile["equilibrium_lower"]), np.asarray(profile["equilibrium_upper"])
    rate, acceleration = (
        np.asarray(profile["equilibrium_rate_limit"]),
        np.asarray(profile["equilibrium_acceleration_limit"]),
    )
    anchor = np.clip(neutral[0], lower, upper)
    delta = raw - anchor
    first, second = _derivative_control_polygons(delta)
    scale = np.ones(6)
    for channel in range(6):
        positive, negative = delta[:, channel] > 0, delta[:, channel] < 0
        if np.any(positive):
            scale[channel] = min(
                scale[channel], float(np.min((upper[channel] - anchor[channel]) / delta[positive, channel]))
            )
        if np.any(negative):
            scale[channel] = min(
                scale[channel], float(np.min((lower[channel] - anchor[channel]) / delta[negative, channel]))
            )
        for polygon, bound in ((first, rate[channel] * duration), (second, acceleration[channel] * duration**2)):
            peak = float(np.max(np.abs(polygon[:, channel])))
            if peak:
                scale[channel] = min(scale[channel], bound / peak)
    scale = np.where(scale < 1.0, scale * (1.0 - 1.0e-12), scale)
    for _ in range(40):
        result = anchor + delta * scale
        if Spline(duration, result).bounds(lower, upper, rate, acceleration):
            return result
        scale *= 0.5
    raise ValueError("Unable to construct a bounded six-channel nominal spline")


def project_shared_residual(
    nominals: np.ndarray,
    residual: np.ndarray,
    durations: np.ndarray,
    profile: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Contract the shared residual independently to each stance's strict bounds.

    Args:
        nominals: Stance-specific six-channel coefficient arrays [stance, 12, 6].
        residual: Shared coefficient offset [12, 6].
        durations: Positive duration for each stance.
        profile: Frozen six-channel bound profile.

    Returns:
        Valid coefficient arrays and one recorded contraction factor per stance.
    """
    nominals = np.asarray(nominals, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    if nominals.ndim != 3 or nominals.shape[1:] != (12, 6) or residual.shape != (12, 6):
        raise ValueError("Expected stance nominals [stance, 12, 6] and one shared [12, 6] residual")
    if durations.shape != (len(nominals),) or np.any(durations <= 0) or not np.isfinite(residual).all():
        raise ValueError("Every nominal must have one positive duration")
    lower, upper = np.asarray(profile["equilibrium_lower"]), np.asarray(profile["equilibrium_upper"])
    rate, acceleration = (
        np.asarray(profile["equilibrium_rate_limit"]),
        np.asarray(profile["equilibrium_acceleration_limit"]),
    )
    scales = np.ones(len(nominals), dtype=np.float64)
    for stance, (duration, nominal) in enumerate(zip(durations, nominals, strict=True)):
        if not Spline(duration, nominal).bounds(lower, upper, rate, acceleration):
            raise ValueError("Nominal spline is not within the frozen profile bounds")
        nominal_first, nominal_second = _derivative_control_polygons(nominal)
        delta_first, delta_second = _derivative_control_polygons(residual)
        alpha_low, alpha_high = 0.0, 1.0

        def constrain(base, slope, lo, hi):
            nonlocal alpha_low, alpha_high
            base, slope = np.asarray(base), np.asarray(slope)
            lo = np.broadcast_to(np.asarray(lo), base.shape)
            hi = np.broadcast_to(np.asarray(hi), base.shape)
            moving = np.abs(slope) > 0
            if np.any((~moving) & ((base < lo) | (base > hi))):
                raise ValueError("Nominal spline is outside a declared bound")
            if np.any(moving):
                bound_a = (lo[moving] - base[moving]) / slope[moving]
                bound_b = (hi[moving] - base[moving]) / slope[moving]
                alpha_low = max(alpha_low, float(np.max(np.minimum(bound_a, bound_b))))
                alpha_high = min(alpha_high, float(np.min(np.maximum(bound_a, bound_b))))

        constrain(nominal, residual, lower[None, :], upper[None, :])
        for channel in range(6):
            constrain(
                nominal_first[:, channel], delta_first[:, channel], -rate[channel] * duration, rate[channel] * duration
            )
            constrain(
                nominal_second[:, channel],
                delta_second[:, channel],
                -acceleration[channel] * duration**2,
                acceleration[channel] * duration**2,
            )
        if alpha_low > alpha_high or alpha_high < 0 or alpha_low > 1:
            raise ValueError("No residual contraction satisfies stance spline bounds")
        scales[stance] = min(1.0, max(0.0, alpha_high))
        if scales[stance] < 1.0:
            scales[stance] *= 1.0 - 1.0e-12
    coefficients = nominals + scales[:, None, None] * residual[None, :, :]
    for stance, (duration, row) in enumerate(zip(durations, coefficients, strict=True)):
        if not Spline(duration, row).bounds(lower, upper, rate, acceleration):
            lo, hi = 0.0, scales[stance]
            for _ in range(64):
                mid = 0.5 * (lo + hi)
                candidate = nominals[stance] + mid * residual
                if Spline(duration, candidate).bounds(lower, upper, rate, acceleration):
                    lo = mid
                else:
                    hi = mid
            scales[stance] = lo
            coefficients[stance] = nominals[stance] + lo * residual
            if not Spline(duration, coefficients[stance]).bounds(lower, upper, rate, acceleration):
                raise ValueError("Numerical shared-residual projection did not satisfy strict spline bounds")
    return coefficients, scales
