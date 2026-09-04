# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Velocity targets for unilateral LOX constraints."""

from __future__ import annotations

import warp as wp

__all__ = [
    "compute_contact_velocity_target",
    "compute_limit_velocity_target",
]

wp.set_module_options({"enable_backward": False})


@wp.func
def compute_contact_velocity_target(
    distance: wp.float32,
    previous_normal_velocity: wp.float32,
    restitution: wp.float32,
    time_step: wp.float32,
    stabilization_fraction: wp.float32,
    dead_zone: wp.float32,
    impact_velocity_threshold: wp.float32,
    recoverable_response: wp.bool,
) -> wp.float32:
    """Compute the minimum end-of-step normal contact velocity.

    Args:
        distance: Margin-shifted signed contact distance [m].
        previous_normal_velocity: Begin-of-step normal velocity [m/s].
        restitution: Newton restitution coefficient.
        time_step: Time step [s].
        stabilization_fraction: Penetration recovery fraction.
        dead_zone: Symmetric distance dead zone [m].
        impact_velocity_threshold: Minimum approaching impact speed [m/s].
        recoverable_response: Whether to permit restitution-recoverable overlap.

    Returns:
        Minimum feasible normal velocity [m/s].
    """
    distance_effective = wp.sign(distance) * wp.max(wp.abs(distance) - dead_zone, 0.0)
    velocity_gap = (
        -(stabilization_fraction * wp.min(distance_effective, 0.0) + wp.max(distance_effective, 0.0)) / time_step
    )
    # Permit the overlap whose next-step recovery matches the unreduced
    # restitution response.
    if (
        recoverable_response
        and distance_effective > 0.0
        and previous_normal_velocity < velocity_gap
        and previous_normal_velocity < -impact_velocity_threshold
        and stabilization_fraction > 0.0
    ):
        recoverable_overlap = -time_step * restitution * previous_normal_velocity / stabilization_fraction
        velocity_gap = -(distance_effective + recoverable_overlap) / time_step

    velocity_target = velocity_gap
    closed = distance <= dead_zone
    approaching = previous_normal_velocity < -impact_velocity_threshold
    if closed and approaching:
        velocity_bounce = -restitution * previous_normal_velocity
        velocity_target = wp.max(velocity_gap, velocity_bounce)
    return velocity_target


@wp.func
def compute_limit_velocity_target(
    violation: wp.float32,
    time_step: wp.float32,
    stabilization_fraction: wp.float32,
) -> wp.float32:
    """Compute the minimum end-of-step joint-limit velocity.

    Args:
        violation: Signed limit residual, negative when violated [m or rad].
        time_step: Time step [s].
        stabilization_fraction: Violation recovery fraction.

    Returns:
        Minimum feasible limit velocity [m/s or rad/s].
    """
    return -stabilization_fraction * wp.min(violation, 0.0) / time_step
