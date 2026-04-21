# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any

import warp as wp

from .base import Clamping


@wp.kernel
def _clamp_dc_motor_kernel(
    current_vel: wp.array[float],
    state_indices: wp.array[wp.uint32],
    saturation_effort: wp.array[float],
    velocity_limit: wp.array[float],
    max_force: wp.array[float],
    src: wp.array[float],
    dst: wp.array[float],
):
    """DC motor velocity-dependent saturation: read src, write to dst.

    τ_max(v) = clamp(τ_sat*(1 - v/v_max),  0,  max_force)
    τ_min(v) = clamp(τ_sat*(-1 - v/v_max), -max_force, 0)
    """
    i = wp.tid()
    state_idx = state_indices[i]
    vel = current_vel[state_idx]
    sat = saturation_effort[i]
    vel_lim = velocity_limit[i]
    max_f = max_force[i]

    max_torque = wp.clamp(sat * (1.0 - vel / vel_lim), 0.0, max_f)
    min_torque = wp.clamp(sat * (-1.0 - vel / vel_lim), -max_f, 0.0)
    dst[i] = wp.clamp(src[i], min_torque, max_torque)


class ClampingDCMotor(Clamping):
    """DC motor velocity-dependent torque saturation.

    Clips controller output using the torque-speed characteristic:
        τ_max(v) = clamp(τ_sat*(1 - v/v_max),  0,  effort_limit)
        τ_min(v) = clamp(τ_sat*(-1 - v/v_max), -effort_limit, 0)

    At zero velocity the motor can produce up to ±τ_sat (capped by
    effort_limit). As velocity approaches v_max, available torque in
    the direction of motion drops to zero.
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        if "velocity_limit" not in args:
            raise ValueError("ClampingDCMotor requires 'velocity_limit'")
        vel_lim = args["velocity_limit"]
        if vel_lim <= 0:
            raise ValueError(f"velocity_limit must be positive, got {vel_lim}")
        if "saturation_effort" not in args:
            raise ValueError("ClampingDCMotor requires 'saturation_effort'")
        sat = args["saturation_effort"]
        if sat <= 0:
            raise ValueError(f"saturation_effort must be positive, got {sat}")
        max_force = args.get("max_force", math.inf)
        if max_force < 0:
            raise ValueError(f"max_force must be non-negative, got {max_force}")
        return {
            "saturation_effort": sat,
            "velocity_limit": vel_lim,
            "max_force": max_force,
        }

    def __init__(
        self,
        saturation_effort: wp.array[float],
        velocity_limit: wp.array[float],
        max_force: wp.array[float],
    ):
        """Initialize DC motor saturation.

        Args:
            saturation_effort: Peak motor torque at stall [N·m]. Shape (N,).
            velocity_limit: Maximum joint velocity [rad/s] for the torque-speed curve. Shape (N,).
            max_force: Absolute effort limits [N or N·m] (continuous-rated). Shape (N,).
        """
        if saturation_effort.shape != velocity_limit.shape:
            raise ValueError(
                f"saturation_effort shape {saturation_effort.shape} "
                f"must match velocity_limit shape {velocity_limit.shape}"
            )
        if saturation_effort.shape != max_force.shape:
            raise ValueError(
                f"saturation_effort shape {saturation_effort.shape} "
                f"must match max_force shape {max_force.shape}"
            )
        self.saturation_effort = saturation_effort
        self.velocity_limit = velocity_limit
        self.max_force = max_force

    def modify_forces(
        self,
        src_forces: wp.array[float],
        dst_forces: wp.array[float],
        positions: wp.array[float],
        velocities: wp.array[float],
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        device: wp.Device | None = None,
    ) -> None:
        wp.launch(
            kernel=_clamp_dc_motor_kernel,
            dim=len(src_forces),
            inputs=[
                velocities,
                vel_indices,
                self.saturation_effort,
                self.velocity_limit,
                self.max_force,
                src_forces,
            ],
            outputs=[dst_forces],
            device=device,
        )
