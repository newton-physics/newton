# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import warp as wp

from .base import Clamping


@wp.kernel
def _clamp_dc_motor_kernel(
    current_vel: wp.array(dtype=float),
    state_indices: wp.array(dtype=wp.uint32),
    saturation_effort: wp.array(dtype=float),
    velocity_limit: wp.array(dtype=float),
    max_force: wp.array(dtype=float),
    src: wp.array(dtype=float),
    dst: wp.array(dtype=float),
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

    Clips controller output using the torque–speed characteristic:
        τ_max(v) = clamp(τ_sat*(1 - v/v_max),  0,  effort_limit)
        τ_min(v) = clamp(τ_sat*(-1 - v/v_max), -effort_limit, 0)

    At zero velocity the motor can produce up to ±τ_sat (capped by
    effort_limit). As velocity approaches v_max, available torque in
    the direction of motion drops to zero.
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        if "velocity_limit" not in args:
            raise ValueError("ClampingDCMotor requires 'velocity_limit' argument")
        return {
            "saturation_effort": args.get("saturation_effort", math.inf),
            "velocity_limit": args["velocity_limit"],
            "max_force": args.get("max_force", math.inf),
        }

    def __init__(
        self,
        saturation_effort: wp.array,
        velocity_limit: wp.array,
        max_force: wp.array,
    ):
        """Initialize DC motor saturation.

        Args:
            saturation_effort: Peak motor torque at stall. Shape (N,).
            velocity_limit: Maximum joint velocity for torque-speed curve. Shape (N,).
            max_force: Absolute effort limits (continuous-rated). Shape (N,).
        """
        self.saturation_effort = saturation_effort
        self.velocity_limit = velocity_limit
        self.max_force = max_force

    def modify_forces(
        self,
        src_forces: wp.array,
        dst_forces: wp.array,
        positions: wp.array,
        velocities: wp.array,
        input_indices: wp.array,
        num_actuators: int,
    ) -> None:
        wp.launch(
            kernel=_clamp_dc_motor_kernel,
            dim=num_actuators,
            inputs=[
                velocities,
                input_indices,
                self.saturation_effort,
                self.velocity_limit,
                self.max_force,
                src_forces,
            ],
            outputs=[dst_forces],
        )
