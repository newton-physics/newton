# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any

import warp as wp

from .base import Clamping


@wp.kernel
def _compute_corner_velocity_kernel(
    saturation_effort: wp.array[float],
    velocity_limit: wp.array[float],
    max_motor_effort: wp.array[float],
    corner_velocity: wp.array[float],
):
    """Find the velocity on the torque-speed curve that intersects max_motor_effort in the second and fourth quadrant."""
    i = wp.tid()
    sat = saturation_effort[i]
    vel_lim = velocity_limit[i]
    max_e = max_motor_effort[i]
    if sat > 0.0:
        corner_velocity[i] = vel_lim * (1.0 + max_e / sat)
    else:
        corner_velocity[i] = vel_lim


@wp.kernel
def _clamp_dc_motor_kernel(
    current_vel: wp.array[float],
    state_indices: wp.array[wp.uint32],
    saturation_effort: wp.array[float],
    velocity_limit: wp.array[float],
    max_motor_effort: wp.array[float],
    corner_velocity: wp.array[float],
    src: wp.array[float],
    dst: wp.array[float],
):
    """DC motor four-quadrant effort-speed saturation: read src, write to dst.

    effort_max(vel) = min(saturation_effort * (1 - vel / velocity_limit),  max_motor_effort)
    effort_min(vel) = max(saturation_effort * (-1 - vel / velocity_limit), -max_motor_effort)
    """
    i = wp.tid()
    state_idx = state_indices[i]
    sat = saturation_effort[i]
    vel_lim = velocity_limit[i]
    max_e = max_motor_effort[i]

    vel = wp.clamp(current_vel[state_idx], -corner_velocity[i], corner_velocity[i])

    effort_max = wp.min(sat * (1.0 - vel / vel_lim), max_e)
    effort_min = wp.max(sat * (-1.0 - vel / vel_lim), -max_e)
    dst[i] = wp.clamp(src[i], effort_min, effort_max)


@wp.func
def _evaluate_dc_motor_clamp(
    value: wp.float64,
    q: wp.float64,
    qd: wp.float64,
    params: wp.array2d[float],
    i: int,
    base: int,
) -> wp.float64:
    """Implicit-solve entry point; params row is ``[saturation, velocity_limit, max_effort, corner_velocity]``.

    Inside the implicit solve ``qd`` is the predicted end-of-step velocity,
    so the effort-speed envelope is enforced self-consistently.
    """
    sat = wp.float64(params[i, base])
    vel_lim = wp.float64(params[i, base + 1])
    max_e = wp.float64(params[i, base + 2])
    corner = wp.float64(params[i, base + 3])

    vel = wp.clamp(qd, -corner, corner)
    effort_max = wp.min(sat * (wp.float64(1.0) - vel / vel_lim), max_e)
    effort_min = wp.max(sat * (wp.float64(-1.0) - vel / vel_lim), -max_e)
    return wp.clamp(value, effort_min, effort_max)


class ClampingDCMotor(Clamping):
    r"""DC motor four-quadrant effort-speed saturation.

    Clips controller output using the linear effort-speed characteristic::

        effort_max(vel) = min(saturation_effort * (1 - vel / velocity_limit),  max_motor_effort)
        effort_min(vel) = max(saturation_effort * (-1 - vel / velocity_limit), -max_motor_effort)

    At zero velocity the motor can produce up to ±\ ``saturation_effort``
    (capped by ``max_motor_effort``). As velocity approaches
    ``velocity_limit``, available effort in the direction of motion drops
    to zero.
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        sat = args.get("saturation_effort", math.inf)
        if sat < 0:
            raise ValueError(f"saturation_effort must be non-negative, got {sat}")
        vel_lim = args.get("velocity_limit", math.inf)
        if vel_lim <= 0:
            raise ValueError(f"velocity_limit must be positive, got {vel_lim}")
        max_motor_effort = args.get("max_motor_effort", math.inf)
        if max_motor_effort < 0:
            raise ValueError(f"max_motor_effort must be non-negative, got {max_motor_effort}")
        return {
            "saturation_effort": sat,
            "velocity_limit": vel_lim,
            "max_motor_effort": max_motor_effort,
        }

    def __init__(
        self,
        saturation_effort: wp.array[float],
        velocity_limit: wp.array[float],
        max_motor_effort: wp.array[float],
    ):
        """Initialize DC motor saturation.

        Args:
            saturation_effort: Peak motor effort at stall [N·m or N]. Shape ``(N,)``.
            velocity_limit: Maximum joint velocity [rad/s or m/s] for
                the effort-speed curve. Shape ``(N,)``.
            max_motor_effort: Effort limit for the effort-speed curve
                [N·m or N]. Shape ``(N,)``.
        """
        if saturation_effort.shape != velocity_limit.shape:
            raise ValueError(
                f"saturation_effort shape {saturation_effort.shape} "
                f"must match velocity_limit shape {velocity_limit.shape}"
            )
        if saturation_effort.shape != max_motor_effort.shape:
            raise ValueError(
                f"saturation_effort shape {saturation_effort.shape} "
                f"must match max_motor_effort shape {max_motor_effort.shape}"
            )
        self.saturation_effort = saturation_effort
        self.velocity_limit = velocity_limit
        self.max_motor_effort = max_motor_effort
        self.corner_velocity = wp.zeros_like(velocity_limit)
        wp.launch(
            kernel=_compute_corner_velocity_kernel,
            dim=len(velocity_limit),
            inputs=[saturation_effort, velocity_limit, max_motor_effort],
            outputs=[self.corner_velocity],
            device=velocity_limit.device,
        )

    evaluate_clamp = _evaluate_dc_motor_clamp

    def clamp_params(self) -> wp.array2d[float]:
        import numpy as np  # noqa: PLC0415

        pack = np.stack(
            [
                self.saturation_effort.numpy(),
                self.velocity_limit.numpy(),
                self.max_motor_effort.numpy(),
                self.corner_velocity.numpy(),
            ],
            axis=1,
        ).astype(np.float32)
        return wp.array(pack, dtype=float, device=self.saturation_effort.device)

    def bind_params(self, block: wp.array2d[float]) -> None:
        self.saturation_effort = block[:, 0]
        self.velocity_limit = block[:, 1]
        self.max_motor_effort = block[:, 2]
        self.corner_velocity = block[:, 3]

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
                self.max_motor_effort,
                self.corner_velocity,
                src_forces,
            ],
            outputs=[dst_forces],
            device=device,
        )
