# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import warnings
from typing import Any

import warp as wp

from .base import Clamping


@wp.kernel(enable_backward=False)
def _corner_velocity_kernel(
    saturation_effort: wp.array[float],
    velocity_limit: wp.array[float],
    max_motor_effort: wp.array[float],
    corner_velocity: wp.array[float],
):
    """Velocity at which the effort-speed envelope reaches ``max_motor_effort``."""
    i = wp.tid()
    sat = saturation_effort[i]
    corner = velocity_limit[i]
    if sat > 0.0:
        ratio = max_motor_effort[i] / sat
        if ratio == ratio:
            corner = velocity_limit[i] * (1.0 + ratio)
    corner_velocity[i] = corner


@wp.func
def _evaluate_dc_motor_clamp(
    value: wp.float64,
    q: wp.float64,
    qd: wp.float64,
    params: wp.array2d[float],
    i: int,
    base: int,
) -> wp.float64:
    """Implicit-solve entry point; params row is ``[saturation, velocity_limit, max_effort]``.

    Inside the implicit solve ``qd`` is the predicted end-of-step velocity,
    so the effort-speed envelope is enforced self-consistently.
    """
    sat = wp.float64(params[i, base])
    vel_lim = wp.float64(params[i, base + 1])
    max_e = wp.float64(params[i, base + 2])
    # Derived, not stored: a cached corner goes stale when the user retunes.
    corner = vel_lim
    if sat > wp.float64(0.0):
        ratio = max_e / sat
        if ratio == ratio:
            corner = vel_lim * (wp.float64(1.0) + ratio)

    vel = wp.clamp(qd, -corner, corner)
    effort_max = wp.min(sat * (wp.float64(1.0) - vel / vel_lim), max_e)
    effort_min = wp.max(sat * (wp.float64(-1.0) - vel / vel_lim), -max_e)
    return wp.clamp(value, effort_min, effort_max)


@wp.kernel
def _clamp_dc_motor_kernel(
    current_vel: wp.array[float],
    state_indices: wp.array[wp.uint32],
    saturation_effort: wp.array[float],
    velocity_limit: wp.array[float],
    max_motor_effort: wp.array[float],
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

    # Derived, not stored: a cached corner goes stale when the user retunes.
    corner = vel_lim
    if sat > 0.0:
        ratio = max_e / sat
        if ratio == ratio:
            corner = vel_lim * (1.0 + ratio)
    vel = wp.clamp(current_vel[state_idx], -corner, corner)

    effort_max = wp.min(sat * (1.0 - vel / vel_lim), max_e)
    effort_min = wp.max(sat * (-1.0 - vel / vel_lim), -max_e)
    dst[i] = wp.clamp(src[i], effort_min, effort_max)


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
        if math.isinf(sat) and not math.isinf(vel_lim):
            # sat*(1 - v/v_lim) is inf*0 = NaN at v == v_lim.
            raise ValueError("saturation_effort must be finite when velocity_limit is finite")
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

    @property
    def corner_velocity(self) -> wp.array[float]:
        """Deprecated. Velocity at which the envelope reaches ``max_motor_effort``.

        .. deprecated:: 1.6
            Kept for compatibility and computed on access from the current
            parameters. It used to be stored at construction, which went stale
            when a parameter was retuned, so the clamp now derives it in-kernel
            and never reads this attribute.
        """
        warnings.warn(
            "ClampingDCMotor.corner_velocity is deprecated and will be removed in a future release. "
            "The clamp derives the corner velocity from the live parameters instead, so a stored "
            "copy is no longer used; compute it as velocity_limit * (1 + max_motor_effort / "
            "saturation_effort) if you need the value.",
            DeprecationWarning,
            stacklevel=2,
        )
        corner = wp.zeros_like(self.velocity_limit)
        wp.launch(
            _corner_velocity_kernel,
            dim=len(self.velocity_limit),
            inputs=[self.saturation_effort, self.velocity_limit, self.max_motor_effort],
            outputs=[corner],
            device=self.velocity_limit.device,
        )
        return corner

    evaluate_clamp = _evaluate_dc_motor_clamp

    def param_width(self) -> int:
        return 3

    def bind_params(self, block: wp.array2d[float]) -> None:
        block[:, 0].assign(self.saturation_effort)
        block[:, 1].assign(self.velocity_limit)
        block[:, 2].assign(self.max_motor_effort)
        self.saturation_effort = block[:, 0]
        self.velocity_limit = block[:, 1]
        self.max_motor_effort = block[:, 2]

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
                src_forces,
            ],
            outputs=[dst_forces],
            device=device,
        )
