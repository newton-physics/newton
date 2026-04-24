# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import warp as wp

from .base import Controller, _masked_zero_1d


@wp.kernel
def _pid_effort_kernel(
    current_pos: wp.array[float],
    current_vel: wp.array[float],
    target_pos: wp.array[float],
    target_vel: wp.array[float],
    feedforward: wp.array[float],
    pos_indices: wp.array[wp.uint32],
    vel_indices: wp.array[wp.uint32],
    target_pos_indices: wp.array[wp.uint32],
    target_vel_indices: wp.array[wp.uint32],
    kp: wp.array[float],
    ki: wp.array[float],
    kd: wp.array[float],
    integral_max: wp.array[float],
    const_effort: wp.array[float],
    dt: float,
    current_integral: wp.array[float],
    efforts: wp.array[float],
    next_integral: wp.array[float],
):
    """effort = const_effort + feedforward + kp*(target_pos - current_pos) + ki*integral(target_pos - current_pos) + kd*(target_vel - current_vel)."""
    i = wp.tid()
    pos_idx = pos_indices[i]
    vel_idx = vel_indices[i]
    tgt_pos_idx = target_pos_indices[i]
    tgt_vel_idx = target_vel_indices[i]

    position_error = target_pos[tgt_pos_idx] - current_pos[pos_idx]
    velocity_error = target_vel[tgt_vel_idx] - current_vel[vel_idx]

    integral = current_integral[i] + position_error * dt
    integral = wp.clamp(integral, -integral_max[i], integral_max[i])

    const_e = float(0.0)
    if const_effort:
        const_e = const_effort[i]

    ff = float(0.0)
    if feedforward:
        ff = feedforward[tgt_vel_idx]

    effort = const_e + ff + kp[i] * position_error + ki[i] * integral + kd[i] * velocity_error
    efforts[i] = effort
    next_integral[i] = integral


class ControllerPID(Controller):
    """Stateful PID (Proportional-Integral-Derivative) controller.

    Effort law::

        effort = const_effort + feedforward + kp * (target_pos - current_pos)
               + ki * integral(target_pos - current_pos) + kd * (target_vel - current_vel)

    Maintains an integral term with anti-windup clamping.
    """

    @dataclass
    class State(Controller.State):
        """Integral state for PID controller."""

        integral: wp.array[float] | None = None
        """Accumulated integral of position error [m·s or rad·s], shape ``(N,)``."""

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            if mask is None:
                self.integral.zero_()
            else:
                wp.launch(_masked_zero_1d, dim=len(mask), inputs=[self.integral, mask])

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        kp = args.get("kp", 0.0)
        if kp < 0:
            raise ValueError(f"kp must be non-negative, got {kp}")
        ki = args.get("ki", 0.0)
        if ki < 0:
            raise ValueError(f"ki must be non-negative, got {ki}")
        kd = args.get("kd", 0.0)
        if kd < 0:
            raise ValueError(f"kd must be non-negative, got {kd}")
        integral_max = args.get("integral_max", math.inf)
        if integral_max < 0:
            raise ValueError(f"integral_max must be non-negative, got {integral_max}")
        return {
            "kp": kp,
            "ki": ki,
            "kd": kd,
            "integral_max": integral_max,
            "const_effort": args.get("const_effort", 0.0),
        }

    def __init__(
        self,
        kp: wp.array[float],
        ki: wp.array[float],
        kd: wp.array[float],
        integral_max: wp.array[float],
        const_effort: wp.array[float] | None = None,
    ):
        """Initialize PID controller.

        Args:
            kp: Proportional gains [N/m or N·m/rad]. Shape ``(N,)``.
            ki: Integral gains [N/(m·s) or N·m/(rad·s)]. Shape ``(N,)``.
            kd: Derivative gains [N·s/m or N·m·s/rad]. Shape ``(N,)``.
            integral_max: Anti-windup limits [m·s or rad·s]. Shape ``(N,)``.
            const_effort: Constant bias effort [N or N·m]. Shape ``(N,)``. ``None`` to skip.
        """
        if kp.shape != ki.shape:
            raise ValueError(f"kp shape {kp.shape} must match ki shape {ki.shape}")
        if kp.shape != kd.shape:
            raise ValueError(f"kp shape {kp.shape} must match kd shape {kd.shape}")
        if kp.shape != integral_max.shape:
            raise ValueError(f"kp shape {kp.shape} must match integral_max shape {integral_max.shape}")
        if const_effort is not None and const_effort.shape != kp.shape:
            raise ValueError(f"const_effort shape {const_effort.shape} must match kp shape {kp.shape}")
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_max = integral_max
        self.const_effort = const_effort
        self._next_integral: wp.array[float] | None = None

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        self._next_integral = wp.zeros(num_actuators, dtype=wp.float32, device=device)

    def is_stateful(self) -> bool:
        return True

    def is_graphable(self) -> bool:
        return True

    def state(self, num_actuators: int, device: wp.Device) -> ControllerPID.State:
        return ControllerPID.State(
            integral=wp.zeros(num_actuators, dtype=wp.float32, device=device),
        )

    def compute(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float] | None,
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        target_pos_indices: wp.array[wp.uint32],
        target_vel_indices: wp.array[wp.uint32],
        forces: wp.array[float],
        state: ControllerPID.State,
        dt: float,
        device: wp.Device | None = None,
    ) -> None:
        wp.launch(
            kernel=_pid_effort_kernel,
            dim=len(forces),
            inputs=[
                positions,
                velocities,
                target_pos,
                target_vel,
                feedforward,
                pos_indices,
                vel_indices,
                target_pos_indices,
                target_vel_indices,
                self.kp,
                self.ki,
                self.kd,
                self.integral_max,
                self.const_effort,
                dt,
                state.integral,
            ],
            outputs=[forces, self._next_integral],
            device=device,
        )

    def update_state(
        self,
        current_state: ControllerPID.State,
        next_state: ControllerPID.State,
    ) -> None:
        wp.copy(next_state.integral, self._next_integral)
