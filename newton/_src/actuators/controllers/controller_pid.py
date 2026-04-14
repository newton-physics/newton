# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass
from typing import Any

import warp as wp

from .base import Controller


@wp.kernel
def _pid_force_kernel(
    current_pos: wp.array(dtype=float),
    current_vel: wp.array(dtype=float),
    target_pos: wp.array(dtype=float),
    target_vel: wp.array(dtype=float),
    control_input: wp.array(dtype=float),
    state_indices: wp.array(dtype=wp.uint32),
    target_indices: wp.array(dtype=wp.uint32),
    force_indices: wp.array(dtype=wp.uint32),
    kp: wp.array(dtype=float),
    ki: wp.array(dtype=float),
    kd: wp.array(dtype=float),
    integral_max: wp.array(dtype=float),
    constant_force: wp.array(dtype=float),
    dt: float,
    current_integral: wp.array(dtype=float),
    forces: wp.array(dtype=float),
    next_integral: wp.array(dtype=float),
):
    """PID force: f = constant + act + kp*e + ki*integral + kd*de."""
    i = wp.tid()
    state_idx = state_indices[i]
    target_idx = target_indices[i]
    force_idx = force_indices[i]

    position_error = target_pos[target_idx] - current_pos[state_idx]
    velocity_error = target_vel[target_idx] - current_vel[state_idx]

    integral = current_integral[i] + position_error * dt
    integral = wp.clamp(integral, -integral_max[i], integral_max[i])

    const_f = float(0.0)
    if constant_force:
        const_f = constant_force[i]

    act = float(0.0)
    if control_input:
        act = control_input[target_idx]

    force = const_f + act + kp[i] * position_error + ki[i] * integral + kd[i] * velocity_error
    forces[force_idx] = force
    next_integral[i] = integral


class ControllerPID(Controller):
    """Stateful PID controller.

    Force law: f = constant + act + Kp*e + Ki*∫e·dt + Kd*de

    Maintains an integral term with integral clamping.
    """

    @dataclass
    class State:
        """Integral state for PID controller."""

        integral: wp.array = None  # Shape (N,)

        def reset(self) -> None:
            self.integral.zero_()

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "kp": args.get("kp", 0.0),
            "ki": args.get("ki", 0.0),
            "kd": args.get("kd", 0.0),
            "integral_max": args.get("integral_max", math.inf),
            "constant_force": args.get("constant_force", 0.0),
        }

    def __init__(
        self,
        kp: wp.array,
        ki: wp.array,
        kd: wp.array,
        integral_max: wp.array,
        constant_force: wp.array | None = None,
    ):
        """Initialize PID controller.

        Args:
            kp: Proportional gains. Shape (N,).
            ki: Integral gains. Shape (N,).
            kd: Derivative gains. Shape (N,).
            integral_max: Anti-windup limits. Shape (N,).
            constant_force: Constant force offsets. Shape (N,). None to skip.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_max = integral_max
        self.constant_force = constant_force
        self._next_integral: wp.array | None = None

    def set_indices(self, input_indices: wp.array, sequential_indices: wp.array) -> None:
        num = len(input_indices)
        device = input_indices.device
        self._next_integral = wp.zeros(num, dtype=wp.float32, device=device)

    def is_stateful(self) -> bool:
        return True

    def state(self, num_actuators: int, device: wp.Device) -> "ControllerPID.State":
        return ControllerPID.State(
            integral=wp.zeros(num_actuators, dtype=wp.float32, device=device),
        )

    def compute(
        self,
        positions: wp.array,
        velocities: wp.array,
        target_pos: wp.array,
        target_vel: wp.array,
        act_input: wp.array | None,
        input_indices: wp.array,
        target_indices: wp.array,
        forces: wp.array,
        force_indices: wp.array,
        num_actuators: int,
        state: "ControllerPID.State",
        dt: float,
    ) -> None:
        wp.launch(
            kernel=_pid_force_kernel,
            dim=num_actuators,
            inputs=[
                positions,
                velocities,
                target_pos,
                target_vel,
                act_input,
                input_indices,
                target_indices,
                force_indices,
                self.kp,
                self.ki,
                self.kd,
                self.integral_max,
                self.constant_force,
                dt,
                state.integral,
            ],
            outputs=[forces, self._next_integral],
        )

    def update_state(
        self,
        current_state: "ControllerPID.State",
        next_state: "ControllerPID.State",
    ) -> None:
        wp.copy(next_state.integral, self._next_integral)
