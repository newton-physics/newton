# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import warp as wp

from ..kernels import pid_controller_kernel, pid_integral_state_kernel
from .base import Actuator

if TYPE_CHECKING:
    from ...sim.control import Control
    from ...sim.state import State


class ActuatorPID(Actuator):
    """Stateful PID controller.

    Control law::

        τ = clamp(constant + act + Kp·(target_pos - q) + Ki·∫e·dt + Kd·(target_vel - v), ±max_force)

    Stateful: maintains integral term with anti-windup clamping.
    """

    @dataclass
    class State:
        """Integral state for PID actuators."""

        integral: wp.array[float] | None = None  # Shape (N,)

        def reset(self) -> None:
            """Zero integral state in-place."""
            self.integral.zero_()

    def is_stateful(self) -> bool:
        return True

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve arguments with defaults.

        Args:
            args: User-provided arguments.

        Returns:
            Arguments with defaults (``kp=0``, ``ki=0``, ``kd=0``,
            ``max_force=inf``, ``integral_max=inf``, ``constant_force=0``).
        """
        return {
            "kp": args.get("kp", 0.0),
            "ki": args.get("ki", 0.0),
            "kd": args.get("kd", 0.0),
            "max_force": args.get("max_force", math.inf),
            "integral_max": args.get("integral_max", math.inf),
            "constant_force": args.get("constant_force", 0.0),
        }

    def __init__(
        self,
        input_indices: wp.array[wp.uint32],
        output_indices: wp.array[wp.uint32],
        kp: wp.array[float],
        ki: wp.array[float],
        kd: wp.array[float],
        max_force: wp.array[float],
        integral_max: wp.array[float],
        constant_force: wp.array[float] | None = None,
        state_pos_attr: str = "joint_q",
        state_vel_attr: str = "joint_qd",
        control_target_pos_attr: str = "joint_target_pos",
        control_target_vel_attr: str = "joint_target_vel",
        control_input_attr: str = "joint_act",
        control_output_attr: str = "joint_f",
    ):
        """Initialize PID actuator.

        Args:
            input_indices: DOF indices for reading state and targets. Shape ``(N,)``.
            output_indices: DOF indices for writing output. Shape ``(N,)``.
            kp: Proportional gains. Shape ``(N,)``.
            ki: Integral gains. Shape ``(N,)``.
            kd: Derivative gains. Shape ``(N,)``.
            max_force: Force limits [N or N·m]. Shape ``(N,)``.
            integral_max: Anti-windup limits. Shape ``(N,)``.
            constant_force: Constant offsets [N or N·m]. Shape ``(N,)``. ``None`` to skip.
            state_pos_attr: Attribute on :class:`~newton.State` for positions.
            state_vel_attr: Attribute on :class:`~newton.State` for velocities.
            control_target_pos_attr: Attribute on :class:`~newton.Control` for target positions.
            control_target_vel_attr: Attribute on :class:`~newton.Control` for target velocities.
            control_input_attr: Attribute on :class:`~newton.Control` for control input. ``None`` to skip.
            control_output_attr: Attribute on :class:`~newton.Control` for output forces.
        """
        super().__init__(input_indices, output_indices, control_output_attr)

        for name, arr in [
            ("kp", kp),
            ("ki", ki),
            ("kd", kd),
            ("max_force", max_force),
            ("integral_max", integral_max),
        ]:
            if len(arr) != self.num_actuators:
                raise ValueError(f"{name} length ({len(arr)}) must match num_actuators ({self.num_actuators})")

        if constant_force is not None and len(constant_force) != self.num_actuators:
            raise ValueError(
                f"constant_force length ({len(constant_force)}) must match num_actuators ({self.num_actuators})"
            )

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_force = max_force
        self.integral_max = integral_max
        self.constant_force = constant_force

        self.state_pos_attr = state_pos_attr
        self.state_vel_attr = state_vel_attr
        self.control_target_pos_attr = control_target_pos_attr
        self.control_target_vel_attr = control_target_vel_attr
        self.control_input_attr = control_input_attr

    def _run_controller(
        self,
        sim_state: State,
        sim_control: Control,
        controller_output: wp.array[float],
        output_indices: wp.array[wp.uint32],
        current_state: ActuatorPID.State,
        dt: float,
    ) -> None:
        """Compute PID control forces."""
        control_input = None
        if self.control_input_attr is not None:
            control_input = getattr(sim_control, self.control_input_attr, None)

        wp.launch(
            kernel=pid_controller_kernel,
            dim=self.num_actuators,
            inputs=[
                getattr(sim_state, self.state_pos_attr),
                getattr(sim_state, self.state_vel_attr),
                getattr(sim_control, self.control_target_pos_attr),
                getattr(sim_control, self.control_target_vel_attr),
                control_input,
                self.input_indices,
                self.input_indices,
                output_indices,
                self.kp,
                self.ki,
                self.kd,
                self.max_force,
                self.integral_max,
                self.constant_force,
                dt,
                current_state.integral,
            ],
            outputs=[controller_output],
        )

    def _run_state_manager(
        self,
        sim_state: State,
        sim_control: Control,
        current_state: ActuatorPID.State,
        next_state: ActuatorPID.State,
        dt: float,
    ) -> None:
        """Update integral state."""
        wp.launch(
            kernel=pid_integral_state_kernel,
            dim=self.num_actuators,
            inputs=[
                getattr(sim_state, self.state_pos_attr),
                getattr(sim_control, self.control_target_pos_attr),
                self.input_indices,
                self.input_indices,
                self.integral_max,
                dt,
                current_state.integral,
            ],
            outputs=[next_state.integral],
        )

    def state(self) -> ActuatorPID.State:
        """Return a new state with zero-initialized integral."""
        device = self.input_indices.device
        return ActuatorPID.State(
            integral=wp.zeros(self.num_actuators, dtype=wp.float32, device=device),
        )
