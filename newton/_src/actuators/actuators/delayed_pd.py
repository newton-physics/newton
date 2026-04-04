# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import warp as wp

from ..kernels import delay_buffer_state_kernel, pd_controller_kernel
from .base import Actuator

if TYPE_CHECKING:
    from ...sim.control import Control
    from ...sim.state import State


class ActuatorDelayedPD(Actuator):
    """PD controller with input delay.

    Control law::

        τ = clamp(constant + act_delayed + Kp·(target_pos_delayed - q)
                  + Kd·(target_vel_delayed - v), ±max_force)

    Stateful: delays targets by N timesteps using circular buffer to model
    actuator lag.
    """

    SCALAR_PARAMS: ClassVar[set[str]] = {"delay"}

    @dataclass
    class State:
        """Circular buffer state for delayed actuators."""

        buffer_pos: wp.array2d[float] | None = None  # Shape (delay, N)
        buffer_vel: wp.array2d[float] | None = None  # Shape (delay, N)
        buffer_act: wp.array2d[float] | None = None  # Shape (delay, N)
        write_idx: int = 0  # Last write position
        is_filled: bool = False  # Buffer filled at least once

        def reset(self) -> None:
            """Zero buffers and reset in-place."""
            self.buffer_pos.zero_()
            self.buffer_vel.zero_()
            self.buffer_act.zero_()
            self.write_idx = self.buffer_pos.shape[0] - 1
            self.is_filled = False

    def is_stateful(self) -> bool:
        return True

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve arguments with defaults. Requires ``delay``.

        Args:
            args: User-provided arguments.

        Returns:
            Arguments with defaults.

        Raises:
            ValueError: If ``delay`` not provided.
        """
        if "delay" not in args:
            raise ValueError("ActuatorDelayedPD requires 'delay' argument")
        return {
            "kp": args.get("kp", 0.0),
            "kd": args.get("kd", 0.0),
            "delay": args["delay"],
            "max_force": args.get("max_force", math.inf),
            "constant_force": args.get("constant_force", 0.0),
        }

    def __init__(
        self,
        input_indices: wp.array[wp.uint32],
        output_indices: wp.array[wp.uint32],
        kp: wp.array[float],
        kd: wp.array[float],
        delay: int,
        max_force: wp.array[float],
        constant_force: wp.array[float] | None = None,
        state_pos_attr: str = "joint_q",
        state_vel_attr: str = "joint_qd",
        control_target_pos_attr: str = "joint_target_pos",
        control_target_vel_attr: str = "joint_target_vel",
        control_input_attr: str = "joint_act",
        control_output_attr: str = "joint_f",
    ):
        """Initialize delayed PD actuator.

        Args:
            input_indices: DOF indices for reading state and targets. Shape ``(N,)``.
            output_indices: DOF indices for writing output. Shape ``(N,)``.
            kp: Proportional gains. Shape ``(N,)``.
            kd: Derivative gains. Shape ``(N,)``.
            delay: Number of timesteps to delay inputs.
            max_force: Force limits [N or N·m]. Shape ``(N,)``.
            constant_force: Constant offsets [N or N·m]. Shape ``(N,)``. ``None`` to skip.
            state_pos_attr: Attribute on :class:`~newton.State` for positions.
            state_vel_attr: Attribute on :class:`~newton.State` for velocities.
            control_target_pos_attr: Attribute on :class:`~newton.Control` for target positions.
            control_target_vel_attr: Attribute on :class:`~newton.Control` for target velocities.
            control_input_attr: Attribute on :class:`~newton.Control` for control input. ``None`` to skip.
            control_output_attr: Attribute on :class:`~newton.Control` for output forces.
        """
        super().__init__(input_indices, output_indices, control_output_attr)

        for name, arr in [("kp", kp), ("kd", kd), ("max_force", max_force)]:
            if len(arr) != self.num_actuators:
                raise ValueError(f"{name} length ({len(arr)}) must match num_actuators ({self.num_actuators})")

        if constant_force is not None and len(constant_force) != self.num_actuators:
            raise ValueError(
                f"constant_force length ({len(constant_force)}) must match num_actuators ({self.num_actuators})"
            )

        self.kp = kp
        self.kd = kd
        self.max_force = max_force
        self.constant_force = constant_force
        self.delay = delay

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
        current_state: ActuatorDelayedPD.State,
        dt: float,
    ) -> None:
        """Compute delayed PD control forces."""
        if current_state is None or not current_state.is_filled:
            return

        read_idx = (current_state.write_idx + 1) % self.delay
        delayed_pos = current_state.buffer_pos[read_idx]
        delayed_vel = current_state.buffer_vel[read_idx]

        delayed_act = None
        if self.control_input_attr is not None:
            delayed_act = current_state.buffer_act[read_idx]

        wp.launch(
            kernel=pd_controller_kernel,
            dim=self.num_actuators,
            inputs=[
                getattr(sim_state, self.state_pos_attr),
                getattr(sim_state, self.state_vel_attr),
                delayed_pos,
                delayed_vel,
                delayed_act,
                self.input_indices,
                self._sequential_indices,
                output_indices,
                self.kp,
                self.kd,
                self.max_force,
                self.constant_force,
                None,  # saturation_effort (DC motor)
                None,  # velocity_limit (DC motor)
                None,  # lookup_angles (remotized)
                None,  # lookup_torques (remotized)
                0,  # lookup_size (remotized)
            ],
            outputs=[controller_output],
        )

    def _run_state_manager(
        self,
        sim_state: State,
        sim_control: Control,
        current_state: ActuatorDelayedPD.State,
        next_state: ActuatorDelayedPD.State,
        dt: float,
    ) -> None:
        """Update circular delay buffer."""
        if next_state is None:
            return

        copy_idx = current_state.write_idx
        write_idx = (current_state.write_idx + 1) % self.delay

        control_input = None
        if self.control_input_attr is not None:
            control_input = getattr(sim_control, self.control_input_attr, None)

        wp.launch(
            kernel=delay_buffer_state_kernel,
            dim=self.num_actuators,
            inputs=[
                getattr(sim_control, self.control_target_pos_attr),
                getattr(sim_control, self.control_target_vel_attr),
                control_input,
                self.input_indices,
                copy_idx,
                write_idx,
                current_state.buffer_pos,
                current_state.buffer_vel,
                current_state.buffer_act,
            ],
            outputs=[
                next_state.buffer_pos,
                next_state.buffer_vel,
                next_state.buffer_act,
            ],
        )

        next_state.write_idx = write_idx
        next_state.is_filled = current_state.is_filled or (write_idx == self.delay - 1)

    def state(self) -> ActuatorDelayedPD.State:
        """Return a new state with allocated circular buffers."""
        device = self.input_indices.device
        return ActuatorDelayedPD.State(
            buffer_pos=wp.zeros((self.delay, self.num_actuators), dtype=wp.float32, device=device),
            buffer_vel=wp.zeros((self.delay, self.num_actuators), dtype=wp.float32, device=device),
            buffer_act=wp.zeros((self.delay, self.num_actuators), dtype=wp.float32, device=device),
            write_idx=self.delay - 1,
            is_filled=False,
        )
