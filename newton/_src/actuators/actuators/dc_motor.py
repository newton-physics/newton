# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import warp as wp

from ..kernels import pd_controller_kernel
from .base import Actuator

if TYPE_CHECKING:
    from ...sim.control import Control
    from ...sim.state import State


class ActuatorDCMotor(Actuator):
    """DC motor actuator with velocity-dependent torque saturation.

    Uses the same PD control law as :class:`ActuatorPD`, but clips torques using
    the DC motor torque-speed characteristic instead of a fixed box limit::

        τ_max(v) = clamp(τ_sat·(1 - v/v_max),  0,  effort_limit)
        τ_min(v) = clamp(τ_sat·(-1 - v/v_max), -effort_limit, 0)
        τ_applied = clamp(τ_computed, τ_min(v), τ_max(v))

    At zero velocity the motor can produce up to ±τ_sat (capped by effort_limit).
    As velocity approaches v_max, available torque in the direction of motion drops to zero.
    Beyond v_max, no torque can be produced in the direction of motion (back-EMF).

    Stateless: no internal memory.
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve arguments with defaults.

        Args:
            args: User-provided arguments.

        Returns:
            Arguments with defaults.

        Raises:
            ValueError: If ``velocity_limit`` not provided.
        """
        if "velocity_limit" not in args:
            raise ValueError("ActuatorDCMotor requires 'velocity_limit' argument")
        return {
            "kp": args.get("kp", 0.0),
            "kd": args.get("kd", 0.0),
            "max_force": args.get("max_force", math.inf),
            "saturation_effort": args.get("saturation_effort", math.inf),
            "velocity_limit": args["velocity_limit"],
            "constant_force": args.get("constant_force", 0.0),
        }

    def __init__(
        self,
        input_indices: wp.array[wp.uint32],
        output_indices: wp.array[wp.uint32],
        kp: wp.array[float],
        kd: wp.array[float],
        max_force: wp.array[float],
        saturation_effort: wp.array[float],
        velocity_limit: wp.array[float],
        constant_force: wp.array[float] | None = None,
        state_pos_attr: str = "joint_q",
        state_vel_attr: str = "joint_qd",
        control_target_pos_attr: str = "joint_target_pos",
        control_target_vel_attr: str = "joint_target_vel",
        control_input_attr: str = "joint_act",
        control_output_attr: str = "joint_f",
    ):
        """Initialize DC motor actuator.

        Args:
            input_indices: DOF indices for reading state and targets. Shape ``(N,)``.
            output_indices: DOF indices for writing output. Shape ``(N,)``.
            kp: Proportional gains. Shape ``(N,)``.
            kd: Derivative gains. Shape ``(N,)``.
            max_force: Absolute effort limits (continuous-rated) [N or N·m]. Shape ``(N,)``.
            saturation_effort: Peak motor torque at stall [N or N·m]. Shape ``(N,)``.
            velocity_limit: Maximum joint velocity for torque-speed curve [rad/s or m/s]. Shape ``(N,)``.
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
            ("kd", kd),
            ("max_force", max_force),
            ("saturation_effort", saturation_effort),
            ("velocity_limit", velocity_limit),
        ]:
            if len(arr) != self.num_actuators:
                raise ValueError(f"{name} length ({len(arr)}) must match num_actuators ({self.num_actuators})")

        if constant_force is not None and len(constant_force) != self.num_actuators:
            raise ValueError(
                f"constant_force length ({len(constant_force)}) must match num_actuators ({self.num_actuators})"
            )

        self.kp = kp
        self.kd = kd
        self.max_force = max_force
        self.saturation_effort = saturation_effort
        self.velocity_limit = velocity_limit
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
        current_state: Any,
        dt: float,
    ) -> None:
        """Compute DC motor PD control forces with velocity-dependent saturation."""
        control_input = None
        if self.control_input_attr is not None:
            control_input = getattr(sim_control, self.control_input_attr, None)

        wp.launch(
            kernel=pd_controller_kernel,
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
                self.kd,
                self.max_force,
                self.constant_force,
                self.saturation_effort,
                self.velocity_limit,
                None,  # lookup_angles (remotized)
                None,  # lookup_torques (remotized)
                0,  # lookup_size (remotized)
            ],
            outputs=[controller_output],
        )
