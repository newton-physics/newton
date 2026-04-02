# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import warp as wp

from ..kernels import pd_controller_kernel
from .delayed_pd import ActuatorDelayedPD

if TYPE_CHECKING:
    from ...sim.control import Control
    from ...sim.state import State


class ActuatorRemotizedPD(ActuatorDelayedPD):
    """PD controller with input delay and angle-dependent torque limits.

    Extends :class:`ActuatorDelayedPD` by replacing the fixed ``max_force`` box
    clamp with angle-dependent torque limits interpolated from a lookup table.
    This models remotized actuators (e.g., linkage-driven joints) where the
    transmission ratio and thus maximum output torque vary with joint angle.

    Control law (with delayed targets)::

        τ = clamp(constant + act_delayed + Kp·(target_pos_delayed - q)
                  + Kd·(target_vel_delayed - v), ±interp(q, lookup))

    Stateful: inherits delay buffers from :class:`ActuatorDelayedPD`.
    """

    SCALAR_PARAMS: ClassVar[set[str]] = {"delay", "lookup_angles", "lookup_torques"}

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve arguments with defaults. Requires ``delay``, ``lookup_angles``, and ``lookup_torques``.

        Args:
            args: User-provided arguments.

        Returns:
            Arguments with defaults.

        Raises:
            ValueError: If required arguments are missing.
        """
        if "delay" not in args:
            raise ValueError("ActuatorRemotizedPD requires 'delay' argument")
        if "lookup_angles" not in args or "lookup_torques" not in args:
            raise ValueError("ActuatorRemotizedPD requires 'lookup_angles' and 'lookup_torques' arguments")
        return {
            "kp": args.get("kp", 0.0),
            "kd": args.get("kd", 0.0),
            "delay": args["delay"],
            "lookup_angles": tuple(args["lookup_angles"]),
            "lookup_torques": tuple(args["lookup_torques"]),
            "constant_force": args.get("constant_force", 0.0),
        }

    def __init__(
        self,
        input_indices: wp.array[wp.uint32],
        output_indices: wp.array[wp.uint32],
        kp: wp.array[float],
        kd: wp.array[float],
        delay: int,
        lookup_angles: wp.array[float] | tuple[float, ...] | list[float],
        lookup_torques: wp.array[float] | tuple[float, ...] | list[float],
        constant_force: wp.array[float] | None = None,
        state_pos_attr: str = "joint_q",
        state_vel_attr: str = "joint_qd",
        control_target_pos_attr: str = "joint_target_pos",
        control_target_vel_attr: str = "joint_target_vel",
        control_input_attr: str = "joint_act",
        control_output_attr: str = "joint_f",
    ):
        """Initialize remotized PD actuator.

        Args:
            input_indices: DOF indices for reading state and targets. Shape ``(N,)``.
            output_indices: DOF indices for writing output. Shape ``(N,)``.
            kp: Proportional gains. Shape ``(N,)``.
            kd: Derivative gains. Shape ``(N,)``.
            delay: Number of timesteps to delay inputs.
            lookup_angles: Sorted joint angles [rad] for the torque lookup table. Shape ``(K,)``.
                Accepts ``wp.array``, ``tuple``, or ``list`` — sequences are converted
                to ``wp.array`` internally.
            lookup_torques: Max output torques [N·m] corresponding to *lookup_angles*.
                Shape ``(K,)``. Accepts ``wp.array``, ``tuple``, or ``list``.
            constant_force: Constant offsets [N or N·m]. Shape ``(N,)``. ``None`` to skip.
            state_pos_attr: Attribute on :class:`~newton.State` for positions.
            state_vel_attr: Attribute on :class:`~newton.State` for velocities.
            control_target_pos_attr: Attribute on :class:`~newton.Control` for target positions.
            control_target_vel_attr: Attribute on :class:`~newton.Control` for target velocities.
            control_input_attr: Attribute on :class:`~newton.Control` for control input. ``None`` to skip.
            control_output_attr: Attribute on :class:`~newton.Control` for output forces.
        """
        max_force_dummy = wp.array(
            np.full(len(input_indices), math.inf, dtype=np.float32),
            device=input_indices.device,
        )
        super().__init__(
            input_indices=input_indices,
            output_indices=output_indices,
            kp=kp,
            kd=kd,
            delay=delay,
            max_force=max_force_dummy,
            constant_force=constant_force,
            state_pos_attr=state_pos_attr,
            state_vel_attr=state_vel_attr,
            control_target_pos_attr=control_target_pos_attr,
            control_target_vel_attr=control_target_vel_attr,
            control_input_attr=control_input_attr,
            control_output_attr=control_output_attr,
        )

        if len(lookup_angles) != len(lookup_torques):
            raise ValueError(
                f"lookup_angles length ({len(lookup_angles)}) must match lookup_torques length ({len(lookup_torques)})"
            )

        device = input_indices.device
        if not isinstance(lookup_angles, wp.array):
            lookup_angles = wp.array(np.array(lookup_angles, dtype=np.float32), device=device)
        if not isinstance(lookup_torques, wp.array):
            lookup_torques = wp.array(np.array(lookup_torques, dtype=np.float32), device=device)

        self.lookup_angles = lookup_angles
        self.lookup_torques = lookup_torques
        self.lookup_size = len(lookup_angles)

    def _run_controller(
        self,
        sim_state: State,
        sim_control: Control,
        controller_output: wp.array[float],
        output_indices: wp.array[wp.uint32],
        current_state: ActuatorDelayedPD.State,
        dt: float,
    ) -> None:
        """Compute delayed PD control forces with angle-dependent torque limits."""
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
                self.lookup_angles,
                self.lookup_torques,
                self.lookup_size,
            ],
            outputs=[controller_output],
        )
