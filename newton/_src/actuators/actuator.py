# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from .clamping.base import Clamping
from .controllers.base import Controller
from .delay import Delay


# TODO: replace with a Transmission class that does J multiplication before accumulating into the output array.
@wp.kernel
def _scatter_add_kernel(
    forces: wp.array[float],
    computed_forces: wp.array[float],
    indices: wp.array[wp.uint32],
    output: wp.array[float],
    computed_output: wp.array[float],
):
    """Scatter-add forces into output; optionally scatter computed forces too."""
    i = wp.tid()
    idx = indices[i]
    output[idx] = output[idx] + forces[i]
    if computed_output:
        computed_output[idx] = computed_output[idx] + computed_forces[i]


class Actuator:
    """Composed actuator: delay → controller → clamping.

    An actuator reads from simulation state/control arrays, optionally
    delays the control targets, computes forces via a controller, applies
    clamping (force limits, saturation, etc.), and writes the result to
    the output array.

    Usage::

        actuator = Actuator(
            indices=indices,
            controller=ControllerPD(kp=kp, kd=kd),
            delay=Delay(delay=wp.array([5, 5], dtype=wp.int32), max_delay=5),
            clamping=[ClampingMaxForce(max_force=max_f)],
        )

        # Simulation loop
        actuator.step(sim_state, sim_control, state_a, state_b, dt=0.01)
    """

    @dataclass
    class State:
        """Composed state for an :class:`Actuator`.

        Holds the delay state (if a delay is present) and the controller
        state. Clamping objects are stateless.
        """

        delay_state: Delay.State | None = None
        """Delay buffer state, or ``None`` if no delay is used."""
        controller_state: Controller.State | None = None
        """Controller-specific state, or ``None`` if stateless."""

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            """Reset composed state.

            Args:
                mask: Boolean mask of length N. ``True`` entries are reset.
                    ``None`` resets all.
            """
            if self.delay_state is not None:
                self.delay_state.reset(mask)
            if self.controller_state is not None:
                self.controller_state.reset(mask)

    def __init__(
        self,
        indices: wp.array[wp.uint32],
        controller: Controller,
        delay: Delay | None = None,
        clamping: list[Clamping] | None = None,
        pos_indices: wp.array[wp.uint32] | None = None,
        frc_indices: wp.array[wp.uint32] | None = None,
        state_pos_attr: str = "joint_q",
        state_vel_attr: str = "joint_qd",
        control_target_pos_attr: str = "joint_target_pos",
        control_target_vel_attr: str = "joint_target_vel",
        control_feedforward_attr: str | None = "joint_act",
        control_output_attr: str = "joint_f",
        control_computed_output_attr: str | None = None,
        requires_grad: bool = False,
    ):
        """Initialize actuator.

        Args:
            indices: DOF indices into ``joint_qd``-shaped arrays (velocities,
                velocity targets, feedforward). Shape ``(N,)``.
            controller: Controller that computes raw forces.
            delay: Optional Delay instance for input delay.
            clamping: List of Clamping objects (post-controller force bounds).
            pos_indices: DOF indices into ``joint_q``-shaped arrays (positions,
                position targets). Defaults to *indices*. Differs from
                *indices* for floating-base or ball-joint articulations
                where ``joint_q`` and ``joint_qd`` have different layouts.
            frc_indices: DOF indices into ``joint_f``-shaped arrays (output
                forces). Defaults to *indices*. Differs from *indices*
                for coupled transmissions or tendon-driven joints.
            state_pos_attr: Attribute on sim_state for positions.
            state_vel_attr: Attribute on sim_state for velocities.
            control_target_pos_attr: Attribute on sim_control for target positions.
            control_target_vel_attr: Attribute on sim_control for target velocities.
            control_feedforward_attr: Attribute on sim_control for control input. None to skip.
            control_output_attr: Attribute on sim_control for clamped output forces.
            control_computed_output_attr: Attribute on sim_control for raw (pre-clamp)
                forces. None to skip writing computed forces.
            requires_grad: Allocate intermediate arrays with gradient support
                for differentiable simulation.
        """
        self.indices = indices
        self.pos_indices = pos_indices if pos_indices is not None else indices
        self.frc_indices = frc_indices if frc_indices is not None else indices
        self.controller = controller
        self.delay = delay
        self.clamping = clamping or []
        self.num_actuators = len(indices)

        self.state_pos_attr = state_pos_attr
        self.state_vel_attr = state_vel_attr
        self.control_target_pos_attr = control_target_pos_attr
        self.control_target_vel_attr = control_target_vel_attr
        self.control_feedforward_attr = control_feedforward_attr
        self.control_output_attr = control_output_attr
        self.control_computed_output_attr = control_computed_output_attr

        self.device = indices.device
        self.requires_grad = requires_grad
        self._sequential_indices = wp.array(np.arange(self.num_actuators, dtype=np.uint32), device=self.device)
        self._computed_forces = wp.zeros(
            self.num_actuators, dtype=wp.float32, device=self.device, requires_grad=requires_grad
        )
        self._applied_forces = (
            wp.zeros(self.num_actuators, dtype=wp.float32, device=self.device, requires_grad=requires_grad)
            if self.clamping
            else None
        )

        controller.finalize(self.device, self.num_actuators)
        if delay is not None:
            delay.finalize(self.device, self.num_actuators, requires_grad=requires_grad)
        for clamp in self.clamping:
            clamp.finalize(self.device, self.num_actuators)

    @property
    def SHARED_PARAMS(self) -> set[str]:
        params: set[str] = set()
        params |= self.controller.SHARED_PARAMS
        for c in self.clamping:
            params |= c.SHARED_PARAMS
        return params

    def is_stateful(self) -> bool:
        """Return True if delay or controller maintains internal state."""
        return self.delay is not None or self.controller.is_stateful()

    def is_graphable(self) -> bool:
        """Return True if all components can be captured in a CUDA graph."""
        return self.controller.is_graphable()

    def state(self) -> Actuator.State | None:
        """Return a new composed state, or None if fully stateless."""
        if not self.is_stateful():
            return None
        return Actuator.State(
            delay_state=(self.delay.state(self.num_actuators, self.device) if self.delay is not None else None),
            controller_state=(
                self.controller.state(self.num_actuators, self.device) if self.controller.is_stateful() else None
            ),
        )

    def step(
        self,
        sim_state: Any,
        sim_control: Any,
        current_act_state: Actuator.State | None = None,
        next_act_state: Actuator.State | None = None,
        dt: float | None = None,
    ) -> None:
        """Execute one control step.

        1. **Delay read** — read per-DOF delayed targets from
           ``current_state`` (falls back to current targets when
           the buffer is empty).
        2. **Controller** — compute raw forces into ``_computed_forces``.
        3. **Clamping** — clamp forces from computed → ``_applied_forces``.
        4. **Scatter** — add applied (and optionally computed) forces to output.
        5. **State updates** — controller state update, then delay
           buffer write (push current targets into ``next_state``).

        Args:
            sim_state: Simulation state with position/velocity arrays.
            sim_control: Control structure with target/output arrays.
            current_act_state: Current composed state (None if stateless).
            next_act_state: Next composed state (None if stateless).
            dt: Timestep [s].
        """
        if self.is_stateful() and (current_act_state is None or next_act_state is None):
            raise ValueError(
                "Stateful actuator requires both current_act_state and next_act_state; create them via actuator.state()"
            )

        positions = getattr(sim_state, self.state_pos_attr)
        velocities = getattr(sim_state, self.state_vel_attr)

        orig_target_pos = getattr(sim_control, self.control_target_pos_attr)
        orig_target_vel = getattr(sim_control, self.control_target_vel_attr)
        orig_feedforward = None
        if self.control_feedforward_attr is not None:
            orig_feedforward = getattr(sim_control, self.control_feedforward_attr, None)

        target_pos = orig_target_pos
        target_vel = orig_target_vel
        feedforward = orig_feedforward
        target_pos_indices = self.pos_indices
        target_vel_indices = self.indices

        # --- 1. Delay read (from current_state) ---
        if self.delay is not None:
            target_pos, target_vel, feedforward = self.delay.get_delayed_targets(
                orig_target_pos,
                orig_target_vel,
                orig_feedforward,
                self.pos_indices,
                self.indices,
                current_act_state.delay_state,
            )
            target_pos_indices = self._sequential_indices
            target_vel_indices = self._sequential_indices

        # --- 2. Controller: compute raw forces ---
        ctrl_state = current_act_state.controller_state if current_act_state else None
        self.controller.compute(
            positions,
            velocities,
            target_pos,
            target_vel,
            feedforward,
            self.pos_indices,
            self.indices,
            target_pos_indices,
            target_vel_indices,
            self._computed_forces,
            ctrl_state,
            dt,
            device=self.device,
        )

        # --- 3. Clamping: computed → applied (fused copy+clamp) ---
        if self.clamping:
            src = self._computed_forces
            for clamp in self.clamping:
                clamp.modify_forces(
                    src,
                    self._applied_forces,
                    positions,
                    velocities,
                    self.pos_indices,
                    self.indices,
                    device=self.device,
                )
                src = self._applied_forces
            output_forces = self._applied_forces
        else:
            output_forces = self._computed_forces

        # --- 4. Scatter-add to output ---
        applied_output = getattr(sim_control, self.control_output_attr)
        computed_output = (
            getattr(sim_control, self.control_computed_output_attr)
            if self.control_computed_output_attr is not None
            else None
        )
        wp.launch(
            kernel=_scatter_add_kernel,
            dim=self.num_actuators,
            inputs=[output_forces, self._computed_forces, self.frc_indices],
            outputs=[applied_output, computed_output],
            device=self.device,
        )

        # --- 5. State updates (write to next_state) ---
        if self.controller.is_stateful():
            self.controller.update_state(
                current_act_state.controller_state,
                next_act_state.controller_state,
            )
        if self.delay is not None:
            self.delay.update_state(
                orig_target_pos,
                orig_target_vel,
                orig_feedforward,
                self.pos_indices,
                self.indices,
                current_act_state.delay_state,
                next_act_state.delay_state,
            )
