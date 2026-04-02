# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from ...sim.control import Control
    from ...sim.state import State


class Actuator:
    """Base class for actuators.

    Subclasses override controller, state manager, and transmission methods.
    The :meth:`step` method executes up to three phases:

    1. **Controller** – computes forces from state error.
    2. **State manager** – updates internal state (e.g., delay buffers).
    3. **Transmission** – maps actuation forces to output (e.g., via Jacobian).

    Class Attributes:
        SCALAR_PARAMS: Set of parameter names that are instance-level (shared across
            all DOFs). These cannot vary per-DOF, so different values require
            separate actuator instances (e.g., ``delay``, ``network``).
    """

    SCALAR_PARAMS: set[str] = set()

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments with defaults.

        Args:
            args: User-provided arguments.

        Returns:
            Complete arguments with defaults filled in.
        """
        raise NotImplementedError(f"{cls.__name__} must implement resolve_arguments")

    def __init__(
        self,
        input_indices: wp.array[wp.uint32],
        output_indices: wp.array[wp.uint32],
        control_output_attr: str = "joint_f",
    ):
        """Initialize actuator.

        Args:
            input_indices: DOF indices for reading state and targets.
                Shape ``(N,)`` for single input per actuator, ``(N, M)`` for
                multiple inputs per actuator.
            output_indices: DOF indices for writing output forces.
                Shape ``(N,)`` for single output per actuator, ``(N, K)`` for
                multiple outputs per actuator.
            control_output_attr: Attribute name on :class:`~newton.Control` for output.
        """
        self.input_indices = input_indices
        self.output_indices = output_indices
        self.control_output_attr = control_output_attr
        self.num_actuators = len(input_indices)

        if len(output_indices) != self.num_actuators:
            raise ValueError(
                f"output_indices length ({len(output_indices)}) must match input_indices length ({self.num_actuators})"
            )

        device = input_indices.device
        self._sequential_indices = wp.array(np.arange(self.num_actuators, dtype=np.uint32), device=device)
        self._actuation_forces = None

    def is_stateful(self) -> bool:
        """Return True if this actuator maintains internal state.

        Users should check this to determine if they need to call the :meth:`state`
        method and manage double-buffered state objects.
        """
        return False

    def is_graphable(self) -> bool:
        """Return True if this actuator's :meth:`step` can be captured in a CUDA graph."""
        return True

    def has_transmission(self) -> bool:
        """Return True if this actuator has a transmission phase."""
        return False

    def _run_controller(
        self,
        sim_state: State,
        sim_control: Control,
        controller_output: wp.array[float],
        output_indices: wp.array[wp.uint32],
        current_state: Any,
        dt: float,
    ) -> None:
        """Compute control forces. Override in subclasses.

        Args:
            sim_state: Simulation state.
            sim_control: Control structure.
            controller_output: Array to write forces to.
            output_indices: Indices into *controller_output*.
            current_state: Current actuator state (``None`` if stateless).
            dt: Time step [s].
        """
        pass

    def _run_state_manager(
        self,
        sim_state: State,
        sim_control: Control,
        current_state: Any,
        next_state: Any,
        dt: float,
    ) -> None:
        """Update internal state. Override in stateful subclasses.

        Args:
            sim_state: Simulation state.
            sim_control: Control structure.
            current_state: Current actuator state.
            next_state: Next actuator state to write.
            dt: Time step [s].
        """
        pass

    def _run_transmission(
        self,
        actuation_forces: wp.array[float],
        sim_control: Control,
    ) -> None:
        """Map actuation forces to output. Override in subclasses with transmission.

        Args:
            actuation_forces: Forces from controller.
            sim_control: Control structure.
        """
        pass

    def step(
        self,
        sim_state: State,
        sim_control: Control,
        current_act_state: Any = None,
        next_act_state: Any = None,
        dt: float = None,
    ) -> None:
        """Execute one control step.

        Args:
            sim_state: Simulation state.
            sim_control: Control structure.
            current_act_state: Current internal state (``None`` if stateless).
            next_act_state: Next internal state (``None`` if stateless).
            dt: Time step [s].
        """
        if self.has_transmission():
            controller_output = self._actuation_forces
            controller_output_indices = self._sequential_indices
        else:
            controller_output = getattr(sim_control, self.control_output_attr)
            controller_output_indices = self.output_indices

        self._run_controller(
            sim_state, sim_control, controller_output, controller_output_indices, current_act_state, dt
        )

        if self.is_stateful():
            self._run_state_manager(sim_state, sim_control, current_act_state, next_act_state, dt)

        if self.has_transmission():
            self._run_transmission(self._actuation_forces, sim_control)

    def state(self) -> Any:
        """Return a new instance of this actuator's internal state.

        Returns:
            State object, or ``None`` if stateless.
        """
        return None
