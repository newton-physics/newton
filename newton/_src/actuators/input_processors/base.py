# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import warp as wp


class InputProcessorBase:
    """Base class for actuator input processors.

    An input processor transforms the inputs of an actuator before they
    reach the drive. The inputs are the simulation state the drive reads
    (positions, velocities) and the commands (target positions, target
    velocities, feedforward). An :class:`~newton.actuators.Actuator` runs its
    processors in list order; each processor receives the inputs returned by
    the previous one.

    Subclasses must override :meth:`resolve_arguments` and :meth:`process`.
    Stateful subclasses also override :meth:`is_stateful`, :meth:`state` and
    :meth:`update_state`.
    """

    @dataclass
    class State:
        """Base state for input processors."""

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            """Reset state to initial values.

            Args:
                mask: Boolean mask of length N. ``True`` entries are reset.
                    ``None`` resets all.
            """

    @dataclass
    class Inputs:
        """Actuator inputs passed through the processor chain.

        Each actuator slot ``i`` reads ``positions[pos_indices[i]]``,
        ``velocities[vel_indices[i]]``, ``target_pos[target_pos_indices[i]]``,
        ``target_vel[target_vel_indices[i]]`` and
        ``feedforward[target_vel_indices[i]]``. A processor that replaces an
        array with one entry per slot also replaces its index array.
        """

        positions: wp.array[float]
        """Positions read by the drive [m or rad]."""
        velocities: wp.array[float]
        """Velocities read by the drive [m/s or rad/s]."""
        target_pos: wp.array[float]
        """Target positions [m or rad]."""
        target_vel: wp.array[float]
        """Target velocities [m/s or rad/s]."""
        feedforward: wp.array[float] | None
        """Feedforward effort [N or N·m], or ``None``."""
        pos_indices: wp.array[wp.uint32]
        """Index of each slot into :attr:`positions`."""
        vel_indices: wp.array[wp.uint32]
        """Index of each slot into :attr:`velocities`."""
        target_pos_indices: wp.array[wp.uint32]
        """Index of each slot into :attr:`target_pos`."""
        target_vel_indices: wp.array[wp.uint32]
        """Index of each slot into :attr:`target_vel` and :attr:`feedforward`."""
        sim_positions: wp.array[float]
        """Unprocessed ``joint_q``-shaped simulation positions [m or rad].
        Index with ``joint_q`` indices to read another joint."""
        sim_velocities: wp.array[float]
        """Unprocessed ``joint_qd``-shaped simulation velocities [m/s or rad/s].
        Index with ``joint_qd`` indices to read another joint."""

    SHARED_PARAMS: ClassVar[set[str]] = set()

    JOINT_PARAMS: ClassVar[set[str]] = set()
    """Names of parameters that reference a joint.

    :meth:`~newton.ModelBuilder.add_actuator` accepts a joint index or a joint
    label for each of these parameters, or ``None`` for the actuated joint.
    It passes the joint to the constructor as two index arrays:
    ``<name>_indices`` into ``joint_qd``-shaped arrays and
    ``<name>_pos_indices`` into ``joint_q``-shaped arrays.
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments with defaults.

        Args:
            args: User-provided arguments.

        Returns:
            Complete arguments with defaults filled in. Return integer
            parameters as ``int`` and real parameters as ``float``:
            :meth:`~newton.ModelBuilder.finalize` stores a parameter as an
            ``int32`` array only when all its values are ``int``.
        """
        raise NotImplementedError(f"{cls.__name__} must implement resolve_arguments")

    def finalize(self, device: wp.Device, num_actuators: int, requires_grad: bool = False) -> None:
        """Called by :class:`~newton.actuators.Actuator` after construction.

        Args:
            device: Warp device to use.
            num_actuators: Number of actuators (DOFs).
            requires_grad: Allocate output arrays with gradient support.
        """

    def process(
        self, inputs: InputProcessorBase.Inputs, state: InputProcessorBase.State | None, dt: float | None
    ) -> InputProcessorBase.Inputs:
        """Return the transformed actuator inputs.

        Return a copy of *inputs* made with :func:`dataclasses.replace`; do
        not modify *inputs* in place. To read a joint other than the
        actuated one, index :attr:`Inputs.sim_positions` and
        :attr:`Inputs.sim_velocities`: earlier processors may have replaced
        :attr:`Inputs.positions` and :attr:`Inputs.velocities`.

        Args:
            inputs: Actuator inputs.
            state: Processor state (``None`` if stateless).
            dt: Timestep [s].

        Returns:
            Transformed actuator inputs.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement process")

    def is_stateful(self) -> bool:
        """Return True if this processor maintains internal state."""
        return False

    def state(self, num_actuators: int, device: wp.Device) -> InputProcessorBase.State | None:
        """Create and return a new state object, or None if stateless."""
        return None

    def update_state(
        self,
        inputs: InputProcessorBase.Inputs,
        current_state: InputProcessorBase.State | None,
        next_state: InputProcessorBase.State | None,
    ) -> None:
        """Advance internal state after the actuator step.

        Args:
            inputs: The inputs this processor received in :meth:`process`.
            current_state: Current processor state.
            next_state: Next processor state to write.
        """
