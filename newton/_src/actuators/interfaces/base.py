# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import warp as wp


class InterfaceBase:
    """Base class for actuator interfaces.

    An interface models how the commands reach the actuator, e.g. a
    communication delay. It transforms only the commands (target positions,
    target velocities, feedforward) before the drive; the drive reads the
    actuator position and velocity from the sensor and the transmission. An
    :class:`~newton.actuators.Actuator` has at most one interface.

    Subclasses must override :meth:`resolve_arguments` and :meth:`process`.
    Stateful subclasses also override :meth:`is_stateful`, :meth:`state` and
    :meth:`update_state`.
    """

    @dataclass
    class State:
        """Base state for interfaces."""

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            """Reset state to initial values.

            Args:
                mask: Boolean mask of length N. ``True`` entries are reset.
                    ``None`` resets all.
            """

    @dataclass
    class Targets:
        """Commands passed through the interface.

        Each actuator slot ``i`` reads ``target_pos[target_pos_indices[i]]``,
        ``target_vel[target_vel_indices[i]]`` and
        ``feedforward[target_vel_indices[i]]``. An interface that replaces a
        target array with one entry per slot also replaces its index array.
        """

        target_pos: wp.array[float]
        """Target positions [m or rad]."""
        target_vel: wp.array[float]
        """Target velocities [m/s or rad/s]."""
        feedforward: wp.array[float] | None
        """Feedforward effort [N or N·m], or ``None``."""
        target_pos_indices: wp.array[wp.uint32]
        """Index of each slot into :attr:`target_pos`."""
        target_vel_indices: wp.array[wp.uint32]
        """Index of each slot into :attr:`target_vel` and :attr:`feedforward`."""

    SHARED_PARAMS: ClassVar[set[str]] = set()

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
            num_actuators: Number of actuators.
            requires_grad: Allocate output arrays with gradient support.
        """

    def process(
        self, targets: InterfaceBase.Targets, state: InterfaceBase.State | None, dt: float | None
    ) -> InterfaceBase.Targets:
        """Return the transformed commands.

        Return a copy of *targets* made with :func:`dataclasses.replace`; do
        not modify *targets* in place.

        Args:
            targets: Commands of the actuators.
            state: Interface state (``None`` if stateless).
            dt: Timestep [s].

        Returns:
            Transformed commands.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement process")

    def is_stateful(self) -> bool:
        """Return True if this interface maintains internal state."""
        return False

    def is_graphable(self) -> bool:
        """Return True if :meth:`process` and :meth:`update_state` can be captured in a CUDA graph."""
        return True

    def state(self, num_actuators: int, device: wp.Device) -> InterfaceBase.State | None:
        """Create and return a new state object, or None if stateless."""
        return None

    def update_state(
        self,
        targets: InterfaceBase.Targets,
        current_state: InterfaceBase.State | None,
        next_state: InterfaceBase.State | None,
    ) -> None:
        """Advance internal state after the actuator step.

        Args:
            targets: The commands this interface received in :meth:`process`.
            current_state: Current interface state.
            next_state: Next interface state to write.
        """
