# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import warp as wp


class InterfaceBase:
    """Base class for actuator interfaces.

    An interface models how the commands reach the actuator, e.g. a
    communication delay. An :class:`~newton.actuators.Actuator` has at most
    one interface.

    Subclasses must override :meth:`resolve_arguments` and :meth:`compute`.
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

    def compute(
        self,
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float] | None,
        target_pos_indices: wp.array[wp.uint32],
        target_vel_indices: wp.array[wp.uint32],
        state: InterfaceBase.State | None,
        dt: float | None = None,
    ) -> tuple[wp.array[float], wp.array[float], wp.array[float]]:
        """Return the transformed targets of each actuator.

        Actuator ``i`` reads ``target_pos[target_pos_indices[i]]``,
        ``target_vel[target_vel_indices[i]]`` and
        ``feedforward[target_vel_indices[i]]``.

        Args:
            target_pos: Target positions [m or rad].
            target_vel: Target velocities [m/s or rad/s].
            feedforward: Feedforward effort [N or N·m], or ``None``.
            target_pos_indices: Index of each actuator into *target_pos*.
            target_vel_indices: Index of each actuator into *target_vel* and
                *feedforward*.
            state: Interface state (``None`` if stateless).
            dt: Timestep [s].

        Returns:
            Target positions, target velocities and feedforward, one entry
            per actuator, shape ``(N,)``; the feedforward is zero when
            *feedforward* is ``None``.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement compute")

    def is_stateful(self) -> bool:
        """Return True if this interface maintains internal state."""
        return False

    def is_graphable(self) -> bool:
        """Return True if :meth:`compute` and :meth:`update_state` can be captured in a CUDA graph."""
        return True

    def state(self, num_actuators: int, device: wp.Device) -> InterfaceBase.State | None:
        """Create and return a new state object, or None if stateless."""
        return None

    def update_state(
        self,
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float] | None,
        target_pos_indices: wp.array[wp.uint32],
        target_vel_indices: wp.array[wp.uint32],
        current_state: InterfaceBase.State | None,
        next_state: InterfaceBase.State | None,
    ) -> None:
        """Advance internal state after the actuator step.

        Receives the same targets as :meth:`compute`.

        Args:
            target_pos: Target positions [m or rad].
            target_vel: Target velocities [m/s or rad/s].
            feedforward: Feedforward effort [N or N·m], or ``None``.
            target_pos_indices: Index of each actuator into *target_pos*.
            target_vel_indices: Index of each actuator into *target_vel* and
                *feedforward*.
            current_state: Current interface state.
            next_state: Next interface state to write.
        """
