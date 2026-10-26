# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import warp as wp


class BatteryBase:
    """Base class for actuator batteries.

    A battery powers one or more actuators.  One object holds several
    batteries of the same type: its parameter arrays have one entry per
    battery.  :meth:`step` and :meth:`reset` read and write arrays on
    ``sim_control`` with one entry per actuator, indexed like
    :attr:`Actuator.actuator_indices <newton.actuators.Actuator.actuator_indices>`.

    Subclasses must override :meth:`resolve_arguments`, :meth:`step` and
    :meth:`reset`.

    **Validation contract:**  :meth:`resolve_arguments` validates scalar
    parameter values before they are batched into Warp arrays.
    ``__init__`` receives pre-built arrays and validates shapes only —
    reading back array contents would force a synchronous device-to-host
    copy on every construction.
    """

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
        actuator_start: wp.array[wp.int32],
        actuators: wp.array[wp.int32],
        world: wp.array[wp.int32] | None = None,
        world_count: int = 0,
    ):
        """Initialize the batteries.

        Args:
            actuator_start: Start of each battery's actuators in *actuators*.
                Shape ``(B + 1,)``.
            actuators: Actuators of all batteries, sorted by battery, so that
                battery ``b`` powers
                ``actuators[actuator_start[b]:actuator_start[b + 1]]``.
            world: World of each battery, ``-1`` for none. Shape ``(B,)``.
                ``None`` places every battery outside any world.
            world_count: Number of worlds, which sets the shape of the mask in
                :meth:`reset`.

        Raises:
            ValueError: *world* does not have one entry per battery.
        """
        self.num_batteries = len(actuator_start) - 1
        """Number of batteries B."""
        self.device = actuator_start.device
        self.actuator_start = actuator_start
        """Start of each battery's actuators in :attr:`actuators`, shape (B + 1,)."""
        self.actuators = actuators
        """Actuators of all batteries, sorted by battery."""
        self.world = world if world is not None else wp.full(self.num_batteries, -1, dtype=wp.int32, device=self.device)
        """World of each battery, ``-1`` for none, shape (B,)."""
        self.world_count = world_count
        """Number of worlds."""
        if self.world.shape != (self.num_batteries,):
            raise ValueError(f"world shape {self.world.shape} must be ({self.num_batteries},)")

    def step(self, sim_control: Any) -> None:
        """Advance the batteries by one step.

        Call once per step, after the actuators have stepped.

        Args:
            sim_control: Object carrying the arrays the battery reads and
                writes, e.g. :class:`~newton.Control`.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement step")

    def reset(self, sim_control: Any, *, world_mask: wp.array[wp.bool] | None = None) -> None:
        """Reset the batteries.

        Call after resetting worlds.

        Args:
            sim_control: Object carrying the arrays the battery writes, e.g.
                :class:`~newton.Control`.
            world_mask: Optional boolean mask of shape ``(world_count + 1,)``
                specifying which worlds to reset. Entries before the last select
                local worlds by index, and the final entry selects batteries
                whose world is ``-1``. If ``None``, all batteries are reset.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement reset")
