# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import warp as wp

if TYPE_CHECKING:
    from ...sim.builder import ModelBuilder
    from ...sim.model import Model

SUPPLY_VOLTAGE_ATTR = "actuator_supply_voltage"
SUPPLY_CURRENT_GAIN_ATTR = "actuator_supply_current_gain"


def register_battery_attributes(builder: ModelBuilder) -> None:
    """Register the per-actuator supply voltage and supply current gain on Control.

    :meth:`~newton.ModelBuilder.add_battery` calls this with the first battery.

    Args:
        builder: Newton model builder on which to register the attributes.
    """
    from ...sim.model import Model  # noqa: PLC0415

    for name in (SUPPLY_VOLTAGE_ATTR, SUPPLY_CURRENT_GAIN_ATTR):
        builder.add_custom_attribute(
            builder.CustomAttribute(
                name=name,
                dtype=wp.float32,
                frequency=Model.AttributeFrequency.ACTUATOR,
                assignment=Model.AttributeAssignment.CONTROL,
                default=0.0,
            )
        )


class BatteryBase:
    """Base class for batteries shared by several actuators.

    A battery sets the supply voltage of its actuators from the current they
    draw. One object holds several batteries of the same type: its parameter
    arrays have one entry per battery. :meth:`~newton.ModelBuilder.add_battery`
    adds batteries, and :meth:`~newton.ModelBuilder.finalize` creates the
    objects in :attr:`Model.batteries <newton.Model.batteries>`.

    :meth:`step` and :meth:`reset` read and write arrays on ``sim_control``
    with one entry per actuator of the scene, indexed like
    :attr:`Actuator.actuator_indices <newton.actuators.Actuator.actuator_indices>`.
    They write ``actuator_supply_voltage`` for the actuators of the batteries.

    Subclasses must override :meth:`resolve_arguments`,
    :meth:`initial_supply_voltage`, :meth:`step` and :meth:`reset`.

    Args:
        actuator_start: Start of each battery's actuators in *actuators*.
            Shape ``(B + 1,)``.
        actuators: Actuators of all batteries, sorted by battery, so that
            battery ``b`` draws from
            ``actuators[actuator_start[b]:actuator_start[b + 1]]``.
        world: World of each battery, ``-1`` for none. Shape ``(B,)``.
            ``None`` places every battery outside any world.

    Raises:
        ValueError: *world* does not have one entry per battery.
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments of one battery with defaults.

        Args:
            args: User-provided arguments, as passed to
                :meth:`~newton.ModelBuilder.add_battery`.

        Returns:
            Complete arguments with defaults filled in.
        """
        raise NotImplementedError(f"{cls.__name__} must implement resolve_arguments")

    @classmethod
    def initial_supply_voltage(cls, args: dict[str, Any]) -> float:
        """Return the supply voltage [V] of a full battery, at which ``actuator_supply_voltage`` starts.

        Args:
            args: Arguments of one battery, as returned by :meth:`resolve_arguments`.
        """
        raise NotImplementedError(f"{cls.__name__} must implement initial_supply_voltage")

    def __init__(
        self,
        actuator_start: wp.array[wp.int32],
        actuators: wp.array[wp.int32],
        world: wp.array[wp.int32] | None = None,
    ):
        self.num_batteries = len(actuator_start) - 1
        """Number of batteries B."""
        self.device = actuator_start.device
        self.actuator_start = actuator_start
        """Start of each battery's actuators in :attr:`actuators`, shape (B + 1,)."""
        self.actuators = actuators
        """Actuators of all batteries, sorted by battery."""
        self.world = world if world is not None else wp.full(self.num_batteries, -1, dtype=wp.int32, device=self.device)
        """World of each battery, ``-1`` for none, shape (B,)."""
        if self.world.shape != (self.num_batteries,):
            raise ValueError(f"world shape {self.world.shape} must be ({self.num_batteries},)")

    def get_attribute_frequency(self, name: str) -> Model.AttributeFrequency:
        """Get the frequency of a parameter array of these batteries.

        The parameter arrays (e.g. :attr:`world`) have one entry per battery
        of this object. Their frequency is
        :attr:`Model.AttributeFrequency.BATTERY
        <newton.Model.AttributeFrequency.BATTERY>` limited to this object.

        Args:
            name: Name of the parameter array, e.g. ``"world"``.

        Returns:
            The frequency of the parameter array.

        Raises:
            KeyError: If there is no parameter array with this name.
        """
        from ...sim.model import Model  # noqa: PLC0415

        value = getattr(self, name, None)
        if isinstance(value, wp.array) and value.shape == (self.num_batteries,):
            return Model.AttributeFrequency.BATTERY
        raise KeyError(f"Attribute frequency of '{name}' is not known")

    def step(self, sim_control: Any) -> None:
        """Update the supply voltage of the actuators from the current they draw.

        Call once per step, after the actuators have stepped.

        Args:
            sim_control: Object carrying the arrays the battery reads and
                writes, e.g. :class:`~newton.Control`.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement step")

    def reset(self, sim_control: Any, *, world_mask: wp.array[wp.bool] | None = None) -> None:
        """Restore the supply voltage of a full battery.

        Call after resetting worlds, so their drives start from a full battery.

        Args:
            sim_control: Object carrying ``actuator_supply_voltage``, e.g.
                :class:`~newton.Control`.
            world_mask: Boolean mask with one entry per world. Batteries in
                ``True`` worlds are reset. ``None`` resets all, including
                batteries outside any world, which a mask never resets.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement reset")
