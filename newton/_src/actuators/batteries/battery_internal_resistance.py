# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import warp as wp

from ..actuator import _require_array
from .base import SUPPLY_CURRENT_GAIN_ATTR, SUPPLY_VOLTAGE_ATTR, BatteryBase


@wp.kernel
def _battery_update_kernel(
    battery_actuator_start: wp.array[wp.int32],
    battery_actuators: wp.array[wp.int32],
    nominal_voltage: wp.array[float],
    resistance: wp.array[float],
    min_voltage: wp.array[float],
    current_gain: wp.array[float],
    effort: wp.array[float],
    supply_voltage: wp.array[float],
):
    b = wp.tid()
    start = battery_actuator_start[b]
    end = battery_actuator_start[b + 1]
    current = float(0.0)
    for k in range(start, end):
        a = battery_actuators[k]
        current += current_gain[a] * effort[a]
    voltage = wp.max(nominal_voltage[b] - resistance[b] * wp.max(current, 0.0), min_voltage[b])
    for k in range(start, end):
        supply_voltage[battery_actuators[k]] = voltage


@wp.kernel
def _battery_reset_kernel(
    battery_actuator_start: wp.array[wp.int32],
    battery_actuators: wp.array[wp.int32],
    battery_world: wp.array[wp.int32],
    nominal_voltage: wp.array[float],
    world_mask: wp.array[wp.bool],
    supply_voltage: wp.array[float],
):
    b = wp.tid()
    world = battery_world[b]
    if world_mask:
        if world < 0 or not world_mask[world]:
            return
    for k in range(battery_actuator_start[b], battery_actuator_start[b + 1]):
        supply_voltage[battery_actuators[k]] = nominal_voltage[b]


class BatteryInternalResistance(BatteryBase):
    """Batteries modeled as an ideal voltage source behind an internal resistance.

    As in `BAM <https://github.com/Rhoban/bam>`_, the voltage of a battery is

    .. math::

        V = \\max\\big(V_{\\text{nominal}} - R \\max(0, \\textstyle\\sum g\\, \\tau),\\;
        V_{\\min}\\big)

    over the actuators assigned to it, where :math:`g \\tau` is the current an
    actuator draws from the battery: :math:`g` is
    ``actuator_supply_current_gain``, written by the drive, e.g.
    :math:`d / k_t` for :class:`~newton.actuators.DriveBAM`, and :math:`\\tau`
    is ``actuator_applied_effort``. The sum is signed, so a regenerating
    actuator offsets a driving one, and regeneration does not raise the
    voltage above nominal. Each battery sums its actuators in a fixed order,
    so the voltage is deterministic.

    The supply and the drives form an explicit feedback loop. For
    :class:`~newton.actuators.DriveBAM` it settles only while
    ``resistance * sum(max_pwm**2 / drive resistance) < 1`` over the
    actuators sharing a battery; above that the voltage oscillates instead
    of converging.

    Args:
        nominal_voltage: Voltage without load [V]. Shape ``(B,)``.
        resistance: Internal resistance [Ω]. Shape ``(B,)``.
        min_voltage: Lowest voltage under load [V]. Shape ``(B,)``.
        actuator_start: See :class:`BatteryBase`.
        actuators: See :class:`BatteryBase`.
        world: See :class:`BatteryBase`.

    Raises:
        ValueError: An array does not have one entry per battery.
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments of one battery with defaults.

        Args:
            args: ``nominal_voltage`` [V] and ``resistance`` [Ω] are required;
                ``min_voltage`` [V] defaults to 0, with which the voltage only
                stays non-negative.

        Returns:
            Complete arguments with defaults filled in.

        Raises:
            ValueError: *nominal_voltage* or *resistance* is missing,
                *min_voltage* is negative or exceeds *nominal_voltage*, or
                *resistance* is negative.
        """
        for name in ("nominal_voltage", "resistance"):
            if name not in args:
                raise ValueError(f"BatteryInternalResistance requires '{name}' argument")
        nominal_voltage, resistance = args["nominal_voltage"], args["resistance"]
        min_voltage = args.get("min_voltage", 0.0)
        if not 0.0 <= min_voltage <= nominal_voltage:
            raise ValueError(
                f"min_voltage must be non-negative and at most nominal_voltage, got {min_voltage} and {nominal_voltage}"
            )
        if resistance < 0.0:
            raise ValueError(f"resistance must be non-negative, got {resistance}")
        return {"nominal_voltage": nominal_voltage, "resistance": resistance, "min_voltage": min_voltage}

    @classmethod
    def initial_supply_voltage(cls, args: dict[str, Any]) -> float:
        """Return the nominal voltage [V]."""
        return args["nominal_voltage"]

    def __init__(
        self,
        nominal_voltage: wp.array[float],
        resistance: wp.array[float],
        min_voltage: wp.array[float],
        actuator_start: wp.array[wp.int32],
        actuators: wp.array[wp.int32],
        world: wp.array[wp.int32] | None = None,
    ):
        super().__init__(actuator_start, actuators, world)
        self.nominal_voltage = nominal_voltage
        """Voltage without load [V], shape (B,)."""
        self.resistance = resistance
        """Internal resistance [Ω], shape (B,)."""
        self.min_voltage = min_voltage
        """Lowest voltage under load [V], shape (B,)."""
        for name in ("nominal_voltage", "resistance", "min_voltage"):
            shape = getattr(self, name).shape
            if shape != (self.num_batteries,):
                raise ValueError(f"{name} shape {shape} must be ({self.num_batteries},)")

    def step(self, sim_control: Any, *, effort: wp.array[float] | None = None) -> None:
        """Drop each battery's voltage with the current of its actuators.

        Writes ``actuator_supply_voltage`` for the actuators of the batteries.
        Drives that read the supply voltage, e.g.
        :class:`~newton.actuators.DriveBAM` with ``supply_from_battery=True``,
        use it on their next step. Call once per step, after the actuators
        have stepped.

        Args:
            sim_control: Object carrying ``actuator_supply_voltage`` and
                ``actuator_supply_current_gain``, and ``actuator_applied_effort``
                when *effort* is ``None``, e.g. :class:`~newton.Control`.
            effort: Effort of each actuator of the scene [N or N·m]. ``None``
                reads ``actuator_applied_effort``.

        Raises:
            AttributeError: *sim_control* has no ``actuator_supply_voltage``
                or ``actuator_supply_current_gain``, or no
                ``actuator_applied_effort`` when *effort* is ``None``.
            ValueError: One of these arrays is ``None``.
        """
        if effort is None:
            effort = _require_array(sim_control, "actuator_applied_effort")
        wp.launch(
            _battery_update_kernel,
            dim=self.num_batteries,
            inputs=[
                self.actuator_start,
                self.actuators,
                self.nominal_voltage,
                self.resistance,
                self.min_voltage,
                _require_array(sim_control, SUPPLY_CURRENT_GAIN_ATTR),
                effort,
            ],
            outputs=[_require_array(sim_control, SUPPLY_VOLTAGE_ATTR)],
            device=self.device,
        )

    def reset(self, sim_control: Any, *, world_mask: wp.array[wp.bool] | None = None) -> None:
        """Restore the nominal voltage in ``actuator_supply_voltage``.

        Call after resetting worlds, so their drives start from a full battery.

        Args:
            sim_control: Object carrying ``actuator_supply_voltage``, e.g.
                :class:`~newton.Control`.
            world_mask: Boolean mask with one entry per world. Batteries in
                ``True`` worlds are reset. ``None`` resets all, including
                batteries outside any world, which a mask never resets.

        Raises:
            AttributeError: *sim_control* has no ``actuator_supply_voltage``.
            ValueError: ``actuator_supply_voltage`` is ``None``.
        """
        wp.launch(
            _battery_reset_kernel,
            dim=self.num_batteries,
            inputs=[self.actuator_start, self.actuators, self.world, self.nominal_voltage, world_mask],
            outputs=[_require_array(sim_control, SUPPLY_VOLTAGE_ATTR)],
            device=self.device,
        )
