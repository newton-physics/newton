# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import warp as wp

from ...core.reset import normalize_reset_world_mask
from ..actuator import _require_array
from .base import BatteryBase


@wp.kernel
def _battery_update_kernel(
    battery_actuator_start: wp.array[wp.int32],
    battery_actuators: wp.array[wp.int32],
    nominal_voltage: wp.array[float],
    resistance: wp.array[float],
    min_voltage: wp.array[float],
    actuator_current: wp.array[float],
    current: wp.array[float],
    supply_voltage: wp.array[float],
):
    b = wp.tid()
    start = battery_actuator_start[b]
    end = battery_actuator_start[b + 1]
    for k in range(start, end):
        current[b] += actuator_current[battery_actuators[k]]
    voltage = wp.max(nominal_voltage[b] - resistance[b] * wp.max(current[b], 0.0), min_voltage[b])
    for k in range(start, end):
        supply_voltage[battery_actuators[k]] = voltage


@wp.kernel
def _battery_reset_kernel(
    battery_actuator_start: wp.array[wp.int32],
    battery_actuators: wp.array[wp.int32],
    battery_world: wp.array[wp.int32],
    world_count: int,
    nominal_voltage: wp.array[float],
    world_mask: wp.array[wp.bool],
    supply_voltage: wp.array[float],
):
    b = wp.tid()
    world = battery_world[b]
    if world < 0:
        world = world_count
    if world_mask:
        if not world_mask[world]:
            return
    for k in range(battery_actuator_start[b], battery_actuator_start[b + 1]):
        supply_voltage[battery_actuators[k]] = nominal_voltage[b]


class BatteryConstantResistance(BatteryBase):
    """Battery from BAM (Better Actuator Models): an ideal voltage source behind a constant internal resistance.

    As in `BAM <https://github.com/Rhoban/bam>`_, the voltage of a battery is

    .. math::

        V = \\max\\big(V_{\\text{nominal}} - R \\max(0, \\textstyle\\sum I),\\;
        V_{\\min}\\big)

    over the currents :math:`I` the actuators assigned to it draw, e.g.
    :math:`d\\, I_{\\text{motor}}` for :class:`~newton.actuators.MotorDC`. The
    sum is signed, so a regenerating actuator offsets a driving one, and
    regeneration does not raise the voltage above nominal. Each battery sums
    its actuators in a fixed order, so the voltage is deterministic.

    The supply and the motors form an explicit feedback loop. For
    :class:`~newton.actuators.MotorDC` without an active current limit it
    settles while ``resistance * sum(max_pwm**2 / motor resistance) < 1``
    over the actuators sharing a battery; above that bound, or while a
    current limit is active, the voltage may oscillate instead of
    converging.
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments with defaults.

        Args:
            args: User-provided arguments.  ``nominal_voltage`` [V] is required; ``resistance`` [Ω]
                defaults to 0, and ``min_voltage`` [V] defaults to 0, with
                which the voltage only stays non-negative.

        Returns:
            Complete arguments with defaults filled in.

        Raises:
            ValueError: *nominal_voltage* is missing, *min_voltage* is
                negative or exceeds *nominal_voltage*, or *resistance* is
                negative.
        """
        if "nominal_voltage" not in args:
            raise ValueError("BatteryConstantResistance requires 'nominal_voltage' argument")
        nominal_voltage, resistance = args["nominal_voltage"], args.get("resistance", 0.0)
        min_voltage = args.get("min_voltage", 0.0)
        if not 0.0 <= min_voltage <= nominal_voltage:
            raise ValueError(
                f"min_voltage must be non-negative and at most nominal_voltage, got {min_voltage} and {nominal_voltage}"
            )
        if resistance < 0.0:
            raise ValueError(f"resistance must be non-negative, got {resistance}")
        return {"nominal_voltage": nominal_voltage, "resistance": resistance, "min_voltage": min_voltage}

    def __init__(
        self,
        nominal_voltage: wp.array[float],
        resistance: wp.array[float],
        min_voltage: wp.array[float],
        actuator_start: wp.array[wp.int32],
        actuators: wp.array[wp.int32],
        world: wp.array[wp.int32] | None = None,
        world_count: int = 0,
    ):
        """Initialize the batteries.

        Args:
            nominal_voltage: Voltage without load [V]. Shape ``(B,)``.
            resistance: Internal resistance [Ω]. Shape ``(B,)``.
            min_voltage: Lowest voltage under load [V]. Shape ``(B,)``.
            actuator_start: See :meth:`BatteryBase.__init__`.
            actuators: See :meth:`BatteryBase.__init__`.
            world: See :meth:`BatteryBase.__init__`.
            world_count: See :meth:`BatteryBase.__init__`.

        Raises:
            ValueError: An array does not have one entry per battery.
        """
        super().__init__(actuator_start, actuators, world, world_count)
        self.nominal_voltage = nominal_voltage
        """Voltage without load [V], shape (B,)."""
        self.resistance = resistance
        """Internal resistance [Ω], shape (B,)."""
        self.min_voltage = min_voltage
        """Lowest voltage under load [V], shape (B,)."""
        self._current = wp.zeros(
            self.num_batteries, dtype=float, device=self.device, requires_grad=nominal_voltage.requires_grad
        )
        for name in ("nominal_voltage", "resistance", "min_voltage"):
            shape = getattr(self, name).shape
            if shape != (self.num_batteries,):
                raise ValueError(f"{name} shape {shape} must be ({self.num_batteries},)")

    def step(self, sim_control: Any) -> None:
        """Drop each battery's voltage with the current of its actuators.

        Raises:
            AttributeError: *sim_control* has no ``actuator_supply_current``
                or ``actuator_supply_voltage``.
            ValueError: One of these arrays is ``None``.

        See :meth:`BatteryBase.step` for the arguments.
        """
        self._current.zero_()
        wp.launch(
            _battery_update_kernel,
            dim=self.num_batteries,
            inputs=[
                self.actuator_start,
                self.actuators,
                self.nominal_voltage,
                self.resistance,
                self.min_voltage,
                _require_array(sim_control, "actuator_supply_current"),
            ],
            outputs=[self._current, _require_array(sim_control, "actuator_supply_voltage")],
            device=self.device,
        )

    def reset(self, sim_control: Any, *, world_mask: wp.array[wp.bool] | None = None) -> None:
        """Restore the nominal voltage of the reset batteries.

        Raises:
            AttributeError: *sim_control* has no ``actuator_supply_voltage``.
            ValueError: ``actuator_supply_voltage`` is ``None``, or
                *world_mask* does not have shape ``(world_count + 1,)``.

        See :meth:`BatteryBase.reset` for the arguments.
        """
        world_mask = normalize_reset_world_mask(world_mask, world_count=self.world_count, device=self.device)
        wp.launch(
            _battery_reset_kernel,
            dim=self.num_batteries,
            inputs=[
                self.actuator_start,
                self.actuators,
                self.world,
                self.world_count,
                self.nominal_voltage,
                world_mask,
            ],
            outputs=[_require_array(sim_control, "actuator_supply_voltage")],
            device=self.device,
        )
