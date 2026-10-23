# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from .actuator import ACTUATOR_FREQUENCY, ACTUATOR_NAMESPACE, _get_attribute, _require_array

if TYPE_CHECKING:
    from ..sim.builder import ModelBuilder
    from ..sim.control import Control
    from ..sim.model import Model

SUPPLY_VOLTAGE_ATTR = "actuator_supply_voltage"
SUPPLY_CURRENT_GAIN_ATTR = "actuator_supply_current_gain"
BATTERY_FREQUENCY = f"{ACTUATOR_NAMESPACE}:battery"


def register_battery_attributes(builder: ModelBuilder) -> None:
    """Register the battery frequency and its attributes on *builder*.

    Registers the battery frequency ``actuation:battery`` with the battery
    parameters, the battery of each actuator ``actuator_battery`` (``-1`` for
    none) and the per-actuator supply voltage and supply current gain on
    Control. It also registers ``battery_actuator_start`` (one entry per battery plus
    one) and ``battery_actuators`` (all actuators, sorted by battery), which
    :meth:`~newton.ModelBuilder.finalize` builds from ``actuator_battery`` so
    that battery ``b`` draws from the actuators
    ``battery_actuators[battery_actuator_start[b]:battery_actuator_start[b + 1]]``.

    :meth:`~newton.ModelBuilder.add_battery` calls this with the first battery.

    Args:
        builder: Newton model builder on which to register the attributes.
    """
    from ..sim.model import Model  # noqa: PLC0415

    builder.add_custom_frequency(builder.CustomFrequency(name="battery", namespace=ACTUATOR_NAMESPACE))
    builder.add_custom_attribute(
        builder.CustomAttribute(
            name="actuator_battery",
            dtype=wp.int32,
            frequency=ACTUATOR_FREQUENCY,
            default=-1,
            references=BATTERY_FREQUENCY,
            namespace=ACTUATOR_NAMESPACE,
        )
    )
    for name in ("battery_nominal_voltage", "battery_resistance", "battery_min_voltage"):
        builder.add_custom_attribute(
            builder.CustomAttribute(
                name=name, dtype=wp.float32, frequency=BATTERY_FREQUENCY, default=0.0, namespace=ACTUATOR_NAMESPACE
            )
        )
    builder.add_custom_attribute(
        builder.CustomAttribute(
            name="battery_world",
            dtype=wp.int32,
            frequency=BATTERY_FREQUENCY,
            default=-1,
            references="world",
            namespace=ACTUATOR_NAMESPACE,
        )
    )
    builder.add_custom_attribute(
        builder.CustomAttribute(
            name="battery_actuator_start",
            dtype=wp.int32,
            frequency=BATTERY_FREQUENCY,
            default=0,
            namespace=ACTUATOR_NAMESPACE,
        )
    )
    builder._add_custom_attribute_model_finalizer(
        f"{ACTUATOR_NAMESPACE}:battery_actuator_start", _finalize_battery_actuator_start
    )
    builder.add_custom_attribute(
        builder.CustomAttribute(
            name="battery_actuators",
            dtype=wp.int32,
            frequency=ACTUATOR_FREQUENCY,
            default=-1,
            references=ACTUATOR_FREQUENCY,
            namespace=ACTUATOR_NAMESPACE,
        )
    )
    builder._add_custom_attribute_model_finalizer(
        f"{ACTUATOR_NAMESPACE}:battery_actuators", _finalize_battery_actuators
    )
    for name in (SUPPLY_VOLTAGE_ATTR, SUPPLY_CURRENT_GAIN_ATTR):
        builder.add_custom_attribute(
            builder.CustomAttribute(
                name=name,
                dtype=wp.float32,
                frequency=ACTUATOR_FREQUENCY,
                assignment=Model.AttributeAssignment.CONTROL,
                default=0.0,
            )
        )


def _battery_actuator_order(builder: ModelBuilder, model: Model) -> tuple[np.ndarray, np.ndarray]:
    """Return all actuator rows sorted by battery, unassigned last, and the start of each battery's rows."""
    battery_count = model.custom_frequency_counts.get(BATTERY_FREQUENCY, 0)
    actuator_battery = np.array(
        [
            -1 if b is None else b
            for b in builder.custom_attributes[f"{ACTUATOR_NAMESPACE}:actuator_battery"].values or ()
        ],
        dtype=np.int64,
    )
    order = np.argsort(np.where(actuator_battery < 0, battery_count, actuator_battery), kind="stable")
    counts = np.bincount(actuator_battery[actuator_battery >= 0], minlength=battery_count)
    return order, np.concatenate(([0], np.cumsum(counts)))


def _finalize_battery_actuator_start(
    builder: ModelBuilder, model: Model, attribute: ModelBuilder.CustomAttribute
) -> None:
    _, start = _battery_actuator_order(builder, model)
    model.add_attribute(
        attribute.name,
        wp.array(start, dtype=wp.int32, device=model.device),
        attribute.frequency,
        attribute.assignment,
        attribute.namespace,
    )


def _finalize_battery_actuators(builder: ModelBuilder, model: Model, attribute: ModelBuilder.CustomAttribute) -> None:
    order, _ = _battery_actuator_order(builder, model)
    model.add_attribute(
        attribute.name,
        wp.array(order, dtype=wp.int32, device=model.device),
        attribute.frequency,
        attribute.assignment,
        attribute.namespace,
        attribute.references,
    )


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


def update_battery_supply(model: Model, control: Control, *, effort: wp.array[float] | None = None) -> None:
    """Drop each battery's voltage with the current of its actuators and write ``control.actuator_supply_voltage``.

    The batteries are added with :meth:`~newton.ModelBuilder.add_battery`. A
    battery is an ideal voltage source behind a resistance, as in
    `BAM <https://github.com/Rhoban/bam>`_. Its voltage is

    .. math::

        V = \\max\\big(V_{\\text{nominal}} - R \\max(0, \\textstyle\\sum g\\, \\tau),\\;
        V_{\\min}\\big)

    over the actuators assigned to it, where :math:`g \\tau` is the current an
    actuator draws from the battery: :math:`g` is
    ``control.actuator_supply_current_gain``, written by the drive, e.g.
    :math:`d / k_t` for :class:`~newton.actuators.DriveBAM`. The sum is signed,
    so a regenerating actuator offsets a driving one, and regeneration does
    not raise the voltage above nominal. The voltage is written into
    ``actuator_supply_voltage`` for those actuators. Drives that read the
    supply voltage, e.g. :class:`~newton.actuators.DriveBAM` with
    ``supply_from_battery=True``, use it on the next actuator step.
    Actuators without a battery are not written. Each battery sums its
    actuators in a fixed order, so the voltage is deterministic.

    Call once per step, after the actuators have stepped. The effort is
    ``control.actuator_computed_effort`` by default: the effort of each
    actuator before clamping, which actuators built by
    :class:`~newton.ModelBuilder` write there on every step. Pass *effort*
    when a later stage represents the motor effort better, e.g. after a
    clamp that models a motor current limit.

    The battery parameters are ``model.actuation.battery_nominal_voltage``,
    ``battery_resistance`` and ``battery_min_voltage``, indexed by battery;
    ``model.actuation.battery_world`` gives the world of each battery, e.g.
    to randomize the parameters per world.

    The supply and the drives form an explicit feedback loop. For
    :class:`~newton.actuators.DriveBAM` it settles only while
    ``battery_resistance * sum(max_pwm**2 / resistance) < 1`` over the
    actuators sharing a battery; above that the voltage oscillates instead
    of converging.

    Args:
        model: Model with batteries.
        control: Object carrying ``actuator_supply_voltage`` and
            ``actuator_supply_current_gain``, and ``actuator_computed_effort``
            when *effort* is ``None``, e.g. :class:`~newton.Control`.
        effort: Effort of each actuator [N or N·m], one entry per actuator.
            ``None`` reads ``control.actuator_computed_effort``.

    Raises:
        AttributeError: *control* has no ``actuator_supply_voltage`` or
            ``actuator_supply_current_gain``.
        ValueError: The model has no batteries, the effort is missing,
            ``actuator_supply_voltage`` or ``actuator_supply_current_gain``
            is ``None``, or any of them does not have one entry per actuator.
    """
    batteries = _require_batteries(model)
    if effort is None:
        effort = _get_attribute(control, "actuator_computed_effort", None)
    supply_voltage = _require_actuator_array(model, _require_array(control, SUPPLY_VOLTAGE_ATTR), SUPPLY_VOLTAGE_ATTR)
    current_gain = _require_actuator_array(
        model, _require_array(control, SUPPLY_CURRENT_GAIN_ATTR), SUPPLY_CURRENT_GAIN_ATTR
    )
    _require_actuator_array(model, effort, "effort")
    wp.launch(
        _battery_update_kernel,
        dim=len(batteries.battery_nominal_voltage),
        inputs=[
            batteries.battery_actuator_start,
            batteries.battery_actuators,
            batteries.battery_nominal_voltage,
            batteries.battery_resistance,
            batteries.battery_min_voltage,
            current_gain,
            effort,
        ],
        outputs=[supply_voltage],
        device=model.device,
    )


def reset_battery_supply(model: Model, control: Control, *, world_mask: wp.array[wp.bool] | None = None) -> None:
    """Restore the nominal voltage of the batteries in ``control.actuator_supply_voltage``.

    Call after resetting worlds, so their drives start from a full battery.

    Args:
        model: Model with batteries.
        control: Object carrying ``actuator_supply_voltage``, e.g.
            :class:`~newton.Control`.
        world_mask: Boolean mask of length ``world_count``. Batteries in
            ``True`` worlds are reset. ``None`` resets all, including
            batteries added outside any world, which a mask never resets.

    Raises:
        AttributeError: *control* has no ``actuator_supply_voltage``.
        ValueError: The model has no batteries, ``actuator_supply_voltage``
            is ``None`` or does not have one entry per actuator, or
            *world_mask* is not shaped ``(world_count,)``.
    """
    batteries = _require_batteries(model)
    supply_voltage = _require_actuator_array(model, _require_array(control, SUPPLY_VOLTAGE_ATTR), SUPPLY_VOLTAGE_ATTR)
    if world_mask is not None and world_mask.shape != (model.world_count,):
        raise ValueError(f"world_mask shape {world_mask.shape} must be ({model.world_count},)")
    wp.launch(
        _battery_reset_kernel,
        dim=len(batteries.battery_nominal_voltage),
        inputs=[
            batteries.battery_actuator_start,
            batteries.battery_actuators,
            batteries.battery_world,
            batteries.battery_nominal_voltage,
            world_mask,
        ],
        outputs=[supply_voltage],
        device=model.device,
    )


def _require_batteries(model: Model) -> Any:
    batteries = getattr(model, ACTUATOR_NAMESPACE, None)
    if getattr(batteries, "battery_nominal_voltage", None) is None:
        raise ValueError("The model has no batteries; add them with ModelBuilder.add_battery()")
    return batteries


def _require_actuator_array(model: Model, array: Any, name: str) -> wp.array[float]:
    count = model.custom_frequency_counts.get(ACTUATOR_FREQUENCY, 0)
    if array is None or array.shape != (count,):
        shape = None if array is None else array.shape
        raise ValueError(f"{name} shape {shape} must be ({count},), one entry per actuator")
    return array
