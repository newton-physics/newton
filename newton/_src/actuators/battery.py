# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import warp as wp

if TYPE_CHECKING:
    from ..sim.builder import ModelBuilder
    from ..sim.control import Control
    from ..sim.model import Model

SUPPLY_VOLTAGE_ATTR = "supply_voltage"
BATTERY_NAMESPACE = "actuator"
BATTERY_FREQUENCY = f"{BATTERY_NAMESPACE}:battery"


@wp.func
def _battery_dof_range(world: int, joint_dof_world_start: wp.array[wp.int32]):
    if world < 0:
        return 0, joint_dof_world_start[joint_dof_world_start.shape[0] - 1]
    return joint_dof_world_start[world], joint_dof_world_start[world + 1]


@wp.kernel
def _battery_update_kernel(
    battery_world: wp.array[wp.int32],
    nominal_voltage: wp.array[float],
    sag_gain: wp.array[float],
    min_voltage: wp.array[float],
    joint_dof_world_start: wp.array[wp.int32],
    joint_dof_battery: wp.array[wp.int32],
    effort: wp.array[float],
    supply_voltage: wp.array[float],
):
    b = wp.tid()
    start, end = _battery_dof_range(battery_world[b], joint_dof_world_start)
    total = float(0.0)
    for d in range(start, end):
        if joint_dof_battery[d] == b:
            total += wp.abs(effort[d])
    voltage = wp.max(nominal_voltage[b] - sag_gain[b] * total, min_voltage[b])
    for d in range(start, end):
        if joint_dof_battery[d] == b:
            supply_voltage[d] = voltage


@wp.kernel
def _battery_reset_kernel(
    battery_world: wp.array[wp.int32],
    nominal_voltage: wp.array[float],
    joint_dof_world_start: wp.array[wp.int32],
    joint_dof_battery: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    supply_voltage: wp.array[float],
):
    b = wp.tid()
    world = battery_world[b]
    if world_mask:
        if world < 0 or not world_mask[world]:
            return
    start, end = _battery_dof_range(world, joint_dof_world_start)
    for d in range(start, end):
        if joint_dof_battery[d] == b:
            supply_voltage[d] = nominal_voltage[b]


def register_battery_attributes(builder: ModelBuilder) -> None:
    """Register the battery frequency and its custom attributes on *builder*.

    :meth:`~newton.ModelBuilder.add_battery` calls this automatically.

    Args:
        builder: Newton model builder on which to register the attributes.
    """
    from ..sim.model import Model  # noqa: PLC0415

    builder.add_custom_frequency(builder.CustomFrequency(name="battery", namespace=BATTERY_NAMESPACE))
    for name in ("battery_nominal_voltage", "battery_sag_gain", "battery_min_voltage"):
        builder.add_custom_attribute(
            builder.CustomAttribute(
                name=name, dtype=wp.float32, frequency=BATTERY_FREQUENCY, default=0.0, namespace=BATTERY_NAMESPACE
            )
        )
    builder.add_custom_attribute(
        builder.CustomAttribute(
            name="battery_world",
            dtype=wp.int32,
            frequency=BATTERY_FREQUENCY,
            default=-1,
            references="world",
            namespace=BATTERY_NAMESPACE,
        )
    )
    builder.add_custom_attribute(
        builder.CustomAttribute(
            name="joint_dof_battery",
            dtype=wp.int32,
            frequency=Model.AttributeFrequency.JOINT_DOF,
            default=-1,
            references=BATTERY_FREQUENCY,
            namespace=BATTERY_NAMESPACE,
        )
    )
    builder.add_custom_attribute(
        builder.CustomAttribute(
            name=SUPPLY_VOLTAGE_ATTR,
            dtype=wp.float32,
            frequency=Model.AttributeFrequency.JOINT_DOF,
            assignment=Model.AttributeAssignment.CONTROL,
            default=0.0,
        )
    )


def update_battery_supply(model: Model, control: Control, *, effort: wp.array[float] | None = None) -> None:
    """Sag each battery's voltage with the effort on its DOFs and write ``control.supply_voltage``.

    The batteries are added with :meth:`~newton.ModelBuilder.add_battery`. The
    voltage of a battery is

    .. math::

        V = \\max\\big(V_{\\text{nominal}} - g_{\\text{sag}} \\sum |\\tau|,\\;
        V_{\\min}\\big)

    over the DOFs assigned to it, and is written into ``supply_voltage`` for
    those DOFs. Drives that read the supply voltage, e.g.
    :class:`~newton.actuators.DriveBAM` with ``supply_from_control=True``,
    use it on the next actuator step. DOFs without a battery are not
    written. Each battery sums its DOFs in a fixed order, so the voltage is
    deterministic.

    Call once per step, after the actuators have stepped and before the
    effort is cleared. The effort is ``control.joint_f`` by default, after
    clamping. To draw the effort before clamping, pass
    ``control_computed_output_attr`` to
    :meth:`~newton.ModelBuilder.add_actuator` and that array as *effort*;
    actuators accumulate into it, so zero it before each actuator step like
    ``joint_f``.

    The battery parameters are ``model.actuator.battery_nominal_voltage``,
    ``battery_sag_gain`` and ``battery_min_voltage``, indexed by battery;
    ``model.actuator.battery_world`` gives the world of each battery, e.g.
    to randomize the parameters per world.

    The supply and the drives form an explicit feedback loop. For
    :class:`~newton.actuators.DriveBAM` it settles only while
    ``sag_gain * sum(kt * max_pwm / resistance) < 1`` over the DOFs sharing
    a battery; above that the voltage oscillates between the minimum and the
    nominal voltage instead of converging.

    Args:
        model: Model with batteries.
        control: Object carrying ``supply_voltage``, and ``joint_f`` when
            *effort* is ``None``, e.g. :class:`~newton.Control`.
        effort: Effort drawn from the batteries [N or N·m], shaped like
            ``joint_qd``. ``None`` reads ``control.joint_f``.

    Raises:
        ValueError: The model has no batteries, or the effort or
            ``supply_voltage`` is missing or not shaped like ``joint_qd``.
    """
    batteries = _require_batteries(model)
    if effort is None:
        effort = getattr(control, "joint_f", None)
    supply_voltage = _require_dof_array(model, control, SUPPLY_VOLTAGE_ATTR)
    if effort is None or effort.shape != (model.joint_dof_count,):
        shape = None if effort is None else effort.shape
        raise ValueError(f"effort shape {shape} must be ({model.joint_dof_count},)")
    wp.launch(
        _battery_update_kernel,
        dim=len(batteries.battery_nominal_voltage),
        inputs=[
            batteries.battery_world,
            batteries.battery_nominal_voltage,
            batteries.battery_sag_gain,
            batteries.battery_min_voltage,
            model.joint_dof_world_start,
            batteries.joint_dof_battery,
            effort,
        ],
        outputs=[supply_voltage],
        device=model.device,
    )


def reset_battery_supply(model: Model, control: Control, *, world_mask: wp.array[wp.bool] | None = None) -> None:
    """Restore the nominal voltage of the batteries in ``control.supply_voltage``.

    Call after resetting worlds, so their drives start from a full battery.

    Args:
        model: Model with batteries.
        control: Object carrying ``supply_voltage``, e.g.
            :class:`~newton.Control`.
        world_mask: Boolean mask of length ``world_count``. Batteries in
            ``True`` worlds are reset. ``None`` resets all, including
            batteries added outside any world, which a mask never resets.

    Raises:
        ValueError: The model has no batteries, ``supply_voltage`` is
            missing or not shaped like ``joint_qd``, or *world_mask* is not
            shaped ``(world_count,)``.
    """
    batteries = _require_batteries(model)
    supply_voltage = _require_dof_array(model, control, SUPPLY_VOLTAGE_ATTR)
    if world_mask is not None and world_mask.shape != (model.world_count,):
        raise ValueError(f"world_mask shape {world_mask.shape} must be ({model.world_count},)")
    wp.launch(
        _battery_reset_kernel,
        dim=len(batteries.battery_nominal_voltage),
        inputs=[
            batteries.battery_world,
            batteries.battery_nominal_voltage,
            model.joint_dof_world_start,
            batteries.joint_dof_battery,
            world_mask,
        ],
        outputs=[supply_voltage],
        device=model.device,
    )


def _require_batteries(model: Model) -> Any:
    batteries = getattr(model, BATTERY_NAMESPACE, None)
    if getattr(batteries, "battery_nominal_voltage", None) is None:
        raise ValueError("The model has no batteries; add them with ModelBuilder.add_battery()")
    return batteries


def _require_dof_array(model: Model, control: Any, name: str) -> wp.array[float]:
    array = getattr(control, name, None)
    if array is None or array.shape != (model.joint_dof_count,):
        shape = None if array is None else array.shape
        raise ValueError(f"control.{name} shape {shape} must be ({model.joint_dof_count},)")
    return array
