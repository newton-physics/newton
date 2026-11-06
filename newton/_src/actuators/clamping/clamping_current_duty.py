# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar

import warp as wp

from ..space import DriveOutputSpace
from .base import ClampingBase


@wp.kernel
def _current_duty_clamp_kernel(
    velocities: wp.array[float],
    vel_indices: wp.array[wp.uint32],
    voltage_estimate: wp.array[float],
    max_current: wp.array[float],
    v_min: wp.array[float],
    kt_estimate: wp.array[float],
    resistance_estimate: wp.array[float],
    src: wp.array[float],
    dst: wp.array[float],
):
    i = wp.tid()
    duty = src[i]
    voltage = voltage_estimate[i]
    if max_current[i] > 0.0 and voltage > v_min[i]:
        back_emf = kt_estimate[i] * velocities[vel_indices[i]]
        span = resistance_estimate[i] * max_current[i]
        duty = wp.clamp(duty, (back_emf - span) / voltage, (back_emf + span) / voltage)
    dst[i] = wp.clamp(duty, -1.0, 1.0)


class ClampingCurrentDuty(ClampingBase):
    r"""Limit signed duty using a DC-motor current estimate, then a duty bound.

    Input and output are dimensionless duty commands. For a voltage
    estimate ``V > v_min``, the current-derived duty window is::

        lower = (kt_estimate * velocity - resistance_estimate * max_current) / V
        upper = (kt_estimate * velocity + resistance_estimate * max_current) / V
        duty = clamp(requested_duty, lower, upper)
        duty = clamp(duty, -1, 1)

    ``voltage_source`` explicitly selects the estimate: ``INPUT`` reads the
    supplied voltage; ``NOMINAL`` uses ``nominal_voltage``. The actuator
    separately uses actual supply voltage to convert the resulting duty into
    motor voltage. This clamp never performs that conversion.

    ``kt_estimate`` and ``resistance_estimate`` are independent firmware
    parameters; they need not match the physical motor.

    A zero ``max_current`` disables the current window. An estimated voltage at or below
    ``v_min`` also skips it, avoiding division by a near-zero voltage. This
    threshold does not change the actual voltage delivered to the motor.
    The fixed duty bound [-1, 1] always applies and takes precedence over the
    current window. Overspeed or an inaccurate voltage estimate can therefore
    cause motor current to exceed the limit.
    This is a quasi-static current estimate, not an inductive current controller.
    """

    class VoltageSource(str, Enum):
        """Voltage used by the firmware's current estimate."""

        INPUT = "input"
        """Read the supplied voltage array."""
        NOMINAL = "nominal"
        """Use the configured nominal voltage, independently of the live supply."""

    input_space = DriveOutputSpace.DUTY

    SHARED_PARAMS: ClassVar[set[str]] = {"voltage_source"}

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        if DriveOutputSpace(args.get("input_space", cls.input_space)) is not cls.input_space:
            raise ValueError("ClampingCurrentDuty accepts only DUTY inputs")
        source = cls.VoltageSource(args.get("voltage_source", cls.VoltageSource.INPUT))
        resolved = {
            "max_current": args.get("max_current", 0.0),
            "v_min": args.get("v_min", 1.0e-6),
            "voltage_source": source,
        }
        if source is cls.VoltageSource.NOMINAL:
            nominal = args.get("nominal_voltage")
            if nominal is None or not 0.0 < nominal < float("inf"):
                raise ValueError("NOMINAL voltage_source requires a finite, positive nominal_voltage")
            resolved["nominal_voltage"] = nominal
        elif "nominal_voltage" in args:
            raise ValueError("nominal_voltage requires voltage_source=NOMINAL; INPUT uses the supplied voltage")
        if resolved["max_current"] < 0.0:
            raise ValueError(f"max_current must be non-negative, got {resolved['max_current']}")
        if not 0.0 < resolved["v_min"] < float("inf"):
            raise ValueError("v_min must be finite and positive")
        for name in ("kt_estimate", "resistance_estimate"):
            value = args.get(name)
            if value is None or not 0.0 < value < float("inf"):
                raise ValueError(f"{name} must be supplied, finite and positive")
            resolved[name] = value
        return resolved

    def __init__(
        self,
        max_current: wp.array[float],
        kt_estimate: wp.array[float],
        resistance_estimate: wp.array[float],
        *,
        voltage_source: VoltageSource = VoltageSource.INPUT,
        nominal_voltage: wp.array[float] | None = None,
        v_min: wp.array[float] | None = None,
    ):
        """Initialize current estimation and duty limits.

        Args:
            max_current: Current limit [A], shape (N,). Zero disables it.
            kt_estimate: Firmware estimate of the motor torque constant [N·m/A or N/A], shape (N,).
            resistance_estimate: Firmware estimate of the motor winding resistance [Ω], shape (N,).
            voltage_source: Select supplied or nominal voltage for the current
                estimate. Shared by all slots in this component.
            nominal_voltage: Positive firmware voltage estimate [V], shape (N,).
                Required in NOMINAL mode and rejected in INPUT mode. As with
                other batched parameters, scalar values are validated by
                resolve_arguments before construction; this constructor checks shapes.
            v_min: Positive voltage threshold [V], shape (N,). Defaults to
                1e-6 V per slot. Skip the current window when the selected
                voltage estimate is at or below this threshold; still bound duty.

        Raises:
            ValueError: Invalid source, missing or unused nominal voltage,
                or inconsistent array shapes.
        """
        self.voltage_source = self.VoltageSource(voltage_source)
        """Source of the firmware's voltage estimate."""
        if self.voltage_source is self.VoltageSource.NOMINAL and nominal_voltage is None:
            raise ValueError("NOMINAL voltage_source requires nominal_voltage")
        if self.voltage_source is self.VoltageSource.INPUT and nominal_voltage is not None:
            raise ValueError("nominal_voltage requires voltage_source=NOMINAL")
        self.nominal_voltage = nominal_voltage
        """Nominal voltage estimate [V], shape (N,), or None in INPUT mode."""
        self.max_current = max_current
        """Current limit [A], shape (N,); zero disables current limiting."""
        self.v_min = (
            wp.full(max_current.shape, 1.0e-6, dtype=float, device=max_current.device) if v_min is None else v_min
        )
        """Minimum voltage estimate for applying the current window [V], shape (N,)."""
        self.kt_estimate = kt_estimate
        """Firmware estimate of the motor torque constant [N·m/A or N/A], shape (N,)."""
        self.resistance_estimate = resistance_estimate
        """Firmware estimate of the motor winding resistance [Ω], shape (N,)."""
        for name in ("kt_estimate", "resistance_estimate", "nominal_voltage", "v_min"):
            value = getattr(self, name)
            if value is not None and value.shape != max_current.shape:
                raise ValueError(f"{name} shape {value.shape} must match max_current shape {max_current.shape}")

    def modify_forces(
        self,
        src_forces: wp.array[float],
        dst_forces: wp.array[float],
        positions: wp.array[float],
        velocities: wp.array[float],
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        device: wp.Device | None = None,
        *,
        supply_voltage: wp.array[float] | None = None,
    ) -> None:
        """Read signed duty from src_forces and write limited duty to dst_forces.

        Both buffers have shape (N,) and are dimensionless. INPUT mode reads
        the supplied supply_voltage array.
        NOMINAL mode needs no runtime voltage input. See the base method for
        the remaining arguments.

        Raises:
            ValueError: INPUT mode has no voltage array, or its shape does not
                match the actuator slots. There is no fallback to nominal voltage.
        """
        if self.voltage_source is self.VoltageSource.NOMINAL:
            estimate = self.nominal_voltage
        else:
            estimate = supply_voltage
            if estimate is None:
                raise ValueError("INPUT voltage_source requires a supplied voltage array")
        if estimate.shape != self.max_current.shape:
            raise ValueError(f"Voltage shape {estimate.shape} must match max_current shape {self.max_current.shape}")
        wp.launch(
            kernel=_current_duty_clamp_kernel,
            dim=len(src_forces),
            inputs=[
                velocities,
                vel_indices,
                estimate,
                self.max_current,
                self.v_min,
                self.kt_estimate,
                self.resistance_estimate,
                src_forces,
            ],
            outputs=[dst_forces],
            device=device,
        )
