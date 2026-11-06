# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Explicit effort mode: control law at the current state, then clamps and the optional motor."""

from __future__ import annotations

from typing import Any

import warp as wp

from .clamping.base import ClampingBase
from .drives.base import DriveBase
from .motor_dc import MotorDC


@wp.kernel
def _duty_to_voltage_kernel(duty: wp.array[float], supply_voltage: wp.array[float], voltage: wp.array[float]):
    i = wp.tid()
    voltage[i] = wp.clamp(duty[i], -1.0, 1.0) * wp.max(supply_voltage[i], 0.0)


class _EffortModeExplicit:
    """Explicit effort mode: control law at the current state, then clamps and the optional motor."""

    def __init__(
        self,
        drive: DriveBase,
        motor: MotorDC | None,
        clamping: list[ClampingBase],
        device: wp.Device,
        drive_output: wp.array[float] | None = None,
        motor_voltage: wp.array[float] | None = None,
    ):
        self._drive = drive
        self._motor = motor
        self._clamping = clamping
        self._device = device
        self._drive_output = drive_output
        self._motor_voltage = motor_voltage

    def is_graphable(self) -> bool:
        return self._drive.is_graphable() and (self._motor is None or self._motor.is_graphable())

    def compute_force(
        self,
        sim_state: Any,
        positions: wp.array[float],
        velocities: wp.array[float],
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float] | None,
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        target_pos_indices: wp.array[wp.uint32],
        target_vel_indices: wp.array[wp.uint32],
        computed_forces: wp.array[float],
        applied_forces: wp.array[float] | None,
        drive_state: DriveBase.State | None,
        dt: float | None,
        custom_inputs: dict[str, Any] | None = None,
        bias_force: wp.array[float] | None = None,
        supply_voltage: wp.array[float] | None = None,
        supply_current: wp.array[float] | None = None,
        actuator_indices: wp.array[wp.uint32] | None = None,
    ) -> wp.array[float]:
        """Compute raw effort into *computed_forces*, clamp into *applied_forces*.

        With a motor, the drive writes the motor command into the
        *drive_output* buffer. Duty clamps write into *applied_forces*, then
        bounded duty is converted to voltage using the actual supply. The
        motor writes its effort into *computed_forces* and its current into
        *supply_current*.

        Returns the buffer holding the final (clamped) effort.
        *bias_force* is ignored.
        """
        compute_kwargs: dict[str, Any] = {"device": self._device}
        if self._drive.custom_inputs:
            compute_kwargs["custom_inputs"] = custom_inputs
        if supply_voltage is not None:
            compute_kwargs["supply_voltage"] = supply_voltage
        self._drive.compute(
            positions,
            velocities,
            target_pos,
            target_vel,
            feedforward,
            pos_indices,
            vel_indices,
            target_pos_indices,
            target_vel_indices,
            computed_forces if self._motor is None else self._drive_output,
            drive_state,
            dt,
            **compute_kwargs,
        )
        forces = computed_forces if self._motor is None else self._drive_output
        for clamp in self._clamping:
            clamp_kwargs = {"supply_voltage": supply_voltage} if self._motor is not None else {}
            clamp.modify_forces(
                forces,
                applied_forces,
                positions,
                velocities,
                pos_indices,
                vel_indices,
                self._device,
                **clamp_kwargs,
            )
            forces = applied_forces
        if self._motor is not None:
            # Preserve duty and voltage separately for their backward passes.
            wp.launch(
                _duty_to_voltage_kernel,
                dim=len(forces),
                inputs=[forces, supply_voltage],
                outputs=[self._motor_voltage],
                device=self._device,
            )
            self._motor.compute(
                self._motor_voltage,
                computed_forces,
                velocities,
                vel_indices,
                supply_voltage,
                actuator_indices,
                supply_current,
                device=self._device,
            )
            forces = computed_forces
        return forces
