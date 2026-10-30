# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Explicit effort mode: control law at the current state, then the motor and clamps."""

from __future__ import annotations

from typing import Any

import warp as wp

from .clamping.base import ClampingBase
from .command_clamping.base import CommandClampingBase
from .drives.base import DriveBase
from .motor_dc import MotorDC


class _EffortModeExplicit:
    """Explicit effort mode: control law at the current state, then command clamps, motor and clamps."""

    def __init__(
        self,
        drive: DriveBase,
        command_clamping: list[CommandClampingBase],
        motor: MotorDC | None,
        clamping: list[ClampingBase],
        device: wp.Device,
        drive_output: wp.array[float] | None = None,
    ):
        self._drive = drive
        self._command_clamping = command_clamping
        self._motor = motor
        self._clamping = clamping
        self._device = device
        self._drive_output = drive_output

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
        *drive_output* buffer, the command clamps bound it in place, and the
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
        if self._motor is not None:
            for clamp in self._command_clamping:
                clamp.modify_forces(
                    self._drive_output,
                    self._drive_output,
                    positions,
                    velocities,
                    pos_indices,
                    vel_indices,
                    self._device,
                    supply_voltage=supply_voltage,
                )
            self._motor.compute(
                self._drive_output,
                computed_forces,
                velocities,
                vel_indices,
                supply_voltage,
                actuator_indices,
                supply_current,
                device=self._device,
            )
        forces = computed_forces
        for clamp in self._clamping:
            clamp.modify_forces(forces, applied_forces, positions, velocities, pos_indices, vel_indices, self._device)
            forces = applied_forces
        return forces
