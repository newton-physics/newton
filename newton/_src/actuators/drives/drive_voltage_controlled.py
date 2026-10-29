# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import warp as wp

from .base import DriveBase


@wp.kernel
def _voltage_controlled_kernel(
    current_pos: wp.array[float],
    target_pos: wp.array[float],
    pos_indices: wp.array[wp.uint32],
    target_pos_indices: wp.array[wp.uint32],
    supply_voltage: wp.array[float],
    kp: wp.array[float],
    error_gain: wp.array[float],
    voltages: wp.array[float],
):
    i = wp.tid()
    position_error = target_pos[target_pos_indices[i]] - current_pos[pos_indices[i]]
    voltages[i] = supply_voltage[i] * error_gain[i] * kp[i] * position_error


class DriveVoltageControlled(DriveBase):
    """Firmware position controller of a voltage-controlled servo, as in BAM (Better Actuator Models).

    Computes the PWM duty cycle :math:`d = g\\, k_p\\, (q_{\\text{target}} - q)`
    and outputs the voltage :math:`u = V d` [V] at the supply voltage
    :math:`V`, where :math:`k_p` is ``kp`` and :math:`g` is ``error_gain``.

    The output is a voltage, so this drive needs a motor that takes a voltage
    command, e.g. :class:`~newton.actuators.MotorDC`. The current and PWM
    limits of the servo are applied by a
    :class:`~newton.actuators.ClampingCurrentPWM`.
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        resolved = {"kp": args.get("kp", 0.0), "error_gain": args.get("error_gain", 1.0)}
        for name, value in resolved.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        return resolved

    def __init__(self, kp: wp.array[float], error_gain: wp.array[float]):
        """Initialize the drive.

        Args:
            kp: Firmware proportional gain, as set on the servo
                [dimensionless]. Shape ``(N,)``.
            error_gain: Converts ``kp`` times the position error to a duty
                cycle [1/rad or 1/m]. Shape ``(N,)``.

        Raises:
            ValueError: The shapes of *kp* and *error_gain* differ.
        """
        if kp.shape != error_gain.shape:
            raise ValueError(f"kp shape {kp.shape} must match error_gain shape {error_gain.shape}")
        self.kp = kp
        """Firmware proportional gain, shape (N,)."""
        self.error_gain = error_gain
        """Conversion of ``kp`` times the position error to a duty cycle, shape (N,)."""

    def is_stateful(self) -> bool:
        return False

    def is_graphable(self) -> bool:
        return True

    def compute(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float] | None,
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        target_pos_indices: wp.array[wp.uint32],
        target_vel_indices: wp.array[wp.uint32],
        forces: wp.array[float],
        state: DriveBase.State | None,
        dt: float,
        device: wp.Device | None = None,
        *,
        custom_inputs: dict[str, Any] | None = None,
        supply_voltage: wp.array[float] | None = None,
    ) -> None:
        """Compute the voltage command and write it to ``forces[i]``.

        Raises:
            ValueError: *supply_voltage* is ``None``, i.e. the actuator has
                no motor.

        See :meth:`DriveBase.compute` for the arguments.
        """
        if supply_voltage is None:
            raise ValueError(
                "DriveVoltageControlled requires a motor, such as MotorDC, that provides the supply voltage."
            )
        wp.launch(
            kernel=_voltage_controlled_kernel,
            dim=len(forces),
            inputs=[positions, target_pos, pos_indices, target_pos_indices, supply_voltage, self.kp, self.error_gain],
            outputs=[forces],
            device=device,
        )
