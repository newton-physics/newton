# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar

import warp as wp


class CommandClampingBase:
    """Base class for clamping of the motor command.

    Command clamping bounds the motor command of an actuator (e.g. a voltage
    for :class:`~newton.actuators.MotorDC`) before the motor turns it into the
    effort, so it requires a motor. It runs before the motor whatever its
    position in the clamping list, and receives the supply voltage of the
    motor. Unlike :class:`~newton.actuators.ClampingBase`, it has no implicit form, since the
    implicit effort mode does not model a motor.

    **Validation contract:** as for :class:`~newton.actuators.ClampingBase`.
    """

    SHARED_PARAMS: ClassVar[set[str]] = set()

    MOTOR_PARAMS: ClassVar[set[str]] = set()
    """Constructor parameters taken from the motor of the actuator.

    :meth:`ModelBuilder.add_actuator <newton.ModelBuilder.add_actuator>`
    requires a motor with these parameters and passes the motor's arrays of
    the same name, so they are authored once, on the motor.
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

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        """Called by :class:`~newton.actuators.Actuator` after construction to set up device-specific resources.

        Args:
            device: Warp device to use.
            num_actuators: Number of actuators (DOFs) this clamping manages.
        """

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
        supply_voltage: wp.array[float],
    ) -> None:
        """Read the motor command from *src*, apply clamping, write to *dst*.

        Args:
            src_forces: Input motor command buffer to read. Shape ``(N,)``.
            dst_forces: Output motor command buffer to write. Shape ``(N,)``.
            positions: Joint positions [m or rad].
            velocities: Joint velocities [m/s or rad/s].
            pos_indices: Indices into *positions* for each DOF.
            vel_indices: Indices into *velocities* for each DOF.
            device: Warp device for kernel launches.
            supply_voltage: Supply voltage of each slot [V], from the
                actuator's motor (see :meth:`MotorDC.compute_supply_voltage
                <newton.actuators.MotorDC.compute_supply_voltage>`). Shape
                ``(N,)``.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement modify_forces")
