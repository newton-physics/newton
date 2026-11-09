# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar

import warp as wp


class StateEncoderBase:
    """Maps the state of the input joints of each actuator onto the position and velocity the drive reads.

    An encoder is the first stage of :meth:`Actuator.step()
    <newton.actuators.Actuator.step>`. It models what the position sensor
    measures, from the input joints of the actuator
    (:attr:`Actuator.pos_indices <newton.actuators.Actuator.pos_indices>`),
    e.g. a sensor mounted behind a gearbox with backlash. It is the sensing
    counterpart of a :class:`~newton.actuators.TransmissionBase`, which models
    where the actuator effort goes.

    Only the drive reads the encoder; clamps and the motor read the motor
    velocity, from the transmission.

    Subclasses must override :meth:`resolve_arguments` and :meth:`compute`.
    """

    SHARED_PARAMS: ClassVar[set[str]] = set()

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments with defaults.

        With :meth:`~newton.ModelBuilder.add_actuator`, every parameter not
        in :attr:`SHARED_PARAMS` holds one value per input joint, or one value
        for all of them, and is passed to the constructor as a ``(joints, N)``
        array.

        Args:
            args: User-provided arguments.

        Returns:
            Complete arguments with defaults filled in.
        """
        raise NotImplementedError(f"{cls.__name__} must implement resolve_arguments")

    def finalize(self, device: wp.Device, num_actuators: int, num_joints: int) -> None:
        """Called by :class:`~newton.actuators.Actuator` after construction.

        Args:
            device: Warp device to use.
            num_actuators: Number of actuators.
            num_joints: Number of input joints per actuator.
        """

    def compute(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        pos_indices: wp.array2d[wp.uint32],
        vel_indices: wp.array2d[wp.uint32],
        output_positions: wp.array[float],
        output_velocities: wp.array[float],
        device: wp.Device | None = None,
    ) -> None:
        """Write the measured position and velocity of each actuator.

        Args:
            positions: Joint positions [m or rad].
            velocities: Joint velocities [m/s or rad/s].
            pos_indices: Index of each input joint of each actuator into
                *positions*, shape ``(joints, N)``.
            vel_indices: Index of each input joint of each actuator into
                *velocities*, shape ``(joints, N)``.
            output_positions: Measured position of each actuator [m or rad],
                shape ``(N,)``.
            output_velocities: Measured velocity of each actuator
                [m/s or rad/s], shape ``(N,)``.
            device: Warp device for kernel launches.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement compute")
