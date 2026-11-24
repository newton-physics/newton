# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar

import warp as wp


class SensorBase:
    """Maps the joint positions of each actuator onto the positions its sensor measures.

    A sensor is the first stage of :meth:`Actuator.step()
    <newton.actuators.Actuator.step>`. It models where the position sensor
    sits, e.g. behind a passive backlash joint, and works in joint space:
    the :class:`~newton.actuators.TransmissionBase` then maps the measured
    joint positions onto the actuator position, so the drive and clamps read
    the measurement in actuator units. The sensor does not change the
    velocities: the transmission maps the true joint velocities.

    Subclasses must override :meth:`resolve_arguments` and :meth:`compute`.
    """

    SHARED_PARAMS: ClassVar[set[str]] = set()

    JOINT_PARAMS: ClassVar[set[str]] = set()
    """Names of parameters that reference a joint.

    :meth:`~newton.ModelBuilder.add_actuator` requires a joint index or a
    joint label for each of these parameters. The joint must be a single-DOF
    joint of the same type as the first joint of *index*.
    It passes the joint to the constructor as two index arrays:
    ``<name>_indices`` into ``joint_qd``-shaped arrays and
    ``<name>_pos_indices`` into ``joint_q``-shaped arrays.
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments with defaults.

        Args:
            args: User-provided arguments.

        Returns:
            Complete arguments with defaults filled in, including every
            parameter in :attr:`JOINT_PARAMS`. Return integer parameters as
            ``int`` and real parameters as ``float``:
            :meth:`~newton.ModelBuilder.finalize` stores a parameter as an
            ``int32`` array only when all its values are ``int``.
        """
        raise NotImplementedError(f"{cls.__name__} must implement resolve_arguments")

    def finalize(self, device: wp.Device, num_actuators: int, num_joints: int, requires_grad: bool = False) -> None:
        """Called by :class:`~newton.actuators.Actuator` after construction.

        Args:
            device: Warp device to use.
            num_actuators: Number of actuators.
            num_joints: Number of joints per actuator.
            requires_grad: Whether buffers the sensor allocates need gradients.
        """

    def compute(
        self,
        positions: wp.array[float],
        pos_indices: wp.array2d[wp.uint32],
        device: wp.Device | None = None,
    ) -> tuple[wp.array[float], wp.array2d[wp.uint32]]:
        """Return the measured joint positions of each actuator.

        Args:
            positions: Joint positions [m or rad].
            pos_indices: Index of each joint of each actuator into
                *positions*, shape ``(joints, N)``.
            device: Warp device for kernel launches.

        Returns:
            An array of measured joint positions [m or rad] and the index of
            each joint of each actuator into it, shape ``(joints, N)``.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement compute")

    def is_graphable(self) -> bool:
        """Return True if :meth:`compute` can be captured in a CUDA graph."""
        return True
