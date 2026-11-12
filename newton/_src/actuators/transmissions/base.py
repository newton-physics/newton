# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar

import warp as wp


class TransmissionBase:
    """Maps between the joints an actuator drives and its one actuator coordinate.

    A transmission starts and concludes :meth:`Actuator.step()
    <newton.actuators.Actuator.step>`. First, :meth:`compute_state` maps the
    joint positions and velocities of the actuator
    (:attr:`Actuator.pos_indices <newton.actuators.Actuator.pos_indices>` and
    :attr:`Actuator.indices <newton.actuators.Actuator.indices>`) onto one
    actuator position and velocity, e.g. the motor angle behind a gearbox or
    the mean finger position of a gripper. The input processors, drive,
    clamps and motor all work with these actuator coordinates. Last,
    :meth:`compute` maps the actuator effort onto the joints in
    :attr:`Actuator.effort_indices <newton.actuators.Actuator.effort_indices>`,
    by default the same DOFs.

    Both maps must be consistent, so that the actuator effort times the
    actuator velocity equals the summed joint effort times joint velocity.
    For example, with a gear ratio of 3, a joint at 0.1 rad is a motor at
    0.3 rad, and a clamp limit of 2 N·m allows 6 N·m at the joint.

    Subclasses must override :meth:`resolve_arguments`, :meth:`compute_state`
    and :meth:`compute`.
    """

    SHARED_PARAMS: ClassVar[set[str]] = set()

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments with defaults.

        With :meth:`~newton.ModelBuilder.add_actuator`, every parameter not
        in :attr:`SHARED_PARAMS` holds one value per joint, or one
        value for all of them, and is passed to the constructor as a
        ``(joints, N)`` array.

        Args:
            args: User-provided arguments.

        Returns:
            Complete arguments with defaults filled in.
        """
        raise NotImplementedError(f"{cls.__name__} must implement resolve_arguments")

    def finalize(self, device: wp.Device, num_actuators: int, num_joints: int, requires_grad: bool = False) -> None:
        """Called by :class:`~newton.actuators.Actuator` after construction.

        Args:
            device: Warp device to use.
            num_actuators: Number of actuators.
            num_joints: Number of joints per actuator.
            requires_grad: Whether buffers the transmission allocates need gradients.
        """

    def compute(
        self,
        forces: wp.array[float],
        computed_forces: wp.array[float],
        dof_indices: wp.array2d[wp.uint32],
        output: wp.array[float],
        computed_output: wp.array[float] | None,
        device: wp.Device | None = None,
    ) -> None:
        """Accumulate the mapped effort into the output arrays.

        Args:
            forces: Applied effort of each actuator [N or N·m], shape ``(N,)``.
            computed_forces: Raw pre-clamp effort of each actuator [N or N·m],
                shape ``(N,)``.
            dof_indices: Output DOF of each joint of each actuator, shape
                ``(joints, N)``.
            output: DOF-indexed array receiving the mapped applied effort.
            computed_output: DOF-indexed array receiving the mapped raw effort,
                or ``None`` to skip it.
            device: Warp device for kernel launches.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement compute")

    def compute_state(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        pos_indices: wp.array2d[wp.uint32],
        dof_indices: wp.array2d[wp.uint32],
        device: wp.Device | None = None,
    ) -> tuple[wp.array[float], wp.array[wp.uint32], wp.array[float], wp.array[wp.uint32]]:
        """Return the actuator position and velocity of each actuator.

        Args:
            positions: Joint positions [m or rad].
            velocities: Joint velocities [m/s or rad/s].
            pos_indices: Index of each joint of each actuator into
                *positions*, shape ``(joints, N)``.
            dof_indices: DOF of each joint of each actuator, shape
                ``(joints, N)``.
            device: Warp device for kernel launches.

        Returns:
            An array of actuator positions [m or rad], the index of each
            actuator into it, an array of actuator velocities [m/s or
            rad/s] and the index of each actuator into it; each index array
            has shape ``(N,)``.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement compute_state")
