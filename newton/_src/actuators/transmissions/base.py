# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar

import warp as wp


class TransmissionBase:
    """Maps the effort of each actuator onto the joints it drives.

    A transmission is the last stage of :meth:`Actuator.step()
    <newton.actuators.Actuator.step>`. The drive, clamping and motor compute one
    effort per actuator; the transmission accumulates that effort into the
    output DOFs of the actuator
    (:attr:`Actuator.effort_indices <newton.actuators.Actuator.effort_indices>`),
    e.g. through a gearbox, a tendon or a differential.

    The same map gives the motor velocity from the output joint velocities
    (:meth:`compute_velocity`), which clamps and the motor read, e.g. for
    back-EMF.

    Subclasses must override :meth:`resolve_arguments`, :meth:`compute` and
    :meth:`compute_velocity`.
    """

    SHARED_PARAMS: ClassVar[set[str]] = set()

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments with defaults.

        With :meth:`~newton.ModelBuilder.add_actuator`, every parameter not
        in :attr:`SHARED_PARAMS` holds one value per output joint, or one
        value for all of them, and is passed to the constructor as a
        ``(joints, N)`` array.

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
            num_joints: Number of output joints per actuator.
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

    def compute_velocity(
        self,
        velocities: wp.array[float],
        dof_indices: wp.array2d[wp.uint32],
        output: wp.array[float],
        device: wp.Device | None = None,
    ) -> None:
        """Write the motor velocity of each actuator into *output*.

        Args:
            velocities: Joint velocities [m/s or rad/s].
            dof_indices: Output DOF of each joint of each actuator, shape
                ``(joints, N)``.
            output: Motor velocity of each actuator [m/s or rad/s], shape
                ``(N,)``.
            device: Warp device for kernel launches.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement compute_velocity")
