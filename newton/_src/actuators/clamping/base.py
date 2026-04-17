# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar

import warp as wp


class Clamping:
    """Base class for post-controller force clamping.

    Clamping objects are stacked on top of a controller to bound
    clamp forces — symmetric limits, velocity-dependent saturation,
    angle-dependent torque curves, etc. They read from a source force
    buffer and write bounded values to a destination buffer.


    Class Attributes:
        SHARED_PARAMS: Parameter names that are instance-level (shared across
            all DOFs). Different values require separate actuator instances.
    """

    SHARED_PARAMS: ClassVar[set[str]] = set()

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

        Override in subclasses that need to move arrays to a specific device.

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
    ) -> None:
        """Read forces from src, apply clamping, write to dst.

        When src and dst are the same array, this is an in-place update.
        The Actuator uses different arrays for the first clamping
        (to preserve the raw controller output) and the same array
        for subsequent clampings.

        Args:
            src_forces: Input force buffer [N or N·m] to read. Shape (N,).
            dst_forces: Output force buffer [N or N·m] to write. Shape (N,).
            positions: Joint positions [m or rad] (global ``joint_q`` array).
            velocities: Joint velocities [m/s or rad/s] (global ``joint_qd`` array).
            pos_indices: Indices into *positions* (``joint_q`` layout).
            vel_indices: Indices into *velocities* (``joint_qd`` layout).
            device: Warp device for kernel launches.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement modify_forces")
