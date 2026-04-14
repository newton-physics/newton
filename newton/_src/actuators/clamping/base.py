# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import warp as wp


class Clamping:
    """Base class for post-controller force clamping.

    Clamping objects are stacked on top of a controller to bound
    output forces — symmetric limits, velocity-dependent saturation,
    angle-dependent torque curves, etc. They read from a source force
    buffer and write bounded values to a destination buffer.

    For input delay, use the ``Delay`` class (passed separately to the
    Actuator, not as a Clamping).

    Class Attributes:
        SHARED_PARAMS: Parameter names that are instance-level (shared across
            all DOFs). Different values require separate actuator instances.
    """

    SHARED_PARAMS: set[str] = set()

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments with defaults.

        Args:
            args: User-provided arguments.

        Returns:
            Complete arguments with defaults filled in.
        """
        raise NotImplementedError(f"{cls.__name__} must implement resolve_arguments")

    def set_device(self, device: wp.Device) -> None:
        """Called by Actuator to set the target device.

        Override in subclasses that need to allocate device arrays
        from raw data (e.g. lookup tables).

        Args:
            device: Warp device to use.
        """
        pass

    def modify_forces(
        self,
        src_forces: wp.array,
        dst_forces: wp.array,
        positions: wp.array,
        velocities: wp.array,
        input_indices: wp.array,
        num_actuators: int,
    ) -> None:
        """Read forces from src, apply clamping, write to dst.

        When src and dst are the same array, this is an in-place update.
        The Actuator uses different arrays for the first clamping
        (to preserve the raw controller output) and the same array
        for subsequent clampings.

        Args:
            src_forces: Input force buffer to read. Shape (N,).
            dst_forces: Output force buffer to write. Shape (N,).
            positions: Joint positions (global array).
            velocities: Joint velocities (global array).
            input_indices: Indices into positions/velocities.
            num_actuators: Number of actuators N.
        """
        pass

    def is_graphable(self) -> bool:
        """Return True if all operations can be captured in a CUDA graph."""
        return True
