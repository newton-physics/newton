# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar

import warp as wp


class ClampingBase:
    """Base class for actuator output effort clamping.

    Clamping stages are stacked on top of a drive to constrain
    actuator output effort — symmetric limits, velocity-dependent
    saturation, position-dependent curves, etc.  They read from a
    source effort buffer and write bounded values to a destination buffer.

    **Validation contract:**  :meth:`resolve_arguments` validates scalar
    parameter values before they are batched into Warp arrays.
    ``__init__`` receives pre-built arrays and validates shapes only —
    reading back array contents would force a synchronous device-to-host
    copy on every construction.
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

    evaluate_clamp: ClassVar[wp.Function | None] = None
    """``@wp.func`` bounding effort inside the implicit solve, at the predicted
    end-of-step state. ``None`` means the clamp has no implicit form::

        evaluate_clamp(value, q, qd, params: wp.array2d[float], i: int, base: int) -> wp.float64

    ``params[i, base:]`` holds this clamp's parameters; see :meth:`bind_params`.

    Clamps apply in list order, innermost first. Order matters only when a
    clamp's feasible interval excludes zero, which :class:`ClampingDCMotor` can
    do above its velocity limit.
    """

    def param_width(self) -> int:
        """Columns this clamp occupies in the packed parameter array."""
        return 0

    def bind_params(self, block: wp.array2d[float]) -> None:
        """Fill *block* with this clamp's parameters and wire attributes to it.

        ``block`` is this clamp's ``(N, param_width())`` slice of the effort
        mode's packed array. Re-pointing the user-facing arrays (e.g.
        ``clamp.max_effort``) at its columns keeps later writes visible to the
        solve kernel.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support implicit actuation")

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
        """Read effort from *src*, apply clamping, write to *dst*.

        When src and dst are the same array, this is an in-place update.
        The Actuator uses different arrays for the first clamping
        (to preserve the raw drive output) and the same array
        for subsequent clampings.

        Args:
            src_forces: Input effort buffer [N or N·m] to read. Shape ``(N,)``.
            dst_forces: Output effort buffer [N or N·m] to write. Shape ``(N,)``.
            positions: Joint positions [m or rad].
            velocities: Joint velocities [m/s or rad/s].
            pos_indices: Indices into *positions* for each DOF.
            vel_indices: Indices into *velocities* for each DOF.
            device: Warp device for kernel launches.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement modify_forces")
