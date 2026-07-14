# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar

import warp as wp


class Clamping:
    """Base class for actuator output effort clamping.

    Clamping objects are stacked on top of a controller to constrain
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
    """``@wp.func`` evaluating this clamp inside the implicit solve kernel.

    The implicit strategy composes the ``evaluate_clamp`` of every capable
    clamp into its Newton residual, so the clamp sees the *predicted*
    end-of-step state. Required signature, evaluated in ``float64``::

        evaluate_clamp(value, q, qd, params: wp.array2d[float], i: int, base: int) -> wp.float64

    where ``params[i, base:]`` holds this clamp's packed per-actuator
    parameters (see :meth:`clamp_params`). ``None`` (the default) means the
    clamp cannot run inside the solve; the implicit strategy rejects such
    clamps at install time.
    """

    def clamp_params(self) -> wp.array[float] | wp.array2d[float]:
        """Return the per-actuator parameter block for :attr:`evaluate_clamp`.

        Shape ``(N,)`` or ``(N, P)``, indexed by actuator slot; the implicit
        strategy packs the blocks of all capable clamps into one array and
        hands each clamp its column offset.
        """
        raise NotImplementedError(f"{type(self).__name__} does not expose clamp_params")

    def bind_params(self, block: wp.array2d[float]) -> None:
        """Re-point parameter attributes to views into the packed block.

        The implicit strategy copies :meth:`clamp_params` into one packed
        array and passes this clamp's ``(N, P)`` slice of it here. Override
        to replace the clamp's user-facing parameter arrays with column views
        of *block*, so that writes to them (e.g. ``clamp.max_effort``) stay
        visible to the solve kernel. The default keeps the original arrays:
        the solve then reads a frozen copy of the construction-time values.
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
        """Read effort from *src*, apply clamping, write to *dst*.

        When src and dst are the same array, this is an in-place update.
        The Actuator uses different arrays for the first clamping
        (to preserve the raw controller output) and the same array
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
