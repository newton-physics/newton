# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
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

    **Stateful clamping:**  A clamp whose bound depends on a quantity that
    evolves over time (e.g. motor temperature) overrides :meth:`is_stateful`,
    :meth:`state`, :meth:`prepare`, and :meth:`update_state`. The
    :class:`~newton.actuators.Actuator` holds the state in
    :attr:`Actuator.State.clamping_states <newton.actuators.Actuator.State.clamping_states>`,
    calls :meth:`prepare` before the effort is computed, and calls
    :meth:`update_state` with the final applied effort after the step.
    """

    @dataclass
    class State:
        """Base state for stateful clamps.

        Subclass this in concrete clamps that maintain internal state.
        """

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            """Reset state to initial values.

            Args:
                mask: Boolean mask of length N. ``True`` entries are reset.
                    ``None`` resets all.
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

    def is_stateful(self) -> bool:
        """Return True if this clamp maintains internal state."""
        return False

    def is_graphable(self) -> bool:
        """Return True if this clamp can be captured in a CUDA graph.

        Override to return False when :meth:`modify_forces`, :meth:`prepare`, or
        :meth:`update_state` synchronize with the host.
        """
        return True

    def state(self, num_actuators: int, device: wp.Device) -> ClampingBase.State | None:
        """Create and return a new state object, or None if stateless."""
        return None

    def prepare(
        self,
        state: ClampingBase.State,
        positions: wp.array[float],
        velocities: wp.array[float],
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        dt: float,
        device: wp.Device | None = None,
    ) -> None:
        """Refresh state-dependent bounds before the effort is computed.

        Called once per step for stateful clamps, before the drive runs, in
        both effort modes. Write the result into this clamp's own parameter
        arrays: :meth:`modify_forces` reads them in the explicit mode and the
        implicit solve reads them through :meth:`bind_params`.

        Args:
            state: Current clamp state.
            positions: Joint positions [m or rad].
            velocities: Joint velocities [m/s or rad/s].
            pos_indices: Indices into *positions* for each DOF.
            vel_indices: Indices into *velocities* for each DOF.
            dt: Timestep [s].
            device: Warp device for kernel launches.
        """

    def update_state(
        self,
        current_state: ClampingBase.State,
        next_state: ClampingBase.State,
        applied_forces: wp.array[float],
        positions: wp.array[float],
        velocities: wp.array[float],
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        dt: float,
        device: wp.Device | None = None,
    ) -> None:
        """Advance internal state after a step.

        Args:
            current_state: Current clamp state.
            next_state: Next clamp state to write.
            applied_forces: Final effort the actuator applied this step,
                after every clamp [N or N·m]. Shape ``(N,)``.
            positions: Joint positions [m or rad].
            velocities: Joint velocities [m/s or rad/s].
            pos_indices: Indices into *positions* for each DOF.
            vel_indices: Indices into *velocities* for each DOF.
            dt: Timestep [s].
            device: Warp device for kernel launches.
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
