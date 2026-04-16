# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import warp as wp


class Controller:
    """Base class for controllers that compute raw forces from state error.

    Controllers are the core computation component in an actuator. They read
    positions, velocities, and targets, then write raw (unclamped) forces to
    a scratch buffer. Clamping and other post-processing is handled by
    Clamping objects composed on top.

    Subclasses must override ``compute`` and ``resolve_arguments``.

    Class Attributes:
        SHARED_PARAMS: Parameter names that are instance-level (shared across
            all DOFs). Different values require separate actuator instances.
    """

    @dataclass
    class State:
        """Base state for controllers.

        Subclass this in concrete controllers that maintain internal
        state (e.g. integral accumulators, history buffers).
        """

        def reset(self) -> None:
            """Reset state to initial values."""

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
        """Called by :class:`Actuator` after construction to set up device-specific resources.

        Override in subclasses that need to place tensors or networks
        on a specific device, or pre-compute index tensors.

        Args:
            device: Warp device to use.
            num_actuators: Number of actuators (DOFs) this controller manages.
        """
        pass

    def compute(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float] | None,
        input_indices: wp.array[wp.uint32],
        target_indices: wp.array[wp.uint32],
        forces: wp.array[float],
        num_actuators: int,
        state: Controller.State | None,
        dt: float,
        device: wp.Device | None = None,
    ) -> None:
        """Compute raw forces and write to ``forces[i]``.

        Args:
            positions: Joint positions [m or rad] (global array).
            velocities: Joint velocities [m/s or rad/s] (global array).
            target_pos: Target positions [m or rad] (global or compact array).
            target_vel: Target velocities [m/s or rad/s] (global or compact array).
            feedforward: Feedforward control input [N or N·m] (may be None).
            input_indices: Indices into positions/velocities.
            target_indices: Indices into target arrays.
            forces: Scratch buffer to write forces [N or N·m] to. Shape (N,).
            num_actuators: Number of actuators N.
            state: Controller state (None if stateless).
            dt: Timestep [s].
            device: Warp device for kernel launches.
        """
        raise NotImplementedError

    def is_stateful(self) -> bool:
        """Return True if this controller maintains internal state."""
        return False

    def is_graphable(self) -> bool:
        """Return True if compute() can be captured in a CUDA graph."""
        return True

    def state(self, num_actuators: int, device: wp.Device) -> Controller.State | None:
        """Create and return a new state object, or None if stateless."""
        return None

    def update_state(
        self,
        current_state: Controller.State,
        next_state: Controller.State,
    ) -> None:
        """Advance internal state after a compute step.

        Args:
            current_state: Current controller state.
            next_state: Next controller state to write.
        """
        pass
