# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from typing import Any

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

        Override in subclasses that need to place tensors or networks
        on a specific device (e.g. neural-network controllers).

        Args:
            device: Warp device to use.
        """
        pass

    def set_indices(self, input_indices: wp.array, sequential_indices: wp.array) -> None:
        """Called by Actuator to provide DOF index arrays.

        Override in subclasses that need to pre-compute index tensors
        (e.g. torch index tensors for neural-network controllers).

        Args:
            input_indices: DOF indices for reading state. Shape (N,)
                for single-input or (N, M) for multi-input actuators.
            sequential_indices: Sequential indices [0..N). Shape (N,).
        """
        pass

    def compute(
        self,
        positions: wp.array,
        velocities: wp.array,
        target_pos: wp.array,
        target_vel: wp.array,
        act_input: wp.array | None,
        input_indices: wp.array,
        target_indices: wp.array,
        forces: wp.array,
        force_indices: wp.array,
        num_actuators: int,
        state: Any,
        dt: float,
    ) -> None:
        """Compute raw forces and write to ``forces[force_indices[i]]``.

        Args:
            positions: Joint positions (global array).
            velocities: Joint velocities (global array).
            target_pos: Target positions (global or compact array).
            target_vel: Target velocities (global or compact array).
            act_input: Feedforward control input (may be None).
            input_indices: Indices into positions/velocities.
            target_indices: Indices into target arrays.
            forces: Scratch buffer to write forces to. Shape (N,).
            force_indices: Indices into forces buffer (typically sequential).
            num_actuators: Number of actuators N.
            state: Controller state (None if stateless).
            dt: Timestep in seconds.
        """
        raise NotImplementedError

    def is_stateful(self) -> bool:
        """Return True if this controller maintains internal state."""
        return False

    def is_graphable(self) -> bool:
        """Return True if compute() can be captured in a CUDA graph."""
        return True

    def state(self, num_actuators: int, device: wp.Device) -> Any:
        """Create and return a new state object, or None if stateless."""
        return None

    def update_state(
        self,
        current_state: Any,
        next_state: Any,
    ) -> None:
        """Advance internal state after a compute step.

        Args:
            current_state: Current controller state.
            next_state: Next controller state to write.
        """
        pass
