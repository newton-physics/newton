# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import warp as wp


@wp.kernel
def _delay_buffer_state_kernel(
    target_pos_global: wp.array[float],
    target_vel_global: wp.array[float],
    feedforward_global: wp.array[float],
    indices: wp.array[wp.uint32],
    copy_idx: int,
    write_idx: int,
    buf_depth: int,
    current_buffer_pos: wp.array2d[float],
    current_buffer_vel: wp.array2d[float],
    current_buffer_act: wp.array2d[float],
    current_num_pushes: wp.array[int],
    next_buffer_pos: wp.array2d[float],
    next_buffer_vel: wp.array2d[float],
    next_buffer_act: wp.array2d[float],
    next_num_pushes: wp.array[int],
):
    """Update delay circular buffer: copy previous entry, write new entry, increment push count."""
    i = wp.tid()
    global_idx = indices[i]

    next_buffer_pos[copy_idx, i] = current_buffer_pos[copy_idx, i]
    next_buffer_vel[copy_idx, i] = current_buffer_vel[copy_idx, i]
    next_buffer_act[copy_idx, i] = current_buffer_act[copy_idx, i]

    next_buffer_pos[write_idx, i] = target_pos_global[global_idx]
    next_buffer_vel[write_idx, i] = target_vel_global[global_idx]

    act = float(0.0)
    if feedforward_global:
        act = feedforward_global[global_idx]
    next_buffer_act[write_idx, i] = act

    next_num_pushes[i] = wp.min(current_num_pushes[i] + 1, buf_depth)


@wp.kernel
def _delay_read_kernel(
    delays: wp.array[int],
    num_pushes: wp.array[int],
    write_idx: int,
    buf_depth: int,
    buffer_pos: wp.array2d[float],
    buffer_vel: wp.array2d[float],
    buffer_act: wp.array2d[float],
    out_pos: wp.array[float],
    out_vel: wp.array[float],
    out_act: wp.array[float],
):
    """Read per-DOF delayed targets, clamping lag to available history."""
    i = wp.tid()
    lag = wp.min(delays[i], wp.max(num_pushes[i] - 1, 0))
    read_idx = (write_idx - lag + buf_depth) % buf_depth
    out_pos[i] = buffer_pos[read_idx, i]
    out_vel[i] = buffer_vel[read_idx, i]
    out_act[i] = buffer_act[read_idx, i]


@wp.kernel
def _delay_masked_reset_kernel(
    mask: wp.array[wp.bool],
    rows: int,
    buf: wp.array2d[float],
    num_pushes: wp.array[int],
):
    """Zero buffer columns and push count where mask is True."""
    i = wp.tid()
    if mask[i]:
        for r in range(rows):
            buf[r, i] = 0.0
        num_pushes[i] = 0


class Delay:
    """Per-DOF input delay for actuator targets.

    Delays targets using a circular buffer of depth ``delay + 1``.
    Each DOF starts with a uniform lag of ``delay`` steps.  Per-DOF
    lags can be modified directly via the :attr:`delays` attribute
    (a ``wp.array[int]`` of shape ``(N,)`` with values in
    ``[1, delay]``) after :meth:`finalize` has been called.

    The delay always produces output.  When fewer than ``lag``
    entries have been written (e.g. right after reset), the lag is
    clamped to the available history so the most recent data is
    returned.

    Class Attributes:
        SHARED_PARAMS: Parameter names that are instance-level (shared across
            all DOFs). Different values require separate actuator instances.
    """

    SHARED_PARAMS: ClassVar[set[str]] = {"delay"}

    @dataclass
    class State:
        """Circular buffer state for delayed targets."""

        buffer_pos: wp.array2d[float] | None = None
        """Delayed target positions [m or rad], shape (buf_depth, N)."""
        buffer_vel: wp.array2d[float] | None = None
        """Delayed target velocities [m/s or rad/s], shape (buf_depth, N)."""
        buffer_act: wp.array2d[float] | None = None
        """Delayed feedforward inputs [N or N·m], shape (buf_depth, N)."""
        num_pushes: wp.array[int] | None = None
        """Per-DOF count of writes since last reset, shape (N,)."""
        write_idx: int = 0
        """Current write position in the circular buffer."""

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            """Reset delay buffer state.

            Args:
                mask: Boolean mask of length N. ``True`` entries have
                    their buffer columns zeroed and push count reset.
                    ``None`` resets all.
            """
            if mask is None:
                self.buffer_pos.zero_()
                self.buffer_vel.zero_()
                self.buffer_act.zero_()
                self.num_pushes.zero_()
                self.write_idx = self.buffer_pos.shape[0] - 1
            else:
                rows = self.buffer_pos.shape[0]
                for buf in (self.buffer_pos, self.buffer_vel, self.buffer_act):
                    wp.launch(
                        _delay_masked_reset_kernel,
                        dim=len(mask),
                        inputs=[mask, rows, buf, self.num_pushes],
                    )

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments with defaults.

        Args:
            args: User-provided arguments.

        Returns:
            Complete arguments with defaults filled in.
        """
        if "delay" not in args:
            raise ValueError("Delay requires 'delay' argument")
        return {"delay": args["delay"]}

    def __init__(self, delay: int):
        """Initialize delay.

        Args:
            delay: Maximum lag in timesteps (>= 1). The internal buffer
                allocates ``delay + 1`` rows to allow write-then-read
                in the same step.
        """
        if delay < 1:
            raise ValueError(f"delay must be >= 1, got {delay}")
        self.delay = delay
        self.buf_depth = delay + 1
        self.delays: wp.array[int] | None = None
        """Per-DOF delay in timesteps [1, delay], shape (N,)."""
        self._indices: wp.array[wp.uint32] | None = None
        self._num_actuators: int = 0
        self._device: wp.Device | None = None

    def finalize(self, indices: wp.array[wp.uint32], num_actuators: int) -> None:
        """Called by :class:`Actuator` after construction.

        Stores indices, allocates the per-DOF :attr:`delays` array
        (initialized to uniform ``delay``).

        Args:
            indices: DOF indices array.
            num_actuators: Number of actuators (DOFs).
        """
        self._indices = indices
        self._num_actuators = num_actuators
        self._device = indices.device
        self.delays = wp.array(np.full(num_actuators, self.delay, dtype=np.int32), dtype=int, device=self._device)

    def state(self, num_actuators: int, device: wp.Device) -> Delay.State:
        """Create a new delay state with zeroed circular buffers.

        Args:
            num_actuators: Number of actuators (buffer width N).
            device: Warp device for buffer allocation.

        Returns:
            Freshly allocated :class:`Delay.State`.
        """
        return Delay.State(
            buffer_pos=wp.zeros((self.buf_depth, num_actuators), dtype=wp.float32, device=device),
            buffer_vel=wp.zeros((self.buf_depth, num_actuators), dtype=wp.float32, device=device),
            buffer_act=wp.zeros((self.buf_depth, num_actuators), dtype=wp.float32, device=device),
            num_pushes=wp.zeros(num_actuators, dtype=int, device=device),
            write_idx=self.buf_depth - 1,
        )

    def get_delayed_targets(
        self,
        feedforward: wp.array[float] | None,
        current_state: Delay.State,
    ) -> tuple[wp.array[float], wp.array[float], wp.array[float] | None]:
        """Return per-DOF delayed targets from the circular buffer.

        Each DOF reads from its own lag offset stored in :attr:`delays`,
        clamped to available history (per-DOF ``num_pushes``).  Always
        produces valid output — no filling gate.

        Args:
            feedforward: Feedforward control input [N or N·m] (may be None).
            current_state: Delay state to read from (the just-updated state).

        Returns:
            ``(delayed_pos, delayed_vel, delayed_feedforward)`` where
            *delayed_feedforward* is ``None`` when *feedforward* is ``None``.
        """
        n = self._num_actuators
        out_pos = wp.zeros(n, dtype=wp.float32, device=self._device)
        out_vel = wp.zeros(n, dtype=wp.float32, device=self._device)
        out_act = wp.zeros(n, dtype=wp.float32, device=self._device)

        wp.launch(
            kernel=_delay_read_kernel,
            dim=n,
            inputs=[
                self.delays,
                current_state.num_pushes,
                current_state.write_idx,
                self.buf_depth,
                current_state.buffer_pos,
                current_state.buffer_vel,
                current_state.buffer_act,
            ],
            outputs=[out_pos, out_vel, out_act],
        )
        return (out_pos, out_vel, out_act if feedforward is not None else None)

    def update_state(
        self,
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float] | None,
        current_state: Delay.State,
        next_state: Delay.State,
    ) -> None:
        """Write current targets into the buffer and advance the write pointer.

        Args:
            target_pos: Current target positions [m or rad] (global array).
            target_vel: Current target velocities [m/s or rad/s] (global array).
            feedforward: Current feedforward input [N or N·m] (global array, may be None).
            current_state: Delay state to read from.
            next_state: Delay state to write into.
        """
        if next_state is None:
            return

        copy_idx = current_state.write_idx
        write_idx = (current_state.write_idx + 1) % self.buf_depth

        wp.launch(
            kernel=_delay_buffer_state_kernel,
            dim=self._num_actuators,
            inputs=[
                target_pos,
                target_vel,
                feedforward,
                self._indices,
                copy_idx,
                write_idx,
                self.buf_depth,
                current_state.buffer_pos,
                current_state.buffer_vel,
                current_state.buffer_act,
                current_state.num_pushes,
            ],
            outputs=[
                next_state.buffer_pos,
                next_state.buffer_vel,
                next_state.buffer_act,
                next_state.num_pushes,
            ],
        )

        next_state.write_idx = write_idx
