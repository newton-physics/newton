# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import warp as wp


@wp.kernel
def _delay_buffer_state_kernel(
    target_pos_global: wp.array[float],
    target_vel_global: wp.array[float],
    feedforward_global: wp.array[float],
    pos_indices: wp.array[wp.uint32],
    vel_indices: wp.array[wp.uint32],
    buf_depth: int,
    current_buffer_pos: wp.array2d[float],
    current_buffer_vel: wp.array2d[float],
    current_buffer_act: wp.array2d[float],
    current_num_pushes: wp.array[int],
    current_write_idx: wp.array[int],
    next_buffer_pos: wp.array2d[float],
    next_buffer_vel: wp.array2d[float],
    next_buffer_act: wp.array2d[float],
    next_num_pushes: wp.array[int],
    next_write_idx: wp.array[int],
):
    """Update delay circular buffer: copy previous entry, write new entry, advance write pointer."""
    i = wp.tid()
    pos_idx = pos_indices[i]
    vel_idx = vel_indices[i]

    copy_idx = current_write_idx[0]
    write_idx = (copy_idx + 1) % buf_depth

    next_buffer_pos[copy_idx, i] = current_buffer_pos[copy_idx, i]
    next_buffer_vel[copy_idx, i] = current_buffer_vel[copy_idx, i]
    next_buffer_act[copy_idx, i] = current_buffer_act[copy_idx, i]

    next_buffer_pos[write_idx, i] = target_pos_global[pos_idx]
    next_buffer_vel[write_idx, i] = target_vel_global[vel_idx]

    act = float(0.0)
    if feedforward_global:
        act = feedforward_global[vel_idx]
    next_buffer_act[write_idx, i] = act

    next_num_pushes[i] = wp.min(current_num_pushes[i] + 1, buf_depth)

    if i == 0:
        next_write_idx[0] = write_idx


@wp.kernel
def _delay_read_kernel(
    delays: wp.array[int],
    num_pushes: wp.array[int],
    write_idx_arr: wp.array[int],
    buf_depth: int,
    buffer_pos: wp.array2d[float],
    buffer_vel: wp.array2d[float],
    buffer_act: wp.array2d[float],
    current_pos: wp.array[float],
    current_vel: wp.array[float],
    current_act: wp.array[float],
    pos_indices: wp.array[wp.uint32],
    vel_indices: wp.array[wp.uint32],
    out_pos: wp.array[float],
    out_vel: wp.array[float],
    out_act: wp.array[float],
):
    """Read per-DOF delayed command inputs, falling back to current inputs when buffer is empty or delay is zero."""
    i = wp.tid()
    n = num_pushes[i]
    if n == 0 or delays[i] == 0:
        pos_idx = pos_indices[i]
        vel_idx = vel_indices[i]
        out_pos[i] = current_pos[pos_idx]
        out_vel[i] = current_vel[vel_idx]
        act = float(0.0)
        if current_act:
            act = current_act[vel_idx]
        out_act[i] = act
    else:
        write_idx = write_idx_arr[0]
        lag = wp.min(delays[i] - 1, n - 1)
        read_idx = (write_idx - lag + buf_depth) % buf_depth
        out_pos[i] = buffer_pos[read_idx, i]
        out_vel[i] = buffer_vel[read_idx, i]
        out_act[i] = buffer_act[read_idx, i]


@wp.kernel
def _delay_masked_reset_kernel(
    mask: wp.array[wp.bool],
    rows: int,
    buf_pos: wp.array2d[float],
    buf_vel: wp.array2d[float],
    buf_act: wp.array2d[float],
    num_pushes: wp.array[int],
):
    """Zero all buffer columns and push count where mask is True."""
    i = wp.tid()
    if mask[i]:
        for r in range(rows):
            buf_pos[r, i] = 0.0
            buf_vel[r, i] = 0.0
            buf_act[r, i] = 0.0
        num_pushes[i] = 0


class Delay:
    """Per-DOF command input delay for actuators.

    Delays command inputs (control targets and feedforward terms) using a
    circular buffer of depth ``max_delay``.  Each DOF has its own lag
    stored in :attr:`delay_steps` (shape ``(N,)``).  The buffer is sized
    for the maximum lag across all DOFs so that DOFs with different delay
    steps can share the same actuator group.

    The delay always produces output.  When the buffer is empty
    (e.g. right after reset) or a DOF has ``delay_steps == 0``, the
    current command inputs are used directly.  When underfilled, the lag
    is clamped to the available history so the oldest available entry is
    returned.
    """

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
        write_idx: wp.array[int] | None = None
        """Current write position in the circular buffer, shape (1,). Device-side for graph capture."""

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
                self.write_idx.fill_(self.buffer_pos.shape[0] - 1)
            else:
                rows = self.buffer_pos.shape[0]
                n = len(mask)
                wp.launch(
                    _delay_masked_reset_kernel,
                    dim=n,
                    inputs=[mask, rows, self.buffer_pos, self.buffer_vel, self.buffer_act, self.num_pushes],
                    device=self.buffer_pos.device,
                )

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments with defaults.

        Args:
            args: User-provided arguments.

        Returns:
            Complete arguments with defaults filled in.
        """
        if "delay_steps" not in args:
            raise ValueError("Delay requires 'delay_steps' argument")
        delay_steps = args["delay_steps"]
        if delay_steps < 0:
            raise ValueError(f"delay_steps must be >= 0, got {delay_steps}")
        return {"delay_steps": delay_steps}

    def __init__(self, delay_steps: wp.array[int], max_delay: int):
        """Initialize delay.

        Args:
            delay_steps: Per-DOF delay values [actuator timesteps], shape ``(N,)``.
            max_delay: Maximum delay across all DOFs.  Determines the
                circular-buffer depth.

        Raises:
            ValueError: If *max_delay* < 1.
        """
        if max_delay < 1:
            raise ValueError(f"max_delay must be >= 1, got {max_delay}")
        self.buf_depth = max_delay
        """Circular-buffer depth (equals ``max_delay``)."""
        self.delay_steps = delay_steps
        """Per-DOF delay values [actuator timesteps], shape (N,)."""
        self._num_actuators: int = 0
        self._device: wp.Device | None = None
        self._requires_grad: bool = False
        self._out_pos: wp.array[float] | None = None
        self._out_vel: wp.array[float] | None = None
        self._out_act: wp.array[float] | None = None

    def finalize(self, device: wp.Device, num_actuators: int, requires_grad: bool = False) -> None:
        """Called by :class:`Actuator` after construction.

        Args:
            device: Warp device to use.
            num_actuators: Number of actuators (DOFs).
            requires_grad: Allocate output arrays with gradient support.
        """
        self._device = device
        self._num_actuators = num_actuators
        self._requires_grad = requires_grad
        self._out_pos = wp.zeros(num_actuators, dtype=wp.float32, device=self._device, requires_grad=requires_grad)
        self._out_vel = wp.zeros(num_actuators, dtype=wp.float32, device=self._device, requires_grad=requires_grad)
        self._out_act = wp.zeros(num_actuators, dtype=wp.float32, device=self._device, requires_grad=requires_grad)

    def state(self, num_actuators: int, device: wp.Device) -> Delay.State:
        """Create a new delay state with zeroed circular buffers.

        Args:
            num_actuators: Number of actuators (buffer width N).
            device: Warp device for buffer allocation.

        Returns:
            Freshly allocated :class:`Delay.State`.
        """
        rg = self._requires_grad
        write_idx_arr = wp.full(1, self.buf_depth - 1, dtype=int, device=device)
        return Delay.State(
            buffer_pos=wp.zeros((self.buf_depth, num_actuators), dtype=wp.float32, device=device, requires_grad=rg),
            buffer_vel=wp.zeros((self.buf_depth, num_actuators), dtype=wp.float32, device=device, requires_grad=rg),
            buffer_act=wp.zeros((self.buf_depth, num_actuators), dtype=wp.float32, device=device, requires_grad=rg),
            num_pushes=wp.zeros(num_actuators, dtype=int, device=device),
            write_idx=write_idx_arr,
        )

    def get_delayed_targets(
        self,
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float] | None,
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        current_state: Delay.State,
    ) -> tuple[wp.array[float], wp.array[float], wp.array[float]]:
        """Read per-DOF delayed command inputs from the circular buffer.

        Each DOF reads from its own lag offset stored in :attr:`delay_steps`,
        clamped to available history (per-DOF ``num_pushes``).  When the
        buffer is empty, falls back to the current command inputs; when
        underfilled, the lag is clamped to the oldest available entry.

        Args:
            target_pos: Current target positions [m or rad].
            target_vel: Current target velocities [m/s or rad/s].
            feedforward: Feedforward control input [N or N·m] (may be ``None``).
            pos_indices: Indices into *target_pos* for each DOF.
            vel_indices: Indices into *target_vel* and *feedforward* for each DOF.
            current_state: Delay state to read from.

        Returns:
            ``(delayed_pos, delayed_vel, delayed_feedforward)``.  When
            *feedforward* is ``None``, *delayed_feedforward* is all zeros.
        """
        wp.launch(
            kernel=_delay_read_kernel,
            dim=self._num_actuators,
            inputs=[
                self.delay_steps,
                current_state.num_pushes,
                current_state.write_idx,  # device-side array (1,)
                self.buf_depth,
                current_state.buffer_pos,
                current_state.buffer_vel,
                current_state.buffer_act,
                target_pos,
                target_vel,
                feedforward,
                pos_indices,
                vel_indices,
            ],
            outputs=[self._out_pos, self._out_vel, self._out_act],
            device=self._device,
        )
        return (self._out_pos, self._out_vel, self._out_act)

    def update_state(
        self,
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float] | None,
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        current_state: Delay.State,
        next_state: Delay.State,
    ) -> None:
        """Write current command inputs into the buffer and advance the write pointer.

        Args:
            target_pos: Current target positions [m or rad].
            target_vel: Current target velocities [m/s or rad/s].
            feedforward: Current feedforward input [N or N·m] (may be ``None``).
            pos_indices: Indices into *target_pos* for each DOF.
            vel_indices: Indices into *target_vel* and *feedforward* for each DOF.
            current_state: Delay state to read from.
            next_state: Delay state to write into.
        """
        if next_state is None:
            return

        wp.launch(
            kernel=_delay_buffer_state_kernel,
            dim=self._num_actuators,
            inputs=[
                target_pos,
                target_vel,
                feedforward,
                pos_indices,
                vel_indices,
                self.buf_depth,
                current_state.buffer_pos,
                current_state.buffer_vel,
                current_state.buffer_act,
                current_state.num_pushes,
                current_state.write_idx,
            ],
            outputs=[
                next_state.buffer_pos,
                next_state.buffer_vel,
                next_state.buffer_act,
                next_state.num_pushes,
                next_state.write_idx,
            ],
            device=self._device,
        )
