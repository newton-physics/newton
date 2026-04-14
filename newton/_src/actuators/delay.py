# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

import warp as wp


@wp.kernel
def _delay_buffer_state_kernel(
    target_pos_global: wp.array(dtype=float),
    target_vel_global: wp.array(dtype=float),
    control_input_global: wp.array(dtype=float),
    indices: wp.array(dtype=wp.uint32),
    copy_idx: int,
    write_idx: int,
    current_buffer_pos: wp.array2d(dtype=float),
    current_buffer_vel: wp.array2d(dtype=float),
    current_buffer_act: wp.array2d(dtype=float),
    next_buffer_pos: wp.array2d(dtype=float),
    next_buffer_vel: wp.array2d(dtype=float),
    next_buffer_act: wp.array2d(dtype=float),
):
    """Update delay circular buffer: copy missing entry, write new entry."""
    i = wp.tid()
    global_idx = indices[i]

    next_buffer_pos[copy_idx, i] = current_buffer_pos[copy_idx, i]
    next_buffer_vel[copy_idx, i] = current_buffer_vel[copy_idx, i]
    next_buffer_act[copy_idx, i] = current_buffer_act[copy_idx, i]

    next_buffer_pos[write_idx, i] = target_pos_global[global_idx]
    next_buffer_vel[write_idx, i] = target_vel_global[global_idx]

    act = float(0.0)
    if control_input_global:
        act = control_input_global[global_idx]
    next_buffer_act[write_idx, i] = act


class Delay:
    """Input delay for actuator targets.

    Delays targets by N timesteps using a circular buffer, modelling
    actuator communication or processing lag. While the buffer is
    filling, the controller produces no output.

    Passed to ``Actuator`` via the ``delay=`` parameter (not in the
    clamping list).
    """

    SHARED_PARAMS = {"delay"}

    @dataclass
    class State:
        """Circular buffer state for delayed targets."""

        buffer_pos: wp.array = None  # Shape (delay, N)
        buffer_vel: wp.array = None  # Shape (delay, N)
        buffer_act: wp.array = None  # Shape (delay, N)
        write_idx: int = 0
        is_filled: bool = False

        def reset(self) -> None:
            self.buffer_pos.zero_()
            self.buffer_vel.zero_()
            self.buffer_act.zero_()
            self.write_idx = self.buffer_pos.shape[0] - 1
            self.is_filled = False

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        if "delay" not in args:
            raise ValueError("Delay requires 'delay' argument")
        return {"delay": args["delay"]}

    def __init__(self, delay: int):
        """Initialize delay.

        Args:
            delay: Number of timesteps to delay inputs.
        """
        self.delay = delay
        self._sequential_indices: wp.array | None = None

    def set_indices(self, num_actuators: int, sequential_indices: wp.array) -> None:
        self._sequential_indices = sequential_indices

    def state(self, num_actuators: int, device: wp.Device) -> "Delay.State":
        return Delay.State(
            buffer_pos=wp.zeros((self.delay, num_actuators), dtype=wp.float32, device=device),
            buffer_vel=wp.zeros((self.delay, num_actuators), dtype=wp.float32, device=device),
            buffer_act=wp.zeros((self.delay, num_actuators), dtype=wp.float32, device=device),
            write_idx=self.delay - 1,
            is_filled=False,
        )

    def is_ready(self, current_state: "Delay.State") -> bool:
        """Return True if the buffer is filled and delayed targets are available."""
        return current_state is not None and current_state.is_filled

    def get_delayed_targets(
        self,
        act_input: wp.array | None,
        current_state: "Delay.State",
    ) -> tuple[wp.array, wp.array, wp.array | None, wp.array]:
        """Return delayed targets from the circular buffer.

        Call only when ``is_ready()`` is True.

        Returns:
            (delayed_pos, delayed_vel, delayed_act, sequential_indices)
        """
        read_idx = (current_state.write_idx + 1) % self.delay
        delayed_pos = current_state.buffer_pos[read_idx]
        delayed_vel = current_state.buffer_vel[read_idx]
        delayed_act = current_state.buffer_act[read_idx] if act_input is not None else None
        return delayed_pos, delayed_vel, delayed_act, self._sequential_indices

    def update_state(
        self,
        target_pos: wp.array,
        target_vel: wp.array,
        act_input: wp.array | None,
        input_indices: wp.array,
        num_actuators: int,
        current_state: "Delay.State",
        next_state: "Delay.State",
        dt: float,
    ) -> None:
        if next_state is None:
            return

        copy_idx = current_state.write_idx
        write_idx = (current_state.write_idx + 1) % self.delay

        wp.launch(
            kernel=_delay_buffer_state_kernel,
            dim=num_actuators,
            inputs=[
                target_pos,
                target_vel,
                act_input,
                input_indices,
                copy_idx,
                write_idx,
                current_state.buffer_pos,
                current_state.buffer_vel,
                current_state.buffer_act,
            ],
            outputs=[
                next_state.buffer_pos,
                next_state.buffer_vel,
                next_state.buffer_act,
            ],
        )

        next_state.write_idx = write_idx
        next_state.is_filled = current_state.is_filled or (write_idx == self.delay - 1)
