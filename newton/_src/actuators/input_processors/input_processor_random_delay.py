# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar

import numpy as np
import warp as wp

from .base import InputProcessorBase

_SHARED_STATE_FIELDS = ("lag_keys", "update_period", "episode")


@wp.func
def _lag_stream(episode_seed: wp.int32, key: wp.uint32) -> wp.uint32:
    return wp.rand_init(episode_seed, wp.int32(key))


@wp.kernel
def _random_delay_buffer_state_kernel(
    target_pos_global: wp.array[float],
    target_vel_global: wp.array[float],
    feedforward_global: wp.array[float],
    pos_indices: wp.array[wp.uint32],
    vel_indices: wp.array[wp.uint32],
    buf_depth: int,
    current_buffer_pos: wp.array2d[float],
    current_buffer_vel: wp.array2d[float],
    current_buffer_act: wp.array2d[float],
    current_num_pushes: wp.array[wp.int32],
    current_write_idx: wp.array[wp.int32],
    current_step_count: wp.array[wp.int32],
    current_phase: wp.array[wp.int32],
    current_seed: wp.array[wp.int32],
    drawn_lag: wp.array[wp.int32],
    next_buffer_pos: wp.array2d[float],
    next_buffer_vel: wp.array2d[float],
    next_buffer_act: wp.array2d[float],
    next_num_pushes: wp.array[wp.int32],
    next_write_idx: wp.array[wp.int32],
    next_lag: wp.array[wp.int32],
    next_step_count: wp.array[wp.int32],
    next_phase: wp.array[wp.int32],
    next_seed: wp.array[wp.int32],
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

    next_lag[i] = drawn_lag[i]
    next_step_count[i] = current_step_count[i] + 1
    next_phase[i] = current_phase[i]
    next_seed[i] = current_seed[i]

    if i == 0:
        next_write_idx[0] = write_idx


@wp.kernel
def _random_delay_read_kernel(
    min_delay: wp.array[wp.int32],
    max_delay: wp.array[wp.int32],
    hold_probability: wp.array[float],
    update_period: wp.array[wp.int32],
    lag_keys: wp.array[wp.uint32],
    num_pushes: wp.array[wp.int32],
    write_idx_arr: wp.array[wp.int32],
    buf_depth: int,
    buffer_pos: wp.array2d[float],
    buffer_vel: wp.array2d[float],
    buffer_act: wp.array2d[float],
    lag: wp.array[wp.int32],
    step_count: wp.array[wp.int32],
    phase: wp.array[wp.int32],
    seed: wp.array[wp.int32],
    current_pos: wp.array[float],
    current_vel: wp.array[float],
    current_act: wp.array[float],
    pos_indices: wp.array[wp.uint32],
    vel_indices: wp.array[wp.uint32],
    out_pos: wp.array[float],
    out_vel: wp.array[float],
    out_act: wp.array[float],
    out_lag: wp.array[wp.int32],
):
    """Draw the lag of this step, then read the delayed command inputs at that lag."""
    i = wp.tid()
    step = step_count[i]
    drawn = wp.max(lag[i], min_delay[i])
    if max_delay[i] > 0:
        draw = step == 0 or update_period[i] == 0
        if not draw:
            draw = (step + phase[i]) % update_period[i] == 0
        stream = _lag_stream(seed[i], lag_keys[i])
        rng = wp.rand_init(wp.int32(wp.randi(stream)), step)
        if draw and step > 0 and hold_probability[i] > 0.0:
            draw = wp.randf(rng) >= hold_probability[i]
        if draw:
            drawn = wp.randi(rng, min_delay[i], max_delay[i] + 1)
    out_lag[i] = drawn

    n = num_pushes[i]
    if n == 0 or drawn == 0:
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
        offset = wp.min(drawn - 1, n - 1)
        read_idx = (write_idx - offset + buf_depth) % buf_depth
        out_pos[i] = buffer_pos[read_idx, i]
        out_vel[i] = buffer_vel[read_idx, i]
        out_act[i] = buffer_act[read_idx, i]


@wp.kernel
def _random_delay_masked_reset_kernel(
    mask: wp.array[wp.bool],
    rows: int,
    buf_pos: wp.array2d[float],
    buf_vel: wp.array2d[float],
    buf_act: wp.array2d[float],
    num_pushes: wp.array[wp.int32],
    advance: int,
    base_seed: int,
    episode: wp.array[wp.int32],
    lag_keys: wp.array[wp.uint32],
    update_period: wp.array[wp.int32],
    lag: wp.array[wp.int32],
    step_count: wp.array[wp.int32],
    phase: wp.array[wp.int32],
    seed: wp.array[wp.int32],
):
    """Zero all buffer columns and push count where mask is True, and start a new random stream."""
    i = wp.tid()
    if mask:
        if not mask[i]:
            return
    for r in range(rows):
        buf_pos[r, i] = 0.0
        buf_vel[r, i] = 0.0
        buf_act[r, i] = 0.0
    num_pushes[i] = 0

    if advance == 1:
        episode[i] = episode[i] + 1
    lag[i] = 0
    step_count[i] = 0
    episode_seed = wp.int32(wp.randi(wp.rand_init(base_seed, episode[i])))
    seed[i] = episode_seed
    phase[i] = 0
    if update_period[i] > 0:
        stream = _lag_stream(episode_seed, lag_keys[i])
        wp.randi(stream)
        phase[i] = wp.randi(stream, 0, update_period[i])


class InputProcessorRandomDelay(InputProcessorBase):
    """Per-DOF command delay with a randomly redrawn lag.

    Like :class:`~newton.actuators.InputProcessorDelay`, but the lag [actuator timesteps] of
    each DOF is drawn uniformly from ``[min_delay, max_delay]`` at the start
    of each episode and redrawn during it:

    * every ``update_period`` steps, at a random phase, or every step when
      ``update_period`` is 0;
    * a due redraw is skipped with probability ``hold_probability``.

    DOFs that name the same ``lag_joint`` draw the same lag sequence, e.g. all
    servos of one robot on one communication bus.  Through
    :meth:`~newton.ModelBuilder.add_actuator`, each DOF names its own joint by
    default and draws independently.  Each :meth:`State.reset` call starts a
    new random stream for the DOFs it resets; DOFs that share a ``lag_joint``
    stay in step only if they are reset the same number of times.  When fewer
    commands than the lag have been seen, the oldest one is returned; with no
    history or a lag of 0, the current command is used.
    """

    SHARED_PARAMS: ClassVar[set[str]] = {"seed"}
    JOINT_PARAMS: ClassVar[set[str]] = {"lag_joint"}

    @dataclass
    class State(InputProcessorBase.State):
        """Circular buffer state for delayed targets, and the lag draw state."""

        buffer_pos: wp.array2d[float] | None = None
        """Delayed target positions [m or rad], shape (buf_depth, N)."""
        buffer_vel: wp.array2d[float] | None = None
        """Delayed target velocities [m/s or rad/s], shape (buf_depth, N)."""
        buffer_act: wp.array2d[float] | None = None
        """Delayed feedforward inputs [N or N·m], shape (buf_depth, N)."""
        num_pushes: wp.array[wp.int32] | None = None
        """Per-DOF count of writes since last reset, shape (N,)."""
        write_idx: wp.array[wp.int32] | None = None
        """Current write position in the circular buffer, shape (1,). Device-side for graph capture."""
        lag: wp.array[wp.int32] | None = None
        """Lag used on the previous step [actuator timesteps], shape (N,)."""
        step_count: wp.array[wp.int32] | None = None
        """Steps since the last reset, shape (N,)."""
        phase: wp.array[wp.int32] | None = None
        """Per-DOF offset of the redraw period, shape (N,)."""
        seed: wp.array[wp.int32] | None = None
        """Per-DOF random seed of the current episode, shape (N,)."""
        lag_keys: wp.array[wp.uint32] | None = None
        """Shared with the processor: random stream key of each DOF, shape (N,)."""
        update_period: wp.array[wp.int32] | None = None
        """Shared with the processor: redraw period of each DOF, shape (N,)."""
        episode: wp.array[wp.int32] | None = None
        """Shared with the processor: number of resets applied to each DOF, shape (N,)."""
        base_seed: int = 0
        """Seed the episode seeds are derived from."""

        def assign(self, other: InputProcessorRandomDelay.State) -> None:
            """Copy *other* into this state without replacing array storage.

            The arrays shared with the processor (:attr:`lag_keys`,
            :attr:`update_period`, :attr:`episode`) are not copied.

            Args:
                other: State to copy from, with matching array shapes.
            """
            for field in fields(self):
                if field.name in _SHARED_STATE_FIELDS:
                    continue
                value = getattr(other, field.name)
                if isinstance(value, wp.array):
                    getattr(self, field.name).assign(value)
                else:
                    setattr(self, field.name, value)

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            """Clear the history of the reset DOFs and start a new random stream for them.

            Args:
                mask: Boolean mask of length N. ``True`` entries have their
                    buffer columns zeroed, push count reset and lag stream
                    advanced. ``None`` resets all.
            """
            self._launch_reset(mask, advance=True)
            if mask is None:
                self.write_idx.fill_(self.buffer_pos.shape[0] - 1)

        def _launch_reset(self, mask: wp.array[wp.bool] | None, advance: bool) -> None:
            wp.launch(
                _random_delay_masked_reset_kernel,
                dim=self.buffer_pos.shape[1],
                inputs=[
                    mask,
                    self.buffer_pos.shape[0],
                    self.buffer_pos,
                    self.buffer_vel,
                    self.buffer_act,
                    self.num_pushes,
                    int(advance),
                    self.base_seed,
                    self.episode,
                    self.lag_keys,
                    self.update_period,
                    self.lag,
                    self.step_count,
                    self.phase,
                    self.seed,
                ],
                device=self.buffer_pos.device,
            )

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments with defaults.

        Args:
            args: User-provided arguments. ``max_delay`` is required.
                ``lag_joint`` defaults to the actuated joint.

        Returns:
            Complete arguments with defaults filled in.
        """
        if "max_delay" not in args:
            raise ValueError("InputProcessorRandomDelay requires 'max_delay' argument")
        min_delay = int(args.get("min_delay", 0))
        max_delay = int(args["max_delay"])
        hold_probability = float(args.get("hold_probability", 0.0))
        update_period = int(args.get("update_period", 0))
        if min_delay < 0:
            raise ValueError(f"min_delay must be >= 0, got {min_delay}")
        if max_delay < min_delay:
            raise ValueError(f"max_delay ({max_delay}) must be >= min_delay ({min_delay})")
        if not 0.0 <= hold_probability <= 1.0:
            raise ValueError(f"hold_probability must lie in [0, 1], got {hold_probability}")
        if update_period < 0:
            raise ValueError(f"update_period must be >= 0, got {update_period}")
        return {
            "min_delay": min_delay,
            "max_delay": max_delay,
            "hold_probability": hold_probability,
            "update_period": update_period,
            "lag_joint": args.get("lag_joint"),
            "seed": int(args.get("seed", 0)),
        }

    def __init__(
        self,
        max_delay: wp.array[wp.int32],
        min_delay: wp.array[wp.int32] | None = None,
        hold_probability: wp.array[float] | None = None,
        update_period: wp.array[wp.int32] | None = None,
        lag_joint_indices: wp.array[wp.uint32] | None = None,
        lag_joint_pos_indices: wp.array[wp.uint32] | None = None,
        seed: int = 0,
    ):
        """Initialize the random delay.

        Args:
            max_delay: Per-DOF largest lag [actuator timesteps], shape ``(N,)``.
            min_delay: Per-DOF smallest lag [actuator timesteps], shape ``(N,)``.
                ``None`` uses 0.
            hold_probability: Per-DOF probability of keeping the lag when a
                redraw is due, shape ``(N,)``. ``None`` uses 0.
            update_period: Per-DOF steps between redraws [actuator timesteps],
                shape ``(N,)``. 0 redraws every step. ``None`` uses 0.
            lag_joint_indices: Per-DOF index into ``joint_qd``-shaped arrays of
                the joint whose lag this DOF shares, shape ``(N,)``. DOFs with
                the same entry draw the same lag. ``None`` uses the actuator
                slot index: the DOFs of this actuator draw independently, but
                slot ``i`` of two such actuators draws the same lags.
            lag_joint_pos_indices: ``joint_q`` layout of *lag_joint_indices*;
                passed by :meth:`~newton.ModelBuilder.add_actuator` and not used.
            seed: Seed of the random lag and phase draws.
        """
        device = max_delay.device
        n = len(max_delay)
        self.max_delay = max_delay
        """Per-DOF largest lag [actuator timesteps], shape (N,)."""
        self.min_delay = min_delay if min_delay is not None else wp.zeros(n, dtype=wp.int32, device=device)
        """Per-DOF smallest lag [actuator timesteps], shape (N,)."""
        self.hold_probability = (
            hold_probability if hold_probability is not None else wp.zeros(n, dtype=wp.float32, device=device)
        )
        """Per-DOF probability of keeping the lag when a redraw is due, shape (N,)."""
        self.update_period = update_period if update_period is not None else wp.zeros(n, dtype=wp.int32, device=device)
        """Per-DOF steps between redraws [actuator timesteps], shape (N,)."""
        self.lag_joint_indices = (
            lag_joint_indices
            if lag_joint_indices is not None
            else wp.array(np.arange(n, dtype=np.uint32), device=device)
        )
        """Per-DOF key of the shared lag stream, shape (N,)."""
        self.seed = int(seed)
        """Seed of the random lag and phase draws."""
        self.buf_depth = max(int(np.max(max_delay.numpy())) if n > 0 else 0, 1)
        """Circular-buffer depth (equals the largest ``max_delay``, at least 1)."""
        self._episode = wp.zeros(n, dtype=wp.int32, device=device)
        self._num_actuators: int = 0
        self._device: wp.Device | None = None
        self._requires_grad: bool = False
        self._out_pos: wp.array[float] | None = None
        self._out_vel: wp.array[float] | None = None
        self._out_act: wp.array[float] | None = None
        self._drawn_lag: wp.array[wp.int32] | None = None
        self._sequential_indices: wp.array[wp.uint32] | None = None

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
        self._out_pos = wp.zeros(num_actuators, dtype=wp.float32, device=device, requires_grad=requires_grad)
        self._out_vel = wp.zeros(num_actuators, dtype=wp.float32, device=device, requires_grad=requires_grad)
        self._out_act = wp.zeros(num_actuators, dtype=wp.float32, device=device, requires_grad=requires_grad)
        self._drawn_lag = wp.zeros(num_actuators, dtype=wp.int32, device=device)
        self._sequential_indices = wp.array(np.arange(num_actuators, dtype=np.uint32), device=device)

    def is_stateful(self) -> bool:
        return True

    def state(self, num_actuators: int, device: wp.Device) -> InputProcessorRandomDelay.State:
        """Create a new delay state with zeroed circular buffers and a drawn lag stream.

        Args:
            num_actuators: Number of actuators (buffer width N).
            device: Warp device for buffer allocation.

        Returns:
            Freshly allocated :class:`InputProcessorRandomDelay.State`.
        """
        rg = self._requires_grad
        shape = (self.buf_depth, num_actuators)
        state = InputProcessorRandomDelay.State(
            buffer_pos=wp.zeros(shape, dtype=wp.float32, device=device, requires_grad=rg),
            buffer_vel=wp.zeros(shape, dtype=wp.float32, device=device, requires_grad=rg),
            buffer_act=wp.zeros(shape, dtype=wp.float32, device=device, requires_grad=rg),
            num_pushes=wp.zeros(num_actuators, dtype=wp.int32, device=device),
            write_idx=wp.full(1, self.buf_depth - 1, dtype=wp.int32, device=device),
            lag=wp.zeros(num_actuators, dtype=wp.int32, device=device),
            step_count=wp.zeros(num_actuators, dtype=wp.int32, device=device),
            phase=wp.zeros(num_actuators, dtype=wp.int32, device=device),
            seed=wp.zeros(num_actuators, dtype=wp.int32, device=device),
            lag_keys=self.lag_joint_indices,
            update_period=self.update_period,
            episode=self._episode,
            base_seed=self.seed,
        )
        state._launch_reset(None, advance=False)
        return state

    def get_delayed_targets(
        self,
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float] | None,
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        current_state: InputProcessorRandomDelay.State,
    ) -> tuple[wp.array[float], wp.array[float], wp.array[float]]:
        """Draw the lag of this step and read the delayed command inputs at that lag.

        The drawn lag is kept for :meth:`update_state`, which stores it in the
        next state.  It is clamped to available history (per-DOF
        ``num_pushes``).  When the buffer is empty, falls back to the current
        command inputs; when underfilled, the lag is clamped to the oldest
        available entry.

        Args:
            target_pos: Current target positions [m or rad].
            target_vel: Current target velocities [m/s or rad/s].
            feedforward: Feedforward control input [N or N·m] (may be ``None``).
            pos_indices: Indices into *target_pos* for each DOF.
            vel_indices: Indices into *target_vel* and *feedforward* for each DOF.
            current_state: InputProcessorRandomDelay state to read from.

        Returns:
            ``(delayed_pos, delayed_vel, delayed_feedforward)``.  When
            *feedforward* is ``None``, *delayed_feedforward* is all zeros.
        """
        wp.launch(
            kernel=_random_delay_read_kernel,
            dim=self._num_actuators,
            inputs=[
                self.min_delay,
                self.max_delay,
                self.hold_probability,
                self.update_period,
                self.lag_joint_indices,
                current_state.num_pushes,
                current_state.write_idx,
                self.buf_depth,
                current_state.buffer_pos,
                current_state.buffer_vel,
                current_state.buffer_act,
                current_state.lag,
                current_state.step_count,
                current_state.phase,
                current_state.seed,
                target_pos,
                target_vel,
                feedforward,
                pos_indices,
                vel_indices,
            ],
            outputs=[self._out_pos, self._out_vel, self._out_act, self._drawn_lag],
            device=self._device,
        )
        return (self._out_pos, self._out_vel, self._out_act)

    def process(
        self, inputs: InputProcessorBase.Inputs, state: InputProcessorRandomDelay.State, dt: float | None
    ) -> InputProcessorBase.Inputs:
        """Replace the command inputs with their delayed values.

        Args:
            inputs: Actuator inputs.
            state: InputProcessorRandomDelay state to read from.
            dt: Timestep [s] (unused).

        Returns:
            *inputs* with delayed ``target_pos``, ``target_vel`` and
            ``feedforward``.
        """
        target_pos, target_vel, feedforward = self.get_delayed_targets(
            inputs.target_pos,
            inputs.target_vel,
            inputs.feedforward,
            inputs.target_pos_indices,
            inputs.target_vel_indices,
            state,
        )
        return replace(
            inputs,
            target_pos=target_pos,
            target_vel=target_vel,
            feedforward=feedforward,
            target_pos_indices=self._sequential_indices,
            target_vel_indices=self._sequential_indices,
        )

    def update_state(
        self,
        inputs: InputProcessorBase.Inputs,
        current_state: InputProcessorRandomDelay.State,
        next_state: InputProcessorRandomDelay.State,
    ) -> None:
        """Push the current command inputs into the buffer.

        Args:
            inputs: the inputs this delay received in :meth:`process`.
            current_state: delay state to read from.
            next_state: delay state to write into.
        """
        self._push_targets(
            inputs.target_pos,
            inputs.target_vel,
            inputs.feedforward,
            inputs.target_pos_indices,
            inputs.target_vel_indices,
            current_state,
            next_state,
        )

    def _push_targets(
        self,
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float] | None,
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        current_state: InputProcessorRandomDelay.State,
        next_state: InputProcessorRandomDelay.State,
    ) -> None:
        """Write command inputs into the buffer and advance the write pointer.

        Args:
            target_pos: Current target positions [m or rad].
            target_vel: Current target velocities [m/s or rad/s].
            feedforward: Current feedforward input [N or N·m] (may be ``None``).
            pos_indices: Indices into *target_pos* for each DOF.
            vel_indices: Indices into *target_vel* and *feedforward* for each DOF.
            current_state: InputProcessorRandomDelay state to read from.
            next_state: InputProcessorRandomDelay state to write into.
        """
        if next_state is None:
            return

        wp.launch(
            kernel=_random_delay_buffer_state_kernel,
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
                current_state.step_count,
                current_state.phase,
                current_state.seed,
                self._drawn_lag,
            ],
            outputs=[
                next_state.buffer_pos,
                next_state.buffer_vel,
                next_state.buffer_act,
                next_state.num_pushes,
                next_state.write_idx,
                next_state.lag,
                next_state.step_count,
                next_state.phase,
                next_state.seed,
            ],
            device=self._device,
        )
