# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar

import numpy as np
import warp as wp

from .base import InputProcessorBase


@wp.func
def _lag_stream(episode_seed: wp.int32, key: wp.uint32) -> wp.uint32:
    return wp.rand_init(episode_seed, wp.int32(key))


@wp.kernel
def _random_delay_read_kernel(
    target_pos: wp.array[float],
    target_vel: wp.array[float],
    feedforward: wp.array[float],
    target_pos_indices: wp.array[wp.uint32],
    target_vel_indices: wp.array[wp.uint32],
    lag_keys: wp.array[wp.uint32],
    min_delay: wp.array[wp.int32],
    max_delay: wp.array[wp.int32],
    hold_probability: wp.array[float],
    update_period: wp.array[wp.int32],
    buffer_pos: wp.array2d[float],
    buffer_vel: wp.array2d[float],
    buffer_act: wp.array2d[float],
    num_pushes: wp.array[wp.int32],
    lag: wp.array[wp.int32],
    step_count: wp.array[wp.int32],
    phase: wp.array[wp.int32],
    seed: wp.array[wp.int32],
    out_pos: wp.array[float],
    out_vel: wp.array[float],
    out_act: wp.array[float],
    out_lag: wp.array[wp.int32],
):
    i = wp.tid()
    step = step_count[i]
    current_lag = wp.max(lag[i], min_delay[i])
    if max_delay[i] > 0:
        draw = step == 0 or update_period[i] == 0
        if not draw:
            draw = (step + phase[i]) % update_period[i] == 0
        stream = _lag_stream(seed[i], lag_keys[i])
        rng = wp.rand_init(wp.int32(wp.randi(stream)), step)
        if draw and step > 0 and hold_probability[i] > 0.0:
            draw = wp.randf(rng) >= hold_probability[i]
        if draw:
            current_lag = wp.randi(rng, min_delay[i], max_delay[i] + 1)
    out_lag[i] = current_lag

    act = float(0.0)
    if feedforward:
        act = feedforward[target_vel_indices[i]]
    n = num_pushes[i]
    if current_lag == 0 or n == 0:
        out_pos[i] = target_pos[target_pos_indices[i]]
        out_vel[i] = target_vel[target_vel_indices[i]]
        out_act[i] = act
    else:
        row = wp.min(current_lag - 1, n - 1)
        out_pos[i] = buffer_pos[row, i]
        out_vel[i] = buffer_vel[row, i]
        out_act[i] = buffer_act[row, i]


@wp.kernel
def _random_delay_push_kernel(
    target_pos: wp.array[float],
    target_vel: wp.array[float],
    feedforward: wp.array[float],
    target_pos_indices: wp.array[wp.uint32],
    target_vel_indices: wp.array[wp.uint32],
    buf_depth: int,
    buffer_pos: wp.array2d[float],
    buffer_vel: wp.array2d[float],
    buffer_act: wp.array2d[float],
    num_pushes: wp.array[wp.int32],
    step_count: wp.array[wp.int32],
    phase: wp.array[wp.int32],
    seed: wp.array[wp.int32],
    drawn_lag: wp.array[wp.int32],
    next_buffer_pos: wp.array2d[float],
    next_buffer_vel: wp.array2d[float],
    next_buffer_act: wp.array2d[float],
    next_num_pushes: wp.array[wp.int32],
    next_lag: wp.array[wp.int32],
    next_step_count: wp.array[wp.int32],
    next_phase: wp.array[wp.int32],
    next_seed: wp.array[wp.int32],
):
    i = wp.tid()
    for row in range(buf_depth - 1, 0, -1):
        next_buffer_pos[row, i] = buffer_pos[row - 1, i]
        next_buffer_vel[row, i] = buffer_vel[row - 1, i]
        next_buffer_act[row, i] = buffer_act[row - 1, i]
    next_buffer_pos[0, i] = target_pos[target_pos_indices[i]]
    next_buffer_vel[0, i] = target_vel[target_vel_indices[i]]
    act = float(0.0)
    if feedforward:
        act = feedforward[target_vel_indices[i]]
    next_buffer_act[0, i] = act
    next_num_pushes[i] = wp.min(num_pushes[i] + 1, buf_depth)
    next_lag[i] = drawn_lag[i]
    next_step_count[i] = step_count[i] + 1
    next_phase[i] = phase[i]
    next_seed[i] = seed[i]


@wp.kernel
def _advance_episode_kernel(episode: wp.array[wp.int32]):
    episode[0] = episode[0] + 1


@wp.kernel
def _random_delay_reset_kernel(
    mask: wp.array[wp.bool],
    base_seed: int,
    episode: wp.array[wp.int32],
    lag_keys: wp.array[wp.uint32],
    update_period: wp.array[wp.int32],
    buffer_pos: wp.array2d[float],
    buffer_vel: wp.array2d[float],
    buffer_act: wp.array2d[float],
    num_pushes: wp.array[wp.int32],
    lag: wp.array[wp.int32],
    step_count: wp.array[wp.int32],
    phase: wp.array[wp.int32],
    seed: wp.array[wp.int32],
):
    i = wp.tid()
    if mask:
        if not mask[i]:
            return
    for row in range(buffer_pos.shape[0]):
        buffer_pos[row, i] = 0.0
        buffer_vel[row, i] = 0.0
        buffer_act[row, i] = 0.0
    num_pushes[i] = 0
    lag[i] = 0
    step_count[i] = 0
    episode_seed = wp.int32(wp.randi(wp.rand_init(base_seed, episode[0])))
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
    new random stream; DOFs of different actuators that share a
    ``lag_joint`` stay in step only if their states are reset the same
    number of times.  When fewer commands than the lag have been seen, the oldest one
    is returned; with no history or a lag of 0, the current command is used.
    """

    SHARED_PARAMS: ClassVar[set[str]] = {"seed"}
    JOINT_PARAMS: ClassVar[set[str]] = {"lag_joint"}

    @dataclass
    class State(InputProcessorBase.State):
        """Command history and lag state."""

        buffer_pos: wp.array2d[float] | None = None
        """Past target positions [m or rad], newest first, shape (buf_depth, N)."""
        buffer_vel: wp.array2d[float] | None = None
        """Past target velocities [m/s or rad/s], newest first, shape (buf_depth, N)."""
        buffer_act: wp.array2d[float] | None = None
        """Past feedforward inputs [N or N·m], newest first, shape (buf_depth, N)."""
        num_pushes: wp.array[wp.int32] | None = None
        """Per-DOF count of stored commands since the last reset, shape (N,)."""
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
        """Shared with the processor: number of resets applied, shape (1,)."""
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
                if field.name in ("lag_keys", "update_period", "episode"):
                    continue
                value = getattr(other, field.name)
                if isinstance(value, wp.array):
                    getattr(self, field.name).assign(value)
                else:
                    setattr(self, field.name, value)

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            """Clear the history and start a new random stream.

            Args:
                mask: Boolean mask of length N. ``True`` entries are reset.
                    ``None`` resets all.
            """
            device = self.lag.device
            wp.launch(_advance_episode_kernel, dim=1, inputs=[self.episode], device=device)
            self._launch_reset(mask)

        def _launch_reset(self, mask: wp.array[wp.bool] | None) -> None:
            wp.launch(
                _random_delay_reset_kernel,
                dim=len(self.lag),
                inputs=[
                    mask,
                    self.base_seed,
                    self.episode,
                    self.lag_keys,
                    self.update_period,
                    self.buffer_pos,
                    self.buffer_vel,
                    self.buffer_act,
                    self.num_pushes,
                    self.lag,
                    self.step_count,
                    self.phase,
                    self.seed,
                ],
                device=self.lag.device,
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
        """History depth (largest ``max_delay``, at least 1)."""
        self._episode = wp.zeros(1, dtype=wp.int32, device=device)
        self._requires_grad = False
        self._num_actuators = 0
        self._device: wp.Device | None = None
        self._out_pos: wp.array[float] | None = None
        self._out_vel: wp.array[float] | None = None
        self._out_act: wp.array[float] | None = None
        self._drawn_lag: wp.array[wp.int32] | None = None
        self._sequential_indices: wp.array[wp.uint32] | None = None

    def finalize(self, device: wp.Device, num_actuators: int, requires_grad: bool = False) -> None:
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
        shape = (self.buf_depth, num_actuators)
        rg = self._requires_grad
        state = InputProcessorRandomDelay.State(
            buffer_pos=wp.zeros(shape, dtype=wp.float32, device=device, requires_grad=rg),
            buffer_vel=wp.zeros(shape, dtype=wp.float32, device=device, requires_grad=rg),
            buffer_act=wp.zeros(shape, dtype=wp.float32, device=device, requires_grad=rg),
            num_pushes=wp.zeros(num_actuators, dtype=wp.int32, device=device),
            lag=wp.zeros(num_actuators, dtype=wp.int32, device=device),
            step_count=wp.zeros(num_actuators, dtype=wp.int32, device=device),
            phase=wp.zeros(num_actuators, dtype=wp.int32, device=device),
            seed=wp.zeros(num_actuators, dtype=wp.int32, device=device),
            lag_keys=self.lag_joint_indices,
            update_period=self.update_period,
            episode=self._episode,
            base_seed=self.seed,
        )
        state._launch_reset(None)
        return state

    def process(
        self, inputs: InputProcessorBase.Inputs, state: InputProcessorRandomDelay.State, dt: float | None
    ) -> InputProcessorBase.Inputs:
        wp.launch(
            _random_delay_read_kernel,
            dim=self._num_actuators,
            inputs=[
                inputs.target_pos,
                inputs.target_vel,
                inputs.feedforward,
                inputs.target_pos_indices,
                inputs.target_vel_indices,
                self.lag_joint_indices,
                self.min_delay,
                self.max_delay,
                self.hold_probability,
                self.update_period,
                state.buffer_pos,
                state.buffer_vel,
                state.buffer_act,
                state.num_pushes,
                state.lag,
                state.step_count,
                state.phase,
                state.seed,
            ],
            outputs=[self._out_pos, self._out_vel, self._out_act, self._drawn_lag],
            device=self._device,
        )
        return replace(
            inputs,
            target_pos=self._out_pos,
            target_vel=self._out_vel,
            feedforward=self._out_act,
            target_pos_indices=self._sequential_indices,
            target_vel_indices=self._sequential_indices,
        )

    def update_state(
        self,
        inputs: InputProcessorBase.Inputs,
        current_state: InputProcessorRandomDelay.State,
        next_state: InputProcessorRandomDelay.State,
    ) -> None:
        wp.launch(
            _random_delay_push_kernel,
            dim=self._num_actuators,
            inputs=[
                inputs.target_pos,
                inputs.target_vel,
                inputs.feedforward,
                inputs.target_pos_indices,
                inputs.target_vel_indices,
                self.buf_depth,
                current_state.buffer_pos,
                current_state.buffer_vel,
                current_state.buffer_act,
                current_state.num_pushes,
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
                next_state.lag,
                next_state.step_count,
                next_state.phase,
                next_state.seed,
            ],
            device=self._device,
        )
