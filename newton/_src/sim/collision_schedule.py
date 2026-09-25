# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device-selected collision scheduling over a fixed simulation substep count."""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence

import numpy as np
import warp as wp

from .collide import CollisionPipeline
from .state import State

__all__ = ["CollisionSubstepScheduler"]


@wp.kernel(enable_backward=False)
def _find_max_point_speed(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    shape_collision_radius: wp.array[wp.float32],
    max_point_speed: wp.array[wp.float32],
    previous_shape_speed: wp.array[wp.float32],
    speed_increased: wp.array[wp.int32],
):
    """Find a conservative instantaneous speed over all moving shape points."""
    shape_id = wp.tid()
    body_id = shape_body[shape_id]
    if body_id < 0:
        return

    X_wb = body_q[body_id]
    X_ws = wp.transform_multiply(X_wb, shape_transform[shape_id])
    shape_origin_world = wp.transform_get_translation(X_ws)
    com_world = wp.transform_point(X_wb, body_com[body_id])
    twist = body_qd[body_id]
    angular_velocity = wp.spatial_bottom(twist)
    shape_origin_velocity = wp.spatial_top(twist) + wp.cross(angular_velocity, shape_origin_world - com_world)

    furthest = wp.max(wp.abs(shape_collision_aabb_lower[shape_id]), wp.abs(shape_collision_aabb_upper[shape_id]))
    angular_radius = wp.max(wp.length(furthest), shape_collision_radius[shape_id])
    point_speed_bound = wp.length(shape_origin_velocity) + wp.length(angular_velocity) * angular_radius
    wp.atomic_max(max_point_speed, 0, point_speed_bound)
    if point_speed_bound > previous_shape_speed[shape_id]:
        wp.atomic_max(speed_increased, 0, 1)
    previous_shape_speed[shape_id] = point_speed_bound


@wp.kernel(enable_backward=False)
def _update_collision_schedule(
    max_point_speed: wp.array[wp.float32],
    speed_increased: wp.array[wp.int32],
    previous_max_point_speed: wp.array[wp.float32],
    substep_dt: float,
    travel_budget: float,
    substep_index: int,
    collision_intervals: wp.array[wp.int32],
    interval_conditions: wp.array[wp.int32],
    collision_due: wp.array[wp.int32],
    collision_deadline: wp.array[wp.int32],
    travel_estimate: wp.array[wp.float32],
    interval_overflow: wp.array[wp.int32],
):
    """Accumulate relative travel and select the next collision horizon."""
    speed = max_point_speed[0]
    # Clear the reduction output for the next substep in the consuming kernel.
    max_point_speed[0] = 0.0
    increased = speed_increased[0]
    speed_increased[0] = 0
    previous_speed = previous_max_point_speed[0]
    relative_travel_per_substep = 2.0 * speed * substep_dt

    accumulated_travel = travel_estimate[0]
    if substep_index == 0:
        accumulated_travel = 0.0
        interval_overflow[0] = 0
    if substep_index > 0 and speed > previous_speed:
        # Correct the preceding estimate when its ending speed is larger than
        # its starting speed. This bounds that step by its two endpoint speeds.
        accumulated_travel += 2.0 * (speed - previous_speed) * substep_dt

    accumulated_overflow = accumulated_travel > travel_budget
    due = (
        substep_index == 0
        or substep_index >= collision_deadline[0]
        or increased > 0
        or accumulated_travel + relative_travel_per_substep > travel_budget
    )
    collision_due[0] = int(due)
    if due:
        accumulated_travel = 0.0
    travel_estimate[0] = accumulated_travel + relative_travel_per_substep
    previous_max_point_speed[0] = speed

    interval_count = collision_intervals.shape[0]
    selected_interval_index = int(0)
    for interval_index in range(interval_count):
        if float(collision_intervals[interval_index]) * relative_travel_per_substep <= travel_budget:
            selected_interval_index = interval_index
    for interval_index in range(interval_count):
        interval_conditions[interval_index] = int(interval_index == selected_interval_index)
    if due:
        # The travel budget can outlast the rounded prediction horizon. Never
        # extend an existing horizon merely because the observed speed falls.
        collision_deadline[0] = substep_index + collision_intervals[selected_interval_index]
    if relative_travel_per_substep > travel_budget or accumulated_overflow:
        interval_overflow[0] = 1


@wp.kernel(enable_backward=False)
def _check_final_substep(
    max_point_speed: wp.array[wp.float32],
    previous_max_point_speed: wp.array[wp.float32],
    travel_estimate: wp.array[wp.float32],
    interval_overflow: wp.array[wp.int32],
    substep_dt: float,
    travel_budget: float,
):
    """Check motion observed after the final solver substep."""
    speed = max_point_speed[0]
    max_point_speed[0] = 0.0
    previous_speed = previous_max_point_speed[0]
    accumulated_travel = travel_estimate[0]
    if speed > previous_speed:
        accumulated_travel += 2.0 * (speed - previous_speed) * substep_dt
    if accumulated_travel > travel_budget:
        interval_overflow[0] = 1


class CollisionSubstepScheduler:
    """Schedule collision refreshes over a fixed number of simulation substeps.

    The scheduler always invokes ``substep_callback`` exactly ``substeps``
    times per frame. It invokes ``collision_callback`` before substep zero and
    thereafter whenever the accumulated global relative-travel estimate would
    exhaust the configured budget, an individual shape's speed increases, or
    the previous collision horizon expires. The speed bound is reevaluated
    after every substep, including the final substep for overflow reporting.
    Selection stays on the device and uses :func:`warp.capture_if`, so it is
    compatible with CUDA graph capture.

    Scheduling is based on instantaneous rigid-shape velocities. It reacts to
    acceleration and impulses after observing their effect on a completed
    substep; it does not predict motion caused within the next substep. This is
    an optimization for speculative rigid contacts, not continuous collision
    detection, and sufficiently abrupt motion can still tunnel through thin
    geometry. Use enough solver substeps for the expected acceleration and
    impulses.

    :meth:`step` works both directly and during CUDA graph capture. When an
    outer capture is active, the complete fixed schedule is recorded inline.
    CUDA graph conditional nodes require CUDA 12.4 or newer.

    The two states are used as ping-pong buffers. ``substeps`` must be even so
    every frame begins and ends in ``states[0]``. Callbacks must be capture-safe
    and must preallocate all storage before the first call.

    .. experimental::

        This scheduling API may change without notice.

    Args:
        collision_pipeline: Speculative collision pipeline whose model and
            extension limit determine the schedule.
        states: Two simulation states used as input/output ping-pong buffers.
        collision_callback: Called as ``callback(state, collision_dt)`` before
            each scheduled collision refresh. ``collision_dt`` is the planned
            horizon until the next refresh [s], capped at the frame boundary;
            a later speed increase may trigger an earlier refresh.
        substep_callback: Called as ``callback(state_in, state_out, dt)`` exactly
            ``substeps`` times per frame.
        frame_dt: Fixed frame duration [s].
        substeps: Fixed positive even number of solver substeps per frame.
        max_collision_dt: Maximum time between collision refreshes [s]. The
            actual interval is rounded down to a whole number of solver
            substeps. If ``None``, refresh frequency is limited only by the
            relative-travel estimate.

    Raises:
        ValueError: If the pipeline contains particles, the states are not two
            distinct compatible rigid-body buffers, or a numeric configuration
            value is invalid.
    """

    def __init__(
        self,
        collision_pipeline: CollisionPipeline,
        states: Sequence[State],
        *,
        collision_callback: Callable[[State, float], None],
        substep_callback: Callable[[State, State, float], None],
        frame_dt: float,
        substeps: int,
        max_collision_dt: float | None = None,
    ):
        if not isinstance(collision_pipeline, CollisionPipeline):
            raise TypeError("collision_pipeline must be a CollisionPipeline")
        speculative_contact_gap_max = collision_pipeline.speculative_contact_gap_max
        if speculative_contact_gap_max is None:
            raise ValueError("collision_pipeline must have speculative contacts enabled")
        model = collision_pipeline.model
        if model.particle_count > 0:
            raise ValueError("CollisionSubstepScheduler supports rigid contacts only; particles are not supported")
        if len(states) != 2:
            raise ValueError(f"states must contain exactly two entries, got {len(states)}")
        if not callable(collision_callback) or not callable(substep_callback):
            raise TypeError("collision_callback and substep_callback must be callable")
        if not np.isfinite(frame_dt) or frame_dt <= 0.0:
            raise ValueError(f"frame_dt must be a positive finite number, got {frame_dt!r}")
        if isinstance(substeps, bool):
            raise TypeError("substeps must be an integer")
        try:
            substeps = operator.index(substeps)
        except TypeError as error:
            raise TypeError("substeps must be an integer") from error
        if substeps <= 0 or substeps % 2 != 0:
            raise ValueError(f"substeps must be a positive even integer, got {substeps}")
        travel_budget = float(speculative_contact_gap_max)
        if travel_budget <= 0.0:
            raise ValueError("adaptive collision scheduling requires a positive speculative extension")
        state_tuple = tuple(states)
        if state_tuple[0] is state_tuple[1]:
            raise ValueError("states must contain two distinct ping-pong buffers")
        for index, state in enumerate(state_tuple):
            if state.body_q is None or state.body_qd is None:
                raise ValueError(f"states[{index}] must contain body_q and body_qd")
            if state.body_q.device != model.device or state.body_qd.device != model.device:
                raise ValueError(f"states[{index}] must be allocated on the model device")
            if state.body_q.shape[0] < model.body_count or state.body_qd.shape[0] < model.body_count:
                raise ValueError(f"states[{index}] must contain all bodies in the collision pipeline model")

        self._model = model
        self._states = state_tuple
        self._collision_callback = collision_callback
        self._substep_callback = substep_callback
        self._substeps = substeps
        self._substep_dt = float(frame_dt) / substeps
        self._travel_budget = travel_budget
        if max_collision_dt is None:
            max_collision_interval = substeps
        else:
            if not np.isfinite(max_collision_dt) or max_collision_dt <= 0.0:
                raise ValueError(f"max_collision_dt must be a positive finite number or None, got {max_collision_dt!r}")
            interval_ratio = float(max_collision_dt) / self._substep_dt
            nearest_interval = round(interval_ratio)
            if np.isclose(interval_ratio, nearest_interval, rtol=1e-12, atol=0.0):
                interval_ratio = nearest_interval
            max_collision_interval = int(interval_ratio)
            if max_collision_interval < 1:
                raise ValueError(
                    f"max_collision_dt must be at least the solver substep duration {self._substep_dt}, "
                    f"got {max_collision_dt!r}"
                )
            max_collision_interval = min(max_collision_interval, substeps)
        self._collision_intervals = tuple(
            value for value in range(1, max_collision_interval + 1) if substeps % value == 0
        )

        device = model.device
        self._interval_values = wp.array(self._collision_intervals, dtype=wp.int32, device=device)
        self._interval_conditions = wp.zeros(len(self._collision_intervals), dtype=wp.int32, device=device)
        self._interval_condition_views = [
            self._interval_conditions[index : index + 1] for index in range(len(self._collision_intervals))
        ]
        self._collision_due = wp.zeros(1, dtype=wp.int32, device=device)
        self._collision_deadline = wp.zeros(1, dtype=wp.int32, device=device)
        self._max_point_speed = wp.zeros(1, dtype=wp.float32, device=device)
        self._previous_shape_speed = wp.zeros(model.shape_count, dtype=wp.float32, device=device)
        self._speed_increased = wp.zeros(1, dtype=wp.int32, device=device)
        self._previous_max_point_speed = wp.zeros(1, dtype=wp.float32, device=device)
        self._travel_estimate = wp.zeros(1, dtype=wp.float32, device=device)
        self.interval_overflow = wp.zeros(1, dtype=wp.int32, device=device)
        """Device scalar set when observed per-substep or accumulated travel exceeds the budget."""

    def _dispatch_collision(self, substep_index: int, interval_index: int | None = None) -> None:
        if interval_index is None:
            if substep_index == 0:
                # Slow frames need only this refresh. Later refreshes favor short
                # horizons so fast motion does not pay for the slow-path shortcut.
                wp.capture_if(
                    self._interval_condition_views[-1],
                    on_true=lambda: self._collision_callback(
                        self._states[0], self._collision_intervals[-1] * self._substep_dt
                    ),
                    on_false=lambda: self._dispatch_collision(substep_index, 0),
                )
                return
            interval_index = 0
        interval = min(self._collision_intervals[interval_index], self._substeps - substep_index)

        def run_collision():
            self._collision_callback(self._states[substep_index % 2], interval * self._substep_dt)

        if interval_index == len(self._collision_intervals) - 1:
            run_collision()
            return
        wp.capture_if(
            self._interval_condition_views[interval_index],
            on_true=run_collision,
            on_false=lambda: self._dispatch_collision(substep_index, interval_index + 1),
        )

    def step(self) -> None:
        """Execute or record one complete fixed-substep frame."""
        model = self._model
        for substep_index in range(self._substeps):
            state_in = substep_index % 2
            state = self._states[state_in]
            wp.launch(
                _find_max_point_speed,
                dim=model.shape_count,
                inputs=[
                    state.body_q,
                    state.body_qd,
                    model.body_com,
                    model.shape_body,
                    model.shape_transform,
                    model.shape_collision_aabb_lower,
                    model.shape_collision_aabb_upper,
                    model.shape_collision_radius,
                    self._max_point_speed,
                    self._previous_shape_speed,
                    self._speed_increased,
                ],
                device=model.device,
            )
            wp.launch(
                _update_collision_schedule,
                dim=1,
                inputs=[
                    self._max_point_speed,
                    self._speed_increased,
                    self._previous_max_point_speed,
                    self._substep_dt,
                    self._travel_budget,
                    substep_index,
                    self._interval_values,
                    self._interval_conditions,
                    self._collision_due,
                    self._collision_deadline,
                    self._travel_estimate,
                    self.interval_overflow,
                ],
                device=model.device,
            )
            if substep_index == 0:
                self._dispatch_collision(substep_index)
            else:
                wp.capture_if(
                    self._collision_due,
                    on_true=lambda substep_index=substep_index: self._dispatch_collision(substep_index),
                )
            self._substep_callback(self._states[state_in], self._states[1 - state_in], self._substep_dt)
        wp.launch(
            _find_max_point_speed,
            dim=model.shape_count,
            inputs=[
                self._states[0].body_q,
                self._states[0].body_qd,
                model.body_com,
                model.shape_body,
                model.shape_transform,
                model.shape_collision_aabb_lower,
                model.shape_collision_aabb_upper,
                model.shape_collision_radius,
                self._max_point_speed,
                self._previous_shape_speed,
                self._speed_increased,
            ],
            device=model.device,
        )
        wp.launch(
            _check_final_substep,
            dim=1,
            inputs=[
                self._max_point_speed,
                self._previous_max_point_speed,
                self._travel_estimate,
                self.interval_overflow,
                self._substep_dt,
                self._travel_budget,
            ],
            device=model.device,
        )
