# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Jacobi scheduling and acceleration for LOX unilateral projections."""

from __future__ import annotations

import warp as wp

from ...core.types import mat36f, mat66f, vec6f
from .contact import project_contact_coulomb_cone
from .projection import (
    PROJECTION_STATUS_VALID,
    _atomic_add_twist,
    _can_fuse_rigid_projection_by_world,
    _check_projected_twist,
    _initialize_jacobi_projection_status,
    _make_direct_projection_index,
    _make_project_contacts_kernel,
    _make_project_scalar_kernel,
    _make_rigid_projection_state,
    _project_rigid_contact_jacobi,
    _project_rigid_friction_jacobi,
    _project_rigid_limit_jacobi,
    _RigidContactProjectionData,
    _RigidProjectionState,
    _sync_threads,
    _validate_projected_twists,
    _warmstart_contacts_jacobi,
    _warmstart_frictions_jacobi,
    _warmstart_limits_jacobi,
)

__all__ = ["project_constraints_jacobi"]

wp.set_module_options({"enable_backward": False})

_JACOBI_CONTACT_BLOCK_DIM = 128
_JACOBI_CONTACT_PROJECTION_BLOCKS_PER_SM = 5
_JACOBI_WORLD_BLOCK_DIM = 128


@wp.kernel
def _project_rigid_constraints_jacobi_by_world(
    projection_iterations: wp.int32,
    warm_start: wp.bool,
    body_offset: wp.array[wp.int32],
    body_count: wp.array[wp.int32],
    world_friction_offset: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    friction_impulse_bound: wp.array[wp.float32],
    friction_delassus: wp.array[wp.float32],
    world_contact_offset: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    world_limit_offset: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    limit_bias: wp.array[wp.float32],
    limit_delassus: wp.array[wp.float32],
    friction_reaction: wp.array[wp.float32],
    limit_reaction: wp.array[wp.float32],
    contact_data: _RigidContactProjectionData,
    state: _RigidProjectionState,
):
    world, lane = wp.tid()
    if not state.world_active[world] or state.world_status[world] != PROJECTION_STATUS_VALID:
        return

    thread_count = wp.block_dim()
    local = lane
    while local < body_count[world]:
        state.twist_delta[body_offset[world] + local] = vec6f(0.0)
        local += thread_count
    _sync_threads()

    if warm_start:
        local = lane
        while local < world_friction_count[world]:
            friction = world_friction_offset[world] + local
            friction_first = friction_body_first[friction]
            friction_second = friction_body_second[friction]
            friction_impulse = friction_reaction[friction]
            if friction_first >= 0:
                _atomic_add_twist(
                    state.twist_delta,
                    friction_first,
                    friction_jacobian_first[friction] * friction_impulse,
                )
            if friction_second >= 0:
                _atomic_add_twist(
                    state.twist_delta,
                    friction_second,
                    friction_jacobian_second[friction] * friction_impulse,
                )
            local += thread_count

        local = lane
        while local < world_contact_count[world]:
            contact = world_contact_offset[world] + local
            contact_first = contact_data.body_first[contact]
            contact_second = contact_data.body_second[contact]
            contact_impulse = contact_data.reaction[contact]
            if contact_first >= 0:
                _atomic_add_twist(
                    state.twist_delta,
                    contact_first,
                    wp.transpose(contact_data.jacobian_first[contact]) @ contact_impulse,
                )
            if contact_second >= 0:
                _atomic_add_twist(
                    state.twist_delta,
                    contact_second,
                    wp.transpose(contact_data.jacobian_second[contact]) @ contact_impulse,
                )
            local += thread_count

        local = lane
        while local < world_limit_count[world]:
            limit = world_limit_offset[world] + local
            limit_first = limit_body_first[limit]
            limit_second = limit_body_second[limit]
            limit_impulse = limit_reaction[limit]
            if limit_first >= 0:
                _atomic_add_twist(
                    state.twist_delta,
                    limit_first,
                    limit_jacobian_first[limit] * limit_impulse,
                )
            if limit_second >= 0:
                _atomic_add_twist(
                    state.twist_delta,
                    limit_second,
                    limit_jacobian_second[limit] * limit_impulse,
                )
            local += thread_count

        _sync_threads()
        local = lane
        while local < body_count[world]:
            body = body_offset[world] + local
            warmstart_correction = state.inverse_weight[body] @ state.twist_delta[body]
            state.projected_twist[body] += warmstart_correction
            state.twist_delta[body] = vec6f(0.0)
            local += thread_count
        _sync_threads()
        if state.world_status[world] != PROJECTION_STATUS_VALID:
            return

    for _sweep in range(projection_iterations):
        local = lane
        while local < world_friction_count[world]:
            constraint = world_friction_offset[world] + local
            _project_rigid_friction_jacobi(
                constraint,
                world,
                0,
                friction_body_first,
                friction_body_second,
                friction_jacobian_first,
                friction_jacobian_second,
                friction_impulse_bound,
                friction_impulse_bound,
                friction_delassus,
                friction_reaction,
                state,
            )
            local += thread_count

        local = lane
        while local < world_contact_count[world]:
            contact = world_contact_offset[world] + local
            _project_rigid_contact_jacobi(contact, world, 0, contact_data, contact_data.delassus, state)
            local += thread_count

        local = lane
        while local < world_limit_count[world]:
            constraint = world_limit_offset[world] + local
            _project_rigid_limit_jacobi(
                constraint,
                world,
                0,
                limit_body_first,
                limit_body_second,
                limit_jacobian_first,
                limit_jacobian_second,
                limit_bias,
                limit_bias,
                limit_delassus,
                limit_reaction,
                state,
            )
            local += thread_count

        _sync_threads()
        local = lane
        while local < body_count[world]:
            body = body_offset[world] + local
            if state.world_status[world] == PROJECTION_STATUS_VALID:
                correction = state.inverse_weight[body] @ state.twist_delta[body]
                state.projected_twist[body] += correction
            state.twist_delta[body] = vec6f(0.0)
            local += thread_count
        _sync_threads()
        if state.world_status[world] != PROJECTION_STATUS_VALID:
            return

    # Validate once after all sweeps, including bodies without constraints.
    local = lane
    while local < body_count[world]:
        body = body_offset[world] + local
        _check_projected_twist(body, world, state.projected_twist, state.world_status)
        local += thread_count


@wp.kernel
def _apply_jacobi_twist_delta(
    body_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
    inverse_weight: wp.array[mat66f],
    twist_delta: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
):
    body = wp.tid()
    world = body_world[body]
    if world_active[world] and world_status[world] == PROJECTION_STATUS_VALID:
        correction = inverse_weight[body] @ twist_delta[body]
        projected_twist[body] += correction
    twist_delta[body] = vec6f(0.0)


@wp.kernel
def _initialize_acceleration_worlds(
    world_active: wp.array[wp.bool],
    prepared_status: wp.array[wp.int32],
    theta: wp.array[wp.float32],
    beta: wp.array[wp.float32],
    restart_dot: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
):
    world = wp.tid()
    if world_active[world]:
        theta[world] = 1.0
        beta[world] = 0.0
        restart_dot[world] = 0.0
        world_status[world] = prepared_status[world]


@wp.kernel
def _initialize_accelerated_reactions(
    friction_capacity: int,
    contact_capacity: int,
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_bound: wp.array[wp.float32],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_friction: wp.array[wp.float32],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    friction_reaction: wp.array[wp.float32],
    friction_trial: wp.array[wp.float32],
    friction_previous: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    contact_trial: wp.array[wp.vec3f],
    contact_previous: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
    limit_trial: wp.array[wp.float32],
    limit_previous: wp.array[wp.float32],
):
    constraint = wp.tid()
    if constraint < friction_capacity:
        world = friction_world[constraint]
        if friction_local[constraint] >= world_friction_count[world] or not world_active[world]:
            return
        friction_value = wp.clamp(
            friction_reaction[constraint], -friction_bound[constraint], friction_bound[constraint]
        )
        friction_reaction[constraint] = friction_value
        friction_trial[constraint] = friction_value
        friction_previous[constraint] = friction_value
        return

    constraint -= friction_capacity
    if constraint < contact_capacity:
        world = contact_world[constraint]
        if contact_local[constraint] >= world_contact_count[world] or not world_active[world]:
            return
        contact_value = wp.vec3f(0.0)
        if contact_body_first[constraint] >= 0 or contact_body_second[constraint] >= 0:
            contact_value = project_contact_coulomb_cone(contact_reaction[constraint], contact_friction[constraint])
        contact_reaction[constraint] = contact_value
        contact_trial[constraint] = contact_value
        contact_previous[constraint] = contact_value
        return

    constraint -= contact_capacity
    world = limit_world[constraint]
    if limit_local[constraint] >= world_limit_count[world] or not world_active[world]:
        return
    limit_value = wp.max(0.0, limit_reaction[constraint])
    limit_reaction[constraint] = limit_value
    limit_trial[constraint] = limit_value
    limit_previous[constraint] = limit_value


@wp.kernel
def _accumulate_rigid_restart(
    friction_capacity: int,
    contact_capacity: int,
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
    friction_reaction: wp.array[wp.float32],
    friction_trial: wp.array[wp.float32],
    friction_previous: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    contact_trial: wp.array[wp.vec3f],
    contact_previous: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
    limit_trial: wp.array[wp.float32],
    limit_previous: wp.array[wp.float32],
    restart_dot: wp.array[wp.float32],
):
    constraint = wp.tid()
    if constraint < friction_capacity:
        world = friction_world[constraint]
        if (
            friction_local[constraint] < world_friction_count[world]
            and world_active[world]
            and world_status[world] == PROJECTION_STATUS_VALID
        ):
            friction_current = friction_reaction[constraint]
            wp.atomic_add(
                restart_dot,
                world,
                (friction_current - friction_trial[constraint]) * (friction_current - friction_previous[constraint]),
            )
        return
    constraint -= friction_capacity
    if constraint < contact_capacity:
        world = contact_world[constraint]
        if (
            contact_local[constraint] < world_contact_count[world]
            and world_active[world]
            and world_status[world] == PROJECTION_STATUS_VALID
        ):
            contact_current = contact_reaction[constraint]
            wp.atomic_add(
                restart_dot,
                world,
                wp.dot(
                    contact_current - contact_trial[constraint],
                    contact_current - contact_previous[constraint],
                ),
            )
        return
    constraint -= contact_capacity
    world = limit_world[constraint]
    if (
        limit_local[constraint] < world_limit_count[world]
        and world_active[world]
        and world_status[world] == PROJECTION_STATUS_VALID
    ):
        limit_current = limit_reaction[constraint]
        wp.atomic_add(
            restart_dot,
            world,
            (limit_current - limit_trial[constraint]) * (limit_current - limit_previous[constraint]),
        )


@wp.kernel
def _finalize_acceleration(
    world_active: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
    restart_dot: wp.array[wp.float32],
    theta: wp.array[wp.float32],
    beta: wp.array[wp.float32],
):
    world = wp.tid()
    if not world_active[world]:
        return
    value = restart_dot[world]
    restart_dot[world] = 0.0
    current = theta[world]
    if (
        world_status[world] != PROJECTION_STATUS_VALID
        or not wp.isfinite(value)
        or value <= 0.0
        or not wp.isfinite(current)
        or current <= 0.0
    ):
        theta[world] = 1.0
        beta[world] = 0.0
        return
    next_theta = 2.0 * current / (wp.sqrt(current * current + 4.0) + current)
    beta[world] = current * (1.0 - current) / (current * current + next_theta)
    theta[world] = next_theta


@wp.kernel
def _extrapolate_rigid_reactions(
    friction_capacity: int,
    contact_capacity: int,
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    world_active: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
    beta: wp.array[wp.float32],
    friction_reaction: wp.array[wp.float32],
    friction_trial: wp.array[wp.float32],
    friction_previous: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    contact_trial: wp.array[wp.vec3f],
    contact_previous: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
    limit_trial: wp.array[wp.float32],
    limit_previous: wp.array[wp.float32],
    twist_delta: wp.array[vec6f],
):
    constraint = wp.tid()
    if constraint < friction_capacity:
        world = friction_world[constraint]
        if (
            friction_local[constraint] >= world_friction_count[world]
            or not world_active[world]
            or world_status[world] != PROJECTION_STATUS_VALID
        ):
            return
        friction_current = friction_reaction[constraint]
        friction_extrapolated = friction_current + beta[world] * (friction_current - friction_previous[constraint])
        friction_delta = friction_extrapolated - friction_current
        friction_previous[constraint] = friction_current
        friction_trial[constraint] = friction_extrapolated
        friction_reaction[constraint] = friction_extrapolated
        first = friction_body_first[constraint]
        second = friction_body_second[constraint]
        _atomic_add_twist(twist_delta, first, friction_jacobian_first[constraint] * friction_delta)
        _atomic_add_twist(twist_delta, second, friction_jacobian_second[constraint] * friction_delta)
        return

    constraint -= friction_capacity
    if constraint < contact_capacity:
        world = contact_world[constraint]
        if (
            contact_local[constraint] >= world_contact_count[world]
            or not world_active[world]
            or world_status[world] != PROJECTION_STATUS_VALID
        ):
            return
        contact_current = contact_reaction[constraint]
        contact_extrapolated = contact_current + beta[world] * (contact_current - contact_previous[constraint])
        contact_delta = contact_extrapolated - contact_current
        contact_previous[constraint] = contact_current
        contact_trial[constraint] = contact_extrapolated
        contact_reaction[constraint] = contact_extrapolated
        first = contact_body_first[constraint]
        second = contact_body_second[constraint]
        if first >= 0:
            _atomic_add_twist(twist_delta, first, wp.transpose(contact_jacobian_first[constraint]) @ contact_delta)
        if second >= 0:
            _atomic_add_twist(twist_delta, second, wp.transpose(contact_jacobian_second[constraint]) @ contact_delta)
        return

    constraint -= contact_capacity
    world = limit_world[constraint]
    if (
        limit_local[constraint] >= world_limit_count[world]
        or not world_active[world]
        or world_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    limit_current = limit_reaction[constraint]
    limit_extrapolated = limit_current + beta[world] * (limit_current - limit_previous[constraint])
    limit_delta = limit_extrapolated - limit_current
    limit_previous[constraint] = limit_current
    limit_trial[constraint] = limit_extrapolated
    limit_reaction[constraint] = limit_extrapolated
    first = limit_body_first[constraint]
    second = limit_body_second[constraint]
    _atomic_add_twist(twist_delta, first, limit_jacobian_first[constraint] * limit_delta)
    _atomic_add_twist(twist_delta, second, limit_jacobian_second[constraint] * limit_delta)


def _apply_jacobi_delta(
    body_world,
    world_active,
    world_status,
    inverse_weight,
    twist_delta,
    projected_twist,
) -> None:
    wp.launch(
        _apply_jacobi_twist_delta,
        dim=projected_twist.shape[0],
        inputs=[body_world, world_active, world_status, inverse_weight, twist_delta],
        outputs=[projected_twist],
        device=projected_twist.device,
    )


def project_constraints_jacobi(
    projection_iterations: int,
    world_active: wp.array[wp.bool],
    body_world: wp.array[wp.int32],
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    friction_impulse_bound: wp.array[wp.float32],
    friction_delassus: wp.array[wp.float32],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    contact_delassus: wp.array[wp.mat33f],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    limit_bias: wp.array[wp.float32],
    limit_delassus: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    twist_delta: wp.array[vec6f],
    contact_reaction: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
    friction_reaction: wp.array[wp.float32],
    prepared_status: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
    warm_start: bool = True,
    world_body_offset: wp.array[wp.int32] | None = None,
    world_body_count: wp.array[wp.int32] | None = None,
    world_friction_offset: wp.array[wp.int32] | None = None,
    world_contact_offset: wp.array[wp.int32] | None = None,
    world_limit_offset: wp.array[wp.int32] | None = None,
    accelerated: bool = False,
    theta: wp.array[wp.float32] | None = None,
    beta: wp.array[wp.float32] | None = None,
    restart_dot: wp.array[wp.float32] | None = None,
    friction_trial: wp.array[wp.float32] | None = None,
    friction_previous: wp.array[wp.float32] | None = None,
    contact_trial: wp.array[wp.vec3f] | None = None,
    contact_previous: wp.array[wp.vec3f] | None = None,
    limit_trial: wp.array[wp.float32] | None = None,
    limit_previous: wp.array[wp.float32] | None = None,
) -> None:
    """Run mass-split Jacobi sweeps over all body-space unilaterals.

    Set ``warm_start`` to false when the projected state already contains the
    current reactions, such as for a final smoothing sweep after Gauss--Seidel.
    """
    if (
        not isinstance(projection_iterations, int)
        or isinstance(projection_iterations, bool)
        or projection_iterations < 1
    ):
        raise ValueError("projection_iterations must be an integer greater than or equal to one.")
    world_count = world_active.shape[0]
    if prepared_status.shape[0] != world_count or world_status.shape[0] != world_count:
        raise ValueError("Active, prepared-status, and status world arrays must have identical lengths.")
    if body_world.shape[0] != projected_twist.shape[0] or twist_delta.shape[0] != projected_twist.shape[0]:
        raise ValueError("Body world, projected twist, and Jacobi delta arrays must have identical lengths.")
    if not isinstance(warm_start, bool):
        raise ValueError("warm_start must be a boolean.")
    if not isinstance(accelerated, bool):
        raise ValueError("accelerated must be a boolean.")
    acceleration_arrays = (
        theta,
        beta,
        restart_dot,
        friction_trial,
        friction_previous,
        contact_trial,
        contact_previous,
        limit_trial,
        limit_previous,
    )
    if accelerated and (not warm_start or any(value is None for value in acceleration_arrays)):
        raise ValueError("Accelerated Jacobi requires warm start and all acceleration arrays.")

    contact_projection_max_blocks = 0
    if projected_twist.device.is_cuda:
        contact_projection_max_blocks = projected_twist.device.sm_count * _JACOBI_CONTACT_PROJECTION_BLOCKS_PER_SM

    use_world_projection = not accelerated and _can_fuse_rigid_projection_by_world(
        projected_twist.device,
        world_count,
        required_world_arrays=(
            world_body_offset,
            world_body_count,
            world_friction_offset,
            world_contact_offset,
            world_limit_offset,
        ),
        parallel_constraint_capacity=friction_world.shape[0] + contact_world.shape[0] + limit_world.shape[0],
        world_block_dim=_JACOBI_WORLD_BLOCK_DIM,
    )

    rigid_capacity = friction_world.shape[0] + contact_world.shape[0] + limit_world.shape[0]
    if accelerated:
        wp.launch(
            _initialize_acceleration_worlds,
            dim=world_count,
            inputs=[world_active, prepared_status],
            outputs=[theta, beta, restart_dot, world_status],
            device=projected_twist.device,
        )
        if rigid_capacity > 0:
            wp.launch(
                _initialize_accelerated_reactions,
                dim=rigid_capacity,
                inputs=[
                    friction_world.shape[0],
                    contact_world.shape[0],
                    friction_world,
                    friction_local,
                    world_friction_count,
                    friction_impulse_bound,
                    contact_world,
                    contact_local,
                    world_contact_count,
                    contact_body_first,
                    contact_body_second,
                    contact_friction,
                    limit_world,
                    limit_local,
                    world_limit_count,
                    world_active,
                ],
                outputs=[
                    friction_reaction,
                    friction_trial,
                    friction_previous,
                    contact_reaction,
                    contact_trial,
                    contact_previous,
                    limit_reaction,
                    limit_trial,
                    limit_previous,
                ],
                device=projected_twist.device,
            )
    elif warm_start:
        wp.launch(
            _initialize_jacobi_projection_status,
            dim=world_count,
            inputs=[world_active, prepared_status],
            outputs=[world_status],
            device=projected_twist.device,
        )
    if not use_world_projection:
        twist_delta.zero_()
    if warm_start and not use_world_projection and contact_world.shape[0] > 0:
        wp.launch(
            _warmstart_contacts_jacobi,
            dim=contact_world.shape[0],
            inputs=[
                contact_world,
                contact_local,
                world_active,
                prepared_status,
                world_contact_count,
                contact_body_first,
                contact_body_second,
                contact_jacobian_first,
                contact_jacobian_second,
                inverse_weight,
                False,
                contact_reaction,
            ],
            outputs=[twist_delta],
            device=projected_twist.device,
        )
    if warm_start and not use_world_projection and friction_world.shape[0] > 0:
        wp.launch(
            _warmstart_frictions_jacobi,
            dim=friction_world.shape[0],
            inputs=[
                friction_world,
                friction_local,
                world_active,
                prepared_status,
                world_friction_count,
                friction_body_first,
                friction_body_second,
                friction_jacobian_first,
                friction_jacobian_second,
                inverse_weight,
                False,
                friction_reaction,
            ],
            outputs=[twist_delta],
            device=projected_twist.device,
        )
    if warm_start and not use_world_projection and limit_world.shape[0] > 0:
        wp.launch(
            _warmstart_limits_jacobi,
            dim=limit_world.shape[0],
            inputs=[
                limit_world,
                limit_local,
                world_active,
                prepared_status,
                world_limit_count,
                limit_body_first,
                limit_body_second,
                limit_jacobian_first,
                limit_jacobian_second,
                inverse_weight,
                False,
                limit_reaction,
            ],
            outputs=[twist_delta],
            device=projected_twist.device,
        )
    if warm_start and not use_world_projection:
        _apply_jacobi_delta(
            body_world,
            world_active,
            world_status,
            inverse_weight,
            twist_delta,
            projected_twist,
        )

    projection_state = _make_rigid_projection_state(
        world_active,
        projected_twist,
        twist_delta,
        world_status,
        inverse_weight=inverse_weight,
    )
    contact_data = _RigidContactProjectionData()
    contact_data.body_first = contact_body_first
    contact_data.body_second = contact_body_second
    contact_data.jacobian_first = contact_jacobian_first
    contact_data.jacobian_second = contact_jacobian_second
    contact_data.delassus = contact_delassus
    contact_data.bias = contact_bias
    contact_data.friction = contact_friction
    contact_data.reaction = contact_reaction

    if use_world_projection:
        wp.launch(
            _project_rigid_constraints_jacobi_by_world,
            dim=(world_count, _JACOBI_WORLD_BLOCK_DIM),
            block_dim=_JACOBI_WORLD_BLOCK_DIM,
            inputs=[
                projection_iterations,
                warm_start,
                world_body_offset,
                world_body_count,
                world_friction_offset,
                world_friction_count,
                friction_body_first,
                friction_body_second,
                friction_jacobian_first,
                friction_jacobian_second,
                friction_impulse_bound,
                friction_delassus,
                world_contact_offset,
                world_contact_count,
                world_limit_offset,
                world_limit_count,
                limit_body_first,
                limit_body_second,
                limit_jacobian_first,
                limit_jacobian_second,
                limit_bias,
                limit_delassus,
            ],
            outputs=[
                friction_reaction,
                limit_reaction,
                contact_data,
                projection_state,
            ],
            device=projected_twist.device,
        )
        return

    friction_index = _make_direct_projection_index(friction_world, friction_local, world_friction_count)
    contact_index = _make_direct_projection_index(contact_world, contact_local, world_contact_count)
    limit_index = _make_direct_projection_index(limit_world, limit_local, world_limit_count)
    for _sweep in range(projection_iterations):
        if friction_world.shape[0] > 0:
            wp.launch(
                _make_project_scalar_kernel(False, False),
                dim=friction_world.shape[0],
                inputs=[
                    friction_world.shape[0],
                    0,
                    friction_index,
                    friction_body_first,
                    friction_body_second,
                    friction_jacobian_first,
                    friction_jacobian_second,
                    friction_impulse_bound,
                    friction_impulse_bound,
                    friction_delassus,
                    friction_reaction,
                    projection_state,
                ],
                device=projected_twist.device,
            )
        if contact_world.shape[0] > 0:
            contact_inputs = [
                contact_world.shape[0],
                0,
                contact_index,
                contact_data,
                projection_state,
            ]
            wp.launch(
                _make_project_contacts_kernel(False),
                dim=contact_world.shape[0],
                inputs=contact_inputs,
                device=projected_twist.device,
                max_blocks=contact_projection_max_blocks,
                block_dim=_JACOBI_CONTACT_BLOCK_DIM,
            )
        if limit_world.shape[0] > 0:
            wp.launch(
                _make_project_scalar_kernel(False, True),
                dim=limit_world.shape[0],
                inputs=[
                    limit_world.shape[0],
                    0,
                    limit_index,
                    limit_body_first,
                    limit_body_second,
                    limit_jacobian_first,
                    limit_jacobian_second,
                    limit_bias,
                    limit_bias,
                    limit_delassus,
                    limit_reaction,
                    projection_state,
                ],
                device=projected_twist.device,
            )
        _apply_jacobi_delta(
            body_world,
            world_active,
            world_status,
            inverse_weight,
            twist_delta,
            projected_twist,
        )
        if accelerated and _sweep + 1 < projection_iterations:
            if rigid_capacity > 0:
                wp.launch(
                    _accumulate_rigid_restart,
                    dim=rigid_capacity,
                    inputs=[
                        friction_world.shape[0],
                        contact_world.shape[0],
                        friction_world,
                        friction_local,
                        world_friction_count,
                        contact_world,
                        contact_local,
                        world_contact_count,
                        limit_world,
                        limit_local,
                        world_limit_count,
                        world_active,
                        world_status,
                        friction_reaction,
                        friction_trial,
                        friction_previous,
                        contact_reaction,
                        contact_trial,
                        contact_previous,
                        limit_reaction,
                        limit_trial,
                        limit_previous,
                    ],
                    outputs=[restart_dot],
                    device=projected_twist.device,
                )
            wp.launch(
                _finalize_acceleration,
                dim=world_count,
                inputs=[world_active, world_status],
                outputs=[restart_dot, theta, beta],
                device=projected_twist.device,
            )
            if rigid_capacity > 0:
                wp.launch(
                    _extrapolate_rigid_reactions,
                    dim=rigid_capacity,
                    inputs=[
                        friction_world.shape[0],
                        contact_world.shape[0],
                        friction_world,
                        friction_local,
                        world_friction_count,
                        friction_body_first,
                        friction_body_second,
                        friction_jacobian_first,
                        friction_jacobian_second,
                        contact_world,
                        contact_local,
                        world_contact_count,
                        contact_body_first,
                        contact_body_second,
                        contact_jacobian_first,
                        contact_jacobian_second,
                        limit_world,
                        limit_local,
                        world_limit_count,
                        limit_body_first,
                        limit_body_second,
                        limit_jacobian_first,
                        limit_jacobian_second,
                        world_active,
                        world_status,
                        beta,
                        friction_reaction,
                        friction_trial,
                        friction_previous,
                        contact_reaction,
                        contact_trial,
                        contact_previous,
                        limit_reaction,
                        limit_trial,
                        limit_previous,
                    ],
                    outputs=[twist_delta],
                    device=projected_twist.device,
                )
            _apply_jacobi_delta(
                body_world,
                world_active,
                world_status,
                inverse_weight,
                twist_delta,
                projected_twist,
            )

    wp.launch(
        _validate_projected_twists,
        dim=projected_twist.shape[0],
        inputs=[body_world, world_active, projected_twist],
        outputs=[world_status],
        device=projected_twist.device,
    )
