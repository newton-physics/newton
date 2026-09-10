# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""LOX splitting updates and convergence reductions."""

from functools import cache

import warp as wp

from ...core.joints import JointCorrectionMode
from ...core.math import compute_body_pose_update_with_logmap
from ...core.types import mat36f, mat66f, vec6f
from ...kinematics.joints import compute_joint_pose_and_relative_motion, make_write_joint_data
from .projection import PROJECTION_STATUS_VALID

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _initialize_bodies(
    body_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    initial_twist: wp.array[vec6f],
    reset_dual: wp.bool,
    projected_twist: wp.array[vec6f],
    projected_twist_previous: wp.array[vec6f],
    global_twist: wp.array[vec6f],
    global_twist_previous: wp.array[vec6f],
    splitting_dual: wp.array[vec6f],
    splitting_dual_impulse: wp.array[vec6f],
):
    body = wp.tid()
    if world_mask and not world_mask[body_world[body]]:
        return
    value = vec6f(0.0)
    if initial_twist:
        value = initial_twist[body]
    projected_twist[body] = value
    projected_twist_previous[body] = value
    global_twist[body] = value
    global_twist_previous[body] = value
    if reset_dual:
        splitting_dual[body] = vec6f(0.0)
        splitting_dual_impulse[body] = vec6f(0.0)


@wp.kernel
def _initialize_worlds(
    world_mask: wp.array[wp.bool],
    world_active: wp.array[wp.bool],
    world_converged: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    world_iteration_limit: wp.array[wp.bool],
    iteration_count: wp.array[wp.int32],
    residual_change: wp.array[wp.float32],
    residual_split: wp.array[wp.float32],
    residual_structural: wp.array[wp.float32],
    residual_structural_projected: wp.array[wp.float32],
    residual_cross_iterate: wp.array[wp.float32],
    residual_lagged_velocity: wp.array[wp.float32],
    residual_total: wp.array[wp.float32],
    iteration_failed: wp.array[wp.int32],
):
    world = wp.tid()
    if world_mask and not world_mask[world]:
        return
    world_active[world] = True
    world_converged[world] = False
    world_failed[world] = False
    world_iteration_limit[world] = False
    iteration_count[world] = 0
    residual_change[world] = 0.0
    residual_split[world] = 0.0
    residual_structural[world] = 0.0
    residual_structural_projected[world] = 0.0
    residual_cross_iterate[world] = 0.0
    residual_lagged_velocity[world] = 0.0
    residual_total[world] = 0.0
    iteration_failed[world] = 0


@wp.kernel
def _prepare_projection(
    body_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    body_split_enabled: wp.array[wp.int32],
    global_solution: wp.array[vec6f],
    splitting_dual: wp.array[vec6f],
    global_twist_previous: wp.array[vec6f],
    global_twist: wp.array[vec6f],
    projected_twist_previous: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
):
    body = wp.tid()
    world = body_world[body]
    if not world_active[world]:
        return
    global_twist_previous[body] = global_twist[body]
    global_twist[body] = global_solution[body]
    projected_twist_previous[body] = projected_twist[body]
    if not body_split_enabled or body_split_enabled[body] != 0:
        projected_twist[body] = global_solution[body] - splitting_dual[body]
    else:
        projected_twist[body] = global_solution[body]
        splitting_dual[body] = vec6f(0.0)


@wp.kernel
def _store_dual_impulse(
    body_has_unilateral: wp.array[wp.int32],
    weight: wp.array[mat66f],
    splitting_dual: wp.array[vec6f],
    splitting_dual_impulse: wp.array[vec6f],
):
    body = wp.tid()
    if body_has_unilateral[body] != 0:
        splitting_dual_impulse[body] = weight[body] @ splitting_dual[body]
    else:
        splitting_dual[body] = vec6f(0.0)
        splitting_dual_impulse[body] = vec6f(0.0)


@wp.kernel
def _restore_dual_from_impulse(
    body_has_unilateral: wp.array[wp.int32],
    inverse_weight: wp.array[mat66f],
    splitting_dual_impulse: wp.array[vec6f],
    splitting_dual: wp.array[vec6f],
):
    body = wp.tid()
    if body_has_unilateral[body] != 0:
        splitting_dual[body] = inverse_weight[body] @ splitting_dual_impulse[body]
    else:
        splitting_dual_impulse[body] = vec6f(0.0)
        splitting_dual[body] = vec6f(0.0)


@wp.kernel
def _initialize_iteration_residuals(
    projection_status: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    iteration_count: wp.array[wp.int32],
    iteration_failed: wp.array[wp.int32],
    residual_change: wp.array[wp.float32],
    residual_split: wp.array[wp.float32],
    residual_cross_iterate: wp.array[wp.float32],
):
    world = wp.tid()
    if not world_active[world]:
        return
    iteration_count[world] += 1
    iteration_failed[world] = 0
    residual_change[world] = 0.0
    residual_split[world] = 0.0
    residual_cross_iterate[world] = 0.0
    if projection_status[world] != PROJECTION_STATUS_VALID:
        world_active[world] = False
        world_failed[world] = True


@wp.kernel
def _update_bodies_and_reduce_residuals(
    time_step: wp.array[wp.float32],
    position_tolerance: wp.float32,
    rotation_tolerance: wp.float32,
    velocity_tolerance: wp.float32,
    body_world: wp.array[wp.int32],
    global_twist_previous: wp.array[vec6f],
    global_twist: wp.array[vec6f],
    projected_twist_previous: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
    world_active: wp.array[wp.bool],
    splitting_dual: wp.array[vec6f],
    iteration_failed: wp.array[wp.int32],
    residual_change: wp.array[wp.float32],
    residual_split: wp.array[wp.float32],
    residual_cross_iterate: wp.array[wp.float32],
):
    body = wp.tid()
    world = body_world[body]
    dt = time_step[world]
    if not world_active[world]:
        return

    previous = global_twist_previous[body]
    current = global_twist[body]
    projected_previous = projected_twist_previous[body]
    projected = projected_twist[body]
    dual = splitting_dual[body]
    finite = (
        wp.isfinite(previous)
        and wp.isfinite(current)
        and wp.isfinite(projected_previous)
        and wp.isfinite(projected)
        and wp.isfinite(dual)
    )
    if not finite:
        wp.atomic_max(iteration_failed, world, 1)
        return

    linear_change = wp.float32(0.0)
    angular_change = wp.float32(0.0)
    linear_split = wp.float32(0.0)
    angular_split = wp.float32(0.0)
    linear_cross_iterate = wp.float32(0.0)
    angular_cross_iterate = wp.float32(0.0)
    for axis in range(3):
        linear_change = wp.max(linear_change, wp.abs(current[axis] - previous[axis]))
        angular_change = wp.max(angular_change, wp.abs(current[axis + 3] - previous[axis + 3]))
        linear_split = wp.max(linear_split, wp.abs(current[axis] - projected[axis]))
        angular_split = wp.max(angular_split, wp.abs(current[axis + 3] - projected[axis + 3]))
        linear_cross_iterate = wp.max(
            linear_cross_iterate,
            wp.abs(current[axis] - projected_previous[axis]),
        )
        angular_cross_iterate = wp.max(
            angular_cross_iterate,
            wp.abs(current[axis + 3] - projected_previous[axis + 3]),
        )
    change = wp.max(
        dt * linear_change / position_tolerance,
        dt * angular_change / rotation_tolerance,
    )
    split = wp.max(linear_split, angular_split) / velocity_tolerance
    cross_iterate = wp.max(
        dt * linear_cross_iterate / position_tolerance,
        dt * angular_cross_iterate / rotation_tolerance,
    )
    wp.atomic_max(residual_change, world, change)
    wp.atomic_max(residual_split, world, split)
    wp.atomic_max(residual_cross_iterate, world, cross_iterate)
    splitting_dual[body] += projected - current


@wp.kernel
def _initialize_fixed_iteration(
    projection_status: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    iteration_count: wp.array[wp.int32],
):
    world = wp.tid()
    if not world_active[world]:
        return
    iteration_count[world] += 1
    if projection_status[world] != PROJECTION_STATUS_VALID:
        world_active[world] = False
        world_failed[world] = True


@wp.kernel
def _update_fixed_iteration_bodies(
    body_world: wp.array[wp.int32],
    global_twist: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    splitting_dual: wp.array[vec6f],
):
    body = wp.tid()
    world = body_world[body]
    if not world_active[world]:
        return
    current = global_twist[body]
    projected = projected_twist[body]
    dual = splitting_dual[body]
    finite = wp.isfinite(current) and wp.isfinite(projected) and wp.isfinite(dual)
    if finite:
        splitting_dual[body] = dual + projected - current
    else:
        world_active[world] = False
        world_failed[world] = True


@wp.kernel
def _finalize_residual_iteration(
    structural_residual: wp.array[wp.float32],
    projected_structural_residual: wp.array[wp.float32],
    lagged_velocity_residual: wp.array[wp.float32],
    lagged_velocity_required: wp.array[wp.int32],
    effort_residual: wp.array[wp.float32],
    iteration_count: wp.array[wp.int32],
    iteration_failed: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_converged: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    residual_change: wp.array[wp.float32],
    residual_split: wp.array[wp.float32],
    residual_structural: wp.array[wp.float32],
    residual_structural_projected: wp.array[wp.float32],
    residual_cross_iterate: wp.array[wp.float32],
    residual_lagged_velocity: wp.array[wp.float32],
    residual_total: wp.array[wp.float32],
):
    world = wp.tid()
    if not world_active[world]:
        return
    if iteration_failed[world] != 0:
        world_active[world] = False
        world_failed[world] = True
        return

    change = residual_change[world]
    split = residual_split[world]
    cross_iterate = residual_cross_iterate[world]
    structural = wp.float32(0.0)
    if structural_residual:
        structural = structural_residual[world]
    projected_structural = wp.float32(0.0)
    if projected_structural_residual:
        projected_structural = projected_structural_residual[world]
    lagged_velocity = wp.float32(0.0)
    if lagged_velocity_residual:
        lagged_velocity = lagged_velocity_residual[world]
    total = wp.max(wp.max(change, split), wp.max(wp.max(structural, cross_iterate), lagged_velocity))
    if effort_residual:
        total = wp.max(total, effort_residual[world])
    residual_structural[world] = structural
    residual_structural_projected[world] = projected_structural
    residual_lagged_velocity[world] = lagged_velocity
    residual_total[world] = total
    lagged_velocity_is_valid = (
        not lagged_velocity_required or lagged_velocity_required[world] == 0 or iteration_count[world] >= 2
    )
    if total <= 1.0 and lagged_velocity_is_valid:
        world_active[world] = False
        world_converged[world] = True


@wp.kernel
def _mark_iteration_limit(
    world_active: wp.array[wp.bool],
    world_iteration_limit: wp.array[wp.bool],
):
    world = wp.tid()
    if world_active[world]:
        world_active[world] = False
        world_iteration_limit[world] = True


@wp.func
def _inverse_mass_quadratic_form(
    jacobian: vec6f,
    body: wp.int32,
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[wp.mat33f],
) -> wp.float32:
    if body < 0:
        return 0.0
    linear = wp.vec3f(jacobian[0], jacobian[1], jacobian[2])
    angular = wp.vec3f(jacobian[3], jacobian[4], jacobian[5])
    return inverse_mass[body] * wp.dot(linear, linear) + wp.dot(angular, inverse_inertia_world[body] @ angular)


@wp.func
def _inverse_mass_bilinear_form(
    first: vec6f,
    second: vec6f,
    body: wp.int32,
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[wp.mat33f],
) -> wp.float32:
    if body < 0:
        return 0.0
    first_linear = wp.vec3f(first[0], first[1], first[2])
    second_linear = wp.vec3f(second[0], second[1], second[2])
    first_angular = wp.vec3f(first[3], first[4], first[5])
    second_angular = wp.vec3f(second[3], second[4], second[5])
    return inverse_mass[body] * wp.dot(first_linear, second_linear) + wp.dot(
        first_angular, inverse_inertia_world[body] @ second_angular
    )


@wp.kernel
def _evaluate_lagged_scalar_velocity_consistency(
    row_world: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    world_active: wp.array[wp.bool],
    global_twist: wp.array[vec6f],
    projected_twist_previous: wp.array[vec6f],
    inverse_velocity_tolerance: wp.float32,
    world_required: wp.array[wp.int32],
    world_residual: wp.array[wp.float32],
):
    row = wp.tid()
    world = row_world[row]
    if not world_active[world]:
        return

    first = body_first[row]
    second = body_second[row]
    if first < 0 and second < 0:
        return
    value = wp.float32(0.0)
    if first >= 0:
        value += wp.dot(jacobian_first[row], global_twist[first] - projected_twist_previous[first])
    if second >= 0:
        value += wp.dot(jacobian_second[row], global_twist[second] - projected_twist_previous[second])
    wp.atomic_max(world_required, world, 1)
    wp.atomic_max(world_residual, world, wp.abs(value) * inverse_velocity_tolerance)


@wp.kernel
def _evaluate_lagged_contact_velocity_consistency(
    world_dynamic_offset: wp.array[wp.int32],
    world_dynamic_count: wp.array[wp.int32],
    dynamic_body_first: wp.array[wp.int32],
    dynamic_body_second: wp.array[wp.int32],
    dynamic_jacobian_first: wp.array[vec6f],
    dynamic_jacobian_second: wp.array[vec6f],
    world_structural_offset: wp.array[wp.int32],
    world_structural_count: wp.array[wp.int32],
    structural_body_first: wp.array[wp.int32],
    structural_body_second: wp.array[wp.int32],
    structural_jacobian_first: wp.array[vec6f],
    structural_jacobian_second: wp.array[vec6f],
    world_friction_offset: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    world_limit_offset: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    world_contact_offset: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    world_active: wp.array[wp.bool],
    global_twist: wp.array[vec6f],
    projected_twist_previous: wp.array[vec6f],
    inverse_velocity_tolerance: wp.float32,
    block_count: wp.int32,
    world_required: wp.array[wp.int32],
    world_residual: wp.array[wp.float32],
):
    world, block, lane = wp.tid()
    if not world_active[world]:
        return

    residual = wp.float32(0.0)
    if block == 0:
        scalar_local = lane
        while scalar_local < world_dynamic_count[world]:
            row = world_dynamic_offset[world] + scalar_local
            first = dynamic_body_first[row]
            second = dynamic_body_second[row]
            scalar_value = wp.float32(0.0)
            if first >= 0:
                scalar_value += wp.dot(
                    dynamic_jacobian_first[row], global_twist[first] - projected_twist_previous[first]
                )
            if second >= 0:
                scalar_value += wp.dot(
                    dynamic_jacobian_second[row], global_twist[second] - projected_twist_previous[second]
                )
            residual = wp.max(residual, wp.abs(scalar_value) * inverse_velocity_tolerance)
            scalar_local += wp.block_dim()
        scalar_local = lane
        while scalar_local < world_structural_count[world]:
            row = world_structural_offset[world] + scalar_local
            first = structural_body_first[row]
            second = structural_body_second[row]
            scalar_value = wp.float32(0.0)
            if first >= 0:
                scalar_value += wp.dot(
                    structural_jacobian_first[row], global_twist[first] - projected_twist_previous[first]
                )
            if second >= 0:
                scalar_value += wp.dot(
                    structural_jacobian_second[row], global_twist[second] - projected_twist_previous[second]
                )
            residual = wp.max(residual, wp.abs(scalar_value) * inverse_velocity_tolerance)
            scalar_local += wp.block_dim()
        scalar_local = lane
        while scalar_local < world_friction_count[world]:
            row = world_friction_offset[world] + scalar_local
            first = friction_body_first[row]
            second = friction_body_second[row]
            scalar_value = wp.float32(0.0)
            if first >= 0:
                scalar_value += wp.dot(
                    friction_jacobian_first[row], global_twist[first] - projected_twist_previous[first]
                )
            if second >= 0:
                scalar_value += wp.dot(
                    friction_jacobian_second[row], global_twist[second] - projected_twist_previous[second]
                )
            residual = wp.max(residual, wp.abs(scalar_value) * inverse_velocity_tolerance)
            scalar_local += wp.block_dim()
        scalar_local = lane
        while scalar_local < world_limit_count[world]:
            row = world_limit_offset[world] + scalar_local
            first = limit_body_first[row]
            second = limit_body_second[row]
            scalar_value = wp.float32(0.0)
            if first >= 0:
                scalar_value += wp.dot(limit_jacobian_first[row], global_twist[first] - projected_twist_previous[first])
            if second >= 0:
                scalar_value += wp.dot(
                    limit_jacobian_second[row], global_twist[second] - projected_twist_previous[second]
                )
            residual = wp.max(residual, wp.abs(scalar_value) * inverse_velocity_tolerance)
            scalar_local += wp.block_dim()

    contact_count = world_contact_count[world]
    local = block * wp.block_dim() + lane
    stride = block_count * wp.block_dim()
    while local < contact_count:
        contact = world_contact_offset[world] + local
        first = contact_body_first[contact]
        second = contact_body_second[contact]
        contact_value = wp.vec3f(0.0)
        if first >= 0:
            contact_value += contact_jacobian_first[contact] @ (global_twist[first] - projected_twist_previous[first])
        if second >= 0:
            contact_value += contact_jacobian_second[contact] @ (
                global_twist[second] - projected_twist_previous[second]
            )
        residual = wp.max(residual, wp.max(wp.abs(contact_value)) * inverse_velocity_tolerance)
        local += stride

    block_residual = wp.tile_max(wp.tile(residual))[0]
    if lane == 0:
        if block == 0:
            scalar_count = (
                world_dynamic_count[world]
                + world_structural_count[world]
                + world_friction_count[world]
                + world_limit_count[world]
            )
            if scalar_count + contact_count > 0:
                world_required[world] = 1
        if block * wp.block_dim() < contact_count or block == 0:
            wp.atomic_max(world_residual, world, block_residual)


@wp.kernel
def _promote_effort_counters(
    effort_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    counter_next: wp.array[wp.float32],
    counter_applied: wp.array[wp.float32],
):
    effort = wp.tid()
    if world_active[effort_world[effort]]:
        counter_applied[effort] = counter_next[effort]


@wp.kernel
def _clear_active_effort_residuals(
    world_active: wp.array[wp.bool],
    residual_scaled: wp.array[wp.float32],
    residual_unscaled: wp.array[wp.float32],
):
    world = wp.tid()
    if world_active[world]:
        residual_scaled[world] = 0.0
        residual_unscaled[world] = 0.0


@wp.kernel
def _update_effort_counters(
    effort_world: wp.array[wp.int32],
    effort_dynamic_row: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    effective_inertia: wp.array[wp.float32],
    projected_twist: wp.array[vec6f],
    intercept: wp.array[wp.float32],
    slope: wp.array[wp.float32],
    impulse_bound: wp.array[wp.float32],
    counter_applied: wp.array[wp.float32],
    time_step: wp.array[wp.float32],
    velocity_tolerance: wp.float32,
    counter_next: wp.array[wp.float32],
    raw_impulse: wp.array[wp.float32],
    net_applied: wp.array[wp.float32],
    net_target: wp.array[wp.float32],
    velocity: wp.array[wp.float32],
    residual: wp.array[wp.float32],
    world_residual_scaled: wp.array[wp.float32],
    world_residual_unscaled: wp.array[wp.float32],
):
    effort = wp.tid()
    world = effort_world[effort]
    if not world_active[world]:
        return

    dynamic_row = effort_dynamic_row[effort]
    current_velocity = wp.float32(0.0)
    first = body_first[dynamic_row]
    second = body_second[dynamic_row]
    if first >= 0:
        current_velocity += wp.dot(jacobian_first[dynamic_row], projected_twist[first])
    if second >= 0:
        current_velocity += wp.dot(jacobian_second[dynamic_row], projected_twist[second])

    raw = time_step[world] * (intercept[effort] - slope[effort] * current_velocity)
    bound = impulse_bound[effort]
    target = wp.clamp(raw, -bound, bound)
    next_counter = target - raw
    applied_counter = counter_applied[effort]
    defect = wp.abs(next_counter - applied_counter)
    scaled_defect = defect / (wp.max(1.0e-6, effective_inertia[dynamic_row]) * velocity_tolerance)

    counter_next[effort] = next_counter
    raw_impulse[effort] = raw
    net_applied[effort] = raw + applied_counter
    net_target[effort] = target
    velocity[effort] = current_velocity
    residual[effort] = defect
    wp.atomic_max(world_residual_scaled, world, scaled_defect)
    wp.atomic_max(world_residual_unscaled, world, defect)


@cache
def make_evaluate_candidate_structural_residual_kernel(correction: JointCorrectionMode):
    """Build a kernel that evaluates exact joint residuals at candidate poses."""

    @wp.kernel
    def _evaluate_candidate_structural_residual(
        time_step: wp.array[wp.float32],
        joint_world: wp.array[wp.int32],
        joint_dof_type: wp.array[wp.int32],
        joint_coords_offset: wp.array[wp.int32],
        joint_dofs_offset: wp.array[wp.int32],
        joint_kinematic_offset: wp.array[wp.int32],
        joint_body_first: wp.array[wp.int32],
        joint_body_second: wp.array[wp.int32],
        joint_first_position: wp.array[wp.vec3f],
        joint_second_position: wp.array[wp.vec3f],
        joint_first_orientation: wp.array[wp.mat33f],
        joint_second_orientation: wp.array[wp.mat33f],
        body_pose: wp.array[wp.transformf],
        candidate_twist: wp.array[vec6f],
        linearization_twist: wp.array[vec6f],
        previous_joint_coordinate: wp.array[wp.float32],
        world_active: wp.array[wp.bool],
        candidate_residual: wp.array[wp.float32],
        scratch_residual_velocity: wp.array[wp.float32],
        scratch_joint_coordinate: wp.array[wp.float32],
        scratch_joint_velocity: wp.array[wp.float32],
    ):
        joint = wp.tid()
        world = joint_world[joint]
        if not world_active[world]:
            return
        dt = time_step[world]

        first = joint_body_first[joint]
        second = joint_body_second[joint]
        first_pose = wp.transform_identity(dtype=wp.float32)
        if first >= 0:
            first_delta = candidate_twist[first] - linearization_twist[first]
            first_pose = compute_body_pose_update_with_logmap(
                dt,
                body_pose[first],
                wp.vec3f(first_delta[0], first_delta[1], first_delta[2]),
                wp.vec3f(first_delta[3], first_delta[4], first_delta[5]),
            )
        second_delta = candidate_twist[second] - linearization_twist[second]
        second_pose = compute_body_pose_update_with_logmap(
            dt,
            body_pose[second],
            wp.vec3f(second_delta[0], second_delta[1], second_delta[2]),
            wp.vec3f(second_delta[3], second_delta[4], second_delta[5]),
        )

        _, relative_position, relative_orientation, relative_twist = compute_joint_pose_and_relative_motion(
            first_pose,
            second_pose,
            wp.spatial_vectorf(0.0),
            wp.spatial_vectorf(0.0),
            joint_first_position[joint],
            joint_second_position[joint],
            joint_first_orientation[joint],
            joint_second_orientation[joint],
        )
        wp.static(make_write_joint_data(correction))(
            joint_dof_type[joint],
            joint_kinematic_offset[joint],
            joint_dofs_offset[joint],
            joint_coords_offset[joint],
            relative_position,
            relative_orientation,
            relative_twist,
            previous_joint_coordinate,
            candidate_residual,
            scratch_residual_velocity,
            scratch_joint_coordinate,
            scratch_joint_velocity,
        )

    return _evaluate_candidate_structural_residual


@wp.kernel
def _blend_structural_candidate_twist(
    global_twist: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
    projected_fraction: wp.float32,
    candidate_twist: wp.array[vec6f],
):
    body = wp.tid()
    candidate_twist[body] = global_twist[body] + projected_fraction * (projected_twist[body] - global_twist[body])


@wp.kernel
def _include_dynamic_compliance_in_structural_effective_mass(
    joint_body_first: wp.array[wp.int32],
    joint_body_second: wp.array[wp.int32],
    joint_structural_offset: wp.array[wp.int32],
    joint_structural_count: wp.array[wp.int32],
    joint_dynamic_offset: wp.array[wp.int32],
    joint_dynamic_count: wp.array[wp.int32],
    structural_jacobian_first: wp.array[vec6f],
    structural_jacobian_second: wp.array[vec6f],
    dynamic_jacobian_first: wp.array[vec6f],
    dynamic_jacobian_second: wp.array[vec6f],
    dynamic_effective_inertia: wp.array[wp.float32],
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[wp.mat33f],
    effective_mass: wp.array[wp.float32],
):
    joint = wp.tid()
    structural_count = joint_structural_count[joint]
    dynamic_count = joint_dynamic_count[joint]
    if structural_count == 0 or dynamic_count == 0:
        return

    first_body = joint_body_first[joint]
    second_body = joint_body_second[joint]
    dynamic_offset = joint_dynamic_offset[joint]

    # Factor the mixed dynamic-row Schur block
    #
    #   J_d M^-1 J_d^T + diag(m_j^-1).
    #
    # Keeping m_j^-1 here retains the full implicit drive compliance for
    # effort-limited drives independently of whether their multiplier is at
    # its bound.
    lower = mat66f(0.0)
    valid = wp.bool(True)
    for row in range(6):
        if row < dynamic_count:
            first_row = dynamic_jacobian_first[dynamic_offset + row]
            second_row = dynamic_jacobian_second[dynamic_offset + row]
            for col in range(6):
                if col <= row and col < dynamic_count:
                    first_col = dynamic_jacobian_first[dynamic_offset + col]
                    second_col = dynamic_jacobian_second[dynamic_offset + col]
                    value = _inverse_mass_bilinear_form(
                        first_row, first_col, first_body, inverse_mass, inverse_inertia_world
                    ) + _inverse_mass_bilinear_form(
                        second_row, second_col, second_body, inverse_mass, inverse_inertia_world
                    )
                    if row == col:
                        inertia = dynamic_effective_inertia[dynamic_offset + row]
                        if inertia > 0.0:
                            value += 1.0 / inertia
                    for inner in range(6):
                        if inner < col:
                            value -= lower[row, inner] * lower[col, inner]
                    if row == col:
                        if not wp.isfinite(value) or value <= 1.0e-12:
                            valid = False
                        elif valid:
                            lower[row, col] = wp.sqrt(value)
                    elif valid:
                        lower[row, col] = value / lower[col, col]

    if not valid:
        return

    structural_offset = joint_structural_offset[joint]
    for structural_local in range(6):
        if structural_local < structural_count:
            structural_row = structural_offset + structural_local
            first_structural = structural_jacobian_first[structural_row]
            second_structural = structural_jacobian_second[structural_row]
            inverse_effective_mass = _inverse_mass_bilinear_form(
                first_structural, first_structural, first_body, inverse_mass, inverse_inertia_world
            ) + _inverse_mass_bilinear_form(
                second_structural, second_structural, second_body, inverse_mass, inverse_inertia_world
            )

            # Subtract the response of the retained mixed drive rows:
            # J_s M^-1 J_d^T S_d^-1 J_d M^-1 J_s^T.
            forward = vec6f(0.0)
            for row in range(6):
                if row < dynamic_count:
                    coupling = _inverse_mass_bilinear_form(
                        first_structural,
                        dynamic_jacobian_first[dynamic_offset + row],
                        first_body,
                        inverse_mass,
                        inverse_inertia_world,
                    ) + _inverse_mass_bilinear_form(
                        second_structural,
                        dynamic_jacobian_second[dynamic_offset + row],
                        second_body,
                        inverse_mass,
                        inverse_inertia_world,
                    )
                    for inner in range(6):
                        if inner < row:
                            coupling -= lower[row, inner] * forward[inner]
                    forward[row] = coupling / lower[row, row]
                    inverse_effective_mass -= forward[row] * forward[row]

            effective_mass[structural_row] = 1.0 / inverse_effective_mass if inverse_effective_mass > 1.0e-12 else 0.0


@wp.kernel
def _reset_structural_multipliers_masked(
    row_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    reaction: wp.array[wp.float32],
):
    row = wp.tid()
    if world_mask[row_world[row]]:
        reaction[row] = 0.0


@wp.kernel
def _reset_effort_rows_masked(
    effort_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    effort_intercept: wp.array[wp.float32],
    effort_slope: wp.array[wp.float32],
    effort_impulse_bound: wp.array[wp.float32],
    effort_raw_impulse: wp.array[wp.float32],
    effort_counter_applied: wp.array[wp.float32],
    effort_counter_next: wp.array[wp.float32],
    effort_net_applied: wp.array[wp.float32],
    effort_net_target: wp.array[wp.float32],
    effort_velocity: wp.array[wp.float32],
    effort_residual: wp.array[wp.float32],
):
    effort = wp.tid()
    if world_mask[effort_world[effort]]:
        effort_intercept[effort] = 0.0
        effort_slope[effort] = 0.0
        effort_impulse_bound[effort] = 0.0
        effort_raw_impulse[effort] = 0.0
        effort_counter_applied[effort] = 0.0
        effort_counter_next[effort] = 0.0
        effort_net_applied[effort] = 0.0
        effort_net_target[effort] = 0.0
        effort_velocity[effort] = 0.0
        effort_residual[effort] = 0.0


@wp.kernel
def _reset_effort_worlds_masked(
    world_mask: wp.array[wp.bool],
    world_effort_residual_max: wp.array[wp.float32],
    world_effort_defect_max: wp.array[wp.float32],
):
    world = wp.tid()
    if world_mask[world]:
        world_effort_residual_max[world] = 0.0
        world_effort_defect_max[world] = 0.0


@wp.kernel
def _reset_friction_reactions_masked(
    friction_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    friction_reaction: wp.array[wp.float32],
):
    friction = wp.tid()
    if world_mask[friction_world[friction]]:
        friction_reaction[friction] = 0.0


@wp.kernel
def _scale_structural_reactions(
    scale: wp.float32,
    reaction: wp.array[wp.float32],
):
    row = wp.tid()
    reaction[row] *= scale


@wp.kernel
def _update_structural_multipliers_from_candidate_rows(
    time_step: wp.array[wp.float32],
    structural_tolerance: wp.float32,
    row_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    body_first_global: wp.array[wp.int32],
    body_second_global: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    candidate_residual: wp.array[wp.float32],
    proximal_defect: wp.array[wp.float32],
    frozen_residual: wp.array[wp.float32],
    penalty: wp.array[wp.float32],
    linearization_twist: wp.array[vec6f],
    candidate_twist: wp.array[vec6f],
    proximal_relaxation: wp.float32,
    body_vector_index: wp.array[wp.int32],
    reaction: wp.array[wp.float32],
    world_residual: wp.array[wp.float32],
    right_hand_side: wp.array[wp.float32],
):
    row = wp.tid()
    world = row_world[row]
    dt = time_step[world]
    if not world_active[world] or projection_status[world] != PROJECTION_STATUS_VALID:
        return

    first_global = body_first_global[row]
    second_global = body_second_global[row]
    first_dynamic = first_global >= 0 and body_vector_index[first_global] >= 0
    second_dynamic = second_global >= 0 and body_vector_index[second_global] >= 0
    if not first_dynamic and not second_dynamic:
        reaction[row] = 0.0
        return
    candidate_velocity = wp.float32(0.0)
    linearization_velocity = wp.float32(0.0)
    if first_global >= 0:
        candidate_velocity += wp.dot(jacobian_first[row], candidate_twist[first_global])
        linearization_velocity += wp.dot(jacobian_first[row], linearization_twist[first_global])
    if second_global >= 0:
        candidate_velocity += wp.dot(jacobian_second[row], candidate_twist[second_global])
        linearization_velocity += wp.dot(jacobian_second[row], linearization_twist[second_global])

    linear_residual = frozen_residual[row] + dt * (candidate_velocity - linearization_velocity)
    update_residual = linear_residual
    if proximal_relaxation > 0.0:
        defect = proximal_defect[row]
        defect += proximal_relaxation * (candidate_residual[row] - linear_residual - defect)
        proximal_defect[row] = defect
        update_residual += defect
    candidate_residual[row] = update_residual
    wp.atomic_max(world_residual, world, wp.abs(update_residual) / structural_tolerance)

    reaction_delta = -penalty[row] * update_residual
    reaction[row] += reaction_delta

    for axis in range(6):
        if first_global >= 0 and body_vector_index[first_global] >= 0:
            wp.atomic_add(
                right_hand_side,
                body_vector_index[first_global] + axis,
                dt * reaction_delta * jacobian_first[row][axis],
            )
        if second_global >= 0 and body_vector_index[second_global] >= 0:
            wp.atomic_add(
                right_hand_side,
                body_vector_index[second_global] + axis,
                dt * reaction_delta * jacobian_second[row][axis],
            )


@wp.kernel
def _reduce_structural_candidate_residual(
    structural_tolerance: wp.float32,
    row_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    body_first_global: wp.array[wp.int32],
    body_second_global: wp.array[wp.int32],
    candidate_residual: wp.array[wp.float32],
    body_vector_index: wp.array[wp.int32],
    world_residual: wp.array[wp.float32],
):
    row = wp.tid()
    world = row_world[row]
    if not world_active[world] or projection_status[world] != PROJECTION_STATUS_VALID:
        return

    first = body_first_global[row]
    second = body_second_global[row]
    first_dynamic = first >= 0 and body_vector_index[first] >= 0
    second_dynamic = second >= 0 and body_vector_index[second] >= 0
    if not first_dynamic and not second_dynamic:
        return
    wp.atomic_max(world_residual, world, wp.abs(candidate_residual[row]) / structural_tolerance)
