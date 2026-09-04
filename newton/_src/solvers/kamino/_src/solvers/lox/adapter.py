# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels bridging Kamino containers with LOX solver storage.

Preparation kernels materialize compact, coalesced row data once per nonlinear
evaluation. LOX reuses those rows across its projection iterations; consumers
therefore avoid repeated sparse-Jacobian indirection and contact preprocessing.
"""

import warp as wp

from ...core.joints import JointActuationType
from ...core.math import contact_wrench_matrix_from_points
from ...core.types import mat36f, vec6f
from ...geometry.contacts import ContactMode
from .bias import compute_contact_velocity_target, compute_limit_velocity_target
from .kernels import _inverse_mass_quadratic_form

wp.set_module_options({"enable_backward": False})


@wp.func
def _load_sparse_jacobian_row(index: wp.int32, jacobian_data: wp.array[vec6f]) -> vec6f:
    if index >= 0:
        return jacobian_data[index]
    return vec6f(0.0)


@wp.kernel
def _prepare_dynamic_rows(
    row_world: wp.array[wp.int32],
    uses_dof_jacobian: wp.array[wp.bool],
    body_first_global: wp.array[wp.int32],
    body_second_global: wp.array[wp.int32],
    value_index: wp.array[wp.int32],
    dof_index: wp.array[wp.int32],
    sparse_first_index: wp.array[wp.int32],
    sparse_second_index: wp.array[wp.int32],
    sparse_jacobian_data: wp.array[vec6f],
    sparse_dof_jacobian_data: wp.array[vec6f],
    joint_inertia: wp.array[wp.float32],
    joint_free_velocity: wp.array[wp.float32],
    dynamic_effort_index: wp.array[wp.int32],
    effort_value_index: wp.array[wp.int32],
    effort_inverse_inertia: wp.array[wp.float32],
    effort_free_velocity: wp.array[wp.float32],
    joint_armature: wp.array[wp.float32],
    joint_position_stiffness: wp.array[wp.float32],
    joint_actuation_type: wp.array[wp.int32],
    joint_velocity: wp.array[wp.float32],
    joint_velocity_begin: wp.array[wp.float32],
    external_effort: wp.array[wp.float32],
    velocity_stiffness: wp.array[wp.float32],
    effort_limit: wp.array[wp.float32],
    linearization_twist: wp.array[vec6f],
    time_step: wp.array[wp.float32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    effective_inertia: wp.array[wp.float32],
    free_velocity: wp.array[wp.float32],
    effort_intercept: wp.array[wp.float32],
    effort_slope: wp.array[wp.float32],
    effort_impulse_bound: wp.array[wp.float32],
):
    row = wp.tid()
    world = row_world[row]
    dt = time_step[world]
    if uses_dof_jacobian[row]:
        jacobian_first[row] = _load_sparse_jacobian_row(sparse_first_index[row], sparse_dof_jacobian_data)
        jacobian_second[row] = _load_sparse_jacobian_row(sparse_second_index[row], sparse_dof_jacobian_data)
    else:
        jacobian_first[row] = _load_sparse_jacobian_row(sparse_first_index[row], sparse_jacobian_data)
        jacobian_second[row] = _load_sparse_jacobian_row(sparse_second_index[row], sparse_jacobian_data)
    source = value_index[row]
    dof = dof_index[row]
    inertia = wp.float32(0.0)
    velocity = wp.float32(0.0)
    if source >= 0:
        inertia = joint_inertia[source]
        velocity = joint_free_velocity[source]
    bounded = dynamic_effort_index[row]
    if bounded >= 0:
        effort_source = effort_value_index[bounded]
        actuator_inverse_inertia = effort_inverse_inertia[effort_source]
        if actuator_inverse_inertia > 0.0:
            actuator_inertia = 1.0 / actuator_inverse_inertia
            velocity = (inertia * velocity + actuator_inertia * effort_free_velocity[effort_source]) / (
                inertia + actuator_inertia
            )
            inertia += actuator_inertia
    effective_inertia[row] = inertia
    mode = joint_actuation_type[dof]
    if inertia > 0.0:
        velocity += joint_armature[dof] * (joint_velocity_begin[dof] - joint_velocity[dof]) / inertia
        if (
            mode == JointActuationType.POSITION
            or mode == JointActuationType.POSITION_VELOCITY
            or mode == JointActuationType.POSITION_VELOCITY_FORCE
        ):
            linearization_velocity = wp.float32(0.0)
            first = body_first_global[row]
            second = body_second_global[row]
            if first >= 0:
                linearization_velocity += wp.dot(jacobian_first[row], linearization_twist[first])
            if second >= 0:
                linearization_velocity += wp.dot(jacobian_second[row], linearization_twist[second])
            velocity += dt * dt * joint_position_stiffness[dof] * linearization_velocity / inertia
    free_velocity[row] = velocity
    if bounded >= 0:
        gradient = wp.float32(0.0)
        if mode == JointActuationType.VELOCITY:
            gradient = velocity_stiffness[dof]
        if (
            mode == JointActuationType.POSITION
            or mode == JointActuationType.POSITION_VELOCITY
            or mode == JointActuationType.POSITION_VELOCITY_FORCE
        ):
            gradient = velocity_stiffness[dof] + dt * joint_position_stiffness[dof]
        beta = inertia * velocity
        effort_intercept[bounded] = (beta - joint_armature[dof] * joint_velocity_begin[dof]) / dt - external_effort[dof]
        effort_slope[bounded] = gradient
        effort_impulse_bound[bounded] = dt * effort_limit[dof]


@wp.kernel
def _prepare_joint_frictions(
    row_world: wp.array[wp.int32],
    body_first_global: wp.array[wp.int32],
    body_second_global: wp.array[wp.int32],
    dof_index: wp.array[wp.int32],
    multiplier_index: wp.array[wp.int32],
    sparse_first_index: wp.array[wp.int32],
    sparse_second_index: wp.array[wp.int32],
    sparse_jacobian_data: wp.array[vec6f],
    friction_force: wp.array[wp.float32],
    source_reaction: wp.array[wp.float32],
    body_velocity_begin: wp.array[vec6f],
    time_step: wp.array[wp.float32],
    import_reactions: wp.bool,
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    impulse_bound: wp.array[wp.float32],
    reaction: wp.array[wp.float32],
    velocity: wp.array[wp.float32],
):
    row = wp.tid()
    world = row_world[row]
    first_jacobian = _load_sparse_jacobian_row(sparse_first_index[row], sparse_jacobian_data)
    second_jacobian = _load_sparse_jacobian_row(sparse_second_index[row], sparse_jacobian_data)
    bound = time_step[world] * friction_force[dof_index[row]]
    jacobian_first[row] = first_jacobian
    jacobian_second[row] = second_jacobian
    impulse_bound[row] = bound
    reaction_guess = reaction[row]
    if import_reactions:
        reaction_guess = time_step[world] * source_reaction[multiplier_index[row]]
        value = wp.float32(0.0)
        first = body_first_global[row]
        second = body_second_global[row]
        if first >= 0:
            value += wp.dot(first_jacobian, body_velocity_begin[first])
        if second >= 0:
            value += wp.dot(second_jacobian, body_velocity_begin[second])
        velocity[row] = value
    reaction[row] = wp.clamp(reaction_guess, -bound, bound)


@wp.kernel
def _prepare_structural_rows(
    body_first_global: wp.array[wp.int32],
    body_second_global: wp.array[wp.int32],
    sparse_first_index: wp.array[wp.int32],
    sparse_second_index: wp.array[wp.int32],
    sparse_jacobian_data: wp.array[vec6f],
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[wp.mat33f],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    effective_mass: wp.array[wp.float32],
):
    row = wp.tid()
    first_jacobian = _load_sparse_jacobian_row(sparse_first_index[row], sparse_jacobian_data)
    second_jacobian = _load_sparse_jacobian_row(sparse_second_index[row], sparse_jacobian_data)
    jacobian_first[row] = first_jacobian
    jacobian_second[row] = second_jacobian

    inverse_effective_mass = _inverse_mass_quadratic_form(
        first_jacobian, body_first_global[row], inverse_mass, inverse_inertia_world
    ) + _inverse_mass_quadratic_form(second_jacobian, body_second_global[row], inverse_mass, inverse_inertia_world)
    effective_mass[row] = 1.0 / inverse_effective_mass if inverse_effective_mass > 1.0e-12 else 0.0


@wp.kernel
def _accumulate_constraint_incidence(
    entity_world: wp.array[wp.int32],
    entity_local: wp.array[wp.int32],
    world_count: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    body_constraint_count: wp.array[wp.int32],
    body_has_unilateral: wp.array[wp.int32],
):
    entity = wp.tid()
    world = entity_world[entity]
    if entity_local[entity] >= world_count[world]:
        return
    first = body_first[entity]
    second = body_second[entity]
    if first >= 0:
        wp.atomic_add(body_constraint_count, first, 1)
        wp.atomic_max(body_has_unilateral, first, 1)
    if second >= 0 and second != first:
        wp.atomic_add(body_constraint_count, second, 1)
        wp.atomic_max(body_has_unilateral, second, 1)


@wp.kernel
def _clear_inactive_limits(
    entity_world: wp.array[wp.int32],
    entity_local: wp.array[wp.int32],
    world_count: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    reaction: wp.array[wp.float32],
    velocity: wp.array[wp.float32],
):
    limit = wp.tid()
    if entity_local[limit] >= world_count[entity_world[limit]]:
        body_first[limit] = -1
        body_second[limit] = -1
        reaction[limit] = 0.0
        velocity[limit] = 0.0


@wp.kernel
def _clear_inactive_contacts(
    entity_world: wp.array[wp.int32],
    entity_local: wp.array[wp.int32],
    world_count: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    reaction: wp.array[wp.vec3f],
    velocity: wp.array[wp.vec3f],
):
    contact = wp.tid()
    if entity_local[contact] >= world_count[entity_world[contact]]:
        body_first[contact] = -1
        body_second[contact] = -1
        reaction[contact] = wp.vec3f(0.0)
        velocity[contact] = wp.vec3f(0.0)


@wp.kernel
def _copy_clamped_world_counts(
    source: wp.array[wp.int32],
    capacity: wp.array[wp.int32],
    destination: wp.array[wp.int32],
):
    world = wp.tid()
    destination[world] = wp.min(wp.max(source[world], 0), capacity[world])


@wp.kernel
def _mark_worlds_with_unilaterals(
    contact_count: wp.array[wp.int32],
    limit_count: wp.array[wp.int32],
    friction_count: wp.array[wp.int32],
    world_has_unilateral: wp.array[wp.bool],
):
    world = wp.tid()
    world_has_unilateral[world] = contact_count[world] > 0 or limit_count[world] > 0 or friction_count[world] > 0


@wp.kernel
def _prepare_limits(
    source_active: wp.array[wp.int32],
    source_capacity: wp.int32,
    source_world: wp.array[wp.int32],
    source_local: wp.array[wp.int32],
    source_bodies: wp.array[wp.vec2i],
    source_violation: wp.array[wp.float32],
    source_reaction: wp.array[wp.float32],
    body_velocity_begin: wp.array[vec6f],
    world_capacity: wp.array[wp.int32],
    world_offset: wp.array[wp.int32],
    body_vector_index: wp.array[wp.int32],
    sparse_jacobian_offsets: wp.array[wp.int32],
    sparse_jacobian_data: wp.array[vec6f],
    time_step: wp.array[wp.float32],
    stabilization_fraction: wp.float32,
    import_reactions: wp.bool,
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    bias: wp.array[wp.float32],
    reaction: wp.array[wp.float32],
    velocity: wp.array[wp.float32],
):
    source = wp.tid()
    if source >= wp.min(source_active[0], source_capacity):
        return
    world = source_world[source]
    local = source_local[source]
    if world < 0 or world >= world_capacity.shape[0] or local < 0 or local >= world_capacity[world]:
        return

    destination = world_offset[world] + local
    bodies = source_bodies[source]
    first_global = bodies[0]
    second_global = bodies[1]
    first_dynamic = first_global >= 0 and body_vector_index[first_global] >= 0
    second_dynamic = second_global >= 0 and body_vector_index[second_global] >= 0
    if not first_dynamic and not second_dynamic:
        body_first[destination] = -1
        body_second[destination] = -1
        jacobian_first[destination] = vec6f(0.0)
        jacobian_second[destination] = vec6f(0.0)
        bias[destination] = 0.0
        reaction[destination] = 0.0
        velocity[destination] = 0.0
        return
    sparse_offset = sparse_jacobian_offsets[source]
    first_sparse_index = sparse_offset + 1 if first_global >= 0 else -1
    first_jacobian = _load_sparse_jacobian_row(first_sparse_index, sparse_jacobian_data)
    second_jacobian = _load_sparse_jacobian_row(sparse_offset, sparse_jacobian_data)
    body_first[destination] = first_global
    body_second[destination] = second_global
    jacobian_first[destination] = first_jacobian
    jacobian_second[destination] = second_jacobian
    velocity_previous = wp.float32(0.0)
    if first_global >= 0:
        velocity_previous += wp.dot(first_jacobian, body_velocity_begin[first_global])
    if second_global >= 0:
        velocity_previous += wp.dot(second_jacobian, body_velocity_begin[second_global])
    dt = time_step[world]
    target = compute_limit_velocity_target(source_violation[source], dt, stabilization_fraction)
    bias[destination] = -target
    if import_reactions:
        reaction[destination] = dt * source_reaction[source]
        velocity[destination] = velocity_previous


@wp.kernel
def _prepare_contacts(
    source_active: wp.array[wp.int32],
    source_capacity: wp.int32,
    source_world: wp.array[wp.int32],
    source_local: wp.array[wp.int32],
    source_bodies: wp.array[wp.vec2i],
    source_position_a: wp.array[wp.vec3f],
    source_position_b: wp.array[wp.vec3f],
    source_frame: wp.array[wp.quatf],
    source_gap: wp.array[wp.vec4f],
    source_material: wp.array[wp.vec2f],
    source_reaction: wp.array[wp.vec3f],
    body_pose: wp.array[wp.transformf],
    body_velocity_begin: wp.array[vec6f],
    world_capacity: wp.array[wp.int32],
    world_count: wp.array[wp.int32],
    world_offset: wp.array[wp.int32],
    body_vector_index: wp.array[wp.int32],
    time_step: wp.array[wp.float32],
    stabilization_fraction: wp.float32,
    dead_zone: wp.float32,
    impact_velocity_threshold: wp.float32,
    recoverable_response: wp.bool,
    import_reactions: wp.bool,
    compact_contacts: wp.bool,
    source_to_internal: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[mat36f],
    jacobian_second: wp.array[mat36f],
    bias: wp.array[wp.vec3f],
    friction: wp.array[wp.float32],
    reaction: wp.array[wp.vec3f],
    velocity: wp.array[wp.vec3f],
):
    source = wp.tid()
    if source >= wp.min(source_active[0], source_capacity):
        return
    world = source_world[source]
    local = source_local[source]
    if world < 0 or world >= world_capacity.shape[0] or local < 0 or local >= world_capacity[world]:
        return

    bodies = source_bodies[source]
    first_global = bodies[0]
    second_global = bodies[1]
    first_dynamic = first_global >= 0 and body_vector_index[first_global] >= 0
    second_dynamic = second_global >= 0 and body_vector_index[second_global] >= 0
    if not first_dynamic and not second_dynamic:
        source_to_internal[source] = -1
        return
    destination = source_to_internal[source]
    if compact_contacts:
        destination = world_offset[world] + wp.atomic_add(world_count, world, 1)
        source_to_internal[source] = destination
    first_jacobian = mat36f(0.0)
    rotation = wp.quat_to_matrix(source_frame[source])
    body_position_b = wp.transform_get_translation(body_pose[second_global])
    jacobian_transpose_b = contact_wrench_matrix_from_points(source_position_b[source], body_position_b) @ rotation
    second_jacobian = wp.transpose(jacobian_transpose_b)
    if first_global >= 0:
        body_position_a = wp.transform_get_translation(body_pose[first_global])
        jacobian_transpose_a = -contact_wrench_matrix_from_points(source_position_a[source], body_position_a) @ rotation
        first_jacobian = wp.transpose(jacobian_transpose_a)

    velocity_previous = wp.vec3f(0.0)
    if first_global >= 0:
        velocity_previous += first_jacobian @ body_velocity_begin[first_global]
    if second_global >= 0:
        velocity_previous += second_jacobian @ body_velocity_begin[second_global]
    gap = source_gap[source]
    material = source_material[source]
    dt = time_step[world]
    target = compute_contact_velocity_target(
        gap[3],
        velocity_previous[2],
        material[1],
        dt,
        stabilization_fraction,
        dead_zone,
        impact_velocity_threshold,
        recoverable_response,
    )

    body_first[destination] = first_global
    body_second[destination] = second_global
    jacobian_first[destination] = first_jacobian
    jacobian_second[destination] = second_jacobian
    bias[destination] = wp.vec3f(0.0, 0.0, -target)
    friction[destination] = material[0]
    if import_reactions:
        reaction[destination] = dt * source_reaction[source]
        velocity[destination] = velocity_previous


@wp.kernel
def _write_dynamic_outputs(
    inverse_time_step: wp.array[wp.float32],
    row_world: wp.array[wp.int32],
    multiplier_index: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    effective_inertia: wp.array[wp.float32],
    free_velocity: wp.array[wp.float32],
    body_velocity: wp.array[vec6f],
    destination: wp.array[wp.float32],
    destination_wrench: wp.array[vec6f],
):
    row = wp.tid()
    velocity = wp.float32(0.0)
    first = body_first[row]
    second = body_second[row]
    if first >= 0:
        velocity += wp.dot(jacobian_first[row], body_velocity[first])
    if second >= 0:
        velocity += wp.dot(jacobian_second[row], body_velocity[second])
    force = inverse_time_step[row_world[row]] * effective_inertia[row] * (free_velocity[row] - velocity)
    destination[multiplier_index[row]] = force
    if first >= 0:
        wp.atomic_add(destination_wrench, first, force * jacobian_first[row])
    if second >= 0:
        wp.atomic_add(destination_wrench, second, force * jacobian_second[row])


@wp.kernel
def _write_dynamic_outputs_with_effort(
    inverse_time_step: wp.array[wp.float32],
    row_world: wp.array[wp.int32],
    multiplier_index: wp.array[wp.int32],
    effort_index: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    effective_inertia: wp.array[wp.float32],
    free_velocity: wp.array[wp.float32],
    effort_counter_applied: wp.array[wp.float32],
    effort_net_applied: wp.array[wp.float32],
    effort_value_index: wp.array[wp.int32],
    body_velocity: wp.array[vec6f],
    destination: wp.array[wp.float32],
    effort_destination: wp.array[wp.float32],
    destination_wrench: wp.array[vec6f],
):
    row = wp.tid()
    velocity = wp.float32(0.0)
    first = body_first[row]
    second = body_second[row]
    if first >= 0:
        velocity += wp.dot(jacobian_first[row], body_velocity[first])
    if second >= 0:
        velocity += wp.dot(jacobian_second[row], body_velocity[second])
    inv_dt = inverse_time_step[row_world[row]]
    multiplier = inv_dt * effective_inertia[row] * (free_velocity[row] - velocity)
    bounded = effort_index[row]
    if bounded >= 0:
        multiplier += inv_dt * effort_counter_applied[bounded]
        effort_destination[effort_value_index[bounded]] = inv_dt * effort_net_applied[bounded]
    destination_index = multiplier_index[row]
    if destination_index >= 0:
        destination[destination_index] = multiplier
    elif bounded >= 0:
        multiplier = inv_dt * effort_net_applied[bounded]
    else:
        return
    if first >= 0:
        wp.atomic_add(destination_wrench, first, multiplier * jacobian_first[row])
    if second >= 0:
        wp.atomic_add(destination_wrench, second, multiplier * jacobian_second[row])


@wp.kernel
def _write_friction_outputs(
    inverse_time_step: wp.array[wp.float32],
    row_world: wp.array[wp.int32],
    multiplier_index: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    source: wp.array[wp.float32],
    destination: wp.array[wp.float32],
    destination_wrench: wp.array[vec6f],
):
    row = wp.tid()
    force = inverse_time_step[row_world[row]] * source[row]
    destination[multiplier_index[row]] = force
    first = body_first[row]
    second = body_second[row]
    if first >= 0:
        wp.atomic_add(destination_wrench, first, force * jacobian_first[row])
    if second >= 0:
        wp.atomic_add(destination_wrench, second, force * jacobian_second[row])


@wp.kernel
def _accumulate_aligned_joint_wrenches(
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    reaction: wp.array[wp.float32],
    destination: wp.array[vec6f],
):
    row = wp.tid()
    scale = reaction[row]
    first = body_first[row]
    second = body_second[row]
    if first >= 0:
        wp.atomic_add(destination, first, scale * jacobian_first[row])
    if second >= 0:
        wp.atomic_add(destination, second, scale * jacobian_second[row])


@wp.kernel
def _write_limit_outputs(
    source_active: wp.array[wp.int32],
    source_capacity: wp.int32,
    source_world: wp.array[wp.int32],
    source_local: wp.array[wp.int32],
    world_capacity: wp.array[wp.int32],
    world_offset: wp.array[wp.int32],
    inverse_time_step: wp.array[wp.float32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    reaction: wp.array[wp.float32],
    velocity: wp.array[wp.float32],
    destination_reaction: wp.array[wp.float32],
    destination_velocity: wp.array[wp.float32],
    destination_wrench: wp.array[vec6f],
):
    source = wp.tid()
    if source >= wp.min(source_active[0], source_capacity):
        return
    world = source_world[source]
    local = source_local[source]
    if world < 0 or world >= world_capacity.shape[0] or local < 0 or local >= world_capacity[world]:
        return
    internal = world_offset[world] + local
    force = inverse_time_step[world] * reaction[internal]
    destination_reaction[source] = force
    destination_velocity[source] = velocity[internal]
    first = body_first[internal]
    second = body_second[internal]
    if first >= 0:
        wp.atomic_add(destination_wrench, first, force * jacobian_first[internal])
    if second >= 0:
        wp.atomic_add(destination_wrench, second, force * jacobian_second[internal])


@wp.kernel
def _write_contact_outputs(
    source_active: wp.array[wp.int32],
    source_capacity: wp.int32,
    source_world: wp.array[wp.int32],
    source_to_internal: wp.array[wp.int32],
    inverse_time_step: wp.array[wp.float32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[mat36f],
    jacobian_second: wp.array[mat36f],
    reaction: wp.array[wp.vec3f],
    velocity: wp.array[wp.vec3f],
    destination_reaction: wp.array[wp.vec3f],
    destination_velocity: wp.array[wp.vec3f],
    destination_mode: wp.array[wp.int32],
    destination_wrench: wp.array[vec6f],
):
    source = wp.tid()
    if source >= wp.min(source_active[0], source_capacity):
        return
    internal = source_to_internal[source]
    if internal < 0:
        zero_velocity = wp.vec3f(0.0)
        destination_reaction[source] = wp.vec3f(0.0)
        destination_velocity[source] = zero_velocity
        destination_mode[source] = wp.static(ContactMode.make_compute_mode_func())(zero_velocity)
        return
    world = source_world[source]
    force = inverse_time_step[world] * reaction[internal]
    destination_reaction[source] = force
    destination_velocity[source] = velocity[internal]
    destination_mode[source] = wp.static(ContactMode.make_compute_mode_func())(velocity[internal])
    first = body_first[internal]
    second = body_second[internal]
    if first >= 0:
        wp.atomic_add(destination_wrench, first, wp.transpose(jacobian_first[internal]) @ force)
    if second >= 0:
        wp.atomic_add(destination_wrench, second, wp.transpose(jacobian_second[internal]) @ force)


@wp.kernel
def _write_integrator_body_inputs(
    body_vector_index: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    world_accepted: wp.array[wp.bool],
    world_time_step: wp.array[wp.float32],
    body_mass: wp.array[wp.float32],
    body_inertia: wp.array[wp.mat33f],
    body_inverse_mass: wp.array[wp.float32],
    body_inverse_inertia: wp.array[wp.mat33f],
    world_gravity: wp.array[wp.vec3f],
    velocity_begin: wp.array[vec6f],
    velocity_projected: wp.array[vec6f],
    body_wrench: wp.array[wp.spatial_vectorf],
    body_velocity: wp.array[wp.spatial_vectorf],
):
    """Encode the accepted LOX velocity as inputs to Kamino integration."""
    body = wp.tid()
    world = body_world[body]
    velocity_end = velocity_begin[body]
    if body_vector_index[body] >= 0 and world_accepted[world]:
        velocity_end = velocity_projected[body]
    linear_velocity_begin = wp.vec3f(
        velocity_begin[body][0],
        velocity_begin[body][1],
        velocity_begin[body][2],
    )
    angular_velocity_begin = wp.vec3f(velocity_begin[body][3], velocity_begin[body][4], velocity_begin[body][5])
    linear_velocity_end = wp.vec3f(velocity_end[0], velocity_end[1], velocity_end[2])
    angular_velocity_end = wp.vec3f(velocity_end[3], velocity_end[4], velocity_end[5])
    inverse_time_step = 1.0 / world_time_step[world]
    force = body_mass[body] * (inverse_time_step * (linear_velocity_end - linear_velocity_begin) - world_gravity[world])
    inertia = body_inertia[body]
    torque = inertia @ (inverse_time_step * (angular_velocity_end - angular_velocity_begin)) + wp.skew(
        angular_velocity_begin
    ) @ (inertia @ angular_velocity_begin)
    body_wrench[body] = wp.spatial_vectorf(
        force[0],
        force[1],
        force[2],
        torque[0],
        torque[1],
        torque[2],
    )
    if body_vector_index[body] >= 0 and (
        body_inverse_mass[body] == 0.0 or wp.determinant(body_inverse_inertia[body]) == 0.0
    ):
        body_velocity[body] = wp.spatial_vectorf(
            velocity_end[0],
            velocity_end[1],
            velocity_end[2],
            velocity_end[3],
            velocity_end[4],
            velocity_end[5],
        )
