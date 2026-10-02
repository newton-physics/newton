# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kernels of the LOX problem.

Loading kernels materialize compact, coalesced row data from Kamino containers once per
time step, so the projection iterations avoid repeated sparse-Jacobian indirection and
contact preprocessing. The joint-row kernels update the structural multipliers and the
effort counters during the iterations, and the output kernels write the reactions back
to Kamino.
"""

from functools import cache

import warp as wp

from ...core.joints import JointActuationType, JointCorrectionMode
from ...core.math import compute_body_pose_update_with_logmap, contact_wrench_matrix_from_points
from ...core.types import mat36f, mat66f, vec6f
from ...geometry.contacts import ContactMode
from ...kinematics.joints import compute_joint_pose_and_relative_motion, make_write_joint_data
from .contact import (
    compute_contact_normal_compliance,
    compute_contact_penetration_bias,
    compute_contact_recovery_fraction,
    compute_contact_velocity_target,
)
from .solver_kernels import atomic_max_nonnegative

###
# Module interface
###

__all__ = [
    "_accumulate_aligned_joint_wrenches",
    "_compute_structural_effective_mass",
    "_count_box_rows",
    "_enable_bodies_in_weighted_blocks",
    "_finish_box_rows",
    "_finish_contacts",
    "_load_contacts",
    "_load_dynamic_rows",
    "_load_joint_frictions",
    "_load_limits",
    "_load_structural_rows",
    "_mark_blocks_with_unilaterals",
    "_reset_angular_reactions",
    "_reset_effort_rows",
    "_reset_effort_worlds",
    "_reset_row_reactions",
    "_scatter_structural_impulses",
    "_update_effort_counters",
    "_write_contact_outputs",
    "_write_dynamic_outputs",
    "_write_friction_outputs",
    "_write_limit_outputs",
    "make_update_structural_multipliers_kernel",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###


@wp.func
def _compute_limit_velocity_target(
    violation: wp.float32,
    time_step: wp.float32,
    stabilization_fraction: wp.float32,
) -> wp.float32:
    """Compute the minimum end-of-step joint-limit velocity.

    Args:
        violation: Signed limit residual, negative when violated [m or rad].
        time_step: Time step [s].
        stabilization_fraction: Violation recovery fraction.

    Returns:
        Minimum feasible limit velocity [m/s or rad/s].
    """
    return -stabilization_fraction * wp.min(violation, 0.0) / time_step


@wp.func
def _load_sparse_jacobian_row(index: wp.int32, jacobian_data: wp.array[vec6f]) -> vec6f:
    """Return one block of a Kamino sparse Jacobian, or zero for a missing block."""
    if index >= 0:
        return jacobian_data[index]
    return vec6f(0.0)


@wp.func
def _inverse_mass_bilinear_form(
    jacobian_a: vec6f,
    jacobian_b: vec6f,
    bid: wp.int32,
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[wp.mat33f],
) -> wp.float32:
    if bid < 0:
        return 0.0
    a_linear = wp.vec3f(jacobian_a[0], jacobian_a[1], jacobian_a[2])
    b_linear = wp.vec3f(jacobian_b[0], jacobian_b[1], jacobian_b[2])
    a_angular = wp.vec3f(jacobian_a[3], jacobian_a[4], jacobian_a[5])
    b_angular = wp.vec3f(jacobian_b[3], jacobian_b[4], jacobian_b[5])
    return inverse_mass[bid] * wp.dot(a_linear, b_linear) + wp.dot(a_angular, inverse_inertia_world[bid] @ b_angular)


###
# Kernels
###


@wp.kernel
def _count_box_rows(
    # Inputs:
    limits_world_num: wp.array[wp.int32],
    world_limit_capacity: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    # Outputs:
    world_box_count: wp.array[wp.int32],
):
    """Box rows of a world are its friction rows followed by its active limits."""
    wid = wp.tid()
    limit_count = wp.int32(0)
    if limits_world_num:
        limit_count = wp.min(wp.max(limits_world_num[wid], 0), world_limit_capacity[wid])
    world_box_count[wid] = world_friction_count[wid] + limit_count


@wp.kernel
def _load_joint_frictions(
    # Inputs:
    friction_row: wp.array[wp.int32],
    dof_index: wp.array[wp.int32],
    multiplier_index: wp.array[wp.int32],
    sparse_a_index: wp.array[wp.int32],
    sparse_b_index: wp.array[wp.int32],
    sparse_jacobian_data: wp.array[vec6f],
    friction_force: wp.array[wp.float32],
    data_joints_lambda_f_j: wp.array[wp.float32],
    row_world: wp.array[wp.int32],
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    body_velocity_begin: wp.array[vec6f],
    model_time_dt: wp.array[wp.float32],
    # Outputs:
    jacobian_a: wp.array[vec6f],
    jacobian_b: wp.array[vec6f],
    lower: wp.array[wp.float32],
    upper: wp.array[wp.float32],
    reaction: wp.array[wp.float32],
    velocity: wp.array[wp.float32],
):
    """Write one bounded friction row ``-b <= lambda <= b`` per frictional joint DOF."""
    tid = wp.tid()
    row = friction_row[tid]
    wid = row_world[row]
    dt = model_time_dt[wid]
    a_jacobian = _load_sparse_jacobian_row(sparse_a_index[tid], sparse_jacobian_data)
    b_jacobian = _load_sparse_jacobian_row(sparse_b_index[tid], sparse_jacobian_data)
    bound = dt * friction_force[dof_index[tid]]
    jacobian_a[row] = a_jacobian
    jacobian_b[row] = b_jacobian
    lower[row] = -bound
    upper[row] = bound
    value = wp.float32(0.0)
    bid_a = body_a[row]
    bid_b = body_b[row]
    if bid_a >= 0:
        value += wp.dot(a_jacobian, body_velocity_begin[bid_a])
    if bid_b >= 0:
        value += wp.dot(b_jacobian, body_velocity_begin[bid_b])
    velocity[row] = value
    reaction[row] = wp.clamp(dt * data_joints_lambda_f_j[multiplier_index[tid]], -bound, bound)


@wp.kernel
def _load_limits(
    # Inputs:
    limits_model_num: wp.array[wp.int32],
    limits_model_max: wp.int32,
    limits_wid: wp.array[wp.int32],
    limits_lid: wp.array[wp.int32],
    limits_bids: wp.array[wp.vec2i],
    limits_r_q: wp.array[wp.float32],
    limits_reaction: wp.array[wp.float32],
    body_velocity_begin: wp.array[vec6f],
    world_capacity: wp.array[wp.int32],
    world_offset: wp.array[wp.int32],
    body_vector_index: wp.array[wp.int32],
    sparse_jacobian_offsets: wp.array[wp.int32],
    sparse_jacobian_data: wp.array[vec6f],
    model_time_dt: wp.array[wp.float32],
    stabilization_fraction: wp.float32,
    # Outputs:
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    jacobian_a: wp.array[vec6f],
    jacobian_b: wp.array[vec6f],
    bias: wp.array[wp.float32],
    reaction: wp.array[wp.float32],
    velocity: wp.array[wp.float32],
):
    """Load each detected joint limit into its box row, with its velocity target and warm-start reaction."""
    lid = wp.tid()
    if lid >= wp.min(limits_model_num[0], limits_model_max):
        return
    wid = limits_wid[lid]
    local = limits_lid[lid]
    if wid < 0 or wid >= world_capacity.shape[0] or local < 0 or local >= world_capacity[wid]:
        return

    destination = world_offset[wid] + local
    bodies = limits_bids[lid]
    bid_a_global = bodies[0]
    bid_b_global = bodies[1]
    a_is_dynamic = bid_a_global >= 0 and body_vector_index[bid_a_global] >= 0
    b_is_dynamic = bid_b_global >= 0 and body_vector_index[bid_b_global] >= 0
    if not a_is_dynamic and not b_is_dynamic:
        body_a[destination] = -1
        body_b[destination] = -1
        jacobian_a[destination] = vec6f(0.0)
        jacobian_b[destination] = vec6f(0.0)
        bias[destination] = 0.0
        reaction[destination] = 0.0
        velocity[destination] = 0.0
        return
    sparse_offset = sparse_jacobian_offsets[lid]
    a_sparse_index = sparse_offset + 1 if bid_a_global >= 0 else -1
    a_jacobian = _load_sparse_jacobian_row(a_sparse_index, sparse_jacobian_data)
    b_jacobian = _load_sparse_jacobian_row(sparse_offset, sparse_jacobian_data)
    body_a[destination] = bid_a_global
    body_b[destination] = bid_b_global
    jacobian_a[destination] = a_jacobian
    jacobian_b[destination] = b_jacobian
    velocity_previous = wp.float32(0.0)
    if bid_a_global >= 0:
        velocity_previous += wp.dot(a_jacobian, body_velocity_begin[bid_a_global])
    if bid_b_global >= 0:
        velocity_previous += wp.dot(b_jacobian, body_velocity_begin[bid_b_global])
    dt = model_time_dt[wid]
    target = _compute_limit_velocity_target(limits_r_q[lid], dt, stabilization_fraction)
    bias[destination] = -target
    reaction[destination] = dt * limits_reaction[lid]
    velocity[destination] = velocity_previous


@wp.kernel
def _finish_box_rows(
    # Inputs:
    row_world: wp.array[wp.int32],
    row_local: wp.array[wp.int32],
    world_row_count: wp.array[wp.int32],
    # Outputs:
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    reaction: wp.array[wp.float32],
    velocity: wp.array[wp.float32],
    body_incidence: wp.array2d[wp.int32],
):
    """Clear the inactive box rows and count the active rows touching each body."""
    row = wp.tid()
    if row_local[row] >= world_row_count[row_world[row]]:
        body_a[row] = -1
        body_b[row] = -1
        reaction[row] = 0.0
        velocity[row] = 0.0
    else:
        bid_a = body_a[row]
        bid_b = body_b[row]
        if bid_a >= 0:
            wp.atomic_add(body_incidence, bid_a, 0, 1)
        if bid_b >= 0 and bid_b != bid_a:
            wp.atomic_add(body_incidence, bid_b, 0, 1)


@wp.kernel
def _load_contacts(
    # Inputs:
    contacts_model_num: wp.array[wp.int32],
    contacts_model_max: wp.int32,
    contacts_wid: wp.array[wp.int32],
    contacts_cid: wp.array[wp.int32],
    contacts_bid_AB: wp.array[wp.vec2i],
    contacts_position_A: wp.array[wp.vec3f],
    contacts_position_B: wp.array[wp.vec3f],
    contacts_frame: wp.array[wp.quatf],
    contacts_gapfunc: wp.array[wp.vec4f],
    contacts_material: wp.array[wp.vec2f],
    contacts_angular_friction: wp.array[wp.vec2f],
    contacts_reaction: wp.array[wp.vec3f],
    data_bodies_q_i: wp.array[wp.transformf],
    body_velocity_begin: wp.array[vec6f],
    world_capacity: wp.array[wp.int32],
    world_count: wp.array[wp.int32],
    world_offset: wp.array[wp.int32],
    body_vector_index: wp.array[wp.int32],
    model_time_dt: wp.array[wp.float32],
    stabilization_fraction: wp.float32,
    dead_zone: wp.float32,
    contacts_stiffness: wp.array[wp.float32],
    contacts_damping: wp.array[wp.float32],
    use_restitution: wp.bool,
    source_to_internal: wp.array[wp.int32],
    # Outputs:
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    jacobian_a: wp.array[mat36f],
    jacobian_b: wp.array[mat36f],
    bias: wp.array[wp.vec3f],
    normal_compliance: wp.array[wp.float32],
    restitution: wp.array[wp.vec4f],
    friction: wp.array[wp.float32],
    reaction: wp.array[wp.vec3f],
    velocity: wp.array[wp.vec3f],
    frame: wp.array[wp.mat33f],
    angular_friction: wp.array[wp.vec2f],
):
    """Compact each detected contact into its contact row, with its Jacobians, contact law, and warm-start reaction."""
    cid = wp.tid()
    if cid >= wp.min(contacts_model_num[0], contacts_model_max):
        return
    wid = contacts_wid[cid]
    local = contacts_cid[cid]
    if wid < 0 or wid >= world_capacity.shape[0] or local < 0 or local >= world_capacity[wid]:
        source_to_internal[cid] = -1
        return

    bodies = contacts_bid_AB[cid]
    bid_a_global = bodies[0]
    bid_b_global = bodies[1]
    a_is_dynamic = bid_a_global >= 0 and body_vector_index[bid_a_global] >= 0
    b_is_dynamic = bid_b_global >= 0 and body_vector_index[bid_b_global] >= 0
    if not a_is_dynamic and not b_is_dynamic:
        source_to_internal[cid] = -1
        return
    destination = world_offset[wid] + wp.atomic_add(world_count, wid, 1)
    source_to_internal[cid] = destination
    a_jacobian = mat36f(0.0)
    rotation = wp.quat_to_matrix(contacts_frame[cid])
    body_position_b = wp.transform_get_translation(data_bodies_q_i[bid_b_global])
    jacobian_transpose_b = contact_wrench_matrix_from_points(contacts_position_B[cid], body_position_b) @ rotation
    b_jacobian = wp.transpose(jacobian_transpose_b)
    if bid_a_global >= 0:
        body_position_a = wp.transform_get_translation(data_bodies_q_i[bid_a_global])
        jacobian_transpose_a = -contact_wrench_matrix_from_points(contacts_position_A[cid], body_position_a) @ rotation
        a_jacobian = wp.transpose(jacobian_transpose_a)

    velocity_previous = wp.vec3f(0.0)
    if bid_a_global >= 0:
        velocity_previous += a_jacobian @ body_velocity_begin[bid_a_global]
    if bid_b_global >= 0:
        velocity_previous += b_jacobian @ body_velocity_begin[bid_b_global]
    gap = contacts_gapfunc[cid][3]
    material = contacts_material[cid]
    dt = model_time_dt[wid]
    stiffness = wp.float32(0.0)
    damping = wp.float32(0.0)
    if contacts_stiffness:
        stiffness = contacts_stiffness[cid]
        damping = contacts_damping[cid]

    # Compliant contacts are implicit spring-dampers: they recover a fraction of their
    # penetration, keep the static target of positive gaps, and get their bounce from
    # the spring-damper alone.
    normal_bias = wp.float32(0.0)
    if stiffness > 0.0:
        fraction = compute_contact_recovery_fraction(stiffness, damping, dt)
        normal_bias = compute_contact_penetration_bias(gap, dt, fraction)
        normal_bias -= compute_contact_velocity_target(
            wp.max(gap, 0.0), velocity_previous[2], 0.0, dt, stabilization_fraction, dead_zone
        )
        if use_restitution:
            # Neutral inputs give a zero in-kernel restitution target
            restitution[destination] = wp.vec4f(0.0, 0.0, 0.0, dt)
    elif use_restitution:
        # The in-kernel restitution replaces the static restitution target; the
        # bias then only recovers penetration.
        normal_bias = compute_contact_penetration_bias(gap, dt, stabilization_fraction)
        restitution[destination] = wp.vec4f(gap, velocity_previous[2], material[1], dt)
    else:
        normal_bias = -compute_contact_velocity_target(
            gap, velocity_previous[2], material[1], dt, stabilization_fraction, dead_zone
        )
    if normal_compliance:
        normal_compliance[destination] = compute_contact_normal_compliance(stiffness, damping, dt)

    body_a[destination] = bid_a_global
    body_b[destination] = bid_b_global
    jacobian_a[destination] = a_jacobian
    jacobian_b[destination] = b_jacobian
    bias[destination] = wp.vec3f(0.0, 0.0, normal_bias)
    friction[destination] = material[0]
    if frame:
        # Spatial contacts also solve their spin and rolling rows about the contact frame axes
        frame[destination] = rotation
        angular_friction[destination] = contacts_angular_friction[cid]
    reaction[destination] = dt * contacts_reaction[cid]
    velocity[destination] = velocity_previous


@wp.kernel
def _finish_contacts(
    # Inputs:
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_count: wp.array[wp.int32],
    # Outputs:
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    reaction: wp.array[wp.vec3f],
    velocity: wp.array[wp.vec3f],
    body_incidence: wp.array2d[wp.int32],
):
    """Clear the inactive contacts and count the active contacts touching each body."""
    cid = wp.tid()
    if contact_local[cid] >= world_count[contact_world[cid]]:
        body_a[cid] = -1
        body_b[cid] = -1
        reaction[cid] = wp.vec3f(0.0)
        velocity[cid] = wp.vec3f(0.0)
    else:
        bid_a = body_a[cid]
        bid_b = body_b[cid]
        if bid_a >= 0:
            wp.atomic_add(body_incidence, bid_a, 0, 1)
        if bid_b >= 0 and bid_b != bid_a:
            wp.atomic_add(body_incidence, bid_b, 0, 1)


@wp.kernel
def _load_structural_rows(
    # Inputs:
    sparse_a_index: wp.array[wp.int32],
    sparse_b_index: wp.array[wp.int32],
    sparse_jacobian_data: wp.array[vec6f],
    warmstart_factor: wp.float32,
    # Outputs:
    jacobian_a: wp.array[vec6f],
    jacobian_b: wp.array[vec6f],
    reaction: wp.array[wp.float32],
):
    """Load each structural row for the time step.

    Gathers the Jacobian blocks from Kamino's sparse constraint Jacobian and damps the
    multiplier warm start of the previous time step.
    """
    row = wp.tid()
    jacobian_a[row] = _load_sparse_jacobian_row(sparse_a_index[row], sparse_jacobian_data)
    jacobian_b[row] = _load_sparse_jacobian_row(sparse_b_index[row], sparse_jacobian_data)
    reaction[row] *= warmstart_factor


@wp.kernel
def _scatter_structural_impulses(
    # Inputs:
    model_time_dt: wp.array[wp.float32],
    row_world: wp.array[wp.int32],
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    jacobian_a: wp.array[vec6f],
    jacobian_b: wp.array[vec6f],
    penalty: wp.array[wp.float32],
    reaction: wp.array[wp.float32],
    # Outputs:
    body_impulse: wp.array[vec6f],
):
    """Accumulate the multiplier impulse ``h J^T lambda`` of each penalized structural row on its bodies."""
    row = wp.tid()
    if penalty[row] <= 0.0:
        return
    impulse = model_time_dt[row_world[row]] * reaction[row]
    bid_a = body_a[row]
    bid_b = body_b[row]
    if bid_a >= 0:
        wp.atomic_add(body_impulse, bid_a, impulse * jacobian_a[row])
    if bid_b >= 0:
        wp.atomic_add(body_impulse, bid_b, impulse * jacobian_b[row])


@wp.kernel
def _load_dynamic_rows(
    # Inputs:
    row_world: wp.array[wp.int32],
    uses_dof_jacobian: wp.array[wp.bool],
    value_index: wp.array[wp.int32],
    dof_index: wp.array[wp.int32],
    sparse_a_index: wp.array[wp.int32],
    sparse_b_index: wp.array[wp.int32],
    sparse_jacobian_data: wp.array[vec6f],
    sparse_dof_jacobian_data: wp.array[vec6f],
    data_joints_m_j: wp.array[wp.float32],
    data_joints_dq_b_j: wp.array[wp.float32],
    dynamic_effort_index: wp.array[wp.int32],
    effort_value_index: wp.array[wp.int32],
    data_joints_inv_m_a: wp.array[wp.float32],
    data_joints_dq_b_a: wp.array[wp.float32],
    model_joints_a_j: wp.array[wp.float32],
    model_joints_k_p_j: wp.array[wp.float32],
    model_joints_dof_act_types: wp.array[wp.int32],
    data_joints_dq_j: wp.array[wp.float32],
    data_joints_tau_j: wp.array[wp.float32],
    model_joints_k_d_j: wp.array[wp.float32],
    model_joints_tau_j_max: wp.array[wp.float32],
    model_time_dt: wp.array[wp.float32],
    # Outputs:
    jacobian_a: wp.array[vec6f],
    jacobian_b: wp.array[vec6f],
    effective_inertia: wp.array[wp.float32],
    free_velocity: wp.array[wp.float32],
    effort_intercept: wp.array[wp.float32],
    effort_slope: wp.array[wp.float32],
    effort_impulse_bound: wp.array[wp.float32],
):
    """Load the Jacobians, implicit inertia, and free velocity of each dynamic row, and the effort model of its drive."""
    row = wp.tid()
    dt = model_time_dt[row_world[row]]
    if uses_dof_jacobian[row]:
        jacobian_a[row] = _load_sparse_jacobian_row(sparse_a_index[row], sparse_dof_jacobian_data)
        jacobian_b[row] = _load_sparse_jacobian_row(sparse_b_index[row], sparse_dof_jacobian_data)
    else:
        jacobian_a[row] = _load_sparse_jacobian_row(sparse_a_index[row], sparse_jacobian_data)
        jacobian_b[row] = _load_sparse_jacobian_row(sparse_b_index[row], sparse_jacobian_data)
    source = value_index[row]
    dof = dof_index[row]
    inertia = wp.float32(0.0)
    velocity = wp.float32(0.0)
    if source >= 0:
        inertia = data_joints_m_j[source]
        velocity = data_joints_dq_b_j[source]
    bounded = dynamic_effort_index[row]
    if bounded >= 0:
        effort_source = effort_value_index[bounded]
        actuator_inverse_inertia = data_joints_inv_m_a[effort_source]
        if actuator_inverse_inertia > 0.0:
            actuator_inertia = 1.0 / actuator_inverse_inertia
            velocity = (inertia * velocity + actuator_inertia * data_joints_dq_b_a[effort_source]) / (
                inertia + actuator_inertia
            )
            inertia += actuator_inertia
    effective_inertia[row] = inertia
    mode = model_joints_dof_act_types[dof]
    free_velocity[row] = velocity
    if bounded >= 0:
        gradient = wp.float32(0.0)
        if mode == JointActuationType.VELOCITY:
            gradient = model_joints_k_d_j[dof]
        if (
            mode == JointActuationType.POSITION
            or mode == JointActuationType.POSITION_VELOCITY
            or mode == JointActuationType.POSITION_VELOCITY_FORCE
        ):
            gradient = model_joints_k_d_j[dof] + dt * model_joints_k_p_j[dof]
        beta = inertia * velocity
        effort_intercept[bounded] = (beta - model_joints_a_j[dof] * data_joints_dq_j[dof]) / dt - data_joints_tau_j[dof]
        effort_slope[bounded] = gradient
        effort_impulse_bound[bounded] = dt * model_joints_tau_j_max[dof]


@wp.kernel
def _compute_structural_effective_mass(
    # Inputs:
    model_joints_bid_B: wp.array[wp.int32],
    model_joints_bid_F: wp.array[wp.int32],
    model_joints_kinematic_cts_offset: wp.array[wp.int32],
    model_joints_num_kinematic_cts: wp.array[wp.int32],
    joint_dynamic_offset: wp.array[wp.int32],
    joint_dynamic_count: wp.array[wp.int32],
    structural_jacobian_a: wp.array[vec6f],
    structural_jacobian_b: wp.array[vec6f],
    dynamic_jacobian_a: wp.array[vec6f],
    dynamic_jacobian_b: wp.array[vec6f],
    dynamic_effective_inertia: wp.array[wp.float32],
    model_bodies_inv_m_i: wp.array[wp.float32],
    data_bodies_inv_I_i: wp.array[wp.mat33f],
    # Outputs:
    effective_mass: wp.array[wp.float32],
):
    """Compute the effective mass of each structural row in the smooth system of its joint.

    The dynamic rows of a joint add ``J_d^T diag(m_j) J_d`` to the inertia of its bodies,
    so by Woodbury the structural rows see

        m_s^-1 = J_s M^-1 J_s^T - (J_d M^-1 J_s^T)^T S_d^-1 (J_d M^-1 J_s^T),
        S_d = J_d M^-1 J_d^T + diag(m_j)^-1.

    Keeping ``m_j^-1`` in ``S_d`` retains the full implicit drive compliance of effort-limited
    drives, whether or not their effort is at its bound. Joints without dynamic rows, or with a
    singular ``S_d``, use the rigid-body value ``m_s^-1 = J_s M^-1 J_s^T``.
    """
    jid = wp.tid()
    structural_count = model_joints_num_kinematic_cts[jid]
    if structural_count == 0:
        return

    bid_a = model_joints_bid_B[jid]
    bid_b = model_joints_bid_F[jid]
    dynamic_offset = joint_dynamic_offset[jid]
    dynamic_count = joint_dynamic_count[jid]

    # Factor S_d, which is empty without dynamic rows
    lower = mat66f(0.0)
    valid = wp.bool(True)
    for row in range(6):
        if row < dynamic_count:
            a_row = dynamic_jacobian_a[dynamic_offset + row]
            b_row = dynamic_jacobian_b[dynamic_offset + row]
            for col in range(6):
                if col <= row and col < dynamic_count:
                    a_col = dynamic_jacobian_a[dynamic_offset + col]
                    b_col = dynamic_jacobian_b[dynamic_offset + col]
                    value = _inverse_mass_bilinear_form(
                        a_row, a_col, bid_a, model_bodies_inv_m_i, data_bodies_inv_I_i
                    ) + _inverse_mass_bilinear_form(b_row, b_col, bid_b, model_bodies_inv_m_i, data_bodies_inv_I_i)
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
        dynamic_count = 0

    structural_offset = model_joints_kinematic_cts_offset[jid]
    for structural_local in range(6):
        if structural_local < structural_count:
            structural_row = structural_offset + structural_local
            a_structural = structural_jacobian_a[structural_row]
            b_structural = structural_jacobian_b[structural_row]
            inverse_effective_mass = _inverse_mass_bilinear_form(
                a_structural, a_structural, bid_a, model_bodies_inv_m_i, data_bodies_inv_I_i
            ) + _inverse_mass_bilinear_form(
                b_structural, b_structural, bid_b, model_bodies_inv_m_i, data_bodies_inv_I_i
            )

            # Subtract the response of the dynamic rows: (J_d M^-1 J_s^T)^T S_d^-1 (J_d M^-1 J_s^T)
            forward = vec6f(0.0)
            for row in range(6):
                if row < dynamic_count:
                    coupling = _inverse_mass_bilinear_form(
                        a_structural,
                        dynamic_jacobian_a[dynamic_offset + row],
                        bid_a,
                        model_bodies_inv_m_i,
                        data_bodies_inv_I_i,
                    ) + _inverse_mass_bilinear_form(
                        b_structural,
                        dynamic_jacobian_b[dynamic_offset + row],
                        bid_b,
                        model_bodies_inv_m_i,
                        data_bodies_inv_I_i,
                    )
                    for inner in range(6):
                        if inner < row:
                            coupling -= lower[row, inner] * forward[inner]
                    forward[row] = coupling / lower[row, row]
                    inverse_effective_mass -= forward[row] * forward[row]

            effective_mass[structural_row] = 1.0 / inverse_effective_mass if inverse_effective_mass > 1.0e-12 else 0.0


@wp.kernel
def _mark_blocks_with_unilaterals(
    # Inputs:
    body_block: wp.array[wp.int32],
    body_has_unilateral: wp.array[wp.int32],
    # Outputs:
    block_has_unilateral: wp.array[wp.int32],
):
    """Mark the factor blocks that contain a body touched by unilateral rows."""
    bid = wp.tid()
    block = body_block[bid]
    if block >= 0 and body_has_unilateral[bid] != 0:
        wp.atomic_max(block_has_unilateral, block, 1)


@wp.kernel
def _enable_bodies_in_weighted_blocks(
    # Inputs:
    body_block: wp.array[wp.int32],
    block_has_unilateral: wp.array[wp.int32],
    # Outputs:
    body_weight_enabled: wp.array[wp.int32],
):
    """Enable the splitting weight of every body of a marked factor block."""
    bid = wp.tid()
    block = body_block[bid]
    body_weight_enabled[bid] = wp.where(block >= 0 and block_has_unilateral[block] != 0, 1, 0)


@cache
def make_update_structural_multipliers_kernel(correction: JointCorrectionMode, proximal: bool):
    """Build the kernel that updates the structural multipliers of each joint from a candidate twist.

    The candidate twist blends the global and projected twists of the joint bodies.
    With ``proximal`` relaxation, the exact joint residual at the candidate poses
    corrects the linearized residual; only the proximal kernel compiles that code, which
    keeps the register use of the default kernel low.
    """

    @wp.kernel
    def _update_structural_multipliers(
        # Inputs:
        model_time_dt: wp.array[wp.float32],
        structural_tolerance: wp.float32,
        model_joints_wid: wp.array[wp.int32],
        model_joints_dof_type: wp.array[wp.int32],
        model_joints_coords_offset: wp.array[wp.int32],
        model_joints_dofs_offset: wp.array[wp.int32],
        model_joints_kinematic_cts_offset: wp.array[wp.int32],
        model_joints_num_kinematic_cts: wp.array[wp.int32],
        model_joints_bid_B: wp.array[wp.int32],
        model_joints_bid_F: wp.array[wp.int32],
        model_joints_B_r_Bj: wp.array[wp.vec3f],
        model_joints_F_r_Fj: wp.array[wp.vec3f],
        model_joints_X_Bj: wp.array[wp.mat33f],
        model_joints_X_Fj: wp.array[wp.mat33f],
        data_bodies_q_i: wp.array[wp.transformf],
        data_joints_q_j_p: wp.array[wp.float32],
        world_active: wp.array[wp.bool],
        world_failed: wp.array[wp.bool],
        body_vector_index: wp.array[wp.int32],
        global_twist: wp.array[vec6f],
        projected_twist: wp.array[vec6f],
        projected_fraction: wp.float32,
        jacobian_a: wp.array[vec6f],
        jacobian_b: wp.array[vec6f],
        frozen_residual: wp.array[wp.float32],
        penalty: wp.array[wp.float32],
        proximal_relaxation: wp.float32,
        joint_max_correction: wp.float32,
        # Outputs:
        proximal_defect: wp.array[wp.float32],
        candidate_residual: wp.array[wp.float32],
        scratch_residual_velocity: wp.array[wp.float32],
        scratch_joint_coordinate: wp.array[wp.float32],
        scratch_joint_velocity: wp.array[wp.float32],
        reaction: wp.array[wp.float32],
        body_impulse: wp.array[vec6f],
        world_residual: wp.array[wp.float32],
    ):
        jid = wp.tid()
        row_begin = model_joints_kinematic_cts_offset[jid]
        row_count = model_joints_num_kinematic_cts[jid]
        wid = model_joints_wid[jid]
        if row_count == 0 or not world_active[wid] or world_failed[wid]:
            return
        dt = model_time_dt[wid]

        bid_a = model_joints_bid_B[jid]
        bid_b = model_joints_bid_F[jid]
        a_vector = -1
        if bid_a >= 0:
            a_vector = body_vector_index[bid_a]
        b_vector = -1
        if bid_b >= 0:
            b_vector = body_vector_index[bid_b]
        if a_vector < 0 and b_vector < 0:
            for row in range(row_begin, row_begin + row_count):
                reaction[row] = 0.0
            return

        # Blend the candidate twist of each body from the global and projected twists
        twist_a = vec6f(0.0)
        if bid_a >= 0:
            twist_a = global_twist[bid_a] + projected_fraction * (projected_twist[bid_a] - global_twist[bid_a])
        twist_b = vec6f(0.0)
        if bid_b >= 0:
            twist_b = global_twist[bid_b] + projected_fraction * (projected_twist[bid_b] - global_twist[bid_b])

        # Evaluate the exact joint residual at the candidate poses
        if wp.static(proximal):
            a_pose = wp.transform_identity(dtype=wp.float32)
            if bid_a >= 0:
                a_pose = compute_body_pose_update_with_logmap(
                    dt,
                    data_bodies_q_i[bid_a],
                    wp.vec3f(twist_a[0], twist_a[1], twist_a[2]),
                    wp.vec3f(twist_a[3], twist_a[4], twist_a[5]),
                )
            b_pose = compute_body_pose_update_with_logmap(
                dt,
                data_bodies_q_i[bid_b],
                wp.vec3f(twist_b[0], twist_b[1], twist_b[2]),
                wp.vec3f(twist_b[3], twist_b[4], twist_b[5]),
            )
            _, relative_position, relative_orientation, relative_twist = compute_joint_pose_and_relative_motion(
                a_pose,
                b_pose,
                wp.spatial_vectorf(0.0),
                wp.spatial_vectorf(0.0),
                model_joints_B_r_Bj[jid],
                model_joints_F_r_Fj[jid],
                model_joints_X_Bj[jid],
                model_joints_X_Fj[jid],
            )
            wp.static(make_write_joint_data(correction))(
                model_joints_dof_type[jid],
                row_begin,
                model_joints_dofs_offset[jid],
                model_joints_coords_offset[jid],
                relative_position,
                relative_orientation,
                relative_twist,
                data_joints_q_j_p,
                candidate_residual,
                scratch_residual_velocity,
                scratch_joint_coordinate,
                scratch_joint_velocity,
            )

        # Update the multipliers of the joint rows, and the multiplier impulses on its bodies
        residual_max = wp.float32(0.0)
        impulse_change_a = vec6f(0.0)
        impulse_change_b = vec6f(0.0)
        for row in range(row_begin, row_begin + row_count):
            row_jacobian_a = jacobian_a[row]
            row_jacobian_b = jacobian_b[row]
            candidate_velocity = wp.float32(0.0)
            if bid_a >= 0:
                candidate_velocity += wp.dot(row_jacobian_a, twist_a)
            if bid_b >= 0:
                candidate_velocity += wp.dot(row_jacobian_b, twist_b)

            row_residual = wp.clamp(frozen_residual[row], -joint_max_correction, joint_max_correction)
            linear_residual = row_residual + dt * candidate_velocity
            update_residual = linear_residual
            if wp.static(proximal):
                defect = proximal_defect[row]
                defect += proximal_relaxation * (candidate_residual[row] - linear_residual - defect)
                proximal_defect[row] = defect
                update_residual += defect
            residual_max = wp.max(residual_max, wp.abs(update_residual))
            reaction_change = -penalty[row] * update_residual
            reaction[row] += reaction_change
            impulse_change_a += reaction_change * row_jacobian_a
            impulse_change_b += reaction_change * row_jacobian_b
        if bid_a >= 0:
            wp.atomic_add(body_impulse, bid_a, dt * impulse_change_a)
        if bid_b >= 0:
            wp.atomic_add(body_impulse, bid_b, dt * impulse_change_b)
        atomic_max_nonnegative(world_residual, wid, residual_max / structural_tolerance)

    return _update_structural_multipliers


@wp.kernel
def _update_effort_counters(
    # Inputs:
    effort_world: wp.array[wp.int32],
    effort_dynamic_row: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    jacobian_a: wp.array[vec6f],
    jacobian_b: wp.array[vec6f],
    effective_inertia: wp.array[wp.float32],
    projected_twist: wp.array[vec6f],
    intercept: wp.array[wp.float32],
    slope: wp.array[wp.float32],
    impulse_bound: wp.array[wp.float32],
    model_time_dt: wp.array[wp.float32],
    velocity_tolerance: wp.float32,
    # Outputs:
    counter: wp.array[wp.float32],
    net_applied: wp.array[wp.float32],
    world_residual_scaled: wp.array[wp.float32],
):
    """Clamp each bounded drive at the projected twist with a counter-impulse for the next candidate solve.

    The change of the counter-impulse, in velocity units, is the effort residual of the world.
    """
    tid = wp.tid()
    wid = effort_world[tid]
    if not world_active[wid]:
        return

    dynamic_row = effort_dynamic_row[tid]
    current_velocity = wp.float32(0.0)
    bid_a = body_a[dynamic_row]
    bid_b = body_b[dynamic_row]
    if bid_a >= 0:
        current_velocity += wp.dot(jacobian_a[dynamic_row], projected_twist[bid_a])
    if bid_b >= 0:
        current_velocity += wp.dot(jacobian_b[dynamic_row], projected_twist[bid_b])

    dt = model_time_dt[wid]
    raw = dt * (intercept[tid] - slope[tid] * current_velocity)
    bound = impulse_bound[tid]
    target = wp.clamp(raw, -bound, bound)
    next_counter = target - raw
    defect = wp.abs(next_counter - counter[tid])
    scaled_defect = defect / (wp.max(1.0e-6, effective_inertia[dynamic_row]) * velocity_tolerance)

    counter[tid] = next_counter
    net_applied[tid] = target
    atomic_max_nonnegative(world_residual_scaled, wid, scaled_defect)


@wp.kernel
def _write_dynamic_outputs(
    # Inputs:
    model_time_inv_dt: wp.array[wp.float32],
    row_world: wp.array[wp.int32],
    multiplier_index: wp.array[wp.int32],
    effort_index: wp.array[wp.int32],
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    jacobian_a: wp.array[vec6f],
    jacobian_b: wp.array[vec6f],
    effective_inertia: wp.array[wp.float32],
    free_velocity: wp.array[wp.float32],
    effort_counter: wp.array[wp.float32],
    effort_net_applied: wp.array[wp.float32],
    effort_value_index: wp.array[wp.int32],
    body_velocity: wp.array[vec6f],
    # Outputs:
    data_joints_lambda_dyn_j: wp.array[wp.float32],
    data_joints_lambda_tau_j: wp.array[wp.float32],
    data_bodies_w_j_i: wp.array[vec6f],
):
    row = wp.tid()
    velocity = wp.float32(0.0)
    bid_a = body_a[row]
    bid_b = body_b[row]
    if bid_a >= 0:
        velocity += wp.dot(jacobian_a[row], body_velocity[bid_a])
    if bid_b >= 0:
        velocity += wp.dot(jacobian_b[row], body_velocity[bid_b])
    inv_dt = model_time_inv_dt[row_world[row]]
    multiplier = inv_dt * effective_inertia[row] * (free_velocity[row] - velocity)
    bounded = effort_index[row]
    if bounded >= 0:
        multiplier += inv_dt * effort_counter[bounded]
        data_joints_lambda_tau_j[effort_value_index[bounded]] = inv_dt * effort_net_applied[bounded]
    destination_index = multiplier_index[row]
    if destination_index >= 0:
        data_joints_lambda_dyn_j[destination_index] = multiplier
    elif bounded >= 0:
        multiplier = inv_dt * effort_net_applied[bounded]
    else:
        return
    if bid_a >= 0:
        wp.atomic_add(data_bodies_w_j_i, bid_a, multiplier * jacobian_a[row])
    if bid_b >= 0:
        wp.atomic_add(data_bodies_w_j_i, bid_b, multiplier * jacobian_b[row])


@wp.kernel
def _write_friction_outputs(
    # Inputs:
    model_time_inv_dt: wp.array[wp.float32],
    friction_row: wp.array[wp.int32],
    multiplier_index: wp.array[wp.int32],
    row_world: wp.array[wp.int32],
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    jacobian_a: wp.array[vec6f],
    jacobian_b: wp.array[vec6f],
    reaction: wp.array[wp.float32],
    world_failed: wp.array[wp.bool],
    # Outputs:
    data_joints_lambda_f_j: wp.array[wp.float32],
    data_bodies_w_j_i: wp.array[vec6f],
):
    """Write the joint friction forces; failed worlds write zero, which also seeds their next time step."""
    tid = wp.tid()
    row = friction_row[tid]
    wid = row_world[row]
    force = wp.float32(0.0)
    if not world_failed[wid]:
        force = model_time_inv_dt[wid] * reaction[row]
    data_joints_lambda_f_j[multiplier_index[tid]] = force
    bid_a = body_a[row]
    bid_b = body_b[row]
    if bid_a >= 0:
        wp.atomic_add(data_bodies_w_j_i, bid_a, force * jacobian_a[row])
    if bid_b >= 0:
        wp.atomic_add(data_bodies_w_j_i, bid_b, force * jacobian_b[row])


@wp.kernel
def _accumulate_aligned_joint_wrenches(
    # Inputs:
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    jacobian_a: wp.array[vec6f],
    jacobian_b: wp.array[vec6f],
    reaction: wp.array[wp.float32],
    # Outputs:
    data_bodies_w_j_i: wp.array[vec6f],
):
    row = wp.tid()
    scale = reaction[row]
    bid_a = body_a[row]
    bid_b = body_b[row]
    if bid_a >= 0:
        wp.atomic_add(data_bodies_w_j_i, bid_a, scale * jacobian_a[row])
    if bid_b >= 0:
        wp.atomic_add(data_bodies_w_j_i, bid_b, scale * jacobian_b[row])


@wp.kernel
def _write_limit_outputs(
    # Inputs:
    limits_model_num: wp.array[wp.int32],
    limits_model_max: wp.int32,
    limits_wid: wp.array[wp.int32],
    limits_lid: wp.array[wp.int32],
    world_capacity: wp.array[wp.int32],
    world_offset: wp.array[wp.int32],
    model_time_inv_dt: wp.array[wp.float32],
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    jacobian_a: wp.array[vec6f],
    jacobian_b: wp.array[vec6f],
    reaction: wp.array[wp.float32],
    velocity: wp.array[wp.float32],
    world_failed: wp.array[wp.bool],
    # Outputs:
    limits_reaction: wp.array[wp.float32],
    limits_velocity: wp.array[wp.float32],
    data_bodies_w_l_i: wp.array[vec6f],
):
    lid = wp.tid()
    if lid >= wp.min(limits_model_num[0], limits_model_max):
        return
    wid = limits_wid[lid]
    local = limits_lid[lid]
    if wid < 0 or wid >= world_capacity.shape[0] or local < 0 or local >= world_capacity[wid]:
        return
    internal = world_offset[wid] + local
    # Failed worlds write zero, which also seeds their next time step
    force = wp.float32(0.0)
    if not world_failed[wid]:
        force = model_time_inv_dt[wid] * reaction[internal]
    limits_reaction[lid] = force
    limits_velocity[lid] = velocity[internal]
    bid_a = body_a[internal]
    bid_b = body_b[internal]
    if bid_a >= 0:
        wp.atomic_add(data_bodies_w_l_i, bid_a, force * jacobian_a[internal])
    if bid_b >= 0:
        wp.atomic_add(data_bodies_w_l_i, bid_b, force * jacobian_b[internal])


@wp.kernel
def _write_contact_outputs(
    # Inputs:
    contacts_model_num: wp.array[wp.int32],
    contacts_model_max: wp.int32,
    contacts_wid: wp.array[wp.int32],
    source_to_internal: wp.array[wp.int32],
    model_time_inv_dt: wp.array[wp.float32],
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    jacobian_a: wp.array[mat36f],
    jacobian_b: wp.array[mat36f],
    reaction: wp.array[wp.vec3f],
    velocity: wp.array[wp.vec3f],
    frame: wp.array[wp.mat33f],
    angular_reaction: wp.array[wp.vec3f],
    world_failed: wp.array[wp.bool],
    # Outputs:
    contacts_reaction: wp.array[wp.vec3f],
    contacts_velocity: wp.array[wp.vec3f],
    contacts_mode: wp.array[wp.int32],
    data_bodies_w_c_i: wp.array[vec6f],
):
    cid = wp.tid()
    if cid >= wp.min(contacts_model_num[0], contacts_model_max):
        return
    internal = source_to_internal[cid]
    if internal < 0:
        zero_velocity = wp.vec3f(0.0)
        contacts_reaction[cid] = wp.vec3f(0.0)
        contacts_velocity[cid] = zero_velocity
        contacts_mode[cid] = wp.static(ContactMode.make_compute_mode_func())(zero_velocity)
        return
    wid = contacts_wid[cid]
    # Failed worlds write zero, which also seeds their next time step
    failed = world_failed[wid]
    force = wp.vec3f(0.0)
    if not failed:
        force = model_time_inv_dt[wid] * reaction[internal]
    contacts_reaction[cid] = force
    contacts_velocity[cid] = velocity[internal]
    contacts_mode[cid] = wp.static(ContactMode.make_compute_mode_func())(velocity[internal])
    wrench_a = wp.transpose(jacobian_a[internal]) @ force
    wrench_b = wp.transpose(jacobian_b[internal]) @ force
    if angular_reaction and not failed:
        # The spin and rolling rows act along the normal and the two tangents
        angular = model_time_inv_dt[wid] * angular_reaction[internal]
        axes = frame[internal]
        moment = angular[0] * wp.vec3f(axes[0, 2], axes[1, 2], axes[2, 2])
        moment += angular[1] * wp.vec3f(axes[0, 0], axes[1, 0], axes[2, 0])
        moment += angular[2] * wp.vec3f(axes[0, 1], axes[1, 1], axes[2, 1])
        torque = vec6f(0.0, 0.0, 0.0, moment[0], moment[1], moment[2])
        wrench_a -= torque
        wrench_b += torque
    bid_a = body_a[internal]
    bid_b = body_b[internal]
    if bid_a >= 0:
        wp.atomic_add(data_bodies_w_c_i, bid_a, wrench_a)
    if bid_b >= 0:
        wp.atomic_add(data_bodies_w_c_i, bid_b, wrench_b)


@wp.kernel
def _reset_row_reactions(
    # Inputs:
    row_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    # Outputs:
    reaction: wp.array[wp.float32],
):
    row = wp.tid()
    if not world_mask or world_mask[row_world[row]]:
        reaction[row] = 0.0


@wp.kernel
def _reset_effort_rows(
    # Inputs:
    effort_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    # Outputs:
    effort_intercept: wp.array[wp.float32],
    effort_slope: wp.array[wp.float32],
    effort_impulse_bound: wp.array[wp.float32],
    effort_counter: wp.array[wp.float32],
    effort_net_applied: wp.array[wp.float32],
):
    tid = wp.tid()
    if not world_mask or world_mask[effort_world[tid]]:
        effort_intercept[tid] = 0.0
        effort_slope[tid] = 0.0
        effort_impulse_bound[tid] = 0.0
        effort_counter[tid] = 0.0
        effort_net_applied[tid] = 0.0


@wp.kernel
def _reset_effort_worlds(
    # Inputs:
    world_mask: wp.array[wp.bool],
    # Outputs:
    world_effort_residual_max: wp.array[wp.float32],
):
    wid = wp.tid()
    if not world_mask or world_mask[wid]:
        world_effort_residual_max[wid] = 0.0


@wp.kernel
def _reset_angular_reactions(
    # Inputs:
    contact_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    # Outputs:
    angular_reaction: wp.array[wp.vec3f],
):
    cid = wp.tid()
    if not world_mask or world_mask[contact_world[cid]]:
        angular_reaction[cid] = wp.vec3f(0.0)
