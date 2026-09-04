# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import functools

import warp as wp

from newton._src.math.spatial import quat_velocity
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    _SMALL_ANGLE_EPS,
    _SMALL_LENGTH_EPS,
    RigidForceElementAdjacencyInfo,
    _load_solve_weight,
    contact_surface_separation,
    evaluate_contact_point_world,
    evaluate_joint_force_hessian,
    evaluate_rigid_contact_from_collision,
    get_body_adjacent_joint_id,
    get_body_num_adjacent_joints,
)

wp.set_module_options({"enable_backward": False})

_NUM_ELASTIC_CONTACT_THREADS_PER_BODY = 32
"""Threads per reduced elastic body for modal contact accumulation."""


@wp.kernel
def copy_elastic_body_transforms_back(
    body_ids: wp.array[wp.int32],
    body_elastic_index: wp.array[wp.int32],
    body_q_in: wp.array[wp.transform],
    body_q_out: wp.array[wp.transform],
):
    """Copy the coupled frame solve result for elastic bodies in one color group."""
    body = body_ids[wp.tid()]
    if body_elastic_index[body] >= 0:
        body_q_out[body] = body_q_in[body]


@wp.func
def _so3_left_jacobian(theta: wp.vec3) -> wp.mat33:
    """Map a rotation-vector increment to a left-trivialized angular increment."""
    theta_sq = wp.dot(theta, theta)
    theta_skew = wp.skew(theta)
    theta_skew_sq = theta_skew * theta_skew
    if theta_sq < _SMALL_ANGLE_EPS * _SMALL_ANGLE_EPS:
        return wp.identity(3, float) + 0.5 * theta_skew + (1.0 / 6.0) * theta_skew_sq

    angle = wp.sqrt(theta_sq)
    a = (1.0 - wp.cos(angle)) / theta_sq
    b = (angle - wp.sin(angle)) / (theta_sq * angle)
    return wp.identity(3, float) + a * theta_skew + b * theta_skew_sq


@wp.func
def _elastic_endpoint_xform(
    joint_index: int,
    body_index: int,
    is_parent_side: bool,
    xform_rest: wp.transform,
    body_elastic_index: wp.array[wp.int32],
    elastic_joint: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    joint_q: wp.array[float],
    joint_q_start: wp.array[wp.int32],
    joint_parent_elastic_endpoint: wp.array[wp.int32],
    joint_child_elastic_endpoint: wp.array[wp.int32],
    elastic_endpoint_phi: wp.array[wp.vec3],
    elastic_endpoint_psi: wp.array[wp.vec3],
    elastic_max_mode_count: int,
):
    if body_index < 0:
        return xform_rest

    elastic_index = body_elastic_index[body_index]
    if elastic_index < 0:
        return xform_rest

    endpoint = joint_child_elastic_endpoint[joint_index]
    if is_parent_side:
        endpoint = joint_parent_elastic_endpoint[joint_index]
    if endpoint < 0:
        return xform_rest

    p = wp.transform_get_translation(xform_rest)
    q = wp.transform_get_rotation(xform_rest)
    owner_joint = elastic_joint[elastic_index]
    q_start = joint_q_start[owner_joint] + 7
    mode_count = elastic_mode_count[elastic_index]

    theta = wp.vec3(0.0, 0.0, 0.0)
    for mode in range(elastic_max_mode_count):
        if mode < mode_count:
            idx = endpoint * elastic_max_mode_count + mode
            p = p + elastic_endpoint_phi[idx] * joint_q[q_start + mode]
            theta = theta + elastic_endpoint_psi[idx] * joint_q[q_start + mode]

    angle = wp.length(theta)
    if angle > _SMALL_ANGLE_EPS:
        q = wp.quat_from_axis_angle(theta / angle, angle) * q

    return wp.transform(p, q)


@wp.kernel
def copy_elastic_joint_frame_to_body(
    elastic_body: wp.array[wp.int32],
    elastic_joint: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    elastic_index = wp.tid()
    body = elastic_body[elastic_index]
    joint = elastic_joint[elastic_index]
    q_start = joint_q_start[joint]
    qd_start = joint_qd_start[joint]

    body_q[body] = wp.transform(
        wp.vec3(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2]),
        wp.quat(joint_q[q_start + 3], joint_q[q_start + 4], joint_q[q_start + 5], joint_q[q_start + 6]),
    )
    body_qd[body] = wp.spatial_vector(
        wp.vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2]),
        wp.vec3(joint_qd[qd_start + 3], joint_qd[qd_start + 4], joint_qd[qd_start + 5]),
    )


@wp.kernel
def integrate_elastic_modes_implicit(
    dt: float,
    elastic_joint: wp.array[wp.int32],
    elastic_mode_start: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    elastic_mode_mass: wp.array[float],
    elastic_mode_stiffness: wp.array[float],
    elastic_mode_damping: wp.array[float],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_f: wp.array[float],
    joint_q_in: wp.array[float],
    joint_qd_in: wp.array[float],
    joint_q_out: wp.array[float],
    joint_qd_out: wp.array[float],
):
    elastic_index = wp.tid()
    joint = elastic_joint[elastic_index]
    q_start = joint_q_start[joint] + 7
    qd_start = joint_qd_start[joint] + 6
    mode_start = elastic_mode_start[elastic_index]
    mode_count = elastic_mode_count[elastic_index]

    for i in range(mode_count):
        mode = mode_start + i
        q_idx = q_start + i
        qd_idx = qd_start + i

        q = joint_q_in[q_idx]
        v = joint_qd_in[qd_idx]
        force = joint_f[qd_idx]
        mass = elastic_mode_mass[mode]
        stiffness = elastic_mode_stiffness[mode]
        damping = elastic_mode_damping[mode]

        denom = mass + dt * damping + dt * dt * stiffness
        if denom <= 0.0:
            joint_q_out[q_idx] = q
            joint_qd_out[qd_idx] = 0.0
        else:
            v_new = (mass * v + dt * (force - stiffness * q)) / denom
            q_new = q + dt * v_new
            joint_q_out[q_idx] = q_new
            joint_qd_out[qd_idx] = v_new


@wp.kernel
def copy_elastic_modes(
    elastic_joint: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_q_src: wp.array[float],
    joint_qd_src: wp.array[float],
    joint_q_dst: wp.array[float],
    joint_qd_dst: wp.array[float],
):
    elastic_index = wp.tid()
    joint = elastic_joint[elastic_index]
    q_start = joint_q_start[joint] + 7
    qd_start = joint_qd_start[joint] + 6
    mode_count = elastic_mode_count[elastic_index]

    for i in range(mode_count):
        joint_q_dst[q_start + i] = joint_q_src[q_start + i]
        joint_qd_dst[qd_start + i] = joint_qd_src[qd_start + i]


@wp.func
def _accumulate_elastic_contact_modes(
    dt: float,
    solve_frame: bool,
    contact_idx: int,
    body: int,
    elastic_index: int,
    body_R: wp.mat33,
    body_R_T: wp.mat33,
    elastic_joint: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    elastic_max_mode_count: int,
    body_elastic_index: wp.array[wp.int32],
    body_inv_mass: wp.array[float],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_offset0: wp.array[wp.vec3],
    rigid_contact_offset1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_margin0: wp.array[float],
    rigid_contact_margin1: wp.array[float],
    rigid_contact_elastic_sample0: wp.array[wp.int32],
    rigid_contact_elastic_sample1: wp.array[wp.int32],
    contact_penalty_k: wp.array[float],
    contact_normal_rho: wp.array[float],
    contact_material_ke: wp.array[float],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    contact_tangent_rho: wp.array[float],
    contact_lambda: wp.array[wp.vec3],
    contact_C0: wp.array[wp.vec3],
    stab_alpha: float,
    legacy_hard_contacts: int,
    contact_compliant_alm: int,
    friction_epsilon: float,
    joint_q_start: wp.array[wp.int32],
    elastic_shape_vertex_local: wp.array[wp.vec3],
    elastic_shape_vertex_phi: wp.array[wp.vec3],
    joint_q_prev: wp.array[float],
    joint_q: wp.array[float],
    elastic_body_block_grad: wp.array2d[float],
    elastic_body_block_matrix: wp.array3d[float],
):
    s0 = rigid_contact_shape0[contact_idx]
    s1 = rigid_contact_shape1[contact_idx]
    if s0 < 0 or s1 < 0:
        return

    b0 = shape_body[s0]
    b1 = shape_body[s1]
    elastic_sample0 = rigid_contact_elastic_sample0[contact_idx]
    elastic_sample1 = rigid_contact_elastic_sample1[contact_idx]

    elastic_body = -1
    elastic_sample = -1
    use_side0 = False
    if elastic_sample0 >= 0:
        elastic_body = b0
        elastic_sample = elastic_sample0
        use_side0 = True
    elif elastic_sample1 >= 0:
        elastic_body = b1
        elastic_sample = elastic_sample1

    if elastic_body != body or elastic_sample < 0:
        return

    cp0_local = rigid_contact_point0[contact_idx]
    cp1_local = rigid_contact_point1[contact_idx]
    cp0_world, cp0_world_prev = evaluate_contact_point_world(
        b0,
        cp0_local,
        elastic_sample0,
        body_q,
        body_q_prev,
        body_elastic_index,
        elastic_joint,
        elastic_mode_count,
        joint_q,
        joint_q_prev,
        joint_q_start,
        elastic_shape_vertex_local,
        elastic_shape_vertex_phi,
        elastic_max_mode_count,
    )
    cp1_world, cp1_world_prev = evaluate_contact_point_world(
        b1,
        cp1_local,
        elastic_sample1,
        body_q,
        body_q_prev,
        body_elastic_index,
        elastic_joint,
        elastic_mode_count,
        joint_q,
        joint_q_prev,
        joint_q_start,
        elastic_shape_vertex_local,
        elastic_shape_vertex_phi,
        elastic_max_mode_count,
    )

    contact_normal = rigid_contact_normal[contact_idx]
    C_n = -contact_surface_separation(
        cp0_world,
        cp1_world,
        contact_normal,
        rigid_contact_margin0[contact_idx],
        rigid_contact_margin1[contact_idx],
    )
    lam_vec = wp.vec3(0.0)
    lam_n = float(0.0)
    C_eff = C_n
    friction_c0 = wp.vec3(0.0)
    normal_solve_weight = _load_solve_weight(contact_penalty_k, contact_normal_rho, contact_idx, contact_compliant_alm)
    material_k = contact_material_ke[contact_idx]
    if legacy_hard_contacts == 0 and contact_compliant_alm == 0:
        normal_solve_weight = material_k
    tangent_solve_weight = contact_tangent_rho[contact_idx]

    if legacy_hard_contacts == 1 or contact_compliant_alm == 1:
        lam_vec = contact_lambda[contact_idx]
        lam_n = wp.dot(lam_vec, contact_normal)
        C0_vec = contact_C0[contact_idx]
        C0_n = wp.dot(contact_normal, C0_vec)
        C_eff = C_n - stab_alpha * C0_n
        friction_c0 = (1.0 - stab_alpha) * (C0_vec - contact_normal * C0_n)

    if C_n <= _SMALL_LENGTH_EPS and lam_n <= 0.0:
        return

    (
        force_0,
        _torque_0,
        h_ll_0,
        h_al_0,
        _h_aa_0,
        force_1,
        _torque_1,
        h_ll_1,
        h_al_1,
        _h_aa_1,
    ) = evaluate_rigid_contact_from_collision(
        b0,
        b1,
        body_q,
        body_q_prev,
        body_com,
        cp0_world,
        cp1_world,
        cp0_world_prev,
        cp1_world_prev,
        rigid_contact_offset0[contact_idx],
        rigid_contact_offset1[contact_idx],
        1,
        contact_normal,
        C_eff,
        normal_solve_weight,
        material_k,
        tangent_solve_weight,
        contact_material_kd[contact_idx],
        lam_vec,
        contact_material_mu[contact_idx],
        friction_epsilon,
        legacy_hard_contacts,
        contact_compliant_alm,
        dt,
        friction_c0,
    )

    elastic_force = force_1
    elastic_h = h_ll_1
    elastic_h_al = h_al_1
    if use_side0:
        elastic_force = force_0
        elastic_h = h_ll_0
        elastic_h_al = h_al_0

    elastic_force_local = body_R_T * elastic_force
    elastic_h_local = body_R_T * (elastic_h * body_R)
    elastic_h_al_local = body_R_T * (elastic_h_al * body_R)
    mode_count = elastic_mode_count[elastic_index]
    max_modes = elastic_max_mode_count

    for i in range(mode_count):
        phi_i_local = elastic_shape_vertex_phi[elastic_sample * max_modes + i]
        wp.atomic_add(
            elastic_body_block_grad,
            elastic_index,
            6 + i,
            -wp.dot(elastic_force_local, phi_i_local),
        )

        if solve_frame and body_inv_mass[body] > 0.0:
            linear_cross = elastic_h_local * phi_i_local
            angular_cross = elastic_h_al_local * phi_i_local
            for axis in range(3):
                wp.atomic_add(
                    elastic_body_block_matrix,
                    elastic_index,
                    axis,
                    6 + i,
                    linear_cross[axis],
                )
                wp.atomic_add(
                    elastic_body_block_matrix,
                    elastic_index,
                    3 + axis,
                    6 + i,
                    angular_cross[axis],
                )

        for j in range(i, mode_count):
            phi_j_local = elastic_shape_vertex_phi[elastic_sample * max_modes + j]
            h_ij = wp.dot(phi_i_local, elastic_h_local * phi_j_local)
            wp.atomic_add(elastic_body_block_matrix, elastic_index, 6 + i, 6 + j, h_ij)


@wp.kernel
def assemble_elastic_joints(
    dt: float,
    solve_frame: bool,
    body_ids: wp.array[wp.int32],
    elastic_joint: wp.array[wp.int32],
    elastic_mode_start: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    elastic_mode_mass: wp.array[float],
    elastic_mode_stiffness: wp.array[float],
    elastic_mode_damping: wp.array[float],
    elastic_mode_coupling_linear: wp.array[wp.vec3],
    elastic_mode_coupling_angular: wp.array[wp.vec3],
    elastic_mode_coupling_centrifugal: wp.array[wp.mat33],
    elastic_mode_coupling_coriolis: wp.array[wp.vec3],
    elastic_implicit_mass_coupling: wp.array[bool],
    adjacency: RigidForceElementAdjacencyInfo,
    elastic_endpoint_phi: wp.array[wp.vec3],
    elastic_endpoint_psi: wp.array[wp.vec3],
    elastic_max_mode_count: int,
    body_elastic_index: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_q_rest: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_inv_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_inertia_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_world: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    external_forces: wp.array[wp.vec3],
    external_torques: wp.array[wp.vec3],
    external_hessian_ll: wp.array[wp.mat33],
    external_hessian_al: wp.array[wp.mat33],
    external_hessian_aa: wp.array[wp.mat33],
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_rod_rest_kb_local: wp.array[wp.vec3],
    joint_rod_rest_twist: wp.array[float],
    joint_qd_start: wp.array[wp.int32],
    joint_target_q_start: wp.array[wp.int32],
    joint_constraint_start: wp.array[wp.int32],
    joint_penalty_k: wp.array[float],
    joint_rho: wp.array[float],
    joint_material_k: wp.array[float],
    joint_penalty_kd: wp.array[float],
    joint_sigma_start: wp.array[wp.vec3],
    joint_C_fric: wp.array[wp.vec3],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_q: wp.array[float],
    joint_target_qd: wp.array[float],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_limit_ke: wp.array[float],
    joint_limit_kd: wp.array[float],
    joint_drive_limit_support: wp.array[float],
    joint_drive_lambda: wp.array[float],
    joint_limit_lambda: wp.array[float],
    joint_lambda_lin: wp.array[wp.vec3],
    joint_lambda_ang: wp.array[wp.vec3],
    joint_C0_lin: wp.array[wp.vec3],
    joint_C0_ang: wp.array[wp.vec3],
    joint_is_hard: wp.array[wp.int32],
    stab_alpha: float,
    joint_compliant_alm: int,
    joint_dof_dim: wp.array2d[int],
    joint_rest_angle: wp.array[float],
    joint_parent_elastic_endpoint: wp.array[wp.int32],
    joint_child_elastic_endpoint: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_f: wp.array[float],
    joint_q_prev: wp.array[float],
    joint_qd_prev: wp.array[float],
    joint_q: wp.array[float],
    elastic_body_block_grad: wp.array2d[float],
    elastic_body_block_delta: wp.array2d[float],
    elastic_body_block_matrix: wp.array3d[float],
):
    body = body_ids[wp.tid()]
    elastic_index = body_elastic_index[body]
    if elastic_index < 0:
        return

    owner_joint = elastic_joint[elastic_index]
    q_start = joint_q_start[owner_joint] + 7
    qd_start = joint_qd_start[owner_joint] + 6
    body_qd_start = joint_qd_start[owner_joint]
    mode_start = elastic_mode_start[elastic_index]
    mode_count = elastic_mode_count[elastic_index]
    max_modes = elastic_max_mode_count
    block_width = max_modes + 6
    mode_coupling_mat_start = elastic_index * max_modes * max_modes

    inv_dt = 1.0 / dt
    inv_dt_sq = inv_dt * inv_dt
    body_rot = wp.transform_get_rotation(body_q[body])
    body_R = wp.quat_to_matrix(body_rot)
    body_R_T = wp.transpose(body_R)
    g_local = body_R_T * gravity[wp.max(body_world[body], 0)]

    com_v_prev = wp.vec3(
        joint_qd_prev[body_qd_start + 0], joint_qd_prev[body_qd_start + 1], joint_qd_prev[body_qd_start + 2]
    )
    omega_prev = wp.vec3(
        joint_qd_prev[body_qd_start + 3], joint_qd_prev[body_qd_start + 4], joint_qd_prev[body_qd_start + 5]
    )
    com_offset = wp.quat_rotate(wp.transform_get_rotation(body_q_prev[body]), body_com[body])
    origin_v_prev = com_v_prev - wp.cross(omega_prev, com_offset)
    origin_v = (wp.transform_get_translation(body_q[body]) - wp.transform_get_translation(body_q_prev[body])) * inv_dt
    a_local = body_R_T * ((origin_v - origin_v_prev) * inv_dt)
    omega = quat_velocity(body_rot, wp.transform_get_rotation(body_q_prev[body]), dt)
    alpha_local = body_R_T * ((omega - omega_prev) * inv_dt)
    omega_local = body_R_T * omega
    omega_skew = wp.skew(omega_local)
    omega_skew_sq = omega_skew * omega_skew

    for i in range(block_width):
        elastic_body_block_grad[elastic_index, i] = 0.0
        elastic_body_block_delta[elastic_index, i] = 0.0
        for j in range(block_width):
            elastic_body_block_matrix[elastic_index, i, j] = 0.0

    frame_dynamic = solve_frame and body_inv_mass[body] > 0.0
    if frame_dynamic:
        q_current = body_q[body]
        q_inertial = body_inertia_q[body]
        body_com_local = body_com[body]
        pos_current = wp.transform_get_translation(q_current)
        rot_current = wp.transform_get_rotation(q_current)
        pos_star = wp.transform_get_translation(q_inertial)
        rot_star = wp.transform_get_rotation(q_inertial)
        com_current = pos_current + wp.quat_rotate(rot_current, body_com_local)
        com_star = pos_star + wp.quat_rotate(rot_star, body_com_local)

        inertial_coeff = body_mass[body] * inv_dt_sq
        frame_force_world = (com_star - com_current) * inertial_coeff + external_forces[body]

        q_delta = wp.mul(wp.quat_inverse(rot_current), rot_star)
        if q_delta[3] < 0.0:
            q_delta = wp.quat(-q_delta[0], -q_delta[1], -q_delta[2], -q_delta[3])
        axis_body, angle_body = wp.quat_to_axis_angle(q_delta)
        frame_torque_local = body_inertia[body] * (axis_body * angle_body * inv_dt_sq)
        frame_torque_world = wp.quat_rotate(rot_current, frame_torque_local) + external_torques[body]

        h_ll_world = external_hessian_ll[body] + inertial_coeff * wp.identity(3, float)
        h_al_world = external_hessian_al[body]
        h_aa_world = external_hessian_aa[body] + body_R * (body_inertia[body] * inv_dt_sq) * body_R_T

        force_local = body_R_T * frame_force_world
        torque_local = body_R_T * frame_torque_world
        h_ll_local = body_R_T * (h_ll_world * body_R)
        h_al_local = body_R_T * (h_al_world * body_R)
        h_aa_local = body_R_T * (h_aa_world * body_R)

        for i in range(3):
            elastic_body_block_grad[elastic_index, i] = -force_local[i]
            elastic_body_block_grad[elastic_index, 3 + i] = -torque_local[i]
            for j in range(3):
                elastic_body_block_matrix[elastic_index, i, j] = h_ll_local[i, j]
                elastic_body_block_matrix[elastic_index, 3 + i, j] = h_al_local[i, j]
                elastic_body_block_matrix[elastic_index, i, 3 + j] = h_al_local[j, i]
                elastic_body_block_matrix[elastic_index, 3 + i, 3 + j] = h_aa_local[i, j]
    else:
        for i in range(6):
            elastic_body_block_matrix[elastic_index, i, i] = 1.0

    for mode in range(mode_count, max_modes):
        elastic_body_block_matrix[elastic_index, 6 + mode, 6 + mode] = 1.0

    for mode in range(mode_count):
        mode_data = mode_start + mode
        q_idx = q_start + mode
        qd_idx = qd_start + mode

        q = joint_q[q_idx]
        q_prev = joint_q_prev[q_idx]
        v_prev = joint_qd_prev[qd_idx]
        mass = elastic_mode_mass[mode_data]
        stiffness = elastic_mode_stiffness[mode_data]
        damping = elastic_mode_damping[mode_data]
        coriolis = wp.vec3(0.0, 0.0, 0.0)
        for j in range(mode_count):
            coriolis = (
                coriolis
                + elastic_mode_coupling_coriolis[mode_coupling_mat_start + mode * max_modes + j]
                * joint_qd_prev[qd_start + j]
            )
        force = (
            joint_f[qd_idx]
            + wp.dot(elastic_mode_coupling_linear[mode_data], g_local - a_local)
            + wp.dot(elastic_mode_coupling_angular[mode_data], alpha_local)
            - wp.ddot(omega_skew_sq, elastic_mode_coupling_centrifugal[mode_data])
            - 2.0 * wp.dot(omega_local, coriolis)
        )

        h = stiffness
        grad = stiffness * q - force
        if mass > 0.0:
            h = h + mass * inv_dt_sq
            grad = grad + mass * inv_dt_sq * (q - q_prev - dt * v_prev)
        if damping > 0.0:
            h = h + damping * inv_dt
            grad = grad + damping * inv_dt * (q - q_prev)

        elastic_body_block_grad[elastic_index, 6 + mode] = grad
        elastic_body_block_matrix[elastic_index, 6 + mode, 6 + mode] = h

        if frame_dynamic and elastic_implicit_mass_coupling[elastic_index]:
            coupling_linear = elastic_mode_coupling_linear[mode_data]
            coupling_torque = elastic_mode_coupling_angular[mode_data] + wp.cross(body_com[body], coupling_linear)
            for axis in range(3):
                elastic_body_block_matrix[elastic_index, axis, 6 + mode] = (
                    elastic_body_block_matrix[elastic_index, axis, 6 + mode] + coupling_linear[axis] * inv_dt_sq
                )
                elastic_body_block_matrix[elastic_index, 3 + axis, 6 + mode] = (
                    elastic_body_block_matrix[elastic_index, 3 + axis, 6 + mode] - coupling_torque[axis] * inv_dt_sq
                )

    adjacent_joint_count = get_body_num_adjacent_joints(adjacency, body)
    for adjacent_joint in range(adjacent_joint_count):
        joint = get_body_adjacent_joint_id(adjacency, body, adjacent_joint)
        is_parent = joint_parent[joint] == body
        endpoint = joint_parent_elastic_endpoint[joint] if is_parent else joint_child_elastic_endpoint[joint]
        if endpoint < 0:
            continue

        joint_force, joint_torque, joint_H_ll, joint_H_al, joint_H_aa = evaluate_joint_force_hessian(
            body,
            joint,
            body_q,
            body_q_prev,
            body_q_rest,
            body_com,
            joint_type,
            joint_enabled,
            joint_parent,
            joint_child,
            joint_X_p,
            joint_X_c,
            body_elastic_index,
            elastic_joint,
            elastic_mode_count,
            joint_q,
            joint_q_prev,
            joint_q_start,
            joint_parent_elastic_endpoint,
            joint_child_elastic_endpoint,
            elastic_endpoint_phi,
            elastic_endpoint_psi,
            elastic_max_mode_count,
            joint_axis,
            joint_rod_rest_kb_local,
            joint_rod_rest_twist,
            joint_qd_start,
            joint_target_q_start,
            joint_constraint_start,
            joint_penalty_k,
            joint_rho,
            joint_material_k,
            joint_penalty_kd,
            joint_sigma_start,
            joint_C_fric,
            joint_target_ke,
            joint_target_kd,
            joint_target_q,
            joint_target_qd,
            joint_limit_lower,
            joint_limit_upper,
            joint_limit_ke,
            joint_limit_kd,
            joint_drive_limit_support,
            joint_drive_lambda,
            joint_limit_lambda,
            joint_lambda_lin,
            joint_lambda_ang,
            joint_C0_lin,
            joint_C0_ang,
            joint_is_hard,
            stab_alpha,
            joint_compliant_alm,
            joint_dof_dim,
            joint_rest_angle,
            dt,
        )

        endpoint_xform = _elastic_endpoint_xform(
            joint,
            body,
            is_parent,
            joint_X_p[joint] if is_parent else joint_X_c[joint],
            body_elastic_index,
            elastic_joint,
            elastic_mode_count,
            joint_q,
            joint_q_start,
            joint_parent_elastic_endpoint,
            joint_child_elastic_endpoint,
            elastic_endpoint_phi,
            elastic_endpoint_psi,
            elastic_max_mode_count,
        )
        endpoint_world = body_q[body] * endpoint_xform
        endpoint_pos_world = wp.transform_get_translation(endpoint_world)
        com_world = wp.transform_point(body_q[body], body_com[body])
        r_world = endpoint_pos_world - com_world
        r_skew = wp.skew(r_world)

        linear_H_aa = wp.transpose(r_skew) * joint_H_ll * r_skew
        angular_torque = joint_torque - wp.cross(r_world, joint_force)
        angular_H_aa = joint_H_aa - linear_H_aa

        force_local = body_R_T * joint_force
        H_ll_local = body_R_T * (joint_H_ll * body_R)
        H_al_local = body_R_T * (joint_H_al * body_R)
        torque_local = body_R_T * joint_torque
        H_aa_local = body_R_T * (joint_H_aa * body_R)
        angular_torque_local = body_R_T * angular_torque
        angular_H_aa_local = body_R_T * (angular_H_aa * body_R)

        if frame_dynamic:
            for i in range(3):
                elastic_body_block_grad[elastic_index, i] = elastic_body_block_grad[elastic_index, i] - force_local[i]
                elastic_body_block_grad[elastic_index, 3 + i] = (
                    elastic_body_block_grad[elastic_index, 3 + i] - torque_local[i]
                )
                for j in range(3):
                    elastic_body_block_matrix[elastic_index, i, j] = (
                        elastic_body_block_matrix[elastic_index, i, j] + H_ll_local[i, j]
                    )
                    elastic_body_block_matrix[elastic_index, 3 + i, j] = (
                        elastic_body_block_matrix[elastic_index, 3 + i, j] + H_al_local[i, j]
                    )
                    elastic_body_block_matrix[elastic_index, i, 3 + j] = (
                        elastic_body_block_matrix[elastic_index, i, 3 + j] + H_al_local[j, i]
                    )
                    elastic_body_block_matrix[elastic_index, 3 + i, 3 + j] = (
                        elastic_body_block_matrix[elastic_index, 3 + i, 3 + j] + H_aa_local[i, j]
                    )

        theta_local = wp.vec3(0.0)
        for mode in range(mode_count):
            theta_local = (
                theta_local + elastic_endpoint_psi[endpoint * elastic_max_mode_count + mode] * joint_q[q_start + mode]
            )
        endpoint_rotation_jacobian = _so3_left_jacobian(theta_local)

        for i in range(mode_count):
            phi_i_local = elastic_endpoint_phi[endpoint * elastic_max_mode_count + i]
            psi_i_local = endpoint_rotation_jacobian * elastic_endpoint_psi[endpoint * elastic_max_mode_count + i]
            mode_i = 6 + i
            elastic_body_block_grad[elastic_index, mode_i] = (
                elastic_body_block_grad[elastic_index, mode_i]
                - wp.dot(force_local, phi_i_local)
                - wp.dot(angular_torque_local, psi_i_local)
            )

            if frame_dynamic:
                linear_cross = H_ll_local * phi_i_local
                angular_cross = H_al_local * phi_i_local + angular_H_aa_local * psi_i_local
                for axis in range(3):
                    elastic_body_block_matrix[elastic_index, axis, mode_i] = (
                        elastic_body_block_matrix[elastic_index, axis, mode_i] + linear_cross[axis]
                    )
                    elastic_body_block_matrix[elastic_index, 3 + axis, mode_i] = (
                        elastic_body_block_matrix[elastic_index, 3 + axis, mode_i] + angular_cross[axis]
                    )

            for j in range(i, mode_count):
                phi_j_local = elastic_endpoint_phi[endpoint * elastic_max_mode_count + j]
                psi_j_local = endpoint_rotation_jacobian * elastic_endpoint_psi[endpoint * elastic_max_mode_count + j]
                mode_j = 6 + j
                elastic_body_block_matrix[elastic_index, mode_i, mode_j] = (
                    elastic_body_block_matrix[elastic_index, mode_i, mode_j]
                    + wp.dot(phi_i_local, H_ll_local * phi_j_local)
                    + wp.dot(psi_i_local, angular_H_aa_local * psi_j_local)
                )

    for i in range(block_width):
        diagonal = elastic_body_block_matrix[elastic_index, i, i]
        elastic_body_block_matrix[elastic_index, i, i] = diagonal + 1.0e-9 * (wp.abs(diagonal) + 1.0)


@wp.kernel
def accumulate_elastic_frame_coupling(
    dt: float,
    elastic_body: wp.array[wp.int32],
    elastic_joint: wp.array[wp.int32],
    elastic_mode_start: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    elastic_mode_coupling_linear: wp.array[wp.vec3],
    elastic_mode_coupling_angular: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_q: wp.array[float],
    joint_q_prev: wp.array[float],
    joint_qd_prev: wp.array[float],
    body_forces: wp.array[wp.vec3],
    body_torques: wp.array[wp.vec3],
):
    elastic_index = wp.tid()
    body = elastic_body[elastic_index]
    owner_joint = elastic_joint[elastic_index]
    q_start = joint_q_start[owner_joint] + 7
    qd_start = joint_qd_start[owner_joint] + 6
    mode_start = elastic_mode_start[elastic_index]
    mode_count = elastic_mode_count[elastic_index]

    inv_dt = 1.0 / dt
    inv_dt_sq = inv_dt * inv_dt
    body_rot = wp.transform_get_rotation(body_q[body])
    body_R_T = wp.transpose(wp.quat_to_matrix(body_rot))
    com = body_com[body]

    origin_v = (wp.transform_get_translation(body_q[body]) - wp.transform_get_translation(body_q_prev[body])) * inv_dt
    origin_v_local = body_R_T * origin_v
    omega = quat_velocity(body_rot, wp.transform_get_rotation(body_q_prev[body]), dt)
    omega_local = body_R_T * omega

    s_qd = wp.vec3(0.0, 0.0, 0.0)
    g_qd = wp.vec3(0.0, 0.0, 0.0)
    s_qdd = wp.vec3(0.0, 0.0, 0.0)
    g_qdd = wp.vec3(0.0, 0.0, 0.0)
    for i in range(mode_count):
        mode_data = mode_start + i
        dq = joint_q[q_start + i] - joint_q_prev[q_start + i]
        qdot = dq * inv_dt
        qddot = (dq - dt * joint_qd_prev[qd_start + i]) * inv_dt_sq
        s_qd = s_qd + elastic_mode_coupling_linear[mode_data] * qdot
        g_qd = g_qd + elastic_mode_coupling_angular[mode_data] * qdot
        s_qdd = s_qdd + elastic_mode_coupling_linear[mode_data] * qddot
        g_qdd = g_qdd + elastic_mode_coupling_angular[mode_data] * qddot

    omega_cross_s_qd = wp.cross(omega_local, s_qd)
    force_local = -s_qdd - omega_cross_s_qd
    torque_local = (
        g_qdd
        + wp.cross(com, s_qdd)
        + wp.cross(omega_local, g_qd)
        + wp.cross(com, omega_cross_s_qd)
        - wp.cross(origin_v_local, s_qd)
    )

    wp.atomic_add(body_forces, body, wp.quat_rotate(body_rot, force_local))
    wp.atomic_add(body_torques, body, wp.quat_rotate(body_rot, torque_local))


@wp.kernel
def assemble_elastic_contacts(
    dt: float,
    solve_frame: bool,
    body_ids: wp.array[wp.int32],
    elastic_joint: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    elastic_max_mode_count: int,
    body_elastic_index: wp.array[wp.int32],
    body_inv_mass: wp.array[float],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    rigid_contact_max: int,
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_offset0: wp.array[wp.vec3],
    rigid_contact_offset1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_margin0: wp.array[float],
    rigid_contact_margin1: wp.array[float],
    rigid_contact_elastic_sample0: wp.array[wp.int32],
    rigid_contact_elastic_sample1: wp.array[wp.int32],
    contact_penalty_k: wp.array[float],
    contact_normal_rho: wp.array[float],
    contact_material_ke: wp.array[float],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    contact_tangent_rho: wp.array[float],
    contact_lambda: wp.array[wp.vec3],
    contact_C0: wp.array[wp.vec3],
    stab_alpha: float,
    legacy_hard_contacts: int,
    contact_compliant_alm: int,
    friction_epsilon: float,
    body_contact_buffer_pre_alloc: int,
    body_contact_counts: wp.array[wp.int32],
    body_contact_indices: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    elastic_shape_vertex_local: wp.array[wp.vec3],
    elastic_shape_vertex_phi: wp.array[wp.vec3],
    joint_q_prev: wp.array[float],
    joint_q: wp.array[float],
    elastic_body_block_grad: wp.array2d[float],
    elastic_body_block_matrix: wp.array3d[float],
):
    tid = wp.tid()
    body_group_index = tid // _NUM_ELASTIC_CONTACT_THREADS_PER_BODY
    thread_id_within_body = tid % _NUM_ELASTIC_CONTACT_THREADS_PER_BODY

    body = body_ids[body_group_index]
    elastic_index = body_elastic_index[body]
    if elastic_index < 0:
        return

    body_rot = wp.transform_get_rotation(body_q[body])
    body_R = wp.quat_to_matrix(body_rot)
    body_R_T = wp.transpose(body_R)

    contact_limit = rigid_contact_count[0]
    if contact_limit > rigid_contact_max:
        contact_limit = rigid_contact_max

    # Use the rigid per-body list when it is complete; otherwise stride the
    # full contact range so contact overflow never truncates reduced forces.
    use_contact_list = False
    contact_count = wp.int32(0)
    if body_contact_buffer_pre_alloc > 0:
        contact_count = body_contact_counts[body]
        use_contact_list = contact_count <= body_contact_buffer_pre_alloc

    if use_contact_list:
        contact_i = thread_id_within_body
        while contact_i < contact_count:
            contact_idx = body_contact_indices[body * body_contact_buffer_pre_alloc + contact_i]
            if contact_idx < contact_limit:
                _accumulate_elastic_contact_modes(
                    dt,
                    solve_frame,
                    contact_idx,
                    body,
                    elastic_index,
                    body_R,
                    body_R_T,
                    elastic_joint,
                    elastic_mode_count,
                    elastic_max_mode_count,
                    body_elastic_index,
                    body_inv_mass,
                    body_q,
                    body_q_prev,
                    body_com,
                    shape_body,
                    rigid_contact_shape0,
                    rigid_contact_shape1,
                    rigid_contact_point0,
                    rigid_contact_point1,
                    rigid_contact_offset0,
                    rigid_contact_offset1,
                    rigid_contact_normal,
                    rigid_contact_margin0,
                    rigid_contact_margin1,
                    rigid_contact_elastic_sample0,
                    rigid_contact_elastic_sample1,
                    contact_penalty_k,
                    contact_normal_rho,
                    contact_material_ke,
                    contact_material_kd,
                    contact_material_mu,
                    contact_tangent_rho,
                    contact_lambda,
                    contact_C0,
                    stab_alpha,
                    legacy_hard_contacts,
                    contact_compliant_alm,
                    friction_epsilon,
                    joint_q_start,
                    elastic_shape_vertex_local,
                    elastic_shape_vertex_phi,
                    joint_q_prev,
                    joint_q,
                    elastic_body_block_grad,
                    elastic_body_block_matrix,
                )

            contact_i += _NUM_ELASTIC_CONTACT_THREADS_PER_BODY
    else:
        contact_idx = thread_id_within_body
        while contact_idx < contact_limit:
            _accumulate_elastic_contact_modes(
                dt,
                solve_frame,
                contact_idx,
                body,
                elastic_index,
                body_R,
                body_R_T,
                elastic_joint,
                elastic_mode_count,
                elastic_max_mode_count,
                body_elastic_index,
                body_inv_mass,
                body_q,
                body_q_prev,
                body_com,
                shape_body,
                rigid_contact_shape0,
                rigid_contact_shape1,
                rigid_contact_point0,
                rigid_contact_point1,
                rigid_contact_offset0,
                rigid_contact_offset1,
                rigid_contact_normal,
                rigid_contact_margin0,
                rigid_contact_margin1,
                rigid_contact_elastic_sample0,
                rigid_contact_elastic_sample1,
                contact_penalty_k,
                contact_normal_rho,
                contact_material_ke,
                contact_material_kd,
                contact_material_mu,
                contact_tangent_rho,
                contact_lambda,
                contact_C0,
                stab_alpha,
                legacy_hard_contacts,
                contact_compliant_alm,
                friction_epsilon,
                joint_q_start,
                elastic_shape_vertex_local,
                elastic_shape_vertex_phi,
                joint_q_prev,
                joint_q,
                elastic_body_block_grad,
                elastic_body_block_matrix,
            )

            contact_idx += _NUM_ELASTIC_CONTACT_THREADS_PER_BODY


@functools.cache
def create_solve_elastic_body_tiled(block_width: int):
    """Create the coupled frame/modal block solve kernel for one block width.

    The assembled ``(6 + mode_count)`` block is solved by a dense Cholesky factorization.
    Blocks that are not positive definite fall back to a Jacobi step.
    """
    width = int(block_width)

    def _solve_elastic_body_tiled(
        dt: float,
        solve_frame: bool,
        body_ids: wp.array[wp.int32],
        body_elastic_index: wp.array[wp.int32],
        elastic_joint: wp.array[wp.int32],
        elastic_mode_count: wp.array[wp.int32],
        body_inv_mass: wp.array[float],
        body_com: wp.array[wp.vec3],
        body_q: wp.array[wp.transform],
        joint_q_start: wp.array[wp.int32],
        joint_qd_start: wp.array[wp.int32],
        joint_q_prev: wp.array[float],
        elastic_body_block_grad: wp.array2d[float],
        elastic_body_block_delta: wp.array2d[float],
        elastic_body_block_matrix: wp.array3d[float],
        elastic_body_block_initial_residual_norm: wp.array[float],
        elastic_body_block_solve_residual_norm: wp.array[float],
        elastic_body_block_applied_residual_norm: wp.array[float],
        elastic_body_block_update_norm: wp.array[float],
        elastic_body_block_update_max: wp.array[float],
        elastic_body_relaxation: float,
        body_q_new: wp.array[wp.transform],
        joint_q: wp.array[float],
        joint_qd: wp.array[float],
    ):
        body = body_ids[wp.tid()]
        elastic_index = body_elastic_index[body]
        if elastic_index < 0:
            return

        h = wp.tile_zeros(shape=(width, width), dtype=float)
        g = wp.tile_zeros(shape=(width,), dtype=float)
        for i in range(width):
            g[i] = elastic_body_block_grad[elastic_index, i]
            for j in range(width):
                if j < i:
                    h[i, j] = elastic_body_block_matrix[elastic_index, j, i]
                else:
                    h[i, j] = elastic_body_block_matrix[elastic_index, i, j]

        rhs = wp.tile_map(wp.neg, g)
        factor = wp.tile_cholesky(h)
        delta = wp.tile_cholesky_solve(factor, rhs)

        block_is_definite = int(1)
        for i in range(width):
            if not wp.isfinite(delta[i]):
                block_is_definite = int(0)

        safe_delta = wp.tile_zeros(shape=(width,), dtype=float)
        for i in range(width):
            if block_is_definite == 1:
                safe_delta[i] = delta[i]
            else:
                diagonal = h[i, i]
                if diagonal > 0.0:
                    safe_delta[i] = -g[i] / diagonal
            elastic_body_block_delta[elastic_index, i] = safe_delta[i]

        owner_joint = elastic_joint[elastic_index]
        q_start = joint_q_start[owner_joint] + 7
        qd_start = joint_qd_start[owner_joint] + 6
        mode_count = elastic_mode_count[elastic_index]

        if elastic_body_block_initial_residual_norm:
            initial_residual_sq = float(0.0)
            solve_residual_sq = float(0.0)
            applied_residual_sq = float(0.0)
            update_norm_sq = float(0.0)
            update_max = float(0.0)
            for row in range(width):
                if row < 6:
                    row_is_solved = solve_frame
                else:
                    row_is_solved = row - 6 < mode_count
                if row_is_solved:
                    grad_i = g[row]
                    solve_residual_i = grad_i
                    applied_residual_i = grad_i
                    for j in range(width):
                        delta_j = safe_delta[j]
                        solve_residual_i = solve_residual_i + h[row, j] * delta_j
                        applied_residual_i = applied_residual_i + h[row, j] * (elastic_body_relaxation * delta_j)

                    applied_delta_i = elastic_body_relaxation * safe_delta[row]
                    initial_residual_sq = initial_residual_sq + grad_i * grad_i
                    solve_residual_sq = solve_residual_sq + solve_residual_i * solve_residual_i
                    applied_residual_sq = applied_residual_sq + applied_residual_i * applied_residual_i
                    update_norm_sq = update_norm_sq + applied_delta_i * applied_delta_i
                    update_max = wp.max(update_max, wp.abs(applied_delta_i))

            elastic_body_block_initial_residual_norm[elastic_index] = wp.sqrt(initial_residual_sq)
            elastic_body_block_solve_residual_norm[elastic_index] = wp.sqrt(solve_residual_sq)
            elastic_body_block_applied_residual_norm[elastic_index] = wp.sqrt(applied_residual_sq)
            elastic_body_block_update_norm[elastic_index] = wp.sqrt(update_norm_sq)
            elastic_body_block_update_max[elastic_index] = update_max

        inv_dt = 1.0 / dt
        q_current = body_q[body]
        if solve_frame and body_inv_mass[body] > 0.0:
            rot_current = wp.transform_get_rotation(q_current)
            body_R = wp.quat_to_matrix(rot_current)
            x_delta = wp.vec3(safe_delta[0], safe_delta[1], safe_delta[2])
            w_delta = wp.vec3(safe_delta[3], safe_delta[4], safe_delta[5])
            x_inc_world = body_R * (elastic_body_relaxation * x_delta)
            w_world = body_R * (elastic_body_relaxation * w_delta)
            angle = wp.length(w_world)
            if angle > _SMALL_ANGLE_EPS:
                dq_world = wp.quat_from_axis_angle(w_world / angle, angle)
            else:
                half_w = 0.5 * w_world
                dq_world = wp.normalize(wp.quat(half_w[0], half_w[1], half_w[2], 1.0))
            rot_new = wp.normalize(dq_world * rot_current)
            com_current = wp.transform_point(q_current, body_com[body])
            com_new = com_current + x_inc_world
            pos_new = com_new - wp.quat_rotate(rot_new, body_com[body])
            body_q_new[body] = wp.transform(pos_new, rot_new)
        else:
            body_q_new[body] = q_current

        for mode in range(mode_count):
            q_idx = q_start + mode
            qd_idx = qd_start + mode
            q_prev = joint_q_prev[q_idx]
            q_old = joint_q[q_idx]
            q_new = q_old + elastic_body_relaxation * safe_delta[6 + mode]
            joint_q[q_idx] = q_new
            joint_qd[qd_idx] = (q_new - q_prev) * inv_dt

    _solve_elastic_body_tiled.__name__ = f"solve_elastic_body_tiled_{width}"
    _solve_elastic_body_tiled.__qualname__ = f"solve_elastic_body_tiled_{width}"
    return wp.kernel(enable_backward=False, module="unique")(_solve_elastic_body_tiled)


@wp.kernel
def copy_body_frame_to_elastic_joint(
    elastic_body: wp.array[wp.int32],
    elastic_joint: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    elastic_index = wp.tid()
    body = elastic_body[elastic_index]
    joint = elastic_joint[elastic_index]
    q_start = joint_q_start[joint]
    qd_start = joint_qd_start[joint]

    X_wb = body_q[body]
    p = wp.transform_get_translation(X_wb)
    q = wp.transform_get_rotation(X_wb)
    v = body_qd[body]
    lin = wp.spatial_top(v)
    ang = wp.spatial_bottom(v)

    joint_q[q_start + 0] = p[0]
    joint_q[q_start + 1] = p[1]
    joint_q[q_start + 2] = p[2]
    joint_q[q_start + 3] = q[0]
    joint_q[q_start + 4] = q[1]
    joint_q[q_start + 5] = q[2]
    joint_q[q_start + 6] = q[3]

    joint_qd[qd_start + 0] = lin[0]
    joint_qd[qd_start + 1] = lin[1]
    joint_qd[qd_start + 2] = lin[2]
    joint_qd[qd_start + 3] = ang[0]
    joint_qd[qd_start + 4] = ang[1]
    joint_qd[qd_start + 5] = ang[2]
