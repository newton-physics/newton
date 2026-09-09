# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the VBD solver."""

import math
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.geometry.tri_mesh_collision import TriMeshCollisionInfo, build_tri_mesh_collision_info
from newton._src.solvers.vbd.particle_vbd_kernels import (
    NUM_THREADS_PER_COLLISION_PRIMITIVE,
    TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
    apply_planar_truncation_parallel_by_collision,
    build_particle_body_contact_adjacency_active,
    create_edge_edge_division_plane_closest_pt,
    create_vertex_triangle_division_plane_closest_pt,
    evaluate_dihedral_angle_based_bending_force_hessian,
    evaluate_neo_hookean_membrane_force_hessian,
    evaluate_self_contact_force_norm,
    evaluate_spring_force_and_hessian,
    evaluate_spring_force_and_hessian_both_vertices,
    evaluate_vertex_triangle_collision_force_hessian_4_vertices,
    evaluate_volumetric_neo_hookean_force_and_hessian,
    gather_particle_body_contact_force_and_hessian,
    make_solve_elasticity_tile,
)
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    RigidContactHistory,
    _alm_relaxed_ascent,
    _compliant_alm_coefficients,
    _contact_tangent_conditioning_scale,
    _eval_body_particle_contact,
    _eval_soft_ef_contact,
    _evaluate_rigid_soft_contact_force_norm,
    _joint_angular_rho_seed,
    build_body_body_contact_lists,
    build_body_particle_contact_lists,
    compute_rigid_contact_forces,
    evaluate_angular_constraint_force_hessian,
    evaluate_body_particle_contact,
    evaluate_linear_constraint_force_hessian,
    evaluate_rigid_contact_from_collision,
    init_body_body_contacts_alm,
    init_body_particle_contacts,
    place_dat_division_plane,
    planar_truncation_t,
    rigid_point_trajectory,
    rigid_trajectory_truncation_t,
    snapshot_body_body_contact_history,
    step_body_body_contact_C0_lambda,
    update_duals_body_body_contacts,
    update_duals_body_particle_contacts,
    update_duals_joint,
)
from newton._src.solvers.vbd.solver_vbd import _PARTICLE_CONTACT_GATHER_BLOCK_DIM, _is_tet_only_elasticity_model
from newton.solvers.experimental.coupled import SolverCoupledProxy
from newton.tests.unittest_utils import (
    add_function_test,
    configure_sdf_for_collision_shapes,
    get_test_devices,
)

devices = get_test_devices()
cpu_devices = [device for device in devices if device.is_cpu]
cuda_devices = [device for device in devices if device.is_cuda]


def _quat_rotate_np(q, v):
    q_vec = np.asarray(q[:3], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(q_vec, v)
    return v + float(q[3]) * t + np.cross(q_vec, t)


def _transform_point_np(xform, point):
    return np.asarray(xform[:3], dtype=np.float64) + _quat_rotate_np(xform[3:], point)


def _transform_contact_point_np(body_q, body_id, local_point):
    if body_id < 0:
        return np.asarray(local_point, dtype=np.float64)
    return _transform_point_np(body_q[body_id], local_point)


def _random_rotation_matrices(count, seed):
    rng = np.random.default_rng(seed)
    quat = rng.normal(size=(count, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return _rotation_matrices_from_quaternions(quat)


def _random_quaternions(count, seed):
    rng = np.random.default_rng(seed)
    quat = rng.normal(size=(count, 4)).astype(np.float32)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return quat


def _rotation_matrices_from_quaternions(quat):
    x = quat[:, 0]
    y = quat[:, 1]
    z = quat[:, 2]
    w = quat[:, 3]

    rotations = np.empty((quat.shape[0], 3, 3), dtype=np.float32)
    rotations[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotations[:, 0, 1] = 2.0 * (x * y - z * w)
    rotations[:, 0, 2] = 2.0 * (x * z + y * w)
    rotations[:, 1, 0] = 2.0 * (x * y + z * w)
    rotations[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotations[:, 1, 2] = 2.0 * (y * z - x * w)
    rotations[:, 2, 0] = 2.0 * (x * z - y * w)
    rotations[:, 2, 1] = 2.0 * (y * z + x * w)
    rotations[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rotations


def _contact_damping_rigid_motion_data(sample_count=100, seed=29):
    quats = _random_quaternions(sample_count, seed)
    rotations = _rotation_matrices_from_quaternions(quats)
    rng = np.random.default_rng(seed + 1)
    translations = rng.uniform(-1.0, 1.0, size=(sample_count, 3)).astype(np.float32)

    normal_rest = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    contact_distance = np.float32(0.04)
    body_rest = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    particle_rest = body_rest + contact_distance * normal_rest
    rigid_a_rest = np.array([-0.2, 0.1, 0.0], dtype=np.float32)
    rigid_b_rest = rigid_a_rest + contact_distance * normal_rest
    soft_rest = np.array(
        [
            [-0.6, -0.5, 0.0],
            [0.7, -0.4, 0.0],
            [0.1, 0.8, 0.0],
            [0.05, 0.1, contact_distance],
        ],
        dtype=np.float32,
    )

    body_q_prev = np.empty((sample_count, 7), dtype=np.float32)
    body_q = np.empty((sample_count, 7), dtype=np.float32)
    rigid_body_q_prev = np.empty((2 * sample_count, 7), dtype=np.float32)
    rigid_body_q = np.empty((2 * sample_count, 7), dtype=np.float32)
    particle_q_prev = np.empty((sample_count, 3), dtype=np.float32)
    particle_q = np.empty((sample_count, 3), dtype=np.float32)
    contact_normal = np.empty((sample_count, 3), dtype=np.float32)
    soft_pos_anchor = np.empty((4 * sample_count, 3), dtype=np.float32)
    soft_pos = np.empty((4 * sample_count, 3), dtype=np.float32)
    tri_indices = np.empty((sample_count, 3), dtype=np.int32)

    identity_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    for sample in range(sample_count):
        R = rotations[sample]
        t = translations[sample]
        q = quats[sample]

        body_q_prev[sample, :3] = body_rest
        body_q_prev[sample, 3:] = identity_quat
        body_q[sample, :3] = body_rest @ R.T + t
        body_q[sample, 3:] = q

        rigid_start = 2 * sample
        rigid_body_q_prev[rigid_start, :3] = rigid_a_rest
        rigid_body_q_prev[rigid_start, 3:] = identity_quat
        rigid_body_q_prev[rigid_start + 1, :3] = rigid_b_rest
        rigid_body_q_prev[rigid_start + 1, 3:] = identity_quat
        rigid_body_q[rigid_start, :3] = rigid_a_rest @ R.T + t
        rigid_body_q[rigid_start, 3:] = q
        rigid_body_q[rigid_start + 1, :3] = rigid_b_rest @ R.T + t
        rigid_body_q[rigid_start + 1, 3:] = q

        particle_q_prev[sample] = particle_rest
        particle_q[sample] = particle_rest @ R.T + t
        contact_normal[sample] = normal_rest @ R.T

        soft_start = 4 * sample
        soft_pos_anchor[soft_start : soft_start + 4] = soft_rest
        soft_pos[soft_start : soft_start + 4] = soft_rest @ R.T + t
        tri_indices[sample] = [soft_start, soft_start + 1, soft_start + 2]

    return {
        "body_q_prev": body_q_prev,
        "body_q": body_q,
        "rigid_body_q_prev": rigid_body_q_prev,
        "rigid_body_q": rigid_body_q,
        "particle_q_prev": particle_q_prev,
        "particle_q": particle_q,
        "contact_normal": contact_normal,
        "soft_pos_anchor": soft_pos_anchor,
        "soft_pos": soft_pos,
        "tri_indices": tri_indices,
    }


def _elastic_damping_rigid_motion_data(sample_count=100, seed=17):
    rest = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.1, 0.0],
            [0.2, 1.1, 0.1],
            [0.1, 0.2, 1.3],
        ],
        dtype=np.float32,
    )

    rotations = _random_rotation_matrices(sample_count, seed)
    rng = np.random.default_rng(seed + 1)
    translations = rng.uniform(-1.0, 1.0, size=(sample_count, 3)).astype(np.float32)

    pos_anchor = np.tile(rest, (sample_count, 1))
    pos = np.empty_like(pos_anchor)
    for sample in range(sample_count):
        start = 4 * sample
        pos[start : start + 4] = rest @ rotations[sample].T + translations[sample]

    particle_ids = np.arange(sample_count, dtype=np.int32)[:, None] * 4
    spring_indices = np.column_stack((particle_ids[:, 0], particle_ids[:, 0] + 1)).astype(np.int32).reshape(-1)
    tri_indices = np.column_stack((particle_ids[:, 0], particle_ids[:, 0] + 1, particle_ids[:, 0] + 2)).astype(np.int32)
    tet_indices = np.column_stack(
        (particle_ids[:, 0], particle_ids[:, 0] + 1, particle_ids[:, 0] + 2, particle_ids[:, 0] + 3)
    ).astype(np.int32)
    edge_indices = np.column_stack(
        (particle_ids[:, 0], particle_ids[:, 0] + 1, particle_ids[:, 0] + 2, particle_ids[:, 0] + 3)
    ).astype(np.int32)

    qp = rest[1] - rest[0]
    rp = rest[2] - rest[0]
    tri_normal = np.cross(qp, rp)
    tri_normal /= np.linalg.norm(tri_normal)
    e1 = qp / np.linalg.norm(qp)
    e2 = np.cross(tri_normal, e1)
    e2 /= np.linalg.norm(e2)
    tri_D = np.array((e1, e2), dtype=np.float32) @ np.array((qp, rp), dtype=np.float32).T
    tri_pose = np.linalg.inv(tri_D).astype(np.float32)
    tri_area = np.float32(np.linalg.det(tri_D) * 0.5)

    tet_Dm = np.array((rest[1] - rest[0], rest[2] - rest[0], rest[3] - rest[0]), dtype=np.float32).T
    tet_pose = np.linalg.inv(tet_Dm).astype(np.float32)

    x1, x2, x3, x4 = rest[0], rest[1], rest[2], rest[3]
    n1 = np.cross(x3 - x1, x4 - x1)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(x4 - x2, x3 - x2)
    n2 /= np.linalg.norm(n2)
    edge_dir = x4 - x3
    edge_dir /= np.linalg.norm(edge_dir)
    edge_rest_angle = np.float32(math.atan2(np.dot(np.cross(n1, n2), edge_dir), np.dot(n1, n2)))
    edge_rest_length = np.float32(np.linalg.norm(x4 - x3))

    return {
        "pos": pos,
        "pos_anchor": pos_anchor,
        "spring_indices": spring_indices,
        "spring_rest_length": np.full(sample_count, np.linalg.norm(rest[1] - rest[0]), dtype=np.float32),
        "spring_stiffness": np.zeros(sample_count, dtype=np.float32),
        "spring_damping": np.full(sample_count, 20.0, dtype=np.float32),
        "tri_indices": tri_indices,
        "tri_poses": np.tile(tri_pose, (sample_count, 1, 1)),
        "tri_areas": np.full(sample_count, tri_area, dtype=np.float32),
        "edge_indices": edge_indices,
        "edge_rest_angle": np.full(sample_count, edge_rest_angle, dtype=np.float32),
        "edge_rest_length": np.full(sample_count, edge_rest_length, dtype=np.float32),
        "tet_indices": tet_indices,
        "tet_poses": np.tile(tet_pose, (sample_count, 1, 1)),
    }


@wp.kernel
def _eval_self_contact_norm_kernel(
    distances: wp.array[float],
    collision_radius: float,
    k: float,
    dEdD_out: wp.array[float],
    d2E_out: wp.array[float],
):
    i = wp.tid()
    dEdD, d2E = evaluate_self_contact_force_norm(distances[i], collision_radius, k)
    dEdD_out[i] = dEdD
    d2E_out[i] = d2E


@wp.kernel
def _eval_rigid_soft_contact_norm_kernel(
    distances: wp.array[float],
    collision_radius: float,
    k: float,
    use_log_barrier: bool,
    dEdD_out: wp.array[float],
    d2E_out: wp.array[float],
):
    i = wp.tid()
    dEdD, d2E = _evaluate_rigid_soft_contact_force_norm(distances[i], collision_radius, k, use_log_barrier)
    dEdD_out[i] = dEdD
    d2E_out[i] = d2E


@wp.kernel
def _eval_compliant_alm_coefficients_kernel(
    material_k: wp.array[float],
    rho: wp.array[float],
    result: wp.array[wp.vec4],
):
    i = wp.tid()
    s, k_eff, a = _compliant_alm_coefficients(material_k[i], rho[i])
    lambda_next = _alm_relaxed_ascent(2.0, 0.25, material_k[i], rho[i])
    result[i] = wp.vec4(s, k_eff, a, lambda_next)


@wp.kernel
def _eval_joint_angular_rho_seed_kernel(
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    inv_dt_sq: float,
    rho_out: wp.array[float],
):
    rho_out[0] = _joint_angular_rho_seed(0, 1, body_inv_mass, body_inv_inertia, inv_dt_sq)


@wp.kernel
def _eval_crossed_contact_tangent_pair_support_kernel(
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    support: wp.array[float],
):
    normal = wp.vec3(0.0, 0.0, 1.0)
    anchor_x = wp.vec3(1.0, 0.0, 0.0)
    anchor_y = wp.vec3(0.0, 1.0, 0.0)
    support[0] = _contact_tangent_conditioning_scale(
        0, 1, anchor_x, anchor_y, normal, shape_body, body_q, body_com, body_inv_mass, body_inv_inertia, 1.0
    )


@wp.kernel
def _eval_compliant_sliding_contact_metric_kernel(
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    force_out: wp.array[wp.vec3],
    hessian_out: wp.array[wp.mat33],
):
    force_a, _torque_a, hessian_a, _hal_a, _haa_a, _force_b, _torque_b, _hessian_b, _hal_b, _haa_b = (
        evaluate_rigid_contact_from_collision(
            0,
            -1,
            body_q,
            body_q_prev,
            body_com,
            wp.vec3(0.0),
            wp.vec3(0.0),
            wp.vec3(0.0),
            wp.vec3(0.0),
            wp.vec3(0.0, 0.0, 1.0),
            0.01,
            100.0,
            1000.0,
            100.0,
            0.0,
            wp.vec3(0.0),
            0.5,
            0.01,
            0,
            1,
            0.01,
            wp.vec3(0.0),
        )
    )
    force_out[0] = force_a
    hessian_out[0] = hessian_a


@wp.kernel
def _eval_directional_joint_projection_kernel(
    linear_force_out: wp.array[wp.vec3],
    angular_torque_out: wp.array[wp.vec3],
):
    a = wp.vec3(1.0, 0.0, 0.0)
    P = wp.identity(3, float) - wp.outer(a, a)
    q_id = wp.quat_identity()
    X_wp = wp.transform(wp.vec3(0.0), q_id)
    X_wc = wp.transform(wp.vec3(4.0, 2.0, 3.0), q_id)
    force, _torque, _Hll, _Hal, _Haa = evaluate_linear_constraint_force_hessian(
        X_wp,
        X_wc,
        X_wp,
        X_wc,
        wp.transform_identity(),
        wp.transform_identity(),
        wp.vec3(0.0),
        wp.vec3(0.0),
        True,
        2.0,
        2.0,
        P,
        wp.vec3(5.0, 7.0, 11.0),
        wp.vec3(0.0),
        0.0,
        0.0,
        0,
        0.01,
    )
    linear_force_out[0] = force

    q_free = wp.quat_from_axis_angle(a, 0.5)
    torque, _Haa_ang, _kappa, _J = evaluate_angular_constraint_force_hessian(
        q_id,
        q_free,
        q_id,
        q_id,
        q_id,
        q_id,
        True,
        2.0,
        2.0,
        P,
        wp.vec3(5.0, 7.0, 11.0),
        wp.vec3(0.0),
        0.0,
        0.0,
        0,
        0.01,
    )
    angular_torque_out[0] = torque


@wp.kernel
def _prepare_body_particle_dual_prefix(
    raw_count: wp.array[int],
    contact_count: wp.array[int],
    penalty_k: wp.array[float],
    initial_penalty: float,
):
    i = wp.tid()
    if i == 0:
        contact_count[0] = raw_count[0]
    penalty_k[i] = initial_penalty


@wp.kernel
def _prepare_particle_contact_gather_replay(
    raw_count: wp.array[int],
    contact_count: wp.array[int],
    particle_contact_head: wp.array[int],
    particle_forces: wp.array[wp.vec3],
    particle_hessians: wp.array[wp.mat33],
):
    i = wp.tid()
    if i == 0:
        contact_count[0] = raw_count[0]
    if i < particle_contact_head.shape[0]:
        particle_contact_head[i] = -1
        particle_forces[i] = wp.vec3(0.0)
        particle_hessians[i] = wp.mat33(0.0)


@wp.kernel
def accumulate_particle_body_contact_force_and_hessian(
    # inputs
    dt: float,
    current_color: int,
    pos_anchor: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    particle_colors: wp.array[int],
    # body-particle contact
    friction_epsilon: float,
    particle_radius: wp.array[float],
    body_particle_contact_indices: wp.array[wp.vec3i],
    body_particle_contact_count: wp.array[int],
    body_particle_contact_max: int,
    # per-contact soft AVBD parameters for body-particle contacts (shared with rigid side)
    body_particle_contact_penalty_k: wp.array[float],
    body_particle_contact_material_ke: wp.array[float],
    body_particle_contact_material_kd: wp.array[float],
    body_particle_contact_material_mu: wp.array[float],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    # Barycentric weights on each record's soft particles; (1, 0, 0) for a particle contact.
    contact_barycentric: wp.array[wp.vec3],
    # outputs: particle force and hessian
    particle_forces: wp.array[wp.vec3],
    particle_hessians: wp.array[wp.mat33],
):
    """Legacy capacity-scan scatter accumulation, kept as the reference oracle for the gather kernel."""
    t_id = wp.tid()

    # One unified soft-contact stream. body_particle_contact_count[0] is the total soft-contact count;
    # each record self-describes via its -1-padded corner ids: (p, -1, -1) is a particle contact,
    # (v0, v1, -1) an edge, (v0, v1, v2) a face. A contact energy E(x) at x = sum_i bary[i]*pos[c_i]
    # contributes bary[i]*force to corner i and bary[i]^2*hessian to its block. VBD solves one color
    # per launch, so only scatter to this record's corners of the active color.
    count = min(body_particle_contact_max, body_particle_contact_count[0])
    if t_id >= count:
        return

    corners = body_particle_contact_indices[t_id]
    # Per-contact AVBD penalty + material properties shared with the rigid side.
    contact_ke = body_particle_contact_penalty_k[t_id]
    contact_kd = body_particle_contact_material_kd[t_id]
    contact_mu = body_particle_contact_material_mu[t_id]

    if corners[1] < 0:
        # Particle contact (p, -1, -1): single-vertex path, unchanged from the pre-unification code.
        particle_idx = corners[0]
        if particle_colors[particle_idx] == current_color:
            body_contact_force, body_contact_hessian = _eval_body_particle_contact(
                particle_idx,
                pos[particle_idx],
                pos_anchor[particle_idx],
                t_id,
                contact_ke,
                contact_kd,
                contact_mu,
                friction_epsilon,
                particle_radius,
                shape_body,
                body_q,
                body_q_prev,
                body_qd,
                body_com,
                contact_shape,
                contact_body_pos,
                contact_body_vel,
                contact_normal,
                shape_margin,
                dt,
            )
            wp.atomic_add(particle_forces, particle_idx, body_contact_force)
            wp.atomic_add(particle_hessians, particle_idx, body_contact_hessian)
    else:
        # Edge/face contact: barycentric point over the record's 2-3 soft particles.
        bary = contact_barycentric[t_id]
        ef_force, ef_hessian, _cp_world = _eval_soft_ef_contact(
            t_id,
            corners,
            bary,
            pos,
            pos_anchor,
            particle_radius,
            contact_ke,
            contact_kd,
            contact_mu,
            friction_epsilon,
            shape_body,
            body_q,
            body_q_prev,
            body_qd,
            body_com,
            contact_shape,
            contact_body_pos,
            contact_body_vel,
            contact_normal,
            shape_margin,
            dt,
        )
        for i in range(3):
            ci = corners[i]
            if ci >= 0:
                w = bary[i]
                if particle_colors[ci] == current_color:
                    wp.atomic_add(particle_forces, ci, w * ef_force)
                    wp.atomic_add(particle_hessians, ci, (w * w) * ef_hessian)


@wp.kernel
def _eval_body_particle_contact_damping_kernel(
    particle_radius: wp.array[float],
    shape_material_mu: wp.array[float],
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    contact_shape: wp.array[wp.int32],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    forces: wp.array[wp.vec3],
):
    i = wp.tid()
    ke = wp.where(i < 2, 400.0, 100.0)
    kd = wp.where((i & 1) == 0, 20.0, 0.0)
    force, _hessian = evaluate_body_particle_contact(
        0,
        wp.vec3(0.0, 0.0, 0.04),
        wp.vec3(0.0, 0.0, 0.05),
        0,
        ke,
        kd,
        0.0,
        0.01,
        particle_radius,
        shape_material_mu,
        shape_body,
        body_q,
        body_q_prev,
        body_qd,
        body_com,
        contact_shape,
        contact_body_pos,
        contact_body_vel,
        contact_normal,
        shape_margin,
        0.1,
    )
    forces[i] = force


@wp.kernel
def _eval_vertex_triangle_uniform_motion_kernel(
    pos: wp.array[wp.vec3],
    pos_prev: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    forces: wp.array[wp.vec3],
    hessians: wp.array[wp.mat33],
):
    i = wp.tid()
    kd = wp.where(i == 1, 50.0, 0.0)
    (
        _has_contact,
        _force_0,
        _force_1,
        _force_2,
        force_3,
        _hessian_0,
        _hessian_1,
        _hessian_2,
        hessian_3,
    ) = evaluate_vertex_triangle_collision_force_hessian_4_vertices(
        3,
        0,
        pos,
        pos_prev,
        tri_indices,
        0.1,
        100.0,
        kd,
        0.0,
        0.01,
        0.1,
    )
    forces[i] = force_3
    hessians[i] = hessian_3


@wp.kernel
def _eval_spring_damping_kernel(
    pos: wp.array[wp.vec3],
    pos_anchor: wp.array[wp.vec3],
    spring_indices: wp.array[int],
    spring_rest_length: wp.array[float],
    spring_stiffness: wp.array[float],
    spring_damping: wp.array[float],
    force: wp.array[wp.vec3],
    hessian: wp.array[wp.mat33],
):
    spring_force, spring_hessian = evaluate_spring_force_and_hessian(
        0,
        0,
        0.1,
        pos,
        pos_anchor,
        spring_indices,
        spring_rest_length,
        spring_stiffness,
        spring_damping,
    )
    force[0] = spring_force
    hessian[0] = spring_hessian


@wp.kernel
def _eval_bending_degenerate_anchor_kernel(
    pos: wp.array[wp.vec3],
    pos_anchor: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    edge_rest_angle: wp.array[float],
    edge_rest_length: wp.array[float],
    force_norms: wp.array[float],
):
    v_order = wp.tid()
    force, hessian = evaluate_dihedral_angle_based_bending_force_hessian(
        0,
        v_order,
        pos,
        pos_anchor,
        edge_indices,
        edge_rest_angle,
        edge_rest_length,
        0.0,
        20.0,
        0.1,
    )
    force_norms[v_order] = wp.length(force) + wp.length(hessian[0]) + wp.length(hessian[1]) + wp.length(hessian[2])


@wp.kernel
def _eval_elastic_damping_rigid_motion_kernel(
    pos: wp.array[wp.vec3],
    pos_anchor: wp.array[wp.vec3],
    spring_indices: wp.array[int],
    spring_rest_length: wp.array[float],
    spring_stiffness: wp.array[float],
    spring_damping: wp.array[float],
    tri_indices: wp.array2d[wp.int32],
    tri_poses: wp.array[wp.mat22],
    tri_areas: wp.array[float],
    edge_indices: wp.array2d[wp.int32],
    edge_rest_angle: wp.array[float],
    edge_rest_length: wp.array[float],
    tet_indices: wp.array2d[wp.int32],
    tet_poses: wp.array[wp.mat33],
    force_norms: wp.array2d[float],
):
    sample = wp.tid()
    dt = 0.1
    damping = 20.0

    _v0, _v1, spring_force_0, spring_force_1, _spring_hessian = evaluate_spring_force_and_hessian_both_vertices(
        sample,
        dt,
        pos,
        pos_anchor,
        spring_indices,
        spring_rest_length,
        spring_stiffness,
        spring_damping,
    )
    force_norms[sample, 0] = wp.max(wp.length(spring_force_0), wp.length(spring_force_1))

    tri_max = float(0.0)
    for v_order in range(3):
        tri_force, _tri_hessian = evaluate_neo_hookean_membrane_force_hessian(
            sample,
            v_order,
            pos,
            pos_anchor,
            tri_indices,
            tri_poses[sample],
            tri_areas[sample],
            0.0,
            1.0,
            damping,
            dt,
        )
        tri_max = wp.max(tri_max, wp.length(tri_force))
    force_norms[sample, 1] = tri_max

    bend_max = float(0.0)
    for v_order in range(4):
        bend_force, _bend_hessian = evaluate_dihedral_angle_based_bending_force_hessian(
            sample,
            v_order,
            pos,
            pos_anchor,
            edge_indices,
            edge_rest_angle,
            edge_rest_length,
            0.0,
            damping,
            dt,
        )
        bend_max = wp.max(bend_max, wp.length(bend_force))
    force_norms[sample, 2] = bend_max

    tet_max = float(0.0)
    for v_order in range(4):
        tet_force, _tet_hessian = evaluate_volumetric_neo_hookean_force_and_hessian(
            sample,
            v_order,
            pos_anchor,
            pos,
            tet_indices,
            tet_poses[sample],
            0.0,
            1.0,
            damping,
            dt,
        )
        tet_max = wp.max(tet_max, wp.length(tet_force))
    force_norms[sample, 3] = tet_max


@wp.kernel
def _eval_body_particle_contact_rigid_motion_kernel(
    particle_q: wp.array[wp.vec3],
    particle_q_prev: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    shape_material_mu: wp.array[float],
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    contact_shape: wp.array[wp.int32],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    damping_delta_norms: wp.array[float],
):
    sample = wp.tid()
    dt = 0.1

    body_particle_force_damped, _body_particle_hessian_damped = evaluate_body_particle_contact(
        sample,
        particle_q[sample],
        particle_q_prev[sample],
        sample,
        100.0,
        20.0,
        0.0,
        0.01,
        particle_radius,
        shape_material_mu,
        shape_body,
        body_q,
        body_q_prev,
        body_qd,
        body_com,
        contact_shape,
        contact_body_pos,
        contact_body_vel,
        contact_normal,
        shape_margin,
        dt,
    )
    body_particle_force_undamped, _body_particle_hessian_undamped = evaluate_body_particle_contact(
        sample,
        particle_q[sample],
        particle_q_prev[sample],
        sample,
        100.0,
        0.0,
        0.0,
        0.01,
        particle_radius,
        shape_material_mu,
        shape_body,
        body_q,
        body_q_prev,
        body_qd,
        body_com,
        contact_shape,
        contact_body_pos,
        contact_body_vel,
        contact_normal,
        shape_margin,
        dt,
    )
    damping_delta_norms[sample] = wp.length(body_particle_force_damped - body_particle_force_undamped)


@wp.kernel
def _eval_rigid_contact_rigid_motion_kernel(
    contact_normal: wp.array[wp.vec3],
    rigid_body_q: wp.array[wp.transform],
    rigid_body_q_prev: wp.array[wp.transform],
    rigid_body_com: wp.array[wp.vec3],
    damping_delta_norms: wp.array[float],
):
    sample = wp.tid()
    dt = 0.1
    rigid_body_a = 2 * sample
    rigid_body_b = rigid_body_a + 1
    (
        force_a_damped,
        torque_a_damped,
        _h_ll_a_damped,
        _h_al_a_damped,
        _h_aa_a_damped,
        force_b_damped,
        torque_b_damped,
        _h_ll_b_damped,
        _h_al_b_damped,
        _h_aa_b_damped,
    ) = evaluate_rigid_contact_from_collision(
        rigid_body_a,
        rigid_body_b,
        rigid_body_q,
        rigid_body_q_prev,
        rigid_body_com,
        wp.vec3(0.2, -0.1, 0.05),
        wp.vec3(0.2, -0.1, 0.05),
        wp.vec3(0.0),
        wp.vec3(0.0),
        contact_normal[sample],
        0.06,
        100.0,
        100.0,
        100.0,
        20.0,
        wp.vec3(0.0),
        0.0,
        0.01,
        0,
        0,
        dt,
        wp.vec3(0.0),
    )
    (
        force_a_undamped,
        torque_a_undamped,
        _h_ll_a_undamped,
        _h_al_a_undamped,
        _h_aa_a_undamped,
        force_b_undamped,
        torque_b_undamped,
        _h_ll_b_undamped,
        _h_al_b_undamped,
        _h_aa_b_undamped,
    ) = evaluate_rigid_contact_from_collision(
        rigid_body_a,
        rigid_body_b,
        rigid_body_q,
        rigid_body_q_prev,
        rigid_body_com,
        wp.vec3(0.2, -0.1, 0.05),
        wp.vec3(0.2, -0.1, 0.05),
        wp.vec3(0.0),
        wp.vec3(0.0),
        contact_normal[sample],
        0.06,
        100.0,
        100.0,
        100.0,
        0.0,
        wp.vec3(0.0),
        0.0,
        0.01,
        0,
        0,
        dt,
        wp.vec3(0.0),
    )
    rigid_delta = wp.max(wp.length(force_a_damped - force_a_undamped), wp.length(force_b_damped - force_b_undamped))
    rigid_delta = wp.max(rigid_delta, wp.length(torque_a_damped - torque_a_undamped))
    rigid_delta = wp.max(rigid_delta, wp.length(torque_b_damped - torque_b_undamped))
    damping_delta_norms[sample] = rigid_delta


@wp.kernel
def _eval_vertex_triangle_contact_rigid_motion_kernel(
    soft_pos: wp.array[wp.vec3],
    soft_pos_anchor: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    damping_delta_norms: wp.array[float],
):
    sample = wp.tid()
    dt = 0.1
    vertex = 4 * sample + 3
    (
        _has_contact_damped,
        force_0_damped,
        force_1_damped,
        force_2_damped,
        force_3_damped,
        _hessian_0_damped,
        _hessian_1_damped,
        _hessian_2_damped,
        _hessian_3_damped,
    ) = evaluate_vertex_triangle_collision_force_hessian_4_vertices(
        vertex,
        sample,
        soft_pos,
        soft_pos_anchor,
        tri_indices,
        0.1,
        100.0,
        20.0,
        0.0,
        0.01,
        dt,
    )
    (
        _has_contact_undamped,
        force_0_undamped,
        force_1_undamped,
        force_2_undamped,
        force_3_undamped,
        _hessian_0_undamped,
        _hessian_1_undamped,
        _hessian_2_undamped,
        _hessian_3_undamped,
    ) = evaluate_vertex_triangle_collision_force_hessian_4_vertices(
        vertex,
        sample,
        soft_pos,
        soft_pos_anchor,
        tri_indices,
        0.1,
        100.0,
        0.0,
        0.0,
        0.01,
        dt,
    )
    soft_delta = wp.max(wp.length(force_0_damped - force_0_undamped), wp.length(force_1_damped - force_1_undamped))
    soft_delta = wp.max(soft_delta, wp.length(force_2_damped - force_2_undamped))
    soft_delta = wp.max(soft_delta, wp.length(force_3_damped - force_3_undamped))
    damping_delta_norms[sample] = soft_delta


def test_self_contact_barrier_c2_at_tau(test, device):
    """Barrier must be C2-continuous at d = tau (= collision_radius / 2).

    The log-barrier region (d_min < d < tau) and the outer linear-penalty
    region (tau <= d < collision_radius) share the boundary d = tau.  For
    C2 continuity both the first derivative (force) and the second
    derivative (Hessian scalar) must agree there.

    Regression for GitHub issue #2154.
    """
    collision_radius = 0.02
    k = 1.0e3
    tau = collision_radius * 0.5
    eps = tau * 1e-5

    distances = wp.array([tau - eps, tau + eps], dtype=float, device=device)
    dEdD_out = wp.zeros(2, dtype=float, device=device)
    d2E_out = wp.zeros(2, dtype=float, device=device)

    wp.launch(
        _eval_self_contact_norm_kernel,
        dim=2,
        inputs=[distances, collision_radius, k, dEdD_out, d2E_out],
        device=device,
    )

    dEdD = dEdD_out.numpy()
    d2E = d2E_out.numpy()

    np.testing.assert_allclose(
        dEdD[0],
        dEdD[1],
        rtol=1e-3,
        err_msg="Self-contact barrier force is not C1-continuous at d = tau",
    )
    np.testing.assert_allclose(
        d2E[0],
        d2E[1],
        rtol=1e-3,
        err_msg="Self-contact barrier Hessian is not C2-continuous at d = tau",
    )


def test_self_contact_barrier_c2_at_d_min(test, device):
    """Barrier must be C2-continuous at d = d_min (= 1e-5).

    The quadratic-extension region (d <= d_min) and the log-barrier region
    (d_min < d < tau) share the boundary d = d_min.  The quadratic is a
    Taylor expansion of the log-barrier at d_min, so both the first and
    second derivatives must match.
    """
    collision_radius = 0.02
    k = 1.0e3
    d_min = 1.0e-5
    eps = d_min * 1e-5

    distances = wp.array([d_min - eps, d_min + eps], dtype=float, device=device)
    dEdD_out = wp.zeros(2, dtype=float, device=device)
    d2E_out = wp.zeros(2, dtype=float, device=device)

    wp.launch(
        _eval_self_contact_norm_kernel,
        dim=2,
        inputs=[distances, collision_radius, k, dEdD_out, d2E_out],
        device=device,
    )

    dEdD = dEdD_out.numpy()
    d2E = d2E_out.numpy()

    np.testing.assert_allclose(
        dEdD[0],
        dEdD[1],
        rtol=1e-3,
        err_msg="Self-contact barrier force is not C1-continuous at d = d_min",
    )
    np.testing.assert_allclose(
        d2E[0],
        d2E[1],
        rtol=1e-3,
        err_msg="Self-contact barrier Hessian is not C2-continuous at d = d_min",
    )


def test_rigid_soft_contact_log_barrier_matches_particle(test, device):
    """The optional rigid-soft law must match self-contact; disabled mode stays quadratic."""
    collision_radius = 0.02
    k = 1.0e3
    distances_np = np.array([0.75, 0.25, 0.0, -0.5], dtype=np.float32) * collision_radius
    distances = wp.array(distances_np, dtype=float, device=device)

    particle_dEdD = wp.zeros(len(distances_np), dtype=float, device=device)
    particle_d2E = wp.zeros(len(distances_np), dtype=float, device=device)
    rigid_dEdD = wp.zeros(len(distances_np), dtype=float, device=device)
    rigid_d2E = wp.zeros(len(distances_np), dtype=float, device=device)
    penalty_dEdD = wp.zeros(len(distances_np), dtype=float, device=device)
    penalty_d2E = wp.zeros(len(distances_np), dtype=float, device=device)

    wp.launch(
        _eval_self_contact_norm_kernel,
        dim=len(distances_np),
        inputs=[distances, collision_radius, k, particle_dEdD, particle_d2E],
        device=device,
    )
    wp.launch(
        _eval_rigid_soft_contact_norm_kernel,
        dim=len(distances_np),
        inputs=[distances, collision_radius, k, True, rigid_dEdD, rigid_d2E],
        device=device,
    )
    wp.launch(
        _eval_rigid_soft_contact_norm_kernel,
        dim=len(distances_np),
        inputs=[distances, collision_radius, k, False, penalty_dEdD, penalty_d2E],
        device=device,
    )

    np.testing.assert_allclose(rigid_dEdD.numpy(), particle_dEdD.numpy(), rtol=1.0e-6)
    np.testing.assert_allclose(rigid_d2E.numpy(), particle_d2E.numpy(), rtol=1.0e-6)
    np.testing.assert_allclose(penalty_dEdD.numpy(), -k * (collision_radius - distances_np), rtol=1.0e-6)
    np.testing.assert_allclose(penalty_d2E.numpy(), k, rtol=1.0e-6)
    test.assertGreater(-rigid_dEdD.numpy()[1], -penalty_dEdD.numpy()[1])


def _rigid_joint_angular_rho_seed_uses_mean_mobility(test, device):
    """Verify the isotropic angular seed averages inverse inertia, not forward inertia."""
    del test
    with wp.ScopedDevice(device):
        body_inv_mass = wp.array([1.0, 1.0], dtype=float, device=device)
        body_inv_inertia = wp.array(
            [np.diag([1.0, 4.0, 9.0]), np.diag([2.0, 3.0, 6.0])],
            dtype=wp.mat33,
            device=device,
        )
        rho = wp.empty(1, dtype=float, device=device)

        wp.launch(
            _eval_joint_angular_rho_seed_kernel,
            dim=1,
            inputs=[body_inv_mass, body_inv_inertia, 100.0],
            outputs=[rho],
            device=device,
        )

        # Mean angular mobility is (1+4+9+2+3+6)/3 = 25/3; rho = inv_dt^2 / mobility = 12.
        np.testing.assert_allclose(rho.numpy(), [12.0], rtol=1.0e-6, atol=1.0e-6)


def _rigid_contact_tangent_support_uses_pair_mobility_eigenvalue(test, device):
    """Verify tangent support preserves endpoint directions until after pair assembly."""
    del test
    with wp.ScopedDevice(device):
        shape_body = wp.array([0, 1], dtype=wp.int32, device=device)
        body_q = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
        body_com = wp.zeros(2, dtype=wp.vec3, device=device)
        body_inv_mass = wp.ones(2, dtype=float, device=device)
        body_inv_inertia = wp.array(
            [np.diag([0.0, 0.0, 9.0]), np.diag([0.0, 0.0, 9.0])],
            dtype=wp.mat33,
            device=device,
        )
        support = wp.empty(1, dtype=float, device=device)

        wp.launch(
            _eval_crossed_contact_tangent_pair_support_kernel,
            dim=1,
            inputs=[shape_body, body_q, body_com, body_inv_mass, body_inv_inertia],
            outputs=[support],
            device=device,
        )

        # diag(1, 10) + diag(10, 1) = 11*I, so reduce after summing.
        np.testing.assert_allclose(support.numpy(), [1.0 / 11.0], rtol=1.0e-6)


def _rigid_compliant_sliding_contact_has_solve_metric(test, device):
    """Verify saturated Coulomb ALM friction keeps projected force and a PSD slip metric."""
    del test
    with wp.ScopedDevice(device):
        body_q_prev = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        body_q = wp.array(
            [wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity())],
            dtype=wp.transform,
            device=device,
        )
        body_com = wp.zeros(1, dtype=wp.vec3, device=device)
        force = wp.empty(1, dtype=wp.vec3, device=device)
        hessian = wp.empty(1, dtype=wp.mat33, device=device)

        wp.launch(
            _eval_compliant_sliding_contact_metric_kernel,
            dim=1,
            inputs=[body_q, body_q_prev, body_com],
            outputs=[force, hessian],
            device=device,
        )

        force_np = force.numpy()[0]
        hessian_np = hessian.numpy()[0]
        np.testing.assert_allclose(abs(force_np[0]), 0.5 * abs(force_np[2]), rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(hessian_np, hessian_np.T, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_array_less(-1.0e-6, np.linalg.eigvalsh(hessian_np))
        np.testing.assert_array_less(0.0, hessian_np[0, 0])


def _assert_rigid_compliant_alm_coefficients(device):
    pairs = np.asarray(
        [(1.0e6, 9.0e6), (1.0e3, 1.0e5), (1.0e5, 1.0e3), (10.0, 10.0), (0.0, 1.0), (1.0, 0.0)],
        dtype=np.float32,
    )
    positive_count = 4

    with wp.ScopedDevice(device):
        material_k = wp.array(pairs[:, 0], dtype=float, device=device)
        rho = wp.array(pairs[:, 1], dtype=float, device=device)
        result = wp.empty(len(pairs), dtype=wp.vec4, device=device)
        wp.launch(
            _eval_compliant_alm_coefficients_kernel,
            dim=len(pairs),
            inputs=[material_k, rho],
            outputs=[result],
            device=device,
        )
        actual = result.numpy()

    material_k_ref = pairs[:positive_count, 0].astype(np.float64)
    rho_ref = pairs[:positive_count, 1].astype(np.float64)
    denominator = material_k_ref + rho_ref
    s_ref = material_k_ref / denominator
    a_ref = rho_ref / denominator
    expected = np.column_stack(
        (
            s_ref,
            material_k_ref * a_ref,
            a_ref,
            s_ref * (2.0 + rho_ref * 0.25),
        )
    )

    np.testing.assert_allclose(actual[:positive_count], expected, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_array_equal(actual[positive_count:], [[0.0, 0.0, 1.0, 0.0]] * 2)


def _rigid_contact_structural_support_conditions_tangent_rho(test, device):
    """Verify step setup caps tangent rho with structural support, not D+S."""
    with wp.ScopedDevice(device):
        # One dynamic body vs world. At the COM, D = inv_dt^2 / inv_mass = 100.
        # Normal rho adds structural: rho_n = D+S = 250. Tangent inertial A_t = D
        # only, so the policy gives rho_t = max(A_t, min(rho_n, S)) = S = 150.
        normal_rho = wp.zeros(1, dtype=float, device=device)
        tangent_rho = wp.zeros(1, dtype=float, device=device)

        wp.launch(
            step_body_body_contact_C0_lambda,
            dim=1,
            inputs=[
                wp.array([1], dtype=int, device=device),
                wp.array([0], dtype=int, device=device),
                wp.array([1], dtype=int, device=device),
                wp.zeros(1, dtype=wp.vec3, device=device),
                wp.zeros(1, dtype=wp.vec3, device=device),
                wp.zeros(1, dtype=wp.vec3, device=device),
                wp.zeros(1, dtype=wp.vec3, device=device),
                wp.array([wp.vec3(0.0, 0.0, 1.0)], dtype=wp.vec3, device=device),
                wp.zeros(1, dtype=float, device=device),
                wp.zeros(1, dtype=float, device=device),
                wp.array([-1, 0], dtype=wp.int32, device=device),
                wp.zeros(1, dtype=wp.int32, device=device),
                wp.array([1.0], dtype=float, device=device),
                wp.zeros(1, dtype=wp.mat33, device=device),
                wp.zeros(1, dtype=wp.vec3, device=device),
                wp.array([150.0], dtype=float, device=device),
                int(newton.BodyFlags.PROXY),
                wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
                0,
                1,
                1.0,
                100.0,
                1.0,
                wp.array([1.0e9], dtype=float, device=device),
                -1.0,
            ],
            outputs=[
                normal_rho,
                wp.zeros(1, dtype=float, device=device),
                wp.zeros(1, dtype=wp.vec3, device=device),
                wp.zeros(1, dtype=wp.vec3, device=device),
                tangent_rho,
            ],
            device=device,
        )

        # A leak of S into the tangent plane metric would lift rho_t to rho_n.
        test.assertLess(tangent_rho.numpy()[0], normal_rho.numpy()[0])
        np.testing.assert_allclose(normal_rho.numpy(), [250.0], rtol=1.0e-6)
        np.testing.assert_allclose(tangent_rho.numpy(), [150.0], rtol=1.0e-6)


def _rigid_contact_history_restore_from_match_index(test, device):
    """Verify legacy hard contact preserves its full-vector warm start."""
    with wp.ScopedDevice(device):
        contact_count = wp.array([4], dtype=int, device=device)
        shape0 = wp.array([0, 0, 0, 0], dtype=int, device=device)
        shape1 = wp.array([1, 1, 1, 1], dtype=int, device=device)
        normal = wp.array([[0.0, 0.0, 1.0]] * 4, dtype=wp.vec3, device=device)

        shape_ke = wp.array([100.0, 200.0], dtype=float, device=device)
        shape_kd = wp.array([1.0, 3.0], dtype=float, device=device)
        shape_mu = wp.array([0.25, 1.0], dtype=float, device=device)
        match_index = wp.array([2, -1, 0, -2], dtype=wp.int32, device=device)

        history = RigidContactHistory()
        history.lambda_ = wp.array([[0.5, 0.0, 1.0], [4.0, 5.0, 6.0], [0.0, 0.0, 7.0]], dtype=wp.vec3, device=device)
        history.penalty_k = wp.array([20.0, 30.0, 40.0], dtype=float, device=device)
        history.normal = wp.array([[0.0, 0.0, 1.0]] * 3, dtype=wp.vec3, device=device)

        penalty_k = wp.zeros(4, dtype=float, device=device)
        lam = wp.zeros(4, dtype=wp.vec3, device=device)
        material_kd = wp.zeros(4, dtype=float, device=device)
        material_mu = wp.zeros(4, dtype=float, device=device)
        material_ke = wp.zeros(4, dtype=float, device=device)

        wp.launch(
            init_body_body_contacts_alm,
            dim=4,
            inputs=[
                contact_count,
                shape0,
                shape1,
                normal,
                shape_ke,
                shape_kd,
                shape_mu,
                1,
                0,
                0,
                match_index,
                history,
                None,
                None,
                None,
                None,
                None,
                10.0,
            ],
            outputs=[
                penalty_k,
                lam,
                material_kd,
                material_mu,
                material_ke,
            ],
            device=device,
        )

        np.testing.assert_allclose(penalty_k.numpy(), [40.0, 10.0, 20.0, 10.0])
        np.testing.assert_allclose(lam.numpy(), [[0.0, 0.0, 7.0], [0.0, 0.0, 0.0], [0.5, 0.0, 1.0], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(material_ke.numpy(), [150.0] * 4)
        np.testing.assert_allclose(material_kd.numpy(), [2.0] * 4)
        np.testing.assert_allclose(material_mu.numpy(), [0.5] * 4)


def _rigid_contact_history_compliant_alm_tangent_warmstart(test, device):
    """Verify ALM sticky zeros tangent warm-start and latest cone-clips it."""
    del test
    cases = (
        # sticky: restore penalty and lambda_n only; zero lambda_t
        ("sticky", [[1.0, 2.0, 3.0]], [0.25, 1.0], 0, [[0.0, 0.0, 3.0]]),
        # latest: hist lambda_t=(3,4) length 5; mu=0.5, lambda_n=5 -> cone 2.5 -> (1.5, 2)
        ("latest", [[3.0, 4.0, 5.0]], [0.5, 0.5], 1, [[1.5, 2.0, 5.0]]),
    )
    with wp.ScopedDevice(device):
        for _name, hist_lambda, shape_mu, latest, expected_lambda in cases:
            contact_count = wp.array([1], dtype=int, device=device)
            shape0 = wp.array([0], dtype=int, device=device)
            shape1 = wp.array([1], dtype=int, device=device)
            normal = wp.array([wp.vec3(0.0, 0.0, 1.0)], dtype=wp.vec3, device=device)
            match_index = wp.array([0], dtype=wp.int32, device=device)

            history = RigidContactHistory()
            history.lambda_ = wp.array(hist_lambda, dtype=wp.vec3, device=device)
            history.penalty_k = wp.array([20.0], dtype=float, device=device)
            history.normal = wp.array([wp.vec3(0.0, 0.0, 1.0)], dtype=wp.vec3, device=device)

            penalty_k = wp.zeros(1, dtype=float, device=device)
            contact_lambda = wp.zeros(1, dtype=wp.vec3, device=device)
            material_kd = wp.zeros(1, dtype=float, device=device)
            material_mu = wp.zeros(1, dtype=float, device=device)
            material_ke = wp.zeros(1, dtype=float, device=device)

            wp.launch(
                init_body_body_contacts_alm,
                dim=1,
                inputs=[
                    contact_count,
                    shape0,
                    shape1,
                    normal,
                    wp.array([100.0, 200.0], dtype=float, device=device),
                    wp.array([1.0, 3.0], dtype=float, device=device),
                    wp.array(shape_mu, dtype=float, device=device),
                    0,
                    1,
                    latest,
                    match_index,
                    history,
                    None,
                    None,
                    None,
                    None,
                    None,
                    10.0,
                ],
                outputs=[
                    penalty_k,
                    contact_lambda,
                    material_kd,
                    material_mu,
                    material_ke,
                ],
                device=device,
            )
            np.testing.assert_allclose(penalty_k.numpy(), [20.0])
            np.testing.assert_allclose(contact_lambda.numpy(), expected_lambda)


def _rigid_contact_history_soft_restores_penalty_only(test, device):
    """Verify legacy soft contacts restore penalty only and never lambda."""
    with wp.ScopedDevice(device):
        contact_count = wp.array([1], dtype=int, device=device)
        shape0 = wp.array([0], dtype=int, device=device)
        shape1 = wp.array([1], dtype=int, device=device)
        normal = wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3, device=device)

        history = RigidContactHistory()
        history.lambda_ = wp.array([[1.0, 2.0, 3.0]], dtype=wp.vec3, device=device)
        history.penalty_k = wp.array([40.0], dtype=float, device=device)
        history.normal = wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3, device=device)

        penalty_k = wp.zeros(1, dtype=float, device=device)
        lam = wp.zeros(1, dtype=wp.vec3, device=device)
        material_kd = wp.zeros(1, dtype=float, device=device)
        material_mu = wp.zeros(1, dtype=float, device=device)
        material_ke = wp.zeros(1, dtype=float, device=device)

        wp.launch(
            init_body_body_contacts_alm,
            dim=1,
            inputs=[
                contact_count,
                shape0,
                shape1,
                normal,
                wp.array([100.0, 200.0], dtype=float, device=device),
                wp.array([1.0, 3.0], dtype=float, device=device),
                wp.array([0.25, 1.0], dtype=float, device=device),
                0,
                0,
                0,
                wp.array([0], dtype=wp.int32, device=device),
                history,
                None,
                None,
                None,
                None,
                None,
                10.0,
            ],
            outputs=[
                penalty_k,
                lam,
                material_kd,
                material_mu,
                material_ke,
            ],
            device=device,
        )

        np.testing.assert_allclose(penalty_k.numpy(), [40.0])
        np.testing.assert_allclose(lam.numpy(), [[0.0, 0.0, 0.0]])


def _rigid_contact_history_capture_requires_preallocation(test, device):
    """Contact history must be allocated before CUDA graph recording."""

    def make_scene(pipeline_first, rigid_contact_max=4):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.2, hy=0.2, hz=0.2)
        builder.color()
        model = builder.finalize(device=device)

        pipeline = contacts = None
        if pipeline_first:
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=rigid_contact_max, contact_matching="latest")
            contacts = pipeline.contacts()

        solver = newton.solvers.SolverVBD(
            model,
            iterations=1,
            rigid_contact_history=True,
            rigid_compliant_alm=True,
        )

        if not pipeline_first:
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=rigid_contact_max, contact_matching="latest")
            contacts = pipeline.contacts()

        state_in = model.state()
        state_out = model.state()
        control = model.control()
        if rigid_contact_max > 0:
            pipeline.collide(state_in, contacts)
        return pipeline, solver, contacts, state_in, state_out, control

    pipeline, solver, contacts, state_in, state_out, control = make_scene(pipeline_first=False)
    with test.assertRaisesRegex(RuntimeError, "contact history must be allocated before graph capture"):
        with wp.ScopedCapture(device=device):
            solver.step(state_in, state_out, control, contacts, 1.0e-3)

    pipeline, solver, contacts, state_in, state_out, control = make_scene(pipeline_first=True)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(state_in, state_out, control, contacts, 1.0e-3)
    test.assertIsNotNone(capture.graph)

    pipeline, solver, contacts, state_in, state_out, control = make_scene(pipeline_first=True, rigid_contact_max=0)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(state_in, state_out, control, contacts, 1.0e-3)
    test.assertIsNotNone(capture.graph)
    test.assertIsNone(solver._prev_contact_lambda)

    pipeline, solver, contacts, state_in, state_out, control = make_scene(pipeline_first=False)
    solver.step(state_in, state_out, control, contacts, 1.0e-3)
    pipeline.collide(state_out, contacts)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(state_out, state_in, control, contacts, 1.0e-3)
    test.assertIsNotNone(capture.graph)


def _rigid_contact_stick_eps_are_deprecated(test, device):
    """Verify deprecated stick options warn and are ignored."""
    builder = newton.ModelBuilder()
    model = builder.finalize(device=device)

    with test.assertWarnsRegex(DeprecationWarning, "deprecated and ignored") as warning:
        solver = newton.solvers.SolverVBD(
            model,
            rigid_compliant_alm=True,
            rigid_contact_stick_motion_eps=1.0e-4,
            rigid_contact_stick_freeze_translation_eps=1.0e-5,
            rigid_contact_stick_freeze_angular_eps=1.0e-5,
        )

    test.assertEqual(warning.filename, __file__)

    # "and ignored": the solver retains no state derived from the deprecated epsilons.
    test.assertFalse(hasattr(solver, "rigid_contact_stick_motion_eps"))
    test.assertFalse(hasattr(solver, "rigid_contact_stick_freeze_translation_eps"))
    test.assertFalse(hasattr(solver, "rigid_contact_stick_freeze_angular_eps"))


def _rigid_compliant_alm_omission_warns_at_caller(test, device):
    """Verify the migration warning identifies the SolverVBD call site."""
    builder = newton.ModelBuilder()
    builder.add_body()
    builder.color()
    model = builder.finalize(device=device)

    with test.assertWarnsRegex(DeprecationWarning, "Omitting rigid_compliant_alm") as warning:
        newton.solvers.SolverVBD(model)
    test.assertEqual(warning.filename, __file__)


def _rigid_contact_dual_update_computes_lambda(test, device):
    """Verify finite-material coefficients and projected contact dual updates."""
    _assert_rigid_compliant_alm_coefficients(device)
    del test
    with wp.ScopedDevice(device):
        # Two contacts, cold-start lambda=0, K=rho_n=rho_t=10, mu=0.5, C_n=0.1
        # from coincident points + 0.05 margins. Contact 0 has unit tangential
        # slip (saturates the cone); contact 1 has none.
        contact_count = wp.array([2], dtype=int, device=device)
        shape0 = wp.array([0, 0], dtype=int, device=device)
        shape1 = wp.array([1, 2], dtype=int, device=device)
        zeros3 = wp.zeros(2, dtype=wp.vec3, device=device)
        normal = wp.array([wp.vec3(0.0, 0.0, 1.0)] * 2, dtype=wp.vec3, device=device)
        margin = wp.array([0.05, 0.05], dtype=float, device=device)
        shape_body = wp.array([0, 1, 2], dtype=int, device=device)

        q = wp.quat_identity()
        body_q = wp.array(
            [
                wp.transform(wp.vec3(0.0, 0.0, 0.0), q),
                wp.transform(wp.vec3(1.0, 0.0, 0.0), q),
                wp.transform(wp.vec3(0.0, 0.0, 0.0), q),
            ],
            dtype=wp.transform,
            device=device,
        )
        body_q_prev = wp.array([wp.transform_identity()] * 3, dtype=wp.transform, device=device)
        contact_mu = wp.array([0.5, 0.5], dtype=float, device=device)
        contact_ke = wp.array([10.0, 10.0], dtype=float, device=device)
        contact_rho = wp.array([10.0, 10.0], dtype=float, device=device)
        penalty_k = wp.array([10.0, 10.0], dtype=float, device=device)
        contact_lambda = wp.zeros(2, dtype=wp.vec3, device=device)

        wp.launch(
            update_duals_body_body_contacts,
            dim=2,
            inputs=[
                contact_count,
                shape0,
                shape1,
                zeros3,
                zeros3,
                zeros3,
                zeros3,
                normal,
                margin,
                margin,
                shape_body,
                body_q,
                body_q_prev,
                contact_mu,
                zeros3,
                0.0,
                0,
                1,
                contact_ke,
                contact_rho,
                contact_rho,
                0.0,
                penalty_k,
                contact_lambda,
            ],
            device=device,
        )

        # With K=rho: s=0.5, k_eff=5 -> lambda_n = k_eff*C_n = 0.5,
        # and the Coulomb limit is mu*lambda_n = 0.25.
        np.testing.assert_allclose(
            contact_lambda.numpy(),
            [
                [-0.25, 0.0, 0.5],
                [0.0, 0.0, 0.5],
            ],
            rtol=1.0e-6,
            atol=1.0e-6,
        )


def _rigid_contact_reset_ownership(test, device):
    """Contact invalidation covers both endpoints and survives nonidentity slots."""
    with wp.ScopedDevice(device):
        # Row 0 owns world 0 through endpoint-0's attached body (its shape is
        # global); row 1 owns world 0 through endpoint-1's direct shape world;
        # row 2 owns unselected world 1. match_index is a nonidentity permutation.
        shape_world = wp.array([-1, -1, 1, 0], dtype=wp.int32, device=device)
        shape_body = wp.array([0, -1, 1, -1], dtype=wp.int32, device=device)
        body_world = wp.array([0, 1], dtype=wp.int32, device=device)
        shape0 = wp.array([0, 1, 2], dtype=int, device=device)
        shape1 = wp.array([1, 3, 1], dtype=int, device=device)
        match_index = wp.array([2, 0, 1], dtype=wp.int32, device=device)
        reset_pending = wp.ones(1, dtype=wp.int32, device=device)
        reset_mask = wp.array([True, False, False], dtype=wp.bool, device=device)

        contact_count = wp.array([3], dtype=int, device=device)
        # Equal current/saved normals make a warm restore reproduce the saved dual exactly.
        normal = wp.array([[0.0, 0.0, 1.0]] * 3, dtype=wp.vec3, device=device)

        history = RigidContactHistory()
        history.lambda_ = wp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=wp.vec3, device=device)
        history.penalty_k = wp.array([40.0, 50.0, 60.0], dtype=float, device=device)
        history.normal = wp.array([[0.0, 0.0, 1.0]] * 3, dtype=wp.vec3, device=device)

        penalty_k = wp.zeros(3, dtype=float, device=device)
        contact_lambda = wp.zeros(3, dtype=wp.vec3, device=device)
        material_kd = wp.zeros(3, dtype=float, device=device)
        material_mu = wp.zeros(3, dtype=float, device=device)
        material_ke = wp.zeros(3, dtype=float, device=device)

        wp.launch(
            init_body_body_contacts_alm,
            dim=3,
            inputs=[
                contact_count,
                shape0,
                shape1,
                normal,
                wp.array([100.0] * 4, dtype=float, device=device),
                wp.zeros(4, dtype=float, device=device),
                # Friction must admit the saved tangential dual (|lam_t| = 6.4 at
                # lam_n = 6), otherwise the ALM restore is masked by cone projection.
                wp.array([2.0] * 4, dtype=float, device=device),
                0,
                1,
                1,
                match_index,
                history,
                reset_pending,
                reset_mask,
                shape_world,
                shape_body,
                body_world,
                -1.0,  # fixed-k sentinel
            ],
            outputs=[
                penalty_k,
                contact_lambda,
                material_kd,
                material_mu,
                material_ke,
            ],
            device=device,
        )

        lam = contact_lambda.numpy()
        # Rows 0 and 1 own the selected world (via endpoint-0 body and endpoint-1
        # shape respectively): both cold-start with a zero dual.
        for row in (0, 1):
            np.testing.assert_allclose(lam[row], 0.0)
        # Row 2 owns unselected world 1 and warm-restores its saved slot (1):
        # the dual comes from history through the nonidentity slot.
        np.testing.assert_allclose(lam[2], [4.0, 5.0, 6.0])
        # The kernel must not mutate the pipeline-owned correspondence.
        np.testing.assert_array_equal(match_index.numpy(), [2, 0, 1])


def _joint_angular_dual_projects_free_axis_lambda(test, device):
    """Angular dual updates should discard lambda on free angular axes."""
    with wp.ScopedDevice(device):
        joint_type = wp.array([int(newton.JointType.REVOLUTE)], dtype=wp.int32, device=device)
        joint_enabled = wp.array([True], dtype=bool, device=device)
        joint_parent = wp.array([-1], dtype=wp.int32, device=device)
        joint_child = wp.array([0], dtype=wp.int32, device=device)
        joint_x_p = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        joint_x_c = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        joint_axis = wp.array([[1.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        joint_rod_rest_kb_local = wp.zeros(1, dtype=wp.vec3, device=device)
        joint_rod_rest_twist = wp.zeros(1, dtype=float, device=device)
        joint_qd_start = wp.array([0], dtype=wp.int32, device=device)
        joint_target_q_start = wp.array([0], dtype=wp.int32, device=device)
        joint_constraint_start = wp.array([0], dtype=wp.int32, device=device)
        body_q = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        body_q_rest = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        joint_dof_dim = wp.array([[0, 0]], dtype=wp.int32, device=device)
        joint_c0_lin = wp.zeros(1, dtype=wp.vec3, device=device)
        joint_c0_ang = wp.zeros(1, dtype=wp.vec3, device=device)
        joint_is_hard = wp.array([1, 1, 0], dtype=wp.int32, device=device)
        joint_material_k = wp.array([10.0, 10.0, 10.0], dtype=float, device=device)
        joint_target_ke = wp.array([0.0], dtype=float, device=device)
        joint_target_kd = wp.array([0.0], dtype=float, device=device)
        joint_target_pos = wp.array([0.0], dtype=float, device=device)
        joint_target_vel = wp.array([0.0], dtype=float, device=device)
        joint_limit_lower = wp.array([-1.0], dtype=float, device=device)
        joint_limit_upper = wp.array([1.0], dtype=float, device=device)
        joint_limit_ke = wp.array([0.0], dtype=float, device=device)
        joint_limit_kd = wp.array([0.0], dtype=float, device=device)
        joint_rest_angle = wp.array([0.0], dtype=float, device=device)
        joint_penalty_k = wp.array([10.0, 10.0, 10.0], dtype=float, device=device)
        lambda_lin = wp.zeros(1, dtype=wp.vec3, device=device)
        lambda_ang = wp.array([[5.0, 2.0, 3.0]], dtype=wp.vec3, device=device)
        drive_limit_support = wp.zeros(1, dtype=float, device=device)
        drive_limit_lambda = wp.zeros(1, dtype=float, device=device)
        limit_lambda = wp.zeros(1, dtype=float, device=device)

        wp.launch(
            update_duals_joint,
            dim=1,
            inputs=[
                joint_type,
                joint_enabled,
                joint_parent,
                joint_child,
                joint_x_p,
                joint_x_c,
                joint_axis,
                joint_rod_rest_kb_local,
                joint_rod_rest_twist,
                joint_qd_start,
                joint_target_q_start,
                joint_constraint_start,
                body_q,
                body_q,
                body_q_rest,
                joint_dof_dim,
                joint_c0_lin,
                joint_c0_ang,
                joint_is_hard,
                0.0,
                joint_material_k,
                joint_material_k,
                0,
                0.0,
                0.0,
                joint_target_ke,
                joint_target_kd,
                joint_target_pos,
                joint_target_vel,
                joint_limit_lower,
                joint_limit_upper,
                joint_limit_ke,
                joint_limit_kd,
                joint_rest_angle,
                drive_limit_support,
                1.0 / 60.0,
            ],
            outputs=[
                joint_penalty_k,
                lambda_lin,
                lambda_ang,
                drive_limit_lambda,
                limit_lambda,
            ],
            device=device,
        )

        np.testing.assert_allclose(lambda_ang.numpy(), [[0.0, 2.0, 3.0]])


def _rod_soft_dual_slots_clear_preserved_lambda(test, device):
    """Verify soft rod slots clear stale lambda components when recombined."""
    with wp.ScopedDevice(device):
        joint_type = wp.array([int(newton.JointType.ROD)], dtype=wp.int32, device=device)
        joint_enabled = wp.array([True], dtype=bool, device=device)
        joint_parent = wp.array([-1], dtype=wp.int32, device=device)
        joint_child = wp.array([0], dtype=wp.int32, device=device)
        joint_x_p = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        joint_x_c = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        joint_axis = wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3, device=device)
        joint_rod_rest_kb_local = wp.zeros(1, dtype=wp.vec3, device=device)
        joint_rod_rest_twist = wp.zeros(1, dtype=float, device=device)
        joint_qd_start = wp.array([0], dtype=wp.int32, device=device)
        joint_target_q_start = wp.array([0], dtype=wp.int32, device=device)
        joint_constraint_start = wp.array([0], dtype=wp.int32, device=device)
        body_q = wp.array(
            [wp.transform(wp.vec3(0.2, 0.3, 0.4), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.3))],
            dtype=wp.transform,
            device=device,
        )
        body_q_rest = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        joint_dof_dim = wp.array([[0, 0]], dtype=wp.int32, device=device)
        joint_c0_lin = wp.zeros(1, dtype=wp.vec3, device=device)
        joint_c0_ang = wp.zeros(1, dtype=wp.vec3, device=device)
        joint_is_hard = wp.array([0, 0, 0, 0], dtype=wp.int32, device=device)
        joint_material_k = wp.array([10.0, 10.0, 10.0, 10.0], dtype=float, device=device)
        joint_rho = wp.zeros(4, dtype=float, device=device)
        joint_target_ke = wp.array([0.0], dtype=float, device=device)
        joint_target_kd = wp.array([0.0], dtype=float, device=device)
        joint_target_pos = wp.array([0.0], dtype=float, device=device)
        joint_target_vel = wp.array([0.0], dtype=float, device=device)
        joint_limit_lower = wp.array([-1.0], dtype=float, device=device)
        joint_limit_upper = wp.array([1.0], dtype=float, device=device)
        joint_limit_ke = wp.array([0.0], dtype=float, device=device)
        joint_limit_kd = wp.array([0.0], dtype=float, device=device)
        joint_rest_angle = wp.array([0.0], dtype=float, device=device)
        drive_limit_support = wp.zeros(1, dtype=float, device=device)
        joint_penalty_k = wp.array([10.0, 10.0, 10.0, 10.0], dtype=float, device=device)
        lambda_lin = wp.array([[1.0, 2.0, 3.0]], dtype=wp.vec3, device=device)
        lambda_ang = wp.array([[4.0, 5.0, 6.0]], dtype=wp.vec3, device=device)
        drive_limit_lambda = wp.zeros(1, dtype=float, device=device)
        limit_lambda = wp.zeros(1, dtype=float, device=device)

        wp.launch(
            update_duals_joint,
            dim=1,
            inputs=[
                joint_type,
                joint_enabled,
                joint_parent,
                joint_child,
                joint_x_p,
                joint_x_c,
                joint_axis,
                joint_rod_rest_kb_local,
                joint_rod_rest_twist,
                joint_qd_start,
                joint_target_q_start,
                joint_constraint_start,
                body_q,
                body_q,
                body_q_rest,
                joint_dof_dim,
                joint_c0_lin,
                joint_c0_ang,
                joint_is_hard,
                0.0,
                joint_material_k,
                joint_rho,
                0,
                0.0,
                0.0,
                joint_target_ke,
                joint_target_kd,
                joint_target_pos,
                joint_target_vel,
                joint_limit_lower,
                joint_limit_upper,
                joint_limit_ke,
                joint_limit_kd,
                joint_rest_angle,
                drive_limit_support,
                1.0 / 60.0,
            ],
            outputs=[
                joint_penalty_k,
                lambda_lin,
                lambda_ang,
                drive_limit_lambda,
                limit_lambda,
            ],
            device=device,
        )

        np.testing.assert_allclose(lambda_lin.numpy(), [[0.0, 0.0, 0.0]])
        np.testing.assert_allclose(lambda_ang.numpy(), [[0.0, 0.0, 0.0]])


def _joint_force_projection_filters_free_direction(test, device):
    """Projected joint force path should not apply force along free directions."""
    with wp.ScopedDevice(device):
        linear_force = wp.zeros(1, dtype=wp.vec3, device=device)
        angular_torque = wp.zeros(1, dtype=wp.vec3, device=device)
        wp.launch(
            _eval_directional_joint_projection_kernel,
            dim=1,
            outputs=[linear_force, angular_torque],
            device=device,
        )

        np.testing.assert_allclose(linear_force.numpy(), [[0.0, 11.0, 17.0]], rtol=1e-6, atol=1e-6)
        angular_torque_np = angular_torque.numpy()
        np.testing.assert_allclose(angular_torque_np[:, 0], [0.0], rtol=1e-6, atol=1e-6)
        test.assertGreater(np.linalg.norm(angular_torque_np[:, 1:]), 0.0)


def _body_particle_contact_damping_is_absolute(test, device):
    """Changing contact stiffness should not change the damping contribution."""
    with wp.ScopedDevice(device):
        particle_radius = wp.array([0.1], dtype=float, device=device)
        shape_material_mu = wp.array([0.0], dtype=float, device=device)
        shape_body = wp.array([-1], dtype=wp.int32, device=device)
        body_q = wp.zeros(0, dtype=wp.transform, device=device)
        body_q_prev = wp.zeros(0, dtype=wp.transform, device=device)
        body_qd = wp.zeros(0, dtype=wp.spatial_vector, device=device)
        body_com = wp.zeros(0, dtype=wp.vec3, device=device)
        contact_shape = wp.array([0], dtype=wp.int32, device=device)
        contact_body_pos = wp.zeros(1, dtype=wp.vec3, device=device)
        contact_body_vel = wp.zeros(1, dtype=wp.vec3, device=device)
        contact_normal = wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3, device=device)
        forces = wp.zeros(4, dtype=wp.vec3, device=device)

        wp.launch(
            _eval_body_particle_contact_damping_kernel,
            dim=4,
            inputs=[
                particle_radius,
                shape_material_mu,
                shape_body,
                body_q,
                body_q_prev,
                body_qd,
                body_com,
                contact_shape,
                contact_body_pos,
                contact_body_vel,
                contact_normal,
                wp.zeros(0, dtype=float, device=device),
            ],
            outputs=[forces],
            device=device,
        )

        force_np = forces.numpy()
        damping_low_ke = force_np[0] - force_np[1]
        damping_high_ke = force_np[2] - force_np[3]
        np.testing.assert_allclose(damping_low_ke, damping_high_ke, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(damping_low_ke, [0.0, 0.0, 2.0], rtol=1.0e-6, atol=1.0e-6)


def _body_particle_contact_damping_ignores_penalty_ramp(test, device):
    """Ramped body-particle contact stiffness must not scale absolute damping."""
    with wp.ScopedDevice(device):
        particle_q = wp.array([[0.0, 0.0, 0.04]] * 4, dtype=wp.vec3, device=device)
        particle_q_prev = wp.array([[0.0, 0.0, 0.05]] * 4, dtype=wp.vec3, device=device)
        particle_radius = wp.array([0.1] * 4, dtype=float, device=device)

        # Single total soft counter; only the particle path is exercised here (records (p, -1, -1)).
        contact_count = wp.array([4], dtype=int, device=device)
        contact_indices = wp.array([[0, -1, -1], [1, -1, -1], [2, -1, -1], [3, -1, -1]], dtype=wp.vec3i, device=device)
        contact_penalty_k = wp.array([400.0, 400.0, 100.0, 100.0], dtype=float, device=device)
        contact_material_kd = wp.array([20.0, 0.0, 20.0, 0.0], dtype=float, device=device)
        contact_material_mu = wp.zeros(4, dtype=float, device=device)

        shape_body = wp.array([-1], dtype=int, device=device)
        body_q = wp.zeros(0, dtype=wp.transform, device=device)
        body_q_prev = wp.zeros(0, dtype=wp.transform, device=device)
        body_qd = wp.zeros(0, dtype=wp.spatial_vector, device=device)
        body_com = wp.zeros(0, dtype=wp.vec3, device=device)
        contact_shape = wp.zeros(4, dtype=int, device=device)
        contact_body_pos = wp.zeros(4, dtype=wp.vec3, device=device)
        contact_body_vel = wp.zeros(4, dtype=wp.vec3, device=device)
        contact_normal = wp.array([[0.0, 0.0, 1.0]] * 4, dtype=wp.vec3, device=device)

        forces = wp.zeros(4, dtype=wp.vec3, device=device)
        hessians = wp.zeros(4, dtype=wp.mat33, device=device)

        # Launch the production gather kernel the way SolverVBD does: build the per-particle
        # incidence lists over the active prefix, then gather one color (all four particles here).
        contact_head = wp.full(4, -1, dtype=int, device=device)
        contact_next = wp.empty(3 * 4, dtype=int, device=device)
        wp.launch(
            build_particle_body_contact_adjacency_active,
            dim=4,
            inputs=[contact_indices, contact_count, 4, contact_head, contact_next],
            device=device,
        )
        color_group = wp.array([0, 1, 2, 3], dtype=wp.int32, device=device)
        wp.launch(
            gather_particle_body_contact_force_and_hessian,
            dim=4,
            block_dim=_PARTICLE_CONTACT_GATHER_BLOCK_DIM,
            inputs=[
                0.1,
                color_group,
                particle_q_prev,
                particle_q,
                0.01,
                False,  # legacy quadratic rigid-soft normal law
                particle_radius,
                contact_indices,
                contact_head,
                contact_next,
                contact_penalty_k,
                contact_material_kd,
                contact_material_mu,
                shape_body,
                body_q,
                body_q_prev,
                body_qd,
                body_com,
                contact_shape,
                contact_body_pos,
                contact_body_vel,
                contact_normal,
                wp.zeros(0, dtype=float, device=device),
                wp.zeros(4, dtype=wp.vec3, device=device),  # barycentric (unused on the particle path)
            ],
            outputs=[forces, hessians],
            device=device,
        )

        force_np = forces.numpy()
        damping_ramped = force_np[0] - force_np[1]
        damping_unramped = force_np[2] - force_np[3]
        np.testing.assert_allclose(damping_ramped, damping_unramped, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(damping_unramped, [0.0, 0.0, 2.0], rtol=1.0e-6, atol=1.0e-6)


def _make_particle_contact_gather_data(device, capacity=17, boundary=5, particle_count=6):
    particle_q = wp.array(
        [[0.03 * i, 0.02 * (i % 2), 0.04] for i in range(particle_count)],
        dtype=wp.vec3,
        device=device,
    )
    particle_q_prev = wp.array(
        [[0.03 * i, 0.02 * (i % 2), 0.05] for i in range(particle_count)],
        dtype=wp.vec3,
        device=device,
    )
    particle_colors = wp.array([i % 3 for i in range(particle_count)], dtype=int, device=device)
    color_groups = [
        wp.array([i for i in range(particle_count) if i % 3 == color], dtype=wp.int32, device=device)
        for color in range(3)
    ]

    indices = []
    barycentric = []
    for contact_index in range(capacity):
        p0 = contact_index % particle_count
        if contact_index % 3 == 0:
            indices.append([p0, -1, -1])
            barycentric.append([1.0, 0.0, 0.0])
        elif contact_index % 3 == 1:
            indices.append([p0, (p0 + 1) % particle_count, -1])
            barycentric.append([0.4, 0.6, 0.0])
        else:
            # Duplicate p0 deliberately: both incidence contributions must be retained.
            indices.append([p0, (p0 + 1) % particle_count, p0])
            barycentric.append([0.2, 0.3, 0.5])

    contact_indices = wp.array(indices, dtype=wp.vec3i, device=device)
    contact_barycentric = wp.array(barycentric, dtype=wp.vec3, device=device)
    return {
        "capacity": capacity,
        "boundary": boundary,
        "particle_count": particle_count,
        "particle_q": particle_q,
        "particle_q_prev": particle_q_prev,
        "particle_colors": particle_colors,
        "particle_radius": wp.full(particle_count, 0.1, dtype=float, device=device),
        "color_groups": color_groups,
        "contact_indices": contact_indices,
        "contact_penalty_k": wp.array([100.0 + i for i in range(capacity)], dtype=float, device=device),
        "contact_material_ke": wp.full(capacity, 200.0, dtype=float, device=device),
        "contact_material_kd": wp.full(capacity, 3.0, dtype=float, device=device),
        "contact_material_mu": wp.zeros(capacity, dtype=float, device=device),
        "shape_body": wp.array([-1], dtype=int, device=device),
        "body_q": wp.zeros(0, dtype=wp.transform, device=device),
        "body_q_prev": wp.zeros(0, dtype=wp.transform, device=device),
        "body_qd": wp.zeros(0, dtype=wp.spatial_vector, device=device),
        "body_com": wp.zeros(0, dtype=wp.vec3, device=device),
        "contact_shape": wp.zeros(capacity, dtype=int, device=device),
        "contact_body_pos": wp.zeros(capacity, dtype=wp.vec3, device=device),
        "contact_body_vel": wp.zeros(capacity, dtype=wp.vec3, device=device),
        "contact_normal": wp.array([[0.0, 0.0, 1.0]] * capacity, dtype=wp.vec3, device=device),
        "shape_margin": wp.zeros(0, dtype=float, device=device),
        "contact_barycentric": contact_barycentric,
    }


def _particle_contact_gather_material_inputs(data):
    return [
        data["contact_penalty_k"],
        data["contact_material_kd"],
        data["contact_material_mu"],
        data["shape_body"],
        data["body_q"],
        data["body_q_prev"],
        data["body_qd"],
        data["body_com"],
        data["contact_shape"],
        data["contact_body_pos"],
        data["contact_body_vel"],
        data["contact_normal"],
        data["shape_margin"],
        data["contact_barycentric"],
    ]


def _launch_particle_contact_gather(data, contact_count, contact_head, contact_next, forces, hessians, device):
    wp.launch(
        build_particle_body_contact_adjacency_active,
        dim=data["capacity"],
        inputs=[
            data["contact_indices"],
            contact_count,
            data["capacity"],
            contact_head,
            contact_next,
        ],
        device=device,
    )
    for color_group in data["color_groups"]:
        wp.launch(
            gather_particle_body_contact_force_and_hessian,
            dim=color_group.size,
            block_dim=_PARTICLE_CONTACT_GATHER_BLOCK_DIM,
            inputs=[
                0.01,
                color_group,
                data["particle_q_prev"],
                data["particle_q"],
                1.0,
                data["particle_radius"],
                data["contact_indices"],
                contact_head,
                contact_next,
                *_particle_contact_gather_material_inputs(data),
            ],
            outputs=[forces, hessians],
            device=device,
        )


def _particle_contact_gather_order_pinned(test, device):
    """Produce identical gather sums for any chain permutation with the same membership.

    The adjacency build's atomic insertions make chain order scheduling-dependent; the gather's
    ascending-node-id consumption must erase that. Reversing every per-particle chain is one such
    permutation, so a regression to plain chain-order walking fails this test.
    """
    with wp.ScopedDevice(device):
        data = _make_particle_contact_gather_data(device)
        capacity = data["capacity"]
        n = data["particle_count"]
        contact_count = wp.array([capacity], dtype=int, device=device)
        head = wp.full(n, -1, dtype=wp.int32, device=device)
        nxt = wp.empty(3 * capacity, dtype=wp.int32, device=device)
        forces = wp.zeros(n, dtype=wp.vec3, device=device)
        hessians = wp.zeros(n, dtype=wp.mat33, device=device)
        _launch_particle_contact_gather(data, contact_count, head, nxt, forces, hessians, device)

        # Reverse every chain on the host: same membership, opposite link order.
        head_np = head.numpy()
        next_np = nxt.numpy()
        rev_head = np.full_like(head_np, -1)
        rev_next = next_np.copy()
        for particle in range(n):
            chain = []
            node = head_np[particle]
            while node >= 0:
                chain.append(node)
                node = next_np[node]
            chain.reverse()
            if chain:
                rev_head[particle] = chain[0]
                for i, node in enumerate(chain):
                    rev_next[node] = chain[i + 1] if i + 1 < len(chain) else -1
        rev_head_wp = wp.array(rev_head, dtype=wp.int32, device=device)
        rev_next_wp = wp.array(rev_next, dtype=wp.int32, device=device)
        forces_rev = wp.zeros(n, dtype=wp.vec3, device=device)
        hessians_rev = wp.zeros(n, dtype=wp.mat33, device=device)
        for color_group in data["color_groups"]:
            wp.launch(
                gather_particle_body_contact_force_and_hessian,
                dim=color_group.size,
                block_dim=_PARTICLE_CONTACT_GATHER_BLOCK_DIM,
                inputs=[
                    0.01,
                    color_group,
                    data["particle_q_prev"],
                    data["particle_q"],
                    1.0,
                    data["particle_radius"],
                    data["contact_indices"],
                    rev_head_wp,
                    rev_next_wp,
                    *_particle_contact_gather_material_inputs(data),
                ],
                outputs=[forces_rev, hessians_rev],
                device=device,
            )

        np.testing.assert_array_equal(forces.numpy().view(np.uint32), forces_rev.numpy().view(np.uint32))
        np.testing.assert_array_equal(hessians.numpy().view(np.uint32), hessians_rev.numpy().view(np.uint32))


def _particle_contact_adjacency_follows_swapped_contacts(test, device):
    """Rebuild the contact adjacency from whichever contacts buffer each step consumes.

    Steps with buffer A, an empty equal-capacity buffer B, then A again; the adjacency must track
    the passed buffer each time, including through the zero-contact intermediate.
    """
    with wp.ScopedDevice(device):
        model, _vertices = _build_edge_over_post(device)
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            soft_contact_gap=0.1,
            enable_rigid_soft_full_surface_contact=True,
        )
        contacts_a = pipeline.contacts()
        contacts_b = pipeline.contacts()  # same capacity, never collided: zero contacts
        state_in = model.state()
        state_out = model.state()
        pipeline.collide(state_in, contacts_a)
        test.assertGreater(int(contacts_a.soft_contact_count.numpy()[0]), 0)

        solver = newton.solvers.SolverVBD(model, iterations=1)
        solver.step(state_in, state_out, None, contacts_a, 1.0 / 120.0)
        members_a = solver._particle_contact_head.numpy() >= 0
        test.assertTrue(np.any(members_a))

        solver.step(state_in, state_out, None, contacts_b, 1.0 / 120.0)
        test.assertTrue(np.all(solver._particle_contact_head.numpy() == -1))

        solver.step(state_in, state_out, None, contacts_a, 1.0 / 120.0)
        # Chain fronts are insertion-racy; membership per particle is the stable invariant.
        np.testing.assert_array_equal(solver._particle_contact_head.numpy() >= 0, members_a)


def _particle_contact_gather_matches_legacy(test, device):
    """Linked particle incidence lists match the legacy mixed-record contact scatter."""
    with wp.ScopedDevice(device):
        data = _make_particle_contact_gather_data(device)
        capacity = data["capacity"]
        boundary = data["boundary"]
        particle_count = data["particle_count"]
        particle_q = data["particle_q"]
        particle_q_prev = data["particle_q_prev"]
        particle_colors = data["particle_colors"]
        particle_radius = data["particle_radius"]
        contact_indices = data["contact_indices"]
        common_material_inputs = _particle_contact_gather_material_inputs(data)

        for raw_count in (0, 1, boundary - 1, boundary, boundary + 1, capacity, capacity + 2):
            contact_count = wp.array([raw_count], dtype=int, device=device)
            contact_head = wp.full(particle_count, -1, dtype=int, device=device)
            contact_next = wp.empty(3 * capacity, dtype=int, device=device)
            legacy_forces = wp.zeros(particle_count, dtype=wp.vec3, device=device)
            legacy_hessians = wp.zeros(particle_count, dtype=wp.mat33, device=device)
            gather_forces = wp.zeros_like(legacy_forces)
            gather_hessians = wp.zeros_like(legacy_hessians)
            for current_color, _color_group in enumerate(data["color_groups"]):
                wp.launch(
                    accumulate_particle_body_contact_force_and_hessian,
                    dim=capacity,
                    inputs=[
                        0.01,
                        current_color,
                        particle_q_prev,
                        particle_q,
                        particle_colors,
                        1.0,
                        particle_radius,
                        contact_indices,
                        contact_count,
                        capacity,
                        common_material_inputs[0],
                        data["contact_material_ke"],
                        *common_material_inputs[1:],
                    ],
                    outputs=[legacy_forces, legacy_hessians],
                    device=device,
                )
            _launch_particle_contact_gather(
                data, contact_count, contact_head, contact_next, gather_forces, gather_hessians, device
            )

            with test.subTest(raw_count=raw_count):
                np.testing.assert_allclose(gather_forces.numpy(), legacy_forces.numpy(), rtol=1.0e-5, atol=1.0e-6)
                np.testing.assert_allclose(gather_hessians.numpy(), legacy_hessians.numpy(), rtol=1.0e-5, atol=1.0e-6)


def _particle_contact_gather_capture_replays_device_count(test, device):
    """A captured adjacency build and gather must consume a changing device-side count."""
    with wp.ScopedDevice(device):
        data = _make_particle_contact_gather_data(device)
        capacity = data["capacity"]
        particle_count = data["particle_count"]
        raw_count = wp.zeros(1, dtype=int, device=device)
        contact_count = wp.zeros(1, dtype=int, device=device)
        contact_head = wp.full(particle_count, -1, dtype=int, device=device)
        contact_next = wp.empty(3 * capacity, dtype=int, device=device)
        forces = wp.zeros(particle_count, dtype=wp.vec3, device=device)
        hessians = wp.zeros(particle_count, dtype=wp.mat33, device=device)

        wp.launch(
            _prepare_particle_contact_gather_replay,
            dim=particle_count,
            inputs=[raw_count, contact_count, contact_head, forces, hessians],
            device=device,
        )
        _launch_particle_contact_gather(data, contact_count, contact_head, contact_next, forces, hessians, device)
        wp.synchronize_device(device)

        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                _prepare_particle_contact_gather_replay,
                dim=particle_count,
                inputs=[raw_count, contact_count, contact_head, forces, hessians],
                device=device,
            )
            _launch_particle_contact_gather(data, contact_count, contact_head, contact_next, forces, hessians, device)
        graph = capture.graph
        test.assertIsNotNone(graph)

        for replay_count in (0, data["boundary"] + 1, capacity + 2, 1):
            reference_count = wp.array([replay_count], dtype=int, device=device)
            reference_head = wp.full(particle_count, -1, dtype=int, device=device)
            reference_next = wp.empty(3 * capacity, dtype=int, device=device)
            reference_forces = wp.zeros_like(forces)
            reference_hessians = wp.zeros_like(hessians)
            _launch_particle_contact_gather(
                data,
                reference_count,
                reference_head,
                reference_next,
                reference_forces,
                reference_hessians,
                device,
            )

            raw_count.assign([replay_count])
            wp.capture_launch(graph)
            with test.subTest(replay_count=replay_count):
                test.assertEqual(int(contact_count.numpy()[0]), replay_count)
                np.testing.assert_allclose(forces.numpy(), reference_forces.numpy(), rtol=1.0e-5, atol=1.0e-6)
                np.testing.assert_allclose(hessians.numpy(), reference_hessians.numpy(), rtol=1.0e-5, atol=1.0e-6)


def _particle_contact_gather_solver_step_dispatch_and_capture(test, device):
    """Exercise production gather dispatch, capture replay, and repeated-step consistency."""
    with wp.ScopedDevice(device):
        model, _vertices = _build_edge_over_post(device)
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            soft_contact_gap=0.1,
            enable_rigid_soft_full_surface_contact=True,
        )
        contacts = pipeline.contacts()
        state_in = model.state()
        state_out = model.state()
        pipeline.collide(state_in, contacts)
        active_count = min(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
        test.assertGreater(active_count, 0)

        solver = newton.solvers.SolverVBD(model, iterations=1)
        solver.step(state_in, state_out, None, contacts, 1.0 / 120.0)
        test.assertTrue(solver._particle_contact_adjacency_initialized)

        raw_count = wp.array([active_count], dtype=int, device=device)
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                _prepare_particle_contact_gather_replay,
                dim=model.particle_count,
                inputs=[
                    raw_count,
                    contacts.soft_contact_count,
                    solver._particle_contact_head,
                    solver.particle_forces,
                    solver.particle_hessians,
                ],
                device=device,
            )
            solver.step(state_in, state_out, None, contacts, 1.0 / 120.0)
        graph = capture.graph
        test.assertIsNotNone(graph)

        for replay_count in (0, active_count):
            raw_count.assign([replay_count])
            wp.capture_launch(graph)
            head = solver._particle_contact_head.numpy()
            with test.subTest(replay_count=replay_count):
                test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), replay_count)
                if replay_count == 0:
                    test.assertTrue(np.all(head == -1))
                else:
                    test.assertTrue(np.any(head >= 0))
                test.assertTrue(np.all(np.isfinite(state_out.particle_q.numpy())))

        # Repeated identical steps on the same contact buffer must produce identical results.
        deterministic_solver = newton.solvers.SolverVBD(
            model,
            iterations=1,
            deterministic=wp.DeterministicMode.RUN_TO_RUN,
        )
        test.assertEqual(deterministic_solver._particle_contact_head.shape[0], model.particle_count)
        contacts.soft_contact_count.assign([active_count])
        results = []
        for _ in range(2):
            state_a = model.state()
            state_b = model.state()
            deterministic_solver._particle_contact_adjacency_initialized = False
            deterministic_solver.step(state_a, state_b, None, contacts, 1.0 / 120.0)
            results.append(state_b.particle_q.numpy().copy())
        np.testing.assert_array_equal(results[0], results[1])


def _make_body_particle_dual_prefix_data(device, capacity):
    particle_count = 6
    particle_q = wp.array(
        [[0.03 * i, 0.02 * (i % 2), 0.0] for i in range(particle_count)],
        dtype=wp.vec3,
        device=device,
    )
    particle_radius = wp.full(particle_count, 0.1, dtype=float, device=device)

    indices = []
    barycentric = []
    for contact_index in range(capacity):
        p0 = contact_index % particle_count
        if contact_index % 3 == 0:
            indices.append([p0, -1, -1])
            barycentric.append([1.0, 0.0, 0.0])
        elif contact_index % 3 == 1:
            indices.append([p0, (p0 + 1) % particle_count, -1])
            barycentric.append([0.4, 0.6, 0.0])
        else:
            indices.append([p0, (p0 + 1) % particle_count, (p0 + 2) % particle_count])
            barycentric.append([0.2, 0.3, 0.5])

    return {
        "indices": wp.array(indices, dtype=wp.vec3i, device=device),
        "shape": wp.zeros(capacity, dtype=int, device=device),
        "body_pos": wp.array([[0.0, 0.0, 0.01 * (i + 1)] for i in range(capacity)], dtype=wp.vec3, device=device),
        "normal": wp.array([[0.0, 0.0, 1.0]] * capacity, dtype=wp.vec3, device=device),
        "barycentric": wp.array(barycentric, dtype=wp.vec3, device=device),
        "particle_q": particle_q,
        "particle_radius": particle_radius,
        "shape_body": wp.array([-1], dtype=int, device=device),
        "shape_margin": wp.zeros(0, dtype=float, device=device),
        "body_q": wp.zeros(0, dtype=wp.transform, device=device),
        "material_ke": wp.full(capacity, 100.0, dtype=float, device=device),
    }


def _launch_body_particle_dual_prefix(data, contact_count, capacity, beta, penalty_k, device):
    wp.launch(
        update_duals_body_particle_contacts,
        dim=capacity,
        inputs=[
            contact_count,
            data["indices"],
            data["shape"],
            data["body_pos"],
            data["normal"],
            data["barycentric"],
            data["particle_q"],
            data["particle_radius"],
            data["shape_body"],
            data["shape_margin"],
            data["body_q"],
            data["material_ke"],
            beta,
            penalty_k,
        ],
        device=device,
    )


def _expected_body_particle_dual_penalties(raw_count, capacity, beta, initial_penalty):
    expected = np.full(capacity, initial_penalty, dtype=np.float32)
    active_count = min(raw_count, capacity)
    penetration = 0.1 + 0.01 * np.arange(1, active_count + 1, dtype=np.float32)
    expected[:active_count] += beta * penetration
    return expected


def _body_particle_dual_active_prefix_boundaries(test, device):
    """Update each clamped active-prefix row once and leave the stale tail untouched."""
    with wp.ScopedDevice(device):
        capacity = 11
        beta = 2.5
        initial_penalty = 1.0
        data = _make_body_particle_dual_prefix_data(device, capacity)
        contact_count = wp.zeros(1, dtype=int, device=device)
        penalty_k = wp.zeros(capacity, dtype=float, device=device)

        counts = [0, 1, 3, 4, 5, 9, capacity, capacity + 2]
        for raw_count in counts:
            contact_count.assign([raw_count])
            penalty_k.fill_(initial_penalty)
            _launch_body_particle_dual_prefix(data, contact_count, capacity, beta, penalty_k, device)

            with test.subTest(raw_count=raw_count):
                np.testing.assert_allclose(
                    penalty_k.numpy(),
                    _expected_body_particle_dual_penalties(raw_count, capacity, beta, initial_penalty),
                    rtol=1.0e-6,
                    atol=1.0e-6,
                )


def _body_particle_dual_capture_replays_device_count(test, device):
    """A captured dual-update launch must consume a changing device-side contact count on replay."""
    with wp.ScopedDevice(device):
        capacity = 11
        beta = 2.5
        initial_penalty = 1.0
        data = _make_body_particle_dual_prefix_data(device, capacity)
        raw_count = wp.zeros(1, dtype=int, device=device)
        contact_count = wp.zeros(1, dtype=int, device=device)
        penalty_k = wp.zeros(capacity, dtype=float, device=device)

        # Compile and validate both kernels before capture. The captured preparation kernel
        # copies the mutable source count on-device and resets all output rows each replay.
        wp.launch(
            _prepare_body_particle_dual_prefix,
            dim=capacity,
            inputs=[raw_count, contact_count, penalty_k, initial_penalty],
            device=device,
        )
        _launch_body_particle_dual_prefix(data, contact_count, capacity, beta, penalty_k, device)
        wp.synchronize_device(device)

        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                _prepare_body_particle_dual_prefix,
                dim=capacity,
                inputs=[raw_count, contact_count, penalty_k, initial_penalty],
                device=device,
            )
            _launch_body_particle_dual_prefix(data, contact_count, capacity, beta, penalty_k, device)
        graph = capture.graph
        test.assertIsNotNone(graph)

        for replay_count in (0, 5, capacity + 2, 1):
            raw_count.assign([replay_count])
            wp.capture_launch(graph)

            with test.subTest(replay_count=replay_count):
                test.assertEqual(int(contact_count.numpy()[0]), replay_count)
                np.testing.assert_allclose(
                    penalty_k.numpy(),
                    _expected_body_particle_dual_penalties(replay_count, capacity, beta, initial_penalty),
                    rtol=1.0e-6,
                    atol=1.0e-6,
                )


def _body_body_contact_damping_ignores_penalty_ramp(test, device):
    """Ramped body-body contact stiffness must not scale absolute damping."""
    with wp.ScopedDevice(device):
        contact_count = wp.array([4], dtype=int, device=device)
        shape0 = wp.zeros(4, dtype=int, device=device)
        shape1 = wp.ones(4, dtype=int, device=device)
        point0 = wp.zeros(4, dtype=wp.vec3, device=device)
        point1 = wp.zeros(4, dtype=wp.vec3, device=device)
        offset0 = wp.zeros(4, dtype=wp.vec3, device=device)
        offset1 = wp.zeros(4, dtype=wp.vec3, device=device)
        normal = wp.array([[0.0, 0.0, 1.0]] * 4, dtype=wp.vec3, device=device)
        margin0 = wp.array([0.1] * 4, dtype=float, device=device)
        margin1 = wp.zeros(4, dtype=float, device=device)

        shape_body = wp.array([-1, 0], dtype=wp.int32, device=device)
        body_q = wp.array(
            [wp.transform(wp.vec3(0.0, 0.0, 0.04), wp.quat_identity())], dtype=wp.transform, device=device
        )
        body_q_prev = wp.array(
            [wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity())], dtype=wp.transform, device=device
        )
        body_com = wp.zeros(1, dtype=wp.vec3, device=device)

        contact_normal_rho = wp.zeros(4, dtype=float, device=device)
        penalty_k = wp.array([400.0, 400.0, 100.0, 100.0], dtype=float, device=device)
        material_ke = wp.array([100.0] * 4, dtype=float, device=device)
        material_kd = wp.array([20.0, 0.0, 20.0, 0.0], dtype=float, device=device)
        material_mu = wp.zeros(4, dtype=float, device=device)
        contact_tangent_rho = wp.zeros(4, dtype=float, device=device)
        contact_lambda = wp.zeros(4, dtype=wp.vec3, device=device)
        contact_c0 = wp.zeros(4, dtype=wp.vec3, device=device)

        body0 = wp.empty(4, dtype=wp.int32, device=device)
        body1 = wp.empty(4, dtype=wp.int32, device=device)
        point0_world = wp.empty(4, dtype=wp.vec3, device=device)
        point1_world = wp.empty(4, dtype=wp.vec3, device=device)
        force_on_body1 = wp.empty(4, dtype=wp.vec3, device=device)

        wp.launch(
            compute_rigid_contact_forces,
            dim=4,
            inputs=[
                0.1,
                contact_count,
                shape0,
                shape1,
                point0,
                point1,
                offset0,
                offset1,
                normal,
                margin0,
                margin1,
                shape_body,
                body_q,
                body_q_prev,
                body_com,
                penalty_k,
                contact_normal_rho,
                material_ke,
                material_kd,
                material_mu,
                contact_tangent_rho,
                contact_lambda,
                contact_c0,
                0.95,
                0,
                0,
                0.01,
            ],
            outputs=[body0, body1, point0_world, point1_world, force_on_body1],
            device=device,
        )

        force_np = force_on_body1.numpy()
        damping_ramped = force_np[0] - force_np[1]
        damping_unramped = force_np[2] - force_np[3]
        np.testing.assert_allclose(damping_ramped, damping_unramped, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(damping_unramped, [0.0, 0.0, 2.0], rtol=1.0e-6, atol=1.0e-6)


def _spring_damping_is_axial(test, device):
    """Spring damping damps length change, not tangential rigid rotation."""
    with wp.ScopedDevice(device):
        theta = 0.1
        pos = wp.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]], dtype=wp.vec3, device=device)
        pos_anchor = wp.array(
            [
                [0.5 * math.cos(theta), 0.5 * math.sin(theta), 0.0],
                [-0.5 * math.cos(theta), -0.5 * math.sin(theta), 0.0],
            ],
            dtype=wp.vec3,
            device=device,
        )
        spring_indices = wp.array([0, 1], dtype=int, device=device)
        spring_rest_length = wp.array([2.0], dtype=float, device=device)
        spring_stiffness = wp.array([0.0], dtype=float, device=device)
        spring_damping = wp.array([20.0], dtype=float, device=device)
        force = wp.zeros(1, dtype=wp.vec3, device=device)
        hessian = wp.zeros(1, dtype=wp.mat33, device=device)

        wp.launch(
            _eval_spring_damping_kernel,
            dim=1,
            inputs=[pos, pos_anchor, spring_indices, spring_rest_length, spring_stiffness, spring_damping],
            outputs=[force, hessian],
            device=device,
        )

        np.testing.assert_allclose(force.numpy()[0], [0.0, 0.0, 0.0], rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(
            hessian.numpy()[0],
            [[200.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            rtol=1.0e-6,
            atol=1.0e-6,
        )


def _bending_damping_handles_degenerate_anchor(test, device):
    """Bending damping skips collapsed previous-step geometry."""
    with wp.ScopedDevice(device):
        pos = wp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.1],
            ],
            dtype=wp.vec3,
            device=device,
        )
        pos_anchor = wp.zeros(4, dtype=wp.vec3, device=device)
        edge_indices = wp.array([[0, 1, 2, 3]], dtype=wp.int32, ndim=2, device=device)
        edge_rest_angle = wp.array([0.0], dtype=float, device=device)
        edge_rest_length = wp.array([1.0], dtype=float, device=device)
        force_norms = wp.zeros(4, dtype=float, device=device)

        wp.launch(
            _eval_bending_degenerate_anchor_kernel,
            dim=4,
            inputs=[pos, pos_anchor, edge_indices, edge_rest_angle, edge_rest_length],
            outputs=[force_norms],
            device=device,
        )

        force_norms_np = force_norms.numpy()

    test.assertTrue(np.all(np.isfinite(force_norms_np)))
    np.testing.assert_allclose(force_norms_np, np.zeros(4), rtol=0.0, atol=1.0e-6)


def _elastic_damping_ignores_rigid_motion(test, device):
    """Elastic damping should not produce force under fixed-seed rigid rotations."""
    sample_count = 100
    data = _elastic_damping_rigid_motion_data(sample_count=sample_count, seed=17)

    with wp.ScopedDevice(device):
        pos = wp.array(data["pos"], dtype=wp.vec3, device=device)
        pos_anchor = wp.array(data["pos_anchor"], dtype=wp.vec3, device=device)
        spring_indices = wp.array(data["spring_indices"], dtype=int, device=device)
        spring_rest_length = wp.array(data["spring_rest_length"], dtype=float, device=device)
        spring_stiffness = wp.array(data["spring_stiffness"], dtype=float, device=device)
        spring_damping = wp.array(data["spring_damping"], dtype=float, device=device)
        tri_indices = wp.array(data["tri_indices"], dtype=wp.int32, ndim=2, device=device)
        tri_poses = wp.array(data["tri_poses"], dtype=wp.mat22, device=device)
        tri_areas = wp.array(data["tri_areas"], dtype=float, device=device)
        edge_indices = wp.array(data["edge_indices"], dtype=wp.int32, ndim=2, device=device)
        edge_rest_angle = wp.array(data["edge_rest_angle"], dtype=float, device=device)
        edge_rest_length = wp.array(data["edge_rest_length"], dtype=float, device=device)
        tet_indices = wp.array(data["tet_indices"], dtype=wp.int32, ndim=2, device=device)
        tet_poses = wp.array(data["tet_poses"], dtype=wp.mat33, device=device)
        force_norms = wp.zeros((sample_count, 4), dtype=float, device=device)

        wp.launch(
            _eval_elastic_damping_rigid_motion_kernel,
            dim=sample_count,
            inputs=[
                pos,
                pos_anchor,
                spring_indices,
                spring_rest_length,
                spring_stiffness,
                spring_damping,
                tri_indices,
                tri_poses,
                tri_areas,
                edge_indices,
                edge_rest_angle,
                edge_rest_length,
                tet_indices,
                tet_poses,
            ],
            outputs=[force_norms],
            device=device,
        )

        max_norms = force_norms.numpy().max(axis=0)

    np.testing.assert_allclose(
        max_norms,
        np.zeros(4),
        rtol=0.0,
        atol=1.0e-4,
        err_msg="Expected zero damping force for spring, membrane, bending, and tet rigid motions",
    )


def _contact_damping_ignores_rigid_motion(test, device):
    """Contact damping should not add force under fixed-seed rigid rotations."""
    sample_count = 100
    data = _contact_damping_rigid_motion_data(sample_count=sample_count, seed=29)

    with wp.ScopedDevice(device):
        particle_q = wp.array(data["particle_q"], dtype=wp.vec3, device=device)
        particle_q_prev = wp.array(data["particle_q_prev"], dtype=wp.vec3, device=device)
        particle_radius = wp.array(np.full(sample_count, 0.1, dtype=np.float32), dtype=float, device=device)
        shape_material_mu = wp.zeros(sample_count, dtype=float, device=device)
        shape_body = wp.array(np.arange(sample_count, dtype=np.int32), dtype=wp.int32, device=device)
        body_q = wp.array(data["body_q"], dtype=wp.transform, device=device)
        body_q_prev = wp.array(data["body_q_prev"], dtype=wp.transform, device=device)
        body_qd = wp.zeros(sample_count, dtype=wp.spatial_vector, device=device)
        body_com = wp.zeros(sample_count, dtype=wp.vec3, device=device)
        contact_shape = wp.array(np.arange(sample_count, dtype=np.int32), dtype=wp.int32, device=device)
        contact_body_pos = wp.zeros(sample_count, dtype=wp.vec3, device=device)
        contact_body_vel = wp.zeros(sample_count, dtype=wp.vec3, device=device)
        contact_normal = wp.array(data["contact_normal"], dtype=wp.vec3, device=device)

        rigid_body_q = wp.array(data["rigid_body_q"], dtype=wp.transform, device=device)
        rigid_body_q_prev = wp.array(data["rigid_body_q_prev"], dtype=wp.transform, device=device)
        rigid_body_com = wp.zeros(2 * sample_count, dtype=wp.vec3, device=device)

        soft_pos = wp.array(data["soft_pos"], dtype=wp.vec3, device=device)
        soft_pos_anchor = wp.array(data["soft_pos_anchor"], dtype=wp.vec3, device=device)
        tri_indices = wp.array(data["tri_indices"], dtype=wp.int32, ndim=2, device=device)
        rigid_delta_norms = wp.zeros(sample_count, dtype=float, device=device)
        body_particle_delta_norms = wp.zeros(sample_count, dtype=float, device=device)
        soft_delta_norms = wp.zeros(sample_count, dtype=float, device=device)

        wp.launch(
            _eval_rigid_contact_rigid_motion_kernel,
            dim=sample_count,
            inputs=[contact_normal, rigid_body_q, rigid_body_q_prev, rigid_body_com],
            outputs=[rigid_delta_norms],
            device=device,
        )
        wp.launch(
            _eval_body_particle_contact_rigid_motion_kernel,
            dim=sample_count,
            inputs=[
                particle_q,
                particle_q_prev,
                particle_radius,
                shape_material_mu,
                shape_body,
                body_q,
                body_q_prev,
                body_qd,
                body_com,
                contact_shape,
                contact_body_pos,
                contact_body_vel,
                contact_normal,
                wp.zeros(0, dtype=float, device=device),
            ],
            outputs=[body_particle_delta_norms],
            device=device,
        )
        wp.launch(
            _eval_vertex_triangle_contact_rigid_motion_kernel,
            dim=sample_count,
            inputs=[soft_pos, soft_pos_anchor, tri_indices],
            outputs=[soft_delta_norms],
            device=device,
        )

        max_delta_norms = np.array(
            [
                rigid_delta_norms.numpy().max(),
                body_particle_delta_norms.numpy().max(),
                soft_delta_norms.numpy().max(),
            ]
        )

    np.testing.assert_allclose(
        max_delta_norms,
        np.zeros(3),
        rtol=0.0,
        atol=1.0e-4,
        err_msg="Expected zero damping contribution for rigid-rigid, rigid-soft, and soft-soft rigid motions",
    )


def _self_contact_damping_uses_relative_gap_rate(test, device):
    """Uniform motion of a contact stencil should not add normal damping."""
    with wp.ScopedDevice(device):
        pos_np = np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.05],
            ],
            dtype=np.float32,
        )
        pos = wp.array(pos_np, dtype=wp.vec3, device=device)
        pos_prev = wp.array(pos_np + np.array([0.0, 0.0, 0.01], dtype=np.float32), dtype=wp.vec3, device=device)
        tri_indices = wp.array(np.array([[0, 1, 2]], dtype=np.int32), dtype=wp.int32, ndim=2, device=device)
        forces = wp.zeros(2, dtype=wp.vec3, device=device)
        hessians = wp.zeros(2, dtype=wp.mat33, device=device)

        wp.launch(
            _eval_vertex_triangle_uniform_motion_kernel,
            dim=2,
            inputs=[pos, pos_prev, tri_indices],
            outputs=[forces, hessians],
            device=device,
        )

        np.testing.assert_allclose(forces.numpy()[1], forces.numpy()[0], rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(hessians.numpy()[1], hessians.numpy()[0], rtol=1.0e-6, atol=1.0e-6)


def _d6_fully_free_structural_slots_are_inactive(test, device):
    """D6 structural slots should be inactive when all axes are free."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link()
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)

    JointDofConfig = newton.ModelBuilder.JointDofConfig
    joint = builder.add_joint_d6(
        -1,
        body,
        linear_axes=[
            JointDofConfig.create_unlimited(newton.Axis.X),
            JointDofConfig.create_unlimited(newton.Axis.Y),
            JointDofConfig.create_unlimited(newton.Axis.Z),
        ],
        angular_axes=[
            JointDofConfig.create_unlimited(newton.Axis.X),
            JointDofConfig.create_unlimited(newton.Axis.Y),
            JointDofConfig.create_unlimited(newton.Axis.Z),
        ],
    )
    builder.add_articulation([joint])

    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(model, rigid_compliant_alm=True)
    start = int(solver.joint_constraint_start.numpy()[joint])

    np.testing.assert_allclose(solver.joint_penalty_k.numpy()[start : start + 2], [0.0, 0.0])
    np.testing.assert_allclose(solver.joint_material_k.numpy()[start : start + 2], [0.0, 0.0])
    np.testing.assert_array_equal(solver.joint_is_hard.numpy()[start : start + 2], [0, 0])

    solver.joint_drive_lambda.fill_(1.0)
    solver.step(model.state(), model.state(), None, None, 1.0e-2)
    np.testing.assert_allclose(solver.joint_drive_limit_support.numpy(), 0.0)
    np.testing.assert_allclose(solver.joint_drive_lambda.numpy(), 0.0)


def _rigid_compliant_drive_preserves_material_equilibrium(test, device):
    """Verify a compliant drive preserves F=K(q-target) and resets its scalar state."""
    builder = newton.ModelBuilder()
    body = builder.add_link(xform=wp.transform_identity(), mass=1.0)
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_prismatic(
        -1,
        body,
        axis=(1.0, 0.0, 0.0),
        target_pos=0.1,
        target_ke=1000.0,
        target_kd=120.0,
        limit_lower=-1.0,
        limit_upper=1.0,
        limit_ke=1000.0,
        limit_kd=120.0,
    )
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))

    solver = newton.solvers.SolverVBD(model, iterations=5, rigid_compliant_alm=True)
    state_0 = model.state()
    state_1 = model.state()

    solver.joint_drive_lambda.fill_(123.0)
    solver.reset(state_0, flags=0)
    np.testing.assert_allclose(solver.joint_drive_lambda.numpy(), 0.0)

    load = 20.0
    wrench = np.array([[load, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    state_0.body_f.assign(wrench)
    state_1.body_f.assign(wrench)
    dt = 1.0 / 600.0
    for _ in range(480):
        solver.step(state_0, state_1, None, None, dt)
        state_0, state_1 = state_1, state_0
        state_0.body_f.assign(wrench)
        state_1.body_f.assign(wrench)

    expected_position = 0.1 + load / 1000.0
    test.assertAlmostEqual(float(state_0.body_q.numpy()[body, 0]), expected_position, delta=2.0e-3)
    test.assertLess(abs(float(state_0.body_qd.numpy()[body, 0])), 1.0e-2)

    dof = int(model.joint_qd_start.numpy()[joint])
    test.assertGreater(float(solver.joint_drive_limit_support.numpy()[dof]), 0.0)
    test.assertAlmostEqual(float(solver.joint_drive_lambda.numpy()[dof]), load, delta=1.0)


def _rigid_compliant_alm_validates_drive_limit_damping(test, device):
    """Verify constructor damping clamps negatives and authored DOF damping stays physical."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=1.0)
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_prismatic(
        -1,
        body,
        axis=(1.0, 0.0, 0.0),
        target_ke=100.0,
        target_kd=1.0,
        limit_lower=-1.0,
        limit_upper=1.0,
        limit_ke=100.0,
        limit_kd=1.0,
    )
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device=device)

    for argument, attribute in (
        ("rigid_joint_linear_kd", "rigid_joint_linear_kd"),
        ("rigid_joint_angular_kd", "rigid_joint_angular_kd"),
    ):
        solver = newton.solvers.SolverVBD(model, rigid_compliant_alm=True, **{argument: -1.0})
        test.assertEqual(getattr(solver, attribute), 0.0)
        for invalid in (np.nan, np.inf):
            with test.assertRaisesRegex(ValueError, argument):
                newton.solvers.SolverVBD(model, rigid_compliant_alm=True, **{argument: invalid})

    for attribute, name in (
        ("joint_target_kd", "model.joint_target_kd"),
        ("joint_limit_kd", "model.joint_limit_kd"),
    ):
        array = getattr(model, attribute)
        original = array.numpy().copy()
        for invalid in (-1.0, np.nan, np.inf):
            values = original.copy()
            values[0] = invalid
            array.assign(values)
            with test.assertRaisesRegex(ValueError, name):
                newton.solvers.SolverVBD(model, rigid_compliant_alm=True)
        array.assign(original)


def _rigid_compliant_alm_validates_contact_materials(test, device):
    """Verify compliant contact rejects invalid physical stiffness, damping, and friction."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=1.0)
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.color()
    model = builder.finalize(device=device)

    for attribute in ("shape_material_ke", "shape_material_kd", "shape_material_mu"):
        array = getattr(model, attribute)
        original = array.numpy().copy()
        for invalid in (-1.0, np.inf):
            values = original.copy()
            values[0] = invalid
            array.assign(values)
            with test.assertRaisesRegex(ValueError, f"model.{attribute}"):
                newton.solvers.SolverVBD(model, rigid_compliant_alm=True)
        array.assign(original)


def _joint_hard_soft_deprecation_describes_legacy_behavior(test, device):
    """Verify the warning distinguishes compliant behavior from the legacy path."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverVBD.register_custom_attributes(builder)
    body = builder.add_link(mass=1.0)
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_fixed(-1, body, custom_attributes={"vbd:joint_is_hard": 0})
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device=device)

    # The latch is solver-local so independently constructed solvers remain testable.
    for _ in range(2):
        with test.assertWarnsRegex(DeprecationWarning, "legacy AVBD still honors it") as warning:
            newton.solvers.SolverVBD(model, rigid_compliant_alm=False)
        test.assertEqual(warning.filename, __file__)


def _rigid_velocity_drive_preserves_legacy_damping_and_adds_compliant_support(test, device):
    """Verify a damping-only drive acts in both modes and gains ALM state only in the new path."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=1.0)
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_prismatic(
        -1,
        body,
        axis=(1.0, 0.0, 0.0),
        target_vel=0.0,
        target_ke=0.0,
        target_kd=120.0,
    )
    builder.joint_qd[-1] = 1.0
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device=device)

    legacy_solver = newton.solvers.SolverVBD(model, iterations=5, rigid_compliant_alm=False)
    legacy_state_0 = model.state()
    legacy_state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, legacy_state_0)
    legacy_initial_speed = float(legacy_state_0.body_qd.numpy()[body, 0])
    legacy_solver.step(legacy_state_0, legacy_state_1, None, None, 1.0 / 600.0)

    dof = int(model.joint_qd_start.numpy()[joint])
    test.assertEqual(float(legacy_solver.joint_drive_limit_support.numpy()[dof]), 0.0)
    test.assertLess(abs(float(legacy_state_1.body_qd.numpy()[body, 0])), abs(legacy_initial_speed))

    solver = newton.solvers.SolverVBD(model, iterations=5, rigid_compliant_alm=True)
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    initial_speed = abs(float(state_0.body_qd.numpy()[body, 0]))
    solver.step(state_0, state_1, None, None, 1.0 / 600.0)

    test.assertGreater(float(solver.joint_drive_limit_support.numpy()[dof]), 0.0)
    test.assertLess(abs(float(state_1.body_qd.numpy()[body, 0])), initial_speed)


def _rigid_drive_ignores_disabled_limit_bounds(test, device):
    """Verify finite bounds without limit material do not clamp a drive target."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=1.0)
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_prismatic(
        -1,
        body,
        axis=(1.0, 0.0, 0.0),
        target_pos=0.5,
        target_ke=1000.0,
        target_kd=50.0,
        limit_lower=-0.1,
        limit_upper=0.1,
        limit_ke=0.0,
        limit_kd=0.0,
    )
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device=device)
    # ModelBuilder sanitizes an out-of-range authored target even when the
    # corresponding limit material is disabled; exercise the runtime model
    # contract directly.
    model.joint_target_q.fill_(0.5)

    state_0 = model.state()
    state_1 = model.state()
    solver = newton.solvers.SolverVBD(model, iterations=5, rigid_compliant_alm=True)
    for _ in range(120):
        solver.step(state_0, state_1, None, None, 1.0 / 600.0)
        state_0, state_1 = state_1, state_0

    test.assertGreater(float(state_0.body_q.numpy()[body, 0]), 0.3)


def _rigid_compliant_limit_holds_under_load(test, device):
    """Verify a compliant projected limit holds an external load at its bound."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=125.0)
    body = builder.add_link()
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=shape_cfg)
    joint = builder.add_joint_prismatic(
        -1,
        body,
        axis=(1.0, 0.0, 0.0),
        limit_lower=-0.1,
        limit_upper=0.1,
        limit_ke=1.0e12,
        limit_kd=2.0e4,
    )
    builder.joint_q[-1] = 0.1
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device=device)
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)

    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    solver = newton.solvers.SolverVBD(model, iterations=2, rigid_compliant_alm=True)

    wrench = np.array([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    dt = 1.0 / 600.0
    for _ in range(120):
        state_0.body_f.assign(wrench)
        solver.step(state_0, state_1, None, None, dt)
        state_0, state_1 = state_1, state_0

    position = float(state_0.body_q.numpy()[body, 0])
    speed = float(state_0.body_qd.numpy()[body, 0])
    test.assertTrue(math.isfinite(position))
    test.assertTrue(math.isfinite(speed))
    test.assertAlmostEqual(position, 0.1, delta=5.0e-4)
    test.assertLess(abs(speed), 1.0e-2)


def _body_structural_k_refreshes_after_joint_enable_notification(test, device):
    """Verify body_structural_k tracks joint enable flips."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body_a = builder.add_link()
    body_b = builder.add_link()
    joint_a = builder.add_joint_fixed(-1, body_a)
    joint_b = builder.add_joint_fixed(-1, body_b, enabled=False)
    builder.add_articulation([joint_a])
    builder.add_articulation([joint_b])
    builder.color()

    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(
        model,
        rigid_compliant_alm=True,
        rigid_joint_linear_ke=1234.0,
    )
    structural_k = solver.body_structural_k

    np.testing.assert_allclose(structural_k.numpy(), [1234.0, 0.0])

    model.joint_enabled.assign([False, True])
    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)

    test.assertIs(solver.body_structural_k, structural_k)
    np.testing.assert_allclose(structural_k.numpy(), [0.0, 1234.0])


def _rigid_reset_state_and_history(test, device):
    """Behavioral reset: constructor baseline, flags, masks, and one-shot deferral."""

    def add_fixed_body(builder, x):
        # Dynamic root fixed to the world; with iterations=0 its velocity is a pure
        # pose finite-difference, identical to a kinematic root here.
        body = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()), mass=1.0)
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        joint = builder.add_joint_fixed(parent=-1, child=body)
        builder.add_articulation([joint])

    template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    add_fixed_body(template, 0.0)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    add_fixed_body(builder, -2.0)  # Global head range.
    builder.add_world(template)
    builder.add_world(template, xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()))
    add_fixed_body(builder, 4.0)  # Global tail range.
    builder.color()
    model = builder.finalize(device=device)

    body_world = model.body_world.numpy()
    joint_world = model.joint_world.numpy()
    np.testing.assert_array_equal(body_world, [-1, 0, 1, -1])
    np.testing.assert_array_equal(joint_world, [-1, 0, 1, -1])

    dt = 1.0e-2
    model_q = model.body_q.numpy()
    model_qd = model.body_qd.numpy()
    selected_bodies = body_world == 0
    selected_joints = joint_world == 0
    global_bodies = body_world < 0
    global_joints = joint_world < 0
    world_mask = wp.array([True, False, False], dtype=wp.bool, device=device)

    solver = newton.solvers.SolverVBD(model, iterations=0, rigid_compliant_alm=True)
    # A history-disabled solver (the default) allocates no contact-reset state.
    test.assertIsNone(solver._contact_history_reset_mask)
    test.assertIsNone(solver._contact_history_reset_pending)

    state = model.state()
    state_out = model.state()

    def step_swap():
        nonlocal state, state_out
        solver.step(state, state_out, None, None, dt)
        state, state_out = state_out, state

    # Phase 1: a non-model first State establishes the pose baseline. The first
    # step reports zero velocity; it would report a jump if it baselined from the
    # model defaults instead.
    first_q = model_q.copy()
    first_q[:, 0] += 5.0
    state.body_q.assign(first_q)
    state.body_qd.zero_()
    step_swap()
    np.testing.assert_allclose(state.body_qd.numpy(), 0.0, atol=1.0e-5)

    # Phase 2: the validation batch is non-mutating (a seeded joint sentinel proves it).
    solver.joint_lambda_lin.fill_(5.0)
    with test.assertRaisesRegex(ValueError, "argument is required"):
        solver.reset(None)
    with test.assertRaisesRegex(TypeError, "dtype bool"):
        solver.reset(state, world_mask=wp.array([1, 0, 0], dtype=wp.int32, device=device))
    with test.assertRaisesRegex(ValueError, "world_mask has size 1, expected 2 or 3"):
        solver.reset(state, world_mask=wp.array([True], dtype=wp.bool, device=device))
    np.testing.assert_allclose(solver.joint_lambda_lin.numpy(), 5.0)

    if device.is_cuda:
        # A requested body array on the wrong device fails.
        good_qd = state.body_qd
        state.body_qd = wp.clone(good_qd, device="cpu")
        with test.assertRaisesRegex(ValueError, "state.body_qd is on device cpu"):
            solver.reset(state, flags=newton.StateFlags.BODY_QD)
        # BODY_Q succeeds: the unrequested wrong-device body_qd never binds and is preserved.
        solver.reset(state, flags=newton.StateFlags.BODY_Q)
        test.assertEqual(str(state.body_qd.device), "cpu")
        state.body_qd = good_qd

    # Phase 3: immediate body-copy and joint selection (no steps; any armed pose
    # intent is consumed before the velocity phases below).
    custom_q = model_q.copy()
    custom_q[:, 0] += 10.0
    custom_qd = np.full_like(model_qd, 3.0)

    state.body_q.assign(custom_q)
    state.body_qd.assign(custom_qd)
    solver.joint_lambda_lin.fill_(7.0)
    solver.reset(state, world_mask=world_mask, flags=newton.StateFlags.BODY_Q)
    result_q = state.body_q.numpy()
    np.testing.assert_allclose(result_q[selected_bodies], model_q[selected_bodies])
    np.testing.assert_allclose(result_q[~selected_bodies], custom_q[~selected_bodies])
    np.testing.assert_allclose(state.body_qd.numpy(), custom_qd)
    np.testing.assert_allclose(solver.joint_lambda_lin.numpy()[selected_joints], 0.0)
    np.testing.assert_allclose(solver.joint_lambda_lin.numpy()[~selected_joints], 7.0)

    state.body_q.assign(custom_q)
    state.body_qd.assign(custom_qd)
    solver.reset(state, world_mask=world_mask, flags=newton.StateFlags.BODY_QD)
    np.testing.assert_allclose(state.body_q.numpy(), custom_q)
    result_qd = state.body_qd.numpy()
    np.testing.assert_allclose(result_qd[selected_bodies], model_qd[selected_bodies])
    np.testing.assert_allclose(result_qd[~selected_bodies], custom_qd[~selected_bodies])

    state.body_q.assign(custom_q)
    state.body_qd.assign(custom_qd)
    solver.reset(state, world_mask=world_mask, flags=0)
    np.testing.assert_allclose(state.body_q.numpy(), custom_q)
    np.testing.assert_allclose(state.body_qd.numpy(), custom_qd)

    # Consume any pose intent armed above and re-establish a known baseline.
    solver.reset(state)
    base_q = model_q.copy()
    base_q[:, 0] += 1.0
    state.body_q.assign(base_q)
    state.body_qd.zero_()
    step_swap()

    # Phase 4: an all-false reset arms nothing, so the next step finite-differences
    # a known delta for every body (a leaked pose baseline would zero some world).
    with test.assertWarnsRegex(DeprecationWarning, "world_count \\+ 1"):
        solver.reset(state, world_mask=wp.array([False, False], dtype=wp.bool, device=device))
    all_false_delta = 2.0
    moved_q = base_q.copy()
    moved_q[:, 0] += all_false_delta
    state.body_q.assign(moved_q)
    state.body_qd.zero_()
    step_swap()
    np.testing.assert_allclose(state.body_qd.numpy()[:, 0], all_false_delta / dt, atol=1.0e-1)

    # Phase 5: a full reset drains all joint history and restores model body State,
    # then defers pose so the next step reports zero velocity everywhere.
    solver.joint_penalty_k.fill_(123.0)
    solver.joint_C0_lin.fill_(11.0)
    solver.joint_C0_ang.fill_(12.0)
    solver.joint_lambda_lin.fill_(13.0)
    solver.joint_lambda_ang.fill_(14.0)
    solver.reset(state)
    np.testing.assert_allclose(state.body_q.numpy(), model_q)
    np.testing.assert_allclose(state.body_qd.numpy(), model_qd)
    np.testing.assert_allclose(solver.joint_penalty_k.numpy(), solver.joint_penalty_k_min.numpy())
    np.testing.assert_allclose(solver.joint_C0_lin.numpy(), 0.0)
    np.testing.assert_allclose(solver.joint_C0_ang.numpy(), 0.0)
    np.testing.assert_allclose(solver.joint_lambda_lin.numpy(), 0.0)
    np.testing.assert_allclose(solver.joint_lambda_ang.numpy(), 0.0)

    final_q = model_q.copy()
    final_q[:, 0] += np.arange(1, model.body_count + 1, dtype=np.float32)
    state.body_q.assign(final_q)
    state.body_qd.zero_()
    step_swap()
    np.testing.assert_allclose(state.body_qd.numpy(), 0.0, atol=1.0e-5)

    # One-shot: the consumed reset does not persist, so an ordinary later delta
    # finite-differences for every body.
    one_shot_delta = 4.0
    moved_final = final_q.copy()
    moved_final[:, 0] += one_shot_delta
    state.body_q.assign(moved_final)
    state.body_qd.zero_()
    step_swap()
    np.testing.assert_allclose(state.body_qd.numpy()[:, 0], one_shot_delta / dt, atol=1.0e-1)

    # Phase 6: a masked flags=0 reset defers only world 0 and drains only its joint
    # history. The next step zeroes selected velocity while the unselected world and
    # the globals finite-difference the jump.
    solver.joint_lambda_lin.fill_(9.0)
    solver.reset(state, world_mask=world_mask, flags=0)
    np.testing.assert_allclose(solver.joint_lambda_lin.numpy()[selected_joints], 0.0)
    np.testing.assert_allclose(solver.joint_lambda_lin.numpy()[~selected_joints], 9.0)
    masked_delta = 3.0
    jump_q = moved_final.copy()
    jump_q[:, 0] += masked_delta
    state.body_q.assign(jump_q)
    state.body_qd.zero_()
    step_swap()
    masked_qd = state.body_qd.numpy()
    np.testing.assert_allclose(masked_qd[selected_bodies, 0], 0.0, atol=1.0e-3)
    np.testing.assert_allclose(masked_qd[~selected_bodies, 0], masked_delta / dt, atol=1.0e-1)

    # Phase 7: the extended mask's final entry selects only global entities.
    global_mask = wp.array([False, False, True], dtype=wp.bool, device=device)
    custom_q = jump_q.copy()
    custom_q[:, 0] += 6.0
    custom_qd = np.full_like(model_qd, 2.0)
    state.body_q.assign(custom_q)
    state.body_qd.assign(custom_qd)
    solver.joint_lambda_lin.fill_(10.0)
    solver.reset(state, world_mask=global_mask, flags=newton.StateFlags.BODY_Q)

    result_q = state.body_q.numpy()
    np.testing.assert_allclose(result_q[global_bodies], model_q[global_bodies])
    np.testing.assert_allclose(result_q[~global_bodies], custom_q[~global_bodies])
    np.testing.assert_allclose(state.body_qd.numpy(), custom_qd)
    np.testing.assert_allclose(solver.joint_lambda_lin.numpy()[global_joints], 0.0)
    np.testing.assert_allclose(solver.joint_lambda_lin.numpy()[~global_joints], 10.0)

    global_delta = 2.0
    final_global_q = jump_q.copy()
    final_global_q[:, 0] += global_delta
    state.body_q.assign(final_global_q)
    state.body_qd.zero_()
    step_swap()
    global_qd = state.body_qd.numpy()
    np.testing.assert_allclose(global_qd[global_bodies], 0.0, atol=1.0e-3)
    np.testing.assert_allclose(global_qd[~global_bodies, 0], global_delta / dt, atol=1.0e-1)

    # An extended all-true mask has the same immediate selection as None.
    state.body_q.assign(custom_q)
    state.body_qd.assign(custom_qd)
    solver.joint_lambda_lin.fill_(11.0)
    solver.reset(
        state,
        world_mask=wp.array([True, True, True], dtype=wp.bool, device=device),
    )
    np.testing.assert_allclose(state.body_q.numpy(), model_q)
    np.testing.assert_allclose(state.body_qd.numpy(), model_qd)
    np.testing.assert_allclose(solver.joint_lambda_lin.numpy(), 0.0)


def _reset_masked_rigid_and_soft(test, device):
    """Verify one masked reset restores rigid bodies and deformables together per world.

    ``reset()`` is one entry point for both maximal body state and particle state.
    With fixed bodies, a cloth grid, and a tetrahedral soft grid sharing the same
    worlds and global (world -1) range, one ``world_mask`` selects both sides:
    ``BODY_Q`` / ``PARTICLE_Q`` restore positions and ``BODY_QD`` / ``PARTICLE_QD``
    velocities in lockstep, ``world_mask=None`` includes globals, and an explicit
    mask's final entry selects only globals. Companion to
    :func:`_rigid_reset_state_and_history`, which covers the rigid history and
    pose-deferral semantics in depth.
    """

    def add_content(builder, x):
        body = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()), mass=1.0)
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        joint = builder.add_joint_fixed(parent=-1, child=body)
        builder.add_articulation([joint])
        builder.add_cloth_grid(
            pos=(x, 0.5, 0.0),
            rot=wp.quat_identity(),
            vel=(0.0, 0.0, 0.0),
            dim_x=2,
            dim_y=2,
            cell_x=0.1,
            cell_y=0.1,
            mass=1.0,
        )
        builder.add_soft_grid(
            pos=wp.vec3(x, 0.5, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=1,
            dim_y=1,
            dim_z=1,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=100.0,
            k_mu=1.0e3,
            k_lambda=1.0e3,
            k_damp=0.0,
        )

    template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    add_content(template, 0.0)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    add_content(builder, -3.0)  # Global head range (world -1).
    builder.add_world(template)  # World 0.
    builder.add_world(template, xform=wp.transform(wp.vec3(3.0, 0.0, 0.0), wp.quat_identity()))  # World 1.
    add_content(builder, 6.0)  # Global tail range (world -1).
    builder.color()
    model = builder.finalize(device=device)

    body_world = model.body_world.numpy()
    particle_world = model.particle_world.numpy()
    np.testing.assert_array_equal(body_world, [-1, 0, 1, -1])
    # Cloth and tet particles populate both local worlds and the global range.
    test.assertTrue((particle_world == 0).any())
    test.assertTrue((particle_world == 1).any())
    test.assertTrue((particle_world < 0).any())

    model_bq = model.body_q.numpy()
    model_bqd = model.body_qd.numpy()
    model_pq = model.particle_q.numpy()
    model_pqd = model.particle_qd.numpy()
    body_selected = body_world == 0
    body_global = body_world < 0
    part_selected = particle_world == 0
    part_global = particle_world < 0

    solver = newton.solvers.SolverVBD(model, iterations=0, rigid_compliant_alm=True)
    state = model.state()

    def perturb():
        bq = model_bq.copy()
        bq[:, 0] += 10.0
        bqd = np.full_like(model_bqd, 3.0)
        pq = model_pq.copy()
        pq[:, 0] += 10.0
        pqd = np.full_like(model_pqd, 5.0)
        state.body_q.assign(bq)
        state.body_qd.assign(bqd)
        state.particle_q.assign(pq)
        state.particle_qd.assign(pqd)
        return bq, bqd, pq, pqd

    world_mask = wp.array([True, False, False], dtype=wp.bool, device=device)

    # World-0 mask, positions only: bodies and particles restore together; velocities untouched.
    bq, bqd, pq, pqd = perturb()
    solver.reset(state, world_mask=world_mask, flags=newton.StateFlags.BODY_Q | newton.StateFlags.PARTICLE_Q)
    result_bq = state.body_q.numpy()
    result_pq = state.particle_q.numpy()
    np.testing.assert_allclose(result_bq[body_selected], model_bq[body_selected])
    np.testing.assert_allclose(result_bq[~body_selected], bq[~body_selected])
    np.testing.assert_allclose(result_pq[part_selected], model_pq[part_selected])
    np.testing.assert_allclose(result_pq[~part_selected], pq[~part_selected])
    np.testing.assert_allclose(state.body_qd.numpy(), bqd)
    np.testing.assert_allclose(state.particle_qd.numpy(), pqd)

    # World-0 mask, velocities only: symmetric, positions untouched.
    bq, bqd, pq, pqd = perturb()
    solver.reset(state, world_mask=world_mask, flags=newton.StateFlags.BODY_QD | newton.StateFlags.PARTICLE_QD)
    np.testing.assert_allclose(state.body_q.numpy(), bq)
    np.testing.assert_allclose(state.particle_q.numpy(), pq)
    result_bqd = state.body_qd.numpy()
    result_pqd = state.particle_qd.numpy()
    np.testing.assert_allclose(result_bqd[body_selected], model_bqd[body_selected])
    np.testing.assert_allclose(result_bqd[~body_selected], bqd[~body_selected])
    np.testing.assert_allclose(result_pqd[part_selected], model_pqd[part_selected])
    np.testing.assert_allclose(result_pqd[~part_selected], pqd[~part_selected])

    # world_mask=None restores every body and particle, the global range included.
    perturb()
    solver.reset(state)
    np.testing.assert_allclose(state.body_q.numpy(), model_bq)
    np.testing.assert_allclose(state.body_qd.numpy(), model_bqd)
    np.testing.assert_allclose(state.particle_q.numpy(), model_pq)
    np.testing.assert_allclose(state.particle_qd.numpy(), model_pqd)

    # The extended mask's final entry selects only the global range, both sides.
    global_mask = wp.array([False, False, True], dtype=wp.bool, device=device)
    bq, bqd, pq, pqd = perturb()
    solver.reset(state, world_mask=global_mask)
    result_bq = state.body_q.numpy()
    result_bqd = state.body_qd.numpy()
    result_pq = state.particle_q.numpy()
    result_pqd = state.particle_qd.numpy()
    np.testing.assert_allclose(result_bq[body_global], model_bq[body_global])
    np.testing.assert_allclose(result_bq[~body_global], bq[~body_global])
    np.testing.assert_allclose(result_bqd[body_global], model_bqd[body_global])
    np.testing.assert_allclose(result_bqd[~body_global], bqd[~body_global])
    np.testing.assert_allclose(result_pq[part_global], model_pq[part_global])
    np.testing.assert_allclose(result_pq[~part_global], pq[~part_global])
    np.testing.assert_allclose(result_pqd[part_global], model_pqd[part_global])
    np.testing.assert_allclose(result_pqd[~part_global], pqd[~part_global])

    if device.is_cuda:
        # A requested particle array on the wrong device fails; an unrequested one
        # never binds and is preserved.
        perturb()
        good_pqd = state.particle_qd
        state.particle_qd = wp.clone(good_pqd, device="cpu")
        with test.assertRaisesRegex(ValueError, "state.particle_qd is on device cpu"):
            solver.reset(state, flags=newton.StateFlags.PARTICLE_QD)
        solver.reset(state, flags=newton.StateFlags.PARTICLE_Q)
        np.testing.assert_allclose(state.particle_q.numpy(), model_pq)
        test.assertEqual(str(state.particle_qd.device), "cpu")
        state.particle_qd = good_pqd

        # The symmetric particle_q guard rejects before any field is written.
        _, _, pq, _ = perturb()
        good_pq = state.particle_q
        state.particle_q = wp.clone(good_pq, device="cpu")
        with test.assertRaisesRegex(ValueError, "state.particle_q is on device cpu"):
            solver.reset(state, flags=newton.StateFlags.PARTICLE_Q)
        np.testing.assert_allclose(state.particle_q.numpy(), pq)
        state.particle_q = good_pq


def _soft_reset_particle_only_and_external(test, device):
    """Verify deformable reset runs when SolverVBD performs no internal rigid integration.

    Covers a particle-only model (no bodies) and a model whose bodies are
    integrated by an external rigid solver; both take the reset path that
    early-returns before any rigid mutation, so particle restoration must happen
    beforehand.
    """
    # Particle-only model: no bodies, so internal_body_reset is False.
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_cloth_grid(
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=(0.0, 0.0, 0.0),
        dim_x=2,
        dim_y=2,
        cell_x=0.1,
        cell_y=0.1,
        mass=1.0,
    )
    builder.color()
    model = builder.finalize(device=device)
    test.assertEqual(model.body_count, 0)

    solver = newton.solvers.SolverVBD(model, iterations=0)
    state = model.state()
    model_q = model.particle_q.numpy()
    model_qd = model.particle_qd.numpy()
    moved_q = model_q.copy()
    moved_q[:, 0] += 4.0
    state.particle_q.assign(moved_q)
    state.particle_qd.assign(np.full_like(model_qd, 2.0))
    solver.reset(state)
    np.testing.assert_allclose(state.particle_q.numpy(), model_q)
    np.testing.assert_allclose(state.particle_qd.numpy(), model_qd)

    # External-rigid coupling: bodies exist but are integrated elsewhere, so the
    # rigid reset path early-returns while masked particle reset still applies.
    template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = template.add_body(mass=1.0)
    template.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    template.add_cloth_grid(
        pos=(0.0, 0.0, 0.5),
        rot=wp.quat_identity(),
        vel=(0.0, 0.0, 0.0),
        dim_x=2,
        dim_y=2,
        cell_x=0.1,
        cell_y=0.1,
        mass=1.0,
    )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_world(template)
    builder.add_world(template, xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()))
    builder.color()
    model = builder.finalize(device=device)
    test.assertGreater(model.body_count, 0)

    solver = newton.solvers.SolverVBD(model, iterations=0, integrate_with_external_rigid_solver=True)
    state = model.state()
    model_q = model.particle_q.numpy()
    particle_world = model.particle_world.numpy()
    selected = particle_world == 0
    moved_q = model_q.copy()
    moved_q[:, 0] += 7.0
    state.particle_q.assign(moved_q)
    world_mask = wp.array([True, False, False], dtype=wp.bool, device=device)
    solver.reset(state, world_mask=world_mask, flags=newton.StateFlags.PARTICLE_Q)
    result_q = state.particle_q.numpy()
    np.testing.assert_allclose(result_q[selected], model_q[selected])
    np.testing.assert_allclose(result_q[~selected], moved_q[~selected])


def _soft_reset_captured_graph_restores_particles(test, device):
    """Verify a captured masked particle reset restores selected-world defaults on replay.

    ``reset()`` launches ``reset_particle_state`` without allocation, so it is
    capturable, and the ``world_mask`` is read device-side, so a replay honors the
    same per-world selection.
    """
    template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    template.add_cloth_grid(
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=(0.0, 0.0, 0.0),
        dim_x=2,
        dim_y=2,
        cell_x=0.1,
        cell_y=0.1,
        mass=1.0,
    )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_world(template)
    builder.add_world(template, xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()))
    builder.color()
    model = builder.finalize(device=device)
    test.assertEqual(model.body_count, 0)

    solver = newton.solvers.SolverVBD(model, iterations=0)
    state = model.state()
    model_q = model.particle_q.numpy()
    particle_world = model.particle_world.numpy()
    selected = particle_world == 0
    world_mask = wp.array([True, False, False], dtype=wp.bool, device=device)

    with wp.ScopedCapture(device=device) as capture:
        solver.reset(state, world_mask=world_mask, flags=newton.StateFlags.PARTICLE_Q)
    graph = capture.graph
    test.assertIsNotNone(graph)

    # In-place edits keep the captured buffer pointers valid; replay must restore
    # only the selected world from the model defaults.
    moved_q = model_q.copy()
    moved_q[:, 0] += 5.0
    state.particle_q.assign(moved_q)
    wp.capture_launch(graph)
    result_q = state.particle_q.numpy()
    np.testing.assert_allclose(result_q[selected], model_q[selected])
    np.testing.assert_allclose(result_q[~selected], moved_q[~selected])


def _soft_reset_then_step_advances_cloth_and_tet(test, device):
    """Verify a finite step after reset advances cloth and tet from the restored state.

    Issue #3400 requires a real solver step after a deformable reset for both cloth
    and volumetric (tet) soft bodies. The step rebaselines from the restored
    positions, so reset-then-step must reproduce a fresh step from the model
    defaults: the pre-reset perturbation must not leak through particle history.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    builder.add_cloth_grid(
        pos=(0.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=(0.0, 0.0, 0.0),
        dim_x=3,
        dim_y=3,
        cell_x=0.1,
        cell_y=0.1,
        mass=1.0,
        tri_ke=1.0e3,
        tri_ka=1.0e3,
        tri_kd=1.0e-1,
    )
    cloth_count = len(builder.particle_q)
    builder.add_soft_grid(
        pos=wp.vec3(1.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=0.1,
        cell_y=0.1,
        cell_z=0.1,
        density=100.0,
        k_mu=1.0e4,
        k_lambda=1.0e4,
        k_damp=1.0e-2,
    )
    builder.color()
    model = builder.finalize(device=device)
    test.assertGreater(cloth_count, 0)
    test.assertGreater(model.particle_count - cloth_count, 0)

    dt = 5.0e-3
    solver = newton.solvers.SolverVBD(model, iterations=5)
    model_q = model.particle_q.numpy()

    # Reference: one step from the model defaults.
    ref_in = model.state()
    ref_out = model.state()
    solver.step(ref_in, ref_out, None, None, dt)
    ref_q = ref_out.particle_q.numpy()
    test.assertTrue(np.all(np.isfinite(ref_q)))

    # Trial: perturb far away, reset back to the defaults, then step. Reusing the
    # same solver means its particle history holds the reference step; matching it
    # proves the reset-then-step rebaselines from the restored state.
    trial_in = model.state()
    trial_out = model.state()
    moved = model_q.copy()
    moved[:, 0] += 3.0
    trial_in.particle_q.assign(moved)
    trial_in.particle_qd.assign(np.full_like(model.particle_qd.numpy(), 1.0))
    solver.reset(trial_in)
    np.testing.assert_allclose(trial_in.particle_q.numpy(), model_q)
    solver.step(trial_in, trial_out, None, None, dt)
    trial_q = trial_out.particle_q.numpy()

    test.assertTrue(np.all(np.isfinite(trial_q)))
    np.testing.assert_allclose(trial_q, ref_q, rtol=1.0e-5, atol=1.0e-6)
    # Both deformable types advanced under gravity (not a trivial no-op).
    test.assertTrue(np.any(np.abs(ref_q[:cloth_count] - model_q[:cloth_count]) > 1.0e-8))
    test.assertTrue(np.any(np.abs(ref_q[cloth_count:] - model_q[cloth_count:]) > 1.0e-8))


def _rigid_reset_replays_captured_step(test, device):
    """A reset issued after capture is consumed by the existing step graph."""
    template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = template.add_body(mass=1.0, is_kinematic=True)
    template.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_world(template)
    builder.add_world(template, xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()))
    builder.color()
    model = builder.finalize(device=device)

    np.testing.assert_array_equal(model.body_world.numpy(), [0, 1])

    solver = newton.solvers.SolverVBD(model, iterations=0, rigid_compliant_alm=True)
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    dt = 1.0e-2

    # Finish lazy initialization and consume the constructor's initial baseline
    # before capturing the fixed state-buffer bindings used below.
    solver.step(state_in, state_out, control, None, dt)
    wp.synchronize_device(device)

    with wp.ScopedCapture(device=device) as capture:
        solver.step(state_in, state_out, control, None, dt)
    graph = capture.graph
    test.assertIsNotNone(graph)

    # reset() runs after capture. Its device-side mask write must be visible when
    # replaying the graph, while post-reset pose preparation remains authoritative.
    world_mask = wp.array([True, False, False], dtype=wp.bool, device=device)
    solver.reset(state_in, world_mask=world_mask, flags=0)
    reset_q = model.body_q.numpy()
    reset_q[:, 0] += 1.0
    state_in.body_q.assign(reset_q)
    state_in.body_qd.zero_()

    wp.capture_launch(graph)

    np.testing.assert_allclose(state_out.body_q.numpy(), reset_q, atol=1.0e-6)
    expected_qd = np.zeros_like(model.body_qd.numpy())
    expected_qd[1, 0] = 1.0 / dt
    np.testing.assert_allclose(state_out.body_qd.numpy(), expected_qd, rtol=1.0e-5, atol=1.0e-3)

    # The captured clear consumes reset intent once. A second replay of the same
    # graph must finite-difference an ordinary pose edit for both worlds.
    delta = 0.25
    next_q = reset_q.copy()
    next_q[:, 0] += delta
    state_in.body_q.assign(next_q)
    state_in.body_qd.zero_()

    wp.capture_launch(graph)

    np.testing.assert_allclose(state_out.body_q.numpy(), next_q, atol=1.0e-6)
    expected_qd[:, 0] = delta / dt
    np.testing.assert_allclose(state_out.body_qd.numpy(), expected_qd, rtol=1.0e-5, atol=1.0e-3)


def _rigid_contact_reset_lifecycle(test, device):
    """Verify legacy contact-history reset and matching provenance."""
    cfg = newton.ModelBuilder.ShapeConfig(ke=100.0, kd=0.0, mu=0.5)
    template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = template.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
        mass=1.0,
        is_kinematic=True,
    )
    template.add_shape_sphere(body, radius=0.1, cfg=cfg)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_ground_plane(cfg=cfg)
    builder.add_world(template)
    builder.add_world(template, xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()))
    builder.color()
    model = builder.finalize(device=device)
    reset_mask = wp.array([True, False, False], dtype=wp.bool, device=device)
    dt = 1.0e-2

    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
    contacts = pipeline.contacts()
    # Legacy AVBD path: fixed-k (no ramping) so cold vs warm is proven by the dual
    # alone; contact_alpha=gamma=1 disable the per-step lambda decay so a seeded
    # dual survives a step unchanged. Compliant ALM uses a different retention /
    # rho live gate, so this contract is legacy-only.
    solver = newton.solvers.SolverVBD(
        model,
        iterations=0,
        rigid_compliant_alm=False,
        rigid_contact_history=True,
        rigid_avbd_contact_alpha=1.0,
        rigid_avbd_gamma=1.0,
    )
    state_in = model.state()
    state_out = model.state()

    def advance(step_contacts):
        nonlocal state_in, state_out
        solver.step(state_in, state_out, None, step_contacts, dt)
        state_in, state_out = state_out, state_in

    shape_body = model.shape_body.numpy()
    body_world = model.body_world.numpy()

    def row_worlds():
        # Each contact pairs a world-local sphere with the global ground plane, so
        # exactly one endpoint carries the owning body/world. Both worlds must be
        # represented so neither assertion below runs on an empty slice.
        n = int(contacts.rigid_contact_count.numpy()[0])
        test.assertGreater(n, 0)
        s0 = contacts.rigid_contact_shape0.numpy()[:n]
        s1 = contacts.rigid_contact_shape1.numpy()[:n]
        rw = np.empty(n, dtype=np.int32)
        for i, (a, b) in enumerate(zip(s0, s1, strict=True)):
            bodies = [bd for bd in (shape_body[a], shape_body[b]) if bd >= 0]
            test.assertEqual(len(bodies), 1)
            rw[i] = body_world[bodies[0]]
        test.assertTrue(bool(np.any(rw == 0)) and bool(np.any(rw == 1)))
        return n, rw

    def seed_saved_dual(selected_mag, unselected_mag):
        # Address the saved dual by each row's match slot (not row index) so the
        # proof does not assume identity matching. Require the slots to be a valid,
        # unique, in-range set so a selected row's later zero can only come from
        # reset invalidation and never from an already-unmatched row. Seed lambda as
        # ``normal * magnitude`` with the matching saved normal so the warm restore
        # is an exact identity rotation.
        n, rw = row_worlds()
        capacity = solver._prev_contact_lambda.shape[0]
        slots = contacts.rigid_contact_match_index.numpy()[:n].astype(np.int64)
        test.assertTrue(np.all(slots >= 0))
        test.assertTrue(np.all(slots < capacity))
        test.assertEqual(len(np.unique(slots)), n)
        normal = contacts.rigid_contact_normal.numpy()[:n]
        saved_lambda = np.zeros((capacity, 3), dtype=np.float32)
        saved_normal = np.zeros((capacity, 3), dtype=np.float32)
        for i in range(n):
            slot = int(slots[i])
            mag = selected_mag if rw[i] == 0 else unselected_mag
            saved_lambda[slot] = normal[i] * mag
            saved_normal[slot] = normal[i]
        solver._prev_contact_lambda.assign(saved_lambda)
        solver._prev_contact_normal.assign(saved_normal)
        return n, rw, normal

    # Frame 1: a cold warm-up populates history from the step's snapshot.
    pipeline.collide(state_in, contacts)
    row_worlds()
    advance(contacts)

    # Reset world 0, then step without contacts: the intent has no fresh geometry
    # to act on and must survive the absent buffer.
    solver.reset(state_in, world_mask=reset_mask, flags=0)
    advance(None)

    # Frame 2: first fresh refresh after reset. Selected-world rows cold-start to a
    # zero dual despite a seeded warm value (proving the intent survived the
    # contactless step); the unselected world warm-restores its exact seed vector.
    pipeline.collide(state_in, contacts)
    n2, rw2, normal2 = seed_saved_dual(7.0, 8.0)
    advance(contacts)
    lam2 = solver.body_body_contact_lambda.numpy()[:n2]
    expected2 = np.where(rw2[:, None] == 0, 0.0, normal2 * 8.0)
    np.testing.assert_allclose(lam2, expected2, atol=1.0e-3)

    # Frame 3: the reset was one-shot, so both worlds warm-restore their exact seeds.
    pipeline.collide(state_in, contacts)
    n3, rw3, normal3 = seed_saved_dual(6.0, 9.0)
    advance(contacts)
    lam3 = solver.body_body_contact_lambda.numpy()[:n3]
    expected3 = np.where(rw3[:, None] == 0, normal3 * 6.0, normal3 * 9.0)
    np.testing.assert_allclose(lam3, expected3, atol=1.0e-3)

    # A disabled pipeline does not overwrite an allocated match-index array, so
    # SolverVBD must reject the buffer instead of restoring stale history.
    disabled_pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    disabled_pipeline.collide(state_in, contacts)
    test.assertEqual(contacts.contact_matching_mode, "disabled")
    with test.assertRaisesRegex(RuntimeError, "valid contact-matching provenance"):
        advance(contacts)


def _vbd_custom_attribute_registration_controls_dahl_defaults(test, device):
    """Verify zero Dahl defaults and rejection of the removed compatibility option."""
    del device

    builder = newton.ModelBuilder()
    newton.solvers.SolverVBD.register_custom_attributes(builder)
    test.assertIn("vbd:joint_is_hard", builder.custom_attributes)
    test.assertIn("vbd:dahl_eps_max", builder.custom_attributes)
    test.assertIn("vbd:dahl_tau", builder.custom_attributes)
    test.assertEqual(builder.custom_attributes["vbd:joint_is_hard"].default, 1)
    test.assertEqual(builder.custom_attributes["vbd:dahl_eps_max"].default, 0.0)
    test.assertEqual(builder.custom_attributes["vbd:dahl_tau"].default, 0.0)

    with test.assertRaisesRegex(TypeError, "dahl_defaults_enabled"):
        newton.solvers.SolverVBD.register_custom_attributes(newton.ModelBuilder(), dahl_defaults_enabled=True)


def _make_vbd_dahl_detection_model(device, *, dahl_eps_max=None, dahl_tau=None):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverVBD.register_custom_attributes(builder)

    parent = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
    child = builder.add_link(xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_box(parent, hx=0.1, hy=0.1, hz=0.1)
    builder.add_shape_box(child, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_rod(
        parent,
        child,
        parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
        bend_stiffness=1.0,
    )
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device=device)
    if dahl_eps_max is not None:
        model.vbd.dahl_eps_max.fill_(float(dahl_eps_max))
    if dahl_tau is not None:
        model.vbd.dahl_tau.fill_(float(dahl_tau))
    return model


def _vbd_dahl_detection_requires_positive_values(test, device):
    model = _make_vbd_dahl_detection_model(device)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        solver = newton.solvers.SolverVBD(model, rigid_compliant_alm=True)
    test.assertFalse(solver.enable_dahl_friction)

    model = _make_vbd_dahl_detection_model(device, dahl_eps_max=0.5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        solver = newton.solvers.SolverVBD(model, rigid_compliant_alm=True)
    test.assertFalse(solver.enable_dahl_friction)

    model = _make_vbd_dahl_detection_model(device, dahl_tau=1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        solver = newton.solvers.SolverVBD(model, rigid_compliant_alm=True)
    test.assertFalse(solver.enable_dahl_friction)

    model = _make_vbd_dahl_detection_model(device, dahl_eps_max=0.5, dahl_tau=1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        solver = newton.solvers.SolverVBD(model, rigid_compliant_alm=True)
    test.assertTrue(solver.enable_dahl_friction)


def _rigid_reset_cable_history(test, device):
    """Reset defers the cable tuple, then rebaselines it from the post-reset pose."""
    model = _make_vbd_dahl_detection_model(device, dahl_eps_max=0.5, dahl_tau=1.0)
    solver = newton.solvers.SolverVBD(model, iterations=0, rigid_compliant_alm=True)

    state_in = model.state()
    state_out = model.state()

    # Warm one step at pose A (the straight rest pose).
    solver.step(state_in, state_out, None, None, 1.0e-2)

    # Seed a distinct nonzero friction tuple so both the deferral and the later
    # rebaseline are observable (an immediate clear would zero these at reset).
    kappa_seed = solver.joint_kappa_prev.numpy()
    sigma_seed = solver.joint_sigma_prev.numpy()
    dkappa_seed = solver.joint_dkappa_prev.numpy()
    kappa_seed[0] = [0.15, -0.2, 0.25]
    sigma_seed[0] = [0.3, -0.4, 0.5]
    dkappa_seed[0] = [0.6, 0.7, -0.8]
    solver.joint_kappa_prev.assign(kappa_seed)
    solver.joint_sigma_prev.assign(sigma_seed)
    solver.joint_dkappa_prev.assign(dkappa_seed)

    # Reset at pose A defers the whole tuple: nothing changes until the next step.
    solver.reset(state_out, flags=0)
    np.testing.assert_allclose(solver.joint_kappa_prev.numpy()[0], [0.15, -0.2, 0.25], atol=1.0e-6)
    np.testing.assert_allclose(solver.joint_sigma_prev.numpy()[0], [0.3, -0.4, 0.5], atol=1.0e-6)
    np.testing.assert_allclose(solver.joint_dkappa_prev.numpy()[0], [0.6, 0.7, -0.8], atol=1.0e-6)

    # Pose editing happens after reset: rotate the child +1 radian about z.
    posed_q = state_out.body_q.numpy()
    posed_q[1, 3:] = [0.0, 0.0, math.sin(0.5), math.cos(0.5)]
    state_out.body_q.assign(posed_q)
    state_out.body_qd.zero_()

    # Poison the per-step Dahl stress output; the rebaseline step must recompute it.
    sigma_start_poison = solver.joint_sigma_start.numpy()
    sigma_start_poison[0] = [9.0, -8.0, 7.0]
    solver.joint_sigma_start.assign(sigma_start_poison)

    reset_state_out = model.state()
    solver.step(state_out, reset_state_out, None, None, 1.0e-2)

    # The step rebaselines curvature from the post-reset pose (a +1 rad z bend) and
    # clears stress, increment, and the recomputed per-step stress output.
    np.testing.assert_allclose(solver.joint_sigma_start.numpy()[0], 0.0, atol=1.0e-6)
    np.testing.assert_allclose(solver.joint_kappa_prev.numpy()[0], [0.0, 0.0, 1.0], atol=1.0e-3)
    np.testing.assert_allclose(solver.joint_sigma_prev.numpy()[0], 0.0, atol=1.0e-6)
    np.testing.assert_allclose(solver.joint_dkappa_prev.numpy()[0], 0.0, atol=1.0e-6)


def _rigid_contact_history_snapshot_copies_active_rows(test, device):
    """Snapshot writes solved state by active contact row and leaves inactive rows untouched."""
    with wp.ScopedDevice(device):
        contact_count = wp.array([2], dtype=int, device=device)
        normal = wp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        lam = wp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=wp.vec3, device=device)
        penalty = wp.array([10.0, 20.0, 30.0], dtype=float, device=device)

        prev_lambda = wp.zeros(3, dtype=wp.vec3, device=device)
        prev_penalty = wp.zeros(3, dtype=float, device=device)
        prev_normal = wp.zeros(3, dtype=wp.vec3, device=device)

        wp.launch(
            snapshot_body_body_contact_history,
            dim=3,
            inputs=[contact_count, normal, lam, penalty],
            outputs=[
                prev_lambda,
                prev_penalty,
                prev_normal,
            ],
            device=device,
        )

        np.testing.assert_allclose(prev_lambda.numpy()[:2], lam.numpy()[:2])
        np.testing.assert_allclose(prev_penalty.numpy()[:2], [10.0, 20.0])
        np.testing.assert_allclose(prev_normal.numpy()[:2], normal.numpy()[:2])
        np.testing.assert_allclose(prev_lambda.numpy()[2], [0.0, 0.0, 0.0])
        test.assertEqual(prev_penalty.numpy()[2], 0.0)


def _capsule_axial_spin_dissipates_via_friction(test, device, hard_contact=True, rigid_compliant_alm=False):
    """An axially-spinning capsule on its side must dissipate spin via Coulomb friction.

    Lays a capsule on the ground (long axis along world X), gives it pure angular
    velocity about that axis (no linear velocity), and checks that translational
    friction couples the spin to lateral motion: angular velocity decays and the
    capsule translates in -Y.
    """
    radius = 0.3
    half_height = 0.7
    omega_init = 5.0  # rad/s about world X (capsule's long axis)

    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e6
    builder.default_shape_cfg.kd = 0.0
    builder.default_shape_cfg.mu = 0.5
    builder.add_ground_plane()

    half = 0.5 * (math.pi / 2)
    q_side = wp.quat(0.0, math.sin(half), 0.0, math.cos(half))
    body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, radius), q=q_side))
    builder.add_shape_capsule(body, radius=radius, half_height=half_height)
    builder.color()

    with wp.ScopedDevice(device):
        model = builder.finalize()
        solver = newton.solvers.SolverVBD(
            model,
            iterations=10,
            rigid_compliant_alm=rigid_compliant_alm,
            rigid_contact_hard=hard_contact,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()

        init_qd = state_0.body_qd.numpy().copy()
        init_qd[0] = [0.0, 0.0, 0.0, omega_init, 0.0, 0.0]
        state_0.body_qd = wp.array(init_qd, dtype=wp.spatial_vector)

        sim_dt = 1.0e-3
        for _ in range(500):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

        qd = state_0.body_qd.numpy()[0]

    v_y = float(qd[1])
    omega_x = float(qd[3])

    test.assertLess(v_y, -0.1, f"capsule failed to translate under axial spin (v_y={v_y:.4f}, omega_x={omega_x:.4f})")
    test.assertLess(omega_x, 4.0, f"axial spin failed to dissipate (omega_x={omega_x:.4f}, v_y={v_y:.4f})")


def _yawed_cable_does_not_inject_energy(test, device, hard_contact=True, rigid_compliant_alm=False):
    """A yawed finite-radius cable settling on a plane must not gain kinetic energy.

    With zero friction there is no energy source, so kinetic energy must decay to rest. A
    non-conservative contact response would instead pump energy and blow the cable up
    (checked for both the hard and soft contact paths).
    """
    num_segments = 12
    segment_length = 0.5 / 19.0
    radius = 0.005
    yaw = math.radians(10.0)
    substeps = 8
    sim_dt = 1.0 / 100.0 / substeps
    num_frames = 200
    settle_frames = 50

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = 100.0
    cfg.mu = 0.0
    cfg.ke = 1.0e3
    cfg.kd = 1.0
    cfg.kf = 0.0
    builder.add_shape_plane(body=-1, cfg=cfg)

    length = num_segments * segment_length
    direction = wp.vec3(float(math.cos(yaw)), float(math.sin(yaw)), 0.0)
    center = wp.vec3(0.0, 0.0, radius + 0.05)
    start = center - 0.5 * length * direction
    rod = newton.Rod.create_straight(
        start=start,
        direction=direction,
        length=length,
        segment_count=num_segments,
        radius=radius,
    )
    bodies, _joints = builder.add_rod(
        rod=rod,
        cfg=cfg,
        stretch_stiffness=1.0e6,
        stretch_damping=1.0e-4,
        bend_stiffness=1.0e-4,
        bend_damping=1.0e-4,
        label="cable",
        body_frame_origin="com",
    )
    builder.color(balance_colors=False)

    with wp.ScopedDevice(device):
        model = builder.finalize()
        solver = newton.solvers.SolverVBD(
            model,
            iterations=20,
            rigid_compliant_alm=rigid_compliant_alm,
            rigid_contact_hard=hard_contact,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()

        masses = model.body_mass.numpy()
        inertias = model.body_inertia.numpy()
        body_idx = [int(b) for b in bodies]

        def kinetic_energy() -> float:
            qd = state_0.body_qd.numpy()
            ke = 0.0
            for b in body_idx:
                vel = qd[b, 0:3]
                omega = qd[b, 3:6]
                ke += 0.5 * float(masses[b]) * float(vel @ vel)
                ke += 0.5 * float(omega @ (inertias[b] @ omega))
            return ke

        max_ke_settled = 0.0
        for frame in range(num_frames):
            for _ in range(substeps):
                state_0.clear_forces()
                collision_pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, sim_dt)
                state_0, state_1 = state_1, state_0
            if frame >= settle_frames:
                max_ke_settled = max(max_ke_settled, kinetic_energy())

        final_ke = kinetic_energy()

    test.assertTrue(np.isfinite(final_ke), f"cable kinetic energy became non-finite ({final_ke})")
    test.assertLess(
        max_ke_settled,
        1.0e-3,
        f"yawed cable injected kinetic energy (max settled KE={max_ke_settled:.3e})",
    )


def _collect_rigid_contact_forces_reports_surface_points(test, device):
    """Rigid contact force reporting returns the same surface anchors used by the solve."""
    radius = 0.3

    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e6
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 0.5
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.95 * radius), q=wp.quat_identity()))
    builder.add_shape_sphere(body, radius=radius)
    builder.color()

    with wp.ScopedDevice(device):
        model = builder.finalize()
        model.set_gravity((0.0, 0.0, 0.0))
        solver = newton.solvers.SolverVBD(model, iterations=2, rigid_compliant_alm=True)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()

        collision_pipeline.collide(state_0, contacts)
        body_q_prev_snapshot = wp.clone(solver.body_q_prev)
        solver.step(state_0, state_1, control, contacts, 1.0e-3)

        c_b0, c_b1, c_p0w, c_p1w, _c_force, c_count = solver.collect_rigid_contact_forces(
            state_1.body_q, body_q_prev_snapshot, contacts, 1.0e-3
        )

        count = int(c_count.numpy()[0])
        body_q_np = state_1.body_q.numpy()
        body0_np = c_b0.numpy()
        body1_np = c_b1.numpy()
        reported0_np = c_p0w.numpy()
        reported1_np = c_p1w.numpy()
        point0_np = contacts.rigid_contact_point0.numpy()
        point1_np = contacts.rigid_contact_point1.numpy()
        offset0_np = contacts.rigid_contact_offset0.numpy()
        offset1_np = contacts.rigid_contact_offset1.numpy()

    test.assertGreater(count, 0, msg="Expected at least one sphere-ground rigid contact")
    max_offset = np.max(
        np.concatenate(
            [
                np.linalg.norm(offset0_np[:count], axis=1),
                np.linalg.norm(offset1_np[:count], axis=1),
            ]
        )
    )
    test.assertGreater(max_offset, 1.0e-4, msg="Test requires a contact with a non-zero surface offset")

    expected0 = np.empty((count, 3), dtype=np.float64)
    expected1 = np.empty((count, 3), dtype=np.float64)
    for i in range(count):
        expected0[i] = _transform_contact_point_np(body_q_np, int(body0_np[i]), point0_np[i] + offset0_np[i])
        expected1[i] = _transform_contact_point_np(body_q_np, int(body1_np[i]), point1_np[i] + offset1_np[i])

    np.testing.assert_allclose(reported0_np[:count], expected0, atol=1.0e-5)
    np.testing.assert_allclose(reported1_np[:count], expected1, atol=1.0e-5)


def _body_body_contact_lists_skip_static_kinematic(test, device):
    """An immovable body must not cause a spurious per-body list overflow."""
    buffer_pre_alloc = 1
    # Effective inverse mass folds together zero-mass and kinematic bodies.
    # Bodies 0 and 2 are dynamic; body 1 is immovable.
    body_inv_mass_effective = wp.array([1.0, 0.0, 1.0], dtype=float, device=device)
    shape_body = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    # Both contacts touch body 1, but each dynamic body has only one contact.
    rigid_contact_count = wp.array([2], dtype=int, device=device)
    rigid_contact_shape0 = wp.array([0, 1], dtype=int, device=device)
    rigid_contact_shape1 = wp.array([1, 2], dtype=int, device=device)

    body_contact_counts = wp.zeros(3, dtype=wp.int32, device=device)
    body_contact_indices = wp.full(3 * buffer_pre_alloc, -1, dtype=wp.int32, device=device)
    body_contact_overflow_max = wp.zeros(1, dtype=wp.int32, device=device)

    wp.launch(
        build_body_body_contact_lists,
        dim=2,
        inputs=[
            rigid_contact_count,
            rigid_contact_shape0,
            rigid_contact_shape1,
            shape_body,
            body_inv_mass_effective,
            buffer_pre_alloc,
        ],
        outputs=[body_contact_counts, body_contact_indices, body_contact_overflow_max],
        device=device,
    )

    np.testing.assert_array_equal(body_contact_counts.numpy(), np.array([1, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(body_contact_indices.numpy(), np.array([0, -1, 1], dtype=np.int32))
    test.assertEqual(int(body_contact_overflow_max.numpy()[0]), 0)


def _body_particle_contact_lists_skip_static_kinematic(test, device):
    """Immovable body-particle contacts must not cause a list overflow."""
    buffer_pre_alloc = 1
    # Body 0 is dynamic; body 1 represents a static or kinematic body.
    body_inv_mass_effective = wp.array([1.0, 0.0], dtype=float, device=device)
    shape_body = wp.array([0, 1], dtype=wp.int32, device=device)
    body_particle_contact_count = wp.array([3], dtype=int, device=device)
    body_particle_contact_shape = wp.array([0, 1, 1], dtype=int, device=device)

    counts = wp.zeros(2, dtype=wp.int32, device=device)
    indices = wp.full(2 * buffer_pre_alloc, -1, dtype=wp.int32, device=device)
    overflow_max = wp.zeros(1, dtype=wp.int32, device=device)

    wp.launch(
        build_body_particle_contact_lists,
        dim=3,
        inputs=[
            body_particle_contact_count,
            body_particle_contact_shape,
            shape_body,
            body_inv_mass_effective,
            buffer_pre_alloc,
        ],
        outputs=[counts, indices, overflow_max],
        device=device,
    )

    np.testing.assert_array_equal(counts.numpy(), np.array([1, 0], dtype=np.int32))
    np.testing.assert_array_equal(indices.numpy(), np.array([0, -1], dtype=np.int32))
    test.assertEqual(int(overflow_max.numpy()[0]), 0)


def _build_multi_world_particle_shape_scene(world_count, device, globals_kind="none"):
    """Build ``world_count`` replicas of a sub-world holding one shape and several free particles.

    ``globals_kind`` puts a global entity in both the head and the tail range: ``"shapes"`` adds
    global shapes, ``"particles"`` adds global particles, ``"none"`` adds neither. The two are never
    mixed: global particles times global shapes contributes a world-count-independent constant, which
    would break the exact 4x scaling the caller checks.
    """
    sub = newton.ModelBuilder()
    sub.add_shape_sphere(body=-1, radius=0.5)
    for i in range(8):
        sub.add_particle(pos=wp.vec3(0.1 * i, 0.0, 2.0), vel=wp.vec3(0.0, 0.0, 0.0), mass=1.0)

    def add_global(builder, z):
        if globals_kind == "shapes":
            builder.add_shape_sphere(body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()), radius=0.25)
        elif globals_kind == "particles":
            builder.add_particle(pos=wp.vec3(0.0, 0.0, z), vel=wp.vec3(0.0, 0.0, 0.0), mass=1.0)

    builder = newton.ModelBuilder()
    add_global(builder, 5.0)  # Global head range.
    for _ in range(world_count):
        builder.add_world(sub)
    add_global(builder, 6.0)  # Global tail range.
    builder.color()
    return builder.finalize(device=device)


def _soft_contact_presize_is_world_aware(test, device):
    """Verify SolverVBD pre-sizes body-particle buffers from world-compatible pairs, not every particle-shape pair."""
    for globals_kind in ("none", "shapes", "particles"):
        sizes = {}
        for world_count in (1, 4):
            model = _build_multi_world_particle_shape_scene(world_count, device, globals_kind=globals_kind)
            if globals_kind != "none":
                # Guard the scene: an empty head or tail range would silently weaken the check below.
                array = model.particle_world_start if globals_kind == "particles" else model.shape_world_start
                start = array.numpy()
                test.assertGreater(start[0], 0, f"{globals_kind=} {world_count=}")
                test.assertGreater(start[-1], start[-2], f"{globals_kind=} {world_count=}")
            # Constructed before any CollisionPipeline exists, as downstream users (Isaac Lab) do.
            solver = newton.solvers.SolverVBD(model)
            sizes[world_count] = solver.body_particle_contact_penalty_k.shape[0]
            test.assertEqual(
                sizes[world_count],
                newton.CollisionPipeline(model, broad_phase="nxn").soft_contact_pair_count,
                f"{globals_kind=} {world_count=}",
            )
        test.assertEqual(sizes[4], 4 * sizes[1], f"{globals_kind=}")


def _two_particle_tile_solve_matches_legacy_bits(test, device):
    """Pack odd-sized, high-valence cloth work without changing any result bit."""
    ring_count = 20
    angles = np.arange(ring_count, dtype=np.float64) * (2.0 * np.pi / ring_count)
    vertices = [wp.vec3(0.0, 0.0, 0.0)]
    vertices.extend(wp.vec3(float(np.cos(a)), float(np.sin(a)), 0.0) for a in angles)
    indices = []
    for i in range(ring_count):
        indices.extend((0, i + 1, (i + 1) % ring_count + 1))

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=vertices,
        indices=indices,
        density=1.0,
        tri_ke=1.0e3,
        tri_ka=7.0e2,
        tri_kd=0.1,
        edge_ke=10.0,
        edge_kd=0.02,
    )
    builder.color()
    model = builder.finalize(device=device)

    test.assertEqual(model.particle_count % 2, 1)
    adjacency_offsets = model.soft_mesh_adjacency_device.v_adj_tris_offsets.numpy()
    test.assertGreater((int(adjacency_offsets[1]) - int(adjacency_offsets[0])) // 2, 16)

    rng = np.random.default_rng(123)
    q_rest = model.particle_q.numpy()
    q = q_rest + (rng.standard_normal(q_rest.shape) * 0.025).astype(np.float32)
    q_prev = q_rest + (rng.standard_normal(q_rest.shape) * 0.01).astype(np.float32)
    inertia = q + (rng.standard_normal(q.shape) * 0.003).astype(np.float32)
    particle_forces = (rng.standard_normal(q.shape) * 0.25).astype(np.float32)
    particle_hessians = np.empty((model.particle_count, 3, 3), dtype=np.float32)
    for i in range(model.particle_count):
        sample = (rng.standard_normal((3, 3)) * 0.02).astype(np.float32)
        particle_hessians[i] = sample @ sample.T + np.eye(3, dtype=np.float32) * 0.1
    initial_displacements = (rng.standard_normal(q.shape) * 0.001).astype(np.float32)

    # Exercise the inactive-particle branch in a real half-warp as well as the
    # padded second group in the final warp.
    mass = model.particle_mass.numpy()
    mass[5] = 0.0
    model.particle_mass.assign(mass)
    particle_ids = np.arange(model.particle_count, dtype=np.int32)
    rng.shuffle(particle_ids)

    ids_device = wp.array(particle_ids, dtype=wp.int32, device=device)
    q_prev_device = wp.array(q_prev, dtype=wp.vec3, device=device)
    q_device = wp.array(q, dtype=wp.vec3, device=device)
    inertia_device = wp.array(inertia, dtype=wp.vec3, device=device)
    forces_device = wp.array(particle_forces, dtype=wp.vec3, device=device)
    hessians_device = wp.array(particle_hessians, dtype=wp.mat33, device=device)
    legacy_displacements = wp.array(initial_displacements, dtype=wp.vec3, device=device)
    packed_displacements = wp.array(initial_displacements, dtype=wp.vec3, device=device)

    common_inputs = [
        0.01,
        ids_device,
        q_prev_device,
        q_device,
        model.particle_mass,
        inertia_device,
        model.particle_flags,
        model.tri_indices,
        model.tri_poses,
        model.tri_materials,
        model.tri_areas,
        model.edge_indices,
        model.edge_rest_angle,
        model.edge_rest_length,
        model.edge_bending_properties,
    ]
    wp.launch(
        make_solve_elasticity_tile(True, True, False),
        dim=model.particle_count * TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
        block_dim=TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
        inputs=[
            *common_inputs,
            model.tet_indices,
            model.tet_poses,
            model.tet_materials,
            model.soft_mesh_adjacency_device,
            forces_device,
            hessians_device,
        ],
        outputs=[legacy_displacements],
        device=device,
    )
    wp.launch(
        make_solve_elasticity_tile(True, False, True),
        dim=((model.particle_count + 1) // 2) * (2 * TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE),
        block_dim=2 * TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
        inputs=[
            *common_inputs,
            model.tet_indices,
            model.tet_poses,
            model.tet_materials,
            model.soft_mesh_adjacency_device,
            forces_device,
            hessians_device,
        ],
        outputs=[packed_displacements],
        device=device,
    )
    legacy_bits = legacy_displacements.numpy().view(np.uint32)
    packed_bits = packed_displacements.numpy().view(np.uint32)
    np.testing.assert_array_equal(packed_bits, legacy_bits)


def _tet_only_elasticity_eligibility_matches_active_materials(test, device):
    """Enable tet-only elasticity only when triangle and edge stiffness are inactive."""
    vertices = [
        wp.vec3(0.0, 0.0, 0.0),
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(0.0, 1.0, 0.0),
        wp.vec3(0.0, 0.0, 1.0),
    ]
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_soft_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=vertices,
        indices=[0, 1, 2, 3],
        density=1.0,
        k_mu=1.0e3,
        k_lambda=2.0e3,
        k_damp=0.0,
    )
    builder.color()
    model = builder.finalize(device=device)

    test.assertGreater(model.tri_count, 0)
    test.assertGreater(model.edge_count, 0)
    test.assertTrue(_is_tet_only_elasticity_model(model))

    tri_materials = model.tri_materials.numpy()
    edge_properties = model.edge_bending_properties.numpy()
    test.assertFalse(np.any(tri_materials[:, :2] > 0.0))
    test.assertFalse(np.any(edge_properties[:, 0] > 0.0))

    # Damping alone is inactive under the full kernel's existing material guards.
    tri_materials[:, 2] = 1.0
    edge_properties[:, 1] = 1.0
    model.tri_materials.assign(tri_materials)
    model.edge_bending_properties.assign(edge_properties)
    test.assertTrue(_is_tet_only_elasticity_model(model))

    tri_materials[0, 0] = 1.0
    model.tri_materials.assign(tri_materials)
    test.assertFalse(_is_tet_only_elasticity_model(model))
    tri_materials[0, 0] = 0.0
    model.tri_materials.assign(tri_materials)

    edge_properties[0, 0] = 1.0
    model.edge_bending_properties.assign(edge_properties)
    test.assertFalse(_is_tet_only_elasticity_model(model))

    empty_model = newton.ModelBuilder().finalize(device=device)
    test.assertFalse(_is_tet_only_elasticity_model(empty_model))


def _tet_only_tile_solve_matches_legacy_bits(test, device):
    """Match the general tile solve bit for bit on a high-valence tet mesh."""
    # The central particle has 17 adjacent tets, exercising a second 16-lane batch.
    vertices = [wp.vec3(0.0, 0.0, 0.0)]
    indices = []
    for tet_index in range(17):
        scale = 1.0 + 0.01 * tet_index
        first = len(vertices)
        vertices.extend(
            (
                wp.vec3(scale, 0.0, 0.0),
                wp.vec3(0.0, scale, 0.0),
                wp.vec3(0.0, 0.0, scale),
            )
        )
        indices.extend((0, first, first + 1, first + 2))

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_soft_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=vertices,
        indices=indices,
        density=1.0,
        k_mu=1.0e3,
        k_lambda=2.0e3,
        k_damp=0.1,
    )
    builder.color()
    model = builder.finalize(device=device)
    test.assertTrue(_is_tet_only_elasticity_model(model))

    tet_offsets = model.soft_mesh_adjacency_device.v_adj_tets_offsets.numpy()
    test.assertEqual((int(tet_offsets[1]) - int(tet_offsets[0])) // 2, 17)

    rng = np.random.default_rng(404)
    q_rest = model.particle_q.numpy()
    q = q_rest + rng.normal(0.0, 0.01, q_rest.shape).astype(np.float32)
    q_prev = q_rest + rng.normal(0.0, 0.005, q_rest.shape).astype(np.float32)
    inertia = q + rng.normal(0.0, 0.001, q.shape).astype(np.float32)
    particle_forces = rng.normal(0.0, 0.1, q.shape).astype(np.float32)
    particle_hessians = np.empty((model.particle_count, 3, 3), dtype=np.float32)
    for particle_index in range(model.particle_count):
        sample = rng.normal(0.0, 0.01, (3, 3)).astype(np.float32)
        particle_hessians[particle_index] = sample @ sample.T + np.eye(3, dtype=np.float32) * 0.1
    initial_displacements = rng.normal(0.0, 0.001, q.shape).astype(np.float32)

    mass = model.particle_mass.numpy()
    mass[5] = 0.0
    model.particle_mass.assign(mass)
    particle_ids = np.arange(model.particle_count, dtype=np.int32)
    rng.shuffle(particle_ids)

    ids_device = wp.array(particle_ids, dtype=wp.int32, device=device)
    q_prev_device = wp.array(q_prev, dtype=wp.vec3, device=device)
    q_device = wp.array(q, dtype=wp.vec3, device=device)
    inertia_device = wp.array(inertia, dtype=wp.vec3, device=device)
    forces_device = wp.array(particle_forces, dtype=wp.vec3, device=device)
    hessians_device = wp.array(particle_hessians, dtype=wp.mat33, device=device)
    legacy_displacements = wp.array(initial_displacements, dtype=wp.vec3, device=device)
    specialized_displacements = wp.array(initial_displacements, dtype=wp.vec3, device=device)

    wp.launch(
        make_solve_elasticity_tile(True, True, False),
        dim=model.particle_count * TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
        block_dim=TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
        inputs=[
            0.01,
            ids_device,
            q_prev_device,
            q_device,
            model.particle_mass,
            inertia_device,
            model.particle_flags,
            model.tri_indices,
            model.tri_poses,
            model.tri_materials,
            model.tri_areas,
            model.edge_indices,
            model.edge_rest_angle,
            model.edge_rest_length,
            model.edge_bending_properties,
            model.tet_indices,
            model.tet_poses,
            model.tet_materials,
            model.soft_mesh_adjacency_device,
            forces_device,
            hessians_device,
        ],
        outputs=[legacy_displacements],
        device=device,
    )
    wp.launch(
        make_solve_elasticity_tile(False, True, False),
        dim=model.particle_count * TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
        block_dim=TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
        inputs=[
            0.01,
            ids_device,
            q_prev_device,
            q_device,
            model.particle_mass,
            inertia_device,
            model.particle_flags,
            model.tri_indices,
            model.tri_poses,
            model.tri_materials,
            model.tri_areas,
            model.edge_indices,
            model.edge_rest_angle,
            model.edge_rest_length,
            model.edge_bending_properties,
            model.tet_indices,
            model.tet_poses,
            model.tet_materials,
            model.soft_mesh_adjacency_device,
            forces_device,
            hessians_device,
        ],
        outputs=[specialized_displacements],
        device=device,
    )

    np.testing.assert_array_equal(
        specialized_displacements.numpy().view(np.uint32),
        legacy_displacements.numpy().view(np.uint32),
    )


class TestSolverVBD(unittest.TestCase):
    pass


add_function_test(
    TestSolverVBD,
    "test_body_body_contact_lists_skip_static_kinematic",
    _body_body_contact_lists_skip_static_kinematic,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_body_particle_contact_lists_skip_static_kinematic",
    _body_particle_contact_lists_skip_static_kinematic,
    devices=devices,
)
add_function_test(
    TestSolverVBD, "test_self_contact_barrier_c2_at_tau", test_self_contact_barrier_c2_at_tau, devices=devices
)
add_function_test(
    TestSolverVBD, "test_self_contact_barrier_c2_at_d_min", test_self_contact_barrier_c2_at_d_min, devices=devices
)
add_function_test(
    TestSolverVBD,
    "test_rigid_soft_contact_log_barrier_matches_particle",
    test_rigid_soft_contact_log_barrier_matches_particle,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_contact_history_restore_from_match_index",
    _rigid_contact_history_restore_from_match_index,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_contact_history_compliant_alm_tangent_warmstart",
    _rigid_contact_history_compliant_alm_tangent_warmstart,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_contact_history_soft_restores_penalty_only",
    _rigid_contact_history_soft_restores_penalty_only,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_joint_angular_rho_seed_uses_mean_mobility",
    _rigid_joint_angular_rho_seed_uses_mean_mobility,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_contact_tangent_support_uses_pair_mobility_eigenvalue",
    _rigid_contact_tangent_support_uses_pair_mobility_eigenvalue,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_compliant_sliding_contact_has_solve_metric",
    _rigid_compliant_sliding_contact_has_solve_metric,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_contact_structural_support_conditions_tangent_rho",
    _rigid_contact_structural_support_conditions_tangent_rho,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_contact_history_capture_requires_preallocation",
    _rigid_contact_history_capture_requires_preallocation,
    devices=cuda_devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_contact_stick_eps_are_deprecated",
    _rigid_contact_stick_eps_are_deprecated,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_compliant_alm_omission_warns_at_caller",
    _rigid_compliant_alm_omission_warns_at_caller,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_contact_dual_update_computes_lambda",
    _rigid_contact_dual_update_computes_lambda,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_contact_reset_ownership",
    _rigid_contact_reset_ownership,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_joint_angular_dual_projects_free_axis_lambda",
    _joint_angular_dual_projects_free_axis_lambda,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rod_soft_dual_slots_clear_preserved_lambda",
    _rod_soft_dual_slots_clear_preserved_lambda,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_joint_force_projection_filters_free_direction",
    _joint_force_projection_filters_free_direction,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_body_particle_contact_damping_is_absolute",
    _body_particle_contact_damping_is_absolute,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_body_particle_contact_damping_ignores_penalty_ramp",
    _body_particle_contact_damping_ignores_penalty_ramp,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_particle_contact_gather_matches_legacy",
    _particle_contact_gather_matches_legacy,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_particle_contact_gather_capture_replays_device_count",
    _particle_contact_gather_capture_replays_device_count,
    devices=cuda_devices,
)
add_function_test(
    TestSolverVBD,
    "test_particle_contact_gather_solver_step_dispatch_and_capture",
    _particle_contact_gather_solver_step_dispatch_and_capture,
    devices=cuda_devices,
)
add_function_test(
    TestSolverVBD,
    "test_particle_contact_gather_order_pinned",
    _particle_contact_gather_order_pinned,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_particle_contact_adjacency_follows_swapped_contacts",
    _particle_contact_adjacency_follows_swapped_contacts,
    devices=cuda_devices,
)
add_function_test(
    TestSolverVBD,
    "test_two_particle_tile_solve_matches_legacy_bits",
    _two_particle_tile_solve_matches_legacy_bits,
    devices=cuda_devices,
)
add_function_test(
    TestSolverVBD,
    "test_tet_only_elasticity_eligibility_matches_active_materials",
    _tet_only_elasticity_eligibility_matches_active_materials,
    devices=cpu_devices,
)
add_function_test(
    TestSolverVBD,
    "test_tet_only_tile_solve_matches_legacy_bits",
    _tet_only_tile_solve_matches_legacy_bits,
    devices=cuda_devices,
)
add_function_test(
    TestSolverVBD,
    "test_body_particle_dual_active_prefix_boundaries",
    _body_particle_dual_active_prefix_boundaries,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_body_particle_dual_capture_replays_device_count",
    _body_particle_dual_capture_replays_device_count,
    devices=cuda_devices,
)
add_function_test(
    TestSolverVBD,
    "test_body_body_contact_damping_ignores_penalty_ramp",
    _body_body_contact_damping_ignores_penalty_ramp,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_spring_damping_is_axial",
    _spring_damping_is_axial,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_bending_damping_handles_degenerate_anchor",
    _bending_damping_handles_degenerate_anchor,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_elastic_damping_ignores_rigid_motion",
    _elastic_damping_ignores_rigid_motion,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_contact_damping_ignores_rigid_motion",
    _contact_damping_ignores_rigid_motion,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_self_contact_damping_uses_relative_gap_rate",
    _self_contact_damping_uses_relative_gap_rate,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_d6_fully_free_structural_slots_are_inactive",
    _d6_fully_free_structural_slots_are_inactive,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_compliant_drive_preserves_material_equilibrium",
    _rigid_compliant_drive_preserves_material_equilibrium,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_compliant_alm_validates_drive_limit_damping",
    _rigid_compliant_alm_validates_drive_limit_damping,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_compliant_alm_validates_contact_materials",
    _rigid_compliant_alm_validates_contact_materials,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_joint_hard_soft_deprecation_describes_legacy_behavior",
    _joint_hard_soft_deprecation_describes_legacy_behavior,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_velocity_drive_preserves_legacy_damping_and_adds_compliant_support",
    _rigid_velocity_drive_preserves_legacy_damping_and_adds_compliant_support,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_drive_ignores_disabled_limit_bounds",
    _rigid_drive_ignores_disabled_limit_bounds,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_compliant_limit_holds_under_load",
    _rigid_compliant_limit_holds_under_load,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_body_structural_k_refreshes_after_joint_enable_notification",
    _body_structural_k_refreshes_after_joint_enable_notification,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_reset_state_and_history",
    _rigid_reset_state_and_history,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_reset_masked_rigid_and_soft",
    _reset_masked_rigid_and_soft,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_soft_reset_particle_only_and_external",
    _soft_reset_particle_only_and_external,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_soft_reset_captured_graph_restores_particles",
    _soft_reset_captured_graph_restores_particles,
    devices=cuda_devices,
)
add_function_test(
    TestSolverVBD,
    "test_soft_reset_then_step_advances_cloth_and_tet",
    _soft_reset_then_step_advances_cloth_and_tet,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_reset_replays_captured_step",
    _rigid_reset_replays_captured_step,
    devices=cuda_devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_contact_reset_lifecycle",
    _rigid_contact_reset_lifecycle,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_vbd_custom_attribute_registration_controls_dahl_defaults",
    _vbd_custom_attribute_registration_controls_dahl_defaults,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_vbd_dahl_detection_requires_positive_values",
    _vbd_dahl_detection_requires_positive_values,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_reset_cable_history",
    _rigid_reset_cable_history,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_rigid_contact_history_snapshot_copies_active_rows",
    _rigid_contact_history_snapshot_copies_active_rows,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_capsule_axial_spin_dissipates_via_friction_hard",
    _capsule_axial_spin_dissipates_via_friction,
    devices=devices,
    hard_contact=True,
)
add_function_test(
    TestSolverVBD,
    "test_capsule_axial_spin_dissipates_via_friction_soft",
    _capsule_axial_spin_dissipates_via_friction,
    devices=devices,
    hard_contact=False,
)
add_function_test(
    TestSolverVBD,
    "test_capsule_axial_spin_dissipates_via_friction_alm",
    _capsule_axial_spin_dissipates_via_friction,
    devices=devices,
    rigid_compliant_alm=True,
)
add_function_test(
    TestSolverVBD,
    "test_yawed_cable_does_not_inject_energy_hard",
    _yawed_cable_does_not_inject_energy,
    devices=devices,
    hard_contact=True,
)
add_function_test(
    TestSolverVBD,
    "test_yawed_cable_does_not_inject_energy_soft",
    _yawed_cable_does_not_inject_energy,
    devices=devices,
    hard_contact=False,
)
add_function_test(
    TestSolverVBD,
    "test_yawed_cable_does_not_inject_energy_alm",
    _yawed_cable_does_not_inject_energy,
    devices=devices,
    rigid_compliant_alm=True,
)
add_function_test(
    TestSolverVBD,
    "test_collect_rigid_contact_forces_reports_surface_points",
    _collect_rigid_contact_forces_reports_surface_points,
    devices=devices,
)
add_function_test(
    TestSolverVBD,
    "test_soft_contact_presize_is_world_aware",
    _soft_contact_presize_is_world_aware,
    devices=devices,
)


def _build_edge_over_post(device):
    """One soft triangle whose v0-v1 edge spans across a narrow tall box ("post").

    All three vertices sit well outside the box's contact margin (so the legacy
    particle-vs-shape pass emits *nothing*: ``soft_contact_count[0] == 0``), while the
    edge interior and the face centroid dip ~0.03 below the box's top (+y) face. Only the
    full-surface EDGE/FACE passes can detect this, and only the new VBD section 2 can act
    on it. Gravity is disabled so the contact push-out is the only force.
    """
    builder = newton.ModelBuilder()
    builder.gravity = (0.0, 0.0, 0.0)

    # Narrow tall post centered at the origin: x,z in [-0.1, 0.1], top face at y = +0.5.
    builder.add_shape_box(
        body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), hx=0.1, hy=0.5, hz=0.1
    )

    # Triangle at y = 0.47 (0.03 below the top face). v0/v1 span the post in x; v2 reaches
    # out in +z. Every vertex is >= 0.3 outside the post in x or z -> outside any margin.
    v0 = builder.add_particle(wp.vec3(-0.4, 0.47, 0.0), wp.vec3(0.0), 0.1)
    v1 = builder.add_particle(wp.vec3(0.4, 0.47, 0.0), wp.vec3(0.0), 0.1)
    v2 = builder.add_particle(wp.vec3(0.0, 0.47, 0.4), wp.vec3(0.0), 0.1)
    builder.add_triangle(v0, v1, v2)

    builder.color()
    configure_sdf_for_collision_shapes(builder)
    model = builder.finalize(device=device)
    return model, (v0, v1, v2)


def test_edge_face_pushes_vertices_out(test, device):
    """A soft edge/face penetrating a rigid box pushes its triangle's vertices out.

    With section 2 absent the particle force stays zero (legacy count is 0, gravity off),
    so the vertices never move. With section 2 present the barycentric distribution drives
    v0 and v1 (the spanning edge) out of the box along the contact normal.

    The +y penetration depth is constant along the part of the edge inside the post, so the
    contact point is degenerate there and lands where the +y and +x exits are nearly
    equidistant (they differ by ~4e-5). Which face the contact resolves to is therefore
    decided by rounding and varies across devices, so assert the push along the emitted
    contact normal rather than along +y.
    """
    model, (v0, v1, _v2) = _build_edge_over_post(device)

    margin = 0.1
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_gap=margin, enable_rigid_soft_full_surface_contact=True
    )
    contacts = pipeline.contacts()
    state_in = model.state()
    state_out = model.state()

    pipeline.collide(state_in, contacts)

    total = int(contacts.soft_contact_count.numpy()[0])
    idx = contacts.soft_contact_indices.numpy()[:total]
    # Precondition: legacy particle pass found nothing; the edge/face passes did.
    test.assertEqual(int(np.sum(idx[:, 1] < 0)), 0, "vertices should be outside the legacy particle margin")
    test.assertGreater(total, 0, "edge/face contacts must be detected")

    # Every record lies on one flat face of the post, so they share a normal. Assert that
    # rather than letting the push check below depend silently on record 0.
    normals = contacts.soft_contact_normal.numpy()[:total]
    test.assertTrue(
        bool(np.allclose(normals, normals[0], atol=1.0e-6)), "edge/face records should share one face normal"
    )
    normal = normals[0]

    solver = newton.solvers.SolverVBD(model)

    q_before = state_in.particle_q.numpy().copy()
    solver.step(state_in, state_out, None, contacts, dt=1.0 / 60.0)
    q_after = state_out.particle_q.numpy()

    # The two vertices of the spanning edge are pushed out along the contact normal.
    for name, v in (("v0", v0), ("v1", v1)):
        push = float(np.dot(q_after[v] - q_before[v], normal))
        test.assertGreater(push, 1.0e-3, f"{name} should be pushed along the contact normal")


def _build_sphere_on_fixed_soft_triangle(device):
    """A dynamic sphere resting on a FIXED soft triangle via a soft FACE contact.

    The triangle's three vertices have mass 0 (kinematic -> VBD never moves them) and lie in
    the z=0 plane, spanning wider than the sphere. The sphere bottom starts just below z=0 so
    the triangle face penetrates immediately, and gravity (-z) pulls the sphere down. Every
    triangle vertex is well outside the sphere, so the legacy particle pass finds nothing:
    only the *body-side* reaction from the soft FACE contact can keep the sphere from falling
    through. A sphere (convex SDF, unambiguous radial normal) keeps the contact normal stable
    as the body moves, isolating the body-side reaction under test.
    """
    builder = newton.ModelBuilder()  # up_axis = Z, gravity = -9.81 along -Z

    v0 = builder.add_particle(wp.vec3(-0.3, -0.3, 0.0), wp.vec3(0.0), 0.0, radius=0.0)
    v1 = builder.add_particle(wp.vec3(0.3, -0.3, 0.0), wp.vec3(0.0), 0.0, radius=0.0)
    v2 = builder.add_particle(wp.vec3(0.0, 0.3, 0.0), wp.vec3(0.0), 0.0, radius=0.0)
    builder.add_triangle(v0, v1, v2)

    # Sphere bottom (z = center - radius) starts slightly below z=0 -> immediate penetration.
    inertia = wp.mat33(2.0e-3, 0.0, 0.0, 0.0, 2.0e-3, 0.0, 0.0, 0.0, 2.0e-3)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.095), wp.quat_identity()),
        mass=0.5,
        inertia=inertia,
        lock_inertia=True,
    )
    builder.add_shape_sphere(body=body, radius=0.1)

    builder.color()
    configure_sdf_for_collision_shapes(builder)
    model = builder.finalize(device=device)
    return model, body


def test_edge_face_reacts_on_rigid_body(test, device):
    """The body-side reaction from a soft FACE contact supports a falling rigid box (S-a).

    Without the body-side section the body gets no reaction and free-falls through the fixed
    triangle (~4.9 m over 1 s); with it, the body is held up near its initial height.
    """
    model, body = _build_sphere_on_fixed_soft_triangle(device)

    margin = 0.1
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_gap=margin, enable_rigid_soft_full_surface_contact=True
    )
    contacts = pipeline.contacts()
    state_in = model.state()
    state_out = model.state()

    pipeline.collide(state_in, contacts)
    total = int(contacts.soft_contact_count.numpy()[0])
    idx = contacts.soft_contact_indices.numpy()[:total]
    test.assertEqual(int(np.sum(idx[:, 1] < 0)), 0, "triangle vertices should be outside the legacy particle margin")
    test.assertGreater(total, 0, "a soft edge/face contact must be detected")

    solver = newton.solvers.SolverVBD(model, rigid_compliant_alm=True)
    dt = 1.0 / 60.0
    z_before = float(state_in.body_q.numpy()[body, 2])

    for _ in range(60):
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, dt)
        state_in, state_out = state_out, state_in

    z_after = float(state_in.body_q.numpy()[body, 2])
    test.assertGreater(z_after, z_before - 0.05, "box should be supported by the soft contact, not free-fall")


def test_edge_face_reacts_through_coupled_proxy(test, device):
    """Verify a detected face contact propagates through proxy coupling."""
    model, body = _build_sphere_on_fixed_soft_triangle(device)
    model.gravity.zero_()
    coupled = SolverCoupledProxy(
        model=model,
        entries=[
            SolverCoupledProxy.Entry(name="body", solver=newton.solvers.SolverSemiImplicit, bodies=[body]),
            SolverCoupledProxy.Entry(
                name="soft",
                solver=lambda view: newton.solvers.SolverVBD(view, iterations=1, rigid_compliant_alm=True),
                particles=list(range(model.particle_count)),
            ),
        ],
        coupling=SolverCoupledProxy.Config(
            proxies=[SolverCoupledProxy.Proxy(source="body", destination="soft", bodies=[body])],
            iterations=1,
        ),
    )
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_gap=0.1, enable_rigid_soft_full_surface_contact=True
    )
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()

    for step in range(2):
        pipeline.collide(state_in, contacts)
        if step == 0:
            total = int(contacts.soft_contact_count.numpy()[0])
            indices = contacts.soft_contact_indices.numpy()[:total]
            test.assertGreater(total, 0)
            test.assertTrue(np.all(indices[:, 1] >= 0), "only edge/face contacts should be detected")
        coupled.step(state_in, state_out, None, contacts, 1.0 / 60.0)
        state_in, state_out = state_out, state_in

    test.assertGreater(float(state_in.body_qd.numpy()[body, 2]), 0.0)


def _set_slot(arr, idx, value):
    a = arr.numpy()
    a[idx] = value
    arr.assign(a)


def _run_face_section2(device, shape_margin):
    """Build a single soft-FACE contact, seed the shared AVBD per-contact material via
    ``init_body_particle_contacts``, then run the production two-kernel sequence
    (``build_particle_body_contact_adjacency_active`` + ``gather_particle_body_contact_force_and_hessian``)
    with the given ``shape_margin`` array. The geometry gives a 0.05 penetration along +z; returns
    ``(forces, hessians, ke, bary, (p0, p1, p2))`` where ``ke`` is the mixed effective stiffness
    section 2 reads. All three vertices form one color group so one gather launch processes the
    whole triangle."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0.0), wp.quat_identity()), hx=1.0, hy=1.0, hz=1.0)
    p0 = builder.add_particle(wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0), 0.1, radius=0.0)
    p1 = builder.add_particle(wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0), 0.1, radius=0.0)
    p2 = builder.add_particle(wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0), 0.1, radius=0.0)
    builder.add_triangle(p0, p1, p2)
    configure_sdf_for_collision_shapes(builder)
    model = builder.finalize(device=device)

    smax = 8
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_gap=0.1, soft_contact_max=smax)
    contacts = pipeline.contacts()
    state = model.state()

    # One FACE record. Contact point x = 0.6 v0 + 0.3 v1 + 0.1 v2 = (0.3, 0.1, 0); put the
    # rigid point 0.05 above it along +z so penetration = -(dot(n, x - bx)) = 0.05 > 0.
    bary = [0.6, 0.3, 0.1]
    contacts.soft_contact_count.assign([1])  # single total soft-contact count
    _set_slot(contacts.soft_contact_indices, 0, [p0, p1, p2])  # unified face record (v0, v1, v2)
    _set_slot(contacts.soft_contact_barycentric, 0, bary)
    _set_slot(contacts.soft_contact_shape, 0, 0)
    _set_slot(contacts.soft_contact_body_pos, 0, [0.3, 0.1, 0.05])
    _set_slot(contacts.soft_contact_body_vel, 0, [0.0, 0.0, 0.0])
    _set_slot(contacts.soft_contact_normal, 0, [0.0, 0.0, 1.0])

    # Dummy single-entry body arrays (the record's shape is on the world, body = -1, so these
    # are never indexed) to avoid passing empty/None body state.
    body_q = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
    body_qd = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    body_com = wp.zeros(1, dtype=wp.vec3, device=device)
    forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=device)
    hessians = wp.zeros(model.particle_count, dtype=wp.mat33, device=device)

    # The edge/face path shares the AVBD per-contact machinery with the particle-vs-surface path:
    # init_body_particle_contacts pre-mixes the global soft material with the contacted shape's
    # material and seeds the penalty. Fixed-k (k_start < 0) seeds it at the mixed ke, reproducing
    # the fully-ramped stiffness section 2 reads at run time in a single launch.
    penalty_k = wp.zeros(smax, dtype=float, device=device)
    material_ke = wp.zeros(smax, dtype=float, device=device)
    material_kd = wp.zeros(smax, dtype=float, device=device)
    material_mu = wp.zeros(smax, dtype=float, device=device)
    wp.launch(
        init_body_particle_contacts,
        dim=smax,
        inputs=[
            contacts.soft_contact_count,
            contacts.soft_contact_shape,
            model.soft_contact_ke,
            model.soft_contact_kd,
            model.soft_contact_mu,
            model.shape_material_ke,
            model.shape_material_kd,
            model.shape_material_mu,
            -1.0,  # k_start < 0 -> fixed-k: penalty seeded at the mixed ke (no ramp)
        ],
        outputs=[penalty_k, material_kd, material_mu, material_ke],
        device=device,
    )

    # Launch the production gather kernel the way SolverVBD does: build the per-particle
    # incidence lists over the active prefix, then gather one color group holding all three
    # triangle vertices.
    contact_head = wp.full(model.particle_count, -1, dtype=int, device=device)
    contact_next = wp.empty(3 * smax, dtype=int, device=device)
    wp.launch(
        build_particle_body_contact_adjacency_active,
        dim=smax,
        inputs=[
            contacts.soft_contact_indices,
            contacts.soft_contact_count,
            smax,
            contact_head,
            contact_next,
        ],
        device=device,
    )
    color_group = wp.array([p0, p1, p2], dtype=wp.int32, device=device)
    wp.launch(
        gather_particle_body_contact_force_and_hessian,
        dim=color_group.shape[0],
        block_dim=_PARTICLE_CONTACT_GATHER_BLOCK_DIM,
        inputs=[
            0.01,  # dt
            color_group,
            state.particle_q,  # pos_anchor == pos -> no damping / friction
            state.particle_q,
            1.0,  # friction_epsilon
            model.particle_radius,
            contacts.soft_contact_indices,
            contact_head,
            contact_next,
            penalty_k,
            material_kd,
            material_mu,
            model.shape_body,
            body_q,
            body_q,
            body_qd,
            body_com,
            contacts.soft_contact_shape,
            contacts.soft_contact_body_pos,
            contacts.soft_contact_body_vel,
            contacts.soft_contact_normal,
            shape_margin,
            contacts.soft_contact_barycentric,
        ],
        outputs=[forces, hessians],
        device=device,
    )
    # Section 2 reads the same per-contact AVBD stiffness the particle path uses; with fixed-k init
    # that equals the mixed ke (arithmetic mean of the global soft ke and the shape's ke). Return it
    # so callers assert against the effective stiffness.
    mixed_ke = float(penalty_k.numpy()[0])
    return forces.numpy(), hessians.numpy(), mixed_ke, bary, (p0, p1, p2)


def test_barycentric_force_distribution(test, device):
    """Section 2 distributes a contact at x = sum_i bary_i*v_i as bary_i*F and bary_i^2*H.

    A single FACE record with an asymmetric barycentric weight isolates the distribution math:
    the per-vertex force must scale with bary_i and the per-vertex Hessian block with bary_i^2.
    """
    f, h, ke, bary, (p0, p1, p2) = _run_face_section2(device, wp.zeros(0, dtype=float, device=device))
    single_force = np.array([0.0, 0.0, 0.05 * ke])  # F = n * penetration * ke

    for i, vi in enumerate([p0, p1, p2]):
        np.testing.assert_allclose(f[vi], bary[i] * single_force, rtol=2e-4, atol=1e-4)
        # Hessian block = bary_i^2 * ke * outer(n, n); only the zz entry is non-zero.
        np.testing.assert_allclose(h[vi][2, 2], bary[i] ** 2 * ke, rtol=2e-4, atol=1e-4)
    # The distributed force sums back to the single-point force (sum of bary == 1).
    np.testing.assert_allclose(f[p0] + f[p1] + f[p2], single_force, rtol=2e-4, atol=1e-4)


def test_edge_face_uses_shape_margin(test, device):
    """A per-shape contact margin (#2994) widens the edge/face penetration by ``margin``.

    Same single-FACE scene; the geometric penetration is 0.05. With ``shape_margin = 0`` the
    total force is ke*0.05; with ``shape_margin = m`` for the contacted shape it is ke*(0.05+m).
    """
    m = 0.02
    # Both runs use a 1-entry per-shape array so only the margin *value* differs (not the
    # array-shape contract). test_barycentric_force_distribution covers the empty-array guard.
    f0, _, ke, _, verts = _run_face_section2(device, wp.array([0.0], dtype=float, device=device))
    fm, _, _, _, _ = _run_face_section2(device, wp.array([m], dtype=float, device=device))  # shape 0 margin
    verts = list(verts)
    np.testing.assert_allclose(f0[verts].sum(axis=0), [0.0, 0.0, 0.05 * ke], rtol=2e-4, atol=1e-4)
    np.testing.assert_allclose(fm[verts].sum(axis=0), [0.0, 0.0, (0.05 + m) * ke], rtol=2e-4, atol=1e-4)


def test_edge_face_mixes_shape_material(test, device):
    """Section 2 mixes the global soft material with the contacted shape's material (ke/kd arithmetic
    mean, mu geometric mean), so per-shape tuning (grippy fingers, low-friction table) reaches
    edge/face contacts. Regression guard: the path previously used only the global soft_contact_*.
    """
    f, _h, mixed_ke, _bary, verts = _run_face_section2(device, wp.array([0.0], dtype=float, device=device))
    fz = float(f[list(verts)].sum(axis=0)[2])
    # The normal force uses the *mixed* stiffness over the 0.05 penetration.
    np.testing.assert_allclose(fz, mixed_ke * 0.05, rtol=2e-4, atol=1e-4)

    # Precondition + regression guard: the box (shape 0) carries the default ShapeConfig.ke, distinct
    # from the global soft_contact_ke, so the mix is observable and differs from a global-only result.
    builder = newton.ModelBuilder()
    builder.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0.0), wp.quat_identity()), hx=1.0, hy=1.0, hz=1.0)
    m = builder.finalize(device=device)
    global_ke = float(m.soft_contact_ke)
    shape_ke = float(m.shape_material_ke.numpy()[0])
    test.assertNotAlmostEqual(shape_ke, global_ke)
    np.testing.assert_allclose(mixed_ke, 0.5 * (global_ke + shape_ke), rtol=1e-6)
    test.assertGreater(abs(fz - global_ke * 0.05), 1e-3, "edge/face force must use the mixed ke, not global-only")


def test_flag_off_is_inert(test, device):
    """With the flag off the edge/face passes produce nothing and section 2 is a pure no-op.

    Reuses the edge-over-post scene (gravity disabled, every vertex outside the legacy
    margin). Flag on pushes the vertices out (test_edge_face_pushes_vertices_out); flag off
    must leave them exactly where they started -- the new path is inert and the legacy path
    is untouched, so flag-off behavior is unchanged.
    """
    model, _verts = _build_edge_over_post(device)
    # Flag OFF at construction: the buffer has no edge/face headroom and the passes never run.
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_gap=0.1, enable_rigid_soft_full_surface_contact=False
    )
    contacts = pipeline.contacts()
    state_in = model.state()
    state_out = model.state()

    pipeline.collide(state_in, contacts)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 0, "flag off => no soft contacts")

    q_before = state_in.particle_q.numpy().copy()
    solver = newton.solvers.SolverVBD(model)
    solver.step(state_in, state_out, None, contacts, dt=1.0 / 60.0)
    q_after = state_out.particle_q.numpy()

    np.testing.assert_allclose(q_after, q_before, atol=1.0e-6, err_msg="flag off must not move the soft body")


def test_full_surface_rejected_for_vbd_proxy_particles(test, device):
    """Reject full-surface contacts during VBD proxy-particle harvesting."""
    builder = newton.ModelBuilder()
    b = builder.add_body()
    builder.add_shape_box(body=b, hx=0.1, hy=0.1, hz=0.1)
    p0 = builder.add_particle(wp.vec3(-0.2, -0.2, 0.6), wp.vec3(0.0), 0.1, radius=0.0)
    p1 = builder.add_particle(wp.vec3(0.2, -0.2, 0.6), wp.vec3(0.0), 0.1, radius=0.0)
    p2 = builder.add_particle(wp.vec3(0.0, 0.2, 0.6), wp.vec3(0.0), 0.1, radius=0.0)
    builder.add_triangle(p0, p1, p2)
    builder.color()  # SolverVBD requires a particle coloring
    model = builder.finalize(device=device)

    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_gap=0.1, enable_rigid_soft_full_surface_contact=True
    )
    contacts = pipeline.contacts()  # capability marker set True
    solver = newton.solvers.SolverVBD(model, rigid_compliant_alm=True)

    harvest_kwargs = {
        "particle_qd_before": wp.zeros(model.particle_count, dtype=wp.vec3, device=device),
        "state": model.state(),
        "state_out": model.state(),
        "dt": 1.0 / 60.0,
    }
    with test.assertRaisesRegex(NotImplementedError, "proxy-particle"):
        solver.coupling_harvest_proxy_particle_forces(
            wp.array([0], dtype=int, device=device),
            wp.zeros(1, dtype=wp.vec3, device=device),
            contacts=contacts,
            **harvest_kwargs,
        )


class TestVBDFullSurfaceContact(unittest.TestCase):
    pass


add_function_test(
    TestVBDFullSurfaceContact,
    "test_edge_face_pushes_vertices_out",
    test_edge_face_pushes_vertices_out,
    devices=devices,
)
add_function_test(
    TestVBDFullSurfaceContact,
    "test_edge_face_reacts_on_rigid_body",
    test_edge_face_reacts_on_rigid_body,
    devices=devices,
)
add_function_test(
    TestVBDFullSurfaceContact,
    "test_edge_face_reacts_through_coupled_proxy",
    test_edge_face_reacts_through_coupled_proxy,
    devices=devices,
)
add_function_test(
    TestVBDFullSurfaceContact,
    "test_barycentric_force_distribution",
    test_barycentric_force_distribution,
    devices=devices,
)
add_function_test(
    TestVBDFullSurfaceContact,
    "test_edge_face_uses_shape_margin",
    test_edge_face_uses_shape_margin,
    devices=devices,
)
add_function_test(
    TestVBDFullSurfaceContact,
    "test_edge_face_mixes_shape_material",
    test_edge_face_mixes_shape_material,
    devices=devices,
)
add_function_test(
    TestVBDFullSurfaceContact,
    "test_flag_off_is_inert",
    test_flag_off_is_inert,
    devices=devices,
)
add_function_test(
    TestVBDFullSurfaceContact,
    "test_full_surface_rejected_for_vbd_proxy_particles",
    test_full_surface_rejected_for_vbd_proxy_particles,
    devices=devices,
)


# =====================================================================================
# DAT (Divide and Truncate) planar and rigid-body truncation
# =====================================================================================

_RIGID_SOFT_DAT_TEST_EPS = 1.0e-6


@wp.kernel
def _planar_truncation_probe(
    signed_distance: wp.array[float],
    normal_displacement: wp.array[float],
    gamma: float,
    minimum_signed_distance: float,
    t_out: wp.array[float],
):
    i = wp.tid()
    t_out[i] = planar_truncation_t(
        wp.vec3(0.0, 0.0, signed_distance[i]),
        wp.vec3(0.0, 0.0, normal_displacement[i]),
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(0.0),
        gamma,
        minimum_signed_distance,
    )


@wp.kernel
def _large_coordinate_epsilon_ablation_probe(
    origin: float,
    initial_gap: float,
    normal_displacement: float,
    separation_eps: wp.array[float],
    t_out: wp.array[float],
    accepted_gap_out: wp.array[float],
):
    """Apply the production plane-placement/truncation math at a large origin."""
    i = wp.tid()
    n = wp.vec3(0.0, 0.0, 1.0)
    rigid_support = wp.vec3(origin, origin, origin)
    soft_start = wp.vec3(origin, origin, origin + initial_gap)
    soft_displacement = wp.vec3(0.0, 0.0, normal_displacement)
    plane_point, _lmbd = place_dat_division_plane(
        n,
        rigid_support,
        initial_gap,
        wp.max(-normal_displacement, 0.0),
        0.0,
        separation_eps[i],
    )
    t = planar_truncation_t(
        soft_start,
        soft_displacement,
        n,
        plane_point,
        0.85,
        separation_eps[i],
    )
    accepted = soft_start + t * soft_displacement
    t_out[i] = t
    # This subtraction happens after ``accepted`` has been stored as a vec3 of
    # FP32 values, exposing whether its intended positive gap is representable.
    accepted_gap_out[i] = accepted[2] - origin


def test_rigid_soft_dat_scale_aware_epsilon_prevents_fp32_gap_collapse(test, device):
    """A fixed micrometer band can round a truncated endpoint onto distant geometry."""
    origin = np.float32(1000.0)
    adjacent = np.nextafter(origin, np.float32(np.inf))
    initial_gap = float(adjacent - origin)
    base_eps = _RIGID_SOFT_DAT_TEST_EPS
    scale_eps = max(base_eps, 4.0 * np.finfo(np.float32).eps * float(origin))

    t_out = wp.empty(2, dtype=float, device=device)
    accepted_gap_out = wp.empty(2, dtype=float, device=device)
    wp.launch(
        _large_coordinate_epsilon_ablation_probe,
        dim=2,
        inputs=[
            float(origin),
            initial_gap,
            -initial_gap,
            wp.array([base_eps, scale_eps], dtype=float, device=device),
        ],
        outputs=[t_out, accepted_gap_out],
        device=device,
    )

    fixed_t, scaled_t = t_out.numpy()
    fixed_gap, scaled_gap = accepted_gap_out.numpy()
    test.assertGreater(fixed_t, 0.0, "the fixed epsilon control must attempt a partial advance")
    test.assertEqual(fixed_gap, 0.0, "that partial endpoint must collapse onto the rigid support in FP32")
    test.assertEqual(scaled_t, 0.0, "a worsening move from inside the representable band must be halted")
    test.assertEqual(scaled_gap, initial_gap, "halting must retain the original one-ULP separation")


def test_planar_truncation_uses_endpoint_signs(test, device):
    """Small normal motion still truncates when the proposed endpoint changes side."""
    s0 = 3.797968e-6
    normal_displacement = -9.053657e-6
    signed_distances = wp.array([s0, s0, -1.0e-6, -1.0e-6], dtype=float, device=device)
    normal_displacements = wp.array([normal_displacement, -1.0e-6, -1.0e-6, 1.0e-6], dtype=float, device=device)
    t_out = wp.empty(4, dtype=float, device=device)
    wp.launch(
        _planar_truncation_probe,
        dim=4,
        inputs=[signed_distances, normal_displacements, 0.85, 0.0],
        outputs=[t_out],
        device=device,
    )
    crossing_t = s0 / -normal_displacement
    expected_crossing_t = 0.85 * crossing_t
    np.testing.assert_allclose(
        t_out.numpy(),
        [expected_crossing_t, 1.0, 0.0, 1.0],
        rtol=0.0,
        atol=1.0e-6,
    )


@wp.kernel
def _soft_self_dat_epsilon_probe(result: wp.array[float]):
    zero = wp.vec3(0.0)

    vertex = wp.vec3(0.0, 0.0, 8.0e-6)
    triangle_0 = wp.vec3(-1.0, -1.0, 0.0)
    triangle_1 = wp.vec3(1.0, -1.0, 0.0)
    triangle_2 = wp.vec3(0.0, 1.0, 0.0)
    vertex_displacement = wp.vec3(0.0, 0.0, -10.0e-6)
    vt_valid, vt_n, vt_d, vt_eps = create_vertex_triangle_division_plane_closest_pt(
        vertex,
        vertex_displacement,
        triangle_0,
        zero,
        triangle_1,
        zero,
        triangle_2,
        zero,
    )
    vt_t = float(-1.0)
    if vt_valid:
        vt_t = planar_truncation_t(vertex, vertex_displacement, vt_n, vt_d, 0.85, vt_eps)
    result[0] = wp.dot(vt_n, vertex - vt_d)
    result[1] = wp.dot(-vt_n, triangle_0 - vt_d)
    result[2] = wp.dot(vt_n, vertex + vt_t * vertex_displacement - triangle_0)
    result[3] = vt_t
    result[4] = vt_eps

    # A point already inside the positive-side half-band may recover freely,
    # but it may not move farther into the band.
    point_inside_band = vt_d + 0.5 * vt_eps * vt_n
    result[5] = planar_truncation_t(point_inside_band, -0.25 * vt_eps * vt_n, vt_n, vt_d, 0.85, vt_eps)
    result[6] = planar_truncation_t(point_inside_band, 0.25 * vt_eps * vt_n, vt_n, vt_d, 0.85, vt_eps)

    edge_0_a = wp.vec3(-1.0, 8.0e-6, 0.0)
    edge_0_b = wp.vec3(1.0, 8.0e-6, 0.0)
    edge_1_a = wp.vec3(-1.0, 0.0, 0.0)
    edge_1_b = wp.vec3(1.0, 0.0, 0.0)
    edge_0_displacement = wp.vec3(0.0, -10.0e-6, 0.0)
    ee_valid, ee_n, ee_d, ee_eps = create_edge_edge_division_plane_closest_pt(
        edge_0_a,
        edge_0_displacement,
        edge_0_b,
        edge_0_displacement,
        edge_1_a,
        zero,
        edge_1_b,
        zero,
    )
    ee_t = float(-1.0)
    if ee_valid:
        ee_t = planar_truncation_t(edge_0_a, edge_0_displacement, ee_n, ee_d, 0.85, ee_eps)
    result[7] = wp.dot(ee_n, edge_0_a - ee_d)
    result[8] = wp.dot(-ee_n, edge_1_a - ee_d)
    result[9] = wp.dot(ee_n, edge_0_a + ee_t * edge_0_displacement - edge_1_a)
    result[10] = ee_t
    result[11] = ee_eps


def test_soft_self_dat_uses_epsilon_separation(test, device):
    """VT and EE planes reserve epsilon, and inside-band motion cannot worsen separation."""
    result = wp.empty(12, dtype=float, device=device)
    wp.launch(_soft_self_dat_epsilon_probe, dim=1, outputs=[result], device=device)
    result = result.numpy()

    vt_positive_margin, vt_negative_margin, vt_accepted_gap, vt_t, vt_eps = result[:5]
    test.assertGreaterEqual(vt_positive_margin, 0.99 * vt_eps)
    test.assertGreaterEqual(vt_negative_margin, 0.99 * vt_eps)
    test.assertGreaterEqual(vt_accepted_gap, 2.0 * vt_eps)
    test.assertGreater(vt_t, 0.0)
    test.assertLess(vt_t, 1.0)
    test.assertEqual(result[5], 0.0)
    test.assertEqual(result[6], 1.0)

    ee_positive_margin, ee_negative_margin, ee_accepted_gap, ee_t, ee_eps = result[7:]
    test.assertGreaterEqual(ee_positive_margin, 0.99 * ee_eps)
    test.assertGreaterEqual(ee_negative_margin, 0.99 * ee_eps)
    test.assertGreaterEqual(ee_accepted_gap, 2.0 * ee_eps)
    test.assertGreater(ee_t, 0.0)
    test.assertLess(ee_t, 1.0)


def _run_soft_self_dat_truncation(
    device,
    positions,
    displacements,
    triangles,
    edges,
    vertex_triangle_pair=None,
    edge_edge_pair=None,
):
    """Run the Planar-DAT kernel for one explicitly prescribed primitive pair."""
    positions = np.asarray(positions, dtype=np.float32)
    displacements = np.asarray(displacements, dtype=np.float32)
    triangles = np.asarray(triangles, dtype=np.int32).reshape((-1, 3))
    edges = np.asarray(edges, dtype=np.int32).reshape((-1, 4))
    particle_count = len(positions)
    edge_count = len(edges)
    collision_info = build_tri_mesh_collision_info(
        particle_count=particle_count,
        tri_count=len(triangles),
        edge_count=edge_count,
        vertex_collision_buffer_pre_alloc=1,
        edge_collision_buffer_pre_alloc=1,
        device=device,
    )

    vertex_counts = np.zeros(particle_count, dtype=np.int32)
    if vertex_triangle_pair is not None:
        vertex_index, triangle_index = vertex_triangle_pair
        vertex_pairs = np.zeros(2 * particle_count, dtype=np.int32)
        vertex_pairs[2 * vertex_index : 2 * vertex_index + 2] = (vertex_index, triangle_index)
        collision_info.vertex_colliding_triangles.assign(vertex_pairs)
        vertex_counts[vertex_index] = 1
    collision_info.vertex_colliding_triangles_count.assign(vertex_counts)

    edge_counts = np.zeros(edge_count, dtype=np.int32)
    if edge_edge_pair is not None:
        first_edge, second_edge = edge_edge_pair
        edge_pairs = np.zeros(2 * edge_count, dtype=np.int32)
        edge_pairs[2 * first_edge : 2 * first_edge + 2] = (first_edge, second_edge)
        collision_info.edge_colliding_edges.assign(edge_pairs)
        edge_counts[first_edge] = 1
    collision_info.edge_colliding_edges_count.assign(edge_counts)

    truncation_t = wp.ones(particle_count, dtype=float, device=device)
    wp.launch(
        apply_planar_truncation_parallel_by_collision,
        dim=max(particle_count, edge_count) * NUM_THREADS_PER_COLLISION_PRIMITIVE,
        inputs=[
            wp.array(positions, dtype=wp.vec3, device=device),
            wp.array(displacements, dtype=wp.vec3, device=device),
            wp.array(triangles, dtype=wp.int32, ndim=2, device=device),
            wp.array(edges, dtype=wp.int32, ndim=2, device=device),
            wp.array([collision_info], dtype=TriMeshCollisionInfo, device=device),
            0.85,
        ],
        outputs=[truncation_t],
        device=device,
    )
    return truncation_t.numpy()


def test_soft_self_dat_truncates_complete_primitive_pairs(test, device):
    """The shared VT/EE separators constrain every vertex, and invalid pairs fail closed."""
    epsilon = _RIGID_SOFT_DAT_TEST_EPS

    with test.subTest(pair="moving VT vertex"):
        positions = np.array([[0.0, 0.0, 1.0], [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
        displacements = np.zeros_like(positions)
        displacements[0, 2] = -2.0
        actual = _run_soft_self_dat_truncation(
            device,
            positions,
            displacements,
            [[1, 2, 3]],
            [],
            vertex_triangle_pair=(0, 0),
        )
        expected = np.array([0.85 * (1.0 - 2.0 * epsilon) / 2.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-6)

    with test.subTest(pair="moving VT triangle"):
        positions = np.array([[0.0, 0.0, 1.0], [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
        displacements = np.zeros_like(positions)
        displacements[1:, 2] = 2.0
        actual = _run_soft_self_dat_truncation(
            device,
            positions,
            displacements,
            [[1, 2, 3]],
            [],
            vertex_triangle_pair=(0, 0),
        )
        moving_t = 0.85 * (1.0 - 2.0 * epsilon) / 2.0
        np.testing.assert_allclose(actual, [1.0, moving_t, moving_t, moving_t], rtol=0.0, atol=1.0e-6)

    with test.subTest(pair="moving EE edges"):
        positions = np.array(
            [
                [-1.0, 0.0, 0.5],
                [1.0, 0.0, 0.5],
                [0.0, 1.0, 2.0],
                [-1.0, 0.0, -0.5],
                [1.0, 0.0, -0.5],
                [0.0, -1.0, -2.0],
            ]
        )
        displacements = np.zeros_like(positions)
        displacements[:2, 2] = -1.0
        displacements[3:5, 2] = 1.0
        actual = _run_soft_self_dat_truncation(
            device,
            positions,
            displacements,
            [[0, 1, 2], [3, 4, 5]],
            [[-1, -1, 0, 1], [-1, -1, 3, 4]],
            edge_edge_pair=(0, 1),
        )
        moving_t = 0.85 * (0.5 - epsilon)
        np.testing.assert_allclose(
            actual,
            [moving_t, moving_t, 1.0, moving_t, moving_t, 1.0],
            rtol=0.0,
            atol=1.0e-6,
        )

    with test.subTest(pair="touching VT fails closed"):
        positions = np.array([[0.0, 0.0, 0.0], [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
        actual = _run_soft_self_dat_truncation(
            device,
            positions,
            np.zeros_like(positions),
            [[1, 2, 3]],
            [],
            vertex_triangle_pair=(0, 0),
        )
        np.testing.assert_array_equal(actual, np.zeros(4))

    with test.subTest(pair="intersecting EE fails closed"):
        positions = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
        actual = _run_soft_self_dat_truncation(
            device,
            positions,
            np.zeros_like(positions),
            [],
            [[-1, -1, 0, 1], [-1, -1, 2, 3]],
            edge_edge_pair=(0, 1),
        )
        np.testing.assert_array_equal(actual, np.zeros(4))


@wp.kernel
def _rigid_trajectory_truncation_probe(
    n: wp.vec3,
    d: wp.vec3,
    c0: wp.vec3,
    dx: wp.vec3,
    axis: wp.vec3,
    angle: float,
    offset0: wp.vec3,
    gamma_r: float,
    use_interval_arithmetic: bool,
    trajectory_samples: int,
    t_out: wp.array[float],
):
    t_out[0] = rigid_trajectory_truncation_t(
        n, d, c0, dx, axis, angle, offset0, gamma_r, 1.0e-3, use_interval_arithmetic, trajectory_samples
    )


@wp.kernel
def _rigid_point_trajectory_probe(
    t: float,
    c0: wp.vec3,
    dx: wp.vec3,
    axis: wp.vec3,
    angle: float,
    offset0: wp.vec3,
    point_out: wp.array[wp.vec3],
):
    point_out[0] = rigid_point_trajectory(t, c0, dx, axis, angle, offset0)


def _probe_rigid_point_trajectory(device, t, c0, dx, axis, angle, offset0):
    point_out = wp.empty(1, dtype=wp.vec3, device=device)
    wp.launch(
        _rigid_point_trajectory_probe,
        dim=1,
        inputs=[t, wp.vec3(*c0), wp.vec3(*dx), wp.vec3(*axis), angle, wp.vec3(*offset0)],
        outputs=[point_out],
        device=device,
    )
    return point_out.numpy()[0]


def _probe_trajectory_truncation(
    device,
    n,
    d,
    c0,
    dx,
    axis,
    angle,
    offset0,
    gamma_r,
    use_interval_arithmetic=False,
    trajectory_samples=8,
):
    t_out = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        _rigid_trajectory_truncation_probe,
        dim=1,
        inputs=[
            wp.vec3(*n),
            wp.vec3(*d),
            wp.vec3(*c0),
            wp.vec3(*dx),
            wp.vec3(*axis),
            angle,
            wp.vec3(*offset0),
            gamma_r,
            use_interval_arithmetic,
            trajectory_samples,
        ],
        outputs=[t_out],
        device=device,
    )
    return float(t_out.numpy()[0])


def _probe_rigid_dat_returning_sdf_arc(
    device,
    use_interval_arithmetic,
    *,
    rigid_anchor=None,
    plane_normal=(1.0, 0.0, 0.0),
    plane_point=None,
    rotation_angle=math.pi,
    trajectory_samples=8,
):
    """Probe a rotating rigid SDF point that crosses and returns through a plane."""

    if rigid_anchor is None:
        phi = math.pi / 16.0
        rigid_anchor = (math.cos(phi), -math.sin(phi), 0.0)
    if plane_point is None:
        plane_point = (0.5 * (0.99 + rigid_anchor[0]), 0.0, 0.0)
    return _probe_trajectory_truncation(
        device,
        plane_normal,
        plane_point,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        rotation_angle,
        rigid_anchor,
        1.0,
        use_interval_arithmetic,
        trajectory_samples,
    )


def test_rigid_dat_trajectory_truncation(test, device):
    """Compare Stage 1 sampling+bisection with optional Stage 2 interval verification."""
    # The production trajectory uses Rodrigues directly for every angle; there
    # is no separate first-order branch at small angles.
    small_angle = 5.0e-8
    small_angle_point = _probe_rigid_point_trajectory(
        device, 1.0, (0, 0, 0), (0, 0, 0), (0, 0, 1), small_angle, (1, 0, 0)
    )
    test.assertTrue(np.allclose(small_angle_point, (math.cos(small_angle), math.sin(small_angle), 0.0), atol=1e-12))
    identity_point = _probe_rigid_point_trajectory(device, 1.0, (0, 0, 0), (0, 0, 0), (0, 0, 0), 0.0, (1, 2, 3))
    test.assertTrue(np.array_equal(identity_point, (1.0, 2.0, 3.0)))

    # A point already behind its assigned plane is outside Algorithm 1's
    # precondition. Both modes use the same explicit recovery policy: strict
    # endpoint improvement is accepted, while further violation is blocked.
    for use_interval_arithmetic in (False, True):
        recovery_t = _probe_trajectory_truncation(
            device,
            (0, 1, 0),
            (0, 0, 0),
            (0, 0, 0),
            (0, -0.2, 0),
            (0, 0, 1),
            0.0,
            (0, 0.1, 0),
            0.85,
            use_interval_arithmetic,
        )
        worsening_t = _probe_trajectory_truncation(
            device,
            (0, 1, 0),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0.1, 0),
            (0, 0, 1),
            0.0,
            (0, 0.1, 0),
            0.85,
            use_interval_arithmetic,
        )
        test.assertEqual(recovery_t, 1.0)
        test.assertEqual(worsening_t, 0.0)

        # Endpoint recovery is insufficient for a curved rigid trajectory. The
        # point starts behind its negative-side boundary, first moves farther
        # into the forbidden band, and only then ends on the safe side. Any
        # penetration-directed portion must halt the complete rigid update.
        returning_recovery_t = _probe_trajectory_truncation(
            device,
            (1.0, 0.0, 0.0),
            (0.8, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            2.0 * math.pi / 3.0,
            (math.cos(-math.pi / 6.0), math.sin(-math.pi / 6.0), 0.0),
            0.85,
            use_interval_arithmetic,
        )
        test.assertEqual(returning_recovery_t, 0.0)

        # The true normal derivative of this quarter-circle recovery is zero
        # at t=0 and strictly negative afterward. Directed interval rounding
        # may return a positive subnormal upper bound for that exact zero; it
        # must not freeze an otherwise monotone separating trajectory.
        monotone_recovery_t = _probe_trajectory_truncation(
            device,
            (1.0, 0.0, 0.0),
            (0.8, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            math.pi / 2.0,
            (1.0, 0.0, 0.0),
            0.85,
            use_interval_arithmetic,
        )
        test.assertEqual(monotone_recovery_t, 1.0)

    # Pure rotation: point at radius 1 rotating pi/2 about z crosses plane y=0.5 at t=1/3
    # (sin(t*pi/2) = 0.5). gamma_r=1 leaves only the fixed 1e-3 safety backoff.
    t = _probe_trajectory_truncation(
        device, (0, 1, 0), (0, 0.5, 0), (0, 0, 0), (0, 0, 0), (0, 0, 1), math.pi / 2, (1, 0, 0), 1.0
    )
    test.assertAlmostEqual(t, 1.0 / 3.0 - 1e-3, delta=2e-3)

    # Same rotation with the opposite handedness moves the point away: no truncation.
    t = _probe_trajectory_truncation(
        device, (0, 1, 0), (0, 0.5, 0), (0, 0, 0), (0, 0, 0), (0, 0, -1), math.pi / 2, (1, 0, 0), 1.0
    )
    test.assertEqual(t, 1.0)

    # Pure translation degenerates to the straight-ray case: crossing at t=0.5.
    t = _probe_trajectory_truncation(
        device, (0, 1, 0), (0, 0.5, 0), (0, 0, 0), (0, 1, 0), (0, 0, 1), 0.0, (0, 0, 0), 0.85
    )
    test.assertAlmostEqual(t, 0.425, delta=2e-3)

    # The same cross-and-return arc starting exactly on the plane must stall;
    # checking only the end point would incorrectly release it.
    phi = math.pi / 16.0
    t_interval = _probe_trajectory_truncation(
        device,
        (1, 0, 0),
        (math.cos(phi), 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 1),
        math.pi,
        (math.cos(-phi), math.sin(-phi), 0),
        1.0,
        True,
    )
    test.assertEqual(t_interval, 0.0, "interval arithmetic must stall a boundary trajectory that returns")

    # Screw motion that stays on the safe side of the plane.
    t = _probe_trajectory_truncation(
        device, (0, 1, 0), (0, 0.5, 0), (0, 0, 0), (0.3, -0.2, 0), (0, 0, 1), 0.3, (0.2, -0.3, 0), 0.85
    )
    test.assertEqual(t, 1.0)

    # A valid start exactly on the plane blocks an approaching update...
    t = _probe_trajectory_truncation(
        device, (0, 1, 0), (0, 0.5, 0), (0, 0, 0), (0, 0.1, 0), (0, 0, 1), 0.0, (0, 0.5, 0), 0.85
    )
    test.assertEqual(t, 0.0)

    # ...while a separating update from the same boundary point stays free.
    t = _probe_trajectory_truncation(
        device, (0, 1, 0), (0, 0.5, 0), (0, 0, 0), (0, -0.1, 0), (0, 0, 1), 0.0, (0, 0.5, 0), 0.85
    )
    test.assertEqual(t, 1.0)


def test_rigid_dat_interval_flag_catches_returning_sdf_arc(test, device):
    """Stage 2 catches a crossing that returns between adjacent Stage-1 samples."""

    stage1_t = _probe_rigid_dat_returning_sdf_arc(device, use_interval_arithmetic=False)
    interval_t = _probe_rigid_dat_returning_sdf_arc(device, use_interval_arithmetic=True)
    test.assertEqual(stage1_t, 1.0, "Stage 1 endpoint samples intentionally miss this returning arc")
    test.assertLess(interval_t, 1.0 / 16.0, "Stage 2 must detect the between-sample crossing")


def test_rigid_dat_interval_catches_quarter_circle_tangent_peak(test, device):
    """Interval verification removes Stage-1's even/odd sampling coincidence."""

    epsilon = 1.0e-4
    plane_rhs = math.sqrt(2.0) - epsilon
    inv_sqrt_two = 1.0 / math.sqrt(2.0)
    probe_args = {
        "rigid_anchor": (1.0, 0.0, 0.0),
        "plane_normal": (inv_sqrt_two, inv_sqrt_two, 0.0),
        "plane_point": (0.5 * plane_rhs, 0.5 * plane_rhs, 0.0),
        "rotation_angle": math.pi / 2.0,
    }

    even_stage1_t = _probe_rigid_dat_returning_sdf_arc(device, False, trajectory_samples=8, **probe_args)
    odd_stage1_t = _probe_rigid_dat_returning_sdf_arc(device, False, trajectory_samples=9, **probe_args)
    odd_interval_t = _probe_rigid_dat_returning_sdf_arc(device, True, trajectory_samples=9, **probe_args)

    first_crossing = (math.pi / 4.0 - math.acos(plane_rhs / math.sqrt(2.0))) / (math.pi / 2.0)
    expected_t = first_crossing - 1.0e-3
    test.assertAlmostEqual(even_stage1_t, expected_t, delta=5.0e-5)
    test.assertEqual(odd_stage1_t, 1.0, "nine samples straddle and miss the narrow peak at t=1/2")
    test.assertAlmostEqual(odd_interval_t, expected_t, delta=5.0e-5)
    test.assertLess(odd_interval_t, 0.5)


def _build_sphere_drop_on_cloth(device):
    """A heavy rigid sphere shot at a pinned cloth grid: a stress scene where penalty
    forces alone cannot prevent penetration within a step."""
    builder = newton.ModelBuilder()  # Z up, gravity -Z
    builder.add_cloth_grid(
        pos=wp.vec3(-0.5, -0.5, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=16,
        dim_y=16,
        cell_x=1.0 / 16.0,
        cell_y=1.0 / 16.0,
        mass=0.05,
        fix_left=True,
        fix_right=True,
        fix_top=True,
        fix_bottom=True,
        tri_ke=1.0e3,
        tri_ka=1.0e3,
        tri_kd=1.0e-1,
        edge_ke=1.0e-2,
        particle_radius=5.0e-3,
    )
    inertia_val = 0.4 * 20.0 * 0.25**2
    inertia = wp.mat33(inertia_val, 0.0, 0.0, 0.0, inertia_val, 0.0, 0.0, 0.0, inertia_val)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.4), wp.quat_identity()),
        mass=20.0,
        inertia=inertia,
        lock_inertia=True,
    )
    builder.add_shape_sphere(body=body, radius=0.25)
    builder.color()
    model = builder.finalize(device=device)
    model.soft_contact_ke = 1.0e4
    model.soft_contact_kd = 1.0e-5
    model.soft_contact_mu = 0.5
    return model, body


def _run_sphere_drop(device, enable_dat, drop_speed=8.0, frames=60):
    model, body = _build_sphere_drop_on_cloth(device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_gap=0.1)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=4,
        rigid_soft_enable_dat=enable_dat,
        rigid_body_particle_contact_buffer_size=1024,
        collision_pipeline=pipeline,
    )
    state_in, state_out = model.state(), model.state()
    qd = state_in.body_qd.numpy()
    qd[body][:3] = [0.0, 0.0, -drop_speed]
    state_in.body_qd.assign(qd)

    worst_pen = 0.0
    body_z = 0.0
    for _frame in range(frames):
        solver.step(state_in, state_out, None, None, 1.0 / 60.0)
        state_in, state_out = state_out, state_in
        q = state_in.particle_q.numpy()
        bq = state_in.body_q.numpy()[body]
        if not (np.isfinite(q).all() and np.isfinite(bq).all()):
            raise AssertionError("simulation produced non-finite state")
        gap = np.linalg.norm(q - bq[None, :3], axis=1) - 0.25
        worst_pen = max(worst_pen, -float(gap.min()))
        body_z = float(bq[2])
    return worst_pen, body_z


def test_rigid_dat_sphere_drop_penetration_free(test, device):
    """Rigid DAT keeps a fast heavy sphere penetration-free against a pinned cloth grid.

    The control run (DAT off) penetrates and tunnels through under the same conditions,
    verifying that the assertion is meaningful.
    """
    worst_pen, body_z = _run_sphere_drop(device, enable_dat=True)
    test.assertLessEqual(worst_pen, 1.0e-4, "rigid DAT must keep cloth vertices outside the sphere")
    test.assertGreater(body_z, -0.5, "sphere must be caught by the cloth, not tunnel through")

    worst_pen_ctrl, body_z_ctrl = _run_sphere_drop(device, enable_dat=False)
    test.assertTrue(
        worst_pen_ctrl > 1.0e-3 or body_z_ctrl < -1.0,
        "control without DAT should penetrate or tunnel; if it no longer does, strengthen this stress",
    )


def test_rigid_dat_requires_owned_pipeline(test, device):
    """Enabling rigid DAT without a solver-owned pipeline raises: the DAT reference poses
    must be snapshotted at the exact detection instants the solver drives."""
    model, _body = _build_sphere_drop_on_cloth(device)
    with test.assertRaises(ValueError):
        newton.solvers.SolverVBD(model, rigid_soft_enable_dat=True)


def test_rigid_dat_requires_positive_rigid_soft_query_radius(test, device):
    """DAT rejects a zero total query radius, not merely a zero pipeline gap."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.2), vel=wp.vec3(0.0), mass=0.1, radius=0.0)
    inertia = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    body = builder.add_body(xform=wp.transform_identity(), mass=1.0, inertia=inertia, lock_inertia=True)
    builder.add_shape_sphere(body, radius=0.1, cfg=newton.ModelBuilder.ShapeConfig(margin=0.0))
    builder.color()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_gap=0.0)
    with test.assertRaisesRegex(ValueError, "positive minimum rigid-soft query radius"):
        newton.solvers.SolverVBD(model, rigid_soft_enable_dat=True, collision_pipeline=pipeline)


def test_rigid_dat_motion_bound_uses_minimum_query_radius(test, device):
    """The shared motion budget includes the smallest particle radius and shape margin.

    VT realizes the global particle-radius minimum directly. TV's barycentric radius and EE's
    maximum endpoint radius cannot be smaller, so this is a lower bound for every feature family.
    """
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.add_particle(pos=wp.vec3(-1.0, 0.0, 0.0), vel=wp.vec3(0.0), mass=0.1, radius=0.02)
    builder.add_particle(pos=wp.vec3(1.0, 0.0, 0.0), vel=wp.vec3(0.0), mass=0.1, radius=0.05)
    inertia = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    body = builder.add_body(xform=wp.transform_identity(), mass=1.0, inertia=inertia, lock_inertia=True)
    builder.add_shape_sphere(
        body,
        radius=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(margin=0.03),
    )
    builder.add_shape_sphere(
        body,
        xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
        radius=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(margin=0.07),
    )
    builder.color()
    model = builder.finalize(device=device)
    soft_gap = 0.01
    relaxation = 0.85
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_gap=soft_gap)
    solver = newton.solvers.SolverVBD(
        model,
        rigid_soft_enable_dat=True,
        rigid_soft_dat_relaxation=relaxation,
        collision_pipeline=pipeline,
    )

    expected_query_radius_min = soft_gap + 0.02 + 0.03
    expected_max_displacement = 0.5 * relaxation * expected_query_radius_min
    test.assertAlmostEqual(solver._rigid_soft_query_radius_min, expected_query_radius_min, places=7)
    test.assertAlmostEqual(solver._rigid_dat_particle_max_displacement, expected_max_displacement, places=7)
    np.testing.assert_allclose(
        solver._rigid_dat_body_max_displacement.numpy(),
        expected_max_displacement,
        rtol=0.0,
        atol=1.0e-7,
    )


def test_rigid_dat_validates_conservative_bound_relaxation(test, device):
    """Rigid DAT requires the strict relaxation interval used by its motion bound."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    model = builder.finalize(device=device)

    for relaxation in (-0.1, 0.0, 1.0, 1.1, np.nan, np.inf):
        with test.subTest(relaxation=relaxation):
            with test.assertRaisesRegex(ValueError, r"must be in \(0, 1\)"):
                newton.solvers.SolverVBD(
                    model,
                    rigid_soft_dat_relaxation=relaxation,
                )

    for relaxation in (1.0e-6, 0.5, 1.0 - 1.0e-6):
        with test.subTest(relaxation=relaxation):
            solver = newton.solvers.SolverVBD(
                model,
                rigid_soft_dat_relaxation=relaxation,
            )
            test.assertEqual(solver.rigid_soft_dat_relaxation, relaxation)


def test_rigid_dat_rejects_missing_body_pose(test, device):
    """A null body pose is valid only for static-world geometry, never for a model with bodies."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.2), vel=wp.vec3(0.0), mass=0.1, radius=0.0)
    inertia = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    body = builder.add_body(mass=1.0, inertia=inertia, lock_inertia=True)
    builder.add_shape_sphere(body, radius=0.1)
    builder.color()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_gap=0.01)
    solver = newton.solvers.SolverVBD(model, rigid_soft_enable_dat=True, collision_pipeline=pipeline)

    state = model.state()
    state.body_q = None
    with test.assertRaisesRegex(ValueError, "requires body_q"):
        solver._penetration_free_truncation(state, solver.contacts)


def _run_rigid_only_contact(device, enable_rigid_soft_dat):
    """Run an ordinary rigid sphere impact with no soft degrees of freedom."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    radius = 0.2
    mass = 1.0
    inertia_value = 0.4 * mass * radius * radius
    inertia = wp.mat33(
        inertia_value,
        0.0,
        0.0,
        0.0,
        inertia_value,
        0.0,
        0.0,
        0.0,
        inertia_value,
    )
    bodies = []
    for x in (-0.5, 0.5):
        body = builder.add_body(
            xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
            mass=mass,
            inertia=inertia,
            lock_inertia=True,
        )
        builder.add_shape_sphere(body, radius=radius)
        bodies.append(body)
    builder.color()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    solver = newton.solvers.SolverVBD(
        model,
        iterations=8,
        rigid_compliant_alm=True,
        rigid_soft_enable_dat=enable_rigid_soft_dat,
        collision_pipeline=pipeline,
    )
    state_in, state_out = model.state(), model.state()
    qd = state_in.body_qd.numpy()
    qd[bodies[0]][0] = 2.0
    qd[bodies[1]][0] = -2.0
    state_in.body_qd.assign(qd)
    saw_contact = False
    for _ in range(30):
        solver.step(state_in, state_out, None, None, 1.0 / 60.0)
        state_in, state_out = state_out, state_in
        saw_contact |= int(solver.contacts.rigid_contact_count.numpy()[0]) > 0
    return (
        state_in.body_q.numpy(),
        state_in.body_qd.numpy(),
        saw_contact,
        solver.rigid_soft_enable_dat,
        solver._rigid_dat_body_max_displacement.numpy(),
        solver._rigid_dat_particle_max_displacement,
    )


def test_rigid_soft_dat_initializes_for_rigid_only_model(test, device):
    """Keep requested DAT initialized while ordinary rigid-only ALM contact remains active."""
    q_dat, qd_dat, saw_contact, dat_enabled, body_max_displacement, particle_max_displacement = _run_rigid_only_contact(
        device, True
    )

    test.assertTrue(dat_enabled)
    test.assertTrue(saw_contact)
    test.assertTrue(np.isfinite(q_dat).all())
    test.assertTrue(np.isfinite(qd_dat).all())
    test.assertTrue(np.isinf(body_max_displacement).all())
    test.assertTrue(np.isinf(particle_max_displacement))


def _run_free_flight_distance(test, device, frequency_type, frequency, speed=6.0, frames=10):
    """Measure free rigid motion under the rigid-soft DAT budget for a schedule."""
    _Frequency = newton.solvers.SolverBase.CollisionFrequencyType
    radius = 0.2
    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_particle(pos=wp.vec3(100.0, 0.0, 0.0), vel=wp.vec3(0.0), mass=0.0, radius=0.0)
    inertia_val = 0.4 * 5.0 * radius * radius
    inertia = wp.mat33(inertia_val, 0.0, 0.0, 0.0, inertia_val, 0.0, 0.0, 0.0, inertia_val)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        mass=5.0,
        inertia=inertia,
        lock_inertia=True,
    )
    builder.add_shape_sphere(body=body, radius=radius)
    builder.color()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        contact_matching="latest",
        soft_contact_gap=0.05,
    )
    solver = newton.solvers.SolverVBD(
        model,
        iterations=10,
        rigid_soft_enable_dat=True,
        collision_pipeline=pipeline,
        collision_frequency={newton.solvers.SolverBase.CollisionSlot.RIGID: frequency},
        collision_frequency_type={newton.solvers.SolverBase.CollisionSlot.RIGID: frequency_type},
    )
    state_in, state_out = model.state(), model.state()
    qd = state_in.body_qd.numpy()
    qd[body][:3] = [speed, 0.0, 0.0]
    state_in.body_qd.assign(qd)
    for _frame in range(frames):
        solver.step(state_in, state_out, None, None, 1.0 / 60.0)
        state_in, state_out = state_out, state_in
    test.assertTrue(np.isfinite(state_in.body_q.numpy()).all())
    return float(state_in.body_q.numpy()[body][0])


def test_rigid_dat_collision_frequency_budget(test, device):
    """Raising rigid-soft detection frequency widens the DAT motion budget.

    The per-body budget is 0.5 * gamma * detection slack PER DETECTION INTERVAL, so a
    body faster than the per-step budget is throttled under PRE_POST_INIT but flies
    (nearly) freely when the reference resets every iteration (ITERATIONS k=1) — the
    configuration fix for the momentum-drain failure mode of infrequent detection.
    """
    _Frequency = newton.solvers.SolverBase.CollisionFrequencyType
    # 6 m/s -> 10 cm per step. soft gap 0.05, gamma 0.85: 2.125 cm per interval.
    # PRE_POST_INIT: 2 intervals/step -> <= ~4.25 cm/step. ITERATIONS k=1 with 10
    # iterations: 11 intervals/step -> unthrottled.
    x_slow = _run_free_flight_distance(test, device, _Frequency.PRE_POST_INIT, 1)
    x_fast = _run_free_flight_distance(test, device, _Frequency.ITERATIONS, 1)
    expected = 6.0 * 10.0 / 60.0  # unthrottled distance over 10 frames
    test.assertGreater(x_fast, 0.9 * expected, "ITERATIONS k=1 must not throttle this speed")
    test.assertLess(x_slow, 0.6 * expected, "PRE_POST_INIT should throttle this speed; retune if not")


def test_rigid_phase_applies_joint_dat_truncation(test, device):
    """A rigid-phase adaptive plane immediately truncates both sides of its SDF particle row."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.add_particle(pos=wp.vec3(0.0, 0.0, 1.0), vel=wp.vec3(0.0), mass=1.0, radius=0.0)
    inertia = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    body = builder.add_body(
        xform=wp.transform_identity(),
        mass=1.0,
        inertia=inertia,
        lock_inertia=True,
    )
    # Analytic box whose top face lies in the body's z=0 plane, so the stored
    # rigid surface point sits at the body origin like the upstream mesh triangle.
    builder.add_shape_box(
        body,
        xform=wp.transform(wp.vec3(0.0, 0.0, -0.5), wp.quat_identity()),
        hx=1.0,
        hy=1.0,
        hz=0.5,
    )
    builder.color()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        soft_contact_gap=2.0,
    )
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=False,
        collision_pipeline=pipeline,
        rigid_soft_enable_dat=True,
    )
    state = model.state()
    pipeline.collide(state, solver.contacts)
    test.assertEqual(int(solver.contacts.soft_contact_count.numpy()[0]), 1)
    np.testing.assert_array_equal(solver.contacts.soft_contact_indices.numpy()[0], [0, -1, -1])
    solver._reset_dat_references(state, reset_rigid_soft=True, reset_particles=True)

    # From a unit reference gap, let both sides propose 0.8 m of approach. The
    # adaptive plane lies halfway between the references, so accepting only one
    # factor would leave the other primitive on the wrong side of that plane.
    solver.particle_displacements.assign(np.array([[0.0, 0.0, -0.8]], dtype=np.float32))
    particle_q = state.particle_q.numpy()
    particle_q[0] = [0.0, 0.0, 0.2]
    state.particle_q.assign(particle_q)
    body_q = state.body_q.numpy()
    body_q[body, :3] = [0.0, 0.0, 0.8]
    state.body_q.assign(body_q)

    solver._rigid_penetration_free_truncation(state, solver.contacts)

    particle_z = float(state.particle_q.numpy()[0, 2])
    rigid_z = float(state.body_q.numpy()[body, 2])
    test.assertLess(float(solver.truncation_ts.numpy()[0]), 1.0)
    test.assertLess(float(solver.body_truncation_ts.numpy()[body]), 1.0)
    test.assertGreater(particle_z, 0.2, "the rigid phase must apply the soft truncation factor")
    test.assertLess(rigid_z, 0.8, "the rigid phase must apply the rigid truncation factor")
    test.assertGreaterEqual(
        particle_z - rigid_z,
        1.9 * _RIGID_SOFT_DAT_TEST_EPS,
        "both accepted primitives must remain outside the epsilon band",
    )
    np.testing.assert_allclose(
        state.particle_q.numpy(),
        solver.pos_prev_collision_detection.numpy() + solver.particle_displacements.numpy(),
        rtol=0.0,
        atol=1.0e-7,
    )


class TestVBDRigidDAT(unittest.TestCase):
    pass


add_function_test(
    TestVBDRigidDAT,
    "test_planar_truncation_uses_endpoint_signs",
    test_planar_truncation_uses_endpoint_signs,
    devices=devices,
)
add_function_test(
    TestVBDRigidDAT,
    "test_rigid_soft_dat_scale_aware_epsilon_prevents_fp32_gap_collapse",
    test_rigid_soft_dat_scale_aware_epsilon_prevents_fp32_gap_collapse,
    devices=devices,
)
add_function_test(
    TestVBDRigidDAT,
    "test_soft_self_dat_uses_epsilon_separation",
    test_soft_self_dat_uses_epsilon_separation,
    devices=devices,
)
add_function_test(
    TestVBDRigidDAT,
    "test_soft_self_dat_truncates_complete_primitive_pairs",
    test_soft_self_dat_truncates_complete_primitive_pairs,
    devices=devices,
)
add_function_test(
    TestVBDRigidDAT,
    "test_rigid_dat_trajectory_truncation",
    test_rigid_dat_trajectory_truncation,
    devices=devices,
)
add_function_test(
    TestVBDRigidDAT,
    "test_rigid_dat_interval_flag_catches_returning_sdf_arc",
    test_rigid_dat_interval_flag_catches_returning_sdf_arc,
    devices=devices,
)
add_function_test(
    TestVBDRigidDAT,
    "test_rigid_dat_interval_catches_quarter_circle_tangent_peak",
    test_rigid_dat_interval_catches_quarter_circle_tangent_peak,
    devices=devices,
)
add_function_test(
    TestVBDRigidDAT,
    "test_rigid_dat_sphere_drop_penetration_free",
    test_rigid_dat_sphere_drop_penetration_free,
    devices=devices,
)
add_function_test(
    TestVBDRigidDAT,
    "test_rigid_dat_requires_owned_pipeline",
    test_rigid_dat_requires_owned_pipeline,
    devices=devices,
)
add_function_test(
    TestVBDRigidDAT,
    "test_rigid_dat_requires_positive_rigid_soft_query_radius",
    test_rigid_dat_requires_positive_rigid_soft_query_radius,
    devices=devices,
)
add_function_test(
    TestVBDRigidDAT,
    "test_rigid_dat_motion_bound_uses_minimum_query_radius",
    test_rigid_dat_motion_bound_uses_minimum_query_radius,
    devices=devices,
)
add_function_test(
    TestVBDRigidDAT,
    "test_rigid_dat_validates_conservative_bound_relaxation",
    test_rigid_dat_validates_conservative_bound_relaxation,
    devices=devices,
)
add_function_test(
    TestVBDRigidDAT,
    "test_rigid_dat_rejects_missing_body_pose",
    test_rigid_dat_rejects_missing_body_pose,
    devices=devices,
)
add_function_test(
    TestVBDRigidDAT,
    "test_rigid_soft_dat_initializes_for_rigid_only_model",
    test_rigid_soft_dat_initializes_for_rigid_only_model,
    devices=devices,
)
add_function_test(
    TestVBDRigidDAT,
    "test_rigid_dat_collision_frequency_budget",
    test_rigid_dat_collision_frequency_budget,
    devices=devices,
)
add_function_test(
    TestVBDRigidDAT,
    "test_rigid_phase_applies_joint_dat_truncation",
    test_rigid_phase_applies_joint_dat_truncation,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
