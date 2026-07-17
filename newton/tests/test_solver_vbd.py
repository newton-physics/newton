# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the VBD solver."""

import math
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.particle_vbd_kernels import (
    accumulate_particle_body_contact_force_and_hessian,
    evaluate_dihedral_angle_based_bending_force_hessian,
    evaluate_neo_hookean_membrane_force_hessian,
    evaluate_self_contact_force_norm,
    evaluate_spring_force_and_hessian,
    evaluate_spring_force_and_hessian_both_vertices,
    evaluate_vertex_triangle_collision_force_hessian_4_vertices,
    evaluate_volumetric_neo_hookean_force_and_hessian,
)
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    RigidContactHistory,
    build_body_body_contact_lists,
    build_body_particle_contact_lists,
    compute_rigid_contact_forces,
    evaluate_angular_constraint_force_hessian,
    evaluate_body_particle_contact,
    evaluate_linear_constraint_force_hessian,
    evaluate_rigid_contact_from_collision,
    init_body_body_contacts_avbd,
    init_body_particle_contacts,
    snapshot_body_body_contact_history,
    update_duals_body_body_contacts,
    update_duals_joint,
)
from newton.tests.unittest_utils import add_function_test, configure_sdf_for_collision_shapes, get_test_devices

devices = get_test_devices()
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
        P,
        wp.vec3(5.0, 7.0, 11.0),
        wp.vec3(0.0),
        0.0,
        0.0,
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
        P,
        wp.vec3(0.0),
        wp.vec3(0.0),
        wp.vec3(5.0, 7.0, 11.0),
        wp.vec3(0.0),
        0.0,
        0.0,
        0.01,
    )
    angular_torque_out[0] = torque


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
        20.0,
        wp.vec3(0.0),
        0.0,
        0.01,
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
        0.0,
        wp.vec3(0.0),
        0.0,
        0.01,
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


def _rigid_contact_history_restore_from_match_index(test, device):
    """VBD warm-start restores from explicit match_index rows."""
    with wp.ScopedDevice(device):
        contact_count = wp.array([4], dtype=int, device=device)
        shape0 = wp.array([0, 0, 0, 0], dtype=int, device=device)
        shape1 = wp.array([1, 1, 1, 1], dtype=int, device=device)
        point0_in = np.array(
            [
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
                [13.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        point1_in = point0_in + np.array([0.0, 0.0, 1.0], dtype=np.float32)
        offset0_in = np.array(
            [
                [0.0, 0.0, 0.1],
                [0.0, 0.0, 0.2],
                [0.0, 0.0, 0.3],
                [0.0, 0.0, 0.4],
            ],
            dtype=np.float32,
        )
        offset1_in = -offset0_in
        point0 = wp.array(point0_in, dtype=wp.vec3, device=device)
        point1 = wp.array(point1_in, dtype=wp.vec3, device=device)
        offset0 = wp.array(offset0_in, dtype=wp.vec3, device=device)
        offset1 = wp.array(offset1_in, dtype=wp.vec3, device=device)
        normal = wp.array([[0.0, 0.0, 1.0]] * 4, dtype=wp.vec3, device=device)

        shape_ke = wp.array([100.0, 200.0], dtype=float, device=device)
        shape_kd = wp.array([1.0, 3.0], dtype=float, device=device)
        shape_mu = wp.array([0.25, 1.0], dtype=float, device=device)
        match_index = wp.array([2, -1, 0, -2], dtype=wp.int32, device=device)

        history = RigidContactHistory()
        history.lambda_ = wp.array([[0.5, 0.0, 1.0], [4.0, 5.0, 6.0], [0.0, 0.0, 7.0]], dtype=wp.vec3, device=device)
        history.stick_flag = wp.array([0, 1, 2], dtype=wp.int32, device=device)
        history.penalty_k = wp.array([20.0, 30.0, 40.0], dtype=float, device=device)
        history.point0 = wp.array([[20.0, 0.0, 0.0], [21.0, 0.0, 0.0], [22.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        history.point1 = wp.array([[20.0, 0.0, 1.0], [21.0, 0.0, 1.0], [22.0, 0.0, 1.0]], dtype=wp.vec3, device=device)
        history.offset0 = wp.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.6], [0.0, 0.0, 0.7]], dtype=wp.vec3, device=device)
        history.offset1 = wp.array([[0.0, 0.0, -0.5], [0.0, 0.0, -0.6], [0.0, 0.0, -0.7]], dtype=wp.vec3, device=device)
        history.normal = wp.array([[0.0, 0.0, 1.0]] * 3, dtype=wp.vec3, device=device)

        penalty_k = wp.zeros(4, dtype=float, device=device)
        lam = wp.zeros(4, dtype=wp.vec3, device=device)
        material_kd = wp.zeros(4, dtype=float, device=device)
        material_mu = wp.zeros(4, dtype=float, device=device)
        material_ke = wp.zeros(4, dtype=float, device=device)

        wp.launch(
            init_body_body_contacts_avbd,
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
                point0,
                point1,
                offset0,
                offset1,
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

        point0_out = point0.numpy()
        point1_out = point1.numpy()
        offset0_out = offset0.numpy()
        offset1_out = offset1.numpy()
        np.testing.assert_allclose(point0_out[0], [22.0, 0.0, 0.0])
        np.testing.assert_allclose(point1_out[0], [22.0, 0.0, 1.0])
        np.testing.assert_allclose(offset0_out[0], [0.0, 0.0, 0.7])
        np.testing.assert_allclose(offset1_out[0], [0.0, 0.0, -0.7])
        np.testing.assert_allclose(point0_out[2], point0_in[2])
        np.testing.assert_allclose(point1_out[2], point1_in[2])
        np.testing.assert_allclose(point0_out[1], point0_in[1])
        np.testing.assert_allclose(point0_out[3], point0_in[3])
        np.testing.assert_allclose(offset0_out[1], offset0_in[1])
        np.testing.assert_allclose(offset0_out[2], offset0_in[2])
        np.testing.assert_allclose(offset0_out[3], offset0_in[3])
        np.testing.assert_allclose(offset1_out[1], offset1_in[1])
        np.testing.assert_allclose(offset1_out[2], offset1_in[2])
        np.testing.assert_allclose(offset1_out[3], offset1_in[3])


def _rigid_contact_history_soft_restores_penalty_only(test, device):
    """Soft contacts restore penalty state only; saved lambda, points, and offsets stay unused."""
    with wp.ScopedDevice(device):
        contact_count = wp.array([1], dtype=int, device=device)
        shape0 = wp.array([0], dtype=int, device=device)
        shape1 = wp.array([1], dtype=int, device=device)
        point0_in = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
        point1_in = np.array([[10.0, 0.0, 1.0]], dtype=np.float32)
        offset0_in = np.array([[0.0, 0.0, 0.1]], dtype=np.float32)
        offset1_in = np.array([[0.0, 0.0, -0.1]], dtype=np.float32)
        point0 = wp.array(point0_in, dtype=wp.vec3, device=device)
        point1 = wp.array(point1_in, dtype=wp.vec3, device=device)
        offset0 = wp.array(offset0_in, dtype=wp.vec3, device=device)
        offset1 = wp.array(offset1_in, dtype=wp.vec3, device=device)
        normal = wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3, device=device)

        history = RigidContactHistory()
        history.lambda_ = wp.array([[1.0, 2.0, 3.0]], dtype=wp.vec3, device=device)
        history.stick_flag = wp.array([1], dtype=wp.int32, device=device)
        history.penalty_k = wp.array([40.0], dtype=float, device=device)
        history.point0 = wp.array([[20.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        history.point1 = wp.array([[20.0, 0.0, 1.0]], dtype=wp.vec3, device=device)
        history.offset0 = wp.array([[0.0, 0.0, 0.5]], dtype=wp.vec3, device=device)
        history.offset1 = wp.array([[0.0, 0.0, -0.5]], dtype=wp.vec3, device=device)
        history.normal = wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3, device=device)

        penalty_k = wp.zeros(1, dtype=float, device=device)
        lam = wp.zeros(1, dtype=wp.vec3, device=device)
        material_kd = wp.zeros(1, dtype=float, device=device)
        material_mu = wp.zeros(1, dtype=float, device=device)
        material_ke = wp.zeros(1, dtype=float, device=device)

        wp.launch(
            init_body_body_contacts_avbd,
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
                point0,
                point1,
                offset0,
                offset1,
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
        np.testing.assert_allclose(point0.numpy(), point0_in)
        np.testing.assert_allclose(point1.numpy(), point1_in)
        np.testing.assert_allclose(offset0.numpy(), offset0_in)
        np.testing.assert_allclose(offset1.numpy(), offset1_in)


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
            contacts = model.contacts(collision_pipeline=pipeline)

        solver = newton.solvers.SolverVBD(model, iterations=1, rigid_contact_history=True)

        if not pipeline_first:
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=rigid_contact_max, contact_matching="latest")
            contacts = model.contacts(collision_pipeline=pipeline)

        state_in = model.state()
        state_out = model.state()
        control = model.control()
        if rigid_contact_max > 0:
            model.collide(state_in, contacts)
        return model, solver, contacts, state_in, state_out, control

    model, solver, contacts, state_in, state_out, control = make_scene(pipeline_first=False)
    with test.assertRaisesRegex(RuntimeError, "contact history must be allocated before CUDA graph capture"):
        with wp.ScopedCapture(device=device):
            solver.step(state_in, state_out, control, contacts, 1.0e-3)

    model, solver, contacts, state_in, state_out, control = make_scene(pipeline_first=True)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(state_in, state_out, control, contacts, 1.0e-3)
    test.assertIsNotNone(capture.graph)

    model, solver, contacts, state_in, state_out, control = make_scene(pipeline_first=True, rigid_contact_max=0)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(state_in, state_out, control, contacts, 1.0e-3)
    test.assertIsNotNone(capture.graph)
    test.assertIsNone(solver._prev_contact_lambda)

    model, solver, contacts, state_in, state_out, control = make_scene(pipeline_first=False)
    solver.step(state_in, state_out, control, contacts, 1.0e-3)
    model.collide(state_out, contacts)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(state_out, state_in, control, contacts, 1.0e-3)
    test.assertIsNotNone(capture.graph)


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
        # Distinct fresh anchors per row; equal current/saved normals so a warm
        # restore reproduces the saved dual exactly.
        point0_in = np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0]], dtype=np.float32)
        point1_in = np.array([[10.0, 0.0, 1.0], [11.0, 0.0, 1.0], [12.0, 0.0, 1.0]], dtype=np.float32)
        offset0_in = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3]], dtype=np.float32)
        offset1_in = np.array([[0.0, 0.0, -0.1], [0.0, 0.0, -0.2], [0.0, 0.0, -0.3]], dtype=np.float32)
        point0 = wp.array(point0_in, dtype=wp.vec3, device=device)
        point1 = wp.array(point1_in, dtype=wp.vec3, device=device)
        offset0 = wp.array(offset0_in, dtype=wp.vec3, device=device)
        offset1 = wp.array(offset1_in, dtype=wp.vec3, device=device)
        normal = wp.array([[0.0, 0.0, 1.0]] * 3, dtype=wp.vec3, device=device)

        # Distinct sticky saved anchors per slot so the warm restore is observable.
        history = RigidContactHistory()
        history.lambda_ = wp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=wp.vec3, device=device)
        history.stick_flag = wp.array([1, 1, 1], dtype=wp.int32, device=device)
        history.penalty_k = wp.array([40.0, 50.0, 60.0], dtype=float, device=device)
        history.point0 = wp.array([[20.0, 0.0, 0.0], [21.0, 0.0, 0.0], [22.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        history.point1 = wp.array([[20.0, 0.0, 1.0], [21.0, 0.0, 1.0], [22.0, 0.0, 1.0]], dtype=wp.vec3, device=device)
        history.offset0 = wp.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.6], [0.0, 0.0, 0.7]], dtype=wp.vec3, device=device)
        history.offset1 = wp.array([[0.0, 0.0, -0.5], [0.0, 0.0, -0.6], [0.0, 0.0, -0.7]], dtype=wp.vec3, device=device)
        history.normal = wp.array([[0.0, 0.0, 1.0]] * 3, dtype=wp.vec3, device=device)

        penalty_k = wp.zeros(3, dtype=float, device=device)
        contact_lambda = wp.zeros(3, dtype=wp.vec3, device=device)
        material_kd = wp.zeros(3, dtype=float, device=device)
        material_mu = wp.zeros(3, dtype=float, device=device)
        material_ke = wp.zeros(3, dtype=float, device=device)

        wp.launch(
            init_body_body_contacts_avbd,
            dim=3,
            inputs=[
                contact_count,
                shape0,
                shape1,
                normal,
                wp.array([100.0] * 4, dtype=float, device=device),
                wp.zeros(4, dtype=float, device=device),
                wp.zeros(4, dtype=float, device=device),
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
                point0,
                point1,
                offset0,
                offset1,
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
        # shape respectively): both cold-start with a zero dual and keep their
        # fresh anchors instead of the saved ones.
        for row in (0, 1):
            np.testing.assert_allclose(lam[row], 0.0)
            np.testing.assert_allclose(point0.numpy()[row], point0_in[row])
            np.testing.assert_allclose(point1.numpy()[row], point1_in[row])
            np.testing.assert_allclose(offset0.numpy()[row], offset0_in[row])
            np.testing.assert_allclose(offset1.numpy()[row], offset1_in[row])
        # Row 2 owns unselected world 1 and warm-restores its saved slot (1):
        # dual and all four anchors come from history through the nonidentity slot.
        np.testing.assert_allclose(lam[2], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(point0.numpy()[2], [21.0, 0.0, 0.0])
        np.testing.assert_allclose(point1.numpy()[2], [21.0, 0.0, 1.0])
        np.testing.assert_allclose(offset0.numpy()[2], [0.0, 0.0, 0.6])
        np.testing.assert_allclose(offset1.numpy()[2], [0.0, 0.0, -0.6])
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
        joint_qd_start = wp.array([0], dtype=wp.int32, device=device)
        joint_target_q_start = wp.array([0], dtype=wp.int32, device=device)
        joint_constraint_start = wp.array([0], dtype=wp.int32, device=device)
        body_q = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        body_q_rest = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        joint_dof_dim = wp.array([[0, 0]], dtype=wp.int32, device=device)
        joint_c0_lin = wp.zeros(1, dtype=wp.vec3, device=device)
        joint_c0_ang = wp.zeros(1, dtype=wp.vec3, device=device)
        joint_is_hard = wp.array([1, 1, 0], dtype=wp.int32, device=device)
        joint_penalty_k_max = wp.array([10.0, 10.0, 10.0], dtype=float, device=device)
        joint_target_ke = wp.array([0.0], dtype=float, device=device)
        joint_target_pos = wp.array([0.0], dtype=float, device=device)
        joint_limit_lower = wp.array([-1.0], dtype=float, device=device)
        joint_limit_upper = wp.array([1.0], dtype=float, device=device)
        joint_limit_ke = wp.array([0.0], dtype=float, device=device)
        joint_rest_angle = wp.array([0.0], dtype=float, device=device)
        joint_penalty_k = wp.array([10.0, 10.0, 10.0], dtype=float, device=device)
        lambda_lin = wp.zeros(1, dtype=wp.vec3, device=device)
        lambda_ang = wp.array([[5.0, 2.0, 3.0]], dtype=wp.vec3, device=device)

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
                joint_qd_start,
                joint_target_q_start,
                joint_constraint_start,
                body_q,
                body_q_rest,
                joint_dof_dim,
                joint_c0_lin,
                joint_c0_ang,
                joint_is_hard,
                0.0,
                joint_penalty_k_max,
                0.0,
                0.0,
                joint_target_ke,
                joint_target_pos,
                joint_limit_lower,
                joint_limit_upper,
                joint_limit_ke,
                joint_rest_angle,
            ],
            outputs=[joint_penalty_k, lambda_lin, lambda_ang],
            device=device,
        )

        np.testing.assert_allclose(lambda_ang.numpy(), [[0.0, 2.0, 3.0]])


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
        particle_colors = wp.zeros(4, dtype=int, device=device)
        particle_radius = wp.array([0.1] * 4, dtype=float, device=device)

        # Single total soft counter; only the particle path is exercised here (records (p, -1, -1)).
        contact_count = wp.array([4], dtype=int, device=device)
        contact_indices = wp.array([[0, -1, -1], [1, -1, -1], [2, -1, -1], [3, -1, -1]], dtype=wp.vec3i, device=device)
        contact_penalty_k = wp.array([400.0, 400.0, 100.0, 100.0], dtype=float, device=device)
        contact_material_ke = wp.array([100.0] * 4, dtype=float, device=device)
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

        wp.launch(
            accumulate_particle_body_contact_force_and_hessian,
            dim=4,
            inputs=[
                0.1,
                0,
                particle_q_prev,
                particle_q,
                particle_colors,
                0.01,
                particle_radius,
                contact_indices,
                contact_count,
                4,
                contact_penalty_k,
                contact_material_ke,
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

        penalty_k = wp.array([400.0, 400.0, 100.0, 100.0], dtype=float, device=device)
        material_ke = wp.array([100.0] * 4, dtype=float, device=device)
        material_kd = wp.array([20.0, 0.0, 20.0, 0.0], dtype=float, device=device)
        material_mu = wp.zeros(4, dtype=float, device=device)
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
                material_ke,
                material_kd,
                material_mu,
                contact_lambda,
                contact_c0,
                0.95,
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
    solver = newton.solvers.SolverVBD(model)
    start = int(solver.joint_constraint_start.numpy()[joint])

    np.testing.assert_allclose(solver.joint_penalty_k.numpy()[start : start + 2], [0.0, 0.0])
    np.testing.assert_allclose(solver.joint_penalty_k_max.numpy()[start : start + 2], [0.0, 0.0])
    np.testing.assert_array_equal(solver.joint_is_hard.numpy()[start : start + 2], [0, 0])


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
    world_mask = wp.array([True, False], dtype=wp.bool, device=device)

    solver = newton.solvers.SolverVBD(model, iterations=0)
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
    with test.assertRaisesRegex(ValueError, "one-dimensional Warp boolean array"):
        solver.reset(state, world_mask=wp.array([1, 0], dtype=wp.int32, device=device))
    with test.assertRaisesRegex(ValueError, "world_mask has length 1, expected 2 or 3"):
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

    solver = newton.solvers.SolverVBD(model, iterations=0)
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
    world_mask = wp.array([True, False], dtype=wp.bool, device=device)
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
    """A reset cold-starts only selected-world contacts, once, on the next refresh."""
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
    reset_mask = wp.array([True, False], dtype=wp.bool, device=device)
    dt = 1.0e-2

    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
    contacts = pipeline.contacts()
    # Fixed-k (no ramping) so cold vs warm is proven by the dual alone;
    # contact_alpha=gamma=1 disable the per-step lambda decay so a seeded dual
    # survives a step unchanged.
    solver = newton.solvers.SolverVBD(
        model,
        iterations=0,
        rigid_contact_history=True,
        rigid_contact_stick_motion_eps=0.0,
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
        solver._prev_contact_stick_flag.zero_()
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


def _vbd_custom_attribute_registration_controls_dahl_defaults(test, device):
    del device

    builder = newton.ModelBuilder()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        newton.solvers.SolverVBD.register_custom_attributes(builder)
    test.assertIn("vbd:joint_is_hard", builder.custom_attributes)
    test.assertIn("vbd:dahl_eps_max", builder.custom_attributes)
    test.assertIn("vbd:dahl_tau", builder.custom_attributes)
    test.assertEqual(builder.custom_attributes["vbd:joint_is_hard"].default, 1)
    test.assertEqual(builder.custom_attributes["vbd:dahl_eps_max"].default, 0.5)
    test.assertEqual(builder.custom_attributes["vbd:dahl_tau"].default, 1.0)
    test.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))

    builder = newton.ModelBuilder()
    newton.solvers.SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)
    test.assertIn("vbd:joint_is_hard", builder.custom_attributes)
    test.assertIn("vbd:dahl_eps_max", builder.custom_attributes)
    test.assertIn("vbd:dahl_tau", builder.custom_attributes)
    test.assertEqual(builder.custom_attributes["vbd:joint_is_hard"].default, 1)
    test.assertEqual(builder.custom_attributes["vbd:dahl_eps_max"].default, 0.0)
    test.assertEqual(builder.custom_attributes["vbd:dahl_tau"].default, 0.0)


def _make_vbd_dahl_detection_model(device, *, dahl_defaults_enabled, dahl_eps_max=None, dahl_tau=None):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        newton.solvers.SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=dahl_defaults_enabled)

    parent = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
    child = builder.add_link(xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_box(parent, hx=0.1, hy=0.1, hz=0.1)
    builder.add_shape_box(child, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_cable(
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
    model = _make_vbd_dahl_detection_model(device, dahl_defaults_enabled=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        solver = newton.solvers.SolverVBD(model)
    test.assertFalse(solver.enable_dahl_friction)

    model = _make_vbd_dahl_detection_model(device, dahl_defaults_enabled=False, dahl_eps_max=0.5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        solver = newton.solvers.SolverVBD(model)
    test.assertFalse(solver.enable_dahl_friction)

    model = _make_vbd_dahl_detection_model(device, dahl_defaults_enabled=False, dahl_tau=1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        solver = newton.solvers.SolverVBD(model)
    test.assertFalse(solver.enable_dahl_friction)

    model = _make_vbd_dahl_detection_model(device, dahl_defaults_enabled=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        solver = newton.solvers.SolverVBD(model)
    test.assertTrue(solver.enable_dahl_friction)

    model = _make_vbd_dahl_detection_model(device, dahl_defaults_enabled=False, dahl_eps_max=0.5, dahl_tau=1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        solver = newton.solvers.SolverVBD(model)
    test.assertTrue(solver.enable_dahl_friction)


def _rigid_reset_cable_history(test, device):
    """Reset defers the cable tuple, then rebaselines it from the post-reset pose."""
    model = _make_vbd_dahl_detection_model(device, dahl_defaults_enabled=False, dahl_eps_max=0.5, dahl_tau=1.0)
    solver = newton.solvers.SolverVBD(model, iterations=0)

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
        point0 = wp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        point1 = wp.array([[1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0]], dtype=wp.vec3, device=device)
        offset0 = wp.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3]], dtype=wp.vec3, device=device)
        offset1 = wp.array([[0.0, 0.0, -0.1], [0.0, 0.0, -0.2], [0.0, 0.0, -0.3]], dtype=wp.vec3, device=device)
        normal = wp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        lam = wp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=wp.vec3, device=device)
        stick = wp.array([1, 2, 3], dtype=wp.int32, device=device)
        penalty = wp.array([10.0, 20.0, 30.0], dtype=float, device=device)

        prev_lambda = wp.zeros(3, dtype=wp.vec3, device=device)
        prev_stick = wp.zeros(3, dtype=wp.int32, device=device)
        prev_penalty = wp.zeros(3, dtype=float, device=device)
        prev_point0 = wp.zeros(3, dtype=wp.vec3, device=device)
        prev_point1 = wp.zeros(3, dtype=wp.vec3, device=device)
        prev_offset0 = wp.zeros(3, dtype=wp.vec3, device=device)
        prev_offset1 = wp.zeros(3, dtype=wp.vec3, device=device)
        prev_normal = wp.zeros(3, dtype=wp.vec3, device=device)

        wp.launch(
            snapshot_body_body_contact_history,
            dim=3,
            inputs=[contact_count, point0, point1, offset0, offset1, normal, lam, stick, penalty],
            outputs=[
                prev_lambda,
                prev_stick,
                prev_penalty,
                prev_point0,
                prev_point1,
                prev_offset0,
                prev_offset1,
                prev_normal,
            ],
            device=device,
        )

        np.testing.assert_allclose(prev_lambda.numpy()[:2], lam.numpy()[:2])
        np.testing.assert_allclose(prev_stick.numpy()[:2], [1, 2])
        np.testing.assert_allclose(prev_penalty.numpy()[:2], [10.0, 20.0])
        np.testing.assert_allclose(prev_point0.numpy()[:2], point0.numpy()[:2])
        np.testing.assert_allclose(prev_point1.numpy()[:2], point1.numpy()[:2])
        np.testing.assert_allclose(prev_offset0.numpy()[:2], offset0.numpy()[:2])
        np.testing.assert_allclose(prev_offset1.numpy()[:2], offset1.numpy()[:2])
        np.testing.assert_allclose(prev_normal.numpy()[:2], normal.numpy()[:2])
        np.testing.assert_allclose(prev_lambda.numpy()[2], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(prev_offset0.numpy()[2], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(prev_offset1.numpy()[2], [0.0, 0.0, 0.0])
        test.assertEqual(prev_stick.numpy()[2], 0)
        test.assertEqual(prev_penalty.numpy()[2], 0.0)


def _rigid_contact_stick_flags_require_cone_and_small_residual(test, device):
    """Contact stick flags require normal load, cone feasibility, and small tangential residual."""
    with wp.ScopedDevice(device):
        contact_count = wp.array([4], dtype=int, device=device)
        shape0 = wp.array([0, 0, 0, 0], dtype=int, device=device)
        shape1 = wp.array([1, 2, 3, 4], dtype=int, device=device)
        point0 = wp.zeros(4, dtype=wp.vec3, device=device)
        point1 = wp.zeros(4, dtype=wp.vec3, device=device)
        offset0 = wp.zeros(4, dtype=wp.vec3, device=device)
        offset1 = wp.zeros(4, dtype=wp.vec3, device=device)
        normal = wp.array([[0.0, 0.0, 1.0]] * 4, dtype=wp.vec3, device=device)
        margin0 = wp.array([0.05, 0.05, 0.05, 0.05], dtype=float, device=device)
        margin1 = wp.array([0.05, 0.05, 0.05, 0.05], dtype=float, device=device)
        shape_body = wp.array([0, 1, 2, 3, 4], dtype=int, device=device)

        q = wp.quat_identity()
        body_q = wp.array(
            [
                wp.transform(wp.vec3(0.0, 0.0, 0.0), q),
                wp.transform(wp.vec3(1.0, 0.0, 0.0), q),
                wp.transform(wp.vec3(0.03, 0.0, 0.0), q),
                wp.transform(wp.vec3(0.01, 0.0, 0.0), q),
                wp.transform(wp.vec3(0.01, 0.0, 0.0), q),
            ],
            dtype=wp.transform,
            device=device,
        )
        body_q_prev = wp.array([wp.transform_identity()] * 5, dtype=wp.transform, device=device)
        contact_mu = wp.array([0.5, 0.5, 0.5, 0.5], dtype=float, device=device)
        contact_c0 = wp.zeros(4, dtype=wp.vec3, device=device)
        body_inv_mass = wp.array([1.0, 0.0, 0.0, 0.0, 1.0], dtype=float, device=device)
        contact_ke = wp.array([10.0, 10.0, 10.0, 10.0], dtype=float, device=device)
        penalty_k = wp.array([10.0, 10.0, 10.0, 10.0], dtype=float, device=device)
        contact_lambda = wp.zeros(4, dtype=wp.vec3, device=device)
        stick_flag = wp.zeros(4, dtype=wp.int32, device=device)

        wp.launch(
            update_duals_body_body_contacts,
            dim=4,
            inputs=[
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
                contact_mu,
                contact_c0,
                0.0,
                0.02,
                1,
                body_inv_mass,
                contact_ke,
                0.0,
            ],
            outputs=[penalty_k, contact_lambda, stick_flag],
            device=device,
        )

        np.testing.assert_allclose(
            contact_lambda.numpy(),
            [
                [-0.5, 0.0, 1.0],
                [-0.3, 0.0, 1.0],
                [-0.1, 0.0, 1.0],
                [-0.1, 0.0, 1.0],
            ],
        )
        np.testing.assert_array_equal(stick_flag.numpy(), [0, 0, 1, 2])

        contact_lambda.zero_()
        stick_flag.zero_()
        penalty_k = wp.array([10.0, 10.0, 10.0, 10.0], dtype=float, device=device)

        wp.launch(
            update_duals_body_body_contacts,
            dim=4,
            inputs=[
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
                contact_mu,
                contact_c0,
                0.0,
                0.0,
                1,
                body_inv_mass,
                contact_ke,
                0.0,
            ],
            outputs=[penalty_k, contact_lambda, stick_flag],
            device=device,
        )

        np.testing.assert_array_equal(stick_flag.numpy(), [0, 0, 0, 0])


def _capsule_axial_spin_dissipates_via_friction(test, device, hard_contact=True):
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
        solver = newton.solvers.SolverVBD(model, iterations=10, rigid_contact_hard=hard_contact)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        init_qd = state_0.body_qd.numpy().copy()
        init_qd[0] = [0.0, 0.0, 0.0, omega_init, 0.0, 0.0]
        state_0.body_qd = wp.array(init_qd, dtype=wp.spatial_vector)

        sim_dt = 1.0e-3
        for _ in range(500):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

        qd = state_0.body_qd.numpy()[0]

    v_y = float(qd[1])
    omega_x = float(qd[3])

    test.assertLess(v_y, -0.1, f"capsule failed to translate under axial spin (v_y={v_y:.4f}, omega_x={omega_x:.4f})")
    test.assertLess(omega_x, 4.0, f"axial spin failed to dissipate (omega_x={omega_x:.4f}, v_y={v_y:.4f})")


def _yawed_cable_does_not_inject_energy(test, device, hard_contact=True):
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
    points = newton.utils.create_straight_cable_points(
        start=start, direction=direction, length=length, num_segments=num_segments
    )
    quaternions = newton.utils.create_parallel_transport_cable_quaternions(points, twist_total=0.0)
    bodies, _joints = builder.add_rod(
        positions=points,
        quaternions=quaternions,
        radius=radius,
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
            rigid_contact_hard=hard_contact,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

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
                model.collide(state_0, contacts)
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
        solver = newton.solvers.SolverVBD(model, iterations=2)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        model.collide(state_0, contacts)
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
    "test_rigid_contact_history_restore_from_match_index",
    _rigid_contact_history_restore_from_match_index,
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
    "test_rigid_contact_history_capture_requires_preallocation",
    _rigid_contact_history_capture_requires_preallocation,
    devices=cuda_devices,
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
    "test_rigid_reset_state_and_history",
    _rigid_reset_state_and_history,
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
    "test_rigid_contact_stick_flags_require_cone_and_small_residual",
    _rigid_contact_stick_flags_require_cone_and_small_residual,
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
    "test_collect_rigid_contact_forces_reports_surface_points",
    _collect_rigid_contact_forces_reports_surface_points,
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
    """A soft edge/face penetrating a rigid box pushes its triangle's vertices out (+y).

    With section 2 absent the particle force stays zero (legacy count is 0, gravity off),
    so the vertices never move. With section 2 present the barycentric distribution drives
    v0 and v1 (the spanning edge) up out of the box.
    """
    model, (v0, v1, _v2) = _build_edge_over_post(device)

    margin = 0.1
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_margin=margin, enable_rigid_soft_full_surface_contact=True
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

    solver = newton.solvers.SolverVBD(model)

    y0_before = state_in.particle_q.numpy()[:, 1].copy()
    solver.step(state_in, state_out, None, contacts, dt=1.0 / 60.0)
    y0_after = state_out.particle_q.numpy()[:, 1]

    # The two vertices of the spanning edge are pushed up out of the +y face.
    test.assertGreater(y0_after[v0] - y0_before[v0], 1.0e-3, "v0 should be pushed +y")
    test.assertGreater(y0_after[v1] - y0_before[v1], 1.0e-3, "v1 should be pushed +y")


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
        model, broad_phase="nxn", soft_contact_margin=margin, enable_rigid_soft_full_surface_contact=True
    )
    contacts = pipeline.contacts()
    state_in = model.state()
    state_out = model.state()

    pipeline.collide(state_in, contacts)
    total = int(contacts.soft_contact_count.numpy()[0])
    idx = contacts.soft_contact_indices.numpy()[:total]
    test.assertEqual(int(np.sum(idx[:, 1] < 0)), 0, "triangle vertices should be outside the legacy particle margin")
    test.assertGreater(total, 0, "a soft edge/face contact must be detected")

    solver = newton.solvers.SolverVBD(model)
    dt = 1.0 / 60.0
    z_before = float(state_in.body_q.numpy()[body, 2])

    for _ in range(60):
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, dt)
        state_in, state_out = state_out, state_in

    z_after = float(state_in.body_q.numpy()[body, 2])
    test.assertGreater(z_after, z_before - 0.05, "box should be supported by the soft contact, not free-fall")


def _set_slot(arr, idx, value):
    a = arr.numpy()
    a[idx] = value
    arr.assign(a)


def _run_face_section2(device, shape_margin):
    """Build a single soft-FACE contact, seed the shared AVBD per-contact material via
    ``init_body_particle_contacts``, then launch the particle-side kernel once with the given
    ``shape_margin`` array. The geometry gives a 0.05 penetration along +z; returns
    ``(forces, hessians, ke, bary, (p0, p1, p2))`` where ``ke`` is the mixed effective stiffness
    section 2 reads. All vertices share color 0 so one launch processes the whole triangle."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(body=-1, xform=wp.transform(wp.vec3(0.0), wp.quat_identity()), hx=1.0, hy=1.0, hz=1.0)
    p0 = builder.add_particle(wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0), 0.1, radius=0.0)
    p1 = builder.add_particle(wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0), 0.1, radius=0.0)
    p2 = builder.add_particle(wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0), 0.1, radius=0.0)
    builder.add_triangle(p0, p1, p2)
    configure_sdf_for_collision_shapes(builder)
    model = builder.finalize(device=device)

    smax = 8
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_margin=0.1, soft_contact_max=smax)
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
    model.particle_colors.assign([0, 0, 0])

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

    wp.launch(
        accumulate_particle_body_contact_force_and_hessian,
        dim=smax,
        inputs=[
            0.01,  # dt
            0,  # current_color
            state.particle_q,  # pos_anchor == pos -> no damping / friction
            state.particle_q,
            model.particle_colors,
            1.0,  # friction_epsilon
            model.particle_radius,
            contacts.soft_contact_indices,
            contacts.soft_contact_count,
            smax,
            penalty_k,
            material_ke,
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
        model, broad_phase="nxn", soft_contact_margin=0.1, enable_rigid_soft_full_surface_contact=False
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


def test_full_surface_rejected_by_vbd_proxy_coupling(test, device):
    """SolverVBD's proxy-coupling hook fails loud on full-surface contacts, which its proxy harvest
    cannot yet consume, instead of silently dropping edge/face force feedback (E5). Standalone
    SolverVBD is unaffected -- this only guards the SolverCoupledProxy path (coupling_* hooks)."""
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
        model, broad_phase="nxn", soft_contact_margin=0.1, enable_rigid_soft_full_surface_contact=True
    )
    contacts = pipeline.contacts()  # capability marker set True
    solver = newton.solvers.SolverVBD(model)
    with test.assertRaises(NotImplementedError):
        solver.coupling_prepare_proxy_contacts(model.state(), contacts)


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
    "test_full_surface_rejected_by_vbd_proxy_coupling",
    test_full_surface_rejected_by_vbd_proxy_coupling,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
