# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optional reconstruction of VBD's incoming joint wrenches."""

import warp as wp

from ...sim import JointType
from ...sim.articulation import transform_2d_rotational_axes, transform_3d_rotational_axes
from .joint_coordinates import JointCoordinateData, evaluate_coordinate
from .rigid_vbd_kernels import (
    _DRIVE_LIMIT_MODE_DRIVE,
    _drive_row_applies_force,
    _evaluate_drive_axis,
    _evaluate_drive_limit_axis,
    _evaluate_joint_dissipation,
    _limit_row_exists,
    _load_joint_axis_drive_limit,
    _resolve_active_drive_row,
    evaluate_joint_force_hessian,
)

wp.set_module_options({"enable_backward": False})


@wp.struct
class JointForceData:
    coordinates: JointCoordinateData
    body_q_rest: wp.array[wp.transform]
    body_articulation_local: wp.array[int]
    joint_enabled: wp.array[bool]
    joint_target_q_start: wp.array[int]
    joint_target_ke: wp.array[float]
    joint_target_kd: wp.array[float]
    joint_limit_lower: wp.array[float]
    joint_limit_upper: wp.array[float]
    joint_limit_ke: wp.array[float]
    joint_limit_kd: wp.array[float]
    joint_friction: wp.array[float]
    joint_damping: wp.array[float]
    joint_rod_rest_kb_local: wp.array[wp.vec3]
    joint_rod_rest_twist: wp.array[float]
    joint_constraint_start: wp.array[int]
    joint_penalty_k: wp.array[float]
    joint_rho: wp.array[float]
    joint_material_k: wp.array[float]
    joint_penalty_kd: wp.array[float]
    joint_sigma_start: wp.array[wp.vec3]
    joint_C_fric: wp.array[wp.vec3]
    joint_drive_limit_support: wp.array[float]
    joint_drive_lambda: wp.array[float]
    joint_limit_lambda: wp.array[float]
    joint_lambda_lin: wp.array[wp.vec3]
    joint_lambda_ang: wp.array[wp.vec3]
    joint_C0_lin: wp.array[wp.vec3]
    joint_C0_ang: wp.array[wp.vec3]
    joint_is_hard: wp.array[int]
    alpha: float
    compliant_alm: int


@wp.func
def _to_child_joint_frame(
    coordinates: JointCoordinateData, joint: int, body_q: wp.array[wp.transform], wrench: wp.spatial_vector
):
    """Shift a world-space child-COM wrench to the child's incoming joint frame."""
    child = coordinates.child[joint]
    pose = body_q[child]
    joint_pose = pose * coordinates.X_c[joint]
    rotation = wp.transform_get_rotation(joint_pose)
    lever = wp.transform_get_translation(joint_pose) - wp.transform_point(pose, coordinates.body_com[child])
    force = wp.spatial_top(wrench)
    torque = wp.spatial_bottom(wrench) - wp.cross(lever, force)
    return wp.spatial_vector(wp.quat_rotate_inv(rotation, force), wp.quat_rotate_inv(rotation, torque))


@wp.func
def _actuation_effort(
    c: JointCoordinateData, joint: int, component: int, body_q: wp.array[wp.transform], actuation: wp.spatial_vector
):
    """Project actual applied actuation onto the coordinate's motion subspace."""
    parent_frame = c.X_p[joint]
    if c.parent[joint] >= 0:
        parent_frame = body_q[c.parent[joint]] * parent_frame
    start = c.qd_start[joint]
    linear_count = c.dof_dim[joint, 0]
    axis = c.axis[start + component]
    if component < linear_count:
        return wp.dot(wp.transform_vector(parent_frame, axis), wp.spatial_top(actuation))
    angular_count = c.dof_dim[joint, 1]
    if angular_count > 1:
        a0 = c.axis[start + linear_count]
        a1 = c.axis[start + linear_count + 1]
        q0, _p0, _c0 = evaluate_coordinate(c, joint, linear_count, body_q)
        if angular_count == 2:
            a0, a1 = transform_2d_rotational_axes(a0, a1, q0)
            axis = a0 if component == linear_count else a1
        else:
            a2 = c.axis[start + linear_count + 2]
            q1, _p1, _c1 = evaluate_coordinate(c, joint, linear_count + 1, body_q)
            a0, a1, a2 = transform_3d_rotational_axes(a0, a1, a2, q0, q1)
            axis = a2
            if component == linear_count:
                axis = a0
            elif component == linear_count + 1:
                axis = a1
    child_pose = body_q[c.child[joint]]
    lever = wp.transform_get_translation(child_pose * c.X_c[joint]) - wp.transform_point(
        child_pose, c.body_com[c.child[joint]]
    )
    torque_at_joint = wp.spatial_bottom(actuation) - wp.cross(lever, wp.spatial_top(actuation))
    return wp.dot(wp.transform_vector(parent_frame, axis), torque_at_joint)


@wp.func
def _drive_effort(
    data: JointForceData,
    joint: int,
    component: int,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    target_q: wp.array[float],
    target_qd: wp.array[float],
    dt: float,
):
    """Reconstruct motor drive effort without including passive limit reactions."""
    c = data.coordinates
    dof = c.qd_start[joint] + component
    axis = _load_joint_axis_drive_limit(
        dof,
        data.joint_target_q_start[joint] + component,
        data.joint_constraint_start[joint] + 2 + component,
        data.joint_target_ke,
        data.joint_target_kd,
        target_q,
        target_qd,
        data.joint_limit_lower,
        data.joint_limit_upper,
        data.joint_limit_ke,
        data.joint_limit_kd,
        data.joint_penalty_k,
        data.compliant_alm,
    )
    q, _g_p, _g_c = evaluate_coordinate(c, joint, component, body_q)
    q_prev, _g_pp, _g_cp = evaluate_coordinate(c, joint, component, body_q_prev)
    displacement = q - q_prev
    if component >= c.dof_dim[joint, 0]:
        displacement = wp.atan2(wp.sin(displacement), wp.cos(displacement))
    rate = displacement / dt
    if component < c.dof_dim[joint, 0]:
        # Linear drives currently project the anchor displacement change onto
        # the CURRENT parent axis; use exactly that law, including moving parents.
        parent_now = c.X_p[joint]
        parent_prev = parent_now
        if c.parent[joint] >= 0:
            parent_now = body_q[c.parent[joint]] * parent_now
            parent_prev = body_q_prev[c.parent[joint]] * parent_prev
        child_now = body_q[c.child[joint]] * c.X_c[joint]
        child_prev = body_q_prev[c.child[joint]] * c.X_c[joint]
        axis_world = wp.normalize(wp.transform_vector(parent_now, c.axis[dof]))
        separation = wp.transform_get_translation(child_now) - wp.transform_get_translation(parent_now)
        separation_prev = wp.transform_get_translation(child_prev) - wp.transform_get_translation(parent_prev)
        q = wp.dot(separation, axis_world)
        rate = wp.dot(separation - separation_prev, axis_world) / dt
    mode, error = _resolve_active_drive_row(
        q,
        axis.target_pos,
        axis.lower,
        axis.upper,
        _drive_row_applies_force(axis.material_drive_ke, axis.drive_kd),
        _limit_row_exists(axis.material_limit_ke, axis.lower, axis.upper),
        data.compliant_alm,
    )
    force = float(0.0)
    if mode == _DRIVE_LIMIT_MODE_DRIVE:
        if data.compliant_alm == 1:
            force, _h = _evaluate_drive_axis(
                error,
                rate,
                axis.target_vel,
                mode,
                axis.drive_ke,
                axis.drive_kd,
                data.joint_drive_limit_support[dof],
                data.joint_drive_lambda[dof],
                1.0 / dt,
            )
        else:
            force, _h = _evaluate_drive_limit_axis(
                error,
                rate,
                axis.target_vel,
                mode,
                axis.drive_ke,
                axis.drive_kd,
                0.0,
                0.0,
                1.0 / dt,
            )
    return -force


@wp.kernel
def evaluate_joint_forces(
    data: JointForceData,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    joint_target_q: wp.array[float],
    joint_target_qd: wp.array[float],
    actuation_wrench: wp.array[wp.spatial_vector],
    joint_armature: wp.array[float],
    joint_qd: wp.array[float],
    joint_qd_prev: wp.array[float],
    references: wp.array[int],
    coeffs: wp.array[wp.vec2],
    dt: float,
    joint_wrench: wp.array[wp.spatial_vector],
    joint_effort: wp.array[float],
):
    joint = wp.tid()
    actuation = wp.spatial_vector()
    if actuation_wrench:
        actuation = actuation_wrench[joint]
    joint_wrench[joint] = wp.spatial_vector()
    if not data.joint_enabled[joint]:
        return
    c = data.coordinates
    child = c.child[joint]
    if child < 0:
        return
    force, torque, _H_ll, _H_al, _H_aa = evaluate_joint_force_hessian(
        child,
        joint,
        body_q,
        body_q_prev,
        data.body_q_rest,
        c.body_com,
        c.joint_type,
        data.joint_enabled,
        c.parent,
        c.child,
        c.X_p,
        c.X_c,
        c.axis,
        data.joint_rod_rest_kb_local,
        data.joint_rod_rest_twist,
        c.qd_start,
        data.joint_target_q_start,
        data.joint_constraint_start,
        data.joint_penalty_k,
        data.joint_rho,
        data.joint_material_k,
        data.joint_penalty_kd,
        data.joint_sigma_start,
        data.joint_C_fric,
        data.joint_target_ke,
        data.joint_target_kd,
        joint_target_q,
        joint_target_qd,
        data.joint_limit_lower,
        data.joint_limit_upper,
        data.joint_limit_ke,
        data.joint_limit_kd,
        data.joint_drive_limit_support,
        data.joint_drive_lambda,
        data.joint_limit_lambda,
        data.joint_lambda_lin,
        data.joint_lambda_ang,
        data.joint_C0_lin,
        data.joint_C0_ang,
        data.joint_is_hard,
        data.alpha,
        data.compliant_alm,
        c.dof_dim,
        c,
        dt,
    )
    passive_f, passive_t, _P_ll, _P_al, _P_aa = _evaluate_joint_dissipation(
        child, joint, body_q, body_q_prev, c, data.joint_enabled, data.joint_friction, data.joint_damping, dt
    )
    wrench = wp.spatial_vector(force + passive_f, torque + passive_t)
    wrench += actuation
    joint_wrench[joint] = _to_child_joint_frame(c, joint, body_q, wrench)
    jt = c.joint_type[joint]
    if joint_effort and (jt == JointType.REVOLUTE or jt == JointType.PRISMATIC or jt == JointType.D6):
        recipient = joint
        ratio = float(1.0)
        reference = references[joint]
        if reference >= 0 and data.joint_enabled[reference]:
            recipient = reference
            ratio = coeffs[joint][1]
        armature_simulated = False
        if data.body_articulation_local and jt == JointType.REVOLUTE:
            armature_simulated = data.body_articulation_local[child] >= 0
        for component in range(c.dof_dim[joint, 0] + c.dof_dim[joint, 1]):
            dof = c.qd_start[joint] + component
            effort = _actuation_effort(c, joint, component, body_q, actuation)
            effort += _drive_effort(data, joint, component, body_q, body_q_prev, joint_target_q, joint_target_qd, dt)
            # Sparse revolute inertia already contributes to the solved drive
            # effort. Only estimate armature that the selected path ignores.
            if not armature_simulated:
                effort += joint_armature[dof] * (joint_qd[dof] - joint_qd_prev[dof]) / dt
            # Virtual work: a follower's effort contributes ratio * effort at
            # its reference. Keep follower entries zero instead of counting twice.
            wp.atomic_add(joint_effort, c.qd_start[recipient] + component, ratio * effort)


@wp.kernel
def add_mimic_joint_forces(
    coordinates: JointCoordinateData,
    joint_enabled: wp.array[bool],
    references: wp.array[int],
    coeffs: wp.array[wp.vec2],
    followers: wp.array[int],
    multipliers: wp.array[float],
    body_q: wp.array[wp.transform],
    joint_wrench: wp.array[wp.spatial_vector],
):
    follower = followers[wp.tid()]
    reference = references[follower]
    if reference < 0 or not joint_enabled[follower] or not joint_enabled[reference]:
        return
    follower_wrench = wp.spatial_vector()
    reference_wrench = wp.spatial_vector()
    for component in range(coordinates.dof_dim[follower, 0] + coordinates.dof_dim[follower, 1]):
        _q_f, _g_fp, g_fc = evaluate_coordinate(coordinates, follower, component, body_q)
        _q_r, _g_rp, g_rc = evaluate_coordinate(coordinates, reference, component, body_q)
        multiplier = multipliers[coordinates.qd_start[follower] + component]
        follower_wrench += multiplier * g_fc
        reference_wrench -= coeffs[follower][1] * multiplier * g_rc
    # Keep reactions per joint, even when serial joints share a body.
    wp.atomic_add(joint_wrench, follower, _to_child_joint_frame(coordinates, follower, body_q, follower_wrench))
    wp.atomic_add(joint_wrench, reference, _to_child_joint_frame(coordinates, reference, body_q, reference_wrench))
