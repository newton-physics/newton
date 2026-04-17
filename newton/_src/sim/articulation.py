# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warp as wp

from ..math import quat_decompose, transform_twist, velocity_at_point
from .enums import BodyFlags, JointType
from .model import Model
from .state import State


@wp.func
def com_twist_to_point_velocity(qd: wp.spatial_vector, X_wb: wp.transform, body_com: wp.vec3, point: wp.vec3):
    """Evaluate a point velocity from a COM-referenced body twist."""
    return velocity_at_point(qd, point - wp.transform_point(X_wb, body_com))


@wp.func
def origin_twist_to_com_twist(qd: wp.spatial_vector, X_wb: wp.transform, body_com: wp.vec3):
    """Shift an origin-referenced body twist to the body COM."""
    omega = wp.spatial_bottom(qd)
    v_com = velocity_at_point(qd, wp.transform_vector(X_wb, body_com))
    return wp.spatial_vector(v_com, omega)


@wp.func
def com_twist_to_origin_twist(qd: wp.spatial_vector, X_wb: wp.transform, body_com: wp.vec3):
    """Shift a COM-referenced body twist to the body origin."""
    omega = wp.spatial_bottom(qd)
    v_origin = wp.spatial_top(qd) - wp.cross(omega, wp.transform_vector(X_wb, body_com))
    return wp.spatial_vector(v_origin, omega)


@wp.func
def compute_2d_rotational_dofs(
    axis_0: wp.vec3,
    axis_1: wp.vec3,
    q0: float,
    q1: float,
    qd0: float,
    qd1: float,
):
    """
    Computes the rotation quaternion and 3D angular velocity given the joint axes, coordinates and velocities.
    """
    q_off = wp.quat_from_matrix(wp.matrix_from_cols(axis_0, axis_1, wp.cross(axis_0, axis_1)))

    # body local axes
    local_0 = wp.quat_rotate(q_off, wp.vec3(1.0, 0.0, 0.0))
    local_1 = wp.quat_rotate(q_off, wp.vec3(0.0, 1.0, 0.0))

    axis_0 = local_0
    q_0 = wp.quat_from_axis_angle(axis_0, q0)

    axis_1 = wp.quat_rotate(q_0, local_1)
    q_1 = wp.quat_from_axis_angle(axis_1, q1)

    rot = q_1 * q_0

    vel = axis_0 * qd0 + axis_1 * qd1

    return rot, vel


@wp.func
def invert_2d_rotational_dofs(
    axis_0: wp.vec3,
    axis_1: wp.vec3,
    q_p: wp.quat,
    q_c: wp.quat,
    w_err: wp.vec3,
):
    """
    Computes generalized joint position and velocity coordinates for a 2D rotational joint given the joint axes, relative orientations and angular velocity differences between the two bodies the joint connects.
    """
    q_off = wp.quat_from_matrix(wp.matrix_from_cols(axis_0, axis_1, wp.cross(axis_0, axis_1)))
    q_pc = wp.quat_inverse(q_off) * wp.quat_inverse(q_p) * q_c * q_off

    # decompose to a compound rotation each axis
    angles = quat_decompose(q_pc)

    # find rotation axes
    local_0 = wp.quat_rotate(q_off, wp.vec3(1.0, 0.0, 0.0))
    local_1 = wp.quat_rotate(q_off, wp.vec3(0.0, 1.0, 0.0))
    local_2 = wp.quat_rotate(q_off, wp.vec3(0.0, 0.0, 1.0))

    axis_0 = local_0
    q_0 = wp.quat_from_axis_angle(axis_0, angles[0])

    axis_1 = wp.quat_rotate(q_0, local_1)
    q_1 = wp.quat_from_axis_angle(axis_1, angles[1])

    axis_2 = wp.quat_rotate(q_1 * q_0, local_2)

    # convert angular velocity to local space
    w_err_p = wp.quat_rotate_inv(q_p, w_err)

    # given joint axes and angular velocity error, solve for joint velocities
    c12 = wp.cross(axis_1, axis_2)
    c02 = wp.cross(axis_0, axis_2)

    vel = wp.vec2(wp.dot(w_err_p, c12) / wp.dot(axis_0, c12), wp.dot(w_err_p, c02) / wp.dot(axis_1, c02))

    return wp.vec2(angles[0], angles[1]), vel


@wp.func
def compute_3d_rotational_dofs(
    axis_0: wp.vec3,
    axis_1: wp.vec3,
    axis_2: wp.vec3,
    q0: float,
    q1: float,
    q2: float,
    qd0: float,
    qd1: float,
    qd2: float,
):
    """
    Computes the rotation quaternion and 3D angular velocity given the joint axes, coordinates and velocities.
    """
    q_off = wp.quat_from_matrix(wp.matrix_from_cols(axis_0, axis_1, axis_2))

    # body local axes
    local_0 = wp.quat_rotate(q_off, wp.vec3(1.0, 0.0, 0.0))
    local_1 = wp.quat_rotate(q_off, wp.vec3(0.0, 1.0, 0.0))
    local_2 = wp.quat_rotate(q_off, wp.vec3(0.0, 0.0, 1.0))

    # reconstruct rotation axes
    axis_0 = local_0
    q_0 = wp.quat_from_axis_angle(axis_0, q0)

    axis_1 = wp.quat_rotate(q_0, local_1)
    q_1 = wp.quat_from_axis_angle(axis_1, q1)

    axis_2 = wp.quat_rotate(q_1 * q_0, local_2)
    q_2 = wp.quat_from_axis_angle(axis_2, q2)

    rot = q_2 * q_1 * q_0
    vel = axis_0 * qd0 + axis_1 * qd1 + axis_2 * qd2

    return rot, vel


@wp.func
def invert_3d_rotational_dofs(
    axis_0: wp.vec3, axis_1: wp.vec3, axis_2: wp.vec3, q_p: wp.quat, q_c: wp.quat, w_err: wp.vec3
):
    """
    Computes generalized joint position and velocity coordinates for a 3D rotational joint given the joint axes, relative orientations and angular velocity differences between the two bodies the joint connects.
    """
    q_off = wp.quat_from_matrix(wp.matrix_from_cols(axis_0, axis_1, axis_2))
    q_pc = wp.quat_inverse(q_off) * wp.quat_inverse(q_p) * q_c * q_off

    # decompose to a compound rotation each axis
    angles = quat_decompose(q_pc)

    # find rotation axes
    local_0 = wp.quat_rotate(q_off, wp.vec3(1.0, 0.0, 0.0))
    local_1 = wp.quat_rotate(q_off, wp.vec3(0.0, 1.0, 0.0))
    local_2 = wp.quat_rotate(q_off, wp.vec3(0.0, 0.0, 1.0))

    axis_0 = local_0
    q_0 = wp.quat_from_axis_angle(axis_0, angles[0])

    axis_1 = wp.quat_rotate(q_0, local_1)
    q_1 = wp.quat_from_axis_angle(axis_1, angles[1])

    axis_2 = wp.quat_rotate(q_1 * q_0, local_2)

    # convert angular velocity to local space
    w_err_p = wp.quat_rotate_inv(q_p, w_err)

    # given joint axes and angular velocity error, solve for joint velocities
    c12 = wp.cross(axis_1, axis_2)
    c02 = wp.cross(axis_0, axis_2)
    c01 = wp.cross(axis_0, axis_1)

    velocities = wp.vec3(
        wp.dot(w_err_p, c12) / wp.dot(axis_0, c12),
        wp.dot(w_err_p, c02) / wp.dot(axis_1, c02),
        wp.dot(w_err_p, c01) / wp.dot(axis_2, c01),
    )

    return angles, velocities


@wp.func
def eval_single_articulation_fk(
    joint_start: int,
    joint_end: int,
    joint_articulation: wp.array[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    body_com: wp.array[wp.vec3],
    body_flags: wp.array[wp.int32],
    body_flag_filter: int,
    # outputs
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    for i in range(joint_start, joint_end):
        articulation = joint_articulation[i]
        if articulation == -1:
            continue

        parent = joint_parent[i]
        child = joint_child[i]

        # compute transform across the joint
        type = joint_type[i]

        X_pj = joint_X_p[i]
        X_cj = joint_X_c[i]

        q_start = joint_q_start[i]
        qd_start = joint_qd_start[i]
        lin_axis_count = joint_dof_dim[i, 0]
        ang_axis_count = joint_dof_dim[i, 1]

        X_j = wp.transform_identity()
        v_j = wp.spatial_vector(wp.vec3(), wp.vec3())

        if type == JointType.PRISMATIC:
            axis = joint_axis[qd_start]

            q = joint_q[q_start]
            qd = joint_qd[qd_start]

            X_j = wp.transform(axis * q, wp.quat_identity())
            v_j = wp.spatial_vector(axis * qd, wp.vec3())

        if type == JointType.REVOLUTE:
            axis = joint_axis[qd_start]

            q = joint_q[q_start]
            qd = joint_qd[qd_start]

            X_j = wp.transform(wp.vec3(), wp.quat_from_axis_angle(axis, q))
            v_j = wp.spatial_vector(wp.vec3(), axis * qd)

        if type == JointType.BALL:
            r = wp.quat(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2], joint_q[q_start + 3])

            w = wp.vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2])

            X_j = wp.transform(wp.vec3(), r)
            v_j = wp.spatial_vector(wp.vec3(), w)

        if type == JointType.FREE or type == JointType.DISTANCE:
            t = wp.transform(
                wp.vec3(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2]),
                wp.quat(joint_q[q_start + 3], joint_q[q_start + 4], joint_q[q_start + 5], joint_q[q_start + 6]),
            )

            v = wp.spatial_vector(
                wp.vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2]),
                wp.vec3(joint_qd[qd_start + 3], joint_qd[qd_start + 4], joint_qd[qd_start + 5]),
            )

            X_j = t
            v_j = v

        if type == JointType.D6:
            pos = wp.vec3(0.0)
            rot = wp.quat_identity()
            vel_v = wp.vec3(0.0)
            vel_w = wp.vec3(0.0)

            # unroll for loop to ensure joint actions remain differentiable
            # (since differentiating through a for loop that updates a local variable is not supported)

            if lin_axis_count > 0:
                axis = joint_axis[qd_start + 0]
                pos += axis * joint_q[q_start + 0]
                vel_v += axis * joint_qd[qd_start + 0]
            if lin_axis_count > 1:
                axis = joint_axis[qd_start + 1]
                pos += axis * joint_q[q_start + 1]
                vel_v += axis * joint_qd[qd_start + 1]
            if lin_axis_count > 2:
                axis = joint_axis[qd_start + 2]
                pos += axis * joint_q[q_start + 2]
                vel_v += axis * joint_qd[qd_start + 2]

            iq = q_start + lin_axis_count
            iqd = qd_start + lin_axis_count
            if ang_axis_count == 1:
                axis = joint_axis[iqd]
                rot = wp.quat_from_axis_angle(axis, joint_q[iq])
                vel_w = joint_qd[iqd] * axis
            if ang_axis_count == 2:
                rot, vel_w = compute_2d_rotational_dofs(
                    joint_axis[iqd + 0],
                    joint_axis[iqd + 1],
                    joint_q[iq + 0],
                    joint_q[iq + 1],
                    joint_qd[iqd + 0],
                    joint_qd[iqd + 1],
                )
            if ang_axis_count == 3:
                rot, vel_w = compute_3d_rotational_dofs(
                    joint_axis[iqd + 0],
                    joint_axis[iqd + 1],
                    joint_axis[iqd + 2],
                    joint_q[iq + 0],
                    joint_q[iq + 1],
                    joint_q[iq + 2],
                    joint_qd[iqd + 0],
                    joint_qd[iqd + 1],
                    joint_qd[iqd + 2],
                )

            X_j = wp.transform(pos, rot)
            v_j = wp.spatial_vector(vel_v, vel_w)

        # transform from world to parent joint anchor frame
        X_wpj = X_pj
        if parent >= 0:
            X_wp = body_q[parent]
            X_wpj = X_wp * X_wpj

        # transform from world to joint anchor frame at child body
        X_wcj = X_wpj * X_j
        # transform from world to child body frame
        X_wc = X_wcj * wp.transform_inverse(X_cj)

        # Velocity must be evaluated at the actual child-body origin. For translated
        # joints, sampling parent motion only at the fixed
        # parent anchor misses the transport term from the current joint displacement.
        x_child_origin = wp.transform_get_translation(X_wc)
        v_parent_origin = wp.vec3()
        w_parent = wp.vec3()
        if parent >= 0:
            v_wp = body_qd[parent]
            w_parent = wp.spatial_bottom(v_wp)
            v_parent_origin = com_twist_to_point_velocity(v_wp, X_wp, body_com[parent], x_child_origin)

        # Transform joint motion into world space.
        linear_joint_world = wp.transform_vector(X_wpj, wp.spatial_top(v_j))
        angular_joint_world = wp.transform_vector(X_wpj, wp.spatial_bottom(v_j))
        if type == JointType.FREE or type == JointType.DISTANCE:
            # FREE / DISTANCE joint linear DOFs follow Newton's COM-velocity
            # convention, so convert the relative child COM twist to an
            # origin-referenced twist before the tree recurrence.
            v_joint_origin = com_twist_to_origin_twist(
                wp.spatial_vector(linear_joint_world, angular_joint_world),
                X_wc,
                body_com[child],
            )
            linear_joint_origin = wp.spatial_top(v_joint_origin)
        else:
            # The linear part of v_j is defined at the child joint anchor; if the
            # child body origin is offset from that anchor, transport the joint
            # angular motion to the body origin.
            child_origin_offset_world = x_child_origin - wp.transform_get_translation(X_wcj)
            linear_joint_origin = linear_joint_world + wp.cross(angular_joint_world, child_origin_offset_world)

        v_wc_origin = wp.spatial_vector(v_parent_origin + linear_joint_origin, w_parent + angular_joint_world)

        if (body_flags[child] & body_flag_filter) != 0:
            body_q[child] = X_wc
            body_qd[child] = origin_twist_to_com_twist(v_wc_origin, X_wc, body_com[child])


@wp.kernel
def eval_articulation_fk(
    articulation_start: wp.array[int],
    articulation_count: int,  # total number of articulations
    articulation_mask: wp.array[
        bool
    ],  # used to enable / disable FK for an articulation, if None then treat all as enabled
    articulation_indices: wp.array[int],  # can be None, articulation indices to process
    joint_articulation: wp.array[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    body_com: wp.array[wp.vec3],
    body_flags: wp.array[wp.int32],
    body_flag_filter: int,
    # outputs
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    tid = wp.tid()

    # Determine which articulation to process
    if articulation_indices:
        # Using indices - get actual articulation ID from array
        articulation_id = articulation_indices[tid]
    else:
        # No indices - articulation ID is just the thread index
        articulation_id = tid

    # Bounds check
    if articulation_id < 0 or articulation_id >= articulation_count:
        return  # Invalid articulation index

    # early out if disabling FK for this articulation
    if articulation_mask:
        if not articulation_mask[articulation_id]:
            return

    joint_start = articulation_start[articulation_id]
    joint_end = articulation_start[articulation_id + 1]

    eval_single_articulation_fk(
        joint_start,
        joint_end,
        joint_articulation,
        joint_q,
        joint_qd,
        joint_q_start,
        joint_qd_start,
        joint_type,
        joint_parent,
        joint_child,
        joint_X_p,
        joint_X_c,
        joint_axis,
        joint_dof_dim,
        body_com,
        body_flags,
        body_flag_filter,
        # outputs
        body_q,
        body_qd,
    )


def eval_fk(
    model: Model,
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    state: State | Model | object,
    mask: wp.array[bool] | None = None,
    indices: wp.array[int] | None = None,
    body_flag_filter: int = BodyFlags.ALL,
):
    """
    Evaluates the model's forward kinematics given the joint coordinates and updates the state's body information (:attr:`State.body_q` and :attr:`State.body_qd`).

    The written :attr:`State.body_qd` values use Newton's public body-twist
    convention ``(v_com_world, omega_world)``.

    Args:
        model: The model to evaluate.
        joint_q: Generalized joint position coordinates, shape [joint_coord_count], float
        joint_qd: Generalized joint velocity coordinates, shape [joint_dof_count], float
        state: The state-like target to update (e.g., :class:`State` or :class:`Model`).
        mask: The mask to use to enable / disable FK for an articulation. If None then treat all as enabled, shape [articulation_count], bool
        indices: Integer indices of articulations to update. If None, updates all articulations.
            Cannot be used together with mask parameter.
        body_flag_filter: Body flag filter controlling which bodies are written to in ``state.body_q`` and
            ``state.body_qd``. Default updates both dynamic and kinematic bodies. Bodies that do not
            match the filter retain their existing values; they are not zeroed or invalidated.
    """
    # Validate inputs
    if mask is not None and indices is not None:
        raise ValueError("Cannot specify both mask and indices parameters")

    # Determine launch dimensions
    if indices is not None:
        num_articulations = len(indices)
    else:
        num_articulations = model.articulation_count

    wp.launch(
        kernel=eval_articulation_fk,
        dim=num_articulations,
        inputs=[
            model.articulation_start,
            model.articulation_count,
            mask,
            indices,
            model.joint_articulation,
            joint_q,
            joint_qd,
            model.joint_q_start,
            model.joint_qd_start,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            model.joint_dof_dim,
            model.body_com,
            model.body_flags,
            body_flag_filter,
        ],
        outputs=[
            state.body_q,
            state.body_qd,
        ],
        device=model.device,
    )


@wp.kernel
def compute_shape_world_transforms(
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    # outputs
    shape_world_transform: wp.array[wp.transform],
):
    """Compute world-space transforms for shapes by concatenating local shape
    transforms with body transforms.

    Args:
        shape_transform: Local shape transforms in body frame,
            shape [shape_count, 7]
        shape_body: Body index for each shape, shape [shape_count]
        body_q: Body transforms in world frame, shape [body_count, 7]
        shape_world_transform: Output world transforms for shapes,
            shape [shape_count, 7]
    """
    shape_idx = wp.tid()

    # Get the local shape transform
    X_bs = shape_transform[shape_idx]

    # Get the body index for this shape
    body_idx = shape_body[shape_idx]

    # If shape is attached to a body (body_idx >= 0), concatenate transforms
    if body_idx >= 0:
        # Get the body transform in world space
        X_wb = body_q[body_idx]

        # Concatenate: world_transform = body_transform * shape_transform
        X_ws = wp.transform_multiply(X_wb, X_bs)
        shape_world_transform[shape_idx] = X_ws
    else:
        # Shape is not attached to a body (static shape), use local
        # transform as world transform
        shape_world_transform[shape_idx] = X_bs


@wp.func
def reconstruct_angular_q_qd(q_pc: wp.quat, w_err: wp.vec3, X_wp: wp.transform, axis: wp.vec3):
    """
    Reconstructs the angular joint coordinates and velocities given the relative rotation and angular velocity
    between a parent and child body.

    Args:
        q_pc (quat): The relative rotation between the parent and child body.
        w_err (vec3): The angular velocity between the parent and child body.
        X_wp (transform): The transform from the parent body frame to the joint parent anchor frame.
        axis (vec3): The joint axis in the joint parent anchor frame.

    Returns:
        q (float): The joint position coordinate.
        qd (float): The joint velocity coordinate.
    """
    axis_p = wp.transform_vector(X_wp, axis)
    twist = wp.quat_twist(axis, q_pc)
    q = wp.acos(twist[3]) * 2.0 * wp.sign(wp.dot(axis, wp.vec3(twist[0], twist[1], twist[2])))
    qd = wp.dot(w_err, axis_p)
    return q, qd


@wp.kernel
def eval_articulation_ik(
    articulation_start: wp.array[int],
    articulation_count: int,  # total number of articulations
    articulation_mask: wp.array[bool],  # can be None, mask to filter articulations
    articulation_indices: wp.array[int],  # can be None, articulation indices to process
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    body_flags: wp.array[wp.int32],
    body_flag_filter: int,
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    art_idx, joint_offset = wp.tid()  # articulation index and joint offset within articulation

    # Determine which articulation to process
    if articulation_indices:
        articulation_id = articulation_indices[art_idx]
    else:
        articulation_id = art_idx

    # Bounds check
    if articulation_id < 0 or articulation_id >= articulation_count:
        return  # Invalid articulation index

    # early out if disabling IK for this articulation
    if articulation_mask:
        if not articulation_mask[articulation_id]:
            return

    # Get joint range for this articulation
    joint_start = articulation_start[articulation_id]
    joint_end = articulation_start[articulation_id + 1]

    # Check if this thread has a valid joint to process
    joint_idx = joint_start + joint_offset
    if joint_idx >= joint_end:
        return  # This thread has no joint (padding thread)

    parent = joint_parent[joint_idx]
    child = joint_child[joint_idx]
    if (body_flags[child] & body_flag_filter) == 0:
        return

    X_pj = joint_X_p[joint_idx]
    X_cj = joint_X_c[joint_idx]

    w_p = wp.vec3()
    v_p = wp.vec3()
    v_wp = wp.spatial_vector()

    # parent anchor frame in world space
    X_wpj = X_pj
    if parent >= 0:
        X_wp = body_q[parent]
        X_wpj = X_wp * X_pj

        v_wp = body_qd[parent]
        w_p = wp.spatial_bottom(v_wp)
        v_p = com_twist_to_point_velocity(v_wp, X_wp, body_com[parent], wp.transform_get_translation(X_wpj))

    # child transform and moment arm
    X_wc = body_q[child]
    X_wcj = X_wc * X_cj

    v_wc = body_qd[child]

    w_c = wp.spatial_bottom(v_wc)
    v_c = com_twist_to_point_velocity(v_wc, X_wc, body_com[child], wp.transform_get_translation(X_wcj))

    # joint properties
    type = joint_type[joint_idx]

    # compute position and orientation differences between anchor frames
    x_p = wp.transform_get_translation(X_wpj)
    x_c = wp.transform_get_translation(X_wcj)

    q_p = wp.transform_get_rotation(X_wpj)
    q_c = wp.transform_get_rotation(X_wcj)

    x_err = x_c - x_p
    v_err = v_c - v_p
    w_err = w_c - w_p

    q_start = joint_q_start[joint_idx]
    qd_start = joint_qd_start[joint_idx]
    lin_axis_count = joint_dof_dim[joint_idx, 0]
    ang_axis_count = joint_dof_dim[joint_idx, 1]

    if type == JointType.PRISMATIC:
        axis = joint_axis[qd_start]

        # world space joint axis
        axis_p = wp.quat_rotate(q_p, axis)

        # evaluate joint coordinates
        q = wp.dot(x_err, axis_p)
        qd = wp.dot(v_err, axis_p)

        joint_q[q_start] = q
        joint_qd[qd_start] = qd

        return

    if type == JointType.REVOLUTE:
        axis = joint_axis[qd_start]
        q_pc = wp.quat_inverse(q_p) * q_c

        q, qd = reconstruct_angular_q_qd(q_pc, w_err, X_wpj, axis)

        joint_q[q_start] = q
        joint_qd[qd_start] = qd

        return

    if type == JointType.BALL:
        q_pc = wp.quat_inverse(q_p) * q_c

        joint_q[q_start + 0] = q_pc[0]
        joint_q[q_start + 1] = q_pc[1]
        joint_q[q_start + 2] = q_pc[2]
        joint_q[q_start + 3] = q_pc[3]

        ang_vel = wp.transform_vector(wp.transform_inverse(X_wpj), w_err)
        joint_qd[qd_start + 0] = ang_vel[0]
        joint_qd[qd_start + 1] = ang_vel[1]
        joint_qd[qd_start + 2] = ang_vel[2]

        return

    if type == JointType.FIXED:
        return

    if type == JointType.FREE or type == JointType.DISTANCE:
        q_pc = wp.quat_inverse(q_p) * q_c

        x_err_c = wp.quat_rotate_inv(q_p, x_err)
        x_child_com_world = wp.transform_point(X_wc, body_com[child])
        v_com_err = wp.spatial_top(v_wc)
        if parent >= 0:
            v_com_err = v_com_err - com_twist_to_point_velocity(v_wp, X_wp, body_com[parent], x_child_com_world)
        v_err_c = wp.quat_rotate_inv(q_p, v_com_err)
        w_err_c = wp.quat_rotate_inv(q_p, w_err)

        joint_q[q_start + 0] = x_err_c[0]
        joint_q[q_start + 1] = x_err_c[1]
        joint_q[q_start + 2] = x_err_c[2]

        joint_q[q_start + 3] = q_pc[0]
        joint_q[q_start + 4] = q_pc[1]
        joint_q[q_start + 5] = q_pc[2]
        joint_q[q_start + 6] = q_pc[3]

        joint_qd[qd_start + 0] = v_err_c[0]
        joint_qd[qd_start + 1] = v_err_c[1]
        joint_qd[qd_start + 2] = v_err_c[2]

        joint_qd[qd_start + 3] = w_err_c[0]
        joint_qd[qd_start + 4] = w_err_c[1]
        joint_qd[qd_start + 5] = w_err_c[2]

        return

    if type == JointType.D6:
        x_err_c = wp.quat_rotate_inv(q_p, x_err)
        v_err_c = wp.quat_rotate_inv(q_p, v_err)
        if lin_axis_count > 0:
            axis = joint_axis[qd_start + 0]
            joint_q[q_start + 0] = wp.dot(x_err_c, axis)
            joint_qd[qd_start + 0] = wp.dot(v_err_c, axis)

        if lin_axis_count > 1:
            axis = joint_axis[qd_start + 1]
            joint_q[q_start + 1] = wp.dot(x_err_c, axis)
            joint_qd[qd_start + 1] = wp.dot(v_err_c, axis)

        if lin_axis_count > 2:
            axis = joint_axis[qd_start + 2]
            joint_q[q_start + 2] = wp.dot(x_err_c, axis)
            joint_qd[qd_start + 2] = wp.dot(v_err_c, axis)

        if ang_axis_count == 1:
            axis = joint_axis[qd_start]
            q_pc = wp.quat_inverse(q_p) * q_c
            q, qd = reconstruct_angular_q_qd(q_pc, w_err, X_wpj, joint_axis[qd_start + lin_axis_count])
            joint_q[q_start + lin_axis_count] = q
            joint_qd[qd_start + lin_axis_count] = qd

        if ang_axis_count == 2:
            axis_0 = joint_axis[qd_start + lin_axis_count + 0]
            axis_1 = joint_axis[qd_start + lin_axis_count + 1]
            qs2, qds2 = invert_2d_rotational_dofs(axis_0, axis_1, q_p, q_c, w_err)
            joint_q[q_start + lin_axis_count + 0] = qs2[0]
            joint_q[q_start + lin_axis_count + 1] = qs2[1]
            joint_qd[qd_start + lin_axis_count + 0] = qds2[0]
            joint_qd[qd_start + lin_axis_count + 1] = qds2[1]

        if ang_axis_count == 3:
            axis_0 = joint_axis[qd_start + lin_axis_count + 0]
            axis_1 = joint_axis[qd_start + lin_axis_count + 1]
            axis_2 = joint_axis[qd_start + lin_axis_count + 2]
            qs3, qds3 = invert_3d_rotational_dofs(axis_0, axis_1, axis_2, q_p, q_c, w_err)
            joint_q[q_start + lin_axis_count + 0] = qs3[0]
            joint_q[q_start + lin_axis_count + 1] = qs3[1]
            joint_q[q_start + lin_axis_count + 2] = qs3[2]
            joint_qd[qd_start + lin_axis_count + 0] = qds3[0]
            joint_qd[qd_start + lin_axis_count + 1] = qds3[1]
            joint_qd[qd_start + lin_axis_count + 2] = qds3[2]

        return


# given maximal coordinate model computes ik (closest point projection)
def eval_ik(
    model: Model,
    state: State | Model | object,
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    mask: wp.array[bool] | None = None,
    indices: wp.array[int] | None = None,
    body_flag_filter: int = BodyFlags.ALL,
):
    """
    Evaluates the model's inverse kinematics given the state's body information (:attr:`State.body_q` and :attr:`State.body_qd`) and updates the generalized joint coordinates `joint_q` and `joint_qd`.

    The input :attr:`State.body_qd` is interpreted using Newton's public body-twist
    convention ``(v_com_world, omega_world)``. For FREE and DISTANCE joints,
    the recovered ``joint_qd`` linear entries are referenced at the child COM
    and expressed in the joint parent frame.

    Args:
        model: The model to evaluate.
        state: The state-like object with the body's maximal coordinates (positions :attr:`State.body_q` and velocities :attr:`State.body_qd`) to use.
        joint_q: Generalized joint position coordinates, shape [joint_coord_count], float
        joint_qd: Generalized joint velocity coordinates, shape [joint_dof_count], float
        mask: Boolean mask indicating which articulations to update. If None, updates all (or those specified by indices).
        indices: Integer indices of articulations to update. If None, updates all articulations.
        body_flag_filter: Body flag filter controlling which joints are written based on each joint's child
            body flag. Default updates joints for both dynamic and kinematic child bodies. Entries that
            do not match the filter retain their existing values in ``joint_q`` and ``joint_qd``.

    Note:
        The mask and indices parameters are mutually exclusive. If both are provided, a ValueError is raised.
    """
    # Check for mutually exclusive parameters
    if mask is not None and indices is not None:
        raise ValueError("mask and indices parameters are mutually exclusive - please use only one")

    # Determine launch dimensions
    if indices is not None:
        num_articulations = len(indices)
    else:
        num_articulations = model.articulation_count

    # Always use 2D launch for joint-level parallelism
    wp.launch(
        kernel=eval_articulation_ik,
        dim=(num_articulations, model.max_joints_per_articulation),
        inputs=[
            model.articulation_start,
            model.articulation_count,
            mask,
            indices,
            state.body_q,
            state.body_qd,
            model.body_com,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            model.joint_dof_dim,
            model.joint_q_start,
            model.joint_qd_start,
            model.body_flags,
            body_flag_filter,
        ],
        outputs=[joint_q, joint_qd],
        device=model.device,
    )


@wp.func
def jcalc_motion_subspace(
    type: int,
    joint_axis: wp.array[wp.vec3],
    lin_axis_count: int,
    ang_axis_count: int,
    X_sc: wp.transform,
    X_wc: wp.transform,
    body_com_child: wp.vec3,
    qd_start: int,
    # outputs
    joint_S_s: wp.array[wp.spatial_vector],
):
    """Compute motion subspace (joint Jacobian columns) for a joint.

    This populates joint_S_s with the motion subspace vectors for each DoF,
    which represent how each joint coordinate affects the spatial velocity.

    Note:
        CABLE joints are not currently supported. CABLE joints have complex,
        configuration-dependent motion subspaces (dynamic stretch direction and
        isotropic angular DOF) and are primarily designed for VBD solver.
        If encountered, their Jacobian columns will remain zero.
    """
    if type == JointType.PRISMATIC:
        axis = joint_axis[qd_start]
        S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
        joint_S_s[qd_start] = S_s

    elif type == JointType.REVOLUTE:
        axis = joint_axis[qd_start]
        S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
        joint_S_s[qd_start] = S_s

    elif type == JointType.D6:
        if lin_axis_count > 0:
            axis = joint_axis[qd_start + 0]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            joint_S_s[qd_start + 0] = S_s
        if lin_axis_count > 1:
            axis = joint_axis[qd_start + 1]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            joint_S_s[qd_start + 1] = S_s
        if lin_axis_count > 2:
            axis = joint_axis[qd_start + 2]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            joint_S_s[qd_start + 2] = S_s
        if ang_axis_count > 0:
            axis = joint_axis[qd_start + lin_axis_count + 0]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            joint_S_s[qd_start + lin_axis_count + 0] = S_s
        if ang_axis_count > 1:
            axis = joint_axis[qd_start + lin_axis_count + 1]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            joint_S_s[qd_start + lin_axis_count + 1] = S_s
        if ang_axis_count > 2:
            axis = joint_axis[qd_start + lin_axis_count + 2]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            joint_S_s[qd_start + lin_axis_count + 2] = S_s

    elif type == JointType.BALL:
        S_0 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        S_1 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        S_2 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
        joint_S_s[qd_start + 0] = S_0
        joint_S_s[qd_start + 1] = S_1
        joint_S_s[qd_start + 2] = S_2

    elif type == JointType.FREE or type == JointType.DISTANCE:
        x_child_com_world = wp.transform_point(X_wc, body_com_child)
        axis_world_x = wp.transform_vector(X_sc, wp.vec3(1.0, 0.0, 0.0))
        axis_world_y = wp.transform_vector(X_sc, wp.vec3(0.0, 1.0, 0.0))
        axis_world_z = wp.transform_vector(X_sc, wp.vec3(0.0, 0.0, 1.0))

        joint_S_s[qd_start + 0] = wp.spatial_vector(axis_world_x, wp.vec3())
        joint_S_s[qd_start + 1] = wp.spatial_vector(axis_world_y, wp.vec3())
        joint_S_s[qd_start + 2] = wp.spatial_vector(axis_world_z, wp.vec3())
        joint_S_s[qd_start + 3] = wp.spatial_vector(-wp.cross(axis_world_x, x_child_com_world), axis_world_x)
        joint_S_s[qd_start + 4] = wp.spatial_vector(-wp.cross(axis_world_y, x_child_com_world), axis_world_y)
        joint_S_s[qd_start + 5] = wp.spatial_vector(-wp.cross(axis_world_z, x_child_com_world), axis_world_z)


@wp.kernel
def eval_articulation_jacobian(
    articulation_start: wp.array[int],
    articulation_count: int,
    articulation_mask: wp.array[bool],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    # outputs
    J: wp.array3d[float],
    joint_S_s: wp.array[wp.spatial_vector],
):
    """Compute spatial Jacobian for articulations.

    The Jacobian J maps joint velocities to spatial velocities of each link.
    Output shape: (articulation_count, max_links*6, max_dofs)
    """
    art_idx = wp.tid()

    if art_idx >= articulation_count:
        return

    if articulation_mask:
        if not articulation_mask[art_idx]:
            return

    joint_start = articulation_start[art_idx]
    joint_end = articulation_start[art_idx + 1]
    joint_count = joint_end - joint_start

    articulation_dof_start = joint_qd_start[joint_start]

    # First pass: compute body transforms and motion subspaces
    for i in range(joint_count):
        j = joint_start + i
        parent = joint_parent[j]
        type = joint_type[j]

        X_pj = joint_X_p[j]

        # parent anchor frame in world space
        X_wpj = X_pj
        if parent >= 0:
            X_wp = body_q[parent]
            X_wpj = X_wp * X_pj

        qd_start = joint_qd_start[j]
        lin_axis_count = joint_dof_dim[j, 0]
        ang_axis_count = joint_dof_dim[j, 1]

        jcalc_motion_subspace(
            type,
            joint_axis,
            lin_axis_count,
            ang_axis_count,
            X_wpj,
            body_q[joint_child[j]],
            body_com[joint_child[j]],
            qd_start,
            joint_S_s,
        )

    # Second pass: build Jacobian by walking kinematic chain
    for i in range(joint_count):
        row_start = i * 6

        j = joint_start + i
        child = joint_child[j]
        x_com_world = wp.transform_point(body_q[child], body_com[child])
        while j != -1:
            joint_dof_start = joint_qd_start[j]
            joint_dof_end = joint_qd_start[j + 1]
            joint_dof_count = joint_dof_end - joint_dof_start

            # Fill out each row of the Jacobian walking up the tree
            for dof in range(joint_dof_count):
                col = (joint_dof_start - articulation_dof_start) + dof
                S = joint_S_s[joint_dof_start + dof]
                S_com = wp.spatial_vector(velocity_at_point(S, x_com_world), wp.spatial_bottom(S))

                for k in range(6):
                    J[art_idx, row_start + k, col] = S_com[k]

            j = joint_ancestor[j]


def eval_jacobian(
    model: Model,
    state: State,
    J: wp.array | None = None,
    joint_S_s: wp.array | None = None,
    mask: wp.array | None = None,
) -> wp.array | None:
    """Evaluate spatial Jacobian for articulations.

    Computes the spatial Jacobian J that maps joint velocities to spatial
    velocities of each link in world frame. The Jacobian is computed for
    each articulation in the model and satisfies ``J_link @ joint_qd ==
    state.body_qd[link]`` under Newton's public COM/world body-twist convention.

    Args:
        model: The model containing articulation definitions.
        state: The state containing body transforms (body_q).
        J: Optional output array for the Jacobian, shape (articulation_count, max_links*6, max_dofs).
           If None, allocates internally.
        joint_S_s: Optional pre-allocated temp array for motion subspaces,
                   shape (joint_dof_count,), dtype wp.spatial_vector.
                   If None, allocates internally.
        mask: Optional boolean mask to select which articulations to compute.
              Shape [articulation_count]. If None, computes for all articulations.

    Returns:
        The Jacobian array J, or None if the model has no articulations.
    """
    if model.articulation_count == 0:
        return None

    # Allocate output if not provided
    if J is None:
        max_links = model.max_joints_per_articulation
        max_dofs = model.max_dofs_per_articulation
        J = wp.empty(
            (model.articulation_count, max_links * 6, max_dofs),
            dtype=float,
            device=model.device,
        )

    # Zero the output buffer
    J.zero_()

    # Allocate temp if not provided
    if joint_S_s is None:
        joint_S_s = wp.zeros(
            model.joint_dof_count,
            dtype=wp.spatial_vector,
            device=model.device,
        )

    wp.launch(
        kernel=eval_articulation_jacobian,
        dim=model.articulation_count,
        inputs=[
            model.articulation_start,
            model.articulation_count,
            mask,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_ancestor,
            model.joint_qd_start,
            model.joint_X_p,
            model.joint_axis,
            model.joint_dof_dim,
            state.body_q,
            model.body_com,
        ],
        outputs=[J, joint_S_s],
        device=model.device,
    )

    return J


@wp.func
def transform_spatial_inertia(t: wp.transform, I: wp.spatial_matrix):
    """Transform a spatial inertia tensor to a new coordinate frame.

    Note: This is duplicated from featherstone/kernels.py to avoid circular imports.
    """
    t_inv = wp.transform_inverse(t)

    q = wp.transform_get_rotation(t_inv)
    p = wp.transform_get_translation(t_inv)

    r1 = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    r2 = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))
    r3 = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))

    R = wp.matrix_from_cols(r1, r2, r3)
    S = wp.skew(p) @ R

    # fmt: off
    T = wp.spatial_matrix(
        R[0, 0], R[0, 1], R[0, 2], S[0, 0], S[0, 1], S[0, 2],
        R[1, 0], R[1, 1], R[1, 2], S[1, 0], S[1, 1], S[1, 2],
        R[2, 0], R[2, 1], R[2, 2], S[2, 0], S[2, 1], S[2, 2],
        0.0, 0.0, 0.0, R[0, 0], R[0, 1], R[0, 2],
        0.0, 0.0, 0.0, R[1, 0], R[1, 1], R[1, 2],
        0.0, 0.0, 0.0, R[2, 0], R[2, 1], R[2, 2],
    )
    # fmt: on

    return wp.mul(wp.mul(wp.transpose(T), I), T)


@wp.kernel
def compute_body_spatial_inertia(
    body_inertia: wp.array[wp.mat33],
    body_mass: wp.array[float],
    body_q: wp.array[wp.transform],
    # outputs
    body_I_s: wp.array[wp.spatial_matrix],
):
    """Compute COM-referenced spatial inertia for each body in world frame."""
    tid = wp.tid()

    I_local = body_inertia[tid]
    m = body_mass[tid]
    X_wb = body_q[tid]
    q = wp.transform_get_rotation(X_wb)

    r1 = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    r2 = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))
    r3 = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))
    R = wp.matrix_from_cols(r1, r2, r3)
    I_world = R * I_local * wp.transpose(R)

    # Spatial inertia about the COM in world coordinates.
    # fmt: off
    body_I_s[tid] = wp.spatial_matrix(
        m,   0.0, 0.0, 0.0,           0.0,           0.0,
        0.0, m,   0.0, 0.0,           0.0,           0.0,
        0.0, 0.0, m,   0.0,           0.0,           0.0,
        0.0, 0.0, 0.0, I_world[0, 0], I_world[0, 1], I_world[0, 2],
        0.0, 0.0, 0.0, I_world[1, 0], I_world[1, 1], I_world[1, 2],
        0.0, 0.0, 0.0, I_world[2, 0], I_world[2, 1], I_world[2, 2],
    )
    # fmt: on


@wp.kernel
def eval_articulation_mass_matrix(
    articulation_start: wp.array[int],
    articulation_count: int,
    articulation_mask: wp.array[bool],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    body_I_s: wp.array[wp.spatial_matrix],
    J: wp.array3d[float],
    # outputs
    H: wp.array3d[float],
):
    """Compute generalized mass matrix H = J^T * M * J.

    The mass matrix H relates joint accelerations to joint forces/torques.
    Output shape: (articulation_count, max_dofs, max_dofs)
    """
    art_idx = wp.tid()

    if art_idx >= articulation_count:
        return

    if articulation_mask:
        if not articulation_mask[art_idx]:
            return

    joint_start = articulation_start[art_idx]
    joint_end = articulation_start[art_idx + 1]
    joint_count = joint_end - joint_start

    articulation_dof_start = joint_qd_start[joint_start]
    articulation_dof_end = joint_qd_start[joint_end]
    articulation_dof_count = articulation_dof_end - articulation_dof_start

    # H = J^T * M * J
    # M is block diagonal with 6x6 spatial inertia blocks
    # We compute this as: for each link i, H += J_i^T * I_i * J_i

    for link_idx in range(joint_count):
        j = joint_start + link_idx
        child = joint_child[j]
        I_s = body_I_s[child]

        row_start = link_idx * 6

        # Compute contribution from this link: H += J_i^T * I_i * J_i
        for dof_i in range(articulation_dof_count):
            for dof_j in range(articulation_dof_count):
                sum_val = float(0.0)

                # J_i^T * I_i * J_j (for the 6 rows of this link)
                for k in range(6):
                    for l in range(6):
                        J_ik = J[art_idx, row_start + k, dof_i]
                        J_jl = J[art_idx, row_start + l, dof_j]
                        sum_val += J_ik * I_s[k, l] * J_jl

                H[art_idx, dof_i, dof_j] = H[art_idx, dof_i, dof_j] + sum_val


def eval_mass_matrix(
    model: Model,
    state: State,
    H: wp.array | None = None,
    J: wp.array | None = None,
    body_I_s: wp.array | None = None,
    joint_S_s: wp.array | None = None,
    mask: wp.array | None = None,
) -> wp.array | None:
    """Evaluate generalized mass matrix for articulations.

    Computes the generalized mass matrix H = J^T * M * J, where J is the spatial
    Jacobian and M is the block-diagonal spatial mass matrix. The mass matrix
    relates joint accelerations to joint forces/torques and is consistent with
    kinetic energy computed from COM-referenced body twists.

    Args:
        model: The model containing articulation definitions.
        state: The state containing body transforms (body_q).
        H: Optional output array for mass matrix, shape (articulation_count, max_dofs, max_dofs).
           If None, allocates internally.
        J: Optional pre-computed Jacobian. If None, computes internally.
           Shape (articulation_count, max_links*6, max_dofs).
        body_I_s: Optional pre-allocated temp array for spatial inertias,
                  shape (body_count,), dtype wp.spatial_matrix. If None, allocates internally.
        joint_S_s: Optional pre-allocated temp array for motion subspaces (only used if J is None),
                   shape (joint_dof_count,), dtype wp.spatial_vector. If None, allocates internally.
        mask: Optional boolean mask to select which articulations to compute.
              Shape [articulation_count]. If None, computes for all articulations.

    Returns:
        The mass matrix array H, or None if the model has no articulations.
    """
    if model.articulation_count == 0:
        return None

    # Allocate output if not provided
    if H is None:
        max_dofs = model.max_dofs_per_articulation
        H = wp.empty(
            (model.articulation_count, max_dofs, max_dofs),
            dtype=float,
            device=model.device,
        )

    # Zero the output buffer
    H.zero_()

    # Allocate or use provided body_I_s
    if body_I_s is None:
        body_I_s = wp.zeros(
            model.body_count,
            dtype=wp.spatial_matrix,
            device=model.device,
        )

    # Compute spatial inertias in world frame
    wp.launch(
        kernel=compute_body_spatial_inertia,
        dim=model.body_count,
        inputs=[
            model.body_inertia,
            model.body_mass,
            state.body_q,
        ],
        outputs=[body_I_s],
        device=model.device,
    )

    # Compute Jacobian if not provided
    if J is None:
        max_links = model.max_joints_per_articulation
        max_dofs = model.max_dofs_per_articulation
        J = wp.zeros(
            (model.articulation_count, max_links * 6, max_dofs),
            dtype=float,
            device=model.device,
        )
        eval_jacobian(model, state, J, joint_S_s=joint_S_s, mask=mask)

    wp.launch(
        kernel=eval_articulation_mass_matrix,
        dim=model.articulation_count,
        inputs=[
            model.articulation_start,
            model.articulation_count,
            mask,
            model.joint_child,
            model.joint_qd_start,
            body_I_s,
            J,
        ],
        outputs=[H],
        device=model.device,
    )

    return H
