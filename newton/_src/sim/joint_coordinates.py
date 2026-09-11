# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pose-derived joint coordinates and their maximal-coordinate gradients."""

import warp as wp

from .articulation import (
    invert_2d_rotational_dofs,
    invert_3d_rotational_dofs,
    transform_3d_rotational_axes,
)
from .enums import JointType


@wp.func
def unwrap_joint_coordinate(coordinate: float, previous: float, component: int, linear_count: int):
    """Lift an angular coordinate to the revolution nearest its previous value.

    Linear coordinates pass through unchanged. The caller owns the continuous
    history and must sample angular motion in increments smaller than pi.
    Quaternion sign changes do not affect the result.
    """
    if component >= linear_count:
        delta = coordinate - previous
        return previous + wp.atan2(wp.sin(delta), wp.cos(delta))
    return coordinate


@wp.func
def _twist_coordinate_gradient(axis: wp.vec3, rotation: wp.quat):
    """Differentiate the signed twist angle, including off-axis swing."""
    v = wp.vec3(rotation[0], rotation[1], rotation[2])
    w = rotation[3]
    s = wp.dot(axis, v)
    denominator = w * w + s * s
    if denominator <= 1.0e-12:
        return axis  # Twist is undefined at a 180-degree orthogonal swing.
    return (w * w * axis + w * wp.cross(v, axis) + s * v) / denominator


@wp.func
def _euler_coordinate_gradient(a0: wp.vec3, a1: wp.vec3, a2: wp.vec3, component: int):
    """Return a row of the inverse angular Jacobian, not a rotation axis."""
    numerator = wp.cross(a1, a2)
    axis = a0
    if component == 1:
        numerator = wp.cross(a2, a0)
        axis = a1
    elif component == 2:
        numerator = wp.cross(a0, a1)
        axis = a2
    denominator = wp.dot(axis, numerator)
    # Euler coordinates are singular at gimbal lock. Bound the inverse there.
    if wp.abs(denominator) < 1.0e-6:
        denominator = wp.where(denominator < 0.0, -1.0e-6, 1.0e-6)
    return numerator / denominator


@wp.func
def _two_axis_coordinate(axis_0: wp.vec3, axis_1: wp.vec3, rotation: wp.quat, component: int):
    """Recover two orthogonal rotations without the artificial third Euler axis."""
    basis = wp.quat_from_matrix(wp.matrix_from_cols(axis_0, axis_1, wp.cross(axis_0, axis_1)))
    R = wp.quat_to_matrix(wp.quat_inverse(basis) * rotation * basis)
    # For R = Rx(q0) Ry(q1), column Y determines q0 and row X determines q1.
    # These atan2 coordinates remain regular when q1 crosses +/-pi/2.
    angle = wp.atan2(R[2, 1], R[1, 1])
    denominator = wp.max(R[1, 1] * R[1, 1] + R[2, 1] * R[2, 1], 1.0e-12)
    gradient = wp.vec3(1.0, -R[0, 1] * R[1, 1] / denominator, -R[0, 1] * R[2, 1] / denominator)
    if component == 1:
        angle = wp.atan2(R[0, 2], R[0, 0])
        denominator = wp.max(R[0, 0] * R[0, 0] + R[0, 2] * R[0, 2], 1.0e-12)
        gradient = wp.vec3(
            0.0,
            (R[0, 0] * R[2, 2] - R[0, 2] * R[2, 0]) / denominator,
            (R[0, 2] * R[1, 0] - R[0, 0] * R[1, 2]) / denominator,
        )
    return angle, wp.quat_rotate(basis, gradient)


@wp.func
def _continuous_euler_angles(angles: wp.vec3, previous: wp.vec3, angular_count: int):
    """Choose the equivalent Euler branch nearest the previous coordinates."""
    direct = wp.vec3()
    alternate = wp.vec3()
    other = wp.vec3(angles[0] + wp.pi, wp.pi - angles[1], angles[2] + wp.pi)
    direct_distance, alternate_distance = float(0.0), float(0.0)
    for axis in range(angular_count):
        direct[axis] = unwrap_joint_coordinate(angles[axis], previous[axis], 0, 0)
        alternate[axis] = unwrap_joint_coordinate(other[axis], previous[axis], 0, 0)
        direct_distance += (direct[axis] - previous[axis]) ** 2.0
        alternate_distance += (alternate[axis] - previous[axis]) ** 2.0
    if alternate_distance < direct_distance:
        return alternate
    return direct


@wp.func
def eval_joint_coordinate(
    joint: int,
    component: int,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_axis: wp.array[wp.vec3],
    angular_reference: wp.vec3,
    continuous: bool,
):
    """Return one joint coordinate and its parent/child maximal-coordinate gradients."""
    type = joint_type[joint]
    parent = joint_parent[joint]
    child = joint_child[joint]

    X_wp = joint_X_p[joint]
    pose_p = X_wp
    if parent >= 0:
        pose_p = body_q[parent]
        X_wp = pose_p * X_wp
    pose_c = body_q[child]
    X_wc = pose_c * joint_X_c[joint]

    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)
    rel_q = wp.quat_inverse(q_p) * q_c
    x_err = wp.transform_get_translation(X_wc) - wp.transform_get_translation(X_wp)
    x_err_p = wp.quat_rotate_inv(q_p, x_err)

    qd_start = joint_qd_start[joint]
    lin_axis_count = joint_dof_dim[joint, 0]
    ang_axis_count = joint_dof_dim[joint, 1]

    coordinate = float(0.0)
    linear_axis = wp.vec3(0.0)
    angular_covector = wp.vec3(0.0)

    if type == JointType.PRISMATIC:
        axis = joint_axis[qd_start]
        coordinate = wp.dot(x_err_p, axis)
        linear_axis = wp.quat_rotate(q_p, axis)
    elif type == JointType.REVOLUTE:
        axis = joint_axis[qd_start]
        coordinate = wp.quat_twist_angle_signed(axis, rel_q)
        angular_covector = wp.quat_rotate(q_p, _twist_coordinate_gradient(axis, rel_q))
    elif type == JointType.D6:
        if component < lin_axis_count:
            axis = joint_axis[qd_start + component]
            coordinate = wp.dot(x_err_p, axis)
            linear_axis = wp.quat_rotate(q_p, axis)
        else:
            angular_component = component - lin_axis_count
            angular_start = qd_start + lin_axis_count
            local_covector = wp.vec3(0.0)
            if ang_axis_count == 1:
                axis = joint_axis[angular_start]
                coordinate = wp.quat_twist_angle_signed(axis, rel_q)
                local_covector = _twist_coordinate_gradient(axis, rel_q)
            elif ang_axis_count == 2:
                axis_0 = joint_axis[angular_start + 0]
                axis_1 = joint_axis[angular_start + 1]
                if continuous:
                    coordinate, local_covector = _two_axis_coordinate(axis_0, axis_1, rel_q, angular_component)
                else:
                    coordinates_2, _unused_velocities_2 = invert_2d_rotational_dofs(
                        axis_0, axis_1, q_p, q_c, wp.vec3(0.0)
                    )
                    coordinate = coordinates_2[angular_component]
                    axis_0_q, axis_1_q, axis_2_q = transform_3d_rotational_axes(
                        axis_0, axis_1, wp.cross(axis_0, axis_1), coordinates_2[0], coordinates_2[1]
                    )
                    local_covector = _euler_coordinate_gradient(axis_0_q, axis_1_q, axis_2_q, angular_component)
            elif ang_axis_count == 3:
                axis_0 = joint_axis[angular_start + 0]
                axis_1 = joint_axis[angular_start + 1]
                axis_2 = joint_axis[angular_start + 2]
                coordinates_3, _unused_velocities_3 = invert_3d_rotational_dofs(
                    axis_0, axis_1, axis_2, q_p, q_c, wp.vec3(0.0)
                )
                if continuous:
                    coordinates_3 = _continuous_euler_angles(coordinates_3, angular_reference, 3)
                coordinate = coordinates_3[angular_component]
                axis_0_q, axis_1_q, axis_2_q = transform_3d_rotational_axes(
                    axis_0, axis_1, axis_2, coordinates_3[0], coordinates_3[1]
                )
                local_covector = _euler_coordinate_gradient(axis_0_q, axis_1_q, axis_2_q, angular_component)
            angular_covector = wp.quat_rotate(q_p, local_covector)

    r_p = wp.vec3(0.0)
    if parent >= 0:
        r_p = wp.transform_get_translation(X_wp) - wp.transform_point(pose_p, body_com[parent])
    r_c = wp.transform_get_translation(X_wc) - wp.transform_point(pose_c, body_com[child])

    # Rotating the parent rotates both its anchor and the coordinate axis.
    gradient_parent = wp.spatial_vector(-linear_axis, -wp.cross(r_p + x_err, linear_axis) - angular_covector)
    gradient_child = wp.spatial_vector(linear_axis, wp.cross(r_c, linear_axis) + angular_covector)
    return coordinate, gradient_parent, gradient_child


@wp.func
def eval_joint_coordinate(
    joint: int,
    component: int,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_axis: wp.array[wp.vec3],
):
    """Evaluate a stateless coordinate, without selecting a continuous Euler branch."""
    return eval_joint_coordinate(
        joint,
        component,
        body_q,
        body_com,
        joint_type,
        joint_parent,
        joint_child,
        joint_X_p,
        joint_X_c,
        joint_qd_start,
        joint_dof_dim,
        joint_axis,
        wp.vec3(),
        False,
    )


@wp.func
def eval_joint_velocity(
    parent: int,
    child: int,
    parent_gradient: wp.spatial_vector,
    child_gradient: wp.spatial_vector,
    body_qd: wp.array[wp.spatial_vector],
) -> float:
    """Return one joint-coordinate velocity from maximal body velocities."""
    velocity = float(0.0)
    if parent >= 0:
        parent_twist = body_qd[parent]
        velocity += wp.dot(wp.spatial_top(parent_gradient), wp.spatial_top(parent_twist))
        velocity += wp.dot(wp.spatial_bottom(parent_gradient), wp.spatial_bottom(parent_twist))
    if child >= 0:
        child_twist = body_qd[child]
        velocity += wp.dot(wp.spatial_top(child_gradient), wp.spatial_top(child_twist))
        velocity += wp.dot(wp.spatial_bottom(child_gradient), wp.spatial_bottom(child_twist))
    return velocity
