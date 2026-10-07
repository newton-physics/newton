# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joint-owned mimic rows in VBD's rigid-body objective."""

import warp as wp

from ...math import quat_decompose
from ...sim import JointType
from ...sim.articulation import transform_3d_rotational_axes
from .rigid_vbd_kernels import (
    _alm_relaxed_ascent,
    _bilateral_auto_rho,
    _compliant_alm_coefficients,
)

wp.set_module_options({"enable_backward": False})

_mat46 = wp.types.matrix(shape=(4, 6), dtype=float)


@wp.struct
class JointMimicData:
    joint_type: wp.array[int]
    joint_enabled: wp.array[bool]
    joint_parent: wp.array[int]
    joint_child: wp.array[int]
    joint_X_p: wp.array[wp.transform]
    joint_X_c: wp.array[wp.transform]
    joint_qd_start: wp.array[int]
    joint_dof_dim: wp.array2d[int]
    joint_axis: wp.array[wp.vec3]
    joint_mimic_joint: wp.array[int]
    joint_mimic_coeffs: wp.array[wp.vec2]
    body_com: wp.array[wp.vec3]
    body_colors: wp.array[int]
    linear_ke: float
    angular_ke: float
    lambda_: wp.array2d[float]


@wp.struct
class JointMimicRow:
    bodies: wp.vec4i
    gradients: _mat46
    error: float
    stiffness: float


@wp.func
def _supported(joint_type: int):
    return joint_type == JointType.PRISMATIC or joint_type == JointType.REVOLUTE or joint_type == JointType.D6


@wp.func
def _active(data: JointMimicData, follower: int):
    reference = data.joint_mimic_joint[follower]
    if reference < 0 or not data.joint_enabled[follower]:
        return False
    return (
        data.joint_enabled[reference]
        and _supported(data.joint_type[follower])
        and _supported(data.joint_type[reference])
    )


@wp.func
def _coordinate(data: JointMimicData, joint: int, component: int, body_q: wp.array[wp.transform]):
    """Evaluate a supported coordinate and its derivatives about each body's COM."""
    parent = data.joint_parent[joint]
    child = data.joint_child[joint]
    X_wp = data.joint_X_p[joint]
    if parent >= 0:
        X_wp = body_q[parent] * X_wp
    X_wc = body_q[child] * data.joint_X_c[joint]
    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)
    linear_count = data.joint_dof_dim[joint, 0]
    start = data.joint_qd_start[joint]
    if component < linear_count:
        axis = wp.quat_rotate(q_p, data.joint_axis[start + component])
        anchor_c = wp.transform_get_translation(X_wc)
        separation = anchor_c - wp.transform_get_translation(X_wp)
        coordinate = wp.dot(separation, axis)
        # The prismatic coordinate's axis rotates with its parent frame.
        r_p = wp.vec3(0.0)
        if parent >= 0:
            r_p = anchor_c - wp.transform_point(body_q[parent], data.body_com[parent])
        r_c = anchor_c - wp.transform_point(body_q[child], data.body_com[child])
        return coordinate, wp.spatial_vector(-axis, -wp.cross(r_p, axis)), wp.spatial_vector(axis, wp.cross(r_c, axis))

    start += linear_count
    angular_count = data.joint_dof_dim[joint, 1]
    angular_component = component - linear_count
    axis_0 = data.joint_axis[start]
    coordinate = float(0.0)
    covector = wp.vec3(0.0)
    if angular_count == 1:
        relative = wp.quat_inverse(q_p) * q_c
        coordinate = wp.quat_twist_angle_signed(axis_0, relative)
        # Differentiate the signed twist even when structural swing error remains.
        v = wp.vec3(relative[0], relative[1], relative[2])
        w = relative[3]
        twist = wp.dot(axis_0, v)
        denom = wp.max(w * w + twist * twist, 1.0e-12)
        covector = (w * w * axis_0 + w * wp.cross(v, axis_0) + twist * v) / denom
    else:
        axis_1 = data.joint_axis[start + 1]
        axis_2 = wp.cross(axis_0, axis_1)
        # Use the same canonical frame as invert_2d/3d_rotational_dofs.
        q_off = wp.quat_from_matrix(wp.matrix_from_cols(axis_0, axis_1, axis_2))
        angles = quat_decompose(wp.quat_inverse(q_off) * wp.quat_inverse(q_p) * q_c * q_off)
        coordinate = angles[angular_component]
        a0 = wp.quat_rotate(q_off, wp.vec3(1.0, 0.0, 0.0))
        a1 = wp.quat_rotate(q_off, wp.vec3(0.0, 1.0, 0.0))
        a2 = wp.quat_rotate(q_off, wp.vec3(0.0, 0.0, 1.0))
        a0, a1, a2 = transform_3d_rotational_axes(a0, a1, a2, angles[0], angles[1])
        # A coordinate gradient is a row of the inverse angular-velocity basis.
        axes = wp.matrix_from_rows(a0, a1, a2)
        covector = wp.cross(axes[(angular_component + 1) % 3], axes[(angular_component + 2) % 3])
        covector /= wp.dot(axes[angular_component], covector)
        if angular_component == 2 and wp.dot(axis_2, data.joint_axis[start + 2]) < 0.0:
            coordinate = -coordinate
            covector = -covector
    covector = wp.quat_rotate(q_p, covector)
    return coordinate, wp.spatial_vector(wp.vec3(0.0), -covector), wp.spatial_vector(wp.vec3(0.0), covector)


@wp.func
def _row(data: JointMimicData, follower: int, component: int, body_q: wp.array[wp.transform]):
    reference = data.joint_mimic_joint[follower]
    q_f, g_fp, g_fc = _coordinate(data, follower, component, body_q)
    q_r, g_rp, g_rc = _coordinate(data, reference, component, body_q)
    coeffs = data.joint_mimic_coeffs[follower]
    row = JointMimicRow()
    row.bodies = wp.vec4i(
        data.joint_parent[follower],
        data.joint_child[follower],
        data.joint_parent[reference],
        data.joint_child[reference],
    )
    for k in range(6):
        row.gradients[0, k] = g_fp[k]
        row.gradients[1, k] = g_fc[k]
        row.gradients[2, k] = -coeffs[1] * g_rp[k]
        row.gradients[3, k] = -coeffs[1] * g_rc[k]
    row.error = q_f - coeffs[0] - coeffs[1] * q_r
    row.stiffness = data.linear_ke
    if component >= data.joint_dof_dim[follower, 0]:
        row.error = wp.atan2(wp.sin(row.error), wp.cos(row.error))
        row.stiffness = data.angular_ke

    # Shared parents and serial pairs need the cross terms of their combined gradient.
    for i in range(4):
        for j in range(i):
            if row.bodies[i] >= 0 and row.bodies[i] == row.bodies[j]:
                for k in range(6):
                    row.gradients[j, k] = row.gradients[j, k] + row.gradients[i, k]
                row.bodies[i] = -1
    return row


@wp.kernel
def accumulate_joint_mimics(
    data: JointMimicData,
    body_q: wp.array[wp.transform],
    body_inv_mass: wp.array[float],
    color: int,
    body_forces: wp.array[wp.vec3],
    body_torques: wp.array[wp.vec3],
    body_hessian_ll: wp.array[wp.mat33],
    body_hessian_al: wp.array[wp.mat33],
    body_hessian_aa: wp.array[wp.mat33],
):
    """Accumulate coupled mimic rows before updating the bodies in one color."""
    follower = wp.tid()
    if not _active(data, follower):
        return
    count = data.joint_dof_dim[follower, 0] + data.joint_dof_dim[follower, 1]
    for component in range(count):
        row = _row(data, follower, component, body_q)
        rho = _bilateral_auto_rho(0.0, row.stiffness)
        s, stiffness, _a = _compliant_alm_coefficients(row.stiffness, rho)
        force = stiffness * row.error + s * data.lambda_[follower, component]
        concurrent = int(0)
        for i in range(4):
            body = row.bodies[i]
            if body >= 0 and body_inv_mass[body] > 0.0 and data.body_colors[body] == color:
                if wp.dot(row.gradients[i], row.gradients[i]) > 0.0:
                    concurrent += 1
        # Existing structural colors can share a mimic row. Majorize its cross
        # terms so their simultaneous block updates descend the same objective.
        hessian = stiffness * float(concurrent)
        for i in range(4):
            body = row.bodies[i]
            if body >= 0 and body_inv_mass[body] > 0.0 and data.body_colors[body] == color:
                gradient = row.gradients[i]
                linear = wp.spatial_top(gradient)
                angular = wp.spatial_bottom(gradient)
                wp.atomic_add(body_forces, body, -force * linear)
                wp.atomic_add(body_torques, body, -force * angular)
                wp.atomic_add(body_hessian_ll, body, hessian * wp.outer(linear, linear))
                wp.atomic_add(body_hessian_al, body, hessian * wp.outer(angular, linear))
                wp.atomic_add(body_hessian_aa, body, hessian * wp.outer(angular, angular))


@wp.kernel
def update_joint_mimic_duals(data: JointMimicData, body_q: wp.array[wp.transform]):
    """Advance the mimic multipliers from the completed body sweep."""
    follower = wp.tid()
    count = data.joint_dof_dim[follower, 0] + data.joint_dof_dim[follower, 1]
    for component in range(count):
        if _active(data, follower):
            row = _row(data, follower, component, body_q)
            rho = _bilateral_auto_rho(0.0, row.stiffness)
            data.lambda_[follower, component] = _alm_relaxed_ascent(
                data.lambda_[follower, component], row.error, row.stiffness, rho
            )
        else:
            data.lambda_[follower, component] = 0.0
