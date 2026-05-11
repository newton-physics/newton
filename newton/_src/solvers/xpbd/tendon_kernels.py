# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for tendon (cable joint) simulation in the XPBD solver.

Implements the Cable Joints method [Müller et al. SCA 2018]. The solver
supports rolling contacts, fixed attachments, frictional pinholes, and finite
capstan slip through the link friction coefficient.
"""

import warp as wp

from ...sim.tendon import TendonLinkType
from ..tendon_kernels import (  # noqa: F401
    advance_point_on_circle,
    signed_arc_length,
    tangent_point_circle,
    update_tendon_attachments,
)


@wp.kernel
def solve_tendon_stretch(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_axis: wp.array[wp.vec3],
    seg_rest_length: wp.array[float],
    seg_attachment_l: wp.array[wp.vec3],
    seg_attachment_r: wp.array[wp.vec3],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_compliance: wp.array[float],
    seg_damping: wp.array[float],
    seg_lambda: wp.array[float],
    seg_delta_lambda: wp.array[float],
    seg_link_l: wp.array[int],
    relaxation: float,
    dt: float,
    # outputs
    body_deltas: wp.array[wp.spatial_vector],
):
    """Phase 3: Solve unilateral distance constraints for each tendon segment.

    Launched with dim = tendon_segment_count. Each segment is a distance
    constraint between attachment points on two rigid bodies.
    """
    seg = wp.tid()
    link_l = seg_link_l[seg]
    link_r = link_l + 1

    body_l = tendon_link_body[link_l]
    body_r = tendon_link_body[link_r]
    link_type_l = tendon_link_type[link_l]
    link_type_r = tendon_link_type[link_r]

    pose_l = body_q[body_l]
    pose_r = body_q[body_r]
    vel_l = wp.spatial_top(body_qd[body_l])  # linear velocity
    omega_l = wp.spatial_bottom(body_qd[body_l])  # angular velocity
    vel_r = wp.spatial_top(body_qd[body_r])
    omega_r = wp.spatial_bottom(body_qd[body_r])

    com_l = body_com[body_l]
    com_r = body_com[body_r]
    m_inv_l = body_inv_mass[body_l]
    m_inv_r = body_inv_mass[body_r]
    I_inv_l = body_inv_inertia[body_l]
    I_inv_r = body_inv_inertia[body_r]

    x_l = wp.transform_point(pose_l, seg_attachment_l_local[seg])
    x_r = wp.transform_point(pose_r, seg_attachment_r_local[seg])
    seg_attachment_l[seg] = x_l
    seg_attachment_r[seg] = x_r
    rest = seg_rest_length[seg]
    compliance = seg_compliance[seg]
    damping = seg_damping[seg]

    diff = x_r - x_l
    d = wp.length(diff)

    # unilateral: only enforce when stretched beyond rest length
    err = d - rest
    if err <= 0.0:
        seg_lambda[seg] = 0.0
        seg_delta_lambda[seg] = 0.0
        return

    # constraint direction
    n = diff / wp.max(d, 1.0e-8)

    world_com_l = wp.transform_point(pose_l, com_l)
    world_com_r = wp.transform_point(pose_r, com_r)

    r_l = x_l - world_com_l
    r_r = x_r - world_com_r

    # Jacobians
    linear_l = -n
    linear_r = n
    angular_l = -wp.cross(r_l, n)
    angular_r = wp.cross(r_r, n)
    if link_type_l == int(TendonLinkType.ROLLING):
        normal_l = wp.transform_vector(pose_l, tendon_link_axis[link_l])
        angular_l = angular_l - wp.dot(angular_l, normal_l) * normal_l
    if link_type_r == int(TendonLinkType.ROLLING):
        normal_r = wp.transform_vector(pose_r, tendon_link_axis[link_r])
        angular_r = angular_r - wp.dot(angular_r, normal_r) * normal_r

    # constraint velocity
    derr = wp.dot(linear_l, vel_l) + wp.dot(linear_r, vel_r) + wp.dot(angular_l, omega_l) + wp.dot(angular_r, omega_r)

    # effective mass
    denom = 0.0
    denom += wp.length_sq(linear_l) * m_inv_l
    denom += wp.length_sq(linear_r) * m_inv_r

    rot_l = wp.transform_get_rotation(pose_l)
    rot_r = wp.transform_get_rotation(pose_r)
    rot_ang_l = wp.quat_rotate_inv(rot_l, angular_l)
    rot_ang_r = wp.quat_rotate_inv(rot_r, angular_r)
    denom += wp.dot(rot_ang_l, I_inv_l * rot_ang_l)
    denom += wp.dot(rot_ang_r, I_inv_r * rot_ang_r)

    alpha = compliance
    gamma = compliance * damping

    lambda_prev = seg_lambda[seg]
    d_lambda = -(err + alpha * lambda_prev + gamma * derr)
    if denom + alpha > 0.0:
        d_lambda = d_lambda / ((dt + gamma) * denom + alpha / dt)

    seg_lambda[seg] = lambda_prev + d_lambda
    seg_delta_lambda[seg] = d_lambda

    # apply positional corrections
    lin_delta_l = linear_l * (d_lambda * relaxation)
    ang_delta_l = angular_l * (d_lambda * relaxation)
    lin_delta_r = linear_r * (d_lambda * relaxation)
    ang_delta_r = angular_r * (d_lambda * relaxation)

    wp.atomic_add(body_deltas, body_l, wp.spatial_vector(lin_delta_l, ang_delta_l))
    wp.atomic_add(body_deltas, body_r, wp.spatial_vector(lin_delta_r, ang_delta_r))


@wp.kernel
def solve_tendon_slip(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    tendon_start: wp.array[int],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_radius: wp.array[float],
    tendon_link_mu: wp.array[float],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    seg_rest_length: wp.array[float],
    seg_attachment_l: wp.array[wp.vec3],
    seg_attachment_r: wp.array[wp.vec3],
    seg_compliance: wp.array[float],
    seg_delta_lambda: wp.array[float],
    relaxation: float,
    # outputs
    body_deltas: wp.array[wp.spatial_vector],
):
    """Solve rolling slip/friction rows for each tendon.

    Stretch carries the common cable load.  This pass handles the tangential
    coupling between adjacent spans and pulley rim motion.  The capstan cone
    controls both rest-length transfer and the admissible spin-axis torque.
    """
    tendon_id = wp.tid()
    link_start = tendon_start[tendon_id]
    link_end = tendon_start[tendon_id + 1]
    num_links = link_end - link_start
    num_segs = num_links - 1
    if num_segs < 2:
        return

    seg_offset = int(0)
    for t in range(tendon_id):
        seg_offset = seg_offset + (tendon_start[t + 1] - tendon_start[t] - 1)

    for i in range(1, num_links - 1):
        link_idx = link_start + i
        if tendon_link_type[link_idx] != int(TendonLinkType.ROLLING):
            continue

        radius = tendon_link_radius[link_idx]
        if radius <= 0.0:
            continue

        seg_left = seg_offset + i - 1
        seg_right = seg_offset + i
        body = tendon_link_body[link_idx]
        pose = body_q[body]
        center = wp.transform_point(pose, tendon_link_offset[link_idx])
        normal = wp.transform_vector(pose, tendon_link_axis[link_idx])

        pt_left = seg_attachment_r[seg_left]
        pt_right = seg_attachment_l[seg_right]
        r_left = pt_left - center
        r_right = pt_right - center
        r_left = r_left - wp.dot(r_left, normal) * normal
        r_right = r_right - wp.dot(r_right, normal) * normal
        len_rl = wp.length(r_left)
        len_rr = wp.length(r_right)
        theta = wp.pi
        if len_rl > 1.0e-8 and len_rr > 1.0e-8:
            u_left = r_left / len_rl
            u_right = r_right / len_rr
            theta = wp.abs(wp.atan2(wp.dot(wp.cross(u_left, u_right), normal), wp.dot(u_left, u_right)))

        cap_ratio = wp.exp(wp.min(wp.max(tendon_link_mu[link_idx], 0.0) * theta, 20.0))
        beta = (cap_ratio - 1.0) / (cap_ratio + 1.0)

        len_l = wp.length(seg_attachment_r[seg_left] - seg_attachment_l[seg_left])
        len_r = wp.length(seg_attachment_r[seg_right] - seg_attachment_l[seg_right])
        d_l = len_l - seg_rest_length[seg_left]
        d_r = len_r - seg_rest_length[seg_right]
        if d_l < 0.0:
            d_l = 0.0
        if d_r < 0.0:
            d_r = 0.0

        comp_l = wp.max(seg_compliance[seg_left], 1.0e-8)
        comp_r = wp.max(seg_compliance[seg_right], 1.0e-8)
        force_l = d_l / comp_l
        force_r = d_r / comp_r

        force_sum = force_l + force_r
        force_diff = wp.abs(force_l - force_r)
        allowed_diff = beta * force_sum
        scale = wp.min(1.0, allowed_diff / wp.max(force_diff, 1.0e-8))

        world_com = wp.transform_point(pose, body_com[body])
        spin_delta = wp.vec3(0.0, 0.0, 0.0)

        x_l = seg_attachment_l[seg_left]
        x_r = seg_attachment_r[seg_left]
        diff = x_r - x_l
        dist = wp.length(diff)
        if dist > 1.0e-8:
            n = diff / dist
            r = x_r - world_com
            angular = wp.cross(r, n)
            candidate = angular * (seg_delta_lambda[seg_left] * relaxation)
            spin_delta = spin_delta + normal * wp.dot(candidate, normal)

        x_l = seg_attachment_l[seg_right]
        x_r = seg_attachment_r[seg_right]
        diff = x_r - x_l
        dist = wp.length(diff)
        if dist > 1.0e-8:
            n = diff / dist
            r = x_l - world_com
            angular = -wp.cross(r, n)
            candidate = angular * (seg_delta_lambda[seg_right] * relaxation)
            spin_delta = spin_delta + normal * wp.dot(candidate, normal)

        spin_delta = spin_delta * scale * beta
        wp.atomic_add(body_deltas, body, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), spin_delta))
