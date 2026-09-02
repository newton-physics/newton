# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for massless routed tendon simulation in the XPBD solver.

Route geometry follows Müller et al., "Cable Joints" (SCA 2018). The solver
supports rolling contacts, fixed attachments, frictional pinholes, and finite
capstan slip through the link friction coefficient.
"""

import warp as wp

from ...sim.tendon import TendonLinkType
from ..tendon_kernels import tendon_material_tension, tendon_segment_length_rate


@wp.kernel
def update_tendon_diagnostics(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    seg_rest_length: wp.array[float],
    seg_attachment_l: wp.array[wp.vec3],
    seg_attachment_r: wp.array[wp.vec3],
    seg_compliance: wp.array[float],
    seg_damping: wp.array[float],
    seg_active: wp.array[int],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
    seg_material_tension: wp.array[float],
    seg_damping_tension: wp.array[float],
):
    """Update XPBD tendon tension diagnostics from the accepted pose."""
    seg = wp.tid()
    seg_material_tension[seg] = 0.0
    seg_damping_tension[seg] = 0.0
    if seg_active[seg] == 0:
        return

    link_l = seg_active_link_l[seg]
    link_r = seg_active_link_r[seg]
    attachment_l = seg_attachment_l[seg]
    attachment_r = seg_attachment_r[seg]
    length = wp.length(attachment_r - attachment_l)
    seg_material_tension[seg] = tendon_material_tension(length, seg_rest_length[seg], seg_compliance[seg])
    seg_damping_tension[seg] = seg_damping[seg] * tendon_segment_length_rate(
        body_q,
        body_qd,
        body_com,
        tendon_link_body,
        tendon_link_type,
        tendon_link_offset,
        tendon_link_axis,
        link_l,
        link_r,
        attachment_l,
        attachment_r,
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
    tendon_link_offset: wp.array[wp.vec3],
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
    seg_active: wp.array[int],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
    compliance_lambda_scale: float,
    relaxation: float,
    dt: float,
    # outputs
    seg_material_tension: wp.array[float],
    body_deltas: wp.array[wp.spatial_vector],
):
    """Solve unilateral distance constraints for each tendon segment.

    Launched with dim = tendon_segment_count. Each segment is a distance
    constraint between attachment points on two rigid bodies.
    """
    seg = wp.tid()
    seg_material_tension[seg] = 0.0
    if seg_active[seg] == 0:
        seg_lambda[seg] = 0.0
        seg_delta_lambda[seg] = 0.0
        return

    link_l = seg_active_link_l[seg]
    link_r = seg_active_link_r[seg]

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

    # Keep signed stretch because positive length rate may activate damping
    # before geometric stretch becomes positive.
    err = d - rest
    if d <= 1.0e-8:
        seg_lambda[seg] = 0.0
        seg_delta_lambda[seg] = 0.0
        return

    material_tension = tendon_material_tension(d, rest, compliance)
    seg_material_tension[seg] = material_tension

    # constraint direction
    n = diff / d

    world_com_l = wp.transform_point(pose_l, com_l)
    world_com_r = wp.transform_point(pose_r, com_r)

    r_l = x_l - world_com_l
    r_r = x_r - world_com_r

    # Jacobians
    linear_l = -n
    linear_r = n
    angular_l = -wp.cross(r_l, n)
    angular_r = wp.cross(r_r, n)
    # Remove only spin about the roller center; motion of an off-center roller
    # remains part of the rigid-body Jacobian.
    if link_type_l == int(TendonLinkType.ROLLING):
        center_l = wp.transform_point(pose_l, tendon_link_offset[link_l])
        normal_l = wp.transform_vector(pose_l, tendon_link_axis[link_l])
        radial_l = x_l - center_l
        angular_l = angular_l - wp.dot(wp.cross(radial_l, linear_l), normal_l) * normal_l
    if link_type_r == int(TendonLinkType.ROLLING):
        center_r = wp.transform_point(pose_r, tendon_link_offset[link_r])
        normal_r = wp.transform_vector(pose_r, tendon_link_axis[link_r])
        radial_r = x_r - center_r
        angular_r = angular_r - wp.dot(wp.cross(radial_r, linear_r), normal_r) * normal_r

    rot_l = wp.transform_get_rotation(pose_l)
    rot_r = wp.transform_get_rotation(pose_r)

    derr = 0.0
    denom = 0.0
    if body_l == body_r:
        # A shared body has one velocity, so combine its endpoint Jacobians before
        # forming the effective mass instead of treating it as two independent bodies.
        linear = linear_l + linear_r
        angular = angular_l + angular_r
        derr = wp.dot(linear, vel_l) + wp.dot(angular, omega_l)
        rot_ang = wp.quat_rotate_inv(rot_l, angular)
        denom = wp.length_sq(linear) * m_inv_l + wp.dot(rot_ang, I_inv_l * rot_ang)
    else:
        derr = (
            wp.dot(linear_l, vel_l) + wp.dot(linear_r, vel_r) + wp.dot(angular_l, omega_l) + wp.dot(angular_r, omega_r)
        )
        denom = wp.length_sq(linear_l) * m_inv_l + wp.length_sq(linear_r) * m_inv_r
        rot_ang_l = wp.quat_rotate_inv(rot_l, angular_l)
        rot_ang_r = wp.quat_rotate_inv(rot_r, angular_r)
        denom += wp.dot(rot_ang_l, I_inv_l * rot_ang_l)
        denom += wp.dot(rot_ang_r, I_inv_r * rot_ang_r)

    lambda_prev = seg_lambda[seg]
    alpha = compliance * compliance_lambda_scale
    gamma = compliance * damping
    d_lambda = -(err + alpha * lambda_prev + gamma * derr)
    linear_denom = (dt + gamma) * denom + compliance / dt
    if linear_denom > 0.0:
        d_lambda = d_lambda / linear_denom

    d_lambda = d_lambda * relaxation

    # A cable may transmit tension but not compression. Applying the
    # unilateral bound to the accumulated multiplier also lets positive
    # length rate activate damping at zero geometric stretch.
    lambda_new = wp.min(lambda_prev + d_lambda, 0.0)
    d_lambda = lambda_new - lambda_prev
    seg_lambda[seg] = lambda_new
    seg_delta_lambda[seg] = d_lambda

    # apply positional corrections
    lin_delta_l = linear_l * d_lambda
    ang_delta_l = angular_l * d_lambda
    lin_delta_r = linear_r * d_lambda
    ang_delta_r = angular_r * d_lambda

    if body_l == body_r:
        wp.atomic_add(
            body_deltas,
            body_l,
            wp.spatial_vector(lin_delta_l + lin_delta_r, ang_delta_l + ang_delta_r),
        )
    else:
        wp.atomic_add(body_deltas, body_l, wp.spatial_vector(lin_delta_l, ang_delta_l))
        wp.atomic_add(body_deltas, body_r, wp.spatial_vector(lin_delta_r, ang_delta_r))


@wp.kernel
def solve_tendon_slip(
    body_q: wp.array[wp.transform],
    tendon_link_seg_left: wp.array[int],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_radius: wp.array[float],
    tendon_link_mu: wp.array[float],
    tendon_link_active: wp.array[bool],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    seg_attachment_l: wp.array[wp.vec3],
    seg_attachment_r: wp.array[wp.vec3],
    seg_compliance: wp.array[float],
    seg_material_tension: wp.array[float],
    seg_damping_tension: wp.array[float],
    seg_delta_lambda: wp.array[float],
    relaxation: float,
    # outputs
    body_deltas: wp.array[wp.spatial_vector],
):
    """Solve one rolling slip/friction row per authored tendon link.

    Stretch carries the common cable load.  This pass handles the tangential
    coupling between adjacent spans and pulley rim motion.  The capstan cone
    controls both rest-length transfer and the admissible spin-axis torque.
    """
    link_idx = wp.tid()
    if tendon_link_type[link_idx] != int(TendonLinkType.ROLLING):
        return
    if not tendon_link_active[link_idx]:
        return

    radius = tendon_link_radius[link_idx]
    seg_left = tendon_link_seg_left[link_idx]
    if radius <= 0.0 or seg_left < 0:
        return

    seg_right = seg_left + 1
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

    # Match the stretch row's constitutive tension and add its damping component.
    compliance_l = seg_compliance[seg_left]
    compliance_r = seg_compliance[seg_right]
    damping_tension_l = seg_damping_tension[seg_left] if compliance_l > 0.0 else 0.0
    damping_tension_r = seg_damping_tension[seg_right] if compliance_r > 0.0 else 0.0
    force_l = wp.max(seg_material_tension[seg_left] + damping_tension_l, 0.0)
    force_r = wp.max(seg_material_tension[seg_right] + damping_tension_r, 0.0)

    force_sum = force_l + force_r
    force_diff = wp.abs(force_l - force_r)
    allowed_diff = beta * force_sum
    scale = wp.min(1.0, allowed_diff / wp.max(force_diff, 1.0e-8))

    # Stretch retains center-motion torque; add only friction-limited spin
    # about the roller center here.
    spin_delta = wp.vec3(0.0, 0.0, 0.0)

    x_l = seg_attachment_l[seg_left]
    x_r = seg_attachment_r[seg_left]
    diff = x_r - x_l
    dist = wp.length(diff)
    if dist > 1.0e-8:
        n = diff / dist
        r = x_r - center
        angular = wp.cross(r, n)
        candidate = angular * seg_delta_lambda[seg_left]
        spin_delta = spin_delta + normal * wp.dot(candidate, normal)

    x_l = seg_attachment_l[seg_right]
    x_r = seg_attachment_r[seg_right]
    diff = x_r - x_l
    dist = wp.length(diff)
    if dist > 1.0e-8:
        n = diff / dist
        r = x_l - center
        angular = -wp.cross(r, n)
        candidate = angular * seg_delta_lambda[seg_right]
        spin_delta = spin_delta + normal * wp.dot(candidate, normal)

    spin_delta = spin_delta * scale * beta
    wp.atomic_add(body_deltas, body, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), spin_delta))
