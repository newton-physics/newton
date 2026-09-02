# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from ...sim.tendon import TendonLinkType
from ..tendon_kernels import (
    tendon_material_tension,
    tendon_segment_length_rate_from_poses,
)

wp.set_module_options({"enable_backward": False})

_MIN_TENDON_COMPLIANCE = wp.constant(1.0e-8)


@wp.struct
class TendonForceElementAdjacencyInfo:
    """CSR adjacency between VBD rigid bodies and tendon segments."""

    body_adj_segments: wp.array[wp.int32]
    body_adj_segments_offsets: wp.array[wp.int32]

    def to(self, device):
        """Copy the adjacency to a device."""
        if device == self.body_adj_segments.device:
            return self

        adjacency = TendonForceElementAdjacencyInfo()
        adjacency.body_adj_segments = self.body_adj_segments.to(device)
        adjacency.body_adj_segments_offsets = self.body_adj_segments_offsets.to(device)
        return adjacency


@wp.kernel
def snapshot_tendon_segment_length_reference(
    dt: float,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_active: wp.array[int],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
    seg_length_prev: wp.array[float],
):
    """Snapshot previous-pose segment lengths for final VBD diagnostics."""
    seg = wp.tid()
    seg_length_prev[seg] = 0.0
    if seg_active[seg] == 0:
        return

    link_l = seg_active_link_l[seg]
    link_r = seg_active_link_r[seg]
    body_l = tendon_link_body[link_l]
    body_r = tendon_link_body[link_r]
    attachment_l = wp.transform_point(body_q[body_l], seg_attachment_l_local[seg])
    attachment_r = wp.transform_point(body_q[body_r], seg_attachment_r_local[seg])
    length_rate = tendon_segment_length_rate_from_poses(
        dt,
        body_q,
        body_q_prev,
        body_com,
        tendon_link_body,
        tendon_link_type,
        tendon_link_offset,
        tendon_link_axis,
        link_l,
        link_r,
        seg_attachment_l_local[seg],
        seg_attachment_r_local[seg],
        attachment_l,
        attachment_r,
    )
    seg_length_prev[seg] = wp.length(attachment_r - attachment_l) - dt * length_rate


@wp.kernel
def update_tendon_segment_diagnostics(
    dt: float,
    body_q: wp.array[wp.transform],
    tendon_link_body: wp.array[int],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_rest_length: wp.array[float],
    seg_active_compliance: wp.array[float],
    seg_active_damping: wp.array[float],
    seg_active: wp.array[int],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
    seg_length_prev: wp.array[float],
    seg_attachment_l: wp.array[wp.vec3],
    seg_attachment_r: wp.array[wp.vec3],
    seg_material_tension: wp.array[float],
    seg_damping_tension: wp.array[float],
    seg_lambda: wp.array[float],
):
    """Update tendon geometry and damping from the accepted VBD pose."""
    seg = wp.tid()
    seg_attachment_l[seg] = wp.vec3(0.0)
    seg_attachment_r[seg] = wp.vec3(0.0)
    seg_material_tension[seg] = 0.0
    seg_damping_tension[seg] = 0.0
    # Native VBD does not solve XPBD constraint multipliers.
    seg_lambda[seg] = 0.0
    if seg_active[seg] == 0:
        return

    link_l = seg_active_link_l[seg]
    link_r = seg_active_link_r[seg]
    body_l = tendon_link_body[link_l]
    body_r = tendon_link_body[link_r]

    attachment_l = wp.transform_point(body_q[body_l], seg_attachment_l_local[seg])
    attachment_r = wp.transform_point(body_q[body_r], seg_attachment_r_local[seg])
    seg_attachment_l[seg] = attachment_l
    seg_attachment_r[seg] = attachment_r
    length = wp.length(attachment_r - attachment_l)
    length_rate = (length - seg_length_prev[seg]) / dt
    compliance = wp.max(seg_active_compliance[seg], _MIN_TENDON_COMPLIANCE)
    seg_material_tension[seg] = tendon_material_tension(length, seg_rest_length[seg], compliance)
    seg_damping_tension[seg] = seg_active_damping[seg] * length_rate


@wp.func
def _rolling_spin_axis_component(
    body_q: wp.array[wp.transform],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_radius: wp.array[float],
    tendon_link_mu: wp.array[float],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    tendon_link_cone_seg_l: wp.array[int],
    tendon_link_cone_seg_r: wp.array[int],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_active: wp.array[int],
    link: int,
    attachment: wp.vec3,
    direction: wp.vec3,
) -> wp.vec3:
    """Return the roller-axis moment removed from a unit-tension span.

    Returns ``(1 - spin_scale) * dot(cross(radial, direction), normal) * normal``
    — the roller-axis part of the span's moment that an ideal (or partially
    slipping) pulley cannot transmit through its rim. Returns zero for
    non-rolling links and when the current wrap geometry is unavailable.
    """
    if tendon_link_type[link] != int(TendonLinkType.ROLLING):
        return wp.vec3(0.0)
    seg_left = tendon_link_cone_seg_l[link]
    seg_right = tendon_link_cone_seg_r[link]
    if seg_left < 0 or seg_right < 0:
        return wp.vec3(0.0)
    if seg_right >= seg_active.shape[0] or seg_active[seg_left] == 0 or seg_active[seg_right] == 0:
        return wp.vec3(0.0)

    body = tendon_link_body[link]
    pose = body_q[body]
    center = wp.transform_point(pose, tendon_link_offset[link])
    normal = wp.normalize(wp.transform_vector(pose, tendon_link_axis[link]))
    point_left = wp.transform_point(pose, seg_attachment_r_local[seg_left])
    point_right = wp.transform_point(pose, seg_attachment_l_local[seg_right])
    radial_left = point_left - center
    radial_right = point_right - center
    radial_left = radial_left - wp.dot(radial_left, normal) * normal
    radial_right = radial_right - wp.dot(radial_right, normal) * normal
    radial_left_length = wp.length(radial_left)
    radial_right_length = wp.length(radial_right)

    theta = float(0.0)
    if tendon_link_radius[link] > 0.0 and radial_left_length > 1.0e-8 and radial_right_length > 1.0e-8:
        unit_left = radial_left / radial_left_length
        unit_right = radial_right / radial_right_length
        theta = wp.abs(
            wp.atan2(
                wp.dot(wp.cross(unit_left, unit_right), normal),
                wp.dot(unit_left, unit_right),
            )
        )

    cap_ratio = wp.exp(wp.min(wp.max(tendon_link_mu[link], 0.0) * theta, 20.0))
    spin_scale = (cap_ratio - 1.0) / (cap_ratio + 1.0)
    radial = attachment - center
    return (1.0 - spin_scale) * wp.dot(wp.cross(radial, direction), normal) * normal


@wp.func
def evaluate_tendon_force_hessians(
    body: int,
    dt: float,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    adjacency: TendonForceElementAdjacencyInfo,
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_radius: wp.array[float],
    tendon_link_mu: wp.array[float],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    tendon_link_cone_seg_l: wp.array[int],
    tendon_link_cone_seg_r: wp.array[int],
    seg_rest_length: wp.array[float],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_active_compliance: wp.array[float],
    seg_active_damping: wp.array[float],
    seg_active: wp.array[int],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
):
    """Evaluate unilateral tendon spring-damper forces for one VBD body."""
    force = wp.vec3(0.0)
    torque = wp.vec3(0.0)
    h_ll = wp.mat33(0.0)
    h_al = wp.mat33(0.0)
    h_aa = wp.mat33(0.0)

    adjacent_start = adjacency.body_adj_segments_offsets[body]
    adjacent_end = adjacency.body_adj_segments_offsets[body + 1]
    for adjacent_index in range(adjacent_start, adjacent_end):
        seg = adjacency.body_adj_segments[adjacent_index]
        if seg_active[seg] == 0:
            continue

        link_l = seg_active_link_l[seg]
        link_r = seg_active_link_r[seg]
        body_l = tendon_link_body[link_l]
        body_r = tendon_link_body[link_r]
        if body != body_l and body != body_r:
            continue

        attachment_l = wp.transform_point(body_q[body_l], seg_attachment_l_local[seg])
        attachment_r = wp.transform_point(body_q[body_r], seg_attachment_r_local[seg])
        direction = attachment_r - attachment_l
        length = wp.length(direction)
        if length <= 1.0e-8:
            continue
        direction = direction / length

        compliance = wp.max(seg_active_compliance[seg], _MIN_TENDON_COMPLIANCE)

        rest_length = seg_rest_length[seg]
        length_rate = tendon_segment_length_rate_from_poses(
            dt,
            body_q,
            body_q_prev,
            body_com,
            tendon_link_body,
            tendon_link_type,
            tendon_link_offset,
            tendon_link_axis,
            link_l,
            link_r,
            seg_attachment_l_local[seg],
            seg_attachment_r_local[seg],
            attachment_l,
            attachment_r,
        )

        stiffness = 1.0 / compliance
        damping = seg_active_damping[seg]
        tension = stiffness * (length - rest_length) + damping * length_rate
        tension = wp.max(tension, 0.0)

        if tension <= 0.0:
            continue

        if body_l == body_r:
            # Both endpoints ride this body: the endpoint forces and their base
            # torques cancel exactly, but the rolling spin corrections are
            # asymmetric. Retain that net roller-axis row, matching XPBD's
            # combined same-body Jacobian.
            fix_l = _rolling_spin_axis_component(
                body_q,
                tendon_link_body,
                tendon_link_type,
                tendon_link_radius,
                tendon_link_mu,
                tendon_link_offset,
                tendon_link_axis,
                tendon_link_cone_seg_l,
                tendon_link_cone_seg_r,
                seg_attachment_l_local,
                seg_attachment_r_local,
                seg_active,
                link_l,
                attachment_l,
                direction,
            )
            fix_r = _rolling_spin_axis_component(
                body_q,
                tendon_link_body,
                tendon_link_type,
                tendon_link_radius,
                tendon_link_mu,
                tendon_link_offset,
                tendon_link_axis,
                tendon_link_cone_seg_l,
                tendon_link_cone_seg_r,
                seg_attachment_l_local,
                seg_attachment_r_local,
                seg_active,
                link_r,
                attachment_r,
                direction,
            )
            net_moment_axis = fix_l - fix_r
            torque = torque - tension * net_moment_axis
            same_body_stiffness = stiffness + damping / dt
            h_aa = h_aa + same_body_stiffness * wp.outer(net_moment_axis, net_moment_axis)
            continue

        world_com = wp.transform_point(body_q[body], body_com[body])
        if body == body_l:
            attachment = attachment_l
            link = link_l
            force_sign = 1.0
            body_force = tension * direction
        else:
            attachment = attachment_r
            link = link_r
            force_sign = -1.0
            body_force = -tension * direction

        moment_arm = attachment - world_com
        moment_axis = wp.cross(moment_arm, direction)

        if tendon_link_type[link] == int(TendonLinkType.ROLLING):
            # Free-span tension still loads the body, but only capstan friction
            # transmits the rolling-axis part of its moment.
            moment_axis = moment_axis - _rolling_spin_axis_component(
                body_q,
                tendon_link_body,
                tendon_link_type,
                tendon_link_radius,
                tendon_link_mu,
                tendon_link_offset,
                tendon_link_axis,
                tendon_link_cone_seg_l,
                tendon_link_cone_seg_r,
                seg_attachment_l_local,
                seg_attachment_r_local,
                seg_active,
                link,
                attachment,
                direction,
            )

        body_torque = force_sign * tension * moment_axis

        effective_stiffness = stiffness + damping / dt

        force = force + body_force
        torque = torque + body_torque
        # The axial Gauss-Newton approximation remains positive semidefinite.
        h_ll = h_ll + effective_stiffness * wp.outer(direction, direction)
        h_al = h_al + effective_stiffness * wp.outer(moment_axis, direction)
        h_aa = h_aa + effective_stiffness * wp.outer(moment_axis, moment_axis)

    return force, torque, h_ll, h_al, h_aa
