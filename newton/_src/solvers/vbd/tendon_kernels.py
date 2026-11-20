# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from ...sim.tendon import TendonGuideType
from ..tendon_kernels import (
    tendon_body_pose,
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
    tendon_guide_body: wp.array[int],
    tendon_guide_type: wp.array[int],
    tendon_guide_offset: wp.array[wp.vec3],
    tendon_guide_axis: wp.array[wp.vec3],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_active: wp.array[int],
    seg_active_guide_l: wp.array[int],
    seg_active_guide_r: wp.array[int],
    seg_length_prev: wp.array[float],
):
    """Snapshot previous-pose segment lengths for final VBD diagnostics."""
    seg = wp.tid()
    seg_length_prev[seg] = 0.0
    if seg_active[seg] == 0:
        return

    guide_l = seg_active_guide_l[seg]
    guide_r = seg_active_guide_r[seg]
    body_l = tendon_guide_body[guide_l]
    body_r = tendon_guide_body[guide_r]
    attachment_l = wp.transform_point(tendon_body_pose(body_q, body_l), seg_attachment_l_local[seg])
    attachment_r = wp.transform_point(tendon_body_pose(body_q, body_r), seg_attachment_r_local[seg])
    length_rate = tendon_segment_length_rate_from_poses(
        dt,
        body_q,
        body_q_prev,
        body_com,
        tendon_guide_body,
        tendon_guide_type,
        tendon_guide_offset,
        tendon_guide_axis,
        guide_l,
        guide_r,
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
    tendon_guide_body: wp.array[int],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_rest_length: wp.array[float],
    seg_active_compliance: wp.array[float],
    seg_active_damping: wp.array[float],
    seg_active: wp.array[int],
    seg_active_guide_l: wp.array[int],
    seg_active_guide_r: wp.array[int],
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

    guide_l = seg_active_guide_l[seg]
    guide_r = seg_active_guide_r[seg]
    body_l = tendon_guide_body[guide_l]
    body_r = tendon_guide_body[guide_r]

    attachment_l = wp.transform_point(tendon_body_pose(body_q, body_l), seg_attachment_l_local[seg])
    attachment_r = wp.transform_point(tendon_body_pose(body_q, body_r), seg_attachment_r_local[seg])
    seg_attachment_l[seg] = attachment_l
    seg_attachment_r[seg] = attachment_r
    length = wp.length(attachment_r - attachment_l)
    length_rate = (length - seg_length_prev[seg]) / dt
    compliance = wp.max(seg_active_compliance[seg], _MIN_TENDON_COMPLIANCE)
    seg_material_tension[seg] = tendon_material_tension(length, seg_rest_length[seg], compliance)
    seg_damping_tension[seg] = seg_active_damping[seg] * length_rate


@wp.func
def _tendon_span_tension(
    seg: int,
    dt: float,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    tendon_guide_body: wp.array[int],
    tendon_guide_type: wp.array[int],
    tendon_guide_offset: wp.array[wp.vec3],
    tendon_guide_axis: wp.array[wp.vec3],
    seg_rest_length: wp.array[float],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_active_compliance: wp.array[float],
    seg_active_damping: wp.array[float],
    seg_active_guide_l: wp.array[int],
    seg_active_guide_r: wp.array[int],
) -> float:
    """Evaluate the selected VBD material law at the current trial poses."""
    guide_l = seg_active_guide_l[seg]
    guide_r = seg_active_guide_r[seg]
    attachment_l = wp.transform_point(tendon_body_pose(body_q, tendon_guide_body[guide_l]), seg_attachment_l_local[seg])
    attachment_r = wp.transform_point(tendon_body_pose(body_q, tendon_guide_body[guide_r]), seg_attachment_r_local[seg])
    length = wp.length(attachment_r - attachment_l)
    rest_length = seg_rest_length[seg]
    if length <= 1.0e-8:
        return 0.0
    compliance = wp.max(seg_active_compliance[seg], _MIN_TENDON_COMPLIANCE)
    length_rate = tendon_segment_length_rate_from_poses(
        dt,
        body_q,
        body_q_prev,
        body_com,
        tendon_guide_body,
        tendon_guide_type,
        tendon_guide_offset,
        tendon_guide_axis,
        guide_l,
        guide_r,
        seg_attachment_l_local[seg],
        seg_attachment_r_local[seg],
        attachment_l,
        attachment_r,
    )
    stiffness = 1.0 / compliance
    material_tension = stiffness * (length - rest_length)
    return wp.max(material_tension + seg_active_damping[seg] * length_rate, 0.0)


@wp.func
def _rolling_spin_axis_component(
    dt: float,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    tendon_guide_body: wp.array[int],
    tendon_guide_type: wp.array[int],
    tendon_guide_offset: wp.array[wp.vec3],
    tendon_guide_axis: wp.array[wp.vec3],
    tendon_guide_cone_seg_l: wp.array[int],
    tendon_guide_cone_seg_r: wp.array[int],
    tendon_guide_cap_ratio: wp.array[float],
    seg_rest_length: wp.array[float],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_active_compliance: wp.array[float],
    seg_active_damping: wp.array[float],
    seg_active: wp.array[int],
    seg_active_guide_l: wp.array[int],
    seg_active_guide_r: wp.array[int],
    guide: int,
    seg: int,
    tension: float,
    attachment: wp.vec3,
    direction: wp.vec3,
) -> wp.vec3:
    """Remove only the rim moment forbidden by the material cone.

    Bound the combined axial moment at the current trial pose. Cached contact
    points can cease to be tangent after another body updates, so a bound on
    tension difference alone does not bound the moment.
    """
    if tendon_guide_type[guide] != int(TendonGuideType.ROLLER):
        return wp.vec3(0.0)
    seg_left = tendon_guide_cone_seg_l[guide]
    seg_right = tendon_guide_cone_seg_r[guide]
    if seg_left < 0 or seg_right < 0 or seg_left >= seg_active.shape[0] or seg_right >= seg_active.shape[0]:
        return wp.vec3(0.0)
    if seg_active[seg_left] == 0 or seg_active[seg_right] == 0:
        return wp.vec3(0.0)
    if seg_active_guide_r[seg_left] != guide or seg_active_guide_l[seg_right] != guide:
        return wp.vec3(0.0)

    pose = tendon_body_pose(body_q, tendon_guide_body[guide])
    center = wp.transform_point(pose, tendon_guide_offset[guide])
    normal = wp.normalize(wp.transform_vector(pose, tendon_guide_axis[guide]))
    radial = attachment - center
    moment = wp.dot(wp.cross(radial, direction), normal)
    spin_scale = float(0.0)
    # Contact points stay fixed in the roller's local frame across all colors,
    # so their angle stays valid for this frozen route. Retangent next iteration
    # and at the accepted pose; forces below still use current body trial poses.
    cap_ratio = tendon_guide_cap_ratio[guide]
    if cap_ratio > 1.0:
        # The caller already evaluated this span at the same trial poses.
        neighbor = seg_right if seg == seg_left else seg_left
        neighbor_tension = _tendon_span_tension(
            neighbor,
            dt,
            body_q,
            body_q_prev,
            body_com,
            tendon_guide_body,
            tendon_guide_type,
            tendon_guide_offset,
            tendon_guide_axis,
            seg_rest_length,
            seg_attachment_l_local,
            seg_attachment_r_local,
            seg_active_compliance,
            seg_active_damping,
            seg_active_guide_l,
            seg_active_guide_r,
        )
        neighbor_l = wp.transform_point(
            tendon_body_pose(body_q, tendon_guide_body[seg_active_guide_l[neighbor]]),
            seg_attachment_l_local[neighbor],
        )
        neighbor_r = wp.transform_point(
            tendon_body_pose(body_q, tendon_guide_body[seg_active_guide_r[neighbor]]),
            seg_attachment_r_local[neighbor],
        )
        neighbor_direction = wp.normalize(neighbor_r - neighbor_l)
        neighbor_attachment = neighbor_l if neighbor == seg_right else neighbor_r
        neighbor_moment = wp.dot(wp.cross(neighbor_attachment - center, neighbor_direction), normal)
        beta = (cap_ratio - 1.0) / (cap_ratio + 1.0)
        bound = wp.length(radial) * beta * (tension + neighbor_tension)
        # The two spans pull in opposite route directions at this guide.
        net_moment = tension * moment - neighbor_tension * neighbor_moment
        spin_scale = float(1.0)
        if wp.abs(net_moment) > bound:
            spin_scale = bound / wp.abs(net_moment)

    return (1.0 - spin_scale) * moment * normal


@wp.func
def evaluate_tendon_force_hessians(
    body: int,
    dt: float,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    adjacency: TendonForceElementAdjacencyInfo,
    tendon_guide_body: wp.array[int],
    tendon_guide_type: wp.array[int],
    tendon_guide_offset: wp.array[wp.vec3],
    tendon_guide_axis: wp.array[wp.vec3],
    tendon_guide_cone_seg_l: wp.array[int],
    tendon_guide_cone_seg_r: wp.array[int],
    tendon_guide_cap_ratio: wp.array[float],
    seg_rest_length: wp.array[float],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_active_compliance: wp.array[float],
    seg_active_damping: wp.array[float],
    seg_active: wp.array[int],
    seg_active_guide_l: wp.array[int],
    seg_active_guide_r: wp.array[int],
):
    """Evaluate unilateral tendon spring-damper forces for one VBD body.

    Freeze the capstan limiter in the positive-semidefinite local Hessian
    approximation; its derivative is not included.
    """
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

        guide_l = seg_active_guide_l[seg]
        guide_r = seg_active_guide_r[seg]
        body_l = tendon_guide_body[guide_l]
        body_r = tendon_guide_body[guide_r]
        if body != body_l and body != body_r:
            continue

        attachment_l = wp.transform_point(tendon_body_pose(body_q, body_l), seg_attachment_l_local[seg])
        attachment_r = wp.transform_point(tendon_body_pose(body_q, body_r), seg_attachment_r_local[seg])
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
            tendon_guide_body,
            tendon_guide_type,
            tendon_guide_offset,
            tendon_guide_axis,
            guide_l,
            guide_r,
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
                dt,
                body_q,
                body_q_prev,
                body_com,
                tendon_guide_body,
                tendon_guide_type,
                tendon_guide_offset,
                tendon_guide_axis,
                tendon_guide_cone_seg_l,
                tendon_guide_cone_seg_r,
                tendon_guide_cap_ratio,
                seg_rest_length,
                seg_attachment_l_local,
                seg_attachment_r_local,
                seg_active_compliance,
                seg_active_damping,
                seg_active,
                seg_active_guide_l,
                seg_active_guide_r,
                guide_l,
                seg,
                tension,
                attachment_l,
                direction,
            )
            fix_r = _rolling_spin_axis_component(
                dt,
                body_q,
                body_q_prev,
                body_com,
                tendon_guide_body,
                tendon_guide_type,
                tendon_guide_offset,
                tendon_guide_axis,
                tendon_guide_cone_seg_l,
                tendon_guide_cone_seg_r,
                tendon_guide_cap_ratio,
                seg_rest_length,
                seg_attachment_l_local,
                seg_attachment_r_local,
                seg_active_compliance,
                seg_active_damping,
                seg_active,
                seg_active_guide_l,
                seg_active_guide_r,
                guide_r,
                seg,
                tension,
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
            guide = guide_l
            force_sign = 1.0
            body_force = tension * direction
        else:
            attachment = attachment_r
            guide = guide_r
            force_sign = -1.0
            body_force = -tension * direction

        moment_arm = attachment - world_com
        moment_axis = wp.cross(moment_arm, direction)

        if tendon_guide_type[guide] == int(TendonGuideType.ROLLER):
            # Free-span tension still loads the body, but only capstan friction
            # transmits the rolling-axis part of its moment.
            moment_axis = moment_axis - _rolling_spin_axis_component(
                dt,
                body_q,
                body_q_prev,
                body_com,
                tendon_guide_body,
                tendon_guide_type,
                tendon_guide_offset,
                tendon_guide_axis,
                tendon_guide_cone_seg_l,
                tendon_guide_cone_seg_r,
                tendon_guide_cap_ratio,
                seg_rest_length,
                seg_attachment_l_local,
                seg_attachment_r_local,
                seg_active_compliance,
                seg_active_damping,
                seg_active,
                seg_active_guide_l,
                seg_active_guide_r,
                guide,
                seg,
                tension,
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
