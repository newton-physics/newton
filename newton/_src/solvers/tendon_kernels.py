# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver-neutral routed tendon geometry kernels."""

from typing import Any

import warp as wp

from ..core.reset import reset_world_selected
from ..math import quat_velocity
from ..sim.tendon import TendonLinkFlags, TendonLinkType


@wp.func
def tendon_material_tension(length: float, rest_length: float, compliance: float) -> float:
    """Evaluate unilateral tendon tension from linear compliance."""
    return wp.max(length - rest_length, 0.0) / wp.max(compliance, 1.0e-30)


@wp.func
def tangent_point_circle(
    p: wp.vec3,
    center: wp.vec3,
    radius: float,
    plane_normal: wp.vec3,
    orientation: int,
) -> wp.vec3:
    """Compute the tangent point on a circle from an external point."""
    d = center - p
    d_proj = d - wp.dot(d, plane_normal) * plane_normal
    dist_in_plane = wp.length(d_proj)
    if dist_in_plane <= radius:
        if dist_in_plane < 1.0e-8:
            basis = wp.vec3(1.0, 0.0, 0.0)
            if wp.abs(plane_normal[0]) > 0.9:
                basis = wp.vec3(0.0, 1.0, 0.0)
            fallback = basis - wp.dot(basis, plane_normal) * plane_normal
            return center + wp.normalize(fallback) * radius
        return center - wp.normalize(d_proj) * radius

    u = d_proj / dist_in_plane
    v = wp.cross(plane_normal, u)
    phi = wp.asin(wp.min(radius / dist_in_plane, 1.0))

    if orientation > 0:
        angle = -1.5707963 - phi
    else:
        angle = 1.5707963 + phi

    return center + radius * (wp.cos(angle) * u + wp.sin(angle) * v)


@wp.func
def signed_arc_length(
    old_pt: wp.vec3,
    new_pt: wp.vec3,
    center: wp.vec3,
    radius: float,
    plane_normal: wp.vec3,
    orientation: int,
) -> float:
    """Signed arc length from old_pt to new_pt on a circular rolling link."""
    r_old = old_pt - center
    r_new = new_pt - center
    r_old = r_old - wp.dot(r_old, plane_normal) * plane_normal
    r_new = r_new - wp.dot(r_new, plane_normal) * plane_normal
    len_old = wp.length(r_old)
    len_new = wp.length(r_new)
    if len_old < 1.0e-8 or len_new < 1.0e-8 or radius <= 0.0:
        return 0.0

    u_old = r_old / len_old
    u_new = r_new / len_new
    cross_val = wp.dot(wp.cross(u_new, u_old), plane_normal)
    dot_val = wp.dot(u_old, u_new)
    angle = wp.atan2(cross_val, dot_val)
    return angle * radius * float(orientation)


@wp.func
def wrapped_arc_length(
    pt_left: wp.vec3,
    pt_right: wp.vec3,
    center: wp.vec3,
    radius: float,
    plane_normal: wp.vec3,
) -> float:
    """Unsigned arc length between two points on a circular rolling link."""
    r_left = pt_left - center
    r_right = pt_right - center
    r_left = r_left - wp.dot(r_left, plane_normal) * plane_normal
    r_right = r_right - wp.dot(r_right, plane_normal) * plane_normal
    len_left = wp.length(r_left)
    len_right = wp.length(r_right)
    if radius <= 0.0 or len_left <= 1.0e-8 or len_right <= 1.0e-8:
        return 0.0

    u_left = r_left / len_left
    u_right = r_right / len_right
    theta = wp.abs(wp.atan2(wp.dot(wp.cross(u_left, u_right), plane_normal), wp.dot(u_left, u_right)))
    return theta * radius


@wp.kernel
def snapshot_tendon_link_active(
    tendon_link_active: wp.array[bool],
    tendon_link_active_step: wp.array[bool],
    tendon_link_flags: wp.array[int],
):
    """Snapshot dynamic state while preserving fixed route ownership."""
    link_idx = wp.tid()
    if (tendon_link_flags[link_idx] & int(TendonLinkFlags.DYNAMIC)) != 0:
        tendon_link_active_step[link_idx] = tendon_link_active[link_idx]
    else:
        tendon_link_active[link_idx] = True
        tendon_link_active_step[link_idx] = True


@wp.kernel(module="unique", enable_backward=False)
def reset_tendon_state_array(
    entity_world: wp.array[int],
    world_mask: wp.array[wp.bool],
    world_count: int,
    initial: wp.array[Any],
    current: wp.array[Any],
):
    """Restore selected rows of one mutable tendon-state array."""
    index = wp.tid()
    if reset_world_selected(entity_world[index], world_mask, world_count):
        current[index] = initial[index]


@wp.kernel
def mark_tendon_rebaseline_worlds(
    world_mask: wp.array[wp.bool],
    rebaseline_world_mask: wp.array[wp.bool],
):
    """Mark reset-selected worlds for tendon geometry rebaselining."""
    world = wp.tid()
    if world_mask[world]:
        rebaseline_world_mask[world] = True


@wp.kernel
def update_tendon_link_active(
    body_q: wp.array[wp.transform],
    tendon_start: wp.array[int],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_flags: wp.array[int],
    tendon_link_radius: wp.array[float],
    tendon_link_orientation: wp.array[int],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    tendon_activation_tol: float,
    tendon_link_active: wp.array[bool],
):
    """Update dynamic rolling links from oriented distance to their bypass span."""
    tendon_id = wp.tid()
    link_start = tendon_start[tendon_id]
    link_end = tendon_start[tendon_id + 1]

    for link_idx in range(link_start + 1, link_end - 1):
        if tendon_link_type[link_idx] != int(TendonLinkType.ROLLING):
            continue
        if (tendon_link_flags[link_idx] & int(TendonLinkFlags.DYNAMIC)) == 0 or tendon_link_radius[link_idx] <= 0.0:
            continue

        prev_link = link_idx - 1
        next_link = link_idx + 1

        prev_pose = body_q[tendon_link_body[prev_link]]
        next_pose = body_q[tendon_link_body[next_link]]
        candidate_pose = body_q[tendon_link_body[link_idx]]
        prev_center = wp.transform_point(prev_pose, tendon_link_offset[prev_link])
        next_center = wp.transform_point(next_pose, tendon_link_offset[next_link])
        candidate_center = wp.transform_point(candidate_pose, tendon_link_offset[link_idx])

        prev_rolling = (
            tendon_link_type[prev_link] == int(TendonLinkType.ROLLING)
            and tendon_link_active[prev_link]
            and tendon_link_radius[prev_link] > 0.0
        )
        next_rolling = (
            tendon_link_type[next_link] == int(TendonLinkType.ROLLING)
            and tendon_link_active[next_link]
            and tendon_link_radius[next_link] > 0.0
        )

        bypass_l = prev_center
        bypass_r = next_center
        if prev_rolling and next_rolling:
            prev_normal = wp.transform_vector(prev_pose, tendon_link_axis[prev_link])
            next_normal = wp.transform_vector(next_pose, tendon_link_axis[next_link])
            for _iter in range(10):
                bypass_r = tangent_point_circle(
                    bypass_l,
                    next_center,
                    tendon_link_radius[next_link],
                    next_normal,
                    tendon_link_orientation[next_link],
                )
                bypass_l = tangent_point_circle(
                    bypass_r,
                    prev_center,
                    tendon_link_radius[prev_link],
                    prev_normal,
                    -tendon_link_orientation[prev_link],
                )
        elif prev_rolling:
            prev_normal = wp.transform_vector(prev_pose, tendon_link_axis[prev_link])
            bypass_l = tangent_point_circle(
                next_center,
                prev_center,
                tendon_link_radius[prev_link],
                prev_normal,
                -tendon_link_orientation[prev_link],
            )
        elif next_rolling:
            next_normal = wp.transform_vector(next_pose, tendon_link_axis[next_link])
            bypass_r = tangent_point_circle(
                prev_center,
                next_center,
                tendon_link_radius[next_link],
                next_normal,
                tendon_link_orientation[next_link],
            )

        # Routing is defined in the candidate's cable plane. Projecting the
        # bypass span makes the test independent of harmless out-of-plane drift.
        normal = wp.transform_vector(candidate_pose, tendon_link_axis[link_idx])
        normal_length = wp.length(normal)
        if normal_length <= 1.0e-8:
            tendon_link_active[link_idx] = False
            continue
        normal = normal / normal_length
        span = bypass_r - bypass_l
        candidate_offset = candidate_center - bypass_l
        span = span - wp.dot(span, normal) * normal
        candidate_offset = candidate_offset - wp.dot(candidate_offset, normal) * normal
        span_length_sq = wp.dot(span, span)
        active = False
        if span_length_sq > 1.0e-12:
            alpha = wp.dot(candidate_offset, span) / span_length_sq
            closest_offset = alpha * span
            span_normal = wp.cross(normal, span) / wp.sqrt(span_length_sq)
            # Orient the signed distance so it is positive on the inactive side.
            distance = wp.dot(candidate_offset - closest_offset, span_normal)
            if tendon_link_orientation[link_idx] <= 0:
                distance = -distance
            activation_radius = tendon_link_radius[link_idx]
            if not tendon_link_active[link_idx]:
                activation_radius = activation_radius * (1.0 - tendon_activation_tol)
            if alpha > 0.0 and alpha < 1.0 and distance <= activation_radius:
                active = True
        tendon_link_active[link_idx] = active


@wp.func
def _tendon_segment_length_rate_from_twists(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    link_l: int,
    link_r: int,
    x_l: wp.vec3,
    x_r: wp.vec3,
    velocity_l: wp.vec3,
    velocity_r: wp.vec3,
    omega_l: wp.vec3,
    omega_r: wp.vec3,
):
    """Return free-span length rate from endpoint body twists."""
    diff = x_r - x_l
    length = wp.length(diff)
    if length <= 1.0e-8:
        return 0.0

    direction = diff / length
    body_l = tendon_link_body[link_l]
    body_r = tendon_link_body[link_r]
    pose_l = body_q[body_l]
    pose_r = body_q[body_r]
    world_com_l = wp.transform_point(pose_l, body_com[body_l])
    world_com_r = wp.transform_point(pose_r, body_com[body_r])

    linear_l = -direction
    linear_r = direction
    angular_l = -wp.cross(x_l - world_com_l, direction)
    angular_r = wp.cross(x_r - world_com_r, direction)

    # Rolling contact does not transmit spin about the roller axis.
    if tendon_link_type[link_l] == int(TendonLinkType.ROLLING):
        center_l = wp.transform_point(pose_l, tendon_link_offset[link_l])
        normal_l = wp.transform_vector(pose_l, tendon_link_axis[link_l])
        radial_l = x_l - center_l
        angular_l = angular_l - wp.dot(wp.cross(radial_l, linear_l), normal_l) * normal_l
    if tendon_link_type[link_r] == int(TendonLinkType.ROLLING):
        center_r = wp.transform_point(pose_r, tendon_link_offset[link_r])
        normal_r = wp.transform_vector(pose_r, tendon_link_axis[link_r])
        radial_r = x_r - center_r
        angular_r = angular_r - wp.dot(wp.cross(radial_r, linear_r), normal_r) * normal_r

    return (
        wp.dot(linear_l, velocity_l)
        + wp.dot(linear_r, velocity_r)
        + wp.dot(angular_l, omega_l)
        + wp.dot(angular_r, omega_r)
    )


@wp.func
def tendon_segment_length_rate(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    link_l: int,
    link_r: int,
    x_l: wp.vec3,
    x_r: wp.vec3,
):
    """Return the free-span length rate used by the stretch constraint."""
    body_l = tendon_link_body[link_l]
    body_r = tendon_link_body[link_r]
    return _tendon_segment_length_rate_from_twists(
        body_q,
        body_com,
        tendon_link_body,
        tendon_link_type,
        tendon_link_offset,
        tendon_link_axis,
        link_l,
        link_r,
        x_l,
        x_r,
        wp.spatial_top(body_qd[body_l]),
        wp.spatial_top(body_qd[body_r]),
        wp.spatial_bottom(body_qd[body_l]),
        wp.spatial_bottom(body_qd[body_r]),
    )


@wp.func
def tendon_segment_length_rate_from_poses(
    dt: float,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    link_l: int,
    link_r: int,
    x_l_local: wp.vec3,
    x_r_local: wp.vec3,
    x_l: wp.vec3,
    x_r: wp.vec3,
):
    """Return free-span length rate from the current and previous body poses."""
    body_l = tendon_link_body[link_l]
    body_r = tendon_link_body[link_r]
    pose_l = body_q[body_l]
    pose_r = body_q[body_r]
    pose_l_prev = body_q_prev[body_l]
    pose_r_prev = body_q_prev[body_r]
    # Preserve VBD's discrete damping except where rolling contact requires removing roller-axis spin.
    if tendon_link_type[link_l] != int(TendonLinkType.ROLLING) and tendon_link_type[link_r] != int(
        TendonLinkType.ROLLING
    ):
        x_l_prev = wp.transform_point(pose_l_prev, x_l_local)
        x_r_prev = wp.transform_point(pose_r_prev, x_r_local)
        return (wp.length(x_r - x_l) - wp.length(x_r_prev - x_l_prev)) / dt

    world_com_l = wp.transform_point(pose_l, body_com[body_l])
    world_com_r = wp.transform_point(pose_r, body_com[body_r])
    world_com_l_prev = wp.transform_point(pose_l_prev, body_com[body_l])
    world_com_r_prev = wp.transform_point(pose_r_prev, body_com[body_r])
    velocity_l = (world_com_l - world_com_l_prev) / dt
    velocity_r = (world_com_r - world_com_r_prev) / dt
    omega_l = quat_velocity(
        wp.transform_get_rotation(pose_l),
        wp.transform_get_rotation(pose_l_prev),
        dt,
    )
    omega_r = quat_velocity(
        wp.transform_get_rotation(pose_r),
        wp.transform_get_rotation(pose_r_prev),
        dt,
    )
    return _tendon_segment_length_rate_from_twists(
        body_q,
        body_com,
        tendon_link_body,
        tendon_link_type,
        tendon_link_offset,
        tendon_link_axis,
        link_l,
        link_r,
        x_l,
        x_r,
        velocity_l,
        velocity_r,
        omega_l,
        omega_r,
    )


@wp.kernel
def prepare_tendon_route(
    body_q: wp.array[wp.transform],
    tendon_start: wp.array[int],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_flags: wp.array[int],
    tendon_link_radius: wp.array[float],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    seg_authored_rest_length: wp.array[float],
    seg_rest_length_step: wp.array[float],
    seg_compliance: wp.array[float],
    seg_damping: wp.array[float],
    tendon_link_active: wp.array[bool],
    tendon_link_active_step: wp.array[bool],
    tendon_link_route_rest_length: wp.array[float],
    seg_attachment_l_local_step: wp.array[wp.vec3],
    seg_attachment_r_local_step: wp.array[wp.vec3],
    initialize: int,
    compliance_floor: float,
    # outputs
    seg_route_rest_length: wp.array[float],
    seg_active: wp.array[int],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
    seg_active_compliance: wp.array[float],
    seg_active_damping: wp.array[float],
):
    """Build the active segment route held fixed during one solver step."""
    tendon_id = wp.tid()
    link_start = tendon_start[tendon_id]
    link_end = tendon_start[tendon_id + 1]
    num_links = link_end - link_start
    num_segs = num_links - 1
    if num_segs < 1:
        return

    seg_offset = link_start - tendon_id
    min_rest = 1.0e-6

    for s in range(num_segs):
        seg = seg_offset + s
        seg_active[seg] = 1
        seg_active_link_l[seg] = link_start + s
        seg_active_link_r[seg] = link_start + s + 1
        seg_active_compliance[seg] = wp.max(seg_compliance[seg], compliance_floor)
        seg_active_damping[seg] = seg_damping[seg]
        seg_route_rest_length[seg] = seg_rest_length_step[seg]

    tendon_link_active[link_start] = True
    tendon_link_active[link_end - 1] = True

    for i in range(1, num_links - 1):
        link_idx = link_start + i
        if (tendon_link_flags[link_idx] & int(TendonLinkFlags.DYNAMIC)) == 0:
            tendon_link_active[link_idx] = True
            tendon_link_active_step[link_idx] = True
        if tendon_link_type[link_idx] != int(TendonLinkType.ROLLING):
            continue
        if tendon_link_route_rest_length[link_idx] <= 0.0:
            continue
        if tendon_link_active[link_idx]:
            continue

        seg_left = seg_offset + i - 1
        seg_right = seg_left + 1
        prev_link = link_idx - 1
        next_link = link_idx + 1

        seg_active[seg_left] = 1
        seg_active[seg_right] = 0
        seg_active_link_l[seg_left] = prev_link
        seg_active_link_r[seg_left] = next_link
        seg_active_compliance[seg_left] = wp.max(seg_compliance[seg_left], compliance_floor) + wp.max(
            seg_compliance[seg_right], compliance_floor
        )
        seg_active_damping[seg_left] = seg_damping[seg_left] + seg_damping[seg_right]

        merged_rest = seg_rest_length_step[seg_left]
        if tendon_link_active_step[link_idx]:
            body = tendon_link_body[link_idx]
            pose = body_q[body]
            center = wp.transform_point(pose, tendon_link_offset[link_idx])
            normal = wp.transform_vector(pose, tendon_link_axis[link_idx])
            pt_left = wp.transform_point(pose, seg_attachment_r_local_step[seg_left])
            pt_right = wp.transform_point(pose, seg_attachment_l_local_step[seg_right])
            merged_rest = (
                seg_rest_length_step[seg_left]
                + seg_rest_length_step[seg_right]
                + wrapped_arc_length(pt_left, pt_right, center, tendon_link_radius[link_idx], normal)
            )
        elif initialize != 0:
            merged_rest = seg_rest_length_step[seg_left] + seg_rest_length_step[seg_right]
            auto_left = seg_authored_rest_length[seg_left] < 0.0
            auto_right = seg_authored_rest_length[seg_right] < 0.0
            if auto_left or auto_right:
                # Automatic material fills the unresolved part of the initial bypass.
                merged_rest = wp.max(merged_rest, tendon_link_route_rest_length[link_idx])
        elif merged_rest <= 0.0:
            merged_rest = tendon_link_route_rest_length[link_idx]

        seg_route_rest_length[seg_left] = wp.max(merged_rest, min_rest)
        seg_route_rest_length[seg_right] = min_rest


@wp.func
def _tendon_segment_attachment_points(
    center_l: wp.vec3,
    center_r: wp.vec3,
    normal_l: wp.vec3,
    normal_r: wp.vec3,
    type_l: int,
    type_r: int,
    radius_l: float,
    radius_r: float,
    orient_l: int,
    orient_r: int,
    seed_al: wp.vec3,
    seed_ar: wp.vec3,
):
    """Compute one active free span's tangent points."""
    new_al = center_l
    new_ar = center_r
    both_rolling = (type_l == int(TendonLinkType.ROLLING)) and (type_r == int(TendonLinkType.ROLLING))

    if both_rolling and radius_l > 0.0 and radius_r > 0.0:
        new_al = seed_al
        new_ar = seed_ar
        for _iter in range(10):
            previous_al = new_al
            previous_ar = new_ar
            new_ar = tangent_point_circle(new_al, center_r, radius_r, normal_r, orient_r)
            new_al = tangent_point_circle(new_ar, center_l, radius_l, normal_l, -orient_l)
            tangent_delta_sq = wp.length_sq(new_al - previous_al) + wp.length_sq(new_ar - previous_ar)
            if tangent_delta_sq == 0.0:
                break
    elif type_l == int(TendonLinkType.ROLLING) and radius_l > 0.0:
        new_ar = center_r
        new_al = tangent_point_circle(center_r, center_l, radius_l, normal_l, -orient_l)
    elif type_r == int(TendonLinkType.ROLLING) and radius_r > 0.0:
        new_al = center_l
        new_ar = tangent_point_circle(center_l, center_r, radius_r, normal_r, orient_r)

    return new_al, new_ar


@wp.kernel
def update_tendon_attachments(
    body_q: wp.array[wp.transform],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_flags: wp.array[int],
    tendon_link_radius: wp.array[float],
    tendon_link_orientation: wp.array[int],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    seg_active: wp.array[int],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
    tendon_link_active: wp.array[bool],
    tendon_link_active_step: wp.array[bool],
    seg_attachment_l_local_step: wp.array[wp.vec3],
    seg_attachment_r_local_step: wp.array[wp.vec3],
    apply_rolling_transfer: int,
    # outputs
    seg_attachment_l: wp.array[wp.vec3],
    seg_attachment_r: wp.array[wp.vec3],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_rolling_delta_l: wp.array[float],
    seg_rolling_delta_r: wp.array[float],
    seg_length: wp.array[float],
):
    """Update one active free span's tangent geometry."""
    seg = wp.tid()
    seg_rolling_delta_l[seg] = 0.0
    seg_rolling_delta_r[seg] = 0.0
    if seg_active[seg] == 0:
        seg_attachment_l[seg] = wp.vec3(0.0, 0.0, 0.0)
        seg_attachment_r[seg] = wp.vec3(0.0, 0.0, 0.0)
        seg_attachment_l_local[seg] = wp.vec3(0.0, 0.0, 0.0)
        seg_attachment_r_local[seg] = wp.vec3(0.0, 0.0, 0.0)
        seg_length[seg] = 0.0
        return

    link_l = seg_active_link_l[seg]
    link_r = seg_active_link_r[seg]

    body_l = tendon_link_body[link_l]
    body_r = tendon_link_body[link_r]
    type_l = tendon_link_type[link_l]
    type_r = tendon_link_type[link_r]
    radius_l = tendon_link_radius[link_l]
    radius_r = tendon_link_radius[link_r]
    orient_l = tendon_link_orientation[link_l]
    orient_r = tendon_link_orientation[link_r]
    offset_l = tendon_link_offset[link_l]
    offset_r = tendon_link_offset[link_r]
    axis_l = tendon_link_axis[link_l]
    axis_r = tendon_link_axis[link_r]

    pose_l = body_q[body_l]
    pose_r = body_q[body_r]
    center_l = wp.transform_point(pose_l, offset_l)
    center_r = wp.transform_point(pose_r, offset_r)
    normal_l = wp.transform_vector(pose_l, axis_l)
    normal_r = wp.transform_vector(pose_r, axis_r)

    seed_al = wp.transform_point(pose_l, seg_attachment_l_local[seg])
    seed_ar = wp.transform_point(pose_r, seg_attachment_r_local[seg])
    base_al = wp.transform_point(pose_l, seg_attachment_l_local_step[seg])
    base_ar_step_seg = seg

    # Route transitions move the persistent right endpoint between adjacent segment slots.
    transition_link = link_l + 1 if link_r == link_l + 2 else link_l
    if (
        tendon_link_type[transition_link] == int(TendonLinkType.ROLLING)
        and (tendon_link_flags[transition_link] & int(TendonLinkFlags.DYNAMIC)) != 0
    ):
        if tendon_link_active[transition_link] and not tendon_link_active_step[transition_link]:
            assert transition_link == link_l and seg > 0
            base_ar_step_seg = seg - 1
        elif not tendon_link_active[transition_link] and tendon_link_active_step[transition_link]:
            assert transition_link == link_l + 1 and link_r == transition_link + 1
            base_ar_step_seg = seg + 1

    base_ar = wp.transform_point(pose_r, seg_attachment_r_local_step[base_ar_step_seg])

    new_al, new_ar = _tendon_segment_attachment_points(
        center_l,
        center_r,
        normal_l,
        normal_r,
        type_l,
        type_r,
        radius_l,
        radius_r,
        orient_l,
        orient_r,
        seed_al,
        seed_ar,
    )

    if apply_rolling_transfer != 0:
        if (
            type_l == int(TendonLinkType.ROLLING)
            and radius_l > 0.0
            and tendon_link_active[link_l] == tendon_link_active_step[link_l]
        ):
            seg_rolling_delta_l[seg] = signed_arc_length(base_al, new_al, center_l, radius_l, normal_l, orient_l)

        if (
            type_r == int(TendonLinkType.ROLLING)
            and radius_r > 0.0
            and tendon_link_active[link_r] == tendon_link_active_step[link_r]
        ):
            seg_rolling_delta_r[seg] = -signed_arc_length(base_ar, new_ar, center_r, radius_r, normal_r, orient_r)

    seg_attachment_l[seg] = new_al
    seg_attachment_r[seg] = new_ar
    seg_attachment_l_local[seg] = wp.transform_point(wp.transform_inverse(pose_l), new_al)
    seg_attachment_r_local[seg] = wp.transform_point(wp.transform_inverse(pose_r), new_ar)
    seg_length[seg] = wp.length(new_ar - new_al)


@wp.kernel
def rebaseline_tendon_attachments(
    body_q: wp.array[wp.transform],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_radius: wp.array[float],
    tendon_link_orientation: wp.array[int],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    seg_active: wp.array[int],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
    seg_world: wp.array[int],
    rebaseline_world_mask: wp.array[wp.bool],
    world_count: int,
    # inputs/outputs
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_attachment_l_local_step: wp.array[wp.vec3],
    seg_attachment_r_local_step: wp.array[wp.vec3],
    # outputs
    seg_attachment_l: wp.array[wp.vec3],
    seg_attachment_r: wp.array[wp.vec3],
    seg_rolling_delta_l: wp.array[float],
    seg_rolling_delta_r: wp.array[float],
    seg_length: wp.array[float],
):
    """Rebaseline reset-selected tangent history from the accepted pose."""
    seg = wp.tid()
    if not reset_world_selected(seg_world[seg], rebaseline_world_mask, world_count):
        return

    seg_rolling_delta_l[seg] = 0.0
    seg_rolling_delta_r[seg] = 0.0
    if seg_active[seg] == 0:
        zero = wp.vec3(0.0, 0.0, 0.0)
        seg_attachment_l[seg] = zero
        seg_attachment_r[seg] = zero
        seg_attachment_l_local[seg] = zero
        seg_attachment_r_local[seg] = zero
        seg_attachment_l_local_step[seg] = zero
        seg_attachment_r_local_step[seg] = zero
        seg_length[seg] = 0.0
        return

    link_l = seg_active_link_l[seg]
    link_r = seg_active_link_r[seg]
    body_l = tendon_link_body[link_l]
    body_r = tendon_link_body[link_r]
    pose_l = body_q[body_l]
    pose_r = body_q[body_r]
    center_l = wp.transform_point(pose_l, tendon_link_offset[link_l])
    center_r = wp.transform_point(pose_r, tendon_link_offset[link_r])
    normal_l = wp.transform_vector(pose_l, tendon_link_axis[link_l])
    normal_r = wp.transform_vector(pose_r, tendon_link_axis[link_r])
    seed_al = wp.transform_point(pose_l, seg_attachment_l_local[seg])
    seed_ar = wp.transform_point(pose_r, seg_attachment_r_local[seg])
    new_al, new_ar = _tendon_segment_attachment_points(
        center_l,
        center_r,
        normal_l,
        normal_r,
        tendon_link_type[link_l],
        tendon_link_type[link_r],
        tendon_link_radius[link_l],
        tendon_link_radius[link_r],
        tendon_link_orientation[link_l],
        tendon_link_orientation[link_r],
        seed_al,
        seed_ar,
    )
    local_l = wp.transform_point(wp.transform_inverse(pose_l), new_al)
    local_r = wp.transform_point(wp.transform_inverse(pose_r), new_ar)
    seg_attachment_l[seg] = new_al
    seg_attachment_r[seg] = new_ar
    seg_attachment_l_local[seg] = local_l
    seg_attachment_r_local[seg] = local_r
    seg_attachment_l_local_step[seg] = local_l
    seg_attachment_r_local_step[seg] = local_r
    seg_length[seg] = wp.length(new_ar - new_al)


@wp.kernel
def update_tendon_cone_rows(
    body_q: wp.array[wp.transform],
    tendon_start: wp.array[int],
    tendon_link_tendon: wp.array[int],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_flags: wp.array[int],
    tendon_link_radius: wp.array[float],
    tendon_link_orientation: wp.array[int],
    tendon_link_mu: wp.array[float],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    tendon_link_active: wp.array[bool],
    seg_active: wp.array[int],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
    seg_attachment_l: wp.array[wp.vec3],
    seg_attachment_r: wp.array[wp.vec3],
    seg_length: wp.array[float],
    report_unsupported_wrap: int,
    # outputs
    tendon_link_cone_seg_l: wp.array[int],
    tendon_link_cone_seg_r: wp.array[int],
    tendon_link_cap_ratio: wp.array[float],
):
    """Cache one capstan-cone row's segment pair and tension ratio."""
    link_idx = wp.tid()
    link_type = tendon_link_type[link_idx]
    link_is_active = tendon_link_active[link_idx]
    is_rolling = link_type == int(TendonLinkType.ROLLING) and link_is_active
    is_pinhole = link_type == int(TendonLinkType.PINHOLE)

    tendon_link_cone_seg_l[link_idx] = -1
    tendon_link_cone_seg_r[link_idx] = -1
    tendon_link_cap_ratio[link_idx] = 1.0

    if not (is_rolling or is_pinhole):
        return

    tendon_id = tendon_link_tendon[link_idx]
    link_start = tendon_start[tendon_id]
    link_end = tendon_start[tendon_id + 1]
    i = link_idx - link_start
    num_links = link_end - link_start
    num_segs = num_links - 1
    if i <= 0 or i >= num_links - 1:
        return

    seg_offset = link_start - tendon_id
    seg_adj_left = seg_offset + i - 1
    seg_adj_right = seg_offset + i
    seg_left = int(-1)
    seg_right = int(-1)
    if is_rolling:
        if seg_active[seg_adj_left] != 0 and seg_active[seg_adj_right] != 0:
            seg_left = seg_adj_left
            seg_right = seg_adj_right
        else:
            if seg_active[seg_adj_left] != 0:
                seg_left = seg_adj_left
            elif i > 1:
                previous_seg = seg_adj_left - 1
                if seg_active[previous_seg] != 0 and seg_active_link_r[previous_seg] == link_idx:
                    seg_left = previous_seg

            if seg_active[seg_adj_right] != 0:
                seg_right = seg_adj_right
            elif i < num_links - 2:
                next_seg = seg_adj_right + 1
                if seg_active[next_seg] != 0 and seg_active_link_l[next_seg] == link_idx:
                    seg_right = next_seg
    else:
        for probe in range(num_segs):
            left_candidate = i - 1 - probe
            if left_candidate >= 0 and seg_left < 0:
                seg = seg_offset + left_candidate
                if seg_active[seg] != 0:
                    if seg_length[seg] > 1.0e-5:
                        seg_left = seg

            right_candidate = i + probe
            if right_candidate < num_segs and seg_right < 0:
                seg = seg_offset + right_candidate
                if seg_active[seg] != 0:
                    if seg_length[seg] > 1.0e-5:
                        seg_right = seg

    if seg_left < 0 or seg_right < 0:
        return

    is_dynamic = (tendon_link_flags[link_idx] & int(TendonLinkFlags.DYNAMIC)) != 0
    if is_rolling and not is_dynamic and report_unsupported_wrap != 0:
        body = tendon_link_body[link_idx]
        pose = body_q[body]
        center = wp.transform_point(pose, tendon_link_offset[link_idx])
        normal = wp.transform_vector(pose, tendon_link_axis[link_idx])
        r_left = seg_attachment_r[seg_left] - center
        r_right = seg_attachment_l[seg_right] - center
        r_left = r_left - wp.dot(r_left, normal) * normal
        r_right = r_right - wp.dot(r_right, normal) * normal
        len_r_left = wp.length(r_left)
        len_r_right = wp.length(r_right)
        if len_r_left > 1.0e-8 and len_r_right > 1.0e-8:
            u_left = r_left / len_r_left
            u_right = r_right / len_r_right
            signed_wrap_angle = wp.atan2(wp.dot(wp.cross(u_left, u_right), normal), wp.dot(u_left, u_right))
            oriented_wrap_angle = signed_wrap_angle * float(tendon_link_orientation[link_idx])
            # atan2 may represent the supported pi boundary as either +pi or -pi.
            wrap_tol = 1.0e-6
            if oriented_wrap_angle < -wrap_tol and oriented_wrap_angle > -wp.pi + wrap_tol:
                wp.printf(
                    "ERROR: Tendon %d ROLLING link %d crossed the supported wrap range [0, pi] "
                    "(oriented angle %f deg). Cable rest length and tension may be invalid. "
                    "Correct the link orientation or route geometry, or use dynamic routing.\n",
                    tendon_id,
                    link_idx,
                    oriented_wrap_angle * 180.0 / wp.pi,
                )

    link_first = seg_active_link_r[seg_left]
    link_last = seg_active_link_l[seg_right]
    if link_last < link_first:
        tmp_link = link_first
        link_first = link_last
        link_last = tmp_link

    cap_ratio = float(1.0)
    if link_first == link_last:
        cone_link_type = tendon_link_type[link_first]
        cone_link_is_active = tendon_link_active[link_first]
        if cone_link_type == int(TendonLinkType.PINHOLE):
            pin = seg_attachment_r[seg_left]
            u_left = seg_attachment_l[seg_left] - pin
            u_right = seg_attachment_r[seg_right] - pin
            len_ul = wp.length(u_left)
            len_ur = wp.length(u_right)
            theta = 0.0
            if len_ul > 1.0e-8 and len_ur > 1.0e-8:
                incoming = -u_left / len_ul
                outgoing = u_right / len_ur
                theta = wp.atan2(wp.length(wp.cross(incoming, outgoing)), wp.dot(incoming, outgoing))
            cap_ratio = wp.exp(wp.min(wp.max(tendon_link_mu[link_first], 0.0) * theta, 20.0))
        elif cone_link_type == int(TendonLinkType.ROLLING) and cone_link_is_active:
            body = tendon_link_body[link_first]
            pose = body_q[body]
            center = wp.transform_point(pose, tendon_link_offset[link_first])
            normal = wp.transform_vector(pose, tendon_link_axis[link_first])
            radius = tendon_link_radius[link_first]
            r_left = seg_attachment_r[seg_left] - center
            r_right = seg_attachment_l[seg_right] - center
            r_left = r_left - wp.dot(r_left, normal) * normal
            r_right = r_right - wp.dot(r_right, normal) * normal
            len_rl = wp.length(r_left)
            len_rr = wp.length(r_right)
            theta = 0.0
            if radius > 0.0 and len_rl > 1.0e-8 and len_rr > 1.0e-8:
                u_left = r_left / len_rl
                u_right = r_right / len_rr
                theta = wp.abs(wp.atan2(wp.dot(wp.cross(u_left, u_right), normal), wp.dot(u_left, u_right)))
            cap_ratio = wp.exp(wp.min(wp.max(tendon_link_mu[link_first], 0.0) * theta, 20.0))
    else:
        max_mu = float(0.0)
        for link in range(link_first, link_last + 1):
            cone_link_type = tendon_link_type[link]
            cone_link_is_active = tendon_link_active[link]
            if cone_link_type == int(TendonLinkType.PINHOLE):
                max_mu = wp.max(max_mu, tendon_link_mu[link])
            elif cone_link_type == int(TendonLinkType.ROLLING) and cone_link_is_active:
                max_mu = wp.max(max_mu, tendon_link_mu[link])
        cap_ratio = wp.exp(wp.min(wp.max(max_mu, 0.0) * wp.pi, 20.0))

    tendon_link_cone_seg_l[link_idx] = seg_left
    tendon_link_cone_seg_r[link_idx] = seg_right
    tendon_link_cap_ratio[link_idx] = wp.max(cap_ratio, 1.0)


@wp.kernel
def solve_tendon_material(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    tendon_start: wp.array[int],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_radius: wp.array[float],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    seg_rest_length: wp.array[float],
    seg_rest_length_step: wp.array[float],
    seg_route_rest_length: wp.array[float],
    seg_stretch: wp.array[float],
    seg_damping_tension: wp.array[float],
    seg_active: wp.array[int],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
    seg_active_compliance: wp.array[float],
    seg_active_damping: wp.array[float],
    tendon_link_active: wp.array[bool],
    tendon_link_active_step: wp.array[bool],
    tendon_link_route_rest_length: wp.array[float],
    seg_attachment_l: wp.array[wp.vec3],
    seg_attachment_r: wp.array[wp.vec3],
    seg_length: wp.array[float],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_rolling_delta_l: wp.array[float],
    seg_rolling_delta_r: wp.array[float],
    tendon_link_cone_seg_l: wp.array[int],
    tendon_link_cone_seg_r: wp.array[int],
    tendon_link_cap_ratio: wp.array[float],
    tendon_cone_sweep_count: wp.array[int],
    damping_from_pose_delta: int,
    dt: float,
    apply_rolling_transfer: int,
    apply_pinhole_slip: int,
    adaptive_cone_sweeps: int,
    tendon_max_sweeps: int,
    tendon_settle_tol: float,
):
    """Update free-span rest-length transfer for one tendon.

    With adaptive sweeps enabled, the capstan cone is relaxed by up to ``tendon_max_sweeps``
    Gauss-Seidel passes and stops when the tension change relative to the first
    sweep's peak tension falls below ``tendon_settle_tol``. Otherwise, the
    established fixed 4/32-sweep policy is used.

    XPBD derives damping from body velocities. VBD sets ``damping_from_pose_delta``
    to use the current and previous accepted poses, matching its force evaluation.
    """
    tendon_id = wp.tid()
    link_start = tendon_start[tendon_id]
    link_end = tendon_start[tendon_id + 1]
    num_links = link_end - link_start
    num_segs = num_links - 1
    tendon_cone_sweep_count[tendon_id] = 0
    if num_segs < 1:
        return

    # Each preceding tendon contributes one fewer segment than links, so the
    # segment prefix is the link prefix minus the number of preceding tendons.
    seg_offset = link_start - tendon_id

    min_rest = 1.0e-6

    for s in range(num_segs):
        seg = seg_offset + s
        seg_damping_tension[seg] = 0.0
        seg_rest_length[seg] = seg_route_rest_length[seg]

    for i in range(1, num_links - 1):
        link_idx = link_start + i
        if tendon_link_type[link_idx] != int(TendonLinkType.ROLLING):
            continue
        if tendon_link_route_rest_length[link_idx] <= 0.0:
            continue
        if not tendon_link_active[link_idx]:
            continue
        if tendon_link_active_step[link_idx]:
            continue

        seg_left = seg_offset + i - 1
        seg_right = seg_left + 1
        body = tendon_link_body[link_idx]
        pose = body_q[body]
        center = wp.transform_point(pose, tendon_link_offset[link_idx])
        normal = wp.transform_vector(pose, tendon_link_axis[link_idx])
        arc_rest = wrapped_arc_length(
            seg_attachment_r[seg_left], seg_attachment_l[seg_right], center, tendon_link_radius[link_idx], normal
        )
        free_rest = seg_rest_length_step[seg_left] - arc_rest
        if free_rest < 2.0 * min_rest:
            free_rest = 2.0 * min_rest

        len_l = seg_length[seg_left]
        len_r = seg_length[seg_right]
        free_len = wp.max(len_l + len_r, 1.0e-8)
        rest_l = wp.max(min_rest, free_rest * len_l / free_len)
        rest_r = wp.max(min_rest, free_rest - rest_l)
        seg_rest_length[seg_left] = rest_l
        seg_rest_length[seg_right] = rest_r

    if apply_rolling_transfer != 0 or apply_pinhole_slip != 0:
        # Snapshot the per-segment stretch d = len - rest ONCE, at its own (~1e-7) scale. The
        # capstan sweeps below run on this stored d, not on a fresh len-rest each sweep: for a
        # stiff cable d is a tiny difference of ~1e-3 lengths, so recomputing it every sweep
        # loses the (even tinier) friction transfers to float32 cancellation. len is constant
        # during this call (bodies move only in the stretch solve), so the cancellation is paid
        # once here; the sweeps then accumulate transfers precisely. rest is rebuilt at the end.
        for s_snap in range(num_segs):
            seg = seg_offset + s_snap
            if seg_active[seg] != 0:
                len_snap = seg_length[seg]
                # raw (signed) stretch: negative when the span is slack -- preserved so the
                # rebuild below conserves total cable length (rest = len - stretch).
                seg_stretch[seg] = len_snap - seg_rest_length[seg]
                if seg_active_damping[seg] != 0.0:
                    if damping_from_pose_delta != 0:
                        seg_damping_tension[seg] = seg_active_damping[seg] * tendon_segment_length_rate_from_poses(
                            dt,
                            body_q,
                            body_q_prev,
                            body_com,
                            tendon_link_body,
                            tendon_link_type,
                            tendon_link_offset,
                            tendon_link_axis,
                            seg_active_link_l[seg],
                            seg_active_link_r[seg],
                            seg_attachment_l_local[seg],
                            seg_attachment_r_local[seg],
                            seg_attachment_l[seg],
                            seg_attachment_r[seg],
                        )
                    else:
                        seg_damping_tension[seg] = seg_active_damping[seg] * tendon_segment_length_rate(
                            body_q,
                            body_qd,
                            body_com,
                            tendon_link_body,
                            tendon_link_type,
                            tendon_link_offset,
                            tendon_link_axis,
                            seg_active_link_l[seg],
                            seg_active_link_r[seg],
                            seg_attachment_l[seg],
                            seg_attachment_r[seg],
                        )

        material_sweep_count = int(4)
        if adaptive_cone_sweeps != 0:
            material_sweep_count = wp.min(tendon_max_sweeps, 256)
        else:
            # Preserve the established fixed policy for callers that disable adaptive convergence.
            for i_policy in range(1, num_links - 1):
                if tendon_link_type[link_start + i_policy] == int(TendonLinkType.PINHOLE):
                    material_sweep_count = int(32)

        converged = int(0)
        settle_tension_reference = float(0.0)
        for material_sweep in range(256):
            if material_sweep >= material_sweep_count or converged != 0:
                break

            tendon_cone_sweep_count[tendon_id] = tendon_cone_sweep_count[tendon_id] + 1
            sweep_dtension = float(0.0)
            sweep_maxtension = float(0.0)
            for order in range(1, num_links - 1):
                i = order
                if material_sweep % 2 == 1:
                    i = num_links - order - 1

                link_idx = link_start + i
                link_type = tendon_link_type[link_idx]
                link_is_active = tendon_link_active[link_idx]
                is_rolling = link_type == int(TendonLinkType.ROLLING) and link_is_active
                is_pinhole = link_type == int(TendonLinkType.PINHOLE)

                if not ((apply_rolling_transfer != 0 and is_rolling) or (apply_pinhole_slip != 0 and is_pinhole)):
                    continue

                seg_left = tendon_link_cone_seg_l[link_idx]
                seg_right = tendon_link_cone_seg_r[link_idx]

                if seg_left < 0 or seg_right < 0:
                    continue

                if material_sweep == 0 and is_rolling:
                    # The common mode is material exchanged with the changing wrapped arc. Apply
                    # it before relaxation so the capstan projection sees the conserved cable.
                    common_rest_delta = 0.5 * (seg_rolling_delta_r[seg_left] + seg_rolling_delta_l[seg_right])
                    seg_stretch[seg_left] = seg_stretch[seg_left] - common_rest_delta
                    seg_stretch[seg_right] = seg_stretch[seg_right] - common_rest_delta

                cap_ratio = tendon_link_cap_ratio[link_idx]

                len_l = seg_length[seg_left]
                len_r = seg_length[seg_right]
                # Project the same unilateral Kelvin-Voigt tension used by
                # solve_tendon_stretch, including damping at zero stretch.
                d_l_raw = seg_stretch[seg_left]
                d_r_raw = seg_stretch[seg_right]

                # use the true per-segment compliance for the cone: the precise stretch state
                # above removes the conditioning need for the old 1e-8 floor, and clamping here
                # would mismatch the cone (d/clamp) against the real tension (d/comp) and inflate
                # interior tensions across a compliance jump (e.g. c = rest/EA short segments).
                compliance_l = seg_active_compliance[seg_left]
                compliance_r = seg_active_compliance[seg_right]
                comp_l = wp.max(compliance_l, 1.0e-30)
                comp_r = wp.max(compliance_r, 1.0e-30)
                damping_tension_l = seg_damping_tension[seg_left] if compliance_l > 0.0 else 0.0
                damping_tension_r = seg_damping_tension[seg_right] if compliance_r > 0.0 else 0.0
                effective_stretch_l = wp.max(d_l_raw + comp_l * damping_tension_l, 0.0)
                effective_stretch_r = wp.max(d_r_raw + comp_r * damping_tension_r, 0.0)
                force_l = effective_stretch_l / comp_l
                force_r = effective_stretch_r / comp_r
                delta = float(0.0)
                max_delta = float(0.0)

                sweep_maxtension = wp.max(sweep_maxtension, wp.max(force_l, force_r))

                if force_l > force_r * cap_ratio:
                    # rest_right -= delta must keep rest_right >= min_rest.
                    max_delta = wp.max(len_r - d_r_raw - min_rest, 0.0)
                    max_delta = wp.min(max_delta, effective_stretch_l)
                    delta = (comp_r * effective_stretch_l - cap_ratio * comp_l * effective_stretch_r) / (
                        comp_r + cap_ratio * comp_l
                    )
                    delta = wp.max(delta, 0.0)
                    delta = wp.min(delta, max_delta)
                    # rest_left += delta => d_l -= delta ; rest_right -= delta => d_r += delta
                    seg_stretch[seg_left] = d_l_raw - delta
                    seg_stretch[seg_right] = d_r_raw + delta
                    sweep_dtension = wp.max(sweep_dtension, wp.max(delta / comp_l, delta / comp_r))
                elif force_r > force_l * cap_ratio:
                    max_delta = wp.max(len_l - d_l_raw - min_rest, 0.0)
                    max_delta = wp.min(max_delta, effective_stretch_r)
                    delta = (comp_l * effective_stretch_r - cap_ratio * comp_r * effective_stretch_l) / (
                        comp_l + cap_ratio * comp_r
                    )
                    delta = wp.max(delta, 0.0)
                    delta = wp.min(delta, max_delta)
                    seg_stretch[seg_left] = d_l_raw + delta
                    seg_stretch[seg_right] = d_r_raw - delta
                    sweep_dtension = wp.max(sweep_dtension, wp.max(delta / comp_l, delta / comp_r))

            # Keep the normalization scale fixed so an unloading cable can settle as both the
            # tension and its change approach zero. A per-sweep scale would shrink with the
            # residual and keep their ratio finite long after the absolute change is negligible.
            if material_sweep == 0:
                settle_tension_reference = sweep_maxtension
            rel_change = sweep_dtension / wp.max(settle_tension_reference, 1.0e-30)
            if adaptive_cone_sweeps != 0 and rel_change < tendon_settle_tol:
                converged = 1

        # Apply the friction-limited differential transport once after material relaxation so
        # cone sweeps cannot iteratively erode it.
        if apply_rolling_transfer != 0:
            for i_roll in range(1, num_links - 1):
                link_idx = link_start + i_roll
                if tendon_link_type[link_idx] != int(TendonLinkType.ROLLING):
                    continue
                if not tendon_link_active[link_idx]:
                    continue

                seg_left = tendon_link_cone_seg_l[link_idx]
                seg_right = tendon_link_cone_seg_r[link_idx]
                if seg_left < 0 or seg_right < 0:
                    continue

                cap_ratio = tendon_link_cap_ratio[link_idx]
                beta = (cap_ratio - 1.0) / (cap_ratio + 1.0)

                # Apply only the friction-limited differential mode after relaxation. The common
                # mode above is independent of friction and already accounts for wrapped material.
                rolling_delta_diff = 0.5 * (seg_rolling_delta_r[seg_left] - seg_rolling_delta_l[seg_right])
                len_al = seg_length[seg_left]
                len_ar = seg_length[seg_right]
                rest_l = len_al - seg_stretch[seg_left]
                rest_r = len_ar - seg_stretch[seg_right]
                rolling_transfer = rolling_delta_diff * beta
                # Bound the zero-sum transfer as a pair. Clamping either span independently would
                # discard the clipped amount and create cable material at the minimum rest length.
                rolling_transfer = wp.max(rolling_transfer, min_rest - rest_l)
                rolling_transfer = wp.min(rolling_transfer, rest_r - min_rest)
                seg_stretch[seg_left] = seg_stretch[seg_left] - rolling_transfer
                seg_stretch[seg_right] = seg_stretch[seg_right] + rolling_transfer

        # rebuild rest lengths from the telescoped stretch state (one cancellation per call,
        # paid once instead of every sweep -- this is what keeps the capstan accurate for stiff
        # cables where d = len - rest would otherwise vanish into float32 noise).
        for s_wb in range(num_segs):
            seg = seg_offset + s_wb
            if seg_active[seg] != 0:
                len_wb = seg_length[seg]
                seg_rest_length[seg] = wp.max(len_wb - seg_stretch[seg], min_rest)
