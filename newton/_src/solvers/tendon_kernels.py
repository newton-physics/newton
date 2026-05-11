# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver-neutral routed tendon geometry kernels."""

import warp as wp

from ..sim.tendon import TendonLinkType


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
            fallback = wp.vec3(1.0, 0.0, 0.0) - wp.dot(wp.vec3(1.0, 0.0, 0.0), plane_normal) * plane_normal
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
def advance_point_on_circle(
    old_pt: wp.vec3,
    center: wp.vec3,
    radius: float,
    plane_normal: wp.vec3,
    orientation: int,
    signed_arc: float,
) -> wp.vec3:
    """Move old_pt along the rolling surface by a signed arc length."""
    r_old = old_pt - center
    r_old = r_old - wp.dot(r_old, plane_normal) * plane_normal
    len_old = wp.length(r_old)
    if len_old < 1.0e-8 or radius <= 0.0:
        return old_pt

    u_old = r_old / len_old
    angle = -signed_arc / (radius * float(orientation))
    tangent = wp.cross(plane_normal, u_old)
    return center + radius * (wp.cos(angle) * u_old + wp.sin(angle) * tangent)


@wp.kernel
def update_tendon_attachments(
    body_q: wp.array[wp.transform],
    tendon_start: wp.array[int],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_radius: wp.array[float],
    tendon_link_orientation: wp.array[int],
    tendon_link_mu: wp.array[float],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    seg_rest_length: wp.array[float],
    seg_compliance: wp.array[float],
    seg_attachment_l: wp.array[wp.vec3],
    seg_attachment_r: wp.array[wp.vec3],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_rolling_delta_l: wp.array[float],
    seg_rolling_delta_r: wp.array[float],
    apply_rolling_transfer: int,
    apply_pinhole_slip: int,
):
    """Update routed tendon tangent points and free-span rest-length transfer."""
    tendon_id = wp.tid()
    link_start = tendon_start[tendon_id]
    link_end = tendon_start[tendon_id + 1]
    num_links = link_end - link_start
    num_segs = num_links - 1
    if num_segs < 1:
        return

    seg_offset = int(0)
    for t in range(tendon_id):
        seg_offset = seg_offset + (tendon_start[t + 1] - tendon_start[t] - 1)

    for s in range(num_segs):
        seg = seg_offset + s
        link_l = link_start + s
        link_r = link_l + 1
        seg_rolling_delta_l[seg] = 0.0
        seg_rolling_delta_r[seg] = 0.0

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

        old_al = wp.transform_point(pose_l, seg_attachment_l_local[seg])
        old_ar = wp.transform_point(pose_r, seg_attachment_r_local[seg])

        new_al = center_l
        new_ar = center_r
        both_rolling = (type_l == int(TendonLinkType.ROLLING)) and (type_r == int(TendonLinkType.ROLLING))

        if both_rolling and radius_l > 0.0 and radius_r > 0.0:
            new_al = old_al
            new_ar = old_ar
            for _iter in range(10):
                new_ar = tangent_point_circle(new_al, center_r, radius_r, normal_r, orient_r)
                new_al = tangent_point_circle(new_ar, center_l, radius_l, normal_l, -orient_l)
        elif type_l == int(TendonLinkType.ROLLING) and radius_l > 0.0:
            new_ar = center_r
            new_al = tangent_point_circle(center_r, center_l, radius_l, normal_l, -orient_l)
        elif type_r == int(TendonLinkType.ROLLING) and radius_r > 0.0:
            new_al = center_l
            new_ar = tangent_point_circle(center_l, center_r, radius_r, normal_r, orient_r)

        if apply_rolling_transfer != 0:
            if type_l == int(TendonLinkType.ROLLING) and radius_l > 0.0:
                delta_l = signed_arc_length(old_al, new_al, center_l, radius_l, normal_l, orient_l)
                seg_rolling_delta_l[seg] = delta_l

            if type_r == int(TendonLinkType.ROLLING) and radius_r > 0.0:
                delta_r = signed_arc_length(old_ar, new_ar, center_r, radius_r, normal_r, orient_r)
                seg_rolling_delta_r[seg] = -delta_r

        seg_attachment_l[seg] = new_al
        seg_attachment_r[seg] = new_ar
        seg_attachment_l_local[seg] = wp.transform_point(wp.transform_inverse(pose_l), new_al)
        seg_attachment_r_local[seg] = wp.transform_point(wp.transform_inverse(pose_r), new_ar)

    if apply_rolling_transfer != 0:
        min_rest = 1.0e-6
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

            rest_l = seg_rest_length[seg_left] + seg_rolling_delta_r[seg_left] * beta
            rest_r = seg_rest_length[seg_right] + seg_rolling_delta_l[seg_right] * beta
            if rest_l < min_rest:
                rest_l = min_rest
            if rest_r < min_rest:
                rest_r = min_rest
            seg_rest_length[seg_left] = rest_l
            seg_rest_length[seg_right] = rest_r

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
            delta = float(0.0)
            max_delta = float(0.0)

            if force_l > force_r * cap_ratio:
                delta = (comp_r * d_l - cap_ratio * comp_l * d_r) / (comp_r + cap_ratio * comp_l)
                if delta < 0.0:
                    delta = 0.0
                max_delta = seg_rest_length[seg_right] - min_rest
                if max_delta < 0.0:
                    max_delta = 0.0
                if delta > max_delta:
                    delta = max_delta
                seg_rest_length[seg_left] = seg_rest_length[seg_left] + delta
                seg_rest_length[seg_right] = seg_rest_length[seg_right] - delta
            elif force_r > force_l * cap_ratio:
                delta = (comp_l * d_r - cap_ratio * comp_r * d_l) / (comp_l + cap_ratio * comp_r)
                if delta < 0.0:
                    delta = 0.0
                max_delta = seg_rest_length[seg_left] - min_rest
                if max_delta < 0.0:
                    max_delta = 0.0
                if delta > max_delta:
                    delta = max_delta
                seg_rest_length[seg_left] = seg_rest_length[seg_left] - delta
                seg_rest_length[seg_right] = seg_rest_length[seg_right] + delta

    if apply_pinhole_slip == 0:
        return

    min_rest = 1.0e-6
    for i in range(1, num_links - 1):
        link_idx = link_start + i
        if tendon_link_type[link_idx] != int(TendonLinkType.PINHOLE):
            continue

        seg_left = seg_offset + i - 1
        seg_right = seg_offset + i
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

        pin = seg_attachment_r[seg_left]
        u_left = seg_attachment_l[seg_left] - pin
        u_right = seg_attachment_r[seg_right] - pin
        len_ul = wp.length(u_left)
        len_ur = wp.length(u_right)
        theta = 0.0
        if len_ul > 1.0e-8 and len_ur > 1.0e-8:
            # Bend angle between incoming cable direction and outgoing direction.
            incoming = -u_left / len_ul
            outgoing = u_right / len_ur
            theta = wp.atan2(wp.length(wp.cross(incoming, outgoing)), wp.dot(incoming, outgoing))

        cap_ratio = wp.exp(wp.min(wp.max(tendon_link_mu[link_idx], 0.0) * theta, 20.0))
        delta = float(0.0)
        max_delta = float(0.0)

        if force_l > force_r * cap_ratio:
            delta = (comp_r * d_l - cap_ratio * comp_l * d_r) / (comp_r + cap_ratio * comp_l)
            if delta < 0.0:
                delta = 0.0
            max_delta = seg_rest_length[seg_right] - min_rest
            if max_delta < 0.0:
                max_delta = 0.0
            if delta > max_delta:
                delta = max_delta
            seg_rest_length[seg_left] = seg_rest_length[seg_left] + delta
            seg_rest_length[seg_right] = seg_rest_length[seg_right] - delta
        elif force_r > force_l * cap_ratio:
            delta = (comp_l * d_r - cap_ratio * comp_r * d_l) / (comp_l + cap_ratio * comp_r)
            if delta < 0.0:
                delta = 0.0
            max_delta = seg_rest_length[seg_left] - min_rest
            if max_delta < 0.0:
                max_delta = 0.0
            if delta > max_delta:
                delta = max_delta
            seg_rest_length[seg_left] = seg_rest_length[seg_left] - delta
            seg_rest_length[seg_right] = seg_rest_length[seg_right] + delta
