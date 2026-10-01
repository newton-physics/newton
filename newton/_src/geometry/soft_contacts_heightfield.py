# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exact heightfield feature contacts for deformable triangle surfaces."""

import warp as wp

from ..utils.heightfield import HeightfieldData, get_triangle_shape_from_heightfield
from .contact_reduction import float_flip
from .flags import ShapeFlags
from .kernels import triangle_closest_point
from .soft_contacts_sdf import _emit_soft_ef_contact, _shape_frames

_HEIGHTFIELD_CELLS_PER_TASK = 256


@wp.func
def _signed_feature_distance(x: wp.vec3, y: wp.vec3, normal: wp.vec3, threshold: float) -> float:
    """Sign nearby feature distances without treating distant back-facing edges as penetration."""
    delta = x - y
    distance = wp.length(delta)
    normal_distance = wp.dot(delta, normal)
    if normal_distance < 0.0:
        # A negative plane distance means penetration only when the two features also overlap
        # tangentially. Without this guard, distant skew edges can appear penetrating merely because
        # their closest-point delta happens to point behind the heightfield triangle's plane.
        tangent = delta - normal_distance * normal
        if wp.length_sq(tangent) >= threshold * threshold:
            return float(1.0e10)
        return -distance
    return distance


@wp.func
def _heightfield_cell_range(a: wp.vec3, b: wp.vec3, c: wp.vec3, hfd: HeightfieldData, threshold: float):
    """Return the conservative inclusive cell rectangle, or an empty rectangle."""
    tri_lower = wp.min(a, wp.min(b, c)) - wp.vec3(threshold)
    tri_upper = wp.max(a, wp.max(b, c)) + wp.vec3(threshold)
    # Terrain is solid below its surface, so only the upper elevation provides a safe Z rejection;
    # deeply penetrating triangles must still reach the exact feature tests.
    if (
        hfd.nrow <= 1
        or hfd.ncol <= 1
        or tri_upper[0] < -hfd.hx
        or tri_lower[0] > hfd.hx
        or tri_upper[1] < -hfd.hy
        or tri_lower[1] > hfd.hy
        or tri_lower[2] > hfd.max_z
    ):
        return wp.vec4i(0, -1, 0, -1)

    dx = 2.0 * hfd.hx / wp.float32(hfd.ncol - 1)
    dy = 2.0 * hfd.hy / wp.float32(hfd.nrow - 1)
    col_begin = wp.max(wp.int32(wp.floor((tri_lower[0] + hfd.hx) / dx)), 0)
    col_end = wp.min(wp.int32(wp.floor((tri_upper[0] + hfd.hx) / dx)), hfd.ncol - 2)
    row_begin = wp.max(wp.int32(wp.floor((tri_lower[1] + hfd.hy) / dy)), 0)
    row_end = wp.min(wp.int32(wp.floor((tri_upper[1] + hfd.hy) / dy)), hfd.nrow - 2)

    return wp.vec4i(col_begin, col_end, row_begin, row_end)


@wp.func
def _closest_heightfield_feature(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    hfd: HeightfieldData,
    elevations: wp.array[wp.float32],
    threshold: float,
    cells: wp.vec4i,
    begin: int,
    end: int,
):
    """Find the missing rigid-vertex/face and rigid-edge/edge contact for one soft triangle."""
    best = float(1.0e10)
    best_bary = wp.vec3(0.0)
    best_y = wp.vec3(0.0)
    best_normal = wp.vec3(0.0, 0.0, 1.0)

    best_cell = int(-1)
    width = cells[1] - cells[0] + 1

    if begin == end:
        return best, best_bary, best_y, best_normal, best_cell
    first_row = cells[2] + begin // width
    last_row = cells[2] + (end - 1) // width
    for row in range(first_row, last_row + 1):
        first_col = cells[0]
        last_col = cells[1]
        if row == first_row:
            first_col = cells[0] + begin % width
        if row == last_row:
            last_col = cells[0] + (end - 1) % width
        for col in range(first_col, last_col + 1):
            cell = (row - cells[2]) * width + col - cells[0]
            for sub in range(2):
                tri_index = (row * (hfd.ncol - 1) + col) * 2 + sub
                rigid_shape, u = get_triangle_shape_from_heightfield(
                    hfd, elevations, wp.transform_identity(), tri_index
                )
                v = u + rigid_shape.scale
                w = u + rigid_shape.auxiliary
                normal = wp.normalize(wp.cross(v - u, w - u))

                # Rigid vertices against the deformable face. The existing particle pass covers
                # the opposite deformable-vertex/rigid-face direction.
                for rigid_vertex in range(3):
                    y = u
                    if rigid_vertex == 1:
                        y = v
                    elif rigid_vertex == 2:
                        y = w
                    x, bary, _feature = triangle_closest_point(a, b, c, y)
                    distance = _signed_feature_distance(x, y, normal, threshold)
                    if distance < best:
                        best = distance
                        best_cell = cell
                        best_bary = bary
                        best_y = y
                        best_normal = normal

                for soft_edge in range(3):
                    p = a
                    q = b
                    if soft_edge == 1:
                        p = b
                        q = c
                    elif soft_edge == 2:
                        p = c
                        q = a
                    for rigid_edge in range(3):
                        r = u
                        s = v
                        if rigid_edge == 1:
                            r = v
                            s = w
                        elif rigid_edge == 2:
                            r = w
                            s = u
                        st = wp.closest_point_edge_edge(p, q, r, s, 1.0e-6)
                        x = p + st[0] * (q - p)
                        y = r + st[1] * (s - r)
                        distance = _signed_feature_distance(x, y, normal, threshold)
                        if distance < best:
                            best = distance
                            best_cell = cell
                            if soft_edge == 0:
                                best_bary = wp.vec3(1.0 - st[0], st[0], 0.0)
                            elif soft_edge == 1:
                                best_bary = wp.vec3(0.0, 1.0 - st[0], st[0])
                            else:
                                best_bary = wp.vec3(st[0], 0.0, 1.0 - st[0])
                            best_y = y
                            best_normal = normal

    return best, best_bary, best_y, best_normal, best_cell


@wp.kernel
def create_soft_heightfield_face_contacts(
    face_pairs: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    tri_indices: wp.array2d[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    shape_heightfield_index: wp.array[wp.int32],
    heightfield_data: wp.array[HeightfieldData],
    heightfield_elevations: wp.array[wp.float32],
    shape_margin: wp.array[float],
    margin: float,
    tid_base: wp.int32,
    task_counts: wp.array[wp.int64],
    winners: wp.array[wp.uint64],
    emit_large: bool,
    cells_per_task: int,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_tids: wp.array[wp.int32],
    soft_contact_particle: wp.array[wp.int32],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):
    """Emit small scans directly, queue large scans, or emit their winning cell on the second pass."""
    tid = wp.tid()
    if emit_large:
        if task_counts[tid] == 0:
            return
    elif task_counts.shape[0] > 0:
        task_counts[tid] = wp.int64(0)
        winners[tid] = wp.uint64(0xFFFFFFFFFFFFFFFF)
    pair = face_pairs[tid]
    tri = pair[0]
    shape = pair[1]
    if (shape_flags[shape] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return

    a_idx = tri_indices[tri, 0]
    b_idx = tri_indices[tri, 1]
    c_idx = tri_indices[tri, 2]
    radius = wp.max(particle_radius[a_idx], wp.max(particle_radius[b_idx], particle_radius[c_idx]))
    shape_contact_margin = shape_margin[shape] if shape_margin.shape[0] > 0 else 0.0
    threshold = margin + shape_contact_margin + radius

    X_bs, X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape)
    a = wp.transform_point(X_sw, particle_q[a_idx])
    b = wp.transform_point(X_sw, particle_q[b_idx])
    c = wp.transform_point(X_sw, particle_q[c_idx])
    hfd = heightfield_data[shape_heightfield_index[shape]]
    cells = _heightfield_cell_range(a, b, c, hfd, threshold)
    begin = int(0)
    end = (cells[1] - cells[0] + 1) * (cells[3] - cells[2] + 1)
    if emit_large:
        if winners[tid] == wp.uint64(0xFFFFFFFFFFFFFFFF):
            return
        begin = int(winners[tid] & wp.uint64(0xFFFFFFFF))
        end = begin + 1
    elif task_counts.shape[0] > 0 and end > cells_per_task:
        task_counts[tid] = wp.int64((end + cells_per_task - 1) // cells_per_task)
        return
    distance, bary, y, normal, _cell = _closest_heightfield_feature(
        a, b, c, hfd, heightfield_elevations, threshold, cells, begin, end
    )
    if distance < threshold:
        _emit_soft_ef_contact(
            tid,
            tid_base,
            soft_contact_max,
            soft_contact_count,
            soft_contact_tids,
            soft_contact_particle,
            soft_contact_indices,
            soft_contact_barycentric,
            soft_contact_shape,
            soft_contact_body_pos,
            soft_contact_body_vel,
            soft_contact_normal,
            wp.vec3i(a_idx, b_idx, c_idx),
            bary,
            shape,
            wp.transform_point(X_bs, y),
            wp.vec3(0.0),
            wp.transform_vector(X_ws, normal),
        )


@wp.kernel
def find_heightfield_task_contacts(
    face_pairs: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    tri_indices: wp.array2d[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    shape_heightfield_index: wp.array[wp.int32],
    heightfield_data: wp.array[HeightfieldData],
    heightfield_elevations: wp.array[wp.float32],
    shape_margin: wp.array[float],
    margin: float,
    offsets: wp.array[wp.int64],
    winners: wp.array[wp.uint64],
    cells_per_task: int,
    grid_size: int,
):
    """Reduce bounded cell batches to one deterministic winning cell per deformable face pair."""
    # Compact ragged tasks avoid allocating face_count * terrain_cell_count scratch.
    pair_count = face_pairs.shape[0]
    task = wp.int64(wp.tid())
    while task < offsets[pair_count]:
        lower = int(0)
        upper = pair_count
        while lower < upper:
            middle = (lower + upper) // 2
            if offsets[middle + 1] <= task:
                lower = middle + 1
            else:
                upper = middle
        tid = lower
        pair = face_pairs[tid]
        tri = pair[0]
        shape = pair[1]
        if (shape_flags[shape] & ShapeFlags.COLLIDE_PARTICLES) == 0:
            task += wp.int64(grid_size)
            continue

        a_idx = tri_indices[tri, 0]
        b_idx = tri_indices[tri, 1]
        c_idx = tri_indices[tri, 2]
        radius = wp.max(particle_radius[a_idx], wp.max(particle_radius[b_idx], particle_radius[c_idx]))
        shape_contact_margin = shape_margin[shape] if shape_margin.shape[0] > 0 else 0.0
        threshold = margin + shape_contact_margin + radius

        _X_bs, _X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape)
        a = wp.transform_point(X_sw, particle_q[a_idx])
        b = wp.transform_point(X_sw, particle_q[b_idx])
        c = wp.transform_point(X_sw, particle_q[c_idx])
        hfd = heightfield_data[shape_heightfield_index[shape]]
        cells = _heightfield_cell_range(a, b, c, hfd, threshold)
        cell_count = (cells[1] - cells[0] + 1) * (cells[3] - cells[2] + 1)
        begin = int(task - offsets[tid]) * cells_per_task
        end = wp.min(begin + cells_per_task, cell_count)
        distance, _bary, _y, _normal, cell = _closest_heightfield_feature(
            a, b, c, hfd, heightfield_elevations, threshold, cells, begin, end
        )
        if distance < threshold:
            # Distance first, then the original row-major cell order for deterministic ties.
            key = (wp.uint64(float_flip(distance)) << wp.uint64(32)) | wp.uint64(cell)
            wp.atomic_min(winners, tid, key)
        task += wp.int64(grid_size)


def launch_soft_heightfield_contacts(*, model, state, contacts, margin: float, device, face_pairs, tid_base: int):
    """Launch exact heightfield feature tests separately from the common SDF face kernel."""
    if len(face_pairs) == 0:
        return
    workspace = getattr(contacts, "_soft_heightfield_work", None)
    task_counts, offsets, winners = (None, None, None) if workspace is None else workspace
    cells_per_task = _HEIGHTFIELD_CELLS_PER_TASK
    inputs = [
        face_pairs,
        state.particle_q,
        model.particle_radius,
        model.tri_indices,
        model.shape_body,
        model.shape_flags,
        model.shape_transform,
        state.body_q,
        model.shape_heightfield_index,
        model.heightfield_data,
        model.heightfield_elevations,
        model.shape_margin,
        margin,
    ]
    outputs = [
        contacts.soft_contact_count,
        contacts.soft_contact_tids,
        contacts.soft_contact_particle,
        contacts.soft_contact_indices,
        contacts.soft_contact_barycentric,
        contacts.soft_contact_shape,
        contacts.soft_contact_body_pos,
        contacts.soft_contact_body_vel,
        contacts.soft_contact_normal,
    ]
    wp.launch(
        create_soft_heightfield_face_contacts,
        dim=len(face_pairs),
        inputs=[*inputs, tid_base, task_counts, winners, False, cells_per_task, contacts.soft_contact_max],
        outputs=outputs,
        device=device,
    )
    if workspace is None:
        return
    wp.utils.array_scan(task_counts, offsets, inclusive=False)
    # A persistent grid consumes every task without a host readback, including in CUDA graphs.
    grid_size = 4096
    wp.launch(
        find_heightfield_task_contacts,
        dim=grid_size,
        inputs=[*inputs, offsets, winners, cells_per_task, grid_size],
        device=device,
    )
    wp.launch(
        create_soft_heightfield_face_contacts,
        dim=len(face_pairs),
        inputs=[*inputs, tid_base, task_counts, winners, True, cells_per_task, contacts.soft_contact_max],
        outputs=outputs,
        device=device,
    )
