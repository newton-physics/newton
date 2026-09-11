# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""BVH-culled deformable face contacts for rigid triangle meshes."""

import warp as wp

from .flags import ShapeFlags
from .kernels import triangle_closest_point
from .sdf_texture import TextureSDFData
from .soft_contacts_sdf import (
    _emit_soft_ef_contact,
    _eval_shape_sdf_lower,
    _shape_frames,
    create_soft_face_sdf_contacts,
)

# Dense overlaps favor the fixed-cost SDF optimizer; bounding candidates also caps local storage
# and traversal work before that fallback.
_MESH_EXACT_MAX_TRIANGLES = wp.constant(6)
_mesh_triangle_candidates_t = wp.types.vector(length=6, dtype=wp.int32)
# Cap the compact fallback launch while leaving enough one-warp blocks to saturate large GPUs.
_MESH_SDF_FALLBACK_GRID_SIZE = 131072

_MESH_FACE_CLASSIFY_BLOCK_DIM = 64


@wp.func
def _closest_triangle_features(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    u: wp.vec3,
    v: wp.vec3,
    w: wp.vec3,
    best_distance_sq: float,
    best_bary: wp.vec3,
    best_y: wp.vec3,
):
    """Update the closest points between a deformable and rigid triangle."""
    for rigid_vertex in range(3):
        y = u
        if rigid_vertex == 1:
            y = v
        elif rigid_vertex == 2:
            y = w
        x, bary, _feature = triangle_closest_point(a, b, c, y)
        distance_sq = wp.length_sq(x - y)
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_bary = bary
            best_y = y

    for soft_vertex in range(3):
        x = a
        bary = wp.vec3(1.0, 0.0, 0.0)
        if soft_vertex == 1:
            x = b
            bary = wp.vec3(0.0, 1.0, 0.0)
        elif soft_vertex == 2:
            x = c
            bary = wp.vec3(0.0, 0.0, 1.0)
        y, _rigid_bary, _feature = triangle_closest_point(u, v, w, x)
        distance_sq = wp.length_sq(x - y)
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_bary = bary
            best_y = y

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
            distance_sq = wp.length_sq(x - y)
            if distance_sq < best_distance_sq:
                best_distance_sq = distance_sq
                if soft_edge == 0:
                    best_bary = wp.vec3(1.0 - st[0], st[0], 0.0)
                elif soft_edge == 1:
                    best_bary = wp.vec3(0.0, 1.0 - st[0], st[0])
                else:
                    best_bary = wp.vec3(st[0], 0.0, 1.0 - st[0])
                best_y = y

    return best_distance_sq, best_bary, best_y


@wp.kernel
def create_soft_mesh_face_contacts(
    face_pairs: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    tri_indices: wp.array2d[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    shape_source: wp.array[wp.uint64],
    shape_sdf_index: wp.array[wp.int32],
    texture_sdf_table: wp.array[TextureSDFData],
    shape_margin: wp.array[float],
    shape_aabb_lower: wp.array[wp.vec3],
    shape_aabb_upper: wp.array[wp.vec3],
    margin: float,
    tid_base: wp.int32,
    soft_contact_max: wp.int32,
    fallback_tids: wp.array[wp.int32],
    fallback_count: wp.array[wp.int32],
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
    """Use exact sparse mesh features, with SDF fallback for dense or penetrating pairs."""
    tid = wp.tid()
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

    # Most Cartesian face/shape pairs are nowhere near the rigid mesh. Reject them against its
    # scaled local AABB without paying for a texture lookup or BVH traversal.
    tri_lower = wp.min(a, wp.min(b, c)) - wp.vec3(threshold)
    tri_upper = wp.max(a, wp.max(b, c)) + wp.vec3(threshold)
    mesh_lower = shape_aabb_lower[shape]
    mesh_upper = shape_aabb_upper[shape]
    if (
        tri_upper[0] < mesh_lower[0]
        or tri_upper[1] < mesh_lower[1]
        or tri_upper[2] < mesh_lower[2]
        or tri_lower[0] > mesh_upper[0]
        or tri_lower[1] > mesh_upper[1]
        or tri_lower[2] > mesh_upper[2]
    ):
        return

    scale = shape_scale[shape]
    geo = shape_type[shape]
    inv_scale = wp.cw_div(wp.vec3(1.0), scale)
    a_mesh = wp.cw_mul(a, inv_scale)
    b_mesh = wp.cw_mul(b, inv_scale)
    c_mesh = wp.cw_mul(c, inv_scale)
    expansion = threshold * wp.abs(inv_scale)
    query = wp.mesh_query_aabb(
        shape_source[shape],
        wp.min(a_mesh, wp.min(b_mesh, c_mesh)) - expansion,
        wp.max(a_mesh, wp.max(b_mesh, c_mesh)) + expansion,
    )
    mesh_tri = wp.int32(0)
    surface_near = bool(False)
    exact_complete = bool(True)
    exact_triangle_count = wp.int32(0)
    exact_triangles = _mesh_triangle_candidates_t(0)
    while wp.mesh_query_aabb_next(query, mesh_tri):
        surface_near = True
        if exact_triangle_count >= _MESH_EXACT_MAX_TRIANGLES:
            exact_complete = False
            break
        exact_triangles[exact_triangle_count] = mesh_tri
        exact_triangle_count += 1

    best_distance_sq = float(1.0e20)
    best_bary = wp.vec3(0.0)
    best_y = wp.vec3(0.0)
    mesh_id = shape_source[shape]
    if exact_complete:
        for candidate_index in range(exact_triangle_count):
            mesh_tri = exact_triangles[candidate_index]
            u = wp.cw_mul(scale, wp.mesh_get_point(mesh_id, mesh_tri * 3 + 0))
            v = wp.cw_mul(scale, wp.mesh_get_point(mesh_id, mesh_tri * 3 + 1))
            w = wp.cw_mul(scale, wp.mesh_get_point(mesh_id, mesh_tri * 3 + 2))
            best_distance_sq, best_bary, best_y = _closest_triangle_features(
                a,
                b,
                c,
                u,
                v,
                w,
                best_distance_sq,
                best_bary,
                best_y,
            )

    sdf_idx = shape_sdf_index[shape]
    if not surface_near:
        # A face wholly inside a closed mesh need not overlap its surface BVH. Preserve the previous
        # penetrating-face contact in that case; any face entering from outside crosses the surface
        # and therefore has a BVH hit above.
        centroid = (a + b + c) / 3.0
        phi_centroid = _eval_shape_sdf_lower(geo, scale, centroid, sdf_idx, texture_sdf_table)
        if phi_centroid >= 0.0:
            return

    use_exact_surface = False
    bary = wp.vec3(0.0)
    x = wp.vec3(0.0)
    phi = float(0.0)
    grad = wp.vec3(0.0)
    if surface_near and exact_complete:
        distance = wp.sqrt(best_distance_sq)
        closest_x = best_bary[0] * a + best_bary[1] * b + best_bary[2] * c
        phi_probe = _eval_shape_sdf_lower(geo, scale, closest_x, sdf_idx, texture_sdf_table)
        if distance > 1.0e-6 and phi_probe >= 0.0:
            if distance >= threshold:
                return
            use_exact_surface = True
            bary = best_bary
            x = closest_x
            phi = distance
            grad = (closest_x - best_y) / distance

    if not use_exact_surface:
        fallback_slot = wp.atomic_add(fallback_count, 0, 1)
        fallback_tids[fallback_slot] = tid
        return
    if phi < threshold:
        y = x - phi * grad
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
            wp.transform_vector(X_ws, grad),
        )


def launch_soft_mesh_face_contacts(
    *,
    model,
    state,
    contacts,
    margin: float,
    device,
    face_pairs,
    fallback_tids,
    fallback_count,
    tid_base: int,
):
    """Launch BVH-guided deformable face contacts for rigid mesh shapes."""
    if len(face_pairs) == 0:
        return
    fallback_count.zero_()
    wp.launch(
        create_soft_mesh_face_contacts,
        dim=len(face_pairs),
        inputs=[
            face_pairs,
            state.particle_q,
            model.particle_radius,
            model.tri_indices,
            model.shape_body,
            model.shape_type,
            model.shape_flags,
            model.shape_transform,
            model.shape_scale,
            state.body_q,
            model.shape_source_ptr,
            model._shape_sdf_index,
            model._texture_sdf_data,
            model.shape_margin,
            model.shape_collision_aabb_lower,
            model.shape_collision_aabb_upper,
            margin,
            tid_base,
            contacts.soft_contact_max,
        ],
        outputs=[
            fallback_tids,
            fallback_count,
            contacts.soft_contact_count,
            contacts.soft_contact_tids,
            contacts.soft_contact_particle,
            contacts.soft_contact_indices,
            contacts.soft_contact_barycentric,
            contacts.soft_contact_shape,
            contacts.soft_contact_body_pos,
            contacts.soft_contact_body_vel,
            contacts.soft_contact_normal,
        ],
        device=device,
        # Two warps balance BVH traversal throughput and register occupancy.
        block_dim=_MESH_FACE_CLASSIFY_BLOCK_DIM,
    )
    fallback_grid_size = min(
        len(face_pairs), _MESH_SDF_FALLBACK_GRID_SIZE if wp.get_device(device).is_cuda else len(face_pairs)
    )
    wp.launch(
        create_soft_face_sdf_contacts,
        dim=fallback_grid_size,
        inputs=[
            fallback_tids,
            fallback_count,
            fallback_grid_size,
            face_pairs,
            state.particle_q,
            model.particle_radius,
            model.tri_indices,
            model.shape_body,
            model.shape_type,
            model.shape_transform,
            model.shape_scale,
            state.body_q,
            model._shape_sdf_index,
            model._texture_sdf_data,
            model.shape_margin,
            margin,
            tid_base,
            contacts.soft_contact_max,
        ],
        outputs=[
            contacts.soft_contact_count,
            contacts.soft_contact_tids,
            contacts.soft_contact_particle,
            contacts.soft_contact_indices,
            contacts.soft_contact_barycentric,
            contacts.soft_contact_shape,
            contacts.soft_contact_body_pos,
            contacts.soft_contact_body_vel,
            contacts.soft_contact_normal,
        ],
        device=device,
        block_dim=32,
    )
