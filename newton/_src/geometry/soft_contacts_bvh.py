# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Full-surface rigid-soft contacts for mesh-backed rigid shapes.

The detector enumerates vertex-face, face-vertex, and edge-edge primitive pairs
with rigid ``wp.Mesh`` and soft grouped BVHs. Detection writes compact integer
candidates without tape recording; a fixed-size second pass reconstructs and
emits differentiable contact geometry, giving every emission thread one replay
slot even when one BVH traversal discovers several pairs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .flags import ParticleFlags, ShapeFlags
from .kernels import counter_increment, triangle_closest_point

if TYPE_CHECKING:
    from ..sim import Contacts, Model, State
    from .tri_mesh_collision import TriMeshCollisionDetector


RIGID_SOFT_BVH_CONTACT_VF = wp.constant(0)
RIGID_SOFT_BVH_CONTACT_FV = wp.constant(1)
RIGID_SOFT_BVH_CONTACT_EE = wp.constant(2)
_NORMAL_EPS = wp.constant(1.0e-12)


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    """Normalize nonzero rows, leaving degenerate rows at zero."""
    result = np.zeros_like(values, dtype=np.float64)
    lengths = np.linalg.norm(values, axis=1)
    valid = lengths > 1.0e-12
    result[valid] = values[valid] / lengths[valid, None]
    return result


def _scaled_mesh_feature_normals(mesh, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return angle-weighted vertex and adjacent-face edge normals after shape scaling.

    Face orientation is transported with the inverse transpose rather than taken
    directly from the scaled winding. This preserves outward orientation under an
    odd number of mirrored scale axes.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)
    edges = np.asarray(mesh.edges, dtype=np.int32).reshape(-1, 2)
    scaled = vertices * scale[None, :]

    face_cross = np.cross(
        scaled[triangles[:, 1]] - scaled[triangles[:, 0]],
        scaled[triangles[:, 2]] - scaled[triangles[:, 0]],
    )
    # A reflected affine map reverses winding, while an outward normal transforms
    # by A^-T. Correct the cross-product orientation by sign(det(A)).
    det_sign = -1.0 if float(np.prod(scale)) < 0.0 else 1.0
    face_normals = _normalized_rows(face_cross * det_sign)

    vertex_accum = np.zeros_like(scaled)
    for face_index, tri in enumerate(triangles):
        normal = face_normals[face_index]
        if not np.any(normal):
            continue
        for corner in range(3):
            vertex = int(tri[corner])
            p = scaled[vertex]
            e0 = scaled[int(tri[(corner + 1) % 3])] - p
            e1 = scaled[int(tri[(corner + 2) % 3])] - p
            l0 = np.linalg.norm(e0)
            l1 = np.linalg.norm(e1)
            if l0 <= 1.0e-12 or l1 <= 1.0e-12:
                continue
            cosine = float(np.clip(np.dot(e0, e1) / (l0 * l1), -1.0, 1.0))
            vertex_accum[vertex] += np.arccos(cosine) * normal
    vertex_normals = _normalized_rows(vertex_accum)

    adjacent: dict[tuple[int, int], list[np.ndarray]] = {}
    for face_index, tri in enumerate(triangles):
        normal = face_normals[face_index]
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (min(int(a), int(b)), max(int(a), int(b)))
            adjacent.setdefault(key, []).append(normal)

    edge_normals = np.zeros((len(edges), 3), dtype=np.float64)
    for edge_index, (a, b) in enumerate(edges):
        normals = adjacent.get((min(int(a), int(b)), max(int(a), int(b))), ())
        if normals:
            summed = np.sum(normals, axis=0)
            length = np.linalg.norm(summed)
            if length <= 1.0e-12:
                summed = np.asarray(normals[0])
                length = np.linalg.norm(summed)
            if length > 1.0e-12:
                edge_normals[edge_index] = summed / length
    return vertex_normals.astype(np.float32), edge_normals.astype(np.float32)


def build_rigid_soft_bvh_rigid_feature_tables(model: Model, shape_mask: np.ndarray):
    """Build per-instance full rigid vertex/edge tables for selected mesh shapes."""
    shape_scales = model.shape_scale.numpy()
    vertex_rows: list[tuple[int, int]] = []
    vertex_positions: list[np.ndarray] = []
    vertex_normals: list[np.ndarray] = []
    edge_rows: list[tuple[int, int, int]] = []
    edge_vertex_rows: list[tuple[int, int]] = []
    edge_normals: list[np.ndarray] = []

    # Cache topology only. Normals remain per instance because non-uniform and
    # mirrored shape scale changes angles and outward directions.
    topology_cache: dict[int, tuple[int, np.ndarray]] = {}
    for shape_index in np.flatnonzero(shape_mask):
        mesh = model.shape_source[int(shape_index)]
        if mesh is None:
            continue
        cache_key = id(mesh)
        cached = topology_cache.get(cache_key)
        if cached is None:
            cached = (len(mesh.vertices), np.asarray(mesh.edges, dtype=np.int32).reshape(-1, 2))
            topology_cache[cache_key] = cached
        vertex_count, edges = cached
        v_normals, e_normals = _scaled_mesh_feature_normals(mesh, np.asarray(shape_scales[shape_index]))
        vertex_offset = len(vertex_rows)
        vertex_rows.extend((int(shape_index), vertex_index) for vertex_index in range(vertex_count))
        vertex_positions.extend(np.asarray(mesh.vertices, dtype=np.float32))
        vertex_normals.extend(v_normals)
        edge_rows.extend((int(shape_index), int(edge[0]), int(edge[1])) for edge in edges)
        edge_vertex_rows.extend((vertex_offset + int(edge[0]), vertex_offset + int(edge[1])) for edge in edges)
        edge_normals.extend(e_normals)

    device = model.device
    vertex_np = np.asarray(vertex_rows, dtype=np.int32).reshape(-1, 2)
    vertex_position_np = np.asarray(vertex_positions, dtype=np.float32).reshape(-1, 3)
    edge_np = np.asarray(edge_rows, dtype=np.int32).reshape(-1, 3)
    edge_vertex_np = np.asarray(edge_vertex_rows, dtype=np.int32).reshape(-1, 2)
    vertex_normal_np = np.asarray(vertex_normals, dtype=np.float32).reshape(-1, 3)
    edge_normal_np = np.asarray(edge_normals, dtype=np.float32).reshape(-1, 3)
    return (
        wp.array(vertex_np, dtype=wp.vec2i, device=device),
        wp.array(vertex_position_np, dtype=wp.vec3, device=device),
        wp.array(vertex_normal_np, dtype=wp.vec3, device=device),
        wp.array(edge_np, dtype=wp.vec3i, device=device),
        wp.array(edge_vertex_np, dtype=wp.vec2i, device=device),
        wp.array(edge_normal_np, dtype=wp.vec3, device=device),
    )


@wp.func
def _append_candidate(
    family: wp.int32,
    soft_feature: wp.int32,
    shape_index: wp.int32,
    rigid_feature: wp.int32,
    candidate_max: wp.int32,
    candidate_count: wp.array[wp.int32],
    candidates: wp.array[wp.vec4i],
):
    index = wp.atomic_add(candidate_count, 0, 1)
    if index < candidate_max:
        candidates[index] = wp.vec4i(family, soft_feature, shape_index, rigid_feature)


@wp.func
def _shape_frames(
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_index: wp.int32,
):
    body_index = shape_body[shape_index]
    X_wb = wp.transform_identity()
    if body_index >= 0:
        X_wb = body_q[body_index]
    X_bs = shape_transform[shape_index]
    X_ws = wp.transform_multiply(X_wb, X_bs)
    return X_bs, X_ws, wp.transform_inverse(X_ws)


@wp.func
def _mesh_face_points(mesh: wp.uint64, face: wp.int32, scale: wp.vec3):
    p0 = wp.cw_mul(wp.mesh_eval_position(mesh, face, 1.0, 0.0), scale)
    p1 = wp.cw_mul(wp.mesh_eval_position(mesh, face, 0.0, 1.0), scale)
    p2 = wp.cw_mul(wp.mesh_eval_position(mesh, face, 0.0, 0.0), scale)
    return p0, p1, p2


@wp.func
def _face_outward(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, scale: wp.vec3):
    normal = wp.cross(p1 - p0, p2 - p0)
    if scale[0] * scale[1] * scale[2] < 0.0:
        normal = -normal
    length = wp.length(normal)
    if length > _NORMAL_EPS:
        return normal / length
    return wp.vec3(0.0)


@wp.func
def _rigid_to_soft_normal(delta: wp.vec3, outward: wp.vec3):
    distance = wp.length(delta)
    outward_length = wp.length(outward)
    if outward_length > _NORMAL_EPS:
        outward = outward / outward_length
    if distance > _NORMAL_EPS:
        normal = delta / distance
        if outward_length > _NORMAL_EPS and wp.dot(normal, outward) < 0.0:
            normal = -normal
        return normal
    if outward_length > _NORMAL_EPS:
        return outward
    return wp.vec3(0.0, 0.0, 1.0)


@wp.kernel
def detect_rigid_soft_bvh_vf_candidates(
    particle_shape_pairs: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_margin: wp.array[float],
    shape_flags: wp.array[wp.int32],
    gap: float,
    candidate_max: wp.int32,
    candidate_count: wp.array[wp.int32],
    candidates: wp.array[wp.vec4i],
):
    tid = wp.tid()
    pair = particle_shape_pairs[tid]
    particle = pair[0]
    shape = pair[1]
    if (particle_flags[particle] & ParticleFlags.ACTIVE) == 0:
        return
    if (shape_flags[shape] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return

    # X_sw maps world coordinates into the shape-local frame. Shape scale is stored separately,
    # so dividing that result by scale below produces the unscaled mesh-local coordinates used by
    # the Warp mesh BVH.
    _X_bs, _X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape)
    x_scaled = wp.transform_point(X_sw, particle_q[particle])
    scale = shape_scale[shape]
    min_scale = wp.min(wp.abs(scale))
    if min_scale <= _NORMAL_EPS:
        return
    threshold = gap + shape_margin[shape] + particle_radius[particle]
    query_point = wp.cw_div(x_scaled, scale)
    query_radius = threshold / min_scale
    lower = query_point - wp.vec3(query_radius)
    upper = query_point + wp.vec3(query_radius)
    mesh = shape_source_ptr[shape]
    query = wp.mesh_query_aabb(mesh, lower, upper)
    face = wp.int32(0)
    while wp.mesh_query_aabb_next(query, face):
        p0, p1, p2 = _mesh_face_points(mesh, face, scale)
        closest, _bary, _feature = triangle_closest_point(p0, p1, p2, x_scaled)
        if wp.length(x_scaled - closest) < threshold:
            _append_candidate(
                RIGID_SOFT_BVH_CONTACT_VF, particle, shape, face, candidate_max, candidate_count, candidates
            )


@wp.kernel
def detect_rigid_soft_bvh_fv_candidates(
    rigid_vertex_table: wp.array[wp.vec2i],
    rigid_vertex_position: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    bvh_id: wp.uint64,
    bvh_group_roots: wp.array[wp.int32],
    world_count: wp.int32,
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_margin: wp.array[float],
    shape_flags: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    particle_max_radius: float,
    gap: float,
    candidate_max: wp.int32,
    candidate_count: wp.array[wp.int32],
    candidates: wp.array[wp.vec4i],
):
    rigid_row = wp.tid()
    entry = rigid_vertex_table[rigid_row]
    shape = entry[0]
    if (shape_flags[shape] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return
    _X_bs, X_ws, _X_sw = _shape_frames(shape_body, body_q, shape_transform, shape)
    rigid_point = wp.transform_point(X_ws, wp.cw_mul(rigid_vertex_position[rigid_row], shape_scale[shape]))
    broad_radius = gap + shape_margin[shape] + particle_max_radius
    lower = rigid_point - wp.vec3(broad_radius)
    upper = rigid_point + wp.vec3(broad_radius)
    rigid_world = shape_world[shape]

    # The soft-feature BVH is one grouped tree. bvh_group_roots[i] is the internal root-node
    # index of local world i for i < world_count; bvh_group_roots[world_count] is the root of
    # the global (particle world -1) feature group. A root of -1 means that group is empty.
    # A local rigid shape must query both its own world (pass 0) and global features (pass 1).
    # A global rigid shape is compatible with every world, so pass 0 queries the whole BVH and
    # pass 1 is skipped.
    for query_pass in range(2):
        run_query = bool(False)
        query_all = bool(False)
        group_root = wp.int32(-1)
        if rigid_world < 0:
            if query_pass == 0:
                run_query = True
                query_all = True  # Global rigid shape: search every soft-feature group once.
        else:
            if query_pass == 0:
                group_root = bvh_group_roots[rigid_world]  # Same local world as the rigid shape.
            else:
                group_root = bvh_group_roots[world_count]  # Global soft features (particle world -1).
            run_query = group_root >= 0  # Skip an empty group, whose root sentinel is -1.
        if run_query:
            query = (
                wp.bvh_query_aabb(bvh_id, lower, upper)
                if query_all
                else wp.bvh_query_aabb(bvh_id, lower, upper, group_root)
            )
            tri = wp.int32(0)
            while wp.bvh_query_next(query, tri):
                v0 = tri_indices[tri, 0]
                v1 = tri_indices[tri, 1]
                v2 = tri_indices[tri, 2]
                if (
                    (particle_flags[v0] & ParticleFlags.ACTIVE) == 0
                    and (particle_flags[v1] & ParticleFlags.ACTIVE) == 0
                    and (particle_flags[v2] & ParticleFlags.ACTIVE) == 0
                ):
                    continue
                closest, _bary, _feature = triangle_closest_point(
                    particle_q[v0], particle_q[v1], particle_q[v2], rigid_point
                )
                radius = wp.max(particle_radius[v0], wp.max(particle_radius[v1], particle_radius[v2]))
                if wp.length(closest - rigid_point) < gap + shape_margin[shape] + radius:
                    _append_candidate(
                        RIGID_SOFT_BVH_CONTACT_FV,
                        tri,
                        shape,
                        rigid_row,
                        candidate_max,
                        candidate_count,
                        candidates,
                    )


@wp.kernel
def detect_rigid_soft_bvh_ee_candidates(
    rigid_edge_table: wp.array[wp.vec3i],
    rigid_edge_vertex_rows: wp.array[wp.vec2i],
    rigid_vertex_position: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    bvh_id: wp.uint64,
    bvh_group_roots: wp.array[wp.int32],
    world_count: wp.int32,
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_margin: wp.array[float],
    shape_flags: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    particle_max_radius: float,
    gap: float,
    parallel_epsilon: float,
    candidate_max: wp.int32,
    candidate_count: wp.array[wp.int32],
    candidates: wp.array[wp.vec4i],
):
    rigid_row = wp.tid()
    entry = rigid_edge_table[rigid_row]
    shape = entry[0]
    if (shape_flags[shape] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return
    _X_bs, X_ws, _X_sw = _shape_frames(shape_body, body_q, shape_transform, shape)
    scale = shape_scale[shape]
    vertex_rows = rigid_edge_vertex_rows[rigid_row]
    r0 = wp.transform_point(X_ws, wp.cw_mul(rigid_vertex_position[vertex_rows[0]], scale))
    r1 = wp.transform_point(X_ws, wp.cw_mul(rigid_vertex_position[vertex_rows[1]], scale))
    broad_radius = gap + shape_margin[shape] + particle_max_radius
    lower = wp.min(r0, r1) - wp.vec3(broad_radius)
    upper = wp.max(r0, r1) + wp.vec3(broad_radius)
    rigid_world = shape_world[shape]

    # Match the FV world-compatibility traversal above: a local rigid edge queries its local
    # soft-edge group and then the global group, while a global rigid edge queries the full BVH once.
    for query_pass in range(2):
        run_query = bool(False)
        query_all = bool(False)
        group_root = wp.int32(-1)
        if rigid_world < 0:
            if query_pass == 0:
                run_query = True
                query_all = True  # Global rigid shape: search every soft-feature group once.
        else:
            if query_pass == 0:
                group_root = bvh_group_roots[rigid_world]  # Same local world as the rigid shape.
            else:
                group_root = bvh_group_roots[world_count]  # Global soft features (particle world -1).
            run_query = group_root >= 0  # Skip an empty group, whose root sentinel is -1.
        if run_query:
            query = (
                wp.bvh_query_aabb(bvh_id, lower, upper)
                if query_all
                else wp.bvh_query_aabb(bvh_id, lower, upper, group_root)
            )
            soft_edge = wp.int32(0)
            while wp.bvh_query_next(query, soft_edge):
                s0_index = edge_indices[soft_edge, 2]
                s1_index = edge_indices[soft_edge, 3]
                if (particle_flags[s0_index] & ParticleFlags.ACTIVE) == 0 and (
                    particle_flags[s1_index] & ParticleFlags.ACTIVE
                ) == 0:
                    continue
                result = wp.closest_point_edge_edge(
                    r0, r1, particle_q[s0_index], particle_q[s1_index], parallel_epsilon
                )
                radius = wp.max(particle_radius[s0_index], particle_radius[s1_index])
                if result[2] < gap + shape_margin[shape] + radius:
                    _append_candidate(
                        RIGID_SOFT_BVH_CONTACT_EE,
                        soft_edge,
                        shape,
                        rigid_row,
                        candidate_max,
                        candidate_count,
                        candidates,
                    )


@wp.kernel
def emit_rigid_soft_bvh_contacts(
    candidate_count: wp.array[wp.int32],
    candidates: wp.array[wp.vec4i],
    particle_q: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    rigid_vertex_table: wp.array[wp.vec2i],
    rigid_vertex_position: wp.array[wp.vec3],
    rigid_vertex_normal: wp.array[wp.vec3],
    rigid_edge_table: wp.array[wp.vec3i],
    rigid_edge_vertex_rows: wp.array[wp.vec2i],
    rigid_edge_outward: wp.array[wp.vec3],
    parallel_epsilon: float,
    tid_base: wp.int32,
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
    tid = wp.tid()
    if tid >= candidate_count[0]:
        return
    candidate = candidates[tid]
    family = candidate[0]
    soft_feature = candidate[1]
    shape = candidate[2]
    rigid_feature = candidate[3]
    X_bs, X_ws, _X_sw = _shape_frames(shape_body, body_q, shape_transform, shape)
    mesh = shape_source_ptr[shape]
    scale = shape_scale[shape]

    corners = wp.vec3i(-1, -1, -1)
    bary = wp.vec3(0.0)
    rigid_point = wp.vec3(0.0)
    soft_point = wp.vec3(0.0)
    outward_local = wp.vec3(0.0)
    body_vel = wp.vec3(0.0)
    particle = wp.int32(-1)

    if family == RIGID_SOFT_BVH_CONTACT_VF:
        particle = soft_feature
        corners = wp.vec3i(particle, -1, -1)
        bary = wp.vec3(1.0, 0.0, 0.0)
        p0, p1, p2 = _mesh_face_points(mesh, rigid_feature, scale)
        soft_local = wp.transform_point(wp.transform_inverse(X_ws), particle_q[particle])
        rigid_point, rigid_bary, _feature = triangle_closest_point(p0, p1, p2, soft_local)
        soft_point = soft_local
        outward_local = _face_outward(p0, p1, p2, scale)
        shape_vel = wp.cw_mul(wp.mesh_eval_velocity(mesh, rigid_feature, rigid_bary[0], rigid_bary[1]), scale)
        body_vel = wp.transform_vector(X_bs, shape_vel)
    elif family == RIGID_SOFT_BVH_CONTACT_FV:
        tri = soft_feature
        v0 = tri_indices[tri, 0]
        v1 = tri_indices[tri, 1]
        v2 = tri_indices[tri, 2]
        corners = wp.vec3i(v0, v1, v2)
        rigid_point = wp.transform_point(X_ws, wp.cw_mul(rigid_vertex_position[rigid_feature], scale))
        soft_point, bary, _feature = triangle_closest_point(particle_q[v0], particle_q[v1], particle_q[v2], rigid_point)
        outward_local = rigid_vertex_normal[rigid_feature]
        local_vertex = rigid_vertex_table[rigid_feature][1]
        shape_vel = wp.cw_mul(wp.mesh_get_velocity(mesh, local_vertex), scale)
        body_vel = wp.transform_vector(X_bs, shape_vel)
    else:
        soft_edge = soft_feature
        s0_index = edge_indices[soft_edge, 2]
        s1_index = edge_indices[soft_edge, 3]
        corners = wp.vec3i(s0_index, s1_index, -1)
        vertex_rows = rigid_edge_vertex_rows[rigid_feature]
        r0 = wp.transform_point(X_ws, wp.cw_mul(rigid_vertex_position[vertex_rows[0]], scale))
        r1 = wp.transform_point(X_ws, wp.cw_mul(rigid_vertex_position[vertex_rows[1]], scale))
        result = wp.closest_point_edge_edge(r0, r1, particle_q[s0_index], particle_q[s1_index], parallel_epsilon)
        rigid_point = (1.0 - result[0]) * r0 + result[0] * r1
        soft_point = (1.0 - result[1]) * particle_q[s0_index] + result[1] * particle_q[s1_index]
        bary = wp.vec3(1.0 - result[1], result[1], 0.0)
        outward_local = rigid_edge_outward[rigid_feature]
        rigid_edge = rigid_edge_table[rigid_feature]
        shape_vel = wp.cw_mul(
            (1.0 - result[0]) * wp.mesh_get_velocity(mesh, rigid_edge[1])
            + result[0] * wp.mesh_get_velocity(mesh, rigid_edge[2]),
            scale,
        )
        body_vel = wp.transform_vector(X_bs, shape_vel)

    if family == RIGID_SOFT_BVH_CONTACT_VF:
        # VF geometry is already shape-local, so X_bs directly stores the rigid point in body space.
        normal = wp.transform_vector(X_ws, _rigid_to_soft_normal(soft_point - rigid_point, outward_local))
        body_pos = wp.transform_point(X_bs, rigid_point)
    else:
        # FV/EE closest points were evaluated in world space and must first be pulled back to body space.
        outward_world = wp.transform_vector(X_ws, outward_local)
        normal = _rigid_to_soft_normal(soft_point - rigid_point, outward_world)
        body_index = shape_body[shape]
        X_wb = wp.transform_identity()
        if body_index >= 0:
            X_wb = body_q[body_index]
        body_pos = wp.transform_point(wp.transform_inverse(X_wb), rigid_point)

    index = counter_increment(soft_contact_count, 0, soft_contact_tids, tid + tid_base, soft_contact_max)
    if index >= 0:
        soft_contact_particle[index] = particle
        soft_contact_indices[index] = corners
        soft_contact_barycentric[index] = bary
        soft_contact_shape[index] = shape
        soft_contact_body_pos[index] = body_pos
        soft_contact_body_vel[index] = body_vel
        soft_contact_normal[index] = wp.normalize(normal)


def launch_rigid_soft_bvh_contacts(
    *,
    model: Model,
    state: State,
    contacts: Contacts,
    detector: TriMeshCollisionDetector | None,
    particle_shape_pairs,
    rigid_vertex_table,
    rigid_vertex_position,
    rigid_vertex_normal,
    rigid_edge_table,
    rigid_edge_vertex_rows,
    rigid_edge_outward,
    candidate_count,
    candidates,
    candidate_max: int,
    gap: float,
    tid_base: int,
    edge_edge_parallel_epsilon: float,
    device,
) -> None:
    """Detect mesh BVH primitive pairs, then emit them through fixed replay threads."""
    candidate_count.zero_()
    common_shape = [
        state.body_q,
        model.shape_transform,
        model.shape_body,
        model.shape_scale,
        model.shape_source_ptr,
        model.shape_margin,
        model.shape_flags,
    ]
    if len(particle_shape_pairs) > 0:
        wp.launch(
            detect_rigid_soft_bvh_vf_candidates,
            dim=len(particle_shape_pairs),
            inputs=[
                particle_shape_pairs,
                state.particle_q,
                model.particle_radius,
                model.particle_flags,
                *common_shape,
                gap,
                candidate_max,
                candidate_count,
                candidates,
            ],
            device=device,
            record_tape=False,
        )
    if detector is not None and len(rigid_vertex_table) > 0 and model.tri_count > 0:
        wp.launch(
            detect_rigid_soft_bvh_fv_candidates,
            dim=len(rigid_vertex_table),
            inputs=[
                rigid_vertex_table,
                rigid_vertex_position,
                state.particle_q,
                model.particle_radius,
                model.particle_flags,
                model.tri_indices,
                detector.bvh_tris.id,
                detector.bvh_tris_group_roots,
                model.world_count,
                *common_shape,
                model.shape_world,
                model.particle_max_radius,
                gap,
                candidate_max,
                candidate_count,
                candidates,
            ],
            device=device,
            record_tape=False,
        )
    if detector is not None and len(rigid_edge_table) > 0 and model.edge_count > 0:
        wp.launch(
            detect_rigid_soft_bvh_ee_candidates,
            dim=len(rigid_edge_table),
            inputs=[
                rigid_edge_table,
                rigid_edge_vertex_rows,
                rigid_vertex_position,
                state.particle_q,
                model.particle_radius,
                model.particle_flags,
                model.edge_indices,
                detector.bvh_edges.id,
                detector.bvh_edges_group_roots,
                model.world_count,
                *common_shape,
                model.shape_world,
                model.particle_max_radius,
                gap,
                edge_edge_parallel_epsilon,
                candidate_max,
                candidate_count,
                candidates,
            ],
            device=device,
            record_tape=False,
        )

    if candidate_max > 0:
        wp.launch(
            emit_rigid_soft_bvh_contacts,
            dim=candidate_max,
            inputs=[
                candidate_count,
                candidates,
                state.particle_q,
                model.tri_indices,
                model.edge_indices,
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_scale,
                model.shape_source_ptr,
                rigid_vertex_table,
                rigid_vertex_position,
                rigid_vertex_normal,
                rigid_edge_table,
                rigid_edge_vertex_rows,
                rigid_edge_outward,
                edge_edge_parallel_epsilon,
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
        )
