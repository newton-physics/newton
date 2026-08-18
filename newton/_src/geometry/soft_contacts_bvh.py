# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""BVH-based full-surface rigid-soft contact for triangle/convex mesh rigid shapes.

The discrete counterpart of :mod:`soft_contacts_sdf` for mesh shapes: three feature-driven
queries -- soft vertex vs rigid triangle (VT), soft triangle vs rigid vertex (TV), soft edge vs rigid
edge (EE) -- forming the classic complete pairing of two triangulated surfaces. Each query
reuses an accelerator that already exists in the pipeline: the per-shape ``wp.Mesh`` BVH for VT,
and the soft self-contact detector's triangle/edge BVHs for TV/EE. One thread per unique feature,
every pair within the query radius reported unfiltered; validity filtering is the consumer's job
at solve time (see the design contract in :class:`~newton.CollisionPipeline`).

Unlike the SDF back-end this needs no provisioned volume SDF and resolves exact mesh geometry
(sharp edges and corners), at the cost of mesh-only scope.

.. note:: **Dense-query contract / consumer responsibility.** Because emission is unfiltered, a
    soft feature near a mesh edge or corner receives one record per nearby rigid primitive, and a
    record whose sign check failed (the soft point lies behind that one triangle's plane, e.g.
    resting on the adjacent face) carries the rigid feature's outward normal. A consumer that
    treats every record as an independent penetration -- as
    :class:`~newton.solvers.SolverVBD`'s soft-contact force currently does -- will apply spurious
    lateral forces at such edges. Updating VBD's collision handling for dense queries (per-record
    validity filtering, the design doc's shared-validity-utility future work) is tracked as
    follow-up work; until then the SDF back-end remains the artifact-free choice for VBD.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .collision_core import transform_normal_with_scale
from .flags import ParticleFlags, ShapeFlags
from .kernels import counter_increment, triangle_closest_point
from .soft_contacts_common import _shape_frames, _write_soft_contact

if TYPE_CHECKING:
    from ..sim.contacts import Contacts
    from ..sim.model import Model
    from ..sim.state import State
    from .tri_mesh_collision import TriMeshCollisionDetector
    from .types import Mesh

# Below this closest-point distance the direction (x_soft - x_rigid)/d is numerically meaningless
# and the rigid feature's outward normal is emitted instead (see _oriented_contact_normal).
# 1e-6 m sits well above float32 cancellation noise for meter-scale coordinates (~1e-8 absolute
# error on the subtraction), where a smaller epsilon would let rounding noise pass as a direction.
CONTACT_NORMAL_DEGENERATE_EPS = wp.constant(1.0e-6)


# ---------------------------------------------------------------------------
# Host-side rigid feature tables
# ---------------------------------------------------------------------------


def _mesh_feature_data(mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-mesh feature tables: unique vertices and edges with outward directions.

    Table entries are *face-vertex* indices into the mesh index buffer, not vertex ids: a single
    such index serves both the position and the velocity fetch in the kernels
    (``wp.mesh_get_point`` and ``wp.mesh_eval_velocity``). Vertices are folded to canonical ids
    (the same 1e-7 quantization as :attr:`Mesh.edges`), so a seam-duplicated vertex yields one
    table row and normals accumulate across the seam. All normals are mesh-local and unit length.

    Returns:
        Four per-mesh numpy arrays, row-aligned in pairs:

        - ``vertex_table``: one representative face-vertex index per canonical vertex.
        - ``vertex_normals``: the row's angle-weighted unit vertex normal.
        - ``edge_table``: one ``(index0, index1)`` face-vertex row per unique edge.
        - ``edge_outward``: the row's unit outward direction (mean of adjacent face normals).
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    idx = np.asarray(mesh.indices, dtype=np.int32).reshape(-1)
    if idx.size == 0 or verts.size == 0:
        empty_i = np.empty(0, dtype=np.int32)
        return empty_i, np.empty((0, 3), np.float32), np.empty((0, 2), np.int32), np.empty((0, 3), np.float32)
    tris = idx.reshape(-1, 3)
    canonical = mesh._canonical_vertex_ids()
    n_canon = int(canonical.max()) + 1

    # Shared per-slot edge topology and face normals -- the same canonical-pair encoding the
    # dihedral filter uses (one entry per (triangle, edge_in_tri) pair; entry s belongs to
    # triangle s // 3). Degenerate (zero-area) faces contribute nothing to the normals.
    orig_edges, slot_keys, _sort_order, _keys_sorted, face_normals, face_norms = mesh._build_edge_slot_topology()
    valid_faces = face_norms > 0.0
    fn_unit = np.zeros_like(face_normals)
    fn_unit[valid_faces] = face_normals[valid_faces] / face_norms[valid_faces, None]

    def _unit(rows: np.ndarray) -> np.ndarray:
        lengths = np.linalg.norm(rows, axis=1)
        out = np.zeros_like(rows)
        nz = lengths > 0.0
        out[nz] = rows[nz] / lengths[nz, None]
        out[~nz] = (0.0, 0.0, 1.0)  # fully degenerate feature: any fixed unit direction
        return out

    # Angle-weighted vertex normals over canonical ids.
    # (preferred over area-weighted in "compute_vertex_normals")
    vertex_normal_accumulation = np.zeros((n_canon, 3), dtype=np.float64)
    corner_edges = ((1, 2), (2, 0), (0, 1))  # corner k spans the edges to the other two corners
    for k, (a, b) in enumerate(corner_edges):
        da = verts[tris[:, a]] - verts[tris[:, k]]
        db = verts[tris[:, b]] - verts[tris[:, k]]
        la = np.linalg.norm(da, axis=1)
        lb = np.linalg.norm(db, axis=1)
        valid_corners = valid_faces & (la > 0.0) & (lb > 0.0)
        cos_angle = np.zeros(len(tris))
        cos_angle[valid_corners] = np.clip(
            np.einsum("ij,ij->i", da[valid_corners], db[valid_corners]) / (la[valid_corners] * lb[valid_corners]),
            -1.0,
            1.0,
        )
        angle = np.where(valid_corners, np.arccos(cos_angle), 0.0)
        np.add.at(vertex_normal_accumulation, canonical[tris[:, k]], fn_unit * angle[:, None])

    # One representative face-vertex index per canonical vertex that appears in a face.
    canon_per_face_vertex = canonical[idx]
    used_canon, first_index = np.unique(canon_per_face_vertex, return_index=True)
    vertex_table = first_index.astype(np.int32)
    vertex_normals = _unit(vertex_normal_accumulation[used_canon]).astype(np.float32)

    # Unique edges over the packed canonical keys; outward = mean of adjacent face normals
    # (a boundary edge keeps its single face normal). Same dedup as Mesh.edges. Slot s maps to
    # triangle s // 3, so repeating each face normal 3x aligns it with the index table.
    index_of_canon = np.full(n_canon, -1, dtype=np.int32)
    index_of_canon[used_canon] = vertex_table
    _, first_idx, inverse = np.unique(slot_keys, return_index=True, return_inverse=True)
    edge_normal_accumulation = np.zeros((len(first_idx), 3), dtype=np.float64)
    np.add.at(edge_normal_accumulation, inverse, np.repeat(fn_unit, 3, axis=0))
    edge_outward = _unit(edge_normal_accumulation).astype(np.float32)
    edge_canon = canonical[orig_edges[first_idx]]
    edge_table = np.column_stack((index_of_canon[edge_canon[:, 0]], index_of_canon[edge_canon[:, 1]])).astype(np.int32)

    return vertex_table, vertex_normals, edge_table, edge_outward


def build_full_surface_bvh_rigid_features(
    model: Model, bvh_shape_mask: np.ndarray
) -> tuple[wp.array[wp.vec2i], wp.array[wp.vec3], wp.array[wp.vec3i], wp.array[wp.vec3]]:
    """Build the flat rigid feature tables over all shapes selected by ``bvh_shape_mask``.

    Per-mesh numpy computation is cached by the mesh content hash (the key the builder already
    dedups ``wp.Mesh`` finalization with), so instanced shapes pay once; each instance still gets
    its own table rows because transforms differ. Host-side and allocation-heavy: call at pipeline
    construction, never inside a CUDA graph capture.

    Returns:
        Four flat, row-aligned arrays:

        - ``rigid_vertex_table`` (``wp.vec2i``): one ``(shape, face_vertex_index)``
          row per canonical vertex of each selected shape.
        - ``rigid_vertex_normals`` (``wp.vec3``): the row's mesh-local unit
          vertex normal (angle-weighted).
        - ``rigid_edge_table`` (``wp.vec3i``): one ``(shape, index0, index1)`` row
          per unique edge; endpoints are face-vertex indices.
        - ``rigid_edge_outward_dirs`` (``wp.vec3``): the row's mesh-local unit
          outward direction (mean of adjacent face normals).
    """
    device = model.device
    vertex_rows: list[np.ndarray] = []
    vertex_normals: list[np.ndarray] = []
    edge_rows: list[np.ndarray] = []
    edge_outwards: list[np.ndarray] = []
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    for shape in np.flatnonzero(bvh_shape_mask):
        mesh = model.shape_source[shape]
        if mesh is None:
            # The mask selects MESH/CONVEX_MESH shapes, which always carry a source Mesh; a
            # missing one would leave the shape in the VT pairs querying a null wp.Mesh id, so
            # fail loudly instead of silently dropping its TV/EE rows.
            raise ValueError(f"mesh/convex shape {int(shape)} has no shape_source Mesh")
        key = hash(mesh)
        data = cache.get(key)
        if data is None:
            data = _mesh_feature_data(mesh)
            cache[key] = data
        v_table, v_normals, e_table, e_outward = data
        if len(v_table):
            vertex_rows.append(np.column_stack((np.full(len(v_table), shape, np.int32), v_table)))
            vertex_normals.append(v_normals)
        if len(e_table):
            edge_rows.append(np.column_stack((np.full(len(e_table), shape, np.int32), e_table)))
            edge_outwards.append(e_outward)

    def _stack(rows: list[np.ndarray], width: int) -> np.ndarray:
        return np.concatenate(rows) if rows else np.empty((0, width), np.int32)

    def _stackf(rows: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(rows) if rows else np.empty((0, 3), np.float32)

    return (
        wp.array(_stack(vertex_rows, 2), dtype=wp.vec2i, device=device),
        wp.array(_stackf(vertex_normals), dtype=wp.vec3, device=device),
        wp.array(_stack(edge_rows, 3), dtype=wp.vec3i, device=device),
        wp.array(_stackf(edge_outwards), dtype=wp.vec3, device=device),
    )


# ---------------------------------------------------------------------------
# Kernels: two-stage detect + emit
# ---------------------------------------------------------------------------
# Differentiable-replay contract: ``counter_increment`` memoizes ONE record index per thread
# (see :func:`soft_contacts_common._write_soft_contact`), so a taped kernel may emit at most
# once per thread. A BVH traversal
# discovers an unbounded number of pairs, so detection and emission are split. The DETECT kernels
# traverse the BVHs and append compact integer candidates with no tape recording (which pairs are
# within range is a discrete, gradient-free decision, like the broad phase). The EMIT kernel runs
# one thread per candidate slot, recomputes the contact geometry differentiably from the live
# state, and emits exactly once -- restoring the one-emission-per-thread contract, so BVH records
# are differentiable like the legacy and SDF records.
#
# Candidate layout: wp.vec4i(family, soft_feature, rigid_shape, rigid_feature)
#
#   family | soft_feature      | rigid_feature
#   -------+-------------------+----------------------------------------------
#   VT     | particle index    | face index local to the rigid shape's mesh
#   TV     | soft triangle     | row in rigid_vertex_table
#   EE     | soft edge         | row in rigid_edge_table
#
# For TV/EE, rigid_shape duplicates the shape id stored in the referenced table row so the emit
# kernel resolves per-shape data without a second lookup.

BVH_CANDIDATE_VT = wp.constant(0)
BVH_CANDIDATE_TV = wp.constant(1)
BVH_CANDIDATE_EE = wp.constant(2)


@wp.func
def _append_bvh_candidate(
    family: wp.int32,
    soft_feature: wp.int32,
    rigid_shape: wp.int32,
    rigid_feature: wp.int32,
    candidate_max: wp.int32,
    candidate_count: wp.array[wp.int32],
    candidates: wp.array[wp.vec4i],
):
    """Append one candidate; the counter keeps counting past capacity (attempted count) while the
    write is guarded, mirroring the record stream's overflow semantics."""
    index = wp.atomic_add(candidate_count, 0, 1)
    if index < candidate_max:
        candidates[index] = wp.vec4i(family, soft_feature, rigid_shape, rigid_feature)


@wp.func
def _oriented_contact_normal(diff: wp.vec3, d: float, reference: wp.vec3):
    """World-space rigid->soft contact normal: the closest-point direction ``diff/d`` when it is
    well-defined and agrees with the rigid feature's outward ``reference``; the reference itself
    when the pair is degenerate (``d < eps``) or penetrated (sign test fails), keeping the sign
    well-defined exactly where a penetration-free consumer needs it."""
    if d > CONTACT_NORMAL_DEGENERATE_EPS and wp.dot(diff, reference) >= 0.0:
        return diff / d
    return reference


@wp.func
def _face_vertex_velocity(mesh: wp.uint64, index: wp.int32):
    """Surface velocity stored on the mesh at a face-vertex index (mesh-local, unscaled)."""
    face = index / 3
    corner = index - face * 3
    u = wp.where(corner == 0, 1.0, 0.0)
    v = wp.where(corner == 1, 1.0, 0.0)
    return wp.mesh_eval_velocity(mesh, face, u, v)


@wp.kernel(enable_backward=False)
def detect_bvh_candidates_vt(
    vt_pairs: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_margin: wp.array[float],
    gap: float,
    candidate_max: wp.int32,
    # outputs
    candidate_count: wp.array[wp.int32],
    candidates: wp.array[wp.vec4i],
):
    """One thread per world-compatible (particle, BVH-backend shape) pair: query the rigid shape's
    own ``wp.Mesh`` BVH and append one candidate per rigid triangle within range. Replaces (not
    supplements) the legacy closest-point record for these shapes -- a pinch between two patches of
    one mesh yields both records. Flag semantics match the legacy particle pass."""
    tid = wp.tid()
    pair = vt_pairs[tid]
    particle_index = pair[0]
    shape_index = pair[1]
    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return
    if (shape_flags[shape_index] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return

    px = particle_q[particle_index]
    radius = particle_radius[particle_index]
    _X_bs, _X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape_index)
    x_local = wp.transform_point(X_sw, px)
    scale = shape_scale[shape_index]
    s_margin = shape_margin[shape_index] if shape_margin.shape[0] > 0 else 0.0
    threshold = gap + s_margin + radius

    mesh = shape_source_ptr[shape_index]
    # The mesh BVH lives in unscaled mesh space; min |scale| conservatively converts the query
    # radius (magnitudes: mirror parity must not shrink the search).
    min_scale = wp.min(wp.min(wp.abs(scale[0]), wp.abs(scale[1])), wp.abs(scale[2]))
    x_mesh = wp.cw_div(x_local, scale)
    r_mesh = threshold / min_scale
    lower = wp.vec3(x_mesh[0] - r_mesh, x_mesh[1] - r_mesh, x_mesh[2] - r_mesh)
    upper = wp.vec3(x_mesh[0] + r_mesh, x_mesh[1] + r_mesh, x_mesh[2] + r_mesh)

    query = wp.mesh_query_aabb(mesh, lower, upper)
    face = wp.int32(0)
    while wp.mesh_query_aabb_next(query, face):
        a = wp.cw_mul(wp.mesh_get_point(mesh, face * 3 + 0), scale)
        b = wp.cw_mul(wp.mesh_get_point(mesh, face * 3 + 1), scale)
        c = wp.cw_mul(wp.mesh_get_point(mesh, face * 3 + 2), scale)
        cp, _bary, _feature = triangle_closest_point(a, b, c, x_local)
        if wp.length(x_local - cp) < threshold:
            if wp.length_sq(wp.cross(b - a, c - a)) == 0.0:
                continue  # degenerate sliver: no meaningful normal, neighbors still report
            _append_bvh_candidate(
                BVH_CANDIDATE_VT, particle_index, shape_index, face, candidate_max, candidate_count, candidates
            )


@wp.kernel(enable_backward=False)
def detect_bvh_candidates_tv(
    rigid_vertex_table: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_world: wp.array[wp.int32],
    shape_margin: wp.array[float],
    bvh_tris_id: wp.uint64,
    bvh_tris_group_roots: wp.array[wp.int32],
    world_count: wp.int32,
    gap: float,
    max_particle_radius: float,
    candidate_max: wp.int32,
    # outputs
    candidate_count: wp.array[wp.int32],
    candidates: wp.array[wp.vec4i],
):
    """One thread per rigid mesh vertex: query the soft triangle BVH (world-grouped, keyed on the
    shape's world -- the :func:`vertex_triangle_collision_detection_kernel` pattern) and append one
    candidate per soft triangle within range. The query AABB is inflated with the model-wide
    maximum particle radius; the accept test applies the barycentric-interpolated per-pair radius."""
    tid = wp.tid()
    entry = rigid_vertex_table[tid]
    shape_index = entry[0]
    index = entry[1]
    if (shape_flags[shape_index] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return

    scale = shape_scale[shape_index]
    _X_bs, X_ws, _X_sw = _shape_frames(shape_body, body_q, shape_transform, shape_index)
    mesh = shape_source_ptr[shape_index]
    x_local = wp.cw_mul(wp.mesh_get_point(mesh, index), scale)
    x_w = wp.transform_point(X_ws, x_local)
    s_margin = shape_margin[shape_index] if shape_margin.shape[0] > 0 else 0.0
    bound = gap + s_margin + max_particle_radius
    lower = wp.vec3(x_w[0] - bound, x_w[1] - bound, x_w[2] - bound)
    upper = wp.vec3(x_w[0] + bound, x_w[1] + bound, x_w[2] + bound)

    rigid_world = shape_world[shape_index]

    # A real-world shape queries two subtrees (its world's, then the global group); a global
    # (world -1) shape can hit any world and runs a single full-tree pass.
    for query_pass in range(2):
        run_query = bool(False)
        query_all = bool(False)
        group_root = wp.int32(-1)

        if rigid_world < 0:
            if query_pass == 0:
                run_query = True
                query_all = True
        else:
            if query_pass == 0:
                group_root = bvh_tris_group_roots[rigid_world]
            else:
                group_root = bvh_tris_group_roots[world_count]
            run_query = group_root >= 0

        if run_query:
            if query_all:
                query = wp.bvh_query_aabb(bvh_tris_id, lower, upper)
            else:
                query = wp.bvh_query_aabb(bvh_tris_id, lower, upper, group_root)

            tri_index = wp.int32(0)
            while wp.bvh_query_next(query, tri_index):
                t0 = tri_indices[tri_index, 0]
                t1 = tri_indices[tri_index, 1]
                t2 = tri_indices[tri_index, 2]
                # Skip only a fully inactive feature; a partially active one still takes forces.
                active = (
                    (particle_flags[t0] & ParticleFlags.ACTIVE)
                    | (particle_flags[t1] & ParticleFlags.ACTIVE)
                    | (particle_flags[t2] & ParticleFlags.ACTIVE)
                )
                if active == 0:
                    continue

                cp, bary, _feature = triangle_closest_point(particle_q[t0], particle_q[t1], particle_q[t2], x_w)
                r_soft = bary[0] * particle_radius[t0] + bary[1] * particle_radius[t1] + bary[2] * particle_radius[t2]
                if wp.length(cp - x_w) < gap + s_margin + r_soft:
                    _append_bvh_candidate(
                        BVH_CANDIDATE_TV, tri_index, shape_index, tid, candidate_max, candidate_count, candidates
                    )


@wp.kernel(enable_backward=False)
def detect_bvh_candidates_ee(
    rigid_edge_table: wp.array[wp.vec3i],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_world: wp.array[wp.int32],
    shape_margin: wp.array[float],
    bvh_edges_id: wp.uint64,
    bvh_edges_group_roots: wp.array[wp.int32],
    world_count: wp.int32,
    edge_edge_parallel_epsilon: float,
    gap: float,
    max_particle_radius: float,
    candidate_max: wp.int32,
    # outputs
    candidate_count: wp.array[wp.int32],
    candidates: wp.array[wp.vec4i],
):
    """One thread per rigid mesh edge: query the soft edge BVH (world-grouped, keyed on the shape's
    world) and append one candidate per soft edge within range."""
    tid = wp.tid()
    entry = rigid_edge_table[tid]
    shape_index = entry[0]
    index0 = entry[1]
    index1 = entry[2]
    if (shape_flags[shape_index] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return

    scale = shape_scale[shape_index]
    _X_bs, X_ws, _X_sw = _shape_frames(shape_body, body_q, shape_transform, shape_index)
    mesh = shape_source_ptr[shape_index]
    r0_w = wp.transform_point(X_ws, wp.cw_mul(wp.mesh_get_point(mesh, index0), scale))
    r1_w = wp.transform_point(X_ws, wp.cw_mul(wp.mesh_get_point(mesh, index1), scale))
    s_margin = shape_margin[shape_index] if shape_margin.shape[0] > 0 else 0.0
    bound = gap + s_margin + max_particle_radius
    lower = wp.min(r0_w, r1_w)
    upper = wp.max(r0_w, r1_w)
    lower = wp.vec3(lower[0] - bound, lower[1] - bound, lower[2] - bound)
    upper = wp.vec3(upper[0] + bound, upper[1] + bound, upper[2] + bound)

    rigid_world = shape_world[shape_index]

    for query_pass in range(2):
        run_query = bool(False)
        query_all = bool(False)
        group_root = wp.int32(-1)

        if rigid_world < 0:
            if query_pass == 0:
                run_query = True
                query_all = True
        else:
            if query_pass == 0:
                group_root = bvh_edges_group_roots[rigid_world]
            else:
                group_root = bvh_edges_group_roots[world_count]
            run_query = group_root >= 0

        if run_query:
            if query_all:
                query = wp.bvh_query_aabb(bvh_edges_id, lower, upper)
            else:
                query = wp.bvh_query_aabb(bvh_edges_id, lower, upper, group_root)

            edge_index = wp.int32(0)
            while wp.bvh_query_next(query, edge_index):
                # edge_indices rows are [o0, o1, v0, v1]; cols 2,3 are the endpoints.
                sv0 = edge_indices[edge_index, 2]
                sv1 = edge_indices[edge_index, 3]
                active = (particle_flags[sv0] & ParticleFlags.ACTIVE) | (particle_flags[sv1] & ParticleFlags.ACTIVE)
                if active == 0:
                    continue

                std = wp.closest_point_edge_edge(
                    r0_w, r1_w, particle_q[sv0], particle_q[sv1], edge_edge_parallel_epsilon
                )
                r_soft = wp.max(particle_radius[sv0], particle_radius[sv1])
                if std[2] < gap + s_margin + r_soft:
                    _append_bvh_candidate(
                        BVH_CANDIDATE_EE, edge_index, shape_index, tid, candidate_max, candidate_count, candidates
                    )


@wp.kernel
def emit_bvh_contacts(
    candidate_count: wp.array[wp.int32],
    candidates: wp.array[wp.vec4i],
    candidate_max: wp.int32,
    particle_q: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    rigid_vertex_table: wp.array[wp.vec2i],
    rigid_vertex_normals: wp.array[wp.vec3],
    rigid_edge_table: wp.array[wp.vec3i],
    rigid_edge_outward_dirs: wp.array[wp.vec3],
    edge_edge_parallel_epsilon: float,
    tid_base: wp.int32,
    soft_contact_max: wp.int32,
    # outputs
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
    """One thread per candidate slot: recompute the contact geometry differentiably and emit once.

    Threads beyond the (clamped) candidate count early-out, so the launch dim is fixed at the
    candidate capacity and the kernel is CUDA-graph-capturable. Exactly one record emission per
    thread (``counter_increment`` + :func:`soft_contacts_common._write_soft_contact`) keeps the
    differentiable replay memo valid, and reading positions from the live ``particle_q``/``body_q``
    (rather than values stored at detection) is what lets gradients flow to the state."""
    tid = wp.tid()
    if tid >= wp.min(candidate_count[0], candidate_max):
        return
    candidate = candidates[tid]
    family = candidate[0]
    soft_feature = candidate[1]
    shape_index = candidate[2]
    rigid_feature = candidate[3]

    X_bs, X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape_index)
    mesh = shape_source_ptr[shape_index]
    scale = shape_scale[shape_index]

    particle = wp.int32(-1)
    corners = wp.vec3i(-1, -1, -1)
    bary = wp.vec3(0.0)
    body_pos = wp.vec3(0.0)
    body_vel = wp.vec3(0.0)
    normal = wp.vec3(0.0)

    if family == BVH_CANDIDATE_VT:
        particle = soft_feature
        corners = wp.vec3i(particle, -1, -1)
        bary = wp.vec3(1.0, 0.0, 0.0)
        face = rigid_feature
        x_local = wp.transform_point(X_sw, particle_q[particle])
        a = wp.cw_mul(wp.mesh_get_point(mesh, face * 3 + 0), scale)
        b = wp.cw_mul(wp.mesh_get_point(mesh, face * 3 + 1), scale)
        c = wp.cw_mul(wp.mesh_get_point(mesh, face * 3 + 2), scale)
        cp, rigid_bary, _feature = triangle_closest_point(a, b, c, x_local)
        diff = x_local - cp
        # A mirrored (negative-determinant) scale flips triangle winding; keep normals outward.
        # Detection skips zero-area faces, so the normalization is safe.
        det_sign = wp.sign(scale[0] * scale[1] * scale[2])
        tri_n = wp.normalize(wp.cross(b - a, c - a)) * det_sign
        normal = wp.transform_vector(X_ws, _oriented_contact_normal(diff, wp.length(diff), tri_n))
        v_local = wp.cw_mul(wp.mesh_eval_velocity(mesh, face, rigid_bary[0], rigid_bary[1]), scale)
        body_pos = wp.transform_point(X_bs, cp)
        body_vel = wp.transform_vector(X_bs, v_local)
    elif family == BVH_CANDIDATE_TV:
        vertex_entry = rigid_vertex_table[rigid_feature]
        index = vertex_entry[1]
        t0 = tri_indices[soft_feature, 0]
        t1 = tri_indices[soft_feature, 1]
        t2 = tri_indices[soft_feature, 2]
        corners = wp.vec3i(t0, t1, t2)
        x_local = wp.cw_mul(wp.mesh_get_point(mesh, index), scale)
        x_w = wp.transform_point(X_ws, x_local)
        cp, bary, _feature = triangle_closest_point(particle_q[t0], particle_q[t1], particle_q[t2], x_w)
        diff = cp - x_w
        n_ref = transform_normal_with_scale(X_ws, scale, rigid_vertex_normals[rigid_feature])
        normal = _oriented_contact_normal(diff, wp.length(diff), n_ref)
        v_local = wp.cw_mul(_face_vertex_velocity(mesh, index), scale)
        body_pos = wp.transform_point(X_bs, x_local)
        body_vel = wp.transform_vector(X_bs, v_local)
    else:
        edge_entry = rigid_edge_table[rigid_feature]
        index0 = edge_entry[1]
        index1 = edge_entry[2]
        sv0 = edge_indices[soft_feature, 2]
        sv1 = edge_indices[soft_feature, 3]
        corners = wp.vec3i(sv0, sv1, -1)
        r0_local = wp.cw_mul(wp.mesh_get_point(mesh, index0), scale)
        r1_local = wp.cw_mul(wp.mesh_get_point(mesh, index1), scale)
        r0_w = wp.transform_point(X_ws, r0_local)
        r1_w = wp.transform_point(X_ws, r1_local)
        s0 = particle_q[sv0]
        s1 = particle_q[sv1]
        std = wp.closest_point_edge_edge(r0_w, r1_w, s0, s1, edge_edge_parallel_epsilon)
        s = std[0]
        t = std[1]
        bary = wp.vec3(1.0 - t, t, 0.0)
        x_rigid = r0_w + s * (r1_w - r0_w)
        x_soft = s0 + t * (s1 - s0)
        e_rigid = r1_w - r0_w
        e_soft = s1 - s0
        cr = wp.cross(e_rigid, e_soft)
        cr_len = wp.length(cr)
        outward_w = transform_normal_with_scale(X_ws, scale, rigid_edge_outward_dirs[rigid_feature])
        # Scale-invariant near-parallel test: |cross| / (|e_r| |e_s|) is sin(angle), matching the
        # epsilon's meaning; a raw magnitude test would go dead for millimeter-scale edges.
        if cr_len > edge_edge_parallel_epsilon * wp.length(e_rigid) * wp.length(e_soft):
            n_ref = cr / cr_len
            if wp.dot(n_ref, outward_w) < 0.0:
                n_ref = -n_ref
        else:
            n_ref = outward_w
        diff = x_soft - x_rigid
        normal = _oriented_contact_normal(diff, wp.length(diff), n_ref)
        v0 = _face_vertex_velocity(mesh, index0)
        v1 = _face_vertex_velocity(mesh, index1)
        body_pos = wp.transform_point(X_bs, r0_local + s * (r1_local - r0_local))
        body_vel = wp.transform_vector(X_bs, wp.cw_mul((1.0 - s) * v0 + s * v1, scale))

    # counter_increment is still needed here: BVH records append to the SHARED stream behind the
    # legacy + SDF records, whose count this step is dynamic, so a thread cannot compute its row
    # statically from tid -- the atomic claims the next free row past that tail, and the tids memo
    # keeps the claimed row reproducible for backward replay (valid again because this kernel
    # emits exactly once per thread). Must be called from the kernel body (not a nested wp.func)
    # for the replay substitution to apply; see _write_soft_contact.
    idx = counter_increment(soft_contact_count, 0, soft_contact_tids, tid + tid_base, soft_contact_max)
    _write_soft_contact(
        idx,
        soft_contact_particle,
        soft_contact_indices,
        soft_contact_barycentric,
        soft_contact_shape,
        soft_contact_body_pos,
        soft_contact_body_vel,
        soft_contact_normal,
        particle,
        corners,
        bary,
        shape_index,
        body_pos,
        body_vel,
        normal,
    )


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


def launch_soft_bvh_contacts(
    *,
    model: Model,
    state: State,
    contacts: Contacts,
    gap: float,
    device,
    vt_pairs: wp.array[wp.vec2i],
    rigid_vertex_table: wp.array[wp.vec2i],
    rigid_vertex_normals: wp.array[wp.vec3],
    rigid_edge_table: wp.array[wp.vec3i],
    rigid_edge_outward_dirs: wp.array[wp.vec3],
    detector: TriMeshCollisionDetector | None,
    max_particle_radius: float,
    tid_base: int,
    candidate_count: wp.array[wp.int32],
    candidates: wp.array[wp.vec4i],
):
    """Launch the BVH full-surface passes: three candidate DETECT kernels, then one EMIT kernel.

    The split exists because the usual single-kernel emission -- detect and emit in one taped
    kernel, appending records through :func:`kernels.counter_increment` as the legacy and SDF
    passes do -- computes silently WRONG gradients here (the adjoint runs; its results are
    misrouted). During backward, Warp re-executes a taped
    kernel's forward body, and :func:`kernels.counter_increment` makes the nondeterministic atomic
    record index reproducible by memoizing it in ONE ``soft_contact_tids`` slot per thread. A BVH
    traversal thread emits once per hit, so a second emission overwrites the memo and backward
    replay routes every adjoint of that thread through the last record's row -- adjoints of the
    earlier records are dropped, the last record's is applied through every emission's Jacobian,
    and the counter is re-incremented on top. Splitting restores
    the one-emission-per-thread contract: the detect kernels run off the tape
    (``record_tape=False`` -- which pairs are within range is a discrete, gradient-free decision)
    and append integer candidates; the emit kernel runs one thread per candidate slot, recomputes
    the geometry differentiably from the live state, and calls ``counter_increment`` exactly once.

    ``tid_base`` is the first free slot in the shared replay-tids array (the legacy + SDF passes
    occupy ``[0, tid_base)``); the emit kernel claims ``len(candidates)`` slots after it, so all
    launch dims and offsets are fixed at pipeline construction and the whole stage stays
    CUDA-graph-capturable. ``detector`` supplies the soft triangle/edge BVHs (and the edge-edge
    parallel epsilon); pass ``None`` when the model has no triangles -- the TV/EE passes are then
    skipped and VT alone is complete (no soft faces or edges exist to miss)."""
    n_vt = int(vt_pairs.shape[0])
    n_tv = int(rigid_vertex_table.shape[0])
    n_ee = int(rigid_edge_table.shape[0])
    candidate_max = int(candidates.shape[0])
    if candidate_max == 0 or (n_vt == 0 and n_tv == 0 and n_ee == 0):
        return

    candidate_count.zero_()

    shape_args = [
        state.body_q,
        model.shape_transform,
        model.shape_body,
        model.shape_flags,
        model.shape_scale,
        model.shape_source_ptr,
    ]
    parallel_epsilon = detector.edge_edge_parallel_epsilon if detector is not None else 1.0e-5

    if n_vt > 0:
        wp.launch(
            detect_bvh_candidates_vt,
            dim=n_vt,
            inputs=[
                vt_pairs,
                state.particle_q,
                model.particle_radius,
                model.particle_flags,
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_flags,
                model.shape_scale,
                model.shape_source_ptr,
                model.shape_margin,
                gap,
                candidate_max,
            ],
            outputs=[candidate_count, candidates],
            device=device,
            record_tape=False,
        )

    if detector is not None and n_tv > 0:
        wp.launch(
            detect_bvh_candidates_tv,
            dim=n_tv,
            inputs=[
                rigid_vertex_table,
                state.particle_q,
                model.particle_radius,
                model.particle_flags,
                model.tri_indices,
                *shape_args,
                model.shape_world,
                model.shape_margin,
                detector.bvh_tris.id,
                detector.bvh_tris_group_roots,
                model.world_count,
                gap,
                max_particle_radius,
                candidate_max,
            ],
            outputs=[candidate_count, candidates],
            device=device,
            record_tape=False,
        )

    if detector is not None and n_ee > 0:
        wp.launch(
            detect_bvh_candidates_ee,
            dim=n_ee,
            inputs=[
                rigid_edge_table,
                state.particle_q,
                model.particle_radius,
                model.particle_flags,
                model.edge_indices,
                *shape_args,
                model.shape_world,
                model.shape_margin,
                detector.bvh_edges.id,
                detector.bvh_edges_group_roots,
                model.world_count,
                parallel_epsilon,
                gap,
                max_particle_radius,
                candidate_max,
            ],
            outputs=[candidate_count, candidates],
            device=device,
            record_tape=False,
        )

    wp.launch(
        emit_bvh_contacts,
        dim=candidate_max,
        inputs=[
            candidate_count,
            candidates,
            candidate_max,
            state.particle_q,
            model.tri_indices,
            model.edge_indices,
            state.body_q,
            model.shape_transform,
            model.shape_body,
            model.shape_scale,
            model.shape_source_ptr,
            rigid_vertex_table,
            rigid_vertex_normals,
            rigid_edge_table,
            rigid_edge_outward_dirs,
            parallel_epsilon,
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
