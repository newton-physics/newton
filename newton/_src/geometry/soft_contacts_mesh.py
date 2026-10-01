# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0


"""Full-surface mesh contacts with local feature validity and final-slot emission.

Vertex-face, face-vertex, and edge-edge pairs describe the same surface-distance
problem at every deformable resolution. Incident-feature cones reject redundant
representations without discarding distinct nearby surface patches. Detection
records immutable feature IDs; a separate differentiable pass evaluates only
the accepted final contacts. No intermediate contact pool is required.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .collision_core import transform_normal_with_scale
from .flags import ParticleFlags, ShapeFlags
from .kernels import triangle_closest_point
from .soft_contacts_sdf import _shape_frames
from .tri_mesh_collision import TriMeshCollisionDetector
from .types import GeoType

if TYPE_CHECKING:
    from ..sim.contacts import Contacts
    from ..sim.model import Model
    from ..sim.state import State
    from .types import Mesh


# One CUDA warp cooperates on each particle query. CPU uses depth zero.
_MESH_QUERY_PARTITION_DEPTH = wp.constant(5)
_MESH_QUERY_PARTITIONS = wp.constant(1 << _MESH_QUERY_PARTITION_DEPTH)


@wp.func_native("""
const wp::Mesh mesh = wp::mesh_get(mesh_id);
int root = *mesh.bvh.root;
if (root < 0) return -2;
for (int level = 0; level < depth; ++level) {
    const auto lower = wp::bvh_load_node(mesh.bvh.node_lowers, root);
    if (lower.b) return (lane & ((1 << (depth - level)) - 1)) == 0 ? root : -2;
    const auto upper = wp::bvh_load_node(mesh.bvh.node_uppers, root);
    root = ((lane >> (depth - 1 - level)) & 1) ? upper.i : lower.i;
}
return root;
""")
def _mesh_query_partition(mesh_id: wp.uint64, lane: int, depth: int) -> int:
    """Partition the existing mesh BVH; -2 marks an unused shallow-tree lane."""
    ...


@wp.func_native("""
int sign = 0;
#if defined(__CUDA_ARCH__)
const unsigned mask = __activemask();
const int leader = (threadIdx.x & 31) & ~(lanes - 1);
if ((threadIdx.x & (lanes - 1)) == 0) {
#endif
    const auto query = wp::mesh_query_point_sign_normal(mesh_id, point, radius);
    sign = query.result ? (query.sign < 0.0f ? -1 : 1) : 0;
#if defined(__CUDA_ARCH__)
}
return __shfl_sync(mask, sign, leader);
#else
return sign;
#endif
""")
def _mesh_partition_sign(mesh_id: wp.uint64, point: wp.vec3, radius: float, lanes: int) -> int:
    """Share an exact nearest query within a power-of-two group of CUDA lanes.

    All lanes in each group must participate with identical query arguments.
    Group size must divide the CUDA warp size and the launch block dimension.
    """
    ...


@wp.func
def _feature_query_capsule(
    scale: wp.vec3,
    transform: wp.transform,
    bounds: wp.vec4,
    error: wp.vec3,
    radius: float,
):
    s = wp.max(wp.abs(scale[0]), wp.max(wp.abs(scale[1]), wp.abs(scale[2])))
    minimum = wp.min(wp.abs(scale[0]), wp.min(wp.abs(scale[1]), wp.abs(scale[2])))
    padding = error[0] * s * s + error[1] * s * radius + error[2] * radius * radius
    axis = wp.normalize(wp.transform_vector(transform, wp.cw_div(wp.vec3(bounds[0], bounds[1], bounds[2]), scale)))
    width = (bounds[3] * s * radius + padding) / minimum
    return axis, wp.min(width, radius)


@lru_cache(maxsize=16)
def _cone_pair_indices(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Reuse pair indices for at most six cone directions or sixteen axes."""
    return np.triu_indices(count, 1)


def _cone_query_bounds(
    point: np.ndarray, positions: np.ndarray, *, edge: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Bound both validity cones by an axis and relative capsule radius.

    Opposing halfspaces imply a slab. Two independent slabs bound transverse
    distance by their smallest Gram eigenvalue. Dropping constraints only
    widens this bound. The returned polynomial propagates the same rounding
    allowance as ``_cone_valid`` through those slab inequalities.
    """
    directions = positions - point
    length = np.linalg.norm(directions, axis=1)
    point_bound = np.linalg.norm(point)
    fallback = np.array([0.0, 0.0, 1.0, 1.0])
    if edge is not None:
        half_length = np.linalg.norm(edge) * 0.5
        if half_length == 0:
            return fallback, np.zeros(3)
        axis = edge / (2 * half_length)
        directions -= (directions @ axis)[:, None] * axis
        length += half_length
        point_bound += half_length
    projected = np.linalg.norm(directions, axis=1)
    valid = projected > 0
    directions = directions[valid] / projected[valid, None]
    p = point_bound + np.linalg.norm(positions[valid], axis=1)
    errors = 2.0e-6 * np.column_stack((p * length[valid], p + length[valid], np.ones(len(p)))) / projected[valid, None]
    if len(directions) < 2:
        return fallback, np.zeros(3)
    # Six extremal halfspaces suffice for a conservative accelerator. Limit
    # precomputation storage/work even at arbitrarily high-valence vertices;
    # the exact acceptance test still uses every incident neighbor.
    if len(directions) > 6:
        support = np.unique(np.concatenate((np.argmin(directions, axis=0), np.argmax(directions, axis=0))))
        directions, errors = directions[support], errors[support]
    i, j = _cone_pair_indices(len(directions))
    delta = directions[i] - directions[j]
    a = np.linalg.norm(delta, axis=1)
    b = np.linalg.norm(directions[i] + directions[j], axis=1)
    valid = a > b
    if not np.any(valid):
        return fallback, np.zeros(3)
    i, j, delta, a, b = i[valid], j[valid], delta[valid], a[valid], b[valid]
    axes = delta / a[:, None]
    width = b / a
    relaxation = 2 * np.maximum(errors[i], errors[j]) / a[:, None]
    if edge is not None:
        axes = np.vstack((axes, axis))
        width = np.append(width, 0.0)
        relaxation = np.vstack((relaxation, np.zeros(3)))
    i, j = _cone_pair_indices(len(axes))
    eigenvalue = 1 - np.abs(np.sum(axes[i] * axes[j], axis=1))
    valid = eigenvalue > 1.0e-12
    if not np.any(valid):
        return fallback, np.zeros(3)
    i, j, eigenvalue = i[valid], j[valid], eigenvalue[valid]
    widths = np.sqrt((width[i] ** 2 + width[j] ** 2) / eigenvalue)
    best = np.argmin(widths)
    if widths[best] >= 1.0:
        return fallback, np.zeros(3)
    direction = np.cross(axes[i[best]], axes[j[best]])
    direction /= np.linalg.norm(direction)
    error = (relaxation[i[best]] + relaxation[j[best]]) / np.sqrt(eigenvalue[best])
    return np.append(direction, widths[best] + 1.0e-5), error


@wp.func
def _cone_valid(
    mesh: wp.uint64, scale: wp.vec3, point: wp.vec3, diff: wp.vec3, span: wp.vec3i, neighbors: wp.array[int]
):
    # Outward separation must not point into any incident feature. The allowance
    # bounds position-subtraction roundoff, rather than imposing a world-unit gap.
    valid = bool(True)
    for k in range(span[0], span[1]):
        neighbor = wp.cw_mul(wp.mesh_get_point(mesh, neighbors[k]), scale)
        direction = neighbor - point
        tolerance = (
            2.0e-6
            * (wp.length(point) + wp.length(neighbor) + wp.length(diff))
            * (wp.length(direction) + wp.length(diff))
        )
        if wp.dot(diff, direction) > tolerance:
            valid = False
    return valid


@wp.func
def _face_valid(
    mesh: wp.uint64,
    scale: wp.vec3,
    point: wp.vec3,
    diff: wp.vec3,
    bary: wp.vec3,
    face: int,
    offset: int,
    vertex_spans: wp.array[wp.vec3i],
    edge_spans: wp.array[wp.vec3i],
    neighbors: wp.array[int],
):
    count = int(0)
    positive = int(0)
    zero = int(0)
    for k in range(3):
        if bary[k] > 0.0:
            count += 1
            positive = k
        else:
            zero = k
    if count == 1:
        span = vertex_spans[offset + 3 * face + positive]
        return span[2] == face and _cone_valid(mesh, scale, point, diff, span, neighbors)
    if count == 2:
        span = edge_spans[offset + 3 * face + zero]
        return span[2] == face and _cone_valid(mesh, scale, point, diff, span, neighbors)
    return True


def _build_feature_adjacency(model: Model, vertex_table: wp.array, edge_table: wp.array):
    """Build fixed incident-feature spans, ownership, and per-shape component counts."""
    et = edge_table.numpy()
    offsets = np.zeros(model.shape_count, dtype=np.int32)
    component_counts = np.zeros(model.shape_count, dtype=np.int32)
    vertex_spans, edge_spans, neighbors = [], [], []
    vertex_bounds, vertex_errors, edge_bounds, edge_errors = [], [], [], []
    ee = np.zeros(len(et), dtype=np.int32)
    cache = {}

    def build(mesh):
        idx = np.asarray(mesh.indices).reshape(-1)
        canon = mesh._canonical_vertex_ids()[idx]
        representative = {}
        incident = {}
        opposite = {}
        edge_owner = {}
        for slot, vertex in enumerate(canon):
            representative.setdefault(int(vertex), slot)
            incident.setdefault(int(vertex), set())
        for face, tri in enumerate(canon.reshape(-1, 3)):
            for k in range(3):
                v = int(tri[k])
                a, b = int(tri[(k + 1) % 3]), int(tri[(k + 2) % 3])
                incident[v].update((a, b))
                key = tuple(sorted((a, b)))
                opposite.setdefault(key, set()).add(v)
                edge_owner.setdefault(key, face)

        unvisited = set(incident)
        component_count = 0
        while unvisited:
            component_count += 1
            pending = [unvisited.pop()]
            while pending:
                connected = incident[pending.pop()] & unvisited
                unvisited.difference_update(connected)
                pending.extend(connected)

        def span(vertices, owner, representatives=representative):
            start = len(neighbors)
            neighbors.extend(representatives[v] for v in sorted(vertices))
            return (start, len(neighbors), owner)

        vs = {v: span(n, representative[v] // 3) for v, n in incident.items()}
        es = {key: span(n, edge_owner[key]) for key, n in opposite.items()}
        points = np.asarray(mesh.vertices, dtype=np.float64)[idx]
        vb = {
            v: (
                _cone_query_bounds(points[representative[v]], points[[representative[n] for n in ns]])
                if len(vertex_table)
                else (np.array([0.0, 0.0, 1.0, 1.0]), np.zeros(3))
            )
            for v, ns in incident.items()
        }
        eb = {}
        edge_slots = {}
        for key, ns in opposite.items():
            p, q = points[[representative[v] for v in key]]
            eb[key] = (
                _cone_query_bounds((p + q) * 0.5, points[[representative[n] for n in ns]], edge=q - p)
                if len(et)
                else (np.array([0.0, 0.0, 1.0, 1.0]), np.zeros(3))
            )
        offset = len(vertex_spans)
        for slot, v in enumerate(canon):
            vertex_spans.append(vs[int(v)])
            vertex_bounds.append(vb[int(v)][0])
            vertex_errors.append(vb[int(v)][1])
            base, k = (slot // 3) * 3, slot % 3
            key = tuple(sorted((int(canon[base + (k + 1) % 3]), int(canon[base + (k + 2) % 3]))))
            edge_spans.append(es[key])
            edge_bounds.append(eb[key][0])
            edge_errors.append(eb[key][1])
            edge_slots.setdefault(key, offset + slot)
        keys = sorted(es)
        edge_keys = np.asarray([(a << 32) | b for a, b in keys], dtype=np.int64)
        edge_data = np.asarray([edge_slots[key] for key in keys], dtype=np.int32)
        return offset, canon, edge_keys, edge_data, component_count

    # Shape instances share immutable local topology. Only the feature rows
    # carry a shape id; constructing full adjacency per world is unnecessary.
    for shape, mesh in enumerate(model.shape_source):
        if mesh is None or not hasattr(mesh, "indices"):
            continue
        key = id(mesh)
        if key not in cache:
            cache[key] = build(mesh)
        offset, canon, edge_keys, edge_data, component_count = cache[key]
        offsets[shape] = offset
        component_counts[shape] = component_count
        start, end = np.searchsorted(et[:, 0], (shape, shape + 1))
        edge_canon = np.sort(canon[et[start:end, 1:]].astype(np.int64), axis=1)
        keys = (edge_canon[:, 0] << 32) | edge_canon[:, 1]
        ee[start:end] = edge_data[np.searchsorted(edge_keys, keys)]
    if max(len(vertex_spans), len(edge_spans), len(neighbors)) > np.iinfo(np.int32).max:
        raise ValueError("Mesh contact topology exceeds 32-bit indexing capacity.")
    arrays = [wp.array(offsets, dtype=int, device=model.device)]
    arrays.extend(
        wp.array(np.asarray(x, dtype=np.int32).reshape(-1, 3), dtype=wp.vec3i, device=model.device)
        for x in (vertex_spans, edge_spans)
    )
    arrays.append(wp.array(ee, dtype=int, device=model.device))
    arrays.append(wp.array(neighbors, dtype=int, device=model.device))
    arrays.extend(
        wp.array(np.asarray(x, dtype=np.float32).reshape(-1, width), dtype=dtype, device=model.device)
        for x, width, dtype in (
            (vertex_bounds, 4, wp.vec4),
            (vertex_errors, 3, wp.vec3),
            (edge_bounds, 4, wp.vec4),
            (edge_errors, 3, wp.vec3),
        )
    )
    return arrays, component_counts


CONTACT_NORMAL_DEGENERATE_EPS = wp.constant(1.0e-6)


def _mesh_feature_data(
    mesh: Mesh, *, collision_edges: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return unique face-vertex slots, edges, and outward reference normals."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    idx = np.asarray(mesh.indices, dtype=np.int32).reshape(-1)
    if idx.size == 0 or verts.size == 0:
        empty_i = np.empty(0, dtype=np.int32)
        return empty_i, np.empty((0, 3), np.float32), np.empty((0, 2), np.int32), np.empty((0, 3), np.float32)
    tris = idx.reshape(-1, 3)
    canonical = mesh._canonical_vertex_ids()
    n_canon = int(canonical.max()) + 1

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

    canon_per_face_vertex = canonical[idx]
    used_canon, first_index = np.unique(canon_per_face_vertex, return_index=True)
    vertex_table = first_index.astype(np.int32)
    vertex_normals = _unit(vertex_normal_accumulation[used_canon]).astype(np.float32)

    index_of_canon = np.full(n_canon, -1, dtype=np.int32)
    index_of_canon[used_canon] = vertex_table
    edge_keys, first_idx, inverse = np.unique(slot_keys, return_index=True, return_inverse=True)
    edge_normal_accumulation = np.zeros((len(first_idx), 3), dtype=np.float64)
    np.add.at(edge_normal_accumulation, inverse, np.repeat(fn_unit, 3, axis=0))
    edge_outward = _unit(edge_normal_accumulation).astype(np.float32)
    edge_canon = canonical[orig_edges[first_idx]]
    if collision_edges is not None:
        # Keep exactly the SDF edge set, including an intentionally empty set.
        # Full triangle adjacency still supplies the normals and validity cones.
        edge_canon = canonical[collision_edges]
        sorted_canon = np.sort(edge_canon.astype(np.int64), axis=1)
        selected_keys = (sorted_canon[:, 0] << 32) | sorted_canon[:, 1]
        edge_outward = edge_outward[np.searchsorted(edge_keys, selected_keys)]
    edge_table = np.column_stack((index_of_canon[edge_canon[:, 0]], index_of_canon[edge_canon[:, 1]])).astype(np.int32)

    return vertex_table, vertex_normals, edge_table, edge_outward


def _build_rigid_features(
    model: Model, bvh_shape_mask: np.ndarray
) -> tuple[wp.array[wp.vec2i], wp.array[wp.vec3], wp.array[wp.vec3i], wp.array[wp.vec3]]:

    device = model.device
    vertex_rows: list[np.ndarray] = []
    vertex_normals: list[np.ndarray] = []
    edge_rows: list[np.ndarray] = []
    edge_outwards: list[np.ndarray] = []
    cache: dict[tuple[int, bytes | None], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    edge_ranges = model.shape_edge_range.numpy()
    packed_edges = model.mesh_edge_indices.numpy()

    for shape in np.flatnonzero(bvh_shape_mask):
        mesh = model.shape_source[shape]
        if mesh is None:
            raise ValueError(f"mesh/convex shape {int(shape)} has no shape_source Mesh")
        start, count = (int(value) for value in edge_ranges[shape])
        # Finalized tables include SDF edges cooked by the builder, which need
        # not be attached to the source Mesh. Slices may differ for one source.
        collision_edges = packed_edges[start : start + count] if start >= 0 else mesh._collision_edges
        key = (id(mesh), None if collision_edges is None else np.asarray(collision_edges).tobytes())
        data = cache.get(key)
        if data is None:
            data = _mesh_feature_data(mesh, collision_edges=collision_edges)
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


_MESH_FEATURE_VT = wp.constant(0)
_MESH_FEATURE_TV = wp.constant(1)
_MESH_FEATURE_EE = wp.constant(2)


@wp.func
def _append_mesh_contact(
    family: wp.int32,
    soft_feature: wp.int32,
    rigid_shape: wp.int32,
    rigid_feature: wp.int32,
    contact_max: wp.int32,
    contact_count: wp.array[wp.int64],
    features: wp.array[wp.vec3i],
    contact_shapes: wp.array[int],
):

    index = wp.atomic_add(contact_count, 0, wp.int64(1))
    if index == wp.int64(contact_max):
        wp.printf("Mesh soft-contact capacity exceeded (%d); this collision result is incomplete.\n", contact_max)
    if index < wp.int64(contact_max):
        features[index] = wp.vec3i(family, soft_feature, rigid_feature)
        contact_shapes[index] = rigid_shape


@wp.kernel(enable_backward=False)
def _begin_mesh_contacts(source: wp.array[int], destination: wp.array[wp.int64]):
    destination[0] = wp.int64(source[0])


@wp.kernel(enable_backward=False)
def _end_mesh_contacts(source: wp.array[wp.int64], destination: wp.array[int], capacity: int):
    destination[0] = wp.int32(wp.min(source[0], wp.int64(capacity) + wp.int64(1)))


@wp.func
def _oriented_contact_normal(diff: wp.vec3, d: float, reference: wp.vec3):

    if d > CONTACT_NORMAL_DEGENERATE_EPS and wp.dot(diff, reference) >= 0.0:
        return diff / d
    return reference


@wp.func
def _face_vertex_velocity(mesh: wp.uint64, index: wp.int32):

    face = index / 3
    corner = index - face * 3
    u = wp.where(corner == 0, 1.0, 0.0)
    v = wp.where(corner == 1, 1.0, 0.0)
    return wp.mesh_eval_velocity(mesh, face, u, v)


@wp.func
def _feature_mesh_sign(mesh: wp.uint64, X_sw: wp.transform, scale: wp.vec3, soft: wp.vec3):
    query_point = wp.cw_div(wp.transform_point(X_sw, soft), scale)
    query = wp.mesh_query_point_sign_normal(mesh, query_point, 1.0e6)
    if not query.result:
        return int(0)
    surface = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
    uncertainty = 2.0e-6 * (wp.length(query_point) + wp.length(surface))
    if wp.length(query_point - surface) <= uncertainty:
        return int(2)
    return wp.where(query.sign < 0.0, -1, 1)


@wp.kernel(enable_backward=False)
def _detect_mesh_vertex_contacts(
    vt_pairs: wp.array[wp.vec2i],
    partition_depth: int,
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_lower: wp.array[wp.vec3],
    shape_upper: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    gap: float,
    contact_max: wp.int32,
    face_offsets: wp.array[int],
    vertex_spans: wp.array[wp.vec3i],
    edge_spans: wp.array[wp.vec3i],
    rigid_edge_slots: wp.array[int],
    neighbors: wp.array[int],
    vertex_bounds: wp.array[wp.vec4],
    vertex_errors: wp.array[wp.vec3],
    edge_bounds: wp.array[wp.vec4],
    edge_errors: wp.array[wp.vec3],
    vertex_outward: wp.array[wp.vec3],
    edge_outward: wp.array[wp.vec3],
    contact_count: wp.array[wp.int64],
    features: wp.array[wp.vec3i],
    contact_shapes: wp.array[int],
):

    partitions = 1 << partition_depth
    tid = wp.tid() >> partition_depth
    lane = wp.tid() & (partitions - 1)
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
    lower_bound = shape_lower[shape_index] - wp.vec3(threshold)
    upper_bound = shape_upper[shape_index] + wp.vec3(threshold)
    if (
        x_local[0] < lower_bound[0]
        or x_local[1] < lower_bound[1]
        or x_local[2] < lower_bound[2]
        or x_local[0] > upper_bound[0]
        or x_local[1] > upper_bound[1]
        or x_local[2] > upper_bound[2]
    ):
        return

    mesh = shape_source_ptr[shape_index]
    min_scale = wp.min(wp.min(wp.abs(scale[0]), wp.abs(scale[1])), wp.abs(scale[2]))
    x_mesh = wp.cw_div(x_local, scale)
    r_mesh = threshold / min_scale
    vertex_sign = _mesh_partition_sign(mesh, x_mesh, r_mesh, partitions)
    if vertex_sign == 0:
        if lane != 0:
            return
        # A finite proximity band must not discard already penetrating points.
        # The expanded shape bounds have already rejected distant particles.
        recovery_radius = wp.length(upper_bound - lower_bound) / min_scale
        recovery = wp.mesh_query_point_sign_normal(mesh, x_mesh, recovery_radius)
        if recovery.result and recovery.sign < 0.0:
            _append_mesh_contact(
                _MESH_FEATURE_VT + 8,
                particle_index,
                shape_index,
                recovery.face,
                contact_max,
                contact_count,
                features,
                contact_shapes,
            )
        return
    root = _mesh_query_partition(mesh, lane, partition_depth)
    if root == -2:
        return
    query = wp.bvh_query_sphere(wp.mesh_get_bvh(mesh), x_mesh, r_mesh, root)
    face = wp.int32(0)
    while wp.bvh_query_next(query, face):
        a = wp.cw_mul(wp.mesh_get_point(mesh, face * 3 + 0), scale)
        b = wp.cw_mul(wp.mesh_get_point(mesh, face * 3 + 1), scale)
        c = wp.cw_mul(wp.mesh_get_point(mesh, face * 3 + 2), scale)
        cp, rigid_bary, _feature = triangle_closest_point(a, b, c, x_local)
        if wp.length(x_local - cp) < threshold:
            if wp.length_sq(wp.cross(b - a, c - a)) == 0.0:
                continue  # degenerate sliver: no meaningful normal, neighbors still report
            if not _face_valid(
                mesh,
                scale,
                cp,
                float(vertex_sign) * (x_local - cp),
                rigid_bary,
                face,
                face_offsets[shape_index],
                vertex_spans,
                edge_spans,
                neighbors,
            ):
                continue
            sign = vertex_sign
            if sign != 0:
                _append_mesh_contact(
                    _MESH_FEATURE_VT + wp.where(sign < 0, 8, 0),
                    particle_index,
                    shape_index,
                    face,
                    contact_max,
                    contact_count,
                    features,
                    contact_shapes,
                )


@wp.kernel(enable_backward=False)
def _detect_mesh_face_contacts(
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
    contact_max: wp.int32,
    face_offsets: wp.array[int],
    vertex_spans: wp.array[wp.vec3i],
    edge_spans: wp.array[wp.vec3i],
    rigid_edge_slots: wp.array[int],
    neighbors: wp.array[int],
    vertex_bounds: wp.array[wp.vec4],
    vertex_errors: wp.array[wp.vec3],
    edge_bounds: wp.array[wp.vec4],
    edge_errors: wp.array[wp.vec3],
    vertex_outward: wp.array[wp.vec3],
    edge_outward: wp.array[wp.vec3],
    contact_count: wp.array[wp.int64],
    features: wp.array[wp.vec3i],
    contact_shapes: wp.array[int],
):

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
    slot = face_offsets[shape_index] + index
    axis, width = _feature_query_capsule(scale, X_ws, vertex_bounds[slot], vertex_errors[slot], bound)
    half_length = wp.where(width < bound, bound, 0.0)
    start = x_w - half_length * axis

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
                group_root = bvh_tris_group_roots[rigid_world]
            else:
                group_root = bvh_tris_group_roots[world_count]
            run_query = group_root >= 0

        if run_query:
            if query_all:
                query = wp.bvh_query_capsule(bvh_tris_id, start, axis, width)
            else:
                query = wp.bvh_query_capsule(bvh_tris_id, start, axis, width, group_root)

            tri_index = wp.int32(0)
            while wp.bvh_query_next(query, tri_index, 2.0 * half_length):
                t0 = tri_indices[tri_index, 0]
                t1 = tri_indices[tri_index, 1]
                t2 = tri_indices[tri_index, 2]
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
                    if bary[0] == 1.0 or bary[1] == 1.0 or bary[2] == 1.0:
                        continue
                    cp_local = wp.transform_point(_X_sw, cp)
                    diff = cp_local - x_local
                    outward = _cone_valid(mesh, scale, x_local, diff, vertex_spans[slot], neighbors)
                    inward = _cone_valid(mesh, scale, x_local, -diff, vertex_spans[slot], neighbors)
                    if not outward and not inward:
                        continue
                    sign = _feature_mesh_sign(mesh, _X_sw, scale, cp)
                    if sign == 2:
                        reference = transform_normal_with_scale(X_ws, scale, vertex_outward[tid])
                        sign = wp.where(wp.dot(cp - x_w, reference) < 0.0, -1, 1)
                    if (sign > 0 and outward) or (sign < 0 and inward):
                        _append_mesh_contact(
                            _MESH_FEATURE_TV + wp.where(sign < 0, 8, 0),
                            tri_index,
                            shape_index,
                            tid,
                            contact_max,
                            contact_count,
                            features,
                            contact_shapes,
                        )


@wp.kernel(enable_backward=False)
def _detect_mesh_edge_contacts(
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
    contact_max: wp.int32,
    face_offsets: wp.array[int],
    vertex_spans: wp.array[wp.vec3i],
    edge_spans: wp.array[wp.vec3i],
    rigid_edge_slots: wp.array[int],
    neighbors: wp.array[int],
    vertex_bounds: wp.array[wp.vec4],
    vertex_errors: wp.array[wp.vec3],
    edge_bounds: wp.array[wp.vec4],
    edge_errors: wp.array[wp.vec3],
    vertex_outward: wp.array[wp.vec3],
    edge_outward: wp.array[wp.vec3],
    contact_count: wp.array[wp.int64],
    features: wp.array[wp.vec3i],
    contact_shapes: wp.array[int],
):

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
    slot = rigid_edge_slots[tid]
    axis, width = _feature_query_capsule(scale, X_ws, edge_bounds[slot], edge_errors[slot], bound)
    half_length = bound + 0.5 * wp.length(r1_w - r0_w)
    width += 0.5 * wp.length(r1_w - r0_w)
    start = 0.5 * (r0_w + r1_w) - half_length * axis
    lower -= wp.vec3(bound)
    upper += wp.vec3(bound)
    capsule_size = 2.0 * (wp.vec3(wp.abs(axis[0]), wp.abs(axis[1]), wp.abs(axis[2])) * half_length + wp.vec3(width))
    box_size = upper - lower
    use_capsule = capsule_size[0] * capsule_size[1] * capsule_size[2] < box_size[0] * box_size[1] * box_size[2]

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
                group_root = -1
            if use_capsule:
                query = wp.bvh_query_capsule(bvh_edges_id, start, axis, width, group_root)
            else:
                query = wp.bvh_query_aabb(bvh_edges_id, lower, upper, group_root)

            edge_index = wp.int32(0)
            while wp.bvh_query_next(query, edge_index, 2.0 * half_length):
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
                    soft_point = particle_q[sv0] + std[1] * (particle_q[sv1] - particle_q[sv0])
                    rigid_point = r0_w + std[0] * (r1_w - r0_w)
                    if std[0] <= 0.0 or std[0] >= 1.0 or std[1] <= 0.0 or std[1] >= 1.0:
                        continue
                    rigid_local = wp.transform_point(_X_sw, rigid_point)
                    diff_local = wp.transform_vector(_X_sw, soft_point - rigid_point)
                    outward = _cone_valid(mesh, scale, rigid_local, diff_local, edge_spans[slot], neighbors)
                    inward = _cone_valid(mesh, scale, rigid_local, -diff_local, edge_spans[slot], neighbors)
                    if not outward and not inward:
                        continue
                    sign = _feature_mesh_sign(mesh, _X_sw, scale, soft_point)
                    if sign == 2:
                        reference = transform_normal_with_scale(X_ws, scale, edge_outward[tid])
                        sign = wp.where(wp.dot(soft_point - rigid_point, reference) < 0.0, -1, 1)
                    if not ((sign > 0 and outward) or (sign < 0 and inward)):
                        continue
                    valid_soft = bool(True)
                    for side in range(2):
                        opposite = edge_indices[edge_index, side]
                        if opposite >= 0:
                            direction = particle_q[opposite] - soft_point
                            diff_soft = float(sign) * (rigid_point - soft_point)
                            if wp.dot(diff_soft, direction) > 2.0e-6 * (
                                wp.length(soft_point) + wp.length(rigid_point) + wp.length(particle_q[opposite])
                            ) * (wp.length(diff_soft) + wp.length(direction)):
                                valid_soft = False
                    if not valid_soft:
                        continue
                    if sign != 0:
                        _append_mesh_contact(
                            _MESH_FEATURE_EE + wp.where(sign < 0, 8, 0),
                            edge_index,
                            shape_index,
                            tid,
                            contact_max,
                            contact_count,
                            features,
                            contact_shapes,
                        )


@wp.kernel
def _evaluate_mesh_contacts(
    contact_count: wp.array[wp.int32],
    features: wp.array[wp.vec3i],
    contact_shapes: wp.array[int],
    contact_max: wp.int32,
    shape_type: wp.array[int],
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
    soft_contact_particle: wp.array[wp.int32],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):

    tid = wp.tid()
    if tid >= wp.min(contact_count[0], contact_max):
        return
    shape = contact_shapes[tid]
    if shape_type[shape] != GeoType.MESH and shape_type[shape] != GeoType.CONVEX_MESH:
        return
    feature = features[tid]
    family = feature[0] & 7
    soft_feature = feature[1]
    if feature[0] < 0:
        return
    shape_index = contact_shapes[tid]
    rigid_feature = feature[2]

    X_bs, X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape_index)
    mesh = shape_source_ptr[shape_index]
    scale = shape_scale[shape_index]

    particle = wp.int32(-1)
    corners = wp.vec3i(-1, -1, -1)
    bary = wp.vec3(0.0)
    body_pos = wp.vec3(0.0)
    body_vel = wp.vec3(0.0)
    normal = wp.vec3(0.0)

    if family == _MESH_FEATURE_VT:
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
        det_sign = wp.sign(scale[0] * scale[1] * scale[2])
        tri_n = wp.normalize(wp.cross(b - a, c - a)) * det_sign
        normal = wp.transform_vector(X_ws, _oriented_contact_normal(diff, wp.length(diff), tri_n))
        v_local = wp.cw_mul(wp.mesh_eval_velocity(mesh, face, rigid_bary[0], rigid_bary[1]), scale)
        body_pos = wp.transform_point(X_bs, cp)
        body_vel = wp.transform_vector(X_bs, v_local)
    elif family == _MESH_FEATURE_TV:
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

    soft_point = bary[0] * particle_q[corners[0]]
    if corners[1] >= 0:
        soft_point += bary[1] * particle_q[corners[1]]
    if corners[2] >= 0:
        soft_point += bary[2] * particle_q[corners[2]]
    rigid_point = wp.transform_point(X_ws, wp.transform_point(wp.transform_inverse(X_bs), body_pos))
    diff_world = soft_point - rigid_point
    distance_world = wp.length(diff_world)
    if distance_world > CONTACT_NORMAL_DEGENERATE_EPS:
        sign = wp.where((feature[0] & 8) != 0, -1.0, 1.0)
        normal = sign * diff_world / distance_world

    soft_contact_particle[tid] = particle
    soft_contact_indices[tid] = corners
    soft_contact_barycentric[tid] = bary
    soft_contact_body_pos[tid] = body_pos
    soft_contact_body_vel[tid] = body_vel
    soft_contact_normal[tid] = normal


def launch_soft_mesh_contacts(*, model: Model, state: State, contacts: Contacts, gap: float, data):
    """Append mesh contacts directly to final slots and evaluate their geometry."""
    device = model.device
    vt_pairs = data.vertex_pairs
    rigid_vertex_table, rigid_vertex_normals, rigid_edge_table, rigid_edge_outward_dirs = data.rigid_features
    detector = data.detector
    max_particle_radius = model.particle_max_radius
    if detector is not None:
        detector.vertex_positions = state.particle_q
        detector.refit_triangles()
        if model.edge_count:
            detector.refit_edges()
    contact_count = data.contact_count
    wp.launch(
        _begin_mesh_contacts,
        dim=1,
        inputs=[contacts.soft_contact_count, contact_count],
        device=device,
        record_tape=False,
    )
    if contacts._soft_contact_mesh_features is None:
        contacts._soft_contact_mesh_features = wp.empty(contacts.soft_contact_max, dtype=wp.vec3i, device=device)
    features = contacts._soft_contact_mesh_features
    contact_shapes = contacts.soft_contact_shape
    n_vt = int(vt_pairs.shape[0])
    n_tv = int(rigid_vertex_table.shape[0])
    n_ee = int(rigid_edge_table.shape[0])
    contact_max = int(features.shape[0])
    if n_vt == 0 and n_tv == 0 and n_ee == 0:
        return

    cone_args = data.adjacency

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
        # CUDA shares the nearest query within a warp. CPU queries keep one
        # worker and the same exact traversal, without redundant sign queries.
        partition_depth = _MESH_QUERY_PARTITION_DEPTH if device.is_cuda else 0
        wp.launch(
            _detect_mesh_vertex_contacts,
            dim=n_vt << partition_depth,
            inputs=[
                vt_pairs,
                partition_depth,
                state.particle_q,
                model.particle_radius,
                model.particle_flags,
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_flags,
                model.shape_scale,
                model.shape_source_ptr,
                model.shape_collision_aabb_lower,
                model.shape_collision_aabb_upper,
                model.shape_margin,
                gap,
                contact_max,
                *cone_args,
            ],
            outputs=[contact_count, features, contact_shapes],
            device=device,
            record_tape=False,
            block_dim=64,
        )

    if detector is not None and n_tv > 0:
        wp.launch(
            _detect_mesh_face_contacts,
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
                contact_max,
                *cone_args,
            ],
            outputs=[contact_count, features, contact_shapes],
            device=device,
            record_tape=False,
            block_dim=64,
        )

    if detector is not None and n_ee > 0:
        wp.launch(
            _detect_mesh_edge_contacts,
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
                contact_max,
                *cone_args,
            ],
            outputs=[contact_count, features, contact_shapes],
            device=device,
            record_tape=False,
            block_dim=64,
        )

    wp.launch(
        _end_mesh_contacts,
        dim=1,
        inputs=[contact_count, contacts.soft_contact_count, contact_max],
        device=device,
        record_tape=False,
    )
    if contact_max == 0:
        return
    wp.launch(
        _evaluate_mesh_contacts,
        dim=contact_max,
        inputs=[
            contacts.soft_contact_count,
            features,
            contact_shapes,
            contact_max,
            model.shape_type,
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
        ],
        outputs=[
            contacts.soft_contact_particle,
            contacts.soft_contact_indices,
            contacts.soft_contact_barycentric,
            contacts.soft_contact_body_pos,
            contacts.soft_contact_body_vel,
            contacts.soft_contact_normal,
        ],
        device=device,
    )


class MeshContactData:
    """Fixed mesh topology and acceleration structures; no candidate-contact buffer."""

    def __init__(self, model: Model, shape_mask: np.ndarray, vertex_pairs: wp.array[wp.vec2i]):
        self.vertex_pairs = vertex_pairs
        if model.tri_count:
            self.rigid_features = _build_rigid_features(model, shape_mask)
            if model.edge_count == 0:
                self.rigid_features = (
                    *self.rigid_features[:2],
                    wp.empty(0, dtype=wp.vec3i, device=model.device),
                    wp.empty(0, dtype=wp.vec3, device=model.device),
                )
            self.detector = TriMeshCollisionDetector(model)
        else:
            self.rigid_features = (
                wp.empty(0, dtype=wp.vec2i, device=model.device),
                wp.empty(0, dtype=wp.vec3, device=model.device),
                wp.empty(0, dtype=wp.vec3i, device=model.device),
                wp.empty(0, dtype=wp.vec3, device=model.device),
            )
            self.detector = None
        vertices, vertex_normals, edges, edge_normals = self.rigid_features
        if max(_MESH_QUERY_PARTITIONS * len(vertex_pairs), len(vertices), len(edges)) > np.iinfo(np.int32).max:
            raise ValueError("Mesh contact queries exceed 32-bit indexing capacity.")
        adjacency, component_counts = _build_feature_adjacency(model, vertices, edges)
        self.adjacency = [*adjacency, vertex_normals, edge_normals]
        # A query emits at most one contact per feature pair. Bound the wide
        # append counter before allocating; final writes remain capacity checked.
        max_faces = max(len(model.shape_source[s].indices) // 3 for s in np.flatnonzero(shape_mask))
        bound = len(vertex_pairs) * max_faces + len(vertices) * model.tri_count + len(edges) * model.edge_count
        if bound > np.iinfo(np.int64).max - np.iinfo(np.int32).max:
            raise ValueError("Mesh contact pairs exceed 64-bit counting capacity.")
        self.contact_count = wp.empty(1, dtype=wp.int64, device=model.device)
        # Size final storage primarily from the deformable workload. Retain a
        # rigid-feature floor for fine flat patches touching very coarse cloth.
        # This remains an estimate; every write checks the caller-sized capacity.
        particle_world = model.particle_world.numpy()
        shape_world = model.shape_world.numpy()[shape_mask]
        shape_counts = np.bincount(shape_world + 1, minlength=model.world_count + 1)

        def count_pairs(indices, column):
            if indices is None:
                return 0
            indices = indices.numpy()
            worlds = particle_world[indices[:, column]]
            counts = np.bincount(worlds + 1, minlength=model.world_count + 1)
            return int(
                counts[0] * len(shape_world)
                + (len(worlds) - counts[0]) * shape_counts[0]
                + counts[1:] @ shape_counts[1:]
            )

        surface_pairs = len(vertex_pairs) + count_pairs(model.tri_indices, 0) + count_pairs(model.edge_indices, 2)
        vertex_pair_shapes = vertex_pairs.numpy()[:, 1]
        vertex_patch_hint = int((4 * component_counts[vertex_pair_shapes]).sum(dtype=np.int64))
        self.contact_capacity_hint = max(vertex_patch_hint, surface_pairs, len(vertices) + len(edges))
