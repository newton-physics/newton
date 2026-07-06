# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast, overload
from urllib.parse import urlparse

import numpy as np
import warp as wp

from ..geometry.types import Mesh


@wp.kernel
def accumulate_vertex_normals(
    points: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    # output
    normals: wp.array[wp.vec3],
):
    """Accumulate per-face normals into per-vertex normals (not normalized)."""
    face = wp.tid()
    i0 = indices[face * 3]
    i1 = indices[face * 3 + 1]
    i2 = indices[face * 3 + 2]
    v0 = points[i0]
    v1 = points[i1]
    v2 = points[i2]
    normal = wp.cross(v1 - v0, v2 - v0)
    wp.atomic_add(normals, i0, normal)
    wp.atomic_add(normals, i1, normal)
    wp.atomic_add(normals, i2, normal)


@wp.kernel
def normalize_vertex_normals(normals: wp.array[wp.vec3]):
    """Normalize per-vertex normals in-place."""
    tid = wp.tid()
    normals[tid] = wp.normalize(normals[tid])


@overload
def compute_vertex_normals(
    points: wp.array,
    indices: wp.array | np.ndarray,
    normals: wp.array | None = None,
    *,
    device: wp.DeviceLike = None,
    normalize: bool = True,
) -> wp.array: ...


@overload
def compute_vertex_normals(
    points: np.ndarray,
    indices: np.ndarray,
    normals: np.ndarray | None = None,
    *,
    device: wp.DeviceLike = None,
    normalize: bool = True,
) -> np.ndarray: ...


def compute_vertex_normals(
    points: wp.array | np.ndarray,
    indices: wp.array | np.ndarray,
    normals: wp.array | np.ndarray | None = None,
    *,
    device: wp.DeviceLike = None,
    normalize: bool = True,
) -> wp.array | np.ndarray:
    """Compute per-vertex normals from triangle indices.

    Supports Warp and NumPy arrays. NumPy inputs run on the CPU via Warp and return
    NumPy output.

    Args:
        points: Vertex positions (wp.vec3 array or Nx3 NumPy array).
        indices: Triangle indices (flattened or Nx3). Warp arrays are expected to be flattened.
        normals: Optional output array to reuse (Warp or NumPy to match ``points``).
        device: Warp device to run on. NumPy inputs default to CPU.
        normalize: Whether to normalize the accumulated normals.

    Returns:
        Per-vertex normals as a Warp array or NumPy array matching the input type.
    """
    if isinstance(points, wp.array):
        if normals is not None and not isinstance(normals, wp.array):
            raise TypeError("normals must be a Warp array when points is a Warp array.")
        device_obj = points.device if device is None else wp.get_device(device)
        indices_wp = indices
        if isinstance(indices, np.ndarray):
            indices_np = np.asarray(indices, dtype=np.int32)
            if indices_np.ndim == 2:
                indices_np = indices_np.reshape(-1)
            elif indices_np.ndim != 1:
                raise ValueError("indices must be flat or (N, 3) for NumPy inputs.")
            indices_wp = wp.array(indices_np, dtype=wp.int32, device=device_obj)
        indices_wp = cast(wp.array, indices_wp)
        if normals is None:
            normals_wp = wp.zeros_like(points)
        else:
            normals_wp = cast(wp.array, normals)
            normals_wp.zero_()
        if len(indices_wp) == 0 or len(points) == 0:
            return normals_wp
        indices_i32 = indices_wp if indices_wp.dtype == wp.int32 else indices_wp.view(dtype=wp.int32)
        wp.launch(
            accumulate_vertex_normals,
            dim=len(indices_i32) // 3,
            inputs=[points, indices_i32],
            outputs=[normals_wp],
            device=device_obj,
        )
        if normalize:
            wp.launch(normalize_vertex_normals, dim=len(normals_wp), inputs=[normals_wp], device=device_obj)
        return normals_wp

    points_np = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    indices_np = np.asarray(indices, dtype=np.int32)
    if indices_np.ndim == 2:
        indices_np = indices_np.reshape(-1)
    elif indices_np.ndim != 1:
        raise ValueError("indices must be flat or (N, 3) for NumPy inputs.")

    normals_np = None
    if normals is not None:
        normals_np = np.asarray(normals, dtype=np.float32).reshape(points_np.shape)
    device_obj = wp.get_device("cpu") if device is None else wp.get_device(device)
    points_wp = wp.array(points_np, dtype=wp.vec3, device=device_obj)
    indices_wp = wp.array(indices_np, dtype=wp.int32, device=device_obj)
    if normals_np is None:
        normals_wp = wp.zeros_like(points_wp)
    else:
        normals_wp = wp.array(normals_np, dtype=wp.vec3, device=device_obj)
        normals_wp.zero_()
    if len(points_wp) == 0 or len(indices_wp) == 0:
        if normals_np is None:
            return np.zeros_like(points_np, dtype=np.float32)
        normals_np[...] = 0.0
        return normals_np
    wp.launch(
        accumulate_vertex_normals,
        dim=len(indices_wp) // 3,
        inputs=[points_wp, indices_wp],
        outputs=[normals_wp],
        device=device_obj,
    )
    if normalize:
        wp.launch(normalize_vertex_normals, dim=len(normals_wp), inputs=[normals_wp], device=device_obj)
    normals_out = normals_wp.numpy()
    if normals_np is not None:
        normals_np[...] = normals_out
        return normals_np
    return normals_out


def smooth_vertex_normals_by_position(
    mesh_vertices: np.ndarray, mesh_faces: np.ndarray, eps: float = 1.0e-6
) -> np.ndarray:
    """Smooth vertex normals by averaging normals of vertices with shared positions."""
    normals = compute_vertex_normals(mesh_vertices, mesh_faces)
    if len(mesh_vertices) == 0:
        return normals
    keys = np.round(mesh_vertices / eps).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    accum = np.zeros((len(unique_keys), 3), dtype=np.float32)
    np.add.at(accum, inverse, normals)
    lengths = np.linalg.norm(accum, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1.0e-8)
    accum = accum / lengths
    return accum[inverse]


# Default number of segments for mesh generation
default_num_segments = 32


@wp.struct
class MeshAdjacencyData:
    """Kernel-facing soft-mesh adjacency -- pure data, no Python state or methods.

    Uploaded from :class:`MeshAdjacency` via :meth:`MeshAdjacency.to`: the vertex-adjacency
    CSR arrays (read by the VBD solver's kernels) plus the ``edge_tri_indices`` /
    ``tri_edge_indices`` topology maps, so kernels can read mesh topology directly on device.
    """

    edge_tri_indices: wp.array2d[wp.int32]
    tri_edge_indices: wp.array2d[wp.int32]
    v_adj_tris: wp.array[wp.int32]
    v_adj_tris_offsets: wp.array[wp.int32]
    v_adj_edges: wp.array[wp.int32]
    v_adj_edges_offsets: wp.array[wp.int32]
    v_adj_springs: wp.array[wp.int32]
    v_adj_springs_offsets: wp.array[wp.int32]
    v_adj_tets: wp.array[wp.int32]
    v_adj_tets_offsets: wp.array[wp.int32]


@wp.func
def get_vertex_num_adjacent_edges(adjacency: MeshAdjacencyData, vertex: wp.int32):
    return (adjacency.v_adj_edges_offsets[vertex + 1] - adjacency.v_adj_edges_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_edge_id_order(adjacency: MeshAdjacencyData, vertex: wp.int32, edge: wp.int32):
    offset = adjacency.v_adj_edges_offsets[vertex]
    return adjacency.v_adj_edges[offset + edge * 2], adjacency.v_adj_edges[offset + edge * 2 + 1]


@wp.func
def get_vertex_num_adjacent_faces(adjacency: MeshAdjacencyData, vertex: wp.int32):
    return (adjacency.v_adj_tris_offsets[vertex + 1] - adjacency.v_adj_tris_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_face_id_order(adjacency: MeshAdjacencyData, vertex: wp.int32, face: wp.int32):
    offset = adjacency.v_adj_tris_offsets[vertex]
    return adjacency.v_adj_tris[offset + face * 2], adjacency.v_adj_tris[offset + face * 2 + 1]


@wp.func
def get_vertex_num_adjacent_springs(adjacency: MeshAdjacencyData, vertex: wp.int32):
    return adjacency.v_adj_springs_offsets[vertex + 1] - adjacency.v_adj_springs_offsets[vertex]


@wp.func
def get_vertex_adjacent_spring_id(adjacency: MeshAdjacencyData, vertex: wp.int32, spring: wp.int32):
    offset = adjacency.v_adj_springs_offsets[vertex]
    return adjacency.v_adj_springs[offset + spring]


@wp.func
def get_vertex_num_adjacent_tets(adjacency: MeshAdjacencyData, vertex: wp.int32):
    return (adjacency.v_adj_tets_offsets[vertex + 1] - adjacency.v_adj_tets_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_tet_id_order(adjacency: MeshAdjacencyData, vertex: wp.int32, tet: wp.int32):
    offset = adjacency.v_adj_tets_offsets[vertex]
    return adjacency.v_adj_tets[offset + tet * 2], adjacency.v_adj_tets[offset + tet * 2 + 1]


class MeshAdjacency:
    """Connectivity tables of a soft (triangle) mesh, consumed by collision and the solvers.

    Three kinds of topology are stored, all derived from a triangle mesh and its bending
    edges. The edge/triangle tables are dense arrays indexed by element and are computed
    eagerly in the constructor (independent of the vertex count). The vertex-adjacency CSR
    tables are built on demand by :meth:`init_vertex_adjacency` (which needs only the particle
    count -- the element topology is stored on the object). Every table is a host NumPy array;
    :meth:`to` copies the vertex-adjacency CSR into a
    :class:`MeshAdjacencyData` struct for use inside Warp kernels.

    Obtain an instance by construction — ``MeshAdjacency(tri_indices)`` — or from a finalized
    model via :attr:`newton.Model.soft_mesh_adjacency`.

    Attributes:
        edge_indices: Undirected edges as ``[o0, o1, v0, v1]`` rows, ``int32`` shape
            ``[edge_count, 4]``: the edge connects ``v0``-``v1``, and ``o0``/``o1`` are the
            opposite vertices of its two adjacent triangles (``o1 == -1`` on a boundary edge).
        edge_tri_indices: The two triangles sharing each edge as ``[f0, f1]`` rows, ``int32``
            shape ``[edge_count, 2]`` (``f1 == -1`` on a boundary edge); ``f0`` is the
            triangle whose opposite vertex is ``o0``.
        tri_edge_indices: Each triangle's three edges, ``int32`` shape ``[tri_count, 3]``;
            column ``k`` is the edge between the triangle's local vertices ``k`` and
            ``(k + 1) % 3``, or ``-1`` if that edge was never registered (e.g. a bare triangle).
        v_adj_edges, v_adj_tris, v_adj_springs, v_adj_tets: Per-vertex CSR adjacency to
            bending edges / triangles / springs / tetrahedra, each paired with its matching
            ``*_offsets``. ``*_offsets`` is ``int32`` of length ``particle_count + 1``, and
            vertex ``i``'s records occupy ``v_adj_X[offsets[i] : offsets[i + 1]]``; a record
            is an ``(element_id, local_slot)`` pair (a single ``element_id`` for springs). All
            eight arrays are ``None`` until :meth:`init_vertex_adjacency` runs.
        vertex_adjacency_initialized: Whether the vertex-adjacency CSR has been built;
            :meth:`init_vertex_adjacency` returns early when this is already ``True``.
        indices, spring_indices, tet_indices: The triangle / spring / tetrahedron topology this
            adjacency is built over, kept from the constructor for :meth:`init_vertex_adjacency`.

    .. note::
        The :attr:`edges` dict is a deprecated compatibility shim (it emits a
        ``DeprecationWarning``); use the ``edge_indices`` / ``edge_tri_indices`` arrays instead.
    """

    @dataclass(slots=True)
    class Edge:
        """Legacy per-edge record: edge ``(v0, v1)`` with opposite vertices
        ``o0``/``o1`` and adjacent triangles ``f0``/``f1`` (``-1`` if boundary)."""

        v0: int
        v1: int
        o0: int
        o1: int
        f0: int
        f1: int

    def __init__(
        self,
        tri_indices: Sequence[Sequence[int]] | np.ndarray | None = None,
        edge_indices: Sequence[Sequence[int]] | np.ndarray | None = None,
        spring_indices: Sequence[int] | np.ndarray | None = None,
        tet_indices: Sequence[Sequence[int]] | np.ndarray | None = None,
        indices: Sequence[Sequence[int]] | np.ndarray | None = None,
    ):
        """Build edge adjacency from triangles and store the element topology as members.

        Args:
            tri_indices: Triangle indices, shape ``[tri_count, 3]``, used to derive the
                edge/triangle maps. ``None`` leaves the edge tables empty.
            edge_indices: Pre-numbered bending edges, shape ``[edge_count, 4]`` as
                ``[o0, o1, v0, v1]``. When given, this exact edge numbering is kept
                (so it stays aligned with externally stored bending materials) and
                only the edge/triangle maps are derived against ``tri_indices``.
                When ``None``, ``edge_indices`` is computed from ``tri_indices``.
            spring_indices: Spring endpoint pairs, flat shape ``[2 * spring_count]``;
                stored for :meth:`init_vertex_adjacency`.
            tet_indices: Tetrahedron vertex ids, shape ``[tet_count, 4]``; stored for
                :meth:`init_vertex_adjacency`.
            indices: Deprecated alias for ``tri_indices``.
        """
        # `indices` is a deprecated alias for `tri_indices`, kept for backward compatibility.
        if indices is not None:
            if tri_indices is not None and not np.array_equal(_numpy_int_array(indices), _numpy_int_array(tri_indices)):
                raise ValueError("Pass `tri_indices` or the deprecated `indices`, not both with different values.")
            warnings.warn(
                "MeshAdjacency `indices` argument is deprecated; use `tri_indices`.",
                DeprecationWarning,
                stacklevel=2,
            )
            tri_indices = indices

        # Element topology kept as members (owned int32 copies, detached from any mutable input
        # list so a finalized model's adjacency can't drift if the builder is modified after
        # finalize()); init_vertex_adjacency builds the CSR from these.
        self.indices = None if tri_indices is None else np.array(tri_indices, dtype=np.int32)
        self.spring_indices = None if spring_indices is None else np.array(spring_indices, dtype=np.int32)
        self.tet_indices = None if tet_indices is None else np.array(tet_indices, dtype=np.int32)

        if edge_indices is not None:
            # Keep the caller's edge numbering; derive only the maps from the triangles.
            self.edge_indices = _numpy_int_rows(edge_indices, 4)
            tris = _numpy_int_rows(tri_indices, 3) if tri_indices is not None else np.empty((0, 3), dtype=np.int32)
            self.edge_tri_indices, self.tri_edge_indices = self._build_maps(self.edge_indices, tris)
        elif tri_indices is not None:
            self.edge_indices, self.edge_tri_indices, self.tri_edge_indices = self._compute_edge_adjacency(tri_indices)
        else:
            self.edge_indices = np.empty((0, 4), dtype=np.int32)
            self.edge_tri_indices = np.empty((0, 2), dtype=np.int32)
            self.tri_edge_indices = np.empty((0, 3), dtype=np.int32)

        # Vertex-to-element CSR tables; built explicitly by init_vertex_adjacency.
        self.v_adj_tris = None
        self.v_adj_tris_offsets = None
        self.v_adj_edges = None
        self.v_adj_edges_offsets = None
        self.v_adj_springs = None
        self.v_adj_springs_offsets = None
        self.v_adj_tets = None
        self.v_adj_tets_offsets = None
        # Set once init_vertex_adjacency has built the CSR tables; guards recomputation.
        self.vertex_adjacency_initialized = False

    @property
    def edges(self) -> dict[tuple[int, int], "MeshAdjacency.Edge"]:
        """Deprecated legacy edge dict, rebuilt on access from ``edge_indices``.

        Maps ``(min(v0, v1), max(v0, v1))`` to an :class:`Edge`. Recomputed on
        every access and never cached; prefer the ``edge_indices`` /
        ``edge_tri_indices`` arrays directly.
        """
        warnings.warn(
            "MeshAdjacency.edges is deprecated; use the edge_indices/edge_tri_indices arrays.",
            DeprecationWarning,
            stacklevel=2,
        )
        edge_indices = _numpy_int_rows(self.edge_indices, 4)
        edge_tri_indices = _numpy_int_rows(self.edge_tri_indices, 2)
        return {
            (min(int(v0), int(v1)), max(int(v0), int(v1))): MeshAdjacency.Edge(
                int(v0), int(v1), int(o0), int(o1), int(f0), int(f1)
            )
            for (o0, o1, v0, v1), (f0, f1) in zip(edge_indices, edge_tri_indices, strict=True)
        }

    def add_edge(self, i0: int, i1: int, o: int, f: int) -> None:
        """Add or update one edge (deprecated; build via ``edge_indices`` instead).

        Legacy incremental API: edge ``(i0, i1)`` with opposite vertex ``o`` in triangle ``f``.
        The first call for an edge fills ``o0``/``f0``, the second fills ``o1``/``f1``; a third
        warns (non-manifold). Updates :attr:`edge_indices` / :attr:`edge_tri_indices` (so
        :attr:`edges` reflects it) and invalidates the vertex-adjacency CSR. It does **not**
        update :attr:`tri_edge_indices`, so an edge added this way will not appear in the
        per-triangle edge map; users can reconstruct via the constructor if they need that. O(edge_count)
        per call -- a compatibility shim, not a hot path.

        Args:
            i0: First edge endpoint.
            i1: Second edge endpoint.
            o: Opposite vertex in triangle ``f``.
            f: Triangle containing this edge.
        """
        warnings.warn(
            "MeshAdjacency.add_edge is deprecated; construct with edge_indices ([o0, o1, v0, v1] rows) instead. "
            "The added edge is not reflected in tri_edge_indices.",
            DeprecationWarning,
            stacklevel=2,
        )
        edge_rows = _numpy_int_rows(self.edge_indices, 4)
        tri_rows = _numpy_int_rows(self.edge_tri_indices, 2)
        lo, hi = (i0, i1) if i0 <= i1 else (i1, i0)
        match = -1
        for e in range(edge_rows.shape[0]):
            if (
                min(int(edge_rows[e, 2]), int(edge_rows[e, 3])) == lo
                and max(int(edge_rows[e, 2]), int(edge_rows[e, 3])) == hi
            ):
                match = e
                break
        if match == -1:
            self.edge_indices = np.concatenate((edge_rows, np.array([[o, -1, i0, i1]], dtype=np.int32)))
            self.edge_tri_indices = np.concatenate((tri_rows, np.array([[f, -1]], dtype=np.int32)))
        elif int(tri_rows[match, 1]) == -1:
            edge_rows[match, 1] = o
            tri_rows[match, 1] = f
            self.edge_indices, self.edge_tri_indices = edge_rows, tri_rows
        else:
            warnings.warn("Detected non-manifold edge", stacklevel=2)
            return
        self.vertex_adjacency_initialized = False

    def to(self, device) -> MeshAdjacencyData:
        """Upload the device-facing adjacency arrays onto ``device`` as a pure data struct.

        Always uploads the edge/triangle topology maps. The vertex-adjacency CSR is uploaded
        only when :meth:`init_vertex_adjacency` has populated it; otherwise the eight
        ``v_adj_*`` fields are left ``None`` and a warning is emitted (the struct is still
        usable for callers that only need the topology maps). This is the only place the host
        NumPy tables become Warp arrays.
        """
        device = wp.get_device(device)
        data = MeshAdjacencyData()
        data.edge_tri_indices = wp.array(self.edge_tri_indices, dtype=wp.int32, device=device)
        data.tri_edge_indices = wp.array(self.tri_edge_indices, dtype=wp.int32, device=device)
        if not self.vertex_adjacency_initialized:
            warnings.warn(
                "MeshAdjacency.to(): vertex adjacency not initialized; v_adj_* are None -- call "
                "init_vertex_adjacency(particle_count) first if your kernels need them.",
                stacklevel=2,
            )
            return data
        data.v_adj_tris = wp.array(self.v_adj_tris, dtype=wp.int32, device=device)
        data.v_adj_tris_offsets = wp.array(self.v_adj_tris_offsets, dtype=wp.int32, device=device)
        data.v_adj_edges = wp.array(self.v_adj_edges, dtype=wp.int32, device=device)
        data.v_adj_edges_offsets = wp.array(self.v_adj_edges_offsets, dtype=wp.int32, device=device)
        data.v_adj_springs = wp.array(self.v_adj_springs, dtype=wp.int32, device=device)
        data.v_adj_springs_offsets = wp.array(self.v_adj_springs_offsets, dtype=wp.int32, device=device)
        data.v_adj_tets = wp.array(self.v_adj_tets, dtype=wp.int32, device=device)
        data.v_adj_tets_offsets = wp.array(self.v_adj_tets_offsets, dtype=wp.int32, device=device)
        return data

    @staticmethod
    def _compute_edge_adjacency(
        indices: Sequence[Sequence[int]] | np.ndarray,
        *,
        tri_start: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the edge / edge-triangle / triangle-edge tables from triangles.

        Returns ``(edge_indices, edge_tri_indices, tri_edge_indices)``.
        ``edge_indices`` rows are ``[o0, o1, v0, v1]`` (opposite vertices, edge
        endpoints); ``edge_tri_indices`` rows are ``[tri0, tri1]`` (``-1`` for a
        boundary side, triangle ids offset by ``tri_start``); ``tri_edge_indices``
        maps each triangle's three local edge slots to edge rows.
        """
        tris = np.asarray(indices, dtype=np.int32).reshape(-1, 3)
        tri_count = tris.shape[0]
        if tri_count == 0:
            return (
                np.empty((0, 4), dtype=np.int32),
                np.empty((0, 2), dtype=np.int32),
                np.empty((0, 3), dtype=np.int32),
            )

        # Local edge slots are: (v0, v1 | opposite v2), (v1, v2 | opposite v0),
        # (v2, v0 | opposite v1).
        entry_v0 = np.stack((tris[:, 0], tris[:, 1], tris[:, 2]), axis=1).reshape(-1)
        entry_v1 = np.stack((tris[:, 1], tris[:, 2], tris[:, 0]), axis=1).reshape(-1)
        entry_opposite = np.stack((tris[:, 2], tris[:, 0], tris[:, 1]), axis=1).reshape(-1)
        entry_tri = np.repeat(np.arange(tri_count, dtype=np.int32), 3)
        entry_slot = np.tile(np.arange(3, dtype=np.int32), tri_count)

        keys = np.stack((np.minimum(entry_v0, entry_v1), np.maximum(entry_v0, entry_v1)), axis=1)
        _, first_entries, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)

        # np.unique returns keys sorted lexicographically. Remap to first-occurrence
        # order so edge rows follow triangle traversal order.
        first_order = np.argsort(first_entries, kind="stable")
        edge_remap = np.empty(len(first_order), dtype=np.int32)
        edge_remap[first_order] = np.arange(len(first_order), dtype=np.int32)
        entry_edge = edge_remap[inverse]

        edge_count = len(first_order)
        edge_indices = np.full((edge_count, 4), -1, dtype=np.int32)
        edge_tri_indices = np.full((edge_count, 2), -1, dtype=np.int32)
        tri_edge_indices = np.full((tri_count, 3), -1, dtype=np.int32)

        fill_counts = np.zeros(edge_count, dtype=np.int32)
        for entry_id, edge_ref in enumerate(entry_edge):
            edge_id = int(edge_ref)
            tri_edge_indices[entry_tri[entry_id], entry_slot[entry_id]] = edge_id

            side = fill_counts[edge_id]
            if side == 0:
                edge_indices[edge_id, 0] = entry_opposite[entry_id]
                edge_indices[edge_id, 2] = entry_v0[entry_id]
                edge_indices[edge_id, 3] = entry_v1[entry_id]
                edge_tri_indices[edge_id, 0] = tri_start + entry_tri[entry_id]
            elif side == 1:
                edge_indices[edge_id, 1] = entry_opposite[entry_id]
                edge_tri_indices[edge_id, 1] = tri_start + entry_tri[entry_id]
            else:
                warnings.warn("Detected non-manifold edge", stacklevel=2)

            fill_counts[edge_id] += 1

        return edge_indices, edge_tri_indices, tri_edge_indices

    @staticmethod
    def _build_maps(edge_indices: np.ndarray, tri_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Derive the edge-triangle and triangle-edge maps from pre-numbered edges.

        Returns ``(edge_tri_indices, tri_edge_indices)`` sized to ``edge_indices`` /
        ``tri_indices``. A triangle is linked to an edge only when the triangle's
        vertex opposite that edge equals one of the edge's stored opposite vertices
        (``o0``/``o1``); placeholder edges (``o0 == o1 == -1``) and bare triangles
        therefore stay unlinked, as if the edge were never registered for them.
        """
        edge_count = edge_indices.shape[0]
        tri_count = tri_indices.shape[0]
        edge_tri_indices = np.full((edge_count, 2), -1, dtype=np.int32)
        tri_edge_indices = np.full((tri_count, 3), -1, dtype=np.int32)
        if edge_count == 0 or tri_count == 0:
            return edge_tri_indices, tri_edge_indices

        edges = edge_indices.astype(np.int64, copy=False)
        tris = tri_indices.astype(np.int64, copy=False)

        # Encode each undirected vertex pair as one key (min * stride + max) so edges and
        # triangle sides can be matched with a single vectorized searchsorted.
        stride = max(int(edges[:, 2:4].max()), int(tris.max())) + 1
        edge_key = np.minimum(edges[:, 2], edges[:, 3]) * stride + np.maximum(edges[:, 2], edges[:, 3])

        # Triangle local slots: slot k spans verts (k, k+1); the opposite vertex is (k+2).
        nxt = np.roll(tris, -1, axis=1)
        slot_key = np.minimum(tris, nxt) * stride + np.maximum(tris, nxt)  # (tri_count, 3)
        slot_opp = np.roll(tris, -2, axis=1)  # (tri_count, 3)

        # Map each slot key to its edge id; -1 where the pair is not a registered edge.
        order = np.argsort(edge_key, kind="stable")
        pos = np.clip(np.searchsorted(edge_key[order], slot_key), 0, edge_count - 1)
        cand = order[pos]  # (tri_count, 3)
        found = edge_key[cand] == slot_key

        # Link a slot to its candidate only if the opposite vertex matches a stored
        # opposite (o0 -> side 0, o1 -> side 1), mirroring _compute_edge_adjacency.
        side0 = found & (slot_opp == edges[cand, 0])
        side1 = found & ~side0 & (slot_opp == edges[cand, 1])

        tri_edge_indices = np.where(side0 | side1, cand, -1).astype(np.int32)
        tri_ids = np.broadcast_to(np.arange(tri_count, dtype=np.int32)[:, None], (tri_count, 3))
        edge_tri_indices[cand[side0], 0] = tri_ids[side0]
        edge_tri_indices[cand[side1], 1] = tri_ids[side1]
        return edge_tri_indices, tri_edge_indices

    @staticmethod
    def compute_vertex_adjacency(
        particle_count: int,
        *,
        edge_indices: "wp.array[wp.int32] | np.ndarray | None" = None,
        tri_indices: "wp.array[wp.int32] | np.ndarray | None" = None,
        spring_indices: "wp.array[wp.int32] | np.ndarray | None" = None,
        tet_indices: "wp.array[wp.int32] | np.ndarray | None" = None,
    ) -> "MeshAdjacency":
        """Build a temporary adjacency over the given elements and its vertex-to-element CSR."""
        return MeshAdjacency(
            tri_indices=tri_indices,
            edge_indices=edge_indices,
            spring_indices=spring_indices,
            tet_indices=tet_indices,
        ).init_vertex_adjacency(particle_count)

    def init_vertex_adjacency(self, particle_count: int) -> "MeshAdjacency":
        """Compute and store the vertex-to-element CSR tables from the stored topology.

        Builds per-vertex adjacency to the bending edges, triangles, springs and tetrahedra held
        on this object (:attr:`edge_indices`, :attr:`indices`, :attr:`spring_indices`,
        :attr:`tet_indices`). Idempotent: returns early once built. The count/fill Warp kernels
        run on the host; :meth:`to` later uploads the result to a device.

        Args:
            particle_count: Number of particles; sizes the per-vertex CSR offsets.
        """
        if self.vertex_adjacency_initialized:
            return self

        if _has_entries(self.edge_indices):
            self.v_adj_edges, self.v_adj_edges_offsets = _build_vertex_adjacency_with_warp(
                _as_cpu_int_array2d(self.edge_indices, 4),
                particle_count,
                count_kernel=_count_num_adjacent_edges,
                fill_kernel=_fill_adjacent_edges,
                values_per_entry=2,
            )
        else:
            self.v_adj_edges, self.v_adj_edges_offsets = _empty_vertex_adjacency()

        if _has_entries(self.indices):
            self.v_adj_tris, self.v_adj_tris_offsets = _build_vertex_adjacency_with_warp(
                _as_cpu_int_array2d(self.indices, 3),
                particle_count,
                count_kernel=_count_num_adjacent_tris,
                fill_kernel=_fill_adjacent_tris,
                values_per_entry=2,
            )
        else:
            self.v_adj_tris, self.v_adj_tris_offsets = _empty_vertex_adjacency()

        if _has_entries(self.tet_indices):
            self.v_adj_tets, self.v_adj_tets_offsets = _build_vertex_adjacency_with_warp(
                _as_cpu_int_array2d(self.tet_indices, 4),
                particle_count,
                count_kernel=_count_num_adjacent_tets,
                fill_kernel=_fill_adjacent_tets,
                values_per_entry=2,
            )
        else:
            self.v_adj_tets, self.v_adj_tets_offsets = _empty_vertex_adjacency()

        if _has_entries(self.spring_indices):
            self.v_adj_springs, self.v_adj_springs_offsets = _build_vertex_adjacency_with_warp(
                _as_cpu_int_array1d(self.spring_indices),
                particle_count,
                count_kernel=_count_num_adjacent_springs,
                fill_kernel=_fill_adjacent_springs,
                values_per_entry=1,
            )
        else:
            self.v_adj_springs, self.v_adj_springs_offsets = _empty_vertex_adjacency()

        self.vertex_adjacency_initialized = True
        return self


def _numpy_int_array(data) -> np.ndarray:
    """Return ``data`` as an int32 NumPy array, accepting Warp arrays."""
    if data is None:
        return np.empty(0, dtype=np.int32)
    if hasattr(data, "numpy"):
        data = data.numpy()
    return np.asarray(data, dtype=np.int32)


def _numpy_int_rows(data, width: int) -> np.ndarray:
    """Return ``data`` as ``(-1, width)`` int32 rows."""
    data_np = _numpy_int_array(data)
    if data_np.size == 0:
        return np.empty((0, width), dtype=np.int32)
    return data_np.reshape(-1, width)


@wp.kernel
def _count_num_adjacent_edges(edge_indices: wp.array2d[wp.int32], num_vertex_adjacent_edges: wp.array[wp.int32]):
    for edge_id in range(edge_indices.shape[0]):
        o0 = edge_indices[edge_id, 0]
        o1 = edge_indices[edge_id, 1]
        v0 = edge_indices[edge_id, 2]
        v1 = edge_indices[edge_id, 3]

        num_vertex_adjacent_edges[v0] = num_vertex_adjacent_edges[v0] + 1
        num_vertex_adjacent_edges[v1] = num_vertex_adjacent_edges[v1] + 1

        if o0 != -1:
            num_vertex_adjacent_edges[o0] = num_vertex_adjacent_edges[o0] + 1
        if o1 != -1:
            num_vertex_adjacent_edges[o1] = num_vertex_adjacent_edges[o1] + 1


@wp.kernel
def _fill_adjacent_edges(
    edge_indices: wp.array2d[wp.int32],
    vertex_adjacent_edges_offsets: wp.array[wp.int32],
    vertex_adjacent_edges_fill_count: wp.array[wp.int32],
    vertex_adjacent_edges: wp.array[wp.int32],
):
    for edge_id in range(edge_indices.shape[0]):
        v0 = edge_indices[edge_id, 2]
        v1 = edge_indices[edge_id, 3]

        fill_count_v0 = vertex_adjacent_edges_fill_count[v0]
        buffer_offset_v0 = vertex_adjacent_edges_offsets[v0]
        vertex_adjacent_edges[buffer_offset_v0 + fill_count_v0 * 2] = edge_id
        vertex_adjacent_edges[buffer_offset_v0 + fill_count_v0 * 2 + 1] = 2
        vertex_adjacent_edges_fill_count[v0] = fill_count_v0 + 1

        fill_count_v1 = vertex_adjacent_edges_fill_count[v1]
        buffer_offset_v1 = vertex_adjacent_edges_offsets[v1]
        vertex_adjacent_edges[buffer_offset_v1 + fill_count_v1 * 2] = edge_id
        vertex_adjacent_edges[buffer_offset_v1 + fill_count_v1 * 2 + 1] = 3
        vertex_adjacent_edges_fill_count[v1] = fill_count_v1 + 1

        o0 = edge_indices[edge_id, 0]
        if o0 != -1:
            fill_count_o0 = vertex_adjacent_edges_fill_count[o0]
            buffer_offset_o0 = vertex_adjacent_edges_offsets[o0]
            vertex_adjacent_edges[buffer_offset_o0 + fill_count_o0 * 2] = edge_id
            vertex_adjacent_edges[buffer_offset_o0 + fill_count_o0 * 2 + 1] = 0
            vertex_adjacent_edges_fill_count[o0] = fill_count_o0 + 1

        o1 = edge_indices[edge_id, 1]
        if o1 != -1:
            fill_count_o1 = vertex_adjacent_edges_fill_count[o1]
            buffer_offset_o1 = vertex_adjacent_edges_offsets[o1]
            vertex_adjacent_edges[buffer_offset_o1 + fill_count_o1 * 2] = edge_id
            vertex_adjacent_edges[buffer_offset_o1 + fill_count_o1 * 2 + 1] = 1
            vertex_adjacent_edges_fill_count[o1] = fill_count_o1 + 1


@wp.kernel
def _count_num_adjacent_tris(tri_indices: wp.array2d[wp.int32], num_vertex_adjacent_tris: wp.array[wp.int32]):
    for tri_id in range(tri_indices.shape[0]):
        v0 = tri_indices[tri_id, 0]
        v1 = tri_indices[tri_id, 1]
        v2 = tri_indices[tri_id, 2]

        num_vertex_adjacent_tris[v0] = num_vertex_adjacent_tris[v0] + 1
        num_vertex_adjacent_tris[v1] = num_vertex_adjacent_tris[v1] + 1
        num_vertex_adjacent_tris[v2] = num_vertex_adjacent_tris[v2] + 1


@wp.kernel
def _fill_adjacent_tris(
    tri_indices: wp.array2d[wp.int32],
    vertex_adjacent_tris_offsets: wp.array[wp.int32],
    vertex_adjacent_tris_fill_count: wp.array[wp.int32],
    vertex_adjacent_tris: wp.array[wp.int32],
):
    for tri_id in range(tri_indices.shape[0]):
        v0 = tri_indices[tri_id, 0]
        v1 = tri_indices[tri_id, 1]
        v2 = tri_indices[tri_id, 2]

        fill_count_v0 = vertex_adjacent_tris_fill_count[v0]
        buffer_offset_v0 = vertex_adjacent_tris_offsets[v0]
        vertex_adjacent_tris[buffer_offset_v0 + fill_count_v0 * 2] = tri_id
        vertex_adjacent_tris[buffer_offset_v0 + fill_count_v0 * 2 + 1] = 0
        vertex_adjacent_tris_fill_count[v0] = fill_count_v0 + 1

        fill_count_v1 = vertex_adjacent_tris_fill_count[v1]
        buffer_offset_v1 = vertex_adjacent_tris_offsets[v1]
        vertex_adjacent_tris[buffer_offset_v1 + fill_count_v1 * 2] = tri_id
        vertex_adjacent_tris[buffer_offset_v1 + fill_count_v1 * 2 + 1] = 1
        vertex_adjacent_tris_fill_count[v1] = fill_count_v1 + 1

        fill_count_v2 = vertex_adjacent_tris_fill_count[v2]
        buffer_offset_v2 = vertex_adjacent_tris_offsets[v2]
        vertex_adjacent_tris[buffer_offset_v2 + fill_count_v2 * 2] = tri_id
        vertex_adjacent_tris[buffer_offset_v2 + fill_count_v2 * 2 + 1] = 2
        vertex_adjacent_tris_fill_count[v2] = fill_count_v2 + 1


@wp.kernel
def _count_num_adjacent_springs(spring_indices: wp.array[wp.int32], num_vertex_adjacent_springs: wp.array[wp.int32]):
    num_springs = spring_indices.shape[0] // 2
    for spring_id in range(num_springs):
        v0 = spring_indices[spring_id * 2]
        v1 = spring_indices[spring_id * 2 + 1]

        num_vertex_adjacent_springs[v0] = num_vertex_adjacent_springs[v0] + 1
        num_vertex_adjacent_springs[v1] = num_vertex_adjacent_springs[v1] + 1


@wp.kernel
def _fill_adjacent_springs(
    spring_indices: wp.array[wp.int32],
    vertex_adjacent_springs_offsets: wp.array[wp.int32],
    vertex_adjacent_springs_fill_count: wp.array[wp.int32],
    vertex_adjacent_springs: wp.array[wp.int32],
):
    num_springs = spring_indices.shape[0] // 2
    for spring_id in range(num_springs):
        v0 = spring_indices[spring_id * 2]
        v1 = spring_indices[spring_id * 2 + 1]

        fill_count_v0 = vertex_adjacent_springs_fill_count[v0]
        buffer_offset_v0 = vertex_adjacent_springs_offsets[v0]
        vertex_adjacent_springs[buffer_offset_v0 + fill_count_v0] = spring_id
        vertex_adjacent_springs_fill_count[v0] = fill_count_v0 + 1

        fill_count_v1 = vertex_adjacent_springs_fill_count[v1]
        buffer_offset_v1 = vertex_adjacent_springs_offsets[v1]
        vertex_adjacent_springs[buffer_offset_v1 + fill_count_v1] = spring_id
        vertex_adjacent_springs_fill_count[v1] = fill_count_v1 + 1


@wp.kernel
def _count_num_adjacent_tets(tet_indices: wp.array2d[wp.int32], num_vertex_adjacent_tets: wp.array[wp.int32]):
    for tet_id in range(tet_indices.shape[0]):
        v0 = tet_indices[tet_id, 0]
        v1 = tet_indices[tet_id, 1]
        v2 = tet_indices[tet_id, 2]
        v3 = tet_indices[tet_id, 3]

        num_vertex_adjacent_tets[v0] = num_vertex_adjacent_tets[v0] + 1
        num_vertex_adjacent_tets[v1] = num_vertex_adjacent_tets[v1] + 1
        num_vertex_adjacent_tets[v2] = num_vertex_adjacent_tets[v2] + 1
        num_vertex_adjacent_tets[v3] = num_vertex_adjacent_tets[v3] + 1


@wp.kernel
def _fill_adjacent_tets(
    tet_indices: wp.array2d[wp.int32],
    vertex_adjacent_tets_offsets: wp.array[wp.int32],
    vertex_adjacent_tets_fill_count: wp.array[wp.int32],
    vertex_adjacent_tets: wp.array[wp.int32],
):
    for tet_id in range(tet_indices.shape[0]):
        v0 = tet_indices[tet_id, 0]
        v1 = tet_indices[tet_id, 1]
        v2 = tet_indices[tet_id, 2]
        v3 = tet_indices[tet_id, 3]

        fill_count_v0 = vertex_adjacent_tets_fill_count[v0]
        buffer_offset_v0 = vertex_adjacent_tets_offsets[v0]
        vertex_adjacent_tets[buffer_offset_v0 + fill_count_v0 * 2] = tet_id
        vertex_adjacent_tets[buffer_offset_v0 + fill_count_v0 * 2 + 1] = 0
        vertex_adjacent_tets_fill_count[v0] = fill_count_v0 + 1

        fill_count_v1 = vertex_adjacent_tets_fill_count[v1]
        buffer_offset_v1 = vertex_adjacent_tets_offsets[v1]
        vertex_adjacent_tets[buffer_offset_v1 + fill_count_v1 * 2] = tet_id
        vertex_adjacent_tets[buffer_offset_v1 + fill_count_v1 * 2 + 1] = 1
        vertex_adjacent_tets_fill_count[v1] = fill_count_v1 + 1

        fill_count_v2 = vertex_adjacent_tets_fill_count[v2]
        buffer_offset_v2 = vertex_adjacent_tets_offsets[v2]
        vertex_adjacent_tets[buffer_offset_v2 + fill_count_v2 * 2] = tet_id
        vertex_adjacent_tets[buffer_offset_v2 + fill_count_v2 * 2 + 1] = 2
        vertex_adjacent_tets_fill_count[v2] = fill_count_v2 + 1

        fill_count_v3 = vertex_adjacent_tets_fill_count[v3]
        buffer_offset_v3 = vertex_adjacent_tets_offsets[v3]
        vertex_adjacent_tets[buffer_offset_v3 + fill_count_v3 * 2] = tet_id
        vertex_adjacent_tets[buffer_offset_v3 + fill_count_v3 * 2 + 1] = 3
        vertex_adjacent_tets_fill_count[v3] = fill_count_v3 + 1


def _has_entries(data) -> bool:
    """Return whether a topology array/list has at least one stored entry."""
    if data is None:
        return False
    if isinstance(data, wp.array):
        return data.size > 0
    return np.asarray(data).size > 0


def _as_cpu_int_array2d(data, width: int) -> wp.array:
    """Return topology data as a CPU Warp int array with shape ``(-1, width)``."""
    if isinstance(data, wp.array):
        if data.ndim == 2:
            return data.to("cpu")
        if data.dtype == wp.int32 and data.is_contiguous:
            return data.reshape((-1, width)).to("cpu")
        return wp.array(data.numpy().reshape(-1, width), dtype=wp.int32, device="cpu")

    return wp.array(_numpy_int_rows(data, width), dtype=wp.int32, device="cpu")


def _as_cpu_int_array1d(data) -> wp.array:
    """Return topology data as a flat CPU Warp int array."""
    if isinstance(data, wp.array):
        if data.ndim == 1:
            return data.to("cpu")
        if data.dtype == wp.int32 and data.is_contiguous:
            return data.flatten().to("cpu")
        return wp.array(data.numpy().reshape(-1), dtype=wp.int32, device="cpu")

    return wp.array(_numpy_int_array(data).reshape(-1), dtype=wp.int32, device="cpu")


def _empty_vertex_adjacency() -> tuple[np.ndarray, np.ndarray]:
    """Return empty (NumPy) adjacency values and offsets arrays."""
    return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)


def _build_vertex_adjacency_with_warp(
    topology: wp.array,
    particle_count: int,
    *,
    count_kernel,
    fill_kernel,
    values_per_entry: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build vertex-adjacency CSR arrays (NumPy) using the VBD count/fill Warp kernels on CPU.

    The kernels run on CPU; results are copied out to NumPy so the host
    ``MeshAdjacency`` stays free of Warp arrays (``to`` re-uploads on demand).
    """
    with wp.ScopedDevice("cpu"):
        counts = wp.zeros(shape=(particle_count,), dtype=wp.int32, device="cpu")
        wp.launch(count_kernel, inputs=[topology, counts], dim=1, device="cpu")

        counts_np = counts.numpy()
        offsets_np = np.empty(shape=(particle_count + 1,), dtype=np.int32)
        offsets_np[0] = 0
        offsets_np[1:] = np.cumsum(values_per_entry * counts_np)[:]
        offsets = wp.array(offsets_np, dtype=wp.int32, device="cpu")

        fill_count = wp.zeros(shape=(particle_count,), dtype=wp.int32, device="cpu")
        values = wp.empty(shape=(int(values_per_entry * counts_np.sum()),), dtype=wp.int32, device="cpu")
        wp.launch(fill_kernel, inputs=[topology, offsets, fill_count, values], dim=1, device="cpu")

    # Copy out of the CPU Warp buffers (which go out of scope here) into owned NumPy arrays.
    return values.numpy().copy(), offsets_np


def create_mesh_sphere(
    radius: float = 1.0,
    *,
    num_latitudes: int = default_num_segments,
    num_longitudes: int = default_num_segments,
    reverse_winding: bool = False,
    compute_normals: bool = True,
    compute_uvs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Create sphere geometry data with optional normals and UVs."""
    positions = []
    normals = [] if compute_normals else None
    uvs = [] if compute_uvs else None
    indices = []

    for i in range(num_latitudes + 1):
        theta = i * np.pi / num_latitudes
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        for j in range(num_longitudes + 1):
            phi = j * 2 * np.pi / num_longitudes
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            x = cos_phi * sin_theta
            y = cos_theta
            z = sin_phi * sin_theta
            positions.append([x * radius, y * radius, z * radius])
            if compute_normals:
                normals.append([x, y, z])
            if compute_uvs:
                u = float(j) / num_longitudes
                v = float(i) / num_latitudes
                uvs.append([u, v])

    for i in range(num_latitudes):
        for j in range(num_longitudes):
            first = i * (num_longitudes + 1) + j
            second = first + num_longitudes + 1
            if reverse_winding:
                indices.extend([first, second, first + 1, second, second + 1, first + 1])
            else:
                indices.extend([first, first + 1, second, second, first + 1, second + 1])

    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
        None if normals is None else np.asarray(normals, dtype=np.float32),
        None if uvs is None else np.asarray(uvs, dtype=np.float32),
    )


def create_mesh_ellipsoid(
    rx: float = 1.0,
    ry: float = 1.0,
    rz: float = 1.0,
    *,
    num_latitudes: int = default_num_segments,
    num_longitudes: int = default_num_segments,
    reverse_winding: bool = False,
    compute_normals: bool = True,
    compute_uvs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Create ellipsoid geometry data with optional normals and UVs."""
    positions = []
    normals = [] if compute_normals else None
    uvs = [] if compute_uvs else None
    indices = []

    for i in range(num_latitudes + 1):
        theta = i * np.pi / num_latitudes
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        for j in range(num_longitudes + 1):
            phi = j * 2 * np.pi / num_longitudes
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            ux = cos_phi * sin_theta
            uy = cos_theta
            uz = sin_phi * sin_theta
            px = ux * rx
            py = uy * ry
            pz = uz * rz
            positions.append([px, py, pz])

            if compute_normals:
                nx = ux / rx
                ny = uy / ry
                nz = uz / rz
                n_len = np.sqrt(nx * nx + ny * ny + nz * nz)
                if n_len > 1e-10:
                    nx /= n_len
                    ny /= n_len
                    nz /= n_len
                normals.append([nx, ny, nz])
            if compute_uvs:
                u = float(j) / num_longitudes
                v = float(i) / num_latitudes
                uvs.append([u, v])

    for i in range(num_latitudes):
        for j in range(num_longitudes):
            first = i * (num_longitudes + 1) + j
            second = first + num_longitudes + 1
            if reverse_winding:
                indices.extend([first, second, first + 1, second, second + 1, first + 1])
            else:
                indices.extend([first, first + 1, second, second, first + 1, second + 1])

    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
        None if normals is None else np.asarray(normals, dtype=np.float32),
        None if uvs is None else np.asarray(uvs, dtype=np.float32),
    )


def _normalize_color(color) -> tuple[float, float, float] | None:
    if color is None:
        return None
    color = np.asarray(color, dtype=np.float32).flatten()
    if color.size >= 3:
        if np.max(color) > 1.0:
            color = color / 255.0
        return (float(color[0]), float(color[1]), float(color[2]))
    return None


def _extract_trimesh_texture(visual_or_material, base_dir: str) -> np.ndarray | str | None:
    """Extract texture from a trimesh visual or a single material object."""
    material = getattr(visual_or_material, "material", visual_or_material)
    if material is None:
        return None

    image = getattr(material, "image", None)
    image_path = getattr(material, "image_path", None)

    if image is None:
        base_color_texture = getattr(material, "baseColorTexture", None)
        if base_color_texture is not None:
            image = getattr(base_color_texture, "image", None)
            image_path = image_path or getattr(base_color_texture, "image_path", None)
            if image is None:
                if isinstance(base_color_texture, (str, os.PathLike)):
                    image_path = image_path or os.fspath(base_color_texture)
                else:
                    image = base_color_texture

    if image is not None:
        try:
            return np.array(image)
        except Exception:
            pass

    if image_path:
        if not os.path.isabs(image_path):
            image_path = os.path.abspath(os.path.join(base_dir, image_path))
        return image_path

    return None


def _extract_trimesh_material_params(
    material,
) -> tuple[float | None, float | None, tuple[float, float, float] | None]:
    if material is None:
        return None, None, None

    base_color = None
    metallic = None
    roughness = None

    color_candidates = [
        getattr(material, "baseColorFactor", None),
        getattr(material, "diffuse", None),
        getattr(material, "diffuseColor", None),
    ]
    for candidate in color_candidates:
        if candidate is not None:
            base_color = _normalize_color(candidate)
            break

    for attr_name in ("metallicFactor", "metallic"):
        value = getattr(material, attr_name, None)
        if value is not None:
            metallic = float(value)
            break

    for attr_name in ("roughnessFactor", "roughness"):
        value = getattr(material, attr_name, None)
        if value is not None:
            roughness = float(value)
            break

    if roughness is None:
        for attr_name in ("glossiness", "shininess"):
            value = getattr(material, attr_name, None)
            if value is not None:
                gloss = float(value)
                if attr_name == "shininess":
                    gloss = min(max(gloss / 1000.0, 0.0), 1.0)
                roughness = 1.0 - min(max(gloss, 0.0), 1.0)
                break

    return roughness, metallic, base_color


def load_meshes_from_file(
    filename: str,
    *,
    scale: np.ndarray | list[float] | tuple[float, ...] = (1.0, 1.0, 1.0),
    maxhullvert: int,
    override_color: np.ndarray | list[float] | tuple[float, float, float] | None = None,
    override_texture: np.ndarray | str | None = None,
) -> list[Mesh]:
    """Load meshes from a file using trimesh and capture texture data if present.

    Args:
        filename: Path to the mesh file.
        scale: Per-axis scale to apply to vertices.
        maxhullvert: Maximum vertices for convex hull approximation.
        override_color: Optional base color override (RGB).
        override_texture: Optional texture path/URL or image override.

    Returns:
        List of Mesh objects.
    """
    import trimesh

    filename = os.fspath(filename)
    scale = np.asarray(scale, dtype=np.float32)
    base_dir = os.path.dirname(filename)

    def _parse_dae_material_colors(
        path: str,
    ) -> tuple[list[str], dict[str, dict[str, float | str | tuple[float, float, float] | None]]]:
        try:
            tree = ET.parse(path)
            root = tree.getroot()
        except Exception:
            return [], {}

        def strip(tag: str) -> str:
            return tag.split("}", 1)[-1] if "}" in tag else tag

        image_paths: dict[str, str] = {}
        for image in root.iter():
            if strip(image.tag) != "image":
                continue
            image_id = image.attrib.get("id")
            image_name = image.attrib.get("name")
            image_path = None
            for child in image.iter():
                if strip(child.tag) == "init_from" and child.text:
                    image_path = child.text.strip()
                    break
            if image_path:
                if image_id:
                    image_paths[image_id] = image_path
                if image_name:
                    image_paths[image_name] = image_path

        def resolve_dae_texture_path(texture_path: str | None) -> str | None:
            if not texture_path:
                return None
            texture_path = image_paths.get(texture_path.lstrip("#"), texture_path)
            parsed = urlparse(texture_path)
            if parsed.scheme in {"file", "http", "https", "data"}:
                return texture_path
            if not os.path.isabs(texture_path):
                texture_path = os.path.abspath(os.path.join(base_dir, texture_path))
            return texture_path

        # Map effect id -> material properties
        effect_props: dict[str, dict[str, float | str | tuple[float, float, float] | None]] = {}
        for effect in root.iter():
            if strip(effect.tag) != "effect":
                continue
            effect_id = effect.attrib.get("id")
            if not effect_id:
                continue
            surface_images: dict[str, str] = {}
            sampler_surfaces: dict[str, str] = {}
            for newparam in effect.iter():
                if strip(newparam.tag) != "newparam":
                    continue
                sid = newparam.attrib.get("sid")
                if not sid:
                    continue
                for child in newparam:
                    child_tag = strip(child.tag)
                    if child_tag == "surface":
                        for init in child.iter():
                            if strip(init.tag) == "init_from" and init.text:
                                surface_images[sid] = init.text.strip()
                                break
                    elif child_tag == "sampler2D":
                        for source in child.iter():
                            if strip(source.tag) == "source" and source.text:
                                sampler_surfaces[sid] = source.text.strip()
                                break
            diffuse_color = None
            diffuse_texture = None
            specular_color = None
            specular_intensity = None
            shininess = None
            for shader_tag in ("phong", "lambert", "blinn"):
                shader = None
                for elem in effect.iter():
                    if strip(elem.tag) == shader_tag:
                        shader = elem
                        break
                if shader is None:
                    continue
                for node in shader.iter():
                    tag = strip(node.tag)
                    if tag == "diffuse":
                        for diffuse_node in node.iter():
                            diffuse_tag = strip(diffuse_node.tag)
                            if diffuse_tag == "texture":
                                sampler_id = diffuse_node.attrib.get("texture")
                                surface_id = sampler_surfaces.get(sampler_id, sampler_id)
                                image_id = surface_images.get(surface_id, surface_id)
                                diffuse_texture = resolve_dae_texture_path(image_id)
                                break
                            if diffuse_tag == "color" and diffuse_node.text:
                                values = [float(x) for x in diffuse_node.text.strip().split()]
                                if len(values) >= 3:
                                    # DAE diffuse colors are commonly authored in linear space.
                                    # Convert to sRGB for the viewer shader (which converts to linear).
                                    diffuse = np.clip(values[:3], 0.0, 1.0)
                                    srgb = np.power(diffuse, 1.0 / 2.2)
                                    diffuse_color = (float(srgb[0]), float(srgb[1]), float(srgb[2]))
                                    break
                        continue
                    if tag == "specular":
                        for col in node.iter():
                            if strip(col.tag) == "color" and col.text:
                                values = [float(x) for x in col.text.strip().split()]
                                if len(values) >= 3:
                                    specular_color = (values[0], values[1], values[2])
                                    break
                        continue
                    if tag == "reflectivity":
                        for val in node.iter():
                            if strip(val.tag) == "float" and val.text:
                                try:
                                    specular_intensity = float(val.text.strip())
                                except ValueError:
                                    specular_intensity = None
                                break
                        continue
                    if tag == "shininess":
                        for val in node.iter():
                            if strip(val.tag) == "float" and val.text:
                                try:
                                    shininess = float(val.text.strip())
                                except ValueError:
                                    shininess = None
                                break
                        continue
                if diffuse_color is not None or diffuse_texture is not None:
                    break
            metallic = None
            if specular_color is not None:
                metallic = float(np.clip(np.max(specular_color), 0.0, 1.0))
            elif specular_intensity is not None:
                metallic = float(np.clip(specular_intensity, 0.0, 1.0))
            roughness = None
            if shininess is not None:
                if shininess > 1.0:
                    shininess = min(shininess / 128.0, 1.0)
                roughness = float(np.clip(1.0 - shininess, 0.0, 1.0))
            if diffuse_color is not None or diffuse_texture is not None:
                effect_props[effect_id] = {
                    "color": diffuse_color,
                    "texture": diffuse_texture,
                    "metallic": metallic,
                    "roughness": roughness,
                }

        # Map material id/name -> material properties
        material_colors: dict[str, dict[str, float | str | tuple[float, float, float] | None]] = {}
        for material in root.iter():
            if strip(material.tag) != "material":
                continue
            mat_id = material.attrib.get("id") or material.attrib.get("name")
            effect_url = None
            for inst in material.iter():
                if strip(inst.tag) == "instance_effect":
                    effect_url = inst.attrib.get("url")
                    break
            if mat_id and effect_url and effect_url.startswith("#"):
                effect_id = effect_url[1:]
                if effect_id in effect_props:
                    material_colors[mat_id] = effect_props[effect_id]

        # Collect triangle material assignments in order
        face_materials: list[str] = []
        for triangles in root.iter():
            if strip(triangles.tag) != "triangles":
                continue
            mat = triangles.attrib.get("material")
            count = triangles.attrib.get("count")
            if not mat or count is None:
                continue
            try:
                tri_count = int(count)
            except ValueError:
                continue
            face_materials.extend([mat] * tri_count)

        return face_materials, material_colors

    dae_face_materials: list[str] = []
    dae_material_colors: dict[str, dict[str, float | str | tuple[float, float, float] | None]] = {}
    if filename.lower().endswith(".dae"):
        dae_face_materials, dae_material_colors = _parse_dae_material_colors(filename)

    if filename.lower().endswith(".dae"):
        with warnings.catch_warnings():
            # Remove when the pycollada floor includes a release that replaces
            # load-time NumPy array shape assignment with reshape.
            warnings.filterwarnings(
                "ignore",
                message=r"Setting the shape on a NumPy array has been deprecated.*",
                category=DeprecationWarning,
                module=r"^collada\.",
            )
            tri = trimesh.load(filename, force="mesh")
    else:
        tri = trimesh.load(filename, force="mesh")
    tri_meshes = tri.geometry.values() if hasattr(tri, "geometry") else [tri]

    meshes = []
    for tri_mesh in tri_meshes:
        vertices = np.array(tri_mesh.vertices, dtype=np.float32) * scale
        faces = np.array(tri_mesh.faces, dtype=np.int32)
        normals = np.array(tri_mesh.vertex_normals, dtype=np.float32) if tri_mesh.vertex_normals is not None else None
        if normals is None or not np.isfinite(normals).all() or np.allclose(normals, 0.0):
            normals = compute_vertex_normals(vertices, faces)

        uvs = None
        if hasattr(tri_mesh, "visual") and getattr(tri_mesh.visual, "uv", None) is not None:
            uvs = np.array(tri_mesh.visual.uv, dtype=np.float32)

        color = _normalize_color(override_color) if override_color is not None else None
        texture = override_texture

        def add_mesh_from_faces(
            face_indices,
            *,
            mat_color=None,
            mat_roughness=None,
            mat_metallic=None,
            mesh_vertices=None,
            mesh_normals=None,
            mesh_uvs=None,
            mesh_texture=None,
        ):
            used = np.unique(face_indices.flatten())
            remap = {int(old): i for i, old in enumerate(used)}
            remapped_faces = np.vectorize(remap.get)(face_indices).astype(np.int32)

            sub_vertices = mesh_vertices[used]
            sub_normals = mesh_normals[used] if mesh_normals is not None else None
            force_smooth = False
            if mat_metallic is not None and mat_metallic > 0.0:
                force_smooth = True
            if mat_roughness is not None and mat_roughness < 0.6:
                force_smooth = True
            if sub_normals is None or force_smooth:
                sub_normals = smooth_vertex_normals_by_position(sub_vertices, remapped_faces)
            sub_uvs = mesh_uvs[used] if mesh_uvs is not None else None
            if mesh_texture is not None and mat_color is None:
                mat_color = (1.0, 1.0, 1.0)

            meshes.append(
                Mesh(
                    sub_vertices,
                    remapped_faces.flatten(),
                    normals=sub_normals,
                    uvs=sub_uvs,
                    maxhullvert=maxhullvert,
                    color=mat_color,
                    texture=mesh_texture,
                    roughness=mat_roughness,
                    metallic=mat_metallic,
                )
            )

        # If a uniform override is provided, skip per-material splitting.
        if color is not None or texture is not None:
            add_mesh_from_faces(
                faces,
                mat_color=color,
                mesh_vertices=vertices,
                mesh_normals=normals,
                mesh_uvs=uvs,
                mesh_texture=texture,
            )
            continue

        # Handle per-face materials if available (e.g. DAE with multiple materials)
        face_materials = getattr(tri_mesh.visual, "face_materials", None) if hasattr(tri_mesh, "visual") else None
        materials = getattr(tri_mesh.visual, "materials", None) if hasattr(tri_mesh, "visual") else None
        if face_materials is not None and materials is not None:
            face_materials = np.array(face_materials, dtype=np.int32).flatten()
            for mat_index in np.unique(face_materials):
                mat_faces = faces[face_materials == mat_index]
                material = materials[int(mat_index)] if int(mat_index) < len(materials) else None
                roughness, metallic, base_color = _extract_trimesh_material_params(material)
                mat_color = base_color
                mat_texture = _extract_trimesh_texture(material, base_dir)
                if mat_color is None and hasattr(tri_mesh.visual, "main_color"):
                    mat_color = _normalize_color(tri_mesh.visual.main_color)
                add_mesh_from_faces(
                    mat_faces,
                    mat_color=mat_color,
                    mat_roughness=roughness,
                    mat_metallic=metallic,
                    mesh_vertices=vertices,
                    mesh_normals=normals,
                    mesh_uvs=uvs,
                    mesh_texture=mat_texture,
                )
            continue

        # DAE fallback: use material groups from the source file if trimesh didn't expose them
        if dae_face_materials and len(dae_face_materials) == len(faces):
            face_materials = np.array(dae_face_materials, dtype=object)
            for mat_name in np.unique(face_materials):
                mat_faces = faces[face_materials == mat_name]
                mat_props = dae_material_colors.get(str(mat_name), {})
                mat_color = mat_props.get("color")
                mat_roughness = mat_props.get("roughness")
                mat_metallic = mat_props.get("metallic")
                mat_texture = mat_props.get("texture", texture)
                add_mesh_from_faces(
                    mat_faces,
                    mat_color=mat_color,
                    mat_roughness=mat_roughness,
                    mat_metallic=mat_metallic,
                    mesh_vertices=vertices,
                    mesh_normals=normals,
                    mesh_uvs=uvs,
                    mesh_texture=mat_texture,
                )
            continue

        # Handle per-face color visuals (common for DAE via ColorVisuals)
        face_colors = getattr(tri_mesh.visual, "face_colors", None) if hasattr(tri_mesh, "visual") else None
        if face_colors is not None:
            face_colors = np.array(face_colors, dtype=np.float32)
            if face_colors.shape[0] == faces.shape[0]:
                # Normalize to 0..1 rgb
                if np.max(face_colors) > 1.0:
                    face_colors = face_colors / 255.0
                rgb = face_colors[:, :3]
                # quantize to avoid tiny float differences
                rgb = np.round(rgb, 4)
                unique_colors, inverse = np.unique(rgb, axis=0, return_inverse=True)
                for color_idx, mat_color in enumerate(unique_colors):
                    mat_faces = faces[inverse == color_idx]
                    add_mesh_from_faces(
                        mat_faces,
                        mat_color=(float(mat_color[0]), float(mat_color[1]), float(mat_color[2])),
                        mesh_vertices=vertices,
                        mesh_normals=normals,
                        mesh_uvs=uvs,
                        mesh_texture=texture,
                    )
                continue

        # Handle per-vertex colors by computing face colors
        vertex_colors = getattr(tri_mesh.visual, "vertex_colors", None) if hasattr(tri_mesh, "visual") else None
        if vertex_colors is not None:
            vertex_colors = np.array(vertex_colors, dtype=np.float32)
            if np.max(vertex_colors) > 1.0:
                vertex_colors = vertex_colors / 255.0
            rgb = vertex_colors[:, :3]
            face_rgb = rgb[faces].mean(axis=1)
            face_rgb = np.round(face_rgb, 4)
            unique_colors, inverse = np.unique(face_rgb, axis=0, return_inverse=True)
            for color_idx, mat_color in enumerate(unique_colors):
                mat_faces = faces[inverse == color_idx]
                add_mesh_from_faces(
                    mat_faces,
                    mat_color=(float(mat_color[0]), float(mat_color[1]), float(mat_color[2])),
                    mesh_vertices=vertices,
                    mesh_normals=normals,
                    mesh_uvs=uvs,
                    mesh_texture=texture,
                )
            continue

        # Single-material mesh fallback
        roughness = None
        metallic = None
        if color is None and hasattr(tri_mesh, "visual") and hasattr(tri_mesh.visual, "main_color"):
            color = _normalize_color(tri_mesh.visual.main_color)

        if hasattr(tri_mesh, "visual") and texture is None:
            texture = _extract_trimesh_texture(tri_mesh.visual, base_dir)
            material = getattr(tri_mesh.visual, "material", None)
            roughness, metallic, base_color = _extract_trimesh_material_params(material)
            if color is None and base_color is not None:
                color = base_color

        meshes.append(
            Mesh(
                vertices,
                faces.flatten(),
                normals=normals,
                uvs=uvs,
                maxhullvert=maxhullvert,
                color=color,
                texture=texture,
                roughness=roughness,
                metallic=metallic,
            )
        )

    return meshes


def create_mesh_capsule(
    radius: float,
    half_height: float,
    *,
    up_axis: int = 1,
    segments: int = default_num_segments,
    compute_normals: bool = True,
    compute_uvs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Create capsule geometry data with optional normals and UVs."""
    positions = []
    normals = [] if compute_normals else None
    uvs = [] if compute_uvs else None
    indices = []

    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be between 0 and 2")

    x_dir, y_dir, z_dir = ((1, 2, 0), (0, 1, 2), (2, 0, 1))[up_axis]
    up_vector = np.zeros(3, dtype=np.float32)
    up_vector[up_axis] = half_height

    for i in range(segments + 1):
        theta = i * np.pi / segments
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        for j in range(segments + 1):
            phi = j * 2 * np.pi / segments
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            z = cos_phi * sin_theta
            y = cos_theta
            x = sin_phi * sin_theta

            xyz = np.array((x, y, z), dtype=np.float32)
            normal = xyz[[x_dir, y_dir, z_dir]]
            pos = normal * radius
            if normal[up_axis] >= 0.0:
                pos += up_vector
            else:
                pos -= up_vector

            positions.append(pos.tolist())
            if compute_normals:
                normals.append(normal.tolist())
            if compute_uvs:
                u = cos_theta * 0.5 + 0.5
                v = cos_phi * sin_theta * 0.5 + 0.5
                uvs.append([u, v])

    nv = len(positions)
    for i in range(segments):
        for j in range(segments):
            first = (i * (segments + 1) + j) % nv
            second = (first + segments + 1) % nv
            indices.extend([first, second, (first + 1) % nv, second, (second + 1) % nv, (first + 1) % nv])

    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
        None if normals is None else np.asarray(normals, dtype=np.float32),
        None if uvs is None else np.asarray(uvs, dtype=np.float32),
    )


def create_mesh_cone(
    radius: float,
    half_height: float,
    *,
    up_axis: int = 1,
    segments: int = default_num_segments,
    compute_normals: bool = True,
    compute_uvs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Create cone geometry data with optional normals and UVs."""
    return create_mesh_cylinder(
        radius,
        half_height,
        up_axis=up_axis,
        segments=segments,
        top_radius=0.0,
        compute_normals=compute_normals,
        compute_uvs=compute_uvs,
    )


def create_mesh_cylinder(
    radius: float,
    half_height: float,
    *,
    up_axis: int = 1,
    segments: int = default_num_segments,
    top_radius: float | None = None,
    compute_normals: bool = True,
    compute_uvs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Create cylinder/truncated cone geometry data with optional normals/UVs."""
    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be between 0 and 2")

    x_dir, y_dir, z_dir = ((1, 2, 0), (0, 1, 2), (2, 0, 1))[up_axis]
    if top_radius is None:
        top_radius = radius

    indices = []
    positions = []
    normals = [] if compute_normals else None
    uvs = [] if compute_uvs else None

    def add_vertex(position: np.ndarray, normal: np.ndarray | None, uv: tuple[float, float] | None) -> int:
        idx = len(positions)
        positions.append(position.tolist())
        if compute_normals:
            assert normals is not None
            normals.append([0.0, 0.0, 0.0] if normal is None else normal.tolist())
        if compute_uvs:
            assert uvs is not None
            uvs.append([0.0, 0.0] if uv is None else [uv[0], uv[1]])
        return idx

    side_radial_component = 2.0 * half_height
    side_axial_component = radius - top_radius

    # Side vertices first (contiguous layout for robust indexing).
    side_bottom_indices = []
    for i in range(segments):
        theta = 2 * np.pi * i / segments
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        position = np.array([radius * cos_theta, -half_height, radius * sin_theta], dtype=np.float32)
        position = position[[x_dir, y_dir, z_dir]]

        side_normal = None
        if compute_normals:
            side_normal = np.array(
                [
                    side_radial_component * cos_theta,
                    side_axial_component,
                    side_radial_component * sin_theta,
                ],
                dtype=np.float32,
            )
            normal_length = np.linalg.norm(side_normal)
            if normal_length > 0.0:
                side_normal = side_normal / normal_length
            side_normal = side_normal[[x_dir, y_dir, z_dir]]

        side_uv = (i / max(segments - 1, 1), 0.0) if compute_uvs else None
        side_bottom_indices.append(add_vertex(position, side_normal, side_uv))

    side_top_indices = []
    side_apex_index: int | None = None
    if top_radius > 0.0:
        for i in range(segments):
            theta = 2 * np.pi * i / segments
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            position = np.array([top_radius * cos_theta, half_height, top_radius * sin_theta], dtype=np.float32)
            position = position[[x_dir, y_dir, z_dir]]

            side_normal = None
            if compute_normals:
                side_normal = np.array(
                    [
                        side_radial_component * cos_theta,
                        side_axial_component,
                        side_radial_component * sin_theta,
                    ],
                    dtype=np.float32,
                )
                normal_length = np.linalg.norm(side_normal)
                if normal_length > 0.0:
                    side_normal = side_normal / normal_length
                side_normal = side_normal[[x_dir, y_dir, z_dir]]

            side_uv = (i / max(segments - 1, 1), 1.0) if compute_uvs else None
            side_top_indices.append(add_vertex(position, side_normal, side_uv))
    else:
        apex_position = np.array([0.0, half_height, 0.0], dtype=np.float32)[[x_dir, y_dir, z_dir]]
        apex_normal = None
        if compute_normals:
            apex_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)[[x_dir, y_dir, z_dir]]
        side_apex_index = add_vertex(apex_position, apex_normal, (0.5, 1.0) if compute_uvs else None)

    # Cap vertices after side vertices (also contiguous per cap).
    cap_center_bottom_idx: int | None = None
    cap_center_top_idx: int | None = None

    if radius > 0.0:
        cap_center_bottom_pos = np.array([0.0, -half_height, 0.0], dtype=np.float32)[[x_dir, y_dir, z_dir]]
        cap_center_bottom_n = (
            np.array([0.0, -1.0, 0.0], dtype=np.float32)[[x_dir, y_dir, z_dir]] if compute_normals else None
        )
        cap_center_bottom_idx = add_vertex(
            cap_center_bottom_pos, cap_center_bottom_n, (0.5, 0.5) if compute_uvs else None
        )

    if top_radius > 0.0:
        cap_center_top_pos = np.array([0.0, half_height, 0.0], dtype=np.float32)[[x_dir, y_dir, z_dir]]
        cap_center_top_n = (
            np.array([0.0, 1.0, 0.0], dtype=np.float32)[[x_dir, y_dir, z_dir]] if compute_normals else None
        )
        cap_center_top_idx = add_vertex(cap_center_top_pos, cap_center_top_n, (0.5, 0.5) if compute_uvs else None)

    cap_ring_bottom_indices = []
    if radius > 0.0:
        for i in range(segments):
            theta = 2 * np.pi * i / segments
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            position = np.array([radius * cos_theta, -half_height, radius * sin_theta], dtype=np.float32)
            position = position[[x_dir, y_dir, z_dir]]
            cap_normal = (
                np.array([0.0, -1.0, 0.0], dtype=np.float32)[[x_dir, y_dir, z_dir]] if compute_normals else None
            )
            cap_uv = (cos_theta * 0.5 + 0.5, sin_theta * 0.5 + 0.5) if compute_uvs else None
            cap_ring_bottom_indices.append(add_vertex(position, cap_normal, cap_uv))

    cap_ring_top_indices = []
    if top_radius > 0.0:
        for i in range(segments):
            theta = 2 * np.pi * i / segments
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            position = np.array([top_radius * cos_theta, half_height, top_radius * sin_theta], dtype=np.float32)
            position = position[[x_dir, y_dir, z_dir]]
            cap_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)[[x_dir, y_dir, z_dir]] if compute_normals else None
            cap_uv = (cos_theta * 0.5 + 0.5, sin_theta * 0.5 + 0.5) if compute_uvs else None
            cap_ring_top_indices.append(add_vertex(position, cap_normal, cap_uv))

    # Bottom cap
    if cap_center_bottom_idx is not None and cap_ring_bottom_indices:
        for i in range(segments):
            i0 = cap_ring_bottom_indices[i]
            i1 = cap_ring_bottom_indices[(i + 1) % segments]
            indices.extend([cap_center_bottom_idx, i0, i1])

    # Top cap
    if cap_center_top_idx is not None and cap_ring_top_indices:
        for i in range(segments):
            i0 = cap_ring_top_indices[i]
            i1 = cap_ring_top_indices[(i + 1) % segments]
            indices.extend([cap_center_top_idx, i1, i0])

    # Side faces
    for i in range(segments):
        bottom_i = side_bottom_indices[i]
        bottom_next = side_bottom_indices[(i + 1) % segments]

        if top_radius > 0.0:
            top_i = side_top_indices[i]
            top_next = side_top_indices[(i + 1) % segments]
            indices.extend([top_i, top_next, bottom_i, top_next, bottom_next, bottom_i])
        else:
            assert side_apex_index is not None
            indices.extend([side_apex_index, bottom_next, bottom_i])

    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
        None if normals is None else np.asarray(normals, dtype=np.float32),
        None if uvs is None else np.asarray(uvs, dtype=np.float32),
    )


def create_mesh_arrow(
    base_radius: float,
    base_height: float,
    *,
    cap_radius: float | None = None,
    cap_height: float | None = None,
    up_axis: int = 1,
    segments: int = default_num_segments,
    compute_normals: bool = True,
    compute_uvs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Create arrow geometry data with optional normals and UVs."""
    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be between 0 and 2")
    if cap_radius is None:
        cap_radius = base_radius * 1.8
    if cap_height is None:
        cap_height = base_height * 0.18

    up_vector = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up_vector[up_axis] = 1.0

    base_positions, base_indices, base_normals, base_uvs = create_mesh_cylinder(
        base_radius,
        base_height / 2,
        up_axis=up_axis,
        segments=segments,
        compute_normals=compute_normals,
        compute_uvs=compute_uvs,
    )
    cap_positions, cap_indices, cap_normals, cap_uvs = create_mesh_cone(
        cap_radius,
        cap_height / 2,
        up_axis=up_axis,
        segments=segments,
        compute_normals=compute_normals,
        compute_uvs=compute_uvs,
    )

    base_positions = base_positions.copy()
    cap_positions = cap_positions.copy()
    base_positions += base_height / 2 * up_vector
    cap_positions += (base_height + cap_height / 2 - 1e-3 * base_height) * up_vector

    positions = np.vstack((base_positions, cap_positions))
    indices = np.hstack((base_indices, cap_indices + len(base_positions)))
    normals = None
    uvs = None
    if compute_normals:
        normals = np.vstack((base_normals, cap_normals))
    if compute_uvs:
        uvs = np.vstack((base_uvs, cap_uvs))
    return positions.astype(np.float32), indices.astype(np.uint32), normals, uvs


def create_mesh_box(
    hx: float,
    hy: float,
    hz: float,
    *,
    duplicate_vertices: bool = True,
    compute_normals: bool = True,
    compute_uvs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Create box geometry data with optional duplicated vertices, normals, and UVs."""
    if duplicate_vertices:
        # fmt: off
        positions = np.array(
            [
                [-hx, -hy, -hz], [-hx, -hy,  hz], [-hx,  hy,  hz], [-hx,  hy, -hz],
                [ hx, -hy, -hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [ hx,  hy, -hz],
                [-hx, -hy, -hz], [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx, -hy, -hz],
                [-hx,  hy, -hz], [-hx,  hy,  hz], [ hx,  hy,  hz], [ hx,  hy, -hz],
                [-hx, -hy, -hz], [-hx,  hy, -hz], [ hx,  hy, -hz], [ hx, -hy, -hz],
                [-hx, -hy,  hz], [-hx,  hy,  hz], [ hx,  hy,  hz], [ hx, -hy,  hz],
            ],
            dtype=np.float32,
        )
        indices = np.array(
            [
                 0,  1,  2,  0,  2,  3,   4,  6,  5,  4,  7,  6,
                 8, 10,  9,  8, 11, 10,  12, 13, 14, 12, 14, 15,
                16, 17, 18, 16, 18, 19,  20, 22, 21, 20, 23, 22,
            ],
            dtype=np.uint32,
        )
        # fmt: on
        normals = None
        uvs = None
        if compute_normals:
            normals = np.array(
                [
                    [-1, 0, 0],
                    [-1, 0, 0],
                    [-1, 0, 0],
                    [-1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 0, -1],
                    [0, 0, -1],
                    [0, 0, -1],
                    [0, 0, -1],
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
        if compute_uvs:
            face_uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
            uvs = np.vstack([face_uv] * 6).astype(np.float32)
        return positions, indices, normals, uvs

    positions = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    # fmt: off
    indices = np.array(
        [
            0, 2, 1, 0, 3, 2,  4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,  2, 3, 7, 2, 7, 6,
            0, 4, 7, 0, 7, 3,  1, 2, 6, 1, 6, 5,
        ],
        dtype=np.uint32,
    )
    # fmt: on
    normals = None
    uvs = None
    if compute_normals:
        normals = compute_vertex_normals(positions, indices).astype(np.float32)
    if compute_uvs:
        uvs = np.zeros((len(positions), 2), dtype=np.float32)
    return positions, indices, normals, uvs


def create_mesh_plane(
    width: float,
    length: float,
    *,
    compute_normals: bool = True,
    compute_uvs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Create plane geometry data with optional normals and UVs."""
    half_width = width / 2
    half_length = length / 2
    positions = np.array(
        [
            [-half_width, -half_length, 0.0],
            [half_width, -half_length, 0.0],
            [half_width, half_length, 0.0],
            [-half_width, half_length, 0.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    normals = None
    uvs = None
    if compute_normals:
        normals = np.array([[0.0, 0.0, 1.0]] * 4, dtype=np.float32)
    if compute_uvs:
        uvs = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    return positions, indices, normals, uvs


@wp.kernel
def solidify_mesh_kernel(
    indices: wp.array2d[int],
    vertices: wp.array[wp.vec3],
    thickness: wp.array[float],
    # outputs
    out_vertices: wp.array[wp.vec3],
    out_indices: wp.array2d[int],
):
    """Extrude each triangle into a triangular prism (wedge) for solidification.

    For each input triangle, creates 6 vertices (3 on each side of the surface)
    and 8 output triangles forming a closed wedge. The extrusion is along the
    face normal, with per-vertex thickness values.

    Launch with dim=num_triangles.

    Args:
        indices: Triangle indices of shape (num_triangles, 3).
        vertices: Vertex positions of shape (num_vertices,).
        thickness: Per-vertex thickness values of shape (num_vertices,).
        out_vertices: Output vertices of shape (num_vertices * 2,). Each input
            vertex produces two output vertices (offset ± thickness along normal).
        out_indices: Output triangle indices of shape (num_triangles * 8, 3).
    """
    tid = wp.tid()
    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]

    vi = vertices[i]
    vj = vertices[j]
    vk = vertices[k]

    normal = wp.normalize(wp.cross(vj - vi, vk - vi))
    ti = normal * thickness[i]
    tj = normal * thickness[j]
    tk = normal * thickness[k]

    # wedge vertices
    vi0 = vi + ti
    vi1 = vi - ti
    vj0 = vj + tj
    vj1 = vj - tj
    vk0 = vk + tk
    vk1 = vk - tk

    i0 = i * 2
    i1 = i * 2 + 1
    j0 = j * 2
    j1 = j * 2 + 1
    k0 = k * 2
    k1 = k * 2 + 1

    out_vertices[i0] = vi0
    out_vertices[i1] = vi1
    out_vertices[j0] = vj0
    out_vertices[j1] = vj1
    out_vertices[k0] = vk0
    out_vertices[k1] = vk1

    oid = tid * 8
    out_indices[oid + 0, 0] = i0
    out_indices[oid + 0, 1] = j0
    out_indices[oid + 0, 2] = k0
    out_indices[oid + 1, 0] = j0
    out_indices[oid + 1, 1] = k1
    out_indices[oid + 1, 2] = k0
    out_indices[oid + 2, 0] = j0
    out_indices[oid + 2, 1] = j1
    out_indices[oid + 2, 2] = k1
    out_indices[oid + 3, 0] = j0
    out_indices[oid + 3, 1] = i1
    out_indices[oid + 3, 2] = j1
    out_indices[oid + 4, 0] = j0
    out_indices[oid + 4, 1] = i0
    out_indices[oid + 4, 2] = i1
    out_indices[oid + 5, 0] = j1
    out_indices[oid + 5, 1] = i1
    out_indices[oid + 5, 2] = k1
    out_indices[oid + 6, 0] = i1
    out_indices[oid + 6, 1] = i0
    out_indices[oid + 6, 2] = k0
    out_indices[oid + 7, 0] = i1
    out_indices[oid + 7, 1] = k0
    out_indices[oid + 7, 2] = k1


def solidify_mesh(
    faces: np.ndarray,
    vertices: np.ndarray,
    thickness: float | list | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a surface mesh into a solid mesh by extruding along face normals.

    Takes a triangle mesh representing a surface and creates a closed solid
    mesh by extruding each triangle into a triangular prism (wedge). Each input
    triangle produces 8 output triangles forming the top, bottom, and sides
    of the prism.

    Args:
        faces: Triangle indices of shape (N, 3), where N is the number of
            triangles.
        vertices: Vertex positions of shape (M, 3), where M is the number of
            vertices.
        thickness: Extrusion distance from the surface. Can be a single float
            (uniform thickness), a list, or an array of shape (M,) for
            per-vertex thickness.

    Returns:
        A tuple containing:
            - faces: Output triangle indices of shape (N * 8, 3).
            - vertices: Output vertex positions of shape (M * 2, 3).
    """
    faces = np.array(faces).reshape(-1, 3)
    out_faces = wp.zeros((len(faces) * 8, 3), dtype=wp.int32)
    out_vertices = wp.zeros(len(vertices) * 2, dtype=wp.vec3)
    if not isinstance(thickness, np.ndarray) and not isinstance(thickness, list):
        thickness = [thickness] * len(vertices)
    wp.launch(
        solidify_mesh_kernel,
        dim=len(faces),
        inputs=[wp.array(faces, dtype=int), wp.array(vertices, dtype=wp.vec3), wp.array(thickness, dtype=float)],
        outputs=[out_vertices, out_faces],
    )
    faces = out_faces.numpy()
    vertices = out_vertices.numpy()
    return faces, vertices


def validate_triangle_mesh(
    vertices: np.ndarray,
    indices: np.ndarray,
    *,
    min_area: float = 1e-6,
    max_aspect_ratio: float = 20.0,
    min_angle_deg: float = 5.0,
    label: str | None = None,
    stacklevel: int = 2,
) -> None:
    """Check a triangle mesh for quality issues and emit warnings.

    Inspects the input triangle mesh for degenerate or sliver triangles
    and extreme interior angles. Non-manifold-edge detection is *not*
    performed here; :class:`MeshAdjacency` emits its own warning during
    construction and is built by every builder path that accepts a
    triangle mesh, so going through ``add_cloth_mesh`` /
    ``add_soft_mesh`` already covers it. Standalone callers who need a
    non-manifold check should construct ``MeshAdjacency(indices)``
    themselves. Each detected problem is reported via
    :func:`warnings.warn`.

    Args:
        vertices: Vertex positions [m], shape ``(N, 3)``.
        indices: Triangle vertex indices, shape ``(F, 3)``.
        min_area: Minimum triangle area [m²]. Default ``1e-6`` (1 mm²).
        max_aspect_ratio: Maximum longest-edge / shortest-altitude ratio.
            Default ``20.0`` — flags slivers whose worst interior angle
            is below ~3° while staying quiet on rough-but-fine
            production meshes.
        min_angle_deg: Minimum interior angle [deg]. Default ``5.0``.
        label: Optional name included in the warning message so callers
            can identify which mesh tripped the warning when validating
            many meshes.
        stacklevel: Passed to :func:`warnings.warn` so the warning points at
            the caller's frame.
    """
    vertices = np.asarray(vertices, dtype=float)
    raw = np.asarray(indices, dtype=np.intp)
    if raw.size > 0 and raw.ndim == 1 and raw.size % 3 != 0:
        warnings.warn("Triangle index array length is not a multiple of 3.", stacklevel=stacklevel)
        return
    try:
        indices = raw.reshape(-1, 3)
    except ValueError:
        warnings.warn("Triangle index array must be flat or have shape (N, 3).", stacklevel=stacklevel)
        return
    n_verts = len(vertices)
    n_faces = len(indices)

    if n_faces == 0:
        warnings.warn("Cloth mesh has no triangles.", stacklevel=stacklevel)
        return

    if n_verts > 0 and (indices.min() < 0 or indices.max() >= n_verts):
        warnings.warn(f"Triangle indices out of range for {n_verts} vertices.", stacklevel=stacklevel)
        return

    v0 = vertices[indices[:, 0]]
    v1 = vertices[indices[:, 1]]
    v2 = vertices[indices[:, 2]]

    e01 = v1 - v0
    e12 = v2 - v1
    e20 = v0 - v2

    len01 = np.linalg.norm(e01, axis=1)
    len12 = np.linalg.norm(e12, axis=1)
    len20 = np.linalg.norm(e20, axis=1)
    longest = np.maximum(len01, np.maximum(len12, len20))

    cross = np.cross(e01, -e20)
    area = 0.5 * np.linalg.norm(cross, axis=1)

    eps = 1e-20
    shortest_alt = 2.0 * area / np.maximum(longest, eps)
    aspect = longest / np.maximum(shortest_alt, eps)

    def _ang(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        an = np.maximum(np.linalg.norm(a, axis=1), eps)
        bn = np.maximum(np.linalg.norm(b, axis=1), eps)
        cos = np.einsum("ij,ij->i", a, b) / (an * bn)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    min_angle_arr = np.minimum(_ang(e01, -e20), np.minimum(_ang(-e01, e12), _ang(-e12, e20)))

    issues: list[str] = []

    n_degen = int(np.sum(area < min_area))
    if n_degen > 0:
        issues.append(f"{n_degen} triangle(s) with area < {min_area} m\u00b2")

    n_sliver = int(np.sum(aspect > max_aspect_ratio))
    if n_sliver > 0:
        issues.append(
            f"{n_sliver} sliver triangle(s) with aspect ratio > {max_aspect_ratio} (worst: {float(aspect.max()):.1f})"
        )

    n_small_angle = int(np.sum(min_angle_arr < min_angle_deg))
    if n_small_angle > 0:
        issues.append(
            f"{n_small_angle} triangle(s) with minimum angle < {min_angle_deg}\u00b0"
            f" (smallest: {float(min_angle_arr.min()):.1f}\u00b0)"
        )

    if not issues:
        return

    prefix = "Mesh quality warning"
    if label is not None:
        prefix += f" [{label}]"
    msg = (
        f"{prefix} ({n_verts} vertices, {n_faces} triangles):\n"
        + "\n".join(f"  - {issue}" for issue in issues)
        + "\nConsider remeshing the input geometry."
    )
    warnings.warn(msg, stacklevel=stacklevel)


def validate_tet_mesh(
    vertices: np.ndarray,
    indices: np.ndarray,
    *,
    min_volume: float = 1e-9,
    min_eta: float = 0.01,
    label: str | None = None,
    stacklevel: int = 2,
) -> None:
    """Check a tetrahedral mesh for quality issues and emit warnings.

    Inspects the input tet mesh for inverted elements, small volumes,
    sliver tetrahedra, and non-manifold faces. Each detected problem is
    reported via :func:`warnings.warn`.

    The shape quality metric used is:

    .. math::

        \\eta = \\frac{12\\,(3\\,|V|)^{2/3}}{\\sum_i l_i^2}

    where *V* is the signed volume and *l_i* are the six edge lengths.
    For a regular tetrahedron :math:`\\eta = 1`; degenerate elements
    approach zero.

    Args:
        vertices: Vertex positions [m], shape ``(N, 3)``.
        indices: Tetrahedron vertex indices, shape ``(T, 4)``.
        min_volume: Minimum absolute tet volume [m³]. Default ``1e-9``
            (1 mm³).
        min_eta: Minimum shape quality eta. Default ``0.01``.
        label: Optional name included in the warning message so callers
            can identify which mesh tripped the warning when validating
            many meshes.
        stacklevel: Passed to :func:`warnings.warn`.
    """
    vertices = np.asarray(vertices, dtype=float)
    raw = np.asarray(indices, dtype=np.intp)
    if raw.size > 0 and raw.ndim == 1 and raw.size % 4 != 0:
        warnings.warn("Tet index array length is not a multiple of 4.", stacklevel=stacklevel)
        return
    try:
        indices = raw.reshape(-1, 4)
    except ValueError:
        warnings.warn("Tet index array must be flat or have shape (N, 4).", stacklevel=stacklevel)
        return
    n_tets = len(indices)

    if n_tets == 0:
        warnings.warn("Soft mesh has no tetrahedra.", stacklevel=stacklevel)
        return

    n_verts = len(vertices)
    if n_verts > 0 and (indices.min() < 0 or indices.max() >= n_verts):
        warnings.warn(f"Tet indices out of range for {n_verts} vertices.", stacklevel=stacklevel)
        return

    v0 = vertices[indices[:, 0]]
    v1 = vertices[indices[:, 1]]
    v2 = vertices[indices[:, 2]]
    v3 = vertices[indices[:, 3]]

    d1 = v1 - v0
    d2 = v2 - v0
    d3 = v3 - v0
    vol = np.einsum("ij,ij->i", d1, np.cross(d2, d3)) / 6.0

    issues: list[str] = []

    n_inverted = int(np.sum(vol < 0))
    if n_inverted > 0:
        issues.append(f"{n_inverted}/{n_tets} inverted tetrahedron(s) (negative volume)")

    n_degen = int(np.sum(np.abs(vol) < min_volume))
    if n_degen > 0:
        issues.append(f"{n_degen}/{n_tets} tetrahedron(s) with volume < {min_volume} m\u00b3")

    e01 = v1 - v0
    e02 = v2 - v0
    e03 = v3 - v0
    e12 = v2 - v1
    e13 = v3 - v1
    e23 = v3 - v2
    l_sq_sum = (
        np.sum(e01**2, axis=1)
        + np.sum(e02**2, axis=1)
        + np.sum(e03**2, axis=1)
        + np.sum(e12**2, axis=1)
        + np.sum(e13**2, axis=1)
        + np.sum(e23**2, axis=1)
    )
    eps = 1e-30
    abs_vol = np.abs(vol)
    eta = 12.0 * np.cbrt(3.0 * abs_vol) ** 2 / np.maximum(l_sq_sum, eps)
    n_sliver = int(np.sum(eta < min_eta))
    if n_sliver > 0:
        issues.append(
            f"{n_sliver}/{n_tets} sliver tetrahedron(s) (shape quality eta < {min_eta}; worst: {float(eta.min()):.4f})"
        )

    face_combos = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    all_faces = np.concatenate([np.sort(indices[:, combo], axis=1) for combo in face_combos])
    _, counts = np.unique(all_faces, axis=0, return_counts=True)
    n_nonmanifold = int(np.sum(counts > 2))
    if n_nonmanifold > 0:
        issues.append(f"{n_nonmanifold} non-manifold face(s) shared by more than 2 tetrahedra")

    if not issues:
        return

    prefix = "Tet mesh quality warning"
    if label is not None:
        prefix += f" [{label}]"
    msg = f"{prefix} ({len(vertices)} vertices, {n_tets} tetrahedra):\n" + "\n".join(f"  - {issue}" for issue in issues)
    warnings.warn(msg, stacklevel=stacklevel)
