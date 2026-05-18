# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Detection of redundant manifold edges via a dihedral-angle pre-filter and
an opt-in box-line absorption pass.

The dihedral-angle pre-filter always runs and gates which manifold edges
participate (and reports them on :class:`EdgeRedundancyResult`). The
box-absorption pass only runs when ``enable_box_absorption=True``: for
every surviving manifold edge we build an oriented box in the edge frame
(``dir``, ``tang``, ``normal``), run the SAP broad phase, and exactly test
whether edge ``a``'s box fully absorbs edge ``b``'s segment. Mutual
absorption marks both edges; :func:`resolve_edge_removals` then picks
winners greedily. Output adjacency is CSR (absorbed edges per box).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from .broad_phase_sap import BroadPhaseSAP
from .flags import ShapeFlags

# -----------------------------------------------------------------------------
# Result type
# -----------------------------------------------------------------------------


@dataclass
class EdgeRedundancyResult:
    """Per-edge containment results from :func:`find_redundant_edges`.

    All per-edge arrays are indexed by the manifold-edge subset
    ``edge_indices`` (not the original :attr:`Mesh.edges` rows).

    Attributes:
        edge_indices [-]: Manifold edge vertex pairs ``(M, 2)``.
        dihedral_angles [rad]: Per-edge dihedral angles.
        adjacent_face_area_sum [m^2]: Sum of the two adjacent triangle areas
            per manifold edge. Used by :func:`resolve_edge_removals` as a
            tiebreaker (larger area wins) when sorting by absorb count.
        candidate_for_removal [-]: Edges absorbed by at least one other box.
        num_absorbers_per_edge [-]: Per-edge count of absorbing boxes.
        absorb_count_per_box [-]: Per-box count of absorbed edges.
        absorbed_offsets [-]: CSR offsets, length ``M + 1``.
        absorbed_indices [-]: CSR values; edges in box ``j`` live in
            ``absorbed_indices[absorbed_offsets[j]:absorbed_offsets[j+1]]``.
            Order within a slice is unspecified (filled by GPU atomics);
            :func:`resolve_edge_removals` is order-insensitive so the
            ``to_remove``/``kept`` masks it produces are still
            bit-deterministic.
        broad_phase_pair_count [-]: AABB pairs returned by SAP.
        aabb_diagonal [m]: Mesh world-space AABB diagonal.
        half_normal [m]: Box half-extent along the edge normal.
        half_lateral [m]: Box half-extent along the in-plane tangent and the
            per-end overhang along the edge direction.
        lower_angle_threshold_rad [rad]: Input gate that was applied. Edges
            below this dihedral angle were dropped before the broad phase.
        upper_angle_threshold_rad [rad]: Default absorption-eligibility
            threshold for :func:`resolve_edge_removals`.
    """

    edge_indices: np.ndarray
    dihedral_angles: np.ndarray
    adjacent_face_area_sum: np.ndarray
    candidate_for_removal: np.ndarray
    num_absorbers_per_edge: np.ndarray
    absorb_count_per_box: np.ndarray
    absorbed_offsets: np.ndarray
    absorbed_indices: np.ndarray
    broad_phase_pair_count: int
    aabb_diagonal: float
    half_normal: float
    half_lateral: float
    lower_angle_threshold_rad: float
    upper_angle_threshold_rad: float


# -----------------------------------------------------------------------------
# Warp kernels
# -----------------------------------------------------------------------------


@wp.kernel
def _build_edge_box_kernel(
    vertices: wp.array[wp.vec3],
    edge_indices: wp.array[wp.vec2i],
    avg_normals: wp.array[wp.vec3],
    half_normal: float,
    half_lateral: float,
    # Outputs
    box_center: wp.array[wp.vec3],
    box_axis_dir: wp.array[wp.vec3],
    box_axis_tang: wp.array[wp.vec3],
    box_axis_normal: wp.array[wp.vec3],
    box_half_extents: wp.array[wp.vec3],
    box_valid: wp.array[wp.int32],
):
    i = wp.tid()

    e = edge_indices[i]
    v0 = vertices[e[0]]
    v1 = vertices[e[1]]

    edge_vec = v1 - v0
    edge_len = wp.length(edge_vec)
    n = avg_normals[i]
    n_len = wp.length(n)

    # Degenerate edge or missing normal (NaN-filled) -> mark as no-box.
    if edge_len <= 1.0e-12 or n_len <= 1.0e-12 or wp.isnan(n[0]):
        box_center[i] = wp.vec3(0.0, 0.0, 0.0)
        box_axis_dir[i] = wp.vec3(1.0, 0.0, 0.0)
        box_axis_tang[i] = wp.vec3(0.0, 1.0, 0.0)
        box_axis_normal[i] = wp.vec3(0.0, 0.0, 1.0)
        box_half_extents[i] = wp.vec3(0.0, 0.0, 0.0)
        box_valid[i] = 0
        return

    dir_e = edge_vec / edge_len
    n_unit = n / n_len

    tang = wp.cross(n_unit, dir_e)
    tang_len = wp.length(tang)
    if tang_len <= 1.0e-12:
        box_center[i] = wp.vec3(0.0, 0.0, 0.0)
        box_axis_dir[i] = wp.vec3(1.0, 0.0, 0.0)
        box_axis_tang[i] = wp.vec3(0.0, 1.0, 0.0)
        box_axis_normal[i] = wp.vec3(0.0, 0.0, 1.0)
        box_half_extents[i] = wp.vec3(0.0, 0.0, 0.0)
        box_valid[i] = 0
        return

    tang = tang / tang_len
    # Re-orthogonalize the normal so the frame stays orthonormal even if
    # avg_normal is not exactly perpendicular to dir_e.
    normal = wp.cross(dir_e, tang)

    box_center[i] = 0.5 * (v0 + v1)
    box_axis_dir[i] = dir_e
    box_axis_tang[i] = tang
    box_axis_normal[i] = normal
    box_half_extents[i] = wp.vec3(0.5 * edge_len + half_lateral, half_lateral, half_normal)
    box_valid[i] = 1


@wp.kernel
def _compute_box_aabb_kernel(
    box_center: wp.array[wp.vec3],
    box_axis_dir: wp.array[wp.vec3],
    box_axis_tang: wp.array[wp.vec3],
    box_axis_normal: wp.array[wp.vec3],
    box_half_extents: wp.array[wp.vec3],
    box_valid: wp.array[wp.int32],
    # Outputs
    aabb_lower: wp.array[wp.vec3],
    aabb_upper: wp.array[wp.vec3],
):
    i = wp.tid()

    if box_valid[i] == 0:
        # Degenerate box collapsed to a single point so it is never overlapped.
        aabb_lower[i] = wp.vec3(1.0e30, 1.0e30, 1.0e30)
        aabb_upper[i] = wp.vec3(-1.0e30, -1.0e30, -1.0e30)
        return

    c = box_center[i]
    h = box_half_extents[i]
    rdir = box_axis_dir[i]
    rtan = box_axis_tang[i]
    rnor = box_axis_normal[i]

    # World half-extents via |R| * h (R has axes as columns).
    hx = wp.abs(rdir[0]) * h[0] + wp.abs(rtan[0]) * h[1] + wp.abs(rnor[0]) * h[2]
    hy = wp.abs(rdir[1]) * h[0] + wp.abs(rtan[1]) * h[1] + wp.abs(rnor[1]) * h[2]
    hz = wp.abs(rdir[2]) * h[0] + wp.abs(rtan[2]) * h[1] + wp.abs(rnor[2]) * h[2]
    world_half = wp.vec3(hx, hy, hz)

    aabb_lower[i] = c - world_half
    aabb_upper[i] = c + world_half


@wp.func
def _box_contains_point(
    p: wp.vec3,
    center: wp.vec3,
    axis_dir: wp.vec3,
    axis_tang: wp.vec3,
    axis_normal: wp.vec3,
    half_extents: wp.vec3,
    eps: float,
) -> int:
    d = p - center
    pd = wp.dot(d, axis_dir)
    pt = wp.dot(d, axis_tang)
    pn = wp.dot(d, axis_normal)
    inside = int(0)
    if (
        wp.abs(pd) <= half_extents[0] + eps
        and wp.abs(pt) <= half_extents[1] + eps
        and wp.abs(pn) <= half_extents[2] + eps
    ):
        inside = 1
    return inside


@wp.func
def _box_contains_edge(
    edge_idx: int,
    box_idx: int,
    vertices: wp.array[wp.vec3],
    edge_indices: wp.array[wp.vec2i],
    box_center: wp.array[wp.vec3],
    box_axis_dir: wp.array[wp.vec3],
    box_axis_tang: wp.array[wp.vec3],
    box_axis_normal: wp.array[wp.vec3],
    box_half_extents: wp.array[wp.vec3],
    box_valid: wp.array[wp.int32],
    eps: float,
) -> int:
    if box_valid[box_idx] == 0:
        return 0
    e = edge_indices[edge_idx]
    v0 = vertices[e[0]]
    v1 = vertices[e[1]]
    c = box_center[box_idx]
    rdir = box_axis_dir[box_idx]
    rtan = box_axis_tang[box_idx]
    rnor = box_axis_normal[box_idx]
    h = box_half_extents[box_idx]
    in0 = _box_contains_point(v0, c, rdir, rtan, rnor, h, eps)
    in1 = _box_contains_point(v1, c, rdir, rtan, rnor, h, eps)
    return in0 * in1


@wp.kernel
def _count_absorbed_per_box_kernel(
    candidate_pair: wp.array[wp.vec2i],
    candidate_pair_count: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    edge_indices: wp.array[wp.vec2i],
    box_center: wp.array[wp.vec3],
    box_axis_dir: wp.array[wp.vec3],
    box_axis_tang: wp.array[wp.vec3],
    box_axis_normal: wp.array[wp.vec3],
    box_half_extents: wp.array[wp.vec3],
    box_valid: wp.array[wp.int32],
    eps: float,
    # In/out
    absorb_count_per_box: wp.array[wp.int32],
    num_absorbers_per_edge: wp.array[wp.int32],
):
    pid = wp.tid()
    if pid >= candidate_pair_count[0]:
        return
    pair = candidate_pair[pid]
    a = pair[0]
    b = pair[1]
    if a == b:
        return

    contains_a_b = _box_contains_edge(
        b,
        a,
        vertices,
        edge_indices,
        box_center,
        box_axis_dir,
        box_axis_tang,
        box_axis_normal,
        box_half_extents,
        box_valid,
        eps,
    )
    contains_b_a = _box_contains_edge(
        a,
        b,
        vertices,
        edge_indices,
        box_center,
        box_axis_dir,
        box_axis_tang,
        box_axis_normal,
        box_half_extents,
        box_valid,
        eps,
    )
    if contains_a_b == 1:
        wp.atomic_add(absorb_count_per_box, a, 1)
        wp.atomic_add(num_absorbers_per_edge, b, 1)
    if contains_b_a == 1:
        wp.atomic_add(absorb_count_per_box, b, 1)
        wp.atomic_add(num_absorbers_per_edge, a, 1)


@wp.kernel
def _scatter_absorbed_per_box_kernel(
    candidate_pair: wp.array[wp.vec2i],
    candidate_pair_count: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    edge_indices: wp.array[wp.vec2i],
    box_center: wp.array[wp.vec3],
    box_axis_dir: wp.array[wp.vec3],
    box_axis_tang: wp.array[wp.vec3],
    box_axis_normal: wp.array[wp.vec3],
    box_half_extents: wp.array[wp.vec3],
    box_valid: wp.array[wp.int32],
    absorbed_offsets: wp.array[wp.int32],
    eps: float,
    # In/out
    write_cursor: wp.array[wp.int32],
    absorbed_indices: wp.array[wp.int32],
):
    pid = wp.tid()
    if pid >= candidate_pair_count[0]:
        return
    pair = candidate_pair[pid]
    a = pair[0]
    b = pair[1]
    if a == b:
        return

    contains_a_b = _box_contains_edge(
        b,
        a,
        vertices,
        edge_indices,
        box_center,
        box_axis_dir,
        box_axis_tang,
        box_axis_normal,
        box_half_extents,
        box_valid,
        eps,
    )
    contains_b_a = _box_contains_edge(
        a,
        b,
        vertices,
        edge_indices,
        box_center,
        box_axis_dir,
        box_axis_tang,
        box_axis_normal,
        box_half_extents,
        box_valid,
        eps,
    )
    if contains_a_b == 1:
        slot = wp.atomic_add(write_cursor, a, 1)
        absorbed_indices[absorbed_offsets[a] + slot] = b
    if contains_b_a == 1:
        slot = wp.atomic_add(write_cursor, b, 1)
        absorbed_indices[absorbed_offsets[b] + slot] = a


@wp.kernel
def _mark_candidates_kernel(
    num_absorbers_per_edge: wp.array[wp.int32],
    candidate_for_removal: wp.array[wp.int32],
):
    i = wp.tid()
    if num_absorbers_per_edge[i] > 0:
        candidate_for_removal[i] = 1
    else:
        candidate_for_removal[i] = 0


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------


def find_redundant_edges(
    mesh,
    *,
    enable_box_absorption: bool = False,
    half_normal: float | None = None,
    half_lateral: float | None = None,
    lower_angle_threshold_rad: float = math.radians(0.1),
    upper_angle_threshold_rad: float = math.radians(10.0),
    initial_pair_capacity_factor: int = 8,
    max_retries: int = 3,
    device=None,
) -> EdgeRedundancyResult:
    """Apply the dihedral-angle pre-filter and (optionally) the box-absorption pass.

    The pre-filter always runs and returns the manifold-edge subset whose
    dihedral angle is at least ``lower_angle_threshold_rad``. Box-absorption
    only runs when ``enable_box_absorption`` is ``True``; otherwise the
    absorption-related fields on the result are zero-initialised.

    Args:
        mesh: A :class:`newton.Mesh` instance.
        enable_box_absorption: When ``True``, build oriented edge boxes and
            populate ``candidate_for_removal`` / CSR adjacency. When
            ``False`` (default), only the dihedral-angle pre-filter runs.
        half_normal [m]: Box half-extent along the edge normal. Defaults to
            ``1e-3 * D`` (``D`` = mesh AABB diagonal). Ignored when
            ``enable_box_absorption`` is ``False``.
        half_lateral [m]: Box half-extent along the in-plane tangent and the
            per-end overhang along the edge. Defaults to ``5e-3 * D``.
            Ignored when ``enable_box_absorption`` is ``False``.
        lower_angle_threshold_rad [rad]: Input gate. Manifold edges with
            dihedral angle below this value are excluded. Default
            ``math.radians(0.1)`` (0.1 deg); set to 0 to keep every
            manifold edge.
        upper_angle_threshold_rad [rad]: Stored on the result as the default
            absorption-eligibility threshold for :func:`resolve_edge_removals`.
            Default 10 deg.
        initial_pair_capacity_factor: SAP pair-buffer size in multiples of
            the manifold-edge count.
        max_retries: Max grow-on-overflow attempts for the SAP pair buffer.
        device: Optional Warp device.
    """
    edges_np, angles_np, normals_np, area_sums_np = mesh._filter_edges_by_dihedral_angle(
        lower_angle_threshold_rad, return_diagnostics=True
    )

    # Manifold edges have finite per-edge diagnostics; non-pair edges hold NaN.
    manifold_mask = np.isfinite(angles_np) & np.all(np.isfinite(normals_np), axis=1)
    edge_indices_np = edges_np[manifold_mask].astype(np.int32, copy=False)
    edge_angles_np = angles_np[manifold_mask].astype(np.float32, copy=False)
    edge_normals_np = normals_np[manifold_mask].astype(np.float32, copy=False)
    edge_area_sums_np = area_sums_np[manifold_mask].astype(np.float32, copy=False)
    n_edges = int(len(edge_indices_np))

    vertices_np = np.asarray(mesh.vertices, dtype=np.float32)

    aabb_min = vertices_np.min(axis=0) if len(vertices_np) > 0 else np.zeros(3, dtype=np.float32)
    aabb_max = vertices_np.max(axis=0) if len(vertices_np) > 0 else np.zeros(3, dtype=np.float32)
    diagonal = float(np.linalg.norm(aabb_max - aabb_min))

    resolved_half_normal = float(half_normal) if half_normal is not None else 1.0e-3 * diagonal
    resolved_half_lateral = float(half_lateral) if half_lateral is not None else 5.0e-3 * diagonal

    # Fast path: absorption disabled by caller, no edges, or non-positive extents.
    boxes_disabled = resolved_half_normal <= 0.0 or resolved_half_lateral <= 0.0
    if not enable_box_absorption or n_edges == 0 or boxes_disabled:
        return EdgeRedundancyResult(
            edge_indices=edge_indices_np.reshape(-1, 2),
            dihedral_angles=edge_angles_np,
            adjacent_face_area_sum=edge_area_sums_np,
            candidate_for_removal=np.zeros(n_edges, dtype=bool),
            num_absorbers_per_edge=np.zeros(n_edges, dtype=np.int32),
            absorb_count_per_box=np.zeros(n_edges, dtype=np.int32),
            absorbed_offsets=np.zeros(n_edges + 1, dtype=np.int32),
            absorbed_indices=np.zeros(0, dtype=np.int32),
            broad_phase_pair_count=0,
            aabb_diagonal=diagonal,
            half_normal=resolved_half_normal,
            half_lateral=resolved_half_lateral,
            lower_angle_threshold_rad=float(lower_angle_threshold_rad),
            upper_angle_threshold_rad=float(upper_angle_threshold_rad),
        )

    if device is None:
        device = wp.get_preferred_device()

    with wp.ScopedDevice(device):
        vertices_wp = wp.array(vertices_np, dtype=wp.vec3)
        edge_indices_wp = wp.array(edge_indices_np.reshape(-1, 2), dtype=wp.vec2i)
        avg_normals_wp = wp.array(edge_normals_np.reshape(-1, 3), dtype=wp.vec3)

        box_center = wp.empty(n_edges, dtype=wp.vec3)
        box_axis_dir = wp.empty(n_edges, dtype=wp.vec3)
        box_axis_tang = wp.empty(n_edges, dtype=wp.vec3)
        box_axis_normal = wp.empty(n_edges, dtype=wp.vec3)
        box_half_extents = wp.empty(n_edges, dtype=wp.vec3)
        box_valid = wp.zeros(n_edges, dtype=wp.int32)

        wp.launch(
            kernel=_build_edge_box_kernel,
            dim=n_edges,
            inputs=[
                vertices_wp,
                edge_indices_wp,
                avg_normals_wp,
                resolved_half_normal,
                resolved_half_lateral,
            ],
            outputs=[
                box_center,
                box_axis_dir,
                box_axis_tang,
                box_axis_normal,
                box_half_extents,
                box_valid,
            ],
        )

        aabb_lower = wp.empty(n_edges, dtype=wp.vec3)
        aabb_upper = wp.empty(n_edges, dtype=wp.vec3)
        wp.launch(
            kernel=_compute_box_aabb_kernel,
            dim=n_edges,
            inputs=[
                box_center,
                box_axis_dir,
                box_axis_tang,
                box_axis_normal,
                box_half_extents,
                box_valid,
            ],
            outputs=[aabb_lower, aabb_upper],
        )

        shape_world_np = np.zeros(n_edges, dtype=np.int32)
        shape_collision_group_np = np.ones(n_edges, dtype=np.int32)
        shape_flags_np = np.full(n_edges, int(ShapeFlags.COLLIDE_SHAPES), dtype=np.int32)
        # Drop COLLIDE_SHAPES on degenerate edges so SAP skips them entirely.
        valid_host = box_valid.numpy()
        shape_flags_np[valid_host == 0] = 0

        shape_world_wp = wp.array(shape_world_np, dtype=wp.int32)
        shape_collision_group_wp = wp.array(shape_collision_group_np, dtype=wp.int32)
        shape_flags_wp = wp.array(shape_flags_np, dtype=wp.int32)

        sap = BroadPhaseSAP(shape_world=shape_world_wp, shape_flags=shape_flags_wp)

        candidate_pair_count = wp.zeros(1, dtype=wp.int32)
        capacity = max(64, initial_pair_capacity_factor * n_edges)
        attempts = 0
        actual_pair_count = 0
        candidate_pair: wp.array | None = None
        while True:
            candidate_pair = wp.empty(capacity, dtype=wp.vec2i)
            sap.launch(
                shape_lower=aabb_lower,
                shape_upper=aabb_upper,
                shape_gap=None,
                shape_collision_group=shape_collision_group_wp,
                shape_world=shape_world_wp,
                shape_count=n_edges,
                candidate_pair=candidate_pair,
                candidate_pair_count=candidate_pair_count,
            )
            actual_pair_count = int(candidate_pair_count.numpy()[0])
            if actual_pair_count <= capacity:
                break
            attempts += 1
            if attempts > max_retries:
                # SAP truncates; surface via broad_phase_pair_count == capacity.
                actual_pair_count = capacity
                break
            capacity = max(actual_pair_count, capacity * 2)

        assert candidate_pair is not None

        eps = 1.0e-6 * max(diagonal, 1.0e-6)

        absorb_count_per_box = wp.zeros(n_edges, dtype=wp.int32)
        num_absorbers_per_edge = wp.zeros(n_edges, dtype=wp.int32)

        if actual_pair_count > 0:
            wp.launch(
                kernel=_count_absorbed_per_box_kernel,
                dim=actual_pair_count,
                inputs=[
                    candidate_pair,
                    candidate_pair_count,
                    vertices_wp,
                    edge_indices_wp,
                    box_center,
                    box_axis_dir,
                    box_axis_tang,
                    box_axis_normal,
                    box_half_extents,
                    box_valid,
                    eps,
                ],
                outputs=[absorb_count_per_box, num_absorbers_per_edge],
            )

        # Exclusive scan -> CSR offsets (inclusive scan into offsets[1:]).
        absorbed_offsets = wp.zeros(n_edges + 1, dtype=wp.int32)
        if n_edges > 0:
            wp.utils.array_scan(absorb_count_per_box, absorbed_offsets[1:], inclusive=True)

        offsets_host = absorbed_offsets.numpy()
        total_pairs = int(offsets_host[-1])
        absorbed_indices = wp.zeros(max(total_pairs, 1), dtype=wp.int32)
        write_cursor = wp.zeros(n_edges, dtype=wp.int32)

        if actual_pair_count > 0 and total_pairs > 0:
            wp.launch(
                kernel=_scatter_absorbed_per_box_kernel,
                dim=actual_pair_count,
                inputs=[
                    candidate_pair,
                    candidate_pair_count,
                    vertices_wp,
                    edge_indices_wp,
                    box_center,
                    box_axis_dir,
                    box_axis_tang,
                    box_axis_normal,
                    box_half_extents,
                    box_valid,
                    absorbed_offsets,
                    eps,
                ],
                outputs=[write_cursor, absorbed_indices],
            )

        candidate_for_removal = wp.zeros(n_edges, dtype=wp.int32)
        wp.launch(
            kernel=_mark_candidates_kernel,
            dim=n_edges,
            inputs=[num_absorbers_per_edge],
            outputs=[candidate_for_removal],
        )

        candidate_host = candidate_for_removal.numpy().astype(bool)
        num_absorbers_host = num_absorbers_per_edge.numpy()
        absorb_count_host = absorb_count_per_box.numpy()
        absorbed_indices_host = absorbed_indices.numpy()[:total_pairs]

    return EdgeRedundancyResult(
        edge_indices=edge_indices_np.reshape(-1, 2),
        dihedral_angles=edge_angles_np,
        adjacent_face_area_sum=edge_area_sums_np,
        candidate_for_removal=candidate_host,
        num_absorbers_per_edge=num_absorbers_host,
        absorb_count_per_box=absorb_count_host,
        absorbed_offsets=offsets_host,
        absorbed_indices=absorbed_indices_host,
        broad_phase_pair_count=actual_pair_count,
        aabb_diagonal=diagonal,
        half_normal=resolved_half_normal,
        half_lateral=resolved_half_lateral,
        lower_angle_threshold_rad=float(lower_angle_threshold_rad),
        upper_angle_threshold_rad=float(upper_angle_threshold_rad),
    )


# -----------------------------------------------------------------------------
# Greedy CPU-side resolution of removal candidates
# -----------------------------------------------------------------------------


@dataclass
class EdgeResolutionResult:
    """Per-edge greedy decision returned by :func:`resolve_edge_removals`.

    Indices are aligned with :attr:`EdgeRedundancyResult.edge_indices`.

    Attributes:
        to_remove [-]: Edges scheduled for definitive removal.
        kept [-]: Containers promoted to "definitely keep" during the greedy
            pass. Always disjoint from :attr:`to_remove`.
        order [-]: Box order used by the greedy loop (descending by
            ``absorb_count_per_box``).
        upper_angle_threshold_rad [rad]: Absorption-eligibility threshold
            that was applied.
    """

    to_remove: np.ndarray
    kept: np.ndarray
    order: np.ndarray
    upper_angle_threshold_rad: float


def resolve_edge_removals(
    result: EdgeRedundancyResult,
    *,
    upper_angle_threshold_rad: float | None = None,
) -> EdgeResolutionResult:
    """Greedy CPU resolution of edge-removal candidates.

    Walks boxes from highest to lowest ``absorb_count_per_box``. For each box:

    1. Skip if the container edge is already scheduled for removal.
    2. Otherwise, promote the container edge to "definitely keep".
    3. Mark each absorbed edge for removal iff its dihedral angle is below
       ``upper_angle_threshold_rad`` and it is not already kept.

    Args:
        result: Output of :func:`find_redundant_edges`.
        upper_angle_threshold_rad [rad]: Upper bound on the dihedral angle of
            a removable edge. Defaults to
            ``result.upper_angle_threshold_rad``.
    """
    if upper_angle_threshold_rad is None:
        upper_angle_threshold_rad = result.upper_angle_threshold_rad
    threshold = float(upper_angle_threshold_rad)

    n = len(result.edge_indices)
    to_remove = np.zeros(n, dtype=bool)
    kept = np.zeros(n, dtype=bool)
    if n == 0:
        return EdgeResolutionResult(
            to_remove=to_remove,
            kept=kept,
            order=np.zeros(0, dtype=np.int32),
            upper_angle_threshold_rad=threshold,
        )

    absorb_count = result.absorb_count_per_box.astype(np.int64, copy=False)
    # Stable descending sort by (absorb_count, adjacent_face_area_sum). The
    # area sum breaks ties on absorb count: when two boxes absorb the same
    # number of others, prefer the one adjacent to larger triangles (heuristic
    # that favours the load-bearing geometry). np.lexsort treats the *last*
    # key as primary, so we put -absorb_count last.
    area_sum = result.adjacent_face_area_sum.astype(np.float64, copy=False)
    order = np.lexsort((-area_sum, -absorb_count)).astype(np.int32, copy=False)

    offsets = result.absorbed_offsets
    indices = result.absorbed_indices
    angles = result.dihedral_angles

    for box_idx in order:
        if absorb_count[box_idx] == 0:
            break
        if to_remove[box_idx]:
            continue

        kept[box_idx] = True

        lo = int(offsets[box_idx])
        hi = int(offsets[box_idx + 1])
        if hi <= lo:
            continue
        absorbed = indices[lo:hi]
        flag = (angles[absorbed] < threshold) & (~kept[absorbed])
        if np.any(flag):
            to_remove[absorbed[flag]] = True

    assert not np.any(kept & to_remove), "kept and to_remove overlap"

    return EdgeResolutionResult(
        to_remove=to_remove,
        kept=kept,
        order=order,
        upper_angle_threshold_rad=threshold,
    )


def remove_redundant_edges(
    mesh,
    *,
    enable_box_absorption: bool = False,
    half_normal: float | None = None,
    half_lateral: float | None = None,
    lower_angle_threshold_rad: float = math.radians(0.1),
    upper_angle_threshold_rad: float = math.radians(10.0),
    initial_pair_capacity_factor: int = 8,
    max_retries: int = 3,
    device=None,
) -> np.ndarray:
    """One-shot wrapper: find redundant edges and return only the kept set.

    Chains :func:`find_redundant_edges` and :func:`resolve_edge_removals`.
    Use the two-step API instead if you need the intermediate diagnostics.

    All keyword arguments are forwarded to :func:`find_redundant_edges`; see
    that function for full semantics.

    Returns:
        Kept manifold-edge vertex pairs ``(M, 2)`` with dtype ``int32``.
    """
    result = find_redundant_edges(
        mesh,
        enable_box_absorption=enable_box_absorption,
        half_normal=half_normal,
        half_lateral=half_lateral,
        lower_angle_threshold_rad=lower_angle_threshold_rad,
        upper_angle_threshold_rad=upper_angle_threshold_rad,
        initial_pair_capacity_factor=initial_pair_capacity_factor,
        max_retries=max_retries,
        device=device,
    )
    resolution = resolve_edge_removals(result)
    return result.edge_indices[~resolution.to_remove]


__all__ = [
    "EdgeRedundancyResult",
    "EdgeResolutionResult",
    "find_redundant_edges",
    "remove_redundant_edges",
    "resolve_edge_removals",
]
