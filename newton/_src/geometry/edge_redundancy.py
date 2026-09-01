# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Redundant-edge detection: dihedral-angle pre-filter + opt-in box absorption.

For each surviving manifold edge we build an oriented box in the edge frame
(``dir``, ``tang``, ``normal``), broad-phase via SAP, and test segment-in-box
exactly. :func:`resolve_edge_removals` then picks kept/removed greedily.
Output adjacency is CSR (absorbed edges per box).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from .broad_phase_sap import BroadPhaseSAP
from .flags import ShapeFlags

# Vectors with length below this are treated as zero (matches
# :mod:`raycast` / :mod:`collision_primitive`).
MINVAL = 1.0e-15

# -----------------------------------------------------------------------------
# Result type
# -----------------------------------------------------------------------------


@dataclass
class EdgeRedundancyResult:
    """Per-edge containment results from :func:`find_redundant_edges`.

    All per-edge arrays are indexed by ``edge_indices`` (the manifold subset),
    not the original :attr:`Mesh.edges` rows.

    Sharp edges (dihedral angle ``>= upper_angle_threshold_rad``) may act as
    containers but are never absorbed: they never appear in ``absorbed_indices``
    and their ``num_absorbers_per_edge`` / ``candidate_for_removal`` are zero.

    Attributes:
        edge_indices [-]: Manifold edge vertex pairs ``(M, 2)``.
        dihedral_angles [rad]: Per-edge dihedral angles.
        adjacent_face_area_sum [m^2]: Sum of the two adjacent triangle areas.
            Tiebreaker for :func:`resolve_edge_removals` (larger wins).
        candidate_for_removal [-]: Edges absorbed by at least one other box.
        num_absorbers_per_edge [-]: Per-edge count of absorbing boxes.
        absorb_count_per_box [-]: Per-box count of absorbed edges.
        absorbed_offsets [-]: CSR offsets, length ``M + 1``.
        absorbed_indices [-]: CSR values; edges in box ``j`` live in
            ``absorbed_indices[absorbed_offsets[j]:absorbed_offsets[j+1]]``.
            Intra-slice order is unspecified (GPU atomics);
            :func:`resolve_edge_removals` is order-insensitive.
        broad_phase_pair_count [-]: AABB pairs returned by SAP.
        aabb_diagonal [m]: Mesh world-space AABB diagonal.
        half_normal [m]: Box half-extent along the edge normal.
        half_lateral [m]: Box half-extent in-plane (across the edge and as
            per-end overhang along it).
        lower_angle_threshold_rad [rad]: Edges below this were excluded
            before the broad phase.
        upper_angle_threshold_rad [rad]: Absorbability gate applied in the
            kernels; also the default for :func:`resolve_edge_removals`.
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

    # Degenerate edge or NaN-filled normal -> no box.
    if edge_len <= wp.static(MINVAL) or n_len <= wp.static(MINVAL) or wp.isnan(n[0]):
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
    if tang_len <= wp.static(MINVAL):
        box_center[i] = wp.vec3(0.0, 0.0, 0.0)
        box_axis_dir[i] = wp.vec3(1.0, 0.0, 0.0)
        box_axis_tang[i] = wp.vec3(0.0, 1.0, 0.0)
        box_axis_normal[i] = wp.vec3(0.0, 0.0, 1.0)
        box_half_extents[i] = wp.vec3(0.0, 0.0, 0.0)
        box_valid[i] = 0
        return

    tang = tang / tang_len
    # Re-orthogonalize: avg_normal isn't guaranteed perpendicular to dir_e.
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
        # Inverted AABB so SAP never overlaps it.
        aabb_lower[i] = wp.vec3(1.0e30, 1.0e30, 1.0e30)
        aabb_upper[i] = wp.vec3(-1.0e30, -1.0e30, -1.0e30)
        return

    c = box_center[i]
    h = box_half_extents[i]
    rdir = box_axis_dir[i]
    rtan = box_axis_tang[i]
    rnor = box_axis_normal[i]

    # World half-extents = |R| * h, with R = [dir | tang | normal].
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
    is_absorbable: wp.array[wp.int32],
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

    # Gate on the absorbee, not the absorber: sharp edges may contain
    # others but must not be marked absorbed themselves.
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
    if contains_a_b == 1 and is_absorbable[b] == 1:
        wp.atomic_add(absorb_count_per_box, a, 1)
        wp.atomic_add(num_absorbers_per_edge, b, 1)
    if contains_b_a == 1 and is_absorbable[a] == 1:
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
    is_absorbable: wp.array[wp.int32],
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

    # Must agree with the count kernel's gate so CSR slot counts match.
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
    if contains_a_b == 1 and is_absorbable[b] == 1:
        slot = wp.atomic_add(write_cursor, a, 1)
        absorbed_indices[absorbed_offsets[a] + slot] = b
    if contains_b_a == 1 and is_absorbable[a] == 1:
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
    precomputed_filter: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> EdgeRedundancyResult:
    """Run the dihedral-angle pre-filter and, optionally, box absorption.

    The pre-filter always runs. Box absorption only runs when
    ``enable_box_absorption`` is ``True``; otherwise the absorption-related
    fields on the result are zero-initialised.

    Args:
        mesh: A :class:`newton.Mesh` instance.
        enable_box_absorption: Build oriented edge boxes and populate the
            CSR adjacency. Defaults to ``False`` (pre-filter only).
        half_normal [m]: Box half-extent along the edge normal. Defaults
            to ``1e-3 * D`` where ``D`` is the mesh AABB diagonal.
        half_lateral [m]: Box half-extent in-plane (across the edge and as
            per-end overhang along it). Defaults to ``5e-3 * D``.
        lower_angle_threshold_rad [rad]: Manifold edges below this
            dihedral angle are excluded. Set to 0 to keep all manifold
            edges. Default 0.1 deg.
        upper_angle_threshold_rad [rad]: Sharp-edge cutoff. Edges at or
            above this angle act as absorbers only; they are excluded
            from ``absorb_count_per_box``, ``num_absorbers_per_edge``,
            ``candidate_for_removal``, and the CSR adjacency. Default 10
            deg. Also stored on the result as the default for
            :func:`resolve_edge_removals`.
        initial_pair_capacity_factor: SAP pair-buffer size in multiples of
            the manifold-edge count.
        max_retries: Max grow-on-overflow attempts for the SAP pair buffer.
        device: Optional Warp device.
        precomputed_filter: Optional ``(edges, angles, avg_normals,
            area_sums)`` from a previous
            :meth:`Mesh._filter_edges_by_dihedral_angle` call at the same
            ``lower_angle_threshold_rad``. Reused verbatim to skip the
            dihedral filter pass.
    """
    if precomputed_filter is not None:
        edges_np, angles_np, normals_np, area_sums_np = precomputed_filter
    else:
        edges_np, angles_np, normals_np, area_sums_np = mesh._filter_edges_by_dihedral_angle(
            lower_angle_threshold_rad, return_diagnostics=True
        )

    # Non-manifold (boundary / 3+-incident) edges carry NaN diagnostics.
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

    # Fast path: absorption disabled, no edges, or zero-extent boxes.
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
        # Clear the collide flag on degenerate boxes so SAP skips them.
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
                # SAP truncated; report via broad_phase_pair_count == capacity.
                actual_pair_count = capacity
                break
            capacity = max(actual_pair_count, capacity * 2)

        assert candidate_pair is not None

        eps = 1.0e-6 * max(diagonal, 1.0e-6)

        # Sharp edges (angle >= upper threshold) act as containers only.
        # Gating in the kernels keeps absorb counts and the CSR adjacency
        # in sync with this rule, so the greedy resolver isn't biased by
        # absorbed-but-unremovable sharp neighbours inflating counts.
        upper_threshold_f = float(upper_angle_threshold_rad)
        is_absorbable_np = (edge_angles_np < upper_threshold_f).astype(np.int32)
        is_absorbable_wp = wp.array(is_absorbable_np, dtype=wp.int32)

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
                    is_absorbable_wp,
                    eps,
                ],
                outputs=[absorb_count_per_box, num_absorbers_per_edge],
            )

        # Exclusive scan = inclusive scan written to offsets[1:].
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
                    is_absorbable_wp,
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
    """Per-edge greedy decision from :func:`resolve_edge_removals`.

    Indices align with :attr:`EdgeRedundancyResult.edge_indices`.

    Attributes:
        to_remove [-]: Edges scheduled for removal.
        kept [-]: Container edges promoted to "definitely keep". Disjoint
            from :attr:`to_remove`.
        order [-]: Box order used by the greedy loop (descending by
            ``absorb_count_per_box``, area-sum tiebreaker).
        upper_angle_threshold_rad [rad]: Threshold that was applied.
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

    Walks boxes from highest to lowest ``absorb_count_per_box``. For each:

    1. Skip if already scheduled for removal.
    2. Otherwise mark the container as kept.
    3. Mark every absorbed edge for removal, unless it has been kept or
       its dihedral angle is above ``upper_angle_threshold_rad``.

    The primary absorbability gate runs in the kernels; the angle check
    here only matters when a caller passes a *stricter* threshold than
    the one baked into the result. A looser threshold has no effect.

    Args:
        result: Output of :func:`find_redundant_edges`.
        upper_angle_threshold_rad [rad]: Upper bound on the dihedral angle
            of a removable edge. Defaults to
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
    # Descending sort on absorb count, with adjacent area as tiebreaker
    # (prefer load-bearing geometry). np.lexsort treats the last key as
    # primary, so -absorb_count goes last.
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
    """Chain :func:`find_redundant_edges` and :func:`resolve_edge_removals`.

    Use the two-step API directly if you need the intermediate diagnostics.
    Keyword arguments are forwarded verbatim to :func:`find_redundant_edges`.

    Returns:
        Kept manifold-edge vertex pairs ``(M, 2)``, dtype ``int32``.
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
