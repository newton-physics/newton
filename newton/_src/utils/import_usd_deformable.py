# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""USD deformable importer passes and their leaf helpers.

This module owns the cable / cloth / volume / attachment / collision-filter import passes:
they traverse the stage, mutate the :class:`~newton.ModelBuilder`, and populate the
``path_*`` result maps via a :class:`_DeformableImportContext` carrying that shared state.
:func:`parse_usd` in :mod:`.import_usd` builds the context and drives the passes in order.
A few leaf helpers (e.g. :func:`validate_mass_array`) are pure and builder-independent.
"""

from __future__ import annotations

import copy
import math
import re
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from pxr import Usd

    from ..sim.builder import ModelBuilder


def validate_mass_array(values: Iterable[float], path: str) -> list[float] | None:
    """Validate an authored per-point ``physics:masses`` array.

    Per-point masses have the highest precedence in the deformable mass resolution, so a poisoned
    value would dominate. Returns the masses as floats when all are finite and non-negative; warns
    and returns ``None`` (so the caller falls back to body / material mass) if any value is
    non-finite or negative, or the array is empty.
    """
    masses = [float(x) for x in values]
    if not masses:
        return None
    if any((not math.isfinite(m)) or m < 0.0 for m in masses):
        warnings.warn(
            f"{path}: physics:masses contains non-finite or negative values; ignoring per-point masses.",
            stacklevel=2,
        )
        return None
    return masses


def _is_ignored_path(path: str, ignore_paths: Sequence[str]) -> bool:
    """Return whether ``path`` matches any of the ``ignore_paths`` regular expressions."""
    return any(re.match(pattern, path) for pattern in ignore_paths)


def _validate_attachment_index_pairs(
    indices0: Sequence[int], count0: int, indices1: Sequence[int], count1: int, path: str
) -> bool:
    """Validate a curve-to-curve junction's paired control-point indices.

    The two index arrays pair element-wise (``indices0[k]`` welds to ``indices1[k]``), so they
    must be non-empty, equal length, and each in range for its source curve's point count.
    Warns and returns ``False`` for a malformed junction so the caller can skip it instead of
    welding unintended points or raising ``IndexError``.
    """
    if not indices0 or not indices1:
        warnings.warn(
            f"{path}: curve-to-curve PhysicsAttachment has empty indices0/indices1; skipping junction.",
            stacklevel=2,
        )
        return False
    if len(indices0) != len(indices1):
        warnings.warn(
            f"{path}: curve-to-curve PhysicsAttachment indices0 (len {len(indices0)}) and indices1 "
            f"(len {len(indices1)}) differ in length; skipping junction.",
            stacklevel=2,
        )
        return False
    for indices, count, which in ((indices0, count0, "src0"), (indices1, count1, "src1")):
        for idx in indices:
            if idx < 0 or idx >= count:
                warnings.warn(
                    f"{path}: curve-to-curve PhysicsAttachment {which} index {idx} is out of range for its "
                    f"curve ({count} points); skipping junction.",
                    stacklevel=2,
                )
                return False
    return True


@dataclass
class _CurveDeformableRecord:
    """A single linear curve deformable eligible for rod-graph welding.

    Positions are already in world space (import transform applied). ``material`` holds
    the authored curve-deformable material values (see
    :func:`.usd.utils._get_curve_deformable_material`); ``radius`` and ``density`` are the
    resolved per-curve values.
    """

    prim: Usd.Prim
    positions: list[wp.vec3]
    closed: bool
    radius: float
    density: float
    material: dict[str, float] = field(default_factory=dict)


def _cable_segment_quaternions(seg_positions: Sequence[wp.vec3], seg_normals: Sequence[wp.vec3]) -> list[wp.quat]:
    """Per-segment capsule orientations for an imported cable.

    Builds one quaternion per segment that maps local ``+Z`` to the segment tangent and local
    ``+Y`` to the authored (world-space) normal; a degenerate normal falls back to a roll-free
    frame. Callers skip zero-length segments, so each segment length is positive here.
    """
    from ..math import quat_between_vectors_robust  # noqa: PLC0415

    z_local = wp.vec3(0.0, 0.0, 1.0)
    y_local = wp.vec3(0.0, 1.0, 0.0)
    eps = 1.0e-8
    quats: list[wp.quat] = []
    for i in range(len(seg_positions) - 1):
        seg = seg_positions[i + 1] - seg_positions[i]
        seg_len = float(wp.length(seg))
        tangent = seg / seg_len
        q = quat_between_vectors_robust(z_local, tangent, eps)
        n_perp = seg_normals[i] - wp.dot(seg_normals[i], tangent) * tangent
        n_len = float(wp.length(n_perp))
        if n_len > eps:
            n_perp = n_perp / n_len
            y0 = wp.quat_rotate(q, y_local)
            roll = math.atan2(float(wp.dot(wp.cross(y0, n_perp), tangent)), float(wp.dot(y0, n_perp)))
            q = wp.mul(wp.quat_from_axis_angle(tangent, roll), q)
        quats.append(q)
    return quats


def _attachment_vec3_list(value) -> list[wp.vec3]:
    """Convert an authored ``coords`` array (or ``None``) to a list of :class:`warp.vec3`."""
    if value is None:
        return []
    return [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in value]


def _attachment_vec3_tuples(values: Sequence[wp.vec3]) -> list[tuple[float, float, float]]:
    """Convert :class:`warp.vec3` values back to plain float tuples for the returned attrs."""
    return [(float(v[0]), float(v[1]), float(v[2])) for v in values]


def _mark_attachment_unsupported(attrs: dict, path: str, reason: str) -> None:
    """Record why a ``PhysicsAttachment`` was not imported and warn, preserving its attrs."""
    attrs["unsupported_reason"] = reason
    warnings.warn(f"{path}: {reason}", stacklevel=2)


def _warn_unsupported_rest_fields(prim: Usd.Prim, path: str, names: Sequence[str], read_attr: Callable) -> None:
    """Warn (once) if any authored rest-state field in ``names`` is present but not yet imported.

    Rest-state import (rest shape, rest dihedral angles) is not implemented yet; warn rather than
    silently drop an authored rest configuration.
    """
    for name in names:
        if read_attr(prim, name) is not None:
            warnings.warn(
                f"{path}: 'physics:{name}' is authored but its import is not yet supported; it is ignored.",
                stacklevel=2,
            )
            return


def _warn_geometry_authored_material_attrs(prim: Usd.Prim, path: str, material_api: str, read_attr: Callable) -> None:
    """Warn for deformable material moduli authored on the geometry instead of the bound material.

    The proposal scopes these moduli to the deformable material APIs, so authoring them on the
    geometry has no effect; warn rather than drop them silently. ``density`` is excluded since it
    may legitimately sit on the body (``PhysicsDeformableBodyAPI``).
    """
    for name in (
        "youngsModulus",
        "poissonsRatio",
        "stretchStiffness",
        "shearStiffness",
        "bendStiffness",
        "twistStiffness",
        "thickness",
    ):
        if read_attr(prim, name) is not None:
            warnings.warn(
                f"{path}: deformable material attribute 'physics:{name}' is authored on the geometry; "
                f"it belongs on the bound material ({material_api}) and is ignored.",
                stacklevel=2,
            )


def _builder_body_xform(builder: ModelBuilder, body_id: int) -> wp.transform:
    """Return body ``body_id``'s current world transform from the builder's ``body_q``."""
    body_q = builder.body_q[body_id]
    return wp.transform(
        wp.vec3(float(body_q[0]), float(body_q[1]), float(body_q[2])),
        wp.quat(float(body_q[3]), float(body_q[4]), float(body_q[5]), float(body_q[6])),
    )


def _resolve_deformable_density(prim: Usd.Prim, material_density: float | None, read_attr: Callable) -> float | None:
    """Resolve the density used for a deformable.

    Mass precedence (proposal): a ``PhysicsDeformableBodyAPI`` body-density override takes
    precedence over the bound material's density.
    """
    from ..usd import utils as usd  # noqa: PLC0415

    _, body_density = usd._get_deformable_body_overrides(prim, read_attr)
    return body_density if body_density is not None else material_density


def _set_body_mass(builder: ModelBuilder, b: int, m: float) -> None:
    """Set body ``b``'s mass and scale its inertia tensor to match (keeps the segment's shape)."""
    orig = builder.body_mass[b]
    s = (m / orig) if orig > 0.0 else 0.0
    builder.body_mass[b] = m
    builder.body_inertia[b] = builder.body_inertia[b] * s
    builder.body_inv_mass[b] = (1.0 / m) if m > 0.0 else 0.0
    builder.body_inv_inertia[b] = wp.inverse(builder.body_inertia[b]) if m > 0.0 else wp.mat33(0.0)


def _apply_particle_masses(builder: ModelBuilder, prim: Usd.Prim, p0: int, p1: int, read_attr: Callable) -> None:
    """Apply the deformable mass override to particles ``[p0, p1)``.

    Per-point ``physics:masses`` (highest precedence) are written directly; otherwise a
    ``PhysicsDeformableBodyAPI`` body-mass total rescales the (density-derived) particle masses.
    """
    from ..usd import utils as usd  # noqa: PLC0415

    n = p1 - p0
    if n <= 0:
        return
    point_masses = usd._get_deformable_point_masses(prim, read_attr)
    if point_masses is not None:
        if len(point_masses) != n:
            warnings.warn(
                f"{prim.GetPath()}: physics:masses length {len(point_masses)} != {n} simulation points; "
                f"ignoring per-point masses.",
                stacklevel=2,
            )
        else:
            for i in range(n):
                builder.particle_mass[p0 + i] = point_masses[i]
            return
    body_mass, _ = usd._get_deformable_body_overrides(prim, read_attr)
    if body_mass is not None:
        current = float(sum(builder.particle_mass[p0:p1]))
        if current > 0.0:
            scale = body_mass / current
            for i in range(p0, p1):
                builder.particle_mass[i] *= scale


def _apply_cable_masses(
    builder: ModelBuilder,
    prim: Usd.Prim,
    body_ids: Sequence[int],
    point_runs: Sequence[tuple[int, int, Sequence[int]]],
    closed: bool,
    read_attr: Callable,
) -> None:
    """Distribute the deformable mass override over a rigid cable's segment bodies.

    Mass precedence for the rigid cable: per-point ``physics:masses`` (highest), else a
    ``PhysicsDeformableBodyAPI`` body-mass total.

    ``physics:masses`` is a per-POINT (vertex) quantity, but add_rod builds one capsule body
    per SEGMENT between consecutive points -- a point is the junction of its neighboring
    segments, not a body. So N points map to N-1 segments (open) or N (closed)::

        points     P0------P1------P2------P3      mass m0 m1 m2 m3
        segments       C0      C1      C2          (open: 3 capsule bodies)

    There is no body at a vertex, so each point's mass is lumped onto the segment(s) it
    borders: an interior point splits its mass between its two segments, an endpoint gives
    its full mass to its single segment (so the total is conserved)::

        C0 = m0 + m1/2,   C1 = m1/2 + m2/2,   C2 = m2/2 + m3

    This preserves the authored distribution (a front-heavy cable stays front-heavy) and its
    total. The mass lands at the segment midpoints rather than the vertices -- the inherent
    approximation of a rigid chain that has no per-vertex DOF. A body-mass total has no
    per-point profile, so it just rescales the (density-derived) segment masses to that total.
    """
    from ..usd import utils as usd  # noqa: PLC0415

    num_points = sum(n for _, n, _ in point_runs)
    point_masses = usd._get_deformable_point_masses(prim, read_attr)
    body_mass, _ = usd._get_deformable_body_overrides(prim, read_attr)
    if point_masses is not None and len(point_masses) != num_points:
        warnings.warn(
            f"{prim.GetPath()}: physics:masses length {len(point_masses)} != {num_points} curve points; "
            f"ignoring per-point masses.",
            stacklevel=2,
        )
        point_masses = None
    if point_masses is not None:
        for start, n, bodies in point_runs:
            pm = [float(point_masses[start + i]) for i in range(n)]
            if closed:
                # Loop: N points -> N segments, every point borders two, so split each in half.
                seg_masses = [0.5 * pm[s] + 0.5 * pm[(s + 1) % n] for s in range(n)]
            else:
                # Open: N points -> N-1 segments. Interior points (the +0.5 terms) split between
                # two segments; the first/last points are endpoints and give their full mass.
                seg_masses = [
                    (pm[s] if s == 0 else 0.5 * pm[s]) + (pm[s + 1] if s + 1 == n - 1 else 0.5 * pm[s + 1])
                    for s in range(n - 1)
                ]
            for b, m in zip(bodies, seg_masses, strict=True):
                _set_body_mass(builder, b, m)
        return
    if body_mass is None:
        return
    current = float(sum(builder.body_mass[b] for b in body_ids))
    if current <= 0.0:
        return
    scale = body_mass / current
    for b in body_ids:
        _set_body_mass(builder, b, builder.body_mass[b] * scale)


def _cable_attachment_anchors(
    attachment_path: str,
    src_path: str,
    site_type: str,
    site_index: int,
    coord: wp.vec3 | None,
    segment_maps: Mapping[str, Mapping[int, tuple[int, float]]],
    point_anchor_maps: Mapping[str, Mapping[int, list[tuple[int, wp.vec3]]]],
) -> list[tuple[int, wp.vec3]] | None:
    """Resolve a cable attachment site to ``(body, local_point)`` anchors.

    ``point`` sites map to the cable's per-point anchors; ``segment`` sites place the anchor on
    the body using the proposal segment coordinate ``coord`` ``(u, s, t)``. Returns ``None`` if
    ``src_path`` is not an imported cable, or ``[]`` (with a warning) for an unresolved site.
    """
    segment_map = segment_maps.get(src_path)
    point_anchors = point_anchor_maps.get(src_path)
    if segment_map is None or point_anchors is None:
        return None

    if site_type == "point":
        anchors = point_anchors.get(site_index)
        if not anchors:
            warnings.warn(
                f"{attachment_path}: point index {site_index} is not an imported cable point on {src_path}; "
                "skipping that attachment site.",
                stacklevel=2,
            )
            return []
        return list(anchors)

    if site_type != "segment":
        return None

    segment = segment_map.get(site_index)
    if segment is None:
        warnings.warn(
            f"{attachment_path}: segment index {site_index} is not an imported cable segment on {src_path}; "
            "skipping that attachment site.",
            stacklevel=2,
        )
        return []
    if coord is None:
        warnings.warn(
            f"{attachment_path}: segment attachment site {site_index} is missing coords0; skipping.",
            stacklevel=2,
        )
        return []

    segment_body, segment_length = segment
    if segment_length <= 1.0e-8:
        warnings.warn(
            f"{attachment_path}: segment index {site_index} has zero length; skipping that attachment site.",
            stacklevel=2,
        )
        return []

    u = float(coord[0])
    s = float(coord[1])
    t = float(coord[2])
    # Imported cable bodies use local +Z along the segment and local +Y for the
    # proposal normal. The proposal binormal is tangent x normal, i.e. local -X.
    local_point = wp.vec3(-t, s, (0.5 - u) * segment_length)
    return [(segment_body, local_point)]


@dataclass
class _DeformableImportContext:
    """Shared state for the deformable import passes (cable / cloth / volume / attachment).

    Bundles the :func:`parse_usd` inputs, the helper closures the passes need, and the result maps
    they populate, so the passes can live in this module instead of as closures in ``parse_usd``.
    The result maps are the same dict objects ``parse_usd`` returns, mutated in place.
    """

    builder: ModelBuilder
    stage: Usd.Stage
    root_prim: Usd.Prim
    resolver: Any
    collect_schema_attrs: bool
    deformable_read: Callable
    get_prim_world_mat: Callable
    get_rigid_body_ancestor_path: Callable
    get_first_target: Callable
    get_tetmesh_cached: Callable
    is_uniform_scale: Callable
    incoming_world_xform: wp.transform
    linear_unit: float
    ignore_paths: Sequence[str]
    verbose: bool
    path_body_map: dict
    path_cable_map: dict
    path_cable_attrs: dict
    path_cable_segments: dict
    path_cable_point_anchors: dict
    path_cloth_map: dict
    path_cloth_attrs: dict
    path_soft_map: dict
    path_soft_attrs: dict
    path_attachment_map: dict
    path_attachment_attrs: dict


def _deformable_import_cable_graphs(ctx: _DeformableImportContext) -> tuple[set[str], set[str]]:
    """Weld curve deformables joined by curve-to-curve ``PhysicsAttachment`` prims into
    rod graphs via :meth:`ModelBuilder.add_rod_graph`.

    A ``point``->``point`` attachment whose ``src0``/``src1`` are both imported curve
    deformables is topology, not a runtime constraint: the two referenced control points are
    the same junction node. Curves transitively joined this way form one graph component, built
    with a single ``add_rod_graph`` call (one capsule body per segment, junction nodes shared).
    Returns the curve prim paths and the junction attachment prim paths consumed here so the
    per-curve cable pass and the attachment post-pass skip them. Single curves and
    curve-to-xform attachments are left to those passes.

    :meth:`ModelBuilder.add_rod_graph` applies one scalar radius/density/stiffness to a whole
    component, so a welded graph uses the first curve's material as the representative for every
    segment (heterogeneous welds warn). Each curve's own authored material is still reported in
    ``path_cable_attrs``.
    """
    from pxr import Usd, UsdGeom

    from ..usd import utils as usd  # noqa: PLC0415
    from .cable import create_cable_stiffness_from_elastic_moduli  # noqa: PLC0415

    builder = ctx.builder
    root_prim = ctx.root_prim
    ignore_paths = ctx.ignore_paths
    incoming_world_xform = ctx.incoming_world_xform
    linear_unit = ctx.linear_unit
    verbose = ctx.verbose
    deformable_read = ctx.deformable_read
    get_prim_world_mat = ctx.get_prim_world_mat
    path_cable_map = ctx.path_cable_map
    path_cable_attrs = ctx.path_cable_attrs
    path_cable_segments = ctx.path_cable_segments
    path_cable_point_anchors = ctx.path_cable_point_anchors

    consumed_curves: set[str] = set()
    consumed_attachments: set[str] = set()
    if not (root_prim and root_prim.IsValid()):
        return consumed_curves, consumed_attachments

    # Collect single-curve curve deformables eligible for graph welding. Junctions reference a
    # whole BasisCurves prim (not an individual curve within it), so a multi-curve prim is left
    # to the per-curve pass.
    curve_recs: dict[str, _CurveDeformableRecord] = {}
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.BasisCurves):
            continue
        if not usd.has_applied_api_schema(prim, "PhysicsCurvesDeformableSimAPI"):
            continue
        path = str(prim.GetPath())
        if _is_ignored_path(path, ignore_paths):
            continue
        curves = UsdGeom.BasisCurves(prim)
        if curves.GetTypeAttr().Get() != UsdGeom.Tokens.linear:
            continue
        pts = curves.GetPointsAttr().Get()
        vcounts = curves.GetCurveVertexCountsAttr().Get()
        if not pts or not vcounts or len(vcounts) != 1 or int(vcounts[0]) < 3:
            continue
        wmat = get_prim_world_mat(prim, None, incoming_world_xform)
        wp_pos, wp_rot, wp_scale = wp.transform_decompose(wmat)
        wxf = wp.transform(wp_pos, wp_rot)
        positions = [
            wp.transform_point(
                wxf, wp.vec3(float(p[0]) * wp_scale[0], float(p[1]) * wp_scale[1], float(p[2]) * wp_scale[2])
            )
            for p in pts
        ]
        mat = usd._get_curve_deformable_material(prim, deformable_read) or {}
        radius = 0.5 * mat["thickness"] if "thickness" in mat else 0.05 / linear_unit
        density = _resolve_deformable_density(prim, mat.get("density"), deformable_read)
        curve_recs[path] = _CurveDeformableRecord(
            prim=prim,
            positions=positions,
            closed=curves.GetWrapAttr().Get() == UsdGeom.Tokens.periodic,
            material=mat,
            radius=radius,
            density=density if density is not None else builder.default_shape_cfg.density,
        )

    if not curve_recs:
        return consumed_curves, consumed_attachments

    # Union-find over curve prim paths; record the per-attachment welded point pairs.
    parent = {p: p for p in curve_recs}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    welds: list[tuple[str, int, str, int]] = []
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        if prim.GetTypeName() != "PhysicsAttachment":
            continue
        # An ignored junction must not alter topology; leave its curves to the per-curve pass.
        if _is_ignored_path(str(prim.GetPath()), ignore_paths):
            continue
        s0 = prim.GetRelationship("physics:src0").GetTargets()
        s1 = prim.GetRelationship("physics:src1").GetTargets()
        if not s0 or not s1:
            continue
        src0, src1 = str(s0[0]), str(s1[0])
        if src0 not in curve_recs or src1 not in curve_recs or src0 == src1:
            continue
        if str(deformable_read(prim, "type0") or "") != "point" or str(deformable_read(prim, "type1") or "") != "point":
            continue
        enabled = deformable_read(prim, "attachmentEnabled")
        if enabled is not None and not bool(enabled):
            continue
        idx0 = [int(i) for i in (deformable_read(prim, "indices0") or [])]
        idx1 = [int(i) for i in (deformable_read(prim, "indices1") or [])]
        if not _validate_attachment_index_pairs(
            idx0, len(curve_recs[src0].positions), idx1, len(curve_recs[src1].positions), str(prim.GetPath())
        ):
            continue  # malformed junction: leave both curves to the per-curve pass
        union(src0, src1)
        for a, b in zip(idx0, idx1, strict=True):
            welds.append((src0, a, src1, b))
        consumed_attachments.add(str(prim.GetPath()))

    components: dict[str, list[str]] = {}
    for p in curve_recs:
        components.setdefault(find(p), []).append(p)
    welds_by_comp: dict[str, list[tuple[str, int, str, int]]] = {}
    for w in welds:
        welds_by_comp.setdefault(find(w[0]), []).append(w)

    def _build_graph_component(cid, comp_paths, comp_welds):
        # Merge welded control points into shared graph nodes (union-find over (path, index)).
        node_parent: dict[tuple[str, int], tuple[str, int]] = {}

        def node_find(node: tuple[str, int]) -> tuple[str, int]:
            node_parent.setdefault(node, node)
            while node_parent[node] != node:
                node_parent[node] = node_parent[node_parent[node]]
                node = node_parent[node]
            return node

        def node_union(a: tuple[str, int], b: tuple[str, int]) -> None:
            ra, rb = node_find(a), node_find(b)
            if ra != rb:
                node_parent[rb] = ra

        for key in comp_paths:
            for i in range(len(curve_recs[key].positions)):
                node_find((key, i))
        for s0, i0, s1, i1 in comp_welds:
            node_union((s0, i0), (s1, i1))

        node_positions: list[wp.vec3] = []
        node_id: dict[tuple[str, int], int] = {}

        def global_node(local: tuple[str, int]) -> int:
            root = node_find(local)
            if root not in node_id:
                node_id[root] = len(node_positions)
                rk, ri = root
                node_positions.append(curve_recs[rk].positions[ri])
            return node_id[root]

        edges: list[tuple[int, int]] = []
        edge_owner: list[tuple[str, int]] = []  # (curve path, local segment index)
        for key in comp_paths:
            rec = curve_recs[key]
            n = len(rec.positions)
            local_edges = [(i, i + 1) for i in range(n - 1)]
            if rec.closed:
                local_edges.append((n - 1, 0))
            for seg, (u, v) in enumerate(local_edges):
                gu, gv = global_node((key, u)), global_node((key, v))
                if gu == gv:
                    continue  # zero-length welded edge
                edges.append((gu, gv))
                edge_owner.append((key, seg))

        if len(node_positions) < 2 or not edges:
            return

        rep = curve_recs[comp_paths[0]]
        # add_rod_graph applies one scalar radius/density/stiffness to the whole component, so a
        # welded graph necessarily flattens its curves to a single representative material. Warn
        # when the welded curves disagree so the flattening is explicit rather than silent.
        if len(comp_paths) > 1:
            sigs = {
                (
                    curve_recs[p].radius,
                    curve_recs[p].density,
                    curve_recs[p].material.get("stretchStiffness"),
                    curve_recs[p].material.get("bendStiffness"),
                )
                for p in comp_paths
            }
            if len(sigs) > 1:
                warnings.warn(
                    f"cable graph '{cid}': welded curves have differing radius/density/stiffness; "
                    f"using '{comp_paths[0]}' as the representative material for the whole component.",
                    stacklevel=2,
                )
        radius = rep.radius
        seg_len = sum(float(wp.length(node_positions[v] - node_positions[u])) for u, v in edges) / len(edges)
        mat = rep.material
        stretch = bend = None
        if seg_len > 0.0:
            if "stretchStiffness" in mat:
                stretch = create_cable_stiffness_from_elastic_moduli(mat["stretchStiffness"], radius, seg_len)[0]
            if "bendStiffness" in mat:
                bend = create_cable_stiffness_from_elastic_moduli(mat["bendStiffness"], radius, seg_len)[1]
        cfg = replace(builder.default_shape_cfg, density=rep.density)
        # Unlike single cables, the graph junction spanning tree is intrinsic topology, not a
        # caller choice, and only a tree (not the all-incident-edges joint set produced when
        # unwrapped) is articulation-safe. So the importer wraps each component into its own
        # articulation here; path_cable_map exposes empty joints for graph curves accordingly.
        body_ids, _graph_joint_ids = builder.add_rod_graph(
            node_positions=node_positions,
            edges=edges,
            radius=radius,
            cfg=cfg,
            stretch_stiffness=stretch,
            bend_stiffness=bend,
            label=cid,
            wrap_in_articulation=True,
            body_frame_origin="com",
        )

        # Partition graph bodies back to their owning curve, and rebuild the per-prim anchor
        # maps the curve-to-xform attachment pass reads (point index / segment index -> body).
        per_prim_segments: dict[str, dict[int, tuple[int, float]]] = {}
        per_prim_bodies: dict[str, list[int]] = {}
        for ge, (key, seg) in enumerate(edge_owner):
            u, v = edges[ge]
            length = float(wp.length(node_positions[v] - node_positions[u]))
            per_prim_segments.setdefault(key, {})[seg] = (body_ids[ge], length)
            per_prim_bodies.setdefault(key, []).append(body_ids[ge])

        for key in comp_paths:
            rec = curve_recs[key]
            n = len(rec.positions)
            segs = per_prim_segments.get(key, {})
            anchors: dict[int, list[tuple[int, wp.vec3]]] = {}
            for pi in range(n):
                if rec.closed:
                    incident = (((pi - 1) % n, "end"), (pi % n, "start"))
                elif pi == 0:
                    incident = ((0, "start"),)
                elif pi == n - 1:
                    incident = ((n - 2, "end"),)
                else:
                    incident = ((pi - 1, "end"), (pi, "start"))
                for seg, role in incident:
                    if seg in segs:
                        body, length = segs[seg]
                        z = -0.5 * length if role == "start" else 0.5 * length
                        anchors.setdefault(pi, []).append((body, wp.vec3(0.0, 0.0, z)))
            path_cable_point_anchors[key] = anchors
            path_cable_segments[key] = segs
            # Graph cables are returned pre-wrapped (see add_rod_graph call above), so joints are
            # empty: callers using the "if joints: add_articulation(joints)" pattern skip them.
            path_cable_map[key] = (per_prim_bodies.get(key, []), [])
            path_cable_attrs[key] = {
                "material": dict(rec.material),
                "resolved_density": rec.density,
                "closed": rec.closed,
                "graph_component": cid,
            }
            key_bodies = per_prim_bodies.get(key, [])
            _apply_cable_masses(builder, rec.prim, key_bodies, [(0, n, key_bodies)], rec.closed, deformable_read)
            consumed_curves.add(key)
        if verbose:
            print(f"Added cable graph {cid} with {len(body_ids)} segments across {len(comp_paths)} curves.")

    for cid, comp_curves in components.items():
        comp_paths = sorted(comp_curves)
        comp_welds = welds_by_comp.get(cid, [])
        if len(comp_paths) == 1 and not comp_welds:
            continue  # plain single curve: leave it to the per-curve pass
        _build_graph_component(cid, comp_paths, comp_welds)

    return consumed_curves, consumed_attachments


def _deformable_import_cable(ctx: _DeformableImportContext, consumed_cable_curve_paths: set[str]) -> None:
    """Import single-curve cable deformables (linear ``GeomBasisCurves`` -> VBD rod via ``add_rod``).

    Curves already welded into a rod graph (``consumed_cable_curve_paths``) are skipped. The cable
    joints are left unwrapped for the caller to wrap before ``finalize()``. Results land in
    ``path_cable_map`` / attrs / segments / point anchors.
    """
    from pxr import Usd, UsdGeom

    from ..usd import utils as usd  # noqa: PLC0415
    from .cable import create_cable_stiffness_from_elastic_moduli  # noqa: PLC0415

    builder = ctx.builder
    root_prim = ctx.root_prim
    ignore_paths = ctx.ignore_paths
    incoming_world_xform = ctx.incoming_world_xform
    linear_unit = ctx.linear_unit
    verbose = ctx.verbose
    deformable_read = ctx.deformable_read
    get_prim_world_mat = ctx.get_prim_world_mat
    path_cable_map = ctx.path_cable_map
    path_cable_attrs = ctx.path_cable_attrs
    path_cable_segments = ctx.path_cable_segments
    path_cable_point_anchors = ctx.path_cable_point_anchors

    if not (root_prim and root_prim.IsValid()):
        return
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.BasisCurves):
            continue
        if not usd.has_applied_api_schema(prim, "PhysicsCurvesDeformableSimAPI"):
            continue

        path = str(prim.GetPath())
        if path in consumed_cable_curve_paths:
            continue  # already built as part of a welded rod graph
        # Per-instance proxies are imported via TraverseInstanceProxies above; USD does
        # not surface prototype masters under a scene-root traversal, so no prototype
        # filter is needed (a non-rendered template is authored as a class/inactive prim).
        if _is_ignored_path(path, ignore_paths):
            continue

        curves = UsdGeom.BasisCurves(prim)
        # The proposal scopes curve deformables to linear basis curves; the
        # importer treats the points as a segment polyline, so a non-linear
        # (e.g. cubic) curve would be misinterpreted.
        if curves.GetTypeAttr().Get() != UsdGeom.Tokens.linear:
            warnings.warn(
                f"{path}: only linear BasisCurves import as cables; skipping non-linear curve.",
                stacklevel=2,
            )
            continue
        points = curves.GetPointsAttr().Get()
        vertex_counts = curves.GetCurveVertexCountsAttr().Get()
        if not points or not vertex_counts:
            warnings.warn(f"{path}: cable curve has no points / curveVertexCounts; skipping.", stacklevel=2)
            continue
        closed = curves.GetWrapAttr().Get() == UsdGeom.Tokens.periodic
        # Rest centerline used for the rest length below (one point per vertex); restNormals
        # and rest bend angles are not imported yet.
        rest_shape_points = deformable_read(prim, "restShapePoints")
        if rest_shape_points is not None and len(rest_shape_points) != len(points):
            warnings.warn(
                f"{path}: restShapePoints length {len(rest_shape_points)} != points {len(points)}; "
                f"ignoring rest shape (rest length taken from the imported points).",
                stacklevel=2,
            )
            rest_shape_points = None
        _warn_unsupported_rest_fields(prim, path, ("restNormals",), deformable_read)
        _warn_geometry_authored_material_attrs(prim, path, "PhysicsCurvesDeformableMaterialAPI", deformable_read)

        world_mat = get_prim_world_mat(prim, None, incoming_world_xform)
        w_pos, w_rot, w_scale = wp.transform_decompose(world_mat)
        world_xf = wp.transform(w_pos, w_rot)

        # Per-point normals give each segment's cross-section frame (twist).
        # ``primvars:normals`` takes precedence over the schema ``normals`` attribute and
        # may be indexed (flattened here); either way the normals are honored only when
        # authored per point: interpolation must be vertex/varying (one normal per control
        # point) and the count must match the points.
        normals_primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar("normals")
        if normals_primvar.HasValue():
            normals = normals_primvar.ComputeFlattened()
            normals_interp = normals_primvar.GetInterpolation()
        else:
            normals = curves.GetNormalsAttr().Get()
            normals_interp = curves.GetNormalsInterpolation()
        if normals is not None and normals_interp not in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying):
            warnings.warn(
                f"{path}: normals interpolation '{normals_interp}' is not per-point (vertex/varying); "
                f"ignoring normals.",
                stacklevel=2,
            )
            normals = None
        if normals is not None and len(normals) != len(points):
            warnings.warn(
                f"{path}: normals length {len(normals)} != points {len(points)}; ignoring normals.",
                stacklevel=2,
            )
            normals = None

        # The proposal authors curve "stretchStiffness" / "bendStiffness" in force/area, i.e.
        # elastic moduli E. create_cable_stiffness_from_elastic_moduli() converts each to the
        # per-joint stiffness add_rod expects via the circular cross-section and segment rest
        # length L (stretch = E*A/L, bend = E*I/L); applied per curve below.
        cable_mat = usd._get_curve_deformable_material(prim, deformable_read) or {}
        if "thickness" in cable_mat:
            radius = 0.5 * cable_mat["thickness"]
        else:
            # No authored thickness: assume a default radius. Express it via the stage's linear
            # unit (meters per unit) so the assumed size is a fixed physical ~0.05 m regardless
            # of cm / mm / m authoring, rather than a meters-flavored literal in stage units.
            radius = 0.05 / linear_unit
            warnings.warn(
                f"{path}: no cable thickness authored (physics:thickness); assuming a default "
                f"radius of {radius:g} stage units (~0.05 m). Author physics:thickness on the "
                f"bound material to set it.",
                stacklevel=2,
            )
        # Density precedence resolved here; total-mass/per-point overrides applied after add_rod.
        cable_density = _resolve_deformable_density(prim, cable_mat.get("density"), deformable_read)
        resolved_cable_density = cable_density if cable_density is not None else builder.default_shape_cfg.density
        cable_cfg = replace(builder.default_shape_cfg, density=resolved_cable_density)
        if "shearStiffness" in cable_mat or "twistStiffness" in cable_mat:
            warnings.warn(
                f"{path}: shearStiffness / twistStiffness are not yet mapped to the VBD cable "
                f"(isotropic bend only); ignoring.",
                stacklevel=2,
            )

        cable_bodies: list[int] = []
        cable_joints: list[int] = []
        # vertex index -> [(segment body, body-local point)]
        cable_point_anchors: dict[int, list[tuple[int, wp.vec3]]] = {}
        # flat segment index -> (segment body, segment length)
        cable_segments: dict[int, tuple[int, float]] = {}
        # Per built curve: (point offset in the prim's masses array, point count, segment bodies),
        # so per-point masses can be lumped onto each curve's segments.
        cable_point_runs: list[tuple[int, int, list[int]]] = []
        imported_point_count = 0
        offset = 0
        flat_segment_index = 0
        for ci, vertex_count in enumerate(vertex_counts):
            n = int(vertex_count)
            start = offset
            local_pts = points[start : start + n]
            offset += n
            curve_segment_count = n if closed else max(0, n - 1)
            # add_rod needs >= 2 segments, i.e. >= 3 centerline points.
            if n < 3:
                warnings.warn(f"{path}: curve {ci} has {n} points (need >= 3); skipping that curve.", stacklevel=2)
                flat_segment_index += curve_segment_count
                continue
            positions = [
                wp.transform_point(
                    world_xf, wp.vec3(float(p[0]) * w_scale[0], float(p[1]) * w_scale[1], float(p[2]) * w_scale[2])
                )
                for p in local_pts
            ]
            # For a periodic curve the closing segment (v[-1] -> v[0]) is a real
            # segment: close the polyline so add_rod builds a body for it (add_rod
            # makes len(positions) - 1 bodies; closed=True then adds the loop joint).
            if closed:
                positions = [*positions, positions[0]]
            num_seg = len(positions) - 1
            # A zero-length segment (duplicate consecutive points) can't be oriented or
            # sized; warn and skip just this curve rather than aborting the whole import.
            seg_lengths = [float(wp.length(positions[i + 1] - positions[i])) for i in range(num_seg)]
            if min(seg_lengths, default=0.0) <= 1.0e-8:
                warnings.warn(
                    f"{path}: curve {ci} has duplicate consecutive points (zero-length segment); skipping that curve.",
                    stacklevel=2,
                )
                flat_segment_index += curve_segment_count
                continue
            imported_point_count += n
            # Authored normals set each segment's cross-section twist. Transform them by the
            # inverse-transpose of the world map (correct under non-uniform scale): for
            # rotation R and per-axis scale S that is R*S^-1 -- divide by the scale, then rotate.
            quaternions = None
            if normals is not None:
                inv_scale = wp.vec3(
                    *(1.0 / s if abs(s) > 1.0e-8 else 1.0 for s in (w_scale[0], w_scale[1], w_scale[2]))
                )
                seg_normals = [
                    wp.quat_rotate(
                        w_rot,
                        wp.vec3(float(nv[0]) * inv_scale[0], float(nv[1]) * inv_scale[1], float(nv[2]) * inv_scale[2]),
                    )
                    for nv in normals[start : start + n]
                ]
                quaternions = _cable_segment_quaternions(positions, seg_normals)
            # Per-joint stiffness needs a per-segment rest length: the mean of the
            # actual segment lengths (the straight-line endpoint distance would
            # underestimate it for curved cables and inflate the stiffness).
            seg_len = sum(seg_lengths) / max(1, num_seg)
            # Use the rest centerline for the rest length when authored (else the imported
            # points), so the cable is not pre-stressed. Only lengths matter -> apply w_scale.
            if rest_shape_points is not None:
                rest_pts = [
                    wp.vec3(float(rp[0]) * w_scale[0], float(rp[1]) * w_scale[1], float(rp[2]) * w_scale[2])
                    for rp in rest_shape_points[start : start + n]
                ]
                if closed:
                    rest_pts = [*rest_pts, rest_pts[0]]
                rest_seg_lengths = [float(wp.length(rest_pts[i + 1] - rest_pts[i])) for i in range(num_seg)]
                if min(rest_seg_lengths, default=0.0) > 1.0e-8:
                    seg_len = sum(rest_seg_lengths) / max(1, num_seg)
            # Convert each authored modulus through the shared cable utility (stretch = E*A/L,
            # bend = E*I/L); the moduli are independent and either may be absent -> None ->
            # builder default. The util returns both from one modulus, so take the matching one.
            stretch_stiffness = bend_stiffness = None
            if seg_len > 0.0:
                if "stretchStiffness" in cable_mat:
                    stretch_stiffness = create_cable_stiffness_from_elastic_moduli(
                        cable_mat["stretchStiffness"], radius, seg_len
                    )[0]
                if "bendStiffness" in cable_mat:
                    bend_stiffness = create_cable_stiffness_from_elastic_moduli(
                        cable_mat["bendStiffness"], radius, seg_len
                    )[1]
            label = path if len(vertex_counts) == 1 else f"{path}_curve{ci}"
            # Imported cables are left unwrapped (wrap_in_articulation=False): the
            # caller wraps the returned joints via add_articulation() before
            # finalize(), so articulation topology (e.g. closing a loop with extra
            # joints, or attaching the cable to other bodies) stays a caller choice.
            bodies, joints = builder.add_rod(
                positions=positions,
                quaternions=quaternions,
                radius=radius,
                cfg=cable_cfg,
                stretch_stiffness=stretch_stiffness,
                bend_stiffness=bend_stiffness,
                closed=closed,
                label=label,
                wrap_in_articulation=False,
                body_frame_origin="com",
            )
            cable_bodies.extend(bodies)
            cable_joints.extend(joints)
            cable_point_runs.append((start, n, bodies))

            for si, body in enumerate(bodies):
                seg_index = flat_segment_index + si
                cable_segments[seg_index] = (body, seg_lengths[si])

            for pi in range(n):
                point_index = start + pi
                anchors = cable_point_anchors.setdefault(point_index, [])
                if closed:
                    incident = ((pi - 1) % n, pi)
                elif pi == 0:
                    incident = (0,)
                elif pi == n - 1:
                    incident = (n - 2,)
                else:
                    incident = (pi - 1, pi)
                for si in incident:
                    z = -0.5 * seg_lengths[si] if si == pi else 0.5 * seg_lengths[si]
                    anchors.append((bodies[si], wp.vec3(0.0, 0.0, z)))

            flat_segment_index += curve_segment_count

        if cable_bodies:
            _apply_cable_masses(builder, prim, cable_bodies, cable_point_runs, closed, deformable_read)
            path_cable_map[path] = (cable_bodies, cable_joints)
            path_cable_point_anchors[path] = cable_point_anchors
            path_cable_segments[path] = cable_segments
            path_cable_attrs[path] = {
                "material": dict(cable_mat),
                "resolved_density": resolved_cable_density,
                "closed": closed,
            }
            if verbose:
                print(f"Added cable {path} with {len(cable_bodies)} segments.")


def _deformable_import_cloth(ctx: _DeformableImportContext) -> None:
    """Import surface deformables (``PhysicsSurfaceDeformableSimAPI`` polygon ``Mesh`` -> cloth).

    n-gon faces are fan-triangulated, so the source need not be pre-triangulated. The surface
    material is mapped onto the isotropic membrane and results land in ``path_cloth_map`` / attrs.
    """
    from pxr import Usd, UsdGeom

    from ..usd import utils as usd  # noqa: PLC0415
    from ..usd.schema_resolver import PrimType  # noqa: PLC0415

    builder = ctx.builder
    root_prim = ctx.root_prim
    ignore_paths = ctx.ignore_paths
    incoming_world_xform = ctx.incoming_world_xform
    verbose = ctx.verbose
    deformable_read = ctx.deformable_read
    get_prim_world_mat = ctx.get_prim_world_mat
    resolver = ctx.resolver
    path_cloth_map = ctx.path_cloth_map
    path_cloth_attrs = ctx.path_cloth_attrs

    if not (root_prim and root_prim.IsValid()):
        return
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if not usd.has_applied_api_schema(prim, "PhysicsSurfaceDeformableSimAPI"):
            continue

        path = str(prim.GetPath())
        # Per-instance proxies are imported via TraverseInstanceProxies above; USD does
        # not surface prototype masters under a scene-root traversal, so no prototype
        # filter is needed (a non-rendered template is authored as a class/inactive prim).
        if _is_ignored_path(path, ignore_paths):
            continue

        mesh = UsdGeom.Mesh(prim)
        mesh_points = mesh.GetPointsAttr().Get()
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not mesh_points or not face_counts or not face_indices:
            warnings.warn(f"{path}: cloth mesh missing points / topology; skipping.", stacklevel=2)
            continue
        if any(int(c) < 3 for c in face_counts):
            warnings.warn(f"{path}: cloth mesh has a face with fewer than 3 vertices; skipping.", stacklevel=2)
            continue
        # Reuse the shared mesh handling from the rigid path: fan-triangulate faces
        # (n-gons such as quads; exact for convex faces, preserving vertex indices so
        # each mesh point stays one particle) and flip winding for left-handed
        # orientation. Subdivision scheme is not consulted -- the polygon cage is simulated.
        tri_faces = usd.fan_triangulate_faces(np.asarray(face_counts), np.asarray(face_indices))
        if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
            tri_faces = tri_faces[:, ::-1]
        tri_vertex_indices = tri_faces.reshape(-1).tolist()
        _warn_unsupported_rest_fields(
            prim,
            path,
            ("restShapePoints", "restBendAngles", "restAdjTriPairs", "restBendAnglesDefault"),
            deformable_read,
        )
        _warn_geometry_authored_material_attrs(prim, path, "PhysicsSurfaceDeformableMaterialAPI", deformable_read)

        world_mat = get_prim_world_mat(prim, None, incoming_world_xform)
        cloth_pos, cloth_rot, cloth_scale = wp.transform_decompose(world_mat)
        # add_cloth_mesh creates one particle per mesh vertex and takes only a uniform
        # scale, unlike add_shape_mesh's per-axis Vec3 shape scale. So bake the full world
        # scale (including non-uniform) into the vertices here and pass scale=1.
        sx, sy, sz = float(cloth_scale[0]), float(cloth_scale[1]), float(cloth_scale[2])
        cloth_vertices = [wp.vec3(float(p[0]) * sx, float(p[1]) * sy, float(p[2]) * sz) for p in mesh_points]
        scale = 1.0

        cloth_mat = usd._get_surface_deformable_material(prim, deformable_read) or {}
        # Surface thickness: prefer the material's authored value; otherwise fall back to a
        # shell mass model's thickness (NewtonMassAPI massModel="shell" / shellThickness,
        # resolved across Newton / MuJoCo like the rigid shape path above).
        thickness = cloth_mat.get("thickness")
        if thickness is None and resolver.get_value(prim, PrimType.SHAPE, "mass_model", default="solid") == "shell":
            shell_thickness_val = resolver.get_value(prim, PrimType.SHAPE, "shell_thickness")
            if shell_thickness_val is not None and math.isfinite(float(shell_thickness_val)):
                if float(shell_thickness_val) > 0.0:
                    thickness = float(shell_thickness_val)

        # Map the surface material onto the SolverVBD / SolverSemiImplicit membrane, whose
        # triangle has three parameters:
        #   tri_ke  = mu     -> in-plane elastic stiffness  <- stretchStiffness
        #   edge_ke          -> dihedral bending stiffness  <- bendStiffness
        #   tri_ka  = lambda -> area preservation (Poisson) <- (no proposal attribute)
        # This membrane is isotropic, so stretch and shear are not separable: both live in mu.
        # We therefore drive mu from stretchStiffness and drop shearStiffness (an anisotropic
        # membrane such as SolverStyle3D's tri_aniso_ke is needed to honor it). tri_ka encodes
        # the Poisson ratio nu = tri_ka / (tri_ka + 2*tri_ke); given a target nu it would be
        # tri_ka = 2*tri_ke*nu / (1 - nu), but the surface material authors no Poisson, so we
        # leave tri_ka at the solver default rather than fabricate one.
        #
        # The proposal authors moduli in force/area; Newton integrates the triangle energy over
        # area, so a modulus is scaled by the shell thickness when one is authored (membrane
        # stiffness ~ E*h, bending ~ E*h^3) -- the surface analog of the cable path's E*A/L.
        #
        # Either way the raw, as-authored moduli (including the dropped shearStiffness) survive
        # in path_cloth_attrs, so another solver can rebuild from them.
        tri_ke = cloth_mat.get("stretchStiffness")
        edge_ke = cloth_mat.get("bendStiffness")
        if thickness is not None:
            tri_ke = tri_ke * thickness if tri_ke is not None else None
            edge_ke = edge_ke * thickness**3 if edge_ke is not None else None
        tri_ka = None  # see above: no proposal attribute -> solver default
        if "shearStiffness" in cloth_mat:
            warnings.warn(
                f"{path}: shearStiffness is not applied -- SolverVBD / SolverSemiImplicit use an "
                f"isotropic membrane where stretch and shear share one modulus. Use SolverStyle3D "
                f"(tri_aniso_ke) for independent shear; the value is preserved in path_cloth_attrs.",
                stacklevel=2,
            )
        # Newton cloth density is areal; convert the volumetric material density with the
        # surface thickness (required for surface mass per the proposal). Body density overrides.
        vol_density = _resolve_deformable_density(prim, cloth_mat.get("density"), deformable_read)
        resolved_cloth_density = vol_density if vol_density is not None else builder.default_shape_cfg.density
        # The areal value is builder-specific; keep it local to add_cloth_mesh.
        density = resolved_cloth_density * thickness if thickness is not None else resolved_cloth_density

        p0, t0, e0 = builder.particle_count, builder.tri_count, builder.edge_count
        builder.add_cloth_mesh(
            pos=cloth_pos,
            rot=cloth_rot,
            scale=scale,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=cloth_vertices,
            indices=tri_vertex_indices,
            density=density,
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            edge_ke=edge_ke,
            label=path,
        )
        _apply_particle_masses(builder, prim, p0, builder.particle_count, deformable_read)
        path_cloth_map[path] = {
            "particle": (p0, builder.particle_count),
            "tri": (t0, builder.tri_count),
            "edge": (e0, builder.edge_count),
        }
        path_cloth_attrs[path] = {
            "material": dict(cloth_mat),
            "resolved_density": resolved_cloth_density,
        }
        if verbose:
            print(f"Added cloth {path} with {builder.particle_count - p0} particles.")


def _deformable_import_volume(ctx: _DeformableImportContext) -> None:
    """Import volume deformables (``UsdGeom.TetMesh`` -> soft body via ``add_soft_mesh``).

    ``PhysicsVolumeDeformableSimAPI`` (or a ``PhysicsDeformableBodyAPI``) opts into the proposal
    mass precedence; a bare TetMesh stays legacy. Results land in ``path_soft_map`` / attrs.
    """
    from pxr import Usd, UsdGeom

    from ..usd import utils as usd  # noqa: PLC0415

    builder = ctx.builder
    root_prim = ctx.root_prim
    ignore_paths = ctx.ignore_paths
    incoming_world_xform = ctx.incoming_world_xform
    verbose = ctx.verbose
    deformable_read = ctx.deformable_read
    get_prim_world_mat = ctx.get_prim_world_mat
    get_tetmesh_cached = ctx.get_tetmesh_cached
    is_uniform_scale = ctx.is_uniform_scale
    resolver = ctx.resolver
    collect_schema_attrs = ctx.collect_schema_attrs
    path_soft_map = ctx.path_soft_map
    path_soft_attrs = ctx.path_soft_attrs

    if not (root_prim and root_prim.IsValid()):
        return
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.TetMesh):
            continue

        path = str(prim.GetPath())
        # Per-instance proxies are imported via TraverseInstanceProxies above; USD does
        # not surface prototype masters under a scene-root traversal, so no prototype
        # filter is needed (a non-rendered template is authored as a class/inactive prim).
        if _is_ignored_path(path, ignore_paths):
            continue

        is_volume_deformable = (
            usd.has_applied_api_schema(prim, "PhysicsVolumeDeformableSimAPI")
            or usd._find_deformable_body_prim(prim) is not None
        )
        if is_volume_deformable:
            _warn_unsupported_rest_fields(prim, path, ("restShapePoints",), deformable_read)
            _warn_geometry_authored_material_attrs(prim, path, "PhysicsVolumeDeformableMaterialAPI", deformable_read)

        if collect_schema_attrs:
            resolver.collect_prim_attrs(prim)

        tetmesh = get_tetmesh_cached(prim)
        tetmesh_for_builder = tetmesh
        if tetmesh.custom_attributes:
            filtered_custom_attributes = {
                k: v for k, v in tetmesh.custom_attributes.items() if k in builder.custom_attributes
            }
            if len(filtered_custom_attributes) != len(tetmesh.custom_attributes):
                # Preserve the cached TetMesh while keeping add_usd's
                # current behavior of dropping unregistered import attrs.
                tetmesh_for_builder = copy.copy(tetmesh)
                tetmesh_for_builder.custom_attributes = filtered_custom_attributes

        soft_mesh_mat = get_prim_world_mat(prim, None, incoming_world_xform)
        soft_mesh_pos, soft_mesh_rot, soft_mesh_scale = wp.transform_decompose(soft_mesh_mat)

        add_soft_mesh_kwargs = {
            "pos": soft_mesh_pos,
            "rot": soft_mesh_rot,
            "scale": 1.0,
            "vel": wp.vec3(0.0, 0.0, 0.0),
            "mesh": tetmesh_for_builder,
            "label": path,
        }
        # Body density overrides the TetMesh's material density.
        if is_volume_deformable:
            resolved_density = _resolve_deformable_density(prim, tetmesh_for_builder.density, deformable_read)
            if resolved_density is not None:
                add_soft_mesh_kwargs["density"] = resolved_density
        if is_uniform_scale(soft_mesh_scale):
            add_soft_mesh_kwargs["scale"] = float(np.array(soft_mesh_scale, dtype=np.float32)[0])
        else:
            add_soft_mesh_kwargs["vertices"] = tetmesh_for_builder.vertices * np.array(
                soft_mesh_scale, dtype=np.float32
            )

        soft_p0, soft_t0 = builder.particle_count, builder.tet_count
        builder.add_soft_mesh(**add_soft_mesh_kwargs)
        if is_volume_deformable:
            _apply_particle_masses(builder, prim, soft_p0, builder.particle_count, deformable_read)
        path_soft_map[path] = {
            "particle": (soft_p0, builder.particle_count),
            "tet": (soft_t0, builder.tet_count),
        }
        path_soft_attrs[path] = {
            # The density actually used: the resolved override if present, else the
            # TetMesh's own material density.
            "resolved_density": add_soft_mesh_kwargs.get("density", tetmesh_for_builder.density),
        }

        if verbose:
            print(f"Added soft mesh {path} with {tetmesh.vertex_count} vertices and {tetmesh.tet_count} tetrahedra.")


def _deformable_import_attachments(ctx: _DeformableImportContext, consumed_junction_attachment_paths: set[str]) -> None:
    """Lower supported AOUSD ``PhysicsAttachment`` prims onto the imported cables.

    Cable ``point`` / ``segment`` sites with ``type1 = "xform"`` become hard ball joints to the
    target xform / rigid body / world frame (``path_attachment_map``); curve-to-curve junctions
    already consumed as rod-graph topology are skipped, and unsupported sites (cloth/volume source,
    non-xform target, ...) are warned and preserved in ``path_attachment_attrs``.
    """
    from pxr import Usd

    builder = ctx.builder
    stage = ctx.stage
    root_prim = ctx.root_prim
    ignore_paths = ctx.ignore_paths
    incoming_world_xform = ctx.incoming_world_xform
    verbose = ctx.verbose
    deformable_read = ctx.deformable_read
    get_prim_world_mat = ctx.get_prim_world_mat
    get_rigid_body_ancestor_path = ctx.get_rigid_body_ancestor_path
    get_first_target = ctx.get_first_target
    path_body_map = ctx.path_body_map
    path_cable_segments = ctx.path_cable_segments
    path_cable_point_anchors = ctx.path_cable_point_anchors
    path_cloth_map = ctx.path_cloth_map
    path_soft_map = ctx.path_soft_map
    path_attachment_map = ctx.path_attachment_map
    path_attachment_attrs = ctx.path_attachment_attrs

    def _attachment_world_point_from_xform(target_path: str, local_point: wp.vec3) -> tuple[int, wp.vec3] | None:
        if target_path in ("", "/"):
            # A world target's coords are authored in stage space, so they ride the same
            # import/up-axis transform applied to the cable geometry (otherwise the anchor
            # stays in original USD coordinates and yanks a translated cable off-position).
            return -1, wp.transform_point(incoming_world_xform, local_point)

        target_prim = stage.GetPrimAtPath(target_path)
        if not target_prim or not target_prim.IsValid():
            return None

        target_mat = get_prim_world_mat(target_prim, None, incoming_world_xform)
        target_pos, target_rot, target_scale = wp.transform_decompose(target_mat)
        scaled_local = wp.vec3(
            float(local_point[0]) * float(target_scale[0]),
            float(local_point[1]) * float(target_scale[1]),
            float(local_point[2]) * float(target_scale[2]),
        )
        world_point = wp.transform_point(wp.transform(target_pos, target_rot), scaled_local)

        body_path = get_rigid_body_ancestor_path(target_prim)
        if body_path is None:
            return -1, world_point

        body_idx = path_body_map.get(body_path, -1)
        if body_idx < 0:
            return None
        local_body_point = wp.transform_point(wp.transform_inverse(_builder_body_xform(builder, body_idx)), world_point)
        return body_idx, local_body_point

    if not (root_prim and root_prim.IsValid()):
        return
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        if str(prim.GetTypeName()) != "PhysicsAttachment":
            continue

        path = str(prim.GetPath())
        if _is_ignored_path(path, ignore_paths):
            continue
        if path in consumed_junction_attachment_paths:
            continue  # already consumed as rod-graph topology (curve-to-curve junction)

        src0 = get_first_target(prim, "physics:src0")
        src1 = get_first_target(prim, "physics:src1")
        type0 = str(deformable_read(prim, "type0") or "")
        type1 = str(deformable_read(prim, "type1") or "")
        indices0 = [int(i) for i in (deformable_read(prim, "indices0") or [])]
        indices1 = [int(i) for i in (deformable_read(prim, "indices1") or [])]
        coords0 = _attachment_vec3_list(deformable_read(prim, "coords0"))
        coords1 = _attachment_vec3_list(deformable_read(prim, "coords1"))
        enabled_val = deformable_read(prim, "attachmentEnabled")
        enabled = True if enabled_val is None else bool(enabled_val)
        stiffness_val = deformable_read(prim, "stiffness")
        damping_val = deformable_read(prim, "damping")
        stiffness = math.inf if stiffness_val is None else float(stiffness_val)
        damping = 0.0 if damping_val is None else float(damping_val)

        attrs: dict[str, Any] = {
            "src0": src0,
            "src1": src1,
            "type0": type0,
            "type1": type1,
            "indices0": list(indices0),
            "indices1": list(indices1),
            "coords0": _attachment_vec3_tuples(coords0),
            "coords1": _attachment_vec3_tuples(coords1),
            "enabled": enabled,
            "stiffness": stiffness,
            "damping": damping,
        }
        path_attachment_attrs[path] = attrs

        if not enabled:
            continue
        if src0 not in path_cable_segments:
            if src0 in path_cloth_map or src0 in path_soft_map:
                _mark_attachment_unsupported(
                    attrs,
                    path,
                    "PhysicsAttachment on cloth/volume deformables is parsed but not imported yet; "
                    "Newton needs a deformable-site attachment constraint for this source type.",
                )
            else:
                _mark_attachment_unsupported(
                    attrs,
                    path,
                    f"physics:src0 target '{src0}' is not an imported cable deformable; skipping.",
                )
            continue
        if type0 not in ("point", "segment"):
            _mark_attachment_unsupported(
                attrs,
                path,
                f"PhysicsAttachment type0='{type0}' is not supported for imported cables; "
                "supported cable site types are 'point' and 'segment'.",
            )
            continue
        if type1 != "xform":
            _mark_attachment_unsupported(
                attrs,
                path,
                f"PhysicsAttachment type1='{type1}' is parsed but not imported yet; only xform targets "
                "are currently supported for cable attachments.",
            )
            continue
        if indices1:
            _mark_attachment_unsupported(
                attrs,
                path,
                "PhysicsAttachment type1='xform' must not author indices1; skipping.",
            )
            continue
        if not indices0:
            _mark_attachment_unsupported(attrs, path, "PhysicsAttachment has no indices0 attachment sites; skipping.")
            continue
        if type0 == "segment" and len(coords0) != len(indices0):
            _mark_attachment_unsupported(
                attrs,
                path,
                f"PhysicsAttachment coords0 length {len(coords0)} does not match indices0 length "
                f"{len(indices0)} for segment sites; skipping.",
            )
            continue
        if coords1 and len(coords1) != len(indices0):
            _mark_attachment_unsupported(
                attrs,
                path,
                f"PhysicsAttachment coords1 length {len(coords1)} does not match indices0 length "
                f"{len(indices0)} for xform sites; skipping.",
            )
            continue
        if not coords1:
            coords1 = [wp.vec3(0.0, 0.0, 0.0) for _ in indices0]

        if math.isfinite(stiffness) or damping != 0.0:
            warnings.warn(
                f"{path}: finite PhysicsAttachment stiffness/damping are not represented by Newton's "
                "current cable-to-xform lowering; importing as hard ball joint(s).",
                stacklevel=2,
            )

        joints: list[int] = []
        for site_idx, src_index in enumerate(indices0):
            coord0 = coords0[site_idx] if type0 == "segment" else None
            cable_anchors = _cable_attachment_anchors(
                path, src0, type0, src_index, coord0, path_cable_segments, path_cable_point_anchors
            )
            if cable_anchors is None:
                _mark_attachment_unsupported(
                    attrs,
                    path,
                    f"PhysicsAttachment type0='{type0}' could not be resolved on cable '{src0}'.",
                )
                break
            target_info = _attachment_world_point_from_xform(src1, coords1[site_idx])
            if target_info is None:
                warnings.warn(
                    f"{path}: physics:src1 target '{src1}' could not be resolved as an xform; "
                    "skipping that attachment site.",
                    stacklevel=2,
                )
                continue
            parent_body, parent_local = target_info
            for anchor_idx, (child_body, child_local) in enumerate(cable_anchors):
                label = f"{path}_site{site_idx}"
                if len(cable_anchors) > 1:
                    label = f"{label}_anchor{anchor_idx}"
                joint_idx = builder.add_joint_ball(
                    parent=parent_body,
                    child=child_body,
                    parent_xform=wp.transform(parent_local, wp.quat_identity()),
                    child_xform=wp.transform(child_local, wp.quat_identity()),
                    label=label,
                    enabled=True,
                )
                joints.append(joint_idx)

        if joints:
            path_attachment_map[path] = joints
            attrs["joint_indices"] = list(joints)
            if verbose:
                print(f"Added PhysicsAttachment {path} with {len(joints)} joint(s).")


def _deformable_remap_collapsed(
    path_cable_map: dict,
    path_attachment_map: dict,
    path_attachment_attrs: dict,
    joint_remap: Mapping[int, int],
    body_remap: Mapping[int, int],
    body_merged_parent: Mapping[int, int],
) -> tuple[dict, dict]:
    """Remap the cable / attachment index maps after ``collapse_fixed_joints``.

    Cable bodies/joints and attachment joints are addressed by index (not prim path), so they must
    ride the collapse remaps to stay valid. Returns the rebuilt ``path_cable_map`` and
    ``path_attachment_map``; ``path_attachment_attrs`` joint indices are refreshed in place.
    """

    def remap_body(body_id: int) -> int:
        # Mirror the path_body_map handling: a reindexed body is in body_remap; a body merged
        # away is resolved via its merge parent.
        if body_id in body_remap:
            return body_remap[body_id]
        if body_id in body_merged_parent:
            parent = body_merged_parent[body_id]
            return body_remap.get(parent, parent)
        return body_id

    if path_cable_map:
        path_cable_map = {
            path: ([remap_body(b) for b in bodies], [joint_remap.get(j, j) for j in joints])
            for path, (bodies, joints) in path_cable_map.items()
        }

    if path_attachment_map:
        path_attachment_map = {
            path: [joint_remap.get(j, j) for j in joints] for path, joints in path_attachment_map.items()
        }
        for path, joints in path_attachment_map.items():
            if path in path_attachment_attrs:
                path_attachment_attrs[path]["joint_indices"] = list(joints)

    return path_cable_map, path_attachment_map


def _deformable_import_element_collision_filters(ctx: _DeformableImportContext) -> None:
    """Lower AOUSD ``PhysicsElementCollisionFilter`` prims to shape collision filter pairs.

    Each prim suppresses collision between selected *elements* of ``src0`` and ``src1``. Supported
    element sources are imported cables (``groupElemIndices`` index the cable's segments) and rigid
    colliders (all of the collider's shapes). An empty index array means *all* elements of that
    source. Cloth/volume (triangle/tet) element sources have no per-element rigid shape in Newton's
    shape-filter model and are warned and skipped.
    """
    from pxr import Usd

    builder = ctx.builder
    root_prim = ctx.root_prim
    ignore_paths = ctx.ignore_paths
    deformable_read = ctx.deformable_read
    get_first_target = ctx.get_first_target
    verbose = ctx.verbose
    path_cable_segments = ctx.path_cable_segments
    path_body_map = ctx.path_body_map
    path_cloth_map = ctx.path_cloth_map
    path_soft_map = ctx.path_soft_map

    if not (root_prim and root_prim.IsValid()):
        return

    def _src_shapes(src_path: str, indices: list[int], filter_path: str) -> list[int] | None:
        # Resolve a source prim + element indices to the builder shape ids to filter.
        if src_path in path_cable_segments:
            segs = path_cable_segments[src_path]  # flat segment index -> (body, length)
            if indices:
                bodies = []
                for idx in indices:
                    if idx not in segs:
                        warnings.warn(
                            f"{filter_path}: element index {idx} is not an imported segment of cable "
                            f"'{src_path}'; skipping that element.",
                            stacklevel=2,
                        )
                        continue
                    bodies.append(segs[idx][0])
            else:
                bodies = [body for body, _length in segs.values()]  # empty indices -> all segments
            shapes: list[int] = []
            for b in bodies:
                shapes.extend(builder.body_shapes.get(b, []))
            return shapes
        if src_path in path_body_map:
            # A rigid collider: filter against all of its shapes (per-element indices not meaningful).
            return list(builder.body_shapes.get(path_body_map[src_path], []))
        if src_path in path_cloth_map or src_path in path_soft_map:
            warnings.warn(
                f"{filter_path}: PhysicsElementCollisionFilter on cloth/volume source '{src_path}' is not "
                "supported (no per-element rigid shapes); skipping.",
                stacklevel=2,
            )
            return None
        warnings.warn(
            f"{filter_path}: PhysicsElementCollisionFilter source '{src_path}' is not an imported "
            "deformable or collider; skipping.",
            stacklevel=2,
        )
        return None

    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        if str(prim.GetTypeName()) != "PhysicsElementCollisionFilter":
            continue
        path = str(prim.GetPath())
        if _is_ignored_path(path, ignore_paths):
            continue
        enabled = deformable_read(prim, "filterEnabled")
        if enabled is not None and not bool(enabled):
            continue
        src0 = get_first_target(prim, "physics:src0")
        src1 = get_first_target(prim, "physics:src1")
        idx0 = [int(i) for i in (deformable_read(prim, "groupElemIndices0") or [])]
        idx1 = [int(i) for i in (deformable_read(prim, "groupElemIndices1") or [])]
        shapes0 = _src_shapes(src0, idx0, path)
        shapes1 = _src_shapes(src1, idx1, path)
        if not shapes0 or not shapes1:
            continue
        pair_count = 0
        for sa in shapes0:
            for sb in shapes1:
                if sa != sb:
                    builder.add_shape_collision_filter_pair(sa, sb)
                    pair_count += 1
        if verbose:
            print(f"Applied PhysicsElementCollisionFilter {path}: {pair_count} shape pair(s).")
