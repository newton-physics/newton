# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""USD cable / curve-deformable import passes.

Imports linear ``UsdGeom.BasisCurves`` deformables as VBD rods, welding curve-to-curve
``PhysicsAttachment`` junctions into shared rod graphs first, then importing remaining single
curves. Driven by :func:`.import_usd.parse_usd` via a
:class:`.import_usd_deformable_utils._DeformableImportContext`.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import warp as wp

from .import_usd_deformable_utils import (
    _apply_cable_masses,
    _cable_segment_quaternions,
    _CurveDeformableRecord,
    _DeformableImportContext,
    _is_ignored_path,
    _resolve_deformable_density,
    _validate_attachment_index_pairs,
    _warn_geometry_authored_material_attrs,
    _warn_unsupported_rest_fields,
)


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
        # Apply the full world affine so non-uniform scale, shear, and reflections are exact.
        positions = [wp.transform_point(wmat, wp.vec3(float(p[0]), float(p[1]), float(p[2]))) for p in pts]
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

        # A welded graph would abort inside add_rod_graph on a degenerate (near-zero-length) edge from
        # duplicate or collapsed points. Reject the component with a warning instead, leaving its curves
        # to the per-curve pass (which warns and skips any individually-degenerate curve).
        if min((float(wp.length(node_positions[v] - node_positions[u])) for u, v in edges), default=0.0) <= 1.0e-8:
            warnings.warn(
                f"cable graph '{cid}': a welded curve has a zero-length segment (duplicate or collapsed "
                f"points); skipping the welded component so its curves import individually.",
                stacklevel=2,
            )
            return

        # add_rod_graph applies one scalar stiffness per component and auto-orients its segments, so a
        # welded curve's authored rest shape and per-point normals cannot be honored. Warn rather than
        # changing the curve's behavior silently (a single, unwelded curve does honor both).
        for key in comp_paths:
            kprim = curve_recs[key].prim
            if deformable_read(kprim, "restShapePoints") is not None:
                warnings.warn(
                    f"{key}: restShapePoints is dropped for a welded cable graph; its stiffness uses the "
                    f"current segment lengths (add_rod_graph's scalar stiffness cannot express per-segment "
                    f"rest lengths).",
                    stacklevel=2,
                )
            normals_attr = UsdGeom.BasisCurves(kprim).GetNormalsAttr()
            if UsdGeom.PrimvarsAPI(kprim).GetPrimvar("normals").HasValue() or (
                normals_attr and normals_attr.Get() is not None
            ):
                warnings.warn(
                    f"{key}: per-point normals are dropped for a welded cable graph; its segments use "
                    f"add_rod_graph's auto-orientation instead of the authored cross-section frame.",
                    stacklevel=2,
                )

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
        # Centerline points use the full affine (below) so reflections are exact; the decomposed
        # rot/scale still frame the authored normals and scale the rest lengths.
        _w_pos, w_rot, w_scale = wp.transform_decompose(world_mat)

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
                wp.transform_point(world_mat, wp.vec3(float(p[0]), float(p[1]), float(p[2]))) for p in local_pts
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
