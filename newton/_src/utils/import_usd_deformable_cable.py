# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""USD cable / curve-deformable import passes.

Imports linear ``UsdGeom.BasisCurves`` deformables as rods (chains of capsule bodies joined
by rod joints, usable by any solver that supports them), welding curve-to-curve
``PhysicsAttachment`` junctions into shared rod graphs first, then importing remaining single
curves. Driven by :func:`.import_usd.parse_usd` via a
:class:`.import_usd_deformable_utils._DeformableImportContext`.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from ..sim.builder import ModelBuilder

from .import_usd_deformable_utils import (
    _AOUSD_DEFAULT_POISSONS_RATIO,
    _AOUSD_DEFAULT_THICKNESS,
    _AOUSD_DEFAULT_YOUNGS_MODULUS,
    _apply_cable_masses,
    _bake_world_points,
    _cable_segment_quaternions,
    _CableMassRun,
    _CurveDeformableRecord,
    _deformable_body_skip_reason,
    _deformable_collision_enabled,
    _DeformableImportContext,
    _is_ignored_path,
    _read_deformable_element_array,
    _resolve_deformable_density,
    _set_cable_body_radius,
    _skip_for_deformable_body_owner,
    _UnionFind,
    _validate_attachment_index_pairs,
    _warn_collision_approximated,
    _warn_dropped_velocities,
    _warn_geometry_authored_material_attrs,
    _warn_subset_material_bindings,
    _warn_unsupported_rest_fields,
)

_AOUSD_CIRCULAR_SECTION_SHEAR_CORRECTION = 0.9
# Attributes introduced after the family-prefix rename; density is shared and intentionally omitted.
_POST_RENAME_CURVE_MATERIAL_ATTRS = (
    "curvesThickness",
    "youngsModulus",
    "poissonsRatio",
    "curvesStretchStiffness",
    "curvesShearStiffness",
    "curvesBendStiffness",
    "curvesTwistStiffness",
)
_LEGACY_CURVE_MATERIAL_ATTRS = (
    "thickness",
    "stretchStiffness",
    "shearStiffness",
    "bendStiffness",
    "twistStiffness",
)
# Removed family-prefixed thickness first, then the earlier unprefixed compatibility name.
_CABLE_THICKNESS_ATTRS = ("curvesThickness", "thickness")
_NEWTON_CURVE_DAMPING_ATTRS = (
    "curvesStretchDamping",
    "curvesShearDamping",
    "curvesBendDamping",
    "curvesTwistDamping",
)


def _curve_thickness_samples(
    authored, vertex_counts: list[int], closed: bool, fallback: float
) -> tuple[list[float], list[float]]:
    """Resolve curve thicknesses at flattened segments and points."""
    segment_counts = [count if closed else max(0, count - 1) for count in vertex_counts]
    if authored is None:
        return [fallback] * sum(segment_counts), [fallback] * sum(vertex_counts)
    if authored.element_type == "constant":
        return [authored.values[0]] * sum(segment_counts), [authored.values[0]] * sum(vertex_counts)
    if authored.element_type == "curve":
        segment_thicknesses = [
            authored.values[curve]
            for curve, segment_count in enumerate(segment_counts)
            for _segment in range(segment_count)
        ]
        point_thicknesses = [
            authored.values[curve] for curve, point_count in enumerate(vertex_counts) for _point in range(point_count)
        ]
        return segment_thicknesses, point_thicknesses
    if authored.element_type == "segment":
        point_thicknesses: list[float] = []
        segment_offset = 0
        for point_count, segment_count in zip(vertex_counts, segment_counts, strict=True):
            values = authored.values[segment_offset : segment_offset + segment_count]
            if point_count == 0:
                pass
            elif segment_count == 0:
                point_thicknesses.extend([fallback] * point_count)
            elif closed:
                point_thicknesses.extend(
                    0.5 * (values[(point - 1) % segment_count] + values[point]) for point in range(point_count)
                )
            else:
                point_thicknesses.append(values[0])
                point_thicknesses.extend(
                    0.5 * (values[point - 1] + values[point]) for point in range(1, point_count - 1)
                )
                point_thicknesses.append(values[-1])
            segment_offset += segment_count
        return list(authored.values), point_thicknesses

    segment_thicknesses: list[float] = []
    point_offset = 0
    for count in vertex_counts:
        values = authored.values[point_offset : point_offset + count]
        if count == 0:
            continue
        segment_thicknesses.extend((values[index] + values[index + 1]) * 0.5 for index in range(count - 1))
        if closed:
            segment_thicknesses.append((values[-1] + values[0]) * 0.5)
        point_offset += count
    return segment_thicknesses, list(authored.values)


def _read_validated_curve_topology(curves, path: str, *, warn: bool = True):
    """Read a cable prim's ``points`` / ``curveVertexCounts`` after validating the partition.

    Counts must be non-negative and sum to exactly ``len(points)``: Python slicing is
    forgiving, so a mismatch would otherwise corrupt every later curve's point offset or
    reach ``add_rod`` with fewer positions than declared (which raises out of the import).
    Shared by the graph prepass and the per-curve pass so the two cannot diverge. Returns
    ``(points, counts)`` with counts as Python ints, or ``None`` for a prim that must be
    skipped whole (warned unless ``warn=False``; the prepass passes ``False`` because an
    unconsumed prim always reaches the per-curve pass, which warns).
    """
    points = curves.GetPointsAttr().Get()
    vertex_counts = curves.GetCurveVertexCountsAttr().Get()
    if not points or not vertex_counts:
        if warn:
            warnings.warn(f"{path}: cable curve has no points / curveVertexCounts; skipping.", stacklevel=2)
        return None
    counts = [int(c) for c in vertex_counts]
    for i, count in enumerate(counts):
        if count < 0:
            if warn:
                warnings.warn(
                    f"{path}: curveVertexCounts[{i}] is {count}; counts must be non-negative; "
                    f"skipping malformed cable.",
                    stacklevel=2,
                )
            return None
    total = sum(counts)
    if total != len(points):
        if warn:
            warnings.warn(
                f"{path}: curveVertexCounts total {total} does not match points length {len(points)}; "
                f"skipping malformed cable.",
                stacklevel=2,
            )
        return None
    return points, counts


def _read_cable_rest_shape_points(prim, path: str, point_count: int, deformable_read):
    """Read count-matched cable ``restShapePoints``."""
    rest_shape_points = deformable_read(prim, "restShapePoints")
    if rest_shape_points is None:
        return None
    if len(rest_shape_points) != point_count:
        warnings.warn(
            f"{path}: restShapePoints length {len(rest_shape_points)} != points {point_count}; "
            f"ignoring rest shape (rest length taken from the imported points).",
            stacklevel=2,
        )
        return None
    return rest_shape_points


def _cable_rest_segment_lengths(rest_points, world_mat, path: str, *, closed: bool) -> list[float] | None:
    """World-space rest segment lengths, or ``None`` (with a warning) if any segment is invalid.

    Applies the full affine so lengths stay exact under reflection / shear (translation cancels in
    the segment differences). Rejecting the whole curve, rather than the offending segment, keeps
    an invalid rest shape from mixing rest and current lengths along one cable.
    """
    points = _bake_world_points(rest_points, world_mat)
    if closed:
        points = [*points, points[0]]
    lengths = [float(wp.length(points[i + 1] - points[i])) for i in range(len(points) - 1)]
    if not lengths or any(not math.isfinite(length) or length <= 1.0e-8 for length in lengths):
        warnings.warn(
            f"{path}: restShapePoints has a non-finite or zero-length segment; ignoring rest shape "
            f"(rest length taken from the imported points).",
            stacklevel=2,
        )
        return None
    return lengths


def _warn_cable_rest_shape_effect(path: str) -> None:
    """Report the intentionally limited effect of a valid cable rest shape."""
    warnings.warn(
        f"{path}: restShapePoints does not establish the simulated rest state; it is used only for "
        f"material-gain discretization when stiffness or damping is available.",
        stacklevel=2,
    )


def _warn_geometry_authored_newton_curve_damping_attrs(prim, path: str) -> None:
    """Warn for Newton curve damping attributes misplaced on curve geometry.

    Args:
        prim: Curve geometry prim to inspect.
        path: Prim path to use in diagnostics.
    """
    for name in _NEWTON_CURVE_DAMPING_ATTRS:
        attr = prim.GetAttribute(f"newton:{name}")
        if attr and attr.HasAuthoredValue():
            warnings.warn(
                f"{path}: deformable material attribute 'newton:{name}' is authored on the geometry; "
                "it belongs on the bound material (NewtonCurvesDeformableMaterialAPI) and is ignored.",
                stacklevel=2,
            )


def _has_legacy_curve_material(material: dict[str, float]) -> bool:
    """Whether attributes from the earlier AOUSD curve-material revision are authored."""
    return any(name in material for name in _LEGACY_CURVE_MATERIAL_ATTRS)


def _is_legacy_only_curve_material(material: dict[str, float]) -> bool:
    """Whether legacy attributes are authored without any current-revision attributes."""
    return _has_legacy_curve_material(material) and not any(
        name in material for name in _POST_RENAME_CURVE_MATERIAL_ATTRS
    )


def _warn_legacy_curve_material(path: str, material: dict[str, float] | None) -> None:
    """Warn when an earlier curve-material attribute is authored."""
    if material is not None and "curvesThickness" in material:
        warnings.warn(
            f"{path}: physics:curvesThickness on the curve material was removed from the AOUSD "
            "proposal and is deprecated; move diameter d to physics:thicknesses = [d] on the "
            "simulation geometry and set physics:thicknesses:elementType = 'constant'.",
            DeprecationWarning,
            stacklevel=2,
        )
    if material is not None and _has_legacy_curve_material(material):
        warnings.warn(
            f"{path}: unprefixed curve material attributes follow an earlier AOUSD proposal revision "
            f"and are deprecated; migrate physics:thickness to physics:thicknesses on the simulation "
            f"geometry with physics:thicknesses:elementType, and convert "
            f"all four resolved legacy modes, including the stretch-to-shear and bend-to-twist "
            f"fallbacks, to structural values before authoring physics:curves*Stiffness.",
            DeprecationWarning,
            stacklevel=2,
        )


def _cable_radius_from_material(material: dict[str, float] | None, linear_unit: float) -> float:
    """Resolve radius from an authored thickness, else the revision-appropriate assumed value.

    Removed material thickness attributes remain compatibility fallbacks; otherwise the proposal's
    one-millimeter diameter default applies in stage units.
    """
    if material is not None:
        for name in _CABLE_THICKNESS_ATTRS:
            if name in material:
                return 0.5 * material[name]
    return 0.5 * _AOUSD_DEFAULT_THICKNESS / linear_unit


def _resolve_cable_structural_stiffnesses(
    material: dict[str, float], radius: float, linear_unit: float
) -> tuple[float | None, float | None, float | None, float | None]:
    """Resolve material-level stretch, shear, bend, and twist structural stiffnesses.

    Per mode, an authored ``physics:curves*Stiffness`` wins, then the deprecated unprefixed
    modulus times its cross-section factor, then the AOUSD derivation from ``E`` and ``nu``.
    A material authoring *only* deprecated attributes keeps its former behavior instead: it
    converts what it authored, inherits shear from stretch and twist from bend, and leaves an
    unresolved mode ``None`` so that the rod-builder default stands.
    """
    area = math.pi * radius**2
    area_moment = 0.25 * math.pi * radius**4
    polar_moment = 0.5 * math.pi * radius**4

    if _is_legacy_only_curve_material(material):
        stretch = material.get("stretchStiffness")
        shear = material.get("shearStiffness")
        bend = material.get("bendStiffness")
        twist = material.get("twistStiffness")
        stretch = stretch * area if stretch is not None else None
        shear = shear * area if shear is not None else stretch
        bend = bend * area_moment if bend is not None else None
        twist = twist * polar_moment if twist is not None else bend
        return stretch, shear, bend, twist

    # Positions and radius use stage distance units. With the supported unit-mass stages, an SI
    # pressure is multiplied by meters-per-unit in stage units; an SI length is divided by it.
    youngs = material.get("youngsModulus", _AOUSD_DEFAULT_YOUNGS_MODULUS * linear_unit)
    poissons = material.get("poissonsRatio", _AOUSD_DEFAULT_POISSONS_RATIO)
    shear_modulus = youngs / (2.0 * (1.0 + poissons))

    def resolve(name: str, legacy_name: str, geometric_factor: float, modulus: float) -> float:
        # A deprecated modulus and the AOUSD derivation both scale the same cross-section factor.
        if name in material:
            return material[name]
        return material.get(legacy_name, modulus) * geometric_factor

    return (
        resolve("curvesStretchStiffness", "stretchStiffness", area, youngs),
        resolve(
            "curvesShearStiffness",
            "shearStiffness",
            area,
            _AOUSD_CIRCULAR_SECTION_SHEAR_CORRECTION * shear_modulus,
        ),
        resolve("curvesBendStiffness", "bendStiffness", area_moment, youngs),
        resolve("curvesTwistStiffness", "twistStiffness", polar_moment, shear_modulus),
    )


def _resolve_cable_structural_dampings(
    material: dict[str, float],
) -> tuple[float | None, float | None, float | None, float | None]:
    """Resolve independently authored Newton stretch, shear, bend, and twist structural damping.

    Args:
        material: Validated curve-material values.
    """
    return tuple(material.get(name) for name in _NEWTON_CURVE_DAMPING_ATTRS)


def _apply_local_rod_material_gains(
    builder: ModelBuilder,
    bodies: list[int],
    joints: list[int],
    segment_rest_lengths: list[float],
    material: dict[str, float] | None,
    segment_radii: list[float],
    joint_radii: list[float],
    linear_unit: float,
) -> None:
    """Discretize cable structural stiffness and damping using each joint's dual rest length.

    A joint spans half of each adjacent segment, so its length is ``0.5 * (L_parent + L_child)``.
    ``segment_rest_lengths[i]`` is the rest length of the segment body ``bodies[i]``. A curve
    material resolves every current AOUSD stiffness independently. Newton damping is independently
    optional per mode, so an unauthored mode leaves the rod-builder default unchanged.

    Args:
        builder: Model builder containing the imported rod joints.
        bodies: Segment body indices corresponding to ``segment_rest_lengths``.
        joints: Rod joint indices to update.
        segment_rest_lengths: Rest length of each segment body.
        material: Validated curve-material values, or ``None`` when no supported material applies.
        segment_radii: Radius of each segment body in stage distance units.
        joint_radii: Radius sampled at each rod joint in stage distance units.
        linear_unit: Meters per stage distance unit.
    """
    if material is None:
        return

    body_rest_lengths = dict(zip(bodies, segment_rest_lengths, strict=True))
    structural_dampings = _resolve_cable_structural_dampings(material)
    if segment_radii and all(radius == segment_radii[0] for radius in segment_radii[1:]):
        uniform_segment_values = _resolve_cable_structural_stiffnesses(material, segment_radii[0], linear_unit)
        body_structural_values = dict.fromkeys(bodies, uniform_segment_values)
    else:
        body_structural_values = {
            body: _resolve_cable_structural_stiffnesses(material, radius, linear_unit)
            for body, radius in zip(bodies, segment_radii, strict=True)
        }
    uniform_joint_values = None
    if joint_radii and all(radius == joint_radii[0] for radius in joint_radii[1:]):
        uniform_joint_values = _resolve_cable_structural_stiffnesses(material, joint_radii[0], linear_unit)

    def series_stiffness(
        parent_value: float | None,
        child_value: float | None,
        parent_segment_length: float,
        child_segment_length: float,
    ) -> float | None:
        if parent_value is None or child_value is None:
            return None
        if parent_value == 0.0 or child_value == 0.0:
            return 0.0
        return 1.0 / (0.5 * parent_segment_length / parent_value + 0.5 * child_segment_length / child_value)

    for joint, joint_radius in zip(joints, joint_radii, strict=True):
        parent = builder.joint_parent[joint]
        child = builder.joint_child[joint]
        parent_length = body_rest_lengths[parent]
        child_length = body_rest_lengths[child]
        parent_values = body_structural_values[parent]
        child_values = body_structural_values[child]

        stretch = series_stiffness(parent_values[0], child_values[0], parent_length, child_length)
        shear = series_stiffness(parent_values[1], child_values[1], parent_length, child_length)
        joint_rest_length = 0.5 * (parent_length + child_length)
        joint_values = (
            uniform_joint_values
            if uniform_joint_values is not None
            else _resolve_cable_structural_stiffnesses(material, joint_radius, linear_unit)
        )
        bend = None if joint_values[2] is None else joint_values[2] / joint_rest_length
        twist = None if joint_values[3] is None else joint_values[3] / joint_rest_length
        stretch_damping, shear_damping, bend_damping, twist_damping = (
            None if damping is None else damping / joint_rest_length for damping in structural_dampings
        )

        builder._set_joint_rod_material_gains(
            joint,
            stretch_stiffness=stretch,
            stretch_damping=stretch_damping,
            shear_stiffness=shear,
            shear_damping=shear_damping,
            bend_stiffness=bend,
            bend_damping=bend_damping,
            twist_stiffness=twist,
            twist_damping=twist_damping,
        )


def _deformable_import_cable_graphs(ctx: _DeformableImportContext) -> tuple[set[str], set[str]]:
    """Weld curve deformables joined by curve-to-curve ``PhysicsAttachment`` prims into
    rod graphs via :meth:`ModelBuilder.add_rod_graph`.

    A hard (unauthored / infinite stiffness) ``point``->``point`` attachment whose
    ``src0``/``src1`` are both imported curve deformables and whose sites are coincident is
    topology, not a runtime constraint: the two referenced control points are the same junction
    node. Curves transitively joined this way form one graph component, built with a single
    ``add_rod_graph`` call (one capsule body per segment, junction nodes shared). Compliant or
    non-coincident curve-to-curve attachments are NOT welded; they warn here and are preserved
    as unsupported in ``path_attachment_attrs`` by the attachment post-pass.
    Returns the curve prim paths and the junction attachment prim paths consumed here so the
    per-curve cable pass and the attachment post-pass skip them. Single curves and
    curve-to-xform attachments are left to those passes.

    A welded graph uses the first curve's stiffness and damping material as representative
    (heterogeneous gains warn), while density and simulation-geometry thickness stay local to each
    segment. Each joint's gains are discretized using its own two adjacent rest lengths, taken per
    curve from ``restShapePoints`` when authored. Each curve's authored material is still reported
    in ``path_cable_attrs``.
    """
    from pxr import UsdGeom

    from ..usd import utils as usd  # noqa: PLC0415

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
    for prim in ctx.prims.cables:
        path = str(prim.GetPath())
        if _is_ignored_path(path, ignore_paths):
            continue
        # Disabled/kinematic curves must not be welded into a graph; the per-curve pass warns.
        if _deformable_body_skip_reason(prim, deformable_read) is not None:
            continue
        # A non-owner curve under a governed deformable body is skipped by the per-curve pass
        # (which warns); it must not silently join a welded graph either.
        if _skip_for_deformable_body_owner(ctx, prim, path, warn=False):
            continue
        curves = UsdGeom.BasisCurves(prim)
        if curves.GetTypeAttr().Get() != UsdGeom.Tokens.linear:
            continue
        topo = _read_validated_curve_topology(curves, path, warn=False)
        if topo is None:
            continue
        pts, vcounts = topo
        if len(vcounts) != 1 or vcounts[0] < 3:
            continue
        wmat = get_prim_world_mat(prim, None, incoming_world_xform)
        # Apply the full world affine so non-uniform scale, shear, and reflections are exact.
        positions = _bake_world_points(pts, wmat)
        mat = usd._get_curve_deformable_material(prim, deformable_read)
        radius = _cable_radius_from_material(mat, linear_unit)
        closed = curves.GetWrapAttr().Get() == UsdGeom.Tokens.periodic
        segment_count = len(pts) if closed else len(pts) - 1
        thicknesses = _read_deformable_element_array(
            prim,
            "thicknesses",
            {"constant": 1, "curve": 1, "segment": segment_count, "point": len(pts)},
            deformable_read,
        )
        segment_thicknesses, point_thicknesses = _curve_thickness_samples(thicknesses, [len(pts)], closed, 2.0 * radius)
        segment_radii = [0.5 * thickness for thickness in segment_thicknesses]
        density = _resolve_deformable_density(
            prim,
            None if mat is None else mat.get("density"),
            deformable_read,
            linear_unit,
            read_base_material=mat is None,
        )
        curve_recs[path] = _CurveDeformableRecord(
            prim=prim,
            positions=positions,
            closed=closed,
            material=mat,
            segment_radii=segment_radii,
            point_radii=[0.5 * thickness for thickness in point_thicknesses],
            density=density,
            thicknesses=thicknesses,
        )

    if not curve_recs:
        return consumed_curves, consumed_attachments

    # Union-find over curve prim paths; record the per-attachment welded point pairs.
    curve_sets = _UnionFind(curve_recs)

    welds: list[tuple[str, int, str, int]] = []
    weld_attachments: list[tuple[str, str]] = []  # (src0 curve path, attachment prim path)
    for prim in ctx.prims.attachments:
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
        # Welding replaces the attachment constraint with shared topology, which is only
        # equivalent for a hard (unauthored / infinite stiffness) attachment whose sites
        # already occupy the same point. A compliant or non-coincident junction is left to
        # the attachment post-pass, which preserves it in path_attachment_attrs as
        # unsupported instead of silently snapping the geometry together.
        stiffness_val = deformable_read(prim, "stiffness")
        damping_val = deformable_read(prim, "damping")
        stiffness, damping = usd._resolve_attachment_gains(prim, stiffness_val, damping_val)
        # Invalid gains must reach the attachment pass instead of being consumed as topology.
        # That pass records one detailed warning and preserves the authored metadata.
        if (
            stiffness is None
            or damping is None
            or math.isnan(stiffness)
            or stiffness < 0.0
            or not (math.isfinite(damping) and damping >= 0.0)
        ):
            continue
        # Hard means the proposal's +inf stiffness sentinel exactly; NaN, -inf, or finite
        # values (compliant or nonconforming) must not weld curves into shared topology.
        # Valid damping does not affect hardness: it only applies when the constraint is not hard.
        hard = stiffness == math.inf
        if not hard:
            warnings.warn(
                f"{prim.GetPath()}: curve-to-curve attachment does not author a hard "
                f"(+inf stiffness) constraint; not welded.",
                stacklevel=2,
            )
            continue
        # A tenth of the thinner attached point's radius: welding then moves geometry by well
        # under the junction bodies' own overlap, so the weld is equivalent to the authored
        # constraint even when thickness varies along either cable.
        if any(
            float(wp.length(curve_recs[src0].positions[a] - curve_recs[src1].positions[b]))
            > 0.1 * min(curve_recs[src0].point_radii[a], curve_recs[src1].point_radii[b])
            for a, b in zip(idx0, idx1, strict=True)
        ):
            warnings.warn(
                f"{prim.GetPath()}: curve-to-curve attachment sites are not coincident; not welded "
                f"(welding would move the authored geometry).",
                stacklevel=2,
            )
            continue
        curve_sets.union(src0, src1)
        for a, b in zip(idx0, idx1, strict=True):
            welds.append((src0, a, src1, b))
        # Consumed only after the component actually builds (below): a failed graph falls back
        # to the per-curve pass, and its junction must reach the attachment pass so the authored
        # constraint is preserved instead of silently dropped.
        weld_attachments.append((src0, str(prim.GetPath())))

    components: dict[str, list[str]] = {}
    for p in curve_recs:
        components.setdefault(curve_sets.find(p), []).append(p)
    welds_by_comp: dict[str, list[tuple[str, int, str, int]]] = {}
    for w in welds:
        welds_by_comp.setdefault(curve_sets.find(w[0]), []).append(w)
    attachments_by_comp: dict[str, set[str]] = {}
    for src0, att_path in weld_attachments:
        attachments_by_comp.setdefault(curve_sets.find(src0), set()).add(att_path)

    def _build_graph_component(cid, comp_paths, comp_welds) -> bool:
        # Merge welded control points into shared graph nodes (union-find over (path, index)).
        nodes = _UnionFind((key, i) for key in comp_paths for i in range(len(curve_recs[key].positions)))
        for s0, i0, s1, i1 in comp_welds:
            nodes.union((s0, i0), (s1, i1))

        node_positions: list[wp.vec3] = []
        node_id: dict[tuple[str, int], int] = {}

        def global_node(local: tuple[str, int]) -> int:
            root = nodes.find(local)
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
                    # Welding merged both endpoints of an authored segment into one node.
                    # Dropping the segment would silently import different topology than
                    # authored (and desynchronize the segment/attachment mappings); reject
                    # the weld instead so the curves import individually and the junction
                    # reaches the attachment pass, mirroring the cycle rejection below.
                    warnings.warn(
                        f"cable graph '{cid}': welding collapses segment {seg} of '{key}' (both "
                        f"endpoints merge into one node); skipping the weld so its curves import "
                        f"individually.",
                        stacklevel=2,
                    )
                    return False
                edges.append((gu, gv))
                edge_owner.append((key, seg))

        if len(node_positions) < 2 or not edges:
            return False
        node_radius_samples: list[list[float]] = [[] for _ in node_positions]
        for key in comp_paths:
            for point, point_radius in enumerate(curve_recs[key].point_radii):
                node_radius_samples[global_node((key, point))].append(point_radius)
        node_radii = [sum(samples) / len(samples) for samples in node_radius_samples]

        # A connected component with as many edges as merged nodes contains a cycle (e.g. a
        # welded periodic curve). add_rod_graph builds a spanning tree and cannot close the
        # loop, which would silently change the authored topology; reject the weld instead so
        # the curves import individually (a periodic curve keeps its loop-closing joint) and
        # the junction reaches the attachment pass.
        if len(edges) >= len(node_positions):
            warnings.warn(
                f"cable graph '{cid}': the welded component contains a cycle, which a rod graph "
                f"cannot close; skipping the weld so its curves import individually.",
                stacklevel=2,
            )
            return False

        # A welded graph would abort inside add_rod_graph on a degenerate (near-zero-length) edge from
        # duplicate or collapsed points. Reject the component with a warning instead, leaving its curves
        # to the per-curve pass (which warns and skips any individually-degenerate curve).
        edge_lengths = [float(wp.length(node_positions[v] - node_positions[u])) for u, v in edges]
        if min(edge_lengths, default=0.0) <= 1.0e-8:
            warnings.warn(
                f"cable graph '{cid}': a welded curve has a zero-length segment (duplicate or collapsed "
                f"points); skipping the welded component so its curves import individually.",
                stacklevel=2,
            )
            return False

        # Rest lengths drive material-gain discretization; edge_lengths stays current for the anchor map below.
        material_edge_lengths = edge_lengths.copy()
        rest_lengths_by_curve: dict[str, list[float]] = {}
        for key in comp_paths:
            rec = curve_recs[key]
            rest_shape_points = _read_cable_rest_shape_points(rec.prim, key, len(rec.positions), deformable_read)
            if rest_shape_points is None:
                continue
            wmat = get_prim_world_mat(rec.prim, None, incoming_world_xform)
            rest_lengths = _cable_rest_segment_lengths(rest_shape_points, wmat, key, closed=rec.closed)
            if rest_lengths is not None:
                rest_lengths_by_curve[key] = rest_lengths
        for edge_index, (key, segment_index) in enumerate(edge_owner):
            if key in rest_lengths_by_curve:
                material_edge_lengths[edge_index] = rest_lengths_by_curve[key][segment_index]

        # add_rod_graph auto-orients segments, so authored cross-section frames cannot be honored.
        for key in comp_paths:
            kprim = curve_recs[key].prim
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
        for key in comp_paths:
            rec = curve_recs[key]
            _warn_geometry_authored_material_attrs(rec.prim, key, "PhysicsCurvesDeformableMaterialAPI", deformable_read)
            _warn_geometry_authored_newton_curve_damping_attrs(rec.prim, key)
            _warn_legacy_curve_material(key, rec.material)
        # A welded graph necessarily flattens stiffness and damping to one representative
        # material. Density and geometry thickness remain per segment.
        if len(comp_paths) > 1:
            sigs = {
                (
                    curve_recs[p].material is not None,
                    frozenset(
                        (name, value) for name, value in (curve_recs[p].material or {}).items() if name != "density"
                    ),
                )
                for p in comp_paths
            }
            if len(sigs) > 1:
                warnings.warn(
                    f"cable graph '{cid}': welded curves have differing stiffness/damping; using "
                    f"'{comp_paths[0]}' as the representative material gains for the whole component. "
                    "Density remains local to each curve.",
                    stacklevel=2,
                )
        radius = rep.segment_radii[0]
        mat = rep.material
        # One rod graph has one shape config, so collision is resolved per component:
        # any collision-enabled member curve makes the whole graph collide.
        collision_states = {p: _deformable_collision_enabled(curve_recs[p].prim, ctx.ignore_paths) for p in comp_paths}
        for p, (_enabled, approximated_from) in collision_states.items():
            _warn_collision_approximated(p, approximated_from)
        collision_enabled = any(enabled for enabled, _src in collision_states.values())
        if collision_enabled and not all(enabled for enabled, _src in collision_states.values()):
            warnings.warn(
                f"cable graph '{cid}': welded cables mix collision-enabled and collision-disabled "
                f"curves; the whole graph collides.",
                stacklevel=2,
            )
        # A zero representative density still needs geometric weights when any member
        # curve authors a body mass total (the per-curve rescale distributes it).
        graph_weight_density = rep.density
        if graph_weight_density <= 0.0 and any(
            usd._get_deformable_body_overrides(curve_recs[p].prim, deformable_read)[0] is not None for p in comp_paths
        ):
            graph_weight_density = 1.0
        cfg = replace(
            builder.default_shape_cfg,
            density=graph_weight_density,
            has_shape_collision=collision_enabled,
            has_particle_collision=collision_enabled,
        )
        # Unlike single cables, the graph junction spanning tree is intrinsic topology, not a
        # caller choice, and only a tree (not the all-incident-edges joint set produced when
        # unwrapped) is articulation-safe. So the importer wraps each component into its own
        # articulation here; path_cable_map exposes empty joints for graph curves accordingly.
        body_ids, graph_joint_ids = builder.add_rod_graph(
            node_positions=node_positions,
            edges=edges,
            radius=radius,
            cfg=cfg,
            label=cid,
            wrap_in_articulation=True,
            body_frame_origin="com",
        )
        edge_radii = [curve_recs[key].segment_radii[segment] for key, segment in edge_owner]
        body_radii = dict(zip(body_ids, edge_radii, strict=True))
        for body, edge_radius in zip(body_ids, edge_radii, strict=True):
            _set_cable_body_radius(builder, body, edge_radius)
        body_nodes = dict(zip(body_ids, edges, strict=True))
        graph_joint_radii = []
        for joint in graph_joint_ids:
            parent = builder.joint_parent[joint]
            child = builder.joint_child[joint]
            shared_nodes = set(body_nodes[parent]).intersection(body_nodes[child])
            if len(shared_nodes) == 1:
                graph_joint_radii.append(node_radii[shared_nodes.pop()])
            else:
                graph_joint_radii.append(0.5 * (body_radii[parent] + body_radii[child]))
        _apply_local_rod_material_gains(
            builder,
            body_ids,
            graph_joint_ids,
            material_edge_lengths,
            mat,
            edge_radii,
            graph_joint_radii,
            linear_unit,
        )
        for key in rest_lengths_by_curve:
            _warn_cable_rest_shape_effect(key)

        # Partition graph bodies back to their owning curve, and rebuild the per-prim anchor
        # maps the curve-to-xform attachment pass reads (point index / segment index -> body).
        per_prim_segments: dict[str, dict[int, tuple[int, float]]] = {}
        per_prim_bodies: dict[str, list[int]] = {}
        for ge, (key, seg) in enumerate(edge_owner):
            length = edge_lengths[ge]
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
                "material": dict(rec.material or {}),
                "resolved_density": rec.density,
                "closed": rec.closed,
                "graph_component": cid,
            }
            key_bodies = per_prim_bodies.get(key, [])
            if key_bodies:
                # Edges are assembled curve-by-curve, so each curve's graph bodies are contiguous.
                # A welded curve owns no individual tree joints (they live in the shared graph
                # articulation, found via articulation_label), so its joint range is empty.
                builder._record_cable_group(
                    key, (key_bodies[0], key_bodies[-1] + 1), (builder.joint_count, builder.joint_count)
                )
            segment_count = n if rec.closed else n - 1
            run = _CableMassRun(
                curve_index=0,
                point_offset=0,
                point_count=n,
                segment_offset=0,
                body_ids=tuple(key_bodies),
                element_volumes=tuple(
                    math.pi * rec.segment_radii[index] ** 2 * segs[index][1] for index in range(segment_count)
                ),
                closed=rec.closed,
            )
            authored_masses = _apply_cable_masses(
                builder,
                rec.prim,
                [run],
                deformable_read,
                n,
                1,
                segment_count,
                rec.density,
            )
            if rec.thicknesses is not None or authored_masses is not None:
                path_cable_attrs[key]["simulation"] = {}
                if rec.thicknesses is not None:
                    path_cable_attrs[key]["simulation"]["thicknesses"] = {
                        "values": list(rec.thicknesses.values),
                        "element_type": rec.thicknesses.element_type,
                    }
                if authored_masses is not None:
                    path_cable_attrs[key]["simulation"]["masses"] = {
                        "values": list(authored_masses.values),
                        "element_type": authored_masses.element_type,
                        "legacy_implicit_type": authored_masses.legacy_implicit_type,
                    }
            consumed_curves.add(key)
        if verbose:
            print(f"Added cable graph {cid} with {len(body_ids)} segments across {len(comp_paths)} curves.")
        return True

    for cid, comp_curves in components.items():
        comp_paths = sorted(comp_curves)
        comp_welds = welds_by_comp.get(cid, [])
        if len(comp_paths) == 1 and not comp_welds:
            continue  # plain single curve: leave it to the per-curve pass
        if _build_graph_component(cid, comp_paths, comp_welds):
            consumed_attachments.update(attachments_by_comp.get(cid, ()))

    return consumed_curves, consumed_attachments


def _deformable_import_cable(ctx: _DeformableImportContext, consumed_cable_curve_paths: set[str]) -> None:
    """Import single-curve cable deformables (linear ``GeomBasisCurves`` -> rod via ``add_rod``).

    Curves already welded into a rod graph (``consumed_cable_curve_paths``) are skipped. Each cable is
    wrapped into its own articulation so the model is finalize-ready. Results land in
    ``path_cable_map`` / attrs / segments / point anchors.
    """
    from pxr import UsdGeom

    from ..usd import utils as usd  # noqa: PLC0415

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
    for prim in ctx.prims.cables:
        path = str(prim.GetPath())
        if path in consumed_cable_curve_paths:
            continue  # already built as part of a welded rod graph
        if _is_ignored_path(path, ignore_paths):
            continue
        skip_reason = _deformable_body_skip_reason(prim, deformable_read)
        if skip_reason is not None:
            warnings.warn(f"{path}: {skip_reason}; skipping cable import.", stacklevel=2)
            continue
        if _skip_for_deformable_body_owner(ctx, prim, path):
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
        topo = _read_validated_curve_topology(curves, path)
        if topo is None:
            continue
        points, vertex_counts = topo
        closed = curves.GetWrapAttr().Get() == UsdGeom.Tokens.periodic
        # Rest centerline used for the rest length below (one point per vertex); restNormals
        # and rest bend angles are not imported yet.
        rest_shape_points = _read_cable_rest_shape_points(prim, path, len(points), deformable_read)
        _warn_unsupported_rest_fields(prim, path, ("restNormals",), deformable_read)
        _warn_dropped_velocities(prim, path)
        _warn_geometry_authored_material_attrs(prim, path, "PhysicsCurvesDeformableMaterialAPI", deformable_read)
        _warn_geometry_authored_newton_curve_damping_attrs(prim, path)
        _warn_subset_material_bindings(prim, path)

        world_mat = get_prim_world_mat(prim, None, incoming_world_xform)
        # Centerline and rest points use the full affine (below) so reflections / shears are exact.
        # The curve normal is a material-frame director, not a surface normal: it co-deforms with
        # the segment tangent, so it transforms by the full linear block M like the points, not by
        # the covector rule M^-T. Recover the linear block from basis images (its columns).
        _o = wp.transform_point(world_mat, wp.vec3(0.0, 0.0, 0.0))
        _cx = wp.transform_point(world_mat, wp.vec3(1.0, 0.0, 0.0)) - _o
        _cy = wp.transform_point(world_mat, wp.vec3(0.0, 1.0, 0.0)) - _o
        _cz = wp.transform_point(world_mat, wp.vec3(0.0, 0.0, 1.0)) - _o
        normal_linear = wp.mat33(_cx[0], _cy[0], _cz[0], _cx[1], _cy[1], _cz[1], _cx[2], _cy[2], _cz[2])

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

        # The current proposal authors structural stiffnesses (EA, kGA, EI, GJ); missing modes
        # derive from E, nu, and the physical curve thickness. Newton damping stays independently
        # optional per mode. Each resolved gain is divided by the joint-local dual rest length
        # after rod construction.
        cable_mat = usd._get_curve_deformable_material(prim, deformable_read)
        _warn_legacy_curve_material(path, cable_mat)
        fallback_radius = _cable_radius_from_material(cable_mat, linear_unit)
        vertex_counts_int = [int(count) for count in vertex_counts]
        authored_segment_count = sum(count if closed else max(0, count - 1) for count in vertex_counts_int)
        authored_thicknesses = _read_deformable_element_array(
            prim,
            "thicknesses",
            {
                "constant": 1,
                "curve": len(vertex_counts_int),
                "segment": authored_segment_count,
                "point": len(points),
            },
            deformable_read,
        )
        segment_thicknesses, point_thicknesses = _curve_thickness_samples(
            authored_thicknesses, vertex_counts_int, closed, 2.0 * fallback_radius
        )
        segment_radii = [0.5 * thickness for thickness in segment_thicknesses]
        point_radii = [0.5 * thickness for thickness in point_thicknesses]
        # Density precedence resolved here; total-mass/per-point overrides applied after add_rod.
        cable_density = _resolve_deformable_density(
            prim,
            None if cable_mat is None else cable_mat.get("density"),
            deformable_read,
            linear_unit,
            read_base_material=cable_mat is None,
        )
        resolved_cable_density = cable_density
        collision_enabled, approximated_from = _deformable_collision_enabled(prim, ctx.ignore_paths)
        _warn_collision_approximated(path, approximated_from)
        cable_cfg = replace(
            builder.default_shape_cfg,
            density=resolved_cable_density,
            has_shape_collision=collision_enabled,
            has_particle_collision=collision_enabled,
        )

        cable_bodies: list[int] = []
        cable_joints: list[int] = []
        # vertex index -> [(segment body, body-local point)]
        cable_point_anchors: dict[int, list[tuple[int, wp.vec3]]] = {}
        # flat segment index -> (segment body, segment length)
        cable_segments: dict[int, tuple[int, float]] = {}
        cable_mass_runs: list[_CableMassRun] = []
        has_valid_rest_shape = False
        offset = 0
        flat_segment_index = 0
        for ci, vertex_count in enumerate(vertex_counts):
            n = int(vertex_count)
            start = offset
            local_pts = points[start : start + n]
            offset += n
            curve_segment_count = n if closed else max(0, n - 1)
            # add_rod needs >= 2 segments: >= 3 centerline points for an open curve, while a
            # periodic 2-point curve closes into 2 segments and is representable.
            min_points = 2 if closed else 3
            if n < min_points:
                warnings.warn(
                    f"{path}: curve {ci} has {n} points (need >= {min_points}); skipping that curve.",
                    stacklevel=2,
                )
                flat_segment_index += curve_segment_count
                continue
            positions = _bake_world_points(local_pts, world_mat)
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
            # Authored normals set each segment's cross-section twist; map them to world via
            # the full linear block computed above. A singular block can only degenerate a
            # normal to (near-)zero, which ``_cable_segment_quaternions`` already falls back
            # from (roll-free frame), so no special-casing here.
            quaternions = None
            if normals is not None:
                seg_normals = [
                    wp.mul(normal_linear, wp.vec3(float(nv[0]), float(nv[1]), float(nv[2])))
                    for nv in normals[start : start + n]
                ]
                quaternions = _cable_segment_quaternions(positions, seg_normals)
            # Use the rest centerline when authored (else the imported points), so each joint's dual
            # length follows the authored rest geometry.
            material_seg_lengths = seg_lengths
            rest_seg_lengths = None
            if rest_shape_points is not None:
                rest_seg_lengths = _cable_rest_segment_lengths(
                    rest_shape_points[start : start + n], world_mat, path, closed=closed
                )
                if rest_seg_lengths is not None:
                    material_seg_lengths = rest_seg_lengths
            label = path if len(vertex_counts) == 1 else f"{path}_curve{ci}"
            curve_radii = segment_radii[flat_segment_index : flat_segment_index + num_seg]
            # Wrap each cable into its own articulation so the model is finalize-ready (add_rod keeps
            # a periodic cable's loop-closing joint out of the tree). Attachment joints to other
            # bodies are loop-closing and stay outside the articulation regardless.
            bodies, joints = builder.add_rod(
                positions=positions,
                quaternions=quaternions,
                radius=curve_radii[0],
                cfg=cable_cfg,
                closed=closed,
                label=label,
                wrap_in_articulation=True,
                body_frame_origin="com",
            )
            for body, segment_radius in zip(bodies, curve_radii, strict=True):
                _set_cable_body_radius(builder, body, segment_radius)
            curve_point_radii = point_radii[start : start + n]
            curve_joint_radii = [*curve_point_radii[1:], curve_point_radii[0]] if closed else curve_point_radii[1:-1]
            _apply_local_rod_material_gains(
                builder,
                bodies,
                joints,
                material_seg_lengths,
                cable_mat,
                curve_radii,
                curve_joint_radii,
                linear_unit,
            )
            has_valid_rest_shape |= rest_seg_lengths is not None
            cable_bodies.extend(bodies)
            cable_joints.extend(joints)
            cable_mass_runs.append(
                _CableMassRun(
                    curve_index=ci,
                    point_offset=start,
                    point_count=n,
                    segment_offset=flat_segment_index,
                    body_ids=tuple(bodies),
                    element_volumes=tuple(
                        math.pi * segment_radius**2 * segment_length
                        for segment_radius, segment_length in zip(curve_radii, seg_lengths, strict=True)
                    ),
                    closed=closed,
                )
            )

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
            if has_valid_rest_shape:
                _warn_cable_rest_shape_effect(path)
            authored_masses = _apply_cable_masses(
                builder,
                prim,
                cable_mass_runs,
                deformable_read,
                len(points),
                len(vertex_counts_int),
                authored_segment_count,
                resolved_cable_density,
            )
            path_cable_map[path] = (cable_bodies, cable_joints)
            # Bodies/joints for a cable prim are built back-to-back, so the index lists are contiguous.
            body_range = (cable_bodies[0], cable_bodies[-1] + 1)
            joint_range = (
                (cable_joints[0], cable_joints[-1] + 1) if cable_joints else (builder.joint_count, builder.joint_count)
            )
            builder._record_cable_group(path, body_range, joint_range)
            path_cable_point_anchors[path] = cable_point_anchors
            path_cable_segments[path] = cable_segments
            path_cable_attrs[path] = {
                "material": dict(cable_mat or {}),
                "resolved_density": resolved_cable_density,
                "closed": closed,
            }
            if authored_thicknesses is not None or authored_masses is not None:
                path_cable_attrs[path]["simulation"] = {}
                if authored_thicknesses is not None:
                    path_cable_attrs[path]["simulation"]["thicknesses"] = {
                        "values": list(authored_thicknesses.values),
                        "element_type": authored_thicknesses.element_type,
                    }
                if authored_masses is not None:
                    path_cable_attrs[path]["simulation"]["masses"] = {
                        "values": list(authored_masses.values),
                        "element_type": authored_masses.element_type,
                        "legacy_implicit_type": authored_masses.legacy_implicit_type,
                    }
            if verbose:
                print(f"Added cable {path} with {len(cable_bodies)} segments.")
