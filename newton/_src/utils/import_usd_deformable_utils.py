# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""USD deformable importer shared leaf helpers and import context.

This module owns the builder-independent leaf helpers (e.g. :func:`_validate_mass_array`) and the
shared mass / density / anchor utilities used by the cable / cloth / volume / attachment /
collision-filter import passes, plus the :class:`_DeformableImportContext` that carries the
:func:`parse_usd` inputs, helper closures, and result maps the passes mutate. The passes
themselves live in the sibling ``import_usd_deformable_{cable,cloth,volume,attachments}`` modules.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from pxr import Usd

    from ..sim.builder import ModelBuilder


def _validate_mass_array(values: Iterable[float], path: str) -> list[float] | None:
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


def _skip_for_deformable_body_owner(ctx, prim, path: str, warn: bool = True) -> bool:
    """True when another simulation geometry already owns this prim's deformable body.

    A ``PhysicsDeformableBodyAPI`` body governs exactly one simulation geometry across all
    families (else its authored mass would be applied once per family). The owner is the
    first candidate in stage traversal order, resolved by the scout.
    """
    from ..usd import utils as usd  # noqa: PLC0415

    body_root = usd._find_deformable_body_prim(prim)
    if body_root is None:
        return False
    owner = ctx.prims.body_owner.get(str(body_root.GetPath()))
    if owner is None or owner == path:
        return False
    if warn:
        warnings.warn(
            f"{path}: deformable body {body_root.GetPath()} already has simulation geometry "
            f"{owner}; skipping additional simulation geometry.",
            stacklevel=2,
        )
    return True


def _is_ignored_path(path: str, ignore_paths: Sequence[str]) -> bool:
    """Return whether ``path`` matches any of the ``ignore_paths`` regular expressions."""
    return any(re.match(pattern, path) for pattern in ignore_paths)


def _world_matrix_reflects(world_mat: wp.mat44) -> bool:
    """Whether the world transform's linear part has a negative determinant (a reflection).

    A reflective (odd-negative-scale) transform flips triangle/tet winding and is not recoverable
    from :func:`warp.transform_decompose` (which always returns a positive scale), so deformable
    points are placed with the full affine and winding is flipped when this is ``True``. The
    determinant sign is transpose-invariant, so the matrix storage convention does not matter here.
    """
    linear = np.array(world_mat, dtype=np.float64).reshape(4, 4)[:3, :3]
    return bool(np.linalg.det(linear) < 0.0)


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


def _warn_dropped_velocities(prim: Usd.Prim, path: str) -> None:
    """Warn if the geometry authors velocities; deformable dynamic state is not imported yet, so the
    body starts at rest rather than being silently reset."""
    from pxr import UsdGeom

    vel = UsdGeom.PointBased(prim).GetVelocitiesAttr()
    if vel and vel.HasAuthoredValue():
        warnings.warn(
            f"{path}: authored velocities are not imported; the deformable starts at rest.",
            stacklevel=2,
        )


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


def _deformable_body_skip_reason(prim: Usd.Prim, read_attr: Callable) -> str | None:
    """Return why a deformable simulation prim must not import as a dynamic object, or None.

    ``physics:bodyEnabled = false`` disables the body outright and
    ``physics:kinematicEnabled = true`` requests a kinematic body, which Newton's deformables
    cannot represent yet; importing either as a dynamic object would silently change the
    authored physical model, so the caller warns and skips the prim.
    ``startsAsleep`` / ``simulationOwner`` are deferred (see the importer limitations doc).
    The flags are read from the governing ``PhysicsDeformableBodyAPI`` prim when one exists,
    else from the simulation prim itself.
    """
    from ..usd import utils as usd  # noqa: PLC0415

    body_prim = usd._find_deformable_body_prim(prim) or prim
    enabled = read_attr(body_prim, "bodyEnabled")
    if enabled is not None and not bool(enabled):
        return "physics:bodyEnabled is false"
    kinematic = read_attr(body_prim, "kinematicEnabled")
    if kinematic is not None and bool(kinematic):
        return "physics:kinematicEnabled is true (kinematic deformables are not supported)"
    return None


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
    authored_point_count: int,
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

    point_masses = usd._get_deformable_point_masses(prim, read_attr)
    body_mass, _ = usd._get_deformable_body_overrides(prim, read_attr)
    # physics:masses is authored per point of the prim's full points array, so validate against
    # the authored count, not the imported one: each run indexes it by its absolute point offset,
    # and a skipped curve leaves its entries unused (a shorter array would be indexed out of range).
    if point_masses is not None and len(point_masses) != authored_point_count:
        warnings.warn(
            f"{prim.GetPath()}: physics:masses length {len(point_masses)} != {authored_point_count} "
            f"authored curve points; ignoring per-point masses.",
            stacklevel=2,
        )
        point_masses = None
    if point_masses is not None:
        lumped: list[tuple[Sequence[int], list[float]]] = []
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
            if len(bodies) != len(seg_masses):
                # A welded graph can collapse a segment, so the surviving body count no longer matches
                # the per-point lumping. Ignore per-point masses (fall back to a body-mass total or the
                # density-derived masses) instead of raising and aborting the whole import.
                warnings.warn(
                    f"{prim.GetPath()}: welded cable collapsed a segment ({len(bodies)} bodies for "
                    f"{len(seg_masses)} point-derived segment masses); ignoring per-point physics:masses.",
                    stacklevel=2,
                )
                point_masses = None
                break
            lumped.append((bodies, seg_masses))
        else:
            for bodies, seg_masses in lumped:
                for b, m in zip(bodies, seg_masses, strict=True):
                    _set_body_mass(builder, b, m)
            return
    # Density-derived masses: add_rod gives each segment a CAPSULE mass (cylinder + two hemispherical
    # caps), but the proposal models a curve element as the cylindrical centerline segment. Rescale
    # each to the cylinder mass = capsule_mass / (1 + 4r/3L) (purely geometric, from the capsule's own
    # radius r and length L = 2*half_height), dropping the cap bias -- large for short, thick segments
    # -- so per-segment masses follow segment length.
    for b in body_ids:
        shapes = builder.body_shapes[b]
        if not shapes:
            continue
        r = float(builder.shape_scale[shapes[0]][0])
        seg_len = 2.0 * float(builder.shape_scale[shapes[0]][1])
        if r > 0.0 and seg_len > 0.0:
            _set_body_mass(builder, b, builder.body_mass[b] / (1.0 + 4.0 * r / (3.0 * seg_len)))
    if body_mass is None:
        return
    # A body-mass total has no per-point profile; rescale the cylinder masses to that total.
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


@dataclass(slots=True)
class _DeformablePrimBuckets:
    """Deformable candidate prims discovered by :func:`_scout_deformable_prims`.

    Each list keeps stage traversal order, so iterating a bucket visits prims in the same order
    the per-family full-stage walks used to. The buckets classify by coarse type only;
    ``ignore_paths`` filtering and per-prim validation stay in the lowering passes so their
    warnings and skip behavior are unchanged.
    """

    cables: list[Usd.Prim] = field(default_factory=list)
    cloth: list[Usd.Prim] = field(default_factory=list)
    tetmeshes: list[Usd.Prim] = field(default_factory=list)
    attachments: list[Usd.Prim] = field(default_factory=list)
    element_filters: list[Usd.Prim] = field(default_factory=list)
    # PhysicsDeformableBodyAPI prim path -> the single simulation geometry it governs (the
    # first candidate of any family in traversal order); a body's mass must not be applied
    # once per family, so the passes skip every other candidate under the same body.
    body_owner: dict[str, str] = field(default_factory=dict)

    def has_candidates(self) -> bool:
        """Whether any deformable lowering pass has candidate prims.

        All five buckets count: bare TetMeshes still take the legacy soft-body path, and
        standalone attachments / element filters must run their passes even when no supported
        deformable was imported (to record their attrs and warn).
        """
        return bool(self.cables or self.cloth or self.tetmeshes or self.attachments or self.element_filters)


def _scout_deformable_prims(root_prim: Usd.Prim) -> _DeformablePrimBuckets:
    """Classify deformable candidate prims in one stage traversal.

    Replaces the per-family full-stage walks: the lowering passes iterate these buckets instead of
    re-traversing the stage, so a stage without deformables pays a single scouting walk. Buckets
    match each pass's coarse type filter: cables/cloth require their applied sim API, but every
    ``TetMesh`` is bucketed because bare TetMeshes still import as legacy soft bodies.
    """
    from pxr import Usd, UsdGeom

    from ..usd import utils as usd  # noqa: PLC0415

    buckets = _DeformablePrimBuckets()
    if not (root_prim and root_prim.IsValid()):
        return buckets

    def claim_body(prim: Usd.Prim) -> None:
        body_root = usd._find_deformable_body_prim(prim)
        if body_root is not None:
            buckets.body_owner.setdefault(str(body_root.GetPath()), str(prim.GetPath()))

    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        type_name = str(prim.GetTypeName())
        if type_name == "PhysicsAttachment":
            buckets.attachments.append(prim)
        elif type_name == "PhysicsElementCollisionFilter":
            buckets.element_filters.append(prim)
        elif prim.IsA(UsdGeom.TetMesh):
            buckets.tetmeshes.append(prim)
            if usd.has_applied_api_schema(prim, "PhysicsVolumeDeformableSimAPI"):
                claim_body(prim)
        elif prim.IsA(UsdGeom.BasisCurves):
            if usd.has_applied_api_schema(prim, "PhysicsCurvesDeformableSimAPI"):
                buckets.cables.append(prim)
                claim_body(prim)
        elif prim.IsA(UsdGeom.Mesh):
            if usd.has_applied_api_schema(prim, "PhysicsSurfaceDeformableSimAPI"):
                buckets.cloth.append(prim)
                claim_body(prim)
    return buckets


@dataclass(slots=True)
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
    incoming_world_xform: wp.transform
    linear_unit: float
    ignore_paths: Sequence[str]
    verbose: bool
    path_body_map: dict
    path_shape_map: dict
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
    # Filled by _scout_deformable_prims so the passes iterate buckets instead of the stage.
    prims: _DeformablePrimBuckets = field(default_factory=_DeformablePrimBuckets)
