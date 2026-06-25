# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Self-contained helpers for the USD deformable importer.

These are pure helpers (no dependence on the :func:`parse_usd` builder state) shared
by the cable / cloth / volume and attachment passes in :mod:`.import_usd`. Builder
mutation and traversal orchestration stay in :mod:`.import_usd`.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from pxr import Usd


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


def is_ignored_path(path: str, ignore_paths: Sequence[str]) -> bool:
    """Return whether ``path`` matches any of the ``ignore_paths`` regular expressions."""
    return any(re.match(pattern, path) for pattern in ignore_paths)


def validate_attachment_index_pairs(
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
class CurveDeformableRecord:
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


def import_element_collision_filters(
    builder,
    root_prim,
    ignore_paths: Sequence[str],
    deformable_read: Callable,
    get_first_target: Callable,
    verbose: bool,
    path_cable_segments: dict,
    path_body_map: dict,
    path_cloth_map: dict,
    path_soft_map: dict,
) -> None:
    """Lower AOUSD ``PhysicsElementCollisionFilter`` prims to shape collision filter pairs.

    Each prim suppresses collision between selected *elements* of ``src0`` and ``src1``. Supported
    element sources are imported cables (``groupElemIndices`` index the cable's segments) and rigid
    colliders (all of the collider's shapes). An empty index array means *all* elements of that
    source. Cloth/volume (triangle/tet) element sources have no per-element rigid shape in Newton's
    shape-filter model and are warned and skipped.
    """
    from pxr import Usd

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
        if is_ignored_path(path, ignore_paths):
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
