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
from collections.abc import Iterable, Sequence
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
