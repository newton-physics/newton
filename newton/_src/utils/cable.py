# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deprecated cable helper compatibility for the public :class:`newton.Rod` API."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import NamedTuple, overload

import warp as wp

from ..sim.rod import (
    _compute_parallel_transport_quaternions,
    _generate_straight_points,
)


class CableStiffness(NamedTuple):
    """Deprecated per-joint cable stiffness tuple retained for compatibility.

    ``stretch`` is measured in N/m; ``bend`` and ``twist`` are measured in
    N·m/rad.

    .. deprecated:: 1.6
        Pass direct stiffness values to :meth:`newton.ModelBuilder.add_rod`.
    """

    stretch: float
    bend: float
    twist: float


def _legacy_straight_points(
    start: wp.vec3,
    direction: wp.vec3,
    length: float,
    num_segments: int,
) -> list[wp.vec3]:
    """Preserve the released point helper's zero-length behavior."""
    if num_segments < 1:
        raise ValueError("num_segments must be >= 1")
    length_m = float(length)
    if not math.isfinite(length_m):
        raise ValueError("length must be finite")
    if length_m < 0.0:
        raise ValueError("length must be >= 0")
    direction_length = float(wp.length(direction))
    if not math.isfinite(direction_length) or direction_length <= 0.0:
        raise ValueError("direction must be finite and non-zero")
    if length_m == 0.0:
        return [start for _ in range(num_segments + 1)]
    return _generate_straight_points(start, direction, length_m, num_segments)


@overload
def create_cable_stiffness_from_elastic_moduli(
    youngs_modulus: float,
    radius: float,
    segment_length: float,
) -> tuple[float, float]: ...


@overload
def create_cable_stiffness_from_elastic_moduli(
    youngs_modulus: float,
    radius: float,
    segment_length: float,
    *,
    poissons_ratio: float,
) -> CableStiffness: ...


@overload
def create_cable_stiffness_from_elastic_moduli(
    youngs_modulus: float,
    radius: float,
    segment_length: float,
    *,
    shear_modulus: float,
) -> CableStiffness: ...


def create_cable_stiffness_from_elastic_moduli(
    youngs_modulus: float,
    radius: float,
    segment_length: float,
    *,
    poissons_ratio: float | None = None,
    shear_modulus: float | None = None,
) -> tuple[float, float] | CableStiffness:
    """Compute per-joint cable stiffness from elastic moduli.

    .. deprecated:: 1.6
        Supply elastic material to :class:`newton.Rod`, or pass direct
        stiffness values to :meth:`newton.ModelBuilder.add_rod`.
    """
    warnings.warn(
        "newton.utils.create_cable_stiffness_from_elastic_moduli() is deprecated in Newton 1.6; "
        "supply material through newton.Rod(...) or direct stiffness to newton.ModelBuilder.add_rod() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Keep this deprecated API's validation order, messages, formulas, and
    # return shapes independent of the replacement Rod material contract.
    youngs = float(youngs_modulus)
    radius_m = float(radius)
    length_m = float(segment_length)
    if not math.isfinite(youngs):
        raise ValueError("youngs_modulus must be finite")
    if not math.isfinite(radius_m):
        raise ValueError("radius must be finite")
    if not math.isfinite(length_m):
        raise ValueError("segment_length must be finite")
    if youngs < 0.0:
        raise ValueError("youngs_modulus must be >= 0")
    if radius_m <= 0.0:
        raise ValueError("radius must be > 0")
    if length_m <= 0.0:
        raise ValueError("segment_length must be > 0")
    if poissons_ratio is not None and shear_modulus is not None:
        raise ValueError("poissons_ratio and shear_modulus are mutually exclusive")

    area = math.pi * radius_m * radius_m
    area_moment = 0.25 * math.pi * radius_m**4
    stretch = youngs * area / length_m
    bend = youngs * area_moment / length_m
    if poissons_ratio is None and shear_modulus is None:
        return stretch, bend

    if shear_modulus is None:
        poisson = float(poissons_ratio)
        if not math.isfinite(poisson):
            raise ValueError("poissons_ratio must be finite")
        if poisson <= -1.0 or poisson >= 0.5:
            raise ValueError("poissons_ratio must satisfy -1 < nu < 0.5")
        shear = youngs / (2.0 * (1.0 + poisson))
    else:
        shear = float(shear_modulus)
        if not math.isfinite(shear):
            raise ValueError("shear_modulus must be finite")
        if shear < 0.0:
            raise ValueError("shear_modulus must be >= 0")

    twist = shear * 0.5 * math.pi * radius_m**4 / length_m
    return CableStiffness(stretch=stretch, bend=bend, twist=twist)


def create_parallel_transport_cable_quaternions(
    points: Sequence[wp.vec3],
    *,
    twist_total: float = 0.0,
) -> list[wp.quat]:
    """Compute per-segment cable frames using parallel transport.

    .. deprecated:: 1.6
        Construct :class:`newton.Rod` with omitted ``quaternions`` and read
        :attr:`newton.Rod.quaternions`. For nonzero ``twist_total``, first call
        :meth:`newton.Rod.compute_frames` with the same value.
    """
    warnings.warn(
        "newton.utils.create_parallel_transport_cable_quaternions() is deprecated in Newton 1.6; "
        "use rod = newton.Rod(points) and read rod.quaternions instead; for nonzero twist_total, "
        "call rod.compute_frames(twist_total=twist_total) first.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _compute_parallel_transport_quaternions(points, twist_total=twist_total)


def create_straight_cable_points(
    start: wp.vec3,
    direction: wp.vec3,
    length: float,
    num_segments: int,
) -> list[wp.vec3]:
    """Generate uniformly spaced points along a straight cable.

    .. deprecated:: 1.6
        Use :meth:`newton.Rod.create_straight` and its :attr:`newton.Rod.points`.
    """
    warnings.warn(
        "newton.utils.create_straight_cable_points() is deprecated in Newton 1.6; "
        "use newton.Rod.create_straight().points instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _legacy_straight_points(start, direction, length, num_segments)


def create_straight_cable_points_and_quaternions(
    start: wp.vec3,
    direction: wp.vec3,
    length: float,
    num_segments: int,
    *,
    twist_total: float = 0.0,
) -> tuple[list[wp.vec3], list[wp.quat]]:
    """Generate straight cable points and matching segment frames.

    .. deprecated:: 1.6
        Use :meth:`newton.Rod.create_straight`.
    """
    warnings.warn(
        "newton.utils.create_straight_cable_points_and_quaternions() is deprecated in Newton 1.6; "
        "use newton.Rod.create_straight() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    points = _legacy_straight_points(start, direction, length, num_segments)
    quaternions = _compute_parallel_transport_quaternions(points, twist_total=twist_total)
    return points, quaternions
