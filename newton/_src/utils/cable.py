# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compatibility helpers for Newton's released cable utility API."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import overload

import warp as wp

from ..rod import (
    RodStiffness,
    compute_parallel_transport_quaternions,
    generate_straight_points,
    generate_straight_points_and_quaternions,
    stiffness_from_elastic_moduli,
)

# Retain the private lookup used by values pickled before the type was renamed.
CableStiffness = RodStiffness


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
) -> RodStiffness: ...


@overload
def create_cable_stiffness_from_elastic_moduli(
    youngs_modulus: float,
    radius: float,
    segment_length: float,
    *,
    shear_modulus: float,
) -> RodStiffness: ...


def create_cable_stiffness_from_elastic_moduli(
    youngs_modulus: float,
    radius: float,
    segment_length: float,
    *,
    poissons_ratio: float | None = None,
    shear_modulus: float | None = None,
) -> tuple[float, float] | RodStiffness:
    """Compute cable stiffness using the released return contract.

    .. deprecated:: 1.6
        Use :func:`newton.rod.stiffness_from_elastic_moduli`. The replacement
        requires exactly one of ``poissons_ratio`` and ``shear_modulus`` and
        always returns :class:`newton.rod.RodStiffness`.

    Returns:
        The released ``(stretch, bend)`` pair when neither torsional material
        input is supplied; otherwise, ``RodStiffness(stretch, bend, twist)``.
    """
    warnings.warn(
        "newton.utils.create_cable_stiffness_from_elastic_moduli() is deprecated in Newton 1.6; "
        "use newton.rod.stiffness_from_elastic_moduli() with poissons_ratio= or shear_modulus= instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if poissons_ratio is None:
        if shear_modulus is not None:
            return stiffness_from_elastic_moduli(
                youngs_modulus,
                radius,
                segment_length,
                shear_modulus=shear_modulus,
            )

        E = float(youngs_modulus)
        r = float(radius)
        L = float(segment_length)

        if not math.isfinite(E):
            raise ValueError("youngs_modulus must be finite")
        if not math.isfinite(r):
            raise ValueError("radius must be finite")
        if not math.isfinite(L):
            raise ValueError("segment_length must be finite")
        if E < 0.0:
            raise ValueError("youngs_modulus must be >= 0")
        if r <= 0.0:
            raise ValueError("radius must be > 0")
        if L <= 0.0:
            raise ValueError("segment_length must be > 0")

        area = math.pi * r * r
        inertia = 0.25 * math.pi * r**4
        return E * area / L, E * inertia / L

    if shear_modulus is not None:
        raise ValueError("poissons_ratio and shear_modulus are mutually exclusive")

    return stiffness_from_elastic_moduli(
        youngs_modulus,
        radius,
        segment_length,
        poissons_ratio=poissons_ratio,
    )


def create_parallel_transport_cable_quaternions(
    points: Sequence[wp.vec3],
    *,
    twist_total: float = 0.0,
) -> list[wp.quat]:
    """Compute cable frames using the released helper name.

    .. deprecated:: 1.6
        Use :func:`newton.rod.compute_parallel_transport_quaternions`.
    """
    warnings.warn(
        "newton.utils.create_parallel_transport_cable_quaternions() is deprecated in Newton 1.6; "
        "use newton.rod.compute_parallel_transport_quaternions() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_parallel_transport_quaternions(points, twist_total=twist_total)


def create_straight_cable_points(
    start: wp.vec3,
    direction: wp.vec3,
    length: float,
    num_segments: int,
) -> list[wp.vec3]:
    """Generate straight cable points using the released helper name.

    .. deprecated:: 1.6
        Use :func:`newton.rod.generate_straight_points`.
    """
    warnings.warn(
        "newton.utils.create_straight_cable_points() is deprecated in Newton 1.6; "
        "use newton.rod.generate_straight_points() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return generate_straight_points(start, direction, length, num_segments)


def create_straight_cable_points_and_quaternions(
    start: wp.vec3,
    direction: wp.vec3,
    length: float,
    num_segments: int,
    *,
    twist_total: float = 0.0,
) -> tuple[list[wp.vec3], list[wp.quat]]:
    """Generate rod points and frames using the released helper name.

    .. deprecated:: 1.6
        Use :func:`newton.rod.generate_straight_points_and_quaternions`.
    """
    warnings.warn(
        "newton.utils.create_straight_cable_points_and_quaternions() is deprecated in Newton 1.6; "
        "use newton.rod.generate_straight_points_and_quaternions() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return generate_straight_points_and_quaternions(
        start,
        direction,
        length,
        num_segments,
        twist_total=twist_total,
    )
