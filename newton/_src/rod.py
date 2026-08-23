# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private implementations for Newton's discrete rod representation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple, overload

import warp as wp

from .math import quat_between_vectors_robust

__all__ = [
    "RodStiffness",
    "compute_parallel_transport_quaternions",
    "generate_straight_points",
    "generate_straight_points_and_quaternions",
    "stiffness_from_elastic_moduli",
]


class RodStiffness(NamedTuple):
    """Per-joint Kirchhoff rod stiffness for a circular isotropic cross-section.

    Returned by :func:`~newton.rod.stiffness_from_elastic_moduli`.

    Fields:

    * ``stretch`` -- axial stiffness ``E * A / L`` [N/m]
    * ``bend``    -- bending stiffness ``E * I / L`` [N·m/rad]
    * ``twist``   -- torsional stiffness ``G * J / L`` [N·m/rad]

    For a circular cross-section the two bending axes are equivalent
    (``EI1 == EI2 == EI``); the single ``bend`` field is used for both axes
    when assembling the per-joint rod stiffness vector.

    No ``shear`` field, by design. Set a sufficiently large finite
    ``shear_stiffness`` separately to approximate an unshearable Kirchhoff rod.

    Being a :class:`typing.NamedTuple`, instances support both attribute
    access (``stiffness.bend``) and tuple unpacking
    (``stretch, bend, twist = stiffness``).
    """

    stretch: float
    bend: float
    twist: float


@overload
def stiffness_from_elastic_moduli(
    youngs_modulus: float,
    radius: float,
    segment_length: float,
    *,
    poissons_ratio: float,
) -> RodStiffness: ...


@overload
def stiffness_from_elastic_moduli(
    youngs_modulus: float,
    radius: float,
    segment_length: float,
    *,
    shear_modulus: float,
) -> RodStiffness: ...


def stiffness_from_elastic_moduli(
    youngs_modulus: float,
    radius: float,
    segment_length: float,
    *,
    poissons_ratio: float | None = None,
    shear_modulus: float | None = None,
) -> RodStiffness:
    """Compute per-joint stretch, bend, and twist stiffness from elastic moduli.

    For a circular cross-section, this computes the stiffness values expected
    by :meth:`newton.ModelBuilder.add_rod` and
    :meth:`newton.ModelBuilder.add_rod_graph`:

    * ``stretch = E * A / L`` [N/m]
    * ``bend = E * I / L`` [N·m/rad]
    * ``twist = G * J / L`` [N·m/rad]

    Here ``A = pi * r^2``, ``I = pi * r^4 / 4``,
    ``J = pi * r^4 / 2``, and ``L = segment_length``. For an isotropic
    material with Poisson's ratio ``nu``,
    ``G = E / (2 * (1 + nu))``.

    No transverse shear stiffness is returned. The rod builder defaults
    ``shear_stiffness`` to ``stretch_stiffness`` when omitted; pass an
    explicit value if that default is not desired.

    Args:
        youngs_modulus: Young's modulus ``E`` [Pa]. Must be finite and
            ``>= 0``.
        radius: Rod radius ``r`` [m]. Must be finite and ``> 0``.
        segment_length: Per-joint rest length ``L`` [m]. Must be finite and
            ``> 0``.
        poissons_ratio: Poisson's ratio ``nu`` used to compute ``G``.
            Must be finite and satisfy ``-1 < nu < 0.5``. Mutually exclusive
            with ``shear_modulus``.
        shear_modulus: Shear modulus ``G`` [Pa]. Must be finite and ``>= 0``.
            Mutually exclusive with ``poissons_ratio``.

    Returns:
        Stretch, bend, and twist stiffness.

    Raises:
        ValueError: If an input is non-finite or out of range, or if exactly
            one of ``poissons_ratio`` and ``shear_modulus`` is not supplied.
    """
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
    stretch = E * area / L
    bend = E * inertia / L

    if poissons_ratio is None:
        if shear_modulus is None:
            raise ValueError("exactly one of poissons_ratio and shear_modulus must be supplied")
        shear = float(shear_modulus)
        if not math.isfinite(shear):
            raise ValueError("shear_modulus must be finite")
        if shear < 0.0:
            raise ValueError("shear_modulus must be >= 0")
    else:
        if shear_modulus is not None:
            raise ValueError("exactly one of poissons_ratio and shear_modulus must be supplied")
        nu = float(poissons_ratio)
        if not math.isfinite(nu):
            raise ValueError("poissons_ratio must be finite")
        if nu <= -1.0 or nu >= 0.5:
            raise ValueError("poissons_ratio must satisfy -1 < nu < 0.5")
        shear = E / (2.0 * (1.0 + nu))

    polar_inertia = 0.5 * math.pi * r**4
    return RodStiffness(stretch=stretch, bend=bend, twist=shear * polar_inertia / L)


def generate_straight_points(
    start: wp.vec3,
    direction: wp.vec3,
    length: float,
    num_segments: int,
) -> list[wp.vec3]:
    """Generate centerline points for a straight rod discretization.

    The returned points form the ``positions`` input for
    :meth:`newton.ModelBuilder.add_rod`.

    Args:
        start: First point in world space [m].
        direction: World-space direction of the rod (need not be normalized).
        length: Total length of the rod [m].
        num_segments: Number of rod segments. The number of points is
            ``num_segments + 1``.

    Returns:
        List of ``wp.vec3`` points of length ``num_segments + 1`` [m].

    Raises:
        ValueError: If the segment count, length, or direction is invalid.
    """
    if num_segments < 1:
        raise ValueError("num_segments must be >= 1")
    length_m = float(length)
    if not math.isfinite(length_m):
        raise ValueError("length must be finite")
    if length_m < 0.0:
        raise ValueError("length must be >= 0")

    dir_len = float(wp.length(direction))
    if dir_len <= 0.0:
        raise ValueError("direction must be non-zero")
    d = direction / dir_len

    ds = length_m / num_segments
    return [start + d * (ds * i) for i in range(num_segments + 1)]


def compute_parallel_transport_quaternions(
    points: Sequence[wp.vec3],
    *,
    twist_total: float = 0.0,
) -> list[wp.quat]:
    """Compute rod segment frames using parallel transport.

    The returned quaternions form the ``quaternions`` input for
    :meth:`newton.ModelBuilder.add_rod`.

    They rotate local +Z to each segment direction while minimizing twist
    between successive segments. Optionally, a total twist can be distributed
    uniformly along the rod.

    Args:
        points: Rod centerline points of length at least two [m].
        twist_total: Total twist [rad] distributed along the rod and applied
            about the segment direction.

    Returns:
        List of ``wp.quat`` of length ``len(points) - 1``.

    Raises:
        ValueError: If there are fewer than two points or consecutive points
            are duplicated.
    """
    if len(points) < 2:
        raise ValueError("points must have length >= 2")

    from_direction = wp.vec3(0.0, 0.0, 1.0)

    num_segments = len(points) - 1
    twist_total_rad = float(twist_total)
    twist_step = (twist_total_rad / num_segments) if twist_total_rad != 0.0 else 0.0
    eps = 1.0e-8

    quats: list[wp.quat] = []
    for i in range(num_segments):
        p0 = points[i]
        p1 = points[i + 1]
        seg = p1 - p0
        seg_len = float(wp.length(seg))
        if seg_len <= 0.0:
            raise ValueError("points must not contain duplicate consecutive points")
        to_direction = seg / seg_len

        # Robustly handle the anti-parallel (180-degree) case, e.g. +Z -> -Z.
        dq_dir = quat_between_vectors_robust(from_direction, to_direction, eps)

        q = dq_dir if i == 0 else wp.mul(dq_dir, quats[i - 1])

        if twist_total_rad != 0.0:
            twist_q = wp.quat_from_axis_angle(to_direction, twist_step)
            q = wp.mul(twist_q, q)

        quats.append(q)
        from_direction = to_direction

    return quats


def generate_straight_points_and_quaternions(
    start: wp.vec3,
    direction: wp.vec3,
    length: float,
    num_segments: int,
    *,
    twist_total: float = 0.0,
) -> tuple[list[wp.vec3], list[wp.quat]]:
    """Generate straight rod points and matching segment frames.

    The returned values form the ``positions`` and ``quaternions`` inputs for
    :meth:`newton.ModelBuilder.add_rod`.

    This combines :func:`newton.rod.generate_straight_points` with
    :func:`newton.rod.compute_parallel_transport_quaternions`.

    Args:
        start: First point in world space [m].
        direction: World-space direction of the rod (need not be normalized).
        length: Total length of the rod [m].
        num_segments: Number of segments. The returned point count is
            ``num_segments + 1``.
        twist_total: Total twist distributed along the rod [rad].

    Returns:
        Pair containing the world-space polyline points [m] and matching segment
        orientations as unit quaternions.

    Raises:
        ValueError: If centerline generation or frame computation receives an
            invalid input.
    """
    points = generate_straight_points(
        start=start,
        direction=direction,
        length=length,
        num_segments=num_segments,
    )
    quats = compute_parallel_transport_quaternions(points, twist_total=twist_total)
    return points, quats
