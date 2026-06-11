# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple, overload

import warp as wp

from ..math import quat_between_vectors_robust


class CableStiffness(NamedTuple):
    """Per-joint Kirchhoff rod stiffness for a circular isotropic cross-section.

    Returned by :func:`create_cable_stiffness_from_elastic_moduli` when the
    caller supplies either ``poissons_ratio`` or ``shear_modulus``.

    Fields:

    * ``stretch`` -- axial stiffness ``E * A / L`` [N/m]
    * ``bend``    -- bending stiffness ``E * I / L`` [N*m / rad]
    * ``twist``   -- torsional stiffness ``G * J / L`` [N*m / rad]

    For a circular cross-section the two bending axes are equivalent
    (``EI1 == EI2 == EI``); the single ``bend`` field is used for both axes
    when assembling the per-joint cable stiffness vector.

    No ``shear`` field, by design: this helper is not a full shearable-beam
    material helper, so the builder defaults ``shear_stiffness`` to
    ``stretch`` as an isotropic positional hold (a constraint stiffness, not a
    transverse-shear material modulus).

    Being a :class:`typing.NamedTuple`, instances support both attribute
    access (``stiffness.bend``) and tuple unpacking
    (``stretch, bend, twist = stiffness``).
    """

    stretch: float
    bend: float
    twist: float


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
    youngs_modulus,
    radius,
    segment_length,
    *,
    poissons_ratio: float | None = None,
    shear_modulus: float | None = None,
):
    """Create per-joint rod/cable stiffness from elastic moduli.

    For a circular cross-section, this computes material stiffnesses and
    converts them to the per-joint stiffness values expected by
    :meth:`ModelBuilder.add_rod` and :meth:`ModelBuilder.add_rod_graph`:

    * ``stretch = E * A / L``     [N/m]
    * ``bend    = E * I / L``     [N*m / rad]
    * ``twist   = G * J / L``     [N*m / rad]    (returned only when
      ``poissons_ratio`` or ``shear_modulus`` is supplied)

    where ``A = pi * r^2``, ``I = pi * r^4 / 4`` (area moment of inertia about
    a diameter), ``J = pi * r^4 / 2`` (polar moment of area), and
    ``L = segment_length``. For an isotropic material with Poisson's ratio
    ``nu``, the shear modulus is ``G = E / (2 * (1 + nu))``.

    No separate transverse shear stiffness is returned. The split cable API
    defaults ``shear_stiffness`` to ``stretch_stiffness`` when omitted; pass an
    explicit ``shear_stiffness`` if that default is not desired. The
    ``shear_modulus`` keyword supplies ``G`` for torsion/twist only.

    The return shape mirrors what the caller asks for:

    * ``create_cable_stiffness_from_elastic_moduli(E, r, L)`` returns the
      plain 2-tuple ``(stretch, bend)`` -- twist is not derivable from
      ``E`` alone, so it is omitted. Suitable for stretch-only or
      bend-only rods, or when the caller manages ``twist_stiffness``
      separately.
    * Supplying ``poissons_ratio`` or ``shear_modulus`` switches the
      return to :class:`CableStiffness` with the additional ``twist``
      term. The result both unpacks as a 3-tuple and exposes named
      fields ``.stretch``, ``.bend``, ``.twist``.

    The 2-tuple has no twist (torsion needs ``G``, not derivable from ``E``).
    Passed through the builder without an explicit ``twist_stiffness``, twist
    defaults to ``bend`` (the previous combined-stiffness model). For
    material-consistent torsion ``twist / bend = G * J / (E * I) = 1 / (1 + nu)``,
    pass ``poissons_ratio`` or ``shear_modulus``.

    Args:
        youngs_modulus: Young's modulus ``E`` [Pa = N/m^2]. Must be finite
            and ``>= 0``.
        radius: Rod/cable radius ``r`` [m]. Must be finite and ``> 0``.
        segment_length: Per-joint rest length ``L`` [m]. Must be finite and
            ``> 0``.
        poissons_ratio: Poisson's ratio ``nu`` used to compute the shear
            modulus ``G = E / (2 * (1 + nu))``. Keyword-only. Must satisfy
            ``-1 < nu < 0.5`` for a stable isotropic 3D material. Mutually
            exclusive with ``shear_modulus``.
        shear_modulus: Shear modulus ``G`` [Pa]. Keyword-only. Mutually
            exclusive with ``poissons_ratio``.

    Returns:
        2-tuple ``(stretch, bend)`` when neither ``poissons_ratio`` nor
        ``shear_modulus`` is supplied; otherwise a :class:`CableStiffness`
        NamedTuple ``(stretch, bend, twist)``.

    Raises:
        ValueError: if any of ``youngs_modulus``, ``radius``,
            ``segment_length``, ``poissons_ratio``, or ``shear_modulus`` is
            non-finite or out of range, or if both ``poissons_ratio`` and
            ``shear_modulus`` are supplied.
    """
    # Accept ints / numpy scalars, but return plain Python floats.
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
    if poissons_ratio is not None and shear_modulus is not None:
        raise ValueError("poissons_ratio and shear_modulus are mutually exclusive")

    area = math.pi * r * r
    inertia = 0.25 * math.pi * r**4
    stretch_stiffness = E * area / L
    bend_stiffness = E * inertia / L

    if poissons_ratio is None and shear_modulus is None:
        return stretch_stiffness, bend_stiffness

    if shear_modulus is None:
        nu = float(poissons_ratio)
        if not math.isfinite(nu):
            raise ValueError("poissons_ratio must be finite")
        if nu <= -1.0 or nu >= 0.5:
            raise ValueError("poissons_ratio must satisfy -1 < nu < 0.5")
        G = E / (2.0 * (1.0 + nu))
    else:
        G = float(shear_modulus)
        if not math.isfinite(G):
            raise ValueError("shear_modulus must be finite")
        if G < 0.0:
            raise ValueError("shear_modulus must be >= 0")

    polar_inertia = 0.5 * math.pi * r**4
    return CableStiffness(
        stretch=stretch_stiffness,
        bend=bend_stiffness,
        twist=G * polar_inertia / L,
    )


def create_straight_cable_points(
    start: wp.vec3,
    direction: wp.vec3,
    length: float,
    num_segments: int,
) -> list[wp.vec3]:
    """Create straight cable polyline points.

    This is a convenience helper for constructing ``positions`` inputs for ``ModelBuilder.add_rod``.

    Args:
        start: First point in world space.
        direction: World-space direction of the cable (need not be normalized).
        length: Total length of the cable (meters).
        num_segments: Number of segments (edges). The number of points is ``num_segments + 1``.

    Returns:
        List of ``wp.vec3`` points of length ``num_segments + 1``.
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


def create_parallel_transport_cable_quaternions(
    points: Sequence[wp.vec3],
    *,
    twist_total: float = 0.0,
) -> list[wp.quat]:
    """Generate per-segment quaternions using a parallel-transport style construction.

    The intended use is for rod/cable capsules whose internal axis is local +Z.
    The returned quaternions rotate local +Z to each segment direction,
    while minimizing twist between successive segments. Optionally, a total twist can be
    distributed uniformly along the cable.

    Args:
        points: Polyline points of length >= 2.
        twist_total: Total twist (radians) distributed along the cable (applied about the segment direction).

    Returns:
        List of ``wp.quat`` of length ``len(points) - 1``.
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


def create_straight_cable_points_and_quaternions(
    start: wp.vec3,
    direction: wp.vec3,
    length: float,
    num_segments: int,
    *,
    twist_total: float = 0.0,
) -> tuple[list[wp.vec3], list[wp.quat]]:
    """Generate straight cable points and matching per-segment quaternions.

    This is a convenience wrapper around:
    - :func:`create_straight_cable_points`
    - :func:`create_parallel_transport_cable_quaternions`
    """
    points = create_straight_cable_points(
        start=start,
        direction=direction,
        length=length,
        num_segments=num_segments,
    )
    quats = create_parallel_transport_cable_quaternions(points, twist_total=twist_total)
    return points, quats
