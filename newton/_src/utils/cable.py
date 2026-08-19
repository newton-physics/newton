# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import NamedTuple, overload

import warp as wp

from ..math import quat_between_vectors_robust


# Rod-level stiffness result.
class RodStiffness(NamedTuple):
    """Per-joint Kirchhoff rod stiffness for a circular isotropic cross-section.

    Returned by :func:`rod_stiffness_from_elastic_moduli` when the caller
    supplies either ``poissons_ratio`` or ``shear_modulus``.

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


# Rod representation
# Deprecated public name retained throughout Newton 1.6.
CableStiffness = RodStiffness


@overload
def rod_stiffness_from_elastic_moduli(
    youngs_modulus: float,
    radius: float,
    segment_length: float,
) -> tuple[float, float]: ...


@overload
def rod_stiffness_from_elastic_moduli(
    youngs_modulus: float,
    radius: float,
    segment_length: float,
    *,
    poissons_ratio: float,
) -> RodStiffness: ...


@overload
def rod_stiffness_from_elastic_moduli(
    youngs_modulus: float,
    radius: float,
    segment_length: float,
    *,
    shear_modulus: float,
) -> RodStiffness: ...


def rod_stiffness_from_elastic_moduli(
    youngs_modulus: float,
    radius: float,
    segment_length: float,
    *,
    poissons_ratio: float | None = None,
    shear_modulus: float | None = None,
) -> tuple[float, float] | RodStiffness:
    """Create per-joint rod stiffness from elastic moduli.

    For a circular cross-section, this computes material stiffnesses and
    converts them to the per-joint stiffness values expected by
    :meth:`ModelBuilder.add_rod` and :meth:`ModelBuilder.add_rod_graph`:

    * ``stretch = E * A / L``     [N/m]
    * ``bend    = E * I / L``     [N·m/rad]
    * ``twist   = G * J / L``     [N·m/rad]    (returned only when
      ``poissons_ratio`` or ``shear_modulus`` is supplied)

    where ``A = pi * r^2``, ``I = pi * r^4 / 4`` (area moment of inertia about
    a diameter), ``J = pi * r^4 / 2`` (polar moment of area), and
    ``L = segment_length``. For an isotropic material with Poisson's ratio
    ``nu``, the shear modulus is ``G = E / (2 * (1 + nu))``.

    No separate transverse shear stiffness is returned. The split rod joint API
    defaults ``shear_stiffness`` to ``stretch_stiffness`` when omitted; pass an
    explicit ``shear_stiffness`` if that default is not desired. The
    ``shear_modulus`` keyword supplies ``G`` for torsion/twist only.

    The return shape mirrors what the caller asks for:

    * ``rod_stiffness_from_elastic_moduli(E, r, L)`` returns the plain
      2-tuple ``(stretch, bend)``. Twist is not derivable from ``E`` alone, so
      it is omitted. This is suitable for stretch-only or bend-only rods, or
      when the caller manages ``twist_stiffness`` separately.
    * Supplying ``poissons_ratio`` or ``shear_modulus`` switches the return to
      :class:`RodStiffness` with the additional ``twist`` term. The result
      both unpacks as a 3-tuple and exposes named fields ``.stretch``,
      ``.bend``, and ``.twist``.

    When the 2-tuple is passed through the builder without an explicit
    ``twist_stiffness``, twist defaults to ``bend`` (the combined-stiffness
    model). For material-consistent torsion,
    ``twist / bend = G * J / (E * I) = 1 / (1 + nu)``; pass
    ``poissons_ratio`` or ``shear_modulus`` to get the third term.

    Args:
        youngs_modulus: Young's modulus ``E`` [Pa = N/m²]. Must be finite
            and ``>= 0``.
        radius: Rod radius ``r`` [m]. Must be finite and ``> 0``.
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
        ``shear_modulus`` is supplied; otherwise a :class:`RodStiffness`
        NamedTuple ``(stretch, bend, twist)``.

    Raises:
        ValueError: If any of ``youngs_modulus``, ``radius``,
            ``segment_length``, ``poissons_ratio``, or ``shear_modulus`` is
            non-finite or out of range, or if both ``poissons_ratio`` and
            ``shear_modulus`` are supplied.
    """
    # Accept ints / NumPy scalars, but return plain Python floats.
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
    return RodStiffness(
        stretch=stretch_stiffness,
        bend=bend_stiffness,
        twist=G * polar_inertia / L,
    )


# Cable-level geometry and frames
def cable_straight_points(
    start: wp.vec3,
    direction: wp.vec3,
    length: float,
    num_segments: int,
) -> list[wp.vec3]:
    """Create straight cable centerline points.

    Args:
        start: First point in world space [m].
        direction: World-space direction of the cable (need not be normalized).
        length: Total length of the cable [m].
        num_segments: Number of segments (edges). The number of points is
            ``num_segments + 1``.

    Returns:
        List of ``wp.vec3`` points of length ``num_segments + 1`` [m].
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


def rod_parallel_transport_quaternions(
    points: Sequence[wp.vec3],
    *,
    twist_total: float = 0.0,
) -> list[wp.quat]:
    """Generate rod segment frames using parallel transport.

    The returned quaternions form the ``quaternions`` input for
    :meth:`ModelBuilder.add_rod`.

    They rotate local +Z to each segment direction while
    minimizing twist between successive segments. Optionally, a total twist
    can be distributed uniformly along the rod.

    Args:
        points: Rod centerline points of length at least two [m].
        twist_total: Total twist [rad] distributed along the rod and applied
            about the segment direction.

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


def rod_straight_points_and_quaternions(
    start: wp.vec3,
    direction: wp.vec3,
    length: float,
    num_segments: int,
    *,
    twist_total: float = 0.0,
) -> tuple[list[wp.vec3], list[wp.quat]]:
    """Generate straight rod points and matching segment frames.

    This is a convenience wrapper around :func:`cable_straight_points` and
    :func:`rod_parallel_transport_quaternions`. The returned values form the
    ``positions`` and ``quaternions`` inputs for :meth:`ModelBuilder.add_rod`.

    Args:
        start: First point in world space [m].
        direction: World-space direction of the rod (need not be normalized).
        length: Total length of the rod [m].
        num_segments: Number of segments. The returned point count is
            ``num_segments + 1``.
        twist_total: Total twist [rad] distributed along the rod.

    Returns:
        Pair containing the world-space polyline points [m] and matching segment
        orientations as unit quaternions.
    """
    points = cable_straight_points(
        start=start,
        direction=direction,
        length=length,
        num_segments=num_segments,
    )
    quats = rod_parallel_transport_quaternions(points, twist_total=twist_total)
    return points, quats


# Deprecated compatibility
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
    """Deprecated alias for :func:`rod_stiffness_from_elastic_moduli`.

    .. deprecated:: 1.6
        Use :func:`rod_stiffness_from_elastic_moduli` instead.
    """
    warnings.warn(
        "create_cable_stiffness_from_elastic_moduli() is deprecated in Newton 1.6; "
        "use rod_stiffness_from_elastic_moduli() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return rod_stiffness_from_elastic_moduli(
        youngs_modulus,
        radius,
        segment_length,
        poissons_ratio=poissons_ratio,
        shear_modulus=shear_modulus,
    )


def create_parallel_transport_cable_quaternions(
    points: Sequence[wp.vec3],
    *,
    twist_total: float = 0.0,
) -> list[wp.quat]:
    """Deprecated alias for :func:`rod_parallel_transport_quaternions`.

    .. deprecated:: 1.6
        Use :func:`rod_parallel_transport_quaternions` instead.
    """
    warnings.warn(
        "create_parallel_transport_cable_quaternions() is deprecated in Newton 1.6; "
        "use rod_parallel_transport_quaternions() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return rod_parallel_transport_quaternions(points, twist_total=twist_total)


def create_straight_cable_points(
    start: wp.vec3,
    direction: wp.vec3,
    length: float,
    num_segments: int,
) -> list[wp.vec3]:
    """Deprecated alias for :func:`cable_straight_points`.

    .. deprecated:: 1.6
        Use :func:`cable_straight_points` instead.
    """
    warnings.warn(
        "create_straight_cable_points() is deprecated in Newton 1.6; use cable_straight_points() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return cable_straight_points(start, direction, length, num_segments)


def create_straight_cable_points_and_quaternions(
    start: wp.vec3,
    direction: wp.vec3,
    length: float,
    num_segments: int,
    *,
    twist_total: float = 0.0,
) -> tuple[list[wp.vec3], list[wp.quat]]:
    """Deprecated alias for :func:`rod_straight_points_and_quaternions`.

    .. deprecated:: 1.6
        Use :func:`rod_straight_points_and_quaternions` instead.
    """
    warnings.warn(
        "create_straight_cable_points_and_quaternions() is deprecated in Newton 1.6; "
        "use rod_straight_points_and_quaternions() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return rod_straight_points_and_quaternions(
        start,
        direction,
        length,
        num_segments,
        twist_total=twist_total,
    )
