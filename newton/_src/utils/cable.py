# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Literal, NamedTuple, overload

import numpy as np
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

    When the 2-tuple is passed through the builder without an explicit
    ``twist_stiffness``, twist defaults to ``bend`` (the combined-stiffness model).
    For material-consistent torsion ``twist / bend = G * J / (E * I) = 1 / (1 + nu)``,
    pass ``poissons_ratio`` or ``shear_modulus`` to get the third term.

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


def _catmull_rom_spans(
    control_points: np.ndarray,
    alpha: float,
    closed: bool,
) -> tuple[list[tuple[tuple[float, float, float, float], np.ndarray]], np.ndarray]:
    """Build per-span evaluation windows for a non-uniform cubic Catmull-Rom spline.

    Knot spacing follows ``t[i+1] - t[i] = |P[i+1] - P[i]|**alpha``. Open curves clamp the
    boundary windows by duplicating the end control points (with duplicated knot values), so the
    curve interpolates the first and last control points. Closed curves wrap the windows.

    Returns:
        A pair ``(spans, knots)`` where ``spans[i]`` is ``((t0, t1, t2, t3), P)`` with ``P`` of
        shape [4, 3]; span ``i`` covers the parameter interval ``[t1, t2] == [knots[i], knots[i+1]]``.
    """
    n = control_points.shape[0]
    spans = []
    if closed:
        deltas = np.linalg.norm(np.roll(control_points, -1, axis=0) - control_points, axis=1) ** alpha
        knots = np.concatenate(([0.0], np.cumsum(deltas)))
        for i in range(n):
            i0, i2, i3 = (i - 1) % n, (i + 1) % n, (i + 2) % n
            t1 = knots[i]
            t2 = knots[i + 1]
            window = ((t1 - deltas[i0], t1, t2, t2 + deltas[i2]), control_points[[i0, i, i2, i3]])
            spans.append(window)
    else:
        deltas = np.linalg.norm(control_points[1:] - control_points[:-1], axis=1) ** alpha
        knots = np.concatenate(([0.0], np.cumsum(deltas)))
        for i in range(n - 1):
            i0 = max(i - 1, 0)
            i3 = min(i + 2, n - 1)
            window = ((knots[i0], knots[i], knots[i + 1], knots[i3]), control_points[[i0, i, i + 1, i3]])
            spans.append(window)
    return spans, knots


def _eval_catmull_rom_span(
    ts4: tuple[float, float, float, float],
    points4: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Evaluate one Catmull-Rom span (Barry-Goldman recursive interpolation) at parameters ``t``."""

    def lerp(a: np.ndarray, b: np.ndarray, ta: float, tb: float) -> np.ndarray:
        if tb == ta:
            # Degenerate window from a clamped (duplicated) boundary control point.
            return np.broadcast_to(a, (t.shape[0], 3))
        w = ((t - ta) / (tb - ta))[:, None]
        return a * (1.0 - w) + b * w

    t0, t1, t2, t3 = ts4
    l01 = lerp(points4[0], points4[1], t0, t1)
    l12 = lerp(points4[1], points4[2], t1, t2)
    l23 = lerp(points4[2], points4[3], t2, t3)
    l012 = lerp(l01, l12, t0, t2)
    l123 = lerp(l12, l23, t1, t3)
    return lerp(l012, l123, t1, t2)


def _eval_catmull_rom(
    spans: list[tuple[tuple[float, float, float, float], np.ndarray]],
    knots: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Evaluate the spline at (sorted or unsorted) parameters ``t`` covering the full domain."""
    num_spans = len(spans)
    span_idx = np.clip(np.searchsorted(knots, t, side="right") - 1, 0, num_spans - 1)
    result = np.empty((t.shape[0], 3), dtype=np.float64)
    for i in range(num_spans):
        mask = span_idx == i
        if mask.any():
            result[mask] = _eval_catmull_rom_span(spans[i][0], spans[i][1], t[mask])
    return result


def _merge_duplicate_control_points(points: np.ndarray, closed: bool) -> np.ndarray:
    """Drop consecutive (and, for closed curves, wrap-around) control points closer than a tolerance."""
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    tol = 1.0e-8 * max(extent, 1.0e-4)
    keep = np.ones(points.shape[0], dtype=bool)
    last = points[0]
    for i in range(1, points.shape[0]):
        if np.linalg.norm(points[i] - last) <= tol:
            keep[i] = False
        else:
            last = points[i]
    merged = points[keep]
    if closed and merged.shape[0] > 1 and np.linalg.norm(merged[-1] - merged[0]) <= tol:
        merged = merged[:-1]
    return merged


def create_cable_spline_points(
    control_points: Sequence[wp.vec3],
    num_segments: int | None = None,
    segment_length: float | None = None,
    *,
    closed: bool = False,
    alpha: float = 0.5,
    oversampling: int = 16,
) -> list[wp.vec3]:
    """Sample a Catmull-Rom spline through the given control points, uniformly in arc length.

    The spline interpolates all control points. Knot spacing uses the ``alpha``
    parameterization (0.5 = centripetal by default, which avoids cusps and local
    self-intersections within spans). The curve is sampled densely to build an arc-length table,
    which is then inverted so the returned polyline nodes are uniformly spaced along the curve
    (equal segment lengths), as expected by :meth:`ModelBuilder.add_rod`.

    Args:
        control_points: Control points the curve interpolates. At least 2 distinct points for an
            open curve, 3 for a closed one; consecutive duplicates are merged.
        num_segments: Number of output segments (>= 2). Exactly one of ``num_segments`` and
            ``segment_length`` must be provided.
        segment_length: Target segment length [m]. The segment count is rounded so segments have
            equal length close to this value (at least 2 segments).
        closed: If True, the spline is a closed loop and the returned polyline ends where it
            starts (the last point equals the first point).
        alpha: Catmull-Rom parameterization exponent: 0.0 uniform, 0.5 centripetal, 1.0 chordal.
        oversampling: Dense samples per output segment used to build the arc-length table.

    Returns:
        List of ``num_segments + 1`` points of type ``wp.vec3``. For ``closed=True`` the last
        point is an exact copy of the first, so passing the result to
        :meth:`ModelBuilder.add_rod` with ``closed=True`` produces a loop with an unstrained
        loop-closing joint.
    """
    if (num_segments is None) == (segment_length is None):
        raise ValueError("create_cable_spline_points: provide exactly one of num_segments and segment_length")
    if num_segments is not None and num_segments < 2:
        raise ValueError("create_cable_spline_points: num_segments must be >= 2")
    if segment_length is not None and (not math.isfinite(segment_length) or segment_length <= 0.0):
        raise ValueError("create_cable_spline_points: segment_length must be positive and finite")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("create_cable_spline_points: alpha must be in [0, 1]")
    if oversampling < 2:
        raise ValueError("create_cable_spline_points: oversampling must be >= 2")

    points = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in control_points], dtype=np.float64)
    if not np.isfinite(points).all():
        raise ValueError("create_cable_spline_points: control_points must be finite")
    points = _merge_duplicate_control_points(points, closed)

    min_points = 3 if closed else 2
    if points.shape[0] < min_points:
        raise ValueError(
            f"create_cable_spline_points: need at least {min_points} distinct control points for a "
            f"{'closed' if closed else 'open'} curve, got {points.shape[0]}"
        )

    spans, knots = _catmull_rom_spans(points, alpha, closed)

    def dense_table(per_span: int) -> tuple[np.ndarray, np.ndarray]:
        t_dense = np.concatenate(
            [np.linspace(knots[i], knots[i + 1], per_span, endpoint=False) for i in range(len(spans))] + [knots[-1:]]
        )
        p_dense = _eval_catmull_rom(spans, knots, t_dense)
        s_dense = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(p_dense, axis=0), axis=1))))
        return t_dense, s_dense

    per_span = 32
    t_dense, s_dense = dense_table(per_span)
    total_length = float(s_dense[-1])
    if total_length <= 0.0:
        raise ValueError("create_cable_spline_points: curve has zero length")

    if num_segments is None:
        num_segments = max(2, round(total_length / segment_length))

    # Ensure the arc-length table is dense enough for the requested resolution.
    per_span_needed = max(per_span, math.ceil(oversampling * num_segments / len(spans)))
    if per_span_needed > per_span:
        t_dense, s_dense = dense_table(per_span_needed)
        total_length = float(s_dense[-1])

    s_targets = np.linspace(0.0, total_length, num_segments + 1)
    t_targets = np.interp(s_targets, s_dense, t_dense)
    samples = _eval_catmull_rom(spans, knots, t_targets)

    if closed:
        samples[-1] = samples[0]

    return [wp.vec3(*p) for p in samples]


def create_rotation_minimizing_cable_quaternions(
    points: Sequence[wp.vec3],
    *,
    twist_total: float = 0.0,
    normal_hint: wp.vec3 | None = None,
    closed: bool = False,
) -> list[wp.quat]:
    """Generate per-segment quaternions using a rotation-minimizing frame (double reflection).

    Like :func:`create_parallel_transport_cable_quaternions`, the returned quaternions rotate the
    capsule's local +Z to each segment (chord) direction while minimizing twist between
    successive segments; the frame is propagated with the double reflection method of Wang et
    al. 2008 ("Computation of rotation minimizing frames"), which is higher-order accurate than
    single-rotation parallel transport and free of the Frenet frame's zero-curvature
    singularities.

    Args:
        points: Polyline points of length >= 2 (segment endpoints).
        twist_total: Total twist [rad] distributed uniformly along the cable (applied about the
            segment directions).
        normal_hint: Optional direction seeding the first cross-section normal (the component
            orthogonal to the first segment is used). If None or (nearly) parallel to the first
            segment, a world axis is chosen automatically.
        closed: If True, treats the polyline as a closed loop (last point coinciding with the
            first) and distributes the loop's holonomy angle evenly along the segments so the
            frame field closes up. For a physically twist-free closed loop, ``twist_total``
            should be a multiple of ``2*pi``.

    Returns:
        List of ``wp.quat`` of length ``len(points) - 1``.
    """
    if len(points) < 2:
        raise ValueError("points must have length >= 2")

    pts = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in points], dtype=np.float64)
    chords = np.diff(pts, axis=0)
    lengths = np.linalg.norm(chords, axis=1)
    if (lengths <= 0.0).any():
        raise ValueError("points must not contain duplicate consecutive points")
    tangents = chords / lengths[:, None]
    midpoints = 0.5 * (pts[:-1] + pts[1:])
    num_segments = tangents.shape[0]
    eps = 1.0e-12

    # Initial cross-section normal: Gram-Schmidt the hint against the first tangent, with an
    # automatic fallback to the world axis least aligned with it.
    normal = None
    if normal_hint is not None:
        hint = np.array([float(normal_hint[0]), float(normal_hint[1]), float(normal_hint[2])], dtype=np.float64)
        projected = hint - tangents[0] * float(hint @ tangents[0])
        norm = np.linalg.norm(projected)
        if norm > 1.0e-6:
            normal = projected / norm
    if normal is None:
        axis = np.zeros(3)
        axis[int(np.argmin(np.abs(tangents[0])))] = 1.0
        projected = axis - tangents[0] * float(axis @ tangents[0])
        normal = projected / np.linalg.norm(projected)

    def double_reflection(r: np.ndarray, x0: np.ndarray, t0: np.ndarray, x1: np.ndarray, t1: np.ndarray) -> np.ndarray:
        v1 = x1 - x0
        c1 = float(v1 @ v1)
        if c1 > eps:
            r_l = r - (2.0 / c1) * float(v1 @ r) * v1
            t_l = t0 - (2.0 / c1) * float(v1 @ t0) * v1
        else:
            # Exact fold-back (coincident sample points): keep the reflected quantities as-is.
            r_l = r
            t_l = t0
        v2 = t1 - t_l
        c2 = float(v2 @ v2)
        if c2 > eps:
            r_l = r_l - (2.0 / c2) * float(v2 @ r_l) * v2
        # Re-orthonormalize against the new tangent to prevent numerical drift.
        r_l = r_l - t1 * float(r_l @ t1)
        return r_l / np.linalg.norm(r_l)

    normals = [normal]
    for k in range(num_segments - 1):
        normals.append(double_reflection(normals[k], midpoints[k], tangents[k], midpoints[k + 1], tangents[k + 1]))

    # Closed loops: transport once more around the wrap to measure the holonomy angle, then
    # distribute it evenly so every joint (including the loop-closing one) sees the same twist.
    holonomy_step = 0.0
    if closed and num_segments >= 2:
        r_loop = double_reflection(normals[-1], midpoints[-1], tangents[-1], midpoints[0], tangents[0])
        phi = math.atan2(float(np.cross(r_loop, normals[0]) @ tangents[0]), float(r_loop @ normals[0]))
        holonomy_step = phi / num_segments

    twist_step = float(twist_total) / num_segments

    quats: list[wp.quat] = []
    for k in range(num_segments):
        roll = (k + 1) * twist_step + k * holonomy_step
        r = normals[k]
        t = tangents[k]
        if roll != 0.0:
            u = np.cross(t, r)
            r = r * math.cos(roll) + u * math.sin(roll)
        u = np.cross(t, r)
        # World-from-local rotation with columns (normal, binormal, tangent): local +Z -> chord.
        q = wp.quat_from_matrix(
            wp.mat33(
                r[0],
                u[0],
                t[0],
                r[1],
                u[1],
                t[1],
                r[2],
                u[2],
                t[2],
            )
        )
        quats.append(wp.normalize(q))

    return quats


class CableSplineShape(NamedTuple):
    """Initial and rest geometry of a spline cable, as produced by :func:`create_cable_spline_shape`.

    ``points``/``quaternions`` describe the cable posed along the spline (the intended
    simulation *initial* configuration); ``rest_points``/``rest_quaternions`` describe the
    *rest* (zero-strain) configuration. When the rest shape is the spline itself, the rest
    fields alias the posed fields.
    """

    points: list[wp.vec3]
    """Polyline points along the spline [m], ``num_segments + 1`` entries."""
    quaternions: list[wp.quat]
    """Per-segment rotation-minimizing frames along the spline, ``num_segments`` entries."""
    rest_points: list[wp.vec3]
    """Rest-configuration polyline points [m], ``num_segments + 1`` entries."""
    rest_quaternions: list[wp.quat]
    """Per-segment rest-configuration frames, ``num_segments`` entries."""


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares plane fit; returns (centroid, in-plane basis u, in-plane basis v)."""
    centroid = points.mean(axis=0)
    # Right singular vectors of the centered cloud: first two span the plane.
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    return centroid, vt[0], vt[1]


def create_cable_spline_shape(
    control_points: Sequence[wp.vec3],
    num_segments: int | None = None,
    segment_length: float | None = None,
    *,
    closed: bool = False,
    twist_total: float = 0.0,
    normal_hint: wp.vec3 | None = None,
    alpha: float = 0.5,
    straight_rest_shape: bool = False,
) -> CableSplineShape:
    """Generate the posed and rest geometry of a cable that follows a Catmull-Rom spline.

    The posed geometry samples the spline uniformly in arc length and generates per-segment
    rotation-minimizing frames (see :func:`create_cable_spline_points` and
    :func:`create_rotation_minimizing_cable_quaternions`). By default the rest geometry is the
    spline itself, so a cable built from it holds the spline shape at equilibrium.

    With ``straight_rest_shape=True`` the rest geometry is instead the *natural* (minimal
    bending, untwisted) configuration of the same cable:

    - **Open** cables: a straight, untwisted chain of the same per-segment lengths, laid out
      from the spline's first point along its first segment direction.
    - **Closed** cables: a straight rest pose cannot close the loop, so the rest shape is a
      regular polygon ("circle") of the same circumference, centered at the spline loop's
      centroid in its least-squares best-fit plane. This placement minimizes the per-joint
      rotation between the rest and posed configurations.

    Build the cable at the rest geometry (``rest_points``/``rest_quaternions``) so
    :attr:`Model.body_q` conveys the rest configuration, then write the posed configuration
    into the simulation state — see ``rest_shape`` in :meth:`ModelBuilder.add_cable_spline`
    and :func:`create_cable_body_transforms`.

    Args:
        control_points: Control points the spline interpolates. See
            :func:`create_cable_spline_points`.
        num_segments: Number of segments (>= 2). Exactly one of ``num_segments`` and
            ``segment_length`` must be provided.
        segment_length: Target segment length [m].
        closed: If True, the spline is a closed loop.
        twist_total: Total twist [rad] distributed uniformly along the *posed* cable. The rest
            configuration is always untwisted, so with ``straight_rest_shape=True`` any twist
            becomes live strain.
        normal_hint: Optional direction seeding the first cross-section normal, shared by the
            posed and rest frames so their rolls match at the first segment.
        alpha: Catmull-Rom parameterization exponent (0.5 = centripetal).
        straight_rest_shape: If True, the rest geometry is the natural configuration described
            above instead of the spline.

    Returns:
        A :class:`CableSplineShape`. When ``straight_rest_shape=False`` the rest fields alias
        the posed fields.
    """
    points = create_cable_spline_points(
        control_points,
        num_segments=num_segments,
        segment_length=segment_length,
        closed=closed,
        alpha=alpha,
    )
    quaternions = create_rotation_minimizing_cable_quaternions(
        points,
        twist_total=twist_total,
        normal_hint=normal_hint,
        closed=closed,
    )

    if not straight_rest_shape:
        return CableSplineShape(points, quaternions, points, quaternions)

    pts = np.array([[p[0], p[1], p[2]] for p in points], dtype=np.float64)
    seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    n = seg_lengths.shape[0]

    if closed:
        # Regular n-gon with the same circumference, in the loop's best-fit plane. The
        # inscribed polygon side is uniform; the posed segments match it to within the
        # arc-length sampling tolerance.
        side = float(seg_lengths.sum()) / n
        circumradius = side / (2.0 * math.sin(math.pi / n))
        centroid, u, v = _fit_plane(pts[:-1])
        angles = 2.0 * math.pi * np.arange(n + 1) / n
        rest_pts = centroid + circumradius * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)
        rest_pts[-1] = rest_pts[0]
        rest_points = [wp.vec3(*p) for p in rest_pts]
        rest_quaternions = create_rotation_minimizing_cable_quaternions(
            rest_points, normal_hint=normal_hint, closed=True
        )
    else:
        # Straight chain preserving the exact per-segment lengths, from the spline start
        # along the first segment direction.
        direction = (pts[1] - pts[0]) / seg_lengths[0]
        offsets = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        rest_pts = pts[0] + offsets[:, None] * direction
        rest_points = [wp.vec3(*p) for p in rest_pts]
        rest_quaternions = create_rotation_minimizing_cable_quaternions(rest_points, normal_hint=normal_hint)

    # Per-joint relative rotation between the rest and posed configurations must stay clearly
    # below pi: the quaternion strain measure snaps to the antipode beyond that.
    max_deviation = 0.0
    joint_pairs = list(zip(range(n - 1), range(1, n), strict=True))
    if closed:
        joint_pairs.append((n - 1, 0))
    for a, b in joint_pairs:
        rel_posed = wp.mul(wp.quat_inverse(quaternions[a]), quaternions[b])
        rel_rest = wp.mul(wp.quat_inverse(rest_quaternions[a]), rest_quaternions[b])
        deviation = wp.mul(wp.quat_inverse(rel_rest), rel_posed)
        angle = 2.0 * math.acos(min(1.0, abs(float(deviation[3]))))
        max_deviation = max(max_deviation, angle)
    if max_deviation > 0.9 * math.pi:
        warnings.warn(
            f"create_cable_spline_shape: maximum per-joint rotation between the rest and posed "
            f"configurations is {max_deviation:.2f} rad, close to the pi limit of the joint "
            f"strain measure; increase the segment count or smooth the spline.",
            UserWarning,
            stacklevel=2,
        )

    return CableSplineShape(points, quaternions, rest_points, rest_quaternions)


def create_cable_body_transforms(
    points: Sequence[wp.vec3],
    quaternions: Sequence[wp.quat],
    *,
    body_frame_origin: Literal["start", "com"] = "com",
) -> list[wp.transform]:
    """Convert cable polyline points and per-segment frames to per-body transforms.

    Useful for writing a posed cable configuration into ``State.body_q`` for rods built with
    :meth:`ModelBuilder.add_rod` or :meth:`ModelBuilder.add_cable_spline` — e.g. to start a
    straight-rest cable from a routed pose (see ``rest_shape`` in
    :meth:`ModelBuilder.add_cable_spline`).

    Args:
        points: Polyline points [m], one more than the number of segments.
        quaternions: Per-segment orientations (local +Z along each segment).
        body_frame_origin: Body-frame placement matching the one used to build the rod:
            ``"com"`` places the body origin at the segment midpoint, ``"start"`` at the
            segment's first point.

    Returns:
        List of ``wp.transform``, one per segment body.
    """
    num_segments = len(points) - 1
    if len(quaternions) != num_segments:
        raise ValueError(
            f"create_cable_body_transforms: expected {num_segments} quaternions for "
            f"{num_segments} segments, got {len(quaternions)}"
        )
    if body_frame_origin not in ("start", "com"):
        raise ValueError("create_cable_body_transforms: body_frame_origin must be 'start' or 'com'")

    transforms = []
    for i in range(num_segments):
        if body_frame_origin == "com":
            origin = 0.5 * (points[i] + points[i + 1])
        else:
            origin = points[i]
        transforms.append(wp.transform(origin, quaternions[i]))
    return transforms


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
