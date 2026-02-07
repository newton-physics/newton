# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
from collections.abc import Sequence

import warp as wp

from ..core.spatial import quat_between_vectors_robust


def create_cable_stiffness_from_elastic_moduli(
    youngs_modulus: float,
    radius: float,
) -> tuple[float, float]:
    """Create rod/cable stiffness parameters from elastic moduli (circular cross-section).

    This returns the *material-like* stiffness values expected by `ModelBuilder.add_rod()` /
    `ModelBuilder.add_rod_graph()`:

    - stretch_stiffness = E * A  [N]
    - bend_stiffness = E * I     [N*m^2]

    where:
    - A = pi * r^2
    - I = (pi * r^4) / 4  (area moment of inertia for a solid circular rod)

    Note: Newton internally converts these into per-joint effective stiffnesses by dividing by
    segment length. This helper intentionally does *not* perform any length normalization.

    Args:
        youngs_modulus: Young's modulus E in Pascals [N/m^2].
        radius: Rod/cable radius r in meters.

    Returns:
        Tuple `(stretch_stiffness, bend_stiffness)` = `(E*A, E*I)`.
    """
    # Accept ints / numpy scalars, but return plain Python floats.
    E = float(youngs_modulus)
    r = float(radius)

    if not math.isfinite(E):
        raise ValueError("youngs_modulus must be finite")
    if not math.isfinite(r):
        raise ValueError("radius must be finite")

    if E < 0.0:
        raise ValueError("youngs_modulus must be >= 0")
    if r <= 0.0:
        raise ValueError("radius must be > 0")

    area = math.pi * r * r
    inertia = 0.25 * math.pi * r**4

    return E * area, E * inertia


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
