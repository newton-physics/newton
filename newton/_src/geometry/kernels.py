# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

import warp as wp

from .broad_phase_common import binary_search
from .flags import ParticleFlags, ShapeFlags
from .types import (
    Axis,
    GeoType,
)


@wp.func
def triangle_closest_point_barycentric(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)

    if d1 <= 0.0 and d2 <= 0.0:
        return wp.vec3(1.0, 0.0, 0.0)

    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)

    if d3 >= 0.0 and d4 <= d3:
        return wp.vec3(0.0, 1.0, 0.0)

    vc = d1 * d4 - d3 * d2
    v = d1 / (d1 - d3)
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return wp.vec3(1.0 - v, v, 0.0)

    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)

    if d6 >= 0.0 and d5 <= d6:
        return wp.vec3(0.0, 0.0, 1.0)

    vb = d5 * d2 - d1 * d6
    w = d2 / (d2 - d6)
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return wp.vec3(1.0 - w, 0.0, w)

    va = d3 * d6 - d5 * d4
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        return wp.vec3(0.0, 1.0 - w, w)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom

    return wp.vec3(1.0 - v - w, v, w)


@wp.func
def triangle_closest_point(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):
    """
    feature_type type:
        TRI_CONTACT_FEATURE_VERTEX_A
        TRI_CONTACT_FEATURE_VERTEX_B
        TRI_CONTACT_FEATURE_VERTEX_C
        TRI_CONTACT_FEATURE_EDGE_AB      : at edge A-B
        TRI_CONTACT_FEATURE_EDGE_AC      : at edge A-C
        TRI_CONTACT_FEATURE_EDGE_BC      : at edge B-C
        TRI_CONTACT_FEATURE_FACE_INTERIOR
    """
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        feature_type = TRI_CONTACT_FEATURE_VERTEX_A
        bary = wp.vec3(1.0, 0.0, 0.0)
        return a, bary, feature_type

    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        feature_type = TRI_CONTACT_FEATURE_VERTEX_B
        bary = wp.vec3(0.0, 1.0, 0.0)
        return b, bary, feature_type

    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        feature_type = TRI_CONTACT_FEATURE_VERTEX_C
        bary = wp.vec3(0.0, 0.0, 1.0)
        return c, bary, feature_type

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        feature_type = TRI_CONTACT_FEATURE_EDGE_AB
        bary = wp.vec3(1.0 - v, v, 0.0)
        return a + v * ab, bary, feature_type

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        v = d2 / (d2 - d6)
        feature_type = TRI_CONTACT_FEATURE_EDGE_AC
        bary = wp.vec3(1.0 - v, 0.0, v)
        return a + v * ac, bary, feature_type

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        v = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        feature_type = TRI_CONTACT_FEATURE_EDGE_BC
        bary = wp.vec3(0.0, 1.0 - v, v)
        return b + v * (c - b), bary, feature_type

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    feature_type = TRI_CONTACT_FEATURE_FACE_INTERIOR
    bary = wp.vec3(1.0 - v - w, v, w)
    return a + v * ab + w * ac, bary, feature_type


@wp.func
def _sdf_point_to_z_up(point: wp.vec3, up_axis: int):
    if up_axis == int(Axis.X):
        return wp.vec3(point[1], point[2], point[0])
    if up_axis == int(Axis.Y):
        return wp.vec3(point[0], point[2], point[1])
    return point


@wp.func
def _sdf_capped_cone_z(bottom_radius: float, top_radius: float, half_height: float, point_z_up: wp.vec3):
    q = wp.vec2(wp.length(wp.vec2(point_z_up[0], point_z_up[1])), point_z_up[2])
    k1 = wp.vec2(top_radius, half_height)
    k2 = wp.vec2(top_radius - bottom_radius, 2.0 * half_height)

    if q[1] < 0.0:
        ca = wp.vec2(q[0] - wp.min(q[0], bottom_radius), wp.abs(q[1]) - half_height)
    else:
        ca = wp.vec2(q[0] - wp.min(q[0], top_radius), wp.abs(q[1]) - half_height)

    denom = wp.dot(k2, k2)
    t = 0.0
    if denom > 0.0:
        t = wp.clamp(wp.dot(k1 - q, k2) / denom, 0.0, 1.0)
    cb = q - k1 + k2 * t

    sign = 1.0
    if cb[0] < 0.0 and ca[1] < 0.0:
        sign = -1.0

    return sign * wp.sqrt(wp.min(wp.dot(ca, ca), wp.dot(cb, cb)))


@wp.func
def sdf_sphere(point: wp.vec3, radius: float):
    """Compute signed distance to a sphere for ``Mesh.create_sphere`` geometry.

    Args:
        point [m]: Query point in the mesh local frame, shape [3], float.
        radius [m]: Sphere radius.

    Returns:
        Signed distance [m], negative inside, zero on surface, positive outside.
    """
    return wp.length(point) - radius


@wp.func
def sdf_sphere_grad(point: wp.vec3, radius: float):
    """Compute outward SDF gradient for ``sdf_sphere``.

    Args:
        point [m]: Query point in the mesh local frame, shape [3], float.
        radius [m]: Sphere radius (unused, kept for API symmetry).

    Returns:
        Unit-length gradient direction in local frame.
    """
    _ = radius
    eps = 1.0e-8
    p_len = wp.length(point)
    if p_len > eps:
        return point / p_len
    return wp.vec3(0.0, 0.0, 1.0)


@wp.func
def sdf_box(point: wp.vec3, hx: float, hy: float, hz: float):
    """Compute signed distance to an axis-aligned box.

    Args:
        point [m]: Query point in the mesh local frame, shape [3], float.
        hx [m]: Half-extent along X.
        hy [m]: Half-extent along Y.
        hz [m]: Half-extent along Z.

    Returns:
        Signed distance [m], negative inside, zero on surface, positive outside.
    """
    # adapted from https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
    qx = abs(point[0]) - hx
    qy = abs(point[1]) - hy
    qz = abs(point[2]) - hz

    e = wp.vec3(wp.max(qx, 0.0), wp.max(qy, 0.0), wp.max(qz, 0.0))

    return wp.length(e) + wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)


@wp.func
def sdf_box_grad(point: wp.vec3, hx: float, hy: float, hz: float):
    """Compute outward SDF gradient for ``sdf_box``.

    Args:
        point [m]: Query point in the mesh local frame, shape [3], float.
        hx [m]: Half-extent along X.
        hy [m]: Half-extent along Y.
        hz [m]: Half-extent along Z.

    Returns:
        Unit-length (or axis-aligned) outward gradient direction.
    """
    qx = abs(point[0]) - hx
    qy = abs(point[1]) - hy
    qz = abs(point[2]) - hz

    # exterior case
    if qx > 0.0 or qy > 0.0 or qz > 0.0:
        x = wp.clamp(point[0], -hx, hx)
        y = wp.clamp(point[1], -hy, hy)
        z = wp.clamp(point[2], -hz, hz)

        return wp.normalize(point - wp.vec3(x, y, z))

    sx = wp.sign(point[0])
    sy = wp.sign(point[1])
    sz = wp.sign(point[2])

    # x projection
    if (qx > qy and qx > qz) or (qy == 0.0 and qz == 0.0):
        return wp.vec3(sx, 0.0, 0.0)

    # y projection
    if (qy > qx and qy > qz) or (qx == 0.0 and qz == 0.0):
        return wp.vec3(0.0, sy, 0.0)

    # z projection
    return wp.vec3(0.0, 0.0, sz)


@wp.func
def sdf_capsule(point: wp.vec3, radius: float, half_height: float, up_axis: int = int(Axis.Y)):
    """Compute signed distance to a capsule for ``Mesh.create_capsule`` geometry.

    Args:
        point [m]: Query point in the mesh local frame, shape [3], float.
        radius [m]: Capsule radius.
        half_height [m]: Half-height of the cylindrical section.
        up_axis: Capsule long axis as ``int(newton.Axis.*)``.

    Returns:
        Signed distance [m], negative inside, zero on surface, positive outside.
    """
    point_z_up = _sdf_point_to_z_up(point, up_axis)
    if point_z_up[2] > half_height:
        return wp.length(wp.vec3(point_z_up[0], point_z_up[1], point_z_up[2] - half_height)) - radius

    if point_z_up[2] < -half_height:
        return wp.length(wp.vec3(point_z_up[0], point_z_up[1], point_z_up[2] + half_height)) - radius

    return wp.length(wp.vec3(point_z_up[0], point_z_up[1], 0.0)) - radius


@wp.func
def _sdf_vector_from_z_up(v: wp.vec3, up_axis: int):
    if up_axis == int(Axis.X):
        return wp.vec3(v[2], v[0], v[1])
    if up_axis == int(Axis.Y):
        return wp.vec3(v[0], v[2], v[1])
    return v


@wp.func
def sdf_capsule_grad(point: wp.vec3, radius: float, half_height: float, up_axis: int = int(Axis.Y)):
    """Compute outward SDF gradient for ``sdf_capsule``.

    Args:
        point [m]: Query point in the mesh local frame, shape [3], float.
        radius [m]: Capsule radius.
        half_height [m]: Half-height of the cylindrical section.
        up_axis: Capsule long axis as ``int(newton.Axis.*)``.

    Returns:
        Unit-length outward gradient direction in local frame.
    """
    _ = radius
    eps = 1.0e-8
    point_z_up = _sdf_point_to_z_up(point, up_axis)
    grad_z_up = wp.vec3()
    if point_z_up[2] > half_height:
        v = wp.vec3(point_z_up[0], point_z_up[1], point_z_up[2] - half_height)
        v_len = wp.length(v)
        grad_z_up = wp.vec3(0.0, 0.0, 1.0)
        if v_len > eps:
            grad_z_up = v / v_len
    elif point_z_up[2] < -half_height:
        v = wp.vec3(point_z_up[0], point_z_up[1], point_z_up[2] + half_height)
        v_len = wp.length(v)
        grad_z_up = wp.vec3(0.0, 0.0, -1.0)
        if v_len > eps:
            grad_z_up = v / v_len
    else:
        v = wp.vec3(point_z_up[0], point_z_up[1], 0.0)
        v_len = wp.length(v)
        grad_z_up = wp.vec3(0.0, 0.0, 1.0)
        if v_len > eps:
            grad_z_up = v / v_len
    return _sdf_vector_from_z_up(grad_z_up, up_axis)


@wp.func
def sdf_cylinder(
    point: wp.vec3,
    radius: float,
    half_height: float,
    up_axis: int = int(Axis.Y),
    top_radius: float = -1.0,
):
    """Compute signed distance to ``Mesh.create_cylinder`` geometry.

    Args:
        point [m]: Query point in the mesh local frame, shape [3], float.
        radius [m]: Bottom radius.
        half_height [m]: Half-height along the cylinder axis.
        up_axis: Cylinder long axis as ``int(newton.Axis.*)``.
        top_radius [m]: Top radius. Negative values use ``radius``.

    Returns:
        Signed distance [m], negative inside, zero on surface, positive outside.
    """
    point_z_up = _sdf_point_to_z_up(point, up_axis)
    if top_radius < 0.0 or wp.abs(top_radius - radius) <= 1.0e-6:
        dx = wp.length(wp.vec3(point_z_up[0], point_z_up[1], 0.0)) - radius
        dy = wp.abs(point_z_up[2]) - half_height
        return wp.min(wp.max(dx, dy), 0.0) + wp.length(wp.vec2(wp.max(dx, 0.0), wp.max(dy, 0.0)))
    return _sdf_capped_cone_z(radius, top_radius, half_height, point_z_up)


@wp.func
def sdf_cylinder_grad(
    point: wp.vec3,
    radius: float,
    half_height: float,
    up_axis: int = int(Axis.Y),
    top_radius: float = -1.0,
):
    """Compute outward SDF gradient for ``sdf_cylinder``.

    Args:
        point [m]: Query point in the mesh local frame, shape [3], float.
        radius [m]: Bottom radius.
        half_height [m]: Half-height along the cylinder axis.
        up_axis: Cylinder long axis as ``int(newton.Axis.*)``.
        top_radius [m]: Top radius. Negative values use ``radius``.

    Returns:
        Unit-length outward gradient direction in local frame.
    """
    eps = 1.0e-8
    point_z_up = _sdf_point_to_z_up(point, up_axis)
    if top_radius >= 0.0 and wp.abs(top_radius - radius) > 1.0e-6:
        # Use finite-difference gradient of the tapered capped-cone SDF.
        fd_eps = 1.0e-4
        dx = _sdf_capped_cone_z(
            radius,
            top_radius,
            half_height,
            point_z_up + wp.vec3(fd_eps, 0.0, 0.0),
        ) - _sdf_capped_cone_z(
            radius,
            top_radius,
            half_height,
            point_z_up - wp.vec3(fd_eps, 0.0, 0.0),
        )
        dy = _sdf_capped_cone_z(
            radius,
            top_radius,
            half_height,
            point_z_up + wp.vec3(0.0, fd_eps, 0.0),
        ) - _sdf_capped_cone_z(
            radius,
            top_radius,
            half_height,
            point_z_up - wp.vec3(0.0, fd_eps, 0.0),
        )
        dz = _sdf_capped_cone_z(
            radius,
            top_radius,
            half_height,
            point_z_up + wp.vec3(0.0, 0.0, fd_eps),
        ) - _sdf_capped_cone_z(
            radius,
            top_radius,
            half_height,
            point_z_up - wp.vec3(0.0, 0.0, fd_eps),
        )
        grad_z_up = wp.vec3(dx, dy, dz)
        grad_len = wp.length(grad_z_up)
        if grad_len > eps:
            grad_z_up = grad_z_up / grad_len
        else:
            grad_z_up = wp.vec3(0.0, 0.0, 1.0)
        return _sdf_vector_from_z_up(grad_z_up, up_axis)

    v = wp.vec3(point_z_up[0], point_z_up[1], 0.0)
    v_len = wp.length(v)
    radial = wp.vec3(0.0, 0.0, 1.0)
    if v_len > eps:
        radial = v / v_len
    axial = wp.vec3(0.0, 0.0, wp.sign(point_z_up[2]))
    dx = v_len - radius
    dy = wp.abs(point_z_up[2]) - half_height
    grad_z_up = wp.vec3()
    if dx > 0.0 and dy > 0.0:
        g = radial * dx + axial * dy
        g_len = wp.length(g)
        if g_len > eps:
            grad_z_up = g / g_len
        else:
            grad_z_up = radial
    elif dx > dy:
        grad_z_up = radial
    else:
        grad_z_up = axial
    return _sdf_vector_from_z_up(grad_z_up, up_axis)


@wp.func
def sdf_ellipsoid(point: wp.vec3, radii: wp.vec3):
    """Compute approximate signed distance to an ellipsoid.

    Args:
        point [m]: Query point in the mesh local frame, shape [3], float.
        radii [m]: Ellipsoid radii along XYZ, shape [3], float.

    Returns:
        Approximate signed distance [m], negative inside, positive outside.
    """
    # Approximate SDF for ellipsoid with radii (rx, ry, rz)
    # Using the approximation: k0 * (k0 - 1) / k1
    eps = 1.0e-8
    r = wp.vec3(
        wp.max(wp.abs(radii[0]), eps),
        wp.max(wp.abs(radii[1]), eps),
        wp.max(wp.abs(radii[2]), eps),
    )
    inv_r = wp.cw_div(wp.vec3(1.0, 1.0, 1.0), r)
    inv_r2 = wp.cw_mul(inv_r, inv_r)
    q0 = wp.cw_mul(point, inv_r)  # p / r
    q1 = wp.cw_mul(point, inv_r2)  # p / r^2
    k0 = wp.length(q0)
    k1 = wp.length(q1)
    if k1 > eps:
        return k0 * (k0 - 1.0) / k1
    # Deep inside / near center fallback
    return -wp.min(wp.min(r[0], r[1]), r[2])


@wp.func
def sdf_ellipsoid_grad(point: wp.vec3, radii: wp.vec3):
    """Compute approximate outward SDF gradient for ``sdf_ellipsoid``.

    Args:
        point [m]: Query point in the mesh local frame, shape [3], float.
        radii [m]: Ellipsoid radii along XYZ, shape [3], float.

    Returns:
        Unit-length approximate outward gradient direction.
    """
    # Gradient of the ellipsoid SDF approximation
    # grad(d) ≈ normalize((k0 / k1) * (p / r^2))
    eps = 1.0e-8
    r = wp.vec3(
        wp.max(wp.abs(radii[0]), eps),
        wp.max(wp.abs(radii[1]), eps),
        wp.max(wp.abs(radii[2]), eps),
    )
    inv_r = wp.cw_div(wp.vec3(1.0, 1.0, 1.0), r)
    inv_r2 = wp.cw_mul(inv_r, inv_r)
    q0 = wp.cw_mul(point, inv_r)  # p / r
    q1 = wp.cw_mul(point, inv_r2)  # p / r^2
    k0 = wp.length(q0)
    k1 = wp.length(q1)
    if k1 < eps:
        return wp.vec3(0.0, 0.0, 1.0)
    # Analytic gradient of the approximation
    grad = q1 * (k0 / k1)
    grad_len = wp.length(grad)
    if grad_len > eps:
        return grad / grad_len
    return wp.vec3(0.0, 0.0, 1.0)


@wp.func
def sdf_cone(point: wp.vec3, radius: float, half_height: float, up_axis: int = int(Axis.Y)):
    """Compute signed distance to a cone for ``Mesh.create_cone`` geometry.

    Args:
        point [m]: Query point in the mesh local frame, shape [3], float.
        radius [m]: Cone base radius.
        half_height [m]: Half-height from center to apex/base.
        up_axis: Cone long axis as ``int(newton.Axis.*)``.

    Returns:
        Signed distance [m], negative inside, zero on surface, positive outside.
    """
    point_z_up = _sdf_point_to_z_up(point, up_axis)
    return _sdf_capped_cone_z(radius, 0.0, half_height, point_z_up)


@wp.func
def sdf_cone_grad(point: wp.vec3, radius: float, half_height: float, up_axis: int = int(Axis.Y)):
    """Compute outward SDF gradient for ``sdf_cone``.

    Args:
        point [m]: Query point in the mesh local frame, shape [3], float.
        radius [m]: Cone base radius.
        half_height [m]: Half-height from center to apex/base.
        up_axis: Cone long axis as ``int(newton.Axis.*)``.

    Returns:
        Unit-length outward gradient direction in local frame.
    """
    # Gradient for cone with apex at +half_height and base at -half_height
    point_z_up = _sdf_point_to_z_up(point, up_axis)
    r = wp.length(wp.vec3(point_z_up[0], point_z_up[1], 0.0))
    dx = r - radius * (half_height - point_z_up[2]) / (2.0 * half_height)
    dy = wp.abs(point_z_up[2]) - half_height
    grad_z_up = wp.vec3()
    if dx > dy:
        # Closest to lateral surface
        if r > 0.0:
            radial_dir = wp.vec3(point_z_up[0], point_z_up[1], 0.0) / r
            # Normal to cone surface
            grad_z_up = wp.normalize(radial_dir + wp.vec3(0.0, 0.0, radius / (2.0 * half_height)))
        else:
            grad_z_up = wp.vec3(0.0, 0.0, 1.0)
    else:
        # Closest to cap
        grad_z_up = wp.vec3(0.0, 0.0, wp.sign(point_z_up[2]))
    return _sdf_vector_from_z_up(grad_z_up, up_axis)


@wp.func
def sdf_plane(point: wp.vec3, width: float, length: float):
    """Compute signed distance to a finite quad in the XY plane.

    Args:
        point [m]: Query point in the mesh local frame, shape [3], float.
        width [m]: Half-extent along X.
        length [m]: Half-extent along Y.

    Returns:
        Distance [m]. For finite extents (``width > 0`` and ``length > 0``), this
        is a Chebyshev (L∞) distance approximation to the quad sheet (not exact
        Euclidean distance). The exact Euclidean distance would be
        ``sqrt(max(|x|-width, 0)^2 + max(|y|-length, 0)^2 + z^2)``.
        Otherwise, for ``width <= 0`` or ``length <= 0``, it reduces to the
        signed distance of the infinite plane (``point.z``).
    """
    # SDF for a quad in the xy plane
    if width > 0.0 and length > 0.0:
        d = wp.max(wp.abs(point[0]) - width, wp.abs(point[1]) - length)
        return wp.max(d, wp.abs(point[2]))
    return point[2]


@wp.func
def sdf_plane_grad(point: wp.vec3, width: float, length: float):
    """Compute a simple upward normal for ``sdf_plane``.

    Args:
        point [m]: Query point in the mesh local frame, shape [3], float.
        width [m]: Half-extent along X.
        length [m]: Half-extent along Y.

    Returns:
        Upward unit normal in local frame.
    """
    _ = (width, length, point)
    return wp.vec3(0.0, 0.0, 1.0)


@wp.func
def closest_point_plane(width: float, length: float, point: wp.vec3):
    # projects the point onto the quad in the xy plane (if width and length > 0.0, otherwise the plane is infinite)
    if width > 0.0:
        x = wp.clamp(point[0], -width, width)
    else:
        x = point[0]
    if length > 0.0:
        y = wp.clamp(point[1], -length, length)
    else:
        y = point[1]
    return wp.vec3(x, y, 0.0)


@wp.func
def closest_point_line_segment(a: wp.vec3, b: wp.vec3, point: wp.vec3):
    ab = b - a
    ap = point - a
    t = wp.dot(ap, ab) / wp.dot(ab, ab)
    t = wp.clamp(t, 0.0, 1.0)
    return a + t * ab


@wp.func
def closest_point_box(upper: wp.vec3, point: wp.vec3):
    # closest point to box surface
    x = wp.clamp(point[0], -upper[0], upper[0])
    y = wp.clamp(point[1], -upper[1], upper[1])
    z = wp.clamp(point[2], -upper[2], upper[2])
    if wp.abs(point[0]) <= upper[0] and wp.abs(point[1]) <= upper[1] and wp.abs(point[2]) <= upper[2]:
        # the point is inside, find closest face
        sx = wp.abs(wp.abs(point[0]) - upper[0])
        sy = wp.abs(wp.abs(point[1]) - upper[1])
        sz = wp.abs(wp.abs(point[2]) - upper[2])
        # return closest point on closest side, handle corner cases
        if (sx < sy and sx < sz) or (sy == 0.0 and sz == 0.0):
            x = wp.sign(point[0]) * upper[0]
        elif (sy < sx and sy < sz) or (sx == 0.0 and sz == 0.0):
            y = wp.sign(point[1]) * upper[1]
        else:
            z = wp.sign(point[2]) * upper[2]
    return wp.vec3(x, y, z)


@wp.func
def get_box_vertex(point_id: int, upper: wp.vec3):
    # box vertex numbering:
    #    6---7
    #    |\  |\       y
    #    | 2-+-3      |
    #    4-+-5 |   z \|
    #     \|  \|      o---x
    #      0---1
    # get the vertex of the box given its ID (0-7)
    sign_x = float(point_id % 2) * 2.0 - 1.0
    sign_y = float((point_id // 2) % 2) * 2.0 - 1.0
    sign_z = float((point_id // 4) % 2) * 2.0 - 1.0
    return wp.vec3(sign_x * upper[0], sign_y * upper[1], sign_z * upper[2])


@wp.func
def get_box_edge(edge_id: int, upper: wp.vec3):
    # get the edge of the box given its ID (0-11)
    if edge_id < 4:
        # edges along x: 0-1, 2-3, 4-5, 6-7
        i = edge_id * 2
        j = i + 1
        return wp.spatial_vector(get_box_vertex(i, upper), get_box_vertex(j, upper))
    elif edge_id < 8:
        # edges along y: 0-2, 1-3, 4-6, 5-7
        edge_id -= 4
        i = edge_id % 2 + edge_id // 2 * 4
        j = i + 2
        return wp.spatial_vector(get_box_vertex(i, upper), get_box_vertex(j, upper))
    # edges along z: 0-4, 1-5, 2-6, 3-7
    edge_id -= 8
    i = edge_id
    j = i + 4
    return wp.spatial_vector(get_box_vertex(i, upper), get_box_vertex(j, upper))


@wp.func
def get_plane_edge(edge_id: int, plane_width: float, plane_length: float):
    # get the edge of the plane given its ID (0-3)
    p0x = (2.0 * float(edge_id % 2) - 1.0) * plane_width
    p0y = (2.0 * float(edge_id // 2) - 1.0) * plane_length
    if edge_id == 0 or edge_id == 3:
        p1x = p0x
        p1y = -p0y
    else:
        p1x = -p0x
        p1y = p0y
    return wp.spatial_vector(wp.vec3(p0x, p0y, 0.0), wp.vec3(p1x, p1y, 0.0))


@wp.func
def closest_edge_coordinate_box(upper: wp.vec3, edge_a: wp.vec3, edge_b: wp.vec3, max_iter: int):
    # find point on edge closest to box, return its barycentric edge coordinate
    # Golden-section search
    a = float(0.0)
    b = float(1.0)
    h = b - a
    invphi = 0.61803398875  # 1 / phi
    invphi2 = 0.38196601125  # 1 / phi^2
    c = a + invphi2 * h
    d = a + invphi * h
    query = (1.0 - c) * edge_a + c * edge_b
    yc = sdf_box(query, upper[0], upper[1], upper[2])
    query = (1.0 - d) * edge_a + d * edge_b
    yd = sdf_box(query, upper[0], upper[1], upper[2])

    for _k in range(max_iter):
        if yc < yd:  # yc > yd to find the maximum
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            query = (1.0 - c) * edge_a + c * edge_b
            yc = sdf_box(query, upper[0], upper[1], upper[2])
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            query = (1.0 - d) * edge_a + d * edge_b
            yd = sdf_box(query, upper[0], upper[1], upper[2])

    if yc < yd:
        return 0.5 * (a + d)
    return 0.5 * (c + b)


@wp.func
def closest_edge_coordinate_plane(
    plane_width: float,
    plane_length: float,
    edge_a: wp.vec3,
    edge_b: wp.vec3,
    max_iter: int,
):
    # find point on edge closest to plane, return its barycentric edge coordinate
    # Golden-section search
    a = float(0.0)
    b = float(1.0)
    h = b - a
    invphi = 0.61803398875  # 1 / phi
    invphi2 = 0.38196601125  # 1 / phi^2
    c = a + invphi2 * h
    d = a + invphi * h
    query = (1.0 - c) * edge_a + c * edge_b
    yc = sdf_plane(query, plane_width, plane_length)
    query = (1.0 - d) * edge_a + d * edge_b
    yd = sdf_plane(query, plane_width, plane_length)

    for _k in range(max_iter):
        if yc < yd:  # yc > yd to find the maximum
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            query = (1.0 - c) * edge_a + c * edge_b
            yc = sdf_plane(query, plane_width, plane_length)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            query = (1.0 - d) * edge_a + d * edge_b
            yd = sdf_plane(query, plane_width, plane_length)

    if yc < yd:
        return 0.5 * (a + d)
    return 0.5 * (c + b)


@wp.func
def closest_edge_coordinate_capsule(radius: float, half_height: float, edge_a: wp.vec3, edge_b: wp.vec3, max_iter: int):
    # find point on edge closest to capsule, return its barycentric edge coordinate
    # Golden-section search
    a = float(0.0)
    b = float(1.0)
    h = b - a
    invphi = 0.61803398875  # 1 / phi
    invphi2 = 0.38196601125  # 1 / phi^2
    c = a + invphi2 * h
    d = a + invphi * h
    query = (1.0 - c) * edge_a + c * edge_b
    yc = sdf_capsule(query, radius, half_height, int(Axis.Z))
    query = (1.0 - d) * edge_a + d * edge_b
    yd = sdf_capsule(query, radius, half_height, int(Axis.Z))

    for _k in range(max_iter):
        if yc < yd:  # yc > yd to find the maximum
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            query = (1.0 - c) * edge_a + c * edge_b
            yc = sdf_capsule(query, radius, half_height, int(Axis.Z))
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            query = (1.0 - d) * edge_a + d * edge_b
            yd = sdf_capsule(query, radius, half_height, int(Axis.Z))

    if yc < yd:
        return 0.5 * (a + d)

    return 0.5 * (c + b)


@wp.func
def closest_edge_coordinate_cylinder(
    radius: float, half_height: float, edge_a: wp.vec3, edge_b: wp.vec3, max_iter: int
):
    # find point on edge closest to cylinder, return its barycentric edge coordinate
    # Golden-section search
    a = float(0.0)
    b = float(1.0)
    h = b - a
    invphi = 0.61803398875  # 1 / phi
    invphi2 = 0.38196601125  # 1 / phi^2
    c = a + invphi2 * h
    d = a + invphi * h
    query = (1.0 - c) * edge_a + c * edge_b
    yc = sdf_cylinder(query, radius, half_height, int(Axis.Z))
    query = (1.0 - d) * edge_a + d * edge_b
    yd = sdf_cylinder(query, radius, half_height, int(Axis.Z))

    for _k in range(max_iter):
        if yc < yd:  # yc > yd to find the maximum
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            query = (1.0 - c) * edge_a + c * edge_b
            yc = sdf_cylinder(query, radius, half_height, int(Axis.Z))
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            query = (1.0 - d) * edge_a + d * edge_b
            yd = sdf_cylinder(query, radius, half_height, int(Axis.Z))

    if yc < yd:
        return 0.5 * (a + d)

    return 0.5 * (c + b)


@wp.func
def mesh_sdf(mesh: wp.uint64, point: wp.vec3, max_dist: float):
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    res = wp.mesh_query_point_sign_normal(mesh, point, max_dist, sign, face_index, face_u, face_v)

    if res:
        closest = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
        return wp.length(point - closest) * sign
    return max_dist


@wp.func
def sdf_mesh(mesh: wp.uint64, point: wp.vec3, max_dist: float):
    """Compute signed distance to a triangle mesh.

    Args:
        mesh: Warp mesh ID (``mesh.id``).
        point [m]: Query point in mesh local frame, shape [3], float.
        max_dist [m]: Maximum query distance.

    Returns:
        Signed distance [m], negative inside, zero on surface, positive outside.
    """
    return mesh_sdf(mesh, point, max_dist)


@wp.func
def closest_point_mesh(mesh: wp.uint64, point: wp.vec3, max_dist: float):
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    res = wp.mesh_query_point_sign_normal(mesh, point, max_dist, sign, face_index, face_u, face_v)

    if res:
        return wp.mesh_eval_position(mesh, face_index, face_u, face_v)
    # return arbitrary point from mesh
    return wp.mesh_eval_position(mesh, 0, 0.0, 0.0)


@wp.func
def closest_edge_coordinate_mesh(mesh: wp.uint64, edge_a: wp.vec3, edge_b: wp.vec3, max_iter: int, max_dist: float):
    # find point on edge closest to mesh, return its barycentric edge coordinate
    # Golden-section search
    a = float(0.0)
    b = float(1.0)
    h = b - a
    invphi = 0.61803398875  # 1 / phi
    invphi2 = 0.38196601125  # 1 / phi^2
    c = a + invphi2 * h
    d = a + invphi * h
    query = (1.0 - c) * edge_a + c * edge_b
    yc = mesh_sdf(mesh, query, max_dist)
    query = (1.0 - d) * edge_a + d * edge_b
    yd = mesh_sdf(mesh, query, max_dist)

    for _k in range(max_iter):
        if yc < yd:  # yc > yd to find the maximum
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            query = (1.0 - c) * edge_a + c * edge_b
            yc = mesh_sdf(mesh, query, max_dist)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            query = (1.0 - d) * edge_a + d * edge_b
            yd = mesh_sdf(mesh, query, max_dist)

    if yc < yd:
        return 0.5 * (a + d)
    return 0.5 * (c + b)


@wp.func
def volume_grad(volume: wp.uint64, p: wp.vec3):
    eps = 0.05  # TODO make this a parameter
    q = wp.volume_world_to_index(volume, p)

    # compute gradient of the SDF using finite differences
    dx = wp.volume_sample_f(volume, q + wp.vec3(eps, 0.0, 0.0), wp.Volume.LINEAR) - wp.volume_sample_f(
        volume, q - wp.vec3(eps, 0.0, 0.0), wp.Volume.LINEAR
    )
    dy = wp.volume_sample_f(volume, q + wp.vec3(0.0, eps, 0.0), wp.Volume.LINEAR) - wp.volume_sample_f(
        volume, q - wp.vec3(0.0, eps, 0.0), wp.Volume.LINEAR
    )
    dz = wp.volume_sample_f(volume, q + wp.vec3(0.0, 0.0, eps), wp.Volume.LINEAR) - wp.volume_sample_f(
        volume, q - wp.vec3(0.0, 0.0, eps), wp.Volume.LINEAR
    )

    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def counter_increment(
    counter: wp.array(dtype=int), counter_index: int, tids: wp.array(dtype=int), tid: int, index_limit: int = -1
):
    """
    Increment the counter but only if it is smaller than index_limit, remember which thread received which counter value.
    This allows the counter increment function to be used in differentiable computations where the backward pass will
    be able to leverage the thread-local counter values.

    If ``index_limit`` is less than zero, the counter is incremented without any limit.

    Args:
        counter: The counter array.
        counter_index: The index of the counter to increment.
        tids: The array to store the thread-local counter values.
        tid: The thread index.
        index_limit: The limit of the counter (optional, default is -1).
    """
    count = wp.atomic_add(counter, counter_index, 1)
    if count < index_limit or index_limit < 0:
        tids[tid] = count
        return count
    tids[tid] = -1
    return -1


@wp.func_replay(counter_increment)
def counter_increment_replay(
    counter: wp.array(dtype=int), counter_index: int, tids: wp.array(dtype=int), tid: int, index_limit: int
):
    return tids[tid]


@wp.kernel
def create_soft_contacts(
    particle_q: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_world: wp.array(dtype=int),  # World indices for particles
    body_q: wp.array(dtype=wp.transform),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_source_ptr: wp.array(dtype=wp.uint64),
    shape_world: wp.array(dtype=int),  # World indices for shapes
    margin: float,
    soft_contact_max: int,
    shape_count: int,
    shape_flags: wp.array(dtype=wp.int32),
    # outputs
    soft_contact_count: wp.array(dtype=int),
    soft_contact_particle: wp.array(dtype=int),
    soft_contact_shape: wp.array(dtype=int),
    soft_contact_body_pos: wp.array(dtype=wp.vec3),
    soft_contact_body_vel: wp.array(dtype=wp.vec3),
    soft_contact_normal: wp.array(dtype=wp.vec3),
    soft_contact_tids: wp.array(dtype=int),
):
    tid = wp.tid()
    particle_index, shape_index = tid // shape_count, tid % shape_count
    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return
    if (shape_flags[shape_index] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return

    # Check world indices
    particle_world_id = particle_world[particle_index]
    shape_world_id = shape_world[shape_index]

    # Skip collision between different worlds (unless one is global)
    if particle_world_id != -1 and shape_world_id != -1 and particle_world_id != shape_world_id:
        return

    rigid_index = shape_body[shape_index]

    px = particle_q[particle_index]
    radius = particle_radius[particle_index]

    X_wb = wp.transform_identity()
    if rigid_index >= 0:
        X_wb = body_q[rigid_index]

    X_bs = shape_transform[shape_index]

    X_ws = wp.transform_multiply(X_wb, X_bs)
    X_sw = wp.transform_inverse(X_ws)

    # transform particle position to shape local space
    x_local = wp.transform_point(X_sw, px)

    # geo description
    geo_type = shape_type[shape_index]
    geo_scale = shape_scale[shape_index]

    # evaluate shape sdf
    d = 1.0e6
    n = wp.vec3()
    v = wp.vec3()

    if geo_type == GeoType.SPHERE:
        d = sdf_sphere(x_local, geo_scale[0])
        n = sdf_sphere_grad(x_local, geo_scale[0])

    if geo_type == GeoType.BOX:
        d = sdf_box(x_local, geo_scale[0], geo_scale[1], geo_scale[2])
        n = sdf_box_grad(x_local, geo_scale[0], geo_scale[1], geo_scale[2])

    if geo_type == GeoType.CAPSULE:
        d = sdf_capsule(x_local, geo_scale[0], geo_scale[1], int(Axis.Z))
        n = sdf_capsule_grad(x_local, geo_scale[0], geo_scale[1], int(Axis.Z))

    if geo_type == GeoType.CYLINDER:
        d = sdf_cylinder(x_local, geo_scale[0], geo_scale[1], int(Axis.Z))
        n = sdf_cylinder_grad(x_local, geo_scale[0], geo_scale[1], int(Axis.Z))

    if geo_type == GeoType.CONE:
        d = sdf_cone(x_local, geo_scale[0], geo_scale[1], int(Axis.Z))
        n = sdf_cone_grad(x_local, geo_scale[0], geo_scale[1], int(Axis.Z))

    if geo_type == GeoType.ELLIPSOID:
        d = sdf_ellipsoid(x_local, geo_scale)
        n = sdf_ellipsoid_grad(x_local, geo_scale)

    if geo_type == GeoType.MESH or geo_type == GeoType.CONVEX_MESH:
        mesh = shape_source_ptr[shape_index]

        face_index = int(0)
        face_u = float(0.0)
        face_v = float(0.0)
        sign = float(0.0)

        min_scale = wp.min(geo_scale)
        if wp.mesh_query_point_sign_normal(
            mesh, wp.cw_div(x_local, geo_scale), margin + radius / min_scale, sign, face_index, face_u, face_v
        ):
            shape_p = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
            shape_v = wp.mesh_eval_velocity(mesh, face_index, face_u, face_v)

            shape_p = wp.cw_mul(shape_p, geo_scale)
            shape_v = wp.cw_mul(shape_v, geo_scale)

            delta = x_local - shape_p

            d = wp.length(delta) * sign
            n = wp.normalize(delta) * sign
            v = shape_v

    if geo_type == GeoType.SDF:
        volume = shape_source_ptr[shape_index]
        xpred_local = wp.volume_world_to_index(volume, wp.cw_div(x_local, geo_scale))
        nn = wp.vec3(0.0, 0.0, 0.0)
        d = wp.volume_sample_grad_f(volume, xpred_local, wp.Volume.LINEAR, nn)
        n = wp.normalize(nn)

    if geo_type == GeoType.PLANE:
        d = sdf_plane(x_local, geo_scale[0] * 0.5, geo_scale[1] * 0.5)
        n = wp.vec3(0.0, 0.0, 1.0)

    if d < margin + radius:
        index = counter_increment(soft_contact_count, 0, soft_contact_tids, tid)

        if index < soft_contact_max:
            # compute contact point in body local space
            body_pos = wp.transform_point(X_bs, x_local - n * d)
            body_vel = wp.transform_vector(X_bs, v)

            world_normal = wp.transform_vector(X_ws, n)

            soft_contact_shape[index] = shape_index
            soft_contact_body_pos[index] = body_pos
            soft_contact_body_vel[index] = body_vel
            soft_contact_particle[index] = particle_index
            soft_contact_normal[index] = world_normal


# --------------------------------------
# region Triangle collision detection

# types of triangle's closest point to a point
TRI_CONTACT_FEATURE_VERTEX_A = wp.constant(0)
TRI_CONTACT_FEATURE_VERTEX_B = wp.constant(1)
TRI_CONTACT_FEATURE_VERTEX_C = wp.constant(2)
TRI_CONTACT_FEATURE_EDGE_AB = wp.constant(3)
TRI_CONTACT_FEATURE_EDGE_AC = wp.constant(4)
TRI_CONTACT_FEATURE_EDGE_BC = wp.constant(5)
TRI_CONTACT_FEATURE_FACE_INTERIOR = wp.constant(6)

# constants used to access TriMeshCollisionDetector.resize_flags
VERTEX_COLLISION_BUFFER_OVERFLOW_INDEX = wp.constant(0)
TRI_COLLISION_BUFFER_OVERFLOW_INDEX = wp.constant(1)
EDGE_COLLISION_BUFFER_OVERFLOW_INDEX = wp.constant(2)
TRI_TRI_COLLISION_BUFFER_OVERFLOW_INDEX = wp.constant(3)


@wp.func
def compute_tri_aabb(
    v1: wp.vec3,
    v2: wp.vec3,
    v3: wp.vec3,
):
    lower = wp.min(wp.min(v1, v2), v3)
    upper = wp.max(wp.max(v1, v2), v3)

    return lower, upper


@wp.kernel
def compute_tri_aabbs(
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    lower_bounds: wp.array(dtype=wp.vec3),
    upper_bounds: wp.array(dtype=wp.vec3),
):
    t_id = wp.tid()

    v1 = pos[tri_indices[t_id, 0]]
    v2 = pos[tri_indices[t_id, 1]]
    v3 = pos[tri_indices[t_id, 2]]

    lower, upper = compute_tri_aabb(v1, v2, v3)

    lower_bounds[t_id] = lower
    upper_bounds[t_id] = upper


@wp.kernel
def compute_edge_aabbs(
    pos: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    lower_bounds: wp.array(dtype=wp.vec3),
    upper_bounds: wp.array(dtype=wp.vec3),
):
    e_id = wp.tid()

    v1 = pos[edge_indices[e_id, 2]]
    v2 = pos[edge_indices[e_id, 3]]

    lower_bounds[e_id] = wp.min(v1, v2)
    upper_bounds[e_id] = wp.max(v1, v2)


@wp.func
def tri_is_neighbor(a_1: wp.int32, a_2: wp.int32, a_3: wp.int32, b_1: wp.int32, b_2: wp.int32, b_3: wp.int32):
    tri_is_neighbor = (
        a_1 == b_1
        or a_1 == b_2
        or a_1 == b_3
        or a_2 == b_1
        or a_2 == b_2
        or a_2 == b_3
        or a_3 == b_1
        or a_3 == b_2
        or a_3 == b_3
    )

    return tri_is_neighbor


@wp.func
def vertex_adjacent_to_triangle(v: wp.int32, a: wp.int32, b: wp.int32, c: wp.int32):
    return v == a or v == b or v == c


@wp.kernel
def init_triangle_collision_data_kernel(
    query_radius: float,
    # outputs
    triangle_colliding_vertices_count: wp.array(dtype=wp.int32),
    triangle_colliding_vertices_min_dist: wp.array(dtype=float),
    resize_flags: wp.array(dtype=wp.int32),
):
    tri_index = wp.tid()

    triangle_colliding_vertices_count[tri_index] = 0
    triangle_colliding_vertices_min_dist[tri_index] = query_radius

    if tri_index == 0:
        for i in range(4):
            resize_flags[i] = 0


@wp.kernel
def vertex_triangle_collision_detection_kernel(
    max_query_radius: float,
    min_query_radius: float,
    bvh_id: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    vertex_colliding_triangles_offsets: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_buffer_sizes: wp.array(dtype=wp.int32),
    triangle_colliding_vertices_offsets: wp.array(dtype=wp.int32),
    triangle_colliding_vertices_buffer_sizes: wp.array(dtype=wp.int32),
    vertex_triangle_filtering_list: wp.array(dtype=wp.int32),
    vertex_triangle_filtering_list_offsets: wp.array(dtype=wp.int32),
    min_distance_filtering_ref_pos: wp.array(dtype=wp.vec3),
    # outputs
    vertex_colliding_triangles: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_count: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_min_dist: wp.array(dtype=float),
    triangle_colliding_vertices: wp.array(dtype=wp.int32),
    triangle_colliding_vertices_count: wp.array(dtype=wp.int32),
    triangle_colliding_vertices_min_dist: wp.array(dtype=float),
    resize_flags: wp.array(dtype=wp.int32),
):
    """
    This function applies discrete collision detection between vertices and triangles. It uses pre-allocated spaces to
    record the collision data. This collision detector works both ways, i.e., it records vertices' colliding triangles to
    `vertex_colliding_triangles`, and records each triangles colliding vertices to `triangle_colliding_vertices`.

    This function assumes that all the vertices are on triangles, and can be indexed from the pos argument.

    Note:

        The collision data buffer is pre-allocated and cannot be changed during collision detection, therefore, the space
        may not be enough. If the space is not enough to record all the collision information, the function will set a
        certain element in resized_flag to be true. The user can reallocate the buffer based on vertex_colliding_triangles_count
        and vertex_colliding_triangles_count.

    Args:
        bvh_id (int): the bvh id you want to collide with
        max_query_radius (float): the upper bound of collision distance.
        min_query_radius (float): the lower bound of collision distance. This distance is evaluated based on min_distance_filtering_ref_pos
        pos (array): positions of all the vertices that make up triangles
        vertex_colliding_triangles_offsets (array): where each vertex' collision buffer starts
        vertex_colliding_triangles_buffer_sizes (array): size of each vertex' collision buffer, will be modified if resizing is needed
        vertex_colliding_triangles_min_dist (array): each vertex' min distance to all (non-neighbor) triangles
        triangle_colliding_vertices_offsets (array): where each triangle's collision buffer starts
        triangle_colliding_vertices_buffer_sizes (array): size of each triangle's collision buffer, will be modified if resizing is needed
        min_distance_filtering_ref_pos (array): the position that minimal collision distance evaluation uses.
        vertex_colliding_triangles (array): flattened buffer of vertices' collision triangles
        vertex_colliding_triangles_count (array): number of triangles each vertex collides with
        triangle_colliding_vertices (array): positions of all the triangles' collision vertices, every two elements
            records the vertex index and a triangle index it collides to
        triangle_colliding_vertices_count (array): number of triangles each vertex collides with
        triangle_colliding_vertices_min_dist (array): each triangle's min distance to all (non-self) vertices
        resized_flag (array): size == 3, (vertex_buffer_resize_required, triangle_buffer_resize_required, edge_buffer_resize_required)
    """

    v_index = wp.tid()
    v = pos[v_index]
    vertex_buffer_offset = vertex_colliding_triangles_offsets[v_index]
    vertex_buffer_size = vertex_colliding_triangles_offsets[v_index + 1] - vertex_buffer_offset

    lower = wp.vec3(v[0] - max_query_radius, v[1] - max_query_radius, v[2] - max_query_radius)
    upper = wp.vec3(v[0] + max_query_radius, v[1] + max_query_radius, v[2] + max_query_radius)

    query = wp.bvh_query_aabb(bvh_id, lower, upper)

    tri_index = wp.int32(0)
    vertex_num_collisions = wp.int32(0)
    min_dis_to_tris = max_query_radius
    while wp.bvh_query_next(query, tri_index):
        t1 = tri_indices[tri_index, 0]
        t2 = tri_indices[tri_index, 1]
        t3 = tri_indices[tri_index, 2]

        if vertex_adjacent_to_triangle(v_index, t1, t2, t3):
            continue

        if vertex_triangle_filtering_list:
            fl_start = vertex_triangle_filtering_list_offsets[v_index]
            fl_end = vertex_triangle_filtering_list_offsets[v_index + 1]  # start of next vertex slice (end exclusive)

            if fl_end > fl_start:
                # Optional fast-fail using first/last elements (remember end is exclusive)
                first_val = vertex_triangle_filtering_list[fl_start]
                last_val = vertex_triangle_filtering_list[fl_end - 1]
                if (tri_index >= first_val) and (tri_index <= last_val):
                    idx = binary_search(vertex_triangle_filtering_list, tri_index, fl_start, fl_end)
                    # `idx` is the first index > tri_index within [fl_start, fl_end)
                    if idx > fl_start and vertex_triangle_filtering_list[idx - 1] == tri_index:
                        continue

        u1 = pos[t1]
        u2 = pos[t2]
        u3 = pos[t3]

        closest_p, _bary, _feature_type = triangle_closest_point(u1, u2, u3, v)

        dist = wp.length(closest_p - v)

        if min_distance_filtering_ref_pos and min_query_radius > 0.0:
            closest_p_ref, _, __ = triangle_closest_point(
                min_distance_filtering_ref_pos[t1],
                min_distance_filtering_ref_pos[t2],
                min_distance_filtering_ref_pos[t3],
                min_distance_filtering_ref_pos[v_index],
            )
            dist_ref = wp.length(closest_p_ref - min_distance_filtering_ref_pos[v_index])

            if dist_ref < min_query_radius:
                continue

        if dist < max_query_radius:
            # record v-f collision to vertex
            min_dis_to_tris = wp.min(min_dis_to_tris, dist)
            if vertex_num_collisions < vertex_buffer_size:
                vertex_colliding_triangles[2 * (vertex_buffer_offset + vertex_num_collisions)] = v_index
                vertex_colliding_triangles[2 * (vertex_buffer_offset + vertex_num_collisions) + 1] = tri_index
            else:
                resize_flags[VERTEX_COLLISION_BUFFER_OVERFLOW_INDEX] = 1

            vertex_num_collisions = vertex_num_collisions + 1

            if triangle_colliding_vertices:
                wp.atomic_min(triangle_colliding_vertices_min_dist, tri_index, dist)
                tri_buffer_size = triangle_colliding_vertices_buffer_sizes[tri_index]
                tri_num_collisions = wp.atomic_add(triangle_colliding_vertices_count, tri_index, 1)

                if tri_num_collisions < tri_buffer_size:
                    tri_buffer_offset = triangle_colliding_vertices_offsets[tri_index]
                    # record v-f collision to triangle
                    triangle_colliding_vertices[tri_buffer_offset + tri_num_collisions] = v_index
                else:
                    resize_flags[TRI_COLLISION_BUFFER_OVERFLOW_INDEX] = 1

    vertex_colliding_triangles_count[v_index] = vertex_num_collisions
    vertex_colliding_triangles_min_dist[v_index] = min_dis_to_tris


@wp.kernel
def edge_colliding_edges_detection_kernel(
    max_query_radius: float,
    min_query_radius: float,
    bvh_id: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_colliding_edges_offsets: wp.array(dtype=wp.int32),
    edge_colliding_edges_buffer_sizes: wp.array(dtype=wp.int32),
    edge_edge_parallel_epsilon: float,
    edge_filtering_list: wp.array(dtype=wp.int32),
    edge_filtering_list_offsets: wp.array(dtype=wp.int32),
    min_distance_filtering_ref_pos: wp.array(dtype=wp.vec3),
    # outputs
    edge_colliding_edges: wp.array(dtype=wp.int32),
    edge_colliding_edges_count: wp.array(dtype=wp.int32),
    edge_colliding_edges_min_dist: wp.array(dtype=float),
    resize_flags: wp.array(dtype=wp.int32),
):
    """
    bvh_id (int): the bvh id you want to do collision detection on
    max_query_radius (float): the upper bound of collision distance.
    min_query_radius (float): the lower bound of collision distance. This distance is evaluated based on min_distance_filtering_ref_pos
    pos (array): positions of all the vertices that make up edges
    edge_colliding_triangles (array): flattened buffer of edges' collision edges
    edge_colliding_edges_count (array): number of edges each edge collides
    edge_colliding_triangles_offsets (array): where each edge's collision buffer starts
    edge_colliding_triangles_buffer_size (array): size of each edge's collision buffer, will be modified if resizing is needed
    edge_min_dis_to_triangles (array): each vertex' min distance to all (non-neighbor) triangles
    resized_flag (array): size == 3, (vertex_buffer_resize_required, triangle_buffer_resize_required, edge_buffer_resize_required)
    """
    e_index = wp.tid()

    e0_v0 = edge_indices[e_index, 2]
    e0_v1 = edge_indices[e_index, 3]

    e0_v0_pos = pos[e0_v0]
    e0_v1_pos = pos[e0_v1]

    lower = wp.min(e0_v0_pos, e0_v1_pos)
    upper = wp.max(e0_v0_pos, e0_v1_pos)

    lower = wp.vec3(lower[0] - max_query_radius, lower[1] - max_query_radius, lower[2] - max_query_radius)
    upper = wp.vec3(upper[0] + max_query_radius, upper[1] + max_query_radius, upper[2] + max_query_radius)

    query = wp.bvh_query_aabb(bvh_id, lower, upper)

    colliding_edge_index = wp.int32(0)
    edge_num_collisions = wp.int32(0)
    min_dis_to_edges = max_query_radius
    while wp.bvh_query_next(query, colliding_edge_index):
        e1_v0 = edge_indices[colliding_edge_index, 2]
        e1_v1 = edge_indices[colliding_edge_index, 3]

        if e0_v0 == e1_v0 or e0_v0 == e1_v1 or e0_v1 == e1_v0 or e0_v1 == e1_v1:
            continue

        if edge_filtering_list:
            fl_start = edge_filtering_list_offsets[e_index]
            fl_end = edge_filtering_list_offsets[e_index + 1]  # start of next vertex slice (end exclusive)

            if fl_end > fl_start:
                # Optional fast-fail using first/last elements (remember end is exclusive)
                first_val = edge_filtering_list[fl_start]
                last_val = edge_filtering_list[fl_end - 1]
                if (colliding_edge_index >= first_val) and (colliding_edge_index <= last_val):
                    idx = binary_search(edge_filtering_list, colliding_edge_index, fl_start, fl_end)
                    if idx > fl_start and edge_filtering_list[idx - 1] == colliding_edge_index:
                        continue
                # else: key is out of range, cannot be present -> skip_this remains False

        e1_v0_pos = pos[e1_v0]
        e1_v1_pos = pos[e1_v1]

        std = wp.closest_point_edge_edge(e0_v0_pos, e0_v1_pos, e1_v0_pos, e1_v1_pos, edge_edge_parallel_epsilon)
        dist = std[2]

        if min_distance_filtering_ref_pos and min_query_radius > 0.0:
            e0_v0_pos_ref, e0_v1_pos_ref, e1_v0_pos_ref, e1_v1_pos_ref = (
                min_distance_filtering_ref_pos[e0_v0],
                min_distance_filtering_ref_pos[e0_v1],
                min_distance_filtering_ref_pos[e1_v0],
                min_distance_filtering_ref_pos[e1_v1],
            )
            std_ref = wp.closest_point_edge_edge(
                e0_v0_pos_ref, e0_v1_pos_ref, e1_v0_pos_ref, e1_v1_pos_ref, edge_edge_parallel_epsilon
            )

            dist_ref = std_ref[2]
            if dist_ref < min_query_radius:
                continue

        if dist < max_query_radius:
            edge_buffer_offset = edge_colliding_edges_offsets[e_index]
            edge_buffer_size = edge_colliding_edges_offsets[e_index + 1] - edge_buffer_offset

            # record e-e collision to e0, and leave e1; e1 will detect this collision from its own thread
            min_dis_to_edges = wp.min(min_dis_to_edges, dist)
            if edge_num_collisions < edge_buffer_size:
                edge_colliding_edges[2 * (edge_buffer_offset + edge_num_collisions)] = e_index
                edge_colliding_edges[2 * (edge_buffer_offset + edge_num_collisions) + 1] = colliding_edge_index
            else:
                resize_flags[EDGE_COLLISION_BUFFER_OVERFLOW_INDEX] = 1

            edge_num_collisions = edge_num_collisions + 1

    edge_colliding_edges_count[e_index] = edge_num_collisions
    edge_colliding_edges_min_dist[e_index] = min_dis_to_edges


@wp.kernel
def triangle_triangle_collision_detection_kernel(
    bvh_id: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    triangle_intersecting_triangles_offsets: wp.array(dtype=wp.int32),
    # outputs
    triangle_intersecting_triangles: wp.array(dtype=wp.int32),
    triangle_intersecting_triangles_count: wp.array(dtype=wp.int32),
    resize_flags: wp.array(dtype=wp.int32),
):
    tri_index = wp.tid()
    t1_v1 = tri_indices[tri_index, 0]
    t1_v2 = tri_indices[tri_index, 1]
    t1_v3 = tri_indices[tri_index, 2]

    v1 = pos[t1_v1]
    v2 = pos[t1_v2]
    v3 = pos[t1_v3]

    lower, upper = compute_tri_aabb(v1, v2, v3)

    buffer_offset = triangle_intersecting_triangles_offsets[tri_index]
    buffer_size = triangle_intersecting_triangles_offsets[tri_index + 1] - buffer_offset

    query = wp.bvh_query_aabb(bvh_id, lower, upper)
    tri_index_2 = wp.int32(0)
    intersection_count = wp.int32(0)
    while wp.bvh_query_next(query, tri_index_2):
        t2_v1 = tri_indices[tri_index_2, 0]
        t2_v2 = tri_indices[tri_index_2, 1]
        t2_v3 = tri_indices[tri_index_2, 2]

        # filter out intersection test with neighbor triangles
        if (
            vertex_adjacent_to_triangle(t1_v1, t2_v1, t2_v2, t2_v3)
            or vertex_adjacent_to_triangle(t1_v2, t2_v1, t2_v2, t2_v3)
            or vertex_adjacent_to_triangle(t1_v3, t2_v1, t2_v2, t2_v3)
        ):
            continue

        u1 = pos[t2_v1]
        u2 = pos[t2_v2]
        u3 = pos[t2_v3]

        if wp.intersect_tri_tri(v1, v2, v3, u1, u2, u3):
            if intersection_count < buffer_size:
                triangle_intersecting_triangles[buffer_offset + intersection_count] = tri_index_2
            else:
                resize_flags[TRI_TRI_COLLISION_BUFFER_OVERFLOW_INDEX] = 1
            intersection_count = intersection_count + 1

    triangle_intersecting_triangles_count[tri_index] = intersection_count


# endregion
