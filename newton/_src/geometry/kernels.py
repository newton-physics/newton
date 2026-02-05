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

from newton._src.core.types import MAXVAL

from . import collision_primitive as primitive
from .broad_phase_common import binary_search
from .flags import ParticleFlags, ShapeFlags
from .types import (
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
def sphere_sdf(center: wp.vec3, radius: float, p: wp.vec3):
    return wp.length(p - center) - radius


@wp.func
def sphere_sdf_grad(center: wp.vec3, radius: float, p: wp.vec3):
    return wp.normalize(p - center)


@wp.func
def box_sdf(upper: wp.vec3, p: wp.vec3):
    # adapted from https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
    qx = abs(p[0]) - upper[0]
    qy = abs(p[1]) - upper[1]
    qz = abs(p[2]) - upper[2]

    e = wp.vec3(wp.max(qx, 0.0), wp.max(qy, 0.0), wp.max(qz, 0.0))

    return wp.length(e) + wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)


@wp.func
def box_sdf_grad(upper: wp.vec3, p: wp.vec3):
    qx = abs(p[0]) - upper[0]
    qy = abs(p[1]) - upper[1]
    qz = abs(p[2]) - upper[2]

    # exterior case
    if qx > 0.0 or qy > 0.0 or qz > 0.0:
        x = wp.clamp(p[0], -upper[0], upper[0])
        y = wp.clamp(p[1], -upper[1], upper[1])
        z = wp.clamp(p[2], -upper[2], upper[2])

        return wp.normalize(p - wp.vec3(x, y, z))

    sx = wp.sign(p[0])
    sy = wp.sign(p[1])
    sz = wp.sign(p[2])

    # x projection
    if (qx > qy and qx > qz) or (qy == 0.0 and qz == 0.0):
        return wp.vec3(sx, 0.0, 0.0)

    # y projection
    if (qy > qx and qy > qz) or (qx == 0.0 and qz == 0.0):
        return wp.vec3(0.0, sy, 0.0)

    # z projection
    return wp.vec3(0.0, 0.0, sz)


@wp.func
def capsule_sdf(radius: float, half_height: float, p: wp.vec3):
    if p[2] > half_height:
        return wp.length(wp.vec3(p[0], p[1], p[2] - half_height)) - radius

    if p[2] < -half_height:
        return wp.length(wp.vec3(p[0], p[1], p[2] + half_height)) - radius

    return wp.length(wp.vec3(p[0], p[1], 0.0)) - radius


@wp.func
def capsule_sdf_grad(radius: float, half_height: float, p: wp.vec3):
    if p[2] > half_height:
        return wp.normalize(wp.vec3(p[0], p[1], p[2] - half_height))

    if p[2] < -half_height:
        return wp.normalize(wp.vec3(p[0], p[1], p[2] + half_height))

    return wp.normalize(wp.vec3(p[0], p[1], 0.0))


@wp.func
def cylinder_sdf(radius: float, half_height: float, p: wp.vec3):
    dx = wp.length(wp.vec3(p[0], p[1], 0.0)) - radius
    dy = wp.abs(p[2]) - half_height
    return wp.min(wp.max(dx, dy), 0.0) + wp.length(wp.vec2(wp.max(dx, 0.0), wp.max(dy, 0.0)))


@wp.func
def cylinder_sdf_grad(radius: float, half_height: float, p: wp.vec3):
    dx = wp.length(wp.vec3(p[0], p[1], 0.0)) - radius
    dy = wp.abs(p[2]) - half_height
    if dx > dy:
        return wp.normalize(wp.vec3(p[0], p[1], 0.0))
    return wp.vec3(0.0, 0.0, wp.sign(p[2]))


@wp.func
def ellipsoid_sdf(radii: wp.vec3, p: wp.vec3):
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
    q0 = wp.cw_mul(p, inv_r)  # p / r
    q1 = wp.cw_mul(p, inv_r2)  # p / r^2
    k0 = wp.length(q0)
    k1 = wp.length(q1)
    if k1 > eps:
        return k0 * (k0 - 1.0) / k1
    # Deep inside / near center fallback
    return -wp.min(wp.min(r[0], r[1]), r[2])


@wp.func
def ellipsoid_sdf_grad(radii: wp.vec3, p: wp.vec3):
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
    q0 = wp.cw_mul(p, inv_r)  # p / r
    q1 = wp.cw_mul(p, inv_r2)  # p / r^2
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
def cone_sdf(radius: float, half_height: float, p: wp.vec3):
    # Cone with apex at +half_height and base at -half_height
    dx = wp.length(wp.vec3(p[0], p[1], 0.0)) - radius * (half_height - p[2]) / (2.0 * half_height)
    dy = wp.abs(p[2]) - half_height
    return wp.min(wp.max(dx, dy), 0.0) + wp.length(wp.vec2(wp.max(dx, 0.0), wp.max(dy, 0.0)))


@wp.func
def cone_sdf_grad(radius: float, half_height: float, p: wp.vec3):
    # Gradient for cone with apex at +half_height and base at -half_height
    r = wp.length(wp.vec3(p[0], p[1], 0.0))
    dx = r - radius * (half_height - p[2]) / (2.0 * half_height)
    dy = wp.abs(p[2]) - half_height
    if dx > dy:
        # Closest to lateral surface
        if r > 0.0:
            radial_dir = wp.vec3(p[0], p[1], 0.0) / r
            # Normal to cone surface
            return wp.normalize(radial_dir + wp.vec3(0.0, 0.0, radius / (2.0 * half_height)))
        else:
            return wp.vec3(0.0, 0.0, 1.0)
    else:
        # Closest to cap
        return wp.vec3(0.0, 0.0, wp.sign(p[2]))


@wp.func
def plane_sdf(width: float, length: float, p: wp.vec3):
    # SDF for a quad in the xy plane
    if width > 0.0 and length > 0.0:
        d = wp.max(wp.abs(p[0]) - width, wp.abs(p[1]) - length)
        return wp.max(d, wp.abs(p[2]))
    return p[2]


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
    yc = box_sdf(upper, query)
    query = (1.0 - d) * edge_a + d * edge_b
    yd = box_sdf(upper, query)

    for _k in range(max_iter):
        if yc < yd:  # yc > yd to find the maximum
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            query = (1.0 - c) * edge_a + c * edge_b
            yc = box_sdf(upper, query)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            query = (1.0 - d) * edge_a + d * edge_b
            yd = box_sdf(upper, query)

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
    yc = plane_sdf(plane_width, plane_length, query)
    query = (1.0 - d) * edge_a + d * edge_b
    yd = plane_sdf(plane_width, plane_length, query)

    for _k in range(max_iter):
        if yc < yd:  # yc > yd to find the maximum
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            query = (1.0 - c) * edge_a + c * edge_b
            yc = plane_sdf(plane_width, plane_length, query)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            query = (1.0 - d) * edge_a + d * edge_b
            yd = plane_sdf(plane_width, plane_length, query)

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
    yc = capsule_sdf(radius, half_height, query)
    query = (1.0 - d) * edge_a + d * edge_b
    yd = capsule_sdf(radius, half_height, query)

    for _k in range(max_iter):
        if yc < yd:  # yc > yd to find the maximum
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            query = (1.0 - c) * edge_a + c * edge_b
            yc = capsule_sdf(radius, half_height, query)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            query = (1.0 - d) * edge_a + d * edge_b
            yd = capsule_sdf(radius, half_height, query)

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
    yc = cylinder_sdf(radius, half_height, query)
    query = (1.0 - d) * edge_a + d * edge_b
    yd = cylinder_sdf(radius, half_height, query)

    for _k in range(max_iter):
        if yc < yd:  # yc > yd to find the maximum
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            query = (1.0 - c) * edge_a + c * edge_b
            yc = cylinder_sdf(radius, half_height, query)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            query = (1.0 - d) * edge_a + d * edge_b
            yd = cylinder_sdf(radius, half_height, query)

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
        d = sphere_sdf(wp.vec3(), geo_scale[0], x_local)
        n = sphere_sdf_grad(wp.vec3(), geo_scale[0], x_local)

    if geo_type == GeoType.BOX:
        d = box_sdf(geo_scale, x_local)
        n = box_sdf_grad(geo_scale, x_local)

    if geo_type == GeoType.CAPSULE:
        d = capsule_sdf(geo_scale[0], geo_scale[1], x_local)
        n = capsule_sdf_grad(geo_scale[0], geo_scale[1], x_local)

    if geo_type == GeoType.CYLINDER:
        d = cylinder_sdf(geo_scale[0], geo_scale[1], x_local)
        n = cylinder_sdf_grad(geo_scale[0], geo_scale[1], x_local)

    if geo_type == GeoType.CONE:
        d = cone_sdf(geo_scale[0], geo_scale[1], x_local)
        n = cone_sdf_grad(geo_scale[0], geo_scale[1], x_local)

    if geo_type == GeoType.ELLIPSOID:
        d = ellipsoid_sdf(geo_scale, x_local)
        n = ellipsoid_sdf_grad(geo_scale, x_local)

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
        d = plane_sdf(geo_scale[0], geo_scale[1], x_local)
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


# region Rigid body collision detection


@wp.func
def count_contact_points_for_pair(
    shape_a: int,
    shape_b: int,
    type_a: int,
    type_b: int,
    shape_scale: wp.array(dtype=wp.vec3),
    shape_source_ptr: wp.array(dtype=wp.uint64),
) -> tuple[int, int]:
    """
    Count the number of potential contact points for a collision pair in both directions
    of the collision pair (collisions from A to B and from B to A).

    Inputs must be canonicalized such that the type of shape A is less than or equal to the type of shape B.

    Args:
        shape_a: First shape index
        shape_b: Second shape index
        type_a: First shape type
        type_b: Second shape type
        shape_scale: Shape scale
        shape_source_ptr: Shape source pointer

    Returns:
        tuple[int, int]: Number of contact points for collisions between A->B and B->A.
    """

    # PLANE against all other types (ordered by GeoType index)
    if type_a == GeoType.PLANE:
        scale_a = wp.vec3(0.0, 0.0, 0.0)
        if shape_a >= 0:
            scale_a = shape_scale[shape_a]
        if type_b == GeoType.PLANE:
            return 0, 0  # no plane-plane contacts
        if type_b == GeoType.SPHERE:
            return 1, 0
        if type_b == GeoType.CAPSULE:
            if scale_a[0] == 0.0 and scale_a[1] == 0.0:
                return 2, 0  # vertex-based collision for infinite plane
            return 2 + 4, 0  # vertex-based collision + plane edges
        if type_b == GeoType.CYLINDER:
            # infinite plane: support max primitive contacts (2 caps + 2 side) = 4
            return 4, 0
        if type_b == GeoType.BOX:
            # elif actual_type_b == GeoType.PLANE:
            if scale_a[0] == 0.0 and scale_a[1] == 0.0:
                return 8, 0  # vertex-based collision
            else:
                return 8 + 4, 0  # vertex-based collision + plane edges
        if type_b == GeoType.MESH or type_b == GeoType.CONVEX_MESH:
            mesh_b = wp.mesh_get(shape_source_ptr[shape_b])
            return mesh_b.points.shape[0], 0

    # SPHERE against all other types (always 1 contact)
    elif type_a == GeoType.SPHERE:
        return 1, 0

    # CAPSULE against all other types
    elif type_a == GeoType.CAPSULE:
        if type_b == GeoType.CAPSULE:
            return 2, 0
        if type_b == GeoType.BOX:
            return 8, 0
        if type_b == GeoType.MESH or type_b == GeoType.CONVEX_MESH:
            num_contacts_a = 2
            mesh_b = wp.mesh_get(shape_source_ptr[shape_b])
            num_contacts_b = mesh_b.points.shape[0]
            return num_contacts_a, num_contacts_b

    # CYLINDER against all other types
    elif type_a == GeoType.CYLINDER:
        # unsupported type combination
        return 0, 0

    # BOX against all other types
    elif type_a == GeoType.BOX:
        if type_b == GeoType.BOX:
            return 12, 12
        if type_b == GeoType.MESH or type_b == GeoType.CONVEX_MESH:
            num_contacts_a = 8
            mesh_b = wp.mesh_get(shape_source_ptr[shape_b])
            num_contacts_b = mesh_b.points.shape[0]
            return num_contacts_a, num_contacts_b

    # MESH against all other types
    elif type_a == GeoType.MESH or type_a == GeoType.CONVEX_MESH:
        mesh_a = wp.mesh_get(shape_source_ptr[shape_a])
        num_contacts_a = mesh_a.points.shape[0]
        if type_b == GeoType.MESH or type_b == GeoType.CONVEX_MESH:
            mesh_b = wp.mesh_get(shape_source_ptr[shape_b])
            num_contacts_b = mesh_b.points.shape[0]
            return num_contacts_a, num_contacts_b
        return 0, 0

    # unsupported type combination
    return 0, 0


# NOTE: Kernel is in a unique module to speed up cold-start ModelBuilder.finalize() time
@wp.kernel(enable_backward=False, module="unique")
def count_contact_points(
    contact_pairs: wp.array(dtype=wp.vec2i),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_source_ptr: wp.array(dtype=wp.uint64),
    max_per_pair: int,
    # outputs
    contact_count: wp.array(dtype=int),
):
    tid = wp.tid()

    shape_ab = contact_pairs[tid]
    shape_a = shape_ab[0]
    shape_b = shape_ab[1]

    if shape_a < 0:
        return

    type_a = shape_type[shape_a]
    if shape_b < 0:
        type_b = GeoType.PLANE
    else:
        type_b = shape_type[shape_b]
    # canonicalize the pair so that type_a <= type_b
    if type_a > type_b:
        shape_tmp = shape_a
        shape_a = shape_b
        shape_b = shape_tmp

        type_tmp = type_a
        type_a = type_b
        type_b = type_tmp

    # determine how many contact points need to be evaluated
    num_contacts_a, num_contacts_b = count_contact_points_for_pair(
        shape_a, shape_b, type_a, type_b, shape_scale, shape_source_ptr
    )
    num_contacts = num_contacts_a + num_contacts_b
    if max_per_pair > 0 and num_contacts > max_per_pair:
        num_contacts = max_per_pair
    wp.atomic_add(contact_count, 0, num_contacts)


@wp.func
def allocate_contact_points(
    num_contacts_a: int,
    num_contacts_b: int,
    shape_a: int,
    shape_b: int,
    rigid_contact_max: int,
    # outputs
    contact_count: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_point_id: wp.array(dtype=int),
) -> bool:
    """
    Allocate contact points for a collision pair.

    Args:
        num_contacts_a: Number of contacts to allocate from shape A->B
        num_contacts_b: Number of contacts to allocate from shape B->A
        shape_a: First shape index
        shape_b: Second shape index
        pair_index_ab: Pair index for shape A to B
        pair_index_ba: Pair index for shape B to A
        rigid_contact_max: Maximum number of rigid contacts allowed
        contact_count: Array to track total contact count
        contact_shape0: Array to store first shape indices
        contact_shape1: Array to store second shape indices
        contact_point_id: Array to store contact point IDs
        contact_point_limit: Array to store contact limits per pair

    Returns:
        bool: True if allocation succeeded, False if limit exceeded
    """
    num_contacts = num_contacts_a + num_contacts_b
    index = wp.atomic_add(contact_count, 0, num_contacts)
    new_end_index = index + num_contacts - 1
    if new_end_index >= rigid_contact_max:
        wp.printf(
            "Number of rigid contacts (%d) exceeded limit (%d). Increase Model.rigid_contact_max.\n",
            new_end_index + 1,
            rigid_contact_max,
        )
        return False

    # allocate contact points from shape A->B
    for i in range(num_contacts_a):
        cp_index = index + i
        contact_shape0[cp_index] = shape_a
        contact_shape1[cp_index] = shape_b
        contact_point_id[cp_index] = i

    # allocate contact points from shape B->A
    for i in range(num_contacts_b):
        cp_index = index + num_contacts_a + i
        contact_shape0[cp_index] = shape_b
        contact_shape1[cp_index] = shape_a
        contact_point_id[cp_index] = i

    return True


@wp.kernel(enable_backward=False)
def broadphase_collision_pairs(
    body_q: wp.array(dtype=wp.transform),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_source_ptr: wp.array(dtype=wp.uint64),
    shape_pairs_filtered: wp.array(dtype=wp.vec2i),
    shape_radius: wp.array(dtype=float),
    num_shapes: int,
    rigid_contact_max: int,
    shape_contact_margin: wp.array(dtype=float),
    max_per_pair: int,
    # outputs
    contact_count: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_point_id: wp.array(dtype=int),
    contact_point_limit: wp.array(dtype=int),
):
    """
    Broadphase collision detection based on bounding spheres testing.
    """
    tid = wp.tid()
    shape_ab = shape_pairs_filtered[tid]
    shape_a = shape_ab[0]
    shape_b = shape_ab[1]

    rigid_a = shape_body[shape_a]
    if rigid_a == -1:
        X_ws_a = shape_transform[shape_a]
    else:
        X_ws_a = body_q[rigid_a] * shape_transform[shape_a]
    rigid_b = shape_body[shape_b]
    if rigid_b == -1:
        X_ws_b = shape_transform[shape_b]
    else:
        X_ws_b = body_q[rigid_b] * shape_transform[shape_b]

    type_a = shape_type[shape_a]
    type_b = shape_type[shape_b]
    # ensure unique ordering of shape pairs
    if type_a > type_b:
        shape_tmp = shape_a
        shape_a = shape_b
        shape_b = shape_tmp

        type_tmp = type_a
        type_a = type_b
        type_b = type_tmp

        X_tmp = X_ws_a
        X_ws_a = X_ws_b
        X_ws_b = X_tmp

    p_b = wp.transform_get_translation(X_ws_b)
    r_b = shape_radius[shape_b]
    if type_a == GeoType.PLANE and type_b == GeoType.PLANE:
        return

    # Use per-shape contact margins (sum for consistency with thickness)
    margin = shape_contact_margin[shape_a] + shape_contact_margin[shape_b]

    # bounding sphere check
    if type_a == GeoType.PLANE:
        query_b = wp.transform_point(wp.transform_inverse(X_ws_a), p_b)
        scale = shape_scale[shape_a]
        closest = closest_point_plane(scale[0], scale[1], query_b)
        d = wp.length(query_b - closest)
        if d > r_b + margin:
            return
    else:
        p_a = wp.transform_get_translation(X_ws_a)
        d = wp.length(p_a - p_b)
        r_a = shape_radius[shape_a]
        r_b = shape_radius[shape_b]
        if d > r_a + r_b + margin:
            return

    pair_index_ab = shape_a * num_shapes + shape_b
    pair_index_ba = shape_b * num_shapes + shape_a

    num_contacts_a, num_contacts_b = count_contact_points_for_pair(
        shape_a, shape_b, type_a, type_b, shape_scale, shape_source_ptr
    )

    if contact_point_limit:
        # assign a limit per contact pair, if max_per_pair is set
        if max_per_pair > 0:
            # distribute maximum number of contact per pair in both directions
            max_per_pair_half = max_per_pair // 2
            if num_contacts_b > 0:
                contact_point_limit[pair_index_ab] = max_per_pair_half
                contact_point_limit[pair_index_ba] = max_per_pair_half
            else:
                contact_point_limit[pair_index_ab] = max_per_pair
                contact_point_limit[pair_index_ba] = 0
        else:
            contact_point_limit[pair_index_ab] = 0
            contact_point_limit[pair_index_ba] = 0

    # Allocate contact points using reusable method
    _success = allocate_contact_points(
        num_contacts_a,
        num_contacts_b,
        shape_a,
        shape_b,
        rigid_contact_max,
        contact_count,
        contact_shape0,
        contact_shape1,
        contact_point_id,
    )


@wp.struct
class GeoData:
    """
    Struct to bundle geometry-related data for collision detection.

    This struct contains all the geometric properties and transforms
    needed for a single shape in collision detection algorithms.
    """

    shape_index: int
    """Index of the shape"""
    rigid_body_index: int
    """Index of the rigid body"""
    geo_type: int
    """Type of the geometry"""
    geo_scale: wp.vec3
    """Scale of the geometry"""
    min_scale: float
    """Minimum scale of the geometry"""
    thickness: float
    """Thickness of the geometry"""
    radius_eff: float
    """Effective radius of the geometry"""
    X_wb: wp.transform  # world-to-body transform
    """World-to-body transform"""
    X_bw: wp.transform  # body-to-world transform (inverse)
    """Body-to-world transform (inverse)"""
    X_bs: wp.transform  # body-to-shape transform
    """Body-to-shape transform"""
    X_ws: wp.transform  # world-to-shape transform
    """World-to-shape transform"""
    X_sw: wp.transform  # shape-to-world transform (inverse)
    """Shape-to-world transform (inverse)"""


@wp.func
def create_geo_data(
    shape_index: int,
    body_q: wp.array(dtype=wp.transform),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_thickness: wp.array(dtype=float),
) -> GeoData:
    """
    Create a GeoData struct from shape arrays.

    Args:
        shape_index: Index of the shape
        body_q: Array of body transforms
        shape_transform: Array of shape transforms
        shape_body: Array mapping shapes to bodies
        shape_type: Array of shape types
        shape_scale: Array of shape scales
        shape_thickness: Array of shape thicknesses

    Returns:
        GeoData: Initialized geometry data struct
    """
    geo_data = GeoData()

    geo_data.shape_index = shape_index
    geo_data.rigid_body_index = shape_body[shape_index]

    # Set up transforms
    if geo_data.rigid_body_index >= 0:
        geo_data.X_wb = body_q[geo_data.rigid_body_index]
    else:
        geo_data.X_wb = wp.transform_identity()

    geo_data.X_bs = shape_transform[shape_index]
    geo_data.X_ws = wp.transform_multiply(geo_data.X_wb, geo_data.X_bs)
    geo_data.X_sw = wp.transform_inverse(geo_data.X_ws)
    geo_data.X_bw = wp.transform_inverse(geo_data.X_wb)

    # Set geometry properties
    geo_data.geo_type = shape_type[shape_index]
    geo_data.geo_scale = shape_scale[shape_index]
    geo_data.min_scale = wp.min(geo_data.geo_scale)
    geo_data.thickness = shape_thickness[shape_index]

    # Determine effective radius
    geo_data.radius_eff = float(0.0)
    if (
        geo_data.geo_type == GeoType.SPHERE
        or geo_data.geo_type == GeoType.CAPSULE
        # or geo_data.geo_type == GeoType.CYLINDER # Cylinder does not have an effective radius - it can't be represented as a minkowski sum between some geometry and a sphere
        or geo_data.geo_type == GeoType.CONE
    ):
        geo_data.radius_eff = geo_data.geo_scale[0]

    return geo_data


@wp.func
def capsule_plane_collision(
    geo_a: GeoData,
    geo_b: GeoData,
    point_id: int,
    edge_sdf_iter: int,
):
    """
    Handle collision between a capsule (geo_a) and a plane (geo_b).

    Returns:
        tuple: (p_a_world, p_b_world, normal, distance)
    """
    plane_width = geo_b.geo_scale[0]
    plane_length = geo_b.geo_scale[1]

    if point_id < 2:
        # vertex-based collision
        half_height_a = geo_a.geo_scale[1]
        side = float(point_id) * 2.0 - 1.0
        p_a_world = wp.transform_point(geo_a.X_ws, wp.vec3(0.0, 0.0, side * half_height_a))
        query_b = wp.transform_point(geo_b.X_sw, p_a_world)
        p_b_body = closest_point_plane(geo_b.geo_scale[0], geo_b.geo_scale[1], query_b)
        p_b_world = wp.transform_point(geo_b.X_ws, p_b_body)
        diff = p_a_world - p_b_world
        if geo_b.geo_scale[0] > 0.0 and geo_b.geo_scale[1] > 0.0:
            normal = wp.normalize(diff)
        else:
            normal = wp.transform_vector(geo_b.X_ws, wp.vec3(0.0, 0.0, 1.0))
        distance = wp.dot(diff, normal)
    else:
        # contact between capsule A and edges of finite plane B
        edge = get_plane_edge(point_id - 2, plane_width, plane_length)
        edge0_world = wp.transform_point(geo_b.X_ws, wp.spatial_top(edge))
        edge1_world = wp.transform_point(geo_b.X_ws, wp.spatial_bottom(edge))
        edge0_a = wp.transform_point(geo_a.X_sw, edge0_world)
        edge1_a = wp.transform_point(geo_a.X_sw, edge1_world)
        max_iter = edge_sdf_iter
        u = closest_edge_coordinate_capsule(geo_a.geo_scale[0], geo_a.geo_scale[1], edge0_a, edge1_a, max_iter)
        p_b_world = (1.0 - u) * edge0_world + u * edge1_world

        # find closest point + contact normal on capsule A
        half_height_a = geo_a.geo_scale[1]
        p0_a_world = wp.transform_point(geo_a.X_ws, wp.vec3(0.0, 0.0, half_height_a))
        p1_a_world = wp.transform_point(geo_a.X_ws, wp.vec3(0.0, 0.0, -half_height_a))
        p_a_world = closest_point_line_segment(p0_a_world, p1_a_world, p_b_world)
        diff = p_a_world - p_b_world
        normal = wp.transform_vector(geo_b.X_ws, wp.vec3(0.0, 0.0, 1.0))
        # normal = wp.normalize(diff)
        distance = wp.dot(diff, normal)

    return p_a_world, p_b_world, normal, distance


@wp.func
def cylinder_plane_collision(
    geo_a: GeoData,
    geo_b: GeoData,
    point_id: int,
    edge_sdf_iter: int,
):
    """
    Handle collision between a cylinder (geo_a) and an infinite plane (geo_b).

    Returns:
        tuple: (p_a_world, p_b_world, normal, distance)
    """
    # World-space plane
    plane_normal_world = wp.transform_vector(geo_b.X_ws, wp.vec3(0.0, 0.0, 1.0))
    plane_pos_world = wp.transform_point(geo_b.X_ws, wp.vec3(0.0, 0.0, 0.0))

    # World-space cylinder params
    cylinder_center_world = wp.transform_point(geo_a.X_ws, wp.vec3(0.0, 0.0, 0.0))
    cylinder_axis_world = wp.normalize(wp.transform_vector(geo_a.X_ws, wp.vec3(0.0, 0.0, 1.0)))
    cylinder_radius = geo_a.geo_scale[0]
    cylinder_half_height = geo_a.geo_scale[1]

    # Use primitive helper (in world space)
    dist_vec, pos_mat, n_world = primitive.collide_plane_cylinder(
        plane_normal_world,
        plane_pos_world,
        cylinder_center_world,
        cylinder_axis_world,
        cylinder_radius,
        cylinder_half_height,
    )

    # Support up to the primitive's 4 contacts (2 caps + 2 side points)
    idx = wp.min(int(point_id), 3)
    distance = dist_vec[idx]
    mid_pos = pos_mat[idx]
    normal = n_world

    # Split midpoint into shape-plane endpoints
    p_a_world = mid_pos + 0.5 * normal * distance
    p_b_world = mid_pos - 0.5 * normal * distance

    return p_a_world, p_b_world, normal, distance


@wp.func
def mesh_box_collision(
    geo_a: GeoData,
    geo_b: GeoData,
    point_id: int,
    shape_source_ptr: wp.array(dtype=wp.uint64),
    shape_a: int,
):
    """
    Handle collision between a mesh (geo_a) and a box (geo_b).

    Returns:
        tuple: (p_a_world, p_b_world, normal, distance)
    """
    # vertex-based contact
    mesh = wp.mesh_get(shape_source_ptr[shape_a])
    body_a_pos = wp.cw_mul(mesh.points[point_id], geo_a.geo_scale)
    p_a_world = wp.transform_point(geo_a.X_ws, body_a_pos)
    # find closest point + contact normal on box B
    query_b = wp.transform_point(geo_b.X_sw, p_a_world)
    p_b_body = closest_point_box(geo_b.geo_scale, query_b)
    p_b_world = wp.transform_point(geo_b.X_ws, p_b_body)
    diff = p_a_world - p_b_world
    # this is more reliable in practice than using the SDF gradient
    normal = wp.normalize(diff)
    if box_sdf(geo_b.geo_scale, query_b) < 0.0:
        normal = -normal
    distance = wp.dot(diff, normal)

    return p_a_world, p_b_world, normal, distance


@wp.func
def mesh_mesh_collision(
    geo_a: GeoData,
    geo_b: GeoData,
    point_id: int,
    shape_source_ptr: wp.array(dtype=wp.uint64),
    shape_a: int,
    shape_b: int,
    rigid_contact_margin: float,
    thickness: float,
):
    """
    Handle collision between two meshes (geo_a and geo_b).

    Returns:
        tuple: (p_a_world, p_b_world, normal, distance, valid)
        where valid indicates if a valid collision was found
    """
    # vertex-based contact
    mesh = wp.mesh_get(shape_source_ptr[shape_a])
    mesh_b = shape_source_ptr[shape_b]

    body_a_pos = wp.cw_mul(mesh.points[point_id], geo_a.geo_scale)
    p_a_world = wp.transform_point(geo_a.X_ws, body_a_pos)
    query_b_local = wp.transform_point(geo_b.X_sw, p_a_world)

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    min_scale = min(geo_a.min_scale, geo_b.min_scale)
    max_dist = (rigid_contact_margin + thickness) / min_scale

    res = wp.mesh_query_point_sign_normal(
        mesh_b, wp.cw_div(query_b_local, geo_b.geo_scale), max_dist, sign, face_index, face_u, face_v
    )

    if res:
        shape_p = wp.mesh_eval_position(mesh_b, face_index, face_u, face_v)
        shape_p = wp.cw_mul(shape_p, geo_b.geo_scale)
        p_b_world = wp.transform_point(geo_b.X_ws, shape_p)
        # contact direction vector in world frame
        diff_b = p_a_world - p_b_world
        normal = wp.normalize(diff_b) * sign
        distance = wp.dot(diff_b, normal)
        valid = True
    else:
        # Return dummy values when no collision found
        p_b_world = wp.vec3(0.0, 0.0, 0.0)
        normal = wp.vec3(0.0, 0.0, 1.0)
        distance = 1.0e6
        valid = False

    return p_a_world, p_b_world, normal, distance, valid


@wp.func
def mesh_plane_collision(
    geo_a: GeoData,
    geo_b: GeoData,
    point_id: int,
    shape_source_ptr: wp.array(dtype=wp.uint64),
    shape_a: int,
    rigid_contact_margin: float,
):
    """
    Handle collision between a mesh (geo_a) and a plane (geo_b).

    Returns:
        tuple: (p_a_world, p_b_world, normal, distance, valid)
        where valid indicates if a valid collision was found
    """
    # vertex-based contact
    mesh = wp.mesh_get(shape_source_ptr[shape_a])
    body_a_pos = wp.cw_mul(mesh.points[point_id], geo_a.geo_scale)
    p_a_world = wp.transform_point(geo_a.X_ws, body_a_pos)
    query_b = wp.transform_point(geo_b.X_sw, p_a_world)
    p_b_body = closest_point_plane(geo_b.geo_scale[0], geo_b.geo_scale[1], query_b)
    p_b_world = wp.transform_point(geo_b.X_ws, p_b_body)
    diff = p_a_world - p_b_world

    # if the plane is infinite or the point is within the plane we fix the normal to prevent intersections
    if (geo_b.geo_scale[0] == 0.0 and geo_b.geo_scale[1] == 0.0) or (
        wp.abs(query_b[0]) < geo_b.geo_scale[0] and wp.abs(query_b[1]) < geo_b.geo_scale[1]
    ):
        normal = wp.transform_vector(geo_b.X_ws, wp.vec3(0.0, 0.0, 1.0))
        distance = wp.dot(diff, normal)
        valid = True
    else:
        normal = wp.normalize(diff)
        distance = wp.dot(diff, normal)
        # ignore extreme penetrations (e.g. when mesh is below the plane)
        if distance < -rigid_contact_margin:
            valid = False
        else:
            valid = True

    return p_a_world, p_b_world, normal, distance, valid


@wp.func
def mesh_capsule_collision(
    geo_a: GeoData,
    geo_b: GeoData,
    point_id: int,
    shape_source_ptr: wp.array(dtype=wp.uint64),
    shape_a: int,
):
    """
    Handle collision between a mesh (geo_a) and a capsule (geo_b).

    Returns:
        tuple: (p_a_world, p_b_world, normal, distance)
    """
    # vertex-based contact
    mesh = wp.mesh_get(shape_source_ptr[shape_a])
    body_a_pos = wp.cw_mul(mesh.points[point_id], geo_a.geo_scale)
    p_a_world = wp.transform_point(geo_a.X_ws, body_a_pos)
    # find closest point + contact normal on capsule B
    half_height_b = geo_b.geo_scale[1]
    A_b = wp.transform_point(geo_b.X_ws, wp.vec3(0.0, 0.0, half_height_b))
    B_b = wp.transform_point(geo_b.X_ws, wp.vec3(0.0, 0.0, -half_height_b))
    p_b_world = closest_point_line_segment(A_b, B_b, p_a_world)
    diff = p_a_world - p_b_world
    # this is more reliable in practice than using the SDF gradient
    normal = wp.normalize(diff)
    distance = wp.dot(diff, normal)

    return p_a_world, p_b_world, normal, distance


@wp.func
def capsule_mesh_collision(
    geo_a: GeoData,
    geo_b: GeoData,
    point_id: int,
    shape_source_ptr: wp.array(dtype=wp.uint64),
    shape_b: int,
    rigid_contact_margin: float,
    thickness: float,
    edge_sdf_iter: int,
):
    """
    Handle collision between a capsule (geo_a) and a mesh (geo_b).

    Returns:
        tuple: (p_a_world, p_b_world, normal, distance, valid)
        where valid indicates if a valid collision was found
    """
    # find closest edge coordinate to mesh SDF B
    half_height_a = geo_a.geo_scale[1]
    # edge from capsule A
    # depending on point id, we query an edge from -h to 0 or 0 to h
    e0 = wp.vec3(0.0, 0.0, -half_height_a * float(point_id % 2))
    e1 = wp.vec3(0.0, 0.0, half_height_a * float((point_id + 1) % 2))
    edge0_world = wp.transform_point(geo_a.X_ws, e0)
    edge1_world = wp.transform_point(geo_a.X_ws, e1)
    edge0_b = wp.transform_point(geo_b.X_sw, edge0_world)
    edge1_b = wp.transform_point(geo_b.X_sw, edge1_world)
    max_iter = edge_sdf_iter
    max_dist = (rigid_contact_margin + thickness) / geo_b.min_scale
    mesh_b = shape_source_ptr[shape_b]
    u = closest_edge_coordinate_mesh(
        mesh_b, wp.cw_div(edge0_b, geo_b.geo_scale), wp.cw_div(edge1_b, geo_b.geo_scale), max_iter, max_dist
    )
    p_a_world = (1.0 - u) * edge0_world + u * edge1_world
    query_b_local = wp.transform_point(geo_b.X_sw, p_a_world)
    mesh_b = shape_source_ptr[shape_b]

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    res = wp.mesh_query_point_sign_normal(
        mesh_b, wp.cw_div(query_b_local, geo_b.geo_scale), max_dist, sign, face_index, face_u, face_v
    )
    if res:
        shape_p = wp.mesh_eval_position(mesh_b, face_index, face_u, face_v)
        shape_p = wp.cw_mul(shape_p, geo_b.geo_scale)
        p_b_world = wp.transform_point(geo_b.X_ws, shape_p)
        p_a_world = closest_point_line_segment(edge0_world, edge1_world, p_b_world)
        # contact direction vector in world frame
        diff = p_a_world - p_b_world
        normal = wp.normalize(diff)
        distance = wp.dot(diff, normal)
        valid = True
    else:
        # Return dummy values when no collision found
        p_b_world = wp.vec3(0.0, 0.0, 0.0)
        normal = wp.vec3(0.0, 0.0, 1.0)
        distance = 1.0e6
        valid = False

    return p_a_world, p_b_world, normal, distance, valid


@wp.func
def capsule_capsule_collision(
    geo_a: GeoData,
    geo_b: GeoData,
    point_id: int,
    edge_sdf_iter: int,
):
    """
    Handle collision between two capsules (geo_a and geo_b).

    Returns:
        tuple: (p_a_world, p_b_world, normal, distance)
    """
    # find closest edge coordinate to capsule SDF B
    half_height_a = geo_a.geo_scale[1]
    half_height_b = geo_b.geo_scale[1]
    # edge from capsule A
    # depending on point id, we query an edge from 0 to 0.5 or 0.5 to 1
    e0 = wp.vec3(0.0, 0.0, half_height_a * float(point_id % 2))
    e1 = wp.vec3(0.0, 0.0, -half_height_a * float((point_id + 1) % 2))
    edge0_world = wp.transform_point(geo_a.X_ws, e0)
    edge1_world = wp.transform_point(geo_a.X_ws, e1)
    edge0_b = wp.transform_point(geo_b.X_sw, edge0_world)
    edge1_b = wp.transform_point(geo_b.X_sw, edge1_world)
    max_iter = edge_sdf_iter
    u = closest_edge_coordinate_capsule(geo_b.geo_scale[0], geo_b.geo_scale[1], edge0_b, edge1_b, max_iter)
    p_a_world = (1.0 - u) * edge0_world + u * edge1_world
    p0_b_world = wp.transform_point(geo_b.X_ws, wp.vec3(0.0, 0.0, half_height_b))
    p1_b_world = wp.transform_point(geo_b.X_ws, wp.vec3(0.0, 0.0, -half_height_b))
    p_b_world = closest_point_line_segment(p0_b_world, p1_b_world, p_a_world)
    diff = p_a_world - p_b_world
    normal = wp.normalize(diff)
    distance = wp.dot(diff, normal)

    return p_a_world, p_b_world, normal, distance


@wp.func
def box_box_collision(
    geo_a: GeoData,
    geo_b: GeoData,
    point_id: int,
    edge_sdf_iter: int,
):
    """
    Handle collision between two boxes (geo_a and geo_b).

    Returns:
        tuple: (p_a_world, p_b_world, normal, distance)
    """
    # edge-based box contact
    edge = get_box_edge(point_id, geo_a.geo_scale)
    edge0_world = wp.transform_point(geo_a.X_ws, wp.spatial_top(edge))
    edge1_world = wp.transform_point(geo_a.X_ws, wp.spatial_bottom(edge))
    edge0_b = wp.transform_point(geo_b.X_sw, edge0_world)
    edge1_b = wp.transform_point(geo_b.X_sw, edge1_world)
    max_iter = edge_sdf_iter
    u = closest_edge_coordinate_box(geo_b.geo_scale, edge0_b, edge1_b, max_iter)
    p_a_world = (1.0 - u) * edge0_world + u * edge1_world

    # find closest point + contact normal on box B
    query_b = wp.transform_point(geo_b.X_sw, p_a_world)
    p_b_body = closest_point_box(geo_b.geo_scale, query_b)
    p_b_world = wp.transform_point(geo_b.X_ws, p_b_body)
    diff = p_a_world - p_b_world

    normal = wp.transform_vector(geo_b.X_ws, box_sdf_grad(geo_b.geo_scale, query_b))
    distance = wp.dot(diff, normal)

    return p_a_world, p_b_world, normal, distance


@wp.func
def box_capsule_collision(
    geo_a: GeoData,
    geo_b: GeoData,
    point_id: int,
    edge_sdf_iter: int,
):
    """
    Handle collision between a box (geo_a) and a capsule (geo_b).

    Returns:
        tuple: (p_a_world, p_b_world, normal, distance)
    """
    half_height_b = geo_b.geo_scale[1]
    # capsule B
    # depending on point id, we query an edge from 0 to 0.5 or 0.5 to 1
    e0 = wp.vec3(0.0, 0.0, -half_height_b * float(point_id % 2))
    e1 = wp.vec3(0.0, 0.0, half_height_b * float((point_id + 1) % 2))
    edge0_world = wp.transform_point(geo_b.X_ws, e0)
    edge1_world = wp.transform_point(geo_b.X_ws, e1)
    edge0_a = wp.transform_point(geo_a.X_sw, edge0_world)
    edge1_a = wp.transform_point(geo_a.X_sw, edge1_world)
    max_iter = edge_sdf_iter
    u = closest_edge_coordinate_box(geo_a.geo_scale, edge0_a, edge1_a, max_iter)
    p_b_world = (1.0 - u) * edge0_world + u * edge1_world
    # find closest point + contact normal on box A
    query_a = wp.transform_point(geo_a.X_sw, p_b_world)
    p_a_body = closest_point_box(geo_a.geo_scale, query_a)
    p_a_world = wp.transform_point(geo_a.X_ws, p_a_body)
    diff = p_a_world - p_b_world
    # the contact point inside the capsule should already be outside the box
    normal = -wp.transform_vector(geo_a.X_ws, box_sdf_grad(geo_a.geo_scale, query_a))
    distance = wp.dot(diff, normal)

    return p_a_world, p_b_world, normal, distance


@wp.func
def box_plane_collision(
    geo_a: GeoData,
    geo_b: GeoData,
    point_id: int,
    edge_sdf_iter: int,
):
    """
    Handle collision between a box (geo_a) and a plane (geo_b).

    Returns:
        tuple: (p_a_world, p_b_world, normal, distance, valid)
        where valid indicates if a valid collision was found
    """
    plane_width = geo_b.geo_scale[0]
    plane_length = geo_b.geo_scale[1]

    if point_id < 8:
        # vertex-based contact
        p_a_body = get_box_vertex(point_id, geo_a.geo_scale)
        p_a_world = wp.transform_point(geo_a.X_ws, p_a_body)
        query_b = wp.transform_point(geo_b.X_sw, p_a_world)
        p_b_body = closest_point_plane(plane_width, plane_length, query_b)
        p_b_world = wp.transform_point(geo_b.X_ws, p_b_body)
        diff = p_a_world - p_b_world
        normal = wp.transform_vector(geo_b.X_ws, wp.vec3(0.0, 0.0, 1.0))
        if plane_width > 0.0 and plane_length > 0.0:
            if wp.abs(query_b[0]) > plane_width or wp.abs(query_b[1]) > plane_length:
                # skip, we will evaluate the plane edge contact with the box later
                valid = False
                distance = 1.0e6
                return p_a_world, p_b_world, normal, distance, valid
            # check whether the COM is above the plane
            # sign = wp.sign(wp.dot(wp.transform_get_translation(geo_a.X_ws) - p_b_world, normal))
            # if sign < 0.0:
            #     # the entire box is most likely below the plane
            #     return
        # the contact point is within plane boundaries
        distance = wp.dot(diff, normal)
        valid = True
    else:
        # contact between box A and edges of finite plane B
        edge = get_plane_edge(point_id - 8, plane_width, plane_length)
        edge0_world = wp.transform_point(geo_b.X_ws, wp.spatial_top(edge))
        edge1_world = wp.transform_point(geo_b.X_ws, wp.spatial_bottom(edge))
        edge0_a = wp.transform_point(geo_a.X_sw, edge0_world)
        edge1_a = wp.transform_point(geo_a.X_sw, edge1_world)
        max_iter = edge_sdf_iter
        u = closest_edge_coordinate_box(geo_a.geo_scale, edge0_a, edge1_a, max_iter)
        p_b_world = (1.0 - u) * edge0_world + u * edge1_world

        # find closest point + contact normal on box A
        query_a = wp.transform_point(geo_a.X_sw, p_b_world)
        p_a_body = closest_point_box(geo_a.geo_scale, query_a)
        p_a_world = wp.transform_point(geo_a.X_ws, p_a_body)
        query_b = wp.transform_point(geo_b.X_sw, p_a_world)
        if wp.abs(query_b[0]) > plane_width or wp.abs(query_b[1]) > plane_length:
            # ensure that the closest point is actually inside the plane
            valid = False
            normal = wp.vec3(0.0, 0.0, 1.0)
            distance = 1.0e6
            return p_a_world, p_b_world, normal, distance, valid
        diff = p_a_world - p_b_world
        com_a = wp.transform_get_translation(geo_a.X_ws)
        query_b = wp.transform_point(geo_b.X_sw, com_a)
        if wp.abs(query_b[0]) > plane_width or wp.abs(query_b[1]) > plane_length:
            # the COM is outside the plane
            normal = wp.normalize(com_a - p_b_world)
        else:
            normal = wp.transform_vector(geo_b.X_ws, wp.vec3(0.0, 0.0, 1.0))
        distance = wp.dot(diff, normal)
        valid = True

    return p_a_world, p_b_world, normal, distance, valid


@wp.func
def box_mesh_collision(
    geo_a: GeoData,
    geo_b: GeoData,
    point_id: int,
    shape_source_ptr: wp.array(dtype=wp.uint64),
    shape_b: int,
    rigid_contact_margin: float,
    thickness: float,
):
    """
    Handle collision between a box (geo_a) and a mesh (geo_b).

    Returns:
        tuple: (p_a_world, p_b_world, normal, distance, valid)
        where valid indicates if a valid collision was found
    """
    # vertex-based contact
    query_a = get_box_vertex(point_id, geo_a.geo_scale)
    p_a_world = wp.transform_point(geo_a.X_ws, query_a)
    query_b_local = wp.transform_point(geo_b.X_sw, p_a_world)
    mesh_b = shape_source_ptr[shape_b]
    max_dist = (rigid_contact_margin + thickness) / geo_b.min_scale
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    res = wp.mesh_query_point_sign_normal(
        mesh_b, wp.cw_div(query_b_local, geo_b.geo_scale), max_dist, sign, face_index, face_u, face_v
    )

    if res:
        shape_p = wp.mesh_eval_position(mesh_b, face_index, face_u, face_v)
        shape_p = wp.cw_mul(shape_p, geo_b.geo_scale)
        p_b_world = wp.transform_point(geo_b.X_ws, shape_p)
        # contact direction vector in world frame
        diff_b = p_a_world - p_b_world
        normal = wp.normalize(diff_b) * sign
        distance = wp.dot(diff_b, normal)
        valid = True
    else:
        # Return dummy values when no collision found
        p_b_world = wp.vec3(0.0, 0.0, 0.0)
        normal = wp.vec3(0.0, 0.0, 1.0)
        distance = 1.0e6
        valid = False

    return p_a_world, p_b_world, normal, distance, valid


@wp.func
def sphere_sphere_collision(
    geo_a: GeoData,
    geo_b: GeoData,
):
    """
    Handle collision between two spheres.

    Returns:
        tuple: (p_a_world, p_b_world, normal, distance)
    """
    p_a_world = wp.transform_get_translation(geo_a.X_ws)
    p_b_world = wp.transform_get_translation(geo_b.X_ws)
    diff = p_a_world - p_b_world
    normal = wp.normalize(diff)
    distance = wp.dot(diff, normal)

    return p_a_world, p_b_world, normal, distance


@wp.func
def sphere_box_collision(
    geo_a: GeoData,
    geo_b: GeoData,
):
    """
    Handle collision between a sphere (geo_a) and a box (geo_b).

    Returns:
        tuple: (p_a_world, p_b_world, normal, distance)
    """
    p_a_world = wp.transform_get_translation(geo_a.X_ws)
    # contact point in frame of body B
    p_a_body = wp.transform_point(geo_b.X_sw, p_a_world)
    p_b_body = closest_point_box(geo_b.geo_scale, p_a_body)
    p_b_world = wp.transform_point(geo_b.X_ws, p_b_body)
    diff = p_a_world - p_b_world
    normal = wp.normalize(diff)
    distance = wp.dot(diff, normal)

    return p_a_world, p_b_world, normal, distance


@wp.func
def sphere_capsule_collision(
    geo_a: GeoData,
    geo_b: GeoData,
):
    """
    Handle collision between a sphere (geo_a) and a capsule (geo_b).

    Returns:
        tuple: (p_a_world, p_b_world, normal, distance)
    """
    p_a_world = wp.transform_get_translation(geo_a.X_ws)
    half_height_b = geo_b.geo_scale[1]
    # capsule B
    A_b = wp.transform_point(geo_b.X_ws, wp.vec3(0.0, 0.0, half_height_b))
    B_b = wp.transform_point(geo_b.X_ws, wp.vec3(0.0, 0.0, -half_height_b))
    p_b_world = closest_point_line_segment(A_b, B_b, p_a_world)
    diff = p_a_world - p_b_world
    normal = wp.normalize(diff)
    distance = wp.dot(diff, normal)

    return p_a_world, p_b_world, normal, distance


@wp.func
def sphere_mesh_collision(
    geo_a: GeoData,
    geo_b: GeoData,
    shape_source_ptr: wp.array(dtype=wp.uint64),
    shape_b: int,
    rigid_contact_margin: float,
    thickness: float,
):
    """
    Handle collision between a sphere (geo_a) and a mesh (geo_b).

    Returns:
        tuple: (p_a_world, p_b_world, normal, distance, valid)
        where valid indicates if a valid collision was found
    """
    p_a_world = wp.transform_get_translation(geo_a.X_ws)
    mesh_b = shape_source_ptr[shape_b]
    query_b_local = wp.transform_point(geo_b.X_sw, p_a_world)
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    max_dist = (thickness + rigid_contact_margin + geo_a.radius_eff) / geo_b.min_scale
    res = wp.mesh_query_point_sign_normal(
        mesh_b, wp.cw_div(query_b_local, geo_b.geo_scale), max_dist, sign, face_index, face_u, face_v
    )
    if res:
        shape_p = wp.mesh_eval_position(mesh_b, face_index, face_u, face_v)
        shape_p = wp.cw_mul(shape_p, geo_b.geo_scale)
        p_b_world = wp.transform_point(geo_b.X_ws, shape_p)
        diff = p_a_world - p_b_world
        normal = wp.normalize(diff)
        distance = wp.dot(diff, normal)
        valid = True
    else:
        # Return dummy values when no collision found
        p_b_world = wp.vec3(0.0, 0.0, 0.0)
        normal = wp.vec3(0.0, 0.0, 1.0)
        distance = 1.0e6
        valid = False

    return p_a_world, p_b_world, normal, distance, valid


@wp.func
def sphere_plane_collision(
    geo_a: GeoData,
    geo_b: GeoData,
):
    """
    Handle collision between a sphere (geo_a) and a plane (geo_b).

    Returns:
        tuple: (p_a_world, p_b_world, normal, distance)
    """
    p_a_world = wp.transform_get_translation(geo_a.X_ws)
    p_b_body = closest_point_plane(geo_b.geo_scale[0], geo_b.geo_scale[1], wp.transform_point(geo_b.X_sw, p_a_world))
    p_b_world = wp.transform_point(geo_b.X_ws, p_b_body)
    diff = p_a_world - p_b_world
    normal = wp.transform_vector(geo_b.X_ws, wp.vec3(0.0, 0.0, 1.0))
    distance = wp.dot(diff, normal)

    return p_a_world, p_b_world, normal, distance


def generate_handle_contact_pairs_kernel(enable_backward: bool):
    @wp.kernel(module="unique", enable_backward=enable_backward)
    def handle_contact_pairs(
        body_q: wp.array(dtype=wp.transform),
        shape_transform: wp.array(dtype=wp.transform),
        shape_body: wp.array(dtype=int),
        shape_type: wp.array(dtype=int),
        shape_scale: wp.array(dtype=wp.vec3),
        shape_source_ptr: wp.array(dtype=wp.uint64),
        shape_thickness: wp.array(dtype=float),
        num_shapes: int,
        shape_contact_margin: wp.array(dtype=float),
        contact_broad_shape0: wp.array(dtype=int),
        contact_broad_shape1: wp.array(dtype=int),
        contact_point_id: wp.array(dtype=int),
        contact_point_limit: wp.array(dtype=int),
        edge_sdf_iter: int,
        # outputs
        contact_count: wp.array(dtype=int),
        contact_shape0: wp.array(dtype=int),
        contact_shape1: wp.array(dtype=int),
        contact_point0: wp.array(dtype=wp.vec3),
        contact_point1: wp.array(dtype=wp.vec3),
        contact_offset0: wp.array(dtype=wp.vec3),
        contact_offset1: wp.array(dtype=wp.vec3),
        contact_normal: wp.array(dtype=wp.vec3),
        contact_thickness0: wp.array(dtype=float),
        contact_thickness1: wp.array(dtype=float),
        contact_pairwise_counter: wp.array(dtype=int),
        contact_tids: wp.array(dtype=int),
    ):
        tid = wp.tid()
        shape_a = contact_broad_shape0[tid]
        shape_b = contact_broad_shape1[tid]
        if shape_a == shape_b:
            return

        if contact_point_limit:
            pair_index = shape_a * num_shapes + shape_b
            contact_limit = contact_point_limit[pair_index]
            if contact_pairwise_counter[pair_index] >= contact_limit:
                # reached limit of contact points per contact pair
                return

        point_id = contact_point_id[tid]

        # Create geometry data structs for both shapes
        geo_a = create_geo_data(shape_a, body_q, shape_transform, shape_body, shape_type, shape_scale, shape_thickness)
        geo_b = create_geo_data(shape_b, body_q, shape_transform, shape_body, shape_type, shape_scale, shape_thickness)

        # Calculate contact margin as sum of per-shape margins (consistent with thickness summing)
        rigid_contact_margin = shape_contact_margin[shape_a] + shape_contact_margin[shape_b]

        distance = 1.0e6
        thickness = geo_a.thickness + geo_b.thickness

        if geo_a.geo_type == GeoType.SPHERE and geo_b.geo_type == GeoType.SPHERE:
            p_a_world, p_b_world, normal, distance = sphere_sphere_collision(geo_a, geo_b)

        elif geo_a.geo_type == GeoType.SPHERE and geo_b.geo_type == GeoType.BOX:
            p_a_world, p_b_world, normal, distance = sphere_box_collision(geo_a, geo_b)

        elif geo_a.geo_type == GeoType.SPHERE and geo_b.geo_type == GeoType.CAPSULE:
            p_a_world, p_b_world, normal, distance = sphere_capsule_collision(geo_a, geo_b)

        elif geo_a.geo_type == GeoType.SPHERE and (
            geo_b.geo_type == GeoType.MESH or geo_b.geo_type == GeoType.CONVEX_MESH
        ):
            p_a_world, p_b_world, normal, distance, valid = sphere_mesh_collision(
                geo_a, geo_b, shape_source_ptr, shape_b, rigid_contact_margin, thickness
            )
            if not valid:
                return

        elif geo_a.geo_type == GeoType.PLANE and geo_b.geo_type == GeoType.SPHERE:
            p_b_world, p_a_world, neg_normal, distance = sphere_plane_collision(geo_b, geo_a)
            # Flip the normal since we flipped the arguments
            normal = -neg_normal

        elif geo_a.geo_type == GeoType.BOX and geo_b.geo_type == GeoType.BOX:
            p_a_world, p_b_world, normal, distance = box_box_collision(geo_a, geo_b, point_id, edge_sdf_iter)

        elif geo_a.geo_type == GeoType.CAPSULE and geo_b.geo_type == GeoType.BOX:
            p_b_world, p_a_world, neg_normal, distance = box_capsule_collision(geo_b, geo_a, point_id, edge_sdf_iter)
            # Flip the normal since we flipped the arguments
            normal = -neg_normal

        elif geo_a.geo_type == GeoType.PLANE and geo_b.geo_type == GeoType.BOX:
            p_b_world, p_a_world, neg_normal, distance, valid = box_plane_collision(
                geo_b, geo_a, point_id, edge_sdf_iter
            )
            # Flip the normal since we flipped the arguments
            normal = -neg_normal
            if not valid:
                return

        elif geo_a.geo_type == GeoType.CAPSULE and geo_b.geo_type == GeoType.CAPSULE:
            p_a_world, p_b_world, normal, distance = capsule_capsule_collision(geo_a, geo_b, point_id, edge_sdf_iter)

        elif geo_a.geo_type == GeoType.CAPSULE and (
            geo_b.geo_type == GeoType.MESH or geo_b.geo_type == GeoType.CONVEX_MESH
        ):
            p_a_world, p_b_world, normal, distance, valid = capsule_mesh_collision(
                geo_a, geo_b, point_id, shape_source_ptr, shape_b, rigid_contact_margin, thickness, edge_sdf_iter
            )
            if not valid:
                return

        elif (
            geo_a.geo_type == GeoType.MESH or geo_a.geo_type == GeoType.CONVEX_MESH
        ) and geo_b.geo_type == GeoType.CAPSULE:
            p_a_world, p_b_world, normal, distance = mesh_capsule_collision(
                geo_a, geo_b, point_id, shape_source_ptr, shape_a
            )

        elif geo_a.geo_type == GeoType.PLANE and geo_b.geo_type == GeoType.CAPSULE:
            p_b_world, p_a_world, neg_normal, distance = capsule_plane_collision(geo_b, geo_a, point_id, edge_sdf_iter)
            # Flip the normal since we flipped the arguments
            normal = -neg_normal

        elif geo_a.geo_type == GeoType.PLANE and geo_b.geo_type == GeoType.CYLINDER:
            p_b_world, p_a_world, neg_normal, distance = cylinder_plane_collision(geo_b, geo_a, point_id, edge_sdf_iter)
            # Flip the normal since we flipped the arguments
            normal = -neg_normal

            # Check if this contact point is valid (primitive function returns MAXVAL for invalid contacts)
            if distance >= MAXVAL:
                return

        elif (
            geo_a.geo_type == GeoType.MESH or geo_a.geo_type == GeoType.CONVEX_MESH
        ) and geo_b.geo_type == GeoType.BOX:
            p_a_world, p_b_world, normal, distance = mesh_box_collision(
                geo_a, geo_b, point_id, shape_source_ptr, shape_a
            )

        elif geo_a.geo_type == GeoType.BOX and (
            geo_b.geo_type == GeoType.MESH or geo_b.geo_type == GeoType.CONVEX_MESH
        ):
            p_a_world, p_b_world, normal, distance, valid = box_mesh_collision(
                geo_a, geo_b, point_id, shape_source_ptr, shape_b, rigid_contact_margin, thickness
            )
            if not valid:
                return

        elif (geo_a.geo_type == GeoType.MESH or geo_a.geo_type == GeoType.CONVEX_MESH) and (
            geo_b.geo_type == GeoType.MESH or geo_b.geo_type == GeoType.CONVEX_MESH
        ):
            p_a_world, p_b_world, normal, distance, valid = mesh_mesh_collision(
                geo_a, geo_b, point_id, shape_source_ptr, shape_a, shape_b, rigid_contact_margin, thickness
            )
            if not valid:
                return

        elif geo_a.geo_type == GeoType.PLANE and (
            geo_b.geo_type == GeoType.MESH or geo_b.geo_type == GeoType.CONVEX_MESH
        ):
            p_b_world, p_a_world, neg_normal, distance, valid = mesh_plane_collision(
                geo_b, geo_a, point_id, shape_source_ptr, shape_b, rigid_contact_margin
            )
            # Flip the normal since we flipped the arguments
            normal = -neg_normal
            if not valid:
                return

        else:
            # print("Unsupported geometry pair in collision handling")
            return

        # Total separation required by radii and additional thicknesses
        total_separation_needed = geo_a.radius_eff + geo_b.radius_eff + thickness
        d = distance - total_separation_needed
        if d < rigid_contact_margin:
            if contact_pairwise_counter:
                pair_contact_id = counter_increment(
                    contact_pairwise_counter, pair_index, contact_tids, tid, index_limit=contact_limit
                )
                if pair_contact_id == -1:
                    # reached contact point limit
                    # wp.printf("Reached contact point limit %d >= %d for shape pair %d and %d (pair_index: %d)\n",
                    #           contact_pairwise_counter[pair_index], contact_limit, shape_a, shape_b, pair_index)
                    return
            index = counter_increment(contact_count, 0, contact_tids, tid)
            if index == -1:
                return
            contact_shape0[index] = shape_a
            contact_shape1[index] = shape_b
            # transform from world into body frame (so the contact point includes the shape transform)
            contact_point0[index] = wp.transform_point(geo_a.X_bw, p_a_world)
            contact_point1[index] = wp.transform_point(geo_b.X_bw, p_b_world)

            offset_magnitude_a = geo_a.radius_eff + geo_a.thickness
            offset_magnitude_b = geo_b.radius_eff + geo_b.thickness

            contact_offset0[index] = wp.transform_vector(geo_a.X_bw, -offset_magnitude_a * normal)
            contact_offset1[index] = wp.transform_vector(geo_b.X_bw, offset_magnitude_b * normal)
            contact_normal[index] = normal
            contact_thickness0[index] = offset_magnitude_a
            contact_thickness1[index] = offset_magnitude_b

    return handle_contact_pairs


# endregion


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
