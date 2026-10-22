# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU setup queries for deformable visual meshes."""

import warp as wp


@wp.func
def _closest_triangle_weights(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d):
    ab, ac, ap = b - a, c - a, p - a
    d00, d01, d11 = wp.dot(ab, ab), wp.dot(ab, ac), wp.dot(ac, ac)
    d20, d21 = wp.dot(ap, ab), wp.dot(ap, ac)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    weights = wp.vec3d(wp.float64(1.0) - v - w, v, w)
    distance = wp.float64(wp.inf)
    if weights[0] >= 0.0 and v >= 0.0 and w >= 0.0:
        distance = wp.length_sq(p - (weights[0] * a + v * b + w * c))
    for edge in range(3):
        start, end = a, b
        if edge == 1:
            start, end = b, c
        elif edge == 2:
            start, end = c, a
        direction = end - start
        t = wp.clamp(wp.dot(p - start, direction) / wp.length_sq(direction), wp.float64(0.0), wp.float64(1.0))
        candidate_distance = wp.length_sq(p - (start + t * direction))
        if candidate_distance < distance:
            distance = candidate_distance
            weights = wp.vec3d(0.0)
            weights[edge] = wp.float64(1.0) - t
            weights[(edge + 1) % 3] = t
    return weights


@wp.kernel
def embed_triangles(
    mesh: wp.uint64,
    vertices: wp.array[wp.vec3],
    triangle_ids: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    weights: wp.array[wp.vec3],
):
    i = wp.tid()
    point = vertices[i]
    p = wp.vec3d(point)
    nearest = wp.mesh_query_point_no_sign(mesh, point, 1.0e30)
    # The builder supplies a nonempty mesh of nondegenerate triangles. Keep a
    # valid candidate even if the float32 query fails or its bounds round inward.
    best = int(0)
    if nearest.result:
        best = nearest.face
    a = wp.vec3d(wp.mesh_get_point(mesh, 3 * best))
    b = wp.vec3d(wp.mesh_get_point(mesh, 3 * best + 1))
    c = wp.vec3d(wp.mesh_get_point(mesh, 3 * best + 2))
    best_weights = _closest_triangle_weights(p, a, b, c)
    distance = wp.length_sq(p - (best_weights[0] * a + best_weights[1] * b + best_weights[2] * c))
    radius = float(wp.sqrt(distance))
    # Query roundoff depends on the distance as well as the point's coordinates.
    scale = wp.max(radius, wp.max(wp.abs(point[0]), wp.max(wp.abs(point[1]), wp.abs(point[2]))))
    radius += 1.0e-5 * (1.0 + scale)
    query = wp.mesh_query_aabb(mesh, point - wp.vec3(radius), point + wp.vec3(radius))
    face = int(0)
    while wp.mesh_query_aabb_next(query, face):
        a = wp.vec3d(wp.mesh_get_point(mesh, 3 * face))
        b = wp.vec3d(wp.mesh_get_point(mesh, 3 * face + 1))
        c = wp.vec3d(wp.mesh_get_point(mesh, 3 * face + 2))
        candidate = _closest_triangle_weights(p, a, b, c)
        d = wp.length_sq(p - (candidate[0] * a + candidate[1] * b + candidate[2] * c))
        if d < distance or (d == distance and face < best):
            best, distance, best_weights = face, d, candidate
    parent[i] = triangle_ids[best]
    weights[i] = wp.vec3(best_weights)


@wp.kernel
def embed_tetrahedra(
    bvh: wp.uint64,
    vertices: wp.array[wp.vec3],
    origins: wp.array[wp.vec3d],
    inverses: wp.array[wp.mat33d],
    centroids: wp.array[wp.vec3d],
    tet_ids: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    weights: wp.array[wp.vec4],
    clamped: wp.array[wp.int32],
):
    i = wp.tid()
    point = vertices[i]
    p = wp.vec3d(point)
    query = wp.bvh_query_aabb(bvh, point, point)
    tet = int(0)
    best = int(-1)
    score = wp.float64(-wp.inf)
    while wp.bvh_query_next(query, tet):
        bary = inverses[tet] * (p - origins[tet])
        minimum = wp.min(wp.float64(1.0) - bary[0] - bary[1] - bary[2], wp.min(bary[0], wp.min(bary[1], bary[2])))
        if minimum >= -1.0e-6 and (minimum > score or (minimum == score and (best < 0 or tet < best))):
            best, score = tet, minimum
    if best < 0:
        # Preserve the documented nearest-centroid fallback for outside points.
        distance = wp.float64(wp.inf)
        for candidate in range(centroids.shape[0]):
            d = wp.length_sq(p - centroids[candidate])
            if d < distance:
                best, distance = candidate, d
        clamped[i] = 1
    bary = inverses[best] * (p - origins[best])
    w = wp.vec4d(
        wp.max(wp.float64(1.0) - bary[0] - bary[1] - bary[2], wp.float64(0.0)),
        wp.max(bary[0], wp.float64(0.0)),
        wp.max(bary[1], wp.float64(0.0)),
        wp.max(bary[2], wp.float64(0.0)),
    )
    parent[i] = tet_ids[best]
    weights[i] = wp.vec4(w / (w[0] + w[1] + w[2] + w[3]))
