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

from newton._src.solvers.vbd.solver_vbd import evaluate_body_particle_contact


@wp.func
def triangle_normal(A: wp.vec3, B: wp.vec3, C: wp.vec3):
    n = wp.cross(B - A, C - A)
    ln = wp.length(n)
    return wp.vec3(0.0) if ln < 1.0e-12 else (n / ln)


@wp.func
def triangle_barycentric(A: wp.vec3, B: wp.vec3, C: wp.vec3, P: wp.vec3):
    v0 = A - C
    v1 = B - C
    v2 = P - C
    dot00 = wp.dot(v0, v0)
    dot01 = wp.dot(v0, v1)
    dot02 = wp.dot(v0, v2)
    dot11 = wp.dot(v1, v1)
    dot12 = wp.dot(v1, v2)
    denom = dot00 * dot11 - dot01 * dot01
    invDenom = 0.0 if wp.abs(denom) < 1.0e-12 else 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    return wp.vec3(u, v, 1.0 - u - v)


@wp.kernel
def eval_body_contact_kernel(
    # inputs
    dt: float,
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    # body-particle contact
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    particle_radius: wp.array(dtype=float),
    soft_contact_particle: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_max: int,
    shape_material_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    # outputs: particle force and hessian
    forces: wp.array(dtype=wp.vec3),
    hessians: wp.array(dtype=wp.mat33),
):
    t_id = wp.tid()

    particle_body_contact_count = wp.min(contact_max, contact_count[0])

    if t_id < particle_body_contact_count:
        particle_idx = soft_contact_particle[t_id]
        body_contact_force, body_contact_hessian = evaluate_body_particle_contact(
            particle_idx,
            pos[particle_idx],
            pos_prev[particle_idx],
            t_id,
            soft_contact_ke,
            soft_contact_kd,
            friction_mu,
            friction_epsilon,
            particle_radius,
            shape_material_mu,
            shape_body,
            body_q,
            body_q_prev,
            body_qd,
            body_com,
            contact_shape,
            contact_body_pos,
            contact_body_vel,
            contact_normal,
            dt,
        )
        wp.atomic_add(forces, particle_idx, body_contact_force)
        wp.atomic_add(hessians, particle_idx, body_contact_hessian)


@wp.kernel
def handle_vertex_triangle_contacts_kernel(
    thickness: float,
    stiff_factor: float,
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=int, ndim=2),
    broad_phase_vf: wp.array(dtype=int, ndim=2),
    static_diags: wp.array(dtype=float),
    # outputs
    forces: wp.array(dtype=wp.vec3),
    hessian_diags: wp.array(dtype=wp.mat33),
):
    vid = wp.tid()

    x0 = pos[vid]
    force0 = wp.vec3(0.0)
    hess0 = wp.identity(n=3, dtype=float) * 0.0
    vert_stiff = static_diags[vid]
    is_collided = wp.int32(0)

    count = broad_phase_vf[0, vid]
    for i in range(count):
        fid = broad_phase_vf[i + 1, vid]
        face = wp.vec3i(tri_indices[fid, 0], tri_indices[fid, 1], tri_indices[fid, 2])
        x1 = pos[face[0]]
        x2 = pos[face[1]]
        x3 = pos[face[2]]
        tri_normal = triangle_normal(x1, x2, x3)
        dist = wp.dot(x0 - x1, tri_normal)
        p = x0 - tri_normal * dist
        bary_coord = triangle_barycentric(x1, x2, x3, p)

        if wp.abs(dist) > thickness:
            continue
        if bary_coord[0] < 0.0 or bary_coord[1] < 0.0 or bary_coord[2] < 0.0:
            continue  # is outside triangle

        face_stiff = (static_diags[face[0]] + static_diags[face[1]] + static_diags[face[2]]) / 3.0
        stiff = stiff_factor * (vert_stiff * face_stiff) / (vert_stiff + face_stiff)

        force = stiff * tri_normal * (thickness - wp.abs(dist)) * wp.sign(dist)
        hess = stiff * wp.outer(tri_normal, tri_normal)

        force0 += force
        wp.atomic_add(forces, face[0], -force * bary_coord[0])
        wp.atomic_add(forces, face[1], -force * bary_coord[1])
        wp.atomic_add(forces, face[2], -force * bary_coord[2])

        hess0 += hess
        wp.atomic_add(hessian_diags, face[0], hess * bary_coord[0] * bary_coord[0])
        wp.atomic_add(hessian_diags, face[1], hess * bary_coord[1] * bary_coord[1])
        wp.atomic_add(hessian_diags, face[2], hess * bary_coord[2] * bary_coord[2])
        is_collided = 1

    if is_collided != 0:
        wp.atomic_add(forces, vid, force0)
        wp.atomic_add(hessian_diags, vid, hess0)


@wp.kernel
def handle_edge_edge_contacts_kernel(
    thickness: float,
    stiff_factor: float,
    pos: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=int, ndim=2),
    broad_phase_ee: wp.array(dtype=int, ndim=2),
    static_diags: wp.array(dtype=float),
    # outputs
    forces: wp.array(dtype=wp.vec3),
    hessian_diags: wp.array(dtype=wp.mat33),
):
    eid = wp.tid()
    edge0 = wp.vec4i(edge_indices[eid, 2], edge_indices[eid, 3], edge_indices[eid, 0], edge_indices[eid, 1])
    x0 = pos[edge0[0]]
    x1 = pos[edge0[1]]
    len0 = wp.length(x0 - x1)

    force0 = wp.vec3(0.0)
    force1 = wp.vec3(0.0)
    hess0 = wp.identity(n=3, dtype=float) * 0.0
    hess1 = wp.identity(n=3, dtype=float) * 0.0
    stiff_0 = (static_diags[edge0[0]] + static_diags[edge0[1]]) / 2.0
    is_collided = wp.int32(0)

    count = broad_phase_ee[0, eid]
    for i in range(count):
        idx = broad_phase_ee[i + 1, eid]
        edge1 = wp.vec4i(edge_indices[idx, 2], edge_indices[idx, 3], edge_indices[idx, 0], edge_indices[idx, 1])
        x2, x3 = pos[edge1[0]], pos[edge1[1]]
        edge_edge_parallel_epsilon = wp.float32(1e-5)

        st = wp.closest_point_edge_edge(x0, x1, x2, x3, edge_edge_parallel_epsilon)
        s, t = st[0], st[1]

        if (s <= 0) or (s >= 1) or (t <= 0) or (t >= 1):
            continue

        c1 = wp.lerp(x0, x1, s)
        c2 = wp.lerp(x2, x3, t)
        dir = c1 - c2
        dist = wp.length(dir)
        limited_thickness = thickness

        len1 = wp.length(x2 - x3)
        avg_len = (len0 + len1) * 0.5
        if edge0[2] == edge1[0] or edge0[3] == edge1[0]:
            limited_thickness = wp.min(limited_thickness, avg_len * 0.5)
        elif edge0[2] == edge1[1] or edge0[3] == edge1[1]:
            limited_thickness = wp.min(limited_thickness, avg_len * 0.5)
        if edge1[2] == edge0[0] or edge1[3] == edge0[0]:
            limited_thickness = wp.min(limited_thickness, avg_len * 0.5)
        elif edge1[2] == edge0[1] or edge1[3] == edge0[1]:
            limited_thickness = wp.min(limited_thickness, avg_len * 0.5)

        if 1e-6 < dist < limited_thickness:
            stiff_1 = (static_diags[edge1[0]] + static_diags[edge1[1]]) / 2.0
            stiff = stiff_factor * (stiff_0 * stiff_1) / (stiff_0 + stiff_1)

            dir = wp.normalize(dir)
            force = stiff * dir * (limited_thickness - dist)
            hess = stiff * wp.outer(dir, dir)

            force0 += force * (1.0 - s)
            force1 += force * s
            wp.atomic_add(forces, edge1[0], -force * (1.0 - t))
            wp.atomic_add(forces, edge1[1], -force * t)

            hess0 += hess * (1.0 - s) * (1.0 - s)
            hess1 += hess * s * s
            wp.atomic_add(hessian_diags, edge1[0], hess * (1.0 - t) * (1.0 - t))
            wp.atomic_add(hessian_diags, edge1[1], hess * t * t)
            is_collided = 1

    if is_collided != 0:
        wp.atomic_add(forces, edge0[0], force0)
        wp.atomic_add(forces, edge0[1], force1)
        wp.atomic_add(hessian_diags, edge0[0], hess0)
        wp.atomic_add(hessian_diags, edge0[1], hess1)


@wp.func
def intersection_gradient_vector(R: wp.vec3, E: wp.vec3, N: wp.vec3):
    """
    Reference: Resolving Surface Collisions through Intersection Contour Minimization, Pascal Volino & Magnenat-Thalmann, 2006.

    Args:
        R: The direction of the intersection segment
        E: Direction vector of the edge
        N: The normals of the polygons
    """
    dot_EN = wp.dot(E, N)
    if wp.abs(dot_EN) > 1e-6:
        return R - 2.0 * N * wp.dot(E, R) / dot_EN
    else:
        return R


@wp.kernel
def solve_untangling_kernel(
    thickness: float,
    stiff_factor: float,
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=int, ndim=2),
    edge_indices: wp.array(dtype=int, ndim=2),
    broad_phase_ef: wp.array(dtype=int, ndim=2),
    static_diags: wp.array(dtype=float),
    # outputs
    forces: wp.array(dtype=wp.vec3),
    hessian_diags: wp.array(dtype=wp.mat33),
):
    eid = wp.tid()
    edge = wp.vec4i(edge_indices[eid, 2], edge_indices[eid, 3], edge_indices[eid, 0], edge_indices[eid, 1])
    v0 = pos[edge[0]]
    v1 = pos[edge[1]]

    # Skip invalid edge
    len0 = wp.length(v0 - v1)
    if len0 < 5e-4:
        return

    force0 = wp.vec3(0.0)
    force1 = wp.vec3(0.0)
    hess0 = wp.identity(n=3, dtype=float) * 0.0
    hess1 = wp.identity(n=3, dtype=float) * 0.0
    stiff_0 = (static_diags[edge[0]] + static_diags[edge[1]]) / 2.0
    is_collided = wp.int32(0)

    # Edge direction
    E = wp.normalize(v0 - v1)
    N2 = wp.vec3(0.0) if edge[2] < 0 else triangle_normal(v0, v1, pos[edge[2]])
    N3 = wp.vec3(0.0) if edge[3] < 0 else triangle_normal(v0, v1, pos[edge[3]])

    count = broad_phase_ef[0, eid]
    for i in range(count):
        fid = broad_phase_ef[i + 1, eid]
        face = wp.vec3i(tri_indices[fid, 0], tri_indices[fid, 1], tri_indices[fid, 2])

        if face[0] == edge[0] or face[0] == edge[1]:
            continue
        if face[1] == edge[0] or face[1] == edge[1]:
            continue
        if face[2] == edge[0] or face[2] == edge[1]:
            continue

        x0 = pos[face[0]]
        x1 = pos[face[1]]
        x2 = pos[face[2]]
        face_normal = wp.cross(x1 - x0, x2 - x1)
        normal_len = wp.length(face_normal)
        if normal_len < 1e-8:
            continue  # invalid triangle

        face_normal = wp.normalize(face_normal)
        d1 = wp.dot(face_normal, v0 - x0)
        d2 = wp.dot(face_normal, v1 - x0)
        if d1 * d2 >= 0.0:
            continue  # on same side

        d1, d2 = wp.abs(d1), wp.abs(d2)
        hit_point = (v0 * d2 + v1 * d1) / (d2 + d1)
        bary_coord = triangle_barycentric(x0, x1, x2, hit_point)

        if (bary_coord[0] < 1e-2) or (bary_coord[1] < 1e-2) or (bary_coord[2] < 1e-2):
            continue  # hit outside

        G = wp.vec3(0.0)

        if edge[2] >= 0:
            R = wp.cross(face_normal, N2)
            R = wp.vec3(0.0) if wp.length(R) < 1e-6 else wp.normalize(R)
            if wp.dot(wp.cross(E, R), wp.cross(E, pos[edge[2]] - hit_point)) < 0.0:
                R *= -1.0
            G += intersection_gradient_vector(R, E, face_normal)

        if edge[3] >= 0:
            R = wp.cross(face_normal, N3)
            R = wp.vec3(0.0) if wp.length(R) < 1e-6 else wp.normalize(R)
            if wp.dot(wp.cross(E, R), wp.cross(E, pos[edge[3]] - hit_point)) < 0.0:
                R *= -1.0
            G += intersection_gradient_vector(R, E, face_normal)

        if wp.length(G) < 1.0e-12:
            continue
        G = wp.normalize(G)

        # Can be precomputed
        stiff_1 = (static_diags[face[0]] + static_diags[face[1]] + static_diags[face[2]]) / 3.0
        stiff = stiff_factor * (stiff_0 * stiff_1) / (stiff_0 + stiff_1)
        disp = 2.0 * thickness

        force = stiff * G * disp
        hess = stiff * wp.outer(G, G)
        edge_bary = wp.vec2(d2, d1) / (d1 + d2)

        force0 += force * edge_bary[0]
        force1 += force * edge_bary[1]
        hess0 += hess * edge_bary[0] * edge_bary[0]
        hess1 += hess * edge_bary[1] * edge_bary[1]

        wp.atomic_add(forces, face[0], -force * bary_coord[0])
        wp.atomic_add(forces, face[1], -force * bary_coord[1])
        wp.atomic_add(forces, face[2], -force * bary_coord[2])

        wp.atomic_add(hessian_diags, face[0], hess * bary_coord[0] * bary_coord[0])
        wp.atomic_add(hessian_diags, face[1], hess * bary_coord[1] * bary_coord[1])
        wp.atomic_add(hessian_diags, face[2], hess * bary_coord[2] * bary_coord[2])

        is_collided = 1

    if is_collided != 0:
        wp.atomic_add(forces, edge[0], force0)
        wp.atomic_add(forces, edge[1], force1)
        wp.atomic_add(hessian_diags, edge[0], hess0)
        wp.atomic_add(hessian_diags, edge[1], hess1)
