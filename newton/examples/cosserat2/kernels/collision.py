# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Collision detection and response kernels for Cosserat rod simulations.

Contains ground plane collision and BVH-based particle-triangle collision.
"""

import warp as wp

from newton._src.geometry.kernels import triangle_closest_point


@wp.kernel
def solve_ground_collision_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    ground_level: float,
    # output
    particle_q_out: wp.array(dtype=wp.vec3),
):
    """Simple ground plane collision constraint.

    Pushes particles above the ground plane if they penetrate.

    Args:
        particle_q: Current particle positions.
        particle_inv_mass: Inverse mass per particle (0 = kinematic).
        particle_radius: Radius per particle.
        ground_level: Z-coordinate of the ground plane.
        particle_q_out: Output corrected positions.
    """
    tid = wp.tid()

    inv_mass = particle_inv_mass[tid]
    pos = particle_q[tid]

    if inv_mass == 0.0:
        particle_q_out[tid] = pos
        return

    radius = particle_radius[tid]
    min_z = ground_level + radius
    penetration = min_z - pos[2]

    if penetration > 0.0:
        particle_q_out[tid] = wp.vec3(pos[0], pos[1], min_z)
    else:
        particle_q_out[tid] = pos


@wp.kernel
def compute_static_tri_aabbs_kernel(
    tri_vertices: wp.array(dtype=wp.vec3f),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    # outputs
    lower_bounds: wp.array(dtype=wp.vec3f),
    upper_bounds: wp.array(dtype=wp.vec3f),
):
    """Compute axis-aligned bounding boxes for triangles in the mesh.

    Args:
        tri_vertices: Array of vertex positions.
        tri_indices: Triangle indices (num_triangles x 3).
        lower_bounds: Output lower bounds of each triangle AABB.
        upper_bounds: Output upper bounds of each triangle AABB.
    """
    tid = wp.tid()

    i0 = tri_indices[tid, 0]
    i1 = tri_indices[tid, 1]
    i2 = tri_indices[tid, 2]

    v0 = tri_vertices[i0]
    v1 = tri_vertices[i1]
    v2 = tri_vertices[i2]

    lower = wp.vec3f(
        wp.min(wp.min(v0[0], v1[0]), v2[0]),
        wp.min(wp.min(v0[1], v1[1]), v2[1]),
        wp.min(wp.min(v0[2], v1[2]), v2[2]),
    )
    upper = wp.vec3f(
        wp.max(wp.max(v0[0], v1[0]), v2[0]),
        wp.max(wp.max(v0[1], v1[1]), v2[1]),
        wp.max(wp.max(v0[2], v1[2]), v2[2]),
    )

    lower_bounds[tid] = lower
    upper_bounds[tid] = upper


@wp.func
def compute_triangle_normal(v0: wp.vec3f, v1: wp.vec3f, v2: wp.vec3f) -> wp.vec3f:
    """Compute the normal vector of a triangle from three vertices.

    Args:
        v0: First vertex of the triangle.
        v1: Second vertex of the triangle.
        v2: Third vertex of the triangle.

    Returns:
        Normalized normal vector of the triangle.
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = wp.cross(edge1, edge2)
    length = wp.length(normal)
    if length > 1e-8:
        return normal / length
    return wp.vec3f(0.0, 1.0, 0.0)  # fallback


@wp.kernel
def collide_particles_vs_triangles_bvh_kernel(
    particle_q: wp.array(dtype=wp.vec3f),
    particle_radius: wp.array(dtype=wp.float32),
    particle_inv_mass: wp.array(dtype=wp.float32),
    tri_vertices: wp.array(dtype=wp.vec3f),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    bvh_id: wp.uint64,
    use_gauss_seidel: wp.bool,
    # outputs
    particle_q_out: wp.array(dtype=wp.vec3f),
):
    """Particle vs triangles collision kernel using BVH broadphase and PBD response.

    For each particle:
    1. Query BVH for triangles within particle radius (broadphase)
    2. For each candidate triangle, compute closest point (narrowphase)
    3. If penetrating, compute and apply position correction (PBD response)

    This is much more efficient than brute-force O(particles * triangles) for large meshes.

    Args:
        particle_q: Current particle positions.
        particle_radius: Radius of each particle.
        particle_inv_mass: Inverse mass of each particle (0 = kinematic/static).
        tri_vertices: Vertex positions of the triangle mesh.
        tri_indices: Triangle indices (num_triangles x 3).
        bvh_id: BVH identifier for broadphase queries.
        use_gauss_seidel: Apply per-triangle corrections immediately when true.
        particle_q_out: Output particle positions after collision response.
    """
    tid = wp.tid()

    inv_mass = particle_inv_mass[tid]

    # Skip kinematic particles
    if inv_mass <= 0.0:
        particle_q_out[tid] = particle_q[tid]
        return

    pos = particle_q[tid]
    radius = particle_radius[tid]

    # Query BVH for triangles within particle's bounding sphere
    query_margin = radius * 1.5  # Small margin for robustness
    lower = wp.vec3f(
        pos[0] - query_margin,
        pos[1] - query_margin,
        pos[2] - query_margin,
    )
    upper = wp.vec3f(
        pos[0] + query_margin,
        pos[1] + query_margin,
        pos[2] + query_margin,
    )

    # Broadphase: query BVH for potentially colliding triangles
    query = wp.bvh_query_aabb(bvh_id, lower, upper)
    tri_idx = wp.int32(0)

    if use_gauss_seidel:
        # Gauss-Seidel: apply each correction immediately to the local position.
        pos_local = pos
        while wp.bvh_query_next(query, tri_idx):
            # Get triangle vertex indices
            i0 = tri_indices[tri_idx, 0]
            i1 = tri_indices[tri_idx, 1]
            i2 = tri_indices[tri_idx, 2]

            # Get triangle vertex positions
            v0 = tri_vertices[i0]
            v1 = tri_vertices[i1]
            v2 = tri_vertices[i2]

            # Narrowphase: find closest point on triangle to particle center
            closest_p, bary, feature_type = triangle_closest_point(v0, v1, v2, pos_local)

            # Compute distance from particle center to closest point
            to_particle = pos_local - closest_p
            dist = wp.length(to_particle)

            # Check for penetration
            penetration = radius - dist

            if penetration > 0.0:
                # Compute correction direction (push particle away from triangle)
                if dist > 1e-8:
                    correction_dir = to_particle / dist  # Normalized direction from triangle to particle
                else:
                    # Particle center is on or very close to the triangle surface
                    # Use triangle normal as fallback
                    correction_dir = compute_triangle_normal(v0, v1, v2)

                    # Make sure normal points towards particle (away from triangle)
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    tri_norm = wp.cross(edge1, edge2)
                    if wp.dot(tri_norm, pos_local - v0) < 0.0:
                        correction_dir = -correction_dir

                # Compute position correction to resolve penetration (PBD stiffness = 1)
                correction = correction_dir * penetration
                pos_local = pos_local + correction

        particle_q_out[tid] = pos_local
    else:
        # Jacobi: accumulate corrections and apply the average at the end.
        total_correction = wp.vec3f(0.0, 0.0, 0.0)
        num_collisions = wp.int32(0)

        while wp.bvh_query_next(query, tri_idx):
            # Get triangle vertex indices
            i0 = tri_indices[tri_idx, 0]
            i1 = tri_indices[tri_idx, 1]
            i2 = tri_indices[tri_idx, 2]

            # Get triangle vertex positions
            v0 = tri_vertices[i0]
            v1 = tri_vertices[i1]
            v2 = tri_vertices[i2]

            # Narrowphase: find closest point on triangle to particle center
            closest_p, bary, feature_type = triangle_closest_point(v0, v1, v2, pos)

            # Compute distance from particle center to closest point
            to_particle = pos - closest_p
            dist = wp.length(to_particle)

            # Check for penetration
            penetration = radius - dist

            if penetration > 0.0:
                # Compute correction direction (push particle away from triangle)
                if dist > 1e-8:
                    correction_dir = to_particle / dist  # Normalized direction from triangle to particle
                else:
                    # Particle center is on or very close to the triangle surface
                    # Use triangle normal as fallback
                    correction_dir = compute_triangle_normal(v0, v1, v2)

                    # Make sure normal points towards particle (away from triangle)
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    tri_norm = wp.cross(edge1, edge2)
                    if wp.dot(tri_norm, pos - v0) < 0.0:
                        correction_dir = -correction_dir

                # Compute position correction to resolve penetration (PBD stiffness = 1)
                correction = correction_dir * penetration

                total_correction = total_correction + correction
                num_collisions = num_collisions + 1

        # Average corrections if multiple collisions (Jacobi-style)
        if num_collisions > 0:
            avg_correction = total_correction / wp.float32(num_collisions)
            particle_q_out[tid] = pos + avg_correction
        else:
            particle_q_out[tid] = pos


