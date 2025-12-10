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

from typing import Any

import warp as wp

from ..geometry.collision_core import (
    build_pair_key2,
)
from ..geometry.contact_data import ContactData
from ..geometry.sdf_utils import SDFData

# Handle both direct execution and module import
from .contact_reduction import (
    ContactReductionFunctions,
    ContactStruct,
    get_shared_memory_pointer_block_dim_plus_2_ints,
    synchronize,
)


@wp.func
def sample_sdf_using_mesh(
    mesh_id: wp.uint64,
    world_pos: wp.vec3,
    max_dist: float,
) -> float:
    """
    Sample signed distance to mesh surface using mesh query.

    Uses wp.mesh_query_point_sign_normal to find the closest point on the mesh
    and compute the signed distance. This is compatible with the return type of
    sample_sdf_extrapolated.

    Args:
        mesh_id: The mesh ID (from wp.Mesh.id)
        world_pos: Query position in mesh local coordinates
        max_dist: Maximum distance to search for closest point

    Returns:
        The signed distance value (negative inside, positive outside)
    """
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    res = wp.mesh_query_point_sign_normal(mesh_id, world_pos, max_dist, sign, face_index, face_u, face_v)

    if res:
        closest = wp.mesh_eval_position(mesh_id, face_index, face_u, face_v)
        return wp.length(world_pos - closest) * sign

    return max_dist


@wp.func
def sample_sdf_grad_using_mesh(
    mesh_id: wp.uint64,
    world_pos: wp.vec3,
    max_dist: float,
) -> tuple[float, wp.vec3]:
    """
    Sample signed distance and gradient to mesh surface using mesh query.

    Uses wp.mesh_query_point_sign_normal to find the closest point on the mesh
    and compute both the signed distance and the gradient direction. This is
    compatible with the return type of sample_sdf_grad_extrapolated.

    The gradient points in the direction of increasing distance (away from the surface
    when outside, toward the surface when inside).

    Args:
        mesh_id: The mesh ID (from wp.Mesh.id)
        world_pos: Query position in mesh local coordinates
        max_dist: Maximum distance to search for closest point

    Returns:
        Tuple of (distance, gradient) where:
        - distance: Signed distance value (negative inside, positive outside)
        - gradient: Normalized direction of increasing distance
    """
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    gradient = wp.vec3(0.0, 0.0, 0.0)

    res = wp.mesh_query_point_sign_normal(mesh_id, world_pos, max_dist, sign, face_index, face_u, face_v)

    if res:
        closest = wp.mesh_eval_position(mesh_id, face_index, face_u, face_v)
        diff = world_pos - closest
        dist = wp.length(diff)

        if dist > 0.0:
            # Gradient points from surface toward query point, scaled by sign
            # When outside (sign > 0): gradient points away from surface (correct for SDF)
            # When inside (sign < 0): gradient points toward surface (correct for SDF)
            gradient = (diff / dist) * sign
        else:
            # Point is exactly on surface - use face normal
            # Get the face normal from the mesh
            mesh = wp.mesh_get(mesh_id)
            i0 = mesh.indices[face_index * 3 + 0]
            i1 = mesh.indices[face_index * 3 + 1]
            i2 = mesh.indices[face_index * 3 + 2]
            v0 = mesh.points[i0]
            v1 = mesh.points[i1]
            v2 = mesh.points[i2]
            face_normal = wp.normalize(wp.cross(v1 - v0, v2 - v0))
            gradient = face_normal * sign

        return dist * sign, gradient

    # No hit found - return max distance with arbitrary gradient
    return max_dist, wp.vec3(0.0, 0.0, 1.0)


@wp.func
def sample_sdf_extrapolated(
    sdf_data: SDFData,
    sdf_pos: wp.vec3,
) -> float:
    """
    Sample SDF with extrapolation for points outside the narrow band or extent.

    This function handles three cases:
    1. Point in narrow band: Returns sparse grid value directly
    2. Point inside extent but outside narrow band: Returns coarse grid value
    3. Point outside extent: Projects to boundary, returns value at boundary + distance to boundary

    Args:
        sdf_data: SDFData struct containing sparse/coarse volumes and extent info
        sdf_pos: Query position in the SDF's local coordinate space

    Returns:
        The signed distance value, extrapolated if necessary
    """
    # Compute extent bounds
    lower = sdf_data.center - sdf_data.half_extents
    upper = sdf_data.center + sdf_data.half_extents

    # Check if point is inside extent
    inside_extent = (
        sdf_pos[0] >= lower[0]
        and sdf_pos[0] <= upper[0]
        and sdf_pos[1] >= lower[1]
        and sdf_pos[1] <= upper[1]
        and sdf_pos[2] >= lower[2]
        and sdf_pos[2] <= upper[2]
    )

    if inside_extent:
        # Sample sparse grid
        sparse_idx = wp.volume_world_to_index(sdf_data.sparse_sdf_ptr, sdf_pos)
        sparse_dist = wp.volume_sample_f(sdf_data.sparse_sdf_ptr, sparse_idx, wp.Volume.LINEAR)

        # Check if we got the background value (outside narrow band)
        # Use a tolerance since we're comparing floats
        background_threshold = sdf_data.background_value * 0.5
        if sparse_dist >= background_threshold:
            # Fallback to coarse grid
            coarse_idx = wp.volume_world_to_index(sdf_data.coarse_sdf_ptr, sdf_pos)
            return wp.volume_sample_f(sdf_data.coarse_sdf_ptr, coarse_idx, wp.Volume.LINEAR)
        else:
            return sparse_dist
    else:
        # Point is outside extent - project to boundary
        clamped_pos = wp.min(wp.max(sdf_pos, lower), upper)
        dist_to_boundary = wp.length(sdf_pos - clamped_pos)

        # Sample at the boundary point using coarse grid (more reliable for extrapolation)
        coarse_idx = wp.volume_world_to_index(sdf_data.coarse_sdf_ptr, clamped_pos)
        boundary_dist = wp.volume_sample_f(sdf_data.coarse_sdf_ptr, coarse_idx, wp.Volume.LINEAR)

        # Extrapolate: value at boundary + distance to boundary
        return boundary_dist + dist_to_boundary


@wp.func
def sample_sdf_grad_extrapolated(
    sdf_data: SDFData,
    sdf_pos: wp.vec3,
) -> tuple[float, wp.vec3]:
    """
    Sample SDF with gradient, with extrapolation for points outside narrow band or extent.

    This function handles three cases:
    1. Point in narrow band: Returns sparse grid value and gradient directly
    2. Point inside extent but outside narrow band: Returns coarse grid value and gradient
    3. Point outside extent: Returns extrapolated distance and direction toward boundary

    Args:
        sdf_data: SDFData struct containing sparse/coarse volumes and extent info
        sdf_pos: Query position in the SDF's local coordinate space

    Returns:
        Tuple of (distance, gradient) where gradient points toward increasing distance
    """
    # Compute extent bounds
    lower = sdf_data.center - sdf_data.half_extents
    upper = sdf_data.center + sdf_data.half_extents

    gradient = wp.vec3(0.0, 0.0, 0.0)

    # Check if point is inside extent
    inside_extent = (
        sdf_pos[0] >= lower[0]
        and sdf_pos[0] <= upper[0]
        and sdf_pos[1] >= lower[1]
        and sdf_pos[1] <= upper[1]
        and sdf_pos[2] >= lower[2]
        and sdf_pos[2] <= upper[2]
    )

    if inside_extent:
        # Sample sparse grid
        sparse_idx = wp.volume_world_to_index(sdf_data.sparse_sdf_ptr, sdf_pos)
        sparse_dist = wp.volume_sample_grad_f(sdf_data.sparse_sdf_ptr, sparse_idx, wp.Volume.LINEAR, gradient)

        # Check if we got the background value (outside narrow band)
        background_threshold = sdf_data.background_value * 0.5
        if sparse_dist >= background_threshold:
            # Fallback to coarse grid
            coarse_idx = wp.volume_world_to_index(sdf_data.coarse_sdf_ptr, sdf_pos)
            coarse_dist = wp.volume_sample_grad_f(sdf_data.coarse_sdf_ptr, coarse_idx, wp.Volume.LINEAR, gradient)
            return coarse_dist, gradient
        else:
            return sparse_dist, gradient
    else:
        # Point is outside extent - project to boundary
        clamped_pos = wp.min(wp.max(sdf_pos, lower), upper)
        diff = sdf_pos - clamped_pos
        dist_to_boundary = wp.length(diff)

        # Sample at the boundary point using coarse grid
        coarse_idx = wp.volume_world_to_index(sdf_data.coarse_sdf_ptr, clamped_pos)
        boundary_dist = wp.volume_sample_f(sdf_data.coarse_sdf_ptr, coarse_idx, wp.Volume.LINEAR)

        # Extrapolate distance: value at boundary + distance to boundary
        extrapolated_dist = boundary_dist + dist_to_boundary

        # Gradient points from boundary toward the query point (direction of increasing distance)
        if dist_to_boundary > 0.0:
            gradient = diff / dist_to_boundary
        else:
            # Fallback: get gradient from coarse grid
            wp.volume_sample_grad_f(sdf_data.coarse_sdf_ptr, coarse_idx, wp.Volume.LINEAR, gradient)

        return extrapolated_dist, gradient


@wp.func
def closest_pt_point_bary_triangle(c: wp.vec3) -> wp.vec3:
    """
    Find the closest point to `c` on the standard barycentric triangle.

    This function projects a barycentric coordinate point onto the valid barycentric
    triangle defined by vertices (1,0,0), (0,1,0), (0,0,1) in barycentric space.
    The valid region is where all coordinates are non-negative and sum to 1.

    This is a specialized version of the general closest-point-on-triangle algorithm
    optimized for the barycentric simplex.

    Args:
        c: Input barycentric coordinates (may be outside valid triangle region)

    Returns:
        The closest valid barycentric coordinates. All components will be >= 0
        and sum to 1.0.

    Note:
        This is used in optimization algorithms that work in barycentric space,
        where gradient descent may produce invalid coordinates that need projection.
    """
    third = 1.0 / 3.0  # constexpr
    c = c - wp.vec3(third * (c[0] + c[1] + c[2] - 1.0))

    # two negative: return positive vertex
    if c[1] < 0.0 and c[2] < 0.0:
        return wp.vec3(1.0, 0.0, 0.0)

    if c[0] < 0.0 and c[2] < 0.0:
        return wp.vec3(0.0, 1.0, 0.0)

    if c[0] < 0.0 and c[1] < 0.0:
        return wp.vec3(0.0, 0.0, 1.0)

    # one negative: return projection onto line if it is on the edge, or the largest vertex otherwise
    if c[0] < 0.0:
        d = c[0] * 0.5
        y = c[1] + d
        z = c[2] + d
        if y > 1.0:
            return wp.vec3(0.0, 1.0, 0.0)
        if z > 1.0:
            return wp.vec3(0.0, 0.0, 1.0)
        return wp.vec3(0.0, y, z)
    if c[1] < 0.0:
        d = c[1] * 0.5
        x = c[0] + d
        z = c[2] + d
        if x > 1.0:
            return wp.vec3(1.0, 0.0, 0.0)
        if z > 1.0:
            return wp.vec3(0.0, 0.0, 1.0)
        return wp.vec3(x, 0.0, z)
    if c[2] < 0.0:
        d = c[2] * 0.5
        x = c[0] + d
        y = c[1] + d
        if x > 1.0:
            return wp.vec3(1.0, 0.0, 0.0)
        if y > 1.0:
            return wp.vec3(0.0, 1.0, 0.0)
        return wp.vec3(x, y, 0.0)
    return c


@wp.func
def do_triangle_sdf_collision(
    sdf_data: SDFData,
    sdf_mesh_id: wp.uint64,
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
    use_bvh_for_sdf: bool,
) -> tuple[float, wp.vec3, wp.vec3]:
    """
    Compute the deepest contact between a triangle and an SDF volume.

    This function uses gradient descent in barycentric coordinates to find the point
    on the triangle that has the minimum (most negative) signed distance to the SDF.
    The optimization starts from either the triangle centroid or one of its vertices
    (whichever has the smallest initial distance).

    Uses extrapolated SDF sampling that handles:
    - Points in narrow band: sparse grid value
    - Points inside extent but outside narrow band: coarse grid value
    - Points outside extent: extrapolated from boundary

    Algorithm:
    1. Evaluate SDF distance at triangle vertices and centroid
    2. Start from the point with minimum distance
    3. Iterate up to 16 times:
       - Compute SDF gradient at current point
       - Project gradient onto triangle edges (in barycentric space)
       - Take a gradient descent step with decreasing step size
       - Project result back onto valid barycentric triangle
    4. Return final distance, contact point, and contact direction

    Args:
        sdf_data: SDFData struct containing sparse/coarse volumes and extent info
        sdf_mesh_id: Mesh ID for BVH-based collision (used when use_bvh_for_sdf is True)
        v0, v1, v2: Triangle vertices in the SDF's local coordinate space
        use_bvh_for_sdf: If True, use BVH-based collision instead of SDF volumes

    Returns:
        Tuple of (distance, contact_point, contact_direction) where:
        - distance: Signed distance to SDF surface (negative = penetration)
        - contact_point: The point on the triangle closest to the SDF surface
        - contact_direction: Normalized direction from surface to contact point
    """
    third = 1.0 / 3.0
    center = (v0 + v1 + v2) * third
    p = center

    # Use extrapolated sampling for initial distance estimates
    max_dist = 1000.0
    if use_bvh_for_sdf:
        dist = sample_sdf_using_mesh(sdf_mesh_id, p, max_dist)
        d0 = sample_sdf_using_mesh(sdf_mesh_id, v0, max_dist)
        d1 = sample_sdf_using_mesh(sdf_mesh_id, v1, max_dist)
        d2 = sample_sdf_using_mesh(sdf_mesh_id, v2, max_dist)
    else:
        dist = sample_sdf_extrapolated(sdf_data, p)
        d0 = sample_sdf_extrapolated(sdf_data, v0)
        d1 = sample_sdf_extrapolated(sdf_data, v1)
        d2 = sample_sdf_extrapolated(sdf_data, v2)

    # choose starting iterate among centroid and triangle vertices
    if d0 < d1 and d0 < d2 and d0 < dist:
        p = v0
        uvw = wp.vec3(1.0, 0.0, 0.0)
    elif d1 < d2 and d1 < dist:
        p = v1
        uvw = wp.vec3(0.0, 1.0, 0.0)
    elif d2 < dist:
        p = v2
        uvw = wp.vec3(0.0, 0.0, 1.0)
    else:
        uvw = wp.vec3(third, third, third)

    difference = wp.sqrt(
        wp.max(
            wp.length_sq(v0 - p),
            wp.max(wp.length_sq(v1 - p), wp.length_sq(v2 - p)),
        )
    )

    difference = wp.max(difference, 1e-8)

    tolerance_sq = 1e-3 * 1e-3

    sdf_gradient = wp.vec3(0.0, 0.0, 0.0)
    step = 1.0 / (2.0 * difference)

    for _iter in range(16):
        # Use extrapolated gradient sampling
        if use_bvh_for_sdf:
            _, sdf_gradient = sample_sdf_grad_using_mesh(sdf_mesh_id, p, max_dist)
        else:
            _, sdf_gradient = sample_sdf_grad_extrapolated(sdf_data, p)

        grad_len = wp.length(sdf_gradient)
        if grad_len == 0.0:
            # We ran into a discontinuity e.g. the exact center of a cube
            # Just pick an arbitrary gradient of unit length to move out of the discontinuity
            sdf_gradient = wp.vec3(0.571846586, 0.705545099, 0.418566116)
            grad_len = 1.0

        sdf_gradient = sdf_gradient / grad_len

        dfdu = wp.dot(sdf_gradient, v0 - p)
        dfdv = wp.dot(sdf_gradient, v1 - p)
        dfdw = wp.dot(sdf_gradient, v2 - p)

        new_uvw = uvw

        new_uvw = wp.vec3(new_uvw[0] - step * dfdu, new_uvw[1] - step * dfdv, new_uvw[2] - step * dfdw)

        step = step * 0.8

        new_uvw = closest_pt_point_bary_triangle(new_uvw)

        p = v0 * new_uvw[0] + v1 * new_uvw[1] + v2 * new_uvw[2]

        if wp.length_sq(uvw - new_uvw) < tolerance_sq:
            break

        uvw = new_uvw

    # Final extrapolated sampling for result
    if use_bvh_for_sdf:
        dist, sdf_gradient = sample_sdf_grad_using_mesh(sdf_mesh_id, p, max_dist)
    else:
        dist, sdf_gradient = sample_sdf_grad_extrapolated(sdf_data, p)

    point = p
    direction = sdf_gradient

    return dist, point, direction


@wp.func
def get_triangle_from_mesh(
    mesh_id: wp.uint64,
    mesh_scale: wp.vec3,
    X_mesh_ws: wp.transform,
    tri_idx: int,
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    """
    Extract a triangle from a mesh and transform it to world space.

    This function retrieves a specific triangle from a mesh by its index,
    applies scaling and transformation, and returns the three vertices
    in world space coordinates.

    Args:
        mesh_id: The mesh ID (use wp.mesh_get to retrieve the mesh object)
        mesh_scale: Scale to apply to mesh vertices (component-wise)
        X_mesh_ws: Mesh world-space transform (position and rotation)
        tri_idx: Triangle index in the mesh (0-based)

    Returns:
        Tuple of (v0_world, v1_world, v2_world) - the three triangle vertices
        in world space after applying scale and transform.

    Note:
        The mesh indices array stores triangle vertex indices as a flat array:
        [tri0_v0, tri0_v1, tri0_v2, tri1_v0, tri1_v1, tri1_v2, ...]
    """

    mesh = wp.mesh_get(mesh_id)

    # Extract triangle vertices from mesh (indices are stored as flat array: i0, i1, i2, i0, i1, i2, ...)
    idx0 = mesh.indices[tri_idx * 3 + 0]
    idx1 = mesh.indices[tri_idx * 3 + 1]
    idx2 = mesh.indices[tri_idx * 3 + 2]

    # Get vertex positions in mesh local space (with scale applied)
    v0_local = wp.cw_mul(mesh.points[idx0], mesh_scale)
    v1_local = wp.cw_mul(mesh.points[idx1], mesh_scale)
    v2_local = wp.cw_mul(mesh.points[idx2], mesh_scale)

    # Transform vertices to world space
    v0_world = wp.transform_point(X_mesh_ws, v0_local)
    v1_world = wp.transform_point(X_mesh_ws, v1_local)
    v2_world = wp.transform_point(X_mesh_ws, v2_local)

    return v0_world, v1_world, v2_world


@wp.func
def get_bounding_sphere(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3) -> tuple[wp.vec3, float]:
    """
    Compute a conservative bounding sphere for a triangle.

    This uses the triangle centroid as the sphere center and the maximum
    distance from the centroid to any vertex as the radius. This is a
    conservative (potentially larger than optimal) but fast bounding sphere.

    Args:
        v0, v1, v2: Triangle vertices in world space

    Returns:
        Tuple of (center, radius) where:
        - center: The centroid of the triangle
        - radius: The maximum distance from centroid to any vertex

    Note:
        This is not the minimal bounding sphere, but it's fast to compute
        and adequate for broad-phase culling.
    """
    center = (v0 + v1 + v2) * (1.0 / 3.0)
    radius = wp.max(wp.max(wp.length_sq(v0 - center), wp.length_sq(v1 - center)), wp.length_sq(v2 - center))
    return center, wp.sqrt(radius)


@wp.func
def add_to_shared_buffer_atomic(
    thread_id: int,
    add_triangle: bool,
    tri_idx: int,
    buffer: wp.array(dtype=wp.int32),
):
    """
    Add a triangle index to a shared memory buffer using atomic operations.

    Buffer layout:
    - [0 .. block_dim-1]: Triangle indices
    - [block_dim]: Current count of triangles in buffer
    - [block_dim+1]: Progress counter (triangles processed so far)

    Args:
        thread_id: The calling thread's index within the thread block
        add_triangle: Whether this thread wants to add a triangle
        tri_idx: The triangle index to add (only used if add_triangle is True)
        buffer: Shared memory buffer for triangle indices
    """
    capacity = wp.block_dim()
    idx = -1

    # Atomic add to get write position
    if add_triangle:
        idx = wp.atomic_add(buffer, capacity, 1)
        if idx < capacity:
            buffer[idx] = tri_idx

    # Thread 0 optimistically advances progress by block_dim
    if thread_id == 0:
        buffer[capacity + 1] += capacity

    synchronize()  # SYNC 1: All atomic writes and progress update complete

    # Cap count at capacity (in case of overflow)
    if thread_id == 0 and buffer[capacity] > capacity:
        buffer[capacity] = capacity

    # Overflow threads correct progress to their tri_idx (minimum wins)
    if add_triangle and idx >= capacity:
        wp.atomic_min(buffer, capacity + 1, tri_idx)

    synchronize()  # SYNC 2: All corrections complete, buffer consistent


@wp.func
def find_interesting_triangles(
    thread_id: int,
    mesh_scale: wp.vec3,
    mesh_to_sdf_transform: wp.transform,
    mesh_id: wp.uint64,
    sdf_data: SDFData,
    sdf_mesh_id: wp.uint64,
    buffer: wp.array(dtype=wp.int32),
    contact_distance: float,
    use_bvh_for_sdf: bool,
):
    """
    Midphase triangle culling for mesh-SDF collision.

    Given a mesh-mesh pair (already identified by broad phase), this function determines
    which triangles from one mesh are close enough to the other mesh's SDF to potentially
    generate contacts. This is an intermediate step between broad phase (which identifies
    colliding object pairs) and narrow phase (which computes exact contact points).

    Each triangle's bounding sphere is tested against the SDF. A triangle is selected if:
        SDF_distance(sphere_center) <= sphere_radius + contact_distance

    Selected triangle indices are stored in a shared memory buffer for subsequent
    narrow-phase contact computation.

    Buffer layout: [0..block_dim-1] = triangle indices, [block_dim] = count, [block_dim+1] = progress
    """
    num_tris = wp.mesh_get(mesh_id).indices.shape[0] // 3
    capacity = wp.block_dim()

    synchronize()  # Ensure buffer state is consistent before starting

    while buffer[capacity + 1] < num_tris and buffer[capacity] < capacity:
        # All threads read the same base index (buffer consistent from previous sync)
        base_tri_idx = buffer[capacity + 1]
        tri_idx = base_tri_idx + thread_id
        add_triangle = False

        if tri_idx < num_tris:
            v0, v1, v2 = get_triangle_from_mesh(mesh_id, mesh_scale, mesh_to_sdf_transform, tri_idx)
            bounding_sphere_center, bounding_sphere_radius = get_bounding_sphere(v0, v1, v2)

            # Use extrapolated SDF distance query for culling
            if use_bvh_for_sdf:
                sdf_dist = sample_sdf_using_mesh(
                    sdf_mesh_id, bounding_sphere_center, 1.01 * (bounding_sphere_radius + contact_distance)
                )
            else:
                sdf_dist = sample_sdf_extrapolated(sdf_data, bounding_sphere_center)
            add_triangle = sdf_dist <= (bounding_sphere_radius + contact_distance)

        synchronize()  # Ensure all threads have read base_tri_idx before any writes
        add_to_shared_buffer_atomic(thread_id, add_triangle, tri_idx, buffer)
        # add_to_shared_buffer_atomic ends with sync, buffer is consistent for next while check

    synchronize()  # Final sync before returning


def create_narrow_phase_process_mesh_mesh_contacts_kernel(
    writer_func: Any,
    contact_reduction_funcs: ContactReductionFunctions | None = None,
):
    @wp.kernel(enable_backward=False)
    def mesh_sdf_collision_kernel(
        shape_data: wp.array(dtype=wp.vec4),
        shape_transform: wp.array(dtype=wp.transform),
        shape_source: wp.array(dtype=wp.uint64),
        shape_sdf_data: wp.array(dtype=SDFData),
        shape_contact_margin: wp.array(dtype=float),
        shape_pairs_mesh_mesh: wp.array(dtype=wp.vec2i),
        shape_pairs_mesh_mesh_count: wp.array(dtype=int),
        betas: wp.array(dtype=wp.float32),  # Unused, kept for API compatibility
        writer_data: Any,
        total_num_blocks: int,
    ):
        """
        Process mesh-mesh collisions using SDF-mesh collision detection.

        Uses a strided loop to process mesh-mesh pairs, with threads within each block
        parallelizing over triangles. This follows the pattern from do_sdf_mesh_collision.

        Args:
            geom_types: Array of geometry types for all shapes
            geom_data: Array of vec4 containing scale (xyz) and thickness (w) for each shape
            geom_transform: Array of world-space transforms for each shape
            geom_source: Array of source pointers (mesh IDs) for each shape
            shape_sdf_data: Array of SDFData structs for mesh shapes
            geom_cutoff: Array of cutoff distances for each shape
            shape_pairs_mesh_mesh: Array of mesh-mesh pairs to process
            shape_pairs_mesh_mesh_count: Number of mesh-mesh pairs
            writer_data: Contact writer data structure
            total_num_blocks: Total number of blocks launched for strided loop
        """
        block_id, t = wp.tid()

        num_pairs = shape_pairs_mesh_mesh_count[0]

        # Strided loop over pairs
        for pair_idx in range(block_id, num_pairs, total_num_blocks):
            pair = shape_pairs_mesh_mesh[pair_idx]
            mesh_shape_a = pair[0]
            mesh_shape_b = pair[1]

            # Get mesh and SDF IDs
            mesh_id_a = shape_source[mesh_shape_a]
            mesh_id_b = shape_source[mesh_shape_b]

            # Get SDF pointers - a value of 0 indicates no SDF is available for this shape
            sdf_ptr_a = shape_sdf_data[mesh_shape_a].sparse_sdf_ptr
            sdf_ptr_b = shape_sdf_data[mesh_shape_b].sparse_sdf_ptr

            # Determine if we should use BVH (mesh queries) instead of SDF for each shape
            # sparse_sdf_ptr == 0 means no SDF is initialized for this shape, use BVH instead
            use_bvh_for_sdf_a = sdf_ptr_a == wp.uint64(0)
            use_bvh_for_sdf_b = sdf_ptr_b == wp.uint64(0)

            # Skip if either mesh is invalid
            if mesh_id_a == wp.uint64(0) or mesh_id_b == wp.uint64(0):
                continue

            # Get mesh objects
            mesh_a = wp.mesh_get(mesh_id_a)
            mesh_b = wp.mesh_get(mesh_id_b)

            # Get mesh scales and transforms
            scale_data_a = shape_data[mesh_shape_a]
            scale_data_b = shape_data[mesh_shape_b]
            mesh_scale_a = wp.vec3(scale_data_a[0], scale_data_a[1], scale_data_a[2])
            mesh_scale_b = wp.vec3(scale_data_b[0], scale_data_b[1], scale_data_b[2])

            X_mesh_a_ws = shape_transform[mesh_shape_a]
            X_mesh_b_ws = shape_transform[mesh_shape_b]

            # Get thickness values
            thickness_a = shape_data[mesh_shape_a][3]
            thickness_b = shape_data[mesh_shape_b][3]

            # Use per-geometry cutoff for contact detection
            cutoff_a = shape_contact_margin[mesh_shape_a]
            cutoff_b = shape_contact_margin[mesh_shape_b]
            margin = wp.max(cutoff_a, cutoff_b)

            # Build pair key for this mesh-mesh pair
            pair_key = build_pair_key2(wp.uint32(mesh_shape_a), wp.uint32(mesh_shape_b))

            # Test both directions: mesh A against SDF B, and mesh B against SDF A
            for mode in range(2):
                # Initialize with dummy values (will be set in mode branches)
                sdf_data = SDFData()
                use_bvh_for_sdf = False

                if mode == 0:
                    # Process mesh A triangles against SDF B
                    # Skip if no SDF available and no BVH fallback possible
                    use_bvh_for_sdf = use_bvh_for_sdf_b
                    if sdf_ptr_b == wp.uint64(0) and not use_bvh_for_sdf:
                        continue

                    mesh_id = mesh_id_a
                    mesh_scale = mesh_scale_a
                    if not use_bvh_for_sdf:
                        sdf_data = shape_sdf_data[mesh_shape_b]
                    sdf_mesh_id = mesh_id_b  # SDF belongs to mesh B
                    # Transform from mesh A space to mesh B space
                    X_mesh_to_sdf = wp.transform_multiply(wp.transform_inverse(X_mesh_b_ws), X_mesh_a_ws)
                    X_sdf_ws = X_mesh_b_ws
                    mesh = mesh_a
                    triangle_mesh_thickness = thickness_a
                else:
                    # Process mesh B triangles against SDF A
                    # Skip if no SDF available and no BVH fallback possible
                    use_bvh_for_sdf = use_bvh_for_sdf_a
                    if sdf_ptr_a == wp.uint64(0) and not use_bvh_for_sdf:
                        continue

                    mesh_id = mesh_id_b
                    mesh_scale = mesh_scale_b
                    if not use_bvh_for_sdf:
                        sdf_data = shape_sdf_data[mesh_shape_a]
                    sdf_mesh_id = mesh_id_a  # SDF belongs to mesh A
                    # Transform from mesh B space to mesh A space
                    X_mesh_to_sdf = wp.transform_multiply(wp.transform_inverse(X_mesh_a_ws), X_mesh_b_ws)
                    X_sdf_ws = X_mesh_a_ws
                    mesh = mesh_b
                    triangle_mesh_thickness = thickness_b

                # (SDF mesh's thickness is already baked into the SDF)
                contact_threshold = margin + triangle_mesh_thickness

                num_tris = mesh.indices.shape[0] // 3
                # strided loop over triangles
                for tri_idx in range(t, num_tris, wp.block_dim()):
                    # Get triangle vertices in SDF's local space
                    v0, v1, v2 = get_triangle_from_mesh(mesh_id, mesh_scale, X_mesh_to_sdf, tri_idx)

                    # Early out: check bounding sphere distance to SDF surface using extrapolated sampling
                    bounding_sphere_center, bounding_sphere_radius = get_bounding_sphere(v0, v1, v2)
                    if use_bvh_for_sdf:
                        sdf_dist = sample_sdf_using_mesh(
                            sdf_mesh_id, bounding_sphere_center, 1.01 * (bounding_sphere_radius + contact_threshold)
                        )
                    else:
                        sdf_dist = sample_sdf_extrapolated(sdf_data, bounding_sphere_center)

                    # Skip triangles that are too far from the SDF surface
                    if sdf_dist > (bounding_sphere_radius + contact_threshold):
                        continue

                    dist, point, direction = do_triangle_sdf_collision(
                        sdf_data, sdf_mesh_id, v0, v1, v2, use_bvh_for_sdf
                    )

                    if dist < contact_threshold:
                        point_world = wp.transform_point(X_sdf_ws, point)

                        direction_world = wp.transform_vector(X_sdf_ws, direction)
                        direction_len = wp.length(direction_world)
                        if direction_len > 0.0:
                            direction_world = direction_world / direction_len

                        if mode == 0:
                            contact_normal = -direction_world
                        else:
                            contact_normal = direction_world

                        # Create contact data
                        # Always use consistent pair ordering (mesh_shape_a, mesh_shape_b) regardless of mode
                        contact_data = ContactData()
                        contact_data.contact_point_center = point_world
                        contact_data.contact_normal_a_to_b = contact_normal
                        contact_data.contact_distance = dist
                        contact_data.radius_eff_a = 0.0
                        contact_data.radius_eff_b = 0.0
                        # SDF mesh's thickness is already baked into the SDF, so set it to 0
                        # Mode 0: mesh_a triangles vs mesh_b's SDF -> thickness_b already in SDF
                        # Mode 1: mesh_b triangles vs mesh_a's SDF -> thickness_a already in SDF
                        if mode == 0:
                            contact_data.thickness_a = thickness_a
                            contact_data.thickness_b = 0.0
                        else:
                            contact_data.thickness_a = 0.0
                            contact_data.thickness_b = thickness_b
                        contact_data.shape_a = mesh_shape_a
                        contact_data.shape_b = mesh_shape_b
                        contact_data.margin = margin
                        if mode == 0:
                            contact_data.feature = wp.uint32(tri_idx + 1)
                        else:
                            contact_data.feature = wp.uint32(tri_idx + 1) | wp.uint32(0x80000000)
                        contact_data.feature_pair_key = pair_key

                        writer_func(contact_data, writer_data, -1)

    # Return early if contact reduction is disabled
    if contact_reduction_funcs is None:
        return mesh_sdf_collision_kernel

    # Extract functions and constants from the contact reduction configuration
    num_reduction_slots = contact_reduction_funcs.num_reduction_slots
    store_reduced_contact_func = contact_reduction_funcs.store_reduced_contact
    filter_unique_contacts_func = contact_reduction_funcs.filter_unique_contacts
    get_smem_slots_plus_1 = contact_reduction_funcs.get_smem_slots_plus_1
    get_smem_slots_contacts = contact_reduction_funcs.get_smem_slots_contacts

    @wp.kernel(enable_backward=False)
    def mesh_sdf_collision_reduce_kernel(
        shape_data: wp.array(dtype=wp.vec4),
        shape_transform: wp.array(dtype=wp.transform),
        shape_source: wp.array(dtype=wp.uint64),
        shape_sdf_data: wp.array(dtype=SDFData),
        shape_contact_margin: wp.array(dtype=float),
        shape_pairs_mesh_mesh: wp.array(dtype=wp.vec2i),
        shape_pairs_mesh_mesh_count: wp.array(dtype=int),
        betas: wp.array(dtype=wp.float32),
        writer_data: Any,
        total_num_blocks: int,
    ):
        block_id, t = wp.tid()
        num_pairs = shape_pairs_mesh_mesh_count[0]
        if block_id >= num_pairs:
            return

        pair = shape_pairs_mesh_mesh[block_id]
        shape_idx_0 = pair[0]
        shape_idx_1 = pair[1]

        mesh0 = shape_source[shape_idx_0]
        mesh1 = shape_source[shape_idx_1]

        # Get SDF data for both shapes
        sdf_data0 = shape_sdf_data[shape_idx_0]
        sdf_data1 = shape_sdf_data[shape_idx_1]

        # Determine if we should use BVH (mesh queries) instead of SDF for each shape
        # sparse_sdf_ptr == 0 means no SDF is initialized for this shape, use BVH instead
        use_bvh_for_sdf_0 = sdf_data0.sparse_sdf_ptr == wp.uint64(0)
        use_bvh_for_sdf_1 = sdf_data1.sparse_sdf_ptr == wp.uint64(0)

        # Extract mesh parameters
        mesh0_data = shape_data[shape_idx_0]
        mesh1_data = shape_data[shape_idx_1]
        mesh0_scale = wp.vec3(mesh0_data[0], mesh0_data[1], mesh0_data[2])
        mesh1_scale = wp.vec3(mesh1_data[0], mesh1_data[1], mesh1_data[2])
        mesh0_transform = shape_transform[shape_idx_0]
        mesh1_transform = shape_transform[shape_idx_1]

        thickness0 = mesh0_data[3]
        thickness1 = mesh1_data[3]

        margin = wp.max(shape_contact_margin[shape_idx_0], shape_contact_margin[shape_idx_1])

        # Initialize (shared memory) buffers for contact reduction
        empty_marker = -1000000000.0
        has_contact = True

        active_contacts_shared_mem = wp.array(
            ptr=wp.static(get_smem_slots_plus_1)(),
            shape=(wp.static(num_reduction_slots) + 1,),
            dtype=wp.int32,
        )
        contacts_shared_mem = wp.array(
            ptr=wp.static(get_smem_slots_contacts)(),
            shape=(wp.static(num_reduction_slots),),
            dtype=ContactStruct,
        )

        for i in range(t, wp.static(num_reduction_slots), wp.block_dim()):
            contacts_shared_mem[i].projection = empty_marker

        if t == 0:
            active_contacts_shared_mem[wp.static(num_reduction_slots)] = 0

        # Initialize (shared memory) buffers for triangle selection
        tri_capacity = wp.block_dim()
        selected_triangles = wp.array(
            ptr=get_shared_memory_pointer_block_dim_plus_2_ints(), shape=(wp.block_dim() + 2,), dtype=wp.int32
        )

        if t == 0:
            selected_triangles[tri_capacity] = 0  # Stores the number of indices inside selected_triangles
            selected_triangles[tri_capacity + 1] = (
                0  # Stores the number of triangles from the mesh that were already investigated
            )

        # Compute midpoint of the two body positions for centering contacts during reduction
        # Using centered world-space coordinates ensures consistent spatial dot products
        midpoint = (wp.transform_get_translation(mesh0_transform) + wp.transform_get_translation(mesh1_transform)) * 0.5

        for mode in range(2):
            synchronize()
            # Determine use_bvh_for_sdf based on which mesh provides the SDF
            use_bvh_for_sdf = False
            if mode == 0:
                mesh = mesh0
                mesh_scale = mesh0_scale
                mesh_sdf_transform = wp.transform_multiply(wp.transform_inverse(mesh1_transform), mesh0_transform)
                sdf_data_current = sdf_data1
                sdf_mesh_id = mesh1  # SDF belongs to mesh1
                use_bvh_for_sdf = use_bvh_for_sdf_1
                triangle_mesh_thickness = thickness0
            else:
                mesh = mesh1
                mesh_scale = mesh1_scale
                mesh_sdf_transform = wp.transform_multiply(wp.transform_inverse(mesh0_transform), mesh1_transform)
                sdf_data_current = sdf_data0
                sdf_mesh_id = mesh0  # SDF belongs to mesh0
                use_bvh_for_sdf = use_bvh_for_sdf_0
                triangle_mesh_thickness = thickness1

            # Contact threshold: margin + triangle mesh's thickness
            # (SDF mesh's thickness is already baked into the SDF)
            contact_threshold = margin + triangle_mesh_thickness

            # Reset progress counter for this mesh
            if t == 0:
                selected_triangles[tri_capacity + 1] = 0
            synchronize()

            num_tris = wp.mesh_get(mesh).indices.shape[0] // 3

            has_contact = wp.bool(False)

            while selected_triangles[tri_capacity + 1] < num_tris:
                find_interesting_triangles(
                    t,
                    mesh_scale,
                    mesh_sdf_transform,
                    mesh,
                    sdf_data_current,
                    sdf_mesh_id,
                    selected_triangles,
                    contact_threshold,
                    use_bvh_for_sdf,
                )

                has_contact = t < selected_triangles[tri_capacity]
                synchronize()
                c = ContactStruct()

                if has_contact:
                    v0, v1, v2 = get_triangle_from_mesh(mesh, mesh_scale, mesh_sdf_transform, selected_triangles[t])
                    dist, point, direction = do_triangle_sdf_collision(
                        sdf_data_current,
                        sdf_mesh_id,
                        v0,
                        v1,
                        v2,
                        use_bvh_for_sdf,
                    )

                    has_contact = dist < contact_threshold

                    if has_contact:
                        # Transform contact to world space, then center by subtracting midpoint
                        # This ensures consistent spatial dot products during reduction
                        # Mode 0: contact in SDF B (mesh1) space -> transform to world
                        # Mode 1: contact in SDF A (mesh0) space -> transform to world
                        if mode == 0:
                            point_world = wp.transform_point(mesh1_transform, point)
                            normal_world = wp.transform_vector(mesh1_transform, direction)
                        else:
                            point_world = wp.transform_point(mesh0_transform, point)
                            normal_world = wp.transform_vector(mesh0_transform, direction)

                        # Center the position by subtracting midpoint (translation only)
                        point_centered = point_world - midpoint

                        normal_len = wp.length(normal_world)
                        if normal_len > 0.0:
                            normal_world = normal_world / normal_len

                        # Normalize normal direction so it always points from pair[0] to pair[1]
                        # Mode 0: gradient points B->A (pair[1]->pair[0]), negate to get pair[0]->pair[1]
                        # Mode 1: gradient points A->B (pair[0]->pair[1]), already correct
                        if mode == 0:
                            normal_world = -normal_world

                        c.position = point_centered  # Centered world-space position
                        c.normal = normal_world  # Normalized world-space normal pointing pair[0]->pair[1]
                        c.depth = dist
                        # Encode mode into feature to distinguish triangles from mesh0 vs mesh1
                        # Mode 0: positive triangle index, Mode 1: negative (-(index+1))
                        tri_idx = selected_triangles[t]
                        c.feature = tri_idx if mode == 0 else -(tri_idx + 1)
                        c.projection = empty_marker

                store_reduced_contact_func(
                    t, has_contact, c, contacts_shared_mem, active_contacts_shared_mem, betas, empty_marker
                )

                # Reset buffer for next batch
                synchronize()
                if t == 0:
                    selected_triangles[tri_capacity] = 0
                synchronize()

        # Now write the reduced contacts to the output array
        # Contacts are in centered world space - add midpoint back to get true world position
        # All contacts use consistent convention: shape_a = pair[0], shape_b = pair[1],
        # normal points from pair[0] to pair[1]
        synchronize()

        # Filter out duplicate contacts (same contact may have won multiple directions)
        filter_unique_contacts_func(t, contacts_shared_mem, active_contacts_shared_mem, empty_marker)

        num_contacts_to_keep = wp.min(
            active_contacts_shared_mem[wp.static(num_reduction_slots)], wp.static(num_reduction_slots)
        )

        for i in range(t, num_contacts_to_keep, wp.block_dim()):
            contact_id = active_contacts_shared_mem[i]
            contact = contacts_shared_mem[contact_id]

            # Add midpoint back to get true world position (contact.position is centered)
            point_world = contact.position + midpoint
            # Normal is already in world space and normalized
            normal_world = contact.normal

            # Create contact data
            contact_data = ContactData()
            contact_data.contact_point_center = point_world
            contact_data.contact_normal_a_to_b = normal_world  # Normalized and pointing pair[0]->pair[1]
            contact_data.contact_distance = contact.depth
            contact_data.radius_eff_a = 0.0
            contact_data.radius_eff_b = 0.0
            # SDF mesh's thickness is already baked into the SDF, so set it to 0
            # contact.feature >= 0 means mode 0: mesh0 triangles vs mesh1's SDF -> thickness1 already in SDF
            # contact.feature < 0 means mode 1: mesh1 triangles vs mesh0's SDF -> thickness0 already in SDF
            if contact.feature >= 0:
                contact_data.thickness_a = thickness0
                contact_data.thickness_b = 0.0
            else:
                contact_data.thickness_a = 0.0
                contact_data.thickness_b = thickness1
            contact_data.shape_a = pair[0]
            contact_data.shape_b = pair[1]
            contact_data.margin = margin
            # The high bit distinguishes contacts from mesh B (mode 1) vs mesh A (mode 0)
            if contact.feature >= 0:
                feature_id = wp.uint32(contact.feature + 1)
            else:
                feature_id = wp.uint32(-contact.feature) | wp.uint32(0x80000000)
            contact_data.feature = feature_id
            contact_data.feature_pair_key = build_pair_key2(wp.uint32(pair[0]), wp.uint32(pair[1]))

            writer_func(contact_data, writer_data, -1)

    return mesh_sdf_collision_reduce_kernel
