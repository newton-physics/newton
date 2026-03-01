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

from newton._src.core.types import MAXVAL

from ..geometry.contact_data import SHAPE_PAIR_HFIELD_BIT, SHAPE_PAIR_INDEX_MASK, ContactData
from ..geometry.sdf_utils import SDFData
from ..geometry.types import GeoType
from ..utils.heightfield import HeightfieldData, sample_sdf_grad_heightfield, sample_sdf_heightfield

# Handle both direct execution and module import
from .contact_reduction import (
    ContactReductionFunctions,
    ContactStruct,
    compute_voxel_index,
    get_shared_memory_pointer_block_dim_plus_2_ints,
    synchronize,
)


@wp.func
def scale_sdf_result_to_world(
    distance: float,
    gradient: wp.vec3,
    sdf_scale: wp.vec3,
    inv_sdf_scale: wp.vec3,
    min_sdf_scale: float,
) -> tuple[float, wp.vec3]:
    """
    Convert SDF distance and gradient from unscaled space to scaled space.

    Args:
        distance: Signed distance in unscaled SDF local space
        gradient: Gradient direction in unscaled SDF local space
        sdf_scale: The SDF shape's scale vector
        inv_sdf_scale: Precomputed 1.0 / sdf_scale
        min_sdf_scale: Precomputed min(sdf_scale) for distance scaling

    Returns:
        Tuple of (scaled_distance, scaled_gradient)
    """
    # Use min scale for conservative distance (won't miss contacts)
    scaled_distance = distance * min_sdf_scale

    # Gradient: apply inverse scale and renormalize
    scaled_grad = wp.cw_mul(gradient, inv_sdf_scale)
    grad_len = wp.length(scaled_grad)
    if grad_len > 0.0:
        scaled_grad = scaled_grad / grad_len
    else:
        scaled_grad = gradient

    return scaled_distance, scaled_grad


@wp.func
def sample_sdf_using_mesh(
    mesh_id: wp.uint64,
    world_pos: wp.vec3,
    max_dist: float = 1000.0,
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
    max_dist: float = 1000.0,
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
        sparse_idx = wp.volume_world_to_index(sdf_data.sparse_sdf_ptr, sdf_pos)
        sparse_dist = wp.volume_sample_f(sdf_data.sparse_sdf_ptr, sparse_idx, wp.Volume.LINEAR)

        if sparse_dist >= wp.static(MAXVAL * 0.99) or wp.isnan(sparse_dist):
            # Fallback to coarse grid when sparse sample is diluted by background
            coarse_idx = wp.volume_world_to_index(sdf_data.coarse_sdf_ptr, sdf_pos)
            return wp.volume_sample_f(sdf_data.coarse_sdf_ptr, coarse_idx, wp.Volume.LINEAR)
        else:
            return sparse_dist
    else:
        # Point is outside extent - project to boundary
        eps = 1e-2 * sdf_data.sparse_voxel_size  # slightly shrink to avoid sampling background
        clamped_pos = wp.min(wp.max(sdf_pos, lower + eps), upper - eps)
        dist_to_boundary = wp.length(sdf_pos - clamped_pos)

        # Sample at the boundary point using coarse grid
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
        sparse_idx = wp.volume_world_to_index(sdf_data.sparse_sdf_ptr, sdf_pos)
        sparse_dist = wp.volume_sample_grad_f(sdf_data.sparse_sdf_ptr, sparse_idx, wp.Volume.LINEAR, gradient)

        if sparse_dist >= wp.static(MAXVAL * 0.99) or wp.isnan(sparse_dist):
            # Fallback to coarse grid when sparse sample is diluted by background
            coarse_idx = wp.volume_world_to_index(sdf_data.coarse_sdf_ptr, sdf_pos)
            coarse_dist = wp.volume_sample_grad_f(sdf_data.coarse_sdf_ptr, coarse_idx, wp.Volume.LINEAR, gradient)
            return coarse_dist, gradient
        else:
            return sparse_dist, gradient
    else:
        # Point is outside extent - project to boundary
        eps = (
            1e-2 * sdf_data.sparse_voxel_size
        )  # slightly shrink the extent to avoid sampling the background value at edge
        clamped_pos = wp.min(wp.max(sdf_pos, lower + eps), upper - eps)
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
    sdf_is_heightfield: bool,
    hfd_sdf: HeightfieldData,
    elevation_data: wp.array(dtype=wp.float32),
) -> tuple[float, wp.vec3, wp.vec3]:
    """
    Compute the deepest contact between a triangle and an SDF volume.

    This function uses gradient descent in barycentric coordinates to find the point
    on the triangle that has the minimum (most negative) signed distance to the SDF.
    The optimization starts from either the triangle centroid or one of its vertices
    (whichever has the smallest initial distance).

    SDF evaluation modes (tried in order):
    1. **Heightfield on-the-fly** (``sdf_is_heightfield``): plane distance to the
       piecewise-linear heightfield surface.
    2. **BVH mesh** (``use_bvh_for_sdf``): ``mesh_query_point_sign_normal``.
    3. **Sparse/coarse volume** (default): trilinear interpolation of precomputed SDF.

    Args:
        sdf_data: SDFData struct containing sparse/coarse volumes and extent info.
        sdf_mesh_id: Mesh ID for BVH-based collision (mode 2).
        v0, v1, v2: Triangle vertices in the SDF's local coordinate space.
        use_bvh_for_sdf: If True, use BVH-based collision instead of SDF volumes.
        sdf_is_heightfield: If True, evaluate SDF on the fly from heightfield data.
        hfd_sdf: HeightfieldData for the SDF shape (used when ``sdf_is_heightfield``).
        elevation_data: Concatenated elevation array (used when ``sdf_is_heightfield``).

    Returns:
        Tuple of:
            - distance: Signed distance to SDF surface (negative = penetration).
            - contact_point: The point on the triangle closest to the SDF surface.
            - contact_direction: Normalized SDF gradient at the contact point.
    """
    third = 1.0 / 3.0
    center = (v0 + v1 + v2) * third
    p = center

    # Sample SDF for initial distance estimates
    if sdf_is_heightfield:
        dist = sample_sdf_heightfield(hfd_sdf, elevation_data, p)
        d0 = sample_sdf_heightfield(hfd_sdf, elevation_data, v0)
        d1 = sample_sdf_heightfield(hfd_sdf, elevation_data, v1)
        d2 = sample_sdf_heightfield(hfd_sdf, elevation_data, v2)
    elif use_bvh_for_sdf:
        dist = sample_sdf_using_mesh(sdf_mesh_id, p)
        d0 = sample_sdf_using_mesh(sdf_mesh_id, v0)
        d1 = sample_sdf_using_mesh(sdf_mesh_id, v1)
        d2 = sample_sdf_using_mesh(sdf_mesh_id, v2)
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
        # Sample SDF gradient
        if sdf_is_heightfield:
            _, sdf_gradient = sample_sdf_grad_heightfield(hfd_sdf, elevation_data, p)
        elif use_bvh_for_sdf:
            _, sdf_gradient = sample_sdf_grad_using_mesh(sdf_mesh_id, p)
        else:
            _, sdf_gradient = sample_sdf_grad_extrapolated(sdf_data, p)

        grad_len = wp.length(sdf_gradient)
        if grad_len == 0.0:
            # Zero gradient means we hit a discontinuity (e.g. the exact center
            # of a cube). Pick an arbitrary unit-length direction to escape it.
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

    # Final SDF sampling for result
    if sdf_is_heightfield:
        dist, sdf_gradient = sample_sdf_grad_heightfield(hfd_sdf, elevation_data, p)
    elif use_bvh_for_sdf:
        dist, sdf_gradient = sample_sdf_grad_using_mesh(sdf_mesh_id, p)
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
def _sphere_aabb_distance(center: wp.vec3, aabb_lower: wp.vec3, aabb_upper: wp.vec3) -> float:
    """Unsigned distance from a point to an AABB (0 when inside, positive outside)."""
    closest = wp.min(wp.max(center, aabb_lower), aabb_upper)
    return wp.length(center - closest)


@wp.func
def get_triangle_from_heightfield(
    hfd: HeightfieldData,
    elevation_data: wp.array(dtype=wp.float32),
    mesh_scale: wp.vec3,
    X_ws: wp.transform,
    tri_idx: int,
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    """Extract a triangle from a heightfield by linear triangle index.

    Each grid cell produces 2 triangles. ``tri_idx`` encodes
    ``(row, col, sub_tri)`` as ``row * (ncol-1) * 2 + col * 2 + sub_tri``.

    Returns ``(v0_world, v1_world, v2_world)`` matching :func:`get_triangle_from_mesh`.
    """
    cells_per_row = hfd.ncol - 1
    cell_idx = tri_idx // 2
    tri_sub = tri_idx - cell_idx * 2
    row = cell_idx // cells_per_row
    col = cell_idx - row * cells_per_row

    dx = 2.0 * hfd.hx / wp.float32(hfd.ncol - 1)
    dy = 2.0 * hfd.hy / wp.float32(hfd.nrow - 1)
    z_range = hfd.max_z - hfd.min_z

    x0 = -hfd.hx + wp.float32(col) * dx
    x1 = x0 + dx
    y0 = -hfd.hy + wp.float32(row) * dy
    y1 = y0 + dy

    base = hfd.data_offset
    h00 = elevation_data[base + row * hfd.ncol + col]
    h10 = elevation_data[base + row * hfd.ncol + (col + 1)]
    h01 = elevation_data[base + (row + 1) * hfd.ncol + col]
    h11 = elevation_data[base + (row + 1) * hfd.ncol + (col + 1)]

    p00 = wp.vec3(x0, y0, hfd.min_z + h00 * z_range)
    p10 = wp.vec3(x1, y0, hfd.min_z + h10 * z_range)
    p01 = wp.vec3(x0, y1, hfd.min_z + h01 * z_range)
    p11 = wp.vec3(x1, y1, hfd.min_z + h11 * z_range)

    if tri_sub == 0:
        v0_local = p00
        v1_local = p10
        v2_local = p11
    else:
        v0_local = p00
        v1_local = p11
        v2_local = p01

    v0_local = wp.cw_mul(v0_local, mesh_scale)
    v1_local = wp.cw_mul(v1_local, mesh_scale)
    v2_local = wp.cw_mul(v2_local, mesh_scale)

    v0_world = wp.transform_point(X_ws, v0_local)
    v1_world = wp.transform_point(X_ws, v1_local)
    v2_world = wp.transform_point(X_ws, v2_local)

    return v0_world, v1_world, v2_world


@wp.func
def get_triangle_count(shape_type: int, mesh_id: wp.uint64, hfd: HeightfieldData) -> int:
    """Return the number of triangles for a mesh or heightfield shape."""
    if shape_type == GeoType.HFIELD:
        if hfd.nrow <= 1 or hfd.ncol <= 1:
            return 0
        return 2 * (hfd.nrow - 1) * (hfd.ncol - 1)
    return wp.mesh_get(mesh_id).indices.shape[0] // 3


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
    inv_sdf_scale: wp.vec3,
    tri_shape_type: int,
    sdf_shape_type: int,
    hfd_tri: HeightfieldData,
    hfd_sdf: HeightfieldData,
    elevation_data: wp.array(dtype=wp.float32),
):
    """
    Midphase triangle culling for mesh-SDF collision.

    Determines which triangles are close enough to the SDF to potentially generate contacts.
    Triangles are transformed to unscaled SDF space before testing.

    Uses a two-level culling strategy:
    1. **AABB early-out (pure ALU, no memory access):** If the triangle's bounding
       sphere is farther from the SDF volume's AABB than ``contact_distance``, the
       triangle is discarded immediately.
    2. **SDF sample:** For triangles that survive the AABB test, sample the SDF at
       the bounding-sphere center to get a tighter distance estimate.

    Buffer layout: [0..block_dim-1] = triangle indices, [block_dim] = count, [block_dim+1] = progress
    """
    num_tris = get_triangle_count(tri_shape_type, mesh_id, hfd_tri)
    capacity = wp.block_dim()

    sdf_is_heightfield = sdf_shape_type == GeoType.HFIELD

    # Precompute SDF AABB in unscaled local space for the volume path.
    # For BVH and heightfield paths, the per-sample functions already have
    # their own distance culling so we skip the AABB pre-filter.
    sdf_aabb_lower = sdf_data.center - sdf_data.half_extents
    sdf_aabb_upper = sdf_data.center + sdf_data.half_extents

    synchronize()  # Ensure buffer state is consistent before starting

    while buffer[capacity + 1] < num_tris and buffer[capacity] < capacity:
        # All threads read the same base index (buffer consistent from previous sync)
        base_tri_idx = buffer[capacity + 1]
        tri_idx = base_tri_idx + thread_id
        add_triangle = False

        if tri_idx < num_tris:
            # Get vertices in scaled SDF local space, then convert to unscaled
            if tri_shape_type == GeoType.HFIELD:
                v0_scaled, v1_scaled, v2_scaled = get_triangle_from_heightfield(
                    hfd_tri, elevation_data, mesh_scale, mesh_to_sdf_transform, tri_idx
                )
            else:
                v0_scaled, v1_scaled, v2_scaled = get_triangle_from_mesh(
                    mesh_id, mesh_scale, mesh_to_sdf_transform, tri_idx
                )
            # Transform to unscaled SDF space for collision detection
            v0 = wp.cw_mul(v0_scaled, inv_sdf_scale)
            v1 = wp.cw_mul(v1_scaled, inv_sdf_scale)
            v2 = wp.cw_mul(v2_scaled, inv_sdf_scale)
            bounding_sphere_center, bounding_sphere_radius = get_bounding_sphere(v0, v1, v2)

            threshold = bounding_sphere_radius + contact_distance

            if sdf_is_heightfield:
                sdf_dist = sample_sdf_heightfield(hfd_sdf, elevation_data, bounding_sphere_center)
                add_triangle = sdf_dist <= threshold
            elif use_bvh_for_sdf:
                sdf_dist = sample_sdf_using_mesh(sdf_mesh_id, bounding_sphere_center, 1.01 * threshold)
                add_triangle = sdf_dist <= threshold
            else:
                aabb_dist = _sphere_aabb_distance(bounding_sphere_center, sdf_aabb_lower, sdf_aabb_upper)
                if aabb_dist <= threshold:
                    sdf_dist = sample_sdf_extrapolated(sdf_data, bounding_sphere_center)
                    add_triangle = sdf_dist <= threshold

        synchronize()  # Ensure all threads have read base_tri_idx before any writes
        add_to_shared_buffer_atomic(thread_id, add_triangle, tri_idx, buffer)
        # add_to_shared_buffer_atomic ends with sync, buffer is consistent for next while check

    synchronize()  # Final sync before returning


def create_narrow_phase_process_mesh_mesh_contacts_kernel(
    writer_func: Any,
    contact_reduction_funcs: ContactReductionFunctions | None = None,
    enable_heightfields: bool = True,
):
    @wp.kernel(enable_backward=False, module="unique")
    def mesh_sdf_collision_kernel(
        shape_data: wp.array(dtype=wp.vec4),
        shape_transform: wp.array(dtype=wp.transform),
        shape_source: wp.array(dtype=wp.uint64),
        sdf_table: wp.array(dtype=SDFData),
        shape_sdf_index: wp.array(dtype=wp.int32),
        shape_gap: wp.array(dtype=float),
        _shape_collision_aabb_lower: wp.array(dtype=wp.vec3),
        _shape_collision_aabb_upper: wp.array(dtype=wp.vec3),
        _shape_voxel_resolution: wp.array(dtype=wp.vec3i),
        shape_pairs_mesh_mesh: wp.array(dtype=wp.vec2i),
        shape_pairs_mesh_mesh_count: wp.array(dtype=int),
        shape_heightfield_data: wp.array(dtype=HeightfieldData),
        heightfield_elevation_data: wp.array(dtype=wp.float32),
        writer_data: Any,
        total_num_blocks: int,
    ):
        """Process mesh-mesh and mesh-heightfield collisions using SDF-based detection."""
        block_id, t = wp.tid()

        pair_count = shape_pairs_mesh_mesh_count[0]

        # Strided loop over pairs
        for pair_idx in range(block_id, pair_count, total_num_blocks):
            pair_encoded = shape_pairs_mesh_mesh[pair_idx]
            if wp.static(enable_heightfields):
                has_hfield = (pair_encoded[0] & SHAPE_PAIR_HFIELD_BIT) != 0
                pair = wp.vec2i(pair_encoded[0] & SHAPE_PAIR_INDEX_MASK, pair_encoded[1])
            else:
                has_hfield = False
                pair = pair_encoded

            margin = shape_gap[pair[0]] + shape_gap[pair[1]]

            for mode in range(2):
                tri_shape = pair[mode]
                sdf_shape = pair[1 - mode]

                if wp.static(enable_heightfields):
                    tri_is_hfield = has_hfield and mode == 0
                    sdf_is_hfield = has_hfield and mode == 1
                else:
                    tri_is_hfield = False
                    sdf_is_hfield = False
                tri_type = GeoType.HFIELD if tri_is_hfield else GeoType.MESH
                sdf_type = GeoType.HFIELD if sdf_is_hfield else GeoType.MESH

                mesh_id_tri = shape_source[tri_shape]
                mesh_id_sdf = shape_source[sdf_shape]

                # Skip invalid sources (heightfields use HeightfieldData instead of mesh id)
                if not tri_is_hfield and mesh_id_tri == wp.uint64(0):
                    continue
                if not sdf_is_hfield and mesh_id_sdf == wp.uint64(0):
                    continue

                hfd_tri = HeightfieldData()
                hfd_sdf = HeightfieldData()
                if wp.static(enable_heightfields):
                    if tri_is_hfield:
                        hfd_tri = shape_heightfield_data[tri_shape]
                    if sdf_is_hfield:
                        hfd_sdf = shape_heightfield_data[sdf_shape]

                # SDF availability: heightfields always use on-the-fly evaluation
                use_bvh_for_sdf = False
                sdf_data = SDFData()
                if not sdf_is_hfield:
                    sdf_idx = shape_sdf_index[sdf_shape]
                    use_bvh_for_sdf = sdf_idx < 0
                    if not use_bvh_for_sdf:
                        sdf_ptr = sdf_table[sdf_idx].sparse_sdf_ptr
                        use_bvh_for_sdf = sdf_ptr == wp.uint64(0)
                    if not use_bvh_for_sdf:
                        sdf_data = sdf_table[sdf_idx]

                scale_data_tri = shape_data[tri_shape]
                scale_data_sdf = shape_data[sdf_shape]
                mesh_scale_tri = wp.vec3(scale_data_tri[0], scale_data_tri[1], scale_data_tri[2])
                mesh_scale_sdf = wp.vec3(scale_data_sdf[0], scale_data_sdf[1], scale_data_sdf[2])

                X_tri_ws = shape_transform[tri_shape]
                X_sdf_ws = shape_transform[sdf_shape]

                sdf_scale = wp.vec3(1.0, 1.0, 1.0)
                if not sdf_is_hfield and not use_bvh_for_sdf:
                    if not sdf_data.scale_baked:
                        sdf_scale = mesh_scale_sdf

                X_mesh_to_sdf = wp.transform_multiply(wp.transform_inverse(X_sdf_ws), X_tri_ws)

                triangle_mesh_thickness = scale_data_tri[3]
                sdf_mesh_thickness = scale_data_sdf[3]

                inv_sdf_scale = wp.cw_div(wp.vec3(1.0, 1.0, 1.0), sdf_scale)
                min_sdf_scale = wp.min(wp.min(sdf_scale[0], sdf_scale[1]), sdf_scale[2])

                contact_threshold = margin + triangle_mesh_thickness + sdf_mesh_thickness
                contact_threshold_unscaled = contact_threshold / min_sdf_scale

                tri_capacity = wp.block_dim()
                selected_triangles = wp.array(
                    ptr=get_shared_memory_pointer_block_dim_plus_2_ints(),
                    shape=(wp.block_dim() + 2,),
                    dtype=wp.int32,
                )

                if t == 0:
                    selected_triangles[tri_capacity] = 0
                    selected_triangles[tri_capacity + 1] = 0
                synchronize()

                num_tris = get_triangle_count(tri_type, mesh_id_tri, hfd_tri)

                while selected_triangles[tri_capacity + 1] < num_tris:
                    find_interesting_triangles(
                        t,
                        mesh_scale_tri,
                        X_mesh_to_sdf,
                        mesh_id_tri,
                        sdf_data,
                        mesh_id_sdf,
                        selected_triangles,
                        contact_threshold_unscaled,
                        use_bvh_for_sdf,
                        inv_sdf_scale,
                        tri_type,
                        sdf_type,
                        hfd_tri,
                        hfd_sdf,
                        heightfield_elevation_data,
                    )

                    has_triangle = t < selected_triangles[tri_capacity]
                    synchronize()

                    if has_triangle:
                        tri_idx = selected_triangles[t]

                        if tri_is_hfield:
                            v0_scaled, v1_scaled, v2_scaled = get_triangle_from_heightfield(
                                hfd_tri, heightfield_elevation_data, mesh_scale_tri, X_mesh_to_sdf, tri_idx
                            )
                        else:
                            v0_scaled, v1_scaled, v2_scaled = get_triangle_from_mesh(
                                mesh_id_tri, mesh_scale_tri, X_mesh_to_sdf, tri_idx
                            )
                        v0 = wp.cw_mul(v0_scaled, inv_sdf_scale)
                        v1 = wp.cw_mul(v1_scaled, inv_sdf_scale)
                        v2 = wp.cw_mul(v2_scaled, inv_sdf_scale)

                        dist_unscaled, point_unscaled, direction_unscaled = do_triangle_sdf_collision(
                            sdf_data,
                            mesh_id_sdf,
                            v0,
                            v1,
                            v2,
                            use_bvh_for_sdf,
                            sdf_is_hfield,
                            hfd_sdf,
                            heightfield_elevation_data,
                        )

                        dist, direction = scale_sdf_result_to_world(
                            dist_unscaled, direction_unscaled, sdf_scale, inv_sdf_scale, min_sdf_scale
                        )
                        point = wp.cw_mul(point_unscaled, sdf_scale)

                        if dist < contact_threshold:
                            point_world = wp.transform_point(X_sdf_ws, point)

                            direction_world = wp.transform_vector(X_sdf_ws, direction)
                            direction_len = wp.length(direction_world)
                            if direction_len > 0.0:
                                direction_world = direction_world / direction_len

                                contact_normal = -direction_world if mode == 0 else direction_world

                                contact_data = ContactData()
                                contact_data.contact_point_center = point_world
                                contact_data.contact_normal_a_to_b = contact_normal
                                contact_data.contact_distance = dist
                                contact_data.radius_eff_a = 0.0
                                contact_data.radius_eff_b = 0.0
                                contact_data.margin_a = shape_data[pair[0]][3]
                                contact_data.margin_b = shape_data[pair[1]][3]
                                contact_data.shape_a = pair[0]
                                contact_data.shape_b = pair[1]
                                contact_data.margin = margin

                                writer_func(contact_data, writer_data, -1)

                    synchronize()
                    if t == 0:
                        selected_triangles[tri_capacity] = 0
                    synchronize()

    # Return early if contact reduction is disabled
    if contact_reduction_funcs is None:
        return mesh_sdf_collision_kernel

    # Extract functions and constants from the contact reduction configuration
    reduction_slot_count = contact_reduction_funcs.reduction_slot_count
    store_reduced_contact_func = contact_reduction_funcs.store_reduced_contact
    filter_unique_contacts_func = contact_reduction_funcs.filter_unique_contacts
    get_smem_slots_plus_1 = contact_reduction_funcs.get_smem_slots_plus_1
    get_smem_slots_contacts = contact_reduction_funcs.get_smem_slots_contacts

    @wp.kernel(enable_backward=False, module="unique")
    def mesh_sdf_collision_reduce_kernel(
        shape_data: wp.array(dtype=wp.vec4),
        shape_transform: wp.array(dtype=wp.transform),
        shape_source: wp.array(dtype=wp.uint64),
        sdf_table: wp.array(dtype=SDFData),
        shape_sdf_index: wp.array(dtype=wp.int32),
        shape_gap: wp.array(dtype=float),
        shape_collision_aabb_lower: wp.array(dtype=wp.vec3),
        shape_collision_aabb_upper: wp.array(dtype=wp.vec3),
        shape_voxel_resolution: wp.array(dtype=wp.vec3i),
        shape_pairs_mesh_mesh: wp.array(dtype=wp.vec2i),
        shape_pairs_mesh_mesh_count: wp.array(dtype=int),
        shape_heightfield_data: wp.array(dtype=HeightfieldData),
        heightfield_elevation_data: wp.array(dtype=wp.float32),
        writer_data: Any,
        total_num_blocks: int,
    ):
        block_id, t = wp.tid()
        pair_count = shape_pairs_mesh_mesh_count[0]

        for pair_idx in range(block_id, pair_count, total_num_blocks):
            pair_encoded = shape_pairs_mesh_mesh[pair_idx]
            if wp.static(enable_heightfields):
                has_hfield = (pair_encoded[0] & SHAPE_PAIR_HFIELD_BIT) != 0
                pair = wp.vec2i(pair_encoded[0] & SHAPE_PAIR_INDEX_MASK, pair_encoded[1])
            else:
                has_hfield = False
                pair = pair_encoded

            margin = shape_gap[pair[0]] + shape_gap[pair[1]]

            empty_marker = wp.static(-MAXVAL)

            active_contacts_shared_mem = wp.array(
                ptr=wp.static(get_smem_slots_plus_1)(),
                shape=(wp.static(reduction_slot_count) + 1,),
                dtype=wp.int32,
            )
            contacts_shared_mem = wp.array(
                ptr=wp.static(get_smem_slots_contacts)(),
                shape=(wp.static(reduction_slot_count),),
                dtype=ContactStruct,
            )

            for i in range(t, wp.static(reduction_slot_count), wp.block_dim()):
                contacts_shared_mem[i].projection = empty_marker

            if t == 0:
                active_contacts_shared_mem[wp.static(reduction_slot_count)] = 0

            for mode in range(2):
                tri_shape = pair[mode]
                sdf_shape = pair[1 - mode]

                if wp.static(enable_heightfields):
                    tri_is_hfield = has_hfield and mode == 0
                    sdf_is_hfield = has_hfield and mode == 1
                else:
                    tri_is_hfield = False
                    sdf_is_hfield = False
                tri_type = GeoType.HFIELD if tri_is_hfield else GeoType.MESH
                sdf_type = GeoType.HFIELD if sdf_is_hfield else GeoType.MESH

                mesh_id_tri = shape_source[tri_shape]
                mesh_id_sdf = shape_source[sdf_shape]

                if not tri_is_hfield and mesh_id_tri == wp.uint64(0):
                    continue
                if not sdf_is_hfield and mesh_id_sdf == wp.uint64(0):
                    continue

                hfd_tri = HeightfieldData()
                hfd_sdf = HeightfieldData()
                if wp.static(enable_heightfields):
                    if tri_is_hfield:
                        hfd_tri = shape_heightfield_data[tri_shape]
                    if sdf_is_hfield:
                        hfd_sdf = shape_heightfield_data[sdf_shape]

                use_bvh_for_sdf = False
                sdf_data = SDFData()
                if not sdf_is_hfield:
                    sdf_idx = shape_sdf_index[sdf_shape]
                    use_bvh_for_sdf = sdf_idx < 0
                    if not use_bvh_for_sdf:
                        sdf_ptr = sdf_table[sdf_idx].sparse_sdf_ptr
                        use_bvh_for_sdf = sdf_ptr == wp.uint64(0)
                    if not use_bvh_for_sdf:
                        sdf_data = sdf_table[sdf_idx]

                scale_data_tri = shape_data[tri_shape]
                scale_data_sdf = shape_data[sdf_shape]
                mesh_scale_tri = wp.vec3(scale_data_tri[0], scale_data_tri[1], scale_data_tri[2])
                mesh_scale_sdf = wp.vec3(scale_data_sdf[0], scale_data_sdf[1], scale_data_sdf[2])

                X_tri_ws = shape_transform[tri_shape]
                X_sdf_ws = shape_transform[sdf_shape]
                X_ws_tri = wp.transform_inverse(X_tri_ws)

                aabb_lower_tri = shape_collision_aabb_lower[tri_shape]
                aabb_upper_tri = shape_collision_aabb_upper[tri_shape]
                voxel_res_tri = shape_voxel_resolution[tri_shape]

                sdf_scale = wp.vec3(1.0, 1.0, 1.0)
                if not sdf_is_hfield and not use_bvh_for_sdf:
                    if not sdf_data.scale_baked:
                        sdf_scale = mesh_scale_sdf

                X_mesh_to_sdf = wp.transform_multiply(wp.transform_inverse(X_sdf_ws), X_tri_ws)

                triangle_mesh_thickness = scale_data_tri[3]
                sdf_mesh_thickness = scale_data_sdf[3]

                midpoint = (wp.transform_get_translation(X_tri_ws) + wp.transform_get_translation(X_sdf_ws)) * 0.5

                inv_sdf_scale = wp.cw_div(wp.vec3(1.0, 1.0, 1.0), sdf_scale)
                min_sdf_scale = wp.min(wp.min(sdf_scale[0], sdf_scale[1]), sdf_scale[2])

                contact_threshold = margin + triangle_mesh_thickness + sdf_mesh_thickness
                contact_threshold_unscaled = contact_threshold / min_sdf_scale

                tri_capacity = wp.block_dim()
                selected_triangles = wp.array(
                    ptr=get_shared_memory_pointer_block_dim_plus_2_ints(),
                    shape=(wp.block_dim() + 2,),
                    dtype=wp.int32,
                )

                if t == 0:
                    selected_triangles[tri_capacity] = 0
                    selected_triangles[tri_capacity + 1] = 0
                synchronize()

                num_tris = get_triangle_count(tri_type, mesh_id_tri, hfd_tri)

                while selected_triangles[tri_capacity + 1] < num_tris:
                    find_interesting_triangles(
                        t,
                        mesh_scale_tri,
                        X_mesh_to_sdf,
                        mesh_id_tri,
                        sdf_data,
                        mesh_id_sdf,
                        selected_triangles,
                        contact_threshold_unscaled,
                        use_bvh_for_sdf,
                        inv_sdf_scale,
                        tri_type,
                        sdf_type,
                        hfd_tri,
                        hfd_sdf,
                        heightfield_elevation_data,
                    )

                    has_triangle = t < selected_triangles[tri_capacity]
                    synchronize()
                    c = ContactStruct()
                    has_contact = wp.bool(False)

                    if has_triangle:
                        tri_idx = selected_triangles[t]

                        if tri_is_hfield:
                            v0_scaled, v1_scaled, v2_scaled = get_triangle_from_heightfield(
                                hfd_tri, heightfield_elevation_data, mesh_scale_tri, X_mesh_to_sdf, tri_idx
                            )
                        else:
                            v0_scaled, v1_scaled, v2_scaled = get_triangle_from_mesh(
                                mesh_id_tri, mesh_scale_tri, X_mesh_to_sdf, tri_idx
                            )
                        v0 = wp.cw_mul(v0_scaled, inv_sdf_scale)
                        v1 = wp.cw_mul(v1_scaled, inv_sdf_scale)
                        v2 = wp.cw_mul(v2_scaled, inv_sdf_scale)

                        dist_unscaled, point_unscaled, direction_unscaled = do_triangle_sdf_collision(
                            sdf_data,
                            mesh_id_sdf,
                            v0,
                            v1,
                            v2,
                            use_bvh_for_sdf,
                            sdf_is_hfield,
                            hfd_sdf,
                            heightfield_elevation_data,
                        )

                        dist, direction = scale_sdf_result_to_world(
                            dist_unscaled, direction_unscaled, sdf_scale, inv_sdf_scale, min_sdf_scale
                        )
                        point = wp.cw_mul(point_unscaled, sdf_scale)

                        if dist < contact_threshold:
                            point_world = wp.transform_point(X_sdf_ws, point)

                            direction_world = wp.transform_vector(X_sdf_ws, direction)
                            direction_len = wp.length(direction_world)
                            if direction_len > 0.0:
                                direction_world = direction_world / direction_len
                                has_contact = True

                                contact_normal = -direction_world if mode == 0 else direction_world

                                c.position = point_world - midpoint
                                c.normal = contact_normal
                                c.depth = dist
                                c.feature = tri_idx if mode == 0 else -(tri_idx + 1)
                                c.projection = empty_marker

                    voxel_idx = int(0)
                    if has_contact:
                        point_tri_local = wp.transform_point(X_ws_tri, point_world)
                        voxel_idx = compute_voxel_index(point_tri_local, aabb_lower_tri, aabb_upper_tri, voxel_res_tri)

                    store_reduced_contact_func(
                        t,
                        has_contact,
                        c,
                        contacts_shared_mem,
                        active_contacts_shared_mem,
                        empty_marker,
                        voxel_idx,
                    )

                    synchronize()
                    if t == 0:
                        selected_triangles[tri_capacity] = 0
                    synchronize()

            # Now write the reduced contacts to the output array
            # Contacts are in centered world space - add midpoint back to get true world position
            # All contacts use consistent convention: shape_a = pair[0], shape_b = pair[1]
            # SYNC: Ensure all contacts from both modes are stored before filtering
            synchronize()

            # Filter out duplicate contacts (same contact may have won multiple directions)
            filter_unique_contacts_func(t, contacts_shared_mem, active_contacts_shared_mem, empty_marker)

            num_contacts_to_keep = wp.min(
                active_contacts_shared_mem[wp.static(reduction_slot_count)], wp.static(reduction_slot_count)
            )

            # Compute midpoint for uncentering contacts (same as computed in mode loop)
            midpoint_out = (
                wp.transform_get_translation(shape_transform[pair[0]])
                + wp.transform_get_translation(shape_transform[pair[1]])
            ) * 0.5

            for i in range(t, num_contacts_to_keep, wp.block_dim()):
                contact_id = active_contacts_shared_mem[i]
                contact = contacts_shared_mem[contact_id]

                # Add midpoint back to get true world position (contact.position is centered)
                point_world = contact.position + midpoint_out

                # Create contact data
                contact_data = ContactData()
                contact_data.contact_point_center = point_world
                contact_data.contact_normal_a_to_b = contact.normal
                contact_data.contact_distance = contact.depth
                contact_data.radius_eff_a = 0.0
                contact_data.radius_eff_b = 0.0
                contact_data.margin_a = shape_data[pair[0]][3]
                contact_data.margin_b = shape_data[pair[1]][3]
                contact_data.shape_a = pair[0]
                contact_data.shape_b = pair[1]
                contact_data.margin = margin

                writer_func(contact_data, writer_data, -1)

    return mesh_sdf_collision_reduce_kernel
