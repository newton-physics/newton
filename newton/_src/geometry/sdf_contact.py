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

from ..geometry.contact_data import SHAPE_PAIR_HFIELD_BIT, SHAPE_PAIR_INDEX_MASK, ContactData
from ..geometry.sdf_texture import TextureSDFData, texture_sample_sdf, texture_sample_sdf_grad
from ..geometry.types import GeoType
from ..utils.heightfield import HeightfieldData, sample_sdf_grad_heightfield, sample_sdf_heightfield

# Handle both direct execution and module import
from .contact_reduction import (
    create_shared_memory_pointer_block_dim_mul_func,
    get_shared_memory_pointer_block_dim_plus_2_ints,
    synchronize,
)
from .contact_reduction_global import GlobalContactReducerData, export_and_reduce_contact_centered

# Shared memory for caching triangle vertices that pass midphase culling.
# block_dim triangles x 3 vertices x 3 floats = 9 int32s per thread.
_get_shared_memory_vertex_cache = create_shared_memory_pointer_block_dim_mul_func(9)


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
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
    vertex_cache: wp.array(dtype=wp.vec3),
    vertex_cache_offset: int,
):
    """Add a triangle index to a shared memory buffer using atomic operations.

    Also caches the triangle's pre-computed vertices in ``vertex_cache`` so
    that downstream consumers can read them from shared memory instead of
    re-fetching from global memory.

    Buffer layout:
    - [0 .. block_dim-1]: Triangle indices
    - [block_dim]: Current count of triangles in buffer
    - [block_dim+1]: Progress counter (triangles processed so far)

    Args:
        thread_id: The calling thread's index within the thread block
        add_triangle: Whether this thread wants to add a triangle
        tri_idx: The triangle index to add (only used if add_triangle is True)
        buffer: Shared memory buffer for triangle indices
        v0: First vertex in unscaled SDF space (stored only if add_triangle is True)
        v1: Second vertex in unscaled SDF space
        v2: Third vertex in unscaled SDF space
        vertex_cache: Shared memory array (double-buffered staging), dtype=vec3
        vertex_cache_offset: Base offset into ``vertex_cache`` for the active staging buffer.
    """
    capacity = wp.block_dim()
    idx = -1

    if add_triangle:
        idx = wp.atomic_add(buffer, capacity, 1)
        if idx < capacity:
            buffer[idx] = tri_idx
            base = vertex_cache_offset + idx * 3
            vertex_cache[base] = v0
            vertex_cache[base + 1] = v1
            vertex_cache[base + 2] = v2

    if thread_id == 0:
        buffer[capacity + 1] += capacity

    synchronize()  # SYNC 1: All atomic writes and progress update complete

    if thread_id == 0 and buffer[capacity] > capacity:
        buffer[capacity] = capacity

    if add_triangle and idx >= capacity:
        wp.atomic_min(buffer, capacity + 1, tri_idx)

    synchronize()  # SYNC 2: All corrections complete, buffer consistent


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


def _create_sdf_contact_funcs(enable_heightfields: bool):
    """Generate SDF contact functions with heightfield branches eliminated at compile time.

    When ``enable_heightfields`` is False, ``wp.static`` strips all heightfield code
    paths from the generated functions, reducing register pressure and instruction
    cache footprint — especially in the 16-iteration gradient descent loop of
    ``do_triangle_sdf_collision``.

    Args:
        enable_heightfields: When False, all heightfield code paths are compiled out.

    Returns:
        Tuple of ``(prefetch_triangle_vertices, find_interesting_triangles,
        do_triangle_sdf_collision)``.
    """

    @wp.func
    def do_triangle_sdf_collision_func(
        texture_sdf: TextureSDFData,
        sdf_mesh_id: wp.uint64,
        v0: wp.vec3,
        v1: wp.vec3,
        v2: wp.vec3,
        use_bvh_for_sdf: bool,
        sdf_is_heightfield: bool,
        hfd_sdf: HeightfieldData,
        elevation_data: wp.array(dtype=wp.float32),
    ) -> tuple[float, wp.vec3, wp.vec3]:
        """Compute the deepest contact between a triangle and an SDF volume.

        Uses gradient descent in barycentric coordinates to find the point on the
        triangle with the minimum signed distance to the SDF. The optimization
        starts from the centroid or whichever vertex has the smallest initial
        distance.

        Returns:
            Tuple of (distance, contact_point, contact_direction).
        """
        third = 1.0 / 3.0
        center = (v0 + v1 + v2) * third
        p = center

        if wp.static(enable_heightfields):
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
                dist = texture_sample_sdf(texture_sdf, p)
                d0 = texture_sample_sdf(texture_sdf, v0)
                d1 = texture_sample_sdf(texture_sdf, v1)
                d2 = texture_sample_sdf(texture_sdf, v2)
        else:
            if use_bvh_for_sdf:
                dist = sample_sdf_using_mesh(sdf_mesh_id, p)
                d0 = sample_sdf_using_mesh(sdf_mesh_id, v0)
                d1 = sample_sdf_using_mesh(sdf_mesh_id, v1)
                d2 = sample_sdf_using_mesh(sdf_mesh_id, v2)
            else:
                dist = texture_sample_sdf(texture_sdf, p)
                d0 = texture_sample_sdf(texture_sdf, v0)
                d1 = texture_sample_sdf(texture_sdf, v1)
                d2 = texture_sample_sdf(texture_sdf, v2)

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

        # Relaxed from 1e-3 to 3e-3: the tighter tolerance required more
        # iterations that pushed float32 precision limits, hurting convergence
        # without measurably improving contact quality.
        tolerance_sq = 3e-3 * 3e-3

        sdf_gradient = wp.vec3(0.0, 0.0, 0.0)
        step = 1.0 / (2.0 * difference)

        for _iter in range(16):
            if wp.static(enable_heightfields):
                if sdf_is_heightfield:
                    _, sdf_gradient = sample_sdf_grad_heightfield(hfd_sdf, elevation_data, p)
                elif use_bvh_for_sdf:
                    _, sdf_gradient = sample_sdf_grad_using_mesh(sdf_mesh_id, p)
                else:
                    _, sdf_gradient = texture_sample_sdf_grad(texture_sdf, p)
            else:
                if use_bvh_for_sdf:
                    _, sdf_gradient = sample_sdf_grad_using_mesh(sdf_mesh_id, p)
                else:
                    _, sdf_gradient = texture_sample_sdf_grad(texture_sdf, p)

            grad_len = wp.length(sdf_gradient)
            if grad_len == 0.0:
                # Arbitrary non-axis-aligned unit vector to break symmetry
                sdf_gradient = wp.vec3(0.571846586, 0.705545099, 0.418566116)
                grad_len = 1.0

            sdf_gradient = sdf_gradient / grad_len

            dfdu = wp.dot(sdf_gradient, v0 - p)
            dfdv = wp.dot(sdf_gradient, v1 - p)
            dfdw = wp.dot(sdf_gradient, v2 - p)

            new_uvw = wp.vec3(uvw[0] - step * dfdu, uvw[1] - step * dfdv, uvw[2] - step * dfdw)

            step = step * 0.8

            new_uvw = closest_pt_point_bary_triangle(new_uvw)

            p = v0 * new_uvw[0] + v1 * new_uvw[1] + v2 * new_uvw[2]

            if wp.length_sq(uvw - new_uvw) < tolerance_sq:
                break

            uvw = new_uvw

        if wp.static(enable_heightfields):
            if sdf_is_heightfield:
                dist, sdf_gradient = sample_sdf_grad_heightfield(hfd_sdf, elevation_data, p)
            elif use_bvh_for_sdf:
                dist, sdf_gradient = sample_sdf_grad_using_mesh(sdf_mesh_id, p)
            else:
                dist, sdf_gradient = texture_sample_sdf_grad(texture_sdf, p)
        else:
            if use_bvh_for_sdf:
                dist, sdf_gradient = sample_sdf_grad_using_mesh(sdf_mesh_id, p)
            else:
                dist, sdf_gradient = texture_sample_sdf_grad(texture_sdf, p)

        return dist, p, sdf_gradient

    @wp.func
    def find_interesting_triangles_func(
        thread_id: int,
        mesh_scale: wp.vec3,
        mesh_to_sdf_transform: wp.transform,
        mesh_id: wp.uint64,
        texture_sdf: TextureSDFData,
        sdf_mesh_id: wp.uint64,
        buffer: wp.array(dtype=wp.int32),
        contact_distance: float,
        use_bvh_for_sdf: bool,
        inv_sdf_scale: wp.vec3,
        tri_end: int,
        tri_shape_type: int,
        sdf_shape_type: int,
        hfd_tri: HeightfieldData,
        hfd_sdf: HeightfieldData,
        elevation_data: wp.array(dtype=wp.float32),
        vertex_cache: wp.array(dtype=wp.vec3),
    ):
        """Midphase triangle culling for mesh-SDF collision.

        Determines which triangles are close enough to the SDF to potentially generate
        contacts. Triangles are transformed to unscaled SDF space before testing.
        Vertices of accepted triangles are cached in ``vertex_cache`` so the consumer
        can read them from shared memory instead of re-fetching from global memory.

        Uses a two-level culling strategy:

        1. **AABB early-out (pure ALU, no memory access):** If the triangle's bounding
           sphere is farther from the SDF volume's AABB than ``contact_distance``, the
           triangle is discarded immediately.
        2. **SDF sample:** For triangles that survive the AABB test, sample the SDF at
           the bounding-sphere center to get a tighter distance estimate.

        Buffer layout: [0..block_dim-1] = triangle indices, [block_dim] = count,
        [block_dim+1] = progress.

        Args:
            tri_end: Maximum triangle index (exclusive).
            vertex_cache: Shared memory array of size block_dim*3, dtype=vec3.
        """
        capacity = wp.block_dim()

        if wp.static(enable_heightfields):
            sdf_is_heightfield = sdf_shape_type == GeoType.HFIELD
        else:
            sdf_is_heightfield = False

        sdf_aabb_lower = texture_sdf.sdf_box_lower
        sdf_aabb_upper = texture_sdf.sdf_box_upper

        synchronize()  # Ensure buffer state is consistent before starting

        while buffer[capacity + 1] < tri_end and buffer[capacity] < capacity:
            base_tri_idx = buffer[capacity + 1]
            tri_idx = base_tri_idx + thread_id
            add_triangle = False
            v0 = wp.vec3(0.0, 0.0, 0.0)
            v1 = wp.vec3(0.0, 0.0, 0.0)
            v2 = wp.vec3(0.0, 0.0, 0.0)

            if tri_idx < tri_end:
                if wp.static(enable_heightfields):
                    if tri_shape_type == GeoType.HFIELD:
                        v0_scaled, v1_scaled, v2_scaled = get_triangle_from_heightfield(
                            hfd_tri, elevation_data, mesh_scale, mesh_to_sdf_transform, tri_idx
                        )
                    else:
                        v0_scaled, v1_scaled, v2_scaled = get_triangle_from_mesh(
                            mesh_id, mesh_scale, mesh_to_sdf_transform, tri_idx
                        )
                else:
                    v0_scaled, v1_scaled, v2_scaled = get_triangle_from_mesh(
                        mesh_id, mesh_scale, mesh_to_sdf_transform, tri_idx
                    )
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
                    culling_radius = threshold
                    clamped = wp.min(wp.max(bounding_sphere_center, sdf_aabb_lower), sdf_aabb_upper)
                    aabb_dist_sq = wp.length_sq(bounding_sphere_center - clamped)
                    if aabb_dist_sq > culling_radius * culling_radius:
                        add_triangle = False
                    else:
                        sdf_dist = texture_sample_sdf(texture_sdf, bounding_sphere_center)
                        add_triangle = sdf_dist <= culling_radius

            synchronize()  # Ensure all threads have read base_tri_idx before any writes
            add_to_shared_buffer_atomic(
                thread_id,
                add_triangle,
                tri_idx,
                buffer,
                v0,
                v1,
                v2,
                vertex_cache,
                0,
            )
            # add_to_shared_buffer_atomic ends with sync, buffer is consistent for next while check

        synchronize()  # Final sync before returning

    return find_interesting_triangles_func, do_triangle_sdf_collision_func


@wp.kernel(enable_backward=False)
def compute_mesh_mesh_block_offsets(
    shape_pairs_mesh_mesh: wp.array(dtype=wp.vec2i),
    shape_pairs_mesh_mesh_count: wp.array(dtype=int),
    shape_source: wp.array(dtype=wp.uint64),
    shape_heightfield_index: wp.array(dtype=wp.int32),
    heightfield_data: wp.array(dtype=HeightfieldData),
    target_blocks: int,
    block_offsets: wp.array(dtype=wp.int32),
):
    """Compute per-pair block counts and prefix sum for dynamic load balancing.

    Block counts are proportional to the total triangle count of both meshes
    (or heightfields) in each pair, so pairs with larger geometry get more
    GPU blocks.

    Args:
        target_blocks: Desired total number of blocks (e.g., sm_count * 4).
        block_offsets: Output array of size ``max_pairs + 1``.
            ``block_offsets[i]`` is the cumulative block count up to pair *i*.
    """
    tid = wp.tid()
    if tid > 0:
        return
    pair_count = wp.min(shape_pairs_mesh_mesh_count[0], shape_pairs_mesh_mesh.shape[0])

    # First pass: sum all triangle counts across all pairs and directions
    total_tris = int(0)
    for i in range(pair_count):
        pair_encoded = shape_pairs_mesh_mesh[i]
        has_hfield = (pair_encoded[0] & SHAPE_PAIR_HFIELD_BIT) != 0
        pair = wp.vec2i(pair_encoded[0] & SHAPE_PAIR_INDEX_MASK, pair_encoded[1])
        for mode in range(2):
            is_hfield = has_hfield and mode == 0
            shape_idx = pair[mode]
            if is_hfield:
                hfd = heightfield_data[shape_heightfield_index[shape_idx]]
                total_tris += get_triangle_count(GeoType.HFIELD, wp.uint64(0), hfd)
            else:
                mesh_id = shape_source[shape_idx]
                if mesh_id != wp.uint64(0):
                    total_tris += wp.mesh_get(mesh_id).indices.shape[0] // 3

    # Compute target triangles per block
    tris_per_block = int(total_tris)
    if target_blocks > 0 and total_tris > 0:
        tris_per_block = wp.max(256, total_tris // target_blocks)

    # Second pass: compute per-pair block counts and prefix sum
    offset = int(0)
    for i in range(pair_count):
        block_offsets[i] = offset
        pair_encoded = shape_pairs_mesh_mesh[i]
        has_hfield = (pair_encoded[0] & SHAPE_PAIR_HFIELD_BIT) != 0
        pair = wp.vec2i(pair_encoded[0] & SHAPE_PAIR_INDEX_MASK, pair_encoded[1])
        pair_tris = int(0)
        for mode in range(2):
            is_hfield = has_hfield and mode == 0
            shape_idx = pair[mode]
            if is_hfield:
                hfd = heightfield_data[shape_heightfield_index[shape_idx]]
                pair_tris += get_triangle_count(GeoType.HFIELD, wp.uint64(0), hfd)
            else:
                mesh_id = shape_source[shape_idx]
                if mesh_id != wp.uint64(0):
                    pair_tris += wp.mesh_get(mesh_id).indices.shape[0] // 3
        blocks = wp.max(1, (pair_tris + tris_per_block - 1) // tris_per_block)
        offset += blocks
    block_offsets[pair_count] = offset


def create_narrow_phase_process_mesh_mesh_contacts_kernel(
    writer_func: Any,
    enable_heightfields: bool = True,
    reduce_contacts: bool = False,
):
    find_interesting_triangles, do_triangle_sdf_collision = _create_sdf_contact_funcs(enable_heightfields)

    @wp.kernel(enable_backward=False, module="unique")
    def mesh_sdf_collision_kernel(
        shape_data: wp.array(dtype=wp.vec4),
        shape_transform: wp.array(dtype=wp.transform),
        shape_source: wp.array(dtype=wp.uint64),
        texture_sdf_table: wp.array(dtype=TextureSDFData),
        shape_sdf_index: wp.array(dtype=wp.int32),
        shape_gap: wp.array(dtype=float),
        _shape_collision_aabb_lower: wp.array(dtype=wp.vec3),
        _shape_collision_aabb_upper: wp.array(dtype=wp.vec3),
        _shape_voxel_resolution: wp.array(dtype=wp.vec3i),
        shape_pairs_mesh_mesh: wp.array(dtype=wp.vec2i),
        shape_pairs_mesh_mesh_count: wp.array(dtype=int),
        shape_heightfield_index: wp.array(dtype=wp.int32),
        heightfield_data: wp.array(dtype=HeightfieldData),
        heightfield_elevations: wp.array(dtype=wp.float32),
        writer_data: Any,
        total_num_blocks: int,
    ):
        """Process mesh-mesh and mesh-heightfield collisions using SDF-based detection."""
        block_id, t = wp.tid()

        pair_count = wp.min(shape_pairs_mesh_mesh_count[0], shape_pairs_mesh_mesh.shape[0])

        # Strided loop over pairs
        for pair_idx in range(block_id, pair_count, total_num_blocks):
            pair_encoded = shape_pairs_mesh_mesh[pair_idx]
            if wp.static(enable_heightfields):
                has_hfield = (pair_encoded[0] & SHAPE_PAIR_HFIELD_BIT) != 0
                pair = wp.vec2i(pair_encoded[0] & SHAPE_PAIR_INDEX_MASK, pair_encoded[1])
            else:
                has_hfield = False
                pair = pair_encoded

            gap_sum = shape_gap[pair[0]] + shape_gap[pair[1]]

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
                        hfd_tri = heightfield_data[shape_heightfield_index[tri_shape]]
                    if sdf_is_hfield:
                        hfd_sdf = heightfield_data[shape_heightfield_index[sdf_shape]]

                # SDF availability: heightfields always use on-the-fly evaluation
                use_bvh_for_sdf = False
                if not sdf_is_hfield:
                    sdf_idx = shape_sdf_index[sdf_shape]
                    use_bvh_for_sdf = sdf_idx < 0 or sdf_idx >= texture_sdf_table.shape[0]
                    if not use_bvh_for_sdf:
                        use_bvh_for_sdf = texture_sdf_table[sdf_idx].coarse_texture.width == 0

                scale_data_tri = shape_data[tri_shape]
                scale_data_sdf = shape_data[sdf_shape]
                mesh_scale_tri = wp.vec3(scale_data_tri[0], scale_data_tri[1], scale_data_tri[2])
                mesh_scale_sdf = wp.vec3(scale_data_sdf[0], scale_data_sdf[1], scale_data_sdf[2])

                X_tri_ws = shape_transform[tri_shape]
                X_sdf_ws = shape_transform[sdf_shape]

                # Determine sdf_scale for the SDF query.
                # Heightfields always use scale=identity, since SDF is directly sampled
                # from elevation grid. For texture SDF, override to identity when scale
                # is already baked. For BVH fallback, use the shape scale.
                texture_sdf = TextureSDFData()
                if sdf_is_hfield:
                    sdf_scale = wp.vec3(1.0, 1.0, 1.0)
                else:
                    sdf_scale = mesh_scale_sdf
                    if not use_bvh_for_sdf:
                        texture_sdf = texture_sdf_table[sdf_idx]
                        if texture_sdf.scale_baked:
                            sdf_scale = wp.vec3(1.0, 1.0, 1.0)

                X_mesh_to_sdf = wp.transform_multiply(wp.transform_inverse(X_sdf_ws), X_tri_ws)

                triangle_mesh_margin = scale_data_tri[3]
                sdf_mesh_margin = scale_data_sdf[3]

                sdf_scale_safe = wp.vec3(
                    wp.max(sdf_scale[0], 1e-10),
                    wp.max(sdf_scale[1], 1e-10),
                    wp.max(sdf_scale[2], 1e-10),
                )
                inv_sdf_scale = wp.cw_div(wp.vec3(1.0, 1.0, 1.0), sdf_scale_safe)
                min_sdf_scale = wp.min(wp.min(sdf_scale_safe[0], sdf_scale_safe[1]), sdf_scale_safe[2])

                contact_threshold = gap_sum + triangle_mesh_margin + sdf_mesh_margin
                contact_threshold_unscaled = contact_threshold / min_sdf_scale

                tri_capacity = wp.block_dim()
                selected_triangles = wp.array(
                    ptr=get_shared_memory_pointer_block_dim_plus_2_ints(),
                    shape=(wp.block_dim() + 2,),
                    dtype=wp.int32,
                )
                vertex_cache = wp.array(
                    ptr=_get_shared_memory_vertex_cache(),
                    shape=(wp.block_dim() * 3,),
                    dtype=wp.vec3,
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
                        texture_sdf,
                        mesh_id_sdf,
                        selected_triangles,
                        contact_threshold_unscaled,
                        use_bvh_for_sdf,
                        inv_sdf_scale,
                        num_tris,
                        tri_type,
                        sdf_type,
                        hfd_tri,
                        hfd_sdf,
                        heightfield_elevations,
                        vertex_cache,
                    )

                    has_triangle = t < selected_triangles[tri_capacity]
                    synchronize()

                    if has_triangle:
                        v0 = vertex_cache[t * 3]
                        v1 = vertex_cache[t * 3 + 1]
                        v2 = vertex_cache[t * 3 + 2]

                        dist_unscaled, point_unscaled, direction_unscaled = do_triangle_sdf_collision(
                            texture_sdf,
                            mesh_id_sdf,
                            v0,
                            v1,
                            v2,
                            use_bvh_for_sdf,
                            sdf_is_hfield,
                            hfd_sdf,
                            heightfield_elevations,
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
                            else:
                                fallback_dir = point_world - wp.transform_get_translation(X_sdf_ws)
                                fallback_len = wp.length(fallback_dir)
                                if fallback_len > 0.0:
                                    direction_world = fallback_dir / fallback_len
                                else:
                                    direction_world = wp.vec3(0.0, 1.0, 0.0)

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
                            contact_data.gap_sum = gap_sum

                            writer_func(contact_data, writer_data, -1)

                    synchronize()
                    if t == 0:
                        selected_triangles[tri_capacity] = 0
                    synchronize()

    # Return early if contact reduction is disabled
    if not reduce_contacts:
        return mesh_sdf_collision_kernel

    # =========================================================================
    # Global reduction variant: uses hashtable instead of shared-memory reduction.
    # Same block_offsets load balancing and shared-memory triangle selection,
    # but contacts are written directly to global buffer + hashtable.
    # =========================================================================

    @wp.kernel(enable_backward=False, module="unique")
    def mesh_sdf_collision_global_reduce_kernel(
        shape_data: wp.array(dtype=wp.vec4),
        shape_transform: wp.array(dtype=wp.transform),
        shape_source: wp.array(dtype=wp.uint64),
        texture_sdf_table: wp.array(dtype=TextureSDFData),
        shape_sdf_index: wp.array(dtype=wp.int32),
        shape_gap: wp.array(dtype=float),
        shape_collision_aabb_lower: wp.array(dtype=wp.vec3),
        shape_collision_aabb_upper: wp.array(dtype=wp.vec3),
        shape_voxel_resolution: wp.array(dtype=wp.vec3i),
        shape_pairs_mesh_mesh: wp.array(dtype=wp.vec2i),
        shape_pairs_mesh_mesh_count: wp.array(dtype=int),
        shape_heightfield_index: wp.array(dtype=wp.int32),
        heightfield_data: wp.array(dtype=HeightfieldData),
        heightfield_elevations: wp.array(dtype=wp.float32),
        block_offsets: wp.array(dtype=wp.int32),
        reducer_data: GlobalContactReducerData,
        total_num_blocks: int,
    ):
        """Process mesh-mesh collisions with global hashtable contact reduction.

        Same load balancing and triangle selection as the thread-block reduce kernel,
        but contacts are written directly to the global buffer and registered in the
        hashtable inline, matching thread-block reduction contact quality:

        - Midpoint-centered position for spatial extreme projection
        - Fixed beta threshold (0.0001 m)
        - Tri-shape AABB for voxel computation (alternates per mode)
        """
        block_id, t = wp.tid()
        pair_count = wp.min(shape_pairs_mesh_mesh_count[0], shape_pairs_mesh_mesh.shape[0])
        total_combos = block_offsets[pair_count]

        for combo_idx in range(block_id, total_combos, total_num_blocks):
            lo = int(0)
            hi = int(pair_count)
            while lo < hi:
                mid = (lo + hi) // 2
                if block_offsets[mid + 1] <= combo_idx:
                    lo = mid + 1
                else:
                    hi = mid
            pair_idx = int(lo)
            pair_block_start = block_offsets[pair_idx]
            block_in_pair = combo_idx - pair_block_start
            blocks_for_pair = block_offsets[pair_idx + 1] - pair_block_start
            pair_encoded = shape_pairs_mesh_mesh[pair_idx]
            if wp.static(enable_heightfields):
                has_hfield = (pair_encoded[0] & SHAPE_PAIR_HFIELD_BIT) != 0
                pair = wp.vec2i(pair_encoded[0] & SHAPE_PAIR_INDEX_MASK, pair_encoded[1])
            else:
                has_hfield = False
                pair = pair_encoded

            gap_sum = shape_gap[pair[0]] + shape_gap[pair[1]]

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
                        hfd_tri = heightfield_data[shape_heightfield_index[tri_shape]]
                    if sdf_is_hfield:
                        hfd_sdf = heightfield_data[shape_heightfield_index[sdf_shape]]

                use_bvh_for_sdf = False
                if not sdf_is_hfield:
                    sdf_idx = shape_sdf_index[sdf_shape]
                    use_bvh_for_sdf = sdf_idx < 0 or sdf_idx >= texture_sdf_table.shape[0]
                    if not use_bvh_for_sdf:
                        use_bvh_for_sdf = texture_sdf_table[sdf_idx].coarse_texture.width == 0

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

                texture_sdf = TextureSDFData()
                if sdf_is_hfield:
                    sdf_scale = wp.vec3(1.0, 1.0, 1.0)
                else:
                    sdf_scale = mesh_scale_sdf
                    if not use_bvh_for_sdf:
                        texture_sdf = texture_sdf_table[sdf_idx]
                        if texture_sdf.scale_baked:
                            sdf_scale = wp.vec3(1.0, 1.0, 1.0)

                X_mesh_to_sdf = wp.transform_multiply(wp.transform_inverse(X_sdf_ws), X_tri_ws)

                triangle_mesh_margin = scale_data_tri[3]
                sdf_mesh_margin = scale_data_sdf[3]

                midpoint = (wp.transform_get_translation(X_tri_ws) + wp.transform_get_translation(X_sdf_ws)) * 0.5

                sdf_scale_safe = wp.vec3(
                    wp.max(sdf_scale[0], 1e-10),
                    wp.max(sdf_scale[1], 1e-10),
                    wp.max(sdf_scale[2], 1e-10),
                )
                inv_sdf_scale = wp.cw_div(wp.vec3(1.0, 1.0, 1.0), sdf_scale_safe)
                min_sdf_scale = wp.min(wp.min(sdf_scale_safe[0], sdf_scale_safe[1]), sdf_scale_safe[2])

                contact_threshold = gap_sum + triangle_mesh_margin + sdf_mesh_margin
                contact_threshold_unscaled = contact_threshold / min_sdf_scale

                tri_capacity = wp.block_dim()
                selected_triangles = wp.array(
                    ptr=get_shared_memory_pointer_block_dim_plus_2_ints(),
                    shape=(wp.block_dim() + 2,),
                    dtype=wp.int32,
                )
                vertex_cache = wp.array(
                    ptr=_get_shared_memory_vertex_cache(),
                    shape=(wp.block_dim() * 3,),
                    dtype=wp.vec3,
                )

                num_tris = get_triangle_count(tri_type, mesh_id_tri, hfd_tri)
                chunk_size = (num_tris + blocks_for_pair - 1) // blocks_for_pair
                tri_start = block_in_pair * chunk_size
                tri_end = wp.min(tri_start + chunk_size, num_tris)

                if t == 0:
                    selected_triangles[tri_capacity] = 0
                    selected_triangles[tri_capacity + 1] = tri_start
                synchronize()

                while selected_triangles[tri_capacity + 1] < tri_end:
                    find_interesting_triangles(
                        t,
                        mesh_scale_tri,
                        X_mesh_to_sdf,
                        mesh_id_tri,
                        texture_sdf,
                        mesh_id_sdf,
                        selected_triangles,
                        contact_threshold_unscaled,
                        use_bvh_for_sdf,
                        inv_sdf_scale,
                        tri_end,
                        tri_type,
                        sdf_type,
                        hfd_tri,
                        hfd_sdf,
                        heightfield_elevations,
                        vertex_cache,
                    )

                    has_triangle = t < selected_triangles[tri_capacity]
                    synchronize()

                    if has_triangle:
                        v0 = vertex_cache[t * 3]
                        v1 = vertex_cache[t * 3 + 1]
                        v2 = vertex_cache[t * 3 + 2]

                        dist_unscaled, point_unscaled, direction_unscaled = do_triangle_sdf_collision(
                            texture_sdf,
                            mesh_id_sdf,
                            v0,
                            v1,
                            v2,
                            use_bvh_for_sdf,
                            sdf_is_hfield,
                            hfd_sdf,
                            heightfield_elevations,
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
                            else:
                                fallback_dir = point_world - wp.transform_get_translation(X_sdf_ws)
                                fallback_len = wp.length(fallback_dir)
                                if fallback_len > 0.0:
                                    direction_world = fallback_dir / fallback_len
                                else:
                                    direction_world = wp.vec3(0.0, 1.0, 0.0)

                            contact_normal = -direction_world if mode == 0 else direction_world

                            export_and_reduce_contact_centered(
                                pair[0],
                                pair[1],
                                point_world,
                                contact_normal,
                                dist,
                                point_world - midpoint,
                                X_ws_tri,
                                aabb_lower_tri,
                                aabb_upper_tri,
                                voxel_res_tri,
                                reducer_data,
                            )

                    synchronize()
                    if t == 0:
                        selected_triangles[tri_capacity] = 0
                    synchronize()

    return mesh_sdf_collision_global_reduce_kernel
