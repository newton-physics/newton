# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

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

# Shared memory for caching midpoint SDF values from culling (1 float per edge slot).
_get_shared_memory_sdf_cache = create_shared_memory_pointer_block_dim_mul_func(1)


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
    add_edge: bool,
    edge_idx: int,
    buffer: wp.array[wp.int32],
    sdf_cache: wp.array[float],
    sdf_value: float,
):
    """Add an edge index to a shared memory buffer using atomic operations.

    Buffer layout:
    - [0 .. block_dim-1]: Edge indices
    - [block_dim]: Current count of edges in buffer
    - [block_dim+1]: Progress counter (edges processed so far)

    Args:
        thread_id: The calling thread's index within the thread block
        add_edge: Whether this thread wants to add an edge
        edge_idx: The edge index to add (only used if add_edge is True)
        buffer: Shared memory buffer for edge indices
        sdf_cache: Shared memory for cached midpoint SDF values
        sdf_value: SDF value at the edge midpoint (from culling)
    """
    capacity = wp.block_dim()
    idx = -1

    if add_edge:
        idx = wp.atomic_add(buffer, capacity, 1)
        if idx < capacity:
            buffer[idx] = edge_idx
            sdf_cache[idx] = sdf_value

    if thread_id == 0:
        buffer[capacity + 1] += capacity

    synchronize()  # SYNC 1: All atomic writes and progress update complete

    if thread_id == 0 and buffer[capacity] > capacity:
        buffer[capacity] = capacity

    if add_edge and idx >= capacity:
        wp.atomic_min(buffer, capacity + 1, edge_idx)

    synchronize()  # SYNC 2: All corrections complete, buffer consistent


@wp.func
def get_edge_from_mesh(
    mesh_id: wp.uint64,
    mesh_edge_indices: wp.array[wp.vec2i],
    edge_range: wp.vec2i,
    mesh_scale: wp.vec3,
    X_mesh_ws: wp.transform,
    edge_idx: int,
) -> tuple[wp.vec3, wp.vec3]:
    """Extract an edge from a mesh and transform it to world space.

    Reads the edge vertex pair from the packed ``mesh_edge_indices`` array
    using the per-shape ``edge_range`` offset, and returns both endpoints
    in world space after applying scale and transform.

    Args:
        mesh_id: The mesh ID (use wp.mesh_get to retrieve the mesh object)
        mesh_edge_indices: Packed array of all mesh edge vertex pairs.
        edge_range: ``(start, count)`` slice for this shape into ``mesh_edge_indices``.
        mesh_scale: Scale to apply to mesh vertices (component-wise)
        X_mesh_ws: Mesh world-space transform (position and rotation)
        edge_idx: Edge index within this shape (0-based)

    Returns:
        Tuple of (v0_world, v1_world) - the two edge endpoints in world space.
    """
    mesh = wp.mesh_get(mesh_id)
    edge = mesh_edge_indices[edge_range[0] + edge_idx]

    idx0 = edge[0]
    idx1 = edge[1]

    v0_local = wp.cw_mul(mesh.points[idx0], mesh_scale)
    v1_local = wp.cw_mul(mesh.points[idx1], mesh_scale)

    v0_world = wp.transform_point(X_mesh_ws, v0_local)
    v1_world = wp.transform_point(X_mesh_ws, v1_local)

    return v0_world, v1_world


@wp.func
def get_edge_from_heightfield(
    hfd: HeightfieldData,
    elevation_data: wp.array[wp.float32],
    X_ws: wp.transform,
    edge_idx: int,
) -> tuple[wp.vec3, wp.vec3]:
    """Extract an edge from a heightfield by linear edge index.

    Heightfield edges are enumerated in three groups:

    - Horizontal edges: ``nrow * (ncol - 1)`` edges along rows.
    - Vertical edges: ``(nrow - 1) * ncol`` edges along columns.
    - Diagonal edges: ``(nrow - 1) * (ncol - 1)`` edges across cells.

    ``hfd`` already carries the per-instance scale baked into ``hx``, ``hy``,
    ``min_z``, and ``max_z`` by the builder, so the returned vertices do not
    need a further scale multiplication.

    Args:
        hfd: Heightfield descriptor (extents are scale-baked).
        elevation_data: Flat elevation array.
        X_ws: World-space transform.
        edge_idx: Linear edge index (0-based).

    Returns:
        Tuple of (v0_world, v1_world) - the two edge endpoints in world space.
    """
    nrow = hfd.nrow
    ncol = hfd.ncol

    dx = 2.0 * hfd.hx / wp.float32(ncol - 1)
    dy = 2.0 * hfd.hy / wp.float32(nrow - 1)
    z_range = hfd.max_z - hfd.min_z
    base = hfd.data_offset

    num_h = nrow * (ncol - 1)
    num_v = (nrow - 1) * ncol

    r0 = int(0)
    c0 = int(0)
    r1 = int(0)
    c1 = int(0)

    if edge_idx < num_h:
        # Horizontal edge
        r0 = edge_idx // (ncol - 1)
        c0 = edge_idx - r0 * (ncol - 1)
        r1 = r0
        c1 = c0 + 1
    elif edge_idx < num_h + num_v:
        # Vertical edge
        local = edge_idx - num_h
        r0 = local // ncol
        c0 = local - r0 * ncol
        r1 = r0 + 1
        c1 = c0
    else:
        # Diagonal edge
        local = edge_idx - num_h - num_v
        r0 = local // (ncol - 1)
        c0 = local - r0 * (ncol - 1)
        r1 = r0 + 1
        c1 = c0 + 1

    x0 = -hfd.hx + wp.float32(c0) * dx
    y0 = -hfd.hy + wp.float32(r0) * dy
    h0 = elevation_data[base + r0 * ncol + c0]
    p0 = wp.vec3(x0, y0, hfd.min_z + h0 * z_range)

    x1 = -hfd.hx + wp.float32(c1) * dx
    y1 = -hfd.hy + wp.float32(r1) * dy
    h1 = elevation_data[base + r1 * ncol + c1]
    p1 = wp.vec3(x1, y1, hfd.min_z + h1 * z_range)

    v0_world = wp.transform_point(X_ws, p0)
    v1_world = wp.transform_point(X_ws, p1)

    return v0_world, v1_world


@wp.func
def get_edge_bounding_sphere(v0: wp.vec3, v1: wp.vec3) -> tuple[wp.vec3, float]:
    """Compute the bounding sphere for an edge (midpoint and half-length).

    Args:
        v0: First edge endpoint.
        v1: Second edge endpoint.

    Returns:
        Tuple of (midpoint, half_length).
    """
    midpoint = (v0 + v1) * 0.5
    half_length = wp.length(v1 - v0) * 0.5
    return midpoint, half_length


@wp.func
def get_triangle_count(shape_type: int, mesh_id: wp.uint64, hfd: HeightfieldData) -> int:
    """Return the number of triangles for a mesh or heightfield shape."""
    if shape_type == GeoType.HFIELD:
        if hfd.nrow <= 1 or hfd.ncol <= 1:
            return 0
        return 2 * (hfd.nrow - 1) * (hfd.ncol - 1)
    return wp.mesh_get(mesh_id).indices.shape[0] // 3


@wp.func
def get_edge_count(shape_type: int, edge_range: wp.vec2i, hfd: HeightfieldData) -> int:
    """Return the number of edges for a mesh or heightfield shape."""
    if shape_type == GeoType.HFIELD:
        if hfd.nrow <= 1 or hfd.ncol <= 1:
            return 0
        return hfd.nrow * (hfd.ncol - 1) + (hfd.nrow - 1) * hfd.ncol + (hfd.nrow - 1) * (hfd.ncol - 1)
    return edge_range[1]


def _create_sdf_contact_funcs(enable_heightfields: bool):
    """Generate SDF contact functions with heightfield branches eliminated at compile time.

    When ``enable_heightfields`` is False, ``wp.static`` strips all heightfield code
    paths from the generated functions, reducing register pressure and instruction
    cache footprint — especially in the 6-iteration Brent's method loop of
    ``do_edge_sdf_collision``.

    Args:
        enable_heightfields: When False, all heightfield code paths are compiled out.

    Returns:
        Tuple of ``(find_interesting_edges, do_edge_sdf_collision)``.
    """

    @wp.func
    def _sample_sdf_at_t(
        texture_sdf: TextureSDFData,
        sdf_mesh_id: wp.uint64,
        v0: wp.vec3,
        edge_dir: wp.vec3,
        tt: float,
        use_bvh_for_sdf: bool,
        sdf_is_heightfield: bool,
        hfd_sdf: HeightfieldData,
        elevation_data: wp.array[wp.float32],
    ) -> float:
        """Sample SDF at the point ``v0 + tt * edge_dir``."""
        pp = v0 + edge_dir * tt
        if wp.static(enable_heightfields):
            if sdf_is_heightfield:
                return sample_sdf_heightfield(hfd_sdf, elevation_data, pp)
            elif use_bvh_for_sdf:
                return sample_sdf_using_mesh(sdf_mesh_id, pp)
            else:
                return texture_sample_sdf(texture_sdf, pp)
        else:
            if use_bvh_for_sdf:
                return sample_sdf_using_mesh(sdf_mesh_id, pp)
            else:
                return texture_sample_sdf(texture_sdf, pp)

    @wp.func
    def do_edge_sdf_collision_func(
        texture_sdf: TextureSDFData,
        sdf_mesh_id: wp.uint64,
        v0: wp.vec3,
        v1: wp.vec3,
        midpoint_sdf: float,
        use_bvh_for_sdf: bool,
        sdf_is_heightfield: bool,
        hfd_sdf: HeightfieldData,
        elevation_data: wp.array[wp.float32],
    ) -> tuple[float, wp.vec3]:
        """Find the deepest point on an edge relative to an SDF volume.

        Uses Brent's method (5 iterations) to minimize the SDF value along the
        edge parameterized as ``p(t) = v0 + t * edge_dir`` for t in [0, 1].
        The initial midpoint SDF value is provided by the caller (cached from
        culling) to avoid a redundant evaluation.

        After the interior search, evaluates the more promising endpoint
        (the one closer to the unconverged bracket boundary) so that vertex
        contacts at edge corners are not missed.

        Returns:
            Tuple of (distance, contact_point).
        """
        golden = 0.3819660112501051  # (3 - sqrt(5)) / 2
        edge_dir = v1 - v0

        # Initialize Brent's method at the midpoint (SDF value from culling)
        a = float(0.0)
        b = float(1.0)
        x = float(0.5)
        w = float(0.5)
        v_brent = float(0.5)
        fx = midpoint_sdf
        fw = fx
        fv = fx
        d_step = float(0.0)
        e_step = float(0.0)

        for _iter in range(5):
            m = 0.5 * (a + b)
            tol = 1.0e-2 * wp.abs(x) + 1.0e-8
            tol2 = 2.0 * tol

            if wp.abs(x - m) <= tol2 - 0.5 * (b - a):
                break

            # Try inverse parabolic interpolation
            use_parabolic = False
            p_num = float(0.0)
            q_denom = float(0.0)

            if wp.abs(e_step) > tol:
                r = (x - w) * (fx - fv)
                q_denom = (x - v_brent) * (fx - fw)
                p_num = (x - v_brent) * q_denom - (x - w) * r
                q_denom = 2.0 * (q_denom - r)
                if q_denom > 0.0:
                    p_num = -p_num
                else:
                    q_denom = -q_denom

                # Check if parabolic step is acceptable
                if wp.abs(p_num) < 0.5 * wp.abs(q_denom * e_step):
                    trial = p_num / q_denom
                    u_trial = x + trial
                    if u_trial - a >= tol2 and b - u_trial >= tol2:
                        use_parabolic = True

            if use_parabolic:
                e_step = d_step
                d_step = p_num / q_denom
            else:
                # Golden section step
                if x >= m:
                    e_step = a - x
                else:
                    e_step = b - x
                d_step = golden * e_step

            # Evaluate new point
            if wp.abs(d_step) >= tol:
                u = x + d_step
            else:
                if d_step > 0.0:
                    u = x + tol
                else:
                    u = x - tol

            fu = _sample_sdf_at_t(
                texture_sdf,
                sdf_mesh_id,
                v0,
                edge_dir,
                u,
                use_bvh_for_sdf,
                sdf_is_heightfield,
                hfd_sdf,
                elevation_data,
            )

            # Update bracket
            if fu <= fx:
                if u < x:
                    b = x
                else:
                    a = x
                v_brent = w
                fv = fw
                w = x
                fw = fx
                x = u
                fx = fu
            else:
                if u < x:
                    a = u
                else:
                    b = u
                if fu <= fw or w == x:
                    v_brent = w
                    fv = fw
                    w = u
                    fw = fu
                elif fu <= fv or v_brent == x or v_brent == w:
                    v_brent = u
                    fv = fu

        # Check the closer endpoint only when Brent converged near a
        # boundary (x < 0.2 or x > 0.8).  When solidly interior the
        # bracket has moved both boundaries inward, so the endpoint
        # cannot beat the interior minimum.
        best_t = x
        best_f = fx
        if x < 0.2 or x > 0.8:
            check_t = 0.0 if x < 0.5 else 1.0
            f_end = _sample_sdf_at_t(
                texture_sdf,
                sdf_mesh_id,
                v0,
                edge_dir,
                check_t,
                use_bvh_for_sdf,
                sdf_is_heightfield,
                hfd_sdf,
                elevation_data,
            )
            if f_end < best_f:
                best_t = check_t
                best_f = f_end

        p = v0 + edge_dir * best_t

        return best_f, p

    @wp.func
    def find_interesting_edges_func(
        thread_id: int,
        mesh_scale: wp.vec3,
        mesh_to_sdf_transform: wp.transform,
        mesh_id: wp.uint64,
        mesh_edge_indices: wp.array[wp.vec2i],
        edge_range: wp.vec2i,
        texture_sdf: TextureSDFData,
        sdf_mesh_id: wp.uint64,
        buffer: wp.array[wp.int32],
        contact_distance: float,
        use_bvh_for_sdf: bool,
        inv_sdf_scale: wp.vec3,
        edge_end: int,
        edge_shape_type: int,
        sdf_shape_type: int,
        hfd_edge: HeightfieldData,
        hfd_sdf: HeightfieldData,
        elevation_data: wp.array[wp.float32],
        sdf_cache: wp.array[float],
    ):
        """Midphase edge culling for mesh-SDF collision.

        Determines which edges are close enough to the SDF to potentially
        generate contacts.  Accepted edge indices are written to the shared
        ``buffer``; the midpoint SDF value is cached in ``sdf_cache`` so
        that Brent's method can skip its initial evaluation. The consumer
        re-fetches vertices from global memory
        (relying on L2 cache locality from this culling pass).

        Buffer layout: [0..block_dim-1] = edge indices, [block_dim] = count,
        [block_dim+1] = progress.
        """
        capacity = wp.block_dim()

        if wp.static(enable_heightfields):
            sdf_is_heightfield = sdf_shape_type == GeoType.HFIELD
        else:
            sdf_is_heightfield = False

        sdf_aabb_lower = texture_sdf.sdf_box_lower
        sdf_aabb_upper = texture_sdf.sdf_box_upper

        synchronize()  # Ensure buffer state is consistent before starting

        while buffer[capacity + 1] < edge_end and buffer[capacity] < capacity:
            base_edge_idx = buffer[capacity + 1]
            edge_idx = base_edge_idx + thread_id
            add_edge = False
            midpoint_sdf = float(0.0)

            if edge_idx < edge_end:
                if wp.static(enable_heightfields):
                    if edge_shape_type == GeoType.HFIELD:
                        v0_scaled, v1_scaled = get_edge_from_heightfield(
                            hfd_edge, elevation_data, mesh_to_sdf_transform, edge_idx
                        )
                    else:
                        v0_scaled, v1_scaled = get_edge_from_mesh(
                            mesh_id, mesh_edge_indices, edge_range, mesh_scale, mesh_to_sdf_transform, edge_idx
                        )
                else:
                    v0_scaled, v1_scaled = get_edge_from_mesh(
                        mesh_id, mesh_edge_indices, edge_range, mesh_scale, mesh_to_sdf_transform, edge_idx
                    )
                v0 = wp.cw_mul(v0_scaled, inv_sdf_scale)
                v1 = wp.cw_mul(v1_scaled, inv_sdf_scale)
                bounding_sphere_center, bounding_sphere_radius = get_edge_bounding_sphere(v0, v1)

                threshold = bounding_sphere_radius + contact_distance

                if sdf_is_heightfield:
                    midpoint_sdf = sample_sdf_heightfield(hfd_sdf, elevation_data, bounding_sphere_center)
                    add_edge = midpoint_sdf <= threshold
                elif use_bvh_for_sdf:
                    midpoint_sdf = sample_sdf_using_mesh(sdf_mesh_id, bounding_sphere_center, 1.01 * threshold)
                    add_edge = midpoint_sdf <= threshold
                else:
                    culling_radius = threshold
                    clamped = wp.min(wp.max(bounding_sphere_center, sdf_aabb_lower), sdf_aabb_upper)
                    aabb_dist_sq = wp.length_sq(bounding_sphere_center - clamped)
                    if aabb_dist_sq > culling_radius * culling_radius:
                        add_edge = False
                    else:
                        midpoint_sdf = texture_sample_sdf(texture_sdf, bounding_sphere_center)
                        add_edge = midpoint_sdf <= culling_radius

            synchronize()  # Ensure all threads have read base_edge_idx before any writes
            add_to_shared_buffer_atomic(
                thread_id,
                add_edge,
                edge_idx,
                buffer,
                sdf_cache,
                midpoint_sdf,
            )
            # add_to_shared_buffer_atomic ends with sync, buffer is consistent for next while check

        synchronize()  # Final sync before returning

    return find_interesting_edges_func, do_edge_sdf_collision_func


@wp.kernel(enable_backward=False)
def compute_mesh_mesh_edge_counts(
    shape_pairs_mesh_mesh: wp.array[wp.vec2i],
    shape_pairs_mesh_mesh_count: wp.array[int],
    shape_edge_range: wp.array[wp.vec2i],
    shape_heightfield_index: wp.array[wp.int32],
    heightfield_data: wp.array[HeightfieldData],
    edge_counts: wp.array[wp.int32],
):
    """Compute per-pair edge counts for mesh-mesh (or heightfield-mesh) pairs.

    Sums the edge counts of both shapes in each pair — each shape may be
    a triangle mesh or a heightfield.  Each thread handles one slot in the
    ``edge_counts`` array.  Slots beyond ``pair_count`` are zeroed so that a
    subsequent ``array_scan`` over the full array produces correct prefix sums.
    """
    i = wp.tid()
    pair_count = wp.min(shape_pairs_mesh_mesh_count[0], shape_pairs_mesh_mesh.shape[0])
    if i >= pair_count:
        edge_counts[i] = 0
        return

    pair_encoded = shape_pairs_mesh_mesh[i]
    has_hfield = (pair_encoded[0] & SHAPE_PAIR_HFIELD_BIT) != 0
    pair = wp.vec2i(pair_encoded[0] & SHAPE_PAIR_INDEX_MASK, pair_encoded[1])
    pair_edges = int(0)
    for mode in range(2):
        is_hfield = has_hfield and mode == 0
        shape_idx = pair[mode]
        if is_hfield:
            hfd = heightfield_data[shape_heightfield_index[shape_idx]]
            pair_edges += get_edge_count(GeoType.HFIELD, wp.vec2i(-1, 0), hfd)
        else:
            pair_edges += shape_edge_range[shape_idx][1]
    edge_counts[i] = wp.int32(pair_edges)


@wp.kernel(enable_backward=False)
def compute_block_counts_from_weights(
    weight_prefix_sums: wp.array[wp.int32],
    weights: wp.array[wp.int32],
    pair_count_arr: wp.array[int],
    max_pairs: int,
    target_blocks: int,
    block_counts: wp.array[wp.int32],
):
    """Convert per-pair weights to block counts using adaptive load balancing.

    Reads the total weight from the inclusive prefix sum to compute the
    adaptive ``weight_per_block`` threshold, then assigns each pair a
    block count proportional to its weight.  Slots beyond ``pair_count``
    are zeroed for a subsequent exclusive ``array_scan``.
    """
    i = wp.tid()
    pair_count = wp.min(pair_count_arr[0], max_pairs)
    if i >= pair_count:
        block_counts[i] = 0
        return

    # Read total from inclusive prefix sum
    total_weight = weight_prefix_sums[pair_count - 1]
    weight_per_block = int(total_weight)
    if target_blocks > 0 and total_weight > 0:
        weight_per_block = wp.max(256, total_weight // target_blocks)

    w = int(weights[i])
    if weight_per_block > 0:
        blocks = wp.max(1, (w + weight_per_block - 1) // weight_per_block)
    else:
        blocks = 1
    block_counts[i] = wp.int32(blocks)


def compute_mesh_mesh_block_offsets_scan(
    shape_pairs_mesh_mesh: wp.array,
    shape_pairs_mesh_mesh_count: wp.array,
    shape_edge_range: wp.array,
    shape_heightfield_index: wp.array,
    heightfield_data: wp.array,
    target_blocks: int,
    block_offsets: wp.array,
    block_counts: wp.array,
    weight_prefix_sums: wp.array,
    device: str | None = None,
    record_tape: bool = True,
):
    """Compute mesh-mesh block offsets using parallel kernels and array_scan.

    Runs a four-stage parallel pipeline: per-pair edge counts →
    inclusive scan → adaptive block counts → exclusive scan into
    ``block_offsets``.
    """
    n = block_counts.shape[0]
    # Step 1: compute per-pair edge counts in parallel
    wp.launch(
        kernel=compute_mesh_mesh_edge_counts,
        dim=n,
        inputs=[
            shape_pairs_mesh_mesh,
            shape_pairs_mesh_mesh_count,
            shape_edge_range,
            shape_heightfield_index,
            heightfield_data,
            block_counts,  # reuse as temp storage for edge counts
        ],
        device=device,
        record_tape=record_tape,
    )
    # Step 2: inclusive scan to get total in last element
    wp.utils.array_scan(block_counts, weight_prefix_sums, inclusive=True)
    # Step 3: compute per-pair block counts using adaptive threshold
    wp.launch(
        kernel=compute_block_counts_from_weights,
        dim=n,
        inputs=[
            weight_prefix_sums,
            block_counts,  # still holds tri counts
            shape_pairs_mesh_mesh_count,
            shape_pairs_mesh_mesh.shape[0],
            target_blocks,
            block_offsets,  # reuse as temp for block counts
        ],
        device=device,
        record_tape=record_tape,
    )
    # Step 4: exclusive scan of block counts → block_offsets
    wp.utils.array_scan(block_offsets, block_offsets, inclusive=False)


def create_narrow_phase_process_mesh_mesh_contacts_kernel(
    writer_func: Any,
    enable_heightfields: bool = True,
    reduce_contacts: bool = False,
):
    find_interesting_edges, do_edge_sdf_collision = _create_sdf_contact_funcs(enable_heightfields)

    # Derive a stable module name from the factory arguments so that
    # identical configurations share the compiled CUDA kernel.  This is
    # critical for deterministic contact generation: two CollisionPipeline
    # instances with the same writer_func must execute the exact same
    # compiled code, otherwise FMA-fusion or register-allocation
    # differences between independent JIT compilations can produce subtly
    # different floating-point results, breaking bit-exact reproducibility.
    _module = f"sdf_contact_{writer_func.__name__}_{enable_heightfields}_{reduce_contacts}"

    @wp.kernel(enable_backward=False, module=_module)
    def mesh_sdf_collision_kernel(
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        texture_sdf_table: wp.array[TextureSDFData],
        shape_sdf_index: wp.array[wp.int32],
        shape_gap: wp.array[float],
        _shape_collision_aabb_lower: wp.array[wp.vec3],
        _shape_collision_aabb_upper: wp.array[wp.vec3],
        _shape_voxel_resolution: wp.array[wp.vec3i],
        shape_pairs_mesh_mesh: wp.array[wp.vec2i],
        shape_pairs_mesh_mesh_count: wp.array[int],
        shape_heightfield_index: wp.array[wp.int32],
        heightfield_data: wp.array[HeightfieldData],
        heightfield_elevations: wp.array[wp.float32],
        mesh_edge_indices: wp.array[wp.vec2i],
        shape_edge_range: wp.array[wp.vec2i],
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

                edge_capacity = wp.block_dim()
                selected_edges = wp.array(
                    ptr=get_shared_memory_pointer_block_dim_plus_2_ints(),
                    shape=(wp.block_dim() + 2,),
                    dtype=wp.int32,
                )
                sdf_cache = wp.array(
                    ptr=_get_shared_memory_sdf_cache(),
                    shape=(wp.block_dim(),),
                    dtype=float,
                )
                if t == 0:
                    selected_edges[edge_capacity] = 0
                    selected_edges[edge_capacity + 1] = 0
                synchronize()

                edge_range_tri = shape_edge_range[tri_shape]
                num_edges = get_edge_count(tri_type, edge_range_tri, hfd_tri)

                while selected_edges[edge_capacity + 1] < num_edges:
                    find_interesting_edges(
                        t,
                        mesh_scale_tri,
                        X_mesh_to_sdf,
                        mesh_id_tri,
                        mesh_edge_indices,
                        edge_range_tri,
                        texture_sdf,
                        mesh_id_sdf,
                        selected_edges,
                        contact_threshold_unscaled,
                        use_bvh_for_sdf,
                        inv_sdf_scale,
                        num_edges,
                        tri_type,
                        sdf_type,
                        hfd_tri,
                        hfd_sdf,
                        heightfield_elevations,
                        sdf_cache,
                    )

                    has_edge = t < selected_edges[edge_capacity]
                    synchronize()

                    if has_edge:
                        edge_idx = selected_edges[t]
                        cached_sdf = sdf_cache[t]
                        if wp.static(enable_heightfields):
                            if tri_type == GeoType.HFIELD:
                                v0s, v1s = get_edge_from_heightfield(
                                    hfd_tri, heightfield_elevations, X_mesh_to_sdf, edge_idx
                                )
                            else:
                                v0s, v1s = get_edge_from_mesh(
                                    mesh_id_tri,
                                    mesh_edge_indices,
                                    edge_range_tri,
                                    mesh_scale_tri,
                                    X_mesh_to_sdf,
                                    edge_idx,
                                )
                        else:
                            v0s, v1s = get_edge_from_mesh(
                                mesh_id_tri,
                                mesh_edge_indices,
                                edge_range_tri,
                                mesh_scale_tri,
                                X_mesh_to_sdf,
                                edge_idx,
                            )
                        v0 = wp.cw_mul(v0s, inv_sdf_scale)
                        v1 = wp.cw_mul(v1s, inv_sdf_scale)

                        dist_unscaled, point_unscaled = do_edge_sdf_collision(
                            texture_sdf,
                            mesh_id_sdf,
                            v0,
                            v1,
                            cached_sdf,
                            use_bvh_for_sdf,
                            sdf_is_hfield,
                            hfd_sdf,
                            heightfield_elevations,
                        )

                        # Quick threshold check before computing the gradient
                        dist_approx = dist_unscaled * min_sdf_scale
                        if dist_approx < contact_threshold:
                            if wp.static(enable_heightfields):
                                if sdf_is_hfield:
                                    dist_unscaled, direction_unscaled = sample_sdf_grad_heightfield(
                                        hfd_sdf, heightfield_elevations, point_unscaled
                                    )
                                elif use_bvh_for_sdf:
                                    dist_unscaled, direction_unscaled = sample_sdf_grad_using_mesh(
                                        mesh_id_sdf, point_unscaled
                                    )
                                else:
                                    dist_unscaled, direction_unscaled = texture_sample_sdf_grad(
                                        texture_sdf, point_unscaled
                                    )
                            else:
                                if use_bvh_for_sdf:
                                    dist_unscaled, direction_unscaled = sample_sdf_grad_using_mesh(
                                        mesh_id_sdf, point_unscaled
                                    )
                                else:
                                    dist_unscaled, direction_unscaled = texture_sample_sdf_grad(
                                        texture_sdf, point_unscaled
                                    )

                            dist, direction = scale_sdf_result_to_world(
                                dist_unscaled, direction_unscaled, sdf_scale, inv_sdf_scale, min_sdf_scale
                            )
                            point = wp.cw_mul(point_unscaled, sdf_scale)
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
                            contact_data.sort_sub_key = (edge_idx << 2) | (mode << 1)

                            writer_func(contact_data, writer_data, -1)

                    synchronize()
                    if t == 0:
                        selected_edges[edge_capacity] = 0
                    synchronize()

    # Return early if contact reduction is disabled
    if not reduce_contacts:
        return mesh_sdf_collision_kernel

    # =========================================================================
    # Global reduction variant: uses hashtable instead of shared-memory reduction.
    # Same block_offsets load balancing and shared-memory triangle selection,
    # but contacts are written directly to global buffer + hashtable.
    # =========================================================================

    @wp.kernel(enable_backward=False, module=_module)
    def mesh_sdf_collision_global_reduce_kernel(
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        texture_sdf_table: wp.array[TextureSDFData],
        shape_sdf_index: wp.array[wp.int32],
        shape_gap: wp.array[float],
        shape_collision_aabb_lower: wp.array[wp.vec3],
        shape_collision_aabb_upper: wp.array[wp.vec3],
        shape_voxel_resolution: wp.array[wp.vec3i],
        shape_pairs_mesh_mesh: wp.array[wp.vec2i],
        shape_pairs_mesh_mesh_count: wp.array[int],
        shape_heightfield_index: wp.array[wp.int32],
        heightfield_data: wp.array[HeightfieldData],
        heightfield_elevations: wp.array[wp.float32],
        mesh_edge_indices: wp.array[wp.vec2i],
        shape_edge_range: wp.array[wp.vec2i],
        block_offsets: wp.array[wp.int32],
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

                edge_capacity = wp.block_dim()
                selected_edges = wp.array(
                    ptr=get_shared_memory_pointer_block_dim_plus_2_ints(),
                    shape=(wp.block_dim() + 2,),
                    dtype=wp.int32,
                )
                sdf_cache = wp.array(
                    ptr=_get_shared_memory_sdf_cache(),
                    shape=(wp.block_dim(),),
                    dtype=float,
                )
                edge_range_tri = shape_edge_range[tri_shape]
                num_edges = get_edge_count(tri_type, edge_range_tri, hfd_tri)
                chunk_size = (num_edges + blocks_for_pair - 1) // blocks_for_pair
                edge_start = block_in_pair * chunk_size
                edge_end = wp.min(edge_start + chunk_size, num_edges)

                if t == 0:
                    selected_edges[edge_capacity] = 0
                    selected_edges[edge_capacity + 1] = edge_start
                synchronize()

                while selected_edges[edge_capacity + 1] < edge_end:
                    find_interesting_edges(
                        t,
                        mesh_scale_tri,
                        X_mesh_to_sdf,
                        mesh_id_tri,
                        mesh_edge_indices,
                        edge_range_tri,
                        texture_sdf,
                        mesh_id_sdf,
                        selected_edges,
                        contact_threshold_unscaled,
                        use_bvh_for_sdf,
                        inv_sdf_scale,
                        edge_end,
                        tri_type,
                        sdf_type,
                        hfd_tri,
                        hfd_sdf,
                        heightfield_elevations,
                        sdf_cache,
                    )

                    has_edge = t < selected_edges[edge_capacity]
                    synchronize()

                    if has_edge:
                        edge_idx = selected_edges[t]
                        cached_sdf = sdf_cache[t]
                        if wp.static(enable_heightfields):
                            if tri_type == GeoType.HFIELD:
                                v0s, v1s = get_edge_from_heightfield(
                                    hfd_tri, heightfield_elevations, X_mesh_to_sdf, edge_idx
                                )
                            else:
                                v0s, v1s = get_edge_from_mesh(
                                    mesh_id_tri,
                                    mesh_edge_indices,
                                    edge_range_tri,
                                    mesh_scale_tri,
                                    X_mesh_to_sdf,
                                    edge_idx,
                                )
                        else:
                            v0s, v1s = get_edge_from_mesh(
                                mesh_id_tri,
                                mesh_edge_indices,
                                edge_range_tri,
                                mesh_scale_tri,
                                X_mesh_to_sdf,
                                edge_idx,
                            )
                        v0 = wp.cw_mul(v0s, inv_sdf_scale)
                        v1 = wp.cw_mul(v1s, inv_sdf_scale)

                        dist_unscaled, point_unscaled = do_edge_sdf_collision(
                            texture_sdf,
                            mesh_id_sdf,
                            v0,
                            v1,
                            cached_sdf,
                            use_bvh_for_sdf,
                            sdf_is_hfield,
                            hfd_sdf,
                            heightfield_elevations,
                        )

                        # Quick threshold check before computing the gradient
                        dist_approx = dist_unscaled * min_sdf_scale
                        if dist_approx < contact_threshold:
                            if wp.static(enable_heightfields):
                                if sdf_is_hfield:
                                    dist_unscaled, direction_unscaled = sample_sdf_grad_heightfield(
                                        hfd_sdf, heightfield_elevations, point_unscaled
                                    )
                                elif use_bvh_for_sdf:
                                    dist_unscaled, direction_unscaled = sample_sdf_grad_using_mesh(
                                        mesh_id_sdf, point_unscaled
                                    )
                                else:
                                    dist_unscaled, direction_unscaled = texture_sample_sdf_grad(
                                        texture_sdf, point_unscaled
                                    )
                            else:
                                if use_bvh_for_sdf:
                                    dist_unscaled, direction_unscaled = sample_sdf_grad_using_mesh(
                                        mesh_id_sdf, point_unscaled
                                    )
                                else:
                                    dist_unscaled, direction_unscaled = texture_sample_sdf_grad(
                                        texture_sdf, point_unscaled
                                    )

                            dist, direction = scale_sdf_result_to_world(
                                dist_unscaled, direction_unscaled, sdf_scale, inv_sdf_scale, min_sdf_scale
                            )
                            point = wp.cw_mul(point_unscaled, sdf_scale)
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
                                (edge_idx << 2) | (mode << 1),
                                point_world - midpoint,
                                X_ws_tri,
                                aabb_lower_tri,
                                aabb_upper_tri,
                                voxel_res_tri,
                                reducer_data,
                            )

                    synchronize()
                    if t == 0:
                        selected_edges[edge_capacity] = 0
                    synchronize()

    return mesh_sdf_collision_global_reduce_kernel
