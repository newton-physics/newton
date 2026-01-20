import warp as wp
from newton._src.geometry.kernels import (
    triangle_closest_point,
    vertex_adjacent_to_triangle,
    VERTEX_COLLISION_BUFFER_OVERFLOW_INDEX,
    TRI_COLLISION_BUFFER_OVERFLOW_INDEX,
    TRI_CONTACT_FEATURE_VERTEX_A,
    TRI_CONTACT_FEATURE_VERTEX_B,
    TRI_CONTACT_FEATURE_VERTEX_C,
    TRI_CONTACT_FEATURE_EDGE_AB,
    TRI_CONTACT_FEATURE_EDGE_AC,
    TRI_CONTACT_FEATURE_EDGE_BC,
    TRI_CONTACT_FEATURE_FACE_INTERIOR,

)

@wp.func
def triangle_normal(v0: wp.vec3f, v1: wp.vec3f, v2: wp.vec3f) -> wp.vec3f:
    """
    Calculate the normal vector of a triangle from three vertices.
    
    Args:
        v0: First vertex of the triangle
        v1: Second vertex of the triangle  
        v2: Third vertex of the triangle
        
    Returns:
        Normalized normal vector of the triangle
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = wp.cross(edge1, edge2)
    return wp.normalize(normal)



@wp.kernel
def collide_triangles_vs_sphere(
    positions: wp.array(dtype=wp.vec3f),
    velocities: wp.array(dtype=wp.vec3f),
    inv_masses: wp.array(dtype=wp.float32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    sphere_center: wp.array(dtype=wp.vec3f),
    sphere_radius: wp.float32,
    restitution: wp.float32,
    dt: wp.float32,
    delta_accumulator: wp.array(dtype=wp.vec3f),
    delta_counter: wp.array(dtype=wp.int32)
):
    tid = wp.tid()
    if tid >= tri_indices.shape[0]:
        return
    
    t1 = tri_indices[tid, 0]
    t2 = tri_indices[tid, 1]
    t3 = tri_indices[tid, 2]

    p1 = positions[t1]
    p2 = positions[t2]
    p3 = positions[t3]

    w1 = inv_masses[t1]
    w2 = inv_masses[t2]
    w3 = inv_masses[t3]
    w = w1 + w2 + w3

    # Skip if all vertices are static (infinite mass)
    if w <= 0.0:
        return

    sp = sphere_center[0] * 0.01

    closest_p, bary, feature_type = triangle_closest_point(p1, p2, p3, sp)
    
    to_sphere = closest_p - sp
    dist = wp.length(to_sphere)
    
    if dist < sphere_radius:
        penetration = sphere_radius - dist
        
        # Avoid division by zero
        if dist > 1e-8:
            correction_dir = to_sphere / dist  # Normalized direction from sphere to triangle
        else:
            # Use triangle normal as fallback if closest point is exactly at sphere center
            tri_normal = triangle_normal(p1, p2, p3)
            correction_dir = tri_normal
        
        # Distribute correction based on barycentric coordinates and masses
        # Barycentric coordinates from triangle_closest_point: bary = (u, v, w) where u+v+w=1
        # p = u*p1 + v*p2 + w*p3, but we need to extract individual weights
        
        # For now, distribute evenly weighted by inverse mass
        total_correction = correction_dir * penetration
        d1 = total_correction * (w1 / w)
        d2 = total_correction * (w2 / w)
        d3 = total_correction * (w3 / w)

        wp.atomic_add(delta_accumulator, t1, d1)
        wp.atomic_add(delta_accumulator, t2, d2)
        wp.atomic_add(delta_accumulator, t3, d3)

        wp.atomic_add(delta_counter, t1, 1)
        wp.atomic_add(delta_counter, t2, 1)
        wp.atomic_add(delta_counter, t3, 1)

@wp.kernel
def vertex_triangle_collision_det(
    query_radius: float,
    bvh_id: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    vertex_colliding_triangles_offsets: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_buffer_sizes: wp.array(dtype=wp.int32),
    # outputs
    vertex_colliding_triangles: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_count: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_min_dist: wp.array(dtype=float),
    triangle_colliding_vertices_min_dist: wp.array(dtype=float),
    resize_flags: wp.array(dtype=wp.int32),
):
    """
    This function applies discrete collision detection between vertices and triangles. It uses pre-allocated spaces to
    record the collision data. Unlike `vertex_triangle_collision_detection_kernel`, this collision detection kernel
    works only in one way, i.e., it only records vertices' colliding triangles to `vertex_colliding_triangles`.

    This function assumes that all the vertices are on triangles, and can be indexed from the pos argument.

    Note:

        The collision date buffer is pre-allocated and cannot be changed during collision detection, therefore, the space
        may not be enough. If the space is not enough to record all the collision information, the function will set a
        certain element in resized_flag to be true. The user can reallocate the buffer based on vertex_colliding_triangles_count
        and vertex_colliding_triangles_count.

    Attributes:
        bvh_id (int): the bvh id you want to collide with
        query_radius (float): the contact radius. vertex-triangle pairs whose distance are less than this will get detected
        pos (array): positions of all the vertices that make up triangles
        vertex_colliding_triangles (array): flattened buffer of vertices' collision triangles, every two elements records
            the vertex index and a triangle index it collides to
        vertex_colliding_triangles_count (array): number of triangles each vertex collides
        vertex_colliding_triangles_offsets (array): where each vertex' collision buffer starts
        vertex_colliding_triangles_buffer_sizes (array): size of each vertex' collision buffer, will be modified if resizing is needed
        vertex_colliding_triangles_min_dist (array): each vertex' min distance to all (non-neighbor) triangles
        triangle_colliding_vertices_min_dist (array): each triangle's min distance to all (non-self) vertices
        resized_flag (array): size == 3, (vertex_buffer_resize_required, triangle_buffer_resize_required, edge_buffer_resize_required)
    """

    v_index = wp.tid()
    v = pos[v_index]
    vertex_buffer_offset = vertex_colliding_triangles_offsets[v_index]
    vertex_buffer_size = vertex_colliding_triangles_offsets[v_index + 1] - vertex_buffer_offset

    lower = wp.vec3(v[0] - query_radius, v[1] - query_radius, v[2] - query_radius)
    upper = wp.vec3(v[0] + query_radius, v[1] + query_radius, v[2] + query_radius)

    query = wp.bvh_query_aabb(bvh_id, lower, upper)

    tri_index = wp.int32(0)
    vertex_num_collisions = wp.int32(0)
    min_dis_to_tris = query_radius
    while wp.bvh_query_next(query, tri_index):
        t1 = tri_indices[tri_index, 0]
        t2 = tri_indices[tri_index, 1]
        t3 = tri_indices[tri_index, 2]
        if vertex_adjacent_to_triangle(v_index, t1, t2, t3):
            continue

        u1 = pos[t1]
        u2 = pos[t2]
        u3 = pos[t3]

        closest_p, bary, feature_type = triangle_closest_point(u1, u2, u3, v)

        dist = wp.length(closest_p - v)

        if dist < query_radius:
            # record v-f collision to vertex
            min_dis_to_tris = wp.min(min_dis_to_tris, dist)
            if vertex_num_collisions < vertex_buffer_size:
                vertex_colliding_triangles[2 * (vertex_buffer_offset + vertex_num_collisions)] = v_index
                vertex_colliding_triangles[2 * (vertex_buffer_offset + vertex_num_collisions) + 1] = tri_index
            else:
                resize_flags[VERTEX_COLLISION_BUFFER_OVERFLOW_INDEX] = 1

            vertex_num_collisions = vertex_num_collisions + 1

            wp.atomic_min(triangle_colliding_vertices_min_dist, tri_index, dist)

    vertex_colliding_triangles_count[v_index] = vertex_num_collisions
    vertex_colliding_triangles_min_dist[v_index] = min_dis_to_tris


@wp.kernel
def collide_particles_vs_triangles(
    particle_q: wp.array(dtype=wp.vec3f),
    particle_radius: wp.array(dtype=wp.float32),
    particle_inv_mass: wp.array(dtype=wp.float32),
    tri_vertices: wp.array(dtype=wp.vec3f),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    num_triangles: wp.int32,
    # outputs
    particle_q_out: wp.array(dtype=wp.vec3f),
):
    """
    Particle vs triangles collision kernel using Position Based Dynamics.
    
    Launches one thread per particle and iterates through all triangles,
    checking for intersection and applying collision response.
    
    For each particle-triangle pair in collision:
    - Computes closest point on triangle to particle center
    - If distance < particle radius, particle is penetrating
    - Applies position correction to push particle outside the triangle surface
    
    Args:
        particle_q: Current particle positions
        particle_radius: Radius of each particle  
        particle_inv_mass: Inverse mass of each particle (0 = kinematic/static)
        tri_vertices: Vertex positions of the triangle mesh
        tri_indices: Triangle indices (num_triangles x 3), each row contains
                     indices into tri_vertices for the 3 vertices of a triangle
        num_triangles: Number of triangles to test against
        particle_q_out: Output particle positions after collision response
    """
    tid = wp.tid()
    
    inv_mass = particle_inv_mass[tid]
    
    # Skip kinematic particles
    if inv_mass <= 0.0:
        particle_q_out[tid] = particle_q[tid]
        return
    
    pos = particle_q[tid]
    radius = particle_radius[tid]
    
    # Accumulate total correction from all triangle collisions
    total_correction = wp.vec3f(0.0, 0.0, 0.0)
    num_collisions = wp.int32(0)
    
    # Iterate through all triangles
    for tri_idx in range(num_triangles):
        # Get triangle vertex indices
        i0 = tri_indices[tri_idx, 0]
        i1 = tri_indices[tri_idx, 1]
        i2 = tri_indices[tri_idx, 2]
        
        # Get triangle vertex positions
        v0 = tri_vertices[i0]
        v1 = tri_vertices[i1]
        v2 = tri_vertices[i2]
        
        # Find closest point on triangle to particle center
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
                correction_dir = triangle_normal(v0, v1, v2)
                
                # Make sure normal points towards particle (away from triangle)
                # Check if particle was approaching from front or back of triangle
                # by checking which side of the triangle plane the particle is on
                edge1 = v1 - v0
                edge2 = v2 - v0
                tri_norm = wp.cross(edge1, edge2)
                if wp.dot(tri_norm, pos - v0) < 0.0:
                    correction_dir = -correction_dir
            
            # Compute position correction to resolve penetration
            # In PBD, we apply the full correction (assuming stiffness = 1)
            correction = correction_dir * penetration
            
            total_correction = total_correction + correction
            num_collisions = num_collisions + 1
    
    # Average corrections if multiple collisions (Jacobi-style)
    if num_collisions > 0:
        avg_correction = total_correction / wp.float32(num_collisions)
        particle_q_out[tid] = pos + avg_correction
    else:
        particle_q_out[tid] = pos


@wp.kernel
def collide_particles_vs_triangles_accumulate(
    particle_q: wp.array(dtype=wp.vec3f),
    particle_radius: wp.array(dtype=wp.float32),
    particle_inv_mass: wp.array(dtype=wp.float32),
    tri_vertices: wp.array(dtype=wp.vec3f),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    num_triangles: wp.int32,
    # outputs
    delta_accumulator: wp.array(dtype=wp.vec3f),
    delta_counter: wp.array(dtype=wp.int32),
):
    """
    Particle vs triangles collision kernel with accumulation for Jacobi-style iteration.
    
    Similar to collide_particles_vs_triangles but accumulates corrections into
    delta_accumulator and delta_counter arrays for later averaging and application.
    This is useful when combining collision constraints with other PBD constraints.
    
    Args:
        particle_q: Current particle positions
        particle_radius: Radius of each particle
        particle_inv_mass: Inverse mass of each particle (0 = kinematic/static)
        tri_vertices: Vertex positions of the triangle mesh
        tri_indices: Triangle indices (num_triangles x 3)
        num_triangles: Number of triangles to test against
        delta_accumulator: Accumulated position corrections (atomic add)
        delta_counter: Number of corrections accumulated per particle (atomic add)
    """
    tid = wp.tid()
    
    inv_mass = particle_inv_mass[tid]
    
    # Skip kinematic particles
    if inv_mass <= 0.0:
        return
    
    pos = particle_q[tid]
    radius = particle_radius[tid]
    
    # Iterate through all triangles
    for tri_idx in range(num_triangles):
        # Get triangle vertex indices
        i0 = tri_indices[tri_idx, 0]
        i1 = tri_indices[tri_idx, 1]
        i2 = tri_indices[tri_idx, 2]
        
        # Get triangle vertex positions
        v0 = tri_vertices[i0]
        v1 = tri_vertices[i1]
        v2 = tri_vertices[i2]
        
        # Find closest point on triangle to particle center
        closest_p, bary, feature_type = triangle_closest_point(v0, v1, v2, pos)
        
        # Compute distance from particle center to closest point
        to_particle = pos - closest_p
        dist = wp.length(to_particle)
        
        # Check for penetration
        penetration = radius - dist
        
        if penetration > 0.0:
            # Compute correction direction (push particle away from triangle)
            if dist > 1e-8:
                correction_dir = to_particle / dist
            else:
                # Use triangle normal as fallback
                correction_dir = triangle_normal(v0, v1, v2)
                edge1 = v1 - v0
                edge2 = v2 - v0
                tri_norm = wp.cross(edge1, edge2)
                if wp.dot(tri_norm, pos - v0) < 0.0:
                    correction_dir = -correction_dir
            
            # Compute position correction
            correction = correction_dir * penetration
            
            # Accumulate correction atomically
            wp.atomic_add(delta_accumulator, tid, correction)
            wp.atomic_add(delta_counter, tid, 1)