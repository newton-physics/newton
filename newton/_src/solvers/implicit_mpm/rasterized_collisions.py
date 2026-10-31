# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp
import warp.fem as fem
import warp.sparse as wps

import newton

from .contact_solver_kernels import solve_coulomb_isotropic

__all__ = [
    "Collider",
    "build_rigidity_operator",
    "interpolate_collider_normals",
    "project_outside_collider",
    "rasterize_collider",
]

_COLLIDER_ACTIVATION_DISTANCE = wp.constant(0.5)
"""Grid contact activation distance in voxel units, divided by the basis degree."""
_PIC_COLLIDER_ACTIVATION_DISTANCE = wp.constant(0.25)
"""Particle contact activation distance in voxel units."""

_INFINITY = wp.constant(1.0e12)
"""Threshold over which values are considered infinite"""

_CLOSEST_POINT_NORMAL_EPSILON = wp.constant(1.0e-3)
"""Maximum neighborhood radius for closest point normal calculation"""

_SLIVER_NORMAL_RATIO = wp.constant(1.0e-6)
"""Faces whose doubled area is below this fraction of their squared longest edge are slivers."""

_NORMAL_RADIUS_EDGE_FRACTION = wp.constant(0.1)
"""Normal neighborhoods are capped at this fraction of each face's shortest edge."""

_CANCELLED_NORMAL_RATIO = wp.constant(1.0e-4)
"""Relative length below which an averaged normal is cancelled by opposing faces.

Sharp convex features keep much larger ratios, e.g. about 0.07 for an 8 degree wedge.
"""

_SDF_SIGN_FROM_AVERAGE_NORMAL = True
"""If true, determine the sign of the sdf from the average normal of the faces around the closest point.
Otherwise, use Warp's default sign determination strategy (raycasts).
"""

_SMALL_ANGLE_EPS = wp.constant(1.0e-4)
"""Small angle threshold to use more robust and faster path for angular velocity calculations"""

_NULL_COLLIDER_ID = -1
"""Indicator for no collider"""

_ALL_COLLIDER_WORLDS = -2
"""Environment sentinel that queries every collider in stable order."""


@wp.struct
class Collider:
    """Packed collider parameters and geometry queried during rasterization."""

    collider_mesh: wp.array[wp.uint64]
    """Mesh of the collider. Shape (collider_count,)."""

    collider_max_thickness: wp.array[float]
    """Max thickness of each collider mesh. Shape (collider_count,)."""

    collider_body_index: wp.array[int]
    """Body index of each collider. Shape (collider_count,)"""

    collider_particle_offsets: wp.array[int]
    """Offsets into collider_particle_ids for deformable collider mesh vertices. Shape (collider_count + 1,)"""

    collider_particle_ids: wp.array[int]
    """Model particle index for each deformable collider mesh vertex. Shape (sum(deformable mesh vertex counts),)"""

    collider_world: wp.array[int]
    """Newton world ID of each stable collider. Shape (collider_count,)."""

    collider_face_offset: wp.array[int]
    """Start of each stable collider's faces in ``face_material_index``. Shape (collider_count,)."""

    world_collider_ids: wp.array[int]
    """Stable collider IDs affecting each world, grouped by world."""

    world_collider_offsets: wp.array[int]
    """Offsets into ``world_collider_ids`` for each world. Shape (world_count + 1,)."""

    face_material_index: wp.array[int]
    """Collider-local material index for each collider mesh face. Shape (sum(mesh.face_count for mesh in meshes),)"""

    collider_material_offset: wp.array[int]
    """Global id of the first material of each stable collider. Shape (collider_count,)."""

    collider_material_count: wp.array[int]
    """Number of materials of each stable collider. Shape (collider_count,)."""

    material_thickness: wp.array[float]
    """Thickness for each collider material. Shape (material_count,)"""

    material_friction: wp.array[float]
    """Friction coefficient for each collider material. Shape (material_count,)"""

    material_adhesion: wp.array[float]
    """Adhesion coefficient for each collider material (Pa). Shape (material_count,)"""

    material_projection_threshold: wp.array[float]
    """Projection threshold for each collider material. Shape (material_count,)"""

    body_com: wp.array[wp.vec3]
    """Body center of mass of each collider. Shape (body_count,)"""

    query_max_dist: float
    """Maximum distance to query collider sdf"""


@wp.func
def _triangle_pseudonormal(q: wp.vec3, e1: wp.vec3, e2: wp.vec3, eps_sq: float):
    """Return the weighted pseudonormal of triangle (0, e1, e2) at its closest point to ``q``.

    Reuse the projection coordinates for vertex/edge classification. Degenerate
    faces and faces beyond the adaptive neighborhood contribute a zero normal.
    """
    e3 = e2 - e1
    e1_sq = wp.length_sq(e1)
    e2_sq = wp.length_sq(e2)
    e3_sq = wp.length_sq(e3)
    e1e2 = wp.dot(e1, e2)
    normal = wp.cross(e1, e2)
    normal_sq = wp.length_sq(normal)
    edge_scale = wp.max(wp.max(e1_sq, e2_sq), e3_sq)
    if normal_sq <= (_SLIVER_NORMAL_RATIO * edge_scale) * (_SLIVER_NORMAL_RATIO * edge_scale):
        return wp.vec3(0.0)

    # Keep vertex neighborhoods disjoint, even on faces smaller than the search radius.
    eps_sq = wp.min(
        eps_sq, _NORMAL_RADIUS_EDGE_FRACTION * _NORMAL_RADIUS_EDGE_FRACTION * wp.min(wp.min(e1_sq, e2_sq), e3_sq)
    )

    e1q = wp.dot(e1, q)
    e2q = wp.dot(e2, q)
    d3 = e1q - e1_sq
    d4 = e2q - e1e2
    d5 = e1q - e1e2
    d6 = e2q - e2_sq
    interior = False
    s = float(0.0)
    t = float(0.0)
    closest = wp.vec3(0.0)
    if e1q <= 0.0 and e2q <= 0.0:
        closest = wp.vec3(0.0)
    elif d3 >= 0.0 and d4 <= d3:
        closest = e1
    elif d6 >= 0.0 and d5 <= d6:
        closest = e2
    else:
        # Voronoi tests select one edge projection instead of trying all three.
        vc = e1q * d4 - d3 * e2q
        vb = d5 * e2q - e1q * d6
        va = d3 * d6 - d5 * d4
        if vc <= 0.0 and e1q >= 0.0 and d3 <= 0.0:
            closest = (e1q / e1_sq) * e1
        elif vb <= 0.0 and e2q >= 0.0 and d6 <= 0.0:
            closest = (e2q / e2_sq) * e2
        elif va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
            closest = e1 + ((d4 - d3) / (d4 - d3 + d5 - d6)) * e3
        else:
            interior = True
            denom = 1.0 / (va + vb + vc)
            s = vb * denom
            t = vc * denom
            closest = s * e1 + t * e2
    if wp.length_sq(q - closest) >= eps_sq:
        return wp.vec3(0.0)

    normal_length = wp.sqrt(normal_sq)
    # Divide all weights by pi; only vertex weights need angles.
    weight = float(2.0)
    if wp.length_sq(closest) < eps_sq:
        weight = wp.atan2(normal_length, e1e2) / wp.pi
    elif wp.length_sq(closest - e1) < eps_sq:
        weight = wp.atan2(normal_length, e1_sq - e1e2) / wp.pi
    elif wp.length_sq(closest - e2) < eps_sq:
        weight = wp.atan2(normal_length, e2_sq - e1e2) / wp.pi
    else:
        u = 1.0 - s - t
        # Barycentric coordinates give edge distances without reprojecting.
        if (
            not interior
            or t * t * normal_sq < eps_sq * e1_sq
            or s * s * normal_sq < eps_sq * e2_sq
            or u * u * normal_sq < eps_sq * e3_sq
        ):
            weight = 1.0
    return normal * (weight / normal_length)


@wp.func
def get_average_face_normal(
    mesh_id: wp.uint64,
    point: wp.vec3,
):
    """Compute an angle-weighted pseudonormal within an adaptive-distance neighborhood.

    Use corner angles at vertices, pi at edges, and 2*pi in face interiors.
    Face selection and feature classification use the same radius, capped at
    one tenth of each face's shortest edge and the absolute search radius.

    Args:
        mesh_id: The mesh to query.
        point: The point to query.

    Returns:
        The average face normal at the point, or zero when opposing faces cancel
        it to roundoff relative to their weights.
    """

    face_normal = wp.vec3(0.0)
    weight_sq_sum = float(0.0)

    vidx = wp.mesh_get(mesh_id).indices
    points = wp.mesh_get(mesh_id).points
    eps_sq = _CLOSEST_POINT_NORMAL_EPSILON * _CLOSEST_POINT_NORMAL_EPSILON

    epsilon = wp.vec3(_CLOSEST_POINT_NORMAL_EPSILON)
    aabb_query = wp.mesh_query_aabb(mesh_id, point - epsilon, point + epsilon)
    face_index = wp.int32(0)
    while wp.mesh_query_aabb_next(aabb_query, face_index):
        V0 = points[vidx[face_index * 3 + 0]]
        V1 = points[vidx[face_index * 3 + 1]]
        V2 = points[vidx[face_index * 3 + 2]]

        weighted_normal = _triangle_pseudonormal(point - V0, V1 - V0, V2 - V0, eps_sq)
        face_normal += weighted_normal
        weight_sq_sum += wp.length_sq(weighted_normal)

    if wp.length_sq(face_normal) <= _CANCELLED_NORMAL_RATIO * _CANCELLED_NORMAL_RATIO * weight_sq_sum:
        return wp.vec3(0.0)
    return wp.normalize(face_normal)


@wp.func
def _query_collider_sdf(
    x: wp.vec3,
    collider: Collider,
    body_q: wp.array[wp.transform],
    stable_collider_id: int,
):
    mesh = collider.collider_mesh[stable_collider_id]
    thickness = collider.collider_max_thickness[stable_collider_id]
    body_id = collider.collider_body_index[stable_collider_id]

    if body_id >= 0:
        b_pos = wp.transform_get_translation(body_q[body_id])
        b_rot = wp.transform_get_rotation(body_q[body_id])
        x_local = wp.quat_rotate_inv(b_rot, x - b_pos)
    else:
        x_local = x

    max_dist = collider.query_max_dist + thickness

    if wp.static(_SDF_SIGN_FROM_AVERAGE_NORMAL):
        query = wp.mesh_query_point_no_sign(mesh, x_local, max_dist)
    else:
        query = wp.mesh_query_point(mesh, x_local, max_dist)

    query_result = query.result
    sdf = float(_INFINITY)
    sdf_grad = wp.vec3(0.0)
    sdf_vel = wp.vec3(0.0)
    closest_point = wp.vec3(0.0)
    material_id = int(0)  # default material, always valid

    if query_result:
        cp = wp.mesh_eval_position(mesh, query.face, query.u, query.v)

        if wp.static(_SDF_SIGN_FROM_AVERAGE_NORMAL):
            face_normal = get_average_face_normal(mesh, cp)
            if wp.length_sq(face_normal) == 0.0:
                # Edge rounding or cancellation can leave the average normal zero.
                face_normal = wp.mesh_eval_face_normal(mesh, query.face)
            sign = wp.where(wp.dot(face_normal, x_local - cp) > 0.0, 1.0, -1.0)
        else:
            face_normal = wp.mesh_eval_face_normal(mesh, query.face)
            sign = query.sign

        local_material_id = collider.face_material_index[collider.collider_face_offset[stable_collider_id] + query.face]
        local_material_id = wp.clamp(local_material_id, 0, collider.collider_material_count[stable_collider_id] - 1)
        mesh_material_id = collider.collider_material_offset[stable_collider_id] + local_material_id
        thickness = collider.material_thickness[mesh_material_id]

        offset = x_local - cp
        d = wp.length(offset) * sign
        sdf = d - thickness

        if wp.abs(d) < _CLOSEST_POINT_NORMAL_EPSILON:
            sdf_grad = face_normal
        else:
            sdf_grad = wp.normalize(offset) * sign

        sdf_vel = wp.mesh_eval_velocity(mesh, query.face, query.u, query.v)
        closest_point = cp
        material_id = mesh_material_id

    return query_result, sdf, sdf_grad, sdf_vel, closest_point, material_id


@wp.func
def collision_sdf(
    x: wp.vec3,
    environment_index: int,
    collider: Collider,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    dt: float,
):
    min_sdf = float(_INFINITY)
    sdf_grad = wp.vec3(0.0)
    sdf_vel = wp.vec3(0.0)
    closest_point = wp.vec3(0.0)
    collider_id = int(_NULL_COLLIDER_ID)
    material_id = int(0)  # default material, always valid

    shared_worlds = environment_index == _ALL_COLLIDER_WORLDS
    query_begin = int(0)
    query_count = collider.collider_mesh.shape[0]

    if not shared_worlds:
        query_begin = collider.world_collider_offsets[environment_index]
        query_count = collider.world_collider_offsets[environment_index + 1] - query_begin

    for query_offset in range(query_count):
        stable_collider_id = query_offset
        if not shared_worlds:
            stable_collider_id = collider.world_collider_ids[query_begin + query_offset]

        query_result, sdf, query_grad, query_vel, query_closest_point, query_material_id = _query_collider_sdf(
            x, collider, body_q, stable_collider_id
        )
        if query_result and sdf < min_sdf:
            min_sdf = sdf
            sdf_grad = query_grad
            sdf_vel = query_vel
            closest_point = query_closest_point
            collider_id = stable_collider_id
            material_id = query_material_id

    # If closest collider has rigid motion, transform back to world frame
    # Do that as a second step to avoid requiring more registers inside bvh query loop
    if collider_id >= 0:
        body_id = collider.collider_body_index[collider_id]
        if body_id >= 0:
            b_xform = body_q[body_id]
            b_rot = wp.transform_get_rotation(b_xform)

            sdf_vel = wp.quat_rotate(b_rot, sdf_vel)
            sdf_grad = wp.normalize(wp.quat_rotate(b_rot, sdf_grad))

            # Compute rigid body velocity at the contact point
            if body_q_prev:
                # backward-differenced velocity from position change
                b_xform_prev = body_q_prev[body_id]
                closest_point_world = wp.transform_point(b_xform, closest_point)
                closest_point_world_prev = wp.transform_point(b_xform_prev, closest_point)
                sdf_vel += (closest_point_world - closest_point_world_prev) / dt

            if body_qd:
                b_v = wp.spatial_top(body_qd[body_id])
                b_w = wp.spatial_bottom(body_qd[body_id])
                b_com = collider.body_com[body_id]
                com_offset_cur = wp.quat_rotate(b_rot, closest_point - b_com)
                ang_vel = wp.length(b_w)
                angle_delta = ang_vel * dt
                if angle_delta > _SMALL_ANGLE_EPS:
                    # forward-differenced velocity from current velocity
                    # (using exponential map)
                    b_rot_delta = wp.quat_from_axis_angle(b_w / ang_vel, angle_delta)
                    com_offset_next = wp.quat_rotate(b_rot_delta, com_offset_cur)
                    sdf_vel += b_v + (com_offset_next - com_offset_cur) / dt
                else:
                    # Instantaneous rigid body velocity (v + omega x r)
                    sdf_vel += b_v + wp.cross(b_w, com_offset_cur)

    return min_sdf, sdf_grad, sdf_vel, collider_id, material_id


@wp.func
def environment_from_offsets(index: int, offsets: wp.array[int]):
    """Return the environment containing a packed-array index."""
    return wp.lower_bound(offsets, index + 1) - 1


@wp.kernel
def collider_volumes_kernel(
    cell_volume: float,
    collider_ids: wp.array[int],
    node_volumes: wp.array[float],
    volumes: wp.array[float],
):
    i = wp.tid()
    collider_id = collider_ids[i]
    if collider_id >= 0:
        wp.atomic_add(volumes, collider_id, node_volumes[i] * cell_volume)


@wp.func
def collider_is_dynamic(collider_id: int, collider: Collider, body_mass: wp.array[float]):
    if collider_id < 0:
        return False
    body_id = collider.collider_body_index[collider_id]
    if body_id < 0:
        return False
    return body_mass[body_id] > 0.0


@wp.kernel
def project_outside_collider(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    velocity_gradients: wp.array[wp.mat33],
    particle_flags: wp.array[wp.int32],
    particle_mass: wp.array[float],
    particle_environment: wp.array[int],
    collider: Collider,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    dt: float,
    particle_radius: wp.array[float],
    positions_out: wp.array[wp.vec3],
    velocities_out: wp.array[wp.vec3],
    velocity_gradients_out: wp.array[wp.mat33],
):
    """Project particles outside colliders and apply Coulomb response.

    For active particles, queries the nearest collider surface, computes the
    penetration at the end of the step, applies a Coulomb friction response
    against the collider velocity, projects positions outside by the required
    signed distance, and rigidifies the particle velocity gradient. Inactive
    and kinematic (zero-mass) particles are passed through unchanged.

    Args:
        positions: Current particle positions.
        velocities: Current particle velocities.
        velocity_gradients: Current particle velocity gradients.
        particle_flags: Per-particle flags; particles without :attr:`ACTIVE` are skipped.
        particle_mass: Per-particle mass; zero-mass (kinematic) particles are skipped.
        particle_environment: Per-particle world IDs, or null to query every collider.
        collider: Collider description and geometry.
        body_q: Rigid body transforms.
        body_qd: Rigid body velocities.
        body_q_prev: Previous rigid body transforms (for finite-difference velocity).
        dt: Timestep length.
        particle_radius: Per-particle radius kept between particle centers and collider surfaces,
            or null to project particle centers onto the surfaces.
        positions_out: Output particle positions.
        velocities_out: Output particle velocities.
        velocity_gradients_out: Output particle velocity gradients.
    """
    i = wp.tid()

    pos_adv = positions[i]
    p_vel = velocities[i]
    vel_grad = velocity_gradients[i]

    if (~particle_flags[i] & newton.ParticleFlags.ACTIVE) or particle_mass[i] == 0.0:
        positions_out[i] = positions[i]
        velocities_out[i] = p_vel
        velocity_gradients_out[i] = vel_grad
        return

    environment_index = int(_ALL_COLLIDER_WORLDS)
    if particle_environment:
        environment_index = particle_environment[i]

    # project outside of collider
    sdf, sdf_gradient, sdf_vel, _collider_id, material_id = collision_sdf(
        pos_adv, environment_index, collider, body_q, body_qd, body_q_prev, dt
    )
    if particle_radius:
        sdf -= particle_radius[i]

    sdf_end = sdf - wp.dot(sdf_vel, sdf_gradient) * dt + collider.material_projection_threshold[material_id]
    if sdf_end < 0:
        # remove normal vel
        friction = collider.material_friction[material_id]
        delta_vel = solve_coulomb_isotropic(friction, sdf_gradient, p_vel - sdf_vel) + sdf_vel - p_vel

        p_vel += delta_vel
        pos_adv += delta_vel * dt

        # project out
        pos_adv -= wp.min(0.0, sdf_end + dt * wp.dot(delta_vel, sdf_gradient)) * sdf_gradient  # delta_vel * dt

        # make velocity gradient rigid
        vel_grad = 0.5 * (vel_grad - wp.transpose(vel_grad))

    positions_out[i] = pos_adv
    velocities_out[i] = p_vel
    velocity_gradients_out[i] = vel_grad


@wp.kernel
def rasterize_collider_kernel(
    collider: Collider,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    voxel_size: float,
    activation_distance: float,
    speculative: bool,
    stop_distance: float,
    dt: float,
    node_positions: wp.array[wp.vec3],
    node_environment_offsets: wp.array[int],
    node_volumes: wp.array[float],
    node_radii: wp.array[float],
    collider_sdf: wp.array[float],
    collider_velocity: wp.array[wp.vec3],
    collider_normals: wp.array[wp.vec3],
    collider_friction: wp.array[float],
    collider_adhesion: wp.array[float],
    collider_ids: wp.array[int],
):
    """Sample collider data at grid nodes.

    Writes per-node signed distance, contact normal, collider velocity, and
    material parameters (friction and adhesion). Nodes that are too far from
    any collider are marked inactive with a null id and zeroed outputs. The
    adhesion value is scaled by ``dt * voxel_size`` to match the nodal impulse
    units used by the solver.

    Args:
        collider: Collider description and geometry.
        body_q: Rigid body transforms.
        body_qd: Rigid body velocities.
        body_q_prev: Previous rigid body transforms (for finite-difference velocity).
        voxel_size: Grid voxel size [m], used to scale the activation distance.
        activation_distance: Distance (in voxels) below which to activate the collider.
        speculative: Whether to offset the collider velocity so a positive gap can close within the step.
        stop_distance: Distance [m] from the surface where speculative approach stops.
        dt: Timestep length (used to scale adhesion and finite-difference velocity).
        node_positions: Grid node positions to sample at.
        node_environment_offsets: Packed node offsets by world, or null to query every collider.
        node_volumes: Per-node integration volumes.
        node_radii: Per-node radius kept from collider surfaces, such as the radius of the particle
            carrying a point contact, or null for none.
        collider_sdf: Output signed distance per node, less its radius.
        collider_velocity: Output collider velocity per node, with the speculative offset if enabled.
        collider_normals: Output contact normals per node.
        collider_friction: Output friction coefficient per node, or -1 if inactive.
        collider_adhesion: Output scaled adhesion per node.
        collider_ids: Output collider id per node, or null id if inactive.
    """
    i = wp.tid()
    x = node_positions[i]

    if x[0] == fem.OUTSIDE:
        bc_active = False
        sdf = _INFINITY
    else:
        environment_index = int(_ALL_COLLIDER_WORLDS)
        if node_environment_offsets:
            environment_index = environment_from_offsets(i, node_environment_offsets)
        sdf, sdf_gradient, sdf_vel, collider_id, material_id = collision_sdf(
            x, environment_index, collider, body_q, body_qd, body_q_prev, dt
        )
        if node_radii:
            sdf -= node_radii[i]
        bc_active = sdf < activation_distance * voxel_size

    collider_sdf[i] = sdf

    if not bc_active:
        collider_velocity[i] = wp.vec3(0.0)
        collider_normals[i] = wp.vec3(0.0)
        collider_friction[i] = -1.0
        collider_adhesion[i] = 0.0
        collider_ids[i] = _NULL_COLLIDER_ID
        return

    collider_ids[i] = collider_id
    collider_normals[i] = sdf_gradient

    collider_friction[i] = collider.material_friction[material_id]
    # Speculative contacts adhere once within their stopping distance, so adhesion pulls no material across the gap.
    adhesion = 0.0
    if not speculative or sdf <= stop_distance:
        adhesion = collider.material_adhesion[material_id] * dt * node_volumes[i] / voxel_size
    collider_adhesion[i] = adhesion

    # Allow approach until the end-of-step gap reaches the stopping distance, in the activation band of
    # grid contacts, so particles near a wall keep sampling the volume around them instead of bunching
    # against it; nodes within that distance keep the collider velocity.
    if speculative and dt > 0.0:
        sdf_vel -= (wp.max(sdf - stop_distance, 0.0) / dt) * sdf_gradient
    collider_velocity[i] = sdf_vel


@wp.kernel
def point_contact_radii(
    collider_space_node: wp.array[int],
    evaluation_particle: wp.array[int],
    particle_radius: wp.array[float],
    node_radii: wp.array[float],
):
    """Radius of the particle carrying each point contact node."""
    n = wp.tid()
    space_node = collider_space_node[n]
    node_radii[n] = wp.where(space_node >= 0, particle_radius[evaluation_particle[wp.max(space_node, 0)]], 0.0)


@wp.func
def quadratic_bspline(x: float):
    x = wp.abs(x)
    if x < 0.5:
        return 0.75 - x * x
    if x < 1.5:
        return 0.5 * (1.5 - x) * (1.5 - x)
    return 0.0


@wp.func
def quadratic_bspline_derivative(x: float):
    a = wp.abs(x)
    if a < 0.5:
        return -2.0 * x
    if a < 1.5:
        return -wp.sign(x) * (1.5 - a)
    return 0.0


def make_cell_locator(grid):
    """Functions locating a position's cell and its neighbor cells, on dense or sparse grids."""

    @fem.cache.dynamic_func(suffix=grid.name)
    def containing_cell(geo: grid.CellArg, x: wp.vec3, environment: int, h: float):
        """Lattice coordinates and center of the cell containing ``x``, and whether it exists."""
        if wp.static(isinstance(grid, fem.Nanogrid)):
            uvw = wp.volume_world_to_index(geo.cell_grid, x) + wp.vec3(0.5)
            local_cell = wp.vec3i(int(wp.floor(uvw[0])), int(wp.floor(uvw[1])), int(wp.floor(uvw[2])))
            return local_cell, wp.volume_index_to_world(geo.cell_grid, wp.vec3(local_cell)), True
        sample = grid.cell_lookup(geo, x, 0.0, environment)
        if sample.element_index < 0:
            return wp.vec3i(0), wp.vec3(0.0), False
        sample.element_coords = wp.vec3(0.5)
        return wp.vec3i(0), grid.cell_position(geo, sample), True

    @fem.cache.dynamic_func(suffix=grid.name)
    def neighbor_cell(
        geo: grid.CellArg, local_cell: wp.vec3i, center: wp.vec3, offset: wp.vec3i, environment: int, h: float
    ):
        """Index of the cell ``offset`` away from the containing cell, or -1 outside the grid."""
        if wp.static(isinstance(grid, fem.Nanogrid)):
            ijk = local_cell + offset + geo.env_offsets[environment]
            cell = wp.volume_lookup_index(geo.cell_grid, ijk[0], ijk[1], ijk[2])
            if cell >= 0:
                if geo.cell_env[cell] != environment:
                    cell = -1
            return cell
        target_center = center + h * wp.vec3(float(offset[0]), float(offset[1]), float(offset[2]))
        target = grid.cell_lookup(geo, target_center, 0.0, environment)
        if target.element_index < 0:
            return -1
        # Grid3D lookups clamp at their boundary: reject clamped cells.
        actual = grid.cell_position(geo, target)
        if wp.length_sq(actual - target_center) >= 1.0e-8 * h * h:
            return -1
        return target.element_index

    return containing_cell, neighbor_cell


def make_cell_distance_kernel(grid):
    """Collider signed distance at every grid cell center, clamped to ``max_distance``."""

    @fem.cache.dynamic_kernel(suffix=grid.name)
    def cell_distance(
        geo: grid.CellArg,
        collider: Collider,
        body_q: wp.array[wp.transform],
        body_qd: wp.array[wp.spatial_vector],
        body_q_prev: wp.array[wp.transform],
        separate_worlds: bool,
        max_distance: float,
        output: wp.array[float],
    ):
        cell = wp.tid()
        if wp.static(isinstance(grid, fem.Nanogrid)):
            if cell >= wp.volume_voxel_count(geo.cell_grid):
                return
        sample = fem.make_free_sample(cell, wp.vec3(0.5))
        center = grid.cell_position(geo, sample)
        environment = grid.cell_environment_index(geo, sample)
        collider_environment = wp.where(separate_worlds, environment, int(_ALL_COLLIDER_WORLDS))
        sdf, _gradient, _velocity, _collider_id, _material_id = collision_sdf(
            center, collider_environment, collider, body_q, body_qd, body_q_prev, 1.0
        )
        output[cell] = wp.min(sdf, max_distance)

    return cell_distance


def make_distance_gradient_kernel(grid):
    """Gradient at each contact node of the collider distance rasterized at cell centers.

    The cell-center distances act as quadratic B-spline coefficients, whose gradient is
    continuous across cells. Cells outside the grid evaluate their distance on the fly.
    """
    containing_cell, neighbor_cell = make_cell_locator(grid)

    @fem.cache.dynamic_kernel(suffix=grid.name)
    def distance_gradient(
        geo: grid.CellArg,
        node_positions: wp.array[wp.vec3],
        node_environment_offsets: wp.array[int],
        cell_distances: wp.array[float],
        collider: Collider,
        body_q: wp.array[wp.transform],
        body_qd: wp.array[wp.spatial_vector],
        body_q_prev: wp.array[wp.transform],
        h: float,
        max_distance: float,
        gradients: wp.array[wp.vec3],
    ):
        i = wp.tid()
        gradients[i] = wp.vec3(0.0)
        x = node_positions[i]
        if x[0] == fem.OUTSIDE:
            return
        environment = int(0)
        collider_environment = int(_ALL_COLLIDER_WORLDS)
        if node_environment_offsets:
            environment = environment_from_offsets(i, node_environment_offsets)
            collider_environment = environment
        local_cell, containing_center, found = containing_cell(geo, x, environment, h)
        if not found:
            return
        t = (x - containing_center) / h
        gradient = wp.vec3(0.0)
        for k in range(27):
            offset = wp.vec3i(k // 9 - 1, (k // 3) % 3 - 1, k % 3 - 1)
            q = t - wp.vec3(float(offset[0]), float(offset[1]), float(offset[2]))
            bx = quadratic_bspline(q[0])
            by = quadratic_bspline(q[1])
            bz = quadratic_bspline(q[2])
            if bx * by * bz == 0.0 and (
                quadratic_bspline_derivative(q[0]) == 0.0
                or quadratic_bspline_derivative(q[1]) == 0.0
                or quadratic_bspline_derivative(q[2]) == 0.0
            ):
                continue
            cell = neighbor_cell(geo, local_cell, containing_center, offset, environment, h)
            distance = float(0.0)
            if cell >= 0 and cell < cell_distances.shape[0]:
                distance = cell_distances[cell]
            else:
                center = containing_center + h * wp.vec3(float(offset[0]), float(offset[1]), float(offset[2]))
                sdf, _g, _v, _cid, _mid = collision_sdf(
                    center, collider_environment, collider, body_q, body_qd, body_q_prev, 1.0
                )
                distance = wp.min(sdf, max_distance)
            weight_gradient = wp.vec3(
                quadratic_bspline_derivative(q[0]) * by * bz,
                bx * quadratic_bspline_derivative(q[1]) * bz,
                bx * by * quadratic_bspline_derivative(q[2]),
            )
            gradient += distance * weight_gradient / h
        gradients[i] = gradient

    return distance_gradient


def collider_distance_gradients(
    collider: Collider,
    body_q: wp.array[wp.transform],
    grid: fem.Geometry,
    voxel_size: float,
    node_positions: wp.array[wp.vec3],
    node_environment_offsets: wp.array | None,
    separate_worlds: bool,
    temporary_store: fem.TemporaryStore,
):
    """Gradient of the collider signed distance rasterized at cell centers, at each contact node.

    Distances are exact within four voxels of colliders, which covers the quadratic B-spline
    support around any active contact, and clamped farther away.
    """
    device = node_positions.device
    max_distance = 4.0 * voxel_size
    cell_distances = fem.borrow_temporary(temporary_store, shape=grid.cell_count(), dtype=float, device=device)
    gradients = fem.borrow_temporary(temporary_store, shape=node_positions.shape[0], dtype=wp.vec3, device=device)
    old_query_distance = collider.query_max_dist
    collider.query_max_dist = max(old_query_distance, max_distance)
    try:
        wp.launch(
            make_cell_distance_kernel(grid),
            dim=grid.cell_count(),
            inputs=[
                grid.cell_arg_value(device),
                collider,
                body_q,
                None,
                None,
                node_environment_offsets is not None,
                max_distance,
            ],
            outputs=[cell_distances],
            device=device,
        )
        wp.launch(
            make_distance_gradient_kernel(grid),
            dim=node_positions.shape[0],
            inputs=[
                grid.cell_arg_value(device),
                node_positions,
                node_environment_offsets,
                cell_distances,
                collider,
                body_q,
                None,
                None,
                voxel_size,
                max_distance,
            ],
            outputs=[gradients],
            device=device,
        )
    finally:
        collider.query_max_dist = old_query_distance
    cell_distances.release()
    return gradients


@wp.func
def collider_is_deformable(collider_id: int, collider: Collider):
    if collider_id < 0:
        return False
    return collider.collider_particle_offsets[collider_id + 1] > collider.collider_particle_offsets[collider_id]


@wp.kernel
def stabilize_collider_velocity(
    dt: float,
    fraction: float,
    close_gap: bool,
    collider_sdf: wp.array[float],
    collider_normals: wp.array[wp.vec3],
    collider_velocity: wp.array[wp.vec3],
):
    """Correct penetration and, with ``close_gap``, allow separated contacts to close their gap.

    Speculative point contacts already close their gap during rasterization.
    """
    i = wp.tid()
    normal = collider_normals[i]
    if wp.length_sq(normal) == 0.0:
        return
    gap = collider_sdf[i]
    target = float(0.0)
    if gap < 0.0:
        target = -fraction * gap / dt
    elif close_gap:
        target = -gap / dt
    collider_velocity[i] += target * normal


@wp.kernel
def fill_collider_coupling_matrices(
    node_positions: wp.array[wp.vec3],
    collider: Collider,
    body_q: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    particle_mass: wp.array[float],
    cell_volume: float,
    particle_block_offset: int,
    collider_ids: wp.array[int],
    J_rows: wp.array[int],
    J_cols: wp.array[int],
    J_values: wp.array[wp.mat33],
    IJtm_values: wp.array[wp.mat33],
):
    i = wp.tid()
    slot = 3 * i

    J_rows[slot + 0] = -1
    J_rows[slot + 1] = -1
    J_rows[slot + 2] = -1
    J_cols[slot + 0] = -1
    J_cols[slot + 1] = -1
    J_cols[slot + 2] = -1

    collider_id = collider_ids[i]

    if collider_is_dynamic(collider_id, collider, body_mass):
        body_id = collider.collider_body_index[collider_id]

        J_rows[slot + 0] = i
        J_rows[slot + 1] = i
        J_cols[slot + 0] = 2 * body_id
        J_cols[slot + 1] = 2 * body_id + 1

        b_pos = wp.transform_get_translation(body_q[body_id])
        b_rot = wp.transform_get_rotation(body_q[body_id])
        R = wp.quat_to_matrix(b_rot)

        x = node_positions[i]
        W = wp.skew(b_pos + R * collider.body_com[body_id] - x)

        Id = wp.identity(n=3, dtype=float)
        J_values[slot + 0] = W
        J_values[slot + 1] = Id

        # Grid impulses need to be scaled by cell_volume

        world_inv_inertia = R @ body_inv_inertia[body_id] @ wp.transpose(R)
        IJtm_values[slot + 0] = -cell_volume * world_inv_inertia @ W
        IJtm_values[slot + 1] = (cell_volume / body_mass[body_id]) * Id

    elif collider_is_deformable(collider_id, collider):
        mesh = collider.collider_mesh[collider_id]
        x = node_positions[i]
        max_dist = collider.query_max_dist + collider.collider_max_thickness[collider_id]
        query = wp.mesh_query_point_no_sign(mesh, x, max_dist)
        if not query.result:
            return

        indices = wp.mesh_get(mesh).indices
        face = query.face
        local_i = indices[3 * face + 0]
        local_j = indices[3 * face + 1]
        local_k = indices[3 * face + 2]

        offset = collider.collider_particle_offsets[collider_id]
        flat_i = offset + local_i
        flat_j = offset + local_j
        flat_k = offset + local_k
        particle_i = collider.collider_particle_ids[flat_i]
        particle_j = collider.collider_particle_ids[flat_j]
        particle_k = collider.collider_particle_ids[flat_k]

        w_i = query.u
        w_j = query.v
        w_k = 1.0 - w_i - w_j

        Id = wp.identity(n=3, dtype=float)

        m_i = particle_mass[particle_i]
        if m_i > 0.0 and w_i > 0.0:
            J_rows[slot + 0] = i
            J_cols[slot + 0] = particle_block_offset + flat_i
            J_values[slot + 0] = w_i * Id
            IJtm_values[slot + 0] = (cell_volume * w_i / m_i) * Id

        m_j = particle_mass[particle_j]
        if m_j > 0.0 and w_j > 0.0:
            J_rows[slot + 1] = i
            J_cols[slot + 1] = particle_block_offset + flat_j
            J_values[slot + 1] = w_j * Id
            IJtm_values[slot + 1] = (cell_volume * w_j / m_j) * Id

        m_k = particle_mass[particle_k]
        if m_k > 0.0 and w_k > 0.0:
            J_rows[slot + 2] = i
            J_cols[slot + 2] = particle_block_offset + flat_k
            J_values[slot + 2] = w_k * Id
            IJtm_values[slot + 2] = (cell_volume * w_k / m_k) * Id


@fem.integrand
def world_position(
    s: fem.Sample,
    domain: fem.Domain,
):
    return domain(s)


@fem.integrand
def collider_gradient_field(s: fem.Sample, domain: fem.Domain, distance: fem.Field, normal: fem.Field):
    min_sdf = float(_INFINITY)
    min_pos = wp.vec3(0.0)
    min_grad = wp.vec3(0.0)

    # min sdf over all nodes in the element
    elem_count = fem.node_count(distance, s)
    for k in range(elem_count):
        s_node = fem.at_node(distance, s, k)
        sdf = distance(s_node, k)
        if sdf < min_sdf:
            min_sdf = sdf
            min_pos = domain(s_node)
            min_grad = normal(s_node, k)

    if min_sdf == _INFINITY:
        return wp.vec3(0.0)

    # compute gradient, filtering invalid values
    sdf_gradient = wp.vec3(0.0)
    for k in range(elem_count):
        s_node = fem.at_node(distance, s, k)
        sdf = distance(s_node, k)
        pos = domain(s_node)

        # if the sdf value is not acceptable (larger than min_sdf + distance between nodes),
        # replace with linearized approximation
        if sdf >= min_sdf + wp.length(pos - min_pos):
            sdf = wp.min(sdf, min_sdf + wp.dot(min_grad, pos - min_pos))

        sdf_gradient += sdf * fem.node_inner_weight_gradient(distance, s, k)

    return sdf_gradient


@wp.kernel
def normalize_gradient(
    gradient: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    sdf: wp.array[float],
    velocity: wp.array[wp.vec3],
    dt: float,
    stop_distance: float,
):
    i = wp.tid()
    previous_normal = normal[i]
    # Fully clamped distance stencils, far from colliders, leave no usable gradient
    if wp.length_sq(gradient[i]) < 1.0e-6:
        return
    corrected_normal = wp.normalize(gradient[i])
    if wp.length_sq(previous_normal) > 0.0 and dt > 0.0:
        velocity[i] += (wp.max(sdf[i] - stop_distance, 0.0) / dt) * (previous_normal - corrected_normal)
    normal[i] = corrected_normal


def speculative_stop_distance(voxel_size: float) -> float:
    """Distance [m] from collider surfaces where speculative particle contacts stop approaching.

    It matches the particle contact activation band, where grid contacts activate too.
    """
    return float(_PIC_COLLIDER_ACTIVATION_DISTANCE) * voxel_size


def _is_speculative(field: fem.DiscreteField) -> bool:
    """Point (particle) contacts close positive gaps within the step; grid contacts keep their activation band."""
    return field.degree == 0


def rasterize_collider(
    collider: Collider,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    voxel_size: float,
    dt: float,
    collider_space_restriction: fem.SpaceRestriction,
    collider_node_volume: wp.array[float],
    collider_position_field: fem.DiscreteField,
    collider_distance_field: fem.DiscreteField,
    collider_normal_field: fem.DiscreteField,
    collider_velocity: wp.array[wp.vec3],
    collider_friction: wp.array[float],
    collider_adhesion: wp.array[float],
    collider_ids: wp.array[int],
    temporary_store: fem.TemporaryStore,
    node_environment_offsets: wp.array | None = None,
    *,
    contact_gap: float = 0.0,
    node_radii: wp.array | None = None,
):
    """Rasterize collider signed-distance, normals, velocity, and material onto grid nodes.

    For each collision node, queries the nearest collider surface and writes the
    signed distance, outward normal, collider velocity, friction, adhesion, and
    collider id to the corresponding output arrays.

    Args:
        collider: Packed collider parameters and geometry.
        body_q: Rigid body transforms.
        body_qd: Rigid body velocities (spatial vectors).
        body_q_prev: Previous rigid body transforms (for finite-difference velocity).
        voxel_size: Grid voxel edge length [m].
        dt: Timestep length [s].
        collider_space_restriction: Space restriction for collision nodes.
        collider_node_volume: Output per-node volume fractions.
        collider_position_field: Output world-space node positions.
        collider_distance_field: Output signed-distance values per node.
        collider_normal_field: Output outward normals per node.
        collider_velocity: Output collider velocity [m/s], with a speculative normal offset for point contacts.
        collider_friction: Output Coulomb friction coefficient per node.
        collider_adhesion: Output adhesion per node [Pa].
        collider_ids: Output collider index per node, or ``_NULL_COLLIDER_ID``.
        temporary_store: Temporary storage for intermediate buffers.
        node_environment_offsets: Packed collision-node offsets by world. If ``None``, every node
            queries every collider in stable order.
        contact_gap: Minimum contact activation distance [m].
        node_radii: Per-node radius subtracted from the signed distance, or ``None`` for none.
            Point contacts pass the radius of their particle, so particles rest on their surface.
    """
    collision_node_count = collider_position_field.dof_values.shape[0]

    collider_position_field.dof_values.fill_(wp.vec3(fem.OUTSIDE))
    fem.interpolate(
        world_position,
        dest=collider_position_field,
        at=collider_space_restriction,
        reduction="first",
        temporary_store=temporary_store,
    )

    activation_distance = (
        _PIC_COLLIDER_ACTIVATION_DISTANCE
        if collider_position_field.degree == 0
        else _COLLIDER_ACTIVATION_DISTANCE / collider_position_field.degree
    )
    activation_distance = max(activation_distance, contact_gap / voxel_size)

    wp.launch(
        rasterize_collider_kernel,
        dim=collision_node_count,
        inputs=[
            collider,
            body_q,
            body_qd,
            body_q_prev,
            voxel_size,
            activation_distance,
            _is_speculative(collider_position_field),
            speculative_stop_distance(voxel_size),
            dt,
            collider_position_field.dof_values,
            node_environment_offsets,
            collider_node_volume,
            node_radii,
            collider_distance_field.dof_values,
            collider_velocity,
            collider_normal_field.dof_values,
            collider_friction,
            collider_adhesion,
            collider_ids,
        ],
    )


def interpolate_collider_normals(
    collider_space_restriction: fem.SpaceRestriction,
    collider_distance_field: fem.DiscreteField,
    collider_normal_field: fem.DiscreteField,
    temporary_store: fem.TemporaryStore,
    *,
    collider_velocity: wp.array[wp.vec3],
    dt: float,
    distance_gradients: wp.array[wp.vec3] | None = None,
    stop_distance: float = 0.0,
):
    """Smooth collider normals by computing the gradient of the distance field.

    Interpolates the gradient of ``collider_distance_field`` at each collision
    node and normalizes the result to produce smoothed outward normals, which
    are written back into ``collider_normal_field``. Point (particle) collision
    nodes carry no distance gradient, so they take ``distance_gradients``, from
    :func:`collider_distance_gradients`, instead.

    Args:
        collider_space_restriction: Space restriction for collision nodes.
        collider_distance_field: Per-node signed-distance field.
        collider_normal_field: Per-node normal field (updated in place).
        temporary_store: Temporary storage for intermediate buffers.
        collider_velocity: Collider constraint velocities, updated for the corrected normals [m/s].
        dt: Timestep length [s].
        distance_gradients: Per-node distance gradients replacing those of
            ``collider_distance_field``, or ``None``.
        stop_distance: Distance [m] where speculative contacts stop approaching, as in
            :func:`rasterize_collider`.
    """
    if distance_gradients is not None:
        corrected_normal = distance_gradients
    else:
        corrected_normal = wp.empty_like(collider_normal_field.dof_values)
        fem.interpolate(
            collider_gradient_field,
            dest=corrected_normal,
            dest_space=collider_normal_field.space,
            at=collider_space_restriction,
            fields={"distance": collider_distance_field, "normal": collider_normal_field},
            reduction="mean",
            temporary_store=temporary_store,
        )

    wp.launch(
        normalize_gradient,
        dim=collider_normal_field.dof_values.shape,
        inputs=[
            corrected_normal,
            collider_normal_field.dof_values,
            collider_distance_field.dof_values,
            collider_velocity,
            dt if _is_speculative(collider_normal_field) else 0.0,
            stop_distance,
        ],
    )


def build_rigidity_operator(
    cell_volume: float,
    node_volumes: wp.array[float],
    node_positions: wp.array[wp.vec3],
    collider: Collider,
    body_q: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    particle_mass: wp.array[float],
    collider_ids: wp.array[int],
) -> tuple[wps.BsrMatrix, wps.BsrMatrix]:
    """Build the collider operator that couples collider impulses to finite-mass DOFs.

    Builds block-sparse maps between active collider nodes and finite-mass
    collider DOFs. Rigid-body DOFs occupy the first ``2 * body_count`` block
    columns; deformable collider vertices use a compact block range after that
    rather than their global model particle ids.

    Internally constructs:
      - J: kinematic Jacobian blocks per node relating collider DOF velocity to nodal velocity.
      - IJtm: mass- and inertia-scaled transpose mapping.

    The returned operator is (J @ IJtm) and corresponds the collider Delassus operator.

    Args:
        cell_volume: Grid cell volume as scaling factor to node_volumes.
        node_volumes: Per-velocity-node volume fractions.
        node_positions: World-space node positions (3D).
        collider: Packed collider parameters and geometry handles.
        body_q: Rigid body transforms.
        body_mass: Rigid body masses.
        body_inv_inertia: Rigid body inverse inertia tensors.
        particle_mass: Particle masses for deformable mesh colliders.
        collider_ids: Per-velocity-node collider id, or `_NULL_COLLIDER_ID` when not active.

    Returns:
        A tuple of ``warp.sparse.BsrMatrix`` (J, IJtm) representing the rigidity coupling operator ``J @ IJtm``
    """

    vel_node_count = node_volumes.shape[0]
    body_count = body_q.shape[0]
    deformable_vertex_count = collider.collider_particle_ids.shape[0]
    particle_block_offset = 2 * body_count

    max_blocks_per_node = 3
    J_rows = wp.empty(vel_node_count * max_blocks_per_node, dtype=int)
    J_cols = wp.empty(vel_node_count * max_blocks_per_node, dtype=int)
    J_values = wp.empty(vel_node_count * max_blocks_per_node, dtype=wp.mat33)
    IJtm_values = wp.empty(vel_node_count * max_blocks_per_node, dtype=wp.mat33)

    wp.launch(
        fill_collider_coupling_matrices,
        dim=vel_node_count,
        inputs=[
            node_positions,
            collider,
            body_q,
            body_mass,
            body_inv_inertia,
            particle_mass,
            cell_volume,
            particle_block_offset,
            collider_ids,
            J_rows,
            J_cols,
            J_values,
            IJtm_values,
        ],
    )

    J = wps.bsr_from_triplets(
        rows_of_blocks=vel_node_count,
        cols_of_blocks=particle_block_offset + deformable_vertex_count,
        rows=J_rows,
        columns=J_cols,
        values=J_values,
    )

    IJtm = wps.bsr_from_triplets(
        cols_of_blocks=vel_node_count,
        rows_of_blocks=particle_block_offset + deformable_vertex_count,
        columns=J_rows,
        rows=J_cols,
        values=IJtm_values,
    )

    return J, IJtm
