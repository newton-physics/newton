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
import warp.fem as fem
import warp.sparse as wps

import newton

from .solve_rheology import solve_coulomb_isotropic

__all__ = [
    "Collider",
    "allot_collider_mass",
    "build_rigidity_operator",
    "interpolate_collider_normals",
    "project_outside_collider",
    "rasterize_collider",
]

_COLLIDER_ACTIVATION_DISTANCE = wp.constant(0.5)
"""Distance below which to activate the collider"""

_INFINITY = wp.constant(1.0e12)
"""Threshold over which values are considered infinite"""

_CLOSEST_POINT_NORMAL_EPSILON = wp.constant(1.0e-3)
"""Epsilon for closest point normal calculation"""

_SDF_SIGN_FROM_AVERAGE_NORMAL = True
"""If true, determine the sign of the sdf from the average normal of the faces around the closest point.
Otherwise, use Warp's default sign determination strategy (raycasts).
"""


_NULL_COLLIDER_ID = -2
_GROUND_COLLIDER_ID = -1
_GROUND_COLLIDER_MATERIAL_INDEX = 0


@wp.struct
class Collider:
    """Packed collider parameters and geometry queried during rasterization."""

    collider_mesh: wp.array(dtype=wp.uint64)
    """Mesh of the collider. Shape (collider_count,)."""

    collider_max_thickness: wp.array(dtype=float)
    """Max thickness of each collider mesh. Shape (collider_count,)."""

    collider_body_index: wp.array(dtype=int)
    """Body index of each collider. Shape (collider_count,)"""

    face_material_index: wp.array(dtype=int)
    """Material index for each collider mesh face. Shape (sum(mesh.face_count for mesh in meshes),)"""

    material_thickness: wp.array(dtype=float)
    """Thickness for each collider material. Shape (material_count,)"""

    material_friction: wp.array(dtype=float)
    """Friction coefficient for each collider material. Shape (material_count,)"""

    material_adhesion: wp.array(dtype=float)
    """Adhesion coefficient for each collider material (Pa). Shape (material_count,)"""

    material_projection_threshold: wp.array(dtype=float)
    """Projection threshold for each collider material. Shape (material_count,)"""

    body_com: wp.array(dtype=wp.vec3)
    """Body center of mass of each collider. Shape (body_count,)"""

    query_max_dist: float
    """Maximum distance to query collider sdf"""

    ground_height: float
    """Y-coordinate of the ground"""

    ground_normal: wp.vec3
    """Normal of the ground"""


@wp.func
def collision_sdf(
    x: wp.vec3, collider: Collider, body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector)
):
    ground_sdf = (
        wp.dot(x, collider.ground_normal)
        - collider.ground_height
        - collider.material_thickness[_GROUND_COLLIDER_MATERIAL_INDEX]
    )

    min_sdf = ground_sdf
    sdf_grad = collider.ground_normal
    sdf_vel = wp.vec3(0.0)
    closest_point = wp.vec3(0.0)
    collider_id = int(_GROUND_COLLIDER_ID)
    material_id = int(_GROUND_COLLIDER_MATERIAL_INDEX)

    # Find closest collider
    global_face_id = int(0)
    for m in range(collider.collider_mesh.shape[0]):
        mesh = collider.collider_mesh[m]
        thickness = collider.collider_max_thickness[m]
        body_id = collider.collider_body_index[m]

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

        if query.result:
            cp = wp.mesh_eval_position(mesh, query.face, query.u, query.v)

            if wp.static(_SDF_SIGN_FROM_AVERAGE_NORMAL):
                face_normal = wp.vec3(0.0)
                epsilon = wp.vec3(_CLOSEST_POINT_NORMAL_EPSILON)
                aabb_query = wp.mesh_query_aabb(mesh, cp - epsilon, cp + epsilon)
                face_index = wp.int32(0)
                while wp.mesh_query_aabb_next(aabb_query, face_index):
                    face_normal += wp.mesh_eval_face_normal(mesh, face_index)

                face_normal = wp.normalize(face_normal)
                sign = wp.where(wp.dot(face_normal, x_local - cp) > 0.0, 1.0, -1.0)
            else:
                face_normal = wp.mesh_eval_face_normal(mesh, query.face)
                sign = query.sign

            mesh_material_id = collider.face_material_index[global_face_id + query.face]
            thickness = collider.material_thickness[mesh_material_id]

            offset = x_local - cp
            d = wp.length(offset) * sign
            sdf = d - thickness

            if sdf < min_sdf:
                min_sdf = sdf
                if wp.abs(d) < _CLOSEST_POINT_NORMAL_EPSILON:
                    sdf_grad = face_normal
                else:
                    sdf_grad = wp.normalize(offset) * sign

                sdf_vel = wp.mesh_eval_velocity(mesh, query.face, query.u, query.v)
                closest_point = cp
                collider_id = m
                material_id = mesh_material_id

        global_face_id += wp.mesh_get(mesh).indices.shape[0] // 3

    # If closest collider has rigid motion, transform back to world frame
    # Do that as a second step to avoid requiring more registers inside bvh query loop
    if collider_id >= 0:
        body_id = collider.collider_body_index[collider_id]
        if body_id >= 0:
            b_pos = wp.transform_get_translation(body_q[body_id])
            b_rot = wp.transform_get_rotation(body_q[body_id])

            sdf_vel = wp.quat_rotate(b_rot, sdf_vel)
            sdf_grad = wp.normalize(wp.quat_rotate(b_rot, sdf_grad))

            # Assumes linear/angular velocities in world frame
            b_v = wp.spatial_top(body_qd[body_id])
            b_w = wp.spatial_bottom(body_qd[body_id])
            sdf_vel = b_v + wp.cross(b_w, wp.quat_rotate(b_rot, closest_point - collider.body_com[body_id]))

    return min_sdf, sdf_grad, sdf_vel, collider_id, material_id


@wp.kernel
def collider_volumes_kernel(
    cell_volume: float,
    collider_ids: wp.array(dtype=int),
    node_volumes: wp.array(dtype=float),
    volumes: wp.array(dtype=float),
):
    i = wp.tid()
    collider_id = collider_ids[i]
    if collider_id >= 0:
        wp.atomic_add(volumes, collider_id, node_volumes[i] * cell_volume)


@wp.func
def collider_density(
    collider_id: int, collider: Collider, collider_volumes: wp.array(dtype=float), body_mass: wp.array(dtype=float)
):
    if collider_id == _GROUND_COLLIDER_ID:
        return _INFINITY
    body_id = collider.collider_body_index[collider_id]
    if body_id < 0:
        return _INFINITY
    return body_mass[body_id] / collider_volumes[collider_id]


@wp.func
def collider_is_dynamic(collider_id: int, collider: Collider, body_mass: wp.array(dtype=float)):
    if collider_id < 0:
        return False
    body_id = collider.collider_body_index[collider_id]
    if body_id < 0:
        return False
    return body_mass[body_id] > 0.0


@wp.kernel
def project_outside_collider(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    velocity_gradients: wp.array(dtype=wp.mat33),
    particle_flags: wp.array(dtype=wp.int32),
    collider: Collider,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    dt: float,
    positions_out: wp.array(dtype=wp.vec3),
    velocities_out: wp.array(dtype=wp.vec3),
    velocity_gradients_out: wp.array(dtype=wp.mat33),
):
    """Project particles outside colliders and apply Coulomb response.

    For active particles, queries the nearest collider surface, computes the
    penetration at the end of the step, applies a Coulomb friction response
    against the collider velocity, projects positions outside by the required
    signed distance, and rigidifies the particle velocity gradient. Inactive
    particles are passed through unchanged.

    Args:
        positions: Current particle positions.
        velocities: Current particle velocities.
        velocity_gradients: Current particle velocity gradients.
        particle_flags: Per-particle flags (used to gate inactive particles).
        collider: Collider description and geometry.
        body_q: Rigid body transforms.
        body_qd: Rigid body velocities.
        dt: Timestep length.
        positions_out: Output particle positions.
        velocities_out: Output particle velocities.
        velocity_gradients_out: Output particle velocity gradients.
    """
    i = wp.tid()

    pos_adv = positions[i]
    p_vel = velocities[i]
    vel_grad = velocity_gradients[i]

    if ~particle_flags[i] & newton.ParticleFlags.ACTIVE:
        positions_out[i] = positions[i]
        velocities_out[i] = p_vel
        velocity_gradients_out[i] = vel_grad
        return

    # project outside of collider
    sdf, sdf_gradient, sdf_vel, _collider_id, material_id = collision_sdf(pos_adv, collider, body_q, body_qd)

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
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    voxel_size: float,
    activation_distance: float,
    dt: float,
    node_positions: wp.array(dtype=wp.vec3),
    node_volumes: wp.array(dtype=float),
    collider_sdf: wp.array(dtype=float),
    collider_velocity: wp.array(dtype=wp.vec3),
    collider_normals: wp.array(dtype=wp.vec3),
    collider_friction: wp.array(dtype=float),
    collider_adhesion: wp.array(dtype=float),
    collider_ids: wp.array(dtype=int),
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
        activation_distance: Distance below which to activate the collider.
        dt: Timestep length (used to scale adhesion).
        node_positions: Grid node positions to sample at.
        collider_sdf: Output signed distance per node.
        collider_velocity: Output collider velocity per node.
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
        sdf, sdf_gradient, sdf_vel, collider_id, material_id = collision_sdf(x, collider, body_q, body_qd)
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
    collider_adhesion[i] = collider.material_adhesion[material_id] * dt * node_volumes[i] / voxel_size

    collider_velocity[i] = sdf_vel


@wp.kernel
def collider_inverse_mass(
    collider: Collider,
    body_mass: wp.array(dtype=float),
    collider_ids: wp.array(dtype=int),
    node_volume: wp.array(dtype=float),
    collider_volumes: wp.array(dtype=float),
    collider_inv_mass_matrix: wp.array(dtype=float),
):
    i = wp.tid()
    collider_id = collider_ids[i]

    bc_vol = node_volume[i]
    if collider_is_dynamic(collider_id, collider, body_mass) and bc_vol > 0.0:
        bc_density = collider_density(collider_id, collider, collider_volumes, body_mass)
        bc_mass = bc_vol * bc_density
        collider_inv_mass_matrix[i] = 1.0 / bc_mass
    else:
        collider_inv_mass_matrix[i] = 0.0


@wp.kernel
def fill_collider_rigidity_matrices(
    node_positions: wp.array(dtype=wp.vec3),
    collider_volumes: wp.array(dtype=float),
    node_volumes: wp.array(dtype=float),
    collider: Collider,
    body_q: wp.array(dtype=wp.transform),
    body_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    cell_volume: float,
    collider_ids: wp.array(dtype=int),
    J_rows: wp.array(dtype=int),
    J_cols: wp.array(dtype=int),
    J_values: wp.array(dtype=wp.mat33),
    IJtm_values: wp.array(dtype=wp.mat33),
    non_rigid_diagonal: wp.array(dtype=wp.mat33),
):
    i = wp.tid()

    collider_id = collider_ids[i]

    if collider_is_dynamic(collider_id, collider, body_mass):
        body_id = collider.collider_body_index[collider_id]

        J_rows[2 * i] = i
        J_rows[2 * i + 1] = i
        J_cols[2 * i] = 2 * body_id
        J_cols[2 * i + 1] = 2 * body_id + 1

        b_pos = wp.transform_get_translation(body_q[body_id])
        b_rot = wp.transform_get_rotation(body_q[body_id])
        R = wp.quat_to_matrix(b_rot)

        x = node_positions[i]
        W = wp.skew(b_pos + R * collider.body_com[body_id] - x)

        Id = wp.identity(n=3, dtype=float)
        J_values[2 * i] = W
        J_values[2 * i + 1] = Id

        bc_mass = node_volumes[i] * cell_volume * collider_density(collider_id, collider, collider_volumes, body_mass)

        world_inv_inertia = R @ body_inv_inertia[body_id] @ wp.transpose(R)
        IJtm_values[2 * i] = -bc_mass * world_inv_inertia @ W
        IJtm_values[2 * i + 1] = bc_mass / body_mass[body_id] * Id

        non_rigid_diagonal[i] = -Id

    else:
        J_cols[2 * i] = -1
        J_cols[2 * i + 1] = -1
        J_rows[2 * i] = -1
        J_rows[2 * i + 1] = -1

        non_rigid_diagonal[i] = wp.mat33(0.0)


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
    gradient: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    normal[i] = wp.normalize(gradient[i])


def rasterize_collider(
    collider: Collider,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    voxel_size: float,
    dt: float,
    collider_space_restriction: fem.SpaceRestriction,
    collider_node_volume: wp.array(dtype=float),
    collider_position_field: fem.DiscreteField,
    collider_distance_field: fem.DiscreteField,
    collider_normal_field: fem.DiscreteField,
    collider_velocity: wp.array(dtype=wp.vec3),
    collider_friction: wp.array(dtype=float),
    collider_adhesion: wp.array(dtype=float),
    collider_ids: wp.array(dtype=int),
):
    collision_node_count = collider_position_field.dof_values.shape[0]

    collider_position_field.dof_values.fill_(wp.vec3(fem.OUTSIDE))
    fem.interpolate(world_position, dest=collider_position_field, at=collider_space_restriction, reduction="first")

    activation_distance = (
        0.0 if collider_position_field.degree == 0 else _COLLIDER_ACTIVATION_DISTANCE / collider_position_field.degree
    )

    wp.launch(
        rasterize_collider_kernel,
        dim=collision_node_count,
        inputs=[
            collider,
            body_q,
            body_qd,
            voxel_size,
            activation_distance,
            dt,
            collider_position_field.dof_values,
            collider_node_volume,
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
):
    # collider_distance_field.dof_values = corrected_distance
    corrected_normal = wp.empty_like(collider_normal_field.dof_values)
    fem.interpolate(
        collider_gradient_field,
        dest=corrected_normal,
        dest_space=collider_normal_field.space,
        at=collider_space_restriction,
        fields={"distance": collider_distance_field, "normal": collider_normal_field},
        reduction="mean",
    )

    wp.launch(
        normalize_gradient,
        dim=collider_normal_field.dof_values.shape,
        inputs=[corrected_normal, collider_normal_field.dof_values],
    )


def allot_collider_mass(
    cell_volume: float,
    node_volumes: wp.array(dtype=float),
    collider: Collider,
    body_mass: wp.array(dtype=float),
    collider_ids: wp.array(dtype=int),
    collider_total_volumes: wp.array(dtype=float),
    collider_inv_mass_matrix: wp.array(dtype=float),
):
    """Accumulate collider mass onto grid nodes and compute inverse masses.

    This function first integrates the per-mesh volume contribution of all
    active collider regions. It then computes per-node inverse masses for
    nodes tagged by ``collider_ids`` as being within the collider influence.

    - Dynamic colliders (finite mass) contribute a non-zero inverse mass that
      is proportional to the local node volume and the collider density
      (mass/total volume per collider mesh).
    - Kinematic colliders (infinite mass) and nodes not influenced by a
      collider receive an inverse mass of zero.

    Args:
        cell_volume: Grid cell volume as scaling factor to node_volumes.
        node_volumes: Per-velocity-node volume fractions (in voxel units).
        collider: Packed collider parameters and geometry handles.
        collider_ids: Per-velocity-node collider id, or `_NULL_COLLIDER_ID` when not active.
        collider_total_volumes: Output per-collider total volumes (accumulated).
        collider_inv_mass_matrix: Output per-node inverse masses due to collider compliance.
    """

    vel_node_count = node_volumes.shape[0]

    collider_total_volumes.zero_()
    wp.launch(
        collider_volumes_kernel,
        dim=vel_node_count,
        inputs=[
            cell_volume,
            collider_ids,
            node_volumes,
            collider_total_volumes,
        ],
    )

    wp.launch(
        collider_inverse_mass,
        dim=vel_node_count,
        inputs=[
            collider,
            body_mass,
            collider_ids,
            node_volumes,
            collider_total_volumes,
            collider_inv_mass_matrix,
        ],
    )


def build_rigidity_operator(
    cell_volume: float,
    node_volumes: wp.array(dtype=float),
    node_positions: wp.array(dtype=wp.vec3),
    collider: Collider,
    body_q: wp.array(dtype=wp.transform),
    body_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    collider_ids: wp.array(dtype=int),
    collider_total_volumes: wp.array(dtype=float),
) -> tuple[wps.BsrMatrix, wps.BsrMatrix, wps.BsrMatrix]:
    """Build the collider rigidity operator that couples node motion to rigid DOFs.

    Builds a block-sparse matrix of size (3 N_vel_nodes) x (3 N_vel_nodes) that
    maps nodal velocity corrections to rigid-body displacements. Only nodes
    with a valid collider id and only dynamic colliders (finite mass) produce
    non-zero blocks.

    Internally constructs:
      - J: kinematic Jacobian blocks per node relating rigid velocity to nodal velocity.
      - IJtm: mass- and inertia-scaled transpose mapping.
      - Iphi_diag: diagonal term that enforces non-rigid (complementary) DOFs.

    The returned matrix is J @ IJtm + diag(Iphi_diag). It is later used to
    propagate rigid coupling when solving collider friction.

    Args:
        cell_volume: Grid cell volume as scaling factor to node_volumes.
        node_volumes: Per-velocity-node volume fractions.
        node_positions: World-space node positions (3D).
        collider: Packed collider parameters and geometry handles.
        body_q: Rigid body transforms.
        body_mass: Rigid body masses.
        body_inv_inertia: Rigid body inverse inertia tensors.
        collider_ids: Per-velocity-node collider id, or -2 when not active.
        collider_total_volumes: Per-collider integrated volumes used to derive densities.

    Returns:
        A tuple of ``warp.sparse.BsrMatrix`` (D, J, IJtm) representing the rigidity coupling operator ``D + J @ IJtm``
    """

    vel_node_count = node_volumes.shape[0]
    body_count = body_q.shape[0]

    J_rows = wp.empty(vel_node_count * 2, dtype=int)
    J_cols = wp.empty(vel_node_count * 2, dtype=int)
    J_values = wp.empty(vel_node_count * 2, dtype=wp.mat33)
    IJtm_values = wp.empty(vel_node_count * 2, dtype=wp.mat33)
    Iphi_diag = wp.empty(vel_node_count, dtype=wp.mat33)

    wp.launch(
        fill_collider_rigidity_matrices,
        dim=vel_node_count,
        inputs=[
            node_positions,
            collider_total_volumes,
            node_volumes,
            collider,
            body_q,
            body_mass,
            body_inv_inertia,
            cell_volume,
            collider_ids,
            J_rows,
            J_cols,
            J_values,
            IJtm_values,
            Iphi_diag,
        ],
    )

    J = wps.bsr_from_triplets(
        rows_of_blocks=vel_node_count,
        cols_of_blocks=2 * body_count,
        rows=J_rows,
        columns=J_cols,
        values=J_values,
    )

    IJtm = wps.bsr_from_triplets(
        cols_of_blocks=vel_node_count,
        rows_of_blocks=2 * body_count,
        columns=J_rows,
        rows=J_cols,
        values=IJtm_values,
    )

    D = wps.bsr_diag(Iphi_diag)
    return D, J, IJtm
