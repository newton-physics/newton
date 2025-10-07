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
    "build_rigidity_matrix",
    "project_outside_collider",
    "rasterize_collider",
]

_COLLIDER_EXTRAPOLATION_DISTANCE = wp.constant(0.25)
"""Distance to extrapolate collider sdf, as a fraction of the voxel size"""

_INFINITY = wp.constant(1.0e12)
"""Mass over which colliders are considered kinematic"""

_NULL_COLLIDER_ID = -2
_GROUND_COLLIDER_ID = -1
_GROUND_FRICTION = 1.0
_GROUND_ADHESION = 0.0
_GROUND_PROJECTION_THRESHOLD = 0.5


@wp.struct
class Collider:
    """Packed collider parameters and geometry queried during rasterization."""

    meshes: wp.array(dtype=wp.uint64)
    """Meshes of the collider"""

    thicknesses: wp.array(dtype=float)
    """Thickness of each collider mesh"""

    projection_threshold: wp.array(dtype=float)
    """Projection threshold for each collider"""

    friction: wp.array(dtype=float)
    """Friction coefficient for each collider"""

    adhesion: wp.array(dtype=float)
    """Adhesion coefficient for each collider (Pa)"""

    masses: wp.array(dtype=float)
    """Mass of each collider"""

    query_max_dist: float
    """Maximum distance to query collider sdf"""

    ground_height: float
    """Y-coordinate of the ground"""

    ground_normal: wp.vec3
    """Normal of the ground"""


@wp.func
def collision_sdf(x: wp.vec3, collider: Collider):
    ground_sdf = wp.dot(x, collider.ground_normal) - collider.ground_height

    min_sdf = ground_sdf
    sdf_grad = collider.ground_normal
    sdf_vel = wp.vec3(0.0)
    collider_id = int(_GROUND_COLLIDER_ID)

    # Find closest collider
    for m in range(collider.meshes.shape[0]):
        mesh = collider.meshes[m]

        thickness = collider.thicknesses[m]
        max_dist = collider.query_max_dist + thickness
        query = wp.mesh_query_point_sign_normal(mesh, x, max_dist)

        if query.result:
            cp = wp.mesh_eval_position(mesh, query.face, query.u, query.v)

            offset = x - cp
            d = wp.length(offset) * query.sign
            sdf = d - thickness

            if sdf < min_sdf:
                min_sdf = sdf
                if wp.abs(d) < 0.0001:
                    sdf_grad = wp.mesh_eval_face_normal(mesh, query.face)
                else:
                    sdf_grad = wp.normalize(offset) * query.sign

                sdf_vel = wp.mesh_eval_velocity(mesh, query.face, query.u, query.v)
                collider_id = m

    return min_sdf, sdf_grad, sdf_vel, collider_id


@wp.func
def collision_is_active(sdf: float, voxel_size: float):
    return sdf < _COLLIDER_EXTRAPOLATION_DISTANCE * voxel_size


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
def collider_friction_coefficient(collider_id: int, collider: Collider):
    if collider_id == _GROUND_COLLIDER_ID:
        return _GROUND_FRICTION
    return collider.friction[collider_id]


@wp.func
def collider_adhesion_coefficient(collider_id: int, collider: Collider):
    if collider_id == _GROUND_COLLIDER_ID:
        return _GROUND_ADHESION
    return collider.adhesion[collider_id]


@wp.func
def collider_density(collider_id: int, collider: Collider, collider_volumes: wp.array(dtype=float)):
    if collider_id == _GROUND_COLLIDER_ID:
        return _INFINITY
    return collider.masses[collider_id] / collider_volumes[collider_id]


@wp.func
def collider_projection_threshold(collider_id: int, collider: Collider):
    if collider_id == _GROUND_COLLIDER_ID:
        return _GROUND_PROJECTION_THRESHOLD
    return collider.projection_threshold[collider_id]


@wp.func
def collider_is_dynamic(collider_id: int, collider: Collider):
    if collider_id < 0:
        return False
    return collider.masses[collider_id] < _INFINITY


@wp.kernel
def project_outside_collider(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    velocity_gradients: wp.array(dtype=wp.mat33),
    particle_flags: wp.array(dtype=wp.int32),
    collider: Collider,
    voxel_size: float,
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
        voxel_size: Grid voxel edge length (used for thresholds/scales).
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
    sdf, sdf_gradient, sdf_vel, collider_id = collision_sdf(pos_adv, collider)

    sdf_end = (
        sdf - wp.dot(sdf_vel, sdf_gradient) * dt + collider_projection_threshold(collider_id, collider) * voxel_size
    )
    if sdf_end < 0:
        # remove normal vel
        friction = collider_friction_coefficient(collider_id, collider)
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
def rasterize_collider(
    collider: Collider,
    voxel_size: float,
    dt: float,
    node_positions: wp.array(dtype=wp.vec3),
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
        voxel_size: Grid voxel edge length (sets query/extrapolation band).
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
        sdf = fem.OUTSIDE
    else:
        sdf, sdf_gradient, sdf_vel, collider_id = collision_sdf(x, collider)
        bc_active = collision_is_active(sdf, voxel_size)

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
    collider_friction[i] = collider_friction_coefficient(collider_id, collider)
    collider_adhesion[i] = collider_adhesion_coefficient(collider_id, collider) * dt * voxel_size
    collider_velocity[i] = sdf_vel


@wp.kernel
def collider_inverse_mass(
    collider: Collider,
    collider_ids: wp.array(dtype=int),
    node_volume: wp.array(dtype=float),
    collider_volumes: wp.array(dtype=float),
    collider_inv_mass_matrix: wp.array(dtype=float),
):
    i = wp.tid()
    collider_id = collider_ids[i]

    bc_vol = node_volume[i]
    if collider_is_dynamic(collider_id, collider) and bc_vol > 0.0:
        bc_density = collider_density(collider_id, collider, collider_volumes)
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
    cell_volume: float,
    collider_ids: wp.array(dtype=int),
    collider_coms: wp.array(dtype=wp.vec3),
    collider_inv_inertia: wp.array(dtype=wp.mat33),
    J_rows: wp.array(dtype=int),
    J_cols: wp.array(dtype=int),
    J_values: wp.array(dtype=wp.mat33),
    IJtm_values: wp.array(dtype=wp.mat33),
    non_rigid_diagonal: wp.array(dtype=wp.mat33),
):
    i = wp.tid()
    x = node_positions[i]

    collider_id = collider_ids[i]
    bc_active = collider_id != _NULL_COLLIDER_ID

    if bc_active and collider_is_dynamic(collider_id, collider):
        J_rows[2 * i] = i
        J_rows[2 * i + 1] = i
        J_cols[2 * i] = 2 * collider_id
        J_cols[2 * i + 1] = 2 * collider_id + 1

        W = wp.skew(collider_coms[collider_id] - x)
        Id = wp.identity(n=3, dtype=float)
        J_values[2 * i] = W
        J_values[2 * i + 1] = Id

        bc_mass = node_volumes[i] * cell_volume * collider_density(collider_id, collider, collider_volumes)

        IJtm_values[2 * i] = -bc_mass * collider_inv_inertia[collider_id] * W
        IJtm_values[2 * i + 1] = bc_mass / collider.masses[collider_id] * Id

        non_rigid_diagonal[i] = -Id

    else:
        J_cols[2 * i] = -1
        J_cols[2 * i + 1] = -1
        J_rows[2 * i] = -1
        J_rows[2 * i + 1] = -1

        non_rigid_diagonal[i] = wp.mat33(0.0)


def allot_collider_mass(
    cell_volume: float,
    node_volumes: wp.array(dtype=float),
    collider: Collider,
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
        cell_volume: Grid cell volume as scalaing factor to node_volumes.
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
            collider_ids,
            node_volumes,
            collider_total_volumes,
            collider_inv_mass_matrix,
        ],
    )


def build_rigidity_matrix(
    cell_volume: float,
    node_volumes: wp.array(dtype=float),
    node_positions: wp.array(dtype=wp.vec3),
    collider: Collider,
    collider_ids: wp.array(dtype=int),
    collider_coms: wp.array(dtype=wp.vec3),
    collider_inv_inertia: wp.array(dtype=wp.mat33),
    collider_total_volumes: wp.array(dtype=float),
):
    """Assemble the collider rigidity matrix that couples node motion to rigid DOFs.

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
        cell_volume: Grid cell volume as scalaing factor to node_volumes.
        node_volumes: Per-velocity-node volume fractions.
        node_positions: World-space node positions (3D).
        collider: Packed collider parameters and geometry handles.
        collider_ids: Per-velocity-node collider id, or -2 when not active.
        collider_coms: Per-collider centers of mass in world space.
        collider_inv_inertia: Per-collider inverse inertia tensors in world space.
        collider_total_volumes: Per-collider integrated volumes used to derive densities.

    Returns:
        A ``warp.sparse.BsrMatrix`` representing the rigidity coupling.
    """

    vel_node_count = node_volumes.shape[0]
    collider_count = collider.meshes.shape[0]

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
            cell_volume,
            collider_ids,
            collider_coms,
            collider_inv_inertia,
            J_rows,
            J_cols,
            J_values,
            IJtm_values,
            Iphi_diag,
        ],
    )

    J = wps.bsr_from_triplets(
        rows_of_blocks=vel_node_count,
        cols_of_blocks=2 * collider_count,
        rows=J_rows,
        columns=J_cols,
        values=J_values,
    )

    IJtm = wps.bsr_from_triplets(
        cols_of_blocks=vel_node_count,
        rows_of_blocks=2 * collider_count,
        columns=J_rows,
        rows=J_cols,
        values=IJtm_values,
    )

    rigid = wps.bsr_diag(Iphi_diag)
    wps.bsr_mm(x=J, y=IJtm, z=rigid, alpha=1.0, beta=1.0, max_new_nnz=10 * rigid.nnz)

    return rigid
