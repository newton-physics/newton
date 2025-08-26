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

"""Implicit MPM solver."""

from dataclasses import dataclass

import numpy as np
import warp as wp
import warp.fem as fem
import warp.sparse as sp
from warp.context import assert_conditional_graph_support

from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase
from .solve_rheology import solve_coulomb_isotropic, solve_rheology

__all__ = ["SolverImplicitMPM"]

vec6 = wp.types.vector(length=6, dtype=wp.float32)
mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
mat63 = wp.types.matrix(shape=(6, 3), dtype=wp.float32)
mat36 = wp.types.matrix(shape=(3, 6), dtype=wp.float32)

VOLUME_CUTOFF = wp.constant(1.0e-4)
"""Volume fraction under which cells are ignored"""

COLLIDER_EXTRAPOLATION_DISTANCE = wp.constant(0.25)
"""Distance to extrapolate collider sdf, as a fraction of the voxel size"""

DEFAULT_PROJECTION_THRESHOLD = wp.constant(0.5)
"""Default threshold for projection outside of collider, as a fraction of the voxel size"""

_DEFAULT_THICKNESS = 0.5

INFINITE_MASS = wp.constant(1.0e12)
"""Mass over which colliders are considered kinematic"""

_NULL_COLLIDER_ID = -2
_GROUND_COLLIDER_ID = -1
_GROUND_FRICTION = 1.0
_DEFAULT_FRICTION = 0.0

_SMALL_STRAINS = True
"""Small-strain approximation -- extension to finite-strain is still WIP"""


@wp.struct
class Collider:
    """Collider data passed to kernels and integrands."""

    meshes: wp.array(dtype=wp.uint64)
    """Meshes of the collider"""

    thicknesses: wp.array(dtype=float)
    """Thickness of each collider mesh"""

    projection_threshold: wp.array(dtype=float)
    """Projection threshold for each collider"""

    friction: wp.array(dtype=float)
    """Friction coefficient for each collider"""

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

        query = wp.mesh_query_point_sign_normal(mesh, x, collider.query_max_dist)

        if query.result:
            cp = wp.mesh_eval_position(mesh, query.face, query.u, query.v)

            offset = x - cp
            d = wp.length(offset) * query.sign
            sdf = d - collider.thicknesses[m]

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
    return sdf < COLLIDER_EXTRAPOLATION_DISTANCE * voxel_size


@fem.integrand
def collider_volumes(
    s: fem.Sample,
    domain: fem.Domain,
    collider: Collider,
    voxel_size: float,
    volumes: wp.array(dtype=float),
):
    x = domain(s)

    sdf, sdf_gradient, sdf_vel, collider_id = collision_sdf(x, collider)
    bc_active = collision_is_active(sdf, voxel_size)

    if bc_active and collider_id >= 0:
        wp.atomic_add(volumes, collider_id, fem.measure(domain, s) * s.qp_weight)


@wp.func
def collider_friction_coefficient(collider_id: int, collider: Collider):
    if collider_id == _GROUND_COLLIDER_ID:
        return _GROUND_FRICTION
    return collider.friction[collider_id]


@wp.func
def collider_density(collider_id: int, collider: Collider, collider_volumes: wp.array(dtype=float)):
    if collider_id == _GROUND_COLLIDER_ID:
        return INFINITE_MASS
    return collider.masses[collider_id] / collider_volumes[collider_id]


@wp.func
def collider_projection_threshold(collider_id: int, collider: Collider):
    if collider_id == _GROUND_COLLIDER_ID:
        return DEFAULT_PROJECTION_THRESHOLD
    return collider.projection_threshold[collider_id]


@wp.func
def collider_is_dynamic(collider_id: int, collider: Collider):
    if collider_id == _GROUND_COLLIDER_ID:
        return False
    return collider.masses[collider_id] < INFINITE_MASS


@fem.integrand
def integrate_fraction(s: fem.Sample, phi: fem.Field, domain: fem.Domain, inv_cell_volume: float):
    return phi(s) * inv_cell_volume


@fem.integrand
def integrate_collider_fraction(
    s: fem.Sample,
    domain: fem.Domain,
    phi: fem.Field,
    sdf: fem.Field,
    inv_cell_volume: float,
):
    return phi(s) * wp.where(sdf(s) <= 0.0, inv_cell_volume, 0.0)


@fem.integrand
def integrate_velocity(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    velocities: wp.array(dtype=wp.vec3),
    velocity_gradients: wp.array(dtype=wp.mat33),
    dt: float,
    gravity: wp.vec3,
    inv_cell_volume: float,
):
    # APIC velocity prediction
    node_offset = domain(fem.at_node(u, s)) - domain(s)
    vel_apic = velocities[s.qp_index] + velocity_gradients[s.qp_index] * node_offset
    vel_adv = vel_apic + dt * gravity

    return wp.dot(u(s), vel_adv) * inv_cell_volume


@fem.integrand
def advect_particles(
    s: fem.Sample,
    grid_vel: fem.Field,
    dt: float,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    vel_grad: wp.array(dtype=wp.mat33),
):
    # Advect particles and project if necessary

    p_vel = grid_vel(s)
    p_vel_grad = fem.grad(grid_vel, s)

    pos_adv = pos_prev[s.qp_index] + dt * p_vel

    pos[s.qp_index] = pos_adv
    vel[s.qp_index] = p_vel
    vel_grad[s.qp_index] = p_vel_grad


@fem.integrand
def update_particle_elastic_strain(
    s: fem.Sample,
    grid_vel: fem.Field,
    grid_strain: fem.Field,
    grid_strain_delta: fem.Field,
    dt: float,
    flip: float,
    elastic_strain_prev: wp.array(dtype=wp.mat33),
    elastic_strain: wp.array(dtype=wp.mat33),
):
    # elastic strain

    p_vel_grad = fem.grad(grid_vel, s)

    strain = grid_strain(s)
    strain_delta = grid_strain_delta(s)
    strain = strain * (1.0 - flip) + flip * (strain_delta + elastic_strain_prev[s.qp_index])

    if wp.static(_SMALL_STRAINS):
        # Jaumann convected derivative
        skew = 0.5 * (p_vel_grad - wp.transpose(p_vel_grad))
        strain += dt * (skew * strain - strain * skew)
        strain = 0.5 * (strain + wp.transpose(strain))

    elastic_strain[s.qp_index] = strain


@wp.kernel
def update_particle_frames(
    dt: float,
    min_stretch: float,
    max_stretch: float,
    vel_grad: wp.array(dtype=wp.mat33),
    transform_prev: wp.array(dtype=wp.mat33),
    transform: wp.array(dtype=wp.mat33),
):
    i = wp.tid()

    p_vel_grad = vel_grad[i]

    # transform, for grain-level rendering
    F_prev = transform_prev[i]
    # dX1/dx = dX1/dX0 dX0/dx
    F = F_prev + dt * p_vel_grad @ F_prev

    # clamp eigenvalues of F
    if min_stretch >= 0.0 and max_stretch >= 0.0:
        U = wp.mat33()
        S = wp.vec3()
        V = wp.mat33()
        wp.svd3(F, U, S, V)
        S = wp.max(wp.min(S, wp.vec3(max_stretch)), wp.vec3(min_stretch))
        F = U @ wp.diag(S) @ wp.transpose(V)

    transform[i] = F


@wp.kernel
def project_outside_collider(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    velocity_gradients: wp.array(dtype=wp.mat33),
    collider: Collider,
    voxel_size: float,
    dt: float,
    positions_out: wp.array(dtype=wp.vec3),
    velocities_out: wp.array(dtype=wp.vec3),
    velocity_gradients_out: wp.array(dtype=wp.mat33),
):
    i = wp.tid()
    pos_adv = positions[i]
    p_vel = velocities[i]

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

        positions_out[i] = pos_adv
        velocities_out[i] = p_vel

        # make velocity gradient rigid
        vel_grad = velocity_gradients[i]
        velocity_gradients_out[i] = vel_grad - wp.transpose(vel_grad)


@wp.kernel
def rasterize_collider(
    collider: Collider,
    voxel_size: float,
    node_positions: wp.array(dtype=wp.vec3),
    collider_sdf: wp.array(dtype=float),
    collider_velocity: wp.array(dtype=wp.vec3),
    collider_normals: wp.array(dtype=wp.vec3),
    collider_friction: wp.array(dtype=float),
    collider_ids: wp.array(dtype=int),
    collider_impulse: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    x = node_positions[i]
    sdf, sdf_gradient, sdf_vel, collider_id = collision_sdf(x, collider)
    bc_active = collision_is_active(sdf, voxel_size)

    collider_sdf[i] = sdf

    if not bc_active:
        collider_velocity[i] = wp.vec3(0.0)
        collider_normals[i] = wp.vec3(0.0)
        collider_friction[i] = -1.0
        collider_ids[i] = _NULL_COLLIDER_ID
        collider_impulse[i] = wp.vec3(0.0)
        return

    collider_ids[i] = collider_id
    collider_normals[i] = sdf_gradient
    collider_friction[i] = collider_friction_coefficient(collider_id, collider)
    collider_velocity[i] = sdf_vel


@wp.kernel
def collider_inverse_mass(
    particle_density: float,
    collider: Collider,
    collider_ids: wp.array(dtype=int),
    node_volume: wp.array(dtype=float),
    collider_volumes: wp.array(dtype=float),
    collider_inv_mass_matrix: wp.array(dtype=float),
):
    i = wp.tid()
    collider_id = collider_ids[i]

    if collider_is_dynamic(collider_id, collider):
        bc_vol = node_volume[i]
        bc_density = collider_density(collider_id, collider, collider_volumes)
        bc_mass = bc_vol * bc_density
        collider_inv_mass_matrix[i] = particle_density / bc_mass
    else:
        collider_inv_mass_matrix[i] = 0.0


@fem.integrand
def free_velocity(
    s: fem.Sample,
    velocity_int: wp.array(dtype=wp.vec3),
    particle_volume: wp.array(dtype=float),
    inv_mass_matrix: wp.array(dtype=float),
):
    pvol = particle_volume[s.qp_index]
    inv_particle_volume = 1.0 / wp.max(pvol, VOLUME_CUTOFF)

    vel = velocity_int[s.qp_index] * inv_particle_volume
    inv_mass_matrix[s.qp_index] = inv_particle_volume

    return vel


@fem.integrand
def small_strain_form(
    s: fem.Sample,
    u: fem.Field,
    tau: fem.Field,
    dt: float,
    domain: fem.Domain,
    inv_cell_volume: float,
    elastic_strain: wp.array(dtype=vec6),
    rotation: wp.array(dtype=wp.quatf),
):
    return wp.ddot(fem.D(u, s), tau(s)) * (dt * inv_cell_volume)


@fem.integrand
def finite_strain_form(
    s: fem.Sample,
    u: fem.Field,
    tau: fem.Field,
    elastic_strain: wp.array(dtype=vec6),
    rotation: wp.array(dtype=wp.quatf),
    dt: float,
    domain: fem.Domain,
    inv_cell_volume: float,
):
    # get polar decomposition at strain node
    tau_index = fem.operator.node_index(tau, s)
    R = wp.quat_to_matrix(rotation[tau_index])
    dS = wp.transpose(R) @ fem.grad(u, s) @ R

    S = fem.SymmetricTensorMapper.dof_to_value_3d(elastic_strain[tau_index])
    dS += dS @ S

    return wp.ddot(dS, tau(s)) * (dt * inv_cell_volume)


@fem.integrand
def integrate_elastic_strain(
    s: fem.Sample,
    elastic_strains: wp.array(dtype=wp.mat33),
    tau: fem.Field,
    domain: fem.Domain,
    inv_cell_volume: float,
):
    return wp.ddot(elastic_strains[s.qp_index], tau(s)) * inv_cell_volume


@wp.kernel
def add_unilateral_strain_offset(
    max_fraction: float,
    particle_volume: wp.array(dtype=float),
    collider_volume: wp.array(dtype=float),
    node_volume: wp.array(dtype=float),
    prev_symmetric_strain: wp.array(dtype=vec6),
    int_symmetric_strain: wp.array(dtype=vec6),
):
    i = wp.tid()

    spherical_part = max_fraction * (node_volume[i] - collider_volume[i]) - particle_volume[i]
    spherical_part = wp.max(spherical_part, 0.0)

    strain_offset = spherical_part / 3.0 * wp.identity(n=3, dtype=float)
    if wp.static(not _SMALL_STRAINS):
        # multiply by (I + S)
        strain_offset += strain_offset * fem.SymmetricTensorMapper.dof_to_value_3d(prev_symmetric_strain[i])

    int_symmetric_strain[i] += fem.SymmetricTensorMapper.value_to_dof_3d(strain_offset)


@wp.func
def polar_decomposition(F: wp.mat33):
    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    wp.svd3(F, U, sig, V)

    Vt = wp.transpose(V)
    R = U * Vt
    S = V * wp.diag(sig) * Vt

    return R, S


@wp.kernel
def scale_strain_stress_matrices(
    stress_strain_mat: mat66,
    particle_volume: wp.array(dtype=float),
    scaled_mat: wp.array(dtype=mat66),
):
    i = wp.tid()

    p_vol = wp.max(particle_volume[i], VOLUME_CUTOFF)
    scaled_mat[i] = stress_strain_mat * p_vol


@wp.kernel
def elastic_strain_decomposition(
    int_elastic_strain: wp.array(dtype=wp.mat33),
    particle_volume: wp.array(dtype=float),
    strain_rotation: wp.array(dtype=wp.quatf),
    int_symmetric_strain: wp.array(dtype=vec6),
    symmetric_strain: wp.array(dtype=vec6),
):
    i = wp.tid()

    Id = wp.identity(n=3, dtype=float)
    V = particle_volume[i]
    dF = int_elastic_strain[i]

    F = dF / wp.max(V, VOLUME_CUTOFF) + Id
    R, S = polar_decomposition(F)

    strain_rotation[i] = wp.quat_from_matrix(R)

    int_symmetric_strain[i] = fem.SymmetricTensorMapper.value_to_dof_3d(wp.transpose(R) @ (dF + V * Id) - V * Id)
    if wp.static(_SMALL_STRAINS):
        symmetric_strain[i] = vec6(0.0)
    else:
        symmetric_strain[i] = fem.SymmetricTensorMapper.value_to_dof_3d(S - Id)


@wp.kernel
def compute_elastic_strain_delta(
    particle_volume: wp.array(dtype=float),
    strain_rotation: wp.array(dtype=wp.quatf),
    sym_elastic_strain: wp.array(dtype=vec6),
    full_strain: wp.array(dtype=wp.mat33),
    elastic_strain: wp.array(dtype=wp.mat33),
    elastic_strain_delta: wp.array(dtype=wp.mat33),
):
    i = wp.tid()

    V_inv = 1.0 / wp.max(particle_volume[i], VOLUME_CUTOFF)
    dFe_prev = elastic_strain[i] * V_inv
    dSe = fem.SymmetricTensorMapper.dof_to_value_3d(sym_elastic_strain[i]) * V_inv

    Id = wp.identity(n=3, dtype=float)
    R = wp.quat_to_matrix(strain_rotation[i])

    if wp.static(_SMALL_STRAINS):
        dR = wp.mat33(0.0)
    else:
        RtdF = full_strain[i] * V_inv
        dR = 0.5 * (RtdF - wp.transpose(RtdF))

    dFe = R @ (Id + dSe + dR) - Id

    elastic_strain_delta[i] = dFe - dFe_prev
    elastic_strain[i] = dFe

    # elastic_strain_delta[i] = R @ (Id + dS - S) @ S
    # elastic_strain_delta[i] = R @ dS


@wp.kernel
def fill_collider_rigidity_matrices(
    node_positions: wp.array(dtype=wp.vec3),
    collider_volumes: wp.array(dtype=float),
    node_volumes: wp.array(dtype=float),
    collider: Collider,
    voxel_size: float,
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

    cvol = voxel_size * voxel_size * voxel_size

    if bc_active and collider_is_dynamic(collider_id, collider):
        J_rows[2 * i] = i
        J_rows[2 * i + 1] = i
        J_cols[2 * i] = 2 * collider_id
        J_cols[2 * i + 1] = 2 * collider_id + 1

        W = wp.skew(collider_coms[collider_id] - x)
        Id = wp.identity(n=3, dtype=float)
        J_values[2 * i] = W
        J_values[2 * i + 1] = Id

        bc_mass = node_volumes[i] * cvol * collider_density(collider_id, collider, collider_volumes)

        IJtm_values[2 * i] = -bc_mass * collider_inv_inertia[collider_id] * W
        IJtm_values[2 * i + 1] = bc_mass / collider.masses[collider_id] * Id

        non_rigid_diagonal[i] = -Id

    else:
        J_cols[2 * i] = -1
        J_cols[2 * i + 1] = -1
        J_rows[2 * i] = -1
        J_rows[2 * i + 1] = -1

        non_rigid_diagonal[i] = wp.mat33(0.0)


@wp.kernel
def sample_grains(
    particles: wp.array(dtype=wp.vec3),
    radius: float,
    positions: wp.array2d(dtype=wp.vec3),
):
    pid, k = wp.tid()

    rng = wp.rand_init(pid * positions.shape[1] + k)

    pos_loc = 2.0 * wp.vec3(wp.randf(rng) - 0.5, wp.randf(rng) - 0.5, wp.randf(rng) - 0.5) * radius
    positions[pid, k] = particles[pid] + pos_loc


@wp.kernel
def transform_grains(
    particle_pos_prev: wp.array(dtype=wp.vec3),
    particle_transform_prev: wp.array(dtype=wp.mat33),
    particle_pos: wp.array(dtype=wp.vec3),
    particle_transform: wp.array(dtype=wp.mat33),
    positions: wp.array2d(dtype=wp.vec3),
):
    pid, k = wp.tid()

    pos_adv = positions[pid, k]

    p_pos = particle_pos[pid]
    p_frame = particle_transform[pid]
    p_pos_prev = particle_pos_prev[pid]
    p_frame_prev = particle_transform_prev[pid]

    pos_loc = wp.inverse(p_frame_prev) @ (pos_adv - p_pos_prev)

    p_pos_adv = p_frame @ pos_loc + p_pos
    positions[pid, k] = p_pos_adv


@fem.integrand
def advect_grains(
    s: fem.Sample,
    domain: fem.Domain,
    grid_vel: fem.Field,
    dt: float,
    positions: wp.array(dtype=wp.vec3),
):
    x = domain(s)
    vel = grid_vel(s)
    pos_adv = x + dt * vel
    positions[s.qp_index] = pos_adv


@wp.kernel
def advect_grains_from_particles(
    dt: float,
    particle_pos_prev: wp.array(dtype=wp.vec3),
    particle_pos: wp.array(dtype=wp.vec3),
    particle_vel_grad: wp.array(dtype=wp.mat33),
    positions: wp.array2d(dtype=wp.vec3),
):
    pid, k = wp.tid()

    p_pos = particle_pos[pid]
    p_pos_prev = particle_pos_prev[pid]

    pos_loc = positions[pid, k] - p_pos_prev

    p_vel_grad = particle_vel_grad[pid]

    displ = dt * p_vel_grad * pos_loc + (p_pos - p_pos_prev)
    positions[pid, k] += displ


@wp.kernel
def project_grains(
    radius: float,
    particle_pos: wp.array(dtype=wp.vec3),
    particle_frames: wp.array(dtype=wp.mat33),
    positions: wp.array2d(dtype=wp.vec3),
):
    pid, k = wp.tid()

    pos_adv = positions[pid, k]

    p_pos = particle_pos[pid]
    p_frame = particle_frames[pid]

    # keep within source particle
    # pos_loc = wp.inverse(p_frame) @ (pos_adv - p_pos)
    # dist = wp.max(wp.abs(pos_loc))
    # if dist > radius:
    #     pos_loc = pos_loc / dist * radius
    # p_pos_adv = p_frame @ pos_loc + p_pos

    p_frame = (radius * radius) * p_frame * wp.transpose(p_frame)
    pos_loc = pos_adv - p_pos
    vn = wp.max(1.0, wp.dot(pos_loc, wp.inverse(p_frame) * pos_loc))
    p_pos_adv = pos_loc / wp.sqrt(vn) + p_pos

    positions[pid, k] = p_pos_adv


@wp.kernel
def pad_voxels(particle_q: wp.array(dtype=wp.vec3i), padded_q: wp.array4d(dtype=wp.vec3i)):
    pid = wp.tid()

    for i in range(3):
        for j in range(3):
            for k in range(3):
                padded_q[pid, i, j, k] = particle_q[pid] + wp.vec3i(i - 1, j - 1, k - 1)


@wp.func
def positive_mod3(x: int):
    return (x % 3 + 3) % 3


@wp.kernel
def node_color_27_stencil(
    voxels: wp.array2d(dtype=int),
    colors: wp.array(dtype=int),
    color_indices: wp.array(dtype=int),
):
    pid = wp.tid()

    c = voxels[pid]
    colors[pid] = positive_mod3(c[0]) * 9 + positive_mod3(c[1]) * 3 + positive_mod3(c[2])
    color_indices[pid] = pid


@wp.kernel
def node_color_8_stencil(
    voxels: wp.array2d(dtype=int),
    colors: wp.array(dtype=int),
    color_indices: wp.array(dtype=int),
):
    pid = wp.tid()

    c = voxels[pid]
    colors[pid] = ((c[0] & 1) << 2) + ((c[1] & 1) << 1) + (c[2] & 1)
    color_indices[pid] = pid


def _allocate_by_voxels(particle_q, voxel_size, padding_voxels: int = 0):
    volume = wp.Volume.allocate_by_voxels(
        voxel_points=particle_q.flatten(),
        voxel_size=voxel_size,
    )

    for _pad_i in range(padding_voxels):
        voxels = wp.empty((volume.get_voxel_count(),), dtype=wp.vec3i)
        volume.get_voxels(voxels)

        padded_voxels = wp.zeros((voxels.shape[0], 3, 3, 3), dtype=wp.vec3i)
        wp.launch(pad_voxels, voxels.shape[0], (voxels, padded_voxels))

        volume = wp.Volume.allocate_by_voxels(
            voxel_points=padded_voxels.flatten(),
            voxel_size=voxel_size,
        )

    return volume


@dataclass
class ImplicitMPMOptions:
    """Implicit MPM solver options."""

    # numerics
    max_iterations: int = 250
    """Maximum number of iterations for the rheology solver."""
    tolerance: float = 1.0e-5
    """Tolerance for the rheology solver."""
    voxel_size: float = 1.0
    """Size of the grid voxels."""
    grid_padding: int = 0
    """Number of empty cells to add around particles."""

    # grid
    dynamic_grid: bool = True
    """Whether to dynamically update the grid from particles at each time step."""
    gauss_seidel: bool = True
    """Whether to use Gauss-Seidel or Jacobi for the rheology solver."""

    # plasticity
    max_fraction: float = 1.0
    """Maximum packing fraction for particles."""
    unilateral: bool = True
    """Whether to use unilateral of full incompressibility."""
    yield_stresses: tuple[float, float, float] = (0.0, -1.0e8, 1.0e8)
    """Yield stresses for the plasticity model."""

    # elasticity (experimental)
    compliance: float = 0.0
    """Compliance for the elasticity model. Experimental."""
    poisson_ratio: float = 0.3
    """Poisson's ratio for the elasticity model."""


class _ImplicitMPMScratchpad:
    """Stratch data for the implicit MPM solver"""

    def __init__(self):
        self.velocity_test = None
        self.velocity_trial = None
        self.fraction_test = None

        self.sym_strain_test = None
        self.full_strain_test = None
        self.divergence_test = None
        self.fraction_field = None

        self.collider_velocity_field = None
        self.elastic_strain_field = None
        self.elastic_strain_delta_field = None

        self.strain_matrix = sp.bsr_zeros(0, 0, mat63)
        self.transposed_strain_matrix = sp.bsr_zeros(0, 0, mat36)

        self.color_offsets, self.color_indices = None, None

        self.collider_quadrature = None

        self.inv_mass_matrix = None
        self.collider_normal = None
        self.collider_friction = None
        self.collider_inv_mass_matrix = None

        self.strain_node_particle_volume = None
        self.strain_node_volume = None
        self.strain_node_collider_volume = None

        self.int_symmetric_strain = None

        self.collider_total_volumes = None
        self.vel_node_volume = None

        self.scaled_stress_strain_mat = None

        self.elastic_rotation = None
        self.prev_symmetric_strain = None

        self.scaled_yield_stress = None

    def create_function_spaces_and_fields(
        self,
        geo_partition: fem.GeometryPartition,
        strain_degree: int = 1,
    ):
        domain = fem.Cells(geo_partition)
        grid = domain.geometry

        # Define function spaces: linear (Q1) for velocity and volume fraction,
        # piecewise-constant for pressure
        velocity_basis = fem.make_polynomial_basis_space(grid, degree=1)
        strain_basis = fem.make_polynomial_basis_space(
            grid,
            strain_degree,
            # discontinuous=True,
            # element_basis=fem.ElementBasis.NONCONFORMING_POLYNOMIAL,
        )

        velocity_space = fem.make_collocated_function_space(velocity_basis, dtype=wp.vec3)
        fraction_space = fem.make_collocated_function_space(velocity_basis, dtype=float)
        full_strain_space = fem.make_collocated_function_space(strain_basis, dtype=wp.mat33)
        sym_strain_space = fem.make_collocated_function_space(
            strain_basis,
            dof_mapper=fem.SymmetricTensorMapper(dtype=wp.mat33, mapping=fem.SymmetricTensorMapper.Mapping.DB16),
        )
        divergence_space = fem.make_collocated_function_space(strain_basis, dtype=float)

        vel_space_partition = fem.make_space_partition(
            space_topology=velocity_space.topology, geometry_partition=domain.geometry_partition, with_halo=False
        )
        strain_space_partition = fem.make_space_partition(
            space_topology=sym_strain_space.topology, geometry_partition=domain.geometry_partition, with_halo=False
        )

        # test, trial and discrete fields
        self.velocity_test = fem.make_test(velocity_space, domain=domain, space_partition=vel_space_partition)
        self.velocity_trial = fem.make_trial(velocity_space, domain=domain, space_partition=vel_space_partition)
        self.fraction_test = fem.make_test(
            fraction_space,
            space_restriction=self.velocity_test.space_restriction,
        )
        self.collider_velocity_field = velocity_space.make_field(space_partition=vel_space_partition)
        self.collider_distance_field = fraction_space.make_field(space_partition=vel_space_partition)

        self.sym_strain_test = fem.make_test(sym_strain_space, domain=domain, space_partition=strain_space_partition)
        self.full_strain_test = fem.make_test(
            full_strain_space, space_restriction=self.sym_strain_test.space_restriction
        )
        self.divergence_test = fem.make_test(
            divergence_space,
            space_restriction=self.sym_strain_test.space_restriction,
        )
        self.elastic_strain_field = full_strain_space.make_field(space_partition=strain_space_partition)
        self.elastic_strain_delta_field = full_strain_space.make_field(space_partition=strain_space_partition)

        collider_quadrature_order = strain_degree + 1
        self.collider_quadrature = fem.RegularQuadrature(
            domain=domain,
            order=collider_quadrature_order,
            family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE,
        )

        self.impulse_field = velocity_space.make_field(space_partition=vel_space_partition)
        self.velocity_field = velocity_space.make_field(space_partition=vel_space_partition)
        self.stress_field = sym_strain_space.make_field(space_partition=strain_space_partition)
        self.collider_ids = wp.empty(velocity_space.node_count(), dtype=int)

    def allocate_temporaries(
        self,
        temporary_store: fem.TemporaryStore,
        collider_count: int,
        has_compliant_bodies: bool,
        elastic: bool,
    ):
        vel_node_count = self.velocity_test.space_partition.node_count()
        strain_node_count = self.sym_strain_test.space_partition.node_count()

        self.inv_mass_matrix = fem.borrow_temporary(temporary_store, shape=(vel_node_count,), dtype=float)
        self.collider_normal = fem.borrow_temporary(temporary_store, shape=(vel_node_count,), dtype=wp.vec3)
        self.collider_friction = fem.borrow_temporary(temporary_store, shape=(vel_node_count,), dtype=float)
        self.collider_inv_mass_matrix = fem.borrow_temporary(temporary_store, shape=(vel_node_count,), dtype=float)

        self.strain_node_particle_volume = fem.borrow_temporary(temporary_store, shape=strain_node_count, dtype=float)
        self.strain_node_volume = fem.borrow_temporary(temporary_store, shape=strain_node_count, dtype=float)
        self.strain_node_collider_volume = fem.borrow_temporary(temporary_store, shape=strain_node_count, dtype=float)

        self.int_symmetric_strain = fem.borrow_temporary(temporary_store, shape=strain_node_count, dtype=vec6)

        self.scaled_yield_stress = fem.borrow_temporary(temporary_store, shape=strain_node_count, dtype=wp.vec3)

        if has_compliant_bodies:
            self.collider_total_volumes = fem.borrow_temporary(temporary_store, shape=collider_count, dtype=float)
            self.vel_node_volume = fem.borrow_temporary(temporary_store, shape=(vel_node_count,), dtype=float)

        if elastic:
            self.scaled_stress_strain_mat = fem.borrow_temporary(temporary_store, shape=strain_node_count, dtype=mat66)

        if elastic or not _SMALL_STRAINS:
            self.elastic_rotation = fem.borrow_temporary(temporary_store, shape=strain_node_count, dtype=wp.quatf)
            self.prev_symmetric_strain = fem.borrow_temporary(temporary_store, shape=strain_node_count, dtype=vec6)

    def compute_coloring(
        self,
        temporary_store: fem.TemporaryStore,
    ):
        grid = self.sym_strain_test.geometry
        strain_node_count = self.sym_strain_test.space.node_count()
        degree = self.sym_strain_test.space.degree
        self.color_offsets, self.color_indices = self._compute_coloring(
            grid, degree, strain_node_count, temporary_store
        )

    def _compute_coloring(self, grid, degree, strain_node_count, temporary_store: fem.TemporaryStore):
        colors = fem.borrow_temporary(temporary_store, shape=strain_node_count * 2, dtype=int)
        color_indices = fem.borrow_temporary(temporary_store, shape=strain_node_count * 2, dtype=int)

        if degree == 1:
            if isinstance(grid, fem.Nanogrid):
                voxels = grid.vertex_grid.get_voxels()
            else:
                voxels = wp.array(
                    np.stack(
                        np.meshgrid(
                            np.arange(grid.res[0] + 1),
                            np.arange(grid.res[1] + 1),
                            np.arange(grid.res[2] + 1),
                            indexing="ij",
                        ),
                        axis=-1,
                    ).reshape(-1, 3),
                    dtype=int,
                )
            wp.launch(
                node_color_27_stencil,
                dim=strain_node_count,
                inputs=[voxels, colors.array, color_indices.array],
            )
        else:
            if isinstance(grid, fem.Nanogrid):
                voxels = grid.cell_grid.get_voxels()
            else:
                voxels = wp.array(
                    np.stack(
                        np.meshgrid(
                            np.arange(grid.res[0]),
                            np.arange(grid.res[1]),
                            np.arange(grid.res[2]),
                            indexing="ij",
                        ),
                        axis=-1,
                    ).reshape(-1, 3),
                    dtype=int,
                )
            wp.launch(
                node_color_8_stencil,
                dim=strain_node_count,
                inputs=[voxels, colors.array, color_indices.array],
            )

        wp.utils.radix_sort_pairs(
            keys=colors.array,
            values=color_indices.array,
            count=strain_node_count,
        )

        unique_colors = colors.array[strain_node_count:]
        color_node_counts = color_indices.array[strain_node_count:]
        color_count = wp.utils.runlength_encode(
            colors.array,
            run_values=unique_colors,
            run_lengths=color_node_counts,
            value_count=strain_node_count,
        )

        color_offsets = np.concatenate([[0], np.cumsum(color_node_counts[:color_count].numpy())])

        return color_offsets, color_indices


@wp.kernel
def clamp_coordinates(
    coords: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    coords[i] = wp.min(wp.max(coords[i], wp.vec3(0.0)), wp.vec3(1.0))


def _get_array(temp):
    return None if temp is None else temp.array


class SolverImplicitMPM(SolverBase):
    """Implicit MPM solver.

    This solver implements an implicit MPM algorithm for granular materials,
    roughly following [1] but with a GPU-friendly rheology solver.

    This variant of MPM is mostly interesting for very stiff materials, especially
    in the fully inelastic limit, but is not as versatile as more traditional explicit approaches.

    [1] https://doi.org/10.1145/2897824.2925877

    Args:
        model: The model to solve.
        options: The solver options.

    Returns:
        The solver.
    """

    Options = ImplicitMPMOptions

    def __init__(
        self,
        model: Model,
        options: ImplicitMPMOptions,
    ):
        super().__init__(model)

        # Compute density from particle mass and radius
        # TODO support for varying properties
        if len(model.particle_mass) > 0 and len(model.particle_radius) > 0:
            self.density = float(
                model.particle_mass[:1].numpy()[0] / (4.0 / 3.0 * np.pi * model.particle_radius[:1].numpy()[0] ** 3)
            )
        else:
            self.density = 1.0

        self.friction_coeff = float(model.particle_mu)
        self.yield_stresses = wp.vec3f(options.yield_stresses)
        self.unilateral = options.unilateral
        self.max_fraction = float(options.max_fraction)

        self.max_iterations = options.max_iterations
        self.tolerance = float(options.tolerance)

        self.voxel_size = float(options.voxel_size)
        self.degree = 1 if options.unilateral else 0

        self.grid_padding = options.grid_padding
        self.dynamic_grid = options.dynamic_grid
        self.coloring = options.gauss_seidel

        # Elastic stress-strain matrix from Poisson's ratio and compliance
        poisson_ratio = options.poisson_ratio
        lame = 1.0 / (1.0 + poisson_ratio) * np.array([poisson_ratio / (1.0 - 2.0 * poisson_ratio), 0.5])
        K = options.compliance
        self.stress_strain_mat = mat66(K / (2.0 * lame[1]) * np.eye(6))
        self.stress_strain_mat[0, 0] = K / (2.0 * lame[1] + 3.0 * lame[0])

        self._elastic = K != 0.0
        self._strain_flip_ratio = 0.95

        self.setup_collider(model)

        self.temporary_store = fem.TemporaryStore()

        self._use_cuda_graph = False
        if self.model.device.is_cuda:
            try:
                assert_conditional_graph_support()
                self._use_cuda_graph = True
            except Exception:
                pass

        self._enable_timers = False
        self._timers_use_nvtx = False

        self._fixed_scratchpad = None

    @staticmethod
    def enrich_state(state: State):
        """Enrich the state with additional fields for tracking particle strain and deformation."""

        device = state.particle_qd.device
        state.particle_qd_grad = wp.zeros(state.particle_qd.shape[0], dtype=wp.mat33, device=device)
        state.particle_elastic_strain = wp.zeros(state.particle_qd.shape[0], dtype=wp.mat33, device=device)
        state.particle_transform = wp.full(
            state.particle_qd.shape[0], value=wp.mat33(np.eye(3)), dtype=wp.mat33, device=device
        )

        state.velocity_field = None
        state.impulse_field = None
        state.stress_field = None
        state.collider_ids = None

    def setup_collider(
        self,
        model: Model,
        # TODO: read colliders from model
        colliders: list[wp.Mesh] | None = None,
        collider_thicknesses: list[float] | None = None,
        collider_projection_threshold: list[float] | None = None,
        collider_masses: list[float] | None = None,
        collider_friction: list[float] | None = None,
    ):
        """Setups the collision geometry for the implicit MPM solver.


        Args:
            model: The model to read ground collision properties from.
            colliders: A list of warp triangular meshes to use as colliders.
            collider_thicknesses: The thicknesses of the colliders.
            collider_projection_threshold: The projection threshold for the colliders, i.e, the maximum acceptable penetration depth before projecting particles out.
            collider_masses: The masses of the colliders.
            collider_friction: The friction coefficients of the colliders.
        """

        self._collider_meshes = colliders  # Keep a ref so that meshes are not garbage collected

        if colliders is None:
            colliders = []

        collider = Collider()

        with wp.ScopedDevice(model.device):
            collider.meshes = wp.array([collider.id for collider in colliders], dtype=wp.uint64)
            collider.thicknesses = (
                wp.full(len(collider.meshes), _DEFAULT_THICKNESS * self.voxel_size, dtype=float)
                if collider_thicknesses is None
                else wp.array(collider_thicknesses, dtype=float)
            )
            collider.friction = (
                wp.full(len(collider.meshes), _DEFAULT_FRICTION, dtype=float)
                if collider_friction is None
                else wp.array(collider_friction, dtype=float)
            )
            collider.masses = (
                wp.full(len(collider.meshes), INFINITE_MASS * 2.0, dtype=float)
                if collider_masses is None
                else wp.array(collider_masses, dtype=float)
            )
            collider.projection_threshold = (
                wp.full(len(collider.meshes), DEFAULT_PROJECTION_THRESHOLD, dtype=float)
                if collider_projection_threshold is None
                else wp.array(collider_projection_threshold, dtype=float)
            )

        collider.query_max_dist = self.voxel_size
        collider.ground_height = 0.0
        collider.ground_normal = wp.vec3(0.0)
        collider.ground_normal[model.up_axis] = 1.0

        self._has_compliant_bodies = len(collider.masses) > 0 and np.min(collider.masses.numpy()) < INFINITE_MASS
        self.collider_coms = wp.zeros(len(collider.meshes), dtype=wp.vec3)
        self.collider_inv_inertia = wp.zeros(len(collider.meshes), dtype=wp.mat33)

        self.collider = collider

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ):
        model = self.model

        with wp.ScopedDevice(model.device):
            if self.dynamic_grid:
                scratch = self._rebuild_scratchpad(state_in)
            else:
                if self._fixed_scratchpad is None:
                    self._fixed_scratchpad = self._rebuild_scratchpad(state_in)
                scratch = self._fixed_scratchpad

            self._step_impl(state_in, state_out, dt, scratch)

    def project_outside(self, state_in: State, state_out: State, dt: float):
        """Projects particles outside of the colliders"""
        wp.launch(
            project_outside_collider,
            dim=state_in.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_in.particle_qd_grad,
                self.collider,
                self.voxel_size,
                dt,
            ],
            outputs=[
                state_out.particle_q,
                state_out.particle_qd,
                state_out.particle_qd_grad,
            ],
            device=state_in.particle_q.device,
        )

    def collect_collider_impulses(self, state: State):
        """Returns the list of collider impulses and the positions at which they are applied.

        Note: the identifier of the collider is not included in the returned values but can be retrieved from the state
        as `state.collider_ids`.
        """
        x = state.velocity_field.space.node_positions()

        collider_impulse = wp.zeros_like(state.impulse_field.dof_values)
        cell_volume = self.voxel_size**3
        fem.utils.array_axpy(
            y=collider_impulse,
            x=state.impulse_field.dof_values,
            alpha=-self.density * cell_volume,
            beta=0.0,
        )

        return collider_impulse, x

    def update_particle_frames(
        self,
        state_prev: State,
        state: State,
        dt: float,
        min_stretch: float = 0.25,
        max_stretch: float = 2.0,
    ):
        """Updates the particle frames to account for the deformation of the particles"""

        wp.launch(
            update_particle_frames,
            dim=state.particle_count,
            inputs=[
                dt,
                min_stretch,
                max_stretch,
                state.particle_qd_grad,
                state_prev.particle_transform,
                state.particle_transform,
            ],
            device=state.particle_qd_grad.device,
        )

    def sample_render_grains(self, state: State, particle_radius: float, grains_per_particle: int):
        """
        Create point samples for rendering at higher resolution than the simulation particles.
        Point samples are advected passievely with the continuum velocity field while being constrained
        to lie within the affinely deformed simulation particles.
        """

        grains = wp.empty((state.particle_count, grains_per_particle), dtype=wp.vec3, device=state.particle_q.device)

        wp.launch(
            sample_grains,
            dim=grains.shape,
            inputs=[
                state.particle_q,
                particle_radius,
                grains,
            ],
            device=state.particle_q.device,
        )

        return grains

    def update_render_grains(
        self,
        state_prev: State,
        state: State,
        grains: wp.array,
        particle_radius: float,
        dt: float,
    ):
        """Advect the render grains at the current time step"""

        if self.velocity_field is None:
            return

        grain_pos = grains.flatten()
        domain = fem.Cells(state.velocity_field.space.geometry)
        grain_pic = fem.PicQuadrature(domain, positions=grain_pos)

        wp.launch(
            advect_grains_from_particles,
            dim=grains.shape,
            inputs=[
                dt,
                state_prev.particle_q,
                state.particle_q,
                state.particle_qd_grad,
                grains,
            ],
            device=grains.device,
        )

        fem.interpolate(
            advect_grains,
            quadrature=grain_pic,
            values={
                "dt": dt,
                "positions": grain_pos,
            },
            fields={
                "grid_vel": state.velocity_field,
            },
            device=grains.device,
        )

        wp.launch(
            project_grains,
            dim=grains.shape,
            inputs=[
                particle_radius,
                state.particle_q,
                state.particle_transform,
                grains,
            ],
            device=grains.device,
        )

    def _allocate_grid(self, positions: wp.array, voxel_size, padding_voxels: int = 0):
        with wp.ScopedTimer(
            "Allocate grid",
            active=self._enable_timers,
            use_nvtx=self._timers_use_nvtx,
            synchronize=True,
        ):
            if self.dynamic_grid:
                volume = _allocate_by_voxels(positions, voxel_size, padding_voxels=padding_voxels)
                grid = fem.Nanogrid(volume)
            else:
                pos_np = positions.numpy()
                bbox_min, bbox_max = np.min(pos_np, axis=0), np.max(pos_np, axis=0)
                bbox_max += padding_voxels * voxel_size
                bbox_min -= padding_voxels * voxel_size
                grid = fem.Grid3D(
                    bounds_lo=wp.vec3(bbox_min),
                    bounds_hi=wp.vec3(bbox_max),
                    res=wp.vec3i(np.ceil((bbox_max - bbox_min) // voxel_size).astype(int)),
                )

        return grid

    def _rebuild_scratchpad(self, state_in: State):
        geo_partition = self._allocate_grid(
            state_in.particle_q, voxel_size=self.voxel_size, padding_voxels=self.grid_padding
        )

        with wp.ScopedTimer(
            "Scratchpad",
            active=self._enable_timers,
            use_nvtx=self._timers_use_nvtx,
            synchronize=True,
        ):
            scratch = _ImplicitMPMScratchpad()
            scratch.create_function_spaces_and_fields(geo_partition, strain_degree=self.degree)

            scratch.allocate_temporaries(
                self.temporary_store,
                collider_count=self.collider.meshes.shape[0],
                has_compliant_bodies=self._has_compliant_bodies,
                elastic=self._elastic,
            )

            if self.coloring:
                scratch.compute_coloring(self.temporary_store)

        return scratch

    def _step_impl(
        self,
        state_in: State,
        state_out: State,
        dt: float,
        scratch: _ImplicitMPMScratchpad,
    ):
        domain = scratch.velocity_test.domain
        inv_cell_volume = 1.0 / self.voxel_size**3

        model = self.model

        with wp.ScopedTimer(
            "Warmstart fields",
            active=self._enable_timers,
            use_nvtx=self._timers_use_nvtx,
            synchronize=True,
        ):
            self._warmstart_fields(scratch, state_in, state_out)

        # Bin particles to grid cells
        with wp.ScopedTimer(
            "Bin particles",
            active=self._enable_timers,
            use_nvtx=self._timers_use_nvtx,
            synchronize=True,
        ):
            pic = fem.PicQuadrature(
                domain=domain,
                positions=state_in.particle_q,
                measures=model.particle_mass,
            )

            if not self.dynamic_grid:
                wp.launch(
                    clamp_coordinates,
                    dim=pic.particle_coords.shape,
                    inputs=[pic.particle_coords],
                )

        vel_node_count = state_out.velocity_field.space.node_count()
        strain_node_count = state_out.stress_field.space.node_count()

        # Velocity right-hand side and inverse mass matrix
        with wp.ScopedTimer(
            "Free velocity",
            active=self._enable_timers,
            use_nvtx=self._timers_use_nvtx,
            synchronize=True,
        ):
            velocity_int = fem.integrate(
                integrate_velocity,
                quadrature=pic,
                fields={"u": scratch.velocity_test},
                values={
                    "velocities": state_in.particle_qd,
                    "velocity_gradients": state_in.particle_qd_grad,
                    "dt": dt,
                    "gravity": model.gravity,
                    "inv_cell_volume": inv_cell_volume,
                },
                output_dtype=wp.vec3,
            )

            node_particle_volume = fem.integrate(
                integrate_fraction,
                quadrature=pic,
                fields={"phi": scratch.fraction_test},
                values={"inv_cell_volume": inv_cell_volume},
                output_dtype=float,
            )
            fem.interpolate(
                free_velocity,
                dest=fem.make_restriction(
                    state_out.velocity_field, space_restriction=scratch.velocity_test.space_restriction
                ),
                values={
                    "velocity_int": velocity_int,
                    "particle_volume": node_particle_volume,
                    "inv_mass_matrix": scratch.inv_mass_matrix.array,
                },
            )

        with wp.ScopedTimer(
            "Rasterize collider",
            active=self._enable_timers,
            use_nvtx=self._timers_use_nvtx,
            synchronize=True,
        ):
            wp.launch(
                rasterize_collider,
                dim=vel_node_count,
                inputs=[
                    self.collider,
                    self.voxel_size,
                    scratch.collider_velocity_field.space.node_positions(),
                    scratch.collider_distance_field.dof_values,
                    scratch.collider_velocity_field.dof_values,
                    scratch.collider_normal.array,
                    scratch.collider_friction.array,
                    state_out.collider_ids,
                    state_out.impulse_field.dof_values,
                ],
            )

        if self._has_compliant_bodies:
            with wp.ScopedTimer(
                "Collider compliance",
                active=self._enable_timers,
                use_nvtx=self._timers_use_nvtx,
                synchronize=True,
            ):
                # Accumulate collider volume so we can distribute mass
                # and record cells with collider to build subdomain
                scratch.collider_total_volumes.array.zero_()
                fem.interpolate(
                    collider_volumes,
                    quadrature=scratch.collider_quadrature,
                    values={
                        "collider": self.collider,
                        "volumes": scratch.collider_total_volumes.array,
                        "voxel_size": self.voxel_size,
                    },
                )
                fem.integrate(
                    integrate_fraction,
                    fields={"phi": scratch.fraction_test},
                    values={"inv_cell_volume": inv_cell_volume},
                    output=scratch.vel_node_volume.array,
                )

                wp.launch(
                    collider_inverse_mass,
                    dim=vel_node_count,
                    inputs=[
                        self.density,
                        self.collider,
                        state_out.collider_ids,
                        scratch.vel_node_volume.array,
                        scratch.collider_total_volumes.array,
                        scratch.collider_inv_mass_matrix.array,
                    ],
                )

                rigidity_matrix = self._build_rigidity_matrix(
                    scratch.collider_total_volumes.array,
                    scratch.vel_node_volume.array,
                    state_out.collider_ids,
                )
        else:
            rigidity_matrix = None
            scratch.collider_inv_mass_matrix.array.zero_()

        with wp.ScopedTimer(
            "Compute strain-node volumes",
            active=self._enable_timers,
            use_nvtx=self._timers_use_nvtx,
            synchronize=True,
        ):
            fem.integrate(
                integrate_fraction,
                quadrature=pic,
                fields={"phi": scratch.divergence_test},
                values={"inv_cell_volume": inv_cell_volume},
                output=scratch.strain_node_particle_volume.array,
            )
            fem.integrate(
                integrate_fraction,
                fields={"phi": scratch.divergence_test},
                values={"inv_cell_volume": inv_cell_volume},
                output=scratch.strain_node_volume.array,
            )
            fem.integrate(
                integrate_collider_fraction,
                quadrature=scratch.collider_quadrature,
                fields={
                    "phi": scratch.divergence_test,
                    "sdf": scratch.collider_distance_field,
                },
                values={
                    "inv_cell_volume": inv_cell_volume,
                },
                output=scratch.strain_node_collider_volume.array,
            )

        if self._elastic:
            with wp.ScopedTimer(
                "Elastic strain rhs",
                active=self._enable_timers,
                use_nvtx=self._timers_use_nvtx,
                synchronize=True,
            ):
                fem.integrate(
                    integrate_elastic_strain,
                    quadrature=pic,
                    fields={"tau": scratch.full_strain_test},
                    values={
                        "elastic_strains": state_in.particle_elastic_strain,
                        "inv_cell_volume": inv_cell_volume,
                    },
                    output=scratch.elastic_strain_field.dof_values,
                )

                wp.launch(
                    elastic_strain_decomposition,
                    dim=strain_node_count,
                    inputs=[
                        scratch.elastic_strain_field.dof_values,
                        scratch.strain_node_particle_volume.array,
                        _get_array(scratch.elastic_rotation),
                        _get_array(scratch.int_symmetric_strain),
                        _get_array(scratch.prev_symmetric_strain),
                    ],
                )

                wp.launch(
                    kernel=scale_strain_stress_matrices,
                    dim=strain_node_count,
                    inputs=[
                        self.stress_strain_mat,
                        scratch.strain_node_particle_volume.array,
                        scratch.scaled_stress_strain_mat.array,
                    ],
                )
        else:
            scratch.int_symmetric_strain.array.zero_()

        # Void fraction (unilateral incompressibility offset)
        if self.unilateral:
            with wp.ScopedTimer(
                "Unilateral offset",
                active=self._enable_timers,
                use_nvtx=self._timers_use_nvtx,
                synchronize=True,
            ):
                wp.launch(
                    add_unilateral_strain_offset,
                    dim=strain_node_count,
                    inputs=[
                        self.max_fraction,
                        scratch.strain_node_particle_volume.array,
                        scratch.strain_node_collider_volume.array,
                        scratch.strain_node_volume.array,
                        _get_array(scratch.prev_symmetric_strain),
                        scratch.int_symmetric_strain.array,
                    ],
                )

        scratch.scaled_yield_stress.array.fill_(self.yield_stresses / self.density)

        # Strain jacobian
        strain_form = small_strain_form if _SMALL_STRAINS else finite_strain_form

        with wp.ScopedTimer(
            "Strain matrix",
            active=self._enable_timers,
            use_nvtx=self._timers_use_nvtx,
            synchronize=True,
        ):
            sp.bsr_set_zero(
                scratch.strain_matrix,
                rows_of_blocks=strain_node_count,
                cols_of_blocks=vel_node_count,
            )
            fem.integrate(
                strain_form,
                quadrature=pic,
                fields={
                    "u": scratch.velocity_trial,
                    "tau": scratch.sym_strain_test,
                },
                values={
                    "dt": dt,
                    "inv_cell_volume": inv_cell_volume,
                    "elastic_strain": _get_array(scratch.prev_symmetric_strain),
                    "rotation": _get_array(scratch.elastic_rotation),
                },
                output_dtype=float,
                output=scratch.strain_matrix,
            )
            scratch.strain_matrix.nnz_sync()

        with wp.ScopedTimer(
            "Strain solve",
            active=self._enable_timers,
            use_nvtx=self._timers_use_nvtx,
            synchronize=True,
        ):
            solve_rheology(
                self.unilateral,
                self.friction_coeff,
                self.max_iterations,
                self.tolerance,
                scratch.strain_matrix,
                scratch.transposed_strain_matrix,
                scratch.inv_mass_matrix.array,
                _get_array(scratch.scaled_yield_stress),
                _get_array(scratch.scaled_stress_strain_mat),
                scratch.int_symmetric_strain.array,
                state_out.stress_field.dof_values,
                state_out.velocity_field.dof_values,
                scratch.collider_friction.array,
                scratch.collider_normal.array,
                scratch.collider_velocity_field.dof_values,
                scratch.collider_inv_mass_matrix.array,
                state_out.impulse_field.dof_values,
                color_offsets=scratch.color_offsets,
                color_indices=_get_array(scratch.color_indices),
                rigidity_mat=rigidity_matrix,
                temporary_store=self.temporary_store,
                use_graph=self._use_cuda_graph,
            )

        # (A)PIC advection
        with wp.ScopedTimer(
            "Advection",
            active=self._enable_timers,
            use_nvtx=self._timers_use_nvtx,
            synchronize=True,
        ):
            fem.interpolate(
                advect_particles,
                quadrature=pic,
                values={
                    "pos": state_out.particle_q,
                    "pos_prev": state_in.particle_q,
                    "vel": state_out.particle_qd,
                    "vel_grad": state_out.particle_qd_grad,
                    "dt": dt,
                },
                fields={
                    "grid_vel": state_out.velocity_field,
                },
            )

        if self._elastic:
            with wp.ScopedTimer(
                "Elastic strain update",
                active=self._enable_timers,
                use_nvtx=self._timers_use_nvtx,
                synchronize=True,
            ):
                # Compute total strain from velocity field
                full_strain = fem.integrate(
                    strain_form,
                    quadrature=pic,
                    fields={
                        "u": state_out.velocity_field,
                        "tau": scratch.full_strain_test,
                    },
                    values={
                        "dt": dt,
                        "inv_cell_volume": inv_cell_volume,
                        "elastic_strain": _get_array(scratch.prev_symmetric_strain),
                        "rotation": _get_array(scratch.elastic_rotation),
                    },
                    output_dtype=wp.mat33,
                )

                # Compute elastic strain delta on grid
                wp.launch(
                    compute_elastic_strain_delta,
                    dim=strain_node_count,
                    inputs=[
                        scratch.strain_node_particle_volume.array,
                        _get_array(scratch.elastic_rotation),
                        _get_array(scratch.int_symmetric_strain),
                        full_strain,
                        scratch.elastic_strain_field.dof_values,
                        scratch.elastic_strain_delta_field.dof_values,
                    ],
                )

                # Update particle elastic strain from grid strain delta
                fem.interpolate(
                    update_particle_elastic_strain,
                    quadrature=pic,
                    values={
                        "dt": dt,
                        "flip": self._strain_flip_ratio,
                        "elastic_strain_prev": state_in.particle_elastic_strain,
                        "elastic_strain": state_out.particle_elastic_strain,
                    },
                    fields={
                        "grid_vel": state_out.velocity_field,
                        "grid_strain": scratch.elastic_strain_field,
                        "grid_strain_delta": scratch.elastic_strain_delta_field,
                    },
                )

    def _build_rigidity_matrix(self, collider_volumes, node_volumes, collider_ids):
        """Assembles the rigidity matrix for the current time step
        (propagates local impulses to the rest of the rigid body)
        """

        vel_node_count = self.velocity_field.space.node_count()
        collider_count = self.collider.meshes.shape[0]

        J_rows = wp.empty(vel_node_count * 2, dtype=int)
        J_cols = wp.empty(vel_node_count * 2, dtype=int)
        J_values = wp.empty(vel_node_count * 2, dtype=wp.mat33)
        IJtm_values = wp.empty(vel_node_count * 2, dtype=wp.mat33)
        Iphi_diag = wp.empty(vel_node_count, dtype=wp.mat33)

        with wp.ScopedTimer("Fill rigidity matrix", synchronize=True, active=self._enable_timers):
            wp.launch(
                fill_collider_rigidity_matrices,
                dim=vel_node_count,
                inputs=[
                    self.velocity_field.space.node_positions(),
                    collider_volumes,
                    node_volumes,
                    self.collider,
                    self.voxel_size,
                    collider_ids,
                    self.collider_coms,
                    self.collider_inv_inertia,
                    J_rows,
                    J_cols,
                    J_values,
                    IJtm_values,
                    Iphi_diag,
                ],
            )

        with wp.ScopedTimer(
            "Build rigidity matrix",
            synchronize=True,
            active=self._enable_timers,
            use_nvtx=self._timers_use_nvtx,
        ):
            J = sp.bsr_from_triplets(
                rows_of_blocks=vel_node_count,
                cols_of_blocks=2 * collider_count,
                rows=J_rows,
                columns=J_cols,
                values=J_values,
            )

            IJtm = sp.bsr_from_triplets(
                cols_of_blocks=vel_node_count,
                rows_of_blocks=2 * collider_count,
                columns=J_rows,
                rows=J_cols,
                values=IJtm_values,
            )

        with wp.ScopedTimer(
            "Assemble rigidity matrix",
            synchronize=True,
            active=self._enable_timers,
            use_nvtx=self._timers_use_nvtx,
        ):
            Iphi = sp.bsr_diag(Iphi_diag)
            Iphi = sp.bsr_from_triplets(
                rows_of_blocks=Iphi.nrow,
                cols_of_blocks=Iphi.ncol,
                columns=Iphi.columns,
                rows=Iphi.uncompress_rows(),
                values=Iphi.values,
            )
            rigid = Iphi + J @ IJtm
            rigid.nnz_sync()

        return rigid

    def _warmstart_fields(self, scratch: _ImplicitMPMScratchpad, state_in: State, state_out: State):
        domain = scratch.velocity_test.domain

        if state_in.impulse_field is not None:
            if self.dynamic_grid:
                prev_impulse_field = fem.NonconformingField(domain, state_in.impulse_field)
                fem.interpolate(
                    prev_impulse_field,
                    dest=fem.make_restriction(
                        scratch.impulse_field, space_restriction=scratch.velocity_test.space_restriction
                    ),
                )
            else:
                scratch.impulse_field.dof_values.assign(state_in.impulse_field.dof_values)

        # Interpolate previous stress
        if state_in.stress_field is not None:
            if self.dynamic_grid:
                prev_stress_field = fem.NonconformingField(domain, state_in.stress_field)
                fem.interpolate(
                    prev_stress_field,
                    dest=fem.make_restriction(
                        scratch.stress_field, space_restriction=scratch.sym_strain_test.space_restriction
                    ),
                )
            else:
                scratch.stress_field.dof_values.assign(state_in.stress_field.dof_values)

        if (
            state_out.velocity_field is None
            or state_out.velocity_field.space_partition != scratch.velocity_field.space_partition
        ):
            state_out.velocity_field = scratch.velocity_field
            state_out.impulse_field = scratch.impulse_field
            state_out.stress_field = scratch.stress_field
            state_out.collider_ids = scratch.collider_ids
        else:
            state_out.velocity_field.dof_values.assign(scratch.velocity_field.dof_values)
            state_out.impulse_field.dof_values.assign(scratch.impulse_field.dof_values)
            state_out.stress_field.dof_values.assign(scratch.stress_field.dof_values)
            state_out.collider_ids.assign(scratch.collider_ids)
