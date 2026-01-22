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

###########################################################################
# Example Cholesky Cosserat Rod - Multi-Tile (3-DOF Tangent Space)
#
# Demonstrates a Block-Jacobi approach with Cholesky solving for Position
# And Orientation Based Cosserat Rods on longer particle chains.
#
# This combines:
#   - Global Cholesky solving with 3-DOF tangent-space from example 08
#   - Multi-tile Block-Jacobi partitioning from example 01
#
# Key difference from examples 07/08_global_cosserat_rod_cholesky:
#   - Uses full 3×3 constraint formulation (96 scalar DOFs per tile)
#   - Each constraint row represents one component of the 3D violation
#
# For 129 particles with 128 stretch + 127 bend constraints:
#   - Stretch tiles: 4 tiles of 32 constraints each (96 scalar DOFs per tile)
#   - Bend tiles: 4 tiles (31, 32, 32, 32 constraints)
#
# Each tile is solved independently in parallel using Cholesky decomposition.
# Boundary particles/quaternions receive corrections from both adjacent tiles
# using atomic operations. Coupling between tiles happens through outer
# constraint iterations.
#
# STATUS: Partially working - explodes when changing rest shape via UI
#
# Command: uv run -m newton.examples cosserat_09_cholesky_cosserat_rod_multitile
#
###########################################################################

import math

import warp as wp

import newton
import newton.examples

# Warp tile configuration
BLOCK_DIM = 128
TILE = 96  # 32 constraints x 3 = 96 scalar DOFs per tile

# Rod configuration
NUM_TILES = 4
CONSTRAINTS_PER_TILE = 32
NUM_PARTICLES = NUM_TILES * CONSTRAINTS_PER_TILE + 1  # 129 particles
NUM_STRETCH = NUM_PARTICLES - 1  # 128 stretch constraints
NUM_BEND = NUM_PARTICLES - 2  # 127 bend constraints


@wp.func
def quat_rotate_e3(q: wp.quat) -> wp.vec3:
    """Compute the third director d3 = q * e3 * conjugate(q) where e3 = (0,0,1)."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    d3_x = 2.0 * (x * z + w * y)
    d3_y = 2.0 * (y * z - w * x)
    d3_z = w * w - x * x - y * y + z * z
    return wp.vec3(d3_x, d3_y, d3_z)


@wp.func
def quat_exp_approx(omega: wp.vec3) -> wp.quat:
    """Approximate quaternion exponential for small angles: exp(omega/2)."""
    half_omega = omega * 0.5
    angle_sq = wp.dot(half_omega, half_omega)

    if angle_sq < 1.0e-8:
        return wp.normalize(wp.quat(half_omega[0], half_omega[1], half_omega[2], 1.0))
    else:
        angle = wp.sqrt(angle_sq)
        s = wp.sin(angle) / angle
        c = wp.cos(angle)
        return wp.quat(s * half_omega[0], s * half_omega[1], s * half_omega[2], c)


@wp.kernel
def integrate_particles_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    gravity: wp.vec3,
    dt: float,
    # outputs
    particle_q_predicted: wp.array(dtype=wp.vec3),
    particle_qd_new: wp.array(dtype=wp.vec3),
):
    """Semi-implicit Euler integration step for particles."""
    tid = wp.tid()
    inv_mass = particle_inv_mass[tid]

    if inv_mass == 0.0:
        particle_q_predicted[tid] = particle_q[tid]
        particle_qd_new[tid] = particle_qd[tid]
        return

    v_new = particle_qd[tid] + gravity * dt
    x_predicted = particle_q[tid] + v_new * dt

    particle_q_predicted[tid] = x_predicted
    particle_qd_new[tid] = v_new


@wp.kernel
def assemble_stretch_block_systems_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    compliance_factor: float,
    stiffness: wp.vec3,
    num_stretch: int,
    constraints_per_tile: int,
    # outputs
    system_matrices: wp.array3d(dtype=float),  # (NUM_TILES, TILE, TILE)
    system_rhs: wp.array2d(dtype=float),  # (NUM_TILES, TILE)
):
    """Assemble block system matrices for stretch/shear constraints."""
    tile_idx = wp.tid()

    constraint_start = tile_idx * constraints_per_tile
    constraint_end = wp.min(constraint_start + constraints_per_tile, num_stretch)
    num_in_tile = constraint_end - constraint_start

    # Initialize to zero
    for i in range(TILE):
        system_rhs[tile_idx, i] = 0.0
        for j in range(TILE):
            system_matrices[tile_idx, i, j] = 0.0

    # For each stretch constraint in this tile
    for local_k in range(num_in_tile):
        global_k = constraint_start + local_k

        p0 = particle_q[global_k]
        p1 = particle_q[global_k + 1]
        q = edge_q[global_k]
        L = rest_length[global_k]

        inv_m0 = particle_inv_mass[global_k]
        inv_m1 = particle_inv_mass[global_k + 1]
        inv_mq = edge_inv_mass[global_k]

        d3 = quat_rotate_e3(q)

        # Constraint violation
        edge_vec = p1 - p0
        gamma = edge_vec / L - d3

        # Transform to local frame and apply stiffness
        gamma_local = wp.quat_rotate_inv(q, gamma)
        gamma_local = wp.vec3(
            gamma_local[0] * stiffness[0],
            gamma_local[1] * stiffness[1],
            gamma_local[2] * stiffness[2],
        )
        gamma = wp.quat_rotate(q, gamma_local)

        # Store RHS
        row_base = local_k * 3
        system_rhs[tile_idx, row_base + 0] = -gamma[0]
        system_rhs[tile_idx, row_base + 1] = -gamma[1]
        system_rhs[tile_idx, row_base + 2] = -gamma[2]

        # Diagonal block
        L_inv = 1.0 / L
        pos_contrib = (inv_m0 + inv_m1) * L_inv * L_inv
        quat_factor = 4.0 * inv_mq

        for i in range(3):
            for j in range(3):
                row = row_base + i
                col = row_base + j

                pos_val = pos_contrib if i == j else 0.0
                quat_val = quat_factor * ((1.0 if i == j else 0.0) - d3[i] * d3[j])
                compliance_val = compliance_factor if i == j else 0.0

                system_matrices[tile_idx, row, col] = pos_val + quat_val + compliance_val

        # Off-diagonal coupling with next constraint in this tile
        if local_k + 1 < num_in_tile:
            L_k1 = rest_length[global_k + 1]
            coupling = -particle_inv_mass[global_k + 1] * L_inv / L_k1

            next_row_base = (local_k + 1) * 3
            for i in range(3):
                system_matrices[tile_idx, row_base + i, next_row_base + i] = coupling
                system_matrices[tile_idx, next_row_base + i, row_base + i] = coupling

    # Pad unused entries with identity
    for i in range(num_in_tile * 3, TILE):
        system_matrices[tile_idx, i, i] = 1.0


@wp.kernel
def cholesky_solve_batched_kernel(
    A: wp.array3d(dtype=float),
    b: wp.array2d(dtype=float),
    # output
    x: wp.array2d(dtype=float),
):
    """Batched Cholesky solve - one tile per thread block."""
    tile_idx = wp.tid()

    a_tile = wp.tile_load(A[tile_idx], shape=(TILE, TILE))
    b_tile = wp.tile_load(b[tile_idx], shape=TILE)

    L = wp.tile_cholesky(a_tile)
    x_tile = wp.tile_cholesky_solve(L, b_tile)

    wp.tile_store(x[tile_idx], x_tile)


@wp.kernel
def zero_vec3_kernel(arr: wp.array(dtype=wp.vec3)):
    """Zero out a vec3 array."""
    tid = wp.tid()
    arr[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def zero_quat_kernel(arr: wp.array(dtype=wp.quat)):
    """Zero out a quaternion array."""
    tid = wp.tid()
    arr[tid] = wp.quat(0.0, 0.0, 0.0, 0.0)


@wp.kernel
def apply_stretch_block_corrections_kernel(
    particle_inv_mass: wp.array(dtype=float),
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    delta_lambdas: wp.array2d(dtype=float),
    num_stretch: int,
    constraints_per_tile: int,
    num_tiles: int,
    # outputs (accumulated via atomics)
    particle_corrections: wp.array(dtype=wp.vec3),
    edge_omega: wp.array(dtype=wp.vec3),  # Angular velocity corrections
):
    """Apply stretch corrections from all tiles using atomic operations."""
    tile_idx = wp.tid()

    constraint_start = tile_idx * constraints_per_tile
    constraint_end = wp.min(constraint_start + constraints_per_tile, num_stretch)
    num_in_tile = constraint_end - constraint_start

    for local_k in range(num_in_tile):
        global_k = constraint_start + local_k

        L = rest_length[global_k]
        q = edge_q[global_k]
        d3 = quat_rotate_e3(q)

        dl = wp.vec3(
            delta_lambdas[tile_idx, local_k * 3 + 0],
            delta_lambdas[tile_idx, local_k * 3 + 1],
            delta_lambdas[tile_idx, local_k * 3 + 2],
        )

        # Particle corrections
        inv_m0 = particle_inv_mass[global_k]
        if inv_m0 > 0.0:
            corr0 = -dl * (inv_m0 / L)
            wp.atomic_add(particle_corrections, global_k, corr0)

        inv_m1 = particle_inv_mass[global_k + 1]
        if inv_m1 > 0.0:
            corr1 = dl * (inv_m1 / L)
            wp.atomic_add(particle_corrections, global_k + 1, corr1)

        # Edge quaternion correction (as angular velocity)
        inv_mq = edge_inv_mass[global_k]
        if inv_mq > 0.0:
            omega = 2.0 * inv_mq * wp.cross(dl, d3)
            wp.atomic_add(edge_omega, global_k, omega)


@wp.kernel
def apply_particle_corrections_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_corrections: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    # output
    particle_q_out: wp.array(dtype=wp.vec3),
):
    """Apply accumulated position corrections to particles."""
    tid = wp.tid()

    if particle_inv_mass[tid] == 0.0:
        particle_q_out[tid] = particle_q[tid]
        return

    particle_q_out[tid] = particle_q[tid] + particle_corrections[tid]


@wp.kernel
def apply_quaternion_omega_kernel(
    edge_q: wp.array(dtype=wp.quat),
    edge_omega: wp.array(dtype=wp.vec3),
    edge_inv_mass: wp.array(dtype=float),
    # output
    edge_q_out: wp.array(dtype=wp.quat),
):
    """Apply accumulated angular velocity corrections to quaternions."""
    tid = wp.tid()

    inv_mass = edge_inv_mass[tid]
    if inv_mass == 0.0:
        edge_q_out[tid] = edge_q[tid]
        return

    q = edge_q[tid]
    omega = edge_omega[tid]

    dq = quat_exp_approx(omega)
    q_new = wp.mul(q, dq)
    q_new = wp.normalize(q_new)

    edge_q_out[tid] = q_new


@wp.kernel
def assemble_bend_block_systems_kernel(
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    rest_darboux: wp.array(dtype=wp.quat),
    compliance_factor: float,
    stiffness: wp.vec3,
    num_bend: int,
    constraints_per_tile: int,
    # outputs
    system_matrices: wp.array3d(dtype=float),
    system_rhs: wp.array2d(dtype=float),
):
    """Assemble block system matrices for bend/twist constraints."""
    tile_idx = wp.tid()

    constraint_start = tile_idx * constraints_per_tile
    constraint_end = wp.min(constraint_start + constraints_per_tile, num_bend)
    num_in_tile = constraint_end - constraint_start

    # Initialize to zero
    for i in range(TILE):
        system_rhs[tile_idx, i] = 0.0
        for j in range(TILE):
            system_matrices[tile_idx, i, j] = 0.0

    # For each bend constraint in this tile
    for local_k in range(num_in_tile):
        global_k = constraint_start + local_k

        q0 = edge_q[global_k]
        q1 = edge_q[global_k + 1]
        rest_d = rest_darboux[global_k]

        inv_m0 = edge_inv_mass[global_k]
        inv_m1 = edge_inv_mass[global_k + 1]

        # Compute Darboux vector
        q0_conj = wp.quat(-q0[0], -q0[1], -q0[2], q0[3])
        omega = wp.mul(q0_conj, q1)

        # Handle quaternion double cover
        omega_plus = wp.vec3(omega[0] + rest_d[0], omega[1] + rest_d[1], omega[2] + rest_d[2])
        omega_minus = wp.vec3(omega[0] - rest_d[0], omega[1] - rest_d[1], omega[2] - rest_d[2])

        norm_plus_sq = wp.dot(omega_plus, omega_plus) + (omega[3] + rest_d[3]) * (omega[3] + rest_d[3])
        norm_minus_sq = wp.dot(omega_minus, omega_minus) + (omega[3] - rest_d[3]) * (omega[3] - rest_d[3])

        if norm_minus_sq > norm_plus_sq:
            kappa = omega_plus
        else:
            kappa = omega_minus

        # Apply stiffness
        kappa = wp.vec3(
            kappa[0] * stiffness[0],
            kappa[1] * stiffness[1],
            kappa[2] * stiffness[2],
        )

        # Store RHS
        row_base = local_k * 3
        system_rhs[tile_idx, row_base + 0] = -kappa[0]
        system_rhs[tile_idx, row_base + 1] = -kappa[1]
        system_rhs[tile_idx, row_base + 2] = -kappa[2]

        # Diagonal block
        diag = 0.25 * (inv_m0 + inv_m1) + compliance_factor
        for i in range(3):
            system_matrices[tile_idx, row_base + i, row_base + i] = diag

        # Off-diagonal coupling
        if local_k + 1 < num_in_tile:
            coupling = -0.25 * edge_inv_mass[global_k + 1]
            next_row_base = (local_k + 1) * 3
            for i in range(3):
                system_matrices[tile_idx, row_base + i, next_row_base + i] = coupling
                system_matrices[tile_idx, next_row_base + i, row_base + i] = coupling

    # Pad unused entries with identity
    for i in range(num_in_tile * 3, TILE):
        system_matrices[tile_idx, i, i] = 1.0


@wp.kernel
def apply_bend_block_corrections_kernel(
    edge_inv_mass: wp.array(dtype=float),
    delta_lambdas: wp.array2d(dtype=float),
    num_bend: int,
    constraints_per_tile: int,
    num_tiles: int,
    # output (accumulated via atomics)
    edge_omega: wp.array(dtype=wp.vec3),
):
    """Apply bend corrections from all tiles using atomic operations."""
    tile_idx = wp.tid()

    constraint_start = tile_idx * constraints_per_tile
    constraint_end = wp.min(constraint_start + constraints_per_tile, num_bend)
    num_in_tile = constraint_end - constraint_start

    for local_k in range(num_in_tile):
        global_k = constraint_start + local_k

        dl = wp.vec3(
            delta_lambdas[tile_idx, local_k * 3 + 0],
            delta_lambdas[tile_idx, local_k * 3 + 1],
            delta_lambdas[tile_idx, local_k * 3 + 2],
        )

        # Quaternion k (q0 role)
        inv_m0 = edge_inv_mass[global_k]
        if inv_m0 > 0.0:
            omega0 = -0.5 * inv_m0 * dl
            wp.atomic_add(edge_omega, global_k, omega0)

        # Quaternion k+1 (q1 role)
        inv_m1 = edge_inv_mass[global_k + 1]
        if inv_m1 > 0.0:
            omega1 = 0.5 * inv_m1 * dl
            wp.atomic_add(edge_omega, global_k + 1, omega1)


@wp.kernel
def normalize_quaternions_kernel(
    edge_q: wp.array(dtype=wp.quat),
    # output
    edge_q_out: wp.array(dtype=wp.quat),
):
    """Normalize quaternions to maintain unit length."""
    tid = wp.tid()
    edge_q_out[tid] = wp.normalize(edge_q[tid])


@wp.kernel
def update_velocities_kernel(
    particle_q_old: wp.array(dtype=wp.vec3),
    particle_q_new: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    dt: float,
    # output
    particle_qd: wp.array(dtype=wp.vec3),
):
    """Update velocities from position change."""
    tid = wp.tid()

    if particle_inv_mass[tid] == 0.0:
        particle_qd[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    delta_x = particle_q_new[tid] - particle_q_old[tid]
    particle_qd[tid] = delta_x / dt


@wp.kernel
def solve_ground_collision_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    ground_level: float,
    # output
    particle_q_out: wp.array(dtype=wp.vec3),
):
    """Simple ground plane collision constraint."""
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
def update_rest_darboux_kernel(
    rest_bend_d1: float,
    rest_bend_d2: float,
    rest_twist: float,
    num_bend: int,
    # output
    rest_darboux: wp.array(dtype=wp.quat),
):
    """Update rest Darboux vectors to define the rod's rest shape."""
    tid = wp.tid()
    if tid >= num_bend:
        return

    half_bend_d1 = rest_bend_d1 * 0.5
    half_bend_d2 = rest_bend_d2 * 0.5
    half_twist = rest_twist * 0.5

    angle_sq = half_bend_d1 * half_bend_d1 + half_bend_d2 * half_bend_d2 + half_twist * half_twist
    angle = wp.sqrt(angle_sq)

    if angle < 1.0e-8:
        rest_darboux[tid] = wp.quat(0.0, 0.0, 0.0, 1.0)
    else:
        s = wp.sin(angle) / angle
        c = wp.cos(angle)
        rest_darboux[tid] = wp.quat(s * half_bend_d1, s * half_bend_d2, s * half_twist, c)


@wp.kernel
def compute_director_lines_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    edge_q: wp.array(dtype=wp.quat),
    num_edges: int,
    axis_length: float,
    # outputs
    line_starts: wp.array(dtype=wp.vec3),
    line_ends: wp.array(dtype=wp.vec3),
    line_colors: wp.array(dtype=wp.vec3),
):
    """Compute line segments for visualizing material frames."""
    tid = wp.tid()
    edge_idx = tid // 3
    axis_idx = tid % 3

    if edge_idx >= num_edges:
        return

    p0 = particle_q[edge_idx]
    p1 = particle_q[edge_idx + 1]
    midpoint = (p0 + p1) * 0.5

    q = edge_q[edge_idx]

    if axis_idx == 0:
        x, y, z, w = q[0], q[1], q[2], q[3]
        d1_x = w * w + x * x - y * y - z * z
        d1_y = 2.0 * (x * y + w * z)
        d1_z = 2.0 * (x * z - w * y)
        director = wp.vec3(d1_x, d1_y, d1_z)
        color = wp.vec3(1.0, 0.0, 0.0)
    elif axis_idx == 1:
        x, y, z, w = q[0], q[1], q[2], q[3]
        d2_x = 2.0 * (x * y - w * z)
        d2_y = w * w - x * x + y * y - z * z
        d2_z = 2.0 * (y * z + w * x)
        director = wp.vec3(d2_x, d2_y, d2_z)
        color = wp.vec3(0.0, 1.0, 0.0)
    else:
        x, y, z, w = q[0], q[1], q[2], q[3]
        d3_x = 2.0 * (x * z + w * y)
        d3_y = 2.0 * (y * z - w * x)
        d3_z = w * w - x * x - y * y + z * z
        director = wp.vec3(d3_x, d3_y, d3_z)
        color = wp.vec3(0.0, 0.0, 1.0)

    line_starts[tid] = midpoint
    line_ends[tid] = midpoint + director * axis_length
    line_colors[tid] = color


class Example:
    def __init__(self, viewer, args=None):
        # Simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.constraint_iterations = 4  # Fewer iterations needed with Cholesky

        self.viewer = viewer
        self.args = args

        # Rod parameters
        self.num_particles = NUM_PARTICLES
        self.num_stretch = NUM_STRETCH
        self.num_bend = NUM_BEND
        self.num_tiles = NUM_TILES
        self.constraints_per_tile = CONSTRAINTS_PER_TILE

        particle_spacing = 0.05
        particle_mass = 0.1
        particle_radius = 0.015
        edge_mass = 0.01
        start_height = 5.0

        # Stiffness parameters
        self.stretch_stiffness = 1.0
        self.shear_stiffness = 1.0
        self.bend_stiffness = 0.1
        self.twist_stiffness = 0.1

        # Compliance for regularization
        self.compliance = 1.0e-6
        self.compliance_factor = self.compliance / (self.sim_dt * self.sim_dt)

        # Rest shape parameters
        self.rest_bend_d1 = 0.0
        self.rest_bend_d2 = 0.0
        self.rest_twist = 0.0

        self.gravity = wp.vec3(0.0, 0.0, -9.81)

        # Build the model
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        for i in range(self.num_particles):
            mass = 0.0 if i == 0 else particle_mass
            builder.add_particle(
                pos=(i * particle_spacing, 0.0, start_height),
                vel=(0.0, 0.0, 0.0),
                mass=mass,
                radius=particle_radius,
            )

        self.model = builder.finalize()

        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 1.0e1
        self.model.soft_contact_mu = 0.5

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, self.args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        device = self.model.device

        # Particle inverse mass array
        inv_mass_np = [0.0] + [1.0 / particle_mass] * (self.num_particles - 1)
        self.particle_inv_mass = wp.array(inv_mass_np, dtype=float, device=device)

        # Edge quaternions
        angle = math.pi / 2.0
        q_init = wp.quat(0.0, math.sin(angle / 2.0), 0.0, math.cos(angle / 2.0))
        edge_q_init = [q_init] * self.num_stretch
        self.edge_q = wp.array(edge_q_init, dtype=wp.quat, device=device)
        self.edge_q_new = wp.array(edge_q_init, dtype=wp.quat, device=device)

        # Edge inverse masses
        edge_inv_mass_np = [1.0 / edge_mass] * self.num_stretch
        self.edge_inv_mass = wp.array(edge_inv_mass_np, dtype=float, device=device)

        # Rest lengths
        rest_length_np = [particle_spacing] * self.num_stretch
        self.rest_length = wp.array(rest_length_np, dtype=float, device=device)

        # Rest Darboux vectors
        rest_darboux_np = [wp.quat(0.0, 0.0, 0.0, 1.0)] * self.num_bend
        self.rest_darboux = wp.array(rest_darboux_np, dtype=wp.quat, device=device)

        # Temporary buffers
        self.particle_q_predicted = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)
        self.particle_q_temp = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)

        # Per-tile system matrices and vectors for Cholesky solve
        self.stretch_matrices = wp.zeros((self.num_tiles, TILE, TILE), dtype=float, device=device)
        self.stretch_rhs = wp.zeros((self.num_tiles, TILE), dtype=float, device=device)
        self.stretch_lambdas = wp.zeros((self.num_tiles, TILE), dtype=float, device=device)

        self.bend_matrices = wp.zeros((self.num_tiles, TILE, TILE), dtype=float, device=device)
        self.bend_rhs = wp.zeros((self.num_tiles, TILE), dtype=float, device=device)
        self.bend_lambdas = wp.zeros((self.num_tiles, TILE), dtype=float, device=device)

        # Correction accumulators (for atomic adds)
        self.particle_corrections = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)
        self.edge_omega = wp.zeros(self.num_stretch, dtype=wp.vec3, device=device)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        # Director visualization buffers
        num_director_lines = self.num_stretch * 3
        self.director_line_starts = wp.zeros(num_director_lines, dtype=wp.vec3, device=device)
        self.director_line_ends = wp.zeros(num_director_lines, dtype=wp.vec3, device=device)
        self.director_line_colors = wp.zeros(num_director_lines, dtype=wp.vec3, device=device)
        self.show_directors = True
        self.director_scale = 0.03

        self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            wp.copy(self.particle_q_temp, self.state_0.particle_q)

            stretch_ks = wp.vec3(self.shear_stiffness, self.shear_stiffness, self.stretch_stiffness)
            bend_ks = wp.vec3(self.bend_stiffness, self.twist_stiffness, self.bend_stiffness)

            # Step 1: Integrate
            wp.launch(
                kernel=integrate_particles_kernel,
                dim=self.num_particles,
                inputs=[
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                    self.particle_inv_mass,
                    self.gravity,
                    self.sim_dt,
                ],
                outputs=[self.particle_q_predicted, self.state_1.particle_qd],
                device=self.model.device,
            )
            wp.copy(self.state_1.particle_q, self.particle_q_predicted)

            # Step 2: Block-Jacobi constraint solving with Cholesky
            for _ in range(self.constraint_iterations):
                # Solve stretch/shear constraints
                wp.launch(
                    kernel=assemble_stretch_block_systems_kernel,
                    dim=self.num_tiles,
                    inputs=[
                        self.state_1.particle_q,
                        self.particle_inv_mass,
                        self.edge_q,
                        self.edge_inv_mass,
                        self.rest_length,
                        self.compliance_factor,
                        stretch_ks,
                        self.num_stretch,
                        self.constraints_per_tile,
                    ],
                    outputs=[self.stretch_matrices, self.stretch_rhs],
                    device=self.model.device,
                )

                wp.launch_tiled(
                    kernel=cholesky_solve_batched_kernel,
                    dim=[self.num_tiles, 1],
                    inputs=[self.stretch_matrices, self.stretch_rhs],
                    outputs=[self.stretch_lambdas],
                    block_dim=BLOCK_DIM,
                    device=self.model.device,
                )

                # Zero correction accumulators
                wp.launch(
                    kernel=zero_vec3_kernel,
                    dim=self.num_particles,
                    inputs=[self.particle_corrections],
                    device=self.model.device,
                )
                wp.launch(
                    kernel=zero_vec3_kernel,
                    dim=self.num_stretch,
                    inputs=[self.edge_omega],
                    device=self.model.device,
                )

                wp.launch(
                    kernel=apply_stretch_block_corrections_kernel,
                    dim=self.num_tiles,
                    inputs=[
                        self.particle_inv_mass,
                        self.edge_q,
                        self.edge_inv_mass,
                        self.rest_length,
                        self.stretch_lambdas,
                        self.num_stretch,
                        self.constraints_per_tile,
                        self.num_tiles,
                    ],
                    outputs=[self.particle_corrections, self.edge_omega],
                    device=self.model.device,
                )

                wp.launch(
                    kernel=apply_particle_corrections_kernel,
                    dim=self.num_particles,
                    inputs=[
                        self.state_1.particle_q,
                        self.particle_corrections,
                        self.particle_inv_mass,
                    ],
                    outputs=[self.particle_q_predicted],
                    device=self.model.device,
                )
                wp.copy(self.state_1.particle_q, self.particle_q_predicted)

                wp.launch(
                    kernel=apply_quaternion_omega_kernel,
                    dim=self.num_stretch,
                    inputs=[
                        self.edge_q,
                        self.edge_omega,
                        self.edge_inv_mass,
                    ],
                    outputs=[self.edge_q_new],
                    device=self.model.device,
                )
                self.edge_q, self.edge_q_new = self.edge_q_new, self.edge_q

                # Solve bend/twist constraints
                if self.num_bend > 0:
                    wp.launch(
                        kernel=assemble_bend_block_systems_kernel,
                        dim=self.num_tiles,
                        inputs=[
                            self.edge_q,
                            self.edge_inv_mass,
                            self.rest_darboux,
                            self.compliance_factor,
                            bend_ks,
                            self.num_bend,
                            self.constraints_per_tile,
                        ],
                        outputs=[self.bend_matrices, self.bend_rhs],
                        device=self.model.device,
                    )

                    wp.launch_tiled(
                        kernel=cholesky_solve_batched_kernel,
                        dim=[self.num_tiles, 1],
                        inputs=[self.bend_matrices, self.bend_rhs],
                        outputs=[self.bend_lambdas],
                        block_dim=BLOCK_DIM,
                        device=self.model.device,
                    )

                    # Zero omega accumulator
                    wp.launch(
                        kernel=zero_vec3_kernel,
                        dim=self.num_stretch,
                        inputs=[self.edge_omega],
                        device=self.model.device,
                    )

                    wp.launch(
                        kernel=apply_bend_block_corrections_kernel,
                        dim=self.num_tiles,
                        inputs=[
                            self.edge_inv_mass,
                            self.bend_lambdas,
                            self.num_bend,
                            self.constraints_per_tile,
                            self.num_tiles,
                        ],
                        outputs=[self.edge_omega],
                        device=self.model.device,
                    )

                    wp.launch(
                        kernel=apply_quaternion_omega_kernel,
                        dim=self.num_stretch,
                        inputs=[
                            self.edge_q,
                            self.edge_omega,
                            self.edge_inv_mass,
                        ],
                        outputs=[self.edge_q_new],
                        device=self.model.device,
                    )
                    self.edge_q, self.edge_q_new = self.edge_q_new, self.edge_q

                # Normalize quaternions
                wp.launch(
                    kernel=normalize_quaternions_kernel,
                    dim=self.num_stretch,
                    inputs=[self.edge_q],
                    outputs=[self.edge_q_new],
                    device=self.model.device,
                )
                self.edge_q, self.edge_q_new = self.edge_q_new, self.edge_q

            # Step 3: Ground collision
            wp.launch(
                kernel=solve_ground_collision_kernel,
                dim=self.num_particles,
                inputs=[
                    self.state_1.particle_q,
                    self.particle_inv_mass,
                    self.model.particle_radius,
                    0.0,
                ],
                outputs=[self.particle_q_predicted],
                device=self.model.device,
            )
            wp.copy(self.state_1.particle_q, self.particle_q_predicted)

            # Step 4: Update velocities
            wp.launch(
                kernel=update_velocities_kernel,
                dim=self.num_particles,
                inputs=[
                    self.particle_q_temp,
                    self.state_1.particle_q,
                    self.particle_inv_mass,
                    self.sim_dt,
                ],
                outputs=[self.state_1.particle_qd],
                device=self.model.device,
            )

            self.state_0, self.state_1 = self.state_1, self.state_0
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        newton.examples.test_particle_state(
            self.state_0,
            "anchor particle is stationary",
            lambda q, qd: wp.length(qd) < 1e-6,
            indices=[0],
        )

        newton.examples.test_particle_state(
            self.state_0,
            "particles are above the ground",
            lambda q, qd: q[2] >= -0.01,
        )

        p_lower = wp.vec3(-3.0, -5.0, -0.1)
        p_upper = wp.vec3(10.0, 5.0, 7.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within reasonable bounds",
            lambda q, qd: newton.utils.vec_inside_limits(q, p_lower, p_upper),
        )

        edge_q_np = self.edge_q.numpy()
        for i, q in enumerate(edge_q_np):
            norm = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2) ** 0.5
            assert abs(norm - 1.0) < 0.1, f"Edge quaternion {i} not normalized: norm={norm}"

    def _update_rest_darboux(self):
        wp.launch(
            kernel=update_rest_darboux_kernel,
            dim=self.num_bend,
            inputs=[
                self.rest_bend_d1,
                self.rest_bend_d2,
                self.rest_twist,
                self.num_bend,
            ],
            outputs=[self.rest_darboux],
            device=self.model.device,
        )

    def gui(self, ui):
        ui.text("Cholesky Cosserat Rod (Multi-Tile)")
        ui.text(f"Particles: {self.num_particles}, Tiles: {self.num_tiles}")
        _changed, self.stretch_stiffness = ui.slider_float("Stretch Stiffness", self.stretch_stiffness, 0.0, 1.0)
        _changed, self.shear_stiffness = ui.slider_float("Shear Stiffness", self.shear_stiffness, 0.0, 1.0)
        _changed, self.bend_stiffness = ui.slider_float("Bend Stiffness", self.bend_stiffness, 0.0, 1.0)
        _changed, self.twist_stiffness = ui.slider_float("Twist Stiffness", self.twist_stiffness, 0.0, 1.0)
        ui.separator()
        ui.text("Rest Shape (Darboux Vector)")
        changed_d1, self.rest_bend_d1 = ui.slider_float("Rest Bend d1", self.rest_bend_d1, -0.5, 0.5)
        changed_d2, self.rest_bend_d2 = ui.slider_float("Rest Bend d2", self.rest_bend_d2, -0.5, 0.5)
        changed_twist, self.rest_twist = ui.slider_float("Rest Twist", self.rest_twist, -0.5, 0.5)
        if changed_d1 or changed_d2 or changed_twist:
            self._update_rest_darboux()
        ui.separator()
        ui.text("Visualization")
        _changed, self.show_directors = ui.checkbox("Show Directors", self.show_directors)
        _changed, self.director_scale = ui.slider_float("Director Scale", self.director_scale, 0.01, 0.1)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)

        if self.show_directors:
            wp.launch(
                kernel=compute_director_lines_kernel,
                dim=self.num_stretch * 3,
                inputs=[
                    self.state_0.particle_q,
                    self.edge_q,
                    self.num_stretch,
                    self.director_scale,
                ],
                outputs=[
                    self.director_line_starts,
                    self.director_line_ends,
                    self.director_line_colors,
                ],
                device=self.model.device,
            )
            self.viewer.log_lines(
                "/directors",
                self.director_line_starts,
                self.director_line_ends,
                self.director_line_colors,
            )
        else:
            self.viewer.log_lines("/directors", None, None, None)

        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()

    if isinstance(viewer, newton.viewer.ViewerGL):
        viewer.show_particles = True

    example = Example(viewer, args)

    newton.examples.run(example, args)
