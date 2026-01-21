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
# Example Global Cosserat Rod with Cholesky Solve - Multi-Tile
#
# Demonstrates a Block-Jacobi approach for TRUE global matrix-based solving
# of Position And Orientation Based Cosserat Rods on longer particle chains.
# The rod is partitioned into multiple 32x32 tiles that are solved in
# parallel using Warp's tile Cholesky API.
#
# This combines:
#   - True global Cholesky solving from example 07
#   - Multi-tile Block-Jacobi partitioning from example 01
#
# Unlike the iterative Jacobi approach in examples 03/04 which uses atomic
# operations, this example:
#   1. Assembles block system matrices A_i = J_i M^{-1} J_i^T per tile
#   2. Solves each block via Cholesky in parallel
#   3. Applies corrections (with atomics only at tile boundaries)
#   4. Iterates for inter-tile coupling
#
# The banded structure (tridiagonal in constraint space) enables efficient
# O(n) solving via tile-parallel Cholesky factorization.
#
# For 129 particles with 128 stretch + 127 bend constraints:
#   - Stretch tiles: 4 tiles of 32 constraints each
#   - Bend tiles: 4 tiles handling up to 32 constraints each
#
# Command: uv run -m newton.examples cosserat_08_global_cosserat_rod_cholesky_multitile
#
###########################################################################

import math

import warp as wp

import newton
import newton.examples

# Warp tile configuration
BLOCK_DIM = 128
TILE = 32  # 32x32 tile size for Cholesky

# Rod configuration - sized for multi-tile solving
NUM_TILES = 4
CONSTRAINTS_PER_TILE = 32
NUM_PARTICLES = NUM_TILES * TILE + 1  # 129 particles
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
def quat_e3_bar(q: wp.quat) -> wp.quat:
    """Compute q * e3_bar where e3_bar is the conjugate of quaternion (0,0,1,0)."""
    return wp.quat(-q[1], q[0], -q[3], q[2])


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
def compute_stretch_constraint_data_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    edge_q: wp.array(dtype=wp.quat),
    rest_length: wp.array(dtype=float),
    num_stretch: int,
    # outputs
    constraint_violation: wp.array(dtype=float),
    constraint_direction: wp.array(dtype=wp.vec3),
    constraint_quat_direction: wp.array(dtype=wp.quat),
):
    """Compute stretch/shear constraint violations and directions."""
    tid = wp.tid()
    if tid >= num_stretch:
        return

    p0 = particle_q[tid]
    p1 = particle_q[tid + 1]
    q0 = edge_q[tid]
    L = rest_length[tid]

    d3 = quat_rotate_e3(q0)
    edge_vec = p1 - p0
    gamma = edge_vec / L - d3

    gamma_mag = wp.length(gamma)

    if gamma_mag > 1.0e-8:
        constraint_direction[tid] = gamma / gamma_mag
    else:
        constraint_direction[tid] = wp.vec3(1.0, 0.0, 0.0)

    constraint_violation[tid] = gamma_mag
    constraint_quat_direction[tid] = quat_e3_bar(q0)


@wp.kernel
def assemble_stretch_block_systems_kernel(
    particle_inv_mass: wp.array(dtype=float),
    edge_inv_mass: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    constraint_direction: wp.array(dtype=wp.vec3),
    constraint_violation: wp.array(dtype=float),
    compliance_factor: float,
    num_stretch: int,
    constraints_per_tile: int,
    # outputs (per-tile arrays)
    system_matrices: wp.array3d(dtype=float),  # (NUM_TILES, TILE, TILE)
    system_rhs: wp.array2d(dtype=float),  # (NUM_TILES, TILE)
):
    """
    Assemble block system matrices for each tile of stretch constraints.

    Each tile handles a contiguous block of constraints.
    A = J M^{-1} J^T + compliance*I (banded/tridiagonal within each tile)
    """
    tile_idx = wp.tid()

    constraint_start = tile_idx * constraints_per_tile
    constraint_end = wp.min(constraint_start + constraints_per_tile, num_stretch)
    num_in_tile = constraint_end - constraint_start

    # Initialize tile matrix
    for i in range(TILE):
        system_rhs[tile_idx, i] = 0.0
        for j in range(TILE):
            system_matrices[tile_idx, i, j] = 0.0

    # Fill tridiagonal structure for this tile
    for local_k in range(num_in_tile):
        global_k = constraint_start + local_k

        w_p0 = particle_inv_mass[global_k]
        w_p1 = particle_inv_mass[global_k + 1]
        w_q0 = edge_inv_mass[global_k]
        L = rest_length[global_k]

        # Diagonal: position contribution + quaternion contribution + compliance
        pos_diag = (w_p0 + w_p1) / (L * L)
        quat_diag = 4.0 * L * L * w_q0
        diag = pos_diag + quat_diag + compliance_factor
        system_matrices[tile_idx, local_k, local_k] = diag

        # RHS
        system_rhs[tile_idx, local_k] = -constraint_violation[global_k]

        # Off-diagonal coupling (within tile only)
        if local_k + 1 < num_in_tile:
            n_k = constraint_direction[global_k]
            n_k1 = constraint_direction[global_k + 1]
            L_k1 = rest_length[global_k + 1]

            coupling = -particle_inv_mass[global_k + 1] * wp.dot(n_k, n_k1) / (L * L_k1)
            system_matrices[tile_idx, local_k, local_k + 1] = coupling
            system_matrices[tile_idx, local_k + 1, local_k] = coupling

    # Pad unused rows with identity
    for i in range(num_in_tile, TILE):
        system_matrices[tile_idx, i, i] = 1.0


@wp.kernel
def cholesky_solve_batched_kernel(
    A: wp.array3d(dtype=float),  # (NUM_TILES, TILE, TILE)
    b: wp.array2d(dtype=float),  # (NUM_TILES, TILE)
    # output
    x: wp.array2d(dtype=float),  # (NUM_TILES, TILE)
):
    """Batched Cholesky solve - one tile per thread block."""
    tile_idx = wp.tid()

    a_tile = wp.tile_load(A[tile_idx], shape=(TILE, TILE))
    b_tile = wp.tile_load(b[tile_idx], shape=TILE)

    L = wp.tile_cholesky(a_tile)
    x_tile = wp.tile_cholesky_solve(L, b_tile)

    wp.tile_store(x[tile_idx], x_tile)


@wp.kernel
def zero_corrections_vec3_kernel(
    corrections: wp.array(dtype=wp.vec3),
):
    """Zero out vec3 corrections buffer."""
    tid = wp.tid()
    corrections[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def zero_corrections_quat_kernel(
    corrections: wp.array(dtype=wp.quat),
):
    """Zero out quat corrections buffer."""
    tid = wp.tid()
    corrections[tid] = wp.quat(0.0, 0.0, 0.0, 0.0)


@wp.kernel
def apply_stretch_block_corrections_kernel(
    particle_inv_mass: wp.array(dtype=float),
    edge_inv_mass: wp.array(dtype=float),
    edge_q: wp.array(dtype=wp.quat),
    rest_length: wp.array(dtype=float),
    constraint_direction: wp.array(dtype=wp.vec3),
    constraint_quat_direction: wp.array(dtype=wp.quat),
    delta_lambdas: wp.array2d(dtype=float),  # (NUM_TILES, TILE)
    num_stretch: int,
    constraints_per_tile: int,
    num_tiles: int,
    # outputs (accumulated via atomics)
    particle_corrections: wp.array(dtype=wp.vec3),
    edge_q_corrections: wp.array(dtype=wp.quat),
):
    """
    Apply position and quaternion corrections from all tiles.
    Uses atomic operations for particles/quaternions at tile boundaries.
    """
    tile_idx = wp.tid()

    constraint_start = tile_idx * constraints_per_tile
    constraint_end = wp.min(constraint_start + constraints_per_tile, num_stretch)
    num_in_tile = constraint_end - constraint_start

    for local_k in range(num_in_tile):
        global_k = constraint_start + local_k
        dl = delta_lambdas[tile_idx, local_k]
        n = constraint_direction[global_k]
        L = rest_length[global_k]

        # Position corrections
        w_p0 = particle_inv_mass[global_k]
        if w_p0 > 0.0:
            corr0 = -n * (dl * w_p0 / L)
            wp.atomic_add(particle_corrections, global_k, corr0)

        w_p1 = particle_inv_mass[global_k + 1]
        if w_p1 > 0.0:
            corr1 = n * (dl * w_p1 / L)
            wp.atomic_add(particle_corrections, global_k + 1, corr1)

        # Quaternion correction
        w_q0 = edge_inv_mass[global_k]
        if w_q0 > 0.0:
            q_e3_bar = constraint_quat_direction[global_k]
            gamma_quat = wp.quat(n[0], n[1], n[2], 0.0)
            corrq_raw = wp.mul(gamma_quat, q_e3_bar)

            scale = 2.0 * w_q0 * L * dl
            corrq = wp.quat(
                corrq_raw[0] * scale,
                corrq_raw[1] * scale,
                corrq_raw[2] * scale,
                corrq_raw[3] * scale,
            )
            wp.atomic_add(edge_q_corrections, global_k, corrq)


@wp.kernel
def apply_particle_corrections_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_corrections: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    # output
    particle_q_out: wp.array(dtype=wp.vec3),
):
    """Apply accumulated corrections to particle positions."""
    tid = wp.tid()

    if particle_inv_mass[tid] == 0.0:
        particle_q_out[tid] = particle_q[tid]
        return

    particle_q_out[tid] = particle_q[tid] + particle_corrections[tid]


@wp.kernel
def apply_quaternion_corrections_kernel(
    edge_q: wp.array(dtype=wp.quat),
    edge_q_corrections: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    # output
    edge_q_out: wp.array(dtype=wp.quat),
):
    """Apply accumulated corrections and normalize quaternions."""
    tid = wp.tid()

    if edge_inv_mass[tid] == 0.0:
        edge_q_out[tid] = edge_q[tid]
        return

    q = edge_q[tid]
    dq = edge_q_corrections[tid]
    q_new = wp.quat(q[0] + dq[0], q[1] + dq[1], q[2] + dq[2], q[3] + dq[3])
    q_new = wp.normalize(q_new)
    edge_q_out[tid] = q_new


@wp.kernel
def compute_bend_constraint_data_kernel(
    edge_q: wp.array(dtype=wp.quat),
    rest_darboux: wp.array(dtype=wp.quat),
    num_bend: int,
    # outputs
    bend_violation: wp.array(dtype=float),
    bend_direction: wp.array(dtype=wp.vec3),
):
    """Compute bend/twist constraint violations and directions."""
    tid = wp.tid()
    if tid >= num_bend:
        return

    q0 = edge_q[tid]
    q1 = edge_q[tid + 1]
    rest_q = rest_darboux[tid]

    q0_conj = wp.quat(-q0[0], -q0[1], -q0[2], q0[3])
    omega = wp.mul(q0_conj, q1)

    omega_plus = wp.vec3(omega[0] + rest_q[0], omega[1] + rest_q[1], omega[2] + rest_q[2])
    omega_minus = wp.vec3(omega[0] - rest_q[0], omega[1] - rest_q[1], omega[2] - rest_q[2])

    norm_plus = wp.dot(omega_plus, omega_plus)
    norm_minus = wp.dot(omega_minus, omega_minus)

    if norm_minus > norm_plus:
        omega_vec = omega_plus
    else:
        omega_vec = omega_minus

    omega_mag = wp.length(omega_vec)

    if omega_mag > 1.0e-8:
        bend_direction[tid] = omega_vec / omega_mag
    else:
        bend_direction[tid] = wp.vec3(1.0, 0.0, 0.0)

    bend_violation[tid] = omega_mag


@wp.kernel
def assemble_bend_block_systems_kernel(
    edge_inv_mass: wp.array(dtype=float),
    bend_direction: wp.array(dtype=wp.vec3),
    bend_violation: wp.array(dtype=float),
    compliance_factor: float,
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

    # Initialize
    for i in range(TILE):
        system_rhs[tile_idx, i] = 0.0
        for j in range(TILE):
            system_matrices[tile_idx, i, j] = 0.0

    for local_k in range(num_in_tile):
        global_k = constraint_start + local_k

        w_q0 = edge_inv_mass[global_k]
        w_q1 = edge_inv_mass[global_k + 1]

        diag = w_q0 + w_q1 + compliance_factor
        system_matrices[tile_idx, local_k, local_k] = diag

        system_rhs[tile_idx, local_k] = -bend_violation[global_k]

        if local_k + 1 < num_in_tile:
            n_k = bend_direction[global_k]
            n_k1 = bend_direction[global_k + 1]

            coupling = -edge_inv_mass[global_k + 1] * wp.dot(n_k, n_k1)
            system_matrices[tile_idx, local_k, local_k + 1] = coupling
            system_matrices[tile_idx, local_k + 1, local_k] = coupling

    for i in range(num_in_tile, TILE):
        system_matrices[tile_idx, i, i] = 1.0


@wp.kernel
def apply_bend_block_corrections_kernel(
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    bend_direction: wp.array(dtype=wp.vec3),
    delta_lambdas: wp.array2d(dtype=float),
    num_bend: int,
    constraints_per_tile: int,
    num_tiles: int,
    # output
    edge_q_corrections: wp.array(dtype=wp.quat),
):
    """Apply bend/twist corrections from all tiles."""
    tile_idx = wp.tid()

    constraint_start = tile_idx * constraints_per_tile
    constraint_end = wp.min(constraint_start + constraints_per_tile, num_bend)
    num_in_tile = constraint_end - constraint_start

    for local_k in range(num_in_tile):
        global_k = constraint_start + local_k
        dl = delta_lambdas[tile_idx, local_k]
        n = bend_direction[global_k]

        # Correction for q0 (from q1 side)
        q1 = edge_q[global_k + 1]
        w_q0 = edge_inv_mass[global_k]
        if w_q0 > 0.0:
            omega_q = wp.quat(n[0], n[1], n[2], 0.0)
            corrq0_raw = wp.mul(q1, omega_q)
            scale = w_q0 * dl
            corrq0 = wp.quat(
                corrq0_raw[0] * scale,
                corrq0_raw[1] * scale,
                corrq0_raw[2] * scale,
                corrq0_raw[3] * scale,
            )
            wp.atomic_add(edge_q_corrections, global_k, corrq0)

        # Correction for q1 (from q0 side)
        q0 = edge_q[global_k]
        w_q1 = edge_inv_mass[global_k + 1]
        if w_q1 > 0.0:
            omega_q = wp.quat(n[0], n[1], n[2], 0.0)
            corrq1_raw = wp.mul(q0, omega_q)
            scale = -w_q1 * dl
            corrq1 = wp.quat(
                corrq1_raw[0] * scale,
                corrq1_raw[1] * scale,
                corrq1_raw[2] * scale,
                corrq1_raw[3] * scale,
            )
            wp.atomic_add(edge_q_corrections, global_k + 1, corrq1)


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
def update_rest_darboux_kernel(
    rest_bend_d1: float,
    rest_bend_d2: float,
    rest_twist: float,
    num_bend: int,
    # output
    rest_darboux: wp.array(dtype=wp.quat),
):
    """Update rest Darboux vectors for rod's rest shape."""
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
    x, y, z, w = q[0], q[1], q[2], q[3]

    if axis_idx == 0:
        d1_x = w * w + x * x - y * y - z * z
        d1_y = 2.0 * (x * y + w * z)
        d1_z = 2.0 * (x * z - w * y)
        director = wp.vec3(d1_x, d1_y, d1_z)
        color = wp.vec3(1.0, 0.0, 0.0)
    elif axis_idx == 1:
        d2_x = 2.0 * (x * y - w * z)
        d2_y = w * w - x * x + y * y - z * z
        d2_z = 2.0 * (y * z + w * x)
        director = wp.vec3(d2_x, d2_y, d2_z)
        color = wp.vec3(0.0, 1.0, 0.0)
    else:
        d3_x = 2.0 * (x * z + w * y)
        d3_y = 2.0 * (y * z - w * x)
        d3_z = w * w - x * x - y * y + z * z
        director = wp.vec3(d3_x, d3_y, d3_z)
        color = wp.vec3(0.0, 0.0, 1.0)

    line_starts[tid] = midpoint
    line_ends[tid] = midpoint + director * axis_length
    line_colors[tid] = color


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


class Example:
    def __init__(self, viewer, args=None):
        # Simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps
        # With global Cholesky + block-Jacobi, we need a few iterations for tile coupling
        self.constraint_iterations = 3

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

        # Compliance parameters
        self.stretch_compliance = 1.0e-6
        self.bend_compliance = 1.0e-5

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

        # Inverse mass arrays
        inv_mass_np = [0.0] + [1.0 / particle_mass] * (self.num_particles - 1)
        self.particle_inv_mass = wp.array(inv_mass_np, dtype=float, device=device)

        # Edge quaternions
        angle = math.pi / 2.0
        q_init = wp.quat(0.0, math.sin(angle / 2.0), 0.0, math.cos(angle / 2.0))
        edge_q_init = [q_init] * self.num_stretch
        self.edge_q = wp.array(edge_q_init, dtype=wp.quat, device=device)
        self.edge_q_new = wp.array(edge_q_init, dtype=wp.quat, device=device)

        edge_inv_mass_np = [1.0 / edge_mass] * self.num_stretch
        self.edge_inv_mass = wp.array(edge_inv_mass_np, dtype=float, device=device)

        rest_length_np = [particle_spacing] * self.num_stretch
        self.rest_length = wp.array(rest_length_np, dtype=float, device=device)

        rest_darboux_np = [wp.quat(0.0, 0.0, 0.0, 1.0)] * self.num_bend
        self.rest_darboux = wp.array(rest_darboux_np, dtype=wp.quat, device=device)

        # Temporary buffers
        self.particle_q_predicted = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)
        self.particle_q_temp = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)

        # Stretch constraint data
        self.stretch_violation = wp.zeros(self.num_stretch, dtype=float, device=device)
        self.stretch_direction = wp.zeros(self.num_stretch, dtype=wp.vec3, device=device)
        self.stretch_quat_direction = wp.zeros(self.num_stretch, dtype=wp.quat, device=device)

        # Bend constraint data
        self.bend_violation = wp.zeros(max(1, self.num_bend), dtype=float, device=device)
        self.bend_direction = wp.zeros(max(1, self.num_bend), dtype=wp.vec3, device=device)

        # Per-tile system matrices
        self.stretch_matrices = wp.zeros((self.num_tiles, TILE, TILE), dtype=float, device=device)
        self.stretch_rhs = wp.zeros((self.num_tiles, TILE), dtype=float, device=device)
        self.stretch_delta_lambdas = wp.zeros((self.num_tiles, TILE), dtype=float, device=device)

        self.bend_matrices = wp.zeros((self.num_tiles, TILE, TILE), dtype=float, device=device)
        self.bend_rhs = wp.zeros((self.num_tiles, TILE), dtype=float, device=device)
        self.bend_delta_lambdas = wp.zeros((self.num_tiles, TILE), dtype=float, device=device)

        # Correction accumulators
        self.particle_corrections = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)
        self.edge_q_corrections = wp.zeros(self.num_stretch, dtype=wp.quat, device=device)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        # Director visualization
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

            stretch_compliance_factor = self.stretch_compliance / (self.sim_dt * self.sim_dt)
            bend_compliance_factor = self.bend_compliance / (self.sim_dt * self.sim_dt)

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

            # Step 2: Block-Jacobi Cholesky constraint solving
            for _ in range(self.constraint_iterations):
                # --- Stretch/Shear constraints ---
                wp.launch(
                    kernel=compute_stretch_constraint_data_kernel,
                    dim=self.num_stretch,
                    inputs=[
                        self.state_1.particle_q,
                        self.edge_q,
                        self.rest_length,
                        self.num_stretch,
                    ],
                    outputs=[
                        self.stretch_violation,
                        self.stretch_direction,
                        self.stretch_quat_direction,
                    ],
                    device=self.model.device,
                )

                # Assemble block systems (one per tile, in parallel)
                wp.launch(
                    kernel=assemble_stretch_block_systems_kernel,
                    dim=self.num_tiles,
                    inputs=[
                        self.particle_inv_mass,
                        self.edge_inv_mass,
                        self.rest_length,
                        self.stretch_direction,
                        self.stretch_violation,
                        stretch_compliance_factor,
                        self.num_stretch,
                        self.constraints_per_tile,
                    ],
                    outputs=[self.stretch_matrices, self.stretch_rhs],
                    device=self.model.device,
                )

                # Batched Cholesky solve (all tiles in parallel)
                wp.launch_tiled(
                    kernel=cholesky_solve_batched_kernel,
                    dim=[self.num_tiles, 1],
                    inputs=[self.stretch_matrices, self.stretch_rhs],
                    outputs=[self.stretch_delta_lambdas],
                    block_dim=BLOCK_DIM,
                    device=self.model.device,
                )

                # Zero correction accumulators
                wp.launch(
                    kernel=zero_corrections_vec3_kernel,
                    dim=self.num_particles,
                    inputs=[self.particle_corrections],
                    device=self.model.device,
                )
                wp.launch(
                    kernel=zero_corrections_quat_kernel,
                    dim=self.num_stretch,
                    inputs=[self.edge_q_corrections],
                    device=self.model.device,
                )

                # Apply stretch corrections (with atomics at boundaries)
                wp.launch(
                    kernel=apply_stretch_block_corrections_kernel,
                    dim=self.num_tiles,
                    inputs=[
                        self.particle_inv_mass,
                        self.edge_inv_mass,
                        self.edge_q,
                        self.rest_length,
                        self.stretch_direction,
                        self.stretch_quat_direction,
                        self.stretch_delta_lambdas,
                        self.num_stretch,
                        self.constraints_per_tile,
                        self.num_tiles,
                    ],
                    outputs=[self.particle_corrections, self.edge_q_corrections],
                    device=self.model.device,
                )

                # Apply accumulated corrections
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
                    kernel=apply_quaternion_corrections_kernel,
                    dim=self.num_stretch,
                    inputs=[
                        self.edge_q,
                        self.edge_q_corrections,
                        self.edge_inv_mass,
                    ],
                    outputs=[self.edge_q_new],
                    device=self.model.device,
                )
                self.edge_q, self.edge_q_new = self.edge_q_new, self.edge_q

                # --- Bend/Twist constraints ---
                if self.num_bend > 0:
                    wp.launch(
                        kernel=compute_bend_constraint_data_kernel,
                        dim=self.num_bend,
                        inputs=[
                            self.edge_q,
                            self.rest_darboux,
                            self.num_bend,
                        ],
                        outputs=[self.bend_violation, self.bend_direction],
                        device=self.model.device,
                    )

                    wp.launch(
                        kernel=assemble_bend_block_systems_kernel,
                        dim=self.num_tiles,
                        inputs=[
                            self.edge_inv_mass,
                            self.bend_direction,
                            self.bend_violation,
                            bend_compliance_factor,
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
                        outputs=[self.bend_delta_lambdas],
                        block_dim=BLOCK_DIM,
                        device=self.model.device,
                    )

                    wp.launch(
                        kernel=zero_corrections_quat_kernel,
                        dim=self.num_stretch,
                        inputs=[self.edge_q_corrections],
                        device=self.model.device,
                    )

                    wp.launch(
                        kernel=apply_bend_block_corrections_kernel,
                        dim=self.num_tiles,
                        inputs=[
                            self.edge_q,
                            self.edge_inv_mass,
                            self.bend_direction,
                            self.bend_delta_lambdas,
                            self.num_bend,
                            self.constraints_per_tile,
                            self.num_tiles,
                        ],
                        outputs=[self.edge_q_corrections],
                        device=self.model.device,
                    )

                    wp.launch(
                        kernel=apply_quaternion_corrections_kernel,
                        dim=self.num_stretch,
                        inputs=[
                            self.edge_q,
                            self.edge_q_corrections,
                            self.edge_inv_mass,
                        ],
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
        """Update rest Darboux vectors from current slider values."""
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
        ui.text("Global Cosserat Rod (Cholesky Multi-Tile)")
        ui.text(f"Particles: {self.num_particles}, Tiles: {self.num_tiles}")
        ui.separator()
        ui.text("Compliance (lower = stiffer)")
        _changed, self.stretch_compliance = ui.slider_float(
            "Stretch Compliance", self.stretch_compliance, 1e-8, 1e-4, format="%.2e"
        )
        _changed, self.bend_compliance = ui.slider_float(
            "Bend Compliance", self.bend_compliance, 1e-8, 1e-4, format="%.2e"
        )
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
