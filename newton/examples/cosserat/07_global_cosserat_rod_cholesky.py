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
# Example Global Cosserat Rod with Cholesky Solve
#
# Demonstrates a TRUE global matrix-based approach for Position And
# Orientation Based Cosserat Rods using Warp's tile Cholesky decomposition.
# Unlike the iterative Jacobi approach in examples 03/04, this example
# assembles a global system matrix A = J M^{-1} J^T and solves it directly.
#
# This guarantees inextensibility (stretch constraint satisfaction) in
# fewer iterations compared to iterative Jacobi methods.
#
# Constraint types:
#   - Stretch/Shear: gamma = (p1-p0)/L - d3(q) = 0
#     Solved via global Cholesky with tridiagonal coupling
#   - Bend/Twist: omega = conj(q0)*q1 - restDarboux = 0
#     Solved via separate global Cholesky system
#
# The banded structure of rod constraints (tridiagonal in constraint space)
# enables efficient O(n) solving via Cholesky factorization.
#
# Command: uv run -m newton.examples cosserat_07_global_cosserat_rod_cholesky
#
###########################################################################

import math

import warp as wp

import newton
import newton.examples

# Warp tile configuration
BLOCK_DIM = 128
TILE = 32  # 32x32 tile for Cholesky

# Rod configuration - sized to fit in single tile
NUM_PARTICLES = 17  # Gives 16 stretch + 15 bend constraints, fits in 32x32
NUM_STRETCH = NUM_PARTICLES - 1  # 16 stretch constraints
NUM_BEND = NUM_PARTICLES - 2  # 15 bend constraints


@wp.func
def quat_rotate_e3(q: wp.quat) -> wp.vec3:
    """Compute the third director d3 = q * e3 * conjugate(q) where e3 = (0,0,1).

    This is an optimized computation of rotating the z-axis by quaternion q.
    """
    x, y, z, w = q[0], q[1], q[2], q[3]
    d3_x = 2.0 * (x * z + w * y)
    d3_y = 2.0 * (y * z - w * x)
    d3_z = w * w - x * x - y * y + z * z
    return wp.vec3(d3_x, d3_y, d3_z)


@wp.func
def quat_e3_bar(q: wp.quat) -> wp.quat:
    """Compute q * e3_bar where e3_bar is the conjugate of quaternion (0,0,1,0).

    In Warp (x,y,z,w) notation: result = (-q.y, q.x, -q.w, q.z)
    """
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
    """
    Compute stretch/shear constraint violations and directions.

    Constraint: gamma = (p1 - p0) / L - d3(q) = 0 (3D vector constraint)

    We linearize this as a scalar constraint with magnitude and direction
    for efficient global solving.
    """
    tid = wp.tid()
    if tid >= num_stretch:
        return

    p0 = particle_q[tid]
    p1 = particle_q[tid + 1]
    q0 = edge_q[tid]
    L = rest_length[tid]

    # Compute third director d3
    d3 = quat_rotate_e3(q0)

    # Compute constraint violation vector: gamma = (p1 - p0) / L - d3
    edge_vec = p1 - p0
    gamma = edge_vec / L - d3

    # Constraint magnitude (how much the constraint is violated)
    gamma_mag = wp.length(gamma)

    if gamma_mag > 1.0e-8:
        # Constraint direction (normalized)
        constraint_direction[tid] = gamma / gamma_mag
    else:
        constraint_direction[tid] = wp.vec3(1.0, 0.0, 0.0)

    constraint_violation[tid] = gamma_mag

    # Quaternion correction direction (for applying corrections later)
    q_e3_bar = quat_e3_bar(q0)
    constraint_quat_direction[tid] = q_e3_bar


@wp.kernel
def assemble_stretch_global_system_kernel(
    particle_inv_mass: wp.array(dtype=float),
    edge_inv_mass: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    constraint_direction: wp.array(dtype=wp.vec3),
    constraint_violation: wp.array(dtype=float),
    compliance_factor: float,
    num_stretch: int,
    # outputs
    system_matrix: wp.array2d(dtype=float),
    system_rhs: wp.array1d(dtype=float),
):
    """
    Assemble the global system matrix A and RHS for stretch/shear constraints.

    System matrix A = J M^{-1} J^T + alpha/dt^2 * I

    For stretch/shear constraints in a chain, A is tridiagonal:
    - A[k,k] = (w_pk + w_p{k+1})/L^2 + 4*L^2*w_qk + compliance
    - A[k,k+1] = coupling through shared particle p_{k+1}

    The position Jacobian gradient is: ∂C/∂p_k = -n/L, ∂C/∂p_{k+1} = +n/L
    The quaternion contribution adds to the diagonal.
    """
    # Initialize matrix to zero
    for i in range(TILE):
        system_rhs[i] = 0.0
        for j in range(TILE):
            system_matrix[i, j] = 0.0

    # Fill in the tridiagonal structure
    for k in range(num_stretch):
        w_p0 = particle_inv_mass[k]
        w_p1 = particle_inv_mass[k + 1]
        w_q0 = edge_inv_mass[k]
        L = rest_length[k]

        # Diagonal contribution from position terms: (w_p0 + w_p1) / L^2
        # For unit direction vectors, dot(n,n) = 1
        pos_diag = (w_p0 + w_p1) / (L * L)

        # Quaternion contribution to diagonal: 4 * L^2 * w_q
        # This comes from the quaternion Jacobian structure
        quat_diag = 4.0 * L * L * w_q0

        # Total diagonal
        diag = pos_diag + quat_diag + compliance_factor
        system_matrix[k, k] = diag

        # RHS: -C_k (constraint violation)
        system_rhs[k] = -constraint_violation[k]

        # Off-diagonal coupling with next constraint (k+1)
        # Constraints k and k+1 share particle k+1
        if k + 1 < num_stretch:
            n_k = constraint_direction[k]
            n_k1 = constraint_direction[k + 1]
            L_k1 = rest_length[k + 1]

            # Coupling: w_{p,k+1} * (n_k/L_k) . (n_{k+1}/L_{k+1})
            # grad_C_k w.r.t. p_{k+1} = +n_k/L_k
            # grad_C_{k+1} w.r.t. p_{k+1} = -n_{k+1}/L_{k+1}
            coupling = -particle_inv_mass[k + 1] * wp.dot(n_k, n_k1) / (L * L_k1)
            system_matrix[k, k + 1] = coupling
            system_matrix[k + 1, k] = coupling

    # Pad unused rows/columns with identity to keep matrix SPD
    for i in range(num_stretch, TILE):
        system_matrix[i, i] = 1.0


@wp.kernel
def cholesky_solve_kernel(
    A: wp.array2d(dtype=float),
    b: wp.array1d(dtype=float),
    # output
    x: wp.array1d(dtype=float),
):
    """Solve Ax = b using tile Cholesky decomposition."""
    a_tile = wp.tile_load(A, shape=(TILE, TILE))
    b_tile = wp.tile_load(b, shape=TILE)

    # Cholesky factorization: A = L L^T
    L = wp.tile_cholesky(a_tile)

    # Solve: L L^T x = b
    x_tile = wp.tile_cholesky_solve(L, b_tile)

    wp.tile_store(x, x_tile)


@wp.kernel
def apply_stretch_corrections_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    edge_q: wp.array(dtype=wp.quat),
    particle_inv_mass: wp.array(dtype=float),
    edge_inv_mass: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    constraint_direction: wp.array(dtype=wp.vec3),
    constraint_quat_direction: wp.array(dtype=wp.quat),
    delta_lambda: wp.array1d(dtype=float),
    num_stretch: int,
    num_particles: int,
    # outputs
    particle_q_out: wp.array(dtype=wp.vec3),
    edge_q_out: wp.array(dtype=wp.quat),
):
    """
    Apply position and quaternion corrections from solved Lagrange multipliers.

    delta_x = M^{-1} J^T delta_lambda
    delta_q = M_q^{-1} J_q^T delta_lambda (then normalize)

    For particle i:
    - Constraint i-1 contributes: +n_{i-1}/L * delta_lambda_{i-1} * w_i (if i > 0)
    - Constraint i contributes: -n_i/L * delta_lambda_i * w_i (if i < num_stretch)
    """
    tid = wp.tid()

    # Handle particles
    if tid < num_particles:
        inv_mass = particle_inv_mass[tid]
        pos = particle_q[tid]

        if inv_mass == 0.0:
            particle_q_out[tid] = pos
        else:
            correction = wp.vec3(0.0, 0.0, 0.0)

            # Contribution from constraint tid-1 (this particle is p_{k+1})
            if tid > 0 and tid - 1 < num_stretch:
                n_prev = constraint_direction[tid - 1]
                L_prev = rest_length[tid - 1]
                dl_prev = delta_lambda[tid - 1]
                # grad_C_{tid-1} w.r.t. p_tid = +n/L
                correction = correction + n_prev * (dl_prev * inv_mass / L_prev)

            # Contribution from constraint tid (this particle is p_k)
            if tid < num_stretch:
                n_curr = constraint_direction[tid]
                L_curr = rest_length[tid]
                dl_curr = delta_lambda[tid]
                # grad_C_tid w.r.t. p_tid = -n/L
                correction = correction - n_curr * (dl_curr * inv_mass / L_curr)

            particle_q_out[tid] = pos + correction

    # Handle quaternions (edges)
    if tid < num_stretch:
        inv_mass_q = edge_inv_mass[tid]
        q = edge_q[tid]

        if inv_mass_q == 0.0:
            edge_q_out[tid] = q
        else:
            n = constraint_direction[tid]
            q_e3_bar = constraint_quat_direction[tid]
            L = rest_length[tid]
            dl = delta_lambda[tid]

            # Quaternion correction: scale * (gamma_quat * q_e3_bar)
            # where gamma_quat = (n, 0)
            gamma_quat = wp.quat(n[0], n[1], n[2], 0.0)
            corrq_raw = wp.mul(gamma_quat, q_e3_bar)

            scale = 2.0 * inv_mass_q * L * dl
            corrq = wp.quat(
                corrq_raw[0] * scale,
                corrq_raw[1] * scale,
                corrq_raw[2] * scale,
                corrq_raw[3] * scale,
            )

            q_new = wp.quat(q[0] + corrq[0], q[1] + corrq[1], q[2] + corrq[2], q[3] + corrq[3])
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
    """
    Compute bend/twist constraint violations and directions.

    Constraint: omega = 2*Im(conj(q0)*q1) should match rest_darboux
    """
    tid = wp.tid()
    if tid >= num_bend:
        return

    q0 = edge_q[tid]
    q1 = edge_q[tid + 1]
    rest_q = rest_darboux[tid]

    # Compute Darboux vector: omega = conj(q0) * q1
    q0_conj = wp.quat(-q0[0], -q0[1], -q0[2], q0[3])
    omega = wp.mul(q0_conj, q1)

    # Handle quaternion double-cover: choose shorter path
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
def assemble_bend_global_system_kernel(
    edge_inv_mass: wp.array(dtype=float),
    bend_direction: wp.array(dtype=wp.vec3),
    bend_violation: wp.array(dtype=float),
    compliance_factor: float,
    num_bend: int,
    # outputs
    system_matrix: wp.array2d(dtype=float),
    system_rhs: wp.array1d(dtype=float),
):
    """
    Assemble global system for bend/twist constraints.

    A = J M^{-1} J^T + compliance*I

    For bend constraints, coupling is through shared quaternions:
    - A[k,k] = w_qk + w_q{k+1} + compliance
    - A[k,k+1] = coupling through shared quaternion q_{k+1}
    """
    # Initialize matrix
    for i in range(TILE):
        system_rhs[i] = 0.0
        for j in range(TILE):
            system_matrix[i, j] = 0.0

    for k in range(num_bend):
        w_q0 = edge_inv_mass[k]
        w_q1 = edge_inv_mass[k + 1]

        # Diagonal
        diag = w_q0 + w_q1 + compliance_factor
        system_matrix[k, k] = diag

        # RHS
        system_rhs[k] = -bend_violation[k]

        # Off-diagonal coupling through shared quaternion
        if k + 1 < num_bend:
            n_k = bend_direction[k]
            n_k1 = bend_direction[k + 1]

            # Coupling: -w_{q,k+1} * (n_k . n_{k+1})
            coupling = -edge_inv_mass[k + 1] * wp.dot(n_k, n_k1)
            system_matrix[k, k + 1] = coupling
            system_matrix[k + 1, k] = coupling

    # Pad with identity
    for i in range(num_bend, TILE):
        system_matrix[i, i] = 1.0


@wp.kernel
def apply_bend_corrections_kernel(
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    bend_direction: wp.array(dtype=wp.vec3),
    delta_lambda: wp.array1d(dtype=float),
    num_bend: int,
    num_edges: int,
    # output
    edge_q_out: wp.array(dtype=wp.quat),
):
    """
    Apply quaternion corrections from bend/twist solve.

    For quaternion k:
    - Constraint k-1 contributes corrq from q1 side
    - Constraint k contributes corrq from q0 side
    """
    tid = wp.tid()
    if tid >= num_edges:
        return

    inv_mass = edge_inv_mass[tid]
    q = edge_q[tid]

    if inv_mass == 0.0:
        edge_q_out[tid] = q
        return

    corrq = wp.quat(0.0, 0.0, 0.0, 0.0)

    # Contribution from constraint tid-1 (this edge is q1)
    if tid > 0 and tid - 1 < num_bend:
        q_prev = edge_q[tid - 1]
        n_prev = bend_direction[tid - 1]
        dl_prev = delta_lambda[tid - 1]

        # corrq1 = q0 * omega * (-invMassq1)
        omega_q = wp.quat(n_prev[0], n_prev[1], n_prev[2], 0.0)
        corrq1_raw = wp.mul(q_prev, omega_q)
        scale = -inv_mass * dl_prev
        corrq = wp.quat(
            corrq[0] + corrq1_raw[0] * scale,
            corrq[1] + corrq1_raw[1] * scale,
            corrq[2] + corrq1_raw[2] * scale,
            corrq[3] + corrq1_raw[3] * scale,
        )

    # Contribution from constraint tid (this edge is q0)
    if tid < num_bend:
        q_next = edge_q[tid + 1]
        n_curr = bend_direction[tid]
        dl_curr = delta_lambda[tid]

        # corrq0 = q1 * omega * invMassq0
        omega_q = wp.quat(n_curr[0], n_curr[1], n_curr[2], 0.0)
        corrq0_raw = wp.mul(q_next, omega_q)
        scale = inv_mass * dl_curr
        corrq = wp.quat(
            corrq[0] + corrq0_raw[0] * scale,
            corrq[1] + corrq0_raw[1] * scale,
            corrq[2] + corrq0_raw[2] * scale,
            corrq[3] + corrq0_raw[3] * scale,
        )

    q_new = wp.quat(q[0] + corrq[0], q[1] + corrq[1], q[2] + corrq[2], q[3] + corrq[3])
    q_new = wp.normalize(q_new)
    edge_q_out[tid] = q_new


@wp.kernel
def update_velocities_kernel(
    particle_q_old: wp.array(dtype=wp.vec3),
    particle_q_new: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    dt: float,
    # output
    particle_qd: wp.array(dtype=wp.vec3),
):
    """Update velocities from position change: v = (x_new - x_old) / dt"""
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
        # With global Cholesky, we need fewer iterations than Jacobi
        self.constraint_iterations = 2

        self.viewer = viewer
        self.args = args

        # Rod parameters
        self.num_particles = NUM_PARTICLES
        self.num_stretch = NUM_STRETCH
        self.num_bend = NUM_BEND

        particle_spacing = 0.1
        particle_mass = 0.1
        particle_radius = 0.02
        edge_mass = 0.01
        start_height = 3.0

        # Compliance parameters (lower = stiffer)
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

        # Edge quaternions (rotate z-axis to x-axis for horizontal rod)
        angle = math.pi / 2.0
        q_init = wp.quat(0.0, math.sin(angle / 2.0), 0.0, math.cos(angle / 2.0))
        edge_q_init = [q_init] * self.num_stretch
        self.edge_q = wp.array(edge_q_init, dtype=wp.quat, device=device)
        self.edge_q_new = wp.array(edge_q_init, dtype=wp.quat, device=device)

        edge_inv_mass_np = [1.0 / edge_mass] * self.num_stretch
        self.edge_inv_mass = wp.array(edge_inv_mass_np, dtype=float, device=device)

        rest_length_np = [particle_spacing] * self.num_stretch
        self.rest_length = wp.array(rest_length_np, dtype=float, device=device)

        # Rest Darboux vectors (identity for straight rod)
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

        # Global system matrices (TILE x TILE for Cholesky)
        self.stretch_matrix = wp.zeros((TILE, TILE), dtype=float, device=device)
        self.stretch_rhs = wp.zeros(TILE, dtype=float, device=device)
        self.stretch_delta_lambda = wp.zeros(TILE, dtype=float, device=device)

        self.bend_matrix = wp.zeros((TILE, TILE), dtype=float, device=device)
        self.bend_rhs = wp.zeros(TILE, dtype=float, device=device)
        self.bend_delta_lambda = wp.zeros(TILE, dtype=float, device=device)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        # Director visualization
        num_director_lines = self.num_stretch * 3
        self.director_line_starts = wp.zeros(num_director_lines, dtype=wp.vec3, device=device)
        self.director_line_ends = wp.zeros(num_director_lines, dtype=wp.vec3, device=device)
        self.director_line_colors = wp.zeros(num_director_lines, dtype=wp.vec3, device=device)
        self.show_directors = True
        self.director_scale = 0.05

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

            # Step 2: Global Cholesky constraint solving
            for _ in range(self.constraint_iterations):
                # --- Stretch/Shear constraints ---
                # Compute constraint data
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

                # Assemble global system
                wp.launch(
                    kernel=assemble_stretch_global_system_kernel,
                    dim=1,
                    inputs=[
                        self.particle_inv_mass,
                        self.edge_inv_mass,
                        self.rest_length,
                        self.stretch_direction,
                        self.stretch_violation,
                        stretch_compliance_factor,
                        self.num_stretch,
                    ],
                    outputs=[self.stretch_matrix, self.stretch_rhs],
                    device=self.model.device,
                )

                # Solve via Cholesky
                wp.launch_tiled(
                    kernel=cholesky_solve_kernel,
                    dim=[1, 1],
                    inputs=[self.stretch_matrix, self.stretch_rhs],
                    outputs=[self.stretch_delta_lambda],
                    block_dim=BLOCK_DIM,
                    device=self.model.device,
                )

                # Apply stretch corrections
                wp.launch(
                    kernel=apply_stretch_corrections_kernel,
                    dim=max(self.num_particles, self.num_stretch),
                    inputs=[
                        self.state_1.particle_q,
                        self.edge_q,
                        self.particle_inv_mass,
                        self.edge_inv_mass,
                        self.rest_length,
                        self.stretch_direction,
                        self.stretch_quat_direction,
                        self.stretch_delta_lambda,
                        self.num_stretch,
                        self.num_particles,
                    ],
                    outputs=[self.particle_q_predicted, self.edge_q_new],
                    device=self.model.device,
                )

                wp.copy(self.state_1.particle_q, self.particle_q_predicted)
                self.edge_q, self.edge_q_new = self.edge_q_new, self.edge_q

                # --- Bend/Twist constraints ---
                if self.num_bend > 0:
                    # Compute bend constraint data
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

                    # Assemble bend system
                    wp.launch(
                        kernel=assemble_bend_global_system_kernel,
                        dim=1,
                        inputs=[
                            self.edge_inv_mass,
                            self.bend_direction,
                            self.bend_violation,
                            bend_compliance_factor,
                            self.num_bend,
                        ],
                        outputs=[self.bend_matrix, self.bend_rhs],
                        device=self.model.device,
                    )

                    # Solve via Cholesky
                    wp.launch_tiled(
                        kernel=cholesky_solve_kernel,
                        dim=[1, 1],
                        inputs=[self.bend_matrix, self.bend_rhs],
                        outputs=[self.bend_delta_lambda],
                        block_dim=BLOCK_DIM,
                        device=self.model.device,
                    )

                    # Apply bend corrections
                    wp.launch(
                        kernel=apply_bend_corrections_kernel,
                        dim=self.num_stretch,
                        inputs=[
                            self.edge_q,
                            self.edge_inv_mass,
                            self.bend_direction,
                            self.bend_delta_lambda,
                            self.num_bend,
                            self.num_stretch,
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

        p_lower = wp.vec3(-2.0, -3.0, -0.1)
        p_upper = wp.vec3(4.0, 3.0, 5.0)
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
        ui.text("Global Cosserat Rod (Cholesky Solve)")
        ui.text(f"Particles: {self.num_particles}, Iterations: {self.constraint_iterations}")
        ui.separator()
        ui.text("Compliance (lower = stiffer)")
        changed_s, self.stretch_compliance = ui.slider_float(
            "Stretch Compliance", self.stretch_compliance, 1e-8, 1e-4, format="%.2e"
        )
        changed_b, self.bend_compliance = ui.slider_float(
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
        _changed, self.director_scale = ui.slider_float("Director Scale", self.director_scale, 0.01, 0.2)

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
