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
# Example Cholesky Cosserat Rod (3-DOF Tangent Space Formulation)
#
# Demonstrates a global Cholesky-based solver for Position And Orientation
# Based Cosserat Rods. Unlike examples 02/04 which use Jacobi iteration,
# this example assembles and solves the global constraint system using
# Warp's tile Cholesky decomposition.
#
# Key difference from example 07:
#   - Example 07 reduces 3D constraints to scalar (tridiagonal system)
#   - This example keeps full 3D constraints (denser coupling)
#   - Each constraint contributes 3 rows to the system matrix
#
# Mathematical formulation (constraint-space):
#   - Constraint violation: C (vector of all constraint values, 3 per constraint)
#   - Jacobian: J (maps DOF corrections to constraint changes)
#   - System matrix: A = J M^{-1} J^T + alpha*I (SPD matrix)
#   - Solve: A * delta_lambda = -C
#   - Apply corrections: delta_x = M^{-1} J^T delta_lambda
#
# The stretch/shear and bend/twist constraints are solved separately:
#   - Stretch/shear: 16 constraints × 3 = 48 scalar constraints -> 64×64 tile
#   - Bend/twist: 15 constraints × 3 = 45 scalar constraints -> 64×64 tile
#
# Uses tangent-space quaternion parameterization (3-DOF angular velocity)
# instead of 4-DOF quaternion, enabling efficient Jacobian computation.
#
# For multi-tile version, see example 09.
#
# Command: uv run -m newton.examples cosserat_08_cholesky_cosserat_rod
#
###########################################################################

import math

import warp as wp

import newton
import newton.examples

# Warp tile configuration
BLOCK_DIM = 128
TILE_STRETCH = 64  # 16 constraints x 3 = 48, padded to 64
TILE_BEND = 64  # 15 constraints x 3 = 45, padded to 64

# Rod configuration - sized to fit in single tiles
NUM_PARTICLES = 17  # Gives 16 stretch + 15 bend constraints
NUM_STRETCH = NUM_PARTICLES - 1  # 16 stretch constraints
NUM_BEND = NUM_PARTICLES - 2  # 15 bend constraints


@wp.func
def quat_rotate_e3(q: wp.quat) -> wp.vec3:
    """Compute the third director d3 = q * e3 * conjugate(q) where e3 = (0,0,1)."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    d3_x = 2.0 * (x * z + w * y)
    d3_y = 2.0 * (y * z - w * x)
    d3_z = w * w - x * x - y * y + z * z
    return wp.vec3(d3_x, d3_y, d3_z)


@wp.func
def skew_matrix(v: wp.vec3) -> wp.mat33:
    """Compute the skew-symmetric matrix [v]_x such that [v]_x * u = v x u."""
    return wp.mat33(
        0.0, -v[2], v[1],
        v[2], 0.0, -v[0],
        -v[1], v[0], 0.0,
    )


@wp.func
def quat_exp_approx(omega: wp.vec3) -> wp.quat:
    """Approximate quaternion exponential for small angles: exp(omega/2).

    For small omega, this is approximately quat(omega/2, 1).
    """
    half_omega = omega * 0.5
    angle_sq = wp.dot(half_omega, half_omega)

    if angle_sq < 1.0e-8:
        # First-order approximation
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
def assemble_stretch_system_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    compliance_factor: float,
    stiffness: wp.vec3,
    num_stretch: int,
    # outputs
    system_matrix: wp.array2d(dtype=float),
    system_rhs: wp.array1d(dtype=float),
):
    """
    Assemble the global system for stretch/shear constraints.

    System: A = J M^{-1} J^T + alpha*I
    RHS: b = -C (scaled constraint violations)

    For stretch constraint k between particles k and k+1 with quaternion k:
    - Constraint: gamma_k = (p_{k+1} - p_k) / L_k - d3(q_k) = 0 (3-vector)
    - J_p0 = -I/L (3x3)
    - J_p1 = +I/L (3x3)
    - J_q = 2 * skew(d3) (3x3, tangent-space)

    The system matrix has block structure where constraints couple through
    shared particles/quaternions.
    """
    # Single thread assembles the entire system
    # Initialize to zero
    for i in range(TILE_STRETCH):
        system_rhs[i] = 0.0
        for j in range(TILE_STRETCH):
            system_matrix[i, j] = 0.0

    # For each stretch constraint
    for k in range(num_stretch):
        p0 = particle_q[k]
        p1 = particle_q[k + 1]
        q = edge_q[k]
        L = rest_length[k]

        inv_m0 = particle_inv_mass[k]
        inv_m1 = particle_inv_mass[k + 1]
        inv_mq = edge_inv_mass[k]

        # Compute d3 director
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

        # Store RHS: -C
        row_base = k * 3
        system_rhs[row_base + 0] = -gamma[0]
        system_rhs[row_base + 1] = -gamma[1]
        system_rhs[row_base + 2] = -gamma[2]

        # Jacobians:
        # J_p0 = -I/L, J_p1 = +I/L
        # J_q = 2 * skew(d3) (for omega -> d3 derivative)
        L_inv = 1.0 / L

        # Diagonal block: J_k M^{-1} J_k^T + alpha*I
        # Contribution from p0: (inv_m0 / L^2) * I
        # Contribution from p1: (inv_m1 / L^2) * I
        # Contribution from q: 4 * inv_mq * skew(d3) * skew(d3)^T = 4 * inv_mq * (I - d3*d3^T)

        pos_contrib = (inv_m0 + inv_m1) * L_inv * L_inv

        # skew(d3) * skew(d3)^T = ||d3||^2 * I - d3 * d3^T (for unit d3, this is I - d3*d3^T)
        # Since d3 is unit: skew(d3) * skew(d3)^T = I - outer(d3, d3)
        quat_factor = 4.0 * inv_mq

        for i in range(3):
            for j in range(3):
                row = row_base + i
                col = row_base + j

                # Position contribution: diagonal identity
                pos_val = pos_contrib if i == j else 0.0

                # Quaternion contribution: I - d3*d3^T
                quat_val = quat_factor * ((1.0 if i == j else 0.0) - d3[i] * d3[j])

                # Compliance regularization on diagonal
                compliance_val = compliance_factor if i == j else 0.0

                system_matrix[row, col] = pos_val + quat_val + compliance_val

        # Off-diagonal blocks: coupling with adjacent constraints
        # Constraint k couples with constraint k+1 through particle k+1
        if k + 1 < num_stretch:
            # Coupling through shared particle k+1
            # A[k, k+1] contribution from p_{k+1}:
            # J_k^{p1} * inv_m1 * J_{k+1}^{p0,T} = (I/L_k) * inv_m1 * (-I/L_{k+1})^T
            #                                    = -inv_m1 / (L_k * L_{k+1}) * I
            L_k1 = rest_length[k + 1]
            coupling = -particle_inv_mass[k + 1] * L_inv / L_k1

            next_row_base = (k + 1) * 3
            for i in range(3):
                system_matrix[row_base + i, next_row_base + i] = coupling
                system_matrix[next_row_base + i, row_base + i] = coupling

    # Pad unused entries with identity
    for i in range(num_stretch * 3, TILE_STRETCH):
        system_matrix[i, i] = 1.0


@wp.kernel
def cholesky_solve_stretch_kernel(
    A: wp.array2d(dtype=float),
    b: wp.array1d(dtype=float),
    # output
    x: wp.array1d(dtype=float),
):
    """Solve stretch system using tile Cholesky."""
    a_tile = wp.tile_load(A, shape=(TILE_STRETCH, TILE_STRETCH))
    b_tile = wp.tile_load(b, shape=TILE_STRETCH)

    L = wp.tile_cholesky(a_tile)
    x_tile = wp.tile_cholesky_solve(L, b_tile)

    wp.tile_store(x, x_tile)


@wp.kernel
def apply_stretch_corrections_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    delta_lambda: wp.array1d(dtype=float),
    num_stretch: int,
    # outputs
    particle_q_out: wp.array(dtype=wp.vec3),
    edge_q_out: wp.array(dtype=wp.quat),
):
    """
    Apply stretch corrections: delta_x = M^{-1} J^T delta_lambda

    Uses exponential map for quaternion updates.
    """
    tid = wp.tid()

    if tid < num_stretch + 1:
        # Update particle positions
        inv_mass = particle_inv_mass[tid]
        pos = particle_q[tid]

        if inv_mass > 0.0:
            correction = wp.vec3(0.0, 0.0, 0.0)

            # Contribution from constraint tid-1 (this particle is p1)
            if tid > 0:
                k = tid - 1
                L = rest_length[k]
                dl = wp.vec3(
                    delta_lambda[k * 3 + 0],
                    delta_lambda[k * 3 + 1],
                    delta_lambda[k * 3 + 2],
                )
                # J_p1 = +I/L, so correction = inv_mass * (I/L) * dl
                correction = correction + dl * (inv_mass / L)

            # Contribution from constraint tid (this particle is p0)
            if tid < num_stretch:
                k = tid
                L = rest_length[k]
                dl = wp.vec3(
                    delta_lambda[k * 3 + 0],
                    delta_lambda[k * 3 + 1],
                    delta_lambda[k * 3 + 2],
                )
                # J_p0 = -I/L, so correction = inv_mass * (-I/L) * dl
                correction = correction - dl * (inv_mass / L)

            pos = pos + correction

        particle_q_out[tid] = pos

    # Update edge quaternions (tid maps to edge index for the second part)
    if tid < num_stretch:
        q = edge_q[tid]
        inv_mq = edge_inv_mass[tid]

        if inv_mq > 0.0:
            d3 = quat_rotate_e3(q)

            dl = wp.vec3(
                delta_lambda[tid * 3 + 0],
                delta_lambda[tid * 3 + 1],
                delta_lambda[tid * 3 + 2],
            )

            # J_q = 2 * skew(d3), so J_q^T = -2 * skew(d3)
            # omega = inv_mq * J_q^T * dl = inv_mq * (-2) * skew(d3) * dl
            #       = -2 * inv_mq * (d3 x dl) = 2 * inv_mq * (dl x d3)
            omega = 2.0 * inv_mq * wp.cross(dl, d3)

            # Apply rotation using exponential map
            dq = quat_exp_approx(omega)
            q_new = wp.mul(q, dq)
            q_new = wp.normalize(q_new)

            edge_q_out[tid] = q_new
        else:
            edge_q_out[tid] = q


@wp.kernel
def assemble_bend_system_kernel(
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    rest_darboux: wp.array(dtype=wp.quat),
    compliance_factor: float,
    stiffness: wp.vec3,
    num_bend: int,
    # outputs
    system_matrix: wp.array2d(dtype=float),
    system_rhs: wp.array1d(dtype=float),
):
    """
    Assemble the global system for bend/twist constraints.

    Bend constraint k between quaternions k and k+1:
    - Darboux vector: omega = Im(conj(q_k) * q_{k+1})
    - Constraint: kappa_k = omega - rest_darboux_k = 0 (3-vector)

    Tangent-space Jacobians:
    - d(omega)/d(omega_0) at identity
    - d(omega)/d(omega_1) at identity
    """
    # Single thread assembles the entire system
    # Initialize to zero
    for i in range(TILE_BEND):
        system_rhs[i] = 0.0
        for j in range(TILE_BEND):
            system_matrix[i, j] = 0.0

    # For each bend constraint
    for k in range(num_bend):
        q0 = edge_q[k]
        q1 = edge_q[k + 1]
        rest_d = rest_darboux[k]

        inv_m0 = edge_inv_mass[k]
        inv_m1 = edge_inv_mass[k + 1]

        # Compute Darboux vector: omega = conj(q0) * q1
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

        # Store RHS: -kappa
        row_base = k * 3
        system_rhs[row_base + 0] = -kappa[0]
        system_rhs[row_base + 1] = -kappa[1]
        system_rhs[row_base + 2] = -kappa[2]

        # Jacobians (simplified for near-identity case):
        # d(Im(conj(q0)*q1))/d(omega_0) ≈ -0.5 * I (at identity)
        # d(Im(conj(q0)*q1))/d(omega_1) ≈ +0.5 * I (at identity)
        #
        # More accurate: depends on current relative rotation
        # For simplicity, use the identity approximation which works well for small deformations

        # Diagonal block: (0.25 * inv_m0 + 0.25 * inv_m1 + compliance) * I
        diag = 0.25 * (inv_m0 + inv_m1) + compliance_factor
        for i in range(3):
            system_matrix[row_base + i, row_base + i] = diag

        # Off-diagonal coupling with adjacent bend constraints
        # Constraint k couples with k+1 through quaternion k+1
        if k + 1 < num_bend:
            # Coupling through q_{k+1}:
            # J_k^{q1} * inv_m1 * J_{k+1}^{q0,T}
            # = (0.5*I) * inv_m1 * (-0.5*I)^T = -0.25 * inv_m1 * I
            coupling = -0.25 * edge_inv_mass[k + 1]

            next_row_base = (k + 1) * 3
            for i in range(3):
                system_matrix[row_base + i, next_row_base + i] = coupling
                system_matrix[next_row_base + i, row_base + i] = coupling

    # Pad unused entries with identity
    for i in range(num_bend * 3, TILE_BEND):
        system_matrix[i, i] = 1.0


@wp.kernel
def cholesky_solve_bend_kernel(
    A: wp.array2d(dtype=float),
    b: wp.array1d(dtype=float),
    # output
    x: wp.array1d(dtype=float),
):
    """Solve bend system using tile Cholesky."""
    a_tile = wp.tile_load(A, shape=(TILE_BEND, TILE_BEND))
    b_tile = wp.tile_load(b, shape=TILE_BEND)

    L = wp.tile_cholesky(a_tile)
    x_tile = wp.tile_cholesky_solve(L, b_tile)

    wp.tile_store(x, x_tile)


@wp.kernel
def apply_bend_corrections_kernel(
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    delta_lambda: wp.array1d(dtype=float),
    num_bend: int,
    num_edges: int,
    # output
    edge_q_out: wp.array(dtype=wp.quat),
):
    """
    Apply bend corrections using exponential map.

    For bend constraint k between q_k and q_{k+1}:
    - omega_k += inv_m_k * J_k^T * dl_k
    - omega_{k+1} += inv_m_{k+1} * J_{k+1}^T * dl_k
    """
    tid = wp.tid()
    if tid >= num_edges:
        return

    q = edge_q[tid]
    inv_m = edge_inv_mass[tid]

    if inv_m == 0.0:
        edge_q_out[tid] = q
        return

    omega = wp.vec3(0.0, 0.0, 0.0)

    # Contribution from constraint tid-1 (this quaternion is q1)
    if tid > 0 and tid - 1 < num_bend:
        k = tid - 1
        dl = wp.vec3(
            delta_lambda[k * 3 + 0],
            delta_lambda[k * 3 + 1],
            delta_lambda[k * 3 + 2],
        )
        # J_q1 = 0.5 * I, so J_q1^T = 0.5 * I
        omega = omega + 0.5 * inv_m * dl

    # Contribution from constraint tid (this quaternion is q0)
    if tid < num_bend:
        k = tid
        dl = wp.vec3(
            delta_lambda[k * 3 + 0],
            delta_lambda[k * 3 + 1],
            delta_lambda[k * 3 + 2],
        )
        # J_q0 = -0.5 * I, so J_q0^T = -0.5 * I
        omega = omega - 0.5 * inv_m * dl

    # Apply rotation using exponential map
    dq = quat_exp_approx(omega)
    q_new = wp.mul(q, dq)
    q_new = wp.normalize(q_new)

    edge_q_out[tid] = q_new


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
        self.constraint_iterations = 3  # Fewer iterations needed with Cholesky

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

        # System matrices and vectors for Cholesky solve
        self.stretch_matrix = wp.zeros((TILE_STRETCH, TILE_STRETCH), dtype=float, device=device)
        self.stretch_rhs = wp.zeros(TILE_STRETCH, dtype=float, device=device)
        self.stretch_lambda = wp.zeros(TILE_STRETCH, dtype=float, device=device)

        self.bend_matrix = wp.zeros((TILE_BEND, TILE_BEND), dtype=float, device=device)
        self.bend_rhs = wp.zeros(TILE_BEND, dtype=float, device=device)
        self.bend_lambda = wp.zeros(TILE_BEND, dtype=float, device=device)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        # Director visualization buffers
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

            # Step 2: Constraint solving with Cholesky
            for _ in range(self.constraint_iterations):
                # Solve stretch/shear constraints
                wp.launch(
                    kernel=assemble_stretch_system_kernel,
                    dim=1,
                    inputs=[
                        self.state_1.particle_q,
                        self.particle_inv_mass,
                        self.edge_q,
                        self.edge_inv_mass,
                        self.rest_length,
                        self.compliance_factor,
                        stretch_ks,
                        self.num_stretch,
                    ],
                    outputs=[self.stretch_matrix, self.stretch_rhs],
                    device=self.model.device,
                )

                wp.launch_tiled(
                    kernel=cholesky_solve_stretch_kernel,
                    dim=[1, 1],
                    inputs=[self.stretch_matrix, self.stretch_rhs],
                    outputs=[self.stretch_lambda],
                    block_dim=BLOCK_DIM,
                    device=self.model.device,
                )

                wp.launch(
                    kernel=apply_stretch_corrections_kernel,
                    dim=max(self.num_particles, self.num_stretch),
                    inputs=[
                        self.state_1.particle_q,
                        self.particle_inv_mass,
                        self.edge_q,
                        self.edge_inv_mass,
                        self.rest_length,
                        self.stretch_lambda,
                        self.num_stretch,
                    ],
                    outputs=[self.particle_q_predicted, self.edge_q_new],
                    device=self.model.device,
                )
                wp.copy(self.state_1.particle_q, self.particle_q_predicted)
                self.edge_q, self.edge_q_new = self.edge_q_new, self.edge_q

                # Solve bend/twist constraints
                if self.num_bend > 0:
                    wp.launch(
                        kernel=assemble_bend_system_kernel,
                        dim=1,
                        inputs=[
                            self.edge_q,
                            self.edge_inv_mass,
                            self.rest_darboux,
                            self.compliance_factor,
                            bend_ks,
                            self.num_bend,
                        ],
                        outputs=[self.bend_matrix, self.bend_rhs],
                        device=self.model.device,
                    )

                    wp.launch_tiled(
                        kernel=cholesky_solve_bend_kernel,
                        dim=[1, 1],
                        inputs=[self.bend_matrix, self.bend_rhs],
                        outputs=[self.bend_lambda],
                        block_dim=BLOCK_DIM,
                        device=self.model.device,
                    )

                    wp.launch(
                        kernel=apply_bend_corrections_kernel,
                        dim=self.num_stretch,
                        inputs=[
                            self.edge_q,
                            self.edge_inv_mass,
                            self.bend_lambda,
                            self.num_bend,
                            self.num_stretch,
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
        ui.text("Cholesky Cosserat Rod")
        ui.text(f"Particles: {self.num_particles}, Iterations: {self.constraint_iterations}")
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
