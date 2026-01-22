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
# Example Full Cosserat Rod with Block-Tridiagonal Global Solver
#
# Implements Position And Orientation Based Cosserat Rods using a global
# block-tridiagonal solver instead of iterative Jacobi projection.
#
# Constraints:
#   - Stretch/Shear (3 DOFs per edge k): γ_k = (p_{k+1} - p_k)/L - d3(q_k) = 0
#     Involves: particles k, k+1 and quaternion k
#   - Bend/Twist (3 DOFs between edges k, k+1): ω_k = vec(conj(q_k) * q_{k+1} - rest) = 0
#     Involves: quaternions k and k+1
#
# Block structure (6×6 per edge):
#   - Rows 0-2: stretch/shear constraint for edge k
#   - Rows 3-5: bend/twist constraint k (between quaternions k and k+1)
#   - This creates coupling: bend_k shares q_k with stretch_k, and q_{k+1} with stretch_{k+1}
#
# IMPORTANT INDEXING NOTE:
#   - Stretch constraint k involves edge k (particle k to k+1, quaternion k)
#   - Bend constraint k involves the JUNCTION between edges k and k+1 (quaternions k, k+1)
#   - The last edge has NO bend constraint after it
#   - Total: n edges -> n stretch constraints, n-1 bend constraints
#
# Uses block Thomas algorithm for O(n) direct solve.
#
# STATUS: NOT WORKING - visualization issue, possibly indexing problems
#
# Reference: "Position And Orientation Based Cosserat Rods"
# by Tassilo Kugelstadt, RWTH Aachen University
# https://animation.rwth-aachen.de/publication/0550/
#
# Command: uv run -m newton.examples cosserat_block_tridiagonal_full
#
###########################################################################

import math

import warp as wp

import newton
import newton.examples

# Block configuration
BLOCK = 6  # 6×6 blocks: 3 stretch-shear + 3 bend-twist
BLOCK_DIM = 128

# Rod configuration
NUM_PARTICLES = 32
NUM_EDGES = NUM_PARTICLES - 1  # 31 edges
NUM_BEND = NUM_EDGES - 1  # 30 bend constraints


# ============================================================================
# Matrix utilities for 6×6 blocks (using array storage, no native mat66)
# ============================================================================

@wp.func
def load_vec6(arr: wp.array2d(dtype=float), idx: int) -> wp.spatial_vectorf:
    """Load a 6-vector from array."""
    return wp.spatial_vectorf(
        arr[idx, 0], arr[idx, 1], arr[idx, 2],
        arr[idx, 3], arr[idx, 4], arr[idx, 5]
    )


@wp.func
def store_vec6(arr: wp.array2d(dtype=float), idx: int, v: wp.spatial_vectorf):
    """Store a 6-vector to array."""
    arr[idx, 0] = v[0]
    arr[idx, 1] = v[1]
    arr[idx, 2] = v[2]
    arr[idx, 3] = v[3]
    arr[idx, 4] = v[4]
    arr[idx, 5] = v[5]


@wp.func
def mat66_mul_vec6(
    M: wp.array3d(dtype=float),
    m_idx: int,
    v: wp.spatial_vectorf,
) -> wp.spatial_vectorf:
    """Multiply 6×6 matrix by 6-vector: result = M[m_idx] @ v"""
    r0 = M[m_idx, 0, 0] * v[0] + M[m_idx, 0, 1] * v[1] + M[m_idx, 0, 2] * v[2] + M[m_idx, 0, 3] * v[3] + M[m_idx, 0, 4] * v[4] + M[m_idx, 0, 5] * v[5]
    r1 = M[m_idx, 1, 0] * v[0] + M[m_idx, 1, 1] * v[1] + M[m_idx, 1, 2] * v[2] + M[m_idx, 1, 3] * v[3] + M[m_idx, 1, 4] * v[4] + M[m_idx, 1, 5] * v[5]
    r2 = M[m_idx, 2, 0] * v[0] + M[m_idx, 2, 1] * v[1] + M[m_idx, 2, 2] * v[2] + M[m_idx, 2, 3] * v[3] + M[m_idx, 2, 4] * v[4] + M[m_idx, 2, 5] * v[5]
    r3 = M[m_idx, 3, 0] * v[0] + M[m_idx, 3, 1] * v[1] + M[m_idx, 3, 2] * v[2] + M[m_idx, 3, 3] * v[3] + M[m_idx, 3, 4] * v[4] + M[m_idx, 3, 5] * v[5]
    r4 = M[m_idx, 4, 0] * v[0] + M[m_idx, 4, 1] * v[1] + M[m_idx, 4, 2] * v[2] + M[m_idx, 4, 3] * v[3] + M[m_idx, 4, 4] * v[4] + M[m_idx, 4, 5] * v[5]
    r5 = M[m_idx, 5, 0] * v[0] + M[m_idx, 5, 1] * v[1] + M[m_idx, 5, 2] * v[2] + M[m_idx, 5, 3] * v[3] + M[m_idx, 5, 4] * v[4] + M[m_idx, 5, 5] * v[5]
    return wp.spatial_vectorf(r0, r1, r2, r3, r4, r5)


@wp.func
def mat66_copy(
    src: wp.array3d(dtype=float),
    src_idx: int,
    dst: wp.array3d(dtype=float),
    dst_idx: int,
):
    """Copy 6×6 matrix from src to dst."""
    for i in range(6):
        for j in range(6):
            dst[dst_idx, i, j] = src[src_idx, i, j]


@wp.func
def mat66_sub_product(
    dst: wp.array3d(dtype=float),
    dst_idx: int,
    L: wp.array3d(dtype=float),
    l_idx: int,
    M_inv: wp.array3d(dtype=float),
    m_idx: int,
    U: wp.array3d(dtype=float),
    u_idx: int,
):
    """Compute dst = dst - L @ M_inv @ U (all 6×6 matrices)."""
    # First compute temp = M_inv @ U
    # Then compute dst -= L @ temp
    for i in range(6):
        for j in range(6):
            val = 0.0
            for k in range(6):
                temp_kj = 0.0
                for m in range(6):
                    temp_kj = temp_kj + M_inv[m_idx, k, m] * U[u_idx, m, j]
                val = val + L[l_idx, i, k] * temp_kj
            dst[dst_idx, i, j] = dst[dst_idx, i, j] - val


@wp.func
def mat66_inverse_lu_inplace(
    A: wp.array3d(dtype=float),
    a_idx: int,
    inv: wp.array3d(dtype=float),
    inv_idx: int,
    work: wp.array3d(dtype=float),
    work_idx: int,
):
    """
    Compute inverse of 6×6 matrix using LU decomposition.
    A is input, inv is output, work is workspace for LU factors.
    """
    # Copy A to work for LU decomposition
    for i in range(6):
        for j in range(6):
            work[work_idx, i, j] = A[a_idx, i, j]

    # Initialize inv to identity
    for i in range(6):
        for j in range(6):
            if i == j:
                inv[inv_idx, i, j] = 1.0
            else:
                inv[inv_idx, i, j] = 0.0

    # LU decomposition with partial pivoting (in-place on work)
    # Also apply pivots to inv (which starts as identity)
    for k in range(6):
        # Find pivot
        max_val = wp.abs(work[work_idx, k, k])
        max_row = k

        for i in range(k + 1, 6):
            if wp.abs(work[work_idx, i, k]) > max_val:
                max_val = wp.abs(work[work_idx, i, k])
                max_row = i

        # Swap rows
        if max_row != k:
            for j in range(6):
                temp = work[work_idx, k, j]
                work[work_idx, k, j] = work[work_idx, max_row, j]
                work[work_idx, max_row, j] = temp

                temp = inv[inv_idx, k, j]
                inv[inv_idx, k, j] = inv[inv_idx, max_row, j]
                inv[inv_idx, max_row, j] = temp

        # Check for singular
        if wp.abs(work[work_idx, k, k]) < 1.0e-12:
            # Return identity
            for i in range(6):
                for j in range(6):
                    if i == j:
                        inv[inv_idx, i, j] = 1.0
                    else:
                        inv[inv_idx, i, j] = 0.0
            return

        # Elimination
        for i in range(k + 1, 6):
            factor = work[work_idx, i, k] / work[work_idx, k, k]
            work[work_idx, i, k] = factor  # Store L factor
            for j in range(k + 1, 6):
                work[work_idx, i, j] = work[work_idx, i, j] - factor * work[work_idx, k, j]

    # Solve for each column of inverse
    for col in range(6):
        # Forward substitution: L * y = inv[:, col] (which has been permuted)
        for i in range(1, 6):
            for j in range(i):
                inv[inv_idx, i, col] = inv[inv_idx, i, col] - work[work_idx, i, j] * inv[inv_idx, j, col]

        # Back substitution: U * x = y
        for i in range(5, -1, -1):
            for j in range(i + 1, 6):
                inv[inv_idx, i, col] = inv[inv_idx, i, col] - work[work_idx, i, j] * inv[inv_idx, j, col]
            inv[inv_idx, i, col] = inv[inv_idx, i, col] / work[work_idx, i, i]


# ============================================================================
# Quaternion utilities
# ============================================================================

@wp.func
def quat_rotate_e3(q: wp.quat) -> wp.vec3:
    """Compute d3 = q * e3 * conj(q) where e3 = (0,0,1)."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    d3_x = 2.0 * (x * z + w * y)
    d3_y = 2.0 * (y * z - w * x)
    d3_z = w * w - x * x - y * y + z * z
    return wp.vec3(d3_x, d3_y, d3_z)


@wp.func
def quat_e3_bar(q: wp.quat) -> wp.quat:
    """Compute q * e3_bar where e3_bar = (0, 0, -1, 0)."""
    return wp.quat(-q[1], q[0], -q[3], q[2])


@wp.func
def quat_to_rotation_matrix(q: wp.quat) -> wp.mat33:
    """Convert quaternion to 3×3 rotation matrix."""
    x, y, z, w = q[0], q[1], q[2], q[3]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return wp.mat33(
        1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy),
        2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
        2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy),
    )


# ============================================================================
# Integration kernels
# ============================================================================

@wp.kernel
def integrate_particles_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    gravity: wp.vec3,
    dt: float,
    # outputs
    particle_q_pred: wp.array(dtype=wp.vec3),
    particle_qd_new: wp.array(dtype=wp.vec3),
):
    """Semi-implicit Euler integration for particles."""
    tid = wp.tid()
    inv_mass = particle_inv_mass[tid]

    if inv_mass == 0.0:
        particle_q_pred[tid] = particle_q[tid]
        particle_qd_new[tid] = particle_qd[tid]
        return

    v_new = particle_qd[tid] + gravity * dt
    x_pred = particle_q[tid] + v_new * dt

    particle_q_pred[tid] = x_pred
    particle_qd_new[tid] = v_new


# ============================================================================
# Constraint computation kernels
# ============================================================================

@wp.kernel
def compute_constraint_data_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    edge_q: wp.array(dtype=wp.quat),
    rest_length: wp.array(dtype=float),
    rest_darboux: wp.array(dtype=wp.quat),
    num_edges: int,
    num_bend: int,
    # outputs
    stretch_violation: wp.array(dtype=wp.vec3),  # γ = (p1-p0)/L - d3
    bend_violation: wp.array(dtype=wp.vec3),  # ω = vec part of conj(q0)*q1 - rest
    edge_d3: wp.array(dtype=wp.vec3),  # cached d3 directors
    edge_rotation: wp.array(dtype=wp.mat33),  # cached rotation matrices
):
    """Compute constraint violations for stretch-shear and bend-twist."""
    tid = wp.tid()

    if tid < num_edges:
        # Stretch-shear constraint
        p0 = particle_q[tid]
        p1 = particle_q[tid + 1]
        q = edge_q[tid]
        L = rest_length[tid]

        d3 = quat_rotate_e3(q)
        edge_d3[tid] = d3

        R = quat_to_rotation_matrix(q)
        edge_rotation[tid] = R

        edge_vec = p1 - p0
        gamma = edge_vec / L - d3
        stretch_violation[tid] = gamma

    if tid < num_bend:
        # Bend-twist constraint
        q0 = edge_q[tid]
        q1 = edge_q[tid + 1]
        rest_q = rest_darboux[tid]

        # Darboux vector: omega = conj(q0) * q1
        q0_conj = wp.quat(-q0[0], -q0[1], -q0[2], q0[3])
        omega = wp.mul(q0_conj, q1)

        # Handle quaternion double-cover
        omega_plus = wp.vec4f(
            omega[0] + rest_q[0],
            omega[1] + rest_q[1],
            omega[2] + rest_q[2],
            omega[3] + rest_q[3],
        )
        omega_minus = wp.vec4f(
            omega[0] - rest_q[0],
            omega[1] - rest_q[1],
            omega[2] - rest_q[2],
            omega[3] - rest_q[3],
        )

        norm_plus_sq = wp.dot(omega_plus, omega_plus)
        norm_minus_sq = wp.dot(omega_minus, omega_minus)

        if norm_minus_sq > norm_plus_sq:
            bend_violation[tid] = wp.vec3(omega_plus[0], omega_plus[1], omega_plus[2])
        else:
            bend_violation[tid] = wp.vec3(omega_minus[0], omega_minus[1], omega_minus[2])


@wp.kernel
def assemble_block_system_kernel(
    particle_inv_mass: wp.array(dtype=float),
    edge_inv_mass: wp.array(dtype=float),
    stretch_violation: wp.array(dtype=wp.vec3),
    bend_violation: wp.array(dtype=wp.vec3),
    edge_d3: wp.array(dtype=wp.vec3),
    edge_rotation: wp.array(dtype=wp.mat33),
    edge_q: wp.array(dtype=wp.quat),
    rest_length: wp.array(dtype=float),
    stretch_stiffness: wp.vec3,
    bend_stiffness: wp.vec3,
    compliance_factor: float,
    num_edges: int,
    num_bend: int,
    # outputs
    D_blocks: wp.array3d(dtype=float),  # (num_edges, 6, 6)
    L_blocks: wp.array3d(dtype=float),  # (num_edges-1, 6, 6)
    U_blocks: wp.array3d(dtype=float),  # (num_edges-1, 6, 6)
    b_blocks: wp.array2d(dtype=float),  # (num_edges, 6)
):
    """
    Assemble block-tridiagonal system for coupled Cosserat constraints.

    Block structure per edge k (6×6 diagonal block D_k):
      - Rows 0-2: stretch-shear constraint k (involves p_k, p_{k+1}, q_k)
      - Rows 3-5: bend-twist constraint k (involves q_k, q_{k+1})

    NOTE: The last edge (k = num_edges-1) has no bend constraint after it,
    so rows 3-5 of D_{num_edges-1} are regularized with identity.

    The system matrix A = J M^{-1} J^T + compliance*I

    Off-diagonal coupling (L_k, U_k connect blocks k and k+1):
      - Stretch_k and stretch_{k+1} share particle p_{k+1}
      - Stretch_k and bend_k both involve q_k (within-block coupling in D_k)
      - Bend_k and stretch_{k+1} share quaternion q_{k+1}
      - Bend_k and bend_{k+1} share quaternion q_{k+1}
    """
    # Single-threaded assembly
    eps = 1.0e-8

    # Initialize to zero
    for k in range(num_edges):
        for i in range(6):
            b_blocks[k, i] = 0.0
            for j in range(6):
                D_blocks[k, i, j] = 0.0
                if k < num_edges - 1:
                    L_blocks[k, i, j] = 0.0
                    U_blocks[k, i, j] = 0.0

    # Assemble each edge's contribution
    for k in range(num_edges):
        L_k = rest_length[k]
        inv_L = 1.0 / L_k
        inv_L_sq = inv_L * inv_L

        w_p0 = particle_inv_mass[k]
        w_p1 = particle_inv_mass[k + 1]
        w_q = edge_inv_mass[k]

        gamma = stretch_violation[k]
        R = edge_rotation[k]

        # ================================================================
        # Stretch-shear diagonal block (rows 0-2, cols 0-2)
        # ================================================================
        # A_stretch = J_p M_p^{-1} J_p^T + J_q M_q^{-1} J_q^T + compliance
        # J_p = [-I/L, I/L] for positions
        # J_q involves quaternion derivatives

        # Position contribution: (w_p0 + w_p1) / L^2 * I
        pos_factor = (w_p0 + w_p1) * inv_L_sq

        # Quaternion contribution: approximately 4 * w_q * L^2 (from paper)
        quat_factor = 4.0 * w_q * L_k * L_k * inv_L_sq  # = 4 * w_q

        stretch_diag = pos_factor + quat_factor + compliance_factor

        for i in range(3):
            D_blocks[k, i, i] = stretch_diag

        # RHS for stretch: -γ (transformed by stiffness in local frame)
        # gamma_loc = R^T * gamma, apply stiffness, gamma = R * gamma_loc
        gamma_loc = wp.vec3(
            R[0, 0] * gamma[0] + R[1, 0] * gamma[1] + R[2, 0] * gamma[2],
            R[0, 1] * gamma[0] + R[1, 1] * gamma[1] + R[2, 1] * gamma[2],
            R[0, 2] * gamma[0] + R[1, 2] * gamma[1] + R[2, 2] * gamma[2],
        )
        gamma_loc = wp.vec3(
            gamma_loc[0] * stretch_stiffness[0],
            gamma_loc[1] * stretch_stiffness[1],
            gamma_loc[2] * stretch_stiffness[2],
        )
        gamma_scaled = wp.vec3(
            R[0, 0] * gamma_loc[0] + R[0, 1] * gamma_loc[1] + R[0, 2] * gamma_loc[2],
            R[1, 0] * gamma_loc[0] + R[1, 1] * gamma_loc[1] + R[1, 2] * gamma_loc[2],
            R[2, 0] * gamma_loc[0] + R[2, 1] * gamma_loc[1] + R[2, 2] * gamma_loc[2],
        )

        b_blocks[k, 0] = -gamma_scaled[0]
        b_blocks[k, 1] = -gamma_scaled[1]
        b_blocks[k, 2] = -gamma_scaled[2]

        # ================================================================
        # Bend-twist diagonal block (rows 3-5, cols 3-5)
        # ================================================================
        if k < num_bend:
            w_q0 = edge_inv_mass[k]
            w_q1 = edge_inv_mass[k + 1]

            # Bend diagonal: (w_q0 + w_q1) + compliance
            bend_diag = w_q0 + w_q1 + compliance_factor

            for i in range(3):
                D_blocks[k, 3 + i, 3 + i] = bend_diag

            # RHS for bend: -ω (with stiffness)
            omega = bend_violation[k]
            b_blocks[k, 3] = -omega[0] * bend_stiffness[0]
            b_blocks[k, 4] = -omega[1] * bend_stiffness[1]
            b_blocks[k, 5] = -omega[2] * bend_stiffness[2]
        else:
            # Last edge has no bend constraint - add identity for regularization
            for i in range(3):
                D_blocks[k, 3 + i, 3 + i] = 1.0

        # ================================================================
        # Off-diagonal coupling: stretch_k <-> bend_k (within same block)
        # Both involve q_k
        # ================================================================
        if k < num_bend:
            # Coupling through shared quaternion q_k
            # This is typically small for well-conditioned problems
            # For simplicity, we approximate as zero (block-diagonal within D)
            pass

        # ================================================================
        # Off-diagonal blocks L[k], U[k]: coupling between blocks k and k+1
        # ================================================================
        if k < num_edges - 1:
            # Stretch_k and stretch_{k+1} share particle k+1
            # Coupling: -w_{p,k+1} / (L_k * L_{k+1}) * (d3_k · d3_{k+1}) for aligned dirs
            L_k1 = rest_length[k + 1]
            d3_k = edge_d3[k]
            d3_k1 = edge_d3[k + 1]
            dot_d3 = wp.dot(d3_k, d3_k1)

            w_shared = particle_inv_mass[k + 1]
            stretch_coupling = -w_shared * inv_L / L_k1 * dot_d3

            # Stretch-stretch coupling (rows 0-2 to rows 0-2 of next block)
            for i in range(3):
                L_blocks[k, i, i] = stretch_coupling
                U_blocks[k, i, i] = stretch_coupling

            # Bend_k involves q_k and q_{k+1}
            # Bend_{k+1} involves q_{k+1} and q_{k+2}
            # They share q_{k+1}, creating coupling
            if k < num_bend - 1:
                w_q_shared = edge_inv_mass[k + 1]
                bend_coupling = -w_q_shared

                for i in range(3):
                    L_blocks[k, 3 + i, 3 + i] = bend_coupling
                    U_blocks[k, 3 + i, 3 + i] = bend_coupling


@wp.kernel
def block_thomas_solve_kernel(
    D_blocks: wp.array3d(dtype=float),
    L_blocks: wp.array3d(dtype=float),
    U_blocks: wp.array3d(dtype=float),
    b_blocks: wp.array2d(dtype=float),
    num_edges: int,
    # workspace
    D_prime: wp.array3d(dtype=float),
    b_prime: wp.array2d(dtype=float),
    D_inv_work: wp.array3d(dtype=float),  # workspace for inverse
    LU_work: wp.array3d(dtype=float),  # workspace for LU decomposition
    # output
    x_blocks: wp.array2d(dtype=float),
):
    """
    Block Thomas algorithm for 6×6 block-tridiagonal system.
    """
    n = num_edges

    # Forward sweep
    # D'[0] = D[0], b'[0] = b[0]
    mat66_copy(D_blocks, 0, D_prime, 0)
    b0 = load_vec6(b_blocks, 0)
    store_vec6(b_prime, 0, b0)

    for i in range(1, n):
        # Copy D[i] to D'[i] first
        mat66_copy(D_blocks, i, D_prime, i)

        # Compute inv(D'[i-1])
        mat66_inverse_lu_inplace(D_prime, i - 1, D_inv_work, 0, LU_work, 0)

        # D'[i] = D[i] - L[i-1] @ inv(D'[i-1]) @ U[i-1]
        mat66_sub_product(D_prime, i, L_blocks, i - 1, D_inv_work, 0, U_blocks, i - 1)

        # b'[i] = b[i] - L[i-1] @ inv(D'[i-1]) @ b'[i-1]
        b_i = load_vec6(b_blocks, i)
        b_prime_im1 = load_vec6(b_prime, i - 1)

        # temp = inv(D'[i-1]) @ b'[i-1]
        temp_vec = mat66_mul_vec6(D_inv_work, 0, b_prime_im1)
        # result = L[i-1] @ temp
        result_vec = mat66_mul_vec6(L_blocks, i - 1, temp_vec)
        # b'[i] = b[i] - result
        b_prime_i = wp.spatial_vectorf(
            b_i[0] - result_vec[0],
            b_i[1] - result_vec[1],
            b_i[2] - result_vec[2],
            b_i[3] - result_vec[3],
            b_i[4] - result_vec[4],
            b_i[5] - result_vec[5],
        )
        store_vec6(b_prime, i, b_prime_i)

    # Back substitution
    # x[n-1] = inv(D'[n-1]) @ b'[n-1]
    mat66_inverse_lu_inplace(D_prime, n - 1, D_inv_work, 0, LU_work, 0)
    b_prime_nm1 = load_vec6(b_prime, n - 1)
    x_nm1 = mat66_mul_vec6(D_inv_work, 0, b_prime_nm1)
    store_vec6(x_blocks, n - 1, x_nm1)

    for i in range(n - 2, -1, -1):
        # x[i] = inv(D'[i]) @ (b'[i] - U[i] @ x[i+1])
        mat66_inverse_lu_inplace(D_prime, i, D_inv_work, 0, LU_work, 0)

        b_prime_i = load_vec6(b_prime, i)
        x_ip1 = load_vec6(x_blocks, i + 1)

        # U[i] @ x[i+1]
        U_x = mat66_mul_vec6(U_blocks, i, x_ip1)

        # b'[i] - U[i] @ x[i+1]
        rhs = wp.spatial_vectorf(
            b_prime_i[0] - U_x[0],
            b_prime_i[1] - U_x[1],
            b_prime_i[2] - U_x[2],
            b_prime_i[3] - U_x[3],
            b_prime_i[4] - U_x[4],
            b_prime_i[5] - U_x[5],
        )

        # inv(D'[i]) @ rhs
        x_i = mat66_mul_vec6(D_inv_work, 0, rhs)
        store_vec6(x_blocks, i, x_i)


@wp.kernel
def apply_particle_stretch_corrections_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    delta_lambda: wp.array2d(dtype=float),
    num_edges: int,
    # output
    particle_q_out: wp.array(dtype=wp.vec3),
):
    """Apply position corrections from stretch constraints."""
    tid = wp.tid()

    inv_mass = particle_inv_mass[tid]
    pos = particle_q[tid]

    if inv_mass == 0.0:
        particle_q_out[tid] = pos
        return

    correction = wp.vec3(0.0, 0.0, 0.0)

    # Contribution from constraint tid-1 (this particle is right particle)
    if tid > 0 and tid - 1 < num_edges:
        L = rest_length[tid - 1]
        dl = wp.vec3(
            delta_lambda[tid - 1, 0],
            delta_lambda[tid - 1, 1],
            delta_lambda[tid - 1, 2],
        )
        # J^T for right particle = +I/L, so corr = -w * dl / L
        correction = correction - dl * inv_mass / L

    # Contribution from constraint tid (this particle is left particle)
    if tid < num_edges:
        L = rest_length[tid]
        dl = wp.vec3(
            delta_lambda[tid, 0],
            delta_lambda[tid, 1],
            delta_lambda[tid, 2],
        )
        # J^T for left particle = -I/L, so corr = +w * dl / L
        correction = correction + dl * inv_mass / L

    particle_q_out[tid] = pos + correction


@wp.kernel
def apply_quaternion_stretch_corrections_kernel(
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    delta_lambda: wp.array2d(dtype=float),
    num_edges: int,
    # output
    edge_q_out: wp.array(dtype=wp.quat),
):
    """Apply quaternion corrections from stretch constraints."""
    tid = wp.tid()
    if tid >= num_edges:
        return

    q = edge_q[tid]
    w_q = edge_inv_mass[tid]
    L = rest_length[tid]

    dl = wp.vec3(
        delta_lambda[tid, 0],
        delta_lambda[tid, 1],
        delta_lambda[tid, 2],
    )

    # corrq = gamma_quat * q_e3_bar * (2 * w_q * L)
    q_e3_bar_val = quat_e3_bar(q)
    gamma_quat = wp.quat(dl[0], dl[1], dl[2], 0.0)
    corrq = wp.mul(gamma_quat, q_e3_bar_val)
    scale = 2.0 * w_q * L

    q_new = wp.quat(
        q[0] + corrq[0] * scale,
        q[1] + corrq[1] * scale,
        q[2] + corrq[2] * scale,
        q[3] + corrq[3] * scale,
    )
    edge_q_out[tid] = wp.normalize(q_new)


@wp.kernel
def apply_quaternion_bend_corrections_kernel(
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    delta_lambda: wp.array2d(dtype=float),
    num_bend: int,
    # output (accumulated via atomics on pre-copied array)
    edge_q_delta: wp.array(dtype=wp.quat),
):
    """Accumulate quaternion corrections from bend constraints."""
    tid = wp.tid()
    if tid >= num_bend:
        return

    dl = wp.vec3(
        delta_lambda[tid, 3],
        delta_lambda[tid, 4],
        delta_lambda[tid, 5],
    )

    q0 = edge_q[tid]
    q1 = edge_q[tid + 1]
    w_q0 = edge_inv_mass[tid]
    w_q1 = edge_inv_mass[tid + 1]

    omega_quat = wp.quat(dl[0], dl[1], dl[2], 0.0)

    # corrq0 = q1 * omega * w_q0
    corrq0 = wp.mul(q1, omega_quat)
    dq0 = wp.quat(
        corrq0[0] * w_q0,
        corrq0[1] * w_q0,
        corrq0[2] * w_q0,
        corrq0[3] * w_q0,
    )
    wp.atomic_add(edge_q_delta, tid, dq0)

    # corrq1 = q0 * omega * (-w_q1)
    corrq1 = wp.mul(q0, omega_quat)
    dq1 = wp.quat(
        -corrq1[0] * w_q1,
        -corrq1[1] * w_q1,
        -corrq1[2] * w_q1,
        -corrq1[3] * w_q1,
    )
    wp.atomic_add(edge_q_delta, tid + 1, dq1)


@wp.kernel
def apply_quaternion_delta_kernel(
    edge_q: wp.array(dtype=wp.quat),
    edge_q_delta: wp.array(dtype=wp.quat),
    num_edges: int,
    # output
    edge_q_out: wp.array(dtype=wp.quat),
):
    """Apply accumulated quaternion delta and normalize."""
    tid = wp.tid()
    if tid >= num_edges:
        return

    q = edge_q[tid]
    dq = edge_q_delta[tid]

    q_new = wp.quat(
        q[0] + dq[0],
        q[1] + dq[1],
        q[2] + dq[2],
        q[3] + dq[3],
    )
    edge_q_out[tid] = wp.normalize(q_new)


@wp.kernel
def zero_quat_kernel(arr: wp.array(dtype=wp.quat)):
    """Zero out quaternion array."""
    tid = wp.tid()
    arr[tid] = wp.quat(0.0, 0.0, 0.0, 0.0)


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
    """Ground collision constraint."""
    tid = wp.tid()

    inv_mass = particle_inv_mass[tid]
    pos = particle_q[tid]

    if inv_mass == 0.0:
        particle_q_out[tid] = pos
        return

    radius = particle_radius[tid]
    min_z = ground_level + radius

    if pos[2] < min_z:
        particle_q_out[tid] = wp.vec3(pos[0], pos[1], min_z)
    else:
        particle_q_out[tid] = pos


class Example:
    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.constraint_iterations = 3

        self.viewer = viewer
        self.args = args

        # Rod parameters
        self.num_particles = NUM_PARTICLES
        self.num_edges = NUM_EDGES
        self.num_bend = NUM_BEND

        particle_spacing = 0.1
        particle_mass = 0.1
        particle_radius = 0.02
        edge_mass = 0.01
        start_height = 3.0

        # Stiffness
        self.stretch_stiffness = wp.vec3(1.0, 1.0, 1.0)
        self.bend_stiffness = wp.vec3(0.5, 0.5, 0.5)

        # Compliance
        self.compliance = 1.0e-6
        self.compliance_factor = self.compliance / (self.sim_dt * self.sim_dt)

        self.gravity = wp.vec3(0.0, 0.0, -9.81)

        # Build model
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
        device = self.model.device

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, self.args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        # Particle inverse masses
        inv_mass_np = [0.0] + [1.0 / particle_mass] * (self.num_particles - 1)
        self.particle_inv_mass = wp.array(inv_mass_np, dtype=float, device=device)

        # Edge quaternions (rotate z to x for horizontal rod)
        angle = math.pi / 2.0
        q_init = wp.quat(0.0, math.sin(angle / 2.0), 0.0, math.cos(angle / 2.0))
        edge_q_init = [q_init] * self.num_edges
        self.edge_q = wp.array(edge_q_init, dtype=wp.quat, device=device)
        self.edge_q_new = wp.array(edge_q_init, dtype=wp.quat, device=device)

        # Edge inverse masses
        edge_inv_mass_np = [1.0 / edge_mass] * self.num_edges
        self.edge_inv_mass = wp.array(edge_inv_mass_np, dtype=float, device=device)

        # Rest lengths
        rest_length_np = [particle_spacing] * self.num_edges
        self.rest_length = wp.array(rest_length_np, dtype=float, device=device)

        # Rest Darboux vectors (identity for straight rod)
        rest_darboux_np = [wp.quat(0.0, 0.0, 0.0, 1.0)] * self.num_bend
        self.rest_darboux = wp.array(rest_darboux_np, dtype=wp.quat, device=device)

        # Constraint data
        self.stretch_violation = wp.zeros(self.num_edges, dtype=wp.vec3, device=device)
        self.bend_violation = wp.zeros(self.num_bend, dtype=wp.vec3, device=device)
        self.edge_d3 = wp.zeros(self.num_edges, dtype=wp.vec3, device=device)
        self.edge_rotation = wp.zeros(self.num_edges, dtype=wp.mat33, device=device)

        # Block system storage
        self.D_blocks = wp.zeros((self.num_edges, BLOCK, BLOCK), dtype=float, device=device)
        self.L_blocks = wp.zeros((self.num_edges - 1, BLOCK, BLOCK), dtype=float, device=device)
        self.U_blocks = wp.zeros((self.num_edges - 1, BLOCK, BLOCK), dtype=float, device=device)
        self.b_blocks = wp.zeros((self.num_edges, BLOCK), dtype=float, device=device)

        # Thomas workspace
        self.D_prime = wp.zeros((self.num_edges, BLOCK, BLOCK), dtype=float, device=device)
        self.b_prime = wp.zeros((self.num_edges, BLOCK), dtype=float, device=device)
        self.D_inv_work = wp.zeros((1, BLOCK, BLOCK), dtype=float, device=device)
        self.LU_work = wp.zeros((1, BLOCK, BLOCK), dtype=float, device=device)

        # Solution
        self.delta_lambda = wp.zeros((self.num_edges, BLOCK), dtype=float, device=device)

        # Temp buffers
        self.particle_q_pred = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)
        self.particle_q_temp = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)
        self.edge_q_delta = wp.zeros(self.num_edges, dtype=wp.quat, device=device)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

    def simulate(self):
        for _ in range(self.sim_substeps):
            wp.copy(self.particle_q_temp, self.state_0.particle_q)

            # Integrate
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
                outputs=[self.particle_q_pred, self.state_1.particle_qd],
                device=self.model.device,
            )
            wp.copy(self.state_1.particle_q, self.particle_q_pred)

            # Constraint iterations
            for _ in range(self.constraint_iterations):
                # Compute constraint data
                wp.launch(
                    kernel=compute_constraint_data_kernel,
                    dim=max(self.num_edges, self.num_bend),
                    inputs=[
                        self.state_1.particle_q,
                        self.edge_q,
                        self.rest_length,
                        self.rest_darboux,
                        self.num_edges,
                        self.num_bend,
                    ],
                    outputs=[
                        self.stretch_violation,
                        self.bend_violation,
                        self.edge_d3,
                        self.edge_rotation,
                    ],
                    device=self.model.device,
                )

                # Assemble block system
                wp.launch(
                    kernel=assemble_block_system_kernel,
                    dim=1,
                    inputs=[
                        self.particle_inv_mass,
                        self.edge_inv_mass,
                        self.stretch_violation,
                        self.bend_violation,
                        self.edge_d3,
                        self.edge_rotation,
                        self.edge_q,
                        self.rest_length,
                        self.stretch_stiffness,
                        self.bend_stiffness,
                        self.compliance_factor,
                        self.num_edges,
                        self.num_bend,
                    ],
                    outputs=[self.D_blocks, self.L_blocks, self.U_blocks, self.b_blocks],
                    device=self.model.device,
                )

                # Solve using block Thomas
                wp.launch(
                    kernel=block_thomas_solve_kernel,
                    dim=1,
                    inputs=[
                        self.D_blocks,
                        self.L_blocks,
                        self.U_blocks,
                        self.b_blocks,
                        self.num_edges,
                        self.D_prime,
                        self.b_prime,
                        self.D_inv_work,
                        self.LU_work,
                    ],
                    outputs=[self.delta_lambda],
                    device=self.model.device,
                )

                # Apply particle corrections from stretch
                wp.launch(
                    kernel=apply_particle_stretch_corrections_kernel,
                    dim=self.num_particles,
                    inputs=[
                        self.state_1.particle_q,
                        self.particle_inv_mass,
                        self.rest_length,
                        self.delta_lambda,
                        self.num_edges,
                    ],
                    outputs=[self.particle_q_pred],
                    device=self.model.device,
                )
                wp.copy(self.state_1.particle_q, self.particle_q_pred)

                # Apply quaternion corrections from stretch
                wp.launch(
                    kernel=apply_quaternion_stretch_corrections_kernel,
                    dim=self.num_edges,
                    inputs=[
                        self.edge_q,
                        self.edge_inv_mass,
                        self.rest_length,
                        self.delta_lambda,
                        self.num_edges,
                    ],
                    outputs=[self.edge_q_new],
                    device=self.model.device,
                )

                # Zero quaternion delta buffer
                wp.launch(
                    kernel=zero_quat_kernel,
                    dim=self.num_edges,
                    inputs=[self.edge_q_delta],
                    device=self.model.device,
                )

                # Accumulate bend corrections
                if self.num_bend > 0:
                    wp.launch(
                        kernel=apply_quaternion_bend_corrections_kernel,
                        dim=self.num_bend,
                        inputs=[
                            self.edge_q_new,
                            self.edge_inv_mass,
                            self.delta_lambda,
                            self.num_bend,
                        ],
                        outputs=[self.edge_q_delta],
                        device=self.model.device,
                    )

                    # Apply bend delta
                    wp.launch(
                        kernel=apply_quaternion_delta_kernel,
                        dim=self.num_edges,
                        inputs=[
                            self.edge_q_new,
                            self.edge_q_delta,
                            self.num_edges,
                        ],
                        outputs=[self.edge_q],
                        device=self.model.device,
                    )
                else:
                    self.edge_q, self.edge_q_new = self.edge_q_new, self.edge_q

            # Ground collision
            wp.launch(
                kernel=solve_ground_collision_kernel,
                dim=self.num_particles,
                inputs=[
                    self.state_1.particle_q,
                    self.particle_inv_mass,
                    self.model.particle_radius,
                    0.0,
                ],
                outputs=[self.particle_q_pred],
                device=self.model.device,
            )
            wp.copy(self.state_1.particle_q, self.particle_q_pred)

            # Update velocities
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
            "particles above ground",
            lambda q, qd: q[2] >= -0.01,
        )

        p_lower = wp.vec3(-2.0, -4.0, -0.1)
        p_upper = wp.vec3(6.0, 4.0, 5.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles within bounds",
            lambda q, qd: newton.utils.vec_inside_limits(q, p_lower, p_upper),
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()

    if isinstance(viewer, newton.viewer.ViewerGL):
        viewer.show_particles = True

    example = Example(viewer, args)

    newton.examples.run(example, args)
