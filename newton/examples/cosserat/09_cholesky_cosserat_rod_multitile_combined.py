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
# Example Cholesky Cosserat Rod - Multi-Tile with Combined Constraints
#
# Implements a TRUE combined stretch+bend system matching the C++ reference
# "Direct Position-Based Solver for Stiff Rods" paper.
#
# This is the most physically accurate formulation among the examples:
#   - Stretch and bend constraints are solved together (not sequentially)
#   - The coupling through shared quaternions is captured in the system matrix
#
# Each joint has 6 DOFs:
#   - 3 DOFs for stretch/shear constraint (position part)
#   - 3 DOFs for bend/twist constraint (orientation part)
#
# The 6×6 block per joint captures the coupling between stretch and bend
# that arises because both constraints affect the edge quaternion:
#
#   A_ii = [ K_stretch,     coupling^T ]
#          [ coupling,      K_bend     ]
#
# Where:
#   - K_stretch: from position Jacobians + quaternion stretch Jacobian
#   - K_bend: from quaternion bend Jacobians
#   - coupling: J_stretch_q * M_q^{-1} * J_bend_q^T (non-zero!)
#
# This matches Equation (19) and the system structure in the paper.
#
# Comparison with other examples:
#   - Examples 02/04/05: Jacobi iteration (stretch then bend sequentially)
#   - Examples 07/08: Cholesky solve (stretch and bend as separate systems)
#   - This example: Cholesky solve (combined stretch+bend system)
#
# Command: uv run -m newton.examples cosserat_09_cholesky_cosserat_rod_multitile_combined
#
###########################################################################

import math

import numpy as np

import warp as wp

import newton
import newton.examples

# Warp tile configuration
BLOCK_DIM = 128
JOINTS_PER_TILE = 16  # 16 joints per tile to keep system size reasonable
JOINT_DOFS = 6  # 3 stretch + 3 bend DOFs per joint
TILE = JOINTS_PER_TILE * JOINT_DOFS  # 96x96 tile

# Rod configuration
NUM_TILES = 8
NUM_PARTICLES = NUM_TILES * JOINTS_PER_TILE + 1  # 129 particles
NUM_JOINTS = NUM_PARTICLES - 1  # 128 joints (stretch constraints)
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
def compute_d3_jacobian_col(q: wp.quat, col: int) -> wp.vec3:
    """
    Compute column 'col' of the 3x4 Jacobian d(d3)/d(q).
    d3 = q * e3 * conj(q), so d(d3)/d(q) is a 3x4 matrix.

    For quaternion q = (x, y, z, w):
    d3 = [2(xz + wy), 2(yz - wx), w² - x² - y² + z²]

    Jacobian columns:
    col 0 (∂/∂x): [2z, -2w, -2x]
    col 1 (∂/∂y): [2w, 2z, -2y]
    col 2 (∂/∂z): [2x, 2y, 2z]
    col 3 (∂/∂w): [2y, -2x, 2w]
    """
    x, y, z, w = q[0], q[1], q[2], q[3]
    if col == 0:
        return wp.vec3(2.0 * z, -2.0 * w, -2.0 * x)
    elif col == 1:
        return wp.vec3(2.0 * w, 2.0 * z, -2.0 * y)
    elif col == 2:
        return wp.vec3(2.0 * x, 2.0 * y, 2.0 * z)
    else:  # col == 3
        return wp.vec3(2.0 * y, -2.0 * x, 2.0 * w)


@wp.func
def compute_G_matrix_col(q: wp.quat, col: int) -> wp.vec4:
    """
    Compute column 'col' of the 4x3 G matrix for quaternion derivatives.
    G maps angular velocity to quaternion rate: dq/dt = G * omega

    G = 0.5 * [ w,  z, -y ]
              [-z,  w,  x ]
              [ y, -x,  w ]
              [-x, -y, -z ]
    """
    x, y, z, w = q[0], q[1], q[2], q[3]
    if col == 0:
        return wp.vec4(0.5 * w, -0.5 * z, 0.5 * y, -0.5 * x)
    elif col == 1:
        return wp.vec4(0.5 * z, 0.5 * w, -0.5 * x, -0.5 * y)
    else:  # col == 2
        return wp.vec4(-0.5 * y, 0.5 * x, 0.5 * w, -0.5 * z)


@wp.func
def compute_bend_jacobian_row(q0: wp.quat, q1: wp.quat, avg_length: float, row: int, is_q0: int) -> wp.vec4:
    """
    Compute row 'row' of the 3x4 Jacobian d(omega)/d(q0) or d(omega)/d(q1).
    omega = 2/L * Im(conj(q0) * q1) is the Darboux vector.

    For d(omega)/d(q0) (is_q0=1):
    jOmega0 = 2/L * [-q1.w, -q1.z,  q1.y,  q1.x]
                    [ q1.z, -q1.w, -q1.x,  q1.y]
                    [-q1.y,  q1.x, -q1.w,  q1.z]

    For d(omega)/d(q1) (is_q0=0):
    jOmega1 = 2/L * [ q0.w,  q0.z, -q0.y, -q0.x]
                    [-q0.z,  q0.w,  q0.x, -q0.y]
                    [ q0.y, -q0.x,  q0.w, -q0.z]
    """
    scale = 2.0 / avg_length
    if is_q0 == 1:
        # d(omega)/d(q0)
        if row == 0:
            return scale * wp.vec4(-q1[3], -q1[2], q1[1], q1[0])
        elif row == 1:
            return scale * wp.vec4(q1[2], -q1[3], -q1[0], q1[1])
        else:
            return scale * wp.vec4(-q1[1], q1[0], -q1[3], q1[2])
    else:
        # d(omega)/d(q1)
        if row == 0:
            return scale * wp.vec4(q0[3], q0[2], -q0[1], -q0[0])
        elif row == 1:
            return scale * wp.vec4(-q0[2], q0[3], q0[0], -q0[1])
        else:
            return scale * wp.vec4(q0[1], -q0[0], q0[3], -q0[2])


@wp.func
def dot4(a: wp.vec4, b: wp.vec4) -> float:
    """Compute dot product of two vec4 values."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


@wp.func
def compute_J_bend(q0: wp.quat, q1: wp.quat, avg_length: float, is_q0: int) -> wp.mat33:
    """Compute 3x3 bend Jacobian block J_bend_q0 or J_bend_q1."""
    j0 = compute_bend_jacobian_row(q0, q1, avg_length, 0, is_q0)
    j1 = compute_bend_jacobian_row(q0, q1, avg_length, 1, is_q0)
    j2 = compute_bend_jacobian_row(q0, q1, avg_length, 2, is_q0)

    if is_q0 == 1:
        q = q0
    else:
        q = q1

    g0 = compute_G_matrix_col(q, 0)
    g1 = compute_G_matrix_col(q, 1)
    g2 = compute_G_matrix_col(q, 2)

    return wp.mat33(
        dot4(j0, g0), dot4(j0, g1), dot4(j0, g2),
        dot4(j1, g0), dot4(j1, g1), dot4(j1, g2),
        dot4(j2, g0), dot4(j2, g1), dot4(j2, g2),
    )


@wp.func
def compute_J_stretch(d3: wp.vec3) -> wp.mat33:
    """Compute 3x3 stretch Jacobian block for quaternion DOFs."""
    return wp.mat33(
        0.0, -2.0 * d3[2], 2.0 * d3[1],
        2.0 * d3[2], 0.0, -2.0 * d3[0],
        -2.0 * d3[1], 2.0 * d3[0], 0.0,
    )


@wp.func
def mat33_add(a: wp.mat33, b: wp.mat33) -> wp.mat33:
    return wp.mat33(
        a[0, 0] + b[0, 0], a[0, 1] + b[0, 1], a[0, 2] + b[0, 2],
        a[1, 0] + b[1, 0], a[1, 1] + b[1, 1], a[1, 2] + b[1, 2],
        a[2, 0] + b[2, 0], a[2, 1] + b[2, 1], a[2, 2] + b[2, 2],
    )


@wp.func
def mat33_sub(a: wp.mat33, b: wp.mat33) -> wp.mat33:
    return wp.mat33(
        a[0, 0] - b[0, 0], a[0, 1] - b[0, 1], a[0, 2] - b[0, 2],
        a[1, 0] - b[1, 0], a[1, 1] - b[1, 1], a[1, 2] - b[1, 2],
        a[2, 0] - b[2, 0], a[2, 1] - b[2, 1], a[2, 2] - b[2, 2],
    )


@wp.func
def mat33_mul(a: wp.mat33, b: wp.mat33) -> wp.mat33:
    return wp.mat33(
        a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0] + a[0, 2] * b[2, 0],
        a[0, 0] * b[0, 1] + a[0, 1] * b[1, 1] + a[0, 2] * b[2, 1],
        a[0, 0] * b[0, 2] + a[0, 1] * b[1, 2] + a[0, 2] * b[2, 2],
        a[1, 0] * b[0, 0] + a[1, 1] * b[1, 0] + a[1, 2] * b[2, 0],
        a[1, 0] * b[0, 1] + a[1, 1] * b[1, 1] + a[1, 2] * b[2, 1],
        a[1, 0] * b[0, 2] + a[1, 1] * b[1, 2] + a[1, 2] * b[2, 2],
        a[2, 0] * b[0, 0] + a[2, 1] * b[1, 0] + a[2, 2] * b[2, 0],
        a[2, 0] * b[0, 1] + a[2, 1] * b[1, 1] + a[2, 2] * b[2, 1],
        a[2, 0] * b[0, 2] + a[2, 1] * b[1, 2] + a[2, 2] * b[2, 2],
    )


@wp.func
def mat33_mul_vec3(a: wp.mat33, v: wp.vec3) -> wp.vec3:
    return wp.vec3(
        a[0, 0] * v[0] + a[0, 1] * v[1] + a[0, 2] * v[2],
        a[1, 0] * v[0] + a[1, 1] * v[1] + a[1, 2] * v[2],
        a[2, 0] * v[0] + a[2, 1] * v[1] + a[2, 2] * v[2],
    )


@wp.func
def mat33_inverse(a: wp.mat33) -> wp.mat33:
    det = (
        a[0, 0] * (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1])
        - a[0, 1] * (a[1, 0] * a[2, 2] - a[1, 2] * a[2, 0])
        + a[0, 2] * (a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0])
    )
    if wp.abs(det) < 1.0e-9:
        det = 1.0e-9
    inv_det = 1.0 / det

    return wp.mat33(
        (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1]) * inv_det,
        (a[0, 2] * a[2, 1] - a[0, 1] * a[2, 2]) * inv_det,
        (a[0, 1] * a[1, 2] - a[0, 2] * a[1, 1]) * inv_det,
        (a[1, 2] * a[2, 0] - a[1, 0] * a[2, 2]) * inv_det,
        (a[0, 0] * a[2, 2] - a[0, 2] * a[2, 0]) * inv_det,
        (a[0, 2] * a[1, 0] - a[0, 0] * a[1, 2]) * inv_det,
        (a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0]) * inv_det,
        (a[0, 1] * a[2, 0] - a[0, 0] * a[2, 1]) * inv_det,
        (a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]) * inv_det,
    )


@wp.func
def load_mat33(system_matrices: wp.array3d(dtype=float), tile_idx: int, row_base: int, col_base: int) -> wp.mat33:
    return wp.mat33(
        system_matrices[tile_idx, row_base + 0, col_base + 0],
        system_matrices[tile_idx, row_base + 0, col_base + 1],
        system_matrices[tile_idx, row_base + 0, col_base + 2],
        system_matrices[tile_idx, row_base + 1, col_base + 0],
        system_matrices[tile_idx, row_base + 1, col_base + 1],
        system_matrices[tile_idx, row_base + 1, col_base + 2],
        system_matrices[tile_idx, row_base + 2, col_base + 0],
        system_matrices[tile_idx, row_base + 2, col_base + 1],
        system_matrices[tile_idx, row_base + 2, col_base + 2],
    )


@wp.func
def store_mat33(system_matrices: wp.array3d(dtype=float), tile_idx: int, row_base: int, col_base: int, m: wp.mat33):
    system_matrices[tile_idx, row_base + 0, col_base + 0] = m[0, 0]
    system_matrices[tile_idx, row_base + 0, col_base + 1] = m[0, 1]
    system_matrices[tile_idx, row_base + 0, col_base + 2] = m[0, 2]
    system_matrices[tile_idx, row_base + 1, col_base + 0] = m[1, 0]
    system_matrices[tile_idx, row_base + 1, col_base + 1] = m[1, 1]
    system_matrices[tile_idx, row_base + 1, col_base + 2] = m[1, 2]
    system_matrices[tile_idx, row_base + 2, col_base + 0] = m[2, 0]
    system_matrices[tile_idx, row_base + 2, col_base + 1] = m[2, 1]
    system_matrices[tile_idx, row_base + 2, col_base + 2] = m[2, 2]


@wp.kernel
def assemble_combined_tridiagonal_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    rest_darboux: wp.array(dtype=wp.quat),
    stretch_stiffness: wp.vec3,
    bend_stiffness: wp.vec3,
    compliance_factor: float,
    num_joints: int,
    num_bend: int,
    # outputs
    block_lower: wp.array3d(dtype=float),
    block_diag: wp.array3d(dtype=float),
    block_upper: wp.array3d(dtype=float),
    block_rhs: wp.array2d(dtype=float),
):
    """Assemble block tridiagonal system for the full chain."""
    j = wp.tid()
    if j >= num_joints:
        return

    # zero blocks
    for r in range(JOINT_DOFS):
        block_rhs[j, r] = 0.0
        for c in range(JOINT_DOFS):
            block_lower[j, r, c] = 0.0
            block_diag[j, r, c] = 0.0
            block_upper[j, r, c] = 0.0

    p0 = particle_q[j]
    p1 = particle_q[j + 1]
    q0 = edge_q[j]
    L = rest_length[j]

    inv_m0 = particle_inv_mass[j]
    inv_m1 = particle_inv_mass[j + 1]
    inv_mq = edge_inv_mass[j]

    # Stretch RHS
    d3 = quat_rotate_e3(q0)
    edge_vec = p1 - p0
    gamma = edge_vec / L - d3
    gamma_local = wp.quat_rotate_inv(q0, gamma)
    gamma_local = wp.vec3(
        gamma_local[0] * stretch_stiffness[0],
        gamma_local[1] * stretch_stiffness[1],
        gamma_local[2] * stretch_stiffness[2],
    )
    gamma = wp.quat_rotate(q0, gamma_local)

    block_rhs[j, 0] = -gamma[0]
    block_rhs[j, 1] = -gamma[1]
    block_rhs[j, 2] = -gamma[2]

    # Stretch-stretch diagonal
    L_inv = 1.0 / L
    pos_contrib = (inv_m0 + inv_m1) * L_inv * L_inv
    quat_factor = 4.0 * inv_mq
    for i in range(3):
        for k in range(3):
            val = pos_contrib if i == k else 0.0
            val += quat_factor * ((1.0 if i == k else 0.0) - d3[i] * d3[k])
            if i == k:
                val += compliance_factor
            block_diag[j, i, k] = val

    J_stretch = compute_J_stretch(d3)

    # Bend block if present
    if j < num_bend:
        q1 = edge_q[j + 1]
        inv_mq1 = edge_inv_mass[j + 1]
        rest_d = rest_darboux[j]
        L_next = rest_length[j + 1]
        avg_length = 0.5 * (L + L_next)

        q0_conj = wp.quat(-q0[0], -q0[1], -q0[2], q0[3])
        omega_quat = wp.mul(q0_conj, q1)
        omega = wp.vec3(omega_quat[0], omega_quat[1], omega_quat[2]) * (2.0 / avg_length)

        rest_vec = wp.vec3(rest_d[0], rest_d[1], rest_d[2])
        kappa_plus = omega + rest_vec
        kappa_minus = omega - rest_vec
        if wp.dot(kappa_minus, kappa_minus) > wp.dot(kappa_plus, kappa_plus):
            kappa = kappa_plus
        else:
            kappa = kappa_minus

        block_rhs[j, 3] = -kappa[0]
        block_rhs[j, 4] = -kappa[1]
        block_rhs[j, 5] = -kappa[2]

        bend_compliance = wp.vec3(
            compliance_factor / wp.max(bend_stiffness[0], 1.0e-6) * (1.0 / avg_length),
            compliance_factor / wp.max(bend_stiffness[1], 1.0e-6) * (1.0 / avg_length),
            compliance_factor / wp.max(bend_stiffness[2], 1.0e-6) * (1.0 / avg_length),
        )

        J_bend_q0 = compute_J_bend(q0, q1, avg_length, 1)
        J_bend_q1 = compute_J_bend(q0, q1, avg_length, 0)

        for i in range(3):
            for k in range(3):
                val = inv_mq * (
                    J_bend_q0[i, 0] * J_bend_q0[k, 0]
                    + J_bend_q0[i, 1] * J_bend_q0[k, 1]
                    + J_bend_q0[i, 2] * J_bend_q0[k, 2]
                )
                val += inv_mq1 * (
                    J_bend_q1[i, 0] * J_bend_q1[k, 0]
                    + J_bend_q1[i, 1] * J_bend_q1[k, 1]
                    + J_bend_q1[i, 2] * J_bend_q1[k, 2]
                )
                if i == k:
                    val += bend_compliance[i]
                block_diag[j, 3 + i, 3 + k] = val

                coupling = inv_mq * (
                    J_stretch[i, 0] * J_bend_q0[k, 0]
                    + J_stretch[i, 1] * J_bend_q0[k, 1]
                    + J_stretch[i, 2] * J_bend_q0[k, 2]
                )
                block_diag[j, i, 3 + k] = coupling
                block_diag[j, 3 + k, i] = coupling
    else:
        for i in range(3):
            block_diag[j, 3 + i, 3 + i] = 1.0

    # Lower coupling (j with j-1)
    if j > 0:
        L_prev = rest_length[j - 1]
        coupling_pos = -particle_inv_mass[j] * (1.0 / L_prev) * (1.0 / L)
        for i in range(3):
            block_lower[j, i, i] = coupling_pos

        if j < num_bend and (j - 1) < num_bend:
            q_prev0 = edge_q[j - 1]
            q_prev1 = q0
            avg_length_prev = 0.5 * (L_prev + L)
            J_bend_q1_prev = compute_J_bend(q_prev0, q_prev1, avg_length_prev, 0)
            J_bend_q0 = compute_J_bend(q0, edge_q[j + 1], 0.5 * (L + rest_length[j + 1]), 1)

            for i in range(3):
                for k in range(3):
                    coupling_bb = inv_mq * (
                        J_bend_q0[i, 0] * J_bend_q1_prev[k, 0]
                        + J_bend_q0[i, 1] * J_bend_q1_prev[k, 1]
                        + J_bend_q0[i, 2] * J_bend_q1_prev[k, 2]
                    )
                    block_lower[j, 3 + i, 3 + k] = coupling_bb

                    coupling_sb = inv_mq * (
                        J_stretch[i, 0] * J_bend_q1_prev[k, 0]
                        + J_stretch[i, 1] * J_bend_q1_prev[k, 1]
                        + J_stretch[i, 2] * J_bend_q1_prev[k, 2]
                    )
                    block_lower[j, i, 3 + k] = coupling_sb

    # Upper coupling (j with j+1)
    if j < num_joints - 1:
        L_next = rest_length[j + 1]
        coupling_pos = -particle_inv_mass[j + 1] * (1.0 / L) * (1.0 / L_next)
        for i in range(3):
            block_upper[j, i, i] = coupling_pos

        if j < num_bend:
            q1 = edge_q[j + 1]
            inv_mq1 = edge_inv_mass[j + 1]
            d3_next = quat_rotate_e3(q1)
            J_stretch_next = compute_J_stretch(d3_next)

            J_bend_q1 = compute_J_bend(q0, q1, 0.5 * (L + L_next), 0)

            for i in range(3):
                for k in range(3):
                    coupling_bs = inv_mq1 * (
                        J_bend_q1[i, 0] * J_stretch_next[k, 0]
                        + J_bend_q1[i, 1] * J_stretch_next[k, 1]
                        + J_bend_q1[i, 2] * J_stretch_next[k, 2]
                    )
                    block_upper[j, 3 + i, k] = coupling_bs

            if (j + 1) < num_bend:
                q_next0 = edge_q[j + 1]
                q_next1 = edge_q[j + 2]
                L_next2 = rest_length[j + 2]
                avg_length_next = 0.5 * (L_next + L_next2)
                J_bend_q0_next = compute_J_bend(q_next0, q_next1, avg_length_next, 1)

                for i in range(3):
                    for k in range(3):
                        coupling_bb = inv_mq1 * (
                            J_bend_q1[i, 0] * J_bend_q0_next[k, 0]
                            + J_bend_q1[i, 1] * J_bend_q0_next[k, 1]
                            + J_bend_q1[i, 2] * J_bend_q0_next[k, 2]
                        )
                        block_upper[j, 3 + i, 3 + k] = coupling_bb


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
def assemble_combined_block_systems_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    rest_darboux: wp.array(dtype=wp.quat),
    stretch_stiffness: wp.vec3,
    bend_stiffness: wp.vec3,
    compliance_factor: float,
    num_joints: int,
    num_bend: int,
    joints_per_tile: int,
    # outputs
    system_matrices: wp.array3d(dtype=float),  # (NUM_TILES, TILE, TILE)
    system_rhs: wp.array2d(dtype=float),  # (NUM_TILES, TILE)
):
    """
    Assemble combined 6x6 block system matrices for stretch+bend constraints.

    Each joint i contributes a 6x6 block:
      - Rows 0-2: stretch constraint on edge i
      - Rows 3-5: bend constraint between edge i and i+1

    The system matrix has the structure:
      A = J * M^{-1} * J^T + compliance * I

    For joint i, the 6x6 diagonal block is:
      [ K_stretch + alpha,    coupling^T        ]
      [ coupling,             K_bend + alpha    ]

    Where coupling arises because both constraints affect q_i.
    """
    tile_idx = wp.tid()

    joint_start = tile_idx * joints_per_tile
    joint_end = wp.min(joint_start + joints_per_tile, num_joints)
    num_in_tile = joint_end - joint_start

    # Initialize to zero
    for i in range(TILE):
        system_rhs[tile_idx, i] = 0.0
        for j in range(TILE):
            system_matrices[tile_idx, i, j] = 0.0

    # For each joint in this tile
    for local_j in range(num_in_tile):
        global_j = joint_start + local_j
        row_base = local_j * JOINT_DOFS

        # Get particle and edge data
        p0 = particle_q[global_j]
        p1 = particle_q[global_j + 1]
        q0 = edge_q[global_j]
        L = rest_length[global_j]

        inv_m0 = particle_inv_mass[global_j]
        inv_m1 = particle_inv_mass[global_j + 1]
        inv_mq = edge_inv_mass[global_j]

        # ========== STRETCH CONSTRAINT (rows 0-2) ==========
        d3 = quat_rotate_e3(q0)

        # Constraint violation: gamma = (p1-p0)/L - d3
        edge_vec = p1 - p0
        gamma = edge_vec / L - d3

        # Transform to local frame and apply stiffness
        gamma_local = wp.quat_rotate_inv(q0, gamma)
        gamma_local = wp.vec3(
            gamma_local[0] * stretch_stiffness[0],
            gamma_local[1] * stretch_stiffness[1],
            gamma_local[2] * stretch_stiffness[2],
        )
        gamma = wp.quat_rotate(q0, gamma_local)

        # Store stretch RHS
        system_rhs[tile_idx, row_base + 0] = -gamma[0]
        system_rhs[tile_idx, row_base + 1] = -gamma[1]
        system_rhs[tile_idx, row_base + 2] = -gamma[2]

        # Stretch-stretch block (K_stretch): J_pos * M_pos^{-1} * J_pos^T + J_q * M_q^{-1} * J_q^T
        # Position part: (inv_m0 + inv_m1) / L^2 * I
        L_inv = 1.0 / L
        pos_contrib = (inv_m0 + inv_m1) * L_inv * L_inv

        # Quaternion part for stretch: J_gamma_q * inv_mq * J_gamma_q^T
        # J_gamma_q = -d(d3)/d(q) * G (3x3 matrix after multiplying out)
        # For d3 rotation, the effective 3x3 is: 4 * inv_mq * (I - d3 * d3^T)
        quat_factor = 4.0 * inv_mq

        for i in range(3):
            for j in range(3):
                row = row_base + i
                col = row_base + j

                pos_val = pos_contrib if i == j else 0.0
                quat_val = quat_factor * ((1.0 if i == j else 0.0) - d3[i] * d3[j])
                compliance_val = compliance_factor if i == j else 0.0

                system_matrices[tile_idx, row, col] = pos_val + quat_val + compliance_val

        # ========== BEND CONSTRAINT (rows 3-5) ==========
        # Only if this joint has a bend constraint (not the last edge)
        if global_j < num_bend:
            q1 = edge_q[global_j + 1]
            inv_mq1 = edge_inv_mass[global_j + 1]
            rest_d = rest_darboux[global_j]

            # Compute Darboux vector omega = 2/L * Im(conj(q0) * q1)
            L_next = rest_length[global_j + 1]
            avg_length = 0.5 * (L + L_next)
            q0_conj = wp.quat(-q0[0], -q0[1], -q0[2], q0[3])
            omega_quat = wp.mul(q0_conj, q1)
            omega = wp.vec3(omega_quat[0], omega_quat[1], omega_quat[2]) * (2.0 / avg_length)

            # Handle quaternion double cover (rest vector is in curvature units)
            rest_vec = wp.vec3(rest_d[0], rest_d[1], rest_d[2])
            kappa_plus = omega + rest_vec
            kappa_minus = omega - rest_vec

            norm_plus_sq = wp.dot(kappa_plus, kappa_plus)
            norm_minus_sq = wp.dot(kappa_minus, kappa_minus)

            if norm_minus_sq > norm_plus_sq:
                kappa = kappa_plus
            else:
                kappa = kappa_minus

            # Store bend RHS
            system_rhs[tile_idx, row_base + 3] = -kappa[0]
            system_rhs[tile_idx, row_base + 4] = -kappa[1]
            system_rhs[tile_idx, row_base + 5] = -kappa[2]

            bend_compliance = wp.vec3(
                compliance_factor / wp.max(bend_stiffness[0], 1.0e-6) * (1.0 / avg_length),
                compliance_factor / wp.max(bend_stiffness[1], 1.0e-6) * (1.0 / avg_length),
                compliance_factor / wp.max(bend_stiffness[2], 1.0e-6) * (1.0 / avg_length),
            )

            # Bend-bend block: J_bend_q0 * inv_mq * J_bend_q0^T + J_bend_q1 * inv_mq1 * J_bend_q1^T
            J_bend_q0 = compute_J_bend(q0, q1, avg_length, 1)
            J_bend_q1 = compute_J_bend(q0, q1, avg_length, 0)

            for i in range(3):
                for j in range(3):
                    val = inv_mq * (
                        J_bend_q0[i, 0] * J_bend_q0[j, 0]
                        + J_bend_q0[i, 1] * J_bend_q0[j, 1]
                        + J_bend_q0[i, 2] * J_bend_q0[j, 2]
                    )
                    val += inv_mq1 * (
                        J_bend_q1[i, 0] * J_bend_q1[j, 0]
                        + J_bend_q1[i, 1] * J_bend_q1[j, 1]
                        + J_bend_q1[i, 2] * J_bend_q1[j, 2]
                    )
                    if i == j:
                        val += bend_compliance[i]

                    row = row_base + 3 + i
                    col = row_base + 3 + j
                    system_matrices[tile_idx, row, col] = val

            # ========== COUPLING BLOCK (stretch-bend) ==========
            # coupling = J_stretch_q * inv_mq * J_bend_q0^T
            J_stretch = compute_J_stretch(d3)
            for i in range(3):
                for j in range(3):
                    coupling = inv_mq * (
                        J_stretch[i, 0] * J_bend_q0[j, 0]
                        + J_stretch[i, 1] * J_bend_q0[j, 1]
                        + J_stretch[i, 2] * J_bend_q0[j, 2]
                    )
                    system_matrices[tile_idx, row_base + i, row_base + 3 + j] = coupling
                    system_matrices[tile_idx, row_base + 3 + j, row_base + i] = coupling

            # Stretch constraint j couples to bend constraint j-1 via shared q_j
            if local_j > 0:
                q_prev0 = edge_q[global_j - 1]
                q_prev1 = q0
                L_prev = rest_length[global_j - 1]
                avg_length_prev = 0.5 * (L_prev + L)
                J_bend_q1_prev = compute_J_bend(q_prev0, q_prev1, avg_length_prev, 0)
                prev_row_base = (local_j - 1) * JOINT_DOFS
                for i in range(3):
                    for j in range(3):
                        coupling = inv_mq * (
                            J_stretch[i, 0] * J_bend_q1_prev[j, 0]
                            + J_stretch[i, 1] * J_bend_q1_prev[j, 1]
                            + J_stretch[i, 2] * J_bend_q1_prev[j, 2]
                        )
                        system_matrices[tile_idx, row_base + i, prev_row_base + 3 + j] += coupling
                        system_matrices[tile_idx, prev_row_base + 3 + j, row_base + i] += coupling

            # Stretch constraint j+1 couples to bend constraint j via shared q_{j+1}
            if local_j + 1 < num_in_tile:
                d3_next = quat_rotate_e3(q1)
                J_stretch_next = compute_J_stretch(d3_next)
                next_row_base = (local_j + 1) * JOINT_DOFS
                for i in range(3):
                    for j in range(3):
                        coupling = inv_mq1 * (
                            J_stretch_next[i, 0] * J_bend_q1[j, 0]
                            + J_stretch_next[i, 1] * J_bend_q1[j, 1]
                            + J_stretch_next[i, 2] * J_bend_q1[j, 2]
                        )
                        system_matrices[tile_idx, next_row_base + i, row_base + 3 + j] += coupling
                        system_matrices[tile_idx, row_base + 3 + j, next_row_base + i] += coupling

            # Bend constraint j couples to bend constraint j+1 via shared edge quaternion
            if local_j + 1 < num_in_tile and global_j + 1 < num_bend:
                q_next0 = edge_q[global_j + 1]
                q_next1 = edge_q[global_j + 2]
                L_next2 = rest_length[global_j + 2]
                avg_length_next = 0.5 * (L_next + L_next2)

                J_bend_q0_next = compute_J_bend(q_next0, q_next1, avg_length_next, 1)
                next_row_base = (local_j + 1) * JOINT_DOFS
                for i in range(3):
                    for j in range(3):
                        coupling = inv_mq1 * (
                            J_bend_q1[i, 0] * J_bend_q0_next[j, 0]
                            + J_bend_q1[i, 1] * J_bend_q0_next[j, 1]
                            + J_bend_q1[i, 2] * J_bend_q0_next[j, 2]
                        )
                        system_matrices[tile_idx, row_base + 3 + i, next_row_base + 3 + j] += coupling
                        system_matrices[tile_idx, next_row_base + 3 + j, row_base + 3 + i] += coupling
        else:
            # Last joint: no bend constraint, just pad with identity
            for i in range(3):
                system_matrices[tile_idx, row_base + 3 + i, row_base + 3 + i] = 1.0

        # ========== OFF-DIAGONAL COUPLING TO NEIGHBORING JOINTS ==========
        # Stretch constraint j couples to stretch constraint j+1 via shared particle
        if local_j + 1 < num_in_tile:
            L_next = rest_length[global_j + 1]
            coupling_pos = -particle_inv_mass[global_j + 1] * L_inv / L_next

            next_row_base = (local_j + 1) * JOINT_DOFS
            for i in range(3):
                # Position coupling (stretch-stretch)
                system_matrices[tile_idx, row_base + i, next_row_base + i] += coupling_pos
                system_matrices[tile_idx, next_row_base + i, row_base + i] += coupling_pos

        # Bend constraint j couples to bend constraint j+1 via shared edge quaternion
        # (handled above using J_bend matrices)

    # Enforce strict diagonal dominance for active rows (SPD guard for Cholesky)
    active_rows = num_in_tile * JOINT_DOFS
    for i in range(active_rows):
        row_sum = float(0.0)
        for j in range(active_rows):
            if i != j:
                row_sum += wp.abs(system_matrices[tile_idx, i, j])
        diag = system_matrices[tile_idx, i, i]
        if diag <= row_sum:
            system_matrices[tile_idx, i, i] = row_sum + 1.0e-6

    # Pad remaining rows with identity
    for i in range(num_in_tile * JOINT_DOFS, TILE):
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
def thomas_solve_block_tridiagonal_kernel(
    system_matrices: wp.array3d(dtype=float),
    system_rhs: wp.array2d(dtype=float),
    num_joints: int,
    joints_per_tile: int,
    # output
    x: wp.array2d(dtype=float),
):
    """Solve block tridiagonal system using block Thomas (single thread per tile)."""
    tile_idx = wp.tid()

    joint_start = tile_idx * joints_per_tile
    joint_end = wp.min(joint_start + joints_per_tile, num_joints)
    num_in_tile = joint_end - joint_start
    if num_in_tile <= 0:
        return

    # Forward elimination: overwrite super-diagonal with C' and RHS with d'
    for i in range(num_in_tile):
        row_base = i * JOINT_DOFS

        B11 = load_mat33(system_matrices, tile_idx, row_base + 0, row_base + 0)
        B12 = load_mat33(system_matrices, tile_idx, row_base + 0, row_base + 3)
        B21 = load_mat33(system_matrices, tile_idx, row_base + 3, row_base + 0)
        B22 = load_mat33(system_matrices, tile_idx, row_base + 3, row_base + 3)

        d1 = wp.vec3(
            system_rhs[tile_idx, row_base + 0],
            system_rhs[tile_idx, row_base + 1],
            system_rhs[tile_idx, row_base + 2],
        )
        d2 = wp.vec3(
            system_rhs[tile_idx, row_base + 3],
            system_rhs[tile_idx, row_base + 4],
            system_rhs[tile_idx, row_base + 5],
        )

        if i > 0:
            # A block from previous row
            prev_row_base = (i - 1) * JOINT_DOFS
            A11 = load_mat33(system_matrices, tile_idx, row_base + 0, prev_row_base + 0)
            A12 = load_mat33(system_matrices, tile_idx, row_base + 0, prev_row_base + 3)
            A21 = load_mat33(system_matrices, tile_idx, row_base + 3, prev_row_base + 0)
            A22 = load_mat33(system_matrices, tile_idx, row_base + 3, prev_row_base + 3)

            # C' from previous step (stored in super-diagonal block)
            C11_prev = load_mat33(system_matrices, tile_idx, prev_row_base + 0, row_base + 0)
            C12_prev = load_mat33(system_matrices, tile_idx, prev_row_base + 0, row_base + 3)
            C21_prev = load_mat33(system_matrices, tile_idx, prev_row_base + 3, row_base + 0)
            C22_prev = load_mat33(system_matrices, tile_idx, prev_row_base + 3, row_base + 3)

            # P = A * C_prev
            P11 = mat33_add(mat33_mul(A11, C11_prev), mat33_mul(A12, C21_prev))
            P12 = mat33_add(mat33_mul(A11, C12_prev), mat33_mul(A12, C22_prev))
            P21 = mat33_add(mat33_mul(A21, C11_prev), mat33_mul(A22, C21_prev))
            P22 = mat33_add(mat33_mul(A21, C12_prev), mat33_mul(A22, C22_prev))

            # M = B - P
            B11 = mat33_sub(B11, P11)
            B12 = mat33_sub(B12, P12)
            B21 = mat33_sub(B21, P21)
            B22 = mat33_sub(B22, P22)

            # d_tilde = d - A * d_prev
            d1_prev = wp.vec3(
                system_rhs[tile_idx, prev_row_base + 0],
                system_rhs[tile_idx, prev_row_base + 1],
                system_rhs[tile_idx, prev_row_base + 2],
            )
            d2_prev = wp.vec3(
                system_rhs[tile_idx, prev_row_base + 3],
                system_rhs[tile_idx, prev_row_base + 4],
                system_rhs[tile_idx, prev_row_base + 5],
            )
            v1 = mat33_mul_vec3(A11, d1_prev) + mat33_mul_vec3(A12, d2_prev)
            v2 = mat33_mul_vec3(A21, d1_prev) + mat33_mul_vec3(A22, d2_prev)
            d1 = d1 - v1
            d2 = d2 - v2

        invB11 = mat33_inverse(B11)
        S = mat33_sub(B22, mat33_mul(B21, mat33_mul(invB11, B12)))
        invS = mat33_inverse(S)

        # Solve for C' if needed
        if i < num_in_tile - 1:
            col_base = (i + 1) * JOINT_DOFS
            C11 = load_mat33(system_matrices, tile_idx, row_base + 0, col_base + 0)
            C12 = load_mat33(system_matrices, tile_idx, row_base + 0, col_base + 3)
            C21 = load_mat33(system_matrices, tile_idx, row_base + 3, col_base + 0)
            C22 = load_mat33(system_matrices, tile_idx, row_base + 3, col_base + 3)

            t11 = mat33_mul(invB11, C11)
            t12 = mat33_mul(invB11, C12)
            rhs21 = mat33_sub(C21, mat33_mul(B21, t11))
            rhs22 = mat33_sub(C22, mat33_mul(B21, t12))
            X21 = mat33_mul(invS, rhs21)
            X22 = mat33_mul(invS, rhs22)
            X11 = mat33_sub(t11, mat33_mul(invB11, mat33_mul(B12, X21)))
            X12 = mat33_sub(t12, mat33_mul(invB11, mat33_mul(B12, X22)))

            store_mat33(system_matrices, tile_idx, row_base + 0, col_base + 0, X11)
            store_mat33(system_matrices, tile_idx, row_base + 0, col_base + 3, X12)
            store_mat33(system_matrices, tile_idx, row_base + 3, col_base + 0, X21)
            store_mat33(system_matrices, tile_idx, row_base + 3, col_base + 3, X22)

        # Solve for d'
        t1 = mat33_mul_vec3(invB11, d1)
        rhs2 = d2 - mat33_mul_vec3(B21, t1)
        x2 = mat33_mul_vec3(invS, rhs2)
        x1 = t1 - mat33_mul_vec3(invB11, mat33_mul_vec3(B12, x2))

        system_rhs[tile_idx, row_base + 0] = x1[0]
        system_rhs[tile_idx, row_base + 1] = x1[1]
        system_rhs[tile_idx, row_base + 2] = x1[2]
        system_rhs[tile_idx, row_base + 3] = x2[0]
        system_rhs[tile_idx, row_base + 4] = x2[1]
        system_rhs[tile_idx, row_base + 5] = x2[2]

    # Back substitution
    for i in range(num_in_tile - 1, -1, -1):
        row_base = i * JOINT_DOFS
        x1 = wp.vec3(
            system_rhs[tile_idx, row_base + 0],
            system_rhs[tile_idx, row_base + 1],
            system_rhs[tile_idx, row_base + 2],
        )
        x2 = wp.vec3(
            system_rhs[tile_idx, row_base + 3],
            system_rhs[tile_idx, row_base + 4],
            system_rhs[tile_idx, row_base + 5],
        )

        if i < num_in_tile - 1:
            col_base = (i + 1) * JOINT_DOFS
            C11 = load_mat33(system_matrices, tile_idx, row_base + 0, col_base + 0)
            C12 = load_mat33(system_matrices, tile_idx, row_base + 0, col_base + 3)
            C21 = load_mat33(system_matrices, tile_idx, row_base + 3, col_base + 0)
            C22 = load_mat33(system_matrices, tile_idx, row_base + 3, col_base + 3)

            x1_next = wp.vec3(
                x[tile_idx, col_base + 0],
                x[tile_idx, col_base + 1],
                x[tile_idx, col_base + 2],
            )
            x2_next = wp.vec3(
                x[tile_idx, col_base + 3],
                x[tile_idx, col_base + 4],
                x[tile_idx, col_base + 5],
            )

            v1 = mat33_mul_vec3(C11, x1_next) + mat33_mul_vec3(C12, x2_next)
            v2 = mat33_mul_vec3(C21, x1_next) + mat33_mul_vec3(C22, x2_next)
            x1 = x1 - v1
            x2 = x2 - v2

        x[tile_idx, row_base + 0] = x1[0]
        x[tile_idx, row_base + 1] = x1[1]
        x[tile_idx, row_base + 2] = x1[2]
        x[tile_idx, row_base + 3] = x2[0]
        x[tile_idx, row_base + 4] = x2[1]
        x[tile_idx, row_base + 5] = x2[2]

    # Zero remaining entries
    active_rows = num_in_tile * JOINT_DOFS
    for i in range(active_rows, TILE):
        x[tile_idx, i] = 0.0


@wp.kernel
def thomas_solve_block_tridiagonal_global_kernel(
    block_lower: wp.array3d(dtype=float),
    block_diag: wp.array3d(dtype=float),
    block_upper: wp.array3d(dtype=float),
    block_rhs: wp.array2d(dtype=float),
    num_joints: int,
    # output
    x: wp.array2d(dtype=float),
):
    """Solve global block tridiagonal system using block Thomas (single thread)."""
    tid = wp.tid()
    if tid > 0:
        return

    # Forward elimination
    for i in range(num_joints):
        B11 = load_mat33(block_diag, i, 0, 0)
        B12 = load_mat33(block_diag, i, 0, 3)
        B21 = load_mat33(block_diag, i, 3, 0)
        B22 = load_mat33(block_diag, i, 3, 3)

        d1 = wp.vec3(block_rhs[i, 0], block_rhs[i, 1], block_rhs[i, 2])
        d2 = wp.vec3(block_rhs[i, 3], block_rhs[i, 4], block_rhs[i, 5])

        if i > 0:
            A11 = load_mat33(block_lower, i, 0, 0)
            A12 = load_mat33(block_lower, i, 0, 3)
            A21 = load_mat33(block_lower, i, 3, 0)
            A22 = load_mat33(block_lower, i, 3, 3)

            C11_prev = load_mat33(block_upper, i - 1, 0, 0)
            C12_prev = load_mat33(block_upper, i - 1, 0, 3)
            C21_prev = load_mat33(block_upper, i - 1, 3, 0)
            C22_prev = load_mat33(block_upper, i - 1, 3, 3)

            P11 = mat33_add(mat33_mul(A11, C11_prev), mat33_mul(A12, C21_prev))
            P12 = mat33_add(mat33_mul(A11, C12_prev), mat33_mul(A12, C22_prev))
            P21 = mat33_add(mat33_mul(A21, C11_prev), mat33_mul(A22, C21_prev))
            P22 = mat33_add(mat33_mul(A21, C12_prev), mat33_mul(A22, C22_prev))

            B11 = mat33_sub(B11, P11)
            B12 = mat33_sub(B12, P12)
            B21 = mat33_sub(B21, P21)
            B22 = mat33_sub(B22, P22)

            d1_prev = wp.vec3(block_rhs[i - 1, 0], block_rhs[i - 1, 1], block_rhs[i - 1, 2])
            d2_prev = wp.vec3(block_rhs[i - 1, 3], block_rhs[i - 1, 4], block_rhs[i - 1, 5])
            v1 = mat33_mul_vec3(A11, d1_prev) + mat33_mul_vec3(A12, d2_prev)
            v2 = mat33_mul_vec3(A21, d1_prev) + mat33_mul_vec3(A22, d2_prev)
            d1 = d1 - v1
            d2 = d2 - v2

        invB11 = mat33_inverse(B11)
        S = mat33_sub(B22, mat33_mul(B21, mat33_mul(invB11, B12)))
        invS = mat33_inverse(S)

        if i < num_joints - 1:
            C11 = load_mat33(block_upper, i, 0, 0)
            C12 = load_mat33(block_upper, i, 0, 3)
            C21 = load_mat33(block_upper, i, 3, 0)
            C22 = load_mat33(block_upper, i, 3, 3)

            t11 = mat33_mul(invB11, C11)
            t12 = mat33_mul(invB11, C12)
            rhs21 = mat33_sub(C21, mat33_mul(B21, t11))
            rhs22 = mat33_sub(C22, mat33_mul(B21, t12))
            X21 = mat33_mul(invS, rhs21)
            X22 = mat33_mul(invS, rhs22)
            X11 = mat33_sub(t11, mat33_mul(invB11, mat33_mul(B12, X21)))
            X12 = mat33_sub(t12, mat33_mul(invB11, mat33_mul(B12, X22)))

            store_mat33(block_upper, i, 0, 0, X11)
            store_mat33(block_upper, i, 0, 3, X12)
            store_mat33(block_upper, i, 3, 0, X21)
            store_mat33(block_upper, i, 3, 3, X22)

        t1 = mat33_mul_vec3(invB11, d1)
        rhs2 = d2 - mat33_mul_vec3(B21, t1)
        x2 = mat33_mul_vec3(invS, rhs2)
        x1 = t1 - mat33_mul_vec3(invB11, mat33_mul_vec3(B12, x2))

        block_rhs[i, 0] = x1[0]
        block_rhs[i, 1] = x1[1]
        block_rhs[i, 2] = x1[2]
        block_rhs[i, 3] = x2[0]
        block_rhs[i, 4] = x2[1]
        block_rhs[i, 5] = x2[2]

    # Back substitution
    for i in range(num_joints - 1, -1, -1):
        x1 = wp.vec3(block_rhs[i, 0], block_rhs[i, 1], block_rhs[i, 2])
        x2 = wp.vec3(block_rhs[i, 3], block_rhs[i, 4], block_rhs[i, 5])

        if i < num_joints - 1:
            C11 = load_mat33(block_upper, i, 0, 0)
            C12 = load_mat33(block_upper, i, 0, 3)
            C21 = load_mat33(block_upper, i, 3, 0)
            C22 = load_mat33(block_upper, i, 3, 3)

            x1_next = wp.vec3(x[i + 1, 0], x[i + 1, 1], x[i + 1, 2])
            x2_next = wp.vec3(x[i + 1, 3], x[i + 1, 4], x[i + 1, 5])

            v1 = mat33_mul_vec3(C11, x1_next) + mat33_mul_vec3(C12, x2_next)
            v2 = mat33_mul_vec3(C21, x1_next) + mat33_mul_vec3(C22, x2_next)
            x1 = x1 - v1
            x2 = x2 - v2

        x[i, 0] = x1[0]
        x[i, 1] = x1[1]
        x[i, 2] = x1[2]
        x[i, 3] = x2[0]
        x[i, 4] = x2[1]
        x[i, 5] = x2[2]


@wp.kernel
def zero_vec3_kernel(arr: wp.array(dtype=wp.vec3)):
    """Zero out a vec3 array."""
    tid = wp.tid()
    arr[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def apply_combined_block_corrections_kernel(
    particle_inv_mass: wp.array(dtype=float),
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    delta_lambdas: wp.array2d(dtype=float),
    num_joints: int,
    num_bend: int,
    joints_per_tile: int,
    num_tiles: int,
    # outputs (accumulated via atomics)
    particle_corrections: wp.array(dtype=wp.vec3),
    edge_omega: wp.array(dtype=wp.vec3),
):
    """
    Apply combined corrections from all tiles.

    Each joint's 6 DOFs contribute:
      - delta_lambda[0:3]: stretch corrections -> particle positions + edge quaternion
      - delta_lambda[3:6]: bend corrections -> edge quaternions
    """
    tile_idx = wp.tid()

    joint_start = tile_idx * joints_per_tile
    joint_end = wp.min(joint_start + joints_per_tile, num_joints)
    num_in_tile = joint_end - joint_start

    for local_j in range(num_in_tile):
        global_j = joint_start + local_j
        row_base = local_j * JOINT_DOFS

        L = rest_length[global_j]
        q0 = edge_q[global_j]
        d3 = quat_rotate_e3(q0)

        # Get stretch delta_lambda (first 3 DOFs)
        dl_stretch = wp.vec3(
            delta_lambdas[tile_idx, row_base + 0],
            delta_lambdas[tile_idx, row_base + 1],
            delta_lambdas[tile_idx, row_base + 2],
        )

        # Apply stretch corrections to particles
        inv_m0 = particle_inv_mass[global_j]
        if inv_m0 > 0.0:
            corr0 = -dl_stretch * (inv_m0 / L)
            wp.atomic_add(particle_corrections, global_j, corr0)

        inv_m1 = particle_inv_mass[global_j + 1]
        if inv_m1 > 0.0:
            corr1 = dl_stretch * (inv_m1 / L)
            wp.atomic_add(particle_corrections, global_j + 1, corr1)

        # Apply stretch corrections to edge quaternion (as angular velocity)
        inv_mq = edge_inv_mass[global_j]
        if inv_mq > 0.0:
            # omega_stretch = 2 * inv_mq * cross(dl_stretch, d3)
            omega_stretch = 2.0 * inv_mq * wp.cross(dl_stretch, d3)
            wp.atomic_add(edge_omega, global_j, omega_stretch)

        # Get bend delta_lambda (last 3 DOFs)
        if global_j < num_bend:
            dl_bend = wp.vec3(
                delta_lambdas[tile_idx, row_base + 3],
                delta_lambdas[tile_idx, row_base + 4],
                delta_lambdas[tile_idx, row_base + 5],
            )

            q1 = edge_q[global_j + 1]
            L_next = rest_length[global_j + 1]
            avg_length = 0.5 * (L + L_next)
            J_bend_q0 = compute_J_bend(q0, q1, avg_length, 1)
            J_bend_q1 = compute_J_bend(q0, q1, avg_length, 0)

            # Apply bend corrections to edge quaternions
            # q0 gets correction from its role as the "first" quaternion
            if inv_mq > 0.0:
                omega0 = wp.vec3(
                    J_bend_q0[0, 0] * dl_bend[0] + J_bend_q0[1, 0] * dl_bend[1] + J_bend_q0[2, 0] * dl_bend[2],
                    J_bend_q0[0, 1] * dl_bend[0] + J_bend_q0[1, 1] * dl_bend[1] + J_bend_q0[2, 1] * dl_bend[2],
                    J_bend_q0[0, 2] * dl_bend[0] + J_bend_q0[1, 2] * dl_bend[1] + J_bend_q0[2, 2] * dl_bend[2],
                )
                omega0 = omega0 * inv_mq
                wp.atomic_add(edge_omega, global_j, omega0)

            # q1 (next edge) gets correction from its role as the "second" quaternion
            inv_mq1 = edge_inv_mass[global_j + 1]
            if inv_mq1 > 0.0:
                omega1 = wp.vec3(
                    J_bend_q1[0, 0] * dl_bend[0] + J_bend_q1[1, 0] * dl_bend[1] + J_bend_q1[2, 0] * dl_bend[2],
                    J_bend_q1[0, 1] * dl_bend[0] + J_bend_q1[1, 1] * dl_bend[1] + J_bend_q1[2, 1] * dl_bend[2],
                    J_bend_q1[0, 2] * dl_bend[0] + J_bend_q1[1, 2] * dl_bend[1] + J_bend_q1[2, 2] * dl_bend[2],
                )
                omega1 = omega1 * inv_mq1
                wp.atomic_add(edge_omega, global_j + 1, omega1)


@wp.kernel
def apply_combined_global_corrections_kernel(
    particle_inv_mass: wp.array(dtype=float),
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    delta_lambdas: wp.array2d(dtype=float),
    num_joints: int,
    num_bend: int,
    # outputs (accumulated via atomics)
    particle_corrections: wp.array(dtype=wp.vec3),
    edge_omega: wp.array(dtype=wp.vec3),
):
    """Apply combined corrections for the global block solve."""
    j = wp.tid()
    if j >= num_joints:
        return

    L = rest_length[j]
    q0 = edge_q[j]
    d3 = quat_rotate_e3(q0)

    dl_stretch = wp.vec3(
        delta_lambdas[j, 0],
        delta_lambdas[j, 1],
        delta_lambdas[j, 2],
    )

    inv_m0 = particle_inv_mass[j]
    if inv_m0 > 0.0:
        corr0 = -dl_stretch * (inv_m0 / L)
        wp.atomic_add(particle_corrections, j, corr0)

    inv_m1 = particle_inv_mass[j + 1]
    if inv_m1 > 0.0:
        corr1 = dl_stretch * (inv_m1 / L)
        wp.atomic_add(particle_corrections, j + 1, corr1)

    inv_mq = edge_inv_mass[j]
    if inv_mq > 0.0:
        omega_stretch = 2.0 * inv_mq * wp.cross(dl_stretch, d3)
        wp.atomic_add(edge_omega, j, omega_stretch)

    if j < num_bend:
        dl_bend = wp.vec3(
            delta_lambdas[j, 3],
            delta_lambdas[j, 4],
            delta_lambdas[j, 5],
        )
        q1 = edge_q[j + 1]
        L_next = rest_length[j + 1]
        avg_length = 0.5 * (L + L_next)
        J_bend_q0 = compute_J_bend(q0, q1, avg_length, 1)
        J_bend_q1 = compute_J_bend(q0, q1, avg_length, 0)

        if inv_mq > 0.0:
            omega0 = wp.vec3(
                J_bend_q0[0, 0] * dl_bend[0] + J_bend_q0[1, 0] * dl_bend[1] + J_bend_q0[2, 0] * dl_bend[2],
                J_bend_q0[0, 1] * dl_bend[0] + J_bend_q0[1, 1] * dl_bend[1] + J_bend_q0[2, 1] * dl_bend[2],
                J_bend_q0[0, 2] * dl_bend[0] + J_bend_q0[1, 2] * dl_bend[1] + J_bend_q0[2, 2] * dl_bend[2],
            )
            omega0 = omega0 * inv_mq
            wp.atomic_add(edge_omega, j, omega0)

        inv_mq1 = edge_inv_mass[j + 1]
        if inv_mq1 > 0.0:
            omega1 = wp.vec3(
                J_bend_q1[0, 0] * dl_bend[0] + J_bend_q1[1, 0] * dl_bend[1] + J_bend_q1[2, 0] * dl_bend[2],
                J_bend_q1[0, 1] * dl_bend[0] + J_bend_q1[1, 1] * dl_bend[1] + J_bend_q1[2, 1] * dl_bend[2],
                J_bend_q1[0, 2] * dl_bend[0] + J_bend_q1[1, 2] * dl_bend[1] + J_bend_q1[2, 2] * dl_bend[2],
            )
            omega1 = omega1 * inv_mq1
            wp.atomic_add(edge_omega, j + 1, omega1)


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

    # Store rest curvature directly in the vector part (units: 1/length).
    rest_darboux[tid] = wp.quat(rest_bend_d1, rest_bend_d2, rest_twist, 0.0)


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
        self.constraint_iterations = 3  # Fewer iterations needed with combined solver

        self.viewer = viewer
        self.args = args

        # Rod parameters
        self.num_particles = NUM_PARTICLES
        self.num_joints = NUM_JOINTS
        self.num_bend = NUM_BEND
        self.num_tiles = NUM_TILES
        self.joints_per_tile = JOINTS_PER_TILE

        particle_spacing = 0.05
        particle_mass = 0.1
        particle_radius = 0.015
        edge_mass = 0.01
        start_height = 5.0

        # Stiffness parameters (per-axis like C++ reference)
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
        edge_q_init = [q_init] * self.num_joints
        self.edge_q = wp.array(edge_q_init, dtype=wp.quat, device=device)
        self.edge_q_new = wp.array(edge_q_init, dtype=wp.quat, device=device)

        # Edge inverse masses
        edge_inv_mass_np = [1.0 / edge_mass] * self.num_joints
        self.edge_inv_mass = wp.array(edge_inv_mass_np, dtype=float, device=device)

        # Rest lengths
        rest_length_np = [particle_spacing] * self.num_joints
        self.rest_length = wp.array(rest_length_np, dtype=float, device=device)

        # Rest Darboux vectors (curvature stored in vector part)
        rest_darboux_np = [wp.quat(0.0, 0.0, 0.0, 0.0)] * max(1, self.num_bend)
        self.rest_darboux = wp.array(rest_darboux_np, dtype=wp.quat, device=device)

        # Temporary buffers
        self.particle_q_predicted = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)
        self.particle_q_temp = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)

        # Global block tridiagonal system for Thomas solver
        self.block_lower = wp.zeros((self.num_joints, JOINT_DOFS, JOINT_DOFS), dtype=float, device=device)
        self.block_diag = wp.zeros((self.num_joints, JOINT_DOFS, JOINT_DOFS), dtype=float, device=device)
        self.block_upper = wp.zeros((self.num_joints, JOINT_DOFS, JOINT_DOFS), dtype=float, device=device)
        self.block_rhs = wp.zeros((self.num_joints, JOINT_DOFS), dtype=float, device=device)
        self.delta_lambdas_global = wp.zeros((self.num_joints, JOINT_DOFS), dtype=float, device=device)

        # Correction accumulators (for atomic adds)
        self.particle_corrections = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)
        self.edge_omega = wp.zeros(self.num_joints, dtype=wp.vec3, device=device)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        # Keyboard control parameters
        self.move_speed = 1.0  # units per second
        self.anchor_min_z = particle_radius
        self.anchor_pos = wp.vec3(0.0, 0.0, start_height)

        # Director visualization buffers
        num_director_lines = self.num_joints * 3
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
            bend_ks = wp.vec3(self.bend_stiffness, self.bend_stiffness, self.twist_stiffness)

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

            # Step 2: Combined constraint solving with Cholesky
            for _ in range(self.constraint_iterations):
                wp.launch(
                    kernel=assemble_combined_tridiagonal_kernel,
                    dim=self.num_joints,
                    inputs=[
                        self.state_1.particle_q,
                        self.particle_inv_mass,
                        self.edge_q,
                        self.edge_inv_mass,
                        self.rest_length,
                        self.rest_darboux,
                        stretch_ks,
                        bend_ks,
                        self.compliance_factor,
                        self.num_joints,
                        self.num_bend,
                    ],
                    outputs=[
                        self.block_lower,
                        self.block_diag,
                        self.block_upper,
                        self.block_rhs,
                    ],
                    device=self.model.device,
                )

                wp.launch(
                    kernel=thomas_solve_block_tridiagonal_global_kernel,
                    dim=1,
                    inputs=[
                        self.block_lower,
                        self.block_diag,
                        self.block_upper,
                        self.block_rhs,
                        self.num_joints,
                    ],
                    outputs=[self.delta_lambdas_global],
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
                    dim=self.num_joints,
                    inputs=[self.edge_omega],
                    device=self.model.device,
                )

                wp.launch(
                    kernel=apply_combined_global_corrections_kernel,
                    dim=self.num_joints,
                    inputs=[
                        self.particle_inv_mass,
                        self.edge_q,
                        self.edge_inv_mass,
                        self.rest_length,
                        self.delta_lambdas_global,
                        self.num_joints,
                        self.num_bend,
                    ],
                    outputs=[self.particle_corrections, self.edge_omega],
                    device=self.model.device,
                )

                # Apply corrections
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
                    dim=self.num_joints,
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
                    dim=self.num_joints,
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
        self._handle_keyboard_input()
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

    def _handle_keyboard_input(self):
        """Move the anchor particle using keyboard controls (J/K/L/M)."""
        if not hasattr(self.viewer, "is_key_down"):
            return

        try:
            import pyglet.window.key as key
        except ImportError:
            return

        dx, dz = 0.0, 0.0

        if self.viewer.is_key_down(key.L):
            dx += self.move_speed * self.frame_dt
        if self.viewer.is_key_down(key.J):
            dx -= self.move_speed * self.frame_dt

        if self.viewer.is_key_down(key.M):
            dz += self.move_speed * self.frame_dt
        if self.viewer.is_key_down(key.K):
            dz -= self.move_speed * self.frame_dt

        if dx == 0.0 and dz == 0.0:
            return

        self.anchor_pos = wp.vec3(
            self.anchor_pos[0] + dx,
            self.anchor_pos[1],
            max(self.anchor_pos[2] + dz, self.anchor_min_z),
        )

        particle_q_np = self.state_0.particle_q.numpy()
        particle_q_np[0] = [self.anchor_pos[0], self.anchor_pos[1], self.anchor_pos[2]]

        particle_q_wp = wp.array(particle_q_np, dtype=wp.vec3, device=self.model.device)
        self.state_0.particle_q = particle_q_wp
        self.state_1.particle_q = particle_q_wp

    def gui(self, ui):
        ui.text("Cholesky Cosserat Rod (Combined 6x6 Blocks)")
        ui.text(f"Particles: {self.num_particles}, Tiles: {self.num_tiles}")
        ui.text(f"System: {TILE}x{TILE} per tile ({JOINTS_PER_TILE} joints × 6 DOFs)")
        _changed, self.stretch_stiffness = ui.slider_float("Stretch Stiffness", self.stretch_stiffness, 0.0, 1.0)
        _changed, self.shear_stiffness = ui.slider_float("Shear Stiffness", self.shear_stiffness, 0.0, 1.0)
        _changed, self.bend_stiffness = ui.slider_float("Bend Stiffness", self.bend_stiffness, 0.0, 1.0)
        _changed, self.twist_stiffness = ui.slider_float("Twist Stiffness", self.twist_stiffness, 0.0, 1.0)
        ui.separator()
        ui.text("Controls")
        ui.text("  J/L: Move anchor in X")
        ui.text("  K/M: Move anchor in Z")
        _changed, self.move_speed = ui.slider_float("Move Speed", self.move_speed, 0.1, 5.0)
        ui.separator()
        ui.text("Rest Shape (Curvature)")
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
                dim=self.num_joints * 3,
                inputs=[
                    self.state_0.particle_q,
                    self.edge_q,
                    self.num_joints,
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
