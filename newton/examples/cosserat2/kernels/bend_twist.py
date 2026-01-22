# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bend and twist constraint kernels for Cosserat rod simulations.

Contains Jacobi-style iterative kernels with friction variants and assembly
kernels for direct solvers.
"""

import warp as wp

from newton.examples.cosserat2.kernels.utilities import compute_darboux_vector


# Small constant for Dahl deadband
DAHL_DEADBAND = 1.0e-6


@wp.kernel
def solve_bend_twist_constraint_kernel(
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    rest_darboux: wp.array(dtype=wp.quat),
    bend_twist_stiffness: wp.vec3,
    num_bend: int,
    # output: accumulated corrections
    edge_q_delta: wp.array(dtype=wp.quat),
):
    """Solve bend and twist constraint for Cosserat rods using Jacobi-style iteration.

    Corrections are accumulated using atomic adds, then applied in a separate pass.

    Darboux vector omega = conjugate(q0) * q1

    Args:
        edge_q: Current edge quaternions.
        edge_inv_mass: Inverse mass per edge.
        rest_darboux: Rest Darboux vectors as quaternions.
        bend_twist_stiffness: Stiffness vector (bend_d1, twist, bend_d2).
        num_bend: Number of bend constraints.
        edge_q_delta: Output accumulated quaternion corrections (atomic add).
    """
    tid = wp.tid()
    if tid >= num_bend:
        return

    eps = 1.0e-6

    # Constraint tid connects edges tid and tid+1
    q0 = edge_q[tid]
    q1 = edge_q[tid + 1]

    inv_mass_q0 = edge_inv_mass[tid]
    inv_mass_q1 = edge_inv_mass[tid + 1]
    rest_darboux_q = rest_darboux[tid]

    # Compute Darboux vector (curvature)
    kappa = compute_darboux_vector(q0, q1, rest_darboux_q)
    omega_x = kappa[0]
    omega_y = kappa[1]
    omega_z = kappa[2]

    # Apply bending and twisting stiffness
    denom = inv_mass_q0 + inv_mass_q1 + eps
    omega_x = omega_x * bend_twist_stiffness[0] / denom
    omega_y = omega_y * bend_twist_stiffness[1] / denom
    omega_z = omega_z * bend_twist_stiffness[2] / denom

    # Omega with w=0 (discrete Darboux vector has vanishing scalar part)
    omega_corrected = wp.quat(omega_x, omega_y, omega_z, 0.0)

    # Compute quaternion corrections
    # corrq0 = q1 * omega * invMassq0
    # corrq1 = q0 * omega * (-invMassq1)
    corrq0_raw = wp.mul(q1, omega_corrected)
    corrq1_raw = wp.mul(q0, omega_corrected)

    corrq0 = wp.quat(
        corrq0_raw[0] * inv_mass_q0,
        corrq0_raw[1] * inv_mass_q0,
        corrq0_raw[2] * inv_mass_q0,
        corrq0_raw[3] * inv_mass_q0,
    )
    corrq1 = wp.quat(
        corrq1_raw[0] * (-inv_mass_q1),
        corrq1_raw[1] * (-inv_mass_q1),
        corrq1_raw[2] * (-inv_mass_q1),
        corrq1_raw[3] * (-inv_mass_q1),
    )

    # Accumulate corrections
    wp.atomic_add(edge_q_delta, tid, corrq0)
    wp.atomic_add(edge_q_delta, tid + 1, corrq1)


@wp.kernel
def solve_bend_twist_with_strain_rate_damping_kernel(
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    rest_darboux: wp.array(dtype=wp.quat),
    bend_twist_stiffness: wp.vec3,
    kappa_prev: wp.array(dtype=wp.vec3),
    damping_coeff: float,
    dt: float,
    num_bend: int,
    # outputs
    edge_q_delta: wp.array(dtype=wp.quat),
    kappa_out: wp.array(dtype=wp.vec3),
):
    """Solve bend and twist constraint with strain-rate damping (Method 2: Rayleigh-style).

    Adds damping proportional to curvature change rate:
    f_damp = damping * k * d(kappa)/dt

    This only damps bending/twisting deformation, not rigid body motion.

    Args:
        edge_q: Current edge quaternions.
        edge_inv_mass: Inverse mass per edge.
        rest_darboux: Rest Darboux vectors as quaternions.
        bend_twist_stiffness: Stiffness vector (bend_d1, twist, bend_d2).
        kappa_prev: Previous curvature values.
        damping_coeff: Strain-rate damping coefficient.
        dt: Time step.
        num_bend: Number of bend constraints.
        edge_q_delta: Output accumulated quaternion corrections (atomic add).
        kappa_out: Output current curvature values.
    """
    tid = wp.tid()
    if tid >= num_bend:
        return

    eps = 1.0e-6

    # Constraint tid connects edges tid and tid+1
    q0 = edge_q[tid]
    q1 = edge_q[tid + 1]

    inv_mass_q0 = edge_inv_mass[tid]
    inv_mass_q1 = edge_inv_mass[tid + 1]
    rest_darboux_q = rest_darboux[tid]

    # Compute current Darboux vector (curvature)
    kappa_now = compute_darboux_vector(q0, q1, rest_darboux_q)

    # Store current curvature for next iteration
    kappa_out[tid] = kappa_now

    # Compute curvature rate
    kappa_old = kappa_prev[tid]
    kappa_dot = (kappa_now - kappa_old) / dt

    # Strain-rate damping: add damping force proportional to curvature rate
    # f_damp = damping * k * kappa_dot
    damp_x = damping_coeff * bend_twist_stiffness[0] * kappa_dot[0]
    damp_y = damping_coeff * bend_twist_stiffness[1] * kappa_dot[1]
    damp_z = damping_coeff * bend_twist_stiffness[2] * kappa_dot[2]

    # Total correction = elastic + damping
    omega_x = kappa_now[0] + damp_x
    omega_y = kappa_now[1] + damp_y
    omega_z = kappa_now[2] + damp_z

    # Apply bending and twisting stiffness
    denom = inv_mass_q0 + inv_mass_q1 + eps
    omega_x = omega_x * bend_twist_stiffness[0] / denom
    omega_y = omega_y * bend_twist_stiffness[1] / denom
    omega_z = omega_z * bend_twist_stiffness[2] / denom

    # Omega with w=0
    omega_corrected = wp.quat(omega_x, omega_y, omega_z, 0.0)

    # Compute quaternion corrections
    corrq0_raw = wp.mul(q1, omega_corrected)
    corrq1_raw = wp.mul(q0, omega_corrected)

    corrq0 = wp.quat(
        corrq0_raw[0] * inv_mass_q0,
        corrq0_raw[1] * inv_mass_q0,
        corrq0_raw[2] * inv_mass_q0,
        corrq0_raw[3] * inv_mass_q0,
    )
    corrq1 = wp.quat(
        corrq1_raw[0] * (-inv_mass_q1),
        corrq1_raw[1] * (-inv_mass_q1),
        corrq1_raw[2] * (-inv_mass_q1),
        corrq1_raw[3] * (-inv_mass_q1),
    )

    # Accumulate corrections
    wp.atomic_add(edge_q_delta, tid, corrq0)
    wp.atomic_add(edge_q_delta, tid + 1, corrq1)


@wp.kernel
def solve_bend_twist_with_dahl_friction_kernel(
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    rest_darboux: wp.array(dtype=wp.quat),
    bend_twist_stiffness: wp.vec3,
    kappa_prev: wp.array(dtype=wp.vec3),
    sigma_prev: wp.array(dtype=wp.vec3),
    dkappa_prev: wp.array(dtype=wp.vec3),
    eps_max: float,
    tau: float,
    num_bend: int,
    # outputs
    edge_q_delta: wp.array(dtype=wp.quat),
    kappa_out: wp.array(dtype=wp.vec3),
    sigma_out: wp.array(dtype=wp.vec3),
    dkappa_out: wp.array(dtype=wp.vec3),
):
    """Solve bend and twist constraint with Dahl hysteresis friction (Method 3).

    Dahl model captures path-dependent friction with hysteresis:
    - sigma (friction stress) evolves based on curvature change direction
    - sigma_max = k * eps_max (friction envelope)
    - tau controls memory decay (how quickly friction "forgets" direction changes)

    Evolution: sigma_new = s * sigma_max * (1 - exp(-s*d_kappa/tau)) + sigma_prev * exp(-s*d_kappa/tau)
    where s = sign(d_kappa) is the direction flag.

    Args:
        edge_q: Current edge quaternions.
        edge_inv_mass: Inverse mass per edge.
        rest_darboux: Rest Darboux vectors as quaternions.
        bend_twist_stiffness: Stiffness vector (bend_d1, twist, bend_d2).
        kappa_prev: Previous curvature values.
        sigma_prev: Previous friction stress values.
        dkappa_prev: Previous curvature change directions.
        eps_max: Maximum persistent strain (friction saturation).
        tau: Memory decay length.
        num_bend: Number of bend constraints.
        edge_q_delta: Output accumulated quaternion corrections (atomic add).
        kappa_out: Output current curvature values.
        sigma_out: Output current friction stress values.
        dkappa_out: Output current curvature change directions.
    """
    tid = wp.tid()
    if tid >= num_bend:
        return

    eps = 1.0e-6

    # Constraint tid connects edges tid and tid+1
    q0 = edge_q[tid]
    q1 = edge_q[tid + 1]

    inv_mass_q0 = edge_inv_mass[tid]
    inv_mass_q1 = edge_inv_mass[tid + 1]
    rest_darboux_q = rest_darboux[tid]

    # Compute current Darboux vector (curvature)
    kappa_now = compute_darboux_vector(q0, q1, rest_darboux_q)

    # Store current curvature
    kappa_out[tid] = kappa_now

    # Read previous state
    kappa_old = kappa_prev[tid]
    sigma_old = sigma_prev[tid]
    dkappa_old = dkappa_prev[tid]

    # Compute sigma_max per component (friction envelope)
    sigma_max_x = bend_twist_stiffness[0] * eps_max
    sigma_max_y = bend_twist_stiffness[1] * eps_max
    sigma_max_z = bend_twist_stiffness[2] * eps_max

    # Process each component independently (3 separate hysteresis loops)
    sigma_new = wp.vec3(0.0, 0.0, 0.0)
    dkappa_new = wp.vec3(0.0, 0.0, 0.0)

    # Component 0 (bend d1)
    d_kappa_0 = kappa_now[0] - kappa_old[0]
    dkappa_new = wp.vec3(d_kappa_0, dkappa_new[1], dkappa_new[2])
    if d_kappa_0 > DAHL_DEADBAND:
        s_0 = 1.0
    elif d_kappa_0 < -DAHL_DEADBAND:
        s_0 = -1.0
    else:
        s_0 = 1.0 if dkappa_old[0] >= 0.0 else -1.0

    if sigma_max_x > 0.0 and tau > 0.0:
        exp_term_0 = wp.exp(-s_0 * d_kappa_0 / tau)
        sigma_0 = s_0 * sigma_max_x * (1.0 - exp_term_0) + sigma_old[0] * exp_term_0
        sigma_0 = wp.clamp(sigma_0, -sigma_max_x, sigma_max_x)
    else:
        sigma_0 = 0.0
    sigma_new = wp.vec3(sigma_0, sigma_new[1], sigma_new[2])

    # Component 1 (twist)
    d_kappa_1 = kappa_now[1] - kappa_old[1]
    dkappa_new = wp.vec3(dkappa_new[0], d_kappa_1, dkappa_new[2])
    if d_kappa_1 > DAHL_DEADBAND:
        s_1 = 1.0
    elif d_kappa_1 < -DAHL_DEADBAND:
        s_1 = -1.0
    else:
        s_1 = 1.0 if dkappa_old[1] >= 0.0 else -1.0

    if sigma_max_y > 0.0 and tau > 0.0:
        exp_term_1 = wp.exp(-s_1 * d_kappa_1 / tau)
        sigma_1 = s_1 * sigma_max_y * (1.0 - exp_term_1) + sigma_old[1] * exp_term_1
        sigma_1 = wp.clamp(sigma_1, -sigma_max_y, sigma_max_y)
    else:
        sigma_1 = 0.0
    sigma_new = wp.vec3(sigma_new[0], sigma_1, sigma_new[2])

    # Component 2 (bend d2)
    d_kappa_2 = kappa_now[2] - kappa_old[2]
    dkappa_new = wp.vec3(dkappa_new[0], dkappa_new[1], d_kappa_2)
    if d_kappa_2 > DAHL_DEADBAND:
        s_2 = 1.0
    elif d_kappa_2 < -DAHL_DEADBAND:
        s_2 = -1.0
    else:
        s_2 = 1.0 if dkappa_old[2] >= 0.0 else -1.0

    if sigma_max_z > 0.0 and tau > 0.0:
        exp_term_2 = wp.exp(-s_2 * d_kappa_2 / tau)
        sigma_2 = s_2 * sigma_max_z * (1.0 - exp_term_2) + sigma_old[2] * exp_term_2
        sigma_2 = wp.clamp(sigma_2, -sigma_max_z, sigma_max_z)
    else:
        sigma_2 = 0.0
    sigma_new = wp.vec3(sigma_new[0], sigma_new[1], sigma_2)

    # Store friction state
    sigma_out[tid] = sigma_new
    dkappa_out[tid] = dkappa_new

    # Total correction = elastic + friction
    # Friction adds a stress that resists bending changes
    omega_x = kappa_now[0] + sigma_new[0] / (bend_twist_stiffness[0] + eps)
    omega_y = kappa_now[1] + sigma_new[1] / (bend_twist_stiffness[1] + eps)
    omega_z = kappa_now[2] + sigma_new[2] / (bend_twist_stiffness[2] + eps)

    # Apply bending and twisting stiffness
    denom = inv_mass_q0 + inv_mass_q1 + eps
    omega_x = omega_x * bend_twist_stiffness[0] / denom
    omega_y = omega_y * bend_twist_stiffness[1] / denom
    omega_z = omega_z * bend_twist_stiffness[2] / denom

    # Omega with w=0
    omega_corrected = wp.quat(omega_x, omega_y, omega_z, 0.0)

    # Compute quaternion corrections
    corrq0_raw = wp.mul(q1, omega_corrected)
    corrq1_raw = wp.mul(q0, omega_corrected)

    corrq0 = wp.quat(
        corrq0_raw[0] * inv_mass_q0,
        corrq0_raw[1] * inv_mass_q0,
        corrq0_raw[2] * inv_mass_q0,
        corrq0_raw[3] * inv_mass_q0,
    )
    corrq1 = wp.quat(
        corrq1_raw[0] * (-inv_mass_q1),
        corrq1_raw[1] * (-inv_mass_q1),
        corrq1_raw[2] * (-inv_mass_q1),
        corrq1_raw[3] * (-inv_mass_q1),
    )

    # Accumulate corrections
    wp.atomic_add(edge_q_delta, tid, corrq0)
    wp.atomic_add(edge_q_delta, tid + 1, corrq1)


@wp.kernel
def compute_bend_constraint_data_kernel(
    edge_q: wp.array(dtype=wp.quat),
    rest_darboux: wp.array(dtype=wp.quat),
    num_bend: int,
    # outputs
    bend_violation: wp.array(dtype=float),
    bend_direction: wp.array(dtype=wp.vec3),
):
    """Compute bend/twist constraint violations and directions for direct solvers.

    Constraint: omega = 2*Im(conj(q0)*q1) should match rest_darboux

    Args:
        edge_q: Current edge quaternions.
        rest_darboux: Rest Darboux vectors as quaternions.
        num_bend: Number of bend constraints.
        bend_violation: Output scalar violation magnitudes.
        bend_direction: Output normalized violation directions.
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
    tile_size: int,
    # outputs
    system_matrix: wp.array2d(dtype=float),
    system_rhs: wp.array1d(dtype=float),
):
    """Assemble global system for bend/twist constraints (Cholesky solver).

    A = J M^{-1} J^T + compliance*I

    For bend constraints, coupling is through shared quaternions:
    - A[k,k] = w_qk + w_q{k+1} + compliance
    - A[k,k+1] = coupling through shared quaternion q_{k+1}

    Args:
        edge_inv_mass: Inverse mass per edge.
        bend_direction: Normalized constraint directions.
        bend_violation: Constraint violation magnitudes.
        compliance_factor: Compliance / dt^2.
        num_bend: Number of bend constraints.
        tile_size: Size of the tile for Cholesky.
        system_matrix: Output system matrix (tile_size x tile_size).
        system_rhs: Output right-hand side vector.
    """
    # Initialize matrix
    for i in range(tile_size):
        system_rhs[i] = 0.0
        for j in range(tile_size):
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
    for i in range(num_bend, tile_size):
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
    """Apply quaternion corrections from bend/twist solve.

    For quaternion k:
    - Constraint k-1 contributes corrq from q1 side
    - Constraint k contributes corrq from q0 side

    Args:
        edge_q: Current edge quaternions.
        edge_inv_mass: Inverse mass per edge.
        bend_direction: Normalized constraint directions.
        delta_lambda: Solved Lagrange multiplier increments.
        num_bend: Number of bend constraints.
        num_edges: Number of edges.
        edge_q_out: Output corrected edge quaternions.
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
def compute_current_kappa_kernel(
    edge_q: wp.array(dtype=wp.quat),
    rest_darboux: wp.array(dtype=wp.quat),
    num_bend: int,
    # output
    kappa_out: wp.array(dtype=wp.vec3),
):
    """Compute and store current curvature for all bend constraints.

    Used to initialize friction state arrays.

    Args:
        edge_q: Current edge quaternions.
        rest_darboux: Rest Darboux vectors as quaternions.
        num_bend: Number of bend constraints.
        kappa_out: Output current curvature values.
    """
    tid = wp.tid()
    if tid >= num_bend:
        return

    q0 = edge_q[tid]
    q1 = edge_q[tid + 1]
    rest_darboux_q = rest_darboux[tid]

    kappa = compute_darboux_vector(q0, q1, rest_darboux_q)
    kappa_out[tid] = kappa
