# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Stretch and shear constraint kernels for Cosserat rod simulations.

Contains both Jacobi-style iterative kernels and assembly kernels for direct solvers.
"""

import warp as wp

from newton.examples.cosserat2.kernels.utilities import quat_rotate_e3, quat_e3_bar


@wp.kernel
def solve_stretch_shear_constraint_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    stretch_shear_stiffness: wp.vec3,
    num_stretch: int,
    # outputs: accumulated corrections
    particle_delta: wp.array(dtype=wp.vec3),
    edge_q_delta: wp.array(dtype=wp.quat),
):
    """Solve stretch and shear constraint for Cosserat rods using Jacobi-style iteration.

    Corrections are accumulated using atomic adds, then applied in a separate pass.
    This allows parallel execution of all constraints.

    Constraint: gamma = (p1 - p0) / L - d3(q) = 0

    Args:
        particle_q: Current particle positions.
        particle_inv_mass: Inverse mass per particle.
        edge_q: Current edge quaternions.
        edge_inv_mass: Inverse mass per edge.
        rest_length: Rest length per edge.
        stretch_shear_stiffness: Stiffness vector (shear_d1, shear_d2, stretch_d3).
        num_stretch: Number of stretch constraints.
        particle_delta: Output accumulated position corrections (atomic add).
        edge_q_delta: Output accumulated quaternion corrections (atomic add).
    """
    tid = wp.tid()
    if tid >= num_stretch:
        return

    eps = 1.0e-6

    p0 = particle_q[tid]
    p1 = particle_q[tid + 1]
    q0 = edge_q[tid]

    inv_mass_p0 = particle_inv_mass[tid]
    inv_mass_p1 = particle_inv_mass[tid + 1]
    inv_mass_q0 = edge_inv_mass[tid]
    L = rest_length[tid]

    # Compute third director d3 = q0 * e3 * conjugate(q0)
    d3 = quat_rotate_e3(q0)

    # Compute constraint violation: gamma = (p1 - p0) / L - d3
    edge_vec = p1 - p0
    gamma = edge_vec / L - d3

    # Compute denominator for constraint scaling
    denom = (inv_mass_p0 + inv_mass_p1) / L + inv_mass_q0 * 4.0 * L + eps

    # Scale gamma by inverse denominator
    gamma = gamma / denom

    # Apply stiffness in LOCAL frame coordinates (material frame)
    # Transform gamma to local space: gamma_loc = R^T(q0) * gamma
    gamma_loc = wp.quat_rotate_inv(q0, gamma)

    # Apply anisotropic stiffness: [shear_d1, shear_d2, stretch_d3]
    gamma_loc = wp.vec3(
        gamma_loc[0] * stretch_shear_stiffness[0],
        gamma_loc[1] * stretch_shear_stiffness[1],
        gamma_loc[2] * stretch_shear_stiffness[2],
    )

    # Transform back to world space: gamma = R(q0) * gamma_loc
    gamma = wp.quat_rotate(q0, gamma_loc)

    # Compute position corrections
    corr0 = gamma * inv_mass_p0
    corr1 = gamma * (-inv_mass_p1)

    # Compute quaternion correction using q * e3_bar formula
    q_e3_bar_val = quat_e3_bar(q0)
    gamma_quat = wp.quat(gamma[0], gamma[1], gamma[2], 0.0)
    corrq0 = wp.mul(gamma_quat, q_e3_bar_val)

    scale = 2.0 * inv_mass_q0 * L
    corrq0 = wp.quat(corrq0[0] * scale, corrq0[1] * scale, corrq0[2] * scale, corrq0[3] * scale)

    # Accumulate corrections using atomic adds
    wp.atomic_add(particle_delta, tid, corr0)
    wp.atomic_add(particle_delta, tid + 1, corr1)
    wp.atomic_add(edge_q_delta, tid, corrq0)


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
    """Compute stretch/shear constraint violations and directions for direct solvers.

    Constraint: gamma = (p1 - p0) / L - d3(q) = 0 (3D vector constraint)

    We linearize this as a scalar constraint with magnitude and direction
    for efficient global solving.

    Args:
        particle_q: Current particle positions.
        edge_q: Current edge quaternions.
        rest_length: Rest length per edge.
        num_stretch: Number of stretch constraints.
        constraint_violation: Output scalar violation magnitudes.
        constraint_direction: Output normalized violation directions.
        constraint_quat_direction: Output quaternion correction directions.
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
def assemble_stretch_tridiagonal_system_kernel(
    particle_inv_mass: wp.array(dtype=float),
    edge_inv_mass: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    constraint_direction: wp.array(dtype=wp.vec3),
    constraint_violation: wp.array(dtype=float),
    compliance_factor: float,
    num_stretch: int,
    # outputs - tridiagonal representation
    diag: wp.array(dtype=float),
    off_diag: wp.array(dtype=float),
    rhs: wp.array(dtype=float),
):
    """Assemble the tridiagonal system for stretch constraints (Thomas algorithm).

    For a tridiagonal matrix:
        [ d[0]  a[0]    0     0   ...  ]
        [ a[0]  d[1]  a[1]    0   ...  ]
        [   0   a[1]  d[2]  a[2]  ...  ]
        [  ...                         ]

    We store:
        diag[i] = d[i] = (w_i + w_{i+1})/L^2 + 4*L^2*w_q + compliance
        off_diag[i] = a[i] = coupling through shared particle

    Args:
        particle_inv_mass: Inverse mass per particle.
        edge_inv_mass: Inverse mass per edge.
        rest_length: Rest length per edge.
        constraint_direction: Normalized constraint directions.
        constraint_violation: Constraint violation magnitudes.
        compliance_factor: Compliance / dt^2.
        num_stretch: Number of stretch constraints.
        diag: Output main diagonal.
        off_diag: Output sub/super diagonal.
        rhs: Output right-hand side.
    """
    # Single thread assembles the system (sequential but O(n))
    for k in range(num_stretch):
        w_p0 = particle_inv_mass[k]
        w_p1 = particle_inv_mass[k + 1]
        w_q0 = edge_inv_mass[k]
        L = rest_length[k]

        # Position contribution
        pos_diag = (w_p0 + w_p1) / (L * L)

        # Quaternion contribution
        quat_diag = 4.0 * L * L * w_q0

        # Diagonal: A[k,k]
        diag[k] = pos_diag + quat_diag + compliance_factor

        # RHS: -C_k
        rhs[k] = -constraint_violation[k]

        # Off-diagonal coupling with next constraint
        if k + 1 < num_stretch:
            n_k = constraint_direction[k]
            n_k1 = constraint_direction[k + 1]
            L_k1 = rest_length[k + 1]

            # A[k, k+1] = -w_{k+1} * (n_k/L_k) . (n_{k+1}/L_{k+1})
            off_diag[k] = -particle_inv_mass[k + 1] * wp.dot(n_k, n_k1) / (L * L_k1)


@wp.kernel
def assemble_stretch_global_system_kernel(
    particle_inv_mass: wp.array(dtype=float),
    edge_inv_mass: wp.array(dtype=float),
    rest_length: wp.array(dtype=float),
    constraint_direction: wp.array(dtype=wp.vec3),
    constraint_violation: wp.array(dtype=float),
    compliance_factor: float,
    num_stretch: int,
    tile_size: int,
    # outputs
    system_matrix: wp.array2d(dtype=float),
    system_rhs: wp.array1d(dtype=float),
):
    """Assemble the global system matrix for stretch constraints (Cholesky solver).

    System matrix A = J M^{-1} J^T + alpha/dt^2 * I

    For stretch/shear constraints in a chain, A is tridiagonal:
    - A[k,k] = (w_pk + w_p{k+1})/L^2 + 4*L^2*w_qk + compliance
    - A[k,k+1] = coupling through shared particle p_{k+1}

    Args:
        particle_inv_mass: Inverse mass per particle.
        edge_inv_mass: Inverse mass per edge.
        rest_length: Rest length per edge.
        constraint_direction: Normalized constraint directions.
        constraint_violation: Constraint violation magnitudes.
        compliance_factor: Compliance / dt^2.
        num_stretch: Number of stretch constraints.
        tile_size: Size of the tile for Cholesky.
        system_matrix: Output system matrix (tile_size x tile_size).
        system_rhs: Output right-hand side vector.
    """
    # Initialize matrix to zero
    for i in range(tile_size):
        system_rhs[i] = 0.0
        for j in range(tile_size):
            system_matrix[i, j] = 0.0

    # Fill in the tridiagonal structure
    for k in range(num_stretch):
        w_p0 = particle_inv_mass[k]
        w_p1 = particle_inv_mass[k + 1]
        w_q0 = edge_inv_mass[k]
        L = rest_length[k]

        # Diagonal contribution from position terms: (w_p0 + w_p1) / L^2
        pos_diag = (w_p0 + w_p1) / (L * L)

        # Quaternion contribution to diagonal: 4 * L^2 * w_q
        quat_diag = 4.0 * L * L * w_q0

        # Total diagonal
        diag = pos_diag + quat_diag + compliance_factor
        system_matrix[k, k] = diag

        # RHS: -C_k (constraint violation)
        system_rhs[k] = -constraint_violation[k]

        # Off-diagonal coupling with next constraint (k+1)
        if k + 1 < num_stretch:
            n_k = constraint_direction[k]
            n_k1 = constraint_direction[k + 1]
            L_k1 = rest_length[k + 1]

            # Coupling: w_{p,k+1} * (n_k/L_k) . (n_{k+1}/L_{k+1})
            coupling = -particle_inv_mass[k + 1] * wp.dot(n_k, n_k1) / (L * L_k1)
            system_matrix[k, k + 1] = coupling
            system_matrix[k + 1, k] = coupling

    # Pad unused rows/columns with identity to keep matrix SPD
    for i in range(num_stretch, tile_size):
        system_matrix[i, i] = 1.0


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
    """Apply position and quaternion corrections from solved Lagrange multipliers.

    delta_x = M^{-1} J^T delta_lambda
    delta_q = M_q^{-1} J_q^T delta_lambda (then normalize)

    For particle i:
    - Constraint i-1 contributes: +n_{i-1}/L * delta_lambda_{i-1} * w_i (if i > 0)
    - Constraint i contributes: -n_i/L * delta_lambda_i * w_i (if i < num_stretch)

    Args:
        particle_q: Current particle positions.
        edge_q: Current edge quaternions.
        particle_inv_mass: Inverse mass per particle.
        edge_inv_mass: Inverse mass per edge.
        rest_length: Rest length per edge.
        constraint_direction: Normalized constraint directions.
        constraint_quat_direction: Quaternion correction directions.
        delta_lambda: Solved Lagrange multiplier increments.
        num_stretch: Number of stretch constraints.
        num_particles: Number of particles.
        particle_q_out: Output corrected particle positions.
        edge_q_out: Output corrected edge quaternions.
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
