# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for Cosserat rod direct solver.

This module contains all Warp kernels needed for the Cosserat rod simulation.
Kernels are designed to work on both CPU and GPU devices.
"""

import warp as wp

# =============================================================================
# Constants
# =============================================================================

# Bandwidth for 6x6 block tri-diagonal system
BAND_KD = 11  # Upper bandwidth
BAND_LDAB = 2 * BAND_KD + 1  # Total banded matrix rows


# =============================================================================
# Quaternion helper functions
# =============================================================================


@wp.func
def quat_mul(a: wp.quat, b: wp.quat) -> wp.quat:
    """Multiply two quaternions."""
    return wp.quat(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    )


@wp.func
def quat_conjugate(q: wp.quat) -> wp.quat:
    """Quaternion conjugate (inverse for unit quaternions)."""
    return wp.quat(-q.x, -q.y, -q.z, q.w)


@wp.func
def quat_normalize(q: wp.quat) -> wp.quat:
    """Normalize quaternion."""
    norm = wp.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm > 1.0e-10:
        inv_norm = 1.0 / norm
        return wp.quat(q.x * inv_norm, q.y * inv_norm, q.z * inv_norm, q.w * inv_norm)
    return wp.quat(0.0, 0.0, 0.0, 1.0)


@wp.func
def quat_rotate_vector(q: wp.quat, v: wp.vec3) -> wp.vec3:
    """Rotate vector by quaternion."""
    tx = 2.0 * (q.y * v.z - q.z * v.y)
    ty = 2.0 * (q.z * v.x - q.x * v.z)
    tz = 2.0 * (q.x * v.y - q.y * v.x)

    return wp.vec3(
        v.x + q.w * tx + q.y * tz - q.z * ty,
        v.y + q.w * ty + q.z * tx - q.x * tz,
        v.z + q.w * tz + q.x * ty - q.y * tx,
    )


@wp.func
def quat_correction_g(q: wp.quat, dtheta: wp.vec3) -> wp.quat:
    """Apply rotation correction using G matrix."""
    norm_sq = dtheta.x * dtheta.x + dtheta.y * dtheta.y + dtheta.z * dtheta.z
    if norm_sq < 1.0e-20:
        return q

    # G matrix maps angular correction to quaternion correction
    corr_x = 0.5 * (q.w * dtheta.x + q.z * dtheta.y - q.y * dtheta.z)
    corr_y = 0.5 * (-q.z * dtheta.x + q.w * dtheta.y + q.x * dtheta.z)
    corr_z = 0.5 * (q.y * dtheta.x - q.x * dtheta.y + q.w * dtheta.z)
    corr_w = 0.5 * (-q.x * dtheta.x - q.y * dtheta.y - q.z * dtheta.z)

    q_new = wp.quat(q.x + corr_x, q.y + corr_y, q.z + corr_z, q.w + corr_w)
    return quat_normalize(q_new)


# =============================================================================
# Jacobian indexing
# =============================================================================


@wp.func
def jacobian_index(edge: int, row: int, col: int) -> int:
    """Index into flattened Jacobian array [n_edges * 6 * 6]."""
    return edge * 36 + row * 6 + col


@wp.func
def jacobian_dot(
    jacobian: wp.array(dtype=wp.float32),
    edge: int,
    col: int,
    dl0: float,
    dl1: float,
    dl2: float,
    dl3: float,
    dl4: float,
    dl5: float,
) -> float:
    """Compute J^T * delta_lambda for a single column."""
    return (
        jacobian[jacobian_index(edge, 0, col)] * dl0
        + jacobian[jacobian_index(edge, 1, col)] * dl1
        + jacobian[jacobian_index(edge, 2, col)] * dl2
        + jacobian[jacobian_index(edge, 3, col)] * dl3
        + jacobian[jacobian_index(edge, 4, col)] * dl4
        + jacobian[jacobian_index(edge, 5, col)] * dl5
    )


# =============================================================================
# Prediction kernels
# =============================================================================


@wp.kernel
def kernel_predict_positions(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    gravity: wp.vec3,
    dt: float,
    damping: float,
    # Output
    predicted_positions: wp.array(dtype=wp.vec3),
):
    """Semi-implicit Euler position prediction."""
    i = wp.tid()
    inv_m = inv_masses[i]

    if inv_m > 0.0:
        v = velocities[i] * (1.0 - damping)
        accel = inv_m * (gravity + forces[i])
        v = v + dt * accel
        velocities[i] = v
        predicted_positions[i] = positions[i] + dt * v
    else:
        predicted_positions[i] = positions[i]


@wp.kernel
def kernel_predict_rotations(
    orientations: wp.array(dtype=wp.quat),
    angular_velocities: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
    quat_inv_masses: wp.array(dtype=wp.float32),
    dt: float,
    damping: float,
    # Output
    predicted_orientations: wp.array(dtype=wp.quat),
):
    """Quaternion rotation prediction."""
    i = wp.tid()
    quat_inv_m = quat_inv_masses[i]

    if quat_inv_m > 0.0:
        omega = angular_velocities[i] * (1.0 - damping)
        omega = omega + dt * torques[i]
        angular_velocities[i] = omega

        q = orientations[i]
        omega_quat = wp.quat(omega.x, omega.y, omega.z, 0.0)
        dq = quat_mul(omega_quat, q)
        q_new = wp.quat(
            q.x + 0.5 * dt * dq.x,
            q.y + 0.5 * dt * dq.y,
            q.z + 0.5 * dt * dq.z,
            q.w + 0.5 * dt * dq.w,
        )
        predicted_orientations[i] = quat_normalize(q_new)
    else:
        predicted_orientations[i] = orientations[i]


# =============================================================================
# Integration kernels
# =============================================================================


@wp.kernel
def kernel_integrate_positions(
    positions: wp.array(dtype=wp.vec3),
    predicted_positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    dt: float,
):
    """Derive velocities and update positions."""
    i = wp.tid()
    inv_dt = 1.0 / dt if dt > 1.0e-10 else 0.0

    if inv_masses[i] > 0.0:
        velocities[i] = (predicted_positions[i] - positions[i]) * inv_dt
        positions[i] = predicted_positions[i]
    else:
        positions[i] = predicted_positions[i]
        velocities[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def kernel_integrate_rotations(
    orientations: wp.array(dtype=wp.quat),
    predicted_orientations: wp.array(dtype=wp.quat),
    prev_orientations: wp.array(dtype=wp.quat),
    angular_velocities: wp.array(dtype=wp.vec3),
    quat_inv_masses: wp.array(dtype=wp.float32),
    dt: float,
):
    """Derive angular velocities and update orientations."""
    i = wp.tid()
    inv_dt = 1.0 / dt if dt > 1.0e-10 else 0.0

    if quat_inv_masses[i] > 0.0:
        q_old = orientations[i]
        prev_orientations[i] = q_old
        q_new = predicted_orientations[i]
        orientations[i] = q_new

        # Derive angular velocity
        q_old_inv = quat_conjugate(q_old)
        q_rel = quat_mul(q_new, q_old_inv)

        sign = 1.0
        if q_rel.w < 0.0:
            sign = -1.0

        angular_velocities[i] = wp.vec3(
            sign * 2.0 * inv_dt * q_rel.x,
            sign * 2.0 * inv_dt * q_rel.y,
            sign * 2.0 * inv_dt * q_rel.z,
        )
    else:
        prev_orientations[i] = orientations[i]
        orientations[i] = predicted_orientations[i]
        angular_velocities[i] = wp.vec3(0.0, 0.0, 0.0)


# =============================================================================
# Constraint kernels
# =============================================================================


@wp.kernel
def kernel_prepare_compliance(
    rest_lengths: wp.array(dtype=wp.float32),
    bend_stiffness: wp.array(dtype=wp.vec3),
    young_modulus: float,
    torsion_modulus: float,
    dt: float,
    # Output
    compliance: wp.array(dtype=wp.float32),
):
    """Compute compliance values for each constraint DOF."""
    i = wp.tid()

    L = rest_lengths[i]
    dt2 = dt * dt
    inv_dt2 = 1.0 / dt2

    # Stretch compliance (nearly inextensible)
    stretch_reg = 1.0e-12
    stretch_compliance = stretch_reg * inv_dt2

    base = i * 6
    compliance[base + 0] = stretch_compliance
    compliance[base + 1] = stretch_compliance
    compliance[base + 2] = stretch_compliance

    # Bend/twist compliance
    ks = bend_stiffness[i]
    eps = 1.0e-10

    K_bend1 = ks.x * young_modulus
    K_bend2 = ks.y * young_modulus
    K_twist = ks.z * torsion_modulus

    compliance[base + 3] = inv_dt2 / (K_bend1 + eps) / L
    compliance[base + 4] = inv_dt2 / (K_bend2 + eps) / L
    compliance[base + 5] = inv_dt2 / (K_twist + eps) / L


@wp.kernel
def kernel_update_constraints(
    predicted_positions: wp.array(dtype=wp.vec3),
    predicted_orientations: wp.array(dtype=wp.quat),
    rest_lengths: wp.array(dtype=wp.float32),
    rest_darboux: wp.array(dtype=wp.vec3),
    # Output
    constraint_values: wp.array(dtype=wp.float32),
):
    """Compute constraint violations."""
    i = wp.tid()

    p0 = predicted_positions[i]
    p1 = predicted_positions[i + 1]
    q0 = predicted_orientations[i]
    q1 = predicted_orientations[i + 1]
    L = rest_lengths[i]

    # Stretch-Shear
    d3_0 = quat_rotate_vector(q0, wp.vec3(0.0, 0.0, 1.0))
    d3_1 = quat_rotate_vector(q1, wp.vec3(0.0, 0.0, 1.0))

    connector0 = p0 + 0.5 * L * d3_0
    connector1 = p1 - 0.5 * L * d3_1
    stretch_violation = connector0 - connector1

    base = i * 6
    constraint_values[base + 0] = stretch_violation.x
    constraint_values[base + 1] = stretch_violation.y
    constraint_values[base + 2] = stretch_violation.z

    # Bend-Twist
    q0_inv = quat_conjugate(q0)
    q_rel = quat_mul(q0_inv, q1)
    omega = wp.vec3(q_rel.x, q_rel.y, q_rel.z)
    omega_rest = rest_darboux[i]

    constraint_values[base + 3] = omega.x - omega_rest.x
    constraint_values[base + 4] = omega.y - omega_rest.y
    constraint_values[base + 5] = omega.z - omega_rest.z


@wp.kernel
def kernel_compute_jacobians(
    predicted_orientations: wp.array(dtype=wp.quat),
    rest_lengths: wp.array(dtype=wp.float32),
    # Output
    jacobian_pos: wp.array(dtype=wp.float32),
    jacobian_rot: wp.array(dtype=wp.float32),
):
    """Compute 6x6 Jacobians for each constraint."""
    i = wp.tid()

    q0 = predicted_orientations[i]
    q1 = predicted_orientations[i + 1]
    L = rest_lengths[i]
    if L < 1.0e-8:
        L = 1.0e-8

    d3_0 = quat_rotate_vector(q0, wp.vec3(0.0, 0.0, 1.0))
    d3_1 = quat_rotate_vector(q1, wp.vec3(0.0, 0.0, 1.0))

    r0 = 0.5 * L * d3_0
    r1 = -0.5 * L * d3_1

    # G matrices
    G0_00 = 0.5 * q0.w
    G0_01 = 0.5 * q0.z
    G0_02 = -0.5 * q0.y
    G0_10 = -0.5 * q0.z
    G0_11 = 0.5 * q0.w
    G0_12 = 0.5 * q0.x
    G0_20 = 0.5 * q0.y
    G0_21 = -0.5 * q0.x
    G0_22 = 0.5 * q0.w
    G0_30 = -0.5 * q0.x
    G0_31 = -0.5 * q0.y
    G0_32 = -0.5 * q0.z

    G1_00 = 0.5 * q1.w
    G1_01 = 0.5 * q1.z
    G1_02 = -0.5 * q1.y
    G1_10 = -0.5 * q1.z
    G1_11 = 0.5 * q1.w
    G1_12 = 0.5 * q1.x
    G1_20 = 0.5 * q1.y
    G1_21 = -0.5 * q1.x
    G1_22 = 0.5 * q1.w
    G1_30 = -0.5 * q1.x
    G1_31 = -0.5 * q1.y
    G1_32 = -0.5 * q1.z

    # jOmega matrices
    jO0_00 = -q1.w
    jO0_01 = -q1.z
    jO0_02 = q1.y
    jO0_03 = q1.x
    jO0_10 = q1.z
    jO0_11 = -q1.w
    jO0_12 = -q1.x
    jO0_13 = q1.y
    jO0_20 = -q1.y
    jO0_21 = q1.x
    jO0_22 = -q1.w
    jO0_23 = q1.z

    jO1_00 = q0.w
    jO1_01 = q0.z
    jO1_02 = -q0.y
    jO1_03 = -q0.x
    jO1_10 = -q0.z
    jO1_11 = q0.w
    jO1_12 = q0.x
    jO1_13 = -q0.y
    jO1_20 = q0.y
    jO1_21 = -q0.x
    jO1_22 = q0.w
    jO1_23 = -q0.z

    # jOmegaG0 = jOmega0 @ G0 (3x3)
    jOG0_00 = jO0_00 * G0_00 + jO0_01 * G0_10 + jO0_02 * G0_20 + jO0_03 * G0_30
    jOG0_01 = jO0_00 * G0_01 + jO0_01 * G0_11 + jO0_02 * G0_21 + jO0_03 * G0_31
    jOG0_02 = jO0_00 * G0_02 + jO0_01 * G0_12 + jO0_02 * G0_22 + jO0_03 * G0_32
    jOG0_10 = jO0_10 * G0_00 + jO0_11 * G0_10 + jO0_12 * G0_20 + jO0_13 * G0_30
    jOG0_11 = jO0_10 * G0_01 + jO0_11 * G0_11 + jO0_12 * G0_21 + jO0_13 * G0_31
    jOG0_12 = jO0_10 * G0_02 + jO0_11 * G0_12 + jO0_12 * G0_22 + jO0_13 * G0_32
    jOG0_20 = jO0_20 * G0_00 + jO0_21 * G0_10 + jO0_22 * G0_20 + jO0_23 * G0_30
    jOG0_21 = jO0_20 * G0_01 + jO0_21 * G0_11 + jO0_22 * G0_21 + jO0_23 * G0_31
    jOG0_22 = jO0_20 * G0_02 + jO0_21 * G0_12 + jO0_22 * G0_22 + jO0_23 * G0_32

    jOG1_00 = jO1_00 * G1_00 + jO1_01 * G1_10 + jO1_02 * G1_20 + jO1_03 * G1_30
    jOG1_01 = jO1_00 * G1_01 + jO1_01 * G1_11 + jO1_02 * G1_21 + jO1_03 * G1_31
    jOG1_02 = jO1_00 * G1_02 + jO1_01 * G1_12 + jO1_02 * G1_22 + jO1_03 * G1_32
    jOG1_10 = jO1_10 * G1_00 + jO1_11 * G1_10 + jO1_12 * G1_20 + jO1_13 * G1_30
    jOG1_11 = jO1_10 * G1_01 + jO1_11 * G1_11 + jO1_12 * G1_21 + jO1_13 * G1_31
    jOG1_12 = jO1_10 * G1_02 + jO1_11 * G1_12 + jO1_12 * G1_22 + jO1_13 * G1_32
    jOG1_20 = jO1_20 * G1_00 + jO1_21 * G1_10 + jO1_22 * G1_20 + jO1_23 * G1_30
    jOG1_21 = jO1_20 * G1_01 + jO1_21 * G1_11 + jO1_22 * G1_21 + jO1_23 * G1_31
    jOG1_22 = jO1_20 * G1_02 + jO1_21 * G1_12 + jO1_22 * G1_22 + jO1_23 * G1_32

    # Position Jacobian (rows 0-2: position contribution)
    # J_fwd[0:3, 0:3] = I, J_bwd[0:3, 0:3] = -I
    jacobian_pos[jacobian_index(i, 0, 0)] = 1.0
    jacobian_pos[jacobian_index(i, 1, 1)] = 1.0
    jacobian_pos[jacobian_index(i, 2, 2)] = 1.0
    jacobian_pos[jacobian_index(i, 0, 3)] = -1.0
    jacobian_pos[jacobian_index(i, 1, 4)] = -1.0
    jacobian_pos[jacobian_index(i, 2, 5)] = -1.0

    # Rotation Jacobian (rows 0-2: rotation contribution via lever arm)
    # J_fwd[0:3, 3:6] = -skew(r0)
    jacobian_rot[jacobian_index(i, 0, 0)] = 0.0
    jacobian_rot[jacobian_index(i, 0, 1)] = r0.z
    jacobian_rot[jacobian_index(i, 0, 2)] = -r0.y
    jacobian_rot[jacobian_index(i, 1, 0)] = -r0.z
    jacobian_rot[jacobian_index(i, 1, 1)] = 0.0
    jacobian_rot[jacobian_index(i, 1, 2)] = r0.x
    jacobian_rot[jacobian_index(i, 2, 0)] = r0.y
    jacobian_rot[jacobian_index(i, 2, 1)] = -r0.x
    jacobian_rot[jacobian_index(i, 2, 2)] = 0.0

    # J_bwd[0:3, 3:6] = skew(r1)
    jacobian_rot[jacobian_index(i, 0, 3)] = 0.0
    jacobian_rot[jacobian_index(i, 0, 4)] = -r1.z
    jacobian_rot[jacobian_index(i, 0, 5)] = r1.y
    jacobian_rot[jacobian_index(i, 1, 3)] = r1.z
    jacobian_rot[jacobian_index(i, 1, 4)] = 0.0
    jacobian_rot[jacobian_index(i, 1, 5)] = -r1.x
    jacobian_rot[jacobian_index(i, 2, 3)] = -r1.y
    jacobian_rot[jacobian_index(i, 2, 4)] = r1.x
    jacobian_rot[jacobian_index(i, 2, 5)] = 0.0

    # Bend-twist Jacobians (rows 3-5)
    jacobian_rot[jacobian_index(i, 3, 0)] = jOG0_00
    jacobian_rot[jacobian_index(i, 3, 1)] = jOG0_01
    jacobian_rot[jacobian_index(i, 3, 2)] = jOG0_02
    jacobian_rot[jacobian_index(i, 4, 0)] = jOG0_10
    jacobian_rot[jacobian_index(i, 4, 1)] = jOG0_11
    jacobian_rot[jacobian_index(i, 4, 2)] = jOG0_12
    jacobian_rot[jacobian_index(i, 5, 0)] = jOG0_20
    jacobian_rot[jacobian_index(i, 5, 1)] = jOG0_21
    jacobian_rot[jacobian_index(i, 5, 2)] = jOG0_22

    jacobian_rot[jacobian_index(i, 3, 3)] = jOG1_00
    jacobian_rot[jacobian_index(i, 3, 4)] = jOG1_01
    jacobian_rot[jacobian_index(i, 3, 5)] = jOG1_02
    jacobian_rot[jacobian_index(i, 4, 3)] = jOG1_10
    jacobian_rot[jacobian_index(i, 4, 4)] = jOG1_11
    jacobian_rot[jacobian_index(i, 4, 5)] = jOG1_12
    jacobian_rot[jacobian_index(i, 5, 3)] = jOG1_20
    jacobian_rot[jacobian_index(i, 5, 4)] = jOG1_21
    jacobian_rot[jacobian_index(i, 5, 5)] = jOG1_22


# =============================================================================
# Assembly and solver kernels
# =============================================================================


@wp.kernel
def kernel_assemble_jmjt_banded(
    jacobian_pos: wp.array(dtype=wp.float32),
    jacobian_rot: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    n_dofs: int,
    # Output
    ab: wp.array(dtype=wp.float32, ndim=2),
):
    """Assemble JMJT into banded matrix storage.

    The Jacobians are stored as [n_edges * 6 * 6] with columns:
    - cols 0-2: segment i (forward) position/rotation
    - cols 3-5: segment i+1 (backward) position/rotation

    JMJT = J * J^T with unit mass/inertia.
    """
    i = wp.tid()
    block_start = 6 * i
    if block_start >= n_dofs:
        return

    kd = BAND_KD

    # Diagonal block
    for row in range(6):
        for col in range(6):
            val = float(0.0)
            # Sum over k=0,1,2 for each segment (position and rotation)
            for k in range(3):
                # Position Jacobians
                j_p0_r = jacobian_pos[jacobian_index(i, row, k)]
                j_p0_c = jacobian_pos[jacobian_index(i, col, k)]
                j_p1_r = jacobian_pos[jacobian_index(i, row, k + 3)]
                j_p1_c = jacobian_pos[jacobian_index(i, col, k + 3)]
                # Rotation Jacobians
                j_t0_r = jacobian_rot[jacobian_index(i, row, k)]
                j_t0_c = jacobian_rot[jacobian_index(i, col, k)]
                j_t1_r = jacobian_rot[jacobian_index(i, row, k + 3)]
                j_t1_c = jacobian_rot[jacobian_index(i, col, k + 3)]
                val = val + (
                    j_p0_r * j_p0_c
                    + j_p1_r * j_p1_c
                    + j_t0_r * j_t0_c
                    + j_t1_r * j_t1_c
                )
            if row == col:
                val = val + compliance[i * 6 + row]

            row_idx = block_start + row
            col_idx = block_start + col
            # Upper triangular storage for symmetric matrix
            if row_idx <= col_idx:
                band_row = kd + row_idx - col_idx
                if band_row >= 0 and band_row <= kd:
                    ab[band_row, col_idx] = val

    # Off-diagonal coupling with previous constraint
    if i > 0:
        prev = i - 1
        prev_block = block_start - 6
        for row in range(6):
            for col in range(6):
                val = float(0.0)
                for k in range(3):
                    # J_bwd[prev] @ J_fwd[i]^T
                    j_p1_prev = jacobian_pos[jacobian_index(prev, row, k + 3)]
                    j_p0_cur = jacobian_pos[jacobian_index(i, col, k)]
                    j_t1_prev = jacobian_rot[jacobian_index(prev, row, k + 3)]
                    j_t0_cur = jacobian_rot[jacobian_index(i, col, k)]
                    val = val + j_p1_prev * j_p0_cur + j_t1_prev * j_t0_cur

                row_idx = prev_block + row
                col_idx = block_start + col
                band_row = kd + row_idx - col_idx
                if band_row >= 0 and band_row <= kd:
                    ab[band_row, col_idx] = val


@wp.kernel
def kernel_build_rhs(
    constraint_values: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    lambda_sum: wp.array(dtype=wp.float32),
    n_dofs: int,
    # Output
    rhs: wp.array(dtype=wp.float32),
):
    """Build RHS vector: -C - alpha * lambda_sum."""
    i = wp.tid()
    if i < n_dofs:
        rhs[i] = -constraint_values[i]


@wp.kernel
def kernel_solve_banded_cholesky(
    n: int,
    ab: wp.array(dtype=wp.float32, ndim=2),
    b: wp.array(dtype=wp.float32),
):
    """In-place banded Cholesky solve (single thread)."""
    tid = wp.tid()
    if tid != 0:
        return

    kd = BAND_KD

    # Cholesky factorization
    for j in range(n):
        sum_sq = float(0.0)
        kmax = int(j)
        if kmax > kd:
            kmax = kd
        for k in range(1, kmax + 1):
            u = ab[kd - k, j]
            sum_sq = sum_sq + u * u

        ajj = ab[kd, j] - sum_sq
        if ajj <= 1.0e-6:
            ajj = float(1.0e-6)
        ujj = wp.sqrt(ajj)
        ab[kd, j] = ujj

        imax = n - j - 1
        if imax > kd:
            imax = kd
        for i_idx in range(1, imax + 1):
            dot = float(0.0)
            k2max = int(j)
            if k2max > kd - i_idx:
                k2max = kd - i_idx
            for k in range(1, k2max + 1):
                dot = dot + ab[kd - k, j] * ab[kd - i_idx - k, j + i_idx]
            aji = ab[kd - i_idx, j + i_idx] - dot
            ab[kd - i_idx, j + i_idx] = aji / ujj

    # Forward solve
    for i in range(n):
        sum_val = float(0.0)
        k0 = int(0)
        if i > kd:
            k0 = i - kd
        for k in range(k0, i):
            sum_val = sum_val + ab[kd + k - i, i] * b[k]
        b[i] = (b[i] - sum_val) / ab[kd, i]

    # Backward solve
    for i in range(n - 1, -1, -1):
        sum_val = float(0.0)
        k1 = i + kd
        if k1 > n - 1:
            k1 = n - 1
        for k in range(i + 1, k1 + 1):
            sum_val = sum_val + ab[kd + i - k, k] * b[k]
        b[i] = (b[i] - sum_val) / ab[kd, i]


@wp.kernel
def kernel_apply_corrections(
    predicted_positions: wp.array(dtype=wp.vec3),
    predicted_orientations: wp.array(dtype=wp.quat),
    inv_masses: wp.array(dtype=wp.float32),
    quat_inv_masses: wp.array(dtype=wp.float32),
    jacobian_pos: wp.array(dtype=wp.float32),
    jacobian_rot: wp.array(dtype=wp.float32),
    delta_lambda: wp.array(dtype=wp.float32),
    lambda_sum: wp.array(dtype=wp.float32),
    n_edges: int,
):
    """Apply corrections from constraint solve (single thread)."""
    tid = wp.tid()
    if tid != 0:
        return

    for edge in range(n_edges):
        base_idx = edge * 6
        dl0 = delta_lambda[base_idx + 0]
        dl1 = delta_lambda[base_idx + 1]
        dl2 = delta_lambda[base_idx + 2]
        dl3 = delta_lambda[base_idx + 3]
        dl4 = delta_lambda[base_idx + 4]
        dl5 = delta_lambda[base_idx + 5]

        # Accumulate lambda
        lambda_sum[base_idx + 0] = lambda_sum[base_idx + 0] + dl0
        lambda_sum[base_idx + 1] = lambda_sum[base_idx + 1] + dl1
        lambda_sum[base_idx + 2] = lambda_sum[base_idx + 2] + dl2
        lambda_sum[base_idx + 3] = lambda_sum[base_idx + 3] + dl3
        lambda_sum[base_idx + 4] = lambda_sum[base_idx + 4] + dl4
        lambda_sum[base_idx + 5] = lambda_sum[base_idx + 5] + dl5

        # Position corrections
        inv_m0 = inv_masses[edge]
        inv_m1 = inv_masses[edge + 1]

        if inv_m0 > 0.0:
            dp0_x = jacobian_dot(jacobian_pos, edge, 0, dl0, dl1, dl2, dl3, dl4, dl5)
            dp0_y = jacobian_dot(jacobian_pos, edge, 1, dl0, dl1, dl2, dl3, dl4, dl5)
            dp0_z = jacobian_dot(jacobian_pos, edge, 2, dl0, dl1, dl2, dl3, dl4, dl5)
            dp0 = wp.vec3(dp0_x * inv_m0, dp0_y * inv_m0, dp0_z * inv_m0)
            predicted_positions[edge] = predicted_positions[edge] + dp0

        if inv_m1 > 0.0:
            dp1_x = jacobian_dot(jacobian_pos, edge, 3, dl0, dl1, dl2, dl3, dl4, dl5)
            dp1_y = jacobian_dot(jacobian_pos, edge, 4, dl0, dl1, dl2, dl3, dl4, dl5)
            dp1_z = jacobian_dot(jacobian_pos, edge, 5, dl0, dl1, dl2, dl3, dl4, dl5)
            dp1 = wp.vec3(dp1_x * inv_m1, dp1_y * inv_m1, dp1_z * inv_m1)
            predicted_positions[edge + 1] = predicted_positions[edge + 1] + dp1

        # Rotation corrections
        if quat_inv_masses[edge] > 0.0:
            dtheta0 = wp.vec3(
                jacobian_dot(jacobian_rot, edge, 0, dl0, dl1, dl2, dl3, dl4, dl5),
                jacobian_dot(jacobian_rot, edge, 1, dl0, dl1, dl2, dl3, dl4, dl5),
                jacobian_dot(jacobian_rot, edge, 2, dl0, dl1, dl2, dl3, dl4, dl5),
            )
            predicted_orientations[edge] = quat_correction_g(
                predicted_orientations[edge], dtheta0
            )

        if quat_inv_masses[edge + 1] > 0.0:
            dtheta1 = wp.vec3(
                jacobian_dot(jacobian_rot, edge, 3, dl0, dl1, dl2, dl3, dl4, dl5),
                jacobian_dot(jacobian_rot, edge, 4, dl0, dl1, dl2, dl3, dl4, dl5),
                jacobian_dot(jacobian_rot, edge, 5, dl0, dl1, dl2, dl3, dl4, dl5),
            )
            predicted_orientations[edge + 1] = quat_correction_g(
                predicted_orientations[edge + 1], dtheta1
            )


@wp.kernel
def kernel_zero_array(arr: wp.array(dtype=wp.float32)):
    """Zero out a float array."""
    i = wp.tid()
    arr[i] = 0.0
