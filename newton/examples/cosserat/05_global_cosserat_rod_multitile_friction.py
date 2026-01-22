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
# Example Cosserat Rod with Internal Friction (Jacobi Iteration)
#
# Extends example 04 with three internal friction models for Cosserat rods.
#
# NOTE: Like example 04, this uses iterative Jacobi-style constraint
# projection, NOT a global matrix solve or tiled Cholesky. The "multi-tile"
# in the filename refers to the extended chain length (129 particles).
#
# Friction Models:
#
# 1. Velocity Damping (simplest):
#    - Directly damps particle velocities: v_new = v * damping_coeff
#    - Simple but not physically accurate, damps all motion uniformly
#
# 2. Strain-Rate Damping (Rayleigh-style):
#    - Damping proportional to curvature change rate
#    - f_damp = damping * k * d(kappa)/dt
#    - Physically motivated, only damps deformation
#
# 3. Dahl Hysteresis (most accurate):
#    - Path-dependent friction with hysteresis loops
#    - Models internal friction in cables/rods realistically
#    - Uses exponential evolution: sigma evolves based on curvature direction
#    - Parameters: eps_max (friction saturation), tau (memory decay)
#
# Command: uv run -m newton.examples cosserat_05_global_cosserat_rod_multitile_friction
#
###########################################################################

import math

import warp as wp

import newton
import newton.examples

# Configuration constants (TILE kept for consistency, not used for Cholesky here)
BLOCK_DIM = 128
TILE = 32  # Used for sizing, not for tiled Cholesky in this example

# Rod configuration - extended chain with 129 particles
NUM_TILES = 4
CONSTRAINTS_PER_TILE = 32
NUM_PARTICLES = NUM_TILES * TILE + 1  # 129 particles
NUM_STRETCH = NUM_PARTICLES - 1  # 128 stretch constraints
NUM_BEND = NUM_PARTICLES - 2  # 127 bend constraints

# Friction method constants
FRICTION_NONE = 0
FRICTION_VELOCITY = 1
FRICTION_STRAIN_RATE = 2
FRICTION_DAHL = 3

# Small constant for Dahl deadband
DAHL_DEADBAND = 1.0e-6


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

    In Eigen (w,x,y,z) notation: q_e3_bar = (q.z, -q.y, q.x, -q.w)
    In Warp (x,y,z,w) notation: result = (-q.y, q.x, -q.w, q.z)

    This is equivalent to: q * quat(0, 0, -1, 0)
    """
    return wp.quat(-q[1], q[0], -q[3], q[2])


@wp.func
def compute_darboux_vector(q0: wp.quat, q1: wp.quat, rest_darboux_q: wp.quat) -> wp.vec3:
    """Compute the Darboux vector (curvature) between two quaternions.

    Returns the imaginary part of omega - rest_darboux (choosing shorter path).
    """
    # Compute Darboux vector: omega = conjugate(q0) * q1
    q0_conj = wp.quat(-q0[0], -q0[1], -q0[2], q0[3])
    omega = wp.mul(q0_conj, q1)

    # Handle quaternion double-cover: choose the shorter path
    omega_plus_x = omega[0] + rest_darboux_q[0]
    omega_plus_y = omega[1] + rest_darboux_q[1]
    omega_plus_z = omega[2] + rest_darboux_q[2]
    omega_plus_w = omega[3] + rest_darboux_q[3]

    omega_minus_x = omega[0] - rest_darboux_q[0]
    omega_minus_y = omega[1] - rest_darboux_q[1]
    omega_minus_z = omega[2] - rest_darboux_q[2]
    omega_minus_w = omega[3] - rest_darboux_q[3]

    # Squared norms
    norm_plus_sq = (
        omega_plus_x * omega_plus_x
        + omega_plus_y * omega_plus_y
        + omega_plus_z * omega_plus_z
        + omega_plus_w * omega_plus_w
    )
    norm_minus_sq = (
        omega_minus_x * omega_minus_x
        + omega_minus_y * omega_minus_y
        + omega_minus_z * omega_minus_z
        + omega_minus_w * omega_minus_w
    )

    # Choose the smaller deviation
    if norm_minus_sq > norm_plus_sq:
        return wp.vec3(omega_plus_x, omega_plus_y, omega_plus_z)
    else:
        return wp.vec3(omega_minus_x, omega_minus_y, omega_minus_z)


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
        # Kinematic particle - don't move
        particle_q_predicted[tid] = particle_q[tid]
        particle_qd_new[tid] = particle_qd[tid]
        return

    # v_new = v + g * dt
    v_new = particle_qd[tid] + gravity * dt
    # x_predicted = x + v_new * dt
    x_predicted = particle_q[tid] + v_new * dt

    particle_q_predicted[tid] = x_predicted
    particle_qd_new[tid] = v_new


@wp.kernel
def apply_velocity_damping_kernel(
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    damping_coeff: float,
    # output
    particle_qd_out: wp.array(dtype=wp.vec3),
):
    """
    Apply velocity damping to particles.

    Method 1: Velocity Damping (simplest)
    Directly damps velocities: v_new = v * damping_coeff
    """
    tid = wp.tid()

    if particle_inv_mass[tid] == 0.0:
        particle_qd_out[tid] = particle_qd[tid]
        return

    particle_qd_out[tid] = particle_qd[tid] * damping_coeff


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
    """
    Solve stretch and shear constraint for Cosserat rods using Jacobi-style iteration.

    Corrections are accumulated using atomic adds, then applied in a separate pass.
    This allows parallel execution of all constraints.

    Constraint: gamma = (p1 - p0) / L - d3(q) = 0
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
def solve_bend_twist_constraint_kernel(
    edge_q: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    rest_darboux: wp.array(dtype=wp.quat),
    bend_twist_stiffness: wp.vec3,
    num_bend: int,
    # output: accumulated corrections
    edge_q_delta: wp.array(dtype=wp.quat),
):
    """
    Solve bend and twist constraint for Cosserat rods using Jacobi-style iteration.

    Corrections are accumulated using atomic adds, then applied in a separate pass.

    Darboux vector omega = conjugate(q0) * q1
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
    """
    Solve bend and twist constraint with strain-rate damping (Method 2: Rayleigh-style).

    Adds damping proportional to curvature change rate:
    f_damp = damping * k * d(kappa)/dt

    This only damps bending/twisting deformation, not rigid body motion.
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
    """
    Solve bend and twist constraint with Dahl hysteresis friction (Method 3).

    Dahl model captures path-dependent friction with hysteresis:
    - sigma (friction stress) evolves based on curvature change direction
    - sigma_max = k * eps_max (friction envelope)
    - tau controls memory decay (how quickly friction "forgets" direction changes)

    Evolution: sigma_new = s * sigma_max * (1 - exp(-s*d_kappa/tau)) + sigma_prev * exp(-s*d_kappa/tau)
    where s = sign(d_kappa) is the direction flag.
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
def apply_particle_corrections_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_delta: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    # output
    particle_q_out: wp.array(dtype=wp.vec3),
):
    """Apply accumulated position corrections to particles."""
    tid = wp.tid()

    inv_mass = particle_inv_mass[tid]
    if inv_mass == 0.0:
        particle_q_out[tid] = particle_q[tid]
        return

    delta = particle_delta[tid]
    particle_q_out[tid] = particle_q[tid] + delta


@wp.kernel
def apply_quaternion_corrections_kernel(
    edge_q: wp.array(dtype=wp.quat),
    edge_q_delta: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    # output
    edge_q_out: wp.array(dtype=wp.quat),
):
    """Apply accumulated quaternion corrections and normalize."""
    tid = wp.tid()

    inv_mass = edge_inv_mass[tid]
    if inv_mass == 0.0:
        edge_q_out[tid] = edge_q[tid]
        return

    q = edge_q[tid]
    dq = edge_q_delta[tid]

    # Add correction
    q_new = wp.quat(q[0] + dq[0], q[1] + dq[1], q[2] + dq[2], q[3] + dq[3])

    # Normalize to maintain unit quaternion
    q_new = wp.normalize(q_new)

    edge_q_out[tid] = q_new


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
    """
    Update rest Darboux vectors to define the rod's rest shape.

    The Darboux vector represents the relative rotation between adjacent frames.
    For small angles, we use: q ≈ (sin(θ/2)*axis, cos(θ/2)) ≈ (θ/2*axis, 1) for small θ

    - rest_bend_d1: bending rate around d1 axis (curvature in d2-d3 plane)
    - rest_bend_d2: bending rate around d2 axis (curvature in d1-d3 plane)
    - rest_twist: twist rate around d3 axis

    Values are in radians per edge segment.
    """
    tid = wp.tid()
    if tid >= num_bend:
        return

    # Build quaternion from axis-angle: half-angles for quaternion representation
    half_bend_d1 = rest_bend_d1 * 0.5
    half_bend_d2 = rest_bend_d2 * 0.5
    half_twist = rest_twist * 0.5

    # For small angles, sin(θ/2) ≈ θ/2 and cos(θ/2) ≈ 1 - θ²/8
    # Use exact formula for robustness
    angle_sq = half_bend_d1 * half_bend_d1 + half_bend_d2 * half_bend_d2 + half_twist * half_twist
    angle = wp.sqrt(angle_sq)

    if angle < 1.0e-8:
        # Near-identity: use limit formula
        rest_darboux[tid] = wp.quat(0.0, 0.0, 0.0, 1.0)
    else:
        # Quaternion from rotation vector (bend_d1, bend_d2, twist)
        # The rotation vector in the material frame corresponds to:
        # x -> d1 axis (bending)
        # y -> d2 axis (bending)
        # z -> d3 axis (twist)
        s = wp.sin(angle) / angle
        c = wp.cos(angle)
        rest_darboux[tid] = wp.quat(s * half_bend_d1, s * half_bend_d2, s * half_twist, c)


@wp.kernel
def compute_director_lines_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    edge_q: wp.array(dtype=wp.quat),
    num_edges: int,
    axis_length: float,
    # outputs: 3 axes per edge (d1=red, d2=green, d3=blue)
    line_starts: wp.array(dtype=wp.vec3),
    line_ends: wp.array(dtype=wp.vec3),
    line_colors: wp.array(dtype=wp.vec3),
):
    """
    Compute line segments for visualizing material frames.
    Each edge has 3 axes (d1, d2, d3) drawn from its midpoint.
    """
    tid = wp.tid()
    edge_idx = tid // 3
    axis_idx = tid % 3  # 0=d1(red), 1=d2(green), 2=d3(blue)

    if edge_idx >= num_edges:
        return

    # Compute edge midpoint
    p0 = particle_q[edge_idx]
    p1 = particle_q[edge_idx + 1]
    midpoint = (p0 + p1) * 0.5

    q = edge_q[edge_idx]

    # Compute the director based on axis_idx
    if axis_idx == 0:
        # d1 = q * e1 * conj(q) where e1 = (1,0,0)
        x, y, z, w = q[0], q[1], q[2], q[3]
        d1_x = w * w + x * x - y * y - z * z
        d1_y = 2.0 * (x * y + w * z)
        d1_z = 2.0 * (x * z - w * y)
        director = wp.vec3(d1_x, d1_y, d1_z)
        color = wp.vec3(1.0, 0.0, 0.0)  # Red
    elif axis_idx == 1:
        # d2 = q * e2 * conj(q) where e2 = (0,1,0)
        x, y, z, w = q[0], q[1], q[2], q[3]
        d2_x = 2.0 * (x * y - w * z)
        d2_y = w * w - x * x + y * y - z * z
        d2_z = 2.0 * (y * z + w * x)
        director = wp.vec3(d2_x, d2_y, d2_z)
        color = wp.vec3(0.0, 1.0, 0.0)  # Green
    else:
        # d3 = q * e3 * conj(q) where e3 = (0,0,1)
        x, y, z, w = q[0], q[1], q[2], q[3]
        d3_x = 2.0 * (x * z + w * y)
        d3_y = 2.0 * (y * z - w * x)
        d3_z = w * w - x * x - y * y + z * z
        director = wp.vec3(d3_x, d3_y, d3_z)
        color = wp.vec3(0.0, 0.0, 1.0)  # Blue

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


@wp.kernel
def compute_current_kappa_kernel(
    edge_q: wp.array(dtype=wp.quat),
    rest_darboux: wp.array(dtype=wp.quat),
    num_bend: int,
    # output
    kappa_out: wp.array(dtype=wp.vec3),
):
    """Compute and store current curvature for all bend constraints."""
    tid = wp.tid()
    if tid >= num_bend:
        return

    q0 = edge_q[tid]
    q1 = edge_q[tid + 1]
    rest_darboux_q = rest_darboux[tid]

    kappa = compute_darboux_vector(q0, q1, rest_darboux_q)
    kappa_out[tid] = kappa


class Example:
    def __init__(self, viewer, args=None):
        # Simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.constraint_iterations = 6  # More iterations for block-Jacobi coupling

        self.viewer = viewer
        self.args = args

        # Rod parameters
        self.num_particles = NUM_PARTICLES
        self.num_stretch = NUM_STRETCH
        self.num_bend = NUM_BEND
        self.num_tiles = NUM_TILES
        self.constraints_per_tile = CONSTRAINTS_PER_TILE

        particle_spacing = 0.05  # rest length between particles
        particle_mass = 0.1
        particle_radius = 0.015
        edge_mass = 0.01  # mass associated with each quaternion
        start_height = 5.0  # Higher start for longer rod

        # Stiffness parameters (0 to 1, where 1 = full constraint projection)
        # These map to PBD stiffness: 1.0 means apply full correction per iteration
        self.stretch_stiffness = 1.0  # Inextensibility
        self.shear_stiffness = 1.0  # d3 director alignment with edge
        self.bend_stiffness = 0.1  # Resist bending
        self.twist_stiffness = 0.1  # Resist twisting

        # Rest shape parameters (Darboux vector components)
        # These control the rod's rest configuration
        self.rest_bend_d1 = 0.0  # Bending rate around d1 axis (rad/segment)
        self.rest_bend_d2 = 0.0  # Bending rate around d2 axis (rad/segment)
        self.rest_twist = 0.0  # Twist rate around d3 axis (rad/segment)

        # Friction parameters
        self.friction_method = FRICTION_NONE  # 0=None, 1=Velocity, 2=Strain-rate, 3=Dahl
        self.velocity_damping = 0.99  # Method 1: velocity damping coefficient (0-1)
        self.strain_rate_damping = 0.1  # Method 2: strain-rate damping coefficient
        self.dahl_eps_max = 0.01  # Method 3: maximum persistent strain [rad]
        self.dahl_tau = 0.005  # Method 3: memory decay length [rad]

        self.gravity = wp.vec3(0.0, 0.0, -9.81)

        # Build the model
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        

        # Create particles: first one is fixed (kinematic)
        for i in range(self.num_particles):
            mass = 0.0 if i == 0 else particle_mass
            builder.add_particle(
                pos=(i * particle_spacing, 0.0, start_height),
                vel=(0.0, 0.0, 0.0),
                mass=mass,
                radius=particle_radius,
            )

        self.model = builder.finalize()

        # Soft contact parameters for particle-ground collision
        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 1.0e1
        self.model.soft_contact_mu = 0.5

        # State buffers
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Collision pipeline
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, self.args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        device = self.model.device

        # Particle inverse mass array
        inv_mass_np = [0.0] + [1.0 / particle_mass] * (self.num_particles - 1)
        self.particle_inv_mass = wp.array(inv_mass_np, dtype=float, device=device)

        # Edge quaternions (initialized to rotate z-axis to x-axis for horizontal rod)
        angle = math.pi / 2.0
        q_init = wp.quat(0.0, math.sin(angle / 2.0), 0.0, math.cos(angle / 2.0))
        edge_q_init = [q_init] * self.num_stretch
        self.edge_q = wp.array(edge_q_init, dtype=wp.quat, device=device)
        self.edge_q_new = wp.array(edge_q_init, dtype=wp.quat, device=device)

        # Edge inverse masses
        edge_inv_mass_np = [1.0 / edge_mass] * self.num_stretch
        self.edge_inv_mass = wp.array(edge_inv_mass_np, dtype=float, device=device)

        # Rest lengths for stretch constraints
        rest_length_np = [particle_spacing] * self.num_stretch
        self.rest_length = wp.array(rest_length_np, dtype=float, device=device)

        # Rest Darboux vectors for bend/twist constraints (identity quaternion for straight rod)
        # Identity quaternion (0,0,0,1) in Warp (x,y,z,w) format
        rest_darboux_np = [wp.quat(0.0, 0.0, 0.0, 1.0)] * self.num_bend
        self.rest_darboux = wp.array(rest_darboux_np, dtype=wp.quat, device=device)

        # Temporary buffer for predicted positions
        self.particle_q_predicted = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)
        self.particle_q_temp = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)
        self.particle_qd_temp = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)

        # Correction accumulators for Jacobi-style iteration
        self.particle_delta = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)
        self.edge_q_delta = wp.zeros(self.num_stretch, dtype=wp.quat, device=device)

        # Friction state arrays (for strain-rate damping and Dahl hysteresis)
        self.kappa_prev = wp.zeros(self.num_bend, dtype=wp.vec3, device=device)
        self.kappa_current = wp.zeros(self.num_bend, dtype=wp.vec3, device=device)
        self.sigma_prev = wp.zeros(self.num_bend, dtype=wp.vec3, device=device)
        self.sigma_current = wp.zeros(self.num_bend, dtype=wp.vec3, device=device)
        self.dkappa_prev = wp.zeros(self.num_bend, dtype=wp.vec3, device=device)
        self.dkappa_current = wp.zeros(self.num_bend, dtype=wp.vec3, device=device)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        # Director visualization buffers (3 axes per edge)
        num_director_lines = self.num_stretch * 3
        self.director_line_starts = wp.zeros(num_director_lines, dtype=wp.vec3, device=device)
        self.director_line_ends = wp.zeros(num_director_lines, dtype=wp.vec3, device=device)
        self.director_line_colors = wp.zeros(num_director_lines, dtype=wp.vec3, device=device)
        self.show_directors = True
        self.director_scale = 0.03  # Length of director axes (smaller for longer rod)

        self.graph = None

        # Keyboard control for the first locked particle
        self.particle_move_speed = 1.0  # units per second

        # Initialize kappa_prev with current curvature
        self._initialize_kappa()

    def _initialize_kappa(self):
        """Initialize curvature state arrays."""
        wp.launch(
            kernel=compute_current_kappa_kernel,
            dim=self.num_bend,
            inputs=[self.edge_q, self.rest_darboux, self.num_bend],
            outputs=[self.kappa_prev],
            device=self.model.device,
        )

    def simulate(self):
        for _ in range(self.sim_substeps):
            # Store old positions for velocity update
            wp.copy(self.particle_q_temp, self.state_0.particle_q)

            # Build stiffness vectors from UI parameters
            # Stretch/shear: (shear_x, shear_y, stretch_z) - z is along rod axis
            stretch_shear_ks = wp.vec3(self.shear_stiffness, self.shear_stiffness, self.stretch_stiffness)
            # Bend/twist: (bend_x, twist_y, bend_z)
            bend_twist_ks = wp.vec3(self.bend_stiffness, self.twist_stiffness, self.bend_stiffness)

            # Step 1: Integrate (predict positions)
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

            # Copy predicted to state_1 for constraint solving
            wp.copy(self.state_1.particle_q, self.particle_q_predicted)

            # Step 2: Iterative Jacobi-style constraint solving
            for _ in range(self.constraint_iterations):
                # Zero out correction accumulators
                wp.launch(
                    kernel=zero_vec3_kernel,
                    dim=self.num_particles,
                    inputs=[self.particle_delta],
                    device=self.model.device,
                )
                wp.launch(
                    kernel=zero_quat_kernel,
                    dim=self.num_stretch,
                    inputs=[self.edge_q_delta],
                    device=self.model.device,
                )

                # Solve stretch/shear constraints (accumulates corrections)
                wp.launch(
                    kernel=solve_stretch_shear_constraint_kernel,
                    dim=self.num_stretch,
                    inputs=[
                        self.state_1.particle_q,
                        self.particle_inv_mass,
                        self.edge_q,
                        self.edge_inv_mass,
                        self.rest_length,
                        stretch_shear_ks,
                        self.num_stretch,
                    ],
                    outputs=[self.particle_delta, self.edge_q_delta],
                    device=self.model.device,
                )

                # Solve bend/twist constraints with selected friction method
                if self.num_bend > 0:
                    if self.friction_method == FRICTION_STRAIN_RATE:
                        # Method 2: Strain-rate damping
                        wp.launch(
                            kernel=solve_bend_twist_with_strain_rate_damping_kernel,
                            dim=self.num_bend,
                            inputs=[
                                self.edge_q,
                                self.edge_inv_mass,
                                self.rest_darboux,
                                bend_twist_ks,
                                self.kappa_prev,
                                self.strain_rate_damping,
                                self.sim_dt,
                                self.num_bend,
                            ],
                            outputs=[self.edge_q_delta, self.kappa_current],
                            device=self.model.device,
                        )
                    elif self.friction_method == FRICTION_DAHL:
                        # Method 3: Dahl hysteresis friction
                        wp.launch(
                            kernel=solve_bend_twist_with_dahl_friction_kernel,
                            dim=self.num_bend,
                            inputs=[
                                self.edge_q,
                                self.edge_inv_mass,
                                self.rest_darboux,
                                bend_twist_ks,
                                self.kappa_prev,
                                self.sigma_prev,
                                self.dkappa_prev,
                                self.dahl_eps_max,
                                self.dahl_tau,
                                self.num_bend,
                            ],
                            outputs=[
                                self.edge_q_delta,
                                self.kappa_current,
                                self.sigma_current,
                                self.dkappa_current,
                            ],
                            device=self.model.device,
                        )
                    else:
                        # No friction or Method 1 (velocity damping applied separately)
                        wp.launch(
                            kernel=solve_bend_twist_constraint_kernel,
                            dim=self.num_bend,
                            inputs=[
                                self.edge_q,
                                self.edge_inv_mass,
                                self.rest_darboux,
                                bend_twist_ks,
                                self.num_bend,
                            ],
                            outputs=[self.edge_q_delta],
                            device=self.model.device,
                        )

                # Apply accumulated position corrections
                wp.launch(
                    kernel=apply_particle_corrections_kernel,
                    dim=self.num_particles,
                    inputs=[
                        self.state_1.particle_q,
                        self.particle_delta,
                        self.particle_inv_mass,
                    ],
                    outputs=[self.particle_q_predicted],
                    device=self.model.device,
                )
                wp.copy(self.state_1.particle_q, self.particle_q_predicted)

                # Apply accumulated quaternion corrections
                wp.launch(
                    kernel=apply_quaternion_corrections_kernel,
                    dim=self.num_stretch,
                    inputs=[
                        self.edge_q,
                        self.edge_q_delta,
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

            # Step 4: Update velocities from position change
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

            # Step 5: Apply velocity damping (Method 1) if selected
            if self.friction_method == FRICTION_VELOCITY:
                wp.launch(
                    kernel=apply_velocity_damping_kernel,
                    dim=self.num_particles,
                    inputs=[
                        self.state_1.particle_qd,
                        self.particle_inv_mass,
                        self.velocity_damping,
                    ],
                    outputs=[self.particle_qd_temp],
                    device=self.model.device,
                )
                wp.copy(self.state_1.particle_qd, self.particle_qd_temp)

            # Update friction state for next substep (for strain-rate and Dahl)
            if self.friction_method == FRICTION_STRAIN_RATE:
                wp.copy(self.kappa_prev, self.kappa_current)
            elif self.friction_method == FRICTION_DAHL:
                wp.copy(self.kappa_prev, self.kappa_current)
                wp.copy(self.sigma_prev, self.sigma_current)
                wp.copy(self.dkappa_prev, self.dkappa_current)

            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

            # Update contacts for visualization
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

    def step(self):
        # Handle keyboard input to move the first locked particle
        self._handle_keyboard_input()

        self.simulate()
        self.sim_time += self.frame_dt

    def _handle_keyboard_input(self):
        """Move the first locked particle using numpad keys.

        Controls:
            - Numpad 4/6: Move in X direction (left/right)
            - Numpad 8/2: Move in Y direction (forward/back)
            - Numpad 9/3: Move in Z direction (up/down)
        """
        if not hasattr(self.viewer, "is_key_down"):
            return

        # Import pyglet for numpad key constants
        try:
            import pyglet.window.key as key
        except ImportError:
            return

        # Calculate movement delta based on numpad keys
        dx = 0.0
        dy = 0.0
        dz = 0.0

        # Numpad 4/6 for X movement (left/right)
        if self.viewer.is_key_down(key.NUM_6):
            dx += self.particle_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_4):
            dx -= self.particle_move_speed * self.frame_dt

        # Numpad 8/2 for Y movement (forward/back)
        if self.viewer.is_key_down(key.NUM_8):
            dy += self.particle_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_2):
            dy -= self.particle_move_speed * self.frame_dt

        # Numpad 9/3 for Z movement (up/down)
        if self.viewer.is_key_down(key.NUM_9):
            dz += self.particle_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_3):
            dz -= self.particle_move_speed * self.frame_dt

        # Apply movement if any key was pressed
        if dx != 0.0 or dy != 0.0 or dz != 0.0:
            # Get current position of particle 0
            particle_q_np = self.state_0.particle_q.numpy()
            pos = particle_q_np[0]

            # Update position
            new_pos = [pos[0] + dx, pos[1] + dy, pos[2] + dz]

            # Clamp Z to stay above ground
            if new_pos[2] < 0.1:
                new_pos[2] = 0.1

            particle_q_np[0] = new_pos

            # Write back to device
            self.state_0.particle_q = wp.array(particle_q_np, dtype=wp.vec3, device=self.model.device)

    def test_final(self):
        # Verify that the anchor particle (first particle) is still at rest
        newton.examples.test_particle_state(
            self.state_0,
            "anchor particle is stationary",
            lambda q, qd: wp.length(qd) < 1e-6,
            indices=[0],
        )

        # Verify all particles are above the ground
        newton.examples.test_particle_state(
            self.state_0,
            "particles are above the ground",
            lambda q, qd: q[2] >= -0.01,
        )

        # Verify particles are within reasonable bounds
        # Rod has 129 particles with 0.05 spacing (~6.4m length)
        p_lower = wp.vec3(-3.0, -5.0, -0.1)
        p_upper = wp.vec3(10.0, 5.0, 7.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within reasonable bounds",
            lambda q, qd: newton.utils.vec_inside_limits(q, p_lower, p_upper),
        )

        # Verify edge quaternions are normalized
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
        ui.text("Cosserat Rod with Internal Friction")
        ui.text(f"Particles: {self.num_particles}, Tiles: {self.num_tiles}")

        ui.separator()
        ui.text("Stiffness Parameters")
        _changed, self.stretch_stiffness = ui.slider_float("Stretch Stiffness", self.stretch_stiffness, 0.0, 1.0)
        _changed, self.shear_stiffness = ui.slider_float("Shear Stiffness", self.shear_stiffness, 0.0, 1.0)
        _changed, self.bend_stiffness = ui.slider_float("Bend Stiffness", self.bend_stiffness, 0.0, 1.0)
        _changed, self.twist_stiffness = ui.slider_float("Twist Stiffness", self.twist_stiffness, 0.0, 1.0)

        ui.separator()
        ui.text("Rest Shape (Darboux Vector)")
        changed_d1, self.rest_bend_d1 = ui.slider_float("Rest Bend d1", self.rest_bend_d1, -0.5, 0.5)
        changed_d2, self.rest_bend_d2 = ui.slider_float("Rest Bend d2", self.rest_bend_d2, -0.5, 0.5)
        changed_twist, self.rest_twist = ui.slider_float("Rest Twist", self.rest_twist, -0.5, 0.5)
        # Update rest Darboux vectors when sliders change
        if changed_d1 or changed_d2 or changed_twist:
            self._update_rest_darboux()

        ui.separator()
        ui.text("Internal Friction")

        # Friction method selector
        friction_methods = ["None", "Velocity Damping", "Strain-Rate Damping", "Dahl Hysteresis"]
        _changed, self.friction_method = ui.combo("Friction Method", self.friction_method, friction_methods)

        # Show relevant parameters based on selected method
        if self.friction_method == FRICTION_VELOCITY:
            ui.text("Method 1: Velocity Damping (simplest)")
            _changed, self.velocity_damping = ui.slider_float("Velocity Damping", self.velocity_damping, 0.9, 1.0)

        elif self.friction_method == FRICTION_STRAIN_RATE:
            ui.text("Method 2: Strain-Rate Damping (Rayleigh)")
            _changed, self.strain_rate_damping = ui.slider_float(
                "Strain-Rate Damping", self.strain_rate_damping, 0.0, 1.0
            )

        elif self.friction_method == FRICTION_DAHL:
            ui.text("Method 3: Dahl Hysteresis (most accurate)")
            _changed, self.dahl_eps_max = ui.slider_float("Eps Max (friction saturation)", self.dahl_eps_max, 0.0, 0.1)
            _changed, self.dahl_tau = ui.slider_float("Tau (memory decay)", self.dahl_tau, 0.001, 0.1)

        ui.separator()
        ui.text("Anchor Control (First Particle)")
        ui.text("  Numpad 4/6: Move left/right (X)")
        ui.text("  Numpad 8/2: Move forward/back (Y)")
        ui.text("  Numpad 9/3: Move up/down (Z)")
        _changed, self.particle_move_speed = ui.slider_float("Move Speed", self.particle_move_speed, 0.1, 5.0)

        ui.separator()
        ui.text("Visualization")
        _changed, self.show_directors = ui.checkbox("Show Directors", self.show_directors)
        _changed, self.director_scale = ui.slider_float("Director Scale", self.director_scale, 0.01, 0.1)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)

        # Visualize material frames (directors)
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
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Enable particle visualization if using GL viewer
    if isinstance(viewer, newton.viewer.ViewerGL):
        viewer.show_particles = True

    # Create and run example
    example = Example(viewer, args)

    newton.examples.run(example, args)
