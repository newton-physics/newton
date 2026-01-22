# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Utility functions and kernels for Cosserat rod simulations.

Contains quaternion helpers and correction application kernels.
"""

import warp as wp


@wp.func
def quat_rotate_e3(q: wp.quat) -> wp.vec3:
    """Compute the third director d3 = q * e3 * conjugate(q) where e3 = (0,0,1).

    This is an optimized computation of rotating the z-axis by quaternion q.

    Args:
        q: Unit quaternion in (x, y, z, w) format.

    Returns:
        The rotated z-axis vector (third director d3).
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

    This is equivalent to: q * quat(0, 0, -1, 0)

    Args:
        q: Unit quaternion in (x, y, z, w) format.

    Returns:
        The product quaternion.
    """
    return wp.quat(-q[1], q[0], -q[3], q[2])


@wp.func
def compute_darboux_vector(q0: wp.quat, q1: wp.quat, rest_darboux_q: wp.quat) -> wp.vec3:
    """Compute the Darboux vector (curvature) between two quaternions.

    Returns the imaginary part of omega - rest_darboux (choosing shorter path).

    Args:
        q0: First quaternion (edge frame).
        q1: Second quaternion (adjacent edge frame).
        rest_darboux_q: Rest Darboux vector as quaternion.

    Returns:
        The Darboux vector (imaginary part of relative rotation minus rest).
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
def zero_vec3_kernel(arr: wp.array(dtype=wp.vec3)):
    """Zero out a vec3 array.

    Args:
        arr: The array to zero out.
    """
    tid = wp.tid()
    arr[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def zero_quat_kernel(arr: wp.array(dtype=wp.quat)):
    """Zero out a quaternion array.

    Args:
        arr: The array to zero out.
    """
    tid = wp.tid()
    arr[tid] = wp.quat(0.0, 0.0, 0.0, 0.0)


@wp.kernel
def apply_particle_corrections_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_delta: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    # output
    particle_q_out: wp.array(dtype=wp.vec3),
):
    """Apply accumulated position corrections to particles.

    Args:
        particle_q: Current particle positions.
        particle_delta: Accumulated position corrections.
        particle_inv_mass: Inverse mass per particle (0 = kinematic).
        particle_q_out: Output corrected positions.
    """
    tid = wp.tid()

    inv_mass = particle_inv_mass[tid]
    if inv_mass == 0.0:
        particle_q_out[tid] = particle_q[tid]
        return

    delta = particle_delta[tid]
    particle_q_out[tid] = particle_q[tid] + delta


@wp.kernel
def apply_particle_corrections_with_velocity_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_delta: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    dt: float,
    # outputs
    particle_q_out: wp.array(dtype=wp.vec3),
    particle_qd_out: wp.array(dtype=wp.vec3),
):
    """Apply accumulated position corrections and update velocities.

    This is the local iterative approach from 02_local_cosserat_rod.py where
    velocity is updated incrementally based on position corrections during
    constraint solving iterations.

    Args:
        particle_q: Current particle positions.
        particle_qd: Current particle velocities.
        particle_delta: Accumulated position corrections.
        particle_inv_mass: Inverse mass per particle (0 = kinematic).
        dt: Time step for velocity update.
        particle_q_out: Output corrected positions.
        particle_qd_out: Output updated velocities.
    """
    tid = wp.tid()

    inv_mass = particle_inv_mass[tid]
    if inv_mass == 0.0:
        particle_q_out[tid] = particle_q[tid]
        particle_qd_out[tid] = particle_qd[tid]
        return

    delta = particle_delta[tid]
    q_new = particle_q[tid] + delta

    # Update velocity based on position change
    qd_new = particle_qd[tid] + delta / dt

    particle_q_out[tid] = q_new
    particle_qd_out[tid] = qd_new


@wp.kernel
def apply_quaternion_corrections_kernel(
    edge_q: wp.array(dtype=wp.quat),
    edge_q_delta: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    # output
    edge_q_out: wp.array(dtype=wp.quat),
):
    """Apply accumulated quaternion corrections and normalize.

    Args:
        edge_q: Current edge quaternions.
        edge_q_delta: Accumulated quaternion corrections.
        edge_inv_mass: Inverse mass per edge (0 = kinematic).
        edge_q_out: Output corrected quaternions (normalized).
    """
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
