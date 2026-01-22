# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Integration and velocity update kernels for Cosserat rod simulations."""

import warp as wp


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
    """Semi-implicit Euler integration step for particles.

    Args:
        particle_q: Current particle positions.
        particle_qd: Current particle velocities.
        particle_inv_mass: Inverse mass per particle (0 = kinematic).
        gravity: Gravity acceleration vector.
        dt: Time step.
        particle_q_predicted: Output predicted positions.
        particle_qd_new: Output updated velocities.
    """
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
def update_velocities_kernel(
    particle_q_old: wp.array(dtype=wp.vec3),
    particle_q_new: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    dt: float,
    # output
    particle_qd: wp.array(dtype=wp.vec3),
):
    """Update velocities from position change: v = (x_new - x_old) / dt.

    Args:
        particle_q_old: Old particle positions (before constraint solving).
        particle_q_new: New particle positions (after constraint solving).
        particle_inv_mass: Inverse mass per particle (0 = kinematic).
        dt: Time step.
        particle_qd: Output updated velocities.
    """
    tid = wp.tid()

    if particle_inv_mass[tid] == 0.0:
        particle_qd[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    delta_x = particle_q_new[tid] - particle_q_old[tid]
    particle_qd[tid] = delta_x / dt


@wp.kernel
def apply_velocity_damping_kernel(
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    damping_coeff: float,
    # output
    particle_qd_out: wp.array(dtype=wp.vec3),
):
    """Apply velocity damping to particles.

    Method 1: Velocity Damping (simplest)
    Directly damps velocities: v_new = v * damping_coeff

    Args:
        particle_qd: Current particle velocities.
        particle_inv_mass: Inverse mass per particle (0 = kinematic).
        damping_coeff: Damping coefficient (0 to 1, where 1 = no damping).
        particle_qd_out: Output damped velocities.
    """
    tid = wp.tid()

    if particle_inv_mass[tid] == 0.0:
        particle_qd_out[tid] = particle_qd[tid]
        return

    particle_qd_out[tid] = particle_qd[tid] * damping_coeff


@wp.kernel
def apply_quaternion_damping_kernel(
    edge_q_old: wp.array(dtype=wp.quat),
    edge_q_new: wp.array(dtype=wp.quat),
    edge_inv_mass: wp.array(dtype=float),
    damping_coeff: float,
    # output
    edge_q_out: wp.array(dtype=wp.quat),
):
    """Apply angular velocity damping to quaternions.

    Damps the rotation change between old and new quaternions:
    q_damped = slerp(q_old, q_new, damping_coeff)

    For small rotations, this is approximated as:
    q_damped = normalize(q_old + damping_coeff * (q_new - q_old))

    Args:
        edge_q_old: Old edge quaternions (before constraint solving).
        edge_q_new: New edge quaternions (after constraint solving).
        edge_inv_mass: Inverse mass per edge (0 = kinematic).
        damping_coeff: Damping coefficient (0 to 1, where 1 = no damping).
        edge_q_out: Output damped quaternions.
    """
    tid = wp.tid()

    if edge_inv_mass[tid] == 0.0:
        edge_q_out[tid] = edge_q_new[tid]
        return

    q_old = edge_q_old[tid]
    q_new = edge_q_new[tid]

    # Ensure we interpolate along the shorter path (handle quaternion double-cover)
    dot = q_old[0] * q_new[0] + q_old[1] * q_new[1] + q_old[2] * q_new[2] + q_old[3] * q_new[3]
    if dot < 0.0:
        # Flip q_new to take shorter path
        q_new = wp.quat(-q_new[0], -q_new[1], -q_new[2], -q_new[3])

    # Linear interpolation (lerp) followed by normalization
    # This is a good approximation for small rotations
    t = damping_coeff
    q_damped = wp.quat(
        q_old[0] + t * (q_new[0] - q_old[0]),
        q_old[1] + t * (q_new[1] - q_old[1]),
        q_old[2] + t * (q_new[2] - q_old[2]),
        q_old[3] + t * (q_new[3] - q_old[3]),
    )

    edge_q_out[tid] = wp.normalize(q_damped)
