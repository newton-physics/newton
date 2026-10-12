# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent contact-law reference fixtures, not a simulation backend."""

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.contact_compliance import normal_coefficients


@wp.kernel
def solve_normal_problem(
    delassus: wp.array3d[float],
    velocity_free: wp.array2d[float],
    bias: wp.array2d[float],
    gamma: wp.array2d[float],
    iterations: int,
    impulses: wp.array2d[float],
):
    """Solve independent worlds of coupled compliant normal rows with PGS."""
    world = wp.tid()
    count = impulses.shape[1]
    for _it in range(iterations):
        for row in range(count):
            old = impulses[world, row]
            residual = velocity_free[world, row] + bias[world, row] + gamma[world, row] * old
            for other in range(count):
                residual += delassus[world, row, other] * impulses[world, other]
            diagonal = delassus[world, row, row] + gamma[world, row]
            impulses[world, row] = wp.max(0.0, old - residual / diagonal)


@wp.kernel
def support_trajectory(
    mass: float,
    gravity: float,
    stiffness: float,
    damping: float,
    dt: float,
    initial_position: float,
    initial_velocity: float,
    iterations: int,
    compliant: int,
    trajectory: wp.array2d[float],
):
    """Integrate one prescribed-plane supported mass using the same PGS law.

    The plane-contact normal row is always a candidate; positive gap produces
    no attractive impulse. This is an analytical contact fixture, not Newton
    collision detection. The hard control uses the current beta=0.05 law.
    """
    x = initial_position
    v = initial_velocity
    inverse_mass = 1.0 / mass
    for step in range(trajectory.shape[0]):
        free_v = v - gravity * dt
        gamma = float(0.0)
        bias = wp.min(x, 0.0) * 0.05 / dt + wp.max(x, 0.0) / dt
        if compliant != 0:
            denominator = dt * stiffness + wp.where(x <= 0.0, damping, 0.0)
            gamma = 1.0 / (dt * denominator)
            bias = stiffness * x / denominator
        impulse = float(0.0)
        for _it in range(iterations):
            residual = free_v + inverse_mass * impulse + bias + gamma * impulse
            impulse = wp.max(0.0, impulse - residual / (inverse_mass + gamma))
        v = free_v + inverse_mass * impulse
        x += dt * v
        trajectory[step, 0] = x
        trajectory[step, 1] = v
        trajectory[step, 2] = impulse / dt


def solve_reference(delassus, velocity_free, separation, stiffness, damping, *, dt, iterations=128, device="cpu"):
    """Solve a small normal-only frozen contact problem for diagnostics.

    This diagnostic rejects invalid matrices and material arrays rather than
    silently reverting to a hard solve. No restitution, warm start, friction,
    bilateral rows, or proximal regularization is included in this prototype.
    """
    matrix = np.asarray(delassus, dtype=np.float32)
    free = np.asarray(velocity_free, dtype=np.float32)
    phi, k, c = (np.asarray(x, dtype=np.float32) for x in (separation, stiffness, damping))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or free.shape != (len(matrix),):
        raise ValueError("Require square Delassus and a matching velocity vector")
    if any(x.shape != free.shape for x in (phi, k, c)):
        raise ValueError("Material and separation arrays must match contact count")
    if not np.isfinite(matrix).all() or not np.isfinite(free).all():
        raise ValueError("Require finite matrix and free velocity")
    if not np.allclose(matrix, matrix.T) or np.linalg.eigvalsh(matrix).min() < -1e-6:
        raise ValueError("Delassus must be symmetric positive semidefinite")
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be positive")
    coefficients = [normal_coefficients(float(a), float(b), float(p), dt=dt) for a, b, p in zip(k, c, phi, strict=True)]
    gamma, bias = np.asarray(coefficients, dtype=np.float32).T
    with wp.ScopedDevice(device):
        impulses = wp.zeros((1, len(free)), dtype=float)
        wp.launch(
            solve_normal_problem,
            1,
            inputs=[
                wp.array(matrix[None]),
                wp.array(free[None]),
                wp.array(bias[None]),
                wp.array(gamma[None]),
                iterations,
                impulses,
            ],
        )
        result = impulses.numpy()[0]
    residual = free + bias + matrix @ result + gamma * result
    return result, residual
