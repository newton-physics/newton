# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# SPH (Smoothed Particle Hydrodynamics) kernels for Newton.
#
# Based on the Müller et al. 2003 SPH formulation:
#   Matthias Müller, David Charypar, and Markus H. Gross.
#   "Particle-based fluid simulation for interactive applications."
#   Symposium on Computer Animation. Vol. 2. 2003.
#
# Kernel functions use the Poly6 (density), Spiky (pressure gradient),
# and viscosity kernels from that paper.

import warp as wp

from ...geometry import ParticleFlags


# ---------------------------------------------------------------------------
# SPH kernel functions (device-callable)
# ---------------------------------------------------------------------------


@wp.func
def poly6_kernel(r_sq: float, h: float):
    """Poly6 kernel for density estimation. Takes squared distance."""
    diff = h * h - r_sq
    if diff > 0.0:
        return diff * diff * diff
    return 0.0


@wp.func
def spiky_gradient(r: wp.vec3, r_len: float, h: float):
    """Spiky kernel gradient for pressure force."""
    if r_len < h and r_len > 1e-7:
        diff = h - r_len
        return -r * (diff * diff) / r_len
    return wp.vec3(0.0)


@wp.func
def viscosity_laplacian(r_len: float, h: float):
    """Viscosity kernel Laplacian for viscous force."""
    if r_len < h:
        return h - r_len
    return 0.0


# ---------------------------------------------------------------------------
# Compute kernels
# ---------------------------------------------------------------------------


@wp.kernel
def compute_density(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    density_normalization: float,
    smoothing_length: float,
    # outputs
    particle_rho: wp.array[float],
):
    """Compute SPH density for each particle using Poly6 kernel."""
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return

    x = particle_x[i]
    world_idx = particle_world[i]

    rho = float(0.0)

    query = wp.hash_grid_query(grid, x, smoothing_length)
    index = int(0)
    while wp.hash_grid_query_next(query, index):
        if (particle_flags[index] & ParticleFlags.ACTIVE) != 0:
            if particle_world[index] == world_idx:
                r = x - particle_x[index]
                r_sq = wp.dot(r, r)
                rho += particle_mass[index] * poly6_kernel(r_sq, smoothing_length)

    particle_rho[i] = density_normalization * rho


@wp.kernel
def compute_sph_forces(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    particle_rho: wp.array[float],
    pressure_stiffness: float,
    rest_density: float,
    pressure_normalization: float,
    viscous_normalization: float,
    smoothing_length: float,
    # outputs
    particle_f: wp.array[wp.vec3],
):
    """Compute SPH pressure and viscosity forces, accumulate into particle_f."""
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return

    x = particle_x[i]
    v = particle_v[i]
    rho_i = particle_rho[i]
    world_idx = particle_world[i]

    # Tait equation of state for pressure
    pressure_i = pressure_stiffness * (rho_i - rest_density)

    pressure_force = wp.vec3(0.0)
    viscous_force = wp.vec3(0.0)

    query = wp.hash_grid_query(grid, x, smoothing_length)
    index = int(0)
    while wp.hash_grid_query_next(query, index):
        if index == i:
            continue
        if (particle_flags[index] & ParticleFlags.ACTIVE) == 0:
            continue
        if particle_world[index] != world_idx:
            continue

        r = particle_x[index] - x
        r_len = wp.length(r)
        rho_j = particle_rho[index]
        pressure_j = pressure_stiffness * (rho_j - rest_density)

        if rho_j < 1e-10:
            continue

        # Pressure force (spiky kernel gradient, symmetric formulation)
        pressure_force += (
            particle_mass[index]
            * (pressure_i + pressure_j)
            / (2.0 * rho_j)
            * spiky_gradient(r, r_len, smoothing_length)
        )

        # Viscosity force
        viscous_force += (
            particle_mass[index]
            * (particle_v[index] - v)
            / rho_j
            * viscosity_laplacian(r_len, smoothing_length)
        )

    # Total SPH force (excluding gravity — gravity is applied by integrate_particles)
    particle_f[i] = (
        pressure_normalization * pressure_force + viscous_normalization * viscous_force
    )
