# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# SPH (Smoothed Particle Hydrodynamics) kernels for Newton.
#
# Supports two formulations:
#   1. Müller et al. 2003 — standard SPH (density + pressure/viscosity forces).
#      Matthias Müller, David Charypar, and Markus Gross.
#      "Particle-based fluid simulation for interactive applications."
#      Symposium on Computer Animation. 2003.
#
#   2. Macklin & Müller 2013 — Position-Based Fluids (PBF).
#      Miles Macklin and Matthias Müller.
#      "Position Based Fluids."
#      ACM SIGGRAPH / Eurographics Symposium on Computer Animation. 2013.
#
# Kernel functions: Poly6 (density), Spiky (pressure gradient / PBF gradient),
# and viscosity Laplacian.

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
    """Spiky kernel gradient for pressure force / PBF constraint gradient."""
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
# Standard SPH kernels (Müller 2003)
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


# ---------------------------------------------------------------------------
# Position-Based Fluids kernels (Macklin & Müller 2013)
# ---------------------------------------------------------------------------


@wp.kernel
def pbf_compute_lambda(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    particle_rho: wp.array[float],
    rest_density: float,
    smoothing_length: float,
    spiky_normalization: float,
    epsilon: float,
    # outputs
    particle_lambda: wp.array[float],
):
    """Compute PBF constraint multiplier lambda_i for each particle.

    For particle i the density constraint is C_i = rho_i / rho_0 - 1.
    The multiplier is:
        lambda_i = -C_i / (sum_k |nabla_k C_i|^2 + epsilon)
    where nabla_i C_i uses the Spiky kernel gradient and nabla_j C_i (j != i)
    uses the same kernel with opposite sign.
    """
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return

    x_i = particle_x[i]
    rho_i = particle_rho[i]
    world_idx = particle_world[i]

    # Constraint value
    C_i = rho_i / rest_density - 1.0

    # Accumulate |nabla_k C_i|^2 over all neighbors k (including self)
    grad_sq_sum = float(0.0)

    # nabla_i C_i (gradient of constraint w.r.t. own position)
    grad_i_C = wp.vec3(0.0)

    query = wp.hash_grid_query(grid, x_i, smoothing_length)
    index = int(0)
    while wp.hash_grid_query_next(query, index):
        if (particle_flags[index] & ParticleFlags.ACTIVE) == 0:
            continue
        if particle_world[index] != world_idx:
            continue

        r = particle_x[index] - x_i
        r_len = wp.length(r)

        # Spiky gradient: nabla W_spiky normalized
        # The factor 1/rho_0 is part of the constraint gradient
        grad_W = spiky_gradient(r, r_len, smoothing_length) * spiky_normalization

        if index == i:
            # Self-contribution: nabla_i C_i = (1/rho_0) * sum_j m_j * grad_W(r_ij, h)
            # (but when j == i, grad_W is zero because r = 0, so nothing to add)
            pass
        else:
            # Contribution of neighbor j to nabla_i C_i:
            #   nabla_i C_i += (m_j / rho_0) * grad_W(r_j - r_i, h)
            # grad_W already points from i toward j (since r = x_j - x_i),
            # and spiky_gradient returns -r * (h-r)^2 / r, so it points from j to i.
            # We negate to get the correct direction for nabla_i C_i.
            contrib = (particle_mass[index] / rest_density) * grad_W
            grad_i_C += contrib

    # Self-gradient magnitude: nabla_i C_i accumulated above
    grad_sq_sum += wp.dot(grad_i_C, grad_i_C)

    # For each neighbor j, nabla_j C_i = -(m_i / rho_0) * grad_W(r_ij, h)
    # |nabla_j C_i|^2 accumulated
    query = wp.hash_grid_query(grid, x_i, smoothing_length)
    index = int(0)
    while wp.hash_grid_query_next(query, index):
        if index == i:
            continue
        if (particle_flags[index] & ParticleFlags.ACTIVE) == 0:
            continue
        if particle_world[index] != world_idx:
            continue

        r = particle_x[index] - x_i
        r_len = wp.length(r)
        grad_W = spiky_gradient(r, r_len, smoothing_length) * spiky_normalization
        grad_j = (particle_mass[i] / rest_density) * grad_W
        grad_sq_sum += wp.dot(grad_j, grad_j)

    lambda_i = -C_i / (grad_sq_sum + epsilon)
    particle_lambda[i] = lambda_i


@wp.kernel
def pbf_compute_position_correction(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    particle_rho: wp.array[float],
    particle_lambda: wp.array[float],
    rest_density: float,
    smoothing_length: float,
    spiky_normalization: float,
    surface_tension: float,
    density_normalization: float,
    dt: float,
    # outputs
    particle_dx: wp.array[wp.vec3],
):
    """Compute position correction delta_p_i for PBF solver.

    The correction has two components:
      1. Constraint projection:
         delta_p_i = (1/rho_0) * sum_j (lambda_i + lambda_j) * spiky_grad(r_ij, h)
      2. XSPH surface tension:
         v_xsph = s * sum_j (m_j / rho_j) * (v_j - v_i) * W_poly6(r_ij, h)
         The XSPH velocity is converted to a position increment: dx_xsph = v_xsph * dt
    """
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return

    x_i = particle_x[i]
    v_i = particle_v[i]
    lambda_i = particle_lambda[i]
    world_idx = particle_world[i]

    delta_p = wp.vec3(0.0)
    xsph_vel = wp.vec3(0.0)

    query = wp.hash_grid_query(grid, x_i, smoothing_length)
    index = int(0)
    while wp.hash_grid_query_next(query, index):
        if index == i:
            continue
        if (particle_flags[index] & ParticleFlags.ACTIVE) == 0:
            continue
        if particle_world[index] != world_idx:
            continue

        r = particle_x[index] - x_i
        r_len = wp.length(r)
        lambda_j = particle_lambda[index]

        # Constraint correction (spiky gradient)
        # spiky_gradient(r) with r=x_j-x_i returns -r_hat*(h-r)^2 (toward self).
        # PBF paper uses nabla W(p_i-p_j) which is +r_hat*(h-r)^2 (toward neighbor).
        # So we negate to match the paper's correction direction.
        s_grad = -spiky_gradient(r, r_len, smoothing_length) * spiky_normalization
        delta_p += (lambda_i + lambda_j) * s_grad

        # XSPH velocity correction (poly6 kernel)
        rho_j = particle_rho[index]
        if rho_j > 1e-10:
            r_sq = wp.dot(r, r)
            w_poly6 = poly6_kernel(r_sq, smoothing_length) * density_normalization
            xsph_vel += (particle_mass[index] / rho_j) * (particle_v[index] - v_i) * w_poly6

    # Scale constraint correction by 1/rho_0
    delta_p *= (1.0 / rest_density)

    # Apply XSPH as a position increment (velocity * dt)
    delta_p += surface_tension * xsph_vel * dt

    particle_dx[i] = delta_p


@wp.kernel
def pbf_apply_position_correction(
    particle_x: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_dx: wp.array[wp.vec3],
    # outputs
    particle_x_new: wp.array[wp.vec3],
):
    """Apply position correction: p_i += delta_p_i."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        particle_x_new[tid] = particle_x[tid]
        return

    particle_x_new[tid] = particle_x[tid] + particle_dx[tid]


@wp.kernel
def pbf_update_velocity(
    particle_x_new: wp.array[wp.vec3],
    particle_x_old: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    dt: float,
    # outputs
    particle_v: wp.array[wp.vec3],
):
    """Update velocity from position change: v_i = (p_new - p_old) / dt."""
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return

    x_new = particle_x_new[tid]
    x_old = particle_x_old[tid]
    particle_v[tid] = (x_new - x_old) / dt
