# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for explicit SPH force accumulation."""

import warp as wp

from ...geometry import ParticleFlags
from .kernels import (
    SPH_EPSILON,
    SPH_ROLE_BOUNDARY,
    SPH_ROLE_FLUID,
    _is_sph_neighbor_role,
    _kernel_gradient,
    _kernel_weight,
    _same_world,
    _support_radius,
    _viscosity_laplacian,
)

wp.set_module_options({"enable_backward": False})


@wp.kernel
def compute_acceleration(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_inv_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    sph_role: wp.array[wp.int32],
    sph_viscosity: wp.array[float],
    sph_surface_tension: wp.array[float],
    sph_adhesion: wp.array[float],
    sph_wetting: wp.array[float],
    sph_contact_angle: wp.array[float],
    sph_smoothing_length: wp.array[float],
    density: wp.array[float],
    pressure: wp.array[float],
    volume: wp.array[float],
    normal: wp.array[wp.vec3],
    boundary_normal: wp.array[wp.vec3],
    default_viscosity: float,
    default_smoothing_length: float,
    kernel_id: int,
    enable_surface_tension: bool,
    surface_tension_normal_threshold: float,
    enable_boundary_adhesion: bool,
    enable_boundary_wetting: bool,
    acceleration: wp.array[wp.vec3],
    surface_acceleration: wp.array[wp.vec3],
    adhesion_acceleration: wp.array[wp.vec3],
    wetting_acceleration: wp.array[wp.vec3],
):
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        i = tid

    inv_mass_i = particle_inv_mass[i]
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0 or inv_mass_i <= 0.0 or sph_role[i] != SPH_ROLE_FLUID:
        acceleration[i] = wp.vec3(0.0)
        surface_acceleration[i] = wp.vec3(0.0)
        adhesion_acceleration[i] = wp.vec3(0.0)
        wetting_acceleration[i] = wp.vec3(0.0)
        return

    x_i = particle_q[i]
    v_i = particle_qd[i]
    world_i = particle_world[i]
    rho_i = wp.max(density[i], SPH_EPSILON)
    p_i = pressure[i]
    h_i = _support_radius(sph_smoothing_length[i], default_smoothing_length)

    viscosity_i = sph_viscosity[i]
    if viscosity_i < 0.0:
        viscosity_i = default_viscosity

    accel = particle_f[i] * inv_mass_i + gravity[wp.max(world_i, 0)]
    surface_accel = wp.vec3(0.0)
    adhesion_accel = wp.vec3(0.0)
    wetting_accel = wp.vec3(0.0)
    n_i = normal[i]
    n_len_i = wp.length(n_i)
    n_hat_i = wp.vec3(0.0)
    if n_len_i > SPH_EPSILON:
        n_hat_i = n_i / n_len_i
    sigma_i = sph_surface_tension[i]
    adhesion_i = sph_adhesion[i]
    wetting_i = sph_wetting[i]
    contact_angle_i = sph_contact_angle[i]
    curvature = float(0.0)

    query = wp.hash_grid_query(grid, x_i, default_smoothing_length)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if (
            j != i
            and (particle_flags[j] & ParticleFlags.ACTIVE) != 0
            and _is_sph_neighbor_role(sph_role[j])
            and _same_world(world_i, particle_world[j])
        ):
            h_j = _support_radius(sph_smoothing_length[j], default_smoothing_length)
            h = wp.max(h_i, h_j)
            r_vec = x_i - particle_q[j]
            r = wp.length(r_vec)

            if r < h and r > SPH_EPSILON:
                rho_j = wp.max(density[j], SPH_EPSILON)
                mass_j = particle_mass[j]
                grad_w = _kernel_gradient(kernel_id, r_vec, r, h)
                pressure_term = p_i / (rho_i * rho_i)
                if sph_role[j] != SPH_ROLE_BOUNDARY:
                    pressure_term += pressure[j] / (rho_j * rho_j)
                accel -= mass_j * pressure_term * grad_w

                viscosity_j = sph_viscosity[j]
                if viscosity_j < 0.0:
                    viscosity_j = default_viscosity
                viscosity = 0.5 * (viscosity_i + viscosity_j)
                accel += viscosity * mass_j * (particle_qd[j] - v_i) * _viscosity_laplacian(r, h) / rho_j

                if enable_boundary_adhesion and sph_role[j] == SPH_ROLE_BOUNDARY and adhesion_i > 0.0:
                    adhesion_j = sph_adhesion[j]
                    pair_adhesion = 0.5 * (adhesion_i + adhesion_j)
                    if adhesion_j <= 0.0:
                        pair_adhesion = adhesion_i
                    adhesion = -pair_adhesion * mass_j * _kernel_weight(kernel_id, r, h) * (r_vec / r) / rho_j
                    adhesion_accel += adhesion
                    accel += adhesion

                if enable_boundary_wetting and sph_role[j] == SPH_ROLE_BOUNDARY and wetting_i > 0.0:
                    wall_normal = boundary_normal[j]
                    wall_normal_len = wp.length(wall_normal)
                    if wall_normal_len > SPH_EPSILON:
                        wetting_j = sph_wetting[j]
                        pair_wetting = 0.5 * (wetting_i + wetting_j)
                        if wetting_j <= 0.0:
                            pair_wetting = wetting_i
                        wetting = (
                            -pair_wetting
                            * wp.cos(contact_angle_i)
                            * mass_j
                            * _kernel_weight(kernel_id, r, h)
                            * (wall_normal / wall_normal_len)
                            / rho_j
                        )
                        wetting_accel += wetting
                        accel += wetting

                if enable_surface_tension and sigma_i > 0.0 and n_len_i > surface_tension_normal_threshold:
                    n_j = normal[j]
                    n_len_j = wp.length(n_j)
                    n_hat_j = wp.vec3(0.0)
                    if n_len_j > SPH_EPSILON:
                        n_hat_j = n_j / n_len_j
                    curvature += volume[j] * wp.dot(n_hat_j - n_hat_i, grad_w)

    if enable_surface_tension and sigma_i > 0.0 and n_len_i > surface_tension_normal_threshold:
        surface_accel = -sigma_i * curvature * n_hat_i / rho_i
        accel += surface_accel

    acceleration[i] = accel
    surface_acceleration[i] = surface_accel
    adhesion_acceleration[i] = adhesion_accel
    wetting_acceleration[i] = wetting_accel


@wp.kernel
def compute_xsph_velocity_delta(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    particle_inv_mass: wp.array[float],
    sph_role: wp.array[wp.int32],
    sph_smoothing_length: wp.array[float],
    density: wp.array[float],
    default_smoothing_length: float,
    kernel_id: int,
    velocity_delta: wp.array[wp.vec3],
):
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        i = tid

    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0 or particle_inv_mass[i] <= 0.0 or sph_role[i] != SPH_ROLE_FLUID:
        velocity_delta[i] = wp.vec3(0.0)
        return

    x_i = particle_q[i]
    v_i = particle_qd[i]
    world_i = particle_world[i]
    h_i = _support_radius(sph_smoothing_length[i], default_smoothing_length)

    delta = wp.vec3(0.0)
    query = wp.hash_grid_query(grid, x_i, default_smoothing_length)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if (
            j != i
            and (particle_flags[j] & ParticleFlags.ACTIVE) != 0
            and sph_role[j] == SPH_ROLE_FLUID
            and _same_world(world_i, particle_world[j])
        ):
            h_j = _support_radius(sph_smoothing_length[j], default_smoothing_length)
            h = wp.max(h_i, h_j)
            r = wp.length(x_i - particle_q[j])

            if r < h:
                rho_j = wp.max(density[j], SPH_EPSILON)
                delta += particle_mass[j] * (particle_qd[j] - v_i) * _kernel_weight(kernel_id, r, h) / rho_j

    velocity_delta[i] = delta


@wp.kernel
def compute_surface_fields(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    particle_inv_mass: wp.array[float],
    sph_role: wp.array[wp.int32],
    sph_smoothing_length: wp.array[float],
    volume: wp.array[float],
    default_smoothing_length: float,
    kernel_id: int,
    color_field: wp.array[float],
    normal: wp.array[wp.vec3],
):
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        i = tid

    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0 or particle_inv_mass[i] <= 0.0 or sph_role[i] != SPH_ROLE_FLUID:
        color_field[i] = 0.0
        normal[i] = wp.vec3(0.0)
        return

    x_i = particle_q[i]
    world_i = particle_world[i]
    h_i = _support_radius(sph_smoothing_length[i], default_smoothing_length)

    c = float(0.0)
    n = wp.vec3(0.0)
    query = wp.hash_grid_query(grid, x_i, default_smoothing_length)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if (
            (particle_flags[j] & ParticleFlags.ACTIVE) != 0
            and _is_sph_neighbor_role(sph_role[j])
            and _same_world(world_i, particle_world[j])
        ):
            h_j = _support_radius(sph_smoothing_length[j], default_smoothing_length)
            h = wp.max(h_i, h_j)
            r_vec = x_i - particle_q[j]
            r = wp.length(r_vec)

            if r < h:
                vol_j = volume[j]
                c += vol_j * _kernel_weight(kernel_id, r, h)
                if j != i and r > SPH_EPSILON:
                    n += vol_j * _kernel_gradient(kernel_id, r_vec, r, h)

    color_field[i] = c
    normal[i] = n
