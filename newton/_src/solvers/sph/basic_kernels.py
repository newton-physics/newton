# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for the default WCSPH runtime path."""

import warp as wp

from ...geometry import ParticleFlags
from .kernels import (
    SPH_EPSILON,
    SPH_ROLE_BOUNDARY,
    SPH_ROLE_FLUID,
    _is_sph_integrated_role,
    _is_sph_neighbor_role,
    _kernel_gradient,
    _kernel_weight,
    _pressure_from_density,
    _same_world,
    _support_radius,
    _viscosity_laplacian,
)

wp.set_module_options({"enable_backward": False})


@wp.kernel
def compute_acceleration_basic(
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
    sph_smoothing_length: wp.array[float],
    density: wp.array[float],
    pressure: wp.array[float],
    default_viscosity: float,
    default_smoothing_length: float,
    kernel_id: int,
    enable_xsph: bool,
    acceleration: wp.array[wp.vec3],
    surface_acceleration: wp.array[wp.vec3],
    adhesion_acceleration: wp.array[wp.vec3],
    wetting_acceleration: wp.array[wp.vec3],
    velocity_delta: wp.array[wp.vec3],
):
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        i = tid

    surface_acceleration[i] = wp.vec3(0.0)
    adhesion_acceleration[i] = wp.vec3(0.0)
    wetting_acceleration[i] = wp.vec3(0.0)

    inv_mass_i = particle_inv_mass[i]
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0 or inv_mass_i <= 0.0 or sph_role[i] != SPH_ROLE_FLUID:
        acceleration[i] = wp.vec3(0.0)
        velocity_delta[i] = wp.vec3(0.0)
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
    xsph_delta = wp.vec3(0.0)
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

            if r < h:
                rho_j = wp.max(density[j], SPH_EPSILON)
                if enable_xsph and sph_role[j] == SPH_ROLE_FLUID:
                    mass_j = particle_mass[j]
                    xsph_delta += mass_j * (particle_qd[j] - v_i) * _kernel_weight(kernel_id, r, h) / rho_j
                if r > SPH_EPSILON:
                    mass_j = particle_mass[j]
                    grad_w = _kernel_gradient(kernel_id, r_vec, r, h)
                    pressure_term = p_i / (rho_i * rho_i)
                    if sph_role[j] != SPH_ROLE_BOUNDARY:
                        pressure_term += pressure[j] / (rho_j * rho_j)
                    accel -= mass_j * pressure_term * grad_w

                    viscosity_j = sph_viscosity[j]
                    if viscosity_j < 0.0:
                        viscosity_j = default_viscosity
                    pair_viscosity = 0.5 * (viscosity_i + viscosity_j)
                    lap_w = _viscosity_laplacian(r, h)
                    accel += pair_viscosity * mass_j * (particle_qd[j] - v_i) * lap_w / rho_j

    acceleration[i] = accel
    velocity_delta[i] = xsph_delta


@wp.kernel
def compute_density_pressure(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    sph_role: wp.array[wp.int32],
    sph_rest_density: wp.array[float],
    sph_sound_speed: wp.array[float],
    sph_stiffness: wp.array[float],
    sph_pressure_exponent: wp.array[float],
    sph_pressure_min: wp.array[float],
    sph_pressure_max: wp.array[float],
    sph_smoothing_length: wp.array[float],
    default_rest_density: float,
    default_sound_speed: float,
    default_stiffness: float,
    default_pressure_exponent: float,
    default_smoothing_length: float,
    kernel_id: int,
    density: wp.array[float],
    pressure: wp.array[float],
    volume: wp.array[float],
):
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        i = tid

    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0 or not _is_sph_neighbor_role(sph_role[i]):
        density[i] = 0.0
        pressure[i] = 0.0
        volume[i] = 0.0
        return

    x_i = particle_q[i]
    world_i = particle_world[i]
    h_i = _support_radius(sph_smoothing_length[i], default_smoothing_length)

    rho = float(0.0)
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
            r = wp.length(x_i - particle_q[j])
            if r < h:
                rho += particle_mass[j] * _kernel_weight(kernel_id, r, h)

    rest_density = sph_rest_density[i]
    if rest_density <= 0.0:
        rest_density = default_rest_density

    sound_speed = sph_sound_speed[i]
    if sound_speed <= 0.0:
        sound_speed = default_sound_speed

    stiffness = sph_stiffness[i]
    if stiffness <= 0.0:
        stiffness = default_stiffness

    pressure_exponent = sph_pressure_exponent[i]
    if pressure_exponent <= 0.0:
        pressure_exponent = default_pressure_exponent

    density[i] = rho
    pressure[i] = _pressure_from_density(
        rho, rest_density, sound_speed, stiffness, pressure_exponent, sph_pressure_min[i], sph_pressure_max[i]
    )
    volume[i] = particle_mass[i] / wp.max(rho, SPH_EPSILON)


@wp.kernel
def integrate_sph_particles(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_inv_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    sph_role: wp.array[wp.int32],
    acceleration: wp.array[wp.vec3],
    velocity_delta: wp.array[wp.vec3],
    xsph: float,
    dt: float,
    max_velocity: float,
    particle_q_out: wp.array[wp.vec3],
    particle_qd_out: wp.array[wp.vec3],
):
    i = wp.tid()

    x = particle_q[i]
    v = particle_qd[i]

    if (
        (particle_flags[i] & ParticleFlags.ACTIVE) == 0
        or particle_inv_mass[i] <= 0.0
        or not _is_sph_integrated_role(sph_role[i])
    ):
        particle_q_out[i] = x
        particle_qd_out[i] = v
        return

    v_new = v + acceleration[i] * dt + xsph * velocity_delta[i]
    speed = wp.length(v_new)
    if speed > max_velocity:
        v_new = v_new * (max_velocity / speed)

    particle_qd_out[i] = v_new
    particle_q_out[i] = x + v_new * dt
