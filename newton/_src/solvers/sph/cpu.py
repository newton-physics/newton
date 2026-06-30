# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU fallbacks for the WCSPH solver."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...geometry import ParticleFlags
from ...sim import State
from .kernels import sph_is_neighbor_role_np as _is_sph_neighbor_role_np
from .kernels import sph_kernel_gradient_np as _kernel_gradient_np
from .kernels import sph_kernel_weight_np as _kernel_weight_np
from .kernels import sph_pressure_from_density_np as _pressure_from_density_np
from .kernels import sph_same_world_np as _same_world_np
from .kernels import sph_viscosity_laplacian_np as _viscosity_laplacian_np
from .sph_model import SPHRole
from .utils import sph_wp_vec3_array as _vec3_array


def _active_neighbor_role_mask(solver: Any) -> np.ndarray:
    flags = solver.model.particle_flags.numpy()
    roles = solver.model.sph.role.numpy()
    return ((flags & int(ParticleFlags.ACTIVE)) != 0) & np.asarray(
        [_is_sph_neighbor_role_np(int(role)) for role in roles],
        dtype=bool,
    )


def _compute_density_pressure_cpu(solver: Any, state: State) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = solver.model
    particle_count = int(model.particle_count)
    q = _vec3_array(state.particle_q)
    mass = np.asarray(model.particle_mass.numpy(), dtype=np.float32)
    worlds = np.asarray(model.particle_world.numpy(), dtype=np.int32)
    rest_density = np.asarray(model.sph.rest_density.numpy(), dtype=np.float32)
    sound_speed = np.asarray(model.sph.sound_speed.numpy(), dtype=np.float32)
    stiffness = np.asarray(model.sph.stiffness.numpy(), dtype=np.float32)
    pressure_exponent = np.asarray(model.sph.pressure_exponent.numpy(), dtype=np.float32)
    pressure_min = np.asarray(model.sph.pressure_min.numpy(), dtype=np.float32)
    pressure_max = np.asarray(model.sph.pressure_max.numpy(), dtype=np.float32)
    support = solver._support_radius_np()
    density = np.zeros(particle_count, dtype=np.float32)
    pressure = np.zeros(particle_count, dtype=np.float32)
    volume = np.zeros(particle_count, dtype=np.float32)
    active_neighbor = _active_neighbor_role_mask(solver)
    active_indices = np.flatnonzero(active_neighbor).astype(np.int32)
    for i in active_indices:
        h_i = max(float(support[i]), 1.0e-6)
        rho = 0.0
        for j in active_indices:
            if not _same_world_np(int(worlds[i]), int(worlds[j])):
                continue
            h = max(h_i, float(support[j]), 1.0e-6)
            r = float(np.linalg.norm(q[i] - q[j]))
            if r < h:
                rho += float(mass[j]) * _kernel_weight_np(solver._sph_model.kernel_id, r, h)

        density[i] = rho
        volume[i] = float(mass[i]) / max(rho, 1.0e-6)

    rest = np.where(rest_density > 0.0, rest_density, float(solver.config.rest_density)).astype(np.float32)
    sound = np.where(sound_speed > 0.0, sound_speed, float(solver.config.sound_speed)).astype(np.float32)
    stiff = np.where(stiffness > 0.0, stiffness, float(solver.config.stiffness)).astype(np.float32)
    exponent = np.where(pressure_exponent > 0.0, pressure_exponent, float(solver.config.pressure_exponent)).astype(
        np.float32
    )
    pressure[active_indices] = _pressure_from_density_np(
        density[active_indices],
        rest[active_indices],
        sound[active_indices],
        stiff[active_indices],
        exponent[active_indices],
        pressure_min[active_indices],
        pressure_max[active_indices],
    )
    return density, pressure, volume


def _integrate_particles_cpu(
    solver: Any, state: State, dt: float, max_velocity: float
) -> tuple[np.ndarray, np.ndarray]:
    q = _vec3_array(state.particle_q)
    qd = _vec3_array(state.particle_qd)
    acceleration = _vec3_array(state.sph.acceleration)
    velocity_delta = _vec3_array(state.sph.velocity_delta)
    q_out = q.copy()
    qd_out = qd.copy()
    active = solver._active_role_mask(SPHRole.FLUID, dynamic=True)
    for i in np.flatnonzero(active).astype(np.int32):
        v_new = qd[i] + acceleration[i] * float(dt) + float(solver.config.xsph) * velocity_delta[i]
        speed = float(np.linalg.norm(v_new))
        if speed > max_velocity:
            v_new = v_new * (max_velocity / speed)
        qd_out[i] = v_new
        q_out[i] = q[i] + v_new * float(dt)
    return q_out, qd_out


def _compute_xsph_velocity_delta_cpu(solver: Any, state: State) -> np.ndarray:
    model = solver.model
    particle_count = int(model.particle_count)
    q = _vec3_array(state.particle_q)
    qd = _vec3_array(state.particle_qd)
    mass = np.asarray(model.particle_mass.numpy(), dtype=np.float32)
    worlds = np.asarray(model.particle_world.numpy(), dtype=np.int32)
    support = solver._support_radius_np()
    density = np.asarray(state.sph.density.numpy(), dtype=np.float32)
    active = solver._active_role_mask(SPHRole.FLUID, dynamic=True)
    fluid_neighbor = solver._active_role_mask(SPHRole.FLUID)
    active_indices = np.flatnonzero(active).astype(np.int32)
    neighbor_indices = np.flatnonzero(fluid_neighbor).astype(np.int32)
    velocity_delta = np.zeros((particle_count, 3), dtype=np.float32)

    for i in active_indices:
        h_i = max(float(support[i]), 1.0e-6)
        delta = np.zeros(3, dtype=np.float32)
        for j in neighbor_indices:
            if i == j or not _same_world_np(int(worlds[i]), int(worlds[j])):
                continue
            h = max(h_i, float(support[j]), 1.0e-6)
            rho_j = max(float(density[j]), 1.0e-6)
            r = float(np.linalg.norm(q[i] - q[j]))
            if r >= h:
                continue
            delta += (
                float(mass[j]) * (qd[j] - qd[i]) * _kernel_weight_np(solver._sph_model.kernel_id, r, h) / rho_j
            ).astype(np.float32)
        velocity_delta[i] = delta
    return velocity_delta


def _compute_acceleration_cpu(
    solver: Any,
    state: State,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = solver.model
    particle_count = int(model.particle_count)
    q = _vec3_array(state.particle_q)
    qd = _vec3_array(state.particle_qd)
    particle_f = _vec3_array(state.particle_f)
    mass = np.asarray(model.particle_mass.numpy(), dtype=np.float32)
    inv_mass = np.asarray(model.particle_inv_mass.numpy(), dtype=np.float32)
    worlds = np.asarray(model.particle_world.numpy(), dtype=np.int32)
    gravity = _vec3_array(model.gravity)
    roles = np.asarray(model.sph.role.numpy(), dtype=np.int32)
    viscosity = np.asarray(model.sph.viscosity.numpy(), dtype=np.float32)
    surface_tension = np.asarray(model.sph.surface_tension.numpy(), dtype=np.float32)
    adhesion = np.asarray(model.sph.adhesion.numpy(), dtype=np.float32)
    wetting = np.asarray(model.sph.wetting.numpy(), dtype=np.float32)
    contact_angle = np.asarray(model.sph.contact_angle.numpy(), dtype=np.float32)
    support = solver._support_radius_np()
    density = np.asarray(state.sph.density.numpy(), dtype=np.float32)
    pressure = np.asarray(state.sph.pressure.numpy(), dtype=np.float32)
    volume = np.asarray(state.sph.volume.numpy(), dtype=np.float32)
    normal = _vec3_array(state.sph.normal)
    boundary_normal = _vec3_array(state.sph.boundary_normal)
    acceleration = np.zeros((particle_count, 3), dtype=np.float32)
    surface_acceleration = np.zeros((particle_count, 3), dtype=np.float32)
    adhesion_acceleration = np.zeros((particle_count, 3), dtype=np.float32)
    wetting_acceleration = np.zeros((particle_count, 3), dtype=np.float32)
    active = solver._active_role_mask(SPHRole.FLUID, dynamic=True)
    neighbor = _active_neighbor_role_mask(solver)
    neighbor_indices = np.flatnonzero(neighbor).astype(np.int32)

    def viscosity_at(index: int) -> float:
        value = float(viscosity[index])
        if value < 0.0:
            value = float(solver.config.viscosity)
        return value

    enable_surface_tension = solver.config.enable_surface_tension
    surface_tension_normal_threshold = solver.config.surface_tension_normal_threshold
    enable_boundary_adhesion = solver.config.enable_boundary_adhesion
    enable_boundary_wetting = solver.config.enable_boundary_wetting

    for i in np.flatnonzero(active).astype(np.int32):
        world_i = int(worlds[i])
        gravity_i = gravity[max(world_i, 0)] if gravity.shape[0] else np.zeros(3, dtype=np.float32)
        accel = particle_f[i] * float(inv_mass[i]) + gravity_i
        rho_i = max(float(density[i]), 1.0e-6)
        p_i = float(pressure[i])
        h_i = max(float(support[i]), 1.0e-6)
        viscosity_i = viscosity_at(int(i))
        n_i = normal[i]
        n_len_i = float(np.linalg.norm(n_i))
        n_hat_i = n_i / n_len_i if n_len_i > 1.0e-6 else np.zeros(3, dtype=np.float32)
        sigma_i = float(surface_tension[i])
        adhesion_i = float(adhesion[i])
        wetting_i = float(wetting[i])
        curvature = 0.0
        surface_accel = np.zeros(3, dtype=np.float32)
        adhesion_accel = np.zeros(3, dtype=np.float32)
        wetting_accel = np.zeros(3, dtype=np.float32)

        for j in neighbor_indices:
            if i == j or not _same_world_np(world_i, int(worlds[j])):
                continue
            h = max(h_i, float(support[j]), 1.0e-6)
            r_vec = q[i] - q[j]
            r = float(np.linalg.norm(r_vec))
            if r >= h or r <= 1.0e-6:
                continue
            rho_j = max(float(density[j]), 1.0e-6)
            mass_j = float(mass[j])
            grad_w = _kernel_gradient_np(solver._sph_model.kernel_id, r_vec, r, h)
            pressure_term = p_i / (rho_i * rho_i)
            if roles[j] != int(SPHRole.BOUNDARY):
                pressure_term += float(pressure[j]) / (rho_j * rho_j)
            accel -= mass_j * pressure_term * grad_w

            viscosity_j = viscosity_at(int(j))
            pair_viscosity = 0.5 * (viscosity_i + viscosity_j)
            accel += pair_viscosity * mass_j * (qd[j] - qd[i]) * _viscosity_laplacian_np(r, h) / rho_j

            if enable_boundary_adhesion and roles[j] == int(SPHRole.BOUNDARY) and adhesion_i > 0.0:
                adhesion_j = float(adhesion[j])
                pair_adhesion = 0.5 * (adhesion_i + adhesion_j) if adhesion_j > 0.0 else adhesion_i
                adhesion_term = (
                    -pair_adhesion * mass_j * _kernel_weight_np(solver._sph_model.kernel_id, r, h) * (r_vec / r) / rho_j
                )
                adhesion_accel += adhesion_term
                accel += adhesion_term

            if enable_boundary_wetting and roles[j] == int(SPHRole.BOUNDARY) and wetting_i > 0.0:
                wall_normal = boundary_normal[j]
                wall_normal_len = float(np.linalg.norm(wall_normal))
                if wall_normal_len > 1.0e-6:
                    wetting_j = float(wetting[j])
                    pair_wetting = 0.5 * (wetting_i + wetting_j) if wetting_j > 0.0 else wetting_i
                    wetting_term = (
                        -pair_wetting
                        * float(np.cos(float(contact_angle[i])))
                        * mass_j
                        * _kernel_weight_np(solver._sph_model.kernel_id, r, h)
                        * (wall_normal / wall_normal_len)
                        / rho_j
                    )
                    wetting_accel += wetting_term
                    accel += wetting_term

            if enable_surface_tension and sigma_i > 0.0 and n_len_i > surface_tension_normal_threshold:
                n_j = normal[j]
                n_len_j = float(np.linalg.norm(n_j))
                n_hat_j = n_j / n_len_j if n_len_j > 1.0e-6 else np.zeros(3, dtype=np.float32)
                curvature += float(volume[j]) * float(np.dot(n_hat_j - n_hat_i, grad_w))

        if enable_surface_tension and sigma_i > 0.0 and n_len_i > surface_tension_normal_threshold:
            surface_accel = -sigma_i * curvature * n_hat_i / rho_i
            accel += surface_accel

        acceleration[i] = accel
        surface_acceleration[i] = surface_accel
        adhesion_acceleration[i] = adhesion_accel
        wetting_acceleration[i] = wetting_accel

    return acceleration, surface_acceleration, adhesion_acceleration, wetting_acceleration


def _compute_surface_fields_cpu(solver: Any, state: State) -> tuple[np.ndarray, np.ndarray]:
    model = solver.model
    particle_count = int(model.particle_count)
    q = _vec3_array(state.particle_q)
    worlds = np.asarray(model.particle_world.numpy(), dtype=np.int32)
    support = solver._support_radius_np()
    volume = np.asarray(state.sph.volume.numpy(), dtype=np.float32)
    color_field = np.zeros(particle_count, dtype=np.float32)
    normal = np.zeros((particle_count, 3), dtype=np.float32)
    active = solver._active_role_mask(SPHRole.FLUID, dynamic=True)
    neighbor = _active_neighbor_role_mask(solver)
    neighbor_indices = np.flatnonzero(neighbor).astype(np.int32)
    for i in np.flatnonzero(active).astype(np.int32):
        h_i = max(float(support[i]), 1.0e-6)
        c = 0.0
        n = np.zeros(3, dtype=np.float32)
        for j in neighbor_indices:
            if not _same_world_np(int(worlds[i]), int(worlds[j])):
                continue
            h = max(h_i, float(support[j]), 1.0e-6)
            vol_j = float(volume[j])
            r_vec = q[i] - q[j]
            r = float(np.linalg.norm(r_vec))
            if r >= h:
                continue
            c += vol_j * _kernel_weight_np(solver._sph_model.kernel_id, r, h)
            if i != j and r > 1.0e-6:
                n += vol_j * _kernel_gradient_np(solver._sph_model.kernel_id, r_vec, r, h)

        color_field[i] = c
        normal[i] = n
    return color_field, normal
