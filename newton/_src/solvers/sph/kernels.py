# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

SPH_ROLE_FLUID = wp.constant(0)
SPH_ROLE_BOUNDARY = wp.constant(1)
SPH_EPSILON = wp.constant(1.0e-6)
SPH_PI = wp.constant(3.141592653589793)
SPH_KERNEL_POLY6 = wp.constant(0)
SPH_KERNEL_CUBIC = wp.constant(1)
SPH_KERNEL_WENDLAND = wp.constant(2)
SPH_KERNEL_SPIKY = wp.constant(3)

_SPH_KERNEL_IDS = {
    "poly6": 0,
    "cubic": 1,
    "wendland": 2,
    "spiky": 3,
}


def sph_kernel_id(name: str) -> int:
    name = str(name).strip()
    try:
        return _SPH_KERNEL_IDS[name]
    except KeyError as exc:
        available = ", ".join(sph_kernel_names())
        raise ValueError(f"Unsupported SPH kernel '{name}'. Available kernels: {available}") from exc


def sph_kernel_names() -> tuple[str, ...]:
    return tuple(_SPH_KERNEL_IDS)


def sph_same_world_np(world_i: int, world_j: int) -> bool:
    """Return whether two SPH world identifiers interact in host-side checks."""

    return world_i == world_j or world_i == -1 or world_j == -1


def sph_is_neighbor_role_np(role: int) -> bool:
    """Return whether a host-side particle role participates in SPH neighbor sampling."""

    return role == 0 or role == 1


def sph_kernel_weight_np(kernel_id: int, r: float, h: float) -> float:
    """Evaluate a built-in SPH smoothing kernel in NumPy/Python host code."""

    if r >= h or h <= 0.0:
        return 0.0
    if kernel_id == 1:
        q = 2.0 * r / h
        coeff = 8.0 / (np.pi * h * h * h)
        if q < 1.0:
            return float(coeff * (1.0 - 1.5 * q * q + 0.75 * q * q * q))
        x = 2.0 - q
        return float(coeff * 0.25 * x * x * x)
    if kernel_id == 2:
        q = r / h
        x = 1.0 - q
        coeff = 21.0 / (2.0 * np.pi * h * h * h)
        return float(coeff * x * x * x * x * (1.0 + 4.0 * q))
    if kernel_id == 3:
        x = h - r
        coeff = 15.0 / (np.pi * h * h * h * h * h * h)
        return float(coeff * x * x * x)

    h2 = h * h
    x = h2 - r * r
    coeff = 315.0 / (64.0 * np.pi * h2 * h2 * h2 * h2 * h)
    return float(coeff * x * x * x)


def sph_kernel_gradient_np(kernel_id: int, r_vec: np.ndarray, r: float, h: float) -> np.ndarray:
    """Evaluate a built-in SPH smoothing-kernel gradient in NumPy/Python host code."""

    if r <= 1.0e-6 or r >= h or h <= 0.0:
        return np.zeros(3, dtype=np.float32)
    if kernel_id == 1:
        q = 2.0 * r / h
        coeff = 16.0 / (np.pi * h * h * h * h)
        if q < 1.0:
            dfdq = -3.0 * q + 2.25 * q * q
        else:
            x = 2.0 - q
            dfdq = -0.75 * x * x
        return np.asarray((coeff * dfdq / r) * r_vec, dtype=np.float32)
    if kernel_id == 2:
        q = r / h
        x = 1.0 - q
        coeff = 21.0 / (2.0 * np.pi * h * h * h)
        return np.asarray((-20.0 * coeff * x * x * x / (h * h)) * r_vec, dtype=np.float32)

    x = h - r
    coeff = -45.0 / (np.pi * h * h * h * h * h * h)
    return np.asarray((coeff * x * x / r) * r_vec, dtype=np.float32)


def sph_viscosity_laplacian_np(r: float, h: float) -> float:
    """Evaluate the SPH viscosity-kernel Laplacian in host-side checks."""

    if r >= h or h <= 0.0:
        return 0.0
    return float(45.0 * (h - r) / (np.pi * h * h * h * h * h * h))


def sph_pressure_from_density_np(
    rho: np.ndarray,
    rest_density: np.ndarray,
    sound_speed: np.ndarray,
    stiffness: np.ndarray,
    pressure_exponent: np.ndarray,
    pressure_min: np.ndarray,
    pressure_max: np.ndarray,
) -> np.ndarray:
    """Evaluate the SPH equation of state in host-side checks."""

    rest = np.maximum(rest_density.astype(np.float32, copy=False), 1.0e-6)
    exponent = np.maximum(pressure_exponent.astype(np.float32, copy=False), 1.0e-6)
    density_ratio = np.maximum(rho.astype(np.float32, copy=False), 1.0e-6) / rest
    pressure = np.zeros_like(rho, dtype=np.float32)

    sound_mask = sound_speed > 0.0
    linear_sound = sound_mask & (exponent == 1.0)
    nonlinear_sound = sound_mask & ~linear_sound
    pressure[linear_sound] = (
        sound_speed[linear_sound] * sound_speed[linear_sound] * (rho[linear_sound] - rest[linear_sound])
    )
    pressure[nonlinear_sound] = (
        rest[nonlinear_sound]
        * sound_speed[nonlinear_sound]
        * sound_speed[nonlinear_sound]
        * (np.power(density_ratio[nonlinear_sound], exponent[nonlinear_sound]) - 1.0)
        / exponent[nonlinear_sound]
    )

    stiffness_mask = (~sound_mask) & (stiffness > 0.0)
    linear_stiffness = stiffness_mask & (exponent == 1.0)
    nonlinear_stiffness = stiffness_mask & ~linear_stiffness
    pressure[linear_stiffness] = (
        stiffness[linear_stiffness] * (rho[linear_stiffness] - rest[linear_stiffness]) / rest[linear_stiffness]
    )
    pressure[nonlinear_stiffness] = (
        stiffness[nonlinear_stiffness]
        * (np.power(density_ratio[nonlinear_stiffness], exponent[nonlinear_stiffness]) - 1.0)
        / exponent[nonlinear_stiffness]
    )

    pressure = np.maximum(pressure, pressure_min)
    clamped = pressure_max > 0.0
    pressure[clamped] = np.minimum(pressure[clamped], pressure_max[clamped])
    return pressure.astype(np.float32)


@wp.func
def _is_sph_neighbor_role(role: int):
    return role == SPH_ROLE_FLUID or role == SPH_ROLE_BOUNDARY


@wp.func
def _is_sph_integrated_role(role: int):
    return role == SPH_ROLE_FLUID


@wp.func
def _same_world(world_i: int, world_j: int):
    return world_i == world_j or world_i == -1 or world_j == -1


@wp.func
def _support_radius(particle_h: float, default_h: float):
    h = particle_h
    if h <= 0.0:
        h = default_h
    return wp.max(h, SPH_EPSILON)


@wp.func
def _poly6_weight(r: float, h: float):
    result = 0.0
    if r < h:
        h2 = h * h
        x = h2 - r * r
        coeff = 315.0 / (64.0 * SPH_PI * h2 * h2 * h2 * h2 * h)
        result = coeff * x * x * x
    return result


@wp.func
def _spiky_weight(r: float, h: float):
    result = 0.0
    if r < h:
        x = h - r
        coeff = 15.0 / (SPH_PI * h * h * h * h * h * h)
        result = coeff * x * x * x
    return result


@wp.func
def _cubic_weight(r: float, h: float):
    result = 0.0
    if r < h:
        q = 2.0 * r / h
        coeff = 8.0 / (SPH_PI * h * h * h)
        if q < 1.0:
            result = coeff * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
        else:
            x = 2.0 - q
            result = coeff * 0.25 * x * x * x
    return result


@wp.func
def _wendland_weight(r: float, h: float):
    result = 0.0
    if r < h:
        q = r / h
        x = 1.0 - q
        coeff = 21.0 / (2.0 * SPH_PI * h * h * h)
        result = coeff * x * x * x * x * (1.0 + 4.0 * q)
    return result


@wp.func
def _spiky_gradient(r_vec: wp.vec3, r: float, h: float):
    grad = wp.vec3(0.0)
    if r > SPH_EPSILON and r < h:
        x = h - r
        coeff = -45.0 / (SPH_PI * h * h * h * h * h * h)
        grad = (coeff * x * x / r) * r_vec
    return grad


@wp.func
def _cubic_gradient(r_vec: wp.vec3, r: float, h: float):
    grad = wp.vec3(0.0)
    if r > SPH_EPSILON and r < h:
        q = 2.0 * r / h
        coeff = 16.0 / (SPH_PI * h * h * h * h)
        dfdq = 0.0
        if q < 1.0:
            dfdq = -3.0 * q + 2.25 * q * q
        else:
            x = 2.0 - q
            dfdq = -0.75 * x * x
        grad = (coeff * dfdq / r) * r_vec
    return grad


@wp.func
def _wendland_gradient(r_vec: wp.vec3, r: float, h: float):
    grad = wp.vec3(0.0)
    if r > SPH_EPSILON and r < h:
        q = r / h
        x = 1.0 - q
        coeff = 21.0 / (2.0 * SPH_PI * h * h * h)
        grad = (-20.0 * coeff * x * x * x / (h * h)) * r_vec
    return grad


@wp.func
def _viscosity_laplacian(r: float, h: float):
    result = 0.0
    if r < h:
        coeff = 45.0 / (SPH_PI * h * h * h * h * h * h)
        result = coeff * (h - r)
    return result


@wp.func
def _kernel_weight(kernel_id: int, r: float, h: float):
    if kernel_id == SPH_KERNEL_CUBIC:
        return _cubic_weight(r, h)
    if kernel_id == SPH_KERNEL_WENDLAND:
        return _wendland_weight(r, h)
    if kernel_id == SPH_KERNEL_SPIKY:
        return _spiky_weight(r, h)
    return _poly6_weight(r, h)


@wp.func
def _kernel_gradient(kernel_id: int, r_vec: wp.vec3, r: float, h: float):
    if kernel_id == SPH_KERNEL_CUBIC:
        return _cubic_gradient(r_vec, r, h)
    if kernel_id == SPH_KERNEL_WENDLAND:
        return _wendland_gradient(r_vec, r, h)
    return _spiky_gradient(r_vec, r, h)


@wp.func
def _pressure_from_density(
    rho: float,
    rest_density: float,
    sound_speed: float,
    stiffness: float,
    pressure_exponent: float,
    pressure_min: float,
    pressure_max: float,
):
    p = float(0.0)
    rest_density = wp.max(rest_density, SPH_EPSILON)
    exponent = wp.max(pressure_exponent, SPH_EPSILON)
    density_ratio = wp.max(rho, SPH_EPSILON) / rest_density

    if sound_speed > 0.0:
        if exponent == 1.0:
            p = sound_speed * sound_speed * (rho - rest_density)
        else:
            p = rest_density * sound_speed * sound_speed * (wp.pow(density_ratio, exponent) - 1.0) / exponent
    elif stiffness > 0.0:
        if exponent == 1.0:
            p = stiffness * (rho - rest_density) / rest_density
        else:
            p = stiffness * (wp.pow(density_ratio, exponent) - 1.0) / exponent

    p = wp.max(p, pressure_min)
    if pressure_max > 0.0:
        p = wp.min(p, pressure_max)
    return p
