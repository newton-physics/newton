# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

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
