# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Double-precision analytic mechanics for the three-body Cartesian leg."""

from __future__ import annotations

import math

import warp as wp

from ..data import validate as validate_reference
from ..profile import validate as validate_profile

Vec5 = wp.types.vector(length=5, dtype=wp.float64)
Mat5 = wp.types.matrix(shape=(5, 5), dtype=wp.float64)
_Jacobian = wp.types.matrix(shape=(6, 5), dtype=wp.float64)
_Vec6 = wp.types.vector(length=6, dtype=wp.float64)
_HALF_PI = wp.constant(wp.float64(math.pi / 2.0))


@wp.struct
class Params:
    """Store lengths [m], masses [kg], local COMs [m], and COM inertias [kg m^2]."""

    lengths: wp.vec2d
    masses: wp.vec3d
    com0: wp.vec2d
    com1: wp.vec2d
    com2: wp.vec2d
    inertias: wp.vec3d


def make_params(reference: dict, profile: dict) -> Params:
    """Validate host inputs once and pack only the three declared bodies."""
    validate_reference(reference)
    return make_params_initial(reference["lengths_m"], profile)


def make_params_initial(lengths_m, profile: dict) -> Params:
    """Pack human geometry and inertial properties without measured targets."""
    validate_profile(profile)
    lengths = tuple(float(value) for value in lengths_m)
    if len(lengths) != 2 or not all(math.isfinite(value) and value > 0.0 for value in lengths):
        raise ValueError("lengths_m must contain two finite, positive segment lengths")
    params = Params()
    params.lengths = wp.vec2d(*lengths)
    params.masses = wp.vec3d(*profile["masses_kg"])
    params.com0 = wp.vec2d(*profile["com_local_m"][0])
    params.com1 = wp.vec2d(*profile["com_local_m"][1])
    params.com2 = wp.vec2d(*profile["com_local_m"][2])
    params.inertias = wp.vec3d(*profile["inertias_kg_m2"])
    return params


@wp.func
def foot_angle(q: Vec5) -> wp.float64:
    """Return the foot angle [rad] in the CPU cumulative-sum order."""
    return ((q[2] + q[3]) + q[4]) + _HALF_PI


@wp.func
def ankle(q: Vec5, p: Params) -> tuple[wp.vec2d, Vec5, Vec5]:
    """Return ankle position [m] and its x/z coordinate Jacobian rows."""
    angle0 = q[2]
    angle1 = q[2] + q[3]
    radius0 = wp.vec2d(wp.cos(angle0) * p.lengths[0], wp.sin(angle0) * p.lengths[0])
    radius1 = wp.vec2d(wp.cos(angle1) * p.lengths[1], wp.sin(angle1) * p.lengths[1])
    zero = wp.float64(0.0)
    one = wp.float64(1.0)
    position = wp.vec2d(q[0], q[1]) + (radius0 + radius1)
    jacobian_x = Vec5(one, zero, -radius0[1] - radius1[1], -radius1[1], zero)
    jacobian_z = Vec5(zero, one, radius0[0] + radius1[0], radius1[0], zero)
    return position, jacobian_x, jacobian_z


@wp.func
def _rotate(angle: wp.float64, local: wp.vec2d) -> wp.vec2d:
    """Rotate a proximal-frame offset into world x/z coordinates."""
    cosine = wp.cos(angle)
    sine = wp.sin(angle)
    return wp.vec2d(cosine * local[0] - sine * local[1], sine * local[0] + cosine * local[1])


@wp.func
def dynamics(q: Vec5, v: Vec5, p: Params, gravity: wp.float64) -> tuple[Mat5, Vec5]:
    """Return mass and centripetal/gravity bias in ``M @ acceleration + bias = load``."""
    angle0 = q[2]
    angle1 = q[2] + q[3]
    angle2 = foot_angle(q)
    radius0 = wp.vec2d(wp.cos(angle0) * p.lengths[0], wp.sin(angle0) * p.lengths[0])
    radius1 = wp.vec2d(wp.cos(angle1) * p.lengths[1], wp.sin(angle1) * p.lengths[1])
    com0 = _rotate(angle0, p.com0)
    com1 = _rotate(angle1, p.com1)
    com2 = _rotate(angle2, p.com2)
    zero = wp.float64(0.0)
    one = wp.float64(1.0)
    jacobian = _Jacobian(
        one,
        zero,
        -com0[1],
        zero,
        zero,
        zero,
        one,
        com0[0],
        zero,
        zero,
        one,
        zero,
        -radius0[1] - com1[1],
        -com1[1],
        zero,
        zero,
        one,
        radius0[0] + com1[0],
        com1[0],
        zero,
        one,
        zero,
        (-radius0[1] - radius1[1]) - com2[1],
        -radius1[1] - com2[1],
        -com2[1],
        zero,
        one,
        (radius0[0] + radius1[0]) + com2[0],
        radius1[0] + com2[0],
        com2[0],
    )
    omega0 = v[2]
    omega1 = v[2] + v[3]
    omega2 = (v[2] + v[3]) + v[4]
    squared0 = omega0 * omega0
    squared1 = omega1 * omega1
    squared2 = omega2 * omega2
    centripetal0 = -com0 * squared0
    centripetal1 = -com1 * squared1 - radius0 * squared0
    centripetal2 = -com2 * squared2 - (radius0 * squared0 + radius1 * squared1)
    force = _Vec6(
        p.masses[0] * centripetal0[0],
        p.masses[0] * (centripetal0[1] + gravity),
        p.masses[1] * centripetal1[0],
        p.masses[1] * (centripetal1[1] + gravity),
        p.masses[2] * centripetal2[0],
        p.masses[2] * (centripetal2[1] + gravity),
    )
    mass = Mat5()
    bias = Vec5()
    for i in range(5):
        for row in range(6):
            bias[i] += jacobian[row, i] * force[row]
        for j in range(5):
            entry = wp.float64(0.0)
            for row in range(6):
                entry = entry + jacobian[row, i] * (p.masses[row // 2] * jacobian[row, j])
            rotational = wp.float64(0.0)
            for body in range(3):
                if i >= 2 and j >= 2 and i <= body + 2 and j <= body + 2:
                    rotational = rotational + p.inertias[body]
            mass[i, j] = entry + rotational
    return mass, bias


@wp.func
def solve(m: Mat5, rhs: Vec5) -> Vec5:
    """Solve the unmodified SPD mass matrix; invalid factors propagate nonfinite values."""
    lower = Mat5()
    for i in range(5):
        for j in range(i + 1):
            value = m[i, j]
            for k in range(j):
                value = value - lower[i, k] * lower[j, k]
            if i == j:
                lower[i, j] = wp.sqrt(value)
            else:
                lower[i, j] = value / lower[j, j]
    intermediate = Vec5()
    for i in range(5):
        value = rhs[i]
        for j in range(i):
            value = value - lower[i, j] * intermediate[j]
        intermediate[i] = value / lower[i, i]
    result = Vec5()
    for reverse_i in range(5):
        i = 4 - reverse_i
        value = intermediate[i]
        for j in range(i + 1, 5):
            value = value - lower[j, i] * result[j]
        result[i] = value / lower[i, i]
    return result


@wp.func_grad(solve)
def _adj_solve(m: Mat5, rhs: Vec5, adj_result: Vec5):
    """Differentiate the symmetric matrix defined by the forward solve's lower triangle."""
    result = solve(m, rhs)
    dual = solve(m, adj_result)
    wp.adjoint[rhs] += dual
    matrix_gradient = Mat5()
    for i in range(5):
        matrix_gradient[i, i] = -dual[i] * result[i]
        for j in range(i):
            # One lower-triangular entry controls both symmetric matrix entries.
            matrix_gradient[i, j] = -dual[i] * result[j] - dual[j] * result[i]
    wp.adjoint[m] += matrix_gradient
