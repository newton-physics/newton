# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Local isotropic Coulomb-contact numerical primitives.

This module is deliberately independent of Kamino containers. Contact vectors
use Kamino's normal-last convention ``(tangent_0, tangent_1, normal)``.

The local solve expects a symmetric positive-definite ``3 x 3`` Delassus
block. Degenerate blocks must be regularized during contact preprocessing.
"""

from __future__ import annotations

import warp as wp

__all__ = [
    "compute_contact_scaled_alart_curnier_residual",
    "project_contact_coulomb_cone",
    "solve_contact_coulomb_newton",
]

wp.set_module_options({"enable_backward": False})


@wp.func
def _solve_symmetric_mat22(
    a00: wp.float32,
    a01: wp.float32,
    a11: wp.float32,
    b0: wp.float32,
    b1: wp.float32,
) -> wp.vec2f:
    det = a00 * a11 - a01 * a01
    inv_det = 1.0 / det
    return wp.vec2f((a11 * b0 - a01 * b1) * inv_det, (a00 * b1 - a01 * b0) * inv_det)


@wp.func
def _compute_sliding_root_value(
    tangent00: wp.float32,
    tangent01: wp.float32,
    tangent11: wp.float32,
    tangent_rhs: wp.vec2f,
    friction_normal_tangent: wp.vec2f,
    friction_normal_rhs: wp.float32,
    alpha: wp.float32,
) -> wp.float32:
    shifted00 = tangent00 + alpha
    shifted11 = tangent11 + alpha
    s = _solve_symmetric_mat22(
        shifted00,
        tangent01,
        shifted11,
        tangent_rhs[0],
        tangent_rhs[1],
    )
    return wp.length(s) - wp.dot(friction_normal_tangent, s) + friction_normal_rhs


@wp.func
def _compute_sliding_root_data(
    tangent00: wp.float32,
    tangent01: wp.float32,
    tangent11: wp.float32,
    tangent_rhs: wp.vec2f,
    friction_normal_tangent: wp.vec2f,
    friction_normal_rhs: wp.float32,
    alpha: wp.float32,
) -> wp.vec4f:
    shifted00 = tangent00 + alpha
    shifted11 = tangent11 + alpha
    determinant = shifted00 * shifted11 - tangent01 * tangent01
    inverse_determinant = 1.0 / determinant
    s = wp.vec2f(
        (shifted11 * tangent_rhs[0] - tangent01 * tangent_rhs[1]) * inverse_determinant,
        (shifted00 * tangent_rhs[1] - tangent01 * tangent_rhs[0]) * inverse_determinant,
    )
    t = wp.vec2f(
        (shifted11 * s[0] - tangent01 * s[1]) * inverse_determinant,
        (shifted00 * s[1] - tangent01 * s[0]) * inverse_determinant,
    )
    s_norm = wp.length(s)
    value = s_norm - wp.dot(friction_normal_tangent, s) + friction_normal_rhs
    derivative = wp.float32(0.0)
    if s_norm > 1.0e-30:
        derivative = -(wp.dot(s, t)) / s_norm + wp.dot(friction_normal_tangent, t)
    return wp.vec4f(value, derivative, s[0], s[1])


@wp.func
def _solve_contact_coulomb_newton_components(
    normal_delassus: wp.float32,
    normal_tangent: wp.vec2f,
    tangent_delassus00: wp.float32,
    tangent_delassus01: wp.float32,
    tangent_delassus11: wp.float32,
    normal_rhs: wp.float32,
    tangent_rhs_raw: wp.vec2f,
    friction: wp.float32,
) -> wp.vec3f:
    """Solve from layout-independent normal and tangential components."""
    # These branches also avoid touching an unused, potentially ill-conditioned
    # tangential block for separating or frictionless contacts.
    if normal_rhs >= 0.0:
        return wp.vec3f(0.0, 0.0, 0.0)
    if friction <= 0.0:
        return wp.vec3f(0.0, 0.0, -normal_rhs / normal_delassus)

    inverse_normal_delassus = 1.0 / normal_delassus
    tangent00 = tangent_delassus00 - normal_tangent[0] * normal_tangent[0] * inverse_normal_delassus
    tangent01 = tangent_delassus01 - normal_tangent[0] * normal_tangent[1] * inverse_normal_delassus
    tangent11 = tangent_delassus11 - normal_tangent[1] * normal_tangent[1] * inverse_normal_delassus
    tangent_rhs = tangent_rhs_raw - (normal_rhs * inverse_normal_delassus) * normal_tangent
    friction_over_normal = friction * inverse_normal_delassus
    friction_normal_tangent = friction_over_normal * normal_tangent
    friction_normal_rhs = friction_over_normal * normal_rhs

    unshifted = _solve_symmetric_mat22(
        tangent00,
        tangent01,
        tangent11,
        tangent_rhs[0],
        tangent_rhs[1],
    )
    value_at_zero = wp.length(unshifted) - wp.dot(friction_normal_tangent, unshifted) + friction_normal_rhs
    if value_at_zero <= 1.0e-7:
        tangent_reaction = -unshifted
        normal_reaction = -(wp.dot(normal_tangent, tangent_reaction) + normal_rhs) * inverse_normal_delassus
        return wp.vec3f(tangent_reaction[0], tangent_reaction[1], normal_reaction)

    # Normalizing alpha by the Schur-complement scale keeps the fixed root
    # tolerances useful across contact blocks with different magnitudes.
    alpha_scale = wp.max(wp.abs(tangent00), wp.abs(tangent01))
    alpha_scale = wp.max(alpha_scale, wp.abs(tangent11))
    alpha_scale = wp.max(alpha_scale, 1.0e-20)

    lower = wp.float32(0.0)
    upper = wp.float32(1.0)
    last_s = unshifted
    for _ in range(64):
        upper_value = _compute_sliding_root_value(
            tangent00,
            tangent01,
            tangent11,
            tangent_rhs,
            friction_normal_tangent,
            friction_normal_rhs,
            upper * alpha_scale,
        )
        if upper_value <= 0.0:
            break
        upper = upper * 2.0

    alpha = 0.5 * (lower + upper)
    for _ in range(12):
        root_data = _compute_sliding_root_data(
            tangent00,
            tangent01,
            tangent11,
            tangent_rhs,
            friction_normal_tangent,
            friction_normal_rhs,
            alpha * alpha_scale,
        )
        value = root_data[0]
        derivative = root_data[1] * alpha_scale
        last_s = wp.vec2f(root_data[2], root_data[3])

        if wp.abs(value) <= 1.0e-7 or wp.abs(upper - lower) <= 1.0e-7 * (1.0 + upper):
            break

        if value > 0.0:
            lower = alpha
        else:
            upper = alpha

        width = upper - lower
        next_alpha = 0.5 * (lower + upper)
        if derivative != 0.0 and wp.isfinite(derivative):
            newton_alpha = alpha - value / derivative
            if (
                newton_alpha > lower
                and newton_alpha < upper
                and wp.abs(newton_alpha - alpha) > 1.0e-7 * wp.max(1.0, width)
            ):
                next_alpha = newton_alpha
        alpha = next_alpha

    tangent_reaction = -last_s
    normal_reaction = -(wp.dot(normal_tangent, tangent_reaction) + normal_rhs) * inverse_normal_delassus
    return wp.vec3f(tangent_reaction[0], tangent_reaction[1], normal_reaction)


@wp.func
def solve_contact_coulomb_newton(
    delassus: wp.mat33f,
    free_velocity: wp.vec3f,
    friction: wp.float32,
) -> wp.vec3f:
    """Solve one normal-last isotropic Coulomb contact.

    The returned impulse ``reaction`` satisfies the contact law for
    ``velocity = delassus @ reaction + free_velocity``. Sliding contacts are
    reduced to a scalar root and solved by bracketed Newton. Every rejected or
    unusable Newton step falls back to bisection of the current bracket.

    Args:
        delassus: Symmetric positive-definite local Delassus block.
        free_velocity: Contact velocity before applying the local impulse.
        friction: Nonnegative isotropic Coulomb friction coefficient.

    Returns:
        The normal-last contact impulse.
    """
    return _solve_contact_coulomb_newton_components(
        delassus[2, 2],
        wp.vec2f(delassus[0, 2], delassus[1, 2]),
        delassus[0, 0],
        delassus[0, 1],
        delassus[1, 1],
        free_velocity[2],
        wp.vec2f(free_velocity[0], free_velocity[1]),
        friction,
    )


@wp.func
def project_contact_coulomb_cone(value: wp.vec3f, friction: wp.float32) -> wp.vec3f:
    """Project a normal-last vector onto an isotropic Coulomb cone.

    Args:
        value: Normal-last vector to project.
        friction: Nonnegative isotropic Coulomb friction coefficient.

    Returns:
        The Euclidean projection of ``value``. Invalid friction disables the
        contact and returns zero, while a non-finite vector is preserved for
        the caller's projection-status check.
    """
    if not wp.isfinite(value):
        return value
    if not wp.isfinite(friction) or friction < 0.0:
        return wp.vec3f(0.0)

    tangent_norm = wp.length(wp.vec2f(value[0], value[1]))
    normal = value[2]
    if normal + friction * tangent_norm <= 0.0:
        return wp.vec3f(0.0)
    if tangent_norm <= friction * normal:
        return value

    projected_normal = (normal + friction * tangent_norm) / (1.0 + friction * friction)
    tangent_scale = friction * projected_normal / tangent_norm
    return wp.vec3f(tangent_scale * value[0], tangent_scale * value[1], projected_normal)


@wp.func
def _compute_contact_delassus_scale(delassus: wp.mat33f) -> wp.float32:
    trace_scale = (wp.abs(delassus[0, 0]) + wp.abs(delassus[1, 1]) + wp.abs(delassus[2, 2])) / 3.0
    return wp.sqrt(wp.max(trace_scale, 1.0e-30))


@wp.func
def compute_contact_scaled_alart_curnier_residual(
    delassus: wp.mat33f,
    reaction: wp.vec3f,
    velocity: wp.vec3f,
    friction: wp.float32,
) -> wp.vec3f:
    """Compute the Delassus-scaled Alart--Curnier contact residual.

    Args:
        delassus: Symmetric positive-definite local Delassus block.
        reaction: Normal-last contact impulse.
        velocity: Resulting normal-last contact velocity.
        friction: Nonnegative isotropic Coulomb friction coefficient.

    Returns:
        The normal-last scaled natural-map residual.
    """
    scale = _compute_contact_delassus_scale(delassus)
    scaled_reaction = scale * reaction
    scaled_velocity = velocity / scale
    modified_velocity = wp.vec3f(
        scaled_velocity[0],
        scaled_velocity[1],
        scaled_velocity[2] + friction * wp.length(wp.vec2f(scaled_velocity[0], scaled_velocity[1])),
    )
    projected = project_contact_coulomb_cone(scaled_reaction - modified_velocity, friction)
    return scaled_reaction - projected
