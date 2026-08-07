# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared contact projection functions for dense and sparse DVI kernels."""

import warp as wp

from ...core.math import FLOAT32_EPS

float32 = wp.float32
vec2f = wp.vec2f


@wp.func
def project_contact_normal_update(
    lambda_old: float32,
    velocity: float32,
    diagonal: float32,
    regularization: float32,
    omega: float32,
) -> float32:
    """Project one normal contact update onto the nonnegative half-line."""
    if diagonal <= FLOAT32_EPS:
        return lambda_old
    return wp.max(float32(0.0), lambda_old - omega * velocity / (diagonal + regularization))


@wp.func
def project_contact_tangent_update(
    lambda_old: vec2f,
    velocity: vec2f,
    diagonal: vec2f,
    off_diagonal: float32,
    regularization: float32,
    omega: float32,
    lambda_max: float32,
) -> vec2f:
    """Project a tangential contact update onto its Coulomb disk."""
    lambda_new = lambda_old
    scalar_diagonal = wp.max(diagonal.x, diagonal.y)
    a00 = diagonal.x + regularization
    a11 = diagonal.y + regularization
    determinant = a00 * a11 - off_diagonal * off_diagonal
    if determinant > FLOAT32_EPS * a00 * a11:
        delta = vec2f(
            (a11 * velocity.x - off_diagonal * velocity.y) / determinant,
            (a00 * velocity.y - off_diagonal * velocity.x) / determinant,
        )
        lambda_new -= omega * delta
    else:
        if scalar_diagonal > FLOAT32_EPS:
            lambda_new -= omega * velocity / (scalar_diagonal + regularization)
    tangent_norm = wp.length(lambda_new)
    if tangent_norm > lambda_max:
        # A non-scalar inverse changes the Euclidean-disk fixed point, so use
        # one shared tangent scale whenever Coulomb sliding is active.
        lambda_new = lambda_old
        if scalar_diagonal > FLOAT32_EPS:
            lambda_new -= omega * velocity / (scalar_diagonal + regularization)
        tangent_norm = wp.length(lambda_new)
    if tangent_norm > lambda_max and tangent_norm > FLOAT32_EPS:
        lambda_new *= lambda_max / tangent_norm
    return lambda_new
