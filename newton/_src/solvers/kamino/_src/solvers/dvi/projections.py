# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared contact projection functions for dense and sparse DVI kernels."""

import warp as wp

from ...core.math import FLOAT32_EPS
from ..padmm.math import project_to_coulomb_cone

float32 = wp.float32
vec3f = wp.vec3f


@wp.func
def project_contact_diagonal_update(
    lambda_old: vec3f,
    v_c: vec3f,
    D_diag: vec3f,
    regularization: float32,
    omega: float32,
    mu: float32,
) -> vec3f:
    """Apply a diagonally preconditioned contact projection.

    Computes ``lambda_next = project_K(lambda - omega * B * v_aug)``.
    """
    lambda_arg = lambda_old
    if D_diag.x > FLOAT32_EPS:
        lambda_arg.x = lambda_old.x - omega * v_c.x / (D_diag.x + regularization)
    if D_diag.y > FLOAT32_EPS:
        lambda_arg.y = lambda_old.y - omega * v_c.y / (D_diag.y + regularization)
    if D_diag.z > FLOAT32_EPS:
        lambda_arg.z = lambda_old.z - omega * v_c.z / (D_diag.z + regularization)
    return project_to_coulomb_cone(lambda_arg, mu)


@wp.func
def contact_diagonal_preconditioner(D_diag: vec3f) -> vec3f:
    """Return the contact preconditioner as a shared tangential and true normal diagonal.

    The two tangential rows must share one scalar, otherwise the friction disk
    ``norm(lambda_t) <= mu * lambda_n`` is scaled into an ellipse and the
    Coulomb-cone projection is no longer a projection. Taking their maximum
    also bounds each row's effective relaxation by ``omega``: preconditioning
    a tangential row by the smaller normal diagonal over-relaxes it whenever
    the contact carries a lever arm, reaching ``D_tt / D_nn = 3.5`` for a
    solid sphere. The normal row keeps its own diagonal, which converges far
    faster than a shared maximum under high mass ratios.
    """
    D_t = wp.max(D_diag.x, D_diag.y)
    return vec3f(D_t, D_t, D_diag.z)
