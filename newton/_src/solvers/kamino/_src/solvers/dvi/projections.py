# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared contact projection functions for dense and sparse DVI kernels."""

import warp as wp

from ...core.math import FLOAT32_EPS
from ..padmm.math import project_to_coulomb_cone

float32 = wp.float32
mat33f = wp.mat33f
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
def project_contact_block_update(
    lambda_old: vec3f,
    v_c: vec3f,
    D_diag: vec3f,
    D_block_inv: mat33f,
    regularization: float32,
    omega: float32,
    mu: float32,
) -> vec3f:
    """Apply a block-preconditioned contact projection.

    Computes ``lambda_next = project_K(lambda - omega * B * v_aug)``.
    """
    inv_diag_norm = wp.abs(D_block_inv[0, 0]) + wp.abs(D_block_inv[1, 1]) + wp.abs(D_block_inv[2, 2])
    if inv_diag_norm > FLOAT32_EPS:
        return project_to_coulomb_cone(lambda_old - omega * (D_block_inv * v_c), mu)
    return project_contact_diagonal_update(lambda_old, v_c, D_diag, regularization, omega, mu)


@wp.func
def contact_normal_preconditioner(D_diag: vec3f) -> vec3f:
    """Return an isotropic contact preconditioner based on normal effective mass."""
    D_eff = D_diag.z
    return vec3f(D_eff, D_eff, D_eff)
