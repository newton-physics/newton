# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Coupled spatial Coulomb contact in normal-first coordinates.

The reaction order is ``(normal, tangent_0, tangent_1, spin, roll_0, roll_1)``.
Sliding, torsional, and rolling friction share one elliptic cone. The local
metric must be positive definite on enabled rows; disabled friction rows are
omitted. Compliance, when present, is included in the supplied metric.
"""

import warp as wp
from warp.fem.linalg import symmetric_eigenvalues_qr

from .contact import solve_contact_coulomb_newton

vec5f = wp.types.vector(length=5, dtype=wp.float32)
vec6f = wp.types.vector(length=6, dtype=wp.float32)
mat55f = wp.types.matrix(shape=(5, 5), dtype=wp.float32)
mat66f = wp.types.matrix(shape=(6, 6), dtype=wp.float32)

wp.set_module_options({"enable_backward": False})


@wp.struct
class SpatialContactPrepared:
    delassus: mat66f
    scaling: vec6f
    eigenvectors: mat55f
    eigenvalues: vec5f
    normal_delassus: wp.float32
    coupling: vec5f
    spectral_coupling: vec5f
    status: wp.int32


@wp.struct
class SpatialContactResult:
    reaction: vec6f
    status: wp.int32


@wp.func
def prepare_spatial_contact(delassus: mat66f, friction: wp.vec3f) -> SpatialContactPrepared:
    """Cache the Schur spectrum for sliding, torsional and rolling coefficients.

    Status zero denotes success; status one denotes an invalid coefficient,
    active metric, or eigendecomposition. No diagonal approximation is used.
    """
    prepared = SpatialContactPrepared()
    prepared.delassus = delassus
    prepared.scaling = vec6f(1.0, friction[0], friction[0], friction[1], friction[2], friction[2])
    prepared.normal_delassus = delassus[0, 0]
    prepared.status = 1
    if not wp.isfinite(friction) or wp.min(friction) < 0.0:
        return prepared
    a = delassus[0, 0]
    if not wp.isfinite(a) or a <= 0.0:
        return prepared

    schur = mat55f(0.0)
    for i in range(6):
        for j in range(6):
            if prepared.scaling[i] > 0.0 and prepared.scaling[j] > 0.0:
                if not wp.isfinite(delassus[i, j]):
                    return prepared
                tolerance = 1.0e-5 * wp.max(a, wp.max(wp.abs(delassus[i, j]), wp.abs(delassus[j, i])))
                if wp.abs(delassus[i, j] - delassus[j, i]) > tolerance:
                    return prepared
    for i in range(5):
        if prepared.scaling[i + 1] > 0.0:
            prepared.coupling[i] = prepared.scaling[i + 1] * delassus[i + 1, 0]
    for i in range(5):
        for j in range(5):
            if prepared.scaling[i + 1] > 0.0 and prepared.scaling[j + 1] > 0.0:
                schur[i, j] = (
                    prepared.scaling[i + 1] * delassus[i + 1, j + 1] * prepared.scaling[j + 1]
                    - prepared.coupling[i] * prepared.coupling[j] / a
                )

    # Retain the existing 3D implementation without paying for a 5D spectrum.
    if friction[1] == 0.0 and friction[2] == 0.0:
        if friction[0] > 0.0:
            # The existing 3D solver uses unscaled friction coordinates. Test
            # their normalized Schur block to avoid mu**4 determinant underflow.
            s00 = delassus[1, 1] - (delassus[1, 0] / a) * delassus[0, 1]
            s01 = delassus[1, 2] - (delassus[1, 0] / a) * delassus[0, 2]
            s11 = delassus[2, 2] - (delassus[2, 0] / a) * delassus[0, 2]
            if s00 <= 0.0 or s11 <= 0.0:
                return prepared
            scale = wp.max(s00, wp.max(wp.abs(s01), s11))
            if (s00 / scale) * (s11 / scale) <= (s01 / scale) * (s01 / scale):
                return prepared
        prepared.status = 0
        return prepared

    spectral_scale = float(0.0)
    for i in range(5):
        for j in range(5):
            spectral_scale = wp.max(spectral_scale, wp.abs(schur[i, j]))
    if not wp.isfinite(spectral_scale) or spectral_scale <= 0.0:
        return prepared
    schur = schur / spectral_scale
    # Padding inactive rows by an uncoupled identity leaves the active solve
    # unchanged: the inactive RHS and coupling are identically zero.
    for i in range(5):
        if prepared.scaling[i + 1] == 0.0:
            schur[i, i] = 1.0
    eigenvalues, eigenvectors_rows = symmetric_eigenvalues_qr(schur, 1.0e-7)
    eigenvectors = wp.transpose(eigenvectors_rows)
    if not wp.isfinite(eigenvalues) or not wp.isfinite(eigenvectors) or wp.min(eigenvalues) <= 0.0:
        return prepared
    # QR failure must not silently remove coupling or invent a valid metric.
    reconstruction = eigenvectors * wp.diag(eigenvalues) * wp.transpose(eigenvectors) - schur
    orthogonality = wp.transpose(eigenvectors) * eigenvectors - wp.identity(n=5, dtype=wp.float32)
    if wp.ddot(reconstruction, reconstruction) > 1.0e-8 or wp.ddot(orthogonality, orthogonality) > 1.0e-8:
        return prepared
    prepared.eigenvalues = spectral_scale * eigenvalues
    prepared.eigenvectors = eigenvectors
    prepared.spectral_coupling = wp.transpose(eigenvectors) * prepared.coupling
    prepared.status = 0
    return prepared


@wp.func
def _spectral_solution(prepared: SpatialContactPrepared, rhs: vec5f, alpha: float) -> vec5f:
    value = vec5f(0.0)
    for i in range(5):
        value[i] = rhs[i] / (prepared.eigenvalues[i] + alpha)
    return value


@wp.func
def _recover_reaction(prepared: SpatialContactPrepared, spectral_s: vec5f, normal_rhs: float) -> vec6f:
    s = prepared.eigenvectors * spectral_s
    reaction = vec6f(0.0)
    reaction[0] = (wp.dot(prepared.spectral_coupling, spectral_s) - normal_rhs) / prepared.normal_delassus
    for i in range(5):
        reaction[i + 1] = -prepared.scaling[i + 1] * s[i]
    return reaction


@wp.func
def solve_spatial_contact(prepared: SpatialContactPrepared, free_velocity: vec6f) -> SpatialContactResult:
    """Solve the unified cone with a safeguarded scalar Schur-complement root.

    Result status is zero for success, one for invalid input/preparation, two
    for failure to bracket the root, and three for failure to converge.
    """
    result = SpatialContactResult()
    result.status = 1
    if prepared.status != 0 or not wp.isfinite(free_velocity):
        return result
    result.status = 0
    normal_rhs = free_velocity[0]
    if normal_rhs >= 0.0:
        return result
    if prepared.scaling[3] == 0.0 and prepared.scaling[4] == 0.0:
        d = prepared.delassus
        linear = wp.mat33f(d[1, 1], d[1, 2], d[1, 0], d[2, 1], d[2, 2], d[2, 0], d[0, 1], d[0, 2], d[0, 0])
        metric_scale = prepared.normal_delassus
        if prepared.scaling[1] > 0.0:
            for i in range(3):
                for j in range(3):
                    metric_scale = wp.max(metric_scale, wp.abs(linear[i, j]))
        r = solve_contact_coulomb_newton(
            linear / metric_scale,
            wp.vec3f(free_velocity[1], free_velocity[2], normal_rhs) / metric_scale,
            prepared.scaling[1],
        )
        result.reaction = vec6f(r[2], r[0], r[1], 0.0, 0.0, 0.0)
        if not wp.isfinite(result.reaction):
            result.status = 3
        return result

    rhs = vec5f(0.0)
    for i in range(5):
        rhs[i] = prepared.scaling[i + 1] * free_velocity[i + 1]
    rhs = wp.transpose(prepared.eigenvectors) * (rhs - (normal_rhs / prepared.normal_delassus) * prepared.coupling)
    coupling = prepared.spectral_coupling / prepared.normal_delassus
    offset = normal_rhs / prepared.normal_delassus
    s = _spectral_solution(prepared, rhs, 0.0)
    value = wp.length(s) - wp.dot(coupling, s) + offset
    tolerance = 2.0e-7 * wp.max(1.0, wp.max(wp.length(s), wp.abs(offset)))
    if value <= tolerance:
        result.reaction = _recover_reaction(prepared, s, normal_rhs)
        if not wp.isfinite(result.reaction):
            result.status = 3
        return result

    alpha_scale = wp.max(prepared.eigenvalues)
    lower = float(0.0)
    upper = float(1.0)
    bracketed = bool(False)
    for _ in range(64):
        s = _spectral_solution(prepared, rhs, upper * alpha_scale)
        value = wp.length(s) - wp.dot(coupling, s) + offset
        if wp.isfinite(value) and value <= 0.0:
            bracketed = True
            break
        upper *= 2.0
    if not bracketed:
        result.status = 2
        return result

    alpha = 0.5 * upper
    result.status = 3
    for _ in range(64):
        s = _spectral_solution(prepared, rhs, alpha * alpha_scale)
        s_norm = wp.length(s)
        value = s_norm - wp.dot(coupling, s) + offset
        tolerance = 2.0e-7 * wp.max(1.0, wp.max(s_norm, wp.abs(offset)))
        if wp.abs(value) <= tolerance:
            result.status = 0
            break
        if value > 0.0:
            lower = alpha
        else:
            upper = alpha
        t = _spectral_solution(prepared, s, alpha * alpha_scale)
        derivative = alpha_scale * (-wp.dot(s, t) / wp.max(s_norm, 1.0e-30) + wp.dot(coupling, t))
        candidate = 0.5 * (lower + upper)
        if wp.isfinite(derivative) and derivative != 0.0:
            newton = alpha - value / derivative
            # Keep a strict contraction even for nonmonotone scalar roots.
            margin = 0.05 * (upper - lower)
            if newton > lower + margin and newton < upper - margin:
                candidate = newton
        alpha = candidate
    result.reaction = _recover_reaction(prepared, s, normal_rhs)
    if not wp.isfinite(result.reaction):
        result.status = 3
    return result


@wp.func
def compute_spatial_contact_residual(prepared: SpatialContactPrepared, reaction: vec6f, velocity: vec6f) -> vec6f:
    """Return the metric-scaled canonical Alart--Curnier natural-map residual.

    ``velocity`` includes the compliant velocity and selected normal target.
    Disabled components report their reaction, which must remain zero.
    """
    rho = vec6f(0.0)
    gamma = vec6f(0.0)
    trace = prepared.normal_delassus
    active_count = float(1.0)
    for i in range(6):
        coefficient = prepared.scaling[i]
        if coefficient > 0.0:
            rho[i] = reaction[i] / coefficient
            gamma[i] = velocity[i] * coefficient
            if i > 0:
                trace += coefficient * coefficient * prepared.delassus[i, i]
                active_count += 1.0
    scale = wp.sqrt(wp.max(trace / active_count, 1.0e-30))
    rho *= scale
    gamma /= scale
    friction_velocity = vec5f(gamma[1], gamma[2], gamma[3], gamma[4], gamma[5])
    gamma[0] += wp.length(friction_velocity)
    value = rho - gamma
    tangent = vec5f(value[1], value[2], value[3], value[4], value[5])
    tangent_norm = wp.length(tangent)
    projected = vec6f(0.0)
    if tangent_norm <= value[0]:
        projected = value
    elif tangent_norm > -value[0]:
        projected[0] = 0.5 * (value[0] + tangent_norm)
        for i in range(5):
            projected[i + 1] = projected[0] * tangent[i] / tangent_norm
    residual = rho - projected
    for i in range(6):
        if prepared.scaling[i] == 0.0:
            residual[i] = reaction[i]
    return residual
