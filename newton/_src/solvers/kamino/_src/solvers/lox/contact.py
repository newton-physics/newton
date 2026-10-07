# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Local contact laws of the LOX projection.

The velocity targets and the normal compliance define the right-hand side of each
contact law. The isotropic Coulomb solve uses Kamino's normal-last convention
``(tangent_0, tangent_1, normal)`` and expects a symmetric positive-definite ``3 x 3``
Delassus block; degenerate blocks must be regularized during contact preprocessing.

The coupled spatial solve uses the normal-first reaction order
``(normal, tangent_0, tangent_1, spin, roll_0, roll_1)``. Sliding, torsional, and
rolling friction share one elliptic cone scaled by ``(1, mu, mu, mu_spin, mu_roll, mu_roll)``.
The local metric must be positive definite on enabled rows; disabled friction rows are
omitted, and compliance, when present, is included in the supplied metric. The solve
eliminates the normal reaction and works on the ``5 x 5`` Schur complement of the scaled
friction rows, whose eigendecomposition is computed once per metric by
:func:`prepare_spatial_contact`.

These primitives are independent of Kamino containers.
"""

from enum import IntEnum

import warp as wp
from warp.fem.linalg import symmetric_eigenvalues_qr

from ...core.types import mat36f, mat66f, vec6f
from ..padmm.math import project_to_coulomb_cone
from .system_kernels import apply_body_weight, normalize_symmetric_matrix

###
# Module interface
###

__all__ = [
    "SpatialContactStatus",
    "angular_contact_reaction",
    "clamp_contact_reaction",
    "compute_contact_delassus_scale",
    "compute_contact_normal_compliance",
    "compute_contact_penetration_bias",
    "compute_contact_recovery_fraction",
    "compute_contact_restitution_target",
    "compute_contact_scaled_alart_curnier_residual",
    "compute_contact_velocity_target",
    "compute_spatial_contact_residual",
    "linear_contact_reaction",
    "mat55f",
    "pack_contact_reaction",
    "pack_spatial_friction",
    "prepare_spatial_contact",
    "scatter_spatial_impulse",
    "solve_contact_coulomb_newton",
    "solve_spatial_contact",
    "spatial_body_velocity",
    "spatial_body_wrench",
    "spatial_contact_delassus",
    "spatial_contact_velocity",
    "spatial_contact_velocity_scale",
    "spatial_friction_scaling",
    "vec5f",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###


_ROOT_EXPANSION_STEPS = 64


_QR_TOLERANCE = 1.0e-7


_DECOMPOSITION_TOLERANCE = 1.0e-8


###
# Types
###


class _SlidingRootStatus(IntEnum):
    """Outcome of the sliding-root solve of a contact."""

    SOLVED = 0
    """The sliding root, or the sticking solution, was found."""

    NOT_BRACKETED = 2
    """No shift made the root function non-positive."""

    NOT_CONVERGED = 3
    """The iteration budget ran out before the root tolerance was met."""


class SpatialContactStatus(IntEnum):
    """Outcome of the spatial contact solve; shares the sliding-root codes."""

    SOLVED = _SlidingRootStatus.SOLVED
    """The spatial contact solution was found."""

    INVALID = 1
    """Reported by callers when :func:`prepare_spatial_contact` rejects the local block."""

    NOT_BRACKETED = _SlidingRootStatus.NOT_BRACKETED
    """No shift made the sliding root function non-positive."""

    NOT_CONVERGED = _SlidingRootStatus.NOT_CONVERGED
    """The iteration budget ran out before the root tolerance was met."""


vec5f = wp.types.vector(length=5, dtype=wp.float32)


mat55f = wp.types.matrix(shape=(5, 5), dtype=wp.float32)


###
# Functions
###


@wp.func
def compute_contact_normal_compliance(
    stiffness: wp.float32,
    damping: wp.float32,
    time_step: wp.float32,
) -> wp.float32:
    """Convert a normal spring-damper to a velocity-impulse compliance [1/kg].

    Backward Euler on the normal force ``k penetration - d v`` gives the compliance
    ``1 / (h (k h + d))``. Add it to the mechanical normal Delassus diagonal before
    solving the local law. Only the law residual includes it: the impulse scattered
    to bodies is ``J^T`` times the reaction change.

    Args:
        stiffness: Normal contact stiffness [N/m]; non-positive selects hard contact.
        damping: Normal contact damping [N s/m].
        time_step: Time step [s].

    Returns:
        The compliance, or zero for hard contact.
    """
    if stiffness <= 0.0:
        return 0.0
    return 1.0 / (time_step * (stiffness * time_step + wp.max(damping, 0.0)))


@wp.func
def compute_contact_recovery_fraction(
    stiffness: wp.float32,
    damping: wp.float32,
    time_step: wp.float32,
) -> wp.float32:
    """Return the penetration recovery fraction ``k h / (k h + d)`` of a normal spring-damper.

    Args:
        stiffness: Positive normal contact stiffness [N/m].
        damping: Normal contact damping [N s/m].
        time_step: Time step [s].
    """
    return stiffness * time_step / (stiffness * time_step + wp.max(damping, 0.0))


@wp.func
def compute_contact_penetration_bias(
    distance: wp.float32,
    time_step: wp.float32,
    stabilization_fraction: wp.float32,
) -> wp.float32:
    """Compute the affine normal bias for compliant penetration recovery.

    Args:
        distance: Beginning-of-step margin-shifted signed distance [m].
        time_step: Time step [s].
        stabilization_fraction: Penetration recovery fraction; one gives backward Euler.

    Returns:
        Normal right-hand-side bias [m/s]. Add once to the reaction-free
        mechanical velocity; subtract any restitution target separately.
    """
    return stabilization_fraction * wp.min(distance, 0.0) / time_step


@wp.func
def compute_contact_restitution_target(
    distance: wp.float32,
    previous_normal_velocity: wp.float32,
    free_normal_velocity: wp.float32,
    restitution: wp.float32,
    time_step: wp.float32,
) -> wp.float32:
    """Select the trial-dependent speculative restitution target.

    Evaluate this at each local update from the mechanical, reaction-free
    right-hand side before adding penetration bias. The beginning-of-step
    distance and velocity remain frozen, while activation is re-evaluated at each update.
    A positive gap that the trial does not consume keeps the speculative gap
    target; otherwise the Newton restitution target applies. Penetration
    recovery is supplied separately by :func:`compute_contact_penetration_bias`.

    Args:
        distance: Beginning-of-step margin-shifted signed distance [m].
        previous_normal_velocity: Beginning-of-step normal velocity [m/s].
        free_normal_velocity: Current reaction-free local normal velocity [m/s].
        restitution: Newton restitution coefficient.
        time_step: Time step [s].

    Returns:
        Normal target [m/s], to subtract from the local right-hand side.
    """
    if distance > 0.0 and free_normal_velocity + distance / time_step >= 0.0:
        return -distance / time_step
    return -restitution * previous_normal_velocity


@wp.func
def compute_contact_velocity_target(
    distance: wp.float32,
    previous_normal_velocity: wp.float32,
    restitution: wp.float32,
    time_step: wp.float32,
    stabilization_fraction: wp.float32,
    dead_zone: wp.float32,
) -> wp.float32:
    """Compute the minimum end-of-step normal contact velocity.

    Args:
        distance: Margin-shifted signed contact distance [m].
        previous_normal_velocity: Begin-of-step normal velocity [m/s].
        restitution: Newton restitution coefficient.
        time_step: Time step [s].
        stabilization_fraction: Penetration recovery fraction.
        dead_zone: Symmetric distance dead zone [m].

    Returns:
        Minimum feasible normal velocity [m/s].
    """
    distance_effective = wp.sign(distance) * wp.max(wp.abs(distance) - dead_zone, 0.0)
    velocity_gap = (
        -(stabilization_fraction * wp.min(distance_effective, 0.0) + wp.max(distance_effective, 0.0)) / time_step
    )
    velocity_target = velocity_gap
    closed = distance <= dead_zone
    approaching = previous_normal_velocity < 0.0
    if closed and approaching:
        velocity_bounce = -restitution * previous_normal_velocity
        velocity_target = wp.max(velocity_gap, velocity_bounce)
    return velocity_target


def _make_sliding_root_solver(
    shifted_solve, system_type, vector_type, *, iterations: int, tolerance: float, newton_margin: float
):
    """Build the bracketed Newton solve of the sliding friction condition.

    Eliminating the normal reaction from a Coulomb cone leaves the friction
    impulse ``-s(alpha)`` with ``s(alpha) = (S + alpha I)^-1 r`` for a proximal
    shift ``alpha >= 0`` of the friction Schur complement ``S``. The contact
    sticks when ``phi(0) <= 0`` and otherwise slides at the root of
    ``phi(alpha) = |s| - c . s + offset``. The root is bracketed by doubling
    ``alpha`` and refined from the Newton step of the sticking solution. The
    Newton steps solve the equivalent ``1 / (c . s - offset) = 1 / |s|``, which
    is nearly linear in ``alpha`` where ``phi`` decays like ``1 / alpha``, and
    fall back to bisection when they leave the bracket. After a step that does
    not halve ``|phi|``, a Newton step also keeps ``newton_margin`` of the
    bracket width from its ends, so that the bracket contracts even for
    nonmonotone roots. A bracket that has shrunk to adjacent floats resolves
    the root to working precision, even when rounding keeps ``phi`` above the
    tolerance.

    Args:
        shifted_solve: ``wp.func`` returning ``(s, t)`` with ``s = (S + alpha I)^-1 rhs``
            and ``t = (S + alpha I)^-1 s`` for a system of type ``system_type``.
        system_type: Warp type describing ``S``.
        vector_type: Warp vector type of ``s``.
        iterations: Maximum number of Newton or bisection steps.
        tolerance: Root tolerance, relative to ``max(|s|, |offset|)`` so that it is invariant to the
            unit system and to the mass of the contacting bodies.
        newton_margin: Fraction of the bracket width that a Newton step keeps from its ends after a step
            that did not halve ``|phi|``.
    """

    @wp.func
    def newton_step(
        s: vector_type,
        t: vector_type,
        coupling: vector_type,
        s_norm: wp.float32,
        value: wp.float32,
        alpha_scale: wp.float32,
    ) -> wp.float32:
        """Return the Newton step of the normalized ``alpha`` toward the sliding root.

        Where the friction budget ``m = c . s - offset = |s| - phi`` is positive, the step solves
        ``1 / m - 1 / |s| = 0``, which is linear in ``alpha`` for a single Schur eigenvalue without
        coupling; otherwise it solves ``phi = 0``.
        """
        budget = s_norm - value
        coupling_rate = wp.dot(coupling, t)
        projection_rate = wp.dot(s, t)
        if budget > 0.0 and s_norm > 0.0:
            inverse_budget = 1.0 / budget
            inverse_norm = 1.0 / s_norm
            derivative = (
                coupling_rate * inverse_budget * inverse_budget
                - projection_rate * inverse_norm * inverse_norm * inverse_norm
            )
            return (inverse_norm - inverse_budget) / (alpha_scale * derivative)
        return -value / (alpha_scale * (coupling_rate - projection_rate / wp.max(s_norm, 1.0e-30)))

    @wp.func
    def solve_sliding_root(
        system: system_type,
        rhs: vector_type,
        coupling: vector_type,
        offset: wp.float32,
        alpha_scale: wp.float32,
    ):
        s, t = shifted_solve(system, rhs, 0.0)
        s_norm = wp.length(s)
        value = s_norm - wp.dot(coupling, s) + offset
        if value <= tolerance * wp.max(s_norm, wp.abs(offset)):
            return s, _SlidingRootStatus.SOLVED
        # Newton step from the sticking solution, the first iterate when it lands inside the bracket
        start = newton_step(s, t, coupling, s_norm, value, alpha_scale)
        start_value = value

        # Normalizing alpha by the Schur-complement scale keeps the fixed root
        # tolerances useful across contact blocks with different magnitudes.
        lower = wp.float32(0.0)
        upper = wp.float32(1.0)
        bracketed = bool(False)
        for _ in range(_ROOT_EXPANSION_STEPS):
            s, t = shifted_solve(system, rhs, upper * alpha_scale)
            value = wp.length(s) - wp.dot(coupling, s) + offset
            if wp.isfinite(value) and value <= 0.0:
                bracketed = True
                break
            if wp.isfinite(value):
                lower = upper
            upper *= 2.0
        if not bracketed:
            return s, _SlidingRootStatus.NOT_BRACKETED

        # The finite bracket rejects a non-finite Newton start or step
        alpha = 0.5 * (lower + upper)
        if start > lower and start < upper:
            alpha = start
        previous_value = start_value
        status = wp.int32(_SlidingRootStatus.NOT_CONVERGED)
        for _ in range(iterations):
            s, t = shifted_solve(system, rhs, alpha * alpha_scale)
            s_norm = wp.length(s)
            value = s_norm - wp.dot(coupling, s) + offset
            if wp.abs(value) <= tolerance * wp.max(s_norm, wp.abs(offset)):
                status = wp.int32(_SlidingRootStatus.SOLVED)
                break
            if value > 0.0:
                lower = alpha
            else:
                upper = alpha
            candidate = 0.5 * (lower + upper)
            if candidate <= lower or candidate >= upper:
                # No float lies strictly inside the bracket, so further steps cannot move alpha.
                status = wp.int32(_SlidingRootStatus.SOLVED)
                break
            newton = alpha + newton_step(s, t, coupling, s_norm, value, alpha_scale)
            if newton > lower and newton < upper:
                candidate = newton
                if wp.abs(value) > 0.5 * wp.abs(previous_value):
                    # The last step did not halve phi: keep the step away from the bracket ends
                    margin = newton_margin * (upper - lower)
                    candidate = wp.clamp(newton, lower + margin, upper - margin)
            previous_value = value
            alpha = candidate
        return s, status

    return solve_sliding_root


@wp.func
def _shifted_solve_mat22(schur: wp.vec3f, rhs: wp.vec2f, alpha: wp.float32):
    """Solve ``(S + alpha I) s = rhs`` and ``(S + alpha I) t = s`` for ``S = [[x, y], [y, z]]``."""
    a00 = schur[0] + alpha
    a01 = schur[1]
    a11 = schur[2] + alpha
    inverse_determinant = 1.0 / (a00 * a11 - a01 * a01)
    s = wp.vec2f(
        (a11 * rhs[0] - a01 * rhs[1]) * inverse_determinant, (a00 * rhs[1] - a01 * rhs[0]) * inverse_determinant
    )
    t = wp.vec2f((a11 * s[0] - a01 * s[1]) * inverse_determinant, (a00 * s[1] - a01 * s[0]) * inverse_determinant)
    return s, t


_solve_sliding_root_mat22 = _make_sliding_root_solver(
    _shifted_solve_mat22, wp.vec3f, wp.vec2f, iterations=12, tolerance=1.0e-7, newton_margin=0.0
)


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
    schur = wp.vec3f(
        tangent_delassus00 - normal_tangent[0] * normal_tangent[0] * inverse_normal_delassus,
        tangent_delassus01 - normal_tangent[0] * normal_tangent[1] * inverse_normal_delassus,
        tangent_delassus11 - normal_tangent[1] * normal_tangent[1] * inverse_normal_delassus,
    )
    tangent_rhs = tangent_rhs_raw - (normal_rhs * inverse_normal_delassus) * normal_tangent
    friction_over_normal = friction * inverse_normal_delassus
    alpha_scale = wp.max(wp.max(wp.abs(schur[0]), wp.abs(schur[1])), wp.max(wp.abs(schur[2]), 1.0e-20))
    # The status is ignored: with a negative normal right-hand side, only non-finite
    # inputs leave the root unbracketed, and their impulse fails the world through the
    # body twists; a root left unconverged by the iteration budget is still bracketed,
    # and the following projection sweeps refine it.
    s, _status = _solve_sliding_root_mat22(
        schur,
        tangent_rhs,
        friction_over_normal * normal_tangent,
        friction_over_normal * normal_rhs,
        alpha_scale,
    )
    tangent_reaction = -s
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
def compute_contact_delassus_scale(delassus: wp.mat33f) -> wp.float32:
    """Return the square root of the mean diagonal of a contact block, which scales the natural-map residual."""
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
    scale = compute_contact_delassus_scale(delassus)
    scaled_reaction = scale * reaction
    scaled_velocity = (1.0 / scale) * velocity
    modified_velocity = wp.vec3f(
        scaled_velocity[0],
        scaled_velocity[1],
        scaled_velocity[2] + friction * wp.length(wp.vec2f(scaled_velocity[0], scaled_velocity[1])),
    )
    projected = project_to_coulomb_cone(scaled_reaction - modified_velocity, friction)
    return scaled_reaction - projected


@wp.func
def spatial_friction_scaling(friction: wp.vec3f) -> vec6f:
    """Return the cone scaling ``(1, mu, mu, mu_spin, mu_roll, mu_roll)``."""
    return vec6f(1.0, friction[0], friction[0], friction[1], friction[2], friction[2])


@wp.func
def _uses_angular_friction(friction: wp.vec3f) -> wp.bool:
    return friction[1] != 0.0 or friction[2] != 0.0


@wp.func
def _scaled_coupling(delassus: mat66f, scaling: vec6f) -> vec5f:
    """Return the scaled normal coupling ``c_i = s_i D[i, 0]`` of the friction rows."""
    coupling = vec5f(0.0)
    for i in range(5):
        if scaling[i + 1] > 0.0:
            coupling[i] = scaling[i + 1] * delassus[i + 1, 0]
    return coupling


@wp.func
def prepare_spatial_contact(delassus: mat66f, friction: wp.vec3f):
    """Eigendecompose the scaled Schur complement of one spatial contact metric.

    Contacts without angular friction use the 3D Coulomb solver; their
    eigenvectors and eigenvalues are left at zero.
    The metric must be exactly symmetric, as :func:`spatial_contact_delassus`
    assembles it.

    Returns:
        The Schur eigenvectors (columns), eigenvalues, and whether the metric is valid.
    """
    eigenvectors = mat55f(0.0)
    eigenvalues = vec5f(0.0)
    scaling = spatial_friction_scaling(friction)
    if not wp.isfinite(friction) or wp.min(friction) < 0.0 or not wp.isfinite(delassus):
        return eigenvectors, eigenvalues, False
    a = delassus[0, 0]
    if a <= 0.0:
        return eigenvectors, eigenvalues, False

    if not _uses_angular_friction(friction):
        if friction[0] > 0.0:
            # The 3D solver uses unscaled friction coordinates. Test their
            # normalized Schur block to avoid mu**4 determinant underflow.
            s00 = delassus[1, 1] - (delassus[1, 0] / a) * delassus[0, 1]
            s01 = delassus[1, 2] - (delassus[1, 0] / a) * delassus[0, 2]
            s11 = delassus[2, 2] - (delassus[2, 0] / a) * delassus[0, 2]
            if s00 <= 0.0 or s11 <= 0.0:
                return eigenvectors, eigenvalues, False
            scale = wp.max(s00, wp.max(wp.abs(s01), s11))
            if (s00 / scale) * (s11 / scale) <= (s01 / scale) * (s01 / scale):
                return eigenvectors, eigenvalues, False
        return eigenvectors, eigenvalues, True

    coupling = _scaled_coupling(delassus, scaling)
    schur = mat55f(0.0)
    for i in range(5):
        for j in range(5):
            if scaling[i + 1] > 0.0 and scaling[j + 1] > 0.0:
                schur[i, j] = scaling[i + 1] * delassus[i + 1, j + 1] * scaling[j + 1] - coupling[i] * coupling[j] / a
    schur, spectral_scale = normalize_symmetric_matrix(schur, 5)
    if not wp.isfinite(spectral_scale) or spectral_scale <= 0.0:
        return eigenvectors, eigenvalues, False
    # Padding inactive rows by an uncoupled identity leaves the active solve
    # unchanged: the inactive right-hand side and coupling are identically zero.
    for i in range(5):
        if scaling[i + 1] == 0.0:
            schur[i, i] = 1.0
    values, vectors_rows = symmetric_eigenvalues_qr(schur, _QR_TOLERANCE)
    vectors = wp.transpose(vectors_rows)
    if not wp.isfinite(values) or not wp.isfinite(vectors) or wp.min(values) <= 0.0:
        return eigenvectors, eigenvalues, False
    # Accept the decomposition only if it reconstructs the Schur block with orthonormal vectors.
    reconstruction = vectors * wp.diag(values) * wp.transpose(vectors) - schur
    orthogonality = wp.transpose(vectors) * vectors - wp.identity(n=5, dtype=wp.float32)
    if (
        wp.ddot(reconstruction, reconstruction) > _DECOMPOSITION_TOLERANCE
        or wp.ddot(orthogonality, orthogonality) > _DECOMPOSITION_TOLERANCE
    ):
        return eigenvectors, eigenvalues, False
    return vectors, spectral_scale * values, True


@wp.func
def _shifted_solve_spectral(eigenvalues: vec5f, rhs: vec5f, alpha: wp.float32):
    """Solve the shifted Schur systems in its eigenbasis, where they are diagonal."""
    s = vec5f(0.0)
    t = vec5f(0.0)
    for i in range(5):
        inverse = 1.0 / (eigenvalues[i] + alpha)
        s[i] = rhs[i] * inverse
        t[i] = s[i] * inverse
    return s, t


# Keep a strict contraction even for nonmonotone roots of the coupled cone.
_solve_sliding_root_spectral = _make_sliding_root_solver(
    _shifted_solve_spectral, vec5f, vec5f, iterations=64, tolerance=2.0e-7, newton_margin=0.05
)


@wp.func
def _recover_reaction(
    eigenvectors: mat55f,
    spectral_coupling: vec5f,
    scaling: vec6f,
    normal_delassus: float,
    spectral_s: vec5f,
    normal_rhs: float,
) -> vec6f:
    s = eigenvectors * spectral_s
    reaction = vec6f(0.0)
    reaction[0] = (wp.dot(spectral_coupling, spectral_s) - normal_rhs) / normal_delassus
    for i in range(5):
        reaction[i + 1] = -scaling[i + 1] * s[i]
    return reaction


@wp.func
def _solve_linear_contact(delassus: mat66f, friction: float, free_velocity: vec6f) -> vec6f:
    """Solve a contact without angular friction with the 3D Coulomb solver.

    Its root tolerances are relative to the metric scale.
    """
    r = _solve_contact_coulomb_newton_components(
        delassus[0, 0],
        wp.vec2f(delassus[1, 0], delassus[2, 0]),
        delassus[1, 1],
        delassus[1, 2],
        delassus[2, 2],
        free_velocity[0],
        wp.vec2f(free_velocity[1], free_velocity[2]),
        friction,
    )
    return vec6f(r[2], r[0], r[1], 0.0, 0.0, 0.0)


@wp.func
def solve_spatial_contact(
    delassus: mat66f,
    eigenvectors: mat55f,
    eigenvalues: vec5f,
    friction: wp.vec3f,
    free_velocity: vec6f,
):
    """Solve the unified cone with a safeguarded scalar Schur-complement root.

    A non-finite reaction reaches the body twists, which fail the world.

    Returns:
        The normal-first reaction and a :class:`SpatialContactStatus`.
    """
    reaction = vec6f(0.0)
    normal_rhs = free_velocity[0]
    if normal_rhs >= 0.0:
        return reaction, SpatialContactStatus.SOLVED
    if not _uses_angular_friction(friction):
        return _solve_linear_contact(delassus, friction[0], free_velocity), SpatialContactStatus.SOLVED

    scaling = spatial_friction_scaling(friction)
    normal_delassus = delassus[0, 0]
    inverse_normal_delassus = 1.0 / normal_delassus
    normal_offset = normal_rhs * inverse_normal_delassus
    coupling = _scaled_coupling(delassus, scaling)
    spectral_coupling = wp.transpose(eigenvectors) * coupling
    rhs = vec5f(0.0)
    for i in range(5):
        rhs[i] = scaling[i + 1] * free_velocity[i + 1]
    rhs = wp.transpose(eigenvectors) * (rhs - normal_offset * coupling)
    s, status = _solve_sliding_root_spectral(
        eigenvalues, rhs, inverse_normal_delassus * spectral_coupling, normal_offset, wp.max(eigenvalues)
    )
    if status == SpatialContactStatus.NOT_BRACKETED:
        return reaction, status
    return _recover_reaction(eigenvectors, spectral_coupling, scaling, normal_delassus, s, normal_rhs), status


@wp.func
def spatial_contact_velocity_scale(delassus: mat66f, friction: wp.vec3f) -> wp.float32:
    """Return the metric velocity scale ``sqrt(mean scaled diagonal)`` of the active rows."""
    scaling = spatial_friction_scaling(friction)
    trace = delassus[0, 0]
    active_count = float(1.0)
    for i in range(1, 6):
        if scaling[i] > 0.0:
            trace += scaling[i] * scaling[i] * delassus[i, i]
            active_count += 1.0
    return wp.sqrt(wp.max(trace / active_count, 1.0e-30))


@wp.func
def compute_spatial_contact_residual(delassus: mat66f, friction: wp.vec3f, reaction: vec6f, velocity: vec6f) -> vec6f:
    """Return the metric-scaled canonical Alart--Curnier natural-map residual.

    ``velocity`` includes the compliant velocity and selected normal target.
    Disabled components report their reaction, which must remain zero.
    """
    scaling = spatial_friction_scaling(friction)
    rho = vec6f(0.0)
    gamma = vec6f(0.0)
    for i in range(6):
        if scaling[i] > 0.0:
            rho[i] = reaction[i] / scaling[i]
            gamma[i] = velocity[i] * scaling[i]
    scale = spatial_contact_velocity_scale(delassus, friction)
    rho *= scale
    gamma *= 1.0 / scale
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
        tangent_scale = projected[0] / tangent_norm
        for i in range(5):
            projected[i + 1] = tangent_scale * tangent[i]
    residual = rho - projected
    for i in range(6):
        if scaling[i] == 0.0:
            residual[i] = reaction[i]
    return residual


@wp.func
def pack_contact_reaction(linear: wp.vec3f, angular: wp.vec3f) -> vec6f:
    """Pack a normal-last linear and an angular reaction into normal-first order."""
    return vec6f(linear[2], linear[0], linear[1], angular[0], angular[1], angular[2])


@wp.func
def pack_spatial_friction(
    cid: wp.int32, friction: wp.array[wp.float32], angular_friction: wp.array[wp.vec2f]
) -> wp.vec3f:
    """Return the sliding, spinning, and rolling friction coefficients ``(mu, mu_spin, mu_roll)`` of a contact."""
    angular = angular_friction[cid]
    return wp.vec3f(friction[cid], angular[0], angular[1])


@wp.func
def linear_contact_reaction(value: vec6f) -> wp.vec3f:
    return wp.vec3f(value[1], value[2], value[0])


@wp.func
def angular_contact_reaction(value: vec6f) -> wp.vec3f:
    return wp.vec3f(value[3], value[4], value[5])


@wp.func
def _spatial_contact_jacobian(linear: mat36f, frame: wp.mat33f, sign: wp.float32) -> mat66f:
    """Return the normal-first 6D Jacobian of one contact body.

    The angular rows are ``sign * (normal, tangent_0, tangent_1)`` acting on
    the angular twist, with ``sign = -1`` for body A.
    """
    jacobian = mat66f(0.0)
    for row in range(3):
        # Normal-first row ``row`` is the normal-last row ``(row + 2) % 3``.
        axis = (row + 2) % 3
        for col in range(6):
            jacobian[row, col] = linear[axis, col]
        for col in range(3):
            jacobian[row + 3, col + 3] = sign * frame[col, axis]
    return jacobian


@wp.func
def _weighted_gram(jacobian: mat66f, inverse_weight: mat66f) -> mat66f:
    """Return ``J W^-1 J^T`` of one body, evaluating each pair of rows once so that it is exactly symmetric."""
    gram = mat66f(0.0)
    for row in range(6):
        weighted = apply_body_weight(inverse_weight, jacobian[row])
        for col in range(6):
            if col >= row:
                value = wp.dot(weighted, jacobian[col])
                gram[row, col] = value
                gram[col, row] = value
    return gram


@wp.func
def spatial_body_velocity(linear: mat36f, frame: wp.mat33f, sign: wp.float32, twist: vec6f) -> vec6f:
    """Return ``J v`` of one contact body in normal-first order (see :func:`_spatial_contact_jacobian`)."""
    linear_velocity = linear @ twist
    angular = wp.transpose(frame) @ wp.vec3f(twist[3], twist[4], twist[5])
    return vec6f(
        linear_velocity[2],
        linear_velocity[0],
        linear_velocity[1],
        sign * angular[2],
        sign * angular[0],
        sign * angular[1],
    )


@wp.func
def spatial_body_wrench(linear: mat36f, frame: wp.mat33f, sign: wp.float32, reaction: vec6f) -> vec6f:
    """Return ``J^T lambda`` of one contact body for a normal-first reaction."""
    wrench = wp.transpose(linear) @ wp.vec3f(reaction[1], reaction[2], reaction[0])
    moment = sign * (frame @ wp.vec3f(reaction[4], reaction[5], reaction[3]))
    wrench[3] += moment[0]
    wrench[4] += moment[1]
    wrench[5] += moment[2]
    return wrench


@wp.func
def scatter_spatial_impulse(
    cid: wp.int32,
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    jacobian_a: wp.array[mat36f],
    jacobian_b: wp.array[mat36f],
    frame: wp.array[wp.mat33f],
    impulse: vec6f,
    inverse_weight: wp.array[mat66f],
    twist_delta: wp.array[vec6f],
):
    """Accumulate the twist change ``W^-1 J^T impulse`` of one normal-first spatial impulse."""
    if wp.length_sq(impulse) == 0.0:
        return
    bid_a = body_a[cid]
    bid_b = body_b[cid]
    contact_frame = frame[cid]
    if bid_a >= 0:
        wrench = spatial_body_wrench(jacobian_a[cid], contact_frame, -1.0, impulse)
        wp.atomic_add(twist_delta, bid_a, apply_body_weight(inverse_weight[bid_a], wrench))
    if bid_b >= 0:
        wrench = spatial_body_wrench(jacobian_b[cid], contact_frame, 1.0, impulse)
        wp.atomic_add(twist_delta, bid_b, apply_body_weight(inverse_weight[bid_b], wrench))


@wp.func
def spatial_contact_velocity(
    cid: wp.int32,
    bid_a: wp.int32,
    bid_b: wp.int32,
    jacobian_a: wp.array[mat36f],
    jacobian_b: wp.array[mat36f],
    frame: wp.array[wp.mat33f],
    twist: wp.array[vec6f],
) -> vec6f:
    velocity = vec6f(0.0)
    contact_frame = frame[cid]
    if bid_a >= 0:
        velocity += spatial_body_velocity(jacobian_a[cid], contact_frame, -1.0, twist[bid_a])
    if bid_b >= 0:
        velocity += spatial_body_velocity(jacobian_b[cid], contact_frame, 1.0, twist[bid_b])
    return velocity


@wp.func
def spatial_contact_delassus(
    cid: wp.int32,
    bid_a: wp.int32,
    bid_b: wp.int32,
    jacobian_a: wp.array[mat36f],
    jacobian_b: wp.array[mat36f],
    frame: wp.array[wp.mat33f],
    inverse_weight_a: mat66f,
    inverse_weight_b: mat66f,
) -> mat66f:
    """Return the symmetric ``J W^-1 J^T`` of one contact in normal-first order, without compliance."""
    delassus = mat66f(0.0)
    contact_frame = frame[cid]
    if bid_a >= 0:
        delassus += _weighted_gram(_spatial_contact_jacobian(jacobian_a[cid], contact_frame, -1.0), inverse_weight_a)
    if bid_b >= 0:
        delassus += _weighted_gram(_spatial_contact_jacobian(jacobian_b[cid], contact_frame, 1.0), inverse_weight_b)
    return delassus


@wp.func
def clamp_contact_reaction(value: vec6f, friction: wp.vec3f) -> vec6f:
    """Scale the friction rows back into the cone while preserving the normal impulse."""
    scale = spatial_friction_scaling(friction)
    result = value
    result[0] = wp.max(value[0], 0.0)
    norm_sq = float(0.0)
    for axis in range(1, 6):
        if scale[axis] > 0.0:
            scaled = value[axis] / scale[axis]
            norm_sq += scaled * scaled
        else:
            result[axis] = 0.0
    factor = wp.min(1.0, result[0] / wp.max(wp.sqrt(norm_sq), 1.0e-30))
    for axis in range(1, 6):
        result[axis] *= factor
    return result
