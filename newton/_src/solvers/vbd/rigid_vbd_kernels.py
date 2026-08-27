# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Rigid body VBD solver kernels and utilities.

This module contains all rigid body-specific kernels, device functions, data structures,
and constants for the VBD solver's rigid body domain.

Every constraint row runs one of two paths, selected by the ``use_compliant_alm``
flag threaded through the kernels: the legacy AVBD ramping penalty, or compliant
ALM, which pairs a finite material stiffness with an augmented-Lagrangian metric.
The two paths share their geometry and differ only in how a row turns a
constraint violation into a force and how its dual state advances.

Organization, in file order:
- Constants: Solver parameters and thresholds
- Compliant ALM coefficient algebra: Row coefficients and automatic rho policies
- Helper classes and device functions: Rigid body dynamics, constraint rows
- Data structures: RigidForceElementAdjacencyInfo and related structs
- Utility kernels: Adjacency building
- Pre-iteration kernels: Forward integration, contact history restore, rho seeding,
  Dahl parameter computation
- Iteration kernels: Contact accumulation, rigid body solve, dual updates
- Post-iteration kernels: Velocity updates, Dahl state updates
"""

from typing import Any

import warp as wp

from newton._src.core.types import MAXVAL
from newton._src.math import orthonormal_basis, quat_velocity
from newton._src.sim import JointType
from newton._src.sim.contacts import contact_surface_point, contact_surface_separation
from newton._src.solvers.solver import integrate_rigid_body

wp.set_module_options({"enable_backward": False})

# ---------------------------------
# Constants
# ---------------------------------

_SMALL_ANGLE_EPS = wp.constant(1.0e-7)
"""Small-angle threshold [rad] for guards and series expansions"""

_DRIVE_LIMIT_MODE_NONE = wp.constant(0)
_DRIVE_LIMIT_MODE_LIMIT_LOWER = wp.constant(1)
_DRIVE_LIMIT_MODE_LIMIT_UPPER = wp.constant(2)
_DRIVE_LIMIT_MODE_DRIVE = wp.constant(3)

_SMALL_LENGTH_EPS = wp.constant(1.0e-9)
"""Small length tolerance (e.g., segment length checks)"""

_USE_SMALL_ANGLE_APPROX = wp.constant(True)
"""If True use first-order small-angle rotation approximation; if False use closed-form rotation update"""

_DAHL_KAPPADOT_DEADBAND = wp.constant(1.0e-6)
"""Deadband threshold for hysteresis direction selection"""

_NUM_CONTACT_THREADS_PER_BODY = wp.constant(4)
"""Threads per body for contact accumulation using strided iteration"""

# DER bend-twist strain measure tolerances (curvature binormal + Bishop transport).
_ROD_KB_FOLD_EPS = wp.constant(1.0e-12)
"""Degenerate-fold scale and denominator floor for the DER curvature binormal.

When both 1 + dot(t0, t1) and |cross(t0, t1)|^2 fall below this value, the
tangents are treated as a true fold with a chosen perpendicular direction. This
is only a divide-by-zero guard; magnitude is bounded by _ROD_KB_CURVATURE_CAP."""

_ROD_KB_CURVATURE_CAP = wp.constant(20.0)
"""Numerical cap for near-fold DER curvature-binormal magnitude.

For |kb| = 2*tan(theta/2), this starts near theta ~= 168.6 deg; it is a
conditioning guard, not a material parameter."""

_ROD_TRANSPORT_DENOM_EPS = wp.constant(1.0e-8)
"""Near-anti-parallel threshold for switching closed-form transport to Bishop.

This is larger than _ROD_KB_FOLD_EPS because transport has no curvature cap;
the closed-form expression must be left before it becomes ill-conditioned."""

_ROD_TWIST_JACOBIAN_DIRECTIONAL_DENOM = wp.constant(2.0e-2)
"""Near-fold threshold for evaluating the twist Jacobian through directional derivatives.

The tangent-bisector row is algebraically exact, but loses float32 consistency
with the normalized transport residual as 1 + dot(t0, t1) approaches zero. This
cutoff is approximately where the curvature-binormal cap starts to engage."""

_ROD_TWIST_ATAN2_DENOM_EPS = wp.constant(1.0e-12)
"""Floor on sin^2 + cos^2 in the transported-twist atan2 derivative."""

_COMPLIANT_ALM_BILATERAL_MIN_MATERIAL_FRACTION = 0.9
"""Smallest local ``k_eff/K`` a bilateral row may realize within one sweep."""

_COMPLIANT_ALM_BILATERAL_MIN_RHO_OVER_K = wp.constant(
    _COMPLIANT_ALM_BILATERAL_MIN_MATERIAL_FRACTION / (1.0 - _COMPLIANT_ALM_BILATERAL_MIN_MATERIAL_FRACTION)
)
"""Floor on ``rho/K`` realizing that fraction, inverted from ``k_eff = K*rho/(K+rho)``.

Applied to bilateral structural and drive rows.
"""

# ---------------------------------
# Compliant ALM coefficient algebra
# ---------------------------------


@wp.func
def _compliant_alm_coefficients(material_k: float, rho: float):
    """Return stable ``(s, k_eff, a)`` coefficients for compliant ALM.

    ``s=K/(K+rho)``, ``a=rho/(K+rho)``, ``k_eff=K*a``, branched to preserve
    whichever of ``s`` and ``a`` is small. A nonpositive input retires the row:
    ``a=1``, not the ``rho -> 0`` limit, lets the ascent clear its dual too.
    """
    if material_k <= 0.0 or rho <= 0.0:
        return 0.0, 0.0, 1.0

    if material_k >= rho:
        r = rho / material_k
        s = 1.0 / (1.0 + r)
        return s, rho * s, r * s

    r = material_k / rho
    a = 1.0 / (1.0 + r)
    return r * a, material_k * a, a


@wp.func
def _alm_relaxed_ascent(lam: Any, R: Any, material_k: float, rho: float) -> Any:
    """Advance one compliant-ALM dual: ``lam <- s*(lam + rho*R)``.

    Both branches evaluate that expression without forming ``rho*R``, which
    can overflow at large rho. The increment form preserves small ``a`` when
    ``rho <= K``; the distributed form preserves small ``s`` otherwise.
    Instantiated for scalar and ``wp.vec3`` duals.
    """
    s, k_eff, a = _compliant_alm_coefficients(material_k, rho)
    if material_k >= rho:
        return lam + (k_eff * R - a * lam)
    return s * lam + k_eff * R


@wp.func
def _material_force_terms(rho: float, material_k: float, dual: Any, use_compliant_alm: int):
    """Return the ``(k, lam_eff)`` pair of a row force ``k*C + lam_eff``.

    Compliant ALM gives ``(k_eff, s*dual)``, legacy ``(rho, dual)``. Coulomb
    tangents bypass this: their dual is not scaled by ``s``.
    """
    if use_compliant_alm == 1:
        s, k_eff, _a = _compliant_alm_coefficients(material_k, rho)
        return k_eff, dual * s

    return rho, dual


# ---------------------------------
# Automatic rho policies
# ---------------------------------
#
# Each policy maps inverse-Delassus row support (scaled by 1/dt^2), plus any
# row-specific material or structural inputs, to an ALM rho. Joint and contact
# supports are computed by step_joint_C0_lambda_rho and
# step_body_body_contact_C0_lambda. Separate wrappers keep each row class's
# policy explicit even when formulas currently coincide.


@wp.func
def _bilateral_auto_rho(row_support: float, material_k: float):
    """Return ``max(H, 9K)``, ensuring ``k_eff >= 0.9K`` in each local solve."""
    if material_k <= 0.0:
        return 0.0
    return wp.max(row_support, _COMPLIANT_ALM_BILATERAL_MIN_RHO_OVER_K * material_k)


@wp.func
def _structural_joint_auto_rho(joint_support: float, material_k: float):
    """Bilateral auto-rho for a structural joint row."""
    return _bilateral_auto_rho(joint_support, material_k)


@wp.func
def _drive_auto_rho(axis_support: float, material_k: float):
    """Bilateral auto-rho for one drive axis."""
    return _bilateral_auto_rho(axis_support, material_k)


@wp.func
def _limit_auto_rho(axis_support: float, material_k: float):
    """Return support as limit rho, or zero if the row is off."""
    if axis_support <= 0.0 or material_k <= 0.0:
        return 0.0
    return axis_support


@wp.func
def _contact_auto_normal_rho(
    normal_support: float,
    material_k: float,
):
    """Return support as contact-normal rho, or zero if the row is off."""
    if normal_support <= 0.0 or material_k <= 0.0:
        return 0.0
    return normal_support


@wp.func
def _contact_auto_tangent_rho(
    tangent_support: float,
    normal_rho: float,
    structural_support: float,
):
    """Use tangent support, strengthened by pair structure up to the normal rho."""
    return wp.max(tangent_support, wp.min(normal_rho, structural_support))


# ---------------------------------
# Helper classes and device functions
# ---------------------------------


@wp.struct
class RigidContactHistory:
    lambda_: wp.array[wp.vec3]
    penalty_k: wp.array[float]
    normal: wp.array[wp.vec3]


@wp.func
def _world_selected(world: int, mask: wp.array[wp.bool]):
    """Query an internal world mask whose final entry selects global entities."""
    mask_index = world
    if world < 0:
        mask_index = mask.shape[0] - 1
    return mask[mask_index]


@wp.func
def _reset_world_selected(
    world: int,
    world_mask: wp.array[wp.bool],
    reset_all: bool,
    world_count: int,
):
    """Query a public reset mask whose final entry selects global entities."""
    if reset_all:
        return True
    if world < 0:
        world = world_count
    return world_mask[world]


@wp.func
def _shape_world_selected(
    shape: int,
    shape_world: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    mask: wp.array[wp.bool],
):
    """Return whether a shape's world (its own, or its attached body's) is selected by ``mask``."""
    if _world_selected(shape_world[shape], mask):
        return True
    body = shape_body[shape]
    return body >= 0 and _world_selected(body_world[body], mask)


@wp.func
def _contact_world_selected(
    shape0: int,
    shape1: int,
    shape_world: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    mask: wp.array[wp.bool],
):
    """Return whether either contact endpoint's world is selected by ``mask``."""
    if _shape_world_selected(shape0, shape_world, shape_body, body_world, mask):
        return True
    return _shape_world_selected(shape1, shape_world, shape_body, body_world, mask)


@wp.func
def ldlt6_solve(h_ll: wp.mat33, h_aa: wp.mat33, h_al: wp.mat33, rhs_lin: wp.vec3, rhs_ang: wp.vec3):
    """Solve the 6x6 SPD block system via direct LDL^T factorization.

    Returns (x_lin, x_ang).
    """
    A11 = h_ll[0, 0]
    A21 = h_ll[1, 0]
    A22 = h_ll[1, 1]
    A31 = h_ll[2, 0]
    A32 = h_ll[2, 1]
    A33 = h_ll[2, 2]
    A41 = h_al[0, 0]
    A42 = h_al[0, 1]
    A43 = h_al[0, 2]
    A44 = h_aa[0, 0]
    A51 = h_al[1, 0]
    A52 = h_al[1, 1]
    A53 = h_al[1, 2]
    A54 = h_aa[1, 0]
    A55 = h_aa[1, 1]
    A61 = h_al[2, 0]
    A62 = h_al[2, 1]
    A63 = h_al[2, 2]
    A64 = h_aa[2, 0]
    A65 = h_aa[2, 1]
    A66 = h_aa[2, 2]

    # LDL^T decomposition
    L21 = A21 / A11
    L31 = A31 / A11
    L41 = A41 / A11
    L51 = A51 / A11
    L61 = A61 / A11

    D2 = A22 - L21 * L21 * A11

    L32 = (A32 - L21 * L31 * A11) / D2
    L42 = (A42 - L21 * L41 * A11) / D2
    L52 = (A52 - L21 * L51 * A11) / D2
    L62 = (A62 - L21 * L61 * A11) / D2

    D3 = A33 - (L31 * L31 * A11 + L32 * L32 * D2)

    L43 = (A43 - L31 * L41 * A11 - L32 * L42 * D2) / D3
    L53 = (A53 - L31 * L51 * A11 - L32 * L52 * D2) / D3
    L63 = (A63 - L31 * L61 * A11 - L32 * L62 * D2) / D3

    D4 = A44 - (L41 * L41 * A11 + L42 * L42 * D2 + L43 * L43 * D3)

    L54 = (A54 - L41 * L51 * A11 - L42 * L52 * D2 - L43 * L53 * D3) / D4
    L64 = (A64 - L41 * L61 * A11 - L42 * L62 * D2 - L43 * L63 * D3) / D4

    D5 = A55 - (L51 * L51 * A11 + L52 * L52 * D2 + L53 * L53 * D3 + L54 * L54 * D4)

    L65 = (A65 - L51 * L61 * A11 - L52 * L62 * D2 - L53 * L63 * D3 - L54 * L64 * D4) / D5

    D6 = A66 - (L61 * L61 * A11 + L62 * L62 * D2 + L63 * L63 * D3 + L64 * L64 * D4 + L65 * L65 * D5)

    # Forward substitution: L y = b
    y1 = rhs_lin[0]
    y2 = rhs_lin[1] - L21 * y1
    y3 = rhs_lin[2] - L31 * y1 - L32 * y2
    y4 = rhs_ang[0] - L41 * y1 - L42 * y2 - L43 * y3
    y5 = rhs_ang[1] - L51 * y1 - L52 * y2 - L53 * y3 - L54 * y4
    y6 = rhs_ang[2] - L61 * y1 - L62 * y2 - L63 * y3 - L64 * y4 - L65 * y5

    # Diagonal solve: D z = y
    z1 = y1 / A11
    z2 = y2 / D2
    z3 = y3 / D3
    z4 = y4 / D4
    z5 = y5 / D5
    z6 = y6 / D6

    # Back-substitution: L^T x = z
    x6 = z6
    x5 = z5 - L65 * x6
    x4 = z4 - L54 * x5 - L64 * x6
    x3 = z3 - L43 * x4 - L53 * x5 - L63 * x6
    x2 = z2 - L32 * x3 - L42 * x4 - L52 * x5 - L62 * x6
    x1 = z1 - L21 * x2 - L31 * x3 - L41 * x4 - L51 * x5 - L61 * x6

    return wp.vec3(x1, x2, x3), wp.vec3(x4, x5, x6)


@wp.func
def compute_kappa(q_wp: wp.quat, q_wc: wp.quat, q_wp_rest: wp.quat, q_wc_rest: wp.quat) -> wp.vec3:
    """Compute rest-relative angular rotation vector kappa in the parent frame.

    Kappa is the rotation vector (theta*axis) from the rest-aligned relative rotation.

    Args:
        q_wp: Parent orientation (world).
        q_wc: Child orientation (world).
        q_wp_rest: Parent rest orientation (world).
        q_wc_rest: Child rest orientation (world).

    Returns:
        wp.vec3: Rotation vector kappa in parent frame.
    """
    # Build R_align = R_rel * R_rel_rest^T using quaternions
    q_rel = wp.quat_inverse(q_wp) * q_wc
    q_rel_rest = wp.quat_inverse(q_wp_rest) * q_wc_rest
    q_align = q_rel * wp.quat_inverse(q_rel_rest)

    # Enforce shortest path (w > 0) to avoid double-cover ambiguity
    if q_align[3] < 0.0:
        q_align = wp.quat(-q_align[0], -q_align[1], -q_align[2], -q_align[3])

    # Log map to rotation vector
    axis, angle = wp.quat_to_axis_angle(q_align)
    return axis * angle


@wp.func
def _quat_rotate_local_z(q: wp.quat) -> wp.vec3:
    """Rotate local +Z by a unit quaternion; the third rotation-matrix column."""
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    return wp.vec3(2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y))


@wp.func
def _quat_rotate_local_x(q: wp.quat) -> wp.vec3:
    """Rotate local +X by a unit quaternion; the first rotation-matrix column."""
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    return wp.vec3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w))


@wp.func
def _normalize_with_fallback(v: wp.vec3, fallback: wp.vec3) -> wp.vec3:
    """Normalize v, then fallback, with a final fixed axis if both are tiny."""
    v_len = wp.length(v)
    if v_len > _SMALL_LENGTH_EPS:
        return v / v_len
    fb_len = wp.length(fallback)
    if fb_len > _SMALL_LENGTH_EPS:
        return fallback / fb_len
    return wp.vec3(1.0, 0.0, 0.0)


@wp.func
def _project_perp(v: wp.vec3, axis: wp.vec3) -> wp.vec3:
    """Project v onto the plane orthogonal to an already-unit axis."""
    return v - axis * wp.dot(v, axis)


@wp.func
def _perpendicular_axis_with_fallback(axis: wp.vec3, preferred: wp.vec3) -> wp.vec3:
    """Return a unit vector perpendicular to axis, preferring preferred when possible."""
    axis = _normalize_with_fallback(axis, wp.vec3(0.0, 0.0, 1.0))
    perp = _project_perp(preferred, axis)
    perp_len = wp.length(perp)
    if perp_len > _SMALL_LENGTH_EPS:
        return perp / perp_len

    alt = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(axis[0]) > 0.9:
        alt = wp.vec3(0.0, 1.0, 0.0)
    return _normalize_with_fallback(_project_perp(alt, axis), wp.vec3(0.0, 0.0, 1.0))


@wp.func
def _bishop_transport_quat(t0: wp.vec3, t1: wp.vec3, fallback_axis: wp.vec3) -> wp.quat:
    """Minimal no-twist rotation that transports unit tangent t0 to unit tangent t1."""
    c = wp.clamp(wp.dot(t0, t1), -1.0, 1.0)
    v = wp.cross(t0, t1)
    s = wp.length(v)
    if s > _SMALL_ANGLE_EPS:
        return wp.quat_from_axis_angle(v / s, wp.atan2(s, c))
    if c > 0.0:
        return wp.quat_identity()

    axis = _perpendicular_axis_with_fallback(t0, fallback_axis)
    return wp.quat_from_axis_angle(axis, wp.pi)


@wp.func
def _finite_curvature_binormal(t0: wp.vec3, t1: wp.vec3, fallback_axis: wp.vec3) -> wp.vec3:
    """Capped DER finite curvature binormal with a true-fold fallback."""
    tangent_dot = wp.clamp(wp.dot(t0, t1), -1.0, 1.0)
    tangent_cross = wp.cross(t0, t1)
    denom = 1.0 + tangent_dot
    cross_sq = wp.dot(tangent_cross, tangent_cross)

    # Use exact DER while cross(t0, t1) gives a direction. At an exact fold the
    # cross product vanishes, so choose a stable perpendicular cap direction.
    if denom <= _ROD_KB_FOLD_EPS and cross_sq <= _ROD_KB_FOLD_EPS:
        axis = _perpendicular_axis_with_fallback(t0, fallback_axis)
        return _ROD_KB_CURVATURE_CAP * axis

    kb = (2.0 / wp.max(_ROD_KB_FOLD_EPS, denom)) * tangent_cross

    kb_len = wp.length(kb)
    if kb_len > _ROD_KB_CURVATURE_CAP:
        kb = (_ROD_KB_CURVATURE_CAP / kb_len) * kb
    return kb


@wp.func
def _finite_curvature_binormal_derivative(
    t0: wp.vec3,
    t1: wp.vec3,
    dt0: wp.vec3,
    dt1: wp.vec3,
) -> wp.vec3:
    """Directional derivative of kb(t0, t1) for tangent perturbations dt0, dt1.

    Pure tangent-space math: it does not know where the perturbations come from.
    The caller forms each one from a world rotation as ``dti = omega_world x ti``
    to build rigid-body angular Jacobian columns.
    """
    tangent_dot = wp.clamp(wp.dot(t0, t1), -1.0, 1.0)
    tangent_cross = wp.cross(t0, t1)
    denom = 1.0 + tangent_dot
    cross_sq = wp.dot(tangent_cross, tangent_cross)
    ddenom = wp.dot(dt0, t1) + wp.dot(t0, dt1)
    dcross = wp.cross(dt0, t1) + wp.cross(t0, dt1)

    if denom <= _ROD_KB_FOLD_EPS and cross_sq <= _ROD_KB_FOLD_EPS:
        # The exact-fold direction is chosen, not geometric, so it has no
        # meaningful derivative; return zero to keep the model bounded.
        return wp.vec3(0.0)

    denom_safe = wp.max(_ROD_KB_FOLD_EPS, denom)
    inv_denom = 1.0 / denom_safe
    kb_raw = (2.0 * inv_denom) * tangent_cross
    dkb_raw = (2.0 * inv_denom) * dcross
    if denom > _ROD_KB_FOLD_EPS:
        dkb_raw = dkb_raw - (2.0 * ddenom * inv_denom * inv_denom) * tangent_cross

    kb_len = wp.length(kb_raw)
    if kb_len > _ROD_KB_CURVATURE_CAP:
        inv_len = 1.0 / kb_len
        scale = _ROD_KB_CURVATURE_CAP * inv_len
        dkb_raw = scale * (dkb_raw - kb_raw * (wp.dot(kb_raw, dkb_raw) * inv_len * inv_len))

    return dkb_raw


@wp.func
def _transport_material_axis(t0: wp.vec3, t1: wp.vec3, m0: wp.vec3, fallback_axis: wp.vec3) -> wp.vec3:
    """Parallel-transport a material normal from tangent t0 to tangent t1."""
    c = wp.clamp(wp.dot(t0, t1), -1.0, 1.0)
    denom = 1.0 + c
    # Closed-form minimal rotation t0 -> t1; assumes m0 perpendicular to t0.
    if denom > _ROD_TRANSPORT_DENOM_EPS:
        w = t0 + t1
        transported = m0 - (wp.dot(m0, t1) / denom) * w
        return _normalize_with_fallback(transported, fallback_axis)

    # Anti-parallel tangents (~180 deg): use a Bishop transport quaternion.
    B = _bishop_transport_quat(t0, t1, fallback_axis)
    return _normalize_with_fallback(wp.quat_rotate(B, m0), fallback_axis)


@wp.func
def _transported_twist_angle_from_material_axes(
    t0: wp.vec3,
    t1: wp.vec3,
    m0: wp.vec3,
    m1: wp.vec3,
    fallback_axis: wp.vec3,
) -> float:
    """Signed twist from transported parent material normal to child material normal."""
    m0_transport = _transport_material_axis(t0, t1, m0, fallback_axis)
    sin_theta = wp.dot(t1, wp.cross(m0_transport, m1))
    cos_theta = wp.dot(m0_transport, m1)
    return wp.atan2(sin_theta, cos_theta)


@wp.struct
class RodBendTwistMeasure:
    """Live bend/twist geometry for one rod joint.

    Measured once per force/Hessian evaluation and reused by the residual and
    analytic Jacobian columns.
    """

    t0: wp.vec3
    t1: wp.vec3
    m0: wp.vec3
    m1: wp.vec3
    kb_world: wp.vec3
    twist: float


@wp.func
def _measure_rod_bend_twist_z(q_wp: wp.quat, q_wc: wp.quat) -> RodBendTwistMeasure:
    """Measure bend/twist for SolverVBD rod joints whose material tangent is local +Z.

    The fixed rod material basis is local ``+X, +Y, +Z``.
    SolverVBD keeps body rotations normalized, so rotated basis axes are already
    orthonormal.
    """
    t0 = _quat_rotate_local_z(q_wp)
    t1 = _quat_rotate_local_z(q_wc)
    m0 = _quat_rotate_local_x(q_wp)
    m1 = _quat_rotate_local_x(q_wc)

    # DER-style split: bend comes from the finite curvature binormal of the two
    # tangents; twist is material spin after no-twist/Bishop transport.
    measure = RodBendTwistMeasure()
    measure.t0 = t0
    measure.t1 = t1
    measure.m0 = m0
    measure.m1 = m1
    measure.twist = _transported_twist_angle_from_material_axes(t0, t1, m0, m1, m0)
    measure.kb_world = _finite_curvature_binormal(t0, t1, m0)
    return measure


@wp.func
def _transported_twist_angle_derivative_from_measure(
    measure: RodBendTwistMeasure,
    omega_world: wp.vec3,
    is_parent: bool,
) -> float:
    """Directional derivative of transported twist for one endpoint rotation.

    Linear in the infinitesimal world-space rotation ``omega_world``; frames
    are read from ``measure``.
    """
    t0 = measure.t0
    t1 = measure.t1
    m0 = measure.m0
    m1 = measure.m1

    dt0 = wp.vec3(0.0)
    dt1 = wp.vec3(0.0)
    dm0 = wp.vec3(0.0)
    dm1 = wp.vec3(0.0)
    if is_parent:
        dt0 = wp.cross(omega_world, t0)
        dm0 = wp.cross(omega_world, m0)
    else:
        dt1 = wp.cross(omega_world, t1)
        dm1 = wp.cross(omega_world, m1)

    c = wp.clamp(wp.dot(t0, t1), -1.0, 1.0)
    denom = 1.0 + c
    if denom <= _ROD_TRANSPORT_DENOM_EPS:
        # At a 180-degree kink the transport derivative is singular. Use bounded
        # tangent spin while the residual fallback supplies the finite angle.
        if is_parent:
            return -wp.dot(omega_world, t0)
        return wp.dot(omega_world, t1)

    w = t0 + t1
    n = wp.dot(m0, t1)
    scale = n / denom
    m0_transport_raw = m0 - scale * w
    m0_transport = _normalize_with_fallback(m0_transport_raw, m0)

    ddenom = wp.dot(dt0, t1) + wp.dot(t0, dt1)
    dn = wp.dot(dm0, t1) + wp.dot(m0, dt1)
    dscale = (dn * denom - n * ddenom) / (denom * denom)
    dm0_transport_raw = dm0 - dscale * w - scale * (dt0 + dt1)

    raw_len = wp.length(m0_transport_raw)
    dm0_transport = wp.vec3(0.0)
    if raw_len > _SMALL_LENGTH_EPS:
        inv_len = 1.0 / raw_len
        dm0_transport = inv_len * (dm0_transport_raw - m0_transport * wp.dot(m0_transport, dm0_transport_raw))

    sin_theta = wp.dot(t1, wp.cross(m0_transport, m1))
    cos_theta = wp.dot(m0_transport, m1)
    dsin = wp.dot(dt1, wp.cross(m0_transport, m1)) + wp.dot(
        t1, wp.cross(dm0_transport, m1) + wp.cross(m0_transport, dm1)
    )
    dcos = wp.dot(dm0_transport, m1) + wp.dot(m0_transport, dm1)
    denom_angle = wp.max(_ROD_TWIST_ATAN2_DENOM_EPS, sin_theta * sin_theta + cos_theta * cos_theta)
    return (cos_theta * dsin - sin_theta * dcos) / denom_angle


@wp.func
def _rod_bend_twist_directional_derivatives_from_measure(
    q_wp: wp.quat,
    measure: RodBendTwistMeasure,
    omega_world: wp.vec3,
    is_parent: bool,
) -> tuple[wp.vec3, float]:
    """Bend/twist derivatives for one endpoint angular perturbation.

    Returns ``(d_bend_local, d_twist)``: the un-projected parent-frame
    curvature-binormal derivative and transported-twist derivative. For a parent
    perturbation the bend term also differentiates the parent-frame coordinates
    (the ``-omega_world x kb`` term).
    """
    t0 = measure.t0
    t1 = measure.t1

    dt0 = wp.vec3(0.0)
    dt1 = wp.vec3(0.0)
    if is_parent:
        dt0 = wp.cross(omega_world, t0)
    else:
        dt1 = wp.cross(omega_world, t1)

    dkb_world = _finite_curvature_binormal_derivative(t0, t1, dt0, dt1)

    # A parent rotation also turns the parent frame the binormal is expressed in,
    # so the parent-frame coordinates pick up the extra -omega x kb term before
    # the single map into the parent frame.
    dkb_parent_frame = dkb_world
    if is_parent:
        dkb_parent_frame = dkb_world - wp.cross(omega_world, measure.kb_world)
    d_bend_local = wp.quat_rotate(wp.quat_inverse(q_wp), dkb_parent_frame)

    d_twist = _transported_twist_angle_derivative_from_measure(measure, omega_world, is_parent)
    return d_bend_local, d_twist


@wp.func
def _finite_curvature_binormal_jacobian(t0: wp.vec3, t1: wp.vec3, is_parent: bool) -> wp.mat33:
    """Jacobian of the finite curvature binormal for one endpoint rotation.

    This is the matrix form of ``_finite_curvature_binormal_derivative`` and
    evaluates geometry shared by the three world-axis columns only once.
    """
    zero = wp.mat33(0.0)
    raw_tangent_dot = wp.dot(t0, t1)
    tangent_dot = wp.clamp(raw_tangent_dot, -1.0, 1.0)
    tangent_cross = wp.cross(t0, t1)
    denom = 1.0 + tangent_dot
    cross_sq = wp.dot(tangent_cross, tangent_cross)
    if denom <= _ROD_KB_FOLD_EPS and cross_sq <= _ROD_KB_FOLD_EPS:
        return zero

    identity = wp.identity(3, float)
    if is_parent:
        ddenom_domega = tangent_cross
        dcross_domega = wp.outer(t0, t1) - raw_tangent_dot * identity
    else:
        ddenom_domega = -tangent_cross
        dcross_domega = raw_tangent_dot * identity - wp.outer(t1, t0)

    denom_safe = wp.max(_ROD_KB_FOLD_EPS, denom)
    inv_denom = 1.0 / denom_safe
    kb_raw = (2.0 * inv_denom) * tangent_cross
    dkb_domega = (2.0 * inv_denom) * dcross_domega
    if denom > _ROD_KB_FOLD_EPS:
        dkb_domega = dkb_domega - (2.0 * inv_denom * inv_denom * wp.outer(tangent_cross, ddenom_domega))

    kb_len = wp.length(kb_raw)
    if kb_len > _ROD_KB_CURVATURE_CAP:
        inv_len = 1.0 / kb_len
        projection = wp.identity(3, float) - (inv_len * inv_len) * wp.outer(kb_raw, kb_raw)
        dkb_domega = (_ROD_KB_CURVATURE_CAP * inv_len) * projection * dkb_domega

    return dkb_domega


@wp.func
def _transported_twist_angle_jacobian_from_measure(
    measure: RodBendTwistMeasure,
    is_parent: bool,
) -> wp.vec3:
    """Jacobian row of transported twist for one endpoint rotation.

    Use the tangent-bisector closed form for well-conditioned tangents. Near a
    fold, evaluate the full directional derivative along each world axis so the
    Jacobian remains consistent with the float32 transport residual.
    """
    t0 = measure.t0
    t1 = measure.t1
    denom = 1.0 + wp.clamp(wp.dot(t0, t1), -1.0, 1.0)
    if denom <= _ROD_TWIST_JACOBIAN_DIRECTIONAL_DENOM:
        e0 = wp.vec3(1.0, 0.0, 0.0)
        e1 = wp.vec3(0.0, 1.0, 0.0)
        e2 = wp.vec3(0.0, 0.0, 1.0)
        return wp.vec3(
            _transported_twist_angle_derivative_from_measure(measure, e0, is_parent),
            _transported_twist_angle_derivative_from_measure(measure, e1, is_parent),
            _transported_twist_angle_derivative_from_measure(measure, e2, is_parent),
        )

    jacobian = (t0 + t1) / denom
    return -jacobian if is_parent else jacobian


@wp.func
def _rod_bend_twist_jacobian_z_from_measure(
    measure: RodBendTwistMeasure,
    is_parent: bool,
) -> wp.mat33:
    """Jacobian of [bend_x, bend_y, twist_z] for rod joints with fixed local +Z.

    The local residual is exactly ``[bend_x, bend_y, twist_z]``, so no bend
    projector or twist-axis vector is needed in this hot path.
    """
    dkb_domega = _finite_curvature_binormal_jacobian(measure.t0, measure.t1, is_parent)
    if is_parent:
        dkb_domega = dkb_domega + wp.skew(measure.kb_world)

    parent_x_world = measure.m0
    parent_y_world = wp.cross(measure.t0, measure.m0)
    bend_jacobian_x = wp.transpose(dkb_domega) * parent_x_world
    bend_jacobian_y = wp.transpose(dkb_domega) * parent_y_world
    twist_jacobian = _transported_twist_angle_jacobian_from_measure(measure, is_parent)
    return wp.matrix_from_rows(bend_jacobian_x, bend_jacobian_y, twist_jacobian)


@wp.func
def _wrap_principal_angle(angle: float) -> float:
    """Wrap a bounded angular difference into ``[-pi, pi]``.

    Args:
        angle: Difference of two principal angles. The caller guarantees a
            value in ``[-2*pi, 2*pi]``.

    Returns:
        Equivalent principal angular difference.
    """
    if angle > wp.pi:
        angle -= 2.0 * wp.pi
    elif angle < -wp.pi:
        angle += 2.0 * wp.pi
    return angle


@wp.func
def _assemble_geometric_rod_kappa_z(
    q_wp: wp.quat,
    kb_now_world: wp.vec3,
    twist_now: float,
    kb_rest_local: wp.vec3,
    twist_rest: float,
) -> wp.vec3:
    """Assemble [bend_x, bend_y, twist_z] for SolverVBD local +Z rod joints."""
    bend_now_local = wp.quat_rotate(wp.quat_inverse(q_wp), kb_now_world)
    bend_residual_local = bend_now_local - kb_rest_local
    # In the local +Z convention, parent-frame x/y are bend and z is twist.
    # Wrap the twist delta of two atan2 angles into [-pi, pi] to avoid a ~2*pi
    # jump when rest/current straddle the branch cut (one revolution per joint).
    twist_residual = _wrap_principal_angle(twist_now - twist_rest)
    return wp.vec3(bend_residual_local[0], bend_residual_local[1], twist_residual)


@wp.func
def _rod_bend_twist_delta(kappa: wp.vec3, kappa_prev: wp.vec3) -> wp.vec3:
    """Return a temporal rod bend/twist strain increment.

    Args:
        kappa: Current ``[bend_x, bend_y, twist_z]`` strain.
        kappa_prev: Previous strain in the same representation.

    Returns:
        Bend increments from ordinary subtraction and the shortest signed
        principal-angle increment for twist.
    """
    return wp.vec3(
        kappa[0] - kappa_prev[0],
        kappa[1] - kappa_prev[1],
        _wrap_principal_angle(kappa[2] - kappa_prev[2]),
    )


@wp.func
def compute_geometric_rod_kappa_cached_z(
    q_wp: wp.quat,
    q_wc: wp.quat,
    kb_rest_local: wp.vec3,
    twist_rest: float,
) -> wp.vec3:
    """Geometric rod-joint strain residual for fixed local +Z."""
    measure = _measure_rod_bend_twist_z(q_wp, q_wc)
    return _assemble_geometric_rod_kappa_z(q_wp, measure.kb_world, measure.twist, kb_rest_local, twist_rest)


@wp.func
def _diag_mul_mat33(d: wp.vec3, m: wp.mat33) -> wp.mat33:
    """Return diag(d) * m without building a dense diagonal matrix."""
    return wp.matrix_from_rows(
        d[0] * wp.vec3(m[0, 0], m[0, 1], m[0, 2]),
        d[1] * wp.vec3(m[1, 0], m[1, 1], m[1, 2]),
        d[2] * wp.vec3(m[2, 0], m[2, 1], m[2, 2]),
    )


@wp.func
def compute_right_jacobian_inverse(kappa: wp.vec3) -> wp.mat33:
    """Inverse right Jacobian Jr^{-1}(kappa) for SO(3) rotation vectors.

    Args:
        kappa: Rotation vector theta*axis (any frame).

    Returns:
        wp.mat33: Jr^{-1}(kappa) in the same frame as kappa.
    """
    theta = wp.length(kappa)
    kappa_skew = wp.skew(kappa)

    if (theta < _SMALL_ANGLE_EPS) or (_USE_SMALL_ANGLE_APPROX):
        return wp.identity(3, float) + 0.5 * kappa_skew + (1.0 / 12.0) * (kappa_skew * kappa_skew)

    sin_theta = wp.sin(theta)
    cos_theta = wp.cos(theta)
    b = (1.0 / (theta * theta)) - (1.0 + cos_theta) / (2.0 * theta * sin_theta)
    return wp.identity(3, float) + 0.5 * kappa_skew + b * (kappa_skew * kappa_skew)


@wp.func
def compute_kappa_dot(
    J_world: wp.mat33,
    omega_p_world: wp.vec3,
    omega_c_world: wp.vec3,
) -> wp.vec3:
    """Time derivative of the rotation-vector residual d(kappa)/dt in parent frame.

    Exploits J_world^T = Jr_inv * R_align^T * R_wp^T, so
    kappa_dot = J_world^T * (omega_c - omega_p).

    Args:
        J_world: World-frame force Jacobian from compute_kappa_and_jacobian.
        omega_p_world: Parent angular velocity (world) [rad/s].
        omega_c_world: Child angular velocity (world) [rad/s].

    Returns:
        wp.vec3: Rotation-vector rate kappa_dot in parent frame [rad/s].
    """
    return wp.transpose(J_world) * (omega_c_world - omega_p_world)


@wp.func
def compute_kappa_and_jacobian(
    q_wp: wp.quat,
    q_wc: wp.quat,
    q_wp_rest: wp.quat,
    q_wc_rest: wp.quat,
):
    """Compute rotation-vector residual and world-frame Jacobian from quaternion poses.

    Returns:
        (kappa, J_world) -- rotation vector and world-frame force Jacobian.
    """
    q_rel = wp.quat_inverse(q_wp) * q_wc
    q_rel_rest = wp.quat_inverse(q_wp_rest) * q_wc_rest
    q_align = q_rel * wp.quat_inverse(q_rel_rest)
    if q_align[3] < 0.0:
        q_align = wp.quat(-q_align[0], -q_align[1], -q_align[2], -q_align[3])
    axis, angle = wp.quat_to_axis_angle(q_align)
    kappa = axis * angle

    Jr_inv = compute_right_jacobian_inverse(kappa)
    R_wp = wp.quat_to_matrix(q_wp)
    R_align = wp.quat_to_matrix(q_align)
    J_world = R_wp * (R_align * wp.transpose(Jr_inv))
    return kappa, J_world


@wp.func
def build_joint_projectors(
    jt: int,
    joint_axis: wp.array[wp.vec3],
    qd_start: int,
    lin_count: int,
    ang_count: int,
    q_wp_rot: wp.quat,
):
    """Build orthogonal-complement projectors P_lin and P_ang.

    P = I - sum(ai * ai^T) over free axes (must be orthonormal).
    P_lin projects the world linear residual: axes rotated by q_wp_rot per call,
      so re-project stored multipliers at each read site.
    P_ang projects the parent-frame angular residual (kappa): axes constant,
      so stored multipliers stay in-basis automatically.
    """
    P_lin = wp.identity(3, float)
    P_ang = wp.identity(3, float)

    if jt == JointType.PRISMATIC:
        a_w = wp.normalize(wp.quat_rotate(q_wp_rot, joint_axis[qd_start]))
        P_lin = P_lin - wp.outer(a_w, a_w)
    elif jt == JointType.D6:
        if lin_count > 0:
            a0_w = wp.normalize(wp.quat_rotate(q_wp_rot, joint_axis[qd_start]))
            P_lin = P_lin - wp.outer(a0_w, a0_w)
        if lin_count > 1:
            a1_w = wp.normalize(wp.quat_rotate(q_wp_rot, joint_axis[qd_start + 1]))
            P_lin = P_lin - wp.outer(a1_w, a1_w)
        if lin_count > 2:
            a2_w = wp.normalize(wp.quat_rotate(q_wp_rot, joint_axis[qd_start + 2]))
            P_lin = P_lin - wp.outer(a2_w, a2_w)

    if jt == JointType.REVOLUTE:
        a = wp.normalize(joint_axis[qd_start])
        P_ang = P_ang - wp.outer(a, a)
    elif jt == JointType.D6:
        if ang_count > 0:
            a0 = wp.normalize(joint_axis[qd_start + lin_count])
            P_ang = P_ang - wp.outer(a0, a0)
        if ang_count > 1:
            a1 = wp.normalize(joint_axis[qd_start + lin_count + 1])
            P_ang = P_ang - wp.outer(a1, a1)
        if ang_count > 2:
            a2 = wp.normalize(joint_axis[qd_start + lin_count + 2])
            P_ang = P_ang - wp.outer(a2, a2)

    return P_lin, P_ang


@wp.func
def _average_contact_material(
    ke0: float,
    kd0: float,
    mu0: float,
    ke1: float,
    kd1: float,
    mu1: float,
):
    """Average material properties for a contact pair.

    ke, kd: arithmetic mean.
    mu: geometric mean.
    """
    avg_ke = 0.5 * (ke0 + ke1)
    avg_kd = 0.5 * (kd0 + kd1)
    avg_mu = wp.sqrt(mu0 * mu1)
    return avg_ke, avg_kd, avg_mu


@wp.func
def _project_coulomb_tangent(tangent: wp.vec3, tangent_length: float, cone_limit: float):
    """Project a tangent vector onto the Coulomb friction disk."""
    if cone_limit <= 0.0:
        return wp.vec3(0.0)
    if tangent_length > cone_limit:
        return tangent * (cone_limit / tangent_length)
    return tangent


@wp.func
def _regularized_coulomb_scale(u_norm: float, eps_u: float) -> float:
    """Return a linearly regularized reciprocal-slip factor.

    A positive ``eps_u`` continues ``1/u`` below the threshold with matching
    value and slope; otherwise the exact reciprocal is used for positive slip.
    """
    if u_norm > eps_u and u_norm > 0.0:
        return 1.0 / u_norm
    if eps_u > 0.0:
        return (-u_norm / eps_u + 2.0) / eps_u
    return 0.0


@wp.func
def _compliant_contact_dual_step(
    lam_old: wp.vec3,
    normal: wp.vec3,
    normal_residual: float,
    tangent_residual: wp.vec3,
    material_k: float,
    mu: float,
    rho_n: float,
    rho_t: float,
):
    """Advance a finite-normal/ideal-Coulomb contact multiplier.

    The normal dual is updated and projected first. The tangent trial is then
    projected into the Coulomb disk defined by that normal update.
    """
    lam_n_old = wp.dot(lam_old, normal)
    lam_t_old = lam_old - normal * lam_n_old

    lam_n_new = wp.max(_alm_relaxed_ascent(lam_n_old, normal_residual, material_k, rho_n), 0.0)

    lam_t_trial = lam_t_old + rho_t * tangent_residual
    tangent_trial_length = wp.length(lam_t_trial)
    cone_limit = mu * lam_n_new
    lam_t_new = _project_coulomb_tangent(lam_t_trial, tangent_trial_length, cone_limit)

    return normal * lam_n_new + lam_t_new


@wp.func
def _contact_force_terms(
    normal_rho: float,
    material_k: float,
    lam: wp.vec3,
    normal: wp.vec3,
):
    """Apply the ALM scaling to a contact multiplier's normal component only.

    Returns the ``(k, lam_eff)`` pair of the force ``k*C + lam_eff``. Friction is
    a Coulomb bound with no material stiffness, so it passes through unscaled.
    """
    lambda_n = wp.dot(lam, normal)
    normal_primal_k, lambda_n_eff = _material_force_terms(normal_rho, material_k, lambda_n, 1)
    lam_n = normal * lambda_n
    return normal_primal_k, normal * lambda_n_eff + (lam - lam_n)


@wp.func
def _load_solve_weight(
    legacy_penalty_k: wp.array[float],
    compliant_rho: wp.array[float],
    idx: int,
    use_compliant_alm: int,
):
    """Load the active solve weight without reading the inactive array.

    Compliant ALM uses ``rho``, constant across the sweeps; legacy AVBD uses the
    mutable ``legacy_penalty_k`` (material stiffness in soft mode, a ramping
    penalty in hard mode).
    """
    if use_compliant_alm == 1:
        return compliant_rho[idx]
    return legacy_penalty_k[idx]


@wp.func
def _structural_row_has_stiffness(solve_weight: float, material_k: float, use_compliant_alm: int):
    """Return whether a structural row has stiffness in the selected formulation."""
    if use_compliant_alm == 1:
        return material_k > 0.0
    return solve_weight > 0.0


@wp.func
def _ramp_penalty_k(
    current: float,
    cap: float,
    beta: float,
    residual: float,
):
    """Ramp the legacy penalty, clamped to the material stiffness."""
    return wp.min(cap, current + beta * residual)


@wp.func
def _contact_delassus_w(
    body_id: int,
    point_local: wp.vec3,
    n: wp.vec3,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
):
    """Return one body's scalar Delassus term along the contact normal.

    ``inv_m + (r x n)^T I^-1 (r x n)`` with ``r`` from the COM, evaluated in the
    body frame where ``body_inv_inertia`` lives. ``body_id`` must be nonnegative.
    """
    w = body_inv_mass[body_id]
    if w > 0.0:
        q = body_q[body_id]
        # Both the contact point and COM are body-local. Keeping the lever arm
        # local avoids cancellation when the body is far from the world origin.
        r_local = point_local - body_com[body_id]
        n_local = wp.quat_rotate_inv(wp.transform_get_rotation(q), n)
        u_local = wp.cross(r_local, n_local)
        w = w + wp.dot(u_local, body_inv_inertia[body_id] * u_local)
    return w


@wp.func
def _contact_body_conditioning_scale(
    shape_id: int,
    point_local: wp.vec3,
    n: wp.vec3,
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    body_structural_k: wp.array[float],
    inv_dt_sq: float,
):
    """Return one body's contact conditioning scale, ``D_b + S_b``.

    ``D_b = 1/(dt^2*w_b)`` is the inertial stiffness at the contact point and
    ``S_b`` the body's structural summary. World, static, and kinematic shapes
    return zero.
    """
    if shape_id < 0:
        return 0.0
    body_id = shape_body[shape_id]
    if body_id < 0:
        return 0.0
    w_b = _contact_delassus_w(body_id, point_local, n, body_q, body_com, body_inv_mass, body_inv_inertia)
    if w_b <= 0.0:
        return 0.0
    return inv_dt_sq / w_b + body_structural_k[body_id]


@wp.func
def _series_scale(a: float, b: float):
    """Combine two stiffness scales in series, ``a*b/(a+b)``, or zero if either is absent.

    The product overflows float32 for two large scales, so the equivalent
    ``lo/(1 + lo/hi)`` is used instead; it stays within ``[lo/2, lo]``.
    """
    if a <= 0.0 or b <= 0.0:
        return 0.0
    lo = wp.min(a, b)
    hi = wp.max(a, b)
    return lo / (1.0 + lo / hi)


@wp.func
def _contact_conditioning_scale(
    shape_id_0: int,
    shape_id_1: int,
    point0_local: wp.vec3,
    point1_local: wp.vec3,
    n: wp.vec3,
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    body_structural_k: wp.array[float],
    inv_dt_sq: float,
):
    """Combine per-body ``D_b + S_b`` scales through contact compliance.

    For dynamic bodies ``b``, the pair scale is
    ``A = 1 / sum_b(1 / (D_b + S_b))``. It reduces to the ordinary normal
    Delassus scale when ``S_b=0`` and to ``D_b+S_b`` against the world.
    """
    scale_0 = _contact_body_conditioning_scale(
        shape_id_0,
        point0_local,
        n,
        shape_body,
        body_q,
        body_com,
        body_inv_mass,
        body_inv_inertia,
        body_structural_k,
        inv_dt_sq,
    )
    scale_1 = _contact_body_conditioning_scale(
        shape_id_1,
        point1_local,
        n,
        shape_body,
        body_q,
        body_com,
        body_inv_mass,
        body_inv_inertia,
        body_structural_k,
        inv_dt_sq,
    )
    if scale_0 > 0.0 and scale_1 > 0.0:
        return _series_scale(scale_0, scale_1)
    return wp.max(scale_0, scale_1)


@wp.func
def _contact_pair_structural_scale(
    body_id_0: int,
    body_id_1: int,
    body_flags: wp.array[wp.int32],
    body_inv_mass: wp.array[float],
    body_structural_k: wp.array[float],
    proxy_flag: int,
):
    """Return the structural stiffness shared by solver-updated contact endpoints."""
    # A proxy's inverse mass is a coupling metric; its pose is not advanced as
    # an independent VBD structural endpoint.
    dynamic_0 = body_id_0 >= 0 and body_inv_mass[body_id_0] > 0.0 and (body_flags[body_id_0] & proxy_flag) == 0
    dynamic_1 = body_id_1 >= 0 and body_inv_mass[body_id_1] > 0.0 and (body_flags[body_id_1] & proxy_flag) == 0
    scale_0 = body_structural_k[body_id_0] if dynamic_0 else 0.0
    scale_1 = body_structural_k[body_id_1] if dynamic_1 else 0.0
    if dynamic_0 and dynamic_1:
        return _series_scale(scale_0, scale_1)
    return wp.max(scale_0, scale_1)


@wp.func
def _contact_body_tangent_block(
    shape_id: int,
    anchor_local: wp.vec3,
    tangent_0: wp.vec3,
    tangent_1: wp.vec3,
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
):
    """Return one body's tangent Delassus block as ``(w00, w01, w11)``."""
    if shape_id < 0:
        return wp.vec3(0.0)
    body_id = shape_body[shape_id]
    if body_id < 0 or body_inv_mass[body_id] <= 0.0:
        return wp.vec3(0.0)

    rot = wp.transform_get_rotation(body_q[body_id])
    tangent_0_local = wp.quat_rotate_inv(rot, tangent_0)
    tangent_1_local = wp.quat_rotate_inv(rot, tangent_1)
    r_local = anchor_local - body_com[body_id]
    u0 = wp.cross(r_local, tangent_0_local)
    u1 = wp.cross(r_local, tangent_1_local)
    inv_inertia = body_inv_inertia[body_id]
    inv_mass = body_inv_mass[body_id]
    w00 = inv_mass + wp.dot(u0, inv_inertia * u0)
    w11 = inv_mass + wp.dot(u1, inv_inertia * u1)
    w01 = wp.dot(u0, inv_inertia * u1)
    return wp.vec3(w00, w01, w11)


@wp.func
def _contact_tangent_conditioning_scale(
    shape_id_0: int,
    shape_id_1: int,
    anchor0_local: wp.vec3,
    anchor1_local: wp.vec3,
    n: wp.vec3,
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    inv_dt_sq: float,
):
    """Return the minimum pair inertial stiffness over all tangent directions.

    Endpoint Delassus blocks add before taking the largest eigenvalue, which is
    the pair's most compliant tangent direction.
    """
    tangent_0, tangent_1 = orthonormal_basis(n)
    w = _contact_body_tangent_block(
        shape_id_0,
        anchor0_local,
        tangent_0,
        tangent_1,
        shape_body,
        body_q,
        body_com,
        body_inv_mass,
        body_inv_inertia,
    ) + _contact_body_tangent_block(
        shape_id_1,
        anchor1_local,
        tangent_0,
        tangent_1,
        shape_body,
        body_q,
        body_com,
        body_inv_mass,
        body_inv_inertia,
    )
    trace = w[0] + w[2]
    if trace <= 0.0:
        return 0.0
    discriminant = wp.sqrt(wp.max((w[0] - w[2]) * (w[0] - w[2]) + 4.0 * w[1] * w[1], 0.0))
    w_max = 0.5 * (trace + discriminant)
    # Direction-agnostic structural support enters separately through
    # _contact_auto_tangent_rho, capped by the normal rho.
    return inv_dt_sq / w_max


@wp.func
def _contact_penalty_floor(material_k: float, k_start: float):
    """Resolve the mutable legacy contact-penalty floor.

    A nonnegative ``k_start`` gives ``min(k_start, material_k)``; a negative value
    uses ``material_k``, disabling the ramp.
    """
    if k_start >= 0.0:
        return wp.min(k_start, material_k)
    return material_k


@wp.func
def _update_dual_vec3(
    C_vec: wp.vec3,
    C0: wp.vec3,
    alpha: float,
    solve_weight: float,
    material_k: float,
    lam: wp.vec3,
    legacy_is_hard: int,
    use_compliant_alm: int,
):
    """Update one vec3 dual in the selected formulation.

    Compliant ALM uses finite-material relaxed ascent; legacy hard accumulates
    the stabilized residual; legacy soft leaves the multiplier unchanged.
    """
    if use_compliant_alm == 0 and legacy_is_hard == 0:
        return lam

    C_update = C_vec - alpha * C0
    if use_compliant_alm == 1:
        return _alm_relaxed_ascent(lam, C_update, material_k, solve_weight)
    return solve_weight * C_update + lam


@wp.func
def _update_joint_structural_dual(
    residual: wp.vec3,
    C0: wp.vec3,
    alpha: float,
    penalty_k: wp.array[float],
    rho: wp.array[float],
    material_k: wp.array[float],
    joint_is_hard: wp.array[wp.int32],
    idx: int,
    lam: wp.vec3,
    use_compliant_alm: int,
    beta: float,
) -> wp.vec3:
    """Update one structural dual and ramp its legacy penalty."""
    row_material_k = material_k[idx]
    legacy_is_hard = int(0)
    if use_compliant_alm == 1:
        solve_weight = rho[idx]
    else:
        solve_weight = penalty_k[idx]
        legacy_is_hard = joint_is_hard[idx]

    lam_new = _update_dual_vec3(
        residual,
        C0,
        alpha,
        solve_weight,
        row_material_k,
        lam,
        legacy_is_hard,
        use_compliant_alm,
    )
    if use_compliant_alm != 1:
        penalty_k[idx] = _ramp_penalty_k(
            solve_weight,
            row_material_k,
            beta,
            wp.length(residual),
        )
    return lam_new


@wp.func
def evaluate_angular_constraint_force_hessian(
    q_wp: wp.quat,
    q_wc: wp.quat,
    q_wp_rest: wp.quat,
    q_wc_rest: wp.quat,
    q_wp_prev: wp.quat,
    q_wc_prev: wp.quat,
    is_parent: bool,
    solve_weight: float,
    material_k: float,
    P: wp.mat33,
    lambda_ang: wp.vec3,
    C0_ang: wp.vec3,
    alpha: float,
    damping: float,
    use_compliant_alm: int,
    dt: float,
):
    """Projected angular constraint force/Hessian using rotation-vector error (kappa).

    Generic evaluator for non-rod angular constraints. Computes force and
    Hessian in the constrained subspace defined by the orthogonal-complement
    projector P. Angular Dahl friction is rod-only and handled separately, so
    this evaluator carries no friction term.

    C0 stabilization: when alpha > 0 and C0_ang is nonzero, the effective
    kappa is kappa - alpha*C0_ang (initial violation snapshot).

    Special cases by projector:
      - P = I: isotropic (FIXED angular)
      - P = I - a*a^T: revolute (1 free angular axis)
      - arbitrary P: D6 (0-3 free angular axes)

    ``solve_weight`` is ALM ``rho`` in compliant mode and the mutable penalty
    stiffness in legacy mode. ``material_k`` supplies the finite material stiffness
    used by compliant ALM.
    Returns:
        (tau_world, H_aa, kappa, J_world) -- constraint torque and Hessian in world
        frame, plus the rotation vector and world-frame Jacobian for reuse by the
        drive/limit block.
    """
    inv_dt = 1.0 / dt

    kappa_now_vec, J_world = compute_kappa_and_jacobian(q_wp, q_wc, q_wp_rest, q_wc_rest)
    kappa_stab = kappa_now_vec - alpha * C0_ang
    kappa_perp = P * kappa_stab
    joint_primal_k, lambda_eff = _material_force_terms(solve_weight, material_k, lambda_ang, use_compliant_alm)

    # P_ang is constant for joint angular residuals, so lambda_ang should already
    # be in-basis. Project here too so stale or externally edited state cannot
    # apply force along a free angular DOF.
    f_local = joint_primal_k * kappa_perp + P * lambda_eff
    H_local = joint_primal_k * P

    if damping > 0.0:
        omega_p_world = quat_velocity(q_wp, q_wp_prev, dt)
        omega_c_world = quat_velocity(q_wc, q_wc_prev, dt)

        dkappa_dt_vec = compute_kappa_dot(J_world, omega_p_world, omega_c_world)
        dkappa_perp = P * dkappa_dt_vec
        f_damp_local = damping * dkappa_perp
        f_local = f_local + f_damp_local

        k_damp = damping * inv_dt
        H_local = H_local + k_damp * P

    H_aa = J_world * (H_local * wp.transpose(J_world))

    tau_world = J_world * f_local
    if not is_parent:
        tau_world = -tau_world

    return tau_world, H_aa, kappa_now_vec, J_world


@wp.func
def evaluate_rod_bend_twist_force_hessian_z(
    q_wp: wp.quat,
    q_wc: wp.quat,
    kb_rest_local: wp.vec3,
    twist_rest: float,
    q_wp_prev: wp.quat,
    q_wc_prev: wp.quat,
    is_parent: bool,
    K_elastic_diag: wp.vec3,
    C0_force: wp.vec3,
    sigma0: wp.vec3,
    H_fric_diag: wp.vec3,
    lambda_projected: wp.vec3,
    K_damp_diag: wp.vec3,
    damping_active: bool,
    dt: float,
):
    """Bend/twist torque and Hessian for SolverVBD local +Z rod joints.

    In the fixed rod material basis, local angular operators are diagonal:
    ``[bend_x, bend_y, twist_z]``. Keep them as vec3 row scales in the hot path
    instead of building dense local matrices.
    """
    inv_dt = 1.0 / dt

    measure = _measure_rod_bend_twist_z(q_wp, q_wc)
    kappa_now_vec = _assemble_geometric_rod_kappa_z(q_wp, measure.kb_world, measure.twist, kb_rest_local, twist_rest)

    # Bend and twist decouple in the material basis: the angular energy is a sum
    # of independent quadratics in [bend_x, bend_y, twist_z], so elastic stiffness
    # and the friction Hessian have no off-diagonal coupling. Carry them as vec3
    # row scales; the dense angular block reappears below via J^T diag(H) J.
    f_local = wp.cw_mul(K_elastic_diag, kappa_now_vec) - C0_force + sigma0 + lambda_projected
    H_local_diag = K_elastic_diag + H_fric_diag

    if damping_active:
        prev_measure = _measure_rod_bend_twist_z(q_wp_prev, q_wc_prev)
        kappa_prev_vec = _assemble_geometric_rod_kappa_z(
            q_wp_prev, prev_measure.kb_world, prev_measure.twist, kb_rest_local, twist_rest
        )
        dkappa_dt = _rod_bend_twist_delta(kappa_now_vec, kappa_prev_vec) * inv_dt
        f_local = f_local + wp.cw_mul(K_damp_diag, dkappa_dt)
        H_local_diag = H_local_diag + inv_dt * K_damp_diag

    J_body = _rod_bend_twist_jacobian_z_from_measure(measure, is_parent)
    # Gauss-Newton self Hessian: J^T diag(H_local_diag) J.
    H_aa = wp.transpose(J_body) * _diag_mul_mat33(H_local_diag, J_body)
    tau_world = -(wp.transpose(J_body) * f_local)

    return tau_world, H_aa, kappa_now_vec, J_body


@wp.func
def evaluate_linear_constraint_force_hessian(
    X_wp: wp.transform,
    X_wc: wp.transform,
    X_wp_prev: wp.transform,
    X_wc_prev: wp.transform,
    parent_pose: wp.transform,
    child_pose: wp.transform,
    parent_com: wp.vec3,
    child_com: wp.vec3,
    is_parent: bool,
    solve_weight: float,
    material_k: float,
    P: wp.mat33,
    lambda_lin: wp.vec3,
    C0_lin: wp.vec3,
    alpha: float,
    damping: float,
    use_compliant_alm: int,
    dt: float,
):
    """Projected linear constraint force/Hessian for anchor coincidence.

    Generic evaluator for non-rod linear constraints. Computes C = x_c - x_p,
    projects with P, and returns force/Hessian in world frame.

    C0 stabilization: when alpha > 0 and C0_lin is nonzero, the effective
    constraint violation is C - alpha*C0 (initial violation snapshot).

    Special cases by projector:
      - P = I: isotropic (BALL, FIXED linear, REVOLUTE linear)
      - P = I - a*a^T: prismatic (1 free linear axis)
      - arbitrary P: D6 (0-3 free linear axes)

    ``solve_weight`` is ALM ``rho`` in compliant mode and the mutable penalty
    stiffness in legacy mode. ``material_k`` supplies the finite material stiffness
    used by compliant ALM.

    Returns:
      - force (wp.vec3): Linear force (world)
      - torque (wp.vec3): Angular torque (world)
      - H_ll (wp.mat33): Linear-linear block
      - H_al (wp.mat33): Angular-linear block
      - H_aa (wp.mat33): Angular-angular block
    """
    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)

    if is_parent:
        com_w = wp.transform_point(parent_pose, parent_com)
        r = x_p - com_w
    else:
        com_w = wp.transform_point(child_pose, child_com)
        r = x_c - com_w

    C_vec = x_c - x_p
    C_stab = C_vec - alpha * C0_lin
    C_perp = P * C_stab
    joint_primal_k, lambda_eff = _material_force_terms(solve_weight, material_k, lambda_lin, use_compliant_alm)

    # P_lin rotates per call -> must re-project lambda_lin (see build_joint_projectors).
    f_attachment = joint_primal_k * C_perp + P * lambda_eff

    K_eff = joint_primal_k * P
    if damping > 0.0:
        inv_dt = 1.0 / dt

        x_p_prev = wp.transform_get_translation(X_wp_prev)
        x_c_prev = wp.transform_get_translation(X_wc_prev)
        C_vec_prev = x_c_prev - x_p_prev
        dC_dt_perp = P * ((C_vec - C_vec_prev) * inv_dt)
        f_attachment = f_attachment + damping * dC_dt_perp
        K_eff = K_eff + (damping * inv_dt) * P

    rx = wp.skew(r)
    H_ll = K_eff
    H_al = rx * K_eff
    H_aa = wp.transpose(rx) * K_eff * rx

    force = f_attachment if is_parent else -f_attachment
    torque = wp.cross(r, force)

    return force, torque, H_ll, H_al, H_aa


@wp.func
def evaluate_rod_stretch_shear_force_hessian(
    X_wp: wp.transform,
    X_wc: wp.transform,
    X_wp_prev: wp.transform,
    X_wc_prev: wp.transform,
    parent_pose: wp.transform,
    child_pose: wp.transform,
    parent_com: wp.vec3,
    child_com: wp.vec3,
    is_parent: bool,
    k_diag: wp.vec3,
    C0_force_local: wp.vec3,
    lambda_local: wp.vec3,
    kd_diag: wp.vec3,
    damping_active: bool,
    dt: float,
):
    """Rod-joint stretch/shear anchor force, torque, and PSD Gauss-Newton self-Hessian.

    All inputs are parent-material: residual ``u = R_p^T (x_c - x_p) =
    [shear_x, shear_y, stretch_z]``, diagonal ``k_diag = (k_shear, k_shear,
    k_stretch)`` / ``kd_diag``, and local ``C0_force_local`` / ``lambda_local``.
    Elastic and AL are energy gradients, damping is dissipative in ``u``, and both
    bodies react at the shared child anchor, so the net internal wrench is zero.
    """
    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)
    q_wp = wp.transform_get_rotation(X_wp)

    if is_parent:
        com_w = wp.transform_point(parent_pose, parent_com)
    else:
        com_w = wp.transform_point(child_pose, child_com)
    r = x_c - com_w

    C_vec = x_c - x_p
    u = wp.quat_rotate_inv(q_wp, C_vec)
    psi = wp.cw_mul(k_diag, u) - C0_force_local + lambda_local

    k_s = k_diag[0]
    k_z = k_diag[2]
    h_s = k_s
    h_z = k_z
    if damping_active:
        inv_dt = 1.0 / dt
        x_p_prev = wp.transform_get_translation(X_wp_prev)
        x_c_prev = wp.transform_get_translation(X_wc_prev)
        u_prev = wp.quat_rotate_inv(wp.transform_get_rotation(X_wp_prev), x_c_prev - x_p_prev)
        psi = psi + wp.cw_mul(kd_diag, (u - u_prev) * inv_dt)
        h_s = h_s + kd_diag[0] * inv_dt
        h_z = h_z + kd_diag[2] * inv_dt

    f_world = wp.quat_rotate(q_wp, psi)
    force = f_world if is_parent else -f_world

    t = _quat_rotate_local_z(q_wp)
    identity = wp.identity(3, float)
    K_eff = h_s * identity + (h_z - h_s) * wp.outer(t, t)
    H_ll = K_eff
    if is_parent:
        # Isotropic elastic energy is frame-invariant, so it takes the parent-anchor arm that
        # evaluate_linear_constraint_force_hessian also uses; min() extracts the largest such block
        # leaving a PSD remainder. Damping keeps the material arm: u_prev is frozen one step back.
        k_iso = wp.min(k_s, k_z)
        K_material = K_eff - k_iso * identity
        r_elastic = x_p - com_w
        rx_material = wp.skew(r)
        H_al = k_iso * wp.skew(r_elastic) + rx_material * K_material
        H_aa = k_iso * (wp.length_sq(r_elastic) * identity - wp.outer(r_elastic, r_elastic))
        H_aa = H_aa + wp.transpose(rx_material) * K_material * rx_material
    else:
        rx = wp.skew(r)
        H_al = rx * K_eff
        H_aa = wp.transpose(rx) * K_eff * rx

    torque = wp.cross(r, force)
    return force, torque, H_ll, H_al, H_aa


# ---------------------------------
# Data structures
# ---------------------------------


@wp.struct
class RigidForceElementAdjacencyInfo:
    r"""
    Stores adjacency information for rigid bodies and their connected joints using CSR (Compressed Sparse Row) format.

    - body_adj_joints: Flattened array of joint IDs. Size is sum over all bodies of N_i, where N_i is the
      number of joints connected to body i.

    - body_adj_joints_offsets: Offset array indicating where each body's joint list starts.
      Size is |B|+1 (number of bodies + 1).
      The number of joints adjacent to body i is: body_adj_joints_offsets[i+1] - body_adj_joints_offsets[i]
    """

    # Rigid body joint adjacency
    body_adj_joints: wp.array[wp.int32]
    body_adj_joints_offsets: wp.array[wp.int32]

    def to(self, device):
        if device == self.body_adj_joints.device:
            return self
        else:
            adjacency_gpu = RigidForceElementAdjacencyInfo()
            adjacency_gpu.body_adj_joints = self.body_adj_joints.to(device)
            adjacency_gpu.body_adj_joints_offsets = self.body_adj_joints_offsets.to(device)

            return adjacency_gpu


@wp.func
def get_body_num_adjacent_joints(adjacency: RigidForceElementAdjacencyInfo, body: wp.int32):
    """Number of joints adjacent to the given body from CSR offsets."""
    return adjacency.body_adj_joints_offsets[body + 1] - adjacency.body_adj_joints_offsets[body]


@wp.func
def get_body_adjacent_joint_id(adjacency: RigidForceElementAdjacencyInfo, body: wp.int32, joint: wp.int32):
    """Joint id at local index `joint` within the body's CSR-adjacent joint list."""
    offset = adjacency.body_adj_joints_offsets[body]
    return adjacency.body_adj_joints[offset + joint]


@wp.func
def evaluate_rigid_contact_from_collision(
    body_a_index: int,
    body_b_index: int,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    contact_point_a_local: wp.vec3,
    contact_point_b_local: wp.vec3,
    contact_offset_a_local: wp.vec3,
    contact_offset_b_local: wp.vec3,
    contact_normal: wp.vec3,
    penetration_depth: float,
    normal_solve_weight: float,
    contact_material_ke: float,
    contact_tangent_rho: float,
    contact_kd: float,
    contact_lam: wp.vec3,
    friction_mu: float,
    friction_epsilon: float,
    legacy_hard_contact: int,
    contact_compliant_alm: int,
    dt: float,
    friction_c0: wp.vec3,
):
    """Compute rigid-contact forces and 3x3 Hessian blocks for a contact pair.

    Legacy hard and compliant ALM contacts use displacement-based projected
    Coulomb friction; legacy soft contact uses velocity-based IPC friction.

    Returns:
        10-tuple: (force_a, torque_a, H_ll_a, H_al_a, H_aa_a,
                   force_b, torque_b, H_ll_b, H_al_b, H_aa_b)
    """
    normal_primal_k = normal_solve_weight
    contact_lam_eff = contact_lam
    if contact_compliant_alm == 1:
        normal_primal_k, contact_lam_eff = _contact_force_terms(
            normal_solve_weight, contact_material_ke, contact_lam, contact_normal
        )

    lam_n = wp.dot(contact_lam_eff, contact_normal)

    if penetration_depth <= _SMALL_LENGTH_EPS and lam_n <= 0.0:
        zero_vec = wp.vec3(0.0)
        zero_mat = wp.mat33(0.0)
        return (zero_vec, zero_vec, zero_mat, zero_mat, zero_mat, zero_vec, zero_vec, zero_mat, zero_mat, zero_mat)

    f_n = normal_primal_k * penetration_depth + lam_n
    if normal_primal_k <= 0.0:
        zero_vec = wp.vec3(0.0)
        zero_mat = wp.mat33(0.0)
        return (zero_vec, zero_vec, zero_mat, zero_mat, zero_mat, zero_vec, zero_vec, zero_mat, zero_mat, zero_mat)
    f_n = wp.max(f_n, 0.0)

    # Only legacy hard retains a zero-force Hessian. Soft and compliant rows release
    # it so separated contacts do not resist free motion.
    if f_n == 0.0 and (legacy_hard_contact == 0 or contact_compliant_alm == 1):
        zero_vec = wp.vec3(0.0)
        zero_mat = wp.mat33(0.0)
        return (zero_vec, zero_vec, zero_mat, zero_mat, zero_mat, zero_vec, zero_vec, zero_mat, zero_mat, zero_mat)

    if body_a_index < 0:
        X_wa = wp.transform_identity()
        X_wa_prev = wp.transform_identity()
        body_a_com_local = wp.vec3(0.0)
    else:
        X_wa = body_q[body_a_index]
        X_wa_prev = body_q_prev[body_a_index]
        body_a_com_local = body_com[body_a_index]

    if body_b_index < 0:
        X_wb = wp.transform_identity()
        X_wb_prev = wp.transform_identity()
        body_b_com_local = wp.vec3(0.0)
    else:
        X_wb = body_q[body_b_index]
        X_wb_prev = body_q_prev[body_b_index]
        body_b_com_local = body_com[body_b_index]

    x_com_a_now = wp.transform_point(X_wa, body_a_com_local)
    x_com_b_now = wp.transform_point(X_wb, body_b_com_local)

    # Normal response uses geometric (skeleton) points; friction uses the surface anchor.
    x_s_a_now = wp.transform_point(X_wa, contact_point_a_local)
    x_s_b_now = wp.transform_point(X_wb, contact_point_b_local)
    x_s_a_prev = wp.transform_point(X_wa_prev, contact_point_a_local)
    x_s_b_prev = wp.transform_point(X_wb_prev, contact_point_b_local)

    x_c_a_now = contact_surface_point(X_wa, contact_point_a_local, contact_offset_a_local)
    x_c_b_now = contact_surface_point(X_wb, contact_point_b_local, contact_offset_b_local)
    x_c_a_prev = contact_surface_point(X_wa_prev, contact_point_a_local, contact_offset_a_local)
    x_c_b_prev = contact_surface_point(X_wb_prev, contact_point_b_local, contact_offset_b_local)

    n_outer = wp.outer(contact_normal, contact_normal)
    I3 = wp.identity(n=3, dtype=float)

    # Normal approach rate from the geometric points (not the rotating anchor).
    v_rel_n = (x_s_b_now - x_s_b_prev - x_s_a_now + x_s_a_prev) / dt
    v_dot_n = wp.dot(contact_normal, v_rel_n)

    # Tangential slip from the surface anchor (required for finite-radius friction).
    v_rel_t = (x_c_b_now - x_c_b_prev - x_c_a_now + x_c_a_prev) / dt
    v_t = v_rel_t - contact_normal * wp.dot(contact_normal, v_rel_t)

    # Normal block (force + optional approach damping), applied at the geometric lever.
    f_n_vec = contact_normal * f_n
    K_n = normal_primal_k * n_outer

    # Tangential friction block, applied at the surface-anchor lever.
    f_t_vec = wp.vec3(0.0)
    K_t = wp.mat33(0.0)

    if legacy_hard_contact == 1 or contact_compliant_alm == 1:
        if friction_mu > 0.0 and f_n > 0.0:
            # Hard/ALM: Coulomb projection of rho_t*(disp + friction_c0) + lambda_t.
            # friction_c0 is (1-alpha)*C0_t from the caller. ALM uses contact_tangent_rho;
            # legacy hard reuses the normal solve weight.
            rho_t = normal_solve_weight
            if contact_compliant_alm == 1:
                rho_t = contact_tangent_rho
            tangential_disp = -(v_t * dt)
            lam_t = contact_lam_eff - contact_normal * lam_n
            f_t_vec = rho_t * (tangential_disp + friction_c0) + lam_t
            f_t_len = wp.length(f_t_vec)
            cone_limit = friction_mu * f_n
            if contact_compliant_alm == 1 and f_t_len > cone_limit:
                # Outside the Coulomb cone, the exact projection Jacobian has zero slip-direction
                # stiffness. Add a conservative PSD solve metric there; force stays unchanged.
                t_hat = f_t_vec / f_t_len
                t_outer = wp.outer(t_hat, t_hat)
                slip_norm = wp.length(tangential_disp)
                sliding_solve_metric = cone_limit * _regularized_coulomb_scale(slip_norm, friction_epsilon * dt)
                K_t = (cone_limit / f_t_len) * rho_t * (I3 - n_outer - t_outer)
                K_t = K_t + sliding_solve_metric * t_outer
            else:
                K_t = rho_t * (I3 - n_outer)
            f_t_vec = _project_coulomb_tangent(f_t_vec, f_t_len, cone_limit)
    else:
        # Soft: IPC regularized Coulomb (force and Hessian).
        if friction_mu > 0.0 and f_n > 0.0:
            f_friction, K_friction = compute_projected_isotropic_friction(
                friction_mu, f_n, contact_normal, v_t * dt, friction_epsilon * dt
            )
            f_t_vec = f_friction
            K_t = K_friction

    if contact_kd > 0.0 and v_dot_n < 0.0 and f_n > 0.0:
        f_n_vec = f_n_vec - contact_kd * v_dot_n * contact_normal
        K_n = K_n + (contact_kd / dt) * n_outer

    f_total = f_n_vec + f_t_vec
    K_total = K_n + K_t

    # Geometric lever for the normal block, surface-anchor lever for friction.
    r_s_a = x_s_a_now - x_com_a_now
    r_c_a = x_c_a_now - x_com_a_now
    r_s_b = x_s_b_now - x_com_b_now
    r_c_b = x_c_b_now - x_com_b_now

    r_s_a_skew_T = wp.transpose(wp.skew(r_s_a))
    r_c_a_skew_T = wp.transpose(wp.skew(r_c_a))
    h_al_a = -(r_s_a_skew_T * K_n + r_c_a_skew_T * K_t)
    h_aa_a = r_s_a_skew_T * K_n * wp.skew(r_s_a) + r_c_a_skew_T * K_t * wp.skew(r_c_a)

    r_s_b_skew_T = wp.transpose(wp.skew(r_s_b))
    r_c_b_skew_T = wp.transpose(wp.skew(r_c_b))
    h_al_b = -(r_s_b_skew_T * K_n + r_c_b_skew_T * K_t)
    h_aa_b = r_s_b_skew_T * K_n * wp.skew(r_s_b) + r_c_b_skew_T * K_t * wp.skew(r_c_b)

    torque_a = wp.cross(r_s_a, -f_n_vec) + wp.cross(r_c_a, -f_t_vec)
    torque_b = wp.cross(r_s_b, f_n_vec) + wp.cross(r_c_b, f_t_vec)

    return (
        -f_total,
        torque_a,
        K_total,
        h_al_a,
        h_aa_a,
        f_total,
        torque_b,
        K_total,
        h_al_b,
        h_aa_b,
    )


@wp.func
def _compute_body_particle_contact_force(
    penetration_depth: float,
    n: wp.vec3,
    relative_translation: wp.vec3,
    ke: float,
    kd: float,
    mu: float,
    friction_epsilon: float,
    dt: float,
):
    """Pure force law for body-particle contacts: normal penalty + damping + friction.

    All geometry and kinematics (penetration, normal, relative displacement) are
    resolved by the caller.  This function only computes the contact force and
    Hessian from those scalar/vector inputs.
    """
    f_n = penetration_depth * ke
    force = n * f_n
    hessian = ke * wp.outer(n, n)

    if wp.dot(n, relative_translation) < 0.0:
        damping_hessian = (kd / dt) * wp.outer(n, n)
        hessian = hessian + damping_hessian
        force = force - damping_hessian * relative_translation

    eps_u = friction_epsilon * dt
    friction_force, friction_hessian = compute_projected_isotropic_friction(mu, f_n, n, relative_translation, eps_u)
    force = force + friction_force
    hessian = hessian + friction_hessian

    return force, hessian


@wp.func
def _eval_body_particle_contact(
    particle_index: int,
    particle_pos: wp.vec3,
    particle_prev_pos: wp.vec3,
    contact_index: int,
    body_particle_contact_ke: float,
    body_particle_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    particle_radius: wp.array[float],
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    dt: float,
):
    """Particle-rigid contact force/Hessian - resolves geometry from arrays then
    delegates to ``_compute_body_particle_contact_force``.

    Prefer calling ``_compute_body_particle_contact_force`` directly when the
    caller already has the contact geometry and relative displacement.
    """
    shape_index = contact_shape[contact_index]
    body_index = shape_body[shape_index]

    X_wb = wp.transform_identity()
    X_com = wp.vec3()
    if body_index >= 0:
        X_wb = body_q[body_index]
        X_com = body_com[body_index]

    bx = wp.transform_point(X_wb, contact_body_pos[contact_index])
    n = contact_normal[contact_index]

    margin = shape_margin[shape_index] if shape_margin.shape[0] > 0 else 0.0
    penetration_depth = -(wp.dot(n, particle_pos - bx) - particle_radius[particle_index] - margin)
    if penetration_depth > 0.0:
        dx = particle_pos - particle_prev_pos

        if body_q_prev:
            X_wb_prev = wp.transform_identity()
            if body_index >= 0:
                X_wb_prev = body_q_prev[body_index]
            bx_prev = wp.transform_point(X_wb_prev, contact_body_pos[contact_index])
            bv = (bx - bx_prev) / dt + wp.transform_vector(X_wb, contact_body_vel[contact_index])
        else:
            r = bx - wp.transform_point(X_wb, X_com)
            body_v_s = wp.spatial_vector()
            if body_index >= 0:
                body_v_s = body_qd[body_index]
            body_w = wp.spatial_bottom(body_v_s)
            body_v = wp.spatial_top(body_v_s)
            bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, contact_body_vel[contact_index])

        relative_translation = dx - bv * dt

        return _compute_body_particle_contact_force(
            penetration_depth,
            n,
            relative_translation,
            body_particle_contact_ke,
            body_particle_contact_kd,
            friction_mu,
            friction_epsilon,
            dt,
        )
    else:
        return wp.vec3(0.0), wp.mat33(0.0)


@wp.func
def _eval_soft_ef_contact(
    contact_index: int,
    corners: wp.vec3i,
    bary: wp.vec3,
    pos: wp.array[wp.vec3],
    pos_prev: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    contact_ke: float,
    contact_kd: float,
    contact_mu: float,
    friction_epsilon: float,
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    dt: float,
):
    """Soft-contact force/Hessian at a barycentric contact point over a record's soft particles.

    The contact point is ``x = sum_i bary[i] * pos[corners[i]]`` over the -1-padded ``corners``:
    a particle record ``(p, -1, -1)`` with ``bary = (1, 0, 0)`` reduces to the single-vertex
    particle-vs-surface case, an edge ``(v0, v1, -1)`` uses two corners, a face all three. Geometry
    and body kinematics are resolved as in :func:`_eval_body_particle_contact`, then the shared force
    law :func:`_compute_body_particle_contact_force` is applied. Returns the contact force and Hessian
    *at the contact point* (the caller distributes them by barycentric weight) and the world-space
    rigid contact point ``bx`` for the body-side reaction. Force/Hessian are zero without penetration.
    """
    v0 = corners[0]  # corner 0 is always valid; corners 1/2 are -1 for edge/particle records
    c1 = corners[1]
    c2 = corners[2]

    x = bary[0] * pos[v0]
    x_prev = bary[0] * pos_prev[v0]
    radius = particle_radius[v0]
    if c1 >= 0:
        x += bary[1] * pos[c1]
        x_prev += bary[1] * pos_prev[c1]
        radius = wp.max(radius, particle_radius[c1])
    if c2 >= 0:
        x += bary[2] * pos[c2]
        x_prev += bary[2] * pos_prev[c2]
        radius = wp.max(radius, particle_radius[c2])

    shape_index = contact_shape[contact_index]
    body_index = shape_body[shape_index]

    X_wb = wp.transform_identity()
    X_com = wp.vec3()
    if body_index >= 0:
        X_wb = body_q[body_index]
        X_com = body_com[body_index]

    bx = wp.transform_point(X_wb, contact_body_pos[contact_index])
    n = contact_normal[contact_index]

    # per-shape contact margin (#2994), applied the same way as the particle-vs-surface path
    margin = shape_margin[shape_index] if shape_margin.shape[0] > 0 else 0.0

    force = wp.vec3(0.0)
    hessian = wp.mat33(0.0)

    penetration_depth = -(wp.dot(n, x - bx) - radius - margin)
    if penetration_depth > 0.0:
        dx = x - x_prev

        if body_q_prev:
            X_wb_prev = wp.transform_identity()
            if body_index >= 0:
                X_wb_prev = body_q_prev[body_index]
            bx_prev = wp.transform_point(X_wb_prev, contact_body_pos[contact_index])
            bv = (bx - bx_prev) / dt + wp.transform_vector(X_wb, contact_body_vel[contact_index])
        else:
            r = bx - wp.transform_point(X_wb, X_com)
            body_v_s = wp.spatial_vector()
            if body_index >= 0:
                body_v_s = body_qd[body_index]
            body_w = wp.spatial_bottom(body_v_s)
            body_v = wp.spatial_top(body_v_s)
            bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, contact_body_vel[contact_index])

        relative_translation = dx - bv * dt

        # contact_ke/kd/mu are the per-contact AVBD values (ramped penalty + pre-mixed material,
        # cached by init_body_particle_contacts) -- the same source the particle path uses.
        force, hessian = _compute_body_particle_contact_force(
            penetration_depth,
            n,
            relative_translation,
            contact_ke,
            contact_kd,
            contact_mu,
            friction_epsilon,
            dt,
        )

    return force, hessian, bx


@wp.func
def evaluate_body_particle_contact(
    particle_index: int,
    particle_pos: wp.vec3,
    particle_prev_pos: wp.vec3,
    contact_index: int,
    body_particle_contact_ke: float,
    body_particle_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    particle_radius: wp.array[float],
    shape_material_mu: wp.array[float],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    dt: float,
):
    """Particle-rigid contact force/Hessian with per-shape mu mixing.

    VBD rigid-side uses ``_eval_body_particle_contact`` directly (mu is
    pre-averaged per contact).  This wrapper is kept for other solvers
    that pass raw mu and need per-shape mixing.
    """
    shape_index = contact_shape[contact_index]
    mixed_mu = wp.sqrt(friction_mu * shape_material_mu[shape_index])
    return _eval_body_particle_contact(
        particle_index,
        particle_pos,
        particle_prev_pos,
        contact_index,
        body_particle_contact_ke,
        body_particle_contact_kd,
        mixed_mu,
        friction_epsilon,
        particle_radius,
        shape_body,
        body_q,
        body_q_prev,
        body_qd,
        body_com,
        contact_shape,
        contact_body_pos,
        contact_body_vel,
        contact_normal,
        shape_margin,
        dt,
    )


@wp.func
def compute_projected_isotropic_friction(
    friction_mu: float,
    normal_load: float,
    n_hat: wp.vec3,
    slip_u: wp.vec3,
    eps_u: float,
) -> tuple[wp.vec3, wp.mat33]:
    """Isotropic Coulomb friction in world frame using projector P = I - n n^T.

    Regularization: if ||u_t|| <= eps_u, uses a linear ramp; otherwise 1/||u_t||.

    Args:
        friction_mu: Coulomb friction coefficient (>= 0).
        normal_load: Normal load magnitude (>= 0).
        n_hat: Unit contact normal (world frame).
        slip_u: Tangential slip displacement over dt (world frame).
        eps_u: Smoothing distance (same units as slip_u, > 0).

    Returns:
        tuple[wp.vec3, wp.mat33]: (force, Hessian) in world frame.
    """
    # Tangential slip in the contact tangent plane without forming P: u_t = u - n * (n dot u)
    dot_nu = wp.dot(n_hat, slip_u)
    u_t = slip_u - n_hat * dot_nu
    u_norm = wp.length(u_t)

    if u_norm > 0.0:
        # Factor common scalar; force aligned with u_t, Hessian proportional to projector
        scale = friction_mu * normal_load * _regularized_coulomb_scale(u_norm, eps_u)
        f = -(scale * u_t)
        K = scale * (wp.identity(3, float) - wp.outer(n_hat, n_hat))
    else:
        f = wp.vec3(0.0)
        K = wp.mat33(0.0)

    return f, K


@wp.func
def resolve_drive_limit_mode(
    q: float,
    target_pos: float,
    lim_lower: float,
    lim_upper: float,
    has_drive: bool,
    has_limits: bool,
):
    """Resolve drive/limit priority and compute position error [m or rad].

    Limits take precedence: if q is outside [lower, upper], the active limit
    wins. Otherwise the drive engages with target clamped to the limit range.

    Returns:
        (mode, err_pos) -- active mode constant and signed position error.
    """
    mode = _DRIVE_LIMIT_MODE_NONE
    err_pos = float(0.0)
    drive_target = target_pos
    if has_limits:
        drive_target = wp.clamp(target_pos, lim_lower, lim_upper)
        if q < lim_lower:
            mode = _DRIVE_LIMIT_MODE_LIMIT_LOWER
            err_pos = q - lim_lower
        elif q > lim_upper:
            mode = _DRIVE_LIMIT_MODE_LIMIT_UPPER
            err_pos = q - lim_upper
    if mode == _DRIVE_LIMIT_MODE_NONE and has_drive:
        mode = _DRIVE_LIMIT_MODE_DRIVE
        err_pos = q - drive_target
    return mode, err_pos


@wp.func
def _resolve_active_drive_row(
    q: float,
    target_pos: float,
    lim_lower: float,
    lim_upper: float,
    has_drive: bool,
    has_limits: bool,
    use_compliant_alm: int,
):
    """Legacy mutex mode, or ALM drive mode with target clamped into limits.

    ALM clamps so the drive never fights the separate limit row.
    """
    if use_compliant_alm == 0:
        return resolve_drive_limit_mode(q, target_pos, lim_lower, lim_upper, has_drive, has_limits)

    if not has_drive:
        return _DRIVE_LIMIT_MODE_NONE, 0.0

    drive_target = target_pos
    if has_limits:
        drive_target = wp.clamp(target_pos, lim_lower, lim_upper)
    return _DRIVE_LIMIT_MODE_DRIVE, q - drive_target


@wp.func
def _nearest_limit_bound(q: float, lim_lower: float, lim_upper: float, has_limits: bool):
    """Pick the nearer finite bound for the limit residual (not the violated one).

    Violation-gated selection chatters at a resting bound where gap is zero but
    the multiplier is nonzero. Proximity keeps the bound stable through that
    fixed point.
    """
    if not has_limits:
        return _DRIVE_LIMIT_MODE_NONE
    if lim_lower <= -MAXVAL:
        return _DRIVE_LIMIT_MODE_LIMIT_UPPER
    if lim_upper >= MAXVAL:
        return _DRIVE_LIMIT_MODE_LIMIT_LOWER
    if q - lim_lower <= lim_upper - q:
        return _DRIVE_LIMIT_MODE_LIMIT_LOWER
    return _DRIVE_LIMIT_MODE_LIMIT_UPPER


@wp.func
def _limit_signed_gap(q: float, lim_lower: float, lim_upper: float, bound: int):
    """Return ``q - bound`` for the selected limit, or zero if none."""
    if bound == _DRIVE_LIMIT_MODE_LIMIT_LOWER:
        return q - lim_lower
    if bound == _DRIVE_LIMIT_MODE_LIMIT_UPPER:
        return q - lim_upper
    return 0.0


@wp.func
def _spring_damper_step(error: float, rate_error: float, material_k: float, material_d: float, inv_dt: float):
    """Return ``(e_row, K + D/dt)`` so ``K*e + D*rate = (K + D/dt)*e_row``."""
    step_k = material_k + material_d * inv_dt
    if step_k <= 0.0:
        return 0.0, 0.0
    return (material_k * error + material_d * rate_error) / step_k, step_k


@wp.func
def _evaluate_drive_limit_axis(
    error: float,
    rate: float,
    target_rate: float,
    active_mode: int,
    drive_k: float,
    drive_d: float,
    limit_k: float,
    limit_d: float,
    inv_dt: float,
):
    """Legacy mutex drive/limit spring: one active mode, no dual.

    The damper is an explicit additive term (not folded into the residual).
    """
    if active_mode == _DRIVE_LIMIT_MODE_DRIVE:
        return (
            drive_k * error + drive_d * (rate - target_rate),
            drive_k + drive_d * inv_dt,
        )
    if active_mode == _DRIVE_LIMIT_MODE_LIMIT_LOWER or active_mode == _DRIVE_LIMIT_MODE_LIMIT_UPPER:
        return limit_k * error + limit_d * rate, limit_k + limit_d * inv_dt
    return 0.0, 0.0


@wp.func
def _evaluate_drive_axis(
    error: float,
    rate: float,
    target_rate: float,
    active_mode: int,
    drive_k: float,
    drive_d: float,
    axis_support: float,
    drive_lambda: float,
    inv_dt: float,
):
    """Compliant ALM bilateral drive force/Hessian."""
    if active_mode != _DRIVE_LIMIT_MODE_DRIVE or axis_support <= 0.0:
        return 0.0, 0.0
    row_error, row_k = _spring_damper_step(error, rate - target_rate, drive_k, drive_d, inv_dt)
    if row_k <= 0.0:
        return 0.0, 0.0
    rho = _drive_auto_rho(axis_support, row_k)
    s, k_eff, _a = _compliant_alm_coefficients(row_k, rho)
    return k_eff * row_error + s * drive_lambda, k_eff


@wp.func
def _update_drive_lambda(
    error: float,
    rate: float,
    target_rate: float,
    active_mode: int,
    drive_k: float,
    drive_d: float,
    axis_support: float,
    drive_lambda: float,
    inv_dt: float,
):
    """Advance a compliant bilateral drive dual.

    Inactive rows clear the multiplier; bilateral drives require no sign projection.
    """
    if active_mode != _DRIVE_LIMIT_MODE_DRIVE or axis_support <= 0.0:
        return 0.0
    row_error, row_k = _spring_damper_step(error, rate - target_rate, drive_k, drive_d, inv_dt)
    if row_k <= 0.0:
        return 0.0
    return _alm_relaxed_ascent(drive_lambda, row_error, row_k, _drive_auto_rho(axis_support, row_k))


@wp.func
def _evaluate_limit_axis(
    q: float,
    lim_lower: float,
    lim_upper: float,
    has_limits: bool,
    limit_k: float,
    limit_d: float,
    rate: float,
    axis_support: float,
    limit_lambda: float,
    inv_dt: float,
):
    """Evaluate a compliant unilateral limit alongside the independent drive row.

    Keep only forces directed into the selected bound, allowing a resting limit
    to retain its multiplier while releasing forces directed away. Legacy mode
    does not call this function.
    """
    bound = _nearest_limit_bound(q, lim_lower, lim_upper, has_limits)
    if bound == _DRIVE_LIMIT_MODE_NONE or limit_k <= 0.0 or axis_support <= 0.0:
        return 0.0, 0.0
    gap = _limit_signed_gap(q, lim_lower, lim_upper, bound)
    row_error, row_k = _spring_damper_step(gap, rate, limit_k, limit_d, inv_dt)
    rho = _limit_auto_rho(axis_support, row_k)
    s, k_eff, _a = _compliant_alm_coefficients(row_k, rho)
    force = k_eff * row_error + s * limit_lambda
    if bound == _DRIVE_LIMIT_MODE_LIMIT_LOWER and force >= 0.0:
        return 0.0, 0.0
    if bound == _DRIVE_LIMIT_MODE_LIMIT_UPPER and force <= 0.0:
        return 0.0, 0.0
    return force, k_eff


@wp.func
def _update_limit_lambda(
    q: float,
    lim_lower: float,
    lim_upper: float,
    has_limits: bool,
    limit_k: float,
    limit_d: float,
    rate: float,
    axis_support: float,
    limit_lambda: float,
    inv_dt: float,
):
    """Advance the compliant unilateral limit multiplier independently of the drive.

    Nearest-bound selection preserves resting reactions. Sign projection keeps
    lower multipliers nonpositive and upper multipliers nonnegative, clearing
    stale state after a bound switch.
    """
    bound = _nearest_limit_bound(q, lim_lower, lim_upper, has_limits)
    if bound == _DRIVE_LIMIT_MODE_NONE or limit_k <= 0.0 or axis_support <= 0.0:
        return 0.0
    gap = _limit_signed_gap(q, lim_lower, lim_upper, bound)
    row_error, row_k = _spring_damper_step(gap, rate, limit_k, limit_d, inv_dt)
    rho = _limit_auto_rho(axis_support, row_k)
    lambda_new = _alm_relaxed_ascent(limit_lambda, row_error, row_k, rho)
    if bound == _DRIVE_LIMIT_MODE_LIMIT_LOWER:
        return wp.min(lambda_new, 0.0)
    return wp.max(lambda_new, 0.0)


# Drive/limit row predicates shared by support seeding, force, and dual paths.
@wp.func
def _drive_row_applies_force(drive_ke: float, drive_kd: float):
    """Whether the drive row contributes force on either path.

    A pure damper (zero stiffness, nonzero damping) is a valid authored drive,
    so damping alone activates the row.
    """
    return drive_ke > 0.0 or drive_kd > 0.0


@wp.func
def _drive_row_needs_state_update(drive_ke: float, drive_kd: float, use_compliant_alm: int):
    """Whether the dual pass must update this drive row.

    Compliant ALM updates a multiplier for stiffness or damping; legacy only
    ramps penalty state for positive stiffness.
    """
    return drive_ke > 0.0 or (use_compliant_alm == 1 and drive_kd > 0.0)


@wp.func
def _limit_row_exists(limit_ke: float, lower: float, upper: float):
    """Whether an authored limit row exists (stiffness and a finite bound)."""
    return limit_ke > 0.0 and (lower > -MAXVAL or upper < MAXVAL)


@wp.func
def _drive_limit_needs_support(
    dof: int,
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_limit_ke: wp.array[float],
):
    """Whether this DOF has an authored drive/limit row that needs support seeding."""
    return _drive_row_applies_force(joint_target_ke[dof], joint_target_kd[dof]) or _limit_row_exists(
        joint_limit_ke[dof], joint_limit_lower[dof], joint_limit_upper[dof]
    )


@wp.struct
class JointAxisDriveLimit:
    """Drive and limit data for one joint DOF.

    ``material_*_ke`` retain authored stiffness for activation and compliant ALM.
    ``drive_ke``/``limit_ke`` are the solve stiffnesses, capped by the legacy
    ramping penalty. Drive damping may independently activate a drive row.
    """

    material_drive_ke: float
    material_limit_ke: float
    drive_ke: float
    limit_ke: float
    drive_kd: float
    limit_kd: float
    target_pos: float
    target_vel: float
    lower: float
    upper: float


@wp.func
def _load_joint_axis_drive_limit(
    dof_idx: int,
    target_q_idx: int,
    penalty_slot: int,
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_q: wp.array[float],
    joint_target_qd: wp.array[float],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_limit_ke: wp.array[float],
    joint_limit_kd: wp.array[float],
    joint_penalty_k: wp.array[float],
    use_compliant_alm: int,
):
    """Gather one DOF's drive/limit coefficients from the model arrays."""
    axis = JointAxisDriveLimit()
    axis.material_drive_ke = joint_target_ke[dof_idx]
    axis.material_limit_ke = joint_limit_ke[dof_idx]
    axis.drive_kd = joint_target_kd[dof_idx]
    axis.limit_kd = joint_limit_kd[dof_idx]
    axis.target_pos = joint_target_q[target_q_idx]
    axis.target_vel = joint_target_qd[dof_idx]
    axis.lower = joint_limit_lower[dof_idx]
    axis.upper = joint_limit_upper[dof_idx]

    axis.drive_ke = axis.material_drive_ke
    axis.limit_ke = axis.material_limit_ke
    if use_compliant_alm == 0:
        penalty_ke = joint_penalty_k[penalty_slot]
        axis.drive_ke = wp.min(penalty_ke, axis.material_drive_ke)
        axis.limit_ke = wp.min(penalty_ke, axis.material_limit_ke)
    return axis


@wp.func
def _eval_joint_axis_drive_limit(
    axis: JointAxisDriveLimit,
    q: float,
    rate: float,
    has_drive: bool,
    has_limits: bool,
    axis_support: float,
    drive_lambda: float,
    limit_lambda: float,
    use_compliant_alm: int,
    inv_dt: float,
):
    """One DOF drive/limit force and Hessian: legacy mutex or ALM superposition."""
    active_mode, err_pos = _resolve_active_drive_row(
        q, axis.target_pos, axis.lower, axis.upper, has_drive, has_limits, use_compliant_alm
    )
    if use_compliant_alm == 0:
        return _evaluate_drive_limit_axis(
            err_pos,
            rate,
            axis.target_vel,
            active_mode,
            axis.drive_ke,
            axis.drive_kd,
            axis.limit_ke,
            axis.limit_kd,
            inv_dt,
        )

    f_drive, H_drive = _evaluate_drive_axis(
        err_pos,
        rate,
        axis.target_vel,
        active_mode,
        axis.drive_ke,
        axis.drive_kd,
        axis_support,
        drive_lambda,
        inv_dt,
    )
    f_limit, H_limit = _evaluate_limit_axis(
        q,
        axis.lower,
        axis.upper,
        has_limits,
        axis.limit_ke,
        axis.limit_kd,
        rate,
        axis_support,
        limit_lambda,
        inv_dt,
    )
    return f_drive + f_limit, H_drive + H_limit


@wp.func
def _update_joint_axis_drive_limit_state(
    axis: JointAxisDriveLimit,
    q: float,
    rate: float,
    has_drive: bool,
    has_limits: bool,
    dof_idx: int,
    penalty_slot: int,
    beta: float,
    inv_dt: float,
    joint_penalty_k: wp.array[float],
    joint_material_k: wp.array[float],
    joint_drive_limit_support: wp.array[float],
    joint_drive_lambda: wp.array[float],
    joint_limit_lambda: wp.array[float],
    use_compliant_alm: int,
):
    """Update compliant drive/limit multipliers or the legacy shared penalty."""
    active_mode, err_pos = _resolve_active_drive_row(
        q, axis.target_pos, axis.lower, axis.upper, has_drive, has_limits, use_compliant_alm
    )
    if use_compliant_alm == 0:
        joint_penalty_k[penalty_slot] = _ramp_penalty_k(
            joint_penalty_k[penalty_slot], joint_material_k[penalty_slot], beta, wp.abs(err_pos)
        )
        return

    axis_support = joint_drive_limit_support[dof_idx]
    joint_drive_lambda[dof_idx] = _update_drive_lambda(
        err_pos,
        rate,
        axis.target_vel,
        active_mode,
        axis.material_drive_ke,
        axis.drive_kd,
        axis_support,
        joint_drive_lambda[dof_idx],
        inv_dt,
    )

    joint_limit_lambda[dof_idx] = _update_limit_lambda(
        q,
        axis.lower,
        axis.upper,
        has_limits,
        axis.material_limit_ke,
        axis.limit_kd,
        rate,
        axis_support,
        joint_limit_lambda[dof_idx],
        inv_dt,
    )


@wp.func
def apply_angular_drive_limit_torque(
    a: wp.vec3,
    J_world: wp.mat33,
    is_parent: bool,
    f_scalar: float,
    H_scalar: float,
):
    """Rank-1 angular drive/limit torque and Hessian along local axis a.

    Maps scalar spring-damper (f_scalar, H_scalar) through J_world to
    world-frame torque and H_aa.
    """
    Ja = J_world * a
    tau = f_scalar * Ja
    Haa = H_scalar * wp.outer(Ja, Ja)
    if not is_parent:
        tau = -tau
    return tau, Haa


@wp.func
def apply_linear_drive_limit_force(
    axis_w: wp.vec3,
    r: wp.vec3,
    is_parent: bool,
    f_scalar: float,
    H_scalar: float,
):
    """Rank-1 linear drive/limit force and Hessian along world axis.

    Maps scalar spring-damper (f_scalar, H_scalar) to world-frame force,
    torque, and Hessian blocks (H_ll, H_al, H_aa) via the moment arm r.
    """
    f_attachment = f_scalar * axis_w
    ra = wp.cross(r, axis_w)
    Hll = H_scalar * wp.outer(axis_w, axis_w)
    Hal = H_scalar * wp.outer(ra, axis_w)
    Haa = H_scalar * wp.outer(ra, ra)
    force = f_attachment if is_parent else -f_attachment
    torque = wp.cross(r, force)
    return force, torque, Hll, Hal, Haa


@wp.func
def _zero_force_hessian():
    """Zero (force, torque, H_ll, H_al, H_aa) tuple for early-exit paths."""
    return wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0)


@wp.func
def evaluate_joint_force_hessian(
    body_index: int,
    joint_index: int,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_q_rest: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_rod_rest_kb_local: wp.array[wp.vec3],
    joint_rod_rest_twist: wp.array[float],
    joint_qd_start: wp.array[int],
    joint_target_q_start: wp.array[int],
    joint_constraint_start: wp.array[int],
    joint_penalty_k: wp.array[float],
    joint_rho: wp.array[float],
    joint_material_k: wp.array[float],
    joint_penalty_kd: wp.array[float],
    joint_sigma_start: wp.array[wp.vec3],
    joint_C_fric: wp.array[wp.vec3],
    # Drive parameters (DOF-indexed via joint_qd_start)
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_q: wp.array[float],
    joint_target_qd: wp.array[float],
    # Limit parameters (DOF-indexed via joint_qd_start)
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_limit_ke: wp.array[float],
    joint_limit_kd: wp.array[float],
    joint_drive_limit_support: wp.array[float],
    joint_drive_lambda: wp.array[float],
    joint_limit_lambda: wp.array[float],
    joint_lambda_lin: wp.array[wp.vec3],
    joint_lambda_ang: wp.array[wp.vec3],
    joint_C0_lin: wp.array[wp.vec3],
    joint_C0_ang: wp.array[wp.vec3],
    joint_is_hard: wp.array[wp.int32],
    stab_alpha: float,
    joint_compliant_alm: int,
    joint_dof_dim: wp.array2d[int],
    joint_rest_angle: wp.array[float],
    dt: float,
):
    """Compute VBD joint force and Hessian contributions for one body.

    Supported joint types: ROD, BALL, FIXED, REVOLUTE, PRISMATIC, D6.
    Rod joints use split stretch/shear and bend/twist helpers; other joints use
    projector-based linear/angular evaluators.

    Indexing:
        joint_constraint_start[j] is a solver-owned start offset into the per-constraint
        arrays (joint_penalty_k, joint_penalty_kd, joint_material_k, joint_rho,
        and joint_is_hard). Layout per joint type:
          - ROD:   4 scalars -> [stretch, shear, bend, twist]
          - BALL:  1 scalar  -> [linear]
          - FIXED: 2 scalars -> [linear, angular]
          - REVOLUTE:  3 scalars -> [linear, angular, ang_drive_limit]
          - PRISMATIC: 3 scalars -> [linear, angular, lin_drive_limit]
          - D6: 2 + lin_count + ang_count scalars -> [linear, angular, per-DOF drive/limit]
        Legacy drive/limit slots use AVBD-ramped penalty stiffness. Under
        compliant ALM, each free DOF is a scalar finite-material row with its
        own multiplier; active limits project onto the appropriate half-line.
    """
    jt = joint_type[joint_index]
    if (
        jt != JointType.ROD
        and jt != JointType.BALL
        and jt != JointType.FIXED
        and jt != JointType.REVOLUTE
        and jt != JointType.PRISMATIC
        and jt != JointType.D6
    ):
        return _zero_force_hessian()

    if not joint_enabled[joint_index]:
        return _zero_force_hessian()

    parent_index = joint_parent[joint_index]
    child_index = joint_child[joint_index]
    if body_index != child_index and (parent_index < 0 or body_index != parent_index):
        return _zero_force_hessian()

    is_parent_body = parent_index >= 0 and body_index == parent_index

    X_pj = joint_X_p[joint_index]
    X_cj = joint_X_c[joint_index]

    if parent_index >= 0:
        parent_pose = body_q[parent_index]
        parent_pose_prev = body_q_prev[parent_index]
        parent_com = body_com[parent_index]
    else:
        parent_pose = wp.transform(wp.vec3(0.0), wp.quat_identity())
        parent_pose_prev = parent_pose
        parent_com = wp.vec3(0.0)

    child_pose = body_q[child_index]
    child_pose_prev = body_q_prev[child_index]
    child_com = body_com[child_index]

    X_wp = parent_pose * X_pj
    X_wc = child_pose * X_cj
    X_wp_prev = parent_pose_prev * X_pj
    X_wc_prev = child_pose_prev * X_cj

    c_start = joint_constraint_start[joint_index]

    # Hoist quaternion extraction (shared by all angular constraints and drive/limits)
    q_wp = wp.transform_get_rotation(X_wp)
    q_wc = wp.transform_get_rotation(X_wc)
    q_wp_prev = wp.transform_get_rotation(X_wp_prev)
    q_wc_prev = wp.transform_get_rotation(X_wc_prev)

    if jt == JointType.ROD:
        stretch_idx = c_start
        shear_idx = c_start + 1
        bend_idx = c_start + 2
        twist_idx = c_start + 3

        solve_weight_stretch = _load_solve_weight(joint_penalty_k, joint_rho, stretch_idx, joint_compliant_alm)
        solve_weight_shear = _load_solve_weight(joint_penalty_k, joint_rho, shear_idx, joint_compliant_alm)
        solve_weight_bend = _load_solve_weight(joint_penalty_k, joint_rho, bend_idx, joint_compliant_alm)
        solve_weight_twist = _load_solve_weight(joint_penalty_k, joint_rho, twist_idx, joint_compliant_alm)

        material_stretch = joint_material_k[stretch_idx]
        material_shear = joint_material_k[shear_idx]
        material_bend = joint_material_k[bend_idx]
        material_twist = joint_material_k[twist_idx]

        kd_stretch = joint_penalty_kd[stretch_idx]
        kd_shear = joint_penalty_kd[shear_idx]
        kd_bend = joint_penalty_kd[bend_idx]
        kd_twist = joint_penalty_kd[twist_idx]

        total_force = wp.vec3(0.0)
        total_torque = wp.vec3(0.0)
        total_H_ll = wp.mat33(0.0)
        total_H_al = wp.mat33(0.0)
        total_H_aa = wp.mat33(0.0)

        bend_stiff = _structural_row_has_stiffness(solve_weight_bend, material_bend, joint_compliant_alm)
        twist_stiff = _structural_row_has_stiffness(solve_weight_twist, material_twist, joint_compliant_alm)
        bend_active = bend_stiff or kd_bend > 0.0
        twist_active = twist_stiff or kd_twist > 0.0
        if bend_active or twist_active:
            lambda_ang = joint_lambda_ang[joint_index]
            C0_ang = joint_C0_ang[joint_index]
            bend_hard = bend_stiff and joint_compliant_alm == 0 and joint_is_hard[bend_idx] == 1
            twist_hard = twist_stiff and joint_compliant_alm == 0 and joint_is_hard[twist_idx] == 1

            bend_dual = wp.vec3(lambda_ang[0], lambda_ang[1], 0.0)
            twist_dual = wp.vec3(0.0, 0.0, lambda_ang[2])
            bend_dual_active = joint_compliant_alm == 1 or bend_hard
            twist_dual_active = joint_compliant_alm == 1 or twist_hard
            if not bend_dual_active:
                bend_dual = wp.vec3(0.0)
            if not twist_dual_active:
                twist_dual = wp.vec3(0.0)

            bend_primal_k, bend_dual_eff = _material_force_terms(
                solve_weight_bend, material_bend, bend_dual, joint_compliant_alm
            )
            twist_primal_k, twist_dual_eff = _material_force_terms(
                solve_weight_twist, material_twist, twist_dual, joint_compliant_alm
            )

            K_elastic_diag = wp.vec3(bend_primal_k, bend_primal_k, twist_primal_k)
            K_damp_diag = wp.vec3(kd_bend, kd_bend, kd_twist)
            damping_active = kd_bend > 0.0 or kd_twist > 0.0

            bend_alpha = float(0.0)
            twist_alpha = float(0.0)
            if joint_compliant_alm == 1 or bend_hard:
                bend_alpha = stab_alpha
            if joint_compliant_alm == 1 or twist_hard:
                twist_alpha = stab_alpha

            sigma = wp.vec3(0.0)
            H_fric_diag = wp.vec3(0.0)
            lambda_projected = bend_dual_eff + twist_dual_eff
            C0_force = bend_primal_k * bend_alpha * wp.vec3(C0_ang[0], C0_ang[1], 0.0)
            C0_force = C0_force + twist_primal_k * twist_alpha * wp.vec3(0.0, 0.0, C0_ang[2])
            dahl_sigma = joint_sigma_start[joint_index]
            dahl_fric = joint_C_fric[joint_index]
            if bend_stiff and not bend_hard:
                sigma = sigma + wp.vec3(dahl_sigma[0], dahl_sigma[1], 0.0)
                H_fric_diag = H_fric_diag + wp.vec3(dahl_fric[0], dahl_fric[1], 0.0)
            if twist_stiff and not twist_hard:
                sigma = sigma + wp.vec3(0.0, 0.0, dahl_sigma[2])
                H_fric_diag = H_fric_diag + wp.vec3(0.0, 0.0, dahl_fric[2])

            rod_torque, rod_H_aa, _rod_kappa, _rod_J = evaluate_rod_bend_twist_force_hessian_z(
                q_wp,
                q_wc,
                joint_rod_rest_kb_local[joint_index],
                joint_rod_rest_twist[joint_index],
                q_wp_prev,
                q_wc_prev,
                is_parent_body,
                K_elastic_diag,
                C0_force,
                sigma,
                H_fric_diag,
                lambda_projected,
                K_damp_diag,
                damping_active,
                dt,
            )
            total_torque = total_torque + rod_torque
            total_H_aa = total_H_aa + rod_H_aa

        stretch_stiff = _structural_row_has_stiffness(solve_weight_stretch, material_stretch, joint_compliant_alm)
        shear_stiff = _structural_row_has_stiffness(solve_weight_shear, material_shear, joint_compliant_alm)
        stretch_active = stretch_stiff or kd_stretch > 0.0
        shear_active = shear_stiff or kd_shear > 0.0
        if stretch_active or shear_active:
            lambda_lin = joint_lambda_lin[joint_index]
            C0_lin = joint_C0_lin[joint_index]
            stretch_hard = stretch_stiff and joint_compliant_alm == 0 and joint_is_hard[stretch_idx] == 1
            shear_hard = shear_stiff and joint_compliant_alm == 0 and joint_is_hard[shear_idx] == 1

            stretch_dual = wp.vec3(0.0, 0.0, lambda_lin[2])
            shear_dual = wp.vec3(lambda_lin[0], lambda_lin[1], 0.0)
            stretch_dual_active = joint_compliant_alm == 1 or stretch_hard
            shear_dual_active = joint_compliant_alm == 1 or shear_hard
            if not stretch_dual_active:
                stretch_dual = wp.vec3(0.0)
            if not shear_dual_active:
                shear_dual = wp.vec3(0.0)

            stretch_primal_k, stretch_dual_eff = _material_force_terms(
                solve_weight_stretch, material_stretch, stretch_dual, joint_compliant_alm
            )
            shear_primal_k, shear_dual_eff = _material_force_terms(
                solve_weight_shear, material_shear, shear_dual, joint_compliant_alm
            )

            k_diag = wp.vec3(shear_primal_k, shear_primal_k, stretch_primal_k)
            kd_diag = wp.vec3(kd_shear, kd_shear, kd_stretch)
            damping_active = kd_stretch > 0.0 or kd_shear > 0.0

            stretch_alpha = float(0.0)
            shear_alpha = float(0.0)
            if joint_compliant_alm == 1 or stretch_hard:
                stretch_alpha = stab_alpha
            if joint_compliant_alm == 1 or shear_hard:
                shear_alpha = stab_alpha

            lambda_local = stretch_dual_eff + shear_dual_eff
            C0_force_local = shear_primal_k * shear_alpha * wp.vec3(C0_lin[0], C0_lin[1], 0.0)
            C0_force_local = C0_force_local + stretch_primal_k * stretch_alpha * wp.vec3(0.0, 0.0, C0_lin[2])

            f_l, t_l, Hll_l, Hal_l, Haa_l = evaluate_rod_stretch_shear_force_hessian(
                X_wp,
                X_wc,
                X_wp_prev,
                X_wc_prev,
                parent_pose,
                child_pose,
                parent_com,
                child_com,
                is_parent_body,
                k_diag,
                C0_force_local,
                lambda_local,
                kd_diag,
                damping_active,
                dt,
            )
            total_force = total_force + f_l
            total_torque = total_torque + t_l
            total_H_ll = total_H_ll + Hll_l
            total_H_al = total_H_al + Hal_l
            total_H_aa = total_H_aa + Haa_l

        return total_force, total_torque, total_H_ll, total_H_al, total_H_aa

    P_I = wp.identity(3, float)

    # AL gating for non-rod structural slots.
    lin_lambda = wp.vec3(0.0)
    lin_C0 = wp.vec3(0.0)
    lin_alpha = float(0.0)
    lin_hard = joint_is_hard[c_start] == 1
    if lin_hard or joint_compliant_alm == 1:
        lin_lambda = joint_lambda_lin[joint_index]
        lin_C0 = joint_C0_lin[joint_index]
        lin_alpha = stab_alpha

    # BALL has no angular structural slot; other non-rod joints do.
    ang_lambda = wp.vec3(0.0)
    ang_C0 = wp.vec3(0.0)
    ang_alpha = float(0.0)
    ang_hard = jt != JointType.BALL and joint_is_hard[c_start + 1] == 1
    if jt != JointType.BALL and (ang_hard or joint_compliant_alm == 1):
        ang_lambda = joint_lambda_ang[joint_index]
        ang_C0 = joint_C0_ang[joint_index]
        ang_alpha = stab_alpha

    if jt == JointType.BALL:
        solve_weight = _load_solve_weight(joint_penalty_k, joint_rho, c_start, joint_compliant_alm)
        damping = joint_penalty_kd[c_start]
        if solve_weight > 0.0:
            return evaluate_linear_constraint_force_hessian(
                X_wp,
                X_wc,
                X_wp_prev,
                X_wc_prev,
                parent_pose,
                child_pose,
                parent_com,
                child_com,
                is_parent_body,
                solve_weight,
                joint_material_k[c_start],
                P_I,
                lin_lambda,
                lin_C0,
                lin_alpha,
                damping,
                joint_compliant_alm,
                dt,
            )
        return _zero_force_hessian()

    if parent_index >= 0:
        X_wp_rest = body_q_rest[parent_index] * X_pj
    else:
        X_wp_rest = X_pj
    X_wc_rest = body_q_rest[child_index] * X_cj
    q_wp_rest = wp.transform_get_rotation(X_wp_rest)
    q_wc_rest = wp.transform_get_rotation(X_wc_rest)

    if jt == JointType.FIXED:
        solve_weight_lin = _load_solve_weight(joint_penalty_k, joint_rho, c_start + 0, joint_compliant_alm)
        kd_lin = joint_penalty_kd[c_start + 0]
        if solve_weight_lin > 0.0:
            f_lin, t_lin, Hll_lin, Hal_lin, Haa_lin = evaluate_linear_constraint_force_hessian(
                X_wp,
                X_wc,
                X_wp_prev,
                X_wc_prev,
                parent_pose,
                child_pose,
                parent_com,
                child_com,
                is_parent_body,
                solve_weight_lin,
                joint_material_k[c_start + 0],
                P_I,
                lin_lambda,
                lin_C0,
                lin_alpha,
                kd_lin,
                joint_compliant_alm,
                dt,
            )
        else:
            f_lin = wp.vec3(0.0)
            t_lin = wp.vec3(0.0)
            Hll_lin = wp.mat33(0.0)
            Hal_lin = wp.mat33(0.0)
            Haa_lin = wp.mat33(0.0)

        solve_weight_ang = _load_solve_weight(joint_penalty_k, joint_rho, c_start + 1, joint_compliant_alm)
        kd_ang = joint_penalty_kd[c_start + 1]
        if solve_weight_ang > 0.0:
            t_ang, Haa_ang, _ang_kappa, _ang_J = evaluate_angular_constraint_force_hessian(
                q_wp,
                q_wc,
                q_wp_rest,
                q_wc_rest,
                q_wp_prev,
                q_wc_prev,
                is_parent_body,
                solve_weight_ang,
                joint_material_k[c_start + 1],
                P_I,
                ang_lambda,
                ang_C0,
                ang_alpha,
                kd_ang,
                joint_compliant_alm,
                dt,
            )
        else:
            t_ang = wp.vec3(0.0)
            Haa_ang = wp.mat33(0.0)

        return f_lin, t_lin + t_ang, Hll_lin, Hal_lin, Haa_lin + Haa_ang

    elif jt == JointType.REVOLUTE:
        qd_start = joint_qd_start[joint_index]
        P_lin, P_ang = build_joint_projectors(jt, joint_axis, qd_start, 0, 1, q_wp)
        a = wp.normalize(joint_axis[qd_start])

        solve_weight_lin = _load_solve_weight(joint_penalty_k, joint_rho, c_start + 0, joint_compliant_alm)
        kd_lin = joint_penalty_kd[c_start + 0]
        if solve_weight_lin > 0.0:
            f_lin, t_lin, Hll_lin, Hal_lin, Haa_lin = evaluate_linear_constraint_force_hessian(
                X_wp,
                X_wc,
                X_wp_prev,
                X_wc_prev,
                parent_pose,
                child_pose,
                parent_com,
                child_com,
                is_parent_body,
                solve_weight_lin,
                joint_material_k[c_start + 0],
                P_lin,
                lin_lambda,
                lin_C0,
                lin_alpha,
                kd_lin,
                joint_compliant_alm,
                dt,
            )
        else:
            f_lin = wp.vec3(0.0)
            t_lin = wp.vec3(0.0)
            Hll_lin = wp.mat33(0.0)
            Hal_lin = wp.mat33(0.0)
            Haa_lin = wp.mat33(0.0)

        solve_weight_ang = _load_solve_weight(joint_penalty_k, joint_rho, c_start + 1, joint_compliant_alm)
        kd_ang = joint_penalty_kd[c_start + 1]

        kappa_cached = wp.vec3(0.0)
        J_world_cached = wp.mat33(0.0)
        has_cached = False

        if solve_weight_ang > 0.0:
            t_ang, Haa_ang, kappa_cached, J_world_cached = evaluate_angular_constraint_force_hessian(
                q_wp,
                q_wc,
                q_wp_rest,
                q_wc_rest,
                q_wp_prev,
                q_wc_prev,
                is_parent_body,
                solve_weight_ang,
                joint_material_k[c_start + 1],
                P_ang,
                ang_lambda,
                ang_C0,
                ang_alpha,
                kd_ang,
                joint_compliant_alm,
                dt,
            )
            has_cached = True
        else:
            t_ang = wp.vec3(0.0)
            Haa_ang = wp.mat33(0.0)

        # Drive + limits on free angular DOF (constraint slot c_start + 2)
        dof_idx = qd_start
        axis_dl = _load_joint_axis_drive_limit(
            dof_idx,
            joint_target_q_start[joint_index],
            c_start + 2,
            joint_target_ke,
            joint_target_kd,
            joint_target_q,
            joint_target_qd,
            joint_limit_lower,
            joint_limit_upper,
            joint_limit_ke,
            joint_limit_kd,
            joint_penalty_k,
            joint_compliant_alm,
        )
        has_drive = _drive_row_applies_force(axis_dl.material_drive_ke, axis_dl.drive_kd)
        has_limits = _limit_row_exists(axis_dl.material_limit_ke, axis_dl.lower, axis_dl.upper)

        if has_drive or has_limits:
            inv_dt = 1.0 / dt

            if has_cached:
                kappa = kappa_cached
                J_world = J_world_cached
            else:
                kappa, J_world = compute_kappa_and_jacobian(q_wp, q_wc, q_wp_rest, q_wc_rest)

            theta = wp.dot(kappa, a)
            theta_abs = theta + joint_rest_angle[dof_idx]
            omega_p = quat_velocity(q_wp, q_wp_prev, dt)
            omega_c = quat_velocity(q_wc, q_wc_prev, dt)
            dkappa_dt = compute_kappa_dot(J_world, omega_p, omega_c)
            dtheta_dt = wp.dot(dkappa_dt, a)

            f_scalar, H_scalar = _eval_joint_axis_drive_limit(
                axis_dl,
                theta_abs,
                dtheta_dt,
                has_drive,
                has_limits,
                joint_drive_limit_support[dof_idx],
                joint_drive_lambda[dof_idx],
                joint_limit_lambda[dof_idx],
                joint_compliant_alm,
                inv_dt,
            )
            if H_scalar > 0.0:
                tau_drive, Haa_drive = apply_angular_drive_limit_torque(a, J_world, is_parent_body, f_scalar, H_scalar)
                t_ang = t_ang + tau_drive
                Haa_ang = Haa_ang + Haa_drive

        return f_lin, t_lin + t_ang, Hll_lin, Hal_lin, Haa_lin + Haa_ang

    elif jt == JointType.PRISMATIC:
        qd_start = joint_qd_start[joint_index]
        axis_local = joint_axis[qd_start]
        P_lin, P_ang = build_joint_projectors(jt, joint_axis, qd_start, 1, 0, q_wp)

        solve_weight_lin = _load_solve_weight(joint_penalty_k, joint_rho, c_start + 0, joint_compliant_alm)
        kd_lin = joint_penalty_kd[c_start + 0]
        if solve_weight_lin > 0.0:
            f_lin, t_lin, Hll_lin, Hal_lin, Haa_lin = evaluate_linear_constraint_force_hessian(
                X_wp,
                X_wc,
                X_wp_prev,
                X_wc_prev,
                parent_pose,
                child_pose,
                parent_com,
                child_com,
                is_parent_body,
                solve_weight_lin,
                joint_material_k[c_start + 0],
                P_lin,
                lin_lambda,
                lin_C0,
                lin_alpha,
                kd_lin,
                joint_compliant_alm,
                dt,
            )
        else:
            f_lin = wp.vec3(0.0)
            t_lin = wp.vec3(0.0)
            Hll_lin = wp.mat33(0.0)
            Hal_lin = wp.mat33(0.0)
            Haa_lin = wp.mat33(0.0)

        solve_weight_ang = _load_solve_weight(joint_penalty_k, joint_rho, c_start + 1, joint_compliant_alm)
        kd_ang = joint_penalty_kd[c_start + 1]
        if solve_weight_ang > 0.0:
            t_ang, Haa_ang, _ang_kappa, _ang_J = evaluate_angular_constraint_force_hessian(
                q_wp,
                q_wc,
                q_wp_rest,
                q_wc_rest,
                q_wp_prev,
                q_wc_prev,
                is_parent_body,
                solve_weight_ang,
                joint_material_k[c_start + 1],
                P_ang,
                ang_lambda,
                ang_C0,
                ang_alpha,
                kd_ang,
                joint_compliant_alm,
                dt,
            )
        else:
            t_ang = wp.vec3(0.0)
            Haa_ang = wp.mat33(0.0)

        # Drive + limits on free linear DOF (constraint slot c_start + 2)
        dof_idx = qd_start
        axis_dl = _load_joint_axis_drive_limit(
            dof_idx,
            joint_target_q_start[joint_index],
            c_start + 2,
            joint_target_ke,
            joint_target_kd,
            joint_target_q,
            joint_target_qd,
            joint_limit_lower,
            joint_limit_upper,
            joint_limit_ke,
            joint_limit_kd,
            joint_penalty_k,
            joint_compliant_alm,
        )
        has_drive = _drive_row_applies_force(axis_dl.material_drive_ke, axis_dl.drive_kd)
        has_limits = _limit_row_exists(axis_dl.material_limit_ke, axis_dl.lower, axis_dl.upper)

        if has_drive or has_limits:
            inv_dt = 1.0 / dt

            x_p = wp.transform_get_translation(X_wp)
            x_c = wp.transform_get_translation(X_wc)
            C_vec = x_c - x_p
            axis_w = wp.normalize(wp.quat_rotate(q_wp, axis_local))

            d_along = wp.dot(C_vec, axis_w)
            x_p_prev = wp.transform_get_translation(X_wp_prev)
            x_c_prev = wp.transform_get_translation(X_wc_prev)
            C_vec_prev = x_c_prev - x_p_prev
            dC_dt = (C_vec - C_vec_prev) * inv_dt
            dd_dt = wp.dot(dC_dt, axis_w)

            f_scalar, H_scalar = _eval_joint_axis_drive_limit(
                axis_dl,
                d_along,
                dd_dt,
                has_drive,
                has_limits,
                joint_drive_limit_support[dof_idx],
                joint_drive_lambda[dof_idx],
                joint_limit_lambda[dof_idx],
                joint_compliant_alm,
                inv_dt,
            )
            if H_scalar > 0.0:
                if is_parent_body:
                    com_w = wp.transform_point(parent_pose, parent_com)
                    r = x_p - com_w
                else:
                    com_w = wp.transform_point(child_pose, child_com)
                    r = x_c - com_w

                force_drive, torque_drive, Hll_drive, Hal_drive, Haa_drive = apply_linear_drive_limit_force(
                    axis_w, r, is_parent_body, f_scalar, H_scalar
                )

                f_lin = f_lin + force_drive
                t_lin = t_lin + torque_drive
                Hll_lin = Hll_lin + Hll_drive
                Hal_lin = Hal_lin + Hal_drive
                Haa_lin = Haa_lin + Haa_drive

        return f_lin, t_lin + t_ang, Hll_lin, Hal_lin, Haa_lin + Haa_ang

    elif jt == JointType.D6:
        lin_count = joint_dof_dim[joint_index, 0]
        ang_count = joint_dof_dim[joint_index, 1]
        qd_start = joint_qd_start[joint_index]

        P_lin, P_ang = build_joint_projectors(
            jt,
            joint_axis,
            qd_start,
            lin_count,
            ang_count,
            q_wp,
        )

        total_force = wp.vec3(0.0)
        total_torque = wp.vec3(0.0)
        total_H_ll = wp.mat33(0.0)
        total_H_al = wp.mat33(0.0)
        total_H_aa = wp.mat33(0.0)

        # Linear constraint (constrained when lin_count < 3)
        solve_weight_lin = _load_solve_weight(joint_penalty_k, joint_rho, c_start + 0, joint_compliant_alm)
        kd_lin = joint_penalty_kd[c_start + 0]

        if lin_count < 3 and solve_weight_lin > 0.0:
            f_l, t_l, Hll_l, Hal_l, Haa_l = evaluate_linear_constraint_force_hessian(
                X_wp,
                X_wc,
                X_wp_prev,
                X_wc_prev,
                parent_pose,
                child_pose,
                parent_com,
                child_com,
                is_parent_body,
                solve_weight_lin,
                joint_material_k[c_start + 0],
                P_lin,
                lin_lambda,
                lin_C0,
                lin_alpha,
                kd_lin,
                joint_compliant_alm,
                dt,
            )
            total_force = total_force + f_l
            total_torque = total_torque + t_l
            total_H_ll = total_H_ll + Hll_l
            total_H_al = total_H_al + Hal_l
            total_H_aa = total_H_aa + Haa_l

        # Angular constraint (constrained when ang_count < 3)
        solve_weight_ang = _load_solve_weight(joint_penalty_k, joint_rho, c_start + 1, joint_compliant_alm)
        kd_ang = joint_penalty_kd[c_start + 1]

        kappa_cached = wp.vec3(0.0)
        J_world_cached = wp.mat33(0.0)
        has_cached = False

        if ang_count < 3 and solve_weight_ang > 0.0:
            t_ang, Haa_ang, kappa_cached, J_world_cached = evaluate_angular_constraint_force_hessian(
                q_wp,
                q_wc,
                q_wp_rest,
                q_wc_rest,
                q_wp_prev,
                q_wc_prev,
                is_parent_body,
                solve_weight_ang,
                joint_material_k[c_start + 1],
                P_ang,
                ang_lambda,
                ang_C0,
                ang_alpha,
                kd_ang,
                joint_compliant_alm,
                dt,
            )
            has_cached = True

            total_torque = total_torque + t_ang
            total_H_aa = total_H_aa + Haa_ang

        # Linear drives/limits (per free linear DOF)
        if lin_count > 0:
            x_p = wp.transform_get_translation(X_wp)
            x_c = wp.transform_get_translation(X_wc)
            C_vec = x_c - x_p
            q_wp_rot = q_wp
            x_p_prev = wp.transform_get_translation(X_wp_prev)
            x_c_prev = wp.transform_get_translation(X_wc_prev)
            C_vec_prev = x_c_prev - x_p_prev
            inv_dt = 1.0 / dt
            dC_dt = (C_vec - C_vec_prev) * inv_dt

            if is_parent_body:
                com_w = wp.transform_point(parent_pose, parent_com)
                r_drive = x_p - com_w
            else:
                com_w = wp.transform_point(child_pose, child_com)
                r_drive = x_c - com_w

            target_q_base = joint_target_q_start[joint_index]
            for li in range(3):
                if li < lin_count:
                    dof_idx = qd_start + li
                    axis_dl = _load_joint_axis_drive_limit(
                        dof_idx,
                        target_q_base + li,
                        c_start + 2 + li,
                        joint_target_ke,
                        joint_target_kd,
                        joint_target_q,
                        joint_target_qd,
                        joint_limit_lower,
                        joint_limit_upper,
                        joint_limit_ke,
                        joint_limit_kd,
                        joint_penalty_k,
                        joint_compliant_alm,
                    )
                    has_drive = _drive_row_applies_force(axis_dl.material_drive_ke, axis_dl.drive_kd)
                    has_limits = _limit_row_exists(axis_dl.material_limit_ke, axis_dl.lower, axis_dl.upper)

                    if has_drive or has_limits:
                        axis_w = wp.normalize(wp.quat_rotate(q_wp_rot, joint_axis[dof_idx]))
                        d_along = wp.dot(C_vec, axis_w)
                        dd_dt = wp.dot(dC_dt, axis_w)

                        f_scalar, H_scalar = _eval_joint_axis_drive_limit(
                            axis_dl,
                            d_along,
                            dd_dt,
                            has_drive,
                            has_limits,
                            joint_drive_limit_support[dof_idx],
                            joint_drive_lambda[dof_idx],
                            joint_limit_lambda[dof_idx],
                            joint_compliant_alm,
                            inv_dt,
                        )
                        if H_scalar > 0.0:
                            force_drive, torque_drive, Hll_drive, Hal_drive, Haa_drive = apply_linear_drive_limit_force(
                                axis_w, r_drive, is_parent_body, f_scalar, H_scalar
                            )

                            total_force = total_force + force_drive
                            total_torque = total_torque + torque_drive
                            total_H_ll = total_H_ll + Hll_drive
                            total_H_al = total_H_al + Hal_drive
                            total_H_aa = total_H_aa + Haa_drive

        # Angular drives/limits (per free angular DOF)
        if ang_count > 0:
            inv_dt = 1.0 / dt

            if has_cached:
                kappa = kappa_cached
                J_world = J_world_cached
            else:
                kappa, J_world = compute_kappa_and_jacobian(q_wp, q_wc, q_wp_rest, q_wc_rest)

            omega_p = quat_velocity(q_wp, q_wp_prev, dt)
            omega_c = quat_velocity(q_wc, q_wc_prev, dt)
            dkappa_dt = compute_kappa_dot(J_world, omega_p, omega_c)

            target_q_base = joint_target_q_start[joint_index]
            for ai in range(3):
                if ai < ang_count:
                    dof_idx = qd_start + lin_count + ai
                    axis_dl = _load_joint_axis_drive_limit(
                        dof_idx,
                        target_q_base + lin_count + ai,
                        c_start + 2 + lin_count + ai,
                        joint_target_ke,
                        joint_target_kd,
                        joint_target_q,
                        joint_target_qd,
                        joint_limit_lower,
                        joint_limit_upper,
                        joint_limit_ke,
                        joint_limit_kd,
                        joint_penalty_k,
                        joint_compliant_alm,
                    )
                    has_drive = _drive_row_applies_force(axis_dl.material_drive_ke, axis_dl.drive_kd)
                    has_limits = _limit_row_exists(axis_dl.material_limit_ke, axis_dl.lower, axis_dl.upper)

                    if has_drive or has_limits:
                        a = wp.normalize(joint_axis[dof_idx])
                        theta = wp.dot(kappa, a)
                        theta_abs = theta + joint_rest_angle[dof_idx]
                        dtheta_dt = wp.dot(dkappa_dt, a)

                        f_scalar, H_scalar = _eval_joint_axis_drive_limit(
                            axis_dl,
                            theta_abs,
                            dtheta_dt,
                            has_drive,
                            has_limits,
                            joint_drive_limit_support[dof_idx],
                            joint_drive_lambda[dof_idx],
                            joint_limit_lambda[dof_idx],
                            joint_compliant_alm,
                            inv_dt,
                        )
                        if H_scalar > 0.0:
                            tau_drive, Haa_drive = apply_angular_drive_limit_torque(
                                a, J_world, is_parent_body, f_scalar, H_scalar
                            )
                            total_torque = total_torque + tau_drive
                            total_H_aa = total_H_aa + Haa_drive

        return total_force, total_torque, total_H_ll, total_H_al, total_H_aa

    return _zero_force_hessian()


# -----------------------------
# Utility kernels
# -----------------------------
@wp.func
def _reset_joint_history(
    joint: int,
    joint_constraint_start: wp.array[wp.int32],
    joint_constraint_dim: wp.array[wp.int32],
    joint_penalty_k_min: wp.array[float],
    joint_penalty_k: wp.array[float],
    joint_C0_lin: wp.array[wp.vec3],
    joint_C0_ang: wp.array[wp.vec3],
    joint_lambda_lin: wp.array[wp.vec3],
    joint_lambda_ang: wp.array[wp.vec3],
):
    """Reset immediately available joint solver history.

    The rod-joint Dahl friction state (kappa/sigma/increment: curvature,
    hysteretic stress, and increment) is left untouched here: an enabled rod
    joint rebaselines it from the next pre-step pose, while a disabled rod joint
    refreshes it in the end-of-step finalizer. DOF-indexed drive/limit state is
    reset by the caller.
    """
    constraint_start = joint_constraint_start[joint]
    constraint_dim = joint_constraint_dim[joint]
    for slot in range(constraint_dim):
        constraint = constraint_start + slot
        joint_penalty_k[constraint] = joint_penalty_k_min[constraint]
    joint_C0_lin[joint] = wp.vec3(0.0)
    joint_C0_ang[joint] = wp.vec3(0.0)
    joint_lambda_lin[joint] = wp.vec3(0.0)
    joint_lambda_ang[joint] = wp.vec3(0.0)


@wp.kernel(enable_backward=False)
def reset_rigid_state(
    world_mask: wp.array[wp.bool],
    reset_all: bool,
    world_count: int,
    body_world: wp.array[wp.int32],
    joint_world: wp.array[wp.int32],
    joint_constraint_start: wp.array[wp.int32],
    joint_constraint_dim: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_dof_dim: wp.array2d[wp.int32],
    model_body_q: wp.array[wp.transform],
    model_body_qd: wp.array[wp.spatial_vector],
    joint_penalty_k_min: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    joint_penalty_k: wp.array[float],
    joint_C0_lin: wp.array[wp.vec3],
    joint_C0_ang: wp.array[wp.vec3],
    joint_lambda_lin: wp.array[wp.vec3],
    joint_lambda_ang: wp.array[wp.vec3],
    joint_drive_lambda: wp.array[float],
    joint_limit_lambda: wp.array[float],
    rigid_pose_rebaseline_mask: wp.array[wp.bool],
    contact_history_reset_mask: wp.array[wp.bool],
    contact_history_reset_pending: wp.array[wp.int32],
):
    """Apply rigid resets in parallel by mask slot, body, and joint."""
    tid = wp.tid()

    if tid < world_count + 1:
        world = tid
        if tid == world_count:
            world = -1
        select_slot = _reset_world_selected(world, world_mask, reset_all, world_count)
        if select_slot:
            rigid_pose_rebaseline_mask[tid] = True
            # Contact-reset state is allocated only with contact warm-starting on.
            if contact_history_reset_mask:
                contact_history_reset_mask[tid] = True
                if reset_all:
                    if tid == world_count:
                        contact_history_reset_pending[0] = 1
                else:
                    wp.atomic_max(contact_history_reset_pending, 0, 1)

    # A non-null output is the caller's request to reset that field.
    if (body_q or body_qd) and tid < body_world.shape[0]:
        body_selected = _reset_world_selected(body_world[tid], world_mask, reset_all, world_count)
        if body_selected:
            if body_q:
                body_q[tid] = model_body_q[tid]
            if body_qd:
                body_qd[tid] = model_body_qd[tid]

    if tid < joint_world.shape[0]:
        joint_selected = _reset_world_selected(joint_world[tid], world_mask, reset_all, world_count)
        if joint_selected:
            _reset_joint_history(
                tid,
                joint_constraint_start,
                joint_constraint_dim,
                joint_penalty_k_min,
                joint_penalty_k,
                joint_C0_lin,
                joint_C0_ang,
                joint_lambda_lin,
                joint_lambda_ang,
            )
            dof_start = joint_qd_start[tid]
            dof_count = joint_dof_dim[tid, 0] + joint_dof_dim[tid, 1]
            for offset in range(dof_count):
                dof = dof_start + offset
                joint_drive_lambda[dof] = 0.0
                joint_limit_lambda[dof] = 0.0


@wp.kernel
def _count_num_adjacent_joints(
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    num_body_adjacent_joints: wp.array[wp.int32],
):
    joint_count = joint_parent.shape[0]
    for joint_id in range(joint_count):
        parent_id = joint_parent[joint_id]
        child_id = joint_child[joint_id]

        # Skip world joints (parent/child == -1)
        if parent_id >= 0:
            num_body_adjacent_joints[parent_id] = num_body_adjacent_joints[parent_id] + 1
        if child_id >= 0:
            num_body_adjacent_joints[child_id] = num_body_adjacent_joints[child_id] + 1


@wp.kernel
def _fill_adjacent_joints(
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    body_adjacent_joints_offsets: wp.array[wp.int32],
    body_adjacent_joints_fill_count: wp.array[wp.int32],
    body_adjacent_joints: wp.array[wp.int32],
):
    joint_count = joint_parent.shape[0]
    for joint_id in range(joint_count):
        parent_id = joint_parent[joint_id]
        child_id = joint_child[joint_id]

        # Add joint to parent body's adjacency list
        if parent_id >= 0:
            fill_count_parent = body_adjacent_joints_fill_count[parent_id]
            buffer_offset_parent = body_adjacent_joints_offsets[parent_id]
            body_adjacent_joints[buffer_offset_parent + fill_count_parent] = joint_id
            body_adjacent_joints_fill_count[parent_id] = fill_count_parent + 1

        # Add joint to child body's adjacency list
        if child_id >= 0:
            fill_count_child = body_adjacent_joints_fill_count[child_id]
            buffer_offset_child = body_adjacent_joints_offsets[child_id]
            body_adjacent_joints[buffer_offset_child + fill_count_child] = joint_id
            body_adjacent_joints_fill_count[child_id] = fill_count_child + 1


@wp.kernel
def refresh_body_structural_k(
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_constraint_start: wp.array[wp.int32],
    joint_constraint_dim: wp.array[wp.int32],
    joint_material_k: wp.array[float],
    body_structural_k: wp.array[float],
):
    """Accumulate enabled linear-joint stiffness into a caller-zeroed output.

    Uses the first constraint slot for ordinary joints. Rod joints take
    ``max(stretch, shear)`` so independently authored shear is not ignored.
    """
    joint_id = wp.tid()
    if not joint_enabled[joint_id] or joint_constraint_dim[joint_id] <= 0:
        return

    c_start = joint_constraint_start[joint_id]
    k_linear = joint_material_k[c_start]
    if joint_type[joint_id] == JointType.ROD and joint_constraint_dim[joint_id] > 1:
        k_linear = wp.max(k_linear, joint_material_k[c_start + 1])
    if k_linear > 0.0:
        parent_id = joint_parent[joint_id]
        child_id = joint_child[joint_id]
        if parent_id >= 0:
            wp.atomic_max(body_structural_k, parent_id, k_linear)
        if child_id >= 0:
            wp.atomic_max(body_structural_k, child_id, k_linear)


# -----------------------------
# Pre-iteration kernels (once per step)
# -----------------------------
@wp.kernel
def forward_step_rigid_bodies(
    # Inputs
    dt: float,
    gravity: wp.array[wp.vec3],
    body_world: wp.array[wp.int32],
    pose_rebaseline_mask: wp.array[wp.bool],
    body_f: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inertia: wp.array[wp.mat33],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    body_inertia_q: wp.array[wp.transform],
):
    """
    Forward integration step for rigid bodies in SolverVBD.

    Args:
        dt: Time step [s].
        gravity: Gravity vector array (world frame).
        body_world: World index for each body.
        pose_rebaseline_mask: Per-world flags for the ``body_q_prev`` rebaseline below.
        body_f: External forces on bodies (spatial wrenches, world frame).
        body_com: Centers of mass (local body frame).
        body_inertia: Inertia tensors (local body frame).
        body_inv_mass: Inverse masses (0 for kinematic bodies).
        body_inv_inertia: Inverse inertia tensors (local body frame).
        body_q: Body transforms (input: start-of-step pose, output: integrated pose).
        body_qd: Body velocities (input: start-of-step velocity, output: integrated velocity).
        body_q_prev: Previous body transforms (output). Rebaselined to the current
            pre-step pose for worlds selected by ``pose_rebaseline_mask`` (first step
            after construction or reset); left unchanged otherwise.
        body_inertia_q: Inertial target body transforms for the AVBD solve (output).
    """
    tid = wp.tid()

    world_idx = body_world[tid]
    q_current = body_q[tid]
    if _world_selected(world_idx, pose_rebaseline_mask):
        # The constructor or reset() may precede state pose preparation.
        body_q_prev[tid] = q_current

    # Early exit for kinematic bodies (inv_mass == 0).
    inv_m = body_inv_mass[tid]
    if inv_m == 0.0:
        body_inertia_q[tid] = q_current
        return

    # Read body state (only for dynamic bodies)
    qd_current = body_qd[tid]
    f_current = body_f[tid]
    com_local = body_com[tid]
    I_local = body_inertia[tid]
    inv_I = body_inv_inertia[tid]
    world_g = gravity[world_idx]

    # Integrate rigid body motion (semi-implicit Euler, no angular damping)
    q_new, qd_new = integrate_rigid_body(
        q_current,
        qd_current,
        f_current,
        com_local,
        I_local,
        inv_m,
        inv_I,
        world_g,
        0.0,  # angular_damping = 0 (consistent with particle VBD)
        dt,
    )

    # Update current transform, velocity, and set inertial target
    body_q[tid] = q_new
    body_qd[tid] = qd_new
    body_inertia_q[tid] = q_new


@wp.kernel
def build_body_body_contact_lists(
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[int],
    rigid_contact_shape1: wp.array[int],
    shape_body: wp.array[wp.int32],
    body_inv_mass_effective: wp.array[float],
    body_contact_buffer_pre_alloc: int,
    body_contact_counts: wp.array[wp.int32],
    body_contact_indices: wp.array[wp.int32],
    body_contact_overflow_max: wp.array[wp.int32],
):
    """
    Build per-body contact lists for body-centric per-color contact evaluation.

    Each contact is listed only under its dynamic bodies (effective inverse
    mass > 0); static/kinematic bodies are skipped since VBD never moves them.
    Overflow is tracked in body_contact_overflow_max for diagnostics.
    """
    t_id = wp.tid()
    if t_id >= rigid_contact_count[0]:
        return

    s0 = rigid_contact_shape0[t_id]
    s1 = rigid_contact_shape1[t_id]
    b0 = shape_body[s0] if s0 >= 0 else -1
    b1 = shape_body[s1] if s1 >= 0 else -1

    if b0 >= 0 and body_inv_mass_effective[b0] > 0.0:
        idx = wp.atomic_add(body_contact_counts, b0, 1)
        if idx < body_contact_buffer_pre_alloc:
            body_contact_indices[b0 * body_contact_buffer_pre_alloc + idx] = t_id
        else:
            wp.atomic_max(body_contact_overflow_max, 0, idx + 1)

    if b1 >= 0 and body_inv_mass_effective[b1] > 0.0:
        idx = wp.atomic_add(body_contact_counts, b1, 1)
        if idx < body_contact_buffer_pre_alloc:
            body_contact_indices[b1 * body_contact_buffer_pre_alloc + idx] = t_id
        else:
            wp.atomic_max(body_contact_overflow_max, 0, idx + 1)


@wp.kernel
def build_body_particle_contact_lists(
    body_particle_contact_count: wp.array[int],
    body_particle_contact_shape: wp.array[int],
    shape_body: wp.array[wp.int32],
    body_inv_mass_effective: wp.array[float],
    body_particle_contact_buffer_pre_alloc: int,
    body_particle_contact_counts: wp.array[wp.int32],
    body_particle_contact_indices: wp.array[wp.int32],
    body_particle_contact_overflow_max: wp.array[wp.int32],
):
    """
    Build per-body contact lists for body-particle contacts.

    Each contact is listed only if its body is dynamic (effective inverse
    mass > 0); static/kinematic bodies are skipped since VBD never moves them.
    Overflow is tracked in body_particle_contact_overflow_max for diagnostics.
    """
    tid = wp.tid()
    # Bucket every soft contact (particle + edge + face; single total count) by its rigid body, so
    # the per-body kernel drives all reactions from one adjacency list.
    if tid >= body_particle_contact_count[0]:
        return

    shape = body_particle_contact_shape[tid]
    body = shape_body[shape] if shape >= 0 else -1

    if body < 0 or body_inv_mass_effective[body] <= 0.0:
        return

    idx = wp.atomic_add(body_particle_contact_counts, body, 1)
    if idx < body_particle_contact_buffer_pre_alloc:
        body_particle_contact_indices[body * body_particle_contact_buffer_pre_alloc + idx] = tid
    else:
        wp.atomic_max(body_particle_contact_overflow_max, 0, idx + 1)


@wp.kernel
def check_contact_overflow(
    overflow_max: wp.array[wp.int32],
    buffer_size: int,
    contact_type: int,
):
    """Print a warning if per-body contact buffer overflowed. Launched with dim=1."""
    omax = overflow_max[0]
    if omax > buffer_size:
        if contact_type == 0:
            wp.printf(
                "Warning: Per-body rigid contact buffer overflowed %d > %d.\n",
                omax,
                buffer_size,
            )
        else:
            wp.printf(
                "Warning: Per-body particle contact buffer overflowed %d > %d.\n",
                omax,
                buffer_size,
            )


@wp.func
def _body_angular_mobility_scalar(body: int, body_inv_mass: wp.array[float], body_inv_inertia: wp.array[wp.mat33]):
    """Return the direction-averaged angular mobility of one dynamic body."""
    if body < 0 or body_inv_mass[body] <= 0.0:
        return 0.0

    inv_I = body_inv_inertia[body]
    return wp.trace(inv_I) / 3.0


@wp.func
def _joint_linear_rho_seed(
    parent: int,
    child: int,
    body_inv_mass: wp.array[float],
    inv_dt_sq: float,
):
    """Reduced-mass scalar seed; deliberately excludes anchor-arm mobility."""
    inv_m_eff = float(0.0)
    if parent >= 0:
        inv_m_eff = inv_m_eff + body_inv_mass[parent]
    if child >= 0:
        inv_m_eff = inv_m_eff + body_inv_mass[child]

    if inv_m_eff > 0.0:
        return inv_dt_sq / inv_m_eff
    return 0.0


@wp.func
def _joint_angular_rho_seed(
    parent: int,
    child: int,
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    inv_dt_sq: float,
):
    """Mean angular-mobility scalar shared by all active angular directions."""
    angular_mobility = float(0.0)
    angular_mobility = angular_mobility + _body_angular_mobility_scalar(parent, body_inv_mass, body_inv_inertia)
    angular_mobility = angular_mobility + _body_angular_mobility_scalar(child, body_inv_mass, body_inv_inertia)

    if angular_mobility > 0.0:
        return inv_dt_sq / angular_mobility
    return 0.0


@wp.func
def _joint_axis_linear_support(
    parent: int,
    child: int,
    parent_pose: wp.transform,
    child_pose: wp.transform,
    parent_anchor: wp.transform,
    child_anchor: wp.transform,
    axis_world: wp.vec3,
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    inv_dt_sq: float,
):
    """Return the scalar inverse-Delassus support of a translational axis."""
    w = 0.0
    if parent >= 0 and body_inv_mass[parent] > 0.0:
        axis_local = wp.quat_rotate_inv(wp.transform_get_rotation(parent_pose), axis_world)
        lever_local = wp.transform_get_translation(parent_anchor) - body_com[parent]
        moment_local = wp.cross(lever_local, axis_local)
        w = body_inv_mass[parent] + wp.dot(moment_local, body_inv_inertia[parent] * moment_local)
    if child >= 0 and body_inv_mass[child] > 0.0:
        axis_local = wp.quat_rotate_inv(wp.transform_get_rotation(child_pose), axis_world)
        lever_local = wp.transform_get_translation(child_anchor) - body_com[child]
        moment_local = wp.cross(lever_local, axis_local)
        w = w + body_inv_mass[child] + wp.dot(moment_local, body_inv_inertia[child] * moment_local)
    if w > 0.0:
        return inv_dt_sq / w
    return 0.0


@wp.func
def _joint_axis_angular_support(
    parent: int,
    child: int,
    parent_pose: wp.transform,
    child_pose: wp.transform,
    angular_gradient_world: wp.vec3,
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    inv_dt_sq: float,
):
    """Return the scalar inverse-Delassus support of an angular axis."""
    w = 0.0
    if parent >= 0 and body_inv_mass[parent] > 0.0:
        gradient_local = wp.quat_rotate_inv(wp.transform_get_rotation(parent_pose), angular_gradient_world)
        w = wp.dot(gradient_local, body_inv_inertia[parent] * gradient_local)
    if child >= 0 and body_inv_mass[child] > 0.0:
        gradient_local = wp.quat_rotate_inv(wp.transform_get_rotation(child_pose), angular_gradient_world)
        w = w + wp.dot(gradient_local, body_inv_inertia[child] * gradient_local)
    if w > 0.0:
        return inv_dt_sq / w
    return 0.0


@wp.kernel
def init_rod_rest_bend_twist(
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    body_q_rest: wp.array[wp.transform],
    joint_rod_rest_kb_local: wp.array[wp.vec3],
    joint_rod_rest_twist: wp.array[float],
):
    """Precompute rod-joint rest angular deformation invariants."""
    j = wp.tid()
    joint_rod_rest_kb_local[j] = wp.vec3(0.0)
    joint_rod_rest_twist[j] = 0.0

    if joint_type[j] != JointType.ROD:
        return

    child = joint_child[j]
    if child < 0:
        return

    parent = joint_parent[j]
    if parent >= 0:
        X_wp_rest = body_q_rest[parent] * joint_X_p[j]
    else:
        X_wp_rest = joint_X_p[j]
    X_wc_rest = body_q_rest[child] * joint_X_c[j]

    q_wp_rest = wp.transform_get_rotation(X_wp_rest)
    q_wc_rest = wp.transform_get_rotation(X_wc_rest)

    # Rest DER bend (parent-local curvature binormal) and rest twist (transported
    # material spin), measured once so a pre-curved rest yields zero strain.
    rest_measure = _measure_rod_bend_twist_z(q_wp_rest, q_wc_rest)
    joint_rod_rest_kb_local[j] = wp.quat_rotate(wp.quat_inverse(q_wp_rest), rest_measure.kb_world)
    joint_rod_rest_twist[j] = rest_measure.twist


@wp.kernel
def step_joint_C0_lambda_rho(
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_rod_rest_kb_local: wp.array[wp.vec3],
    joint_rod_rest_twist: wp.array[float],
    body_q_prev: wp.array[wp.transform],
    body_q_rest: wp.array[wp.transform],
    joint_constraint_start: wp.array[wp.int32],
    joint_constraint_dim: wp.array[wp.int32],
    joint_is_hard: wp.array[wp.int32],
    joint_compliant_alm: int,
    lambda_retention: float,
    penalty_decay: float,
    joint_penalty_k_min: wp.array[float],
    joint_material_k: wp.array[float],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_limit_ke: wp.array[float],
    inv_dt_sq: float,
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    joint_penalty_k: wp.array[float],
    joint_C0_lin: wp.array[wp.vec3],
    joint_C0_ang: wp.array[wp.vec3],
    joint_lambda_lin: wp.array[wp.vec3],
    joint_lambda_ang: wp.array[wp.vec3],
    joint_rho: wp.array[float],
    joint_drive_limit_support: wp.array[float],
    joint_drive_lambda: wp.array[float],
    joint_limit_lambda: wp.array[float],
):
    """Once-per-step joint setup before the iteration loop (dim = joint_count).

    Sole owner of all per-step joint maintenance:
      1. penalty-k decay (runs even for disabled joints);
      2. compliant-ALM auto-``rho`` refresh for structural rows;
      3. directional support + multiplier retention for drive/limit rows;
      4. C0 snapshot + lambda retention for stabilized structural rows.

    Bilateral structural and drive auto-rho keep ``k_eff >= 0.9K`` in each local
    solve. Limits use their directional support directly as the ALM metric.
    """
    j = wp.tid()
    zero = wp.vec3(0.0)
    c_start = int(joint_constraint_start[j])
    c_dim = int(joint_constraint_dim[j])
    child = joint_child[j]
    parent = joint_parent[j]

    # 1. Penalty-k decay runs unconditionally (even for disabled joints).
    for s in range(c_dim):
        idx = c_start + s
        joint_penalty_k[idx] = wp.clamp(
            penalty_decay * joint_penalty_k[idx], joint_penalty_k_min[idx], joint_material_k[idx]
        )

    # 2. Auto-rho refresh under compliant ALM. Unconditional like the decay above
    #    (no enabled gate, to avoid a divergent branch); a disabled joint's rho is unused.
    if joint_compliant_alm == 1 and c_dim > 0 and child >= 0:
        jt = joint_type[j]
        if jt == JointType.ROD:
            linear_support = _joint_linear_rho_seed(parent, child, body_inv_mass, inv_dt_sq)
            angular_support = _joint_angular_rho_seed(parent, child, body_inv_mass, body_inv_inertia, inv_dt_sq)
            joint_rho[c_start] = _bilateral_auto_rho(linear_support, joint_material_k[c_start])
            joint_rho[c_start + 1] = _bilateral_auto_rho(linear_support, joint_material_k[c_start + 1])
            joint_rho[c_start + 2] = _bilateral_auto_rho(angular_support, joint_material_k[c_start + 2])
            joint_rho[c_start + 3] = _bilateral_auto_rho(angular_support, joint_material_k[c_start + 3])
        else:
            if joint_material_k[c_start] > 0.0:
                linear_support = _joint_linear_rho_seed(parent, child, body_inv_mass, inv_dt_sq)
                joint_rho[c_start] = _structural_joint_auto_rho(linear_support, joint_material_k[c_start])

            if c_dim > 1 and joint_material_k[c_start + 1] > 0.0:
                angular_support = _joint_angular_rho_seed(parent, child, body_inv_mass, body_inv_inertia, inv_dt_sq)
                joint_rho[c_start + 1] = _structural_joint_auto_rho(angular_support, joint_material_k[c_start + 1])

    # 3. Free DOFs use their exact scalar support; structural vec3 slots use
    #    the isotropic seeds above.
    if joint_compliant_alm == 1:
        jt = joint_type[j]
        linear_count = 0
        angular_count = 0
        if jt == JointType.PRISMATIC:
            linear_count = 1
        elif jt == JointType.REVOLUTE:
            angular_count = 1
        elif jt == JointType.D6:
            linear_count = joint_dof_dim[j, 0]
            angular_count = joint_dof_dim[j, 1]

        qd_start = joint_qd_start[j]
        has_material_row = bool(False)
        has_angular_material_row = bool(False)
        for axis in range(linear_count + angular_count):
            dof = qd_start + axis
            joint_drive_limit_support[dof] = 0.0
            if _drive_limit_needs_support(
                dof,
                joint_target_ke,
                joint_target_kd,
                joint_limit_lower,
                joint_limit_upper,
                joint_limit_ke,
            ):
                has_material_row = True
                if axis >= linear_count:
                    has_angular_material_row = True
                if joint_enabled[j] and child >= 0:
                    joint_drive_lambda[dof] = lambda_retention * joint_drive_lambda[dof]
                    joint_limit_lambda[dof] = lambda_retention * joint_limit_lambda[dof]
                else:
                    joint_drive_lambda[dof] = 0.0
                    joint_limit_lambda[dof] = 0.0
            else:
                joint_drive_lambda[dof] = 0.0
                joint_limit_lambda[dof] = 0.0

        if joint_enabled[j] and child >= 0 and has_material_row:
            parent_pose = wp.transform_identity()
            if parent >= 0:
                parent_pose = body_q_prev[parent]
            child_pose = body_q_prev[child]
            X_wp = parent_pose * joint_X_p[j]
            X_wc = child_pose * joint_X_c[j]
            q_wp = wp.transform_get_rotation(X_wp)
            q_wc = wp.transform_get_rotation(X_wc)
            for axis in range(linear_count):
                dof = qd_start + axis
                if _drive_limit_needs_support(
                    dof,
                    joint_target_ke,
                    joint_target_kd,
                    joint_limit_lower,
                    joint_limit_upper,
                    joint_limit_ke,
                ):
                    axis_world = wp.normalize(wp.quat_rotate(q_wp, joint_axis[dof]))
                    joint_drive_limit_support[dof] = _joint_axis_linear_support(
                        parent,
                        child,
                        parent_pose,
                        child_pose,
                        joint_X_p[j],
                        joint_X_c[j],
                        axis_world,
                        body_com,
                        body_inv_mass,
                        body_inv_inertia,
                        inv_dt_sq,
                    )

            if has_angular_material_row:
                q_wp_rest = wp.transform_get_rotation(
                    (body_q_rest[parent] * joint_X_p[j]) if parent >= 0 else joint_X_p[j]
                )
                q_wc_rest = wp.transform_get_rotation(body_q_rest[child] * joint_X_c[j])
                _kappa, angular_jacobian_world = compute_kappa_and_jacobian(q_wp, q_wc, q_wp_rest, q_wc_rest)
                for axis in range(angular_count):
                    dof = qd_start + linear_count + axis
                    if _drive_limit_needs_support(
                        dof,
                        joint_target_ke,
                        joint_target_kd,
                        joint_limit_lower,
                        joint_limit_upper,
                        joint_limit_ke,
                    ):
                        gradient_world = angular_jacobian_world * wp.normalize(joint_axis[dof])
                        joint_drive_limit_support[dof] = _joint_axis_angular_support(
                            parent,
                            child,
                            parent_pose,
                            child_pose,
                            gradient_world,
                            body_inv_mass,
                            body_inv_inertia,
                            inv_dt_sq,
                        )

    # 4. C0 snapshot + lambda retention for stabilized structural slots.
    if not joint_enabled[j] or c_dim == 0 or child < 0:
        joint_C0_lin[j] = zero
        joint_lambda_lin[j] = zero
        joint_C0_ang[j] = zero
        joint_lambda_ang[j] = zero
        return

    jt = joint_type[j]

    # Rod joints have four structural slots, but AL state is stored as two vec3
    # blocks: linear = stretch/shear, angular = bend/twist.
    if jt == JointType.ROD:
        stretch_idx = c_start
        shear_idx = c_start + 1
        bend_idx = c_start + 2
        twist_idx = c_start + 3

        has_linear = int(0)
        has_angular = int(0)
        if joint_is_hard[stretch_idx] == 1 or joint_is_hard[shear_idx] == 1:
            has_linear = 1
        if joint_is_hard[bend_idx] == 1 or joint_is_hard[twist_idx] == 1:
            has_angular = 1
        if joint_compliant_alm == 1:
            if joint_material_k[stretch_idx] > 0.0 or joint_material_k[shear_idx] > 0.0:
                has_linear = 1
            if joint_material_k[bend_idx] > 0.0 or joint_material_k[twist_idx] > 0.0:
                has_angular = 1

        if has_linear == 0 and has_angular == 0:
            joint_C0_lin[j] = zero
            joint_lambda_lin[j] = zero
            joint_C0_ang[j] = zero
            joint_lambda_ang[j] = zero
            return

        if parent >= 0:
            X_wp = body_q_prev[parent] * joint_X_p[j]
        else:
            X_wp = joint_X_p[j]
        X_wc = body_q_prev[child] * joint_X_c[j]

        if has_linear == 1:
            x_p = wp.transform_get_translation(X_wp)
            x_c = wp.transform_get_translation(X_wc)
            # Parent-material residual: [shear_x, shear_y, stretch_z].
            joint_C0_lin[j] = wp.quat_rotate_inv(wp.transform_get_rotation(X_wp), x_c - x_p)
            joint_lambda_lin[j] = joint_lambda_lin[j] * lambda_retention
        else:
            joint_C0_lin[j] = zero
            joint_lambda_lin[j] = zero

        if has_angular == 1:
            q_wp = wp.transform_get_rotation(X_wp)
            q_wc = wp.transform_get_rotation(X_wc)
            joint_C0_ang[j] = compute_geometric_rod_kappa_cached_z(
                q_wp,
                q_wc,
                joint_rod_rest_kb_local[j],
                joint_rod_rest_twist[j],
            )
            joint_lambda_ang[j] = joint_lambda_ang[j] * lambda_retention
        else:
            joint_C0_ang[j] = zero
            joint_lambda_ang[j] = zero
        return

    # Non-rod joints have at most two structural slots here; drive/limit
    # slots are maintained separately.
    has_linear = int(0)
    has_angular = int(0)
    if joint_is_hard[c_start] == 1:
        has_linear = 1
    if c_dim > 1 and joint_is_hard[c_start + 1] == 1:
        has_angular = 1
    if joint_compliant_alm == 1:
        if joint_material_k[c_start] > 0.0:
            has_linear = 1
        if c_dim > 1 and joint_material_k[c_start + 1] > 0.0:
            has_angular = 1

    if has_linear == 0 and has_angular == 0:
        joint_C0_lin[j] = zero
        joint_lambda_lin[j] = zero
        joint_C0_ang[j] = zero
        joint_lambda_ang[j] = zero
        return

    if parent >= 0:
        X_wp = body_q_prev[parent] * joint_X_p[j]
    else:
        X_wp = joint_X_p[j]
    X_wc = body_q_prev[child] * joint_X_c[j]

    if has_linear == 1:
        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        joint_C0_lin[j] = x_c - x_p
        joint_lambda_lin[j] = joint_lambda_lin[j] * lambda_retention
    else:
        joint_C0_lin[j] = zero
        joint_lambda_lin[j] = zero

    if has_angular == 1:
        q_wp = wp.transform_get_rotation(X_wp)
        q_wc = wp.transform_get_rotation(X_wc)
        if parent >= 0:
            X_wp_rest = body_q_rest[parent] * joint_X_p[j]
        else:
            X_wp_rest = joint_X_p[j]
        X_wc_rest = body_q_rest[child] * joint_X_c[j]
        q_wp_rest = wp.transform_get_rotation(X_wp_rest)
        q_wc_rest = wp.transform_get_rotation(X_wc_rest)
        joint_C0_ang[j] = compute_kappa(q_wp, q_wc, q_wp_rest, q_wc_rest)
        joint_lambda_ang[j] = joint_lambda_ang[j] * lambda_retention
    else:
        joint_C0_ang[j] = zero
        joint_lambda_ang[j] = zero


@wp.kernel
def init_body_body_contact_materials(
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[int],
    rigid_contact_shape1: wp.array[int],
    shape_material_ke: wp.array[float],
    shape_material_kd: wp.array[float],
    shape_material_mu: wp.array[float],
    k_start: float,
    # Outputs
    contact_penalty_k: wp.array[float],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    contact_material_ke: wp.array[float],
):
    """Cold-start body-body contact penalties and cache material properties.

    Averages both shapes' material. Legacy penalty is seeded at
    ``min(k_start, avg_ke)`` when ramping (k_start >= 0) or at ``avg_ke`` when
    fixed-k (k_start < 0). Compliant-ALM contact keeps contact material
    stiffness as physical ``k``. Rho has a single owner: the per-step setup
    kernel ``step_body_body_contact_C0_lambda`` resolves it every step.
    """
    i = wp.tid()
    if i >= rigid_contact_count[0]:
        return

    shape_id_0 = rigid_contact_shape0[i]
    shape_id_1 = rigid_contact_shape1[i]
    if shape_id_0 < 0 or shape_id_1 < 0:
        return

    avg_ke, avg_kd, avg_mu = _average_contact_material(
        shape_material_ke[shape_id_0],
        shape_material_kd[shape_id_0],
        shape_material_mu[shape_id_0],
        shape_material_ke[shape_id_1],
        shape_material_kd[shape_id_1],
        shape_material_mu[shape_id_1],
    )

    contact_material_kd[i] = avg_kd
    contact_material_mu[i] = avg_mu
    contact_material_ke[i] = avg_ke

    contact_penalty_k[i] = _contact_penalty_floor(avg_ke, k_start)


@wp.kernel
def init_body_body_contacts_alm(
    # Dimensioning
    rigid_contact_count: wp.array[int],
    # Constraint data
    rigid_contact_shape0: wp.array[int],
    rigid_contact_shape1: wp.array[int],
    rigid_contact_normal: wp.array[wp.vec3],
    # Material
    shape_material_ke: wp.array[float],
    shape_material_kd: wp.array[float],
    shape_material_mu: wp.array[float],
    legacy_hard_contacts: int,
    contact_compliant_alm: int,
    restore_compliant_tangent_warmstart: int,
    # Pipeline-owned correspondence and VBD-owned cross-step state
    match_index: wp.array[wp.int32],
    history: RigidContactHistory,
    # Optional reset context; a null pending array disables masked invalidation.
    contact_history_reset_pending: wp.array[wp.int32],
    contact_history_reset_mask: wp.array[wp.bool],
    shape_world: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    # Scalar parameters
    k_start: float,
    # Outputs
    contact_penalty_k: wp.array[float],
    contact_lambda: wp.array[wp.vec3],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    contact_material_ke: wp.array[float],
):
    """Warm-start body-body contact state from match indices.

    ALM: always restore matched ``lambda_n``; with ``latest`` matching also
    restore cone-clamped ``lambda_t``. Sticky matching keeps tangent memory in
    the replayed material anchor (via C0), not in ``lambda_t``. Legacy hard:
    full-vector warm start. Legacy soft: ``penalty_k`` only. C0/decay live in
    :func:`step_body_body_contact_C0_lambda`.
    """
    i = wp.tid()
    if i >= rigid_contact_count[0]:
        return

    # Every unmatched row remains zero; matched multiplier paths overwrite it.
    contact_lambda[i] = wp.vec3(0.0)
    s0 = rigid_contact_shape0[i]
    s1 = rigid_contact_shape1[i]
    if s0 < 0 or s1 < 0:
        return

    avg_ke, avg_kd, avg_mu = _average_contact_material(
        shape_material_ke[s0],
        shape_material_kd[s0],
        shape_material_mu[s0],
        shape_material_ke[s1],
        shape_material_kd[s1],
        shape_material_mu[s1],
    )
    contact_material_ke[i] = avg_ke
    contact_material_kd[i] = avg_kd
    contact_material_mu[i] = avg_mu

    k_floor = _contact_penalty_floor(avg_ke, k_start)
    slot = match_index[i]
    # Drop the saved match for reset-selected worlds so they cold-start instead
    # of warm-starting from pre-reset history.
    if slot >= 0 and contact_history_reset_pending:
        if contact_history_reset_pending[0] != 0:
            if _contact_world_selected(s0, s1, shape_world, shape_body, body_world, contact_history_reset_mask):
                slot = -1

    if slot >= 0:
        contact_penalty_k[i] = wp.clamp(history.penalty_k[slot], k_floor, avg_ke)
        use_legacy_hard = legacy_hard_contacts == 1 and contact_compliant_alm == 0
        restore_multiplier = use_legacy_hard or contact_compliant_alm == 1
        if restore_multiplier:
            lam_hist = history.lambda_[slot]
            n_new = rigid_contact_normal[i]
            n_old = history.normal[slot]
            lam_n = wp.dot(lam_hist, n_old)
            lam_new = n_new * lam_n
            if use_legacy_hard or restore_compliant_tangent_warmstart == 1:
                lam_t_old = lam_hist - n_old * lam_n
                lam_t_new = lam_t_old - n_new * wp.dot(lam_t_old, n_new)
                if contact_compliant_alm == 1:
                    # Compliant ALM restores only a valid Coulomb-cone warm start.
                    lam_t_new = _project_coulomb_tangent(
                        lam_t_new,
                        wp.length(lam_t_new),
                        avg_mu * wp.max(lam_n, 0.0),
                    )
                lam_new += lam_t_new
            contact_lambda[i] = lam_new
    else:
        contact_penalty_k[i] = k_floor


@wp.kernel
def snapshot_body_body_contact_history(
    rigid_contact_count: wp.array[int],
    rigid_contact_normal: wp.array[wp.vec3],
    contact_lambda: wp.array[wp.vec3],
    contact_penalty_k: wp.array[float],
    # Persistent outputs, in RigidContactHistory order
    prev_lambda: wp.array[wp.vec3],
    prev_penalty_k: wp.array[float],
    prev_normal: wp.array[wp.vec3],
):
    """Snapshot post-iteration contact state by contact row.

    The next match_index refers to the rows written here, so VBD history is
    stored directly by contact row index.
    """
    i = wp.tid()
    if i >= rigid_contact_count[0]:
        return

    prev_lambda[i] = contact_lambda[i]
    prev_penalty_k[i] = contact_penalty_k[i]
    prev_normal[i] = rigid_contact_normal[i]


@wp.kernel
def step_body_body_contact_C0_lambda(
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[int],
    rigid_contact_shape1: wp.array[int],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_offset0: wp.array[wp.vec3],
    rigid_contact_offset1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_margin0: wp.array[float],
    rigid_contact_margin1: wp.array[float],
    shape_body: wp.array[int],
    body_flags: wp.array[int],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    body_com: wp.array[wp.vec3],
    body_structural_k: wp.array[float],
    proxy_flag: int,
    body_q: wp.array[wp.transform],
    legacy_hard_contacts: int,
    contact_compliant_alm: int,
    lambda_retention: float,
    inv_dt_sq: float,
    penalty_decay: float,
    contact_material_ke: wp.array[float],
    k_start: float,
    # In/out
    contact_normal_rho: wp.array[float],
    contact_penalty_k: wp.array[float],
    contact_C0: wp.array[wp.vec3],
    contact_lambda: wp.array[wp.vec3],
    contact_tangent_rho: wp.array[float],
):
    """Per-step contact penalty decay, lambda retention, C0, and ALM rho.

    ALM retention needs a compressive warm-start (``lambda·n > 0``), a live row
    (``ke > 0`` and ``rho_n > 0``), and positive ``lambda_retention``. Legacy
    AVBD applies its ``alpha*gamma`` retention to the full multiplier vector
    without those gates. ALM rho is automatic from support and independent of
    history. C0 is refreshed for legacy hard and ALM contacts.
    """
    i = wp.tid()
    if i >= rigid_contact_count[0]:
        return

    contact_normal_rho[i] = 0.0
    contact_tangent_rho[i] = 0.0

    ke = contact_material_ke[i]
    s0 = rigid_contact_shape0[i]
    s1 = rigid_contact_shape1[i]
    k_min = _contact_penalty_floor(ke, k_start)
    contact_penalty_k[i] = wp.clamp(penalty_decay * contact_penalty_k[i], k_min, ke)

    # Snapshot the fresh residual before validating warm-start support; the
    # dual uses it to release truly separated rows.
    n = rigid_contact_normal[i]
    if legacy_hard_contacts == 1 or contact_compliant_alm == 1:
        b0 = shape_body[s0] if s0 >= 0 else -1
        b1 = shape_body[s1] if s1 >= 0 else -1
        p0 = rigid_contact_point0[i]
        p1 = rigid_contact_point1[i]
        anchor0_local = p0 + rigid_contact_offset0[i]
        anchor1_local = p1 + rigid_contact_offset1[i]
        cp0 = wp.transform_point(body_q[b0], p0) if b0 >= 0 else p0
        cp1 = wp.transform_point(body_q[b1], p1) if b1 >= 0 else p1
        C0_n = -contact_surface_separation(cp0, cp1, n, rigid_contact_margin0[i], rigid_contact_margin1[i])
        a0 = wp.transform_point(body_q[b0], anchor0_local) if b0 >= 0 else anchor0_local
        a1 = wp.transform_point(body_q[b1], anchor1_local) if b1 >= 0 else anchor1_local
        d_surf = a1 - a0
        C0_t = -(d_surf - n * wp.dot(n, d_surf))
        contact_C0[i] = n * C0_n + C0_t

        if contact_compliant_alm == 1:
            # Resolve rho before using it to validate retained history.
            normal_support = _contact_conditioning_scale(
                s0,
                s1,
                p0,
                p1,
                n,
                shape_body,
                body_q,
                body_com,
                body_inv_mass,
                body_inv_inertia,
                body_structural_k,
                inv_dt_sq,
            )
            contact_normal_rho[i] = _contact_auto_normal_rho(normal_support, ke)
            tangent_support = _contact_tangent_conditioning_scale(
                s0,
                s1,
                anchor0_local,
                anchor1_local,
                n,
                shape_body,
                body_q,
                body_com,
                body_inv_mass,
                body_inv_inertia,
                inv_dt_sq,
            )
            structural_support = _contact_pair_structural_scale(
                b0,
                b1,
                body_flags,
                body_inv_mass,
                body_structural_k,
                proxy_flag,
            )
            contact_tangent_rho[i] = _contact_auto_tangent_rho(
                tangent_support,
                contact_normal_rho[i],
                structural_support,
            )

    lam = contact_lambda[i]
    if contact_compliant_alm == 0:
        # Preserve legacy full-vector retention.
        contact_lambda[i] = lam * lambda_retention
        return

    has_compressive_warmstart = wp.dot(lam, n) > 0.0
    live = ke > 0.0 and contact_normal_rho[i] > 0.0
    if has_compressive_warmstart and live and lambda_retention > 0.0:
        contact_lambda[i] = lam * lambda_retention
    else:
        contact_lambda[i] = wp.vec3(0.0)


@wp.kernel
def init_body_particle_contacts(
    body_particle_contact_count: wp.array[int],
    body_particle_contact_shape: wp.array[int],
    soft_contact_ke: float,
    soft_contact_kd: float,
    soft_contact_mu: float,
    shape_material_ke: wp.array[float],
    shape_material_kd: wp.array[float],
    shape_material_mu: wp.array[float],
    k_start: float,
    # Outputs
    body_particle_contact_penalty_k: wp.array[float],
    body_particle_contact_material_kd: wp.array[float],
    body_particle_contact_material_mu: wp.array[float],
    body_particle_contact_material_ke: wp.array[float],
):
    """Cold-start body-particle contact penalties and cache material properties.

    Averages particle-side material (scalar `soft_contact_ke/kd/mu`) with the
    rigid shape's material. Penalty is seeded at ``min(k_start, avg_ke)`` when
    ramping (k_start >= 0) or at ``avg_ke`` when fixed-k (k_start < 0).
    """
    i = wp.tid()
    # Process every soft contact (single total count) so edge/face records get the same pre-mixed
    # material and seeded penalty as particle contacts.
    if i >= body_particle_contact_count[0]:
        return

    shape_idx = body_particle_contact_shape[i]

    avg_ke, avg_kd, avg_mu = _average_contact_material(
        soft_contact_ke,
        soft_contact_kd,
        soft_contact_mu,
        shape_material_ke[shape_idx],
        shape_material_kd[shape_idx],
        shape_material_mu[shape_idx],
    )

    body_particle_contact_material_ke[i] = avg_ke
    body_particle_contact_material_kd[i] = avg_kd
    body_particle_contact_material_mu[i] = avg_mu

    body_particle_contact_penalty_k[i] = _contact_penalty_floor(avg_ke, k_start)


@wp.func
def _rod_dahl_active_stiffness(
    c_start: int,
    joint_penalty_k: wp.array[float],
    joint_material_k: wp.array[float],
    joint_is_hard: wp.array[wp.int32],
    joint_compliant_alm: int,
) -> wp.vec3:
    """Return bend/twist stiffness that carries Dahl hysteresis."""
    bend_idx = c_start + 2
    twist_idx = c_start + 3

    k_bend = float(0.0)
    k_twist = float(0.0)
    if joint_compliant_alm == 1:
        k_bend = joint_material_k[bend_idx]
        k_twist = joint_material_k[twist_idx]
    else:
        if joint_is_hard[bend_idx] == 0:
            k_bend = joint_penalty_k[bend_idx]
        if joint_is_hard[twist_idx] == 0:
            k_twist = joint_penalty_k[twist_idx]

    return wp.vec3(k_bend, k_bend, k_twist)


@wp.func
def _dahl_axis_direction(d_kappa: float, d_kappa_prev: float) -> float:
    """Loading direction for one scalar Dahl component."""
    direction = float(1.0)
    if d_kappa > _DAHL_KAPPADOT_DEADBAND:
        direction = 1.0
    elif d_kappa < -_DAHL_KAPPADOT_DEADBAND:
        direction = -1.0
    else:
        direction = 1.0 if d_kappa_prev >= 0.0 else -1.0
    return direction


@wp.func
def _advance_dahl_axis(
    d_kappa: float,
    d_kappa_prev: float,
    sigma_prev: float,
    sigma_max: float,
    tau: float,
):
    """Advance one scalar Dahl component from a supplied strain increment.

    Args:
        d_kappa: Current strain increment.
        d_kappa_prev: Previous increment used inside the direction deadband.
        sigma_prev: Previous Dahl stress.
        sigma_max: Magnitude of the Dahl stress envelope.
        tau: Dahl transition length in strain units.

    Returns:
        Updated Dahl stress and loading direction.
    """
    direction = _dahl_axis_direction(d_kappa, d_kappa_prev)
    exp_term = wp.exp(-direction * d_kappa / tau)
    sigma = direction * sigma_max * (1.0 - exp_term) + sigma_prev * exp_term
    return sigma, direction


@wp.kernel
def compute_rod_dahl_parameters(
    # Inputs
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_world: wp.array[wp.int32],
    pose_rebaseline_mask: wp.array[wp.bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_constraint_start: wp.array[int],
    joint_penalty_k: wp.array[float],
    joint_material_k: wp.array[float],
    joint_is_hard: wp.array[wp.int32],
    joint_compliant_alm: int,
    joint_rod_rest_kb_local: wp.array[wp.vec3],
    joint_rod_rest_twist: wp.array[float],
    body_q: wp.array[wp.transform],
    joint_sigma_prev: wp.array[wp.vec3],
    joint_kappa_prev: wp.array[wp.vec3],
    joint_dkappa_prev: wp.array[wp.vec3],
    joint_eps_max: wp.array[float],
    joint_tau: wp.array[float],
    # Outputs
    joint_sigma_start: wp.array[wp.vec3],
    joint_C_fric: wp.array[wp.vec3],
):
    """
    Compute shared rod Dahl hysteresis parameters (sigma0, C_fric) from
    the current bend/twist strain and the stored previous Dahl state.

    The outputs are:
      - sigma0: linearized friction stress at the start of the step (per component)
      - C_fric: tangent stiffness d(sigma)/d(kappa) (per component)

    Dahl ``eps_max`` and ``tau`` are per-joint scalars. Bend and twist
    receive separate envelopes from their respective active stiffnesses.
    Legacy-hard and inactive subspaces produce zero Dahl stress and tangent
    stiffness.

    On a selected first/reset step, curvature is rebased to the current
    start-of-step pose and the stored stress and curvature increment are
    cleared. This pre-solve rebaseline covers enabled rod joints; a disabled rod joint
    refreshes its history in ``update_rod_dahl_state`` (the end-of-step
    finalizer) instead, so a reset while disabled is applied there.
    """
    j = wp.tid()
    zero = wp.vec3(0.0)

    # Default to no friction; the success path overwrites both outputs below.
    joint_sigma_start[j] = zero
    joint_C_fric[j] = zero

    # Only rod joints own Dahl state. Disabled rods are not solved, and
    # the finalizer refreshes their Dahl history every step, so they need no
    # begin-of-step rebaseline.
    if not joint_enabled[j] or joint_type[j] != JointType.ROD:
        return

    parent = joint_parent[j]
    child = joint_child[j]
    # World-parent joints are valid; child body must exist.
    if child < 0:
        return

    rebaseline = _world_selected(joint_world[j], pose_rebaseline_mask)

    eps_max = joint_eps_max[j]
    tau = joint_tau[j]
    c_start = joint_constraint_start[j]
    k_dahl = _rod_dahl_active_stiffness(
        c_start,
        joint_penalty_k,
        joint_material_k,
        joint_is_hard,
        joint_compliant_alm,
    )
    # A gated joint (no Dahl this step) still owes state clearing on a
    # rebaseline step, so stale pre-reset stress can never resurface when a
    # subspace later reactivates.
    dahl_active = tau > 0.0 and eps_max > 0.0 and (k_dahl[0] > 0.0 or k_dahl[1] > 0.0 or k_dahl[2] > 0.0)
    if not dahl_active and not rebaseline:
        return

    # Compute joint frames in world space and the current bend/twist strain.
    if parent >= 0:
        X_wp = body_q[parent] * joint_X_p[j]
    else:
        X_wp = joint_X_p[j]
    X_wc = body_q[child] * joint_X_c[j]
    q_wp = wp.transform_get_rotation(X_wp)
    q_wc = wp.transform_get_rotation(X_wc)
    kappa_now = compute_geometric_rod_kappa_cached_z(
        q_wp,
        q_wc,
        joint_rod_rest_kb_local[j],
        joint_rod_rest_twist[j],
    )

    # Previous Dahl state (from last converged timestep).
    kappa_prev = joint_kappa_prev[j]
    d_kappa_prev = joint_dkappa_prev[j]
    sigma_prev = joint_sigma_prev[j]
    if rebaseline:
        kappa_prev = kappa_now
        d_kappa_prev = wp.vec3(0.0)
        sigma_prev = wp.vec3(0.0)
        joint_kappa_prev[j] = kappa_prev
        joint_dkappa_prev[j] = d_kappa_prev
        joint_sigma_prev[j] = sigma_prev

    if not dahl_active:
        return

    d_kappa = _rod_bend_twist_delta(kappa_now, kappa_prev)
    sigma_out = zero
    C_fric_out = zero
    for axis in range(3):
        sigma_max = k_dahl[axis] * eps_max
        if sigma_max <= 0.0:
            continue

        sigma0_i, direction = _advance_dahl_axis(
            d_kappa[axis],
            d_kappa_prev[axis],
            sigma_prev[axis],
            sigma_max,
            tau,
        )
        sigma0_i = wp.clamp(sigma0_i, -sigma_max, sigma_max)

        # Tangent stiffness K = (sigma_max - dir*sigma0) / (tau + |d_kappa|).
        numerator = sigma_max - direction * sigma0_i
        denominator = tau + wp.abs(d_kappa[axis])
        sigma_out[axis] = sigma0_i
        C_fric_out[axis] = wp.max(numerator / denominator, 0.0)

    joint_sigma_start[j] = sigma_out
    joint_C_fric[j] = C_fric_out


# -----------------------------
# Iteration kernels (per color per iteration)
# -----------------------------
@wp.kernel
def accumulate_body_body_contacts_per_body(
    dt: float,
    color_group: wp.array[wp.int32],
    body_q_prev: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    friction_epsilon: float,
    contact_penalty_k: wp.array[float],
    contact_normal_rho: wp.array[float],
    contact_material_ke: wp.array[float],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    contact_tangent_rho: wp.array[float],
    contact_lambda: wp.array[wp.vec3],
    contact_C0: wp.array[wp.vec3],
    stab_alpha: float,
    legacy_hard_contacts: int,
    contact_compliant_alm: int,
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[int],
    rigid_contact_shape1: wp.array[int],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_offset0: wp.array[wp.vec3],
    rigid_contact_offset1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_margin0: wp.array[float],
    rigid_contact_margin1: wp.array[float],
    shape_body: wp.array[wp.int32],
    body_contact_buffer_pre_alloc: int,
    body_contact_counts: wp.array[wp.int32],
    body_contact_indices: wp.array[wp.int32],
    body_forces: wp.array[wp.vec3],
    body_torques: wp.array[wp.vec3],
    body_hessian_ll: wp.array[wp.mat33],
    body_hessian_al: wp.array[wp.mat33],
    body_hessian_aa: wp.array[wp.mat33],
):
    """
    Per-body contact force/Hessian accumulation (compliant ALM or legacy penalty)
    with _NUM_CONTACT_THREADS_PER_BODY strided threads.
    """
    tid = wp.tid()
    body_idx_in_group = tid // _NUM_CONTACT_THREADS_PER_BODY
    thread_id_within_body = tid % _NUM_CONTACT_THREADS_PER_BODY

    if body_idx_in_group >= color_group.shape[0]:
        return

    body_id = color_group[body_idx_in_group]
    if body_inv_mass[body_id] <= 0.0:
        return

    num_contacts = body_contact_counts[body_id]
    if num_contacts > body_contact_buffer_pre_alloc:
        num_contacts = body_contact_buffer_pre_alloc

    contact_count = rigid_contact_count[0]

    force_acc = wp.vec3(0.0)
    torque_acc = wp.vec3(0.0)
    h_ll_acc = wp.mat33(0.0)
    h_al_acc = wp.mat33(0.0)
    h_aa_acc = wp.mat33(0.0)

    i = thread_id_within_body
    while i < num_contacts:
        contact_idx = body_contact_indices[body_id * body_contact_buffer_pre_alloc + i]
        if contact_idx >= contact_count:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        s0 = rigid_contact_shape0[contact_idx]
        s1 = rigid_contact_shape1[contact_idx]
        b0 = shape_body[s0] if s0 >= 0 else -1
        b1 = shape_body[s1] if s1 >= 0 else -1

        if b0 != body_id and b1 != body_id:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        cp0_local = rigid_contact_point0[contact_idx]
        cp1_local = rigid_contact_point1[contact_idx]
        cp0_offset_local = rigid_contact_offset0[contact_idx]
        cp1_offset_local = rigid_contact_offset1[contact_idx]
        contact_normal = rigid_contact_normal[contact_idx]
        # Normal C_n uses the unprojected (skeleton) points: ``thickness`` already accounts
        # for the radial extent, so adding the offset here would double-count it.
        cp0_world = wp.transform_point(body_q[b0], cp0_local) if b0 >= 0 else cp0_local
        cp1_world = wp.transform_point(body_q[b1], cp1_local) if b1 >= 0 else cp1_local
        C_n = -contact_surface_separation(
            cp0_world, cp1_world, contact_normal, rigid_contact_margin0[contact_idx], rigid_contact_margin1[contact_idx]
        )

        lam_n = float(0.0)
        C_eff = C_n
        lam_vec = wp.vec3(0.0)
        normal_solve_weight = _load_solve_weight(
            contact_penalty_k, contact_normal_rho, contact_idx, contact_compliant_alm
        )
        material_k = contact_material_ke[contact_idx]
        friction_c0 = wp.vec3(0.0)

        if legacy_hard_contacts == 1 or contact_compliant_alm == 1:
            lam_vec = contact_lambda[contact_idx]
            lam_n = wp.dot(lam_vec, contact_normal)
            C0_vec = contact_C0[contact_idx]
            C0_n = wp.dot(contact_normal, C0_vec)
            # C0 stabilization: normal uses C_n - alpha*C0_n;
            # tangent caches (1 - alpha)*C0_t for the later tangential update.
            C_eff = C_n - stab_alpha * C0_n
            friction_c0 = (1.0 - stab_alpha) * (C0_vec - contact_normal * C0_n)

        if C_n <= _SMALL_LENGTH_EPS and lam_n <= 0.0:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        normal_primal_k, lambda_n_eff = _material_force_terms(
            normal_solve_weight, material_k, lam_n, contact_compliant_alm
        )
        f_n_check = normal_primal_k * C_eff + lambda_n_eff
        if f_n_check <= 0.0 and lam_n <= 0.0:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        contact_kd = contact_material_kd[contact_idx]
        contact_mu = contact_material_mu[contact_idx]

        (
            force_0,
            torque_0,
            h_ll_0,
            h_al_0,
            h_aa_0,
            force_1,
            torque_1,
            h_ll_1,
            h_al_1,
            h_aa_1,
        ) = evaluate_rigid_contact_from_collision(
            b0,
            b1,
            body_q,
            body_q_prev,
            body_com,
            cp0_local,
            cp1_local,
            cp0_offset_local,
            cp1_offset_local,
            contact_normal,
            C_eff,
            normal_solve_weight,
            material_k,
            contact_tangent_rho[contact_idx],
            contact_kd,
            lam_vec,
            contact_mu,
            friction_epsilon,
            legacy_hard_contacts,
            contact_compliant_alm,
            dt,
            friction_c0,
        )

        if body_id == b0:
            force_acc += force_0
            torque_acc += torque_0
            h_ll_acc += h_ll_0
            h_al_acc += h_al_0
            h_aa_acc += h_aa_0
        else:
            force_acc += force_1
            torque_acc += torque_1
            h_ll_acc += h_ll_1
            h_al_acc += h_al_1
            h_aa_acc += h_aa_1

        i += _NUM_CONTACT_THREADS_PER_BODY

    wp.atomic_add(body_forces, body_id, force_acc)
    wp.atomic_add(body_torques, body_id, torque_acc)
    wp.atomic_add(body_hessian_ll, body_id, h_ll_acc)
    wp.atomic_add(body_hessian_al, body_id, h_al_acc)
    wp.atomic_add(body_hessian_aa, body_id, h_aa_acc)


@wp.kernel
def compute_rigid_contact_forces(
    dt: float,
    # Contact data
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[int],
    rigid_contact_shape1: wp.array[int],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_offset0: wp.array[wp.vec3],
    rigid_contact_offset1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_margin0: wp.array[float],
    rigid_contact_margin1: wp.array[float],
    # Model/state
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    # Contact material properties (per-contact)
    contact_penalty_k: wp.array[float],
    contact_normal_rho: wp.array[float],
    contact_material_ke: wp.array[float],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    contact_tangent_rho: wp.array[float],
    contact_lambda: wp.array[wp.vec3],
    contact_C0: wp.array[wp.vec3],
    stab_alpha: float,
    legacy_hard_contacts: int,
    contact_compliant_alm: int,
    friction_epsilon: float,
    # Outputs (length = rigid_contact_max)
    out_body0: wp.array[wp.int32],
    out_body1: wp.array[wp.int32],
    out_point0_world: wp.array[wp.vec3],
    out_point1_world: wp.array[wp.vec3],
    out_force_on_body1: wp.array[wp.vec3],
):
    """Compute per-contact forces in world space."""
    contact_idx = wp.tid()

    rc = rigid_contact_count[0]
    if contact_idx >= rc:
        # Fill sentinel values for inactive entries (useful when launching with rigid_contact_max)
        out_body0[contact_idx] = wp.int32(-1)
        out_body1[contact_idx] = wp.int32(-1)
        out_point0_world[contact_idx] = wp.vec3(0.0)
        out_point1_world[contact_idx] = wp.vec3(0.0)
        out_force_on_body1[contact_idx] = wp.vec3(0.0)
        return

    s0 = rigid_contact_shape0[contact_idx]
    s1 = rigid_contact_shape1[contact_idx]
    if s0 < 0 or s1 < 0:
        out_body0[contact_idx] = wp.int32(-1)
        out_body1[contact_idx] = wp.int32(-1)
        out_point0_world[contact_idx] = wp.vec3(0.0)
        out_point1_world[contact_idx] = wp.vec3(0.0)
        out_force_on_body1[contact_idx] = wp.vec3(0.0)
        return

    b0 = shape_body[s0]
    b1 = shape_body[s1]
    out_body0[contact_idx] = b0
    out_body1[contact_idx] = b1

    cp0_local = rigid_contact_point0[contact_idx]
    cp1_local = rigid_contact_point1[contact_idx]
    cp0_offset_local = rigid_contact_offset0[contact_idx]
    cp1_offset_local = rigid_contact_offset1[contact_idx]
    contact_normal = rigid_contact_normal[contact_idx]

    # Normal C_n uses the unprojected (skeleton) points: ``thickness`` already accounts
    # for the radial extent, so adding the offset here would double-count it.
    cp0_world = wp.transform_point(body_q[b0], cp0_local) if b0 >= 0 else cp0_local
    cp1_world = wp.transform_point(body_q[b1], cp1_local) if b1 >= 0 else cp1_local
    out_point0_world[contact_idx] = (
        wp.transform_point(body_q[b0], cp0_local + cp0_offset_local) if b0 >= 0 else cp0_local + cp0_offset_local
    )
    out_point1_world[contact_idx] = (
        wp.transform_point(body_q[b1], cp1_local + cp1_offset_local) if b1 >= 0 else cp1_local + cp1_offset_local
    )

    C_n = -contact_surface_separation(
        cp0_world, cp1_world, contact_normal, rigid_contact_margin0[contact_idx], rigid_contact_margin1[contact_idx]
    )

    lam_n = float(0.0)
    C_eff = C_n
    lam_vec = wp.vec3(0.0)
    normal_solve_weight = _load_solve_weight(contact_penalty_k, contact_normal_rho, contact_idx, contact_compliant_alm)
    material_k = contact_material_ke[contact_idx]
    friction_c0 = wp.vec3(0.0)

    if legacy_hard_contacts == 1 or contact_compliant_alm == 1:
        lam_vec = contact_lambda[contact_idx]
        lam_n = wp.dot(lam_vec, contact_normal)
        C0_vec = contact_C0[contact_idx]
        C0_n = wp.dot(contact_normal, C0_vec)
        # C0 stabilization: normal uses C_n - alpha*C0_n;
        # tangent caches (1 - alpha)*C0_t for the later tangential update.
        C_eff = C_n - stab_alpha * C0_n
        friction_c0 = (1.0 - stab_alpha) * (C0_vec - contact_normal * C0_n)

    normal_primal_k, lambda_n_eff = _material_force_terms(normal_solve_weight, material_k, lam_n, contact_compliant_alm)
    f_n_check = normal_primal_k * C_eff + lambda_n_eff
    if (C_n <= _SMALL_LENGTH_EPS or f_n_check <= 0.0) and lam_n <= 0.0:
        out_force_on_body1[contact_idx] = wp.vec3(0.0)
        return

    contact_kd = contact_material_kd[contact_idx]
    contact_mu = contact_material_mu[contact_idx]

    (
        _force_0,
        _torque_0,
        _h_ll_0,
        _h_al_0,
        _h_aa_0,
        force_1,
        _torque_1,
        _h_ll_1,
        _h_al_1,
        _h_aa_1,
    ) = evaluate_rigid_contact_from_collision(
        int(b0),
        int(b1),
        body_q,
        body_q_prev,
        body_com,
        cp0_local,
        cp1_local,
        cp0_offset_local,
        cp1_offset_local,
        contact_normal,
        C_eff,
        normal_solve_weight,
        material_k,
        contact_tangent_rho[contact_idx],
        contact_kd,
        lam_vec,
        contact_mu,
        friction_epsilon,
        legacy_hard_contacts,
        contact_compliant_alm,
        dt,
        friction_c0,
    )

    out_force_on_body1[contact_idx] = force_1


@wp.kernel
def accumulate_body_particle_contacts_per_body(
    dt: float,
    color_group: wp.array[wp.int32],
    # Particle state
    particle_q: wp.array[wp.vec3],
    particle_q_prev: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    # Rigid body state
    body_q_prev: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    shape_body: wp.array[int],
    # AVBD body-particle soft contact penalties and material properties
    friction_epsilon: float,
    body_particle_contact_penalty_k: wp.array[float],
    body_particle_contact_material_ke: wp.array[float],
    body_particle_contact_material_kd: wp.array[float],
    body_particle_contact_material_mu: wp.array[float],
    # Soft contact data (body-particle)
    body_particle_contact_count: wp.array[int],
    soft_contact_indices: wp.array[wp.vec3i],
    body_particle_contact_shape: wp.array[int],
    body_particle_contact_body_pos: wp.array[wp.vec3],
    body_particle_contact_body_vel: wp.array[wp.vec3],
    body_particle_contact_normal: wp.array[wp.vec3],
    # Barycentric weights on each record's soft particles; (1, 0, 0) for a particle contact.
    soft_contact_barycentric: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    # Per-body soft-contact adjacency (body-particle)
    body_particle_contact_buffer_pre_alloc: int,
    body_particle_contact_counts: wp.array[wp.int32],
    body_particle_contact_indices: wp.array[wp.int32],
    # Outputs
    body_forces: wp.array[wp.vec3],
    body_torques: wp.array[wp.vec3],
    body_hessian_ll: wp.array[wp.mat33],
    body_hessian_al: wp.array[wp.mat33],
    body_hessian_aa: wp.array[wp.mat33],
):
    """
    Per-body accumulation of body-particle soft contact forces and Hessians on rigid bodies.

    Handles both contact kinds from one per-body adjacency list, dispatching on each record's
    -1-padded ``soft_contact_indices``: a particle record ``(p, -1, -1)`` resolves single-particle
    geometry inline; an edge/face record evaluates the barycentric contact point over its 2-3 soft
    particles via ``_eval_soft_ef_contact``. Both apply the shared force law
    ``_compute_body_particle_contact_force`` and the equal-and-opposite body reaction. Body surface
    velocity uses the displacement-based path (body_q_prev).

    Notes:
      - Only dynamic bodies (inv_mass > 0) are updated.
      - Hessian contributions are accumulated into body_hessian_ll/al/aa.
      - Uses per-contact effective penalty/material parameters initialized once per step.
    """
    tid = wp.tid()
    body_idx_in_group = tid // _NUM_CONTACT_THREADS_PER_BODY
    thread_id_within_body = tid % _NUM_CONTACT_THREADS_PER_BODY

    if body_idx_in_group >= color_group.shape[0]:
        return

    body_id = color_group[body_idx_in_group]
    if body_inv_mass[body_id] <= 0.0:
        return

    num_contacts = body_particle_contact_counts[body_id]
    if num_contacts > body_particle_contact_buffer_pre_alloc:
        num_contacts = body_particle_contact_buffer_pre_alloc

    max_contacts = body_particle_contact_count[0]  # single total soft-contact count

    X_wb = body_q[body_id]
    X_wb_prev = body_q_prev[body_id]
    com_world = wp.transform_point(X_wb, body_com[body_id])

    force_acc = wp.vec3(0.0)
    torque_acc = wp.vec3(0.0)
    h_ll_acc = wp.mat33(0.0)
    h_al_acc = wp.mat33(0.0)
    h_aa_acc = wp.mat33(0.0)

    i = thread_id_within_body
    while i < num_contacts:
        contact_idx = body_particle_contact_indices[body_id * body_particle_contact_buffer_pre_alloc + i]
        i += _NUM_CONTACT_THREADS_PER_BODY
        if contact_idx >= max_contacts:
            continue

        f_soft = wp.vec3(0.0)
        h_soft = wp.mat33(0.0)
        cp_world = wp.vec3(0.0)

        corners = soft_contact_indices[contact_idx]

        if corners[1] < 0:
            # Particle-vs-surface (p, -1, -1): single-particle geometry, resolved inline.
            particle_idx = corners[0]
            if particle_idx < 0:
                continue

            particle_pos = particle_q[particle_idx]
            cp_local = body_particle_contact_body_pos[contact_idx]
            cp_world = wp.transform_point(X_wb, cp_local)
            n = body_particle_contact_normal[contact_idx]
            radius = particle_radius[particle_idx]
            s_idx = body_particle_contact_shape[contact_idx]
            margin = shape_margin[s_idx] if s_idx >= 0 and shape_margin.shape[0] > 0 else 0.0
            penetration_depth = -(wp.dot(n, particle_pos - cp_world) - radius - margin)
            if penetration_depth <= 0.0:
                continue

            bx_prev = wp.transform_point(X_wb_prev, cp_local)
            bv = (cp_world - bx_prev) / dt + wp.transform_vector(X_wb, body_particle_contact_body_vel[contact_idx])
            dx = particle_pos - particle_q_prev[particle_idx]
            relative_translation = dx - bv * dt

            f_soft, h_soft = _compute_body_particle_contact_force(
                penetration_depth,
                n,
                relative_translation,
                body_particle_contact_penalty_k[contact_idx],
                body_particle_contact_material_kd[contact_idx],
                body_particle_contact_material_mu[contact_idx],
                friction_epsilon,
                dt,
            )
        else:
            # Edge/face: barycentric contact point over the record's 2-3 soft particles. Uses the
            # shared force law via _eval_soft_ef_contact -- the same evaluation as the particle side.
            bary = soft_contact_barycentric[contact_idx]
            f_soft, h_soft, cp_world = _eval_soft_ef_contact(
                contact_idx,
                corners,
                bary,
                particle_q,
                particle_q_prev,
                particle_radius,
                body_particle_contact_penalty_k[contact_idx],
                body_particle_contact_material_kd[contact_idx],
                body_particle_contact_material_mu[contact_idx],
                friction_epsilon,
                shape_body,
                body_q,
                body_q_prev,
                body_qd,
                body_com,
                body_particle_contact_shape,
                body_particle_contact_body_pos,
                body_particle_contact_body_vel,
                body_particle_contact_normal,
                shape_margin,
                dt,
            )

        # Equal-and-opposite reaction on the body at the rigid contact point (shared by both kinds).
        f_body = -f_soft
        r = cp_world - com_world
        tau_body = wp.cross(r, f_body)
        r_skew = wp.skew(r)
        r_skew_T_K = wp.transpose(r_skew) * h_soft

        force_acc += f_body
        torque_acc += tau_body
        h_ll_acc += h_soft
        h_al_acc += -r_skew_T_K
        h_aa_acc += r_skew_T_K * r_skew

    wp.atomic_add(body_forces, body_id, force_acc)
    wp.atomic_add(body_torques, body_id, torque_acc)
    wp.atomic_add(body_hessian_ll, body_id, h_ll_acc)
    wp.atomic_add(body_hessian_al, body_id, h_al_acc)
    wp.atomic_add(body_hessian_aa, body_id, h_aa_acc)


@wp.kernel
def solve_rigid_body(
    dt: float,
    body_ids_in_color: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_q_rest: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_inv_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_inertia_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    adjacency: RigidForceElementAdjacencyInfo,
    # Joint data
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_rod_rest_kb_local: wp.array[wp.vec3],
    joint_rod_rest_twist: wp.array[float],
    joint_qd_start: wp.array[int],
    joint_target_q_start: wp.array[int],
    joint_constraint_start: wp.array[int],
    # Joint constraint state (scalar constraints indexed via joint_constraint_start)
    joint_penalty_k: wp.array[float],
    joint_rho: wp.array[float],
    joint_material_k: wp.array[float],
    joint_penalty_kd: wp.array[float],
    # Dahl hysteresis parameters (frozen for this timestep, component-wise vec3 per joint)
    joint_sigma_start: wp.array[wp.vec3],
    joint_C_fric: wp.array[wp.vec3],
    # Drive parameters (DOF-indexed via joint_qd_start)
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_q: wp.array[float],
    joint_target_qd: wp.array[float],
    # Limit parameters (DOF-indexed via joint_qd_start)
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_limit_ke: wp.array[float],
    joint_limit_kd: wp.array[float],
    joint_drive_limit_support: wp.array[float],
    joint_drive_lambda: wp.array[float],
    joint_limit_lambda: wp.array[float],
    joint_lambda_lin: wp.array[wp.vec3],
    joint_lambda_ang: wp.array[wp.vec3],
    joint_C0_lin: wp.array[wp.vec3],
    joint_C0_ang: wp.array[wp.vec3],
    joint_is_hard: wp.array[wp.int32],
    stab_alpha: float,
    joint_compliant_alm: int,
    joint_dof_dim: wp.array2d[int],
    joint_rest_angle: wp.array[float],
    external_forces: wp.array[wp.vec3],
    external_torques: wp.array[wp.vec3],
    # Preaccumulated rigid-contact Hessian contributions
    external_hessian_ll: wp.array[wp.mat33],
    external_hessian_al: wp.array[wp.mat33],
    external_hessian_aa: wp.array[wp.mat33],
    # Output
    body_q_new: wp.array[wp.transform],
):
    """
    AVBD solve step for rigid bodies.

    Assembles inertial, joint, and collision contributions into a 6x6 SPD
    block system and solves via direct LDL^T.

    Algorithm:
      1. Compute inertial forces/Hessians
      2. Accumulate contact and joint forces/Hessians
      3. Update the pose from one 6x6 LDL^T solve

    Args:
        dt: Time step.
        body_ids_in_color: Body indices in current color group (for parallel coloring).
        body_q_prev: Previous body transforms (for damping and friction).
        body_q_rest: Rest transforms (for joint targets).
        body_mass: Body masses.
        body_inv_mass: Inverse masses (0 for kinematic bodies).
        body_inertia: Inertia tensors (local body frame).
        body_inertia_q: Inertial target transforms (from forward integration).
        body_com: Center of mass offsets (local body frame).
        adjacency: Body-joint adjacency (CSR format).
        joint_*: Joint configuration arrays.
        joint_penalty_k: Per-constraint legacy penalty stiffness.
        joint_sigma_start: Dahl hysteresis state at start of step.
        joint_C_fric: Dahl friction configuration per joint.
        external_forces: External linear forces from rigid contacts.
        external_torques: External angular torques from rigid contacts.
        external_hessian_ll: Preaccumulated rigid-contact linear block.
        external_hessian_al: Preaccumulated rigid-contact angular-linear block.
        external_hessian_aa: Preaccumulated rigid-contact angular block.
        body_q: Current body transforms (input).
        body_q_new: Updated body transforms (output) for the current solve sweep.

    Note:
      - All forces, torques, and Hessian blocks are expressed in the world frame.
    """
    tid = wp.tid()
    body_index = body_ids_in_color[tid]

    q_current = body_q[body_index]

    # Early exit for kinematic bodies
    if body_inv_mass[body_index] == 0.0:
        body_q_new[body_index] = q_current
        return

    # Inertial force and Hessian
    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # Read body properties
    q_inertial = body_inertia_q[body_index]
    body_com_local = body_com[body_index]
    m = body_mass[body_index]
    I_body = body_inertia[body_index]

    # Extract poses
    pos_current = wp.transform_get_translation(q_current)
    rot_current = wp.transform_get_rotation(q_current)
    pos_star = wp.transform_get_translation(q_inertial)
    rot_star = wp.transform_get_rotation(q_inertial)

    # Compute COM positions
    com_current = pos_current + wp.quat_rotate(rot_current, body_com_local)
    com_star = pos_star + wp.quat_rotate(rot_star, body_com_local)

    # Linear inertial force and Hessian
    inertial_coeff = m * dt_sqr_reciprocal
    f_lin = (com_star - com_current) * inertial_coeff

    # Compute relative rotation via quaternion difference
    # dq = q_current^-1 * q_star
    q_delta = wp.mul(wp.quat_inverse(rot_current), rot_star)

    # Enforce shortest path (w > 0) to avoid double-cover ambiguity
    if q_delta[3] < 0.0:
        q_delta = wp.quat(-q_delta[0], -q_delta[1], -q_delta[2], -q_delta[3])

    # Rotation vector
    axis_body, angle_body = wp.quat_to_axis_angle(q_delta)
    theta_body = axis_body * angle_body

    # Angular inertial torque
    tau_body = I_body * (theta_body * dt_sqr_reciprocal)
    tau_world = wp.quat_rotate(rot_current, tau_body)

    # Angular Hessian in world frame: use full inertia (supports off-diagonal products of inertia)
    R_cur = wp.quat_to_matrix(rot_current)
    I_world = R_cur * I_body * wp.transpose(R_cur)
    angular_hessian = dt_sqr_reciprocal * I_world

    # Accumulate external forces (rigid contacts)
    # Read external contributions
    ext_torque = external_torques[body_index]
    ext_force = external_forces[body_index]
    ext_h_aa = external_hessian_aa[body_index]
    ext_h_al = external_hessian_al[body_index]
    ext_h_ll = external_hessian_ll[body_index]

    f_torque = tau_world + ext_torque
    f_force = f_lin + ext_force

    h_aa = angular_hessian + ext_h_aa
    h_al = ext_h_al
    h_ll = wp.mat33(
        ext_h_ll[0, 0] + inertial_coeff,
        ext_h_ll[0, 1],
        ext_h_ll[0, 2],
        ext_h_ll[1, 0],
        ext_h_ll[1, 1] + inertial_coeff,
        ext_h_ll[1, 2],
        ext_h_ll[2, 0],
        ext_h_ll[2, 1],
        ext_h_ll[2, 2] + inertial_coeff,
    )

    # Accumulate joint forces (constraints)
    num_adj_joints = get_body_num_adjacent_joints(adjacency, body_index)
    for joint_counter in range(num_adj_joints):
        joint_idx = get_body_adjacent_joint_id(adjacency, body_index, joint_counter)
        joint_force, joint_torque, joint_H_ll, joint_H_al, joint_H_aa = evaluate_joint_force_hessian(
            body_index,
            joint_idx,
            body_q,
            body_q_prev,
            body_q_rest,
            body_com,
            joint_type,
            joint_enabled,
            joint_parent,
            joint_child,
            joint_X_p,
            joint_X_c,
            joint_axis,
            joint_rod_rest_kb_local,
            joint_rod_rest_twist,
            joint_qd_start,
            joint_target_q_start,
            joint_constraint_start,
            joint_penalty_k,
            joint_rho,
            joint_material_k,
            joint_penalty_kd,
            joint_sigma_start,
            joint_C_fric,
            joint_target_ke,
            joint_target_kd,
            joint_target_q,
            joint_target_qd,
            joint_limit_lower,
            joint_limit_upper,
            joint_limit_ke,
            joint_limit_kd,
            joint_drive_limit_support,
            joint_drive_lambda,
            joint_limit_lambda,
            joint_lambda_lin,
            joint_lambda_ang,
            joint_C0_lin,
            joint_C0_ang,
            joint_is_hard,
            stab_alpha,
            joint_compliant_alm,
            joint_dof_dim,
            joint_rest_angle,
            dt,
        )

        f_force = f_force + joint_force
        f_torque = f_torque + joint_torque

        h_ll = h_ll + joint_H_ll
        h_al = h_al + joint_H_al
        h_aa = h_aa + joint_H_aa

    # Regularize angular Hessian
    trA = wp.trace(h_aa) / 3.0
    epsA = 1.0e-9 * (trA + 1.0)
    h_aa[0, 0] = h_aa[0, 0] + epsA
    h_aa[1, 1] = h_aa[1, 1] + epsA
    h_aa[2, 2] = h_aa[2, 2] + epsA

    # Solve 6x6 system via direct LDL^T
    x_inc, w_world = ldlt6_solve(h_ll, h_aa, h_al, f_force, f_torque)

    # Update pose from increments
    # Convert angular increment to quaternion
    if _USE_SMALL_ANGLE_APPROX:
        half_w = w_world * 0.5
        dq_world = wp.quat(half_w[0], half_w[1], half_w[2], 1.0)
        dq_world = wp.normalize(dq_world)
    else:
        ang_mag = wp.length(w_world)
        if ang_mag > _SMALL_ANGLE_EPS:
            dq_world = wp.quat_from_axis_angle(w_world / ang_mag, ang_mag)
        else:
            half_w = w_world * 0.5
            dq_world = wp.quat(half_w[0], half_w[1], half_w[2], 1.0)
            dq_world = wp.normalize(dq_world)

    # Apply rotation
    rot_new = wp.mul(dq_world, rot_current)
    rot_new = wp.normalize(rot_new)

    # Update position
    com_new = com_current + x_inc
    pos_new = com_new - wp.quat_rotate(rot_new, body_com_local)

    body_q_new[body_index] = wp.transform(pos_new, rot_new)


@wp.kernel
def update_duals_joint(
    # Inputs
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_rod_rest_kb_local: wp.array[wp.vec3],
    joint_rod_rest_twist: wp.array[float],
    joint_qd_start: wp.array[int],
    joint_target_q_start: wp.array[int],
    joint_constraint_start: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_q_rest: wp.array[wp.transform],
    joint_dof_dim: wp.array2d[int],
    joint_C0_lin: wp.array[wp.vec3],
    joint_C0_ang: wp.array[wp.vec3],
    joint_is_hard: wp.array[wp.int32],
    stab_alpha: float,
    joint_material_k: wp.array[float],
    joint_rho: wp.array[float],
    joint_compliant_alm: int,
    beta_lin: float,
    beta_ang: float,
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_q: wp.array[float],
    joint_target_qd: wp.array[float],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_limit_ke: wp.array[float],
    joint_limit_kd: wp.array[float],
    joint_rest_angle: wp.array[float],
    joint_drive_limit_support: wp.array[float],
    dt: float,
    # Input/output
    joint_penalty_k: wp.array[float],
    joint_lambda_lin: wp.array[wp.vec3],
    joint_lambda_ang: wp.array[wp.vec3],
    joint_drive_lambda: wp.array[float],
    joint_limit_lambda: wp.array[float],
):
    """Update joint duals / legacy penalties each iteration.

    ALM structural and legacy-hard slots update vector multipliers; legacy soft
    only ramps penalty. Drive/limit: legacy ramps penalty, ALM uses separate
    scalar lambdas (limits half-line projected). Rho is fixed within the step.
    """
    j = wp.tid()

    if not joint_enabled[j]:
        return

    parent = joint_parent[j]
    child = joint_child[j]

    # Early exit for invalid joints
    if child < 0:
        return

    jt = joint_type[j]
    if (
        jt != JointType.ROD
        and jt != JointType.BALL
        and jt != JointType.FIXED
        and jt != JointType.REVOLUTE
        and jt != JointType.PRISMATIC
        and jt != JointType.D6
    ):
        return

    # Read solver constraint start index
    c_start = joint_constraint_start[j]

    # Compute joint frames in world space
    if parent >= 0:
        X_wp = body_q[parent] * joint_X_p[j]
        X_wp_prev = body_q_prev[parent] * joint_X_p[j]
    else:
        X_wp = joint_X_p[j]
        X_wp_prev = joint_X_p[j]
    X_wc = body_q[child] * joint_X_c[j]
    X_wc_prev = body_q_prev[child] * joint_X_c[j]

    # ROD joint: fixed stretch/shear/bend/twist slots.
    if jt == JointType.ROD:
        q_wp = wp.transform_get_rotation(X_wp)
        q_wc = wp.transform_get_rotation(X_wc)

        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        C_vec = x_c - x_p

        kappa = compute_geometric_rod_kappa_cached_z(
            q_wp,
            q_wc,
            joint_rod_rest_kb_local[j],
            joint_rod_rest_twist[j],
        )

        # Linear penalty update in the parent-material frame: local
        # u = [shear_x, shear_y, stretch_z], so stretch is z and shear is xy.
        stretch_idx = c_start
        shear_idx = c_start + 1
        u = wp.quat_rotate_inv(q_wp, C_vec)
        lambda_lin = joint_lambda_lin[j]
        C0_lin = joint_C0_lin[j]

        u_stretch = wp.vec3(0.0, 0.0, u[2])
        lam_stretch = _update_joint_structural_dual(
            u_stretch,
            wp.vec3(0.0, 0.0, C0_lin[2]),
            stab_alpha,
            joint_penalty_k,
            joint_rho,
            joint_material_k,
            joint_is_hard,
            stretch_idx,
            wp.vec3(0.0, 0.0, lambda_lin[2]),
            joint_compliant_alm,
            beta_lin,
        )
        if joint_compliant_alm == 0 and joint_is_hard[stretch_idx] == 0:
            lam_stretch = wp.vec3(0.0)

        u_shear = wp.vec3(u[0], u[1], 0.0)
        lam_shear = _update_joint_structural_dual(
            u_shear,
            wp.vec3(C0_lin[0], C0_lin[1], 0.0),
            stab_alpha,
            joint_penalty_k,
            joint_rho,
            joint_material_k,
            joint_is_hard,
            shear_idx,
            wp.vec3(lambda_lin[0], lambda_lin[1], 0.0),
            joint_compliant_alm,
            beta_lin,
        )
        if joint_compliant_alm == 0 and joint_is_hard[shear_idx] == 0:
            lam_shear = wp.vec3(0.0)
        joint_lambda_lin[j] = lam_stretch + lam_shear

        bend_idx = c_start + 2
        twist_idx = c_start + 3
        lambda_ang = joint_lambda_ang[j]
        C0_ang = joint_C0_ang[j]

        kappa_bend = wp.vec3(kappa[0], kappa[1], 0.0)
        lam_bend = _update_joint_structural_dual(
            kappa_bend,
            wp.vec3(C0_ang[0], C0_ang[1], 0.0),
            stab_alpha,
            joint_penalty_k,
            joint_rho,
            joint_material_k,
            joint_is_hard,
            bend_idx,
            wp.vec3(lambda_ang[0], lambda_ang[1], 0.0),
            joint_compliant_alm,
            beta_ang,
        )
        if joint_compliant_alm == 0 and joint_is_hard[bend_idx] == 0:
            lam_bend = wp.vec3(0.0)

        kappa_twist = wp.vec3(0.0, 0.0, kappa[2])
        lam_twist = _update_joint_structural_dual(
            kappa_twist,
            wp.vec3(0.0, 0.0, C0_ang[2]),
            stab_alpha,
            joint_penalty_k,
            joint_rho,
            joint_material_k,
            joint_is_hard,
            twist_idx,
            wp.vec3(0.0, 0.0, lambda_ang[2]),
            joint_compliant_alm,
            beta_ang,
        )
        if joint_compliant_alm == 0 and joint_is_hard[twist_idx] == 0:
            lam_twist = wp.vec3(0.0)
        joint_lambda_ang[j] = lam_bend + lam_twist
        return

    # BALL joint: update isotropic linear anchor-coincidence penalty (single scalar).
    if jt == JointType.BALL:
        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        C_vec = x_c - x_p

        i0 = c_start
        joint_lambda_lin[j] = _update_joint_structural_dual(
            C_vec,
            joint_C0_lin[j],
            stab_alpha,
            joint_penalty_k,
            joint_rho,
            joint_material_k,
            joint_is_hard,
            i0,
            joint_lambda_lin[j],
            joint_compliant_alm,
            beta_lin,
        )
        return

    if parent >= 0:
        X_wp_rest = body_q_rest[parent] * joint_X_p[j]
    else:
        X_wp_rest = joint_X_p[j]
    X_wc_rest = body_q_rest[child] * joint_X_c[j]

    # FIXED joint: update isotropic linear + isotropic angular penalties (2 scalars).
    if jt == JointType.FIXED:
        i_lin = c_start + 0
        i_ang = c_start + 1

        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        C_vec_lin = x_c - x_p
        joint_lambda_lin[j] = _update_joint_structural_dual(
            C_vec_lin,
            joint_C0_lin[j],
            stab_alpha,
            joint_penalty_k,
            joint_rho,
            joint_material_k,
            joint_is_hard,
            i_lin,
            joint_lambda_lin[j],
            joint_compliant_alm,
            beta_lin,
        )

        q_wp = wp.transform_get_rotation(X_wp)
        q_wc = wp.transform_get_rotation(X_wc)
        q_wp_rest = wp.transform_get_rotation(X_wp_rest)
        q_wc_rest = wp.transform_get_rotation(X_wc_rest)
        kappa = compute_kappa(q_wp, q_wc, q_wp_rest, q_wc_rest)
        joint_lambda_ang[j] = _update_joint_structural_dual(
            kappa,
            joint_C0_ang[j],
            stab_alpha,
            joint_penalty_k,
            joint_rho,
            joint_material_k,
            joint_is_hard,
            i_ang,
            joint_lambda_ang[j],
            joint_compliant_alm,
            beta_ang,
        )
        return

    # REVOLUTE joint: isotropic linear + perpendicular angular penalties (2 scalars).
    if jt == JointType.REVOLUTE:
        i_lin = c_start + 0
        i_ang = c_start + 1
        qd_start = joint_qd_start[j]
        q_wp = wp.transform_get_rotation(X_wp)
        P_lin, P_ang = build_joint_projectors(jt, joint_axis, qd_start, 0, 1, q_wp)

        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        C_vec_lin = P_lin * (x_c - x_p)
        joint_lambda_lin[j] = _update_joint_structural_dual(
            C_vec_lin,
            P_lin * joint_C0_lin[j],
            stab_alpha,
            joint_penalty_k,
            joint_rho,
            joint_material_k,
            joint_is_hard,
            i_lin,
            joint_lambda_lin[j],
            joint_compliant_alm,
            beta_lin,
        )

        q_wc = wp.transform_get_rotation(X_wc)
        q_wp_rest = wp.transform_get_rotation(X_wp_rest)
        q_wc_rest = wp.transform_get_rotation(X_wc_rest)
        kappa = compute_kappa(q_wp, q_wc, q_wp_rest, q_wc_rest)
        kappa_perp = P_ang * kappa
        lam_old = P_ang * joint_lambda_ang[j]
        joint_lambda_ang[j] = _update_joint_structural_dual(
            kappa_perp,
            P_ang * joint_C0_ang[j],
            stab_alpha,
            joint_penalty_k,
            joint_rho,
            joint_material_k,
            joint_is_hard,
            i_ang,
            lam_old,
            joint_compliant_alm,
            beta_ang,
        )

        # Drive/limit dual update for free angular DOF (slot c_start + 2)
        dof_idx = qd_start
        axis_dl = _load_joint_axis_drive_limit(
            dof_idx,
            joint_target_q_start[j],
            c_start + 2,
            joint_target_ke,
            joint_target_kd,
            joint_target_q,
            joint_target_qd,
            joint_limit_lower,
            joint_limit_upper,
            joint_limit_ke,
            joint_limit_kd,
            joint_penalty_k,
            joint_compliant_alm,
        )
        has_drive = _drive_row_needs_state_update(axis_dl.material_drive_ke, axis_dl.drive_kd, joint_compliant_alm)
        has_limits = _limit_row_exists(axis_dl.material_limit_ke, axis_dl.lower, axis_dl.upper)

        if has_drive or has_limits:
            a = wp.normalize(joint_axis[qd_start])
            theta_abs = wp.dot(kappa, a) + joint_rest_angle[dof_idx]
            q_wp_prev = wp.transform_get_rotation(X_wp_prev)
            q_wc_prev = wp.transform_get_rotation(X_wc_prev)
            omega_p = quat_velocity(q_wp, q_wp_prev, dt)
            omega_c = quat_velocity(q_wc, q_wc_prev, dt)
            _kappa, J_world = compute_kappa_and_jacobian(q_wp, q_wc, q_wp_rest, q_wc_rest)
            rate = wp.dot(compute_kappa_dot(J_world, omega_p, omega_c), a)
            _update_joint_axis_drive_limit_state(
                axis_dl,
                theta_abs,
                rate,
                has_drive,
                has_limits,
                dof_idx,
                c_start + 2,
                beta_ang,
                1.0 / dt,
                joint_penalty_k,
                joint_material_k,
                joint_drive_limit_support,
                joint_drive_lambda,
                joint_limit_lambda,
                joint_compliant_alm,
            )
        return

    # PRISMATIC joint: perpendicular linear + isotropic angular penalties (2 scalars).
    if jt == JointType.PRISMATIC:
        i_lin = c_start + 0
        i_ang = c_start + 1
        qd_start = joint_qd_start[j]
        q_wp = wp.transform_get_rotation(X_wp)
        P_lin, P_ang = build_joint_projectors(jt, joint_axis, qd_start, 1, 0, q_wp)

        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        C_vec = x_c - x_p
        C_vec_perp = P_lin * C_vec
        # P_lin rotates with the parent; re-project stored lambda into the current
        # constrained subspace before accumulating.
        lam_old = P_lin * joint_lambda_lin[j]
        joint_lambda_lin[j] = _update_joint_structural_dual(
            C_vec_perp,
            P_lin * joint_C0_lin[j],
            stab_alpha,
            joint_penalty_k,
            joint_rho,
            joint_material_k,
            joint_is_hard,
            i_lin,
            lam_old,
            joint_compliant_alm,
            beta_lin,
        )

        q_wc = wp.transform_get_rotation(X_wc)
        q_wp_rest = wp.transform_get_rotation(X_wp_rest)
        q_wc_rest = wp.transform_get_rotation(X_wc_rest)
        kappa = compute_kappa(q_wp, q_wc, q_wp_rest, q_wc_rest)
        kappa_perp = P_ang * kappa
        joint_lambda_ang[j] = _update_joint_structural_dual(
            kappa_perp,
            P_ang * joint_C0_ang[j],
            stab_alpha,
            joint_penalty_k,
            joint_rho,
            joint_material_k,
            joint_is_hard,
            i_ang,
            joint_lambda_ang[j],
            joint_compliant_alm,
            beta_ang,
        )

        # Drive/limit dual update for free linear DOF (slot c_start + 2)
        dof_idx = qd_start
        axis_dl = _load_joint_axis_drive_limit(
            dof_idx,
            joint_target_q_start[j],
            c_start + 2,
            joint_target_ke,
            joint_target_kd,
            joint_target_q,
            joint_target_qd,
            joint_limit_lower,
            joint_limit_upper,
            joint_limit_ke,
            joint_limit_kd,
            joint_penalty_k,
            joint_compliant_alm,
        )
        has_drive = _drive_row_needs_state_update(axis_dl.material_drive_ke, axis_dl.drive_kd, joint_compliant_alm)
        has_limits = _limit_row_exists(axis_dl.material_limit_ke, axis_dl.lower, axis_dl.upper)

        if has_drive or has_limits:
            axis_w_dl = wp.normalize(wp.quat_rotate(q_wp, joint_axis[qd_start]))
            d_along = wp.dot(C_vec, axis_w_dl)
            x_p_prev = wp.transform_get_translation(X_wp_prev)
            x_c_prev = wp.transform_get_translation(X_wc_prev)
            rate = wp.dot((C_vec - (x_c_prev - x_p_prev)) / dt, axis_w_dl)
            _update_joint_axis_drive_limit_state(
                axis_dl,
                d_along,
                rate,
                has_drive,
                has_limits,
                dof_idx,
                c_start + 2,
                beta_lin,
                1.0 / dt,
                joint_penalty_k,
                joint_material_k,
                joint_drive_limit_support,
                joint_drive_lambda,
                joint_limit_lambda,
                joint_compliant_alm,
            )
        return

    # D6 joint: projected linear + projected angular penalties (2 scalars).
    if jt == JointType.D6:
        i_lin = c_start + 0
        i_ang = c_start + 1
        lin_count = joint_dof_dim[j, 0]
        ang_count = joint_dof_dim[j, 1]
        qd_start = joint_qd_start[j]
        q_wp_rot = wp.transform_get_rotation(X_wp)
        P_lin, P_ang = build_joint_projectors(jt, joint_axis, qd_start, lin_count, ang_count, q_wp_rot)

        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        C_vec = x_c - x_p
        x_p_prev = wp.transform_get_translation(X_wp_prev)
        x_c_prev = wp.transform_get_translation(X_wc_prev)
        C_vec_prev = x_c_prev - x_p_prev
        if lin_count < 3:
            C_vec_perp = P_lin * C_vec
            # P_lin rotates with the parent; re-project stored lambda into the current
            # constrained subspace before accumulating.
            lam_old = P_lin * joint_lambda_lin[j]
            joint_lambda_lin[j] = _update_joint_structural_dual(
                C_vec_perp,
                P_lin * joint_C0_lin[j],
                stab_alpha,
                joint_penalty_k,
                joint_rho,
                joint_material_k,
                joint_is_hard,
                i_lin,
                lam_old,
                joint_compliant_alm,
                beta_lin,
            )

        q_wc = wp.transform_get_rotation(X_wc)
        q_wp_prev = wp.transform_get_rotation(X_wp_prev)
        q_wc_prev = wp.transform_get_rotation(X_wc_prev)
        q_wp_rest = wp.transform_get_rotation(X_wp_rest)
        q_wc_rest = wp.transform_get_rotation(X_wc_rest)
        kappa, J_world = compute_kappa_and_jacobian(q_wp_rot, q_wc, q_wp_rest, q_wc_rest)
        omega_p = quat_velocity(q_wp_rot, q_wp_prev, dt)
        omega_c = quat_velocity(q_wc, q_wc_prev, dt)
        dkappa_dt = compute_kappa_dot(J_world, omega_p, omega_c)
        inv_dt = 1.0 / dt
        if ang_count < 3:
            kappa_perp = P_ang * kappa
            lam_old = P_ang * joint_lambda_ang[j]
            joint_lambda_ang[j] = _update_joint_structural_dual(
                kappa_perp,
                P_ang * joint_C0_ang[j],
                stab_alpha,
                joint_penalty_k,
                joint_rho,
                joint_material_k,
                joint_is_hard,
                i_ang,
                lam_old,
                joint_compliant_alm,
                beta_ang,
            )

        # Drive/limit dual update for D6 free DOFs
        target_q_base = joint_target_q_start[j]
        for li in range(3):
            if li < lin_count:
                dof_idx = qd_start + li
                axis_dl = _load_joint_axis_drive_limit(
                    dof_idx,
                    target_q_base + li,
                    c_start + 2 + li,
                    joint_target_ke,
                    joint_target_kd,
                    joint_target_q,
                    joint_target_qd,
                    joint_limit_lower,
                    joint_limit_upper,
                    joint_limit_ke,
                    joint_limit_kd,
                    joint_penalty_k,
                    joint_compliant_alm,
                )
                has_drive = _drive_row_needs_state_update(
                    axis_dl.material_drive_ke, axis_dl.drive_kd, joint_compliant_alm
                )
                has_limits = _limit_row_exists(axis_dl.material_limit_ke, axis_dl.lower, axis_dl.upper)

                if has_drive or has_limits:
                    axis_w_dl = wp.normalize(wp.quat_rotate(q_wp_rot, joint_axis[dof_idx]))
                    _update_joint_axis_drive_limit_state(
                        axis_dl,
                        wp.dot(C_vec, axis_w_dl),
                        wp.dot((C_vec - C_vec_prev) / dt, axis_w_dl),
                        has_drive,
                        has_limits,
                        dof_idx,
                        c_start + 2 + li,
                        beta_lin,
                        inv_dt,
                        joint_penalty_k,
                        joint_material_k,
                        joint_drive_limit_support,
                        joint_drive_lambda,
                        joint_limit_lambda,
                        joint_compliant_alm,
                    )

        for ai in range(3):
            if ai < ang_count:
                dof_idx = qd_start + lin_count + ai
                axis_dl = _load_joint_axis_drive_limit(
                    dof_idx,
                    target_q_base + lin_count + ai,
                    c_start + 2 + lin_count + ai,
                    joint_target_ke,
                    joint_target_kd,
                    joint_target_q,
                    joint_target_qd,
                    joint_limit_lower,
                    joint_limit_upper,
                    joint_limit_ke,
                    joint_limit_kd,
                    joint_penalty_k,
                    joint_compliant_alm,
                )
                has_drive = _drive_row_needs_state_update(
                    axis_dl.material_drive_ke, axis_dl.drive_kd, joint_compliant_alm
                )
                has_limits = _limit_row_exists(axis_dl.material_limit_ke, axis_dl.lower, axis_dl.upper)

                if has_drive or has_limits:
                    a_dl = wp.normalize(joint_axis[dof_idx])
                    _update_joint_axis_drive_limit_state(
                        axis_dl,
                        wp.dot(kappa, a_dl) + joint_rest_angle[dof_idx],
                        wp.dot(dkappa_dt, a_dl),
                        has_drive,
                        has_limits,
                        dof_idx,
                        c_start + 2 + lin_count + ai,
                        beta_ang,
                        inv_dt,
                        joint_penalty_k,
                        joint_material_k,
                        joint_drive_limit_support,
                        joint_drive_lambda,
                        joint_limit_lambda,
                        joint_compliant_alm,
                    )
        return


@wp.kernel
def update_duals_body_body_contacts(
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[int],
    rigid_contact_shape1: wp.array[int],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_offset0: wp.array[wp.vec3],
    rigid_contact_offset1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_margin0: wp.array[float],
    rigid_contact_margin1: wp.array[float],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    contact_material_mu: wp.array[float],
    contact_C0: wp.array[wp.vec3],
    stab_alpha: float,
    legacy_hard_contacts: int,
    contact_compliant_alm: int,
    contact_material_ke: wp.array[float],
    contact_tangent_rho: wp.array[float],
    contact_normal_rho: wp.array[float],
    beta: float,
    # Input/output
    contact_penalty_k: wp.array[float],
    contact_lambda: wp.array[wp.vec3],
):
    """Update body-body contact duals and legacy penalty stiffness."""
    idx = wp.tid()
    if idx >= rigid_contact_count[0]:
        return

    shape_id_0 = rigid_contact_shape0[idx]
    shape_id_1 = rigid_contact_shape1[idx]
    if shape_id_0 < 0 or shape_id_1 < 0:
        return
    body_id_0 = shape_body[shape_id_0]
    body_id_1 = shape_body[shape_id_1]

    if body_id_0 < 0 and body_id_1 < 0:
        return

    cp0_local = rigid_contact_point0[idx]
    cp1_local = rigid_contact_point1[idx]
    anchor0_local = cp0_local + rigid_contact_offset0[idx]
    anchor1_local = cp1_local + rigid_contact_offset1[idx]

    if body_id_0 >= 0:
        p0_world = wp.transform_point(body_q[body_id_0], cp0_local)
        a0_world = wp.transform_point(body_q[body_id_0], anchor0_local)
        a0_prev = wp.transform_point(body_q_prev[body_id_0], anchor0_local)
    else:
        p0_world = cp0_local
        a0_world = anchor0_local
        a0_prev = anchor0_local

    if body_id_1 >= 0:
        p1_world = wp.transform_point(body_q[body_id_1], cp1_local)
        a1_world = wp.transform_point(body_q[body_id_1], anchor1_local)
        a1_prev = wp.transform_point(body_q_prev[body_id_1], anchor1_local)
    else:
        p1_world = cp1_local
        a1_world = anchor1_local
        a1_prev = anchor1_local

    n = rigid_contact_normal[idx]
    C_n_raw = -contact_surface_separation(p0_world, p1_world, n, rigid_contact_margin0[idx], rigid_contact_margin1[idx])

    if legacy_hard_contacts == 1 or contact_compliant_alm == 1:
        if contact_compliant_alm == 1:
            rho_n = contact_normal_rho[idx]
        else:
            rho_n = contact_penalty_k[idx]
        material_k = contact_material_ke[idx]
        lam_vec = contact_lambda[idx]
        mu = contact_material_mu[idx]

        C0_vec = contact_C0[idx]
        C0_n = wp.dot(n, C0_vec)
        C_stab_n = C_n_raw - stab_alpha * C0_n
        C0_t_vec = C0_vec - n * C0_n

        # Bypass C0 stabilization on separation so normal support releases fully.
        if C_n_raw < 0.0:
            C_stab_n = C_n_raw

        rel_disp = (a0_world - a0_prev) - (a1_world - a1_prev)
        tangential_disp = rel_disp - n * wp.dot(n, rel_disp)
        tangent_residual = tangential_disp + (1.0 - stab_alpha) * C0_t_vec

        if contact_compliant_alm == 1:
            contact_lambda[idx] = _compliant_contact_dual_step(
                lam_vec,
                n,
                C_stab_n,
                tangent_residual,
                material_k,
                mu,
                rho_n,
                contact_tangent_rho[idx],
            )
        else:
            lam_n_old = wp.dot(lam_vec, n)
            lam_t_old = lam_vec - n * lam_n_old
            lam_n_new = wp.max(lam_n_old + rho_n * C_stab_n, 0.0)
            lam_t_new = lam_t_old + rho_n * tangent_residual
            cone_limit = mu * lam_n_new
            lam_t_new = _project_coulomb_tangent(lam_t_new, wp.length(lam_t_new), cone_limit)
            contact_lambda[idx] = n * lam_n_new + lam_t_new

    if contact_compliant_alm == 0 and C_n_raw > 0.0:
        contact_penalty_k[idx] = _ramp_penalty_k(contact_penalty_k[idx], contact_material_ke[idx], beta, C_n_raw)


@wp.kernel
def update_duals_body_particle_contacts(
    body_particle_contact_count: wp.array[int],
    soft_contact_indices: wp.array[wp.vec3i],
    body_particle_contact_shape: wp.array[int],
    body_particle_contact_body_pos: wp.array[wp.vec3],
    body_particle_contact_normal: wp.array[wp.vec3],
    soft_contact_barycentric: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    shape_body: wp.array[int],
    shape_margin: wp.array[float],
    body_q: wp.array[wp.transform],
    body_particle_contact_material_ke: wp.array[float],
    beta: float,
    body_particle_contact_penalty_k: wp.array[float],
):
    """
    Update AVBD penalty parameters for body-particle soft contacts (per-iteration).

    Ramps each contact's penalty by beta * penetration, clamped to the per-contact material
    stiffness ceiling. Covers all soft contacts via the record's -1-padded corners: a particle
    record uses the particle position; an edge/face record uses the barycentric point over its
    2-3 soft particles -- matching _eval_soft_ef_contact.
    """
    idx = wp.tid()
    if idx >= body_particle_contact_count[0]:
        return

    corners = soft_contact_indices[idx]
    shape_idx = body_particle_contact_shape[idx]
    body_idx = shape_body[shape_idx] if shape_idx >= 0 else -1

    stiffness = body_particle_contact_material_ke[idx]

    X_wb = wp.transform_identity()
    if body_idx >= 0:
        X_wb = body_q[body_idx]

    cp_world = wp.transform_point(X_wb, body_particle_contact_body_pos[idx])
    margin = shape_margin[shape_idx] if shape_idx >= 0 and shape_margin.shape[0] > 0 else 0.0
    n = body_particle_contact_normal[idx]

    if corners[1] < 0:
        # Particle contact (p, -1, -1).
        p = corners[0]
        contact_pos = particle_q[p]
        radius = particle_radius[p]
    else:
        # Edge/face: barycentric point + max radius over the record's valid corners.
        bary = soft_contact_barycentric[idx]
        v0 = corners[0]
        c1 = corners[1]
        c2 = corners[2]
        contact_pos = bary[0] * particle_q[v0]
        radius = particle_radius[v0]
        if c1 >= 0:
            contact_pos += bary[1] * particle_q[c1]
            radius = wp.max(radius, particle_radius[c1])
        if c2 >= 0:
            contact_pos += bary[2] * particle_q[c2]
            radius = wp.max(radius, particle_radius[c2])

    penetration = -(wp.dot(n, contact_pos - cp_world) - radius - margin)
    penetration = wp.max(0.0, penetration)

    k = body_particle_contact_penalty_k[idx]
    body_particle_contact_penalty_k[idx] = wp.min(k + beta * penetration, stiffness)


# -----------------------------
# Post-iteration kernels (after all iterations)
# -----------------------------
@wp.kernel
def update_body_velocity(
    dt: float,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_qd_mirror: wp.array[wp.spatial_vector],
    body_q_out: wp.array[wp.transform],
):
    """
    Update body velocities from position changes (world frame).

    Computes linear and angular velocities using finite differences.
    Also transfers the final body poses to body_q_out (fused copy from
    the in-place Gauss-Seidel iteration buffer to state_out).

    Linear: v = (com_current - com_prev) / dt
    Angular: omega from quaternion difference dq = q * q_prev^-1

    Args:
        dt: Time step.
        body_q: Current body transforms (world), from state_in (in-place iteration buffer).
        body_com: Center of mass offsets (local frame).
        body_q_prev: Previous body transforms (input/output), advanced to the
            current pose for the next step. ``SolverVBD.reset()`` is the supported
            way to establish a new baseline after a discontinuous pose change.
        body_qd: Output body velocities (spatial vectors, world frame), bound to state_out.
        body_qd_mirror: Output body velocities, bound to state_in. Mirrors body_qd so the
            next step's forward integrator sees the finalized velocity even when the
            caller's Python-level state swap is not recorded in a captured CUDA graph.
        body_q_out: Output body transforms (state_out), fused copy of body_q.
    """
    tid = wp.tid()

    # Read transforms
    pose = body_q[tid]
    pose_prev = body_q_prev[tid]

    x = wp.transform_get_translation(pose)
    x_prev = wp.transform_get_translation(pose_prev)
    q = wp.transform_get_rotation(pose)
    q_prev = wp.transform_get_rotation(pose_prev)

    # Compute COM positions
    com_local = body_com[tid]
    x_com = x + wp.quat_rotate(q, com_local)
    x_com_prev = x_prev + wp.quat_rotate(q_prev, com_local)

    # Linear velocity
    v = (x_com - x_com_prev) / dt

    # Angular velocity
    omega = quat_velocity(q, q_prev, dt)

    body_qd[tid] = wp.spatial_vector(v, omega)

    # Mirror to state_in (CUDA-graph-capture safety).
    body_qd_mirror[tid] = wp.spatial_vector(v, omega)

    # Advance body_q_prev for next step (for kinematic bodies this is the only write).
    body_q_prev[tid] = pose

    body_q_out[tid] = pose


@wp.kernel
def update_rod_dahl_state(
    # Joint geometry
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_constraint_start: wp.array[int],
    joint_penalty_k: wp.array[float],
    joint_material_k: wp.array[float],
    joint_is_hard: wp.array[wp.int32],
    joint_compliant_alm: int,
    joint_rod_rest_kb_local: wp.array[wp.vec3],
    joint_rod_rest_twist: wp.array[float],
    # Body states (final, after solver convergence)
    body_q: wp.array[wp.transform],
    # Dahl model parameters (PER-JOINT arrays, isotropic)
    joint_eps_max: wp.array[float],
    joint_tau: wp.array[float],
    # Dahl state (inputs - from previous timestep, outputs - to next timestep) - component-wise (vec3)
    joint_sigma_prev: wp.array[wp.vec3],  # input/output
    joint_kappa_prev: wp.array[wp.vec3],  # input/output
    joint_dkappa_prev: wp.array[wp.vec3],  # input/output (stores Delta kappa)
):
    """
    Persist rod Dahl hysteresis state after solver convergence.

    State is diagonal in [bend_x, bend_y, twist_z]. Compliant ALM uses authored
    material stiffness; legacy AVBD advances only active soft rod modes.
    """
    j = wp.tid()
    zero = wp.vec3(0.0)

    if joint_type[j] != JointType.ROD:
        return

    parent = joint_parent[j]
    child = joint_child[j]
    if child < 0:
        return

    if parent >= 0:
        X_wp = body_q[parent] * joint_X_p[j]
    else:
        X_wp = joint_X_p[j]
    X_wc = body_q[child] * joint_X_c[j]

    q_wp = wp.transform_get_rotation(X_wp)
    q_wc = wp.transform_get_rotation(X_wc)

    kappa_final = compute_geometric_rod_kappa_cached_z(
        q_wp,
        q_wc,
        joint_rod_rest_kb_local[j],
        joint_rod_rest_twist[j],
    )

    c_start = joint_constraint_start[j]
    k_dahl = _rod_dahl_active_stiffness(
        c_start,
        joint_penalty_k,
        joint_material_k,
        joint_is_hard,
        joint_compliant_alm,
    )

    # Inactive modes clear stress and use the current strain as the next baseline.
    if not joint_enabled[j] or (k_dahl[0] <= 0.0 and k_dahl[1] <= 0.0 and k_dahl[2] <= 0.0):
        joint_kappa_prev[j] = kappa_final
        joint_sigma_prev[j] = zero
        joint_dkappa_prev[j] = zero
        return

    # Stored Dahl state from the previous converged timestep.
    kappa_old = joint_kappa_prev[j]
    d_kappa_old = joint_dkappa_prev[j]
    sigma_old = joint_sigma_prev[j]
    d_kappa = _rod_bend_twist_delta(kappa_final, kappa_old)

    eps_max = joint_eps_max[j]  # Maximum persistent strain [rad]
    tau = joint_tau[j]  # Memory decay length [rad]

    if eps_max <= 0.0 or tau <= 0.0:
        joint_sigma_prev[j] = zero
        joint_kappa_prev[j] = kappa_final
        joint_dkappa_prev[j] = d_kappa
        return

    sigma_final_out = zero
    d_kappa_out = zero

    for axis in range(3):
        sigma_max = k_dahl[axis] * eps_max  # [N*m]
        if sigma_max <= 0.0:
            continue

        sigma_i_next, _direction = _advance_dahl_axis(
            d_kappa[axis],
            d_kappa_old[axis],
            sigma_old[axis],
            sigma_max,
            tau,
        )
        sigma_final_out[axis] = sigma_i_next
        d_kappa_out[axis] = d_kappa[axis]

    # Store final vector state for next timestep: [bend_x, bend_y, twist_z].
    joint_sigma_prev[j] = sigma_final_out
    joint_kappa_prev[j] = kappa_final
    joint_dkappa_prev[j] = d_kappa_out
