# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared discrete rod bend/twist geometry for solver kernels.

Endpoint quaternions describe normalized material-frame rotations into world
space. Local +Z is the rod tangent and local +X is its material normal. Bend
residuals use the parent material frame; twist is the signed angle after Bishop
transport, with residuals wrapped to [-pi, pi]. Jacobians map infinitesimal
world-space endpoint rotations to these local strain coordinates.

Curvature is capped near folds. At an exact fold, a material-axis fallback
chooses the curvature direction and its derivative is zero; twist derivatives
use bounded tangent spin when transport is singular. These guards preserve the
existing rod formulation rather than defining a smooth derivative at a fold.

Force assembly, material laws, and solver state updates belong to each solver.
"""

import warp as wp

_SMALL_ANGLE_EPS = wp.constant(1.0e-7)
"""Small-angle threshold for Bishop transport."""

_SMALL_LENGTH_EPS = wp.constant(1.0e-9)
"""Small-length threshold for normalizing material axes."""

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


@wp.func
def quat_rotate_local_z(q: wp.quat) -> wp.vec3:
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
def measure_rod_bend_twist_z(q_wp: wp.quat, q_wc: wp.quat) -> RodBendTwistMeasure:
    """Measure bend/twist for rod joints whose material tangent is local +Z.

    The fixed rod material basis is local ``+X, +Y, +Z``.
    Both input quaternions must be normalized so the rotated basis axes are
    orthonormal.
    """
    t0 = quat_rotate_local_z(q_wp)
    t1 = quat_rotate_local_z(q_wc)
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
def rod_bend_twist_jacobian_z_from_measure(
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
def assemble_geometric_rod_kappa_z(
    q_wp: wp.quat,
    kb_now_world: wp.vec3,
    twist_now: float,
    kb_rest_local: wp.vec3,
    twist_rest: float,
) -> wp.vec3:
    """Assemble [bend_x, bend_y, twist_z] for local +Z rod joints."""
    bend_now_local = wp.quat_rotate(wp.quat_inverse(q_wp), kb_now_world)
    bend_residual_local = bend_now_local - kb_rest_local
    # In the local +Z convention, parent-frame x/y are bend and z is twist.
    # Wrap the twist delta of two atan2 angles into [-pi, pi] to avoid a ~2*pi
    # jump when rest/current straddle the branch cut (one revolution per joint).
    twist_residual = _wrap_principal_angle(twist_now - twist_rest)
    return wp.vec3(bend_residual_local[0], bend_residual_local[1], twist_residual)


@wp.func
def rod_bend_twist_delta(kappa: wp.vec3, kappa_prev: wp.vec3) -> wp.vec3:
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
    """Geometric rod-joint strain residual for fixed local +Z.

    ``kb_rest_local`` is the rest curvature binormal in the parent material
    frame and ``twist_rest`` the rest twist, both measured by
    ``measure_rod_bend_twist_z`` at the rest relative rotation. The residual
    then vanishes at rest up to float rounding.
    """
    measure = measure_rod_bend_twist_z(q_wp, q_wc)
    return assemble_geometric_rod_kappa_z(q_wp, measure.kb_world, measure.twist, kb_rest_local, twist_rest)
