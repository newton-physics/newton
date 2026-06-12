# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Rigid body VBD solver kernels and utilities.

This module contains all rigid body-specific kernels, device functions, data structures,
and constants for the VBD solver's rigid body domain (AVBD algorithm).

Organization:
- Constants: Solver parameters and thresholds
- Data structures: RigidForceElementAdjacencyInfo and related structs
- Device functions: Helper functions for rigid body dynamics
- Utility kernels: Adjacency building
- Pre-iteration kernels: Forward integration, contact history restore, Dahl parameter computation
- Iteration kernels: Contact accumulation, rigid body solve, dual updates
- Post-iteration kernels: Velocity updates, Dahl state updates
"""

import warp as wp

from newton._src.core.types import MAXVAL
from newton._src.math import quat_velocity
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

_STICK_FLAG_ANCHOR = wp.constant(1)
"""contact_stick_flag value: frozen anchor (sticking kinematic/static contacts)"""

_STICK_FLAG_DEADZONE = wp.constant(2)
"""contact_stick_flag value: anti-creep deadzone (sticking dynamic-dynamic contacts)"""

# ---------------------------------
# Helper classes and device functions
# ---------------------------------


@wp.struct
class RigidContactHistory:
    lambda_: wp.array[wp.vec3]
    stick_flag: wp.array[wp.int32]
    penalty_k: wp.array[float]
    point0: wp.array[wp.vec3]
    point1: wp.array[wp.vec3]
    offset0: wp.array[wp.vec3]
    offset1: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]


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
def _normalize_with_fallback(v: wp.vec3, fallback: wp.vec3) -> wp.vec3:
    v_len = wp.length(v)
    if v_len > _SMALL_LENGTH_EPS:
        return v / v_len
    fb_len = wp.length(fallback)
    if fb_len > _SMALL_LENGTH_EPS:
        return fallback / fb_len
    return wp.vec3(1.0, 0.0, 0.0)


@wp.func
def _project_perp(v: wp.vec3, axis: wp.vec3) -> wp.vec3:
    return v - axis * wp.dot(v, axis)


@wp.func
def _project_along_unit_axis(v: wp.vec3, axis: wp.vec3) -> wp.vec3:
    """Project v onto an already-unit axis."""
    return axis * wp.dot(v, axis)


@wp.func
def _quat_rotate_local_z(q: wp.quat) -> wp.vec3:
    """Rotate local +Z by a unit quaternion; the third rotation-matrix column."""
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    return wp.vec3(2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y))


@wp.func
def _korner_cable_deformation_z(q_wp: wp.quat, q_wc: wp.quat) -> wp.vec3:
    """Korner/Audoly bend-twist deformation measure for local +Z cable frames.

    The measure is ``2 * Im(inv(q_parent) * q_child)`` with the relative
    quaternion sign chosen so the scalar component is non-negative. It is
    small-angle equivalent to the DER bend/twist vector, but stays bounded at
    sharp turns instead of requiring the finite-curvature-binormal cap.
    """
    q_rel = wp.mul(wp.quat_inverse(q_wp), q_wc)
    sign = 1.0
    if q_rel[3] < 0.0:
        sign = -1.0
    return sign * 2.0 * wp.vec3(q_rel[0], q_rel[1], q_rel[2])


@wp.func
def _korner_rest_quat(kb_rest_local: wp.vec3) -> wp.quat:
    """Reconstruct the rest relative rotation from the stored Korner rest vector.

    ``kb_rest_local`` is ``2 * Im(q_rel_rest)`` with a non-negative scalar part
    (see :func:`init_cable_rest_bend_twist`), so the unit rotation is recovered
    on the principal branch as ``(Im, sqrt(1 - |Im|^2))``.
    """
    v = 0.5 * kb_rest_local
    w = wp.sqrt(wp.max(0.0, 1.0 - wp.dot(v, v)))
    return wp.quat(v[0], v[1], v[2], w)


@wp.func
def _korner_cable_residual_z(q_wp: wp.quat, q_wc: wp.quat, kb_rest_local: wp.vec3) -> wp.vec3:
    """Rest-relative Korner/Audoly bend-twist strain for local +Z cables.

    The strain composes the current relative rotation against the rest one in
    rotation space, ``2 * Im(inv(q_rel_rest) * inv(q_wp) * q_wc)``, rather than
    linearly subtracting two deformation vectors. Composition keeps pre-curved
    rest shapes exact: pure twist on a bent rest produces no bend, and a rigid
    global rotation of the rest shape produces no strain. For a straight rest
    (``q_rel_rest`` identity) it reduces to ``2 * Im(inv(q_wp) * q_wc)``.
    """
    q_def = wp.mul(wp.quat_inverse(_korner_rest_quat(kb_rest_local)), wp.mul(wp.quat_inverse(q_wp), q_wc))
    sign = 1.0
    if q_def[3] < 0.0:
        sign = -1.0
    return sign * 2.0 * wp.vec3(q_def[0], q_def[1], q_def[2])


@wp.func
def _korner_cable_residual_directional_derivative_z(
    q_wp: wp.quat,
    q_wc: wp.quat,
    kb_rest_local: wp.vec3,
    omega_world: wp.vec3,
    is_parent: bool,
) -> wp.vec3:
    """Directional derivative of the rest-relative Korner strain for a world rotation.

    With ``A = inv(q_rel_rest) * inv(q_wp)`` and ``q_def = A * q_wc = (v, w)``
    (sign chosen so ``w >= 0``), the child-rotation derivative is
    ``(w I - [v]x)(R_A omega_world)`` and the parent derivative is its negation.
    Reduces to the rest-free derivative when ``q_rel_rest`` is the identity.
    """
    A = wp.mul(wp.quat_inverse(_korner_rest_quat(kb_rest_local)), wp.quat_inverse(q_wp))
    q_def = wp.mul(A, q_wc)
    sign = 1.0
    if q_def[3] < 0.0:
        sign = -1.0
    v = sign * wp.vec3(q_def[0], q_def[1], q_def[2])
    w = sign * q_def[3]
    a = wp.quat_rotate(A, omega_world)

    # (w I - [v]x) a == w a - v x a
    dkappa = w * a - wp.cross(v, a)
    if is_parent:
        dkappa = -dkappa
    return dkappa


@wp.func
def _korner_cable_jacobian_z(
    q_wp: wp.quat,
    q_wc: wp.quat,
    kb_rest_local: wp.vec3,
    is_parent: bool,
) -> wp.mat33:
    """Jacobian of rest-relative Korner [bend_x, bend_y, twist_z] for local +Z cables."""
    e0 = wp.vec3(1.0, 0.0, 0.0)
    e1 = wp.vec3(0.0, 1.0, 0.0)
    e2 = wp.vec3(0.0, 0.0, 1.0)

    j0 = _korner_cable_residual_directional_derivative_z(q_wp, q_wc, kb_rest_local, e0, is_parent)
    j1 = _korner_cable_residual_directional_derivative_z(q_wp, q_wc, kb_rest_local, e1, is_parent)
    j2 = _korner_cable_residual_directional_derivative_z(q_wp, q_wc, kb_rest_local, e2, is_parent)
    return wp.matrix_from_cols(j0, j1, j2)


@wp.func
def compute_geometric_cable_kappa_cached_z(
    q_wp: wp.quat,
    q_wc: wp.quat,
    kb_rest_local: wp.vec3,
    twist_rest: float,
) -> wp.vec3:
    """Geometric cable strain residual for fixed local +Z cables."""
    return _korner_cable_residual_z(q_wp, q_wc, kb_rest_local)


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
def _update_dual_vec3(
    C_vec: wp.vec3,
    C0: wp.vec3,
    alpha: float,
    k: float,
    lam: wp.vec3,
    is_hard: int,
):
    """Shared AVBD dual update for a vec3 constraint slot.

    Hard mode: stabilized constraint + lambda accumulation.
    Soft mode: lambda unchanged.

    Args:
        C_vec: Current constraint violation vector.
        C0: Initial constraint violation snapshot for stabilization.
        alpha: C0 stabilization factor.
        k: Current penalty stiffness.
        lam: Current Lagrange multiplier.
        is_hard: 1 for hard (AL), 0 for soft (penalty-only).

    Returns:
        wp.vec3: Updated Lagrange multiplier.
    """
    if is_hard == 1:
        C_stab = C_vec - alpha * C0
        lam_new = k * C_stab + lam
    else:
        lam_new = lam
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
    penalty_k: float,
    P: wp.mat33,
    lambda_ang: wp.vec3,
    C0_ang: wp.vec3,
    alpha: float,
    damping: float,
    dt: float,
):
    """Projected angular constraint force/Hessian using rotation-vector error (kappa).

    Generic evaluator for non-cable angular constraints. Computes force and
    Hessian in the constrained subspace defined by the orthogonal-complement
    projector P. Angular Dahl friction is cable-only and handled separately, so
    this evaluator carries no friction term.

    C0 stabilization: when alpha > 0 and C0_ang is nonzero, the effective
    kappa is kappa - alpha*C0_ang (initial violation snapshot).

    Special cases by projector:
      - P = I: isotropic (FIXED angular)
      - P = I - a*a^T: revolute (1 free angular axis)
      - arbitrary P: D6 (0-3 free angular axes)

    Returns:
        (tau_world, H_aa, kappa, J_world) -- constraint torque and Hessian in world
        frame, plus the rotation vector and world-frame Jacobian for reuse by the
        drive/limit block.
    """
    inv_dt = 1.0 / dt

    kappa_now_vec, J_world = compute_kappa_and_jacobian(q_wp, q_wc, q_wp_rest, q_wc_rest)
    kappa_stab = kappa_now_vec - alpha * C0_ang
    kappa_perp = P * kappa_stab

    # P_ang is constant for joint angular residuals, so lambda_ang should already
    # be in-basis. Project here too so stale or externally edited state cannot
    # apply force along a free angular DOF.
    f_local = penalty_k * kappa_perp + P * lambda_ang

    H_local = penalty_k * P

    if damping > 0.0:
        omega_p_world = quat_velocity(q_wp, q_wp_prev, dt)
        omega_c_world = quat_velocity(q_wc, q_wc_prev, dt)

        dkappa_dt_vec = compute_kappa_dot(J_world, omega_p_world, omega_c_world)
        dkappa_perp = P * dkappa_dt_vec
        f_damp_local = (damping * penalty_k) * dkappa_perp
        f_local = f_local + f_damp_local

        k_damp = (damping * inv_dt) * penalty_k
        H_local = H_local + k_damp * P

    H_aa = J_world * (H_local * wp.transpose(J_world))

    tau_world = J_world * f_local
    if not is_parent:
        tau_world = -tau_world

    return tau_world, H_aa, kappa_now_vec, J_world


@wp.func
def evaluate_cable_bend_twist_force_hessian_z(
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
    """Bend/twist torque and Hessian for SolverVBD local +Z cables.

    In the fixed cable material basis, local angular operators are diagonal:
    ``[bend_x, bend_y, twist_z]``. Keep them as vec3 row scales in the hot path
    instead of building dense local matrices.
    """
    inv_dt = 1.0 / dt

    kappa_now_vec = _korner_cable_residual_z(q_wp, q_wc, kb_rest_local)

    # Bend and twist decouple in the material basis: the angular energy is a sum
    # of independent quadratics in [bend_x, bend_y, twist_z], so elastic stiffness
    # and the friction Hessian have no off-diagonal coupling. Carry them as vec3
    # row scales; the dense angular block reappears below via J^T diag(H) J.
    f_local = wp.cw_mul(K_elastic_diag, kappa_now_vec) - C0_force + sigma0 + lambda_projected
    H_local_diag = K_elastic_diag + H_fric_diag

    if damping_active:
        kappa_prev_vec = _korner_cable_residual_z(q_wp_prev, q_wc_prev, kb_rest_local)
        dkappa_dt = (kappa_now_vec - kappa_prev_vec) * inv_dt
        f_local = f_local + wp.cw_mul(K_damp_diag, dkappa_dt)
        H_local_diag = H_local_diag + inv_dt * K_damp_diag

    J_body = _korner_cable_jacobian_z(q_wp, q_wc, kb_rest_local, is_parent)
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
    penalty_k: float,
    P: wp.mat33,
    lambda_lin: wp.vec3,
    C0_lin: wp.vec3,
    alpha: float,
    damping: float,
    dt: float,
):
    """Projected linear constraint force/Hessian for anchor coincidence.

    Generic evaluator for non-cable linear constraints. Computes C = x_c - x_p,
    projects with P, and returns force/Hessian in world frame.

    C0 stabilization: when alpha > 0 and C0_lin is nonzero, the effective
    constraint violation is C - alpha*C0 (initial violation snapshot).

    Special cases by projector:
      - P = I: isotropic (BALL, FIXED linear, REVOLUTE linear)
      - P = I - a*a^T: prismatic (1 free linear axis)
      - arbitrary P: D6 (0-3 free linear axes)

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

    # P_lin rotates per call -> must re-project lambda_lin (see build_joint_projectors).
    f_attachment = penalty_k * C_perp + P * lambda_lin

    # Fold damping into effective stiffness: K_eff = k*(1 + d/dt)*P
    if damping > 0.0:
        inv_dt = 1.0 / dt
        K_eff = penalty_k * (1.0 + damping * inv_dt) * P

        x_p_prev = wp.transform_get_translation(X_wp_prev)
        x_c_prev = wp.transform_get_translation(X_wc_prev)
        C_vec_prev = x_c_prev - x_p_prev
        dC_dt_perp = P * ((C_vec - C_vec_prev) * inv_dt)
        f_attachment = f_attachment + (damping * penalty_k) * dC_dt_perp
    else:
        K_eff = penalty_k * P

    rx = wp.skew(r)
    H_ll = K_eff
    H_al = rx * K_eff
    H_aa = wp.transpose(rx) * K_eff * rx

    force = f_attachment if is_parent else -f_attachment
    torque = wp.cross(r, force)

    return force, torque, H_ll, H_al, H_aa


@wp.func
def evaluate_cable_stretch_shear_force_hessian(
    X_wp: wp.transform,
    X_wc: wp.transform,
    X_wp_prev: wp.transform,
    X_wc_prev: wp.transform,
    parent_pose: wp.transform,
    child_pose: wp.transform,
    parent_com: wp.vec3,
    child_com: wp.vec3,
    is_parent: bool,
    K_linear: wp.mat33,
    C0_force: wp.vec3,
    lambda_projected: wp.vec3,
    K_damp: wp.mat33,
    damping_active: bool,
    dt: float,
):
    """Combined cable stretch/shear anchor force and VBD self-Hessian.

    The caller supplies the current-frame split matrix
    ``K = k_shear I + (k_stretch - k_shear) t t^T`` and mode-filtered
    C0/lambda terms. The Hessian treats K as fixed for this local solve,
    matching the generic linear projector path.
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
    f_attachment = K_linear * C_vec - C0_force + lambda_projected
    K_eff = K_linear

    if damping_active:
        inv_dt = 1.0 / dt
        x_p_prev = wp.transform_get_translation(X_wp_prev)
        x_c_prev = wp.transform_get_translation(X_wc_prev)
        C_vec_prev = x_c_prev - x_p_prev
        f_attachment = f_attachment + K_damp * ((C_vec - C_vec_prev) * inv_dt)
        K_eff = K_eff + inv_dt * K_damp

    rx = wp.skew(r)
    H_ll = K_eff
    H_al = rx * K_eff
    H_aa = wp.transpose(rx) * K_eff * rx

    force = f_attachment if is_parent else -f_attachment
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
    contact_ke: float,
    contact_ke_t: float,
    contact_kd: float,
    contact_lam: wp.vec3,
    friction_mu: float,
    friction_epsilon: float,
    hard_contact: int,
    dt: float,
    friction_c0: wp.vec3,
):
    """Compute augmented-Lagrangian contact forces and 3x3 Hessian blocks for a rigid contact pair.

    Hard contacts: ALM normal + displacement-based tangential friction with Coulomb cone clamping.
    The tangential constraint is the relative tangential displacement from body_q_prev to body_q,
    which correctly captures kinematic body motion.
    Soft contacts: velocity-based IPC friction with scalar penalty.

    Returns:
        10-tuple: (force_a, torque_a, H_ll_a, H_al_a, H_aa_a,
                   force_b, torque_b, H_ll_b, H_al_b, H_aa_b)
    """
    lam_n = wp.dot(contact_lam, contact_normal)

    if penetration_depth <= _SMALL_LENGTH_EPS and lam_n <= 0.0:
        zero_vec = wp.vec3(0.0)
        zero_mat = wp.mat33(0.0)
        return (zero_vec, zero_vec, zero_mat, zero_mat, zero_mat, zero_vec, zero_vec, zero_mat, zero_mat, zero_mat)

    f_n = contact_ke * penetration_depth + lam_n
    if contact_ke <= 0.0:
        zero_vec = wp.vec3(0.0)
        zero_mat = wp.mat33(0.0)
        return (zero_vec, zero_vec, zero_mat, zero_mat, zero_mat, zero_vec, zero_vec, zero_mat, zero_mat, zero_mat)
    f_n = wp.max(f_n, 0.0)

    if f_n == 0.0 and hard_contact == 0:
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

    x_c_a_now = contact_surface_point(X_wa, contact_point_a_local, contact_offset_a_local)
    x_c_b_now = contact_surface_point(X_wb, contact_point_b_local, contact_offset_b_local)
    x_c_a_prev = contact_surface_point(X_wa_prev, contact_point_a_local, contact_offset_a_local)
    x_c_b_prev = contact_surface_point(X_wb_prev, contact_point_b_local, contact_offset_b_local)

    n_outer = wp.outer(contact_normal, contact_normal)

    v_rel = (x_c_b_now - x_c_b_prev - x_c_a_now + x_c_a_prev) / dt
    v_dot_n = wp.dot(contact_normal, v_rel)

    if hard_contact == 1:
        f_total = contact_normal * f_n
        K_total = contact_ke * n_outer

        if friction_mu > 0.0 and f_n > 0.0:
            # ALM tangential friction with Coulomb cone clamping.
            # Tangential constraint: rel_disp + friction_c0
            # (friction_c0 = (1 - alpha) * C0_t, pre-scaled by the caller).
            v_t = v_rel - contact_normal * v_dot_n
            tangential_disp = -(v_t * dt)
            lam_t = contact_lam - contact_normal * lam_n
            f_t_vec = contact_ke_t * (tangential_disp + friction_c0) + lam_t
            f_t_len = wp.length(f_t_vec)
            cone_limit = friction_mu * f_n
            if f_t_len > cone_limit and f_t_len > 0.0:
                cone_ratio = cone_limit / f_t_len
                f_t_vec = f_t_vec * cone_ratio
            f_total = f_total + f_t_vec
            I3 = wp.identity(n=3, dtype=float)
            K_total = K_total + contact_ke_t * (I3 - n_outer)
    else:
        # Soft contact: normal penalty + IPC velocity-based friction
        f_total = contact_normal * f_n
        K_total = contact_ke * n_outer

        if friction_mu > 0.0 and f_n > 0.0:
            v_t = v_rel - contact_normal * v_dot_n
            f_friction, K_friction = compute_projected_isotropic_friction(
                friction_mu, f_n, contact_normal, v_t * dt, friction_epsilon * dt
            )
            f_total = f_total + f_friction
            K_total = K_total + K_friction

    if contact_kd > 0.0 and v_dot_n < 0.0 and f_n > 0.0:
        damping_coeff = contact_kd * contact_ke
        f_total = f_total - damping_coeff * v_dot_n * contact_normal
        K_total = K_total + (damping_coeff / dt) * n_outer

    r_a = x_c_a_now - x_com_a_now
    r_b = x_c_b_now - x_com_b_now

    r_a_skew = wp.skew(r_a)
    r_a_skew_T_K = wp.transpose(r_a_skew) * K_total
    h_aa_a = r_a_skew_T_K * r_a_skew
    h_al_a = -r_a_skew_T_K

    r_b_skew = wp.skew(r_b)
    r_b_skew_T_K = wp.transpose(r_b_skew) * K_total
    h_aa_b = r_b_skew_T_K * r_b_skew
    h_al_b = -r_b_skew_T_K

    return (
        -f_total,
        wp.cross(r_a, -f_total),
        K_total,
        h_al_a,
        h_aa_a,
        f_total,
        wp.cross(r_b, f_total),
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
        damping_coeff = kd * ke
        damping_hessian = (damping_coeff / dt) * wp.outer(n, n)
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
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
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

    penetration_depth = -(wp.dot(n, particle_pos - bx) - particle_radius[particle_index])
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
        # IPC-style regularization
        if u_norm > eps_u:
            f1_SF_over_x = 1.0 / u_norm
        else:
            f1_SF_over_x = (-u_norm / eps_u + 2.0) / eps_u

        # Factor common scalar; force aligned with u_t, Hessian proportional to projector
        scale = friction_mu * normal_load * f1_SF_over_x
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
    joint_cable_kb_rest_local: wp.array[wp.vec3],
    joint_cable_twist_rest: wp.array[float],
    joint_qd_start: wp.array[int],
    joint_target_q_start: wp.array[int],
    joint_constraint_start: wp.array[int],
    joint_penalty_k: wp.array[float],
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
    joint_lambda_lin: wp.array[wp.vec3],
    joint_lambda_ang: wp.array[wp.vec3],
    joint_C0_lin: wp.array[wp.vec3],
    joint_C0_ang: wp.array[wp.vec3],
    joint_is_hard: wp.array[wp.int32],
    avbd_alpha: float,
    joint_dof_dim: wp.array2d[int],
    joint_rest_angle: wp.array[float],
    dt: float,
):
    """Compute AVBD joint force and Hessian contributions for one body.

    Supported joint types: CABLE, BALL, FIXED, REVOLUTE, PRISMATIC, D6.
    Cable uses split stretch/shear and bend/twist helpers; other joints use
    projector-based linear/angular evaluators.

    Indexing:
        joint_constraint_start[j] is a solver-owned start offset into the per-constraint
        arrays (joint_penalty_k, joint_penalty_kd). Layout per joint type:
          - CABLE: 4 scalars -> [stretch, shear, bend, twist]
          - BALL:  1 scalar  -> [linear]
          - FIXED: 2 scalars -> [linear, angular]
          - REVOLUTE:  3 scalars -> [linear, angular, ang_drive_limit]
          - PRISMATIC: 3 scalars -> [linear, angular, lin_drive_limit]
          - D6: 2 + lin_count + ang_count scalars -> [linear, angular, per-DOF drive/limit]
        Drive/limit slots use AVBD-ramped stiffness via min(avbd_ke, model_ke).
        Drive/limit forces remain penalty-only (no lambda or C0 state).
    """
    jt = joint_type[joint_index]
    if (
        jt != JointType.CABLE
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

    if jt == JointType.CABLE:
        stretch_idx = c_start
        shear_idx = c_start + 1
        bend_idx = c_start + 2
        twist_idx = bend_idx + 1
        k_stretch = joint_penalty_k[stretch_idx]
        k_shear = joint_penalty_k[shear_idx]
        kd_stretch = joint_penalty_kd[stretch_idx]
        kd_shear = joint_penalty_kd[shear_idx]
        k_bend = joint_penalty_k[bend_idx]
        k_twist = joint_penalty_k[twist_idx]
        kd_bend = joint_penalty_kd[bend_idx]
        kd_twist = joint_penalty_kd[twist_idx]

        total_force = wp.vec3(0.0)
        total_torque = wp.vec3(0.0)
        total_H_ll = wp.mat33(0.0)
        total_H_al = wp.mat33(0.0)
        total_H_aa = wp.mat33(0.0)

        bend_active = k_bend > 0.0
        twist_active = k_twist > 0.0
        if bend_active or twist_active:
            K_elastic_diag = wp.vec3(k_bend, k_bend, k_twist)
            K_damp_diag = wp.vec3(kd_bend * k_bend, kd_bend * k_bend, kd_twist * k_twist)
            damping_active = (kd_bend > 0.0 and k_bend > 0.0) or (kd_twist > 0.0 and k_twist > 0.0)

            sigma = wp.vec3(0.0)
            H_fric_diag = wp.vec3(0.0)
            lambda_projected = wp.vec3(0.0)
            C0_force = wp.vec3(0.0)
            dahl_sigma = joint_sigma_start[joint_index]
            dahl_fric = joint_C_fric[joint_index]
            bend_hard = bend_active and joint_is_hard[bend_idx] == 1
            twist_hard = twist_active and joint_is_hard[twist_idx] == 1
            lambda_ang = wp.vec3(0.0)
            C0_ang = wp.vec3(0.0)
            if bend_hard or twist_hard:
                lambda_ang = joint_lambda_ang[joint_index]
                C0_ang = joint_C0_ang[joint_index]

            if bend_hard:
                lambda_projected = lambda_projected + wp.vec3(lambda_ang[0], lambda_ang[1], 0.0)
                C0_force = C0_force + (k_bend * avbd_alpha) * wp.vec3(C0_ang[0], C0_ang[1], 0.0)
            elif bend_active:
                sigma = sigma + wp.vec3(dahl_sigma[0], dahl_sigma[1], 0.0)
                H_fric_diag = H_fric_diag + wp.vec3(dahl_fric[0], dahl_fric[1], 0.0)

            if twist_hard:
                lambda_projected = lambda_projected + wp.vec3(0.0, 0.0, lambda_ang[2])
                C0_force = C0_force + (k_twist * avbd_alpha) * wp.vec3(0.0, 0.0, C0_ang[2])
            elif twist_active:
                sigma = sigma + wp.vec3(0.0, 0.0, dahl_sigma[2])
                H_fric_diag = H_fric_diag + wp.vec3(0.0, 0.0, dahl_fric[2])

            cable_torque, cable_H_aa, _cable_kappa, _cable_J = evaluate_cable_bend_twist_force_hessian_z(
                q_wp,
                q_wc,
                joint_cable_kb_rest_local[joint_index],
                joint_cable_twist_rest[joint_index],
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
            total_torque = total_torque + cable_torque
            total_H_aa = total_H_aa + cable_H_aa

        stretch_active = k_stretch > 0.0
        shear_active = k_shear > 0.0
        if stretch_active or shear_active:
            t_world = _quat_rotate_local_z(q_wp)
            P_stretch = wp.outer(t_world, t_world)
            I = wp.identity(3, float)

            # K = k_shear I + (k_stretch - k_shear) t t^T.
            K_linear = k_shear * I + (k_stretch - k_shear) * P_stretch
            k_damp_stretch = kd_stretch * k_stretch
            k_damp_shear = kd_shear * k_shear
            K_damp = k_damp_shear * I + (k_damp_stretch - k_damp_shear) * P_stretch
            damping_active = (kd_stretch > 0.0 and k_stretch > 0.0) or (kd_shear > 0.0 and k_shear > 0.0)

            lambda_projected = wp.vec3(0.0)
            C0_force = wp.vec3(0.0)
            stretch_hard = stretch_active and joint_is_hard[stretch_idx] == 1
            shear_hard = shear_active and joint_is_hard[shear_idx] == 1
            if stretch_hard or shear_hard:
                lambda_lin = joint_lambda_lin[joint_index]
                C0_lin = joint_C0_lin[joint_index]
                lambda_stretch = _project_along_unit_axis(lambda_lin, t_world)
                C0_stretch = _project_along_unit_axis(C0_lin, t_world)
                if stretch_hard:
                    lambda_projected = lambda_projected + lambda_stretch
                    C0_force = C0_force + (k_stretch * avbd_alpha) * C0_stretch
                if shear_hard:
                    lambda_projected = lambda_projected + (lambda_lin - lambda_stretch)
                    C0_force = C0_force + (k_shear * avbd_alpha) * (C0_lin - C0_stretch)

            f_l, t_l, Hll_l, Hal_l, Haa_l = evaluate_cable_stretch_shear_force_hessian(
                X_wp,
                X_wc,
                X_wp_prev,
                X_wc_prev,
                parent_pose,
                child_pose,
                parent_com,
                child_com,
                is_parent_body,
                K_linear,
                C0_force,
                lambda_projected,
                K_damp,
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

    # Hard/soft AL gating for the non-cable linear structural slot.
    lin_lambda = wp.vec3(0.0)
    lin_C0 = wp.vec3(0.0)
    lin_alpha = float(0.0)
    if joint_is_hard[c_start] == 1:
        lin_lambda = joint_lambda_lin[joint_index]
        lin_C0 = joint_C0_lin[joint_index]
        lin_alpha = avbd_alpha

    # BALL has no angular structural slot; other non-cable joints do.
    ang_lambda = wp.vec3(0.0)
    ang_C0 = wp.vec3(0.0)
    ang_alpha = float(0.0)
    if jt != JointType.BALL and joint_is_hard[c_start + 1] == 1:
        ang_lambda = joint_lambda_ang[joint_index]
        ang_C0 = joint_C0_ang[joint_index]
        ang_alpha = avbd_alpha

    if jt == JointType.BALL:
        k = joint_penalty_k[c_start]
        damping = joint_penalty_kd[c_start]
        if k > 0.0:
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
                k,
                P_I,
                lin_lambda,
                lin_C0,
                lin_alpha,
                damping,
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
        k_lin = joint_penalty_k[c_start + 0]
        kd_lin = joint_penalty_kd[c_start + 0]
        if k_lin > 0.0:
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
                k_lin,
                P_I,
                lin_lambda,
                lin_C0,
                lin_alpha,
                kd_lin,
                dt,
            )
        else:
            f_lin = wp.vec3(0.0)
            t_lin = wp.vec3(0.0)
            Hll_lin = wp.mat33(0.0)
            Hal_lin = wp.mat33(0.0)
            Haa_lin = wp.mat33(0.0)

        k_ang = joint_penalty_k[c_start + 1]
        kd_ang = joint_penalty_kd[c_start + 1]
        if k_ang > 0.0:
            t_ang, Haa_ang, _ang_kappa, _ang_J = evaluate_angular_constraint_force_hessian(
                q_wp,
                q_wc,
                q_wp_rest,
                q_wc_rest,
                q_wp_prev,
                q_wc_prev,
                is_parent_body,
                k_ang,
                P_I,
                ang_lambda,
                ang_C0,
                ang_alpha,
                kd_ang,
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

        k_lin = joint_penalty_k[c_start + 0]
        kd_lin = joint_penalty_kd[c_start + 0]
        if k_lin > 0.0:
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
                k_lin,
                P_lin,
                lin_lambda,
                lin_C0,
                lin_alpha,
                kd_lin,
                dt,
            )
        else:
            f_lin = wp.vec3(0.0)
            t_lin = wp.vec3(0.0)
            Hll_lin = wp.mat33(0.0)
            Hal_lin = wp.mat33(0.0)
            Haa_lin = wp.mat33(0.0)

        k_ang = joint_penalty_k[c_start + 1]
        kd_ang = joint_penalty_kd[c_start + 1]

        kappa_cached = wp.vec3(0.0)
        J_world_cached = wp.mat33(0.0)
        has_cached = False

        if k_ang > 0.0:
            t_ang, Haa_ang, kappa_cached, J_world_cached = evaluate_angular_constraint_force_hessian(
                q_wp,
                q_wc,
                q_wp_rest,
                q_wc_rest,
                q_wp_prev,
                q_wc_prev,
                is_parent_body,
                k_ang,
                P_ang,
                ang_lambda,
                ang_C0,
                ang_alpha,
                kd_ang,
                dt,
            )
            has_cached = True
        else:
            t_ang = wp.vec3(0.0)
            Haa_ang = wp.mat33(0.0)

        # Drive + limits on free angular DOF (AVBD slot c_start + 2)
        dof_idx = qd_start
        target_q_idx = joint_target_q_start[joint_index]
        model_drive_ke = joint_target_ke[dof_idx]
        drive_kd = joint_target_kd[dof_idx]
        target_pos = joint_target_q[target_q_idx]
        target_vel = joint_target_qd[dof_idx]
        lim_lower = joint_limit_lower[dof_idx]
        lim_upper = joint_limit_upper[dof_idx]
        model_limit_ke = joint_limit_ke[dof_idx]
        lim_kd = joint_limit_kd[dof_idx]

        has_drive = model_drive_ke > 0.0 or drive_kd > 0.0
        has_limits = model_limit_ke > 0.0 and (lim_lower > -MAXVAL or lim_upper < MAXVAL)

        avbd_ke = joint_penalty_k[c_start + 2]
        drive_ke = wp.min(avbd_ke, model_drive_ke)
        lim_ke = wp.min(avbd_ke, model_limit_ke)

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

            mode, err_pos = resolve_drive_limit_mode(theta_abs, target_pos, lim_lower, lim_upper, has_drive, has_limits)
            f_scalar = float(0.0)
            H_scalar = float(0.0)
            if mode == _DRIVE_LIMIT_MODE_LIMIT_LOWER or mode == _DRIVE_LIMIT_MODE_LIMIT_UPPER:
                lim_d = lim_kd * lim_ke
                f_scalar = lim_ke * err_pos + lim_d * dtheta_dt
                H_scalar = lim_ke + lim_d * inv_dt
            elif mode == _DRIVE_LIMIT_MODE_DRIVE:
                drive_d = drive_kd * drive_ke
                vel_err = dtheta_dt - target_vel
                f_scalar = drive_ke * err_pos + drive_d * vel_err
                H_scalar = drive_ke + drive_d * inv_dt

            if H_scalar > 0.0:
                tau_drive, Haa_drive = apply_angular_drive_limit_torque(a, J_world, is_parent_body, f_scalar, H_scalar)
                t_ang = t_ang + tau_drive
                Haa_ang = Haa_ang + Haa_drive

        return f_lin, t_lin + t_ang, Hll_lin, Hal_lin, Haa_lin + Haa_ang

    elif jt == JointType.PRISMATIC:
        qd_start = joint_qd_start[joint_index]
        axis_local = joint_axis[qd_start]
        P_lin, P_ang = build_joint_projectors(jt, joint_axis, qd_start, 1, 0, q_wp)

        k_lin = joint_penalty_k[c_start + 0]
        kd_lin = joint_penalty_kd[c_start + 0]
        if k_lin > 0.0:
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
                k_lin,
                P_lin,
                lin_lambda,
                lin_C0,
                lin_alpha,
                kd_lin,
                dt,
            )
        else:
            f_lin = wp.vec3(0.0)
            t_lin = wp.vec3(0.0)
            Hll_lin = wp.mat33(0.0)
            Hal_lin = wp.mat33(0.0)
            Haa_lin = wp.mat33(0.0)

        k_ang = joint_penalty_k[c_start + 1]
        kd_ang = joint_penalty_kd[c_start + 1]
        if k_ang > 0.0:
            t_ang, Haa_ang, _ang_kappa, _ang_J = evaluate_angular_constraint_force_hessian(
                q_wp,
                q_wc,
                q_wp_rest,
                q_wc_rest,
                q_wp_prev,
                q_wc_prev,
                is_parent_body,
                k_ang,
                P_ang,
                ang_lambda,
                ang_C0,
                ang_alpha,
                kd_ang,
                dt,
            )
        else:
            t_ang = wp.vec3(0.0)
            Haa_ang = wp.mat33(0.0)

        # Drive + limits on free linear DOF (AVBD slot c_start + 2)
        dof_idx = qd_start
        target_q_idx = joint_target_q_start[joint_index]
        model_drive_ke = joint_target_ke[dof_idx]
        drive_kd = joint_target_kd[dof_idx]
        target_pos = joint_target_q[target_q_idx]
        target_vel = joint_target_qd[dof_idx]
        lim_lower = joint_limit_lower[dof_idx]
        lim_upper = joint_limit_upper[dof_idx]
        model_limit_ke = joint_limit_ke[dof_idx]
        lim_kd = joint_limit_kd[dof_idx]

        has_drive = model_drive_ke > 0.0 or drive_kd > 0.0
        has_limits = model_limit_ke > 0.0 and (lim_lower > -MAXVAL or lim_upper < MAXVAL)

        avbd_ke = joint_penalty_k[c_start + 2]
        drive_ke = wp.min(avbd_ke, model_drive_ke)
        lim_ke = wp.min(avbd_ke, model_limit_ke)

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

            mode, err_pos = resolve_drive_limit_mode(d_along, target_pos, lim_lower, lim_upper, has_drive, has_limits)
            f_scalar = float(0.0)
            H_scalar = float(0.0)
            if mode == _DRIVE_LIMIT_MODE_LIMIT_LOWER or mode == _DRIVE_LIMIT_MODE_LIMIT_UPPER:
                lim_d = lim_kd * lim_ke
                f_scalar = lim_ke * err_pos + lim_d * dd_dt
                H_scalar = lim_ke + lim_d * inv_dt
            elif mode == _DRIVE_LIMIT_MODE_DRIVE:
                drive_d = drive_kd * drive_ke
                vel_err = dd_dt - target_vel
                f_scalar = drive_ke * err_pos + drive_d * vel_err
                H_scalar = drive_ke + drive_d * inv_dt

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
        k_lin = joint_penalty_k[c_start + 0]
        kd_lin = joint_penalty_kd[c_start + 0]

        if lin_count < 3 and k_lin > 0.0:
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
                k_lin,
                P_lin,
                lin_lambda,
                lin_C0,
                lin_alpha,
                kd_lin,
                dt,
            )
            total_force = total_force + f_l
            total_torque = total_torque + t_l
            total_H_ll = total_H_ll + Hll_l
            total_H_al = total_H_al + Hal_l
            total_H_aa = total_H_aa + Haa_l

        # Angular constraint (constrained when ang_count < 3)
        k_ang = joint_penalty_k[c_start + 1]
        kd_ang = joint_penalty_kd[c_start + 1]

        kappa_cached = wp.vec3(0.0)
        J_world_cached = wp.mat33(0.0)
        has_cached = False

        if ang_count < 3 and k_ang > 0.0:
            t_ang, Haa_ang, kappa_cached, J_world_cached = evaluate_angular_constraint_force_hessian(
                q_wp,
                q_wc,
                q_wp_rest,
                q_wc_rest,
                q_wp_prev,
                q_wc_prev,
                is_parent_body,
                k_ang,
                P_ang,
                ang_lambda,
                ang_C0,
                ang_alpha,
                kd_ang,
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
                    target_q_idx = target_q_base + li
                    model_drive_ke = joint_target_ke[dof_idx]
                    drive_kd = joint_target_kd[dof_idx]
                    target_pos = joint_target_q[target_q_idx]
                    target_vel = joint_target_qd[dof_idx]
                    lim_lower = joint_limit_lower[dof_idx]
                    lim_upper = joint_limit_upper[dof_idx]
                    model_limit_ke = joint_limit_ke[dof_idx]
                    lim_kd = joint_limit_kd[dof_idx]

                    has_drive = model_drive_ke > 0.0 or drive_kd > 0.0
                    has_limits = model_limit_ke > 0.0 and (lim_lower > -MAXVAL or lim_upper < MAXVAL)

                    avbd_ke = joint_penalty_k[c_start + 2 + li]
                    drive_ke = wp.min(avbd_ke, model_drive_ke)
                    lim_ke = wp.min(avbd_ke, model_limit_ke)

                    if has_drive or has_limits:
                        axis_w = wp.normalize(wp.quat_rotate(q_wp_rot, joint_axis[dof_idx]))
                        d_along = wp.dot(C_vec, axis_w)
                        dd_dt = wp.dot(dC_dt, axis_w)

                        mode, err_pos = resolve_drive_limit_mode(
                            d_along, target_pos, lim_lower, lim_upper, has_drive, has_limits
                        )
                        f_scalar = float(0.0)
                        H_scalar = float(0.0)
                        if mode == _DRIVE_LIMIT_MODE_LIMIT_LOWER or mode == _DRIVE_LIMIT_MODE_LIMIT_UPPER:
                            lim_d = lim_kd * lim_ke
                            f_scalar = lim_ke * err_pos + lim_d * dd_dt
                            H_scalar = lim_ke + lim_d * inv_dt
                        elif mode == _DRIVE_LIMIT_MODE_DRIVE:
                            drive_d = drive_kd * drive_ke
                            vel_err = dd_dt - target_vel
                            f_scalar = drive_ke * err_pos + drive_d * vel_err
                            H_scalar = drive_ke + drive_d * inv_dt

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
                    target_q_idx = target_q_base + lin_count + ai
                    model_drive_ke = joint_target_ke[dof_idx]
                    drive_kd = joint_target_kd[dof_idx]
                    target_pos = joint_target_q[target_q_idx]
                    target_vel = joint_target_qd[dof_idx]
                    lim_lower = joint_limit_lower[dof_idx]
                    lim_upper = joint_limit_upper[dof_idx]
                    model_limit_ke = joint_limit_ke[dof_idx]
                    lim_kd = joint_limit_kd[dof_idx]

                    has_drive = model_drive_ke > 0.0 or drive_kd > 0.0
                    has_limits = model_limit_ke > 0.0 and (lim_lower > -MAXVAL or lim_upper < MAXVAL)

                    avbd_ke = joint_penalty_k[c_start + 2 + lin_count + ai]
                    drive_ke = wp.min(avbd_ke, model_drive_ke)
                    lim_ke = wp.min(avbd_ke, model_limit_ke)

                    if has_drive or has_limits:
                        a = wp.normalize(joint_axis[dof_idx])
                        theta = wp.dot(kappa, a)
                        theta_abs = theta + joint_rest_angle[dof_idx]
                        dtheta_dt = wp.dot(dkappa_dt, a)

                        mode, err_pos = resolve_drive_limit_mode(
                            theta_abs, target_pos, lim_lower, lim_upper, has_drive, has_limits
                        )
                        f_scalar = float(0.0)
                        H_scalar = float(0.0)
                        if mode == _DRIVE_LIMIT_MODE_LIMIT_LOWER or mode == _DRIVE_LIMIT_MODE_LIMIT_UPPER:
                            lim_d = lim_kd * lim_ke
                            f_scalar = lim_ke * err_pos + lim_d * dtheta_dt
                            H_scalar = lim_ke + lim_d * inv_dt
                        elif mode == _DRIVE_LIMIT_MODE_DRIVE:
                            drive_d = drive_kd * drive_ke
                            vel_err = dtheta_dt - target_vel
                            f_scalar = drive_ke * err_pos + drive_d * vel_err
                            H_scalar = drive_ke + drive_d * inv_dt

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


# -----------------------------
# Pre-iteration kernels (once per step)
# -----------------------------
@wp.kernel
def forward_step_rigid_bodies(
    # Inputs
    dt: float,
    gravity: wp.array[wp.vec3],
    body_world: wp.array[wp.int32],
    body_f: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inertia: wp.array[wp.mat33],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_inertia_q: wp.array[wp.transform],
):
    """
    Forward integration step for rigid bodies in the AVBD/VBD solver.

    Args:
        dt: Time step [s].
        gravity: Gravity vector array (world frame).
        body_world: World index for each body.
        body_f: External forces on bodies (spatial wrenches, world frame).
        body_com: Centers of mass (local body frame).
        body_inertia: Inertia tensors (local body frame).
        body_inv_mass: Inverse masses (0 for kinematic bodies).
        body_inv_inertia: Inverse inertia tensors (local body frame).
        body_q: Body transforms (input: start-of-step pose, output: integrated pose).
        body_qd: Body velocities (input: start-of-step velocity, output: integrated velocity).
        body_inertia_q: Inertial target body transforms for the AVBD solve (output).
    """
    tid = wp.tid()

    q_current = body_q[tid]

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
    world_idx = body_world[tid]
    world_g = gravity[wp.max(world_idx, 0)]

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
    body_contact_buffer_pre_alloc: int,
    body_contact_counts: wp.array[wp.int32],
    body_contact_indices: wp.array[wp.int32],
    body_contact_overflow_max: wp.array[wp.int32],
):
    """
    Build per-body contact lists for body-centric per-color contact evaluation.
    Tracks overflow into body_contact_overflow_max for diagnostics.
    """
    t_id = wp.tid()
    if t_id >= rigid_contact_count[0]:
        return

    s0 = rigid_contact_shape0[t_id]
    s1 = rigid_contact_shape1[t_id]
    b0 = shape_body[s0] if s0 >= 0 else -1
    b1 = shape_body[s1] if s1 >= 0 else -1

    if b0 >= 0:
        idx = wp.atomic_add(body_contact_counts, b0, 1)
        if idx < body_contact_buffer_pre_alloc:
            body_contact_indices[b0 * body_contact_buffer_pre_alloc + idx] = t_id
        else:
            wp.atomic_max(body_contact_overflow_max, 0, idx + 1)

    if b1 >= 0:
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
    body_particle_contact_buffer_pre_alloc: int,
    body_particle_contact_counts: wp.array[wp.int32],
    body_particle_contact_indices: wp.array[wp.int32],
    body_particle_contact_overflow_max: wp.array[wp.int32],
):
    """
    Build per-body contact lists for body-particle contacts.
    Tracks overflow into body_particle_contact_overflow_max for diagnostics.
    """
    tid = wp.tid()
    if tid >= body_particle_contact_count[0]:
        return

    shape = body_particle_contact_shape[tid]
    body = shape_body[shape] if shape >= 0 else -1

    if body < 0:
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


@wp.kernel
def init_cable_rest_bend_twist(
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    body_q_rest: wp.array[wp.transform],
    joint_cable_kb_rest_local: wp.array[wp.vec3],
    joint_cable_twist_rest: wp.array[float],
):
    """Precompute cable rest angular deformation invariants."""
    j = wp.tid()
    joint_cable_kb_rest_local[j] = wp.vec3(0.0)
    joint_cable_twist_rest[j] = 0.0

    if joint_type[j] != JointType.CABLE:
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
    # Historical storage name: this vec3 now stores the full Korner/Audoly
    # rest angular deformation [bend_x, bend_y, twist_z].
    joint_cable_kb_rest_local[j] = _korner_cable_deformation_z(q_wp_rest, q_wc_rest)
    joint_cable_twist_rest[j] = 0.0


@wp.kernel
def step_joint_C0_lambda(
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_cable_kb_rest_local: wp.array[wp.vec3],
    joint_cable_twist_rest: wp.array[float],
    body_q_prev: wp.array[wp.transform],
    body_q_rest: wp.array[wp.transform],
    joint_constraint_start: wp.array[wp.int32],
    joint_constraint_dim: wp.array[wp.int32],
    joint_is_hard: wp.array[wp.int32],
    lambda_decay: float,
    penalty_decay: float,
    joint_penalty_k_min: wp.array[float],
    joint_penalty_k_max: wp.array[float],
    joint_penalty_k: wp.array[float],
    joint_C0_lin: wp.array[wp.vec3],
    joint_C0_ang: wp.array[wp.vec3],
    joint_lambda_lin: wp.array[wp.vec3],
    joint_lambda_ang: wp.array[wp.vec3],
):
    """Per-step joint AVBD maintenance: k decay + C0 snapshot + lambda decay.

    Sole owner of joint decay. Cable stretch/shear and bend/twist share linear
    and angular AL state blocks; non-cable drive/limit slots stay soft.
    """
    j = wp.tid()
    zero = wp.vec3(0.0)
    c_start = int(joint_constraint_start[j])
    c_dim = int(joint_constraint_dim[j])

    # K decay runs unconditionally (even for disabled joints).
    for s in range(c_dim):
        idx = c_start + s
        joint_penalty_k[idx] = wp.clamp(
            penalty_decay * joint_penalty_k[idx], joint_penalty_k_min[idx], joint_penalty_k_max[idx]
        )

    child = joint_child[j]
    if not joint_enabled[j] or c_dim == 0 or child < 0:
        joint_C0_lin[j] = zero
        joint_lambda_lin[j] = zero
        joint_C0_ang[j] = zero
        joint_lambda_ang[j] = zero
        return

    jt = joint_type[j]

    # Cable has four structural slots, but AL state is stored as two vec3
    # blocks: linear = stretch/shear, angular = bend/twist.
    if jt == JointType.CABLE:
        stretch_idx = c_start
        shear_idx = c_start + 1
        bend_idx = c_start + 2
        twist_idx = c_start + 3

        has_linear_hard = int(0)
        has_angular_hard = int(0)
        if joint_is_hard[stretch_idx] == 1 or joint_is_hard[shear_idx] == 1:
            has_linear_hard = 1
        if joint_is_hard[bend_idx] == 1 or joint_is_hard[twist_idx] == 1:
            has_angular_hard = 1

        if has_linear_hard == 0 and has_angular_hard == 0:
            joint_C0_lin[j] = zero
            joint_lambda_lin[j] = zero
            joint_C0_ang[j] = zero
            joint_lambda_ang[j] = zero
            return

        parent = joint_parent[j]
        if parent >= 0:
            X_wp = body_q_prev[parent] * joint_X_p[j]
        else:
            X_wp = joint_X_p[j]
        X_wc = body_q_prev[child] * joint_X_c[j]

        if has_linear_hard == 1:
            x_p = wp.transform_get_translation(X_wp)
            x_c = wp.transform_get_translation(X_wc)
            joint_C0_lin[j] = x_c - x_p
            joint_lambda_lin[j] = joint_lambda_lin[j] * lambda_decay
        else:
            joint_C0_lin[j] = zero
            joint_lambda_lin[j] = zero

        if has_angular_hard == 1:
            q_wp = wp.transform_get_rotation(X_wp)
            q_wc = wp.transform_get_rotation(X_wc)
            joint_C0_ang[j] = compute_geometric_cable_kappa_cached_z(
                q_wp,
                q_wc,
                joint_cable_kb_rest_local[j],
                joint_cable_twist_rest[j],
            )
            joint_lambda_ang[j] = joint_lambda_ang[j] * lambda_decay
        else:
            joint_C0_ang[j] = zero
            joint_lambda_ang[j] = zero
        return

    # Non-cable joints have at most two structural hard slots here: linear and
    # angular. Drive/limit slots are always soft and ignored by this snapshot.
    has_linear_hard = int(joint_is_hard[c_start])
    has_angular_hard = int(0)
    if c_dim > 1 and joint_is_hard[c_start + 1] == 1:
        has_angular_hard = 1

    if has_linear_hard == 0 and has_angular_hard == 0:
        joint_C0_lin[j] = zero
        joint_lambda_lin[j] = zero
        joint_C0_ang[j] = zero
        joint_lambda_ang[j] = zero
        return

    parent = joint_parent[j]
    if parent >= 0:
        X_wp = body_q_prev[parent] * joint_X_p[j]
    else:
        X_wp = joint_X_p[j]
    X_wc = body_q_prev[child] * joint_X_c[j]

    if has_linear_hard == 1:
        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        joint_C0_lin[j] = x_c - x_p
        joint_lambda_lin[j] = joint_lambda_lin[j] * lambda_decay
    else:
        joint_C0_lin[j] = zero
        joint_lambda_lin[j] = zero

    if has_angular_hard == 1:
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
        joint_lambda_ang[j] = joint_lambda_ang[j] * lambda_decay
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

    Averages both shapes' material.  Penalty is seeded at ``min(k_start, avg_ke)``
    when ramping (k_start >= 0) or at ``avg_ke`` when fixed-k (k_start < 0).
    """
    i = wp.tid()
    if i >= rigid_contact_count[0]:
        return

    shape_id_0 = rigid_contact_shape0[i]
    shape_id_1 = rigid_contact_shape1[i]

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

    k_floor = avg_ke if k_start < 0.0 else wp.min(k_start, avg_ke)
    contact_penalty_k[i] = k_floor


@wp.kernel
def init_body_body_contacts_avbd(
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
    hard_contacts: int,
    # Pipeline-owned correspondence and VBD-owned cross-step state
    match_index: wp.array[wp.int32],
    history: RigidContactHistory,
    # Scalar parameters
    k_start: float,
    # In/out: replayed only for matched hard contacts that were sticking.
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_offset0: wp.array[wp.vec3],
    rigid_contact_offset1: wp.array[wp.vec3],
    # Outputs
    contact_penalty_k: wp.array[float],
    contact_lambda: wp.array[wp.vec3],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    contact_material_ke: wp.array[float],
):
    """Restore body-body contact state from match indices.

    For hard contacts: restores lambda (rotated from old to new contact frame),
    penalty_k, and stick-anchor points when the previous matched contact stuck.
    For soft contacts: restores penalty_k only; lambda stays zero because the
    soft path is penalty-only.
    Sticky hard contacts may overwrite rigid_contact_point0/1 and
    rigid_contact_offset0/1 in place with the previously saved contact anchors.
    C0 and decay are handled by step_body_body_contact_C0_lambda.

    match_index[i] addresses saved contact rows from the last snapshot.
    Negative values (-1 unmatched, -2 broken) cold-start identically.
    """
    i = wp.tid()
    if i >= rigid_contact_count[0]:
        return

    s0 = rigid_contact_shape0[i]
    s1 = rigid_contact_shape1[i]

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

    k_floor = avg_ke if k_start < 0.0 else wp.min(k_start, avg_ke)
    slot = match_index[i]

    if slot >= 0:
        contact_penalty_k[i] = wp.clamp(history.penalty_k[slot], k_floor, avg_ke)
        if hard_contacts == 1:
            lam_hist = history.lambda_[slot]
            n_new = rigid_contact_normal[i]
            n_old = history.normal[slot]
            lam_n = wp.dot(lam_hist, n_old)
            lam_t_old = lam_hist - n_old * lam_n
            lam_t_new = lam_t_old - n_new * wp.dot(lam_t_old, n_new)
            contact_lambda[i] = n_new * lam_n + lam_t_new

            stick_flag = history.stick_flag[slot]
            # Replay saved points and offsets only for contacts whose saved
            # state was sticking. Point and offset must move together; the
            # surface anchor is ``point + offset``.
            if stick_flag == _STICK_FLAG_ANCHOR or stick_flag == _STICK_FLAG_DEADZONE:
                rigid_contact_point0[i] = history.point0[slot]
                rigid_contact_point1[i] = history.point1[slot]
                rigid_contact_offset0[i] = history.offset0[slot]
                rigid_contact_offset1[i] = history.offset1[slot]
        else:
            contact_lambda[i] = wp.vec3(0.0)
    else:
        contact_penalty_k[i] = k_floor
        contact_lambda[i] = wp.vec3(0.0)


@wp.kernel
def snapshot_body_body_contact_history(
    rigid_contact_count: wp.array[int],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_offset0: wp.array[wp.vec3],
    rigid_contact_offset1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    contact_lambda: wp.array[wp.vec3],
    contact_stick_flag: wp.array[wp.int32],
    contact_penalty_k: wp.array[float],
    # Outputs, same order as RigidContactHistory
    prev_lambda: wp.array[wp.vec3],
    prev_stick_flag: wp.array[wp.int32],
    prev_penalty_k: wp.array[float],
    prev_point0: wp.array[wp.vec3],
    prev_point1: wp.array[wp.vec3],
    prev_offset0: wp.array[wp.vec3],
    prev_offset1: wp.array[wp.vec3],
    prev_normal: wp.array[wp.vec3],
):
    """Snapshot converged contact state by contact row.

    The next match_index refers to the rows written here, so VBD history is
    stored directly by contact row index.
    """
    i = wp.tid()
    if i >= rigid_contact_count[0]:
        return

    prev_lambda[i] = contact_lambda[i]
    prev_stick_flag[i] = contact_stick_flag[i]
    prev_penalty_k[i] = contact_penalty_k[i]
    prev_point0[i] = rigid_contact_point0[i]
    prev_point1[i] = rigid_contact_point1[i]
    prev_offset0[i] = rigid_contact_offset0[i]
    prev_offset1[i] = rigid_contact_offset1[i]
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
    body_q: wp.array[wp.transform],
    hard_contacts: int,
    lambda_decay: float,
    penalty_decay: float,
    contact_material_ke: wp.array[float],
    k_start: float,
    # In/out
    contact_penalty_k: wp.array[float],
    contact_C0: wp.array[wp.vec3],
    contact_lambda: wp.array[wp.vec3],
):
    """Per-step k decay + lambda decay + C0 snapshot.

    Runs every step. K decay is unconditional (hard and soft). Lambda decay
    uses lambda_decay when retaining hard-contact lambda across steps or reused
    contact rows. C0 is always recomputed for hard contacts.
    """
    i = wp.tid()
    if i >= rigid_contact_count[0]:
        return

    ke = contact_material_ke[i]
    k_min = ke if k_start < 0.0 else wp.min(k_start, ke)
    contact_penalty_k[i] = wp.clamp(penalty_decay * contact_penalty_k[i], k_min, ke)

    contact_lambda[i] = contact_lambda[i] * lambda_decay

    if hard_contacts == 1:
        s0 = rigid_contact_shape0[i]
        s1 = rigid_contact_shape1[i]
        b0 = shape_body[s0] if s0 >= 0 else -1
        b1 = shape_body[s1] if s1 >= 0 else -1
        p0 = rigid_contact_point0[i]
        p1 = rigid_contact_point1[i]
        anchor0_local = p0 + rigid_contact_offset0[i]
        anchor1_local = p1 + rigid_contact_offset1[i]
        n = rigid_contact_normal[i]
        # Normal: thickness already accounts for the radial extent, so use
        # the unprojected skeleton points (matches update_duals_body_body_contacts).
        cp0 = wp.transform_point(body_q[b0], p0) if b0 >= 0 else p0
        cp1 = wp.transform_point(body_q[b1], p1) if b1 >= 0 else p1
        C0_n = -contact_surface_separation(cp0, cp1, n, rigid_contact_margin0[i], rigid_contact_margin1[i])
        # Tangential: use surface anchors so spin about a body's symmetry axis
        # registers in the frozen tangential offset, matching tangential_disp
        # in update_duals_body_body_contacts.
        a0 = wp.transform_point(body_q[b0], anchor0_local) if b0 >= 0 else anchor0_local
        a1 = wp.transform_point(body_q[b1], anchor1_local) if b1 >= 0 else anchor1_local
        d_surf = a1 - a0
        C0_t = -(d_surf - n * wp.dot(n, d_surf))
        contact_C0[i] = n * C0_n + C0_t


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
    rigid shape's material.  Penalty is seeded at ``min(k_start, avg_ke)`` when
    ramping (k_start >= 0) or at ``avg_ke`` when fixed-k (k_start < 0).
    """
    i = wp.tid()
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

    k_floor = avg_ke if k_start < 0.0 else wp.min(k_start, avg_ke)
    body_particle_contact_penalty_k[i] = k_floor


@wp.func
def _cable_dahl_active_stiffness(
    c_start: int,
    joint_penalty_k: wp.array[float],
    joint_is_hard: wp.array[wp.int32],
) -> wp.vec3:
    """Current stiffness for soft cable bend/twist modes; hard modes return zero."""
    bend_idx = c_start + 2
    twist_idx = bend_idx + 1

    k_bend = float(0.0)
    k_bend_active = joint_penalty_k[bend_idx]
    if joint_is_hard[bend_idx] == 0 and k_bend_active > 0.0:
        k_bend = k_bend_active

    k_twist = float(0.0)
    k_twist_active = joint_penalty_k[twist_idx]
    if joint_is_hard[twist_idx] == 0 and k_twist_active > 0.0:
        k_twist = k_twist_active

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
    kappa: float,
    kappa_prev: float,
    d_kappa_prev: float,
    sigma_prev: float,
    sigma_max: float,
    tau: float,
):
    """Advance one scalar Dahl component and return (sigma, delta_kappa, direction)."""
    d_kappa = kappa - kappa_prev
    direction = _dahl_axis_direction(d_kappa, d_kappa_prev)
    exp_term = wp.exp(-direction * d_kappa / tau)
    sigma = direction * sigma_max * (1.0 - exp_term) + sigma_prev * exp_term
    return sigma, d_kappa, direction


@wp.kernel
def compute_cable_dahl_parameters(
    # Inputs
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_constraint_start: wp.array[int],
    joint_penalty_k: wp.array[float],
    joint_is_hard: wp.array[wp.int32],
    joint_cable_kb_rest_local: wp.array[wp.vec3],
    joint_cable_twist_rest: wp.array[float],
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
    Compute shared cable Dahl hysteresis parameters (sigma0, C_fric) from
    the current bend/twist strain and the stored previous Dahl state.

    The outputs are:
      - sigma0: linearized friction stress at the start of the step (per component)
      - C_fric: tangent stiffness d(sigma)/d(kappa) (per component)

    Dahl eps_max/tau remain per-joint scalars for compatibility with main's
    custom attributes. Bend and twist still get separate envelopes through live
    active stiffness. Hard or inactive subspaces produce zero Dahl stress and
    tangent stiffness.
    """
    j = wp.tid()
    zero = wp.vec3(0.0)

    # Default to no friction; the success path overwrites both outputs below.
    joint_sigma_start[j] = zero
    joint_C_fric[j] = zero

    if not joint_enabled[j] or joint_type[j] != JointType.CABLE:
        return

    parent = joint_parent[j]
    child = joint_child[j]
    # World-parent joints are valid; child body must exist.
    if child < 0:
        return

    eps_max = joint_eps_max[j]
    tau = joint_tau[j]
    if tau <= 0.0 or eps_max <= 0.0:
        return

    c_start = joint_constraint_start[j]
    k_dahl = _cable_dahl_active_stiffness(c_start, joint_penalty_k, joint_is_hard)
    if k_dahl[0] <= 0.0 and k_dahl[1] <= 0.0 and k_dahl[2] <= 0.0:
        return

    # Compute joint frames in world space and the current bend/twist strain.
    if parent >= 0:
        X_wp = body_q[parent] * joint_X_p[j]
    else:
        X_wp = joint_X_p[j]
    X_wc = body_q[child] * joint_X_c[j]
    q_wp = wp.transform_get_rotation(X_wp)
    q_wc = wp.transform_get_rotation(X_wc)
    kappa_now = compute_geometric_cable_kappa_cached_z(
        q_wp,
        q_wc,
        joint_cable_kb_rest_local[j],
        joint_cable_twist_rest[j],
    )

    # Previous Dahl state (from last converged timestep).
    kappa_prev = joint_kappa_prev[j]
    d_kappa_prev = joint_dkappa_prev[j]
    sigma_prev = joint_sigma_prev[j]

    sigma_out = zero
    C_fric_out = zero
    for axis in range(3):
        sigma_max = k_dahl[axis] * eps_max
        if sigma_max <= 0.0:
            continue

        sigma0_i, d_kappa_i, direction = _advance_dahl_axis(
            kappa_now[axis],
            kappa_prev[axis],
            d_kappa_prev[axis],
            sigma_prev[axis],
            sigma_max,
            tau,
        )
        sigma0_i = wp.clamp(sigma0_i, -sigma_max, sigma_max)

        # Tangent stiffness K = (sigma_max - dir*sigma0) / (tau + |d_kappa|).
        numerator = sigma_max - direction * sigma0_i
        denominator = tau + wp.abs(d_kappa_i)
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
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    contact_lambda: wp.array[wp.vec3],
    contact_C0: wp.array[wp.vec3],
    avbd_alpha: float,
    hard_contacts: int,
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
    Per-body augmented-Lagrangian contact accumulation with _NUM_CONTACT_THREADS_PER_BODY strided threads.
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
        k = contact_penalty_k[contact_idx]
        friction_c0 = wp.vec3(0.0)

        if hard_contacts == 1:
            lam_vec = contact_lambda[contact_idx]
            lam_n = wp.dot(lam_vec, contact_normal)
            C0_vec = contact_C0[contact_idx]
            C0_n = wp.dot(contact_normal, C0_vec)
            # Hard-contact stabilization: normal uses C_n - alpha*C0_n; tangent caches
            # (1 - alpha)*C0_t for the later tangential update.
            C_eff = C_n - avbd_alpha * C0_n
            friction_c0 = (1.0 - avbd_alpha) * (C0_vec - contact_normal * C0_n)

        if C_n <= _SMALL_LENGTH_EPS and lam_n <= 0.0:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        f_n_check = k * C_eff + lam_n
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
            k,
            k,
            contact_kd,
            lam_vec,
            contact_mu,
            friction_epsilon,
            hard_contacts,
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
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    contact_lambda: wp.array[wp.vec3],
    contact_C0: wp.array[wp.vec3],
    avbd_alpha: float,
    hard_contacts: int,
    friction_epsilon: float,
    # Outputs (length = rigid_contact_max)
    out_body0: wp.array[wp.int32],
    out_body1: wp.array[wp.int32],
    out_point0_world: wp.array[wp.vec3],
    out_point1_world: wp.array[wp.vec3],
    out_force_on_body1: wp.array[wp.vec3],
):
    """Compute per-contact forces in world space (hard: ALM, soft: penalty)."""
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
    k = contact_penalty_k[contact_idx]
    friction_c0 = wp.vec3(0.0)

    if hard_contacts == 1:
        lam_vec = contact_lambda[contact_idx]
        lam_n = wp.dot(lam_vec, contact_normal)
        C0_vec = contact_C0[contact_idx]
        C0_n = wp.dot(contact_normal, C0_vec)
        # Hard-contact stabilization: normal uses C_n - alpha*C0_n; tangent caches
        # (1 - alpha)*C0_t for the later tangential update.
        C_eff = C_n - avbd_alpha * C0_n
        friction_c0 = (1.0 - avbd_alpha) * (C0_vec - contact_normal * C0_n)

    f_n_check = k * C_eff + lam_n
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
        k,
        k,
        contact_kd,
        lam_vec,
        contact_mu,
        friction_epsilon,
        hard_contacts,
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
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    # AVBD body-particle soft contact penalties and material properties
    friction_epsilon: float,
    body_particle_contact_penalty_k: wp.array[float],
    body_particle_contact_material_kd: wp.array[float],
    body_particle_contact_material_mu: wp.array[float],
    # Soft contact data (body-particle)
    body_particle_contact_count: wp.array[int],
    body_particle_contact_particle: wp.array[int],
    body_particle_contact_body_pos: wp.array[wp.vec3],
    body_particle_contact_body_vel: wp.array[wp.vec3],
    body_particle_contact_normal: wp.array[wp.vec3],
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
    Per-body accumulation of body-particle (particle-rigid) soft contact forces and
    Hessians on rigid bodies.

    This kernel resolves contact geometry and relative displacement inline, then
    calls ``_compute_body_particle_contact_force`` for the pure force law.
    Body surface velocity uses the displacement-based path (body_q_prev).

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

    max_contacts = body_particle_contact_count[0]

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
        if contact_idx >= max_contacts:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        particle_idx = body_particle_contact_particle[contact_idx]
        if particle_idx < 0:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        particle_pos = particle_q[particle_idx]
        cp_local = body_particle_contact_body_pos[contact_idx]
        cp_world = wp.transform_point(X_wb, cp_local)
        n = body_particle_contact_normal[contact_idx]
        radius = particle_radius[particle_idx]
        penetration_depth = -(wp.dot(n, particle_pos - cp_world) - radius)

        if penetration_depth <= 0.0:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        bx_prev = wp.transform_point(X_wb_prev, cp_local)
        bv = (cp_world - bx_prev) / dt + wp.transform_vector(X_wb, body_particle_contact_body_vel[contact_idx])
        dx = particle_pos - particle_q_prev[particle_idx]
        relative_translation = dx - bv * dt

        force_on_particle, hessian_particle = _compute_body_particle_contact_force(
            penetration_depth,
            n,
            relative_translation,
            body_particle_contact_penalty_k[contact_idx],
            body_particle_contact_material_kd[contact_idx],
            body_particle_contact_material_mu[contact_idx],
            friction_epsilon,
            dt,
        )

        f_body = -force_on_particle

        r = cp_world - com_world
        tau_body = wp.cross(r, f_body)

        r_skew = wp.skew(r)
        r_skew_T_K = wp.transpose(r_skew) * hessian_particle

        force_acc += f_body
        torque_acc += tau_body
        h_ll_acc += hessian_particle
        h_al_acc += -r_skew_T_K
        h_aa_acc += r_skew_T_K * r_skew

        i += _NUM_CONTACT_THREADS_PER_BODY

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
    joint_cable_kb_rest_local: wp.array[wp.vec3],
    joint_cable_twist_rest: wp.array[float],
    joint_qd_start: wp.array[int],
    joint_target_q_start: wp.array[int],
    joint_constraint_start: wp.array[int],
    # AVBD per-constraint penalty state (scalar constraints indexed via joint_constraint_start)
    joint_penalty_k: wp.array[float],
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
    joint_lambda_lin: wp.array[wp.vec3],
    joint_lambda_ang: wp.array[wp.vec3],
    joint_C0_lin: wp.array[wp.vec3],
    joint_C0_ang: wp.array[wp.vec3],
    joint_is_hard: wp.array[wp.int32],
    avbd_alpha: float,
    joint_dof_dim: wp.array2d[int],
    joint_rest_angle: wp.array[float],
    external_forces: wp.array[wp.vec3],
    external_torques: wp.array[wp.vec3],
    external_hessian_ll: wp.array[wp.mat33],  # Linear-linear block from rigid contacts
    external_hessian_al: wp.array[wp.mat33],  # Angular-linear coupling block from rigid contacts
    external_hessian_aa: wp.array[wp.mat33],  # Angular-angular block from rigid contacts
    # Output
    body_q_new: wp.array[wp.transform],
):
    """
    AVBD solve step for rigid bodies.

    Assembles inertial, joint, and collision contributions into a 6x6 SPD
    block system and solves via direct LDL^T.

    Algorithm:
      1. Compute inertial forces/Hessians
      2. Accumulate external forces/Hessians from rigid contacts
      3. Accumulate joint forces/Hessians from adjacent joints
      4. Solve 6x6 system via LDL^T
      5. Update pose: rotation from angular increment, position from linear increment

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
        joint_penalty_k: AVBD per-constraint penalty stiffness (one scalar per solver constraint component).
        joint_sigma_start: Dahl hysteresis state at start of step.
        joint_C_fric: Dahl friction configuration per joint.
        external_forces: External linear forces from rigid contacts.
        external_torques: External angular torques from rigid contacts.
        external_hessian_ll: Linear-linear Hessian block (3x3) from rigid contacts.
        external_hessian_al: Angular-linear coupling Hessian block (3x3) from rigid contacts.
        external_hessian_aa: Angular-angular Hessian block (3x3) from rigid contacts.
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
            joint_cable_kb_rest_local,
            joint_cable_twist_rest,
            joint_qd_start,
            joint_target_q_start,
            joint_constraint_start,
            joint_penalty_k,
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
            joint_lambda_lin,
            joint_lambda_ang,
            joint_C0_lin,
            joint_C0_ang,
            joint_is_hard,
            avbd_alpha,
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
    joint_cable_kb_rest_local: wp.array[wp.vec3],
    joint_cable_twist_rest: wp.array[float],
    joint_qd_start: wp.array[int],
    joint_target_q_start: wp.array[int],
    joint_constraint_start: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_rest: wp.array[wp.transform],
    joint_dof_dim: wp.array2d[int],
    joint_C0_lin: wp.array[wp.vec3],
    joint_C0_ang: wp.array[wp.vec3],
    joint_is_hard: wp.array[wp.int32],
    avbd_alpha: float,
    joint_penalty_k_max: wp.array[float],
    beta_lin: float,
    beta_ang: float,
    joint_target_ke: wp.array[float],
    joint_target_q: wp.array[float],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_limit_ke: wp.array[float],
    joint_rest_angle: wp.array[float],
    # Input/output
    joint_penalty_k: wp.array[float],
    joint_lambda_lin: wp.array[wp.vec3],
    joint_lambda_ang: wp.array[wp.vec3],
):
    """
    Update augmented-Lagrangian duals for joint constraints (per-iteration).

    Hard structural slots update lambda via ALM; all structural slots ramp k.
    Drive/limit slots ramp k only (no lambda);
    k is capped at ``joint_penalty_k_max`` while the force kernel applies
    the mode-specific stiffness cap (``min(avbd_ke, model_ke)``).
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
        jt != JointType.CABLE
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
    else:
        X_wp = joint_X_p[j]
    X_wc = body_q[child] * joint_X_c[j]

    # CABLE joint: fixed stretch/shear/bend/twist slots.
    if jt == JointType.CABLE:
        q_wp = wp.transform_get_rotation(X_wp)
        q_wc = wp.transform_get_rotation(X_wc)

        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        C_vec = x_c - x_p

        kappa = compute_geometric_cable_kappa_cached_z(
            q_wp,
            q_wc,
            joint_cable_kb_rest_local[j],
            joint_cable_twist_rest[j],
        )

        # Linear penalty update: axial stretch / transverse shear.
        stretch_idx = c_start
        shear_idx = c_start + 1
        t_world = _quat_rotate_local_z(q_wp)
        lambda_lin = joint_lambda_lin[j]
        C0_lin = joint_C0_lin[j]

        C_stretch = _project_along_unit_axis(C_vec, t_world)
        C0_stretch = _project_along_unit_axis(C0_lin, t_world)
        lambda_stretch = _project_along_unit_axis(lambda_lin, t_world)
        lam_stretch = _update_dual_vec3(
            C_stretch,
            C0_stretch,
            avbd_alpha,
            joint_penalty_k[stretch_idx],
            lambda_stretch,
            joint_is_hard[stretch_idx],
        )
        joint_penalty_k[stretch_idx] = wp.min(
            joint_penalty_k_max[stretch_idx], joint_penalty_k[stretch_idx] + beta_lin * wp.length(C_stretch)
        )

        C_shear = C_vec - C_stretch
        lam_shear = _update_dual_vec3(
            C_shear,
            C0_lin - C0_stretch,
            avbd_alpha,
            joint_penalty_k[shear_idx],
            lambda_lin - lambda_stretch,
            joint_is_hard[shear_idx],
        )
        joint_lambda_lin[j] = lam_stretch + lam_shear
        joint_penalty_k[shear_idx] = wp.min(
            joint_penalty_k_max[shear_idx], joint_penalty_k[shear_idx] + beta_lin * wp.length(C_shear)
        )

        # Bend penalty update (first angular constraint slot)
        bend_idx = c_start + 2
        lambda_ang = joint_lambda_ang[j]
        C0_ang = joint_C0_ang[j]
        kappa_bend = wp.vec3(kappa[0], kappa[1], 0.0)
        lam_bend = _update_dual_vec3(
            kappa_bend,
            wp.vec3(C0_ang[0], C0_ang[1], 0.0),
            avbd_alpha,
            joint_penalty_k[bend_idx],
            wp.vec3(lambda_ang[0], lambda_ang[1], 0.0),
            joint_is_hard[bend_idx],
        )
        joint_penalty_k[bend_idx] = wp.min(
            joint_penalty_k_max[bend_idx], joint_penalty_k[bend_idx] + beta_ang * wp.length(kappa_bend)
        )

        twist_idx = bend_idx + 1
        kappa_twist = wp.vec3(0.0, 0.0, kappa[2])
        lam_twist = _update_dual_vec3(
            kappa_twist,
            wp.vec3(0.0, 0.0, C0_ang[2]),
            avbd_alpha,
            joint_penalty_k[twist_idx],
            wp.vec3(0.0, 0.0, lambda_ang[2]),
            joint_is_hard[twist_idx],
        )
        joint_lambda_ang[j] = lam_bend + lam_twist
        joint_penalty_k[twist_idx] = wp.min(
            joint_penalty_k_max[twist_idx], joint_penalty_k[twist_idx] + beta_ang * wp.length(kappa_twist)
        )
        return

    if parent >= 0:
        X_wp_rest = body_q_rest[parent] * joint_X_p[j]
    else:
        X_wp_rest = joint_X_p[j]
    X_wc_rest = body_q_rest[child] * joint_X_c[j]

    # BALL joint: update isotropic linear anchor-coincidence penalty (single scalar).
    if jt == JointType.BALL:
        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        C_vec = x_c - x_p

        i0 = c_start
        lam_new = _update_dual_vec3(
            C_vec,
            joint_C0_lin[j],
            avbd_alpha,
            joint_penalty_k[i0],
            joint_lambda_lin[j],
            joint_is_hard[i0],
        )
        joint_lambda_lin[j] = lam_new
        joint_penalty_k[i0] = wp.min(joint_penalty_k_max[i0], joint_penalty_k[i0] + beta_lin * wp.length(C_vec))
        return

    # FIXED joint: update isotropic linear + isotropic angular penalties (2 scalars).
    if jt == JointType.FIXED:
        i_lin = c_start + 0
        i_ang = c_start + 1

        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        C_vec_lin = x_c - x_p
        lam_new = _update_dual_vec3(
            C_vec_lin,
            joint_C0_lin[j],
            avbd_alpha,
            joint_penalty_k[i_lin],
            joint_lambda_lin[j],
            joint_is_hard[i_lin],
        )
        joint_lambda_lin[j] = lam_new
        joint_penalty_k[i_lin] = wp.min(
            joint_penalty_k_max[i_lin], joint_penalty_k[i_lin] + beta_lin * wp.length(C_vec_lin)
        )

        q_wp = wp.transform_get_rotation(X_wp)
        q_wc = wp.transform_get_rotation(X_wc)
        q_wp_rest = wp.transform_get_rotation(X_wp_rest)
        q_wc_rest = wp.transform_get_rotation(X_wc_rest)
        kappa = compute_kappa(q_wp, q_wc, q_wp_rest, q_wc_rest)
        lam_new = _update_dual_vec3(
            kappa,
            joint_C0_ang[j],
            avbd_alpha,
            joint_penalty_k[i_ang],
            joint_lambda_ang[j],
            joint_is_hard[i_ang],
        )
        joint_lambda_ang[j] = lam_new
        joint_penalty_k[i_ang] = wp.min(
            joint_penalty_k_max[i_ang], joint_penalty_k[i_ang] + beta_ang * wp.length(kappa)
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
        lam_new = _update_dual_vec3(
            C_vec_lin,
            P_lin * joint_C0_lin[j],
            avbd_alpha,
            joint_penalty_k[i_lin],
            joint_lambda_lin[j],
            joint_is_hard[i_lin],
        )
        joint_lambda_lin[j] = lam_new
        joint_penalty_k[i_lin] = wp.min(
            joint_penalty_k_max[i_lin], joint_penalty_k[i_lin] + beta_lin * wp.length(C_vec_lin)
        )

        q_wc = wp.transform_get_rotation(X_wc)
        q_wp_rest = wp.transform_get_rotation(X_wp_rest)
        q_wc_rest = wp.transform_get_rotation(X_wc_rest)
        kappa = compute_kappa(q_wp, q_wc, q_wp_rest, q_wc_rest)
        kappa_perp = P_ang * kappa
        lam_old = P_ang * joint_lambda_ang[j]
        lam_new = _update_dual_vec3(
            kappa_perp,
            P_ang * joint_C0_ang[j],
            avbd_alpha,
            joint_penalty_k[i_ang],
            lam_old,
            joint_is_hard[i_ang],
        )
        joint_lambda_ang[j] = lam_new
        joint_penalty_k[i_ang] = wp.min(
            joint_penalty_k_max[i_ang], joint_penalty_k[i_ang] + beta_ang * wp.length(kappa_perp)
        )

        # Drive/limit dual update for free angular DOF (slot c_start + 2)
        dof_idx = qd_start
        model_drive_ke = joint_target_ke[dof_idx]
        model_limit_ke = joint_limit_ke[dof_idx]
        lim_lower = joint_limit_lower[dof_idx]
        lim_upper = joint_limit_upper[dof_idx]
        has_drive = model_drive_ke > 0.0
        has_limits = model_limit_ke > 0.0 and (lim_lower > -MAXVAL or lim_upper < MAXVAL)

        if has_drive or has_limits:
            a = wp.normalize(joint_axis[qd_start])
            theta = wp.dot(kappa, a)
            theta_abs = theta + joint_rest_angle[dof_idx]
            target_pos = joint_target_q[joint_target_q_start[j]]
            _mode, err_pos = resolve_drive_limit_mode(
                theta_abs, target_pos, lim_lower, lim_upper, has_drive, has_limits
            )
            i_dl = c_start + 2
            C_dl = wp.abs(err_pos)
            joint_penalty_k[i_dl] = wp.min(joint_penalty_k_max[i_dl], joint_penalty_k[i_dl] + beta_ang * C_dl)
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
        lam_new = _update_dual_vec3(
            C_vec_perp,
            P_lin * joint_C0_lin[j],
            avbd_alpha,
            joint_penalty_k[i_lin],
            lam_old,
            joint_is_hard[i_lin],
        )
        joint_lambda_lin[j] = lam_new
        joint_penalty_k[i_lin] = wp.min(
            joint_penalty_k_max[i_lin], joint_penalty_k[i_lin] + beta_lin * wp.length(C_vec_perp)
        )

        q_wc = wp.transform_get_rotation(X_wc)
        q_wp_rest = wp.transform_get_rotation(X_wp_rest)
        q_wc_rest = wp.transform_get_rotation(X_wc_rest)
        kappa = compute_kappa(q_wp, q_wc, q_wp_rest, q_wc_rest)
        kappa_perp = P_ang * kappa
        lam_new = _update_dual_vec3(
            kappa_perp,
            P_ang * joint_C0_ang[j],
            avbd_alpha,
            joint_penalty_k[i_ang],
            joint_lambda_ang[j],
            joint_is_hard[i_ang],
        )
        joint_lambda_ang[j] = lam_new
        joint_penalty_k[i_ang] = wp.min(
            joint_penalty_k_max[i_ang], joint_penalty_k[i_ang] + beta_ang * wp.length(kappa_perp)
        )

        # Drive/limit dual update for free linear DOF (slot c_start + 2)
        dof_idx = qd_start
        model_drive_ke = joint_target_ke[dof_idx]
        model_limit_ke = joint_limit_ke[dof_idx]
        lim_lower = joint_limit_lower[dof_idx]
        lim_upper = joint_limit_upper[dof_idx]
        has_drive = model_drive_ke > 0.0
        has_limits = model_limit_ke > 0.0 and (lim_lower > -MAXVAL or lim_upper < MAXVAL)

        if has_drive or has_limits:
            axis_local = joint_axis[qd_start]
            axis_w_dl = wp.normalize(wp.quat_rotate(q_wp, axis_local))
            d_along = wp.dot(C_vec, axis_w_dl)
            target_pos = joint_target_q[joint_target_q_start[j]]
            _mode, err_pos = resolve_drive_limit_mode(d_along, target_pos, lim_lower, lim_upper, has_drive, has_limits)
            i_dl = c_start + 2
            C_dl = wp.abs(err_pos)
            joint_penalty_k[i_dl] = wp.min(joint_penalty_k_max[i_dl], joint_penalty_k[i_dl] + beta_lin * C_dl)
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
        if lin_count < 3:
            C_vec_perp = P_lin * C_vec
            # P_lin rotates with the parent; re-project stored lambda into the current
            # constrained subspace before accumulating.
            lam_old = P_lin * joint_lambda_lin[j]
            lam_new = _update_dual_vec3(
                C_vec_perp,
                P_lin * joint_C0_lin[j],
                avbd_alpha,
                joint_penalty_k[i_lin],
                lam_old,
                joint_is_hard[i_lin],
            )
            joint_lambda_lin[j] = lam_new
            joint_penalty_k[i_lin] = wp.min(
                joint_penalty_k_max[i_lin], joint_penalty_k[i_lin] + beta_lin * wp.length(C_vec_perp)
            )

        q_wc = wp.transform_get_rotation(X_wc)
        q_wp_rest = wp.transform_get_rotation(X_wp_rest)
        q_wc_rest = wp.transform_get_rotation(X_wc_rest)
        kappa = compute_kappa(q_wp_rot, q_wc, q_wp_rest, q_wc_rest)
        if ang_count < 3:
            kappa_perp = P_ang * kappa
            lam_old = P_ang * joint_lambda_ang[j]
            lam_new = _update_dual_vec3(
                kappa_perp,
                P_ang * joint_C0_ang[j],
                avbd_alpha,
                joint_penalty_k[i_ang],
                lam_old,
                joint_is_hard[i_ang],
            )
            joint_lambda_ang[j] = lam_new
            joint_penalty_k[i_ang] = wp.min(
                joint_penalty_k_max[i_ang], joint_penalty_k[i_ang] + beta_ang * wp.length(kappa_perp)
            )

        # Drive/limit dual update for D6 free DOFs
        target_q_base = joint_target_q_start[j]
        for li in range(3):
            if li < lin_count:
                dof_idx = qd_start + li
                target_q_idx = target_q_base + li
                model_drive_ke = joint_target_ke[dof_idx]
                model_limit_ke = joint_limit_ke[dof_idx]
                lim_lower = joint_limit_lower[dof_idx]
                lim_upper = joint_limit_upper[dof_idx]
                has_drive = model_drive_ke > 0.0
                has_limits = model_limit_ke > 0.0 and (lim_lower > -MAXVAL or lim_upper < MAXVAL)

                if has_drive or has_limits:
                    axis_w_dl = wp.normalize(wp.quat_rotate(q_wp_rot, joint_axis[dof_idx]))
                    d_along = wp.dot(C_vec, axis_w_dl)
                    target_pos_dl = joint_target_q[target_q_idx]
                    _mode, err_pos = resolve_drive_limit_mode(
                        d_along, target_pos_dl, lim_lower, lim_upper, has_drive, has_limits
                    )
                    i_dl = c_start + 2 + li
                    C_dl = wp.abs(err_pos)
                    joint_penalty_k[i_dl] = wp.min(joint_penalty_k_max[i_dl], joint_penalty_k[i_dl] + beta_lin * C_dl)

        for ai in range(3):
            if ai < ang_count:
                dof_idx = qd_start + lin_count + ai
                target_q_idx = target_q_base + lin_count + ai
                model_drive_ke = joint_target_ke[dof_idx]
                model_limit_ke = joint_limit_ke[dof_idx]
                lim_lower = joint_limit_lower[dof_idx]
                lim_upper = joint_limit_upper[dof_idx]
                has_drive = model_drive_ke > 0.0
                has_limits = model_limit_ke > 0.0 and (lim_lower > -MAXVAL or lim_upper < MAXVAL)

                if has_drive or has_limits:
                    a_dl = wp.normalize(joint_axis[dof_idx])
                    theta = wp.dot(kappa, a_dl)
                    theta_abs = theta + joint_rest_angle[dof_idx]
                    target_pos_dl = joint_target_q[target_q_idx]
                    _mode, err_pos = resolve_drive_limit_mode(
                        theta_abs, target_pos_dl, lim_lower, lim_upper, has_drive, has_limits
                    )
                    i_dl = c_start + 2 + lin_count + ai
                    C_dl = wp.abs(err_pos)
                    joint_penalty_k[i_dl] = wp.min(joint_penalty_k_max[i_dl], joint_penalty_k[i_dl] + beta_ang * C_dl)
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
    avbd_alpha: float,
    stick_motion_eps: float,
    hard_contacts: int,
    body_inv_mass: wp.array[float],
    contact_material_ke: wp.array[float],
    beta: float,
    # Input/output
    contact_penalty_k: wp.array[float],
    contact_lambda: wp.array[wp.vec3],
    # Output
    contact_stick_flag: wp.array[wp.int32],
):
    """
    Update AVBD augmented-Lagrangian duals for contact constraints (per-iteration).
    Hard mode: scalar isotropic k with vec3 lambda. Normal uses C_stab_n, tangential
    uses displacement (body_q_prev -> body_q) for kinematic friction support.
    Coulomb cone clamping on tangential lambda. Soft mode: no lambda update.
    K ramp runs unconditionally for both hard and soft contacts.
    """
    idx = wp.tid()
    if idx >= rigid_contact_count[0]:
        return

    shape_id_0 = rigid_contact_shape0[idx]
    shape_id_1 = rigid_contact_shape1[idx]
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

    if hard_contacts == 1:
        k = contact_penalty_k[idx]
        C0_vec = contact_C0[idx]
        lam_vec = contact_lambda[idx]
        mu = contact_material_mu[idx]

        C_n_raw = -contact_surface_separation(
            p0_world, p1_world, n, rigid_contact_margin0[idx], rigid_contact_margin1[idx]
        )
        C0_n = wp.dot(n, C0_vec)
        C_stab_n = C_n_raw - avbd_alpha * C0_n

        # Release lambda_n at full rate on separation (bypass C0 stabilization).
        if C_n_raw < 0.0:
            C_stab_n = C_n_raw

        lam_n_old = wp.dot(lam_vec, n)
        lam_n_new = wp.max(lam_n_old + k * C_stab_n, 0.0)

        rel_disp = (a0_world - a0_prev) - (a1_world - a1_prev)
        tangential_disp = rel_disp - n * wp.dot(n, rel_disp)
        C0_t_vec = C0_vec - n * C0_n
        lam_t_old = lam_vec - n * lam_n_old
        tangent_residual = tangential_disp + (1.0 - avbd_alpha) * C0_t_vec
        lam_t_new = lam_t_old + k * tangent_residual
        lam_t_len = wp.length(lam_t_new)
        cone_limit = mu * lam_n_new
        if lam_t_len > cone_limit and lam_t_len > 0.0:
            lam_t_new = lam_t_new * (cone_limit / lam_t_len)
        contact_lambda[idx] = n * lam_n_new + lam_t_new

        has_kinematic = int(0)
        if body_id_0 < 0 or body_id_1 < 0:
            has_kinematic = int(1)
        elif body_id_0 >= 0 and body_inv_mass[body_id_0] == 0.0:
            has_kinematic = int(1)
        elif body_id_1 >= 0 and body_inv_mass[body_id_1] == 0.0:
            has_kinematic = int(1)

        flag = int(0)
        if lam_n_new > 0.0 and lam_t_len <= cone_limit and wp.length(tangent_residual) < stick_motion_eps:
            if has_kinematic == 1:
                flag = _STICK_FLAG_ANCHOR
            else:
                flag = _STICK_FLAG_DEADZONE
        contact_stick_flag[idx] = flag
    else:
        contact_stick_flag[idx] = int(0)

    C_n = -contact_surface_separation(p0_world, p1_world, n, rigid_contact_margin0[idx], rigid_contact_margin1[idx])
    if C_n > 0.0:
        contact_penalty_k[idx] = wp.min(contact_material_ke[idx], contact_penalty_k[idx] + beta * C_n)


@wp.kernel
def update_duals_body_particle_contacts(
    body_particle_contact_count: wp.array[int],
    body_particle_contact_particle: wp.array[int],
    body_particle_contact_shape: wp.array[int],
    body_particle_contact_body_pos: wp.array[wp.vec3],
    body_particle_contact_normal: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_particle_contact_material_ke: wp.array[float],
    beta: float,
    body_particle_contact_penalty_k: wp.array[float],
):
    """
    Update AVBD penalty parameters for body-particle soft contacts (per-iteration).

    Ramps each contact's penalty by beta * penetration, clamped to the
    per-contact material stiffness ceiling.
    """
    idx = wp.tid()
    if idx >= body_particle_contact_count[0]:
        return

    particle_idx = body_particle_contact_particle[idx]
    shape_idx = body_particle_contact_shape[idx]
    body_idx = shape_body[shape_idx] if shape_idx >= 0 else -1

    stiffness = body_particle_contact_material_ke[idx]

    X_wb = wp.transform_identity()
    if body_idx >= 0:
        X_wb = body_q[body_idx]

    cp_world = wp.transform_point(X_wb, body_particle_contact_body_pos[idx])
    particle_pos = particle_q[particle_idx]
    radius = particle_radius[particle_idx]
    n = body_particle_contact_normal[idx]

    penetration = -(wp.dot(n, particle_pos - cp_world) - radius)
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
    body_contact_buffer_pre_alloc: int,
    body_contact_counts: wp.array[wp.int32],
    body_contact_indices: wp.array[wp.int32],
    contact_stick_flag: wp.array[wp.int32],
    apply_stick_deadzone: int,
    stick_freeze_translation_eps: float,
    stick_freeze_angular_eps: float,
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_qd_mirror: wp.array[wp.spatial_vector],
    body_q_out: wp.array[wp.transform],
):
    """
    Update body velocities from position changes (world frame).

    Optionally applies a tiny body-level stick-contact deadzone before
    finite-difference velocity computation.
    Computes linear and angular velocities using finite differences.
    Also transfers the final body poses to body_q_out (fused copy from
    the in-place Gauss-Seidel iteration buffer to state_out).

    Linear: v = (com_current - com_prev) / dt
    Angular: omega from quaternion difference dq = q * q_prev^-1

    Args:
        dt: Time step.
        body_q: Current body transforms (world), from state_in (in-place iteration buffer).
        body_com: Center of mass offsets (local frame).
        body_contact_buffer_pre_alloc: Per-body contact-list capacity.
        body_contact_counts: Number of body-body contacts adjacent to each body.
        body_contact_indices: Flat per-body contact index lists.
        contact_stick_flag: Per-contact flag (0=none, ANCHOR=sticking kinematic/static,
            DEADZONE=sticking dynamic-dynamic).
        apply_stick_deadzone: If nonzero, enable anti-creep deadzone for bodies whose
            contacts carry DEADZONE but not ANCHOR.
        stick_freeze_translation_eps: Translation deadzone [m] for anti-creep snapping.
        stick_freeze_angular_eps: Angular deadzone [rad] for anti-creep snapping.
        body_q_prev: Previous body transforms (input/output, advanced to current
            pose for next step). For kinematic bodies set body_q. For dynamic
            teleportation also set body_q_prev and body_qd.
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

    if apply_stick_deadzone != 0:
        count = wp.min(body_contact_counts[tid], body_contact_buffer_pre_alloc)
        offset = tid * body_contact_buffer_pre_alloc
        has_anchor = int(0)
        has_deadzone = int(0)
        for i in range(count):
            contact_idx = body_contact_indices[offset + i]
            f = contact_stick_flag[contact_idx]
            if f == _STICK_FLAG_ANCHOR:
                has_anchor = int(1)
            elif f == _STICK_FLAG_DEADZONE:
                has_deadzone = int(1)

        if has_deadzone != 0 and has_anchor == 0:
            translation_delta = wp.length(x - x_prev)
            angular_delta = wp.length(quat_velocity(q, q_prev, 1.0))  # dt=1 gives angular displacement [rad]
            if translation_delta < stick_freeze_translation_eps and angular_delta < stick_freeze_angular_eps:
                pose = pose_prev
                x = x_prev
                q = q_prev

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
def update_cable_dahl_state(
    # Joint geometry
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_constraint_start: wp.array[int],
    joint_penalty_k: wp.array[float],
    joint_is_hard: wp.array[wp.int32],
    joint_cable_kb_rest_local: wp.array[wp.vec3],
    joint_cable_twist_rest: wp.array[float],
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
    Persist cable Dahl hysteresis state after solver convergence.

    State is diagonal in [bend_x, bend_y, twist_z]. Only soft modes with active
    stiffness are advanced; inactive modes clear stress and use final strain as
    the next baseline.
    """
    j = wp.tid()
    zero = wp.vec3(0.0)

    if joint_type[j] != JointType.CABLE:
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

    kappa_final = compute_geometric_cable_kappa_cached_z(
        q_wp,
        q_wc,
        joint_cable_kb_rest_local[j],
        joint_cable_twist_rest[j],
    )

    c_start = joint_constraint_start[j]
    k_dahl = _cable_dahl_active_stiffness(c_start, joint_penalty_k, joint_is_hard)

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

    eps_max = joint_eps_max[j]  # Maximum persistent strain [rad]
    tau = joint_tau[j]  # Memory decay length [rad]

    if eps_max <= 0.0 or tau <= 0.0:
        joint_sigma_prev[j] = zero
        joint_kappa_prev[j] = kappa_final
        joint_dkappa_prev[j] = kappa_final - kappa_old  # store Delta kappa
        return

    sigma_final_out = zero
    d_kappa_out = zero

    for axis in range(3):
        sigma_max = k_dahl[axis] * eps_max  # [N*m]
        if sigma_max <= 0.0:
            continue

        sigma_i_next, d_kappa_i, _direction = _advance_dahl_axis(
            kappa_final[axis],
            kappa_old[axis],
            d_kappa_old[axis],
            sigma_old[axis],
            sigma_max,
            tau,
        )
        sigma_final_out[axis] = sigma_i_next
        d_kappa_out[axis] = d_kappa_i

    # Store final vector state for next timestep: [bend_x, bend_y, twist_z].
    joint_sigma_prev[j] = sigma_final_out
    joint_kappa_prev[j] = kappa_final
    joint_dkappa_prev[j] = d_kappa_out
