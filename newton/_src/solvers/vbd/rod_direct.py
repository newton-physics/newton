# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Direct solve of rod joints inside :class:`~newton.solvers.SolverVBD`'s rigid iteration.

VBD updates one body at a time. A rod segment cannot rotate without stretching
its anchors against neighbours that are frozen during its local solve, so for
slender, light rods (stretch-to-bend stiffness ratio ``k_s (l/2)^2 / k_b``
beyond ~25) bending converges extremely slowly, and softening the primal stretch
term instead loses the ability to carry axial load.

With ``rigid_rod_direct_solve=True``, every connected set of dynamic bodies
joined by :attr:`~newton.JointType.ROD` joints is solved as one block per VBD
iteration: the rod's incremental potential (inertia, the contact and non-rod
joint terms VBD accumulates per body, and the exact rod stretch/shear and
bend/twist energies) is minimized by a Newton step whose Hessian is Gauss-Newton
plus the tension part of the anchors' geometric stiffness (which carries a
tensioned rod's transverse stiffness), solved exactly with a block tree
elimination in float64 that is linear in the number of segments (the tree
analogue of Deul et al. 2018, "Direct Position-Based Solver for Stiff Rods").
A backtracking line search on the same potential keeps the step a descent step.
Rod joints that close a loop, or join a dynamic body to a kinematic one, enter
the block diagonal only.
"""

from __future__ import annotations

import warp as wp

from .rigid_vbd_kernels import (
    _assemble_geometric_rod_kappa_z,
    _measure_rod_bend_twist_z,
    _rod_bend_twist_delta,
    _rod_bend_twist_jacobian_z_from_measure,
)

vec6d = wp.types.vector(length=6, dtype=wp.float64)
mat66d = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
mat36d = wp.types.matrix(shape=(3, 6), dtype=wp.float64)

ROD_DIRECT_FAILED = wp.constant(1)  # status bit: a step was refused (pose kept)
ROD_DIRECT_STALLED = wp.constant(2)  # status bit: no decrease above the estimated round-off
ROD_DIRECT_STALE = wp.constant(4)  # status bit: refused because a body's dynamic state changed under capture
FLOAT32_EPS = wp.constant(wp.float64(1.1920929e-7))
EXTERNAL_HESSIAN_WEIGHT = wp.constant(wp.float64(2.0))
"""Weight of the accumulated contact, attachment and non-rod joint Hessian in the rod step.

All rods are solved in parallel, so a term between two directly solved bodies (two rods, or two segments of one rod)
couples simultaneous block solves, and each solve sees only its own diagonal block. For a PSD metric K and
simultaneous endpoint increments u, v, ``|u + v|_K^2 <= 2 |u|_K^2 + 2 |v|_K^2``, so twice the diagonal curvature
majorizes the omitted cross terms; with weight 1, stiff contacts between rods overshoot and diverge. On one-sided terms
(static, kinematic or colour-solved partners, particles) the weight under-relaxes the step: their friction and active
set enter the line search only as this quadratic model, and with weight 1 rods resting on frictional contacts keep
two to three times more residual motion. The gradient is not weighted, so the force balance at convergence is
unchanged; the weight only shortens steps."""


@wp.func
def _d3(v: wp.vec3) -> wp.vec3d:
    return wp.vec3d(wp.float64(v[0]), wp.float64(v[1]), wp.float64(v[2]))


@wp.func
def _d33(m: wp.mat33) -> wp.mat33d:
    out = wp.mat33d()
    for i in range(3):
        for j in range(3):
            out[i, j] = wp.float64(m[i, j])
    return out


@wp.func
def _skewd(v: wp.vec3d) -> wp.mat33d:
    return wp.mat33d(wp.float64(0.0), -v[2], v[1], v[2], wp.float64(0.0), -v[0], -v[1], v[0], wp.float64(0.0))


@wp.func
def _blocks66(ll: wp.mat33d, al: wp.mat33d, aa: wp.mat33d) -> mat66d:
    """[[ll, al^T], [al, aa]] (VBD's ldlt6_solve layout: angular rows, linear columns in al)."""
    out = mat66d()
    for i in range(3):
        for j in range(3):
            out[i, j] = ll[i, j]
            out[i, j + 3] = al[j, i]
            out[i + 3, j] = al[i, j]
            out[i + 3, j + 3] = aa[i, j]
    return out


@wp.func
def _vec6(lin: wp.vec3d, ang: wp.vec3d) -> vec6d:
    return vec6d(lin[0], lin[1], lin[2], ang[0], ang[1], ang[2])


@wp.func
def _jtkj(Ja: mat36d, k: wp.vec3d, Jb: mat36d) -> mat66d:
    """Ja^T diag(k) Jb."""
    out = mat66d()
    for i in range(6):
        for j in range(6):
            s = wp.float64(0.0)
            for r in range(3):
                s = s + Ja[r, i] * k[r] * Jb[r, j]
            out[i, j] = s
    return out


@wp.func
def _jte(J: mat36d, e: wp.vec3d) -> vec6d:
    out = vec6d()
    for i in range(6):
        s = wp.float64(0.0)
        for r in range(3):
            s = s + J[r, i] * e[r]
        out[i] = s
    return out


@wp.func
def _lin_jac(Rt: wp.mat33d, sgn: wp.float64, ang: wp.mat33d) -> mat36d:
    """Rows of Rt [sgn * I, ang]."""
    out = mat36d()
    for r in range(3):
        for j in range(3):
            out[r, j] = sgn * Rt[r, j]
    a = Rt * ang
    for r in range(3):
        for j in range(3):
            out[r, j + 3] = a[r, j]
    return out


@wp.func
def _ang_jac(J: wp.mat33) -> mat36d:
    out = mat36d()
    for r in range(3):
        for j in range(3):
            out[r, j + 3] = wp.float64(J[r, j])
    return out


@wp.func
def _add33(M: mat66d, r0: int, c0: int, m: wp.mat33d) -> mat66d:
    out = M
    for i in range(3):
        for j in range(3):
            out[r0 + i, c0 + j] = out[r0 + i, c0 + j] + m[i, j]
    return out


@wp.func
def roundoff_estimate(g: vec6d, T: wp.transform) -> wp.float64:
    """Estimate of |g . dq| for the float32 rounding dq of pose T, taking ``|dx| <= eps |x|`` and ``|dtheta| <= eps``.

    It covers the pose rounding only, not the float32 strain, anchor and quaternion arithmetic or second-order
    terms, so it is an estimate of the energy's round-off rather than a bound.
    """
    x = _d3(wp.transform_get_translation(T))
    lin = wp.length(wp.vec3d(g[0], g[1], g[2]))
    ang = wp.length(wp.vec3d(g[3], g[4], g[5]))
    return FLOAT32_EPS * (lin * wp.length(x) + ang)


@wp.func
def _lever_stiffness(f: wp.vec3d, r: wp.vec3d) -> wp.mat33d:
    """max(f . r, 0) (I - r r^T / |r|^2): rotational stiffness of a body pulled by force f at lever arm r."""
    out = wp.mat33d()
    rr = wp.dot(r, r)
    fr = wp.dot(f, r)
    if fr <= wp.float64(0.0) or rr <= wp.float64(0.0):
        return out
    out = -(fr / rr) * wp.outer(r, r)
    for i in range(3):
        out[i, i] = out[i, i] + fr
    return out


@wp.func
def chol6_checked(A: mat66d):
    """Cholesky factor of A and whether every pivot was positive (relative to A's diagonal)."""
    L = mat66d()
    ok = bool(True)
    for i in range(6):
        for j in range(6):
            if j <= i:
                s = A[i, j]
                for k in range(6):
                    if k < j:
                        s = s - L[i, k] * L[j, k]
                if i == j:
                    if s <= wp.float64(1.0e-14) * wp.abs(A[i, i]):
                        ok = False
                    L[i, i] = wp.sqrt(wp.max(s, wp.float64(1.0e-300)))
                else:
                    L[i, j] = s / L[j, j]
    return L, ok


@wp.func
def chol6_solve(L: mat66d, b: vec6d) -> vec6d:
    y = vec6d()
    for i in range(6):
        s = b[i]
        for k in range(6):
            if k < i:
                s = s - L[i, k] * y[k]
        y[i] = s / L[i, i]
    x = vec6d()
    for ii in range(6):
        i = 5 - ii
        s = y[i]
        for k in range(6):
            if k > i:
                s = s - L[k, i] * x[k]
        x[i] = s / L[i, i]
    return x


@wp.func
def chol6_solve_mat(L: mat66d, B: mat66d) -> mat66d:
    X = mat66d()
    for c in range(6):
        col = vec6d()
        for r in range(6):
            col[r] = B[r, c]
        sol = chol6_solve(L, col)
        for r in range(6):
            X[r, c] = sol[r]
    return X


# ---------------------------------------------------------------------------------------------------------------------
# energies
# ---------------------------------------------------------------------------------------------------------------------


@wp.struct
class RodJointTerms:
    energy: wp.float64
    g_p: vec6d
    g_c: vec6d
    H_pp: mat66d
    H_pc: mat66d
    H_cc: mat66d


@wp.func
def _rod_strains(
    Tp: wp.transform, Tc: wp.transform, Xp: wp.transform, Xc: wp.transform, kb_rest: wp.vec3, twist_rest: float
):
    X_wp = wp.transform_multiply(Tp, Xp)
    X_wc = wp.transform_multiply(Tc, Xc)
    q_wp = wp.transform_get_rotation(X_wp)
    q_wc = wp.transform_get_rotation(X_wc)
    C = wp.transform_get_translation(X_wc) - wp.transform_get_translation(X_wp)
    u = wp.quat_rotate_inv(q_wp, C)
    measure = _measure_rod_bend_twist_z(q_wp, q_wc)
    kappa = _assemble_geometric_rod_kappa_z(q_wp, measure.kb_world, measure.twist, kb_rest, twist_rest)
    return X_wp, X_wc, C, u, measure, kappa


@wp.func
def rod_joint_terms(
    Tp: wp.transform,
    Tc: wp.transform,
    Tp_prev: wp.transform,
    Tc_prev: wp.transform,
    com_p_local: wp.vec3,
    com_c_local: wp.vec3,
    Xp: wp.transform,
    Xc: wp.transform,
    kb_rest: wp.vec3,
    twist_rest: float,
    k_lin: wp.vec3d,
    k_ang: wp.vec3d,
    c_lin: wp.vec3d,
    c_ang: wp.vec3d,
) -> RodJointTerms:
    """Elastic rod energy plus the implicit damping potential, gradients and Hessian blocks.

    ``u = R_p^T (x_c - x_p)`` is (shear_x, shear_y, stretch) and ``kappa`` is (bend_x, bend_y, twist) in the parent
    anchor frame; ``k_*`` are per-axis stiffnesses and ``c_* = kd / dt`` per-axis damping coefficients. Unknowns per
    body are (COM translation, world rotation). The blocks are Gauss-Newton plus the tension part of the anchors'
    geometric stiffness, which carries a tensioned rod's transverse (string) stiffness that Gauss-Newton alone misses.
    The other second-order terms (compression, transverse anchor forces, the bend/twist measures' curvature) are
    omitted, so the blocks stay positive semi-definite; the line search on the exact energy covers the difference.
    """
    X_wp, X_wc, C, u, measure, kappa = _rod_strains(Tp, Tc, Xp, Xc, kb_rest, twist_rest)
    _p0, _c0, _C0, u_prev, _m0, kappa_prev = _rod_strains(Tp_prev, Tc_prev, Xp, Xc, kb_rest, twist_rest)
    du = _d3(u - u_prev)
    dk = _d3(_rod_bend_twist_delta(kappa, kappa_prev))
    ud, kd = _d3(u), _d3(kappa)

    q_wp = wp.transform_get_rotation(X_wp)
    Rt = wp.transpose(_d33(wp.quat_to_matrix(q_wp)))
    r_p = _d3(wp.transform_get_translation(X_wp) - wp.transform_point(Tp, com_p_local))
    r_c = _d3(wp.transform_get_translation(X_wc) - wp.transform_point(Tc, com_c_local))
    Jl_p = _lin_jac(Rt, wp.float64(-1.0), _skewd(r_p) + _skewd(_d3(C)))
    Jl_c = _lin_jac(Rt, wp.float64(1.0), -_skewd(r_c))
    Ja_p = _ang_jac(_rod_bend_twist_jacobian_z_from_measure(measure, True))
    Ja_c = _ang_jac(_rod_bend_twist_jacobian_z_from_measure(measure, False))

    e_lin = wp.cw_mul(k_lin, ud) + wp.cw_mul(c_lin, du)
    e_ang = wp.cw_mul(k_ang, kd) + wp.cw_mul(c_ang, dk)
    h_lin = k_lin + c_lin
    h_ang = k_ang + c_ang

    t = RodJointTerms()
    half = wp.float64(0.5)
    t.energy = half * (
        wp.dot(ud, wp.cw_mul(k_lin, ud))
        + wp.dot(du, wp.cw_mul(c_lin, du))
        + wp.dot(kd, wp.cw_mul(k_ang, kd))
        + wp.dot(dk, wp.cw_mul(c_ang, dk))
    )
    t.g_p = _jte(Jl_p, e_lin) + _jte(Ja_p, e_ang)
    t.g_c = _jte(Jl_c, e_lin) + _jte(Ja_c, e_ang)
    t.H_pp = _jtkj(Jl_p, h_lin, Jl_p) + _jtkj(Ja_p, h_ang, Ja_p)
    t.H_pc = _jtkj(Jl_p, h_lin, Jl_c) + _jtkj(Ja_p, h_ang, Ja_c)
    t.H_cc = _jtkj(Jl_c, h_lin, Jl_c) + _jtkj(Ja_c, h_ang, Ja_c)
    # Tension part of the anchors' geometric stiffness f . d2x_anchor: for a body pulled by force f at lever r it is
    # max(f . r, 0) (I - r r^T / |r|^2), exact when lever and force align (a taut rod) and PSD always.
    psi = wp.transpose(Rt) * e_lin  # dE/dx_c: the spring pulls the child anchor by -psi, the parent's by +psi
    t.H_pp = _add33(t.H_pp, 3, 3, _lever_stiffness(psi, r_p))
    t.H_cc = _add33(t.H_cc, 3, 3, _lever_stiffness(-psi, r_c))
    return t


@wp.func
def rod_material(j: int, joint_qd_start: wp.array[int], ke: wp.array[float], kd: wp.array[float], dt: float):
    d = joint_qd_start[j]
    inv_dt = wp.float64(1.0 / dt)
    k_lin = wp.vec3d(wp.float64(ke[d]), wp.float64(ke[d + 1]), wp.float64(ke[d + 2]))
    k_ang = wp.vec3d(wp.float64(ke[d + 3]), wp.float64(ke[d + 4]), wp.float64(ke[d + 5]))
    c_lin = inv_dt * wp.vec3d(wp.float64(kd[d]), wp.float64(kd[d + 1]), wp.float64(kd[d + 2]))
    c_ang = inv_dt * wp.vec3d(wp.float64(kd[d + 3]), wp.float64(kd[d + 4]), wp.float64(kd[d + 5]))
    return k_lin, k_ang, c_lin, c_ang


@wp.func
def apply_increment(T: wp.transform, com_local: wp.vec3, delta: vec6d, alpha: wp.float64) -> wp.transform:
    """Pose after moving the COM by alpha*delta[0:3] and rotating by alpha*delta[3:6] (world)."""
    rot = wp.transform_get_rotation(T)
    com = wp.transform_point(T, com_local)
    dx = wp.vec3(wp.float32(alpha * delta[0]), wp.float32(alpha * delta[1]), wp.float32(alpha * delta[2]))
    w = wp.vec3(wp.float32(alpha * delta[3]), wp.float32(alpha * delta[4]), wp.float32(alpha * delta[5]))
    ang = wp.length(w)
    dq = wp.quat_identity()
    if ang > 1.0e-12:
        dq = wp.quat_from_axis_angle(w / ang, ang)
    rot_new = wp.normalize(wp.mul(dq, rot))
    com_new = com + dx
    return wp.transform(com_new - wp.quat_rotate(rot_new, com_local), rot_new)


@wp.func
def inertial_terms(T: wp.transform, T_star: wp.transform, com_local: wp.vec3, m: float, I_body: wp.mat33, dt: float):
    """VBD's inertial force (negative gradient), Hessian and energy for one body."""
    inv_h2 = wp.float64(1.0 / (dt * dt))
    rot = wp.transform_get_rotation(T)
    com = wp.transform_point(T, com_local)
    com_star = wp.transform_point(T_star, com_local)
    q_delta = wp.mul(wp.quat_inverse(rot), wp.transform_get_rotation(T_star))
    if q_delta[3] < 0.0:
        q_delta = wp.quat(-q_delta[0], -q_delta[1], -q_delta[2], -q_delta[3])
    axis, angle = wp.quat_to_axis_angle(q_delta)
    theta = _d3(axis * angle)
    Ib = _d33(I_body)
    md = wp.float64(m)
    dcom = _d3(com_star - com)
    f_lin = md * inv_h2 * dcom
    R = _d33(wp.quat_to_matrix(rot))
    tau = R * (Ib * theta) * inv_h2
    H_aa = R * Ib * wp.transpose(R) * inv_h2
    H_ll = wp.mat33d(
        md * inv_h2,
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        md * inv_h2,
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        md * inv_h2,
    )
    energy = wp.float64(0.5) * inv_h2 * (md * wp.dot(dcom, dcom) + wp.dot(theta, Ib * theta))
    return _vec6(f_lin, tau), _blocks66(H_ll, wp.mat33d(), H_aa), energy


@wp.func
def rod_joint_energy(
    Tp: wp.transform,
    Tc: wp.transform,
    Tp_prev: wp.transform,
    Tc_prev: wp.transform,
    Xp: wp.transform,
    Xc: wp.transform,
    kb_rest: wp.vec3,
    twist_rest: float,
    k_lin: wp.vec3d,
    k_ang: wp.vec3d,
    c_lin: wp.vec3d,
    c_ang: wp.vec3d,
) -> wp.float64:
    """The energy part of :func:`rod_joint_terms` (line search)."""
    _a, _b, _c, u, _m, kappa = _rod_strains(Tp, Tc, Xp, Xc, kb_rest, twist_rest)
    _a0, _b0, _c0, u_prev, _m0, kappa_prev = _rod_strains(Tp_prev, Tc_prev, Xp, Xc, kb_rest, twist_rest)
    ud, kd = _d3(u), _d3(kappa)
    du = _d3(u - u_prev)
    dk = _d3(_rod_bend_twist_delta(kappa, kappa_prev))
    return wp.float64(0.5) * (
        wp.dot(ud, wp.cw_mul(k_lin, ud))
        + wp.dot(du, wp.cw_mul(c_lin, du))
        + wp.dot(kd, wp.cw_mul(k_ang, kd))
        + wp.dot(dk, wp.cw_mul(c_ang, dk))
    )
