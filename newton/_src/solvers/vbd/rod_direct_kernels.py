# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kernels of the direct rod solve (see :mod:`.rod_direct`)."""

from __future__ import annotations

import warp as wp

from ...sim import JointType
from .rigid_vbd_kernels import (
    RigidForceElementAdjacencyInfo,
    evaluate_joint_force_hessian,
    get_body_adjacent_joint_id,
    get_body_num_adjacent_joints,
)
from .rod_direct import (
    EXTERNAL_HESSIAN_WEIGHT,
    ROD_DIRECT_FAILED,
    ROD_DIRECT_STALE,
    ROD_DIRECT_STALLED,
    _blocks66,
    _d3,
    _d33,
    _vec6,
    apply_increment,
    chol6_checked,
    chol6_solve,
    chol6_solve_mat,
    inertial_terms,
    mat66d,
    rod_joint_terms,
    rod_material,
    roundoff_estimate,
    vec6d,
)
from .rod_direct_energy import _com_or_zero, _pose_or_world, _rod_potential


@wp.kernel(enable_backward=False)
def accumulate_rod_body_joint_terms(
    rod_bodies: wp.array[wp.int32],
    body_node: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_q_rest: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    adjacency: RigidForceElementAdjacencyInfo,
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
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_q: wp.array[float],
    joint_target_qd: wp.array[float],
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
    joint_angle_prev: wp.array[float],
    dt: float,
    # outputs (accumulated)
    body_forces: wp.array[wp.vec3],
    body_torques: wp.array[wp.vec3],
    body_hessian_ll: wp.array[wp.mat33],
    body_hessian_al: wp.array[wp.mat33],
    body_hessian_aa: wp.array[wp.mat33],
):
    """Add every non-rod joint's force/Hessian on a directly solved body (one thread per candidate body)."""
    b = rod_bodies[wp.tid()]
    if body_node[b] < 0:
        return
    for n in range(get_body_num_adjacent_joints(adjacency, b)):
        j = get_body_adjacent_joint_id(adjacency, b, n)
        if joint_type[j] == JointType.ROD:
            continue
        f, t, h_ll, h_al, h_aa = evaluate_joint_force_hessian(
            b,
            j,
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
            joint_angle_prev,
            dt,
        )
        body_forces[b] = body_forces[b] + f
        body_torques[b] = body_torques[b] + t
        body_hessian_ll[b] = body_hessian_ll[b] + h_ll
        body_hessian_al[b] = body_hessian_al[b] + h_al
        body_hessian_aa[b] = body_hessian_aa[b] + h_aa


@wp.func
def _flag_rod(status: wp.array[wp.int32], r: int, bit: int) -> bool:
    """Set a sticky status bit of rod ``r``; ``True`` on its first report since the status was cleared, so warnings
    print once rather than on every iteration."""
    first = (status[r] & bit) == 0
    status[r] = status[r] | bit
    return first


@wp.kernel(enable_backward=False)
def solve_rod_direct(
    dt: float,
    max_halvings: int,
    rod_start: wp.array[wp.int32],
    node_body: wp.array[wp.int32],
    node_parent: wp.array[wp.int32],
    node_joint: wp.array[wp.int32],
    node_side_parent: wp.array[wp.int32],
    bnd_start: wp.array[wp.int32],
    bnd_joint: wp.array[wp.int32],
    body_node: wp.array[wp.int32],
    body_q_prev: wp.array[wp.transform],
    body_inertia_q: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_com: wp.array[wp.vec3],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_qd_start: wp.array[int],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_rod_rest_kb_local: wp.array[wp.vec3],
    joint_rod_rest_twist: wp.array[float],
    ext_force: wp.array[wp.vec3],
    ext_torque: wp.array[wp.vec3],
    ext_h_ll: wp.array[wp.mat33],
    ext_h_al: wp.array[wp.mat33],
    ext_h_aa: wp.array[wp.mat33],
    node_slot: wp.array[wp.int32],
    stale: wp.array[wp.int32],
    global_slot: int,
    # scratch, one entry per node
    D: wp.array[mat66d],
    B: wp.array[mat66d],
    F: wp.array[vec6d],
    S: wp.array[mat66d],
    Y: wp.array[vec6d],
    DEL: wp.array[vec6d],
    HM: wp.array[mat66d],
    FM: wp.array[vec6d],
    T0: wp.array[wp.transform],
    # in/out
    body_q: wp.array[wp.transform],
    status: wp.array[wp.int32],
):
    """One Newton step of a whole rod (one thread per rod): assemble, eliminate leaves-to-root, back-substitute, then
    backtrack on the rod's incremental potential. A singular block or a non-finite step keeps the pose and flags the
    rod in ``status``. Rods beyond the current layout are empty and return at once. A rod whose layout is stale
    (see :func:`check_rod_membership`) is left to the color loop, which ``split_rod_inverse_mass`` gave its bodies."""
    r = wp.tid()
    s = rod_start[r]
    e = rod_start[r + 1]
    if e <= s:
        return
    slot = node_slot[s]
    if stale[global_slot] != 0:
        slot = global_slot
    if stale[slot] != 0:  # rod membership changed under graph capture; the layout no longer matches the model
        status[r] = status[r] | ROD_DIRECT_STALE
        if wp.atomic_max(stale, slot, 2) == 1:  # first report of this slot since the layout was written
            world = slot
            if slot >= global_slot:
                world = -1
            wp.printf(
                "Warning: SolverVBD rigid_rod_direct_solve: a body became or stopped being dynamic under graph "
                "capture (world %d; -1 for global bodies or rods spanning worlds); those rods use the per-body solve "
                "until notify_model_changed() runs outside capture.\n",
                world,
            )
        return

    ok = bool(True)
    noise = wp.float64(0.0)  # estimate of the round-off of the line search's energy differences
    for k in range(s, e):
        b = node_body[k]
        T = body_q[b]
        T0[k] = T
        f_in, H_in, _E = inertial_terms(T, body_inertia_q[b], body_com[b], body_mass[b], body_inertia[b], dt)
        fm = _vec6(_d3(ext_force[b]), _d3(ext_torque[b]))
        noise = noise + roundoff_estimate(f_in, T) + roundoff_estimate(fm, T)
        hm = EXTERNAL_HESSIAN_WEIGHT * _blocks66(_d33(ext_h_ll[b]), _d33(ext_h_al[b]), _d33(ext_h_aa[b]))
        for q in range(bnd_start[k], bnd_start[k + 1]):
            j = bnd_joint[q]
            p = joint_parent[j]
            c = joint_child[j]
            k_lin, k_ang, c_lin, c_ang = rod_material(j, joint_qd_start, joint_target_ke, joint_target_kd, dt)
            t = rod_joint_terms(
                _pose_or_world(body_q, p),
                _pose_or_world(body_q, c),
                _pose_or_world(body_q_prev, p),
                _pose_or_world(body_q_prev, c),
                _com_or_zero(body_com, p),
                _com_or_zero(body_com, c),
                joint_X_p[j],
                joint_X_c[j],
                joint_rod_rest_kb_local[j],
                joint_rod_rest_twist[j],
                k_lin,
                k_ang,
                c_lin,
                c_ang,
            )
            if p == b:
                f_in = f_in - t.g_p
                H_in = H_in + t.H_pp
                noise = noise + roundoff_estimate(t.g_p, T)
            if c == b:
                f_in = f_in - t.g_c
                H_in = H_in + t.H_cc
                noise = noise + roundoff_estimate(t.g_c, T)
        FM[k] = fm
        HM[k] = hm
        D[k] = H_in + hm
        F[k] = f_in + fm
        B[k] = mat66d()

    for k in range(s, e):
        j = node_joint[k]
        if j < 0:
            continue
        P = node_parent[k]
        pb = joint_parent[j]
        cb = joint_child[j]
        k_lin, k_ang, c_lin, c_ang = rod_material(j, joint_qd_start, joint_target_ke, joint_target_kd, dt)
        t = rod_joint_terms(
            body_q[pb],
            body_q[cb],
            body_q_prev[pb],
            body_q_prev[cb],
            body_com[pb],
            body_com[cb],
            joint_X_p[j],
            joint_X_c[j],
            joint_rod_rest_kb_local[j],
            joint_rod_rest_twist[j],
            k_lin,
            k_ang,
            c_lin,
            c_ang,
        )
        noise = noise + roundoff_estimate(t.g_p, body_q[pb]) + roundoff_estimate(t.g_c, body_q[cb])
        if node_side_parent[k] == 1:
            D[k] = D[k] + t.H_pp
            F[k] = F[k] - t.g_p
            D[P] = D[P] + t.H_cc
            F[P] = F[P] - t.g_c
            B[k] = t.H_pc
        else:
            D[k] = D[k] + t.H_cc
            F[k] = F[k] - t.g_c
            D[P] = D[P] + t.H_pp
            F[P] = F[P] - t.g_p
            B[k] = wp.transpose(t.H_pc)

    # leaves-to-root elimination: children precede their parent in node order
    pred = wp.float64(
        0.0
    )  # F^T H^-1 F = sum of the eliminated F_k^T D_k^-1 F_k: the step's predicted first-order decrease
    for k in range(s, e):
        Dk = D[k]
        tr = wp.float64(0.0)
        for i in range(6):
            tr = tr + Dk[i, i]
        eps = wp.float64(1.0e-12) * (tr / wp.float64(6.0)) + wp.float64(1.0e-300)
        for i in range(6):
            Dk[i, i] = Dk[i, i] + eps
        L, pivots_ok = chol6_checked(Dk)
        if not pivots_ok:
            ok = False
        Y[k] = chol6_solve(L, F[k])
        pred = pred + wp.dot(F[k], Y[k])
        P = node_parent[k]
        if P >= 0:
            Sk = chol6_solve_mat(L, B[k])
            S[k] = Sk
            Bt = wp.transpose(B[k])
            D[P] = D[P] - Bt * Sk
            F[P] = F[P] - Bt * Y[k]

    for kk in range(s, e):
        k = s + e - 1 - kk
        P = node_parent[k]
        d = Y[k]
        if P >= 0:
            d = d - S[k] * DEL[P]
        DEL[k] = d
        for i in range(6):
            if not wp.isfinite(d[i]):
                ok = False
    if not ok:
        if _flag_rod(status, r, ROD_DIRECT_FAILED):
            wp.printf(
                "Warning: SolverVBD rigid_rod_direct_solve: the rod containing body %d gave a singular or non-finite "
                "step; pose kept.\n",
                node_body[s],
            )
        return

    alpha = wp.float64(1.0)
    phi0 = _rod_potential(
        s,
        e,
        wp.float64(0.0),
        node_body,
        node_parent,
        node_joint,
        node_side_parent,
        body_q_prev,
        body_inertia_q,
        body_mass,
        body_inertia,
        body_com,
        joint_parent,
        joint_child,
        joint_X_p,
        joint_X_c,
        joint_qd_start,
        joint_target_ke,
        joint_target_kd,
        joint_rod_rest_kb_local,
        joint_rod_rest_twist,
        DEL,
        HM,
        FM,
        T0,
        bnd_start,
        bnd_joint,
        body_node,
        body_q,
        dt,
    )
    if not wp.isfinite(phi0):
        if _flag_rod(status, r, ROD_DIRECT_FAILED):
            wp.printf(
                "Warning: SolverVBD rigid_rod_direct_solve: the rod containing body %d has a non-finite energy; pose "
                "kept.\n",
                node_body[s],
            )
        return
    accepted = bool(False)
    finite_trial = bool(True)
    for _it in range(max_halvings + 1):
        if not accepted:
            phi = _rod_potential(
                s,
                e,
                alpha,
                node_body,
                node_parent,
                node_joint,
                node_side_parent,
                body_q_prev,
                body_inertia_q,
                body_mass,
                body_inertia,
                body_com,
                joint_parent,
                joint_child,
                joint_X_p,
                joint_X_c,
                joint_qd_start,
                joint_target_ke,
                joint_target_kd,
                joint_rod_rest_kb_local,
                joint_rod_rest_twist,
                DEL,
                HM,
                FM,
                T0,
                bnd_start,
                bnd_joint,
                body_node,
                body_q,
                dt,
            )
            finite_trial = wp.isfinite(phi)
            if phi <= phi0:
                accepted = True
            else:
                alpha = alpha * wp.float64(0.5)
    if not accepted:
        # The step is a descent direction (positive definite matrix). A non-finite energy even at the smallest trial
        # step is a failure; otherwise flag the rod only when that step predicted a decrease larger than the
        # estimated round-off of the energy.
        if not finite_trial:
            if _flag_rod(status, r, ROD_DIRECT_FAILED):
                wp.printf(
                    "Warning: SolverVBD rigid_rod_direct_solve: the rod containing body %d has a non-finite energy "
                    "at the smallest trial step; pose kept.\n",
                    node_body[s],
                )
        elif alpha * wp.float64(2.0) * pred > noise:
            status[r] = status[r] | ROD_DIRECT_STALLED
        alpha = wp.float64(0.0)
    for k in range(s, e):
        b = node_body[k]
        body_q[b] = apply_increment(T0[k], body_com[b], DEL[k], alpha)
