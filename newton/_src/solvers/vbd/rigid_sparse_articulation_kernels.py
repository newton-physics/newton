# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for VBD rigid articulation sparse solves."""

import warp as wp

from newton._src.math import quat_velocity
from newton._src.sim import JointType
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    _drive_row_applies_force,
    _eval_joint_axis_drive_limit,
    _limit_row_exists,
    _load_joint_axis_drive_limit,
    _load_solve_weight,
    _material_force_terms,
    _measure_rod_bend_twist_z,
    _rod_bend_twist_jacobian_z_from_measure,
    _structural_row_has_stiffness,
    compute_kappa_and_jacobian,
    compute_kappa_dot,
    evaluate_rod_joint_force_hessian,
)

wp.set_module_options({"enable_backward": False})


class vec6f(wp.types.vector(length=6, dtype=wp.float32)):
    pass


class mat66f(wp.types.matrix(shape=(6, 6), dtype=wp.float32)):
    pass


_SPARSE_BLOCK_DIM = wp.constant(6)
_SPARSE_BLOCK_SIZE = wp.constant(36)
_SPARSE_CTA_THREADS = wp.constant(32)
SPARSE_ARTICULATION_CTA_THREADS = 128


# Warp has no public device-side block or warp barriers for ordinary kernels.
# The cooperative CUDA factorization requires both between dependent phases.
@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _block_sync():
    pass


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncwarp();
#endif
""")
def _warp_sync():
    pass


@wp.func
def _vec6_from_parts(linear: wp.vec3, angular: wp.vec3) -> vec6f:
    return vec6f(linear[0], linear[1], linear[2], angular[0], angular[1], angular[2])


@wp.func
def _scalar_block_index(slot: int, row: int, col: int) -> int:
    return slot * _SPARSE_BLOCK_SIZE + row * _SPARSE_BLOCK_DIM + col


@wp.func
def _scalar_vec_index(row: int, comp: int) -> int:
    return row * _SPARSE_BLOCK_DIM + comp


@wp.func
def _scalar_block_get(values: wp.array[float], slot: int, row: int, col: int) -> float:
    return values[_scalar_block_index(slot, row, col)]


@wp.func
def _scalar_block_set(values: wp.array[float], slot: int, row: int, col: int, value: float):
    values[_scalar_block_index(slot, row, col)] = value


@wp.func
def _scalar_vec_get(values: wp.array[float], row: int, comp: int) -> float:
    return values[_scalar_vec_index(row, comp)]


@wp.func
def _scalar_vec_set(values: wp.array[float], row: int, comp: int, value: float):
    values[_scalar_vec_index(row, comp)] = value


@wp.func
def _joint_kappa(q_wp: wp.quat, q_wc: wp.quat, q_wp_rest: wp.quat, q_wc_rest: wp.quat) -> wp.vec3:
    q_rel = wp.mul(wp.quat_inverse(q_wp), q_wc)
    q_rel_rest = wp.mul(wp.quat_inverse(q_wp_rest), q_wc_rest)
    q_align = wp.mul(q_rel, wp.quat_inverse(q_rel_rest))
    if q_align[3] < 0.0:
        q_align = wp.quat(-q_align[0], -q_align[1], -q_align[2], -q_align[3])
    axis, angle = wp.quat_to_axis_angle(q_align)
    return axis * angle


@wp.func
def _joint_projectors(
    jt: int,
    joint_axis: wp.array[wp.vec3],
    qd_start: int,
    lin_count: int,
    ang_count: int,
    parent_anchor_q: wp.quat,
):
    P_lin = wp.identity(3, float)
    P_ang = wp.identity(3, float)

    if jt == JointType.PRISMATIC:
        axis_w = wp.normalize(wp.quat_rotate(parent_anchor_q, joint_axis[qd_start]))
        P_lin = P_lin - wp.outer(axis_w, axis_w)
    elif jt == JointType.D6:
        if lin_count > 0:
            axis_l0 = wp.normalize(wp.quat_rotate(parent_anchor_q, joint_axis[qd_start]))
            P_lin = P_lin - wp.outer(axis_l0, axis_l0)
        if lin_count > 1:
            axis_l1 = wp.normalize(wp.quat_rotate(parent_anchor_q, joint_axis[qd_start + 1]))
            P_lin = P_lin - wp.outer(axis_l1, axis_l1)
        if lin_count > 2:
            axis_l2 = wp.normalize(wp.quat_rotate(parent_anchor_q, joint_axis[qd_start + 2]))
            P_lin = P_lin - wp.outer(axis_l2, axis_l2)

    if jt == JointType.REVOLUTE:
        axis = wp.normalize(joint_axis[qd_start])
        P_ang = P_ang - wp.outer(axis, axis)
    elif jt == JointType.D6:
        if ang_count > 0:
            axis_a0 = wp.normalize(joint_axis[qd_start + lin_count])
            P_ang = P_ang - wp.outer(axis_a0, axis_a0)
        if ang_count > 1:
            axis_a1 = wp.normalize(joint_axis[qd_start + lin_count + 1])
            P_ang = P_ang - wp.outer(axis_a1, axis_a1)
        if ang_count > 2:
            axis_a2 = wp.normalize(joint_axis[qd_start + lin_count + 2])
            P_ang = P_ang - wp.outer(axis_a2, axis_a2)

    return P_lin, P_ang


@wp.func
def _find_block_slot(
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    row_local: int,
    col_local: int,
) -> int:
    row = body_start + row_local
    begin = articulation_block_row_offsets[row]
    end = articulation_block_row_offsets[row + 1]
    slot = int(-1)
    for s in range(begin, end):
        if articulation_block_cols[s] == col_local:
            slot = s
    return slot


@wp.func
def _add_mat66(values: wp.array[mat66f], slot: int, block: mat66f):
    if slot >= 0:
        values[slot] = values[slot] + block


@wp.func
def _add_rhs(rhs: wp.array[vec6f], index: int, value: vec6f):
    rhs[index] = rhs[index] + value


@wp.func
def _mat66_from_blocks(ll: wp.mat33, al: wp.mat33, aa: wp.mat33) -> mat66f:
    A = mat66f(0.0)
    for i in range(3):
        for j in range(3):
            A[i, j] = ll[i, j]
            A[i + 3, j] = al[i, j]
            A[i, j + 3] = al[j, i]
            A[i + 3, j + 3] = aa[i, j]
    return A


@wp.func
def _mat66_from_angular_block(aa: wp.mat33) -> mat66f:
    A = mat66f(0.0)
    for i in range(3):
        for j in range(3):
            A[i + 3, j + 3] = aa[i, j]
    return A


@wp.func
def _cholesky66_serial(A_in: mat66f) -> mat66f:
    """Factor one diagonal block in the serial block-sparse Cholesky path."""
    L = mat66f(0.0)
    for i in range(6):
        for j in range(6):
            if j <= i:
                value = A_in[i, j]
                for k in range(6):
                    if k < j:
                        value = value - L[i, k] * L[j, k]
                if i == j:
                    L[i, j] = wp.sqrt(value)
                else:
                    L[i, j] = value / L[j, j]
    return L


@wp.func
def _solve_lower66(L: mat66f, b: vec6f) -> vec6f:
    x = vec6f(0.0)
    for i in range(6):
        value = b[i]
        for j in range(6):
            if j < i:
                value = value - L[i, j] * x[j]
        x[i] = value / L[i, i]
    return x


@wp.func
def _solve_upper66_from_lower(L: mat66f, b: vec6f) -> vec6f:
    x = vec6f(0.0)
    for ii in range(6):
        i = 5 - ii
        value = b[i]
        for jj in range(6):
            j = 5 - jj
            if j > i:
                value = value - L[j, i] * x[j]
        x[i] = value / L[i, i]
    return x


@wp.func
def _solve_right_lower_transpose66(A: mat66f, L: mat66f) -> mat66f:
    X = mat66f(0.0)
    for row in range(6):
        b = vec6f(A[row, 0], A[row, 1], A[row, 2], A[row, 3], A[row, 4], A[row, 5])
        x = _solve_lower66(L, b)
        for col in range(6):
            X[row, col] = x[col]
    return X


@wp.func
def _joint_linear_jacobian_value(P: wp.mat33, r: wp.vec3, is_parent: bool, constraint_row: int, dof: int) -> float:
    value = float(0.0)
    sign = float(1.0)
    if is_parent:
        sign = -1.0

    if dof < 3:
        value = sign * P[constraint_row, dof]
    else:
        S = wp.skew(r)
        local_dof = dof - 3
        if is_parent:
            for k in range(3):
                value = value + P[constraint_row, k] * S[k, local_dof]
        else:
            for k in range(3):
                value = value - P[constraint_row, k] * S[k, local_dof]
    return value


@wp.func
def _joint_angular_jacobian_value(P: wp.mat33, is_parent: bool, constraint_row: int, dof: int) -> float:
    if dof < 3:
        return 0.0
    sign = float(1.0)
    if is_parent:
        sign = -1.0
    return sign * P[constraint_row, dof - 3]


@wp.func
def _assemble_constraint_pair(
    values: wp.array[mat66f],
    rhs: wp.array[vec6f],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    local_a: int,
    local_b: int,
    residual: wp.vec3,
    force_scale: float,
    hessian_scale: float,
    P: wp.mat33,
    r_a: wp.vec3,
    r_b: wp.vec3,
    is_parent_a: bool,
    is_parent_b: bool,
    angular_only: bool,
):
    if force_scale == 0.0 and hessian_scale <= 0.0:
        return

    rhs_a = vec6f(0.0)
    rhs_b = vec6f(0.0)
    H_aa = mat66f(0.0)
    H_ab = mat66f(0.0)
    H_bb = mat66f(0.0)

    for i in range(6):
        accum_a = float(0.0)
        accum_b = float(0.0)
        for c in range(3):
            if angular_only:
                Ja = _joint_angular_jacobian_value(P, is_parent_a, c, i)
                Jb = _joint_angular_jacobian_value(P, is_parent_b, c, i)
            else:
                Ja = _joint_linear_jacobian_value(P, r_a, is_parent_a, c, i)
                Jb = _joint_linear_jacobian_value(P, r_b, is_parent_b, c, i)
            accum_a = accum_a + Ja * residual[c]
            accum_b = accum_b + Jb * residual[c]
            for j in range(6):
                if angular_only:
                    Ja_j = _joint_angular_jacobian_value(P, is_parent_a, c, j)
                    Jb_j = _joint_angular_jacobian_value(P, is_parent_b, c, j)
                else:
                    Ja_j = _joint_linear_jacobian_value(P, r_a, is_parent_a, c, j)
                    Jb_j = _joint_linear_jacobian_value(P, r_b, is_parent_b, c, j)
                H_aa[i, j] = H_aa[i, j] + hessian_scale * Ja * Ja_j
                H_ab[i, j] = H_ab[i, j] + hessian_scale * Ja * Jb_j
                H_bb[i, j] = H_bb[i, j] + hessian_scale * Jb * Jb_j
        rhs_a[i] = -force_scale * accum_a
        rhs_b[i] = -force_scale * accum_b

    _add_rhs(rhs, body_start + local_a, rhs_a)
    _add_rhs(rhs, body_start + local_b, rhs_b)

    slot_aa = _find_block_slot(articulation_block_row_offsets, articulation_block_cols, body_start, local_a, local_a)
    slot_bb = _find_block_slot(articulation_block_row_offsets, articulation_block_cols, body_start, local_b, local_b)
    _add_mat66(values, slot_aa, H_aa)
    _add_mat66(values, slot_bb, H_bb)

    if local_a >= local_b:
        slot_ab = _find_block_slot(
            articulation_block_row_offsets, articulation_block_cols, body_start, local_a, local_b
        )
        _add_mat66(values, slot_ab, H_ab)
    else:
        slot_ba = _find_block_slot(
            articulation_block_row_offsets, articulation_block_cols, body_start, local_b, local_a
        )
        _add_mat66(values, slot_ba, wp.transpose(H_ab))


@wp.func
def _assemble_constraint_single(
    values: wp.array[mat66f],
    rhs: wp.array[vec6f],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    local_body: int,
    residual: wp.vec3,
    force_scale: float,
    hessian_scale: float,
    P: wp.mat33,
    r: wp.vec3,
    is_parent: bool,
    angular_only: bool,
):
    if force_scale == 0.0 and hessian_scale <= 0.0:
        return

    rhs_body = vec6f(0.0)
    H = mat66f(0.0)
    for i in range(6):
        accum = float(0.0)
        for c in range(3):
            if angular_only:
                Ji = _joint_angular_jacobian_value(P, is_parent, c, i)
            else:
                Ji = _joint_linear_jacobian_value(P, r, is_parent, c, i)
            accum = accum + Ji * residual[c]
            for j in range(6):
                if angular_only:
                    Jj = _joint_angular_jacobian_value(P, is_parent, c, j)
                else:
                    Jj = _joint_linear_jacobian_value(P, r, is_parent, c, j)
                H[i, j] = H[i, j] + hessian_scale * Ji * Jj
        rhs_body[i] = -force_scale * accum

    _add_rhs(rhs, body_start + local_body, rhs_body)
    slot = _find_block_slot(articulation_block_row_offsets, articulation_block_cols, body_start, local_body, local_body)
    _add_mat66(values, slot, H)


@wp.func
def _assemble_angular_direct_pair(
    values: wp.array[mat66f],
    rhs: wp.array[vec6f],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    parent_local: int,
    child_local: int,
    parent_body: int,
    torque_parent: wp.vec3,
    H_aa: wp.mat33,
):
    rhs_parent = _vec6_from_parts(wp.vec3(0.0), torque_parent)
    rhs_child = _vec6_from_parts(wp.vec3(0.0), -torque_parent)
    H_block = _mat66_from_angular_block(H_aa)

    if parent_body >= 0 and parent_local >= 0:
        _add_rhs(rhs, body_start + parent_local, rhs_parent)
        _add_rhs(rhs, body_start + child_local, rhs_child)

        slot_pp = _find_block_slot(
            articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, parent_local
        )
        slot_cc = _find_block_slot(
            articulation_block_row_offsets, articulation_block_cols, body_start, child_local, child_local
        )
        _add_mat66(values, slot_pp, H_block)
        _add_mat66(values, slot_cc, H_block)

        H_cross = _mat66_from_angular_block(-H_aa)
        if parent_local >= child_local:
            slot_pc = _find_block_slot(
                articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, child_local
            )
            _add_mat66(values, slot_pc, H_cross)
        else:
            slot_cp = _find_block_slot(
                articulation_block_row_offsets, articulation_block_cols, body_start, child_local, parent_local
            )
            _add_mat66(values, slot_cp, H_cross)
    else:
        _add_rhs(rhs, body_start + child_local, rhs_child)
        slot_cc = _find_block_slot(
            articulation_block_row_offsets, articulation_block_cols, body_start, child_local, child_local
        )
        _add_mat66(values, slot_cc, H_block)


@wp.func
def _angular_constraint_force_hessian(
    parent_anchor_q: wp.quat,
    child_anchor_q: wp.quat,
    parent_anchor_q_prev: wp.quat,
    child_anchor_q_prev: wp.quat,
    parent_rest_q: wp.quat,
    child_rest_q: wp.quat,
    stiffness: float,
    P: wp.mat33,
    sigma0: wp.vec3,
    C_fric: wp.vec3,
    lambda_ang: wp.vec3,
    C0_ang: wp.vec3,
    alpha: float,
    damping: float,
    dt: float,
):
    kappa, J_world = compute_kappa_and_jacobian(parent_anchor_q, child_anchor_q, parent_rest_q, child_rest_q)
    kappa_stab = kappa - alpha * C0_ang
    f_local = stiffness * (P * kappa_stab) + sigma0 + P * lambda_ang

    H_local = stiffness * P + wp.mat33(
        C_fric[0],
        0.0,
        0.0,
        0.0,
        C_fric[1],
        0.0,
        0.0,
        0.0,
        C_fric[2],
    )

    if damping > 0.0:
        omega_parent = quat_velocity(parent_anchor_q, parent_anchor_q_prev, dt)
        omega_child = quat_velocity(child_anchor_q, child_anchor_q_prev, dt)
        dkappa_dt = compute_kappa_dot(J_world, omega_parent, omega_child)
        f_local = f_local + damping * (P * dkappa_dt)
        H_local = H_local + (damping / dt) * P

    torque_parent = J_world * f_local
    H_aa = J_world * (H_local * wp.transpose(J_world))
    return torque_parent, H_aa, kappa, J_world


@wp.func
def _assemble_linear_joint(
    values: wp.array[mat66f],
    rhs: wp.array[vec6f],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    parent_local: int,
    child_local: int,
    parent_body: int,
    child_body: int,
    parent_anchor: wp.vec3,
    child_anchor: wp.vec3,
    parent_anchor_prev: wp.vec3,
    child_anchor_prev: wp.vec3,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    stiffness: float,
    damping: float,
    dt: float,
    P: wp.mat33,
    lambda_lin: wp.vec3,
    C0_lin: wp.vec3,
    alpha: float,
):
    C_vec = child_anchor - parent_anchor
    C_stab = C_vec - alpha * C0_lin
    force_residual = stiffness * (P * C_stab) + P * lambda_lin
    hessian_scale = stiffness
    if damping > 0.0:
        residual_prev = child_anchor_prev - parent_anchor_prev
        dC_dt = (C_vec - residual_prev) / dt
        force_residual = force_residual + damping * (P * dC_dt)
        hessian_scale = stiffness + damping / dt

    child_pose = body_q[child_body]
    r_child = child_anchor - wp.transform_point(child_pose, body_com[child_body])

    if parent_body >= 0 and parent_local >= 0:
        parent_pose = body_q[parent_body]
        r_parent = parent_anchor - wp.transform_point(parent_pose, body_com[parent_body])
        _assemble_constraint_pair(
            values,
            rhs,
            articulation_block_row_offsets,
            articulation_block_cols,
            body_start,
            parent_local,
            child_local,
            force_residual,
            1.0,
            hessian_scale,
            P,
            r_parent,
            r_child,
            True,
            False,
            False,
        )
    else:
        _assemble_constraint_single(
            values,
            rhs,
            articulation_block_row_offsets,
            articulation_block_cols,
            body_start,
            child_local,
            force_residual,
            1.0,
            hessian_scale,
            P,
            r_child,
            False,
            False,
        )


@wp.func
def _assemble_angular_joint(
    values: wp.array[mat66f],
    rhs: wp.array[vec6f],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    parent_local: int,
    child_local: int,
    parent_body: int,
    child_body: int,
    parent_anchor_q: wp.quat,
    child_anchor_q: wp.quat,
    parent_anchor_q_prev: wp.quat,
    child_anchor_q_prev: wp.quat,
    parent_rest_q: wp.quat,
    child_rest_q: wp.quat,
    stiffness: float,
    damping: float,
    dt: float,
    P: wp.mat33,
    sigma0: wp.vec3,
    C_fric: wp.vec3,
    lambda_ang: wp.vec3,
    C0_ang: wp.vec3,
    alpha: float,
):
    torque_parent, H_aa, kappa, J_world = _angular_constraint_force_hessian(
        parent_anchor_q,
        child_anchor_q,
        parent_anchor_q_prev,
        child_anchor_q_prev,
        parent_rest_q,
        child_rest_q,
        stiffness,
        P,
        sigma0,
        C_fric,
        lambda_ang,
        C0_ang,
        alpha,
        damping,
        dt,
    )
    _assemble_angular_direct_pair(
        values,
        rhs,
        articulation_block_row_offsets,
        articulation_block_cols,
        body_start,
        parent_local,
        child_local,
        parent_body,
        torque_parent,
        H_aa,
    )
    return kappa, J_world


@wp.func
def _assemble_linear_axis_row(
    values: wp.array[mat66f],
    rhs: wp.array[vec6f],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    parent_local: int,
    child_local: int,
    parent_body: int,
    child_body: int,
    parent_anchor: wp.vec3,
    child_anchor: wp.vec3,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    axis_world: wp.vec3,
    force_scalar: float,
    hessian_scalar: float,
):
    P = wp.outer(axis_world, axis_world)
    residual = axis_world
    child_pose = body_q[child_body]
    r_child = child_anchor - wp.transform_point(child_pose, body_com[child_body])

    if parent_body >= 0 and parent_local >= 0:
        parent_pose = body_q[parent_body]
        r_parent = parent_anchor - wp.transform_point(parent_pose, body_com[parent_body])
        _assemble_constraint_pair(
            values,
            rhs,
            articulation_block_row_offsets,
            articulation_block_cols,
            body_start,
            parent_local,
            child_local,
            residual,
            force_scalar,
            hessian_scalar,
            P,
            r_parent,
            r_child,
            True,
            False,
            False,
        )
    else:
        _assemble_constraint_single(
            values,
            rhs,
            articulation_block_row_offsets,
            articulation_block_cols,
            body_start,
            child_local,
            residual,
            force_scalar,
            hessian_scalar,
            P,
            r_child,
            False,
            False,
        )


@wp.func
def _assemble_angular_axis_row(
    values: wp.array[mat66f],
    rhs: wp.array[vec6f],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    parent_local: int,
    child_local: int,
    parent_body: int,
    angular_jacobian_world: wp.vec3,
    force_scalar: float,
    hessian_scalar: float,
):
    torque_parent = force_scalar * angular_jacobian_world
    H_aa = hessian_scalar * wp.outer(angular_jacobian_world, angular_jacobian_world)
    _assemble_angular_direct_pair(
        values,
        rhs,
        articulation_block_row_offsets,
        articulation_block_cols,
        body_start,
        parent_local,
        child_local,
        parent_body,
        torque_parent,
        H_aa,
    )


@wp.func
def _add_mat66_scalar_atomic(values_scalar: wp.array[float], slot: int, block: mat66f):
    if slot >= 0:
        for i in range(6):
            for j in range(6):
                wp.atomic_add(values_scalar, _scalar_block_index(slot, i, j), block[i, j])


@wp.func
def _add_rhs_scalar_atomic(rhs_scalar: wp.array[float], index: int, value: vec6f):
    for i in range(6):
        wp.atomic_add(rhs_scalar, _scalar_vec_index(index, i), value[i])


@wp.func
def _assemble_constraint_pair_scalar(
    values_scalar: wp.array[float],
    rhs_scalar: wp.array[float],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    local_a: int,
    local_b: int,
    residual: wp.vec3,
    force_scale: float,
    hessian_scale: float,
    P: wp.mat33,
    r_a: wp.vec3,
    r_b: wp.vec3,
    is_parent_a: bool,
    is_parent_b: bool,
    angular_only: bool,
):
    if force_scale == 0.0 and hessian_scale <= 0.0:
        return

    rhs_a = vec6f(0.0)
    rhs_b = vec6f(0.0)
    H_aa = mat66f(0.0)
    H_ab = mat66f(0.0)
    H_bb = mat66f(0.0)

    for i in range(6):
        accum_a = float(0.0)
        accum_b = float(0.0)
        for c in range(3):
            if angular_only:
                Ja = _joint_angular_jacobian_value(P, is_parent_a, c, i)
                Jb = _joint_angular_jacobian_value(P, is_parent_b, c, i)
            else:
                Ja = _joint_linear_jacobian_value(P, r_a, is_parent_a, c, i)
                Jb = _joint_linear_jacobian_value(P, r_b, is_parent_b, c, i)
            accum_a = accum_a + Ja * residual[c]
            accum_b = accum_b + Jb * residual[c]
            for j in range(6):
                if angular_only:
                    Ja_j = _joint_angular_jacobian_value(P, is_parent_a, c, j)
                    Jb_j = _joint_angular_jacobian_value(P, is_parent_b, c, j)
                else:
                    Ja_j = _joint_linear_jacobian_value(P, r_a, is_parent_a, c, j)
                    Jb_j = _joint_linear_jacobian_value(P, r_b, is_parent_b, c, j)
                H_aa[i, j] = H_aa[i, j] + hessian_scale * Ja * Ja_j
                H_ab[i, j] = H_ab[i, j] + hessian_scale * Ja * Jb_j
                H_bb[i, j] = H_bb[i, j] + hessian_scale * Jb * Jb_j
        rhs_a[i] = -force_scale * accum_a
        rhs_b[i] = -force_scale * accum_b

    _add_rhs_scalar_atomic(rhs_scalar, body_start + local_a, rhs_a)
    _add_rhs_scalar_atomic(rhs_scalar, body_start + local_b, rhs_b)

    slot_aa = _find_block_slot(articulation_block_row_offsets, articulation_block_cols, body_start, local_a, local_a)
    slot_bb = _find_block_slot(articulation_block_row_offsets, articulation_block_cols, body_start, local_b, local_b)
    _add_mat66_scalar_atomic(values_scalar, slot_aa, H_aa)
    _add_mat66_scalar_atomic(values_scalar, slot_bb, H_bb)

    if local_a >= local_b:
        slot_ab = _find_block_slot(
            articulation_block_row_offsets, articulation_block_cols, body_start, local_a, local_b
        )
        _add_mat66_scalar_atomic(values_scalar, slot_ab, H_ab)
    else:
        slot_ba = _find_block_slot(
            articulation_block_row_offsets, articulation_block_cols, body_start, local_b, local_a
        )
        _add_mat66_scalar_atomic(values_scalar, slot_ba, wp.transpose(H_ab))


@wp.func
def _assemble_constraint_single_scalar(
    values_scalar: wp.array[float],
    rhs_scalar: wp.array[float],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    local_body: int,
    residual: wp.vec3,
    force_scale: float,
    hessian_scale: float,
    P: wp.mat33,
    r: wp.vec3,
    is_parent: bool,
    angular_only: bool,
):
    if force_scale == 0.0 and hessian_scale <= 0.0:
        return

    rhs_body = vec6f(0.0)
    H = mat66f(0.0)
    for i in range(6):
        accum = float(0.0)
        for c in range(3):
            if angular_only:
                Ji = _joint_angular_jacobian_value(P, is_parent, c, i)
            else:
                Ji = _joint_linear_jacobian_value(P, r, is_parent, c, i)
            accum = accum + Ji * residual[c]
            for j in range(6):
                if angular_only:
                    Jj = _joint_angular_jacobian_value(P, is_parent, c, j)
                else:
                    Jj = _joint_linear_jacobian_value(P, r, is_parent, c, j)
                H[i, j] = H[i, j] + hessian_scale * Ji * Jj
        rhs_body[i] = -force_scale * accum

    _add_rhs_scalar_atomic(rhs_scalar, body_start + local_body, rhs_body)
    slot = _find_block_slot(articulation_block_row_offsets, articulation_block_cols, body_start, local_body, local_body)
    _add_mat66_scalar_atomic(values_scalar, slot, H)


@wp.func
def _assemble_angular_direct_pair_scalar(
    values_scalar: wp.array[float],
    rhs_scalar: wp.array[float],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    parent_local: int,
    child_local: int,
    parent_body: int,
    torque_parent: wp.vec3,
    H_aa: wp.mat33,
):
    rhs_parent = _vec6_from_parts(wp.vec3(0.0), torque_parent)
    rhs_child = _vec6_from_parts(wp.vec3(0.0), -torque_parent)
    H_block = _mat66_from_angular_block(H_aa)

    if parent_body >= 0 and parent_local >= 0:
        _add_rhs_scalar_atomic(rhs_scalar, body_start + parent_local, rhs_parent)
        _add_rhs_scalar_atomic(rhs_scalar, body_start + child_local, rhs_child)
        slot_pp = _find_block_slot(
            articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, parent_local
        )
        slot_cc = _find_block_slot(
            articulation_block_row_offsets, articulation_block_cols, body_start, child_local, child_local
        )
        _add_mat66_scalar_atomic(values_scalar, slot_pp, H_block)
        _add_mat66_scalar_atomic(values_scalar, slot_cc, H_block)

        H_cross = _mat66_from_angular_block(-H_aa)
        if parent_local >= child_local:
            slot_pc = _find_block_slot(
                articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, child_local
            )
            _add_mat66_scalar_atomic(values_scalar, slot_pc, H_cross)
        else:
            slot_cp = _find_block_slot(
                articulation_block_row_offsets, articulation_block_cols, body_start, child_local, parent_local
            )
            _add_mat66_scalar_atomic(values_scalar, slot_cp, H_cross)
    else:
        _add_rhs_scalar_atomic(rhs_scalar, body_start + child_local, rhs_child)
        slot_cc = _find_block_slot(
            articulation_block_row_offsets, articulation_block_cols, body_start, child_local, child_local
        )
        _add_mat66_scalar_atomic(values_scalar, slot_cc, H_block)


@wp.func
def _assemble_linear_joint_scalar(
    values_scalar: wp.array[float],
    rhs_scalar: wp.array[float],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    parent_local: int,
    child_local: int,
    parent_body: int,
    child_body: int,
    parent_anchor: wp.vec3,
    child_anchor: wp.vec3,
    parent_anchor_prev: wp.vec3,
    child_anchor_prev: wp.vec3,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    stiffness: float,
    damping: float,
    dt: float,
    P: wp.mat33,
    lambda_lin: wp.vec3,
    C0_lin: wp.vec3,
    alpha: float,
):
    C_vec = child_anchor - parent_anchor
    C_stab = C_vec - alpha * C0_lin
    force_residual = stiffness * (P * C_stab) + P * lambda_lin
    hessian_scale = stiffness
    if damping > 0.0:
        residual_prev = child_anchor_prev - parent_anchor_prev
        dC_dt = (C_vec - residual_prev) / dt
        force_residual = force_residual + damping * (P * dC_dt)
        hessian_scale = stiffness + damping / dt

    child_pose = body_q[child_body]
    r_child = child_anchor - wp.transform_point(child_pose, body_com[child_body])

    if parent_body >= 0 and parent_local >= 0:
        parent_pose = body_q[parent_body]
        r_parent = parent_anchor - wp.transform_point(parent_pose, body_com[parent_body])
        _assemble_constraint_pair_scalar(
            values_scalar,
            rhs_scalar,
            articulation_block_row_offsets,
            articulation_block_cols,
            body_start,
            parent_local,
            child_local,
            force_residual,
            1.0,
            hessian_scale,
            P,
            r_parent,
            r_child,
            True,
            False,
            False,
        )
    else:
        _assemble_constraint_single_scalar(
            values_scalar,
            rhs_scalar,
            articulation_block_row_offsets,
            articulation_block_cols,
            body_start,
            child_local,
            force_residual,
            1.0,
            hessian_scale,
            P,
            r_child,
            False,
            False,
        )


@wp.func
def _assemble_angular_joint_scalar(
    values_scalar: wp.array[float],
    rhs_scalar: wp.array[float],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    parent_local: int,
    child_local: int,
    parent_body: int,
    parent_anchor_q: wp.quat,
    child_anchor_q: wp.quat,
    parent_anchor_q_prev: wp.quat,
    child_anchor_q_prev: wp.quat,
    parent_rest_q: wp.quat,
    child_rest_q: wp.quat,
    stiffness: float,
    damping: float,
    dt: float,
    P: wp.mat33,
    sigma0: wp.vec3,
    C_fric: wp.vec3,
    lambda_ang: wp.vec3,
    C0_ang: wp.vec3,
    alpha: float,
):
    torque_parent, H_aa, kappa, J_world = _angular_constraint_force_hessian(
        parent_anchor_q,
        child_anchor_q,
        parent_anchor_q_prev,
        child_anchor_q_prev,
        parent_rest_q,
        child_rest_q,
        stiffness,
        P,
        sigma0,
        C_fric,
        lambda_ang,
        C0_ang,
        alpha,
        damping,
        dt,
    )
    _assemble_angular_direct_pair_scalar(
        values_scalar,
        rhs_scalar,
        articulation_block_row_offsets,
        articulation_block_cols,
        body_start,
        parent_local,
        child_local,
        parent_body,
        torque_parent,
        H_aa,
    )
    return kappa, J_world


@wp.func
def _assemble_linear_axis_row_scalar(
    values_scalar: wp.array[float],
    rhs_scalar: wp.array[float],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    parent_local: int,
    child_local: int,
    parent_body: int,
    child_body: int,
    parent_anchor: wp.vec3,
    child_anchor: wp.vec3,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    axis_world: wp.vec3,
    force_scalar: float,
    hessian_scalar: float,
):
    P = wp.outer(axis_world, axis_world)
    residual = axis_world
    child_pose = body_q[child_body]
    r_child = child_anchor - wp.transform_point(child_pose, body_com[child_body])

    if parent_body >= 0 and parent_local >= 0:
        parent_pose = body_q[parent_body]
        r_parent = parent_anchor - wp.transform_point(parent_pose, body_com[parent_body])
        _assemble_constraint_pair_scalar(
            values_scalar,
            rhs_scalar,
            articulation_block_row_offsets,
            articulation_block_cols,
            body_start,
            parent_local,
            child_local,
            residual,
            force_scalar,
            hessian_scalar,
            P,
            r_parent,
            r_child,
            True,
            False,
            False,
        )
    else:
        _assemble_constraint_single_scalar(
            values_scalar,
            rhs_scalar,
            articulation_block_row_offsets,
            articulation_block_cols,
            body_start,
            child_local,
            residual,
            force_scalar,
            hessian_scalar,
            P,
            r_child,
            False,
            False,
        )


@wp.func
def _assemble_angular_axis_row_scalar(
    values_scalar: wp.array[float],
    rhs_scalar: wp.array[float],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    parent_local: int,
    child_local: int,
    parent_body: int,
    angular_jacobian_world: wp.vec3,
    force_scalar: float,
    hessian_scalar: float,
):
    torque_parent = force_scalar * angular_jacobian_world
    H_aa = hessian_scalar * wp.outer(angular_jacobian_world, angular_jacobian_world)
    _assemble_angular_direct_pair_scalar(
        values_scalar,
        rhs_scalar,
        articulation_block_row_offsets,
        articulation_block_cols,
        body_start,
        parent_local,
        child_local,
        parent_body,
        torque_parent,
        H_aa,
    )


@wp.func
def _body_diagonal_contribution(
    dt: float,
    local_body: int,
    articulation_bodies: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_inv_mass: wp.array[float],
    body_com: wp.array[wp.vec3],
    body_inertia: wp.array[wp.mat33],
    body_inertia_q: wp.array[wp.transform],
    body_forces: wp.array[wp.vec3],
    body_torques: wp.array[wp.vec3],
    body_hessian_ll: wp.array[wp.mat33],
    body_hessian_al: wp.array[wp.mat33],
    body_hessian_aa: wp.array[wp.mat33],
):
    body = articulation_bodies[local_body]
    q_current = body_q[body]

    diag = mat66f(0.0)
    rhs_value = vec6f(0.0)
    if body_inv_mass[body] == 0.0:
        diag = wp.identity(6, float) * 1.0e30
    else:
        q_star = body_inertia_q[body]
        body_com_local = body_com[body]
        pos_current = wp.transform_get_translation(q_current)
        rot_current = wp.transform_get_rotation(q_current)
        pos_star = wp.transform_get_translation(q_star)
        rot_star = wp.transform_get_rotation(q_star)
        com_current = pos_current + wp.quat_rotate(rot_current, body_com_local)
        com_star = pos_star + wp.quat_rotate(rot_star, body_com_local)

        dt_inv_sqr = 1.0 / (dt * dt)
        inertial_coeff = body_mass[body] * dt_inv_sqr
        f_lin = (com_star - com_current) * inertial_coeff + body_forces[body]

        q_delta = wp.mul(wp.quat_inverse(rot_current), rot_star)
        if q_delta[3] < 0.0:
            q_delta = wp.quat(-q_delta[0], -q_delta[1], -q_delta[2], -q_delta[3])

        axis_body, angle_body = wp.quat_to_axis_angle(q_delta)
        theta_body = axis_body * angle_body
        I_body = body_inertia[body]
        tau_world = wp.quat_rotate(rot_current, I_body * (theta_body * dt_inv_sqr)) + body_torques[body]
        R = wp.quat_to_matrix(rot_current)
        I_world = R * I_body * wp.transpose(R)

        H_ll = body_hessian_ll[body]
        H_ll[0, 0] = H_ll[0, 0] + inertial_coeff
        H_ll[1, 1] = H_ll[1, 1] + inertial_coeff
        H_ll[2, 2] = H_ll[2, 2] + inertial_coeff
        H_al = body_hessian_al[body]
        H_aa = body_hessian_aa[body] + I_world * dt_inv_sqr
        H_aa[0, 0] = H_aa[0, 0] + 1.0e-6
        H_aa[1, 1] = H_aa[1, 1] + 1.0e-6
        H_aa[2, 2] = H_aa[2, 2] + 1.0e-6

        diag = _mat66_from_blocks(H_ll, H_al, H_aa)
        rhs_value = _vec6_from_parts(f_lin, tau_world)

    return diag, rhs_value


@wp.func
def _apply_sparse_delta_value_to_body(
    local_body: int,
    articulation_bodies: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_inv_mass: wp.array[float],
    body_com: wp.array[wp.vec3],
    update_relaxation: float,
    dx_in: vec6f,
    body_q_new: wp.array[wp.transform],
):
    body = articulation_bodies[local_body]
    q_current = body_q[body]
    if body_inv_mass[body] == 0.0:
        body_q_new[body] = q_current
    else:
        dx = dx_in * update_relaxation
        pos = wp.transform_get_translation(q_current)
        rot = wp.transform_get_rotation(q_current)
        body_com_local = body_com[body]
        com = pos + wp.quat_rotate(rot, body_com_local)
        w = wp.vec3(dx[3], dx[4], dx[5])
        angle = wp.length(w)
        if angle > 1.0e-7:
            dq = wp.quat_from_axis_angle(w / angle, angle)
        else:
            half_w = 0.5 * w
            dq = wp.normalize(wp.quat(half_w[0], half_w[1], half_w[2], 1.0))
        rot_new = wp.normalize(wp.mul(dq, rot))
        com_new = com + wp.vec3(dx[0], dx[1], dx[2])
        pos_new = com_new - wp.quat_rotate(rot_new, body_com_local)
        body_q_new[body] = wp.transform(pos_new, rot_new)


@wp.func
def _apply_sparse_delta_to_body(
    local_body: int,
    articulation_bodies: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_inv_mass: wp.array[float],
    body_com: wp.array[wp.vec3],
    update_relaxation: float,
    delta: wp.array[vec6f],
    body_q_new: wp.array[wp.transform],
):
    _apply_sparse_delta_value_to_body(
        local_body,
        articulation_bodies,
        body_q,
        body_inv_mass,
        body_com,
        update_relaxation,
        delta[local_body],
        body_q_new,
    )


@wp.kernel
def regularize_articulation_body_hessian(
    articulation_bodies: wp.array[wp.int32],
    regularization: float,
    body_hessian_ll: wp.array[wp.mat33],
    body_hessian_aa: wp.array[wp.mat33],
):
    body = articulation_bodies[wp.tid()]
    H_ll = body_hessian_ll[body]
    H_aa = body_hessian_aa[body]
    for i in range(3):
        H_ll[i, i] = H_ll[i, i] + regularization
        H_aa[i, i] = H_aa[i, i] + regularization
    body_hessian_ll[body] = H_ll
    body_hessian_aa[body] = H_aa


@wp.func
def _load_sparse_structural_coefficients(
    jt: int,
    c_start: int,
    joint_penalty_k: wp.array[float],
    joint_rho: wp.array[float],
    joint_material_k: wp.array[float],
    joint_penalty_kd: wp.array[float],
    joint_compliant_alm: int,
):
    solve_weight_linear = float(0.0)
    solve_weight_angular = float(0.0)
    material_k_linear = float(0.0)
    material_k_angular = float(0.0)
    kd_linear = float(0.0)
    kd_angular = float(0.0)

    if jt == JointType.BALL:
        solve_weight_linear = _load_solve_weight(joint_penalty_k, joint_rho, c_start, joint_compliant_alm)
        material_k_linear = joint_material_k[c_start]
        kd_linear = joint_penalty_kd[c_start]
    else:
        solve_weight_linear = _load_solve_weight(joint_penalty_k, joint_rho, c_start, joint_compliant_alm)
        solve_weight_angular = _load_solve_weight(joint_penalty_k, joint_rho, c_start + 1, joint_compliant_alm)
        material_k_linear = joint_material_k[c_start]
        material_k_angular = joint_material_k[c_start + 1]
        kd_linear = joint_penalty_kd[c_start]
        kd_angular = joint_penalty_kd[c_start + 1]

    return (
        solve_weight_linear,
        solve_weight_angular,
        material_k_linear,
        material_k_angular,
        kd_linear,
        kd_angular,
    )


@wp.func
def _evaluate_sparse_rod_body(
    body: int,
    joint: int,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_rod_rest_kb_local: wp.array[wp.vec3],
    joint_rod_rest_twist: wp.array[float],
    joint_constraint_start: wp.array[int],
    joint_penalty_k: wp.array[float],
    joint_rho: wp.array[float],
    joint_material_k: wp.array[float],
    joint_penalty_kd: wp.array[float],
    joint_sigma_start: wp.array[wp.vec3],
    joint_C_fric: wp.array[wp.vec3],
    joint_lambda_lin: wp.array[wp.vec3],
    joint_lambda_ang: wp.array[wp.vec3],
    joint_C0_lin: wp.array[wp.vec3],
    joint_C0_ang: wp.array[wp.vec3],
    joint_is_hard: wp.array[wp.int32],
    avbd_alpha: float,
    joint_compliant_alm: int,
    dt: float,
):
    return evaluate_rod_joint_force_hessian(
        body,
        joint,
        body_q,
        body_q_prev,
        body_com,
        joint_parent,
        joint_child,
        joint_X_p,
        joint_X_c,
        joint_rod_rest_kb_local,
        joint_rod_rest_twist,
        joint_constraint_start,
        joint_penalty_k,
        joint_rho,
        joint_material_k,
        joint_penalty_kd,
        joint_sigma_start,
        joint_C_fric,
        joint_lambda_lin,
        joint_lambda_ang,
        joint_C0_lin,
        joint_C0_ang,
        joint_is_hard,
        avbd_alpha,
        joint_compliant_alm,
        dt,
    )


@wp.func
def _add_rod_body_contribution(
    values: wp.array[mat66f],
    rhs: wp.array[vec6f],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    local_body: int,
    force: wp.vec3,
    torque: wp.vec3,
    H_ll: wp.mat33,
    H_al: wp.mat33,
    H_aa: wp.mat33,
):
    row = body_start + local_body
    _add_rhs(rhs, row, _vec6_from_parts(force, torque))
    slot = _find_block_slot(articulation_block_row_offsets, articulation_block_cols, body_start, local_body, local_body)
    _add_mat66(values, slot, _mat66_from_blocks(H_ll, H_al, H_aa))


@wp.func
def _add_rod_body_contribution_scalar(
    values_scalar: wp.array[float],
    rhs_scalar: wp.array[float],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    local_body: int,
    force: wp.vec3,
    torque: wp.vec3,
    H_ll: wp.mat33,
    H_al: wp.mat33,
    H_aa: wp.mat33,
):
    row = body_start + local_body
    _add_rhs_scalar_atomic(rhs_scalar, row, _vec6_from_parts(force, torque))
    slot = _find_block_slot(articulation_block_row_offsets, articulation_block_cols, body_start, local_body, local_body)
    _add_mat66_scalar_atomic(values_scalar, slot, _mat66_from_blocks(H_ll, H_al, H_aa))


@wp.func
def _rod_cross_hessian(
    dt: float,
    joint: int,
    parent: int,
    child: int,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_constraint_start: wp.array[int],
    joint_penalty_k: wp.array[float],
    joint_rho: wp.array[float],
    joint_material_k: wp.array[float],
    joint_penalty_kd: wp.array[float],
    joint_C_fric: wp.array[wp.vec3],
    joint_is_hard: wp.array[wp.int32],
    joint_compliant_alm: int,
):
    """Return the parent-child Gauss-Newton block for one ROD joint."""
    cross = mat66f(0.0)
    c_start = joint_constraint_start[joint]
    stretch_idx = c_start
    shear_idx = c_start + 1
    bend_idx = c_start + 2
    twist_idx = c_start + 3

    stretch_weight = _load_solve_weight(joint_penalty_k, joint_rho, stretch_idx, joint_compliant_alm)
    shear_weight = _load_solve_weight(joint_penalty_k, joint_rho, shear_idx, joint_compliant_alm)
    bend_weight = _load_solve_weight(joint_penalty_k, joint_rho, bend_idx, joint_compliant_alm)
    twist_weight = _load_solve_weight(joint_penalty_k, joint_rho, twist_idx, joint_compliant_alm)

    k_stretch, _ = _material_force_terms(
        stretch_weight, joint_material_k[stretch_idx], wp.vec3(0.0), joint_compliant_alm
    )
    k_shear, _ = _material_force_terms(shear_weight, joint_material_k[shear_idx], wp.vec3(0.0), joint_compliant_alm)
    k_bend, _ = _material_force_terms(bend_weight, joint_material_k[bend_idx], wp.vec3(0.0), joint_compliant_alm)
    k_twist, _ = _material_force_terms(twist_weight, joint_material_k[twist_idx], wp.vec3(0.0), joint_compliant_alm)

    X_wp = body_q[parent] * joint_X_p[joint]
    X_wc = body_q[child] * joint_X_c[joint]
    q_wp = wp.transform_get_rotation(X_wp)
    q_wc = wp.transform_get_rotation(X_wc)

    h_shear = k_shear + joint_penalty_kd[shear_idx] / dt
    h_stretch = k_stretch + joint_penalty_kd[stretch_idx] / dt
    if h_shear > 0.0 or h_stretch > 0.0:
        tangent = wp.quat_rotate(q_wp, wp.vec3(0.0, 0.0, 1.0))
        K_world = h_shear * wp.identity(3, float) + (h_stretch - h_shear) * wp.outer(tangent, tangent)

        # Match the PSD split used by evaluate_rod_stretch_shear_force_hessian:
        # the isotropic elastic part acts at the parent anchor while anisotropy
        # and damping use the shared child-anchor arm on the parent body.
        k_iso = wp.min(k_shear, k_stretch)
        K_material = K_world - k_iso * wp.identity(3, float)
        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        parent_com_world = wp.transform_point(body_q[parent], body_com[parent])
        child_com_world = wp.transform_point(body_q[child], body_com[child])
        r_parent_anchor = x_p - parent_com_world
        r_parent_material = x_c - parent_com_world
        r_child = x_c - child_com_world
        identity = wp.identity(3, float)

        for i in range(6):
            for j in range(6):
                value = float(0.0)
                for row in range(3):
                    J_parent_iso = _joint_linear_jacobian_value(identity, r_parent_anchor, True, row, i)
                    J_child_value = _joint_linear_jacobian_value(identity, r_child, False, row, j)
                    value = value + k_iso * J_parent_iso * J_child_value
                    J_parent_material = _joint_linear_jacobian_value(identity, r_parent_material, True, row, i)
                    for col in range(3):
                        J_child_col = _joint_linear_jacobian_value(identity, r_child, False, col, j)
                        value = value + J_parent_material * K_material[row, col] * J_child_col
                cross[i, j] = cross[i, j] + value

    bend_stiff = _structural_row_has_stiffness(bend_weight, joint_material_k[bend_idx], joint_compliant_alm)
    twist_stiff = _structural_row_has_stiffness(twist_weight, joint_material_k[twist_idx], joint_compliant_alm)
    bend_hard = bend_stiff and joint_compliant_alm == 0 and joint_is_hard[bend_idx] == 1
    twist_hard = twist_stiff and joint_compliant_alm == 0 and joint_is_hard[twist_idx] == 1
    friction = joint_C_fric[joint]
    h_bend_x = k_bend + joint_penalty_kd[bend_idx] / dt
    h_bend_y = h_bend_x
    h_twist = k_twist + joint_penalty_kd[twist_idx] / dt
    if bend_stiff and not bend_hard:
        h_bend_x = h_bend_x + friction[0]
        h_bend_y = h_bend_y + friction[1]
    if twist_stiff and not twist_hard:
        h_twist = h_twist + friction[2]

    if h_bend_x > 0.0 or h_bend_y > 0.0 or h_twist > 0.0:
        measure = _measure_rod_bend_twist_z(q_wp, q_wc)
        J_parent_angular = _rod_bend_twist_jacobian_z_from_measure(measure, True)
        J_child_angular = _rod_bend_twist_jacobian_z_from_measure(measure, False)
        H_local_diag = wp.vec3(h_bend_x, h_bend_y, h_twist)
        for i in range(3):
            for j in range(3):
                value = float(0.0)
                for row in range(3):
                    value = value + J_parent_angular[row, i] * H_local_diag[row] * J_child_angular[row, j]
                cross[i + 3, j + 3] = cross[i + 3, j + 3] + value

    return cross


@wp.func
def _add_rod_cross_block(
    values: wp.array[mat66f],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    parent_local: int,
    child_local: int,
    cross: mat66f,
):
    if parent_local >= child_local:
        slot = _find_block_slot(
            articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, child_local
        )
        _add_mat66(values, slot, cross)
    else:
        slot = _find_block_slot(
            articulation_block_row_offsets, articulation_block_cols, body_start, child_local, parent_local
        )
        _add_mat66(values, slot, wp.transpose(cross))


@wp.func
def _add_rod_cross_block_scalar(
    values_scalar: wp.array[float],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    parent_local: int,
    child_local: int,
    cross: mat66f,
):
    if parent_local >= child_local:
        slot = _find_block_slot(
            articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, child_local
        )
        _add_mat66_scalar_atomic(values_scalar, slot, cross)
    else:
        slot = _find_block_slot(
            articulation_block_row_offsets, articulation_block_cols, body_start, child_local, parent_local
        )
        _add_mat66_scalar_atomic(values_scalar, slot, wp.transpose(cross))


@wp.func
def _evaluate_sparse_drive_limit(
    dof_idx: int,
    target_q_idx: int,
    penalty_slot: int,
    q: float,
    rate: float,
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_q: wp.array[float],
    joint_target_vel: wp.array[float],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_limit_ke: wp.array[float],
    joint_limit_kd: wp.array[float],
    joint_penalty_k: wp.array[float],
    joint_drive_limit_support: wp.array[float],
    joint_drive_lambda: wp.array[float],
    joint_limit_lambda: wp.array[float],
    joint_compliant_alm: int,
    dt: float,
):
    axis = _load_joint_axis_drive_limit(
        dof_idx,
        target_q_idx,
        penalty_slot,
        joint_target_ke,
        joint_target_kd,
        joint_target_q,
        joint_target_vel,
        joint_limit_lower,
        joint_limit_upper,
        joint_limit_ke,
        joint_limit_kd,
        joint_penalty_k,
        joint_compliant_alm,
    )
    has_drive = _drive_row_applies_force(axis.material_drive_ke, axis.drive_kd)
    has_limits = _limit_row_exists(axis.material_limit_ke, axis.lower, axis.upper)
    if has_drive or has_limits:
        return _eval_joint_axis_drive_limit(
            axis,
            q,
            rate,
            has_drive,
            has_limits,
            joint_drive_limit_support[dof_idx],
            joint_drive_lambda[dof_idx],
            joint_limit_lambda[dof_idx],
            joint_compliant_alm,
            1.0 / dt,
        )
    return float(0.0), float(0.0)


@wp.kernel
def assemble_articulation_body_diagonal_scalar(
    dt: float,
    articulation_bodies: wp.array[wp.int32],
    articulation_diag_slots: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_inv_mass: wp.array[float],
    body_com: wp.array[wp.vec3],
    body_inertia: wp.array[wp.mat33],
    body_inertia_q: wp.array[wp.transform],
    body_forces: wp.array[wp.vec3],
    body_torques: wp.array[wp.vec3],
    body_hessian_ll: wp.array[wp.mat33],
    body_hessian_al: wp.array[wp.mat33],
    body_hessian_aa: wp.array[wp.mat33],
    values_scalar: wp.array[float],
    rhs_scalar: wp.array[float],
):
    local_body = wp.tid()
    diag, rhs_value = _body_diagonal_contribution(
        dt,
        local_body,
        articulation_bodies,
        body_q,
        body_mass,
        body_inv_mass,
        body_com,
        body_inertia,
        body_inertia_q,
        body_forces,
        body_torques,
        body_hessian_ll,
        body_hessian_al,
        body_hessian_aa,
    )
    diag_slot = articulation_diag_slots[local_body]
    for i in range(6):
        _scalar_vec_set(rhs_scalar, local_body, i, rhs_value[i])
        for j in range(6):
            _scalar_block_set(values_scalar, diag_slot, i, j, diag[i, j])


@wp.kernel
def assemble_articulation_joints_scalar(
    dt: float,
    articulation_joints: wp.array[wp.int32],
    articulation_joint_body_start: wp.array[wp.int32],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_articulation_local: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_q_rest: wp.array[wp.transform],
    body_inertia_q: wp.array[wp.transform],
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
    joint_dof_dim: wp.array2d[int],
    joint_rest_angle: wp.array[float],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_armature: wp.array[float],
    joint_target_q: wp.array[float],
    joint_target_vel: wp.array[float],
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
    avbd_alpha: float,
    joint_compliant_alm: int,
    values_scalar: wp.array[float],
    rhs_scalar: wp.array[float],
):
    joint_cursor = wp.tid()
    joint = articulation_joints[joint_cursor]
    if not joint_enabled[joint]:
        return

    jt = joint_type[joint]
    if (
        jt != JointType.ROD
        and jt != JointType.BALL
        and jt != JointType.FIXED
        and jt != JointType.REVOLUTE
        and jt != JointType.PRISMATIC
        and jt != JointType.D6
    ):
        return

    child = joint_child[joint]
    parent = joint_parent[joint]
    if child < 0:
        return

    body_start = articulation_joint_body_start[joint_cursor]
    child_local = body_articulation_local[child]
    parent_local = -1
    if parent >= 0:
        parent_local = body_articulation_local[parent]

    if jt == JointType.ROD:
        child_force, child_torque, child_H_ll, child_H_al, child_H_aa = _evaluate_sparse_rod_body(
            child,
            joint,
            body_q,
            body_q_prev,
            body_com,
            joint_parent,
            joint_child,
            joint_X_p,
            joint_X_c,
            joint_rod_rest_kb_local,
            joint_rod_rest_twist,
            joint_constraint_start,
            joint_penalty_k,
            joint_rho,
            joint_material_k,
            joint_penalty_kd,
            joint_sigma_start,
            joint_C_fric,
            joint_lambda_lin,
            joint_lambda_ang,
            joint_C0_lin,
            joint_C0_ang,
            joint_is_hard,
            avbd_alpha,
            joint_compliant_alm,
            dt,
        )
        _add_rod_body_contribution_scalar(
            values_scalar,
            rhs_scalar,
            articulation_block_row_offsets,
            articulation_block_cols,
            body_start,
            child_local,
            child_force,
            child_torque,
            child_H_ll,
            child_H_al,
            child_H_aa,
        )
        if parent >= 0:
            parent_force, parent_torque, parent_H_ll, parent_H_al, parent_H_aa = _evaluate_sparse_rod_body(
                parent,
                joint,
                body_q,
                body_q_prev,
                body_com,
                joint_parent,
                joint_child,
                joint_X_p,
                joint_X_c,
                joint_rod_rest_kb_local,
                joint_rod_rest_twist,
                joint_constraint_start,
                joint_penalty_k,
                joint_rho,
                joint_material_k,
                joint_penalty_kd,
                joint_sigma_start,
                joint_C_fric,
                joint_lambda_lin,
                joint_lambda_ang,
                joint_C0_lin,
                joint_C0_ang,
                joint_is_hard,
                avbd_alpha,
                joint_compliant_alm,
                dt,
            )
            _add_rod_body_contribution_scalar(
                values_scalar,
                rhs_scalar,
                articulation_block_row_offsets,
                articulation_block_cols,
                body_start,
                parent_local,
                parent_force,
                parent_torque,
                parent_H_ll,
                parent_H_al,
                parent_H_aa,
            )
            cross = _rod_cross_hessian(
                dt,
                joint,
                parent,
                child,
                body_q,
                body_com,
                joint_X_p,
                joint_X_c,
                joint_constraint_start,
                joint_penalty_k,
                joint_rho,
                joint_material_k,
                joint_penalty_kd,
                joint_C_fric,
                joint_is_hard,
                joint_compliant_alm,
            )
            _add_rod_cross_block_scalar(
                values_scalar,
                articulation_block_row_offsets,
                articulation_block_cols,
                body_start,
                parent_local,
                child_local,
                cross,
            )
        return

    child_pose = body_q[child]
    child_prev_pose = body_q_prev[child]
    child_rest_pose = body_q_rest[child]
    X_c = joint_X_c[joint]
    child_anchor = wp.transform_point(child_pose, wp.transform_get_translation(X_c))
    child_anchor_prev = wp.transform_point(child_prev_pose, wp.transform_get_translation(X_c))
    child_anchor_q = wp.mul(wp.transform_get_rotation(child_pose), wp.transform_get_rotation(X_c))
    child_anchor_q_prev = wp.mul(wp.transform_get_rotation(child_prev_pose), wp.transform_get_rotation(X_c))
    child_rest_q = wp.mul(wp.transform_get_rotation(child_rest_pose), wp.transform_get_rotation(X_c))

    X_p = joint_X_p[joint]
    if parent >= 0:
        parent_pose = body_q[parent]
        parent_prev_pose = body_q_prev[parent]
        parent_rest_pose = body_q_rest[parent]
        parent_anchor = wp.transform_point(parent_pose, wp.transform_get_translation(X_p))
        parent_anchor_prev = wp.transform_point(parent_prev_pose, wp.transform_get_translation(X_p))
        parent_anchor_q = wp.mul(wp.transform_get_rotation(parent_pose), wp.transform_get_rotation(X_p))
        parent_anchor_q_prev = wp.mul(wp.transform_get_rotation(parent_prev_pose), wp.transform_get_rotation(X_p))
        parent_rest_q = wp.mul(wp.transform_get_rotation(parent_rest_pose), wp.transform_get_rotation(X_p))
    else:
        parent_anchor = wp.transform_get_translation(X_p)
        parent_anchor_prev = parent_anchor
        parent_anchor_q = wp.transform_get_rotation(X_p)
        parent_anchor_q_prev = parent_anchor_q
        parent_rest_q = parent_anchor_q

    c_start = joint_constraint_start[joint]
    qd_start = joint_qd_start[joint]
    lin_count = int(0)
    ang_count = int(0)
    if jt == JointType.REVOLUTE:
        ang_count = 1
    elif jt == JointType.PRISMATIC:
        lin_count = 1
    elif jt == JointType.D6:
        lin_count = joint_dof_dim[joint, 0]
        ang_count = joint_dof_dim[joint, 1]

    P_lin = wp.identity(3, float)
    P_ang = wp.identity(3, float)
    if jt == JointType.REVOLUTE or jt == JointType.PRISMATIC or jt == JointType.D6:
        P_lin, P_ang = _joint_projectors(jt, joint_axis, qd_start, lin_count, ang_count, parent_anchor_q)

    lin_lambda = wp.vec3(0.0)
    lin_C0 = wp.vec3(0.0)
    lin_alpha = float(0.0)
    linear_hard = joint_is_hard[c_start] == 1
    if linear_hard or joint_compliant_alm == 1:
        lin_lambda = joint_lambda_lin[joint]
        lin_C0 = joint_C0_lin[joint]
        lin_alpha = avbd_alpha

    ang_lambda = wp.vec3(0.0)
    ang_C0 = wp.vec3(0.0)
    ang_alpha = float(0.0)
    ang_hard = int(0)
    if jt != JointType.BALL:
        ang_hard = joint_is_hard[c_start + 1]
        if ang_hard == 1 or joint_compliant_alm == 1:
            ang_lambda = joint_lambda_ang[joint]
            ang_C0 = joint_C0_ang[joint]
            ang_alpha = avbd_alpha

    solve_weight_linear, solve_weight_angular, material_k_linear, material_k_angular, kd_linear, kd_angular = (
        _load_sparse_structural_coefficients(
            jt,
            c_start,
            joint_penalty_k,
            joint_rho,
            joint_material_k,
            joint_penalty_kd,
            joint_compliant_alm,
        )
    )
    k_linear, lin_lambda = _material_force_terms(
        solve_weight_linear, material_k_linear, lin_lambda, joint_compliant_alm
    )
    k_angular, ang_lambda = _material_force_terms(
        solve_weight_angular, material_k_angular, ang_lambda, joint_compliant_alm
    )

    if k_linear > 0.0 and (jt != JointType.D6 or lin_count < 3):
        _assemble_linear_joint_scalar(
            values_scalar,
            rhs_scalar,
            articulation_block_row_offsets,
            articulation_block_cols,
            body_start,
            parent_local,
            child_local,
            parent,
            child,
            parent_anchor,
            child_anchor,
            parent_anchor_prev,
            child_anchor_prev,
            body_q,
            body_com,
            k_linear,
            kd_linear,
            dt,
            P_lin,
            lin_lambda,
            lin_C0,
            lin_alpha,
        )

    kappa_cached = wp.vec3(0.0)
    J_world_cached = wp.mat33(0.0)
    has_angular_cache = False
    if k_angular > 0.0 and jt != JointType.BALL and (jt != JointType.D6 or ang_count < 3):
        sigma0 = wp.vec3(0.0)
        C_fric = wp.vec3(0.0)
        kappa_cached, J_world_cached = _assemble_angular_joint_scalar(
            values_scalar,
            rhs_scalar,
            articulation_block_row_offsets,
            articulation_block_cols,
            body_start,
            parent_local,
            child_local,
            parent,
            parent_anchor_q,
            child_anchor_q,
            parent_anchor_q_prev,
            child_anchor_q_prev,
            parent_rest_q,
            child_rest_q,
            k_angular,
            kd_angular,
            dt,
            P_ang,
            sigma0,
            C_fric,
            ang_lambda,
            ang_C0,
            ang_alpha,
        )
        has_angular_cache = True

    if jt == JointType.REVOLUTE:
        armature = joint_armature[qd_start]
        if armature > 0.0:
            child_inertia_q = wp.mul(wp.transform_get_rotation(body_inertia_q[child]), wp.transform_get_rotation(X_c))
            parent_inertia_q = wp.transform_get_rotation(X_p)
            if parent >= 0:
                parent_inertia_q = wp.mul(
                    wp.transform_get_rotation(body_inertia_q[parent]), wp.transform_get_rotation(X_p)
                )
            armature_kappa, armature_J_world = compute_kappa_and_jacobian(
                parent_anchor_q, child_anchor_q, parent_inertia_q, child_inertia_q
            )
            axis_local = wp.normalize(joint_axis[qd_start])
            armature_jacobian_world = armature_J_world * axis_local
            armature_hessian = armature / (dt * dt)
            armature_force = armature_hessian * wp.dot(armature_kappa, axis_local)
            _assemble_angular_axis_row_scalar(
                values_scalar,
                rhs_scalar,
                articulation_block_row_offsets,
                articulation_block_cols,
                body_start,
                parent_local,
                child_local,
                parent,
                armature_jacobian_world,
                armature_force,
                armature_hessian,
            )

    if jt == JointType.REVOLUTE:
        dof_idx = qd_start
        target_q_idx = joint_target_q_start[joint]
        has_drive = _drive_row_applies_force(joint_target_ke[dof_idx], joint_target_kd[dof_idx])
        has_limits = _limit_row_exists(joint_limit_ke[dof_idx], joint_limit_lower[dof_idx], joint_limit_upper[dof_idx])
        if has_drive or has_limits:
            axis_local = wp.normalize(joint_axis[dof_idx])
            kappa = kappa_cached
            J_world = J_world_cached
            if not has_angular_cache:
                kappa, J_world = compute_kappa_and_jacobian(
                    parent_anchor_q, child_anchor_q, parent_rest_q, child_rest_q
                )
            theta = wp.dot(kappa, axis_local)
            theta_abs = theta + joint_rest_angle[dof_idx]
            omega_parent = quat_velocity(parent_anchor_q, parent_anchor_q_prev, dt)
            omega_child = quat_velocity(child_anchor_q, child_anchor_q_prev, dt)
            dkappa_dt = compute_kappa_dot(J_world, omega_parent, omega_child)
            dtheta_dt = wp.dot(dkappa_dt, axis_local)

            force_scalar, hessian_scalar = _evaluate_sparse_drive_limit(
                dof_idx,
                target_q_idx,
                c_start + 2,
                theta_abs,
                dtheta_dt,
                joint_target_ke,
                joint_target_kd,
                joint_target_q,
                joint_target_vel,
                joint_limit_lower,
                joint_limit_upper,
                joint_limit_ke,
                joint_limit_kd,
                joint_penalty_k,
                joint_drive_limit_support,
                joint_drive_lambda,
                joint_limit_lambda,
                joint_compliant_alm,
                dt,
            )

            if hessian_scalar > 0.0:
                angular_jacobian_world = J_world * axis_local
                _assemble_angular_axis_row_scalar(
                    values_scalar,
                    rhs_scalar,
                    articulation_block_row_offsets,
                    articulation_block_cols,
                    body_start,
                    parent_local,
                    child_local,
                    parent,
                    angular_jacobian_world,
                    force_scalar,
                    hessian_scalar,
                )

    if jt == JointType.PRISMATIC:
        dof_idx = qd_start
        target_q_idx = joint_target_q_start[joint]
        has_drive = _drive_row_applies_force(joint_target_ke[dof_idx], joint_target_kd[dof_idx])
        has_limits = _limit_row_exists(joint_limit_ke[dof_idx], joint_limit_lower[dof_idx], joint_limit_upper[dof_idx])
        if has_drive or has_limits:
            axis_world = wp.normalize(wp.quat_rotate(parent_anchor_q, joint_axis[dof_idx]))
            C_vec = child_anchor - parent_anchor
            C_vec_prev = child_anchor_prev - parent_anchor_prev
            d_along = wp.dot(C_vec, axis_world)
            dd_dt = wp.dot((C_vec - C_vec_prev) / dt, axis_world)

            force_scalar, hessian_scalar = _evaluate_sparse_drive_limit(
                dof_idx,
                target_q_idx,
                c_start + 2,
                d_along,
                dd_dt,
                joint_target_ke,
                joint_target_kd,
                joint_target_q,
                joint_target_vel,
                joint_limit_lower,
                joint_limit_upper,
                joint_limit_ke,
                joint_limit_kd,
                joint_penalty_k,
                joint_drive_limit_support,
                joint_drive_lambda,
                joint_limit_lambda,
                joint_compliant_alm,
                dt,
            )

            if hessian_scalar > 0.0:
                _assemble_linear_axis_row_scalar(
                    values_scalar,
                    rhs_scalar,
                    articulation_block_row_offsets,
                    articulation_block_cols,
                    body_start,
                    parent_local,
                    child_local,
                    parent,
                    child,
                    parent_anchor,
                    child_anchor,
                    body_q,
                    body_com,
                    axis_world,
                    force_scalar,
                    hessian_scalar,
                )

    if jt == JointType.D6:
        C_vec = child_anchor - parent_anchor
        C_vec_prev = child_anchor_prev - parent_anchor_prev
        target_q_base = joint_target_q_start[joint]
        for li in range(3):
            if li < lin_count:
                dof_idx = qd_start + li
                target_q_idx = target_q_base + li
                has_drive = _drive_row_applies_force(joint_target_ke[dof_idx], joint_target_kd[dof_idx])
                has_limits = _limit_row_exists(
                    joint_limit_ke[dof_idx], joint_limit_lower[dof_idx], joint_limit_upper[dof_idx]
                )
                if has_drive or has_limits:
                    axis_world = wp.normalize(wp.quat_rotate(parent_anchor_q, joint_axis[dof_idx]))
                    d_along = wp.dot(C_vec, axis_world)
                    dd_dt = wp.dot((C_vec - C_vec_prev) / dt, axis_world)
                    force_scalar, hessian_scalar = _evaluate_sparse_drive_limit(
                        dof_idx,
                        target_q_idx,
                        c_start + 2 + li,
                        d_along,
                        dd_dt,
                        joint_target_ke,
                        joint_target_kd,
                        joint_target_q,
                        joint_target_vel,
                        joint_limit_lower,
                        joint_limit_upper,
                        joint_limit_ke,
                        joint_limit_kd,
                        joint_penalty_k,
                        joint_drive_limit_support,
                        joint_drive_lambda,
                        joint_limit_lambda,
                        joint_compliant_alm,
                        dt,
                    )

                    if hessian_scalar > 0.0:
                        _assemble_linear_axis_row_scalar(
                            values_scalar,
                            rhs_scalar,
                            articulation_block_row_offsets,
                            articulation_block_cols,
                            body_start,
                            parent_local,
                            child_local,
                            parent,
                            child,
                            parent_anchor,
                            child_anchor,
                            body_q,
                            body_com,
                            axis_world,
                            force_scalar,
                            hessian_scalar,
                        )

        if ang_count > 0:
            kappa = kappa_cached
            J_world = J_world_cached
            if not has_angular_cache:
                kappa, J_world = compute_kappa_and_jacobian(
                    parent_anchor_q, child_anchor_q, parent_rest_q, child_rest_q
                )
            omega_parent = quat_velocity(parent_anchor_q, parent_anchor_q_prev, dt)
            omega_child = quat_velocity(child_anchor_q, child_anchor_q_prev, dt)
            dkappa_dt = compute_kappa_dot(J_world, omega_parent, omega_child)
            for ai in range(3):
                if ai < ang_count:
                    dof_idx = qd_start + lin_count + ai
                    target_q_idx = target_q_base + lin_count + ai
                    has_drive = _drive_row_applies_force(joint_target_ke[dof_idx], joint_target_kd[dof_idx])
                    has_limits = _limit_row_exists(
                        joint_limit_ke[dof_idx], joint_limit_lower[dof_idx], joint_limit_upper[dof_idx]
                    )
                    if has_drive or has_limits:
                        axis_local = wp.normalize(joint_axis[dof_idx])
                        theta = wp.dot(kappa, axis_local)
                        theta_abs = theta + joint_rest_angle[dof_idx]
                        dtheta_dt = wp.dot(dkappa_dt, axis_local)
                        force_scalar, hessian_scalar = _evaluate_sparse_drive_limit(
                            dof_idx,
                            target_q_idx,
                            c_start + 2 + lin_count + ai,
                            theta_abs,
                            dtheta_dt,
                            joint_target_ke,
                            joint_target_kd,
                            joint_target_q,
                            joint_target_vel,
                            joint_limit_lower,
                            joint_limit_upper,
                            joint_limit_ke,
                            joint_limit_kd,
                            joint_penalty_k,
                            joint_drive_limit_support,
                            joint_drive_lambda,
                            joint_limit_lambda,
                            joint_compliant_alm,
                            dt,
                        )

                        if hessian_scalar > 0.0:
                            angular_jacobian_world = J_world * axis_local
                            _assemble_angular_axis_row_scalar(
                                values_scalar,
                                rhs_scalar,
                                articulation_block_row_offsets,
                                articulation_block_cols,
                                body_start,
                                parent_local,
                                child_local,
                                parent,
                                angular_jacobian_world,
                                force_scalar,
                                hessian_scalar,
                            )


@wp.func
def _cholesky66_scalar(values: wp.array[float], slot: int):
    for i in range(6):
        for j in range(6):
            if j <= i:
                value = _scalar_block_get(values, slot, i, j)
                for k in range(6):
                    if k < j:
                        value = value - _scalar_block_get(values, slot, i, k) * _scalar_block_get(values, slot, j, k)
                if i == j:
                    value = wp.sqrt(value)
                else:
                    value = value / _scalar_block_get(values, slot, j, j)
                _scalar_block_set(values, slot, i, j, value)
            else:
                _scalar_block_set(values, slot, i, j, 0.0)


@wp.func
def _right_solve_lower_transpose66_scalar_row(values: wp.array[float], slot: int, diag_slot: int, row: int):
    for col in range(6):
        value = _scalar_block_get(values, slot, row, col)
        for k in range(6):
            if k < col:
                value = value - _scalar_block_get(values, slot, row, k) * _scalar_block_get(values, diag_slot, col, k)
        value = value / _scalar_block_get(values, diag_slot, col, col)
        _scalar_block_set(values, slot, row, col, value)


@wp.func
def _lower_solve66_scalar(values: wp.array[float], diag_slot: int, delta_scalar: wp.array[float], row: int):
    for i in range(6):
        value = _scalar_vec_get(delta_scalar, row, i)
        for j in range(6):
            if j < i:
                value = value - _scalar_block_get(values, diag_slot, i, j) * _scalar_vec_get(delta_scalar, row, j)
        value = value / _scalar_block_get(values, diag_slot, i, i)
        _scalar_vec_set(delta_scalar, row, i, value)


@wp.func
def _upper_solve66_scalar(values: wp.array[float], diag_slot: int, delta_scalar: wp.array[float], row: int):
    for ii in range(6):
        i = 5 - ii
        value = _scalar_vec_get(delta_scalar, row, i)
        for jj in range(6):
            j = 5 - jj
            if j > i:
                value = value - _scalar_block_get(values, diag_slot, j, i) * _scalar_vec_get(delta_scalar, row, j)
        value = value / _scalar_block_get(values, diag_slot, i, i)
        _scalar_vec_set(delta_scalar, row, i, value)


@wp.kernel
def solve_articulation_sparse_block32_scalar(
    articulation_body_offsets: wp.array[wp.int32],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    articulation_block_col_offsets: wp.array[wp.int32],
    articulation_block_col_rows: wp.array[wp.int32],
    articulation_block_col_slots: wp.array[wp.int32],
    articulation_factor_update_offsets: wp.array[wp.int32],
    articulation_factor_update_dst_slots: wp.array[wp.int32],
    articulation_factor_update_left_slots: wp.array[wp.int32],
    articulation_factor_update_right_slots: wp.array[wp.int32],
    articulation_diag_slots: wp.array[wp.int32],
    values_scalar: wp.array[float],
    rhs_scalar: wp.array[float],
    delta_scalar: wp.array[float],
):
    thread_id = wp.tid()
    thread_count = wp.block_dim()
    articulation_id = thread_id // thread_count
    lane = thread_id - articulation_id * thread_count
    warp = lane // _SPARSE_CTA_THREADS
    warp_lane = lane - warp * _SPARSE_CTA_THREADS
    warp_count = thread_count // _SPARSE_CTA_THREADS

    body_start = articulation_body_offsets[articulation_id]
    body_end = articulation_body_offsets[articulation_id + 1]
    body_count = body_end - body_start

    for local_k in range(body_count):
        row_k = body_start + local_k
        diag_slot = articulation_diag_slots[row_k]

        if lane == 0:
            _cholesky66_scalar(values_scalar, diag_slot)

        _block_sync()

        col_begin = articulation_block_col_offsets[row_k]
        col_end = articulation_block_col_offsets[row_k + 1]
        for col_entry in range(col_begin + warp, col_end, warp_count):
            slot_ik = articulation_block_col_slots[col_entry]
            if warp_lane < _SPARSE_BLOCK_DIM:
                _right_solve_lower_transpose66_scalar_row(values_scalar, slot_ik, diag_slot, warp_lane)

        _block_sync()

        factor_update_begin = articulation_factor_update_offsets[row_k]
        factor_update_end = articulation_factor_update_offsets[row_k + 1]
        for factor_update_entry in range(factor_update_begin + warp, factor_update_end, warp_count):
            dst_slot = articulation_factor_update_dst_slots[factor_update_entry]
            left_slot = articulation_factor_update_left_slots[factor_update_entry]
            right_slot = articulation_factor_update_right_slots[factor_update_entry]
            elem = warp_lane
            for _elem_pass in range(2):
                if elem < _SPARSE_BLOCK_SIZE:
                    block_row = elem // _SPARSE_BLOCK_DIM
                    block_col = elem - block_row * _SPARSE_BLOCK_DIM
                    value = _scalar_block_get(values_scalar, dst_slot, block_row, block_col)
                    accum = float(0.0)
                    for p in range(6):
                        accum = accum + _scalar_block_get(values_scalar, left_slot, block_row, p) * (
                            _scalar_block_get(values_scalar, right_slot, block_col, p)
                        )
                    _scalar_block_set(values_scalar, dst_slot, block_row, block_col, value - accum)
                elem = elem + _SPARSE_CTA_THREADS

        _block_sync()

    for local_i in range(body_count):
        row_i = body_start + local_i
        if warp == 0 and warp_lane < _SPARSE_BLOCK_DIM:
            value = _scalar_vec_get(rhs_scalar, row_i, warp_lane)
            row_begin = articulation_block_row_offsets[row_i]
            row_end = articulation_block_row_offsets[row_i + 1]
            for row_entry in range(row_begin, row_end):
                local_j = articulation_block_cols[row_entry]
                if local_j < local_i:
                    accum = float(0.0)
                    for col in range(6):
                        accum = accum + _scalar_block_get(values_scalar, row_entry, warp_lane, col) * _scalar_vec_get(
                            delta_scalar, body_start + local_j, col
                        )
                    value = value - accum
            _scalar_vec_set(delta_scalar, row_i, warp_lane, value)

        if warp == 0:
            _warp_sync()

        if lane == 0:
            _lower_solve66_scalar(values_scalar, articulation_diag_slots[row_i], delta_scalar, row_i)

        if warp == 0:
            _warp_sync()

    for local_ii in range(body_count):
        local_i = body_count - 1 - local_ii
        row_i = body_start + local_i
        if warp == 0 and warp_lane < _SPARSE_BLOCK_DIM:
            value = _scalar_vec_get(delta_scalar, row_i, warp_lane)
            col_begin = articulation_block_col_offsets[row_i]
            col_end = articulation_block_col_offsets[row_i + 1]
            for col_entry in range(col_begin, col_end):
                local_j = articulation_block_col_rows[col_entry]
                slot_ji = articulation_block_col_slots[col_entry]
                accum = float(0.0)
                for col in range(6):
                    accum = accum + _scalar_block_get(values_scalar, slot_ji, col, warp_lane) * _scalar_vec_get(
                        delta_scalar, body_start + local_j, col
                    )
                value = value - accum
            _scalar_vec_set(delta_scalar, row_i, warp_lane, value)

        if warp == 0:
            _warp_sync()

        if lane == 0:
            _upper_solve66_scalar(values_scalar, articulation_diag_slots[row_i], delta_scalar, row_i)

        if warp == 0:
            _warp_sync()


@wp.kernel
def apply_articulation_sparse_delta_scalar(
    articulation_bodies: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_inv_mass: wp.array[float],
    body_com: wp.array[wp.vec3],
    update_relaxation: float,
    delta_scalar: wp.array[float],
    body_q_new: wp.array[wp.transform],
):
    local_body = wp.tid()
    dx = vec6f(
        _scalar_vec_get(delta_scalar, local_body, 0),
        _scalar_vec_get(delta_scalar, local_body, 1),
        _scalar_vec_get(delta_scalar, local_body, 2),
        _scalar_vec_get(delta_scalar, local_body, 3),
        _scalar_vec_get(delta_scalar, local_body, 4),
        _scalar_vec_get(delta_scalar, local_body, 5),
    )
    _apply_sparse_delta_value_to_body(
        local_body,
        articulation_bodies,
        body_q,
        body_inv_mass,
        body_com,
        update_relaxation,
        dx,
        body_q_new,
    )


@wp.kernel
def solve_articulation_sparse_serial(
    dt: float,
    articulation_body_offsets: wp.array[wp.int32],
    articulation_joint_offsets: wp.array[wp.int32],
    articulation_bodies: wp.array[wp.int32],
    articulation_joints: wp.array[wp.int32],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    articulation_diag_slots: wp.array[wp.int32],
    body_articulation_local: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_q_rest: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_inv_mass: wp.array[float],
    body_com: wp.array[wp.vec3],
    body_inertia: wp.array[wp.mat33],
    body_inertia_q: wp.array[wp.transform],
    body_forces: wp.array[wp.vec3],
    body_torques: wp.array[wp.vec3],
    body_hessian_ll: wp.array[wp.mat33],
    body_hessian_al: wp.array[wp.mat33],
    body_hessian_aa: wp.array[wp.mat33],
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
    joint_dof_dim: wp.array2d[int],
    joint_rest_angle: wp.array[float],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_armature: wp.array[float],
    joint_target_q: wp.array[float],
    joint_target_vel: wp.array[float],
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
    avbd_alpha: float,
    joint_compliant_alm: int,
    update_relaxation: float,
    values: wp.array[mat66f],
    rhs: wp.array[vec6f],
    delta: wp.array[vec6f],
    body_q_new: wp.array[wp.transform],
):
    articulation_id = wp.tid()

    body_start = articulation_body_offsets[articulation_id]
    body_end = articulation_body_offsets[articulation_id + 1]
    joint_start = articulation_joint_offsets[articulation_id]
    joint_end = articulation_joint_offsets[articulation_id + 1]

    for local_body in range(body_start, body_end):
        diag, rhs_value = _body_diagonal_contribution(
            dt,
            local_body,
            articulation_bodies,
            body_q,
            body_mass,
            body_inv_mass,
            body_com,
            body_inertia,
            body_inertia_q,
            body_forces,
            body_torques,
            body_hessian_ll,
            body_hessian_al,
            body_hessian_aa,
        )
        rhs[local_body] = rhs_value
        diag_slot = articulation_diag_slots[local_body]
        values[diag_slot] = values[diag_slot] + diag

    for joint_cursor in range(joint_start, joint_end):
        joint = articulation_joints[joint_cursor]
        if not joint_enabled[joint]:
            continue

        jt = joint_type[joint]
        if (
            jt != JointType.ROD
            and jt != JointType.BALL
            and jt != JointType.FIXED
            and jt != JointType.REVOLUTE
            and jt != JointType.PRISMATIC
            and jt != JointType.D6
        ):
            continue

        child = joint_child[joint]
        parent = joint_parent[joint]
        if child < 0:
            continue

        child_local = body_articulation_local[child]
        parent_local = -1
        if parent >= 0:
            parent_local = body_articulation_local[parent]

        if jt == JointType.ROD:
            child_force, child_torque, child_H_ll, child_H_al, child_H_aa = _evaluate_sparse_rod_body(
                child,
                joint,
                body_q,
                body_q_prev,
                body_com,
                joint_parent,
                joint_child,
                joint_X_p,
                joint_X_c,
                joint_rod_rest_kb_local,
                joint_rod_rest_twist,
                joint_constraint_start,
                joint_penalty_k,
                joint_rho,
                joint_material_k,
                joint_penalty_kd,
                joint_sigma_start,
                joint_C_fric,
                joint_lambda_lin,
                joint_lambda_ang,
                joint_C0_lin,
                joint_C0_ang,
                joint_is_hard,
                avbd_alpha,
                joint_compliant_alm,
                dt,
            )
            _add_rod_body_contribution(
                values,
                rhs,
                articulation_block_row_offsets,
                articulation_block_cols,
                body_start,
                child_local,
                child_force,
                child_torque,
                child_H_ll,
                child_H_al,
                child_H_aa,
            )
            if parent >= 0:
                parent_force, parent_torque, parent_H_ll, parent_H_al, parent_H_aa = _evaluate_sparse_rod_body(
                    parent,
                    joint,
                    body_q,
                    body_q_prev,
                    body_com,
                    joint_parent,
                    joint_child,
                    joint_X_p,
                    joint_X_c,
                    joint_rod_rest_kb_local,
                    joint_rod_rest_twist,
                    joint_constraint_start,
                    joint_penalty_k,
                    joint_rho,
                    joint_material_k,
                    joint_penalty_kd,
                    joint_sigma_start,
                    joint_C_fric,
                    joint_lambda_lin,
                    joint_lambda_ang,
                    joint_C0_lin,
                    joint_C0_ang,
                    joint_is_hard,
                    avbd_alpha,
                    joint_compliant_alm,
                    dt,
                )
                _add_rod_body_contribution(
                    values,
                    rhs,
                    articulation_block_row_offsets,
                    articulation_block_cols,
                    body_start,
                    parent_local,
                    parent_force,
                    parent_torque,
                    parent_H_ll,
                    parent_H_al,
                    parent_H_aa,
                )
                cross = _rod_cross_hessian(
                    dt,
                    joint,
                    parent,
                    child,
                    body_q,
                    body_com,
                    joint_X_p,
                    joint_X_c,
                    joint_constraint_start,
                    joint_penalty_k,
                    joint_rho,
                    joint_material_k,
                    joint_penalty_kd,
                    joint_C_fric,
                    joint_is_hard,
                    joint_compliant_alm,
                )
                _add_rod_cross_block(
                    values,
                    articulation_block_row_offsets,
                    articulation_block_cols,
                    body_start,
                    parent_local,
                    child_local,
                    cross,
                )
            continue

        child_pose = body_q[child]
        child_prev_pose = body_q_prev[child]
        child_rest_pose = body_q_rest[child]
        X_c = joint_X_c[joint]
        child_anchor = wp.transform_point(child_pose, wp.transform_get_translation(X_c))
        child_anchor_prev = wp.transform_point(child_prev_pose, wp.transform_get_translation(X_c))
        child_anchor_q = wp.mul(wp.transform_get_rotation(child_pose), wp.transform_get_rotation(X_c))
        child_anchor_q_prev = wp.mul(wp.transform_get_rotation(child_prev_pose), wp.transform_get_rotation(X_c))
        child_rest_q = wp.mul(wp.transform_get_rotation(child_rest_pose), wp.transform_get_rotation(X_c))

        X_p = joint_X_p[joint]
        if parent >= 0:
            parent_pose = body_q[parent]
            parent_prev_pose = body_q_prev[parent]
            parent_rest_pose = body_q_rest[parent]
            parent_anchor = wp.transform_point(parent_pose, wp.transform_get_translation(X_p))
            parent_anchor_prev = wp.transform_point(parent_prev_pose, wp.transform_get_translation(X_p))
            parent_anchor_q = wp.mul(wp.transform_get_rotation(parent_pose), wp.transform_get_rotation(X_p))
            parent_anchor_q_prev = wp.mul(wp.transform_get_rotation(parent_prev_pose), wp.transform_get_rotation(X_p))
            parent_rest_q = wp.mul(wp.transform_get_rotation(parent_rest_pose), wp.transform_get_rotation(X_p))
        else:
            parent_anchor = wp.transform_get_translation(X_p)
            parent_anchor_prev = parent_anchor
            parent_anchor_q = wp.transform_get_rotation(X_p)
            parent_anchor_q_prev = parent_anchor_q
            parent_rest_q = parent_anchor_q

        c_start = joint_constraint_start[joint]
        qd_start = joint_qd_start[joint]
        lin_count = int(0)
        ang_count = int(0)
        if jt == JointType.REVOLUTE:
            ang_count = 1
        elif jt == JointType.PRISMATIC:
            lin_count = 1
        elif jt == JointType.D6:
            lin_count = joint_dof_dim[joint, 0]
            ang_count = joint_dof_dim[joint, 1]

        P_lin = wp.identity(3, float)
        P_ang = wp.identity(3, float)
        if jt == JointType.REVOLUTE or jt == JointType.PRISMATIC or jt == JointType.D6:
            P_lin, P_ang = _joint_projectors(jt, joint_axis, qd_start, lin_count, ang_count, parent_anchor_q)

        lin_lambda = wp.vec3(0.0)
        lin_C0 = wp.vec3(0.0)
        lin_alpha = float(0.0)
        linear_hard = joint_is_hard[c_start] == 1
        if linear_hard or joint_compliant_alm == 1:
            lin_lambda = joint_lambda_lin[joint]
            lin_C0 = joint_C0_lin[joint]
            lin_alpha = avbd_alpha

        ang_lambda = wp.vec3(0.0)
        ang_C0 = wp.vec3(0.0)
        ang_alpha = float(0.0)
        ang_hard = int(0)
        if jt != JointType.BALL:
            ang_hard = joint_is_hard[c_start + 1]
            if ang_hard == 1 or joint_compliant_alm == 1:
                ang_lambda = joint_lambda_ang[joint]
                ang_C0 = joint_C0_ang[joint]
                ang_alpha = avbd_alpha

        solve_weight_linear, solve_weight_angular, material_k_linear, material_k_angular, kd_linear, kd_angular = (
            _load_sparse_structural_coefficients(
                jt,
                c_start,
                joint_penalty_k,
                joint_rho,
                joint_material_k,
                joint_penalty_kd,
                joint_compliant_alm,
            )
        )
        k_linear, lin_lambda = _material_force_terms(
            solve_weight_linear, material_k_linear, lin_lambda, joint_compliant_alm
        )
        k_angular, ang_lambda = _material_force_terms(
            solve_weight_angular, material_k_angular, ang_lambda, joint_compliant_alm
        )

        if k_linear > 0.0 and (jt != JointType.D6 or lin_count < 3):
            _assemble_linear_joint(
                values,
                rhs,
                articulation_block_row_offsets,
                articulation_block_cols,
                body_start,
                parent_local,
                child_local,
                parent,
                child,
                parent_anchor,
                child_anchor,
                parent_anchor_prev,
                child_anchor_prev,
                body_q,
                body_com,
                k_linear,
                kd_linear,
                dt,
                P_lin,
                lin_lambda,
                lin_C0,
                lin_alpha,
            )

        kappa_cached = wp.vec3(0.0)
        J_world_cached = wp.mat33(0.0)
        has_angular_cache = False
        if k_angular > 0.0 and jt != JointType.BALL and (jt != JointType.D6 or ang_count < 3):
            sigma0 = wp.vec3(0.0)
            C_fric = wp.vec3(0.0)
            kappa_cached, J_world_cached = _assemble_angular_joint(
                values,
                rhs,
                articulation_block_row_offsets,
                articulation_block_cols,
                body_start,
                parent_local,
                child_local,
                parent,
                child,
                parent_anchor_q,
                child_anchor_q,
                parent_anchor_q_prev,
                child_anchor_q_prev,
                parent_rest_q,
                child_rest_q,
                k_angular,
                kd_angular,
                dt,
                P_ang,
                sigma0,
                C_fric,
                ang_lambda,
                ang_C0,
                ang_alpha,
            )
            has_angular_cache = True

        if jt == JointType.REVOLUTE:
            armature = joint_armature[qd_start]
            if armature > 0.0:
                child_inertia_q = wp.mul(
                    wp.transform_get_rotation(body_inertia_q[child]), wp.transform_get_rotation(X_c)
                )
                parent_inertia_q = wp.transform_get_rotation(X_p)
                if parent >= 0:
                    parent_inertia_q = wp.mul(
                        wp.transform_get_rotation(body_inertia_q[parent]), wp.transform_get_rotation(X_p)
                    )
                armature_kappa, armature_J_world = compute_kappa_and_jacobian(
                    parent_anchor_q, child_anchor_q, parent_inertia_q, child_inertia_q
                )
                axis_local = wp.normalize(joint_axis[qd_start])
                armature_jacobian_world = armature_J_world * axis_local
                armature_hessian = armature / (dt * dt)
                armature_force = armature_hessian * wp.dot(armature_kappa, axis_local)
                _assemble_angular_axis_row(
                    values,
                    rhs,
                    articulation_block_row_offsets,
                    articulation_block_cols,
                    body_start,
                    parent_local,
                    child_local,
                    parent,
                    armature_jacobian_world,
                    armature_force,
                    armature_hessian,
                )

        if jt == JointType.REVOLUTE:
            dof_idx = qd_start
            target_q_idx = joint_target_q_start[joint]
            has_drive = _drive_row_applies_force(joint_target_ke[dof_idx], joint_target_kd[dof_idx])
            has_limits = _limit_row_exists(
                joint_limit_ke[dof_idx], joint_limit_lower[dof_idx], joint_limit_upper[dof_idx]
            )
            if has_drive or has_limits:
                axis_local = wp.normalize(joint_axis[dof_idx])
                kappa = kappa_cached
                J_world = J_world_cached
                if not has_angular_cache:
                    kappa, J_world = compute_kappa_and_jacobian(
                        parent_anchor_q, child_anchor_q, parent_rest_q, child_rest_q
                    )
                theta = wp.dot(kappa, axis_local)
                theta_abs = theta + joint_rest_angle[dof_idx]
                omega_parent = quat_velocity(parent_anchor_q, parent_anchor_q_prev, dt)
                omega_child = quat_velocity(child_anchor_q, child_anchor_q_prev, dt)
                dkappa_dt = compute_kappa_dot(J_world, omega_parent, omega_child)
                dtheta_dt = wp.dot(dkappa_dt, axis_local)

                force_scalar, hessian_scalar = _evaluate_sparse_drive_limit(
                    dof_idx,
                    target_q_idx,
                    c_start + 2,
                    theta_abs,
                    dtheta_dt,
                    joint_target_ke,
                    joint_target_kd,
                    joint_target_q,
                    joint_target_vel,
                    joint_limit_lower,
                    joint_limit_upper,
                    joint_limit_ke,
                    joint_limit_kd,
                    joint_penalty_k,
                    joint_drive_limit_support,
                    joint_drive_lambda,
                    joint_limit_lambda,
                    joint_compliant_alm,
                    dt,
                )

                if hessian_scalar > 0.0:
                    angular_jacobian_world = J_world * axis_local
                    _assemble_angular_axis_row(
                        values,
                        rhs,
                        articulation_block_row_offsets,
                        articulation_block_cols,
                        body_start,
                        parent_local,
                        child_local,
                        parent,
                        angular_jacobian_world,
                        force_scalar,
                        hessian_scalar,
                    )

        if jt == JointType.PRISMATIC:
            dof_idx = qd_start
            target_q_idx = joint_target_q_start[joint]
            has_drive = _drive_row_applies_force(joint_target_ke[dof_idx], joint_target_kd[dof_idx])
            has_limits = _limit_row_exists(
                joint_limit_ke[dof_idx], joint_limit_lower[dof_idx], joint_limit_upper[dof_idx]
            )
            if has_drive or has_limits:
                axis_world = wp.normalize(wp.quat_rotate(parent_anchor_q, joint_axis[dof_idx]))
                C_vec = child_anchor - parent_anchor
                C_vec_prev = child_anchor_prev - parent_anchor_prev
                d_along = wp.dot(C_vec, axis_world)
                dd_dt = wp.dot((C_vec - C_vec_prev) / dt, axis_world)

                force_scalar, hessian_scalar = _evaluate_sparse_drive_limit(
                    dof_idx,
                    target_q_idx,
                    c_start + 2,
                    d_along,
                    dd_dt,
                    joint_target_ke,
                    joint_target_kd,
                    joint_target_q,
                    joint_target_vel,
                    joint_limit_lower,
                    joint_limit_upper,
                    joint_limit_ke,
                    joint_limit_kd,
                    joint_penalty_k,
                    joint_drive_limit_support,
                    joint_drive_lambda,
                    joint_limit_lambda,
                    joint_compliant_alm,
                    dt,
                )

                if hessian_scalar > 0.0:
                    _assemble_linear_axis_row(
                        values,
                        rhs,
                        articulation_block_row_offsets,
                        articulation_block_cols,
                        body_start,
                        parent_local,
                        child_local,
                        parent,
                        child,
                        parent_anchor,
                        child_anchor,
                        body_q,
                        body_com,
                        axis_world,
                        force_scalar,
                        hessian_scalar,
                    )

        if jt == JointType.D6:
            C_vec = child_anchor - parent_anchor
            C_vec_prev = child_anchor_prev - parent_anchor_prev
            target_q_base = joint_target_q_start[joint]
            for li in range(3):
                if li < lin_count:
                    dof_idx = qd_start + li
                    target_q_idx = target_q_base + li
                    has_drive = _drive_row_applies_force(joint_target_ke[dof_idx], joint_target_kd[dof_idx])
                    has_limits = _limit_row_exists(
                        joint_limit_ke[dof_idx], joint_limit_lower[dof_idx], joint_limit_upper[dof_idx]
                    )
                    if has_drive or has_limits:
                        axis_world = wp.normalize(wp.quat_rotate(parent_anchor_q, joint_axis[dof_idx]))
                        d_along = wp.dot(C_vec, axis_world)
                        dd_dt = wp.dot((C_vec - C_vec_prev) / dt, axis_world)
                        force_scalar, hessian_scalar = _evaluate_sparse_drive_limit(
                            dof_idx,
                            target_q_idx,
                            c_start + 2 + li,
                            d_along,
                            dd_dt,
                            joint_target_ke,
                            joint_target_kd,
                            joint_target_q,
                            joint_target_vel,
                            joint_limit_lower,
                            joint_limit_upper,
                            joint_limit_ke,
                            joint_limit_kd,
                            joint_penalty_k,
                            joint_drive_limit_support,
                            joint_drive_lambda,
                            joint_limit_lambda,
                            joint_compliant_alm,
                            dt,
                        )

                        if hessian_scalar > 0.0:
                            _assemble_linear_axis_row(
                                values,
                                rhs,
                                articulation_block_row_offsets,
                                articulation_block_cols,
                                body_start,
                                parent_local,
                                child_local,
                                parent,
                                child,
                                parent_anchor,
                                child_anchor,
                                body_q,
                                body_com,
                                axis_world,
                                force_scalar,
                                hessian_scalar,
                            )

            if ang_count > 0:
                kappa = kappa_cached
                J_world = J_world_cached
                if not has_angular_cache:
                    kappa, J_world = compute_kappa_and_jacobian(
                        parent_anchor_q, child_anchor_q, parent_rest_q, child_rest_q
                    )
                omega_parent = quat_velocity(parent_anchor_q, parent_anchor_q_prev, dt)
                omega_child = quat_velocity(child_anchor_q, child_anchor_q_prev, dt)
                dkappa_dt = compute_kappa_dot(J_world, omega_parent, omega_child)
                for ai in range(3):
                    if ai < ang_count:
                        dof_idx = qd_start + lin_count + ai
                        target_q_idx = target_q_base + lin_count + ai
                        has_drive = _drive_row_applies_force(joint_target_ke[dof_idx], joint_target_kd[dof_idx])
                        has_limits = _limit_row_exists(
                            joint_limit_ke[dof_idx], joint_limit_lower[dof_idx], joint_limit_upper[dof_idx]
                        )
                        if has_drive or has_limits:
                            axis_local = wp.normalize(joint_axis[dof_idx])
                            theta = wp.dot(kappa, axis_local)
                            theta_abs = theta + joint_rest_angle[dof_idx]
                            dtheta_dt = wp.dot(dkappa_dt, axis_local)
                            force_scalar, hessian_scalar = _evaluate_sparse_drive_limit(
                                dof_idx,
                                target_q_idx,
                                c_start + 2 + lin_count + ai,
                                theta_abs,
                                dtheta_dt,
                                joint_target_ke,
                                joint_target_kd,
                                joint_target_q,
                                joint_target_vel,
                                joint_limit_lower,
                                joint_limit_upper,
                                joint_limit_ke,
                                joint_limit_kd,
                                joint_penalty_k,
                                joint_drive_limit_support,
                                joint_drive_lambda,
                                joint_limit_lambda,
                                joint_compliant_alm,
                                dt,
                            )

                            if hessian_scalar > 0.0:
                                angular_jacobian_world = J_world * axis_local
                                _assemble_angular_axis_row(
                                    values,
                                    rhs,
                                    articulation_block_row_offsets,
                                    articulation_block_cols,
                                    body_start,
                                    parent_local,
                                    child_local,
                                    parent,
                                    angular_jacobian_world,
                                    force_scalar,
                                    hessian_scalar,
                                )

    body_count = body_end - body_start

    for local_k in range(body_count):
        row_k = body_start + local_k
        diag_slot = articulation_diag_slots[row_k]
        Akk = values[diag_slot]

        for local_s in range(local_k):
            slot_ks = _find_block_slot(
                articulation_block_row_offsets, articulation_block_cols, body_start, local_k, local_s
            )
            if slot_ks >= 0:
                Lks = values[slot_ks]
                Akk = Akk - Lks * wp.transpose(Lks)

        Lkk = _cholesky66_serial(Akk)
        values[diag_slot] = Lkk

        for local_i in range(local_k + 1, body_count):
            slot_ik = _find_block_slot(
                articulation_block_row_offsets, articulation_block_cols, body_start, local_i, local_k
            )
            if slot_ik >= 0:
                Aik = values[slot_ik]
                for local_s in range(local_k):
                    slot_is = _find_block_slot(
                        articulation_block_row_offsets, articulation_block_cols, body_start, local_i, local_s
                    )
                    slot_ks = _find_block_slot(
                        articulation_block_row_offsets, articulation_block_cols, body_start, local_k, local_s
                    )
                    if slot_is >= 0 and slot_ks >= 0:
                        Aik = Aik - values[slot_is] * wp.transpose(values[slot_ks])
                values[slot_ik] = _solve_right_lower_transpose66(Aik, Lkk)

    for local_i in range(body_count):
        row_i = body_start + local_i
        accum = rhs[row_i]
        for local_j in range(local_i):
            slot_ij = _find_block_slot(
                articulation_block_row_offsets, articulation_block_cols, body_start, local_i, local_j
            )
            if slot_ij >= 0:
                accum = accum - values[slot_ij] * delta[body_start + local_j]
        Lii = values[articulation_diag_slots[row_i]]
        delta[row_i] = _solve_lower66(Lii, accum)

    for local_ii in range(body_count):
        local_i = body_count - 1 - local_ii
        row_i = body_start + local_i
        accum = delta[row_i]
        for local_j in range(local_i + 1, body_count):
            slot_ji = _find_block_slot(
                articulation_block_row_offsets, articulation_block_cols, body_start, local_j, local_i
            )
            if slot_ji >= 0:
                accum = accum - wp.transpose(values[slot_ji]) * delta[body_start + local_j]
        Lii = values[articulation_diag_slots[row_i]]
        delta[row_i] = _solve_upper66_from_lower(Lii, accum)

    for local_body in range(body_start, body_end):
        _apply_sparse_delta_to_body(
            local_body,
            articulation_bodies,
            body_q,
            body_inv_mass,
            body_com,
            update_relaxation,
            delta,
            body_q_new,
        )
