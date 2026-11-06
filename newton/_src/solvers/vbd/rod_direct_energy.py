# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Trial poses and the incremental potential evaluated by the direct rod solve's line search."""

from __future__ import annotations

import warp as wp

from .rod_direct import (
    apply_increment,
    inertial_terms,
    mat66d,
    rod_joint_energy,
    rod_material,
    vec6d,
)


@wp.func
def _pose_or_world(body_q: wp.array[wp.transform], b: int) -> wp.transform:
    if b < 0:
        return wp.transform_identity()
    return body_q[b]


@wp.func
def _com_or_zero(body_com: wp.array[wp.vec3], b: int) -> wp.vec3:
    if b < 0:
        return wp.vec3(0.0)
    return body_com[b]


@wp.func
def _trial_pose(
    node: int,
    alpha: wp.float64,
    node_body: wp.array[wp.int32],
    body_com: wp.array[wp.vec3],
    DEL: wp.array[vec6d],
    T0: wp.array[wp.transform],
) -> wp.transform:
    return apply_increment(T0[node], body_com[node_body[node]], DEL[node], alpha)


@wp.func
def _trial_body_pose(
    b: int,
    alpha: wp.float64,
    body_node: wp.array[wp.int32],
    node_body: wp.array[wp.int32],
    body_com: wp.array[wp.vec3],
    DEL: wp.array[vec6d],
    T0: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
) -> wp.transform:
    if b < 0:
        return wp.transform_identity()
    k = body_node[b]
    if k < 0:
        return body_q[b]
    return _trial_pose(k, alpha, node_body, body_com, DEL, T0)


@wp.func
def _rod_potential(
    s: int,
    e: int,
    alpha: wp.float64,
    node_body: wp.array[wp.int32],
    node_parent: wp.array[wp.int32],
    node_joint: wp.array[wp.int32],
    node_side_parent: wp.array[wp.int32],
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
    DEL: wp.array[vec6d],
    HM: wp.array[mat66d],
    FM: wp.array[vec6d],
    T0: wp.array[wp.transform],
    bnd_start: wp.array[wp.int32],
    bnd_joint: wp.array[wp.int32],
    body_node: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    dt: float,
) -> wp.float64:
    """Inertia + exact in-rod joint energies + the quadratic model of everything else, at step fraction alpha.

    A tree joint's two bodies are this node and its parent node; their trial poses come from T0 and DEL.
    """
    phi = wp.float64(0.0)
    for k in range(s, e):
        b = node_body[k]
        T = _trial_pose(k, alpha, node_body, body_com, DEL, T0)
        _f, _H, E = inertial_terms(T, body_inertia_q[b], body_com[b], body_mass[b], body_inertia[b], dt)
        d = alpha * DEL[k]
        phi = phi + E - wp.dot(FM[k], d) + wp.float64(0.5) * wp.dot(d, HM[k] * d)
    for k in range(s, e):
        j = node_joint[k]
        if j < 0:
            continue
        pb = joint_parent[j]
        cb = joint_child[j]
        T_self = _trial_pose(k, alpha, node_body, body_com, DEL, T0)
        T_up = _trial_pose(node_parent[k], alpha, node_body, body_com, DEL, T0)
        Tp = T_up
        Tc = T_self
        if node_side_parent[k] == 1:
            Tp = T_self
            Tc = T_up
        k_lin, k_ang, c_lin, c_ang = rod_material(j, joint_qd_start, joint_target_ke, joint_target_kd, dt)
        phi = phi + rod_joint_energy(
            Tp,
            Tc,
            body_q_prev[pb],
            body_q_prev[cb],
            joint_X_p[j],
            joint_X_c[j],
            joint_rod_rest_kb_local[j],
            joint_rod_rest_twist[j],
            k_lin,
            k_ang,
            c_lin,
            c_ang,
        )
    for k in range(s, e):  # one-sided rod joints, exactly; a loop-closing joint is counted from its lower node
        b = node_body[k]
        for q in range(bnd_start[k], bnd_start[k + 1]):
            j = bnd_joint[q]
            pb = joint_parent[j]
            cb = joint_child[j]
            other = pb
            if pb == b:
                other = cb
            if other >= 0 and body_node[other] >= 0 and body_node[other] < k:
                continue
            k_lin, k_ang, c_lin, c_ang = rod_material(j, joint_qd_start, joint_target_ke, joint_target_kd, dt)
            Tp = _trial_body_pose(pb, alpha, body_node, node_body, body_com, DEL, T0, body_q)
            Tc = _trial_body_pose(cb, alpha, body_node, node_body, body_com, DEL, T0, body_q)
            phi = phi + rod_joint_energy(
                Tp,
                Tc,
                _pose_or_world(body_q_prev, pb),
                _pose_or_world(body_q_prev, cb),
                joint_X_p[j],
                joint_X_c[j],
                joint_rod_rest_kb_local[j],
                joint_rod_rest_twist[j],
                k_lin,
                k_ang,
                c_lin,
                c_ang,
            )
    return phi
