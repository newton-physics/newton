# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""VBD mimic constraints using the local body Hessians and force multipliers."""

import warp as wp

from ...sim import JointType, Model
from ...sim.joint_mimic import eval_joint_mimic_coordinate
from .rigid_vbd_kernels import ldlt6_solve

wp.set_module_options({"enable_backward": False})

_Mat46 = wp.types.matrix(shape=(4, 6), dtype=wp.float32)


@wp.struct
class _JointData:
    body_com: wp.array[wp.vec3]
    joint_type: wp.array[int]
    enabled: wp.array[bool]
    parent: wp.array[int]
    child: wp.array[int]
    X_p: wp.array[wp.transform]
    X_c: wp.array[wp.transform]
    qd_start: wp.array[int]
    dof_dim: wp.array2d[int]
    axis: wp.array[wp.vec3]
    reference: wp.array[int]
    coeffs: wp.array[wp.vec2]


@wp.func
def _evaluate_row(data: _JointData, follower: int, component: int, body_q: wp.array[wp.transform]):
    reference = data.reference[follower]
    q_f, g_fp, g_fc = eval_joint_mimic_coordinate(
        follower,
        component,
        body_q,
        data.body_com,
        data.joint_type,
        data.parent,
        data.child,
        data.X_p,
        data.X_c,
        data.qd_start,
        data.dof_dim,
        data.axis,
    )
    q_r, g_rp, g_rc = eval_joint_mimic_coordinate(
        reference,
        component,
        body_q,
        data.body_com,
        data.joint_type,
        data.parent,
        data.child,
        data.X_p,
        data.X_c,
        data.qd_start,
        data.dof_dim,
        data.axis,
    )
    offset, multiplier = data.coeffs[follower][0], data.coeffs[follower][1]
    error = q_f - offset - multiplier * q_r
    if component >= data.dof_dim[follower, 0]:
        error = wp.atan2(wp.sin(error), wp.cos(error))
    bodies = wp.vec4i(data.parent[follower], data.child[follower], data.parent[reference], data.child[reference])
    gradients = wp.matrix_from_rows(g_fp, g_fc, -multiplier * g_rp, -multiplier * g_rc)
    # Shared parents and serial joints must use the SUM of their gradients
    # before evaluating J H^-1 J^T; otherwise the cross terms are lost.
    for i in range(4):
        if bodies[i] >= 0:
            for j in range(i):
                if bodies[i] == bodies[j]:
                    gradients[j] = gradients[j] + gradients[i]
                    bodies[i] = -1
    return error, bodies, gradients


@wp.kernel
def _accumulate_reactions(
    data: _JointData,
    followers: wp.array[int],
    body_q: wp.array[wp.transform],
    multipliers: wp.array[float],
    forces: wp.array[wp.vec3],
    torques: wp.array[wp.vec3],
):
    follower = followers[wp.tid()]
    reference = data.reference[follower]
    if reference < 0 or not data.enabled[follower] or not data.enabled[reference]:
        return
    for component in range(data.dof_dim[follower, 0] + data.dof_dim[follower, 1]):
        _error, bodies, gradients = _evaluate_row(data, follower, component, body_q)
        multiplier = multipliers[data.qd_start[follower] + component]
        for i in range(4):
            if bodies[i] >= 0:
                reaction = multiplier * gradients[i]
                wp.atomic_add(forces, bodies[i], wp.spatial_top(reaction))
                wp.atomic_add(torques, bodies[i], wp.spatial_bottom(reaction))


@wp.kernel
def _solve_mimics(
    data: _JointData,
    followers: wp.array[int],
    body_inv_mass: wp.array[float],
    h_ll: wp.array[wp.mat33],
    h_al: wp.array[wp.mat33],
    h_aa: wp.array[wp.mat33],
    body_q: wp.array[wp.transform],
    multipliers: wp.array[float],
):
    follower = followers[wp.tid()]
    reference = data.reference[follower]
    if reference < 0 or not data.enabled[follower] or not data.enabled[reference]:
        return
    # Rows in a color have disjoint bodies. Components within a joint are
    # sequential, so neither multi-axis joints nor shared leaders race.
    for component in range(data.dof_dim[follower, 0] + data.dof_dim[follower, 1]):
        error, bodies, gradients = _evaluate_row(data, follower, component, body_q)
        responses = _Mat46()
        compliance = float(0.0)
        for i in range(4):
            body = bodies[i]
            if body >= 0 and body_inv_mass[body] > 0.0:
                gradient = gradients[i]
                dx, dw = ldlt6_solve(
                    h_ll[body], h_aa[body], h_al[body], wp.spatial_top(gradient), wp.spatial_bottom(gradient)
                )
                response = wp.spatial_vector(dx, dw)
                responses[i] = response
                compliance += wp.dot(gradient, response)
        if compliance <= 0.0:
            continue
        delta_lambda = -error / compliance
        multipliers[data.qd_start[follower] + component] += delta_lambda
        for i in range(4):
            body = bodies[i]
            if body >= 0 and body_inv_mass[body] > 0.0:
                delta = delta_lambda * responses[i]
                pose = body_q[body]
                rotation = wp.transform_get_rotation(pose)
                com = wp.transform_point(pose, data.body_com[body])
                rotation_new = wp.normalize(rotation + 0.5 * wp.quat(wp.spatial_bottom(delta), 0.0) * rotation)
                position_new = com + wp.spatial_top(delta) - wp.quat_rotate(rotation_new, data.body_com[body])
                body_q[body] = wp.transform(position_new, rotation_new)


class JointMimicSolver:
    """Solve mimic rows once per VBD iteration, including their reaction forces.

    A mass-only projection discards friction/drive/contact stiffness and can
    satisfy the mimic equation while violating force balance. Instead, use
    the Schur complement J H^-1 J^T of VBD's assembled body blocks, and feed
    the resulting force multipliers into the next body sweep. Multipliers
    are reset each timestep; no additional iteration setting is needed.
    """

    def __init__(self, model: Model):
        self.model = model
        self.data = _JointData()
        self.data.body_com = model.body_com
        for field, source in (
            ("joint_type", "joint_type"),
            ("enabled", "joint_enabled"),
            ("parent", "joint_parent"),
            ("child", "joint_child"),
            ("X_p", "joint_X_p"),
            ("X_c", "joint_X_c"),
            ("qd_start", "joint_qd_start"),
            ("dof_dim", "joint_dof_dim"),
            ("axis", "joint_axis"),
            ("reference", "joint_mimic_joint"),
            ("coeffs", "joint_mimic_coeffs"),
        ):
            setattr(self.data, field, getattr(model, source))
        self.multipliers = wp.zeros_like(model.joint_qd)
        references = model.joint_mimic_joint.numpy()
        parents, children = model.joint_parent.numpy(), model.joint_child.numpy()
        types = model.joint_type.numpy()
        supported = (JointType.REVOLUTE, JointType.PRISMATIC, JointType.D6)
        followers = [j for j, r in enumerate(references) if r >= 0 and types[j] in supported and types[r] in supported]
        self.followers = wp.array(followers, dtype=int, device=model.device)
        # Greedy edge coloring of four-body constraints, independent of the
        # ordinary body coloring (which only includes structural joints).
        used_colors: dict[int, set[int]] = {}
        colors: list[list[int]] = []
        for follower in followers:
            reference = references[follower]
            bodies = {
                int(b)
                for b in (parents[follower], children[follower], parents[reference], children[reference])
                if b >= 0
            }
            unavailable = set().union(*(used_colors.get(b, set()) for b in bodies))
            color = 0
            while color in unavailable:
                color += 1
            if color == len(colors):
                colors.append([])
            colors[color].append(follower)
            for body in bodies:
                used_colors.setdefault(body, set()).add(color)
        self.colors = [wp.array(group, dtype=int, device=model.device) for group in colors]

    def reset(self):
        self.multipliers.zero_()

    def accumulate_reactions(self, body_q, forces, torques):
        wp.launch(
            _accumulate_reactions,
            dim=self.followers.size,
            inputs=[self.data, self.followers, body_q, self.multipliers],
            outputs=[forces, torques],
            device=self.model.device,
        )

    def solve(self, body_q, body_inv_mass, h_ll, h_al, h_aa):
        for group in self.colors:
            wp.launch(
                _solve_mimics,
                dim=group.size,
                inputs=[self.data, group, body_inv_mass, h_ll, h_al, h_aa],
                outputs=[body_q, self.multipliers],
                device=self.model.device,
            )
