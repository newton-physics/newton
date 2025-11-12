# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import warp as wp

from ...core import (
    quat_decompose,
    quat_twist,
)
from ...sim import (
    Control,
    JointType,
    Model,
    State,
)


@wp.func
def joint_force(
    q: float,
    qd: float,
    joint_target_pos: float,
    joint_target_vel: float,
    target_ke: float,
    target_kd: float,
    limit_lower: float,
    limit_upper: float,
    limit_ke: float,
    limit_kd: float,
) -> float:
    """Joint force evaluation for a single degree of freedom."""

    limit_f = 0.0
    damping_f = 0.0
    target_f = 0.0

    target_f = target_ke * (joint_target_pos - q) + target_kd * (joint_target_vel - qd)

    # When limit violated: apply limit restoration forces and disable target control
    if q < limit_lower:
        limit_f = limit_ke * (limit_lower - q)
        damping_f = -limit_kd * qd
        target_f = 0.0
    elif q > limit_upper:
        limit_f = limit_ke * (limit_upper - q)
        damping_f = -limit_kd * qd
        target_f = 0.0

    return limit_f + damping_f + target_f


@wp.kernel
def eval_body_joints(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    joint_qd_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_enabled: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    joint_f: wp.array(dtype=float),
    joint_target_pos: wp.array(dtype=float),
    joint_target_vel: wp.array(dtype=float),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    joint_attach_ke: float,
    joint_attach_kd: float,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    type = joint_type[tid]

    c_child = joint_child[tid]
    c_parent = joint_parent[tid]

    if joint_enabled[tid] == 0:
        return

    qd_start = joint_qd_start[tid]
    if type == JointType.FREE or type == JointType.DISTANCE:
        wrench = wp.spatial_vector(
            joint_f[qd_start + 0],
            joint_f[qd_start + 1],
            joint_f[qd_start + 2],
            joint_f[qd_start + 3],
            joint_f[qd_start + 4],
            joint_f[qd_start + 5],
        )

        wp.atomic_add(body_f, c_child, wrench)
        return

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]

    X_wp = X_pj
    r_p = wp.vec3()
    w_p = wp.vec3()
    v_p = wp.vec3()

    # parent transform and moment arm
    if c_parent >= 0:
        X_wp = body_q[c_parent] * X_wp
        r_p = wp.transform_get_translation(X_wp) - wp.transform_point(body_q[c_parent], body_com[c_parent])

        twist_p = body_qd[c_parent]

        w_p = wp.spatial_bottom(twist_p)
        v_p = wp.spatial_top(twist_p) + wp.cross(w_p, r_p)

    # child transform and moment arm
    X_wc = body_q[c_child] * X_cj
    r_c = wp.transform_get_translation(X_wc) - wp.transform_point(body_q[c_child], body_com[c_child])

    twist_c = body_qd[c_child]

    w_c = wp.spatial_bottom(twist_c)
    v_c = wp.spatial_top(twist_c) + wp.cross(w_c, r_c)

    lin_axis_count = joint_dof_dim[tid, 0]
    ang_axis_count = joint_dof_dim[tid, 1]

    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)

    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)

    # translational error
    x_err = x_c - x_p
    r_err = wp.quat_inverse(q_p) * q_c
    v_err = v_c - v_p
    w_err = w_c - w_p

    # total force/torque on the parent
    t_total = wp.vec3()
    f_total = wp.vec3()

    # reduce angular damping stiffness for stability
    angular_damping_scale = 0.01

    if type == JointType.FIXED:
        ang_err = wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2])) * wp.acos(r_err[3]) * 2.0

        f_total += x_err * joint_attach_ke + v_err * joint_attach_kd
        t_total += (
            wp.transform_vector(X_wp, ang_err) * joint_attach_ke + w_err * joint_attach_kd * angular_damping_scale
        )

    if type == JointType.PRISMATIC:
        axis = joint_axis[qd_start]

        # world space joint axis
        axis_p = wp.transform_vector(X_wp, axis)

        # evaluate joint coordinates
        q = wp.dot(x_err, axis_p)
        qd = wp.dot(v_err, axis_p)

        f_total = axis_p * (
            -joint_f[qd_start]
            - joint_force(
                q,
                qd,
                joint_target_pos[qd_start],
                joint_target_vel[qd_start],
                joint_target_ke[qd_start],
                joint_target_kd[qd_start],
                joint_limit_lower[qd_start],
                joint_limit_upper[qd_start],
                joint_limit_ke[qd_start],
                joint_limit_kd[qd_start],
            )
        )

        # attachment dynamics
        ang_err = wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2])) * wp.acos(r_err[3]) * 2.0

        # project off any displacement along the joint axis
        f_total += (x_err - q * axis_p) * joint_attach_ke + (v_err - qd * axis_p) * joint_attach_kd
        t_total += (
            wp.transform_vector(X_wp, ang_err) * joint_attach_ke + w_err * joint_attach_kd * angular_damping_scale
        )

    if type == JointType.REVOLUTE:
        axis = joint_axis[qd_start]

        axis_p = wp.transform_vector(X_wp, axis)
        axis_c = wp.transform_vector(X_wc, axis)

        # swing twist decomposition
        twist = quat_twist(axis, r_err)

        q = wp.acos(twist[3]) * 2.0 * wp.sign(wp.dot(axis, wp.vec3(twist[0], twist[1], twist[2])))
        qd = wp.dot(w_err, axis_p)

        t_total = axis_p * (
            -joint_f[qd_start]
            - joint_force(
                q,
                qd,
                joint_target_pos[qd_start],
                joint_target_vel[qd_start],
                joint_target_ke[qd_start],
                joint_target_kd[qd_start],
                joint_limit_lower[qd_start],
                joint_limit_upper[qd_start],
                joint_limit_ke[qd_start],
                joint_limit_kd[qd_start],
            )
        )

        # attachment dynamics
        swing_err = wp.cross(axis_p, axis_c)

        f_total += x_err * joint_attach_ke + v_err * joint_attach_kd
        t_total += swing_err * joint_attach_ke + (w_err - qd * axis_p) * joint_attach_kd * angular_damping_scale

    if type == JointType.BALL:
        ang_err = wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2])) * wp.acos(r_err[3]) * 2.0

        # TODO joint limits
        # TODO expose target_kd or target_ke for ball joints
        # t_total += target_kd * w_err + target_ke * wp.transform_vector(X_wp, ang_err)
        f_total += x_err * joint_attach_ke + v_err * joint_attach_kd
        t_total += wp.vec3(-joint_f[qd_start], -joint_f[qd_start + 1], -joint_f[qd_start + 2])

    if type == JointType.D6:
        pos = wp.vec3(0.0)
        vel = wp.vec3(0.0)
        if lin_axis_count >= 1:
            axis_0 = wp.transform_vector(X_wp, joint_axis[qd_start + 0])
            q0 = wp.dot(x_err, axis_0)
            qd0 = wp.dot(v_err, axis_0)

            f_total += axis_0 * (
                -joint_f[qd_start]
                - joint_force(
                    q0,
                    qd0,
                    joint_target_pos[qd_start + 0],
                    joint_target_vel[qd_start + 0],
                    joint_target_ke[qd_start + 0],
                    joint_target_kd[qd_start + 0],
                    joint_limit_lower[qd_start + 0],
                    joint_limit_upper[qd_start + 0],
                    joint_limit_ke[qd_start + 0],
                    joint_limit_kd[qd_start + 0],
                )
            )

            pos += q0 * axis_0
            vel += qd0 * axis_0

        if lin_axis_count >= 2:
            axis_1 = wp.transform_vector(X_wp, joint_axis[qd_start + 1])
            q1 = wp.dot(x_err, axis_1)
            qd1 = wp.dot(v_err, axis_1)

            f_total += axis_1 * (
                -joint_f[qd_start + 1]
                - joint_force(
                    q1,
                    qd1,
                    joint_target_pos[qd_start + 1],
                    joint_target_vel[qd_start + 1],
                    joint_target_ke[qd_start + 1],
                    joint_target_kd[qd_start + 1],
                    joint_limit_lower[qd_start + 1],
                    joint_limit_upper[qd_start + 1],
                    joint_limit_ke[qd_start + 1],
                    joint_limit_kd[qd_start + 1],
                )
            )

            pos += q1 * axis_1
            vel += qd1 * axis_1

        if lin_axis_count == 3:
            axis_2 = wp.transform_vector(X_wp, joint_axis[qd_start + 2])
            q2 = wp.dot(x_err, axis_2)
            qd2 = wp.dot(v_err, axis_2)

            f_total += axis_2 * (
                -joint_f[qd_start + 2]
                - joint_force(
                    q2,
                    qd2,
                    joint_target_pos[qd_start + 2],
                    joint_target_vel[qd_start + 2],
                    joint_target_ke[qd_start + 2],
                    joint_target_kd[qd_start + 2],
                    joint_limit_lower[qd_start + 2],
                    joint_limit_upper[qd_start + 2],
                    joint_limit_ke[qd_start + 2],
                    joint_limit_kd[qd_start + 2],
                )
            )

            pos += q2 * axis_2
            vel += qd2 * axis_2

        f_total += (x_err - pos) * joint_attach_ke + (v_err - vel) * joint_attach_kd

        if ang_axis_count == 0:
            ang_err = wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2])) * wp.acos(r_err[3]) * 2.0
            t_total += (
                wp.transform_vector(X_wp, ang_err) * joint_attach_ke + w_err * joint_attach_kd * angular_damping_scale
            )

        i_0 = lin_axis_count + qd_start + 0
        i_1 = lin_axis_count + qd_start + 1
        i_2 = lin_axis_count + qd_start + 2
        qdi_start = qd_start + lin_axis_count

        if ang_axis_count == 1:
            axis = joint_axis[i_0]

            axis_p = wp.transform_vector(X_wp, axis)
            axis_c = wp.transform_vector(X_wc, axis)

            # swing twist decomposition
            twist = quat_twist(axis, r_err)

            q = wp.acos(twist[3]) * 2.0 * wp.sign(wp.dot(axis, wp.vec3(twist[0], twist[1], twist[2])))
            qd = wp.dot(w_err, axis_p)

            t_total = axis_p * (
                -joint_f[qdi_start]
                - joint_force(
                    q,
                    qd,
                    joint_target_pos[i_0],
                    joint_target_vel[i_0],
                    joint_target_ke[i_0],
                    joint_target_kd[i_0],
                    joint_limit_lower[i_0],
                    joint_limit_upper[i_0],
                    joint_limit_ke[i_0],
                    joint_limit_kd[i_0],
                )
            )

            # attachment dynamics
            swing_err = wp.cross(axis_p, axis_c)

            t_total += swing_err * joint_attach_ke + (w_err - qd * axis_p) * joint_attach_kd * angular_damping_scale

        if ang_axis_count == 2:
            q_pc = wp.quat_inverse(q_p) * q_c

            # decompose to a compound rotation each axis
            angles = quat_decompose(q_pc)

            orig_axis_0 = joint_axis[i_0]
            orig_axis_1 = joint_axis[i_1]
            orig_axis_2 = wp.cross(orig_axis_0, orig_axis_1)

            # reconstruct rotation axes
            axis_0 = orig_axis_0
            q_0 = wp.quat_from_axis_angle(axis_0, angles[0])

            axis_1 = wp.quat_rotate(q_0, orig_axis_1)
            q_1 = wp.quat_from_axis_angle(axis_1, angles[1])

            axis_2 = wp.quat_rotate(q_1 * q_0, orig_axis_2)

            axis_0 = wp.transform_vector(X_wp, axis_0)
            axis_1 = wp.transform_vector(X_wp, axis_1)
            axis_2 = wp.transform_vector(X_wp, axis_2)

            # joint dynamics

            t_total += axis_0 * (
                -joint_f[qdi_start]
                - joint_force(
                    angles[0],
                    wp.dot(axis_0, w_err),
                    joint_target_pos[i_0],
                    joint_target_vel[i_0],
                    joint_target_ke[i_0],
                    joint_target_kd[i_0],
                    joint_limit_lower[i_0],
                    joint_limit_upper[i_0],
                    joint_limit_ke[i_0],
                    joint_limit_kd[i_0],
                )
            )
            t_total += axis_1 * (
                -joint_f[qdi_start + 1]
                - joint_force(
                    angles[1],
                    wp.dot(axis_1, w_err),
                    joint_target_pos[i_1],
                    joint_target_vel[i_1],
                    joint_target_ke[i_1],
                    joint_target_kd[i_1],
                    joint_limit_lower[i_1],
                    joint_limit_upper[i_1],
                    joint_limit_ke[i_1],
                    joint_limit_kd[i_1],
                )
            )

            # last axis (fixed)
            t_total += axis_2 * -joint_force(
                angles[2],
                wp.dot(axis_2, w_err),
                0.0,
                0.0,
                joint_attach_ke,
                joint_attach_kd * angular_damping_scale,
                0.0,
                0.0,
                0.0,
                0.0,
            )

        if ang_axis_count == 3:
            q_pc = wp.quat_inverse(q_p) * q_c

            # decompose to a compound rotation each axis
            angles = quat_decompose(q_pc)

            orig_axis_0 = joint_axis[i_0]
            orig_axis_1 = joint_axis[i_1]
            orig_axis_2 = joint_axis[i_2]

            # reconstruct rotation axes
            axis_0 = orig_axis_0
            q_0 = wp.quat_from_axis_angle(axis_0, angles[0])

            axis_1 = wp.quat_rotate(q_0, orig_axis_1)
            q_1 = wp.quat_from_axis_angle(axis_1, angles[1])

            axis_2 = wp.quat_rotate(q_1 * q_0, orig_axis_2)

            axis_0 = wp.transform_vector(X_wp, axis_0)
            axis_1 = wp.transform_vector(X_wp, axis_1)
            axis_2 = wp.transform_vector(X_wp, axis_2)

            t_total += axis_0 * (
                -joint_f[qdi_start]
                - joint_force(
                    angles[0],
                    wp.dot(axis_0, w_err),
                    joint_target_pos[i_0],
                    joint_target_vel[i_0],
                    joint_target_ke[i_0],
                    joint_target_kd[i_0],
                    joint_limit_lower[i_0],
                    joint_limit_upper[i_0],
                    joint_limit_ke[i_0],
                    joint_limit_kd[i_0],
                )
            )
            t_total += axis_1 * (
                -joint_f[qdi_start + 1]
                - joint_force(
                    angles[1],
                    wp.dot(axis_1, w_err),
                    joint_target_pos[i_1],
                    joint_target_vel[i_1],
                    joint_target_ke[i_1],
                    joint_target_kd[i_1],
                    joint_limit_lower[i_1],
                    joint_limit_upper[i_1],
                    joint_limit_ke[i_1],
                    joint_limit_kd[i_1],
                )
            )
            t_total += axis_2 * (
                -joint_f[qdi_start + 2]
                - joint_force(
                    angles[2],
                    wp.dot(axis_2, w_err),
                    joint_target_pos[i_2],
                    joint_target_vel[i_2],
                    joint_target_ke[i_2],
                    joint_target_kd[i_2],
                    joint_limit_lower[i_2],
                    joint_limit_upper[i_2],
                    joint_limit_ke[i_2],
                    joint_limit_kd[i_2],
                )
            )

    # write forces
    if c_parent >= 0:
        wp.atomic_add(body_f, c_parent, wp.spatial_vector(f_total, t_total + wp.cross(r_p, f_total)))

    wp.atomic_sub(body_f, c_child, wp.spatial_vector(f_total, t_total + wp.cross(r_c, f_total)))


def eval_body_joint_forces(
    model: Model, state: State, control: Control, body_f: wp.array, joint_attach_ke: float, joint_attach_kd: float
):
    if model.joint_count:
        wp.launch(
            kernel=eval_body_joints,
            dim=model.joint_count,
            inputs=[
                state.body_q,
                state.body_qd,
                model.body_com,
                model.joint_qd_start,
                model.joint_type,
                model.joint_enabled,
                model.joint_child,
                model.joint_parent,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis,
                model.joint_dof_dim,
                control.joint_f,
                control.joint_target_pos,
                control.joint_target_vel,
                model.joint_target_ke,
                model.joint_target_kd,
                model.joint_limit_lower,
                model.joint_limit_upper,
                model.joint_limit_ke,
                model.joint_limit_kd,
                joint_attach_ke,
                joint_attach_kd,
            ],
            outputs=[body_f],
            device=model.device,
        )
