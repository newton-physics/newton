# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

from typing import ClassVar

import warp as wp

from ...math.spatial import transform_twist
from ...sim import JointType
from ...sim.articulation import (
    compute_2d_rotational_dofs,
    compute_3d_rotational_dofs,
)

PGS_CONSTRAINT_TYPE_CONTACT = 0
PGS_CONSTRAINT_TYPE_JOINT_TARGET = 1
PGS_CONSTRAINT_TYPE_FRICTION = 2
PGS_CONSTRAINT_TYPE_JOINT_LIMIT = 3
# Joint velocity-limit row. Mirrors the PhysX per-DOF velocity clamp (see
# ``notes/investigations/velocity-spike/physx-deep-dive.md`` §4, math
# appendix). Activated when ``|qdot_i| > model.joint_velocity_limit[i]``;
# solved as a unilateral row with ``lambda >= 0`` (the sign of J flips so
# the two sides of the bilateral ``[-qdot_max, +qdot_max]`` box are both
# handled by the same PGS projector). No Baumgarte / ERP bias.
PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT = 4


@wp.kernel
def copy_int_array_masked(
    src: wp.array[int],
    mask: wp.array[int],
    # outputs
    dst: wp.array[int],
):
    tid = wp.tid()
    if mask[tid] != 0:
        dst[tid] = src[tid]


@wp.kernel
def compute_spatial_inertia(
    body_inertia: wp.array[wp.mat33],
    body_mass: wp.array[float],
    # outputs
    body_I_m: wp.array[wp.spatial_matrix],
):
    tid = wp.tid()
    I = body_inertia[tid]
    m = body_mass[tid]
    # fmt: off
    body_I_m[tid] = wp.spatial_matrix(
        m,   0.0, 0.0, 0.0,     0.0,     0.0,
        0.0, m,   0.0, 0.0,     0.0,     0.0,
        0.0, 0.0, m,   0.0,     0.0,     0.0,
        0.0, 0.0, 0.0, I[0, 0], I[0, 1], I[0, 2],
        0.0, 0.0, 0.0, I[1, 0], I[1, 1], I[1, 2],
        0.0, 0.0, 0.0, I[2, 0], I[2, 1], I[2, 2],
    )
    # fmt: on


@wp.kernel
def compute_com_transforms(
    body_com: wp.array[wp.vec3],
    # outputs
    body_X_com: wp.array[wp.transform],
):
    tid = wp.tid()
    com = body_com[tid]
    body_X_com[tid] = wp.transform(com, wp.quat_identity())


@wp.kernel
def update_articulation_origins(
    articulation_start: wp.array[int],
    joint_child: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    # outputs
    articulation_origin: wp.array[wp.vec3],
):
    art = wp.tid()

    start = articulation_start[art]
    end = articulation_start[art + 1]

    if start >= end:
        articulation_origin[art] = wp.vec3()
        return

    root_body = joint_child[start]
    if root_body >= 0:
        # Store the absolute world-space COM position of the articulation root body.
        articulation_origin[art] = wp.transform_point(body_q[root_body], body_com[root_body])
    else:
        articulation_origin[art] = wp.vec3()


@wp.kernel
def update_articulation_root_com_offsets(
    articulation_start: wp.array[int],
    joint_child: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    # outputs
    articulation_root_com_offset: wp.array[wp.vec3],
):
    # NOTE: This helper keeps the rotated root COM offset in world orientation.
    # FeatherPGS currently uses update_articulation_origins() instead, which
    # stores the absolute root COM world position for its free-root convention.
    art = wp.tid()

    start = articulation_start[art]
    end = articulation_start[art + 1]

    if start >= end:
        articulation_root_com_offset[art] = wp.vec3()
        return

    root_body = joint_child[start]
    if root_body >= 0:
        rot = wp.transform_get_rotation(body_q[root_body])
        articulation_root_com_offset[art] = wp.quat_rotate(rot, body_com[root_body])
    else:
        articulation_root_com_offset[art] = wp.vec3()


@wp.kernel
def convert_root_free_qd_world_to_local(
    articulation_root_is_free: wp.array[int],
    articulation_root_dof_start: wp.array[int],
    articulation_root_com_offset: wp.array[wp.vec3],
    # in/out
    qd: wp.array[float],
):
    art = wp.tid()
    if articulation_root_is_free[art] == 0:
        return

    ds = articulation_root_dof_start[art]
    v_com = wp.vec3(qd[ds + 0], qd[ds + 1], qd[ds + 2])
    w = wp.vec3(qd[ds + 3], qd[ds + 4], qd[ds + 5])
    com_offset = articulation_root_com_offset[art]

    # Shift linear velocity from the public CoM convention to the internal
    # root-body-origin linear term used by FeatherPGS integration/ID state.
    v_local = v_com - wp.cross(w, com_offset)

    qd[ds + 0] = v_local[0]
    qd[ds + 1] = v_local[1]
    qd[ds + 2] = v_local[2]


@wp.kernel
def convert_root_free_qd_local_to_world(
    articulation_root_is_free: wp.array[int],
    articulation_root_dof_start: wp.array[int],
    articulation_root_com_offset: wp.array[wp.vec3],
    # in/out
    qd: wp.array[float],
):
    art = wp.tid()
    if articulation_root_is_free[art] == 0:
        return

    ds = articulation_root_dof_start[art]
    v_local = wp.vec3(qd[ds + 0], qd[ds + 1], qd[ds + 2])
    w = wp.vec3(qd[ds + 3], qd[ds + 4], qd[ds + 5])
    com_offset = articulation_root_com_offset[art]

    # Convert the internal root-body-origin linear term back to the public CoM convention.
    v_com = v_local + wp.cross(w, com_offset)

    qd[ds + 0] = v_com[0]
    qd[ds + 1] = v_com[1]
    qd[ds + 2] = v_com[2]


@wp.func
def transform_spatial_inertia(t: wp.transform, I: wp.spatial_matrix):
    """
    Transform a spatial inertia tensor to a new coordinate frame.

    This computes the change of coordinates for a spatial inertia tensor under a rigid-body
    transformation `t`. The result is mathematically equivalent to:

        adj_t^-T * I * adj_t^-1

    where `adj_t` is the adjoint transformation matrix of `t`, and `I` is the spatial inertia
    tensor in the original frame. This operation is described in Frank & Park, "Modern Robotics",
    Section 8.2.3 (pg. 290).

    Args:
        t (wp.transform): The rigid-body transform (destination ← source).
        I (wp.spatial_matrix): The spatial inertia tensor in the source frame.

    Returns:
        wp.spatial_matrix: The spatial inertia tensor expressed in the destination frame.
    """
    t_inv = wp.transform_inverse(t)

    q = wp.transform_get_rotation(t_inv)
    p = wp.transform_get_translation(t_inv)

    r1 = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    r2 = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))
    r3 = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))

    R = wp.matrix_from_cols(r1, r2, r3)
    S = wp.skew(p) @ R

    T = wp.spatial_matrix(
        R[0, 0],
        R[0, 1],
        R[0, 2],
        S[0, 0],
        S[0, 1],
        S[0, 2],
        R[1, 0],
        R[1, 1],
        R[1, 2],
        S[1, 0],
        S[1, 1],
        S[1, 2],
        R[2, 0],
        R[2, 1],
        R[2, 2],
        S[2, 0],
        S[2, 1],
        S[2, 2],
        0.0,
        0.0,
        0.0,
        R[0, 0],
        R[0, 1],
        R[0, 2],
        0.0,
        0.0,
        0.0,
        R[1, 0],
        R[1, 1],
        R[1, 2],
        0.0,
        0.0,
        0.0,
        R[2, 0],
        R[2, 1],
        R[2, 2],
    )

    return wp.mul(wp.mul(wp.transpose(T), I), T)


# compute transform across a joint
@wp.func
def jcalc_transform(
    type: int,
    joint_axis: wp.array[wp.vec3],
    axis_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    joint_q: wp.array[float],
    q_start: int,
):
    if type == JointType.PRISMATIC:
        q = joint_q[q_start]
        axis = joint_axis[axis_start]
        X_jc = wp.transform(axis * q, wp.quat_identity())
        return X_jc

    if type == JointType.REVOLUTE:
        q = joint_q[q_start]
        axis = joint_axis[axis_start]
        X_jc = wp.transform(wp.vec3(), wp.quat_from_axis_angle(axis, q))
        return X_jc

    if type == JointType.BALL:
        qx = joint_q[q_start + 0]
        qy = joint_q[q_start + 1]
        qz = joint_q[q_start + 2]
        qw = joint_q[q_start + 3]

        X_jc = wp.transform(wp.vec3(), wp.quat(qx, qy, qz, qw))
        return X_jc

    if type == JointType.FIXED:
        X_jc = wp.transform_identity()
        return X_jc

    if type == JointType.FREE or type == JointType.DISTANCE:
        px = joint_q[q_start + 0]
        py = joint_q[q_start + 1]
        pz = joint_q[q_start + 2]

        qx = joint_q[q_start + 3]
        qy = joint_q[q_start + 4]
        qz = joint_q[q_start + 5]
        qw = joint_q[q_start + 6]

        X_jc = wp.transform(wp.vec3(px, py, pz), wp.quat(qx, qy, qz, qw))
        return X_jc

    if type == JointType.D6:
        pos = wp.vec3(0.0)
        rot = wp.quat_identity()

        # unroll for loop to ensure joint actions remain differentiable
        # (since differentiating through a for loop that updates a local variable is not supported)

        if lin_axis_count > 0:
            axis = joint_axis[axis_start + 0]
            pos += axis * joint_q[q_start + 0]
        if lin_axis_count > 1:
            axis = joint_axis[axis_start + 1]
            pos += axis * joint_q[q_start + 1]
        if lin_axis_count > 2:
            axis = joint_axis[axis_start + 2]
            pos += axis * joint_q[q_start + 2]

        ia = axis_start + lin_axis_count
        iq = q_start + lin_axis_count
        if ang_axis_count == 1:
            axis = joint_axis[ia]
            rot = wp.quat_from_axis_angle(axis, joint_q[iq])
        if ang_axis_count == 2:
            rot, _ = compute_2d_rotational_dofs(
                joint_axis[ia + 0],
                joint_axis[ia + 1],
                joint_q[iq + 0],
                joint_q[iq + 1],
                0.0,
                0.0,
            )
        if ang_axis_count == 3:
            rot, _ = compute_3d_rotational_dofs(
                joint_axis[ia + 0],
                joint_axis[ia + 1],
                joint_axis[ia + 2],
                joint_q[iq + 0],
                joint_q[iq + 1],
                joint_q[iq + 2],
                0.0,
                0.0,
                0.0,
            )

        X_jc = wp.transform(pos, rot)
        return X_jc

    # default case
    return wp.transform_identity()


# compute motion subspace and velocity for a joint
@wp.func
def jcalc_motion(
    type: int,
    joint_axis: wp.array[wp.vec3],
    lin_axis_count: int,
    ang_axis_count: int,
    X_sc: wp.transform,
    joint_qd: wp.array[float],
    qd_start: int,
    # outputs
    joint_S_s: wp.array[wp.spatial_vector],
):
    if type == JointType.PRISMATIC:
        axis = joint_axis[qd_start]
        S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
        v_j_s = S_s * joint_qd[qd_start]
        joint_S_s[qd_start] = S_s
        return v_j_s

    if type == JointType.REVOLUTE:
        axis = joint_axis[qd_start]
        S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
        v_j_s = S_s * joint_qd[qd_start]
        joint_S_s[qd_start] = S_s
        return v_j_s

    if type == JointType.D6:
        v_j_s = wp.spatial_vector()
        if lin_axis_count > 0:
            axis = joint_axis[qd_start + 0]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + 0]
            joint_S_s[qd_start + 0] = S_s
        if lin_axis_count > 1:
            axis = joint_axis[qd_start + 1]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + 1]
            joint_S_s[qd_start + 1] = S_s
        if lin_axis_count > 2:
            axis = joint_axis[qd_start + 2]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + 2]
            joint_S_s[qd_start + 2] = S_s
        if ang_axis_count > 0:
            axis = joint_axis[qd_start + lin_axis_count + 0]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 0]
            joint_S_s[qd_start + lin_axis_count + 0] = S_s
        if ang_axis_count > 1:
            axis = joint_axis[qd_start + lin_axis_count + 1]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 1]
            joint_S_s[qd_start + lin_axis_count + 1] = S_s
        if ang_axis_count > 2:
            axis = joint_axis[qd_start + lin_axis_count + 2]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 2]
            joint_S_s[qd_start + lin_axis_count + 2] = S_s

        return v_j_s

    if type == JointType.BALL:
        S_0 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        S_1 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        S_2 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

        joint_S_s[qd_start + 0] = S_0
        joint_S_s[qd_start + 1] = S_1
        joint_S_s[qd_start + 2] = S_2

        return S_0 * joint_qd[qd_start + 0] + S_1 * joint_qd[qd_start + 1] + S_2 * joint_qd[qd_start + 2]

    if type == JointType.FIXED:
        return wp.spatial_vector()

    if type == JointType.FREE or type == JointType.DISTANCE:
        # For FREE/DISTANCE joints we treat linear/angular velocity components as
        # referenced at the root COM world point to avoid world-origin conditioning.
        q_sc = wp.transform_get_rotation(X_sc)

        v_local = wp.vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2])
        w_local = wp.vec3(joint_qd[qd_start + 3], joint_qd[qd_start + 4], joint_qd[qd_start + 5])
        v_j_s = wp.spatial_vector(wp.quat_rotate(q_sc, v_local), wp.quat_rotate(q_sc, w_local))

        ex = wp.quat_rotate(q_sc, wp.vec3(1.0, 0.0, 0.0))
        ey = wp.quat_rotate(q_sc, wp.vec3(0.0, 1.0, 0.0))
        ez = wp.quat_rotate(q_sc, wp.vec3(0.0, 0.0, 1.0))

        joint_S_s[qd_start + 0] = wp.spatial_vector(ex, wp.vec3())
        joint_S_s[qd_start + 1] = wp.spatial_vector(ey, wp.vec3())
        joint_S_s[qd_start + 2] = wp.spatial_vector(ez, wp.vec3())
        joint_S_s[qd_start + 3] = wp.spatial_vector(wp.vec3(), ex)
        joint_S_s[qd_start + 4] = wp.spatial_vector(wp.vec3(), ey)
        joint_S_s[qd_start + 5] = wp.spatial_vector(wp.vec3(), ez)

        return v_j_s

    wp.printf("jcalc_motion not implemented for joint type %d\n", type)

    # default case
    return wp.spatial_vector()


# computes joint space forces/torques in tau
@wp.func
def jcalc_tau(
    type: int,
    joint_S_s: wp.array[wp.spatial_vector],
    joint_f: wp.array[float],
    dof_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    body_f_s: wp.spatial_vector,
    # outputs
    tau: wp.array[float],
):
    if type == JointType.BALL:
        # target_ke = joint_target_ke[dof_start]
        # target_kd = joint_target_kd[dof_start]

        for i in range(3):
            S_s = joint_S_s[dof_start + i]

            # w = joint_qd[dof_start + i]
            # r = joint_q[coord_start + i]

            tau[dof_start + i] = -wp.dot(S_s, body_f_s) + joint_f[dof_start + i]
            # tau -= w * target_kd - r * target_ke

        return

    if type == JointType.FREE or type == JointType.DISTANCE:
        for i in range(6):
            S_s = joint_S_s[dof_start + i]
            tau[dof_start + i] = -wp.dot(S_s, body_f_s) + joint_f[dof_start + i]

        return

    if type == JointType.PRISMATIC or type == JointType.REVOLUTE or type == JointType.D6:
        axis_count = lin_axis_count + ang_axis_count

        for i in range(axis_count):
            j = dof_start + i
            S_s = joint_S_s[j]
            # total torque / force on the joint (drive forces handled via augmented mass)
            tau[j] = -wp.dot(S_s, body_f_s) + joint_f[j]

        return


@wp.func
def jcalc_integrate(
    type: int,
    child: int,
    body_com: wp.array[wp.vec3],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_qdd: wp.array[float],
    coord_start: int,
    dof_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    dt: float,
    # outputs
    joint_q_new: wp.array[float],
    joint_qd_new: wp.array[float],
):
    if type == JointType.FIXED:
        return

    # prismatic / revolute
    if type == JointType.PRISMATIC or type == JointType.REVOLUTE:
        qdd = joint_qdd[dof_start]
        qd = joint_qd[dof_start]
        q = joint_q[coord_start]

        qd_new = qd + qdd * dt
        q_new = q + qd_new * dt

        joint_qd_new[dof_start] = qd_new
        joint_q_new[coord_start] = q_new

        return

    # ball
    if type == JointType.BALL:
        m_j = wp.vec3(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2])
        w_j = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])

        r_j = wp.quat(
            joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2], joint_q[coord_start + 3]
        )

        # symplectic Euler
        w_j_new = w_j + m_j * dt

        drdt_j = wp.quat(w_j_new, 0.0) * r_j * 0.5

        # new orientation (normalized)
        r_j_new = wp.normalize(r_j + drdt_j * dt)

        # update joint coords
        joint_q_new[coord_start + 0] = r_j_new[0]
        joint_q_new[coord_start + 1] = r_j_new[1]
        joint_q_new[coord_start + 2] = r_j_new[2]
        joint_q_new[coord_start + 3] = r_j_new[3]

        # update joint vel
        joint_qd_new[dof_start + 0] = w_j_new[0]
        joint_qd_new[dof_start + 1] = w_j_new[1]
        joint_qd_new[dof_start + 2] = w_j_new[2]

        return

    if type == JointType.FREE or type == JointType.DISTANCE:
        a_s = wp.vec3(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2])
        m_s = wp.vec3(joint_qdd[dof_start + 3], joint_qdd[dof_start + 4], joint_qdd[dof_start + 5])

        v_com = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])
        w_s = wp.vec3(joint_qd[dof_start + 3], joint_qd[dof_start + 4], joint_qd[dof_start + 5])

        # symplectic Euler
        w_s = w_s + m_s * dt
        v_com = v_com + a_s * dt

        p_s = wp.vec3(joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2])

        r_s = wp.quat(
            joint_q[coord_start + 3], joint_q[coord_start + 4], joint_q[coord_start + 5], joint_q[coord_start + 6]
        )
        com_offset_world = wp.quat_rotate(r_s, body_com[child])
        dpdt_s = v_com - wp.cross(w_s, com_offset_world)

        drdt_s = wp.quat(w_s, 0.0) * r_s * 0.5

        # new orientation (normalized)
        p_s_new = p_s + dpdt_s * dt
        r_s_new = wp.normalize(r_s + drdt_s * dt)

        # update transform
        joint_q_new[coord_start + 0] = p_s_new[0]
        joint_q_new[coord_start + 1] = p_s_new[1]
        joint_q_new[coord_start + 2] = p_s_new[2]

        joint_q_new[coord_start + 3] = r_s_new[0]
        joint_q_new[coord_start + 4] = r_s_new[1]
        joint_q_new[coord_start + 5] = r_s_new[2]
        joint_q_new[coord_start + 6] = r_s_new[3]

        joint_qd_new[dof_start + 0] = v_com[0]
        joint_qd_new[dof_start + 1] = v_com[1]
        joint_qd_new[dof_start + 2] = v_com[2]
        joint_qd_new[dof_start + 3] = w_s[0]
        joint_qd_new[dof_start + 4] = w_s[1]
        joint_qd_new[dof_start + 5] = w_s[2]

        return

    # other joint types (compound, universal, D6)
    if type == JointType.D6:
        axis_count = lin_axis_count + ang_axis_count

        for i in range(axis_count):
            qdd = joint_qdd[dof_start + i]
            qd = joint_qd[dof_start + i]
            q = joint_q[coord_start + i]

            qd_new = qd + qdd * dt
            q_new = q + qd_new * dt

            joint_qd_new[dof_start + i] = qd_new
            joint_q_new[coord_start + i] = q_new

        return


@wp.func
def compute_link_transform(
    i: int,
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q: wp.array[float],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    body_X_com: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    # outputs
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
):
    # parent transform
    parent = joint_parent[i]
    child = joint_child[i]

    # parent transform in spatial coordinates
    X_pj = joint_X_p[i]
    X_cj = joint_X_c[i]
    # parent anchor frame in world space
    X_wpj = X_pj
    if parent >= 0:
        X_wp = body_q[parent]
        X_wpj = X_wp * X_wpj

    type = joint_type[i]
    qd_start = joint_qd_start[i]
    lin_axis_count = joint_dof_dim[i, 0]
    ang_axis_count = joint_dof_dim[i, 1]
    coord_start = joint_q_start[i]

    # compute transform across joint
    X_j = jcalc_transform(type, joint_axis, qd_start, lin_axis_count, ang_axis_count, joint_q, coord_start)

    # transform from world to joint anchor frame at child body
    X_wcj = X_wpj * X_j
    # transform from world to child body frame
    X_wc = X_wcj * wp.transform_inverse(X_cj)

    # compute transform of center of mass
    X_cm = body_X_com[child]
    X_sm = X_wc * X_cm

    # store geometry transforms
    body_q[child] = X_wc
    body_q_com[child] = X_sm


@wp.kernel
def eval_rigid_fk(
    articulation_start: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q: wp.array[float],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    body_X_com: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    # outputs
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
):
    # one thread per-articulation
    index = wp.tid()

    start = articulation_start[index]
    end = articulation_start[index + 1]

    for i in range(start, end):
        compute_link_transform(
            i,
            joint_type,
            joint_parent,
            joint_child,
            joint_q_start,
            joint_qd_start,
            joint_q,
            joint_X_p,
            joint_X_c,
            body_X_com,
            joint_axis,
            joint_dof_dim,
            body_q,
            body_q_com,
        )


@wp.func
def spatial_cross(a: wp.spatial_vector, b: wp.spatial_vector):
    w_a = wp.spatial_bottom(a)
    v_a = wp.spatial_top(a)

    w_b = wp.spatial_bottom(b)
    v_b = wp.spatial_top(b)

    w = wp.cross(w_a, w_b)
    v = wp.cross(w_a, v_b) + wp.cross(v_a, w_b)

    return wp.spatial_vector(v, w)


@wp.func
def spatial_cross_dual(a: wp.spatial_vector, b: wp.spatial_vector):
    w_a = wp.spatial_bottom(a)
    v_a = wp.spatial_top(a)

    w_b = wp.spatial_bottom(b)
    v_b = wp.spatial_top(b)

    w = wp.cross(w_a, w_b) + wp.cross(v_a, v_b)
    v = wp.cross(w_a, v_b)

    return wp.spatial_vector(v, w)


@wp.func
def dense_index(stride: int, i: int, j: int):
    return i * stride + j


@wp.func
def compute_link_velocity(
    i: int,
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_articulation: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_qd: wp.array[float],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    body_I_m: wp.array[wp.spatial_matrix],
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    joint_X_p: wp.array[wp.transform],
    articulation_origin: wp.array[wp.vec3],
    gravity: wp.array[wp.vec3],
    # outputs
    joint_S_s: wp.array[wp.spatial_vector],
    body_I_s: wp.array[wp.spatial_matrix],
    body_v_s: wp.array[wp.spatial_vector],
    body_f_s: wp.array[wp.spatial_vector],
    body_a_s: wp.array[wp.spatial_vector],
):
    type = joint_type[i]
    child = joint_child[i]
    parent = joint_parent[i]
    articulation = joint_articulation[i]
    qd_start = joint_qd_start[i]
    origin = wp.vec3()
    if articulation >= 0:
        origin = articulation_origin[articulation]

    X_pj = joint_X_p[i]
    # X_cj = joint_X_c[i]

    # parent anchor frame in world space
    X_wpj = X_pj
    if parent >= 0:
        X_wp = body_q[parent]
        X_wpj = X_wp * X_wpj
    X_wpj_local = wp.transform(
        wp.transform_get_translation(X_wpj) - origin,
        wp.transform_get_rotation(X_wpj),
    )

    # compute motion subspace and velocity across the joint (also stores S_s to global memory)
    lin_axis_count = joint_dof_dim[i, 0]
    ang_axis_count = joint_dof_dim[i, 1]
    v_j_s = jcalc_motion(
        type,
        joint_axis,
        lin_axis_count,
        ang_axis_count,
        X_wpj_local,
        joint_qd,
        qd_start,
        joint_S_s,
    )

    # parent velocity
    v_parent_s = wp.spatial_vector()
    a_parent_s = wp.spatial_vector()

    if parent >= 0:
        v_parent_s = body_v_s[parent]
        a_parent_s = body_a_s[parent]

    # body velocity, acceleration
    v_s = v_parent_s + v_j_s
    a_s = a_parent_s + spatial_cross(v_s, v_j_s)

    # compute body forces
    X_sm = body_q_com[child]
    X_sm_local = wp.transform(
        wp.transform_get_translation(X_sm) - origin,
        wp.transform_get_rotation(X_sm),
    )
    I_m = body_I_m[child]

    # gravity and external forces (expressed in frame aligned with s but centered at body mass)
    m = I_m[0, 0]

    f_g = m * gravity[0]
    r_com = wp.transform_get_translation(X_sm_local)
    f_g_s = wp.spatial_vector(f_g, wp.cross(r_com, f_g))

    # body forces
    I_s = transform_spatial_inertia(X_sm_local, I_m)

    f_b_s = I_s * a_s + spatial_cross_dual(v_s, I_s * v_s)

    body_v_s[child] = v_s
    body_a_s[child] = a_s
    body_f_s[child] = f_b_s - f_g_s
    body_I_s[child] = I_s


# Inverse dynamics via Recursive Newton-Euler algorithm (Featherstone Table 5.1)
@wp.kernel
def eval_rigid_id(
    articulation_start: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_articulation: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_qd: wp.array[float],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    body_I_m: wp.array[wp.spatial_matrix],
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    joint_X_p: wp.array[wp.transform],
    articulation_origin: wp.array[wp.vec3],
    gravity: wp.array[wp.vec3],
    # outputs
    joint_S_s: wp.array[wp.spatial_vector],
    body_I_s: wp.array[wp.spatial_matrix],
    body_v_s: wp.array[wp.spatial_vector],
    body_f_s: wp.array[wp.spatial_vector],
    body_a_s: wp.array[wp.spatial_vector],
):
    # one thread per-articulation
    index = wp.tid()

    start = articulation_start[index]
    end = articulation_start[index + 1]

    # compute link velocities and coriolis forces
    for i in range(start, end):
        compute_link_velocity(
            i,
            joint_type,
            joint_parent,
            joint_child,
            joint_articulation,
            joint_qd_start,
            joint_qd,
            joint_axis,
            joint_dof_dim,
            body_I_m,
            body_q,
            body_q_com,
            joint_X_p,
            articulation_origin,
            gravity,
            joint_S_s,
            body_I_s,
            body_v_s,
            body_f_s,
            body_a_s,
        )


@wp.kernel
def eval_rigid_tau(
    articulation_start: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_articulation: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_f: wp.array[float],
    joint_S_s: wp.array[wp.spatial_vector],
    body_fb_s: wp.array[wp.spatial_vector],
    body_f_ext: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    articulation_origin: wp.array[wp.vec3],
    # outputs
    body_ft_s: wp.array[wp.spatial_vector],
    tau: wp.array[float],
):
    # one thread per-articulation
    index = wp.tid()

    start = articulation_start[index]
    end = articulation_start[index + 1]
    count = end - start

    # compute joint forces
    for offset in range(count):
        # for backwards traversal
        i = end - offset - 1

        type = joint_type[i]
        parent = joint_parent[i]
        child = joint_child[i]
        articulation = joint_articulation[i]
        dof_start = joint_qd_start[i]
        lin_axis_count = joint_dof_dim[i, 0]
        ang_axis_count = joint_dof_dim[i, 1]
        origin = wp.vec3()
        if articulation >= 0:
            origin = articulation_origin[articulation]

        # body forces in Featherstone frame (origin)
        f_b_s = body_fb_s[child]
        f_t_s = body_ft_s[child]

        # external wrench is provided at COM in world frame; shift torque to origin
        f_ext_com = body_f_ext[child]
        f_ext_f = wp.spatial_bottom(f_ext_com)
        f_ext_t = wp.spatial_top(f_ext_com)

        X_wb = body_q[child]
        com_local = body_com[child]
        com_world = wp.transform_point(X_wb, com_local)
        com_rel = com_world - origin
        tau_origin = f_ext_f + wp.cross(com_rel, f_ext_t)
        f_ext_origin = wp.spatial_vector(f_ext_t, tau_origin)

        # subtract external wrench to get net wrench on body
        f_s = f_b_s + f_t_s - f_ext_origin

        # compute joint-space forces, writes out tau
        jcalc_tau(
            type,
            joint_S_s,
            joint_f,
            dof_start,
            lin_axis_count,
            ang_axis_count,
            f_s,
            tau,
        )

        if parent >= 0:
            # update parent forces, todo: check that this is valid for the backwards pass
            wp.atomic_add(body_ft_s, parent, f_s)


@wp.kernel
def compute_body_parent_f(
    body_q_com: wp.array[wp.transform],
    body_f_s: wp.array[wp.spatial_vector],
    body_ft_s: wp.array[wp.spatial_vector],
    body_f_ext: wp.array[wp.spatial_vector],
    body_to_articulation: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_parent_f: wp.array[wp.spatial_vector],
):
    """Populate ``State.body_parent_f`` from FeatherPGS' RNEA backward pass."""
    tid = wp.tid()

    art = body_to_articulation[tid]
    origin = wp.vec3()
    if art >= 0:
        origin = articulation_origin[art]

    com_world = wp.transform_get_translation(body_q_com[tid])

    f_ext_com = body_f_ext[tid]
    f_ext_f = wp.spatial_bottom(f_ext_com)
    f_ext_t = wp.spatial_top(f_ext_com)
    com_rel = com_world - origin
    tau_origin = f_ext_f + wp.cross(com_rel, f_ext_t)
    f_ext_origin = wp.spatial_vector(f_ext_t, tau_origin)

    f_s = body_f_s[tid] + body_ft_s[tid] - f_ext_origin
    f_lin = wp.spatial_top(f_s)
    f_ang_at_origin = wp.spatial_bottom(f_s)
    f_ang_at_com = f_ang_at_origin - wp.cross(com_rel, f_lin)

    body_parent_f[tid] = wp.spatial_vector(f_lin, f_ang_at_com)


@wp.kernel
def eval_rigid_mass(
    articulation_start: wp.array[int],
    articulation_M_start: wp.array[int],
    mass_update_mask: wp.array[int],
    body_I_s: wp.array[wp.spatial_matrix],
    # outputs
    M_blocks: wp.array[float],
):
    # one thread per-articulation
    index = wp.tid()

    if mass_update_mask[index] == 0:
        return

    joint_start = articulation_start[index]
    joint_end = articulation_start[index + 1]
    joint_count = joint_end - joint_start

    M_offset = articulation_M_start[index]

    for l in range(joint_count):
        I = body_I_s[joint_start + l]
        block = M_offset + l * 36
        for row in range(6):
            for col in range(6):
                M_blocks[block + row * 6 + col] = I[row, col]


@wp.kernel
def compute_composite_inertia(
    articulation_start: wp.array[int],
    mass_update_mask: wp.array[int],
    joint_ancestor: wp.array[int],
    body_I_s: wp.array[wp.spatial_matrix],
    # outputs
    body_I_c: wp.array[wp.spatial_matrix],
):
    art_idx = wp.tid()

    if mass_update_mask[art_idx] == 0:
        return

    start = articulation_start[art_idx]
    end = articulation_start[art_idx + 1]
    count = end - start

    for i in range(count):
        idx = start + i
        body_I_c[idx] = body_I_s[idx]

    for i in range(count - 1, -1, -1):
        child_idx = start + i
        parent_idx = joint_ancestor[child_idx]

        if parent_idx >= start:
            body_I_c[parent_idx] += body_I_c[child_idx]


@wp.kernel
def integrate_generalized_joints(
    joint_type: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    body_com: wp.array[wp.vec3],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_qdd: wp.array[float],
    dt: float,
    # outputs
    joint_q_new: wp.array[float],
    joint_qd_new: wp.array[float],
):
    # one thread per-articulation
    index = wp.tid()

    type = joint_type[index]
    child = joint_child[index]
    coord_start = joint_q_start[index]
    dof_start = joint_qd_start[index]
    lin_axis_count = joint_dof_dim[index, 0]
    ang_axis_count = joint_dof_dim[index, 1]

    jcalc_integrate(
        type,
        child,
        body_com,
        joint_q,
        joint_qd,
        joint_qdd,
        coord_start,
        dof_start,
        lin_axis_count,
        ang_axis_count,
        dt,
        joint_q_new,
        joint_qd_new,
    )


@wp.kernel
def compute_velocity_predictor(
    joint_qd: wp.array[float],
    joint_qdd: wp.array[float],
    dt: float,
    # outputs
    v_hat: wp.array[float],
):
    tid = wp.tid()
    v_hat[tid] = joint_qd[tid] + joint_qdd[tid] * dt


@wp.kernel
def update_qdd_from_velocity(
    joint_qd: wp.array[float],
    v_new: wp.array[float],
    inv_dt: float,
    # outputs
    joint_qdd: wp.array[float],
):
    tid = wp.tid()
    joint_qdd[tid] = (v_new[tid] - joint_qd[tid]) * inv_dt


@wp.func
def contact_tangent_basis(n: wp.vec3):
    # pick an arbitrary perpendicular vector and orthonormalize
    tangent0 = wp.cross(n, wp.vec3(1.0, 0.0, 0.0))
    if wp.length_sq(tangent0) < 1.0e-12:
        tangent0 = wp.cross(n, wp.vec3(0.0, 1.0, 0.0))
    tangent0 = wp.normalize(tangent0)
    tangent1 = wp.normalize(wp.cross(n, tangent0))
    return tangent0, tangent1


@wp.kernel
def compute_contact_linear_force_from_impulses(
    contact_count: wp.array[wp.int32],
    contact_normal: wp.array[wp.vec3],
    contact_world: wp.array[wp.int32],
    contact_slot: wp.array[wp.int32],
    contact_path: wp.array[wp.int32],
    world_impulses: wp.array2d[wp.float32],
    mf_impulses: wp.array2d[wp.float32],
    enable_friction: int,
    inv_dt: float,
    # outputs
    rigid_contact_force: wp.array[wp.vec3],
):
    """Convert solved FeatherPGS contact impulses into world-frame forces."""
    c = wp.tid()
    total_contacts = contact_count[0]
    if c >= total_contacts:
        return

    force = wp.vec3(0.0)
    slot = contact_slot[c]
    path = contact_path[c]

    if slot >= 0 and path >= 0 and inv_dt > 0.0:
        world = contact_world[c]
        # Contacts store normals from shape 0 toward shape 1 (A-to-B). FeatherPGS
        # solves along the opposite direction internally, which corresponds to the
        # force on shape/body 0 from shape/body 1.
        normal = -contact_normal[c]

        lam_n = 0.0
        lam_t0 = 0.0
        lam_t1 = 0.0
        if path == 0:
            lam_n = world_impulses[world, slot]
            if enable_friction != 0:
                lam_t0 = world_impulses[world, slot + 1]
                lam_t1 = world_impulses[world, slot + 2]
        elif path == 1:
            lam_n = mf_impulses[world, slot]
            if enable_friction != 0:
                lam_t0 = mf_impulses[world, slot + 1]
                lam_t1 = mf_impulses[world, slot + 2]

        force = lam_n * normal
        if enable_friction != 0:
            tangent0, tangent1 = contact_tangent_basis(normal)
            force += lam_t0 * tangent0 + lam_t1 * tangent1
        force *= inv_dt

    rigid_contact_force[c] = force


@wp.kernel
def pack_contact_linear_force_as_spatial(
    contact_count: wp.array[wp.int32],
    rigid_contact_force: wp.array[wp.vec3],
    # outputs
    contact_force: wp.array[wp.spatial_vector],
):
    """Pack linear contact forces into Newton's spatial-force contact buffer."""
    c = wp.tid()
    total_contacts = contact_count[0]
    if c >= total_contacts:
        return

    contact_force[c] = wp.spatial_vector(rigid_contact_force[c], wp.vec3(0.0))


# Computes J*v contribution on the fly by walking the tree
# This keeps the S vectors in L2 cache and avoids reading the large J matrix.
@wp.func
def accumulate_contact_jacobian_matrix_free(
    articulation: int,
    body_index: int,
    weight: float,
    point_world: wp.vec3,
    n_vec: wp.vec3,
    body_to_joint: wp.array[int],
    body_to_articulation: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    articulation_origin: wp.array[wp.vec3],
    articulation_dof_start: int,
    # Outputs
    row_base_index: int,
    Jc_out: wp.array[float],
):
    if body_index < 0:
        return

    origin = articulation_origin[articulation]
    point_rel = point_world - origin

    curr_joint = body_to_joint[body_index]

    while curr_joint != -1:
        dof_start = joint_qd_start[curr_joint]
        dof_end = joint_qd_start[curr_joint + 1]
        dof_count = dof_end - dof_start

        for k in range(dof_count):
            global_dof = dof_start + k

            S = joint_S_s[global_dof]

            linear = wp.vec3(S[0], S[1], S[2])
            angular = wp.vec3(S[3], S[4], S[5])

            lin_vel_at_point = linear + wp.cross(angular, point_rel)
            proj = wp.dot(n_vec, lin_vel_at_point)

            local_dof = global_dof - articulation_dof_start

            Jc_out[row_base_index + local_dof] += weight * proj

        curr_joint = joint_ancestor[curr_joint]


@wp.kernel
def build_contact_rows_normal(
    contact_count: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_material_mu: wp.array[float],
    articulation_start: wp.array[int],
    articulation_H_rows: wp.array[int],
    articulation_dof_start: wp.array[int],
    body_to_joint: wp.array[int],
    body_to_articulation: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    articulation_origin: wp.array[wp.vec3],
    max_constraints: int,
    max_dofs: int,
    contact_beta: float,
    contact_cfm: float,
    enable_friction: int,
    # Outputs
    constraint_counts: wp.array[int],
    Jc_out: wp.array[float],
    phi_out: wp.array[float],
    row_beta: wp.array[float],
    row_cfm: wp.array[float],
    row_types: wp.array[int],
    target_velocity: wp.array[float],
    row_parent: wp.array[int],
    row_mu: wp.array[float],
):
    tid = wp.tid()
    total_contacts = contact_count[0]
    if tid >= total_contacts:
        return

    # contact normal stored as A-to-B; negate to get B-to-A used internally
    n = -contact_normal[tid]
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]

    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    articulation_a = -1
    articulation_b = -1
    if body_a >= 0:
        articulation_a = body_to_articulation[body_a]
    if body_b >= 0:
        articulation_b = body_to_articulation[body_b]

    articulation = articulation_a
    if articulation < 0:
        articulation = articulation_b
    elif articulation_b >= 0 and articulation_b != articulation:
        return
    if articulation < 0:
        return

    thickness_a = contact_thickness0[tid]
    thickness_b = contact_thickness1[tid]
    mu = 0.0
    mat_count = 0
    if shape_a >= 0:
        mu += shape_material_mu[shape_a]
        mat_count += 1
    if shape_b >= 0:
        mu += shape_material_mu[shape_b]
        mat_count += 1
    if mat_count > 0:
        mu /= float(mat_count)

    point_a_local = contact_point0[tid]
    point_b_local = contact_point1[tid]
    point_a_world = wp.vec3(0.0)
    point_b_world = wp.vec3(0.0)

    if body_a >= 0:
        X_wb_a = body_q[body_a]  # World-from-Body transform
        # Contact points are stored in body frame by collision detection
        point_a_world = wp.transform_point(X_wb_a, point_a_local) - thickness_a * n
    else:
        point_a_world = point_a_local - thickness_a * n

    if body_b >= 0:
        X_wb_b = body_q[body_b]  # World-from-Body transform
        # Contact points are stored in body frame by collision detection
        point_b_world = wp.transform_point(X_wb_b, point_b_local) + thickness_b * n
    else:
        point_b_world = point_b_local + thickness_b * n

    phi = wp.dot(n, point_a_world - point_b_world)

    if phi > 0.001:
        return

    # Determine upfront if we'll add friction rows (needed for atomic slot allocation)
    dof_count = articulation_H_rows[articulation]
    will_add_friction = enable_friction != 0 and mu > 0.0 and dof_count > 0

    # Allocate all slots (normal + 2 friction) in a single atomic operation
    # This guarantees contiguous layout: [normal, friction1, friction2]
    slots_needed = 3 if will_add_friction else 1
    base_slot = wp.atomic_add(constraint_counts, articulation, slots_needed)

    # Check for overflow (all slots must fit)
    if base_slot + slots_needed > max_constraints:
        return

    art_dof_start = articulation_dof_start[articulation]

    # --- Normal contact row at base_slot ---
    phi_index = articulation * max_constraints + base_slot
    phi_out[phi_index] = phi
    row_beta[phi_index] = contact_beta
    row_cfm[phi_index] = contact_cfm
    row_types[phi_index] = PGS_CONSTRAINT_TYPE_CONTACT
    target_velocity[phi_index] = 0.0
    row_parent[phi_index] = -1
    row_mu[phi_index] = mu

    row_base = phi_index * max_dofs
    for col in range(max_dofs):
        Jc_out[row_base + col] = 0.0

    accumulate_contact_jacobian_matrix_free(
        articulation,
        body_a,
        1.0,
        point_a_world,
        n,
        body_to_joint,
        body_to_articulation,
        joint_ancestor,
        joint_qd_start,
        joint_S_s,
        articulation_origin,
        art_dof_start,
        row_base,
        Jc_out,
    )

    accumulate_contact_jacobian_matrix_free(
        articulation,
        body_b,
        -1.0,
        point_b_world,
        n,
        body_to_joint,
        body_to_articulation,
        joint_ancestor,
        joint_qd_start,
        joint_S_s,
        articulation_origin,
        art_dof_start,
        row_base,
        Jc_out,
    )

    # --- Friction rows at base_slot + 1 and base_slot + 2 ---
    if will_add_friction:
        t0, t1 = contact_tangent_basis(n)

        # Friction row 1 at base_slot + 1
        row_index_1 = articulation * max_constraints + base_slot + 1
        tangent_base_1 = row_index_1 * max_dofs

        for col in range(max_dofs):
            Jc_out[tangent_base_1 + col] = 0.0

        row_beta[row_index_1] = 0.0
        row_cfm[row_index_1] = contact_cfm
        row_types[row_index_1] = PGS_CONSTRAINT_TYPE_FRICTION
        target_velocity[row_index_1] = 0.0
        phi_out[row_index_1] = 0.0
        row_parent[row_index_1] = phi_index
        row_mu[row_index_1] = mu

        accumulate_contact_jacobian_matrix_free(
            articulation,
            body_a,
            1.0,
            point_a_world,
            t0,
            body_to_joint,
            body_to_articulation,
            joint_ancestor,
            joint_qd_start,
            joint_S_s,
            articulation_origin,
            art_dof_start,
            tangent_base_1,
            Jc_out,
        )

        accumulate_contact_jacobian_matrix_free(
            articulation,
            body_b,
            -1.0,
            point_b_world,
            t0,
            body_to_joint,
            body_to_articulation,
            joint_ancestor,
            joint_qd_start,
            joint_S_s,
            articulation_origin,
            art_dof_start,
            tangent_base_1,
            Jc_out,
        )

        # Friction row 2 at base_slot + 2
        row_index_2 = articulation * max_constraints + base_slot + 2
        tangent_base_2 = row_index_2 * max_dofs

        for col in range(max_dofs):
            Jc_out[tangent_base_2 + col] = 0.0

        row_beta[row_index_2] = 0.0
        row_cfm[row_index_2] = contact_cfm
        row_types[row_index_2] = PGS_CONSTRAINT_TYPE_FRICTION
        target_velocity[row_index_2] = 0.0
        phi_out[row_index_2] = 0.0
        row_parent[row_index_2] = phi_index
        row_mu[row_index_2] = mu

        accumulate_contact_jacobian_matrix_free(
            articulation,
            body_a,
            1.0,
            point_a_world,
            t1,
            body_to_joint,
            body_to_articulation,
            joint_ancestor,
            joint_qd_start,
            joint_S_s,
            articulation_origin,
            art_dof_start,
            tangent_base_2,
            Jc_out,
        )

        accumulate_contact_jacobian_matrix_free(
            articulation,
            body_b,
            -1.0,
            point_b_world,
            t1,
            body_to_joint,
            body_to_articulation,
            joint_ancestor,
            joint_qd_start,
            joint_S_s,
            articulation_origin,
            art_dof_start,
            tangent_base_2,
            Jc_out,
        )


@wp.kernel
def build_augmented_joint_rows(
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    articulation_H_rows: wp.array[int],
    joint_type: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_target_pos: wp.array[float],
    joint_target_vel: wp.array[float],
    max_dofs: int,
    dt: float,
    # outputs
    row_counts: wp.array[int],
    row_dof_index: wp.array[int],
    row_K: wp.array[float],
    row_u0: wp.array[float],
    limit_counts: wp.array[int],
):
    articulation = wp.tid()
    if max_dofs == 0:
        row_counts[articulation] = 0
        limit_counts[articulation] = 0
        return

    dof_count = articulation_H_rows[articulation]
    if dof_count == 0:
        row_counts[articulation] = 0
        limit_counts[articulation] = 0
        return

    joint_start = articulation_start[articulation]
    joint_end = articulation_start[articulation + 1]

    slot = int(0)
    limit_counts[articulation] = 0

    for joint_index in range(joint_start, joint_end):
        type = joint_type[joint_index]
        if type != JointType.PRISMATIC and type != JointType.REVOLUTE and type != JointType.D6:
            continue

        lin_axis_count = joint_dof_dim[joint_index, 0]
        ang_axis_count = joint_dof_dim[joint_index, 1]
        axis_count = lin_axis_count + ang_axis_count

        qd_start = joint_qd_start[joint_index]
        coord_start = joint_q_start[joint_index]

        for axis in range(axis_count):
            if slot >= max_dofs:
                break
            dof_index = qd_start + axis
            coord_index = coord_start + axis

            ke = joint_target_ke[dof_index]
            kd = joint_target_kd[dof_index]
            if ke <= 0.0 and kd <= 0.0:
                continue

            K = ke * dt * dt + kd * dt
            if K <= 0.0:
                continue

            row_index = articulation * max_dofs + slot
            row_dof_index[row_index] = dof_index
            q = joint_q[coord_index]
            qd_val = joint_qd[dof_index]
            target_pos = joint_target_pos[dof_index]
            target_vel = joint_target_vel[dof_index]
            u0 = -(ke * (q - target_pos + dt * qd_val) + kd * (qd_val - target_vel))
            row_K[row_index] = K
            row_u0[row_index] = u0

            slot += 1
            if slot >= max_dofs:
                break

    row_counts[articulation] = slot
    limit_counts[articulation] = 0


@wp.kernel
def detect_limit_count_changes(
    limit_counts: wp.array[int],
    prev_limit_counts: wp.array[int],
    # outputs
    limit_change_mask: wp.array[int],
):
    tid = wp.tid()
    change = 1 if limit_counts[tid] != prev_limit_counts[tid] else 0
    limit_change_mask[tid] = change


@wp.kernel
def build_mass_update_mask(
    global_flag: int,
    limit_change_mask: wp.array[int],
    # outputs
    mass_update_mask: wp.array[int],
):
    tid = wp.tid()
    flag = 1 if global_flag != 0 else 0
    if limit_change_mask[tid] != 0:
        flag = 1
    mass_update_mask[tid] = flag


# =============================================================================
# Joint Limit Constraint Kernels
# =============================================================================


@wp.kernel
def allocate_joint_limit_slots(
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    articulation_H_rows: wp.array[int],
    joint_type: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_q: wp.array[float],
    art_to_world: wp.array[int],
    max_constraints: int,
    # outputs
    limit_slot: wp.array[int],
    limit_sign: wp.array[float],
    world_slot_counter: wp.array[int],
):
    """Allocate constraint slots for violated joint limits.

    For each articulation, checks all DOFs of PRISMATIC, REVOLUTE, and D6
    joints against their limits.  When a DOF violates its lower or upper
    limit, a single constraint slot is atomically reserved in the world's
    slot counter (the same counter used by contacts).

    Outputs per-DOF arrays ``limit_slot`` (world-constraint row, or -1) and
    ``limit_sign`` (+1 for lower-limit violation, -1 for upper).
    """
    art = wp.tid()
    world = art_to_world[art]

    # Initialize all DOFs of this articulation to "no limit active"
    dof_base = articulation_dof_start[art]
    dof_count = articulation_H_rows[art]
    for d in range(dof_count):
        limit_slot[dof_base + d] = -1
        limit_sign[dof_base + d] = 0.0

    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]

    for j in range(joint_start, joint_end):
        jtype = joint_type[j]
        if jtype != JointType.PRISMATIC and jtype != JointType.REVOLUTE and jtype != JointType.D6:
            continue

        lin_count = joint_dof_dim[j, 0]
        ang_count = joint_dof_dim[j, 1]
        axis_count = lin_count + ang_count
        qd_start = joint_qd_start[j]
        q_start = joint_q_start[j]

        for axis in range(axis_count):
            dof = qd_start + axis
            q_val = joint_q[q_start + axis]
            lower = joint_limit_lower[dof]
            upper = joint_limit_upper[dof]

            # Lower limit violation (q < lower)
            if q_val < lower:
                slot = wp.atomic_add(world_slot_counter, world, 1)
                if slot < max_constraints:
                    limit_slot[dof] = slot
                    limit_sign[dof] = 1.0
            # Upper limit violation (q > upper)
            elif q_val > upper:
                slot = wp.atomic_add(world_slot_counter, world, 1)
                if slot < max_constraints:
                    limit_slot[dof] = slot
                    limit_sign[dof] = -1.0


@wp.kernel
def populate_joint_limit_J_for_size(
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    joint_type: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_q: wp.array[float],
    art_to_world: wp.array[int],
    limit_slot: wp.array[int],
    limit_sign: wp.array[float],
    group_to_art: wp.array[int],
    pgs_beta: float,
    pgs_cfm: float,
    # outputs
    J_group: wp.array3d[float],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_cfm: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
):
    """Populate Jacobian and metadata for joint limit constraints.

    Launched once per size group with ``dim = n_arts_of_size``.  Each thread
    walks the joints of one articulation and, for every DOF whose
    ``limit_slot`` is non-negative (i.e. the limit was activated by
    :func:`allocate_joint_limit_slots`), writes:

    * A single ±1 entry in the Jacobian at the DOF's local column.
    * Constraint metadata (type, phi, beta, cfm, etc.).
    """
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    world = art_to_world[art]
    dof_start = articulation_dof_start[art]

    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]

    for j in range(joint_start, joint_end):
        jtype = joint_type[j]
        if jtype != JointType.PRISMATIC and jtype != JointType.REVOLUTE and jtype != JointType.D6:
            continue

        lin_count = joint_dof_dim[j, 0]
        ang_count = joint_dof_dim[j, 1]
        axis_count = lin_count + ang_count
        qd_start = joint_qd_start[j]
        q_start = joint_q_start[j]

        for axis in range(axis_count):
            dof = qd_start + axis
            slot = limit_slot[dof]
            if slot < 0:
                continue

            sign = limit_sign[dof]
            q_val = joint_q[q_start + axis]
            lower = joint_limit_lower[dof]
            upper = joint_limit_upper[dof]

            # phi is negative when violating
            phi = 0.0
            if sign > 0.0:
                phi = q_val - lower
            else:
                phi = upper - q_val

            # Jacobian: single ±1 entry at the local DOF column
            local_dof = dof - dof_start
            J_group[group_idx, slot, local_dof] = sign

            # Constraint metadata
            world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_JOINT_LIMIT
            world_row_parent[world, slot] = -1
            world_row_mu[world, slot] = 0.0
            world_row_beta[world, slot] = pgs_beta
            world_row_cfm[world, slot] = pgs_cfm
            world_phi[world, slot] = phi
            world_target_velocity[world, slot] = 0.0


# =============================================================================
# Joint Velocity-Limit Constraint Kernels
# =============================================================================
# These kernels mirror the PhysX per-DOF velocity-limit formulation documented
# in ``notes/investigations/velocity-spike/physx-deep-dive.md`` §4 and the math
# appendix. They reuse the same allocation / populate shape as the
# joint-position-limit kernels above, but the activation test is on ``qdot_i``
# against ``model.joint_velocity_limit[i]`` and there is no Baumgarte bias.


@wp.kernel
def allocate_joint_velocity_limit_slots(
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    articulation_H_rows: wp.array[int],
    joint_type: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_velocity_limit: wp.array[float],
    joint_qd: wp.array[float],
    art_to_world: wp.array[int],
    max_constraints: int,
    # outputs
    velocity_limit_slot: wp.array[int],
    velocity_limit_sign: wp.array[float],
    world_slot_counter: wp.array[int],
):
    """Allocate constraint slots for DOFs that exceed their velocity limit.

    Mirrors :func:`allocate_joint_limit_slots` for the PhysX velocity-limit
    formulation. For each non-locked DOF of a PRISMATIC / REVOLUTE / D6 joint
    a slot is atomically reserved in the per-world counter iff
    ``|qdot_i| > joint_velocity_limit[i]``. The sign encodes which side of
    the bilateral box ``[-qdot_max, +qdot_max]`` is violated so the
    corresponding row can be written as a single signed ±1 entry in the
    grouped Jacobian:

    * ``sign = +1`` — lower-limit violation (``qdot_i < -qdot_max``). The row
      pushes velocity back up (``J = +e_i``, ``target_vel = -qdot_max``).
    * ``sign = -1`` — upper-limit violation (``qdot_i > +qdot_max``). The row
      pushes velocity back down (``J = -e_i``, ``target_vel = -qdot_max``).

    The unilateral PGS projector (``lambda >= 0``) combined with the sign
    flip on ``J`` reproduces PhysX's bilateral projection onto the box, at
    most one side at a time.

    Outputs per-DOF arrays ``velocity_limit_slot`` (world-constraint row, or
    -1) and ``velocity_limit_sign`` (+1 / -1).
    """
    art = wp.tid()
    world = art_to_world[art]

    # Initialize all DOFs of this articulation to "no limit active"
    dof_base = articulation_dof_start[art]
    dof_count = articulation_H_rows[art]
    for d in range(dof_count):
        velocity_limit_slot[dof_base + d] = -1
        velocity_limit_sign[dof_base + d] = 0.0

    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]

    for j in range(joint_start, joint_end):
        jtype = joint_type[j]
        if jtype != JointType.PRISMATIC and jtype != JointType.REVOLUTE and jtype != JointType.D6:
            continue

        lin_count = joint_dof_dim[j, 0]
        ang_count = joint_dof_dim[j, 1]
        axis_count = lin_count + ang_count
        qd_start = joint_qd_start[j]

        for axis in range(axis_count):
            dof = qd_start + axis
            qdot = joint_qd[dof]
            qdot_max = joint_velocity_limit[dof]

            # Guard against degenerate limits. PhysX pins ``recipResponse``
            # off for ``unitResponse <= 0``; here we drop the row entirely if
            # the stored limit is non-positive (treated as "unlimited").
            if qdot_max <= 0.0:
                continue

            if qdot > qdot_max:
                slot = wp.atomic_add(world_slot_counter, world, 1)
                if slot < max_constraints:
                    velocity_limit_slot[dof] = slot
                    velocity_limit_sign[dof] = -1.0
            elif qdot < -qdot_max:
                slot = wp.atomic_add(world_slot_counter, world, 1)
                if slot < max_constraints:
                    velocity_limit_slot[dof] = slot
                    velocity_limit_sign[dof] = 1.0


@wp.kernel
def populate_joint_velocity_limit_J_for_size(
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    joint_type: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_velocity_limit: wp.array[float],
    art_to_world: wp.array[int],
    velocity_limit_slot: wp.array[int],
    velocity_limit_sign: wp.array[float],
    group_to_art: wp.array[int],
    pgs_cfm: float,
    # outputs
    J_group: wp.array3d[float],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_cfm: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
):
    """Populate Jacobian and metadata for joint velocity-limit rows.

    Launched once per size group with ``dim = n_arts_of_size``. For every DOF
    whose ``velocity_limit_slot`` is non-negative, writes a single signed ±1
    entry into the local DOF column of the grouped Jacobian and sets the
    constraint metadata. The row has **no Baumgarte bias** (``beta = 0``,
    ``phi = 0``) — PhysX's velocity-limit row has no ``data.erp`` either.
    The target velocity is ``-qdot_max`` for both sides of the box: combined
    with the sign flip on ``J``, this encodes the bilateral projection as a
    unilateral ``J*v >= target_vel`` row with ``lambda >= 0``.
    """
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    world = art_to_world[art]
    dof_start = articulation_dof_start[art]

    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]

    for j in range(joint_start, joint_end):
        jtype = joint_type[j]
        if jtype != JointType.PRISMATIC and jtype != JointType.REVOLUTE and jtype != JointType.D6:
            continue

        lin_count = joint_dof_dim[j, 0]
        ang_count = joint_dof_dim[j, 1]
        axis_count = lin_count + ang_count
        qd_start = joint_qd_start[j]

        for axis in range(axis_count):
            dof = qd_start + axis
            slot = velocity_limit_slot[dof]
            if slot < 0:
                continue

            sign = velocity_limit_sign[dof]
            qdot_max = joint_velocity_limit[dof]

            # Single signed ±1 entry at the local DOF column. The selector
            # row on generalised velocity is ``J = sign * e_i``; the
            # articulated-body response ``J M^-1 J^T`` is exactly PhysX's
            # ``recipResponse`` on the same axis and is computed by the
            # tiled H^-1 J^T / ``diag_from_JY_par_art`` path.
            local_dof = dof - dof_start
            J_group[group_idx, slot, local_dof] = sign

            world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT
            world_row_parent[world, slot] = -1
            world_row_mu[world, slot] = 0.0
            # No Baumgarte / ERP — matches PhysX vel-limit row (§4).
            world_row_beta[world, slot] = 0.0
            world_row_cfm[world, slot] = pgs_cfm
            world_phi[world, slot] = 0.0
            # ``target_vel = -qdot_max`` for both signs: rhs = -target + J*v
            # = qdot_max +/- qdot_i, which is negative exactly when the
            # corresponding side of the box is violated.
            world_target_velocity[world, slot] = -qdot_max


# =============================================================================
# Multi-Articulation Contact Building Kernels
# =============================================================================
# These kernels enable contacts between multiple articulations within the same
# world. The constraint system becomes world-level instead of per-articulation.


@wp.kernel
def allocate_world_contact_slots(
    contact_count: wp.array[int],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
    is_free_rigid: wp.array[int],
    has_free_rigid: int,
    max_constraints: int,
    mf_max_constraints: int,
    enable_friction: int,
    # outputs
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    world_slot_counter: wp.array[int],
    contact_path: wp.array[int],
    mf_slot_counter: wp.array[int],
):
    """
    Phase 1 of multi-articulation contact building.

    Allocates world-level constraint slots for each contact and records
    which articulations are involved. Contacts where both sides are free
    rigid bodies (or ground) are routed to the matrix-free path.

    Each contact reserves 3 slots (normal + 2 friction) in its world's constraint buffer.
    """
    c = wp.tid()
    total_contacts = contact_count[0]
    if c >= total_contacts:
        contact_slot[c] = -1
        contact_path[c] = -1
        return

    shape_a = contact_shape0[c]
    shape_b = contact_shape1[c]

    # Get bodies and articulations
    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    art_a = -1
    art_b = -1
    if body_a >= 0:
        art_a = body_to_articulation[body_a]
    if body_b >= 0:
        art_b = body_to_articulation[body_b]

    # Determine world (both bodies must be in same world, or one is ground)
    world = -1
    if art_a >= 0:
        world = art_to_world[art_a]
    if art_b >= 0:
        world_b = art_to_world[art_b]
        if world >= 0 and world_b != world:
            # Cross-world contact - shouldn't happen, skip
            contact_slot[c] = -1
            contact_path[c] = -1
            return
        world = world_b

    if world < 0:
        # No articulation involved (ground-ground?)
        contact_slot[c] = -1
        contact_path[c] = -1
        return

    # Compute phi (same logic as populate_world_J_for_size)
    # contact normal stored as A-to-B; negate to get B-to-A used internally
    normal = -contact_normal[c]
    point_a_local = contact_point0[c]
    point_b_local = contact_point1[c]
    thickness_a = contact_thickness0[c]
    thickness_b = contact_thickness1[c]

    point_a_world = wp.vec3(0.0)
    point_b_world = wp.vec3(0.0)

    if body_a >= 0:
        X_wb_a = body_q[body_a]
        # Contact points are stored in body frame by collision detection
        point_a_world = wp.transform_point(X_wb_a, point_a_local) - thickness_a * normal
    else:
        point_a_world = point_a_local - thickness_a * normal

    if body_b >= 0:
        X_wb_b = body_q[body_b]
        # Contact points are stored in body frame by collision detection
        point_b_world = wp.transform_point(X_wb_b, point_b_local) + thickness_b * normal
    else:
        point_b_world = point_b_local + thickness_b * normal
    phi = wp.dot(normal, point_a_world - point_b_world)

    # Gate on margin
    if phi >= 0.001:
        contact_slot[c] = -1
        contact_path[c] = -1
        return

    # Classify: MF path if both sides are free rigid or ground
    is_mf = 0
    if has_free_rigid != 0:
        a_is_free_or_ground = (art_a < 0) or (is_free_rigid[art_a] != 0)
        b_is_free_or_ground = (art_b < 0) or (is_free_rigid[art_b] != 0)
        if a_is_free_or_ground and b_is_free_or_ground:
            is_mf = 1

    # Allocate slots (1 normal + 2 friction)
    slots_needed = 1
    if enable_friction != 0:
        slots_needed = 3

    if is_mf != 0:
        # Matrix-free path
        slot = wp.atomic_add(mf_slot_counter, world, slots_needed)
        if slot + slots_needed > mf_max_constraints:
            # Roll back the counter so finalize sees only filled slots
            wp.atomic_add(mf_slot_counter, world, -slots_needed)
            contact_slot[c] = -1
            contact_path[c] = -1
            return
        contact_world[c] = world
        contact_slot[c] = slot
        contact_art_a[c] = art_a
        contact_art_b[c] = art_b
        contact_path[c] = 1
    else:
        # Dense path
        slot = wp.atomic_add(world_slot_counter, world, slots_needed)
        if slot + slots_needed > max_constraints:
            # Roll back the counter so finalize sees only filled slots
            wp.atomic_add(world_slot_counter, world, -slots_needed)
            contact_slot[c] = -1
            contact_path[c] = -1
            return
        contact_world[c] = world
        contact_slot[c] = slot
        contact_art_a[c] = art_a
        contact_art_b[c] = art_b
        contact_path[c] = 0


@wp.func
def accumulate_jacobian_row_world(
    body_index: int,
    sign: float,
    point_world: wp.vec3,
    origin: wp.vec3,
    direction: wp.vec3,
    body_to_joint: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    art_dof_start: int,
    n_dofs: int,
    group_idx: int,
    row: int,
    J_group: wp.array3d[float],
):
    """Accumulate Jacobian contributions by walking up the kinematic tree."""
    if body_index < 0:
        return

    point_rel = point_world - origin
    curr_joint = body_to_joint[body_index]

    while curr_joint >= 0:
        dof_start = joint_qd_start[curr_joint]
        dof_end = joint_qd_start[curr_joint + 1]

        for global_dof in range(dof_start, dof_end):
            S = joint_S_s[global_dof]
            lin = wp.vec3(S[0], S[1], S[2])
            ang = wp.vec3(S[3], S[4], S[5])

            # Velocity at contact point from this joint
            v = lin + wp.cross(ang, point_rel)
            proj = wp.dot(direction, v)

            local_dof = global_dof - art_dof_start
            if local_dof >= 0 and local_dof < n_dofs:
                J_group[group_idx, row, local_dof] += sign * proj

        curr_joint = joint_ancestor[curr_joint]


@wp.kernel
def populate_world_J_for_size(
    contact_count: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    contact_path: wp.array[int],
    target_size: int,
    art_size: wp.array[int],
    art_group_idx: wp.array[int],
    art_dof_start: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_to_joint: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_material_mu: wp.array[float],
    enable_friction: int,
    pgs_beta: float,
    pgs_cfm: float,
    # outputs
    J_group: wp.array3d[float],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_cfm: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
):
    """
    Phase 2 of multi-articulation contact building (per size group).

    Populates the Jacobian matrix for articulations of a specific DOF size.
    Each contact may contribute to multiple articulations' J matrices.
    Contacts routed to the matrix-free path (contact_path==1) are skipped.
    """
    c = wp.tid()
    total_contacts = contact_count[0]
    if c >= total_contacts:
        return

    # Skip contacts routed to MF path
    if contact_path[c] != 0:
        return

    slot = contact_slot[c]
    if slot < 0:
        return

    world = contact_world[c]
    art_a = contact_art_a[c]
    art_b = contact_art_b[c]

    # Get contact geometry
    # contact normal stored as A-to-B; negate to get B-to-A used internally
    normal = -contact_normal[c]
    shape_a = contact_shape0[c]
    shape_b = contact_shape1[c]

    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    thickness_a = contact_thickness0[c]
    thickness_b = contact_thickness1[c]

    # Compute contact points in world frame
    # Contact points are stored in body frame by collision detection
    point_a_local = contact_point0[c]
    point_b_local = contact_point1[c]
    point_a_world = wp.vec3(0.0)
    point_b_world = wp.vec3(0.0)

    if body_a >= 0:
        X_wb_a = body_q[body_a]
        point_a_world = wp.transform_point(X_wb_a, point_a_local) - thickness_a * normal
    else:
        point_a_world = point_a_local - thickness_a * normal

    if body_b >= 0:
        X_wb_b = body_q[body_b]
        point_b_world = wp.transform_point(X_wb_b, point_b_local) + thickness_b * normal
    else:
        point_b_world = point_b_local + thickness_b * normal

    # Compute penetration depth
    phi = wp.dot(normal, point_a_world - point_b_world)

    # Compute friction coefficient
    mu = 0.0
    mat_count = 0
    if shape_a >= 0:
        mu += shape_material_mu[shape_a]
        mat_count += 1
    if shape_b >= 0:
        mu += shape_material_mu[shape_b]
        mat_count += 1
    if mat_count > 0:
        mu /= float(mat_count)

    # Compute tangent basis for friction
    t0, t1 = contact_tangent_basis(normal)

    # Handle articulation A if it matches target size
    if art_a >= 0 and art_size[art_a] == target_size:
        group_idx_a = art_group_idx[art_a]
        dof_start_a = art_dof_start[art_a]
        origin_a = articulation_origin[art_a]

        # Normal row (slot + 0)
        accumulate_jacobian_row_world(
            body_a,
            1.0,
            point_a_world,
            origin_a,
            normal,
            body_to_joint,
            joint_ancestor,
            joint_qd_start,
            joint_S_s,
            dof_start_a,
            target_size,
            group_idx_a,
            slot,
            J_group,
        )

        if enable_friction != 0:
            # Friction row 1 (slot + 1)
            accumulate_jacobian_row_world(
                body_a,
                1.0,
                point_a_world,
                origin_a,
                t0,
                body_to_joint,
                joint_ancestor,
                joint_qd_start,
                joint_S_s,
                dof_start_a,
                target_size,
                group_idx_a,
                slot + 1,
                J_group,
            )
            # Friction row 2 (slot + 2)
            accumulate_jacobian_row_world(
                body_a,
                1.0,
                point_a_world,
                origin_a,
                t1,
                body_to_joint,
                joint_ancestor,
                joint_qd_start,
                joint_S_s,
                dof_start_a,
                target_size,
                group_idx_a,
                slot + 2,
                J_group,
            )

    # Handle articulation B if it matches target size
    if art_b >= 0 and art_size[art_b] == target_size:
        group_idx_b = art_group_idx[art_b]
        dof_start_b = art_dof_start[art_b]
        origin_b = articulation_origin[art_b]

        # Opposite sign for body B
        accumulate_jacobian_row_world(
            body_b,
            -1.0,
            point_b_world,
            origin_b,
            normal,
            body_to_joint,
            joint_ancestor,
            joint_qd_start,
            joint_S_s,
            dof_start_b,
            target_size,
            group_idx_b,
            slot,
            J_group,
        )

        if enable_friction != 0:
            accumulate_jacobian_row_world(
                body_b,
                -1.0,
                point_b_world,
                origin_b,
                t0,
                body_to_joint,
                joint_ancestor,
                joint_qd_start,
                joint_S_s,
                dof_start_b,
                target_size,
                group_idx_b,
                slot + 1,
                J_group,
            )
            accumulate_jacobian_row_world(
                body_b,
                -1.0,
                point_b_world,
                origin_b,
                t1,
                body_to_joint,
                joint_ancestor,
                joint_qd_start,
                joint_S_s,
                dof_start_b,
                target_size,
                group_idx_b,
                slot + 2,
                J_group,
            )

    # Set row metadata (only once per contact, from whichever articulation runs first)
    # Use art_a preferentially to avoid double-writes
    if art_a >= 0 and art_size[art_a] == target_size:
        # Normal contact row
        world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_CONTACT
        world_row_parent[world, slot] = -1
        world_row_mu[world, slot] = mu
        world_row_beta[world, slot] = pgs_beta
        world_row_cfm[world, slot] = pgs_cfm
        world_phi[world, slot] = phi
        world_target_velocity[world, slot] = 0.0

        if enable_friction != 0:
            # Friction row 1
            world_row_type[world, slot + 1] = PGS_CONSTRAINT_TYPE_FRICTION
            world_row_parent[world, slot + 1] = slot
            world_row_mu[world, slot + 1] = mu
            world_row_beta[world, slot + 1] = 0.0
            world_row_cfm[world, slot + 1] = pgs_cfm
            world_phi[world, slot + 1] = 0.0
            world_target_velocity[world, slot + 1] = 0.0

            # Friction row 2
            world_row_type[world, slot + 2] = PGS_CONSTRAINT_TYPE_FRICTION
            world_row_parent[world, slot + 2] = slot
            world_row_mu[world, slot + 2] = mu
            world_row_beta[world, slot + 2] = 0.0
            world_row_cfm[world, slot + 2] = pgs_cfm
            world_phi[world, slot + 2] = 0.0
            world_target_velocity[world, slot + 2] = 0.0

    elif art_b >= 0 and art_size[art_b] == target_size:
        # Only write metadata from art_b if art_a didn't match this size
        world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_CONTACT
        world_row_parent[world, slot] = -1
        world_row_mu[world, slot] = mu
        world_row_beta[world, slot] = pgs_beta
        world_row_cfm[world, slot] = pgs_cfm
        world_phi[world, slot] = phi
        world_target_velocity[world, slot] = 0.0

        if enable_friction != 0:
            world_row_type[world, slot + 1] = PGS_CONSTRAINT_TYPE_FRICTION
            world_row_parent[world, slot + 1] = slot
            world_row_mu[world, slot + 1] = mu
            world_row_beta[world, slot + 1] = 0.0
            world_row_cfm[world, slot + 1] = pgs_cfm
            world_phi[world, slot + 1] = 0.0
            world_target_velocity[world, slot + 1] = 0.0

            world_row_type[world, slot + 2] = PGS_CONSTRAINT_TYPE_FRICTION
            world_row_parent[world, slot + 2] = slot
            world_row_mu[world, slot + 2] = mu
            world_row_beta[world, slot + 2] = 0.0
            world_row_cfm[world, slot + 2] = pgs_cfm
            world_phi[world, slot + 2] = 0.0
            world_target_velocity[world, slot + 2] = 0.0


@wp.kernel
def finalize_world_constraint_counts(
    world_slot_counter: wp.array[int],
    max_constraints: int,
    slots_per_contact: int,
    # outputs
    world_constraint_count: wp.array[int],
):
    """Copy and clamp the slot counter to constraint counts.

    When the atomic slot counter exceeds ``max_constraints``, clamping can
    leave "gap" slots that were reserved by a rejected contact but never
    written.  Those gap slots have zero Jacobians and will be harmlessly
    skipped by PGS (zero diagonal → ``continue``).

    The ``slots_per_contact`` argument is accepted for backwards
    compatibility but is no longer used for rounding, because the
    constraint buffer may now contain a mix of 3-row contact groups and
    single-row joint-limit constraints.
    """
    world = wp.tid()
    count = world_slot_counter[world]
    if count > max_constraints:
        count = max_constraints
    world_constraint_count[world] = count


@wp.kernel
def clamp_contact_counts(
    constraint_counts: wp.array[int],
    max_constraints: int,
):
    articulation = wp.tid()
    count = constraint_counts[articulation]
    if count > max_constraints:
        constraint_counts[articulation] = max_constraints


@wp.kernel
def apply_augmented_mass_diagonal(
    articulation_H_start: wp.array[int],
    articulation_H_rows: wp.array[int],
    articulation_dof_start: wp.array[int],
    max_dofs: int,
    mass_update_mask: wp.array[int],
    row_counts: wp.array[int],
    row_dof_index: wp.array[int],
    row_K: wp.array[float],
    # outputs
    H: wp.array[float],
):
    articulation = wp.tid()
    if mass_update_mask[articulation] == 0:
        return

    n = articulation_H_rows[articulation]
    if n == 0 or max_dofs == 0:
        return

    count = row_counts[articulation]
    if count == 0:
        return

    H_start = articulation_H_start[articulation]
    dof_start = articulation_dof_start[articulation]

    for i in range(count):
        row_index = articulation * max_dofs + i
        dof = row_dof_index[row_index]
        local = dof - dof_start
        if local < 0 or local >= n:
            continue

        K = row_K[row_index]
        if K <= 0.0:
            continue

        diag_index = H_start + dense_index(n, local, local)
        H[diag_index] += K


@wp.kernel
def apply_augmented_mass_diagonal_grouped(
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    n_dofs: int,
    max_dofs: int,
    mass_update_mask: wp.array[int],
    row_counts: wp.array[int],
    row_dof_index: wp.array[int],
    row_K: wp.array[float],
    # outputs
    H_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
):
    """Apply augmented mass diagonal for grouped H storage."""
    idx = wp.tid()
    articulation = group_to_art[idx]

    if mass_update_mask[articulation] == 0:
        return

    count = row_counts[articulation]
    if count == 0:
        return

    dof_start = articulation_dof_start[articulation]

    for i in range(count):
        row_index = articulation * max_dofs + i
        dof = row_dof_index[row_index]
        local = dof - dof_start
        if local < 0 or local >= n_dofs:
            continue

        K = row_K[row_index]
        if K <= 0.0:
            continue

        H_group[idx, local, local] += K


@wp.kernel
def apply_augmented_joint_tau(
    max_dofs: int,
    row_counts: wp.array[int],
    row_dof_index: wp.array[int],
    row_u0: wp.array[float],
    # outputs
    joint_tau: wp.array[float],
):
    articulation = wp.tid()
    if max_dofs == 0:
        return

    count = row_counts[articulation]
    if count == 0:
        return

    for i in range(count):
        row_index = articulation * max_dofs + i
        dof = row_dof_index[row_index]
        u0 = row_u0[row_index]
        if u0 == 0.0:
            continue

        wp.atomic_add(joint_tau, dof, u0)


@wp.kernel
def clamp_augmented_joint_u0(
    max_dofs: int,
    row_counts: wp.array[int],
    row_dof_index: wp.array[int],
    joint_effort_limit: wp.array[float],
    # outputs
    row_u0: wp.array[float],
):
    """Actuator-drive-only effort-limit clamp.

    Clamps each augmented row's explicit PD-drive output ``u0`` to
    ``+/- joint_effort_limit[dof]`` *before* that row is accumulated into
    ``joint_tau``. The rigid / passive / Coriolis / gravity / external /
    :attr:`~newton.Control.joint_f` bucket already sitting in ``joint_tau``
    is not touched by any clamp, matching the convention used by MuJoCo's
    ``actuatorfrcrange`` and PhysX articulation drive ``maxForce``. A
    ``joint_effort_limit`` value of ``<= 0`` is treated as "unlimited".
    """
    articulation = wp.tid()
    if max_dofs == 0:
        return

    count = row_counts[articulation]
    if count == 0:
        return

    for i in range(count):
        row_index = articulation * max_dofs + i
        dof = row_dof_index[row_index]

        limit = joint_effort_limit[dof]
        if limit <= 0.0:
            continue

        u0 = row_u0[row_index]
        if u0 > limit:
            row_u0[row_index] = limit
        elif u0 < -limit:
            row_u0[row_index] = -limit


# --- Tile configuration for contact system build ---
# Kernel naming: {op}_{parallelism}
# parallelism: tiled | loop | par_row | par_row_col | par_dof

# Max generalized dofs per articulation we support in the tiled path.
# joint_dof_count per articulation must be <= TILE_DOF or we use fall back
TILE_DOF = wp.constant(49)

# Max constraints per articulation we support in the tiled path.
# Threads per tile/block for tile kernels
TILE_THREADS = 64


@wp.kernel
def update_body_qd_from_featherstone(
    body_v_s: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_to_articulation: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_qd_out: wp.array[wp.spatial_vector],
):
    tid = wp.tid()

    twist = body_v_s[tid]  # spatial twist about origin
    v0 = wp.spatial_top(twist)
    w = wp.spatial_bottom(twist)

    X_wb = body_q[tid]
    com_local = body_com[tid]
    com_world = wp.transform_point(X_wb, com_local)
    art = body_to_articulation[tid]
    origin = wp.vec3()
    if art >= 0:
        origin = articulation_origin[art]
    com_rel = com_world - origin

    v_com = v0 + wp.cross(w, com_rel)

    body_qd_out[tid] = wp.spatial_vector(v_com, w)


# =============================================================================
# World-Level PGS and Velocity Kernels for Multi-Articulation
# =============================================================================


@wp.kernel
def compute_world_contact_bias(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    world_phi: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_type: wp.array2d[int],
    world_target_velocity: wp.array2d[float],
    dt: float,
    # outputs
    world_rhs: wp.array2d[float],
):
    """Compute the RHS bias term for world-level PGS solve.

    The RHS follows the convention: rhs = J*v + stabilization
    For contacts with penetration (phi < 0): rhs = J*v + beta * phi / dt (negative)
    This leads to positive impulses when resolved by PGS.
    """
    world = wp.tid()
    m = world_constraint_count[world]

    inv_dt = 1.0 / dt

    for i in range(m):
        phi = world_phi[world, i]
        beta = world_row_beta[world, i]
        row_type = world_row_type[world, i]
        target_vel = world_target_velocity[world, i]

        # Initialize with -target_velocity (will add J*v later)
        rhs = -target_vel

        # For contacts and joint limits: add Baumgarte stabilization when violating
        if row_type == PGS_CONSTRAINT_TYPE_CONTACT or row_type == PGS_CONSTRAINT_TYPE_JOINT_LIMIT:
            if phi < 0.0:
                rhs += beta * phi * inv_dt  # Negative for penetration / violation
        elif row_type == PGS_CONSTRAINT_TYPE_JOINT_TARGET:
            rhs += beta * phi * inv_dt
        # PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT: no phi-based bias. The
        # constraint is an instantaneous velocity-space projection; the only
        # RHS contribution is ``-target_vel`` (already set above). The fused
        # PGS kernel computes ``J*v`` from the current velocity each iteration.

        world_rhs[world, i] = rhs


@wp.kernel
def prepare_world_impulses(
    max_constraints: int,
    # in/out
    world_impulses: wp.array2d[float],
):
    """Initialize world impulses to zero outside the active rows."""
    world = wp.tid()
    for i in range(max_constraints):
        world_impulses[world, i] = 0.0


# =============================================================================
# Fully Matrix-Free PGS Kernels (velocity-space Jacobi)
# =============================================================================


@wp.kernel
def diag_from_JY_par_art(
    J_group: wp.array3d[float],  # [n_arts_of_size, max_constraints, n_dofs]
    Y_group: wp.array3d[float],  # [n_arts_of_size, max_constraints, n_dofs]
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    world_constraint_count: wp.array[int],
    n_dofs: int,
    max_constraints: int,
    n_arts: int,
    # output
    world_diag: wp.array2d[float],
):
    """Compute diagonal of Delassus from J and Y without assembling the full matrix.

    diag[w,c] += sum_k J[idx,c,k] * Y[idx,c,k]. Thread dim: n_arts * max_constraints.
    """
    tid = wp.tid()
    c = tid % max_constraints
    idx = tid // max_constraints
    if idx >= n_arts:
        return
    art = group_to_art[idx]
    world = art_to_world[art]
    if c >= world_constraint_count[world]:
        return
    val = float(0.0)
    for k in range(n_dofs):
        val += J_group[idx, c, k] * Y_group[idx, c, k]
    if val != 0.0:
        wp.atomic_add(world_diag, world, c, val)


@wp.kernel
def gather_JY_to_world(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    art_dof_start: wp.array[int],
    world_constraint_count: wp.array[int],
    world_dof_start: wp.array[int],
    J_group: wp.array3d[float],
    Y_group: wp.array3d[float],
    n_dofs: int,
    max_constraints: int,
    n_arts: int,
    # outputs
    J_world: wp.array3d[float],
    Y_world: wp.array3d[float],
):
    """Gather per-size-group J/Y into world-indexed arrays.

    Thread dim: n_arts * max_constraints * n_dofs.
    """
    tid = wp.tid()
    d = tid % n_dofs
    remainder = tid // n_dofs
    c = remainder % max_constraints
    idx = remainder // max_constraints
    if idx >= n_arts:
        return
    art = group_to_art[idx]
    world = art_to_world[art]
    if c >= world_constraint_count[world]:
        return
    dof_start = art_dof_start[art]
    w_dof_start = world_dof_start[world]
    local_d = (dof_start - w_dof_start) + d
    # Write unconditionally (including zeros) so J_world/Y_world don't need pre-zeroing
    J_world[world, c, local_d] = J_group[idx, c, d]
    Y_world[world, c, local_d] = Y_group[idx, c, d]


# =============================================================================
# Matrix-Free PGS Kernels for Free Rigid Bodies
# =============================================================================


@wp.kernel
def build_mf_contact_rows(
    contact_count: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_path: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    shape_material_mu: wp.array[float],
    enable_friction: int,
    pgs_beta: float,
    # outputs
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_row_type: wp.array2d[int],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    mf_phi: wp.array2d[float],
):
    """Build MF constraint rows for contacts between free rigid bodies / ground.

    For root free joints, the internal qd used here stores the COM-point linear
    term and angular velocity, i.e. `qd = [v_com_point, omega]`, where the COM
    point is the root body's world-space COM position. The MF contact Jacobian
    uses contact position relative to that point:
        J = [d, r x d]   (r = p_contact - p_com_world)
    """
    c = wp.tid()
    total_contacts = contact_count[0]
    if c >= total_contacts:
        return

    if contact_path[c] != 1:
        return

    slot = contact_slot[c]
    if slot < 0:
        return

    world = contact_world[c]
    shape_a = contact_shape0[c]
    shape_b = contact_shape1[c]
    # contact normal stored as A-to-B; negate to get B-to-A used internally
    normal = -contact_normal[c]

    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    thickness_a = contact_thickness0[c]
    thickness_b = contact_thickness1[c]

    # Compute contact points in world frame
    point_a_local = contact_point0[c]
    point_b_local = contact_point1[c]
    point_a_world = wp.vec3(0.0)
    point_b_world = wp.vec3(0.0)

    if body_a >= 0:
        X_wb_a = body_q[body_a]
        point_a_world = wp.transform_point(X_wb_a, point_a_local) - thickness_a * normal
    else:
        point_a_world = point_a_local - thickness_a * normal

    if body_b >= 0:
        X_wb_b = body_q[body_b]
        point_b_world = wp.transform_point(X_wb_b, point_b_local) + thickness_b * normal
    else:
        point_b_world = point_b_local + thickness_b * normal

    phi = wp.dot(normal, point_a_world - point_b_world)

    # Friction coefficient
    mu = 0.0
    mat_count = 0
    if shape_a >= 0:
        mu += shape_material_mu[shape_a]
        mat_count += 1
    if shape_b >= 0:
        mu += shape_material_mu[shape_b]
        mat_count += 1
    if mat_count > 0:
        mu /= float(mat_count)

    # Tangent basis
    t0, t1 = contact_tangent_basis(normal)

    # Write rows for normal + friction
    for row_offset in range(3):
        if row_offset > 0 and enable_friction == 0:
            break

        row_idx = slot + row_offset

        if row_offset == 0:
            d = normal
        elif row_offset == 1:
            d = t0
        else:
            d = t1

        # Body A Jacobian in articulation-local frame: J = [d, r_a x d], where
        # r_a is the contact point relative to articulation A's fixed origin.
        if body_a >= 0:
            art_a = contact_art_a[c]
            origin_a = articulation_origin[art_a]
            r_a = point_a_world - origin_a
            ang_a = wp.cross(r_a, d)
            mf_J_a[world, row_idx, 0] = d[0]
            mf_J_a[world, row_idx, 1] = d[1]
            mf_J_a[world, row_idx, 2] = d[2]
            mf_J_a[world, row_idx, 3] = ang_a[0]
            mf_J_a[world, row_idx, 4] = ang_a[1]
            mf_J_a[world, row_idx, 5] = ang_a[2]

        # Body B Jacobian in articulation-local frame (opposite sign).
        if body_b >= 0:
            art_b = contact_art_b[c]
            origin_b = articulation_origin[art_b]
            r_b = point_b_world - origin_b
            ang_b = wp.cross(r_b, d)
            mf_J_b[world, row_idx, 0] = -d[0]
            mf_J_b[world, row_idx, 1] = -d[1]
            mf_J_b[world, row_idx, 2] = -d[2]
            mf_J_b[world, row_idx, 3] = -ang_b[0]
            mf_J_b[world, row_idx, 4] = -ang_b[1]
            mf_J_b[world, row_idx, 5] = -ang_b[2]

        mf_body_a[world, row_idx] = body_a
        mf_body_b[world, row_idx] = body_b

        if row_offset == 0:
            mf_row_type[world, row_idx] = PGS_CONSTRAINT_TYPE_CONTACT
            mf_row_parent[world, row_idx] = -1
            mf_phi[world, row_idx] = phi
        else:
            mf_row_type[world, row_idx] = PGS_CONSTRAINT_TYPE_FRICTION
            mf_row_parent[world, row_idx] = slot
            mf_phi[world, row_idx] = 0.0
        mf_row_mu[world, row_idx] = mu


@wp.func
def spatial_matrix_block_inverse(M: wp.spatial_matrix):
    """Invert a 6x6 spatial matrix using 3x3 block inversion.

    Partition M = [A B; C D] into 3x3 blocks, then:
        S = D - C * A^-1 * B   (Schur complement)
        M^-1 = [A^-1 + A^-1*B*S^-1*C*A^-1,  -A^-1*B*S^-1]
               [-S^-1*C*A^-1,                 S^-1]
    """
    A = wp.mat33(
        M[0, 0],
        M[0, 1],
        M[0, 2],
        M[1, 0],
        M[1, 1],
        M[1, 2],
        M[2, 0],
        M[2, 1],
        M[2, 2],
    )
    B = wp.mat33(
        M[0, 3],
        M[0, 4],
        M[0, 5],
        M[1, 3],
        M[1, 4],
        M[1, 5],
        M[2, 3],
        M[2, 4],
        M[2, 5],
    )
    C = wp.mat33(
        M[3, 0],
        M[3, 1],
        M[3, 2],
        M[4, 0],
        M[4, 1],
        M[4, 2],
        M[5, 0],
        M[5, 1],
        M[5, 2],
    )
    D = wp.mat33(
        M[3, 3],
        M[3, 4],
        M[3, 5],
        M[4, 3],
        M[4, 4],
        M[4, 5],
        M[5, 3],
        M[5, 4],
        M[5, 5],
    )

    Ainv = wp.inverse(A)
    AinvB = Ainv * B
    S = D - C * AinvB
    Sinv = wp.inverse(S)
    SinvCAinv = Sinv * C * Ainv

    # Top-left: Ainv + AinvB * SinvCAinv
    TL = Ainv + AinvB * SinvCAinv
    # Top-right: -AinvB * Sinv
    TR = -AinvB * Sinv
    # Bottom-left: -SinvCAinv
    BL = -SinvCAinv
    # Bottom-right: Sinv
    BR = Sinv

    return wp.spatial_matrix(
        TL[0, 0],
        TL[0, 1],
        TL[0, 2],
        TR[0, 0],
        TR[0, 1],
        TR[0, 2],
        TL[1, 0],
        TL[1, 1],
        TL[1, 2],
        TR[1, 0],
        TR[1, 1],
        TR[1, 2],
        TL[2, 0],
        TL[2, 1],
        TL[2, 2],
        TR[2, 0],
        TR[2, 1],
        TR[2, 2],
        BL[0, 0],
        BL[0, 1],
        BL[0, 2],
        BR[0, 0],
        BR[0, 1],
        BR[0, 2],
        BL[1, 0],
        BL[1, 1],
        BL[1, 2],
        BR[1, 0],
        BR[1, 1],
        BR[1, 2],
        BL[2, 0],
        BL[2, 1],
        BL[2, 2],
        BR[2, 0],
        BR[2, 1],
        BR[2, 2],
    )


@wp.kernel
def compute_mf_body_Hinv(
    body_I_s: wp.array[wp.spatial_matrix],
    is_free_rigid: wp.array[int],
    body_to_articulation: wp.array[int],
    # outputs
    mf_body_Hinv: wp.array[wp.spatial_matrix],
):
    """Compute H^-1 = inverse(body_I_s) for free rigid bodies.

    For root free joints, H = body_I_s in articulation-local coordinates.
    This remains a full 6x6 matrix for bodies with non-zero CoM offsets.
    """
    b = wp.tid()
    art = body_to_articulation[b]
    if art < 0:
        return
    if is_free_rigid[art] == 0:
        return

    mf_body_Hinv[b] = spatial_matrix_block_inverse(body_I_s[b])


@wp.kernel
def compute_mf_effective_mass_and_rhs(
    mf_constraint_count: wp.array[int],
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_body_Hinv: wp.array[wp.spatial_matrix],
    mf_phi: wp.array2d[float],
    mf_row_type: wp.array2d[int],
    pgs_cfm: float,
    pgs_beta: float,
    dt: float,
    mf_max_constraints: int,
    # outputs
    mf_eff_mass_inv: wp.array2d[float],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    mf_rhs: wp.array2d[float],
):
    """Compute effective mass diagonal, H^-1*J^T, and RHS bias for MF constraints.

    The effective mass for constraint i is:
        d_ii = J_a^T * H_a_inv * J_a + J_b^T * H_b_inv * J_b + cfm

    H_inv is the full 6x6 inverse of the spatial inertia in articulation-local
    coordinates for each free rigid articulation.

    RHS stores only the stabilization bias (not J*v), since the MF PGS
    recomputes J*v each iteration from the live velocity array.
    """
    tid = wp.tid()
    world = tid // mf_max_constraints
    i = tid % mf_max_constraints
    if i >= mf_constraint_count[world]:
        return

    ba = mf_body_a[world, i]
    bb = mf_body_b[world, i]

    # Load Jacobian as spatial_vector
    Ja = wp.spatial_vector(
        mf_J_a[world, i, 0],
        mf_J_a[world, i, 1],
        mf_J_a[world, i, 2],
        mf_J_a[world, i, 3],
        mf_J_a[world, i, 4],
        mf_J_a[world, i, 5],
    )
    Jb = wp.spatial_vector(
        mf_J_b[world, i, 0],
        mf_J_b[world, i, 1],
        mf_J_b[world, i, 2],
        mf_J_b[world, i, 3],
        mf_J_b[world, i, 4],
        mf_J_b[world, i, 5],
    )

    d = pgs_cfm

    # Side A: MiJt_a = H_a_inv * J_a, d += J_a^T * MiJt_a
    if ba >= 0:
        Hinv_a = mf_body_Hinv[ba]
        MiJt_a = Hinv_a * Ja
        d += wp.dot(Ja, MiJt_a)
        mf_MiJt_a[world, i, 0] = MiJt_a[0]
        mf_MiJt_a[world, i, 1] = MiJt_a[1]
        mf_MiJt_a[world, i, 2] = MiJt_a[2]
        mf_MiJt_a[world, i, 3] = MiJt_a[3]
        mf_MiJt_a[world, i, 4] = MiJt_a[4]
        mf_MiJt_a[world, i, 5] = MiJt_a[5]

    # Side B
    if bb >= 0:
        Hinv_b = mf_body_Hinv[bb]
        MiJt_b = Hinv_b * Jb
        d += wp.dot(Jb, MiJt_b)
        mf_MiJt_b[world, i, 0] = MiJt_b[0]
        mf_MiJt_b[world, i, 1] = MiJt_b[1]
        mf_MiJt_b[world, i, 2] = MiJt_b[2]
        mf_MiJt_b[world, i, 3] = MiJt_b[3]
        mf_MiJt_b[world, i, 4] = MiJt_b[4]
        mf_MiJt_b[world, i, 5] = MiJt_b[5]

    if d > 0.0:
        mf_eff_mass_inv[world, i] = 1.0 / d
    else:
        mf_eff_mass_inv[world, i] = 0.0

    # Baumgarte stabilization bias only (not J*v -- recomputed each PGS iter)
    bias = float(0.0)
    rtype = mf_row_type[world, i]
    if rtype == PGS_CONSTRAINT_TYPE_CONTACT:
        phi_val = mf_phi[world, i]
        bias = pgs_beta / dt * wp.min(phi_val, 0.0)

    mf_rhs[world, i] = bias


@wp.kernel
def finalize_mf_constraint_counts(
    mf_slot_counter: wp.array[int],
    mf_max_constraints: int,
    slots_per_contact: int,
    # outputs
    mf_constraint_count: wp.array[int],
):
    """Clamp MF slot counter to max and store as constraint count.

    See :func:`finalize_world_constraint_counts` for the gap-avoidance
    rationale behind the ``slots_per_contact`` rounding.
    """
    world = wp.tid()
    count = mf_slot_counter[world]
    if count > mf_max_constraints:
        count = mf_max_constraints
    count = (count // slots_per_contact) * slots_per_contact
    mf_constraint_count[world] = count


@wp.kernel
def compute_mf_world_dof_offsets(
    mf_constraint_count: wp.array[int],
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    body_to_articulation: wp.array[int],
    art_dof_start: wp.array[int],
    world_dof_start: wp.array[int],
    mf_max_constraints: int,
    # outputs
    mf_dof_a: wp.array2d[int],
    mf_dof_b: wp.array2d[int],
):
    """Compute world-relative DOF offsets for each MF contact body.

    For each MF constraint, stores the articulation DOF start minus the
    world DOF start for body A and B.  The two-phase GS kernel uses
    these offsets to index into the shared velocity vector.
    """
    tid = wp.tid()
    world = tid // mf_max_constraints
    c = tid % mf_max_constraints
    if c >= mf_constraint_count[world]:
        return
    w_dof = world_dof_start[world]
    ba = mf_body_a[world, c]
    bb = mf_body_b[world, c]
    if ba >= 0:
        mf_dof_a[world, c] = art_dof_start[body_to_articulation[ba]] - w_dof
    else:
        mf_dof_a[world, c] = -1
    if bb >= 0:
        mf_dof_b[world, c] = art_dof_start[body_to_articulation[bb]] - w_dof
    else:
        mf_dof_b[world, c] = -1


@wp.kernel
def finalize_world_diag_cfm(
    world_constraint_count: wp.array[int],
    world_row_cfm: wp.array2d[float],
    # in/out
    world_diag: wp.array2d[float],
):
    """Add CFM to world diagonal after Delassus accumulation."""
    world = wp.tid()
    m = world_constraint_count[world]

    for i in range(m):
        world_diag[world, i] += world_row_cfm[world, i]


@wp.kernel
def add_dense_contact_compliance_to_diag(
    world_constraint_count: wp.array[int],
    world_row_type: wp.array2d[int],
    contact_alpha: float,
    # in/out
    world_diag: wp.array2d[float],
):
    """Add normal-contact compliance to the dense PGS diagonal.

    The dense articulated contact path uses a Delassus diagonal in impulse
    space. A compliance ``alpha = compliance / dt^2`` contributes an additional
    diagonal term for normal contact rows only, yielding a softer normal
    response without changing friction or joint-limit rows.
    """
    world = wp.tid()
    m = world_constraint_count[world]

    for i in range(m):
        if world_row_type[world, i] == PGS_CONSTRAINT_TYPE_CONTACT:
            world_diag[world, i] += contact_alpha


# =============================================================================
# Parallelized Non-Tiled Kernels for Heterogeneous Multi-Articulation
# =============================================================================
# These kernels parallelize across constraints (and constraint pairs) to achieve
# much better GPU utilization than the single-thread-per-articulation versions.


# =============================================================================
# Tiled kernels for homogenous multi-articulation support
# =============================================================================


@wp.kernel
def crba_fill_par_dof(
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    mass_update_mask: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_S_s: wp.array[wp.spatial_vector],
    body_I_c: wp.array[wp.spatial_matrix],
    # Size-group parameters
    group_to_art: wp.array[int],
    n_dofs: int,  # = TILE_DOF for tiled path
    # outputs
    H_group: wp.array3d[float],  # [n_arts_of_size, n_dofs, n_dofs]
):
    """
    CRBA fill kernel that writes directly to size-grouped H storage.

    Thread dimension: n_arts_of_size * n_dofs (one thread per articulation-column pair)

    This version is for homogenous multi-articulation where all articulations have
    the same DOF count equal to TILE_DOF.
    """
    tid = wp.tid()

    group_idx = tid // n_dofs
    col_idx = tid % n_dofs

    art_idx = group_to_art[group_idx]

    if mass_update_mask[art_idx] == 0:
        return

    # All articulations in this group have exactly n_dofs DOFs
    if col_idx >= n_dofs:
        return

    global_dof_start = articulation_dof_start[art_idx]
    target_dof_global = global_dof_start + col_idx

    joint_start = articulation_start[art_idx]
    joint_end = articulation_start[art_idx + 1]

    # Find the joint that owns this DOF
    pivot_joint = int(-1)
    for j in range(joint_start, joint_end):
        q_start = joint_qd_start[j]
        q_end = joint_qd_start[j + 1]
        if target_dof_global >= q_start and target_dof_global < q_end:
            pivot_joint = j
            break

    if pivot_joint == -1:
        return

    # Compute Force F = I_c[pivot] * S[column]
    S_col = joint_S_s[target_dof_global]
    I_comp = body_I_c[pivot_joint]
    F = I_comp * S_col

    # Walk up the tree and project F onto ancestors
    # H[row, col] = S[row] * F
    curr = pivot_joint

    while curr != -1:
        if curr < joint_start:
            break

        q_start = joint_qd_start[curr]
        q_dim = joint_dof_dim[curr]
        count = q_dim[0] + q_dim[1]

        dof_offset_local = q_start - global_dof_start

        for k in range(count):
            row_idx = dof_offset_local + k

            S_row = joint_S_s[q_start + k]
            val = wp.dot(S_row, F)

            # Write to grouped 3D array
            H_group[group_idx, row_idx, col_idx] = val
            H_group[group_idx, col_idx, row_idx] = val

        curr = joint_ancestor[curr]


@wp.kernel
def gather_tau_to_groups(
    joint_tau: wp.array[float],  # [total_dofs]
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    n_dofs: int,
    tau_group: wp.array3d[float],  # [n_arts, n_dofs, 1]
):
    """Gather joint_tau from 1D array into grouped 3D buffer for tiled solve.

    Thread dimension: n_arts_of_size (one thread per articulation in this size group)
    """
    idx = wp.tid()
    art = group_to_art[idx]
    dof_start = articulation_dof_start[art]
    for i in range(n_dofs):
        tau_group[idx, i, 0] = joint_tau[dof_start + i]


@wp.kernel
def scatter_qdd_from_groups(
    qdd_group: wp.array3d[float],  # [n_arts, n_dofs, 1]
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    n_dofs: int,
    joint_qdd: wp.array[float],  # [total_dofs]
):
    """Scatter qdd from grouped 3D buffer back to 1D array after tiled solve.

    Thread dimension: n_arts_of_size (one thread per articulation in this size group)
    """
    idx = wp.tid()
    art = group_to_art[idx]
    dof_start = articulation_dof_start[art]
    for i in range(n_dofs):
        joint_qdd[dof_start + i] = qdd_group[idx, i, 0]


# =============================================================================
# PGS Convergence Diagnostic Kernel (velocity-space mode)
# =============================================================================


# =============================================================================
# PGS NCP / MDP Residual Diagnostic Kernel (velocity-space mode)
# =============================================================================


@wp.kernel
def pack_contact_triplets_vec3(
    src: wp.array2d[float],
    dst: wp.array2d[wp.vec3],
):
    """Pack flat scalar rows into vec3 contact triplets: dst[w,c] = vec3(src[w,c*3], src[w,c*3+1], src[w,c*3+2])."""
    world, c = wp.tid()
    row0 = c * 3
    dst[world, c] = wp.vec3(src[world, row0], src[world, row0 + 1], src[world, row0 + 2])


@wp.kernel
def pack_mf_meta(
    mf_dof_a: wp.array2d[int],
    mf_dof_b: wp.array2d[int],
    mf_eff_mass_inv: wp.array2d[float],
    mf_rhs: wp.array2d[float],
    mf_row_type: wp.array2d[int],
    mf_row_parent: wp.array2d[int],
    mf_meta: wp.array2d[wp.vec4i],
):
    """Pack 6 MF metadata arrays into a single vec4i for 16-byte vectorized loads."""
    world, i = wp.tid()
    dof_a = mf_dof_a[world, i]
    dof_b = mf_dof_b[world, i]
    eff_mass = mf_eff_mass_inv[world, i]
    rhs = mf_rhs[world, i]
    row_type = mf_row_type[world, i]
    row_parent = mf_row_parent[world, i]

    packed_dofs = (dof_a << 16) | (dof_b & 0xFFFF)
    packed_tp = (row_parent << 16) | (row_type & 0xFFFF)

    mf_meta[world, i] = wp.vec4i(
        packed_dofs,
        wp.cast(eff_mass, wp.int32),
        wp.cast(rhs, wp.int32),
        packed_tp,
    )


class TiledKernelFactory:
    """Factory for generating size-specialized tiled kernels for heterogeneous multi-articulation.

    This factory generates and caches tiled kernels specialized for specific DOF counts,
    enabling optimal tiled operations (Cholesky, triangular solves) for articulations
    with different numbers of degrees of freedom.

    The pattern follows ik_lbfgs_optimizer.py: kernels are generated on-demand with
    wp.constant() captured via closure, then cached by (dimensions, device.arch).
    """

    # Class-level caches: key -> compiled kernel
    _hinv_jt_cache: ClassVar[dict[tuple[int, int, str], "wp.Kernel"]] = {}
    _cholesky_cache: ClassVar[dict[tuple[int, str], "wp.Kernel"]] = {}
    _pgs_fused_warp_cache: ClassVar[dict] = {}
    _triangular_solve_cache: ClassVar[dict[tuple[int, str], "wp.Kernel"]] = {}

    @classmethod
    def get_hinv_jt_kernel(cls, n_dofs: int, max_constraints: int, device: "wp.Device") -> "wp.Kernel":
        """Get or create a tiled H^-1*J^T kernel for the given dimensions."""
        key = (n_dofs, max_constraints, device.arch)
        if key not in cls._hinv_jt_cache:
            cls._hinv_jt_cache[key] = cls._build_hinv_jt_kernel(n_dofs, max_constraints)
        return cls._hinv_jt_cache[key]

    @classmethod
    def _build_hinv_jt_kernel(cls, n_dofs: int, max_constraints: int) -> "wp.Kernel":
        """Build specialized H^-1*J^T kernel for given dimensions.

        Solves Y = H^-1 * J^T using tiled Cholesky solve:
          L * L^T * Y = J^T
          => L * Z = J^T (forward solve)
          => L^T * Y = Z (backward solve)
        """
        # Create compile-time constants via closure
        # Convert to Python int to ensure wp.constant() accepts them
        TILE_DOF_LOCAL = wp.constant(int(n_dofs))
        TILE_CONSTRAINTS_LOCAL = wp.constant(int(max_constraints))

        def hinv_jt_tiled_template(
            L_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
            J_group: wp.array3d[float],  # [n_arts, max_c, n_dofs]
            group_to_art: wp.array[int],
            art_to_world: wp.array[int],
            world_constraint_count: wp.array[int],
            # output
            Y_group: wp.array3d[float],  # [n_arts, max_c, n_dofs]
        ):
            idx = wp.tid()
            art = group_to_art[idx]
            world = art_to_world[art]
            n_constraints = world_constraint_count[world]

            if n_constraints == 0:
                return

            # Load L (Cholesky factor) and J (Jacobian rows)
            L_tile = wp.tile_load(L_group[idx], shape=(TILE_DOF_LOCAL, TILE_DOF_LOCAL), bounds_check=False)
            J_tile = wp.tile_load(J_group[idx], shape=(TILE_CONSTRAINTS_LOCAL, TILE_DOF_LOCAL), bounds_check=False)

            # Solve L * Z = J^T (forward substitution)
            # J_tile is (max_c x n_dofs), J^T is (n_dofs x max_c)
            Jt_tile = wp.tile_transpose(J_tile)
            Z_tile = wp.tile_lower_solve(L_tile, Jt_tile)

            # Solve L^T * Y = Z (backward substitution)
            Lt_tile = wp.tile_transpose(L_tile)
            X_tile = wp.tile_upper_solve(Lt_tile, Z_tile)

            # Store Y = H^-1 * J^T (transpose back to row layout)
            Y_out_tile = wp.tile_transpose(X_tile)
            wp.tile_store(Y_group[idx], Y_out_tile)

        hinv_jt_tiled_template.__name__ = f"hinv_jt_tiled_{n_dofs}_{max_constraints}"
        hinv_jt_tiled_template.__qualname__ = f"hinv_jt_tiled_{n_dofs}_{max_constraints}"
        return wp.kernel(enable_backward=False, module="unique")(hinv_jt_tiled_template)

    @classmethod
    def get_cholesky_kernel(cls, n_dofs: int, device: "wp.Device") -> "wp.Kernel":
        """Get or create a tiled Cholesky kernel for the given DOF count."""
        key = (n_dofs, device.arch)
        if key not in cls._cholesky_cache:
            cls._cholesky_cache[key] = cls._build_cholesky_kernel(n_dofs)
        return cls._cholesky_cache[key]

    @classmethod
    def _build_cholesky_kernel(cls, n_dofs: int) -> "wp.Kernel":
        """Build specialized Cholesky kernel for given DOF count.

        Computes L such that H + diag(armature) = L * L^T.
        """
        # Convert to Python int to ensure wp.constant() accepts them
        TILE_DOF_LOCAL = wp.constant(int(n_dofs))

        def cholesky_tiled_template(
            H_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
            R_group: wp.array2d[float],  # [n_arts, n_dofs] armature
            group_to_art: wp.array[int],
            mass_update_mask: wp.array[int],
            # output
            L_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
        ):
            idx = wp.tid()
            art = group_to_art[idx]

            if mass_update_mask[art] == 0:
                return

            # Load H and armature
            H_tile = wp.tile_load(H_group[idx], shape=(TILE_DOF_LOCAL, TILE_DOF_LOCAL), bounds_check=False)
            armature = wp.tile_load(R_group[idx], shape=(TILE_DOF_LOCAL,), bounds_check=False)

            # Add armature to diagonal
            H_tile = wp.tile_diag_add(H_tile, armature)

            # Compute Cholesky factorization
            L_tile = wp.tile_cholesky(H_tile)

            # Store result
            wp.tile_store(L_group[idx], L_tile)

        cholesky_tiled_template.__name__ = f"cholesky_tiled_{n_dofs}"
        cholesky_tiled_template.__qualname__ = f"cholesky_tiled_{n_dofs}"
        return wp.kernel(enable_backward=False, module="unique")(cholesky_tiled_template)

    @classmethod
    def get_triangular_solve_kernel(cls, n_dofs: int, device: "wp.Device") -> "wp.Kernel":
        """Get or create a tiled triangular solve kernel for the given DOF count."""
        key = (n_dofs, device.arch)
        if key not in cls._triangular_solve_cache:
            cls._triangular_solve_cache[key] = cls._build_triangular_solve_kernel(n_dofs)
        return cls._triangular_solve_cache[key]

    @classmethod
    def _build_triangular_solve_kernel(cls, n_dofs: int) -> "wp.Kernel":
        """Build specialized triangular solve kernel for given DOF count.

        Solves L * L^T * x = b for x using tiled forward and backward substitution.
        """
        TILE_DOF_LOCAL = wp.constant(int(n_dofs))

        def trisolve_tiled_template(
            L_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
            tau_group: wp.array3d[float],  # [n_arts, n_dofs, 1]
            qdd_group: wp.array3d[float],  # [n_arts, n_dofs, 1]
        ):
            idx = wp.tid()
            L_tile = wp.tile_load(L_group[idx], shape=(TILE_DOF_LOCAL, TILE_DOF_LOCAL), bounds_check=False)
            tau_tile = wp.tile_load(tau_group[idx], shape=(TILE_DOF_LOCAL, 1), bounds_check=False)

            # Forward substitution: L * z = tau
            z_tile = wp.tile_lower_solve(L_tile, tau_tile)

            # Backward substitution: L^T * qdd = z
            Lt_tile = wp.tile_transpose(L_tile)
            qdd_tile = wp.tile_upper_solve(Lt_tile, z_tile)

            wp.tile_store(qdd_group[idx], qdd_tile)

        trisolve_tiled_template.__name__ = f"trisolve_tiled_{n_dofs}"
        trisolve_tiled_template.__qualname__ = f"trisolve_tiled_{n_dofs}"
        return wp.kernel(enable_backward=False, module="unique")(trisolve_tiled_template)

    @classmethod
    def get_pgs_fused_warp_kernel(
        cls,
        max_constraints: int,
        max_contact_triplets: int,
        mf_max_constraints: int,
        max_world_dofs: int,
        device: "wp.Device",
    ) -> "wp.Kernel":
        key = (max_constraints, max_contact_triplets, mf_max_constraints, max_world_dofs, device.arch)
        if key not in cls._pgs_fused_warp_cache:
            cls._pgs_fused_warp_cache[key] = cls._build_pgs_fused_warp_kernel(
                max_constraints, max_contact_triplets, mf_max_constraints, max_world_dofs
            )
        return cls._pgs_fused_warp_cache[key]

    @classmethod
    def _build_pgs_fused_warp_kernel(
        cls, max_constraints: int, max_contact_triplets: int, mf_max_constraints: int, max_world_dofs: int
    ) -> "wp.Kernel":
        """Fused two-phase GS PGS kernel — pure Warp tile API.

        Phase 1 (dense): tile-parallel dot/update over D DOFs using
        cooperative tile_load + tile_dot + tile_axpy. J/Y loads are
        software-pipelined (prefetch next row while computing current).
        Scoped @wp.func helpers limit register lifetimes. Contact metadata
        (diag, rhs) loaded as vec3 triplets to reduce broadcast loads.

        Phase 2 (MF): SIMT scalar code within tiled kernel. J/MiJt loads
        are software-pipelined and lane-parallel (lanes 0-5 body A, 6-11
        body B). Metadata packed into vec4i for single 16-byte loads.
        tile_extract (sync-free on shared) for velocity reads,
        tile_scatter_add(atomic=False) for velocity writes (lanes write
        distinct DOF indices), tile_scatter_masked for single-lane impulse
        writes.

        Both phases share s_v in shared memory. All PGS iterations run
        inside the kernel (no global round-trip for v between phases).
        One warp (32 threads) per world.
        """
        M_D = wp.constant(max_constraints)
        M_CT = wp.constant(max_contact_triplets)
        M_MF = mf_max_constraints
        M_MF_CONST = wp.constant(mf_max_constraints)
        D_val = max_world_dofs
        D = wp.constant(D_val)

        # Phase 1 helpers — scoped to limit register lifetimes
        @wp.func
        def dot_Jv(
            s_v: wp.tile(dtype=float, shape=(D_val,), storage="shared"),
            J_row: wp.tile(dtype=float, shape=(D_val,), storage="register"),
        ) -> float:
            """Compute J·v from a pre-loaded J row. Scoped — register tiles freed on return."""
            return wp.tile_extract(wp.tile_dot(J_row, s_v), 0)

        @wp.func
        def load_J_row(
            J_world: wp.array3d[float],
            world: int,
            row: int,
        ) -> wp.tile(dtype=float, shape=(D_val,), storage="register"):
            """Load a J/Y row into a register tile."""
            return wp.tile_load(J_world[world, row], shape=(D_val,), storage="register", bounds_check=False)

        @wp.func
        def velocity_update_preloaded(
            s_v: wp.tile(dtype=float, shape=(D_val,), storage="shared"),
            Y_row: wp.tile(dtype=float, shape=(D_val,), storage="register"),
            delta_impulse: float,
        ):
            """Apply s_v += Y * delta using a pre-loaded Y row."""
            wp.tile_axpy(delta_impulse, Y_row, s_v)

        @wp.func
        def velocity_update(
            s_v: wp.tile(dtype=float, shape=(D_val,), storage="shared"),
            Y_world: wp.array3d[float],
            world: int,
            row: int,
            delta_impulse: float,
        ):
            """Load Y row and apply s_v += Y * delta."""
            Y_row = wp.tile_load(Y_world[world, row], shape=(D_val,), storage="register", bounds_check=False)
            wp.tile_axpy(delta_impulse, Y_row, s_v)

        # Phase 2: SIMT within tiled kernel — each lane holds one element of the
        # 6-DOF body A/B operations (lanes 0-5 body A, 6-11 body B). Velocity reads
        # use tile_extract on shared s_v (sync-free). Impulse writes use
        # tile_scatter_masked; velocity writes use tile_scatter_add(atomic=False)
        # since body A and body B DOF ranges don't overlap within a constraint.
        @wp.func
        def pgs_mf_phase(
            world: int,
            lane: int,
            s_v: wp.tile(dtype=float, shape=(D_val,), storage="shared"),
            s_lam_mf: wp.tile(dtype=float, shape=(M_MF,), storage="shared"),
            mf_constraint_count: wp.array[int],
            mf_meta: wp.array2d[wp.vec4i],
            mf_J_a: wp.array3d[float],
            mf_J_b: wp.array3d[float],
            mf_MiJt_a: wp.array3d[float],
            mf_MiJt_b: wp.array3d[float],
            mf_row_mu: wp.array2d[float],
            omega: float,
        ):
            m_mf = mf_constraint_count[world]
            if m_mf > M_MF_CONST:
                m_mf = M_MF_CONST
            if m_mf == 0:
                return

            # Software-pipelined Phase 2: prefetch J/MiJt/metadata for constraint
            # i+1 while computing constraint i. Hides global memory latency.
            pre_Ja = float(0.0)
            pre_Jb = float(0.0)
            pre_MiJta = float(0.0)
            pre_MiJtb = float(0.0)
            pre_meta = wp.vec4i(0, 0, 0, 0)

            # Prefetch constraint 0
            if m_mf > 0:
                pre_meta = mf_meta[world, 0]
                if lane < 6:
                    pre_Ja = mf_J_a[world, 0, lane]
                    pre_MiJta = mf_MiJt_a[world, 0, lane]
                if lane >= 6 and lane < 12:
                    pre_Jb = mf_J_b[world, 0, lane - 6]
                    pre_MiJtb = mf_MiJt_b[world, 0, lane - 6]

            for i in range(m_mf):
                # Consume prefetched data
                cur_Ja = pre_Ja
                cur_Jb = pre_Jb
                cur_MiJta = pre_MiJta
                cur_MiJtb = pre_MiJtb
                meta = pre_meta

                # Prefetch i+1
                if i + 1 < m_mf:
                    pre_meta = mf_meta[world, i + 1]
                    if lane < 6:
                        pre_Ja = mf_J_a[world, i + 1, lane]
                        pre_MiJta = mf_MiJt_a[world, i + 1, lane]
                    if lane >= 6 and lane < 12:
                        pre_Jb = mf_J_b[world, i + 1, lane - 6]
                        pre_MiJtb = mf_MiJt_b[world, i + 1, lane - 6]

                # Unpack metadata (already prefetched)
                dof_a = meta[0] >> 16
                dof_b = (meta[0] << 16) >> 16  # sign-extend lower 16 bits
                mf_diag = wp.cast(meta[1], wp.float32)
                if mf_diag <= 0.0:
                    continue

                # J*v dot product — lane-parallel using prefetched J values
                my_jv = float(0.0)
                if lane < 6 and dof_a >= 0:
                    my_jv = cur_Ja * wp.tile_extract(s_v, dof_a + lane)
                if lane >= 6 and lane < 12 and dof_b >= 0:
                    my_jv = cur_Jb * wp.tile_extract(s_v, dof_b + lane - 6)
                # Reduce per-thread values: tile_full creates a register tile where
                # each thread's element is its own my_jv, then tile_sum reduces via shuffles.
                jv_tile = wp.tile_sum(wp.tile_full(shape=(32,), value=my_jv, dtype=float, storage="register"))
                jv = wp.tile_extract(jv_tile, 0)

                # PGS projection
                rhs_val = wp.cast(meta[2], wp.float32)
                residual = jv + rhs_val
                delta = -residual * mf_diag
                old_impulse = wp.tile_extract(s_lam_mf, i)
                new_impulse = old_impulse + omega * delta
                mf_rt = meta[3] & 0xFFFF

                if mf_rt == 0:
                    if new_impulse < 0.0:
                        new_impulse = 0.0
                elif mf_rt == 2:
                    mf_par = meta[3] >> 16
                    lambda_n = wp.tile_extract(s_lam_mf, mf_par)
                    mu = mf_row_mu[world, i]
                    radius = wp.max(mu * lambda_n, 0.0)

                    if radius <= 0.0:
                        new_impulse = 0.0
                    else:
                        sib = wp.where(i == mf_par + 1, mf_par + 2, mf_par + 1)
                        wp.tile_scatter_masked(s_lam_mf, i, new_impulse, lane == 0)
                        a_val = new_impulse
                        b_val = wp.tile_extract(s_lam_mf, sib)
                        mag = wp.sqrt(a_val * a_val + b_val * b_val)
                        if mag > radius:
                            scale = radius / mag
                            new_impulse = a_val * scale
                            sib_new = b_val * scale
                            sib_delta = sib_new - b_val
                            wp.tile_scatter_masked(s_lam_mf, sib, sib_new, lane == 0)

                            # Sibling velocity update — unpack sibling dofs from meta
                            sib_meta = mf_meta[world, sib]
                            sib_dof_a = sib_meta[0] >> 16
                            sib_dof_b = (sib_meta[0] << 16) >> 16
                            sib_idx = -1
                            sib_val = float(0.0)
                            if lane < 6 and sib_dof_a >= 0:
                                sib_idx = sib_dof_a + lane
                                sib_val = mf_MiJt_a[world, sib, lane] * sib_delta
                            elif lane >= 6 and lane < 12 and sib_dof_b >= 0:
                                sib_idx = sib_dof_b + lane - 6
                                sib_val = mf_MiJt_b[world, sib, lane - 6] * sib_delta
                            wp.tile_scatter_add(s_v, sib_idx, sib_val, sib_idx >= 0, atomic=False)

                delta_impulse = new_impulse - old_impulse
                wp.tile_scatter_masked(s_lam_mf, i, new_impulse, lane == 0)

                # Velocity update — lane-parallel scatter using prefetched MiJt
                if delta_impulse != 0.0:
                    idx = -1
                    val = float(0.0)
                    if lane < 6 and dof_a >= 0:
                        idx = dof_a + lane
                        val = cur_MiJta * delta_impulse
                    elif lane >= 6 and lane < 12 and dof_b >= 0:
                        idx = dof_b + lane - 6
                        val = cur_MiJtb * delta_impulse
                    wp.tile_scatter_add(s_v, idx, val, idx >= 0, atomic=False)

        def pgs_fused_warp(
            # Dense
            world_constraint_count: wp.array[int],
            dense_contact_row_count: wp.array[int],
            world_dof_start: wp.array[int],
            rhs_bias: wp.array2d[float],
            rhs_bias_vec3: wp.array2d[wp.vec3],
            world_diag: wp.array2d[float],
            world_diag_vec3: wp.array2d[wp.vec3],
            impulses_vec3: wp.array2d[wp.vec3],
            impulses_flat: wp.array2d[float],
            J_world: wp.array3d[float],
            Y_world: wp.array3d[float],
            world_row_mu: wp.array2d[float],
            # MF
            mf_constraint_count: wp.array[int],
            mf_meta: wp.array2d[wp.vec4i],
            mf_impulses: wp.array2d[float],
            mf_J_a: wp.array3d[float],
            mf_J_b: wp.array3d[float],
            mf_MiJt_a: wp.array3d[float],
            mf_MiJt_b: wp.array3d[float],
            mf_row_mu: wp.array2d[float],
            # Shared
            iterations: int,
            omega: float,
            # Output
            v_out: wp.array[float],
        ):
            world, thread = wp.tid()

            m_total = world_constraint_count[world]
            m_contact_rows = dense_contact_row_count[world]
            if m_total > M_D:
                m_total = M_D
            if m_contact_rows > M_D:
                m_contact_rows = M_D
            n_contacts = m_contact_rows // 3

            w_dof_start = world_dof_start[world]

            # ── LOAD PHASE ──
            s_v = wp.tile_load(v_out, shape=(D,), offset=(w_dof_start,), storage="shared", bounds_check=False)
            s_lam_contact = wp.tile_load(impulses_vec3[world], shape=(M_CT,), storage="shared", bounds_check=False)
            s_lam_mf = wp.tile_load(mf_impulses[world], shape=(M_MF_CONST,), storage="shared", bounds_check=False)
            # ── SOLVE PHASE ──
            for _iter in range(iterations):
                # ── Phase 1: Dense contacts (tile API, pipelined J+Y loads) ──
                # Prefetch first J and Y rows before loop
                pre_J = load_J_row(J_world, world, 0)
                pre_Y = load_J_row(Y_world, world, 0)

                for c in range(n_contacts):
                    row0 = c * 3
                    lam3 = wp.tile_extract(s_lam_contact, c)
                    diag3 = world_diag_vec3[world, c]
                    rhs3 = rhs_bias_vec3[world, c]

                    # Normal — consume prefetched J+Y, prefetch friction 1
                    cur_J = pre_J
                    cur_Y = pre_Y
                    pre_J = load_J_row(J_world, world, row0 + 1)
                    pre_Y = load_J_row(Y_world, world, row0 + 1)
                    if diag3[0] > 0.0:
                        jv = dot_Jv(s_v, cur_J)
                        new_n = wp.max(lam3[0] + omega * (-(jv + rhs3[0]) / diag3[0]), 0.0)
                        delta_n = new_n - lam3[0]
                        lam3 = wp.vec3(new_n, lam3[1], lam3[2])
                        if delta_n != 0.0:
                            velocity_update_preloaded(s_v, cur_Y, delta_n)

                    # Friction 1 — consume prefetched J+Y, prefetch friction 2
                    cur_J = pre_J
                    cur_Y = pre_Y
                    pre_J = load_J_row(J_world, world, row0 + 2)
                    pre_Y = load_J_row(Y_world, world, row0 + 2)
                    if diag3[1] > 0.0:
                        jv = dot_Jv(s_v, cur_J)
                        new_f1 = lam3[1] + omega * (-(jv + rhs3[1]) / diag3[1])
                        delta_f1 = new_f1 - lam3[1]
                        lam3 = wp.vec3(lam3[0], new_f1, lam3[2])
                        if delta_f1 != 0.0:
                            velocity_update_preloaded(s_v, cur_Y, delta_f1)

                    # Friction 2 — consume prefetched J+Y, prefetch next contact
                    cur_J = pre_J
                    cur_Y = pre_Y
                    if c + 1 < n_contacts:
                        pre_J = load_J_row(J_world, world, (c + 1) * 3)
                        pre_Y = load_J_row(Y_world, world, (c + 1) * 3)
                    if diag3[2] > 0.0:
                        jv = dot_Jv(s_v, cur_J)
                        new_f2 = lam3[2] + omega * (-(jv + rhs3[2]) / diag3[2])
                        delta_f2 = new_f2 - lam3[2]
                        lam3 = wp.vec3(lam3[0], lam3[1], new_f2)
                        if delta_f2 != 0.0:
                            velocity_update_preloaded(s_v, cur_Y, delta_f2)

                    # Friction cone projection (uses non-pipelined velocity_update
                    # for random-access sibling rows)
                    mu = world_row_mu[world, row0 + 1]
                    radius = wp.max(mu * lam3[0], 0.0)
                    if radius <= 0.0:
                        if lam3[1] != 0.0:
                            velocity_update(s_v, Y_world, world, row0 + 1, -lam3[1])
                        if lam3[2] != 0.0:
                            velocity_update(s_v, Y_world, world, row0 + 2, -lam3[2])
                        lam3 = wp.vec3(lam3[0], 0.0, 0.0)
                    else:
                        mag = wp.sqrt(lam3[1] * lam3[1] + lam3[2] * lam3[2])
                        if mag > radius:
                            scale = radius / mag
                            old_f1 = lam3[1]
                            old_f2 = lam3[2]
                            lam3 = wp.vec3(lam3[0], old_f1 * scale, old_f2 * scale)
                            velocity_update(s_v, Y_world, world, row0 + 1, lam3[1] - old_f1)
                            velocity_update(s_v, Y_world, world, row0 + 2, lam3[2] - old_f2)

                    wp.tile_scatter_masked(s_lam_contact, c, lam3, thread == 0)

                # ── Phase 1: Joint limits (tile API, scoped helpers) ──
                for i in range(m_contact_rows, m_total):
                    denom = world_diag[world, i]
                    if denom <= 0.0:
                        continue
                    J_row_i = load_J_row(J_world, world, i)
                    jv = dot_Jv(s_v, J_row_i)
                    old_impulse = impulses_flat[world, i]
                    new_impulse = wp.max(old_impulse + omega * (-(jv + rhs_bias[world, i]) / denom), 0.0)
                    delta_impulse = new_impulse - old_impulse
                    impulses_flat[world, i] = new_impulse
                    if delta_impulse != 0.0:
                        velocity_update(s_v, Y_world, world, i, delta_impulse)

                # ── Phase 2: MF constraints (SIMT Warp, shared tile access) ──
                pgs_mf_phase(
                    world,
                    thread,
                    s_v,
                    s_lam_mf,
                    mf_constraint_count,
                    mf_meta,
                    mf_J_a,
                    mf_J_b,
                    mf_MiJt_a,
                    mf_MiJt_b,
                    mf_row_mu,
                    omega,
                )

            # ── STORE PHASE ──
            wp.tile_store(v_out, s_v, offset=(w_dof_start,), bounds_check=False)
            wp.tile_store(impulses_vec3[world], s_lam_contact, bounds_check=False)
            wp.tile_store(mf_impulses[world], s_lam_mf, bounds_check=False)

        name = f"pgs_fused_warp_{max_constraints}_{max_contact_triplets}_{mf_max_constraints}_{max_world_dofs}"
        pgs_fused_warp.__name__ = name
        pgs_fused_warp.__qualname__ = name
        return wp.kernel(enable_backward=False, module="unique", launch_bounds=(32, 12))(pgs_fused_warp)
