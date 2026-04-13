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


@wp.func
def dense_cholesky(
    n: int,
    A: wp.array[float],
    R: wp.array[float],
    A_start: int,
    R_start: int,
    # outputs
    L: wp.array[float],
):
    # compute the Cholesky factorization of A = L L^T with diagonal regularization R
    for j in range(n):
        s = A[A_start + dense_index(n, j, j)] + R[R_start + j]

        for k in range(j):
            r = L[A_start + dense_index(n, j, k)]
            s -= r * r

        s = wp.sqrt(s)
        invS = 1.0 / s

        L[A_start + dense_index(n, j, j)] = s

        for i in range(j + 1, n):
            s = A[A_start + dense_index(n, i, j)]

            for k in range(j):
                s -= L[A_start + dense_index(n, i, k)] * L[A_start + dense_index(n, j, k)]

            L[A_start + dense_index(n, i, j)] = s * invS


@wp.kernel
def cholesky_loop(
    H_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
    R_group: wp.array2d[float],  # [n_arts, n_dofs]
    group_to_art: wp.array[int],
    mass_update_mask: wp.array[int],
    n_dofs: int,
    # output
    L_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
):
    """Non-tiled Cholesky for grouped articulation storage.

    One thread per articulation, loop-based Cholesky decomposition.
    Efficient for small articulations where tile overhead dominates.
    """
    group_idx = wp.tid()
    art_idx = group_to_art[group_idx]

    if mass_update_mask[art_idx] == 0:
        return

    # Cholesky decomposition with regularization: L L^T = H + diag(R)
    for j in range(n_dofs):
        # Compute diagonal element L[j,j]
        s = H_group[group_idx, j, j] + R_group[group_idx, j]

        for k in range(j):
            r = L_group[group_idx, j, k]
            s -= r * r

        s = wp.sqrt(s)
        inv_s = 1.0 / s
        L_group[group_idx, j, j] = s

        # Compute off-diagonal elements L[i,j] for i > j
        for i in range(j + 1, n_dofs):
            s = H_group[group_idx, i, j]

            for k in range(j):
                s -= L_group[group_idx, i, k] * L_group[group_idx, j, k]

            L_group[group_idx, i, j] = s * inv_s


@wp.func
def dense_subs(
    n: int,
    L_start: int,
    b_start: int,
    L: wp.array[float],
    b: wp.array[float],
    # outputs
    x: wp.array[float],
):
    # Solves (L L^T) x = b for x given the Cholesky factor L
    # forward substitution solves the lower triangular system L y = b for y
    for i in range(n):
        s = b[b_start + i]

        for j in range(i):
            s -= L[L_start + dense_index(n, i, j)] * x[b_start + j]

        x[b_start + i] = s / L[L_start + dense_index(n, i, i)]

    # backward substitution solves the upper triangular system L^T x = y for x
    for i in range(n - 1, -1, -1):
        s = x[b_start + i]

        for j in range(i + 1, n):
            s -= L[L_start + dense_index(n, j, i)] * x[b_start + j]

        x[b_start + i] = s / L[L_start + dense_index(n, i, i)]


@wp.func
def dense_solve(
    n: int,
    L_start: int,
    b_start: int,
    A: wp.array[float],
    L: wp.array[float],
    b: wp.array[float],
    # outputs
    x: wp.array[float],
    tmp: wp.array[float],
):
    # helper function to include tmp argument for backward pass
    dense_subs(n, L_start, b_start, L, b, x)


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
def prepare_impulses(
    constraint_counts: wp.array[int],
    max_constraints: int,
    warmstart: int,
    # outputs
    impulses: wp.array[float],
):
    articulation = wp.tid()
    m = constraint_counts[articulation]
    base = articulation * max_constraints

    for i in range(max_constraints):
        if warmstart == 0 or i >= m:
            impulses[base + i] = 0.0


@wp.kernel
def clamp_joint_tau(
    joint_tau: wp.array[float],
    joint_effort_limit: wp.array[float],
):
    tid = wp.tid()

    # Per-DoF effort limit (same convention as MuJoCo actuators)
    limit = joint_effort_limit[tid]

    # If limit <= 0, treat as unlimited
    if limit <= 0.0:
        return

    t = joint_tau[tid]

    if t > limit:
        t = limit
    elif t < -limit:
        t = -limit

    joint_tau[tid] = t


# --- Tile configuration for contact system build ---
# Kernel naming: {op}_{parallelism}
# parallelism: tiled | loop | par_row | par_row_col | par_dof

# Max generalized dofs per articulation we support in the tiled path.
# joint_dof_count per articulation must be <= TILE_DOF or we use fall back
TILE_DOF = wp.constant(49)

# Max constraints per articulation we support in the tiled path.
# dense_max_constraints must be <= TILE_CONSTRAINTS or we use fall back
TILE_CONSTRAINTS = wp.constant(128)

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

        world_rhs[world, i] = rhs


@wp.kernel
def rhs_accum_world_par_art(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    art_to_world: wp.array[int],
    art_size: wp.array[int],
    art_group_idx: wp.array[int],
    art_dof_start: wp.array[int],
    v_hat: wp.array[float],
    group_to_art: wp.array[int],
    J_group: wp.array3d[float],
    n_dofs: int,
    # outputs
    world_rhs: wp.array2d[float],
):
    """
    Accumulate J*v_hat into world RHS for a single size group.

    RHS = J*v + stabilization (already includes stabilization from compute_world_contact_bias)
    This kernel is launched once per size group to accumulate velocity contributions.
    """
    idx = wp.tid()
    art = group_to_art[idx]
    world = art_to_world[art]
    n_constraints = world_constraint_count[world]

    if n_constraints == 0:
        return

    dof_start = art_dof_start[art]

    for c in range(n_constraints):
        jv = float(0.0)
        for d in range(n_dofs):
            jv += J_group[idx, c, d] * v_hat[dof_start + d]
        wp.atomic_add(world_rhs, world, c, jv)  # Add J*v (positive)


@wp.kernel
def prepare_world_impulses(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    warmstart: int,
    # in/out
    world_impulses: wp.array2d[float],
):
    """Initialize world impulses (zero or warmstart)."""
    world = wp.tid()
    m = world_constraint_count[world]

    for i in range(max_constraints):
        if warmstart == 0 or i >= m:
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
def pgs_solve_mf_loop(
    mf_constraint_count: wp.array[int],
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_eff_mass_inv: wp.array2d[float],
    mf_rhs: wp.array2d[float],
    mf_row_type: wp.array2d[int],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    body_to_articulation: wp.array[int],
    art_dof_start: wp.array[int],
    iterations: int,
    omega: float,
    # in/out
    mf_impulses: wp.array2d[float],
    v_out: wp.array[float],
):
    """Matrix-free PGS solver for free rigid body contacts.

    Operates directly on body velocities stored in v_out (generalized coordinates).
    Each iteration recomputes J*v from v_out and applies velocity corrections
    immediately (Gauss-Seidel style).
    """
    world = wp.tid()
    m_count = mf_constraint_count[world]
    if m_count == 0:
        return

    for _ in range(iterations):
        for i in range(m_count):
            eff_inv = mf_eff_mass_inv[world, i]
            if eff_inv <= 0.0:
                continue

            ba = mf_body_a[world, i]
            bb = mf_body_b[world, i]

            # Compute current J * v
            jv = float(0.0)
            if ba >= 0:
                art_a = body_to_articulation[ba]
                ds_a = art_dof_start[art_a]
                for k in range(6):
                    jv += mf_J_a[world, i, k] * v_out[ds_a + k]
            if bb >= 0:
                art_b = body_to_articulation[bb]
                ds_b = art_dof_start[art_b]
                for k in range(6):
                    jv += mf_J_b[world, i, k] * v_out[ds_b + k]

            # PGS update: delta = -(J*v_current + bias) / d_ii
            delta = -(jv + mf_rhs[world, i]) * eff_inv
            new_impulse = mf_impulses[world, i] + omega * delta
            old_impulse = mf_impulses[world, i]

            row_type = mf_row_type[world, i]

            # Project
            if row_type == PGS_CONSTRAINT_TYPE_CONTACT:
                if new_impulse < 0.0:
                    new_impulse = 0.0
            elif row_type == PGS_CONSTRAINT_TYPE_FRICTION:
                parent_idx = mf_row_parent[world, i]
                lambda_n = mf_impulses[world, parent_idx]
                mu_val = mf_row_mu[world, i]
                radius = wp.max(mu_val * lambda_n, 0.0)

                if radius <= 0.0:
                    new_impulse = 0.0
                else:
                    # Sibling friction row
                    if i == parent_idx + 1:
                        sib = parent_idx + 2
                    else:
                        sib = parent_idx + 1

                    mf_impulses[world, i] = new_impulse
                    a = new_impulse
                    b = mf_impulses[world, sib]
                    mag = wp.sqrt(a * a + b * b)
                    if mag > radius:
                        scale = radius / mag
                        new_impulse = a * scale
                        mf_impulses[world, sib] = b * scale
                        # Apply sibling correction to velocities
                        sib_delta = b * scale - b
                        sib_ba = mf_body_a[world, sib]
                        sib_bb = mf_body_b[world, sib]
                        if sib_ba >= 0:
                            sib_art_a = body_to_articulation[sib_ba]
                            sib_ds_a = art_dof_start[sib_art_a]
                            for k in range(6):
                                v_out[sib_ds_a + k] = v_out[sib_ds_a + k] + mf_MiJt_a[world, sib, k] * sib_delta
                        if sib_bb >= 0:
                            sib_art_b = body_to_articulation[sib_bb]
                            sib_ds_b = art_dof_start[sib_art_b]
                            for k in range(6):
                                v_out[sib_ds_b + k] = v_out[sib_ds_b + k] + mf_MiJt_b[world, sib, k] * sib_delta

            delta_impulse = new_impulse - old_impulse
            mf_impulses[world, i] = new_impulse

            # Apply velocity correction: v += M_inv * J^T * delta_impulse
            if ba >= 0:
                art_a2 = body_to_articulation[ba]
                ds_a2 = art_dof_start[art_a2]
                for k in range(6):
                    v_out[ds_a2 + k] = v_out[ds_a2 + k] + mf_MiJt_a[world, i, k] * delta_impulse
            if bb >= 0:
                art_b2 = body_to_articulation[bb]
                ds_b2 = art_dof_start[art_b2]
                for k in range(6):
                    v_out[ds_b2 + k] = v_out[ds_b2 + k] + mf_MiJt_b[world, i, k] * delta_impulse


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
def build_mf_body_map(
    mf_constraint_count: wp.array[int],
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    body_to_articulation: wp.array[int],
    art_dof_start: wp.array[int],
    max_mf_bodies: int,
    # outputs
    mf_body_list: wp.array2d[int],
    mf_body_dof_start: wp.array2d[int],
    mf_body_count: wp.array[int],
    mf_local_body_a: wp.array2d[int],
    mf_local_body_b: wp.array2d[int],
):
    """Build per-world compact body table and local body index mapping.

    Scans all MF constraint body indices, builds a unique body list per world,
    and maps each constraint's body indices to local (compact) indices.
    """
    world = wp.tid()
    m = mf_constraint_count[world]
    if m == 0:
        mf_body_count[world] = 0
        return

    n_bodies = int(0)

    for i in range(m):
        # Process body A
        ba = mf_body_a[world, i]
        if ba >= 0:
            # Search for ba in body_list
            found_a = int(-1)
            for b in range(n_bodies):
                if mf_body_list[world, b] == ba:
                    found_a = b
                    break
            if found_a < 0 and n_bodies < max_mf_bodies:
                found_a = n_bodies
                mf_body_list[world, n_bodies] = ba
                art_a = body_to_articulation[ba]
                mf_body_dof_start[world, n_bodies] = art_dof_start[art_a]
                n_bodies += 1
            mf_local_body_a[world, i] = found_a
        else:
            mf_local_body_a[world, i] = -1

        # Process body B
        bb = mf_body_b[world, i]
        if bb >= 0:
            found_b = int(-1)
            for b in range(n_bodies):
                if mf_body_list[world, b] == bb:
                    found_b = b
                    break
            if found_b < 0 and n_bodies < max_mf_bodies:
                found_b = n_bodies
                mf_body_list[world, n_bodies] = bb
                mf_body_dof_start[world, n_bodies] = art_dof_start[body_to_articulation[bb]]
                n_bodies += 1
            mf_local_body_b[world, i] = found_b
        else:
            mf_local_body_b[world, i] = -1

    mf_body_count[world] = n_bodies


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
def pgs_solve_loop(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    world_diag: wp.array2d[float],
    world_C: wp.array3d[float],
    world_rhs: wp.array2d[float],
    world_impulses: wp.array2d[float],
    iterations: int,
    omega: float,
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
):
    """
    World-level Projected Gauss-Seidel solver.

    Similar to pgs_solve_contacts but operates on 2D world-indexed arrays.
    """
    world = wp.tid()
    m = world_constraint_count[world]

    if m == 0:
        return

    for _ in range(iterations):
        for i in range(m):
            # Compute residual: w = rhs_i + sum_j C_ij * lambda_j
            w = world_rhs[world, i]
            for j in range(m):
                w += world_C[world, i, j] * world_impulses[world, j]

            denom = world_diag[world, i]
            if denom <= 0.0:
                continue

            delta = -w / denom
            new_impulse = world_impulses[world, i] + omega * delta
            row_type = world_row_type[world, i]

            # --- Normal contact or joint limit: lambda_n >= 0 ---
            if row_type == PGS_CONSTRAINT_TYPE_CONTACT or row_type == PGS_CONSTRAINT_TYPE_JOINT_LIMIT:
                if new_impulse < 0.0:
                    new_impulse = 0.0
                world_impulses[world, i] = new_impulse

            # --- Friction: isotropic Coulomb ---
            elif row_type == PGS_CONSTRAINT_TYPE_FRICTION:
                parent_idx = world_row_parent[world, i]
                lambda_n = world_impulses[world, parent_idx]
                mu = world_row_mu[world, i]
                radius = wp.max(mu * lambda_n, 0.0)

                if radius <= 0.0:
                    world_impulses[world, i] = 0.0
                    continue

                world_impulses[world, i] = new_impulse

                # Sibling friction row: constraints are laid out as [normal, friction1, friction2]
                # so friction rows are at parent_idx+1 and parent_idx+2
                if i == parent_idx + 1:
                    sib = parent_idx + 2
                else:
                    sib = parent_idx + 1

                # Project tangent impulses onto friction disk
                a = world_impulses[world, i]
                b = world_impulses[world, sib]

                mag = wp.sqrt(a * a + b * b)
                if mag > radius:
                    scale = radius / mag
                    world_impulses[world, i] = a * scale
                    world_impulses[world, sib] = b * scale

            else:
                world_impulses[world, i] = new_impulse


@wp.kernel
def apply_impulses_world_par_dof(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    art_dof_start: wp.array[int],
    n_dofs: int,
    n_arts: int,
    world_constraint_count: wp.array[int],
    max_constraints: int,
    Y_group: wp.array3d[float],
    world_impulses: wp.array2d[float],
    v_hat: wp.array[float],
    # outputs
    v_out: wp.array[float],
):
    """
    Accumulate velocity changes from world impulses for a single size group.
    DOF-parallelized: each thread handles one (articulation, DOF) pair.

    v_out = v_hat + Y * impulses
    """
    tid = wp.tid()

    # Decode thread index
    local_dof = tid % n_dofs
    idx = tid // n_dofs  # group index

    if idx >= n_arts:
        return

    art = group_to_art[idx]
    world = art_to_world[art]
    n_constraints = world_constraint_count[world]
    dof_start = art_dof_start[art]

    # Inner loop only over constraints
    delta_v = float(0.0)
    for c in range(n_constraints):
        delta_v += Y_group[idx, c, local_dof] * world_impulses[world, c]

    global_dof = dof_start + local_dof
    v_out[global_dof] = v_hat[global_dof] + delta_v


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


@wp.kernel
def hinv_jt_par_row(
    # Grouped Cholesky factor storage [n_arts, n_dofs, n_dofs]
    L_group: wp.array3d[float],
    # Size-grouped Jacobian [n_arts_of_size, max_constraints, n_dofs]
    J_group: wp.array3d[float],
    # Indirection arrays
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    world_constraint_count: wp.array[int],
    # Size parameters
    n_dofs: int,
    max_constraints: int,
    n_arts: int,
    # Output: Y = H^-1 * J^T [n_arts_of_size, max_constraints, n_dofs]
    Y_group: wp.array3d[float],
):
    """
    Compute Y = H^-1 * J^T for one size group using forward/backward substitution.

    Uses L_group (3D array) grouped by DOF size.
    Efficient for small articulations where tile overhead dominates.

    Each thread handles one (articulation, constraint) pair.

    For each articulation in the group, solves:
        L * L^T * Y = J^T
    Using:
        1. Forward substitution: L * Z = J^T
        2. Backward substitution: L^T * Y = Z

    Thread dimension: n_arts_of_size * max_constraints
    """
    tid = wp.tid()

    # Decode thread index
    c = tid % max_constraints  # constraint index
    idx = tid // max_constraints  # group index (articulation within size group)

    # Bounds check for articulation
    if idx >= n_arts:
        return

    art = group_to_art[idx]
    world = art_to_world[art]
    n_constraints = world_constraint_count[world]

    # Early exit if this constraint is beyond the actual count
    if c >= n_constraints:
        return

    # ----------------------------------------------------------------
    # Forward substitution: L * z = j
    # L is lower triangular, so solve from top to bottom
    # ----------------------------------------------------------------
    for i in range(n_dofs):
        # z[i] = (j[i] - sum_{k<i} L[i,k] * z[k]) / L[i,i]
        val = J_group[idx, c, i]

        for k in range(i):
            # z[k] is stored in Y_group temporarily
            val -= L_group[idx, i, k] * Y_group[idx, c, k]

        L_ii = L_group[idx, i, i]
        if L_ii != 0.0:
            Y_group[idx, c, i] = val / L_ii
        else:
            Y_group[idx, c, i] = 0.0

    # ----------------------------------------------------------------
    # Backward substitution: L^T * y = z
    # L^T is upper triangular, so solve from bottom to top
    # z is currently stored in Y_group, we overwrite with y
    # ----------------------------------------------------------------
    for i_rev in range(n_dofs):
        i = n_dofs - 1 - i_rev

        # y[i] = (z[i] - sum_{k>i} L[k,i] * y[k]) / L[i,i]
        # Note: L^T[i,k] = L[k,i], so we read L[k,i] for k > i
        val = Y_group[idx, c, i]  # This is z[i] from forward pass

        for k in range(i + 1, n_dofs):
            val -= L_group[idx, k, i] * Y_group[idx, c, k]

        L_ii = L_group[idx, i, i]
        if L_ii != 0.0:
            Y_group[idx, c, i] = val / L_ii
        else:
            Y_group[idx, c, i] = 0.0


@wp.kernel
def delassus_par_row_col(
    # Size-grouped arrays
    J_group: wp.array3d[float],  # [n_arts_of_size, max_constraints, n_dofs]
    Y_group: wp.array3d[float],  # [n_arts_of_size, max_constraints, n_dofs]
    # Indirection arrays
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    world_constraint_count: wp.array[int],
    # Size parameters
    n_dofs: int,
    max_constraints: int,
    n_arts: int,
    # Output: Delassus matrix C and diagonal (accumulated via atomics)
    world_C: wp.array3d[float],  # [world_count, max_constraints, max_constraints]
    world_diag: wp.array2d[float],  # [world_count, max_constraints]
):
    """
    Accumulate Delassus matrix contribution C += J * Y^T from one size group.

    PARALLELIZED VERSION: Each thread handles one (articulation, i, j) triplet.

    The Delassus matrix is: C = sum_art J_art * H_art^-1 * J_art^T = sum_art J_art * Y_art^T

    Since Y is stored as [constraint, dof], we compute:
        C[i,j] = sum_k J[i,k] * Y[j,k]

    Thread dimension: n_arts_of_size * max_constraints * max_constraints
    """
    tid = wp.tid()

    # Decode thread index
    j = tid % max_constraints
    i = (tid // max_constraints) % max_constraints
    idx = tid // (max_constraints * max_constraints)

    # Bounds check for articulation
    if idx >= n_arts:
        return

    art = group_to_art[idx]
    world = art_to_world[art]
    n_constraints = world_constraint_count[world]

    # Early exit if this (i, j) is beyond the actual constraint count
    if i >= n_constraints or j >= n_constraints:
        return

    # Compute C[i,j] = sum_k J[i,k] * Y[j,k]
    val = float(0.0)
    for k in range(n_dofs):
        val += J_group[idx, i, k] * Y_group[idx, j, k]

    if val != 0.0:
        wp.atomic_add(world_C, world, i, j, val)

    # Also accumulate diagonal separately (only when i == j)
    if i == j and val != 0.0:
        wp.atomic_add(world_diag, world, i, val)


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
def trisolve_loop(
    L_group: wp.array3d[float],  # [n_arts_of_size, n_dofs, n_dofs]
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    n_dofs: int,
    joint_tau: wp.array[float],  # [total_dofs]
    # output
    joint_qdd: wp.array[float],  # [total_dofs]
):
    """
    Solve L * L^T * qdd = tau for grouped articulations using forward/backward substitution.

    Thread dimension: n_arts_of_size (one thread per articulation in this size group)
    """
    idx = wp.tid()
    art = group_to_art[idx]
    dof_start = articulation_dof_start[art]

    # Forward substitution: L * z = tau
    # z is stored temporarily in joint_qdd
    for i in range(n_dofs):
        val = joint_tau[dof_start + i]
        for k in range(i):
            L_ik = L_group[idx, i, k]
            val -= L_ik * joint_qdd[dof_start + k]

        L_ii = L_group[idx, i, i]
        if L_ii != 0.0:
            joint_qdd[dof_start + i] = val / L_ii
        else:
            joint_qdd[dof_start + i] = 0.0

    # Backward substitution: L^T * qdd = z
    for i_rev in range(n_dofs):
        i = n_dofs - 1 - i_rev

        val = joint_qdd[dof_start + i]
        for k in range(i + 1, n_dofs):
            L_ki = L_group[idx, k, i]
            val -= L_ki * joint_qdd[dof_start + k]

        L_ii = L_group[idx, i, i]
        if L_ii != 0.0:
            joint_qdd[dof_start + i] = val / L_ii
        else:
            joint_qdd[dof_start + i] = 0.0


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


@wp.kernel
def vector_add_inplace(a: wp.array[float], b: wp.array[float]):
    """a[i] += b[i]"""
    i = wp.tid()
    a[i] = a[i] + b[i]


@wp.kernel
def compute_delta_and_accumulate(
    v_out: wp.array[float],
    v_snap: wp.array[float],
    v_accum: wp.array[float],
):
    """delta = v_out - v_snap; v_accum += delta; v_snap = delta (reuse buffer for rhs_accum input)"""
    i = wp.tid()
    delta = v_out[i] - v_snap[i]
    v_accum[i] = v_accum[i] + delta
    v_snap[i] = delta


# =============================================================================
# PGS Convergence Diagnostic Kernel (velocity-space mode)
# =============================================================================


@wp.kernel
def pgs_convergence_diagnostic_velocity(
    # Dense constraints
    constraint_count: wp.array[int],
    world_dof_start: wp.array[int],
    rhs: wp.array2d[float],
    impulses: wp.array2d[float],
    prev_impulses: wp.array2d[float],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    row_mu: wp.array2d[float],
    J_world: wp.array3d[float],
    max_constraints: int,
    max_world_dofs: int,
    # MF constraints
    mf_constraint_count: wp.array[int],
    mf_rhs: wp.array2d[float],
    mf_impulses: wp.array2d[float],
    prev_mf_impulses: wp.array2d[float],
    mf_row_type: wp.array2d[int],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_dof_a: wp.array2d[int],
    mf_dof_b: wp.array2d[int],
    mf_max_constraints: int,
    # Velocity
    v_out: wp.array[float],
    # Output: [worlds, 4]
    metrics: wp.array2d[float],
):
    """Compute per-world PGS convergence metrics for velocity-space mode.

    Metrics:
        [0] max|delta_lambda| across all constraint rows
        [1] sum(lambda_n * residual_n) for normal contacts (complementarity gap)
        [2] sum(residual_t^2) for sticking friction contacts (tangent residual energy)
        [3] sum(FB(lambda_n, residual_n)^2) for normal contacts (Fischer-Burmeister)
    """
    world = wp.tid()

    m_dense = constraint_count[world]
    m_mf = mf_constraint_count[world]
    w_dof_start = world_dof_start[world]

    max_dl = float(0.0)
    comp_gap = float(0.0)
    tang_res = float(0.0)
    fb_merit = float(0.0)

    # --- Dense constraints ---
    for i in range(m_dense):
        lam = impulses[world, i]
        prev_lam = prev_impulses[world, i]
        dl = wp.abs(lam - prev_lam)
        if dl > max_dl:
            max_dl = dl

        # Compute residual: r_i = J_i * v + bias_i
        jv = float(0.0)
        for d in range(max_world_dofs):
            jv += J_world[world, i, d] * v_out[w_dof_start + d]
        residual = jv + rhs[world, i]

        rt = row_type[world, i]
        if rt == PGS_CONSTRAINT_TYPE_CONTACT:
            # Normal: complementarity gap and FB
            comp_gap += lam * residual
            fb_val = wp.sqrt(lam * lam + residual * residual) - lam - residual
            fb_merit += fb_val * fb_val
        elif rt == PGS_CONSTRAINT_TYPE_FRICTION:
            # Friction: tangent residual for sticking contacts
            parent_idx = row_parent[world, i]
            lambda_n = impulses[world, parent_idx]
            mu = row_mu[world, i]
            radius = mu * lambda_n
            if radius > 0.0:
                # Check if sticking: |lambda_t| < mu * lambda_n
                # Get sibling
                if i == parent_idx + 1:
                    sib = parent_idx + 2
                else:
                    sib = parent_idx + 1
                lam_t1 = impulses[world, i]
                lam_t2 = impulses[world, sib]
                t_mag = wp.sqrt(lam_t1 * lam_t1 + lam_t2 * lam_t2)
                if t_mag < radius * 0.999:  # sticking (with small tolerance)
                    tang_res += residual * residual

    # --- MF constraints ---
    for i in range(m_mf):
        lam = mf_impulses[world, i]
        prev_lam = prev_mf_impulses[world, i]
        dl = wp.abs(lam - prev_lam)
        if dl > max_dl:
            max_dl = dl

        # Compute residual: r = J_a * v_a + J_b * v_b + bias
        dof_a = mf_dof_a[world, i]
        dof_b = mf_dof_b[world, i]
        jv = float(0.0)
        if dof_a >= 0:
            for k in range(6):
                jv += mf_J_a[world, i, k] * v_out[dof_a + k]
        if dof_b >= 0:
            for k in range(6):
                jv += mf_J_b[world, i, k] * v_out[dof_b + k]
        residual = jv + mf_rhs[world, i]

        rt = mf_row_type[world, i]
        if rt == PGS_CONSTRAINT_TYPE_CONTACT:
            comp_gap += lam * residual
            fb_val = wp.sqrt(lam * lam + residual * residual) - lam - residual
            fb_merit += fb_val * fb_val
        elif rt == PGS_CONSTRAINT_TYPE_FRICTION:
            parent_idx = mf_row_parent[world, i]
            lambda_n = mf_impulses[world, parent_idx]
            mu = mf_row_mu[world, i]
            radius = mu * lambda_n
            if radius > 0.0:
                if i == parent_idx + 1:
                    sib = parent_idx + 2
                else:
                    sib = parent_idx + 1
                lam_t1 = mf_impulses[world, i]
                lam_t2 = mf_impulses[world, sib]
                t_mag = wp.sqrt(lam_t1 * lam_t1 + lam_t2 * lam_t2)
                if t_mag < radius * 0.999:
                    tang_res += residual * residual

    metrics[world, 0] = max_dl
    metrics[world, 1] = comp_gap
    metrics[world, 2] = tang_res
    metrics[world, 3] = fb_merit
