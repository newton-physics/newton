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

"""Warp kernels for SolverMuJoCo."""

from __future__ import annotations

from typing import Any

import warp as wp

from ...sim import JointType

# Custom vector types
vec5 = wp.types.vector(length=5, dtype=wp.float32)
vec10 = wp.types.vector(length=10, dtype=wp.float32)


# Constants
MJ_MINVAL = 2.220446049250313e-16


# Utility functions
@wp.func
def orthogonals(a: wp.vec3):
    y = wp.vec3(0.0, 1.0, 0.0)
    z = wp.vec3(0.0, 0.0, 1.0)
    b = wp.where((-0.5 < a[1]) and (a[1] < 0.5), y, z)
    b = b - a * wp.dot(a, b)
    b = wp.normalize(b)
    if wp.length(a) == 0.0:
        b = wp.vec3(0.0, 0.0, 0.0)
    c = wp.cross(a, b)

    return b, c


@wp.func
def make_frame(a: wp.vec3):
    a = wp.normalize(a)
    b, c = orthogonals(a)

    # fmt: off
    return wp.mat33(
    a.x, a.y, a.z,
    b.x, b.y, b.z,
    c.x, c.y, c.z
  )
    # fmt: on


@wp.func
def write_contact(
    # Data in:
    # In:
    dist_in: float,
    pos_in: wp.vec3,
    frame_in: wp.mat33,
    margin_in: float,
    gap_in: float,
    condim_in: int,
    friction_in: vec5,
    solref_in: wp.vec2f,
    solreffriction_in: wp.vec2f,
    solimp_in: vec5,
    geoms_in: wp.vec2i,
    worldid_in: int,
    contact_id_in: int,
    # Data out:
    contact_dist_out: wp.array(dtype=float),
    contact_pos_out: wp.array(dtype=wp.vec3),
    contact_frame_out: wp.array(dtype=wp.mat33),
    contact_includemargin_out: wp.array(dtype=float),
    contact_friction_out: wp.array(dtype=vec5),
    contact_solref_out: wp.array(dtype=wp.vec2),
    contact_solreffriction_out: wp.array(dtype=wp.vec2),
    contact_solimp_out: wp.array(dtype=vec5),
    contact_dim_out: wp.array(dtype=int),
    contact_geom_out: wp.array(dtype=wp.vec2i),
    contact_worldid_out: wp.array(dtype=int),
):
    # See function write_contact in mujoco_warp, file collision_primitive.py

    cid = contact_id_in
    contact_dist_out[cid] = dist_in
    contact_pos_out[cid] = pos_in
    contact_frame_out[cid] = frame_in
    contact_geom_out[cid] = geoms_in
    contact_worldid_out[cid] = worldid_in
    contact_includemargin_out[cid] = margin_in - gap_in
    contact_dim_out[cid] = condim_in
    contact_friction_out[cid] = friction_in
    contact_solref_out[cid] = solref_in
    contact_solreffriction_out[cid] = solreffriction_in
    contact_solimp_out[cid] = solimp_in


@wp.func
def contact_params(
    geom_condim: wp.array(dtype=int),
    geom_priority: wp.array(dtype=int),
    geom_solmix: wp.array2d(dtype=float),
    geom_solref: wp.array2d(dtype=wp.vec2),
    geom_solimp: wp.array2d(dtype=vec5),
    geom_friction: wp.array2d(dtype=wp.vec3),
    geom_margin: wp.array2d(dtype=float),
    geom_gap: wp.array2d(dtype=float),
    geoms: wp.vec2i,
    worldid: int,
):
    # See function contact_params in mujoco_warp, file collision_primitive.py

    g1 = geoms[0]
    g2 = geoms[1]

    p1 = geom_priority[g1]
    p2 = geom_priority[g2]

    solmix1 = geom_solmix[worldid, g1]
    solmix2 = geom_solmix[worldid, g2]

    mix = solmix1 / (solmix1 + solmix2)
    mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 < MJ_MINVAL), 0.5, mix)
    mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 >= MJ_MINVAL), 0.0, mix)
    mix = wp.where((solmix1 >= MJ_MINVAL) and (solmix2 < MJ_MINVAL), 1.0, mix)
    mix = wp.where(p1 == p2, mix, wp.where(p1 > p2, 1.0, 0.0))

    margin = wp.max(geom_margin[worldid, g1], geom_margin[worldid, g2])
    gap = wp.max(geom_gap[worldid, g1], geom_gap[worldid, g2])

    condim1 = geom_condim[g1]
    condim2 = geom_condim[g2]
    condim = wp.where(p1 == p2, wp.max(condim1, condim2), wp.where(p1 > p2, condim1, condim2))

    max_geom_friction = wp.max(geom_friction[worldid, g1], geom_friction[worldid, g2])
    friction = vec5(
        max_geom_friction[0],
        max_geom_friction[0],
        max_geom_friction[1],
        max_geom_friction[2],
        max_geom_friction[2],
    )

    if geom_solref[worldid, g1].x > 0.0 and geom_solref[worldid, g2].x > 0.0:
        solref = mix * geom_solref[worldid, g1] + (1.0 - mix) * geom_solref[worldid, g2]
    else:
        solref = wp.min(geom_solref[worldid, g1], geom_solref[worldid, g2])

    solreffriction = wp.vec2(0.0, 0.0)

    solimp = mix * geom_solimp[worldid, g1] + (1.0 - mix) * geom_solimp[worldid, g2]

    return margin, gap, condim, friction, solref, solreffriction, solimp


# Kernel functions
@wp.kernel
def convert_newton_contacts_to_mjwarp_kernel(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    # Model:
    geom_condim: wp.array(dtype=int),
    geom_priority: wp.array(dtype=int),
    geom_solmix: wp.array2d(dtype=float),
    geom_solref: wp.array2d(dtype=wp.vec2),
    geom_solimp: wp.array2d(dtype=vec5),
    geom_friction: wp.array2d(dtype=wp.vec3),
    geom_margin: wp.array2d(dtype=float),
    geom_gap: wp.array2d(dtype=float),
    # Newton contacts
    rigid_contact_count: wp.array(dtype=wp.int32),
    rigid_contact_shape0: wp.array(dtype=wp.int32),
    rigid_contact_shape1: wp.array(dtype=wp.int32),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_thickness0: wp.array(dtype=wp.float32),
    rigid_contact_thickness1: wp.array(dtype=wp.float32),
    bodies_per_world: int,
    to_mjc_geom_index: wp.array(dtype=wp.vec2i),
    # Mujoco warp contacts
    nacon_out: wp.array(dtype=int),
    contact_dist_out: wp.array(dtype=float),
    contact_pos_out: wp.array(dtype=wp.vec3),
    contact_frame_out: wp.array(dtype=wp.mat33),
    contact_includemargin_out: wp.array(dtype=float),
    contact_friction_out: wp.array(dtype=vec5),
    contact_solref_out: wp.array(dtype=wp.vec2),
    contact_solreffriction_out: wp.array(dtype=wp.vec2),
    contact_solimp_out: wp.array(dtype=vec5),
    contact_dim_out: wp.array(dtype=int),
    contact_geom_out: wp.array(dtype=wp.vec2i),
    contact_worldid_out: wp.array(dtype=int),
    # Values to clear - see _zero_collision_arrays kernel from mujoco_warp
    nworld_in: int,
    ncollision_out: wp.array(dtype=int),
):
    # See kernel solve_body_contact_positions for reference

    tid = wp.tid()

    # Set number of contacts (for a single world)
    if tid == 0:
        nacon_out[0] = rigid_contact_count[0]
        ncollision_out[0] = 0

    if tid >= rigid_contact_count[0]:
        return

    shape_a = rigid_contact_shape0[tid]
    shape_b = rigid_contact_shape1[tid]

    body_a = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    body_b = -1
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    X_wb_a = wp.transform_identity()
    X_wb_b = wp.transform_identity()
    if body_a >= 0:
        X_wb_a = body_q[body_a]

    if body_b >= 0:
        X_wb_b = body_q[body_b]

    bx_a = wp.transform_point(X_wb_a, rigid_contact_point0[tid])
    bx_b = wp.transform_point(X_wb_b, rigid_contact_point1[tid])

    thickness = rigid_contact_thickness0[tid] + rigid_contact_thickness1[tid]

    n = -rigid_contact_normal[tid]
    dist = wp.dot(n, bx_b - bx_a) - thickness

    # Contact position: use midpoint between contact points (as in XPBD kernel)
    pos = 0.5 * (bx_a + bx_b)

    # Build contact frame
    frame = make_frame(n)

    world_geom_a = to_mjc_geom_index[shape_a]
    world_geom_b = to_mjc_geom_index[shape_b]
    geoms = wp.vec2i(world_geom_a[1], world_geom_b[1])

    # See kernel update_body_mass_ipos_kernel, line below:
    #     worldid = wp.tid() // bodies_per_world
    # which uses the same strategy to determine the world id
    worldid = world_geom_a[0]
    if worldid < 0:
        worldid = world_geom_b[0]

    margin, gap, condim, friction, solref, solreffriction, solimp = contact_params(
        geom_condim,
        geom_priority,
        geom_solmix,
        geom_solref,
        geom_solimp,
        geom_friction,
        geom_margin,
        geom_gap,
        geoms,
        worldid,
    )

    # Use the write_contact function to write all the data
    write_contact(
        dist_in=dist,
        pos_in=pos,
        frame_in=frame,
        margin_in=margin,
        gap_in=gap,
        condim_in=condim,
        friction_in=friction,
        solref_in=solref,
        solreffriction_in=solreffriction,
        solimp_in=solimp,
        geoms_in=geoms,
        worldid_in=worldid,
        contact_id_in=tid,
        contact_dist_out=contact_dist_out,
        contact_pos_out=contact_pos_out,
        contact_frame_out=contact_frame_out,
        contact_includemargin_out=contact_includemargin_out,
        contact_friction_out=contact_friction_out,
        contact_solref_out=contact_solref_out,
        contact_solreffriction_out=contact_solreffriction_out,
        contact_solimp_out=contact_solimp_out,
        contact_dim_out=contact_dim_out,
        contact_geom_out=contact_geom_out,
        contact_worldid_out=contact_worldid_out,
    )


@wp.kernel
def convert_mj_coords_to_warp_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    joints_per_world: int,
    up_axis: int,
    joint_type: wp.array(dtype=wp.int32),
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_dof_dim: wp.array(dtype=wp.int32, ndim=2),
    # outputs
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
):
    worldid, jntid = wp.tid()

    type = joint_type[jntid]
    q_i = joint_q_start[jntid]
    qd_i = joint_qd_start[jntid]
    wq_i = joint_q_start[joints_per_world * worldid + jntid]
    wqd_i = joint_qd_start[joints_per_world * worldid + jntid]

    if type == JointType.FREE:
        # convert position components
        for i in range(3):
            joint_q[wq_i + i] = qpos[worldid, q_i + i]

        # change quaternion order from wxyz to xyzw
        rot = wp.quat(
            qpos[worldid, q_i + 4],
            qpos[worldid, q_i + 5],
            qpos[worldid, q_i + 6],
            qpos[worldid, q_i + 3],
        )
        joint_q[wq_i + 3] = rot[0]
        joint_q[wq_i + 4] = rot[1]
        joint_q[wq_i + 5] = rot[2]
        joint_q[wq_i + 6] = rot[3]
        # for i in range(6):
        #     # convert velocity components
        #     joint_qd[wqd_i + i] = qvel[worldid, qd_i + i]

        joint_qd[wqd_i + 0] = qvel[worldid, qd_i + 0]
        joint_qd[wqd_i + 1] = qvel[worldid, qd_i + 1]
        joint_qd[wqd_i + 2] = qvel[worldid, qd_i + 2]

        w = wp.vec3(qvel[worldid, qd_i + 3], qvel[worldid, qd_i + 4], qvel[worldid, qd_i + 5])
        w = wp.quat_rotate(rot, w)
        joint_qd[wqd_i + 3] = w[0]
        joint_qd[wqd_i + 4] = w[1]
        joint_qd[wqd_i + 5] = w[2]
    elif type == JointType.BALL:
        # change quaternion order from wxyz to xyzw
        rot = wp.quat(
            qpos[worldid, q_i + 1],
            qpos[worldid, q_i + 2],
            qpos[worldid, q_i + 3],
            qpos[worldid, q_i],
        )
        joint_q[wq_i] = rot[0]
        joint_q[wq_i + 1] = rot[1]
        joint_q[wq_i + 2] = rot[2]
        joint_q[wq_i + 3] = rot[3]
        for i in range(3):
            # convert velocity components
            joint_qd[wqd_i + i] = qvel[worldid, qd_i + i]
    else:
        axis_count = joint_dof_dim[jntid, 0] + joint_dof_dim[jntid, 1]
        for i in range(axis_count):
            # convert position components
            joint_q[wq_i + i] = qpos[worldid, q_i + i]
        for i in range(axis_count):
            # convert velocity components
            joint_qd[wqd_i + i] = qvel[worldid, qd_i + i]


@wp.kernel
def convert_warp_coords_to_mj_kernel(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    joints_per_world: int,
    up_axis: int,
    joint_type: wp.array(dtype=wp.int32),
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_dof_dim: wp.array(dtype=wp.int32, ndim=2),
    # outputs
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
):
    worldid, jntid = wp.tid()

    type = joint_type[jntid]
    q_i = joint_q_start[jntid]
    qd_i = joint_qd_start[jntid]
    wq_i = joint_q_start[joints_per_world * worldid + jntid]
    wqd_i = joint_qd_start[joints_per_world * worldid + jntid]

    if type == JointType.FREE:
        # convert position components
        for i in range(3):
            qpos[worldid, q_i + i] = joint_q[wq_i + i]

        rot = wp.quat(
            joint_q[wq_i + 3],
            joint_q[wq_i + 4],
            joint_q[wq_i + 5],
            joint_q[wq_i + 6],
        )
        # change quaternion order from xyzw to wxyz
        qpos[worldid, q_i + 3] = rot[3]
        qpos[worldid, q_i + 4] = rot[0]
        qpos[worldid, q_i + 5] = rot[1]
        qpos[worldid, q_i + 6] = rot[2]
        # for i in range(6):
        #     # convert velocity components
        #     qvel[worldid, qd_i + i] = joint_qd[qd_i + i]

        qvel[worldid, qd_i + 0] = joint_qd[wqd_i + 0]
        qvel[worldid, qd_i + 1] = joint_qd[wqd_i + 1]
        qvel[worldid, qd_i + 2] = joint_qd[wqd_i + 2]

        w = wp.vec3(joint_qd[wqd_i + 3], joint_qd[wqd_i + 4], joint_qd[wqd_i + 5])
        w = wp.quat_rotate_inv(rot, w)
        qvel[worldid, qd_i + 3] = w[0]
        qvel[worldid, qd_i + 4] = w[1]
        qvel[worldid, qd_i + 5] = w[2]

    elif type == JointType.BALL:
        # change quaternion order from xyzw to wxyz
        qpos[worldid, q_i + 0] = joint_q[wq_i + 1]
        qpos[worldid, q_i + 1] = joint_q[wq_i + 2]
        qpos[worldid, q_i + 2] = joint_q[wq_i + 3]
        qpos[worldid, q_i + 3] = joint_q[wq_i + 0]
        for i in range(3):
            # convert velocity components
            qvel[worldid, qd_i + i] = joint_qd[wqd_i + i]
    else:
        axis_count = joint_dof_dim[jntid, 0] + joint_dof_dim[jntid, 1]
        for i in range(axis_count):
            # convert position components
            qpos[worldid, q_i + i] = joint_q[wq_i + i]
        for i in range(axis_count):
            # convert velocity components
            qvel[worldid, qd_i + i] = joint_qd[wqd_i + i]


@wp.kernel
def convert_mjw_contact_to_warp_kernel(
    # inputs
    contact_geom_mapping: wp.array2d(dtype=wp.int32),
    pyramidal_cone: bool,
    mj_nacon: wp.array(dtype=wp.int32),
    mj_contact_frame: wp.array(dtype=wp.mat33f),
    mj_contact_dim: wp.array(dtype=int),
    mj_contact_geom: wp.array(dtype=wp.vec2i),
    mj_contact_efc_address: wp.array2d(dtype=int),
    mj_contact_worldid: wp.array(dtype=wp.int32),
    mj_efc_force: wp.array2d(dtype=float),
    # outputs
    contact_pair: wp.array(dtype=wp.vec2i),
    contact_normal: wp.array(dtype=wp.vec3f),
    contact_force: wp.array(dtype=float),
):
    n_contacts = mj_nacon[0]
    contact_idx = wp.tid()

    if contact_idx >= n_contacts:
        return

    worldid = mj_contact_worldid[contact_idx]
    geoms_mjw = mj_contact_geom[contact_idx]

    normalforce = wp.float(-1.0)

    efc_address0 = mj_contact_efc_address[contact_idx, 0]
    if efc_address0 >= 0:
        normalforce = mj_efc_force[worldid, efc_address0]

        if pyramidal_cone:
            dim = mj_contact_dim[contact_idx]
            for i in range(1, 2 * (dim - 1)):
                normalforce += mj_efc_force[worldid, mj_contact_efc_address[contact_idx, i]]

    pair = wp.vec2i()
    for i in range(2):
        pair[i] = contact_geom_mapping[worldid, geoms_mjw[i]]
    contact_pair[contact_idx] = pair
    contact_normal[contact_idx] = wp.transpose(mj_contact_frame[contact_idx])[0]
    contact_force[contact_idx] = wp.where(normalforce > 0.0, normalforce, 0.0)


@wp.kernel
def apply_mjc_control_kernel(
    joint_target_pos: wp.array(dtype=wp.float32),
    joint_target_vel: wp.array(dtype=wp.float32),
    axis_to_actuator: wp.array2d(dtype=wp.int32),
    axes_per_world: int,
    # outputs
    mj_act: wp.array2d(dtype=wp.float32),
):
    worldid, axisid = wp.tid()
    # Position actuator
    actuator_id = axis_to_actuator[axisid, 0]
    if actuator_id != -1:
        mj_act[worldid, actuator_id] = joint_target_pos[worldid * axes_per_world + axisid]
    # Velocity actuator
    actuator_id = axis_to_actuator[axisid, 1]
    if actuator_id != -1:
        mj_act[worldid, actuator_id] = joint_target_vel[worldid * axes_per_world + axisid]


@wp.kernel
def apply_mjc_body_f_kernel(
    up_axis: int,
    body_q: wp.array(dtype=wp.transform),
    body_f: wp.array(dtype=wp.spatial_vector),
    to_mjc_body_index: wp.array(dtype=wp.int32),
    bodies_per_world: int,
    # outputs
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, bodyid = wp.tid()
    mj_body_id = to_mjc_body_index[bodyid]
    if mj_body_id != -1:
        f = body_f[worldid * bodies_per_world + bodyid]
        v = wp.vec3(f[0], f[1], f[2])
        w = wp.vec3(f[3], f[4], f[5])
        xfrc_applied[worldid, mj_body_id] = wp.spatial_vector(v, w)


@wp.kernel
def apply_mjc_qfrc_kernel(
    body_q: wp.array(dtype=wp.transform),
    joint_f: wp.array(dtype=wp.float32),
    joint_type: wp.array(dtype=wp.int32),
    body_com: wp.array(dtype=wp.vec3),
    joint_child: wp.array(dtype=wp.int32),
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_dof_dim: wp.array2d(dtype=wp.int32),
    joints_per_world: int,
    bodies_per_world: int,
    # outputs
    qfrc_applied: wp.array2d(dtype=wp.float32),
):
    worldid, jntid = wp.tid()
    child = joint_child[jntid]
    # q_i = joint_q_start[jntid]
    qd_i = joint_qd_start[jntid]
    # wq_i = joint_q_start[joints_per_world * worldid + jntid]
    wqd_i = joint_qd_start[joints_per_world * worldid + jntid]
    jtype = joint_type[jntid]
    if jtype == JointType.FREE or jtype == JointType.DISTANCE:
        tf = body_q[worldid * bodies_per_world + child]
        rot = wp.transform_get_rotation(tf)
        # com_world = wp.transform_point(tf, body_com[child])
        v = wp.vec3(joint_f[wqd_i + 0], joint_f[wqd_i + 1], joint_f[wqd_i + 2])
        w = wp.vec3(joint_f[wqd_i + 3], joint_f[wqd_i + 4], joint_f[wqd_i + 5])

        # rotate angular torque to world frame
        w = wp.quat_rotate_inv(rot, w)

        qfrc_applied[worldid, qd_i + 0] = v[0]
        qfrc_applied[worldid, qd_i + 1] = v[1]
        qfrc_applied[worldid, qd_i + 2] = v[2]
        qfrc_applied[worldid, qd_i + 3] = w[0]
        qfrc_applied[worldid, qd_i + 4] = w[1]
        qfrc_applied[worldid, qd_i + 5] = w[2]
    elif jtype == JointType.BALL:
        qfrc_applied[worldid, qd_i + 0] = joint_f[wqd_i + 0]
        qfrc_applied[worldid, qd_i + 1] = joint_f[wqd_i + 1]
        qfrc_applied[worldid, qd_i + 2] = joint_f[wqd_i + 2]
    else:
        for i in range(joint_dof_dim[jntid, 0] + joint_dof_dim[jntid, 1]):
            qfrc_applied[worldid, qd_i + i] = joint_f[wqd_i + i]


@wp.func
def eval_single_articulation_fk(
    joint_start: int,
    joint_end: int,
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_com: wp.array(dtype=wp.vec3),
    # outputs
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    for i in range(joint_start, joint_end):
        parent = joint_parent[i]
        child = joint_child[i]

        # compute transform across the joint
        type = joint_type[i]

        X_pj = joint_X_p[i]
        X_cj = joint_X_c[i]

        # parent anchor frame in world space
        X_wpj = X_pj
        # velocity of parent anchor point in world space
        v_wpj = wp.spatial_vector()
        if parent >= 0:
            X_wp = body_q[parent]
            X_wpj = X_wp * X_wpj
            r_p = wp.transform_get_translation(X_wpj) - wp.transform_point(X_wp, body_com[parent])

            v_wp = body_qd[parent]
            w_p = wp.spatial_bottom(v_wp)
            v_p = wp.spatial_top(v_wp) + wp.cross(w_p, r_p)
            v_wpj = wp.spatial_vector(v_p, w_p)

        q_start = joint_q_start[i]
        qd_start = joint_qd_start[i]
        lin_axis_count = joint_dof_dim[i, 0]
        ang_axis_count = joint_dof_dim[i, 1]

        X_j = wp.transform_identity()
        v_j = wp.spatial_vector(wp.vec3(), wp.vec3())

        if type == JointType.PRISMATIC:
            axis = joint_axis[qd_start]

            q = joint_q[q_start]
            qd = joint_qd[qd_start]

            X_j = wp.transform(axis * q, wp.quat_identity())
            v_j = wp.spatial_vector(axis * qd, wp.vec3())

        if type == JointType.REVOLUTE:
            axis = joint_axis[qd_start]

            q = joint_q[q_start]
            qd = joint_qd[qd_start]

            X_j = wp.transform(wp.vec3(), wp.quat_from_axis_angle(axis, q))
            v_j = wp.spatial_vector(wp.vec3(), axis * qd)

        if type == JointType.BALL:
            r = wp.quat(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2], joint_q[q_start + 3])

            w = wp.vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2])

            X_j = wp.transform(wp.vec3(), r)
            v_j = wp.spatial_vector(wp.vec3(), w)

        if type == JointType.FREE or type == JointType.DISTANCE:
            t = wp.transform(
                wp.vec3(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2]),
                wp.quat(joint_q[q_start + 3], joint_q[q_start + 4], joint_q[q_start + 5], joint_q[q_start + 6]),
            )

            v = wp.spatial_vector(
                wp.vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2]),
                wp.vec3(joint_qd[qd_start + 3], joint_qd[qd_start + 4], joint_qd[qd_start + 5]),
            )

            X_j = t
            v_j = v

        if type == JointType.D6:
            pos = wp.vec3(0.0)
            rot = wp.quat_identity()
            vel_v = wp.vec3(0.0)
            vel_w = wp.vec3(0.0)

            for j in range(lin_axis_count):
                axis = joint_axis[qd_start + j]
                pos += axis * joint_q[q_start + j]
                vel_v += axis * joint_qd[qd_start + j]

            iq = q_start + lin_axis_count
            iqd = qd_start + lin_axis_count
            for j in range(ang_axis_count):
                axis = joint_axis[iqd + j]
                rot = rot * wp.quat_from_axis_angle(axis, joint_q[iq + j])
                vel_w += joint_qd[iqd + j] * axis

            X_j = wp.transform(pos, rot)
            v_j = wp.spatial_vector(vel_v, vel_w)  # vel_v=linear, vel_w=angular

        # transform from world to joint anchor frame at child body
        X_wcj = X_wpj * X_j
        # transform from world to child body frame
        X_wc = X_wcj * wp.transform_inverse(X_cj)

        # transform velocity across the joint to world space
        linear_vel = wp.transform_vector(X_wpj, wp.spatial_top(v_j))
        angular_vel = wp.transform_vector(X_wpj, wp.spatial_bottom(v_j))

        v_wc = v_wpj + wp.spatial_vector(linear_vel, angular_vel)  # spatial vector with (linear, angular) ordering

        body_q[child] = X_wc
        body_qd[child] = v_wc


@wp.kernel
def eval_articulation_fk(
    articulation_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_com: wp.array(dtype=wp.vec3),
    # outputs
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    joint_start = articulation_start[tid]
    joint_end = articulation_start[tid + 1]

    eval_single_articulation_fk(
        joint_start,
        joint_end,
        joint_q,
        joint_qd,
        joint_q_start,
        joint_qd_start,
        joint_type,
        joint_parent,
        joint_child,
        joint_X_p,
        joint_X_c,
        joint_axis,
        joint_dof_dim,
        body_com,
        # outputs
        body_q,
        body_qd,
    )


@wp.kernel
def convert_body_xforms_to_warp_kernel(
    xpos: wp.array2d(dtype=wp.vec3),
    xquat: wp.array2d(dtype=wp.quat),
    to_mjc_body_index: wp.array(dtype=wp.int32),
    bodies_per_world: int,
    # outputs
    body_q: wp.array(dtype=wp.transform),
):
    worldid, bodyid = wp.tid()
    wbi = bodies_per_world * worldid + bodyid
    mbi = to_mjc_body_index[bodyid]
    pos = xpos[worldid, mbi]
    quat = xquat[worldid, mbi]
    # convert from wxyz to xyzw
    quat = wp.quat(quat[1], quat[2], quat[3], quat[0])
    # quat = wp.quat(quat[3], quat[0], quat[1], quat[2])
    # quat = wp.quat_identity()
    # quat = wp.quat_inverse(quat)
    body_q[wbi] = wp.transform(pos, quat)


@wp.kernel
def update_body_mass_ipos_kernel(
    body_com: wp.array(dtype=wp.vec3f),
    body_mass: wp.array(dtype=float),
    bodies_per_world: int,
    up_axis: int,
    body_mapping: wp.array(dtype=int),
    # outputs
    body_ipos: wp.array2d(dtype=wp.vec3f),
    body_mass_out: wp.array2d(dtype=float),
):
    tid = wp.tid()
    worldid = wp.tid() // bodies_per_world
    index_in_world = wp.tid() % bodies_per_world
    mjc_idx = body_mapping[index_in_world]
    if mjc_idx == -1:
        return

    # update COM position
    if up_axis == 1:
        body_ipos[worldid, mjc_idx] = wp.vec3f(body_com[tid][0], -body_com[tid][2], body_com[tid][1])
    else:
        body_ipos[worldid, mjc_idx] = body_com[tid]

    # update mass
    body_mass_out[worldid, mjc_idx] = body_mass[tid]


@wp.kernel
def update_body_inertia_kernel(
    body_inertia: wp.array(dtype=wp.mat33f),
    bodies_per_world: int,
    body_mapping: wp.array(dtype=int),
    # outputs
    body_inertia_out: wp.array2d(dtype=wp.vec3f),
    body_iquat_out: wp.array2d(dtype=wp.quatf),
):
    tid = wp.tid()
    worldid = wp.tid() // bodies_per_world
    index_in_world = wp.tid() % bodies_per_world
    mjc_idx = body_mapping[index_in_world]
    if mjc_idx == -1:
        return

    # Get inertia tensor
    I = body_inertia[tid]

    # Calculate eigenvalues and eigenvectors
    eigenvectors, eigenvalues = wp.eig3(I)

    # transpose eigenvectors to allow reshuffling by indexing rows.
    vecs_transposed = wp.transpose(eigenvectors)

    # Bubble sort for 3 elements in descending order
    for i in range(2):
        for j in range(2 - i):
            if eigenvalues[j] < eigenvalues[j + 1]:
                # Swap eigenvalues
                temp_val = eigenvalues[j]
                eigenvalues[j] = eigenvalues[j + 1]
                eigenvalues[j + 1] = temp_val
                # Swap eigenvectors
                temp_vec = vecs_transposed[j]
                vecs_transposed[j] = vecs_transposed[j + 1]
                vecs_transposed[j + 1] = temp_vec

    # Convert eigenvectors to quaternion (xyzw format)
    q = wp.quat_from_matrix(wp.transpose(vecs_transposed))
    q = wp.normalize(q)

    # Convert from xyzw to wxyz format
    q = wp.quat(q[1], q[2], q[3], q[0])

    # Store results
    body_inertia_out[worldid, mjc_idx] = eigenvalues
    body_iquat_out[worldid, mjc_idx] = q


@wp.kernel(module="unique", enable_backward=False)
def repeat_array_kernel(
    src: wp.array(dtype=Any),
    nelems_per_world: int,
    dst: wp.array(dtype=Any),
):
    tid = wp.tid()
    src_idx = tid % nelems_per_world
    dst[tid] = src[src_idx]


@wp.kernel
def update_axis_properties_kernel(
    joint_target_kp: wp.array(dtype=float),
    joint_target_kv: wp.array(dtype=float),
    joint_effort_limit: wp.array(dtype=float),
    axis_to_actuator: wp.array2d(dtype=wp.int32),
    axes_per_world: int,
    # outputs
    actuator_bias: wp.array2d(dtype=vec10),
    actuator_gain: wp.array2d(dtype=vec10),
    actuator_forcerange: wp.array2d(dtype=wp.vec2f),
):
    """Update actuator force ranges based on joint effort limits."""
    tid = wp.tid()
    worldid = tid // axes_per_world
    axis_in_world = tid % axes_per_world

    kp = joint_target_kp[tid]
    kv = joint_target_kv[tid]
    effort_limit = joint_effort_limit[tid]

    # Update position actuator (index 0)
    pos_actuator_idx = axis_to_actuator[axis_in_world, 0]
    if pos_actuator_idx >= 0:  # Valid actuator
        actuator_bias[worldid, pos_actuator_idx][1] = -kp
        actuator_gain[worldid, pos_actuator_idx][0] = kp
        actuator_forcerange[worldid, pos_actuator_idx] = wp.vec2f(-effort_limit, effort_limit)

    # Update velocity actuator (index 1)
    vel_actuator_idx = axis_to_actuator[axis_in_world, 1]
    if vel_actuator_idx >= 0:  # Valid actuator
        actuator_bias[worldid, vel_actuator_idx][2] = -kv
        actuator_gain[worldid, vel_actuator_idx][0] = kv
        actuator_forcerange[worldid, vel_actuator_idx] = wp.vec2f(-effort_limit, effort_limit)


@wp.kernel
def update_dof_properties_kernel(
    joint_armature: wp.array(dtype=float),
    joint_friction: wp.array(dtype=float),
    dofs_per_world: int,
    # outputs
    dof_armature: wp.array2d(dtype=float),
    dof_frictionloss: wp.array2d(dtype=float),
):
    """Update DOF armature and friction loss values."""
    tid = wp.tid()
    worldid = tid // dofs_per_world
    dof_in_world = tid % dofs_per_world

    # update armature
    dof_armature[worldid, dof_in_world] = joint_armature[tid]

    # update friction loss
    dof_frictionloss[worldid, dof_in_world] = joint_friction[tid]


@wp.kernel
def update_joint_margin_kernel(
    joint_dof_margin: wp.array(dtype=float),
    joint_qd_start: wp.array(dtype=wp.int32),
    dofs_per_world: int,
    joints_per_world: int,
    # outputs
    jnt_margin: wp.array2d(dtype=float),
):
    """Update joint margin values from per-DOF margins in Newton to per-joint margins in MuJoCo.

    In Newton, margin is stored per-DOF, but in MuJoCo it's stored per-joint.
    We use the first DOF's margin value for each joint.
    """
    tid = wp.tid()
    worldid = tid // dofs_per_world
    dof_in_world = tid % dofs_per_world

    # Find which joint this DOF belongs to
    # We need to find the joint index where qd_start <= dof_in_world < qd_start + dof_count
    for joint_idx in range(joints_per_world):
        qd_start = joint_qd_start[joint_idx]
        if qd_start == dof_in_world:
            # This is the first DOF of this joint, update the joint's margin
            jnt_margin[worldid, joint_idx] = joint_dof_margin[tid]
            break


@wp.kernel
def update_joint_transforms_kernel(
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_dof_start: wp.array(dtype=wp.int32),
    joint_dof_dim: wp.array2d(dtype=wp.int32),
    joint_original_axis: wp.array(dtype=wp.vec3),
    joint_child: wp.array(dtype=wp.int32),
    joint_type: wp.array(dtype=wp.int32),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    joint_mjc_dof_start: wp.array(dtype=wp.int32),
    body_mapping: wp.array(dtype=wp.int32),
    joints_per_world: int,
    # outputs
    joint_pos: wp.array2d(dtype=wp.vec3),
    joint_axis: wp.array2d(dtype=wp.vec3),
    joint_solref: wp.array2d(dtype=wp.vec2),
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
):
    tid = wp.tid()
    worldid = tid // joints_per_world
    joint_in_world = tid % joints_per_world

    jtype = joint_type[tid]
    if jtype == JointType.FREE:
        # we do not set joint transforms for free joints
        return

    child_xform = joint_X_c[tid]
    parent_xform = joint_X_p[tid]
    lin_axis_count = joint_dof_dim[tid, 0]
    ang_axis_count = joint_dof_dim[tid, 1]
    newton_dof_start = joint_dof_start[tid]
    mjc_dof_start = joint_mjc_dof_start[joint_in_world]
    if mjc_dof_start == -1:
        # this should not happen
        wp.printf("Joint %i has no MuJoCo DOF start index\n", joint_in_world)
        return

    # update linear dofs
    for i in range(lin_axis_count):
        newton_dof_index = newton_dof_start + i
        axis = joint_original_axis[newton_dof_index]
        ai = mjc_dof_start + i
        joint_axis[worldid, ai] = wp.quat_rotate(child_xform.q, axis)
        joint_pos[worldid, ai] = child_xform.p
        # update joint limit solref using negative convention
        if joint_limit_ke[newton_dof_index] > 0:
            joint_solref[worldid, ai] = wp.vec2(-joint_limit_ke[newton_dof_index], -joint_limit_kd[newton_dof_index])

    # update angular dofs
    for i in range(ang_axis_count):
        newton_dof_index = newton_dof_start + lin_axis_count + i
        axis = joint_original_axis[newton_dof_index]
        ai = mjc_dof_start + lin_axis_count + i
        joint_axis[worldid, ai] = wp.quat_rotate(child_xform.q, axis)
        joint_pos[worldid, ai] = child_xform.p
        # update joint limit solref using negative convention
        if joint_limit_ke[newton_dof_index] > 0:
            joint_solref[worldid, ai] = wp.vec2(-joint_limit_ke[newton_dof_index], -joint_limit_kd[newton_dof_index])

    # update body pos and quat from parent joint transform
    child = joint_child[joint_in_world]  # Newton body id
    body_id = body_mapping[child]  # MuJoCo body id
    tf = parent_xform * wp.transform_inverse(child_xform)
    body_pos[worldid, body_id] = tf.p
    body_quat[worldid, body_id] = wp.quat(tf.q.w, tf.q.x, tf.q.y, tf.q.z)


@wp.kernel(enable_backward=False)
def update_shape_mappings_kernel(
    geom_to_shape_idx: wp.array(dtype=wp.int32),
    geom_is_static: wp.array(dtype=bool),
    shape_range_len: int,
    first_env_shape_base: int,
    # output
    full_shape_mapping: wp.array(dtype=wp.vec2i),
    reverse_shape_mapping: wp.array(dtype=wp.int32, ndim=2),
):
    env_idx, geom_idx = wp.tid()
    template_or_static_idx = geom_to_shape_idx[geom_idx]
    if template_or_static_idx < 0:
        return

    # Check if this is a static shape using the precomputed mask
    # For static shapes, template_or_static_idx is the absolute Newton shape index
    # For non-static shapes, template_or_static_idx is 0-based offset from first env's first shape
    is_static = geom_is_static[geom_idx]

    if is_static:
        # Static shape - use absolute index
        # Store world ID as -1 (sentinel) since static shapes exist in all worlds
        # The actual world ID will be determined from the non-static shape in the contact pair
        global_shape_idx = template_or_static_idx
        if env_idx == 0:
            # Only store the mapping once for static shapes (use env 0's geom index)
            full_shape_mapping[global_shape_idx] = wp.vec2i(-1, geom_idx)
        reverse_shape_mapping[env_idx, geom_idx] = global_shape_idx
    else:
        # Non-static shape - compute the absolute Newton shape index for this environment
        # template_or_static_idx is 0-based offset within first_group shapes
        global_shape_idx = first_env_shape_base + template_or_static_idx + env_idx * shape_range_len
        full_shape_mapping[global_shape_idx] = wp.vec2i(env_idx, geom_idx)
        reverse_shape_mapping[env_idx, geom_idx] = global_shape_idx


@wp.kernel
def update_model_properties_kernel(
    # Newton model properties
    gravity_src: wp.array(dtype=wp.vec3),
    # MuJoCo model properties
    gravity_dst: wp.array(dtype=wp.vec3f),
):
    world_idx = wp.tid()
    gravity_dst[world_idx] = gravity_src[0]


@wp.kernel
def update_geom_properties_kernel(
    shape_collision_radius: wp.array(dtype=float),
    shape_mu: wp.array(dtype=float),
    shape_ke: wp.array(dtype=float),
    shape_kd: wp.array(dtype=float),
    shape_size: wp.array(dtype=wp.vec3f),
    shape_transform: wp.array(dtype=wp.transform),
    to_newton_shape_index: wp.array2d(dtype=wp.int32),
    geom_type: wp.array(dtype=int),
    GEOM_TYPE_MESH: int,
    geom_dataid: wp.array(dtype=int),
    mesh_pos: wp.array(dtype=wp.vec3),
    mesh_quat: wp.array(dtype=wp.quat),
    torsional_friction: float,
    rolling_friction: float,
    # outputs
    geom_rbound: wp.array2d(dtype=float),
    geom_friction: wp.array2d(dtype=wp.vec3f),
    geom_solref: wp.array2d(dtype=wp.vec2f),
    geom_size: wp.array2d(dtype=wp.vec3f),
    geom_pos: wp.array2d(dtype=wp.vec3f),
    geom_quat: wp.array2d(dtype=wp.quatf),
):
    """Update geom properties from Newton shape properties."""
    worldid, geom_idx = wp.tid()

    shape_idx = to_newton_shape_index[worldid, geom_idx]
    if shape_idx < 0:
        return

    # update bounding radius
    geom_rbound[worldid, geom_idx] = shape_collision_radius[shape_idx]

    # update friction (slide, torsion, roll)
    mu = shape_mu[shape_idx]
    geom_friction[worldid, geom_idx] = wp.vec3f(mu, torsional_friction * mu, rolling_friction * mu)

    # update geom_solref (timeconst, dampratio) using stiffness and damping
    # we don't use negative convention for geom_solref because MJWarp's code
    # combining geoms' negative solrefs looks suspicious
    ke, kd = shape_ke[shape_idx], shape_kd[shape_idx]
    if ke > 0.0 and kd > 0.0:
        # kd = 2 / timeconst -> timeconst = 2 / kd
        # ke = 1 / (timeconst^2 * dampratio^2) -> dampratio = sqrt(1 / (timeconst^2 * ke))
        timeconst = 2.0 / kd
        dampratio = wp.sqrt(1.0 / (timeconst * timeconst * ke))
        geom_solref[worldid, geom_idx] = wp.vec2f(timeconst, dampratio)
    else:
        geom_solref[worldid, geom_idx] = wp.vec2f(0.02, 1.0)

    # update size
    geom_size[worldid, geom_idx] = shape_size[shape_idx]

    # update position and orientation

    # get shape transform
    tf = shape_transform[shape_idx]

    # check if this is a mesh geom and apply mesh transformation
    if geom_type[geom_idx] == GEOM_TYPE_MESH:
        mesh_id = geom_dataid[geom_idx]
        mesh_p = mesh_pos[mesh_id]
        mesh_q = mesh_quat[mesh_id]
        mesh_tf = wp.transform(mesh_p, wp.quat(mesh_q.y, mesh_q.z, mesh_q.w, mesh_q.x))
        tf = tf * mesh_tf

    # store position and orientation
    geom_pos[worldid, geom_idx] = tf.p
    geom_quat[worldid, geom_idx] = wp.quat(tf.q.w, tf.q.x, tf.q.y, tf.q.z)
