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

import os
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ...core.types import nparray, override
from ...geometry import MESH_MAXHULLVERT, GeoType, ShapeFlags
from ...sim import (
    Contacts,
    Control,
    EqType,
    JointMode,
    JointType,
    Model,
    State,
    color_graph,
    plot_graph,
)
from ...utils import topological_sort
from ..flags import SolverNotifyFlags
from ..solver import SolverBase


class vec10f(wp.types.vector(length=10, dtype=float)):
    pass


if TYPE_CHECKING:
    from mujoco import MjData, MjModel
    from mujoco_warp import Data as MjWarpData
    from mujoco_warp import Model as MjWarpModel
else:
    MjModel = object
    MjData = object
    MjWarpModel = object
    MjWarpData = object


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


# Define vec5 as a 5-element vector of float32, matching MuJoCo's convention
vec5 = wp.types.vector(length=5, dtype=wp.float32)


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


MJ_MINVAL = 2.220446049250313e-16


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
    joint_target: wp.array(dtype=wp.float32),
    axis_mode: wp.array(dtype=wp.int32),
    axis_to_actuator: wp.array(dtype=wp.int32),
    axes_per_world: int,
    # outputs
    mj_act: wp.array2d(dtype=wp.float32),
):
    worldid, axisid = wp.tid()
    actuator_id = axis_to_actuator[axisid]
    if actuator_id != -1:
        if axis_mode[axisid] != JointMode.NONE:
            mj_act[worldid, actuator_id] = joint_target[worldid * axes_per_world + axisid]
        else:
            mj_act[worldid, actuator_id] = 0.0


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
    joint_dof_mode: wp.array(dtype=int),
    joint_target_kp: wp.array(dtype=float),
    joint_target_kv: wp.array(dtype=float),
    joint_effort_limit: wp.array(dtype=float),
    axis_to_actuator: wp.array(dtype=wp.int32),
    axes_per_world: int,
    # outputs
    actuator_bias: wp.array2d(dtype=vec10f),
    actuator_gain: wp.array2d(dtype=vec10f),
    actuator_forcerange: wp.array2d(dtype=wp.vec2f),
):
    """Update actuator force ranges based on joint effort limits."""
    tid = wp.tid()
    worldid = tid // axes_per_world
    axis_in_world = tid % axes_per_world

    actuator_idx = axis_to_actuator[axis_in_world]
    if actuator_idx >= 0:  # Valid actuator
        kp = joint_target_kp[tid]
        kv = joint_target_kv[tid]
        mode = joint_dof_mode[tid]

        if mode == JointMode.TARGET_POSITION:
            # bias = vec10f(0.0, -kp, -kv, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            # gain = vec10f(kp, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            actuator_bias[worldid, actuator_idx][1] = -kp
            actuator_bias[worldid, actuator_idx][2] = -kv
            actuator_gain[worldid, actuator_idx][0] = kp
        elif mode == JointMode.TARGET_VELOCITY:
            # bias = vec10f(0.0, 0.0, -kv, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            # gain = vec10f(kv, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            actuator_bias[worldid, actuator_idx][1] = 0.0
            actuator_bias[worldid, actuator_idx][2] = -kv
            actuator_gain[worldid, actuator_idx][0] = kv
        else:
            # bias = [0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0]
            # gain = [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            actuator_bias[worldid, actuator_idx][1] = 0.0
            actuator_bias[worldid, actuator_idx][2] = 0.0
            actuator_gain[worldid, actuator_idx][0] = 1.0

        effort_limit = joint_effort_limit[tid]
        actuator_forcerange[worldid, actuator_idx] = wp.vec2f(-effort_limit, effort_limit)


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
def update_joint_transforms_kernel(
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_dof_start: wp.array(dtype=wp.int32),
    joint_dof_dim: wp.array2d(dtype=wp.int32),
    joint_original_axis: wp.array(dtype=wp.vec3),
    joint_child: wp.array(dtype=wp.int32),
    joint_type: wp.array(dtype=wp.int32),
    joint_mjc_dof_start: wp.array(dtype=wp.int32),
    body_mapping: wp.array(dtype=wp.int32),
    joints_per_world: int,
    # outputs
    joint_pos: wp.array2d(dtype=wp.vec3),
    joint_axis: wp.array2d(dtype=wp.vec3),
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
        axis = joint_original_axis[newton_dof_start + i]
        ai = mjc_dof_start + i
        joint_axis[worldid, ai] = wp.quat_rotate(child_xform.q, axis)
        joint_pos[worldid, ai] = child_xform.p

    # update angular dofs
    for i in range(ang_axis_count):
        axis = joint_original_axis[newton_dof_start + lin_axis_count + i]
        ai = mjc_dof_start + lin_axis_count + i
        joint_axis[worldid, ai] = wp.quat_rotate(child_xform.q, axis)
        joint_pos[worldid, ai] = child_xform.p

    # update body pos and quat from parent joint transform
    child = joint_child[joint_in_world]  # Newton body id
    body_id = body_mapping[child]  # MuJoCo body id
    tf = parent_xform * wp.transform_inverse(child_xform)
    body_pos[worldid, body_id] = tf.p
    body_quat[worldid, body_id] = wp.quat(tf.q.w, tf.q.x, tf.q.y, tf.q.z)


@wp.kernel(enable_backward=False)
def update_incoming_shape_xform_kernel(
    geom_to_shape_idx: wp.array(dtype=wp.int32),
    geom_is_static: wp.array(dtype=bool),
    shape_transform: wp.array(dtype=wp.transform),
    shape_range_len: int,
    first_env_shape_base: int,
    geom_pos: wp.array(dtype=wp.vec3),
    geom_quat: wp.array(dtype=wp.quat),
    # output
    full_shape_mapping: wp.array(dtype=wp.vec2i),
    reverse_shape_mapping: wp.array(dtype=wp.int32, ndim=2),
    shape_incoming_xform: wp.array(dtype=wp.transform),
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
    # Update incoming shape transforms
    # compute the difference between the original shape transform
    # and the transform after applying the joint child transform
    # and the transform MuJoCo does on mesh geoms
    original_tf = shape_transform[global_shape_idx]
    mjc_p = geom_pos[geom_idx]
    q = geom_quat[geom_idx]
    # convert quat from wxyz to xyzw
    mjc_q = wp.quat(q[1], q[2], q[3], q[0])
    mjc_tf = wp.transform(mjc_p, mjc_q)
    shape_incoming_xform[global_shape_idx] = mjc_tf * wp.transform_inverse(original_tf)


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
    shape_incoming_xform: wp.array(dtype=wp.transform),
    torsional_friction: float,
    rolling_friction: float,
    contact_stiffness_time_const: float,
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

    # update solref (stiffness, damping as time constants)
    # MuJoCo uses time constants, Newton uses direct stiffness/damping
    # convert using heuristic: time_const = sqrt(mass/stiffness)
    ke = shape_ke[shape_idx]
    kd = shape_kd[shape_idx]
    if ke > 0.0:
        # use provided time constant for stiffness
        time_const_stiff = contact_stiffness_time_const
        if kd > 0.0:
            time_const_damp = kd / (2.0 * wp.sqrt(ke))
        else:
            time_const_damp = 1.0
    else:
        time_const_stiff = contact_stiffness_time_const
        time_const_damp = 1.0
    geom_solref[worldid, geom_idx] = wp.vec2f(time_const_stiff, time_const_damp)

    # update size
    geom_size[worldid, geom_idx] = shape_size[shape_idx]

    # update position and orientation
    tf = shape_transform[shape_idx]
    incoming_xform = shape_incoming_xform[shape_idx]
    tf = incoming_xform * tf
    pos = tf.p
    quat = tf.q
    geom_pos[worldid, geom_idx] = pos
    geom_quat[worldid, geom_idx] = wp.quatf(quat.w, quat.x, quat.y, quat.z)


class SolverMuJoCo(SolverBase):
    """
    This solver provides an interface to simulate physics using the `MuJoCo <https://github.com/google-deepmind/mujoco>`_ physics engine,
    optimized with GPU acceleration through `mujoco_warp <https://github.com/google-deepmind/mujoco_warp>`_. It supports both MuJoCo and
    mujoco_warp backends, enabling efficient simulation of articulated systems with
    contacts and constraints.

    .. note::

        - This solver requires `mujoco_warp`_ and its dependencies to be installed.
        - For installation instructions, see the `mujoco_warp`_ repository.

    Example
    -------

    .. code-block:: python

        solver = newton.solvers.SolverMuJoCo(model)

        # simulation loop
        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

    Debugging
    ---------

    To debug the SolverMuJoCo, you can save the MuJoCo model that is created from the :class:`newton.Model` in the constructor of the SolverMuJoCo:

    .. code-block:: python

        solver = newton.solvers.SolverMuJoCo(model, save_to_mjcf="model.xml")

    This will save the MuJoCo model as an MJCF file, which can be opened in the MuJoCo simulator.

    It is also possible to visualize the simulation running in the SolverMuJoCo through MuJoCo's own viewer.
    This may help to debug the simulation and see how the MuJoCo model looks like when it is created from the Newton model.

    .. code-block:: python

        import newton

        solver = newton.solvers.SolverMuJoCo(model)

        for _ in range(num_frames):
            # step the solver
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

            solver.render_mujoco_viewer()
    """

    # Class variables to cache the imported modules
    _mujoco = None
    _mujoco_warp = None

    @classmethod
    def import_mujoco(cls):
        """Import the MuJoCo Warp dependencies and cache them as class variables."""
        if cls._mujoco is None or cls._mujoco_warp is None:
            try:
                import mujoco  # noqa: PLC0415
                import mujoco_warp  # noqa: PLC0415

                cls._mujoco = mujoco
                cls._mujoco_warp = mujoco_warp
            except ImportError as e:
                raise ImportError(
                    "MuJoCo backend not installed. Please refer to https://github.com/google-deepmind/mujoco_warp for installation instructions."
                ) from e
        return cls._mujoco, cls._mujoco_warp

    def __init__(
        self,
        model: Model,
        *,
        mjw_model: MjWarpModel | None = None,
        mjw_data: MjWarpData | None = None,
        separate_worlds: bool | None = None,
        njmax: int | None = None,
        ncon_per_world: int | None = None,
        iterations: int = 20,
        ls_iterations: int = 10,
        solver: int | str = "cg",
        integrator: int | str = "implicitfast",
        cone: int | str = "pyramidal",
        impratio: float = 1.0,
        use_mujoco_cpu: bool = False,
        disable_contacts: bool = False,
        default_actuator_gear: float | None = None,
        actuator_gears: dict[str, float] | None = None,
        update_data_interval: int = 1,
        save_to_mjcf: str | None = None,
        contact_stiffness_time_const: float = 0.02,
        ls_parallel: bool = False,
        use_mujoco_contacts: bool = True,
        joint_solref_limit: tuple[float, float] | None = None,
        joint_solimp_limit: tuple[float, float, float, float, float] | None = None,
        tolerance: float = 1e-6,
        ls_tolerance: float = 0.01,
    ):
        """
        Args:
            model (Model): the model to be simulated.
            mjw_model (MjWarpModel | None): Optional pre-existing MuJoCo Warp model. If provided with `mjw_data`, conversion from Newton model is skipped.
            mjw_data (MjWarpData | None): Optional pre-existing MuJoCo Warp data. If provided with `mjw_model`, conversion from Newton model is skipped.
            separate_worlds (bool | None): If True, each Newton world is mapped to a separate MuJoCo world. Defaults to `not use_mujoco_cpu`.
            njmax (int): Maximum number of constraints per world. If None, a default value is estimated from the initial state. Note that the larger of the user-provided value or the default value is used.
            ncon_per_world (int | None): Number of contact points per world. If None, a default value is estimated from the initial state. Note that the larger of the user-provided value or the default value is used.
            iterations (int): Number of solver iterations.
            ls_iterations (int): Number of line search iterations for the solver.
            solver (int | str): Solver type. Can be "cg" or "newton", or their corresponding MuJoCo integer constants.
            integrator (int | str): Integrator type. Can be "euler", "rk4", or "implicitfast", or their corresponding MuJoCo integer constants.
            cone (int | str): The type of contact friction cone. Can be "pyramidal", "elliptic", or their corresponding MuJoCo integer constants.
            impratio (float): Frictional-to-normal constraint impedance ratio.
            use_mujoco_cpu (bool): If True, use the MuJoCo-C CPU backend instead of `mujoco_warp`.
            disable_contacts (bool): If True, disable contact computation in MuJoCo.
            register_collision_groups (bool): If True, register collision groups from the Newton model in MuJoCo.
            default_actuator_gear (float | None): Default gear ratio for all actuators. Can be overridden by `actuator_gears`.
            actuator_gears (dict[str, float] | None): Dictionary mapping joint names to specific gear ratios, overriding the `default_actuator_gear`.
            update_data_interval (int): Frequency (in simulation steps) at which to update the MuJoCo Data object from the Newton state. If 0, Data is never updated after initialization.
            save_to_mjcf (str | None): Optional path to save the generated MJCF model file.
            contact_stiffness_time_const (float): Time constant for contact stiffness in MuJoCo's solver reference model. Defaults to 0.02 (20ms). Can be set to match the simulation timestep for tighter coupling.
            ls_parallel (bool): If True, enable parallel line search in MuJoCo. Defaults to False.
            use_mujoco_contacts (bool): If True, use the MuJoCo contact solver. If False, use the Newton contact solver (newton contacts must be passed in through the step function in that case).
            joint_solref_limit (tuple[float, float] | None): Global solver reference parameters for all joint limits. If provided, applies these solref values to all joints created. Defaults to None (uses MuJoCo defaults).
            joint_solimp_limit (tuple[float, float, float, float, float] | None): Global solver impedance parameters for all joint limits. If provided, applies these solimp values to all joints created. Defaults to None (uses MuJoCo defaults).
            tolerance (float | None): Solver tolerance for early termination of the iterative solver. Defaults to 1e-6.
            ls_tolerance (float | None): Solver tolerance for early termination of the line search. Defaults to 0.01.
        """
        super().__init__(model)
        # Import and cache MuJoCo modules (only happens once per class)
        mujoco, _ = self.import_mujoco()
        self.contact_stiffness_time_const = contact_stiffness_time_const
        self.joint_solref_limit = joint_solref_limit
        self.joint_solimp_limit = joint_solimp_limit

        if use_mujoco_cpu and not use_mujoco_contacts:
            print("Setting use_mujoco_contacts to False has no effect when use_mujoco_cpu is True")

        self.shape_incoming_xform: wp.array(dtype=wp.transform) | None = None
        """The transform applied to Newton's shape frame to match MuJoCo's geom frame. This only affects mesh shapes (MuJoCo aligns them with their inertial frames). Shape [shape_count], dtype transform."""
        self.joint_mjc_dof_start: wp.array(dtype=wp.int32) | None = None
        """Mapping from Newton joint index to the start index of its joint axes in MuJoCo. Only defined for the joint indices of the first world in Newton, defaults to -1 otherwise. Shape [joint_count], dtype int32."""
        self.mjc_axis_to_actuator: wp.array(dtype=int) | None = None
        """Mapping from Newton joint axis index to MJC actuator index. Shape [dof_count], dtype int32."""
        self.to_mjc_body_index: wp.array(dtype=int) | None = None
        """Mapping from MuJoCo body index to Newton body index (skip world body index -1). Shape [bodies_per_world], dtype int32."""
        self.to_newton_shape_index: wp.array(dtype=int, ndim=2) | None = None
        """Mapping from MuJoCo [worldid, geom index] to Newton shape index. This is used to map MuJoCo geoms to Newton shapes."""
        self.to_mjc_geom_index: wp.array(dtype=wp.vec2i) | None = None
        """Mapping from Newton shape index to MuJoCo [worldid, geom index]."""

        self.selected_shapes: wp.array(dtype=int) | None = None
        """Indices of Newton shapes that are used in the MuJoCo model (includes non-instantiated visual-only shapes) for the first world as a basis for replicating the nworlds worlds in MuJoCo Warp."""
        self.selected_joints: wp.array(dtype=int) | None = None
        """Indices of Newton joints that are used in the MuJoCo model for the first world as a basis for replicating the nworlds worlds in MuJoCo Warp."""
        self.selected_bodies: wp.array(dtype=int) | None = None
        """Indices of Newton bodies that are used in the MuJoCo model for the first world as a basis for replicating the nworlds worlds in MuJoCo Warp."""

        self._viewer = None
        """Instance of the MuJoCo viewer for debugging."""

        disableflags = 0
        if disable_contacts:
            disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        if mjw_model is not None and mjw_data is not None:
            self.mjw_model = mjw_model
            self.mjw_data = mjw_data
            self.use_mujoco_cpu = False
        else:
            self.use_mujoco_cpu = use_mujoco_cpu
            if separate_worlds is None:
                separate_worlds = not use_mujoco_cpu
            with wp.ScopedTimer("convert_model_to_mujoco", active=False):
                self.convert_to_mjc(
                    model,
                    disableflags=disableflags,
                    disable_contacts=disable_contacts,
                    separate_worlds=separate_worlds,
                    njmax=njmax,
                    ncon_per_world=ncon_per_world,
                    iterations=iterations,
                    ls_iterations=ls_iterations,
                    cone=cone,
                    impratio=impratio,
                    solver=solver,
                    integrator=integrator,
                    default_actuator_gear=default_actuator_gear,
                    actuator_gears=actuator_gears,
                    target_filename=save_to_mjcf,
                    ls_parallel=ls_parallel,
                    tolerance=tolerance,
                    ls_tolerance=ls_tolerance,
                )
        self.update_data_interval = update_data_interval
        self._step = 0

        if self.mjw_model is not None:
            self.mjw_model.opt.run_collision_detection = use_mujoco_contacts

    @override
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):
        if self.use_mujoco_cpu:
            self.apply_mjc_control(self.model, state_in, control, self.mj_data)
            if self.update_data_interval > 0 and self._step % self.update_data_interval == 0:
                # XXX updating the mujoco state at every step may introduce numerical instability
                self.update_mjc_data(self.mj_data, self.model, state_in)
            self.mj_model.opt.timestep = dt
            self._mujoco.mj_step(self.mj_model, self.mj_data)
            self.update_newton_state(self.model, state_out, self.mj_data)
        else:
            self.apply_mjc_control(self.model, state_in, control, self.mjw_data)
            if self.update_data_interval > 0 and self._step % self.update_data_interval == 0:
                self.update_mjc_data(self.mjw_data, self.model, state_in)
            self.mjw_model.opt.timestep.fill_(dt)
            with wp.ScopedDevice(self.model.device):
                if self.mjw_model.opt.run_collision_detection:
                    self._mujoco_warp.step(self.mjw_model, self.mjw_data)
                else:
                    self.convert_contacts_to_mjwarp(self.model, state_in, contacts)
                    self._mujoco_warp.step(self.mjw_model, self.mjw_data)

            self.update_newton_state(self.model, state_out, self.mjw_data)
        self._step += 1
        return state_out

    def convert_contacts_to_mjwarp(self, model: Model, state_in: State, contacts: Contacts):
        bodies_per_world = self.model.body_count // self.model.num_worlds
        wp.launch(
            convert_newton_contacts_to_mjwarp_kernel,
            dim=(contacts.rigid_contact_max,),
            inputs=[
                state_in.body_q,
                model.shape_body,
                self.mjw_model.geom_condim,
                self.mjw_model.geom_priority,
                self.mjw_model.geom_solmix,
                self.mjw_model.geom_solref,
                self.mjw_model.geom_solimp,
                self.mjw_model.geom_friction,
                self.mjw_model.geom_margin,
                self.mjw_model.geom_gap,
                # Newton contacts
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_thickness0,
                contacts.rigid_contact_thickness1,
                bodies_per_world,
                self.to_mjc_geom_index,
                # Mujoco warp contacts
                self.mjw_data.nacon,
                self.mjw_data.contact.dist,
                self.mjw_data.contact.pos,
                self.mjw_data.contact.frame,
                self.mjw_data.contact.includemargin,
                self.mjw_data.contact.friction,
                self.mjw_data.contact.solref,
                self.mjw_data.contact.solreffriction,
                self.mjw_data.contact.solimp,
                self.mjw_data.contact.dim,
                self.mjw_data.contact.geom,
                self.mjw_data.contact.worldid,
                # Data to clear
                self.mjw_data.nworld,
                self.mjw_data.ncollision,
            ],
        )

    @override
    def notify_model_changed(self, flags: int):
        if flags & SolverNotifyFlags.BODY_INERTIAL_PROPERTIES:
            self.update_model_inertial_properties()
        if flags & SolverNotifyFlags.JOINT_PROPERTIES:
            self.update_joint_properties()
        if flags & SolverNotifyFlags.JOINT_DOF_PROPERTIES:
            self.update_joint_dof_properties()
        if flags & SolverNotifyFlags.SHAPE_PROPERTIES:
            self.update_geom_properties()
        if flags & SolverNotifyFlags.MODEL_PROPERTIES:
            self.update_model_properties()

    @staticmethod
    def _data_is_mjwarp(data):
        # Check if the data is a mujoco_warp Data object
        return hasattr(data, "nworld")

    def apply_mjc_control(self, model: Model, state: State, control: Control | None, mj_data: MjWarpData | MjData):
        if control is None or control.joint_f is None:
            if state.body_f is None:
                return
        is_mjwarp = SolverMuJoCo._data_is_mjwarp(mj_data)
        if is_mjwarp:
            ctrl = mj_data.ctrl
            qfrc = mj_data.qfrc_applied
            xfrc = mj_data.xfrc_applied
            nworld = mj_data.nworld
        else:
            ctrl = wp.zeros((1, len(mj_data.ctrl)), dtype=wp.float32, device=model.device)
            qfrc = wp.zeros((1, len(mj_data.qfrc_applied)), dtype=wp.float32, device=model.device)
            xfrc = wp.zeros((1, len(mj_data.xfrc_applied)), dtype=wp.spatial_vector, device=model.device)
            nworld = 1
        axes_per_world = model.joint_dof_count // nworld
        joints_per_world = model.joint_count // nworld
        bodies_per_world = model.body_count // nworld
        if control is not None:
            wp.launch(
                apply_mjc_control_kernel,
                dim=(nworld, axes_per_world),
                inputs=[
                    control.joint_target,
                    model.joint_dof_mode,
                    self.mjc_axis_to_actuator,
                    axes_per_world,
                ],
                outputs=[
                    ctrl,
                ],
                device=model.device,
            )
            wp.launch(
                apply_mjc_qfrc_kernel,
                dim=(nworld, joints_per_world),
                inputs=[
                    state.body_q,
                    control.joint_f,
                    model.joint_type,
                    model.body_com,
                    model.joint_child,
                    model.joint_q_start,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    joints_per_world,
                    bodies_per_world,
                ],
                outputs=[
                    qfrc,
                ],
                device=model.device,
            )

        if state.body_f is not None:
            wp.launch(
                apply_mjc_body_f_kernel,
                dim=(nworld, bodies_per_world),
                inputs=[
                    model.up_axis,
                    state.body_q,
                    state.body_f,
                    self.to_mjc_body_index,
                    bodies_per_world,
                ],
                outputs=[
                    xfrc,
                ],
                device=model.device,
            )
        if not is_mjwarp:
            mj_data.xfrc_applied = xfrc.numpy()
            mj_data.ctrl[:] = ctrl.numpy().flatten()
            mj_data.qfrc_applied[:] = qfrc.numpy()

    def update_mjc_data(self, mj_data: MjWarpData | MjData, model: Model, state: State | None = None):
        is_mjwarp = SolverMuJoCo._data_is_mjwarp(mj_data)
        if is_mjwarp:
            # we have an MjWarp Data object
            qpos = mj_data.qpos
            qvel = mj_data.qvel
            nworld = mj_data.nworld
        else:
            # we have an MjData object from Mujoco
            qpos = wp.empty((1, model.joint_coord_count), dtype=wp.float32, device=model.device)
            qvel = wp.empty((1, model.joint_dof_count), dtype=wp.float32, device=model.device)
            nworld = 1
        if state is None:
            joint_q = model.joint_q
            joint_qd = model.joint_qd
        else:
            joint_q = state.joint_q
            joint_qd = state.joint_qd
        joints_per_world = model.joint_count // nworld
        wp.launch(
            convert_warp_coords_to_mj_kernel,
            dim=(nworld, joints_per_world),
            inputs=[
                joint_q,
                joint_qd,
                joints_per_world,
                model.up_axis,
                model.joint_type,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_dof_dim,
            ],
            outputs=[qpos, qvel],
            device=model.device,
        )
        if not is_mjwarp:
            mj_data.qpos[:] = qpos.numpy().flatten()[: len(mj_data.qpos)]
            mj_data.qvel[:] = qvel.numpy().flatten()[: len(mj_data.qvel)]

    def update_newton_state(
        self,
        model: Model,
        state: State,
        mj_data: MjWarpData | MjData,
        eval_fk: bool = True,
    ):
        is_mjwarp = SolverMuJoCo._data_is_mjwarp(mj_data)
        if is_mjwarp:
            # we have an MjWarp Data object
            qpos = mj_data.qpos
            qvel = mj_data.qvel
            nworld = mj_data.nworld

            xpos = mj_data.xpos
            xquat = mj_data.xquat
        else:
            # we have an MjData object from Mujoco
            qpos = wp.array([mj_data.qpos], dtype=wp.float32, device=model.device)
            qvel = wp.array([mj_data.qvel], dtype=wp.float32, device=model.device)
            nworld = 1

            xpos = wp.array([mj_data.xpos], dtype=wp.vec3, device=model.device)
            xquat = wp.array([mj_data.xquat], dtype=wp.quat, device=model.device)
        joints_per_world = model.joint_count // nworld
        wp.launch(
            convert_mj_coords_to_warp_kernel,
            dim=(nworld, joints_per_world),
            inputs=[
                qpos,
                qvel,
                joints_per_world,
                int(model.up_axis),
                model.joint_type,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_dof_dim,
            ],
            outputs=[state.joint_q, state.joint_qd],
            device=model.device,
        )

        if eval_fk:
            # custom forward kinematics for handling multi-dof joints
            wp.launch(
                kernel=eval_articulation_fk,
                dim=model.articulation_count,
                inputs=[
                    model.articulation_start,
                    state.joint_q,
                    state.joint_qd,
                    model.joint_q_start,
                    model.joint_qd_start,
                    model.joint_type,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    model.joint_axis,
                    model.joint_dof_dim,
                    model.body_com,
                ],
                outputs=[
                    state.body_q,
                    state.body_qd,
                ],
                device=model.device,
            )
        else:
            bodies_per_world = model.body_count // model.num_worlds
            wp.launch(
                convert_body_xforms_to_warp_kernel,
                dim=(nworld, bodies_per_world),
                inputs=[
                    xpos,
                    xquat,
                    self.to_mjc_body_index,
                    bodies_per_world,
                ],
                outputs=[state.body_q],
                device=model.device,
            )

    @staticmethod
    def find_body_collision_filter_pairs(
        model: Model,
        selected_bodies: nparray,
        colliding_shapes: nparray,
    ):
        """For shape collision filter pairs, find body collision filter pairs that are contained within."""

        body_exclude_pairs = []
        shape_set = set(colliding_shapes)

        body_shapes = {}
        for body in selected_bodies:
            shapes = model.body_shapes[body]
            shapes = [s for s in shapes if s in shape_set]
            body_shapes[body] = shapes

        bodies_a, bodies_b = np.triu_indices(len(selected_bodies), k=1)
        for body_a, body_b in zip(bodies_a, bodies_b, strict=True):
            b1, b2 = selected_bodies[body_a], selected_bodies[body_b]
            shapes_1 = body_shapes[b1]
            shapes_2 = body_shapes[b2]
            excluded = True
            for shape_1 in shapes_1:
                for shape_2 in shapes_2:
                    if shape_1 > shape_2:
                        s1, s2 = shape_2, shape_1
                    else:
                        s1, s2 = shape_1, shape_2
                    if (s1, s2) not in model.shape_collision_filter_pairs:
                        excluded = False
                        break
            if excluded:
                body_exclude_pairs.append((b1, b2))
        return body_exclude_pairs

    @staticmethod
    def color_collision_shapes(
        model: Model, selected_shapes: nparray, visualize_graph: bool = False, shape_keys: list[str] | None = None
    ) -> nparray:
        """
        Find a graph coloring of the collision filter pairs in the model.
        Shapes within the same color cannot collide with each other.
        Shapes can only collide with shapes of different colors.

        Args:
            model (Model): The model to color the collision shapes of.
            selected_shapes (nparray): The indices of the collision shapes to color.
            visualize_graph (bool): Whether to visualize the graph coloring.
            shape_keys (list[str]): The keys of the shapes, only used for visualization.

        Returns:
            nparray: An integer array of shape (num_shapes,), where each element is the color of the corresponding shape.
        """
        # we first create a mapping from selected shape to local color shape index
        # to reduce the number of nodes in the graph to only the number of selected shapes
        # without any gaps between the indices (otherwise we have to allocate max(selected_shapes) + 1 nodes)
        to_color_shape_index = {}
        for i, shape in enumerate(selected_shapes):
            to_color_shape_index[shape] = i
        # find graph coloring of collision filter pairs
        num_shapes = len(selected_shapes)
        shape_a, shape_b = np.triu_indices(num_shapes, k=1)
        cgroup = [model.shape_collision_group[i] for i in selected_shapes]
        # edges representing colliding shape pairs
        graph_edges = [
            (i, j)
            for i, j in zip(shape_a, shape_b, strict=True)
            if (
                (selected_shapes[i], selected_shapes[j]) not in model.shape_collision_filter_pairs
                and (cgroup[i] == cgroup[j] or cgroup[i] == -1 or cgroup[j] == -1)
            )
        ]
        shape_color = np.zeros(model.shape_count, dtype=np.int32)
        if len(graph_edges) > 0:
            color_groups = color_graph(
                num_nodes=num_shapes,
                graph_edge_indices=wp.array(graph_edges, dtype=wp.int32),
                balance_colors=False,
            )
            num_colors = 0
            for group in color_groups:
                num_colors += 1
                shape_color[selected_shapes[group]] = num_colors
            if visualize_graph:
                plot_graph(
                    vertices=np.arange(num_shapes),
                    edges=graph_edges,
                    node_labels=[shape_keys[i] for i in selected_shapes] if shape_keys is not None else None,
                    node_colors=[shape_color[i] for i in selected_shapes],
                )

        return shape_color

    @override
    def update_contacts(self, contacts: Contacts) -> None:
        # TODO: ensure that class invariants are preserved
        # TODO: fill actual contact arrays instead of creating new ones
        mj_data = self.mjw_data
        naconmax = mj_data.naconmax
        mj_contact = mj_data.contact

        contacts.rigid_contact_max = naconmax
        contacts.rigid_contact_count = mj_data.nacon
        contacts.position = mj_contact.pos
        contacts.separation = mj_contact.dist

        if not hasattr(contacts, "pair"):
            contacts.pair = wp.empty(naconmax, dtype=wp.vec2i, device=self.model.device)

        if not hasattr(contacts, "normal"):
            contacts.normal = wp.empty(naconmax, dtype=wp.vec3f, device=self.model.device)

        if not hasattr(contacts, "force"):
            contacts.force = wp.empty(naconmax, dtype=wp.float32, device=self.model.device)

        wp.launch(
            convert_mjw_contact_to_warp_kernel,
            dim=mj_data.naconmax,
            inputs=[
                self.to_newton_shape_index,
                self.mjw_model.opt.cone == int(self._mujoco.mjtCone.mjCONE_PYRAMIDAL),
                mj_data.nacon,
                mj_contact.frame,
                mj_contact.dim,
                mj_contact.geom,
                mj_contact.efc_address,
                mj_contact.worldid,
                mj_data.efc.force,
            ],
            outputs=[
                contacts.pair,
                contacts.normal,
                contacts.force,
            ],
            device=self.model.device,
        )
        contacts.n_contacts = mj_data.nacon

    def convert_to_mjc(
        self,
        model: Model,
        state: State | None = None,
        *,
        separate_worlds: bool = True,
        iterations: int = 20,
        ls_iterations: int = 10,
        njmax: int | None = None,  # number of constraints per world
        ncon_per_world: int | None = None,
        solver: int | str = "cg",
        integrator: int | str = "implicitfast",
        disableflags: int = 0,
        disable_contacts: bool = False,
        impratio: float = 1.0,
        tolerance: float | None = None,
        ls_tolerance: float | None = None,
        cone: int | str = "pyramidal",
        # maximum absolute joint limit value after which the joint is considered not limited
        joint_limit_threshold: float = 1e3,
        geom_solref: tuple[float, float] | None = None,
        geom_solimp: tuple[float, float, float, float, float] = (0.9, 0.95, 0.001, 0.5, 2.0),
        geom_friction: tuple[float, float, float] | None = None,
        geom_condim: int = 3,
        target_filename: str | None = None,
        default_actuator_args: dict | None = None,
        default_actuator_gear: float | None = None,
        actuator_gears: dict[str, float] | None = None,
        actuated_axes: list[int] | None = None,
        skip_visual_only_geoms: bool = True,
        add_axes: bool = False,
        mesh_maxhullvert: int = MESH_MAXHULLVERT,
        ls_parallel: bool = False,
    ) -> tuple[MjWarpModel, MjWarpData, MjModel, MjData]:
        """
        Convert a Newton model and state to MuJoCo (Warp) model and data.

        Args:
            Model (newton.Model): The Newton model to convert.
            State (newton.State): The Newton state to convert.

        Returns:
            tuple[MjWarpModel, MjWarpData, MjModel, MjData]: A tuple containing the model and data objects for ``mujoco_warp`` and MuJoCo.
        """

        if not model.joint_count:
            raise ValueError("The model must have at least one joint to be able to convert it to MuJoCo.")

        # Validate that separate_worlds=False is only used with single world
        if not separate_worlds and model.num_worlds > 1:
            raise ValueError(
                f"separate_worlds=False is only supported for single-world models. "
                f"Got num_worlds={model.num_worlds}. Use separate_worlds=True for multi-world models."
            )

        mujoco, mujoco_warp = self.import_mujoco()

        actuator_args = {
            # "ctrllimited": True,
            # "ctrlrange": (-1.0, 1.0),
            "gear": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "trntype": mujoco.mjtTrn.mjTRN_JOINT,
            # motor actuation properties (already the default settings in Mujoco)
            "gainprm": [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "biasprm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "dyntype": mujoco.mjtDyn.mjDYN_NONE,
            "gaintype": mujoco.mjtGain.mjGAIN_FIXED,
            "biastype": mujoco.mjtBias.mjBIAS_AFFINE,
        }
        if default_actuator_args is not None:
            actuator_args.update(default_actuator_args)
        if default_actuator_gear is not None:
            actuator_args["gear"][0] = default_actuator_gear
        if actuator_gears is None:
            actuator_gears = {}

        def _resolve_mj_opt(val, opts: dict[str, int], kind: str):
            if isinstance(val, str):
                key = val.strip().lower()
                try:
                    return opts[key]
                except KeyError as e:
                    options = "', '".join(sorted(opts))
                    raise ValueError(f"Unknown {kind} '{val}'. Valid options: '{options}'.") from e
            return val

        solver = _resolve_mj_opt(
            solver, {"cg": mujoco.mjtSolver.mjSOL_CG, "newton": mujoco.mjtSolver.mjSOL_NEWTON}, "solver"
        )
        integrator = _resolve_mj_opt(
            integrator,
            {
                "euler": mujoco.mjtIntegrator.mjINT_EULER,
                "rk4": mujoco.mjtIntegrator.mjINT_RK4,
                "implicit": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
                "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
            },
            "integrator",
        )
        cone = _resolve_mj_opt(
            cone, {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL, "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC}, "cone"
        )

        def quat_to_mjc(q):
            # convert from xyzw to wxyz
            return [q[3], q[0], q[1], q[2]]

        def quat_from_mjc(q):
            # convert from wxyz to xyzw
            return [q[1], q[2], q[3], q[0]]

        def fill_arr_from_dict(arr: nparray, d: dict[int, Any]):
            # fast way to fill an array from a dictionary
            # keys and values can also be tuples of integers
            keys = np.array(list(d.keys()), dtype=int)
            vals = np.array(list(d.values()), dtype=int)
            if keys.ndim == 1:
                arr[keys] = vals
            else:
                arr[tuple(keys.T)] = vals

        spec = mujoco.MjSpec()
        spec.option.disableflags = disableflags
        spec.option.gravity = np.array([*model.gravity.numpy()[0]])
        spec.option.solver = solver
        spec.option.integrator = integrator
        spec.option.iterations = iterations
        spec.option.ls_iterations = ls_iterations
        spec.option.cone = cone
        spec.option.impratio = impratio
        spec.option.tolerance = tolerance
        spec.option.ls_tolerance = ls_tolerance
        spec.option.jacobian = mujoco.mjtJacobian.mjJAC_AUTO

        defaults = spec.default
        if callable(defaults):
            defaults = defaults()
        defaults.geom.condim = geom_condim
        # Use provided or default contact stiffness time constant
        if geom_solref is None:
            geom_solref = (self.contact_stiffness_time_const, 1.0)
        defaults.geom.solref = geom_solref
        defaults.geom.solimp = geom_solimp
        # Use model's friction parameters if geom_friction is not provided
        if geom_friction is None:
            geom_friction = (1.0, model.rigid_contact_torsional_friction, model.rigid_contact_rolling_friction)
        defaults.geom.friction = geom_friction
        # defaults.geom.contype = 0
        spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_AUTO

        if add_axes:
            # add axes for debug visualization in MuJoCo viewer when loading the generated XML
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                name="axis_x",
                fromto=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                rgba=[1.0, 0.0, 0.0, 1.0],
                size=[0.01, 0.01, 0.01],
                contype=0,
                conaffinity=0,
            )
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                name="axis_y",
                fromto=[0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                rgba=[0.0, 1.0, 0.0, 1.0],
                size=[0.01, 0.01, 0.01],
                contype=0,
                conaffinity=0,
            )
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                name="axis_z",
                fromto=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                rgba=[0.0, 0.0, 1.0, 1.0],
                size=[0.01, 0.01, 0.01],
                contype=0,
                conaffinity=0,
            )

        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_parent_xform = model.joint_X_p.numpy()
        joint_child_xform = model.joint_X_c.numpy()
        joint_limit_lower = model.joint_limit_lower.numpy()
        joint_limit_upper = model.joint_limit_upper.numpy()
        joint_type = model.joint_type.numpy()
        joint_axis = model.joint_axis.numpy()
        joint_dof_dim = model.joint_dof_dim.numpy()
        joint_dof_mode = model.joint_dof_mode.numpy()
        joint_target_kd = model.joint_target_kd.numpy()
        joint_target_ke = model.joint_target_ke.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        joint_armature = model.joint_armature.numpy()
        joint_effort_limit = model.joint_effort_limit.numpy()
        # MoJoCo doesn't have velocity limit
        # joint_velocity_limit = model.joint_velocity_limit.numpy()
        joint_friction = model.joint_friction.numpy()
        joint_world = model.joint_world.numpy()
        body_mass = model.body_mass.numpy()
        body_inertia = model.body_inertia.numpy()
        body_com = model.body_com.numpy()
        body_world = model.body_world.numpy()
        shape_transform = model.shape_transform.numpy()
        shape_type = model.shape_type.numpy()
        shape_size = model.shape_scale.numpy()
        shape_flags = model.shape_flags.numpy()
        shape_world = model.shape_world.numpy()
        shape_mu = model.shape_material_mu.numpy()

        eq_constraint_type = model.equality_constraint_type.numpy()
        eq_constraint_body1 = model.equality_constraint_body1.numpy()
        eq_constraint_body2 = model.equality_constraint_body2.numpy()
        eq_constraint_anchor = model.equality_constraint_anchor.numpy()
        eq_constraint_torquescale = model.equality_constraint_torquescale.numpy()
        eq_constraint_relpose = model.equality_constraint_relpose.numpy()
        eq_constraint_joint1 = model.equality_constraint_joint1.numpy()
        eq_constraint_joint2 = model.equality_constraint_joint2.numpy()
        eq_constraint_polycoef = model.equality_constraint_polycoef.numpy()
        eq_constraint_enabled = model.equality_constraint_enabled.numpy()

        INT32_MAX = np.iinfo(np.int32).max
        collision_mask_everything = INT32_MAX

        # mapping from joint axis to actuator index
        axis_to_actuator = np.zeros((model.joint_dof_count,), dtype=np.int32) - 1
        actuator_count = 0

        # supported non-fixed joint types in MuJoCo (fixed joints are handled by nesting bodies)
        supported_joint_types = {
            JointType.FREE,
            JointType.BALL,
            JointType.PRISMATIC,
            JointType.REVOLUTE,
            JointType.D6,
        }

        geom_type_mapping = {
            GeoType.SPHERE: mujoco.mjtGeom.mjGEOM_SPHERE,
            GeoType.PLANE: mujoco.mjtGeom.mjGEOM_PLANE,
            GeoType.CAPSULE: mujoco.mjtGeom.mjGEOM_CAPSULE,
            GeoType.CYLINDER: mujoco.mjtGeom.mjGEOM_CYLINDER,
            GeoType.BOX: mujoco.mjtGeom.mjGEOM_BOX,
            GeoType.MESH: mujoco.mjtGeom.mjGEOM_MESH,
            GeoType.CONVEX_MESH: mujoco.mjtGeom.mjGEOM_MESH,
        }

        mj_bodies = [spec.worldbody]
        # mapping from Newton body id to MuJoCo body id
        body_mapping = {-1: 0}
        # mapping from Newton shape id to MuJoCo geom name
        shape_mapping = {}

        # ensure unique names
        body_name_counts = {}
        joint_names = {}

        # number of shapes which are replicated per world (excludes singular static shapes from a negative group)
        shape_range_len = 0

        if separate_worlds:
            # determine which shapes, bodies and joints belong to the first world
            # based on the shape world: we pick objects from the first world and global shapes
            non_negatives = shape_world[shape_world >= 0]
            if len(non_negatives) > 0:
                first_group = np.min(non_negatives)
                shape_range_len = len(np.where(shape_world == first_group)[0])
            else:
                first_group = -1
                shape_range_len = model.shape_count
            selected_shapes = np.where((shape_world == first_group) | (shape_world < 0))[0]
            selected_bodies = np.where((body_world == first_group) | (body_world < 0))[0]
            selected_joints = np.where((joint_world == first_group) | (joint_world < 0))[0]
        else:
            # if we are not separating environments to worlds, we use all shapes, bodies, joints
            first_group = 0
            shape_range_len = model.shape_count

            # if we are not separating worlds, we use all shapes, bodies, joints
            selected_shapes = np.arange(model.shape_count, dtype=np.int32)
            selected_bodies = np.arange(model.body_count, dtype=np.int32)
            selected_joints = np.arange(model.joint_count, dtype=np.int32)

        # sort joints topologically depth-first since this is the order that will also be used
        # for placing bodies in the MuJoCo model
        joints_simple = list(zip(joint_parent[selected_joints], joint_child[selected_joints], strict=False))
        joint_order = topological_sort(joints_simple, use_dfs=True)
        if any(joint_order[i] != i for i in range(len(joints_simple))):
            warnings.warn(
                "Joint order is not in depth-first topological order while converting Newton model to MuJoCo, this may lead to diverging kinematics between MuJoCo and Newton.",
                stacklevel=2,
            )

        # find graph coloring of collision filter pairs
        # filter out shapes that are not colliding with anything
        colliding_shapes = selected_shapes[shape_flags[selected_shapes] & ShapeFlags.COLLIDE_SHAPES != 0]

        # number of shapes we are instantiating in MuJoCo (which will be replicated for the number of envs)
        colliding_shapes_per_world = len(colliding_shapes)

        # filter out non-colliding bodies using excludes
        body_filters = self.find_body_collision_filter_pairs(
            model,
            selected_bodies,
            colliding_shapes,
        )

        shape_color = self.color_collision_shapes(
            model, colliding_shapes, visualize_graph=False, shape_keys=model.shape_key
        )

        # store selected shapes, bodies, joints for later use in update_geom_properties
        self.selected_shapes = wp.array(selected_shapes, dtype=wp.int32, device=model.device)
        self.selected_joints = wp.array(selected_joints, dtype=wp.int32, device=model.device)
        self.selected_bodies = wp.array(selected_bodies, dtype=wp.int32, device=model.device)
        selected_shapes_set = set(selected_shapes)

        def add_geoms(newton_body_id: int):
            body = mj_bodies[body_mapping[newton_body_id]]
            shapes = model.body_shapes.get(newton_body_id)
            if not shapes:
                return
            for shape in shapes:
                if shape not in selected_shapes_set:
                    # skip shapes that are not selected for this world
                    continue
                if skip_visual_only_geoms and not (shape_flags[shape] & ShapeFlags.COLLIDE_SHAPES):
                    continue
                stype = shape_type[shape]
                name = f"{model.shape_key[shape]}_{shape}"
                if stype == GeoType.PLANE and newton_body_id != -1:
                    raise ValueError("Planes can only be attached to static bodies")
                geom_params = {
                    "type": geom_type_mapping[stype],
                    "name": name,
                }
                tf = wp.transform(*shape_transform[shape])
                if stype == GeoType.MESH or stype == GeoType.CONVEX_MESH:
                    mesh_src = model.shape_source[shape]
                    # use mesh-specific maxhullvert or fall back to the default
                    maxhullvert = getattr(mesh_src, "maxhullvert", mesh_maxhullvert)
                    # apply scaling
                    size = shape_size[shape]
                    vertices = mesh_src.vertices * size
                    spec.add_mesh(
                        name=name,
                        uservert=vertices.flatten(),
                        userface=mesh_src.indices.flatten(),
                        maxhullvert=maxhullvert,
                    )
                    geom_params["meshname"] = name
                geom_params["pos"] = tf.p
                geom_params["quat"] = quat_to_mjc(tf.q)
                size = shape_size[shape]
                if np.any(size > 0.0):
                    # duplicate nonzero entries at places where size is 0
                    nonzero = size[size > 0.0][0]
                    size[size == 0.0] = nonzero
                    geom_params["size"] = size
                else:
                    assert stype == GeoType.PLANE, "Only plane shapes are allowed to have a size of zero"
                    # planes are always infinite for collision purposes in mujoco
                    geom_params["size"] = [5.0, 5.0, 5.0]
                    # make ground plane blue in the MuJoCo viewer (only used for debugging)
                    geom_params["rgba"] = [0.0, 0.3, 0.6, 1.0]

                # encode collision filtering information
                if not (shape_flags[shape] & ShapeFlags.COLLIDE_SHAPES):
                    # this shape is not colliding with anything
                    geom_params["contype"] = 0
                    geom_params["conaffinity"] = 0
                else:
                    color = shape_color[shape]
                    if color < 32:
                        contype = 1 << color
                        geom_params["contype"] = contype
                        # collide with anything except shapes from the same color
                        geom_params["conaffinity"] = collision_mask_everything & ~contype

                # set friction from Newton shape materials using model's friction parameters
                mu = shape_mu[shape]
                geom_params["friction"] = [
                    mu,
                    model.rigid_contact_torsional_friction * mu,
                    model.rigid_contact_rolling_friction * mu,
                ]

                body.add_geom(**geom_params)
                # store the geom name instead of assuming index
                shape_mapping[shape] = name

        # add static geoms attached to the worldbody
        add_geoms(-1)

        # maps from Newton joint index to the start index of its joint axes in MuJoCo
        # (only defined for the joints of the first world)
        joint_mjc_dof_start = np.full(model.joint_count, -1, dtype=np.int32)

        # add joints, bodies and geoms
        for ji in joint_order:
            parent, child = joints_simple[ji]
            if child in body_mapping:
                raise ValueError(f"Body {child} already exists in the mapping")

            # add body
            body_mapping[child] = len(mj_bodies)

            # use the correct global joint index
            j = selected_joints[ji]

            # this assumes that the joint position is 0
            tf = wp.transform(*joint_parent_xform[j])
            tf = tf * wp.transform_inverse(wp.transform(*joint_child_xform[j]))

            jc_xform = wp.transform(*joint_child_xform[j])
            joint_pos = jc_xform.p
            joint_rot = jc_xform.q

            # ensure unique body name
            name = model.body_key[child]
            if name not in body_name_counts:
                body_name_counts[name] = 1
            else:
                while name in body_name_counts:
                    body_name_counts[name] += 1
                    name = f"{name}_{body_name_counts[name]}"

            inertia = body_inertia[child]
            body = mj_bodies[body_mapping[parent]].add_body(
                name=name,
                pos=tf.p,
                quat=quat_to_mjc(tf.q),
                mass=body_mass[child],
                ipos=body_com[child, :],
                fullinertia=[inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]],
                explicitinertial=True,
            )
            mj_bodies.append(body)

            # add joint
            j_type = joint_type[j]
            qd_start = joint_qd_start[j]
            name = model.joint_key[j]
            if name not in joint_names:
                joint_names[name] = 1
            else:
                while name in joint_names:
                    joint_names[name] += 1
                    name = f"{name}_{joint_names[name]}"

            joint_mjc_dof_start[ji] = len(spec.joints)

            if j_type == JointType.FREE:
                body.add_joint(
                    name=name,
                    type=mujoco.mjtJoint.mjJNT_FREE,
                    damping=0.0,
                    limited=False,
                )
            elif j_type in supported_joint_types:
                lin_axis_count, ang_axis_count = joint_dof_dim[j]
                # linear dofs
                for i in range(lin_axis_count):
                    ai = qd_start + i

                    axis = wp.quat_rotate(joint_rot, wp.vec3(*joint_axis[ai]))

                    joint_params = {
                        "armature": joint_armature[qd_start + i],
                        "pos": joint_pos,
                    }
                    # Set friction
                    joint_params["frictionloss"] = joint_friction[ai]
                    lower, upper = joint_limit_lower[ai], joint_limit_upper[ai]
                    if lower == upper or (abs(lower) > joint_limit_threshold and abs(upper) > joint_limit_threshold):
                        joint_params["limited"] = False
                    else:
                        joint_params["limited"] = True
                        joint_params["range"] = (lower, upper)
                        # Add global solver parameters if provided
                        if self.joint_solref_limit is not None:
                            joint_params["solref_limit"] = self.joint_solref_limit
                        if self.joint_solimp_limit is not None:
                            joint_params["solimp_limit"] = self.joint_solimp_limit
                    axname = name
                    if lin_axis_count > 1 or ang_axis_count > 1:
                        axname += "_lin"
                    if lin_axis_count > 1:
                        axname += str(i)
                    body.add_joint(
                        name=axname,
                        type=mujoco.mjtJoint.mjJNT_SLIDE,
                        axis=axis,
                        **joint_params,
                    )
                    if actuated_axes is None or ai in actuated_axes:
                        # add actuator for this axis
                        gear = actuator_gears.get(axname)
                        if gear is not None:
                            args = {}
                            args.update(actuator_args)
                            args["gear"] = [gear, 0.0, 0.0, 0.0, 0.0, 0.0]
                        else:
                            args = actuator_args

                        if joint_dof_mode[ai] == JointMode.TARGET_POSITION:
                            kp = joint_target_ke[ai]
                            kv = joint_target_kd[ai]
                            args["biasprm"] = [0.0, -kp, -kv, 0, 0, 0, 0, 0, 0, 0]
                            args["gainprm"] = [kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        elif joint_dof_mode[ai] == JointMode.TARGET_VELOCITY:
                            kv = joint_target_kd[ai]
                            args["biasprm"] = [0.0, 0.0, -kv, 0, 0, 0, 0, 0, 0, 0]
                            args["gainprm"] = [kv, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            # no target position or velocity, just use the default gain
                            args["biasprm"] = [0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0]
                            args["gainprm"] = [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                        # Add effort limits from Newton model
                        effort_limit = joint_effort_limit[ai]
                        args["forcerange"] = [-effort_limit, effort_limit]

                        spec.add_actuator(target=axname, **args)
                        axis_to_actuator[ai] = actuator_count
                        actuator_count += 1

                # angular dofs
                for i in range(lin_axis_count, lin_axis_count + ang_axis_count):
                    ai = qd_start + i

                    axis = wp.quat_rotate(joint_rot, wp.vec3(*joint_axis[ai]))

                    joint_params = {
                        "armature": joint_armature[qd_start + i],
                        "pos": joint_pos,
                    }
                    # Set friction
                    joint_params["frictionloss"] = joint_friction[ai]
                    lower, upper = joint_limit_lower[ai], joint_limit_upper[ai]
                    if lower == upper or (abs(lower) > joint_limit_threshold and abs(upper) > joint_limit_threshold):
                        joint_params["limited"] = False
                    else:
                        joint_params["limited"] = True
                        joint_params["range"] = (np.rad2deg(lower), np.rad2deg(upper))
                        # Add global solver parameters if provided
                        if self.joint_solref_limit is not None:
                            joint_params["solref_limit"] = self.joint_solref_limit
                        if self.joint_solimp_limit is not None:
                            joint_params["solimp_limit"] = self.joint_solimp_limit
                    axname = name
                    if lin_axis_count > 1 or ang_axis_count > 1:
                        axname += "_ang"
                    if ang_axis_count > 1:
                        axname += str(i - lin_axis_count)
                    body.add_joint(
                        name=axname,
                        type=mujoco.mjtJoint.mjJNT_HINGE,
                        axis=axis,
                        **joint_params,
                    )
                    if actuated_axes is None or ai in actuated_axes:
                        # add actuator for this axis
                        gear = actuator_gears.get(axname)
                        if gear is not None:
                            args = {}
                            args.update(actuator_args)
                            args["gear"] = [gear, 0.0, 0.0, 0.0, 0.0, 0.0]
                        else:
                            args = actuator_args

                        if joint_dof_mode[ai] == JointMode.TARGET_POSITION:
                            kp = joint_target_ke[ai]
                            kv = joint_target_kd[ai]
                            args["biasprm"] = [0.0, -kp, -kv, 0, 0, 0, 0, 0, 0, 0]
                            args["gainprm"] = [kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        elif joint_dof_mode[ai] == JointMode.TARGET_VELOCITY:
                            kv = joint_target_kd[ai]
                            args["biasprm"] = [0.0, 0.0, -kv, 0, 0, 0, 0, 0, 0, 0]
                            args["gainprm"] = [kv, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            # no target position or velocity, just use the default gain
                            args["biasprm"] = [0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0]
                            args["gainprm"] = [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                        # Add effort limits from Newton model
                        effort_limit = joint_effort_limit[ai]
                        args["forcerange"] = [-effort_limit, effort_limit]

                        spec.add_actuator(target=axname, **args)
                        axis_to_actuator[ai] = actuator_count
                        actuator_count += 1

            elif j_type != JointType.FIXED:
                raise NotImplementedError(f"Joint type {j_type} is not supported yet")

            add_geoms(child)

        for i, constraint_type in enumerate(eq_constraint_type):
            if constraint_type == EqType.CONNECT:
                eq = spec.add_equality(objtype=mujoco.mjtObj.mjOBJ_BODY)
                eq.type = mujoco.mjtEq.mjEQ_CONNECT
                eq.active = eq_constraint_enabled[i]
                eq.name1 = model.body_key[eq_constraint_body1[i]]
                eq.name2 = model.body_key[eq_constraint_body2[i]]
                eq.data[0:3] = eq_constraint_anchor[i]

            elif constraint_type == EqType.JOINT:
                eq = spec.add_equality(objtype=mujoco.mjtObj.mjOBJ_JOINT)
                eq.type = mujoco.mjtEq.mjEQ_JOINT
                eq.active = eq_constraint_enabled[i]
                eq.name1 = model.joint_key[eq_constraint_joint1[i]]
                eq.name2 = model.joint_key[eq_constraint_joint2[i]]
                eq.data[0:5] = eq_constraint_polycoef[i]

            elif constraint_type == EqType.WELD:
                eq = spec.add_equality(objtype=mujoco.mjtObj.mjOBJ_BODY)
                eq.type = mujoco.mjtEq.mjEQ_WELD
                eq.active = eq_constraint_enabled[i]
                eq.name1 = model.body_key[eq_constraint_body1[i]]
                eq.name2 = model.body_key[eq_constraint_body2[i]]
                cns_relpose = wp.transform(*eq_constraint_relpose[i])
                eq.data[0:3] = eq_constraint_anchor[i]
                eq.data[3:6] = wp.transform_get_translation(cns_relpose)
                eq.data[6:10] = wp.transform_get_rotation(cns_relpose)
                eq.data[10] = eq_constraint_torquescale[i]

        assert len(spec.geoms) == colliding_shapes_per_world, (
            "The number of geoms in the MuJoCo model does not match the number of colliding shapes in the Newton model."
        )

        # add contact exclusions between bodies to ensure parent <> child collisions are ignored
        # even when one of the bodies is static
        for b1, b2 in body_filters:
            mb1, mb2 = body_mapping[b1], body_mapping[b2]
            spec.add_exclude(bodyname1=spec.bodies[mb1].name, bodyname2=spec.bodies[mb2].name)

        self.mj_model = spec.compile()
        self.mj_data = mujoco.MjData(self.mj_model)

        self.update_mjc_data(self.mj_data, model, state)

        # fill some MjWarp model fields that are outdated after update_mjc_data.
        # just setting qpos0 to d.qpos leads to weird behavior here, needs
        # to be investigated.

        mujoco.mj_forward(self.mj_model, self.mj_data)

        if target_filename:
            with open(target_filename, "w") as f:
                f.write(spec.to_xml())
                print(f"Saved mujoco model to {os.path.abspath(target_filename)}")

        # now that the model is compiled, get the actual geom indices and compute
        # shape transform corrections
        shape_to_geom_idx = {}
        geom_to_shape_idx = {}
        for shape, geom_name in shape_mapping.items():
            geom_idx = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_idx >= 0:
                shape_to_geom_idx[shape] = geom_idx
                geom_to_shape_idx[geom_idx] = shape

        with wp.ScopedDevice(model.device):
            # create the MuJoCo Warp model
            self.mjw_model = mujoco_warp.put_model(self.mj_model)

            # build the geom index mappings now that we have the actual indices
            # geom_to_shape_idx maps from MuJoCo geom index to absolute Newton shape index.
            # Convert non-static shapes to template-relative indices for the kernel.
            geom_to_shape_idx_np = np.full((self.mj_model.ngeom,), -1, dtype=np.int32)

            # Find the minimum shape index for the first non-static group to use as the base
            first_env_shapes = np.where(shape_world == first_group)[0]
            first_env_shape_base = int(np.min(first_env_shapes)) if len(first_env_shapes) > 0 else 0

            # Per-geom static mask (True if static, False otherwise)
            geom_is_static_np = np.zeros((self.mj_model.ngeom,), dtype=bool)

            for geom_idx, abs_shape_idx in geom_to_shape_idx.items():
                if shape_world[abs_shape_idx] < 0:
                    # Static shape - use absolute index and mark mask
                    geom_to_shape_idx_np[geom_idx] = abs_shape_idx
                    geom_is_static_np[geom_idx] = True
                else:
                    # Non-static shape - convert to template-relative offset from first env base
                    geom_to_shape_idx_np[geom_idx] = abs_shape_idx - first_env_shape_base

            geom_to_shape_idx_wp = wp.array(geom_to_shape_idx_np, dtype=wp.int32)
            geom_is_static_wp = wp.array(geom_is_static_np, dtype=bool)

            # use the actual number of geoms from the MuJoCo model
            self.to_newton_shape_index = wp.full((model.num_worlds, self.mj_model.ngeom), -1, dtype=wp.int32)

            # create mapping from Newton shape index to MuJoCo [world, geom index]
            self.to_mjc_geom_index = wp.full(model.shape_count, -1, dtype=wp.vec2i)

            # mapping from Newton shape id to a corrective transform
            # that maps from Newton's shape frame to MuJoCo's internal geom frame
            self.shape_incoming_xform = wp.full(model.shape_count, wp.transform_identity(), dtype=wp.transform)

            # mapping from Newton joint index to the start index of its joint axes in MuJoCo
            self.joint_mjc_dof_start = wp.array(joint_mjc_dof_start, dtype=wp.int32)

            if self.mjw_model.geom_pos.size:
                wp.launch(
                    update_incoming_shape_xform_kernel,
                    dim=(self.model.num_worlds, self.mj_model.ngeom),
                    inputs=[
                        geom_to_shape_idx_wp,
                        geom_is_static_wp,
                        self.model.shape_transform,
                        shape_range_len,
                        first_env_shape_base,
                        self.mjw_model.geom_pos[0],
                        self.mjw_model.geom_quat[0],
                    ],
                    outputs=[
                        self.to_mjc_geom_index,
                        self.to_newton_shape_index,
                        self.shape_incoming_xform,
                    ],
                )

            # mapping from Newton joint axis index to MJC actuator index
            self.mjc_axis_to_actuator = wp.array(axis_to_actuator, dtype=wp.int32)
            # mapping from MJC body index to Newton body index (skip world index -1)
            to_mjc_body_index = np.fromiter(body_mapping.keys(), dtype=int)[1:] + 1
            self.to_mjc_body_index = wp.array(to_mjc_body_index, dtype=wp.int32)

            # set mjwarp-only settings
            self.mjw_model.opt.ls_parallel = ls_parallel

            if separate_worlds:
                nworld = model.num_worlds
            else:
                nworld = 1

            # expand model fields that can be expanded:
            self.expand_model_fields(self.mjw_model, nworld)

            # so far we have only defined the first world,
            # now complete the data from the Newton model
            self.notify_model_changed(SolverNotifyFlags.ALL)

            # TODO find better heuristics to determine naconmax and njmax
            if disable_contacts:
                naconmax = 0
            else:
                if ncon_per_world is not None:
                    rigid_contact_max = nworld * ncon_per_world
                    if rigid_contact_max < self.mj_data.ncon * nworld:
                        warnings.warn(
                            f"[WARNING] Value for ncon_per_world is changed from {ncon_per_world} to {self.mj_data.ncon} following an MjWarp requirement.",
                            stacklevel=2,
                        )
                        naconmax = self.mj_data.ncon * nworld
                    else:
                        naconmax = rigid_contact_max
                else:
                    naconmax = max(512, self.mj_data.ncon * nworld)

            if njmax is not None:
                if njmax < self.mj_data.nefc:
                    warnings.warn(
                        f"[WARNING] Value for njmax is changed from {njmax} to {self.mj_data.nefc} following an MjWarp requirement.",
                        stacklevel=2,
                    )
                    njmax = self.mj_data.nefc
            else:
                njmax = self.mj_data.nefc

            self.mjw_data = mujoco_warp.put_data(
                self.mj_model,
                self.mj_data,
                nworld=nworld,
                naconmax=naconmax,
                njmax=njmax,
            )

    def expand_model_fields(self, mj_model: MjWarpModel, nworld: int):
        if nworld == 1:
            return

        model_fields_to_expand = {
            # "qpos0",
            # "qpos_spring",
            "body_pos",
            "body_quat",
            "body_ipos",
            "body_iquat",
            "body_mass",
            # "body_subtreemass",
            "body_inertia",
            # "body_invweight0",
            # "body_gravcomp",
            # "jnt_solref",
            # "jnt_solimp",
            "jnt_pos",
            "jnt_axis",
            # "jnt_stiffness",
            # "jnt_range",
            # "jnt_actfrcrange",
            # "jnt_margin",
            "dof_armature",
            # "dof_damping",
            # "dof_invweight0",
            "dof_frictionloss",
            # "dof_solimp",
            # "dof_solref",
            # "geom_matid",
            # "geom_solmix",
            "geom_solref",
            # "geom_solimp",
            "geom_size",
            "geom_rbound",
            "geom_pos",
            "geom_quat",
            "geom_friction",
            # "geom_margin",
            # "geom_gap",
            # "geom_rgba",
            # "site_pos",
            # "site_quat",
            # "cam_pos",
            # "cam_quat",
            # "cam_poscom0",
            # "cam_pos0",
            # "cam_mat0",
            # "light_pos",
            # "light_dir",
            # "light_poscom0",
            # "light_pos0",
            # "eq_solref",
            # "eq_solimp",
            # "eq_data",
            # "actuator_dynprm",
            "actuator_gainprm",
            "actuator_biasprm",
            # "actuator_ctrlrange",
            "actuator_forcerange",
            # "actuator_actrange",
            # "actuator_gear",
            # "pair_solref",
            # "pair_solreffriction",
            # "pair_solimp",
            # "pair_margin",
            # "pair_gap",
            # "pair_friction",
            # "tendon_solref_lim",
            # "tendon_solimp_lim",
            # "tendon_range",
            # "tendon_margin",
            # "tendon_length0",
            # "tendon_invweight0",
            # "mat_rgba",
        }

        def tile(x: wp.array):
            # Create new array with same shape but first dim multiplied by nworld
            new_shape = list(x.shape)
            new_shape[0] = nworld
            wp_array = {1: wp.array, 2: wp.array2d, 3: wp.array3d, 4: wp.array4d}[len(new_shape)]
            dst = wp_array(shape=new_shape, dtype=x.dtype, device=x.device)

            # Flatten arrays for kernel
            src_flat = x.flatten()
            dst_flat = dst.flatten()

            # Launch kernel to repeat data - one thread per destination element
            n_elems_per_world = dst_flat.shape[0] // nworld
            wp.launch(
                repeat_array_kernel,
                dim=dst_flat.shape[0],
                inputs=[src_flat, n_elems_per_world],
                outputs=[dst_flat],
                device=x.device,
            )
            return dst

        for field in mj_model.__dataclass_fields__:
            if field in model_fields_to_expand:
                array = getattr(mj_model, field)
                setattr(mj_model, field, tile(array))

    def update_model_inertial_properties(self):
        if self.model.body_count == 0:
            return

        bodies_per_world = self.model.body_count // self.model.num_worlds

        wp.launch(
            update_body_mass_ipos_kernel,
            dim=self.model.body_count,
            inputs=[
                self.model.body_com,
                self.model.body_mass,
                bodies_per_world,
                self.model.up_axis,
                self.to_mjc_body_index,
            ],
            outputs=[self.mjw_model.body_ipos, self.mjw_model.body_mass],
            device=self.model.device,
        )

        wp.launch(
            update_body_inertia_kernel,
            dim=self.model.body_count,
            inputs=[
                self.model.body_inertia,
                bodies_per_world,
                self.to_mjc_body_index,
            ],
            outputs=[self.mjw_model.body_inertia, self.mjw_model.body_iquat],
            device=self.model.device,
        )

    def update_joint_dof_properties(self):
        """Update all joint dof properties including effort limits, velocity limits, friction, and armature in the MuJoCo model."""
        if self.model.joint_dof_count == 0:
            return

        dofs_per_world = self.model.joint_dof_count // self.model.num_worlds

        # Update actuator force ranges (effort limits) if actuators exist
        if self.mjc_axis_to_actuator is not None:
            wp.launch(
                update_axis_properties_kernel,
                dim=self.model.joint_dof_count,
                inputs=[
                    self.model.joint_dof_mode,
                    self.model.joint_target_ke,
                    self.model.joint_target_kd,
                    self.model.joint_effort_limit,
                    self.mjc_axis_to_actuator,
                    dofs_per_world,
                ],
                outputs=[
                    self.mjw_model.actuator_biasprm,
                    self.mjw_model.actuator_gainprm,
                    self.mjw_model.actuator_forcerange,
                ],
                device=self.model.device,
            )

        # Update DOF properties (armature and friction)
        wp.launch(
            update_dof_properties_kernel,
            dim=self.model.joint_dof_count,
            inputs=[
                self.model.joint_armature,
                self.model.joint_friction,
                dofs_per_world,
            ],
            outputs=[self.mjw_model.dof_armature, self.mjw_model.dof_frictionloss],
            device=self.model.device,
        )

    def update_joint_properties(self):
        """Update joint properties including joint positions, joint axes, and relative body transforms in the MuJoCo model."""
        if self.model.joint_count == 0:
            return

        joints_per_world = self.model.joint_count // self.model.num_worlds

        # Update joint positions, joint axes, and relative body transforms
        wp.launch(
            update_joint_transforms_kernel,
            dim=self.model.joint_count,
            inputs=[
                self.model.joint_X_p,
                self.model.joint_X_c,
                self.model.joint_qd_start,
                self.model.joint_dof_dim,
                self.model.joint_axis,
                self.model.joint_child,
                self.model.joint_type,
                self.joint_mjc_dof_start,
                self.to_mjc_body_index,
                joints_per_world,
            ],
            outputs=[
                self.mjw_model.jnt_pos,
                self.mjw_model.jnt_axis,
                self.mjw_model.body_pos,
                self.mjw_model.body_quat,
            ],
            device=self.model.device,
        )

    def update_geom_properties(self):
        """Update geom properties including collision radius, friction, and contact parameters in the MuJoCo model."""

        # Get number of geoms and worlds from MuJoCo model
        num_geoms = self.mj_model.ngeom
        if num_geoms == 0:
            return

        num_worlds = self.model.num_worlds

        wp.launch(
            update_geom_properties_kernel,
            dim=(num_worlds, num_geoms),
            inputs=[
                self.model.shape_collision_radius,
                self.model.shape_material_mu,
                self.model.shape_material_ke,
                self.model.shape_material_kd,
                self.model.shape_scale,
                self.model.shape_transform,
                self.to_newton_shape_index,
                self.shape_incoming_xform,
                self.model.rigid_contact_torsional_friction,
                self.model.rigid_contact_rolling_friction,
                self.contact_stiffness_time_const,
            ],
            outputs=[
                self.mjw_model.geom_rbound,
                self.mjw_model.geom_friction,
                self.mjw_model.geom_solref,
                self.mjw_model.geom_size,
                self.mjw_model.geom_pos,
                self.mjw_model.geom_quat,
            ],
            device=self.model.device,
        )

    def update_model_properties(self):
        """Update model properties including gravity in the MuJoCo model."""
        if self.use_mujoco_cpu:
            self.mj_model.opt.gravity[:] = np.array([*self.model.gravity.numpy()[0]])
        else:
            if hasattr(self, "mjw_data"):
                wp.launch(
                    kernel=update_model_properties_kernel,
                    dim=self.mjw_data.nworld,
                    inputs=[
                        self.model.gravity,
                    ],
                    outputs=[
                        self.mjw_model.opt.gravity,
                    ],
                    device=self.model.device,
                )

    def render_mujoco_viewer(
        self,
        show_left_ui: bool = True,
        show_right_ui: bool = True,
        show_contact_points: bool = True,
        show_contact_forces: bool = False,
        show_transparent_geoms: bool = True,
    ):
        """Create and synchronize the MuJoCo viewer.
        The viewer will be created if it is not already open.

        .. note::

            The MuJoCo viewer only supports rendering Newton models with a single world,
            unless :attr:`use_mujoco_cpu` is :obj:`True` or the solver was initialized with
            :attr:`separate_worlds` set to :obj:`False`.

            The MuJoCo viewer is only meant as a debugging tool.

        Args:
            show_left_ui: Whether to show the left UI.
            show_right_ui: Whether to show the right UI.
            show_contact_points: Whether to show contact points.
            show_contact_forces: Whether to show contact forces.
            show_transparent_geoms: Whether to show transparent geoms.
        """
        if self._viewer is None:
            import mujoco  # noqa: PLC0415
            import mujoco.viewer  # noqa: PLC0415

            # make the headlights brighter to improve visibility
            # in the MuJoCo viewer
            self.mj_model.vis.headlight.ambient[:] = [0.3, 0.3, 0.3]
            self.mj_model.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
            self.mj_model.vis.headlight.specular[:] = [0.9, 0.9, 0.9]

            self._viewer = mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data, show_left_ui=show_left_ui, show_right_ui=show_right_ui
            )
            # Enter the context manager to keep the viewer alive
            self._viewer.__enter__()

            self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = show_contact_points
            self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = show_contact_forces
            self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = show_transparent_geoms

        if self._viewer.is_running():
            if not self.use_mujoco_cpu:
                self._mujoco_warp.get_data_into(self.mj_data, self.mj_model, self.mjw_data)

            self._viewer.sync()

    def close_mujoco_viewer(self):
        """Close the MuJoCo viewer if it exists."""
        if hasattr(self, "_viewer") and self._viewer is not None:
            try:
                self._viewer.__exit__(None, None, None)
            except Exception:
                pass  # Ignore errors during cleanup
            finally:
                self._viewer = None

    def __del__(self):
        """Cleanup method to close the viewer when the solver is destroyed."""
        self.close_mujoco_viewer()
