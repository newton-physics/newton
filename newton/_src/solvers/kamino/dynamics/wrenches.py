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

"""
KAMINO: Dynamics: Wrenches
"""

from __future__ import annotations

import warp as wp

from ..core.model import Model, ModelData
from ..core.types import float32, int32, mat63f, vec2i, vec3f, vec4f, vec6f
from ..geometry.contacts import ContactsData
from ..kinematics.jacobians import DenseSystemJacobiansData
from ..kinematics.limits import LimitsData

###
# Module interface
###

__all__ = [
    "compute_constraint_body_wrenches",
    "compute_joint_dof_body_wrenches",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


@wp.kernel
def _compute_joint_dof_body_wrenches(
    # Inputs:
    model_info_num_body_dofs: wp.array(dtype=int32),
    model_info_bodies_offset: wp.array(dtype=int32),
    model_info_joint_dofs_offset: wp.array(dtype=int32),
    model_joints_num_dofs: wp.array(dtype=int32),
    model_joints_dofs_offset: wp.array(dtype=int32),
    model_joints_wid: wp.array(dtype=int32),
    model_joints_bid_B: wp.array(dtype=int32),
    model_joints_bid_F: wp.array(dtype=int32),
    state_joints_tau_j: wp.array(dtype=float32),
    jacobian_dofs_offsets: wp.array(dtype=int32),
    jacobian_dofs_data: wp.array(dtype=float32),
    # Outputs:
    state_bodies_w_a: wp.array(dtype=vec6f),
):
    # Retrieve the thread index as the joint index
    jid = wp.tid()

    # Retrieve the world index of the joint
    wid = model_joints_wid[jid]

    # Retrieve the body indices of the joint
    # NOTE: these indices are w.r.t the model
    bid_F_j = model_joints_bid_F[jid]
    bid_B_j = model_joints_bid_B[jid]

    # Retrieve the size and index offset of the joint DoFs
    d_j = model_joints_num_dofs[jid]
    dio_j = model_joints_dofs_offset[jid]

    # Retrieve the number of body DoFs in the world
    nbd = model_info_num_body_dofs[wid]

    # Retrieve the element index offset of the bodies of the world
    bio = model_info_bodies_offset[wid]

    # Retrieve the DoF block index offsets of the world's actuation
    # Jacobian matrix and generalized joint actuation force vector
    mio = jacobian_dofs_offsets[wid]
    vio = model_info_joint_dofs_offset[wid]

    # Append offsets to the current joint's DoFs
    vio += dio_j
    mio += nbd * dio_j

    # Compute and store the joint actuation wrench for the Follower body
    w_j_F = vec6f(0.0)
    dio_F = 6 * (bid_F_j - bio)
    for j in range(d_j):
        mio_j = mio + nbd * j + dio_F
        vio_j = vio + j
        tau_j = state_joints_tau_j[vio_j]
        for i in range(6):
            w_j_F[i] += jacobian_dofs_data[mio_j + i] * tau_j
    wp.atomic_add(state_bodies_w_a, bid_F_j, w_j_F)

    # Compute and store the joint actuation wrench for the Base body if bid_B >= 0
    if bid_B_j >= 0:
        w_j_B = vec6f(0.0)
        dio_B = 6 * (bid_B_j - bio)
        for j in range(d_j):
            mio_j = mio + nbd * j + dio_B
            vio_j = vio + j
            tau_j = state_joints_tau_j[vio_j]
            for i in range(6):
                w_j_B[i] += jacobian_dofs_data[mio_j + i] * tau_j
        wp.atomic_add(state_bodies_w_a, bid_B_j, w_j_B)


@wp.kernel
def _compute_joint_cts_body_wrenches(
    # Inputs:
    model_info_num_body_dofs: wp.array(dtype=int32),
    model_info_bodies_offset: wp.array(dtype=int32),
    model_time_inv_dt: wp.array(dtype=float32),
    model_joints_wid: wp.array(dtype=int32),
    model_joints_m: wp.array(dtype=int32),
    model_joints_cio: wp.array(dtype=int32),
    model_joints_bid_B: wp.array(dtype=int32),
    model_joints_bid_F: wp.array(dtype=int32),
    jacobian_cts_offset: wp.array(dtype=int32),
    jacobian_cts_data: wp.array(dtype=float32),
    lambdas_offsets: wp.array(dtype=int32),
    lambdas_data: wp.array(dtype=float32),
    # Outputs:
    state_bodies_w_j: wp.array(dtype=vec6f),
):
    # Retrieve the thread index as the joint index
    jid = wp.tid()

    # Retrieve the world index of the joint
    wid = model_joints_wid[jid]

    # Retrieve the body indices of the joint
    # NOTE: these indices are w.r.t the model
    bid_F_j = model_joints_bid_F[jid]
    bid_B_j = model_joints_bid_B[jid]

    # Retrieve the size and index offset of the joint constraint
    m_j = model_joints_m[jid]
    cio_j = model_joints_cio[jid]

    # Retrieve the number of body DoFs in the world
    nbd = model_info_num_body_dofs[wid]

    # Retrieve the element index offset of the bodies of the world
    bio = model_info_bodies_offset[wid]

    # Retrieve the inverse time-step of the world
    inv_dt = model_time_inv_dt[wid]

    # Retrieve the constraint block index offsets of the
    # Jacobian matrix and multipliers vector of the world
    mio = jacobian_cts_offset[wid]
    vio = lambdas_offsets[wid]

    # Compute and store the joint constraint wrench for the Follower body
    # NOTE: We need to scale by the time-step because the lambdas are impulses
    w_j_F = vec6f(0.0)
    dio_F = 6 * (bid_F_j - bio)
    for j in range(m_j):
        mio_j = mio + nbd * (cio_j + j) + dio_F
        vio_j = vio + cio_j + j
        lambda_j = lambdas_data[vio_j]
        for i in range(6):
            w_j_F[i] += jacobian_cts_data[mio_j + i] * lambda_j
    w_j_F *= inv_dt
    wp.atomic_add(state_bodies_w_j, bid_F_j, w_j_F)

    # Compute and store the joint constraint wrench for the Base body if bid_B >= 0
    # NOTE: We need to scale by the time-step because the lambdas are impulses
    if bid_B_j >= 0:
        w_j_B = vec6f(0.0)
        dio_B = 6 * (bid_B_j - bio)
        for j in range(m_j):
            mio_j = mio + nbd * (cio_j + j) + dio_B
            vio_j = vio + cio_j + j
            lambda_j = lambdas_data[vio_j]
            for i in range(6):
                w_j_B[i] += jacobian_cts_data[mio_j + i] * lambda_j
        w_j_B *= inv_dt
        wp.atomic_add(state_bodies_w_j, bid_B_j, w_j_B)


@wp.kernel
def _compute_limit_cts_body_wrenches(
    # Inputs:
    model_info_num_body_dofs: wp.array(dtype=int32),
    model_info_bodies_offset: wp.array(dtype=int32),
    state_info_limit_cts_group_offset: wp.array(dtype=int32),
    model_time_inv_dt: wp.array(dtype=float32),
    limits_model_num: wp.array(dtype=int32),
    limits_wid: wp.array(dtype=int32),
    limits_lid: wp.array(dtype=int32),
    limits_bids: wp.array(dtype=vec2i),
    jacobian_cts_offset: wp.array(dtype=int32),
    jacobian_cts_data: wp.array(dtype=float32),
    lambdas_offsets: wp.array(dtype=int32),
    lambdas_data: wp.array(dtype=float32),
    # Outputs:
    state_bodies_w_l: wp.array(dtype=vec6f),
):
    # Retrieve the thread index
    tid = wp.tid()

    # Skip if tid is greater than the number of active contacts in the model
    if tid >= limits_model_num[0]:
        return

    # Retrieve the contact index of the contact w.r.t the world
    lid = limits_lid[tid]

    # Retrieve the world index of the contact
    wid = limits_wid[tid]

    # Extract the body indices associated with the contact
    # NOTE: These indices are w.r.t the model
    bids = limits_bids[tid]
    bid_B = bids[0]
    bid_F = bids[1]

    # Retrieve the inverse time-step of the world
    inv_dt = model_time_inv_dt[wid]

    # Retrieve the world-specific info data
    nbd = model_info_num_body_dofs[wid]
    bio = model_info_bodies_offset[wid]
    mio = jacobian_cts_offset[wid]
    vio = lambdas_offsets[wid]

    # Retrieve the index offset of the active contact constraints of the world
    lcgo = state_info_limit_cts_group_offset[wid]

    # Compute the index offsets of the contact constraint
    cio_l = lcgo + lid
    vio_l = vio + cio_l
    mio_l = mio + nbd * cio_l

    # Extract the limit force/torque from the impulse
    # NOTE: We need to scale by the time-step because the lambdas are impulses
    lambda_l = inv_dt * lambdas_data[vio_l]

    # Extract the contact constraint Jacobian for the follower body
    JT_l_F = vec6f(0.0)
    dio_F = 6 * (bid_F - bio)
    for _j in range(3):
        mio_lF = mio_l + dio_F
        for i in range(6):
            JT_l_F[i] = jacobian_cts_data[mio_lF + i]

    # Compute the contact constraint wrench for the follower body
    w_c_F = JT_l_F * lambda_l

    # Store the contact constraint wrench for the follower body
    wp.atomic_add(state_bodies_w_l, bid_F, w_c_F)

    # Compute the limit constraint wrench for the joint base body if bid_B >= 0
    if bid_B >= 0:
        # Extract the contact constraint Jacobian for the base body
        JT_l_B = vec6f(0.0)
        dio_B = 6 * (bid_B - bio)
        mio_lB = mio_l + dio_B
        for i in range(6):
            JT_l_B[i] = jacobian_cts_data[mio_lB + i]

        # Compute the contact constraint wrench for the base body
        w_c_B = JT_l_B * lambda_l

        # Store the contact constraint wrench for the base body
        wp.atomic_add(state_bodies_w_l, bid_B, w_c_B)


@wp.kernel
def _compute_contact_cts_body_wrenches(
    # Inputs:
    model_info_num_body_dofs: wp.array(dtype=int32),
    model_info_bodies_offset: wp.array(dtype=int32),
    state_info_contact_cts_group_offset: wp.array(dtype=int32),
    model_time_inv_dt: wp.array(dtype=float32),
    contacts_model_num: wp.array(dtype=int32),
    contacts_wid: wp.array(dtype=int32),
    contacts_cid: wp.array(dtype=int32),
    contacts_body_A: wp.array(dtype=vec4f),
    contacts_body_B: wp.array(dtype=vec4f),
    jacobian_cts_offset: wp.array(dtype=int32),
    jacobian_cts_data: wp.array(dtype=float32),
    lambdas_offsets: wp.array(dtype=int32),
    lambdas_data: wp.array(dtype=float32),
    # Outputs:
    state_bodies_w_c: wp.array(dtype=vec6f),
):
    # Retrieve the thread index
    tid = wp.tid()

    # Skip if tid is greater than the number of active contacts in the model
    if tid >= contacts_model_num[0]:
        return

    # Retrieve the contact index of the contact w.r.t the world
    cid = contacts_cid[tid]

    # Retrieve the world index of the contact
    wid = contacts_wid[tid]

    # Extract the body indices associated with the contact
    # NOTE: These indices are w.r.t the model
    bid_B = int(contacts_body_B[tid][3])
    bid_A = int(contacts_body_A[tid][3])

    # Retrieve the inverse time-step of the world
    inv_dt = model_time_inv_dt[wid]

    # Retrieve the world-specific info data
    nbd = model_info_num_body_dofs[wid]
    bio = model_info_bodies_offset[wid]
    mio = jacobian_cts_offset[wid]
    vio = lambdas_offsets[wid]

    # Retrieve the index offset of the active contact constraints of the world
    ccgo = state_info_contact_cts_group_offset[wid]

    # Compute the index offsets of the contact constraint
    k = 3 * cid
    cio_k = ccgo + k
    vio_k = vio + cio_k
    mio_k = mio + nbd * cio_k

    # Extract the 3D contact force
    # NOTE: We need to scale by the time-step because the lambdas are impulses
    # TODO: Add helper function to extract 3D vectors from flat arrays
    lambda_c = inv_dt * vec3f(lambdas_data[vio_k], lambdas_data[vio_k + 1], lambdas_data[vio_k + 2])

    # Extract the contact constraint Jacobian for body B
    JT_c_B = mat63f(0.0)
    dio_B = 6 * (bid_B - bio)
    for j in range(3):
        mio_kj = mio_k + nbd * j + dio_B
        for i in range(6):
            JT_c_B[i, j] = jacobian_cts_data[mio_kj + i]

    # Compute the contact constraint wrench for body B
    w_c_B = JT_c_B @ lambda_c

    # Store the contact constraint wrench for body B
    wp.atomic_add(state_bodies_w_c, bid_B, w_c_B)

    # Compute the contact constraint wrench for body A if bid_A >= 0
    if bid_A >= 0:
        # Extract the contact constraint Jacobian for body A
        JT_c_A = mat63f(0.0)
        dio_A = 6 * (bid_A - bio)
        for j in range(3):
            mio_kj = mio_k + nbd * j + dio_A
            for i in range(6):
                JT_c_A[i, j] = jacobian_cts_data[mio_kj + i]

        # Compute the contact constraint wrench for body A
        w_c_A = JT_c_A @ lambda_c

        # Store the contact constraint wrench for body A
        wp.atomic_add(state_bodies_w_c, bid_A, w_c_A)


###
# Launchers
###


def compute_joint_dof_body_wrenches(
    model: Model, data: ModelData, jacobians: DenseSystemJacobiansData, reset_to_zero: bool = True
) -> None:
    """
    Update the actuation wrenches of the bodies based on the active joint torques.
    """
    # First clear the previous actuation wrenches, because the kernel computing them
    # uses an atomic add to accumulate contributions from each joint DoF, and thus
    # assumes the target array is zeroed out before each call
    if reset_to_zero:
        data.bodies.w_a_i.zero_()

    # Then compute the body wrenches resulting from the current generalized actuation forces
    wp.launch(
        _compute_joint_dof_body_wrenches,
        dim=model.size.sum_of_num_joints,
        inputs=[
            # Inputs:
            model.info.num_body_dofs,
            model.info.bodies_offset,
            model.info.joint_dofs_offset,
            model.joints.num_dofs,
            model.joints.dofs_offset,
            model.joints.wid,
            model.joints.bid_B,
            model.joints.bid_F,
            data.joints.tau_j,
            jacobians.J_dofs_offsets,
            jacobians.J_dofs_data,
            # Outputs:
            data.bodies.w_a_i,
        ],
    )


def compute_constraint_body_wrenches(
    model: Model,
    data: ModelData,
    limits: LimitsData,
    contacts: ContactsData,
    jacobians: DenseSystemJacobiansData,
    lambdas_offsets: wp.array(dtype=int32),
    lambdas_data: wp.array(dtype=float32),
    reset_to_zero: bool = True,
):
    """
    Launches the kernels to compute the body-wise constraint wrenches.
    """

    if model.size.sum_of_num_joints > 0:
        if reset_to_zero:
            data.bodies.w_j_i.zero_()
        wp.launch(
            _compute_joint_cts_body_wrenches,
            dim=model.size.sum_of_num_joints,
            inputs=[
                # Inputs:
                model.info.num_body_dofs,
                model.info.bodies_offset,
                model.time.inv_dt,
                model.joints.wid,
                model.joints.num_cts,
                model.joints.cts_offset,
                model.joints.bid_B,
                model.joints.bid_F,
                jacobians.J_cts_offsets,
                jacobians.J_cts_data,
                lambdas_offsets,
                lambdas_data,
                # Outputs:
                data.bodies.w_j_i,
            ],
        )

    if limits is not None:
        if reset_to_zero:
            data.bodies.w_l_i.zero_()
        wp.launch(
            _compute_limit_cts_body_wrenches,
            dim=limits.num_model_max_limits,
            inputs=[
                # Inputs:
                model.info.num_body_dofs,
                model.info.bodies_offset,
                data.info.limit_cts_group_offset,
                model.time.inv_dt,
                limits.model_num_limits,
                limits.wid,
                limits.lid,
                limits.bids,
                jacobians.J_cts_offsets,
                jacobians.J_cts_data,
                lambdas_offsets,
                lambdas_data,
                # Outputs:
                data.bodies.w_l_i,
            ],
        )

    if contacts is not None:
        if reset_to_zero:
            data.bodies.w_c_i.zero_()
        wp.launch(
            _compute_contact_cts_body_wrenches,
            dim=contacts.num_model_max_contacts,
            inputs=[
                # Inputs:
                model.info.num_body_dofs,
                model.info.bodies_offset,
                data.info.contact_cts_group_offset,
                model.time.inv_dt,
                contacts.model_num_contacts,
                contacts.wid,
                contacts.cid,
                contacts.body_A,
                contacts.body_B,
                jacobians.J_cts_offsets,
                jacobians.J_cts_data,
                lambdas_offsets,
                lambdas_data,
                # Outputs:
                data.bodies.w_c_i,
            ],
        )
