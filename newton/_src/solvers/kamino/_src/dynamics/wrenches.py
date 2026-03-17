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

import warp as wp

from ..core.data import DataKamino
from ..core.model import ModelKamino
from ..core.types import float32, int32, mat63f, vec2i, vec3f, vec6f
from ..geometry.contacts import ContactsKamino
from ..kinematics.jacobians import DenseSystemJacobians, SparseSystemJacobians
from ..kinematics.limits import LimitsKamino

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
def _compute_joint_dof_body_wrenches_dense(
    # Inputs:
    model_info_num_body_dofs: wp.array(dtype=int32),
    model_info_bodies_offset: wp.array(dtype=int32),
    model_info_joint_dofs_offset: wp.array(dtype=int32),
    model_joints_num_dofs: wp.array(dtype=int32),
    model_joints_dofs_offset: wp.array(dtype=int32),
    model_joints_wid: wp.array(dtype=int32),
    model_joints_bid_B: wp.array(dtype=int32),
    model_joints_bid_F: wp.array(dtype=int32),
    data_joints_tau_j: wp.array(dtype=float32),
    jacobian_dofs_offsets: wp.array(dtype=int32),
    jacobian_dofs_data: wp.array(dtype=float32),
    # Outputs:
    data_bodies_w_a: wp.array(dtype=vec6f),
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
        tau_j = data_joints_tau_j[vio_j]
        for i in range(6):
            w_j_F[i] += jacobian_dofs_data[mio_j + i] * tau_j
    wp.atomic_add(data_bodies_w_a, bid_F_j, w_j_F)

    # Compute and store the joint actuation wrench for the Base body if bid_B >= 0
    if bid_B_j >= 0:
        w_j_B = vec6f(0.0)
        dio_B = 6 * (bid_B_j - bio)
        for j in range(d_j):
            mio_j = mio + nbd * j + dio_B
            vio_j = vio + j
            tau_j = data_joints_tau_j[vio_j]
            for i in range(6):
                w_j_B[i] += jacobian_dofs_data[mio_j + i] * tau_j
        wp.atomic_add(data_bodies_w_a, bid_B_j, w_j_B)


@wp.kernel
def _compute_joint_dof_body_wrenches_sparse(
    # Inputs:
    model_info_joint_dofs_offset: wp.array(dtype=int32),
    model_joints_num_dofs: wp.array(dtype=int32),
    model_joints_dofs_offset: wp.array(dtype=int32),
    model_joints_wid: wp.array(dtype=int32),
    model_joints_bid_B: wp.array(dtype=int32),
    model_joints_bid_F: wp.array(dtype=int32),
    data_joints_tau_j: wp.array(dtype=float32),
    jac_joint_nzb_offsets: wp.array(dtype=int32),
    jac_nzb_values: wp.array(dtype=vec6f),
    # Outputs:
    data_bodies_w_a: wp.array(dtype=vec6f),
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

    # Retrieve the DoF block index offsets of the world's actuation
    # Jacobian matrix and generalized joint actuation force vector
    vio = model_info_joint_dofs_offset[wid]

    # Append offsets to the current joint's DoFs
    vio += dio_j

    # Retrieve the starting index for the non-zero blocks for the current joint
    jac_j_nzb_start = jac_joint_nzb_offsets[jid]

    # Compute and store the joint actuation wrench for the Follower body
    w_j_F = vec6f(0.0)
    for j in range(d_j):
        jac_block = jac_nzb_values[jac_j_nzb_start + j]
        vio_j = vio + j
        tau_j = data_joints_tau_j[vio_j]
        w_j_F += jac_block * tau_j
    wp.atomic_add(data_bodies_w_a, bid_F_j, w_j_F)

    # Compute and store the joint actuation wrench for the Base body if bid_B >= 0
    if bid_B_j >= 0:
        w_j_B = vec6f(0.0)
        for j in range(d_j):
            jac_block = jac_nzb_values[jac_j_nzb_start + d_j + j]
            vio_j = vio + j
            tau_j = data_joints_tau_j[vio_j]
            w_j_B += jac_block * tau_j
        wp.atomic_add(data_bodies_w_a, bid_B_j, w_j_B)


@wp.kernel
def _compute_joint_cts_body_wrenches_dense(
    # Inputs:
    model_info_num_body_dofs: wp.array(dtype=int32),
    model_info_bodies_offset: wp.array(dtype=int32),
    model_info_joint_dynamic_cts_group_offset: wp.array(dtype=int32),
    model_info_joint_kinematic_cts_group_offset: wp.array(dtype=int32),
    model_time_inv_dt: wp.array(dtype=float32),
    model_joints_wid: wp.array(dtype=int32),
    model_joints_num_dynamic_cts: wp.array(dtype=int32),
    model_joints_num_kinematic_cts: wp.array(dtype=int32),
    model_joints_dynamic_cts_offset: wp.array(dtype=int32),
    model_joints_kinematic_cts_offset: wp.array(dtype=int32),
    model_joints_bid_B: wp.array(dtype=int32),
    model_joints_bid_F: wp.array(dtype=int32),
    jacobian_cts_offset: wp.array(dtype=int32),
    jacobian_cts_data: wp.array(dtype=float32),
    lambdas_offsets: wp.array(dtype=int32),
    lambdas_data: wp.array(dtype=float32),
    # Outputs:
    data_bodies_w_j: wp.array(dtype=vec6f),
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
    num_dyn_cts_j = model_joints_num_dynamic_cts[jid]
    num_kin_cts_j = model_joints_num_kinematic_cts[jid]
    dyn_cts_start_j = model_joints_dynamic_cts_offset[jid]
    kin_cts_start_j = model_joints_kinematic_cts_offset[jid]

    # Retrieve the number of body DoFs in the world
    nbd = model_info_num_body_dofs[wid]

    # Retrieve the element index offset of the bodies of the world
    bio = model_info_bodies_offset[wid]

    # Retrieve the index offsets of the active joint dynamic and kinematic constraints of the world
    world_jdcgo = model_info_joint_dynamic_cts_group_offset[wid]
    world_jkcgo = model_info_joint_kinematic_cts_group_offset[wid]

    # Retrieve the inverse time-step of the world
    inv_dt = model_time_inv_dt[wid]

    # Retrieve the constraint block index offsets of the
    # Jacobian matrix and multipliers vector of the world
    world_jacobian_start = jacobian_cts_offset[wid]
    world_cts_start = lambdas_offsets[wid]

    # Compute and store the joint constraint wrench for the Follower body
    # NOTE: We need to scale by the time-step because the lambdas are impulses
    w_j_F = vec6f(0.0)
    col_F_start = 6 * (bid_F_j - bio)
    for j in range(num_dyn_cts_j):
        row_j = world_jdcgo + dyn_cts_start_j + j
        mio_j = world_jacobian_start + nbd * row_j + col_F_start
        vio_j = world_cts_start + row_j
        lambda_j = inv_dt * lambdas_data[vio_j]
        for i in range(6):
            w_j_F[i] += jacobian_cts_data[mio_j + i] * lambda_j
    for j in range(num_kin_cts_j):
        row_j = world_jkcgo + kin_cts_start_j + j
        mio_j = world_jacobian_start + nbd * row_j + col_F_start
        vio_j = world_cts_start + row_j
        lambda_j = inv_dt * lambdas_data[vio_j]
        for i in range(6):
            w_j_F[i] += jacobian_cts_data[mio_j + i] * lambda_j
    wp.atomic_add(data_bodies_w_j, bid_F_j, w_j_F)

    # Compute and store the joint constraint wrench for the Base body if bid_B >= 0
    # NOTE: We need to scale by the time-step because the lambdas are impulses
    if bid_B_j >= 0:
        w_j_B = vec6f(0.0)
        col_B_start = 6 * (bid_B_j - bio)
        for j in range(num_dyn_cts_j):
            row_j = world_jdcgo + dyn_cts_start_j + j
            mio_j = world_jacobian_start + nbd * row_j + col_B_start
            vio_j = world_cts_start + row_j
            lambda_j = inv_dt * lambdas_data[vio_j]
            for i in range(6):
                w_j_B[i] += jacobian_cts_data[mio_j + i] * lambda_j
        for j in range(num_kin_cts_j):
            row_j = world_jkcgo + kin_cts_start_j + j
            mio_j = world_jacobian_start + nbd * row_j + col_B_start
            vio_j = world_cts_start + row_j
            lambda_j = inv_dt * lambdas_data[vio_j]
            for i in range(6):
                w_j_B[i] += jacobian_cts_data[mio_j + i] * lambda_j
        wp.atomic_add(data_bodies_w_j, bid_B_j, w_j_B)


@wp.kernel
def _compute_limit_cts_body_wrenches_dense(
    # Inputs:
    model_info_num_body_dofs: wp.array(dtype=int32),
    model_info_bodies_offset: wp.array(dtype=int32),
    data_info_limit_cts_group_offset: wp.array(dtype=int32),
    model_time_inv_dt: wp.array(dtype=float32),
    limits_model_num: wp.array(dtype=int32),
    limits_model_max: int32,
    limits_wid: wp.array(dtype=int32),
    limits_lid: wp.array(dtype=int32),
    limits_bids: wp.array(dtype=vec2i),
    jacobian_cts_offset: wp.array(dtype=int32),
    jacobian_cts_data: wp.array(dtype=float32),
    lambdas_offsets: wp.array(dtype=int32),
    lambdas_data: wp.array(dtype=float32),
    # Outputs:
    data_bodies_w_l: wp.array(dtype=vec6f),
):
    # Retrieve the thread index
    tid = wp.tid()

    # Skip if tid is greater than the number of active limits in the model
    if tid >= wp.min(limits_model_num[0], limits_model_max):
        return

    # Retrieve the limit index of the limit w.r.t the world
    lid = limits_lid[tid]

    # Retrieve the world index of the limit
    wid = limits_wid[tid]

    # Extract the body indices associated with the limit
    # NOTE: These indices are w.r.t the model
    bids = limits_bids[tid]
    bid_B = bids[0]
    bid_F = bids[1]

    # Retrieve the inverse time-step of the world
    inv_dt = model_time_inv_dt[wid]

    # Retrieve the world-specific info
    nbd = model_info_num_body_dofs[wid]
    bio = model_info_bodies_offset[wid]
    mio = jacobian_cts_offset[wid]
    vio = lambdas_offsets[wid]

    # Retrieve the index offset of the active limit constraints of the world
    lcgo = data_info_limit_cts_group_offset[wid]

    # Compute the index offsets of the limit constraint
    cio_l = lcgo + lid
    vio_l = vio + cio_l
    mio_l = mio + nbd * cio_l

    # Extract the limit force/torque from the impulse
    # NOTE: We need to scale by the time-step because the lambdas are impulses
    lambda_l = inv_dt * lambdas_data[vio_l]

    # Extract the limit constraint Jacobian for the follower body
    JT_l_F = vec6f(0.0)
    dio_F = 6 * (bid_F - bio)
    mio_lF = mio_l + dio_F
    for i in range(6):
        JT_l_F[i] = jacobian_cts_data[mio_lF + i]

    # Compute the limit constraint wrench for the follower body
    w_l_F = JT_l_F * lambda_l

    # Store the limit constraint wrench for the follower body
    wp.atomic_add(data_bodies_w_l, bid_F, w_l_F)

    # Compute the limit constraint wrench for the joint base body if bid_B >= 0
    if bid_B >= 0:
        # Extract the limit constraint Jacobian for the base body
        JT_l_B = vec6f(0.0)
        dio_B = 6 * (bid_B - bio)
        mio_lB = mio_l + dio_B
        for i in range(6):
            JT_l_B[i] = jacobian_cts_data[mio_lB + i]

        # Compute the limit constraint wrench for the base body
        w_l_B = JT_l_B * lambda_l

        # Store the limit constraint wrench for the base body
        wp.atomic_add(data_bodies_w_l, bid_B, w_l_B)


@wp.kernel
def _compute_contact_cts_body_wrenches_dense(
    # Inputs:
    model_info_num_body_dofs: wp.array(dtype=int32),
    model_info_bodies_offset: wp.array(dtype=int32),
    data_info_contact_cts_group_offset: wp.array(dtype=int32),
    model_time_inv_dt: wp.array(dtype=float32),
    contacts_model_num: wp.array(dtype=int32),
    contacts_model_max: int32,
    contacts_wid: wp.array(dtype=int32),
    contacts_cid: wp.array(dtype=int32),
    contacts_bid_AB: wp.array(dtype=vec2i),
    jacobian_cts_offset: wp.array(dtype=int32),
    jacobian_cts_data: wp.array(dtype=float32),
    lambdas_offsets: wp.array(dtype=int32),
    lambdas_data: wp.array(dtype=float32),
    # Outputs:
    data_bodies_w_c: wp.array(dtype=vec6f),
):
    # Retrieve the thread index
    tid = wp.tid()

    # Skip if tid is greater than the number of active contacts in the model
    if tid >= wp.min(contacts_model_num[0], contacts_model_max):
        return

    # Retrieve the contact index of the contact w.r.t the world
    cid = contacts_cid[tid]

    # Retrieve the world index of the contact
    wid = contacts_wid[tid]

    # Extract the body indices associated with the contact
    # NOTE: These indices are w.r.t the model
    bid_AB = contacts_bid_AB[tid]
    bid_A = bid_AB[0]
    bid_B = bid_AB[1]

    # Retrieve the inverse time-step of the world
    inv_dt = model_time_inv_dt[wid]

    # Retrieve the world-specific info data
    nbd = model_info_num_body_dofs[wid]
    bio = model_info_bodies_offset[wid]
    mio = jacobian_cts_offset[wid]
    vio = lambdas_offsets[wid]

    # Retrieve the index offset of the active contact constraints of the world
    ccgo = data_info_contact_cts_group_offset[wid]

    # Compute the index offsets of the contact constraint
    k = 3 * cid
    cio_k = ccgo + k
    vio_k = vio + cio_k
    mio_k = mio + nbd * cio_k

    # Extract the 3D contact force
    # NOTE: We need to scale by the time-step because the lambdas are impulses
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
    wp.atomic_add(data_bodies_w_c, bid_B, w_c_B)

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
        wp.atomic_add(data_bodies_w_c, bid_A, w_c_A)


@wp.kernel
def _compute_cts_body_wrenches_sparse(
    # Inputs:
    model_time_inv_dt: wp.array(dtype=float32),
    model_info_bodies_offset: wp.array(dtype=int32),
    data_info_limit_cts_group_offset: wp.array(dtype=int32),
    data_info_contact_cts_group_offset: wp.array(dtype=int32),
    jac_num_nzb: wp.array(dtype=int32),
    jac_nzb_start: wp.array(dtype=int32),
    jac_nzb_coords: wp.array2d(dtype=int32),
    jac_nzb_values: wp.array(dtype=vec6f),
    lambdas_offsets: wp.array(dtype=int32),
    lambdas_data: wp.array(dtype=float32),
    # Outputs:
    data_bodies_w_j_i: wp.array(dtype=vec6f),
    data_bodies_w_l_i: wp.array(dtype=vec6f),
    data_bodies_w_c_i: wp.array(dtype=vec6f),
):
    # Retrieve the world and non-zero
    # block indices from the thread grid
    wid, nzbid = wp.tid()

    # Skip if the non-zero block index is greater than
    # the number of active non-zero blocks for the world
    if nzbid >= jac_num_nzb[wid]:
        return

    # Retrieve the inverse time-step of the world
    inv_dt = model_time_inv_dt[wid]

    # Retrieve world-specific index offsets
    world_bid_start = model_info_bodies_offset[wid]
    J_cts_nzb_start = jac_nzb_start[wid]
    world_cts_start = lambdas_offsets[wid]
    limit_cts_group_start = data_info_limit_cts_group_offset[wid]
    contact_cts_group_start = data_info_contact_cts_group_offset[wid]

    # Retrieve the Jacobian matrix block coordinates
    # and values for the current non-zero block
    global_nzb_idx = J_cts_nzb_start + nzbid
    J_ji_coords = jac_nzb_coords[global_nzb_idx]
    J_ji = jac_nzb_values[global_nzb_idx]

    # Get constraint and body from the block coordinates
    cts_row = J_ji_coords[0]
    bid_j = J_ji_coords[1] // 6

    # Get global body index, i.e. w.r.t the model
    global_bid_j = world_bid_start + bid_j

    # Retrieve the constraint reaction of the current constraint row
    # NOTE: We need to scale by the time-step because the lambdas are impulses
    lambda_j = inv_dt * lambdas_data[world_cts_start + cts_row]

    # Compute the joint constraint wrench for the body
    w_ij = lambda_j * J_ji

    # Add the wrench to the appropriate array
    if cts_row >= contact_cts_group_start:
        wp.atomic_add(data_bodies_w_c_i, global_bid_j, w_ij)
    elif cts_row >= limit_cts_group_start:
        wp.atomic_add(data_bodies_w_l_i, global_bid_j, w_ij)
    else:
        wp.atomic_add(data_bodies_w_j_i, global_bid_j, w_ij)


###
# Launchers
###


def compute_joint_dof_body_wrenches_dense(
    model: ModelKamino, data: DataKamino, jacobians: DenseSystemJacobians, reset_to_zero: bool = True
):
    """
    Update the actuation wrenches of the bodies based on the active joint torques.
    """
    # First check that the Jacobians are dense
    if not isinstance(jacobians, DenseSystemJacobians):
        raise ValueError(f"Expected `DenseSystemJacobians` but got {type(jacobians)}.")

    # Clear the previous actuation wrenches, because the kernel computing them
    # uses an atomic add to accumulate contributions from each joint DoF, and
    # thus assumes the target array is zeroed out before each call
    if reset_to_zero:
        data.bodies.w_a_i.zero_()

    # Then compute the body wrenches resulting from the current generalized actuation forces
    wp.launch(
        _compute_joint_dof_body_wrenches_dense,
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
            jacobians.data.J_dofs_offsets,
            jacobians.data.J_dofs_data,
            # Outputs:
            data.bodies.w_a_i,
        ],
    )


def compute_joint_dof_body_wrenches_sparse(
    model: ModelKamino, data: DataKamino, jacobians: SparseSystemJacobians, reset_to_zero: bool = True
) -> None:
    """
    Update the actuation wrenches of the bodies based on the active joint torques.
    """
    # First check that the Jacobians are sparse
    if not isinstance(jacobians, SparseSystemJacobians):
        raise ValueError(f"Expected `SparseSystemJacobians` but got {type(jacobians)}.")

    # Clear the previous actuation wrenches, because the kernel computing them
    # uses an atomic add to accumulate contributions from each joint DoF, and
    # thus assumes the target array is zeroed out before each call
    if reset_to_zero:
        data.bodies.w_a_i.zero_()

    # Then compute the body wrenches resulting from the current generalized actuation forces
    wp.launch(
        _compute_joint_dof_body_wrenches_sparse,
        dim=model.size.sum_of_num_joints,
        inputs=[
            # Inputs:
            model.info.joint_dofs_offset,
            model.joints.num_dofs,
            model.joints.dofs_offset,
            model.joints.wid,
            model.joints.bid_B,
            model.joints.bid_F,
            data.joints.tau_j,
            jacobians._J_dofs_joint_nzb_offsets,
            jacobians._J_dofs.bsm.nzb_values,
            # Outputs:
            data.bodies.w_a_i,
        ],
    )


def compute_joint_dof_body_wrenches(
    model: ModelKamino,
    data: DataKamino,
    jacobians: DenseSystemJacobians | SparseSystemJacobians,
    reset_to_zero: bool = True,
) -> None:
    """
    Update the actuation wrenches of the bodies based on the active joint torques.
    """
    if isinstance(jacobians, DenseSystemJacobians):
        compute_joint_dof_body_wrenches_dense(model, data, jacobians, reset_to_zero)
    elif isinstance(jacobians, SparseSystemJacobians):
        compute_joint_dof_body_wrenches_sparse(model, data, jacobians, reset_to_zero)
    else:
        raise ValueError(f"Expected `DenseSystemJacobians` or `SparseSystemJacobians` but got {type(jacobians)}.")


def compute_constraint_body_wrenches_dense(
    model: ModelKamino,
    data: DataKamino,
    jacobians: DenseSystemJacobians,
    lambdas_offsets: wp.array,
    lambdas_data: wp.array,
    limits: LimitsKamino | None = None,
    contacts: ContactsKamino | None = None,
    reset_to_zero: bool = True,
):
    """
    Launches the kernels to compute the body-wise constraint wrenches.
    """
    # First check that the Jacobians are dense
    if not isinstance(jacobians, DenseSystemJacobians):
        raise ValueError(f"Expected `DenseSystemJacobians` but got {type(jacobians)}.")

    # Proceed by constraint type, since the Jacobian and lambda data are
    # stored in separate blocks for each constraint type in the dense case
    if model.size.sum_of_num_joints > 0:
        if reset_to_zero:
            data.bodies.w_j_i.zero_()
        wp.launch(
            _compute_joint_cts_body_wrenches_dense,
            dim=model.size.sum_of_num_joints,
            inputs=[
                # Inputs:
                model.info.num_body_dofs,
                model.info.bodies_offset,
                model.info.joint_dynamic_cts_group_offset,
                model.info.joint_kinematic_cts_group_offset,
                model.time.inv_dt,
                model.joints.wid,
                model.joints.num_dynamic_cts,
                model.joints.num_kinematic_cts,
                model.joints.dynamic_cts_offset,
                model.joints.kinematic_cts_offset,
                model.joints.bid_B,
                model.joints.bid_F,
                jacobians.data.J_cts_offsets,
                jacobians.data.J_cts_data,
                lambdas_offsets,
                lambdas_data,
                # Outputs:
                data.bodies.w_j_i,
            ],
        )

    if limits is not None and limits.model_max_limits_host > 0:
        if reset_to_zero:
            data.bodies.w_l_i.zero_()
        wp.launch(
            _compute_limit_cts_body_wrenches_dense,
            dim=limits.model_max_limits_host,
            inputs=[
                # Inputs:
                model.info.num_body_dofs,
                model.info.bodies_offset,
                data.info.limit_cts_group_offset,
                model.time.inv_dt,
                limits.model_active_limits,
                limits.model_max_limits_host,
                limits.wid,
                limits.lid,
                limits.bids,
                jacobians.data.J_cts_offsets,
                jacobians.data.J_cts_data,
                lambdas_offsets,
                lambdas_data,
                # Outputs:
                data.bodies.w_l_i,
            ],
        )

    if contacts is not None and contacts.model_max_contacts_host > 0:
        if reset_to_zero:
            data.bodies.w_c_i.zero_()
        wp.launch(
            _compute_contact_cts_body_wrenches_dense,
            dim=contacts.model_max_contacts_host,
            inputs=[
                # Inputs:
                model.info.num_body_dofs,
                model.info.bodies_offset,
                data.info.contact_cts_group_offset,
                model.time.inv_dt,
                contacts.model_active_contacts,
                contacts.model_max_contacts_host,
                contacts.wid,
                contacts.cid,
                contacts.bid_AB,
                jacobians.data.J_cts_offsets,
                jacobians.data.J_cts_data,
                lambdas_offsets,
                lambdas_data,
                # Outputs:
                data.bodies.w_c_i,
            ],
        )


def compute_constraint_body_wrenches_sparse(
    model: ModelKamino,
    data: DataKamino,
    jacobians: SparseSystemJacobians,
    lambdas_offsets: wp.array,
    lambdas_data: wp.array,
    reset_to_zero: bool = True,
):
    """
    Launches the kernels to compute the body-wise constraint wrenches.
    """
    # First check that the Jacobians are sparse
    if not isinstance(jacobians, SparseSystemJacobians):
        raise ValueError(f"Expected `SparseSystemJacobians` but got {type(jacobians)}.")

    # Optionally clear the previous constraint wrenches, because the kernel computing them
    # uses an `wp.atomic_add` op to accumulate contributions from each constraint non-zero
    # block, and thus assumes the target arrays are zeroed out before each call
    if reset_to_zero:
        data.bodies.w_j_i.zero_()
        data.bodies.w_l_i.zero_()
        data.bodies.w_c_i.zero_()

    # Then compute the body wrenches resulting from the current active constraints
    wp.launch(
        _compute_cts_body_wrenches_sparse,
        dim=(model.size.num_worlds, jacobians._J_cts.bsm.max_of_num_nzb),
        inputs=[
            # Inputs:
            model.time.inv_dt,
            model.info.bodies_offset,
            data.info.limit_cts_group_offset,
            data.info.contact_cts_group_offset,
            jacobians._J_cts.bsm.num_nzb,
            jacobians._J_cts.bsm.nzb_start,
            jacobians._J_cts.bsm.nzb_coords,
            jacobians._J_cts.bsm.nzb_values,
            lambdas_offsets,
            lambdas_data,
            # Outputs:
            data.bodies.w_j_i,
            data.bodies.w_l_i,
            data.bodies.w_c_i,
        ],
    )


def compute_constraint_body_wrenches(
    model: ModelKamino,
    data: DataKamino,
    jacobians: DenseSystemJacobians | SparseSystemJacobians,
    lambdas_offsets: wp.array,
    lambdas_data: wp.array,
    limits: LimitsKamino | None = None,
    contacts: ContactsKamino | None = None,
    reset_to_zero: bool = True,
):
    """
    Launches the kernels to compute the body-wise constraint wrenches.
    """
    if isinstance(jacobians, DenseSystemJacobians):
        compute_constraint_body_wrenches_dense(
            model=model,
            data=data,
            jacobians=jacobians,
            lambdas_offsets=lambdas_offsets,
            lambdas_data=lambdas_data,
            limits=limits,
            contacts=contacts,
            reset_to_zero=reset_to_zero,
        )
    elif isinstance(jacobians, SparseSystemJacobians):
        compute_constraint_body_wrenches_sparse(
            model=model,
            data=data,
            jacobians=jacobians,
            lambdas_offsets=lambdas_offsets,
            lambdas_data=lambdas_data,
            reset_to_zero=reset_to_zero,
        )
    else:
        raise ValueError(f"Expected `DenseSystemJacobians` or `SparseSystemJacobians` but got {type(jacobians)}.")
