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
KAMINO: Kinematics: Jacobians
"""

from __future__ import annotations

from typing import Any

import warp as wp
from warp.context import Devicelike

from ..core.joints import JointDoFType
from ..core.math import I_6
from ..core.model import Model, ModelData
from ..core.types import (
    float32,
    int32,
    mat33f,
    mat63f,
    mat66f,
    transformf,
    vec2i,
    vec3f,
    vec4f,
)
from ..geometry.contacts import Contacts, ContactsData
from ..kinematics.limits import Limits, LimitsData

###
# Module interface
###

__all__ = [
    "DenseSystemJacobians",
    "DenseSystemJacobiansData",
    "build_contact_jacobians",
    "build_jacobians",
    "build_joint_jacobians",
    "build_limit_jacobians",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###

W_C_I = wp.constant(mat63f(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0))
"""Identify-like wrench matrix to initialize contact wrench matrices."""


###
# Functions
###


def make_store_joint_jacobian_func(axes: Any):
    """
    Generates a warp function to store body-pair Jacobian blocks into a target flat
    data array given a vector of Jacobian row indices (i.e. selection vector).
    """

    @wp.func
    def store_joint_jacobian(
        J_offset: int,
        row_offset: int,
        row_size: int,
        bid_offset: int,
        bid_B: int,
        bid_F: int,
        JT_B_j: mat66f,
        JT_F_j: mat66f,
        J_data: wp.array(dtype=float32),
    ):
        """
        Stores the Jacobian blocks of a joint into the provided flat data array at the specified offset.

        Args:
            J_offset (int): The offset at which the Jacobian matrix block of the corresponding world starts.
            row_offset (int): The offset at which the first target row starts.
            row_size (int): The number of columns in the world's Jacobian block.
            bid_offset (int): The body index offset of the world's bodies w.r.t the model.
            bid_B (int): The body index of the base body of the joint w.r.t the model.
            bid_F (int): The body index of the follower body of the joint w.r.t the model.
            JT_B_j (mat66f): The 6x6 Jacobian transpose block of the joint's base body.
            JT_F_j (mat66f): The 6x6 Jacobian transpose block of the joint's follower body.
            J_data (wp.array(dtype=float32)): The flat data array holding the Jacobian matrix blocks.
        """
        # Set the number of rows in the output Jacobian block
        # NOTE: This is evaluated statically at compile time
        num_jac_rows = wp.static(len(axes))

        # Append the row offset to the Jacobian matrix block offset
        # NOTE: This sets the adjusts the start index to the first target row
        J_offset += row_size * row_offset

        # Store the Jacobian block for the follower body
        J_offset_F = J_offset + 6 * (bid_F - bid_offset)
        for j in range(num_jac_rows):
            kj = J_offset_F + row_size * j
            for i in range(6):
                J_data[kj + i] = JT_F_j[i, axes[j]]

        # If the base body is not the world (:= -1), store the respective Jacobian block
        if bid_B > -1:
            J_offset_B = J_offset + 6 * (bid_B - bid_offset)
            for j in range(num_jac_rows):
                kj = J_offset_B + row_size * j
                for i in range(6):
                    J_data[kj + i] = JT_B_j[i, axes[j]]

    # Return the function
    return store_joint_jacobian


@wp.func
def store_joint_cts_jacobian(
    dof_type: int,
    J_cts_offset: int,
    cts_offset: int,
    num_body_dofs: int,
    bid_offset: int,
    bid_B: int,
    bid_F: int,
    JT_B: mat66f,
    JT_F: mat66f,
    J_cts_data: wp.array(dtype=float32),
):
    """
    Stores the constraints Jacobian block of a joint into the provided flat data array at the given offset.
    """

    if dof_type == JointDoFType.REVOLUTE:
        wp.static(make_store_joint_jacobian_func(JointDoFType.REVOLUTE.cts_axes))(
            J_cts_offset, cts_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_cts_data
        )

    elif dof_type == JointDoFType.PRISMATIC:
        wp.static(make_store_joint_jacobian_func(JointDoFType.PRISMATIC.cts_axes))(
            J_cts_offset, cts_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_cts_data
        )

    elif dof_type == JointDoFType.CYLINDRICAL:
        wp.static(make_store_joint_jacobian_func(JointDoFType.CYLINDRICAL.cts_axes))(
            J_cts_offset, cts_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_cts_data
        )

    elif dof_type == JointDoFType.UNIVERSAL:
        wp.static(make_store_joint_jacobian_func(JointDoFType.UNIVERSAL.cts_axes))(
            J_cts_offset, cts_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_cts_data
        )

    elif dof_type == JointDoFType.SPHERICAL:
        wp.static(make_store_joint_jacobian_func(JointDoFType.SPHERICAL.cts_axes))(
            J_cts_offset, cts_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_cts_data
        )

    elif dof_type == JointDoFType.GIMBAL:
        wp.static(make_store_joint_jacobian_func(JointDoFType.GIMBAL.cts_axes))(
            J_cts_offset, cts_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_cts_data
        )

    elif dof_type == JointDoFType.CARTESIAN:
        wp.static(make_store_joint_jacobian_func(JointDoFType.CARTESIAN.cts_axes))(
            J_cts_offset, cts_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_cts_data
        )

    elif dof_type == JointDoFType.FIXED:
        wp.static(make_store_joint_jacobian_func(JointDoFType.FIXED.cts_axes))(
            J_cts_offset, cts_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_cts_data
        )


@wp.func
def store_joint_dofs_jacobian(
    dof_type: int,
    J_dofs_offset: int,
    dofs_offset: int,
    num_body_dofs: int,
    bid_offset: int,
    bid_B: int,
    bid_F: int,
    JT_B: mat66f,
    JT_F: mat66f,
    J_dofs_data: wp.array(dtype=float32),
):
    """
    Stores the DoFs Jacobian block of a joint into the provided flat data array at the given offset.
    """

    if dof_type == JointDoFType.REVOLUTE:
        wp.static(make_store_joint_jacobian_func(JointDoFType.REVOLUTE.dofs_axes))(
            J_dofs_offset, dofs_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_dofs_data
        )

    elif dof_type == JointDoFType.PRISMATIC:
        wp.static(make_store_joint_jacobian_func(JointDoFType.PRISMATIC.dofs_axes))(
            J_dofs_offset, dofs_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_dofs_data
        )

    elif dof_type == JointDoFType.CYLINDRICAL:
        wp.static(make_store_joint_jacobian_func(JointDoFType.CYLINDRICAL.dofs_axes))(
            J_dofs_offset, dofs_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_dofs_data
        )

    elif dof_type == JointDoFType.UNIVERSAL:
        wp.static(make_store_joint_jacobian_func(JointDoFType.UNIVERSAL.dofs_axes))(
            J_dofs_offset, dofs_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_dofs_data
        )

    elif dof_type == JointDoFType.SPHERICAL:
        wp.static(make_store_joint_jacobian_func(JointDoFType.SPHERICAL.dofs_axes))(
            J_dofs_offset, dofs_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_dofs_data
        )

    elif dof_type == JointDoFType.GIMBAL:
        wp.static(make_store_joint_jacobian_func(JointDoFType.GIMBAL.dofs_axes))(
            J_dofs_offset, dofs_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_dofs_data
        )

    elif dof_type == JointDoFType.CARTESIAN:
        wp.static(make_store_joint_jacobian_func(JointDoFType.CARTESIAN.dofs_axes))(
            J_dofs_offset, dofs_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_dofs_data
        )

    elif dof_type == JointDoFType.FREE:
        wp.static(make_store_joint_jacobian_func(JointDoFType.FREE.dofs_axes))(
            J_dofs_offset, dofs_offset, num_body_dofs, bid_offset, bid_B, bid_F, JT_B, JT_F, J_dofs_data
        )


@wp.func
def wrench_matrix_from_points(r_j: vec3f, r_i: vec3f) -> mat66f:
    """
    Generates a 6x6 wrench matrix from the absolute positions (in world coordinates) of the joint and body.

    W_j = [I_3  , 0_3] , where S_ji is the skew-symmetric matrix of the vector r_ji = r_j - r_i.
          [S_ji , I_3]

    Args:
        r_j (vec3f): Position of the joint in world coordinates.
        r_i (vec3f): Position of the body in world coordinates.

    Returns:
        mat66f: The 6x6 wrench matrix.
    """
    # Initialize the wrench matrix
    W_j = I_6

    # Fill the lower left block with the skew-symmetric matrix
    S_rj = wp.skew(r_j - r_i)
    for i in range(3):
        for j in range(3):
            W_j[3 + i, j] = S_rj[i, j]

    # Return the wrench matrix
    return W_j


@wp.func
def contact_wrench_matrix_from_points(r_k: vec3f, r_i: vec3f) -> mat63f:
    """
    Generates a 6x3 wrench matrix from the absolute positions (in world coordinates) of the joint and body.

    W_ki = [ I_3  ] , where S_ki is the skew-symmetric matrix of the vector r_ki = r_k - r_i.
           [ S_ki ]

    Args:
        r_k (vec3f): Position of the contact on the body in world coordinates.
        r_i (vec3f): Position of the body CoM in world coordinates.

    Returns:
        mat63f: The 6x3 wrench matrix.
    """
    # Initialize the wrench matrix
    W_ki = W_C_I

    # Fill the lower left block with the skew-symmetric matrix
    S_ki = wp.skew(r_k - r_i)
    for i in range(3):
        for j in range(3):
            W_ki[3 + i, j] = S_ki[i, j]

    # Return the wrench matrix
    return W_ki


@wp.func
def expand6d(X: mat33f) -> mat66f:
    """
    Expands a 3x3 rotation matrix to a 6x6 matrix operator by filling
    the upper left and lower right blocks with the input matrix.

    Args:
        X (mat33f): The 3x3 matrix to be expanded.

    Returns:
        mat66: The expanded 6x6 matrix.
    """
    # Initialize the 6D matrix
    X_6d = mat66f(0.0)

    # Fill the upper left 3x3 block with the input matrix
    for i in range(3):
        for j in range(3):
            X_6d[i, j] = X[i, j]
            X_6d[3 + i, 3 + j] = X[i, j]

    # Return the expanded matrix
    return X_6d


###
# Kernels
###


@wp.kernel
def _build_joint_jacobians(
    # Inputs
    model_info_num_body_dofs: wp.array(dtype=int32),
    model_info_bodies_offset: wp.array(dtype=int32),
    model_joints_wid: wp.array(dtype=int32),
    model_joints_dof_type: wp.array(dtype=int32),
    model_joints_cts_offset: wp.array(dtype=int32),
    model_joints_dofs_offset: wp.array(dtype=int32),
    model_joints_bid_B: wp.array(dtype=int32),
    model_joints_bid_F: wp.array(dtype=int32),
    model_joints_X: wp.array(dtype=mat33f),
    state_joints_p: wp.array(dtype=transformf),
    state_bodies_q: wp.array(dtype=transformf),
    jacobian_cts_offsets: wp.array(dtype=int32),
    jacobian_dofs_offsets: wp.array(dtype=int32),
    # Outputs
    jacobian_cts_data: wp.array(dtype=float32),
    jacobian_dofs_data: wp.array(dtype=float32),
):
    """
    A kernel to compute the Jacobians (constraints and actuated DoFs) for the joints in a model.
    """
    # Retrieve the thread index as the joint index
    jid = wp.tid()

    # Retrieve the joint model data
    wid = model_joints_wid[jid]
    dof_type = model_joints_dof_type[jid]
    cio = model_joints_cts_offset[jid]
    dio = model_joints_dofs_offset[jid]
    bid_B = model_joints_bid_B[jid]
    bid_F = model_joints_bid_F[jid]
    X_j = model_joints_X[jid]

    # Retrieve the number of body DoFs for corresponding world
    nbd = model_info_num_body_dofs[wid]
    bio = model_info_bodies_offset[wid]

    # Retrieve the Jacobian block offset for this joint
    cjmio = jacobian_cts_offsets[wid]
    djmio = jacobian_dofs_offsets[wid]

    # Retrieve the pose transform of the joint
    T_j = state_joints_p[jid]
    r_j = wp.transform_get_translation(T_j)
    R_j = wp.quat_to_matrix(wp.transform_get_rotation(T_j))

    # Retrieve the pose transforms of each body
    # NOTE: If the base body is the world (bid=-1), use the identity transform (frame of the world's origin)
    T_B_j = wp.transform_identity()
    if bid_B > -1:
        T_B_j = state_bodies_q[bid_B]
    T_F_j = state_bodies_q[bid_F]
    r_B_j = wp.transform_get_translation(T_B_j)
    r_F_j = wp.transform_get_translation(T_F_j)

    # Compute the wrench matrices
    # TODO: Since the lever-arm is a relative position, can we just use B_r_Bj and F_r_Fj instead?
    W_j_B = wrench_matrix_from_points(r_j, r_B_j)
    W_j_F = wrench_matrix_from_points(r_j, r_F_j)

    # Compute the effective projector to joint frame and expand to 6D
    R_X_j = R_j @ X_j
    R_X_bar_j = expand6d(R_X_j)

    # Compute the extended jacobians, i.e. without the selection-matrix multiplication
    JT_B_j = -W_j_B @ R_X_bar_j  # Reaction is on the Base body body ; (6 x 6)
    JT_F_j = W_j_F @ R_X_bar_j  # Action is on the Follower body    ; (6 x 6)

    # Store the constraint Jacobian block
    store_joint_cts_jacobian(dof_type, cjmio, cio, nbd, bio, bid_B, bid_F, JT_B_j, JT_F_j, jacobian_cts_data)

    # Store the actuation Jacobian block if the joint is actuated
    store_joint_dofs_jacobian(dof_type, djmio, dio, nbd, bio, bid_B, bid_F, JT_B_j, JT_F_j, jacobian_dofs_data)


@wp.kernel
def _build_limit_jacobians(
    # Inputs:
    model_info_num_body_dofs: wp.array(dtype=int32),
    model_info_bodies_offset: wp.array(dtype=int32),
    state_info_limit_cts_group_offset: wp.array(dtype=int32),
    limits_model_num: wp.array(dtype=int32),
    limits_wid: wp.array(dtype=int32),
    limits_lid: wp.array(dtype=int32),
    limits_bids: wp.array(dtype=vec2i),
    limits_dof: wp.array(dtype=int32),
    limits_side: wp.array(dtype=float32),
    jacobian_dofs_offsets: wp.array(dtype=int32),
    jacobian_cts_offsets: wp.array(dtype=int32),
    jacobian_dofs_data: wp.array(dtype=float32),
    # Outputs:
    jacobian_cts_data: wp.array(dtype=float32),
):
    """
    A kernel to compute the Jacobians (constraints and actuated DoFs) for the joints in a model.
    """
    # Retrieve the thread index as the limit index
    lid = wp.tid()

    # Skip if cid is greater than the total number of active limits in the model
    if lid >= limits_model_num[0]:
        return

    # Retrieve the world index of the active limit
    wid_l = limits_wid[lid]

    # Retrieve the limit description info
    # NOTE: *_l is used to denote a subscript for the limit index
    lid_l = limits_lid[lid]
    bids_l = limits_bids[lid]
    dof_l = limits_dof[lid]
    side_l = limits_side[lid]

    # Retrieve the relevant model info of the world
    nbd = model_info_num_body_dofs[wid_l]
    bio = model_info_bodies_offset[wid_l]
    lcgo = state_info_limit_cts_group_offset[wid_l]
    ajmio = jacobian_dofs_offsets[wid_l]
    cjmio = jacobian_cts_offsets[wid_l]

    # Append the index offsets to the corresponding rows of the Jacobians
    ajmio += nbd * dof_l
    cjmio += nbd * (lcgo + lid_l)

    # Extract the body ids
    bid_B_l = bids_l[0]
    bid_F_l = bids_l[1]

    # Set the constraint Jacobian block for the follower body from the actuation Jacobian block
    bio_F = 6 * (bid_F_l - bio)
    act_kj = ajmio + bio_F
    cts_kj = cjmio + bio_F
    for i in range(6):
        jacobian_cts_data[cts_kj + i] = side_l * jacobian_dofs_data[act_kj + i]

    # If not the world body, set the constraint Jacobian block for the base body from the actuation Jacobian block
    if bid_B_l > -1:
        bio_B = 6 * (bid_B_l - bio)
        act_kj = ajmio + bio_B
        cts_kj = cjmio + bio_B
        for i in range(6):
            jacobian_cts_data[cts_kj + i] = side_l * jacobian_dofs_data[act_kj + i]


@wp.kernel
def _build_contact_jacobians(
    # Inputs:
    model_info_num_body_dofs: wp.array(dtype=int32),
    model_info_bodies_offset: wp.array(dtype=int32),
    state_info_contact_cts_group_offset: wp.array(dtype=int32),
    state_bodies_q: wp.array(dtype=transformf),
    contacts_model_num: wp.array(dtype=int32),
    contacts_wid: wp.array(dtype=int32),
    contacts_cid: wp.array(dtype=int32),
    contacts_body_A: wp.array(dtype=vec4f),
    contacts_body_B: wp.array(dtype=vec4f),
    contacts_frames: wp.array(dtype=mat33f),
    jacobian_cts_offsets: wp.array(dtype=int32),
    # Outputs:
    jacobian_cts_data: wp.array(dtype=float32),
):
    """
    A kernel to compute the Jacobians (constraints and actuated DoFs) for the joints in a model.
    """
    # Retrieve the thread index as the contact index
    cid = wp.tid()

    # Skip if cid is greater than the total number of active contacts in the model
    if cid >= contacts_model_num[0]:
        return

    # Retrieve the contact index w.r.t the world
    # NOTE: k denotes a notational subscript for the
    # contact index, i.e. C_k is the k-th contact entity
    cid_k = contacts_cid[cid]

    # Retrieve the the contact frame and body contact points
    R_k = contacts_frames[cid]
    body_A_k = contacts_body_A[cid]
    body_B_k = contacts_body_B[cid]

    # Retrieve the world index of the contact
    wid = contacts_wid[cid]

    # Retrieve the relevant model info for the world
    nbd = model_info_num_body_dofs[wid]
    bio = model_info_bodies_offset[wid]
    ccgo = state_info_contact_cts_group_offset[wid]
    cjmio = jacobian_cts_offsets[wid]

    # Append the index offset for the contact Jacobian block in the constraint Jacobian
    cjmio += ccgo * nbd

    # Extract the body ids
    bid_A_k = int32(body_A_k[3])
    bid_B_k = int32(body_B_k[3])

    # Extract the contact points on each body geom
    r_Ac_k = vec3f(body_A_k[0], body_A_k[1], body_A_k[2])
    r_Bc_k = vec3f(body_B_k[0], body_B_k[1], body_B_k[2])

    # Set the constraint index offset for this contact
    cio_k = 3 * cid_k

    # Compute and store the revolute Jacobian block for the follower body (subject of action)
    r_B_k = wp.transform_get_translation(state_bodies_q[bid_B_k])
    W_B_k = contact_wrench_matrix_from_points(r_Bc_k, r_B_k)
    JT_c_B_k = W_B_k @ R_k  # Action is on the follower body (B)  ; (6 x 3)
    bio_B = 6 * (bid_B_k - bio)
    for j in range(3):
        kj = cjmio + nbd * (cio_k + j) + bio_B
        for i in range(6):
            jacobian_cts_data[kj + i] = JT_c_B_k[i, j]

    # If not the world body, compute and store the revolute Jacobian block for the base body (subject of reaction)
    if bid_A_k > -1:
        r_A_k = wp.transform_get_translation(state_bodies_q[bid_A_k])
        W_A_k = contact_wrench_matrix_from_points(r_Ac_k, r_A_k)
        JT_c_A_k = -W_A_k @ R_k  # Reaction is on the base body (A)    ; (6 x 3)
        bio_A = 6 * (bid_A_k - bio)
        for j in range(3):
            kj = cjmio + nbd * (cio_k + j) + bio_A
            for i in range(6):
                jacobian_cts_data[kj + i] = JT_c_A_k[i, j]


###
# Launchers
###


def build_joint_jacobians(
    model: Model,
    data: ModelData,
    jacobian_cts_offsets: wp.array,
    jacobian_cts_data: wp.array,
    jacobian_dofs_offsets: wp.array,
    jacobian_dofs_data: wp.array,
    reset_to_zero: bool = False,
):
    # Optionally reset the Jacobian arrays to zero
    if reset_to_zero:
        jacobian_cts_data.zero_()
        jacobian_dofs_data.zero_()

    # Launch the kernel to build the joint Jacobians
    wp.launch(
        _build_joint_jacobians,
        dim=model.size.sum_of_num_joints,
        inputs=[
            # Inputs:
            model.info.num_body_dofs,
            model.info.bodies_offset,
            model.joints.wid,
            model.joints.dof_type,
            model.joints.cts_offset,
            model.joints.dofs_offset,
            model.joints.bid_B,
            model.joints.bid_F,
            model.joints.X_j,
            data.joints.p_j,
            data.bodies.q_i,
            jacobian_cts_offsets,
            jacobian_dofs_offsets,
            # Outputs:
            jacobian_cts_data,
            jacobian_dofs_data,
        ],
    )


def build_limit_jacobians(
    model: Model,
    data: ModelData,
    limits: LimitsData,
    jacobian_cts_offsets: wp.array,
    jacobian_cts_data: wp.array,
    jacobian_dofs_offsets: wp.array,
    jacobian_dofs_data: wp.array,
    reset_to_zero: bool = False,
):
    # Optionally reset the Jacobian array data to zero
    if reset_to_zero:
        jacobian_cts_data.zero_()

    # Build the limit constraints Jacobians
    wp.launch(
        _build_limit_jacobians,
        dim=limits.num_model_max_limits,
        inputs=[
            # Inputs:
            model.info.num_body_dofs,
            model.info.bodies_offset,
            data.info.limit_cts_group_offset,
            limits.model_num_limits,
            limits.wid,
            limits.lid,
            limits.bids,
            limits.dof,
            limits.side,
            jacobian_dofs_offsets,
            jacobian_dofs_data,
            jacobian_cts_offsets,
            # Outputs:
            jacobian_cts_data,
        ],
    )


def build_contact_jacobians(
    model: Model,
    data: ModelData,
    contacts: ContactsData,
    jacobian_cts_offsets: wp.array,
    jacobian_cts_data: wp.array,
    reset_to_zero: bool = False,
):
    # Optionally reset the Jacobian array data to zero
    if reset_to_zero:
        jacobian_cts_data.zero_()

    # Build the contact constraints Jacobians
    wp.launch(
        _build_contact_jacobians,
        dim=contacts.num_model_max_contacts,
        inputs=[
            # Inputs:
            model.info.num_body_dofs,
            model.info.bodies_offset,
            data.info.contact_cts_group_offset,
            data.bodies.q_i,
            contacts.model_num_contacts,
            contacts.wid,
            contacts.cid,
            contacts.body_A,
            contacts.body_B,
            contacts.frame,
            jacobian_cts_offsets,
            # Outputs:
            jacobian_cts_data,
        ],
    )


def build_jacobians(
    model: Model,
    data: ModelData,
    limits: LimitsData | None,
    contacts: ContactsData | None,
    jacobian_cts_offsets: wp.array,
    jacobian_cts_data: wp.array,
    jacobian_dofs_offsets: wp.array,
    jacobian_dofs_data: wp.array,
    reset_to_zero: bool = True,
):
    # Optionally reset the Jacobian array data to zero
    if reset_to_zero:
        jacobian_cts_data.zero_()
        jacobian_dofs_data.zero_()

    # Build the joint constraints and actuation Jacobians
    wp.launch(
        _build_joint_jacobians,
        dim=model.size.sum_of_num_joints,
        inputs=[
            # Inputs:
            model.info.num_body_dofs,
            model.info.bodies_offset,
            model.joints.wid,
            model.joints.dof_type,
            model.joints.cts_offset,
            model.joints.dofs_offset,
            model.joints.bid_B,
            model.joints.bid_F,
            model.joints.X_j,
            data.joints.p_j,
            data.bodies.q_i,
            jacobian_cts_offsets,
            jacobian_dofs_offsets,
            # Outputs:
            jacobian_cts_data,
            jacobian_dofs_data,
        ],
    )

    # Build the limit constraints Jacobians if a limits data container is provided
    if limits is not None:
        if not isinstance(limits, LimitsData):
            raise TypeError(f"`limits` is required to be of type `LimitsData` but got {type(limits)}.")
        wp.launch(
            _build_limit_jacobians,
            dim=limits.num_model_max_limits,
            inputs=[
                # Inputs:
                model.info.num_body_dofs,
                model.info.bodies_offset,
                data.info.limit_cts_group_offset,
                limits.model_num_limits,
                limits.wid,
                limits.lid,
                limits.bids,
                limits.dof,
                limits.side,
                jacobian_dofs_offsets,
                jacobian_cts_offsets,
                jacobian_dofs_data,
                # Outputs:
                jacobian_cts_data,
            ],
        )

    # Build the contact constraints Jacobians if a contacts data container is provided
    if contacts is not None:
        if not isinstance(contacts, ContactsData):
            raise TypeError(f"`contacts` is required to be of type `ContactsData` but got {type(contacts)}.")
        wp.launch(
            _build_contact_jacobians,
            dim=contacts.num_model_max_contacts,
            inputs=[
                # Inputs:
                model.info.num_body_dofs,
                model.info.bodies_offset,
                data.info.contact_cts_group_offset,
                data.bodies.q_i,
                contacts.model_num_contacts,
                contacts.wid,
                contacts.cid,
                contacts.body_A,
                contacts.body_B,
                contacts.frame,
                jacobian_cts_offsets,
                # Outputs:
                jacobian_cts_data,
            ],
        )


###
# Dense System Jacobians
###


class DenseSystemJacobiansData:
    """
    Container to hold time-varying Jacobians of the system.
    """

    def __init__(self):
        ###
        # Constraint Jacobian
        ###

        self.J_cts_offsets: wp.array(dtype=int32) | None = None
        """
        The index offset of the constraint Jacobian matrix block of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.\n
        """

        self.J_cts_data: wp.array(dtype=float32) | None = None
        """
        A flat array containing the joint constraint Jacobian matrix data of all worlds.\n
        Shape of ``(sum(ncts_w * nbd_w),)`` and type :class:`float32`.
        """

        ###
        # DoFs Jacobian
        ###

        self.J_dofs_offsets: wp.array(dtype=int32) | None = None
        """
        The index offset of the DoF Jacobian matrix block of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.\n
        """

        self.J_dofs_data: wp.array(dtype=float32) | None = None
        """
        A flat array containing the joint DoF Jacobian matrix data of all worlds.\n
        Shape of ``(sum(njad_w * nbd_w),)`` and type :class:`float32`.
        """


class DenseSystemJacobians:
    """
    Container to hold time-varying Jacobians of the system.
    """

    def __init__(
        self,
        model: Model | None = None,
        limits: Limits | None = None,
        contacts: Contacts | None = None,
        device: Devicelike = None,
    ):
        # Declare and initialize the Jacobian data container
        self._data = DenseSystemJacobiansData()

        # If a model is provided, allocate the Jacobians data
        if model is not None:
            self.allocate(model=model, limits=limits, contacts=contacts, device=device)

    @property
    def data(self) -> DenseSystemJacobiansData:
        """
        Returns the internal data container holding the Jacobians data.
        """
        return self._data

    def allocate(
        self, model: Model, limits: Limits | None = None, contacts: Contacts | None = None, device: Devicelike = None
    ):
        # Ensure the model container is valid
        if model is None:
            raise ValueError("`model` is required but got `None`.")
        else:
            if not isinstance(model, Model):
                raise TypeError(f"`model` is required to be of type `Model` but got {type(model)}.")

        # Ensure the limits container is valid
        if limits is not None:
            if not isinstance(limits, Limits):
                raise TypeError(f"`limits` is required to be of type `Limits` but got {type(limits)}.")

        # Ensure the contacts container is valid
        if contacts is not None:
            if not isinstance(contacts, Contacts):
                raise TypeError(f"`contacts` is required to be of type `Contacts` but got {type(contacts)}.")

        # Extract the constraint and DoF sizes of each world
        nw = model.info.num_worlds
        nbd = [model.worlds[w].num_body_dofs for w in range(nw)]
        njc = [model.worlds[w].num_joint_cts for w in range(nw)]
        njd = [model.worlds[w].num_joint_dofs for w in range(nw)]
        maxnl = limits.num_world_max_limits if limits is not None else [0] * nw
        maxnc = contacts.num_world_max_contacts if contacts is not None else [0] * nw
        maxncts = [njc[w] + maxnl[w] + 3 * maxnc[w] for w in range(nw)]

        # Compute the sizes of the Jacobian matrix data for each world
        J_cts_sizes = [maxncts[i] * nbd[i] for i in range(nw)]
        J_dofs_sizes = [njd[i] * nbd[i] for i in range(nw)]

        # Compute the total size of the Jacobian matrix data
        total_J_cts_size = sum(J_cts_sizes)
        total_J_dofs_size = sum(J_dofs_sizes)

        # Compute matrix index offsets of each Jacobian block
        J_cts_offsets = [0] * nw
        J_dofs_offsets = [0] * nw
        for w in range(1, nw):
            J_cts_offsets[w] = J_cts_offsets[w - 1] + J_cts_sizes[w - 1]
            J_dofs_offsets[w] = J_dofs_offsets[w - 1] + J_dofs_sizes[w - 1]

        # Allocate the Jacobian arrays
        with wp.ScopedDevice(device):
            self._data.J_cts_offsets = wp.array(J_cts_offsets, dtype=int32)
            self._data.J_dofs_offsets = wp.array(J_dofs_offsets, dtype=int32)
            self._data.J_cts_data = wp.zeros(shape=(total_J_cts_size,), dtype=float32)
            self._data.J_dofs_data = wp.zeros(shape=(total_J_dofs_size,), dtype=float32)

    def build(
        self, model: Model, data: ModelData, limits: LimitsData, contacts: ContactsData, reset_to_zero: bool = True
    ):
        """
        Builds the system DoF and constraint Jacobians for the given
        data of the provided model, data, limits and contacts containers.
        """
        build_jacobians(
            model=model,
            data=data,
            limits=limits,
            contacts=contacts,
            jacobian_cts_offsets=self._data.J_cts_offsets,
            jacobian_cts_data=self._data.J_cts_data,
            jacobian_dofs_offsets=self._data.J_dofs_offsets,
            jacobian_dofs_data=self._data.J_dofs_data,
            reset_to_zero=reset_to_zero,
        )
