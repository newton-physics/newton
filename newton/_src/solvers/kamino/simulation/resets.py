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

"""Provides a set of operations to reset the state of a physics simulation."""

import warp as wp

from ..core.bodies import transform_body_inertial_properties
from ..core.model import Model, ModelData
from ..core.state import State
from ..core.types import float32, int32, mat33f, transformf, vec3f, vec6f
from ..kinematics.joints import compute_joint_pose_and_relative_motion, make_write_joint_data

###
# Module interface
###

__all__ = [
    "reset_select_worlds_to_initial_state",
    "reset_select_worlds_to_state",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###


###
# Kernels
###


@wp.kernel
def _reset_time_of_select_worlds(
    # Inputs:
    mask: wp.array(dtype=int32),
    # Outputs:
    data_time: wp.array(dtype=float32),
    data_steps: wp.array(dtype=int32),
):
    # Retrieve the world index from the 1D thread index
    wid = wp.tid()

    # Retrieve the reset flag for the corresponding world
    world_has_reset = mask[wid]

    # Skip resetting time if the world has not been marked for reset
    if not world_has_reset:
        return

    # Reset both the physical time and step count to zero
    data_time[wid] = 0.0
    data_steps[wid] = 0


@wp.kernel
def _reset_bodies_of_select_worlds(
    # Inputs:
    mask: wp.array(dtype=int32),
    # Inputs:
    model_bid: wp.array(dtype=int32),
    model_i_I_i: wp.array(dtype=mat33f),
    model_inv_i_I_i: wp.array(dtype=mat33f),
    state_q_i: wp.array(dtype=transformf),
    state_u_i: wp.array(dtype=vec6f),
    # Outputs:
    data_q_i: wp.array(dtype=transformf),
    data_u_i: wp.array(dtype=vec6f),
    data_I_i: wp.array(dtype=mat33f),
    data_inv_I_i: wp.array(dtype=mat33f),
    data_w_i: wp.array(dtype=vec6f),
    data_w_a_i: wp.array(dtype=vec6f),
    data_w_j_i: wp.array(dtype=vec6f),
    data_w_l_i: wp.array(dtype=vec6f),
    data_w_c_i: wp.array(dtype=vec6f),
    data_w_e_i: wp.array(dtype=vec6f),
):
    # Retrieve the body index from the 1D thread index
    bid = wp.tid()

    # Retrieve the world index for this body
    wid = model_bid[bid]

    # Retrieve the reset flag for the corresponding world
    world_has_reset = mask[wid]

    # Skip resetting this body if the world has not been marked for reset
    if not world_has_reset:
        return

    # Create a zero-valued vec6 to zero-out wrenches
    zero6 = vec6f(0.0)

    # Retrieve the target state for this body
    q_i_0 = state_q_i[bid]
    u_i_0 = state_u_i[bid]

    # Retrieve the model data for this body
    i_I_i = model_i_I_i[bid]
    inv_i_I_i = model_inv_i_I_i[bid]

    # Compute the moment of inertia matrices in world coordinates
    I_i, inv_I_i = transform_body_inertial_properties(q_i_0, i_I_i, inv_i_I_i)

    # Store the reset state and inertial properties
    # in the output arrays and zero-out wrenches
    data_q_i[bid] = q_i_0
    data_u_i[bid] = u_i_0
    data_I_i[bid] = I_i
    data_inv_I_i[bid] = inv_I_i
    data_w_i[bid] = zero6
    data_w_a_i[bid] = zero6
    data_w_j_i[bid] = zero6
    data_w_l_i[bid] = zero6
    data_w_c_i[bid] = zero6
    data_w_e_i[bid] = zero6


@wp.kernel
def _reset_joints_of_select_worlds(
    # Inputs:
    reset_constraints: bool,
    mask: wp.array(dtype=int32),
    model_info_joint_coords_offset: wp.array(dtype=int32),
    model_info_joint_dofs_offset: wp.array(dtype=int32),
    model_info_joint_cts_offset: wp.array(dtype=int32),
    model_joint_wid: wp.array(dtype=int32),
    model_joint_dof_type: wp.array(dtype=int32),
    model_joint_num_cts: wp.array(dtype=int32),
    model_joint_coords_offset: wp.array(dtype=int32),
    model_joint_dofs_offset: wp.array(dtype=int32),
    model_joint_cts_offset: wp.array(dtype=int32),
    model_joint_bid_B: wp.array(dtype=int32),
    model_joint_bid_F: wp.array(dtype=int32),
    model_joint_B_r_Bj: wp.array(dtype=vec3f),
    model_joint_F_r_Fj: wp.array(dtype=vec3f),
    model_joint_X_j: wp.array(dtype=mat33f),
    model_joint_q_j_ref: wp.array(dtype=float32),
    state_q_i: wp.array(dtype=transformf),
    state_u_i: wp.array(dtype=vec6f),
    state_lambda_j: wp.array(dtype=float32),
    # Outputs:
    data_p_j: wp.array(dtype=transformf),
    data_r_j: wp.array(dtype=float32),
    data_dr_j: wp.array(dtype=float32),
    data_q_j: wp.array(dtype=float32),
    data_dq_j: wp.array(dtype=float32),
    data_lambda_j: wp.array(dtype=float32),
    data_j_w_j: wp.array(dtype=vec6f),
    data_j_w_a_j: wp.array(dtype=vec6f),
    data_j_w_c_j: wp.array(dtype=vec6f),
    data_j_w_l_j: wp.array(dtype=vec6f),
):
    # Retrieve the body index from the 1D thread index
    jid = wp.tid()

    # Retrieve the world index for this body
    wid = model_joint_wid[jid]

    # Retrieve the reset flag for the corresponding world
    world_has_reset = mask[wid]

    # Skip resetting this joint if the world has not been marked for reset
    if not world_has_reset:
        return

    # Retrieve the joint model data
    dof_type = model_joint_dof_type[jid]
    num_cts = model_joint_num_cts[jid]
    coords_offset = model_joint_coords_offset[jid]
    dofs_offset = model_joint_dofs_offset[jid]
    cts_offset = model_joint_cts_offset[jid]
    bid_B = model_joint_bid_B[jid]
    bid_F = model_joint_bid_F[jid]
    B_r_Bj = model_joint_B_r_Bj[jid]
    F_r_Fj = model_joint_F_r_Fj[jid]
    X_j = model_joint_X_j[jid]

    # Retrieve the index offsets of the joint's constraint and DoF dimensions
    world_joint_coords_offset = model_info_joint_coords_offset[wid]
    world_joint_dofs_offset = model_info_joint_dofs_offset[wid]
    world_joint_cts_offset = model_info_joint_cts_offset[wid]

    # If the Base body is the world (bid=-1), use the identity transform (frame
    # of the world's origin), otherwise retrieve the Base body's pose and twist
    T_B_j = wp.transform_identity(dtype=float32)
    u_B_j = vec6f(0.0)
    if bid_B > -1:
        T_B_j = state_q_i[bid_B]
        u_B_j = state_u_i[bid_B]

    # Retrieve the Follower body's pose and twist
    T_F_j = state_q_i[bid_F]
    u_F_j = state_u_i[bid_F]

    # Append the index offsets of the world's joint blocks
    coords_offset += world_joint_coords_offset
    dofs_offset += world_joint_dofs_offset
    cts_offset += world_joint_cts_offset

    # Compute the joint frame pose and relative motion
    p_j, j_r_j, j_q_j, j_u_j = compute_joint_pose_and_relative_motion(T_B_j, T_F_j, u_B_j, u_F_j, B_r_Bj, F_r_Fj, X_j)

    # Store the absolute pose of the joint frame in world coordinates
    data_p_j[jid] = p_j

    # Store the joint constraint residuals and motion
    wp.static(make_write_joint_data())(
        dof_type,
        cts_offset,
        dofs_offset,
        coords_offset,
        j_r_j,
        j_q_j,
        j_u_j,
        model_joint_q_j_ref,
        data_r_j,
        data_dr_j,
        data_q_j,
        data_dq_j,
    )

    # Reset the joint-related wrenches to zero
    zero6 = vec6f(0.0)
    data_j_w_j[jid] = zero6
    data_j_w_a_j[jid] = zero6
    data_j_w_c_j[jid] = zero6
    data_j_w_l_j[jid] = zero6

    # If requested, reset the joint constraint reactions to zero
    if reset_constraints:
        for k in range(num_cts):
            data_lambda_j[cts_offset + k] = 0.0
    # Otherwise, copy the target constraint reactions from the target state
    else:
        for k in range(num_cts):
            data_lambda_j[cts_offset + k] = state_lambda_j[cts_offset + k]


###
# Launchers
###


def reset_select_worlds_to_initial_state(
    model: Model,
    mask: wp.array,
    data: ModelData,
    reset_constraints: bool = True,
):
    """
    Reset the state of the selected worlds to the initial state
    defined in the model given an array of per-world flags.

    Args:
        model: Input model container holding the time-invariant data of the system.
        state: Input state container specifying the target state to be reset to.
        mask: Array of per-world flags indicating which worlds should be reset.
        data: Output solver data to be configured for the target state.
        reset_constraints: Whether to reset joint constraint reactions to zero.
    """
    # Reset time
    wp.launch(
        _reset_time_of_select_worlds,
        dim=model.size.num_worlds,
        inputs=[
            # Inputs:
            mask,
            # Outputs:
            data.time.time,
            data.time.steps,
        ],
    )

    # Reset bodies
    wp.launch(
        _reset_bodies_of_select_worlds,
        dim=model.size.sum_of_num_bodies,
        inputs=[
            # Inputs:
            mask,
            model.bodies.wid,
            model.bodies.i_I_i,
            model.bodies.inv_i_I_i,
            model.bodies.q_i_0,
            model.bodies.u_i_0,
            # Outputs:
            data.bodies.q_i,
            data.bodies.u_i,
            data.bodies.I_i,
            data.bodies.inv_I_i,
            data.bodies.w_i,
            data.bodies.w_a_i,
            data.bodies.w_j_i,
            data.bodies.w_l_i,
            data.bodies.w_c_i,
            data.bodies.w_e_i,
        ],
    )

    # Reset joints
    wp.launch(
        _reset_joints_of_select_worlds,
        dim=model.size.sum_of_num_joints,
        inputs=[
            # Inputs:
            reset_constraints,
            mask,
            model.info.joint_coords_offset,
            model.info.joint_dofs_offset,
            model.info.joint_cts_offset,
            model.joints.wid,
            model.joints.dof_type,
            model.joints.num_cts,
            model.joints.coords_offset,
            model.joints.dofs_offset,
            model.joints.cts_offset,
            model.joints.bid_B,
            model.joints.bid_F,
            model.joints.B_r_Bj,
            model.joints.F_r_Fj,
            model.joints.X_j,
            model.joints.q_j_ref,
            model.bodies.q_i_0,
            model.bodies.u_i_0,
            data.joints.lambda_j,
            # Outputs:
            data.joints.p_j,
            data.joints.r_j,
            data.joints.dr_j,
            data.joints.q_j,
            data.joints.dq_j,
            data.joints.lambda_j,
            data.joints.j_w_j,
            data.joints.j_w_a_j,
            data.joints.j_w_c_j,
            data.joints.j_w_l_j,
        ],
    )


def reset_select_worlds_to_state(
    model: Model,
    state: State,
    mask: wp.array,
    data: ModelData,
    reset_constraints: bool = True,
):
    """
    Reset the state of the selected worlds given an array of per-world flags.

    Args:
        model: Input model container holding the time-invariant data of the system.
        state: Input state container specifying the target state to be reset to.
        mask: Array of per-world flags indicating which worlds should be reset.
        data: Output solver data to be configured for the target state.
    """
    # Reset time
    wp.launch(
        _reset_time_of_select_worlds,
        dim=model.size.num_worlds,
        inputs=[
            # Inputs:
            mask,
            # Outputs:
            data.time.time,
            data.time.steps,
        ],
    )

    # Reset bodies
    wp.launch(
        _reset_bodies_of_select_worlds,
        dim=model.size.sum_of_num_bodies,
        inputs=[
            # Inputs:
            mask,
            model.bodies.wid,
            model.bodies.i_I_i,
            model.bodies.inv_i_I_i,
            state.q_i,
            state.u_i,
            # Outputs:
            data.bodies.q_i,
            data.bodies.u_i,
            data.bodies.I_i,
            data.bodies.inv_I_i,
            data.bodies.w_i,
            data.bodies.w_a_i,
            data.bodies.w_j_i,
            data.bodies.w_l_i,
            data.bodies.w_c_i,
            data.bodies.w_e_i,
        ],
    )

    # Reset joints
    wp.launch(
        _reset_joints_of_select_worlds,
        dim=model.size.sum_of_num_joints,
        inputs=[
            # Inputs:
            reset_constraints,
            mask,
            model.info.joint_coords_offset,
            model.info.joint_dofs_offset,
            model.info.joint_cts_offset,
            model.joints.wid,
            model.joints.dof_type,
            model.joints.num_cts,
            model.joints.coords_offset,
            model.joints.dofs_offset,
            model.joints.cts_offset,
            model.joints.bid_B,
            model.joints.bid_F,
            model.joints.B_r_Bj,
            model.joints.F_r_Fj,
            model.joints.X_j,
            model.joints.q_j_ref,
            state.q_i,
            state.u_i,
            state.lambda_j,
            # Outputs:
            data.joints.p_j,
            data.joints.r_j,
            data.joints.dr_j,
            data.joints.q_j,
            data.joints.dq_j,
            data.joints.lambda_j,
            data.joints.j_w_j,
            data.joints.j_w_a_j,
            data.joints.j_w_c_j,
            data.joints.j_w_l_j,
        ],
    )
