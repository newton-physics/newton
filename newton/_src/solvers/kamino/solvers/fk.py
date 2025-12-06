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
KAMINO: Solvers: Forward Kinematics
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import cache

import numpy as np
import warp as wp
from warp.context import Devicelike

from ..core.joints import JointActuationType, JointDoFType
from ..core.math import (
    G_of,
    quat_left_jacobian_inverse,
    quat_log,
    unit_quat_apply,
    unit_quat_apply_jacobian,
    unit_quat_conj_apply,
    unit_quat_conj_apply_jacobian,
    unit_quat_conj_to_rotation_matrix,
)
from ..core.model import Model
from ..core.types import vec6f
from ..linalg.factorize.llt_blocked_semi_sparse import SemiSparseBlockCholeskySolverBatched

###
# Module interface
###

__all__ = ["ForwardKinematicsSolver", "ForwardKinematicsSolverSettings", "ForwardKinematicsSolverStatus"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###


@wp.func
def read_quat_from_array(array: wp.array(dtype=wp.float32), offset: int) -> wp.quatf:
    """
    Utility function to read a quaternion from a flat array
    """
    return wp.quatf(array[offset], array[offset + 1], array[offset + 2], array[offset + 3])


###
# Kernels
###


@wp.kernel
def _reset_state(
    # Inputs
    num_bodies: wp.array(dtype=wp.int32),
    first_body_id: wp.array(dtype=wp.int32),
    bodies_q_0_flat: wp.array(dtype=wp.float32),
    world_mask: wp.array(dtype=wp.int32),
    # Outputs
    bodies_q_flat: wp.array(dtype=wp.float32),
):
    """
    A kernel resetting the fk state (body poses) to the reference state

    Inputs:
        num_bodies: Num bodies per world
        first_body_id: First body id per world
        bodies_q_0_flat: Reference state, flattened
        world_mask: Per-world flag to perform the operation (0 = skip)
    Outputs:
        bodies_q_flat: State to reset, flattened
    """
    wd_id, state_id_loc = wp.tid()  # Thread indices (= world index, state index)
    rb_id_loc = state_id_loc // 7
    if wd_id < num_bodies.shape[0] and world_mask[wd_id] != 0 and rb_id_loc < num_bodies[wd_id]:
        state_id_tot = 7 * first_body_id[wd_id] + state_id_loc
        bodies_q_flat[state_id_tot] = bodies_q_0_flat[state_id_tot]


@wp.kernel
def _eval_fk_actuated_dofs_or_coords(
    # Inputs
    model_base_dofs: wp.array(dtype=wp.float32),
    model_actuated_dofs: wp.array(dtype=wp.float32),
    actuated_dofs_map: wp.array(dtype=wp.int32),
    # Outputs
    fk_actuated_dofs: wp.array(dtype=wp.float32),
):
    """
    A kernel mapping actuated and base dofs/coordinates of the main model to actuated dofs/coordinates of the fk model,
    which has a modified version of the joints, notably actuated free joints to control floating bases.

    This uses a map from fk to model dofs/cords, that has >= 0 indices for fk dofs/coords that correspond to
    main model actuated dofs/coords, and negative indices for base dofs/coords (base dof/coord i is stored as -i - 1)

    Inputs:
        model_base_dofs: Base dofs or coordinates of the main model (as a flat vector with 6 dofs or 7 coordinates per world)
        model_actuated_dofs: Actuated dofs/coords of the main model
        actuated_dofs_map: Map of fk to main model actuated/base dofs/coords
    Outputs:
        fk_actuated_dofs: Actuated dofs or coordinates of the fk model
    """

    # Retrieve the thread index (= fk actuated dof or coordinate index)
    # Note: we use "dof" in variables naming to mean either dof or coordinate
    fk_dof_id = wp.tid()

    if fk_dof_id < fk_actuated_dofs.shape[0]:
        model_dof_id = actuated_dofs_map[fk_dof_id]
        if model_dof_id >= 0:
            fk_actuated_dofs[fk_dof_id] = model_actuated_dofs[model_dof_id]
        else:  # Base dofs/coordinates are encoded as negative indices
            base_dof_id = -(model_dof_id + 1)  # Recover base dof/coord id
            fk_actuated_dofs[fk_dof_id] = model_base_dofs[base_dof_id]


@wp.kernel
def _eval_position_control_transformations(
    # Inputs
    joints_dof_type: wp.array(dtype=wp.int32),
    joints_act_type: wp.array(dtype=wp.int32),
    actuated_coords_offset: wp.array(dtype=wp.int32),
    joints_X: wp.array(dtype=wp.mat33f),
    actuators_q: wp.array(dtype=wp.float32),
    # Outputs
    pos_control_transforms: wp.array(dtype=wp.transformf),
):
    """
    A kernel computing a transformation per joint corresponding to position-control parameters
    More specifically, this is the identity (no translation, no rotation) for passive joints
    and a transformation corresponding to joint generalized coordinates for actuators

    The translation part is expressed in joint frame (e.g., translation is along [1,0,0] for a prismatic joint)
    The rotation part is expressed in body frame (e.g., rotation is about X[:,0] for a revolute joint)

    Inputs:
        joints_dof_type: Joint dof type (i.e. revolute, spherical, ...)
        joints_act_type: Joint actuation type (i.e. passive or actuated)
        actuated_coords_offset: Joint first actuated coordinate id, among all actuated coordinates in all worlds
        joints_X: Joint frame (local axes, valid both on base and follower)
        actuators_q: Actuated coordinates
    Outputs:
        pos_control_transforms: Joint position-control transformation
    """

    # Retrieve the thread index (= joint index)
    jt_id = wp.tid()

    if jt_id < joints_dof_type.shape[0]:
        # Retrieve the joint model data
        dof_type_j = joints_dof_type[jt_id]
        act_type_j = joints_act_type[jt_id]
        X = joints_X[jt_id]

        # Initialize transform to identity (already covers the passive case)
        t = wp.vec3f(0.0, 0.0, 0.0)
        q = wp.quatf(0.0, 0.0, 0.0, 1.0)

        # In the actuated case, set translation/rotation as per joint generalized coordinates
        if act_type_j == JointActuationType.FORCE:
            offset_q_j = actuated_coords_offset[jt_id]
            if dof_type_j == JointDoFType.CARTESIAN:
                t[0] = actuators_q[offset_q_j]
                t[1] = actuators_q[offset_q_j + 1]
                t[2] = actuators_q[offset_q_j + 2]
            elif dof_type_j == JointDoFType.CYLINDRICAL:
                t[0] = actuators_q[offset_q_j]
                q = wp.quat_from_axis_angle(wp.vec3f(X[0, 0], X[1, 0], X[2, 0]), actuators_q[offset_q_j + 1])
            elif dof_type_j == JointDoFType.FIXED:
                pass  # No dofs to apply
            elif dof_type_j == JointDoFType.FREE:
                t[0] = actuators_q[offset_q_j]
                t[1] = actuators_q[offset_q_j + 1]
                t[2] = actuators_q[offset_q_j + 2]
                q_X = wp.quat_from_matrix(X)
                q_loc = read_quat_from_array(actuators_q, offset_q_j + 3)
                q = q_X * q_loc * wp.quat_inverse(q_X)
            elif dof_type_j == JointDoFType.PRISMATIC:
                t[0] = actuators_q[offset_q_j]
            elif dof_type_j == JointDoFType.REVOLUTE:
                q = wp.quat_from_axis_angle(wp.vec3f(X[0, 0], X[1, 0], X[2, 0]), actuators_q[offset_q_j])
            elif dof_type_j == JointDoFType.SPHERICAL:
                q_X = wp.quat_from_matrix(X)
                q_loc = read_quat_from_array(actuators_q, offset_q_j)
                q = q_X * q_loc * wp.quat_inverse(q_X)
            elif dof_type_j == JointDoFType.UNIVERSAL:
                q_x = wp.quat_from_axis_angle(wp.vec3f(X[0, 0], X[1, 0], X[2, 0]), actuators_q[offset_q_j])
                q_y = wp.quat_from_axis_angle(wp.vec3f(X[0, 1], X[1, 1], X[2, 1]), actuators_q[offset_q_j + 1])
                q = q_x * q_y

        # Write out transformation
        pos_control_transforms[jt_id] = wp.transformf(t, q)


@wp.kernel
def _eval_unit_quaternion_constraints(
    # Inputs
    num_bodies: wp.array(dtype=wp.int32),
    first_body_id: wp.array(dtype=wp.int32),
    bodies_q: wp.array(dtype=wp.transformf),
    world_mask: wp.array(dtype=wp.int32),
    # Outputs
    constraints: wp.array2d(dtype=wp.float32),
):
    """
        A kernel computing unit norm quaternion constraints for each body, written at the top of the constraints vector

        Inputs:
            num_bodies: Num bodies per world
            first_body_id: First body id per world
            bodies_q: Body poses
            world_mask: Per-world flag to perform the computation (0 = skip)
        Outputs:
            constraints: Constraint vector per world
    ):
    """

    # Retrieve the thread indices (= world index, body index)
    wd_id, rb_id_loc = wp.tid()

    if wd_id < num_bodies.shape[0] and world_mask[wd_id] != 0 and rb_id_loc < num_bodies[wd_id]:
        # Get overall body id
        rb_id_tot = first_body_id[wd_id] + rb_id_loc

        # Evaluate unit quaternion constraint
        q = wp.transform_get_rotation(bodies_q[rb_id_tot])
        constraints[wd_id, rb_id_loc] = wp.dot(q, q) - 1.0


@cache
def create_eval_joint_constraints_kernel(has_universal_joints: bool):
    """
    Returns the joint constraints evaluation kernel, statically baking in whether there are universal joints
    or not (these joints need a separate handling)
    """

    @wp.kernel
    def _eval_joint_constraints(
        # Inputs
        num_joints: wp.array(dtype=wp.int32),
        first_joint_id: wp.array(dtype=wp.int32),
        joints_dof_type: wp.array(dtype=wp.int32),
        joints_act_type: wp.array(dtype=wp.int32),
        joints_bid_B: wp.array(dtype=wp.int32),
        joints_bid_F: wp.array(dtype=wp.int32),
        joints_X: wp.array(dtype=wp.mat33f),
        joints_B_r_B: wp.array(dtype=wp.vec3f),
        joints_F_r_F: wp.array(dtype=wp.vec3f),
        bodies_q: wp.array(dtype=wp.transformf),
        pos_control_transforms: wp.array(dtype=wp.transformf),
        ct_full_to_red_map: wp.array2d(dtype=wp.int32),
        world_mask: wp.array(dtype=wp.int32),
        # Outputs
        constraints: wp.array2d(dtype=wp.float32),
    ):
        """
        A kernel computing joint constraints with the log map formulation, first computing 6 constraints per
        joint (treating it as a fixed joint), then writing out the relevant subset of constraints (only along
        relevant directions) using a precomputed full to reduced map.

        Note: the log map formulation doesn't allow to formulate passive universal joints. If such joints are
        present, the right number of (incorrect) constraints is first written with the log map, then the result
        is overwritten in a second pass with the correct constraints.

        Inputs:
            num_joints: Num joints per world
            first_joint_id: First joint id per world
            joints_dof_type: Joint dof type (i.e. revolute, spherical, ...)
            joints_act_type: Joint actuation type (i.e. passive or actuated)
            joints_bid_B: Joint base body id
            joints_bid_F: Joint follower body id
            joints_X: Joint frame (local axes, valid both on base and follower)
            joints_B_r_B: Joint local position on base body
            joints_F_r_F: Joint local position on follower body
            bodies_q: Body poses
            pos_control_transforms: Joint position-control transformation
            ct_full_to_red_map: Map from full to reduced constraint id (per world)
            world_mask: Per-world flag to perform the computation (0 = skip)
        Outputs:
            constraints: Constraint vector per world
        """

        # Retrieve the thread indices (= world index, joint index)
        wd_id, jt_id_loc = wp.tid()

        if wd_id < num_joints.shape[0] and world_mask[wd_id] != 0 and jt_id_loc < num_joints[wd_id]:
            # Get overall joint id
            jt_id_tot = first_joint_id[wd_id] + jt_id_loc

            # Get reduced constraint ids (-1 meaning constraint is not used)
            first_ct_id_full = 6 * jt_id_loc
            trans_ct_ids_red = wp.vec3i(
                ct_full_to_red_map[wd_id, first_ct_id_full],
                ct_full_to_red_map[wd_id, first_ct_id_full + 1],
                ct_full_to_red_map[wd_id, first_ct_id_full + 2],
            )
            rot_ct_ids_red = wp.vec3i(
                ct_full_to_red_map[wd_id, first_ct_id_full + 3],
                ct_full_to_red_map[wd_id, first_ct_id_full + 4],
                ct_full_to_red_map[wd_id, first_ct_id_full + 5],
            )

            # Get joint local positions and orientation
            x_base = joints_B_r_B[jt_id_tot]
            x_follower = joints_F_r_F[jt_id_tot]
            X_T = wp.transpose(joints_X[jt_id_tot])

            # Get base and follower transformations
            base_id = joints_bid_B[jt_id_tot]
            if base_id < 0:
                c_base = wp.vec3f(0.0, 0.0, 0.0)
                q_base = wp.quatf(0.0, 0.0, 0.0, 1.0)
            else:
                c_base = wp.transform_get_translation(bodies_q[base_id])
                q_base = wp.transform_get_rotation(bodies_q[base_id])
            follower_id = joints_bid_F[jt_id_tot]
            c_follower = wp.transform_get_translation(bodies_q[follower_id])
            q_follower = wp.transform_get_rotation(bodies_q[follower_id])

            # Get position control transformation, in joint/body frame for translation/rotation part
            t_control_joint = wp.transform_get_translation(pos_control_transforms[jt_id_tot])
            q_control_body = wp.transform_get_rotation(pos_control_transforms[jt_id_tot])

            # Translation constraints: compute "error" translation, in joint frame
            pos_follower_world = unit_quat_apply(q_follower, x_follower) + c_follower
            pos_follower_base = unit_quat_conj_apply(q_base, pos_follower_world - c_base)
            pos_rel_base = (
                pos_follower_base - x_base
            )  # Relative position on base body (should match translation from controls)
            t_error = X_T * pos_rel_base - t_control_joint  # Error in joint frame

            # Rotation constraints: compute "error" rotation with the log map, in joint frame
            q_error_base = wp.quat_inverse(q_base) * q_follower * wp.quat_inverse(q_control_body)
            rot_error = X_T * quat_log(q_error_base)

            # Write out constraint
            for i in range(3):
                if trans_ct_ids_red[i] >= 0:
                    constraints[wd_id, trans_ct_ids_red[i]] = t_error[i]
                if rot_ct_ids_red[i] >= 0:
                    constraints[wd_id, rot_ct_ids_red[i]] = rot_error[i]

            # Correct constraints for passive universal joints
            if wp.static(has_universal_joints):
                # Check for a passive universal joint
                dof_type_j = joints_dof_type[jt_id_tot]
                act_type_j = joints_act_type[jt_id_tot]
                if dof_type_j != int(JointDoFType.UNIVERSAL) or act_type_j != int(JointActuationType.PASSIVE):
                    return

                # Compute constraint (dot product between x axis on base and y axis on follower)
                a_x = X_T[0]
                a_y = X_T[1]
                a_x_base = unit_quat_apply(q_base, a_x)
                a_y_follower = unit_quat_apply(q_follower, a_y)
                ct = -wp.dot(a_x_base, a_y_follower)

                # Set constraint in output (at a location corresponding to z rotational constraint)
                constraints[wd_id, rot_ct_ids_red[2]] = ct

    return _eval_joint_constraints


@wp.kernel
def _eval_unit_quaternion_constraints_jacobian(
    # Inputs
    num_bodies: wp.array(dtype=wp.int32),
    first_body_id: wp.array(dtype=wp.int32),
    bodies_q: wp.array(dtype=wp.transformf),
    world_mask: wp.array(dtype=wp.int32),
    # Outputs
    constraints_jacobian: wp.array3d(dtype=wp.float32),
):
    """
    A kernel computing the Jacobian of unit norm quaternion constraints for each body, written at the top of the
    constraints Jacobian

    Inputs:
        num_bodies: Num bodies per world
        first_body_id: First body id per world
        bodies_q: Body poses
        world_mask: Per-world flag to perform the computation (0 = skip)
    Outputs:
        constraints_jacobian: Constraints Jacobian per world
    """

    # Retrieve the thread indices (= world index, body index)
    wd_id, rb_id_loc = wp.tid()

    if wd_id < num_bodies.shape[0] and world_mask[wd_id] != 0 and rb_id_loc < num_bodies[wd_id]:
        # Get overall body id
        rb_id_tot = first_body_id[wd_id] + rb_id_loc

        # Evaluate constraint Jacobian
        q = wp.transform_get_rotation(bodies_q[rb_id_tot])
        state_offset = 7 * rb_id_loc + 3
        constraints_jacobian[wd_id, rb_id_loc, state_offset] = 2.0 * q.x
        constraints_jacobian[wd_id, rb_id_loc, state_offset + 1] = 2.0 * q.y
        constraints_jacobian[wd_id, rb_id_loc, state_offset + 2] = 2.0 * q.z
        constraints_jacobian[wd_id, rb_id_loc, state_offset + 3] = 2.0 * q.w


@cache
def create_eval_joint_constraints_jacobian_kernel(has_universal_joints: bool):
    """
    Returns the joint constraints Jacobian evaluation kernel, statically baking in whether there are universal joints
    or not (these joints need a separate handling)
    """

    @wp.kernel
    def _eval_joint_constraints_jacobian(
        # Inputs
        num_joints: wp.array(dtype=wp.int32),
        first_joint_id: wp.array(dtype=wp.int32),
        first_body_id: wp.array(dtype=wp.int32),
        joints_dof_type: wp.array(dtype=wp.int32),
        joints_act_type: wp.array(dtype=wp.int32),
        joints_bid_B: wp.array(dtype=wp.int32),
        joints_bid_F: wp.array(dtype=wp.int32),
        joints_X: wp.array(dtype=wp.mat33f),
        joints_B_r_B: wp.array(dtype=wp.vec3f),
        joints_F_r_F: wp.array(dtype=wp.vec3f),
        bodies_q: wp.array(dtype=wp.transformf),
        pos_control_transforms: wp.array(dtype=wp.transformf),
        ct_full_to_red_map: wp.array2d(dtype=wp.int32),
        world_mask: wp.array(dtype=wp.int32),
        # Outputs
        constraints_jacobian: wp.array3d(dtype=wp.float32),
    ):
        """
        A kernel computing the Jacobian of the joint constraints.
        The Jacobian is assumed to have already been filled with zeros, at least in the coefficients that
        are always zero due to joint connectivity.

        Inputs:
            num_joints: Num joints per world
            first_joint_id: First joint id per world
            first_body_id: First body id per world
            joints_dof_type: Joint dof type (i.e. revolute, spherical, ...)
            joints_act_type: Joint actuation type (i.e. passive or actuated)
            joints_bid_B: Joint base body id
            joints_bid_F: Joint follower body id
            joints_X: Joint frame (local axes, valid both on base and follower)
            joints_B_r_B: Joint local position on base body
            joints_F_r_F: Joint local position on follower body
            bodies_q: Body poses
            pos_control_transforms: Joint position-control transformation
            ct_full_to_red_map: Map from full to reduced constraint id (per world)
            world_mask: Per-world flag to perform the computation (0 = skip)
        Outputs:
            constraints_jacobian: Constraint Jacobian per world
        """

        # Retrieve the thread indices (= world index, joint index)
        wd_id, jt_id_loc = wp.tid()

        if wd_id < num_joints.shape[0] and world_mask[wd_id] != 0 and jt_id_loc < num_joints[wd_id]:
            # Get overall joint id
            jt_id_tot = first_joint_id[wd_id] + jt_id_loc

            # Get reduced constraint ids (-1 meaning constraint is not used)
            first_ct_id_full = 6 * jt_id_loc
            trans_ct_ids_red = wp.vec3i(
                ct_full_to_red_map[wd_id, first_ct_id_full],
                ct_full_to_red_map[wd_id, first_ct_id_full + 1],
                ct_full_to_red_map[wd_id, first_ct_id_full + 2],
            )
            rot_ct_ids_red = wp.vec3i(
                ct_full_to_red_map[wd_id, first_ct_id_full + 3],
                ct_full_to_red_map[wd_id, first_ct_id_full + 4],
                ct_full_to_red_map[wd_id, first_ct_id_full + 5],
            )

            # Get joint local positions and orientation
            x_follower = joints_F_r_F[jt_id_tot]
            X_T = wp.transpose(joints_X[jt_id_tot])

            # Get base and follower transformations
            base_id_tot = joints_bid_B[jt_id_tot]
            if base_id_tot < 0:
                c_base = wp.vec3f(0.0, 0.0, 0.0)
                q_base = wp.quatf(0.0, 0.0, 0.0, 1.0)
            else:
                c_base = wp.transform_get_translation(bodies_q[base_id_tot])
                q_base = wp.transform_get_rotation(bodies_q[base_id_tot])
            follower_id_tot = joints_bid_F[jt_id_tot]
            c_follower = wp.transform_get_translation(bodies_q[follower_id_tot])
            q_follower = wp.transform_get_rotation(bodies_q[follower_id_tot])
            base_id_loc = base_id_tot - first_body_id[wd_id]
            follower_id_loc = follower_id_tot - first_body_id[wd_id]

            # Get position control transformation (rotation part only, as translation part doesn't affect the Jacobian)
            q_control_body = wp.transform_get_rotation(pos_control_transforms[jt_id_tot])

            # Translation constraints
            X_T_R_base_T = X_T * unit_quat_conj_to_rotation_matrix(q_base)
            if base_id_tot >= 0:
                jac_trans_c_base = -X_T_R_base_T
                delta_pos = unit_quat_apply(q_follower, x_follower) + c_follower - c_base
                jac_trans_q_base = X_T * unit_quat_conj_apply_jacobian(q_base, delta_pos)
            jac_trans_c_follower = X_T_R_base_T
            jac_trans_q_follower = X_T_R_base_T * unit_quat_apply_jacobian(q_follower, x_follower)

            # Rotation constraints
            q_base_sq_norm = wp.dot(q_base, q_base)
            q_follower_sq_norm = wp.dot(q_follower, q_follower)
            R_base_T = unit_quat_conj_to_rotation_matrix(q_base / wp.sqrt(q_base_sq_norm))
            q_rel = q_follower * wp.quat_inverse(q_control_body) * wp.quat_inverse(q_base)
            temp = X_T * R_base_T * quat_left_jacobian_inverse(q_rel)
            if base_id_tot >= 0:
                jac_rot_q_base = (-2.0 / q_base_sq_norm) * temp * G_of(q_base)
            jac_rot_q_follower = (2.0 / q_follower_sq_norm) * temp * G_of(q_follower)
            # Note: we need X^T * R_base^T both for translation and rotation constraints, but to get the correct
            # derivatives for non-unit quaternions (which may be encountered before convergence) we end up needing
            # to use a separate formula to evaluate R_base in either case

            # Write out Jacobian
            base_offset = 7 * base_id_loc
            follower_offset = 7 * follower_id_loc
            for i in range(3):
                trans_ct_id_red = trans_ct_ids_red[i]
                if trans_ct_id_red >= 0:
                    for j in range(3):
                        if base_id_tot >= 0:
                            constraints_jacobian[wd_id, trans_ct_id_red, base_offset + j] = jac_trans_c_base[i, j]
                        constraints_jacobian[wd_id, trans_ct_id_red, follower_offset + j] = jac_trans_c_follower[i, j]
                    for j in range(4):
                        if base_id_tot >= 0:
                            constraints_jacobian[wd_id, trans_ct_id_red, base_offset + 3 + j] = jac_trans_q_base[i, j]
                        constraints_jacobian[wd_id, trans_ct_id_red, follower_offset + 3 + j] = jac_trans_q_follower[
                            i, j
                        ]
                rot_ct_id_red = rot_ct_ids_red[i]
                if rot_ct_id_red >= 0:
                    for j in range(4):
                        if base_id_tot >= 0:
                            constraints_jacobian[wd_id, rot_ct_id_red, base_offset + 3 + j] = jac_rot_q_base[i, j]
                        constraints_jacobian[wd_id, rot_ct_id_red, follower_offset + 3 + j] = jac_rot_q_follower[i, j]

            # Correct Jacobian for passive universal joints
            if wp.static(has_universal_joints):
                # Check for a passive universal joint
                dof_type_j = joints_dof_type[jt_id_tot]
                act_type_j = joints_act_type[jt_id_tot]
                if dof_type_j != int(JointDoFType.UNIVERSAL) or act_type_j != int(JointActuationType.PASSIVE):
                    return

                # Compute constraint Jacobian (cross product between x axis on base and y axis on follower)
                a_x = X_T[0]
                a_y = X_T[1]
                if base_id_tot >= 0:
                    a_y_follower = unit_quat_apply(q_follower, a_y)
                    jac_q_base = -a_y_follower * unit_quat_apply_jacobian(q_base, a_x)
                a_x_base = unit_quat_apply(q_base, a_x)
                jac_q_follower = -a_x_base * unit_quat_apply_jacobian(q_follower, a_y)

                # Write out Jacobian
                for i in range(4):
                    rot_ct_id_red = rot_ct_ids_red[2]
                    if base_id_tot >= 0:
                        constraints_jacobian[wd_id, rot_ct_id_red, base_offset + 3 + i] = jac_q_base[i]
                    constraints_jacobian[wd_id, rot_ct_id_red, follower_offset + 3 + i] = jac_q_follower[i]

    return _eval_joint_constraints_jacobian


@cache
def create_tile_based_kernels(TILE_SIZE_CTS: wp.int32, TILE_SIZE_VRS: wp.int32):
    """
    Generates and returns all tile-based kernels in this module, given the tile size to use along the constraints
    and variables (i.e. body poses) dimensions in the constraint vector, Jacobian, step vector etc.

    These are _eval_pattern_T_pattern, _eval_max_constraint, _eval_jacobian_T_jacobian, eval_jacobian_T_constraints,
    _eval_merit_function, _eval_merit_function_gradient (returned in this order)
    """

    @wp.func
    def clip_to_one(x: wp.float32):
        """
        Clips an number to 1 if it is above
        """
        return wp.min(x, 1.0)

    @wp.kernel
    def _eval_pattern_T_pattern(
        # Inputs
        sparsity_pattern: wp.array3d(dtype=wp.float32),
        # Outputs
        pattern_T_pattern: wp.array3d(dtype=wp.float32),
    ):
        """
        A kernel computing the sparsity pattern of J^T * J given that of J, in each world
        More specifically, given an integer matrix of zeros and ones representing a sparsity pattern, multiply it by
        its transpose and clip values to [0, 1] to get the sparsity pattern of J^T * J
        Note: mostly redundant with _eval_jacobian_T_jacobian apart from the clipping, could possibly be removed
        (was initially written to take int32, but float32 is actually faster)

        Inputs:
            sparsity_pattern: Jacobian sparsity pattern per world
        Outputs:
            pattern_T_pattern: Jacobian^T * Jacobian sparsity pattern per world
        """
        wd_id, i, j = wp.tid()  # Thread indices (= world index, output tile indices)

        if (
            wd_id < pattern_T_pattern.shape[0]
            and i * TILE_SIZE_VRS < pattern_T_pattern.shape[1]
            and j * TILE_SIZE_VRS < pattern_T_pattern.shape[2]
        ):
            tile_out = wp.tile_zeros(shape=(TILE_SIZE_VRS, TILE_SIZE_VRS), dtype=wp.float32)

            num_cts = sparsity_pattern.shape[1]
            num_tiles_K = (num_cts + TILE_SIZE_CTS - 1) // TILE_SIZE_CTS  # Equivalent to ceil(num_cts / TILE_SIZE_CTS)

            for k in range(num_tiles_K):
                tile_i_3d = wp.tile_load(
                    sparsity_pattern,
                    shape=(1, TILE_SIZE_CTS, TILE_SIZE_VRS),
                    offset=(wd_id, k * TILE_SIZE_CTS, i * TILE_SIZE_VRS),
                )
                tile_i = wp.tile_reshape(tile_i_3d, (TILE_SIZE_CTS, TILE_SIZE_VRS))
                tile_i_T = wp.tile_transpose(tile_i)
                tile_j_3d = wp.tile_load(
                    sparsity_pattern,
                    shape=(1, TILE_SIZE_CTS, TILE_SIZE_VRS),
                    offset=(wd_id, k * TILE_SIZE_CTS, j * TILE_SIZE_VRS),
                )
                tile_j = wp.tile_reshape(tile_j_3d, (TILE_SIZE_CTS, TILE_SIZE_VRS))
                wp.tile_matmul(tile_i_T, tile_j, tile_out)

            tile_out_3d = wp.tile_reshape(tile_out, (1, TILE_SIZE_VRS, TILE_SIZE_VRS))
            tile_out_3d_clipped = wp.tile_map(clip_to_one, tile_out_3d)
            wp.tile_store(pattern_T_pattern, tile_out_3d_clipped, offset=(wd_id, i * TILE_SIZE_VRS, j * TILE_SIZE_VRS))

    @wp.func
    def _isnan(x: wp.float32) -> wp.int32:
        """Calls wp.isnan and converts the result to int32"""
        return wp.int32(wp.isnan(x))

    @wp.kernel
    def _eval_max_constraint(
        # Inputs
        constraints: wp.array2d(dtype=wp.float32),
        # Outputs
        max_constraint: wp.array(dtype=wp.float32),
    ):
        """
        A kernel computing the max absolute constraint from the constraints vector, in each world.

        Inputs:
            constraints: Constraint vector per world
        Outputs:
            max_constraint: Max absolute constraint per world; must be zero-initialized
        """
        wd_id, i, tid = wp.tid()  # Thread indices (= world index, input tile index, thread index in block)

        if wd_id < constraints.shape[0] and i * TILE_SIZE_CTS < constraints.shape[1]:
            segment = wp.tile_load(constraints, shape=(1, TILE_SIZE_CTS), offset=(wd_id, i * TILE_SIZE_CTS))
            segment_max = wp.tile_max(wp.tile_map(wp.abs, segment))[0]
            segment_has_nan = wp.tile_max(wp.tile_map(_isnan, segment))[0]

            if tid == 0:
                if segment_has_nan:
                    # Write NaN in max (non-atomically, as this will overwrite any non-NaN value)
                    max_constraint[wd_id] = wp.nan
                else:
                    # Atomically update the max, only if it is not yet NaN (in CUDA, the max() operation only
                    # considers non-NaN values, so the NaN value would get overwritten by a non-NaN otherwise)
                    while True:
                        curr_val = max_constraint[wd_id]
                        if wp.isnan(curr_val):
                            break
                        check_val = wp.atomic_cas(max_constraint, wd_id, curr_val, wp.max(curr_val, segment_max))
                        if check_val == curr_val:
                            break

    @wp.kernel
    def _eval_jacobian_T_jacobian(
        # Inputs
        constraints_jacobian: wp.array3d(dtype=wp.float32),
        world_mask: wp.array(dtype=wp.int32),
        # Outputs
        jacobian_T_jacobian: wp.array3d(dtype=wp.float32),
    ):
        """
        A kernel computing the matrix product J^T * J given the Jacobian J, in each world

        Inputs:
            constraints_jacobian: Constraint Jacobian per world
            world_mask: Per-world flag to perform the computation (0 = skip)
        Outputs:
            jacobian_T_jacobian: Jacobian^T * Jacobian per world
        """
        wd_id, i, j = wp.tid()  # Thread indices (= world index, output tile indices)

        if (
            wd_id < jacobian_T_jacobian.shape[0]
            and world_mask[wd_id] != 0
            and i * TILE_SIZE_VRS < jacobian_T_jacobian.shape[1]
            and j * TILE_SIZE_VRS < jacobian_T_jacobian.shape[2]
        ):
            tile_out = wp.tile_zeros(shape=(TILE_SIZE_VRS, TILE_SIZE_VRS), dtype=wp.float32)

            num_cts = constraints_jacobian.shape[1]
            num_tiles_K = (num_cts + TILE_SIZE_CTS - 1) // TILE_SIZE_CTS  # Equivalent to ceil(num_cts / TILE_SIZE_CTS)

            for k in range(num_tiles_K):
                tile_i_3d = wp.tile_load(
                    constraints_jacobian,
                    shape=(1, TILE_SIZE_CTS, TILE_SIZE_VRS),
                    offset=(wd_id, k * TILE_SIZE_CTS, i * TILE_SIZE_VRS),
                )
                tile_i = wp.tile_reshape(tile_i_3d, (TILE_SIZE_CTS, TILE_SIZE_VRS))
                tile_i_T = wp.tile_transpose(tile_i)
                tile_j_3d = wp.tile_load(
                    constraints_jacobian,
                    shape=(1, TILE_SIZE_CTS, TILE_SIZE_VRS),
                    offset=(wd_id, k * TILE_SIZE_CTS, j * TILE_SIZE_VRS),
                )
                tile_j = wp.tile_reshape(tile_j_3d, (TILE_SIZE_CTS, TILE_SIZE_VRS))
                wp.tile_matmul(tile_i_T, tile_j, tile_out)

            tile_out_3d = wp.tile_reshape(tile_out, (1, TILE_SIZE_VRS, TILE_SIZE_VRS))
            wp.tile_store(jacobian_T_jacobian, tile_out_3d, offset=(wd_id, i * TILE_SIZE_VRS, j * TILE_SIZE_VRS))

    @wp.kernel
    def _eval_jacobian_T_constraints(
        # Inputs
        constraints_jacobian: wp.array3d(dtype=wp.float32),
        constraints: wp.array2d(dtype=wp.float32),
        world_mask: wp.array(dtype=wp.int32),
        # Outputs
        jacobian_T_constraints: wp.array2d(dtype=wp.float32),
    ):
        """
        A kernel computing the matrix product J^T * C given the Jacobian J and the constraints vector C, in each world

        Inputs:
            constraints_jacobian: Constraint Jacobian per world
            constraints: Constraint vector per world
            world_mask: Per-world flag to perform the computation (0 = skip)
        Outputs:
            jacobian_T_constraints: Jacobian^T * Constraints per world
        """
        wd_id, i = wp.tid()  # Thread indices (= world index, output tile index)

        if (
            wd_id < jacobian_T_constraints.shape[0]
            and world_mask[wd_id] != 0
            and i * TILE_SIZE_VRS < jacobian_T_constraints.shape[1]
        ):
            segment_out = wp.tile_zeros(shape=(TILE_SIZE_VRS, 1), dtype=wp.float32)

            num_cts = constraints_jacobian.shape[1]
            num_tiles_K = (num_cts + TILE_SIZE_CTS - 1) // TILE_SIZE_CTS  # Equivalent to ceil(num_cts / TILE_SIZE_CTS)

            for k in range(num_tiles_K):
                tile_i_3d = wp.tile_load(
                    constraints_jacobian,
                    shape=(1, TILE_SIZE_CTS, TILE_SIZE_VRS),
                    offset=(wd_id, k * TILE_SIZE_CTS, i * TILE_SIZE_VRS),
                )
                tile_i = wp.tile_reshape(tile_i_3d, (TILE_SIZE_CTS, TILE_SIZE_VRS))
                tile_i_T = wp.tile_transpose(tile_i)
                segment_k_2d = wp.tile_load(constraints, shape=(1, TILE_SIZE_CTS), offset=(wd_id, k * TILE_SIZE_CTS))
                segment_k = wp.tile_reshape(segment_k_2d, (TILE_SIZE_CTS, 1))  # Technically still 2d...
                wp.tile_matmul(tile_i_T, segment_k, segment_out)

            segment_out_2d = wp.tile_reshape(
                segment_out,
                (
                    1,
                    TILE_SIZE_VRS,
                ),
            )
            wp.tile_store(
                jacobian_T_constraints,
                segment_out_2d,
                offset=(
                    wd_id,
                    i * TILE_SIZE_VRS,
                ),
            )

    @wp.kernel
    def _eval_merit_function(
        # Inputs
        constraints: wp.array2d(dtype=wp.float32),
        # Outputs
        merit_function_val: wp.array(dtype=wp.float32),
    ):
        """
        A kernel computing the merit function, i.e. the least-squares error 1/2 * ||C||^2, from the constraints
        vector C, in each world

        Inputs:
            constraints: Constraint vector per world
        Outputs:
            merit_function_val: Merit function value per world; must be zero-initialized
        """
        wd_id, i, tid = wp.tid()  # Thread indices (= world index, input tile index, thread index in block)

        if wd_id < constraints.shape[0] and i * TILE_SIZE_CTS < constraints.shape[1]:
            segment = wp.tile_load(constraints, shape=(1, TILE_SIZE_CTS), offset=(wd_id, i * TILE_SIZE_CTS))
            segment_error = 0.5 * wp.tile_sum(wp.tile_map(wp.mul, segment, segment))[0]

            if tid == 0:
                wp.atomic_add(merit_function_val, wd_id, segment_error)

    @wp.kernel
    def _eval_merit_function_gradient(
        # Inputs
        step: wp.array2d(dtype=wp.float32),
        grad: wp.array2d(dtype=wp.float32),
        # Outputs
        merit_function_grad: wp.array(dtype=wp.float32),
    ):
        """
        A kernel computing the merit function gradient w.r.t. line search step size, from the step direction
        and the gradient in state space (= dC_ds^T * C). This is simply the dot product between these two vectors.

        Inputs:
            step: Step in variables per world
            grad: Gradient w.r.t. state (i.e. body poses) per world
        Outputs:
            merit_function_grad: Merit function gradient per world; must be zero-initialized
        """
        wd_id, i, tid = wp.tid()  # Thread indices (= world index, input tile index, thread index in block)

        if wd_id < step.shape[0] and i * TILE_SIZE_VRS < step.shape[1]:
            step_segment = wp.tile_load(step, shape=(1, TILE_SIZE_VRS), offset=(wd_id, i * TILE_SIZE_VRS))
            grad_segment = wp.tile_load(grad, shape=(1, TILE_SIZE_VRS), offset=(wd_id, i * TILE_SIZE_VRS))
            tile_dot_prod = wp.tile_sum(wp.tile_map(wp.mul, step_segment, grad_segment))[0]

            if tid == 0:
                wp.atomic_add(merit_function_grad, wd_id, tile_dot_prod)

    return (
        _eval_pattern_T_pattern,
        _eval_max_constraint,
        _eval_jacobian_T_jacobian,
        _eval_jacobian_T_constraints,
        _eval_merit_function,
        _eval_merit_function_gradient,
    )


@wp.kernel
def _eval_rhs(
    # Inputs
    grad: wp.array2d(dtype=wp.float32),
    # Outputs
    rhs: wp.array2d(dtype=wp.float32),
):
    """
    A kernel computing rhs := -grad (where rhs has shape (num_worlds, num_states_max, 1))

    Inputs:
        grad: Merit function gradient w.r.t. state (i.e. body poses) per world
    Outputs:
        rhs: Gauss-Newton right-hand side per world
    """
    wd_id, state_id_loc = wp.tid()  # Thread indices (= world index, state index)
    if wd_id < grad.shape[0] and state_id_loc < grad.shape[1]:
        rhs[wd_id, state_id_loc] = -grad[wd_id, state_id_loc]


@wp.kernel
def _eval_stepped_state(
    # Inputs
    num_bodies: wp.array(dtype=wp.int32),
    first_body_id: wp.array(dtype=wp.int32),
    bodies_q_0_flat: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    step: wp.array2d(dtype=wp.float32),
    world_mask: wp.array(dtype=wp.int32),
    # Outputs
    bodies_q_alpha_flat: wp.array(dtype=wp.float32),
):
    """
    A kernel computing states_alpha := states_0 + alpha * step

    Inputs:
        num_bodies: Num bodies per world
        first_body_id: First body id per world
        bodies_q_0_flat: Previous state (for step size 0), flattened
        alpha: Step size per world
        step: Step direction per world
        world_mask: Per-world flag to perform the computation (0 = skip)
    Outputs:
        bodies_q_alpha_flat: New state (for step size alpha), flattened
    """
    wd_id, state_id_loc = wp.tid()  # Thread indices (= world index, state index)
    rb_id_loc = state_id_loc // 7
    if wd_id < num_bodies.shape[0] and world_mask[wd_id] != 0 and rb_id_loc < num_bodies[wd_id]:
        state_id_tot = 7 * first_body_id[wd_id] + state_id_loc
        bodies_q_alpha_flat[state_id_tot] = bodies_q_0_flat[state_id_tot] + alpha[wd_id] * step[wd_id, state_id_loc]


@wp.kernel
def _apply_line_search_step(
    # Inputs
    num_bodies: wp.array(dtype=wp.int32),
    first_body_id: wp.array(dtype=wp.int32),
    bodies_q_alpha: wp.array(dtype=wp.transformf),
    line_search_success: wp.array(dtype=wp.int32),
    # Outputs
    bodies_q: wp.array(dtype=wp.transformf),
):
    """
    A kernel replacing the state with the line search result, in worlds where line search succeeded
    Note: relies on the fact that the success flag is left at zero for worlds that don't run line search
    (otherwise would also need to check against line search mask)

    Inputs
        num_bodies: Num bodies per world
        first_body_id: First body id per world
        bodies_q_alpha: Stepped states (line search result)
        line_search_success: Per-world line search success flag
    Outputs
        bodies_q: Output state (rigid body poses)
    """
    wd_id, rb_id_loc = wp.tid()  # Thread indices (= world index, body index)
    if wd_id < num_bodies.shape[0] and line_search_success[wd_id] != 0 and rb_id_loc < num_bodies[wd_id]:
        rb_id_tot = first_body_id[wd_id] + rb_id_loc
        bodies_q[rb_id_tot] = bodies_q_alpha[rb_id_tot]


@wp.kernel
def _line_search_check(
    # Inputs
    val_0: wp.array(dtype=wp.float32),
    grad_0: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    val_alpha: wp.array(dtype=wp.float32),
    iteration: wp.array(dtype=wp.int32),
    max_iterations: wp.array(dtype=wp.int32, shape=(1,)),
    # Outputs
    line_search_success: wp.array(dtype=wp.int32),
    line_search_mask: wp.array(dtype=wp.int32),
    line_search_loop_condition: wp.array(dtype=wp.int32, shape=(1,)),
):
    """
    A kernel checking the sufficient decrease condition in line search in each world, and updating the looping
    condition (zero if max iterations reached, or all worlds successful)

    Inputs:
        val_0: Merit function value at 0, per world
        grad_0: Merit function gradient at 0, per world
        alpha: Step size per world (in/out)
        val_alpha: Merit function value at alpha, per world
        iteration: Iteration count, per world
        max_iterations: Max iterations
    Outputs:
        line_search_success: Convergence per world
        line_search_mask: Per-world flag to continue line search (0 = skip)
        line_search_loop_condition: Loop condition; must be zero-initialized
    """
    wd_id = wp.tid()  # Thread index (= world index)
    if wd_id < val_0.shape[0] and line_search_mask[wd_id] != 0:
        iteration[wd_id] += 1
        line_search_success[wd_id] = int(
            wp.isfinite(val_alpha[wd_id]) and val_alpha[wd_id] <= val_0[wd_id] + 1e-4 * alpha[wd_id] * grad_0[wd_id]
        )
        continue_loop_world = iteration[wd_id] < max_iterations[0] and not line_search_success[wd_id]
        line_search_mask[wd_id] = int(continue_loop_world)
        if continue_loop_world:
            alpha[wd_id] *= 0.5
        wp.atomic_max(line_search_loop_condition, 0, int(continue_loop_world))


@wp.kernel
def _newton_check(
    # Inputs
    max_constraint: wp.array(dtype=wp.float32),
    tolerance: wp.array(dtype=wp.float32, shape=(1,)),
    iteration: wp.array(dtype=wp.int32),
    max_iterations: wp.array(dtype=wp.int32, shape=(1,)),
    line_search_success: wp.array(dtype=wp.int32),
    # Outputs
    newton_success: wp.array(dtype=wp.int32),
    newton_mask: wp.array(dtype=wp.int32),
    newton_loop_condition: wp.array(dtype=wp.int32, shape=(1,)),
):
    """
    A kernel checking the convergence (max constraint vs tolerance) in each world, and updating the looping
    condition (zero if max iterations reached, or all worlds successful)

    Inputs
        max_constraint: Max absolute constraint per world
        tolerance: Tolerance on max constraint
        iteration: Iteration count, per world
        max_iterations: Max iterations
        line_search_success: Per-world line search success flag
    Outputs
        newton_success: Convergence per world
        newton_mask: Flag to keep iterating per world
        newton_loop_condition: Loop condition; must be zero-initialized
    """
    wd_id = wp.tid()  # Thread index (= world index)
    if wd_id < max_constraint.shape[0] and newton_mask[wd_id] != 0:
        iteration[wd_id] += 1
        max_constraint_wd = max_constraint[wd_id]
        is_finite = wp.isfinite(max_constraint_wd)
        newton_success[wd_id] = int(is_finite and max_constraint_wd <= tolerance[0])
        newton_continue_world = int(
            iteration[wd_id] < max_iterations[0]
            and not newton_success[wd_id]
            and is_finite  # Abort when encountering NaN / Inf values
            and line_search_success[wd_id]  # Abort in case of line search failure
        )
        newton_mask[wd_id] = newton_continue_world
        wp.atomic_max(newton_loop_condition, 0, newton_continue_world)


@wp.kernel
def _eval_target_constraint_velocities(
    # Inputs
    num_joints: wp.array(dtype=wp.int32),
    first_joint_id: wp.array(dtype=wp.int32),
    joints_dof_type: wp.array(dtype=wp.int32),
    joints_act_type: wp.array(dtype=wp.int32),
    actuated_dofs_offset: wp.array(dtype=wp.int32),
    ct_full_to_red_map: wp.array2d(dtype=wp.int32),
    actuators_u: wp.array(dtype=wp.float32),
    world_mask: wp.array(dtype=wp.int32),
    # Outputs
    target_cts_u: wp.array2d(dtype=wp.float32),
):
    """
    A kernel computing the target constraint velocities, i.e. zero for passive constraints
    and the prescribed dof velocity for actuated constraints.

    Inputs:
        num_joints: Num joints per world
        first_joint_id: First joint id per world
        joints_dof_type: Joint dof type (i.e. revolute, spherical, ...)
        joints_act_type: Joint actuation type (i.e. passive or actuated)
        actuated_dofs_offset: Joint first actuated dof id, among all actuated dofs in all worlds
        ct_full_to_red_map: Map from full to reduced constraint id (per world)
        actuators_u: Actuated joint velocities
        world_mask: Per-world flag to perform the computation (0 = skip)
    Outputs:
        target_cts_u: Target constraint velocities (assumed to be zero-initialized)
    """
    # Retrieve the thread indices (= world index, joint index)
    wd_id, jt_id_loc = wp.tid()

    if wd_id < world_mask.shape[0] and world_mask[wd_id] != 0 and jt_id_loc < num_joints[wd_id]:
        # Retrieve the joint model data
        jt_id_tot = first_joint_id[wd_id] + jt_id_loc
        if joints_act_type[jt_id_tot] != JointActuationType.FORCE:
            return
        dof_type_j = joints_dof_type[jt_id_tot]
        offset_u_j = actuated_dofs_offset[jt_id_tot]
        offset_cts_j = ct_full_to_red_map[wd_id, 6 * jt_id_loc]

        if dof_type_j == JointDoFType.CARTESIAN:
            target_cts_u[wd_id, offset_cts_j] = actuators_u[offset_u_j]
            target_cts_u[wd_id, offset_cts_j + 1] = actuators_u[offset_u_j + 1]
            target_cts_u[wd_id, offset_cts_j + 2] = actuators_u[offset_u_j + 2]
        elif dof_type_j == JointDoFType.CYLINDRICAL:
            target_cts_u[wd_id, offset_cts_j] = actuators_u[offset_u_j]
            target_cts_u[wd_id, offset_cts_j + 3] = actuators_u[offset_u_j + 1]
        elif dof_type_j == JointDoFType.FIXED:
            pass  # No dofs to apply
        elif dof_type_j == JointDoFType.FREE:
            target_cts_u[wd_id, offset_cts_j] = actuators_u[offset_u_j]
            target_cts_u[wd_id, offset_cts_j + 1] = actuators_u[offset_u_j + 1]
            target_cts_u[wd_id, offset_cts_j + 2] = actuators_u[offset_u_j + 2]
            target_cts_u[wd_id, offset_cts_j + 3] = actuators_u[offset_u_j + 3]
            target_cts_u[wd_id, offset_cts_j + 4] = actuators_u[offset_u_j + 4]
            target_cts_u[wd_id, offset_cts_j + 5] = actuators_u[offset_u_j + 5]
        elif dof_type_j == JointDoFType.PRISMATIC:
            target_cts_u[wd_id, offset_cts_j] = actuators_u[offset_u_j]
        elif dof_type_j == JointDoFType.REVOLUTE:
            target_cts_u[wd_id, offset_cts_j + 3] = actuators_u[offset_u_j]
        elif dof_type_j == JointDoFType.SPHERICAL:
            target_cts_u[wd_id, offset_cts_j + 3] = actuators_u[offset_u_j]
            target_cts_u[wd_id, offset_cts_j + 4] = actuators_u[offset_u_j + 1]
            target_cts_u[wd_id, offset_cts_j + 5] = actuators_u[offset_u_j + 2]
        elif dof_type_j == JointDoFType.UNIVERSAL:
            target_cts_u[wd_id, offset_cts_j + 3] = actuators_u[offset_u_j]
            target_cts_u[wd_id, offset_cts_j + 4] = actuators_u[offset_u_j + 1]


@wp.kernel
def _eval_body_velocities(
    # Inputs
    num_bodies: wp.array(dtype=wp.int32),
    first_body_id: wp.array(dtype=wp.int32),
    bodies_q: wp.array(dtype=wp.transformf),
    bodies_q_dot: wp.array2d(dtype=wp.float32),
    world_mask: wp.array(dtype=wp.int32),
    # Outputs
    bodies_u: wp.array(dtype=vec6f),
):
    """
    A kernel computing the body velocities (twists) from the time derivative of body poses,
    computing in particular angular velocities omega = G(q)q_dot

    Inputs:
        num_bodies: Number of bodies per world
        first_body_id: First body id per world
        bodies_q: Body poses
        bodies_q_dot: Time derivative of body poses
        world_mask: Per-world flag to perform the computation (0 = skip)
    Outputs:
        bodies_u: Body velocities (twists)
    """
    wd_id, rb_id_loc = wp.tid()  # Thread indices (= world index, body index)
    if wd_id < world_mask.shape[0] and world_mask[wd_id] != 0 and rb_id_loc < num_bodies[wd_id]:
        # Indices / offsets
        rb_id_tot = first_body_id[wd_id] + rb_id_loc
        offset_q_dot = 7 * rb_id_loc

        # Copy linear velocity
        bodies_u[rb_id_tot][0] = bodies_q_dot[wd_id, offset_q_dot]
        bodies_u[rb_id_tot][1] = bodies_q_dot[wd_id, offset_q_dot + 1]
        bodies_u[rb_id_tot][2] = bodies_q_dot[wd_id, offset_q_dot + 2]

        # Compute angular velocities
        q = wp.transform_get_rotation(bodies_q[rb_id_tot])
        q_dot = wp.vec4f(
            bodies_q_dot[wd_id, offset_q_dot + 3],
            bodies_q_dot[wd_id, offset_q_dot + 4],
            bodies_q_dot[wd_id, offset_q_dot + 5],
            bodies_q_dot[wd_id, offset_q_dot + 6],
        )
        omega = 2.0 * (G_of(q) * q_dot)
        bodies_u[rb_id_tot][3] = omega[0]
        bodies_u[rb_id_tot][4] = omega[1]
        bodies_u[rb_id_tot][5] = omega[2]


###
# Classes
###


@dataclass
class ForwardKinematicsSolverSettings:
    """
    Host-side class to store settings for the forward kinematics solve.
    """

    max_newton_iterations: wp.int32 = 30
    """Maximal number of Gauss-Newton iterations (default: 30).
       Changes to this setting after the solver's initialization will have no effect."""

    max_line_search_iterations: wp.int32 = 20
    """Maximal line search iterations in the inner loop (default: 20).
       Changes to this setting after the solver's initialization will have no effect."""

    tolerance: wp.float32 = 1e-6
    """Maximal absolute kinematic constraint value that is acceptable at the solution (default: 1e-6).
       Changes to this setting after the solver's initialization will have no effect."""

    reset_state: bool = True
    """Whether to reset the state to initial states, to use as initial guess (default: True).
       Changes to this setting after graph capture will have no effect."""

    TILE_SIZE_CTS: wp.int32 = 8
    """Tile size for kernels along the dimension of kinematic constraints (default: 8).
       Changes to this setting after the solver's initialization will have no effect."""

    TILE_SIZE_VRS: wp.int32 = 8
    """Tile size for kernels along the dimension of rigid body pose variables (default: 8).
       Changes to this setting after the solver's initialization will have no effect."""


@dataclass
class ForwardKinematicsSolverStatus:
    """
    Class containing detailed data on the success/failure status of a forward kinematics solve

    Attributes
    ----------
    iterations : np.ndarray
        the number of Newton iterations run per world
    max_constraints : np.ndarray
        the maximal kinematic constraint residual per world
    success : np.ndarray
        the solver success flag per world (i.e., constraint residual below tolerance within max iterations)
    """

    iterations: np.ndarray(dtype=np.int32)

    max_constraints: np.ndarray(dtype=np.float32)

    success: np.ndarray(dtype=np.int32)


class ForwardKinematicsSolver:
    """
    Forward Kinematics solver class
    """

    def __init__(self, model: Model | None = None, settings: ForwardKinematicsSolverSettings | None = None):
        """
        Initializes the solver to solve forward kinematics for a given model.

        Parameters
        ----------
        model : Model, optional
            Model for which to solve forward kinematics. If not provided, the finalize() method
            must be called at a later time for deferred initialization (default: None).
        settings : ForwardKinematicsSolverSettings, optional
            Solver settings. If not provided, default settings will be used (default: None).
        """

        self.model: Model | None = None
        """Underlying model"""

        self.device: Devicelike = None
        """Device for data allocations"""

        self.settings: ForwardKinematicsSolverSettings = ForwardKinematicsSolverSettings()
        """Solver settings"""

        self.linear_solver: SemiSparseBlockCholeskySolverBatched | None = None
        """Semi-sparse Cholesky solver for the J^T * J linear system"""

        self.graph: wp.Graph | None = None
        """Cuda graph for the convenience function with verbosity options"""

        # Note: there are many other internal data members below, which are not documented here

        # Set model and settings, and finalize if model was provided
        self.model = model
        if settings is not None:
            self.settings = settings
        if model is not None:
            self.finalize()

    def finalize(self, model: Model | None = None, settings: ForwardKinematicsSolverSettings | None = None):
        """
        Finishes the solver initialization, performing necessary allocations and precomputations.
        This method only needs to be called manually if a model was not provided in the constructor,
        or to reset the solver for a new model.

        Parameters
        ----------
        model : Model, optional
            Model for which to solve forward kinematics. If not provided, the model given to the
            constructor will be used. Must be provided if not given to the constructor (default: None).
        settings : ForwardKinematicsSolverSettings, optional
            Solver settings. If not provided, the settings given to the constructor, or if not, the
            default settings will be used (default: None).
        """

        # Initialize the model and settings if provided
        if model is not None:
            self.model = model
        if settings is not None:
            self.settings = settings
        if self.model is None:
            raise ValueError("ForwardKinematicsSolver: error, provided model is None.")

        # Initialize device
        self.device = self.model.device

        # Retrieve / compute dimensions - Worlds
        self.num_worlds = self.model.size.num_worlds  # For convenience

        # Retrieve / compute dimensions - Bodies
        num_bodies = self.model.info.num_bodies.numpy()  # Number of bodies per world
        first_body_id = np.concatenate(([0], num_bodies.cumsum()))  # Index of first body per world
        self.num_bodies_max = self.model.size.max_of_num_bodies  # Max number of bodies across worlds

        # Retrieve / compute dimensions - States (i.e., body poses)
        num_states = 7 * num_bodies  # Number of state dimensions per world
        self.num_states_tot = 7 * self.model.size.sum_of_num_bodies  # State dimensions for the whole model
        self.num_states_max = 7 * self.num_bodies_max  # Max state dimension across worlds

        # Retrieve / compute dimensions - Joints (main model)
        num_joints_prev = self.model.info.num_joints.numpy()  # Number of joints per world
        first_joint_id_prev = np.concatenate(([0], num_joints_prev.cumsum()))  # Index of first joint per world

        # Retrieve / compute dimensions - Actuated coordinates (main model)
        num_actuated_coords_prev = (
            self.model.info.num_actuated_joint_coords.numpy()
        )  # Number of actuated joint coordinates per world
        first_actuated_coord_prev = np.concatenate(
            ([0], num_actuated_coords_prev.cumsum())
        )  # Index of first actuated coordinate per world
        actuated_coord_offsets_prev = (
            self.model.joints.actuated_coords_offset.numpy()
        )  # Index of first joint actuated coordinate, among actuated coordinates of a single world
        for wd_id in range(self.num_worlds):  # Convert into offsets among all actuated coordinates
            actuated_coord_offsets_prev[first_joint_id_prev[wd_id] : first_joint_id_prev[wd_id + 1]] += (
                first_actuated_coord_prev[wd_id]
            )
            # Note: will currently produce garbage for passive joints (because for these the offsets are set to -1)
            # but we won't read these values below anyway.

        # Retrieve / compute dimensions - Actuated dofs (main model)
        num_actuated_dofs_prev = (
            self.model.info.num_actuated_joint_dofs.numpy()
        )  # Number of actuated joint dofs per world
        first_actuated_dof_prev = np.concatenate(
            ([0], num_actuated_dofs_prev.cumsum())
        )  # Index of first actuated dof per world
        actuated_dof_offsets_prev = (
            self.model.joints.actuated_dofs_offset.numpy().copy()
        )  # Index of first joind actuated dof, among actuated dofs of a single world
        for wd_id in range(self.num_worlds):  # Convert into offsets among all actuated dofs
            actuated_dof_offsets_prev[first_joint_id_prev[wd_id] : first_joint_id_prev[wd_id + 1]] += (
                first_actuated_dof_prev[wd_id]
            )
            # Note: will currently produce garbage for passive joints (because for these the offsets are set to -1)
            # but we won't read these values below anyway.

        # Create a copy of the model's joints with added actuated free joints as needed to reset the base position/orientation
        joints_dof_type_prev = self.model.joints.dof_type.numpy()
        joints_act_type_prev = self.model.joints.act_type.numpy()
        joints_bid_B_prev = self.model.joints.bid_B.numpy()
        joints_bid_F_prev = self.model.joints.bid_F.numpy()
        joints_B_r_Bj_prev = self.model.joints.B_r_Bj.numpy()
        joints_F_r_Fj_prev = self.model.joints.F_r_Fj.numpy()
        joints_X_j_prev = self.model.joints.X_j.numpy()
        joints_num_coords_prev = self.model.joints.num_coords.numpy()
        joints_num_dofs_prev = self.model.joints.num_dofs.numpy()
        joints_dof_type = []
        joints_act_type = []
        joints_bid_B = []
        joints_bid_F = []
        joints_B_r_Bj = []
        joints_F_r_Fj = []
        joints_X_j = []
        joints_num_actuated_coords = []  # Number of actuated coordinates per joint (0 for passive joints)
        joints_num_actuated_dofs = []  # Number of actuated dofs per joint (0 for passive joints)
        num_joints = np.zeros(self.num_worlds, dtype=np.int32)  # Number of joints per world
        self.num_joints_tot = 0  # Number of joints for all worlds
        actuated_coords_map = []  # Map of new actuated coordinates to these in the model or to the base coordinates
        actuated_dofs_map = []  # Map of new actuated dofs to these in the model or to the base dofs
        base_q_default = np.zeros(7 * self.num_worlds, dtype=np.float32)  # Default base pose
        bodies_q_0 = self.model.bodies.q_i_0.numpy()
        for wd_id in range(self.num_worlds):
            # Retrieve base joint id if set
            base_joint_id = -1
            if self.model.worlds[wd_id].has_base_joint:
                base_joint_id = first_joint_id_prev[wd_id] + self.model.worlds[wd_id].base_joint_idx

            # Copy data for all kept joints
            world_joint_ids = [
                i for i in range(first_joint_id_prev[wd_id], first_joint_id_prev[wd_id + 1]) if i != base_joint_id
            ]
            for jt_id_prev in world_joint_ids:
                joints_dof_type.append(joints_dof_type_prev[jt_id_prev])
                joints_act_type.append(joints_act_type_prev[jt_id_prev])
                joints_bid_B.append(joints_bid_B_prev[jt_id_prev])
                joints_bid_F.append(joints_bid_F_prev[jt_id_prev])
                joints_B_r_Bj.append(joints_B_r_Bj_prev[jt_id_prev])
                joints_F_r_Fj.append(joints_F_r_Fj_prev[jt_id_prev])
                joints_X_j.append(joints_X_j_prev[jt_id_prev])
                if joints_act_type[-1] == JointActuationType.FORCE:
                    num_coords_jt = joints_num_coords_prev[jt_id_prev]
                    joints_num_actuated_coords.append(num_coords_jt)
                    coord_offset = actuated_coord_offsets_prev[jt_id_prev]
                    actuated_coords_map.extend(range(coord_offset, coord_offset + num_coords_jt))

                    num_dofs_jt = joints_num_dofs_prev[jt_id_prev]
                    joints_num_actuated_dofs.append(num_dofs_jt)
                    dof_offset = actuated_dof_offsets_prev[jt_id_prev]
                    actuated_dofs_map.extend(range(dof_offset, dof_offset + num_dofs_jt))
                else:
                    joints_num_actuated_coords.append(0)
                    joints_num_actuated_dofs.append(0)

            # Add joint for base joint / base body
            if base_joint_id >= 0:  # Replace base joint with an actuated free joint
                joints_dof_type.append(JointDoFType.FREE)
                joints_act_type.append(JointActuationType.FORCE)
                joints_bid_B.append(-1)
                joints_bid_F.append(joints_bid_F_prev[base_joint_id])
                joints_B_r_Bj.append(joints_B_r_Bj_prev[base_joint_id])
                joints_F_r_Fj.append(joints_F_r_Fj_prev[base_joint_id])
                joints_X_j.append(joints_X_j_prev[base_joint_id])
                joints_num_actuated_coords.append(7)
                coord_offset = -7 * wd_id - 1  # We encode offsets in base_q negatively with i -> -i - 1
                actuated_coords_map.extend(range(coord_offset, coord_offset - 7, -1))
                base_q_default[7 * wd_id : 7 * wd_id + 7] = [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.1,
                ]  # Default to zero of free joint
                joints_num_actuated_dofs.append(6)
                dof_offset = -6 * wd_id - 1  # We encode offsets in base_u negatively with i -> -i - 1
                actuated_dofs_map.extend(range(dof_offset, dof_offset - 6, -1))
            elif self.model.worlds[wd_id].has_base_body:  # Add an actuated free joint to the base body
                base_body_id = first_body_id[wd_id] + self.model.worlds[wd_id].base_body_idx
                joints_dof_type.append(JointDoFType.FREE)
                joints_act_type.append(JointActuationType.FORCE)
                joints_bid_B.append(-1)
                joints_bid_F.append(base_body_id)
                joints_B_r_Bj.append(np.zeros(3, dtype=np.float32))
                joints_F_r_Fj.append(np.zeros(3, dtype=np.float32))
                joints_X_j.append(np.eye(3, 3, dtype=np.float32))
                joints_num_actuated_coords.append(7)
                # Note: we rely on the initial body orientations being identity
                # Only then will the corresponding joint coordinates be interpretable as
                # specifying the absolute base position and orientation
                coord_offset = -7 * wd_id - 1  # We encode offsets in base_q negatively with i -> -i - 1
                actuated_coords_map.extend(range(coord_offset, coord_offset - 7, -1))
                base_q_default[7 * wd_id : 7 * wd_id + 7] = bodies_q_0[base_body_id]  # Default to initial body pose
                joints_num_actuated_dofs.append(6)
                dof_offset = -6 * wd_id - 1  # We encode offsets in base_u negatively with i -> -i - 1
                actuated_dofs_map.extend(range(dof_offset, dof_offset - 6, -1))

            # Record number of joints
            num_joints_world = len(joints_dof_type) - self.num_joints_tot
            self.num_joints_tot += num_joints_world
            num_joints[wd_id] = num_joints_world

        # Retrieve / compute dimensions - Joints (FK model)
        first_joint_id = np.concatenate(([0], num_joints.cumsum()))  # Index of first joint per world
        self.num_joints_max = max(num_joints)  # Max number of joints across worlds

        # Retrieve / compute dimensions - Actuated coordinates (FK model)
        joints_num_actuated_coords = np.array(joints_num_actuated_coords)  # Number of actuated coordinates per joint
        actuated_coord_offsets = np.concatenate(
            ([0], joints_num_actuated_coords.cumsum())
        )  # First actuated coordinate offset per joint, among all actuated coordinates
        self.num_actuated_coords = actuated_coord_offsets[-1]

        # Retrieve / compute dimensions - Actuated dofs (FK model)
        joints_num_actuated_dofs = np.array(joints_num_actuated_dofs)  # Number of actuated dofs per joint
        actuated_dof_offsets = np.concatenate(
            ([0], joints_num_actuated_dofs.cumsum())
        )  # First actuated dof offset per joint, among all actuated dofs
        self.num_actuated_dofs = actuated_dof_offsets[-1]

        # Retrieve / compute dimensions - Constraints
        num_constraints = num_bodies.copy()  # Number of kinematic constraints per world (unit quat. + joints)
        has_universal_joints = False  # Whether the model has a least one passive universal joint
        constraint_full_to_red_map = -1 * np.ones((self.num_worlds, 6 * self.num_joints_max), dtype=np.int32)
        for wd_id in range(self.num_worlds):
            ct_count = num_constraints[wd_id]
            for jt_id_loc in range(num_joints[wd_id]):
                jt_id_tot = first_joint_id[wd_id] + jt_id_loc  # Joint id among all joints
                act_type = joints_act_type[jt_id_tot]
                if act_type == JointActuationType.FORCE:  # Actuator: select all six constraints
                    for i in range(6):
                        constraint_full_to_red_map[wd_id, 6 * jt_id_loc + i] = ct_count + i
                    ct_count += 6
                else:
                    dof_type = joints_dof_type[jt_id_tot]
                    if dof_type == JointDoFType.CARTESIAN:
                        for i in range(3):
                            constraint_full_to_red_map[wd_id, 6 * jt_id_loc + 3 + i] = ct_count + i
                        ct_count += 3
                    elif dof_type == JointDoFType.CYLINDRICAL:
                        constraint_full_to_red_map[wd_id, 6 * jt_id_loc + 1] = ct_count
                        constraint_full_to_red_map[wd_id, 6 * jt_id_loc + 2] = ct_count + 1
                        constraint_full_to_red_map[wd_id, 6 * jt_id_loc + 4] = ct_count + 2
                        constraint_full_to_red_map[wd_id, 6 * jt_id_loc + 5] = ct_count + 3
                        ct_count += 4
                    elif dof_type == JointDoFType.FIXED:
                        for i in range(6):
                            constraint_full_to_red_map[wd_id, 6 * jt_id_loc + i] = ct_count + i
                        ct_count += 6
                    elif dof_type == JointDoFType.FREE:
                        pass
                    elif dof_type == JointDoFType.PRISMATIC:
                        constraint_full_to_red_map[wd_id, 6 * jt_id_loc + 1] = ct_count
                        constraint_full_to_red_map[wd_id, 6 * jt_id_loc + 2] = ct_count + 1
                        for i in range(3):
                            constraint_full_to_red_map[wd_id, 6 * jt_id_loc + 3 + i] = ct_count + 2 + i
                        ct_count += 5
                    elif dof_type == JointDoFType.REVOLUTE:
                        for i in range(3):
                            constraint_full_to_red_map[wd_id, 6 * jt_id_loc + i] = ct_count + i
                        constraint_full_to_red_map[wd_id, 6 * jt_id_loc + 4] = ct_count + 3
                        constraint_full_to_red_map[wd_id, 6 * jt_id_loc + 5] = ct_count + 4
                        ct_count += 5
                    elif dof_type == JointDoFType.SPHERICAL:
                        for i in range(3):
                            constraint_full_to_red_map[wd_id, 6 * jt_id_loc + i] = ct_count + i
                        ct_count += 3
                    elif dof_type == JointDoFType.UNIVERSAL:
                        for i in range(3):
                            constraint_full_to_red_map[wd_id, 6 * jt_id_loc + i] = ct_count + i
                        constraint_full_to_red_map[wd_id, 6 * jt_id_loc + 5] = ct_count + 3
                        ct_count += 4
                        has_universal_joints = True
                    else:
                        raise RuntimeError("Unknown joint dof type")
            num_constraints[wd_id] = ct_count
        self.num_constraints_max = np.max(num_constraints)

        # Retrieve / compute dimensions - Number of tiles (for kernels using Tile API)
        self.num_tiles_constraints = (
            self.num_constraints_max + self.settings.TILE_SIZE_CTS - 1
        ) // self.settings.TILE_SIZE_CTS
        self.num_tiles_states = (self.num_states_max + self.settings.TILE_SIZE_VRS - 1) // self.settings.TILE_SIZE_VRS

        # Data allocation or transfer from numpy to warp
        with wp.ScopedDevice(self.device):
            # Dimensions
            self.first_body_id = wp.from_numpy(first_body_id, dtype=wp.int32)
            self.num_joints = wp.from_numpy(num_joints, dtype=wp.int32)
            self.first_joint_id = wp.from_numpy(first_joint_id, dtype=wp.int32)
            self.actuated_coord_offsets = wp.from_numpy(actuated_coord_offsets, dtype=wp.int32)
            self.actuated_coords_map = wp.from_numpy(np.array(actuated_coords_map), dtype=wp.int32)
            self.actuated_dof_offsets = wp.from_numpy(actuated_dof_offsets, dtype=wp.int32)
            self.actuated_dofs_map = wp.from_numpy(np.array(actuated_dofs_map), dtype=wp.int32)
            self.num_states = wp.from_numpy(num_states, dtype=wp.int32)
            self.constraint_full_to_red_map = wp.from_numpy(constraint_full_to_red_map, dtype=wp.int32)

            # Modified joints
            self.joints_dof_type = wp.from_numpy(joints_dof_type, dtype=wp.int32)
            self.joints_act_type = wp.from_numpy(joints_act_type, dtype=wp.int32)
            self.joints_bid_B = wp.from_numpy(joints_bid_B, dtype=wp.int32)
            self.joints_bid_F = wp.from_numpy(joints_bid_F, dtype=wp.int32)
            self.joints_B_r_Bj = wp.from_numpy(joints_B_r_Bj, dtype=wp.vec3f)
            self.joints_F_r_Fj = wp.from_numpy(joints_F_r_Fj, dtype=wp.vec3f)
            self.joints_X_j = wp.from_numpy(joints_X_j, dtype=wp.mat33f)

            # Default base state
            self.base_q_default = wp.from_numpy(base_q_default, dtype=wp.transformf)
            self.base_u_default = wp.zeros(shape=(self.num_worlds,), dtype=vec6f)

            # Line search
            self.max_line_search_iterations = wp.array(dtype=wp.int32, shape=(1,))  # Max iterations
            self.max_line_search_iterations.fill_(self.settings.max_line_search_iterations)
            self.line_search_iteration = wp.array(dtype=wp.int32, shape=(self.num_worlds,))  # Iteration count
            self.line_search_loop_condition = wp.array(dtype=wp.int32, shape=(1,))  # Loop condition
            self.line_search_success = wp.array(dtype=wp.int32, shape=(self.num_worlds,))  # Convergence, per world
            self.line_search_mask = wp.array(
                dtype=wp.int32, shape=(self.num_worlds,)
            )  # Flag to keep iterating per world
            self.val_0 = wp.array(dtype=wp.float32, shape=(self.num_worlds,))  # Merit function value at 0, per world
            self.grad_0 = wp.array(
                dtype=wp.float32, shape=(self.num_worlds,)
            )  # Merit function gradient at 0, per world
            self.alpha = wp.array(dtype=wp.float32, shape=(self.num_worlds,))  # Step size, per world
            self.bodies_q_alpha = wp.array(dtype=wp.transformf, shape=(self.model.size.sum_of_num_bodies,))  # New state
            self.val_alpha = wp.array(dtype=wp.float32, shape=(self.num_worlds,))  # New merit function value, per world

            # Gauss-Newton
            self.max_newton_iterations = wp.array(dtype=wp.int32, shape=(1,))  # Max iterations
            self.max_newton_iterations.fill_(self.settings.max_newton_iterations)
            self.newton_iteration = wp.array(dtype=wp.int32, shape=(self.num_worlds,))  # Iteration count
            self.newton_loop_condition = wp.array(dtype=wp.int32, shape=(1,))  # Loop condition
            self.newton_success = wp.array(dtype=wp.int32, shape=(self.num_worlds,))  # Convergence per world
            self.newton_mask = wp.array(dtype=wp.int32, shape=(self.num_worlds,))  # Flag to keep iterating per world
            self.tolerance = wp.array(dtype=wp.float32, shape=(1,))  # Tolerance on max constraint
            self.tolerance.fill_(self.settings.tolerance)
            self.actuators_q = wp.array(dtype=wp.float32, shape=(self.num_actuated_coords,))  # Actuated coordinates
            self.pos_control_transforms = wp.array(
                dtype=wp.transformf, shape=(self.num_joints_tot,)
            )  # Position-control transformations at joints
            self.constraints = wp.zeros(
                dtype=wp.float32,
                shape=(
                    self.num_worlds,
                    self.num_constraints_max,
                ),
            )  # Constraints vector per world
            self.max_constraint = wp.array(dtype=wp.float32, shape=(self.num_worlds,))  # Maximal constraint per world
            self.jacobian = wp.zeros(
                dtype=wp.float32, shape=(self.num_worlds, self.num_constraints_max, self.num_states_max)
            )  # Constraints Jacobian per world
            self.lhs = wp.zeros(
                dtype=wp.float32, shape=(self.num_worlds, self.num_states_max, self.num_states_max)
            )  # Gauss-Newton left-hand side per world
            self.grad = wp.zeros(
                dtype=wp.float32, shape=(self.num_worlds, self.num_states_max)
            )  # Merit function gradient w.r.t. state per world
            self.rhs = wp.zeros(
                dtype=wp.float32, shape=(self.num_worlds, self.num_states_max)
            )  # Gauss-Newton right-hand side per world (=-grad)
            self.step = wp.zeros(
                dtype=wp.float32, shape=(self.num_worlds, self.num_states_max)
            )  # Step in state variables per world

            # Velocity solver
            self.actuators_u = wp.array(
                dtype=wp.float32, shape=(self.num_actuated_dofs,)
            )  # Velocities for actuated dofs of fk model
            self.target_cts_u = wp.zeros(
                dtype=wp.float32,
                shape=(
                    self.num_worlds,
                    self.num_constraints_max,
                ),
            )  # Target velocity per constraint
            self.bodies_q_dot = wp.array(
                ptr=self.step.ptr, dtype=wp.float32, shape=(self.num_worlds, self.num_states_max), copy=False
            )  # Time derivative of body poses (alias of self.step for data re-use)
            # Note: we also re-use self.jacobian, self.lhs and self.rhs for the velocity solver

        # Initialize kernels that depend on static values
        self._eval_joint_constraints_kernel = create_eval_joint_constraints_kernel(has_universal_joints)
        self._eval_joint_constraints_jacobian_kernel = create_eval_joint_constraints_jacobian_kernel(
            has_universal_joints
        )
        (
            self._eval_pattern_T_pattern_kernel,
            self._eval_max_constraint_kernel,
            self._eval_jacobian_T_jacobian_kernel,
            self._eval_jacobian_T_constraints_kernel,
            self._eval_merit_function_kernel,
            self._eval_merit_function_gradient_kernel,
        ) = create_tile_based_kernels(self.settings.TILE_SIZE_CTS, self.settings.TILE_SIZE_VRS)

        # Compute sparsity pattern and initialize linear solver (running symbolic factorization)

        # Jacobian sparsity pattern
        sparsity_pattern = np.zeros((self.num_worlds, self.num_constraints_max, self.num_states_max), dtype=int)
        for wd_id in range(self.num_worlds):
            for rb_id_loc in range(num_bodies[wd_id]):
                sparsity_pattern[wd_id, rb_id_loc, 7 * rb_id_loc + 3 : 7 * rb_id_loc + 7] = 1
            for jt_id_loc in range(num_joints[wd_id]):
                jt_id_tot = first_joint_id[wd_id] + jt_id_loc
                base_id_tot = joints_bid_B[jt_id_tot]
                follower_id_tot = joints_bid_F[jt_id_tot]
                rb_ids_tot = [base_id_tot, follower_id_tot] if base_id_tot >= 0 else [follower_id_tot]
                for rb_id_tot in rb_ids_tot:
                    rb_id_loc = rb_id_tot - first_body_id[wd_id]
                    state_offset = 7 * rb_id_loc
                    for i in range(3):
                        ct_offset = constraint_full_to_red_map[wd_id, 6 * jt_id_loc + i]  # ith translation constraint
                        if ct_offset >= 0:
                            sparsity_pattern[wd_id, ct_offset, state_offset : state_offset + 7] = 1
                        ct_offset = constraint_full_to_red_map[wd_id, 6 * jt_id_loc + 3 + i]  # ith rotation constraint
                        if ct_offset >= 0:
                            sparsity_pattern[wd_id, ct_offset, state_offset + 3 : state_offset + 7] = 1

        # Jacobian^T * Jacobian sparsity pattern
        sparsity_pattern_wp = wp.from_numpy(sparsity_pattern, dtype=wp.float32, device=self.device)
        sparsity_pattern_lhs_wp = wp.zeros(
            dtype=wp.float32, shape=(self.num_worlds, self.num_states_max, self.num_states_max), device=self.device
        )
        wp.launch_tiled(
            self._eval_pattern_T_pattern_kernel,
            dim=(self.num_worlds, self.num_tiles_states, self.num_tiles_states),
            inputs=[sparsity_pattern_wp, sparsity_pattern_lhs_wp],
            block_dim=64,
            device=self.device,
        )
        sparsity_pattern_lhs = sparsity_pattern_lhs_wp.numpy().astype("int32")

        # Initialize linear solver
        self.linear_solver = SemiSparseBlockCholeskySolverBatched(
            self.num_worlds,
            self.num_states_max,
            block_size=16,  # TODO: optimize this (e.g. 14 ?)
            device=self.device,
            enable_reordering=True,
        )
        self.linear_solver.capture_sparsity_pattern(sparsity_pattern_lhs, num_states)

    ###
    # Internal evaluators (graph-capturable functions working on pre-allocated data)
    ###

    def _reset_state(
        self,
        bodies_q: wp.array(dtype=wp.transformf),
        world_mask: wp.array(dtype=wp.int32),
    ):
        """
        Internal function resetting the bodies state to the reference state stored in the model.
        """
        wp.launch(
            _reset_state,
            dim=(self.num_worlds, self.num_states_max),
            inputs=[
                self.model.info.num_bodies,
                self.first_body_id,
                wp.array(
                    ptr=self.model.bodies.q_i_0.ptr,
                    dtype=wp.float32,
                    shape=(self.num_states_tot,),
                    device=self.device,
                    copy=False,
                ),
                world_mask,
                wp.array(
                    ptr=bodies_q.ptr, dtype=wp.float32, shape=(self.num_states_tot,), device=self.device, copy=False
                ),
            ],
            device=self.device,
        )

    def _eval_position_control_transformations(
        self,
        base_q: wp.array(dtype=wp.transformf),
        actuators_q: wp.array(dtype=wp.float32),
        pos_control_transforms: wp.array(dtype=wp.transformf),
    ):
        """
        Internal evaluator for position control transformations, from actuated & base coordinates of the main model.
        """
        # Compute actuators_q of fk model with modified joints
        wp.launch(
            _eval_fk_actuated_dofs_or_coords,
            dim=(self.num_actuated_coords,),
            inputs=[
                wp.array(
                    ptr=base_q.ptr, dtype=wp.float32, shape=(7 * self.num_worlds,), device=self.device, copy=False
                ),
                actuators_q,
                self.actuated_coords_map,
                self.actuators_q,
            ],
            device=self.device,
        )

        # Compute position control transformations
        wp.launch(
            _eval_position_control_transformations,
            dim=(self.num_joints_tot,),
            inputs=[
                self.joints_dof_type,
                self.joints_act_type,
                self.actuated_coord_offsets,
                self.joints_X_j,
                self.actuators_q,
                pos_control_transforms,
            ],
            device=self.device,
        )

    def _eval_kinematic_constraints(
        self,
        bodies_q: wp.array(dtype=wp.transformf),
        pos_control_transforms: wp.array(dtype=wp.transformf),
        world_mask: wp.array(dtype=wp.int32),
        constraints: wp.array2d(dtype=wp.float32),
    ):
        """
        Internal evaluator for the kinematic constraints vector, from body poses and position-control transformations
        """

        # Evaluate unit norm quaternion constraints
        wp.launch(
            _eval_unit_quaternion_constraints,
            dim=(self.num_worlds, self.num_bodies_max),
            inputs=[self.model.info.num_bodies, self.first_body_id, bodies_q, world_mask, constraints],
            device=self.device,
        )
        # Evaluate joint constraints
        wp.launch(
            self._eval_joint_constraints_kernel,
            dim=(self.num_worlds, self.num_joints_max),
            inputs=[
                self.num_joints,
                self.first_joint_id,
                self.joints_dof_type,
                self.joints_act_type,
                self.joints_bid_B,
                self.joints_bid_F,
                self.joints_X_j,
                self.joints_B_r_Bj,
                self.joints_F_r_Fj,
                bodies_q,
                pos_control_transforms,
                self.constraint_full_to_red_map,
                world_mask,
                constraints,
            ],
            device=self.device,
        )

    def _eval_max_constraint(
        self, constraints: wp.array2d(dtype=wp.float32), max_constraint: wp.array(dtype=wp.float32)
    ):
        """
        Internal evaluator for the maximal absolute constraint, from the constraints vector, in each world
        """
        max_constraint.zero_()
        wp.launch_tiled(
            self._eval_max_constraint_kernel,
            dim=(self.num_worlds, self.num_tiles_constraints),
            inputs=[constraints, max_constraint],
            block_dim=64,
            device=self.device,
        )

    def _eval_kinematic_constraints_jacobian(
        self,
        bodies_q: wp.array(dtype=wp.transformf),
        pos_control_transforms: wp.array(dtype=wp.transformf),
        world_mask: wp.array(dtype=wp.int32),
        constraints_jacobian: wp.array3d(dtype=wp.float32),
    ):
        """
        Internal evaluator for the kinematic constraints Jacobian with respect to body poses, from body poses
        and position-control transformations
        """

        # Evaluate unit norm quaternion constraints Jacobian
        wp.launch(
            _eval_unit_quaternion_constraints_jacobian,
            dim=(self.num_worlds, self.num_bodies_max),
            inputs=[self.model.info.num_bodies, self.first_body_id, bodies_q, world_mask, constraints_jacobian],
            device=self.device,
        )

        # Evaluate joint constraints Jacobian
        wp.launch(
            self._eval_joint_constraints_jacobian_kernel,
            dim=(self.num_worlds, self.num_joints_max),
            inputs=[
                self.num_joints,
                self.first_joint_id,
                self.first_body_id,
                self.joints_dof_type,
                self.joints_act_type,
                self.joints_bid_B,
                self.joints_bid_F,
                self.joints_X_j,
                self.joints_B_r_Bj,
                self.joints_F_r_Fj,
                bodies_q,
                pos_control_transforms,
                self.constraint_full_to_red_map,
                world_mask,
                constraints_jacobian,
            ],
            device=self.device,
        )

    def _eval_merit_function(self, constraints: wp.array2d(dtype=wp.float32), error: wp.array(dtype=wp.float32)):
        """
        Internal evaluator for the line search merit function, i.e. the least-squares error 1/2 * ||C||^2,
        from the constraints vector C, in each world
        """
        error.zero_()
        wp.launch_tiled(
            self._eval_merit_function_kernel,
            dim=(self.num_worlds, self.num_tiles_constraints),
            inputs=[constraints, error],
            block_dim=64,
            device=self.device,
        )

    def _eval_merit_function_gradient(
        self,
        step: wp.array2d(dtype=wp.float32),
        grad: wp.array2d(dtype=wp.float32),
        error_grad: wp.array(dtype=wp.float32),
    ):
        """
        Internal evaluator for the merit function gradient w.r.t. line search step size, from the step direction
        and the gradient in state space (= dC_ds^T * C). This is simply the dot product between these two vectors.
        """
        error_grad.zero_()
        wp.launch_tiled(
            self._eval_merit_function_gradient_kernel,
            dim=(self.num_worlds, self.num_tiles_states),
            inputs=[step, grad, error_grad],
            block_dim=64,
            device=self.device,
        )

    def _run_line_search_iteration(self, bodies_q: wp.array(dtype=wp.transformf)):
        """
        Internal function running one iteration of line search, checking the Armijo sufficient descent condition
        """
        # Eval stepped state
        wp.launch(
            _eval_stepped_state,
            dim=(self.num_worlds, self.num_states_max),
            inputs=[
                self.model.info.num_bodies,
                self.first_body_id,
                wp.array(
                    ptr=bodies_q.ptr, dtype=wp.float32, shape=(self.num_states_tot,), device=self.device, copy=False
                ),
                self.alpha,
                self.step,
                self.line_search_mask,
                wp.array(
                    ptr=self.bodies_q_alpha.ptr,
                    dtype=wp.float32,
                    shape=(self.num_states_tot,),
                    device=self.device,
                    copy=False,
                ),
            ],
            device=self.device,
        )

        # Evaluate new constraints and merit function (least squares norm of constraints)
        self._eval_kinematic_constraints(
            self.bodies_q_alpha, self.pos_control_transforms, self.line_search_mask, self.constraints
        )
        self._eval_merit_function(self.constraints, self.val_alpha)

        # Check decrease and update step
        self.line_search_loop_condition.zero_()
        wp.launch(
            _line_search_check,
            dim=(self.num_worlds,),
            inputs=[
                self.val_0,
                self.grad_0,
                self.alpha,
                self.val_alpha,
                self.line_search_iteration,
                self.max_line_search_iterations,
                self.line_search_success,
                self.line_search_mask,
                self.line_search_loop_condition,
            ],
            device=self.device,
        )

    def _run_newton_iteration(self, bodies_q: wp.array(dtype=wp.transformf)):
        """
        Internal function running one iteration of Gauss-Newton. Assumes the constraints vector to be already
        up-to-date (because we will already have checked convergence before the first loop iteration)
        """
        # Evaluate constraints Jacobian
        self._eval_kinematic_constraints_jacobian(
            bodies_q, self.pos_control_transforms, self.newton_mask, self.jacobian
        )

        # Evaluate Gauss-Newton left-hand side (J^T * J) and right-hand side (-J^T * C)
        wp.launch_tiled(
            self._eval_jacobian_T_jacobian_kernel,
            dim=(self.num_worlds, self.num_tiles_states, self.num_tiles_states),
            inputs=[self.jacobian, self.newton_mask, self.lhs],
            block_dim=64,
            device=self.device,
        )
        wp.launch_tiled(
            self._eval_jacobian_T_constraints_kernel,
            dim=(self.num_worlds, self.num_tiles_states),
            inputs=[self.jacobian, self.constraints, self.newton_mask, self.grad],
            block_dim=64,
            device=self.device,
        )
        wp.launch(
            _eval_rhs,
            dim=(self.num_worlds, self.num_states_max),
            inputs=[self.grad, self.rhs],
            device=self.device,
        )

        # Compute step (system solve)
        self.linear_solver.factorize(self.lhs, self.num_states, self.newton_mask)
        self.linear_solver.solve(
            self.rhs.reshape((self.num_worlds, self.num_states_max, 1)),
            self.step.reshape((self.num_worlds, self.num_states_max, 1)),
            self.newton_mask,
        )

        # Line search
        self.line_search_iteration.zero_()
        self.line_search_success.zero_()
        wp.copy(self.line_search_mask, self.newton_mask)
        self.line_search_loop_condition.fill_(1)
        self._eval_merit_function(self.constraints, self.val_0)
        self._eval_merit_function_gradient(self.step, self.grad, self.grad_0)
        self.alpha.fill_(1.0)
        wp.capture_while(self.line_search_loop_condition, lambda: self._run_line_search_iteration(bodies_q))

        # Apply line search step and update max constraint
        wp.launch(
            _apply_line_search_step,
            dim=(self.num_worlds, self.num_bodies_max),
            inputs=[
                self.model.info.num_bodies,
                self.first_body_id,
                self.bodies_q_alpha,
                self.line_search_success,
                bodies_q,
            ],
            device=self.device,
        )
        self._eval_max_constraint(self.constraints, self.max_constraint)

        # Check convergence
        self.newton_loop_condition.zero_()
        wp.launch(
            _newton_check,
            dim=(self.num_worlds,),
            inputs=[
                self.max_constraint,
                self.tolerance,
                self.newton_iteration,
                self.max_newton_iterations,
                self.line_search_success,
                self.newton_success,
                self.newton_mask,
                self.newton_loop_condition,
            ],
            device=self.device,
        )

    def _solve_for_body_velocities(
        self,
        pos_control_transforms: wp.array(dtype=wp.transformf),
        base_u: wp.array(dtype=vec6f),
        actuators_u: wp.array(dtype=wp.float32),
        bodies_q: wp.array(dtype=wp.transformf),
        bodies_u: wp.array(dtype=vec6f),
        world_mask: wp.array(dtype=wp.int32),
    ):
        """
        Internal function solving for body velocities, so that constraint velocities are zero,
        except at actuated dofs and at the base joint, where they must match prescribed velocities.
        """
        # Compute actuators_u of fk model with modified joints
        wp.launch(
            _eval_fk_actuated_dofs_or_coords,
            dim=(self.num_actuated_dofs,),
            inputs=[
                wp.array(
                    ptr=base_u.ptr, dtype=wp.float32, shape=(6 * self.num_worlds,), device=self.device, copy=False
                ),
                actuators_u,
                self.actuated_dofs_map,
                self.actuators_u,
            ],
            device=self.device,
        )

        # Compute target constraint velocities (prescribed for actuated dofs, zero for passive constraints)
        self.target_cts_u.zero_()
        wp.launch(
            _eval_target_constraint_velocities,
            dim=(
                self.num_worlds,
                self.num_joints_max,
            ),
            inputs=[
                self.num_joints,
                self.first_joint_id,
                self.joints_dof_type,
                self.joints_act_type,
                self.actuated_dof_offsets,
                self.constraint_full_to_red_map,
                self.actuators_u,
                world_mask,
                self.target_cts_u,
            ],
            device=self.device,
        )

        # Update constraints Jacobian
        self._eval_kinematic_constraints_jacobian(bodies_q, pos_control_transforms, world_mask, self.jacobian)

        # Evaluate system left-hand side (J^T * J) and right-hand side (J^T * targets_cts_u)
        wp.launch_tiled(
            self._eval_jacobian_T_jacobian_kernel,
            dim=(self.num_worlds, self.num_tiles_states, self.num_tiles_states),
            inputs=[self.jacobian, world_mask, self.lhs],
            block_dim=64,
            device=self.device,
        )
        wp.launch_tiled(
            self._eval_jacobian_T_constraints_kernel,
            dim=(self.num_worlds, self.num_tiles_states),
            inputs=[self.jacobian, self.target_cts_u, world_mask, self.rhs],
            block_dim=64,
            device=self.device,
        )

        # Compute body velocities (system solve)
        self.linear_solver.factorize(self.lhs, self.num_states, world_mask)
        self.linear_solver.solve(
            self.rhs.reshape((self.num_worlds, self.num_states_max, 1)),
            self.bodies_q_dot.reshape((self.num_worlds, self.num_states_max, 1)),
            world_mask,
        )
        wp.launch(
            _eval_body_velocities,
            dim=(self.num_worlds, self.num_bodies_max),
            inputs=[self.model.info.num_bodies, self.first_body_id, bodies_q, self.bodies_q_dot, world_mask, bodies_u],
            device=self.device,
        )

    ###
    # Exposed functions (overall solve_fk() function + constraints (Jacobian) evaluators for debugging)
    ###

    def eval_position_control_transformations(
        self, actuators_q: wp.array(dtype=wp.float32), base_q: wp.array(dtype=wp.transformf) | None = None
    ):
        """
        Evaluates and returns position control transformations (an intermediary quantity needed for the
        kinematic constraints/Jacobian evaluation) for a model given actuated coordinates, and optionally
        the base pose (the default base pose is used if not provided).
        """
        assert base_q is None or base_q.device == self.device
        assert actuators_q.device == self.device

        if base_q is None:
            base_q = self.base_q_default

        pos_control_transforms = wp.array(dtype=wp.transformf, shape=(self.num_joints_tot,), device=self.device)
        self._eval_position_control_transformations(base_q, actuators_q, pos_control_transforms)
        return pos_control_transforms

    def eval_kinematic_constraints(
        self, bodies_q: wp.array(dtype=wp.transformf), pos_control_transforms: wp.array(dtype=wp.transformf)
    ):
        """
        Evaluates and returns the kinematic constraints vector given the body poses and the position
        control transformations.
        """
        assert bodies_q.device == self.device
        assert pos_control_transforms.device == self.device

        constraints = wp.zeros(
            dtype=wp.float32,
            shape=(
                self.num_worlds,
                self.num_constraints_max,
            ),
            device=self.device,
        )
        world_mask = wp.ones(dtype=wp.int32, shape=(self.num_worlds,), device=self.device)
        self._eval_kinematic_constraints(bodies_q, pos_control_transforms, world_mask, constraints)
        return constraints

    def eval_kinematic_constraints_jacobian(
        self, bodies_q: wp.array(dtype=wp.transformf), pos_control_transforms: wp.array(dtype=wp.transformf)
    ):
        """
        Evaluates and returns the kinematic constraints Jacobian (w.r.t. body poses) given the body poses
        and the position control transformations.
        """
        assert bodies_q.device == self.device
        assert pos_control_transforms.device == self.device

        constraints_jacobian = wp.zeros(
            dtype=wp.float32, shape=(self.num_worlds, self.num_constraints_max, self.num_states_max), device=self.device
        )
        world_mask = wp.ones(dtype=wp.int32, shape=(self.num_worlds,), device=self.device)
        self._eval_kinematic_constraints_jacobian(bodies_q, pos_control_transforms, world_mask, constraints_jacobian)
        return constraints_jacobian

    def solve_for_body_velocities(
        self,
        pos_control_transforms: wp.array(dtype=wp.transformf),
        actuators_u: wp.array(dtype=wp.float32),
        bodies_q: wp.array(dtype=wp.transformf),
        bodies_u: wp.array(dtype=vec6f),
        base_u: wp.array(dtype=vec6f) | None = None,
        world_mask: wp.array(dtype=wp.int32) | None = None,
    ):
        """
        Graph-capturable function solving for body velocities as a post-processing to the FK solve.
        More specifically, solves for body twists yielding zero constraint velocities, except at
        actuated dofs and at the base joint, where velocities must match prescribed velocities.

        Parameters
        ----------
        pos_control_transforms : wp.array
            Array of position-control transforms, encoding actuated coordinates and base pose.
            Expects shape of ``(num_fk_joints,)`` and type :class:`transform`
        actuators_u : wp.array
            Array of actuated joint velocities.
            Expects shape of ``(sum_of_num_actuated_joint_dofs,)`` and type :class:`float`.
        bodies_q : wp.array
            Array of rigid body poses. Must be the solution of FK given the position-control transforms.
            Expects shape of ``(num_bodies,)`` and type :class:`transform`.
        bodies_u : wp.array
            Array of rigid body velocities (twists), written out by the solver.
            Expects shape of ``(num_bodies,)`` and type :class:`vec6`.
        base_u : wp.array, optional
            Velocity (twist) of the base body for each world, in the frame of the base joint if it was set, or
            absolute otherwise.
            If not provided, will default to zero. Ignored if no base body or joint was set for this model.
            If this function is captured in a graph, must be either always or never provided.
            Expects shape of ``(num_worlds,)`` and type :class:`vec6`.
        world_mask : wp.array, optional
            Array of per-world flags that indicate which worlds should be processed (0 = leave that world unchanged).
            If not provided, all worlds will be processed.
            If this function is captured in a graph, must be either always or never provided.
        """
        assert pos_control_transforms.device == self.device
        assert actuators_u.device == self.device
        assert bodies_q.device == self.device
        assert bodies_u.device == self.device
        assert base_u is None or base_u.device == self.device
        assert world_mask is None or world_mask.device == self.device

        # Use default base velocity if not provided
        if base_u is None:
            base_u = self.base_u_default

        # Compute velocities
        self._solve_for_body_velocities(pos_control_transforms, base_u, actuators_u, bodies_q, bodies_u, world_mask)

    def run_fk_solve(
        self,
        actuators_q: wp.array(dtype=wp.float32),
        bodies_q: wp.array(dtype=wp.transformf),
        base_q: wp.array(dtype=wp.transformf) | None = None,
        actuators_u: wp.array(dtype=wp.float32) | None = None,
        base_u: wp.array(dtype=vec6f) | None = None,
        bodies_u: wp.array(dtype=vec6f) | None = None,
        world_mask: wp.array(dtype=wp.int32) | None = None,
    ):
        """
        Graph-capturable function solving forward kinematics with Gauss-Newton.
        More specifically, solves for the rigid body poses satisfying kinematic constraints, given actuated joint coordinates
        and base pose. Optionally also solves for rigid body velocities given actuator and base body velocities.

        Parameters
        ----------
        actuators_q : wp.array
            Array of actuated joint coordinates.
            Expects shape of ``(sum_of_num_actuated_joint_coords,)`` and type :class:`float`.
        bodies_q : wp.array
            Array of rigid body poses, written out by the solver and read in as initial guess if the reset_state
            solver setting is False.
            Expects shape of ``(num_bodies,)`` and type :class:`transform`.
        base_q : wp.array, optional
            Pose of the base body for each world, in the frame of the base joint if it was set, or absolute otherwise.
            If not provided, will default to zero coordinates of the base joint, or the initial pose of the base body.
            If no base body or joint was set for this model, will be ignored.
            If this function is captured in a graph, must be either always or never provided.
            Expects shape of ``(num_worlds,)`` and type :class:`transform`.
        actuators_u : wp.array, optional
            Array of actuated joint velocities.
            Must be provided when solving for body velocities, i.e. if bodies_u is provided.
            If this function is captured in a graph, must be either always or never provided.
            Expects shape of ``(sum_of_num_actuated_joint_dofs,)`` and type :class:`float`.
        base_u : wp.array, optional
            Velocity (twist) of the base body for each world, in the frame of the base joint if it was set, or
            absolute otherwise.
            If not provided, will default to zero. Ignored if no base body or joint was set for this model.
            If this function is captured in a graph, must be either always or never provided.
            Expects shape of ``(num_worlds,)`` and type :class:`vec6`.
        bodies_u : wp.array, optional
            Array of rigid body velocities (twists), written out by the solver if provided.
            If this function is captured in a graph, must be either always or never provided.
            Expects shape of ``(num_bodies,)`` and type :class:`vec6`.
        world_mask : wp.array, optional
            Array of per-world flags that indicate which worlds should be processed (0 = leave that world unchanged).
            If not provided, all worlds will be processed.
            If this function is captured in a graph, must be either always or never provided.
        """
        # Check that actuators_u are provided if we need to solve for bodies_u
        if bodies_u is not None and actuators_u is None:
            raise ValueError(
                "run_fk_solve: actuators_u must be provided to solve for velocities (i.e. if bodies_u is provided)."
            )

        # Use default base state if not provided
        if base_q is None:
            base_q = self.base_q_default
        if bodies_u is not None and base_u is None:
            base_u = self.base_u_default

        # Compute position control transforms (independent of body poses, depends on controls only)
        self._eval_position_control_transformations(base_q, actuators_q, self.pos_control_transforms)

        # Reset iteration count and success/continuation flags
        self.newton_iteration.fill_(-1)  # The initial loop condition check will increment this to zero
        self.newton_success.zero_()
        if world_mask is not None:
            self.newton_mask.assign(world_mask)
        else:
            self.newton_mask.fill_(1)

        # Optionally reset state
        if self.settings.reset_state:
            self._reset_state(bodies_q, self.newton_mask)

        # Evaluate constraints, and initialize loop condition (might not even need to loop)
        self._eval_kinematic_constraints(bodies_q, self.pos_control_transforms, self.newton_mask, self.constraints)
        self._eval_max_constraint(self.constraints, self.max_constraint)
        self.newton_loop_condition.zero_()
        wp.copy(self.line_search_success, self.newton_mask)  # Newton check will abort in case of line search failure
        wp.launch(
            _newton_check,
            dim=(self.num_worlds,),
            inputs=[
                self.max_constraint,
                self.tolerance,
                self.newton_iteration,
                self.max_newton_iterations,
                self.line_search_success,
                self.newton_success,
                self.newton_mask,
                self.newton_loop_condition,
            ],
            device=self.device,
        )

        # Main loop
        wp.capture_while(self.newton_loop_condition, lambda: self._run_newton_iteration(bodies_q))

        # Velocity solve, for worlds where FK ran and was successful
        if bodies_u is not None:
            self._solve_for_body_velocities(
                self.pos_control_transforms, base_u, actuators_u, bodies_q, bodies_u, self.newton_success
            )

    def solve_fk(
        self,
        actuators_q: wp.array(dtype=wp.float32),
        bodies_q: wp.array(dtype=wp.transformf),
        base_q: wp.array(dtype=wp.transformf) | None = None,
        actuators_u: wp.array(dtype=wp.float32) | None = None,
        base_u: wp.array(dtype=vec6f) | None = None,
        bodies_u: wp.array(dtype=vec6f) | None = None,
        world_mask: wp.array(dtype=wp.int32) | None = None,
        verbose: bool = False,
        return_status: bool = False,
        use_graph: bool = True,
    ):
        """
        Convenience function with verbosity options (non graph-capturable), solving forward kinematics with Gauss-Newton.
        More specifically, solves for the rigid body poses satisfying kinematic constraints, given actuated joint coordinates
        and base pose. Optionally also solves for rigid body velocities given actuator and base body velocities.

        Parameters
        ----------
        actuators_q : wp.array
            Array of actuated joint coordinates.
            Expects shape of ``(sum_of_num_actuated_joint_coords,)`` and type :class:`float`.
        bodies_q : wp.array
            Array of rigid body poses, written out by the solver and read in as initial guess if the reset_state
            solver setting is False.
            Expects shape of ``(num_bodies,)`` and type :class:`transform`.
        base_q : wp.array, optional
            Pose of the base body for each world, in the frame of the base joint if it was set, or absolute otherwise.
            If not provided, will default to zero coordinates of the base joint, or the initial pose of the base body.
            If no base body or joint was set for this model, will be ignored.
            Expects shape of ``(num_worlds,)`` and type :class:`transform`.
        actuators_u : wp.array, optional
            Array of actuated joint velocities.
            Must be provided when solving for body velocities, i.e. if bodies_u is provided.
            Expects shape of ``(sum_of_num_actuated_joint_dofs,)`` and type :class:`float`.
        base_u : wp.array, optional
            Velocity (twist) of the base body for each world, in the frame of the base joint if it was set, or
            absolute otherwise.
            If not provided, will default to zero. Ignored if no base body or joint was set for this model.
            Expects shape of ``(num_worlds,)`` and type :class:`vec6`.
        bodies_u : wp.array, optional
            Array of rigid body velocities (twists), written out by the solver if provided.
            Expects shape of ``(num_bodies,)`` and type :class:`vec6`.
        world_mask : wp.array, optional
            Array of per-world flags that indicate which worlds should be processed (0 = leave that world unchanged).
            If not provided, all worlds will be processed.
        verbose : bool, optional
            whether to write a status message at the end (default: False)
        return_status : bool, optional
            whether to return the detailed solver status (default: False)
        use_graph : bool, optional
            whether to use graph capture internally to accelerate multiple calls to this function. Can be turned
            off for profiling individual kernels (default: True)

        Returns
        -------
        solver_status : ForwardKinematicsSolverStatus, optional
            the detailed solver status with success flag, number of iterations and constraint residual per world
        """
        assert base_q is None or base_q.device == self.device
        assert actuators_q.device == self.device
        assert bodies_q.device == self.device
        assert base_u is None or base_u.device == self.device
        assert actuators_u is None or actuators_u.device == self.device
        assert bodies_u is None or bodies_u.device == self.device

        # Run solve (with or without graph)
        if use_graph:
            if self.graph is None:
                wp.capture_begin(self.device)
                self.run_fk_solve(actuators_q, bodies_q, base_q, actuators_u, base_u, bodies_u, world_mask)
                self.graph = wp.capture_end()
            wp.capture_launch(self.graph)
        else:
            self.run_fk_solve(actuators_q, bodies_q, base_q, actuators_u, base_u, bodies_u, world_mask)

        # Status message
        if verbose or return_status:
            success = self.newton_success.numpy().copy()
            iterations = self.newton_iteration.numpy().copy()
            max_constraints = self.max_constraint.numpy().copy()
            num_active_worlds = self.num_worlds if world_mask is None else world_mask.sum()
            if verbose:
                sys.__stdout__.write(f"Newton success for {success.sum()}/{num_active_worlds} worlds; ")
                sys.__stdout__.write(f"num iterations={iterations.max()}; ")
                sys.__stdout__.write(f"max constraint={max_constraints.max()}\n")

        # Return solver status
        if return_status:
            return ForwardKinematicsSolverStatus(
                iterations=iterations, max_constraints=max_constraints, success=success
            )
