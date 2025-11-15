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
from ..core.model import Model, ModelData
from ..linalg.factorize.llt_blocked_semi_sparse import SemiSparseBlockCholeskySolverBatched

###
# Module interface
###

__all__ = ["ForwardKinematicsSolver"]


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
def _eval_position_control_transformations(
    # Inputs
    joints_dof_type: wp.array(dtype=wp.int32),  # Joint dof type (i.e. revolute, spherical, ...)
    joints_act_type: wp.array(dtype=wp.int32),  # Joint actuation type (i.e. passive or actuated)
    joints_dofs_offset: wp.array(dtype=wp.int32),  # Joint first dof id, amond all dofs of all joints in all worlds
    joints_X: wp.array(dtype=wp.mat33f),  # Joint frame (local axes, valid both on base and follower)
    joints_q_j: wp.array(dtype=wp.float32),  # Joint dofs
    # Outputs
    pos_control_transforms: wp.array(dtype=wp.transformf),  # Joint position-control transformation
):
    """
    A kernel computing a transformation per joint corresponding to position-control parameters
    More specifically, this is the identity (no translation, no rotation) for passive joints
    and a transformation corresponding to joint generalized coordinates for actuators

    The translation part is expressed in joint frame (e.g., translation is along [1,0,0] for a prismatic joint)
    The rotation part is expressed in body frame (e.g., rotation is about X[:,0] for a revolute joint)
    """

    # Retrieve the thread index (= joint index)
    jt_id = wp.tid()

    if jt_id < joints_dof_type.shape[0]:
        # Retrieve the joint model data
        dof_type_j = joints_dof_type[jt_id]
        act_type_j = joints_act_type[jt_id]
        offset_q_j = joints_dofs_offset[jt_id]
        X = joints_X[jt_id]

        # Initialize transform to identity (already covers the passive case)
        t = wp.vec3f(0.0, 0.0, 0.0)
        q = wp.quatf(0.0, 0.0, 0.0, 1.0)

        # In the actuated case, set translation/rotation as per joint generalized coordinates
        if act_type_j == JointActuationType.FORCE:
            if dof_type_j == JointDoFType.CARTESIAN:
                t[0] = joints_q_j[offset_q_j]
                t[1] = joints_q_j[offset_q_j + 1]
                t[2] = joints_q_j[offset_q_j + 2]
            elif dof_type_j == JointDoFType.CYLINDRICAL:
                t[0] = joints_q_j[offset_q_j]
                q = wp.quat_from_axis_angle(wp.vec3f(X[0, 0], X[1, 0], X[2, 0]), joints_q_j[offset_q_j + 1])
            elif dof_type_j == JointDoFType.FIXED:
                pass  # No dofs to apply
            elif dof_type_j == JointDoFType.FREE:
                t[0] = joints_q_j[offset_q_j]
                t[1] = joints_q_j[offset_q_j + 1]
                t[2] = joints_q_j[offset_q_j + 2]
                q_X = wp.quat_from_matrix(X)
                q_loc = read_quat_from_array(joints_q_j, offset_q_j + 3)
                q = q_X * q_loc * wp.quat_inverse(q_X)
            elif dof_type_j == JointDoFType.PRISMATIC:
                t[0] = joints_q_j[offset_q_j]
            elif dof_type_j == JointDoFType.REVOLUTE:
                q = wp.quat_from_axis_angle(wp.vec3f(X[0, 0], X[1, 0], X[2, 0]), joints_q_j[offset_q_j])
            elif dof_type_j == JointDoFType.SPHERICAL:
                q_X = wp.quat_from_matrix(X)
                q_loc = read_quat_from_array(joints_q_j, offset_q_j)
                q = q_X * q_loc * wp.quat_inverse(q_X)
            elif dof_type_j == JointDoFType.UNIVERSAL:
                q_x = wp.quat_from_axis_angle(wp.vec3f(X[0, 0], X[1, 0], X[2, 0]), joints_q_j[offset_q_j])
                q_y = wp.quat_from_axis_angle(wp.vec3f(X[0, 1], X[1, 1], X[2, 1]), joints_q_j[offset_q_j + 1])
                q = q_x * q_y

        # Write out transformation
        pos_control_transforms[jt_id] = wp.transformf(t, q)


@wp.kernel
def _eval_unit_quaternion_constraints(
    # Inputs
    num_bodies: wp.array(dtype=wp.int32),  # Num bodies per world
    first_body_id: wp.array(dtype=wp.int32),  # First body id per world
    bodies_q_i: wp.array(dtype=wp.transformf),  # Body poses
    skip_computation: wp.array(dtype=wp.int32),  # Per-world flag to skip the computation
    # Outputs
    constraints: wp.array2d(dtype=wp.float32),  # Constraint vector per world
):
    """
    A kernel computing unit norm quaternion constraints for each body, written at the top of the constraints vector
    """

    # Retrieve the thread indices (= world index, body index)
    wd_id, rb_id_loc = wp.tid()

    if wd_id < num_bodies.shape[0] and skip_computation[wd_id] == 0 and rb_id_loc < num_bodies[wd_id]:
        # Get overall body id
        rb_id_tot = first_body_id[wd_id] + rb_id_loc

        # Evaluate unit quaternion constraint
        q = wp.transform_get_rotation(bodies_q_i[rb_id_tot])
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
        num_joints: wp.array(dtype=wp.int32),  # Num joints per world
        first_joint_id: wp.array(dtype=wp.int32),  # First joint id per world
        joints_dof_type: wp.array(dtype=wp.int32),  # Joint dof type (i.e. revolute, spherical, ...)
        joints_act_type: wp.array(dtype=wp.int32),  # Joint actuation type (i.e. passive or actuated)
        joints_bid_B: wp.array(dtype=wp.int32),  # Joint base body id
        joints_bid_F: wp.array(dtype=wp.int32),  # Joint follower body id
        joints_X: wp.array(dtype=wp.mat33f),  # Joint frame (local axes, valid both on base and follower)
        joints_B_r_B: wp.array(dtype=wp.vec3f),  # Joint local position on base body
        joints_F_r_F: wp.array(dtype=wp.vec3f),  # Joint local position on follower body
        bodies_q_i: wp.array(dtype=wp.transformf),  # Body poses
        pos_control_transforms: wp.array(dtype=wp.transformf),  # Joint position-control transformation
        ct_full_to_red_map: wp.array2d(dtype=wp.int32),  # Map from full to reduced constraint id (per world)
        skip_computation: wp.array(dtype=wp.int32),  # Per-world flag to skip the computation
        # Outputs
        constraints: wp.array2d(dtype=wp.float32),  # Constraint vector per world
    ):
        """
        A kernel computing joint constraints with the log map formulation, first computing 6 constraints per
        joint (treating it as a fixed joint), then writing out the relevant subset of constraints (only along
        relevant directions) using a precomputed full to reduced map.

        Note: the log map formulation doesn't allow to formulate passive universal joints. If such joints are
        present, the right number of (incorrect) constraints is first written with the log map, then the result
        is overwritten in a second pass with the correct constraints.
        """

        # Retrieve the thread indices (= world index, joint index)
        wd_id, jt_id_loc = wp.tid()

        if wd_id < num_joints.shape[0] and skip_computation[wd_id] == 0 and jt_id_loc < num_joints[wd_id]:
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
                c_base = wp.transform_get_translation(bodies_q_i[base_id])
                q_base = wp.transform_get_rotation(bodies_q_i[base_id])
            follower_id = joints_bid_F[jt_id_tot]
            c_follower = wp.transform_get_translation(bodies_q_i[follower_id])
            q_follower = wp.transform_get_rotation(bodies_q_i[follower_id])

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
    num_bodies: wp.array(dtype=wp.int32),  # Num bodies per world
    first_body_id: wp.array(dtype=wp.int32),  # First body id per world
    bodies_q_i: wp.array(dtype=wp.transformf),  # Body poses
    skip_computation: wp.array(dtype=wp.int32),  # Per-world flag to skip the computation
    # Outputs
    constraints_jacobian: wp.array3d(dtype=wp.float32),  # Constraints Jacobian per world
):
    """
    A kernel computing the Jacobian of unit norm quaternion constraints for each body, written at the top of the
    constraints Jacobian
    """

    # Retrieve the thread indices (= world index, body index)
    wd_id, rb_id_loc = wp.tid()

    if wd_id < num_bodies.shape[0] and skip_computation[wd_id] == 0 and rb_id_loc < num_bodies[wd_id]:
        # Get overall body id
        rb_id_tot = first_body_id[wd_id] + rb_id_loc

        # Evaluate constraint Jacobian
        q = wp.transform_get_rotation(bodies_q_i[rb_id_tot])
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
        num_joints: wp.array(dtype=wp.int32),  # Num joints per world
        first_joint_id: wp.array(dtype=wp.int32),  # First joint id per world
        first_body_id: wp.array(dtype=wp.int32),  # First body id per world
        joints_dof_type: wp.array(dtype=wp.int32),  # Joint dof type (i.e. revolute, spherical, ...)
        joints_act_type: wp.array(dtype=wp.int32),  # Joint actuation type (i.e. passive or actuated)
        joints_bid_B: wp.array(dtype=wp.int32),  # Joint base body id
        joints_bid_F: wp.array(dtype=wp.int32),  # Joint follower body id
        joints_X: wp.array(dtype=wp.mat33f),  # Joint frame (local axes, valid both on base and follower)
        joints_B_r_B: wp.array(dtype=wp.vec3f),  # Joint local position on base body
        joints_F_r_F: wp.array(dtype=wp.vec3f),  # Joint local position on follower body
        bodies_q_i: wp.array(dtype=wp.transformf),  # Body poses
        pos_control_transforms: wp.array(dtype=wp.transformf),  # Joint position-control transformation
        ct_full_to_red_map: wp.array2d(dtype=wp.int32),  # Map from full to reduced constraint id (per world)
        skip_computation: wp.array(dtype=wp.int32),  # Per-world flag to skip the computation
        # Outputs
        constraints_jacobian: wp.array3d(dtype=wp.float32),  # Constraint Jacobian per world
    ):
        """
        A kernel computing the Jacobian of the joint constraints.
        The Jacobian is assumed to have already been filled with zeros, at least in the coefficients that
        are always zero due to joint connectivity.
        """

        # Retrieve the thread indices (= world index, joint index)
        wd_id, jt_id_loc = wp.tid()

        if wd_id < num_joints.shape[0] and skip_computation[wd_id] == 0 and jt_id_loc < num_joints[wd_id]:
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
                c_base = wp.transform_get_translation(bodies_q_i[base_id_tot])
                q_base = wp.transform_get_rotation(bodies_q_i[base_id_tot])
            follower_id_tot = joints_bid_F[jt_id_tot]
            c_follower = wp.transform_get_translation(bodies_q_i[follower_id_tot])
            q_follower = wp.transform_get_rotation(bodies_q_i[follower_id_tot])
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
    and variables (i.e. states) dimensions in the constraint vector, Jacobian, step vector etc.

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
        sparsity_pattern: wp.array3d(dtype=wp.float32),  # Jacobian sparsity pattern per world
        # Outputs
        pattern_T_pattern: wp.array3d(dtype=wp.float32),  # Jacobian^T * Jacobian sparsity pattern per world
    ):
        """
        A kernel computing the sparsity pattern of J^T * J given that of J, in each world
        More specifically, given an integer matrix of zeros and ones representing a sparsity pattern, multiply it by
        its transpose and clip values to [0, 1] to get the sparsity pattern of J^T * J
        Note: mostly redundant with _eval_jacobian_T_jacobian apart from the clipping, could possibly be removed
        (was initially written to take int32, but float32 is actually faster)
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
        constraints: wp.array2d(dtype=wp.float32),  # Constraint vector per world
        # Outputs
        max_constraint: wp.array(dtype=wp.float32),  # Max absolute constraint per world
    ):
        """
        A kernel computing the max absolute constraint from the constraints vector, in each world.
        max_constraint must be zero-initialized
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
        constraints_jacobian: wp.array3d(dtype=wp.float32),  # Constraint Jacobian per world
        skip_computation: wp.array(dtype=wp.int32),  # Per-world flag to skip the computation
        # Outputs
        jacobian_T_jacobian: wp.array3d(dtype=wp.float32),  # Jacobian^T * Jacobian per world
    ):
        """
        A kernel computing the matrix product J^T * J given the Jacobian J, in each world
        """
        wd_id, i, j = wp.tid()  # Thread indices (= world index, output tile indices)

        if (
            wd_id < jacobian_T_jacobian.shape[0]
            and skip_computation[wd_id] == 0
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
        constraints_jacobian: wp.array3d(dtype=wp.float32),  # Constraint Jacobian per world
        constraints: wp.array2d(dtype=wp.float32),  # Constraint vector per world
        skip_computation: wp.array(dtype=wp.int32),  # Per-world flag to skip the computation
        # Outputs
        jacobian_T_constraints: wp.array2d(dtype=wp.float32),  # Jacobian^T * Constraints per world
    ):
        """
        A kernel computing the matrix product J^T * C given the Jacobian J and the constraints vector C, in each world
        """
        wd_id, i = wp.tid()  # Thread indices (= world index, output tile index)

        if (
            wd_id < jacobian_T_constraints.shape[0]
            and skip_computation[wd_id] == 0
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
        constraints: wp.array2d(dtype=wp.float32),  # Constraint vector per world
        # Outputs
        merit_function_val: wp.array(dtype=wp.float32),  # Merit function value per world
    ):
        """
        A kernel computing the merit function, i.e. the least-squares error 1/2 * ||C||^2, from the constraints
        vector C, in each world
        merit_function_val must be zero-initialized
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
        step: wp.array2d(dtype=wp.float32),  # Step in variables per world
        grad: wp.array2d(dtype=wp.float32),  # Gradient w.r.t. state per world
        # Outputs
        merit_function_grad: wp.array(dtype=wp.float32),  # Merit function gradient per world
    ):
        """
        A kernel computing the merit function gradient w.r.t. line search step size, from the step direction
        and the gradient in state space (= dC_ds^T * C). This is simply the dot product between these two vectors.
        merit_function_grad must be zero-initialized
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
    grad: wp.array2d(dtype=wp.float32),  # Merit function gradient w.r.t. state per world
    # Outputs
    rhs: wp.array2d(dtype=wp.float32),  # Gauss-Newton right-hand side per world
):
    """
    A kernel computing rhs := -grad (where rhs has shape (num_worlds, num_states_max, 1))
    """
    wd_id, state_id_loc = wp.tid()  # Thread indices (= world index, state index)
    if wd_id < grad.shape[0] and state_id_loc < grad.shape[1]:
        rhs[wd_id, state_id_loc] = -grad[wd_id, state_id_loc]


@wp.kernel
def _eval_stepped_state(
    # Inputs
    num_bodies: wp.array(dtype=wp.int32),  # Num bodies per world
    first_body_id: wp.array(dtype=wp.int32),  # First body id per world
    rb_states_0_flat: wp.array(dtype=wp.float32),  # Previous state (for step size 0), flattened
    alpha: wp.array(dtype=wp.float32),  # Step size per world
    step: wp.array2d(dtype=wp.float32),  # Step in variables per world
    skip_computation: wp.array(dtype=wp.int32),  # Per-world flag to skip the computation
    # Outputs
    rb_states_alpha_flat: wp.array(dtype=wp.float32),  # New state (for step size alpha), flattened
):
    """
    A kernel computing states_alpha := states_0 + alpha * step
    """
    wd_id, state_id_loc = wp.tid()  # Thread indices (= world index, state index)
    rb_id_loc = state_id_loc // 7
    if wd_id < num_bodies.shape[0] and skip_computation[wd_id] == 0 and rb_id_loc < num_bodies[wd_id]:
        state_id_tot = 7 * first_body_id[wd_id] + state_id_loc
        rb_states_alpha_flat[state_id_tot] = rb_states_0_flat[state_id_tot] + alpha[wd_id] * step[wd_id, state_id_loc]


@wp.kernel
def _apply_line_search_step(
    # Inputs
    num_bodies: wp.array(dtype=wp.int32),  # Num bodies per world
    first_body_id: wp.array(dtype=wp.int32),  # First body id per world
    rb_states_alpha: wp.array(dtype=wp.transformf),  # Stepped states (line search result)
    line_search_success: wp.array(dtype=wp.int32),  # Per-world line search success flag
    # Outputs
    rb_states: wp.array(dtype=wp.transformf),  # Rigid body states
):
    """
    A kernel replacing the state with the line search result, in worlds where line search succeeded
    """
    wd_id, rb_id_loc = wp.tid()  # Thread indices (= world index, body index)
    if wd_id < num_bodies.shape[0] and line_search_success[wd_id] and rb_id_loc < num_bodies[wd_id]:
        rb_id_tot = first_body_id[wd_id] + rb_id_loc
        rb_states[rb_id_tot] = rb_states_alpha[rb_id_tot]


@wp.kernel
def _line_search_check(
    # Inputs
    val_0: wp.array(dtype=wp.float32),  # Merit function value at 0, per world
    grad_0: wp.array(dtype=wp.float32),  # Merit function gradient at 0, per world
    alpha: wp.array(dtype=wp.float32),  # Step size per world (in/out)
    val_alpha: wp.array(dtype=wp.float32),  # Merit function value at alpha, per world
    iteration: wp.array(dtype=wp.int32),  # Iteration count, per world
    max_iterations: wp.array(dtype=wp.int32, shape=(1,)),  # Max iterations
    # Outputs
    line_search_success: wp.array(dtype=wp.int32),  # Convergence per world
    line_search_loop_condition: wp.array(dtype=wp.int32, shape=(1,)),  # Loop condition
):
    """
    A kernel checking the sufficient decrease condition in line search in each world, and updating the looping
    condition (zero if max iterations reached, or all worlds successful)
    line_search_loop_condition must be zero-initialized
    """
    wd_id = wp.tid()  # Thread index (= world index)
    if wd_id < val_0.shape[0] and line_search_success[wd_id] == 0:
        iteration[wd_id] += 1
        line_search_success[wd_id] = int(
            wp.isfinite(val_alpha[wd_id]) and val_alpha[wd_id] <= val_0[wd_id] + 1e-4 * alpha[wd_id] * grad_0[wd_id]
        )
        continue_loop_world = iteration[wd_id] < max_iterations[0] and not line_search_success[wd_id]
        if continue_loop_world:
            alpha[wd_id] *= 0.5
        wp.atomic_max(line_search_loop_condition, 0, int(continue_loop_world))


@wp.kernel
def _newton_check(
    # Inputs
    max_constraint: wp.array(dtype=wp.float32),  # Max absolute constraint per world
    tolerance: wp.array(dtype=wp.float32, shape=(1,)),  # Tolerance on max constraint
    iteration: wp.array(dtype=wp.int32),  # Iteration count, per world
    max_iterations: wp.array(dtype=wp.int32, shape=(1,)),  # Max iterations
    line_search_success: wp.array(dtype=wp.int32),  # Per-world line search success flag
    # Outputs
    newton_success: wp.array(dtype=wp.int32),  # Convergence per world
    newton_skip: wp.array(dtype=wp.int32),  # Flag to stop iterating per world
    newton_loop_condition: wp.array(dtype=wp.int32, shape=(1,)),  # Loop condition
):
    """
    A kernel checking the convergence (max constraint vs tolerance) in each world, and updating the looping
    condition (zero if max iterations reached, or all worlds successful)
    newton_loop_condition must be zero-initialized
    """
    wd_id = wp.tid()  # Thread index (= world index)
    if wd_id < max_constraint.shape[0] and newton_skip[wd_id] == 0:
        iteration[wd_id] += 1
        max_constraint_wd = max_constraint[wd_id]
        is_finite = wp.isfinite(max_constraint_wd)
        newton_success[wd_id] = int(is_finite and max_constraint_wd <= tolerance[0])
        newton_continue_world = int(
            iteration[wd_id] < max_iterations[0]
            and not newton_success[wd_id]
            and is_finite  # Abort when encountering NaN / Inf values
            and line_search_success[wd_id]
        )  # Abort in case of line search failure
        newton_skip[wd_id] = 1 - newton_continue_world
        wp.atomic_max(newton_loop_condition, 0, newton_continue_world)


###
# Classes
###


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

    iterations: np.ndarray(dtype="int32")

    max_constraints: np.ndarray(dtype="float32")

    success: np.ndarray(dtype="int32")


class ForwardKinematicsSolver:
    """
    Forward Kinematics solver class
    """

    def __init__(self, model: Model, TILE_SIZE_CTS: wp.int32 = 8, TILE_SIZE_VRS: wp.int32 = 8):
        """
        Initializes the solver to solve forward kinematics for a given model.
        Note: will use the same device as the model
        """

        self.model: Model | None = None
        """Underlying model"""

        self.device: Devicelike = None
        """Device for data allocations"""

        self.linear_solver: SemiSparseBlockCholeskySolverBatched | None = None
        """Semi-sparse Cholesky solver for the J^T * J linear system"""

        self.graph: wp.Graph | None = None
        """Cuda graph for the forward kinematics solve"""

        # Note: many other data members below, for internal use only (currently not documented)

        # Initialize model and device
        self.model = model
        self.device = model.device

        # Retrieve / compute dimensions - Worlds
        self.num_worlds = self.model.size.num_worlds  # For convenience

        # Retrieve / compute dimensions - Bodies
        num_bodies = model.info.num_bodies.numpy()  # Number of bodies per world
        first_body_id = np.concatenate(([0], num_bodies.cumsum()))  # Index of first body per world
        self.num_bodies_max = model.size.max_of_num_bodies  # Max number of bodies across worlds

        # Retrieve / compute dimensions - Joints
        num_joints = model.info.num_joints.numpy()  # Number of joints per world
        first_joint_id = np.concatenate(([0], num_joints.cumsum()))  # Index of first joint per world
        self.num_joints_max = model.size.max_of_num_joints  # Max number of joints across worlds

        # Retrieve / compute dimensions - Joint DoFs
        num_dofs = model.info.num_joint_dofs.numpy()  # Number of joint dofs per world
        first_joint_dof = np.concatenate(([0], num_dofs.cumsum()))  # Index of first dof per world
        joint_dof_offsets = model.joints.dofs_offset.numpy()  # Dofs offsets, among dofs of a single world only
        for wd_id in range(self.num_worlds):  # Convert into dofs offsets among all dofs
            joint_dof_offsets[first_joint_id[wd_id] : first_joint_id[wd_id + 1]] += first_joint_dof[wd_id]

        # Retrieve / compute dimensions - States
        num_states = 7 * num_bodies  # Number of body states per world
        self.num_states_tot = 7 * model.size.sum_of_num_bodies  # Number of body states for the whole model
        self.num_states_max = 7 * self.num_bodies_max  # Max state dimension across worlds

        # Retrieve / compute dimensions - Constraints
        num_constraints = num_bodies.copy()  # Number of kinematic constraints per world (unit quat. + joints)
        has_universal_joints = False  # Whether the model has a least one passive universal joint
        constraint_full_to_red_map = -1 * np.ones((self.num_worlds, 6 * self.num_joints_max), dtype="int32")
        act_types = model.joints.act_type.numpy()
        dof_types = model.joints.dof_type.numpy()
        for wd_id in range(self.num_worlds):
            ct_count = num_constraints[wd_id]
            for jt_id_loc in range(num_joints[wd_id]):
                jt_id_tot = first_joint_id[wd_id] + jt_id_loc  # Joint id among all joints
                act_type = act_types[jt_id_tot]
                if act_type == JointActuationType.FORCE:  # Actuator: select all six constraints
                    for i in range(6):
                        constraint_full_to_red_map[wd_id, 6 * jt_id_loc + i] = ct_count + i
                    ct_count += 6
                else:
                    dof_type = dof_types[jt_id_tot]
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
        self.num_tiles_constraints = (self.num_constraints_max + TILE_SIZE_CTS - 1) // TILE_SIZE_CTS
        self.num_tiles_states = (self.num_states_max + TILE_SIZE_VRS - 1) // TILE_SIZE_VRS

        # Data allocation or transfer from numpy to warp
        with wp.ScopedDevice(self.device):
            # Dimensions
            self.first_body_id = wp.from_numpy(first_body_id, dtype=wp.int32)
            self.first_joint_id = wp.from_numpy(first_joint_id, dtype=wp.int32)
            self.joint_dof_offsets = wp.from_numpy(joint_dof_offsets, dtype=wp.int32)
            self.num_states = wp.from_numpy(num_states, dtype=wp.int32)
            self.constraint_full_to_red_map = wp.from_numpy(constraint_full_to_red_map, dtype=wp.int32)

            # Line search
            self.max_line_search_iterations = wp.array(dtype=wp.int32, shape=(1,))  # Max iterations
            self.line_search_iteration = wp.array(dtype=wp.int32, shape=(self.num_worlds,))  # Iteration count
            self.line_search_loop_condition = wp.array(dtype=wp.int32, shape=(1,))  # Loop condition
            self.line_search_success = wp.array(dtype=wp.int32, shape=(self.num_worlds,))  # Convergence, per world
            self.val_0 = wp.array(dtype=wp.float32, shape=(self.num_worlds,))  # Merit function value at 0, per world
            self.grad_0 = wp.array(
                dtype=wp.float32, shape=(self.num_worlds,)
            )  # Merit function gradient at 0, per world
            self.alpha = wp.array(dtype=wp.float32, shape=(self.num_worlds,))  # Step size, per world
            self.rb_states_alpha = wp.array(dtype=wp.transformf, shape=(model.size.sum_of_num_bodies,))  # New state
            self.val_alpha = wp.array(dtype=wp.float32, shape=(self.num_worlds,))  # New merit function value, per world

            # Gauss-Newton
            self.max_newton_iterations = wp.array(dtype=wp.int32, shape=(1,))  # Max iterations
            self.newton_iteration = wp.array(dtype=wp.int32, shape=(self.num_worlds,))  # Iteration count
            self.newton_loop_condition = wp.array(dtype=wp.int32, shape=(1,))  # Loop condition
            self.newton_success = wp.array(dtype=wp.int32, shape=(self.num_worlds,))  # Convergence per world
            self.newton_skip = wp.array(dtype=wp.int32, shape=(self.num_worlds,))  # Flag to stop iterating per world
            self.tolerance = wp.array(dtype=wp.float32, shape=(1,))  # Tolerance on max constraint
            self.joints_q_j = wp.array(dtype=wp.float32, shape=(model.size.sum_of_num_joint_coords,))  # Coordinates
            self.pos_control_transforms = wp.array(
                dtype=wp.transformf, shape=(model.size.sum_of_num_joints,)
            )  # Position-control transformations at joints
            self.rb_states = wp.array(dtype=wp.transformf, shape=(model.size.sum_of_num_bodies,))  # Rigid body poses
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
        ) = create_tile_based_kernels(TILE_SIZE_CTS, TILE_SIZE_VRS)

        # Compute sparsity pattern and initialize linear solver (running symbolic factorization)

        # Jacobian sparsity pattern
        base_ids = model.joints.bid_B.numpy()
        follower_ids = model.joints.bid_F.numpy()
        sparsity_pattern = np.zeros((self.num_worlds, self.num_constraints_max, self.num_states_max), dtype=int)
        for wd_id in range(self.num_worlds):
            for rb_id_loc in range(num_bodies[wd_id]):
                sparsity_pattern[wd_id, rb_id_loc, 7 * rb_id_loc + 3 : 7 * rb_id_loc + 7] = 1
            for jt_id_loc in range(num_joints[wd_id]):
                jt_id_tot = first_joint_id[wd_id] + jt_id_loc
                base_id_tot = base_ids[jt_id_tot]
                follower_id_tot = follower_ids[jt_id_tot]
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

    def _eval_position_control_transformations(
        self, joints_q_j: wp.array(dtype=wp.float32), pos_control_transforms: wp.array(dtype=wp.transformf)
    ):
        """
        Internal evaluator for position control transformations, from joint coordinates
        """
        wp.launch(
            _eval_position_control_transformations,
            dim=(self.model.size.sum_of_num_joints,),
            inputs=[
                self.model.joints.dof_type,
                self.model.joints.act_type,
                self.joint_dof_offsets,
                self.model.joints.X_j,
                joints_q_j,
                pos_control_transforms,
            ],
        )
        return pos_control_transforms

    def _eval_kinematic_constraints(
        self,
        bodies_q_i: wp.array(dtype=wp.float32),
        pos_control_transforms: wp.array(dtype=wp.transformf),
        skip_computation: wp.array(dtype=wp.int32),
        constraints: wp.array2d(dtype=wp.float32),
    ):
        """
        Internal evaluator for the kinematic constraints vector, from body states and position-control transformations
        """

        # Evaluate unit norm quaternion constraints
        wp.launch(
            _eval_unit_quaternion_constraints,
            dim=(self.num_worlds, self.num_bodies_max),
            inputs=[self.model.info.num_bodies, self.first_body_id, bodies_q_i, skip_computation, constraints],
        )
        # Evaluate joint constraints
        wp.launch(
            self._eval_joint_constraints_kernel,
            dim=(self.num_worlds, self.num_joints_max),
            inputs=[
                self.model.info.num_joints,
                self.first_joint_id,
                self.model.joints.dof_type,
                self.model.joints.act_type,
                self.model.joints.bid_B,
                self.model.joints.bid_F,
                self.model.joints.X_j,
                self.model.joints.B_r_Bj,
                self.model.joints.F_r_Fj,
                bodies_q_i,
                pos_control_transforms,
                self.constraint_full_to_red_map,
                skip_computation,
                constraints,
            ],
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
        )

    def _eval_kinematic_constraints_jacobian(
        self,
        bodies_q_i: wp.array(dtype=wp.transformf),
        pos_control_transforms: wp.array(dtype=wp.transformf),
        skip_computation: wp.array(dtype=wp.int32),
        constraints_jacobian: wp.array3d(dtype=wp.float32),
    ):
        """
        Internal evaluator for the kinematic constraints Jacobian with respect to body states, from body states
        and position-control transformations
        """

        # Evaluate unit norm quaternion constraints Jacobian
        wp.launch(
            _eval_unit_quaternion_constraints_jacobian,
            dim=(self.num_worlds, self.num_bodies_max),
            inputs=[self.model.info.num_bodies, self.first_body_id, bodies_q_i, skip_computation, constraints_jacobian],
        )

        # Evaluate joint constraints Jacobian
        wp.launch(
            self._eval_joint_constraints_jacobian_kernel,
            dim=(self.num_worlds, self.num_joints_max),
            inputs=[
                self.model.info.num_joints,
                self.first_joint_id,
                self.first_body_id,
                self.model.joints.dof_type,
                self.model.joints.act_type,
                self.model.joints.bid_B,
                self.model.joints.bid_F,
                self.model.joints.X_j,
                self.model.joints.B_r_Bj,
                self.model.joints.F_r_Fj,
                bodies_q_i,
                pos_control_transforms,
                self.constraint_full_to_red_map,
                skip_computation,
                constraints_jacobian,
            ],
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
        )

    def _run_line_search_iteration(self):
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
                wp.array(ptr=self.rb_states.ptr, dtype=wp.float32, shape=(self.num_states_tot,)),
                self.alpha,
                self.step,
                self.line_search_success,
                wp.array(ptr=self.rb_states_alpha.ptr, dtype=wp.float32, shape=(self.num_states_tot,)),
            ],
        )

        # Evaluate new constraints and merit function (least squares norm of constraints)
        self._eval_kinematic_constraints(
            self.rb_states_alpha, self.pos_control_transforms, self.line_search_success, self.constraints
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
                self.line_search_loop_condition,
            ],
        )

    def _run_newton_iteration(self):
        """
        Internal function running one iteration of Gauss-Newton. Assumes the constraints vector to be already
        up-to-date (because we will already have checked convergence before the first loop iteration)
        """
        # Evaluate constraints Jacobian
        self._eval_kinematic_constraints_jacobian(
            self.rb_states, self.pos_control_transforms, self.newton_skip, self.jacobian
        )

        # Evaluate Gauss-Newton left-hand side (J^T * J) and right-hand side (-J^T * C)
        wp.launch_tiled(
            self._eval_jacobian_T_jacobian_kernel,
            dim=(self.num_worlds, self.num_tiles_states, self.num_tiles_states),
            inputs=[self.jacobian, self.newton_skip, self.lhs],
            block_dim=64,
        )
        wp.launch_tiled(
            self._eval_jacobian_T_constraints_kernel,
            dim=(self.num_worlds, self.num_tiles_states),
            inputs=[self.jacobian, self.constraints, self.newton_skip, self.grad],
            block_dim=64,
        )
        wp.launch(_eval_rhs, dim=(self.num_worlds, self.num_states_max), inputs=[self.grad, self.rhs])

        # Compute step (system solve)
        self.linear_solver.factorize(self.lhs, self.num_states, self.newton_skip)
        self.linear_solver.solve(
            self.rhs.reshape((self.num_worlds, self.num_states_max, 1)),
            self.step.reshape((self.num_worlds, self.num_states_max, 1)),
            self.newton_skip,
        )

        # Line search
        self.line_search_iteration.zero_()
        self.line_search_success.zero_()
        self.line_search_loop_condition.fill_(1)
        self._eval_merit_function(self.constraints, self.val_0)
        self._eval_merit_function_gradient(self.step, self.grad, self.grad_0)
        self.alpha.fill_(1.0)
        wp.capture_while(self.line_search_loop_condition, lambda: self._run_line_search_iteration())

        # Apply line search step and update max constraint
        wp.launch(
            _apply_line_search_step,
            dim=(self.num_worlds, self.num_bodies_max),
            inputs=[
                self.model.info.num_bodies,
                self.first_body_id,
                self.rb_states_alpha,
                self.line_search_success,
                self.rb_states,
            ],
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
                self.newton_skip,
                self.newton_loop_condition,
            ],
        )

    def _read_state(self, state: ModelData, reset_state):
        """
        Internal function copying the joint positions and rigid body poses from the ModelData into internal arrays
        """
        # Read current joint coordinates
        wp.copy(self.joints_q_j, state.joints.q_j)

        # Initialize body states (initial guess)
        if reset_state:
            wp.copy(self.rb_states, self.model.bodies.q_i_0)
        else:
            wp.copy(self.rb_states, state.bodies.q_i)

    def _write_state(self, state: ModelData):
        """
        Internal function writing out the computed rigid body poses into the ModelData
        """
        wp.copy(state.bodies.q_i, self.rb_states)

    def _run_fk_solve(self):
        """
        Internal function running the core FK solve
        """
        # Compute position control transforms (independent of state, depends on controls only)
        self._eval_position_control_transformations(self.joints_q_j, self.pos_control_transforms)

        # Reset iteration count and success/continuation flags
        self.newton_iteration.fill_(-1)  # The initial loop condition check will increment this to zero
        self.newton_success.zero_()
        self.newton_skip.zero_()

        # Evaluate constraints, and initialize loop condition (might not even need to loop)
        self._eval_kinematic_constraints(
            self.rb_states, self.pos_control_transforms, self.newton_skip, self.constraints
        )
        self._eval_max_constraint(self.constraints, self.max_constraint)
        self.newton_loop_condition.zero_()
        self.line_search_success.fill_(1)  # Newton check will abort in case of line search failure
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
                self.newton_skip,
                self.newton_loop_condition,
            ],
        )

        # Main loop
        wp.capture_while(self.newton_loop_condition, lambda: self._run_newton_iteration())

    ###
    # Exposed functions (overall solve_fk() function + constraints (Jacobian) evaluators for debugging)
    ###

    def eval_position_control_transformations(self, state: ModelData):
        """
        Evaluates and returns position control transformations for a model in a given state
        (intermediary quantity needed for kinematic constraints/Jacobian evaluation)
        """
        assert state.joints.q_j.device == self.device

        pos_control_transforms = wp.array(
            dtype=wp.transformf, shape=(self.model.size.sum_of_num_joints,), device=self.device
        )
        self._eval_position_control_transformations(state.joints.q_j, pos_control_transforms)
        return pos_control_transforms

    def eval_kinematic_constraints(
        self, state: ModelData, pos_control_transforms: wp.array(dtype=wp.transformf) | None = None
    ):
        """
        Evaluates and returns the kinematic constraints vector for a model in a given state
        """
        assert state.bodies.q_i.device == self.device

        if pos_control_transforms is None:
            pos_control_transforms = self.eval_position_control_transformations(state)

        constraints = wp.zeros(
            dtype=wp.float32,
            shape=(
                self.num_worlds,
                self.num_constraints_max,
            ),
            device=self.device,
        )
        skip_computation = wp.zeros(dtype=wp.int32, shape=(self.num_worlds,), device=self.device)
        self._eval_kinematic_constraints(state.bodies.q_i, pos_control_transforms, skip_computation, constraints)
        return constraints

    def eval_kinematic_constraints_jacobian(
        self, state: ModelData, pos_control_transforms: wp.array(dtype=wp.transformf) | None = None
    ):
        """
        Evaluates and returns the kinematic constraints Jacobian (w.r.t. body states) for a model in a given state
        """
        assert state.bodies.q_i.device == self.device

        if pos_control_transforms is None:
            pos_control_transforms = self.eval_position_control_transformations(state)

        constraints_jacobian = wp.zeros(
            dtype=wp.float32, shape=(self.num_worlds, self.num_constraints_max, self.num_states_max), device=self.device
        )
        skip_computation = wp.zeros(dtype=wp.int32, shape=(self.num_worlds,), device=self.device)
        self._eval_kinematic_constraints_jacobian(
            state.bodies.q_i, pos_control_transforms, skip_computation, constraints_jacobian
        )
        return constraints_jacobian

    def run_fk_solve(self, state: ModelData, reset_state: bool = True):
        """
        Graph-capturable function solving forward kinematics with Gauss-Newton. More specifically, solves for the
        rigid body poses satisfying kinematic constraints, given the current actuator generalized coordinates
        (i.e. position-control inputs)

        Parameters
        ----------
        state : ModelData
            provides the generalized coordinates to solve for, and the initial guess for body states if reset_state
            is False. The computed rigid body states will be updated into this state.
            Must be allocated on the same device as the underlying model.
        reset_state : bool, optional
            whether to reset the state to initial states, to use as initial guess (default: True). This parameter
            will be baked into the graph in case of graph capture.
        """
        # Initialize joint positions and rigid body poses
        self._read_state(state, reset_state)

        # Solve forward kinematics (working on internal arrays)
        self._run_fk_solve()

        # Write out result
        self._write_state(state)

    def solve_fk(
        self,
        state: ModelData,
        reset_state: bool = True,
        max_newton_iterations: wp.int32 = 30,
        max_line_search_iterations: wp.int32 = 20,
        tolerance: wp.float32 = 1e-6,
        use_graph: bool = True,
        verbose: bool = False,
        return_status: bool = False,
    ):
        """
        Convenience function (non graph-capturable) solving forward kinematics with Gauss-Newton. More specifically,
        solves for the rigid body poses satisfying kinematic constraints, given the current actuator generalized
        coordinates (i.e. position-control inputs)

        Parameters
        ----------
        state : ModelData
            provides the generalized coordinates to solve for, and the initial guess for body states if reset_state
            is False. The computed rigid body states will be updated into this state.
            Must be allocated on the same device as the underlying model.
        reset_state : bool, optional
            whether to reset the state to initial states, to use as initial guess (default: True)
        max_newton_iterations : int, optional
            maximal number of Gauss-Newton iterations (default: 30)
        max_line_search_iterations : int, optional
            maximal line search iterations in the inner loop (default: 20)
        tolerance : float, optional
            maximal absolute kinematic constraint value that is acceptable at the solution (default: 1e-6)
        use_graph : bool, optional
            whether to use graph capture internally to accelerate multiple calls to this function. Can be turned
            off for profiling individual kernels (default: True)
        verbose : bool, optional
            whether to write a status message at the end (default: False)
        return_status : bool, optional
            whether to return the detailed solver status (default: False)

        Returns
        -------
        solver_status : ForwardKinematicsSolverStatus, optional
            the detailed solver status with success flag, number of iterations and constraint residual per world
        """

        # Initialize joint positions and rigid body poses
        self._read_state(state, reset_state)

        # Read solver parameters
        self.max_newton_iterations.fill_(max_newton_iterations)
        self.max_line_search_iterations.fill_(max_line_search_iterations)
        self.tolerance.fill_(tolerance)

        # Run solve (with or without graph)
        if use_graph:
            if self.graph is None:
                wp.capture_begin(self.device)
                self._run_fk_solve()
                self.graph = wp.capture_end()
            wp.capture_launch(self.graph)
        else:
            self._run_fk_solve()

        # Status message
        if verbose or return_status:
            success = self.newton_success.numpy()
            iterations = self.newton_iteration.numpy()
            max_constraints = self.max_constraint.numpy()
            if verbose:
                sys.__stdout__.write(f"Newton success for {success.sum()}/{self.num_worlds} worlds; ")
                sys.__stdout__.write(f"num iterations={iterations.max()}; ")
                sys.__stdout__.write(f"max constraint={max_constraints.max()}\n")

        # Write out result
        self._write_state(state)

        # Return solver status
        if return_status:
            return ForwardKinematicsSolverStatus(
                iterations=iterations, max_constraints=max_constraints, success=success
            )
