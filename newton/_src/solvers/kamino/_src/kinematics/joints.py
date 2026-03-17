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
KAMINO: Kinematics: Joints
"""

from __future__ import annotations

from functools import cache

import warp as wp

from ..core.data import DataKamino
from ..core.joints import JointActuationType, JointCorrectionMode, JointDoFType
from ..core.math import (
    FLOAT32_MAX,
    TWO_PI,
    quat_log,
    quat_to_euler_xyz,
    quat_to_vec4,
    quat_twist_angle,
    screw,
    screw_angular,
    screw_linear,
    squared_norm,
)
from ..core.model import ModelKamino
from ..core.types import (
    float32,
    int32,
    mat33f,
    quatf,
    transformf,
    vec1f,
    vec2f,
    vec3f,
    vec4f,
    vec6f,
    vec7f,
)

###
# Module interface
###

__all__ = [
    "compute_joints_data",
    "extract_actuators_state_from_joints",
    "extract_joints_state_from_actuators",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###


DEFAULT_LIMIT_V1F = vec1f(FLOAT32_MAX)
DEFAULT_LIMIT_V2F = vec2f(FLOAT32_MAX)
DEFAULT_LIMIT_V3F = vec3f(FLOAT32_MAX)
DEFAULT_LIMIT_V4F = vec4f(FLOAT32_MAX)
DEFAULT_LIMIT_V7F = vec7f(FLOAT32_MAX)


###
# Functions - Coordinate Correction
###


@wp.func
def correct_rotational_coord(q_j_in: float32, q_j_ref: float32 = 0.0, q_j_limit: float32 = FLOAT32_MAX) -> float32:
    """
    Corrects a rotational joint coordinate to be as close as possible to a reference coordinate.
    """
    q_j_in += wp.round((q_j_ref - q_j_in) / TWO_PI) * TWO_PI
    q_j_in = wp.mod(q_j_in, q_j_limit)
    return q_j_in


@wp.func
def correct_quat_vector_coord(q_j_in: vec4f, q_j_ref: vec4f) -> vec4f:
    """
    Corrects a quaternion joint coordinate to be as close as possible to a reference coordinate.

    This ensures that the quaternion `q_j_in` is chosen such that it is
    closer to the reference quaternion `q_j_ref`, accounting for the fact
    that quaternions `q` and `-q` represent the same rotation.
    """
    if squared_norm(q_j_in + q_j_ref) < squared_norm(q_j_in - q_j_ref):
        q_j_in *= -1.0
    return q_j_in


@wp.func
def correct_joint_coord_free(q_j_in: vec7f, q_j_ref: vec7f, q_j_limit: vec7f = DEFAULT_LIMIT_V7F) -> vec7f:
    """Corrects the orientation quaternion coordinate of a free joint."""
    q_j_in[0:4] = correct_quat_vector_coord(q_j_in[0:4], q_j_ref[0:4])
    return q_j_in


@wp.func
def correct_joint_coord_revolute(q_j_in: vec1f, q_j_ref: vec1f, q_j_limit: vec1f = DEFAULT_LIMIT_V1F) -> vec1f:
    """Corrects the rotational joint coordinate."""
    q_j_in[0] = correct_rotational_coord(q_j_in[0], q_j_ref[0], q_j_limit[0])
    return q_j_in


@wp.func
def correct_joint_coord_prismatic(q_j_in: vec1f, q_j_ref: vec1f, q_j_limit: vec1f = DEFAULT_LIMIT_V1F) -> vec1f:
    """No correction needed for prismatic coordinates."""
    return q_j_in


@wp.func
def correct_joint_coord_cylindrical(q_j_in: vec2f, q_j_ref: vec2f, q_j_limit: vec2f = DEFAULT_LIMIT_V2F) -> vec2f:
    """Corrects only the rotational joint coordinate."""
    q_j_in[1] = correct_rotational_coord(q_j_in[1], q_j_ref[1], q_j_limit[1])
    return q_j_in


@wp.func
def correct_joint_coord_universal(q_j_in: vec2f, q_j_ref: vec2f, q_j_limit: vec2f = DEFAULT_LIMIT_V2F) -> vec2f:
    """Corrects each of the two rotational joint coordinates individually."""
    q_j_in[0] = correct_rotational_coord(q_j_in[0], q_j_ref[0], q_j_limit[0])
    q_j_in[1] = correct_rotational_coord(q_j_in[1], q_j_ref[1], q_j_limit[1])
    return q_j_in


@wp.func
def correct_joint_coord_spherical(q_j_in: vec4f, q_j_ref: vec4f, q_j_limit: vec4f = DEFAULT_LIMIT_V4F) -> vec4f:
    """Corrects a quaternion joint coordinate to be as close as possible to a reference."""
    return correct_quat_vector_coord(q_j_in, q_j_ref)


@wp.func
def correct_joint_coord_gimbal(q_j_in: vec3f, q_j_ref: vec3f, q_j_limit: vec3f = DEFAULT_LIMIT_V3F) -> vec3f:
    """Corrects each of the XYZ Euler angles individually."""
    q_j_in[0] = correct_rotational_coord(q_j_in[0], q_j_ref[0], q_j_limit[0])
    q_j_in[1] = correct_rotational_coord(q_j_in[1], q_j_ref[1], q_j_limit[1])
    q_j_in[2] = correct_rotational_coord(q_j_in[2], q_j_ref[2], q_j_limit[2])
    return q_j_in


@wp.func
def correct_joint_coord_cartesian(q_j_in: vec3f, q_j_ref: vec3f, q_j_limit: vec3f = DEFAULT_LIMIT_V3F) -> vec3f:
    """No correction needed for Cartesian coordinates."""
    return q_j_in


def get_joint_coord_correction_function(dof_type: JointDoFType):
    """
    Retrieves the function to correct joint
    coordinates based on the type of joint DoF.
    """
    if dof_type == JointDoFType.FREE:
        return correct_joint_coord_free
    elif dof_type == JointDoFType.REVOLUTE:
        return correct_joint_coord_revolute
    elif dof_type == JointDoFType.PRISMATIC:
        return correct_joint_coord_prismatic
    elif dof_type == JointDoFType.CYLINDRICAL:
        return correct_joint_coord_cylindrical
    elif dof_type == JointDoFType.UNIVERSAL:
        return correct_joint_coord_universal
    elif dof_type == JointDoFType.SPHERICAL:
        return correct_joint_coord_spherical
    elif dof_type == JointDoFType.GIMBAL:
        return correct_joint_coord_gimbal
    elif dof_type == JointDoFType.CARTESIAN:
        return correct_joint_coord_cartesian
    elif dof_type == JointDoFType.FIXED:
        return None
    else:
        raise ValueError(f"Unknown joint DoF type: {dof_type}")


###
# Functions - Coordinate Mappings
###


@wp.func
def map_to_joint_coords_free(j_r_j: vec3f, j_q_j: quatf) -> vec7f:
    """Returns the full 7D representation of joint pose (3D translation + 4D rotation)."""
    # TODO: Is there a more efficient way to construct a vec7f?
    return vec7f(j_r_j[0], j_r_j[1], j_r_j[2], j_q_j.x, j_q_j.y, j_q_j.z, j_q_j.w)


@wp.func
def map_to_joint_coords_revolute(j_r_j: vec3f, j_q_j: quatf) -> vec1f:
    """Returns the 1D rotation angle about the local X-axis."""
    # Measure rotation around the x-axis only
    axis = vec3f(1.0, 0.0, 0.0)
    return vec1f(quat_twist_angle(j_q_j, axis))


@wp.func
def map_to_joint_coords_prismatic(j_r_j: vec3f, j_q_j: quatf) -> vec1f:
    """Returns the 1D translation distance along the local X-axis."""
    return vec1f(j_r_j[0])


@wp.func
def map_to_joint_coords_cylindrical(j_r_j: vec3f, j_q_j: quatf) -> vec2f:
    """Returns the 2D vector of translation and rotation about the local X-axis."""
    j_p_j = quat_log(j_q_j)
    return vec2f(j_r_j[0], j_p_j[0])


@wp.func
def map_to_joint_coords_universal(j_r_j: vec3f, j_q_j: quatf) -> vec2f:
    """Returns the 2D vector of joint angles for the two revolute DoFs."""
    # TODO: Fix this so that the order of rotations is consistent
    j_theta_j = quat_log(j_q_j)
    return j_theta_j[0:2]


@wp.func
def map_to_joint_coords_spherical(j_r_j: vec3f, j_q_j: quatf) -> vec4f:
    """Returns the 4D unit-quaternion representing the joint rotation."""
    return quat_to_vec4(j_q_j)


@wp.func
def map_to_joint_coords_gimbal(j_r_j: vec3f, j_q_j: quatf) -> vec3f:
    """Returns the 3D XYZ Euler angles (roll, pitch, yaw)."""
    # How can we make this safer?
    return quat_to_euler_xyz(j_q_j)


@wp.func
def map_to_joint_coords_cartesian(j_r_j: vec3f, j_q_j: quatf) -> vec3f:
    """Returns the 3D translational."""
    return j_r_j


def get_joint_coords_mapping_function(dof_type: JointDoFType):
    """
    Retrieves the function to map joint relative poses to
    joint coordinates based on the type of joint DoF.
    """
    if dof_type == JointDoFType.FREE:
        return map_to_joint_coords_free
    elif dof_type == JointDoFType.REVOLUTE:
        return map_to_joint_coords_revolute
    elif dof_type == JointDoFType.PRISMATIC:
        return map_to_joint_coords_prismatic
    elif dof_type == JointDoFType.CYLINDRICAL:
        return map_to_joint_coords_cylindrical
    elif dof_type == JointDoFType.UNIVERSAL:
        return map_to_joint_coords_universal
    elif dof_type == JointDoFType.SPHERICAL:
        return map_to_joint_coords_spherical
    elif dof_type == JointDoFType.GIMBAL:
        return map_to_joint_coords_gimbal
    elif dof_type == JointDoFType.CARTESIAN:
        return map_to_joint_coords_cartesian
    elif dof_type == JointDoFType.FIXED:
        return None
    else:
        raise ValueError(f"Unknown joint DoF type: {dof_type}")


###
# Functions - Constraint residual
###


@wp.func
def joint_constraint_angular_residual_free(j_q_j: quatf) -> vec3f:
    return vec3f(0.0, 0.0, 0.0)


@wp.func
def joint_constraint_angular_residual_revolute(j_q_j: quatf) -> vec3f:
    """Returns the joint constraint residual for a revolute joint that rotates about the local X-axis."""
    # x-axis attached to the base body
    j_x_B = vec3f(1.0, 0.0, 0.0)

    # x-axis attached to the follower body, expressed in the joint frame on the base body.
    j_x_F = wp.quat_rotate(j_q_j, j_x_B)

    # Residual vector = sin(res_angle) * axis, where axis only has y, and z components.
    # For small angles this is equal to res_angle * axis
    return wp.cross(j_x_B, j_x_F)


@wp.func
def joint_constraint_angular_residual_universal(j_q_j: quatf) -> vec3f:
    """Returns the joint constraint residual for a universal joint."""
    # TODO: Fix, using log-based residual as a placeholder
    return quat_log(j_q_j)


@wp.func
def joint_constraint_angular_residual_fixed(j_q_j: quatf) -> vec3f:
    """Returns the joint constraint residual for a fixed joint."""
    return quat_log(j_q_j)


def get_joint_constraint_angular_residual_function(dof_type: JointDoFType):
    """
    Retrieves the function that computes the joint constraint residual as a 6D local vector
    """

    # Use the fixed joint residual as the generic implementation.
    if dof_type == JointDoFType.FREE:
        return joint_constraint_angular_residual_free
    elif dof_type == JointDoFType.REVOLUTE:
        return joint_constraint_angular_residual_revolute
    elif dof_type == JointDoFType.PRISMATIC:
        return joint_constraint_angular_residual_fixed
    elif dof_type == JointDoFType.CYLINDRICAL:
        return joint_constraint_angular_residual_revolute
    elif dof_type == JointDoFType.UNIVERSAL:
        return joint_constraint_angular_residual_universal
    elif dof_type == JointDoFType.SPHERICAL:
        return joint_constraint_angular_residual_free
    elif dof_type == JointDoFType.GIMBAL:
        return joint_constraint_angular_residual_free
    elif dof_type == JointDoFType.CARTESIAN:
        return joint_constraint_angular_residual_fixed
    elif dof_type == JointDoFType.FIXED:
        return joint_constraint_angular_residual_fixed
    else:
        raise ValueError(f"Unknown joint DoF type: {dof_type}")


###
# Functions - State Writes
###


def make_typed_write_joint_data(dof_type: JointDoFType, correction: JointCorrectionMode = JointCorrectionMode.TWOPI):
    """
    Generates functions to store the joint state according to the
    constraint and DoF dimensions specific to the type of joint.
    """
    # Retrieve the joint constraint and DoF axes
    dof_axes = dof_type.dofs_axes
    cts_axes = dof_type.cts_axes

    # Retrieve the number of constraints and dofs
    num_coords = dof_type.num_coords
    num_dofs = dof_type.num_dofs
    num_cts = dof_type.num_cts

    # Define a vector type for the joint coordinates
    _coordsvec = dof_type.coords_storage_type

    # Define the coordinate bound for correction
    q_j_limit = _coordsvec(dof_type.coords_bound(correction)) if _coordsvec is not None else None

    # Generate a joint type-specific function to write the
    # computed joint state into the model data arrays
    @wp.func
    def _write_typed_joint_data(
        # Inputs:
        cts_offset: int32,  # Index offset of the joint constraints
        dofs_offset: int32,  # Index offset of the joint DoFs
        coords_offset: int32,  # Index offset of the joint coordinates
        j_r_j: vec3f,  # 3D vector of the joint-local relative pose
        j_q_j: quatf,  # 4D unit-quaternion of the joint-local relative pose
        j_u_j: vec6f,  # 6D vector ofthe joint-local relative twist
        q_j_p: wp.array(dtype=float32),  # Reference joint coordinates for correction
        # Outputs:
        r_j_out: wp.array(dtype=float32),  # Flat array of joint constraint residuals
        dr_j_out: wp.array(dtype=float32),  # Flat array of joint constraint velocities
        q_j_out: wp.array(dtype=float32),  # Flat array of joint DoF coordinates
        dq_j_out: wp.array(dtype=float32),  # Flat array of joint DoF velocities
    ):
        # Only write the constraint residual and velocity if the joint defines constraints
        # NOTE: This will be disabled for free joints
        if wp.static(num_cts > 0):
            # Construct a 6D residual vector
            j_theta_j = wp.static(get_joint_constraint_angular_residual_function(dof_type))(j_q_j)
            j_p_j = screw(j_r_j, j_theta_j)
            # Store the joint constraint residuals
            for j in range(num_cts):
                r_j_out[cts_offset + j] = j_p_j[cts_axes[j]]
                dr_j_out[cts_offset + j] = j_u_j[cts_axes[j]]

        # Only write the DoF coordinates and velocities if the joint defines DoFs
        # NOTE: This will be disabled for fixed joints
        if wp.static(num_dofs > 0):
            # Map the joint relative pose to joint DoF coordinates
            q_j = wp.static(get_joint_coords_mapping_function(dof_type))(j_r_j, j_q_j)

            # Optionally generate code to correct the joint coordinates
            if wp.static(correction != JointCorrectionMode.NONE):
                q_j_prev = _coordsvec()
                for j in range(num_coords):
                    q_j_prev[j] = q_j_p[coords_offset + j]
                q_j = wp.static(get_joint_coord_correction_function(dof_type))(q_j, q_j_prev, q_j_limit)

            # Store the joint DoF coordinates
            for j in range(num_coords):
                q_j_out[coords_offset + j] = q_j[j]
            # Store the joint DoF velocities
            for j in range(num_dofs):
                dq_j_out[dofs_offset + j] = j_u_j[dof_axes[j]]

    # Return the function
    return _write_typed_joint_data


def make_write_joint_data(correction: JointCorrectionMode = JointCorrectionMode.TWOPI):
    """
    Generates functions to store the joint state according to the
    constraint and DoF dimensions specific to the type of joint.
    """

    @wp.func
    def _write_joint_data(
        # Inputs:
        dof_type: int32,
        cts_offset: int32,
        dofs_offset: int32,
        coords_offset: int32,
        j_r_j: vec3f,
        j_q_j: quatf,
        j_u_j: vec6f,
        q_j_p: wp.array(dtype=float32),
        # Outputs:
        data_r_j: wp.array(dtype=float32),
        data_dr_j: wp.array(dtype=float32),
        data_q_j: wp.array(dtype=float32),
        data_dq_j: wp.array(dtype=float32),
    ):
        """
        Stores the joint constraint residuals and DoF motion based on the joint type.

        Args:
            dof_type (int32): The type of joint DoF.
            cts_offset (int32): Index offset of the joint constraints.
            dofs_offset (int32): Index offset of the joint DoFs.
            coords_offset (int32): Index offset of the joint coordinates.
            j_r_j (vec3f): 3D vector of the joint-local relative translation.
            j_q_j (quatf): 4D unit-quaternion of the joint-local relative rotation.
            j_u_j (vec6f): 6D vector of the joint-local relative twist.
            data_r_j (wp.array): Flat array of joint constraint residuals.
            data_dr_j (wp.array): Flat array of joint constraint residuals.
            data_q_j (wp.array): Flat array of joint DoF coordinates.
            data_dq_j (wp.array): Flat array of joint DoF velocities.
        """
        # TODO: Use wp.static to include conditionals at compile time based on the joint types present in the builder

        if dof_type == JointDoFType.REVOLUTE:
            wp.static(make_typed_write_joint_data(JointDoFType.REVOLUTE, correction))(
                cts_offset,
                dofs_offset,
                coords_offset,
                j_r_j,
                j_q_j,
                j_u_j,
                q_j_p,
                data_r_j,
                data_dr_j,
                data_q_j,
                data_dq_j,
            )

        elif dof_type == JointDoFType.PRISMATIC:
            wp.static(make_typed_write_joint_data(JointDoFType.PRISMATIC))(
                cts_offset,
                dofs_offset,
                coords_offset,
                j_r_j,
                j_q_j,
                j_u_j,
                q_j_p,
                data_r_j,
                data_dr_j,
                data_q_j,
                data_dq_j,
            )

        elif dof_type == JointDoFType.CYLINDRICAL:
            wp.static(make_typed_write_joint_data(JointDoFType.CYLINDRICAL, correction))(
                cts_offset,
                dofs_offset,
                coords_offset,
                j_r_j,
                j_q_j,
                j_u_j,
                q_j_p,
                data_r_j,
                data_dr_j,
                data_q_j,
                data_dq_j,
            )

        elif dof_type == JointDoFType.UNIVERSAL:
            wp.static(make_typed_write_joint_data(JointDoFType.UNIVERSAL, correction))(
                cts_offset,
                dofs_offset,
                coords_offset,
                j_r_j,
                j_q_j,
                j_u_j,
                q_j_p,
                data_r_j,
                data_dr_j,
                data_q_j,
                data_dq_j,
            )

        elif dof_type == JointDoFType.SPHERICAL:
            wp.static(make_typed_write_joint_data(JointDoFType.SPHERICAL))(
                cts_offset,
                dofs_offset,
                coords_offset,
                j_r_j,
                j_q_j,
                j_u_j,
                q_j_p,
                data_r_j,
                data_dr_j,
                data_q_j,
                data_dq_j,
            )

        elif dof_type == JointDoFType.GIMBAL:
            wp.static(make_typed_write_joint_data(JointDoFType.GIMBAL, correction))(
                cts_offset,
                dofs_offset,
                coords_offset,
                j_r_j,
                j_q_j,
                j_u_j,
                q_j_p,
                data_r_j,
                data_dr_j,
                data_q_j,
                data_dq_j,
            )

        elif dof_type == JointDoFType.CARTESIAN:
            wp.static(make_typed_write_joint_data(JointDoFType.CARTESIAN))(
                cts_offset,
                dofs_offset,
                coords_offset,
                j_r_j,
                j_q_j,
                j_u_j,
                q_j_p,
                data_r_j,
                data_dr_j,
                data_q_j,
                data_dq_j,
            )

        elif dof_type == JointDoFType.FIXED:
            wp.static(make_typed_write_joint_data(JointDoFType.FIXED))(
                cts_offset,
                dofs_offset,
                coords_offset,
                j_r_j,
                j_q_j,
                j_u_j,
                q_j_p,
                data_r_j,
                data_dr_j,
                data_q_j,
                data_dq_j,
            )

        elif dof_type == JointDoFType.FREE:
            wp.static(make_typed_write_joint_data(JointDoFType.FREE))(
                cts_offset,
                dofs_offset,
                coords_offset,
                j_r_j,
                j_q_j,
                j_u_j,
                q_j_p,
                data_r_j,
                data_dr_j,
                data_q_j,
                data_dq_j,
            )

    # Return the function
    return _write_joint_data


###
# Functions - State Computation
###


@wp.func
def compute_joint_pose_and_relative_motion(
    T_B_j: transformf,
    T_F_j: transformf,
    u_B_j: vec6f,
    u_F_j: vec6f,
    B_r_Bj: vec3f,
    F_r_Fj: vec3f,
    X_j: mat33f,
) -> tuple[transformf, vec3f, quatf, vec6f]:
    """
    Computes the relative motion of a joint given the states of its Base and Follower bodies.

    Args:
        T_B_j (transformf): The absolute pose of the Base body in world coordinates.
        T_F_j (transformf): The absolute pose of the Follower body in world coordinates.
        u_B_j (vec6f): The absolute twist of the Base body in world coordinates.
        u_F_j (vec6f): The absolute twist of the Follower body in world coordinates.
        B_r_Bj (vec3f): The position of the joint frame in the Base body's local coordinates.
        F_r_Fj (vec3f): The position of the joint frame in the Follower body's local coordinates.
        X_j (mat33f): The joint transformation matrix.

    Returns:
        tuple[transformf, vec6f, vec6f]: The absolute pose of the joint frame in world coordinates,
        and two 6D vectors encoding the relative motion of the bodies in the frame of the joint.
    """

    # Joint frame as quaternion
    q_X_j = wp.quat_from_matrix(X_j)

    # Extract the decomposed state of the Base body
    r_B_j = wp.transform_get_translation(T_B_j)
    q_B_j = wp.transform_get_rotation(T_B_j)
    v_B_j = screw_linear(u_B_j)
    omega_B_j = screw_angular(u_B_j)

    # Extract the decomposed state of the Follower body
    r_F_j = wp.transform_get_translation(T_F_j)
    q_F_j = wp.transform_get_rotation(T_F_j)
    v_F_j = screw_linear(u_F_j)
    omega_F_j = screw_angular(u_F_j)

    # Local joint frame quantities
    r_Bj = wp.quat_rotate(q_B_j, B_r_Bj)
    q_Bj = q_B_j * q_X_j
    r_Fj = wp.quat_rotate(q_F_j, F_r_Fj)
    q_Fj = q_F_j * q_X_j

    # Compute the pose of the joint frame via the Base body
    r_j_B = r_B_j + r_Bj
    p_j = wp.transformation(r_j_B, q_Bj, dtype=float32)

    # Compute the relative pose between the representations of joint frame w.r.t. the two bodies
    # NOTE: The pose is decomposed into a translation vector `j_r_j` and a rotation quaternion `j_q_j`
    j_r_j = wp.quat_rotate_inv(q_Bj, r_F_j + r_Fj - r_j_B)
    j_q_j = wp.quat_inverse(q_Bj) * q_Fj

    # Compute the 6D relative twist vector between the representations of joint frame w.r.t. the two bodies
    # TODO: How can we simplify this expression and make it more efficient?
    j_v_j = wp.quat_rotate_inv(q_Bj, v_F_j - v_B_j + wp.cross(omega_F_j, r_Fj) - wp.cross(omega_B_j, r_Bj))
    j_omega_j = wp.quat_rotate_inv(q_Bj, omega_F_j - omega_B_j)
    j_u_j = screw(j_v_j, j_omega_j)

    # Return the computed joint frame pose and relative motion vectors
    return p_j, j_r_j, j_q_j, j_u_j


@wp.func
def compute_and_write_joint_implicit_dynamics(
    # Constants:
    dt: float32,
    coords_offset: int32,
    dofs_offset: int32,
    num_dynamic_cts: int32,
    dynamic_cts_offset: int32,
    # Inputs:
    model_joint_a_j: wp.array(dtype=float32),
    model_joint_b_j: wp.array(dtype=float32),
    model_joint_k_p_j: wp.array(dtype=float32),
    model_joint_k_d_j: wp.array(dtype=float32),
    data_joint_q_j: wp.array(dtype=float32),
    data_joint_dq_j: wp.array(dtype=float32),
    data_joint_q_j_ref: wp.array(dtype=float32),
    data_joint_dq_j_ref: wp.array(dtype=float32),
    data_joint_tau_j_ref: wp.array(dtype=float32),
    # Outputs:
    data_joint_m_j: wp.array(dtype=float32),
    data_joint_inv_m_j: wp.array(dtype=float32),
    data_joint_dq_b_j: wp.array(dtype=float32),
):
    # Iterate over the dynamic constraints of the joint and
    # compute and store the implicit dynamics intermediates
    # TODO: We currently do not handle implicit dynamics of
    # multi-dof joints, but we should generalize this.
    for j in range(num_dynamic_cts):
        coords_offset_j = coords_offset + j
        dofs_offset_j = dofs_offset + j
        dynamic_cts_offset_j = dynamic_cts_offset + j

        # Retrieve the current joint state
        # TODO: How can we avoid the extra memory load and
        # instead just get them from `make_write_joint_data`?
        q_j = data_joint_q_j[dofs_offset_j]
        dq_j = data_joint_dq_j[dofs_offset_j]

        # Retrieve the implicit joint dynamics and PD control parameters
        a_j = model_joint_a_j[dofs_offset_j]
        b_j = model_joint_b_j[dofs_offset_j]
        k_p_j = model_joint_k_p_j[dofs_offset_j]
        k_d_j = model_joint_k_d_j[dofs_offset_j]

        # Retrieve PD control references
        pd_q_j_ref = data_joint_q_j_ref[coords_offset_j]
        pd_dq_j_ref = data_joint_dq_j_ref[dofs_offset_j]
        pd_tau_j_ff = data_joint_tau_j_ref[dofs_offset_j]

        # Compute the implicit joint dynamics intermedates
        m_j = a_j + dt * (b_j + k_d_j) + dt * dt * k_p_j
        inv_m_j = 1.0 / m_j
        tau_j = pd_tau_j_ff + k_p_j * (pd_q_j_ref - q_j) + k_d_j * pd_dq_j_ref
        h_j = a_j * dq_j + dt * tau_j
        dq_b_j = inv_m_j * h_j

        # Store the resulting joint dynamics intermadiates
        data_joint_m_j[dynamic_cts_offset_j] = m_j
        data_joint_inv_m_j[dynamic_cts_offset_j] = inv_m_j
        data_joint_dq_b_j[dynamic_cts_offset_j] = dq_b_j


###
# Kernels
###


@cache
def make_compute_joints_data_kernel(correction: JointCorrectionMode = JointCorrectionMode.TWOPI):
    """
    Generates the kernel to compute the joint states based on the current body states.
    """

    @wp.kernel
    def _compute_joints_data(
        # Inputs:
        model_info_joint_coords_offset: wp.array(dtype=int32),
        model_info_joint_dofs_offset: wp.array(dtype=int32),
        model_info_joint_dynamic_cts_offset: wp.array(dtype=int32),
        model_info_joint_kinematic_cts_offset: wp.array(dtype=int32),
        model_time_dt: wp.array(dtype=float32),
        model_joint_wid: wp.array(dtype=int32),
        model_joint_dof_type: wp.array(dtype=int32),
        model_joint_num_dynamic_cts: wp.array(dtype=int32),
        model_joint_coords_offset: wp.array(dtype=int32),
        model_joint_dofs_offset: wp.array(dtype=int32),
        model_joint_dynamic_cts_offset: wp.array(dtype=int32),
        model_joint_kinematic_cts_offset: wp.array(dtype=int32),
        model_joint_bid_B: wp.array(dtype=int32),
        model_joint_bid_F: wp.array(dtype=int32),
        model_joint_B_r_Bj: wp.array(dtype=vec3f),
        model_joint_F_r_Fj: wp.array(dtype=vec3f),
        model_joint_X_j: wp.array(dtype=mat33f),
        model_joint_a_j: wp.array(dtype=float32),
        model_joint_b_j: wp.array(dtype=float32),
        model_joint_k_p_j: wp.array(dtype=float32),
        model_joint_k_d_j: wp.array(dtype=float32),
        data_body_q_i: wp.array(dtype=transformf),
        data_body_u_i: wp.array(dtype=vec6f),
        data_joint_q_j_ref: wp.array(dtype=float32),
        data_joint_dq_j_ref: wp.array(dtype=float32),
        data_joint_tau_j_ref: wp.array(dtype=float32),
        q_j_p: wp.array(dtype=float32),
        # Outputs:
        data_joint_p_j: wp.array(dtype=transformf),
        data_joint_r_j: wp.array(dtype=float32),
        data_joint_dr_j: wp.array(dtype=float32),
        data_joint_q_j: wp.array(dtype=float32),
        data_joint_dq_j: wp.array(dtype=float32),
        data_joint_m_j: wp.array(dtype=float32),
        data_joint_inv_m_j: wp.array(dtype=float32),
        data_joint_dq_b_j: wp.array(dtype=float32),
    ):
        # Retrieve the thread index
        jid = wp.tid()

        # Retrieve the joint model data
        wid = model_joint_wid[jid]
        dof_type = model_joint_dof_type[jid]
        num_dynamic_cts = model_joint_num_dynamic_cts[jid]
        bid_B = model_joint_bid_B[jid]
        bid_F = model_joint_bid_F[jid]
        B_r_Bj = model_joint_B_r_Bj[jid]
        F_r_Fj = model_joint_F_r_Fj[jid]
        X_j = model_joint_X_j[jid]

        # Retrieve the time step
        dt = model_time_dt[wid]

        # Retrieve world offsets
        # NOTE: We load data together for better memory coalescing
        world_coords_offset = model_info_joint_coords_offset[wid]
        world_dofs_offset = model_info_joint_dofs_offset[wid]
        world_dynamic_cts_offset = model_info_joint_dynamic_cts_offset[wid]
        world_kinematic_cts_offset = model_info_joint_kinematic_cts_offset[wid]

        # Retrieve joint-specific offsets
        # NOTE: We load data together for better memory coalescing
        joint_coords_offset = model_joint_coords_offset[jid]
        joint_dofs_offset = model_joint_dofs_offset[jid]
        joint_dynamic_cts_offset = model_joint_dynamic_cts_offset[jid]
        joint_kinematic_cts_offset = model_joint_kinematic_cts_offset[jid]

        # Construct offsets from world + current joint offset
        coords_offset = world_coords_offset + joint_coords_offset
        dofs_offset = world_dofs_offset + joint_dofs_offset
        dynamic_cts_offset = world_dynamic_cts_offset + joint_dynamic_cts_offset
        kinematic_cts_offset = world_kinematic_cts_offset + joint_kinematic_cts_offset

        # If the Base body is the world (bid=-1), use the identity transform (frame
        # of the world's origin), otherwise retrieve the Base body's pose and twist
        T_B_j = wp.transform_identity(dtype=float32)
        u_B_j = vec6f(0.0)
        if bid_B > -1:
            T_B_j = data_body_q_i[bid_B]
            u_B_j = data_body_u_i[bid_B]

        # Retrieve the Follower body's pose and twist
        T_F_j = data_body_q_i[bid_F]
        u_F_j = data_body_u_i[bid_F]

        # Compute the joint frame pose and relative motion
        p_j, j_r_j, j_q_j, j_u_j = compute_joint_pose_and_relative_motion(
            T_B_j, T_F_j, u_B_j, u_F_j, B_r_Bj, F_r_Fj, X_j
        )

        # Store the absolute pose of the joint frame in world coordinates
        data_joint_p_j[jid] = p_j

        # Store the joint constraint residuals and motion
        wp.static(make_write_joint_data(correction))(
            dof_type,
            kinematic_cts_offset,
            dofs_offset,
            coords_offset,
            j_r_j,
            j_q_j,
            j_u_j,
            q_j_p,
            data_joint_r_j,
            data_joint_dr_j,
            data_joint_q_j,
            data_joint_dq_j,
        )

        # Compute and store the implicit dynamics
        # for the dynamic constraints of the joint
        compute_and_write_joint_implicit_dynamics(
            dt,
            coords_offset,
            dofs_offset,
            num_dynamic_cts,
            dynamic_cts_offset,
            model_joint_a_j,
            model_joint_b_j,
            model_joint_k_p_j,
            model_joint_k_d_j,
            data_joint_q_j,
            data_joint_dq_j,
            data_joint_q_j_ref,
            data_joint_dq_j_ref,
            data_joint_tau_j_ref,
            data_joint_m_j,
            data_joint_inv_m_j,
            data_joint_dq_b_j,
        )

    # Return the kernel
    return _compute_joints_data


@wp.kernel
def _extract_actuators_state_from_joints(
    # Inputs:
    world_mask: wp.array(dtype=int32),
    model_info_joint_coords_offset: wp.array(dtype=int32),
    model_info_joint_dofs_offset: wp.array(dtype=int32),
    model_info_joint_actuated_coords_offset: wp.array(dtype=int32),
    model_info_joint_actuated_dofs_offset: wp.array(dtype=int32),
    model_joint_wid: wp.array(dtype=int32),
    model_joint_act_type: wp.array(dtype=int32),
    model_joint_num_coords: wp.array(dtype=int32),
    model_joint_num_dofs: wp.array(dtype=int32),
    model_joint_coords_offset: wp.array(dtype=int32),
    model_joint_dofs_offset: wp.array(dtype=int32),
    model_joint_actuated_coords_offset: wp.array(dtype=int32),
    model_joint_actuated_dofs_offset: wp.array(dtype=int32),
    joint_q: wp.array(dtype=float32),
    joint_u: wp.array(dtype=float32),
    # Outputs:
    actuator_q: wp.array(dtype=float32),
    actuator_u: wp.array(dtype=float32),
):
    # Retrieve the joint index from the thread grid
    jid = wp.tid()

    # Retrieve the world index and actuation type of the joint
    wid = model_joint_wid[jid]
    act_type = model_joint_act_type[jid]

    # Early exit the operation if the joint's world is flagged as skipped or if the joint is not actuated
    if world_mask[wid] == 0 or act_type == JointActuationType.PASSIVE:
        return

    # Retrieve the joint model data
    num_coords = model_joint_num_coords[jid]
    num_dofs = model_joint_num_dofs[jid]
    coords_offset = model_joint_coords_offset[jid]
    dofs_offset = model_joint_dofs_offset[jid]
    actuated_coords_offset = model_joint_actuated_coords_offset[jid]
    actuated_dofs_offset = model_joint_actuated_dofs_offset[jid]

    # Retrieve the index offsets of the joint's constraint and DoF dimensions
    world_joint_coords_offset = model_info_joint_coords_offset[wid]
    world_joint_dofs_offset = model_info_joint_dofs_offset[wid]
    world_joint_actuated_coords_offset = model_info_joint_actuated_coords_offset[wid]
    world_joint_actuated_dofs_offset = model_info_joint_actuated_dofs_offset[wid]

    # Append the index offsets of the world's joint blocks
    jq_start = world_joint_coords_offset + coords_offset
    jd_start = world_joint_dofs_offset + dofs_offset
    aq_start = world_joint_actuated_coords_offset + actuated_coords_offset
    ad_start = world_joint_actuated_dofs_offset + actuated_dofs_offset

    # TODO: Change to use array slice assignment when supported in Warp
    # # Store the actuated joint coordinates and velocities
    # actuator_q[aq_start : aq_start + num_coords] = joint_q[jq_start : jq_start + num_coords]
    # actuator_u[ad_start : ad_start + num_dofs] = joint_u[jd_start : jd_start + num_dofs]

    # Store the actuated joint coordinates and velocities
    for j in range(num_coords):
        actuator_q[aq_start + j] = joint_q[jq_start + j]
    for j in range(num_dofs):
        actuator_u[ad_start + j] = joint_u[jd_start + j]


@wp.kernel
def _extract_joints_state_from_actuators(
    # Inputs:
    world_mask: wp.array(dtype=int32),
    model_info_joint_coords_offset: wp.array(dtype=int32),
    model_info_joint_dofs_offset: wp.array(dtype=int32),
    model_info_joint_actuated_coords_offset: wp.array(dtype=int32),
    model_info_joint_actuated_dofs_offset: wp.array(dtype=int32),
    model_joint_wid: wp.array(dtype=int32),
    model_joint_act_type: wp.array(dtype=int32),
    model_joint_num_coords: wp.array(dtype=int32),
    model_joint_num_dofs: wp.array(dtype=int32),
    model_joint_coords_offset: wp.array(dtype=int32),
    model_joint_dofs_offset: wp.array(dtype=int32),
    model_joint_actuated_coords_offset: wp.array(dtype=int32),
    model_joint_actuated_dofs_offset: wp.array(dtype=int32),
    actuator_q: wp.array(dtype=float32),
    actuator_u: wp.array(dtype=float32),
    # Outputs:
    joint_q: wp.array(dtype=float32),
    joint_u: wp.array(dtype=float32),
):
    # Retrieve the joint index from the thread grid
    jid = wp.tid()

    # Retrieve the world index and actuation type of the joint
    wid = model_joint_wid[jid]
    act_type = model_joint_act_type[jid]

    # Early exit the operation if the joint's world is flagged as skipped or if the joint is not actuated
    if world_mask[wid] == 0 or act_type == JointActuationType.PASSIVE:
        return

    # Retrieve the joint model data
    num_coords = model_joint_num_coords[jid]
    num_dofs = model_joint_num_dofs[jid]
    coords_offset = model_joint_coords_offset[jid]
    dofs_offset = model_joint_dofs_offset[jid]
    actuated_coords_offset = model_joint_actuated_coords_offset[jid]
    actuated_dofs_offset = model_joint_actuated_dofs_offset[jid]

    # Retrieve the index offsets of the joint's constraint and DoF dimensions
    world_joint_coords_offset = model_info_joint_coords_offset[wid]
    world_joint_dofs_offset = model_info_joint_dofs_offset[wid]
    world_joint_actuated_coords_offset = model_info_joint_actuated_coords_offset[wid]
    world_joint_actuated_dofs_offset = model_info_joint_actuated_dofs_offset[wid]

    # Append the index offsets of the world's joint blocks
    jq_start = world_joint_coords_offset + coords_offset
    jd_start = world_joint_dofs_offset + dofs_offset
    aq_start = world_joint_actuated_coords_offset + actuated_coords_offset
    ad_start = world_joint_actuated_dofs_offset + actuated_dofs_offset

    # TODO: Change to use array slice assignment when supported in Warp
    # # Store the actuated joint coordinates and velocities
    # joint_q[jq_start : jq_start + num_coords] = actuator_q[aq_start : aq_start + num_coords]
    # joint_u[jd_start : jd_start + num_dofs] = actuator_u[ad_start : ad_start + num_dofs]

    # Store the actuated joint coordinates and velocities
    for j in range(num_coords):
        joint_q[jq_start + j] = actuator_q[aq_start + j]
    for j in range(num_dofs):
        joint_u[jd_start + j] = actuator_u[ad_start + j]


###
# Launchers
###


def compute_joints_data(
    model: ModelKamino,
    data: DataKamino,
    q_j_p: wp.array,
    correction: JointCorrectionMode = JointCorrectionMode.TWOPI,
) -> None:
    """
    Computes the states of the joints based on the current body states.

    The computed joint state data includes both the generalized coordinates and velocities
    corresponding to the respective degrees of freedom (DoFs), as well as the constraint-space
    residuals and velocities of the applied bilateral constraints.

    Args:
        model (`ModelKamino`):
            The model container holding the time-invariant data of the simulation.
        q_j_p (`wp.array`):
            An array of previous joint DoF coordinates used for coordinate correction.\n
            Only used for revolute DoFs of the relevant joints to enforce angle continuity.\n
            Shape of ``(sum_of_num_joint_coords,)`` and type :class:`float`.
        data (`DataKamino`):
            The solver data container holding the internal time-varying state of the simulation.
    """

    # Generate the kernel to compute the joint states
    # conditioned on the type coordinate correction
    _kernel = make_compute_joints_data_kernel(correction)

    # Launch the kernel to compute the joint states
    wp.launch(
        _kernel,
        dim=model.size.sum_of_num_joints,
        inputs=[
            # Inputs:
            model.info.joint_coords_offset,
            model.info.joint_dofs_offset,
            model.info.joint_dynamic_cts_offset,
            model.info.joint_kinematic_cts_offset,
            model.time.dt,
            model.joints.wid,
            model.joints.dof_type,
            model.joints.num_dynamic_cts,
            model.joints.coords_offset,
            model.joints.dofs_offset,
            model.joints.dynamic_cts_offset,
            model.joints.kinematic_cts_offset,
            model.joints.bid_B,
            model.joints.bid_F,
            model.joints.B_r_Bj,
            model.joints.F_r_Fj,
            model.joints.X_j,
            model.joints.a_j,
            model.joints.b_j,
            model.joints.k_p_j,
            model.joints.k_d_j,
            data.bodies.q_i,
            data.bodies.u_i,
            data.joints.q_j_ref,
            data.joints.dq_j_ref,
            data.joints.tau_j_ref,
            q_j_p,
            # Outputs:
            data.joints.p_j,
            data.joints.r_j,
            data.joints.dr_j,
            data.joints.q_j,
            data.joints.dq_j,
            data.joints.m_j,
            data.joints.inv_m_j,
            data.joints.dq_b_j,
        ],
        device=model.device,
    )


def extract_actuators_state_from_joints(
    model: ModelKamino,
    world_mask: wp.array,
    joint_q: wp.array,
    joint_u: wp.array,
    actuator_q: wp.array,
    actuator_u: wp.array,
):
    """
    Extracts the states of the actuated joints from the full joint state arrays.

    Only joints that are marked as actuated and belong to worlds
    that are not masked will have their states extracted.

    Args:
        model (`ModelKamino`):
            The model container holding the time-invariant data of the simulation.
        joint_q (`wp.array`):
            The full array of joint coordinates.\n
            Shape of ``(sum_of_num_joint_coords,)`` and type :class:`float`.
        joint_u (`wp.array`):
            The full array of joint velocities.\n
            Shape of ``(sum_of_num_joint_dofs,)`` and type :class:`float`.
        actuator_q (`wp.array`):
            The output array to store the actuated joint coordinates.\n
            Shape of ``(sum_of_num_actuated_joint_coords,)`` and type :class:`float`.
        actuator_u (`wp.array`):
            The output array to store the actuated joint velocities.\n
            Shape of ``(sum_of_actuated_joint_dofs,)`` and type :class:`float`.
        world_mask (`wp.array`):
            An array indicating which worlds are active (1) or skipped (0).\n
            Shape of ``(num_worlds,)`` and type :class:`int32`.
    """
    wp.launch(
        _extract_actuators_state_from_joints,
        dim=model.size.sum_of_num_joints,
        inputs=[
            world_mask,
            model.info.joint_coords_offset,
            model.info.joint_dofs_offset,
            model.info.joint_actuated_coords_offset,
            model.info.joint_actuated_dofs_offset,
            model.joints.wid,
            model.joints.act_type,
            model.joints.num_coords,
            model.joints.num_dofs,
            model.joints.coords_offset,
            model.joints.dofs_offset,
            model.joints.actuated_coords_offset,
            model.joints.actuated_dofs_offset,
            joint_q,
            joint_u,
        ],
        outputs=[
            actuator_q,
            actuator_u,
        ],
        device=model.device,
    )


def extract_joints_state_from_actuators(
    model: ModelKamino,
    world_mask: wp.array,
    actuator_q: wp.array,
    actuator_u: wp.array,
    joint_q: wp.array,
    joint_u: wp.array,
):
    """
    Extracts the states of the actuated joints from the full joint state arrays.

    Only joints that are marked as actuated and belong to worlds
    that are not masked will have their states extracted.

    Args:
        model (`ModelKamino`):
            The model container holding the time-invariant data of the simulation.
        joint_q (`wp.array`):
            The full array of joint coordinates.\n
            Shape of ``(sum_of_num_joint_coords,)`` and type :class:`float`.
        joint_u (`wp.array`):
            The full array of joint velocities.\n
            Shape of ``(sum_of_num_joint_dofs,)`` and type :class:`float`.
        actuator_q (`wp.array`):
            The output array to store the actuated joint coordinates.\n
            Shape of ``(sum_of_num_actuated_joint_coords,)`` and type :class:`float`.
        actuator_u (`wp.array`):
            The output array to store the actuated joint velocities.\n
            Shape of ``(sum_of_actuated_joint_dofs,)`` and type :class:`float`.
        world_mask (`wp.array`):
            An array indicating which worlds are active (1) or skipped (0).\n
            Shape of ``(num_worlds,)`` and type :class:`int32`.
    """
    wp.launch(
        _extract_joints_state_from_actuators,
        dim=model.size.sum_of_num_joints,
        inputs=[
            world_mask,
            model.info.joint_coords_offset,
            model.info.joint_dofs_offset,
            model.info.joint_actuated_coords_offset,
            model.info.joint_actuated_dofs_offset,
            model.joints.wid,
            model.joints.act_type,
            model.joints.num_coords,
            model.joints.num_dofs,
            model.joints.coords_offset,
            model.joints.dofs_offset,
            model.joints.actuated_coords_offset,
            model.joints.actuated_dofs_offset,
            actuator_q,
            actuator_u,
        ],
        outputs=[
            joint_q,
            joint_u,
        ],
        device=model.device,
    )
