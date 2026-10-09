# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: Kinematics: Joints
"""

from __future__ import annotations

from functools import cache

import warp as wp

from .....sim.articulation import transform_3d_rotational_axes
from ..core.data import DataKamino
from ..core.joints import DofActuationPath, JointActuationType, JointCorrectionMode, JointDoFType
from ..core.math import FLOAT32_MAX, quat_log, quat_twist_angle
from ..core.model import ModelKamino
from ..core.types import (
    vec1f,
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

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


###
# Constants
###


DEFAULT_LIMIT_V1F = vec1f(FLOAT32_MAX)
DEFAULT_LIMIT_V2F = wp.vec2f(FLOAT32_MAX)
DEFAULT_LIMIT_V3F = wp.vec3f(FLOAT32_MAX)
DEFAULT_LIMIT_V4F = wp.vec4f(FLOAT32_MAX)
DEFAULT_LIMIT_V7F = vec7f(FLOAT32_MAX)


###
# Functions - Coordinate Correction
###


@wp.func
def correct_rotational_coord(q_j_in: wp.float32, q_j_ref: wp.float32 = 0.0) -> wp.float32:
    """
    Corrects a rotational joint coordinate to be as close as possible to a reference coordinate.
    """
    return q_j_in + wp.round((q_j_ref - q_j_in) / wp.tau) * wp.tau  # Note: wp.tau is 2 * pi


@wp.func
def correct_rotational_coord_with_limit(
    q_j_in: wp.float32, q_j_ref: wp.float32 = 0.0, q_j_limit: wp.float32 = FLOAT32_MAX
) -> wp.float32:
    """Corrects a rotational coordinate relative to a reference and wraps it within a limit."""
    return wp.mod(correct_rotational_coord(q_j_in, q_j_ref), q_j_limit)


@wp.func
def correct_quat_vector_coord(q_j_in: wp.vec4f, q_j_ref: wp.vec4f) -> wp.vec4f:
    """
    Corrects a quaternion joint coordinate to be as close as possible to a reference coordinate.

    This ensures that the quaternion `q_j_in` is chosen such that it is
    closer to the reference quaternion `q_j_ref`, accounting for the fact
    that quaternions `q` and `-q` represent the same rotation.
    """
    if wp.length_sq(q_j_in + q_j_ref) < wp.length_sq(q_j_in - q_j_ref):
        q_j_in *= -1.0
    return q_j_in


@wp.func
def correct_joint_coord_free(q_j_in: vec7f, q_j_ref: vec7f, q_j_limit: vec7f = DEFAULT_LIMIT_V7F) -> vec7f:
    """Corrects the orientation quaternion coordinate of a free joint."""
    q_j_in[3:] = correct_quat_vector_coord(q_j_in[3:], q_j_ref[3:])
    return q_j_in


@wp.func
def correct_joint_coord_revolute(q_j_in: vec1f, q_j_ref: vec1f, q_j_limit: vec1f = DEFAULT_LIMIT_V1F) -> vec1f:
    """Corrects the rotational joint coordinate."""
    q_j_in[0] = correct_rotational_coord_with_limit(q_j_in[0], q_j_ref[0], q_j_limit[0])
    return q_j_in


@wp.func
def correct_joint_coord_prismatic(q_j_in: vec1f, q_j_ref: vec1f, q_j_limit: vec1f = DEFAULT_LIMIT_V1F) -> vec1f:
    """No correction needed for prismatic coordinates."""
    return q_j_in


@wp.func
def correct_joint_coord_cylindrical(
    q_j_in: wp.vec2f, q_j_ref: wp.vec2f, q_j_limit: wp.vec2f = DEFAULT_LIMIT_V2F
) -> wp.vec2f:
    """Corrects only the rotational joint coordinate."""
    q_j_in[1] = correct_rotational_coord_with_limit(q_j_in[1], q_j_ref[1], q_j_limit[1])
    return q_j_in


@wp.func
def correct_joint_coord_universal(
    q_j_in: wp.vec2f, q_j_ref: wp.vec2f, q_j_limit: wp.vec2f = DEFAULT_LIMIT_V2F
) -> wp.vec2f:
    """Corrects each of the two rotational joint coordinates individually."""
    q_j_in[0] = correct_rotational_coord_with_limit(q_j_in[0], q_j_ref[0], q_j_limit[0])
    q_j_in[1] = correct_rotational_coord_with_limit(q_j_in[1], q_j_ref[1], q_j_limit[1])
    return q_j_in


@wp.func
def correct_joint_coord_gimbal(
    q_j_in: wp.vec3f, q_j_ref: wp.vec3f, q_j_limit: wp.vec3f = DEFAULT_LIMIT_V3F
) -> wp.vec3f:
    """Correct each intrinsic Euler coordinate against its previous value."""
    for i in range(3):
        q_j_in[i] = correct_rotational_coord_with_limit(q_j_in[i], q_j_ref[i], q_j_limit[i])
    return q_j_in


@wp.func
def correct_joint_coord_spherical(
    q_j_in: wp.vec4f, q_j_ref: wp.vec4f, q_j_limit: wp.vec4f = DEFAULT_LIMIT_V4F
) -> wp.vec4f:
    """Corrects a quaternion joint coordinate to be as close as possible to a reference."""
    return correct_quat_vector_coord(q_j_in, q_j_ref)


@wp.func
def correct_joint_coord_cartesian(
    q_j_in: wp.vec3f, q_j_ref: wp.vec3f, q_j_limit: wp.vec3f = DEFAULT_LIMIT_V3F
) -> wp.vec3f:
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
    elif dof_type == JointDoFType.GIMBAL:
        return correct_joint_coord_gimbal
    elif dof_type == JointDoFType.GIMBAL_LEFT_HANDED:
        return correct_joint_coord_gimbal
    elif dof_type == JointDoFType.SPHERICAL:
        return correct_joint_coord_spherical
    elif dof_type == JointDoFType.CARTESIAN:
        return correct_joint_coord_cartesian
    elif dof_type == JointDoFType.FIXED:
        return None
    else:
        raise ValueError(f"Unknown joint DoF type: {dof_type}")


@wp.func
def _correct_rotational_coord_in_place(coords: wp.array[wp.float32], coords_ref: wp.array[wp.float32], slot: wp.int32):
    """Correct one angular coordinate stored at ``slot`` against its reference (in-place, ±2π)."""
    coords[slot] = correct_rotational_coord(coords[slot], coords_ref[slot])


@wp.func
def _correct_quat_coord_in_place(coords: wp.array[wp.float32], coords_ref: wp.array[wp.float32], slot: wp.int32):
    """Correct a quaternion stored at ``slot..slot+3`` against its reference (in-place, up to sign)."""
    quat = wp.vec4f(coords[slot], coords[slot + 1], coords[slot + 2], coords[slot + 3])
    quat_ref = wp.vec4f(coords_ref[slot], coords_ref[slot + 1], coords_ref[slot + 2], coords_ref[slot + 3])
    quat = correct_quat_vector_coord(quat, quat_ref)
    for i in range(4):
        coords[slot + i] = quat[i]


@wp.func
def correct_joint_coords_in_place(
    dof_type: wp.int32,
    coords: wp.array[wp.float32],
    coords_ref: wp.array[wp.float32],
    offset: wp.int32,
):
    """
    Correct the joint coordinates stored at ``coords[offset : offset + num_coords]`` against a
    reference block in ``coords_ref``, in place, using ±2π wrapping for angles and sign correction
    for quaternions. No limit wrapping is applied; use :func:`get_joint_coord_correction_function`
    if bounded wrapping is needed.

    The runtime ``dof_type`` dispatches to the appropriate per-type correction, matching the
    joint DoF layouts defined in :class:`JointDoFType`. Types with no correction (Cartesian,
    Fixed, Prismatic) are no-ops, and unrecognized values are also silently skipped to accommodate
    solver-specific extensions with disjoint coord slots (e.g. FK's Axis type).
    """
    if dof_type == JointDoFType.CARTESIAN or dof_type == JointDoFType.FIXED or dof_type == JointDoFType.PRISMATIC:
        return
    elif dof_type == JointDoFType.CYLINDRICAL:
        _correct_rotational_coord_in_place(coords, coords_ref, offset + 1)
    elif dof_type == JointDoFType.FREE:
        _correct_quat_coord_in_place(coords, coords_ref, offset + 3)
    elif dof_type == JointDoFType.REVOLUTE:
        _correct_rotational_coord_in_place(coords, coords_ref, offset)
    elif dof_type == JointDoFType.SPHERICAL:
        _correct_quat_coord_in_place(coords, coords_ref, offset)
    elif dof_type == JointDoFType.UNIVERSAL:
        _correct_rotational_coord_in_place(coords, coords_ref, offset)
        _correct_rotational_coord_in_place(coords, coords_ref, offset + 1)
    elif dof_type == JointDoFType.GIMBAL or dof_type == JointDoFType.GIMBAL_LEFT_HANDED:
        for i in range(3):
            _correct_rotational_coord_in_place(coords, coords_ref, offset + i)


@wp.func
def select_gimbal_coords(j_q_j: wp.quatf, reference: wp.vec3f, third_axis_sign: wp.float32) -> wp.vec3f:
    """Select the authored intrinsic-XYZ chart nearest to a reference triple."""
    principal = wp.quat_to_euler(j_q_j, 2, 1, 0)
    alternative = wp.vec3f(principal[0] + wp.pi, wp.pi - principal[1], principal[2] + wp.pi)
    principal[2] *= third_axis_sign
    alternative[2] *= third_axis_sign
    principal = wp.vec3f(
        correct_rotational_coord(principal[0], reference[0]),
        correct_rotational_coord(principal[1], reference[1]),
        correct_rotational_coord(principal[2], reference[2]),
    )
    alternative = wp.vec3f(
        correct_rotational_coord(alternative[0], reference[0]),
        correct_rotational_coord(alternative[1], reference[1]),
        correct_rotational_coord(alternative[2], reference[2]),
    )
    if wp.length_sq(alternative - reference) < wp.length_sq(principal - reference):
        return alternative
    return principal


@wp.func
def gimbal_transported_axes(coords: wp.vec3f, third_axis_sign: wp.float32) -> wp.mat33f:
    """Return the physical angular-velocity columns for authored Euler rates."""
    a0, a1, a2 = transform_3d_rotational_axes(
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(0.0, 1.0, 0.0),
        wp.vec3(0.0, 0.0, third_axis_sign),
        coords[0],
        coords[1],
    )
    return wp.matrix_from_cols(wp.vec3f(a0), wp.vec3f(a1), wp.vec3f(a2))


@wp.func
def gimbal_reciprocal_axes(coords: wp.vec3f, third_axis_sign: wp.float32) -> wp.mat33f:
    """Return reciprocal angular axes ``E^-T`` for an authored gimbal chart."""
    a0, a1, a2 = transform_3d_rotational_axes(
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(0.0, 1.0, 0.0),
        wp.vec3(0.0, 0.0, third_axis_sign),
        coords[0],
        coords[1],
    )
    c12 = wp.cross(a1, a2)
    c02 = wp.cross(a0, a2)
    c01 = wp.cross(a0, a1)
    return wp.matrix_from_cols(
        c12 / wp.dot(a0, c12),
        c02 / wp.dot(a1, c02),
        c01 / wp.dot(a2, c01),
    )


@wp.func
def map_gimbal_angular_velocity_to_rates(coords: wp.vec3f, omega: wp.vec3f, third_axis_sign: wp.float32) -> wp.vec3f:
    """Project joint-frame angular velocity onto selected intrinsic Euler rates."""
    reciprocal_axes = gimbal_reciprocal_axes(coords, third_axis_sign)
    return wp.vec3f(
        wp.dot(wp.vec3f(reciprocal_axes[:, 0]), omega),
        wp.dot(wp.vec3f(reciprocal_axes[:, 1]), omega),
        wp.dot(wp.vec3f(reciprocal_axes[:, 2]), omega),
    )


###
# Functions - Coordinate Mappings
###


@wp.func
def map_to_joint_coords_free(j_r_j: wp.vec3f, j_q_j: wp.quatf) -> vec7f:
    """Returns the full 7D representation of joint pose (3D translation + 4D rotation)."""
    # TODO: Is there a more efficient way to construct a vec7f?
    return vec7f(j_r_j[0], j_r_j[1], j_r_j[2], j_q_j.x, j_q_j.y, j_q_j.z, j_q_j.w)


@wp.func
def map_to_joint_coords_revolute(j_r_j: wp.vec3f, j_q_j: wp.quatf) -> vec1f:
    """Returns the 1D rotation angle about the local X-axis."""
    # Measure rotation around the x-axis only
    axis = wp.vec3f(1.0, 0.0, 0.0)
    return vec1f(quat_twist_angle(j_q_j, axis))


@wp.func
def map_to_joint_coords_prismatic(j_r_j: wp.vec3f, j_q_j: wp.quatf) -> vec1f:
    """Returns the 1D translation distance along the local X-axis."""
    return vec1f(j_r_j[0])


@wp.func
def map_to_joint_coords_cylindrical(j_r_j: wp.vec3f, j_q_j: wp.quatf) -> wp.vec2f:
    """Returns the 2D vector of translation and rotation about the local X-axis."""
    j_p_j = quat_log(j_q_j)
    return wp.vec2f(j_r_j[0], j_p_j[0])


@wp.func
def map_to_joint_coords_universal(j_r_j: wp.vec3f, j_q_j: wp.quatf) -> wp.vec2f:
    """Returns the 2D vector of joint angles for the two revolute DoFs."""
    # X and Y axis in base frame
    a_x_B = wp.vec3f(1.0, 0.0, 0.0)
    a_y_B = wp.vec3f(0.0, 1.0, 0.0)

    # X and Y axis in follower frame
    a_x_F = wp.quat_rotate(j_q_j, a_x_B)
    a_y_F = wp.quat_rotate(j_q_j, a_y_B)

    # Extract theta_x, as the angle of the rotation about a_x_B mapping a_y_B to a_y_F
    theta_x = wp.atan2(a_y_F[2], a_y_F[1])

    # Extract theta_y, as the angle of the rotation about a_y_F mapping a_x_B to a_x_F
    theta_y = wp.atan2(wp.dot(wp.cross(a_x_B, a_x_F), a_y_F), a_x_F[0])

    return wp.vec2f(theta_x, theta_y)


@wp.func
def map_to_joint_coords_gimbal(j_r_j: wp.vec3f, j_q_j: wp.quatf) -> wp.vec3f:
    """Return intrinsic XYZ Euler coordinates for a canonical gimbal frame."""
    return wp.quat_to_euler(j_q_j, 2, 1, 0)


@wp.func
def map_to_joint_coords_gimbal_left_handed(j_r_j: wp.vec3f, j_q_j: wp.quatf) -> wp.vec3f:
    """Return authored coordinates for a left-handed gimbal frame."""
    coords = map_to_joint_coords_gimbal(j_r_j, j_q_j)
    return wp.vec3f(coords[0], coords[1], -coords[2])


@wp.func
def map_to_joint_coords_spherical(j_r_j: wp.vec3f, j_q_j: wp.quatf) -> wp.vec4f:
    """Returns the 4D unit-quaternion representing the joint rotation."""
    return wp.vec4f(*j_q_j)


@wp.func
def map_to_joint_coords_cartesian(j_r_j: wp.vec3f, j_q_j: wp.quatf) -> wp.vec3f:
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
    elif dof_type == JointDoFType.GIMBAL:
        return map_to_joint_coords_gimbal
    elif dof_type == JointDoFType.GIMBAL_LEFT_HANDED:
        return map_to_joint_coords_gimbal_left_handed
    elif dof_type == JointDoFType.SPHERICAL:
        return map_to_joint_coords_spherical
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
def joint_constraint_angular_residual_free(j_q_j: wp.quatf) -> wp.vec3f:
    return wp.vec3f(0.0, 0.0, 0.0)


@wp.func
def joint_constraint_angular_residual_revolute(j_q_j: wp.quatf) -> wp.vec3f:
    """Returns the joint constraint residual for a revolute joint that rotates about the local X-axis."""
    # x-axis attached to the base body
    j_x_B = wp.vec3f(1.0, 0.0, 0.0)

    # x-axis attached to the follower body, expressed in the joint frame on the base body.
    j_x_F = wp.quat_rotate(j_q_j, j_x_B)

    # Residual vector = sin(res_angle) * axis, where axis only has y, and z components.
    # For small angles this is equal to res_angle * axis
    return wp.cross(j_x_B, j_x_F)


@wp.func
def joint_constraint_angular_residual_universal(j_q_j: wp.quatf) -> wp.vec3f:
    """Returns the joint constraint residual for a universal joint."""
    e_y = wp.vec3f(0.0, 1.0, 0.0)
    a_x_dot_a_y = wp.quat_rotate(j_q_j, e_y)[0]

    return wp.vec3f(0.0, 0.0, -a_x_dot_a_y)


@wp.func
def joint_constraint_angular_residual_fixed(j_q_j: wp.quatf) -> wp.vec3f:
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
    elif dof_type == JointDoFType.GIMBAL_LEFT_HANDED:
        return joint_constraint_angular_residual_free
    elif dof_type == JointDoFType.CARTESIAN:
        return joint_constraint_angular_residual_fixed
    elif dof_type == JointDoFType.FIXED:
        return joint_constraint_angular_residual_fixed
    else:
        raise ValueError(f"Unknown joint DoF type: {dof_type}")


@wp.func
def universal_intermediary_axes(rel_ori_joint: wp.quatf) -> wp.mat33f:
    """
    Compute the columns of the universal-joint intermediary body frame, expressed as a rotation
    matrix in the Base-side joint frame. Column 1 is the x-axis on the Base; column 2 is the y-axis
    on the Follower (orthogonalized against the Base x-axis in case constraints are violated);
    column 3 is their cross product.

    Args:
        rel_ori_joint: Relative orientation of the Follower joint frame w.r.t. the Base joint frame.
    """
    e_x = wp.vec3f(1.0, 0.0, 0.0)
    e_y = wp.vec3f(0.0, 1.0, 0.0)
    a_x = e_x
    a_y = wp.quat_rotate(rel_ori_joint, e_y)
    a_y = wp.normalize(a_y - wp.dot(a_y, a_x) * a_x)
    a_z = wp.cross(a_x, a_y)
    return wp.matrix_from_cols(a_x, a_y, a_z)


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
    third_axis_sign = -1.0 if dof_type == JointDoFType.GIMBAL_LEFT_HANDED else 1.0

    # Generate a joint type-specific function to write the
    # computed joint state into the model data arrays
    @wp.func
    def _write_typed_joint_data(
        # Inputs:
        cts_offset: wp.int32,  # Index offset of the joint constraints
        dofs_offset: wp.int32,  # Index offset of the joint DoFs
        coords_offset: wp.int32,  # Index offset of the joint coordinates
        j_r_j: wp.vec3f,  # 3D vector of the joint-local relative pose
        j_q_j: wp.quatf,  # 4D unit-quaternion of the joint-local relative pose
        j_u_j: wp.spatial_vectorf,  # 6D vector of the joint-local relative twist
        q_j_p: wp.array[wp.float32],  # Reference joint coordinates for correction
        # Outputs:
        r_j_out: wp.array[wp.float32],  # Flat array of joint constraint residuals
        dr_j_out: wp.array[wp.float32],  # Flat array of joint constraint velocities
        q_j_out: wp.array[wp.float32],  # Flat array of joint DoF coordinates
        dq_j_out: wp.array[wp.float32],  # Flat array of joint DoF velocities
    ):
        # Convert angular velocity to intermediary body frame for universal joint
        if wp.static(dof_type == JointDoFType.UNIVERSAL):
            axes = universal_intermediary_axes(j_q_j)
            omega_intermediary = wp.transpose(axes) @ wp.spatial_bottom(j_u_j)
            j_u_j = wp.spatial_vectorf(*wp.spatial_top(j_u_j), *omega_intermediary)

        # Only write the constraint residual and velocity if the joint defines constraints
        # NOTE: This will be disabled for free joints
        if wp.static(num_cts > 0):
            if cts_offset >= wp.int32(0):
                # Construct a 6D residual vector
                j_theta_j = wp.static(get_joint_constraint_angular_residual_function(dof_type))(j_q_j)
                j_p_j = wp.spatial_vectorf(*j_r_j, *j_theta_j)
                # Store the joint constraint residuals
                for j in range(num_cts):
                    r_j_out[cts_offset + j] = j_p_j[cts_axes[j]]
                    dr_j_out[cts_offset + j] = j_u_j[cts_axes[j]]

        # Only write the DoF coordinates and velocities if the joint defines DoFs
        # NOTE: This will be disabled for fixed joints
        if wp.static(num_dofs > 0):
            # Map the joint relative pose to joint DoF coordinates
            q_j = wp.static(get_joint_coords_mapping_function(dof_type))(j_r_j, j_q_j)

            if wp.static(dof_type == JointDoFType.GIMBAL or dof_type == JointDoFType.GIMBAL_LEFT_HANDED):
                q_j_prev = _coordsvec()
                for j in range(num_coords):
                    q_j_prev[j] = q_j_p[coords_offset + j]
                q_j = select_gimbal_coords(j_q_j, q_j_prev, third_axis_sign)
            # Optionally generate code to correct the joint coordinates
            elif wp.static(correction != JointCorrectionMode.NONE):
                q_j_prev = _coordsvec()
                for j in range(num_coords):
                    q_j_prev[j] = q_j_p[coords_offset + j]
                q_j = wp.static(get_joint_coord_correction_function(dof_type))(q_j, q_j_prev, q_j_limit)

            # Store the joint DoF coordinates
            for j in range(num_coords):
                q_j_out[coords_offset + j] = q_j[j]
            # Store the joint DoF velocities
            if wp.static(dof_type == JointDoFType.GIMBAL or dof_type == JointDoFType.GIMBAL_LEFT_HANDED):
                rates = map_gimbal_angular_velocity_to_rates(q_j, wp.spatial_bottom(j_u_j), third_axis_sign)
                for j in range(3):
                    dq_j_out[dofs_offset + j] = rates[j]
            else:
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
        dof_type: wp.int32,
        cts_offset: wp.int32,
        dofs_offset: wp.int32,
        coords_offset: wp.int32,
        j_r_j: wp.vec3f,
        j_q_j: wp.quatf,
        j_u_j: wp.spatial_vectorf,
        q_j_p: wp.array[wp.float32],
        # Outputs:
        data_r_j: wp.array[wp.float32],
        data_dr_j: wp.array[wp.float32],
        data_q_j: wp.array[wp.float32],
        data_dq_j: wp.array[wp.float32],
    ):
        """
        Stores the joint constraint residuals and DoF motion based on the joint type.

        Args:
            dof_type: The type of joint DoF.
            cts_offset: Index offset of the joint constraints.
            dofs_offset: Index offset of the joint DoFs.
            coords_offset: Index offset of the joint coordinates.
            j_r_j: 3D vector of the joint-local relative translation.
            j_q_j: 4D unit-quaternion of the joint-local relative rotation.
            j_u_j: 6D vector of the joint-local relative twist.
            data_r_j: Flat array of joint constraint residuals.
            data_dr_j: Flat array of joint constraint residuals.
            data_q_j: Flat array of joint DoF coordinates.
            data_dq_j: Flat array of joint DoF velocities.
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

        elif dof_type == JointDoFType.GIMBAL_LEFT_HANDED:
            wp.static(make_typed_write_joint_data(JointDoFType.GIMBAL_LEFT_HANDED, correction))(
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
    T_B_j: wp.transformf,
    T_F_j: wp.transformf,
    u_B_j: wp.spatial_vectorf,
    u_F_j: wp.spatial_vectorf,
    B_r_Bj: wp.vec3f,
    F_r_Fj: wp.vec3f,
    X_Bj: wp.mat33f,
    X_Fj: wp.mat33f,
) -> tuple[wp.transformf, wp.vec3f, wp.quatf, wp.spatial_vectorf]:
    """
    Computes the relative motion of a joint given the states of its Base and Follower bodies.

    Args:
        T_B_j: The absolute pose of the Base body, in world coordinates.
        T_F_j: The absolute pose of the Follower body, in world coordinates.
        u_B_j: The absolute twist of the Base body, in world coordinates.
        u_F_j: The absolute twist of the Follower body, in world coordinates.
        B_r_Bj: The position of the joint on the Base body, in local coordinates.
        F_r_Fj: The position of the joint on the Follower body, in local coordinates.
        X_Bj: The joint frame on the Base body, in local coordinates.
        X_Fj: The joint frame on the Follower body, in local coordinates.

    Returns:
        The absolute pose of the joint frame in world coordinates,
        and two 6D vectors encoding the relative motion of the bodies in the frame of the joint.
    """

    # Joint frames as quaternions, on the parent and follower sides
    q_X_Bj = wp.quat_from_matrix(X_Bj)
    q_X_Fj = wp.quat_from_matrix(X_Fj)

    # Extract the decomposed state of the Base body
    r_B_j = wp.transform_get_translation(T_B_j)
    q_B_j = wp.transform_get_rotation(T_B_j)
    v_B_j = wp.spatial_top(u_B_j)
    omega_B_j = wp.spatial_bottom(u_B_j)

    # Extract the decomposed state of the Follower body
    r_F_j = wp.transform_get_translation(T_F_j)
    q_F_j = wp.transform_get_rotation(T_F_j)
    v_F_j = wp.spatial_top(u_F_j)
    omega_F_j = wp.spatial_bottom(u_F_j)

    # Local joint frame quantities
    r_Bj = wp.quat_rotate(q_B_j, B_r_Bj)
    q_Bj = q_B_j * q_X_Bj
    r_Fj = wp.quat_rotate(q_F_j, F_r_Fj)
    q_Fj = q_F_j * q_X_Fj

    # Compute the pose of the joint frame via the Base body
    r_j_B = r_B_j + r_Bj
    r_j_F = r_F_j + r_Fj
    p_j = wp.transformation(r_j_B, q_Bj, dtype=wp.float32)
    r_j = r_j_F - r_j_B

    # Compute the relative pose between the representations of joint frame w.r.t. the two bodies
    # NOTE: The pose is decomposed into a translation vector `j_r_j` and a rotation quaternion `j_q_j`
    j_r_j = wp.quat_rotate_inv(q_Bj, r_j)
    j_q_j = wp.quat_inverse(q_Bj) * q_Fj

    # Compute the 6D relative twist vector between the representations of joint frame w.r.t. the two bodies
    # TODO: How can we simplify this expression and make it more efficient?
    j_v_j = wp.quat_rotate_inv(q_Bj, v_F_j - v_B_j + wp.cross(omega_F_j, r_Fj) - wp.cross(omega_B_j, r_Bj + r_j))
    j_omega_j = wp.quat_rotate_inv(q_Bj, omega_F_j - omega_B_j)
    j_u_j = wp.spatial_vectorf(*j_v_j, *j_omega_j)

    # Return the computed joint frame pose and relative motion vectors
    return p_j, j_r_j, j_q_j, j_u_j


@wp.func
def joint_pd_spherical_position_error(
    coords_offset: wp.int32,
    data_joint_q_j: wp.array[wp.float32],
    data_joint_q_j_ref: wp.array[wp.float32],
) -> wp.vec3f:
    """Return the 3D rotation-vector position error for a spherical joint."""
    q_j = wp.quatf(
        data_joint_q_j[coords_offset + 0],
        data_joint_q_j[coords_offset + 1],
        data_joint_q_j[coords_offset + 2],
        data_joint_q_j[coords_offset + 3],
    )
    q_j_ref = wp.quatf(
        data_joint_q_j_ref[coords_offset + 0],
        data_joint_q_j_ref[coords_offset + 1],
        data_joint_q_j_ref[coords_offset + 2],
        data_joint_q_j_ref[coords_offset + 3],
    )
    return quat_log(q_j_ref * wp.quat_inverse(q_j))


@wp.func
def compute_and_write_joint_implicit_dynamics(
    # Constants:
    dt: wp.float32,
    dof_type: wp.int32,
    coords_offset: wp.int32,
    dofs_offset: wp.int32,
    num_dynamic_cts: wp.int32,
    dynamic_cts_offset: wp.int32,
    dynamic_cts_axis: wp.array[wp.int32],
    # Inputs:
    model_joint_a_j: wp.array[wp.float32],
    model_joint_b_j: wp.array[wp.float32],
    model_joint_k_p_j: wp.array[wp.float32],
    model_joint_k_d_j: wp.array[wp.float32],
    model_joint_dof_act_types: wp.array[wp.int32],
    model_joint_dof_act_paths: wp.array[wp.int32],
    data_joint_q_j: wp.array[wp.float32],
    data_joint_dq_j: wp.array[wp.float32],
    data_joint_tau_j: wp.array[wp.float32],
    data_joint_q_j_ref: wp.array[wp.float32],
    data_joint_dq_j_ref: wp.array[wp.float32],
    data_joint_tau_j_ref: wp.array[wp.float32],  # Can be `None`
    # Outputs:
    data_joint_m_j: wp.array[wp.float32],
    data_joint_inv_m_j: wp.array[wp.float32],
    data_joint_dq_b_j: wp.array[wp.float32],
):
    # Iterate over the dynamic constraints of the joint and
    # compute and store the implicit dynamics intermediates.
    q_j_err_spherical = wp.vec3f(0.0)
    if dof_type == JointDoFType.SPHERICAL:
        q_j_err_spherical = joint_pd_spherical_position_error(coords_offset, data_joint_q_j, data_joint_q_j_ref)

    for j in range(num_dynamic_cts):
        dynamic_cts_offset_j = dynamic_cts_offset + j
        axis = dynamic_cts_axis[dynamic_cts_offset_j]
        dofs_offset_j = dofs_offset + axis
        act_type = model_joint_dof_act_types[dofs_offset_j]
        include_actuation = model_joint_dof_act_paths[dofs_offset_j] == DofActuationPath.DYNAMIC_CTS

        # Retrieve the current joint state
        # TODO: How can we avoid the extra memory load and
        # instead just get them from `make_write_joint_data`?
        dq_j = data_joint_dq_j[dofs_offset_j]

        # Retrieve the implicit joint dynamics and PD control parameters
        a_j = model_joint_a_j[dofs_offset_j]
        b_j = model_joint_b_j[dofs_offset_j]
        k_p_j = model_joint_k_p_j[dofs_offset_j]
        k_d_j = model_joint_k_d_j[dofs_offset_j]

        # Retrieve external load
        tau_j = data_joint_tau_j[dofs_offset_j]

        # Retrieve PD control references
        pd_dq_j_ref = data_joint_dq_j_ref[dofs_offset_j]
        pd_tau_j_ff = data_joint_tau_j_ref[dofs_offset_j] if data_joint_tau_j_ref else 0.0
        if dof_type == JointDoFType.SPHERICAL:
            q_j_err = q_j_err_spherical[axis]
        else:
            coord = coords_offset + axis
            q_j_err = data_joint_q_j_ref[coord] - data_joint_q_j[coord]

        m_j = a_j + dt * b_j
        tau_j_tot = 0.0
        if include_actuation:
            tau_j_tot = tau_j
            if act_type == JointActuationType.FORCE:
                tau_j_tot += pd_tau_j_ff
            elif act_type == JointActuationType.POSITION:
                m_j += dt * k_d_j + dt * dt * k_p_j
                tau_j_tot += k_p_j * q_j_err
            elif act_type == JointActuationType.VELOCITY:
                m_j += dt * k_d_j
                tau_j_tot += k_d_j * pd_dq_j_ref
            elif act_type == JointActuationType.POSITION_VELOCITY:
                m_j += dt * k_d_j + dt * dt * k_p_j
                tau_j_tot += k_p_j * q_j_err + k_d_j * pd_dq_j_ref
            elif act_type == JointActuationType.POSITION_VELOCITY_FORCE:
                m_j += dt * k_d_j + dt * dt * k_p_j
                tau_j_tot += pd_tau_j_ff + k_p_j * q_j_err + k_d_j * pd_dq_j_ref

        # Enforce minimum mass to avoid division by zero
        m_j = wp.max(1e-6, m_j)
        inv_m_j = 1.0 / m_j
        h_j = a_j * dq_j + dt * tau_j_tot
        dq_b_j = inv_m_j * h_j

        # Store the resulting joint dynamics intermediates
        data_joint_m_j[dynamic_cts_offset_j] = m_j
        data_joint_inv_m_j[dynamic_cts_offset_j] = inv_m_j
        data_joint_dq_b_j[dynamic_cts_offset_j] = dq_b_j


@wp.func
def compute_and_write_joint_effort_dynamics(
    # Constants:
    dt: wp.float32,
    dof_type: wp.int32,
    coords_offset: wp.int32,
    dofs_offset: wp.int32,
    num_effort_cts: wp.int32,
    effort_cts_offset: wp.int32,
    effort_cts_axis: wp.array[wp.int32],
    # Inputs:
    model_joint_k_p_j: wp.array[wp.float32],
    model_joint_k_d_j: wp.array[wp.float32],
    model_joint_tau_j_max: wp.array[wp.float32],
    model_joint_dof_act_types: wp.array[wp.int32],
    data_joint_q_j: wp.array[wp.float32],
    data_joint_tau_j: wp.array[wp.float32],
    data_joint_q_j_ref: wp.array[wp.float32],
    data_joint_dq_j_ref: wp.array[wp.float32],
    data_joint_tau_j_ref: wp.array[wp.float32],  # Can be `None`
    # Outputs:
    data_joint_inv_m_a: wp.array[wp.float32],
    data_joint_dq_b_a: wp.array[wp.float32],
    data_joint_bound_a: wp.array[wp.float32],
):
    # Iterate over the effort constraints of the joint and
    # compute and store the effort dynamics intermediates.
    q_j_err_spherical = wp.vec3f(0.0)
    if dof_type == JointDoFType.SPHERICAL:
        q_j_err_spherical = joint_pd_spherical_position_error(coords_offset, data_joint_q_j, data_joint_q_j_ref)

    for j in range(num_effort_cts):
        effort_cts_offset_j = effort_cts_offset + j
        axis = effort_cts_axis[effort_cts_offset_j]
        dofs_offset_j = dofs_offset + axis
        act_type = model_joint_dof_act_types[dofs_offset_j]

        # Retrieve the effort limit and PD control parameters
        k_p_j = model_joint_k_p_j[dofs_offset_j]
        k_d_j = model_joint_k_d_j[dofs_offset_j]
        tau_j_max = model_joint_tau_j_max[dofs_offset_j]

        # Retrieve external load
        tau_j = data_joint_tau_j[dofs_offset_j]

        # Retrieve PD control references
        pd_dq_j_ref = data_joint_dq_j_ref[dofs_offset_j]
        pd_tau_j_ff = data_joint_tau_j_ref[dofs_offset_j] if data_joint_tau_j_ref else 0.0
        if dof_type == JointDoFType.SPHERICAL:
            q_j_err = q_j_err_spherical[axis]
        else:
            coord = coords_offset + axis
            q_j_err = data_joint_q_j_ref[coord] - data_joint_q_j[coord]

        # All actuation types possible for effort constraints have an active k_d,
        # even POSITION, which has k_d with zero reference.
        # All except VELOCITY have an active k_p.
        m_a = dt * k_d_j
        if act_type != JointActuationType.VELOCITY:
            m_a += dt * dt * k_p_j

        # If the effort constraint exists, it always contains the actuation.
        tau_j_tot = tau_j
        if act_type == JointActuationType.POSITION:
            tau_j_tot += k_p_j * q_j_err
        elif act_type == JointActuationType.VELOCITY:
            tau_j_tot += k_d_j * pd_dq_j_ref
        elif act_type == JointActuationType.POSITION_VELOCITY:
            tau_j_tot += k_p_j * q_j_err + k_d_j * pd_dq_j_ref
        elif act_type == JointActuationType.POSITION_VELOCITY_FORCE:
            tau_j_tot += pd_tau_j_ff + k_p_j * q_j_err + k_d_j * pd_dq_j_ref

        # Enforce minimum effective actuator inertia to cap compliance and avoid division by zero.
        m_a = wp.max(1e-6, m_a)
        inv_m_a = 1.0 / m_a
        dq_b_a = inv_m_a * dt * tau_j_tot
        bound_a = dt * tau_j_max

        # Store the resulting effort dynamics intermediates
        data_joint_inv_m_a[effort_cts_offset_j] = inv_m_a
        data_joint_dq_b_a[effort_cts_offset_j] = dq_b_a
        data_joint_bound_a[effort_cts_offset_j] = bound_a


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
        model_time_dt: wp.array[wp.float32],
        model_joint_wid: wp.array[wp.int32],
        model_joint_dof_type: wp.array[wp.int32],
        model_joint_dof_act_types: wp.array[wp.int32],
        model_joint_coords_offset: wp.array[wp.int32],
        model_joint_dofs_offset: wp.array[wp.int32],
        model_joint_dynamic_cts_offset: wp.array[wp.int32],
        model_joint_kinematic_cts_offset: wp.array[wp.int32],
        model_joint_bid_B: wp.array[wp.int32],
        model_joint_bid_F: wp.array[wp.int32],
        model_joint_B_r_Bj: wp.array[wp.vec3f],
        model_joint_F_r_Fj: wp.array[wp.vec3f],
        model_joint_X_Bj: wp.array[wp.mat33f],
        model_joint_X_Fj: wp.array[wp.mat33f],
        model_joint_a_j: wp.array[wp.float32],
        model_joint_b_j: wp.array[wp.float32],
        model_joint_k_p_j: wp.array[wp.float32],
        model_joint_k_d_j: wp.array[wp.float32],
        model_joint_tau_j_max: wp.array[wp.float32],
        model_joint_num_effort_cts: wp.array[wp.int32],
        model_joint_effort_cts_offset: wp.array[wp.int32],
        model_joint_dynamic_cts_axis: wp.array[wp.int32],
        model_joint_effort_cts_axis: wp.array[wp.int32],
        model_joint_dof_act_paths: wp.array[wp.int32],
        data_body_q_i: wp.array[wp.transformf],
        data_body_u_i: wp.array[wp.spatial_vectorf],
        data_joint_tau_j: wp.array[wp.float32],
        data_joint_q_j_ref: wp.array[wp.float32],
        data_joint_dq_j_ref: wp.array[wp.float32],
        data_joint_tau_j_ref: wp.array[wp.float32],
        q_j_p: wp.array[wp.float32],
        # Outputs:
        data_joint_p_j: wp.array[wp.transformf],
        data_joint_r_j: wp.array[wp.float32],
        data_joint_dr_j: wp.array[wp.float32],
        data_joint_q_j: wp.array[wp.float32],
        data_joint_dq_j: wp.array[wp.float32],
        data_joint_m_j: wp.array[wp.float32],
        data_joint_inv_m_j: wp.array[wp.float32],
        data_joint_dq_b_j: wp.array[wp.float32],
        data_joint_inv_m_a: wp.array[wp.float32],
        data_joint_dq_b_a: wp.array[wp.float32],
        data_joint_bound_a: wp.array[wp.float32],
    ):
        # Retrieve the thread index
        jid = wp.tid()

        # Retrieve the joint model data
        wid = model_joint_wid[jid]
        dof_type = model_joint_dof_type[jid]
        bid_B = model_joint_bid_B[jid]
        bid_F = model_joint_bid_F[jid]
        B_r_Bj = model_joint_B_r_Bj[jid]
        F_r_Fj = model_joint_F_r_Fj[jid]
        X_Bj = model_joint_X_Bj[jid]
        X_Fj = model_joint_X_Fj[jid]

        # Retrieve the time step
        dt = model_time_dt[wid]

        # Retrieve joint-specific offsets/sizes
        coords_offset = model_joint_coords_offset[jid]
        dofs_offset = model_joint_dofs_offset[jid]
        dynamic_cts_offset = model_joint_dynamic_cts_offset[jid]
        num_dynamic_cts = model_joint_dynamic_cts_offset[jid + 1] - dynamic_cts_offset
        kinematic_cts_offset = model_joint_kinematic_cts_offset[jid]
        num_kinematic_cts = model_joint_kinematic_cts_offset[jid + 1] - kinematic_cts_offset
        if num_kinematic_cts == wp.int32(0):
            kinematic_cts_offset = wp.int32(-1)

        # If the Base body is the world (bid=-1), use the identity transform (frame
        # of the world's origin), otherwise retrieve the Base body's pose and twist
        T_B_j = wp.transform_identity(dtype=wp.float32)
        u_B_j = wp.spatial_vectorf(0.0)
        if bid_B > -1:
            T_B_j = data_body_q_i[bid_B]
            u_B_j = data_body_u_i[bid_B]

        # Retrieve the Follower body's pose and twist
        T_F_j = data_body_q_i[bid_F]
        u_F_j = data_body_u_i[bid_F]

        # Compute the joint frame pose and relative motion
        p_j, j_r_j, j_q_j, j_u_j = compute_joint_pose_and_relative_motion(
            T_B_j, T_F_j, u_B_j, u_F_j, B_r_Bj, F_r_Fj, X_Bj, X_Fj
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
            dof_type,
            coords_offset,
            dofs_offset,
            num_dynamic_cts,
            dynamic_cts_offset,
            model_joint_dynamic_cts_axis,
            model_joint_a_j,
            model_joint_b_j,
            model_joint_k_p_j,
            model_joint_k_d_j,
            model_joint_dof_act_types,
            model_joint_dof_act_paths,
            data_joint_q_j,
            data_joint_dq_j,
            data_joint_tau_j,
            data_joint_q_j_ref,
            data_joint_dq_j_ref,
            data_joint_tau_j_ref,
            data_joint_m_j,
            data_joint_inv_m_j,
            data_joint_dq_b_j,
        )
        # Compute and store the effort dynamics
        # for the effort constraints of the joint
        compute_and_write_joint_effort_dynamics(
            dt,
            dof_type,
            coords_offset,
            dofs_offset,
            model_joint_num_effort_cts[jid],
            model_joint_effort_cts_offset[jid],
            model_joint_effort_cts_axis,
            model_joint_k_p_j,
            model_joint_k_d_j,
            model_joint_tau_j_max,
            model_joint_dof_act_types,
            data_joint_q_j,
            data_joint_tau_j,
            data_joint_q_j_ref,
            data_joint_dq_j_ref,
            data_joint_tau_j_ref,
            data_joint_inv_m_a,
            data_joint_dq_b_a,
            data_joint_bound_a,
        )

    # Return the kernel
    return _compute_joints_data


@wp.kernel
def _extract_actuators_state_from_joints(
    # Inputs:
    world_mask: wp.array[wp.bool],  # None also supported
    model_joint_wid: wp.array[wp.int32],
    model_joint_act_type: wp.array[wp.int32],
    model_joint_coords_offset: wp.array[wp.int32],
    model_joint_dofs_offset: wp.array[wp.int32],
    model_joint_actuated_coords_offset: wp.array[wp.int32],
    model_joint_actuated_dofs_offset: wp.array[wp.int32],
    joint_q: wp.array[wp.float32],
    joint_u: wp.array[wp.float32],
    # Outputs:
    actuator_q: wp.array[wp.float32],
    actuator_u: wp.array[wp.float32],
):
    # Retrieve the joint index from the thread grid
    jid = wp.tid()

    # Retrieve the world index and actuation type of the joint
    wid = model_joint_wid[jid]
    act_type = model_joint_act_type[jid]

    # Early exit the operation if the joint's world is flagged as skipped or if the joint is not actuated
    if (world_mask and not world_mask[wid]) or act_type == JointActuationType.PASSIVE:
        return

    # Retrieve the joint model data
    jq_start = model_joint_coords_offset[jid]
    num_coords = model_joint_coords_offset[jid + 1] - jq_start
    jd_start = model_joint_dofs_offset[jid]
    num_dofs = model_joint_dofs_offset[jid + 1] - jd_start
    aq_start = model_joint_actuated_coords_offset[jid]
    ad_start = model_joint_actuated_dofs_offset[jid]

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
    world_mask: wp.array[wp.bool],
    model_joint_wid: wp.array[wp.int32],
    model_joint_act_type: wp.array[wp.int32],
    model_joint_coords_offset: wp.array[wp.int32],
    model_joint_dofs_offset: wp.array[wp.int32],
    model_joint_actuated_coords_offset: wp.array[wp.int32],
    model_joint_actuated_dofs_offset: wp.array[wp.int32],
    actuator_q: wp.array[wp.float32],
    actuator_u: wp.array[wp.float32],
    # Outputs:
    joint_q: wp.array[wp.float32],
    joint_u: wp.array[wp.float32],
):
    # Retrieve the joint index from the thread grid
    jid = wp.tid()

    # Retrieve the world index and actuation type of the joint
    wid = model_joint_wid[jid]
    act_type = model_joint_act_type[jid]

    # Early exit the operation if the joint's world is flagged as skipped or if the joint is not actuated
    if not world_mask[wid] or act_type == JointActuationType.PASSIVE:
        return

    # Retrieve the joint model data
    jq_start = model_joint_coords_offset[jid]
    num_coords = model_joint_coords_offset[jid + 1] - jq_start
    jd_start = model_joint_dofs_offset[jid]
    num_dofs = model_joint_dofs_offset[jid + 1] - jd_start
    aq_start = model_joint_actuated_coords_offset[jid]
    ad_start = model_joint_actuated_dofs_offset[jid]

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
    q_j_p: wp.array[wp.float32],
    correction: JointCorrectionMode = JointCorrectionMode.TWOPI,
) -> None:
    """
    Computes the states of the joints based on the current body states.

    The computed joint state data includes both the generalized coordinates and velocities
    corresponding to the respective degrees of freedom (DoFs), as well as the constraint-space
    residuals and velocities of the applied bilateral constraints.

    Args:
        model: The model container holding the time-invariant data of the simulation.
        q_j_p: An array of previous joint DoF coordinates used for coordinate correction.
            Only used for revolute DoFs of the relevant joints to enforce angle continuity.
            Shape of ``(sum_of_num_joint_coords,)``.
        data: The solver data container holding the internal time-varying state of the simulation.
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
            model.time.dt,
            model.joints.wid,
            model.joints.dof_type,
            model.joints.dof_act_types,
            model.joints.coords_offset,
            model.joints.dofs_offset,
            model.joints.dynamic_cts_offset,
            model.joints.kinematic_cts_offset,
            model.joints.bid_B,
            model.joints.bid_F,
            model.joints.B_r_Bj,
            model.joints.F_r_Fj,
            model.joints.X_Bj,
            model.joints.X_Fj,
            model.joints.a_j,
            model.joints.b_j,
            model.joints.k_p_j,
            model.joints.k_d_j,
            model.joints.tau_j_max,
            model.joints.num_effort_cts,
            model.joints.effort_cts_offset,
            model.joints.dynamic_cts_axis,
            model.joints.effort_cts_axis,
            model.joints.dof_act_paths,
            data.bodies.q_i,
            data.bodies.u_i,
            data.joints.tau_j,
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
            data.joints.inv_m_a,
            data.joints.dq_b_a,
            data.joints.bound_a,
        ],
        device=model.device,
    )


def extract_actuators_state_from_joints(
    model: ModelKamino,
    joint_q: wp.array[wp.float32],
    joint_u: wp.array[wp.float32],
    actuator_q: wp.array[wp.float32],
    actuator_u: wp.array[wp.float32],
    world_mask: wp.array[wp.bool] | None = None,
):
    """
    Extracts the states of the actuated joints from the full joint state arrays.

    Only joints that are marked as actuated and belong to worlds
    that are not masked will have their states extracted.

    Args:
        model: The model container holding the time-invariant data of the simulation.
        joint_q: The full array of joint coordinates.
            Shape of ``(sum_of_num_joint_coords,)``.
        joint_u: The full array of joint velocities.
            Shape of ``(sum_of_num_joint_dofs,)``.
        actuator_q: The output array to store the actuated joint coordinates.
            Shape of ``(sum_of_num_actuated_joint_coords,)``.
        actuator_u: The output array to store the actuated joint velocities.
            Shape of ``(sum_of_actuated_joint_dofs,)``.
        world_mask: Per-world boolean mask. If provided, indicates in which worlds to perform the operation.
            Shape of ``(num_worlds,)``.
    """
    wp.launch(
        _extract_actuators_state_from_joints,
        dim=model.size.sum_of_num_joints,
        inputs=[
            world_mask,
            model.joints.wid,
            model.joints.act_type,
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
    world_mask: wp.array[wp.bool],
    actuator_q: wp.array[wp.float32],
    actuator_u: wp.array[wp.float32],
    joint_q: wp.array[wp.float32],
    joint_u: wp.array[wp.float32],
):
    """
    Extracts the states of the actuated joints from the full joint state arrays.

    Only joints that are marked as actuated and belong to worlds
    that are not masked will have their states extracted.

    Args:
        model: The model container holding the time-invariant data of the simulation.
        joint_q: The full array of joint coordinates.
            Shape of ``(sum_of_num_joint_coords,)``.
        joint_u: The full array of joint velocities.
            Shape of ``(sum_of_num_joint_dofs,)``.
        actuator_q: The output array to store the actuated joint coordinates.
            Shape of ``(sum_of_num_actuated_joint_coords,)``.
        actuator_u: The output array to store the actuated joint velocities.
            Shape of ``(sum_of_actuated_joint_dofs,)``.
        world_mask: An array indicating which worlds are active (True) or skipped (False).
            Shape of ``(num_worlds,)``.
    """
    wp.launch(
        _extract_joints_state_from_actuators,
        dim=model.size.sum_of_num_joints,
        inputs=[
            world_mask,
            model.joints.wid,
            model.joints.act_type,
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
