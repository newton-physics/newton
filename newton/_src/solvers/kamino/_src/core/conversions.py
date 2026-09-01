# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Provides a set of conversion utilities to bridge Kamino and Newton."""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .....geometry import ShapeFlags
from .....sim.model import Model
from ....coupled.model_view import ModelView
from ..utils import logger as msg
from .bodies import (
    RigidBodiesModel,
    convert_body_origin_to_com,
    convert_geom_offset_origin_to_com,
)
from .geometry import GeometriesModel
from .joints import (
    JOINT_QMAX,
    JOINT_QMIN,
    DofActuationPath,
    JointActuationType,
    JointDoFType,
    JointsModel,
    _validate_implicit_pd_gains,
    has_dynamic_cts_wp,
    has_effort_cts_wp,
    has_friction_cts_wp,
)
from .materials import MaterialDescriptor, MaterialManager
from .shapes import max_contacts_for_shape_pair
from .size import SizeKamino
from .types import mat63f, to_warp_int32_array, vec6f

if TYPE_CHECKING:
    from ..core.model import ModelKamino, ModelKaminoInfo

###
# Module interface
###

__all__ = [
    "StructuralUpdateViolation",
    "convert_geometries",
    "convert_joints",
    "convert_model_joint_actuation",
    "convert_model_joint_transforms",
    "convert_model_materials",
    "convert_rigid_bodies",
    "convert_target_coords_to_target_dofs",
    "convert_target_dofs_to_target_coords",
    "validate_model_structural_updates",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


class StructuralUpdateViolation(IntEnum):
    """Indices into the structural-update validation violations array."""

    DYNAMIC_CTS = 0
    LIMIT_FINITE = 1
    ACTUATION_PARTITION = 2
    INVALID_TARGET_MODE = 3
    NONORTHONORMAL_AXES = 4
    GIMBAL_HANDEDNESS = 5
    MASSLESS = 6
    FRICTION_CTS = 7
    EFFORT_CTS = 8


###
# Kernels
###


@wp.kernel
def world_max_contacts_kernel(
    # Inputs:
    max_contacts_per_pair: int,
    model_shape_type: wp.array[wp.int32],
    model_shape_world: wp.array[wp.int32],
    model_shape_contact_pair: wp.array[wp.vec2i],
    # Outputs:
    world_max_contacts: wp.array[wp.int32],
):
    # Retrieve the shape pair index from the thread grid
    shape_pair_id = wp.tid()

    # Extract the shape types for this pair.
    shape_pair = model_shape_contact_pair[shape_pair_id]
    shape_type_a = model_shape_type[shape_pair[0]]
    shape_type_b = model_shape_type[shape_pair[1]]

    # Determine the world for this pair — fall back to other shape if one is global
    world_id_a = model_shape_world[shape_pair[0]]
    world_id_b = model_shape_world[shape_pair[1]]
    world_id = world_id_a if world_id_a >= 0 else world_id_b
    if world_id < 0:
        return  # Both shapes are global — skip

    # Compute max contact count for this pair and add to world total,
    # ensuring shapes are ordered by type for consistent contact counts.
    if shape_type_a > shape_type_b:
        shape_type_a, shape_type_b = shape_type_b, shape_type_a
    num_contacts_a, num_contacts_b = max_contacts_for_shape_pair(
        type_a=shape_type_a,
        type_b=shape_type_b,
    )
    num_contacts = num_contacts_a + num_contacts_b
    if max_contacts_per_pair >= 0:
        num_contacts = min(num_contacts, max_contacts_per_pair)
    wp.atomic_add(world_max_contacts, world_id, num_contacts)


@wp.kernel
def material_first_shape_kernel(
    # Inputs:
    geom_material: wp.array[wp.int32],
    # Outputs:
    first_shape: wp.array[wp.int32],
):
    """Record the first shape index associated with each material."""
    shape = wp.tid()
    material = geom_material[shape]
    if material >= 0:
        wp.atomic_min(first_shape, material, shape)


@wp.kernel
def validate_material_update_kernel(
    shape_friction: wp.array[wp.float32],
    shape_restitution: wp.array[wp.float32],
    geom_material: wp.array[wp.int32],
    first_shape: wp.array[wp.int32],
    conflict_material: wp.array[wp.int32],
):
    """Find the first material whose shapes have conflicting properties."""
    shape = wp.tid()
    material = geom_material[shape]
    if material < 0:
        return
    representative = first_shape[material]
    if (
        shape_friction[shape] != shape_friction[representative]
        or shape_restitution[shape] != shape_restitution[representative]
    ):
        wp.atomic_min(conflict_material, 0, material)


@wp.kernel
def update_materials_kernel(
    # Inputs:
    shape_friction: wp.array[wp.float32],
    shape_restitution: wp.array[wp.float32],
    first_shape: wp.array[wp.int32],
    shape_count: int,
    # Outputs:
    restitution: wp.array[wp.float32],
    static_friction: wp.array[wp.float32],
    dynamic_friction: wp.array[wp.float32],
    pair_restitution: wp.array[wp.float32],
    pair_static_friction: wp.array[wp.float32],
    pair_dynamic_friction: wp.array[wp.float32],
):
    """Update Kamino material properties from cached representative shapes.

    The material-zero properties are also copied to the default material pair.
    """
    material = wp.tid()
    shape = first_shape[material]
    if shape < shape_count:
        friction = shape_friction[shape]
        restitution[material] = shape_restitution[shape]
        static_friction[material] = friction
        dynamic_friction[material] = friction
        if material == 0:
            pair_restitution[0] = shape_restitution[shape]
            pair_static_friction[0] = friction
            pair_dynamic_friction[0] = friction


@wp.kernel
def validate_joint_dof_updates_kernel(
    # Inputs:
    joint_qd_start: wp.array[wp.int32],
    joint_armature: wp.array[wp.float32],
    joint_damping: wp.array[wp.float32],
    joint_target_ke: wp.array[wp.float32],
    joint_target_kd: wp.array[wp.float32],
    joint_target_mode: wp.array[wp.int32],
    joint_effort_limit: wp.array[wp.float32],
    joint_friction: wp.array[wp.float32],
    joint_dof_type: wp.array[wp.int32],
    dynamic_cts_offset: wp.array[wp.int32],
    dynamic_cts_axis: wp.array[wp.int32],
    friction_cts_offset: wp.array[wp.int32],
    friction_cts_axis: wp.array[wp.int32],
    effort_cts_offset: wp.array[wp.int32],
    effort_cts_axis: wp.array[wp.int32],
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
    built_limit_finite: wp.array[wp.int32],
    joint_count: int,
    dof_count: int,
    # Outputs:
    violations: wp.array[wp.int32],
):
    """Find the first structural change to joint degree-of-freedom properties."""
    tid = wp.tid()
    if tid < joint_count:
        dof_start = joint_qd_start[tid]
        dof_end = joint_qd_start[tid + 1]
        dynamic_row = dynamic_cts_offset[tid]
        dynamic_row_end = dynamic_cts_offset[tid + 1]
        friction_row = friction_cts_offset[tid]
        friction_row_end = friction_cts_offset[tid + 1]
        effort_row = effort_cts_offset[tid]
        effort_row_end = effort_cts_offset[tid + 1]
        for axis in range(dof_end - dof_start):
            dof = dof_start + axis
            act_type = JointActuationType.from_newton_wp(joint_target_mode[dof])
            if act_type < 0:
                wp.atomic_min(violations, StructuralUpdateViolation.INVALID_TARGET_MODE, tid)
                return
            dynamic_required = has_dynamic_cts_wp(
                act_type,
                joint_target_ke[dof],
                joint_target_kd[dof],
                joint_effort_limit[dof],
                joint_armature[dof],
                joint_damping[dof],
            )
            dynamic_built = dynamic_row < dynamic_row_end and dynamic_cts_axis[dynamic_row] == axis
            if dynamic_required != dynamic_built:
                wp.atomic_min(violations, StructuralUpdateViolation.DYNAMIC_CTS, tid)
            if dynamic_built:
                dynamic_row += 1

            friction_required = has_friction_cts_wp(joint_dof_type[tid], joint_friction[dof])
            friction_built = friction_row < friction_row_end and friction_cts_axis[friction_row] == axis
            if friction_required and not friction_built:
                wp.atomic_min(violations, StructuralUpdateViolation.FRICTION_CTS, tid)
            if friction_built:
                friction_row += 1

            effort_required = has_effort_cts_wp(
                act_type,
                joint_target_ke[dof],
                joint_target_kd[dof],
                joint_effort_limit[dof],
            )
            effort_built = effort_row < effort_row_end and effort_cts_axis[effort_row] == axis
            if effort_required != effort_built:
                wp.atomic_min(violations, StructuralUpdateViolation.EFFORT_CTS, tid)
            if effort_built:
                effort_row += 1

    if tid < dof_count:
        current_finite = joint_limit_lower[tid] > JOINT_QMIN or joint_limit_upper[tid] < JOINT_QMAX
        if current_finite != (built_limit_finite[tid] != 0):
            wp.atomic_min(violations, StructuralUpdateViolation.LIMIT_FINITE, tid)


@wp.kernel
def validate_joint_actuation_updates_kernel(
    # Inputs:
    joint_qd_start: wp.array[wp.int32],
    joint_target_mode: wp.array[wp.int32],
    act_type: wp.array[wp.int32],
    # Outputs:
    violations: wp.array[wp.int32],
):
    """Find the first joint with an invalid or structurally changed actuation type."""
    joint = wp.tid()
    current_actuation = JointActuationType.aggregate_from_newton_wp(
        joint_qd_start[joint],
        joint_qd_start[joint + 1],
        joint_target_mode,
    )
    if current_actuation < 0:
        wp.atomic_min(violations, StructuralUpdateViolation.INVALID_TARGET_MODE, joint)
    elif (current_actuation == JointActuationType.PASSIVE) != (act_type[joint] == JointActuationType.PASSIVE):
        wp.atomic_min(violations, StructuralUpdateViolation.ACTUATION_PARTITION, joint)


@wp.kernel
def validate_joint_axes_kernel(
    # Inputs:
    joint_qd_start: wp.array[wp.int32],
    joint_axis: wp.array[wp.vec3f],
    joint_dof_type: wp.array[wp.int32],
    # Outputs:
    violations: wp.array[wp.int32],
):
    """Find the first universal or gimbal joint with invalid axis configuration."""
    joint = wp.tid()
    dof_type = joint_dof_type[joint]
    is_universal = dof_type == JointDoFType.UNIVERSAL
    is_gimbal = dof_type == JointDoFType.GIMBAL or dof_type == JointDoFType.GIMBAL_LEFT_HANDED
    if not is_universal and not is_gimbal:
        return

    dof_start = joint_qd_start[joint]
    axis_0 = joint_axis[dof_start]
    axis_1 = joint_axis[dof_start + 1]
    valid = (
        wp.isfinite(axis_0[0])
        and wp.isfinite(axis_0[1])
        and wp.isfinite(axis_0[2])
        and wp.isfinite(axis_1[0])
        and wp.isfinite(axis_1[1])
        and wp.isfinite(axis_1[2])
        and wp.abs(wp.dot(axis_0, axis_0) - 1.0) <= 1.0e-6
        and wp.abs(wp.dot(axis_1, axis_1) - 1.0) <= 1.0e-6
        and wp.abs(wp.dot(axis_0, axis_1)) <= 1.0e-6
    )
    if is_gimbal:
        axis_2 = joint_axis[dof_start + 2]
        valid = (
            valid
            and wp.isfinite(axis_2[0])
            and wp.isfinite(axis_2[1])
            and wp.isfinite(axis_2[2])
            and wp.abs(wp.dot(axis_2, axis_2) - 1.0) <= 1.0e-6
            and wp.abs(wp.dot(axis_0, axis_2)) <= 1.0e-6
            and wp.abs(wp.dot(axis_1, axis_2)) <= 1.0e-6
        )
        if valid:
            left_handed = wp.dot(wp.cross(axis_0, axis_1), axis_2) < 0.0
            expected_left_handed = dof_type == JointDoFType.GIMBAL_LEFT_HANDED
            if left_handed != expected_left_handed:
                wp.atomic_min(violations, StructuralUpdateViolation.GIMBAL_HANDEDNESS, joint)
                return
    if not valid:
        wp.atomic_min(violations, StructuralUpdateViolation.NONORTHONORMAL_AXES, joint)


@wp.func
def has_zero_inverse_inertia(inv_inertia: wp.mat33f) -> bool:
    """Return whether every element of an inverse inertia matrix is zero."""
    return (
        inv_inertia[0, 0] == 0.0
        and inv_inertia[0, 1] == 0.0
        and inv_inertia[0, 2] == 0.0
        and inv_inertia[1, 0] == 0.0
        and inv_inertia[1, 1] == 0.0
        and inv_inertia[1, 2] == 0.0
        and inv_inertia[2, 0] == 0.0
        and inv_inertia[2, 1] == 0.0
        and inv_inertia[2, 2] == 0.0
    )


@wp.kernel
def validate_body_inertial_updates_kernel(
    # Inputs:
    body_inv_mass: wp.array[wp.float32],
    body_inv_inertia: wp.array[wp.mat33f],
    built_massless: wp.array[wp.int32],
    # Outputs:
    violations: wp.array[wp.int32],
):
    """Find the first body made massless after constructing SolverKamino."""
    body = wp.tid()
    is_massless = body_inv_mass[body] == 0.0 or has_zero_inverse_inertia(body_inv_inertia[body])
    if is_massless and built_massless[body] == 0:
        wp.atomic_min(violations, StructuralUpdateViolation.MASSLESS, body)


@wp.kernel
def update_joint_actuation_kernel(
    # Inputs:
    joint_qd_start: wp.array[wp.int32],
    dof_act_types: wp.array[wp.int32],
    # Outputs:
    act_type: wp.array[wp.int32],
):
    """Aggregate each joint's Kamino actuation type from its DoF modes."""
    joint = wp.tid()
    act_type[joint] = JointActuationType.aggregate_wp(
        joint_qd_start[joint],
        joint_qd_start[joint + 1],
        dof_act_types,
    )


@wp.kernel
def update_joint_dof_actuation_kernel(
    # Inputs:
    joint_target_mode: wp.array[wp.int32],
    # Outputs:
    dof_act_types: wp.array[wp.int32],
):
    """Update each DoF's Kamino actuation type from its Newton target mode."""
    dof = wp.tid()
    dof_act_types[dof] = JointActuationType.from_newton_wp(joint_target_mode[dof])


@wp.kernel
def rigid_bodies_indexing_kernel(
    # Inputs:
    model_body_world_start: wp.array[wp.int32],
    model_shape_world_start: wp.array[wp.int32],
    # Outputs:
    body_bid: wp.array[wp.int32],
    num_bodies: wp.array[wp.int32],
    num_shapes: wp.array[wp.int32],
    num_body_dofs: wp.array[wp.int32],
    world_body_offset: wp.array[wp.int32],
    world_shape_offset: wp.array[wp.int32],
    world_body_dof_offset: wp.array[wp.int32],
):
    # Retrieve the world index
    world_id = wp.tid()

    # Compute number of bodies/shapes based on world starts
    bodies_start = model_body_world_start[world_id]
    num_bodies_w = model_body_world_start[world_id + 1] - bodies_start
    num_bodies[world_id] = num_bodies_w
    num_shapes[world_id] = model_shape_world_start[world_id + 1] - model_shape_world_start[world_id]
    num_body_dofs[world_id] = 6 * num_bodies[world_id]

    # Fill in in-world index for bodies
    for i in range(num_bodies_w):
        body_bid[bodies_start + i] = i

    # Set world offsets
    world_body_offset[world_id] = model_body_world_start[world_id]
    world_shape_offset[world_id] = model_shape_world_start[world_id]
    world_body_dof_offset[world_id] = 6 * model_body_world_start[world_id]


@wp.kernel
def joint_conversion_kernel(
    # Inputs:
    model_joint_world: wp.array[wp.int32],
    model_joint_world_start: wp.array[wp.int32],
    model_joint_type: wp.array[wp.int32],
    model_joint_dof_dim: wp.array2d[wp.int32],
    model_joint_q_start: wp.array[wp.int32],
    model_joint_qd_start: wp.array[wp.int32],
    model_joint_axis: wp.array[wp.vec3f],
    model_joint_target_mode: wp.array[wp.int32],
    model_joint_target_ke: wp.array[wp.float32],
    model_joint_target_kd: wp.array[wp.float32],
    model_joint_effort_limit: wp.array[wp.float32],
    model_joint_armature: wp.array[wp.float32],
    model_joint_damping: wp.array[wp.float32],
    model_joint_friction: wp.array[wp.float32],
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
    # Outputs:
    joint_jid: wp.array[wp.int32],
    joint_dof_type: wp.array[wp.int32],
    joint_act_type: wp.array[wp.int32],
    joint_dof_act_types: wp.array[wp.int32],
    joint_dof_act_paths: wp.array[wp.int32],
    joint_num_coords: wp.array[wp.int32],
    joint_num_dofs: wp.array[wp.int32],
    joint_num_bilateral_cts: wp.array[wp.int32],
    joint_num_dynamic_cts: wp.array[wp.int32],
    joint_num_kinematic_cts: wp.array[wp.int32],
    joint_num_bounded_cts: wp.array[wp.int32],
    joint_num_friction_cts: wp.array[wp.int32],
    joint_num_effort_cts: wp.array[wp.int32],
):
    # Retrieve the joint index
    joint_id = wp.tid()

    world_id = model_joint_world[joint_id]
    joint_jid[joint_id] = joint_id - model_joint_world_start[world_id]

    # Determine Kamino joint type
    type_j = model_joint_type[joint_id]
    dof_dim_j = wp.vec2i(model_joint_dof_dim[joint_id, 0], model_joint_dof_dim[joint_id, 1])
    q_count_j = model_joint_q_start[joint_id + 1] - model_joint_q_start[joint_id]
    dofs_start_j = model_joint_qd_start[joint_id]
    qd_count_j = model_joint_qd_start[joint_id + 1] - dofs_start_j
    limit_upper_j = vec6f()
    limit_lower_j = vec6f()
    dof_axes_j = mat63f()
    for i in range(qd_count_j):
        limit_upper_j[i] = joint_limit_upper[dofs_start_j + i]
        limit_lower_j[i] = joint_limit_lower[dofs_start_j + i]
        dof_axes_j[i] = model_joint_axis[dofs_start_j + i]
    dof_type_j = JointDoFType.from_newton_wp(
        type_j, q_count_j, qd_count_j, dof_dim_j, limit_lower_j, limit_upper_j, dof_axes_j
    )
    assert dof_type_j >= 0, "Joint DoF type must be valid"

    # Get joint type properties
    ncoords_j = JointDoFType.num_coords_wp(dof_type_j)
    ndofs_j = JointDoFType.num_dofs_wp(dof_type_j)
    num_kinematic_cts_j = JointDoFType.num_cts_wp(dof_type_j)
    assert ncoords_j >= 0, "Number of joint coordinates must be valid"
    assert ndofs_j >= 0, "Number of joint DoFs must be valid"
    assert num_kinematic_cts_j >= 0, "Number of joint constraints must be valid"
    joint_dof_type[joint_id] = dof_type_j
    joint_num_coords[joint_id] = ncoords_j
    joint_num_dofs[joint_id] = ndofs_j
    joint_num_kinematic_cts[joint_id] = num_kinematic_cts_j

    act_type_j = int(JointActuationType.PASSIVE)
    num_dynamic_cts_j = int(0)
    num_friction_cts_j = int(0)
    num_effort_cts_j = int(0)
    for axis in range(qd_count_j):
        dof = dofs_start_j + axis
        dof_act_types = JointActuationType.from_newton_wp(model_joint_target_mode[dof])
        assert dof_act_types >= 0, "Joint actuation type must be valid"
        joint_dof_act_types[dof] = dof_act_types
        act_type_j = max(act_type_j, dof_act_types)

        effort = has_effort_cts_wp(
            dof_act_types,
            model_joint_target_ke[dof],
            model_joint_target_kd[dof],
            model_joint_effort_limit[dof],
        )
        dynamic = has_dynamic_cts_wp(
            dof_act_types,
            model_joint_target_ke[dof],
            model_joint_target_kd[dof],
            model_joint_effort_limit[dof],
            model_joint_armature[dof],
            model_joint_damping[dof],
        )
        friction = has_friction_cts_wp(dof_type_j, model_joint_friction[dof])
        if dynamic:
            num_dynamic_cts_j += 1
        if friction:
            num_friction_cts_j += 1
        if effort:
            num_effort_cts_j += 1
            joint_dof_act_paths[dof] = DofActuationPath.EFFORT_CTS
        elif dynamic:
            joint_dof_act_paths[dof] = DofActuationPath.DYNAMIC_CTS
        else:
            joint_dof_act_paths[dof] = DofActuationPath.BODY_WRENCHES

    joint_act_type[joint_id] = act_type_j
    joint_num_dynamic_cts[joint_id] = num_dynamic_cts_j
    joint_num_friction_cts[joint_id] = num_friction_cts_j
    joint_num_effort_cts[joint_id] = num_effort_cts_j
    joint_num_bilateral_cts[joint_id] = num_kinematic_cts_j + num_dynamic_cts_j
    joint_num_bounded_cts[joint_id] = num_friction_cts_j + num_effort_cts_j


@wp.kernel
def joint_frame_conversion_kernel(
    # Inputs:
    model_joint_parent: wp.array[wp.int32],
    model_joint_child: wp.array[wp.int32],
    model_joint_qd_start: wp.array[wp.int32],
    model_joint_axis: wp.array[wp.vec3f],
    model_body_com: wp.array[wp.vec3f],
    model_joint_X_p: wp.array[wp.transformf],
    model_joint_X_c: wp.array[wp.transformf],
    joint_dof_type: wp.array[wp.int32],
    joint_num_dofs: wp.array[wp.int32],
    # Outputs:
    joint_B_r_B: wp.array[wp.vec3f],
    joint_F_r_F: wp.array[wp.vec3f],
    joint_X_B: wp.array[wp.mat33f],
    joint_X_F: wp.array[wp.mat33f],
):
    # Retrieve the joint index
    joint_id = wp.tid()

    # Get joint type properties
    dof_type_j = joint_dof_type[joint_id]
    ndofs_j = joint_num_dofs[joint_id]

    # Get Newton joint transforms and joint axes
    parent_bid = model_joint_parent[joint_id]
    p_r_p_com = wp.vec3f(model_body_com[parent_bid]) if parent_bid >= 0 else wp.vec3f(0.0, 0.0, 0.0)
    c_r_c_com = wp.vec3f(model_body_com[model_joint_child[joint_id]])
    T_X_p_j = model_joint_X_p[joint_id]
    T_X_c_j = model_joint_X_c[joint_id]
    q_p_j = wp.transform_get_rotation(T_X_p_j)
    q_c_j = wp.transform_get_rotation(T_X_c_j)
    p_r_p_j = wp.transform_get_translation(T_X_p_j)
    c_r_c_j = wp.transform_get_translation(T_X_c_j)

    # Convert positions by subtracting CoM
    B_r_Bj = p_r_p_j - p_r_p_com
    F_r_Fj = c_r_c_j - c_r_c_com

    # Convert rotations by absorbing the DoF axis basis
    dof_axes_j = mat63f()
    dofs_start_j = model_joint_qd_start[joint_id]
    for i in range(ndofs_j):
        dof_axes_j[i] = model_joint_axis[dofs_start_j + i]
    R_axis_j = JointDoFType.axes_matrix_from_joint_type(dof_type_j, dof_axes_j)
    X_B_j = wp.quat_to_matrix(q_p_j) @ R_axis_j
    X_F_j = wp.quat_to_matrix(q_c_j) @ R_axis_j

    # Write converted joint transforms
    joint_B_r_B[joint_id] = B_r_Bj
    joint_F_r_F[joint_id] = F_r_Fj
    joint_X_B[joint_id] = X_B_j
    joint_X_F[joint_id] = X_F_j


@wp.kernel
def joint_indexing_kernel(
    # Inputs:
    model_joint_world_start: wp.array[wp.int32],
    joint_act_type: wp.array[wp.int32],
    joint_num_coords: wp.array[wp.int32],
    joint_num_dofs: wp.array[wp.int32],
    joint_num_kinematic_cts: wp.array[wp.int32],
    joint_num_dynamic_cts: wp.array[wp.int32],
    joint_num_bounded_cts: wp.array[wp.int32],
    joint_num_friction_cts: wp.array[wp.int32],
    joint_num_effort_cts: wp.array[wp.int32],
    model_fk_act_flag: wp.array[wp.int32],
    # Outputs:
    num_passive_joints: wp.array[wp.int32],
    num_actuated_joints: wp.array[wp.int32],
    num_dynamic_joints: wp.array[wp.int32],
    num_joint_coords: wp.array[wp.int32],
    num_joint_dofs: wp.array[wp.int32],
    num_joint_passive_coords: wp.array[wp.int32],
    num_joint_passive_dofs: wp.array[wp.int32],
    num_joint_actuated_coords: wp.array[wp.int32],
    num_joint_fk_actuated_coords: wp.array[wp.int32],
    num_joint_actuated_dofs: wp.array[wp.int32],
    num_joint_fk_actuated_dofs: wp.array[wp.int32],
    num_joint_bilateral_cts: wp.array[wp.int32],
    num_joint_dynamic_cts: wp.array[wp.int32],
    num_joint_kinematic_cts: wp.array[wp.int32],
    num_joint_bounded_cts: wp.array[wp.int32],
    num_joint_friction_cts: wp.array[wp.int32],
    num_joint_effort_cts: wp.array[wp.int32],
    joint_coord_start: wp.array[wp.int32],
    joint_dofs_start: wp.array[wp.int32],
    joint_actuated_coord_start: wp.array[wp.int32],
    joint_actuated_dofs_start: wp.array[wp.int32],
    joint_passive_coord_start: wp.array[wp.int32],
    joint_passive_dofs_start: wp.array[wp.int32],
    joint_bilateral_cts_start: wp.array[wp.int32],
    joint_dynamic_cts_start: wp.array[wp.int32],
    joint_kinematic_cts_start: wp.array[wp.int32],
    joint_bounded_cts_start: wp.array[wp.int32],
    joint_friction_cts_start: wp.array[wp.int32],
    joint_effort_cts_start: wp.array[wp.int32],
):
    world_id = wp.tid()

    joints_world_start = model_joint_world_start[world_id]
    num_joints_world = model_joint_world_start[world_id + 1] - joints_world_start

    # Initialize sizes for this world
    num_passive_j = int(0)
    num_actuated_j = int(0)
    num_dynamic_j = int(0)
    num_coords = int(0)
    num_dofs = int(0)
    num_actuated_coords = int(0)
    num_fk_actuated_coords = int(0)
    num_actuated_dofs = int(0)
    num_fk_actuated_dofs = int(0)
    num_passive_coords = int(0)
    num_passive_dofs = int(0)
    num_bilateral_cts = int(0)
    num_dynamic_cts = int(0)
    num_kinematic_cts = int(0)
    num_bounded = int(0)
    num_friction = int(0)
    num_effort = int(0)

    for jid in range(num_joints_world):
        joint_id = joints_world_start + jid

        # Updating the start indices within the world
        joint_coord_start[joint_id] = num_coords
        joint_dofs_start[joint_id] = num_dofs
        joint_actuated_coord_start[joint_id] = num_actuated_coords
        joint_actuated_dofs_start[joint_id] = num_actuated_dofs
        joint_passive_coord_start[joint_id] = num_passive_coords
        joint_passive_dofs_start[joint_id] = num_passive_dofs
        joint_bilateral_cts_start[joint_id] = num_bilateral_cts
        joint_dynamic_cts_start[joint_id] = num_dynamic_cts
        joint_kinematic_cts_start[joint_id] = num_kinematic_cts
        joint_bounded_cts_start[joint_id] = num_bounded
        joint_friction_cts_start[joint_id] = num_friction
        joint_effort_cts_start[joint_id] = num_effort

        # Reading off joint properties from previous kernel
        ncoords_j = joint_num_coords[joint_id]
        ndofs_j = joint_num_dofs[joint_id]
        n_kin_cts_j = joint_num_kinematic_cts[joint_id]
        n_dyn_cts_j = joint_num_dynamic_cts[joint_id]
        n_bounded_cts_j = joint_num_bounded_cts[joint_id]
        n_friction_cts_j = joint_num_friction_cts[joint_id]
        n_effort_cts_j = joint_num_effort_cts[joint_id]
        act_type_j = joint_act_type[joint_id]

        # Update world sizes based on joint sizes
        num_coords += ncoords_j
        num_dofs += ndofs_j
        num_bilateral_cts += n_kin_cts_j
        num_kinematic_cts += n_kin_cts_j

        # Update sizes based on passive/active joint distinction
        if act_type_j > JointActuationType.PASSIVE:
            num_actuated_j += 1
            num_actuated_coords += ncoords_j
            num_actuated_dofs += ndofs_j
            if not model_fk_act_flag or model_fk_act_flag[joint_id] == -1:
                num_fk_actuated_coords += ncoords_j
                num_fk_actuated_dofs += ndofs_j
        else:
            num_passive_j += 1
            num_passive_coords += ncoords_j
            num_passive_dofs += ndofs_j
        if model_fk_act_flag and model_fk_act_flag[joint_id] == 1:
            num_fk_actuated_coords += ncoords_j
            num_fk_actuated_dofs += ndofs_j

        # Update sizes based on whether joint is dynamic
        if n_dyn_cts_j > 0:
            num_dynamic_cts += n_dyn_cts_j
            num_bilateral_cts += n_dyn_cts_j
            num_dynamic_j += 1

        num_bounded += n_bounded_cts_j
        num_friction += n_friction_cts_j
        num_effort += n_effort_cts_j

    # Write sizes for this world
    num_passive_joints[world_id] = num_passive_j
    num_actuated_joints[world_id] = num_actuated_j
    num_dynamic_joints[world_id] = num_dynamic_j
    num_joint_coords[world_id] = num_coords
    num_joint_dofs[world_id] = num_dofs
    num_joint_bilateral_cts[world_id] = num_bilateral_cts
    num_joint_kinematic_cts[world_id] = num_kinematic_cts
    num_joint_dynamic_cts[world_id] = num_dynamic_cts
    num_joint_bounded_cts[world_id] = num_bounded
    num_joint_friction_cts[world_id] = num_friction
    num_joint_effort_cts[world_id] = num_effort
    num_joint_actuated_coords[world_id] = num_actuated_coords
    num_joint_fk_actuated_coords[world_id] = num_fk_actuated_coords
    num_joint_actuated_dofs[world_id] = num_actuated_dofs
    num_joint_fk_actuated_dofs[world_id] = num_fk_actuated_dofs
    num_joint_passive_coords[world_id] = num_passive_coords
    num_joint_passive_dofs[world_id] = num_passive_dofs


@wp.kernel
def _globalize_joint_offsets(
    # Inputs:
    joint_world: wp.array[wp.int32],
    world_coord_offset: wp.array[wp.int32],
    world_dof_offset: wp.array[wp.int32],
    world_passive_coord_offset: wp.array[wp.int32],
    world_passive_dof_offset: wp.array[wp.int32],
    world_actuated_coord_offset: wp.array[wp.int32],
    world_actuated_dof_offset: wp.array[wp.int32],
    world_bilateral_cts_offset: wp.array[wp.int32],
    world_dynamic_cts_offset: wp.array[wp.int32],
    world_kinematic_cts_offset: wp.array[wp.int32],
    world_bounded_cts_offset: wp.array[wp.int32],
    world_friction_cts_offset: wp.array[wp.int32],
    world_effort_cts_offset: wp.array[wp.int32],
    # Outputs:
    joint_coord_start: wp.array[wp.int32],
    joint_dofs_start: wp.array[wp.int32],
    joint_passive_coord_start: wp.array[wp.int32],
    joint_passive_dofs_start: wp.array[wp.int32],
    joint_actuated_coord_start: wp.array[wp.int32],
    joint_actuated_dofs_start: wp.array[wp.int32],
    joint_bilateral_cts_start: wp.array[wp.int32],
    joint_dynamic_cts_start: wp.array[wp.int32],
    joint_kinematic_cts_start: wp.array[wp.int32],
    joint_bounded_cts_start: wp.array[wp.int32],
    joint_friction_cts_start: wp.array[wp.int32],
    joint_effort_cts_start: wp.array[wp.int32],
):
    jid = wp.tid()
    w = joint_world[jid]
    joint_coord_start[jid] += world_coord_offset[w]
    joint_dofs_start[jid] += world_dof_offset[w]
    joint_passive_coord_start[jid] += world_passive_coord_offset[w]
    joint_passive_dofs_start[jid] += world_passive_dof_offset[w]
    joint_actuated_coord_start[jid] += world_actuated_coord_offset[w]
    joint_actuated_dofs_start[jid] += world_actuated_dof_offset[w]
    joint_bilateral_cts_start[jid] += world_bilateral_cts_offset[w]
    joint_dynamic_cts_start[jid] += world_dynamic_cts_offset[w]
    joint_kinematic_cts_start[jid] += world_kinematic_cts_offset[w]
    joint_bounded_cts_start[jid] += world_bounded_cts_offset[w]
    joint_friction_cts_start[jid] += world_friction_cts_offset[w]
    joint_effort_cts_start[jid] += world_effort_cts_offset[w]


@wp.kernel
def pack_joint_constraint_axes_kernel(
    # Inputs:
    model_joint_qd_start: wp.array[wp.int32],
    model_joint_target_mode: wp.array[wp.int32],
    model_joint_target_ke: wp.array[wp.float32],
    model_joint_target_kd: wp.array[wp.float32],
    model_joint_effort_limit: wp.array[wp.float32],
    model_joint_armature: wp.array[wp.float32],
    model_joint_damping: wp.array[wp.float32],
    model_joint_friction: wp.array[wp.float32],
    joint_dof_type: wp.array[wp.int32],
    joint_dynamic_cts_start: wp.array[wp.int32],
    joint_friction_cts_start: wp.array[wp.int32],
    joint_effort_cts_start: wp.array[wp.int32],
    # Outputs:
    dynamic_cts_axis: wp.array[wp.int32],
    friction_cts_axis: wp.array[wp.int32],
    effort_cts_axis: wp.array[wp.int32],
):
    """Pack ascending joint-local DoF axes for compact constraint rows."""
    joint = wp.tid()
    dof_start = model_joint_qd_start[joint]
    dof_end = model_joint_qd_start[joint + 1]
    dynamic_row = joint_dynamic_cts_start[joint]
    friction_row = joint_friction_cts_start[joint]
    effort_row = joint_effort_cts_start[joint]
    dof_type = joint_dof_type[joint]
    for axis in range(dof_end - dof_start):
        dof = dof_start + axis
        act_type = JointActuationType.from_newton_wp(model_joint_target_mode[dof])
        dynamic = has_dynamic_cts_wp(
            act_type,
            model_joint_target_ke[dof],
            model_joint_target_kd[dof],
            model_joint_effort_limit[dof],
            model_joint_armature[dof],
            model_joint_damping[dof],
        )
        friction = has_friction_cts_wp(dof_type, model_joint_friction[dof])
        effort = has_effort_cts_wp(
            act_type,
            model_joint_target_ke[dof],
            model_joint_target_kd[dof],
            model_joint_effort_limit[dof],
        )
        if dynamic:
            dynamic_cts_axis[dynamic_row] = axis
            dynamic_row += 1
        if friction:
            friction_cts_axis[friction_row] = axis
            friction_row += 1
        if effort:
            effort_cts_axis[effort_row] = axis
            effort_row += 1


@wp.kernel
def geometry_conversion_kernel(
    # Inputs:
    model_shape_world: wp.array[wp.int32],
    model_shape_world_start: wp.array[wp.int32],
    model_shape_flags: wp.array[wp.int32],
    model_shape_collision_groups: wp.array[wp.int32],
    geom_material: wp.array[wp.int32],
    # Outputs:
    geom_gid: wp.array[wp.int32],
    model_num_collidable_geoms: wp.array[wp.int32],
):
    # Retrieve the geom/shape index from the thread grid
    shape_id = wp.tid()

    # Determine the world for this shape and compute in-world geom index
    world_id = model_shape_world[shape_id]
    if world_id >= 0:
        geom_gid[shape_id] = shape_id - model_shape_world_start[world_id]
    else:
        # Handle global shapes that don't belong to any world (world_id=-1)
        if shape_id < model_shape_world_start[0]:
            # Global shapes at the head are indexed as-is before all world shapes
            geom_gid[shape_id] = shape_id
        else:
            # Global shapes at the tail are indexed after all world shapes
            geom_gid[shape_id] = shape_id - model_shape_world_start[-2]

    # Determine if this shape is collidable and update collidable geom count
    # for the world. If not collidable, also ensure no material is assigned.
    shape_flags = model_shape_flags[shape_id]
    if (shape_flags & ShapeFlags.COLLIDE_SHAPES) != 0 and model_shape_collision_groups[shape_id] > 0:
        wp.atomic_add(model_num_collidable_geoms, 0, 1)
    else:
        geom_material[shape_id] = -1


@wp.kernel
def target_dofs_to_coords_conversion_kernel(
    # Inputs
    model_joints_dof_type: wp.array[wp.int32],
    model_joints_dofs_offset: wp.array[wp.int32],
    model_joints_coords_offset: wp.array[wp.int32],
    joint_target_dofs: wp.array[wp.float32],
    # Outputs
    joint_target_coords: wp.array[wp.float32],
):
    # Read thread id (= joint id)
    jid = wp.tid()

    # Get dof/coords offsets and number of dofs
    dof_offset = model_joints_dofs_offset[jid]
    num_dofs = model_joints_dofs_offset[jid + 1] - dof_offset
    coord_offset = model_joints_coords_offset[jid]

    # Check whether coords = dofs for this joint
    dof_type = model_joints_dof_type[jid]
    orientation_dofs_offset = -1  # Offset of orientation dofs to convert
    if dof_type == JointDoFType.FREE or dof_type == JointDoFType.SPHERICAL:
        # Spherical/free joint: the last 3 dofs / 4 coords differ (Euler angles vs unit quaternion)
        orientation_dofs_offset = num_dofs - 3
        num_dofs -= 3

    # Copy all dofs/coords that match
    for k in range(num_dofs):
        joint_target_coords[coord_offset + k] = joint_target_dofs[dof_offset + k]

    # Convert Euler angles to unit quaternion if needed
    if orientation_dofs_offset >= 0:
        angles_offset = dof_offset + orientation_dofs_offset
        angles = wp.vec3f(
            joint_target_dofs[angles_offset],
            joint_target_dofs[angles_offset + 1],
            joint_target_dofs[angles_offset + 2],
        )
        quat = wp.quat_from_euler(angles, 2, 1, 0)
        quat_offset = coord_offset + orientation_dofs_offset
        for k in range(4):
            joint_target_coords[quat_offset + k] = quat[k]


@wp.kernel
def target_coords_to_dofs_conversion_kernel(
    # Inputs
    model_joints_dof_type: wp.array[wp.int32],
    model_joints_dofs_offset: wp.array[wp.int32],
    model_joints_coords_offset: wp.array[wp.int32],
    joint_target_coords: wp.array[wp.float32],
    # Outputs
    joint_target_dofs: wp.array[wp.float32],
):
    # Read thread id (= joint id)
    jid = wp.tid()

    # Get dof/coords offsets and number of dofs
    dof_offset = model_joints_dofs_offset[jid]
    num_dofs = model_joints_dofs_offset[jid + 1] - dof_offset
    coord_offset = model_joints_coords_offset[jid]

    # Check whether coords = dofs for this joint
    dof_type = model_joints_dof_type[jid]
    orientation_dofs_offset = -1  # Offset of orientation dofs to convert
    if dof_type == JointDoFType.FREE or dof_type == JointDoFType.SPHERICAL:
        # Spherical/free joint: the last 3 dofs / 4 coords differ (Euler angles vs unit quaternion)
        orientation_dofs_offset = num_dofs - 3
        num_dofs -= 3

    # Copy all dofs/coords that match
    for k in range(num_dofs):
        joint_target_dofs[dof_offset + k] = joint_target_coords[coord_offset + k]

    # Convert unit quaternion to Euler angles if needed
    if orientation_dofs_offset >= 0:
        quat_offset = coord_offset + orientation_dofs_offset
        quat = wp.quat(
            joint_target_coords[quat_offset],
            joint_target_coords[quat_offset + 1],
            joint_target_coords[quat_offset + 2],
            joint_target_coords[quat_offset + 3],
        )
        angles = wp.quat_to_euler(quat, 2, 1, 0)
        angles_offset = dof_offset + orientation_dofs_offset
        for k in range(3):
            joint_target_dofs[angles_offset + k] = angles[k]


@wp.kernel
def write_coeff_kernel(a: wp.array[wp.int32], idx: int, v: int):
    """Helper kernel writing a single array coefficient"""
    a[idx] = v


###
# Functions
###


def compute_required_contact_capacity(
    model: Model,
    max_contacts_per_pair: int | None = None,
    max_contacts_per_world: int | None = None,
) -> tuple[int, list[int]]:
    """
    Computes the required contact capacity for a given Newton model.

    The outputs are used to determine the minimum number of contacts
    to be allocated, according to the shapes present in the model.

    Args:
        model: The Newton model for which to compute the required contact capacity.
        max_contacts_per_pair: Optional maximum number of contacts to allocate per shape pair.
            If `None`, no per-pair limit is applied.
        max_contacts_per_world: Optional maximum number of contacts to allocate per world.
            If `None`, no per-world limit is applied, otherwise caps the computed
            per-world requirements at this value.

    Returns:
        (model_required_contacts, world_required_contacts):
            A tuple containing:
            - `model_required_contacts` (int):
                The total number of contacts required for the entire model.
            - `world_required_contacts` (list[int]):
                A list of required contacts per world, where the length of the
                list is equal to `model.world_count` and each entry corresponds
                to the required contacts for that world.

    """
    # First check if there are any collision geometries
    if model.shape_count == 0:
        return 0, [0] * model.world_count

    # Compute maximum contacts per world
    world_max_contacts_wp = wp.zeros((model.world_count,), dtype=wp.int32, device=model.device)
    wp.launch(
        kernel=world_max_contacts_kernel,
        dim=model.shape_contact_pair_count,
        inputs=[
            max_contacts_per_pair if max_contacts_per_pair is not None else -1,
            model.shape_type,
            model.shape_world,
            model.shape_contact_pairs,
        ],
        outputs=[world_max_contacts_wp],
        device=model.device,
    )
    world_max_contacts = world_max_contacts_wp.numpy()

    # Cap per-world totals when a per-world maximum is specified
    if max_contacts_per_world is not None:
        world_max_contacts = np.minimum(world_max_contacts, max_contacts_per_world)

    # Return the per-world maximum contacts list
    return int(np.sum(world_max_contacts)), world_max_contacts.astype(int).tolist()


def validate_model_structural_updates(
    model: Model,
    joints: JointsModel,
    built_limit_finite: wp.array[wp.int32],
    built_massless: wp.array[wp.int32],
    violations: wp.array[wp.int32],
    *,
    check_dof: bool,
    check_actuation: bool,
    check_axes: bool,
    check_inertial: bool,
) -> int:
    """Validate that runtime edits preserve Kamino's structural layout.

    ``violations`` is a ``len(StructuralUpdateViolation)``-entry array
    containing the first index for each violation type:

    - :attr:`StructuralUpdateViolation.DYNAMIC_CTS`: dynamic-constraint topology changed
    - :attr:`StructuralUpdateViolation.LIMIT_FINITE`: finite-limit state changed
    - :attr:`StructuralUpdateViolation.ACTUATION_PARTITION`: passive/actuated partition changed
    - :attr:`StructuralUpdateViolation.INVALID_TARGET_MODE`: unsupported target-mode combination
    - :attr:`StructuralUpdateViolation.NONORTHONORMAL_AXES`: nonorthonormal universal/gimbal axes
    - :attr:`StructuralUpdateViolation.GIMBAL_HANDEDNESS`: gimbal axis handedness changed
    - :attr:`StructuralUpdateViolation.MASSLESS`: a built massive body became massless
    - :attr:`StructuralUpdateViolation.FRICTION_CTS`: joint friction constraint topology changed
    - :attr:`StructuralUpdateViolation.EFFORT_CTS`: effort-row topology changed

    An entry equal to the maximum of the body, joint, and DoF counts indicates that no
    violation of that type was found.

    Args:
        model: The Newton model containing the updated properties to validate.
        joints: The current Kamino joint model, before applying the updates.
        built_limit_finite: The built finite limit state for each DoF.
        built_massless: Whether each body was massless at solver construction.
        violations: The array to store the violations.
        check_dof: Whether to check the DoF updates.
        check_actuation: Whether to check the actuation updates.
        check_axes: Whether to check universal and gimbal axes.
        check_inertial: Whether to check body inertial updates.

    Returns:
        The sentinel value indicating no violations.
    """
    dim = max(model.body_count, model.joint_count, model.joint_dof_count)
    violations.fill_(dim)
    if (check_dof or check_actuation) and dim > 0:
        wp.launch(
            kernel=validate_joint_dof_updates_kernel,
            dim=dim,
            inputs=[
                # Inputs:
                model.joint_qd_start,
                model.joint_armature,
                model.joint_damping,
                model.joint_target_ke,
                model.joint_target_kd,
                model.joint_target_mode,
                model.joint_effort_limit,
                model.joint_friction,
                joints.dof_type,
                joints.dynamic_cts_offset,
                joints.dynamic_cts_axis,
                joints.friction_cts_offset,
                joints.friction_cts_axis,
                joints.effort_cts_offset,
                joints.effort_cts_axis,
                model.joint_limit_lower,
                model.joint_limit_upper,
                built_limit_finite,
                model.joint_count,
                model.joint_dof_count,
                # Outputs:
                violations,
            ],
            device=model.device,
        )
    if check_actuation and model.joint_count > 0:
        wp.launch(
            kernel=validate_joint_actuation_updates_kernel,
            dim=model.joint_count,
            inputs=[
                # Inputs:
                model.joint_qd_start,
                model.joint_target_mode,
                joints.act_type,
                # Outputs:
                violations,
            ],
            device=model.device,
        )
    if check_axes and model.joint_count > 0:
        wp.launch(
            kernel=validate_joint_axes_kernel,
            dim=model.joint_count,
            inputs=[
                # Inputs:
                model.joint_qd_start,
                model.joint_axis,
                joints.dof_type,
                # Outputs:
                violations,
            ],
            device=model.device,
        )
    if check_inertial and model.body_count > 0:
        wp.launch(
            kernel=validate_body_inertial_updates_kernel,
            dim=model.body_count,
            inputs=[
                # Inputs:
                model.body_inv_mass,
                model.body_inv_inertia,
                built_massless,
                # Outputs:
                violations,
            ],
            device=model.device,
        )

    return dim


def convert_model_joint_actuation(model: Model, joints: JointsModel) -> None:
    """Update Kamino's joint and DoF actuation types from Newton target modes."""
    if model.joint_count == 0:
        return
    if model.joint_dof_count > 0:
        wp.launch(
            kernel=update_joint_dof_actuation_kernel,
            dim=model.joint_dof_count,
            inputs=[
                # Inputs:
                model.joint_target_mode,
                # Outputs:
                joints.dof_act_types,
            ],
            device=model.device,
        )
    wp.launch(
        kernel=update_joint_actuation_kernel,
        dim=model.joint_count,
        inputs=[
            # Inputs:
            model.joint_qd_start,
            joints.dof_act_types,
            # Outputs:
            joints.act_type,
        ],
        device=model.device,
    )


def _validate_joint_axes(
    model: Model,
    joint_dof_type: wp.array[wp.int32],
    violations: wp.array[wp.int32],
) -> None:
    """Validate universal and gimbal axes before Warp frame conversion."""
    violations.fill_(model.joint_count)
    if model.joint_count > 0:
        wp.launch(
            kernel=validate_joint_axes_kernel,
            dim=model.joint_count,
            inputs=[
                # Inputs:
                model.joint_qd_start,
                model.joint_axis,
                joint_dof_type,
                # Outputs:
                violations,
            ],
            device=model.device,
        )
    violations_np = violations.numpy()
    invalid_joint = int(violations_np[StructuralUpdateViolation.NONORTHONORMAL_AXES])
    if invalid_joint < model.joint_count:
        raise ValueError(
            f"Invalid joint configuration for SolverKamino:\n"
            f"  - joint {invalid_joint} ({model.joint_label[invalid_joint]!r}): "
            "universal and gimbal axes must be unit length and orthogonal"
        )
    invalid_joint = int(violations_np[StructuralUpdateViolation.GIMBAL_HANDEDNESS])
    if invalid_joint < model.joint_count:
        raise ValueError(
            f"Invalid joint configuration for SolverKamino:\n"
            f"  - joint {invalid_joint} ({model.joint_label[invalid_joint]!r}): "
            "gimbal axes must preserve the solver's original handedness"
        )


def convert_model_joint_transforms(model: Model, joints: JointsModel) -> None:
    """
    Converts the joint model parameterization of Newton's to Kamino's format.

    Computes :attr:`JointsModel.B_r_Bj`, :attr:`JointsModel.F_r_Fj`, :attr:`JointsModel.X_Bj`
    and :attr:`JointsModel.X_Fj` from Newton's ``model.joint_X_p`` / ``model.joint_X_c``
    transforms and writes them in-place into ``joints``.

    Args:
    - model:
        The input Newton model containing the joint information to be converted.
    - joints:
        The output JointsModel instance where the converted joint data will be stored.
        This function modifies the `joints` object in-place.
    """
    wp.launch(
        kernel=joint_frame_conversion_kernel,
        dim=model.joint_count,
        inputs=[
            # Inputs:
            model.joint_parent,
            model.joint_child,
            model.joint_qd_start,
            model.joint_axis,
            model.body_com,
            model.joint_X_p,
            model.joint_X_c,
            joints.dof_type,
            joints.num_dofs,
            # Outputs:
            joints.B_r_Bj,
            joints.F_r_Fj,
            joints.X_Bj,
            joints.X_Fj,
        ],
        device=model.device,
    )


def compute_material_first_shape(
    geom_material: wp.array[wp.int32],
    num_materials: int,
) -> wp.array[wp.int32]:
    """Compute the first shape associated with each fixed material ID.

    Args:
        geom_material: Material ID for each shape.
        num_materials: Number of registered materials.

    Returns:
        Per-material shape indices. Materials without an associated shape use
        the shape count as a sentinel.
    """
    shape_count = geom_material.shape[0]
    first_shape = wp.full(num_materials, shape_count, dtype=wp.int32, device=geom_material.device)
    if shape_count > 0:
        wp.launch(
            kernel=material_first_shape_kernel,
            dim=shape_count,
            inputs=[
                # Inputs:
                geom_material,
                # Outputs:
                first_shape,
            ],
            device=geom_material.device,
        )
    return first_shape


def convert_model_materials(
    model: Model,
    model_kamino: ModelKamino,
    first_shape: wp.array[wp.int32],
    conflict: wp.array[wp.int32],
) -> None:
    """Update Kamino's material properties in place from Newton shape materials.

    Recomputes per-material friction and restitution from
    ``model.shape_material_mu`` and ``model.shape_material_restitution`` while
    preserving the material arrays referenced by Kamino's collision detector.

    Args:
        model: Newton model containing the updated shape materials.
        model_kamino: Kamino model whose material tables are updated.
        first_shape: Cached first shape associated with each fixed material ID.
        conflict: Scratch scalar for reporting conflicting material updates.

    Raises:
        RuntimeError: If shapes assigned to the same material ID have different
            material properties and would require splitting that material.
    """
    materials = model_kamino.materials
    conflict.fill_(materials.num_materials)

    # Check each shape against the cached representative for its material.
    wp.launch(
        kernel=validate_material_update_kernel,
        dim=model.shape_count,
        inputs=[
            # Inputs:
            model.shape_material_mu,
            model.shape_material_restitution,
            model_kamino.geoms.material,
            first_shape,
            # Outputs:
            conflict,
        ],
        device=model.device,
    )

    conflict_material = int(conflict.numpy()[0])
    if conflict_material < materials.num_materials:
        raise RuntimeError(
            f"Multiple shapes assigned to contact material {conflict_material} attempted to update it with "
            "different friction or restitution values; recreate SolverKamino to split the material."
        )

    # Once conflicts have been ruled out, update the material properties in place.
    wp.launch(
        kernel=update_materials_kernel,
        dim=materials.num_materials,
        inputs=[
            # Inputs:
            model.shape_material_mu,
            model.shape_material_restitution,
            first_shape,
            model.shape_count,
            # Outputs:
            materials.restitution,
            materials.static_friction,
            materials.dynamic_friction,
            model_kamino.material_pairs.restitution,
            model_kamino.material_pairs.static_friction,
            model_kamino.material_pairs.dynamic_friction,
        ],
        device=model.device,
    )


def convert_rigid_bodies(
    model: Model | ModelView,
    model_size: SizeKamino,
    model_info: ModelKaminoInfo,
) -> RigidBodiesModel:
    """
    Converts the rigid bodies from a Newton model into Kamino's format. The function
    will create a new `RigidBodiesModel` object and fill in the rigid body and shape
    entries of the provided `SizeKamino` and `ModelKaminoInfo` objects. The input model
    is treated as read-only (data is neither modified nor aliased).

    Args:
        model: Newton model.
        model_size: Model size object, to be filled in by the function.
        model_info: Model info object, to be filled in by the function.

    Returns:
        Fully converted rigid bodies model in Kamino's format.
    """

    # Compute the offsets and number of entities per world
    with wp.ScopedDevice(model.device):
        body_bid = wp.zeros((model.body_count,), dtype=wp.int32)
        num_bodies = wp.zeros((model.world_count,), dtype=wp.int32)
        num_shapes = wp.zeros((model.world_count,), dtype=wp.int32)
        num_body_dofs = wp.zeros((model.world_count,), dtype=wp.int32)
        world_body_offset = wp.zeros((model.world_count + 1,), dtype=wp.int32)
        world_shape_offset = wp.zeros((model.world_count,), dtype=wp.int32)
        world_body_dof_offset = wp.zeros((model.world_count,), dtype=wp.int32)
    wp.launch(
        kernel=rigid_bodies_indexing_kernel,
        dim=model.world_count,
        inputs=[
            model.body_world_start,
            model.shape_world_start,
        ],
        outputs=[
            body_bid,
            num_bodies,
            num_shapes,
            num_body_dofs,
            world_body_offset,
            world_shape_offset,
            world_body_dof_offset,
        ],
        device=model.device,
    )

    # model.body_q stores body-origin world poses, but Kamino expects
    # COM world poses (joint attachment vectors are COM-relative).
    q_i_0 = wp.empty((model.body_count,), dtype=wp.transformf, device=model.device)
    convert_body_origin_to_com(model.body_com, model.body_q, q_i_0)

    # Fill in size data for bodies
    model_size.sum_of_num_bodies = model.body_count
    model_size.max_of_num_bodies = int(num_bodies.numpy().max())
    model_size.sum_of_num_geoms = model.shape_count
    model_size.max_of_num_geoms = int(num_shapes.numpy().max())
    model_size.sum_of_num_body_dofs = 6 * model.body_count
    model_size.max_of_num_body_dofs = int(num_body_dofs.numpy().max())

    # Write the N+1 entry (grand total) into the bodies offset array.
    wp.launch(
        write_coeff_kernel,
        dim=1,
        inputs=[world_body_offset, model_size.num_worlds, model_size.sum_of_num_bodies],
        device=model.device,
    )

    # Per-world heterogeneous model info
    model_info.num_bodies = num_bodies
    model_info.num_geoms = num_shapes
    model_info.num_body_dofs = num_body_dofs
    model_info.bodies_offset = world_body_offset
    model_info.geoms_offset = world_shape_offset
    model_info.body_dofs_offset = world_body_dof_offset

    model_bodies = RigidBodiesModel(
        num_bodies=model.body_count,
        label=model.body_label,
        wid=model.body_world,
        bid=body_bid,  # TODO: Remove
        m_i=model.body_mass,
        inv_m_i=model.body_inv_mass,
        i_r_com_i=model.body_com,
        i_I_i=model.body_inertia,
        inv_i_I_i=model.body_inv_inertia,
        q_i_0=q_i_0,
        u_i_0=model.body_qd,
    )
    return model_bodies


def _warn_ignored_free_joint_friction(
    model: Model | ModelView,
    joint_dof_type: wp.array[wp.int32],
) -> None:
    """Warn when friction is assigned to a FREE joint (ignored by Kamino)."""
    if model.joint_count == 0:
        return
    dof_type_np = joint_dof_type.numpy()
    friction_np = model.joint_friction.numpy()
    qd_start_np = model.joint_qd_start.numpy()
    for jid in range(model.joint_count):
        if dof_type_np[jid] != JointDoFType.FREE:
            continue
        dof_start = int(qd_start_np[jid])
        dof_end = int(qd_start_np[jid + 1])
        if np.any(friction_np[dof_start:dof_end] > 0.0):
            msg.warning("Ignoring joint friction on FREE joint %d (%r).", jid, model.joint_label[jid])


def _validate_model_joint_pd_gains(model: Model | ModelView) -> None:
    """Raises if a Newton joint's selected implicit-PD mode has no effective gain."""
    target_mode = model.joint_target_mode.numpy()
    k_p = model.joint_target_ke.numpy()
    k_d = model.joint_target_kd.numpy()
    for dof in range(model.joint_dof_count):
        act_type = JointActuationType.from_newton(target_mode[dof])
        _validate_implicit_pd_gains(act_type, k_p[dof], k_d[dof], label=f"DoF={dof}")


def convert_joints(
    model: Model | ModelView,
    model_size: SizeKamino,
    model_info: ModelKaminoInfo,
) -> JointsModel:
    """
    Converts the joints from a Newton model into Kamino's format. The function will
    create a new `JointsModel` object and fill in the joint entries of the provided
    `SizeKamino` and `ModelKaminoInfo` objects. The input model is treated as read-only
    (data is neither modified nor aliased).

    Args:
        model: Newton model.
        model_size: Model size object, to be filled in by the function.
        model_info: Model info object, to be filled in by the function.

    Returns:
        Fully converted joints model in Kamino's format.
    """
    _validate_model_joint_pd_gains(model)

    # Compute the number of joints per world
    joint_world_start_np = model.joint_world_start.numpy()
    num_joints_np = joint_world_start_np[1 : model.world_count + 1] - joint_world_start_np[: model.world_count]

    # Create joint property arrays
    with wp.ScopedDevice(model.device):
        joint_jid = wp.empty(shape=(model.joint_count,), dtype=wp.int32)
        joint_dof_type = wp.zeros(shape=(model.joint_count,), dtype=wp.int32)
        joint_act_type = wp.zeros(shape=(model.joint_count,), dtype=wp.int32)
        joint_dof_act_types = wp.zeros(shape=(model.joint_dof_count,), dtype=wp.int32)
        joint_dof_act_paths = wp.zeros(shape=(model.joint_dof_count,), dtype=wp.int32)
        joint_num_coords = wp.zeros(shape=(model.joint_count,), dtype=wp.int32)
        joint_num_dofs = wp.zeros(shape=(model.joint_count,), dtype=wp.int32)
        joint_num_bilateral_cts = wp.zeros(shape=(model.joint_count,), dtype=wp.int32)
        joint_num_dynamic_cts = wp.zeros(shape=(model.joint_count,), dtype=wp.int32)
        joint_num_kinematic_cts = wp.zeros(shape=(model.joint_count,), dtype=wp.int32)
        joint_num_bounded_cts = wp.zeros(shape=(model.joint_count,), dtype=wp.int32)
        joint_num_friction_cts = wp.zeros(shape=(model.joint_count,), dtype=wp.int32)
        joint_num_effort_cts = wp.zeros(shape=(model.joint_count,), dtype=wp.int32)
        joint_B_r_B = wp.empty(shape=(model.joint_count,), dtype=wp.vec3f)
        joint_F_r_F = wp.empty(shape=(model.joint_count,), dtype=wp.vec3f)
        joint_X_B = wp.empty(shape=(model.joint_count,), dtype=wp.mat33f)
        joint_X_F = wp.empty(shape=(model.joint_count,), dtype=wp.mat33f)

    # First classify each DoF and count its joint dynamics, friction, and
    # effort-limit constraints. The indexing pass then needs those counts to
    # prefix-sum the world-local offsets; they cannot be inferred from joint
    # type alone because they depend on per-DoF actuation and parameters.
    wp.launch(
        kernel=joint_conversion_kernel,
        dim=model.joint_count,
        inputs=[
            # Inputs:
            model.joint_world,
            model.joint_world_start,
            model.joint_type,
            model.joint_dof_dim,
            model.joint_q_start,
            model.joint_qd_start,
            model.joint_axis,
            model.joint_target_mode,
            model.joint_target_ke,
            model.joint_target_kd,
            model.joint_effort_limit,
            model.joint_armature,
            model.joint_damping,
            model.joint_friction,
            model.joint_limit_lower,
            model.joint_limit_upper,
            # Outputs:
            joint_jid,
            joint_dof_type,
            joint_act_type,
            joint_dof_act_types,
            joint_dof_act_paths,
            joint_num_coords,
            joint_num_dofs,
            joint_num_bilateral_cts,
            joint_num_dynamic_cts,
            joint_num_kinematic_cts,
            joint_num_bounded_cts,
            joint_num_friction_cts,
            joint_num_effort_cts,
        ],
        device=model.device,
    )

    _warn_ignored_free_joint_friction(model, joint_dof_type)

    axis_validation_violations = wp.empty(len(StructuralUpdateViolation), dtype=wp.int32, device=model.device)
    _validate_joint_axes(model, joint_dof_type, axis_validation_violations)

    wp.launch(
        kernel=joint_frame_conversion_kernel,
        dim=model.joint_count,
        inputs=[
            # Inputs:
            model.joint_parent,
            model.joint_child,
            model.joint_qd_start,
            model.joint_axis,
            model.body_com,
            model.joint_X_p,
            model.joint_X_c,
            joint_dof_type,
            joint_num_dofs,
            # Outputs:
            joint_B_r_B,
            joint_F_r_F,
            joint_X_B,
            joint_X_F,
        ],
        device=model.device,
    )

    # Compute sizes and indices for all joint properties
    with wp.ScopedDevice(model.device):
        num_passive_joints = wp.zeros(shape=(model.world_count,), dtype=wp.int32)
        num_actuated_joints = wp.zeros(shape=(model.world_count,), dtype=wp.int32)
        num_dynamic_joints = wp.zeros(shape=(model.world_count,), dtype=wp.int32)
        num_joint_coords = wp.zeros(shape=(model.world_count,), dtype=wp.int32)
        num_joint_dofs = wp.zeros(shape=(model.world_count,), dtype=wp.int32)
        num_joint_passive_coords = wp.zeros(shape=(model.world_count,), dtype=wp.int32)
        num_joint_passive_dofs = wp.zeros(shape=(model.world_count,), dtype=wp.int32)
        num_joint_actuated_coords = wp.zeros(shape=(model.world_count,), dtype=wp.int32)
        num_joint_fk_actuated_coords = wp.zeros(shape=(model.world_count,), dtype=wp.int32)
        num_joint_actuated_dofs = wp.zeros(shape=(model.world_count,), dtype=wp.int32)
        num_joint_fk_actuated_dofs = wp.zeros(shape=(model.world_count,), dtype=wp.int32)
        num_joint_bilateral_cts = wp.zeros(shape=(model.world_count,), dtype=wp.int32)
        num_joint_dynamic_cts = wp.zeros(shape=(model.world_count,), dtype=wp.int32)
        num_joint_kinematic_cts = wp.zeros(shape=(model.world_count,), dtype=wp.int32)
        num_joint_bounded_cts = wp.zeros(shape=(model.world_count,), dtype=wp.int32)
        num_joint_friction_cts = wp.zeros(shape=(model.world_count,), dtype=wp.int32)
        num_joint_effort_cts = wp.zeros(shape=(model.world_count,), dtype=wp.int32)
        joint_coord_start = wp.zeros(shape=(model.joint_count + 1,), dtype=wp.int32)
        joint_dofs_start = wp.zeros(shape=(model.joint_count + 1,), dtype=wp.int32)
        joint_actuated_coord_start = wp.zeros(shape=(model.joint_count + 1,), dtype=wp.int32)
        joint_actuated_dofs_start = wp.zeros(shape=(model.joint_count + 1,), dtype=wp.int32)
        joint_passive_coord_start = wp.zeros(shape=(model.joint_count + 1,), dtype=wp.int32)
        joint_passive_dofs_start = wp.zeros(shape=(model.joint_count + 1,), dtype=wp.int32)
        joint_bilateral_cts_start = wp.zeros(shape=(model.joint_count + 1,), dtype=wp.int32)
        joint_dynamic_cts_start = wp.zeros(shape=(model.joint_count + 1,), dtype=wp.int32)
        joint_kinematic_cts_start = wp.zeros(shape=(model.joint_count + 1,), dtype=wp.int32)
        joint_bounded_cts_start = wp.zeros(shape=(model.joint_count + 1,), dtype=wp.int32)
        joint_friction_cts_start = wp.zeros(shape=(model.joint_count + 1,), dtype=wp.int32)
        joint_effort_cts_start = wp.zeros(shape=(model.joint_count + 1,), dtype=wp.int32)

    wp.launch(
        kernel=joint_indexing_kernel,
        dim=model.world_count,
        inputs=[
            model.joint_world_start,
            joint_act_type,
            joint_num_coords,
            joint_num_dofs,
            joint_num_kinematic_cts,
            joint_num_dynamic_cts,
            joint_num_bounded_cts,
            joint_num_friction_cts,
            joint_num_effort_cts,
            model.fk_actuation_flag if hasattr(model, "fk_actuation_flag") else None,
        ],
        outputs=[
            num_passive_joints,
            num_actuated_joints,
            num_dynamic_joints,
            num_joint_coords,
            num_joint_dofs,
            num_joint_passive_coords,
            num_joint_passive_dofs,
            num_joint_actuated_coords,
            num_joint_fk_actuated_coords,
            num_joint_actuated_dofs,
            num_joint_fk_actuated_dofs,
            num_joint_bilateral_cts,
            num_joint_dynamic_cts,
            num_joint_kinematic_cts,
            num_joint_bounded_cts,
            num_joint_friction_cts,
            num_joint_effort_cts,
            joint_coord_start,
            joint_dofs_start,
            joint_actuated_coord_start,
            joint_actuated_dofs_start,
            joint_passive_coord_start,
            joint_passive_dofs_start,
            joint_bilateral_cts_start,
            joint_dynamic_cts_start,
            joint_kinematic_cts_start,
            joint_bounded_cts_start,
            joint_friction_cts_start,
            joint_effort_cts_start,
        ],
        device=model.device,
    )

    # Get on-device copies of the per-world sizes
    num_passive_joints_np = num_passive_joints.numpy()
    num_actuated_joints_np = num_actuated_joints.numpy()
    num_dynamic_joints_np = num_dynamic_joints.numpy()
    num_joint_coords_np = num_joint_coords.numpy()
    num_joint_dofs_np = num_joint_dofs.numpy()
    num_joint_passive_coords_np = num_joint_passive_coords.numpy()
    num_joint_passive_dofs_np = num_joint_passive_dofs.numpy()
    num_joint_actuated_coords_np = num_joint_actuated_coords.numpy()
    num_joint_fk_actuated_coords_np = num_joint_fk_actuated_coords.numpy()
    num_joint_actuated_dofs_np = num_joint_actuated_dofs.numpy()
    num_joint_fk_actuated_dofs_np = num_joint_fk_actuated_dofs.numpy()
    num_joint_bilateral_cts_np = num_joint_bilateral_cts.numpy()
    num_joint_dynamic_cts_np = num_joint_dynamic_cts.numpy()
    num_joint_kinematic_cts_np = num_joint_kinematic_cts.numpy()
    num_joint_bounded_cts_np = num_joint_bounded_cts.numpy()
    num_joint_friction_cts_np = num_joint_friction_cts.numpy()
    num_joint_effort_cts_np = num_joint_effort_cts.numpy()

    # Compute offsets per world
    world_joint_offset_np = np.zeros((model.world_count,), dtype=int)
    world_joint_coord_offset_np = np.zeros((model.world_count,), dtype=int)
    world_joint_dof_offset_np = np.zeros((model.world_count,), dtype=int)
    world_actuated_joint_coord_offset_np = np.zeros((model.world_count,), dtype=int)
    world_actuated_joint_dofs_offset_np = np.zeros((model.world_count,), dtype=int)
    world_passive_joint_coord_offset_np = np.zeros((model.world_count,), dtype=int)
    world_passive_joint_dofs_offset_np = np.zeros((model.world_count,), dtype=int)
    world_joint_bilateral_cts_offset_np = np.zeros((model.world_count,), dtype=int)
    world_joint_dynamic_cts_offset_np = np.zeros((model.world_count,), dtype=int)
    world_joint_kinematic_cts_offset_np = np.zeros((model.world_count,), dtype=int)
    world_joint_bounded_cts_offset_np = np.zeros((model.world_count,), dtype=int)
    world_joint_friction_cts_offset_np = np.zeros((model.world_count,), dtype=int)
    world_joint_effort_cts_offset_np = np.zeros((model.world_count,), dtype=int)
    for w in range(1, model.world_count):
        world_joint_offset_np[w] = world_joint_offset_np[w - 1] + num_joints_np[w - 1]
        world_joint_coord_offset_np[w] = world_joint_coord_offset_np[w - 1] + num_joint_coords_np[w - 1]
        world_joint_dof_offset_np[w] = world_joint_dof_offset_np[w - 1] + num_joint_dofs_np[w - 1]
        world_actuated_joint_coord_offset_np[w] = (
            world_actuated_joint_coord_offset_np[w - 1] + num_joint_actuated_coords_np[w - 1]
        )
        world_actuated_joint_dofs_offset_np[w] = (
            world_actuated_joint_dofs_offset_np[w - 1] + num_joint_actuated_dofs_np[w - 1]
        )
        world_passive_joint_coord_offset_np[w] = (
            world_passive_joint_coord_offset_np[w - 1] + num_joint_passive_coords_np[w - 1]
        )
        world_passive_joint_dofs_offset_np[w] = (
            world_passive_joint_dofs_offset_np[w - 1] + num_joint_passive_dofs_np[w - 1]
        )
        world_joint_bilateral_cts_offset_np[w] = (
            world_joint_bilateral_cts_offset_np[w - 1] + num_joint_bilateral_cts_np[w - 1]
        )
        world_joint_dynamic_cts_offset_np[w] = (
            world_joint_dynamic_cts_offset_np[w - 1] + num_joint_dynamic_cts_np[w - 1]
        )
        world_joint_kinematic_cts_offset_np[w] = (
            world_joint_kinematic_cts_offset_np[w - 1] + num_joint_kinematic_cts_np[w - 1]
        )
        world_joint_bounded_cts_offset_np[w] = (
            world_joint_bounded_cts_offset_np[w - 1] + num_joint_bounded_cts_np[w - 1]
        )
        world_joint_friction_cts_offset_np[w] = (
            world_joint_friction_cts_offset_np[w - 1] + num_joint_friction_cts_np[w - 1]
        )
        world_joint_effort_cts_offset_np[w] = world_joint_effort_cts_offset_np[w - 1] + num_joint_effort_cts_np[w - 1]

    # Determine the base body and joint indices per world
    base_body_idx_np = np.full((model.world_count,), -1, dtype=int)
    base_joint_idx_np = np.full((model.world_count,), -1, dtype=int)
    body_world_start_np = model.body_world_start.numpy()
    joint_world_start_np = model.joint_world_start.numpy()
    joint_child_np = model.joint_child.numpy()
    joint_parent_np = model.joint_parent.numpy()
    joint_dof_type_np = joint_dof_type.numpy()

    # Assign base bodies based on articulation roots (if articulations are present)
    world_has_non_floating_root = np.zeros((model.world_count,), dtype=bool)
    if model.articulation_count > 0:
        articulation_start_np = model.articulation_start.numpy()
        articulation_world_np = model.articulation_world.numpy()
        # NOTE: We only assign the first articulation rooted by a unary free joint in each world
        for aid in range(model.articulation_count):
            wid = articulation_world_np[aid]
            base_joint = articulation_start_np[aid]
            base_body = joint_child_np[base_joint]
            if base_body_idx_np[wid] == -1 and base_joint_idx_np[wid] == -1:
                if joint_dof_type_np[base_joint] != JointDoFType.FREE or joint_parent_np[base_joint] != -1:
                    world_has_non_floating_root[wid] = True
                    continue
                base_body_idx_np[wid] = base_body
                base_joint_idx_np[wid] = base_joint

    # For worlds without articulations, look for a unary free joint, or use the first body
    for wid in range(model.world_count):
        if base_body_idx_np[wid] != -1:  # World already has a base body
            continue
        # Look for a unary joint, and use it as base joint if it is a free joint
        has_unary_joint = False
        for jid in range(joint_world_start_np[wid], joint_world_start_np[wid + 1]):
            if joint_parent_np[jid] == -1:
                has_unary_joint = True
                if joint_dof_type_np[jid] == JointDoFType.FREE:
                    base_joint_idx_np[wid] = jid
                    base_body_idx_np[wid] = int(joint_child_np[jid])
                    break
        # As a last fallback, set first body in that world as base body (no base joint), if no unary
        # joints were found (else this is not a floating-base model and we assign no base body).
        if base_body_idx_np[wid] == -1 and not has_unary_joint:
            if body_world_start_np[wid] == body_world_start_np[wid + 1]:
                continue
            base_body_idx_np[wid] = body_world_start_np[wid]

    # Record whether there is a world that has no base body.
    has_world_without_base_body = np.any(base_body_idx_np == -1)

    # Update size object
    model_size.sum_of_num_joints = int(num_joints_np.sum())
    model_size.max_of_num_joints = int(num_joints_np.max())
    model_size.sum_of_num_passive_joints = int(num_passive_joints_np.sum())
    model_size.max_of_num_passive_joints = int(num_passive_joints_np.max())
    model_size.sum_of_num_actuated_joints = int(num_actuated_joints_np.sum())
    model_size.max_of_num_actuated_joints = int(num_actuated_joints_np.max())
    model_size.sum_of_num_dynamic_joints = int(num_dynamic_joints_np.sum())
    model_size.max_of_num_dynamic_joints = int(num_dynamic_joints_np.max())
    model_size.sum_of_num_joint_coords = int(num_joint_coords_np.sum())
    model_size.max_of_num_joint_coords = int(num_joint_coords_np.max())
    model_size.sum_of_num_joint_dofs = int(num_joint_dofs_np.sum())
    model_size.max_of_num_joint_dofs = int(num_joint_dofs_np.max())
    model_size.sum_of_num_passive_joint_coords = int(num_joint_passive_coords_np.sum())
    model_size.max_of_num_passive_joint_coords = int(num_joint_passive_coords_np.max())
    model_size.sum_of_num_passive_joint_dofs = int(num_joint_passive_dofs_np.sum())
    model_size.max_of_num_passive_joint_dofs = int(num_joint_passive_dofs_np.max())
    model_size.sum_of_num_actuated_joint_coords = int(num_joint_actuated_coords_np.sum())
    model_size.max_of_num_actuated_joint_coords = int(num_joint_actuated_coords_np.max())
    model_size.sum_of_num_fk_actuated_joint_coords = int(num_joint_fk_actuated_coords_np.sum())
    model_size.max_of_num_fk_actuated_joint_coords = int(num_joint_fk_actuated_coords_np.max())
    model_size.sum_of_num_actuated_joint_dofs = int(num_joint_actuated_dofs_np.sum())
    model_size.max_of_num_actuated_joint_dofs = int(num_joint_actuated_dofs_np.max())
    model_size.sum_of_num_fk_actuated_joint_dofs = int(num_joint_fk_actuated_dofs_np.sum())
    model_size.max_of_num_fk_actuated_joint_dofs = int(num_joint_fk_actuated_dofs_np.max())
    model_size.sum_of_num_bilateral_joint_cts = int(num_joint_bilateral_cts_np.sum())
    model_size.max_of_num_bilateral_joint_cts = int(num_joint_bilateral_cts_np.max())
    model_size.sum_of_num_dynamic_joint_cts = int(num_joint_dynamic_cts_np.sum())
    model_size.max_of_num_dynamic_joint_cts = int(num_joint_dynamic_cts_np.max())
    model_size.sum_of_num_kinematic_joint_cts = int(num_joint_kinematic_cts_np.sum())
    model_size.max_of_num_kinematic_joint_cts = int(num_joint_kinematic_cts_np.max())
    model_size.sum_of_num_bounded_joint_cts = int(num_joint_bounded_cts_np.sum())
    model_size.max_of_num_bounded_joint_cts = int(num_joint_bounded_cts_np.max())
    model_size.sum_of_num_friction_joint_cts = int(num_joint_friction_cts_np.sum())
    model_size.max_of_num_friction_joint_cts = int(num_joint_friction_cts_np.max())
    model_size.sum_of_num_effort_joint_cts = int(num_joint_effort_cts_np.sum())
    model_size.max_of_num_effort_joint_cts = int(num_joint_effort_cts_np.max())
    model_size.sum_of_max_total_cts = int(num_joint_bilateral_cts_np.sum() + num_joint_bounded_cts_np.sum())
    model_size.max_of_max_total_cts = int(np.max(num_joint_bilateral_cts_np + num_joint_bounded_cts_np))

    # Update per-world heterogeneous model info
    model_info.num_passive_joints = num_passive_joints
    model_info.num_actuated_joints = num_actuated_joints
    model_info.num_dynamic_joints = num_dynamic_joints
    model_info.num_joint_coords = num_joint_coords
    model_info.num_joint_dofs = num_joint_dofs
    model_info.num_passive_joint_coords = num_joint_passive_coords
    model_info.num_passive_joint_dofs = num_joint_passive_dofs
    model_info.num_actuated_joint_coords = num_joint_actuated_coords
    model_info.num_actuated_joint_dofs = num_joint_actuated_dofs
    model_info.num_joint_bilateral_cts = num_joint_bilateral_cts
    model_info.num_joint_dynamic_cts = num_joint_dynamic_cts
    model_info.num_joint_kinematic_cts = num_joint_kinematic_cts
    model_info.has_world_without_base_body = has_world_without_base_body
    model_info.num_joint_bounded_cts = num_joint_bounded_cts
    model_info.num_joint_friction_cts = num_joint_friction_cts
    model_info.num_joint_effort_cts = num_joint_effort_cts
    with wp.ScopedDevice(model.device):
        model_info.num_joints = to_warp_int32_array(num_joints_np)
        model_info.joints_offset = to_warp_int32_array(world_joint_offset_np)
        model_info.joint_coords_offset = to_warp_int32_array(world_joint_coord_offset_np)
        model_info.joint_dofs_offset = to_warp_int32_array(world_joint_dof_offset_np)
        model_info.joint_passive_coords_offset = to_warp_int32_array(world_passive_joint_coord_offset_np)
        model_info.joint_passive_dofs_offset = to_warp_int32_array(world_passive_joint_dofs_offset_np)
        model_info.joint_actuated_coords_offset = to_warp_int32_array(world_actuated_joint_coord_offset_np)
        model_info.joint_actuated_dofs_offset = to_warp_int32_array(world_actuated_joint_dofs_offset_np)
        model_info.joint_bilateral_cts_offset = to_warp_int32_array(world_joint_bilateral_cts_offset_np)
        model_info.joint_dynamic_cts_offset = to_warp_int32_array(world_joint_dynamic_cts_offset_np)
        model_info.joint_kinematic_cts_offset = to_warp_int32_array(world_joint_kinematic_cts_offset_np)
        model_info.joint_bounded_cts_offset = to_warp_int32_array(world_joint_bounded_cts_offset_np)
        model_info.joint_friction_cts_offset = to_warp_int32_array(world_joint_friction_cts_offset_np)
        model_info.joint_effort_cts_offset = to_warp_int32_array(world_joint_effort_cts_offset_np)
        model_info.base_body_index = to_warp_int32_array(base_body_idx_np)
        model_info.base_joint_index = to_warp_int32_array(base_joint_idx_np)
        # Can only be allocated after the model size is updated
        dynamic_cts_axis = wp.empty(shape=(model_size.sum_of_num_dynamic_joint_cts,), dtype=wp.int32)
        friction_cts_axis = wp.empty(shape=(model_size.sum_of_num_friction_joint_cts,), dtype=wp.int32)
        effort_cts_axis = wp.empty(shape=(model_size.sum_of_num_effort_joint_cts,), dtype=wp.int32)

    # Convert local (per-world) joint offsets to global by adding per-world prefix offsets in-place
    wp.launch(
        kernel=_globalize_joint_offsets,
        dim=model.joint_count,
        inputs=[
            model.joint_world,
            model_info.joint_coords_offset,
            model_info.joint_dofs_offset,
            model_info.joint_passive_coords_offset,
            model_info.joint_passive_dofs_offset,
            model_info.joint_actuated_coords_offset,
            model_info.joint_actuated_dofs_offset,
            model_info.joint_bilateral_cts_offset,
            model_info.joint_dynamic_cts_offset,
            model_info.joint_kinematic_cts_offset,
            model_info.joint_bounded_cts_offset,
            model_info.joint_friction_cts_offset,
            model_info.joint_effort_cts_offset,
        ],
        outputs=[
            joint_coord_start,
            joint_dofs_start,
            joint_passive_coord_start,
            joint_passive_dofs_start,
            joint_actuated_coord_start,
            joint_actuated_dofs_start,
            joint_bilateral_cts_start,
            joint_dynamic_cts_start,
            joint_kinematic_cts_start,
            joint_bounded_cts_start,
            joint_friction_cts_start,
            joint_effort_cts_start,
        ],
        device=model.device,
    )

    wp.launch(
        kernel=pack_joint_constraint_axes_kernel,
        dim=model.joint_count,
        inputs=[
            model.joint_qd_start,
            model.joint_target_mode,
            model.joint_target_ke,
            model.joint_target_kd,
            model.joint_effort_limit,
            model.joint_armature,
            model.joint_damping,
            model.joint_friction,
            joint_dof_type,
            joint_dynamic_cts_start,
            joint_friction_cts_start,
            joint_effort_cts_start,
            dynamic_cts_axis,
            friction_cts_axis,
            effort_cts_axis,
        ],
        device=model.device,
    )

    # Write the N+1 entry (grand total) into each offset array.
    for start_array, total in (
        (joint_coord_start, model_size.sum_of_num_joint_coords),
        (joint_dofs_start, model_size.sum_of_num_joint_dofs),
        (joint_passive_coord_start, model_size.sum_of_num_passive_joint_coords),
        (joint_passive_dofs_start, model_size.sum_of_num_passive_joint_dofs),
        (joint_actuated_coord_start, model_size.sum_of_num_actuated_joint_coords),
        (joint_actuated_dofs_start, model_size.sum_of_num_actuated_joint_dofs),
        (joint_bilateral_cts_start, model_size.sum_of_num_bilateral_joint_cts),
        (joint_dynamic_cts_start, model_size.sum_of_num_dynamic_joint_cts),
        (joint_kinematic_cts_start, model_size.sum_of_num_kinematic_joint_cts),
        (joint_bounded_cts_start, model_size.sum_of_num_bounded_joint_cts),
        (joint_friction_cts_start, model_size.sum_of_num_friction_joint_cts),
        (joint_effort_cts_start, model_size.sum_of_num_effort_joint_cts),
    ):
        wp.launch(
            write_coeff_kernel,
            dim=1,
            inputs=[start_array, model_size.sum_of_num_joints, total],
            device=model.device,
        )

    # Joints
    model_joints = JointsModel(
        num_joints=model.joint_count,
        label=model.joint_label,
        wid=model.joint_world,
        jid=joint_jid,  # TODO: Remove
        dof_type=joint_dof_type,
        act_type=joint_act_type,
        dof_act_types=joint_dof_act_types,
        dof_act_paths=joint_dof_act_paths,
        fk_act_flag=model.fk_actuation_flag if hasattr(model, "fk_actuation_flag") else None,
        bid_B=model.joint_parent,
        bid_F=model.joint_child,
        B_r_Bj=joint_B_r_B,
        F_r_Fj=joint_F_r_F,
        X_Bj=joint_X_B,
        X_Fj=joint_X_F,
        q_j_min=model.joint_limit_lower,
        q_j_max=model.joint_limit_upper,
        dq_j_max=model.joint_velocity_limit,
        tau_j_max=model.joint_effort_limit,
        a_j=model.joint_armature,
        b_j=model.joint_damping,
        f_j=model.joint_friction,
        k_p_j=model.joint_target_ke,
        k_d_j=model.joint_target_kd,
        q_j_0=model.joint_q,
        dq_j_0=model.joint_qd,
        num_coords=joint_num_coords,
        num_dofs=joint_num_dofs,
        num_bilateral_cts=joint_num_bilateral_cts,
        num_dynamic_cts=joint_num_dynamic_cts,
        num_kinematic_cts=joint_num_kinematic_cts,
        num_bounded_cts=joint_num_bounded_cts,
        num_friction_cts=joint_num_friction_cts,
        num_effort_cts=joint_num_effort_cts,
        coords_offset=joint_coord_start,
        dofs_offset=joint_dofs_start,
        passive_coords_offset=joint_passive_coord_start,
        passive_dofs_offset=joint_passive_dofs_start,
        actuated_coords_offset=joint_actuated_coord_start,
        actuated_dofs_offset=joint_actuated_dofs_start,
        bilateral_cts_offset=joint_bilateral_cts_start,
        dynamic_cts_offset=joint_dynamic_cts_start,
        kinematic_cts_offset=joint_kinematic_cts_start,
        bounded_cts_offset=joint_bounded_cts_start,
        friction_cts_offset=joint_friction_cts_start,
        effort_cts_offset=joint_effort_cts_start,
        dynamic_cts_axis=dynamic_cts_axis,
        friction_cts_axis=friction_cts_axis,
        effort_cts_axis=effort_cts_axis,
    )
    return model_joints


def register_materials(model: Model, materials_manager: MaterialManager) -> np.ndarray:
    """
    Registers all materials from the given model in the materials manager.

    Args:
        model: Newton model.
        materials_manager: Materials manager to register the materials to.

    Returns:
        NumPy array of material indices for each geom.
    """
    # Set up material parameter dictionary
    material_param_indices: dict[tuple[float, float], int] = {}
    for i, material in enumerate(materials_manager.materials):
        # Adding already existing (default) materials from material manager, making sure the values
        # undergo the same transformation as any material parameters in the Newton model (conversion
        # to np.float32)
        mu = float(np.float32(material.static_friction))
        restitution = float(np.float32(material.restitution))
        material_param_indices[(mu, restitution)] = i

    # Newton material parameters
    shape_friction = model.shape_material_mu.numpy().tolist()
    shape_restitution = model.shape_material_restitution.numpy().tolist()
    # Mapping from geom to material index
    geom_material = np.zeros((model.shape_count,), dtype=int)
    # TODO: Integrate world index for shape material
    # shape_world_np = model.shape_world.numpy()

    for s in range(model.shape_count):
        # Check if material with these parameters already exists
        material_desc = (shape_friction[s], shape_restitution[s])
        if material_desc in material_param_indices:
            material_id = material_param_indices[material_desc]
        else:
            material = MaterialDescriptor(
                name=f"{model.shape_label[s]}_material",
                restitution=shape_restitution[s],
                static_friction=shape_friction[s],
                dynamic_friction=shape_friction[s],
                # wid=shape_world_np[s],
            )
            material_id = materials_manager.register(material)
            material_param_indices[material_desc] = material_id
        geom_material[s] = material_id

    return geom_material


def convert_geometries(
    model: Model | ModelView,
    model_size: SizeKamino,
    model_bodies: RigidBodiesModel,
    materials_manager: MaterialManager,
) -> GeometriesModel:
    # Set up materials
    geom_material_np = register_materials(model, materials_manager)

    # Update size object
    model_size.sum_of_num_materials = materials_manager.num_materials
    model_size.max_of_num_materials = materials_manager.num_materials
    model_size.sum_of_num_material_pairs = materials_manager.num_material_pairs
    model_size.max_of_num_material_pairs = materials_manager.num_material_pairs

    # Convert shapes to the Kamino data structure
    with wp.ScopedDevice(model.device):
        geom_gid = wp.zeros((model.shape_count,), dtype=wp.int32)
        geom_material = to_warp_int32_array(geom_material_np)
        model_num_collidable_geoms = wp.zeros((1,), dtype=wp.int32)

    wp.launch(
        kernel=geometry_conversion_kernel,
        dim=model.shape_count,
        inputs=[
            model.shape_world,
            model.shape_world_start,
            model.shape_flags,
            model.shape_collision_group,
            geom_material,
        ],
        outputs=[
            geom_gid,
            model_num_collidable_geoms,
        ],
        device=model.device,
    )

    # Compute total number of required contacts per world
    if model.rigid_contact_max > 0:
        model_min_contacts = int(model.rigid_contact_max)
        min_contacts_per_world = model.rigid_contact_max // model.world_count
        world_min_contacts = [min_contacts_per_world] * model.world_count
    else:
        model_min_contacts, world_min_contacts = compute_required_contact_capacity(model)

    # Convert shape offsets from body-frame-relative to COM-relative
    offset = wp.zeros_like(model.shape_transform)
    convert_geom_offset_origin_to_com(
        model_bodies.i_r_com_i,
        model.shape_body,
        model.shape_transform,
        offset,
    )

    # Create additional collision detection meta-data
    sorted_excluded_pairs = model.shape_collision_filter_pairs_array()
    excluded_pairs = wp.array(sorted_excluded_pairs, dtype=wp.vec2i, device=model.device)

    return GeometriesModel(
        num_geoms=model.shape_count,
        num_collidable=model_num_collidable_geoms.numpy()[0],
        num_collidable_pairs=model.shape_contact_pair_count,
        num_excluded_pairs=len(sorted_excluded_pairs),
        model_minimum_contacts=model_min_contacts,
        world_minimum_contacts=world_min_contacts,
        label=model.shape_label,
        wid=model.shape_world,
        gid=geom_gid,
        bid=model.shape_body,
        type=model.shape_type,
        flags=model.shape_flags,
        ptr=model.shape_source_ptr,
        params=model.shape_scale,
        offset=offset,
        material=geom_material,
        group=model.shape_collision_group,
        gap=model.shape_gap,
        margin=model.shape_margin,
        collidable_pairs=model.shape_contact_pairs,
        excluded_pairs=excluded_pairs,
        heightfield_index=model.shape_heightfield_index,
        heightfield_data=model.heightfield_data,
        heightfield_elevations=model.heightfield_elevations,
        collision_aabb_lower=model.shape_collision_aabb_lower,
        collision_aabb_upper=model.shape_collision_aabb_upper,
        voxel_resolution=model._shape_voxel_resolution,
        collision_radius=model.shape_collision_radius,
    )


def convert_target_dofs_to_target_coords(
    joint_target_dofs: wp.array[wp.float32], joint_target_coords: wp.array[wp.float32], model: ModelKamino
):
    wp.launch(
        target_dofs_to_coords_conversion_kernel,
        dim=model.size.sum_of_num_joints,
        inputs=[
            model.joints.dof_type,
            model.joints.dofs_offset,
            model.joints.coords_offset,
            joint_target_dofs,
            joint_target_coords,
        ],
        device=model.device,
    )


def convert_target_coords_to_target_dofs(
    joint_target_coords: wp.array[wp.float32], joint_target_dofs: wp.array[wp.float32], model: ModelKamino
):
    wp.launch(
        target_coords_to_dofs_conversion_kernel,
        dim=model.size.sum_of_num_joints,
        inputs=[
            model.joints.dof_type,
            model.joints.dofs_offset,
            model.joints.coords_offset,
            joint_target_coords,
            joint_target_dofs,
        ],
        device=model.device,
    )
