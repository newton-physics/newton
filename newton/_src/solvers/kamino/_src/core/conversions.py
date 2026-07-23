# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Provides a set of conversion utilities to bridge Kamino and Newton."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .....geometry import ShapeFlags
from .....sim.model import Model
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
    JointActuationType,
    JointDoFType,
    JointsModel,
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
    "convert_geometries",
    "convert_joints",
    "convert_model_joint_actuation",
    "convert_model_joint_transforms",
    "convert_model_materials",
    "convert_rigid_bodies",
    "convert_target_coords_to_target_dofs",
    "convert_target_dofs_to_target_coords",
    "validate_model_joint_updates",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

###
# Kernels
###


@wp.func
def joint_actuation_type_from_dofs(
    dof_start: int,
    dof_end: int,
    target_mode: wp.array[wp.int32],
) -> int:
    """Aggregate Newton's per-DoF target modes into a Kamino joint actuation type."""
    joint_target_mode = int(0)
    for dof in range(dof_start, dof_end):
        joint_target_mode = max(joint_target_mode, target_mode[dof])
    return JointActuationType.from_newton_wp(joint_target_mode)


@wp.func
def joint_requires_dynamic_constraints(
    dof_start: int,
    dof_end: int,
    armature: wp.array[wp.float32],
    damping: wp.array[wp.float32],
    target_ke: wp.array[wp.float32],
    target_kd: wp.array[wp.float32],
) -> bool:
    """Return whether any DoF makes a joint dynamic."""
    dynamic = bool(False)
    for dof in range(dof_start, dof_end):
        dynamic = dynamic or (armature[dof] > 0.0 or damping[dof] > 0.0 or target_ke[dof] > 0.0 or target_kd[dof] > 0.0)
    return dynamic


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
    num_dynamic_cts: wp.array[wp.int32],
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
        if joint_requires_dynamic_constraints(
            dof_start,
            dof_end,
            joint_armature,
            joint_damping,
            joint_target_ke,
            joint_target_kd,
        ) != (num_dynamic_cts[tid] > 0):
            wp.atomic_min(violations, 0, tid)

    if tid < dof_count:
        current_finite = joint_limit_lower[tid] > JOINT_QMIN or joint_limit_upper[tid] < JOINT_QMAX
        if current_finite != (built_limit_finite[tid] != 0):
            wp.atomic_min(violations, 1, tid)


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
    current_actuation = joint_actuation_type_from_dofs(
        joint_qd_start[joint],
        joint_qd_start[joint + 1],
        joint_target_mode,
    )
    if current_actuation < 0:
        wp.atomic_min(violations, 3, joint)
    elif (current_actuation == JointActuationType.PASSIVE) != (act_type[joint] == JointActuationType.PASSIVE):
        wp.atomic_min(violations, 2, joint)


@wp.kernel
def update_joint_actuation_kernel(
    # Inputs:
    joint_qd_start: wp.array[wp.int32],
    joint_target_mode: wp.array[wp.int32],
    # Outputs:
    act_type: wp.array[wp.int32],
):
    """Update each joint's Kamino actuation type from its target modes."""
    joint = wp.tid()
    act_type[joint] = joint_actuation_type_from_dofs(
        joint_qd_start[joint],
        joint_qd_start[joint + 1],
        joint_target_mode,
    )


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
    model_joint_target_mode: wp.array[wp.int32],
    model_joint_dof_dim: wp.array2d[wp.int32],
    model_joint_q_start: wp.array[wp.int32],
    model_joint_qd_start: wp.array[wp.int32],
    model_joint_armature: wp.array[wp.float32],
    model_joint_damping: wp.array[wp.float32],
    model_joint_target_ke: wp.array[wp.float32],
    model_joint_target_kd: wp.array[wp.float32],
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
    # Outputs:
    joint_jid: wp.array[wp.int32],
    joint_dof_type: wp.array[wp.int32],
    joint_act_type: wp.array[wp.int32],
    joint_num_coords: wp.array[wp.int32],
    joint_num_dofs: wp.array[wp.int32],
    joint_num_cts: wp.array[wp.int32],
    joint_num_dynamic_cts: wp.array[wp.int32],
    joint_num_kinematic_cts: wp.array[wp.int32],
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
    for i in range(qd_count_j):
        limit_upper_j[i] = joint_limit_upper[dofs_start_j + i]
        limit_lower_j[i] = joint_limit_lower[dofs_start_j + i]
    dof_type_j = JointDoFType.from_newton_wp(type_j, q_count_j, qd_count_j, dof_dim_j, limit_lower_j, limit_upper_j)
    assert dof_type_j >= 0, "Joint DoF type must be valid"

    # Get joint type properties
    ncoords_j = JointDoFType.num_coords_wp(dof_type_j)
    ndofs_j = JointDoFType.num_dofs_wp(dof_type_j)
    ncts_j = JointDoFType.num_cts_wp(dof_type_j)
    assert ncoords_j >= 0, "Number of joint coordinates must be valid"
    assert ndofs_j >= 0, "Number of joint DoFs must be valid"
    assert ncts_j >= 0, "Number of joint constraints must be valid"
    joint_dof_type[joint_id] = dof_type_j
    joint_num_coords[joint_id] = ncoords_j
    joint_num_dofs[joint_id] = ndofs_j

    # Determine Kamino actuation mode for joint
    act_type_j = joint_actuation_type_from_dofs(dofs_start_j, dofs_start_j + ndofs_j, model_joint_target_mode)
    assert act_type_j >= 0, "Joint actuation type must be valid"
    joint_act_type[joint_id] = act_type_j

    # Infer if the joint requires dynamic constraints
    is_dynamic_j = joint_requires_dynamic_constraints(
        dofs_start_j,
        dofs_start_j + ndofs_j,
        model_joint_armature,
        model_joint_damping,
        model_joint_target_ke,
        model_joint_target_kd,
    )

    # Set joint dimensions
    joint_num_kinematic_cts[joint_id] = ncts_j
    if is_dynamic_j:
        joint_num_dynamic_cts[joint_id] = ndofs_j
    joint_num_cts[joint_id] = joint_num_dynamic_cts[joint_id] + joint_num_kinematic_cts[joint_id]


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
    num_joint_cts: wp.array[wp.int32],
    num_joint_dynamic_cts: wp.array[wp.int32],
    num_joint_kinematic_cts: wp.array[wp.int32],
    joint_coord_start: wp.array[wp.int32],
    joint_dofs_start: wp.array[wp.int32],
    joint_actuated_coord_start: wp.array[wp.int32],
    joint_actuated_dofs_start: wp.array[wp.int32],
    joint_passive_coord_start: wp.array[wp.int32],
    joint_passive_dofs_start: wp.array[wp.int32],
    joint_cts_start: wp.array[wp.int32],
    joint_dynamic_cts_start: wp.array[wp.int32],
    joint_kinematic_cts_start: wp.array[wp.int32],
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
    num_cts = int(0)
    num_dynamic_cts = int(0)
    num_kinematic_cts = int(0)

    for jid in range(num_joints_world):
        joint_id = joints_world_start + jid

        # Updating the start indices within the world
        joint_coord_start[joint_id] = num_coords
        joint_dofs_start[joint_id] = num_dofs
        joint_actuated_coord_start[joint_id] = num_actuated_coords
        joint_actuated_dofs_start[joint_id] = num_actuated_dofs
        joint_passive_coord_start[joint_id] = num_passive_coords
        joint_passive_dofs_start[joint_id] = num_passive_dofs
        joint_cts_start[joint_id] = num_cts
        joint_dynamic_cts_start[joint_id] = num_dynamic_cts
        joint_kinematic_cts_start[joint_id] = num_kinematic_cts

        # Reading off joint properties from previous kernel
        ncoords_j = joint_num_coords[joint_id]
        ndofs_j = joint_num_dofs[joint_id]
        n_kin_cts_j = joint_num_kinematic_cts[joint_id]
        n_dyn_cts_j = joint_num_dynamic_cts[joint_id]
        act_type_j = joint_act_type[joint_id]

        # Update world sizes based on joint sizes
        num_coords += ncoords_j
        num_dofs += ndofs_j
        num_cts += n_kin_cts_j
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
            num_cts += n_dyn_cts_j
            num_dynamic_j += 1

    # Write sizes for this world
    num_passive_joints[world_id] = num_passive_j
    num_actuated_joints[world_id] = num_actuated_j
    num_dynamic_joints[world_id] = num_dynamic_j
    num_joint_coords[world_id] = num_coords
    num_joint_dofs[world_id] = num_dofs
    num_joint_cts[world_id] = num_cts
    num_joint_kinematic_cts[world_id] = num_kinematic_cts
    num_joint_dynamic_cts[world_id] = num_dynamic_cts
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
    world_cts_offset: wp.array[wp.int32],
    world_dynamic_cts_offset: wp.array[wp.int32],
    world_kinematic_cts_offset: wp.array[wp.int32],
    # Outputs:
    joint_coord_start: wp.array[wp.int32],
    joint_dofs_start: wp.array[wp.int32],
    joint_passive_coord_start: wp.array[wp.int32],
    joint_passive_dofs_start: wp.array[wp.int32],
    joint_actuated_coord_start: wp.array[wp.int32],
    joint_actuated_dofs_start: wp.array[wp.int32],
    joint_cts_start: wp.array[wp.int32],
    joint_dynamic_cts_start: wp.array[wp.int32],
    joint_kinematic_cts_start: wp.array[wp.int32],
):
    jid = wp.tid()
    w = joint_world[jid]
    joint_coord_start[jid] += world_coord_offset[w]
    joint_dofs_start[jid] += world_dof_offset[w]
    joint_passive_coord_start[jid] += world_passive_coord_offset[w]
    joint_passive_dofs_start[jid] += world_passive_dof_offset[w]
    joint_actuated_coord_start[jid] += world_actuated_coord_offset[w]
    joint_actuated_dofs_start[jid] += world_actuated_dof_offset[w]
    joint_cts_start[jid] += world_cts_offset[w]
    joint_dynamic_cts_start[jid] += world_dynamic_cts_offset[w]
    joint_kinematic_cts_start[jid] += world_kinematic_cts_offset[w]


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
            If `None`, no per-world limit is applied, otherwise it will
            override the computed per-world requirements if it is larger.

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

    # Override the per-world maximum contacts if specified in the settings
    if max_contacts_per_world is not None:
        world_max_contacts = np.minimum(world_max_contacts, max_contacts_per_world)

    # Return the per-world maximum contacts list
    return int(np.sum(world_max_contacts)), world_max_contacts.astype(int).tolist()


def validate_model_joint_updates(
    model: Model,
    joints: JointsModel,
    built_limit_finite: wp.array[wp.int32],
    violations: wp.array[wp.int32],
    *,
    check_dof: bool,
    check_actuation: bool,
) -> int:
    """Validate that runtime joint edits preserve Kamino's structural layout.

    ``violations`` is a four-entry array containing the first index for each
    violation type:
       0: a joint whose dynamic-constraint topology changed
       1: a DoF whose finite-limit state changed
       2: a joint whose passive/actuated partition changed
       3: a joint with an unsupported combination of target modes

    An entry equal to the maximum of the joint and DoF counts indicates that no
    violation of that type was found.

    Args:
        model: The Newton model containing the updated joints to validate.
        joints: The current Kamino joint model, before applying the updates.
        built_limit_finite: The built finite limit state for each DoF.
        violations: The array to store the violations.
        check_dof: Whether to check the DoF updates.
        check_actuation: Whether to check the actuation updates.

    Returns:
        The sentinel value indicating no violations.
    """
    dim = max(model.joint_count, model.joint_dof_count)
    violations.fill_(dim)
    if check_dof and dim > 0:
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
                joints.num_dynamic_cts,
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

    return dim


def convert_model_joint_actuation(model: Model, joints: JointsModel) -> None:
    """Update Kamino's per-joint actuation types from Newton target modes."""
    if model.joint_count == 0:
        return
    wp.launch(
        kernel=update_joint_actuation_kernel,
        dim=model.joint_count,
        inputs=[
            # Inputs:
            model.joint_qd_start,
            model.joint_target_mode,
            # Outputs:
            joints.act_type,
        ],
        device=model.device,
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
    model: Model,
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


def convert_joints(
    model: Model,
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
    # Compute the number of joints per world
    joint_world_start_np = model.joint_world_start.numpy()
    num_joints_np = joint_world_start_np[1 : model.world_count + 1] - joint_world_start_np[: model.world_count]

    # Create joint property arrays
    with wp.ScopedDevice(model.device):
        joint_jid = wp.empty(shape=(model.joint_count,), dtype=wp.int32)
        joint_dof_type = wp.zeros(shape=(model.joint_count,), dtype=wp.int32)
        joint_act_type = wp.zeros(shape=(model.joint_count,), dtype=wp.int32)
        joint_num_coords = wp.zeros(shape=(model.joint_count,), dtype=wp.int32)
        joint_num_dofs = wp.zeros(shape=(model.joint_count,), dtype=wp.int32)
        joint_num_cts = wp.zeros(shape=(model.joint_count,), dtype=wp.int32)
        joint_num_dynamic_cts = wp.zeros(shape=(model.joint_count,), dtype=wp.int32)
        joint_num_kinematic_cts = wp.zeros(shape=(model.joint_count,), dtype=wp.int32)
        joint_B_r_B = wp.empty(shape=(model.joint_count,), dtype=wp.vec3f)
        joint_F_r_F = wp.empty(shape=(model.joint_count,), dtype=wp.vec3f)
        joint_X_B = wp.empty(shape=(model.joint_count,), dtype=wp.mat33f)
        joint_X_F = wp.empty(shape=(model.joint_count,), dtype=wp.mat33f)

    wp.launch(
        kernel=joint_conversion_kernel,
        dim=model.joint_count,
        inputs=[
            # Inputs:
            model.joint_world,
            model.joint_world_start,
            model.joint_type,
            model.joint_target_mode,
            model.joint_dof_dim,
            model.joint_q_start,
            model.joint_qd_start,
            model.joint_armature,
            model.joint_damping,
            model.joint_target_ke,
            model.joint_target_kd,
            model.joint_limit_lower,
            model.joint_limit_upper,
            # Outputs:
            joint_jid,
            joint_dof_type,
            joint_act_type,
            joint_num_coords,
            joint_num_dofs,
            joint_num_cts,
            joint_num_dynamic_cts,
            joint_num_kinematic_cts,
        ],
        device=model.device,
    )

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
        num_joint_cts = wp.zeros(shape=(model.world_count,), dtype=wp.int32)
        num_joint_dynamic_cts = wp.zeros(shape=(model.world_count,), dtype=wp.int32)
        num_joint_kinematic_cts = wp.zeros(shape=(model.world_count,), dtype=wp.int32)
        joint_coord_start = wp.zeros(shape=(model.joint_count + 1,), dtype=wp.int32)
        joint_dofs_start = wp.zeros(shape=(model.joint_count + 1,), dtype=wp.int32)
        joint_actuated_coord_start = wp.zeros(shape=(model.joint_count + 1,), dtype=wp.int32)
        joint_actuated_dofs_start = wp.zeros(shape=(model.joint_count + 1,), dtype=wp.int32)
        joint_passive_coord_start = wp.zeros(shape=(model.joint_count + 1,), dtype=wp.int32)
        joint_passive_dofs_start = wp.zeros(shape=(model.joint_count + 1,), dtype=wp.int32)
        joint_cts_start = wp.zeros(shape=(model.joint_count + 1,), dtype=wp.int32)
        joint_dynamic_cts_start = wp.zeros(shape=(model.joint_count + 1,), dtype=wp.int32)
        joint_kinematic_cts_start = wp.zeros(shape=(model.joint_count + 1,), dtype=wp.int32)

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
            num_joint_cts,
            num_joint_dynamic_cts,
            num_joint_kinematic_cts,
            joint_coord_start,
            joint_dofs_start,
            joint_actuated_coord_start,
            joint_actuated_dofs_start,
            joint_passive_coord_start,
            joint_passive_dofs_start,
            joint_cts_start,
            joint_dynamic_cts_start,
            joint_kinematic_cts_start,
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
    num_joint_cts_np = num_joint_cts.numpy()
    num_joint_dynamic_cts_np = num_joint_dynamic_cts.numpy()
    num_joint_kinematic_cts_np = num_joint_kinematic_cts.numpy()

    # Compute offsets per world
    world_joint_offset_np = np.zeros((model.world_count,), dtype=int)
    world_joint_coord_offset_np = np.zeros((model.world_count,), dtype=int)
    world_joint_dof_offset_np = np.zeros((model.world_count,), dtype=int)
    world_actuated_joint_coord_offset_np = np.zeros((model.world_count,), dtype=int)
    world_actuated_joint_dofs_offset_np = np.zeros((model.world_count,), dtype=int)
    world_passive_joint_coord_offset_np = np.zeros((model.world_count,), dtype=int)
    world_passive_joint_dofs_offset_np = np.zeros((model.world_count,), dtype=int)
    world_joint_cts_offset_np = np.zeros((model.world_count,), dtype=int)
    world_joint_dynamic_cts_offset_np = np.zeros((model.world_count,), dtype=int)
    world_joint_kinematic_cts_offset_np = np.zeros((model.world_count,), dtype=int)
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
        world_joint_cts_offset_np[w] = world_joint_cts_offset_np[w - 1] + num_joint_cts_np[w - 1]
        world_joint_dynamic_cts_offset_np[w] = (
            world_joint_dynamic_cts_offset_np[w - 1] + num_joint_dynamic_cts_np[w - 1]
        )
        world_joint_kinematic_cts_offset_np[w] = (
            world_joint_kinematic_cts_offset_np[w - 1] + num_joint_kinematic_cts_np[w - 1]
        )

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
                msg.warning(f"Zero bodies in world {wid}, no base body assigned.")
                continue
            base_body_idx_np[wid] = body_world_start_np[wid]

    # Only warn for worlds where a skipped root left the world without a base
    if np.any(world_has_non_floating_root & (base_body_idx_np == -1)):
        msg.warning(
            "Model has articulations whose root is not a free joint attached to the world, "
            "disabling floating base resets for those worlds."
        )

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
    model_size.sum_of_num_joint_cts = int(num_joint_cts_np.sum())
    model_size.max_of_num_joint_cts = int(num_joint_cts_np.max())
    model_size.sum_of_num_dynamic_joint_cts = int(num_joint_dynamic_cts_np.sum())
    model_size.max_of_num_dynamic_joint_cts = int(num_joint_dynamic_cts_np.max())
    model_size.sum_of_num_kinematic_joint_cts = int(num_joint_kinematic_cts_np.sum())
    model_size.max_of_num_kinematic_joint_cts = int(num_joint_kinematic_cts_np.max())
    model_size.sum_of_max_total_cts = int(num_joint_cts_np.sum())
    model_size.max_of_max_total_cts = int(num_joint_cts_np.max())

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
    model_info.num_joint_cts = num_joint_cts
    model_info.num_joint_dynamic_cts = num_joint_dynamic_cts
    model_info.num_joint_kinematic_cts = num_joint_kinematic_cts
    with wp.ScopedDevice(model.device):
        model_info.num_joints = to_warp_int32_array(num_joints_np)
        model_info.joints_offset = to_warp_int32_array(world_joint_offset_np)
        model_info.joint_coords_offset = to_warp_int32_array(world_joint_coord_offset_np)
        model_info.joint_dofs_offset = to_warp_int32_array(world_joint_dof_offset_np)
        model_info.joint_passive_coords_offset = to_warp_int32_array(world_passive_joint_coord_offset_np)
        model_info.joint_passive_dofs_offset = to_warp_int32_array(world_passive_joint_dofs_offset_np)
        model_info.joint_actuated_coords_offset = to_warp_int32_array(world_actuated_joint_coord_offset_np)
        model_info.joint_actuated_dofs_offset = to_warp_int32_array(world_actuated_joint_dofs_offset_np)
        model_info.joint_cts_offset = to_warp_int32_array(world_joint_cts_offset_np)
        model_info.joint_dynamic_cts_offset = to_warp_int32_array(world_joint_dynamic_cts_offset_np)
        model_info.joint_kinematic_cts_offset = to_warp_int32_array(world_joint_kinematic_cts_offset_np)
        model_info.base_body_index = to_warp_int32_array(base_body_idx_np)
        model_info.base_joint_index = to_warp_int32_array(base_joint_idx_np)

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
            model_info.joint_cts_offset,
            model_info.joint_dynamic_cts_offset,
            model_info.joint_kinematic_cts_offset,
        ],
        outputs=[
            joint_coord_start,
            joint_dofs_start,
            joint_passive_coord_start,
            joint_passive_dofs_start,
            joint_actuated_coord_start,
            joint_actuated_dofs_start,
            joint_cts_start,
            joint_dynamic_cts_start,
            joint_kinematic_cts_start,
        ],
        device=model.device,
    )

    # Write the N+1 entry (grand total) into each offset array.
    wp.launch(
        write_coeff_kernel,
        dim=1,
        inputs=[joint_coord_start, model_size.sum_of_num_joints, model_size.sum_of_num_joint_coords],
        device=model.device,
    )
    wp.launch(
        write_coeff_kernel,
        dim=1,
        inputs=[joint_dofs_start, model_size.sum_of_num_joints, model_size.sum_of_num_joint_dofs],
        device=model.device,
    )
    wp.launch(
        write_coeff_kernel,
        dim=1,
        inputs=[joint_passive_coord_start, model_size.sum_of_num_joints, model_size.sum_of_num_passive_joint_coords],
        device=model.device,
    )
    wp.launch(
        write_coeff_kernel,
        dim=1,
        inputs=[joint_passive_dofs_start, model_size.sum_of_num_joints, model_size.sum_of_num_passive_joint_dofs],
        device=model.device,
    )
    wp.launch(
        write_coeff_kernel,
        dim=1,
        inputs=[joint_actuated_coord_start, model_size.sum_of_num_joints, model_size.sum_of_num_actuated_joint_coords],
        device=model.device,
    )
    wp.launch(
        write_coeff_kernel,
        dim=1,
        inputs=[joint_actuated_dofs_start, model_size.sum_of_num_joints, model_size.sum_of_num_actuated_joint_dofs],
        device=model.device,
    )
    wp.launch(
        write_coeff_kernel,
        dim=1,
        inputs=[joint_cts_start, model_size.sum_of_num_joints, model_size.sum_of_num_joint_cts],
        device=model.device,
    )
    wp.launch(
        write_coeff_kernel,
        dim=1,
        inputs=[joint_dynamic_cts_start, model_size.sum_of_num_joints, model_size.sum_of_num_dynamic_joint_cts],
        device=model.device,
    )
    wp.launch(
        write_coeff_kernel,
        dim=1,
        inputs=[joint_kinematic_cts_start, model_size.sum_of_num_joints, model_size.sum_of_num_kinematic_joint_cts],
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
        k_p_j=model.joint_target_ke,
        k_d_j=model.joint_target_kd,
        q_j_0=model.joint_q,
        dq_j_0=model.joint_qd,
        num_coords=joint_num_coords,
        num_dofs=joint_num_dofs,
        num_cts=joint_num_cts,
        num_dynamic_cts=joint_num_dynamic_cts,
        num_kinematic_cts=joint_num_kinematic_cts,
        coords_offset=joint_coord_start,
        dofs_offset=joint_dofs_start,
        passive_coords_offset=joint_passive_coord_start,
        passive_dofs_offset=joint_passive_dofs_start,
        actuated_coords_offset=joint_actuated_coord_start,
        actuated_dofs_offset=joint_actuated_dofs_start,
        cts_offset=joint_cts_start,
        dynamic_cts_offset=joint_dynamic_cts_start,
        kinematic_cts_offset=joint_kinematic_cts_start,
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
    model: Model,
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
