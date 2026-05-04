# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Provides a set of conversion utilities to bridge Kamino and Newton."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .....geometry import ShapeFlags
from .....sim.model import Model
from ..core.bodies import RigidBodiesModel, convert_body_origin_to_com
from ..core.size import SizeKamino
from .builder import JointActuationType
from .geometry import GeometriesModel
from .joints import (
    JOINT_DQMAX,
    JOINT_QMAX,
    JOINT_QMIN,
    JOINT_TAUMAX,
    JointDoFType,
    JointsModel,
)
from .materials import MaterialDescriptor, MaterialManager
from .shapes import max_contacts_for_shape_pair
from .types import float32, int32, mat33f, mat63f, quatf, transformf, vec2i, vec3f, vec6f

if TYPE_CHECKING:
    from ..core.model import ModelKamino, ModelKaminoInfo

###
# Module interface
###

__all__ = [
    "convert_entity_local_transforms",
    "convert_geometries",
    "convert_joints",
    "convert_model_joint_transforms",
    "convert_rigid_bodies",
    "convert_target_coords_to_target_dofs",
    "convert_target_dofs_to_target_coords",
]


###
# Kernels
###


@wp.kernel
def entity_local_transform_conversion_kernel(
    # Inputs:
    model_joint_world_start: wp.array[int32],
    model_joint_parent: wp.array[int32],
    model_joint_child: wp.array[int32],
    # Outputs:
    body_corr: wp.array[quatf],
    body_com: wp.array[vec3f],
    body_q: wp.array[transformf],
    body_qd: wp.array[vec6f],
    body_inertia: wp.array[mat33f],
    body_inv_inertia: wp.array[mat33f],
    joint_X_p: wp.array[transformf],
    joint_X_c: wp.array[transformf],
):
    # Retrieve the world index
    world_id = wp.tid()
    # Retrieve the joint index range for this world
    joint_id_start = model_joint_world_start[world_id]
    joint_id_end = model_joint_world_start[world_id + 1] - 1

    for joint_id in range(joint_id_start, joint_id_end + 1):
        parent_id = model_joint_parent[joint_id]
        child_id = model_joint_child[joint_id]

        # If the parent body was previously corrected, first update this
        # joint's parent-side transform to the new parent frame.
        joint_X_p_j = joint_X_p[joint_id]
        if parent_id >= 0:
            parent_corr = body_corr[parent_id]
            if not parent_corr[3] == 1.0:
                p_pos = wp.transform_get_translation(joint_X_p_j)
                wp.transform_set_translation(joint_X_p_j, wp.quat_rotate_inv(parent_corr, p_pos))
                p_quat = wp.transform_get_rotation(joint_X_p_j)
                wp.transform_set_rotation(joint_X_p_j, wp.quat_inverse(parent_corr) * p_quat)
                joint_X_p[joint_id] = joint_X_p_j

        # Now compute the correction for this joint's child body
        joint_X_c_j = joint_X_c[joint_id]
        q_cj = wp.transform_get_rotation(joint_X_c_j)
        q_pj = wp.transform_get_rotation(joint_X_p_j)
        q_corr = q_cj * wp.quat_inverse(q_pj)

        if child_id < 0 or wp.abs(q_corr[3] - 1.0) < 1e-5:
            continue

        if not body_corr[child_id][3] == 1.0:
            print(
                "A body is the child of multiple joints requiring joint_X_c "
                "correction. The previous correction will be overwritten, which "
                "may produce incorrect joint constraints for loop-closing joints."
            )
        body_corr[child_id] = q_corr

        # Update child-side joint transform: rotation becomes identity,
        # position re-expressed in the new child frame
        q_corr_inv = wp.quat_inverse(q_corr)
        c_pos = wp.transform_get_translation(joint_X_c_j)
        wp.transform_set_translation(joint_X_c_j, wp.quat_rotate(q_corr_inv, c_pos))
        wp.transform_set_rotation(joint_X_c_j, wp.quat_identity())
        joint_X_c[joint_id] = joint_X_c_j

        # Rotate the child body's local quantities
        body_q_c = body_q[child_id]
        q_old = wp.transform_get_rotation(body_q_c)
        wp.transform_set_rotation(body_q_c, q_old * q_corr)
        body_q[child_id] = body_q_c

        body_com[child_id] = wp.quat_rotate(q_corr_inv, body_com[child_id])

        R_inv_corr = wp.quat_to_matrix(q_corr_inv)
        body_inertia[child_id] = R_inv_corr @ body_inertia[child_id] @ wp.transpose(R_inv_corr)
        body_inv_inertia[child_id] = R_inv_corr @ body_inv_inertia[child_id] @ wp.transpose(R_inv_corr)

        # TODO: Do these need be converted? Aren't they already computed at body CoM?
        body_qd_c = body_qd[child_id]
        body_qd_c[0:3] = R_inv_corr @ body_qd_c[0:3]
        body_qd_c[3:6] = R_inv_corr @ body_qd_c[3:6]
        body_qd[child_id] = body_qd_c


@wp.kernel
def shape_transform_conversion_kernel(
    # Inputs:
    model_shape_body: wp.array[int32],
    body_corr: wp.array[quatf],
    # Outputs:
    shape_transform: wp.array[transformf],
):
    # Retrieve the shape index
    shape_id = wp.tid()

    body_id = model_shape_body[shape_id]
    if body_id < 0:
        return

    q_corr_inv = wp.quat_inverse(body_corr[body_id])

    st = shape_transform[shape_id]
    s_pos = wp.transform_get_translation(st)
    s_quat = wp.transform_get_rotation(st)
    wp.transform_set_translation(st, wp.quat_rotate(q_corr_inv, s_pos))
    wp.transform_set_rotation(st, q_corr_inv * s_quat)
    shape_transform[shape_id] = st


@wp.kernel
def world_max_contacts_kernel(
    # Inputs:
    max_contacts_per_pair: int,
    model_shape_type: wp.array[int32],
    model_shape_world: wp.array[int32],
    model_shape_contact_pair: wp.array[vec2i],
    # Outputs:
    world_max_contacts: wp.array[int32],
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
def rigid_bodies_indexing_kernel(
    # Inputs:
    model_body_world_start: wp.array[int32],
    model_shape_world_start: wp.array[int32],
    # Outputs:
    body_bid: wp.array[int32],
    num_bodies: wp.array[int32],
    num_shapes: wp.array[int32],
    num_body_dofs: wp.array[int32],
    world_body_offset: wp.array[int32],
    world_shape_offset: wp.array[int32],
    world_body_dof_offset: wp.array[int32],
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
def mass_prop_accumulation_kernel(
    # Inputs:
    model_body_world_start: wp.array[int32],
    model_body_mass: wp.array[float32],
    body_inertia: wp.array[mat33f],
    # Outputs:
    mass_total: wp.array[float32],
    mass_min: wp.array[float32],
    mass_max: wp.array[float32],
    inertia_total: wp.array[float32],
):
    # Retrieve the world index
    world_id = wp.tid()
    # Retrieve the body index range for this world
    body_id_start = model_body_world_start[world_id]
    body_id_end = model_body_world_start[world_id + 1] - 1

    mass = float32(0.0)
    m_min = float32(1e10)
    m_max = float32(0.0)
    inertia = float32(0.0)

    for body_id in range(body_id_start, body_id_end + 1):
        mass_b = model_body_mass[body_id]
        mass += mass_b
        if mass_b < m_min:
            m_min = mass_b
        if mass_b > m_max:
            m_max = mass_b
        inertia_diag = wp.get_diag(body_inertia[body_id])
        inertia += 3.0 * mass_b + inertia_diag[0] + inertia_diag[1] + inertia_diag[2]

    mass_total[world_id] = mass
    mass_min[world_id] = m_min
    mass_max[world_id] = m_max
    inertia_total[world_id] = inertia


@wp.kernel
def joint_conversion_kernel(
    # Inputs:
    model_joint_world: wp.array[int32],
    model_joint_world_start: wp.array[int32],
    model_joint_type: wp.array[int32],
    model_joint_target_mode: wp.array[int32],
    model_joint_parent: wp.array[int32],
    model_joint_child: wp.array[int32],
    model_joint_dof_dim: wp.array2d[int32],
    model_joint_q_start: wp.array[int32],
    model_joint_qd_start: wp.array[int32],
    model_joint_armature: wp.array[float32],
    model_joint_friction: wp.array[float32],
    model_joint_target_ke: wp.array[float32],
    model_joint_target_kd: wp.array[float32],
    joint_axis: wp.array[vec3f],
    body_com: wp.array[vec3f],
    joint_X_p: wp.array[transformf],
    joint_X_c: wp.array[transformf],
    joint_limit_lower: wp.array[float32],
    joint_limit_upper: wp.array[float32],
    joint_velocity_limit: wp.array[float32],
    joint_effort_limit: wp.array[float32],
    # Outputs:
    joint_jid: wp.array[int32],
    joint_dof_type: wp.array[int32],
    joint_act_type: wp.array[int32],
    joint_num_coords: wp.array[int32],
    joint_num_dofs: wp.array[int32],
    joint_num_cts: wp.array[int32],
    joint_num_dynamic_cts: wp.array[int32],
    joint_num_kinematic_cts: wp.array[int32],
    joint_B_r_B: wp.array[vec3f],
    joint_F_r_F: wp.array[vec3f],
    joint_X: wp.array[mat33f],
):
    # Retrieve the joint index
    joint_id = wp.tid()

    world_id = model_joint_world[joint_id]
    joint_jid[joint_id] = joint_id - model_joint_world_start[world_id]

    # Determine Kamino joint type
    type_j = model_joint_type[joint_id]
    dof_dim_j = vec2i(model_joint_dof_dim[joint_id, 0], model_joint_dof_dim[joint_id, 1])
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
    joint_dofs_target_mode_j = int(0)
    for dof_id in range(ndofs_j):
        joint_dofs_target_mode_j = max(joint_dofs_target_mode_j, model_joint_target_mode[dofs_start_j + dof_id])
    act_type_j = JointActuationType.from_newton_wp(joint_dofs_target_mode_j)
    assert act_type_j >= 0, "Joint actuation type must be valid"
    joint_act_type[joint_id] = act_type_j

    # Infer if the joint requires dynamic constraints
    is_dynamic_j = bool(False)
    for dof_id in range(ndofs_j):
        a_j = model_joint_armature[dofs_start_j + dof_id]
        b_j = model_joint_friction[dofs_start_j + dof_id]
        ke_j = model_joint_target_ke[dofs_start_j + dof_id]
        kd_j = model_joint_target_kd[dofs_start_j + dof_id]
        is_dynamic_j = is_dynamic_j or (a_j > 0.0) or (b_j > 0.0) or (ke_j > 0.0) or (kd_j > 0.0)

    # Set joint dimensions
    joint_num_kinematic_cts[joint_id] = ncts_j
    if is_dynamic_j:
        joint_num_dynamic_cts[joint_id] = ndofs_j
    joint_num_cts[joint_id] = joint_num_dynamic_cts[joint_id] + joint_num_kinematic_cts[joint_id]

    # Compute joint axes and offsets
    parent_bid = model_joint_parent[joint_id]
    p_r_p_com = vec3f(body_com[parent_bid]) if parent_bid >= 0 else vec3f(0.0, 0.0, 0.0)
    c_r_c_com = vec3f(body_com[model_joint_child[joint_id]])
    X_p_j = joint_X_p[joint_id]
    X_c_j = joint_X_c[joint_id]
    q_p_j = wp.transform_get_rotation(X_p_j)
    p_r_p_j = wp.transform_get_translation(X_p_j)
    c_r_c_j = wp.transform_get_translation(X_c_j)

    dof_axes_j = mat63f()
    for i in range(ndofs_j):
        dof_axes_j[i] = joint_axis[dofs_start_j + i]
    R_axis_j = JointDoFType.axes_matrix_from_joint_type(dof_type_j, dof_axes_j)
    B_r_Bj = p_r_p_j - p_r_p_com
    F_r_Fj = c_r_c_j - c_r_c_com
    X_j = wp.quat_to_matrix(q_p_j) @ R_axis_j
    joint_B_r_B[joint_id] = B_r_Bj
    joint_F_r_F[joint_id] = F_r_Fj
    joint_X[joint_id] = X_j

    # Clip joint limits and effort/velocity limits to supported ranges
    for i in range(qd_count_j):
        joint_limit_lower[dofs_start_j + i] = wp.clamp(joint_limit_lower[dofs_start_j + i], JOINT_QMIN, JOINT_QMAX)
        joint_limit_upper[dofs_start_j + i] = wp.clamp(joint_limit_upper[dofs_start_j + i], JOINT_QMIN, JOINT_QMAX)
        joint_velocity_limit[dofs_start_j + i] = wp.clamp(
            joint_velocity_limit[dofs_start_j + i], -JOINT_DQMAX, JOINT_DQMAX
        )
        joint_effort_limit[dofs_start_j + i] = wp.clamp(
            joint_effort_limit[dofs_start_j + i], -JOINT_TAUMAX, JOINT_TAUMAX
        )


@wp.kernel
def joint_indexing_kernel(
    # Inputs:
    model_joint_world_start: wp.array[int32],
    joint_act_type: wp.array[int32],
    joint_num_coords: wp.array[int32],
    joint_num_dofs: wp.array[int32],
    joint_num_kinematic_cts: wp.array[int32],
    joint_num_dynamic_cts: wp.array[int32],
    # Outputs:
    num_passive_joints: wp.array[int32],
    num_actuated_joints: wp.array[int32],
    num_dynamic_joints: wp.array[int32],
    num_joint_coords: wp.array[int32],
    num_joint_dofs: wp.array[int32],
    num_joint_passive_coords: wp.array[int32],
    num_joint_passive_dofs: wp.array[int32],
    num_joint_actuated_coords: wp.array[int32],
    num_joint_actuated_dofs: wp.array[int32],
    num_joint_cts: wp.array[int32],
    num_joint_dynamic_cts: wp.array[int32],
    num_joint_kinematic_cts: wp.array[int32],
    joint_coord_start: wp.array[int32],
    joint_dofs_start: wp.array[int32],
    joint_actuated_coord_start: wp.array[int32],
    joint_actuated_dofs_start: wp.array[int32],
    joint_passive_coord_start: wp.array[int32],
    joint_passive_dofs_start: wp.array[int32],
    joint_cts_start: wp.array[int32],
    joint_dynamic_cts_start: wp.array[int32],
    joint_kinematic_cts_start: wp.array[int32],
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
    num_actuated_dofs = int(0)
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
        else:
            num_passive_j += 1
            num_passive_coords += ncoords_j
            num_passive_dofs += ndofs_j

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
    num_joint_actuated_dofs[world_id] = num_actuated_dofs
    num_joint_passive_coords[world_id] = num_passive_coords
    num_joint_passive_dofs[world_id] = num_passive_dofs


@wp.kernel
def _globalize_joint_offsets(
    # Inputs:
    joint_world: wp.array[int32],
    world_coord_offset: wp.array[int32],
    world_dof_offset: wp.array[int32],
    world_passive_coord_offset: wp.array[int32],
    world_passive_dof_offset: wp.array[int32],
    world_actuated_coord_offset: wp.array[int32],
    world_actuated_dof_offset: wp.array[int32],
    world_cts_offset: wp.array[int32],
    world_dynamic_cts_offset: wp.array[int32],
    world_kinematic_cts_offset: wp.array[int32],
    # Outputs:
    joint_coord_start: wp.array[int32],
    joint_dofs_start: wp.array[int32],
    joint_passive_coord_start: wp.array[int32],
    joint_passive_dofs_start: wp.array[int32],
    joint_actuated_coord_start: wp.array[int32],
    joint_actuated_dofs_start: wp.array[int32],
    joint_cts_start: wp.array[int32],
    joint_dynamic_cts_start: wp.array[int32],
    joint_kinematic_cts_start: wp.array[int32],
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
    model_shape_world: wp.array[int32],
    model_shape_world_start: wp.array[int32],
    model_shape_flags: wp.array[int32],
    model_shape_collision_groups: wp.array[int32],
    geom_material: wp.array[int32],
    # Outputs:
    geom_gid: wp.array[int32],
    model_num_collidable_geoms: wp.array[int32],
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
    model_joints_dof_type: wp.array[int32],
    model_joints_dofs_offset: wp.array[int32],
    model_joints_coords_offset: wp.array[int32],
    joint_target_dofs: wp.array[float32],
    # Outputs
    joint_target_coords: wp.array[float32],
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
        angles = vec3f(
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
    model_joints_dof_type: wp.array[int32],
    model_joints_dofs_offset: wp.array[int32],
    model_joints_coords_offset: wp.array[int32],
    joint_target_coords: wp.array[float32],
    # Outputs
    joint_target_dofs: wp.array[float32],
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
def write_coeff_kernel(a: wp.array[int32], idx: int, v: int):
    """Helper kernel writing a single array coefficient"""
    a[idx] = v


###
# Functions
###


def convert_entity_local_transforms(model: Model) -> dict[str, wp.array]:
    """
    Converts all entity-local transforms (i.e. of bodies, joints and shapes) in the
    given Newton model to a format that is compatible with Kamino's constraint system.

    This involves absorbing any non-identity :attr:`Model.joint_X_c`
    rotations into the child body frames, and updating all downstream
    :attr:`Model.joint_X_p` and :attr:`Model.shape_transform` accordingly.

    Args:
        model (Model): Newton model to be modified in-place.
    """
    # ---------------------------------------------------------------------------
    # Pre-processing: absorb non-identity joint_X_c rotations into child body
    # frames so that Kamino sees aligned joint frames on both sides.
    #
    # Kamino's constraint system assumes a single joint frame X_j valid for both
    # the base (parent) and follower (child) bodies.  At q = 0 it requires
    #   q_base^{-1} * q_follower = identity
    #
    # Newton, however, allows different parent / child joint-frame orientations
    # via joint_X_p and joint_X_c.  At q = 0 Newton's FK gives:
    #   q_follower = q_parent * q_pj * inv(q_cj)
    # so q_base^{-1} * q_follower = q_pj * inv(q_cj) which is generally not
    # identity.
    #
    # To fix this we apply a per-body correction rotation q_corr = q_cj * inv(q_pj)
    # (applied on the right) to each child body's frame:
    #   q_body_new = q_body_old * q_corr
    #
    # This makes q_base^{-1} * q_follower_new = identity at q = 0, and the joint
    # rotation axis R(q_pj) * axis is preserved.
    #
    # All body-local quantities (CoM, inertia, shapes) are re-expressed in the
    # rotated frame, and downstream joint_X_p transforms are updated to account
    # for the parent body's frame change.
    # ---------------------------------------------------------------------------

    # Work on copies so the original Newton model is not mutated
    body_com = wp.clone(model.body_com)
    body_q = wp.clone(model.body_q)
    body_qd = wp.clone(model.body_qd)
    body_inertia = wp.clone(model.body_inertia)
    body_inv_inertia = wp.clone(model.body_inv_inertia)
    joint_X_p = wp.clone(model.joint_X_p)
    joint_X_c = wp.clone(model.joint_X_c)
    shape_transform = wp.clone(model.shape_transform)

    # Process joints in tree order (Newton stores them parent-before-child).
    # For each joint whose q_pj * inv(q_cj) is not identity, we apply a
    # correction q_corr to the child body's frame and immediately propagate
    # to all downstream joints that reference the corrected body as parent.
    # body_corr: dict[int, np.ndarray] = {}  # body_index -> cumulative q_corr
    body_corr = wp.full(
        shape=(model.body_count,), value=wp.quat_identity(dtype=float32), dtype=quatf, device=model.device
    )

    # Convert bodies, sequentially per world
    wp.launch(
        kernel=entity_local_transform_conversion_kernel,
        dim=model.world_count,
        inputs=[
            model.joint_world_start,
            model.joint_parent,
            model.joint_child,
        ],
        outputs=[
            body_corr,
            body_com,
            body_q,
            body_qd,
            body_inertia,
            body_inv_inertia,
            joint_X_p,
            joint_X_c,
        ],
        device=model.device,
    )

    # Convert shapes based on body corrections
    wp.launch(
        kernel=shape_transform_conversion_kernel,
        dim=model.shape_count,
        inputs=[model.shape_body, body_corr],
        outputs=[shape_transform],
        device=model.device,
    )

    # Return the converted transforms as warp arrays
    # to be used for constructing the Kamino model
    return {
        "body_q": body_q,
        "body_qd": body_qd,
        "body_com": body_com,
        "body_inertia": body_inertia,
        "body_inv_inertia": body_inv_inertia,
        "shape_transform": shape_transform,
        "joint_X_p": joint_X_p,
        "joint_X_c": joint_X_c,
    }


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
        model (Model):
            The Newton model for which to compute the required contact capacity.
        max_contacts_per_pair (int, optional):
            Optional maximum number of contacts to allocate per shape pair.
            If `None`, no per-pair limit is applied.
        max_contacts_per_world (int, optional):
            Optional maximum number of contacts to allocate per world.
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
    world_max_contacts_wp = wp.zeros((model.world_count,), dtype=int32, device=model.device)
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


# TODO: Re-implement this using a kernel to run in parallel on the GPU if possible
# TODO: FIX THIS: This is not correct, it is not using the new joint_X_p and joint_X_c transforms.
def convert_model_joint_transforms(model: Model, joints: JointsModel) -> None:
    """
    Converts the joint model parameterization of Newton's to Kamino's format.

    This essentially involves computing the B_r_Bj, F_r_Fj and
    X_j arrays from the joint_X_p and joint_X_c transforms.

    Args:
    - model (Model):
        The input Newton model containing the joint information to be converted.
    - joints (JointsModel):
        The output JointsModel instance where the converted joint data will be stored.
        This function modifies the `joints` object in-place.
    """
    joint_X_p_np = model.joint_X_p.numpy()
    joint_X_c_np = model.joint_X_c.numpy()
    body_com_np = model.body_com.numpy()
    joint_parent_np = model.joint_parent.numpy()
    joint_child_np = model.joint_child.numpy()
    joint_axis_np = model.joint_axis.numpy()
    joint_qd_start_np = model.joint_qd_start.numpy()
    dof_type_np = joints.dof_type.numpy()

    n_joints = model.joint_count
    B_r_Bj_np = np.zeros((n_joints, 3), dtype=np.float32)
    F_r_Fj_np = np.zeros((n_joints, 3), dtype=np.float32)
    X_j_np = np.zeros((n_joints, 9), dtype=np.float32)

    for j in range(n_joints):
        dof_type_j = JointDoFType(int(dof_type_np[j]))
        dofs_start_j = int(joint_qd_start_np[j])
        ndofs_j = dof_type_j.num_dofs
        joint_axes_j = mat63f(joint_axis_np[dofs_start_j : dofs_start_j + ndofs_j])
        R_axis_j = JointDoFType.axes_matrix_from_joint_type(dof_type=dof_type_j, dof_axes=joint_axes_j)

        parent_bid = int(joint_parent_np[j])
        p_r_p_com = wp.vec3f(body_com_np[parent_bid]) if parent_bid >= 0 else wp.vec3f(0.0)
        c_r_c_com = wp.vec3f(body_com_np[int(joint_child_np[j])])

        X_p_j = wp.transformf(*joint_X_p_np[j, :])
        X_c_j = wp.transformf(*joint_X_c_np[j, :])
        q_p_j = wp.transform_get_rotation(X_p_j)
        p_r_p_j = wp.transform_get_translation(X_p_j)
        c_r_c_j = wp.transform_get_translation(X_c_j)

        B_r_Bj_np[j, :] = p_r_p_j - p_r_p_com
        F_r_Fj_np[j, :] = c_r_c_j - c_r_c_com
        X_j_np[j, :] = wp.quat_to_matrix(q_p_j) @ R_axis_j

    joints.B_r_Bj.assign(B_r_Bj_np)
    joints.F_r_Fj.assign(F_r_Fj_np)
    joints.X_j.assign(X_j_np.reshape((n_joints, 3, 3)))


def convert_rigid_bodies(
    model: Model,
    model_size: SizeKamino,
    model_info: ModelKaminoInfo,
    body_com: wp.array,
    body_q: wp.array,
    body_qd: wp.array,
    body_inertia: wp.array,
    body_inv_inertia: wp.array,
) -> RigidBodiesModel:
    """
    Converts the rigid bodies from a Newton model into Kamino's format. The function
    will create a new `RigidBodiesModel` object and fill in the rigid body and shape
    entries of the provided `SizeKamino` and `ModelKaminoInfo` objects.

    This function requires that the preprocessing that absorbs non-identity joint
    rotations into child body frames has already been computed, and the conversion
    result is passed in as arguments.

    Args:
        model: Newton model.
        model_size: Model size object, to be filled in by the function.
        model_info: Model info object, to be filled in by the function.
        body_com: Preprocessed rigid body center of mass positions.
        body_q: Preprocessed initial rigid body poses.
        body_qd: Preprocessed initial rigid body velocities.
        body_inertia: Preprocessed rigid body inertias.
        body_inv_inertia: Preprocessed inverse rigid body inertias.

    Returns:
        Fully converted rigid bodies model in Kamino's format.
    """

    # Compute the offsets and number of entities per world
    with wp.ScopedDevice(model.device):
        body_bid = wp.zeros((model.body_count,), dtype=int32)
        num_bodies = wp.zeros((model.world_count,), dtype=int32)
        num_shapes = wp.zeros((model.world_count,), dtype=int32)
        num_body_dofs = wp.zeros((model.world_count,), dtype=int32)
        world_body_offset = wp.zeros((model.world_count + 1,), dtype=int32)
        world_shape_offset = wp.zeros((model.world_count,), dtype=int32)
        world_body_dof_offset = wp.zeros((model.world_count,), dtype=int32)
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

    # Construct per-world inertial summaries
    with wp.ScopedDevice(model.device):
        mass_total = wp.empty((model.world_count,), dtype=float32)
        mass_min = wp.empty((model.world_count,), dtype=float32)
        mass_max = wp.empty((model.world_count,), dtype=float32)
        inertia_total = wp.empty((model.world_count,), dtype=float32)
    wp.launch(
        kernel=mass_prop_accumulation_kernel,
        dim=model.world_count,
        inputs=[
            model.body_world_start,
            model.body_mass,
            body_inertia,
        ],
        outputs=[
            mass_total,
            mass_min,
            mass_max,
            inertia_total,
        ],
        device=model.device,
    )

    # model.body_q stores body-origin world poses, but Kamino expects
    # COM world poses (joint attachment vectors are COM-relative).
    q_i_0 = wp.empty((model.body_count,), dtype=transformf, device=model.device)
    convert_body_origin_to_com(body_com, body_q, q_i_0)

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
    model_info.mass_min = mass_min
    model_info.mass_max = mass_max
    model_info.mass_total = mass_total
    model_info.inertia_total = inertia_total

    model_bodies = RigidBodiesModel(
        num_bodies=model.body_count,
        label=model.body_label,
        wid=model.body_world,
        bid=body_bid,  # TODO: Remove
        m_i=model.body_mass,
        inv_m_i=model.body_inv_mass,
        i_r_com_i=body_com,
        i_I_i=body_inertia,
        inv_i_I_i=body_inv_inertia,
        q_i_0=q_i_0,
        u_i_0=body_qd,
    )
    return model_bodies


def convert_joints(
    model: Model,
    model_size: SizeKamino,
    model_info: ModelKaminoInfo,
    body_com: wp.array,
    joint_X_p: wp.array,
    joint_X_c: wp.array,
) -> JointsModel:
    """
    Converts the joints from a Newton model into Kamino's format. The function will
    create a new `JointsModel` object and fill in the joint entries of the provided
    `SizeKamino` and `ModelKaminoInfo` objects.

    This function requires that the preprocessing that absorbs non-identity joint
    rotations into child body frames has already been computed, and the conversion
    result is passed in as arguments.

    Args:
        model: Newton model.
        model_size: Model size object, to be filled in by the function.
        model_info: Model info object, to be filled in by the function.
        body_com: Preprocessed rigid body center of mass positions.
        joint_X_p: Preprocessed joint frames in parent frame.
        joint_X_c: Preprocessed joint frames in child frame.

    Returns:
        Fully converted joints model in Kamino's format.
    """
    # Compute the number of joints per world
    joint_world_start_np = model.joint_world_start.numpy()
    num_joints_np = joint_world_start_np[1 : model.world_count + 1] - joint_world_start_np[: model.world_count]

    # Create joint property arrays
    with wp.ScopedDevice(model.device):
        joint_jid = wp.empty(shape=(model.joint_count,), dtype=int32)
        joint_dof_type = wp.zeros(shape=(model.joint_count,), dtype=int32)
        joint_act_type = wp.zeros(shape=(model.joint_count,), dtype=int32)
        joint_num_coords = wp.zeros(shape=(model.joint_count,), dtype=int32)
        joint_num_dofs = wp.zeros(shape=(model.joint_count,), dtype=int32)
        joint_num_cts = wp.zeros(shape=(model.joint_count,), dtype=int32)
        joint_num_dynamic_cts = wp.zeros(shape=(model.joint_count,), dtype=int32)
        joint_num_kinematic_cts = wp.zeros(shape=(model.joint_count,), dtype=int32)
        joint_B_r_B = wp.empty(shape=(model.joint_count,), dtype=vec3f)
        joint_F_r_F = wp.empty(shape=(model.joint_count,), dtype=vec3f)
        joint_X = wp.empty(shape=(model.joint_count,), dtype=mat33f)

    # Copy limit arrays
    joint_limit_lower = wp.clone(model.joint_limit_lower)
    joint_limit_upper = wp.clone(model.joint_limit_upper)
    joint_velocity_limit = wp.clone(model.joint_velocity_limit)
    joint_effort_limit = wp.clone(model.joint_effort_limit)

    wp.launch(
        kernel=joint_conversion_kernel,
        dim=model.joint_count,
        inputs=[
            # Inputs:
            model.joint_world,
            model.joint_world_start,
            model.joint_type,
            model.joint_target_mode,
            model.joint_parent,
            model.joint_child,
            model.joint_dof_dim,
            model.joint_q_start,
            model.joint_qd_start,
            model.joint_armature,
            model.joint_friction,
            model.joint_target_ke,
            model.joint_target_kd,
            model.joint_axis,
            body_com,
            joint_X_p,
            joint_X_c,
            joint_limit_lower,
            joint_limit_upper,
            joint_velocity_limit,
            joint_effort_limit,
            # Outputs:
            joint_jid,
            joint_dof_type,
            joint_act_type,
            joint_num_coords,
            joint_num_dofs,
            joint_num_cts,
            joint_num_dynamic_cts,
            joint_num_kinematic_cts,
            joint_B_r_B,
            joint_F_r_F,
            joint_X,
        ],
        device=model.device,
    )

    # Compute sizes and indices for all joint properties
    with wp.ScopedDevice(model.device):
        num_passive_joints = wp.zeros(shape=(model.world_count,), dtype=int32)
        num_actuated_joints = wp.zeros(shape=(model.world_count,), dtype=int32)
        num_dynamic_joints = wp.zeros(shape=(model.world_count,), dtype=int32)
        num_joint_coords = wp.zeros(shape=(model.world_count,), dtype=int32)
        num_joint_dofs = wp.zeros(shape=(model.world_count,), dtype=int32)
        num_joint_passive_coords = wp.zeros(shape=(model.world_count,), dtype=int32)
        num_joint_passive_dofs = wp.zeros(shape=(model.world_count,), dtype=int32)
        num_joint_actuated_coords = wp.zeros(shape=(model.world_count,), dtype=int32)
        num_joint_actuated_dofs = wp.zeros(shape=(model.world_count,), dtype=int32)
        num_joint_cts = wp.zeros(shape=(model.world_count,), dtype=int32)
        num_joint_dynamic_cts = wp.zeros(shape=(model.world_count,), dtype=int32)
        num_joint_kinematic_cts = wp.zeros(shape=(model.world_count,), dtype=int32)
        joint_coord_start = wp.zeros(shape=(model.joint_count + 1,), dtype=int32)
        joint_dofs_start = wp.zeros(shape=(model.joint_count + 1,), dtype=int32)
        joint_actuated_coord_start = wp.zeros(shape=(model.joint_count + 1,), dtype=int32)
        joint_actuated_dofs_start = wp.zeros(shape=(model.joint_count + 1,), dtype=int32)
        joint_passive_coord_start = wp.zeros(shape=(model.joint_count + 1,), dtype=int32)
        joint_passive_dofs_start = wp.zeros(shape=(model.joint_count + 1,), dtype=int32)
        joint_cts_start = wp.zeros(shape=(model.joint_count + 1,), dtype=int32)
        joint_dynamic_cts_start = wp.zeros(shape=(model.joint_count + 1,), dtype=int32)
        joint_kinematic_cts_start = wp.zeros(shape=(model.joint_count + 1,), dtype=int32)

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
            num_joint_actuated_dofs,
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
    num_joint_actuated_dofs_np = num_joint_actuated_dofs.numpy()
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
    body_world_np = model.body_world.numpy()
    joint_world_np = model.joint_world.numpy()
    body_world_start_np = model.body_world_start.numpy()

    # Check for articulations
    if model.articulation_count > 0:
        articulation_start_np = model.articulation_start.numpy()
        articulation_world_np = model.articulation_world.numpy()
        joint_child_np = model.joint_child.numpy()
        # For each articulation, assign its base body and joint to the corresponding world
        # NOTE: We only assign the first articulation found in each world
        for aid in range(model.articulation_count):
            wid = articulation_world_np[aid]
            base_joint = articulation_start_np[aid]
            base_body = joint_child_np[base_joint]
            if base_body_idx_np[wid] == -1 and base_joint_idx_np[wid] == -1:
                base_body_idx_np[wid] = base_body
                base_joint_idx_np[wid] = base_joint

    # Check for root joint (i.e. joint with no parent body (= -1))
    elif model.joint_count > 0:
        joint_parent_np = model.joint_parent.numpy()
        joint_child_np = model.joint_child.numpy()
        # TODO: How to handle no free joint being defined?
        # Create a list of joint indices with parent body == -1 for each world
        world_parent_joints: dict[int, list[int]] = {w: [] for w in range(model.world_count)}
        for j in range(model.joint_count):
            wid_j = joint_world_np[j]
            parent_j = joint_parent_np[j]
            if parent_j == -1:
                world_parent_joints[wid_j].append(j)
        # For each world, assign the base body and joint based on the first joint with parent == -1,
        # If no joint with parent == -1 is found in a world, then assign the first body as base
        # If multiple joints with parent == -1 are found in a world, then assign the first one as the base
        for w in range(model.world_count):
            if len(world_parent_joints[w]) > 0:
                j = world_parent_joints[w][0]
                base_joint_idx_np[w] = j
                base_body_idx_np[w] = int(joint_child_np[j])
            else:
                base_body_idx_np[w] = int(body_world_start_np[w])
                base_joint_idx_np[w] = -1

    # Fall-back: first body and joint in the world
    else:
        for w in range(model.world_count):
            # Base body: first body in the world
            for b in range(model.body_count):
                if body_world_np[b] == w:
                    base_body_idx_np[w] = b
                    break
            # Base joint: first joint in the world
            for j in range(model.joint_count):
                if joint_world_np[j] == w:
                    base_joint_idx_np[w] = j
                    break

    # Ensure that all worlds have a base body assigned
    for w in range(model.world_count):
        if base_body_idx_np[w] == -1:
            raise ValueError(f"World {w} does not have a base body assigned (index is -1).")

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
    model_size.sum_of_num_actuated_joint_dofs = int(num_joint_actuated_dofs_np.sum())
    model_size.max_of_num_actuated_joint_dofs = int(num_joint_actuated_dofs_np.max())
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
        model_info.num_joints = wp.array(num_joints_np, dtype=int32)
        model_info.joints_offset = wp.array(world_joint_offset_np, dtype=int32)
        model_info.joint_coords_offset = wp.array(world_joint_coord_offset_np, dtype=int32)
        model_info.joint_dofs_offset = wp.array(world_joint_dof_offset_np, dtype=int32)
        model_info.joint_passive_coords_offset = wp.array(world_passive_joint_coord_offset_np, dtype=int32)
        model_info.joint_passive_dofs_offset = wp.array(world_passive_joint_dofs_offset_np, dtype=int32)
        model_info.joint_actuated_coords_offset = wp.array(world_actuated_joint_coord_offset_np, dtype=int32)
        model_info.joint_actuated_dofs_offset = wp.array(world_actuated_joint_dofs_offset_np, dtype=int32)
        model_info.joint_cts_offset = wp.array(world_joint_cts_offset_np, dtype=int32)
        model_info.joint_dynamic_cts_offset = wp.array(world_joint_dynamic_cts_offset_np, dtype=int32)
        model_info.joint_kinematic_cts_offset = wp.array(world_joint_kinematic_cts_offset_np, dtype=int32)
        model_info.base_body_index = wp.array(base_body_idx_np, dtype=int32)
        model_info.base_joint_index = wp.array(base_joint_idx_np, dtype=int32)

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
        bid_B=model.joint_parent,
        bid_F=model.joint_child,
        B_r_Bj=joint_B_r_B,
        F_r_Fj=joint_F_r_F,
        X_j=joint_X,
        q_j_min=joint_limit_lower,
        q_j_max=joint_limit_upper,
        dq_j_max=joint_velocity_limit,
        tau_j_max=joint_effort_limit,
        a_j=model.joint_armature,
        b_j=model.joint_friction,  # TODO: Is this the right attribute?
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
    model: Model, model_size: SizeKamino, materials_manager: MaterialManager, shape_transform: wp.array
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
        geom_gid = wp.zeros((model.shape_count,), dtype=int32)
        geom_material = wp.from_numpy(geom_material_np, dtype=int32)
        model_num_collidable_geoms = wp.zeros((1,), dtype=int32)

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

    # Geometries
    model_geoms = GeometriesModel(
        num_geoms=model.shape_count,
        num_collidable=model_num_collidable_geoms.numpy()[0],
        num_collidable_pairs=model.shape_contact_pair_count,
        num_excluded_pairs=len(model.shape_collision_filter_pairs),
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
        offset=wp.zeros_like(model.shape_transform),
        material=geom_material,
        group=model.shape_collision_group,
        gap=model.shape_gap,
        margin=model.shape_margin,
        collidable_pairs=model.shape_contact_pairs,
        excluded_pairs=wp.array(sorted(model.shape_collision_filter_pairs), dtype=vec2i, device=model.device),
        # Mesh / heightfield data pass-through from Newton model
        heightfield_index=model.shape_heightfield_index,
        heightfield_data=model.heightfield_data,
        heightfield_elevations=model.heightfield_elevations,
        collision_aabb_lower=model.shape_collision_aabb_lower,
        collision_aabb_upper=model.shape_collision_aabb_upper,
        voxel_resolution=model._shape_voxel_resolution,
        collision_radius=model.shape_collision_radius,
    )

    return model_geoms


def convert_target_dofs_to_target_coords(
    joint_target_dofs: wp.array, joint_target_coords: wp.array, model: ModelKamino
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
    joint_target_coords: wp.array, joint_target_dofs: wp.array, model: ModelKamino
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
