# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: UNIT TESTS: DEBUG-PRINT HELPERS
"""

import numpy as np

from newton._src.solvers.kamino._src.core.data import DataKamino
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.utils import logger as msg

###
# Model Functions
###


def debug_print_model_size(model: ModelKamino):
    """Print host- and device-side ``model.size`` fields in the debug log."""
    if not msg.is_enabled_for(msg.LogLevel.DEBUG):
        return
    msg.debug("Model Size:")

    # Print the host-side model size meta-data
    msg.debug("model.size.num_worlds: %s", model.size.num_worlds)

    # Print the device-side model size data
    msg.debug("model.size.sum_of_num_bodies: %s", model.size.sum_of_num_bodies)
    msg.debug("model.size.max_of_num_bodies: %s", model.size.max_of_num_bodies)
    msg.debug("model.size.sum_of_num_joints: %s", model.size.sum_of_num_joints)
    msg.debug("model.size.max_of_num_joints: %s", model.size.max_of_num_joints)
    msg.debug("model.size.sum_of_num_material_pairs: %s", model.size.sum_of_num_material_pairs)
    msg.debug("model.size.max_of_num_material_pairs: %s", model.size.max_of_num_material_pairs)
    msg.debug("model.size.sum_of_num_body_dofs: %s", model.size.sum_of_num_body_dofs)
    msg.debug("model.size.max_of_num_body_dofs: %s", model.size.max_of_num_body_dofs)
    msg.debug("model.size.sum_of_num_joint_dofs: %s", model.size.sum_of_num_joint_dofs)
    msg.debug("model.size.max_of_num_joint_dofs: %s", model.size.max_of_num_joint_dofs)
    msg.debug("model.size.sum_of_max_inequalities: %s", model.size.sum_of_max_inequalities)
    msg.debug("model.size.max_of_max_inequalities: %s", model.size.max_of_max_inequalities)
    msg.debug("model.size.sum_of_num_bounded_joint_cts: %s", model.size.sum_of_num_bounded_joint_cts)
    msg.debug("model.size.max_of_num_bounded_joint_cts: %s", model.size.max_of_num_bounded_joint_cts)
    msg.debug("model.size.sum_of_num_friction_joint_cts: %s", model.size.sum_of_num_friction_joint_cts)
    msg.debug("model.size.max_of_num_friction_joint_cts: %s", model.size.max_of_num_friction_joint_cts)
    msg.debug("model.size.sum_of_num_effort_joint_cts: %s", model.size.sum_of_num_effort_joint_cts)
    msg.debug("model.size.max_of_num_effort_joint_cts: %s", model.size.max_of_num_effort_joint_cts)
    msg.debug("model.size.sum_of_max_total_cts: %s", model.size.sum_of_max_total_cts)
    msg.debug("model.size.max_of_max_total_cts: %s", model.size.max_of_max_total_cts)


def debug_print_model_info(model: ModelKamino):
    """Print ``model.info`` and all per-world count arrays in the debug log."""
    if not msg.is_enabled_for(msg.LogLevel.DEBUG):
        return
    msg.debug("===============================================================================")
    msg.debug("Model Info:")
    # Print the host-side model info meta-data
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug("model.info.num_worlds: %s", model.info.num_worlds)
    # Print the device-side model info data
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug("model.info.num_bodies: %s", model.info.num_bodies)
    msg.debug("model.info.num_joints: %s", model.info.num_joints)
    msg.debug("model.info.num_passive_joints: %s", model.info.num_passive_joints)
    msg.debug("model.info.num_actuated_joints: %s", model.info.num_actuated_joints)
    msg.debug("model.info.num_dynamic_joints: %s", model.info.num_dynamic_joints)
    msg.debug("model.info.num_geoms: %s", model.info.num_geoms)
    msg.debug("model.info.max_limits: %s", model.info.max_limits)
    msg.debug("model.info.max_contacts: %s", model.info.max_contacts)
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug("model.info.num_body_dofs: %s", model.info.num_body_dofs)
    msg.debug("model.info.num_joint_coords: %s", model.info.num_joint_coords)
    msg.debug("model.info.num_joint_dofs: %s", model.info.num_joint_dofs)
    msg.debug("model.info.num_passive_joint_coords: %s", model.info.num_passive_joint_coords)
    msg.debug("model.info.num_passive_joint_dofs: %s", model.info.num_passive_joint_dofs)
    msg.debug("model.info.num_actuated_joint_coords: %s", model.info.num_actuated_joint_coords)
    msg.debug("model.info.num_actuated_joint_dofs: %s", model.info.num_actuated_joint_dofs)
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug("model.info.num_joint_bilateral_cts: %s", model.info.num_joint_bilateral_cts)
    msg.debug("model.info.num_joint_dynamic_cts: %s", model.info.num_joint_dynamic_cts)
    msg.debug("model.info.num_joint_kinematic_cts: %s", model.info.num_joint_kinematic_cts)
    msg.debug("model.info.num_joint_bounded_cts: %s", model.info.num_joint_bounded_cts)
    msg.debug("model.info.num_joint_friction_cts: %s", model.info.num_joint_friction_cts)
    msg.debug("model.info.num_joint_effort_cts: %s", model.info.num_joint_effort_cts)
    msg.debug("model.info.max_limit_cts: %s", model.info.max_limit_cts)
    msg.debug("model.info.max_contact_cts: %s", model.info.max_contact_cts)
    msg.debug("model.info.max_total_cts: %s", model.info.max_total_cts)
    # Print the element offsets
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug("model.info.bodies_offset: %s", model.info.bodies_offset)
    msg.debug("model.info.joints_offset: %s", model.info.joints_offset)
    msg.debug("model.info.limits_offset: %s", model.info.limits_offset)
    msg.debug("model.info.contacts_offset: %s", model.info.contacts_offset)
    msg.debug("model.info.inequalities_offset: %s", model.info.inequalities_offset)
    # Print the coords, DoFs and constraint offsets
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug("model.info.body_dofs_offset: %s", model.info.body_dofs_offset)
    msg.debug("model.info.joint_coords_offset: %s", model.info.joint_coords_offset)
    msg.debug("model.info.joint_dofs_offset: %s", model.info.joint_dofs_offset)
    msg.debug("model.info.joint_passive_coords_offset: %s", model.info.joint_passive_coords_offset)
    msg.debug("model.info.joint_passive_dofs_offset: %s", model.info.joint_passive_dofs_offset)
    msg.debug("model.info.joint_actuated_coords_offset: %s", model.info.joint_actuated_coords_offset)
    msg.debug("model.info.joint_actuated_dofs_offset: %s", model.info.joint_actuated_dofs_offset)
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug("model.info.joint_bilateral_cts_offset: %s", model.info.joint_bilateral_cts_offset)
    msg.debug("model.info.joint_dynamic_cts_offset: %s", model.info.joint_dynamic_cts_offset)
    msg.debug("model.info.joint_kinematic_cts_offset: %s", model.info.joint_kinematic_cts_offset)
    msg.debug("model.info.joint_bounded_cts_offset: %s", model.info.joint_bounded_cts_offset)
    msg.debug("model.info.joint_friction_cts_offset: %s", model.info.joint_friction_cts_offset)
    msg.debug("model.info.joint_effort_cts_offset: %s", model.info.joint_effort_cts_offset)
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug("model.info.total_cts_offset: %s", model.info.total_cts_offset)
    msg.debug("model.info.joint_dynamic_cts_group_offset: %s", model.info.joint_dynamic_cts_group_offset)
    msg.debug("model.info.joint_kinematic_cts_group_offset: %s", model.info.joint_kinematic_cts_group_offset)
    msg.debug("model.info.joint_bounded_cts_group_offset: %s", model.info.joint_bounded_cts_group_offset)
    msg.debug("model.info.joint_friction_cts_group_offset: %s", model.info.joint_friction_cts_group_offset)
    msg.debug("model.info.joint_effort_cts_group_offset: %s", model.info.joint_effort_cts_group_offset)


def debug_print_model_constraint_info(model: ModelKamino):
    """Print constraint counts and per-world sizes from ``model.info`` in the debug log."""
    if not msg.is_enabled_for(msg.LogLevel.DEBUG):
        return
    msg.debug("Model Constraint Info:")
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug("model.info.max_limits: %s", model.info.max_limits)
    msg.debug("model.info.max_contacts: %s", model.info.max_contacts)
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug("model.info.num_joint_bilateral_cts: %s", model.info.num_joint_bilateral_cts)
    msg.debug("model.info.num_joint_dynamic_cts: %s", model.info.num_joint_dynamic_cts)
    msg.debug("model.info.num_joint_kinematic_cts: %s", model.info.num_joint_kinematic_cts)
    msg.debug("model.info.num_joint_bounded_cts: %s", model.info.num_joint_bounded_cts)
    msg.debug("model.info.num_joint_friction_cts: %s", model.info.num_joint_friction_cts)
    msg.debug("model.info.num_joint_effort_cts: %s", model.info.num_joint_effort_cts)
    msg.debug("model.info.max_limit_cts: %s", model.info.max_limit_cts)
    msg.debug("model.info.max_contact_cts: %s", model.info.max_contact_cts)
    msg.debug("model.info.max_total_cts: %s", model.info.max_total_cts)
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug("model.info.limits_offset: %s", model.info.limits_offset)
    msg.debug("model.info.contacts_offset: %s", model.info.contacts_offset)
    msg.debug("model.info.inequalities_offset: %s", model.info.inequalities_offset)
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug("model.info.body_dofs_offset: %s", model.info.body_dofs_offset)
    msg.debug("model.info.joint_coords_offset: %s", model.info.joint_coords_offset)
    msg.debug("model.info.joint_dofs_offset: %s", model.info.joint_dofs_offset)
    msg.debug("model.info.joint_passive_coords_offset: %s", model.info.joint_passive_coords_offset)
    msg.debug("model.info.joint_passive_dofs_offset: %s", model.info.joint_passive_dofs_offset)
    msg.debug("model.info.joint_actuated_coords_offset: %s", model.info.joint_actuated_coords_offset)
    msg.debug("model.info.joint_actuated_dofs_offset: %s", model.info.joint_actuated_dofs_offset)
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug("model.info.joint_bilateral_cts_offset: %s", model.info.joint_bilateral_cts_offset)
    msg.debug("model.info.joint_dynamic_cts_offset: %s", model.info.joint_dynamic_cts_offset)
    msg.debug("model.info.joint_kinematic_cts_offset: %s", model.info.joint_kinematic_cts_offset)
    msg.debug("model.info.joint_bounded_cts_offset: %s", model.info.joint_bounded_cts_offset)
    msg.debug("model.info.joint_friction_cts_offset: %s", model.info.joint_friction_cts_offset)
    msg.debug("model.info.joint_effort_cts_offset: %s", model.info.joint_effort_cts_offset)
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug("model.info.total_cts_offset: %s", model.info.total_cts_offset)
    msg.debug("model.info.joint_dynamic_cts_group_offset: %s", model.info.joint_dynamic_cts_group_offset)
    msg.debug("model.info.joint_kinematic_cts_group_offset: %s", model.info.joint_kinematic_cts_group_offset)
    msg.debug("model.info.joint_bounded_cts_group_offset: %s", model.info.joint_bounded_cts_group_offset)
    msg.debug("model.info.joint_friction_cts_group_offset: %s", model.info.joint_friction_cts_group_offset)
    msg.debug("model.info.joint_effort_cts_group_offset: %s", model.info.joint_effort_cts_group_offset)


def debug_print_model_bodies(model: ModelKamino, inertias=True, initial_states=True):
    """Print ``model.bodies`` (optionally inertias and initial states) in the debug log."""
    if not msg.is_enabled_for(msg.LogLevel.DEBUG):
        return
    msg.debug("model.bodies.num_bodies: %s", model.bodies.num_bodies)
    msg.debug("model.bodies.wid: %s", model.bodies.wid)
    msg.debug("model.bodies.bid: %s", model.bodies.bid)
    if inertias:
        msg.debug("model.bodies.m_i: %s", model.bodies.m_i)
        msg.debug("model.bodies.inv_m_i:\n%s", model.bodies.inv_m_i)
        msg.debug("model.bodies.i_I_i:\n%s", model.bodies.i_I_i)
        msg.debug("model.bodies.inv_i_I_i:\n%s", model.bodies.inv_i_I_i)
    if initial_states:
        msg.debug("model.bodies.q_i_0:\n%s", model.bodies.q_i_0)
        msg.debug("model.bodies.u_i_0:\n%s", model.bodies.u_i_0)


def debug_print_model_joints(
    model: ModelKamino,
    dimensions=True,
    offsets=True,
    parameters=True,
    limits=True,
    dynamics=True,
):
    """Print ``model.joints`` (dims, offsets, parameters, limits, dynamics as requested) in the debug log."""
    if not msg.is_enabled_for(msg.LogLevel.DEBUG):
        return
    msg.debug("model.joints.num_joints: %s", model.joints.num_joints)
    msg.debug("model.joints.wid: %s", model.joints.wid)
    msg.debug("model.joints.jid: %s", model.joints.jid)
    msg.debug("model.joints.dof_type: %s", model.joints.dof_type)
    msg.debug("model.joints.act_type: %s", model.joints.act_type)
    msg.debug("model.joints.num_dynamic_cts: %s", model.joints.num_dynamic_cts)
    msg.debug("model.joints.num_kinematic_cts: %s", model.joints.num_kinematic_cts)
    msg.debug("model.joints.bid_B: %s", model.joints.bid_B)
    msg.debug("model.joints.bid_F: %s", model.joints.bid_F)
    msg.debug("model.joints.B_r_Bj:\n%s", model.joints.B_r_Bj)
    msg.debug("model.joints.F_r_Fj:\n%s", model.joints.F_r_Fj)
    msg.debug("model.joints.X_Bj:\n%s", model.joints.X_Bj)
    msg.debug("model.joints.X_Fj:\n%s", model.joints.X_Fj)
    msg.debug("model.joints.q_j_0: %s", model.joints.q_j_0)
    msg.debug("model.joints.dq_j_0: %s", model.joints.dq_j_0)
    if dimensions:
        msg.debug("model.joints.num_coords: %s", model.joints.num_coords)
        msg.debug("model.joints.num_dofs: %s", model.joints.num_dofs)
        msg.debug("model.joints.num_bilateral_cts: %s", model.joints.num_bilateral_cts)
        msg.debug("model.joints.num_dynamic_cts: %s", model.joints.num_dynamic_cts)
        msg.debug("model.joints.num_kinematic_cts: %s", model.joints.num_kinematic_cts)
        msg.debug("model.joints.num_bounded_cts: %s", model.joints.num_bounded_cts)
        msg.debug("model.joints.num_friction_cts: %s", model.joints.num_friction_cts)
        msg.debug("model.joints.num_effort_cts: %s", model.joints.num_effort_cts)
    if offsets:
        msg.debug("model.joints.coords_offset: %s", model.joints.coords_offset)
        msg.debug("model.joints.dofs_offset: %s", model.joints.dofs_offset)
        msg.debug("model.joints.passive_coords_offset: %s", model.joints.passive_coords_offset)
        msg.debug("model.joints.passive_dofs_offset: %s", model.joints.passive_dofs_offset)
        msg.debug("model.joints.actuated_coords_offset: %s", model.joints.actuated_coords_offset)
        msg.debug("model.joints.actuated_dofs_offset: %s", model.joints.actuated_dofs_offset)
        msg.debug("model.joints.bilateral_cts_offset: %s", model.joints.bilateral_cts_offset)
        msg.debug("model.joints.dynamic_cts_offset: %s", model.joints.dynamic_cts_offset)
        msg.debug("model.joints.kinematic_cts_offset: %s", model.joints.kinematic_cts_offset)
        msg.debug("model.joints.bounded_cts_offset: %s", model.joints.bounded_cts_offset)
        msg.debug("model.joints.friction_cts_offset: %s", model.joints.friction_cts_offset)
        msg.debug("model.joints.effort_cts_offset: %s", model.joints.effort_cts_offset)
        msg.debug("model.joints.dynamic_cts_offset_total_cts: %s", model.joints.dynamic_cts_offset_total_cts)
        msg.debug("model.joints.kinematic_cts_offset_total_cts: %s", model.joints.kinematic_cts_offset_total_cts)
        msg.debug("model.joints.friction_cts_offset_total_cts: %s", model.joints.friction_cts_offset_total_cts)
        msg.debug("model.joints.effort_cts_offset_total_cts: %s", model.joints.effort_cts_offset_total_cts)
    if parameters:
        msg.debug("model.joints.B_r_Bj: %s", model.joints.B_r_Bj)
        msg.debug("model.joints.F_r_Fj: %s", model.joints.F_r_Fj)
        msg.debug("model.joints.X_Bj: %s", model.joints.X_Bj)
        msg.debug("model.joints.X_Fj: %s", model.joints.X_Fj)
    if limits:
        msg.debug("model.joints.q_j_min: %s", model.joints.q_j_min)
        msg.debug("model.joints.q_j_max: %s", model.joints.q_j_max)
        msg.debug("model.joints.dq_j_max: %s", model.joints.dq_j_max)
        msg.debug("model.joints.tau_j_max: %s", model.joints.tau_j_max)
    if dynamics:
        msg.debug("model.joints.a_j: %s", model.joints.a_j)
        msg.debug("model.joints.b_j: %s", model.joints.b_j)
        msg.debug("model.joints.f_j: %s", model.joints.f_j)
        msg.debug("model.joints.k_p_j: %s", model.joints.k_p_j)
        msg.debug("model.joints.k_d_j: %s", model.joints.k_d_j)
        msg.debug("model.joints.dynamic_cts_axis: %s", model.joints.dynamic_cts_axis)
        msg.debug("model.joints.friction_cts_axis: %s", model.joints.friction_cts_axis)
        msg.debug("model.joints.effort_cts_axis: %s", model.joints.effort_cts_axis)


def debug_print_data_info(data: DataKamino):
    """Print ``data.info`` (constraint counts, offsets) in the debug log."""
    if not msg.is_enabled_for(msg.LogLevel.DEBUG):
        return
    msg.debug("===============================================================================")
    msg.debug("data.info.num_limits:  %s", data.info.num_limits)
    msg.debug("data.info.num_contacts:  %s", data.info.num_contacts)
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug("data.info.num_total_cts:  %s", data.info.num_total_cts)
    msg.debug("data.info.num_limit_cts:  %s", data.info.num_limit_cts)
    msg.debug("data.info.num_contact_cts:  %s", data.info.num_contact_cts)
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug("data.info.limit_cts_group_offset:  %s", data.info.limit_cts_group_offset)
    msg.debug("data.info.contact_cts_group_offset:  %s", data.info.contact_cts_group_offset)


def debug_print_data(data: DataKamino, info=True):
    """Print ``data`` (bodies, joints, actuators, and optionally ``data.info``) in the debug log."""
    if not msg.is_enabled_for(msg.LogLevel.DEBUG):
        return
    # Print the state info
    if info:
        debug_print_data_info(data)
    # Print body state data
    msg.debug("data.bodies.I_i: %s", data.bodies.I_i)
    msg.debug("data.bodies.inv_I_i: %s", data.bodies.inv_I_i)
    msg.debug("data.bodies.q_i: %s", data.bodies.q_i)
    msg.debug("data.bodies.u_i: %s", data.bodies.u_i)
    msg.debug("data.bodies.w_i: %s", data.bodies.w_i)
    msg.debug("data.bodies.w_a_i: %s", data.bodies.w_a_i)
    msg.debug("data.bodies.w_j_i: %s", data.bodies.w_j_i)
    msg.debug("data.bodies.w_l_i: %s", data.bodies.w_l_i)
    msg.debug("data.bodies.w_f_i: %s", data.bodies.w_f_i)
    msg.debug("data.bodies.w_c_i: %s", data.bodies.w_c_i)
    msg.debug("data.bodies.w_e_i: %s", data.bodies.w_e_i)
    # Print joint state data
    msg.debug("data.joints.p_j: %s", data.joints.p_j)
    msg.debug("data.joints.q_j: %s", data.joints.q_j)
    msg.debug("data.joints.dq_j: %s", data.joints.dq_j)
    msg.debug("data.joints.tau_j: %s", data.joints.tau_j)
    msg.debug("data.joints.r_j: %s", data.joints.r_j)
    msg.debug("data.joints.dr_j: %s", data.joints.dr_j)
    msg.debug("data.joints.lambda_kin_j: %s", data.joints.lambda_kin_j)
    msg.debug("data.joints.lambda_dyn_j: %s", data.joints.lambda_dyn_j)
    msg.debug("data.joints.lambda_f_j: %s", data.joints.lambda_f_j)
    msg.debug("data.joints.lambda_tau_j: %s", data.joints.lambda_tau_j)
    msg.debug("data.joints.m_j: %s", data.joints.m_j)
    msg.debug("data.joints.inv_m_j: %s", data.joints.inv_m_j)
    msg.debug("data.joints.q_j_ref: %s", data.joints.q_j_ref)
    msg.debug("data.joints.dq_j_ref: %s", data.joints.dq_j_ref)
    msg.debug("data.joints.j_w_j: %s", data.joints.j_w_j)
    msg.debug("data.joints.j_w_a_j: %s", data.joints.j_w_a_j)
    msg.debug("data.joints.j_w_c_j: %s", data.joints.j_w_c_j)
    msg.debug("data.joints.j_w_f_j: %s", data.joints.j_w_f_j)
    msg.debug("data.joints.j_w_l_j: %s", data.joints.j_w_l_j)
    # Print the geometry state data
    msg.debug("data.geoms.pose: %s", data.geoms.pose)


###
# General-Purpose Functions
###


def debug_print_error_stats(name, arr, ref, n, show_errors=False):
    """Print abs/rel error statistics between ``arr`` and ``ref`` in the debug log."""
    if not msg.is_enabled_for(msg.LogLevel.DEBUG):
        return
    err = arr - ref
    err_abs = np.abs(err)
    err_l2 = np.linalg.norm(err)
    err_mean = np.sum(err_abs) / n
    err_max = np.max(err_abs)
    if show_errors:
        msg.debug("%s_err (%s):\n%s", name, err.shape, err)
    msg.debug("%s_err_l2: %s", name, err_l2)
    msg.debug("%s_err_mean: %s", name, err_mean)
    msg.debug("%s_err_max: %s\n\n", name, err_max)
