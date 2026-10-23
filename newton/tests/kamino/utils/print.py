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
    msg.debug(f"model.size.num_worlds: {model.size.num_worlds}")

    # Print the device-side model size data
    msg.debug(f"model.size.sum_of_num_bodies: {model.size.sum_of_num_bodies}")
    msg.debug(f"model.size.max_of_num_bodies: {model.size.max_of_num_bodies}")
    msg.debug(f"model.size.sum_of_num_joints: {model.size.sum_of_num_joints}")
    msg.debug(f"model.size.max_of_num_joints: {model.size.max_of_num_joints}")
    msg.debug(f"model.size.sum_of_num_material_pairs: {model.size.sum_of_num_material_pairs}")
    msg.debug(f"model.size.max_of_num_material_pairs: {model.size.max_of_num_material_pairs}")
    msg.debug(f"model.size.sum_of_num_body_dofs: {model.size.sum_of_num_body_dofs}")
    msg.debug(f"model.size.max_of_num_body_dofs: {model.size.max_of_num_body_dofs}")
    msg.debug(f"model.size.sum_of_num_joint_dofs: {model.size.sum_of_num_joint_dofs}")
    msg.debug(f"model.size.max_of_num_joint_dofs: {model.size.max_of_num_joint_dofs}")
    msg.debug(f"model.size.sum_of_max_inequalities: {model.size.sum_of_max_inequalities}")
    msg.debug(f"model.size.max_of_max_inequalities: {model.size.max_of_max_inequalities}")
    msg.debug(f"model.size.sum_of_num_bounded_joint_cts: {model.size.sum_of_num_bounded_joint_cts}")
    msg.debug(f"model.size.max_of_num_bounded_joint_cts: {model.size.max_of_num_bounded_joint_cts}")
    msg.debug(f"model.size.sum_of_num_friction_joint_cts: {model.size.sum_of_num_friction_joint_cts}")
    msg.debug(f"model.size.max_of_num_friction_joint_cts: {model.size.max_of_num_friction_joint_cts}")
    msg.debug(f"model.size.sum_of_num_effort_joint_cts: {model.size.sum_of_num_effort_joint_cts}")
    msg.debug(f"model.size.max_of_num_effort_joint_cts: {model.size.max_of_num_effort_joint_cts}")
    msg.debug(f"model.size.sum_of_max_total_cts: {model.size.sum_of_max_total_cts}")
    msg.debug(f"model.size.max_of_max_total_cts: {model.size.max_of_max_total_cts}")


def debug_print_model_info(model: ModelKamino):
    """Print ``model.info`` and all per-world count arrays in the debug log."""
    if not msg.is_enabled_for(msg.LogLevel.DEBUG):
        return
    msg.debug("===============================================================================")
    msg.debug("Model Info:")
    # Print the host-side model info meta-data
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug(f"model.info.num_worlds: {model.info.num_worlds}")
    # Print the device-side model info data
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug(f"model.info.num_bodies: {model.info.num_bodies}")
    msg.debug(f"model.info.num_joints: {model.info.num_joints}")
    msg.debug(f"model.info.num_passive_joints: {model.info.num_passive_joints}")
    msg.debug(f"model.info.num_actuated_joints: {model.info.num_actuated_joints}")
    msg.debug(f"model.info.num_dynamic_joints: {model.info.num_dynamic_joints}")
    msg.debug(f"model.info.num_geoms: {model.info.num_geoms}")
    msg.debug(f"model.info.max_limits: {model.info.max_limits}")
    msg.debug(f"model.info.max_contacts: {model.info.max_contacts}")
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug(f"model.info.num_body_dofs: {model.info.num_body_dofs}")
    msg.debug(f"model.info.num_joint_coords: {model.info.num_joint_coords}")
    msg.debug(f"model.info.num_joint_dofs: {model.info.num_joint_dofs}")
    msg.debug(f"model.info.num_passive_joint_coords: {model.info.num_passive_joint_coords}")
    msg.debug(f"model.info.num_passive_joint_dofs: {model.info.num_passive_joint_dofs}")
    msg.debug(f"model.info.num_actuated_joint_coords: {model.info.num_actuated_joint_coords}")
    msg.debug(f"model.info.num_actuated_joint_dofs: {model.info.num_actuated_joint_dofs}")
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug(f"model.info.num_joint_bilateral_cts: {model.info.num_joint_bilateral_cts}")
    msg.debug(f"model.info.num_joint_dynamic_cts: {model.info.num_joint_dynamic_cts}")
    msg.debug(f"model.info.num_joint_kinematic_cts: {model.info.num_joint_kinematic_cts}")
    msg.debug(f"model.info.num_joint_bounded_cts: {model.info.num_joint_bounded_cts}")
    msg.debug(f"model.info.num_joint_friction_cts: {model.info.num_joint_friction_cts}")
    msg.debug(f"model.info.num_joint_effort_cts: {model.info.num_joint_effort_cts}")
    msg.debug(f"model.info.max_limit_cts: {model.info.max_limit_cts}")
    msg.debug(f"model.info.max_contact_cts: {model.info.max_contact_cts}")
    msg.debug(f"model.info.max_total_cts: {model.info.max_total_cts}")
    # Print the element offsets
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug(f"model.info.bodies_offset: {model.info.bodies_offset}")
    msg.debug(f"model.info.joints_offset: {model.info.joints_offset}")
    msg.debug(f"model.info.limits_offset: {model.info.limits_offset}")
    msg.debug(f"model.info.contacts_offset: {model.info.contacts_offset}")
    msg.debug(f"model.info.inequalities_offset: {model.info.inequalities_offset}")
    # Print the coords, DoFs and constraint offsets
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug(f"model.info.body_dofs_offset: {model.info.body_dofs_offset}")
    msg.debug(f"model.info.joint_coords_offset: {model.info.joint_coords_offset}")
    msg.debug(f"model.info.joint_dofs_offset: {model.info.joint_dofs_offset}")
    msg.debug(f"model.info.joint_passive_coords_offset: {model.info.joint_passive_coords_offset}")
    msg.debug(f"model.info.joint_passive_dofs_offset: {model.info.joint_passive_dofs_offset}")
    msg.debug(f"model.info.joint_actuated_coords_offset: {model.info.joint_actuated_coords_offset}")
    msg.debug(f"model.info.joint_actuated_dofs_offset: {model.info.joint_actuated_dofs_offset}")
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug(f"model.info.joint_bilateral_cts_offset: {model.info.joint_bilateral_cts_offset}")
    msg.debug(f"model.info.joint_dynamic_cts_offset: {model.info.joint_dynamic_cts_offset}")
    msg.debug(f"model.info.joint_kinematic_cts_offset: {model.info.joint_kinematic_cts_offset}")
    msg.debug(f"model.info.joint_bounded_cts_offset: {model.info.joint_bounded_cts_offset}")
    msg.debug(f"model.info.joint_friction_cts_offset: {model.info.joint_friction_cts_offset}")
    msg.debug(f"model.info.joint_effort_cts_offset: {model.info.joint_effort_cts_offset}")
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug(f"model.info.total_cts_offset: {model.info.total_cts_offset}")
    msg.debug(f"model.info.joint_dynamic_cts_group_offset: {model.info.joint_dynamic_cts_group_offset}")
    msg.debug(f"model.info.joint_kinematic_cts_group_offset: {model.info.joint_kinematic_cts_group_offset}")
    msg.debug(f"model.info.joint_bounded_cts_group_offset: {model.info.joint_bounded_cts_group_offset}")
    msg.debug(f"model.info.joint_friction_cts_group_offset: {model.info.joint_friction_cts_group_offset}")
    msg.debug(f"model.info.joint_effort_cts_group_offset: {model.info.joint_effort_cts_group_offset}")


def debug_print_model_constraint_info(model: ModelKamino):
    """Print constraint counts and per-world sizes from ``model.info`` in the debug log."""
    if not msg.is_enabled_for(msg.LogLevel.DEBUG):
        return
    msg.debug("Model Constraint Info:")
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug(f"model.info.max_limits: {model.info.max_limits}")
    msg.debug(f"model.info.max_contacts: {model.info.max_contacts}")
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug(f"model.info.num_joint_bilateral_cts: {model.info.num_joint_bilateral_cts}")
    msg.debug(f"model.info.num_joint_dynamic_cts: {model.info.num_joint_dynamic_cts}")
    msg.debug(f"model.info.num_joint_kinematic_cts: {model.info.num_joint_kinematic_cts}")
    msg.debug(f"model.info.num_joint_bounded_cts: {model.info.num_joint_bounded_cts}")
    msg.debug(f"model.info.num_joint_friction_cts: {model.info.num_joint_friction_cts}")
    msg.debug(f"model.info.num_joint_effort_cts: {model.info.num_joint_effort_cts}")
    msg.debug(f"model.info.max_limit_cts: {model.info.max_limit_cts}")
    msg.debug(f"model.info.max_contact_cts: {model.info.max_contact_cts}")
    msg.debug(f"model.info.max_total_cts: {model.info.max_total_cts}")
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug(f"model.info.limits_offset: {model.info.limits_offset}")
    msg.debug(f"model.info.contacts_offset: {model.info.contacts_offset}")
    msg.debug(f"model.info.inequalities_offset: {model.info.inequalities_offset}")
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug(f"model.info.body_dofs_offset: {model.info.body_dofs_offset}")
    msg.debug(f"model.info.joint_coords_offset: {model.info.joint_coords_offset}")
    msg.debug(f"model.info.joint_dofs_offset: {model.info.joint_dofs_offset}")
    msg.debug(f"model.info.joint_passive_coords_offset: {model.info.joint_passive_coords_offset}")
    msg.debug(f"model.info.joint_passive_dofs_offset: {model.info.joint_passive_dofs_offset}")
    msg.debug(f"model.info.joint_actuated_coords_offset: {model.info.joint_actuated_coords_offset}")
    msg.debug(f"model.info.joint_actuated_dofs_offset: {model.info.joint_actuated_dofs_offset}")
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug(f"model.info.joint_bilateral_cts_offset: {model.info.joint_bilateral_cts_offset}")
    msg.debug(f"model.info.joint_dynamic_cts_offset: {model.info.joint_dynamic_cts_offset}")
    msg.debug(f"model.info.joint_kinematic_cts_offset: {model.info.joint_kinematic_cts_offset}")
    msg.debug(f"model.info.joint_bounded_cts_offset: {model.info.joint_bounded_cts_offset}")
    msg.debug(f"model.info.joint_friction_cts_offset: {model.info.joint_friction_cts_offset}")
    msg.debug(f"model.info.joint_effort_cts_offset: {model.info.joint_effort_cts_offset}")
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug(f"model.info.total_cts_offset: {model.info.total_cts_offset}")
    msg.debug(f"model.info.joint_dynamic_cts_group_offset: {model.info.joint_dynamic_cts_group_offset}")
    msg.debug(f"model.info.joint_kinematic_cts_group_offset: {model.info.joint_kinematic_cts_group_offset}")
    msg.debug(f"model.info.joint_bounded_cts_group_offset: {model.info.joint_bounded_cts_group_offset}")
    msg.debug(f"model.info.joint_friction_cts_group_offset: {model.info.joint_friction_cts_group_offset}")
    msg.debug(f"model.info.joint_effort_cts_group_offset: {model.info.joint_effort_cts_group_offset}")


def debug_print_model_bodies(model: ModelKamino, inertias=True, initial_states=True):
    """Print ``model.bodies`` (optionally inertias and initial states) in the debug log."""
    if not msg.is_enabled_for(msg.LogLevel.DEBUG):
        return
    msg.debug(f"model.bodies.num_bodies: {model.bodies.num_bodies}")
    msg.debug(f"model.bodies.wid: {model.bodies.wid}")
    msg.debug(f"model.bodies.bid: {model.bodies.bid}")
    if inertias:
        msg.debug(f"model.bodies.m_i: {model.bodies.m_i}")
        msg.debug(f"model.bodies.inv_m_i:\n{model.bodies.inv_m_i}")
        msg.debug(f"model.bodies.i_I_i:\n{model.bodies.i_I_i}")
        msg.debug(f"model.bodies.inv_i_I_i:\n{model.bodies.inv_i_I_i}")
    if initial_states:
        msg.debug(f"model.bodies.q_i_0:\n{model.bodies.q_i_0}")
        msg.debug(f"model.bodies.u_i_0:\n{model.bodies.u_i_0}")


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
    msg.debug(f"model.joints.num_joints: {model.joints.num_joints}")
    msg.debug(f"model.joints.wid: {model.joints.wid}")
    msg.debug(f"model.joints.jid: {model.joints.jid}")
    msg.debug(f"model.joints.dof_type: {model.joints.dof_type}")
    msg.debug(f"model.joints.act_type: {model.joints.act_type}")
    msg.debug(f"model.joints.num_dynamic_cts: {model.joints.num_dynamic_cts}")
    msg.debug(f"model.joints.num_kinematic_cts: {model.joints.num_kinematic_cts}")
    msg.debug(f"model.joints.bid_B: {model.joints.bid_B}")
    msg.debug(f"model.joints.bid_F: {model.joints.bid_F}")
    msg.debug(f"model.joints.B_r_Bj:\n{model.joints.B_r_Bj}")
    msg.debug(f"model.joints.F_r_Fj:\n{model.joints.F_r_Fj}")
    msg.debug(f"model.joints.X_Bj:\n{model.joints.X_Bj}")
    msg.debug(f"model.joints.X_Fj:\n{model.joints.X_Fj}")
    msg.debug(f"model.joints.q_j_0: {model.joints.q_j_0}")
    msg.debug(f"model.joints.dq_j_0: {model.joints.dq_j_0}")
    if dimensions:
        msg.debug(f"model.joints.num_coords: {model.joints.num_coords}")
        msg.debug(f"model.joints.num_dofs: {model.joints.num_dofs}")
        msg.debug(f"model.joints.num_bilateral_cts: {model.joints.num_bilateral_cts}")
        msg.debug(f"model.joints.num_dynamic_cts: {model.joints.num_dynamic_cts}")
        msg.debug(f"model.joints.num_kinematic_cts: {model.joints.num_kinematic_cts}")
        msg.debug(f"model.joints.num_bounded_cts: {model.joints.num_bounded_cts}")
        msg.debug(f"model.joints.num_friction_cts: {model.joints.num_friction_cts}")
        msg.debug(f"model.joints.num_effort_cts: {model.joints.num_effort_cts}")
    if offsets:
        msg.debug(f"model.joints.coords_offset: {model.joints.coords_offset}")
        msg.debug(f"model.joints.dofs_offset: {model.joints.dofs_offset}")
        msg.debug(f"model.joints.passive_coords_offset: {model.joints.passive_coords_offset}")
        msg.debug(f"model.joints.passive_dofs_offset: {model.joints.passive_dofs_offset}")
        msg.debug(f"model.joints.actuated_coords_offset: {model.joints.actuated_coords_offset}")
        msg.debug(f"model.joints.actuated_dofs_offset: {model.joints.actuated_dofs_offset}")
        msg.debug(f"model.joints.bilateral_cts_offset: {model.joints.bilateral_cts_offset}")
        msg.debug(f"model.joints.dynamic_cts_offset: {model.joints.dynamic_cts_offset}")
        msg.debug(f"model.joints.kinematic_cts_offset: {model.joints.kinematic_cts_offset}")
        msg.debug(f"model.joints.bounded_cts_offset: {model.joints.bounded_cts_offset}")
        msg.debug(f"model.joints.friction_cts_offset: {model.joints.friction_cts_offset}")
        msg.debug(f"model.joints.effort_cts_offset: {model.joints.effort_cts_offset}")
        msg.debug(f"model.joints.dynamic_cts_offset_total_cts: {model.joints.dynamic_cts_offset_total_cts}")
        msg.debug(f"model.joints.kinematic_cts_offset_total_cts: {model.joints.kinematic_cts_offset_total_cts}")
        msg.debug(f"model.joints.friction_cts_offset_total_cts: {model.joints.friction_cts_offset_total_cts}")
        msg.debug(f"model.joints.effort_cts_offset_total_cts: {model.joints.effort_cts_offset_total_cts}")
    if parameters:
        msg.debug(f"model.joints.B_r_Bj: {model.joints.B_r_Bj}")
        msg.debug(f"model.joints.F_r_Fj: {model.joints.F_r_Fj}")
        msg.debug(f"model.joints.X_Bj: {model.joints.X_Bj}")
        msg.debug(f"model.joints.X_Fj: {model.joints.X_Fj}")
    if limits:
        msg.debug(f"model.joints.q_j_min: {model.joints.q_j_min}")
        msg.debug(f"model.joints.q_j_max: {model.joints.q_j_max}")
        msg.debug(f"model.joints.dq_j_max: {model.joints.dq_j_max}")
        msg.debug(f"model.joints.tau_j_max: {model.joints.tau_j_max}")
    if dynamics:
        msg.debug(f"model.joints.a_j: {model.joints.a_j}")
        msg.debug(f"model.joints.b_j: {model.joints.b_j}")
        msg.debug(f"model.joints.f_j: {model.joints.f_j}")
        msg.debug(f"model.joints.k_p_j: {model.joints.k_p_j}")
        msg.debug(f"model.joints.k_d_j: {model.joints.k_d_j}")
        msg.debug(f"model.joints.dynamic_cts_axis: {model.joints.dynamic_cts_axis}")
        msg.debug(f"model.joints.friction_cts_axis: {model.joints.friction_cts_axis}")
        msg.debug(f"model.joints.effort_cts_axis: {model.joints.effort_cts_axis}")


def debug_print_data_info(data: DataKamino):
    """Print ``data.info`` (constraint counts, offsets) in the debug log."""
    if not msg.is_enabled_for(msg.LogLevel.DEBUG):
        return
    msg.debug("===============================================================================")
    msg.debug(f"data.info.num_limits:  {data.info.num_limits}")
    msg.debug(f"data.info.num_contacts:  {data.info.num_contacts}")
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug(f"data.info.num_total_cts:  {data.info.num_total_cts}")
    msg.debug(f"data.info.num_limit_cts:  {data.info.num_limit_cts}")
    msg.debug(f"data.info.num_contact_cts:  {data.info.num_contact_cts}")
    msg.debug("-------------------------------------------------------------------------------")
    msg.debug(f"data.info.limit_cts_group_offset:  {data.info.limit_cts_group_offset}")
    msg.debug(f"data.info.contact_cts_group_offset:  {data.info.contact_cts_group_offset}")


def debug_print_data(data: DataKamino, info=True):
    """Print ``data`` (bodies, joints, actuators, and optionally ``data.info``) in the debug log."""
    if not msg.is_enabled_for(msg.LogLevel.DEBUG):
        return
    # Print the state info
    if info:
        debug_print_data_info(data)
    # Print body state data
    msg.debug(f"data.bodies.I_i: {data.bodies.I_i}")
    msg.debug(f"data.bodies.inv_I_i: {data.bodies.inv_I_i}")
    msg.debug(f"data.bodies.q_i: {data.bodies.q_i}")
    msg.debug(f"data.bodies.u_i: {data.bodies.u_i}")
    msg.debug(f"data.bodies.w_i: {data.bodies.w_i}")
    msg.debug(f"data.bodies.w_a_i: {data.bodies.w_a_i}")
    msg.debug(f"data.bodies.w_j_i: {data.bodies.w_j_i}")
    msg.debug(f"data.bodies.w_l_i: {data.bodies.w_l_i}")
    msg.debug(f"data.bodies.w_f_i: {data.bodies.w_f_i}")
    msg.debug(f"data.bodies.w_c_i: {data.bodies.w_c_i}")
    msg.debug(f"data.bodies.w_e_i: {data.bodies.w_e_i}")
    # Print joint state data
    msg.debug(f"data.joints.p_j: {data.joints.p_j}")
    msg.debug(f"data.joints.q_j: {data.joints.q_j}")
    msg.debug(f"data.joints.dq_j: {data.joints.dq_j}")
    msg.debug(f"data.joints.tau_j: {data.joints.tau_j}")
    msg.debug(f"data.joints.r_j: {data.joints.r_j}")
    msg.debug(f"data.joints.dr_j: {data.joints.dr_j}")
    msg.debug(f"data.joints.lambda_kin_j: {data.joints.lambda_kin_j}")
    msg.debug(f"data.joints.lambda_dyn_j: {data.joints.lambda_dyn_j}")
    msg.debug(f"data.joints.lambda_f_j: {data.joints.lambda_f_j}")
    msg.debug(f"data.joints.lambda_tau_j: {data.joints.lambda_tau_j}")
    msg.debug(f"data.joints.m_j: {data.joints.m_j}")
    msg.debug(f"data.joints.inv_m_j: {data.joints.inv_m_j}")
    msg.debug(f"data.joints.q_j_ref: {data.joints.q_j_ref}")
    msg.debug(f"data.joints.dq_j_ref: {data.joints.dq_j_ref}")
    msg.debug(f"data.joints.j_w_j: {data.joints.j_w_j}")
    msg.debug(f"data.joints.j_w_a_j: {data.joints.j_w_a_j}")
    msg.debug(f"data.joints.j_w_c_j: {data.joints.j_w_c_j}")
    msg.debug(f"data.joints.j_w_f_j: {data.joints.j_w_f_j}")
    msg.debug(f"data.joints.j_w_l_j: {data.joints.j_w_l_j}")
    # Print the geometry state data
    msg.debug(f"data.geoms.pose: {data.geoms.pose}")


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
        msg.debug(f"{name}_err ({err.shape}):\n{err}")
    msg.debug(f"{name}_err_l2: {err_l2}")
    msg.debug(f"{name}_err_mean: {err_mean}")
    msg.debug(f"{name}_err_max: {err_max}\n\n")
