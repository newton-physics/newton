# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: UNIT TESTS: DEBUG-PRINT HELPERS
"""

import logging

import numpy as np

from newton._src.solvers.kamino._src.core.data import DataKamino
from newton._src.solvers.kamino._src.core.model import ModelKamino

logger = logging.getLogger(__name__)

###
# Model Functions
###


def debug_print_model_size(model: ModelKamino):
    """Print host- and device-side ``model.size`` fields in the debug log."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug("Model Size:")

    # Print the host-side model size meta-data
    logger.debug("model.size.num_worlds: %s", model.size.num_worlds)

    # Print the device-side model size data
    logger.debug("model.size.sum_of_num_bodies: %s", model.size.sum_of_num_bodies)
    logger.debug("model.size.max_of_num_bodies: %s", model.size.max_of_num_bodies)
    logger.debug("model.size.sum_of_num_joints: %s", model.size.sum_of_num_joints)
    logger.debug("model.size.max_of_num_joints: %s", model.size.max_of_num_joints)
    logger.debug("model.size.sum_of_num_material_pairs: %s", model.size.sum_of_num_material_pairs)
    logger.debug("model.size.max_of_num_material_pairs: %s", model.size.max_of_num_material_pairs)
    logger.debug("model.size.sum_of_num_body_dofs: %s", model.size.sum_of_num_body_dofs)
    logger.debug("model.size.max_of_num_body_dofs: %s", model.size.max_of_num_body_dofs)
    logger.debug("model.size.sum_of_num_joint_dofs: %s", model.size.sum_of_num_joint_dofs)
    logger.debug("model.size.max_of_num_joint_dofs: %s", model.size.max_of_num_joint_dofs)
    logger.debug("model.size.sum_of_max_inequalities: %s", model.size.sum_of_max_inequalities)
    logger.debug("model.size.max_of_max_inequalities: %s", model.size.max_of_max_inequalities)
    logger.debug("model.size.sum_of_num_bounded_joint_cts: %s", model.size.sum_of_num_bounded_joint_cts)
    logger.debug("model.size.max_of_num_bounded_joint_cts: %s", model.size.max_of_num_bounded_joint_cts)
    logger.debug("model.size.sum_of_num_friction_joint_cts: %s", model.size.sum_of_num_friction_joint_cts)
    logger.debug("model.size.max_of_num_friction_joint_cts: %s", model.size.max_of_num_friction_joint_cts)
    logger.debug("model.size.sum_of_num_effort_joint_cts: %s", model.size.sum_of_num_effort_joint_cts)
    logger.debug("model.size.max_of_num_effort_joint_cts: %s", model.size.max_of_num_effort_joint_cts)
    logger.debug("model.size.sum_of_max_total_cts: %s", model.size.sum_of_max_total_cts)
    logger.debug("model.size.max_of_max_total_cts: %s", model.size.max_of_max_total_cts)


def debug_print_model_info(model: ModelKamino):
    """Print ``model.info`` and all per-world count arrays in the debug log."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug("===============================================================================")
    logger.debug("Model Info:")
    # Print the host-side model info meta-data
    logger.debug("-------------------------------------------------------------------------------")
    logger.debug("model.info.num_worlds: %s", model.info.num_worlds)
    # Print the device-side model info data
    logger.debug("-------------------------------------------------------------------------------")
    logger.debug("model.info.num_bodies: %s", model.info.num_bodies)
    logger.debug("model.info.num_joints: %s", model.info.num_joints)
    logger.debug("model.info.num_passive_joints: %s", model.info.num_passive_joints)
    logger.debug("model.info.num_actuated_joints: %s", model.info.num_actuated_joints)
    logger.debug("model.info.num_dynamic_joints: %s", model.info.num_dynamic_joints)
    logger.debug("model.info.num_geoms: %s", model.info.num_geoms)
    logger.debug("model.info.max_limits: %s", model.info.max_limits)
    logger.debug("model.info.max_contacts: %s", model.info.max_contacts)
    logger.debug("-------------------------------------------------------------------------------")
    logger.debug("model.info.num_body_dofs: %s", model.info.num_body_dofs)
    logger.debug("model.info.num_joint_coords: %s", model.info.num_joint_coords)
    logger.debug("model.info.num_joint_dofs: %s", model.info.num_joint_dofs)
    logger.debug("model.info.num_passive_joint_coords: %s", model.info.num_passive_joint_coords)
    logger.debug("model.info.num_passive_joint_dofs: %s", model.info.num_passive_joint_dofs)
    logger.debug("model.info.num_actuated_joint_coords: %s", model.info.num_actuated_joint_coords)
    logger.debug("model.info.num_actuated_joint_dofs: %s", model.info.num_actuated_joint_dofs)
    logger.debug("-------------------------------------------------------------------------------")
    logger.debug("model.info.num_joint_bilateral_cts: %s", model.info.num_joint_bilateral_cts)
    logger.debug("model.info.num_joint_dynamic_cts: %s", model.info.num_joint_dynamic_cts)
    logger.debug("model.info.num_joint_kinematic_cts: %s", model.info.num_joint_kinematic_cts)
    logger.debug("model.info.num_joint_bounded_cts: %s", model.info.num_joint_bounded_cts)
    logger.debug("model.info.num_joint_friction_cts: %s", model.info.num_joint_friction_cts)
    logger.debug("model.info.num_joint_effort_cts: %s", model.info.num_joint_effort_cts)
    logger.debug("model.info.max_limit_cts: %s", model.info.max_limit_cts)
    logger.debug("model.info.max_contact_cts: %s", model.info.max_contact_cts)
    logger.debug("model.info.max_total_cts: %s", model.info.max_total_cts)
    # Print the element offsets
    logger.debug("-------------------------------------------------------------------------------")
    logger.debug("model.info.bodies_offset: %s", model.info.bodies_offset)
    logger.debug("model.info.joints_offset: %s", model.info.joints_offset)
    logger.debug("model.info.limits_offset: %s", model.info.limits_offset)
    logger.debug("model.info.contacts_offset: %s", model.info.contacts_offset)
    logger.debug("model.info.inequalities_offset: %s", model.info.inequalities_offset)
    # Print the coords, DoFs and constraint offsets
    logger.debug("-------------------------------------------------------------------------------")
    logger.debug("model.info.body_dofs_offset: %s", model.info.body_dofs_offset)
    logger.debug("model.info.joint_coords_offset: %s", model.info.joint_coords_offset)
    logger.debug("model.info.joint_dofs_offset: %s", model.info.joint_dofs_offset)
    logger.debug("model.info.joint_passive_coords_offset: %s", model.info.joint_passive_coords_offset)
    logger.debug("model.info.joint_passive_dofs_offset: %s", model.info.joint_passive_dofs_offset)
    logger.debug("model.info.joint_actuated_coords_offset: %s", model.info.joint_actuated_coords_offset)
    logger.debug("model.info.joint_actuated_dofs_offset: %s", model.info.joint_actuated_dofs_offset)
    logger.debug("-------------------------------------------------------------------------------")
    logger.debug("model.info.joint_bilateral_cts_offset: %s", model.info.joint_bilateral_cts_offset)
    logger.debug("model.info.joint_dynamic_cts_offset: %s", model.info.joint_dynamic_cts_offset)
    logger.debug("model.info.joint_kinematic_cts_offset: %s", model.info.joint_kinematic_cts_offset)
    logger.debug("model.info.joint_bounded_cts_offset: %s", model.info.joint_bounded_cts_offset)
    logger.debug("model.info.joint_friction_cts_offset: %s", model.info.joint_friction_cts_offset)
    logger.debug("model.info.joint_effort_cts_offset: %s", model.info.joint_effort_cts_offset)
    logger.debug("-------------------------------------------------------------------------------")
    logger.debug("model.info.total_cts_offset: %s", model.info.total_cts_offset)
    logger.debug("model.info.joint_dynamic_cts_group_offset: %s", model.info.joint_dynamic_cts_group_offset)
    logger.debug("model.info.joint_kinematic_cts_group_offset: %s", model.info.joint_kinematic_cts_group_offset)
    logger.debug("model.info.joint_bounded_cts_group_offset: %s", model.info.joint_bounded_cts_group_offset)
    logger.debug("model.info.joint_friction_cts_group_offset: %s", model.info.joint_friction_cts_group_offset)
    logger.debug("model.info.joint_effort_cts_group_offset: %s", model.info.joint_effort_cts_group_offset)


def debug_print_model_constraint_info(model: ModelKamino):
    """Print constraint counts and per-world sizes from ``model.info`` in the debug log."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug("Model Constraint Info:")
    logger.debug("-------------------------------------------------------------------------------")
    logger.debug("model.info.max_limits: %s", model.info.max_limits)
    logger.debug("model.info.max_contacts: %s", model.info.max_contacts)
    logger.debug("-------------------------------------------------------------------------------")
    logger.debug("model.info.num_joint_bilateral_cts: %s", model.info.num_joint_bilateral_cts)
    logger.debug("model.info.num_joint_dynamic_cts: %s", model.info.num_joint_dynamic_cts)
    logger.debug("model.info.num_joint_kinematic_cts: %s", model.info.num_joint_kinematic_cts)
    logger.debug("model.info.num_joint_bounded_cts: %s", model.info.num_joint_bounded_cts)
    logger.debug("model.info.num_joint_friction_cts: %s", model.info.num_joint_friction_cts)
    logger.debug("model.info.num_joint_effort_cts: %s", model.info.num_joint_effort_cts)
    logger.debug("model.info.max_limit_cts: %s", model.info.max_limit_cts)
    logger.debug("model.info.max_contact_cts: %s", model.info.max_contact_cts)
    logger.debug("model.info.max_total_cts: %s", model.info.max_total_cts)
    logger.debug("-------------------------------------------------------------------------------")
    logger.debug("model.info.limits_offset: %s", model.info.limits_offset)
    logger.debug("model.info.contacts_offset: %s", model.info.contacts_offset)
    logger.debug("model.info.inequalities_offset: %s", model.info.inequalities_offset)
    logger.debug("-------------------------------------------------------------------------------")
    logger.debug("model.info.body_dofs_offset: %s", model.info.body_dofs_offset)
    logger.debug("model.info.joint_coords_offset: %s", model.info.joint_coords_offset)
    logger.debug("model.info.joint_dofs_offset: %s", model.info.joint_dofs_offset)
    logger.debug("model.info.joint_passive_coords_offset: %s", model.info.joint_passive_coords_offset)
    logger.debug("model.info.joint_passive_dofs_offset: %s", model.info.joint_passive_dofs_offset)
    logger.debug("model.info.joint_actuated_coords_offset: %s", model.info.joint_actuated_coords_offset)
    logger.debug("model.info.joint_actuated_dofs_offset: %s", model.info.joint_actuated_dofs_offset)
    logger.debug("-------------------------------------------------------------------------------")
    logger.debug("model.info.joint_bilateral_cts_offset: %s", model.info.joint_bilateral_cts_offset)
    logger.debug("model.info.joint_dynamic_cts_offset: %s", model.info.joint_dynamic_cts_offset)
    logger.debug("model.info.joint_kinematic_cts_offset: %s", model.info.joint_kinematic_cts_offset)
    logger.debug("model.info.joint_bounded_cts_offset: %s", model.info.joint_bounded_cts_offset)
    logger.debug("model.info.joint_friction_cts_offset: %s", model.info.joint_friction_cts_offset)
    logger.debug("model.info.joint_effort_cts_offset: %s", model.info.joint_effort_cts_offset)
    logger.debug("-------------------------------------------------------------------------------")
    logger.debug("model.info.total_cts_offset: %s", model.info.total_cts_offset)
    logger.debug("model.info.joint_dynamic_cts_group_offset: %s", model.info.joint_dynamic_cts_group_offset)
    logger.debug("model.info.joint_kinematic_cts_group_offset: %s", model.info.joint_kinematic_cts_group_offset)
    logger.debug("model.info.joint_bounded_cts_group_offset: %s", model.info.joint_bounded_cts_group_offset)
    logger.debug("model.info.joint_friction_cts_group_offset: %s", model.info.joint_friction_cts_group_offset)
    logger.debug("model.info.joint_effort_cts_group_offset: %s", model.info.joint_effort_cts_group_offset)


def debug_print_model_bodies(model: ModelKamino, inertias=True, initial_states=True):
    """Print ``model.bodies`` (optionally inertias and initial states) in the debug log."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug("model.bodies.num_bodies: %s", model.bodies.num_bodies)
    logger.debug("model.bodies.wid: %s", model.bodies.wid)
    logger.debug("model.bodies.bid: %s", model.bodies.bid)
    if inertias:
        logger.debug("model.bodies.m_i: %s", model.bodies.m_i)
        logger.debug("model.bodies.inv_m_i:\n%s", model.bodies.inv_m_i)
        logger.debug("model.bodies.i_I_i:\n%s", model.bodies.i_I_i)
        logger.debug("model.bodies.inv_i_I_i:\n%s", model.bodies.inv_i_I_i)
    if initial_states:
        logger.debug("model.bodies.q_i_0:\n%s", model.bodies.q_i_0)
        logger.debug("model.bodies.u_i_0:\n%s", model.bodies.u_i_0)


def debug_print_model_joints(
    model: ModelKamino,
    dimensions=True,
    offsets=True,
    parameters=True,
    limits=True,
    dynamics=True,
):
    """Print ``model.joints`` (dims, offsets, parameters, limits, dynamics as requested) in the debug log."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug("model.joints.num_joints: %s", model.joints.num_joints)
    logger.debug("model.joints.wid: %s", model.joints.wid)
    logger.debug("model.joints.jid: %s", model.joints.jid)
    logger.debug("model.joints.dof_type: %s", model.joints.dof_type)
    logger.debug("model.joints.act_type: %s", model.joints.act_type)
    logger.debug("model.joints.num_dynamic_cts: %s", model.joints.num_dynamic_cts)
    logger.debug("model.joints.num_kinematic_cts: %s", model.joints.num_kinematic_cts)
    logger.debug("model.joints.bid_B: %s", model.joints.bid_B)
    logger.debug("model.joints.bid_F: %s", model.joints.bid_F)
    logger.debug("model.joints.B_r_Bj:\n%s", model.joints.B_r_Bj)
    logger.debug("model.joints.F_r_Fj:\n%s", model.joints.F_r_Fj)
    logger.debug("model.joints.X_Bj:\n%s", model.joints.X_Bj)
    logger.debug("model.joints.X_Fj:\n%s", model.joints.X_Fj)
    logger.debug("model.joints.q_j_0: %s", model.joints.q_j_0)
    logger.debug("model.joints.dq_j_0: %s", model.joints.dq_j_0)
    if dimensions:
        logger.debug("model.joints.num_coords: %s", model.joints.num_coords)
        logger.debug("model.joints.num_dofs: %s", model.joints.num_dofs)
        logger.debug("model.joints.num_bilateral_cts: %s", model.joints.num_bilateral_cts)
        logger.debug("model.joints.num_dynamic_cts: %s", model.joints.num_dynamic_cts)
        logger.debug("model.joints.num_kinematic_cts: %s", model.joints.num_kinematic_cts)
        logger.debug("model.joints.num_bounded_cts: %s", model.joints.num_bounded_cts)
        logger.debug("model.joints.num_friction_cts: %s", model.joints.num_friction_cts)
        logger.debug("model.joints.num_effort_cts: %s", model.joints.num_effort_cts)
    if offsets:
        logger.debug("model.joints.coords_offset: %s", model.joints.coords_offset)
        logger.debug("model.joints.dofs_offset: %s", model.joints.dofs_offset)
        logger.debug("model.joints.passive_coords_offset: %s", model.joints.passive_coords_offset)
        logger.debug("model.joints.passive_dofs_offset: %s", model.joints.passive_dofs_offset)
        logger.debug("model.joints.actuated_coords_offset: %s", model.joints.actuated_coords_offset)
        logger.debug("model.joints.actuated_dofs_offset: %s", model.joints.actuated_dofs_offset)
        logger.debug("model.joints.bilateral_cts_offset: %s", model.joints.bilateral_cts_offset)
        logger.debug("model.joints.dynamic_cts_offset: %s", model.joints.dynamic_cts_offset)
        logger.debug("model.joints.kinematic_cts_offset: %s", model.joints.kinematic_cts_offset)
        logger.debug("model.joints.bounded_cts_offset: %s", model.joints.bounded_cts_offset)
        logger.debug("model.joints.friction_cts_offset: %s", model.joints.friction_cts_offset)
        logger.debug("model.joints.effort_cts_offset: %s", model.joints.effort_cts_offset)
        logger.debug("model.joints.dynamic_cts_offset_total_cts: %s", model.joints.dynamic_cts_offset_total_cts)
        logger.debug("model.joints.kinematic_cts_offset_total_cts: %s", model.joints.kinematic_cts_offset_total_cts)
        logger.debug("model.joints.friction_cts_offset_total_cts: %s", model.joints.friction_cts_offset_total_cts)
        logger.debug("model.joints.effort_cts_offset_total_cts: %s", model.joints.effort_cts_offset_total_cts)
    if parameters:
        logger.debug("model.joints.B_r_Bj: %s", model.joints.B_r_Bj)
        logger.debug("model.joints.F_r_Fj: %s", model.joints.F_r_Fj)
        logger.debug("model.joints.X_Bj: %s", model.joints.X_Bj)
        logger.debug("model.joints.X_Fj: %s", model.joints.X_Fj)
    if limits:
        logger.debug("model.joints.q_j_min: %s", model.joints.q_j_min)
        logger.debug("model.joints.q_j_max: %s", model.joints.q_j_max)
        logger.debug("model.joints.dq_j_max: %s", model.joints.dq_j_max)
        logger.debug("model.joints.tau_j_max: %s", model.joints.tau_j_max)
    if dynamics:
        logger.debug("model.joints.a_j: %s", model.joints.a_j)
        logger.debug("model.joints.b_j: %s", model.joints.b_j)
        logger.debug("model.joints.f_j: %s", model.joints.f_j)
        logger.debug("model.joints.k_p_j: %s", model.joints.k_p_j)
        logger.debug("model.joints.k_d_j: %s", model.joints.k_d_j)
        logger.debug("model.joints.dynamic_cts_axis: %s", model.joints.dynamic_cts_axis)
        logger.debug("model.joints.friction_cts_axis: %s", model.joints.friction_cts_axis)
        logger.debug("model.joints.effort_cts_axis: %s", model.joints.effort_cts_axis)


def debug_print_data_info(data: DataKamino):
    """Print ``data.info`` (constraint counts, offsets) in the debug log."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug("===============================================================================")
    logger.debug("data.info.num_limits:  %s", data.info.num_limits)
    logger.debug("data.info.num_contacts:  %s", data.info.num_contacts)
    logger.debug("-------------------------------------------------------------------------------")
    logger.debug("data.info.num_total_cts:  %s", data.info.num_total_cts)
    logger.debug("data.info.num_limit_cts:  %s", data.info.num_limit_cts)
    logger.debug("data.info.num_contact_cts:  %s", data.info.num_contact_cts)
    logger.debug("-------------------------------------------------------------------------------")
    logger.debug("data.info.limit_cts_group_offset:  %s", data.info.limit_cts_group_offset)
    logger.debug("data.info.contact_cts_group_offset:  %s", data.info.contact_cts_group_offset)


def debug_print_data(data: DataKamino, info=True):
    """Print ``data`` (bodies, joints, actuators, and optionally ``data.info``) in the debug log."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    # Print the state info
    if info:
        debug_print_data_info(data)
    # Print body state data
    logger.debug("data.bodies.I_i: %s", data.bodies.I_i)
    logger.debug("data.bodies.inv_I_i: %s", data.bodies.inv_I_i)
    logger.debug("data.bodies.q_i: %s", data.bodies.q_i)
    logger.debug("data.bodies.u_i: %s", data.bodies.u_i)
    logger.debug("data.bodies.w_i: %s", data.bodies.w_i)
    logger.debug("data.bodies.w_a_i: %s", data.bodies.w_a_i)
    logger.debug("data.bodies.w_j_i: %s", data.bodies.w_j_i)
    logger.debug("data.bodies.w_l_i: %s", data.bodies.w_l_i)
    logger.debug("data.bodies.w_f_i: %s", data.bodies.w_f_i)
    logger.debug("data.bodies.w_c_i: %s", data.bodies.w_c_i)
    logger.debug("data.bodies.w_e_i: %s", data.bodies.w_e_i)
    # Print joint state data
    logger.debug("data.joints.p_j: %s", data.joints.p_j)
    logger.debug("data.joints.q_j: %s", data.joints.q_j)
    logger.debug("data.joints.dq_j: %s", data.joints.dq_j)
    logger.debug("data.joints.tau_j: %s", data.joints.tau_j)
    logger.debug("data.joints.r_j: %s", data.joints.r_j)
    logger.debug("data.joints.dr_j: %s", data.joints.dr_j)
    logger.debug("data.joints.lambda_kin_j: %s", data.joints.lambda_kin_j)
    logger.debug("data.joints.lambda_dyn_j: %s", data.joints.lambda_dyn_j)
    logger.debug("data.joints.lambda_f_j: %s", data.joints.lambda_f_j)
    logger.debug("data.joints.lambda_tau_j: %s", data.joints.lambda_tau_j)
    logger.debug("data.joints.m_j: %s", data.joints.m_j)
    logger.debug("data.joints.inv_m_j: %s", data.joints.inv_m_j)
    logger.debug("data.joints.q_j_ref: %s", data.joints.q_j_ref)
    logger.debug("data.joints.dq_j_ref: %s", data.joints.dq_j_ref)
    logger.debug("data.joints.j_w_j: %s", data.joints.j_w_j)
    logger.debug("data.joints.j_w_a_j: %s", data.joints.j_w_a_j)
    logger.debug("data.joints.j_w_c_j: %s", data.joints.j_w_c_j)
    logger.debug("data.joints.j_w_f_j: %s", data.joints.j_w_f_j)
    logger.debug("data.joints.j_w_l_j: %s", data.joints.j_w_l_j)
    # Print the geometry state data
    logger.debug("data.geoms.pose: %s", data.geoms.pose)


###
# General-Purpose Functions
###


def debug_print_error_stats(name, arr, ref, n, show_errors=False):
    """Print abs/rel error statistics between ``arr`` and ``ref`` in the debug log."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    err = arr - ref
    err_abs = np.abs(err)
    err_l2 = np.linalg.norm(err)
    err_mean = np.sum(err_abs) / n
    err_max = np.max(err_abs)
    if show_errors:
        logger.debug("%s_err (%s):\n%s", name, err.shape, err)
    logger.debug("%s_err_l2: %s", name, err_l2)
    logger.debug("%s_err_mean: %s", name, err_mean)
    logger.debug("%s_err_max: %s\n\n", name, err_max)
