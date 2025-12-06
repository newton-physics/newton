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
KAMINO: UNIT TESTS: GENERAL UTILITIES
"""

import numpy as np

from newton._src.solvers.kamino.core.model import Model, ModelData

###
# Model Functions
###


def print_model_size(model: Model):
    print("Model Size:")

    # Print the host-side model size meta-data
    print(f"model.size.num_worlds: {model.size.num_worlds}")

    # Print the device-side model size data
    print(f"model.size.sum_of_num_bodies: {model.size.sum_of_num_bodies}")
    print(f"model.size.max_of_num_bodies: {model.size.max_of_num_bodies}")
    print(f"model.size.sum_of_num_joints: {model.size.sum_of_num_joints}")
    print(f"model.size.max_of_num_joints: {model.size.max_of_num_joints}")
    print(f"model.size.sum_of_num_material_pairs: {model.size.sum_of_num_material_pairs}")
    print(f"model.size.max_of_num_material_pairs: {model.size.max_of_num_material_pairs}")
    print(f"model.size.sum_of_num_body_dofs: {model.size.sum_of_num_body_dofs}")
    print(f"model.size.max_of_num_body_dofs: {model.size.max_of_num_body_dofs}")
    print(f"model.size.sum_of_num_joint_dofs: {model.size.sum_of_num_joint_dofs}")
    print(f"model.size.max_of_num_joint_dofs: {model.size.max_of_num_joint_dofs}")
    print(f"model.size.sum_of_max_unilaterals: {model.size.sum_of_max_unilaterals}")
    print(f"model.size.max_of_max_unilaterals: {model.size.max_of_max_unilaterals}")


def print_model_info(model: Model):
    print("Model Info:")

    # Print the host-side model info meta-data
    print(f"model.info.num_worlds: {model.info.num_worlds}")

    # Print the device-side model info data
    print(f"model.info.num_bodies: {model.info.num_bodies}")
    print(f"model.info.num_joints: {model.info.num_joints}")
    print(f"model.info.num_passive_joints: {model.info.num_passive_joints}")
    print(f"model.info.num_actuated_joints: {model.info.num_actuated_joints}")
    print(f"model.info.num_collision_geoms: {model.info.num_collision_geoms}")
    print(f"model.info.num_physical_geoms: {model.info.num_physical_geoms}")
    print(f"model.info.max_limits: {model.info.max_limits}")
    print(f"model.info.max_contacts: {model.info.max_contacts}")
    print(f"model.info.num_body_dofs: {model.info.num_body_dofs}")
    print(f"model.info.num_joint_dofs: {model.info.num_joint_dofs}")
    print(f"model.info.num_passive_joint_dofs: {model.info.num_passive_joint_dofs}")
    print(f"model.info.num_actuated_joint_dofs: {model.info.num_actuated_joint_dofs}")
    print(f"model.info.num_joint_cts: {model.info.num_joint_cts}")
    print(f"model.info.max_limit_cts: {model.info.max_limit_cts}")
    print(f"model.info.max_contact_cts: {model.info.max_contact_cts}")
    print(f"model.info.max_total_cts: {model.info.max_total_cts}")

    # Print the element offsets
    print(f"model.info.bodies_offset: {model.info.bodies_offset}")
    print(f"model.info.joints_offset: {model.info.joints_offset}")
    print(f"model.info.limits_offset: {model.info.limits_offset}")
    print(f"model.info.contacts_offset: {model.info.contacts_offset}")
    print(f"model.info.unilaterals_offset: {model.info.unilaterals_offset}")

    # Print the DoF and constraint offsets
    print(f"model.info.body_dofs_offset: {model.info.body_dofs_offset}")
    print(f"model.info.joint_dofs_offset: {model.info.joint_dofs_offset}")
    print(f"model.info.joint_passive_dofs_offset: {model.info.joint_passive_dofs_offset}")
    print(f"model.info.joint_actuated_dofs_offset: {model.info.joint_actuated_dofs_offset}")
    print(f"model.info.joint_cts_offset: {model.info.joint_cts_offset}")
    print(f"model.info.limit_cts_offset: {model.info.limit_cts_offset}")
    print(f"model.info.contact_cts_offset: {model.info.contact_cts_offset}")
    print(f"model.info.unilateral_cts_offset: {model.info.unilateral_cts_offset}")
    print(f"model.info.total_cts_offset: {model.info.total_cts_offset}")

    # Print the inertial properties
    print(f"model.info.mass_min: {model.info.mass_min}")
    print(f"model.info.mass_max: {model.info.mass_max}")
    print(f"model.info.mass_total: {model.info.mass_total}")
    print(f"model.info.inertia_total: {model.info.inertia_total}")


def print_model_constraint_info(model: Model):
    print("Model Constraint Info:")
    print(f"model.info.max_limits: {model.info.max_limits}")
    print(f"model.info.max_contacts: {model.info.max_contacts}")
    print(f"model.info.num_joint_cts: {model.info.num_joint_cts}")
    print(f"model.info.max_limit_cts: {model.info.max_limit_cts}")
    print(f"model.info.max_contact_cts: {model.info.max_contact_cts}")
    print(f"model.info.max_total_cts: {model.info.max_total_cts}")
    print(f"model.info.limits_offset: {model.info.limits_offset}")
    print(f"model.info.contacts_offset: {model.info.contacts_offset}")
    print(f"model.info.unilaterals_offset: {model.info.unilaterals_offset}")
    print(f"model.info.joint_cts_offset: {model.info.joint_cts_offset}")
    print(f"model.info.limit_cts_offset: {model.info.limit_cts_offset}")
    print(f"model.info.contact_cts_offset: {model.info.contact_cts_offset}")
    print(f"model.info.unilateral_cts_offset: {model.info.unilateral_cts_offset}")
    print(f"model.info.total_cts_offset: {model.info.total_cts_offset}")


def print_model_bodies(model: Model, inertias=True, initial_states=True):
    print(f"model.bodies.num_bodies: {model.bodies.num_bodies}")
    print(f"model.bodies.wid: {model.bodies.wid}")
    print(f"model.bodies.bid: {model.bodies.bid}")
    if inertias:
        print(f"model.bodies.m_i: {model.bodies.m_i}")
        print(f"model.bodies.inv_m_i:\n{model.bodies.inv_m_i}")
        print(f"model.bodies.i_I_i:\n{model.bodies.i_I_i}")
        print(f"model.bodies.inv_i_I_i:\n{model.bodies.inv_i_I_i}")
    if initial_states:
        print(f"model.bodies.q_i_0:\n{model.bodies.q_i_0}")
        print(f"model.bodies.u_i_0:\n{model.bodies.u_i_0}")


def print_model_joints(model: Model, offsets=True, parameters=True, limits=True):
    print(f"model.joints.num_joints: {model.joints.num_joints}")
    print(f"model.joints.wid: {model.joints.wid}")
    print(f"model.joints.jid: {model.joints.jid}")
    print(f"model.joints.dof_type: {model.joints.dof_type}")
    print(f"model.joints.act_type: {model.joints.act_type}")
    print(f"model.joints.num_cts: {model.joints.num_cts}")
    print(f"model.joints.num_dofs: {model.joints.num_dofs}")
    print(f"model.joints.bid_B: {model.joints.bid_B}")
    print(f"model.joints.bid_F: {model.joints.bid_F}")
    if offsets:
        print(f"model.joints.cts_offset: {model.joints.cts_offset}")
        print(f"model.joints.dofs_offset: {model.joints.dofs_offset}")
        print(f"model.joints.passive_dofs_offset: {model.joints.passive_dofs_offset}")
        print(f"model.joints.actuated_dofs_offset: {model.joints.actuated_dofs_offset}")
    if parameters:
        print(f"model.joints.B_r_Bj: {model.joints.B_r_Bj}")
        print(f"model.joints.F_r_Fj: {model.joints.F_r_Fj}")
        print(f"model.joints.X_j: {model.joints.X_j}")
    if limits:
        print(f"model.joints.q_j_min: {model.joints.q_j_min}")
        print(f"model.joints.q_j_max: {model.joints.q_j_max}")
        print(f"model.joints.dq_j_max: {model.joints.dq_j_max}")
        print(f"model.joints.tau_j_max: {model.joints.tau_j_max}")


def print_model_data_info(data: ModelData):
    print("data.info.num_total_cts: ", data.info.num_total_cts)
    print("data.info.num_limits: ", data.info.num_limits)
    print("data.info.num_limit_cts: ", data.info.num_limit_cts)
    print("data.info.limit_cts_group_offset: ", data.info.limit_cts_group_offset)
    print("data.info.num_contacts: ", data.info.num_contacts)
    print("data.info.num_contact_cts: ", data.info.num_contact_cts)
    print("data.info.contact_cts_group_offset: ", data.info.contact_cts_group_offset)


def print_model_state(data: ModelData, info=True):
    # Print the state info
    if info:
        print_model_data_info(data)
    # Print body state data
    print(f"data.bodies.I_i: {data.bodies.I_i}")
    print(f"data.bodies.inv_I_i: {data.bodies.inv_I_i}")
    print(f"data.bodies.q_i: {data.bodies.q_i}")
    print(f"data.bodies.u_i: {data.bodies.u_i}")
    print(f"data.bodies.w_i: {data.bodies.w_i}")
    print(f"data.bodies.w_a_i: {data.bodies.w_a_i}")
    print(f"data.bodies.w_j_i: {data.bodies.w_j_i}")
    print(f"data.bodies.w_l_i: {data.bodies.w_l_i}")
    print(f"data.bodies.w_c_i: {data.bodies.w_c_i}")
    print(f"data.bodies.w_e_i: {data.bodies.w_e_i}")
    # Print joint state data
    print(f"data.joints.p_j: {data.joints.p_j}")
    print(f"data.joints.r_j: {data.joints.r_j}")
    print(f"data.joints.dr_j: {data.joints.dr_j}")
    print(f"data.joints.lambda_j: {data.joints.lambda_j}")
    print(f"data.joints.q_j: {data.joints.q_j}")
    print(f"data.joints.dq_j: {data.joints.dq_j}")
    print(f"data.joints.tau_j: {data.joints.tau_j}")
    print(f"data.joints.j_w_j: {data.joints.j_w_j}")
    print(f"data.joints.j_w_c_j: {data.joints.j_w_c_j}")
    print(f"data.joints.j_w_a_j: {data.joints.j_w_a_j}")
    print(f"data.joints.j_w_l_j: {data.joints.j_w_l_j}")
    # Print the geometry state data
    print(f"data.cgeoms.pose: {data.cgeoms.pose}")
    print(f"data.cgeoms.aabb: {data.cgeoms.aabb}")
    print(f"data.cgeoms.radius: {data.cgeoms.radius}")


###
# General-Purpose Functions
###


def print_error_stats(name, arr, ref, n, show_errors=False):
    err = arr - ref
    err_abs = np.abs(err)
    err_l2 = np.linalg.norm(err)
    err_mean = np.sum(err_abs) / n
    err_max = np.max(err_abs)
    if show_errors:
        print(f"{name}_err ({err.shape}):\n{err}")
    print(f"{name}_err_l2: {err_l2}")
    print(f"{name}_err_mean: {err_mean}")
    print(f"{name}_err_max: {err_max}\n\n")
