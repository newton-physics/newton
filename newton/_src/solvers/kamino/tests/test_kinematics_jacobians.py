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
KAMINO: UNIT TESTS: KINEMATICS: JACOBIANS
"""

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.core.math import quat_exp, screw, screw_angular, screw_linear
from newton._src.solvers.kamino.core.model import Model, ModelData
from newton._src.solvers.kamino.core.types import float32, int32, mat33f, transformf, vec3f, vec6f
from newton._src.solvers.kamino.geometry.contacts import Contacts
from newton._src.solvers.kamino.geometry.detector import CollisionDetector
from newton._src.solvers.kamino.kinematics.constraints import make_unilateral_constraints_info, update_constraints_info

# Module to be tested
from newton._src.solvers.kamino.kinematics.jacobians import DenseSystemJacobians
from newton._src.solvers.kamino.kinematics.joints import compute_joints_state
from newton._src.solvers.kamino.kinematics.limits import Limits
from newton._src.solvers.kamino.models.builders import build_boxes_fourbar
from newton._src.solvers.kamino.models.utils import (
    make_heterogeneous_builder,
    make_homogeneous_builder,
    make_single_builder,
)

# Test utilities
from newton._src.solvers.kamino.tests.utils.print import (
    print_model_constraint_info,
    print_model_data_info,
)

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Helper Functions
###


def extract_cts_jacobians(
    model: Model,
    limits: Limits | None,
    contacts: Contacts | None,
    jacobians: DenseSystemJacobians,
    verbose: bool = False,
) -> list[np.ndarray]:
    # Retrieve the number of worlds in the model
    num_worlds = model.info.num_worlds

    # Reshape the flat Jacobian as a set of matrices
    cjmio = jacobians.data.J_cts_offsets.numpy()
    J_cts_flat = jacobians.data.J_cts_data.numpy()
    J_cts_flat_total_size = J_cts_flat.size
    J_cts_flat_offsets = [int(cjmio[i]) for i in range(num_worlds)]
    J_cts_flat_sizes = [0] * num_worlds
    J_cts_flat_offsets_ext = [*J_cts_flat_offsets, J_cts_flat_total_size]
    for i in range(num_worlds - 1, -1, -1):
        J_cts_flat_sizes[i] = J_cts_flat_offsets_ext[i + 1] - J_cts_flat_offsets_ext[i]

    # Extract each Jacobian as a matrix
    num_bdofs = [model.worlds[w].num_body_dofs for w in range(num_worlds)]
    num_jcts = [model.worlds[w].num_joint_cts for w in range(num_worlds)]
    maxnl = limits.num_world_max_limits if limits is not None else [0] * num_worlds
    maxnc = contacts.num_world_max_contacts if contacts is not None else [0] * num_worlds
    J_cts_mat: list[np.ndarray] = []
    for i in range(num_worlds):
        maxncts = num_jcts[i] + maxnl[i] + 3 * maxnc[i]
        start = J_cts_flat_offsets[i]
        end = J_cts_flat_offsets[i] + J_cts_flat_sizes[i]
        J_cts_mat.append(J_cts_flat[start:end].reshape((maxncts, num_bdofs[i])))

    # Optional verbose output
    if verbose:
        print(f"J_cts_flat_total_size: {J_cts_flat_total_size}")
        print(f"sum(J_cts_flat_sizes): {sum(J_cts_flat_sizes)}")
        print(f"J_cts_flat_sizes: {J_cts_flat_sizes}")
        print(f"J_cts_flat_offsets: {J_cts_flat_offsets}")
        print("")  # Add a newline for better readability
        for i in range(num_worlds):
            print(f"{i}: start={J_cts_flat_offsets[i]}, end={J_cts_flat_offsets[i] + J_cts_flat_sizes[i]}")
            print(f"J_cts_mat[{i}] ({J_cts_mat[i].shape}):\n{J_cts_mat[i]}\n")

    # Return the extracted Jacobians
    return J_cts_mat


def extract_dofs_jacobians(model: Model, jacobians: DenseSystemJacobians, verbose: bool = False) -> list[np.ndarray]:
    # Retrieve the number of worlds in the model
    num_worlds = model.info.num_worlds

    # Reshape the flat Jacobian as a set of matrices
    ajmio = jacobians.data.J_dofs_offsets.numpy()
    J_dofs_flat = jacobians.data.J_dofs_data.numpy()
    J_dofs_flat_total_size = J_dofs_flat.size
    J_dofs_flat_offsets = [int(ajmio[i]) for i in range(num_worlds)]
    J_dofs_flat_sizes = [0] * num_worlds
    J_dofs_flat_offsets_ext = [*J_dofs_flat_offsets, J_dofs_flat_total_size]
    for i in range(num_worlds - 1, -1, -1):
        J_dofs_flat_sizes[i] = J_dofs_flat_offsets_ext[i + 1] - J_dofs_flat_offsets_ext[i]

    # Extract each Jacobian as a matrix
    num_bdofs = [model.worlds[w].num_body_dofs for w in range(num_worlds)]
    num_jdofs = [model.worlds[w].num_joint_dofs for w in range(num_worlds)]
    J_dofs_mat: list[np.ndarray] = []
    for i in range(num_worlds):
        start = J_dofs_flat_offsets[i]
        end = J_dofs_flat_offsets[i] + J_dofs_flat_sizes[i]
        J_dofs_mat.append(J_dofs_flat[start:end].reshape((num_jdofs[i], num_bdofs[i])))

    # Optional verbose output
    if verbose:
        print(f"J_dofs_flat_total_size: {J_dofs_flat_total_size}")
        print(f"sum(J_dofs_flat_sizes): {sum(J_dofs_flat_sizes)}")
        print(f"J_dofs_flat_sizes: {J_dofs_flat_sizes}")
        print(f"J_dofs_flat_offsets: {J_dofs_flat_offsets}")
        print("")  # Add a newline for better readability
        for i in range(num_worlds):
            print(f"{i}: start={J_dofs_flat_offsets[i]}, end={J_dofs_flat_offsets[i] + J_dofs_flat_sizes[i]}")
            print(f"J_dofs_mat[{i}] ({J_dofs_mat[i].shape}):\n{J_dofs_mat[i]}\n")

    # Return the extracted Jacobians
    return J_dofs_mat


###
# Constants
###

Q_X_J = 0.3 * math.pi
THETA_Y_J = 0.0
THETA_Z_J = 0.0
J_DR_J = vec3f(0.0)
J_DV_J = vec3f(0.0)
J_DOMEGA_J = vec3f(0.0)


###
# Kernels
###


@wp.kernel
def _set_fourbar_body_states(
    model_joint_bid_B: wp.array(dtype=int32),
    model_joint_bid_F: wp.array(dtype=int32),
    model_joint_B_r_Bj: wp.array(dtype=vec3f),
    model_joint_F_r_Fj: wp.array(dtype=vec3f),
    model_joint_X_j: wp.array(dtype=mat33f),
    state_body_q_i: wp.array(dtype=transformf),
    state_body_u_i: wp.array(dtype=vec6f),
):
    """
    Set the state of the bodies to a certain values in order to check computations of joint states.
    """
    # Retrieve the thread index as the joint index
    jid = wp.tid()

    # Retrieve the joint parameters
    bid_B = model_joint_bid_B[jid]
    bid_F = model_joint_bid_F[jid]
    B_r_Bj = model_joint_B_r_Bj[jid]
    F_r_Fj = model_joint_F_r_Fj[jid]
    X_j = model_joint_X_j[jid]

    # Retrieve the current state of the Base body
    p_B = state_body_q_i[bid_B]
    u_B = state_body_u_i[bid_B]

    # Extract the position and orientation of the Base body
    r_B = wp.transform_get_translation(p_B)
    q_B = wp.transform_get_rotation(p_B)
    R_B = wp.quat_to_matrix(q_B)

    # Extract the linear and angular velocity of the Base body
    v_B = screw_linear(u_B)
    omega_B = screw_angular(u_B)

    # Define the joint rotation offset
    # NOTE: X_j projects quantities into the joint frame
    # NOTE: X_j^T projects quantities into the outer frame (world or body)
    q_x_j = Q_X_J * wp.pow(-1.0, float(jid))  # Alternate sign for each joint
    theta_y_j = THETA_Y_J
    theta_z_j = THETA_Z_J
    j_dR_j = vec3f(q_x_j, theta_y_j, theta_z_j)  # Joint offset as rotation vector
    q_jq = quat_exp(j_dR_j)  # Joint offset as rotation quaternion
    R_jq = wp.quat_to_matrix(q_jq)  # Joint offset as rotation matrix

    # Define the joint translation offset
    j_dr_j = J_DR_J

    # Define the joint twist offset
    j_dv_j = J_DV_J
    j_domega_j = J_DOMEGA_J

    # Follower body rotation via the Base and joint frames
    R_B_X_j = R_B @ X_j
    R_F_new = R_B_X_j @ R_jq @ wp.transpose(X_j)
    q_F_new = wp.quat_from_matrix(R_F_new)

    # Follower body position via the Base and joint frames
    r_Fj = R_F_new @ F_r_Fj
    r_F_new = r_B + R_B @ B_r_Bj + R_B_X_j @ j_dr_j - r_Fj

    # Follower body twist via the Base and joint frames
    r_Bj = R_B @ B_r_Bj
    r_Fj = R_F_new @ F_r_Fj
    omega_F_new = R_B_X_j @ j_domega_j + omega_B
    v_F_new = R_B_X_j @ j_dv_j + v_B + wp.cross(omega_B, r_Bj) - wp.cross(omega_F_new, r_Fj)

    # Offset the bose of the body by a fixed amount
    state_body_q_i[bid_F] = wp.transformation(r_F_new, q_F_new, dtype=float32)
    state_body_u_i[bid_F] = screw(v_F_new, omega_F_new)


###
# Launchers
###


def set_fourbar_body_states(model: Model, data: ModelData):
    wp.launch(
        _set_fourbar_body_states,
        dim=3,  # Set to three because we only need to set the first three joints
        inputs=[
            model.joints.bid_B,
            model.joints.bid_F,
            model.joints.B_r_Bj,
            model.joints.F_r_Fj,
            model.joints.X_j,
            data.bodies.q_i,
            data.bodies.u_i,
        ],
    )


###
# Tests
###


class TestKinematicsJacobians(unittest.TestCase):
    def setUp(self):
        self.verbose = False  # Set to True for verbose output
        self.default_device = wp.get_device()

    def tearDown(self):
        self.default_device = None

    def test_01_allocate_single_dense_system_jacobians_only_joints(self):
        # Construct the model description using the ModelBuilder
        builder, _, _ = make_single_builder(build_func=build_boxes_fourbar)

        # Create the model from the builder
        model = builder.finalize(device=self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"model.size.sum_of_num_bodies: {model.size.sum_of_num_bodies}")
            print(f"model.size.sum_of_num_joints: {model.size.sum_of_num_joints}")
            print(f"model.size.sum_of_num_joint_cts: {model.size.sum_of_num_joint_cts}")
            print(f"model.size.sum_of_num_joint_dofs: {model.size.sum_of_num_joint_dofs}")

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(model=model, device=self.default_device)
        if self.verbose:
            print(
                f"jacobians.data.J_cts_offsets (shape={jacobians.data.J_cts_offsets.shape}): {jacobians.data.J_cts_offsets}"
            )
            print(
                f"jacobians.data.J_dofs_offsets (shape={jacobians.data.J_dofs_offsets.shape}): {jacobians.data.J_dofs_offsets}"
            )
            print(f"jacobians.data.J_cts_data: shape={jacobians.data.J_cts_data.shape}")
            print(f"jacobians.data.J_dofs_data: shape={jacobians.data.J_dofs_data.shape}")

        # Check the allocations of Jacobians
        model_num_cts = model.size.sum_of_num_joint_cts
        self.assertEqual(jacobians.data.J_dofs_offsets.size, 1)
        self.assertEqual(jacobians.data.J_cts_offsets.size, 1)
        self.assertEqual(jacobians.data.J_dofs_offsets.numpy()[0], 0)
        self.assertEqual(jacobians.data.J_cts_offsets.numpy()[0], 0)
        self.assertEqual(
            jacobians.data.J_dofs_data.shape, (model.size.sum_of_num_joint_dofs * model.size.sum_of_num_body_dofs,)
        )
        self.assertEqual(jacobians.data.J_cts_data.shape, (model_num_cts * model.size.sum_of_num_body_dofs,))

    def test_02_allocate_single_dense_system_jacobians_with_limits(self):
        # Construct the model description using the ModelBuilder
        builder, _, _ = make_single_builder(build_func=build_boxes_fourbar)

        # Create the model from the builder
        model = builder.finalize(device=self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"model.size.sum_of_num_bodies: {model.size.sum_of_num_bodies}")
            print(f"model.size.sum_of_num_joints: {model.size.sum_of_num_joints}")
            print(f"model.size.sum_of_num_joint_cts: {model.size.sum_of_num_joint_cts}")
            print(f"model.size.sum_of_num_joint_dofs: {model.size.sum_of_num_joint_dofs}")

        # Construct and allocate the limits container
        limits = Limits(builder=builder, device=self.default_device)
        if self.verbose:
            print("limits.num_model_max_limits: ", limits.num_model_max_limits)
            print("limits.num_world_max_limits: ", limits.num_world_max_limits)

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(model=model, limits=limits, device=self.default_device)
        if self.verbose:
            print(
                f"jacobians.data.J_dofs_offsets (shape={jacobians.data.J_dofs_offsets.shape}): {jacobians.data.J_dofs_offsets}"
            )
            print(
                f"jacobians.data.J_cts_offsets (shape={jacobians.data.J_cts_offsets.shape}): {jacobians.data.J_cts_offsets}"
            )
            print(f"jacobians.data.J_dofs_data: shape={jacobians.data.J_dofs_data.shape}")
            print(f"jacobians.data.J_cts_data: shape={jacobians.data.J_cts_data.shape}")

        # Check the allocations of Jacobians
        model_num_cts = model.size.sum_of_num_joint_cts + limits.num_model_max_limits
        self.assertEqual(jacobians.data.J_dofs_offsets.size, 1)
        self.assertEqual(jacobians.data.J_cts_offsets.size, 1)
        self.assertEqual(jacobians.data.J_dofs_offsets.numpy()[0], 0)
        self.assertEqual(jacobians.data.J_cts_offsets.numpy()[0], 0)
        self.assertEqual(
            jacobians.data.J_dofs_data.shape, (model.size.sum_of_num_joint_dofs * model.size.sum_of_num_body_dofs,)
        )
        self.assertEqual(jacobians.data.J_cts_data.shape, (model_num_cts * model.size.sum_of_num_body_dofs,))

    def test_03_allocate_single_dense_system_jacobians_with_contacts(self):
        # Problem constants
        max_world_contacts = 12

        # Construct the model description using the ModelBuilder
        builder, _, _ = make_single_builder(build_func=build_boxes_fourbar)

        # Create the model from the builder
        model = builder.finalize(device=self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"model.size.sum_of_num_bodies: {model.size.sum_of_num_bodies}")
            print(f"model.size.sum_of_num_joints: {model.size.sum_of_num_joints}")
            print(f"model.size.sum_of_num_joint_cts: {model.size.sum_of_num_joint_cts}")
            print(f"model.size.sum_of_num_joint_dofs: {model.size.sum_of_num_joint_dofs}")

        # Extract the contact allocation capacities required by the model
        required_model_max_contacts, required_world_max_contacts = builder.required_contact_capacity()
        if self.verbose:
            print("required_model_max_contacts: ", required_model_max_contacts)
            print("required_world_max_contacts: ", required_world_max_contacts)

        # Construct and allocate the contacts container
        contacts = Contacts(
            capacity=required_world_max_contacts, default_max_contacts=max_world_contacts, device=self.default_device
        )
        if self.verbose:
            print("contacts.default_max_world_contacts: ", contacts.default_max_world_contacts)
            print("contacts.num_model_max_contacts: ", contacts.num_model_max_contacts)
            print("contacts.num_world_max_contacts: ", contacts.num_world_max_contacts)

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(model=model, contacts=contacts, device=self.default_device)
        if self.verbose:
            print(
                f"jacobians.data.J_dofs_offsets (shape={jacobians.data.J_dofs_offsets.shape}): {jacobians.data.J_dofs_offsets}"
            )
            print(
                f"jacobians.data.J_cts_offsets (shape={jacobians.data.J_cts_offsets.shape}): {jacobians.data.J_cts_offsets}"
            )
            print(f"jacobians.data.J_dofs_data: shape={jacobians.data.J_dofs_data.shape}")
            print(f"jacobians.data.J_cts_data: shape={jacobians.data.J_cts_data.shape}")

        # Check the allocations of Jacobians
        model_num_cts = model.size.sum_of_num_joint_cts + 3 * contacts.num_model_max_contacts
        self.assertEqual(jacobians.data.J_dofs_offsets.size, 1)
        self.assertEqual(jacobians.data.J_cts_offsets.size, 1)
        self.assertEqual(jacobians.data.J_dofs_offsets.numpy()[0], 0)
        self.assertEqual(jacobians.data.J_cts_offsets.numpy()[0], 0)
        self.assertEqual(
            jacobians.data.J_dofs_data.shape, (model.size.sum_of_num_joint_dofs * model.size.sum_of_num_body_dofs,)
        )
        self.assertEqual(jacobians.data.J_cts_data.shape, (model_num_cts * model.size.sum_of_num_body_dofs,))

    def test_04_allocate_single_dense_system_jacobians_with_limits_and_contacts(self):
        # Problem constants
        max_world_contacts = 12

        # Construct the model description using the ModelBuilder
        builder, _, _ = make_single_builder(build_func=build_boxes_fourbar)

        # Create the model from the builder
        model = builder.finalize(device=self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"model.size.sum_of_num_bodies: {model.size.sum_of_num_bodies}")
            print(f"model.size.sum_of_num_joints: {model.size.sum_of_num_joints}")
            print(f"model.size.sum_of_num_joint_cts: {model.size.sum_of_num_joint_cts}")
            print(f"model.size.sum_of_num_joint_dofs: {model.size.sum_of_num_joint_dofs}")

        # Construct and allocate the limits container
        limits = Limits(builder=builder, device=self.default_device)
        if self.verbose:
            print("limits.num_model_max_limits: ", limits.num_model_max_limits)
            print("limits.num_world_max_limits: ", limits.num_world_max_limits)

        # Extract the contact allocation capacities required by the model
        required_model_max_contacts, required_world_max_contacts = builder.required_contact_capacity()
        if self.verbose:
            print("required_model_max_contacts: ", required_model_max_contacts)
            print("required_world_max_contacts: ", required_world_max_contacts)

        # Construct and allocate the contacts container
        contacts = Contacts(
            capacity=required_world_max_contacts, device=self.default_device, default_max_contacts=max_world_contacts
        )
        if self.verbose:
            print("contacts.default_max_world_contacts: ", contacts.default_max_world_contacts)
            print("contacts.num_model_max_contacts: ", contacts.num_model_max_contacts)
            print("contacts.num_world_max_contacts: ", contacts.num_world_max_contacts)

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(model=model, limits=limits, contacts=contacts, device=self.default_device)
        if self.verbose:
            print(
                f"jacobians.data.J_dofs_offsets (shape={jacobians.data.J_dofs_offsets.shape}): {jacobians.data.J_dofs_offsets}"
            )
            print(
                f"jacobians.data.J_cts_offsets (shape={jacobians.data.J_cts_offsets.shape}): {jacobians.data.J_cts_offsets}"
            )
            print(f"jacobians.data.J_dofs_data: shape={jacobians.data.J_dofs_data.shape}")
            print(f"jacobians.data.J_cts_data: shape={jacobians.data.J_cts_data.shape}")

        # Check the allocations of Jacobians
        model_num_cts = (
            model.size.sum_of_num_joint_cts + limits.num_model_max_limits + 3 * contacts.num_model_max_contacts
        )
        self.assertEqual(jacobians.data.J_dofs_offsets.size, 1)
        self.assertEqual(jacobians.data.J_cts_offsets.size, 1)
        self.assertEqual(jacobians.data.J_dofs_offsets.numpy()[0], 0)
        self.assertEqual(jacobians.data.J_cts_offsets.numpy()[0], 0)
        self.assertEqual(
            jacobians.data.J_dofs_data.shape, (model.size.sum_of_num_joint_dofs * model.size.sum_of_num_body_dofs,)
        )
        self.assertEqual(jacobians.data.J_cts_data.shape, (model_num_cts * model.size.sum_of_num_body_dofs,))

    def test_05_allocate_homogeneous_dense_system_jacobians(self):
        # Problem constants
        num_worlds = 3
        max_world_contacts = 12

        # Construct the model description using the ModelBuilder
        builder = make_homogeneous_builder(num_worlds=num_worlds, build_func=build_boxes_fourbar)

        # Create the model from the builder
        model = builder.finalize(device=self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"model.size.sum_of_num_bodies: {model.size.sum_of_num_bodies}")
            print(f"model.size.sum_of_num_joints: {model.size.sum_of_num_joints}")
            print(f"model.size.sum_of_num_joint_cts: {model.size.sum_of_num_joint_cts}")
            print(f"model.size.sum_of_num_joint_dofs: {model.size.sum_of_num_joint_dofs}")

        # Construct and allocate the limits container
        limits = Limits(builder=builder, device=self.default_device)
        if self.verbose:
            print("limits.num_model_max_limits: ", limits.num_model_max_limits)
            print("limits.num_world_max_limits: ", limits.num_world_max_limits)

        # Extract the contact allocation capacities required by the model
        required_model_max_contacts, required_world_max_contacts = builder.required_contact_capacity()
        if self.verbose:
            print("required_model_max_contacts: ", required_model_max_contacts)
            print("required_world_max_contacts: ", required_world_max_contacts)

        # Construct and allocate the contacts container
        contacts = Contacts(
            capacity=required_world_max_contacts, device=self.default_device, default_max_contacts=max_world_contacts
        )
        if self.verbose:
            print("contacts.default_max_world_contacts: ", contacts.default_max_world_contacts)
            print("contacts.num_model_max_contacts: ", contacts.num_model_max_contacts)
            print("contacts.num_world_max_contacts: ", contacts.num_world_max_contacts)

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(model=model, limits=limits, contacts=contacts, device=self.default_device)
        if self.verbose:
            print(
                f"jacobians.data.J_dofs_offsets (shape={jacobians.data.J_dofs_offsets.shape}): {jacobians.data.J_dofs_offsets}"
            )
            print(
                f"jacobians.data.J_cts_offsets (shape={jacobians.data.J_cts_offsets.shape}): {jacobians.data.J_cts_offsets}"
            )
            print(f"jacobians.data.J_dofs_data: shape={jacobians.data.J_dofs_data.shape}")
            print(f"jacobians.data.J_cts_data: shape={jacobians.data.J_cts_data.shape}")

        # Compute the total maximum number of constraints
        num_body_dofs = [model.worlds[w].num_body_dofs for w in range(num_worlds)]
        num_joint_dofs = [model.worlds[w].num_joint_dofs for w in range(num_worlds)]
        num_total_cts = [
            (model.worlds[w].num_joint_cts + limits.num_world_max_limits[w] + 3 * contacts.num_world_max_contacts[w])
            for w in range(num_worlds)
        ]
        if self.verbose:
            print("num_body_dofs: ", num_body_dofs)
            print("num_total_cts: ", num_total_cts)
            print("num_joint_dofs: ", num_joint_dofs)

        # Compute Jacobian sizes
        J_dofs_size: list[int] = [0] * num_worlds
        J_cts_size: list[int] = [0] * num_worlds
        for w in range(num_worlds):
            J_dofs_size[w] = num_joint_dofs[w] * num_body_dofs[w]
            J_cts_size[w] = num_total_cts[w] * num_body_dofs[w]

        # Compute Jacobian offsets
        J_dofs_offsets: list[int] = [0] + [sum(J_dofs_size[:w]) for w in range(1, num_worlds)]
        J_cts_offsets: list[int] = [0] + [sum(J_cts_size[:w]) for w in range(1, num_worlds)]

        # Check the allocations of Jacobians
        self.assertEqual(jacobians.data.J_dofs_offsets.size, num_worlds)
        self.assertEqual(jacobians.data.J_cts_offsets.size, num_worlds)
        J_dofs_mio_np = jacobians.data.J_dofs_offsets.numpy()
        J_cts_mio_np = jacobians.data.J_cts_offsets.numpy()
        for w in range(num_worlds):
            self.assertEqual(J_dofs_mio_np[w], J_dofs_offsets[w])
            self.assertEqual(J_cts_mio_np[w], J_cts_offsets[w])
        self.assertEqual(jacobians.data.J_dofs_data.size, sum(J_dofs_size))
        self.assertEqual(jacobians.data.J_cts_data.size, sum(J_cts_size))

    def test_06_allocate_heterogeneous_dense_system_jacobians(self):
        # Problem constants
        max_world_contacts = 12

        # Construct the model description using the ModelBuilder
        builder, _, _ = make_heterogeneous_builder()
        num_worlds = builder.num_worlds

        # Create the model from the builder
        model = builder.finalize(device=self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"model.size.sum_of_num_bodies: {model.size.sum_of_num_bodies}")
            print(f"model.size.sum_of_num_joints: {model.size.sum_of_num_joints}")
            print(f"model.size.sum_of_num_joint_cts: {model.size.sum_of_num_joint_cts}")
            print(f"model.size.sum_of_num_joint_dofs: {model.size.sum_of_num_joint_dofs}")

        # Construct and allocate the limits container
        limits = Limits(builder=builder, device=self.default_device)
        if self.verbose:
            print("limits.num_model_max_limits: ", limits.num_model_max_limits)
            print("limits.num_world_max_limits: ", limits.num_world_max_limits)

        # Extract the contact allocation capacities required by the model
        required_model_max_contacts, required_world_max_contacts = builder.required_contact_capacity()
        if self.verbose:
            print("required_model_max_contacts: ", required_model_max_contacts)
            print("required_world_max_contacts: ", required_world_max_contacts)

        # Construct and allocate the contacts container
        contacts = Contacts(
            capacity=required_world_max_contacts, device=self.default_device, default_max_contacts=max_world_contacts
        )
        if self.verbose:
            print("contacts.default_max_world_contacts: ", contacts.default_max_world_contacts)
            print("contacts.num_model_max_contacts: ", contacts.num_model_max_contacts)
            print("contacts.num_world_max_contacts: ", contacts.num_world_max_contacts)

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(model=model, limits=limits, contacts=contacts, device=self.default_device)
        if self.verbose:
            print(
                f"jacobians.data.J_dofs_offsets (shape={jacobians.data.J_dofs_offsets.shape}): {jacobians.data.J_dofs_offsets}"
            )
            print(
                f"jacobians.data.J_cts_offsets (shape={jacobians.data.J_cts_offsets.shape}): {jacobians.data.J_cts_offsets}"
            )
            print(f"jacobians.data.J_dofs_data: shape={jacobians.data.J_dofs_data.shape}")
            print(f"jacobians.data.J_cts_data: shape={jacobians.data.J_cts_data.shape}")

        # Compute the total maximum number of constraints
        num_body_dofs = [model.worlds[w].num_body_dofs for w in range(num_worlds)]
        num_joint_dofs = [model.worlds[w].num_joint_dofs for w in range(num_worlds)]
        num_total_cts = [
            (model.worlds[w].num_joint_cts + limits.num_world_max_limits[w] + 3 * contacts.num_world_max_contacts[w])
            for w in range(num_worlds)
        ]
        if self.verbose:
            print("num_body_dofs: ", num_body_dofs)
            print("num_total_cts: ", num_total_cts)
            print("num_joint_dofs: ", num_joint_dofs)

        # Compute Jacobian sizes
        J_dofs_size: list[int] = [0] * num_worlds
        J_cts_size: list[int] = [0] * num_worlds
        for w in range(num_worlds):
            J_dofs_size[w] = num_joint_dofs[w] * num_body_dofs[w]
            J_cts_size[w] = num_total_cts[w] * num_body_dofs[w]

        # Compute Jacobian offsets
        J_dofs_offsets: list[int] = [0] + [sum(J_dofs_size[:w]) for w in range(1, num_worlds)]
        J_cts_offsets: list[int] = [0] + [sum(J_cts_size[:w]) for w in range(1, num_worlds)]

        # Check the allocations of Jacobians
        self.assertEqual(jacobians.data.J_dofs_offsets.size, num_worlds)
        self.assertEqual(jacobians.data.J_cts_offsets.size, num_worlds)
        J_dofs_mio_np = jacobians.data.J_dofs_offsets.numpy()
        J_cts_mio_np = jacobians.data.J_cts_offsets.numpy()
        for w in range(num_worlds):
            self.assertEqual(J_dofs_mio_np[w], J_dofs_offsets[w])
            self.assertEqual(J_cts_mio_np[w], J_cts_offsets[w])
        self.assertEqual(jacobians.data.J_dofs_data.size, sum(J_dofs_size))
        self.assertEqual(jacobians.data.J_cts_data.size, sum(J_cts_size))

    def test_07_build_single_dense_system_jacobians(self):
        # Constants
        max_world_contacts = 12

        # Construct the model description using the ModelBuilder
        builder, _, _ = make_single_builder(build_func=build_boxes_fourbar)

        # Create the model from the builder
        model = builder.finalize(device=self.default_device)

        # Create a model state container
        data = model.data(device=self.default_device)

        # Construct and allocate the limits container
        limits = Limits(builder=builder, device=self.default_device)

        # Create the collision detector
        detector = CollisionDetector(
            builder=builder, default_max_contacts=max_world_contacts, device=self.default_device
        )

        # Create the constraints info
        make_unilateral_constraints_info(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            device=self.default_device,
        )
        if self.verbose:
            print("")  # Add a newline for better readability
            print_model_constraint_info(model)
            print_model_data_info(data)

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(
            model=model, limits=limits, contacts=detector.contacts, device=self.default_device
        )
        wp.synchronize()

        # Perturn the fourbar bodies in poses that trigger the joint limits
        set_fourbar_body_states(model=model, data=data)
        wp.synchronize()
        if self.verbose:
            print("data.bodies.q_i:\n", data.bodies.q_i)
            print("data.bodies.u_i:\n", data.bodies.u_i)

        # Compute the joints state
        compute_joints_state(model=model, q_j_p=wp.zeros_like(data.joints.q_j), data=data)
        wp.synchronize()
        if self.verbose:
            print("data.joints.p_j:\n", data.joints.p_j)
            print("data.joints.r_j:\n", data.joints.r_j)
            print("data.joints.dr_j:\n", data.joints.dr_j)
            print("data.joints.q_j:\n", data.joints.q_j)
            print("data.joints.dq_j:\n", data.joints.dq_j)

        # Run limit detection to generate active limits
        limits.detect(model, data)
        wp.synchronize()
        if self.verbose:
            print(f"limits.world_num_limits: {limits.world_num_limits}")
            print(f"data.info.num_limits: {data.info.num_limits}")

        # Run collision detection to generate active contacts
        detector.collide(model, data)
        wp.synchronize()
        if self.verbose:
            print(f"contacts.world_num_contacts: {detector.contacts.world_num_contacts}")
            print(f"data.info.num_contacts: {data.info.num_contacts}")

        # Update the constraints info
        update_constraints_info(model=model, data=data)
        if self.verbose:
            print("")  # Add a newline for better readability
            print_model_data_info(data)
        wp.synchronize()

        # Build the dense system Jacobians
        jacobians.build(model=model, data=data, limits=limits.data, contacts=detector.contacts.data)
        wp.synchronize()

        # Reshape the flat actuation Jacobian as a matrix
        J_dofs_offsets = jacobians.data.J_dofs_offsets.numpy()
        J_dofs_flat = jacobians.data.J_dofs_data.numpy()
        njd = J_dofs_flat.size // model.size.sum_of_num_body_dofs
        J_dofs_mat = J_dofs_flat.reshape((njd, model.size.sum_of_num_body_dofs))

        # Reshape the flat constraintJacobian as a matrix
        J_cts_offsets = jacobians.data.J_cts_offsets.numpy()
        J_cts_flat = jacobians.data.J_cts_data.numpy()
        maxncts = J_cts_flat.size // model.size.sum_of_num_body_dofs
        J_cts_mat = J_cts_flat.reshape((maxncts, model.size.sum_of_num_body_dofs))

        # Check the shapes of the Jacobians
        self.assertEqual(J_dofs_offsets.size, 1)
        self.assertEqual(J_cts_offsets.size, 1)
        self.assertEqual(
            maxncts,
            model.size.sum_of_num_joint_cts
            + limits.num_model_max_limits
            + 3 * detector.contacts.num_model_max_contacts,
        )
        self.assertEqual(njd, model.size.sum_of_num_joint_dofs)

        # Optional verbose output
        if self.verbose:
            print(
                f"jacobians.data.J_cts_offsets (shape={jacobians.data.J_cts_offsets.shape}): {jacobians.data.J_cts_offsets}"
            )
            print(f"J_cts_flat (shape={J_cts_flat.shape}):\n{J_cts_flat}")
            print(f"J_cts_mat (shape={J_cts_mat.shape}):\n{J_cts_mat}")
            print(
                f"jacobians.data.J_dofs_offsets (shape={jacobians.data.J_dofs_offsets.shape}): {jacobians.data.J_dofs_offsets}"
            )
            print(f"J_dofs_flat (shape={J_dofs_flat.shape}):\n{J_dofs_flat}")
            print(f"J_dofs_mat (shape={J_dofs_mat.shape}):\n{J_dofs_mat}")

    def test_08_build_homogeneous_dense_system_jacobians(self):
        # Problem constants
        num_worlds = 3
        max_world_contacts = 12

        # Construct the model description using the ModelBuilder
        builder = make_homogeneous_builder(num_worlds=num_worlds, build_func=build_boxes_fourbar)

        # Create the model from the builder
        model = builder.finalize(device=self.default_device)

        # Create a model state container
        data = model.data(device=self.default_device)

        # Construct and allocate the limits container
        limits = Limits(builder=builder, device=self.default_device)

        # Create the collision detector
        detector = CollisionDetector(
            builder=builder, default_max_contacts=max_world_contacts, device=self.default_device
        )

        # Create the constraints info
        make_unilateral_constraints_info(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            device=self.default_device,
        )
        if self.verbose:
            print("")  # Add a newline for better readability
            print_model_constraint_info(model)
            print_model_data_info(data)

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(
            model=model, limits=limits, contacts=detector.contacts, device=self.default_device
        )
        wp.synchronize()

        # Perturn the fourbar bodies in poses that trigger the joint limits
        set_fourbar_body_states(model=model, data=data)
        wp.synchronize()
        if self.verbose:
            print("data.bodies.q_i:\n", data.bodies.q_i)
            print("data.bodies.u_i:\n", data.bodies.u_i)

        # Compute the joints state
        compute_joints_state(model=model, q_j_p=wp.zeros_like(data.joints.q_j), data=data)
        wp.synchronize()
        if self.verbose:
            print("data.joints.p_j:\n", data.joints.p_j)
            print("data.joints.r_j:\n", data.joints.r_j)
            print("data.joints.dr_j:\n", data.joints.dr_j)
            print("data.joints.q_j:\n", data.joints.q_j)
            print("data.joints.dq_j:\n", data.joints.dq_j)

        # Run limit detection to generate active limits
        limits.detect(model, data)
        wp.synchronize()
        if self.verbose:
            print(f"limits.world_num_limits: {limits.world_num_limits}")
            print(f"data.info.num_limits: {data.info.num_limits}")

        # Run collision detection to generate active contacts
        detector.collide(model, data)
        wp.synchronize()
        if self.verbose:
            print(f"contacts.world_num_contacts: {detector.contacts.world_num_contacts}")
            print(f"data.info.num_contacts: {data.info.num_contacts}")

        # Update the constraints info
        update_constraints_info(model=model, data=data)
        if self.verbose:
            print("")  # Add a newline for better readability
            print_model_data_info(data)
        wp.synchronize()

        # Build the dense system Jacobians
        jacobians.build(model=model, data=data, limits=limits.data, contacts=detector.contacts.data)
        wp.synchronize()

        # Extract the Jacobian matrices
        extract_cts_jacobians(
            model=model, limits=limits, contacts=detector.contacts, jacobians=jacobians, verbose=self.verbose
        )
        extract_dofs_jacobians(model=model, jacobians=jacobians, verbose=self.verbose)
        # TODO: Add checks for the Jacobian matrices

    def test_09_build_heterogeneous_dense_system_jacobians(self):
        # Problem constants
        max_world_contacts = 12

        # Construct the model description using the ModelBuilder
        builder, _, _ = make_heterogeneous_builder()

        # Create the model from the builder
        model = builder.finalize(device=self.default_device)

        # Create a model state container
        data = model.data(device=self.default_device)

        # Construct and allocate the limits container
        limits = Limits(builder=builder, device=self.default_device)

        # Create the collision detector
        detector = CollisionDetector(
            builder=builder, default_max_contacts=max_world_contacts, device=self.default_device
        )

        # Create the constraints info
        make_unilateral_constraints_info(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            device=self.default_device,
        )
        if self.verbose:
            print("")  # Add a newline for better readability
            print_model_constraint_info(model)
            print_model_data_info(data)

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(
            model=model, limits=limits, contacts=detector.contacts, device=self.default_device
        )
        wp.synchronize()

        # Perturn the fourbar bodies in poses that trigger the joint limits
        set_fourbar_body_states(model=model, data=data)
        wp.synchronize()
        if self.verbose:
            print("data.bodies.q_i:\n", data.bodies.q_i)
            print("data.bodies.u_i:\n", data.bodies.u_i)

        # Compute the joints state
        compute_joints_state(model=model, q_j_p=wp.zeros_like(data.joints.q_j), data=data)
        wp.synchronize()
        if self.verbose:
            print("data.joints.p_j:\n", data.joints.p_j)
            print("data.joints.r_j:\n", data.joints.r_j)
            print("data.joints.dr_j:\n", data.joints.dr_j)
            print("data.joints.q_j:\n", data.joints.q_j)
            print("data.joints.dq_j:\n", data.joints.dq_j)

        # Run limit detection to generate active limits
        limits.detect(model, data)
        wp.synchronize()
        if self.verbose:
            print(f"limits.world_num_limits: {limits.world_num_limits}")
            print(f"data.info.num_limits: {data.info.num_limits}")

        # Run collision detection to generate active contacts
        detector.collide(model, data)
        wp.synchronize()
        if self.verbose:
            print(f"contacts.world_num_contacts: {detector.contacts.world_num_contacts}")
            print(f"data.info.num_contacts: {data.info.num_contacts}")

        # Update the constraints info
        update_constraints_info(model=model, data=data)
        if self.verbose:
            print("")  # Add a newline for better readability
            print_model_data_info(data)
        wp.synchronize()

        # Build the dense system Jacobians
        jacobians.build(model=model, data=data, limits=limits.data, contacts=detector.contacts.data)
        wp.synchronize()

        # Extract the Jacobian matrices
        extract_cts_jacobians(
            model=model, limits=limits, contacts=detector.contacts, jacobians=jacobians, verbose=self.verbose
        )
        extract_dofs_jacobians(model=model, jacobians=jacobians, verbose=self.verbose)
        # TODO: Add checks for the Jacobian matrices


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=1000, precision=6, threshold=10000, suppress=True)  # Suppress scientific notation

    # Global warp configurations
    wp.config.enable_backward = False
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
