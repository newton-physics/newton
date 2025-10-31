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
KAMINO: UNIT TESTS: KINEMATICS: JOINTS
"""

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.core.math import quat_exp, screw, screw_angular, screw_linear
from newton._src.solvers.kamino.core.model import Model, ModelData
from newton._src.solvers.kamino.core.types import float32, int32, mat33f, transformf, vec3f, vec6f

# Module to be tested
from newton._src.solvers.kamino.kinematics.joints import compute_joints_state
from newton._src.solvers.kamino.models.builders import build_revolute_joint_test_system
from newton._src.solvers.kamino.models.utils import make_homogeneous_builder, make_single_builder

###
# Module configs
###

wp.set_module_options({"enable_backward": False})

###
# Constants
###

Q_X_J = 0.5 * math.pi
THETA_Y_J = 0.1
THETA_Z_J = -0.1
J_DR_J = vec3f(0.01, 0.02, 0.03)
J_DV_J = vec3f(0.1, -0.2, 0.3)
J_DOMEGA_J = vec3f(-1.0, 0.04, -0.05)


###
# Kernels
###


@wp.kernel
def _set_joint_follower_body_state(
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
    q_x_j = Q_X_J
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


def set_joint_follower_body_state(model: Model, data: ModelData):
    wp.launch(
        _set_joint_follower_body_state,
        dim=model.size.sum_of_num_joints,
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


class TestKinematicsJoints(unittest.TestCase):
    def setUp(self):
        self.verbose = False  # Set to True for verbose output
        self.default_device = wp.get_device()

    def tearDown(self):
        self.default_device = None

    def test_01_single_revolute_joint(self):
        # Construct the model description using the ModelBuilder
        builder, _, _ = make_single_builder(build_func=build_revolute_joint_test_system)

        # Create the model and state
        model = builder.finalize(device=self.default_device)
        data = model.data(device=self.default_device)

        # Set the state of the Follower body to a known state
        set_joint_follower_body_state(model, data)
        if self.verbose:
            print("data.bodies.q_i:\n", data.bodies.q_i)
            print("data.bodies.u_i:\n", data.bodies.u_i)

        # Update the state of the joints
        compute_joints_state(model=model, q_j_p=wp.zeros_like(data.joints.q_j), data=data)
        if self.verbose:
            print("data.joints.p_j:\n", data.joints.p_j)
            print("data.joints.r_j:\n", data.joints.r_j)
            print("data.joints.dr_j:\n", data.joints.dr_j)
            print("data.joints.q_j:\n", data.joints.q_j)
            print("data.joints.dq_j:\n", data.joints.dq_j)

        # Check the joint state values
        r_j_np = data.joints.r_j.numpy()
        dr_j_np = data.joints.dr_j.numpy()
        q_j_np = data.joints.q_j.numpy()
        dq_j_np = data.joints.dq_j.numpy()
        self.assertAlmostEqual(q_j_np[0], Q_X_J, places=6)
        self.assertAlmostEqual(r_j_np[0], J_DR_J[0], places=6)
        self.assertAlmostEqual(r_j_np[1], J_DR_J[1], places=6)
        self.assertAlmostEqual(r_j_np[2], J_DR_J[2], places=6)
        self.assertAlmostEqual(r_j_np[3], THETA_Y_J, places=6)
        self.assertAlmostEqual(r_j_np[4], THETA_Z_J, places=6)
        self.assertAlmostEqual(dq_j_np[0], J_DOMEGA_J[0], places=6)
        self.assertAlmostEqual(dr_j_np[0], J_DV_J[0], places=6)
        self.assertAlmostEqual(dr_j_np[1], J_DV_J[1], places=6)
        self.assertAlmostEqual(dr_j_np[2], J_DV_J[2], places=6)
        self.assertAlmostEqual(dr_j_np[3], J_DOMEGA_J[1], places=6)
        self.assertAlmostEqual(dr_j_np[4], J_DOMEGA_J[2], places=6)

    def test_02_multiple_revolute_joints(self):
        # Construct the model description using the ModelBuilder
        builder = make_homogeneous_builder(num_worlds=4, build_func=build_revolute_joint_test_system)

        # Create the model and state
        model = builder.finalize(device=self.default_device)
        data = model.data(device=self.default_device)

        # Set the state of the Follower body to a known state
        set_joint_follower_body_state(model, data)
        if self.verbose:
            print("data.bodies.q_i:\n", data.bodies.q_i)
            print("data.bodies.u_i:\n", data.bodies.u_i)

        # Update the state of the joints
        compute_joints_state(model=model, q_j_p=wp.zeros_like(data.joints.q_j), data=data)
        if self.verbose:
            print("data.joints.p_j:\n", data.joints.p_j)
            print("data.joints.r_j:\n", data.joints.r_j)
            print("data.joints.dr_j:\n", data.joints.dr_j)
            print("data.joints.q_j:\n", data.joints.q_j)
            print("data.joints.dq_j:\n", data.joints.dq_j)

        # Check the joint state values
        nj = model.size.sum_of_num_joints
        cio_j_np = model.joints.cts_offset.numpy()
        r_j_np = data.joints.r_j.numpy()
        dr_j_np = data.joints.dr_j.numpy()
        dio_j_np = model.joints.dofs_offset.numpy()
        q_j_np = data.joints.q_j.numpy()
        dq_j_np = data.joints.dq_j.numpy()
        for j in range(nj):
            self.assertAlmostEqual(q_j_np[dio_j_np[j] + 0], Q_X_J, places=6)
            self.assertAlmostEqual(r_j_np[cio_j_np[j] + 0], J_DR_J[0], places=6)
            self.assertAlmostEqual(r_j_np[cio_j_np[j] + 1], J_DR_J[1], places=6)
            self.assertAlmostEqual(r_j_np[cio_j_np[j] + 2], J_DR_J[2], places=6)
            self.assertAlmostEqual(r_j_np[cio_j_np[j] + 3], THETA_Y_J, places=6)
            self.assertAlmostEqual(r_j_np[cio_j_np[j] + 4], THETA_Z_J, places=6)
            self.assertAlmostEqual(dq_j_np[dio_j_np[j] + 0], J_DOMEGA_J[0], places=6)
            self.assertAlmostEqual(dr_j_np[cio_j_np[j] + 0], J_DV_J[0], places=6)
            self.assertAlmostEqual(dr_j_np[cio_j_np[j] + 1], J_DV_J[1], places=6)
            self.assertAlmostEqual(dr_j_np[cio_j_np[j] + 2], J_DV_J[2], places=6)
            self.assertAlmostEqual(dr_j_np[cio_j_np[j] + 3], J_DOMEGA_J[1], places=6)
            self.assertAlmostEqual(dr_j_np[cio_j_np[j] + 4], J_DOMEGA_J[2], places=6)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=500, precision=10, suppress=True)  # Suppress scientific notation

    # Global warp configurations
    wp.config.enable_backward = False
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
