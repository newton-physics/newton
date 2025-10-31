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
KAMINO: UNIT TESTS: KINEMATICS: LIMITS
"""

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.core.math import quat_exp, screw, screw_angular, screw_linear
from newton._src.solvers.kamino.core.model import Model, ModelData
from newton._src.solvers.kamino.core.types import float32, int32, mat33f, transformf, vec3f, vec6f
from newton._src.solvers.kamino.kinematics.joints import compute_joints_state

# Module to be tested
from newton._src.solvers.kamino.kinematics.limits import Limits
from newton._src.solvers.kamino.models.builders import (
    build_boxes_fourbar,
    build_revolute_joint_test_system,
)
from newton._src.solvers.kamino.models.utils import make_homogeneous_builder

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###

Q_X_J = 0.3 * math.pi
Q_X_J_MAX = 0.25 * math.pi


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
    theta_y_j = 0.0
    theta_z_j = 0.0
    j_dR_j = vec3f(q_x_j, theta_y_j, theta_z_j)  # Joint offset as rotation vector
    q_jq = quat_exp(j_dR_j)  # Joint offset as rotation quaternion
    R_jq = wp.quat_to_matrix(q_jq)  # Joint offset as rotation matrix

    # Define the joint translation offset
    j_dr_j = vec3f(0.0)

    # Define the joint twist offset
    j_dv_j = vec3f(0.0)
    j_domega_j = vec3f(0.0)

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


class TestKinematicsLimits(unittest.TestCase):
    def setUp(self):
        self.verbose = False  # Set to True for detailed output
        self.default_device = wp.get_device()

    def tearDown(self):
        self.default_device = None

    def test_01_create_empty_limits_container(self):
        """
        Tests the creation of an empty Limits container (for deferred allocation).
        """
        # Create a Limits container
        limits = Limits(device=self.default_device)

        # Check the initial state of the limits
        self.assertIsNone(limits.model_max_limits)
        self.assertIsNone(limits.model_num_limits)
        self.assertIsNone(limits.world_max_limits)
        self.assertIsNone(limits.world_max_limits)
        self.assertIsNone(limits.wid)
        self.assertIsNone(limits.lid)
        self.assertIsNone(limits.jid)
        self.assertIsNone(limits.bids)
        self.assertIsNone(limits.dof)
        self.assertIsNone(limits.side)
        self.assertIsNone(limits.r_q)
        self.assertIsNone(limits.r_dq)
        self.assertIsNone(limits.r_tau)

    def test_02_allocate_limits_container_from_homogeneous_builder(self):
        """
        Tests the allocation of a Limits container.
        """
        # Construct the model description using the ModelBuilder
        builder = make_homogeneous_builder(num_worlds=3, build_func=build_boxes_fourbar)

        # Create a Limits container
        limits = Limits(builder=builder, device=self.default_device)

        # Check the initial state of the limits
        self.assertIsNotNone(limits.model_max_limits)
        self.assertIsNotNone(limits.model_num_limits)
        self.assertIsNotNone(limits.world_max_limits)
        self.assertIsNotNone(limits.world_max_limits)
        self.assertIsNotNone(limits.wid)
        self.assertIsNotNone(limits.lid)
        self.assertIsNotNone(limits.jid)
        self.assertIsNotNone(limits.bids)
        self.assertIsNotNone(limits.dof)
        self.assertIsNotNone(limits.side)
        self.assertIsNotNone(limits.r_q)
        self.assertIsNotNone(limits.r_dq)
        self.assertIsNotNone(limits.r_tau)

        # Check the shapes of the limits arrays
        self.assertEqual(limits.num_model_max_limits, 12)
        self.assertEqual(limits.num_world_max_limits, [4, 4, 4])
        self.assertEqual(limits.model_num_limits.shape, (1,))
        self.assertEqual(limits.model_num_limits.shape, (1,))
        self.assertEqual(limits.world_max_limits.shape, (3,))
        self.assertEqual(limits.world_num_limits.shape, (3,))

        # Optional verbose output
        if self.verbose:
            print("limits.num_model_max_limits:", limits.num_model_max_limits)
            print("limits.num_world_max_limits:", limits.num_world_max_limits)
            print("limits.model_max_limits:", limits.model_max_limits)
            print("limits.model_num_limits:", limits.model_num_limits)
            print("limits.world_max_limits:", limits.world_max_limits)
            print("limits.world_num_limits:", limits.world_num_limits)
            print("limits.wid:", limits.wid)
            print("limits.lid:", limits.lid)
            print("limits.jid:", limits.jid)
            print("limits.bids:\n", limits.bids)
            print("limits.dof:", limits.dof)
            print("limits.side:", limits.side)
            print("limits.r_q:", limits.r_q)
            print("limits.r_dq:", limits.r_dq)
            print("limits.r_tau:", limits.r_tau)

    def test_03_check_revolute_joint(self):
        # Construct the model description using the ModelBuilder
        builder = make_homogeneous_builder(num_worlds=4, build_func=build_revolute_joint_test_system)
        num_worlds = builder.num_worlds

        # Create the model and state
        model = builder.finalize(device=self.default_device)
        data = model.data(device=self.default_device)

        # Set the state of the Follower body to a known state
        set_joint_follower_body_state(model, data)

        # Update the state of the joints
        compute_joints_state(model=model, q_j_p=wp.zeros_like(data.joints.q_j), data=data)

        # Optional verbose output
        if self.verbose:
            print("model.joints.q_j_min: ", model.joints.q_j_min)
            print("model.joints.q_j_max: ", model.joints.q_j_max)
            print("model.joints.dq_j_max: ", model.joints.dq_j_max)
            print("model.joints.tau_j_max: ", model.joints.tau_j_max)
            print("data.bodies.q_i:\n", data.bodies.q_i)
            print("data.bodies.u_i:\n", data.bodies.u_i)
            print("data.joints.p_j:\n", data.joints.p_j)
            print("data.joints.r_j: ", data.joints.r_j)
            print("data.joints.dr_j: ", data.joints.dr_j)
            print("data.joints.q_j: ", data.joints.q_j)
            print("data.joints.dq_j: ", data.joints.dq_j)

        # Create a Limits container
        limits = Limits(builder=builder, device=self.default_device)

        # Optional verbose output
        if self.verbose:
            print("[before]: limits.num_model_max_limits:", limits.num_model_max_limits)
            print("[before]: limits.num_world_max_limits:", limits.num_world_max_limits)
            print("[before]: limits.model_max_limits:", limits.model_max_limits)
            print("[before]: limits.model_num_limits:", limits.model_num_limits)
            print("[before]: limits.world_max_limits:", limits.world_max_limits)
            print("[before]: limits.world_num_limits:", limits.world_num_limits)
            print("[before]: limits.wid:", limits.wid)
            print("[before]: limits.lid:", limits.lid)
            print("[before]: limits.jid:", limits.jid)
            print("[before]: limits.bids:\n", limits.bids)
            print("[before]: limits.dof:", limits.dof)
            print("[before]: limits.side:", limits.side)
            print("[before]: limits.r_q:", limits.r_q)
            print("[before]: limits.r_dq:", limits.r_dq)
            print("[before]: limits.r_tau:", limits.r_tau)

        # Check for active joint limits
        limits.detect(model, data)

        # Optional verbose output
        if self.verbose:
            print("[after]: limits.num_model_max_limits:", limits.num_model_max_limits)
            print("[after]: limits.num_world_max_limits:", limits.num_world_max_limits)
            print("[after]: limits.model_max_limits:", limits.model_max_limits)
            print("[after]: limits.model_num_limits:", limits.model_num_limits)
            print("[after]: limits.world_max_limits:", limits.world_max_limits)
            print("[after]: limits.world_num_limits:", limits.world_num_limits)
            print("[after]: limits.wid:", limits.wid)
            print("[after]: limits.lid:", limits.lid)
            print("[after]: limits.jid:", limits.jid)
            print("[after]: limits.bids:\n", limits.bids)
            print("[after]: limits.dof:", limits.dof)
            print("[after]: limits.side:", limits.side)
            print("[after]: limits.r_q:", limits.r_q)
            print("[after]: limits.r_dq:", limits.r_dq)
            print("[after]: limits.r_tau:", limits.r_tau)

        # Check the limits
        limits_num_np = limits.world_num_limits.numpy()
        limits_wid_np = limits.wid.numpy()
        limits_lid_np = limits.lid.numpy()
        limits_jid_np = limits.jid.numpy()
        limits_dof_np = limits.dof.numpy()
        limits_side_np = limits.side.numpy()
        limits_r_q_np = limits.r_q.numpy()
        for i in range(num_worlds):
            # Check the number of limits for this world
            self.assertEqual(limits_num_np[i], 1)
            for j in range(limits_num_np[i]):
                # Check the limits for this world
                self.assertEqual(limits_wid_np[i], i)
                self.assertEqual(limits_lid_np[i], j)
                self.assertEqual(limits_jid_np[i], i * limits_num_np[i] + j)
                self.assertEqual(limits_dof_np[i], j)
                self.assertEqual(limits_side_np[i], -1)
                self.assertAlmostEqual(limits_r_q_np[i * limits_num_np[i] + j], Q_X_J_MAX - Q_X_J, places=6)


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
