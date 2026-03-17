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

"""Unit tests for the `kamino.kinematics.joints` module"""

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.data import DataKamino
from newton._src.solvers.kamino._src.core.math import quat_exp, screw, screw_angular, screw_linear
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.core.types import float32, int32, mat33f, transformf, vec3f, vec6f
from newton._src.solvers.kamino._src.kinematics.joints import compute_joints_data
from newton._src.solvers.kamino._src.models.builders.testing import build_unary_revolute_joint_test
from newton._src.solvers.kamino._src.models.builders.utils import make_homogeneous_builder
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context

###
# Module configs
###

wp.set_module_options({"enable_backward": False})

###
# Constants
###

Q_X_J = 0.5 * math.pi
THETA_Y_J = 0.1
THETA_Z_J = -0.2
J_DR_J = vec3f(0.01, 0.02, 0.03)
J_DV_J = vec3f(0.1, -0.2, 0.3)
J_DOMEGA_J = vec3f(-1.0, 0.04, -0.05)

# Compute revolute joint rotational residual: sin(angle) * axis
ROT_RES_VEC = np.array([0.0, THETA_Y_J, THETA_Z_J])
ROT_RES_ANGLE = np.linalg.norm(ROT_RES_VEC)
ROT_RES = (np.sin(ROT_RES_ANGLE) / ROT_RES_ANGLE) * ROT_RES_VEC

###
# Kernels
###


@wp.kernel
def _set_joint_follower_body_state(
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
    bid_F = model_joint_bid_F[jid]
    B_r_Bj = model_joint_B_r_Bj[jid]
    F_r_Fj = model_joint_F_r_Fj[jid]
    X_j = model_joint_X_j[jid]

    # The base body is assumed to be at the origin with no rotation or twist
    p_B = transformf(vec3f(0.0), wp.quat_identity())
    u_B = vec6f(0.0)
    r_B = wp.transform_get_translation(p_B)
    q_B = wp.transform_get_rotation(p_B)
    R_B = wp.quat_to_matrix(q_B)
    v_B = screw_linear(u_B)
    omega_B = screw_angular(u_B)

    # Define the joint rotation offset
    # NOTE: X_j projects quantities into the joint frame
    # NOTE: X_j^T projects quantities into the outer frame (world or body)
    j_dR_yz_j = vec3f(0.0, THETA_Y_J, THETA_Z_J)  # Joint residual as rotation vector
    j_dR_x_j = vec3f(Q_X_J, 0.0, 0.0)  # Joint dof rotation as rotation vector
    q_jq = quat_exp(j_dR_yz_j) * quat_exp(j_dR_x_j)  # Total joint offset
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


def set_joint_follower_body_state(model: ModelKamino, data: DataKamino):
    wp.launch(
        _set_joint_follower_body_state,
        dim=model.size.sum_of_num_joints,
        inputs=[
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
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True to enable verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            msg.info("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_single_revolute_joint(self):
        # Construct the model description using the ModelBuilderKamino
        builder = build_unary_revolute_joint_test()

        # Create the model and state
        model = builder.finalize(device=self.default_device)
        data = model.data(device=self.default_device)

        # Set the state of the Follower body to a known state
        set_joint_follower_body_state(model, data)
        msg.info("data.bodies.q_i: %s", data.bodies.q_i)
        msg.info("data.bodies.u_i: %s", data.bodies.u_i)

        # Update the state of the joints
        compute_joints_data(model=model, data=data, q_j_p=wp.zeros_like(data.joints.q_j))
        msg.info("data.joints.p_j: %s", data.joints.p_j)

        # Extract joint data
        r_j_np = data.joints.r_j.numpy().copy()
        dr_j_np = data.joints.dr_j.numpy().copy()
        q_j_np = data.joints.q_j.numpy().copy()
        dq_j_np = data.joints.dq_j.numpy().copy()
        msg.info("[measured]:  r_j: %s", r_j_np)
        msg.info("[measured]: dr_j: %s", dr_j_np)
        msg.info("[measured]:  q_j: %s", q_j_np)
        msg.info("[measured]: dq_j: %s", dq_j_np)

        # Construct expected joint data
        r_j_expected = np.array([J_DR_J[0], J_DR_J[1], J_DR_J[2], ROT_RES[1], ROT_RES[2]], dtype=np.float32)
        dr_j_expected = np.array([J_DV_J[0], J_DV_J[1], J_DV_J[2], J_DOMEGA_J[1], J_DOMEGA_J[2]], dtype=np.float32)
        q_j_expected = np.array([Q_X_J], dtype=np.float32)
        dq_j_expected = np.array([J_DOMEGA_J[0]], dtype=np.float32)
        msg.info("[expected]:  r_j: %s", r_j_expected)
        msg.info("[expected]: dr_j: %s", dr_j_expected)
        msg.info("[expected]:  q_j: %s", q_j_expected)
        msg.info("[expected]: dq_j: %s", dq_j_expected)

        # Check the joint state values
        np.testing.assert_almost_equal(r_j_np, r_j_expected)
        np.testing.assert_almost_equal(dr_j_np, dr_j_expected)
        np.testing.assert_almost_equal(q_j_np, q_j_expected)
        np.testing.assert_almost_equal(dq_j_np, dq_j_expected)

    def test_02_multiple_revolute_joints(self):
        # Construct the model description using the ModelBuilderKamino
        builder = make_homogeneous_builder(num_worlds=4, build_fn=build_unary_revolute_joint_test)

        # Create the model and state
        model = builder.finalize(device=self.default_device)
        data = model.data(device=self.default_device)

        # Set the state of the Follower body to a known state
        set_joint_follower_body_state(model, data)
        msg.info("data.bodies.q_i:\n%s", data.bodies.q_i)
        msg.info("data.bodies.u_i:\n%s", data.bodies.u_i)

        # Update the state of the joints
        compute_joints_data(model=model, data=data, q_j_p=wp.zeros_like(data.joints.q_j))
        msg.info("data.joints.p_j: %s", data.joints.p_j)

        # Extract joint data
        r_j_np = data.joints.r_j.numpy().copy()
        dr_j_np = data.joints.dr_j.numpy().copy()
        q_j_np = data.joints.q_j.numpy().copy()
        dq_j_np = data.joints.dq_j.numpy().copy()
        msg.info("[measured]:  r_j: %s", r_j_np)
        msg.info("[measured]: dr_j: %s", dr_j_np)
        msg.info("[measured]:  q_j: %s", q_j_np)
        msg.info("[measured]: dq_j: %s", dq_j_np)

        # Construct expected joint data
        r_j_expected = np.array([J_DR_J[0], J_DR_J[1], J_DR_J[2], ROT_RES[1], ROT_RES[2]], dtype=np.float32)
        dr_j_expected = np.array([J_DV_J[0], J_DV_J[1], J_DV_J[2], J_DOMEGA_J[1], J_DOMEGA_J[2]], dtype=np.float32)
        q_j_expected = np.array([Q_X_J], dtype=np.float32)
        dq_j_expected = np.array([J_DOMEGA_J[0]], dtype=np.float32)

        # Tile expected values for all joints
        r_j_expected = np.tile(r_j_expected, builder.num_worlds)
        dr_j_expected = np.tile(dr_j_expected, builder.num_worlds)
        q_j_expected = np.tile(q_j_expected, builder.num_worlds)
        dq_j_expected = np.tile(dq_j_expected, builder.num_worlds)
        msg.info("[expected]:  r_j: %s", r_j_expected)
        msg.info("[expected]: dr_j: %s", dr_j_expected)
        msg.info("[expected]:  q_j: %s", q_j_expected)
        msg.info("[expected]: dq_j: %s", dq_j_expected)

        # Check the joint state values
        np.testing.assert_almost_equal(r_j_np, r_j_expected)
        np.testing.assert_almost_equal(dr_j_np, dr_j_expected)
        np.testing.assert_almost_equal(q_j_np, q_j_expected)
        np.testing.assert_almost_equal(dq_j_np, dq_j_expected)

    def test_03_single_dynamic_revolute_joint(self):
        # Construct the model description using the ModelBuilderKamino
        builder = build_unary_revolute_joint_test(dynamic=True, implicit_pd=True)

        # Create the model and state
        model = builder.finalize(device=self.default_device)
        data = model.data(device=self.default_device)
        model.time.set_uniform_timestep(0.01)

        # Optionally print model parameters for debugging
        msg.info("model.time.dt: %s", model.time.dt)
        msg.info("model.joints.a_j: %s", model.joints.a_j)
        msg.info("model.joints.b_j: %s", model.joints.b_j)
        msg.info("model.joints.k_p_j: %s", model.joints.k_p_j)
        msg.info("model.joints.k_d_j: %s\n", model.joints.k_d_j)
        msg.info("model.joints.num_cts: %s", model.joints.num_cts)
        msg.info("model.joints.num_dynamic_cts: %s", model.joints.num_dynamic_cts)
        msg.info("model.joints.num_kinematic_cts: %s", model.joints.num_kinematic_cts)
        msg.info("model.joints.dynamic_cts_offset: %s", model.joints.dynamic_cts_offset)
        msg.info("model.joints.kinematic_cts_offset: %s\n", model.joints.kinematic_cts_offset)
        msg.info("model.info.num_joint_dynamic_cts: %s", model.info.num_joint_dynamic_cts)
        msg.info("model.info.joint_dynamic_cts_offset: %s\n", model.info.joint_dynamic_cts_offset)

        # Set the state of the Follower body to a known state
        set_joint_follower_body_state(model, data)
        msg.info("data.bodies.q_i: %s", data.bodies.q_i)
        msg.info("data.bodies.u_i: %s\n", data.bodies.u_i)

        # Update the state of the joints
        compute_joints_data(model=model, data=data, q_j_p=wp.zeros_like(data.joints.q_j))
        msg.info("data.joints.p_j: %s\n", data.joints.p_j)

        # Extract measured joint data
        r_j_np = data.joints.r_j.numpy().copy()
        dr_j_np = data.joints.dr_j.numpy().copy()
        q_j_np = data.joints.q_j.numpy().copy()
        dq_j_np = data.joints.dq_j.numpy().copy()
        m_j_np = data.joints.m_j.numpy().copy()
        inv_m_j_np = data.joints.inv_m_j.numpy().copy()
        dq_b_j_np = data.joints.dq_b_j.numpy().copy()
        q_j_ref_np = data.joints.q_j_ref.numpy().copy()
        dq_j_ref_np = data.joints.dq_j_ref.numpy().copy()
        tau_j_ref_np = data.joints.tau_j_ref.numpy().copy()
        msg.info("[measured]:  r_j: %s", r_j_np)
        msg.info("[measured]: dr_j: %s", dr_j_np)
        msg.info("[measured]:  q_j: %s", q_j_np)
        msg.info("[measured]: dq_j: %s\n", dq_j_np)
        msg.info("[measured]: m_j: %s", m_j_np)
        msg.info("[measured]: inv_m_j: %s", inv_m_j_np)
        msg.info("[measured]: dq_b_j: %s\n", dq_b_j_np)
        msg.info("[measured]: q_j_ref: %s", q_j_ref_np)
        msg.info("[measured]: dq_j_ref: %s\n", dq_j_ref_np)
        msg.info("[measured]: tau_j_ref: %s\n", tau_j_ref_np)

        # Compute expected joint dynamics values based on the PD control
        # law and the equations of motion for a single revolute joint
        dt = model.time.dt.numpy().copy()[0]
        a_j_np = model.joints.a_j.numpy().copy()
        b_j_np = model.joints.b_j.numpy().copy()
        k_p_j_np = model.joints.k_p_j.numpy().copy()
        k_d_j_np = model.joints.k_d_j.numpy().copy()
        m_j_exp_val = a_j_np[0] + dt * (b_j_np[0] + k_d_j_np[0]) + dt * dt * k_p_j_np[0]
        inv_m_j_exp_val = 1.0 / m_j_exp_val
        tau_j_exp_val = tau_j_ref_np[0] + k_p_j_np[0] * (q_j_ref_np[0] - q_j_np[0]) + k_d_j_np[0] * dq_j_ref_np[0]
        h_j_exp_val = a_j_np[0] * dq_j_np[0] + dt * tau_j_exp_val
        dq_b_j_exp_val = inv_m_j_exp_val * h_j_exp_val

        # Construct expected joint data
        r_j_expected = np.array([J_DR_J[0], J_DR_J[1], J_DR_J[2], ROT_RES[1], ROT_RES[2]], dtype=np.float32)
        dr_j_expected = np.array([J_DV_J[0], J_DV_J[1], J_DV_J[2], J_DOMEGA_J[1], J_DOMEGA_J[2]], dtype=np.float32)
        q_j_expected = np.array([Q_X_J], dtype=np.float32)
        dq_j_expected = np.array([J_DOMEGA_J[0]], dtype=np.float32)
        m_j_expected = np.array([m_j_exp_val], dtype=np.float32)
        tau_j_expected = np.array([tau_j_exp_val], dtype=np.float32)
        h_j_expected = np.array([h_j_exp_val], dtype=np.float32)
        inv_m_j_expected = np.array([inv_m_j_exp_val], dtype=np.float32)
        dq_b_j_expected = np.array([dq_b_j_exp_val], dtype=np.float32)
        msg.info("[expected]:  r_j: %s", r_j_expected)
        msg.info("[expected]: dr_j: %s", dr_j_expected)
        msg.info("[expected]:  q_j: %s", q_j_expected)
        msg.info("[expected]: dq_j: %s\n", dq_j_expected)
        msg.info("[expected]: m_j: %s", m_j_expected)
        msg.info("[expected]: tau_j: %s", tau_j_expected)
        msg.info("[expected]: h_j: %s", h_j_expected)
        msg.info("[expected]: inv_m_j: %s", inv_m_j_expected)
        msg.info("[expected]: dq_b_j: %s\n", dq_b_j_expected)

        # Check the joint data values
        np.testing.assert_almost_equal(r_j_np, r_j_expected)
        np.testing.assert_almost_equal(dr_j_np, dr_j_expected)
        np.testing.assert_almost_equal(q_j_np, q_j_expected)
        np.testing.assert_almost_equal(dq_j_np, dq_j_expected)
        np.testing.assert_almost_equal(m_j_np, m_j_expected)
        np.testing.assert_almost_equal(inv_m_j_np, inv_m_j_expected)
        np.testing.assert_almost_equal(dq_b_j_np, dq_b_j_expected)

    def test_04_multiple_dynamic_revolute_joints(self):
        # Construct the model description using the ModelBuilderKamino
        builder = make_homogeneous_builder(
            num_worlds=4, build_fn=build_unary_revolute_joint_test, dynamic=True, implicit_pd=True
        )

        # Create the model and data
        model = builder.finalize(device=self.default_device)
        data = model.data(device=self.default_device)
        model.time.set_uniform_timestep(0.01)

        # Optionally print model parameters for debugging
        msg.info("model.time.dt: %s", model.time.dt)
        msg.info("model.joints.a_j: %s", model.joints.a_j)
        msg.info("model.joints.b_j: %s", model.joints.b_j)
        msg.info("model.joints.k_p_j: %s", model.joints.k_p_j)
        msg.info("model.joints.k_d_j: %s\n", model.joints.k_d_j)
        msg.info("model.joints.num_cts: %s", model.joints.num_cts)
        msg.info("model.joints.num_dynamic_cts: %s", model.joints.num_dynamic_cts)
        msg.info("model.joints.num_kinematic_cts: %s", model.joints.num_kinematic_cts)
        msg.info("model.joints.dynamic_cts_offset: %s", model.joints.dynamic_cts_offset)
        msg.info("model.joints.kinematic_cts_offset: %s\n", model.joints.kinematic_cts_offset)
        msg.info("model.info.num_joint_dynamic_cts: %s", model.info.num_joint_dynamic_cts)
        msg.info("model.info.joint_dynamic_cts_offset: %s\n", model.info.joint_dynamic_cts_offset)

        # Set the state of the Follower body to a known state
        set_joint_follower_body_state(model, data)
        msg.info("data.bodies.q_i:\n%s", data.bodies.q_i)
        msg.info("data.bodies.u_i:\n%s\n", data.bodies.u_i)

        # Update the state of the joints
        compute_joints_data(model=model, data=data, q_j_p=wp.zeros_like(data.joints.q_j))
        msg.info("data.joints.p_j:\n%s", data.joints.p_j)

        # Extract measured joint data
        r_j_np = data.joints.r_j.numpy().copy()
        dr_j_np = data.joints.dr_j.numpy().copy()
        q_j_np = data.joints.q_j.numpy().copy()
        dq_j_np = data.joints.dq_j.numpy().copy()
        m_j_np = data.joints.m_j.numpy().copy()
        inv_m_j_np = data.joints.inv_m_j.numpy().copy()
        dq_b_j_np = data.joints.dq_b_j.numpy().copy()
        q_j_ref_np = data.joints.q_j_ref.numpy().copy()
        dq_j_ref_np = data.joints.dq_j_ref.numpy().copy()
        tau_j_ref_np = data.joints.tau_j_ref.numpy().copy()
        msg.info("[measured]:  r_j: %s", r_j_np)
        msg.info("[measured]: dr_j: %s", dr_j_np)
        msg.info("[measured]:  q_j: %s", q_j_np)
        msg.info("[measured]: dq_j: %s\n", dq_j_np)
        msg.info("[measured]: m_j: %s", m_j_np)
        msg.info("[measured]: inv_m_j: %s", inv_m_j_np)
        msg.info("[measured]: dq_b_j: %s\n", dq_b_j_np)
        msg.info("[measured]: q_j_ref: %s", q_j_ref_np)
        msg.info("[measured]: dq_j_ref: %s\n", dq_j_ref_np)
        msg.info("[measured]: tau_j_ref: %s\n", tau_j_ref_np)

        # Compute expected joint dynamics values based on the PD control
        # law and the equations of motion for a single revolute joint
        dt = model.time.dt.numpy().copy()[0]
        a_j_np = model.joints.a_j.numpy().copy()
        b_j_np = model.joints.b_j.numpy().copy()
        k_p_j_np = model.joints.k_p_j.numpy().copy()
        k_d_j_np = model.joints.k_d_j.numpy().copy()
        m_j_exp_val = a_j_np[0] + dt * (b_j_np[0] + k_d_j_np[0]) + dt * dt * k_p_j_np[0]
        inv_m_j_exp_val = 1.0 / m_j_exp_val
        tau_j_exp_val = tau_j_ref_np[0] + k_p_j_np[0] * (q_j_ref_np[0] - q_j_np[0]) + k_d_j_np[0] * dq_j_ref_np[0]
        h_j_exp_val = a_j_np[0] * dq_j_np[0] + dt * tau_j_exp_val
        dq_b_j_exp_val = inv_m_j_exp_val * h_j_exp_val

        # Construct expected joint data
        r_j_expected = np.array([J_DR_J[0], J_DR_J[1], J_DR_J[2], ROT_RES[1], ROT_RES[2]], dtype=np.float32)
        dr_j_expected = np.array([J_DV_J[0], J_DV_J[1], J_DV_J[2], J_DOMEGA_J[1], J_DOMEGA_J[2]], dtype=np.float32)
        q_j_expected = np.array([Q_X_J], dtype=np.float32)
        dq_j_expected = np.array([J_DOMEGA_J[0]], dtype=np.float32)
        m_j_expected = np.array([m_j_exp_val], dtype=np.float32)
        h_j_expected = np.array([h_j_exp_val], dtype=np.float32)
        inv_m_j_expected = np.array([inv_m_j_exp_val], dtype=np.float32)
        dq_b_j_expected = np.array([dq_b_j_exp_val], dtype=np.float32)

        # Tile expected values for all joints
        r_j_expected = np.tile(r_j_expected, builder.num_worlds)
        dr_j_expected = np.tile(dr_j_expected, builder.num_worlds)
        q_j_expected = np.tile(q_j_expected, builder.num_worlds)
        dq_j_expected = np.tile(dq_j_expected, builder.num_worlds)
        m_j_expected = np.tile(m_j_expected, builder.num_worlds)
        h_j_expected = np.tile(h_j_expected, builder.num_worlds)
        inv_m_j_expected = np.tile(inv_m_j_expected, builder.num_worlds)
        dq_b_j_expected = np.tile(dq_b_j_expected, builder.num_worlds)
        msg.info("[expected]:  r_j: %s", r_j_expected)
        msg.info("[expected]: dr_j: %s", dr_j_expected)
        msg.info("[expected]:  q_j: %s", q_j_expected)
        msg.info("[expected]: dq_j: %s\n", dq_j_expected)
        msg.info("[expected]: m_j: %s", m_j_expected)
        msg.info("[expected]: h_j: %s", h_j_expected)
        msg.info("[expected]: inv_m_j: %s", inv_m_j_expected)
        msg.info("[expected]: dq_b_j: %s\n", dq_b_j_expected)

        # Check the joint data values
        np.testing.assert_almost_equal(r_j_np, r_j_expected)
        np.testing.assert_almost_equal(dr_j_np, dr_j_expected)
        np.testing.assert_almost_equal(q_j_np, q_j_expected)
        np.testing.assert_almost_equal(dq_j_np, dq_j_expected)
        np.testing.assert_almost_equal(m_j_np, m_j_expected)
        np.testing.assert_almost_equal(inv_m_j_np, inv_m_j_expected)
        np.testing.assert_almost_equal(dq_b_j_np, dq_b_j_expected)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
