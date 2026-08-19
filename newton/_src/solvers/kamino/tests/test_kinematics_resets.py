# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the `kamino.kinematics.resets` module"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.kinematics.joints import JointCorrectionMode, compute_joints_data
from newton._src.solvers.kamino._src.kinematics.resets import reset_joints_state_from_bodies_state
from newton._src.solvers.kamino._src.models.builders.testing import build_all_joints_test_model
from newton._src.solvers.kamino._src.solvers import ForwardKinematicsSolver
from newton._src.solvers.kamino._src.utils import logger as msg
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino.utils.sampling import (
    sample_actuator_coords,
    sample_actuator_velocities,
    sample_base_state,
)

###
# Utils
###

rtol = 1e-6
atol = 1e-5


def sample_base_state_wp(model: ModelKamino, rng: np.random.Generator):
    base_q_np, base_u_np = sample_base_state(model.size.num_worlds, rng, max_pos=0.02, max_angle=np.radians(5.0))
    base_q = wp.from_numpy(base_q_np[0], dtype=wp.transformf, device=model.device)
    base_u = wp.from_numpy(base_u_np[0], dtype=wp.spatial_vectorf, device=model.device)
    return base_q, base_u


def sample_actuator_state_wp(model: ModelKamino, rng: np.random.Generator):
    actuator_q_np = sample_actuator_coords(model, rng, max_pos=0.02, max_angle=np.radians(5.0))[0]
    actuator_u_np = sample_actuator_velocities(model, rng)[0]
    actuator_q = wp.from_numpy(actuator_q_np, dtype=wp.float32, device=model.device)
    actuator_u = wp.from_numpy(actuator_u_np, dtype=wp.float32, device=model.device)
    return actuator_q, actuator_u


def set_model_to_random_pose(
    test_case: unittest.TestCase,
    model: ModelKamino,
    rng: np.random.Generator,
):
    """
    Helper sampling a random valid pose & velocity for a model, setting the model
    into this pose with FK, and computing joint data as a post-processing.
    """
    # Sample random pose
    base_q, base_u = sample_base_state_wp(model, rng)
    actuator_q, actuator_u = sample_actuator_state_wp(model, rng)

    # Set the model into generated non-trivial pose using FK
    fk_solver = ForwardKinematicsSolver(model=model)
    data = model.data(unilateral_cts=False, joint_wrenches=False, device=model.device)
    fk_solver.run_fk_solve(
        actuators_q=actuator_q,
        actuators_u=actuator_u,
        base_q=base_q,
        base_u=base_u,
        bodies_q=data.bodies.q_i,
        bodies_u=data.bodies.u_i,
    )
    test_case.assertTrue(fk_solver.newton_success.numpy().sum() == model.size.num_worlds)

    # Evaluate joint state and check constraint residuals
    compute_joints_data(model=model, data=data, q_j_p=model.joints.q_j_0, correction=JointCorrectionMode.CONTINUOUS)
    np.testing.assert_allclose(data.joints.r_j.numpy(), 0.0, rtol=0, atol=atol)
    np.testing.assert_allclose(data.joints.dr_j.numpy(), 0.0, rtol=0, atol=atol)

    return data


###
# Tests
###


class TestJointBodyStateConversions(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True to enable verbose output
        self.progress = test_context.verbose  # Set to True to show progress bars during long tests
        self.seed = 42

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_reset_joint_states_from_body_state(self):
        """
        Validate reset_joints_state_from_bodies_state() against compute_joints_data()
        on a model with all joint types.
        """
        # Initialize rng
        rng = np.random.default_rng(self.seed)

        # Setup a model with all joint types
        builder = build_all_joints_test_model(binary_joints=True, unary_joints=False, actuated=True, floating_base=True)
        model = builder.finalize(device=self.default_device)

        # Set the model into a non-trivial pose
        data = set_model_to_random_pose(self, model, rng)

        # Compute joint states from bodies state
        state = model.state()
        wp.copy(state.q_i, data.bodies.q_i)
        wp.copy(state.u_i, data.bodies.u_i)
        all_worlds_mask = wp.ones(shape=model.size.num_worlds, dtype=wp.bool, device=model.device)
        reset_joints_state_from_bodies_state(model, state, world_mask=all_worlds_mask)

        # Compare against joint state in joint data
        # Note: both functions are correcting coords w.r.t. initial coords, so values are directly comparable
        np.testing.assert_allclose(state.q_j.numpy(), data.joints.q_j.numpy(), rtol=rtol, atol=atol)
        np.testing.assert_allclose(state.dq_j.numpy(), data.joints.dq_j.numpy(), rtol=rtol, atol=atol)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
