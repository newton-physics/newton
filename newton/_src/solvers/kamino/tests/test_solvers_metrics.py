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

"""Unit tests for `solvers/metrics.py`."""

import os
import unittest

import numpy as np
import warp as wp
from warp.context import Devicelike

from newton._src.solvers.kamino.integrators.euler import integrate_semi_implicit_euler
from newton._src.solvers.kamino.models.builders import (
    build_box_on_plane,
    build_boxes_hinged,
)
from newton._src.solvers.kamino.solvers.metrics import SolutionMetrics
from newton._src.solvers.kamino.solvers.padmm import PADMMSolver
from newton._src.solvers.kamino.tests.test_solvers_padmm import TestSetup
from newton._src.solvers.kamino.utils import logger as msg

###
# Tests
###


class TestSolverMetrics(unittest.TestCase):
    def setUp(self):
        self.default_device: Devicelike = wp.get_device()
        self.verbose = False  # Set to True for detailed output
        self.savefig = False  # Set to True to generate solver info plots
        self.output_dir = os.path.dirname(os.path.realpath(__file__)) + "/output/test_solvers_padmm"

        # Create output directory if saving figures
        if self.savefig:
            os.makedirs(self.output_dir, exist_ok=True)

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.set_log_level(msg.LogLevel.WARNING)

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_00_make_default(self):
        """
        Test creating a SolutionMetrics instance with default initialization.
        """
        # Creating a default solver metrics evaluator without any model
        # should result in an instance without any memory allocation.
        metrics = SolutionMetrics()
        self.assertIsNone(metrics._device)
        self.assertIsNone(metrics._data)
        self.assertIsNone(metrics._buffer_s)
        self.assertIsNone(metrics._buffer_v)

        # Requesting the solver data container when the
        # solver has not been finalized should raise an
        # error since no allocations have been made.
        self.assertRaises(RuntimeError, lambda: metrics.data)

    def test_01_finalize_default(self):
        """
        Test creating a SolutionMetrics instance with default initialization and then finalizing all memory allocations.
        """
        # Create a test setup
        test = TestSetup(builder_fn=build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Creating a default solver metrics evaluator without any model
        # should result in an instance without any memory allocation.
        metrics = SolutionMetrics()

        # Finalize the solver with a model
        metrics.finalize(test.model)

        # Check that the solver has been properly allocated
        self.assertIsNotNone(metrics._data)
        self.assertIsNotNone(metrics._device)
        self.assertIs(metrics._device, test.model.device)
        self.assertIsNotNone(metrics._buffer_s)
        self.assertIsNotNone(metrics._buffer_v)

        # Check allocation sizes
        msg.info("num_worlds: %s", test.model.size.num_worlds)
        msg.info("sum_of_max_total_cts: %s", test.model.size.sum_of_max_total_cts)
        msg.info("buffer_s size: %s", metrics._buffer_s.size)
        msg.info("buffer_v size: %s", metrics._buffer_v.size)
        self.assertEqual(metrics._buffer_s.size, test.model.size.sum_of_max_total_cts)
        self.assertEqual(metrics._buffer_v.size, test.model.size.sum_of_max_total_cts)
        self.assertEqual(metrics.data.r_eom.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_eom_argmax.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_kinematics.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_kinematics_argmax.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_cts_joints.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_cts_joints_argmax.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_cts_limits.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_cts_limits_argmax.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_cts_contacts.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_cts_contacts_argmax.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_v_plus.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_v_plus_argmax.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_ncp_primal.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_ncp_primal_argmax.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_ncp_dual.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_ncp_dual_argmax.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_ncp_compl.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_ncp_compl_argmax.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_vi_natmap.size, test.model.size.num_worlds)
        self.assertEqual(metrics.data.r_vi_natmap_argmax.size, test.model.size.num_worlds)

    def test_02_evaluate_trivial_solution(self):
        """
        Tests evaluating metrics on an all-zeros trivial solution.
        """
        # Create the test problem
        test = TestSetup(
            builder_fn=build_box_on_plane,
            max_world_contacts=4,
            gravity=False,
            perturb=False,
            device=self.default_device,
        )

        # Creating a default solver metrics evaluator from the test model
        metrics = SolutionMetrics(model=test.model)

        # Define a trivial solution (all zeros)
        with wp.ScopedDevice(test.model.device):
            sigma = wp.zeros(test.model.size.num_worlds, dtype=wp.vec2f)
            lambdas = wp.zeros(test.model.size.sum_of_max_total_cts, dtype=wp.float32)
            v_plus = wp.zeros(test.model.size.sum_of_max_total_cts, dtype=wp.float32)

        # Build the test problem and integrate the state over a single time-step
        test.build()
        integrate_semi_implicit_euler(model=test.model, data=test.data)

        nl = test.limits.model_num_limits.numpy()[0] if test.limits.num_model_max_limits > 0 else 0
        nc = test.contacts.model_num_contacts.numpy()[0] if test.contacts.num_model_max_contacts > 0 else 0
        msg.info("num active limits: %s", nl)
        msg.info("num active contacts: %s\n", nc)
        self.assertEqual(nl, 0)
        self.assertEqual(nc, 4)

        # Compute the metrics on the trivial solution
        metrics.reset()
        metrics.evaluate(
            sigma=sigma,
            lambdas=lambdas,
            v_plus=v_plus,
            model=test.model,
            data=test.data,
            state_p=test.state_p,
            problem=test.problem,
            jacobians=test.jacobians,
            limits=test.limits,
            contacts=test.contacts,
        )

        # Optional verbose output
        msg.info("metrics.r_eom: %s", metrics.data.r_eom)
        msg.info("metrics.r_kinematics: %s", metrics.data.r_kinematics)
        msg.info("metrics.r_cts_joints: %s", metrics.data.r_cts_joints)
        msg.info("metrics.r_cts_limits: %s", metrics.data.r_cts_limits)
        msg.info("metrics.r_cts_contacts: %s", metrics.data.r_cts_contacts)
        msg.info("metrics.r_v_plus: %s", metrics.data.r_v_plus)
        msg.info("metrics.r_ncp_primal: %s", metrics.data.r_ncp_primal)
        msg.info("metrics.r_ncp_dual: %s", metrics.data.r_ncp_dual)
        msg.info("metrics.r_ncp_compl: %s", metrics.data.r_ncp_compl)
        msg.info("metrics.r_vi_natmap: %s\n", metrics.data.r_vi_natmap)

        # Extract the maximum contact penetration to use for validation
        nc = test.contacts.model_num_contacts.numpy()[0]
        max_contact_penetration = 0.0
        for cid in range(nc):
            pen = test.contacts.gapfunc.numpy()[cid][3]
            max_contact_penetration = max(max_contact_penetration, pen)

        # Check that all metrics are zero
        np.testing.assert_allclose(metrics.data.r_eom.numpy()[0], 0.0)
        np.testing.assert_allclose(metrics.data.r_kinematics.numpy()[0], 0.0)
        np.testing.assert_allclose(metrics.data.r_cts_joints.numpy()[0], 0.0)
        np.testing.assert_allclose(metrics.data.r_cts_limits.numpy()[0], 0.0)
        np.testing.assert_allclose(metrics.data.r_cts_contacts.numpy()[0], max_contact_penetration)
        np.testing.assert_allclose(metrics.data.r_ncp_primal.numpy()[0], 0.0)
        np.testing.assert_allclose(metrics.data.r_ncp_dual.numpy()[0], 0.0)
        np.testing.assert_allclose(metrics.data.r_ncp_compl.numpy()[0], 0.0)
        np.testing.assert_allclose(metrics.data.r_vi_natmap.numpy()[0], 0.0)

        # Optional verbose output
        msg.info("metrics.r_eom_argmax: %s", metrics.data.r_eom_argmax)
        msg.info("metrics.r_kinematics_argmax: %s", metrics.data.r_kinematics_argmax)
        msg.info("metrics.r_cts_joints_argmax: %s", metrics.data.r_cts_joints_argmax)
        msg.info("metrics.r_cts_limits_argmax: %s", metrics.data.r_cts_limits_argmax)
        msg.info("metrics.r_cts_contacts_argmax: %s", metrics.data.r_cts_contacts_argmax)
        msg.info("metrics.r_v_plus_argmax: %s", metrics.data.r_v_plus_argmax)
        msg.info("metrics.r_ncp_primal_argmax: %s", metrics.data.r_ncp_primal_argmax)
        msg.info("metrics.r_ncp_dual_argmax: %s", metrics.data.r_ncp_dual_argmax)
        msg.info("metrics.r_ncp_compl_argmax: %s", metrics.data.r_ncp_compl_argmax)
        msg.info("metrics.r_vi_natmap_argmax: %s\n", metrics.data.r_vi_natmap_argmax)

        # Check that all argmax indices are correct
        np.testing.assert_allclose(metrics.data.r_eom_argmax.numpy()[0], 0)  # only one body
        np.testing.assert_allclose(metrics.data.r_kinematics_argmax.numpy()[0], -1)  # no joints
        np.testing.assert_allclose(metrics.data.r_cts_joints_argmax.numpy()[0], -1)  # no joints
        np.testing.assert_allclose(metrics.data.r_cts_limits_argmax.numpy()[0], -1)  # no limits
        # NOTE: all contacts will have the same residual,
        # so the argmax will evaluate to the last constraint
        np.testing.assert_allclose(metrics.data.r_v_plus_argmax.numpy()[0], 11)
        # NOTE: all contacts will have the same penetration,
        # so the argmax will evaluate to the last contact
        np.testing.assert_allclose(metrics.data.r_cts_contacts_argmax.numpy()[0], 3)
        np.testing.assert_allclose(metrics.data.r_ncp_primal_argmax.numpy()[0], 3)
        np.testing.assert_allclose(metrics.data.r_ncp_dual_argmax.numpy()[0], 3)
        np.testing.assert_allclose(metrics.data.r_ncp_compl_argmax.numpy()[0], 3)
        np.testing.assert_allclose(metrics.data.r_vi_natmap_argmax.numpy()[0], 3)

    def test_03_evaluate_padmm_solution_box_on_plane(self):
        """
        Tests evaluating metrics on a solution computed with the Proximal-ADMM (PADMM) solver.
        """
        # Create the test problem
        test = TestSetup(
            builder_fn=build_box_on_plane, max_world_contacts=4, gravity=True, perturb=True, device=self.default_device
        )

        # Create the PADMM solver
        solver = PADMMSolver(model=test.model, use_acceleration=False, collect_info=True)

        # Creating a default solver metrics evaluator from the test model
        metrics = SolutionMetrics(model=test.model)

        # Solve the test problem
        test.build()
        solver.reset()
        solver.coldstart()
        solver.solve(problem=test.problem)
        integrate_semi_implicit_euler(model=test.model, data=test.data)

        # Compute the metrics on the trivial solution
        metrics.reset()
        metrics.evaluate(
            sigma=solver.data.state.sigma,
            lambdas=solver.data.solution.lambdas,
            v_plus=solver.data.solution.v_plus,
            model=test.model,
            data=test.data,
            state_p=test.state_p,
            problem=test.problem,
            jacobians=test.jacobians,
            limits=test.limits,
            contacts=test.contacts,
        )

        nl = test.limits.model_num_limits.numpy()[0] if test.limits.num_model_max_limits > 0 else 0
        nc = test.contacts.model_num_contacts.numpy()[0] if test.contacts.num_model_max_contacts > 0 else 0
        msg.info("num active limits: %s", nl)
        msg.info("num active contacts: %s\n", nc)

        # Optional verbose output
        msg.info("metrics.r_eom: %s", metrics.data.r_eom)
        msg.info("metrics.r_kinematics: %s", metrics.data.r_kinematics)
        msg.info("metrics.r_cts_joints: %s", metrics.data.r_cts_joints)
        msg.info("metrics.r_cts_limits: %s", metrics.data.r_cts_limits)
        msg.info("metrics.r_cts_contacts: %s", metrics.data.r_cts_contacts)
        msg.info("metrics.r_v_plus: %s", metrics.data.r_v_plus)
        msg.info("metrics.r_ncp_primal: %s", metrics.data.r_ncp_primal)
        msg.info("metrics.r_ncp_dual: %s", metrics.data.r_ncp_dual)
        msg.info("metrics.r_ncp_compl: %s", metrics.data.r_ncp_compl)
        msg.info("metrics.r_vi_natmap: %s\n", metrics.data.r_vi_natmap)

        # Extract the maximum contact penetration to use for validation
        nc = test.contacts.model_num_contacts.numpy()[0]
        max_contact_penetration = 0.0
        for cid in range(nc):
            pen = test.contacts.gapfunc.numpy()[cid][3]
            max_contact_penetration = max(max_contact_penetration, pen)

        # Check that all metrics are zero
        accuracy = 5  # number of decimal places for accuracy
        self.assertAlmostEqual(metrics.data.r_eom.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_kinematics.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_cts_joints.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_cts_limits.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_cts_contacts.numpy()[0], max_contact_penetration, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_ncp_primal.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_ncp_dual.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_ncp_compl.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_vi_natmap.numpy()[0], 0.0, places=accuracy)

        # Optional verbose output
        msg.info("metrics.r_eom_argmax: %s", metrics.data.r_eom_argmax)
        msg.info("metrics.r_kinematics_argmax: %s", metrics.data.r_kinematics_argmax)
        msg.info("metrics.r_cts_joints_argmax: %s", metrics.data.r_cts_joints_argmax)
        msg.info("metrics.r_cts_limits_argmax: %s", metrics.data.r_cts_limits_argmax)
        msg.info("metrics.r_cts_contacts_argmax: %s", metrics.data.r_cts_contacts_argmax)
        msg.info("metrics.r_v_plus_argmax: %s", metrics.data.r_v_plus_argmax)
        msg.info("metrics.r_ncp_primal_argmax: %s", metrics.data.r_ncp_primal_argmax)
        msg.info("metrics.r_ncp_dual_argmax: %s", metrics.data.r_ncp_dual_argmax)
        msg.info("metrics.r_ncp_compl_argmax: %s", metrics.data.r_ncp_compl_argmax)
        msg.info("metrics.r_vi_natmap_argmax: %s\n", metrics.data.r_vi_natmap_argmax)

    def test_04_evaluate_padmm_solution_boxes_hinged(self):
        """
        Tests evaluating metrics on a solution computed with the Proximal-ADMM (PADMM) solver.
        """
        # Create the test problem
        test = TestSetup(
            builder_fn=build_boxes_hinged, max_world_contacts=8, gravity=True, perturb=True, device=self.default_device
        )

        # Create the PADMM solver
        solver = PADMMSolver(model=test.model, use_acceleration=False, collect_info=True)

        # Creating a default solver metrics evaluator from the test model
        metrics = SolutionMetrics(model=test.model)

        # Solve the test problem
        test.build()
        solver.reset()
        solver.coldstart()
        solver.solve(problem=test.problem)
        integrate_semi_implicit_euler(model=test.model, data=test.data)

        # Compute the metrics on the trivial solution
        metrics.evaluate(
            sigma=solver.data.state.sigma,
            lambdas=solver.data.solution.lambdas,
            v_plus=solver.data.solution.v_plus,
            model=test.model,
            data=test.data,
            state_p=test.state_p,
            problem=test.problem,
            jacobians=test.jacobians,
            limits=test.limits,
            contacts=test.contacts,
        )

        nl = test.limits.model_num_limits.numpy()[0] if test.limits.num_model_max_limits > 0 else 0
        nc = test.contacts.model_num_contacts.numpy()[0] if test.contacts.num_model_max_contacts > 0 else 0
        msg.info("num active limits: %s", nl)
        msg.info("num active contacts: %s\n", nc)

        # Optional verbose output
        msg.info("metrics.r_eom: %s", metrics.data.r_eom)
        msg.info("metrics.r_kinematics: %s", metrics.data.r_kinematics)
        msg.info("metrics.r_cts_joints: %s", metrics.data.r_cts_joints)
        msg.info("metrics.r_cts_limits: %s", metrics.data.r_cts_limits)
        msg.info("metrics.r_cts_contacts: %s", metrics.data.r_cts_contacts)
        msg.info("metrics.r_v_plus: %s", metrics.data.r_v_plus)
        msg.info("metrics.r_ncp_primal: %s", metrics.data.r_ncp_primal)
        msg.info("metrics.r_ncp_dual: %s", metrics.data.r_ncp_dual)
        msg.info("metrics.r_ncp_compl: %s", metrics.data.r_ncp_compl)
        msg.info("metrics.r_vi_natmap: %s\n", metrics.data.r_vi_natmap)

        # Extract the maximum contact penetration to use for validation
        max_contact_penetration = 0.0
        for cid in range(nc):
            pen = test.contacts.gapfunc.numpy()[cid][3]
            max_contact_penetration = max(max_contact_penetration, pen)

        # Check that all metrics are zero
        accuracy = 5  # number of decimal places for accuracy
        self.assertAlmostEqual(metrics.data.r_eom.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_kinematics.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_cts_joints.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_cts_limits.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_cts_contacts.numpy()[0], max_contact_penetration, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_ncp_primal.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_ncp_dual.numpy()[0], 0.0, places=4)  # less accurate, but still correct
        self.assertAlmostEqual(metrics.data.r_ncp_compl.numpy()[0], 0.0, places=accuracy)
        self.assertAlmostEqual(metrics.data.r_vi_natmap.numpy()[0], 0.0, places=accuracy)

        # Optional verbose output
        msg.info("metrics.r_eom_argmax: %s", metrics.data.r_eom_argmax)
        msg.info("metrics.r_kinematics_argmax: %s", metrics.data.r_kinematics_argmax)
        msg.info("metrics.r_cts_joints_argmax: %s", metrics.data.r_cts_joints_argmax)
        msg.info("metrics.r_cts_limits_argmax: %s", metrics.data.r_cts_limits_argmax)
        msg.info("metrics.r_cts_contacts_argmax: %s", metrics.data.r_cts_contacts_argmax)
        msg.info("metrics.r_v_plus_argmax: %s", metrics.data.r_v_plus_argmax)
        msg.info("metrics.r_ncp_primal_argmax: %s", metrics.data.r_ncp_primal_argmax)
        msg.info("metrics.r_ncp_dual_argmax: %s", metrics.data.r_ncp_dual_argmax)
        msg.info("metrics.r_ncp_compl_argmax: %s", metrics.data.r_ncp_compl_argmax)
        msg.info("metrics.r_vi_natmap_argmax: %s\n", metrics.data.r_vi_natmap_argmax)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=1000, precision=10, threshold=10000, suppress=True)  # Suppress scientific notation

    # Global warp configurations
    wp.config.enable_backward = False
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
