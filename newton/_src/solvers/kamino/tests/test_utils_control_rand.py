# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the RandomController class."""

import unittest

import numpy as np
import warp as wp

from newton import ModelBuilder
from newton._src.solvers.kamino._src import ModelKamino
from newton._src.solvers.kamino._src.core.time import advance_time
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino._src.utils.control.rand import RandomJointController
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton.tests.utils.basics import build_boxes_fourbar

###
# Tests
###


class TestRandomController(unittest.TestCase):
    def setUp(self):
        # Configs
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.seed = 42
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for verbose output

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

    def test_00_make_default(self):
        # Create a default random controller
        controller = RandomJointController()
        # Check default values
        self.assertIsNotNone(controller)
        self.assertEqual(controller._model, None)
        self.assertEqual(controller._data, None)
        self.assertRaises(RuntimeError, lambda: controller.device)
        self.assertRaises(RuntimeError, lambda: controller.seed)
        self.assertRaises(RuntimeError, lambda: controller.model)
        self.assertRaises(RuntimeError, lambda: controller.data)

    def test_01_make_for_single_fourbar(self):
        # Define a model builder for the boxes_fourbar problem with 1 world
        builder = build_boxes_fourbar()
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device))
        data = model.data()
        control = model.control()

        # Create a random controller with default arguments
        controller = RandomJointController(model=model, seed=self.seed)

        # Check contents
        self.assertIsNotNone(controller)
        self.assertIsNotNone(controller._model, None)
        self.assertIsNotNone(controller._data, None)
        self.assertIs(controller.device, model.device)

        # Check dimensions of the interval array
        self.assertEqual(controller.data.interval.shape, (model.size.num_worlds,))
        self.assertTrue((controller.data.interval.numpy() == 1.0).all())

        # Check that the seed is set correctly
        self.assertEqual(controller.seed, self.seed)

        # Check that the generated control inputs are different than the default values
        self.assertEqual(np.linalg.norm(control.tau_j.numpy()), 0.0)
        controller.compute(time=data.time, control=control)
        tau_j_np_0 = control.tau_j.numpy().copy()
        msg.info("control.tau_j: %s", tau_j_np_0)
        self.assertGreaterEqual(np.linalg.norm(control.tau_j.numpy()), 0.0)

    def test_02_make_for_multiple_fourbar(self):
        # Define a model builder for the boxes_fourbar problem with 4 worlds
        builder = ModelBuilder()
        builder.replicate(builder=build_boxes_fourbar(), world_count=4)
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device))
        data = model.data()
        control = model.control()

        # Create a random controller with default arguments
        controller = RandomJointController(model=model, seed=self.seed)

        # Check contents
        self.assertIsNotNone(controller)
        self.assertIsNotNone(controller._model, None)
        self.assertIsNotNone(controller._data, None)
        self.assertIs(controller.device, model.device)

        # Check dimensions of the interval array
        self.assertEqual(controller.data.interval.shape, (model.size.num_worlds,))
        self.assertTrue((controller.data.interval.numpy() == 1.0).all())

        # Check that the seed is set correctly
        self.assertEqual(controller.seed, self.seed)

        # Check that the generated control inputs are different than the default values
        self.assertEqual(np.linalg.norm(control.tau_j.numpy()), 0.0)
        controller.compute(time=data.time, control=control)
        tau_j_np_0 = control.tau_j.numpy().copy()
        msg.info("control.tau_j: %s", tau_j_np_0)
        self.assertGreaterEqual(np.linalg.norm(control.tau_j.numpy()), 0.0)

    def test_03_time_based_scheduling_multi_world(self):
        """Behavioral test of the per-world elapsed-time scheduling.

        Configures four worlds with distinct ``(dt, interval)`` pairs and drives
        the controller through the simulation loop, asserting that ``tau_j`` is
        regenerated exactly at hand-computed elapsed-time boundaries and stays
        bit-for-bit identical between them. Worlds 0 and 3 share the same
        update-step schedule despite different ``dt``, which confirms the
        controller schedules on simulated time rather than step counts.
        """
        # Build a replicated 4-world fourbar model
        num_worlds = 4
        builder = ModelBuilder()
        builder.replicate(builder=build_boxes_fourbar(), world_count=num_worlds)
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device))
        data = model.data()
        control = model.control()

        # Configure distinct per-world time-steps (powers of two, exactly
        # representable in float32 to avoid accumulated drift over the loop)
        per_world_dt = np.array([0.125, 0.25, 0.125, 0.25], dtype=np.float32)
        model.time.set_timesteps(per_world_dt)

        # Configure distinct per-world intervals for torque regeneration
        per_world_interval = np.array([0.5, 0.75, 0.25, 1.0], dtype=np.float32)
        controller = RandomJointController(model=model, interval=per_world_interval, seed=self.seed)

        # Ground-truth per-world update-step sets, computed by hand from the
        # ``t = k * dt`` schedule crossing multiples of ``interval``:
        #   world 0: dt=0.125, interval=0.5  -> t in {0.0, 0.5, 1.0, 1.5}
        #   world 1: dt=0.25,  interval=0.75 -> t in {0.0, 0.75, 1.5, 2.25, 3.0}
        #   world 2: dt=0.125, interval=0.25 -> t in {0.0, 0.25, ..., 1.5}
        #   world 3: dt=0.25,  interval=1.0  -> t in {0.0, 1.0, 2.0, 3.0}
        # Note: worlds 0 and 3 update at the same step indices with different
        # ``dt``, demonstrating that scheduling is driven by elapsed time.
        expected_update_steps = {
            0: {0, 4, 8, 12},
            1: {0, 3, 6, 9, 12},
            2: {0, 2, 4, 6, 8, 10, 12},
            3: {0, 4, 8, 12},
        }
        num_steps = 13  # spans t in [0, 1.5] for world 0 and [0, 3.0] for world 3

        # Map each DoF to its owning world via the joint layout, so per-world
        # slices of ``tau_j`` can be compared independently
        joints_wid = model.joints.wid.numpy()
        dofs_offset = model.joints.dofs_offset.numpy()
        dof_wid = np.empty(int(model.size.sum_of_num_joint_dofs), dtype=np.int32)
        for j in range(joints_wid.shape[0]):
            dof_wid[dofs_offset[j] : dofs_offset[j + 1]] = joints_wid[j]
        for w in range(num_worlds):
            self.assertTrue((dof_wid == w).any(), f"world {w} has no DoFs to observe")

        # Walk the simulation loop, snapshotting tau_j after each compute; the
        # kernel writes only on update steps, so on non-update steps the slice
        # must remain byte-for-byte identical to the previous snapshot
        prev_tau_j = control.tau_j.numpy().copy()
        self.assertEqual(np.linalg.norm(prev_tau_j), 0.0)
        for step in range(num_steps):
            controller.compute(time=data.time, control=control)
            tau_j = control.tau_j.numpy().copy()
            for w in range(num_worlds):
                mask = dof_wid == w
                prev_slice = prev_tau_j[mask]
                curr_slice = tau_j[mask]
                if step in expected_update_steps[w]:
                    self.assertFalse(
                        np.array_equal(prev_slice, curr_slice),
                        f"world {w} step {step}: torques did not update at expected boundary",
                    )
                else:
                    np.testing.assert_array_equal(
                        curr_slice,
                        prev_slice,
                        err_msg=f"world {w} step {step}: torques changed between boundaries",
                    )
            advance_time(model=model.time, data=data.time)
            prev_tau_j = tau_j


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
