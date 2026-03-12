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

"""Unit tests for the RandomController class."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.models.builders.basics import build_boxes_fourbar
from newton._src.solvers.kamino._src.models.builders.utils import make_homogeneous_builder
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino._src.utils.control.rand import RandomJointController
from newton._src.solvers.kamino.tests import setup_tests, test_context

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
        builder = make_homogeneous_builder(num_worlds=1, build_fn=build_boxes_fourbar)
        model = builder.finalize(device=self.default_device)
        data = model.data()
        control = model.control()

        # Create a random controller with default arguments
        controller = RandomJointController(model=model, seed=self.seed)

        # Check contents
        self.assertIsNotNone(controller)
        self.assertIsNotNone(controller._model, None)
        self.assertIsNotNone(controller._data, None)
        self.assertIs(controller.device, model.device)

        # Check dimensions of the decimation array
        self.assertEqual(controller.data.decimation.shape, (model.size.num_worlds,))
        self.assertTrue((controller.data.decimation.numpy() == 1).all())

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
        builder = make_homogeneous_builder(num_worlds=4, build_fn=build_boxes_fourbar)
        model = builder.finalize(device=self.default_device)
        data = model.data()
        control = model.control()

        # Create a random controller with default arguments
        controller = RandomJointController(model=model, seed=self.seed)

        # Check contents
        self.assertIsNotNone(controller)
        self.assertIsNotNone(controller._model, None)
        self.assertIsNotNone(controller._data, None)
        self.assertIs(controller.device, model.device)

        # Check dimensions of the decimation array
        self.assertEqual(controller.data.decimation.shape, (model.size.num_worlds,))
        self.assertTrue((controller.data.decimation.numpy() == 1).all())

        # Check that the seed is set correctly
        self.assertEqual(controller.seed, self.seed)

        # Check that the generated control inputs are different than the default values
        self.assertEqual(np.linalg.norm(control.tau_j.numpy()), 0.0)
        controller.compute(time=data.time, control=control)
        tau_j_np_0 = control.tau_j.numpy().copy()
        msg.info("control.tau_j: %s", tau_j_np_0)
        self.assertGreaterEqual(np.linalg.norm(control.tau_j.numpy()), 0.0)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
