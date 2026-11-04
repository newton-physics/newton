# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the JointSpacePIDController class."""

import logging
import unittest

import warp as wp

from newton._src.solvers.kamino._src.utils.control.pid import JointSpacePIDController
from newton._src.solvers.kamino._src.utils.logger import reset_log_level, set_log_level
from newton._src.solvers.kamino.tests import setup_tests, test_context

logger = logging.getLogger(__name__)

###
# Tests
###


class TestJointSpacePIDController(unittest.TestCase):
    def setUp(self):
        # Configs
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.seed = 42
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            set_log_level(logging.DEBUG)
            logger.debug("\n")  # Add newline before test output for better readability
        else:
            reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            reset_log_level()

    def test_make_default(self):
        # Create a default PID controller
        controller = JointSpacePIDController()
        # Check default values
        self.assertIsNotNone(controller)
        self.assertEqual(controller.device, None)
        self.assertEqual(controller._data, None)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
