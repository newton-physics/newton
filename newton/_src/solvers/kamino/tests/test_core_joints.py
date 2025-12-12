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

"""Unit tests for the `kamino.core.joints` module"""

import unittest

import warp as wp

from newton._src.solvers.kamino.core.joints import JointDoFType
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.utils import logger as msg

###
# Tests
###


class TestCoreJoints(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True to enable verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_joint_dof_type_enum(self):
        doftype = JointDoFType.REVOLUTE

        # Optional verbose output
        msg.info(f"doftype: {doftype}")
        msg.info(f"doftype.value: {doftype.value}")
        msg.info(f"doftype.name: {doftype.name}")
        msg.info(f"doftype.num_cts: {doftype.num_cts}")
        msg.info(f"doftype.num_dofs: {doftype.num_dofs}")
        msg.info(f"doftype.cts_axes: {doftype.cts_axes}")
        msg.info(f"doftype.dofs_axes: {doftype.dofs_axes}")

        # Check the enum values
        self.assertEqual(doftype.value, JointDoFType.REVOLUTE)
        self.assertEqual(doftype.name, "REVOLUTE")
        self.assertEqual(doftype.num_cts, 5)
        self.assertEqual(doftype.num_dofs, 1)
        self.assertEqual(doftype.cts_axes, (0, 1, 2, 4, 5))
        self.assertEqual(doftype.dofs_axes, (3,))


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
