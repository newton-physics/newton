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
KAMINO: UNIT TESTS: CORE: BUILDER
"""

import unittest

import numpy as np
import warp as wp

# Module to be tested
from newton._src.solvers.kamino.core.joints import JointDoFType

###
# Tests
###


class TestCoreJoints(unittest.TestCase):
    def setUp(self):
        self.verbose = False  # Set to True to enable verbose output
        self.default_device = wp.get_device()

    def tearDown(self):
        self.default_device = None

    def test_joint_dof_type_enum(self):
        doftype = JointDoFType.REVOLUTE

        # Optional verbose output
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"doftype: {doftype}")
            print(f"doftype.value: {doftype.value}")
            print(f"doftype.name: {doftype.name}")
            print(f"doftype.num_cts: {doftype.num_cts}")
            print(f"doftype.num_dofs: {doftype.num_dofs}")

        # Check the enum values
        self.assertEqual(doftype.value, JointDoFType.REVOLUTE)
        self.assertEqual(doftype.name, "REVOLUTE")
        self.assertEqual(doftype.num_cts, 5)
        self.assertEqual(doftype.num_dofs, 1)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=500, precision=10, suppress=True)  # Suppress scientific notation

    # Global warp configurations
    wp.config.verbose = True
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
