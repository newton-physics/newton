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
KAMINO: UNIT TESTS: CORE: MODEL
"""

import unittest

import warp as wp

# Module to be tested
from newton._src.solvers.kamino.core.model import Model

# Test utilities
from newton._src.solvers.kamino.models.builders.basics import (
    build_boxes_hinged,
    build_boxes_nunchaku,
    make_basics_heterogeneous_builder,
)
from newton._src.solvers.kamino.models.builders.utils import make_homogeneous_builder
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.utils.print import (
    print_model_bodies,
    print_model_data_info,
    print_model_info,
    print_model_joints,
)
from newton._src.solvers.kamino.utils import logger as msg

###
# Tests
###


class TestModel(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for verbose output

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

    def test_01_single_model(self):
        # Create a model builder
        builder = build_boxes_hinged()

        # Finalize the model
        model: Model = builder.finalize(self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print_model_info(model)

        # Create a model state
        state = model.data()
        if self.verbose:
            print("")  # Add a newline for better readability
            print_model_data_info(state)

        # Check the model info entries
        self.assertEqual(model.size.sum_of_num_bodies, builder.num_bodies)
        self.assertEqual(model.size.sum_of_num_joints, builder.num_joints)
        self.assertEqual(model.size.sum_of_num_collision_geoms, builder.num_collision_geoms)
        self.assertEqual(model.size.sum_of_num_physical_geoms, 0)
        self.assertEqual(model.device, self.default_device)

    def test_02_double_model(self):
        # Create a model builder
        builder1 = build_boxes_hinged()
        builder2 = build_boxes_nunchaku()

        # Compute the total number of elements from the two builders
        total_nb = builder1.num_bodies + builder2.num_bodies
        total_nj = builder1.num_joints + builder2.num_joints
        total_ng = builder1.num_collision_geoms + builder2.num_collision_geoms

        # Add the second builder to the first one
        builder1.add_builder(builder2)

        # Finalize the model
        model: Model = builder1.finalize(self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print_model_info(model)

        # Create a model state
        data = model.data()
        if self.verbose:
            print("")  # Add a newline for better readability
            print_model_data_info(data)

        # Check the model info entries
        self.assertEqual(model.size.sum_of_num_bodies, total_nb)
        self.assertEqual(model.size.sum_of_num_joints, total_nj)
        self.assertEqual(model.size.sum_of_num_collision_geoms, total_ng)
        self.assertEqual(model.size.sum_of_num_physical_geoms, 0)

    def test_03_homogeneous_model(self):
        # Constants
        num_worlds = 4

        # Create a model builder
        builder = make_homogeneous_builder(num_worlds=num_worlds, build_fn=build_boxes_hinged)

        # Finalize the model
        model: Model = builder.finalize(self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print_model_info(model)

        # Create a model state
        state = model.data()
        if self.verbose:
            print("")  # Add a newline for better readability
            print_model_data_info(state)

        # Check the model info entries
        self.assertEqual(model.size.sum_of_num_bodies, num_worlds * 2)
        self.assertEqual(model.size.sum_of_num_joints, num_worlds * 1)
        self.assertEqual(model.size.sum_of_num_collision_geoms, num_worlds * 3)
        self.assertEqual(model.size.sum_of_num_physical_geoms, num_worlds * 0)
        self.assertEqual(model.device, self.default_device)

    def test_04_hetereogeneous_model(self):
        # Create a model builder
        builder = make_basics_heterogeneous_builder()
        num_worlds = builder.num_worlds

        # Finalize the model
        model: Model = builder.finalize(self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print_model_info(model)
            print("")  # Add a newline for better readability
            print_model_bodies(model)
            print("")  # Add a newline for better readability
            print_model_joints(model)

        # Create a model state
        state = model.data()
        if self.verbose:
            print("")  # Add a newline for better readability
            print_model_data_info(state)

        # Check the model info entries
        self.assertEqual(model.info.num_worlds, num_worlds)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
