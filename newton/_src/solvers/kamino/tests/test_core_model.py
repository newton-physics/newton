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

import numpy as np
import warp as wp

from newton._src.solvers.kamino.core.builder import ModelBuilder

# Module to be tested
from newton._src.solvers.kamino.core.model import Model

# Test utilities
from newton._src.solvers.kamino.models.builders import (
    build_boxes_hinged,
    build_boxes_nunchaku,
)
from newton._src.solvers.kamino.models.utils import (
    make_heterogeneous_builder,
    make_homogeneous_builder,
)
from newton._src.solvers.kamino.tests.utils.print import (
    print_model_bodies,
    print_model_data_info,
    print_model_info,
    print_model_joints,
)

###
# Tests
###


class TestModel(unittest.TestCase):
    def setUp(self):
        self.verbose = False  # Set to True for verbose output
        self.default_device = wp.get_device()

    def tearDown(self):
        self.default_device = None

    def test_01_single_model(self):
        # Create a model builder
        builder = ModelBuilder()

        # Construct a first model
        bids, jids, gids = build_boxes_hinged(builder)

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
        self.assertEqual(model.size.sum_of_num_bodies, len(bids))
        self.assertEqual(model.size.sum_of_num_joints, len(jids))
        self.assertEqual(model.size.sum_of_num_collision_geoms, len(gids))
        self.assertEqual(model.size.sum_of_num_physical_geoms, 0)
        self.assertEqual(model.device, self.default_device)

    def test_02_double_model(self):
        # Create a model builder
        builder1 = ModelBuilder()
        bids1, jids1, gids1 = build_boxes_hinged(builder1)

        # Create a second model builder
        builder2 = ModelBuilder()
        bids2, jids2, gids2 = build_boxes_nunchaku(builder2)

        # Add the second builder to the first one
        builder1.add_builder(builder2)

        # Compute the total number of elements from the two builders
        total_nb = len(bids1) + len(bids2)
        total_nj = len(jids1) + len(jids2)
        total_ng = len(gids1) + len(gids2)

        # Finalize the model
        model: Model = builder1.finalize(self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print_model_info(model)

        # Create a model state
        state = model.data()
        if self.verbose:
            print("")  # Add a newline for better readability
            print_model_data_info(state)

        # Check the model info entries
        self.assertEqual(model.size.sum_of_num_bodies, total_nb)
        self.assertEqual(model.size.sum_of_num_joints, total_nj)
        self.assertEqual(model.size.sum_of_num_collision_geoms, total_ng)
        self.assertEqual(model.size.sum_of_num_physical_geoms, 0)

    def test_03_homogeneous_model(self):
        # Constants
        num_worlds = 4

        # Create a model builder
        builder = make_homogeneous_builder(num_worlds=num_worlds, build_func=build_boxes_hinged)

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
        builder, _, _ = make_heterogeneous_builder()
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
    # Global numpy configurations
    np.set_printoptions(linewidth=500, precision=6, suppress=True)  # Suppress scientific notation

    # Global warp configurations
    wp.config.verbose = True
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
