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
KAMINO: UNIT TESTS
"""

import unittest

import numpy as np
import warp as wp

# Module to be tested
from newton._src.solvers.kamino.geometry.collisions import Collisions, make_collision_pairs
from newton._src.solvers.kamino.models.builders import (
    build_boxes_nunchaku,
)
from newton._src.solvers.kamino.models.utils import (
    make_homogeneous_builder,
    make_single_builder,
)

###
# Tests
###


class TestGeometryCollisions(unittest.TestCase):
    def setUp(self):
        self.verbose = False  # Set to True for detailed output
        self.default_device = wp.get_device()

        # Set the common build function and geometry parameters
        self.build_func = build_boxes_nunchaku
        self.num_geoms = 4  # NOTE: specialized to build_boxes_nunchaku
        self.num_geom_pairs = 4  # NOTE: specialized to build_boxes_nunchaku
        self.max_geom_pairs = self.num_geoms * (self.num_geoms - 1) // 2
        if self.verbose:
            print("")
            print(f"build_func: {self.build_func.__name__}")
            print(f"  num_geoms: {self.num_geoms}")
            print(f"  max_geom_pairs: {self.max_geom_pairs}")
            print(f"  num_geom_pairs: {self.num_geom_pairs}")

    def tearDown(self):
        self.default_device = None

    def test_01_make_model_collision_pairs_single_world(self):
        # Create and set up a model builder
        builder = make_single_builder(build_fn=self.build_func)

        # Calculate the maximum number of geometry pairs
        max_geom_pairs = self.num_geoms * (self.num_geoms - 1) // 2
        num_geom_pairs = self.num_geom_pairs
        if self.verbose:
            print(f"max_geom_pairs: {max_geom_pairs}")
            print(f"num_geom_pairs: {num_geom_pairs}")

        # Construct collision pairs
        world_nxn_num_geom_pairs, model_nxn_geom_pair, model_nxn_pairid, model_nxn_wid = make_collision_pairs(builder)
        if self.verbose:
            print(f"world_nxn_num_geom_pairs (size={len(world_nxn_num_geom_pairs)}): {world_nxn_num_geom_pairs}")
            print(f"model_nxn_geom_pair (size={len(model_nxn_geom_pair)}): {model_nxn_geom_pair}")
            print(f"model_nxn_pairid (size={len(model_nxn_pairid)}): {model_nxn_pairid}")
            print(f"model_nxn_wid (size={len(model_nxn_wid)}): {model_nxn_wid}")

        # Check collision pairs allocations
        self.assertLessEqual(len(model_nxn_geom_pair), max_geom_pairs)
        self.assertLessEqual(len(model_nxn_pairid), max_geom_pairs)
        self.assertLessEqual(len(model_nxn_wid), max_geom_pairs)
        self.assertEqual(len(model_nxn_geom_pair), num_geom_pairs)
        self.assertEqual(len(model_nxn_pairid), num_geom_pairs)
        self.assertEqual(len(model_nxn_wid), num_geom_pairs)

    def test_02_make_model_collision_pairs_multiple_world(self):
        # Create and set up a model builder
        builder = make_homogeneous_builder(num_worlds=3, build_fn=self.build_func)
        num_worlds = builder.num_worlds

        # Calculate the maximum number of geometry pairs
        max_geom_pairs = num_worlds * self.num_geoms * (self.num_geoms - 1) // 2
        num_geom_pairs = num_worlds * self.num_geom_pairs
        if self.verbose:
            print(f"max_geom_pairs: {max_geom_pairs}")
            print(f"num_geom_pairs: {num_geom_pairs}")

        # Construct collision pairs
        world_nxn_num_geom_pairs, model_nxn_geom_pair, model_nxn_pairid, model_nxn_wid = make_collision_pairs(builder)
        if self.verbose:
            print(f"world_nxn_num_geom_pairs (size={len(world_nxn_num_geom_pairs)}):\n{world_nxn_num_geom_pairs}")
            print(f"model_nxn_geom_pair (size={len(model_nxn_geom_pair)}):\n{model_nxn_geom_pair}")
            print(f"model_nxn_pairid (size={len(model_nxn_pairid)}):\n{model_nxn_pairid}")
            print(f"model_nxn_wid (size={len(model_nxn_wid)}):\n{model_nxn_wid}")

        # Check collision pairs allocations
        self.assertLessEqual(len(world_nxn_num_geom_pairs), num_worlds)
        self.assertLessEqual(len(model_nxn_geom_pair), max_geom_pairs)
        self.assertLessEqual(len(model_nxn_pairid), max_geom_pairs)
        self.assertLessEqual(len(model_nxn_wid), max_geom_pairs)
        self.assertEqual(len(model_nxn_geom_pair), num_geom_pairs)
        self.assertEqual(len(model_nxn_pairid), num_geom_pairs)
        self.assertEqual(len(model_nxn_wid), num_geom_pairs)

    def test_03_create_collisions_container(self):
        # Create and set up a model builder
        builder = make_homogeneous_builder(num_worlds=3, build_fn=self.build_func)
        num_worlds = builder.num_worlds

        # Calculate the maximum number of geometry pairs
        max_geom_pairs = num_worlds * self.num_geoms * (self.num_geoms - 1) // 2
        num_geom_pairs = num_worlds * self.num_geom_pairs
        num_collision_pairs = num_geom_pairs
        if self.verbose:
            print(f"max_geom_pairs: {max_geom_pairs}")
            print(f"num_geom_pairs: {num_geom_pairs}")
            print(f"num_collision_pairs: {num_collision_pairs}")

        # Create a collision container
        collisions = Collisions(builder=builder, device=self.default_device)
        if self.verbose:
            print(f"collisions.cmodel.num_model_geom_pairs: {collisions.cmodel.num_model_geom_pairs}")
            print(f"collisions.cmodel.num_world_geom_pairs: {collisions.cmodel.num_world_geom_pairs}")
            print(
                f"collisions.cmodel.model_num_pairs (size={len(collisions.cmodel.model_num_pairs)}):\n{collisions.cmodel.model_num_pairs}"
            )
            print(
                f"collisions.cmodel.world_num_pairs (size={len(collisions.cmodel.world_num_pairs)}):\n{collisions.cmodel.world_num_pairs}"
            )
            print(f"collisions.cmodel.wid (size={len(collisions.cmodel.wid)}):\n{collisions.cmodel.wid}")
            print(f"collisions.cmodel.pairid (size={len(collisions.cmodel.pairid)}):\n{collisions.cmodel.pairid}")
            print(
                f"collisions.cmodel.geom_pair (size={len(collisions.cmodel.geom_pair)}):\n{collisions.cmodel.geom_pair}"
            )
            print(
                f"collisions.cdata.model_num_collisions (size={len(collisions.cdata.model_num_collisions)}):\n{collisions.cdata.model_num_collisions}"
            )
            print(
                f"collisions.cdata.world_num_collisions (size={len(collisions.cdata.world_num_collisions)}):\n{collisions.cdata.world_num_collisions}"
            )
            print(f"collisions.cdata.wid (size={len(collisions.cdata.wid)}):\n{collisions.cdata.wid}")
            print(f"collisions.cdata.geom_pair (size={len(collisions.cdata.geom_pair)}):\n{collisions.cdata.geom_pair}")

        # Check collision container allocations
        self.assertLessEqual(collisions.cmodel.num_model_geom_pairs, max_geom_pairs)
        self.assertLessEqual(sum(collisions.cmodel.num_world_geom_pairs), max_geom_pairs)
        self.assertEqual(collisions.cmodel.num_model_geom_pairs, num_geom_pairs)
        self.assertEqual(sum(collisions.cmodel.num_world_geom_pairs), num_geom_pairs)
        self.assertEqual(len(collisions.cmodel.model_num_pairs), 1)
        self.assertEqual(len(collisions.cmodel.world_num_pairs), num_worlds)
        self.assertEqual(len(collisions.cmodel.wid), num_collision_pairs)
        self.assertEqual(len(collisions.cmodel.pairid), num_collision_pairs)
        self.assertEqual(len(collisions.cmodel.geom_pair), num_collision_pairs)
        self.assertEqual(len(collisions.cdata.model_num_collisions), 1)
        self.assertEqual(len(collisions.cdata.world_num_collisions), num_worlds)
        self.assertEqual(len(collisions.cdata.wid), num_collision_pairs)
        self.assertEqual(len(collisions.cdata.geom_pair), num_collision_pairs)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=500, precision=10, suppress=True)  # Suppress scientific notation

    # Global warp configurations
    wp.config.enable_backward = False
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
