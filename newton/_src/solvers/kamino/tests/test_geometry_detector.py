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

# Modules to be tested
from newton._src.solvers.kamino.core.geometry import update_collision_geometries_state
from newton._src.solvers.kamino.geometry.broadphase import nxn_broadphase
from newton._src.solvers.kamino.geometry.collisions import Collisions
from newton._src.solvers.kamino.geometry.contacts import Contacts
from newton._src.solvers.kamino.geometry.detector import CollisionDetector
from newton._src.solvers.kamino.geometry.primitives import primitive_narrowphase
from newton._src.solvers.kamino.models.builders import build_boxes_nunchaku
from newton._src.solvers.kamino.models.utils import make_homogeneous_builder

###
# Tests
###


class TestGeometryCollisionDetector(unittest.TestCase):
    def setUp(self):
        self.verbose = False  # Set to True for detailed output
        self.default_device = wp.get_device()

        # Set the common build function and geometry parameters
        self.build_func = build_boxes_nunchaku
        self.num_collisions = 3  # NOTE: specialized to build_boxes_nunchaku
        self.num_contacts = 9  # NOTE: specialized to build_boxes_nunchaku
        self.max_contacts = 12  # NOTE: This is specialized to the nunchaku model
        if self.verbose:
            print("")
            print(f"build_func: {self.build_func.__name__}")
            print(f"  num_collisions: {self.num_collisions}")
            print(f"  num_contacts: {self.num_contacts}")
            print(f"  max_contacts: {self.max_contacts}")
            print("")

    def tearDown(self):
        self.default_device = None

    def test_01_update_collision_geometry_state(self):
        # Create and set up a model builder
        builder = make_homogeneous_builder(num_worlds=3, build_fn=self.build_func)

        # Finalize the model
        model = builder.finalize(self.default_device)

        # Create a state container
        state = model.data()

        # Update the absolute poses (i.e. computed in world coordinates) of the collision geometries
        update_collision_geometries_state(state.bodies.q_i, model.cgeoms, state.cgeoms)

        # Optional verbose output
        if self.verbose:
            print(f"state.bodies.q_i:\n{state.bodies.q_i}")
            print(f"state.cgeoms.pose:\n{state.cgeoms.pose}")
            print(f"state.cgeoms.aabb:\n{state.cgeoms.aabb}")
            print(f"state.cgeoms.radius:\n{state.cgeoms.radius}")

    def test_02_nxn_broadphase(self):
        # Create and set up a model builder
        builder = make_homogeneous_builder(num_worlds=4, build_fn=self.build_func)
        num_worlds = builder.num_worlds

        # Finalize the model
        model = builder.finalize(self.default_device)

        # Create a state container
        state = model.data()

        # Update the state of the collision geometries
        update_collision_geometries_state(state.bodies.q_i, model.cgeoms, state.cgeoms)

        # Create collisions container
        collisions = Collisions(builder=builder, device=self.default_device)

        # Execute brute-force (NxN) broadphase
        with wp.ScopedTimer("nxn_broadphase"):
            nxn_broadphase(model.cgeoms, state.cgeoms, collisions.cmodel, collisions.cdata)

        # Check collision output
        self.assertEqual(collisions.cdata.model_num_collisions.numpy()[0], num_worlds * self.num_collisions)
        for i in range(num_worlds):
            self.assertEqual(collisions.cdata.world_num_collisions.numpy()[i], self.num_collisions)

        # Optional verbose output
        if self.verbose:
            print(f"collisions.cmodel.num_model_geom_pairs: {collisions.cmodel.num_model_geom_pairs}")
            print(f"collisions.cmodel.num_world_geom_pairs: {collisions.cmodel.num_world_geom_pairs}")
            print(
                f"collisions.cmodel.model_num_pairs (size={len(collisions.cmodel.model_num_pairs)}): {collisions.cmodel.model_num_pairs}"
            )
            print(
                f"collisions.cmodel.world_num_pairs (size={len(collisions.cmodel.world_num_pairs)}): {collisions.cmodel.world_num_pairs}"
            )
            print(f"collisions.cmodel.wid (size={len(collisions.cmodel.wid)}): {collisions.cmodel.wid}")
            print(f"collisions.cmodel.pairid (size={len(collisions.cmodel.pairid)}): {collisions.cmodel.pairid}")
            print(
                f"collisions.cmodel.geom_pair (size={len(collisions.cmodel.geom_pair)}):\n{collisions.cmodel.geom_pair}"
            )
            print(
                f"collisions.cdata.model_num_collisions (size={len(collisions.cdata.model_num_collisions)}): {collisions.cdata.model_num_collisions}"
            )
            print(
                f"collisions.cdata.world_num_collisions (size={len(collisions.cdata.world_num_collisions)}): {collisions.cdata.world_num_collisions}"
            )
            print(f"collisions.cdata.wid (size={len(collisions.cdata.wid)}): {collisions.cdata.wid}")
            print(f"collisions.cdata.geom_pair (size={len(collisions.cdata.geom_pair)}):\n{collisions.cdata.geom_pair}")

    def test_03_primitive_narrowphase(self):
        # Create and set up a model builder
        builder = make_homogeneous_builder(num_worlds=4, build_fn=self.build_func)
        num_worlds = builder.num_worlds

        # Finalize the model
        model = builder.finalize(self.default_device)

        # Create a state container
        state = model.data()

        # Update the state of the collision geometries
        update_collision_geometries_state(state.bodies.q_i, model.cgeoms, state.cgeoms)

        # Create collisions container
        collisions = Collisions(builder=builder, device=self.default_device)

        # Execute brute-force (NxN) broadphase
        nxn_broadphase(model.cgeoms, state.cgeoms, collisions.cmodel, collisions.cdata)

        # Create a contacts container
        capacity = [self.max_contacts] * num_worlds  # Custom capacity for each world
        contacts = Contacts(capacity=capacity, device=self.default_device)

        # Execute narrowphase for primitive shapes
        with wp.ScopedTimer("primitive_narrowphase"):
            primitive_narrowphase(model, state, collisions, contacts)

        # Optional verbose output
        if self.verbose:
            print(f"contacts.num_model_max_contacts: {contacts.num_model_max_contacts}")
            print(f"contacts.num_world_max_contacts: {contacts.num_world_max_contacts}")
            print(f"contacts.model_max_contacts: {contacts.model_max_contacts}")
            print(f"contacts.model_num_contacts: {contacts.model_num_contacts}")
            print(f"contacts.world_max_contacts: {contacts.world_max_contacts}")
            print(f"contacts.world_num_contacts: {contacts.world_num_contacts}")
            print(f"contacts.wid: {contacts.wid}")
            print(f"contacts.cid: {contacts.cid}")
            print(f"contacts.body_A:\n{contacts.body_A}")
            print(f"contacts.body_B:\n{contacts.body_B}")
            print(f"contacts.gapfunc:\n{contacts.gapfunc}")
            print(f"contacts.frame:\n{contacts.frame}")
            print(f"contacts.material:\n{contacts.material}")

    def test_04_collision_detector(self):
        # Create and set up a model builder
        builder = make_homogeneous_builder(num_worlds=10, build_fn=self.build_func)

        # Finalize the model
        model = builder.finalize(self.default_device)

        # Create a state container
        state = model.data()

        # Create a collision detector
        detector = CollisionDetector(
            builder=builder, default_max_contacts=self.max_contacts, device=self.default_device
        )

        # Peroform collision detection
        with wp.ScopedTimer("detector.collide"):
            detector.collide(model, state)

        # Optional verbose output
        if self.verbose:
            print(f"detector.contacts.num_model_max_contacts: {detector.contacts.num_model_max_contacts}")
            print(f"detector.contacts.num_world_max_contacts: {detector.contacts.num_world_max_contacts}")
            print(f"detector.contacts.model_max_contacts: {detector.contacts.model_max_contacts}")
            print(f"detector.contacts.model_num_contacts: {detector.contacts.model_num_contacts}")
            print(f"detector.contacts.world_max_contacts: {detector.contacts.world_max_contacts}")
            print(f"detector.contacts.world_num_contacts: {detector.contacts.world_num_contacts}")
            print(f"detector.contacts.wid: {detector.contacts.wid}")
            print(f"detector.contacts.cid: {detector.contacts.cid}")
            print(f"detector.contacts.body_A:\n{detector.contacts.body_A}")
            print(f"detector.contacts.body_B:\n{detector.contacts.body_B}")
            print(f"detector.contacts.gapfunc:\n{detector.contacts.gapfunc}")
            print(f"detector.contacts.frame:\n{detector.contacts.frame}")
            print(f"detector.contacts.material:\n{detector.contacts.material}")


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
