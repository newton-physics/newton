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

"""Unit tests for the collider functions of narrow-phase collision detection"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.core import ModelBuilder
from newton._src.solvers.kamino.core.geometry import update_collision_geometries_state
from newton._src.solvers.kamino.core.shapes import (
    BoxShape,
    SphereShape,
)
from newton._src.solvers.kamino.core.types import mat33f, transformf, vec6f
from newton._src.solvers.kamino.geometry.broadphase import nxn_broadphase
from newton._src.solvers.kamino.geometry.collisions import Collisions
from newton._src.solvers.kamino.geometry.contacts import Contacts
from newton._src.solvers.kamino.geometry.primitives import primitive_narrowphase
from newton._src.solvers.kamino.utils import logger as msg

###
# Builders
###


def build_sphere_on_sphere(dz_0: float = 0.0) -> ModelBuilder:
    """
    Builds a simple model with two spheres positioned along the z-axis.

    The first body (sphere 0) is placed below the second body (sphere 1) along the z-axis.

    The spheres are initially separated by a distance dz_0 (negative for penetration, positive for separation).

    Args:
        dz_0 (float): Initial distance between the two spheres along the z-axis.

    Returns:
        ModelBuilder: The constructed model builder with two spheres.
    """
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid0 = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid1 = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_layer("default")
    builder.add_collision_geometry(body=bid0, shape=SphereShape(0.5))
    builder.add_collision_geometry(body=bid1, shape=SphereShape(0.5))
    return builder


def build_box_on_box(dz_0: float = 0.0) -> ModelBuilder:
    """
    Builds a simple model with two boxes positioned along the z-axis.

    The first body (box 0) is placed below the second body (box 1) along the z-axis.

    The boxes are initially separated by a distance dz_0 (negative for penetration, positive for separation).

    Args:
        dz_0 (float): Initial distance between the two boxes along the z-axis.

    Returns:
        ModelBuilder: The constructed model builder with two boxes.
    """
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid0 = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid1 = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_layer("default")
    builder.add_collision_geometry(body=bid0, shape=BoxShape(2.0, 2.0, 1.0))
    builder.add_collision_geometry(body=bid1, shape=BoxShape(1.0, 1.0, 1.0))
    return builder


def build_sphere_on_box(dz_0: float = 0.0) -> ModelBuilder:
    """
    Builds a simple model with a sphere positioned above a box along the z-axis.

    The first body (box 0) is placed below the second body (sphere 1) along the z-axis.

    The bodies are initially separated by a distance dz_0 (negative for penetration, positive for separation).

    Args:
        dz_0 (float): Initial distance between the two boxes along the z-axis.

    Returns:
        ModelBuilder: The constructed model builder with two boxes.
    """
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid0 = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, -0.5 - 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bid1 = builder.add_rigid_body(
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(0.0, 0.0, 0.5 + 0.5 * dz_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_collision_layer("default")
    builder.add_collision_geometry(body=bid0, shape=SphereShape(0.5))
    builder.add_collision_geometry(body=bid1, shape=BoxShape(1.0, 1.0, 1.0))
    return builder


###
# Tests
###


class TestGeometryContacts(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()
        self.max_contacts = 16  # Maximum number of contacts for the test
        self.verbose = False  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.set_log_level(msg.LogLevel.WARNING)

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_sphere_on_sphere(self):
        # Define the initial body/geom separation distance
        # NOTE: This is the expected penetration depth (negative gap)
        dz_0 = -0.01

        # Create and set up a model builder
        builder = build_sphere_on_sphere(dz_0=dz_0)
        num_worlds = builder.num_worlds

        # Finalize the model and data
        model = builder.finalize(self.default_device)
        data = model.data()

        # Update the state of the collision geometries
        update_collision_geometries_state(data.bodies.q_i, model.cgeoms, data.cgeoms)

        # Create collisions container
        collisions = Collisions(builder=builder, device=self.default_device)

        # Execute brute-force (NxN) broadphase
        nxn_broadphase(model.cgeoms, data.cgeoms, collisions.cmodel, collisions.cdata)

        # Create a contacts container
        capacity = [self.max_contacts] * num_worlds  # Custom capacity for each world
        contacts = Contacts(capacity=capacity, device=self.default_device)

        # Execute narrowphase for primitive shapes
        primitive_narrowphase(model, data, collisions, contacts)

        # Check results
        self.assertEqual(contacts.model_num_contacts.numpy()[0], 1)
        self.assertEqual(contacts.world_num_contacts.numpy()[0], 1)
        self.assertEqual(contacts.wid.numpy()[0], 0)
        self.assertEqual(contacts.cid.numpy()[0], 0)
        self.assertEqual(int(contacts.body_A.numpy()[0][3]), 0)  # Body-A index
        self.assertEqual(int(contacts.body_B.numpy()[0][3]), 1)  # Body-B index
        np.testing.assert_almost_equal(contacts.body_A.numpy()[0][0:3], [0.0, 0.0, -0.5 * dz_0])  # Body-A position
        np.testing.assert_almost_equal(contacts.body_B.numpy()[0][0:3], [0.0, 0.0, 0.5 * dz_0])  # Body-B position
        np.testing.assert_almost_equal(contacts.gapfunc.numpy()[0], [0.0, 0.0, 1.0, dz_0], decimal=6)
        np.testing.assert_almost_equal(contacts.frame.numpy()[0], np.eye(3), decimal=6)

        # Optional verbose output
        msg.info(f"bodies.q_i:\n{data.bodies.q_i}")
        msg.info(f"contacts.model_num_contacts: {contacts.model_num_contacts}")
        msg.info(f"contacts.world_num_contacts: {contacts.world_num_contacts}")
        msg.info(f"contacts.wid: {contacts.wid}")
        msg.info(f"contacts.cid: {contacts.cid}")
        msg.info(f"contacts.body_A:\n{contacts.body_A}")
        msg.info(f"contacts.body_B:\n{contacts.body_B}")
        msg.info(f"contacts.gapfunc:\n{contacts.gapfunc}")
        msg.info(f"contacts.frame:\n{contacts.frame}")
        msg.info(f"contacts.material:\n{contacts.material}")

    def test_02_box_on_box(self):
        # Define the initial body/geom separation distance
        dz_0 = -0.01

        # Create and set up a model builder
        builder = build_box_on_box(dz_0=dz_0)
        num_worlds = builder.num_worlds

        # Finalize the model and data
        model = builder.finalize(self.default_device)
        data = model.data()

        # Update the state of the collision geometries
        update_collision_geometries_state(data.bodies.q_i, model.cgeoms, data.cgeoms)

        # Create collisions container
        collisions = Collisions(builder=builder, device=self.default_device)

        # Execute brute-force (NxN) broadphase
        nxn_broadphase(model.cgeoms, data.cgeoms, collisions.cmodel, collisions.cdata)

        # Create a contacts container
        capacity = [self.max_contacts] * num_worlds  # Custom capacity for each world
        contacts = Contacts(capacity=capacity, device=self.default_device)

        # Execute narrowphase for primitive shapes
        primitive_narrowphase(model, data, collisions, contacts)

        # Check results
        self.assertEqual(contacts.model_num_contacts.numpy()[0], 4)
        self.assertEqual(contacts.world_num_contacts.numpy()[0], 4)
        for i in range(4):
            self.assertEqual(contacts.wid.numpy()[i], 0)
            self.assertEqual(contacts.cid.numpy()[i], i)
            self.assertEqual(int(contacts.body_A.numpy()[i][3]), 0)  # Body-A index
            self.assertEqual(int(contacts.body_B.numpy()[i][3]), 1)  # Body-B index
            np.testing.assert_almost_equal(contacts.gapfunc.numpy()[i], [0.0, 0.0, 1.0, dz_0], decimal=6)
            np.testing.assert_almost_equal(contacts.frame.numpy()[i], np.eye(3), decimal=6)

        # Optional verbose output
        msg.info(f"bodies.q_i:\n{data.bodies.q_i}")
        msg.info(f"contacts.model_num_contacts: {contacts.model_num_contacts}")
        msg.info(f"contacts.world_num_contacts: {contacts.world_num_contacts}")
        msg.info(f"contacts.wid: {contacts.wid}")
        msg.info(f"contacts.cid: {contacts.cid}")
        msg.info(f"contacts.body_A:\n{contacts.body_A}")
        msg.info(f"contacts.body_B:\n{contacts.body_B}")
        msg.info(f"contacts.gapfunc:\n{contacts.gapfunc}")
        msg.info(f"contacts.frame:\n{contacts.frame}")
        msg.info(f"contacts.material:\n{contacts.material}")

    def test_03_box_on_sphere(self):
        # Define the initial body/geom separation distance
        # NOTE: This is the expected penetration depth (negative gap)
        dz_0 = -0.01

        # Create and set up a model builder
        builder = build_sphere_on_box(dz_0=dz_0)
        num_worlds = builder.num_worlds

        # Finalize the model and data
        model = builder.finalize(self.default_device)
        data = model.data()

        # Update the state of the collision geometries
        update_collision_geometries_state(data.bodies.q_i, model.cgeoms, data.cgeoms)

        # Create collisions container
        collisions = Collisions(builder=builder, device=self.default_device)

        # Execute brute-force (NxN) broadphase
        nxn_broadphase(model.cgeoms, data.cgeoms, collisions.cmodel, collisions.cdata)

        # Create a contacts container
        capacity = [self.max_contacts] * num_worlds  # Custom capacity for each world
        contacts = Contacts(capacity=capacity, device=self.default_device)

        # Execute narrowphase for primitive shapes
        primitive_narrowphase(model, data, collisions, contacts)

        # Check results
        self.assertEqual(contacts.model_num_contacts.numpy()[0], 1)
        self.assertEqual(contacts.world_num_contacts.numpy()[0], 1)
        self.assertEqual(contacts.wid.numpy()[0], 0)
        self.assertEqual(contacts.cid.numpy()[0], 0)
        self.assertEqual(int(contacts.body_A.numpy()[0][3]), 0)  # Body-A index
        self.assertEqual(int(contacts.body_B.numpy()[0][3]), 1)  # Body-B index
        np.testing.assert_almost_equal(contacts.body_A.numpy()[0][0:3], [0.0, 0.0, -0.5 * dz_0])  # Body-A position
        np.testing.assert_almost_equal(contacts.body_B.numpy()[0][0:3], [0.0, 0.0, 0.5 * dz_0])  # Body-B position
        np.testing.assert_almost_equal(contacts.gapfunc.numpy()[0], [0.0, 0.0, 1.0, dz_0], decimal=6)
        np.testing.assert_almost_equal(contacts.frame.numpy()[0], np.eye(3), decimal=6)

        # Optional verbose output
        msg.info(f"bodies.q_i:\n{data.bodies.q_i}")
        msg.info(f"contacts.model_num_contacts: {contacts.model_num_contacts}")
        msg.info(f"contacts.world_num_contacts: {contacts.world_num_contacts}")
        msg.info(f"contacts.wid: {contacts.wid}")
        msg.info(f"contacts.cid: {contacts.cid}")
        msg.info(f"contacts.body_A:\n{contacts.body_A}")
        msg.info(f"contacts.body_B:\n{contacts.body_B}")
        msg.info(f"contacts.gapfunc:\n{contacts.gapfunc}")
        msg.info(f"contacts.frame:\n{contacts.frame}")
        msg.info(f"contacts.material:\n{contacts.material}")


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=20000, threshold=20000, precision=7, suppress=True)

    # Global warp configurations
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
