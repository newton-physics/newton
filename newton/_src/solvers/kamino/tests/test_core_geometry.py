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

"""Unit tests for core geometry containers and operations"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.core.geometry import CollisionGeometryDescriptor, GeometryDescriptor
from newton._src.solvers.kamino.core.shapes import (
    MeshShape,
    ShapeType,
    SphereShape,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.utils import logger as msg

###
# Tests
###


class TestGeometryDescriptor(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for detailed output

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

    def test_00_primitive_shape_geom(self):
        # Create a geometry descriptor with a sphere shape
        geom = GeometryDescriptor(name="sphere_geom", shape=SphereShape(radius=0.42))
        # Check default values
        self.assertEqual(geom.name, "sphere_geom")
        self.assertEqual(geom.layer, "default")
        self.assertEqual(geom.bid, -1)
        self.assertEqual(geom.shape.type, ShapeType.SPHERE)
        self.assertEqual(geom.shape.num_params, 1)
        self.assertEqual(geom.shape.params, 0.42)
        self.assertEqual(geom.shape.name, "sphere")
        self.assertIsInstance(geom.shape.uid, str)

        # Check hash function
        geom_hash = hash(geom)
        geom_hash2 = hash(geom)
        shape_hash = hash(geom.shape)
        self.assertEqual(geom_hash, geom_hash2)
        self.assertEqual(shape_hash, shape_hash)
        msg.info(f"geom_hash: {geom_hash}")
        msg.info(f"geom_hash2: {geom_hash2}")
        msg.info(f"shape_hash: {shape_hash}")

    def test_01_mesh_shape_geom(self):
        # Create a geometry descriptor with a mesh shape
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        indices = [(0, 1, 2)]
        geom = GeometryDescriptor(name="mesh_geom", shape=MeshShape(vertices, indices))

        # Check default values
        self.assertEqual(geom.name, "mesh_geom")
        self.assertEqual(geom.layer, "default")
        self.assertEqual(geom.bid, -1)
        self.assertEqual(geom.shape.type, ShapeType.MESH)
        self.assertEqual(geom.shape.num_params, -1)
        self.assertEqual(geom.shape.params, 1.0)
        self.assertEqual(geom.shape.name, "mesh")
        self.assertIsInstance(geom.shape.uid, str)
        self.assertTrue(np.array_equal(geom.shape.vertices, np.array(vertices)))
        self.assertTrue(np.array_equal(geom.shape.indices, np.array(indices).flatten()))

        # Check hash function
        geom_hash = hash(geom)
        geom_hash2 = hash(geom)
        shape_hash = hash(geom.shape)
        self.assertEqual(geom_hash, geom_hash2)
        self.assertEqual(shape_hash, shape_hash)
        msg.info(f"geom_hash: {geom_hash}")
        msg.info(f"geom_hash2: {geom_hash2}")
        msg.info(f"shape_hash: {shape_hash}")


class TestCollisionGeometryDescriptor(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for detailed output

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

    def test_00_primitive_shape_geom(self):
        # Create a geometry descriptor with a sphere shape
        geom = CollisionGeometryDescriptor(name="sphere_geom", shape=SphereShape(radius=0.42))
        # Check default values
        self.assertEqual(geom.name, "sphere_geom")
        self.assertEqual(geom.layer, "default")
        self.assertEqual(geom.bid, -1)
        self.assertEqual(geom.shape.type, ShapeType.SPHERE)
        self.assertEqual(geom.shape.num_params, 1)
        self.assertEqual(geom.shape.params, 0.42)
        self.assertEqual(geom.shape.name, "sphere")
        self.assertIsInstance(geom.shape.uid, str)

        # Check hash function
        geom_hash = hash(geom)
        geom_hash2 = hash(geom)
        shape_hash = hash(geom.shape)
        self.assertEqual(geom_hash, geom_hash2)
        self.assertEqual(shape_hash, shape_hash)
        msg.info(f"geom_hash: {geom_hash}")
        msg.info(f"geom_hash2: {geom_hash2}")
        msg.info(f"shape_hash: {shape_hash}")

    def test_01_mesh_shape_geom(self):
        # Create a geometry descriptor with a mesh shape
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        indices = [(0, 1, 2)]
        geom = CollisionGeometryDescriptor(name="mesh_geom", shape=MeshShape(vertices, indices))

        # Check default values
        self.assertEqual(geom.name, "mesh_geom")
        self.assertEqual(geom.layer, "default")
        self.assertEqual(geom.bid, -1)
        self.assertEqual(geom.shape.type, ShapeType.MESH)
        self.assertEqual(geom.shape.num_params, -1)
        self.assertEqual(geom.shape.params, 1.0)
        self.assertEqual(geom.shape.name, "mesh")
        self.assertIsInstance(geom.shape.uid, str)
        self.assertTrue(np.array_equal(geom.shape.vertices, np.array(vertices)))
        self.assertTrue(np.array_equal(geom.shape.indices, np.array(indices).flatten()))

        # Check hash function
        geom_hash = hash(geom)
        geom_hash2 = hash(geom)
        shape_hash = hash(geom.shape)
        self.assertEqual(geom_hash, geom_hash2)
        self.assertEqual(shape_hash, shape_hash)
        msg.info(f"geom_hash: {geom_hash}")
        msg.info(f"geom_hash2: {geom_hash2}")
        msg.info(f"shape_hash: {shape_hash}")


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
