# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: UNIT TESTS: CORE: SHAPES
"""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.types import GeoType
from newton._src.solvers.kamino._src.core.shapes import (
    BoxShape,
    CapsuleShape,
    ConeShape,
    CylinderShape,
    EllipsoidShape,
    EmptyShape,
    MeshShape,
    PlaneShape,
    SphereShape,
)
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context

###
# Tests
###


class TestShapeDescriptors(unittest.TestCase):
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

    def test_00_empty_shape(self):
        # Create a default-constructed surface material
        shape = EmptyShape()
        # Check default values
        self.assertEqual(shape.type, GeoType.NONE)
        self.assertEqual(shape.params, None)
        self.assertEqual(shape.name, "empty")
        self.assertIsInstance(shape.uid, str)

        # Check hash function
        shape_hash = hash(shape)
        shape_hash2 = hash(shape)
        base_hash = super(EmptyShape, shape).__hash__()
        self.assertEqual(shape_hash, shape_hash2)
        self.assertEqual(shape_hash, base_hash)

    def test_01_sphere_shape(self):
        # Create a sphere shape
        radius = 1.0
        shape = SphereShape(radius)
        # Check default values
        self.assertEqual(shape.name, "sphere")
        self.assertEqual(shape.type, GeoType.SPHERE)
        self.assertEqual(shape.params, radius)

        # Check hash function
        shape_hash = hash(shape)
        shape_hash2 = hash(shape)
        base_hash = super(SphereShape, shape).__hash__()
        self.assertEqual(shape_hash, shape_hash2)
        self.assertEqual(shape_hash, base_hash)

    def test_02_cylinder_shape(self):
        # Create a cylinder shape (Newton convention: half-height)
        radius = 0.5
        half_height = 1.0
        shape = CylinderShape(radius, half_height)
        # Check default values
        self.assertEqual(shape.name, "cylinder")
        self.assertEqual(shape.type, GeoType.CYLINDER)
        self.assertEqual(shape.params, (radius, half_height))

        # Check hash function
        shape_hash = hash(shape)
        shape_hash2 = hash(shape)
        base_hash = super(CylinderShape, shape).__hash__()
        self.assertEqual(shape_hash, shape_hash2)
        self.assertEqual(shape_hash, base_hash)

    def test_03_cone_shape(self):
        # Create a cone shape (Newton convention: half-height)
        radius = 0.5
        half_height = 1.0
        shape = ConeShape(radius, half_height)
        # Check default values
        self.assertEqual(shape.name, "cone")
        self.assertEqual(shape.type, GeoType.CONE)
        self.assertEqual(shape.params, (radius, half_height))

        # Check hash function
        shape_hash = hash(shape)
        shape_hash2 = hash(shape)
        base_hash = super(ConeShape, shape).__hash__()
        self.assertEqual(shape_hash, shape_hash2)
        self.assertEqual(shape_hash, base_hash)

    def test_04_capsule_shape(self):
        # Create a capsule shape (Newton convention: half-height)
        radius = 0.5
        half_height = 1.0
        shape = CapsuleShape(radius, half_height)
        # Check default values
        self.assertEqual(shape.name, "capsule")
        self.assertEqual(shape.type, GeoType.CAPSULE)
        self.assertEqual(shape.params, (radius, half_height))

        # Check hash function
        shape_hash = hash(shape)
        shape_hash2 = hash(shape)
        base_hash = super(CapsuleShape, shape).__hash__()
        self.assertEqual(shape_hash, shape_hash2)
        self.assertEqual(shape_hash, base_hash)

    def test_05_box_shape(self):
        # Create a box shape (Newton convention: half-extents)
        half_extents = (0.5, 1.0, 1.5)
        shape = BoxShape(*half_extents)
        # Check default values
        self.assertEqual(shape.name, "box")
        self.assertEqual(shape.type, GeoType.BOX)
        self.assertEqual(shape.params, half_extents)

        # Check hash function
        shape_hash = hash(shape)
        shape_hash2 = hash(shape)
        base_hash = super(BoxShape, shape).__hash__()
        self.assertEqual(shape_hash, shape_hash2)
        self.assertEqual(shape_hash, base_hash)

    def test_06_ellipsoid_shape(self):
        # Create an ellipsoid shape
        radii = (1.0, 2.0, 3.0)
        shape = EllipsoidShape(*radii)
        # Check default values
        self.assertEqual(shape.name, "ellipsoid")
        self.assertEqual(shape.type, GeoType.ELLIPSOID)
        self.assertEqual(shape.params, radii)

        # Check hash function
        shape_hash = hash(shape)
        shape_hash2 = hash(shape)
        base_hash = super(EllipsoidShape, shape).__hash__()
        self.assertEqual(shape_hash, shape_hash2)
        self.assertEqual(shape_hash, base_hash)

    def test_07_plane_shape(self):
        # Create a plane shape
        normal = (0.0, 1.0, 0.0)
        distance = 0.5
        shape = PlaneShape(normal, distance)
        # Check default values
        self.assertEqual(shape.name, "plane")
        self.assertEqual(shape.type, GeoType.PLANE)
        self.assertEqual(shape.params, (*normal, distance))

        # Check hash function
        shape_hash = hash(shape)
        shape_hash2 = hash(shape)
        base_hash = super(PlaneShape, shape).__hash__()
        self.assertEqual(shape_hash, shape_hash2)
        self.assertEqual(shape_hash, base_hash)

    def test_08_mesh_shape(self):
        # Create a mesh shape
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        indices = [(0, 1, 2)]
        shape = MeshShape(vertices, indices)
        # Check default values
        self.assertEqual(shape.name, "mesh")
        self.assertEqual(shape.type, GeoType.MESH)
        self.assertEqual(shape.params, (1.0, 1.0, 1.0))
        self.assertTrue(np.array_equal(shape.vertices, np.array(vertices)))
        self.assertTrue(np.array_equal(shape.indices, np.array(indices).flatten()))

        # Check hash function
        shape_hash = hash(shape)
        shape_hash2 = hash(shape)
        base_hash = super(MeshShape, shape).__hash__()
        self.assertEqual(shape_hash, shape_hash2)
        self.assertNotEqual(shape_hash, base_hash)

    def test_09_convex_shape(self):
        # Create a convex mesh shape
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        indices = [(0, 1, 2)]
        shape = MeshShape(vertices, indices, is_convex=True)
        # Check default values
        self.assertEqual(shape.name, "convex")
        self.assertEqual(shape.type, GeoType.CONVEX_MESH)
        self.assertEqual(shape.params, (1.0, 1.0, 1.0))
        self.assertTrue(np.array_equal(shape.vertices, np.array(vertices)))
        self.assertTrue(np.array_equal(shape.indices, np.array(indices).flatten()))

        # Check hash function
        shape_hash = hash(shape)
        shape_hash2 = hash(shape)
        base_hash = super(MeshShape, shape).__hash__()
        self.assertEqual(shape_hash, shape_hash2)
        self.assertNotEqual(shape_hash, base_hash)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
