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
KAMINO: UNIT TESTS: CORE: SHAPES
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.core.shapes import (
    BoxShape,
    CapsuleShape,
    ConeShape,
    CylinderShape,
    EllipsoidShape,
    EmptyShape,
    MeshShape,
    PlaneShape,
    SDFShape,
    ShapeType,
    SphereShape,
)
from newton._src.solvers.kamino.core.types import mat33f, vec3f
from newton._src.solvers.kamino.utils import logger as msg

###
# Tests
###


class TestShapeType(unittest.TestCase):
    def test_00_empty_shape(self):
        type = ShapeType.EMPTY
        self.assertEqual(type, 0)
        self.assertEqual(type.num_params, 0)

    def test_01_sphere_shape(self):
        type = ShapeType.SPHERE
        self.assertEqual(type, 1)
        self.assertEqual(type.num_params, 1)

    def test_02_cylinder_shape(self):
        type = ShapeType.CYLINDER
        self.assertEqual(type, 2)
        self.assertEqual(type.num_params, 2)

    def test_03_cone_shape(self):
        type = ShapeType.CONE
        self.assertEqual(type, 3)
        self.assertEqual(type.num_params, 2)

    def test_04_capsule_shape(self):
        type = ShapeType.CAPSULE
        self.assertEqual(type, 4)
        self.assertEqual(type.num_params, 2)

    def test_05_box_shape(self):
        type = ShapeType.BOX
        self.assertEqual(type, 5)
        self.assertEqual(type.num_params, 3)

    def test_06_ellipsoid_shape(self):
        type = ShapeType.ELLIPSOID
        self.assertEqual(type, 6)
        self.assertEqual(type.num_params, 3)

    def test_07_plane_shape(self):
        type = ShapeType.PLANE
        self.assertEqual(type, 7)
        self.assertEqual(type.num_params, 4)

    def test_08_mesh_shape(self):
        type = ShapeType.MESH
        self.assertEqual(type, 8)
        self.assertEqual(type.num_params, -1)

    def test_09_convex_shape(self):
        type = ShapeType.CONVEX
        self.assertEqual(type, 9)
        self.assertEqual(type.num_params, -1)

    def test_10_hfield_shape(self):
        type = ShapeType.HFIELD
        self.assertEqual(type, 10)
        self.assertEqual(type.num_params, -1)

    def test_11_sdf_shape(self):
        type = ShapeType.SDF
        self.assertEqual(type, 11)
        self.assertEqual(type.num_params, -1)


class TestShapeDescriptors(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()
        self.verbose = False  # Set to True to enable verbose output

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

    def test_00_empty_shape(self):
        # Create a default-constructed surface material
        shape = EmptyShape()
        # Check default values
        self.assertEqual(shape.type, ShapeType.EMPTY)
        self.assertEqual(shape.num_params, 0)
        self.assertEqual(shape.params, None)
        self.assertEqual(shape.name, "empty")
        self.assertIsInstance(shape.uid, str)

        # Check hash function
        shape_hash = hash(shape)
        shape_hash2 = hash(shape)
        base_hash = super(EmptyShape, shape).__hash__()
        self.assertEqual(shape_hash, shape_hash2)
        self.assertEqual(shape_hash, base_hash)
        msg.info(f"EmptyShape hash: {shape_hash}")
        msg.info(f"EmptyShape hash (2): {shape_hash2}")
        msg.info(f"EmptyShape base hash: {base_hash}")

    def test_01_sphere_shape(self):
        # Create a sphere shape
        radius = 1.0
        shape = SphereShape(radius)
        # Check default values
        self.assertEqual(shape.name, "sphere")
        self.assertEqual(shape.type, ShapeType.SPHERE)
        self.assertEqual(shape.num_params, 1)
        self.assertEqual(shape.params, radius)

        # Check hash function
        shape_hash = hash(shape)
        shape_hash2 = hash(shape)
        base_hash = super(SphereShape, shape).__hash__()
        self.assertEqual(shape_hash, shape_hash2)
        self.assertEqual(shape_hash, base_hash)
        msg.info(f"SphereShape hash: {shape_hash}")
        msg.info(f"SphereShape hash (2): {shape_hash2}")
        msg.info(f"SphereShape base hash: {base_hash}")

    def test_02_cylinder_shape(self):
        # Create a cylinder shape
        radius = 0.5
        height = 2.0
        shape = CylinderShape(radius, height)
        # Check default values
        self.assertEqual(shape.name, "cylinder")
        self.assertEqual(shape.type, ShapeType.CYLINDER)
        self.assertEqual(shape.num_params, 2)
        self.assertEqual(shape.params, (radius, height))

        # Check hash function
        shape_hash = hash(shape)
        shape_hash2 = hash(shape)
        base_hash = super(CylinderShape, shape).__hash__()
        self.assertEqual(shape_hash, shape_hash2)
        self.assertEqual(shape_hash, base_hash)
        msg.info(f"CylinderShape hash: {shape_hash}")
        msg.info(f"CylinderShape hash (2): {shape_hash2}")
        msg.info(f"CylinderShape base hash: {base_hash}")

    def test_03_cone_shape(self):
        # Create a cone shape
        radius = 0.5
        height = 2.0
        shape = ConeShape(radius, height)
        # Check default values
        self.assertEqual(shape.name, "cone")
        self.assertEqual(shape.type, ShapeType.CONE)
        self.assertEqual(shape.num_params, 2)
        self.assertEqual(shape.params, (radius, height))

        # Check hash function
        shape_hash = hash(shape)
        shape_hash2 = hash(shape)
        base_hash = super(ConeShape, shape).__hash__()
        self.assertEqual(shape_hash, shape_hash2)
        self.assertEqual(shape_hash, base_hash)
        msg.info(f"ConeShape hash: {shape_hash}")
        msg.info(f"ConeShape hash (2): {shape_hash2}")
        msg.info(f"ConeShape base hash: {base_hash}")

    def test_04_capsule_shape(self):
        # Create a capsule shape
        radius = 0.5
        height = 2.0
        shape = CapsuleShape(radius, height)
        # Check default values
        self.assertEqual(shape.name, "capsule")
        self.assertEqual(shape.type, ShapeType.CAPSULE)
        self.assertEqual(shape.num_params, 2)
        self.assertEqual(shape.params, (radius, height))

        # Check hash function
        shape_hash = hash(shape)
        shape_hash2 = hash(shape)
        base_hash = super(CapsuleShape, shape).__hash__()
        self.assertEqual(shape_hash, shape_hash2)
        self.assertEqual(shape_hash, base_hash)
        msg.info(f"CapsuleShape hash: {shape_hash}")
        msg.info(f"CapsuleShape hash (2): {shape_hash2}")
        msg.info(f"CapsuleShape base hash: {base_hash}")

    def test_05_box_shape(self):
        # Create a box shape
        dimensions = (1.0, 2.0, 3.0)
        shape = BoxShape(*dimensions)
        # Check default values
        self.assertEqual(shape.name, "box")
        self.assertEqual(shape.type, ShapeType.BOX)
        self.assertEqual(shape.num_params, 3)
        self.assertEqual(shape.params, dimensions)

        # Check hash function
        shape_hash = hash(shape)
        shape_hash2 = hash(shape)
        base_hash = super(BoxShape, shape).__hash__()
        self.assertEqual(shape_hash, shape_hash2)
        self.assertEqual(shape_hash, base_hash)
        msg.info(f"BoxShape hash: {shape_hash}")
        msg.info(f"BoxShape hash (2): {shape_hash2}")
        msg.info(f"BoxShape base hash: {base_hash}")

    def test_06_ellipsoid_shape(self):
        # Create an ellipsoid shape
        radii = (1.0, 2.0, 3.0)
        shape = EllipsoidShape(*radii)
        # Check default values
        self.assertEqual(shape.name, "ellipsoid")
        self.assertEqual(shape.type, ShapeType.ELLIPSOID)
        self.assertEqual(shape.num_params, 3)
        self.assertEqual(shape.params, radii)

        # Check hash function
        shape_hash = hash(shape)
        shape_hash2 = hash(shape)
        base_hash = super(EllipsoidShape, shape).__hash__()
        self.assertEqual(shape_hash, shape_hash2)
        self.assertEqual(shape_hash, base_hash)
        msg.info(f"EllipsoidShape hash: {shape_hash}")
        msg.info(f"EllipsoidShape hash (2): {shape_hash2}")
        msg.info(f"EllipsoidShape base hash: {base_hash}")

    def test_07_plane_shape(self):
        # Create a plane shape
        normal = (0.0, 1.0, 0.0)
        distance = 0.5
        shape = PlaneShape(normal, distance)
        # Check default values
        self.assertEqual(shape.name, "plane")
        self.assertEqual(shape.type, ShapeType.PLANE)
        self.assertEqual(shape.num_params, 4)
        self.assertEqual(shape.params, (*normal, distance))

        # Check hash function
        shape_hash = hash(shape)
        shape_hash2 = hash(shape)
        base_hash = super(PlaneShape, shape).__hash__()
        self.assertEqual(shape_hash, shape_hash2)
        self.assertEqual(shape_hash, base_hash)
        msg.info(f"PlaneShape hash: {shape_hash}")
        msg.info(f"PlaneShape hash (2): {shape_hash2}")
        msg.info(f"PlaneShape base hash: {base_hash}")

    def test_08_mesh_shape(self):
        # Create a mesh shape
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        indices = [(0, 1, 2)]
        shape = MeshShape(vertices, indices)
        # Check default values
        self.assertEqual(shape.name, "mesh")
        self.assertEqual(shape.type, ShapeType.MESH)
        self.assertEqual(shape.num_params, -1)
        self.assertEqual(shape.params, 1.0)
        self.assertTrue(np.array_equal(shape.vertices, np.array(vertices)))
        self.assertTrue(np.array_equal(shape.indices, np.array(indices).flatten()))

        # Check hash function
        shape_hash = hash(shape)
        shape_hash2 = hash(shape)
        base_hash = super(MeshShape, shape).__hash__()
        self.assertEqual(shape_hash, shape_hash2)
        self.assertNotEqual(shape_hash, base_hash)
        msg.info(f"MeshShape hash: {shape_hash}")
        msg.info(f"MeshShape hash (2): {shape_hash2}")
        msg.info(f"MeshShape base hash: {base_hash}")

    def test_09_convex_shape(self):
        # Create a mesh shape
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        indices = [(0, 1, 2)]
        shape = MeshShape(vertices, indices, is_convex=True)
        # Check default values
        self.assertEqual(shape.name, "convex")
        self.assertEqual(shape.type, ShapeType.CONVEX)
        self.assertEqual(shape.num_params, -1)
        self.assertEqual(shape.params, 1.0)
        self.assertTrue(np.array_equal(shape.vertices, np.array(vertices)))
        self.assertTrue(np.array_equal(shape.indices, np.array(indices).flatten()))

        # Check hash function
        shape_hash = hash(shape)
        shape_hash2 = hash(shape)
        base_hash = super(MeshShape, shape).__hash__()
        self.assertEqual(shape_hash, shape_hash2)
        self.assertNotEqual(shape_hash, base_hash)
        msg.info(f"MeshShape hash: {shape_hash}")
        msg.info(f"MeshShape hash (2): {shape_hash2}")
        msg.info(f"MeshShape base hash: {base_hash}")

    # TODO: Re-enable when HFieldShape is implemented
    # def test_10_hfield_shape(self):
    #     # Create a height-field shape
    #     vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    #     indices = [(0, 1, 2)]
    #     shape = HFieldShape(vertices, indices)
    #     # Check default values
    #     self.assertEqual(shape.name, "hfield")
    #     self.assertEqual(shape.type, ShapeType.HFIELD)
    #     self.assertEqual(shape.num_params, -1)
    #     self.assertEqual(shape.params, None)
    #     self.assertTrue(np.array_equal(shape.vertices, np.array(vertices)))
    #     self.assertTrue(np.array_equal(shape.indices, np.array(indices).flatten()))

    def test_11_sdf_shape(self):
        if self.default_device.is_cpu:
            self.skipTest("SDFShape tests are skipped on CPU device.")
        # Create an SDF shape
        volume = wp.Volume.allocate((-10, -10, -10), (10, 10, 10), voxel_size=0.5, device=self.default_device)
        shape = SDFShape(volume)
        # Check default values
        self.assertEqual(shape.name, "sdf")
        self.assertEqual(shape.type, ShapeType.SDF)
        self.assertEqual(shape.num_params, -1)
        self.assertEqual(shape.params, None)
        self.assertEqual(shape.mass, 1.0)
        self.assertEqual(shape.com, vec3f(0.0))
        self.assertEqual(shape.inertia, mat33f(np.eye(3)))

        # Check hash function
        shape_hash = hash(shape)
        shape_hash2 = hash(shape)
        base_hash = super(SDFShape, shape).__hash__()
        self.assertEqual(shape_hash, shape_hash2)
        self.assertNotEqual(shape_hash, base_hash)
        msg.info(f"SDFShape hash: {shape_hash}")
        msg.info(f"SDFShape hash (2): {shape_hash2}")
        msg.info(f"SDFShape base hash: {base_hash}")


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=2000, precision=10, threshold=20000, suppress=True)

    # Global warp configurations
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
