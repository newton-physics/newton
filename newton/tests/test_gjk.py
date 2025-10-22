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

"""Tests for GJK distance computation using the new simplex solver."""

import unittest

import warp as wp

from newton import GeoType
from newton._src.geometry.simplex_solver import create_solve_closest_distance
from newton._src.geometry.support_function import GenericShapeData, SupportMapDataProvider, support_map

MAX_ITERATIONS = 30


@wp.kernel
def _gjk_kernel(
    type_a: int,
    size_a: wp.vec3,
    pos_a: wp.vec3,
    quat_a: wp.quat,
    type_b: int,
    size_b: wp.vec3,
    pos_b: wp.vec3,
    quat_b: wp.quat,
    # Outputs:
    collision_out: wp.array(dtype=int),
    dist_out: wp.array(dtype=float),
    point_out: wp.array(dtype=wp.vec3),
    normal_out: wp.array(dtype=wp.vec3),
):
    """Kernel to compute GJK distance between two shapes."""
    # Create shape data for both geometries
    shape_a = GenericShapeData()
    shape_a.shape_type = type_a
    shape_a.scale = size_a
    shape_a.auxiliary = wp.vec3(0.0)

    shape_b = GenericShapeData()
    shape_b.shape_type = type_b
    shape_b.scale = size_b
    shape_b.auxiliary = wp.vec3(0.0)

    data_provider = SupportMapDataProvider()

    # Call GJK solver
    collision, distance, point, normal, _feature_a, _feature_b = wp.static(create_solve_closest_distance(support_map))(
        shape_a,
        shape_b,
        quat_a,
        quat_b,
        pos_a,
        pos_b,
        0.0,  # sum_of_contact_offsets
        data_provider,
        MAX_ITERATIONS,
        1e-6,  # COLLIDE_EPSILON
    )

    collision_out[0] = int(collision)
    dist_out[0] = distance
    point_out[0] = point
    normal_out[0] = normal


def _geom_dist(
    geom_type1: int,
    size1: wp.vec3,
    pos1: wp.vec3,
    quat1: wp.quat,
    geom_type2: int,
    size2: wp.vec3,
    pos2: wp.vec3,
    quat2: wp.quat,
):
    """
    Compute distance between two geometries using GJK algorithm.

    Returns:
        Tuple of (distance, midpoint_contact_point, normal, collision_flag)
    """
    # Convert GeoType enums to int if needed
    type1 = int(geom_type1)
    type2 = int(geom_type2)

    collision_out = wp.zeros(1, dtype=int)
    dist_out = wp.zeros(1, dtype=float)
    point_out = wp.zeros(1, dtype=wp.vec3)
    normal_out = wp.zeros(1, dtype=wp.vec3)

    wp.launch(
        _gjk_kernel,
        dim=1,
        inputs=[type1, size1, pos1, quat1, type2, size2, pos2, quat2],
        outputs=[collision_out, dist_out, point_out, normal_out],
    )

    return (
        dist_out.numpy()[0],
        point_out.numpy()[0],
        normal_out.numpy()[0],
        collision_out.numpy()[0],
    )


class TestGJK(unittest.TestCase):
    """Tests for GJK distance computation using the new simplex solver."""

    def test_spheres_distance(self):
        """Test distance between two separated spheres."""
        # Two spheres of radius 1.0, separated by distance 3.0
        # Expected distance: 3.0 - 1.0 - 1.0 = 1.0
        dist, _point, _normal, collision = _geom_dist(
            GeoType.SPHERE,
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(-1.5, 0.0, 0.0),
            wp.quat_identity(),
            GeoType.SPHERE,
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(1.5, 0.0, 0.0),
            wp.quat_identity(),
        )
        self.assertAlmostEqual(1.0, dist, places=5)
        self.assertEqual(0, collision)  # No collision

    def test_spheres_touching(self):
        """Test two touching spheres have zero distance."""
        # Two spheres of radius 1.0, centers at distance 2.0
        # Expected distance: 0.0 (just touching)
        dist, _point, _normal, _collision = _geom_dist(
            GeoType.SPHERE,
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(-1.0, 0.0, 0.0),
            wp.quat_identity(),
            GeoType.SPHERE,
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.quat_identity(),
        )
        self.assertAlmostEqual(0.0, dist, places=5)

    def test_sphere_sphere_overlapping(self):
        """Test overlapping spheres return collision=True and distance=0."""
        # Two spheres of radius 3.0, centers at distance 4.0
        # Expected overlap: 3.0 + 3.0 - 4.0 = 2.0
        # Note: GJK returns collision=True and distance=0 for overlapping shapes (MPR would give penetration depth)
        dist, _point, _normal, collision = _geom_dist(
            GeoType.SPHERE,
            wp.vec3(3.0, 0.0, 0.0),
            wp.vec3(-1.0, 0.0, 0.0),
            wp.quat_identity(),
            GeoType.SPHERE,
            wp.vec3(3.0, 0.0, 0.0),
            wp.vec3(3.0, 0.0, 0.0),
            wp.quat_identity(),
        )
        self.assertAlmostEqual(0.0, dist, places=5)
        self.assertEqual(1, collision)  # GJK reports collision=True for overlapping shapes

    def test_box_box_separated(self):
        """Test distance between two separated boxes."""
        # Two boxes: first is 5x5x5 (half-extents 2.5), second is 2x2x2 (half-extents 1.0)
        # First centered at (-1, 0, 0), second at (1.5, 0, 0)
        # Distance between centers: 2.5, half-extents sum: 3.5
        # Expected separation: 2.5 - 2.5 - 1.0 = -1.0 (overlapping)
        # But let's test a separated case
        dist, _point, normal, collision = _geom_dist(
            GeoType.BOX,
            wp.vec3(1.0, 1.0, 1.0),
            wp.vec3(-2.0, 0.0, 0.0),
            wp.quat_identity(),
            GeoType.BOX,
            wp.vec3(1.0, 1.0, 1.0),
            wp.vec3(2.5, 0.0, 0.0),
            wp.quat_identity(),
        )
        # Centers at distance 4.5, half-extents sum: 2.0
        # Expected distance: 4.5 - 1.0 - 1.0 = 2.5
        self.assertAlmostEqual(2.5, dist, places=5)
        self.assertEqual(0, collision)
        # Normal should point from A to B (positive X direction)
        self.assertAlmostEqual(normal[0], 1.0, places=5)
        self.assertAlmostEqual(normal[1], 0.0, places=5)
        self.assertAlmostEqual(normal[2], 0.0, places=5)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
