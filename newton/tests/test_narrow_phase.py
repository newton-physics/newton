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

"""Test NarrowPhase collision detection API.

This test suite validates the NarrowPhase API by testing various primitive collision scenarios.
The tests follow the same conventions as test_collision_primitives.py:

1. **Normal Direction**: Contact normals point from shape A (first geom) toward shape B (second geom)
2. **Penetration Depth**: Negative values indicate penetration, positive values indicate separation
3. **Surface Reconstruction**: Moving ±penetration_depth/2 along the normal from the contact point
   should land on the respective surfaces of each geometry
4. **Unit Normals**: All contact normals should have unit length
5. **Perpendicular Tangents**: Contact tangents should be perpendicular to normals

These validations ensure the NarrowPhase follows the same contact conventions as the
primitive collision functions.
"""

import unittest

import numpy as np
import warp as wp
from warp.tests.unittest_utils import StdOutCapture

from newton._src.geometry.flags import ShapeFlags
from newton._src.geometry.narrow_phase import NarrowPhase
from newton._src.geometry.sdf_utils import SDFData
from newton._src.geometry.types import GeoType


def check_normal_direction(pos_a, pos_b, normal, tolerance=1e-5):
    """Check that normal points from shape A toward shape B."""
    expected_direction = pos_b - pos_a
    expected_direction_norm = np.linalg.norm(expected_direction)
    if expected_direction_norm > tolerance:
        expected_direction = expected_direction / expected_direction_norm
        dot_product = np.dot(normal, expected_direction)
        return dot_product > (1.0 - tolerance)
    return True  # Can't determine direction if centers coincide


def check_contact_position_midpoint_spheres(
    contact_pos, normal, penetration_depth, pos_a, radius_a, pos_b, radius_b, tolerance=0.05
):
    """Check that contact position is at the midpoint between the two sphere surfaces.

    For sphere-sphere collision:
    - Moving from contact_pos by -penetration_depth/2 along normal should reach surface of sphere A
    - Moving from contact_pos by +penetration_depth/2 along normal should reach surface of sphere B
    """
    if penetration_depth >= 0:
        # For separated or just touching cases, position is still at midpoint
        # but we can't validate surface points the same way
        return True

    # Point on surface of geom 0 (sphere A)
    surface_point_0 = contact_pos - normal * (penetration_depth / 2.0)
    # Distance from this point to sphere A center should equal radius_a
    dist_to_sphere_a = np.linalg.norm(surface_point_0 - pos_a)

    # Point on surface of geom 1 (sphere B)
    surface_point_1 = contact_pos + normal * (penetration_depth / 2.0)
    # Distance from this point to sphere B center should equal radius_b
    dist_to_sphere_b = np.linalg.norm(surface_point_1 - pos_b)

    return abs(dist_to_sphere_a - radius_a) < tolerance and abs(dist_to_sphere_b - radius_b) < tolerance


def distance_point_to_box(point, box_pos, box_rot, box_size):
    """Calculate distance from a point to a box surface.

    Args:
        point: Point to check (world space)
        box_pos: Box center position
        box_rot: Box rotation matrix (3x3)
        box_size: Box half-extents
    """
    # Transform point to box local coordinates
    local_point = np.dot(box_rot.T, point - box_pos)

    # Clamp to box bounds
    clamped = np.clip(local_point, -box_size, box_size)

    # Distance from point to closest point on/in box
    return np.linalg.norm(local_point - clamped)


def distance_point_to_capsule(point, capsule_pos, capsule_axis, capsule_radius, capsule_half_length):
    """Calculate distance from a point to a capsule surface."""
    segment = capsule_axis * capsule_half_length
    start = capsule_pos - segment
    end = capsule_pos + segment

    # Find closest point on capsule centerline
    ab = end - start
    t = np.dot(point - start, ab) / (np.dot(ab, ab) + 1e-6)
    t = np.clip(t, 0.0, 1.0)
    closest_on_line = start + t * ab

    # Distance to capsule surface
    dist_to_centerline = np.linalg.norm(point - closest_on_line)
    return abs(dist_to_centerline - capsule_radius)


def distance_point_to_plane(point, plane_pos, plane_normal):
    """Calculate signed distance from a point to a plane."""
    return np.dot(point - plane_pos, plane_normal)


def distance_point_to_ellipsoid(point, ellipsoid_pos, ellipsoid_rot, semi_axes):
    """Calculate approximate distance from a point to an ellipsoid surface.

    Args:
        point: Point to check (world space)
        ellipsoid_pos: Ellipsoid center position
        ellipsoid_rot: Ellipsoid rotation matrix (3x3)
        semi_axes: Semi-axes (a, b, c) along local x, y, z

    Returns:
        Approximate distance to ellipsoid surface
    """
    # Transform point to ellipsoid local coordinates
    local_point = np.dot(ellipsoid_rot.T, point - ellipsoid_pos)

    # Scale to unit sphere
    a, b, c = semi_axes
    scaled_point = local_point / np.array([a, b, c])

    # Distance from unit sphere surface
    dist_to_unit_sphere = np.linalg.norm(scaled_point) - 1.0

    # Approximate distance (this is not exact but good enough for tests)
    avg_scale = (a + b + c) / 3.0
    return dist_to_unit_sphere * avg_scale


def check_surface_reconstruction(contact_pos, normal, penetration_depth, dist_func_a, dist_func_b, tolerance=0.08):
    """Verify that contact position is at midpoint between surfaces.

    Args:
        contact_pos: Contact position in world space
        normal: Contact normal (pointing from A to B)
        penetration_depth: Penetration depth (negative for penetration)
        dist_func_a: Function that calculates distance to surface A
        dist_func_b: Function that calculates distance to surface B
        tolerance: Tolerance for distance checks

    Returns:
        True if surface reconstruction is valid
    """
    if penetration_depth >= 0:
        # For separated or just touching cases, we can't validate the same way
        return True

    # Point on surface of geom A (shape 0)
    surface_point_a = contact_pos - normal * (penetration_depth / 2.0)
    dist_to_surface_a = dist_func_a(surface_point_a)

    # Point on surface of geom B (shape 1)
    surface_point_b = contact_pos + normal * (penetration_depth / 2.0)
    dist_to_surface_b = dist_func_b(surface_point_b)

    return dist_to_surface_a < tolerance and dist_to_surface_b < tolerance


class TestNarrowPhase(unittest.TestCase):
    """Test NarrowPhase collision detection API with various primitive pairs."""

    def setUp(self):
        """Set up narrow phase instance for tests."""
        # Use reasonable buffer sizes for tests
        # Tests typically use small numbers of shapes (< 100)
        max_pairs = 10000  # Conservative estimate for test scenarios
        self.narrow_phase = NarrowPhase(
            max_candidate_pairs=max_pairs,
            max_triangle_pairs=100000,
            device=None,
        )
        self.contact_margin = 0.01

    def _create_geometry_arrays(self, geom_list):
        """Create geometry arrays from a list of geometry descriptions.

        Each geometry is a dict with:
            - type: GeoType value
            - transform: (position, quaternion) tuple
            - data: scale/size as vec3, thickness as float
            - source: mesh pointer (default 0)
            - cutoff: contact margin (default 0.0)

        Returns:
            Tuple of (geom_types, geom_data, geom_transform, geom_source, shape_contact_margin, geom_collision_radius)
        """
        n = len(geom_list)

        geom_types = np.zeros(n, dtype=np.int32)
        geom_data = np.zeros(n, dtype=wp.vec4)
        geom_transforms = []
        geom_source = np.zeros(n, dtype=np.uint64)
        shape_contact_margin = np.zeros(n, dtype=np.float32)
        geom_collision_radius = np.zeros(n, dtype=np.float32)

        for i, geom in enumerate(geom_list):
            geom_types[i] = int(geom["type"])

            # Data: (scale_x, scale_y, scale_z, thickness)
            data = geom.get("data", ([1.0, 1.0, 1.0], 0.0))
            if isinstance(data, tuple):
                scale, thickness = data
            else:
                scale = data
                thickness = 0.0
            geom_data[i] = wp.vec4(scale[0], scale[1], scale[2], thickness)

            # Transform: position and quaternion
            pos, quat = geom.get("transform", ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]))
            geom_transforms.append(
                wp.transform(wp.vec3(pos[0], pos[1], pos[2]), wp.quat(quat[0], quat[1], quat[2], quat[3]))
            )

            geom_source[i] = geom.get("source", 0)
            shape_contact_margin[i] = geom.get("cutoff", 0.0)

            # Compute collision radius for AABB fallback (used for planes/meshes)
            geo_type = geom_types[i]
            scale_array = np.array(scale)
            if geo_type == int(GeoType.SPHERE):
                geom_collision_radius[i] = scale_array[0]
            elif geo_type == int(GeoType.BOX):
                geom_collision_radius[i] = np.linalg.norm(scale_array)
            elif geo_type == int(GeoType.CAPSULE) or geo_type == int(GeoType.CYLINDER) or geo_type == int(GeoType.CONE):
                geom_collision_radius[i] = scale_array[0] + scale_array[1]
            elif geo_type == int(GeoType.ELLIPSOID):
                # Bounding sphere radius is the largest semi-axis
                geom_collision_radius[i] = max(scale_array[0], scale_array[1], scale_array[2])
            elif geo_type == int(GeoType.PLANE):
                if scale_array[0] > 0.0 and scale_array[1] > 0.0:
                    # finite plane
                    geom_collision_radius[i] = np.linalg.norm(scale_array)
                else:
                    # infinite plane
                    geom_collision_radius[i] = 1.0e6
            else:
                # Default for other types (mesh, etc.)
                geom_collision_radius[i] = np.linalg.norm(scale_array) if len(scale_array) >= 3 else 10.0

        return (
            wp.array(geom_types, dtype=wp.int32),
            wp.array(geom_data, dtype=wp.vec4),
            wp.array(geom_transforms, dtype=wp.transform),
            wp.array(geom_source, dtype=wp.uint64),
            wp.array(shape_contact_margin, dtype=wp.float32),
            wp.array(geom_collision_radius, dtype=wp.float32),
            wp.zeros(len(geom_list), dtype=SDFData),  # shape_sdf_data - empty for non-mesh tests
            wp.full(
                len(geom_list), ShapeFlags.COLLIDE_SHAPES, dtype=wp.int32
            ),  # shape_flags - collision enabled, no hydroelastic
            wp.zeros(len(geom_list), dtype=wp.vec3),  # shape_collision_aabb_lower - dummy for non-mesh tests
            wp.ones(len(geom_list), dtype=wp.vec3),  # shape_collision_aabb_upper - dummy for non-mesh tests
            wp.full(len(geom_list), wp.vec3i(4, 4, 4), dtype=wp.vec3i),  # shape_voxel_resolution - dummy
        )

    def _run_narrow_phase(self, geom_list, pairs):
        """Run narrow phase on given geometry and pairs.

        Args:
            geom_list: List of geometry descriptions
            pairs: List of (i, j) tuples indicating which geometries to test

        Returns:
            Tuple of (contact_count, contact_pairs, positions, normals, penetrations, tangents)
        """
        (
            geom_types,
            geom_data,
            geom_transform,
            geom_source,
            shape_contact_margin,
            geom_collision_radius,
            shape_sdf_data,
            shape_flags,
            shape_collision_aabb_lower,
            shape_collision_aabb_upper,
            shape_voxel_resolution,
        ) = self._create_geometry_arrays(geom_list)

        # Create candidate pairs
        candidate_pair = wp.array(np.array(pairs, dtype=np.int32).reshape(-1, 2), dtype=wp.vec2i)
        candidate_pair_count = wp.array([len(pairs)], dtype=wp.int32)

        # Allocate output arrays
        max_contacts = len(pairs) * 10  # Allow multiple contacts per pair
        contact_pair = wp.zeros(max_contacts, dtype=wp.vec2i)
        contact_position = wp.zeros(max_contacts, dtype=wp.vec3)
        contact_normal = wp.zeros(max_contacts, dtype=wp.vec3)
        contact_penetration = wp.zeros(max_contacts, dtype=float)
        contact_tangent = wp.zeros(max_contacts, dtype=wp.vec3)
        contact_count = wp.zeros(1, dtype=int)

        # Launch narrow phase
        self.narrow_phase.launch(
            candidate_pair=candidate_pair,
            candidate_pair_count=candidate_pair_count,
            shape_types=geom_types,
            shape_data=geom_data,
            shape_transform=geom_transform,
            shape_source=geom_source,
            shape_sdf_data=shape_sdf_data,
            shape_contact_margin=shape_contact_margin,
            shape_collision_radius=geom_collision_radius,
            shape_flags=shape_flags,
            shape_collision_aabb_lower=shape_collision_aabb_lower,
            shape_collision_aabb_upper=shape_collision_aabb_upper,
            shape_voxel_resolution=shape_voxel_resolution,
            contact_pair=contact_pair,
            contact_position=contact_position,
            contact_normal=contact_normal,
            contact_penetration=contact_penetration,
            contact_count=contact_count,
            contact_tangent=contact_tangent,
        )

        wp.synchronize()

        # Return numpy arrays
        count = contact_count.numpy()[0]
        return (
            count,
            contact_pair.numpy()[:count],
            contact_position.numpy()[:count],
            contact_normal.numpy()[:count],
            contact_penetration.numpy()[:count],
            contact_tangent.numpy()[:count],
        )

    def test_sphere_sphere_separated(self):
        """Test sphere-sphere collision when separated."""
        # Two spheres separated by distance 1.5
        geom_list = [
            {
                "type": GeoType.SPHERE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
            {
                "type": GeoType.SPHERE,
                "transform": ([3.5, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
        ]

        count, _pairs, _positions, normals, penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        # Separated spheres should produce no contacts (or contacts with positive separation)
        if count > 0:
            # If contact is generated, penetration should be positive (separation)
            # Distance between centers = 3.5, sum of radii = 2.0, expected separation = 1.5
            self.assertGreater(penetrations[0], 0.0, "Separated spheres should have positive penetration (separation)")
            self.assertAlmostEqual(
                penetrations[0], 1.5, places=1, msg=f"Expected separation ~1.5, got {penetrations[0]}"
            )

            # Normal should be unit length
            normal_length = np.linalg.norm(normals[0])
            self.assertAlmostEqual(normal_length, 1.0, places=5, msg="Normal should be unit length")

    def test_sphere_sphere_touching(self):
        """Test sphere-sphere collision with small overlap."""
        # Two unit spheres with small penetration at x=1.998
        # Distance = 1.998, sum of radii = 2.0, penetration = -0.002
        geom_list = [
            {
                "type": GeoType.SPHERE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
            {
                "type": GeoType.SPHERE,
                "transform": ([1.998, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
        ]

        count, pairs, positions, normals, penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        # Should generate contact with small overlap
        self.assertGreater(count, 0, "Spheres with small overlap should generate contact")
        self.assertLess(penetrations[0], 0.0, "Should have negative penetration (overlap)")
        self.assertAlmostEqual(
            penetrations[0], -0.002, delta=0.001, msg=f"Expected penetration ~-0.002, got {penetrations[0]}"
        )

        # Normal should be unit length
        normal_length = np.linalg.norm(normals[0])
        self.assertAlmostEqual(normal_length, 1.0, places=5, msg="Normal should be unit length")

        # Specifically check it's along +X
        self.assertAlmostEqual(normals[0][0], 1.0, places=2, msg="Normal should point along +X")
        self.assertAlmostEqual(normals[0][1], 0.0, places=2, msg="Normal Y should be 0")
        self.assertAlmostEqual(normals[0][2], 0.0, places=2, msg="Normal Z should be 0")

        # Verify surface reconstruction
        if penetrations[0] < 0:
            # Get actual pair indices from narrow phase result
            pair = pairs[0]
            shape_a_idx = pair[0]
            shape_b_idx = pair[1]

            pos_a = np.array([0.0, 0.0, 0.0]) if shape_a_idx == 0 else np.array([1.998, 0.0, 0.0])
            pos_b = np.array([1.998, 0.0, 0.0]) if shape_b_idx == 1 else np.array([0.0, 0.0, 0.0])
            radius_a = 1.0
            radius_b = 1.0

            self.assertTrue(
                check_contact_position_midpoint_spheres(
                    positions[0], normals[0], penetrations[0], pos_a, radius_a, pos_b, radius_b
                ),
                msg="Contact position should be at midpoint between sphere surfaces",
            )

    def test_sphere_sphere_penetrating(self):
        """Test sphere-sphere collision with penetration."""
        test_cases = [
            # (separation, expected_penetration)
            (1.8, -0.2),  # Small penetration
            (1.5, -0.5),  # Medium penetration
            (1.2, -0.8),  # Large penetration
        ]

        for separation, expected_penetration in test_cases:
            with self.subTest(separation=separation):
                pos_a = np.array([0.0, 0.0, 0.0])
                pos_b = np.array([separation, 0.0, 0.0])
                radius_a = 1.0
                radius_b = 1.0

                geom_list = [
                    {
                        "type": GeoType.SPHERE,
                        "transform": (pos_a.tolist(), [0.0, 0.0, 0.0, 1.0]),
                        "data": ([radius_a, radius_a, radius_a], 0.0),
                    },
                    {
                        "type": GeoType.SPHERE,
                        "transform": (pos_b.tolist(), [0.0, 0.0, 0.0, 1.0]),
                        "data": ([radius_b, radius_b, radius_b], 0.0),
                    },
                ]

                count, _pairs, positions, normals, penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

                self.assertGreater(count, 0, "Penetrating spheres should generate contact")
                self.assertAlmostEqual(
                    penetrations[0],
                    expected_penetration,
                    places=2,
                    msg=f"Expected penetration {expected_penetration}, got {penetrations[0]}",
                )

                # Normal should be unit length
                normal_length = np.linalg.norm(normals[0])
                self.assertAlmostEqual(normal_length, 1.0, places=2, msg="Normal should be unit length")

                # Normal should point from sphere 0 toward sphere 1
                self.assertTrue(
                    check_normal_direction(pos_a, pos_b, normals[0]),
                    msg="Normal should point from sphere 0 toward sphere 1",
                )

                # Verify surface reconstruction - contact position should be at midpoint between surfaces
                if penetrations[0] < 0:
                    self.assertTrue(
                        check_contact_position_midpoint_spheres(
                            positions[0], normals[0], penetrations[0], pos_a, radius_a, pos_b, radius_b
                        ),
                        msg="Contact position should be at midpoint between sphere surfaces",
                    )

    def test_sphere_sphere_different_radii(self):
        """Test sphere-sphere collision with different radii."""
        # Sphere at origin with radius 0.5, sphere at x=1.499 with radius 1.0
        # Distance between centers = 1.499
        # Sum of radii = 1.5
        # Expected penetration = 0.001 (very slight penetration)
        geom_list = [
            {
                "type": GeoType.SPHERE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 0.5, 0.5], 0.0),
            },
            {
                "type": GeoType.SPHERE,
                "transform": ([1.499, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
        ]

        count, _pairs, positions, normals, penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        self.assertGreater(count, 0, "Nearly touching spheres should generate contact")
        self.assertAlmostEqual(penetrations[0], 0.0, places=2, msg="Should have near-zero penetration")

        # Normal should be unit length
        normal_length = np.linalg.norm(normals[0])
        self.assertAlmostEqual(normal_length, 1.0, places=5, msg="Normal should be unit length")

        # Verify surface reconstruction if penetrating
        if penetrations[0] < 0:
            pos_a = np.array([0.0, 0.0, 0.0])
            radius_a = 0.5
            pos_b = np.array([1.499, 0.0, 0.0])
            radius_b = 1.0
            self.assertTrue(
                check_contact_position_midpoint_spheres(
                    positions[0], normals[0], penetrations[0], pos_a, radius_a, pos_b, radius_b
                ),
                msg="Contact position should be at midpoint between sphere surfaces",
            )

    def test_sphere_box_penetrating(self):
        """Test sphere-box collision with penetration."""
        # Unit sphere at origin (radius 1.0), box at (1.999, 0, 0) with half-size 1.0
        # Sphere surface at x=1.0, box left surface at x=0.999
        # Expected penetration = 0.001
        sphere_pos = np.array([0.0, 0.0, 0.0])
        sphere_radius = 1.0
        box_pos = np.array([1.999, 0.0, 0.0])
        box_size = np.array([1.0, 1.0, 1.0])

        geom_list = [
            {
                "type": GeoType.SPHERE,
                "transform": (sphere_pos.tolist(), [0.0, 0.0, 0.0, 1.0]),
                "data": ([sphere_radius, sphere_radius, sphere_radius], 0.0),
            },
            {
                "type": GeoType.BOX,
                "transform": (box_pos.tolist(), [0.0, 0.0, 0.0, 1.0]),
                "data": (box_size.tolist(), 0.0),
            },
        ]

        count, _pairs, positions, normals, penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        # Should generate contact
        self.assertGreater(count, 0, "Sphere-box should generate contact")

        # Normal should be unit length
        normal_length = np.linalg.norm(normals[0])
        self.assertAlmostEqual(normal_length, 1.0, places=5, msg="Normal should be unit length")

        # Check penetration depth: sphere surface at x=1.0, box left surface at x=0.999, overlap = 0.001
        self.assertLess(penetrations[0], 0.0, "Sphere-box should be penetrating")
        self.assertAlmostEqual(
            penetrations[0], -0.001, places=2, msg=f"Expected penetration ~-0.001, got {penetrations[0]}"
        )

        # Normal should point approximately from sphere toward box (+X direction)
        self.assertTrue(
            check_normal_direction(sphere_pos, box_pos, normals[0]),
            msg="Normal should point from sphere toward box",
        )
        self.assertGreater(abs(normals[0][0]), 0.9, msg="Normal should be primarily along X axis")

        # Verify surface reconstruction if penetrating
        if penetrations[0] < 0:
            box_rot = np.eye(3)

            def dist_to_sphere(p):
                return abs(np.linalg.norm(p - sphere_pos) - sphere_radius)

            def dist_to_box(p):
                return distance_point_to_box(p, box_pos, box_rot, box_size)

            self.assertTrue(
                check_surface_reconstruction(positions[0], normals[0], penetrations[0], dist_to_sphere, dist_to_box),
                msg="Contact position should be at midpoint between surfaces",
            )

    def test_sphere_box_corner_collision(self):
        """Test sphere-box collision at box corner."""
        # Sphere approaching box corner
        offset = 1.5  # Distance to corner
        corner_dir = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)  # Unit vector toward corner
        sphere_pos = corner_dir * offset

        geom_list = [
            {
                "type": GeoType.SPHERE,
                "transform": (sphere_pos.tolist(), [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 0.5, 0.5], 0.0),
            },
            {"type": GeoType.BOX, "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]), "data": ([1.0, 1.0, 1.0], 0.0)},
        ]

        count, _pairs, positions, normals, penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        # May or may not have contact depending on exact distance
        if count > 0:
            # Normal should point approximately along corner direction
            normal_length = np.linalg.norm(normals[0])
            self.assertAlmostEqual(normal_length, 1.0, places=2, msg="Normal should be unit length")

            # Verify surface reconstruction if penetrating
            if penetrations[0] < 0:
                sphere_radius = 0.5
                box_pos = np.array([0.0, 0.0, 0.0])
                box_size = np.array([1.0, 1.0, 1.0])
                box_rot = np.eye(3)

                def dist_to_sphere(p):
                    return abs(np.linalg.norm(p - sphere_pos) - sphere_radius)

                def dist_to_box(p):
                    return distance_point_to_box(p, box_pos, box_rot, box_size)

                self.assertTrue(
                    check_surface_reconstruction(
                        positions[0], normals[0], penetrations[0], dist_to_sphere, dist_to_box
                    ),
                    msg="Contact position should be at midpoint between surfaces",
                )

    def test_box_box_face_collision(self):
        """Test box-box collision with face contact."""
        # Two unit boxes, one at origin, one offset by 1.8 along X
        # Box surfaces at x=1.0 and x=0.8, overlap = 0.2
        box_a_pos = np.array([0.0, 0.0, 0.0])
        box_a_size = np.array([1.0, 1.0, 1.0])
        box_a_rot = np.eye(3)

        box_b_pos = np.array([1.8, 0.0, 0.0])
        box_b_size = np.array([1.0, 1.0, 1.0])
        box_b_rot = np.eye(3)

        geom_list = [
            {
                "type": GeoType.BOX,
                "transform": (box_a_pos.tolist(), [0.0, 0.0, 0.0, 1.0]),
                "data": (box_a_size.tolist(), 0.0),
            },
            {
                "type": GeoType.BOX,
                "transform": (box_b_pos.tolist(), [0.0, 0.0, 0.0, 1.0]),
                "data": (box_b_size.tolist(), 0.0),
            },
        ]

        count, _pairs, positions, normals, penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        self.assertGreater(count, 0, "Penetrating boxes should generate contact(s)")

        # Check that at least one contact has normal along X axis and correct penetration
        has_x_normal = False
        for i in range(count):
            # Normal should be unit length
            normal_length = np.linalg.norm(normals[i])
            self.assertAlmostEqual(normal_length, 1.0, places=5, msg=f"Contact {i} normal should be unit length")

            if abs(normals[i][0]) > 0.9:
                has_x_normal = True

                # Check penetration depth: box A right face at x=1.0, box B left face at x=0.8, overlap = 0.2
                self.assertLess(penetrations[i], 0.0, f"Contact {i} should have negative penetration")
                self.assertAlmostEqual(
                    penetrations[i],
                    -0.2,
                    places=1,
                    msg=f"Contact {i} expected penetration ~-0.2, got {penetrations[i]}",
                )

                # Normal should point from box A toward box B
                self.assertTrue(
                    check_normal_direction(box_a_pos, box_b_pos, normals[i]),
                    msg=f"Contact {i} normal should point from box A toward box B",
                )

                # Verify surface reconstruction for this contact
                if penetrations[i] < 0:

                    def dist_to_box_a(p):
                        return distance_point_to_box(p, box_a_pos, box_a_rot, box_a_size)

                    def dist_to_box_b(p):
                        return distance_point_to_box(p, box_b_pos, box_b_rot, box_b_size)

                    self.assertTrue(
                        check_surface_reconstruction(
                            positions[i], normals[i], penetrations[i], dist_to_box_a, dist_to_box_b
                        ),
                        msg=f"Contact {i} position should be at midpoint between surfaces",
                    )

                break
        self.assertTrue(has_x_normal, "At least one contact should have normal along X axis")

    def test_box_box_edge_collision(self):
        """Test box-box collision with edge contact."""
        # Two boxes, one rotated 45 degrees around Z axis
        # This creates an edge-edge contact scenario
        angle = np.pi / 4.0  # 45 degrees
        quat = [0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)]

        geom_list = [
            {"type": GeoType.BOX, "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]), "data": ([0.5, 0.5, 0.5], 0.0)},
            {"type": GeoType.BOX, "transform": ([1.2, 0.0, 0.0], quat), "data": ([0.5, 0.5, 0.5], 0.0)},
        ]

        count, _pairs, _positions, normals, _penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        # Edge-edge collision should generate contact
        self.assertGreater(count, 0, "Edge-edge collision should generate contact")

        # Normal should be unit length
        normal_length = np.linalg.norm(normals[0])
        self.assertAlmostEqual(normal_length, 1.0, places=5, msg="Normal should be unit length")

    def test_sphere_capsule_cylinder_side(self):
        """Test sphere collision with capsule cylinder side."""
        # Capsule along Z axis, sphere approaching from +Y side
        # Capsule: radius=0.5, half_length=1.0 (extends from z=-1 to z=1)
        # Sphere: radius=0.5, at (0, 1.5, 0)
        # Distance = 0.5 (separation)
        geom_list = [
            {
                "type": GeoType.SPHERE,
                "transform": ([0.0, 1.5, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 0.5, 0.5], 0.0),
            },
            {
                "type": GeoType.CAPSULE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 1.0, 0.0], 0.0),
            },
        ]

        count, _pairs, _positions, normals, penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        # May not generate contact if separated beyond margin
        if count > 0:
            # Normal should be unit length
            normal_length = np.linalg.norm(normals[0])
            self.assertAlmostEqual(normal_length, 1.0, places=5, msg="Normal should be unit length")

            # Normal should point primarily along Y axis
            self.assertGreater(abs(normals[0][1]), 0.9, msg="Normal should be along Y axis for cylinder side collision")

            # Check separation: distance = 1.5 - (0.5 + 0.5) = 0.5
            self.assertGreater(penetrations[0], 0.0, "Separated shapes should have positive penetration")
            self.assertAlmostEqual(
                penetrations[0], 0.5, delta=0.1, msg=f"Expected separation ~0.5, got {penetrations[0]}"
            )

    def test_sphere_capsule_cap(self):
        """Test sphere collision with capsule hemispherical cap."""
        # Capsule along Z axis, sphere approaching from above
        # Capsule: radius=0.5, half_length=1.0
        # Sphere: radius=0.5, at (0, 0, 2.2)
        # Top cap center at z=1.0, combined radii = 1.0, distance = 1.2
        # Expected separation = 0.2
        geom_list = [
            {
                "type": GeoType.SPHERE,
                "transform": ([0.0, 0.0, 2.2], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 0.5, 0.5], 0.0),
            },
            {
                "type": GeoType.CAPSULE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 1.0, 0.0], 0.0),
            },
        ]

        count, _pairs, _positions, normals, penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        if count > 0:
            # Normal should be unit length
            normal_length = np.linalg.norm(normals[0])
            self.assertAlmostEqual(normal_length, 1.0, places=5, msg="Normal should be unit length")

            # Normal should point primarily along Z axis
            self.assertGreater(abs(normals[0][2]), 0.9, msg="Normal should be along Z axis for cap collision")

            # Check separation: distance = 2.2 - 1.0 = 1.2, combined radii = 1.0, separation = 0.2
            self.assertGreater(penetrations[0], 0.0, "Separated shapes should have positive penetration")
            self.assertAlmostEqual(
                penetrations[0], 0.2, delta=0.05, msg=f"Expected separation ~0.2, got {penetrations[0]}"
            )

    def test_capsule_capsule_parallel(self):
        """Test capsule-capsule collision when parallel."""
        # Two capsules parallel along Z axis, offset in Y direction
        geom_list = [
            {
                "type": GeoType.CAPSULE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 1.0, 0.0], 0.0),
            },
            {
                "type": GeoType.CAPSULE,
                "transform": ([0.0, 1.5, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 1.0, 0.0], 0.0),
            },
        ]

        count, _pairs, _positions, _normals, penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        # Capsules with combined radius 1.0 and separation 1.5 should be separated
        if count > 0:
            self.assertGreater(penetrations[0], 0.0, "Separated capsules should have positive penetration")

    def test_capsule_capsule_crossed(self):
        """Test capsule-capsule collision when crossed (perpendicular)."""
        # Two capsules perpendicular: one along Z, one along X
        # Rotate second capsule 90 degrees around Y axis
        # Offset second capsule in Y direction to create moderate penetration
        angle = np.pi / 2.0
        quat = [0.0, np.sin(angle / 2.0), 0.0, np.cos(angle / 2.0)]

        geom_list = [
            {
                "type": GeoType.CAPSULE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 1.0, 0.0], 0.0),
            },
            # Capsule along X-axis at y=0.8 (crosses capsule 1 with moderate penetration)
            # Distance between centerlines = 0.8, combined radii = 1.0, expected penetration = -0.2
            {"type": GeoType.CAPSULE, "transform": ([0.0, 0.8, 0.0], quat), "data": ([0.5, 1.0, 0.0], 0.0)},
        ]

        count, pairs, positions, normals, penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        # Crossed capsules with radius 0.5 each should be penetrating
        self.assertGreater(count, 0, "Crossed capsules should generate contact")

        # Normal should be unit length
        normal_length = np.linalg.norm(normals[0])
        self.assertAlmostEqual(normal_length, 1.0, places=5, msg="Normal should be unit length")

        # Check penetration depth: distance between centerlines = 0.8, combined radii = 1.0
        # Expected penetration = 0.8 - 1.0 = -0.2
        self.assertLess(penetrations[0], 0.0, "Crossed capsules should have negative penetration")
        self.assertAlmostEqual(
            penetrations[0], -0.2, places=1, msg=f"Expected penetration ~-0.2, got {penetrations[0]}"
        )

        # Verify surface reconstruction
        if penetrations[0] < 0:
            # Get actual pair indices from narrow phase result
            pair = pairs[0]
            shape_a_idx = pair[0]
            shape_b_idx = pair[1]

            # Capsule 0: along Z at (0,0,0), Capsule 1: along X at (0,0.8,0)
            if shape_a_idx == 0:
                capsule_a_pos = np.array([0.0, 0.0, 0.0])
                capsule_a_axis = np.array([0.0, 0.0, 1.0])
            else:
                capsule_a_pos = np.array([0.0, 0.8, 0.0])
                capsule_a_axis = np.array([1.0, 0.0, 0.0])

            if shape_b_idx == 1:
                capsule_b_pos = np.array([0.0, 0.8, 0.0])
                capsule_b_axis = np.array([1.0, 0.0, 0.0])
            else:
                capsule_b_pos = np.array([0.0, 0.0, 0.0])
                capsule_b_axis = np.array([0.0, 0.0, 1.0])

            capsule_radius = 0.5
            capsule_half_length = 1.0

            def dist_to_capsule_a(p):
                return distance_point_to_capsule(p, capsule_a_pos, capsule_a_axis, capsule_radius, capsule_half_length)

            def dist_to_capsule_b(p):
                return distance_point_to_capsule(p, capsule_b_pos, capsule_b_axis, capsule_radius, capsule_half_length)

            self.assertTrue(
                check_surface_reconstruction(
                    positions[0], normals[0], penetrations[0], dist_to_capsule_a, dist_to_capsule_b
                ),
                msg="Contact position should be at midpoint between capsule surfaces",
            )

    def test_plane_sphere_above(self):
        """Test plane-sphere collision when sphere is above plane."""
        # Infinite plane at z=0, normal pointing up (+Z)
        # Sphere radius 1.0 at z=2.0 (center)
        # Distance from center to plane = 2.0, minus radius = 1.0 separation
        geom_list = [
            {
                "type": GeoType.PLANE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.0, 0.0, 0.0], 0.0),
            },
            {
                "type": GeoType.SPHERE,
                "transform": ([0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
        ]

        count, _pairs, _positions, _normals, penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        # Separated - may not generate contact
        if count > 0:
            self.assertGreater(penetrations[0], 0.0, "Sphere above plane should have positive penetration")

    def test_plane_sphere_touching(self):
        """Test plane-sphere collision with small overlap."""
        # Infinite plane at z=0, sphere radius 1.0 at z=0.999 (small penetration)
        # Sphere bottom at z=-0.001, penetration = -0.001
        geom_list = [
            {
                "type": GeoType.PLANE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.0, 0.0, 0.0], 0.0),
            },
            {
                "type": GeoType.SPHERE,
                "transform": ([0.0, 0.0, 0.999], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
        ]

        count, pairs, positions, normals, penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        self.assertGreater(count, 0, "Sphere-plane with small overlap should generate contact")
        self.assertLess(penetrations[0], 0.0, "Should have negative penetration (overlap)")
        self.assertAlmostEqual(
            penetrations[0], -0.001, delta=0.001, msg=f"Expected penetration ~-0.001, got {penetrations[0]}"
        )

        # Normal should be unit length
        normal_length = np.linalg.norm(normals[0])
        self.assertAlmostEqual(normal_length, 1.0, places=5, msg="Normal should be unit length")

        # Verify surface reconstruction
        if penetrations[0] < 0:
            # Get actual pair indices from narrow phase result
            pair = pairs[0]
            shape_a_idx = pair[0]

            plane_pos = np.array([0.0, 0.0, 0.0])
            plane_normal = np.array([0.0, 0.0, 1.0])
            sphere_pos = np.array([0.0, 0.0, 0.999])
            sphere_radius = 1.0

            # Determine which is plane and which is sphere based on pair indices
            if shape_a_idx == 0:
                # Shape A is plane, Shape B is sphere
                def dist_to_a(p):
                    return abs(distance_point_to_plane(p, plane_pos, plane_normal))

                def dist_to_b(p):
                    return abs(np.linalg.norm(p - sphere_pos) - sphere_radius)
            else:
                # Shape A is sphere, Shape B is plane
                def dist_to_a(p):
                    return abs(np.linalg.norm(p - sphere_pos) - sphere_radius)

                def dist_to_b(p):
                    return abs(distance_point_to_plane(p, plane_pos, plane_normal))

            self.assertTrue(
                check_surface_reconstruction(positions[0], normals[0], penetrations[0], dist_to_a, dist_to_b),
                msg="Contact position should be at midpoint between surfaces",
            )

    def test_plane_sphere_penetrating(self):
        """Test plane-sphere collision when sphere penetrates plane."""
        # Infinite plane at z=0, sphere radius 1.0 at z=0.5
        # Penetration depth = radius - distance = 1.0 - 0.5 = 0.5
        plane_pos = np.array([0.0, 0.0, 0.0])
        sphere_pos = np.array([0.0, 0.0, 0.5])
        sphere_radius = 1.0

        geom_list = [
            {
                "type": GeoType.PLANE,
                "transform": (plane_pos.tolist(), [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.0, 0.0, 0.0], 0.0),
            },
            {
                "type": GeoType.SPHERE,
                "transform": (sphere_pos.tolist(), [0.0, 0.0, 0.0, 1.0]),
                "data": ([sphere_radius, sphere_radius, sphere_radius], 0.0),
            },
        ]

        count, _pairs, positions, normals, penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        self.assertGreater(count, 0, "Penetrating sphere-plane should generate contact")
        self.assertLess(penetrations[0], 0.0, "Penetration should be negative")

        # Normal should be unit length
        normal_length = np.linalg.norm(normals[0])
        self.assertAlmostEqual(normal_length, 1.0, places=5, msg="Normal should be unit length")

        # Normal should point in plane normal direction (+Z)
        self.assertGreater(normals[0][2], 0.9, msg="Normal should point in +Z direction")

        # Verify surface reconstruction
        plane_normal = np.array([0.0, 0.0, 1.0])

        def dist_to_plane(p):
            return abs(distance_point_to_plane(p, plane_pos, plane_normal))

        def dist_to_sphere(p):
            return abs(np.linalg.norm(p - sphere_pos) - sphere_radius)

        self.assertTrue(
            check_surface_reconstruction(positions[0], normals[0], penetrations[0], dist_to_plane, dist_to_sphere),
            msg="Contact position should be at midpoint between surfaces",
        )

    def test_plane_box_resting(self):
        """Test plane-box collision when box is resting on plane."""
        # Infinite plane at z=0, box with size 1.0 at z=0.999 (very slightly penetrating)
        # Box bottom face at z=-0.001, top at z=1.999, so penetration depth ~0.001
        plane_pos = np.array([0.0, 0.0, 0.0])
        plane_normal = np.array([0.0, 0.0, 1.0])
        box_pos = np.array([0.0, 0.0, 0.999])
        box_size = np.array([1.0, 1.0, 1.0])
        box_rot = np.eye(3)

        geom_list = [
            {
                "type": GeoType.PLANE,
                "transform": (plane_pos.tolist(), [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.0, 0.0, 0.0], 0.0),
            },
            {
                "type": GeoType.BOX,
                "transform": (box_pos.tolist(), [0.0, 0.0, 0.0, 1.0]),
                "data": (box_size.tolist(), 0.0),
            },
        ]

        count, _pairs, positions, normals, penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        # Box resting on plane should generate contact(s)
        self.assertGreater(count, 0, "Box on plane should generate contact")

        # All contacts should have normals pointing up and near-zero penetration
        for i in range(count):
            # Normal should be unit length
            normal_length = np.linalg.norm(normals[i])
            self.assertAlmostEqual(normal_length, 1.0, places=5, msg=f"Contact {i} normal should be unit length")

            self.assertGreater(normals[i][2], 0.5, msg=f"Contact {i} normal should point upward")

            # Check penetration depth: box bottom at z=-0.001, plane at z=0, penetration ~-0.001
            self.assertAlmostEqual(
                penetrations[i], 0.0, places=2, msg=f"Contact {i} expected near-zero penetration, got {penetrations[i]}"
            )

            # Verify surface reconstruction for penetrating contacts
            if penetrations[i] < 0:

                def dist_to_plane(p):
                    return abs(distance_point_to_plane(p, plane_pos, plane_normal))

                def dist_to_box(p):
                    return distance_point_to_box(p, box_pos, box_rot, box_size)

                self.assertTrue(
                    check_surface_reconstruction(positions[i], normals[i], penetrations[i], dist_to_plane, dist_to_box),
                    msg=f"Contact {i} position should be at midpoint between surfaces",
                )

    def test_plane_capsule_resting(self):
        """Test plane-capsule collision with small overlap."""
        # Infinite plane at z=0, capsule with radius 0.5, half_length 1.0
        # Capsule center at z=1.499 so bottom cap has small penetration with plane
        # (centerline from z=0.499 to z=2.499, with radius 0.5, bottom at z=-0.001)
        # Penetration = -0.001
        geom_list = [
            {
                "type": GeoType.PLANE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.0, 0.0, 0.0], 0.0),
            },
            {
                "type": GeoType.CAPSULE,
                "transform": ([0.0, 0.0, 1.499], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 1.0, 0.0], 0.0),
            },
        ]

        count, _pairs, positions, normals, penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        self.assertGreater(count, 0, "Capsule on plane should generate contact")

        # Normal should be unit length
        normal_length = np.linalg.norm(normals[0])
        self.assertAlmostEqual(normal_length, 1.0, places=5, msg="Normal should be unit length")

        # Normal should point up
        self.assertGreater(normals[0][2], 0.9, msg="Normal should point in +Z direction")

        # Check penetration depth: capsule bottom at z=-0.001, plane at z=0, small overlap
        self.assertLess(penetrations[0], 0.0, "Should have negative penetration (overlap)")
        self.assertAlmostEqual(
            penetrations[0], -0.001, delta=0.001, msg=f"Expected penetration ~-0.001, got {penetrations[0]}"
        )

        # Verify surface reconstruction if penetrating
        if penetrations[0] < 0:
            plane_pos = np.array([0.0, 0.0, 0.0])
            plane_normal = np.array([0.0, 0.0, 1.0])
            capsule_pos = np.array([0.0, 0.0, 1.499])
            capsule_axis = np.array([0.0, 0.0, 1.0])
            capsule_radius = 0.5
            capsule_half_length = 1.0

            def dist_to_plane(p):
                return abs(distance_point_to_plane(p, plane_pos, plane_normal))

            def dist_to_capsule(p):
                return distance_point_to_capsule(p, capsule_pos, capsule_axis, capsule_radius, capsule_half_length)

            self.assertTrue(
                check_surface_reconstruction(positions[0], normals[0], penetrations[0], dist_to_plane, dist_to_capsule),
                msg="Contact position should be at midpoint between surfaces",
            )

    def test_multiple_pairs(self):
        """Test narrow phase with multiple collision pairs."""
        # Create 3 spheres in a line, test all pairs
        geom_list = [
            {
                "type": GeoType.SPHERE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
            {
                "type": GeoType.SPHERE,
                "transform": ([1.8, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
            {
                "type": GeoType.SPHERE,
                "transform": ([3.6, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
        ]

        # Test pairs (0,1), (1,2), and (0,2)
        pairs = [(0, 1), (1, 2), (0, 2)]
        count, contact_pairs, _positions, _normals, _penetrations, _tangents = self._run_narrow_phase(geom_list, pairs)

        # Should get contacts for (0,1) and (1,2) which are penetrating
        # Pair (0,2) is separated so may not generate contact
        self.assertGreaterEqual(count, 2, "Should have at least 2 contacts for penetrating pairs")

        # Verify pairs are correct
        pair_set = {tuple(p) for p in contact_pairs}
        self.assertIn((0, 1), pair_set, "Should have contact for pair (0, 1)")
        self.assertIn((1, 2), pair_set, "Should have contact for pair (1, 2)")

    def test_cylinder_sphere(self):
        """Test cylinder-sphere collision."""
        # Cylinder along Z axis, sphere approaching from side
        geom_list = [
            {
                "type": GeoType.CYLINDER,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 1.0, 0.0], 0.0),
            },
            {
                "type": GeoType.SPHERE,
                "transform": ([1.5, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 0.5, 0.5], 0.0),
            },
        ]

        count, _pairs, _positions, _normals, penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        # Cylinder radius 0.5 + sphere radius 0.5 = 1.0, distance = 1.5, so separation = 0.5
        if count > 0:
            # If contact generated, should have positive penetration (separation)
            self.assertGreater(penetrations[0], 0.0, "Separated should have positive penetration")

    def test_no_self_collision(self):
        """Test that narrow phase doesn't generate self-collisions."""
        geom_list = [
            {
                "type": GeoType.SPHERE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
        ]

        # Try to test sphere against itself
        count, _pairs, _positions, _normals, _penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 0)])

        # Should not generate any contacts for self-collision
        self.assertEqual(count, 0, "Self-collision should not generate contacts")

    def test_contact_normal_unit_length(self):
        """Test that all contact normals are unit length."""
        # Create various collision scenarios
        geom_list = [
            {
                "type": GeoType.SPHERE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
            {
                "type": GeoType.SPHERE,
                "transform": ([1.5, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
            {"type": GeoType.BOX, "transform": ([0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]), "data": ([0.5, 0.5, 0.5], 0.0)},
            {
                "type": GeoType.CAPSULE,
                "transform": ([3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 1.0, 0.0], 0.0),
            },
        ]

        pairs = [(0, 1), (0, 2), (1, 3)]
        count, _contact_pairs, _positions, normals, _penetrations, _tangents = self._run_narrow_phase(geom_list, pairs)

        # Check all normals are unit length
        for i in range(count):
            normal_length = np.linalg.norm(normals[i])
            self.assertAlmostEqual(
                normal_length, 1.0, places=2, msg=f"Contact {i} normal should be unit length, got {normal_length}"
            )

    def test_contact_tangent_perpendicular(self):
        """Test that contact tangents are perpendicular to normals."""
        geom_list = [
            {
                "type": GeoType.SPHERE,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
            {
                "type": GeoType.SPHERE,
                "transform": ([1.5, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 1.0, 1.0], 0.0),
            },
        ]

        count, _pairs, _positions, normals, _penetrations, tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        for i in range(count):
            # Tangent should be perpendicular to normal (dot product ~ 0)
            dot_product = np.dot(normals[i], tangents[i])
            self.assertAlmostEqual(
                dot_product,
                0.0,
                places=2,
                msg=f"Contact {i} tangent should be perpendicular to normal, dot product = {dot_product}",
            )

    def test_per_shape_contact_margin(self):
        """
        Test that per-shape contact margins work correctly by testing two spheres
        with different margins approaching a plane.
        """
        # Create geometries: plane + 2 spheres with different margins
        geom_types = wp.array(
            [int(GeoType.PLANE), int(GeoType.SPHERE), int(GeoType.SPHERE)],
            dtype=wp.int32,
        )
        geom_data = wp.array(
            [
                wp.vec4(0.0, 0.0, 1.0, 0.0),  # Plane (infinite)
                wp.vec4(0.2, 0.2, 0.2, 0.0),  # Sphere A radius=0.2
                wp.vec4(0.2, 0.2, 0.2, 0.0),  # Sphere B radius=0.2
            ],
            dtype=wp.vec4,
        )
        geom_source = wp.zeros(3, dtype=wp.uint64)
        shape_sdf_data = wp.zeros(3, dtype=SDFData)  # SDF data (not used in this test)
        geom_collision_radius = wp.array([1e6, 0.2, 0.2], dtype=wp.float32)
        shape_flags = wp.full(3, ShapeFlags.COLLIDE_SHAPES, dtype=wp.int32)  # Collision enabled, no hydroelastic

        # Contact margins: plane=0.01, sphereA=0.02, sphereB=0.06
        shape_contact_margin = wp.array([0.01, 0.02, 0.06], dtype=wp.float32)

        # Dummy AABB arrays (not used for primitive tests)
        shape_collision_aabb_lower = wp.zeros(3, dtype=wp.vec3)
        shape_collision_aabb_upper = wp.ones(3, dtype=wp.vec3)
        shape_voxel_resolution = wp.full(3, wp.vec3i(4, 4, 4), dtype=wp.vec3i)

        # Allocate output arrays
        max_contacts = 10
        contact_pair = wp.zeros(max_contacts, dtype=wp.vec2i)
        contact_position = wp.zeros(max_contacts, dtype=wp.vec3)
        contact_normal = wp.zeros(max_contacts, dtype=wp.vec3)
        contact_penetration = wp.zeros(max_contacts, dtype=float)
        contact_tangent = wp.zeros(max_contacts, dtype=wp.vec3)
        contact_count = wp.zeros(1, dtype=int)

        # Test 1: Sphere A at z=0.25 (outside combined margin 0.03) - no contact
        geom_transform = wp.array(
            [
                wp.transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                wp.transform((0.0, 0.0, 0.25), (0.0, 0.0, 0.0, 1.0)),
                wp.transform((10.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
            ],
            dtype=wp.transform,
        )
        pairs = wp.array([wp.vec2i(0, 1)], dtype=wp.vec2i)
        pair_count = wp.array([1], dtype=wp.int32)

        contact_count.zero_()
        self.narrow_phase.launch(
            candidate_pair=pairs,
            candidate_pair_count=pair_count,
            shape_types=geom_types,
            shape_data=geom_data,
            shape_transform=geom_transform,
            shape_source=geom_source,
            shape_sdf_data=shape_sdf_data,
            shape_contact_margin=shape_contact_margin,
            shape_collision_radius=geom_collision_radius,
            shape_flags=shape_flags,
            shape_collision_aabb_lower=shape_collision_aabb_lower,
            shape_collision_aabb_upper=shape_collision_aabb_upper,
            shape_voxel_resolution=shape_voxel_resolution,
            contact_pair=contact_pair,
            contact_position=contact_position,
            contact_normal=contact_normal,
            contact_penetration=contact_penetration,
            contact_count=contact_count,
            contact_tangent=contact_tangent,
        )
        wp.synchronize()
        self.assertEqual(contact_count.numpy()[0], 0, "Sphere A outside margin should have no contact")

        # Test 2: Sphere A at z=0.15 (inside margin) - contact!
        geom_transform = wp.array(
            [
                wp.transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                wp.transform((0.0, 0.0, 0.15), (0.0, 0.0, 0.0, 1.0)),
                wp.transform((10.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
            ],
            dtype=wp.transform,
        )

        contact_count.zero_()
        self.narrow_phase.launch(
            candidate_pair=pairs,
            candidate_pair_count=pair_count,
            shape_types=geom_types,
            shape_data=geom_data,
            shape_transform=geom_transform,
            shape_source=geom_source,
            shape_sdf_data=shape_sdf_data,
            shape_contact_margin=shape_contact_margin,
            shape_collision_radius=geom_collision_radius,
            shape_flags=shape_flags,
            shape_collision_aabb_lower=shape_collision_aabb_lower,
            shape_collision_aabb_upper=shape_collision_aabb_upper,
            shape_voxel_resolution=shape_voxel_resolution,
            contact_pair=contact_pair,
            contact_position=contact_position,
            contact_normal=contact_normal,
            contact_penetration=contact_penetration,
            contact_count=contact_count,
            contact_tangent=contact_tangent,
        )
        wp.synchronize()
        self.assertGreater(contact_count.numpy()[0], 0, "Sphere A inside margin should have contact")

        # Test 3: Sphere B at z=0.23 (inside its larger margin 0.07) - contact!
        geom_transform = wp.array(
            [
                wp.transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                wp.transform((10.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
                wp.transform((0.0, 0.0, 0.23), (0.0, 0.0, 0.0, 1.0)),
            ],
            dtype=wp.transform,
        )
        pairs = wp.array([wp.vec2i(0, 2)], dtype=wp.vec2i)

        contact_count.zero_()
        self.narrow_phase.launch(
            candidate_pair=pairs,
            candidate_pair_count=pair_count,
            shape_types=geom_types,
            shape_data=geom_data,
            shape_transform=geom_transform,
            shape_source=geom_source,
            shape_sdf_data=shape_sdf_data,
            shape_contact_margin=shape_contact_margin,
            shape_collision_radius=geom_collision_radius,
            shape_flags=shape_flags,
            shape_collision_aabb_lower=shape_collision_aabb_lower,
            shape_collision_aabb_upper=shape_collision_aabb_upper,
            shape_voxel_resolution=shape_voxel_resolution,
            contact_pair=contact_pair,
            contact_position=contact_position,
            contact_normal=contact_normal,
            contact_penetration=contact_penetration,
            contact_count=contact_count,
            contact_tangent=contact_tangent,
        )
        wp.synchronize()
        self.assertGreater(contact_count.numpy()[0], 0, "Sphere B with larger margin should have contact")

    # ================================================================================
    # Ellipsoid collision tests
    # ================================================================================

    def test_ellipsoid_ellipsoid_separated(self):
        """Test ellipsoid-ellipsoid collision when separated."""
        # Two ellipsoids separated along X axis
        geom_list = [
            {
                "type": GeoType.ELLIPSOID,
                "transform": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 0.5, 0.3], 0.0),  # semi-axes a=1.0, b=0.5, c=0.3
            },
            {
                "type": GeoType.ELLIPSOID,
                "transform": ([3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 0.5, 0.3], 0.0),
            },
        ]

        count, _pairs, _positions, normals, penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        # Separated ellipsoids should produce no contacts (or contacts with positive separation)
        if count > 0:
            # If contact is generated, penetration should be positive (separation)
            self.assertGreater(
                penetrations[0], 0.0, "Separated ellipsoids should have positive penetration (separation)"
            )

            # Normal should be unit length
            normal_length = np.linalg.norm(normals[0])
            self.assertAlmostEqual(normal_length, 1.0, places=5, msg="Normal should be unit length")

    def test_ellipsoid_ellipsoid_penetrating(self):
        """Test ellipsoid-ellipsoid collision with penetration."""
        # Two ellipsoids with overlap along X axis
        # Ellipsoid A centered at origin with a=1.0 extends to x=1.0
        # Ellipsoid B centered at x=1.8 with a=1.0 extends to x=0.8
        # Overlap region from x=0.8 to x=1.0 = 0.2 overlap
        pos_a = np.array([0.0, 0.0, 0.0])
        pos_b = np.array([1.8, 0.0, 0.0])

        geom_list = [
            {
                "type": GeoType.ELLIPSOID,
                "transform": (pos_a.tolist(), [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 0.5, 0.3], 0.0),
            },
            {
                "type": GeoType.ELLIPSOID,
                "transform": (pos_b.tolist(), [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 0.5, 0.3], 0.0),
            },
        ]

        count, _pairs, _positions, normals, penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        self.assertGreater(count, 0, "Penetrating ellipsoids should generate contact")
        self.assertLess(penetrations[0], 0.0, "Should have negative penetration (overlap)")

        # Normal should be unit length
        normal_length = np.linalg.norm(normals[0])
        self.assertAlmostEqual(normal_length, 1.0, places=5, msg="Normal should be unit length")

        # Normal should point from ellipsoid 0 toward ellipsoid 1 (approximately +X)
        self.assertTrue(
            check_normal_direction(pos_a, pos_b, normals[0]),
            msg="Normal should point from ellipsoid 0 toward ellipsoid 1",
        )

    def test_ellipsoid_sphere_penetrating(self):
        """Test ellipsoid-sphere collision with penetration."""
        # Ellipsoid at origin, sphere approaching from +X
        # Note: Narrow phase may swap shapes to ensure consistent ordering (lower type first)
        # SPHERE=2 < ELLIPSOID=4, so sphere becomes shape A
        ellipsoid_pos = np.array([0.0, 0.0, 0.0])
        sphere_pos = np.array([1.4, 0.0, 0.0])
        sphere_radius = 0.5

        geom_list = [
            {
                "type": GeoType.ELLIPSOID,
                "transform": (ellipsoid_pos.tolist(), [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 0.5, 0.3], 0.0),
            },
            {
                "type": GeoType.SPHERE,
                "transform": (sphere_pos.tolist(), [0.0, 0.0, 0.0, 1.0]),
                "data": ([sphere_radius, sphere_radius, sphere_radius], 0.0),
            },
        ]

        count, pairs, _positions, normals, _penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        self.assertGreater(count, 0, "Ellipsoid-sphere should generate contact")

        # Normal should be unit length
        normal_length = np.linalg.norm(normals[0])
        self.assertAlmostEqual(normal_length, 1.0, places=5, msg="Normal should be unit length")

        # Get actual pair to determine shape order (narrow phase may swap)
        pair = pairs[0]
        shape_a_idx = pair[0]
        if shape_a_idx == 0:
            # Ellipsoid is shape A, normal points toward sphere (+X)
            pos_a, pos_b = ellipsoid_pos, sphere_pos
        else:
            # Sphere is shape A, normal points toward ellipsoid (-X)
            pos_a, pos_b = sphere_pos, ellipsoid_pos

        # Normal should point from shape A toward shape B
        self.assertTrue(
            check_normal_direction(pos_a, pos_b, normals[0]),
            msg="Normal should point from shape A toward shape B",
        )

    def test_ellipsoid_box_penetrating(self):
        """Test ellipsoid-box collision with penetration."""
        # Ellipsoid at origin, box approaching from +X
        ellipsoid_pos = np.array([0.0, 0.0, 0.0])
        box_pos = np.array([1.4, 0.0, 0.0])
        box_size = np.array([0.5, 0.5, 0.5])

        geom_list = [
            {
                "type": GeoType.ELLIPSOID,
                "transform": (ellipsoid_pos.tolist(), [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 0.5, 0.3], 0.0),
            },
            {
                "type": GeoType.BOX,
                "transform": (box_pos.tolist(), [0.0, 0.0, 0.0, 1.0]),
                "data": (box_size.tolist(), 0.0),
            },
        ]

        count, _pairs, _positions, normals, _penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        self.assertGreater(count, 0, "Ellipsoid-box should generate contact")

        # Normal should be unit length
        normal_length = np.linalg.norm(normals[0])
        self.assertAlmostEqual(normal_length, 1.0, places=5, msg="Normal should be unit length")

        # Normal should point from ellipsoid toward box
        self.assertTrue(
            check_normal_direction(ellipsoid_pos, box_pos, normals[0]),
            msg="Normal should point from ellipsoid toward box",
        )

    def test_ellipsoid_plane_penetrating(self):
        """Test ellipsoid-plane collision with penetration."""
        # Infinite plane at z=0, ellipsoid resting on plane with small penetration
        # Ellipsoid with c=0.3 semi-axis along Z, positioned so bottom just penetrates
        plane_pos = np.array([0.0, 0.0, 0.0])
        ellipsoid_pos = np.array([0.0, 0.0, 0.29])  # Bottom at z=-0.01 (small penetration)

        geom_list = [
            {
                "type": GeoType.PLANE,
                "transform": (plane_pos.tolist(), [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.0, 0.0, 0.0], 0.0),  # infinite plane
            },
            {
                "type": GeoType.ELLIPSOID,
                "transform": (ellipsoid_pos.tolist(), [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 0.5, 0.3], 0.0),
            },
        ]

        count, _pairs, _positions, normals, penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        self.assertGreater(count, 0, "Ellipsoid-plane should generate contact")
        self.assertLess(penetrations[0], 0.0, "Should have negative penetration (overlap)")

        # Normal should be unit length
        normal_length = np.linalg.norm(normals[0])
        self.assertAlmostEqual(normal_length, 1.0, places=5, msg="Normal should be unit length")

        # Normal should point in plane normal direction (+Z)
        self.assertGreater(abs(normals[0][2]), 0.9, msg="Normal should be along Z axis")

    def test_ellipsoid_capsule_penetrating(self):
        """Test ellipsoid-capsule collision with penetration."""
        # Ellipsoid at origin, capsule approaching from +Y
        ellipsoid_pos = np.array([0.0, 0.0, 0.0])
        capsule_pos = np.array([0.0, 0.9, 0.0])

        geom_list = [
            {
                "type": GeoType.ELLIPSOID,
                "transform": (ellipsoid_pos.tolist(), [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 0.5, 0.3], 0.0),
            },
            {
                "type": GeoType.CAPSULE,
                "transform": (capsule_pos.tolist(), [0.0, 0.0, 0.0, 1.0]),
                "data": ([0.5, 1.0, 0.0], 0.0),  # radius=0.5, half_length=1.0
            },
        ]

        count, _pairs, _positions, normals, _penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        self.assertGreater(count, 0, "Ellipsoid-capsule should generate contact")

        # Normal should be unit length
        normal_length = np.linalg.norm(normals[0])
        self.assertAlmostEqual(normal_length, 1.0, places=5, msg="Normal should be unit length")

    def test_ellipsoid_different_orientations(self):
        """Test ellipsoid collision with rotated ellipsoids."""
        # Two ellipsoids, one rotated 90 degrees around Z axis
        angle = np.pi / 2.0
        quat = [0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)]

        pos_a = np.array([0.0, 0.0, 0.0])
        pos_b = np.array([1.3, 0.0, 0.0])

        geom_list = [
            {
                "type": GeoType.ELLIPSOID,
                "transform": (pos_a.tolist(), [0.0, 0.0, 0.0, 1.0]),
                "data": ([1.0, 0.3, 0.3], 0.0),  # elongated along X
            },
            {
                "type": GeoType.ELLIPSOID,
                "transform": (pos_b.tolist(), quat),  # rotated, now elongated along Y
                "data": ([1.0, 0.3, 0.3], 0.0),
            },
        ]

        count, _pairs, _positions, normals, _penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        # Ellipsoid A extends to x=1.0, ellipsoid B after rotation has semi-axis 0.3 along X
        # Starting at x=1.3, B extends from x=1.0 to x=1.6, so they just touch
        if count > 0:
            # Normal should be unit length
            normal_length = np.linalg.norm(normals[0])
            self.assertAlmostEqual(normal_length, 1.0, places=5, msg="Normal should be unit length")

    def test_ellipsoid_sphere_equivalent(self):
        """Test that an ellipsoid with equal semi-axes behaves like a sphere."""
        # Two ellipsoids with a=b=c should behave like spheres
        # Sphere 1 at origin with radius 1.0, sphere 2 at x=1.8 with radius 1.0
        # Expected: same behavior as sphere-sphere with penetration ~-0.2
        pos_a = np.array([0.0, 0.0, 0.0])
        pos_b = np.array([1.8, 0.0, 0.0])
        radius = 1.0

        geom_list = [
            {
                "type": GeoType.ELLIPSOID,
                "transform": (pos_a.tolist(), [0.0, 0.0, 0.0, 1.0]),
                "data": ([radius, radius, radius], 0.0),  # sphere-like ellipsoid
            },
            {
                "type": GeoType.ELLIPSOID,
                "transform": (pos_b.tolist(), [0.0, 0.0, 0.0, 1.0]),
                "data": ([radius, radius, radius], 0.0),
            },
        ]

        count, _pairs, _positions, normals, penetrations, _tangents = self._run_narrow_phase(geom_list, [(0, 1)])

        self.assertGreater(count, 0, "Sphere-like ellipsoids should generate contact")
        self.assertLess(penetrations[0], 0.0, "Should have negative penetration")

        # Expected penetration for sphere-sphere: distance - 2*radius = 1.8 - 2.0 = -0.2
        self.assertAlmostEqual(
            penetrations[0], -0.2, places=1, msg=f"Expected penetration ~-0.2, got {penetrations[0]}"
        )

        # Normal should be unit length
        normal_length = np.linalg.norm(normals[0])
        self.assertAlmostEqual(normal_length, 1.0, places=5, msg="Normal should be unit length")

        # Normal should point along +X
        self.assertAlmostEqual(normals[0][0], 1.0, places=1, msg="Normal should point along +X")


class TestBufferOverflowWarnings(unittest.TestCase):
    """Test that buffer overflow produces warnings and does not crash."""

    @staticmethod
    def _make_ellipsoids(n, spacing=1.5):
        """Create n overlapping ellipsoids along the X axis (routes to GJK)."""
        geom_list = []
        for i in range(n):
            geom_list.append(
                {
                    "type": GeoType.ELLIPSOID,
                    "transform": ([i * spacing, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                    "data": ([1.0, 0.8, 0.6], 0.0),
                }
            )
        return geom_list

    @staticmethod
    def _make_spheres(n, spacing=1.5):
        """Create n overlapping unit spheres along the X axis."""
        geom_list = []
        for i in range(n):
            geom_list.append(
                {
                    "type": GeoType.SPHERE,
                    "transform": ([i * spacing, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                    "data": ([1.0, 1.0, 1.0], 0.0),
                }
            )
        return geom_list

    def _create_geometry_arrays(self, geom_list):
        """Create geometry arrays from geometry descriptions."""
        n = len(geom_list)
        geom_types = np.zeros(n, dtype=np.int32)
        geom_data = np.zeros(n, dtype=wp.vec4)
        geom_transforms = []
        geom_source = np.zeros(n, dtype=np.uint64)
        shape_contact_margin = np.zeros(n, dtype=np.float32)
        geom_collision_radius = np.zeros(n, dtype=np.float32)

        for i, geom in enumerate(geom_list):
            geom_types[i] = int(geom["type"])
            data = geom.get("data", ([1.0, 1.0, 1.0], 0.0))
            if isinstance(data, tuple):
                scale, thickness = data
            else:
                scale = data
                thickness = 0.0
            geom_data[i] = wp.vec4(scale[0], scale[1], scale[2], thickness)
            pos, quat = geom.get("transform", ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]))
            geom_transforms.append(
                wp.transform(wp.vec3(pos[0], pos[1], pos[2]), wp.quat(quat[0], quat[1], quat[2], quat[3]))
            )
            geom_source[i] = geom.get("source", 0)
            shape_contact_margin[i] = geom.get("cutoff", 0.0)
            geom_collision_radius[i] = max(scale[0], scale[1], scale[2])

        return (
            wp.array(geom_types, dtype=wp.int32),
            wp.array(geom_data, dtype=wp.vec4),
            wp.array(geom_transforms, dtype=wp.transform),
            wp.array(geom_source, dtype=wp.uint64),
            wp.array(shape_contact_margin, dtype=wp.float32),
            wp.array(geom_collision_radius, dtype=wp.float32),
            wp.zeros(n, dtype=SDFData),
            wp.full(n, ShapeFlags.COLLIDE_SHAPES, dtype=wp.int32),
            wp.zeros(n, dtype=wp.vec3),
            wp.ones(n, dtype=wp.vec3),
            wp.full(n, wp.vec3i(4, 4, 4), dtype=wp.vec3i),
        )

    def test_gjk_buffer_overflow(self):
        """Test that GJK buffer overflow produces a warning and no crash."""
        # 4 overlapping ellipsoids -> 3 adjacent pairs routed to GJK, but buffer has capacity 1
        geom_list = self._make_ellipsoids(4)
        all_pairs = [(i, j) for i in range(4) for j in range(i + 1, 4) if abs(i - j) == 1]

        narrow_phase = NarrowPhase(
            max_candidate_pairs=1,
            has_meshes=False,
            device=None,
        )

        arrays = self._create_geometry_arrays(geom_list)
        candidate_pair = wp.array(np.array(all_pairs, dtype=np.int32).reshape(-1, 2), dtype=wp.vec2i)
        num_candidate_pair = wp.array([len(all_pairs)], dtype=wp.int32)

        contact_count = wp.zeros(1, dtype=int)
        max_contacts = 20
        contact_pair = wp.zeros(max_contacts, dtype=wp.vec2i)
        contact_position = wp.zeros(max_contacts, dtype=wp.vec3)
        contact_normal = wp.zeros(max_contacts, dtype=wp.vec3)
        contact_penetration = wp.zeros(max_contacts, dtype=float)

        capture = StdOutCapture()
        capture.begin()
        narrow_phase.launch(
            candidate_pair=candidate_pair,
            candidate_pair_count=num_candidate_pair,
            shape_types=arrays[0],
            shape_data=arrays[1],
            shape_transform=arrays[2],
            shape_source=arrays[3],
            shape_sdf_data=arrays[6],
            shape_contact_margin=arrays[4],
            shape_collision_radius=arrays[5],
            shape_flags=arrays[7],
            shape_local_aabb_lower=arrays[8],
            shape_local_aabb_upper=arrays[9],
            shape_voxel_resolution=arrays[10],
            contact_pair=contact_pair,
            contact_position=contact_position,
            contact_normal=contact_normal,
            contact_penetration=contact_penetration,
            contact_count=contact_count,
        )
        wp.synchronize()
        output = capture.end()

        # Verify overflow was detected (counter exceeds buffer capacity)
        gjk_count = narrow_phase.gjk_candidate_pairs_count.numpy()[0]
        gjk_capacity = narrow_phase.gjk_candidate_pairs.shape[0]
        self.assertGreater(gjk_count, gjk_capacity, "GJK buffer should have overflowed")

        # Verify warning was printed (wp.printf capture is unreliable on CPU/Windows)
        if wp.get_preferred_device().is_cuda:
            self.assertIn("GJK candidate pair buffer overflowed", output)

        # Verify some contacts were still produced (from the pairs that fit)
        count = contact_count.numpy()[0]
        self.assertGreater(count, 0, "Should still produce contacts for pairs that fit in the buffer")

    def test_broad_phase_buffer_overflow(self):
        """Test that broad phase buffer overflow produces a warning and no crash."""
        # 4 overlapping spheres -> 3 adjacent pairs, but broad phase buffer has capacity 1
        geom_list = self._make_spheres(4)
        all_pairs = [(i, j) for i in range(4) for j in range(i + 1, 4) if abs(i - j) == 1]

        narrow_phase = NarrowPhase(
            max_candidate_pairs=1000,
            has_meshes=False,
            device=None,
        )

        arrays = self._create_geometry_arrays(geom_list)
        # Broad phase buffer has capacity 1, but we feed 3 pairs
        candidate_pair = wp.zeros(1, dtype=wp.vec2i)
        candidate_pair_full = wp.array(np.array(all_pairs, dtype=np.int32).reshape(-1, 2), dtype=wp.vec2i)
        # Copy first pair only into the tiny buffer
        wp.copy(candidate_pair, candidate_pair_full, count=1)
        # But set the count to the full number of pairs (simulating broad phase overflow)
        num_candidate_pair = wp.array([len(all_pairs)], dtype=wp.int32)

        contact_count = wp.zeros(1, dtype=int)
        max_contacts = 20
        contact_pair_out = wp.zeros(max_contacts, dtype=wp.vec2i)
        contact_position = wp.zeros(max_contacts, dtype=wp.vec3)
        contact_normal = wp.zeros(max_contacts, dtype=wp.vec3)
        contact_penetration = wp.zeros(max_contacts, dtype=float)

        capture = StdOutCapture()
        capture.begin()
        narrow_phase.launch(
            candidate_pair=candidate_pair,
            candidate_pair_count=num_candidate_pair,
            shape_types=arrays[0],
            shape_data=arrays[1],
            shape_transform=arrays[2],
            shape_source=arrays[3],
            shape_sdf_data=arrays[6],
            shape_contact_margin=arrays[4],
            shape_collision_radius=arrays[5],
            shape_flags=arrays[7],
            shape_local_aabb_lower=arrays[8],
            shape_local_aabb_upper=arrays[9],
            shape_voxel_resolution=arrays[10],
            contact_pair=contact_pair_out,
            contact_position=contact_position,
            contact_normal=contact_normal,
            contact_penetration=contact_penetration,
            contact_count=contact_count,
        )
        wp.synchronize()
        output = capture.end()

        # Verify broad phase overflow warning was printed (wp.printf capture is unreliable on CPU/Windows)
        if wp.get_preferred_device().is_cuda:
            self.assertIn("Broad phase pair buffer overflowed", output)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
