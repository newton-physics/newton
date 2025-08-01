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

import unittest

import numpy as np
import warp as wp

from newton.geometry.narrow_phase import NarrowPhaseContactGeneration
from newton.geometry.types import GEO_BOX, GEO_CAPSULE, GEO_CYLINDER, GEO_PLANE, GEO_SPHERE


def quat_from_euler_degree(e: wp.vec3, i: int, j: int, k: int) -> wp.quat:
    """
    Convert Euler angles to a quaternion.

    :math:`i, j, k` are the indices in :math:`[0, 1, 2]` of the axes in which the Euler angles are provided
    (:math:`i \\neq j, j \\neq k`), e.g. (0, 1, 2) for Euler sequence XYZ.

    Args:
        e (vec3): The Euler angles (in radians)
        i (int): The index of the first axis
        j (int): The index of the second axis
        k (int): The index of the third axis

    Returns:
        quat: The quaternion
    """
    # Half angles
    half_e = (3.14159265358979323846 / 180.0) * e / 2.0

    # Precompute sines and cosines of half angles
    cr = wp.cos(half_e[i])
    sr = wp.sin(half_e[i])
    cp = wp.cos(half_e[j])
    sp = wp.sin(half_e[j])
    cy = wp.cos(half_e[k])
    sy = wp.sin(half_e[k])

    # Components of the quaternion based on the rotation sequence
    return wp.quat(
        (cy * sr * cp - sy * cr * sp),
        (cy * cr * sp + sy * sr * cp),
        (sy * cr * cp - cy * sr * sp),
        (cy * cr * cp + sy * sr * sp),
    )


class TestNarrowPhase(unittest.TestCase):
    def setUp(self):
        self.narrow_phase = NarrowPhaseContactGeneration()

        # Common test parameters
        self.max_contacts = 100
        self.cutoff = 0.1

        # Initialize output arrays
        self.contact_pair = wp.zeros(self.max_contacts, dtype=wp.vec2i)
        self.contact_position = wp.zeros(self.max_contacts, dtype=wp.vec3)
        self.contact_normal = wp.zeros(self.max_contacts, dtype=wp.vec3)
        self.contact_penetration = wp.zeros(self.max_contacts, dtype=float)
        self.contact_tangent = wp.zeros(self.max_contacts, dtype=wp.vec3)
        self.contact_count = wp.zeros(1, dtype=int)

    def _create_geometry_arrays(self, geom_types, geom_data_list, transforms):
        """Helper to create geometry arrays from lists"""
        n_geoms = len(geom_types)

        candidate_pairs = wp.array([(0, 1)], dtype=wp.vec2i)
        num_candidate_pairs = wp.array([1], dtype=int)

        geom_types_array = wp.array(geom_types, dtype=wp.int32)
        geom_data_array = wp.array(geom_data_list, dtype=wp.vec4)
        geom_transforms = wp.array(transforms, dtype=wp.transform)
        geom_source = wp.zeros(n_geoms, dtype=wp.uint64)
        geom_cutoff_array = wp.array([self.cutoff] * n_geoms, dtype=float)

        return (
            candidate_pairs,
            num_candidate_pairs,
            geom_types_array,
            geom_data_array,
            geom_transforms,
            geom_source,
            geom_cutoff_array,
        )

    def _run_narrow_phase_test(self, geom_types, geom_data_list, transforms, expected_contacts=True):
        """Common test runner for narrow phase collision detection"""
        arrays = self._create_geometry_arrays(geom_types, geom_data_list, transforms)

        self.narrow_phase.launch(
            *arrays,
            self.contact_pair,
            self.contact_position,
            self.contact_normal,
            self.contact_penetration,
            self.contact_tangent,
            self.contact_count,
        )

        wp.synchronize()

        contact_count = self.contact_count.numpy()[0]

        if expected_contacts:
            self.assertGreater(contact_count, 0, "Expected contacts but none were generated")

            # Validate contact data
            pairs = self.contact_pair.numpy()[:contact_count]
            # positions = self.contact_position.numpy()[:contact_count]
            normals = self.contact_normal.numpy()[:contact_count]
            # penetrations = self.contact_penetration.numpy()[:contact_count]
            tangents = self.contact_tangent.numpy()[:contact_count]

            # Basic sanity checks
            for i in range(contact_count):
                # Check that contact pairs reference valid geometries
                self.assertIn(pairs[i][0], [0, 1])
                self.assertIn(pairs[i][1], [0, 1])

                # Check that normals are normalized (approximately)
                normal_length = np.linalg.norm(normals[i])
                self.assertAlmostEqual(normal_length, 1.0, places=5)

                # Check that tangents are normalized (approximately) if not zero
                tangent_length = np.linalg.norm(tangents[i])
                if tangent_length > 1e-6:
                    self.assertAlmostEqual(tangent_length, 1.0, places=5)
        else:
            self.assertEqual(contact_count, 0, "Expected no contacts but some were generated")

        return contact_count

    def test_plane_sphere_collision(self):
        """Test Plane-Sphere collision detection"""
        geom_types = [GEO_PLANE, GEO_SPHERE]

        # Plane at origin, sphere slightly penetrating
        plane_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        sphere_transform = wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity())  # Sphere above plane

        geom_data = [
            wp.vec4(0.0, 0.0, 0.0, 0.0),  # Plane data (unused)
            wp.vec4(1.0, 0.0, 0.0, 0.0),  # Sphere radius = 1.0
        ]

        transforms = [plane_transform, sphere_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_plane_box_collision(self):
        """Test Plane-Box collision detection"""
        geom_types = [GEO_PLANE, GEO_BOX]

        plane_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        box_transform = wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity())

        geom_data = [
            wp.vec4(0.0, 0.0, 0.0, 0.0),  # Plane data
            wp.vec4(0.5, 0.5, 0.5, 0.0),  # Box half-extents
        ]

        transforms = [plane_transform, box_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_plane_capsule_collision(self):
        """Test Plane-Capsule collision detection"""
        geom_types = [GEO_PLANE, GEO_CAPSULE]

        plane_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        capsule_transform = wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity())

        geom_data = [
            wp.vec4(0.0, 0.0, 0.0, 0.0),  # Plane data
            wp.vec4(0.5, 1.0, 0.0, 0.0),  # Capsule: radius=0.5, half_length=1.0
        ]

        transforms = [plane_transform, capsule_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_plane_cylinder_collision(self):
        """Test Plane-Cylinder collision detection"""
        geom_types = [GEO_PLANE, GEO_CYLINDER]

        plane_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        cylinder_transform = wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity())

        geom_data = [
            wp.vec4(0.0, 0.0, 0.0, 0.0),  # Plane data
            wp.vec4(0.5, 1.0, 0.0, 0.0),  # Cylinder: radius=0.5, half_height=1.0
        ]

        transforms = [plane_transform, cylinder_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_sphere_sphere_collision(self):
        """Test Sphere-Sphere collision detection"""
        geom_types = [GEO_SPHERE, GEO_SPHERE]

        sphere1_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        sphere2_transform = wp.transform(wp.vec3(1.5, 0.0, 0.0), wp.quat_identity())  # Overlapping spheres

        geom_data = [
            wp.vec4(1.0, 0.0, 0.0, 0.0),  # Sphere 1 radius = 1.0
            wp.vec4(1.0, 0.0, 0.0, 0.0),  # Sphere 2 radius = 1.0
        ]

        transforms = [sphere1_transform, sphere2_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_sphere_box_collision(self):
        """Test Sphere-Box collision detection"""
        geom_types = [GEO_SPHERE, GEO_BOX]

        sphere_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        box_transform = wp.transform(wp.vec3(0.75, 0.0, 0.0), wp.quat_identity())

        geom_data = [
            wp.vec4(1.0, 0.0, 0.0, 0.0),  # Sphere radius = 1.0
            wp.vec4(0.5, 0.5, 0.5, 0.0),  # Box half-extents
        ]

        transforms = [sphere_transform, box_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_sphere_capsule_collision(self):
        """Test Sphere-Capsule collision detection"""
        geom_types = [GEO_SPHERE, GEO_CAPSULE]

        sphere_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        capsule_transform = wp.transform(wp.vec3(1.2, 0.0, 0.0), wp.quat_identity())

        geom_data = [
            wp.vec4(1.0, 0.0, 0.0, 0.0),  # Sphere radius = 1.0
            wp.vec4(0.5, 1.0, 0.0, 0.0),  # Capsule: radius=0.5, half_length=1.0
        ]

        transforms = [sphere_transform, capsule_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_sphere_cylinder_collision(self):
        """Test Sphere-Cylinder collision detection"""
        geom_types = [GEO_SPHERE, GEO_CYLINDER]

        sphere_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        cylinder_transform = wp.transform(wp.vec3(1.2, 0.0, 0.0), wp.quat_identity())

        geom_data = [
            wp.vec4(1.0, 0.0, 0.0, 0.0),  # Sphere radius = 1.0
            wp.vec4(0.5, 1.0, 0.0, 0.0),  # Cylinder: radius=0.5, half_height=1.0
        ]

        transforms = [sphere_transform, cylinder_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_box_box_collision(self):
        """Test Box-Box collision detection"""
        geom_types = [GEO_BOX, GEO_BOX]

        box1_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        box2_transform = wp.transform(wp.vec3(0.75, 0.0, 0.0), wp.quat_identity())  # Overlapping boxes

        geom_data = [
            wp.vec4(0.5, 0.5, 0.5, 0.0),  # Box 1 half-extents
            wp.vec4(0.5, 0.5, 0.5, 0.0),  # Box 2 half-extents
        ]

        transforms = [box1_transform, box2_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_capsule_capsule_collision(self):
        """Test Capsule-Capsule collision detection"""
        geom_types = [GEO_CAPSULE, GEO_CAPSULE]

        capsule1_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        capsule2_transform = wp.transform(wp.vec3(0.9, 0.0, 0.0), wp.quat_identity())

        geom_data = [
            wp.vec4(0.5, 1.0, 0.0, 0.0),  # Capsule 1: radius=0.5, half_length=1.0
            wp.vec4(0.5, 1.0, 0.0, 0.0),  # Capsule 2: radius=0.5, half_length=1.0
        ]

        transforms = [capsule1_transform, capsule2_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_capsule_box_collision(self):
        """Test Capsule-Box collision detection"""
        geom_types = [GEO_CAPSULE, GEO_BOX]

        capsule_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        box_transform = wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity())

        geom_data = [
            wp.vec4(0.5, 1.0, 0.0, 0.0),  # Capsule: radius=0.5, half_length=1.0
            wp.vec4(0.5, 0.5, 0.5, 0.0),  # Box half-extents
        ]

        transforms = [capsule_transform, box_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_box_plane_rotated(self):
        """Test Box-Plane collision with rotated box (adapted from box_plane fixture)"""
        geom_types = [GEO_PLANE, GEO_BOX]

        plane_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        # Box at pos="0 0 0.3" with euler="45 0 0" rotation
        box_quat = quat_from_euler_degree(wp.vec3(45, 0, 0), 0, 1, 2)
        box_transform = wp.transform(wp.vec3(0.0, 0.0, 0.3), box_quat)

        geom_data = [
            wp.vec4(0.0, 0.0, 0.0, 0.0),  # Plane data
            wp.vec4(0.5, 0.5, 0.5, 0.0),  # Box half-extents
        ]

        transforms = [plane_transform, box_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_box_box_vertex_face(self):
        """Test Box-Box vertex-face collision (adapted from box_box_vf fixture)"""
        geom_types = [GEO_BOX, GEO_BOX]

        # First box at origin
        box1_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

        # Second box at pos="0 -0.2 1.2" with euler="44 46 0" rotation
        box2_quat = quat_from_euler_degree(wp.vec3(44, 46, 0), 0, 1, 2)
        box2_transform = wp.transform(wp.vec3(0.0, -0.2, 1.2), box2_quat)

        geom_data = [
            wp.vec4(0.5, 0.5, 0.5, 0.0),  # Box 1 half-extents
            wp.vec4(0.6, 0.4, 0.7, 0.0),  # Box 2 half-extents
        ]

        transforms = [box1_transform, box2_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_box_box_edge_edge(self):
        """Test Box-Box edge-edge collision (adapted from box_box_ee fixture)"""
        geom_types = [GEO_BOX, GEO_BOX]
        self.cutoff = 1

        # First box with euler="0 45 0" rotation
        box1_quat = quat_from_euler_degree(wp.vec3(0, 45, 0), 0, 1, 2)
        box1_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), box1_quat)

        # Second box at pos="0 0 1.6" with euler="44 0 90" rotation
        box2_quat = quat_from_euler_degree(wp.vec3(44, 0, 90), 0, 1, 2)
        box2_transform = wp.transform(wp.vec3(0.0, 0.0, 1.6), box2_quat)

        geom_data = [
            wp.vec4(0.5, 0.5, 0.5, 0.0),  # Box 1 half-extents
            wp.vec4(0.6, 0.4, 0.7, 0.0),  # Box 2 half-extents
        ]

        transforms = [box1_transform, box2_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_plane_sphere_rotated(self):
        """Test Plane-Sphere collision with rotated sphere (adapted from plane_sphere fixture)"""
        geom_types = [GEO_PLANE, GEO_SPHERE]

        plane_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        # Sphere at pos="0 0 0.2" with euler="45 0 0" rotation (rotation doesn't affect sphere but keeping for consistency)
        sphere_quat = quat_from_euler_degree(wp.vec3(45, 0, 0), 0, 1, 2)
        sphere_transform = wp.transform(wp.vec3(0.0, 0.0, 0.2), sphere_quat)

        geom_data = [
            wp.vec4(0.0, 0.0, 0.0, 0.0),  # Plane data
            wp.vec4(0.5, 0.0, 0.0, 0.0),  # Sphere radius = 0.5
        ]

        transforms = [plane_transform, sphere_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_plane_capsule_angled(self):
        """Test Plane-Capsule collision with angled capsule (adapted from plane_capsule fixture)"""
        geom_types = [GEO_PLANE, GEO_CAPSULE]

        plane_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        # Capsule at pos="0 0 0.0" with euler="30 30 0" rotation
        capsule_quat = quat_from_euler_degree(wp.vec3(30, 30, 0), 0, 1, 2)
        capsule_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), capsule_quat)

        geom_data = [
            wp.vec4(0.0, 0.0, 0.0, 0.0),  # Plane data
            wp.vec4(0.05, 0.05, 0.0, 0.0),  # Capsule: radius=0.05, half_length=0.05
        ]

        transforms = [plane_transform, capsule_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_capsule_capsule_parallel(self):
        """Test Capsule-Capsule collision with parallel capsules (adapted from capsule_capsule fixture)"""
        geom_types = [GEO_CAPSULE, GEO_CAPSULE]

        # Converting fromto="0.62235904 0.58846647 0.651046 1.5330081 0.33564585 0.977849"
        # to position and orientation - using center position and rotation to align with direction
        center1 = wp.vec3(1.0778, 0.4620, 0.8144)  # midpoint of fromto
        capsule1_transform = wp.transform(center1, wp.quat_identity())

        # Converting fromto="0.5505271 0.60345304 0.476661 1.3900293 0.30709633 0.932082"
        center2 = wp.vec3(0.9703, 0.4553, 0.7044)  # midpoint of fromto
        capsule2_transform = wp.transform(center2, wp.quat_identity())

        geom_data = [
            wp.vec4(0.05, 0.5, 0.0, 0.0),  # Capsule 1: radius=0.05, approx half_length
            wp.vec4(0.05, 0.5, 0.0, 0.0),  # Capsule 2: radius=0.05, approx half_length
        ]

        transforms = [capsule1_transform, capsule2_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_sphere_sphere_close(self):
        """Test Sphere-Sphere collision with close spheres (adapted from sphere_sphere fixture)"""
        geom_types = [GEO_SPHERE, GEO_SPHERE]

        sphere1_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        sphere2_transform = wp.transform(wp.vec3(0.0, 0.3, 0.0), wp.quat_identity())

        geom_data = [
            wp.vec4(0.2, 0.0, 0.0, 0.0),  # Sphere 1 radius = 0.2
            wp.vec4(0.11, 0.0, 0.0, 0.0),  # Sphere 2 radius = 0.11
        ]

        transforms = [sphere1_transform, sphere2_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_sphere_capsule_side(self):
        """Test Sphere-Capsule collision from side (adapted from sphere_capsule fixture)"""
        geom_types = [GEO_SPHERE, GEO_CAPSULE]

        sphere_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        # Capsule fromto="0.3 0 0 0.7 0 0" - horizontal capsule
        # This defines a capsule along the x-axis, centered at x=0.5, with half-length 0.2
        capsule_quat = quat_from_euler_degree(wp.vec3(0.0, 90.0, 0.0), 0, 1, 2)  # Rotate to align with x-axis
        capsule_transform = wp.transform(wp.vec3(0.5, 0.0, 0.0), capsule_quat)

        geom_data = [
            wp.vec4(0.25, 0.0, 0.0, 0.0),  # Sphere radius = 0.25
            wp.vec4(0.1, 0.2, 0.0, 0.0),  # Capsule: radius=0.1, half_length=0.2
        ]

        transforms = [sphere_transform, capsule_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_sphere_cylinder_corner(self):
        """Test Sphere-Cylinder collision at corner (adapted from sphere_cylinder_corner fixture)"""
        geom_types = [GEO_SPHERE, GEO_CYLINDER]

        sphere_transform = wp.transform(wp.vec3(0.33, 0.0, 0.0), wp.quat_identity())

        # Cylinder with euler="30 45 0" rotation
        cylinder_quat = quat_from_euler_degree(wp.vec3(30, 45, 0), 0, 1, 2)
        cylinder_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), cylinder_quat)

        geom_data = [
            wp.vec4(0.1, 0.0, 0.0, 0.0),  # Sphere radius = 0.1
            wp.vec4(0.15, 0.2, 0.0, 0.0),  # Cylinder: radius=0.15, half_height=0.2
        ]

        transforms = [sphere_transform, cylinder_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_sphere_cylinder_cap(self):
        """Test Sphere-Cylinder collision at cap (adapted from sphere_cylinder_cap fixture)"""
        geom_types = [GEO_SPHERE, GEO_CYLINDER]

        sphere_transform = wp.transform(wp.vec3(0.26, -0.14, 0.1), wp.quat_identity())

        # Cylinder with euler="30 45 0" rotation
        cylinder_quat = quat_from_euler_degree(wp.vec3(30, 45, 0), 0, 1, 2)
        cylinder_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), cylinder_quat)

        geom_data = [
            wp.vec4(0.1, 0.0, 0.0, 0.0),  # Sphere radius = 0.1
            wp.vec4(0.15, 0.2, 0.0, 0.0),  # Cylinder: radius=0.15, half_height=0.2
        ]

        transforms = [sphere_transform, cylinder_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_sphere_cylinder_side(self):
        """Test Sphere-Cylinder collision at side (adapted from sphere_cylinder_side fixture)"""
        geom_types = [GEO_SPHERE, GEO_CYLINDER]

        sphere_transform = wp.transform(wp.vec3(0.0, -0.26, 0.0), wp.quat_identity())

        # Cylinder with euler="30 45 0" rotation
        cylinder_quat = quat_from_euler_degree(wp.vec3(30, 45, 0), 0, 1, 2)
        cylinder_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), cylinder_quat)

        geom_data = [
            wp.vec4(0.1, 0.0, 0.0, 0.0),  # Sphere radius = 0.1
            wp.vec4(0.15, 0.2, 0.0, 0.0),  # Cylinder: radius=0.15, half_height=0.2
        ]

        transforms = [sphere_transform, cylinder_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_plane_cylinder_variations(self):
        """Test Plane-Cylinder collision variations (adapted from plane_cylinder fixtures)"""
        geom_types = [GEO_PLANE, GEO_CYLINDER]

        # Test variation 1: plane_cylinder_1 - pos="0 0 0.1" euler="30 30 0"
        plane_quat = quat_from_euler_degree(wp.vec3(3, 0, 0), 0, 1, 2)
        plane_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), plane_quat)

        cylinder_quat = quat_from_euler_degree(wp.vec3(30, 30, 0), 0, 1, 2)
        cylinder_transform = wp.transform(wp.vec3(0.0, 0.0, 0.1), cylinder_quat)

        geom_data = [
            wp.vec4(0.0, 0.0, 0.0, 0.0),  # Plane data
            wp.vec4(0.05, 0.1, 0.0, 0.0),  # Cylinder: radius=0.05, half_height=0.1
        ]

        transforms = [plane_transform, cylinder_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_sphere_box_shallow(self):
        """Test Sphere-Box shallow collision (adapted from sphere_box_shallow fixture)"""
        geom_types = [GEO_BOX, GEO_SPHERE]

        box_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        sphere_transform = wp.transform(wp.vec3(-0.6, -0.6, 0.7), wp.quat_identity())

        geom_data = [
            wp.vec4(0.5, 0.5, 0.5, 0.0),  # Box half-extents
            wp.vec4(0.5, 0.0, 0.0, 0.0),  # Sphere radius = 0.5
        ]

        transforms = [box_transform, sphere_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_capsule_box_edge(self):
        """Test Capsule-Box edge collision (adapted from capsule_box_edge fixture)"""
        geom_types = [GEO_BOX, GEO_CAPSULE]

        box_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

        # Capsule at pos="0.4 0.2 0.8" with euler="0 -40 0" rotation
        capsule_quat = quat_from_euler_degree(wp.vec3(0, -40, 0), 0, 1, 2)
        capsule_transform = wp.transform(wp.vec3(0.4, 0.2, 0.8), capsule_quat)

        geom_data = [
            wp.vec4(0.5, 0.4, 0.9, 0.0),  # Box half-extents
            wp.vec4(0.5, 0.8, 0.0, 0.0),  # Capsule: radius=0.5, half_length=0.8
        ]

        transforms = [box_transform, capsule_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_capsule_box_corner(self):
        """Test Capsule-Box corner collision (adapted from capsule_box_corner fixture)"""
        geom_types = [GEO_BOX, GEO_CAPSULE]

        box_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        capsule_transform = wp.transform(wp.vec3(0.55, 0.6, 0.65), wp.quat_identity())

        geom_data = [
            wp.vec4(0.5, 0.55, 0.6, 0.0),  # Box half-extents
            wp.vec4(0.4, 0.6, 0.0, 0.0),  # Capsule: radius=0.4, half_length=0.6
        ]

        transforms = [box_transform, capsule_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_capsule_box_face_tip(self):
        """Test Capsule-Box face tip collision (adapted from capsule_box_face_tip fixture)"""
        geom_types = [GEO_BOX, GEO_CAPSULE]
        self.cutoff = 1

        box_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        capsule_transform = wp.transform(wp.vec3(0.0, 0.0, 1.5), wp.quat_identity())

        geom_data = [
            wp.vec4(0.5, 0.4, 0.9, 0.0),  # Box half-extents
            wp.vec4(0.5, 0.8, 0.0, 0.0),  # Capsule: radius=0.5, half_length=0.8
        ]

        transforms = [box_transform, capsule_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_capsule_box_face_flat(self):
        """Test Capsule-Box face flat collision (adapted from capsule_box_face_flat fixture)"""
        geom_types = [GEO_BOX, GEO_CAPSULE]

        box_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        capsule_transform = wp.transform(wp.vec3(0.5, 0.2, 0.0), wp.quat_identity())

        geom_data = [
            wp.vec4(0.5, 0.7, 0.9, 0.0),  # Box half-extents
            wp.vec4(0.2, 0.4, 0.0, 0.0),  # Capsule: radius=0.2, half_length=0.4
        ]

        transforms = [box_transform, capsule_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)

    def test_no_collision_distant_spheres(self):
        """Test no collision detection for distant spheres"""
        geom_types = [GEO_SPHERE, GEO_SPHERE]

        sphere1_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        sphere2_transform = wp.transform(wp.vec3(10.0, 0.0, 0.0), wp.quat_identity())  # Distant spheres

        geom_data = [
            wp.vec4(1.0, 0.0, 0.0, 0.0),  # Sphere 1 radius = 1.0
            wp.vec4(1.0, 0.0, 0.0, 0.0),  # Sphere 2 radius = 1.0
        ]

        transforms = [sphere1_transform, sphere2_transform]

        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=False)
        self.assertEqual(contact_count, 0)

    def test_multiple_contact_pairs(self):
        """Test with multiple geometry pairs"""
        # Create multiple pairs to test
        pairs = [(0, 1), (0, 2), (1, 2)]
        candidate_pairs = wp.array(pairs, dtype=wp.vec2i)
        num_candidate_pairs = wp.array([len(pairs)], dtype=int)

        # Three spheres in a line, close enough to generate contacts
        geom_types = [GEO_SPHERE, GEO_SPHERE, GEO_SPHERE]
        transforms = [
            wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            wp.transform(wp.vec3(1.5, 0.0, 0.0), wp.quat_identity()),
            wp.transform(wp.vec3(3.0, 0.0, 0.0), wp.quat_identity()),
        ]

        geom_data = [
            wp.vec4(1.0, 0.0, 0.0, 0.0),  # Sphere 1 radius = 1.0
            wp.vec4(1.0, 0.0, 0.0, 0.0),  # Sphere 2 radius = 1.0
            wp.vec4(1.0, 0.0, 0.0, 0.0),  # Sphere 3 radius = 1.0
        ]

        geom_types_array = wp.array(geom_types, dtype=wp.int32)
        geom_data_array = wp.array(geom_data, dtype=wp.vec4)
        geom_transforms = wp.array(transforms, dtype=wp.transform)
        geom_source = wp.zeros(len(geom_types), dtype=wp.uint64)
        geom_cutoff_array = wp.array([self.cutoff] * len(geom_types), dtype=float)

        self.narrow_phase.launch(
            candidate_pairs,
            num_candidate_pairs,
            geom_types_array,
            geom_data_array,
            geom_transforms,
            geom_source,
            geom_cutoff_array,
            self.contact_pair,
            self.contact_position,
            self.contact_normal,
            self.contact_penetration,
            self.contact_tangent,
            self.contact_count,
        )

        wp.synchronize()

        contact_count = self.contact_count.numpy()[0]

        # Should get contacts for the two overlapping pairs (0,1) and (1,2)
        # Pair (0,2) should not generate contacts as they're too far apart
        self.assertGreater(contact_count, 0)
        self.assertLessEqual(contact_count, 2)  # At most 2 contacts

    def test_contact_cutoff_distance(self):
        """Test that contact cutoff distance is respected"""
        geom_types = [GEO_SPHERE, GEO_SPHERE]

        sphere1_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        sphere2_transform = wp.transform(wp.vec3(2.05, 0.0, 0.0), wp.quat_identity())  # Just outside cutoff

        geom_data = [
            wp.vec4(1.0, 0.0, 0.0, 0.0),  # Sphere 1 radius = 1.0
            wp.vec4(1.0, 0.0, 0.0, 0.0),  # Sphere 2 radius = 1.0
        ]

        transforms = [sphere1_transform, sphere2_transform]

        # Test with small cutoff - should not generate contacts
        self.cutoff = 0.01
        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=False)
        self.assertEqual(contact_count, 0)

        # Test with larger cutoff - should generate contacts
        self.cutoff = 0.1
        self.setUp()  # Reset arrays with new cutoff
        contact_count = self._run_narrow_phase_test(geom_types, geom_data, transforms, expected_contacts=True)
        self.assertGreater(contact_count, 0)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
