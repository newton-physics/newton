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

from newton import geometry


@wp.kernel
def test_plane_sphere_kernel(
    plane_normals: wp.array(dtype=wp.vec3),
    plane_positions: wp.array(dtype=wp.vec3),
    sphere_positions: wp.array(dtype=wp.vec3),
    sphere_radii: wp.array(dtype=float),
    distances: wp.array(dtype=float),
    contact_positions: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    dist, pos = geometry.collide_plane_sphere(
        plane_normals[tid], plane_positions[tid], sphere_positions[tid], sphere_radii[tid]
    )
    distances[tid] = dist
    contact_positions[tid] = pos


@wp.kernel
def test_sphere_sphere_kernel(
    pos1: wp.array(dtype=wp.vec3),
    radius1: wp.array(dtype=float),
    pos2: wp.array(dtype=wp.vec3),
    radius2: wp.array(dtype=float),
    distances: wp.array(dtype=float),
    contact_positions: wp.array(dtype=wp.vec3),
    contact_normals: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    dist, pos, normal = geometry.collide_sphere_sphere(pos1[tid], radius1[tid], pos2[tid], radius2[tid])
    distances[tid] = dist
    contact_positions[tid] = pos
    contact_normals[tid] = normal


@wp.kernel
def test_sphere_capsule_kernel(
    sphere_positions: wp.array(dtype=wp.vec3),
    sphere_radii: wp.array(dtype=float),
    capsule_positions: wp.array(dtype=wp.vec3),
    capsule_axes: wp.array(dtype=wp.vec3),
    capsule_radii: wp.array(dtype=float),
    capsule_half_lengths: wp.array(dtype=float),
    distances: wp.array(dtype=float),
    contact_positions: wp.array(dtype=wp.vec3),
    contact_normals: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    dist, pos, normal = geometry.collide_sphere_capsule(
        sphere_positions[tid],
        sphere_radii[tid],
        capsule_positions[tid],
        capsule_axes[tid],
        capsule_radii[tid],
        capsule_half_lengths[tid],
    )
    distances[tid] = dist
    contact_positions[tid] = pos
    contact_normals[tid] = normal


@wp.kernel
def test_capsule_capsule_kernel(
    cap1_positions: wp.array(dtype=wp.vec3),
    cap1_axes: wp.array(dtype=wp.vec3),
    cap1_radii: wp.array(dtype=float),
    cap1_half_lengths: wp.array(dtype=float),
    cap2_positions: wp.array(dtype=wp.vec3),
    cap2_axes: wp.array(dtype=wp.vec3),
    cap2_radii: wp.array(dtype=float),
    cap2_half_lengths: wp.array(dtype=float),
    distances: wp.array(dtype=float),
    contact_positions: wp.array(dtype=wp.vec3),
    contact_normals: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    dist, pos, normal = geometry.collide_capsule_capsule(
        cap1_positions[tid],
        cap1_axes[tid],
        cap1_radii[tid],
        cap1_half_lengths[tid],
        cap2_positions[tid],
        cap2_axes[tid],
        cap2_radii[tid],
        cap2_half_lengths[tid],
    )
    distances[tid] = dist
    contact_positions[tid] = pos
    contact_normals[tid] = normal


@wp.kernel
def test_plane_ellipsoid_kernel(
    plane_normals: wp.array(dtype=wp.vec3),
    plane_positions: wp.array(dtype=wp.vec3),
    ellipsoid_positions: wp.array(dtype=wp.vec3),
    ellipsoid_rotations: wp.array(dtype=wp.mat33),
    ellipsoid_sizes: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=float),
    contact_positions: wp.array(dtype=wp.vec3),
    contact_normals: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    dist, pos, normal = geometry.collide_plane_ellipsoid(
        plane_normals[tid],
        plane_positions[tid],
        ellipsoid_positions[tid],
        ellipsoid_rotations[tid],
        ellipsoid_sizes[tid],
    )
    distances[tid] = dist
    contact_positions[tid] = pos
    contact_normals[tid] = normal


@wp.kernel
def test_sphere_cylinder_kernel(
    sphere_positions: wp.array(dtype=wp.vec3),
    sphere_radii: wp.array(dtype=float),
    cylinder_positions: wp.array(dtype=wp.vec3),
    cylinder_axes: wp.array(dtype=wp.vec3),
    cylinder_radii: wp.array(dtype=float),
    cylinder_half_heights: wp.array(dtype=float),
    distances: wp.array(dtype=float),
    contact_positions: wp.array(dtype=wp.vec3),
    contact_normals: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    dist, pos, normal = geometry.collide_sphere_cylinder(
        sphere_positions[tid],
        sphere_radii[tid],
        cylinder_positions[tid],
        cylinder_axes[tid],
        cylinder_radii[tid],
        cylinder_half_heights[tid],
    )
    distances[tid] = dist
    contact_positions[tid] = pos
    contact_normals[tid] = normal


@wp.kernel
def test_sphere_box_kernel(
    sphere_positions: wp.array(dtype=wp.vec3),
    sphere_radii: wp.array(dtype=float),
    box_positions: wp.array(dtype=wp.vec3),
    box_rotations: wp.array(dtype=wp.mat33),
    box_sizes: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=float),
    contact_positions: wp.array(dtype=wp.vec3),
    contact_normals: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    dist, pos, normal = geometry.collide_sphere_box(
        sphere_positions[tid], sphere_radii[tid], box_positions[tid], box_rotations[tid], box_sizes[tid]
    )
    distances[tid] = dist
    contact_positions[tid] = pos
    contact_normals[tid] = normal


@wp.kernel
def test_plane_capsule_kernel(
    plane_normals: wp.array(dtype=wp.vec3),
    plane_positions: wp.array(dtype=wp.vec3),
    capsule_positions: wp.array(dtype=wp.vec3),
    capsule_axes: wp.array(dtype=wp.vec3),
    capsule_radii: wp.array(dtype=float),
    capsule_half_lengths: wp.array(dtype=float),
    distances: wp.array(dtype=wp.vec2),
    contact_positions: wp.array(dtype=wp.types.matrix((2, 3), wp.float32)),
    contact_frames: wp.array(dtype=wp.mat33),
):
    tid = wp.tid()
    dist, pos, frame = geometry.collide_plane_capsule(
        plane_normals[tid],
        plane_positions[tid],
        capsule_positions[tid],
        capsule_axes[tid],
        capsule_radii[tid],
        capsule_half_lengths[tid],
    )
    distances[tid] = dist
    contact_positions[tid] = pos
    contact_frames[tid] = frame


@wp.kernel
def test_plane_box_kernel(
    plane_normals: wp.array(dtype=wp.vec3),
    plane_positions: wp.array(dtype=wp.vec3),
    box_positions: wp.array(dtype=wp.vec3),
    box_rotations: wp.array(dtype=wp.mat33),
    box_sizes: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=wp.vec4),
    contact_positions: wp.array(dtype=wp.types.matrix((4, 3), wp.float32)),
    contact_normals: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    dist, pos, normal = geometry.collide_plane_box(
        plane_normals[tid], plane_positions[tid], box_positions[tid], box_rotations[tid], box_sizes[tid]
    )
    distances[tid] = dist
    contact_positions[tid] = pos
    contact_normals[tid] = normal


@wp.kernel
def test_plane_cylinder_kernel(
    plane_normals: wp.array(dtype=wp.vec3),
    plane_positions: wp.array(dtype=wp.vec3),
    cylinder_centers: wp.array(dtype=wp.vec3),
    cylinder_axes: wp.array(dtype=wp.vec3),
    cylinder_radii: wp.array(dtype=float),
    cylinder_half_heights: wp.array(dtype=float),
    distances: wp.array(dtype=wp.vec4),
    contact_positions: wp.array(dtype=wp.types.matrix((4, 3), wp.float32)),
    contact_normals: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    dist, pos, normal = geometry.collide_plane_cylinder(
        plane_normals[tid],
        plane_positions[tid],
        cylinder_centers[tid],
        cylinder_axes[tid],
        cylinder_radii[tid],
        cylinder_half_heights[tid],
    )
    distances[tid] = dist
    contact_positions[tid] = pos
    contact_normals[tid] = normal


@wp.kernel
def test_box_box_kernel(
    box1_positions: wp.array(dtype=wp.vec3),
    box1_rotations: wp.array(dtype=wp.mat33),
    box1_sizes: wp.array(dtype=wp.vec3),
    box2_positions: wp.array(dtype=wp.vec3),
    box2_rotations: wp.array(dtype=wp.mat33),
    box2_sizes: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=wp.types.vector(8, wp.float32)),
    contact_positions: wp.array(dtype=wp.types.matrix((8, 3), wp.float32)),
    contact_normals: wp.array(dtype=wp.types.matrix((8, 3), wp.float32)),
):
    tid = wp.tid()
    dist, pos, normals = geometry.collide_box_box(
        box1_positions[tid],
        box1_rotations[tid],
        box1_sizes[tid],
        box2_positions[tid],
        box2_rotations[tid],
        box2_sizes[tid],
    )
    distances[tid] = dist
    contact_positions[tid] = pos
    contact_normals[tid] = normals


@wp.kernel
def test_box_box_with_margin_kernel(
    box1_positions: wp.array(dtype=wp.vec3),
    box1_rotations: wp.array(dtype=wp.mat33),
    box1_sizes: wp.array(dtype=wp.vec3),
    box2_positions: wp.array(dtype=wp.vec3),
    box2_rotations: wp.array(dtype=wp.mat33),
    box2_sizes: wp.array(dtype=wp.vec3),
    margins: wp.array(dtype=float),
    distances: wp.array(dtype=wp.types.vector(8, wp.float32)),
    contact_positions: wp.array(dtype=wp.types.matrix((8, 3), wp.float32)),
    contact_normals: wp.array(dtype=wp.types.matrix((8, 3), wp.float32)),
):
    tid = wp.tid()
    dist, pos, normals = geometry.collide_box_box(
        box1_positions[tid],
        box1_rotations[tid],
        box1_sizes[tid],
        box2_positions[tid],
        box2_rotations[tid],
        box2_sizes[tid],
        margins[tid],
    )
    distances[tid] = dist
    contact_positions[tid] = pos
    contact_normals[tid] = normals


@wp.kernel
def test_capsule_box_kernel(
    capsule_positions: wp.array(dtype=wp.vec3),
    capsule_axes: wp.array(dtype=wp.vec3),
    capsule_radii: wp.array(dtype=float),
    capsule_half_lengths: wp.array(dtype=float),
    box_positions: wp.array(dtype=wp.vec3),
    box_rotations: wp.array(dtype=wp.mat33),
    box_sizes: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=wp.vec2),
    contact_positions: wp.array(dtype=wp.types.matrix((2, 3), wp.float32)),
    contact_normals: wp.array(dtype=wp.types.matrix((2, 3), wp.float32)),
):
    tid = wp.tid()
    dist, pos, normals = geometry.collide_capsule_box(
        capsule_positions[tid],
        capsule_axes[tid],
        capsule_radii[tid],
        capsule_half_lengths[tid],
        box_positions[tid],
        box_rotations[tid],
        box_sizes[tid],
    )
    distances[tid] = dist
    contact_positions[tid] = pos
    contact_normals[tid] = normals


class TestCollisionPrimitives(unittest.TestCase):
    def test_plane_sphere(self):
        """Test plane-sphere collision."""
        test_cases = [
            # Plane normal, plane pos, sphere pos, sphere radius, expected distance
            ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0], 1.0, 1.0),  # Above plane
            ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.5], 1.0, -0.5),  # Intersecting plane
            ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0], 0.5, 0.0),  # Touching plane
        ]

        plane_normals = wp.array([wp.vec3(tc[0][0], tc[0][1], tc[0][2]) for tc in test_cases], dtype=wp.vec3)
        plane_positions = wp.array([wp.vec3(tc[1][0], tc[1][1], tc[1][2]) for tc in test_cases], dtype=wp.vec3)
        sphere_positions = wp.array([wp.vec3(tc[2][0], tc[2][1], tc[2][2]) for tc in test_cases], dtype=wp.vec3)
        sphere_radii = wp.array([tc[3] for tc in test_cases], dtype=float)
        distances = wp.array([0.0] * len(test_cases), dtype=float)
        contact_positions = wp.array([wp.vec3(0.0, 0.0, 0.0)] * len(test_cases), dtype=wp.vec3)

        wp.launch(
            test_plane_sphere_kernel,
            dim=len(test_cases),
            inputs=[plane_normals, plane_positions, sphere_positions, sphere_radii, distances, contact_positions],
        )
        wp.synchronize()

        distances_np = distances.numpy()

        # Verify expected distances
        for i, expected_dist in enumerate([tc[4] for tc in test_cases]):
            self.assertAlmostEqual(distances_np[i], expected_dist, places=6, msg=f"Distance failed for test case {i}")

    def test_sphere_sphere(self):
        """Test sphere-sphere collision."""
        test_cases = [
            # pos1, radius1, pos2, radius2, expected_distance
            ([0.0, 0.0, 0.0], 1.0, [3.0, 0.0, 0.0], 1.0, 1.0),  # Separated
            ([0.0, 0.0, 0.0], 1.0, [1.5, 0.0, 0.0], 1.0, -0.5),  # Overlapping
            ([0.0, 0.0, 0.0], 1.0, [2.0, 0.0, 0.0], 1.0, 0.0),  # Exactly touching
        ]

        pos1 = wp.array([wp.vec3(tc[0][0], tc[0][1], tc[0][2]) for tc in test_cases], dtype=wp.vec3)
        radius1 = wp.array([tc[1] for tc in test_cases], dtype=float)
        pos2 = wp.array([wp.vec3(tc[2][0], tc[2][1], tc[2][2]) for tc in test_cases], dtype=wp.vec3)
        radius2 = wp.array([tc[3] for tc in test_cases], dtype=float)
        distances = wp.array([0.0] * len(test_cases), dtype=float)
        contact_positions = wp.array([wp.vec3(0.0, 0.0, 0.0)] * len(test_cases), dtype=wp.vec3)
        contact_normals = wp.array([wp.vec3(0.0, 0.0, 0.0)] * len(test_cases), dtype=wp.vec3)

        wp.launch(
            test_sphere_sphere_kernel,
            dim=len(test_cases),
            inputs=[pos1, radius1, pos2, radius2, distances, contact_positions, contact_normals],
        )
        wp.synchronize()

        distances_np = distances.numpy()
        normals_np = contact_normals.numpy()

        # Verify expected distances
        for i, expected_dist in enumerate([tc[4] for tc in test_cases]):
            self.assertAlmostEqual(distances_np[i], expected_dist, places=6, msg=f"Distance failed for test case {i}")

        # Check normal vectors are unit length (except for zero distance case)
        for i in range(len(test_cases)):
            if i != 2:  # Skip exact touching case
                normal_length = np.linalg.norm(normals_np[i])
                self.assertAlmostEqual(normal_length, 1.0, places=6, msg=f"Normal not unit length for test case {i}")

    def test_sphere_capsule(self):
        """Test sphere-capsule collision."""
        test_cases = [
            # Sphere intersects capsule cylinder part - moved closer for overlap
            ([0.0, 0.8, 0.0], 0.5, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.5, 1.0),
            # Sphere intersects capsule cap - moved closer for overlap
            ([0.0, 0.0, 1.3], 0.5, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.5, 1.0),
        ]

        sphere_positions = wp.array([wp.vec3(tc[0][0], tc[0][1], tc[0][2]) for tc in test_cases], dtype=wp.vec3)
        sphere_radii = wp.array([tc[1] for tc in test_cases], dtype=float)
        capsule_positions = wp.array([wp.vec3(tc[2][0], tc[2][1], tc[2][2]) for tc in test_cases], dtype=wp.vec3)
        capsule_axes = wp.array([wp.vec3(tc[3][0], tc[3][1], tc[3][2]) for tc in test_cases], dtype=wp.vec3)
        capsule_radii = wp.array([tc[4] for tc in test_cases], dtype=float)
        capsule_half_lengths = wp.array([tc[5] for tc in test_cases], dtype=float)
        distances = wp.array([0.0] * len(test_cases), dtype=float)
        contact_positions = wp.array([wp.vec3(0.0, 0.0, 0.0)] * len(test_cases), dtype=wp.vec3)
        contact_normals = wp.array([wp.vec3(0.0, 0.0, 0.0)] * len(test_cases), dtype=wp.vec3)

        wp.launch(
            test_sphere_capsule_kernel,
            dim=len(test_cases),
            inputs=[
                sphere_positions,
                sphere_radii,
                capsule_positions,
                capsule_axes,
                capsule_radii,
                capsule_half_lengths,
                distances,
                contact_positions,
                contact_normals,
            ],
        )
        wp.synchronize()

        distances_np = distances.numpy()

        # Basic sanity checks - should have negative distances indicating penetration
        for i in range(len(test_cases)):
            self.assertLess(
                distances_np[i],
                0.0,
                msg=f"Expected penetration (negative distance) for test case {i}, got {distances_np[i]}",
            )

    def test_capsule_capsule(self):
        """Test capsule-capsule collision."""
        test_cases = [
            # Parallel capsules
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.5, 1.0, [0.0, 1.5, 0.0], [1.0, 0.0, 0.0], 0.5, 1.0),
            # Perpendicular capsules
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.5, 1.0, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.5, 1.0),
        ]

        cap1_positions = wp.array([wp.vec3(tc[0][0], tc[0][1], tc[0][2]) for tc in test_cases], dtype=wp.vec3)
        cap1_axes = wp.array([wp.vec3(tc[1][0], tc[1][1], tc[1][2]) for tc in test_cases], dtype=wp.vec3)
        cap1_radii = wp.array([tc[2] for tc in test_cases], dtype=float)
        cap1_half_lengths = wp.array([tc[3] for tc in test_cases], dtype=float)
        cap2_positions = wp.array([wp.vec3(tc[4][0], tc[4][1], tc[4][2]) for tc in test_cases], dtype=wp.vec3)
        cap2_axes = wp.array([wp.vec3(tc[5][0], tc[5][1], tc[5][2]) for tc in test_cases], dtype=wp.vec3)
        cap2_radii = wp.array([tc[6] for tc in test_cases], dtype=float)
        cap2_half_lengths = wp.array([tc[7] for tc in test_cases], dtype=float)
        distances = wp.array([0.0] * len(test_cases), dtype=float)
        contact_positions = wp.array([wp.vec3(0.0, 0.0, 0.0)] * len(test_cases), dtype=wp.vec3)
        contact_normals = wp.array([wp.vec3(0.0, 0.0, 0.0)] * len(test_cases), dtype=wp.vec3)

        wp.launch(
            test_capsule_capsule_kernel,
            dim=len(test_cases),
            inputs=[
                cap1_positions,
                cap1_axes,
                cap1_radii,
                cap1_half_lengths,
                cap2_positions,
                cap2_axes,
                cap2_radii,
                cap2_half_lengths,
                distances,
                contact_positions,
                contact_normals,
            ],
        )
        wp.synchronize()

        distances_np = distances.numpy()

        # Basic sanity checks - first case should be separated, second overlapping
        self.assertGreater(distances_np[0], 0.0, msg="Parallel capsules should be separated")
        self.assertLess(distances_np[1], 0.0, msg="Intersecting capsules should overlap")

    def test_plane_ellipsoid(self):
        """Test plane-ellipsoid collision."""
        # Identity rotation matrix
        identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        test_cases = [
            # Ellipsoid above plane
            ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0], identity, [1.0, 1.0, 1.5]),
            # Ellipsoid intersecting plane
            ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], identity, [1.0, 1.0, 1.5]),
        ]

        plane_normals = wp.array([wp.vec3(tc[0][0], tc[0][1], tc[0][2]) for tc in test_cases], dtype=wp.vec3)
        plane_positions = wp.array([wp.vec3(tc[1][0], tc[1][1], tc[1][2]) for tc in test_cases], dtype=wp.vec3)
        ellipsoid_positions = wp.array([wp.vec3(tc[2][0], tc[2][1], tc[2][2]) for tc in test_cases], dtype=wp.vec3)
        ellipsoid_rotations = wp.array([tc[3] for tc in test_cases], dtype=wp.mat33)
        ellipsoid_sizes = wp.array([wp.vec3(tc[4][0], tc[4][1], tc[4][2]) for tc in test_cases], dtype=wp.vec3)
        distances = wp.array([0.0] * len(test_cases), dtype=float)
        contact_positions = wp.array([wp.vec3(0.0, 0.0, 0.0)] * len(test_cases), dtype=wp.vec3)
        contact_normals = wp.array([wp.vec3(0.0, 0.0, 0.0)] * len(test_cases), dtype=wp.vec3)

        wp.launch(
            test_plane_ellipsoid_kernel,
            dim=len(test_cases),
            inputs=[
                plane_normals,
                plane_positions,
                ellipsoid_positions,
                ellipsoid_rotations,
                ellipsoid_sizes,
                distances,
                contact_positions,
                contact_normals,
            ],
        )
        wp.synchronize()

        distances_np = distances.numpy()

        # First case should be separated, second should be penetrating
        self.assertGreater(distances_np[0], 0.0, msg="Ellipsoid should be above plane")
        self.assertLess(distances_np[1], 0.0, msg="Ellipsoid should penetrate plane")

    def test_sphere_cylinder(self):
        """Test sphere-cylinder collision."""
        test_cases = [
            # Sphere penetrating cylinder side
            ([1.4, 0.0, 0.0], 0.5, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0, 1.0),
            # Sphere penetrating cylinder cap
            ([0.0, 0.0, 1.4], 0.5, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0, 1.0),
            # Sphere penetrating cylinder corner
            ([1.4, 0.0, 1.2], 0.5, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0, 1.0),
        ]

        sphere_positions = wp.array([wp.vec3(tc[0][0], tc[0][1], tc[0][2]) for tc in test_cases], dtype=wp.vec3)
        sphere_radii = wp.array([tc[1] for tc in test_cases], dtype=float)
        cylinder_positions = wp.array([wp.vec3(tc[2][0], tc[2][1], tc[2][2]) for tc in test_cases], dtype=wp.vec3)
        cylinder_axes = wp.array([wp.vec3(tc[3][0], tc[3][1], tc[3][2]) for tc in test_cases], dtype=wp.vec3)
        cylinder_radii = wp.array([tc[4] for tc in test_cases], dtype=float)
        cylinder_half_heights = wp.array([tc[5] for tc in test_cases], dtype=float)
        distances = wp.array([0.0] * len(test_cases), dtype=float)
        contact_positions = wp.array([wp.vec3(0.0, 0.0, 0.0)] * len(test_cases), dtype=wp.vec3)
        contact_normals = wp.array([wp.vec3(0.0, 0.0, 0.0)] * len(test_cases), dtype=wp.vec3)

        wp.launch(
            test_sphere_cylinder_kernel,
            dim=len(test_cases),
            inputs=[
                sphere_positions,
                sphere_radii,
                cylinder_positions,
                cylinder_axes,
                cylinder_radii,
                cylinder_half_heights,
                distances,
                contact_positions,
                contact_normals,
            ],
        )
        wp.synchronize()

        distances_np = distances.numpy()

        # All test cases should result in penetration (dist <= 0)
        for i in range(len(test_cases)):
            self.assertLessEqual(
                distances_np[i], 0.0, msg=f"Expected penetration for test case {i}, got {distances_np[i]}"
            )

    def test_sphere_box(self):
        """Test sphere-box collision."""
        # Identity rotation matrix
        identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        test_cases = [
            # Sphere outside box
            ([2.0, 0.0, 0.0], 0.5, [0.0, 0.0, 0.0], identity, [1.0, 1.0, 1.0]),
            # Sphere intersecting box face
            ([1.2, 0.0, 0.0], 0.5, [0.0, 0.0, 0.0], identity, [1.0, 1.0, 1.0]),
            # Sphere inside box
            ([0.0, 0.0, 0.0], 0.5, [0.0, 0.0, 0.0], identity, [1.0, 1.0, 1.0]),
        ]

        sphere_positions = wp.array([wp.vec3(tc[0][0], tc[0][1], tc[0][2]) for tc in test_cases], dtype=wp.vec3)
        sphere_radii = wp.array([tc[1] for tc in test_cases], dtype=float)
        box_positions = wp.array([wp.vec3(tc[2][0], tc[2][1], tc[2][2]) for tc in test_cases], dtype=wp.vec3)
        box_rotations = wp.array([tc[3] for tc in test_cases], dtype=wp.mat33)
        box_sizes = wp.array([wp.vec3(tc[4][0], tc[4][1], tc[4][2]) for tc in test_cases], dtype=wp.vec3)
        distances = wp.array([0.0] * len(test_cases), dtype=float)
        contact_positions = wp.array([wp.vec3(0.0, 0.0, 0.0)] * len(test_cases), dtype=wp.vec3)
        contact_normals = wp.array([wp.vec3(0.0, 0.0, 0.0)] * len(test_cases), dtype=wp.vec3)

        wp.launch(
            test_sphere_box_kernel,
            dim=len(test_cases),
            inputs=[
                sphere_positions,
                sphere_radii,
                box_positions,
                box_rotations,
                box_sizes,
                distances,
                contact_positions,
                contact_normals,
            ],
        )
        wp.synchronize()

        distances_np = distances.numpy()

        # Verify expected collision states
        self.assertGreater(distances_np[0], 0.0, msg="Sphere should be outside box")
        self.assertLess(distances_np[1], 0.5, msg="Sphere should be close to or intersecting box")
        self.assertLess(distances_np[2], 0.0, msg="Sphere inside box should have negative distance")

    def test_plane_capsule(self):
        """Test plane-capsule collision."""
        test_cases = [
            # Capsule above plane
            ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.0, 0.0, 0.0], 0.5, 1.0),
            # Capsule intersecting plane
            ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.3], [1.0, 0.0, 0.0], 0.5, 1.0),
        ]

        plane_normals = wp.array([wp.vec3(tc[0][0], tc[0][1], tc[0][2]) for tc in test_cases], dtype=wp.vec3)
        plane_positions = wp.array([wp.vec3(tc[1][0], tc[1][1], tc[1][2]) for tc in test_cases], dtype=wp.vec3)
        capsule_positions = wp.array([wp.vec3(tc[2][0], tc[2][1], tc[2][2]) for tc in test_cases], dtype=wp.vec3)
        capsule_axes = wp.array([wp.vec3(tc[3][0], tc[3][1], tc[3][2]) for tc in test_cases], dtype=wp.vec3)
        capsule_radii = wp.array([tc[4] for tc in test_cases], dtype=float)
        capsule_half_lengths = wp.array([tc[5] for tc in test_cases], dtype=float)
        distances = wp.array([wp.vec2(0.0, 0.0)] * len(test_cases), dtype=wp.vec2)
        contact_positions = wp.array(
            [wp.types.matrix((2, 3), wp.float32)()] * len(test_cases), dtype=wp.types.matrix((2, 3), wp.float32)
        )
        contact_frames = wp.array([wp.mat33()] * len(test_cases), dtype=wp.mat33)

        wp.launch(
            test_plane_capsule_kernel,
            dim=len(test_cases),
            inputs=[
                plane_normals,
                plane_positions,
                capsule_positions,
                capsule_axes,
                capsule_radii,
                capsule_half_lengths,
                distances,
                contact_positions,
                contact_frames,
            ],
        )
        wp.synchronize()

        distances_np = distances.numpy()

        # Plane-capsule always returns 2 finite distances: positive=separation, negative=penetration
        # Case 0: Above plane should have positive distances (separation)
        self.assertGreater(distances_np[0][0], 0.0, msg="Capsule above plane should have positive distance")
        self.assertGreater(distances_np[0][1], 0.0, msg="Capsule above plane should have positive distance")

        # Case 1: Intersecting plane should have negative distances (penetration)
        self.assertLess(distances_np[1][0], 0.0, msg="Intersecting capsule should penetrate plane")
        self.assertLess(distances_np[1][1], 0.0, msg="Intersecting capsule should penetrate plane")

    def test_plane_box(self):
        """Test plane-box collision."""
        # Identity rotation matrix
        identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        test_cases = [
            # Box above plane (should have no contacts)
            ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0], identity, [1.0, 1.0, 1.0], 0),
            # Box intersecting plane (should have contacts)
            ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.5], identity, [1.0, 1.0, 1.0], 4),
        ]

        plane_normals = wp.array([wp.vec3(tc[0][0], tc[0][1], tc[0][2]) for tc in test_cases], dtype=wp.vec3)
        plane_positions = wp.array([wp.vec3(tc[1][0], tc[1][1], tc[1][2]) for tc in test_cases], dtype=wp.vec3)
        box_positions = wp.array([wp.vec3(tc[2][0], tc[2][1], tc[2][2]) for tc in test_cases], dtype=wp.vec3)
        box_rotations = wp.array([tc[3] for tc in test_cases], dtype=wp.mat33)
        box_sizes = wp.array([wp.vec3(tc[4][0], tc[4][1], tc[4][2]) for tc in test_cases], dtype=wp.vec3)
        distances = wp.array([wp.vec4()] * len(test_cases), dtype=wp.vec4)
        contact_positions = wp.array(
            [wp.types.matrix((4, 3), wp.float32)()] * len(test_cases), dtype=wp.types.matrix((4, 3), wp.float32)
        )
        contact_normals = wp.array([wp.vec3()] * len(test_cases), dtype=wp.vec3)

        wp.launch(
            test_plane_box_kernel,
            dim=len(test_cases),
            inputs=[
                plane_normals,
                plane_positions,
                box_positions,
                box_rotations,
                box_sizes,
                distances,
                contact_positions,
                contact_normals,
            ],
        )
        wp.synchronize()

        distances_np = distances.numpy()

        # Check expected contact counts
        for i in range(len(test_cases)):
            valid_contacts = sum(1 for d in distances_np[i] if d != float("inf"))
            expected_contacts = test_cases[i][5]  # Expected contact count
            self.assertEqual(
                valid_contacts,
                expected_contacts,
                msg=f"Expected {expected_contacts} contacts but got {valid_contacts} for test case {i}",
            )

    def test_plane_cylinder(self):
        """Test plane-cylinder collision."""
        test_cases = [
            # Cylinder far above plane (should have no contacts)
            ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, 1.0], 1.0, 1.0, 0),
            # Cylinder intersecting plane (should have contacts)
            ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 1.0], 1.0, 1.0, 4),
        ]

        plane_normals = wp.array([wp.vec3(tc[0][0], tc[0][1], tc[0][2]) for tc in test_cases], dtype=wp.vec3)
        plane_positions = wp.array([wp.vec3(tc[1][0], tc[1][1], tc[1][2]) for tc in test_cases], dtype=wp.vec3)
        cylinder_centers = wp.array([wp.vec3(tc[2][0], tc[2][1], tc[2][2]) for tc in test_cases], dtype=wp.vec3)
        cylinder_axes = wp.array([wp.vec3(tc[3][0], tc[3][1], tc[3][2]) for tc in test_cases], dtype=wp.vec3)
        cylinder_radii = wp.array([tc[4] for tc in test_cases], dtype=float)
        cylinder_half_heights = wp.array([tc[5] for tc in test_cases], dtype=float)
        distances = wp.array([wp.vec4()] * len(test_cases), dtype=wp.vec4)
        contact_positions = wp.array(
            [wp.types.matrix((4, 3), wp.float32)()] * len(test_cases), dtype=wp.types.matrix((4, 3), wp.float32)
        )
        contact_normals = wp.array([wp.vec3()] * len(test_cases), dtype=wp.vec3)

        wp.launch(
            test_plane_cylinder_kernel,
            dim=len(test_cases),
            inputs=[
                plane_normals,
                plane_positions,
                cylinder_centers,
                cylinder_axes,
                cylinder_radii,
                cylinder_half_heights,
                distances,
                contact_positions,
                contact_normals,
            ],
        )
        wp.synchronize()

        distances_np = distances.numpy()

        # Check collision behavior - far cylinder should have fewer contacts than intersecting one
        valid_contacts_0 = sum(1 for d in distances_np[0] if d != float("inf"))
        valid_contacts_1 = sum(1 for d in distances_np[1] if d != float("inf"))

        # The intersecting cylinder should have more contacts than the far one
        self.assertGreaterEqual(
            valid_contacts_1,
            valid_contacts_0,
            msg=f"Intersecting cylinder should have >= contacts than far cylinder: {valid_contacts_1} vs {valid_contacts_0}",
        )
        self.assertGreater(valid_contacts_1, 0, msg="Intersecting cylinder should have at least one contact")

        # Check that contact distances make sense - closer cylinder should have smaller/negative distances
        if valid_contacts_0 > 0 and valid_contacts_1 > 0:
            min_dist_0 = min(d for d in distances_np[0] if d != float("inf"))
            min_dist_1 = min(d for d in distances_np[1] if d != float("inf"))
            self.assertLess(
                min_dist_1,
                min_dist_0,
                msg=f"Intersecting cylinder should have smaller contact distance: {min_dist_1} vs {min_dist_0}",
            )

    def test_box_box(self):
        """Test box-box collision."""
        # Identity rotation matrix
        identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        test_cases = [
            # Separated boxes
            ([0.0, 0.0, 0.0], identity, [1.0, 1.0, 1.0], [3.0, 0.0, 0.0], identity, [1.0, 1.0, 1.0]),
            # Overlapping boxes
            ([0.0, 0.0, 0.0], identity, [1.0, 1.0, 1.0], [1.5, 0.0, 0.0], identity, [1.0, 1.0, 1.0]),
        ]

        box1_positions = wp.array([wp.vec3(tc[0][0], tc[0][1], tc[0][2]) for tc in test_cases], dtype=wp.vec3)
        box1_rotations = wp.array([tc[1] for tc in test_cases], dtype=wp.mat33)
        box1_sizes = wp.array([wp.vec3(tc[2][0], tc[2][1], tc[2][2]) for tc in test_cases], dtype=wp.vec3)
        box2_positions = wp.array([wp.vec3(tc[3][0], tc[3][1], tc[3][2]) for tc in test_cases], dtype=wp.vec3)
        box2_rotations = wp.array([tc[4] for tc in test_cases], dtype=wp.mat33)
        box2_sizes = wp.array([wp.vec3(tc[5][0], tc[5][1], tc[5][2]) for tc in test_cases], dtype=wp.vec3)
        distances = wp.array([wp.types.vector(8, wp.float32)()] * len(test_cases), dtype=wp.types.vector(8, wp.float32))
        contact_positions = wp.array(
            [wp.types.matrix((8, 3), wp.float32)()] * len(test_cases), dtype=wp.types.matrix((8, 3), wp.float32)
        )
        contact_normals = wp.array(
            [wp.types.matrix((8, 3), wp.float32)()] * len(test_cases), dtype=wp.types.matrix((8, 3), wp.float32)
        )

        wp.launch(
            test_box_box_kernel,
            dim=len(test_cases),
            inputs=[
                box1_positions,
                box1_rotations,
                box1_sizes,
                box2_positions,
                box2_rotations,
                box2_sizes,
                distances,
                contact_positions,
                contact_normals,
            ],
        )
        wp.synchronize()

        distances_np = distances.numpy()

        # Count valid contacts for each test case
        for i in range(len(test_cases)):
            valid_contacts = sum(1 for j in range(8) if distances_np[i][j] != float("inf"))

            if i == 0:  # Separated boxes
                self.assertEqual(valid_contacts, 0, msg="Separated boxes should have no contacts")
            elif i == 1:  # Overlapping boxes
                self.assertGreater(valid_contacts, 0, msg="Overlapping boxes should have contacts")

    def test_box_box_margin(self):
        """Test box-box collision with margin parameter.

        This test verifies that the margin parameter works correctly:
        - Two boxes stacked vertically with a gap of 0.2
        - With margin=0.0, no contacts should be found (boxes separated)
        - With margin=0.3, contacts should be found (margin > gap)
        - With margin=0.1, no contacts should be found (margin < gap)
        """
        # Identity rotation matrix
        identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        # Box sizes (half-extents)
        box_size = [0.5, 0.5, 0.5]

        # Box positions: stacked vertically with gap of 0.2
        # Box 1 at z=0, top face at z=0.5
        # Box 2 at z=1.2, bottom face at z=0.7
        # Gap = 0.7 - 0.5 = 0.2
        test_cases = [
            # box1_pos, box1_rot, box1_size, box2_pos, box2_rot, box2_size, margin, expect_contacts
            (
                [0.0, 0.0, 0.0],
                identity,
                box_size,
                [0.0, 0.0, 1.2],
                identity,
                box_size,
                0.0,
                False,
            ),  # No margin, no contact
            (
                [0.0, 0.0, 0.0],
                identity,
                box_size,
                [0.0, 0.0, 1.2],
                identity,
                box_size,
                0.3,
                True,
            ),  # Margin > gap, contact
            (
                [0.0, 0.0, 0.0],
                identity,
                box_size,
                [0.0, 0.0, 1.2],
                identity,
                box_size,
                0.1,
                False,
            ),  # Margin < gap, no contact
            (
                [0.0, 0.0, 0.0],
                identity,
                box_size,
                [0.0, 0.0, 1.2],
                identity,
                box_size,
                0.201,
                True,
            ),  # Margin = gap, contact
        ]

        box1_positions = wp.array([wp.vec3(tc[0][0], tc[0][1], tc[0][2]) for tc in test_cases], dtype=wp.vec3)
        box1_rotations = wp.array([tc[1] for tc in test_cases], dtype=wp.mat33)
        box1_sizes = wp.array([wp.vec3(tc[2][0], tc[2][1], tc[2][2]) for tc in test_cases], dtype=wp.vec3)
        box2_positions = wp.array([wp.vec3(tc[3][0], tc[3][1], tc[3][2]) for tc in test_cases], dtype=wp.vec3)
        box2_rotations = wp.array([tc[4] for tc in test_cases], dtype=wp.mat33)
        box2_sizes = wp.array([wp.vec3(tc[5][0], tc[5][1], tc[5][2]) for tc in test_cases], dtype=wp.vec3)
        margins = wp.array([tc[6] for tc in test_cases], dtype=float)
        distances = wp.array([wp.types.vector(8, wp.float32)()] * len(test_cases), dtype=wp.types.vector(8, wp.float32))
        contact_positions = wp.array(
            [wp.types.matrix((8, 3), wp.float32)()] * len(test_cases), dtype=wp.types.matrix((8, 3), wp.float32)
        )
        contact_normals = wp.array(
            [wp.types.matrix((8, 3), wp.float32)()] * len(test_cases), dtype=wp.types.matrix((8, 3), wp.float32)
        )

        wp.launch(
            test_box_box_with_margin_kernel,
            dim=len(test_cases),
            inputs=[
                box1_positions,
                box1_rotations,
                box1_sizes,
                box2_positions,
                box2_rotations,
                box2_sizes,
                margins,
                distances,
                contact_positions,
                contact_normals,
            ],
        )
        wp.synchronize()

        distances_np = distances.numpy()

        # Verify expected contact behavior for each test case
        for i in range(len(test_cases)):
            valid_contacts = sum(1 for j in range(8) if distances_np[i][j] != float("inf"))
            expect_contacts = test_cases[i][7]
            margin = test_cases[i][6]

            if expect_contacts:
                self.assertGreater(
                    valid_contacts,
                    0,
                    msg=f"Test case {i}: Expected contacts with margin={margin}, but found {valid_contacts}",
                )
            else:
                self.assertEqual(
                    valid_contacts,
                    0,
                    msg=f"Test case {i}: Expected no contacts with margin={margin}, but found {valid_contacts}",
                )

    def test_capsule_box(self):
        """Test capsule-box collision."""
        # Identity rotation matrix
        identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        test_cases = [
            # Capsule intersecting box
            ([0.0, 0.0, 1.5], [0.0, 0.0, 1.0], 0.5, 1.0, [0.0, 0.0, 0.0], identity, [1.0, 1.0, 1.0]),
            # Capsule above box
            ([0.0, 0.0, 3.0], [0.0, 0.0, 1.0], 0.5, 1.0, [0.0, 0.0, 0.0], identity, [1.0, 1.0, 1.0]),
        ]

        capsule_positions = wp.array([wp.vec3(tc[0][0], tc[0][1], tc[0][2]) for tc in test_cases], dtype=wp.vec3)
        capsule_axes = wp.array([wp.vec3(tc[1][0], tc[1][1], tc[1][2]) for tc in test_cases], dtype=wp.vec3)
        capsule_radii = wp.array([tc[2] for tc in test_cases], dtype=float)
        capsule_half_lengths = wp.array([tc[3] for tc in test_cases], dtype=float)
        box_positions = wp.array([wp.vec3(tc[4][0], tc[4][1], tc[4][2]) for tc in test_cases], dtype=wp.vec3)
        box_rotations = wp.array([tc[5] for tc in test_cases], dtype=wp.mat33)
        box_sizes = wp.array([wp.vec3(tc[6][0], tc[6][1], tc[6][2]) for tc in test_cases], dtype=wp.vec3)
        distances = wp.array([wp.vec2()] * len(test_cases), dtype=wp.vec2)
        contact_positions = wp.array(
            [wp.types.matrix((2, 3), wp.float32)()] * len(test_cases), dtype=wp.types.matrix((2, 3), wp.float32)
        )
        contact_normals = wp.array(
            [wp.types.matrix((2, 3), wp.float32)()] * len(test_cases), dtype=wp.types.matrix((2, 3), wp.float32)
        )

        wp.launch(
            test_capsule_box_kernel,
            dim=len(test_cases),
            inputs=[
                capsule_positions,
                capsule_axes,
                capsule_radii,
                capsule_half_lengths,
                box_positions,
                box_rotations,
                box_sizes,
                distances,
                contact_positions,
                contact_normals,
            ],
        )
        wp.synchronize()

        distances_np = distances.numpy()

        # Capsule-box behavior: intersecting has penetration, separated has positive distances
        # Case 0 (intersecting): Should have at least one penetrating contact (negative distance)
        has_penetration = any(d < 0.0 for d in distances_np[0] if d != float("inf"))
        self.assertTrue(has_penetration, msg="Intersecting capsule-box should have penetrating contact")

        # Case 1 (separated): All finite distances should be positive (separation)
        finite_distances = [d for d in distances_np[1] if d != float("inf")]
        for dist in finite_distances:
            self.assertGreater(dist, 0.0, msg=f"Separated capsule-box should have positive distance, got {dist}")


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
