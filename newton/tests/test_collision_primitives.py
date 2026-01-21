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
from newton._src.core.types import MAXVAL


def check_normal_direction_sphere_sphere(pos1, pos2, normal, tolerance=1e-5):
    """Check that normal points from sphere 1 toward sphere 2."""
    expected_direction = pos2 - pos1
    expected_direction_norm = np.linalg.norm(expected_direction)
    if expected_direction_norm > tolerance:
        expected_direction = expected_direction / expected_direction_norm
        dot_product = np.dot(normal, expected_direction)
        return dot_product > (1.0 - tolerance)
    return True  # Can't determine direction if centers coincide


def check_contact_position_midpoint(
    contact_pos, normal, penetration_depth, pos1, radius1, pos2, radius2, tolerance=0.05
):
    """Check that contact position is at the midpoint between the two surfaces.

    For sphere-sphere collision:
    - Moving from contact_pos by -penetration_depth/2 along normal should reach surface of sphere 1
    - Moving from contact_pos by +penetration_depth/2 along normal should reach surface of sphere 2
    """
    if penetration_depth >= 0:
        # For separated or just touching cases, position is still at midpoint
        # but we can't validate surface points the same way
        return True

    # Point on surface of geom 0 (sphere 1)
    surface_point_0 = contact_pos - normal * (penetration_depth / 2.0)
    # Distance from this point to sphere 1 center should equal radius1
    dist_to_sphere1 = np.linalg.norm(surface_point_0 - pos1)

    # Point on surface of geom 1 (sphere 2)
    surface_point_1 = contact_pos + normal * (penetration_depth / 2.0)
    # Distance from this point to sphere 2 center should equal radius2
    dist_to_sphere2 = np.linalg.norm(surface_point_1 - pos2)

    return abs(dist_to_sphere1 - radius1) < tolerance and abs(dist_to_sphere2 - radius2) < tolerance


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


def distance_point_to_box(point, box_pos, box_rot, box_size):
    """Calculate distance from a point to a box surface."""
    # Transform point to box local coordinates
    local_point = np.dot(box_rot.T, point - box_pos)

    # Clamp to box bounds
    clamped = np.clip(local_point, -box_size, box_size)

    # Distance from point to closest point on/in box
    return np.linalg.norm(local_point - clamped)


def distance_point_to_cylinder(point, cylinder_pos, cylinder_axis, cylinder_radius, cylinder_half_height):
    """Calculate distance from a point to a cylinder surface."""
    vec = point - cylinder_pos
    x = np.dot(vec, cylinder_axis)

    # Project onto axis and perpendicular component
    a_proj = cylinder_axis * x
    p_proj = vec - a_proj
    p_proj_dist = np.linalg.norm(p_proj)

    # Check if within cylinder height
    if abs(x) <= cylinder_half_height:
        # Side of cylinder
        if p_proj_dist <= cylinder_radius:
            # Inside cylinder - distance to nearest surface
            dist_to_side = abs(cylinder_radius - p_proj_dist)
            dist_to_cap = cylinder_half_height - abs(x)
            return min(dist_to_side, dist_to_cap)
        else:
            # Outside cylinder radially
            return p_proj_dist - cylinder_radius
    else:
        # Beyond cylinder caps
        if p_proj_dist <= cylinder_radius:
            # Above/below cap center
            return abs(x) - cylinder_half_height
        else:
            # Corner region
            cap_offset = cylinder_axis * (np.sign(x) * cylinder_half_height)
            p_proj_normalized = (p_proj / p_proj_dist) * cylinder_radius if p_proj_dist > 1e-6 else np.zeros(3)
            corner_pos = cylinder_pos + cap_offset + p_proj_normalized
            return np.linalg.norm(point - corner_pos)


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
    distances: wp.array(dtype=wp.vec2),
    contact_positions: wp.array(dtype=wp.types.matrix((2, 3), wp.float32)),
    contact_normals: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    dists, positions, normal = geometry.collide_capsule_capsule(
        cap1_positions[tid],
        cap1_axes[tid],
        cap1_radii[tid],
        cap1_half_lengths[tid],
        cap2_positions[tid],
        cap2_axes[tid],
        cap2_radii[tid],
        cap2_half_lengths[tid],
    )
    distances[tid] = dists
    contact_positions[tid] = positions
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
        """Test plane-sphere collision with analytical penetration depth validation.

        Analytical calculation:
        - Distance = (sphere_center - plane_point) · plane_normal - sphere_radius
        - Negative distance indicates penetration
        """
        test_cases = [
            # Plane normal, plane pos, sphere pos, sphere radius, expected distance
            ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0], 1.0, 1.0),  # Above plane, separation = 1.0
            ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.5], 1.0, 0.5),  # Above plane, separation = 0.5
            ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0, 0.0),  # Just touching
            ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.8], 1.0, -0.2),  # Penetration = 0.2
            ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.5], 1.0, -0.5),  # Penetration = 0.5
            ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.2], 1.0, -0.8),  # Penetration = 0.8
            ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], 0.5, 0.5),  # X-axis, separation = 0.5
            ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0], 0.5, 0.0),  # X-axis, touching
            ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.3, 0.0, 0.0], 0.5, -0.2),  # X-axis, penetration = 0.2
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
        positions_np = contact_positions.numpy()

        # Verify expected distances with analytical validation
        for i, expected_dist in enumerate([tc[4] for tc in test_cases]):
            self.assertAlmostEqual(
                distances_np[i],
                expected_dist,
                places=5,
                msg=f"Test case {i}: Expected distance {expected_dist:.4f}, got {distances_np[i]:.4f}",
            )

        # Check that contact position lies between sphere and plane
        for i in range(len(test_cases)):
            if distances_np[i] >= 0:
                # Skip separated cases
                continue

            plane_normal = np.array(test_cases[i][0])
            plane_pos = np.array(test_cases[i][1])
            sphere_pos = np.array(test_cases[i][2])
            sphere_radius = test_cases[i][3]
            contact_pos = positions_np[i]

            # Contact position should be between sphere surface and plane
            # Distance from contact to sphere center should be less than sphere radius
            dist_to_sphere_center = np.linalg.norm(contact_pos - sphere_pos)
            self.assertLess(
                dist_to_sphere_center,
                sphere_radius + 0.01,
                msg=f"Test case {i}: Contact position too far from sphere (dist: {dist_to_sphere_center:.4f})",
            )

            # Contact position should be on the plane side of the sphere center
            # (or at most slightly past the plane)
            dist_contact_to_plane = np.dot(contact_pos - plane_pos, plane_normal)
            dist_sphere_to_plane = np.dot(sphere_pos - plane_pos, plane_normal)
            self.assertLessEqual(
                dist_contact_to_plane,
                dist_sphere_to_plane + 0.01,
                msg=f"Test case {i}: Contact position on wrong side of sphere center",
            )

    def test_sphere_sphere(self):
        """Test sphere-sphere collision with analytical penetration depth validation.

        Analytical calculation:
        - Distance = ||center2 - center1|| - (radius1 + radius2)
        - Negative distance indicates penetration
        """
        test_cases = [
            # pos1, radius1, pos2, radius2, expected_distance
            ([0.0, 0.0, 0.0], 1.0, [3.5, 0.0, 0.0], 1.0, 1.5),  # Separated by 1.5
            ([0.0, 0.0, 0.0], 1.0, [3.0, 0.0, 0.0], 1.0, 1.0),  # Separated by 1.0
            ([0.0, 0.0, 0.0], 1.0, [2.5, 0.0, 0.0], 1.0, 0.5),  # Separated by 0.5
            ([0.0, 0.0, 0.0], 1.0, [2.0, 0.0, 0.0], 1.0, 0.0),  # Exactly touching
            ([0.0, 0.0, 0.0], 1.0, [1.8, 0.0, 0.0], 1.0, -0.2),  # Penetration = 0.2
            ([0.0, 0.0, 0.0], 1.0, [1.5, 0.0, 0.0], 1.0, -0.5),  # Penetration = 0.5
            ([0.0, 0.0, 0.0], 1.0, [1.2, 0.0, 0.0], 1.0, -0.8),  # Penetration = 0.8
            # Different radii
            ([0.0, 0.0, 0.0], 0.5, [2.0, 0.0, 0.0], 1.0, 0.5),  # Separated
            ([0.0, 0.0, 0.0], 0.5, [1.5, 0.0, 0.0], 1.0, 0.0),  # Touching
            ([0.0, 0.0, 0.0], 0.5, [1.2, 0.0, 0.0], 1.0, -0.3),  # Penetration = 0.3
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
        positions_np = contact_positions.numpy()

        # Verify expected distances with analytical validation
        for i, expected_dist in enumerate([tc[4] for tc in test_cases]):
            self.assertAlmostEqual(
                distances_np[i],
                expected_dist,
                places=5,
                msg=f"Test case {i}: Expected distance {expected_dist:.4f}, got {distances_np[i]:.4f}",
            )

        # Check normal vectors are unit length (except for zero distance case)
        for i in range(len(test_cases)):
            if abs(test_cases[i][4]) > 1e-6:  # Skip near-zero distance cases
                normal_length = np.linalg.norm(normals_np[i])
                self.assertAlmostEqual(
                    normal_length, 1.0, places=5, msg=f"Test case {i}: Normal not unit length: {normal_length:.4f}"
                )

        # Check that normal points from geom 0 (sphere 1) into geom 1 (sphere 2)
        for i in range(len(test_cases)):
            pos1 = np.array(test_cases[i][0])
            pos2 = np.array(test_cases[i][2])
            normal = normals_np[i]
            self.assertTrue(
                check_normal_direction_sphere_sphere(pos1, pos2, normal),
                msg=f"Test case {i}: Normal does not point from sphere 1 toward sphere 2",
            )

        # Check that contact position is at midpoint between surfaces
        for i in range(len(test_cases)):
            pos1 = np.array(test_cases[i][0])
            radius1 = test_cases[i][1]
            pos2 = np.array(test_cases[i][2])
            radius2 = test_cases[i][3]
            contact_pos = positions_np[i]
            normal = normals_np[i]
            penetration_depth = distances_np[i]

            self.assertTrue(
                check_contact_position_midpoint(contact_pos, normal, penetration_depth, pos1, radius1, pos2, radius2),
                msg=f"Test case {i}: Contact position is not at midpoint between surfaces",
            )

    def test_sphere_capsule(self):
        """Test sphere-capsule collision with analytical penetration depth validation.

        Capsule: center at origin, axis along Z, radius=0.5, half-length=1.0
        - Cylinder part extends from z=-1.0 to z=1.0
        - Hemisphere caps at top and bottom
        - Total length from cap center to cap center = 2.0
        """
        capsule_center = [0.0, 0.0, 0.0]
        capsule_axis = [0.0, 0.0, 1.0]
        capsule_radius = 0.5
        capsule_half_length = 1.0
        sphere_radius = 0.5

        test_cases = [
            # Sphere approaching capsule cylinder side (from +Y direction)
            ([0.0, 1.5, 0.0], sphere_radius, capsule_center, capsule_axis, capsule_radius, capsule_half_length, 0.5),
            ([0.0, 1.0, 0.0], sphere_radius, capsule_center, capsule_axis, capsule_radius, capsule_half_length, 0.0),
            ([0.0, 0.9, 0.0], sphere_radius, capsule_center, capsule_axis, capsule_radius, capsule_half_length, -0.1),
            ([0.0, 0.8, 0.0], sphere_radius, capsule_center, capsule_axis, capsule_radius, capsule_half_length, -0.2),
            # Sphere approaching capsule cap (from +Z direction, aligned with axis)
            ([0.0, 0.0, 2.5], sphere_radius, capsule_center, capsule_axis, capsule_radius, capsule_half_length, 0.5),
            ([0.0, 0.0, 2.0], sphere_radius, capsule_center, capsule_axis, capsule_radius, capsule_half_length, 0.0),
            ([0.0, 0.0, 1.9], sphere_radius, capsule_center, capsule_axis, capsule_radius, capsule_half_length, -0.1),
            ([0.0, 0.0, 1.8], sphere_radius, capsule_center, capsule_axis, capsule_radius, capsule_half_length, -0.2),
        ]

        sphere_positions = wp.array([wp.vec3(tc[0][0], tc[0][1], tc[0][2]) for tc in test_cases], dtype=wp.vec3)
        sphere_radii = wp.array([tc[1] for tc in test_cases], dtype=float)
        capsule_positions = wp.array([wp.vec3(tc[2][0], tc[2][1], tc[2][2]) for tc in test_cases], dtype=wp.vec3)
        capsule_axes = wp.array([wp.vec3(tc[3][0], tc[3][1], tc[3][2]) for tc in test_cases], dtype=wp.vec3)
        capsule_radii = wp.array([tc[4] for tc in test_cases], dtype=float)
        capsule_half_lengths = wp.array([tc[5] for tc in test_cases], dtype=float)
        expected_distances = [tc[6] for tc in test_cases]
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
        normals_np = contact_normals.numpy()
        positions_np = contact_positions.numpy()

        # Verify expected distances with analytical validation
        tolerance = 0.01  # Small tolerance for numerical precision
        for i, expected_dist in enumerate(expected_distances):
            self.assertAlmostEqual(
                distances_np[i],
                expected_dist,
                delta=tolerance,
                msg=f"Test case {i}: Expected distance {expected_dist:.4f}, got {distances_np[i]:.4f}",
            )

        # Check that normal points from geom 0 (sphere) into geom 1 (capsule)
        # and contact position is at midpoint between surfaces
        for i in range(len(test_cases)):
            if distances_np[i] >= 0:
                # Skip separated cases for now
                continue

            sphere_pos = np.array(test_cases[i][0])
            sphere_radius = test_cases[i][1]
            capsule_pos = np.array(test_cases[i][2])
            capsule_axis = np.array(test_cases[i][3])
            capsule_radius = test_cases[i][4]
            capsule_half_length = test_cases[i][5]

            contact_pos = positions_np[i]
            normal = normals_np[i]
            penetration_depth = distances_np[i]

            # Check midpoint property: going half penetration depth in each direction should land on surfaces
            surface_point_0 = contact_pos - normal * (penetration_depth / 2.0)
            surface_point_1 = contact_pos + normal * (penetration_depth / 2.0)

            # Distance from surface_point_0 to sphere surface should be small
            dist_to_sphere = abs(np.linalg.norm(surface_point_0 - sphere_pos) - sphere_radius)

            # Distance from surface_point_1 to capsule surface should be small
            dist_to_capsule = distance_point_to_capsule(
                surface_point_1, capsule_pos, capsule_axis, capsule_radius, capsule_half_length
            )

            self.assertLess(
                dist_to_sphere,
                0.05,
                msg=f"Test case {i}: Point at -penetration_depth/2 not on sphere surface (error: {dist_to_sphere:.4f})",
            )
            self.assertLess(
                dist_to_capsule,
                0.05,
                msg=f"Test case {i}: Point at +penetration_depth/2 not on capsule surface (error: {dist_to_capsule:.4f})",
            )

    def test_capsule_capsule(self):
        """Test capsule-capsule collision with analytical penetration depth validation.

        Test parallel capsules moving closer together.
        Capsule 1: center at origin, axis along X, radius=0.5, half-length=1.0
        Capsule 2: center at various Y positions, axis along X, radius=0.5, half-length=1.0
        Distance between parallel capsules = Y_separation - (radius1 + radius2)
        """
        cap1_pos = [0.0, 0.0, 0.0]
        cap1_axis = [1.0, 0.0, 0.0]
        cap1_radius = 0.5
        cap1_half_length = 1.0
        cap2_axis = [1.0, 0.0, 0.0]
        cap2_radius = 0.5
        cap2_half_length = 1.0

        test_cases = [
            # Parallel capsules at various Y separations
            (
                cap1_pos,
                cap1_axis,
                cap1_radius,
                cap1_half_length,
                [0.0, 2.0, 0.0],
                cap2_axis,
                cap2_radius,
                cap2_half_length,
                1.0,
            ),
            (
                cap1_pos,
                cap1_axis,
                cap1_radius,
                cap1_half_length,
                [0.0, 1.5, 0.0],
                cap2_axis,
                cap2_radius,
                cap2_half_length,
                0.5,
            ),
            (
                cap1_pos,
                cap1_axis,
                cap1_radius,
                cap1_half_length,
                [0.0, 1.0, 0.0],
                cap2_axis,
                cap2_radius,
                cap2_half_length,
                0.0,
            ),
            (
                cap1_pos,
                cap1_axis,
                cap1_radius,
                cap1_half_length,
                [0.0, 0.9, 0.0],
                cap2_axis,
                cap2_radius,
                cap2_half_length,
                -0.1,
            ),
            (
                cap1_pos,
                cap1_axis,
                cap1_radius,
                cap1_half_length,
                [0.0, 0.8, 0.0],
                cap2_axis,
                cap2_radius,
                cap2_half_length,
                -0.2,
            ),
            (
                cap1_pos,
                cap1_axis,
                cap1_radius,
                cap1_half_length,
                [0.0, 0.6, 0.0],
                cap2_axis,
                cap2_radius,
                cap2_half_length,
                -0.4,
            ),
            # Perpendicular capsules (intersecting at center)
            (
                cap1_pos,
                cap1_axis,
                cap1_radius,
                cap1_half_length,
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                cap2_radius,
                cap2_half_length,
                -1.0,
            ),
        ]

        cap1_positions = wp.array([wp.vec3(tc[0][0], tc[0][1], tc[0][2]) for tc in test_cases], dtype=wp.vec3)
        cap1_axes = wp.array([wp.vec3(tc[1][0], tc[1][1], tc[1][2]) for tc in test_cases], dtype=wp.vec3)
        cap1_radii = wp.array([tc[2] for tc in test_cases], dtype=float)
        cap1_half_lengths = wp.array([tc[3] for tc in test_cases], dtype=float)
        cap2_positions = wp.array([wp.vec3(tc[4][0], tc[4][1], tc[4][2]) for tc in test_cases], dtype=wp.vec3)
        cap2_axes = wp.array([wp.vec3(tc[5][0], tc[5][1], tc[5][2]) for tc in test_cases], dtype=wp.vec3)
        cap2_radii = wp.array([tc[6] for tc in test_cases], dtype=float)
        cap2_half_lengths = wp.array([tc[7] for tc in test_cases], dtype=float)
        expected_distances = [tc[8] for tc in test_cases]
        distances = wp.zeros(len(test_cases), dtype=wp.vec2)
        contact_positions = wp.zeros((len(test_cases),), dtype=wp.types.matrix((2, 3), wp.float32))
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
        normals_np = contact_normals.numpy()
        positions_np = contact_positions.numpy()

        # Verify expected distances with analytical validation (use first contact)
        tolerance = 0.01  # Small tolerance for numerical precision
        for i, expected_dist in enumerate(expected_distances):
            # Use first contact distance (index 0)
            self.assertAlmostEqual(
                distances_np[i][0],
                expected_dist,
                delta=tolerance,
                msg=f"Test case {i}: Expected distance {expected_dist:.4f}, got {distances_np[i][0]:.4f}",
            )

        # Check that contact position is at midpoint between surfaces
        for i in range(len(test_cases)):
            # Use first contact distance
            if distances_np[i][0] >= 0:
                # Skip separated cases for now
                continue

            cap1_pos = np.array(test_cases[i][0])
            cap1_axis = np.array(test_cases[i][1])
            cap1_radius = test_cases[i][2]
            cap1_half_length = test_cases[i][3]
            cap2_pos = np.array(test_cases[i][4])
            cap2_axis = np.array(test_cases[i][5])
            cap2_radius = test_cases[i][6]
            cap2_half_length = test_cases[i][7]

            # Skip perpendicular/complex cases where simple midpoint validation doesn't apply
            axis_alignment = abs(np.dot(cap1_axis, cap2_axis))
            if axis_alignment < 0.9:  # Not parallel enough
                continue

            # Use first contact position (row 0 of the 2x3 matrix)
            contact_pos = positions_np[i][0]
            normal = normals_np[i]
            penetration_depth = distances_np[i][0]

            # Check midpoint property: going half penetration depth in each direction should land on surfaces
            surface_point_0 = contact_pos - normal * (penetration_depth / 2.0)
            surface_point_1 = contact_pos + normal * (penetration_depth / 2.0)

            # Distance from surface_point_0 to capsule 1 surface should be small
            dist_to_cap1 = distance_point_to_capsule(
                surface_point_0, cap1_pos, cap1_axis, cap1_radius, cap1_half_length
            )

            # Distance from surface_point_1 to capsule 2 surface should be small
            dist_to_cap2 = distance_point_to_capsule(
                surface_point_1, cap2_pos, cap2_axis, cap2_radius, cap2_half_length
            )

            self.assertLess(
                dist_to_cap1,
                0.05,
                msg=f"Test case {i}: Point at -penetration_depth/2 not on capsule 1 surface (error: {dist_to_cap1:.4f})",
            )
            self.assertLess(
                dist_to_cap2,
                0.05,
                msg=f"Test case {i}: Point at +penetration_depth/2 not on capsule 2 surface (error: {dist_to_cap2:.4f})",
            )

    def test_plane_ellipsoid(self):
        """Test plane-ellipsoid collision with analytical penetration depth validation.

        Plane at z=0, normal pointing up (+Z)
        Ellipsoid: center at various Z positions, half-axes=[1.0, 1.0, 1.5]
        - Bottom of ellipsoid is at center_z - z_axis_size = center_z - 1.5
        """
        # Identity rotation matrix
        identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        plane_normal = [0.0, 0.0, 1.0]
        plane_pos = [0.0, 0.0, 0.0]
        ellipsoid_size = [1.0, 1.0, 1.5]  # Half-axes (x, y, z)

        test_cases = [
            # Ellipsoid above plane (bottom at z > 0)
            (plane_normal, plane_pos, [0.0, 0.0, 2.5], identity, ellipsoid_size, 1.0),  # bottom at z=1.0
            (plane_normal, plane_pos, [0.0, 0.0, 2.0], identity, ellipsoid_size, 0.5),  # bottom at z=0.5
            (plane_normal, plane_pos, [0.0, 0.0, 1.5], identity, ellipsoid_size, 0.0),  # just touching
            # Ellipsoid penetrating plane
            (plane_normal, plane_pos, [0.0, 0.0, 1.4], identity, ellipsoid_size, -0.1),  # penetration = 0.1
            (plane_normal, plane_pos, [0.0, 0.0, 1.3], identity, ellipsoid_size, -0.2),  # penetration = 0.2
            (plane_normal, plane_pos, [0.0, 0.0, 1.0], identity, ellipsoid_size, -0.5),  # penetration = 0.5
        ]

        plane_normals = wp.array([wp.vec3(tc[0][0], tc[0][1], tc[0][2]) for tc in test_cases], dtype=wp.vec3)
        plane_positions = wp.array([wp.vec3(tc[1][0], tc[1][1], tc[1][2]) for tc in test_cases], dtype=wp.vec3)
        ellipsoid_positions = wp.array([wp.vec3(tc[2][0], tc[2][1], tc[2][2]) for tc in test_cases], dtype=wp.vec3)
        ellipsoid_rotations = wp.array([tc[3] for tc in test_cases], dtype=wp.mat33)
        ellipsoid_sizes = wp.array([wp.vec3(tc[4][0], tc[4][1], tc[4][2]) for tc in test_cases], dtype=wp.vec3)
        expected_distances = [tc[5] for tc in test_cases]
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
        normals_np = contact_normals.numpy()

        # Verify expected distances with analytical validation
        tolerance = 0.01
        for i, expected_dist in enumerate(expected_distances):
            self.assertAlmostEqual(
                distances_np[i],
                expected_dist,
                delta=tolerance,
                msg=f"Test case {i}: Expected distance {expected_dist:.4f}, got {distances_np[i]:.4f}",
            )

        # Check that normal points in correct direction (plane normal direction)
        for i in range(len(test_cases)):
            plane_normal = np.array(test_cases[i][0])
            contact_normal = normals_np[i]
            # Normal should match plane normal (pointing from plane into ellipsoid)
            dot_product = np.dot(plane_normal, contact_normal)
            self.assertGreater(
                dot_product,
                0.99,
                msg=f"Test case {i}: Contact normal doesn't match plane normal (dot product: {dot_product:.4f})",
            )

    def test_sphere_cylinder(self):
        """Test sphere-cylinder collision with analytical penetration depth validation.

        Cylinder: center at origin, axis along Z, radius=1.0, half-height=1.0
        - Side surface at radial distance 1.0 from Z-axis
        - Top cap at z=1.0, bottom cap at z=-1.0
        """
        cylinder_center = [0.0, 0.0, 0.0]
        cylinder_axis = [0.0, 0.0, 1.0]
        cylinder_radius = 1.0
        cylinder_half_height = 1.0
        sphere_radius = 0.5

        test_cases = [
            # Sphere approaching cylinder side (from +X direction)
            (
                [2.0, 0.0, 0.0],
                sphere_radius,
                cylinder_center,
                cylinder_axis,
                cylinder_radius,
                cylinder_half_height,
                0.5,
            ),
            (
                [1.5, 0.0, 0.0],
                sphere_radius,
                cylinder_center,
                cylinder_axis,
                cylinder_radius,
                cylinder_half_height,
                0.0,
            ),
            (
                [1.4, 0.0, 0.0],
                sphere_radius,
                cylinder_center,
                cylinder_axis,
                cylinder_radius,
                cylinder_half_height,
                -0.1,
            ),
            (
                [1.3, 0.0, 0.0],
                sphere_radius,
                cylinder_center,
                cylinder_axis,
                cylinder_radius,
                cylinder_half_height,
                -0.2,
            ),
            # Sphere approaching cylinder top cap (from +Z direction)
            (
                [0.0, 0.0, 2.0],
                sphere_radius,
                cylinder_center,
                cylinder_axis,
                cylinder_radius,
                cylinder_half_height,
                0.5,
            ),
            (
                [0.0, 0.0, 1.5],
                sphere_radius,
                cylinder_center,
                cylinder_axis,
                cylinder_radius,
                cylinder_half_height,
                0.0,
            ),
            (
                [0.0, 0.0, 1.4],
                sphere_radius,
                cylinder_center,
                cylinder_axis,
                cylinder_radius,
                cylinder_half_height,
                -0.1,
            ),
            (
                [0.0, 0.0, 1.3],
                sphere_radius,
                cylinder_center,
                cylinder_axis,
                cylinder_radius,
                cylinder_half_height,
                -0.2,
            ),
        ]

        sphere_positions = wp.array([wp.vec3(tc[0][0], tc[0][1], tc[0][2]) for tc in test_cases], dtype=wp.vec3)
        sphere_radii = wp.array([tc[1] for tc in test_cases], dtype=float)
        cylinder_positions = wp.array([wp.vec3(tc[2][0], tc[2][1], tc[2][2]) for tc in test_cases], dtype=wp.vec3)
        cylinder_axes = wp.array([wp.vec3(tc[3][0], tc[3][1], tc[3][2]) for tc in test_cases], dtype=wp.vec3)
        cylinder_radii = wp.array([tc[4] for tc in test_cases], dtype=float)
        cylinder_half_heights = wp.array([tc[5] for tc in test_cases], dtype=float)
        expected_distances = [tc[6] for tc in test_cases]
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
        normals_np = contact_normals.numpy()
        positions_np = contact_positions.numpy()

        # Verify expected distances with analytical validation
        tolerance = 0.01  # Small tolerance for numerical precision
        for i, expected_dist in enumerate(expected_distances):
            self.assertAlmostEqual(
                distances_np[i],
                expected_dist,
                delta=tolerance,
                msg=f"Test case {i}: Expected distance {expected_dist:.4f}, got {distances_np[i]:.4f}",
            )

        # Check that contact position is at midpoint between surfaces
        for i in range(len(test_cases)):
            if distances_np[i] >= 0:
                # Skip separated cases for now
                continue

            sphere_pos = np.array(test_cases[i][0])
            sphere_radius = test_cases[i][1]
            cylinder_pos = np.array(test_cases[i][2])
            cylinder_axis = np.array(test_cases[i][3])
            cylinder_radius = test_cases[i][4]
            cylinder_half_height = test_cases[i][5]

            contact_pos = positions_np[i]
            normal = normals_np[i]
            penetration_depth = distances_np[i]

            # Check midpoint property: going half penetration depth in each direction should land on surfaces
            surface_point_0 = contact_pos - normal * (penetration_depth / 2.0)
            surface_point_1 = contact_pos + normal * (penetration_depth / 2.0)

            # Distance from surface_point_0 to sphere surface should be small
            dist_to_sphere = abs(np.linalg.norm(surface_point_0 - sphere_pos) - sphere_radius)

            # Distance from surface_point_1 to cylinder surface should be small
            dist_to_cylinder = distance_point_to_cylinder(
                surface_point_1, cylinder_pos, cylinder_axis, cylinder_radius, cylinder_half_height
            )

            self.assertLess(
                dist_to_sphere,
                0.05,
                msg=f"Test case {i}: Point at -penetration_depth/2 not on sphere surface (error: {dist_to_sphere:.4f})",
            )
            self.assertLess(
                dist_to_cylinder,
                0.05,
                msg=f"Test case {i}: Point at +penetration_depth/2 not on cylinder surface (error: {dist_to_cylinder:.4f})",
            )

    def test_sphere_box(self):
        """Test sphere-box collision with analytical penetration depth validation.

        For sphere approaching box face along normal:
        - Distance = (sphere_center_to_face) - sphere_radius
        - Box center at origin, half-extents = 1.0, so face at x=1.0
        - Sphere at x position with radius 0.5
        """
        # Identity rotation matrix
        identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        box_size = [1.0, 1.0, 1.0]
        sphere_radius = 0.5

        test_cases = [
            # Sphere approaching box face from +X direction
            # Box face at x=1.0, sphere at various x positions
            ([2.5, 0.0, 0.0], sphere_radius, [0.0, 0.0, 0.0], identity, box_size, 1.0),  # Separated by 1.0
            ([2.0, 0.0, 0.0], sphere_radius, [0.0, 0.0, 0.0], identity, box_size, 0.5),  # Separated by 0.5
            ([1.5, 0.0, 0.0], sphere_radius, [0.0, 0.0, 0.0], identity, box_size, 0.0),  # Just touching
            ([1.4, 0.0, 0.0], sphere_radius, [0.0, 0.0, 0.0], identity, box_size, -0.1),  # Penetration = 0.1
            ([1.3, 0.0, 0.0], sphere_radius, [0.0, 0.0, 0.0], identity, box_size, -0.2),  # Penetration = 0.2
            ([1.2, 0.0, 0.0], sphere_radius, [0.0, 0.0, 0.0], identity, box_size, -0.3),  # Penetration = 0.3
            # Sphere approaching from +Z direction
            ([0.0, 0.0, 2.0], sphere_radius, [0.0, 0.0, 0.0], identity, box_size, 0.5),  # Separated by 0.5
            ([0.0, 0.0, 1.5], sphere_radius, [0.0, 0.0, 0.0], identity, box_size, 0.0),  # Just touching
            ([0.0, 0.0, 1.3], sphere_radius, [0.0, 0.0, 0.0], identity, box_size, -0.2),  # Penetration = 0.2
            # Sphere center inside box
            ([0.0, 0.0, 0.4], 0.3, [0.0, 0.0, 0.0], identity, box_size, -0.9),  # Sphere center inside
        ]

        sphere_positions = wp.array([wp.vec3(tc[0][0], tc[0][1], tc[0][2]) for tc in test_cases], dtype=wp.vec3)
        sphere_radii = wp.array([tc[1] for tc in test_cases], dtype=float)
        box_positions = wp.array([wp.vec3(tc[2][0], tc[2][1], tc[2][2]) for tc in test_cases], dtype=wp.vec3)
        box_rotations = wp.array([tc[3] for tc in test_cases], dtype=wp.mat33)
        box_sizes = wp.array([wp.vec3(tc[4][0], tc[4][1], tc[4][2]) for tc in test_cases], dtype=wp.vec3)
        expected_distances = [tc[5] for tc in test_cases]
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
        normals_np = contact_normals.numpy()
        positions_np = contact_positions.numpy()

        # Verify expected distances with analytical validation
        tolerance = 0.01  # Small tolerance for numerical precision
        for i, expected_dist in enumerate(expected_distances):
            self.assertAlmostEqual(
                distances_np[i],
                expected_dist,
                delta=tolerance,
                msg=f"Test case {i}: Expected distance {expected_dist:.4f}, got {distances_np[i]:.4f}",
            )

        # Check that normal points from geom 0 (sphere) into geom 1 (box)
        # and contact position is at midpoint between surfaces
        for i in range(len(test_cases)):
            if distances_np[i] >= 0:
                # Skip separated cases for now
                continue

            sphere_pos = np.array(test_cases[i][0])
            sphere_radius = test_cases[i][1]
            box_pos = np.array(test_cases[i][2])
            # Convert wp.mat33 to numpy array
            box_rot_mat = test_cases[i][3]
            box_rot = np.array(
                [
                    [box_rot_mat[0, 0], box_rot_mat[0, 1], box_rot_mat[0, 2]],
                    [box_rot_mat[1, 0], box_rot_mat[1, 1], box_rot_mat[1, 2]],
                    [box_rot_mat[2, 0], box_rot_mat[2, 1], box_rot_mat[2, 2]],
                ]
            )
            box_size = np.array(test_cases[i][4])

            contact_pos = positions_np[i]
            normal = normals_np[i]
            penetration_depth = distances_np[i]

            # Check midpoint property: going half penetration depth in each direction should land on surfaces
            surface_point_0 = contact_pos - normal * (penetration_depth / 2.0)
            surface_point_1 = contact_pos + normal * (penetration_depth / 2.0)

            # Distance from surface_point_0 to sphere surface should be small
            dist_to_sphere = abs(np.linalg.norm(surface_point_0 - sphere_pos) - sphere_radius)

            # Distance from surface_point_1 to box surface should be small
            dist_to_box = distance_point_to_box(surface_point_1, box_pos, box_rot, box_size)

            self.assertLess(
                dist_to_sphere,
                0.05,
                msg=f"Test case {i}: Point at -penetration_depth/2 not on sphere surface (error: {dist_to_sphere:.4f})",
            )
            self.assertLess(
                dist_to_box,
                0.05,
                msg=f"Test case {i}: Point at +penetration_depth/2 not on box surface (error: {dist_to_box:.4f})",
            )

    def test_plane_capsule(self):
        """Test plane-capsule collision with analytical penetration depth validation.

        Plane at z=0, normal pointing up (+Z)
        Capsule: axis horizontal (along X), radius=0.5, half-length=1.0
        - Capsule endpoints are at center ± (half_length * axis)
        - Lowest point is center_z - radius
        """
        plane_normal = [0.0, 0.0, 1.0]
        plane_pos = [0.0, 0.0, 0.0]
        capsule_axis = [1.0, 0.0, 0.0]  # Horizontal capsule
        capsule_radius = 0.5
        capsule_half_length = 1.0

        test_cases = [
            # Capsule above plane at various heights
            (plane_normal, plane_pos, [0.0, 0.0, 2.0], capsule_axis, capsule_radius, capsule_half_length, 1.5),
            (plane_normal, plane_pos, [0.0, 0.0, 1.5], capsule_axis, capsule_radius, capsule_half_length, 1.0),
            (plane_normal, plane_pos, [0.0, 0.0, 1.0], capsule_axis, capsule_radius, capsule_half_length, 0.5),
            (plane_normal, plane_pos, [0.0, 0.0, 0.5], capsule_axis, capsule_radius, capsule_half_length, 0.0),
            (plane_normal, plane_pos, [0.0, 0.0, 0.4], capsule_axis, capsule_radius, capsule_half_length, -0.1),
            (plane_normal, plane_pos, [0.0, 0.0, 0.3], capsule_axis, capsule_radius, capsule_half_length, -0.2),
            (plane_normal, plane_pos, [0.0, 0.0, 0.2], capsule_axis, capsule_radius, capsule_half_length, -0.3),
        ]

        plane_normals = wp.array([wp.vec3(tc[0][0], tc[0][1], tc[0][2]) for tc in test_cases], dtype=wp.vec3)
        plane_positions = wp.array([wp.vec3(tc[1][0], tc[1][1], tc[1][2]) for tc in test_cases], dtype=wp.vec3)
        capsule_positions = wp.array([wp.vec3(tc[2][0], tc[2][1], tc[2][2]) for tc in test_cases], dtype=wp.vec3)
        capsule_axes = wp.array([wp.vec3(tc[3][0], tc[3][1], tc[3][2]) for tc in test_cases], dtype=wp.vec3)
        capsule_radii = wp.array([tc[4] for tc in test_cases], dtype=float)
        capsule_half_lengths = wp.array([tc[5] for tc in test_cases], dtype=float)
        expected_distances = [tc[6] for tc in test_cases]
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
        frames_np = contact_frames.numpy()

        # Verify expected distances with analytical validation
        # Capsule generates 2 contacts (one at each end)
        tolerance = 0.01
        for i, expected_dist in enumerate(expected_distances):
            # Both contacts should have approximately the same distance for horizontal capsule
            for j in range(2):
                if distances_np[i][j] < MAXVAL * 0.99:
                    self.assertAlmostEqual(
                        distances_np[i][j],
                        expected_dist,
                        delta=tolerance,
                        msg=f"Test case {i}, contact {j}: Expected distance {expected_dist:.4f}, got {distances_np[i][j]:.4f}",
                    )

        # Check that contact frame normal (first row) matches plane normal
        for i in range(len(test_cases)):
            plane_normal = np.array(test_cases[i][0])
            frame = frames_np[i]
            # Extract first row of contact frame (the normal)
            contact_normal = np.array([frame[0, 0], frame[0, 1], frame[0, 2]])
            dot_product = np.dot(plane_normal, contact_normal)
            self.assertGreater(
                dot_product,
                0.99,
                msg=f"Test case {i}: Contact frame normal doesn't match plane normal (dot product: {dot_product:.4f})",
            )

    def test_plane_box(self):
        """Test plane-box collision with analytical penetration depth validation.

        Plane at z=0, normal pointing up (+Z)
        Box: center at various Z positions, half-extents=[1.0, 1.0, 1.0]
        - Box bottom face is at center_z - 1.0
        - Penetration depth = -bottom_face_z when penetrating
        """
        # Identity rotation matrix
        identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        plane_normal = [0.0, 0.0, 1.0]
        plane_pos = [0.0, 0.0, 0.0]
        box_size = [1.0, 1.0, 1.0]

        test_cases = [
            # Box above plane (no contacts expected)
            (plane_normal, plane_pos, [0.0, 0.0, 2.0], identity, box_size, 0, 1.0),
            # Box just touching plane (4 corner contacts)
            (plane_normal, plane_pos, [0.0, 0.0, 1.0], identity, box_size, 4, 0.0),
            # Box penetrating plane slightly
            (plane_normal, plane_pos, [0.0, 0.0, 0.9], identity, box_size, 4, -0.1),
            (plane_normal, plane_pos, [0.0, 0.0, 0.8], identity, box_size, 4, -0.2),
            (plane_normal, plane_pos, [0.0, 0.0, 0.7], identity, box_size, 4, -0.3),
            (plane_normal, plane_pos, [0.0, 0.0, 0.5], identity, box_size, 4, -0.5),
        ]

        plane_normals = wp.array([wp.vec3(tc[0][0], tc[0][1], tc[0][2]) for tc in test_cases], dtype=wp.vec3)
        plane_positions = wp.array([wp.vec3(tc[1][0], tc[1][1], tc[1][2]) for tc in test_cases], dtype=wp.vec3)
        box_positions = wp.array([wp.vec3(tc[2][0], tc[2][1], tc[2][2]) for tc in test_cases], dtype=wp.vec3)
        box_rotations = wp.array([tc[3] for tc in test_cases], dtype=wp.mat33)
        box_sizes = wp.array([wp.vec3(tc[4][0], tc[4][1], tc[4][2]) for tc in test_cases], dtype=wp.vec3)
        expected_contact_counts = [tc[5] for tc in test_cases]
        expected_distances = [tc[6] for tc in test_cases]
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
        normals_np = contact_normals.numpy()

        # Verify contact counts and distances
        tolerance = 0.01
        for i in range(len(test_cases)):
            valid_contacts = sum(1 for d in distances_np[i] if d < MAXVAL * 0.99)
            expected_count = expected_contact_counts[i]
            expected_dist = expected_distances[i]

            self.assertEqual(
                valid_contacts,
                expected_count,
                msg=f"Test case {i}: Expected {expected_count} contacts but got {valid_contacts}",
            )

            # Check that all valid contact distances match expected value
            if valid_contacts > 0:
                for j in range(4):
                    if distances_np[i][j] < MAXVAL * 0.99:
                        self.assertAlmostEqual(
                            distances_np[i][j],
                            expected_dist,
                            delta=tolerance,
                            msg=f"Test case {i}, contact {j}: Expected distance {expected_dist:.4f}, got {distances_np[i][j]:.4f}",
                        )

        # Check that contact normal matches plane normal for cases with contacts
        for i in range(len(test_cases)):
            if expected_contact_counts[i] == 0:
                continue

            plane_normal = np.array(test_cases[i][0])
            contact_normal = normals_np[i]
            dot_product = np.dot(plane_normal, contact_normal)
            self.assertGreater(
                dot_product,
                0.99,
                msg=f"Test case {i}: Contact normal doesn't match plane normal (dot product: {dot_product:.4f})",
            )

    def test_plane_cylinder(self):
        """Test plane-cylinder collision with analytical penetration depth validation.

        Plane at z=0, normal pointing up (+Z)
        Cylinder: axis along Z, radius=1.0, half-height=1.0
        - Cylinder bottom face is at center_z - half_height
        - When axis is vertical, bottom edge contacts form a circle
        """
        plane_normal = [0.0, 0.0, 1.0]
        plane_pos = [0.0, 0.0, 0.0]
        cylinder_axis = [0.0, 0.0, 1.0]  # Vertical cylinder
        cylinder_radius = 1.0
        cylinder_half_height = 1.0

        test_cases = [
            # Cylinder above plane (separated, min distance should be positive)
            (plane_normal, plane_pos, [0.0, 0.0, 5.0], cylinder_axis, cylinder_radius, cylinder_half_height, 4.0),
            # Cylinder just touching plane
            (plane_normal, plane_pos, [0.0, 0.0, 1.0], cylinder_axis, cylinder_radius, cylinder_half_height, 0.0),
            # Cylinder penetrating plane
            (plane_normal, plane_pos, [0.0, 0.0, 0.9], cylinder_axis, cylinder_radius, cylinder_half_height, -0.1),
            (plane_normal, plane_pos, [0.0, 0.0, 0.8], cylinder_axis, cylinder_radius, cylinder_half_height, -0.2),
            (plane_normal, plane_pos, [0.0, 0.0, 0.7], cylinder_axis, cylinder_radius, cylinder_half_height, -0.3),
            (plane_normal, plane_pos, [0.0, 0.0, 0.5], cylinder_axis, cylinder_radius, cylinder_half_height, -0.5),
        ]

        plane_normals = wp.array([wp.vec3(tc[0][0], tc[0][1], tc[0][2]) for tc in test_cases], dtype=wp.vec3)
        plane_positions = wp.array([wp.vec3(tc[1][0], tc[1][1], tc[1][2]) for tc in test_cases], dtype=wp.vec3)
        cylinder_centers = wp.array([wp.vec3(tc[2][0], tc[2][1], tc[2][2]) for tc in test_cases], dtype=wp.vec3)
        cylinder_axes = wp.array([wp.vec3(tc[3][0], tc[3][1], tc[3][2]) for tc in test_cases], dtype=wp.vec3)
        cylinder_radii = wp.array([tc[4] for tc in test_cases], dtype=float)
        cylinder_half_heights = wp.array([tc[5] for tc in test_cases], dtype=float)
        expected_distances = [tc[6] for tc in test_cases]
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
        normals_np = contact_normals.numpy()

        # Verify minimum distances (closest point between cylinder and plane)
        tolerance = 0.01
        for i in range(len(test_cases)):
            expected_dist = expected_distances[i]

            # Get minimum distance from all contact points
            valid_dists = [d for d in distances_np[i] if np.isfinite(d)]

            # For separated cases (positive distance), contacts may or may not be returned
            # For touching/penetrating cases (distance <= 0), require contacts
            if expected_dist <= 0.0:
                self.assertGreater(
                    len(valid_dists),
                    0,
                    msg=f"Test case {i}: Expected contacts for touching/penetrating case but got none",
                )

            if len(valid_dists) > 0:
                min_dist = min(valid_dists)
                self.assertAlmostEqual(
                    min_dist,
                    expected_dist,
                    delta=tolerance,
                    msg=f"Test case {i}: Expected min distance {expected_dist:.4f}, got {min_dist:.4f}",
                )

        # Check that contact normal matches plane normal for cases with contacts
        for i in range(len(test_cases)):
            expected_dist = expected_distances[i]
            if expected_dist > 0.0:  # Skip separated cases
                continue

            plane_normal = np.array(test_cases[i][0])
            contact_normal = normals_np[i]
            dot_product = np.dot(plane_normal, contact_normal)
            self.assertGreater(
                dot_product,
                0.99,
                msg=f"Test case {i}: Contact normal doesn't match plane normal (dot product: {dot_product:.4f})",
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
        normals_np = contact_normals.numpy()

        # Count valid contacts for each test case
        for i in range(len(test_cases)):
            valid_contacts = sum(1 for j in range(8) if distances_np[i][j] < MAXVAL * 0.99)

            if i == 0:  # Separated boxes
                self.assertEqual(valid_contacts, 0, msg="Separated boxes should have no contacts")
            elif i == 1:  # Overlapping boxes
                self.assertGreater(valid_contacts, 0, msg="Overlapping boxes should have contacts")

        # Check that contact normals are unit length and point from box1 into box2
        for i in range(len(test_cases)):
            for j in range(8):
                if distances_np[i][j] >= MAXVAL * 0.99:
                    continue

                # Check normal is unit length
                normal = normals_np[i][j]
                normal_length = np.linalg.norm(normal)
                self.assertAlmostEqual(
                    normal_length,
                    1.0,
                    delta=0.01,
                    msg=f"Test case {i}, contact {j}: Normal not unit length: {normal_length:.4f}",
                )

                # For overlapping boxes, normal should point from box1 toward box2
                if i == 1:
                    box1_pos = np.array(test_cases[i][0])
                    box2_pos = np.array(test_cases[i][3])
                    direction = box2_pos - box1_pos
                    direction = direction / np.linalg.norm(direction)
                    # Normal should have positive component in direction from box1 to box2
                    dot_product = np.dot(normal, direction)
                    self.assertGreater(
                        dot_product,
                        -0.1,  # Allow some tolerance for edge cases
                        msg=f"Test case {i}, contact {j}: Normal points away from box2 (dot: {dot_product:.4f})",
                    )

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
            ),  # Margin > gap, contact
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
        normals_np = contact_normals.numpy()

        # Verify expected contact behavior for each test case
        for i in range(len(test_cases)):
            valid_contacts = sum(1 for j in range(8) if distances_np[i][j] < MAXVAL * 0.99)
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

        # Check that contact normals are unit length for cases with contacts
        for i in range(len(test_cases)):
            expect_contacts = test_cases[i][7]
            if not expect_contacts:
                continue

            for j in range(8):
                if distances_np[i][j] >= MAXVAL * 0.99:
                    continue

                normal = normals_np[i][j]
                normal_length = np.linalg.norm(normal)
                self.assertAlmostEqual(
                    normal_length,
                    1.0,
                    delta=0.01,
                    msg=f"Test case {i}, contact {j}: Normal not unit length: {normal_length:.4f}",
                )

    def test_capsule_box(self):
        """Test capsule-box collision with analytical penetration depth validation.

        Capsule: axis along Z, radius=0.5, half-length=1.0
        Box: center at origin, half-extents=[1.0, 1.0, 1.0]
        - Box top face at z=1.0
        - Capsule bottom hemisphere center at capsule_z - half_length
        """
        # Identity rotation matrix
        identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        box_center = [0.0, 0.0, 0.0]
        box_size = [1.0, 1.0, 1.0]
        capsule_axis = [0.0, 0.0, 1.0]
        capsule_radius = 0.5
        capsule_half_length = 1.0

        test_cases = [
            # Capsule approaching box top face from above
            # Capsule at z=3.0, bottom hemisphere center at z=2.0, bottom surface at z=1.5
            ([0.0, 0.0, 3.0], capsule_axis, capsule_radius, capsule_half_length, box_center, identity, box_size, 0.5),
            # Capsule at z=2.5, bottom hemisphere center at z=1.5, bottom surface at z=1.0 (touching)
            ([0.0, 0.0, 2.5], capsule_axis, capsule_radius, capsule_half_length, box_center, identity, box_size, 0.0),
            # Capsule at z=2.4, penetration = 0.1
            ([0.0, 0.0, 2.4], capsule_axis, capsule_radius, capsule_half_length, box_center, identity, box_size, -0.1),
            # Capsule at z=2.3, penetration = 0.2
            ([0.0, 0.0, 2.3], capsule_axis, capsule_radius, capsule_half_length, box_center, identity, box_size, -0.2),
            # Capsule at z=2.2, penetration = 0.3
            ([0.0, 0.0, 2.2], capsule_axis, capsule_radius, capsule_half_length, box_center, identity, box_size, -0.3),
        ]

        capsule_positions = wp.array([wp.vec3(tc[0][0], tc[0][1], tc[0][2]) for tc in test_cases], dtype=wp.vec3)
        capsule_axes = wp.array([wp.vec3(tc[1][0], tc[1][1], tc[1][2]) for tc in test_cases], dtype=wp.vec3)
        capsule_radii = wp.array([tc[2] for tc in test_cases], dtype=float)
        capsule_half_lengths = wp.array([tc[3] for tc in test_cases], dtype=float)
        box_positions = wp.array([wp.vec3(tc[4][0], tc[4][1], tc[4][2]) for tc in test_cases], dtype=wp.vec3)
        box_rotations = wp.array([tc[5] for tc in test_cases], dtype=wp.mat33)
        box_sizes = wp.array([wp.vec3(tc[6][0], tc[6][1], tc[6][2]) for tc in test_cases], dtype=wp.vec3)
        expected_min_distances = [tc[7] for tc in test_cases]
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
        normals_np = contact_normals.numpy()
        positions_np = contact_positions.numpy()

        # Verify expected distances with analytical validation
        tolerance = 0.05  # Slightly larger tolerance for capsule-box collision
        for i, expected_min_dist in enumerate(expected_min_distances):
            # Find the minimum distance among valid contacts
            valid_distances = [d for d in distances_np[i] if d < MAXVAL * 0.99]
            if len(valid_distances) > 0:
                min_distance = min(valid_distances)
                self.assertAlmostEqual(
                    min_distance,
                    expected_min_dist,
                    delta=tolerance,
                    msg=f"Test case {i}: Expected min distance {expected_min_dist:.4f}, got {min_distance:.4f}",
                )
            elif expected_min_dist > 0:
                # Separated case might not have contacts in some implementations
                pass
            else:
                # Should have contacts for penetrating/touching cases
                self.fail(f"Test case {i}: Expected contacts but found none")

        # Check midpoint property for penetrating contacts
        for i in range(len(test_cases)):
            capsule_pos = np.array(test_cases[i][0])
            capsule_axis = np.array(test_cases[i][1])
            capsule_radius = test_cases[i][2]
            capsule_half_length = test_cases[i][3]
            box_pos = np.array(test_cases[i][4])
            box_rot_mat = test_cases[i][5]
            box_rot = np.array(
                [
                    [box_rot_mat[0, 0], box_rot_mat[0, 1], box_rot_mat[0, 2]],
                    [box_rot_mat[1, 0], box_rot_mat[1, 1], box_rot_mat[1, 2]],
                    [box_rot_mat[2, 0], box_rot_mat[2, 1], box_rot_mat[2, 2]],
                ]
            )
            box_size = np.array(test_cases[i][6])

            for j in range(2):  # Check up to 2 contacts
                if distances_np[i][j] >= MAXVAL * 0.99 or distances_np[i][j] >= 0:
                    continue

                contact_pos = positions_np[i][j]
                normal = normals_np[i][j]
                penetration_depth = distances_np[i][j]

                # Check midpoint property
                surface_point_0 = contact_pos - normal * (penetration_depth / 2.0)
                surface_point_1 = contact_pos + normal * (penetration_depth / 2.0)

                dist_to_capsule = distance_point_to_capsule(
                    surface_point_0, capsule_pos, capsule_axis, capsule_radius, capsule_half_length
                )
                dist_to_box = distance_point_to_box(surface_point_1, box_pos, box_rot, box_size)

                self.assertLess(
                    dist_to_capsule,
                    0.08,
                    msg=f"Test case {i}, contact {j}: Point at -penetration_depth/2 not on capsule surface (error: {dist_to_capsule:.4f})",
                )
                self.assertLess(
                    dist_to_box,
                    0.08,
                    msg=f"Test case {i}, contact {j}: Point at +penetration_depth/2 not on box surface (error: {dist_to_box:.4f})",
                )

    def test_box_box_penetration_depths(self):
        """Test box-box collision with analytical validation of penetration depths.

        This test validates three scenarios:
        1. Face-to-face penetration: Two aligned cubes moved into each other in steps
        2. Edge penetration: Rotated cube with edge penetrating another cube's face
        3. Corner penetration: Rotated cube with corner penetrating another cube's face
        """
        # Identity rotation matrix
        identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        # Cube size (half-extents)
        cube_size = [0.5, 0.5, 0.5]

        # --- Test Case 1: Face-to-face penetration ---
        # Bottom cube at origin, top cube moving down in steps
        # Box 1: center at (0, 0, 0), top face at z=0.5
        # Box 2 configurations with different penetration depths
        face_to_face_cases = []

        # Step 1: Initially separated by 0.1
        # Box 2 center at z=1.1, bottom face at z=0.6, gap = 0.1
        face_to_face_cases.append(
            ([0.0, 0.0, 0.0], identity, cube_size, [0.0, 0.0, 1.1], identity, cube_size, 0.1, False)
        )

        # Step 2: Just touching (gap = 0)
        # Box 2 center at z=1.0, bottom face at z=0.5
        face_to_face_cases.append(
            ([0.0, 0.0, 0.0], identity, cube_size, [0.0, 0.0, 1.0], identity, cube_size, 0.0, True)
        )

        # Step 3: Penetration of 0.1
        # Box 2 center at z=0.9, bottom face at z=0.4
        # Penetration = 0.5 - 0.4 = 0.1
        face_to_face_cases.append(
            ([0.0, 0.0, 0.0], identity, cube_size, [0.0, 0.0, 0.9], identity, cube_size, -0.1, True)
        )

        # Step 4: Penetration of 0.2
        # Box 2 center at z=0.8, bottom face at z=0.3
        # Penetration = 0.5 - 0.3 = 0.2
        face_to_face_cases.append(
            ([0.0, 0.0, 0.0], identity, cube_size, [0.0, 0.0, 0.8], identity, cube_size, -0.2, True)
        )

        # Step 5: Penetration of 0.3
        # Box 2 center at z=0.7, bottom face at z=0.2
        # Penetration = 0.5 - 0.2 = 0.3
        face_to_face_cases.append(
            ([0.0, 0.0, 0.0], identity, cube_size, [0.0, 0.0, 0.7], identity, cube_size, -0.3, True)
        )

        # --- Test Case 2: Edge penetration ---
        # Rotate top cube 45 degrees around X-axis, so its edge penetrates
        # cos(45°) = sin(45°) = sqrt(2)/2 ≈ 0.707107
        cos45 = np.cos(np.pi / 4)
        sin45 = np.sin(np.pi / 4)

        # Rotation matrix around X-axis by 45 degrees
        rot_x_45 = wp.mat33(1.0, 0.0, 0.0, 0.0, cos45, -sin45, 0.0, sin45, cos45)

        # For a cube rotated 45° around X-axis:
        # The diagonal edge in YZ plane now points at 45° to Z-axis
        # The maximum Z extent becomes: 0.5 * sqrt(2) ≈ 0.707107
        # Place box2 such that the rotated edge penetrates box1's top face

        edge_cases = []

        # Edge slightly penetrating: Box2 center at z = 0.5 + 0.5*sqrt(2) - epsilon
        # (Small epsilon to avoid numerical precision issues with exact touching)
        edge_z_touching = 0.5 + 0.5 * np.sqrt(2)
        epsilon = 0.02
        edge_cases.append(
            (
                [0.0, 0.0, 0.0],
                identity,
                cube_size,
                [0.0, 0.0, edge_z_touching - epsilon],
                rot_x_45,
                cube_size,
                -epsilon,
                True,
            )
        )

        # Edge penetration of 0.1
        edge_z_penetrate_01 = edge_z_touching - 0.1
        edge_cases.append(
            ([0.0, 0.0, 0.0], identity, cube_size, [0.0, 0.0, edge_z_penetrate_01], rot_x_45, cube_size, -0.1, True)
        )

        # Edge penetration of 0.2
        edge_z_penetrate_02 = edge_z_touching - 0.2
        edge_cases.append(
            ([0.0, 0.0, 0.0], identity, cube_size, [0.0, 0.0, edge_z_penetrate_02], rot_x_45, cube_size, -0.2, True)
        )

        # --- Test Case 3: Corner penetration ---
        # Rotate cube to point a corner downward by composing rotations
        # Strategy: Rotate 45° around X, then 45° around Y
        # This aligns a corner of the cube to point approximately downward

        # Compute R_y(45) @ R_x(45) to point a corner downward
        # Using numpy for correct matrix multiplication
        Rx = np.array([[1, 0, 0], [0, cos45, -sin45], [0, sin45, cos45]], dtype=np.float32)
        Ry = np.array([[cos45, 0, sin45], [0, 1, 0], [-sin45, 0, cos45]], dtype=np.float32)
        M = Ry @ Rx  # R_y(45) @ R_x(45)
        rot_corner = wp.mat33(M[0, 0], M[0, 1], M[0, 2], M[1, 0], M[1, 1], M[1, 2], M[2, 0], M[2, 1], M[2, 2])

        # Distance from cube center to corner: 0.5 * sqrt(3) ≈ 0.866025
        corner_dist = 0.5 * np.sqrt(3)

        corner_cases = []

        # Corner slightly penetrating: Box2 center at z = 0.5 + corner_dist - epsilon
        # (Small epsilon to avoid numerical precision issues with exact touching)
        corner_z_touching = 0.5 + corner_dist
        corner_cases.append(
            (
                [0.0, 0.0, 0.0],
                identity,
                cube_size,
                [0.0, 0.0, corner_z_touching - epsilon],
                rot_corner,
                cube_size,
                -epsilon,
                True,
            )
        )

        # Corner penetration of 0.1
        corner_z_penetrate_01 = corner_z_touching - 0.1
        corner_cases.append(
            (
                [0.0, 0.0, 0.0],
                identity,
                cube_size,
                [0.0, 0.0, corner_z_penetrate_01],
                rot_corner,
                cube_size,
                -0.1,
                True,
            )
        )

        # Corner penetration of 0.15
        corner_z_penetrate_015 = corner_z_touching - 0.15
        corner_cases.append(
            (
                [0.0, 0.0, 0.0],
                identity,
                cube_size,
                [0.0, 0.0, corner_z_penetrate_015],
                rot_corner,
                cube_size,
                -0.15,
                True,
            )
        )

        # Combine all test cases
        all_test_cases = face_to_face_cases + edge_cases + corner_cases

        # Prepare arrays for kernel launch
        box1_positions = wp.array([wp.vec3(tc[0][0], tc[0][1], tc[0][2]) for tc in all_test_cases], dtype=wp.vec3)
        box1_rotations = wp.array([tc[1] for tc in all_test_cases], dtype=wp.mat33)
        box1_sizes = wp.array([wp.vec3(tc[2][0], tc[2][1], tc[2][2]) for tc in all_test_cases], dtype=wp.vec3)
        box2_positions = wp.array([wp.vec3(tc[3][0], tc[3][1], tc[3][2]) for tc in all_test_cases], dtype=wp.vec3)
        box2_rotations = wp.array([tc[4] for tc in all_test_cases], dtype=wp.mat33)
        box2_sizes = wp.array([wp.vec3(tc[5][0], tc[5][1], tc[5][2]) for tc in all_test_cases], dtype=wp.vec3)
        distances = wp.array(
            [wp.types.vector(8, wp.float32)()] * len(all_test_cases), dtype=wp.types.vector(8, wp.float32)
        )
        contact_positions = wp.array(
            [wp.types.matrix((8, 3), wp.float32)()] * len(all_test_cases), dtype=wp.types.matrix((8, 3), wp.float32)
        )
        contact_normals = wp.array(
            [wp.types.matrix((8, 3), wp.float32)()] * len(all_test_cases), dtype=wp.types.matrix((8, 3), wp.float32)
        )

        # Launch kernel
        wp.launch(
            test_box_box_kernel,
            dim=len(all_test_cases),
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
        normals_np = contact_normals.numpy()

        # Validate results
        for i, tc in enumerate(all_test_cases):
            expected_penetration = tc[6]  # Expected minimum penetration depth
            expect_contacts = tc[7]  # Whether contacts are expected

            # Count valid contacts and find deepest penetration
            valid_contacts = []
            for j in range(8):
                if distances_np[i][j] < MAXVAL * 0.99:
                    valid_contacts.append(distances_np[i][j])

            if expect_contacts:
                self.assertGreater(
                    len(valid_contacts),
                    0,
                    msg=f"Test case {i}: Expected contacts but found none. Box2 at z={all_test_cases[i][3][2]:.4f}",
                )

                # Find deepest penetration (most negative distance)
                min_distance = min(valid_contacts)

                # For penetrating contacts, verify the penetration depth
                if expected_penetration < 0:
                    # Allow some tolerance for numerical precision and collision detection approximation
                    tolerance = 0.05  # 5cm tolerance for collision detection
                    self.assertLess(
                        min_distance,
                        0.0,
                        msg=f"Test case {i}: Expected penetration but got separation distance {min_distance:.4f}",
                    )
                    self.assertAlmostEqual(
                        min_distance,
                        expected_penetration,
                        delta=tolerance,
                        msg=f"Test case {i}: Expected penetration depth {expected_penetration:.4f}, "
                        f"got {min_distance:.4f}. Box2 z={all_test_cases[i][3][2]:.4f}",
                    )
                elif expected_penetration == 0:
                    # Just touching case - should be very close to 0
                    tolerance = 0.05
                    self.assertAlmostEqual(
                        min_distance,
                        0.0,
                        delta=tolerance,
                        msg=f"Test case {i}: Expected touching (distance ≈ 0), got {min_distance:.4f}",
                    )
            else:
                self.assertEqual(
                    len(valid_contacts),
                    0,
                    msg=f"Test case {i}: Expected no contacts but found {len(valid_contacts)}",
                )

        # Check that contact normals are unit length for penetrating cases
        for i, tc in enumerate(all_test_cases):
            expected_penetration = tc[6]
            if expected_penetration >= 0:
                continue

            for j in range(8):
                if distances_np[i][j] >= MAXVAL * 0.99:
                    continue

                normal = normals_np[i][j]
                normal_length = np.linalg.norm(normal)
                self.assertAlmostEqual(
                    normal_length,
                    1.0,
                    delta=0.01,
                    msg=f"Test case {i}, contact {j}: Normal not unit length: {normal_length:.4f}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
