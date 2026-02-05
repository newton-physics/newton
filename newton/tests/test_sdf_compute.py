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

"""Test compute_sdf function for SDF generation.

This test suite validates:
1. SDF values inside the extent are smaller than the background value
2. Sparse and coarse SDFs have consistent values
3. SDF gradients point away from the surface
4. Points inside the mesh have negative SDF values
5. Points outside the mesh have positive SDF values

Note: These tests require GPU (CUDA) since wp.Volume only supports CUDA devices.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton import GeoType, Mesh
from newton._src.geometry.sdf_contact import sample_sdf_extrapolated, sample_sdf_grad_extrapolated
from newton.geometry import SDFData, compute_sdf
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

# Skip all tests in this module if CUDA is not available
# wp.Volume only supports CUDA devices
_cuda_available = wp.is_cuda_available()


def create_box_mesh(half_extents: tuple[float, float, float]) -> Mesh:
    """Create a simple box mesh for testing."""
    hx, hy, hz = half_extents
    vertices = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            # Bottom face (z = -hz)
            0,
            2,
            1,
            0,
            3,
            2,
            # Top face (z = hz)
            4,
            5,
            6,
            4,
            6,
            7,
            # Front face (y = -hy)
            0,
            1,
            5,
            0,
            5,
            4,
            # Back face (y = hy)
            2,
            3,
            7,
            2,
            7,
            6,
            # Left face (x = -hx)
            0,
            4,
            7,
            0,
            7,
            3,
            # Right face (x = hx)
            1,
            2,
            6,
            1,
            6,
            5,
        ],
        dtype=np.int32,
    )
    return Mesh(vertices, indices)


def create_sphere_mesh(radius: float, subdivisions: int = 2) -> Mesh:
    """Create a sphere mesh by subdividing an icosahedron."""
    # Golden ratio
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    # Icosahedron vertices (normalized and scaled by radius)
    verts_list = [
        [-1, phi, 0],
        [1, phi, 0],
        [-1, -phi, 0],
        [1, -phi, 0],
        [0, -1, phi],
        [0, 1, phi],
        [0, -1, -phi],
        [0, 1, -phi],
        [phi, 0, -1],
        [phi, 0, 1],
        [-phi, 0, -1],
        [-phi, 0, 1],
    ]
    norm_factor = np.linalg.norm(verts_list[0])
    verts_list = [
        [v[0] / norm_factor * radius, v[1] / norm_factor * radius, v[2] / norm_factor * radius] for v in verts_list
    ]

    # Icosahedron faces (CCW winding for outward normals)
    faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]

    # Subdivide
    for _ in range(subdivisions):
        new_faces = []
        edge_midpoints = {}

        def get_midpoint(i0, i1, _edge_midpoints=edge_midpoints):
            key = (min(i0, i1), max(i0, i1))
            if key not in _edge_midpoints:
                v0, v1 = verts_list[i0], verts_list[i1]
                mid = [(v0[0] + v1[0]) / 2, (v0[1] + v1[1]) / 2, (v0[2] + v1[2]) / 2]
                length = np.sqrt(mid[0] ** 2 + mid[1] ** 2 + mid[2] ** 2)
                mid = [mid[0] / length * radius, mid[1] / length * radius, mid[2] / length * radius]
                _edge_midpoints[key] = len(verts_list)
                verts_list.append(mid)
            return _edge_midpoints[key]

        for f in faces:
            a = get_midpoint(f[0], f[1])
            b = get_midpoint(f[1], f[2])
            c = get_midpoint(f[2], f[0])
            new_faces.extend([[f[0], a, c], [f[1], b, a], [f[2], c, b], [a, b, c]])
        faces = new_faces

    verts = np.array(verts_list, dtype=np.float32)
    indices = np.array(faces, dtype=np.int32).flatten()
    return Mesh(verts, indices)


def invert_mesh_winding(mesh: Mesh) -> Mesh:
    """Create a mesh with inverted winding by swapping triangle indices."""
    indices = mesh.indices.copy()
    # Swap second and third vertex of each triangle to flip winding
    for i in range(0, len(indices), 3):
        indices[i + 1], indices[i + 2] = indices[i + 2], indices[i + 1]
    return Mesh(mesh.vertices.copy(), indices)


# Warp kernel for sampling SDF values
@wp.kernel
def sample_sdf_kernel(
    volume_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    values: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    point = points[tid]
    index_pos = wp.volume_world_to_index(volume_id, point)
    values[tid] = wp.volume_sample_f(volume_id, index_pos, wp.Volume.LINEAR)


# Warp kernel for sampling SDF gradients
@wp.kernel
def sample_sdf_gradient_kernel(
    volume_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    values: wp.array(dtype=wp.float32),
    gradients: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    point = points[tid]
    index_pos = wp.volume_world_to_index(volume_id, point)
    grad = wp.vec3(0.0, 0.0, 0.0)
    values[tid] = wp.volume_sample_grad_f(volume_id, index_pos, wp.Volume.LINEAR, grad)
    gradients[tid] = grad


def sample_sdf_at_points(volume, points_np: np.ndarray) -> np.ndarray:
    """Sample SDF values at given points using a Warp kernel."""
    n_points = len(points_np)
    points = wp.array(points_np, dtype=wp.vec3)
    values = wp.zeros(n_points, dtype=wp.float32)

    wp.launch(
        sample_sdf_kernel,
        dim=n_points,
        inputs=[volume.id, points, values],
    )
    wp.synchronize()

    return values.numpy()


def sample_sdf_with_gradient(volume, points_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample SDF values and gradients at given points using a Warp kernel."""
    n_points = len(points_np)
    points = wp.array(points_np, dtype=wp.vec3)
    values = wp.zeros(n_points, dtype=wp.float32)
    gradients = wp.zeros(n_points, dtype=wp.vec3)

    wp.launch(
        sample_sdf_gradient_kernel,
        dim=n_points,
        inputs=[volume.id, points, values, gradients],
    )
    wp.synchronize()

    return values.numpy(), gradients.numpy()


@unittest.skipUnless(_cuda_available, "wp.Volume requires CUDA device")
class TestComputeSDF(unittest.TestCase):
    """Test the compute_sdf function."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        wp.init()

    def setUp(self):
        """Set up test fixtures."""
        self.half_extents = (0.5, 0.5, 0.5)
        self.mesh = create_box_mesh(self.half_extents)

    def test_sdf_returns_valid_data(self):
        """Test that compute_sdf returns valid data."""
        sdf_data, sparse_volume, coarse_volume, _ = compute_sdf(
            mesh_src=self.mesh,
            shape_type=GeoType.MESH,
            shape_thickness=0.0,
        )

        self.assertIsNotNone(sparse_volume)
        self.assertIsNotNone(coarse_volume)
        self.assertNotEqual(sdf_data.sparse_sdf_ptr, 0)
        self.assertNotEqual(sdf_data.coarse_sdf_ptr, 0)

    def test_sdf_extents_are_valid(self):
        """Test that SDF extents match the mesh bounds."""
        sdf_data, _, _, _ = compute_sdf(
            mesh_src=self.mesh,
            shape_type=GeoType.MESH,
            shape_thickness=0.0,
            margin=0.05,
        )

        # Half extents should be at least as large as mesh half extents + margin
        min_half_extent = min(self.half_extents) + 0.05
        self.assertGreaterEqual(sdf_data.half_extents[0], min_half_extent - 0.01)
        self.assertGreaterEqual(sdf_data.half_extents[1], min_half_extent - 0.01)
        self.assertGreaterEqual(sdf_data.half_extents[2], min_half_extent - 0.01)

    def test_sparse_sdf_values_near_surface(self):
        """Test that sparse SDF values near the surface are smaller than background.

        Note: The sparse SDF is a narrow-band SDF, so only values near the surface
        (within narrow_band_distance) will have valid values. Points far from the
        surface will return the background value.
        """
        sdf_data, sparse_volume, _, _ = compute_sdf(
            mesh_src=self.mesh,
            shape_type=GeoType.MESH,
            shape_thickness=0.0,
            narrow_band_distance=(-0.1, 0.1),
        )

        # Test points near the surface (within narrow band)
        # These are just inside and just outside each face of the box
        test_points = np.array(
            [
                [0.45, 0.0, 0.0],  # Near +X face (inside)
                [0.55, 0.0, 0.0],  # Near +X face (outside)
                [0.0, 0.45, 0.0],  # Near +Y face (inside)
                [0.0, 0.0, 0.45],  # Near +Z face (inside)
                [-0.45, 0.0, 0.0],  # Near -X face (inside)
            ],
            dtype=np.float32,
        )

        values = sample_sdf_at_points(sparse_volume, test_points)

        for _i, (point, value) in enumerate(zip(test_points, values, strict=False)):
            self.assertLess(
                value,
                sdf_data.background_value,
                f"SDF value {value} at {point} (near surface) should be less than background {sdf_data.background_value}",
            )

    def test_coarse_sdf_values_inside_extent(self):
        """Test that coarse SDF values inside the extent are smaller than background."""
        sdf_data, _, coarse_volume, _ = compute_sdf(
            mesh_src=self.mesh,
            shape_type=GeoType.MESH,
            shape_thickness=0.0,
        )

        # Sample points inside the SDF extent
        center = np.array([sdf_data.center[0], sdf_data.center[1], sdf_data.center[2]])
        half_ext = np.array([sdf_data.half_extents[0], sdf_data.half_extents[1], sdf_data.half_extents[2]])

        # Generate test points inside the extent
        test_points = np.array(
            [
                center,  # Center
                center + half_ext * 0.5,  # Offset from center
                center - half_ext * 0.5,  # Other offset
            ],
            dtype=np.float32,
        )

        values = sample_sdf_at_points(coarse_volume, test_points)

        for _i, (point, value) in enumerate(zip(test_points, values, strict=False)):
            self.assertLess(
                value,
                sdf_data.background_value,
                f"Coarse SDF value {value} at {point} should be less than background {sdf_data.background_value}",
            )

    def test_coarse_sdf_values_at_extent_boundary(self):
        """Test that coarse SDF values at the extent boundary are valid.

        The extent boundary is at center ± half_extents. With margin=0.05 and
        mesh half_extents of 0.5, the boundary is at approximately ±0.55.
        Points at or near this boundary should still have valid SDF values.
        """
        margin = 0.05
        sdf_data, _, coarse_volume, _ = compute_sdf(
            mesh_src=self.mesh,
            shape_type=GeoType.MESH,
            shape_thickness=0.0,
            margin=margin,
        )

        center = np.array([sdf_data.center[0], sdf_data.center[1], sdf_data.center[2]])
        half_ext = np.array([sdf_data.half_extents[0], sdf_data.half_extents[1], sdf_data.half_extents[2]])

        # Verify the extent includes the margin
        expected_half_ext = self.half_extents[0] + margin  # 0.5 + 0.05 = 0.55
        self.assertAlmostEqual(
            half_ext[0],
            expected_half_ext,
            places=2,
            msg=f"Expected half_extent ~{expected_half_ext}, got {half_ext[0]}",
        )

        # Test points at extent boundary corners (slightly inside to ensure we're in the volume)
        boundary_factor = 0.99  # Just inside the boundary
        test_points = np.array(
            [
                # Corners of the extent (outside the mesh, inside the extent)
                center + half_ext * np.array([boundary_factor, boundary_factor, boundary_factor]),
                center + half_ext * np.array([boundary_factor, boundary_factor, -boundary_factor]),
                center + half_ext * np.array([boundary_factor, -boundary_factor, boundary_factor]),
                center + half_ext * np.array([boundary_factor, -boundary_factor, -boundary_factor]),
                center + half_ext * np.array([-boundary_factor, boundary_factor, boundary_factor]),
                center + half_ext * np.array([-boundary_factor, boundary_factor, -boundary_factor]),
                center + half_ext * np.array([-boundary_factor, -boundary_factor, boundary_factor]),
                center + half_ext * np.array([-boundary_factor, -boundary_factor, -boundary_factor]),
                # Face centers at the boundary
                center + half_ext * np.array([boundary_factor, 0.0, 0.0]),
                center + half_ext * np.array([-boundary_factor, 0.0, 0.0]),
                center + half_ext * np.array([0.0, boundary_factor, 0.0]),
                center + half_ext * np.array([0.0, -boundary_factor, 0.0]),
                center + half_ext * np.array([0.0, 0.0, boundary_factor]),
                center + half_ext * np.array([0.0, 0.0, -boundary_factor]),
            ],
            dtype=np.float32,
        )

        values = sample_sdf_at_points(coarse_volume, test_points)

        for i, (point, value) in enumerate(zip(test_points, values, strict=False)):
            self.assertLess(
                value,
                sdf_data.background_value,
                f"Coarse SDF at extent boundary point {i} = {point} should be < {sdf_data.background_value}, got {value}",
            )
            # Corners are outside the mesh (which is at ±0.5), so SDF should be positive
            # Face center points at ±0.55 on one axis and 0 on others are also outside mesh
            self.assertGreater(
                value,
                0.0,
                f"Coarse SDF at extent boundary (outside mesh at ±0.5) should be positive, got {value} at {point}",
            )

    def test_sparse_sdf_values_at_extent_boundary(self):
        """Test that sparse SDF values at the actual extent boundary are valid.

        The extent boundary is at center ± half_extents. With margin=0.05 and
        mesh half_extents of 0.5, the extent boundary is at approximately ±0.55.

        The narrow band extends ±0.1 from the surface (at ±0.5), so the narrow
        band covers [0.4, 0.6] for each face. The extent boundary at 0.55 is
        within this narrow band, so we should get valid values there.
        """
        margin = 0.05
        sdf_data, sparse_volume, _, _ = compute_sdf(
            mesh_src=self.mesh,
            shape_type=GeoType.MESH,
            shape_thickness=0.0,
            margin=margin,
        )

        center = np.array([sdf_data.center[0], sdf_data.center[1], sdf_data.center[2]])
        half_ext = np.array([sdf_data.half_extents[0], sdf_data.half_extents[1], sdf_data.half_extents[2]])

        # Verify the extent is what we expect (mesh half_extents + margin)
        expected_half_ext = self.half_extents[0] + margin  # 0.5 + 0.05 = 0.55
        self.assertAlmostEqual(
            half_ext[0],
            expected_half_ext,
            places=2,
            msg=f"Expected half_extent ~{expected_half_ext}, got {half_ext[0]}",
        )

        # Test points AT the extent boundary (0.99 * half_ext to stay just inside)
        # These should be within the narrow band since:
        # - Surface is at 0.5
        # - Narrow band extends to 0.5 + 0.1 = 0.6
        # - Extent boundary is at ~0.55, which is < 0.6
        boundary_factor = 0.99
        boundary_points = np.array(
            [
                # Face centers at extent boundary
                center + half_ext * np.array([boundary_factor, 0.0, 0.0]),
                center + half_ext * np.array([-boundary_factor, 0.0, 0.0]),
                center + half_ext * np.array([0.0, boundary_factor, 0.0]),
                center + half_ext * np.array([0.0, -boundary_factor, 0.0]),
                center + half_ext * np.array([0.0, 0.0, boundary_factor]),
                center + half_ext * np.array([0.0, 0.0, -boundary_factor]),
            ],
            dtype=np.float32,
        )

        values = sample_sdf_at_points(sparse_volume, boundary_points)

        for i, (point, value) in enumerate(zip(boundary_points, values, strict=False)):
            self.assertLess(
                value,
                sdf_data.background_value,
                f"Sparse SDF at extent boundary point {i} = {point} should be < {sdf_data.background_value}, got {value}",
            )
            # These points are outside the mesh surface, so SDF should be positive
            self.assertGreater(
                value,
                0.0,
                f"Sparse SDF at extent boundary (outside mesh) should be positive, got {value} at {point}",
            )

    def test_sdf_negative_inside_mesh(self):
        """Test that SDF values are negative inside the mesh.

        For the sparse SDF, we test a point just inside the surface (within the narrow band).
        For the coarse SDF, we can test the center since it covers the entire volume.
        """
        _sdf_data, sparse_volume, coarse_volume, _ = compute_sdf(
            mesh_src=self.mesh,
            shape_type=GeoType.MESH,
            shape_thickness=0.0,
        )

        # For sparse SDF: test point just inside a face (within narrow band)
        near_surface_inside = np.array([[0.45, 0.0, 0.0]], dtype=np.float32)
        sparse_values = sample_sdf_at_points(sparse_volume, near_surface_inside)
        self.assertLess(
            sparse_values[0], 0.0, f"Sparse SDF just inside surface should be negative, got {sparse_values[0]}"
        )

        # For coarse SDF: test at center (coarse SDF covers entire volume)
        center_point = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        coarse_values = sample_sdf_at_points(coarse_volume, center_point)
        self.assertLess(coarse_values[0], 0.0, f"Coarse SDF at center should be negative, got {coarse_values[0]}")

    def test_sdf_positive_outside_mesh(self):
        """Test that SDF values are positive outside the mesh."""
        _sdf_data, sparse_volume, coarse_volume, _ = compute_sdf(
            mesh_src=self.mesh,
            shape_type=GeoType.MESH,
            shape_thickness=0.0,
        )

        # Point well outside the box
        outside_point = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)

        # Test sparse SDF (may hit background value if outside narrow band)
        sparse_values = sample_sdf_at_points(sparse_volume, outside_point)
        self.assertGreater(sparse_values[0], 0.0, f"Sparse SDF outside should be positive, got {sparse_values[0]}")

        # Test coarse SDF
        coarse_values = sample_sdf_at_points(coarse_volume, outside_point)
        self.assertGreater(coarse_values[0], 0.0, f"Coarse SDF outside should be positive, got {coarse_values[0]}")

    def test_sdf_gradient_points_outward(self):
        """Test that SDF gradient points away from the surface (outward)."""
        _sdf_data, sparse_volume, _, _ = compute_sdf(
            mesh_src=self.mesh,
            shape_type=GeoType.MESH,
            shape_thickness=0.0,
        )

        # Test gradient at a point slightly inside the +X face
        test_points = np.array([[0.4, 0.0, 0.0]], dtype=np.float32)  # Inside the box, close to +X face

        _values, gradients = sample_sdf_with_gradient(sparse_volume, test_points)

        gradient = gradients[0]
        gradient_norm = np.linalg.norm(gradient)

        if gradient_norm > 1e-6:
            gradient_normalized = gradient / gradient_norm
            # X component should be positive (pointing outward toward +X face)
            self.assertGreater(
                gradient_normalized[0],
                0.5,
                f"Gradient should point toward +X, got {gradient_normalized}",
            )

    def test_sparse_and_coarse_consistency(self):
        """Test that sparse and coarse SDFs have consistent signs near the surface.

        We test at a point near the surface (within the narrow band) where both
        SDFs should have valid values.
        """
        _sdf_data, sparse_volume, coarse_volume, _ = compute_sdf(
            mesh_src=self.mesh,
            shape_type=GeoType.MESH,
            shape_thickness=0.0,
        )

        # Sample at a point near the surface (within narrow band)
        near_surface = np.array([[0.45, 0.0, 0.0]], dtype=np.float32)

        sparse_values = sample_sdf_at_points(sparse_volume, near_surface)
        coarse_values = sample_sdf_at_points(coarse_volume, near_surface)

        # Both should have the same sign (both negative inside)
        self.assertEqual(
            np.sign(sparse_values[0]),
            np.sign(coarse_values[0]),
            f"Sparse ({sparse_values[0]}) and coarse ({coarse_values[0]}) should have same sign near surface",
        )

    def test_thickness_offset(self):
        """Test that thickness offsets the SDF values.

        We test near the surface where the sparse SDF has valid values.
        """
        thickness = 0.1

        _, sparse_no_thickness, _, _ = compute_sdf(
            mesh_src=self.mesh,
            shape_type=GeoType.MESH,
            shape_thickness=0.0,
        )

        _, sparse_with_thickness, _, _ = compute_sdf(
            mesh_src=self.mesh,
            shape_type=GeoType.MESH,
            shape_thickness=thickness,
        )

        # Sample near the surface (within narrow band)
        near_surface = np.array([[0.45, 0.0, 0.0]], dtype=np.float32)

        values_no_thick = sample_sdf_at_points(sparse_no_thickness, near_surface)
        values_with_thick = sample_sdf_at_points(sparse_with_thickness, near_surface)

        # With thickness, SDF should be offset (more negative = thicker shell)
        self.assertAlmostEqual(
            values_with_thick[0],
            values_no_thick[0] - thickness,
            places=2,
            msg=f"Thickness should offset SDF by -{thickness}",
        )

    def test_inverted_winding_sphere(self):
        """Test SDF computation for a sphere mesh with inverted winding.

        Verifies that:
        1. The inverted winding is detected (winding threshold becomes -0.5)
        2. Points inside the sphere still have negative SDF values
        3. Points outside the sphere still have positive SDF values
        """
        radius = 0.5
        sphere = create_sphere_mesh(radius, subdivisions=2)
        inverted_sphere = invert_mesh_winding(sphere)

        # Compute SDF at low resolution for speed, with wider narrow band
        _, sparse_volume, coarse_volume, _ = compute_sdf(
            mesh_src=inverted_sphere,
            shape_type=GeoType.MESH,
            shape_thickness=0.0,
            max_resolution=32,
            narrow_band_distance=(-0.2, 0.2),  # Wider band for testing
        )

        self.assertIsNotNone(sparse_volume)
        self.assertIsNotNone(coarse_volume)

        # Test points inside the sphere (should be negative)
        inside_points = np.array(
            [
                [0.0, 0.0, 0.0],  # Center
                [0.1, 0.0, 0.0],  # Slightly off center
                [0.0, 0.2, 0.0],  # Another inside point
                [0.1, 0.1, 0.1],  # Inside diagonal
            ],
            dtype=np.float32,
        )

        inside_values = sample_sdf_at_points(coarse_volume, inside_points)
        for i, (point, value) in enumerate(zip(inside_points, inside_values, strict=False)):
            self.assertLess(value, 0.0, f"Point {i} at {point} should be inside (negative), got {value}")

        # Test points near but inside sphere surface (should be negative)
        # The SDF extent is ~1.1, so stay well within bounds
        near_inside_points = np.array(
            [
                [radius - 0.05, 0.0, 0.0],  # Just inside +X
                [0.0, radius - 0.05, 0.0],  # Just inside +Y
                [0.0, 0.0, radius - 0.05],  # Just inside +Z
            ],
            dtype=np.float32,
        )

        near_inside_values = sample_sdf_at_points(coarse_volume, near_inside_points)
        for i, (point, value) in enumerate(zip(near_inside_points, near_inside_values, strict=False)):
            self.assertLess(value, 0.0, f"Point {i} at {point} should be inside (negative), got {value}")

        # Test points just outside sphere surface (should be positive)
        # Use small offset (0.02) to stay well within the narrow band and volume extent
        outside_offset = 0.02
        outside_points = np.array(
            [
                [radius + outside_offset, 0.0, 0.0],  # Just outside +X
                [0.0, radius + outside_offset, 0.0],  # Just outside +Y
                [0.0, 0.0, radius + outside_offset],  # Just outside +Z
                [-(radius + outside_offset), 0.0, 0.0],  # Just outside -X
                [0.0, -(radius + outside_offset), 0.0],  # Just outside -Y
                [0.0, 0.0, -(radius + outside_offset)],  # Just outside -Z
            ],
            dtype=np.float32,
        )

        outside_values = sample_sdf_at_points(coarse_volume, outside_points)
        for i, (point, value) in enumerate(zip(outside_points, outside_values, strict=False)):
            self.assertGreater(value, 0.0, f"Point {i} at {point} should be outside (positive), got {value}")


@unittest.skipUnless(_cuda_available, "wp.Volume requires CUDA device")
class TestComputeSDFGridSampling(unittest.TestCase):
    """Test SDF by sampling on a grid of points."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        wp.init()

    def setUp(self):
        """Set up test fixtures."""
        self.half_extents = (0.5, 0.5, 0.5)
        self.mesh = create_box_mesh(self.half_extents)

    def test_grid_sampling_sparse_sdf_near_surface(self):
        """Sample sparse SDF on a grid near the surface and verify values are valid.

        Since the sparse SDF is a narrow-band SDF, we sample points near the surface
        (on a shell around the box) where the SDF should have valid values.
        """
        sdf_data, sparse_volume, _, _ = compute_sdf(
            mesh_src=self.mesh,
            shape_type=GeoType.MESH,
            shape_thickness=0.0,
        )

        # Sample points on a grid near the +X face of the box (within narrow band)
        test_points = []
        for j in range(5):
            for k in range(5):
                # Grid on the YZ plane, at x = 0.45 (just inside the surface)
                y = (j / 4.0 - 0.5) * 0.8  # Range [-0.4, 0.4]
                z = (k / 4.0 - 0.5) * 0.8
                test_points.append([0.45, y, z])
                # Also test just outside
                test_points.append([0.55, y, z])

        test_points = np.array(test_points, dtype=np.float32)
        values = sample_sdf_at_points(sparse_volume, test_points)

        for i, (point, value) in enumerate(zip(test_points, values, strict=False)):
            self.assertLess(
                value,
                sdf_data.background_value,
                f"SDF at point {i} = {point} (near surface) should be < {sdf_data.background_value}, got {value}",
            )

    def test_grid_sampling_coarse_sdf(self):
        """Sample coarse SDF on a grid and verify all values are less than background."""
        sdf_data, _, coarse_volume, _ = compute_sdf(
            mesh_src=self.mesh,
            shape_type=GeoType.MESH,
            shape_thickness=0.0,
        )

        # Create a grid of test points inside the extent
        center = np.array([sdf_data.center[0], sdf_data.center[1], sdf_data.center[2]])
        half_ext = np.array([sdf_data.half_extents[0], sdf_data.half_extents[1], sdf_data.half_extents[2]])

        # Sample on a 5x5x5 grid inside the extent
        test_points = []
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    # Normalized coordinates [-0.8, 0.8] to stay inside extent
                    u = (i / 4.0 - 0.5) * 1.6
                    v = (j / 4.0 - 0.5) * 1.6
                    w = (k / 4.0 - 0.5) * 1.6
                    point = center + half_ext * np.array([u, v, w])
                    test_points.append(point)

        test_points = np.array(test_points, dtype=np.float32)
        values = sample_sdf_at_points(coarse_volume, test_points)

        for i, (point, value) in enumerate(zip(test_points, values, strict=False)):
            self.assertLess(
                value,
                sdf_data.background_value,
                f"Coarse SDF at grid point {i} = {point} should be < {sdf_data.background_value}, got {value}",
            )


@wp.kernel
def sample_sdf_extrapolated_kernel(
    sdf_data: SDFData,
    points: wp.array(dtype=wp.vec3),
    values: wp.array(dtype=wp.float32),
):
    """Kernel to test sample_sdf_extrapolated function."""
    tid = wp.tid()
    values[tid] = sample_sdf_extrapolated(sdf_data, points[tid])


@wp.kernel
def sample_sdf_grad_extrapolated_kernel(
    sdf_data: SDFData,
    points: wp.array(dtype=wp.vec3),
    values: wp.array(dtype=wp.float32),
    gradients: wp.array(dtype=wp.vec3),
):
    """Kernel to test sample_sdf_grad_extrapolated function."""
    tid = wp.tid()
    dist, grad = sample_sdf_grad_extrapolated(sdf_data, points[tid])
    values[tid] = dist
    gradients[tid] = grad


def sample_extrapolated_at_points(sdf_data: SDFData, points_np: np.ndarray) -> np.ndarray:
    """Sample extrapolated SDF values at given points."""
    n_points = len(points_np)
    points = wp.array(points_np, dtype=wp.vec3)
    values = wp.zeros(n_points, dtype=wp.float32)

    wp.launch(
        sample_sdf_extrapolated_kernel,
        dim=n_points,
        inputs=[sdf_data, points, values],
    )
    wp.synchronize()

    return values.numpy()


def sample_extrapolated_with_gradient(sdf_data: SDFData, points_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample extrapolated SDF values and gradients at given points."""
    n_points = len(points_np)
    points = wp.array(points_np, dtype=wp.vec3)
    values = wp.zeros(n_points, dtype=wp.float32)
    gradients = wp.zeros(n_points, dtype=wp.vec3)

    wp.launch(
        sample_sdf_grad_extrapolated_kernel,
        dim=n_points,
        inputs=[sdf_data, points, values, gradients],
    )
    wp.synchronize()

    return values.numpy(), gradients.numpy()


@unittest.skipUnless(_cuda_available, "wp.Volume requires CUDA device")
class TestSDFExtrapolation(unittest.TestCase):
    """Test the SDF extrapolation functions."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        wp.init()

    def setUp(self):
        """Set up test fixtures."""
        self.half_extents = (0.5, 0.5, 0.5)
        self.mesh = create_box_mesh(self.half_extents)
        # Create SDF with known parameters
        self.sdf_data, self.sparse_volume, self.coarse_volume, _ = compute_sdf(
            mesh_src=self.mesh,
            shape_type=GeoType.MESH,
            shape_thickness=0.0,
            narrow_band_distance=(-0.1, 0.1),
            margin=0.05,
        )

    def test_extrapolated_inside_narrow_band(self):
        """Test that points inside narrow band return sparse grid values."""
        # Points near surface (within narrow band of ±0.1 from surface at 0.5)
        test_points = np.array(
            [
                [0.45, 0.0, 0.0],  # Just inside +X face
                [0.55, 0.0, 0.0],  # Just outside +X face
                [0.0, 0.45, 0.0],  # Just inside +Y face
                [0.0, 0.0, 0.45],  # Just inside +Z face
            ],
            dtype=np.float32,
        )

        extrapolated_values = sample_extrapolated_at_points(self.sdf_data, test_points)
        direct_values = sample_sdf_at_points(self.sparse_volume, test_points)

        for i, (ext_val, direct_val) in enumerate(zip(extrapolated_values, direct_values, strict=False)):
            # Within narrow band, extrapolated should match sparse grid
            self.assertAlmostEqual(
                ext_val,
                direct_val,
                places=4,
                msg=f"Point {i}: extrapolated ({ext_val}) should match sparse ({direct_val})",
            )

    def test_extrapolated_inside_extent_outside_narrow_band(self):
        """Test that points inside extent but outside narrow band return coarse grid values."""
        # Center of the box - inside extent but outside narrow band
        test_points = np.array(
            [
                [0.0, 0.0, 0.0],  # Center
                [0.1, 0.1, 0.1],  # Near center
                [0.2, 0.0, 0.0],  # Partway to surface but outside narrow band
            ],
            dtype=np.float32,
        )

        extrapolated_values = sample_extrapolated_at_points(self.sdf_data, test_points)
        coarse_values = sample_sdf_at_points(self.coarse_volume, test_points)

        for i, (ext_val, coarse_val) in enumerate(zip(extrapolated_values, coarse_values, strict=False)):
            # Inside extent but outside narrow band, should use coarse grid
            self.assertAlmostEqual(
                ext_val,
                coarse_val,
                places=4,
                msg=f"Point {i}: extrapolated ({ext_val}) should match coarse ({coarse_val})",
            )

    def test_extrapolated_outside_extent(self):
        """Test that points outside extent return extrapolated values."""
        center = np.array([self.sdf_data.center[0], self.sdf_data.center[1], self.sdf_data.center[2]])
        half_ext = np.array(
            [self.sdf_data.half_extents[0], self.sdf_data.half_extents[1], self.sdf_data.half_extents[2]]
        )

        # Points outside the extent (beyond center ± half_extents)
        outside_distance = 0.5  # Distance beyond boundary
        test_points = np.array(
            [
                center + np.array([half_ext[0] + outside_distance, 0.0, 0.0]),  # Outside +X
                center + np.array([0.0, half_ext[1] + outside_distance, 0.0]),  # Outside +Y
                center + np.array([0.0, 0.0, half_ext[2] + outside_distance]),  # Outside +Z
            ],
            dtype=np.float32,
        )

        # Get boundary points (clamped to extent)
        boundary_points = np.array(
            [
                center + np.array([half_ext[0] - 1e-6, 0.0, 0.0]),  # +X boundary
                center + np.array([0.0, half_ext[1] - 1e-6, 0.0]),  # +Y boundary
                center + np.array([0.0, 0.0, half_ext[2] - 1e-6]),  # +Z boundary
            ],
            dtype=np.float32,
        )

        extrapolated_values = sample_extrapolated_at_points(self.sdf_data, test_points)
        boundary_values = sample_sdf_at_points(self.coarse_volume, boundary_points)

        for i in range(len(test_points)):
            # Extrapolated value should be boundary_value + distance_to_boundary
            expected = boundary_values[i] + outside_distance
            self.assertAlmostEqual(
                extrapolated_values[i],
                expected,
                places=2,
                msg=f"Point {i}: extrapolated ({extrapolated_values[i]}) should be boundary ({boundary_values[i]}) + distance ({outside_distance}) = {expected}",
            )

    def test_extrapolated_values_are_continuous(self):
        """Test that extrapolated values are continuous across the extent boundary."""
        center = np.array([self.sdf_data.center[0], self.sdf_data.center[1], self.sdf_data.center[2]])
        half_ext = np.array(
            [self.sdf_data.half_extents[0], self.sdf_data.half_extents[1], self.sdf_data.half_extents[2]]
        )

        # Sample along a line crossing the extent boundary
        epsilon = 0.01
        test_points = np.array(
            [
                center + np.array([half_ext[0] - epsilon, 0.0, 0.0]),  # Just inside
                center + np.array([half_ext[0], 0.0, 0.0]),  # At boundary
                center + np.array([half_ext[0] + epsilon, 0.0, 0.0]),  # Just outside
            ],
            dtype=np.float32,
        )

        values = sample_extrapolated_at_points(self.sdf_data, test_points)

        # Values should be monotonically increasing (moving away from mesh surface)
        self.assertLess(
            values[0],
            values[1] + 0.02,  # Small tolerance for numerical precision
            f"Value inside ({values[0]}) should be less than at boundary ({values[1]})",
        )
        self.assertLess(
            values[1],
            values[2] + 0.02,
            f"Value at boundary ({values[1]}) should be less than outside ({values[2]})",
        )

    def test_extrapolated_gradient_inside_narrow_band(self):
        """Test that gradients inside narrow band match sparse grid gradients."""
        test_points = np.array(
            [
                [0.45, 0.0, 0.0],  # Just inside +X face
                [0.0, 0.45, 0.0],  # Just inside +Y face
            ],
            dtype=np.float32,
        )

        ext_values, ext_gradients = sample_extrapolated_with_gradient(self.sdf_data, test_points)
        direct_values, direct_gradients = sample_sdf_with_gradient(self.sparse_volume, test_points)

        for i in range(len(test_points)):
            # Values should match
            self.assertAlmostEqual(
                ext_values[i],
                direct_values[i],
                places=4,
                msg=f"Point {i}: extrapolated value ({ext_values[i]}) should match sparse ({direct_values[i]})",
            )
            # Gradients should match
            for j in range(3):
                self.assertAlmostEqual(
                    ext_gradients[i][j],
                    direct_gradients[i][j],
                    places=3,
                    msg=f"Point {i}, component {j}: gradient mismatch",
                )

    def test_extrapolated_gradient_outside_extent(self):
        """Test that gradients outside extent point toward the boundary."""
        center = np.array([self.sdf_data.center[0], self.sdf_data.center[1], self.sdf_data.center[2]])
        half_ext = np.array(
            [self.sdf_data.half_extents[0], self.sdf_data.half_extents[1], self.sdf_data.half_extents[2]]
        )

        # Points outside extent along each axis
        outside_distance = 0.5
        test_points = np.array(
            [
                center + np.array([half_ext[0] + outside_distance, 0.0, 0.0]),  # Outside +X
                center + np.array([-half_ext[0] - outside_distance, 0.0, 0.0]),  # Outside -X
                center + np.array([0.0, half_ext[1] + outside_distance, 0.0]),  # Outside +Y
            ],
            dtype=np.float32,
        )

        _values, gradients = sample_extrapolated_with_gradient(self.sdf_data, test_points)

        # Gradients should point outward (toward the query point from boundary)
        # For point outside +X, gradient should point in +X direction
        self.assertGreater(
            gradients[0][0],
            0.5,
            f"Gradient outside +X should point in +X direction, got {gradients[0]}",
        )
        # For point outside -X, gradient should point in -X direction
        self.assertLess(
            gradients[1][0],
            -0.5,
            f"Gradient outside -X should point in -X direction, got {gradients[1]}",
        )
        # For point outside +Y, gradient should point in +Y direction
        self.assertGreater(
            gradients[2][1],
            0.5,
            f"Gradient outside +Y should point in +Y direction, got {gradients[2]}",
        )

    def test_extrapolated_always_less_than_background(self):
        """Test that extrapolated values are always less than background value."""
        center = np.array([self.sdf_data.center[0], self.sdf_data.center[1], self.sdf_data.center[2]])
        half_ext = np.array(
            [self.sdf_data.half_extents[0], self.sdf_data.half_extents[1], self.sdf_data.half_extents[2]]
        )

        # Sample at various points: inside, at boundary, and outside
        test_points = np.array(
            [
                center,  # Center
                center + half_ext * 0.5,  # Inside
                center + half_ext * 0.99,  # Near boundary
                center + half_ext * 1.5,  # Outside
                center + half_ext * 2.0,  # Far outside
            ],
            dtype=np.float32,
        )

        values = sample_extrapolated_at_points(self.sdf_data, test_points)

        for i, value in enumerate(values):
            self.assertLess(
                value,
                self.sdf_data.background_value,
                f"Point {i}: extrapolated value ({value}) should be less than background ({self.sdf_data.background_value})",
            )


class TestMeshSDFCollisionFlag(unittest.TestCase):
    """Test per-shape SDF generation behavior."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        wp.init()

    def setUp(self):
        """Set up test fixtures."""
        self.half_extents = (0.5, 0.5, 0.5)
        self.mesh = create_box_mesh(self.half_extents)

    def test_sdf_max_resolution_raises_on_cpu(self):
        """Test that sdf_max_resolution != None raises ValueError on CPU."""
        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.sdf_max_resolution = 64  # Request SDF generation

        # Add a mesh shape to trigger SDF computation
        builder.add_body()
        builder.add_shape_mesh(body=-1, mesh=self.mesh, cfg=cfg)

        # Should raise ValueError when finalizing on CPU
        with self.assertRaises(ValueError) as context:
            builder.finalize(device="cpu")

        self.assertIn("CUDA", str(context.exception))
        self.assertIn("sdf_max_resolution", str(context.exception))

    def test_sdf_disabled_works_on_cpu(self):
        """Test that sdf_max_resolution=None (default) works on CPU."""
        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.sdf_max_resolution = None  # No SDF generation (default)

        # Add a mesh shape
        builder.add_body()
        builder.add_shape_mesh(body=-1, mesh=self.mesh, cfg=cfg)

        # Should NOT raise when finalizing on CPU
        model = builder.finalize(device="cpu")

        # SDF data array should still exist (one empty entry per shape)
        self.assertEqual(model.shape_sdf_data.shape[0], 1)
        # But the SDF pointer should be zero (no SDF generated)
        self.assertEqual(model.shape_sdf_data.numpy()[0]["sparse_sdf_ptr"], 0)

    @unittest.skipUnless(_cuda_available, "Requires CUDA device")
    def test_sdf_enabled_works_on_gpu(self):
        """Test that sdf_max_resolution != None works on GPU."""
        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.sdf_max_resolution = 64  # Request SDF generation

        # Add a mesh shape
        builder.add_body()
        builder.add_shape_mesh(body=-1, mesh=self.mesh, cfg=cfg)

        # Should work on GPU
        model = builder.finalize(device="cuda:0")

        # SDF data should be populated
        self.assertGreater(model.shape_sdf_data.shape[0], 0)
        # SDF pointer should be non-zero (SDF was generated)
        self.assertNotEqual(model.shape_sdf_data.numpy()[0]["sparse_sdf_ptr"], 0)


class TestSDFNonUniformScaleBrickPyramid(unittest.TestCase):
    """Test SDF collision with non-uniform scaling using a brick pyramid."""

    pass


def test_brick_pyramid_stability(test, device):
    """Test that a pyramid of non-uniformly scaled mesh bricks remains stable.

    Creates a small pyramid using a unit cube mesh with non-uniform scale
    applied to make brick-shaped objects. Verifies that the top brick
    stays in place after simulation.
    """
    builder = newton.ModelBuilder()
    builder.rigid_contact_margin = 0.005

    # Add ground plane
    builder.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

    # Create unit cube mesh (will be scaled non-uniformly)
    cube_mesh = create_box_mesh((0.5, 0.5, 0.5))

    # Configure shape with SDF enabled
    mesh_cfg = newton.ModelBuilder.ShapeConfig()
    mesh_cfg.sdf_max_resolution = 32

    # Brick dimensions via non-uniform scale
    brick_scale = (0.4, 0.2, 0.1)  # Wide, medium depth, thin
    brick_width = brick_scale[0]
    brick_height = brick_scale[2]
    gap = 0.005

    # Build a small 3-row pyramid
    pyramid_rows = 3
    for row in range(pyramid_rows):
        bricks_in_row = pyramid_rows - row
        z_pos = brick_height / 2 + row * (brick_height + gap)

        row_width = bricks_in_row * brick_width + (bricks_in_row - 1) * gap
        start_x = -row_width / 2 + brick_width / 2

        for i in range(bricks_in_row):
            x_pos = start_x + i * (brick_width + gap)

            body = builder.add_body(xform=wp.transform(wp.vec3(x_pos, 0.0, z_pos), wp.quat_identity()))
            builder.add_shape_mesh(
                body,
                mesh=cube_mesh,
                scale=brick_scale,  # Non-uniform scale
                cfg=mesh_cfg,
            )
            joint = builder.add_joint_free(body)
            builder.add_articulation([joint])

    # Finalize model on the specified CUDA device
    model = builder.finalize(device=device)

    # Get initial position of top brick (last body added)
    top_brick_body = model.body_count - 1
    initial_state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, initial_state)
    initial_top_pos = initial_state.body_q.numpy()[top_brick_body][:3].copy()

    # Create collision pipeline and solver
    collision_pipeline = newton.CollisionPipelineUnified(
        model,
        broad_phase_mode=newton.BroadPhaseMode.NXN,
    )
    solver = newton.solvers.SolverXPBD(model, iterations=10, rigid_contact_relaxation=0.8)

    # Simulate for a short time
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    dt = 1.0 / 60.0 / 4
    num_steps = 120  # ~0.5 seconds

    for _ in range(num_steps):
        state_0.clear_forces()
        contacts = model.collide(state_0, collision_pipeline=collision_pipeline)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

    # Get final position of top brick
    final_top_pos = state_0.body_q.numpy()[top_brick_body][:3]

    # Top brick should not have fallen significantly
    # Allow small settling but it should stay roughly in place
    z_drop = initial_top_pos[2] - final_top_pos[2]
    xy_drift = np.sqrt((final_top_pos[0] - initial_top_pos[0]) ** 2 + (final_top_pos[1] - initial_top_pos[1]) ** 2)

    # The top brick should settle slightly but not fall through
    test.assertLess(
        z_drop,
        brick_height,  # Should not drop more than its own height
        f"Top brick dropped too much: {z_drop:.4f} (max allowed: {brick_height})",
    )
    test.assertLess(
        xy_drift,
        brick_width * 0.5,  # Should not drift too far horizontally
        f"Top brick drifted too far: {xy_drift:.4f}",
    )

    # Final Z should still be positive (above ground)
    test.assertGreater(
        final_top_pos[2],
        0.0,
        f"Top brick fell through ground: z = {final_top_pos[2]}",
    )


# Register CUDA-only tests using the standard pattern
cuda_devices = get_cuda_test_devices()

add_function_test(
    TestSDFNonUniformScaleBrickPyramid,
    "test_brick_pyramid_stability",
    test_brick_pyramid_stability,
    devices=cuda_devices,
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
