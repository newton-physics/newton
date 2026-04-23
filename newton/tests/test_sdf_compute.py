# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test compute_sdf_from_shape function for SDF generation.

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
from newton._src.geometry.sdf_texture import TextureSDFData, texture_sample_sdf
from newton._src.geometry.sdf_utils import (
    SDF,
    SDFData,
    compute_isomesh,
    compute_offset_mesh,
    compute_offset_mesh_analytical,
    compute_sdf_from_shape,
    sample_sdf_extrapolated,
    sample_sdf_grad_extrapolated,
)
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
    points: wp.array[wp.vec3],
    values: wp.array[wp.float32],
):
    tid = wp.tid()
    point = points[tid]
    index_pos = wp.volume_world_to_index(volume_id, point)
    values[tid] = wp.volume_sample_f(volume_id, index_pos, wp.Volume.LINEAR)


# Warp kernel for sampling SDF gradients
@wp.kernel
def sample_sdf_gradient_kernel(
    volume_id: wp.uint64,
    points: wp.array[wp.vec3],
    values: wp.array[wp.float32],
    gradients: wp.array[wp.vec3],
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

    return values.numpy(), gradients.numpy()


@unittest.skipUnless(_cuda_available, "Texture SDF requires CUDA device")
class TestComputeSDF(unittest.TestCase):
    """Test mesh SDF construction via the texture-based path.

    On CUDA, ``SDF.create_from_mesh`` builds only a texture SDF (no NanoVDB
    volumes).  These tests validate sign, distance, and extent correctness
    using the texture sampling path.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        wp.init()
        cls.half_extents = (0.5, 0.5, 0.5)
        cls.mesh = create_box_mesh(cls.half_extents)

    def _build_sdf(self, **kwargs):
        """Helper: build SDF from mesh using the texture path."""
        return SDF.create_from_mesh(self.mesh, **kwargs)

    def test_sdf_returns_valid_data(self):
        """Test that create_from_mesh returns valid texture SDF data."""
        sdf = self._build_sdf()
        self.assertIsNotNone(sdf.texture_data)
        # NanoVDB volumes are not built for mesh SDFs on CUDA.
        self.assertIsNone(sdf.sparse_volume)
        self.assertIsNone(sdf.coarse_volume)

    def test_sdf_extents_are_valid(self):
        """Test that SDF extents match the mesh bounds."""
        sdf = self._build_sdf(margin=0.05)
        td = sdf.texture_data

        lower = np.array([td.sdf_box_lower[0], td.sdf_box_lower[1], td.sdf_box_lower[2]])
        upper = np.array([td.sdf_box_upper[0], td.sdf_box_upper[1], td.sdf_box_upper[2]])
        extent_size = upper - lower

        # Each axis extent should be at least 2*(half_extent + margin)
        expected_min_size = 2.0 * (min(self.half_extents) + 0.05)
        for i in range(3):
            self.assertGreaterEqual(extent_size[i], expected_min_size - 0.02, f"Extent axis {i} too small")

    def test_sdf_values_near_surface(self):
        """Test that texture SDF values near the surface have correct sign and magnitude."""
        sdf = self._build_sdf(narrow_band_range=(-0.1, 0.1))

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

        values = _sample_texture_sdf_at_points(sdf, test_points)

        self.assertLess(float(values[0]), 0.0, "Inside +X face should be negative")
        self.assertGreater(float(values[1]), 0.0, "Outside +X face should be positive")
        self.assertLess(float(values[2]), 0.0, "Inside +Y face should be negative")
        self.assertLess(float(values[3]), 0.0, "Inside +Z face should be negative")
        self.assertLess(float(values[4]), 0.0, "Inside -X face should be negative")

    def test_sdf_values_inside_extent(self):
        """Test that SDF values at the center and interior points are negative."""
        sdf = self._build_sdf()

        test_points = np.array(
            [
                [0.0, 0.0, 0.0],  # Center
                [0.2, 0.2, 0.2],  # Interior
                [-0.2, -0.2, -0.2],
            ],
            dtype=np.float32,
        )

        values = _sample_texture_sdf_at_points(sdf, test_points)
        for i, value in enumerate(values):
            self.assertLess(float(value), 0.0, f"Interior point {i} should be negative, got {value}")

    def test_sdf_negative_inside_mesh(self):
        """Test that SDF values are negative inside the mesh."""
        sdf = self._build_sdf()

        test_points = np.array(
            [
                [0.45, 0.0, 0.0],  # Just inside +X face
                [0.0, 0.0, 0.0],  # Center
            ],
            dtype=np.float32,
        )

        values = _sample_texture_sdf_at_points(sdf, test_points)
        self.assertLess(float(values[0]), 0.0, "Just inside surface should be negative")
        self.assertLess(float(values[1]), 0.0, "Center should be negative")

    def test_sdf_positive_outside_mesh(self):
        """Test that SDF values are positive outside the mesh."""
        sdf = self._build_sdf()

        outside_point = np.array([[0.6, 0.0, 0.0]], dtype=np.float32)
        values = _sample_texture_sdf_at_points(sdf, outside_point)
        self.assertGreater(float(values[0]), 0.0, "Outside mesh should be positive")

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

        sdf = SDF.create_from_mesh(
            inverted_sphere,
            max_resolution=32,
            narrow_band_range=(-0.2, 0.2),
        )
        self.assertIsNotNone(sdf.texture_data)

        inside_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.2, 0.0],
                [0.1, 0.1, 0.1],
            ],
            dtype=np.float32,
        )

        inside_values = _sample_texture_sdf_at_points(sdf, inside_points)
        for i, (point, value) in enumerate(zip(inside_points, inside_values, strict=False)):
            self.assertLess(float(value), 0.0, f"Point {i} at {point} should be inside (negative), got {value}")

        near_inside_points = np.array(
            [
                [radius - 0.05, 0.0, 0.0],
                [0.0, radius - 0.05, 0.0],
                [0.0, 0.0, radius - 0.05],
            ],
            dtype=np.float32,
        )

        near_inside_values = _sample_texture_sdf_at_points(sdf, near_inside_points)
        for i, (point, value) in enumerate(zip(near_inside_points, near_inside_values, strict=False)):
            self.assertLess(float(value), 0.0, f"Point {i} at {point} should be inside (negative), got {value}")

        outside_offset = 0.02
        outside_points = np.array(
            [
                [radius + outside_offset, 0.0, 0.0],
                [0.0, radius + outside_offset, 0.0],
                [0.0, 0.0, radius + outside_offset],
                [-(radius + outside_offset), 0.0, 0.0],
                [0.0, -(radius + outside_offset), 0.0],
                [0.0, 0.0, -(radius + outside_offset)],
            ],
            dtype=np.float32,
        )

        outside_values = _sample_texture_sdf_at_points(sdf, outside_points)
        for i, (point, value) in enumerate(zip(outside_points, outside_values, strict=False)):
            self.assertGreater(float(value), 0.0, f"Point {i} at {point} should be outside (positive), got {value}")


@unittest.skipUnless(_cuda_available, "Texture SDF requires CUDA device")
class TestComputeSDFGridSampling(unittest.TestCase):
    """Test texture SDF by sampling on a grid of points."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        wp.init()
        cls.half_extents = (0.5, 0.5, 0.5)
        cls.mesh = create_box_mesh(cls.half_extents)

    def test_grid_sampling_near_surface(self):
        """Sample texture SDF on a grid near the surface and verify sign correctness."""
        sdf = SDF.create_from_mesh(self.mesh)

        test_points_inside = []
        test_points_outside = []
        for j in range(5):
            for k in range(5):
                y = (j / 4.0 - 0.5) * 0.8
                z = (k / 4.0 - 0.5) * 0.8
                test_points_inside.append([0.45, y, z])
                test_points_outside.append([0.55, y, z])

        inside_np = np.array(test_points_inside, dtype=np.float32)
        outside_np = np.array(test_points_outside, dtype=np.float32)

        vals_in = _sample_texture_sdf_at_points(sdf, inside_np)
        vals_out = _sample_texture_sdf_at_points(sdf, outside_np)

        for i, (pt, v) in enumerate(zip(test_points_inside, vals_in, strict=False)):
            self.assertLess(float(v), 0.0, f"Inside point {i} at {pt} should be negative, got {v}")
        for i, (pt, v) in enumerate(zip(test_points_outside, vals_out, strict=False)):
            self.assertGreater(float(v), 0.0, f"Outside point {i} at {pt} should be positive, got {v}")

    def test_grid_sampling_interior(self):
        """Sample texture SDF on an interior 5^3 grid and verify all points are negative."""
        sdf = SDF.create_from_mesh(self.mesh)

        test_points = []
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    x = (i / 4.0 - 0.5) * 0.8
                    y = (j / 4.0 - 0.5) * 0.8
                    z = (k / 4.0 - 0.5) * 0.8
                    test_points.append([x, y, z])

        test_np = np.array(test_points, dtype=np.float32)
        values = _sample_texture_sdf_at_points(sdf, test_np)

        for i, (pt, v) in enumerate(zip(test_points, values, strict=False)):
            self.assertLess(float(v), 0.0, f"Interior point {i} at {pt} should be negative, got {v}")


@wp.kernel
def sample_sdf_extrapolated_kernel(
    sdf_data: SDFData,
    points: wp.array[wp.vec3],
    values: wp.array[wp.float32],
):
    """Kernel to test sample_sdf_extrapolated function."""
    tid = wp.tid()
    values[tid] = sample_sdf_extrapolated(sdf_data, points[tid])


@wp.kernel
def sample_sdf_grad_extrapolated_kernel(
    sdf_data: SDFData,
    points: wp.array[wp.vec3],
    values: wp.array[wp.float32],
    gradients: wp.array[wp.vec3],
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

    return values.numpy(), gradients.numpy()


@unittest.skipUnless(_cuda_available, "wp.Volume requires CUDA device")
class TestSDFExtrapolation(unittest.TestCase):
    """Test the SDF extrapolation functions (NanoVDB utility).

    These tests exercise :func:`sample_sdf_extrapolated` and
    :func:`sample_sdf_grad_extrapolated`, which rely on NanoVDB volumes.
    Since the production mesh SDF path no longer builds NanoVDB, we build
    them explicitly via :func:`_compute_sdf_from_shape_impl`.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        from newton._src.geometry.sdf_utils import _compute_sdf_from_shape_impl  # noqa: PLC0415

        wp.init()
        cls.half_extents = (0.5, 0.5, 0.5)
        cls.mesh = create_box_mesh(cls.half_extents)
        cls.sdf_data, cls.sparse_volume, cls.coarse_volume, _ = _compute_sdf_from_shape_impl(
            shape_type=GeoType.MESH,
            shape_geo=cls.mesh,
            shape_scale=(1.0, 1.0, 1.0),
            shape_margin=0.0,
            narrow_band_distance=(-0.1, 0.1),
            margin=0.05,
            max_resolution=64,
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
        cls.half_extents = (0.5, 0.5, 0.5)
        cls.mesh = create_box_mesh(cls.half_extents)

    def test_mesh_cfg_sdf_conflict_raises(self):
        """Mesh shapes should reject cfg.sdf_* and require mesh.build_sdf()."""
        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.sdf_max_resolution = 64
        builder.add_body()
        with self.assertRaises(ValueError) as context:
            builder.add_shape_mesh(body=-1, mesh=self.mesh, cfg=cfg)
        self.assertIn("mesh.build_sdf", str(context.exception))

    def test_mesh_cfg_sdf_narrow_band_conflict_raises(self):
        """Mesh shapes should reject cfg.sdf_narrow_band_range overrides."""
        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.sdf_narrow_band_range = (-0.2, 0.2)
        builder.add_body()
        with self.assertRaises(ValueError) as context:
            builder.add_shape_mesh(body=-1, mesh=self.mesh, cfg=cfg)
        self.assertIn("mesh.build_sdf", str(context.exception))

    def test_sdf_disabled_works_on_cpu(self):
        """Mesh without mesh.sdf should still finalize on CPU."""
        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig()

        # Add a mesh shape
        builder.add_body()
        builder.add_shape_mesh(body=-1, mesh=self.mesh, cfg=cfg)

        # Should NOT raise when finalizing on CPU
        model = builder.finalize(device="cpu")

        # No compact SDF entry should exist for this shape
        self.assertEqual(int(model.shape_sdf_index.numpy()[0]), -1)
        self.assertEqual(model.texture_sdf_data.shape[0], 0)

    @unittest.skipUnless(_cuda_available, "Requires CUDA device")
    def test_mesh_build_sdf_works_on_gpu(self):
        """Mesh SDF built via mesh.build_sdf() should be used by builder."""
        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig()
        mesh = create_box_mesh(self.half_extents)
        mesh.build_sdf(max_resolution=64)

        # Add a mesh shape
        builder.add_body()
        builder.add_shape_mesh(body=-1, mesh=mesh, cfg=cfg)

        # Should work on GPU
        model = builder.finalize(device="cuda:0")

        # Texture SDF data should be populated in compact table
        sdf_idx = int(model.shape_sdf_index.numpy()[0])
        self.assertGreaterEqual(sdf_idx, 0)
        self.assertGreater(model.texture_sdf_data.shape[0], sdf_idx)

    @unittest.skipUnless(_cuda_available, "Requires CUDA device")
    def test_mesh_build_sdf_guard_and_clear(self):
        """build_sdf() should guard overwrite until clear_sdf() is called."""
        mesh = create_box_mesh((0.2, 0.2, 0.2))
        mesh.build_sdf(max_resolution=32)
        with self.assertRaises(RuntimeError):
            mesh.build_sdf(max_resolution=32)
        mesh.clear_sdf()
        mesh.build_sdf(max_resolution=32)
        self.assertIsNotNone(mesh.sdf)

    @unittest.skipUnless(_cuda_available, "Requires CUDA device")
    def test_sdf_create_from_data_roundtrip(self):
        """Round-trip SDF reconstruction from generated data."""
        mesh = create_box_mesh((0.3, 0.2, 0.1))
        mesh.build_sdf(max_resolution=32)
        sdf = mesh.sdf
        assert sdf is not None

        rebuilt = newton.SDF.create_from_data(
            sparse_volume=sdf.sparse_volume,
            coarse_volume=sdf.coarse_volume,
            block_coords=sdf.block_coords,
            center=tuple(sdf.data.center),
            half_extents=tuple(sdf.data.half_extents),
            background_value=float(sdf.data.background_value),
            scale_baked=bool(sdf.data.scale_baked),
        )
        # On CUDA, sparse/coarse volumes are None; ptrs should both be 0.
        self.assertEqual(int(rebuilt.data.sparse_sdf_ptr), int(sdf.data.sparse_sdf_ptr))
        self.assertEqual(int(rebuilt.data.coarse_sdf_ptr), int(sdf.data.coarse_sdf_ptr))
        np.testing.assert_allclose(np.array(rebuilt.data.sparse_voxel_size), np.array(sdf.data.sparse_voxel_size))
        np.testing.assert_allclose(np.array(rebuilt.data.coarse_voxel_size), np.array(sdf.data.coarse_voxel_size))

    @unittest.skipUnless(_cuda_available, "Requires CUDA device")
    def test_sdf_static_create_methods(self):
        """SDF static creation methods should construct valid SDF handles with texture data."""
        mesh = create_box_mesh((0.3, 0.2, 0.1))

        sdf_from_mesh = newton.SDF.create_from_mesh(mesh, max_resolution=32)
        self.assertIsNotNone(sdf_from_mesh.texture_data)

        sdf_from_points = newton.SDF.create_from_points(mesh.vertices, mesh.indices, max_resolution=32)
        self.assertIsNotNone(sdf_from_points.texture_data)

        rebuilt = newton.SDF.create_from_data(
            sparse_volume=sdf_from_mesh.sparse_volume,
            coarse_volume=sdf_from_mesh.coarse_volume,
            block_coords=sdf_from_mesh.block_coords,
            center=tuple(sdf_from_mesh.data.center),
            half_extents=tuple(sdf_from_mesh.data.half_extents),
            background_value=float(sdf_from_mesh.data.background_value),
            scale_baked=bool(sdf_from_mesh.data.scale_baked),
        )
        self.assertEqual(int(rebuilt.data.sparse_sdf_ptr), int(sdf_from_mesh.data.sparse_sdf_ptr))

    def test_standalone_sdf_shape_api_removed(self):
        """GeoType.SDF and add_shape_sdf should not exist."""
        self.assertFalse(hasattr(newton.GeoType, "SDF"))
        self.assertFalse(hasattr(newton.ModelBuilder, "add_shape_sdf"))


class TestSDFPublicApi(unittest.TestCase):
    """Test public API shape for SDF creators."""

    def test_top_level_sdf_exported(self):
        """Top-level package should expose SDF as newton.SDF."""
        self.assertTrue(hasattr(newton, "SDF"))
        self.assertFalse(hasattr(newton.geometry, "SDF"))

    def test_module_level_sdf_creators_removed(self):
        """Module-level SDF creators should not be exposed in public API."""
        self.assertFalse(hasattr(newton.geometry, "create_sdf_from_mesh"))
        self.assertFalse(hasattr(newton.geometry, "create_sdf_from_data"))

    @unittest.skipUnless(_cuda_available, "Requires CUDA device")
    def test_hydroelastic_primitive_generates_sdf_on_gpu(self):
        """Hydroelastic primitives should generate per-shape SDF data."""
        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.sdf_max_resolution = 32
        cfg.is_hydroelastic = True

        body = builder.add_body()
        builder.add_shape_box(body=body, hx=0.5, hy=0.4, hz=0.3, cfg=cfg)

        model = builder.finalize(device="cuda:0")
        sdf_idx = int(model.shape_sdf_index.numpy()[0])
        self.assertGreaterEqual(sdf_idx, 0)
        self.assertGreater(model.texture_sdf_data.shape[0], sdf_idx)


@unittest.skipUnless(_cuda_available, "wp.Volume requires CUDA device")
class TestComputeOffsetMesh(unittest.TestCase):
    """Test compute_offset_mesh for various shapes and offset magnitudes.

    Validates that the offset isosurface is geometrically correct even when
    the offset pushes the surface well beyond the original shape AABB.
    """

    device = "cuda:0"

    @staticmethod
    def _analytical_sdf(v, shape_type, shape_scale):
        """Evaluate analytical SDF for a primitive at point v using NumPy."""
        if shape_type == GeoType.SPHERE:
            return np.linalg.norm(v) - shape_scale[0]
        if shape_type == GeoType.BOX:
            q = np.abs(v) - np.array(shape_scale[:3])
            return float(np.linalg.norm(np.maximum(q, 0.0)) + min(max(q[0], q[1], q[2]), 0.0))
        if shape_type == GeoType.CAPSULE:
            r, hh = shape_scale[0], shape_scale[1]
            pz = max(-hh, min(float(v[2]), hh))
            return np.linalg.norm(v - np.array([0, 0, pz])) - r
        if shape_type == GeoType.CYLINDER:
            r, hh = shape_scale[0], shape_scale[1]
            dxy = np.linalg.norm(v[:2]) - r
            dz = abs(v[2]) - hh
            return float(np.linalg.norm(np.maximum([dxy, dz], 0.0)) + min(max(dxy, dz), 0.0))
        return None

    def _assert_vertices_at_offset(self, mesh, shape_type, shape_scale, offset, atol=None):
        """Assert every vertex of *mesh* is approximately *offset* from the base surface.

        For each vertex **v**, computes the analytical SDF of the un-inflated
        shape.  That distance should be approximately equal to *offset* because
        ``compute_offset_mesh`` bakes the offset into the SDF volume so the
        zero-isosurface sits where ``sdf(v) == offset``.
        """
        if atol is None:
            atol = offset * 0.15 + 0.02

        verts = mesh.vertices
        self.assertGreater(len(verts), 0, "Offset mesh has no vertices")
        max_err = 0.0
        for v in verts:
            d = self._analytical_sdf(v, shape_type, shape_scale)
            if d is None:
                continue
            err = abs(d - offset)
            max_err = max(max_err, err)

        self.assertLess(
            max_err,
            atol,
            f"Max vertex distance error {max_err:.4f} exceeds tolerance {atol:.4f} "
            f"for shape {shape_type}, scale {shape_scale}, offset {offset}",
        )

    def test_box_small_offset(self):
        """Box with a small offset that stays within the original AABB."""
        mesh = compute_offset_mesh(GeoType.BOX, shape_scale=(0.5, 0.35, 0.25), offset=0.05, device=self.device)
        self.assertIsNotNone(mesh)
        self.assertGreater(mesh.vertices.shape[0], 0)
        self._assert_vertices_at_offset(mesh, GeoType.BOX, (0.5, 0.35, 0.25), 0.05)

    def test_box_large_offset(self):
        """Box with an offset larger than its smallest half-extent."""
        mesh = compute_offset_mesh(GeoType.BOX, shape_scale=(0.5, 0.35, 0.25), offset=0.5, device=self.device)
        self.assertIsNotNone(mesh)
        self.assertGreater(mesh.vertices.shape[0], 0)
        self._assert_vertices_at_offset(mesh, GeoType.BOX, (0.5, 0.35, 0.25), 0.5)

    def test_box_very_large_offset(self):
        """Box with an offset much larger than the shape itself."""
        mesh = compute_offset_mesh(GeoType.BOX, shape_scale=(0.2, 0.2, 0.2), offset=1.0, device=self.device)
        self.assertIsNotNone(mesh)
        self.assertGreater(mesh.vertices.shape[0], 0)
        self._assert_vertices_at_offset(mesh, GeoType.BOX, (0.2, 0.2, 0.2), 1.0)

        extent = np.max(np.abs(mesh.vertices), axis=0)
        for i in range(3):
            self.assertGreater(
                extent[i],
                0.2 + 0.8,
                f"Offset mesh extent along axis {i} ({extent[i]:.3f}) should exceed shape_scale + offset = {0.2 + 1.0}",
            )

    def test_sphere_large_offset(self):
        """Sphere with a large offset — surface should be roughly spherical."""
        r = 0.3
        off = 0.7
        mesh = compute_offset_mesh(GeoType.SPHERE, shape_scale=(r, r, r), offset=off, device=self.device)
        self.assertIsNotNone(mesh)
        self.assertGreater(mesh.vertices.shape[0], 0)
        dists = np.linalg.norm(mesh.vertices, axis=1)
        expected_radius = r + off
        np.testing.assert_allclose(dists, expected_radius, atol=0.05)

    def test_capsule_large_offset(self):
        """Capsule with offset exceeding its radius."""
        r, hh = 0.2, 0.4
        off = 0.6
        mesh = compute_offset_mesh(GeoType.CAPSULE, shape_scale=(r, hh, 0.0), offset=off, device=self.device)
        self.assertIsNotNone(mesh)
        self._assert_vertices_at_offset(mesh, GeoType.CAPSULE, (r, hh, 0.0), off)

    def test_cylinder_large_offset(self):
        """Cylinder with offset exceeding its radius."""
        r, hh = 0.3, 0.5
        off = 0.8
        mesh = compute_offset_mesh(GeoType.CYLINDER, shape_scale=(r, hh, 0.0), offset=off, device=self.device)
        self.assertIsNotNone(mesh)
        self._assert_vertices_at_offset(mesh, GeoType.CYLINDER, (r, hh, 0.0), off)

    def test_plane_returns_none(self):
        """Plane should return None (not supported)."""
        mesh = compute_offset_mesh(GeoType.PLANE, shape_scale=(1.0, 1.0, 1.0), offset=0.1, device=self.device)
        self.assertIsNone(mesh)

    def test_hfield_returns_none(self):
        """Heightfield should return None (not supported)."""
        mesh = compute_offset_mesh(GeoType.HFIELD, shape_scale=(1.0, 1.0, 1.0), offset=0.1, device=self.device)
        self.assertIsNone(mesh)

    def test_zero_offset(self):
        """Zero offset should produce a mesh approximating the original surface."""
        mesh = compute_offset_mesh(GeoType.SPHERE, shape_scale=(0.5, 0.5, 0.5), offset=0.0, device=self.device)
        self.assertIsNotNone(mesh)
        self.assertGreater(mesh.vertices.shape[0], 0)
        dists = np.linalg.norm(mesh.vertices, axis=1)
        np.testing.assert_allclose(dists, 0.5, atol=0.03)

    def test_mesh_shape_large_offset(self):
        """Mesh (box geometry) with large offset."""
        box_mesh = create_box_mesh((0.3, 0.3, 0.3))
        off = 0.5
        mesh = compute_offset_mesh(GeoType.MESH, shape_geo=box_mesh, offset=off, device=self.device)
        self.assertIsNotNone(mesh)
        extent = np.max(np.abs(mesh.vertices), axis=0)
        for i in range(3):
            self.assertGreater(
                extent[i],
                0.3 + off * 0.5,
                f"Mesh offset extent along axis {i} ({extent[i]:.3f}) too small",
            )


@unittest.skipUnless(_cuda_available, "wp.Volume requires CUDA device")
class TestExtractIsomesh(unittest.TestCase):
    """Test SDF.extract_isomesh and compute_isomesh with isovalue parameter.

    Uses a box mesh with a pre-built SDF.  Validates that every vertex of the
    extracted isosurface sits at the correct signed distance from the original
    box, measured with the analytical box SDF as ground truth.
    """

    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.half_extents = (0.3, 0.3, 0.3)
        cls.mesh = create_box_mesh(cls.half_extents)
        cls.mesh.build_sdf(max_resolution=64)

    @staticmethod
    def _box_sdf(v, hx, hy, hz):
        q = np.abs(v) - np.array([hx, hy, hz])
        return float(np.linalg.norm(np.maximum(q, 0.0)) + min(max(q[0], q[1], q[2]), 0.0))

    def _assert_box_vertices_at_isovalue(self, iso_mesh, isovalue, atol=0.03):
        """Assert every vertex of *iso_mesh* sits at *isovalue* from the box surface."""
        hx, hy, hz = self.half_extents
        verts = iso_mesh.vertices
        errors = np.array([abs(self._box_sdf(v, hx, hy, hz) - isovalue) for v in verts])
        max_err = float(errors.max())
        self.assertLess(
            max_err,
            atol,
            f"Max vertex SDF error {max_err:.4f} exceeds {atol} for isovalue={isovalue} (mean {errors.mean():.4f})",
        )

    def test_extract_isomesh_zero_isovalue(self):
        """extract_isomesh at isovalue=0: every vertex should be on the original box surface."""
        sdf = self.mesh.sdf
        self.assertIsNotNone(sdf)
        result = sdf.extract_isomesh(isovalue=0.0)
        self.assertIsNotNone(result)
        self.assertGreater(result.vertices.shape[0], 0)
        self._assert_box_vertices_at_isovalue(result, 0.0)

    def test_extract_isomesh_positive_isovalue(self):
        """extract_isomesh at positive isovalue: vertices at the inflated distance."""
        sdf = self.mesh.sdf
        self.assertIsNotNone(sdf)
        offset = 0.04
        result = sdf.extract_isomesh(isovalue=offset)
        self.assertIsNotNone(result)
        self.assertGreater(result.vertices.shape[0], 0)
        self._assert_box_vertices_at_isovalue(result, offset)

    def test_extract_isomesh_returns_none_outside_band(self):
        """extract_isomesh at isovalue far outside the narrow band returns None."""
        sdf = self.mesh.sdf
        self.assertIsNotNone(sdf)
        result = sdf.extract_isomesh(isovalue=10.0)
        self.assertIsNone(result)

    def test_compute_offset_mesh_with_prebuilt_sdf(self):
        """compute_offset_mesh via pre-built SDF: vertices at the offset distance."""
        offset = 0.04
        result = compute_offset_mesh(GeoType.MESH, shape_geo=self.mesh, offset=offset)
        self.assertIsNotNone(result)
        self.assertGreater(result.vertices.shape[0], 0)
        self._assert_box_vertices_at_isovalue(result, offset)

    def test_isovalue_changes_surface_consistently(self):
        """Larger isovalue produces a strictly larger mesh than smaller isovalue."""
        sdf = self.mesh.sdf
        self.assertIsNotNone(sdf)
        mesh_small = sdf.extract_isomesh(isovalue=0.02)
        mesh_large = sdf.extract_isomesh(isovalue=0.06)
        self.assertIsNotNone(mesh_small)
        self.assertIsNotNone(mesh_large)
        extent_small = np.max(np.abs(mesh_small.vertices), axis=0)
        extent_large = np.max(np.abs(mesh_large.vertices), axis=0)
        for i in range(3):
            self.assertGreater(
                extent_large[i],
                extent_small[i],
                f"Larger isovalue should produce larger extent on axis {i}: "
                f"{extent_large[i]:.4f} vs {extent_small[i]:.4f}",
            )

    def test_extract_isomesh_texture_with_shape_margin(self):
        """extract_isomesh via texture path correctly handles baked shape_margin."""
        hx, hy, hz = 0.3, 0.3, 0.3
        margin_val = 0.04
        mesh = create_box_mesh((hx, hy, hz))
        sdf = SDF.create_from_mesh(mesh, shape_margin=margin_val, max_resolution=64)
        self.assertIsNotNone(sdf)
        self.assertEqual(sdf.shape_margin, margin_val)
        self.assertIsNotNone(sdf.texture_data)

        result = sdf.extract_isomesh(isovalue=margin_val)
        self.assertIsNotNone(result)
        self.assertGreater(result.vertices.shape[0], 0)

        errors = np.array([abs(self._box_sdf(v, hx, hy, hz) - margin_val) for v in result.vertices])
        max_err = float(errors.max())
        self.assertLess(
            max_err,
            0.03,
            f"Texture path with shape_margin: max vertex error {max_err:.4f} "
            f"exceeds tolerance 0.03 (shape_margin={margin_val}). "
            f"Mean error: {float(errors.mean()):.4f}",
        )


@unittest.skipUnless(_cuda_available, "wp.Volume requires CUDA device")
class TestComputeOffsetMeshAdditionalPrimitives(unittest.TestCase):
    """Test compute_offset_mesh for primitives not covered by TestComputeOffsetMesh.

    Adds analytical SDF references for ellipsoid and cone, validating vertex
    positions with the same rigour as ``TestComputeOffsetMesh._assert_vertices_at_offset``.
    """

    device = "cuda:0"

    @staticmethod
    def _analytical_sdf(v, shape_type, shape_scale):
        """Evaluate analytical SDF for a primitive at point *v*."""
        if shape_type == GeoType.ELLIPSOID:
            rx, ry, rz = shape_scale[:3]
            eps = 1e-8
            r = np.array([max(abs(rx), eps), max(abs(ry), eps), max(abs(rz), eps)])
            q0 = v / r
            q1 = v / (r * r)
            k0 = np.linalg.norm(q0)
            k1 = np.linalg.norm(q1)
            if k1 > eps:
                return float(k0 * (k0 - 1.0) / k1)
            return float(-min(r))
        if shape_type == GeoType.CONE:
            bottom_r, hh = shape_scale[0], shape_scale[1]
            top_r = 0.0
            # cone SDF with Z up-axis
            r_xy = np.linalg.norm(v[:2])
            q = np.array([r_xy, v[2]])
            k1 = np.array([top_r, hh])
            k2 = np.array([top_r - bottom_r, 2.0 * hh])
            if q[1] < 0.0:
                ca = np.array([q[0] - min(q[0], bottom_r), abs(q[1]) - hh])
            else:
                ca = np.array([q[0] - min(q[0], top_r), abs(q[1]) - hh])
            denom = np.dot(k2, k2)
            t = 0.0
            if denom > 0.0:
                t = float(np.clip(np.dot(k1 - q, k2) / denom, 0.0, 1.0))
            cb = q - k1 + k2 * t
            sign = -1.0 if cb[0] < 0.0 and ca[1] < 0.0 else 1.0
            return float(sign * np.sqrt(min(np.dot(ca, ca), np.dot(cb, cb))))
        return None

    def _assert_vertices_at_offset(self, mesh, shape_type, shape_scale, offset, atol=None):
        """Assert every vertex is approximately *offset* from the base surface."""
        if atol is None:
            atol = offset * 0.15 + 0.02
        self.assertGreater(len(mesh.vertices), 0, "Offset mesh has no vertices")
        max_err = 0.0
        for v in mesh.vertices:
            d = self._analytical_sdf(v, shape_type, shape_scale)
            if d is None:
                continue
            max_err = max(max_err, abs(d - offset))
        self.assertLess(
            max_err,
            atol,
            f"Max vertex distance error {max_err:.4f} exceeds {atol:.4f} "
            f"for shape {shape_type}, scale {shape_scale}, offset {offset}",
        )

    def test_ellipsoid_offset(self):
        """Ellipsoid offset mesh: every vertex at the correct signed distance."""
        sx, sy, sz = 0.4, 0.3, 0.2
        off = 0.3
        mesh = compute_offset_mesh(GeoType.ELLIPSOID, shape_scale=(sx, sy, sz), offset=off, device=self.device)
        self.assertIsNotNone(mesh)
        self.assertGreater(mesh.vertices.shape[0], 0)
        self._assert_vertices_at_offset(mesh, GeoType.ELLIPSOID, (sx, sy, sz), off)

    def test_cone_offset(self):
        """Cone offset mesh: every vertex at the correct signed distance."""
        r, hh = 0.25, 0.4
        off = 0.3
        mesh = compute_offset_mesh(GeoType.CONE, shape_scale=(r, hh, 0.0), offset=off, device=self.device)
        self.assertIsNotNone(mesh)
        self.assertGreater(mesh.vertices.shape[0], 0)
        self._assert_vertices_at_offset(mesh, GeoType.CONE, (r, hh, 0.0), off)

    def test_compute_offset_mesh_analytical_unsupported_type(self):
        """compute_offset_mesh_analytical returns None for non-analytical types."""
        result = compute_offset_mesh_analytical(GeoType.MESH, shape_scale=(1, 1, 1), offset=0.1, device=self.device)
        self.assertIsNone(result)

    def test_tiny_sphere_offset_mesh(self):
        """A 1 mm sphere should still produce a valid offset mesh with adaptive resolution."""
        r = 0.001
        off = 0.0005
        expected_radius = r + off  # 0.0015
        mesh = compute_offset_mesh(GeoType.SPHERE, shape_scale=(r, r, r), offset=off, device=self.device)
        self.assertIsNotNone(mesh, "Tiny sphere offset mesh should not be None with adaptive resolution")
        self.assertGreater(mesh.vertices.shape[0], 0)
        dists = np.linalg.norm(mesh.vertices, axis=1)
        # Tolerance is 15% of expected radius — tight enough to catch a
        # coarse-grid failure while allowing for marching-cubes discretization.
        np.testing.assert_allclose(dists, expected_radius, atol=expected_radius * 0.15)

    def test_convex_mesh_offset_mesh(self):
        """compute_offset_mesh with CONVEX_MESH produces a geometrically correct offset surface."""
        hx, hy, hz = 0.3, 0.3, 0.3
        box_mesh = create_box_mesh((hx, hy, hz))
        off = 0.1
        mesh = compute_offset_mesh(GeoType.CONVEX_MESH, shape_geo=box_mesh, offset=off, device=self.device)
        self.assertIsNotNone(mesh)
        self.assertGreater(mesh.vertices.shape[0], 0)
        max_err = 0.0
        for v in mesh.vertices:
            q = np.abs(v) - np.array([hx, hy, hz])
            box_dist = float(np.linalg.norm(np.maximum(q, 0.0)) + min(max(q[0], q[1], q[2]), 0.0))
            max_err = max(max_err, abs(box_dist - off))
        self.assertLess(
            max_err,
            off * 0.15 + 0.02,
            f"CONVEX_MESH offset mesh: max vertex distance error {max_err:.4f} for offset {off}",
        )

    def test_compute_isomesh_empty_volume(self):
        """compute_isomesh with isovalue far from any surface returns None."""
        _, sparse_vol, _, _ = compute_sdf_from_shape(
            shape_type=GeoType.BOX,
            shape_scale=(0.2, 0.2, 0.2),
            shape_margin=0.0,
            max_resolution=16,
            narrow_band_distance=(-0.05, 0.05),
        )
        self.assertIsNotNone(sparse_vol)
        result = compute_isomesh(sparse_vol, isovalue=10.0)
        self.assertIsNone(result)


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
    builder.rigid_gap = 0.005

    # Add ground plane
    builder.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

    # Create unit cube mesh (will be scaled non-uniformly)
    cube_mesh = create_box_mesh((0.5, 0.5, 0.5))
    cube_mesh.build_sdf(max_resolution=32, device=device)

    # Configure shape with SDF enabled
    mesh_cfg = newton.ModelBuilder.ShapeConfig()

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
    collision_pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
    )
    contacts = collision_pipeline.contacts()
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
        collision_pipeline.collide(state_0, contacts)
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


class TestMeshIsWatertight(unittest.TestCase):
    """Test the Mesh.is_watertight property."""

    def test_box_mesh_is_watertight(self):
        """A closed box mesh should be watertight."""
        mesh = create_box_mesh((0.5, 0.5, 0.5))
        self.assertTrue(mesh.is_watertight)

    def test_sphere_mesh_is_watertight(self):
        """A subdivided icosphere should be watertight."""
        mesh = create_sphere_mesh(1.0, subdivisions=2)
        self.assertTrue(mesh.is_watertight)

    def test_open_mesh_is_not_watertight(self):
        """A mesh missing a face should not be watertight."""
        verts = np.array(
            [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
            dtype=np.float32,
        )
        # 5 faces of a cube (missing one face -> boundary edges)
        indices = np.array(
            [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 0, 4, 7, 0, 7, 3],
            dtype=np.int32,
        )
        mesh = Mesh(verts, indices, compute_inertia=False)
        self.assertFalse(mesh.is_watertight)

    def test_empty_mesh_is_not_watertight(self):
        """An empty mesh should not be watertight."""
        mesh = Mesh(np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int32), compute_inertia=False)
        self.assertFalse(mesh.is_watertight)

    def test_is_watertight_cache_invalidation(self):
        """Changing vertices or indices should invalidate the cache."""
        mesh = create_box_mesh((0.5, 0.5, 0.5))
        self.assertTrue(mesh.is_watertight)
        # Mutate vertices (keeps topology, should still be watertight)
        mesh.vertices = mesh.vertices * 2.0
        self.assertTrue(mesh.is_watertight)
        # Remove some triangles to break watertightness
        mesh.indices = mesh.indices[:30]  # 5 faces instead of 6
        self.assertFalse(mesh.is_watertight)


@wp.kernel
def _sample_texture_sdf_kernel(
    sdf: TextureSDFData,
    query_points: wp.array[wp.vec3],
    out_values: wp.array[float],
):
    tid = wp.tid()
    out_values[tid] = texture_sample_sdf(sdf, query_points[tid])


def _sample_texture_sdf_at_points(sdf_obj, points_np):
    """Sample texture SDF at world-space points, returns numpy array of SDF values."""
    n = len(points_np)
    query_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device="cuda:0")
    out_wp = wp.zeros(n, dtype=float, device="cuda:0")
    wp.launch(
        _sample_texture_sdf_kernel,
        dim=n,
        inputs=[sdf_obj.texture_data, query_wp, out_wp],
        device="cuda:0",
    )
    return out_wp.numpy()


@unittest.skipUnless(_cuda_available, "wp.Volume requires CUDA device")
class TestSDFWatertightFastPath(unittest.TestCase):
    """Test that the watertight unsigned-BVH fast path produces accurate texture SDF values."""

    device = "cuda:0"

    def test_watertight_sdf_matches_winding_path(self):
        """Texture SDF from watertight fast path should closely match the winding-number path."""
        mesh = create_sphere_mesh(0.5, subdivisions=2)
        self.assertTrue(mesh.is_watertight)

        # Build SDF via normal (winding) path by forcing is_watertight to False
        mesh._is_watertight = False
        sdf_winding = SDF.create_from_mesh(mesh, max_resolution=32, texture_format="float32")
        mesh._is_watertight = None  # reset cache

        # Build SDF via watertight fast path
        self.assertTrue(mesh.is_watertight)
        sdf_fast = SDF.create_from_mesh(mesh, max_resolution=32, texture_format="float32")

        self.assertIsNotNone(sdf_winding.texture_data, "Winding SDF should have texture data")
        self.assertIsNotNone(sdf_fast.texture_data, "Fast SDF should have texture data")

        test_points = np.array(
            [
                [0.0, 0.0, 0.0],  # center (inside)
                [0.3, 0.0, 0.0],  # near surface (inside)
                [0.6, 0.0, 0.0],  # outside
                [0.0, 0.45, 0.0],  # near surface
            ],
            dtype=np.float32,
        )

        vals_winding = _sample_texture_sdf_at_points(sdf_winding, test_points)
        vals_fast = _sample_texture_sdf_at_points(sdf_fast, test_points)

        for i, pt in enumerate(test_points):
            vw = float(vals_winding[i])
            vf = float(vals_fast[i])
            self.assertEqual(
                np.sign(vw),
                np.sign(vf),
                msg=f"Sign mismatch at {pt}: winding={vw:.4f}, fast={vf:.4f}",
            )
            self.assertAlmostEqual(
                vw,
                vf,
                delta=0.1,
                msg=f"SDF mismatch at {pt}: winding={vw:.4f}, fast={vf:.4f}",
            )

        mesh.clear_sdf()

    def test_sphere_watertight_sign_grid(self):
        """Every voxel should have matching sign between fast-path and winding-path."""
        mesh = create_sphere_mesh(0.5, subdivisions=2)

        mesh._is_watertight = False
        sdf_winding = SDF.create_from_mesh(mesh, max_resolution=16, texture_format="float32")
        mesh._is_watertight = None

        mesh._is_watertight = True
        sdf_fast = SDF.create_from_mesh(mesh, max_resolution=16, texture_format="float32")
        mesh._is_watertight = None

        pts = []
        for z in np.linspace(-0.7, 0.7, 8):
            for y in np.linspace(-0.7, 0.7, 8):
                for x in np.linspace(-0.7, 0.7, 8):
                    pts.append([x, y, z])
        pts_np = np.array(pts, dtype=np.float32)

        vals_w = _sample_texture_sdf_at_points(sdf_winding, pts_np)
        vals_f = _sample_texture_sdf_at_points(sdf_fast, pts_np)

        sign_w = np.sign(vals_w)
        sign_f = np.sign(vals_f)
        mismatches = np.sum(sign_w != sign_f)
        self.assertEqual(
            mismatches,
            0,
            f"{mismatches}/{len(pts)} voxels have sign mismatch between unsigned-BVH and winding path",
        )

        mesh.clear_sdf()

    def test_box_watertight_sdf_sign_correctness(self):
        """Watertight SDF for a box should have correct inside/outside signs."""
        mesh = create_box_mesh((0.5, 0.5, 0.5))
        self.assertTrue(mesh.is_watertight)

        sdf = SDF.create_from_mesh(mesh, max_resolution=32, texture_format="float32")
        self.assertIsNotNone(sdf.texture_data, "SDF should have texture data")

        test_points = np.array(
            [
                [0.0, 0.0, 0.0],  # center (inside)
                [1.0, 0.0, 0.0],  # well outside
            ],
            dtype=np.float32,
        )
        vals = _sample_texture_sdf_at_points(sdf, test_points)

        self.assertLess(float(vals[0]), 0.0, "Center of box should have negative SDF")
        self.assertGreater(float(vals[1]), 0.0, "Point outside box should have positive SDF")

        mesh.clear_sdf()

    def test_torus_watertight_sign_grid(self):
        """Non-convex torus: sign mismatches should be < 5% of sampled points.

        The scanline sign fill can differ from winding numbers at a few
        boundary voxels on complex non-convex geometry.  We allow up to 5%
        mismatches (all near the surface) while catching gross errors.
        """
        mesh = _create_torus_mesh(major_r=0.4, minor_r=0.15, n_major=24, n_minor=12)
        self.assertTrue(mesh.is_watertight, "Torus mesh should be watertight")

        mesh._is_watertight = False
        sdf_winding = SDF.create_from_mesh(mesh, max_resolution=32, texture_format="float32")
        mesh._is_watertight = None

        mesh._is_watertight = True
        sdf_fast = SDF.create_from_mesh(mesh, max_resolution=32, texture_format="float32")
        mesh._is_watertight = None

        pts = []
        for z in np.linspace(-0.7, 0.7, 10):
            for y in np.linspace(-0.7, 0.7, 10):
                for x in np.linspace(-0.7, 0.7, 10):
                    pts.append([x, y, z])
        pts_np = np.array(pts, dtype=np.float32)

        vals_w = _sample_texture_sdf_at_points(sdf_winding, pts_np)
        vals_f = _sample_texture_sdf_at_points(sdf_fast, pts_np)

        sign_w = np.sign(vals_w)
        sign_f = np.sign(vals_f)
        mismatches = int(np.sum(sign_w != sign_f))
        max_allowed = int(0.05 * len(pts))
        self.assertLessEqual(
            mismatches,
            max_allowed,
            f"{mismatches}/{len(pts)} voxels ({mismatches / len(pts) * 100:.1f}%) have sign mismatch on torus "
            f"(tolerance: {max_allowed}, 5%)",
        )
        mesh.clear_sdf()

    def test_watertight_distance_accuracy_sphere(self):
        """Distance error between fast and winding paths should be within 0.5 voxel-sizes.

        The unsigned-BVH fast path uses exact mesh distance queries (same BVH
        as the winding path) so distances should match closely.  Small
        differences arise from the different sign methods affecting which
        subgrids are allocated and how boundary voxels are classified.
        """
        mesh = create_sphere_mesh(0.5, subdivisions=2)
        res = 32

        mesh._is_watertight = False
        sdf_w = SDF.create_from_mesh(mesh, max_resolution=res, texture_format="float32")
        mesh._is_watertight = None

        mesh._is_watertight = True
        sdf_f = SDF.create_from_mesh(mesh, max_resolution=res, texture_format="float32")
        mesh._is_watertight = None

        pts = []
        for z in np.linspace(-0.6, 0.6, 12):
            for y in np.linspace(-0.6, 0.6, 12):
                for x in np.linspace(-0.6, 0.6, 12):
                    pts.append([x, y, z])
        pts_np = np.array(pts, dtype=np.float32)

        vals_w = _sample_texture_sdf_at_points(sdf_w, pts_np)
        vals_f = _sample_texture_sdf_at_points(sdf_f, pts_np)

        points_np = mesh.vertices
        extent = np.max(points_np, axis=0) - np.min(points_np, axis=0)
        voxel_size = float(np.max(extent)) / res

        max_err = float(np.max(np.abs(vals_w - vals_f)))
        self.assertLess(
            max_err,
            0.5 * voxel_size,
            f"Max distance error {max_err:.6f} exceeds 0.5*voxel_size={0.5 * voxel_size:.6f}",
        )
        mesh.clear_sdf()

    def test_watertight_distance_accuracy_box(self):
        """Distance error between fast and winding paths should be within 0.5 voxel-sizes."""
        mesh = create_box_mesh((0.4, 0.3, 0.5))
        res = 32

        mesh._is_watertight = False
        sdf_w = SDF.create_from_mesh(mesh, max_resolution=res, texture_format="float32")
        mesh._is_watertight = None

        mesh._is_watertight = True
        sdf_f = SDF.create_from_mesh(mesh, max_resolution=res, texture_format="float32")
        mesh._is_watertight = None

        pts = []
        for z in np.linspace(-0.6, 0.6, 12):
            for y in np.linspace(-0.4, 0.4, 12):
                for x in np.linspace(-0.5, 0.5, 12):
                    pts.append([x, y, z])
        pts_np = np.array(pts, dtype=np.float32)

        vals_w = _sample_texture_sdf_at_points(sdf_w, pts_np)
        vals_f = _sample_texture_sdf_at_points(sdf_f, pts_np)

        points_np = mesh.vertices
        extent = np.max(points_np, axis=0) - np.min(points_np, axis=0)
        voxel_size = float(np.max(extent)) / res

        max_err = float(np.max(np.abs(vals_w - vals_f)))
        self.assertLess(
            max_err,
            0.5 * voxel_size,
            f"Max distance error {max_err:.6f} exceeds 0.5*voxel_size={0.5 * voxel_size:.6f}",
        )
        mesh.clear_sdf()

    def test_sign_method_parity_override_on_non_watertight_mesh(self):
        """``sign_method='parity'`` should still build a texture-backed SDF even when
        auto-detection reports the mesh as non-watertight.

        The API documents that parity-sign results on non-watertight meshes are
        undefined, so this test only verifies that the override runs the parity
        path to completion and produces a populated SDF.  It intentionally does
        not assert specific inside/outside signs, which would lock undefined
        behavior into the test contract.
        """
        # Closed cube with its top face (+Y) removed: not watertight by topology.
        verts = np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ],
            dtype=np.float32,
        )
        indices = np.array(
            [
                0,
                2,
                1,
                0,
                3,
                2,  # -Z face
                4,
                5,
                6,
                4,
                6,
                7,  # +Z face
                0,
                1,
                5,
                0,
                5,
                4,  # -Y face
                0,
                4,
                7,
                0,
                7,
                3,  # -X face
                1,
                2,
                6,
                1,
                6,
                5,  # +X face
                # +Y face omitted: mesh is open at the top.
            ],
            dtype=np.int32,
        )
        mesh = Mesh(verts, indices, compute_inertia=False)
        self.assertFalse(mesh.is_watertight, "Open cube should auto-detect as non-watertight")

        sdf_parity = SDF.create_from_mesh(mesh, max_resolution=32, texture_format="float32", sign_method="parity")
        self.assertIsNotNone(sdf_parity.texture_data, "SDF should have texture data")
        self.assertFalse(sdf_parity.is_empty(), "Texture-backed SDF should not report as empty")

        mesh.clear_sdf()

    def test_watertight_uint16_texture_format(self):
        """Watertight path with ``texture_format='uint16'`` should produce a valid
        quantized SDF matching the float32 result within the quantization bound.
        """
        mesh = create_box_mesh((0.5, 0.5, 0.5))
        self.assertTrue(mesh.is_watertight)

        sdf_f32 = SDF.create_from_mesh(mesh, max_resolution=32, texture_format="float32")
        sdf_u16 = SDF.create_from_mesh(mesh, max_resolution=32, texture_format="uint16")

        self.assertIsNotNone(sdf_f32.texture_data)
        self.assertIsNotNone(sdf_u16.texture_data)

        pts = []
        for z in np.linspace(-0.6, 0.6, 8):
            for y in np.linspace(-0.6, 0.6, 8):
                for x in np.linspace(-0.6, 0.6, 8):
                    pts.append([x, y, z])
        pts_np = np.array(pts, dtype=np.float32)

        vals_f32 = _sample_texture_sdf_at_points(sdf_f32, pts_np)
        vals_u16 = _sample_texture_sdf_at_points(sdf_u16, pts_np)

        # Sign should always agree outside a tight band around the surface.
        # Inside the band, uint16 quantization error is bounded by narrow_band
        # range / 2**16, so rely on signed value tolerance instead.
        max_err = float(np.max(np.abs(vals_f32 - vals_u16)))
        # Default narrow band is ±0.1 m; uint16 SNORM quantization has
        # LSB ≈ 0.2 / 65535 ≈ 3e-6 m.  Allow 1e-3 m for rounding-plus-trilinear
        # interpolation noise across texels.
        self.assertLess(
            max_err,
            1e-3,
            f"uint16 watertight SDF deviates by {max_err:.6f} from float32 reference",
        )

        mesh.clear_sdf()


def _create_torus_mesh(major_r: float = 0.4, minor_r: float = 0.15, n_major: int = 24, n_minor: int = 12) -> Mesh:
    """Create a watertight torus mesh centered at the origin (non-convex)."""
    verts = []
    for i in range(n_major):
        theta = 2.0 * np.pi * i / n_major
        ct, st = np.cos(theta), np.sin(theta)
        for j in range(n_minor):
            phi = 2.0 * np.pi * j / n_minor
            cp, sp = np.cos(phi), np.sin(phi)
            r = major_r + minor_r * cp
            verts.append([r * ct, minor_r * sp, r * st])
    verts = np.array(verts, dtype=np.float32)

    faces = []
    for i in range(n_major):
        i_next = (i + 1) % n_major
        for j in range(n_minor):
            j_next = (j + 1) % n_minor
            a = i * n_minor + j
            b = i_next * n_minor + j
            c = i_next * n_minor + j_next
            d = i * n_minor + j_next
            faces.append([a, b, c])
            faces.append([a, c, d])
    indices = np.array(faces, dtype=np.int32).flatten()

    return Mesh(verts, indices)


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
