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

from newton._src.geometry.terrain_generator import (
    _box_terrain,
    _flat_terrain,
    _gap_terrain,
    _pyramid_stairs_terrain,
    _random_grid_terrain,
    _wave_terrain,
    generate_terrain_grid,
    heightfield_to_mesh,
)
from newton.tests.unittest_utils import assert_np_equal


class TestTerrainGenerator(unittest.TestCase):
    """Test suite for terrain generation functions."""

    # =========================================================================
    # Tests for heightfield_to_mesh function
    # =========================================================================

    def test_heightfield_to_mesh_basic(self):
        """Test basic heightfield to mesh conversion with valid inputs."""
        # Create a simple 3x3 heightfield
        heightfield = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)

        vertices, indices = heightfield_to_mesh(heightfield=heightfield, extent_x=2.0, extent_y=2.0)

        # Check output types
        self.assertEqual(vertices.dtype, np.float32)
        self.assertEqual(indices.dtype, np.int32)

        # Check vertex count: 3x3 grid = 9 top vertices + 9 bottom vertices = 18 total
        self.assertEqual(len(vertices), 18)

        # Check that vertices have 3 coordinates
        self.assertEqual(vertices.shape[1], 3)

        # Check that indices are flattened (1D array)
        self.assertEqual(len(indices.shape), 1)

        # Check that all indices are valid (within vertex range)
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < len(vertices)))

    def test_heightfield_to_mesh_minimum_grid_size(self):
        """Test heightfield to mesh with minimum valid grid size (2x2)."""
        # Minimum valid grid: 2x2
        heightfield = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        vertices, indices = heightfield_to_mesh(heightfield=heightfield, extent_x=1.0, extent_y=1.0)

        # Check that mesh was created successfully
        self.assertEqual(len(vertices), 8)  # 2x2 grid = 4 top + 4 bottom vertices
        self.assertGreater(len(indices), 0)

    def test_heightfield_to_mesh_invalid_dimensions(self):
        """Test that non-2D arrays raise ValueError."""
        # 1D array
        heightfield_1d = np.array([0.0, 1.0, 2.0])
        with self.assertRaises(ValueError) as context:
            heightfield_to_mesh(heightfield=heightfield_1d, extent_x=1.0, extent_y=1.0)
        self.assertIn("must be 2D array", str(context.exception))

        # 3D array
        heightfield_3d = np.zeros((3, 3, 3))
        with self.assertRaises(ValueError) as context:
            heightfield_to_mesh(heightfield=heightfield_3d, extent_x=1.0, extent_y=1.0)
        self.assertIn("must be 2D array", str(context.exception))

    def test_heightfield_to_mesh_too_small_grid(self):
        """Test that grid sizes smaller than 2x2 raise ValueError."""
        # 1x1 grid (too small)
        heightfield_1x1 = np.array([[1.0]])
        with self.assertRaises(ValueError) as context:
            heightfield_to_mesh(heightfield=heightfield_1x1, extent_x=1.0, extent_y=1.0)
        self.assertIn("at least 2x2", str(context.exception))

        # 1x3 grid (one dimension too small)
        heightfield_1x3 = np.array([[0.0, 1.0, 2.0]])
        with self.assertRaises(ValueError) as context:
            heightfield_to_mesh(heightfield=heightfield_1x3, extent_x=1.0, extent_y=1.0)
        self.assertIn("at least 2x2", str(context.exception))

        # 3x1 grid (one dimension too small)
        heightfield_3x1 = np.array([[0.0], [1.0], [2.0]])
        with self.assertRaises(ValueError) as context:
            heightfield_to_mesh(heightfield=heightfield_3x1, extent_x=1.0, extent_y=1.0)
        self.assertIn("at least 2x2", str(context.exception))

    def test_heightfield_to_mesh_non_positive_extent(self):
        """Test that non-positive extent values raise ValueError."""
        heightfield = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        # Zero extent_x
        with self.assertRaises(ValueError) as context:
            heightfield_to_mesh(heightfield=heightfield, extent_x=0.0, extent_y=1.0)
        self.assertIn("must be positive", str(context.exception))

        # Negative extent_x
        with self.assertRaises(ValueError) as context:
            heightfield_to_mesh(heightfield=heightfield, extent_x=-1.0, extent_y=1.0)
        self.assertIn("must be positive", str(context.exception))

        # Zero extent_y
        with self.assertRaises(ValueError) as context:
            heightfield_to_mesh(heightfield=heightfield, extent_x=1.0, extent_y=0.0)
        self.assertIn("must be positive", str(context.exception))

        # Negative extent_y
        with self.assertRaises(ValueError) as context:
            heightfield_to_mesh(heightfield=heightfield, extent_x=1.0, extent_y=-1.0)
        self.assertIn("must be positive", str(context.exception))

    def test_heightfield_to_mesh_flat_terrain(self):
        """Test heightfield to mesh with flat terrain (all zeros)."""
        heightfield = np.zeros((5, 5), dtype=np.float32)

        vertices, indices = heightfield_to_mesh(
            heightfield=heightfield, extent_x=4.0, extent_y=4.0, ground_z=-1.0
        )

        # Check that top surface is at z=0
        top_vertices = vertices[: len(vertices) // 2]
        assert_np_equal(top_vertices[:, 2], np.zeros(len(top_vertices)), tol=1e-6)

        # Check that bottom surface is at ground_z=-1.0
        bottom_vertices = vertices[len(vertices) // 2 :]
        assert_np_equal(bottom_vertices[:, 2], np.full(len(bottom_vertices), -1.0), tol=1e-6)

    def test_heightfield_to_mesh_sloped_terrain(self):
        """Test heightfield to mesh with sloped terrain."""
        # Create a simple slope from 0 to 1
        heightfield = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]], dtype=np.float32)

        vertices, indices = heightfield_to_mesh(heightfield=heightfield, extent_x=2.0, extent_y=2.0)

        # Check that heights are preserved in top vertices
        top_vertices = vertices[: len(vertices) // 2]
        expected_heights = heightfield.ravel()
        assert_np_equal(top_vertices[:, 2], expected_heights, tol=1e-6)

    def test_heightfield_to_mesh_random_terrain(self):
        """Test heightfield to mesh with random terrain."""
        np.random.seed(42)
        heightfield = np.random.uniform(-1.0, 1.0, size=(10, 10)).astype(np.float32)

        vertices, indices = heightfield_to_mesh(heightfield=heightfield, extent_x=5.0, extent_y=5.0)

        # Check vertex count
        expected_vertices = 10 * 10 * 2  # top + bottom
        self.assertEqual(len(vertices), expected_vertices)

        # Check that heights are preserved
        top_vertices = vertices[: len(vertices) // 2]
        assert_np_equal(top_vertices[:, 2], heightfield.ravel(), tol=1e-6)

    def test_heightfield_to_mesh_center_offset(self):
        """Test heightfield to mesh with custom center coordinates."""
        heightfield = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        vertices, indices = heightfield_to_mesh(
            heightfield=heightfield, extent_x=2.0, extent_y=2.0, center_x=5.0, center_y=10.0
        )

        # Check that vertices are centered around (5.0, 10.0)
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]

        # X coordinates should be centered around 5.0
        self.assertAlmostEqual(np.mean(x_coords), 5.0, delta=0.1)

        # Y coordinates should be centered around 10.0
        self.assertAlmostEqual(np.mean(y_coords), 10.0, delta=0.1)

    def test_heightfield_to_mesh_face_count(self):
        """Test that the correct number of faces are generated."""
        # For an NxM grid, we expect:
        # - Top surface: 2 * (N-1) * (M-1) triangles
        # - Bottom surface: 2 * (N-1) * (M-1) triangles
        # - 4 side walls: 2 * (N-1) + 2 * (M-1) + 2 * (N-1) + 2 * (M-1) = 4 * (N-1 + M-1) triangles
        N, M = 5, 7
        heightfield = np.random.rand(N, M).astype(np.float32)

        vertices, indices = heightfield_to_mesh(heightfield=heightfield, extent_x=4.0, extent_y=6.0)

        # Each triangle has 3 indices
        num_triangles = len(indices) // 3

        # Expected: top + bottom + 4 walls
        expected_top_bottom = 2 * 2 * (N - 1) * (M - 1)
        expected_walls = 4 * ((N - 1) + (M - 1))
        expected_total = expected_top_bottom + expected_walls

        self.assertEqual(num_triangles, expected_total)

    def test_heightfield_to_mesh_watertight(self):
        """Test that the generated mesh is watertight (closed)."""
        heightfield = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)

        vertices, indices = heightfield_to_mesh(
            heightfield=heightfield, extent_x=2.0, extent_y=2.0, ground_z=-0.5
        )

        # Check that we have both top and bottom vertices
        num_grid_points = heightfield.size
        self.assertEqual(len(vertices), 2 * num_grid_points)

        # Check that bottom vertices are at ground_z
        bottom_vertices = vertices[num_grid_points:]
        assert_np_equal(bottom_vertices[:, 2], np.full(num_grid_points, -0.5), tol=1e-6)

    # =========================================================================
    # Tests for existing terrain generation functions
    # =========================================================================

    def test_flat_terrain(self):
        """Test flat terrain generation."""
        size = (10.0, 10.0)
        height = 0.5

        vertices, indices = _flat_terrain(size, height)

        # Check output types
        self.assertEqual(vertices.dtype, np.float32)
        self.assertEqual(indices.dtype, np.int32)

        # Flat terrain should have 4 vertices (rectangle)
        self.assertEqual(len(vertices), 4)

        # All vertices should be at the same height
        assert_np_equal(vertices[:, 2], np.full(4, height), tol=1e-6)

    def test_pyramid_stairs_terrain(self):
        """Test pyramid stairs terrain generation."""
        size = (10.0, 10.0)

        vertices, indices = _pyramid_stairs_terrain(size, step_width=0.5, step_height=0.1, platform_width=1.0)

        # Check output types
        self.assertEqual(vertices.dtype, np.float32)
        self.assertEqual(indices.dtype, np.int32)

        # Should have vertices and indices
        self.assertGreater(len(vertices), 0)
        self.assertGreater(len(indices), 0)

        # All indices should be valid
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < len(vertices)))

    def test_random_grid_terrain(self):
        """Test random grid terrain generation."""
        size = (10.0, 10.0)
        seed = 42

        vertices1, indices1 = _random_grid_terrain(size, grid_width=0.5, seed=seed)
        vertices2, indices2 = _random_grid_terrain(size, grid_width=0.5, seed=seed)

        # Same seed should produce same terrain
        assert_np_equal(vertices1, vertices2, tol=1e-6)
        assert_np_equal(indices1, indices2, tol=0.0)

        # Different seed should produce different terrain
        vertices3, indices3 = _random_grid_terrain(size, grid_width=0.5, seed=123)
        self.assertFalse(np.allclose(vertices1, vertices3))

    def test_wave_terrain(self):
        """Test wave terrain generation."""
        size = (10.0, 10.0)

        vertices, indices = _wave_terrain(size, wave_amplitude=0.3, wave_frequency=2.0, resolution=50)

        # Check output types
        self.assertEqual(vertices.dtype, np.float32)
        self.assertEqual(indices.dtype, np.int32)

        # Check that vertices exist
        self.assertGreater(len(vertices), 0)

        # Wave should have variation in Z
        z_coords = vertices[:, 2]
        self.assertGreater(np.max(z_coords) - np.min(z_coords), 0.1)

    def test_box_terrain(self):
        """Test box terrain generation."""
        size = (10.0, 10.0)

        vertices, indices = _box_terrain(size, box_height=0.5, platform_width=1.5)

        # Check output types
        self.assertEqual(vertices.dtype, np.float32)
        self.assertEqual(indices.dtype, np.int32)

        # Should have vertices and indices
        self.assertGreater(len(vertices), 0)
        self.assertGreater(len(indices), 0)

    def test_gap_terrain(self):
        """Test gap terrain generation."""
        size = (10.0, 10.0)

        vertices, indices = _gap_terrain(size, gap_width=0.8, platform_width=1.2)

        # Check output types
        self.assertEqual(vertices.dtype, np.float32)
        self.assertEqual(indices.dtype, np.int32)

        # Should have vertices and indices
        self.assertGreater(len(vertices), 0)
        self.assertGreater(len(indices), 0)

    # =========================================================================
    # Tests for generate_terrain_grid
    # =========================================================================

    def test_generate_terrain_grid_single_block(self):
        """Test terrain grid generation with a single block."""
        vertices, indices = generate_terrain_grid(grid_size=(1, 1), block_size=(5.0, 5.0), terrain_types="flat")

        # Check output types
        self.assertEqual(vertices.dtype, np.float32)
        self.assertEqual(indices.dtype, np.int32)

        # Should have vertices and indices
        self.assertGreater(len(vertices), 0)
        self.assertGreater(len(indices), 0)

    def test_generate_terrain_grid_multiple_blocks(self):
        """Test terrain grid generation with multiple blocks."""
        vertices, indices = generate_terrain_grid(
            grid_size=(2, 2), block_size=(5.0, 5.0), terrain_types=["flat", "wave"]
        )

        # Should have more vertices than a single block
        self.assertGreater(len(vertices), 4)
        self.assertGreater(len(indices), 6)

    def test_generate_terrain_grid_with_seed(self):
        """Test that terrain grid generation is deterministic with seed."""
        vertices1, indices1 = generate_terrain_grid(
            grid_size=(2, 2), block_size=(5.0, 5.0), terrain_types="random_grid", seed=42
        )

        vertices2, indices2 = generate_terrain_grid(
            grid_size=(2, 2), block_size=(5.0, 5.0), terrain_types="random_grid", seed=42
        )

        # Same seed should produce same terrain
        assert_np_equal(vertices1, vertices2, tol=1e-6)
        assert_np_equal(indices1, indices2, tol=0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

