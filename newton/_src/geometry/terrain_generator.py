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


"""Compact procedural terrain generator for Newton physics examples.

Provides various terrain generation functions that output Newton-compatible triangle meshes.
Supports creating grids of terrain blocks with different procedural patterns.
"""

import numpy as np

# ============================================================================
# Helper Functions
# ============================================================================


def _create_box(
    size: tuple[float, float, float], position: tuple[float, float, float] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Create a box mesh as (vertices, faces) from dimensions and position.

    Each face has its own vertices to ensure sharp edges with per-face normals.

    Args:
        size: (width, depth, height) dimensions of the box
        position: (x, y, z) center position of the box. If None, centered at origin.

    Returns:
        Tuple of (vertices, faces) where vertices is (24, 3) float32 array
        and faces is (12, 3) int32 array (will be flattened in caller)
    """
    w, d, h = size
    half_w, half_d, half_h = w / 2, d / 2, h / 2

    # Create separate vertices for each face to get sharp edges
    # Each face gets 4 vertices (one per corner)
    # Order: bottom, top, front, back, left, right faces

    # Bottom face (z = -half_h, normal pointing down)
    bottom_vertices = np.array(
        [
            [-half_w, -half_d, -half_h],  # 0: front-left
            [half_w, -half_d, -half_h],  # 1: front-right
            [half_w, half_d, -half_h],  # 2: back-right
            [-half_w, half_d, -half_h],  # 3: back-left
        ],
        dtype=np.float32,
    )

    # Top face (z = half_h, normal pointing up)
    top_vertices = np.array(
        [
            [-half_w, -half_d, half_h],  # 4: front-left
            [half_w, -half_d, half_h],  # 5: front-right
            [half_w, half_d, half_h],  # 6: back-right
            [-half_w, half_d, half_h],  # 7: back-left
        ],
        dtype=np.float32,
    )

    # Front face (y = -half_d, normal pointing forward)
    front_vertices = np.array(
        [
            [-half_w, -half_d, -half_h],  # 8: bottom-left
            [half_w, -half_d, -half_h],  # 9: bottom-right
            [half_w, -half_d, half_h],  # 10: top-right
            [-half_w, -half_d, half_h],  # 11: top-left
        ],
        dtype=np.float32,
    )

    # Back face (y = half_d, normal pointing backward)
    back_vertices = np.array(
        [
            [half_w, half_d, -half_h],  # 12: bottom-right
            [-half_w, half_d, -half_h],  # 13: bottom-left
            [-half_w, half_d, half_h],  # 14: top-left
            [half_w, half_d, half_h],  # 15: top-right
        ],
        dtype=np.float32,
    )

    # Left face (x = -half_w, normal pointing left)
    left_vertices = np.array(
        [
            [-half_w, half_d, -half_h],  # 16: back-bottom
            [-half_w, -half_d, -half_h],  # 17: front-bottom
            [-half_w, -half_d, half_h],  # 18: front-top
            [-half_w, half_d, half_h],  # 19: back-top
        ],
        dtype=np.float32,
    )

    # Right face (x = half_w, normal pointing right)
    right_vertices = np.array(
        [
            [half_w, -half_d, -half_h],  # 20: front-bottom
            [half_w, half_d, -half_h],  # 21: back-bottom
            [half_w, half_d, half_h],  # 22: back-top
            [half_w, -half_d, half_h],  # 23: front-top
        ],
        dtype=np.float32,
    )

    # Combine all vertices
    vertices = np.vstack(
        [
            bottom_vertices,
            top_vertices,
            front_vertices,
            back_vertices,
            left_vertices,
            right_vertices,
        ]
    )

    # Translate to position if provided
    if position is not None:
        vertices += np.array(position, dtype=np.float32)

    # Define faces (12 triangles for a box)
    # Each face is two triangles, counter-clockwise when viewed from outside
    # Vertex indices: bottom (0-3), top (4-7), front (8-11), back (12-15), left (16-19), right (20-23)
    faces = np.array(
        [
            # Bottom face (z = -half_h)
            [0, 2, 1],
            [0, 3, 2],
            # Top face (z = half_h)
            [4, 5, 6],
            [4, 6, 7],
            # Front face (y = -half_d)
            [8, 9, 10],
            [8, 10, 11],
            # Back face (y = half_d)
            [12, 13, 14],
            [12, 14, 15],
            # Left face (x = -half_w)
            [16, 17, 18],
            [16, 18, 19],
            # Right face (x = half_w)
            [20, 21, 22],
            [20, 22, 23],
        ],
        dtype=np.int32,
    )

    return vertices, faces


# ============================================================================
# Primitive Terrain Functions
# ============================================================================


def _flat_terrain(size: tuple[float, float], height: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate a flat plane terrain.

    Args:
        size: (width, height) size of the terrain plane in meters
        height: Z-coordinate height of the terrain plane

    Returns:
        tuple of (vertices, indices) where vertices is (N, 3) float32 array
        and indices is (M,) int32 array of triangle indices
    """
    x0 = [size[0], size[1], height]
    x1 = [size[0], 0.0, height]
    x2 = [0.0, size[1], height]
    x3 = [0.0, 0.0, height]
    vertices = np.array([x0, x1, x2, x3], dtype=np.float32)
    faces = np.array([[1, 0, 2], [2, 3, 1]], dtype=np.int32)
    return vertices, faces.flatten()


def _pyramid_stairs_terrain(
    size: tuple[float, float], step_width: float = 0.5, step_height: float = 0.1, platform_width: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate pyramid stairs terrain with steps converging to center platform.

    Args:
        size: (width, height) size of the terrain in meters
        step_width: Width of each step ring
        step_height: Height increment for each step
        platform_width: Width of the center platform

    Returns:
        tuple of (vertices, indices) where vertices is (N, 3) float32 array
        and indices is (M,) int32 array of triangle indices
    """
    meshes = []
    center = [size[0] / 2, size[1] / 2, 0.0]

    num_steps_x = int((size[0] - platform_width) / (2 * step_width))
    num_steps_y = int((size[1] - platform_width) / (2 * step_width))
    num_steps = min(num_steps_x, num_steps_y)

    # Add ground plane
    ground_pos = (center[0], center[1], -step_height / 2)
    meshes.append(_create_box((size[0], size[1], step_height), ground_pos))

    # Create concentric rectangular steps (including final ring around platform)
    for k in range(num_steps + 1):
        box_size = (size[0] - 2 * k * step_width, size[1] - 2 * k * step_width)
        box_z = center[2] + (k + 1) * step_height / 2.0
        box_offset = (k + 0.5) * step_width
        box_height = (k + 1) * step_height

        # Skip if this would be smaller than the platform
        if box_size[0] <= platform_width or box_size[1] <= platform_width:
            continue

        # Top/bottom/left/right boxes
        for dx, dy, sx, sy in [
            (0, size[1] / 2 - box_offset, box_size[0], step_width),  # top
            (0, -size[1] / 2 + box_offset, box_size[0], step_width),  # bottom
            (size[0] / 2 - box_offset, 0, step_width, box_size[1] - 2 * step_width),  # right
            (-size[0] / 2 + box_offset, 0, step_width, box_size[1] - 2 * step_width),  # left
        ]:
            pos = (center[0] + dx, center[1] + dy, box_z)
            meshes.append(_create_box((sx, sy, box_height), pos))

    # Center platform (two steps higher than the last step ring)
    platform_height = (num_steps + 2) * step_height
    box_dims = (platform_width, platform_width, platform_height)
    box_pos = (center[0], center[1], center[2] + platform_height / 2)
    meshes.append(_create_box(box_dims, box_pos))

    return _combine_meshes(meshes)


def _random_grid_terrain(
    size: tuple[float, float],
    grid_width: float = 0.5,
    grid_height_range: tuple[float, float] = (-0.15, 0.15),
    platform_width: float | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate terrain with randomized height grid cells.

    Args:
        size: (width, height) size of the terrain in meters
        grid_width: Width of each grid cell
        grid_height_range: (min_height, max_height) range for random height variation
        platform_width: Unused parameter (kept for API compatibility)
        seed: Random seed for reproducibility

    Returns:
        tuple of (vertices, indices) where vertices is (N, 3) float32 array
        and indices is (M,) int32 array of triangle indices
    """
    rng = np.random.default_rng(seed)

    num_boxes_x = int(size[0] / grid_width)
    num_boxes_y = int(size[1] / grid_width)

    # Template box for a grid cell
    template_vertices, template_faces = _create_box((grid_width, grid_width, 1.0))

    # Create grid with random heights
    all_vertices = []
    all_faces = []
    vertex_count = 0

    for ix in range(num_boxes_x):
        for it in range(num_boxes_y):
            # Position grid cells starting from (0, 0) with proper alignment
            x = ix * grid_width + grid_width / 2
            y = it * grid_width + grid_width / 2
            h_noise = rng.uniform(*grid_height_range)

            # Offset vertices (box is centered at origin)
            v = template_vertices.copy()
            v[:, 0] += x
            v[:, 1] += y
            v[:, 2] -= 0.5

            # Raise top face vertices (indices 4-7) by random height
            v[4:8, 2] += h_noise

            all_vertices.append(v)
            all_faces.append(template_faces + vertex_count)
            vertex_count += 24  # Each box has 24 vertices (4 per face, 6 faces)

    vertices = np.vstack(all_vertices).astype(np.float32)
    faces = np.vstack(all_faces).astype(np.int32)

    return vertices, faces.flatten()


def _wave_terrain(
    size: tuple[float, float], wave_amplitude: float = 0.3, wave_frequency: float = 2.0, resolution: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Generate 2D sine wave terrain with zero boundaries.

    Args:
        size: (width, height) size of the terrain in meters
        wave_amplitude: Amplitude of the sine wave
        wave_frequency: Frequency of the sine wave
        resolution: Number of grid points per dimension

    Returns:
        tuple of (vertices, indices) where vertices is (N, 3) float32 array
        and indices is (M,) int32 array of triangle indices
    """
    x = np.linspace(0, size[0], resolution)
    y = np.linspace(0, size[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Create 2D sine pattern that is naturally zero at all boundaries
    # sin(n*pi*x/L) is zero at x=0 and x=L for integer n
    Z = wave_amplitude * np.sin(wave_frequency * np.pi * X / size[0]) * np.sin(wave_frequency * np.pi * Y / size[1])

    # Create vertices and faces
    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)

    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            v0 = i * resolution + j
            v1 = i * resolution + (j + 1)
            v2 = (i + 1) * resolution + j
            v3 = (i + 1) * resolution + (j + 1)
            # Counter-clockwise winding for upward-facing triangles
            faces.append([v0, v1, v2])
            faces.append([v2, v1, v3])

    return vertices, np.array(faces, dtype=np.int32).flatten()


def _box_terrain(
    size: tuple[float, float], box_height: float = 0.5, platform_width: float = 1.5
) -> tuple[np.ndarray, np.ndarray]:
    """Generate terrain with a raised box platform in center.

    Args:
        size: (width, height) size of the terrain in meters
        box_height: Height of the raised platform
        platform_width: Width of the raised platform

    Returns:
        tuple of (vertices, indices) where vertices is (N, 3) float32 array
        and indices is (M,) int32 array of triangle indices
    """
    meshes = []

    # Ground plane
    ground_pos = (size[0] / 2, size[1] / 2, -0.5)
    meshes.append(_create_box((size[0], size[1], 1.0), ground_pos))

    # Raised platform
    platform_pos = (size[0] / 2, size[1] / 2, box_height / 2 - 0.5)
    meshes.append(_create_box((platform_width, platform_width, 1.0 + box_height), platform_pos))

    return _combine_meshes(meshes)


def _gap_terrain(
    size: tuple[float, float], gap_width: float = 0.8, platform_width: float = 1.2
) -> tuple[np.ndarray, np.ndarray]:
    """Generate terrain with a gap around the center platform.

    Args:
        size: (width, height) size of the terrain in meters
        gap_width: Width of the gap around the platform
        platform_width: Width of the center platform

    Returns:
        tuple of (vertices, indices) where vertices is (N, 3) float32 array
        and indices is (M,) int32 array of triangle indices
    """
    meshes = []
    center = (size[0] / 2, size[1] / 2, -0.5)

    # Outer border
    thickness_x = (size[0] - platform_width - 2 * gap_width) / 2
    thickness_y = (size[1] - platform_width - 2 * gap_width) / 2

    for dx, dy, sx, sy in [
        (0, (size[1] - thickness_y) / 2, size[0], thickness_y),  # top
        (0, -(size[1] - thickness_y) / 2, size[0], thickness_y),  # bottom
        ((size[0] - thickness_x) / 2, 0, thickness_x, platform_width + 2 * gap_width),  # right
        (-(size[0] - thickness_x) / 2, 0, thickness_x, platform_width + 2 * gap_width),  # left
    ]:
        pos = (center[0] + dx, center[1] + dy, center[2])
        meshes.append(_create_box((sx, sy, 1.0), pos))

    # Center platform
    meshes.append(_create_box((platform_width, platform_width, 1.0), center))

    return _combine_meshes(meshes)


# ============================================================================
# Terrain Grid Generator
# ============================================================================


def generate_terrain_grid(
    grid_size: tuple[int, int] = (4, 4),
    block_size: tuple[float, float] = (5.0, 5.0),
    terrain_types: list[str] | str | object | None = None,
    terrain_params: dict[str, dict[str, float]] | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a grid of procedural terrain blocks.

    This is the main public API function for generating terrain grids.

    Args:
        grid_size: (rows, cols) number of terrain blocks
        block_size: (width, height) size of each terrain block in meters
        terrain_types: List of terrain type names, single terrain type string,
                      or callable function (any object with __call__). If None, uses all types.
                      Available types: 'flat', 'pyramid_stairs', 'random_grid', 'wave', 'box', 'gap'
        terrain_params: Dictionary mapping terrain types to their parameter dicts
        seed: Random seed for reproducibility

    Returns:
        tuple of (vertices, indices) where:
        - vertices: (N, 3) float32 array of vertex positions
        - indices: (M,) int32 array of triangle indices (flattened)
    """

    # Default terrain types
    if terrain_types is None:
        terrain_types = ["flat", "pyramid_stairs", "random_grid", "wave", "box", "gap"]

    terrain_funcs = {
        "flat": _flat_terrain,
        "pyramid_stairs": _pyramid_stairs_terrain,
        "random_grid": _random_grid_terrain,
        "wave": _wave_terrain,
        "box": _box_terrain,
        "gap": _gap_terrain,
    }

    if terrain_params is None:
        terrain_params = {}

    all_vertices = []
    all_indices = []
    vertex_offset = 0

    rows, cols = grid_size

    for row in range(rows):
        for col in range(cols):
            # Select terrain type (cycle or random)
            if isinstance(terrain_types, list):
                terrain_idx = (row * cols + col) % len(terrain_types)
                terrain_name = terrain_types[terrain_idx]
            else:
                terrain_name = terrain_types

            # Get terrain function
            if callable(terrain_name):
                terrain_func = terrain_name
            else:
                terrain_func = terrain_funcs[terrain_name]

            # Get parameters for this terrain type
            params = terrain_params.get(terrain_name, {})

            # Generate terrain block
            vertices, indices = terrain_func(block_size, **params)

            # Offset to grid position
            offset_x = col * block_size[0]
            offset_y = row * block_size[1]
            vertices[:, 0] += offset_x
            vertices[:, 1] += offset_y

            # Accumulate geometry
            all_vertices.append(vertices)
            all_indices.append(indices + vertex_offset)
            vertex_offset += len(vertices)

    # Combine all blocks
    vertices = np.vstack(all_vertices).astype(np.float32)
    indices = np.concatenate(all_indices).astype(np.int32)

    return vertices, indices


# ============================================================================
# Helper Functions
# ============================================================================


def _combine_meshes(meshes: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Combine multiple (vertices, faces) tuples into a single mesh.

    Args:
        meshes: List of (vertices, faces) tuples to combine

    Returns:
        tuple of (vertices, indices) where vertices is (N, 3) float32 array
        and indices is (M,) int32 array of triangle indices (flattened)
    """
    if len(meshes) == 1:
        vertices, faces = meshes[0]
        return vertices.astype(np.float32), faces.flatten().astype(np.int32)

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for vertices, faces in meshes:
        all_vertices.append(vertices)
        all_faces.append(faces + vertex_offset)
        vertex_offset += len(vertices)

    combined_vertices = np.vstack(all_vertices).astype(np.float32)
    combined_faces = np.vstack(all_faces).astype(np.int32)

    return combined_vertices, combined_faces.flatten()


def _to_newton_mesh(vertices: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert terrain geometry to Newton mesh format.

    This is a convenience function that ensures proper dtypes.

    Args:
        vertices: (N, 3) array of vertex positions
        indices: (M,) array of triangle indices (flattened)

    Returns:
        tuple of (vertices, indices) with proper dtypes for Newton (float32 and int32)
    """
    return vertices.astype(np.float32), indices.astype(np.int32)
