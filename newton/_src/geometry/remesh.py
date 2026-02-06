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

"""Point cloud extraction and surface reconstruction for mesh repair.

This module provides GPU-accelerated utilities to extract dense point clouds with
reliable surface normals from triangle meshes. The extraction uses multi-view
orthographic raycasting from camera directions distributed on an icosphere, with
online voxel-based downsampling for memory efficiency. Optional secondary "cavity
cameras" improve coverage of deep cavities and occluded regions.

The point cloud can then be used to reconstruct a clean, watertight mesh using
Poisson surface reconstruction.

Key features:
    - GPU-accelerated raycasting using Warp
    - Online downsampling via sparse voxel hash grid (Morton-encoded keys)
    - Random camera roll and randomized processing order to reduce sampling bias
    - Optional cavity cameras for improved coverage of occluded regions
    - Probability-scaled selection of cavity candidates (favors deeper hits)
    - Consistent outward-facing normals

Requirements:
    - Point cloud extraction (PointCloudExtractor): Only requires Warp (included with Newton)
    - Surface reconstruction (SurfaceReconstructor): Requires Open3D (`pip install open3d`)

This is useful for repairing meshes with:
    - Inconsistent or flipped triangle winding
    - Missing or incorrect vertex normals
    - Non-manifold geometry
    - Holes or self-intersections
    - Deep cavities that are hard to capture from external viewpoints

Example:
    Remesh a problematic mesh to get a clean, watertight version::

        import numpy as np
        from newton._src.geometry.remesh import PointCloudExtractor, SurfaceReconstructor

        # Load your mesh (vertices: Nx3, indices: Mx3 or flattened)
        vertices = np.array(...)  # your mesh vertices
        indices = np.array(...)  # your mesh triangle indices

        # Step 1: Extract point cloud with reliable normals
        # edge_segments controls view count: views = 20 * n^2
        # cavity_cameras adds secondary cameras for deep cavities
        extractor = PointCloudExtractor(
            edge_segments=4,  # 320 views
            resolution=1000,  # 1000x1000 rays per view
            cavity_cameras=100,  # 100 secondary hemisphere cameras
        )
        points, normals = extractor.extract(vertices, indices)
        print(f"Extracted {len(points)} points")

        # Step 2: Reconstruct clean mesh using Poisson reconstruction
        reconstructor = SurfaceReconstructor(
            depth=10,
            simplify_tolerance=1e-7,  # fraction of mesh diagonal
        )
        clean_mesh = reconstructor.reconstruct(points, normals)
        print(f"Reconstructed {len(clean_mesh.indices) // 3} triangles")

        # Use the clean mesh
        new_vertices = clean_mesh.vertices  # (N, 3) float32
        new_indices = clean_mesh.indices  # (M,) int32, flattened
"""

import math
import warnings

import numpy as np
import warp as wp

from newton._src.geometry.hashtable import HashTable, hashtable_find_or_insert
from newton._src.geometry.types import Mesh

# -----------------------------------------------------------------------------
# Morton encoding for sparse voxel grid (21 bits per axis = 63 bits total)
# -----------------------------------------------------------------------------

# Offset to handle negative coordinates: shift by 2^20 so range [-2^20, 2^20) maps to [0, 2^21)
VOXEL_COORD_OFFSET = wp.constant(wp.int32(1 << 20))  # 1,048,576
VOXEL_COORD_MASK = wp.constant(wp.uint64(0x1FFFFF))  # 21 bits = 2,097,151


@wp.func
def _split_by_3(x: wp.uint64) -> wp.uint64:
    """Spread 21-bit integer into 63 bits with 2 zeros between each bit (for Morton encoding)."""
    # x = ---- ---- ---- ---- ---- ---- ---x xxxx xxxx xxxx xxxx xxxx (21 bits)
    x = x & wp.uint64(0x1FFFFF)  # Mask to 21 bits
    # Spread bits apart using magic numbers (interleave with zeros)
    x = (x | (x << wp.uint64(32))) & wp.uint64(0x1F00000000FFFF)
    x = (x | (x << wp.uint64(16))) & wp.uint64(0x1F0000FF0000FF)
    x = (x | (x << wp.uint64(8))) & wp.uint64(0x100F00F00F00F00F)
    x = (x | (x << wp.uint64(4))) & wp.uint64(0x10C30C30C30C30C3)
    x = (x | (x << wp.uint64(2))) & wp.uint64(0x1249249249249249)
    return x


@wp.func
def morton_encode_3d(cx: wp.int32, cy: wp.int32, cz: wp.int32) -> wp.uint64:
    """Encode 3 signed integers into a 63-bit Morton code.

    Each coordinate is shifted by VOXEL_COORD_OFFSET to handle negatives,
    then the 21-bit values are interleaved: z takes bits 2,5,8,..., y takes 1,4,7,..., x takes 0,3,6,...
    """
    # Shift to unsigned range
    ux = wp.uint64(cx + VOXEL_COORD_OFFSET) & VOXEL_COORD_MASK
    uy = wp.uint64(cy + VOXEL_COORD_OFFSET) & VOXEL_COORD_MASK
    uz = wp.uint64(cz + VOXEL_COORD_OFFSET) & VOXEL_COORD_MASK
    # Interleave bits
    return _split_by_3(ux) | (_split_by_3(uy) << wp.uint64(1)) | (_split_by_3(uz) << wp.uint64(2))


@wp.func
def compute_voxel_key(
    point: wp.vec3,
    inv_voxel_size: wp.float32,
) -> wp.uint64:
    """Compute Morton-encoded voxel key for a point."""
    # Quantize to integer voxel coordinates
    cx = wp.int32(wp.floor(point[0] * inv_voxel_size))
    cy = wp.int32(wp.floor(point[1] * inv_voxel_size))
    cz = wp.int32(wp.floor(point[2] * inv_voxel_size))
    return morton_encode_3d(cx, cy, cz)


# -----------------------------------------------------------------------------
# Random number generation (LCG-based, suitable for GPU)
# -----------------------------------------------------------------------------

# LCG constants (same as glibc)
_LCG_A = wp.constant(wp.uint32(1103515245))
_LCG_C = wp.constant(wp.uint32(12345))


@wp.func
def rand_init(seed: wp.uint32, thread_id: wp.uint32) -> wp.uint32:
    """Initialize random state from a seed and thread ID.

    Combines seed with thread_id using XOR and applies one LCG step
    to ensure different threads have different starting states.
    """
    state = seed ^ thread_id
    # Apply one LCG step to mix the bits
    return state * _LCG_A + _LCG_C


@wp.func
def rand_next(state: wp.uint32) -> wp.uint32:
    """Advance the random state and return the new state."""
    return state * _LCG_A + _LCG_C


@wp.func
def rand_float(state: wp.uint32) -> float:
    """Convert random state to a float in [0, 1]."""
    # Use upper 31 bits (better quality in LCG)
    return wp.float32(state & wp.uint32(0x7FFFFFFF)) / wp.float32(0x7FFFFFFF)


@wp.func
def rand_next_float(state: wp.uint32) -> tuple[wp.uint32, float]:
    """Advance state and return (new_state, random_float).

    Use this when you need multiple random numbers in sequence.
    """
    new_state = state * _LCG_A + _LCG_C
    rand_val = wp.float32(new_state & wp.uint32(0x7FFFFFFF)) / wp.float32(0x7FFFFFFF)
    return new_state, rand_val


# -----------------------------------------------------------------------------
# VoxelHashGrid - sparse voxel grid with online accumulation
# -----------------------------------------------------------------------------


@wp.kernel
def _accumulate_point_kernel(
    point: wp.vec3,
    normal: wp.vec3,
    inv_voxel_size: wp.float32,
    # Hash table arrays
    keys: wp.array(dtype=wp.uint64),
    active_slots: wp.array(dtype=wp.int32),
    # Accumulator arrays
    sum_positions_x: wp.array(dtype=wp.float32),
    sum_positions_y: wp.array(dtype=wp.float32),
    sum_positions_z: wp.array(dtype=wp.float32),
    sum_normals_x: wp.array(dtype=wp.float32),
    sum_normals_y: wp.array(dtype=wp.float32),
    sum_normals_z: wp.array(dtype=wp.float32),
    counts: wp.array(dtype=wp.int32),
):
    """Accumulate a single point into the voxel grid (for testing)."""
    key = compute_voxel_key(point, inv_voxel_size)
    idx = hashtable_find_or_insert(key, keys, active_slots)
    if idx >= 0:
        old_count = wp.atomic_add(counts, idx, 1)
        # Only store position on first hit
        if old_count == 0:
            sum_positions_x[idx] = point[0]
            sum_positions_y[idx] = point[1]
            sum_positions_z[idx] = point[2]
        # Always accumulate normals
        wp.atomic_add(sum_normals_x, idx, normal[0])
        wp.atomic_add(sum_normals_y, idx, normal[1])
        wp.atomic_add(sum_normals_z, idx, normal[2])


@wp.kernel
def _finalize_voxels_kernel(
    active_slots: wp.array(dtype=wp.int32),
    num_active: wp.int32,
    # Accumulator arrays (input)
    sum_positions_x: wp.array(dtype=wp.float32),
    sum_positions_y: wp.array(dtype=wp.float32),
    sum_positions_z: wp.array(dtype=wp.float32),
    sum_normals_x: wp.array(dtype=wp.float32),
    sum_normals_y: wp.array(dtype=wp.float32),
    sum_normals_z: wp.array(dtype=wp.float32),
    counts: wp.array(dtype=wp.int32),
    # Output arrays
    out_points: wp.array(dtype=wp.vec3),
    out_normals: wp.array(dtype=wp.vec3),
):
    """Finalize voxel averages and write to output arrays."""
    tid = wp.tid()
    if tid >= num_active:
        return

    idx = active_slots[tid]
    count = counts[idx]
    if count <= 0:
        return

    # Position: use the stored position directly (first hit, no averaging)
    # This avoids position drift artifacts (bumps) from averaging hits
    # at slightly different depths from different ray angles
    pos = wp.vec3(
        sum_positions_x[idx],
        sum_positions_y[idx],
        sum_positions_z[idx],
    )

    # Normal: average and normalize (averaging normals is good for smoothness)
    avg_normal = wp.vec3(
        sum_normals_x[idx],
        sum_normals_y[idx],
        sum_normals_z[idx],
    )
    normal_len = wp.length(avg_normal)
    if normal_len > 1e-8:
        avg_normal = avg_normal / normal_len
    else:
        avg_normal = wp.vec3(0.0, 1.0, 0.0)  # Fallback

    out_points[tid] = pos
    out_normals[tid] = avg_normal


class VoxelHashGrid:
    """Sparse voxel grid with online accumulation of positions and normals.

    Uses a GPU hash table to map voxel coordinates (Morton-encoded) to
    accumulator slots. Points and normals are accumulated using atomic
    operations, allowing fully parallel insertion from multiple threads.

    This is useful for voxel-based downsampling of point clouds directly
    on the GPU without intermediate storage.

    Args:
        capacity: Maximum number of unique voxels. Rounded up to power of two.
        voxel_size: Size of each cubic voxel.
        device: Warp device for computation.

    Example:
        >>> grid = VoxelHashGrid(capacity=1_000_000, voxel_size=0.01)
        >>> # Accumulate points (typically done in a kernel)
        >>> # ...
        >>> points, normals, count = grid.finalize()
    """

    def __init__(
        self,
        capacity: int,
        voxel_size: float,
        device: str | None = None,
    ):
        if voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {voxel_size}")

        self.voxel_size = voxel_size
        self.inv_voxel_size = 1.0 / voxel_size
        self.device = device

        # Hash table for voxel keys
        self._hashtable = HashTable(capacity, device=device)
        self.capacity = self._hashtable.capacity

        # Accumulator arrays (separate x/y/z for atomic_add compatibility)
        self.sum_positions_x = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        self.sum_positions_y = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        self.sum_positions_z = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        self.sum_normals_x = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        self.sum_normals_y = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        self.sum_normals_z = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        self.counts = wp.zeros(self.capacity, dtype=wp.int32, device=device)
        # Max confidence per voxel (for two-pass best-hit selection)
        self.max_confidences = wp.zeros(self.capacity, dtype=wp.float32, device=device)

    @property
    def keys(self) -> wp.array:
        """Hash table keys array (for use in kernels)."""
        return self._hashtable.keys

    @property
    def active_slots(self) -> wp.array:
        """Active slots tracking array (for use in kernels)."""
        return self._hashtable.active_slots

    def clear(self):
        """Clear all voxels and reset accumulators."""
        self._hashtable.clear()
        self.sum_positions_x.zero_()
        self.sum_positions_y.zero_()
        self.sum_positions_z.zero_()
        self.sum_normals_x.zero_()
        self.sum_normals_y.zero_()
        self.sum_normals_z.zero_()
        self.counts.zero_()
        self.max_confidences.zero_()

    def get_num_voxels(self) -> int:
        """Get the current number of occupied voxels."""
        return int(self._hashtable.active_slots.numpy()[self.capacity])

    def finalize(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Finalize accumulation and return averaged points and normals.

        Computes the average position and normalized normal for each occupied
        voxel and returns the results as numpy arrays.

        Returns:
            Tuple of (points, normals, num_points) where:
            - points: (N, 3) float32 array of averaged positions
            - normals: (N, 3) float32 array of normalized normals
            - num_points: number of occupied voxels
        """
        num_active = self.get_num_voxels()
        if num_active == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                0,
            )

        # Allocate output buffers
        out_points = wp.zeros(num_active, dtype=wp.vec3, device=self.device)
        out_normals = wp.zeros(num_active, dtype=wp.vec3, device=self.device)

        # Launch finalization kernel
        wp.launch(
            _finalize_voxels_kernel,
            dim=num_active,
            inputs=[
                self.active_slots,
                num_active,
                self.sum_positions_x,
                self.sum_positions_y,
                self.sum_positions_z,
                self.sum_normals_x,
                self.sum_normals_y,
                self.sum_normals_z,
                self.counts,
                out_points,
                out_normals,
            ],
            device=self.device,
        )

        wp.synchronize()

        return (
            out_points.numpy(),
            out_normals.numpy(),
            num_active,
        )


def compute_bounding_sphere(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute a bounding sphere for a set of vertices.

    Uses Ritter's algorithm for a reasonable approximation.

    Args:
        vertices: (N, 3) array of vertex positions.

    Returns:
        Tuple of (center, radius) where center is (3,) array.

    Raises:
        ValueError: If vertices array is empty.
    """
    if len(vertices) == 0:
        raise ValueError("Cannot compute bounding sphere for empty vertex array")

    # Start with axis-aligned bounding box center
    min_pt = np.min(vertices, axis=0)
    max_pt = np.max(vertices, axis=0)
    center = (min_pt + max_pt) / 2.0

    # Compute radius as max distance from center
    distances = np.linalg.norm(vertices - center, axis=1)
    radius = float(np.max(distances))

    # Handle single-vertex case: use small positive radius
    if radius == 0.0:
        radius = 1e-6

    return center, radius


def create_icosahedron_directions(edge_segments: int = 2) -> np.ndarray:
    """Create camera directions from subdivided icosahedron face centers.

    An icosahedron has 20 faces. Each face is subdivided into n^2 smaller
    triangles where n is the number of segments per edge. This gives
    fine-grained control over the number of directions.

    Args:
        edge_segments: Number of segments per triangle edge (n >= 1).
            Total faces = 20 * n^2. Examples:
            - n=1: 20 faces (original icosahedron)
            - n=2: 80 faces
            - n=3: 180 faces
            - n=4: 320 faces
            - n=5: 500 faces

    Returns:
        (N, 3) array of unit direction vectors, one per face.
    """
    if edge_segments < 1:
        raise ValueError(f"edge_segments must be >= 1, got {edge_segments}")

    n = edge_segments

    # Golden ratio
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    # Icosahedron vertices (normalized)
    ico_verts = np.array(
        [
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
        ],
        dtype=np.float64,
    )
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)

    # Icosahedron faces (20 triangles)
    ico_faces = np.array(
        [
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
        ],
        dtype=np.int32,
    )

    # Get vertices for all 20 faces: (20, 3, 3) - 20 faces, 3 vertices each, 3 coords
    v0_all = ico_verts[ico_faces[:, 0]]  # (20, 3)
    v1_all = ico_verts[ico_faces[:, 1]]  # (20, 3)
    v2_all = ico_verts[ico_faces[:, 2]]  # (20, 3)

    if n == 1:
        # No subdivision needed - just return face centers
        centers = (v0_all + v1_all + v2_all) / 3.0
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        return centers.astype(np.float32)

    # Subdivide each face into n^2 triangles using barycentric coordinates
    # Total sub-faces = 20 * n^2

    # Pre-compute barycentric coordinates for all sub-triangle centers
    # For upward triangles: centers at (i + 1/3, j + 1/3) in barycentric grid coords
    # For downward triangles: centers at (i + 2/3, j + 2/3) in barycentric grid coords

    # Generate all upward triangle barycentric centers
    # Upward triangles exist for i in [0, n-1], j in [0, n-i-1]
    up_coords = []
    for i in range(n):
        for j in range(n - i):
            # Center of triangle with vertices at (i,j), (i+1,j), (i,j+1)
            # Barycentric center: ((i + i+1 + i)/3, (j + j + j+1)/3) = (i + 1/3, j + 1/3)
            bi = (i + (i + 1) + i) / 3.0
            bj = (j + j + (j + 1)) / 3.0
            up_coords.append((bi, bj))

    # Generate all downward triangle barycentric centers
    down_coords = []
    for i in range(n):
        for j in range(n - i - 1):
            # Center of triangle with vertices at (i+1,j), (i+1,j+1), (i,j+1)
            bi = ((i + 1) + (i + 1) + i) / 3.0
            bj = (j + (j + 1) + (j + 1)) / 3.0
            down_coords.append((bi, bj))

    # Combine all sub-triangle centers
    all_bary = np.array(up_coords + down_coords, dtype=np.float64)  # (n^2, 2)

    # Convert barycentric (i, j) to weights (w0, w1, w2) where w0 + w1 + w2 = 1
    # p = w0*v0 + w1*v1 + w2*v2 where w0 = (n - i - j)/n, w1 = j/n, w2 = i/n
    # Wait, need to be careful: barycentric coords (i, j, k) where k = n - i - j
    # p = (i*v0 + j*v1 + k*v2) / n
    # So w0 = i/n (weight for v0), w1 = j/n (weight for v1), w2 = k/n = (n-i-j)/n (weight for v2)

    bi = all_bary[:, 0]  # (num_subtris,)
    bj = all_bary[:, 1]  # (num_subtris,)
    bk = n - bi - bj

    w0 = bi / n  # (num_subtris,)
    w1 = bj / n
    w2 = bk / n

    # Compute centers for all 20 faces x num_subtris sub-triangles
    # Result shape: (20, num_subtris, 3)
    # centers[f, s] = w0[s]*v0_all[f] + w1[s]*v1_all[f] + w2[s]*v2_all[f]

    # Use broadcasting: (20, 1, 3) * (1, num_subtris, 1) -> (20, num_subtris, 3)
    centers = (
        v0_all[:, np.newaxis, :] * w0[np.newaxis, :, np.newaxis]
        + v1_all[:, np.newaxis, :] * w1[np.newaxis, :, np.newaxis]
        + v2_all[:, np.newaxis, :] * w2[np.newaxis, :, np.newaxis]
    )  # (20, num_subtris, 3)

    # Normalize to unit sphere
    centers = centers / np.linalg.norm(centers, axis=2, keepdims=True)

    # Reshape to (20 * num_subtris, 3)
    centers = centers.reshape(-1, 3)

    return centers.astype(np.float32)


def compute_hemisphere_edge_segments(target_rays: int) -> int:
    """Compute the icosahedron edge segments to get approximately target_rays hemisphere directions.

    The number of hemisphere directions is approximately half of the full sphere directions:
    - n edge segments gives 20 * n^2 full sphere directions
    - Hemisphere has ~10 * n^2 directions

    We solve: 10 * n^2 >= target_rays => n >= sqrt(target_rays / 10)

    Args:
        target_rays: Target number of hemisphere directions.

    Returns:
        Edge segments value that gives at least target_rays hemisphere directions.
    """
    # Direct formula: n = ceil(sqrt(target_rays / 10))
    n = max(1, math.ceil(math.sqrt(target_rays / 10.0)))
    return n


def create_hemisphere_directions(target_rays: int) -> np.ndarray:
    """Create hemisphere directions from a subdivided icosahedron.

    Generates approximately target_rays directions distributed over a hemisphere
    (local Z > 0). These can be rotated to align with any surface normal.

    Args:
        target_rays: Target number of hemisphere directions. The actual count
            will be the smallest icosahedron subdivision that meets or exceeds this.

    Returns:
        (N, 3) array of unit direction vectors in the upper hemisphere (z > 0).
    """
    # Find edge segments that gives enough rays
    edge_segments = compute_hemisphere_edge_segments(target_rays)

    # Generate full sphere directions
    all_directions = create_icosahedron_directions(edge_segments)

    # Filter to upper hemisphere (z > 0)
    hemisphere_directions = all_directions[all_directions[:, 2] > 0]

    return hemisphere_directions


def compute_camera_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute orthonormal camera basis vectors from a view direction.

    Args:
        direction: Unit direction vector the camera is looking along.

    Returns:
        Tuple of (right, up) unit vectors forming an orthonormal basis with direction.

    Raises:
        ValueError: If direction vector has zero or near-zero length.
    """
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        raise ValueError("Direction vector has zero or near-zero length")
    direction = direction / norm

    # Choose an arbitrary up vector that's not parallel to direction
    if abs(direction[1]) < 0.9:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    right = np.cross(world_up, direction)
    right = right / np.linalg.norm(right)

    up = np.cross(direction, right)
    up = up / np.linalg.norm(up)

    return right, up


@wp.kernel
def raycast_orthographic_kernel(
    # Mesh
    mesh_id: wp.uint64,
    # Camera parameters
    cam_origin: wp.vec3,
    cam_dir: wp.vec3,
    cam_right: wp.vec3,
    cam_up: wp.vec3,
    pixel_size: wp.float32,
    resolution: wp.int32,
    max_ray_dist: wp.float32,
    # Voxel hash grid parameters
    inv_voxel_size: wp.float32,
    # Hash table arrays
    keys: wp.array(dtype=wp.uint64),
    active_slots: wp.array(dtype=wp.int32),
    # Accumulator arrays
    sum_positions_x: wp.array(dtype=wp.float32),
    sum_positions_y: wp.array(dtype=wp.float32),
    sum_positions_z: wp.array(dtype=wp.float32),
    sum_normals_x: wp.array(dtype=wp.float32),
    sum_normals_y: wp.array(dtype=wp.float32),
    sum_normals_z: wp.array(dtype=wp.float32),
    counts: wp.array(dtype=wp.int32),
    max_confidences: wp.array(dtype=wp.float32),
    # Two-pass mode: 0 = confidence pass, 1 = position pass
    pass_mode: wp.int32,
    # Cavity camera candidate buffers (optional - pass empty arrays to disable)
    cavity_origins: wp.array(dtype=wp.vec3),
    cavity_directions: wp.array(dtype=wp.vec3),
    cavity_hit_distances: wp.array(dtype=wp.float32),
    cavity_count: wp.array(dtype=wp.int32),  # Single-element array for atomic counter
    max_cavity_candidates: wp.int32,
    camera_offset: wp.float32,
    cavity_prob_scale: wp.float32,  # Scale factor to control acceptance rate
    random_seed: wp.uint32,
):
    """Raycast kernel for orthographic projection with two-pass best-hit selection.

    Uses a two-pass approach for highest quality point cloud extraction:
    - Pass 0 (confidence): Find max confidence per voxel (confidence = |dot(ray, normal)|)
    - Pass 1 (position): Only write position/normal if confidence matches max

    This ensures we keep the most perpendicular hit per voxel, which has the most
    accurate position (least depth error from ray angle).

    Normals are flipped to always point toward the camera (outward from surface).

    Additionally, hits with larger distances (deeper in cavities) are probabilistically
    selected as cavity camera candidates.
    """
    px, py = wp.tid()

    if px >= resolution or py >= resolution:
        return

    # Compute ray origin on the image plane
    # Center the grid around the camera origin
    half_res = wp.float32(resolution) * 0.5
    offset_x = (wp.float32(px) - half_res + 0.5) * pixel_size
    offset_y = (wp.float32(py) - half_res + 0.5) * pixel_size

    ray_origin = cam_origin + cam_right * offset_x + cam_up * offset_y
    ray_direction = cam_dir

    # Query mesh intersection
    query = wp.mesh_query_ray(mesh_id, ray_origin, ray_direction, max_ray_dist)

    if query.result:
        # Compute hit point
        hit_point = ray_origin + ray_direction * query.t

        # Get surface normal - ensure it points toward camera (opposite to ray direction)
        normal = query.normal
        if wp.dot(normal, ray_direction) > 0.0:
            normal = -normal
        normal = wp.normalize(normal)

        # Confidence = how perpendicular the ray is to the surface
        # Higher confidence = more accurate position (less depth error)
        confidence = wp.abs(wp.dot(ray_direction, normal))

        # Get or create voxel slot
        key = compute_voxel_key(hit_point, inv_voxel_size)
        idx = hashtable_find_or_insert(key, keys, active_slots)

        if idx >= 0:
            if pass_mode == 0:
                # Pass 0: Confidence pass - find max confidence per voxel
                wp.atomic_max(max_confidences, idx, confidence)
            else:
                # Pass 1: Position pass - only write if we have the best confidence
                # Use small epsilon for floating point comparison
                max_conf = max_confidences[idx]
                if confidence >= max_conf - 1.0e-6:
                    # We're the best (or tied) - write position and accumulate normal
                    sum_positions_x[idx] = hit_point[0]
                    sum_positions_y[idx] = hit_point[1]
                    sum_positions_z[idx] = hit_point[2]

                    # Accumulate normals (averaging is good for smoothness)
                    wp.atomic_add(sum_normals_x, idx, normal[0])
                    wp.atomic_add(sum_normals_y, idx, normal[1])
                    wp.atomic_add(sum_normals_z, idx, normal[2])
                    wp.atomic_add(counts, idx, 1)

        # Probabilistically select this hit as a cavity camera candidate (only in pass 1)
        # Higher hit distance = higher probability of selection
        if pass_mode == 1 and max_cavity_candidates > 0:
            # Generate random number
            thread_id = wp.uint32(px * resolution + py)
            rand_state = rand_init(random_seed, thread_id)
            rand_val = rand_float(rand_state)

            # Acceptance probability: scaled by hit distance to favor deeper hits
            # cavity_prob_scale controls overall rate to match expected ray count
            accept_prob = (query.t / max_ray_dist) * cavity_prob_scale
            if rand_val < accept_prob:
                # Atomically claim a slot
                slot = wp.atomic_add(cavity_count, 0, 1)
                if slot < max_cavity_candidates:
                    # Origin: offset back along ray direction (guaranteed outside mesh)
                    cavity_origin = hit_point - ray_direction * camera_offset
                    # Direction: use surface normal (points into cavity perpendicular to surface)
                    cavity_dir = normal
                    # cavity_dir = -ray_direction # Alternative: use ray direction
                    cavity_origins[slot] = cavity_origin
                    cavity_directions[slot] = cavity_dir
                    cavity_hit_distances[slot] = query.t


@wp.kernel
def raycast_hemisphere_kernel(
    # Mesh
    mesh_id: wp.uint64,
    # Camera parameters
    cam_origin: wp.vec3,
    cam_right: wp.vec3,
    cam_up: wp.vec3,
    cam_forward: wp.vec3,
    min_ray_dist: wp.float32,
    max_ray_dist: wp.float32,
    # Hemisphere directions (local frame, z > 0)
    hemisphere_dirs: wp.array(dtype=wp.vec3),
    num_directions: wp.int32,
    # Voxel hash grid parameters
    inv_voxel_size: wp.float32,
    # Hash table arrays
    keys: wp.array(dtype=wp.uint64),
    active_slots: wp.array(dtype=wp.int32),
    # Accumulator arrays
    sum_positions_x: wp.array(dtype=wp.float32),
    sum_positions_y: wp.array(dtype=wp.float32),
    sum_positions_z: wp.array(dtype=wp.float32),
    sum_normals_x: wp.array(dtype=wp.float32),
    sum_normals_y: wp.array(dtype=wp.float32),
    sum_normals_z: wp.array(dtype=wp.float32),
    counts: wp.array(dtype=wp.int32),
    max_confidences: wp.array(dtype=wp.float32),
    # Two-pass mode: 0 = confidence pass, 1 = position pass
    pass_mode: wp.int32,
):
    """Raycast kernel for hemisphere projection from a cavity camera (two-pass).

    Uses a two-pass approach for highest quality point cloud extraction:
    - Pass 0 (confidence): Find max confidence per voxel
    - Pass 1 (position): Only write position/normal if confidence matches max

    The camera origin and forward direction come from cavity candidates collected
    during primary raycasting.
    """
    tid = wp.tid()

    if tid >= num_directions:
        return

    # Get local hemisphere direction (z > 0 in local frame)
    local_dir = hemisphere_dirs[tid]

    # Transform to world space: local (x, y, z) -> world (right, up, forward)
    # local z (forward) maps to cam_forward (the surface normal)
    # local x maps to cam_right
    # local y maps to cam_up
    world_dir = cam_right * local_dir[0] + cam_up * local_dir[1] + cam_forward * local_dir[2]
    world_dir = wp.normalize(world_dir)

    # Query mesh intersection
    query = wp.mesh_query_ray(mesh_id, cam_origin, world_dir, max_ray_dist)

    if query.result and query.t > min_ray_dist:
        # Compute hit point
        hit_point = cam_origin + world_dir * query.t

        # Get surface normal - ensure it points toward camera (opposite to ray direction)
        normal = query.normal
        if wp.dot(normal, world_dir) > 0.0:
            normal = -normal
        normal = wp.normalize(normal)

        # Confidence = how perpendicular the ray is to the surface
        confidence = wp.abs(wp.dot(world_dir, normal))

        # Get or create voxel slot
        key = compute_voxel_key(hit_point, inv_voxel_size)
        idx = hashtable_find_or_insert(key, keys, active_slots)

        if idx >= 0:
            if pass_mode == 0:
                # Pass 0: Confidence pass - find max confidence per voxel
                wp.atomic_max(max_confidences, idx, confidence)
            else:
                # Pass 1: Position pass - only write if we have the best confidence
                max_conf = max_confidences[idx]
                if confidence >= max_conf - 1.0e-6:
                    # We're the best (or tied) - write position and accumulate normal
                    sum_positions_x[idx] = hit_point[0]
                    sum_positions_y[idx] = hit_point[1]
                    sum_positions_z[idx] = hit_point[2]

                    wp.atomic_add(sum_normals_x, idx, normal[0])
                    wp.atomic_add(sum_normals_y, idx, normal[1])
                    wp.atomic_add(sum_normals_z, idx, normal[2])
                    wp.atomic_add(counts, idx, 1)


class PointCloudExtractor:
    """Extract dense point clouds with normals from triangle meshes.

    Uses multi-view orthographic raycasting from directions distributed on
    an icosphere to capture the complete surface of a mesh. Normals are
    guaranteed to be consistent (always pointing outward toward the camera).

    Points are accumulated directly into a sparse voxel hash grid during
    raycasting, providing built-in downsampling and dramatically reducing
    memory usage compared to storing all ray hits.

    Optionally, secondary "cavity cameras" can shoot hemisphere rays to improve
    coverage of deep cavities and occluded regions. During primary raycasting,
    hits with large ray distances are probabilistically collected as cavity camera
    candidates (acceptance probability scales with hit distance and is auto-tuned
    to match buffer capacity). The camera origin is offset back along the ray
    direction to guarantee it's outside the mesh. Primary cameras are processed
    in randomized order to ensure even distribution of cavity candidates.

    Args:
        edge_segments: Number of segments per icosahedron edge for camera directions.
            Total views = 20 * n^2. Examples:
            - n=1: 20 views
            - n=2: 80 views
            - n=3: 180 views
            - n=4: 320 views
            - n=5: 500 views
            Higher values provide better coverage with finer control than recursive
            subdivision.
        resolution: Pixel resolution of the orthographic camera (resolution x resolution).
            Must be between 1 and 10000. Also determines the number of rays per cavity
            camera (~resolution^2 hemisphere directions).
        voxel_size: Size of voxels for point accumulation. If None (default), automatically
            computed as 0.1% of the mesh bounding sphere radius. Smaller values give
            denser point clouds but require more memory.
        max_voxels: Maximum number of unique voxels (hash table capacity). If None (default),
            automatically estimated based on voxel_size and mesh extent to keep hash table
            load factor around 50%. Set explicitly if you know your requirements.
        device: Warp device to use for computation.
        seed: Random seed for reproducibility. Controls camera roll angles, camera
            processing order, and cavity candidate selection. Set to None for
            non-deterministic behavior.
        cavity_cameras: Number of secondary hemisphere cameras for improved cavity
            coverage. Set to 0 (default) to disable. Camera positions are collected
            during primary raycasting from hits with large ray distances (deep in
            cavities). Each camera shoots ~resolution^2 rays in a hemisphere pattern,
            with position and direction guaranteed to be outside the mesh surface.

    Note:
        Memory usage is dominated by the voxel hash grid, which scales with
        ``max_voxels`` (~32 bytes per voxel slot), not with ``resolution^2 * num_views``.
        This makes high-resolution extraction practical even on memory-constrained systems.

    Example:
        >>> extractor = PointCloudExtractor(edge_segments=4, resolution=1000)
        >>> points, normals = extractor.extract(vertices, indices)
        >>> print(f"Extracted {len(points)} points with normals")

        >>> # With cavity cameras for better coverage of occluded regions
        >>> extractor = PointCloudExtractor(edge_segments=4, resolution=500, cavity_cameras=100)
        >>> points, normals = extractor.extract(vertices, indices)
    """

    def __init__(
        self,
        edge_segments: int = 2,
        resolution: int = 1000,
        voxel_size: float | None = None,
        max_voxels: int | None = None,
        device: str | None = None,
        seed: int | None = 42,
        cavity_cameras: int = 0,
    ):
        # Validate parameters
        if edge_segments < 1:
            raise ValueError(f"edge_segments must be >= 1, got {edge_segments}")
        if resolution < 1 or resolution > 10000:
            raise ValueError(f"resolution must be between 1 and 10000 (inclusive), got {resolution}")
        if voxel_size is not None and voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {voxel_size}")
        if max_voxels is not None and max_voxels < 1:
            raise ValueError(f"max_voxels must be >= 1, got {max_voxels}")
        if cavity_cameras < 0:
            raise ValueError(f"cavity_cameras must be >= 0, got {cavity_cameras}")

        self.edge_segments = edge_segments
        self.resolution = resolution
        self.voxel_size = voxel_size  # None means auto-compute
        self.max_voxels = max_voxels  # None means auto-compute
        self.device = device if device is not None else wp.get_device()
        self.seed = seed
        self.cavity_cameras = cavity_cameras

        # Pre-compute camera directions for primary pass
        self.directions = create_icosahedron_directions(edge_segments)
        self.num_views = len(self.directions)

        # Pre-compute hemisphere directions for cavity cameras
        if cavity_cameras > 0:
            target_rays = resolution * resolution
            self.hemisphere_directions = create_hemisphere_directions(target_rays)
            self.num_hemisphere_dirs = len(self.hemisphere_directions)
        else:
            self.hemisphere_directions = None
            self.num_hemisphere_dirs = 0

    def extract(
        self,
        vertices: np.ndarray,
        indices: np.ndarray,
        padding_factor: float = 1.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract point cloud from a triangle mesh.

        Performs multi-view orthographic raycasting with online voxel-based accumulation:

        1. Primary pass: Rays from icosphere-distributed cameras (processed in random
           order) with random roll per camera. Hits with large ray distances are
           probabilistically collected as cavity camera candidates, with acceptance
           probability auto-scaled based on expected hit count and buffer capacity.
        2. Secondary pass (if cavity_cameras > 0): Hemisphere rays from sampled cavity
           camera candidates (weighted by hit distance to favor deeper cavities).
           Camera positions are offset back along the original ray direction to
           guarantee they're outside the mesh.

        Points are accumulated into a sparse voxel hash grid, automatically averaging
        multiple hits per voxel. This provides built-in downsampling with minimal memory.

        Args:
            vertices: (N, 3) array of vertex positions.
            indices: (M,) or (M/3, 3) array of triangle indices.
            padding_factor: Multiplier for bounding sphere radius to ensure
                rays start outside the mesh.

        Returns:
            Tuple of (points, normals) where:
            - points: (N, 3) float32 array of world-space intersection points
            - normals: (N, 3) float32 array of world-space surface normals

        Raises:
            ValueError: If vertices or indices are empty, or indices are invalid.
        """
        # Ensure correct shapes
        vertices = np.asarray(vertices, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int32).flatten()

        # Validate inputs
        if len(vertices) == 0:
            raise ValueError("Vertices array cannot be empty")
        if len(indices) == 0:
            raise ValueError("Indices array cannot be empty")
        if len(indices) % 3 != 0:
            raise ValueError(f"Indices length must be a multiple of 3, got {len(indices)}")
        if np.any(indices < 0) or np.any(indices >= len(vertices)):
            raise ValueError(
                f"Indices must be in range [0, {len(vertices)}), got range [{indices.min()}, {indices.max()}]"
            )

        # Compute bounding sphere in original space
        center, radius = compute_bounding_sphere(vertices)

        # Normalize mesh to unit sphere centered at origin
        # This ensures voxel coordinates are always in a predictable range (~±1000)
        # regardless of input mesh scale, preventing hash coordinate overflow
        if radius > 0:
            normalized_vertices = (vertices - center) / radius
        else:
            # Degenerate case: single point or zero-size mesh
            normalized_vertices = vertices - center

        # In normalized space, the mesh fits in unit sphere (radius=1)
        # All parameters are now in normalized space
        normalized_radius = 1.0 if radius > 0 else 1e-6
        padded_radius = normalized_radius * padding_factor

        # Compute pixel size to cover the bounding sphere diameter
        pixel_size = (2.0 * padded_radius) / self.resolution

        # Maximum ray distance (diameter of bounding sphere with padding)
        max_ray_dist = 2.0 * padded_radius * 1.5

        # Voxel size in normalized space
        # Auto: 0.001 gives ~1000 voxels across the diameter, well within ±1M Morton range
        if self.voxel_size is None:
            voxel_size = 0.0005  # Fixed in normalized space
        else:
            # User specified voxel_size is in original space, convert to normalized
            voxel_size = self.voxel_size / radius if radius > 0 else self.voxel_size

        # Compute max_voxels (auto or user-specified)
        if self.max_voxels is None:
            # In normalized space, radius=1, so surface voxels ≈ 4π / voxel_size²
            # Use 4x for hash table load factor (~25%)
            # Cap at 16M to avoid excessive memory for small voxels
            estimated_surface_voxels = 4.0 * np.pi / (voxel_size**2)
            max_voxels = min(1 << 26, max(1 << 20, int(estimated_surface_voxels * 4)))
        else:
            max_voxels = self.max_voxels

        # Create sparse voxel hash grid for accumulation (in normalized space)
        voxel_grid = VoxelHashGrid(
            capacity=max_voxels,
            voxel_size=voxel_size,
            device=self.device,
        )

        # Create Warp mesh from normalized vertices
        wp_vertices = wp.array(normalized_vertices, dtype=wp.vec3, device=self.device)
        wp_indices = wp.array(indices, dtype=wp.int32, device=self.device)
        mesh = wp.Mesh(points=wp_vertices, indices=wp_indices)

        # Create random generator for camera roll angles
        rng = np.random.default_rng(self.seed)

        # Pre-compute all camera bases and random rotations (vectorized)
        # directions is (num_views, 3)
        directions = self.directions

        # Compute camera bases for all directions at once
        # Choose world_up based on direction[1] magnitude
        world_ups = np.where(
            np.abs(directions[:, 1:2]) < 0.9,
            np.array([[0.0, 1.0, 0.0]]),
            np.array([[0.0, 0.0, 1.0]]),
        )  # (num_views, 3)

        # right = cross(world_up, direction), then normalize
        rights = np.cross(world_ups, directions)
        rights /= np.linalg.norm(rights, axis=1, keepdims=True)

        # up = cross(direction, right)
        ups = np.cross(directions, rights)

        # Pre-generate all random roll angles and apply rotation
        thetas = rng.uniform(0, 2 * np.pi, size=self.num_views)
        cos_thetas = np.cos(thetas)[:, np.newaxis]  # (num_views, 1)
        sin_thetas = np.sin(thetas)[:, np.newaxis]

        # Rotated: right' = cos*right + sin*up, up' = cos*up - sin*right
        rights_rot = cos_thetas * rights + sin_thetas * ups
        ups_rot = cos_thetas * ups - sin_thetas * rights

        # Camera origins in normalized space (mesh is centered at origin)
        # Cameras are placed at distance padded_radius from origin along each direction
        cam_origins = -directions * padded_radius  # Origin is at (0,0,0) in normalized space

        # Camera offset for cavity candidates (0.1% of normalized radius = 0.001)
        camera_offset = 0.001

        # Allocate cavity camera candidate buffers if needed
        if self.cavity_cameras > 0:
            # Large buffer to collect candidates - at least 100K or 100x requested cameras
            max_cavity_candidates = max(100_000, self.cavity_cameras * 100)
            cavity_origins = wp.zeros(max_cavity_candidates, dtype=wp.vec3, device=self.device)
            cavity_directions = wp.zeros(max_cavity_candidates, dtype=wp.vec3, device=self.device)
            cavity_hit_distances = wp.zeros(max_cavity_candidates, dtype=wp.float32, device=self.device)
            cavity_count = wp.zeros(1, dtype=wp.int32, device=self.device)

            # Calculate probability scale to target ~2x buffer size candidates
            # Total rays = num_views * resolution^2, assume ~50% hit rate
            total_expected_hits = self.num_views * self.resolution * self.resolution * 0.5
            # Average hit distance ratio is ~0.5, so base acceptance would be 0.5
            # We want: 0.5 * prob_scale * total_hits ≈ 2 * max_cavity_candidates
            # prob_scale = 4 * max_cavity_candidates / total_hits
            cavity_prob_scale = float(4.0 * max_cavity_candidates / max(total_expected_hits, 1.0))
            # Clamp to reasonable range
            cavity_prob_scale = min(1.0, max(1e-6, cavity_prob_scale))
        else:
            # Empty arrays when cavity cameras disabled
            max_cavity_candidates = 0
            cavity_origins = wp.empty(0, dtype=wp.vec3, device=self.device)
            cavity_directions = wp.empty(0, dtype=wp.vec3, device=self.device)
            cavity_hit_distances = wp.empty(0, dtype=wp.float32, device=self.device)
            cavity_count = wp.zeros(1, dtype=wp.int32, device=self.device)
            cavity_prob_scale = 0.0

        # Randomize camera order to get even distribution of cavity candidates
        # (prevents later cameras from always overflowing the buffer)
        camera_order = rng.permutation(self.num_views)

        # Two-pass approach for best-hit selection:
        # Pass 0: Find max confidence (|dot(ray, normal)|) per voxel across all cameras
        # Pass 1: Only write position/normal for hits that match max confidence

        # Helper function to run all cameras with given pass mode
        def run_primary_cameras(pass_mode: int):
            for i in camera_order:
                direction = directions[i]
                right = rights_rot[i]
                up = ups_rot[i]
                cam_origin = cam_origins[i]

                # Different random seed per view for cavity candidate selection
                random_seed = rng.integers(0, 2**31, dtype=np.uint32)

                wp.launch(
                    kernel=raycast_orthographic_kernel,
                    dim=(self.resolution, self.resolution),
                    inputs=[
                        mesh.id,
                        wp.vec3(cam_origin[0], cam_origin[1], cam_origin[2]),
                        wp.vec3(direction[0], direction[1], direction[2]),
                        wp.vec3(right[0], right[1], right[2]),
                        wp.vec3(up[0], up[1], up[2]),
                        float(pixel_size),
                        self.resolution,
                        float(max_ray_dist),
                        float(voxel_grid.inv_voxel_size),
                        voxel_grid.keys,
                        voxel_grid.active_slots,
                        voxel_grid.sum_positions_x,
                        voxel_grid.sum_positions_y,
                        voxel_grid.sum_positions_z,
                        voxel_grid.sum_normals_x,
                        voxel_grid.sum_normals_y,
                        voxel_grid.sum_normals_z,
                        voxel_grid.counts,
                        voxel_grid.max_confidences,
                        pass_mode,
                        cavity_origins,
                        cavity_directions,
                        cavity_hit_distances,
                        cavity_count,
                        max_cavity_candidates,
                        float(camera_offset),
                        float(cavity_prob_scale),
                        int(random_seed),
                    ],
                    device=self.device,
                )

        # Pass 0: Find max confidence per voxel
        run_primary_cameras(pass_mode=0)

        # Pass 1: Write positions for best-confidence hits
        run_primary_cameras(pass_mode=1)

        # Check hash table load factor and warn if too high
        num_voxels_after_primary = voxel_grid.get_num_voxels()
        load_factor = num_voxels_after_primary / voxel_grid.capacity
        if load_factor > 0.7:
            warnings.warn(
                f"Voxel hash table is {load_factor:.0%} full ({num_voxels_after_primary}/{voxel_grid.capacity}). "
                f"This may cause slowdowns. Consider increasing max_voxels or using a larger voxel_size.",
                stacklevel=2,
            )

        # Secondary pass: cavity cameras for improved coverage of occluded regions
        if self.cavity_cameras > 0:
            wp.synchronize()

            # Get the number of cavity candidates that were attempted to write
            total_attempts = int(cavity_count.numpy()[0])
            # Clamp to buffer size (some may have been dropped due to overflow)
            num_candidates = min(total_attempts, max_cavity_candidates)

            # Report buffer status
            if total_attempts > max_cavity_candidates:
                overflow_count = total_attempts - max_cavity_candidates
                print(
                    f"Cavity candidates: {num_candidates:,} collected, "
                    f"{overflow_count:,} dropped (buffer overflow, not critical)"
                )
            elif num_candidates > 0:
                print(f"Cavity candidates: {num_candidates:,} collected")

            if num_candidates > 0:
                # Prepare hemisphere directions on GPU
                wp_hemisphere_dirs = wp.array(self.hemisphere_directions, dtype=wp.vec3, device=self.device)

                # Minimum ray distance to avoid self-occlusion
                min_ray_dist = camera_offset * 2.0

                # Get cavity candidate data from GPU
                origins_np = cavity_origins.numpy()[:num_candidates]
                directions_np = cavity_directions.numpy()[:num_candidates]
                hit_dists_np = cavity_hit_distances.numpy()[:num_candidates]

                # Sample cavity cameras with weighted random choice
                # Weight by hit distance to favor deeper cavities
                weights = hit_dists_np.copy()
                weights_sum = weights.sum()
                if weights_sum > 0:
                    weights /= weights_sum
                else:
                    weights = np.ones(num_candidates) / num_candidates

                # Sample up to cavity_cameras, but no more than available candidates
                num_to_sample = min(self.cavity_cameras, num_candidates)
                sample_indices = rng.choice(num_candidates, size=num_to_sample, p=weights, replace=True)

                # Pre-generate all random roll angles
                thetas = rng.uniform(0, 2 * np.pi, size=num_to_sample)

                # Pre-compute camera bases for all sampled cavity cameras
                cavity_cam_data = []
                for i in range(num_to_sample):
                    sample_idx = sample_indices[i]
                    cam_origin = origins_np[sample_idx]
                    cam_forward = directions_np[sample_idx]  # Already points into mesh

                    # Compute camera basis (cam_forward is the forward direction)
                    right, up = compute_camera_basis(cam_forward)

                    # Apply random roll around forward direction
                    theta = thetas[i]
                    cos_theta = np.cos(theta)
                    sin_theta = np.sin(theta)
                    right_rot = cos_theta * right + sin_theta * up
                    up_rot = cos_theta * up - sin_theta * right
                    right, up = right_rot, up_rot

                    cavity_cam_data.append((cam_origin, right, up, cam_forward))

                # Helper function to run all cavity cameras with given pass mode
                def run_cavity_cameras(pass_mode: int):
                    for cam_origin, right, up, cam_forward in cavity_cam_data:
                        wp.launch(
                            kernel=raycast_hemisphere_kernel,
                            dim=self.num_hemisphere_dirs,
                            inputs=[
                                mesh.id,
                                wp.vec3(cam_origin[0], cam_origin[1], cam_origin[2]),
                                wp.vec3(right[0], right[1], right[2]),
                                wp.vec3(up[0], up[1], up[2]),
                                wp.vec3(cam_forward[0], cam_forward[1], cam_forward[2]),
                                float(min_ray_dist),
                                float(max_ray_dist),
                                wp_hemisphere_dirs,
                                self.num_hemisphere_dirs,
                                float(voxel_grid.inv_voxel_size),
                                voxel_grid.keys,
                                voxel_grid.active_slots,
                                voxel_grid.sum_positions_x,
                                voxel_grid.sum_positions_y,
                                voxel_grid.sum_positions_z,
                                voxel_grid.sum_normals_x,
                                voxel_grid.sum_normals_y,
                                voxel_grid.sum_normals_z,
                                voxel_grid.counts,
                                voxel_grid.max_confidences,
                                pass_mode,
                            ],
                            device=self.device,
                        )

                # Two-pass for cavity cameras as well
                run_cavity_cameras(pass_mode=0)  # Confidence pass
                run_cavity_cameras(pass_mode=1)  # Position pass

        # Finalize voxel grid to get averaged points and normals
        wp.synchronize()
        final_num_voxels = voxel_grid.get_num_voxels()
        final_load_factor = final_num_voxels / voxel_grid.capacity
        print(
            f"Voxel grid: {final_num_voxels:,} voxels, "
            f"{final_load_factor:.1%} load factor "
            f"({final_num_voxels:,}/{voxel_grid.capacity:,})"
        )

        points_np, normals_np, _num_points = voxel_grid.finalize()

        # Transform points back from normalized space to original space
        # normalized = (original - center) / radius
        # original = normalized * radius + center
        if radius > 0:
            points_np = points_np * radius + center
        else:
            points_np = points_np + center
        # Normals are unit vectors, no transformation needed

        return points_np, normals_np


class SurfaceReconstructor:
    """Reconstruct triangle meshes from point clouds using Poisson reconstruction.

    Uses Open3D's implementation of Screened Poisson Surface Reconstruction.

    Note:
        When used with PointCloudExtractor, the point cloud is already downsampled
        via the built-in voxel hash grid accumulation. No additional downsampling
        is needed.

    Args:
        depth: Octree depth for Poisson reconstruction (higher = more detail, slower).
            Default is 10, which provides good detail.
        scale: Scale factor for the reconstruction bounding box. Default 1.1.
        linear_fit: Use linear interpolation for iso-surface extraction. Default False.
        density_threshold_quantile: Quantile for removing low-density vertices
            (boundary artifacts). Default 0.01 removes bottom 1%.
        simplify_ratio: Target ratio to reduce triangle count (e.g., 0.1 = keep 10%).
            If None, no simplification is performed. Uses quadric decimation which
            preserves shape well and removes unnecessary triangles in flat areas.
        target_triangles: Target number of triangles after simplification.
            Overrides simplify_ratio if both are set.
        simplify_tolerance: Maximum geometric error allowed during simplification,
            as a fraction of the mesh bounding box diagonal (e.g., 0.0000001 = 0.00001% of diagonal).
            Only coplanar/nearly-coplanar triangles within this tolerance are merged.
            The mesh keeps all triangles it needs to stay within tolerance.
            This is the recommended option for quality-preserving simplification.
            Overrides simplify_ratio and target_triangles if set.
        fast_simplification: If True (default), use pyfqmr for fast mesh simplification.
            If False, use Open3D's simplify_quadric_decimation which may produce
            slightly higher quality results but is significantly slower.

    Example:
        >>> extractor = PointCloudExtractor(edge_segments=4, resolution=1000)
        >>> points, normals = extractor.extract(vertices, indices)
        >>> reconstructor = SurfaceReconstructor(depth=10, simplify_tolerance=1e-7)
        >>> mesh = reconstructor.reconstruct(points, normals)
        >>> print(f"Reconstructed {len(mesh.indices) // 3} triangles")
    """

    def __init__(
        self,
        depth: int = 10,
        scale: float = 1.1,
        linear_fit: bool = False,
        density_threshold_quantile: float = 0.0,
        simplify_ratio: float | None = None,
        target_triangles: int | None = None,
        simplify_tolerance: float | None = None,
        fast_simplification: bool = True,
    ):
        # Validate parameters
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        if not (0.0 <= density_threshold_quantile <= 1.0):
            raise ValueError(f"density_threshold_quantile must be in [0, 1], got {density_threshold_quantile}")
        if simplify_ratio is not None and (simplify_ratio <= 0 or simplify_ratio > 1):
            raise ValueError(f"simplify_ratio must be in (0, 1], got {simplify_ratio}")
        if target_triangles is not None and target_triangles < 1:
            raise ValueError(f"target_triangles must be >= 1, got {target_triangles}")
        if simplify_tolerance is not None and simplify_tolerance < 0:
            raise ValueError(f"simplify_tolerance must be >= 0, got {simplify_tolerance}")

        self.depth = depth
        self.scale = scale
        self.linear_fit = linear_fit
        self.density_threshold_quantile = density_threshold_quantile
        self.simplify_ratio = simplify_ratio
        self.target_triangles = target_triangles
        self.simplify_tolerance = simplify_tolerance
        self.fast_simplification = fast_simplification

    def reconstruct(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        verbose: bool = True,
    ) -> Mesh:
        """Reconstruct a triangle mesh from a point cloud.

        Args:
            points: (N, 3) array of point positions.
            normals: (N, 3) array of surface normals (should be unit length).
            verbose: Print progress information.

        Returns:
            Mesh containing vertices and triangle indices.
        """
        import open3d as o3d  # noqa: PLC0415  # lazy import, open3d is optional

        points = np.asarray(points, dtype=np.float32)
        normals = np.asarray(normals, dtype=np.float32)

        # Validate inputs
        if len(points) == 0:
            raise ValueError("Cannot reconstruct from empty point cloud")

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

        # Run Poisson reconstruction
        if verbose:
            print(f"Running Poisson reconstruction (depth={self.depth})...")

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=self.depth,
            scale=self.scale,
            linear_fit=self.linear_fit,
        )

        # Remove low-density vertices (boundary artifacts)
        if self.density_threshold_quantile > 0:
            densities = np.asarray(densities)
            threshold = np.quantile(densities, self.density_threshold_quantile)
            vertices_to_remove = densities < threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)

        num_triangles_before = len(mesh.triangles)

        if verbose:
            print(f"Reconstructed mesh: {len(mesh.vertices)} vertices, {num_triangles_before} triangles")

        # Simplify mesh if requested
        needs_simplification = (
            self.simplify_tolerance is not None or self.target_triangles is not None or self.simplify_ratio is not None
        )

        if needs_simplification:
            if self.fast_simplification:
                # Use pyfqmr (fast quadric mesh reduction)
                vertices, faces = self._simplify_pyfqmr(mesh, num_triangles_before, verbose)
            else:
                # Use Open3D (slower but potentially higher quality)
                vertices, faces = self._simplify_open3d(mesh, num_triangles_before, verbose)
        else:
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.triangles, dtype=np.int32)

        # Convert to output format
        indices = faces.flatten().astype(np.int32)

        if verbose and needs_simplification:
            num_triangles_after = len(faces)
            if num_triangles_before > 0:
                reduction = 100 * (1 - num_triangles_after / num_triangles_before)
                print(
                    f"Simplified mesh: {len(vertices)} vertices, {num_triangles_after} triangles ({reduction:.1f}% reduction)"
                )
            else:
                print(f"Simplified mesh: {len(vertices)} vertices, {num_triangles_after} triangles")

        return Mesh(vertices=vertices, indices=indices, compute_inertia=False)

    def _simplify_pyfqmr(self, mesh, num_triangles_before: int, verbose: bool) -> tuple[np.ndarray, np.ndarray]:
        """Simplify mesh using pyfqmr (fast)."""
        from pyfqmr import Simplify  # noqa: PLC0415  # lazy import

        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.triangles, dtype=np.int32)

        mesh_simplifier = Simplify()
        mesh_simplifier.setMesh(vertices, faces)

        if self.simplify_tolerance is not None:
            # Error-based: use lossless mode with epsilon threshold
            # Scale tolerance by mesh bounding box diagonal to make it scale-independent
            min_coords = vertices.min(axis=0)
            max_coords = vertices.max(axis=0)
            diagonal = np.linalg.norm(max_coords - min_coords)
            absolute_tolerance = self.simplify_tolerance * diagonal
            if verbose:
                print(
                    f"Simplifying mesh with pyfqmr (tolerance={self.simplify_tolerance} = {absolute_tolerance:.6f} absolute, diagonal={diagonal:.4f})..."
                )
            mesh_simplifier.simplify_mesh_lossless(epsilon=absolute_tolerance, verbose=False)
        elif self.target_triangles is not None:
            target = self.target_triangles
            if verbose:
                print(f"Simplifying mesh with pyfqmr to {target} triangles...")
            mesh_simplifier.simplify_mesh(target_count=target, verbose=False)
        elif self.simplify_ratio is not None:
            target = int(num_triangles_before * self.simplify_ratio)
            if verbose:
                print(f"Simplifying mesh with pyfqmr to {self.simplify_ratio:.1%} ({target} triangles)...")
            mesh_simplifier.simplify_mesh(target_count=target, verbose=False)

        vertices, faces, _ = mesh_simplifier.getMesh()
        return np.asarray(vertices, dtype=np.float32), faces

    def _simplify_open3d(self, mesh, num_triangles_before: int, verbose: bool) -> tuple[np.ndarray, np.ndarray]:
        """Simplify mesh using Open3D (higher quality, slower)."""
        if self.simplify_tolerance is not None:
            # Error-based: aggressively target 1 triangle, but stop when error exceeds tolerance
            bbox = mesh.get_axis_aligned_bounding_box()
            diagonal = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
            # Open3D QEM uses squared distances, so square the tolerance
            absolute_tolerance = (self.simplify_tolerance * diagonal) ** 2
            if verbose:
                print(
                    f"Simplifying mesh with Open3D (tolerance={self.simplify_tolerance} = {self.simplify_tolerance * diagonal:.6f} absolute, diagonal={diagonal:.4f})..."
                )
            mesh = mesh.simplify_quadric_decimation(
                target_number_of_triangles=1,
                maximum_error=absolute_tolerance,
            )
        elif self.target_triangles is not None:
            target = self.target_triangles
            if verbose:
                print(f"Simplifying mesh with Open3D to {target} triangles...")
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
        elif self.simplify_ratio is not None:
            target = int(num_triangles_before * self.simplify_ratio)
            if verbose:
                print(f"Simplifying mesh with Open3D to {self.simplify_ratio:.1%} ({target} triangles)...")
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.int32)
        return vertices, faces


def extract_largest_island(
    vertices: np.ndarray,
    indices: np.ndarray,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the largest connected component (island) from a mesh.

    Uses edge-based connectivity: triangles sharing an edge are considered connected.
    Optimized using NumPy vectorization and SciPy sparse connected components.

    This is useful as post-processing after Poisson reconstruction, which can
    sometimes create small floating fragments in areas with sparse point coverage.

    Args:
        vertices: Vertex positions, shape (N, 3).
        indices: Triangle indices, shape (M*3,) or (M, 3).
        verbose: Print progress information. Default is False.

    Returns:
        Tuple of (new_vertices, new_indices) containing only the largest island.
        Indices are returned as a flattened array (M*3,).
    """
    from scipy import sparse  # noqa: PLC0415

    # Ensure indices are flattened
    indices = np.asarray(indices).flatten()
    num_triangles = len(indices) // 3

    if num_triangles == 0:
        return vertices, indices

    # Reshape indices to (num_triangles, 3)
    triangles = indices.reshape(-1, 3)

    # Build edges array: each triangle contributes 3 edges
    # Edge format: (min_vertex, max_vertex) for consistent ordering
    # Shape: (num_triangles * 3, 2)
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]

    edges = np.stack(
        [
            np.stack([np.minimum(v0, v1), np.maximum(v0, v1)], axis=1),
            np.stack([np.minimum(v1, v2), np.maximum(v1, v2)], axis=1),
            np.stack([np.minimum(v2, v0), np.maximum(v2, v0)], axis=1),
        ],
        axis=1,
    ).reshape(-1, 2)  # Shape: (num_triangles * 3, 2)

    # Triangle index for each edge
    tri_indices = np.repeat(np.arange(num_triangles), 3)

    # Encode edges as single integers for fast grouping
    # Use a large multiplier to avoid collisions
    max_vertex = int(vertices.shape[0])
    edge_keys = edges[:, 0].astype(np.int64) * max_vertex + edges[:, 1].astype(np.int64)

    # Sort edges to group identical edges together
    sort_idx = np.argsort(edge_keys)
    sorted_keys = edge_keys[sort_idx]
    sorted_tris = tri_indices[sort_idx]

    # Find where consecutive edges are the same (shared edges)
    same_as_next = sorted_keys[:-1] == sorted_keys[1:]

    # For each shared edge, connect the two triangles
    # Get pairs of triangles that share an edge
    tri_a = sorted_tris[:-1][same_as_next]
    tri_b = sorted_tris[1:][same_as_next]

    # Build sparse adjacency matrix for triangles
    # Each shared edge creates a connection between two triangles
    if len(tri_a) > 0:
        # Create symmetric adjacency matrix
        row = np.concatenate([tri_a, tri_b])
        col = np.concatenate([tri_b, tri_a])
        data = np.ones(len(row), dtype=np.int8)
        adjacency = sparse.csr_matrix((data, (row, col)), shape=(num_triangles, num_triangles))
    else:
        # No shared edges - each triangle is its own component
        adjacency = sparse.csr_matrix((num_triangles, num_triangles), dtype=np.int8)

    # Find connected components using SciPy (highly optimized C code)
    num_components, labels = sparse.csgraph.connected_components(adjacency, directed=False)

    # If only one component, return as-is
    if num_components == 1:
        if verbose:
            print("Island filtering: 1 component (mesh is fully connected)")
        return vertices, indices

    # Count triangles per component
    component_sizes = np.bincount(labels)
    largest_component = np.argmax(component_sizes)
    largest_size = component_sizes[largest_component]

    # Get mask of triangles to keep
    keep_mask = labels == largest_component
    keep_triangles = triangles[keep_mask]

    # Find unique vertices used by kept triangles
    used_vertices = np.unique(keep_triangles.flatten())

    # Create vertex remapping
    vertex_remap = np.full(len(vertices), -1, dtype=np.int32)
    vertex_remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int32)

    # Remap triangle indices
    new_indices = vertex_remap[keep_triangles.flatten()]
    new_vertices = vertices[used_vertices]

    if verbose:
        print(f"Island filtering: {num_components} components found")
        print(f"  Kept largest: {largest_size} triangles ({largest_size * 100.0 / num_triangles:.1f}%)")
        print(f"  Removed: {num_triangles - largest_size} triangles from {num_components - 1} smaller islands")

    return new_vertices, new_indices


def remesh_poisson(
    vertices,
    faces,
    # Point cloud extraction parameters
    edge_segments: int = 2,
    resolution: int = 1000,
    voxel_size: float | None = None,
    cavity_cameras: int = 0,
    # Surface reconstruction parameters
    depth: int = 10,
    density_threshold_quantile: float = 0.0,
    simplify_tolerance: float | None = 1e-7,
    simplify_ratio: float | None = None,
    target_triangles: int | None = None,
    fast_simplification: bool = True,
    # Post-processing parameters
    keep_largest_island: bool = True,
    # Control parameters
    device: str | None = None,
    seed: int | None = 42,
    verbose: bool = False,
):
    """Remesh a 3D triangular surface mesh using Poisson surface reconstruction.

    This function extracts a dense point cloud from the input mesh using GPU-accelerated
    multi-view raycasting, then reconstructs a clean, watertight mesh using Screened
    Poisson Surface Reconstruction.

    This is useful for repairing meshes with:
        - Inconsistent or flipped triangle winding
        - Missing or incorrect vertex normals
        - Non-manifold geometry
        - Holes or self-intersections

    Args:
        vertices: A numpy array of shape (N, 3) containing the vertex positions.
        faces: A numpy array of shape (M, 3) containing the vertex indices of the faces.
        edge_segments: Number of segments per icosahedron edge for camera directions.
            Total views = 20 * n^2. Higher values provide better surface coverage.
            Default is 2 (80 views).
        resolution: Pixel resolution of the orthographic camera (resolution x resolution).
            Higher values capture finer details. Default is 1000.
        voxel_size: Size of voxels for point accumulation. If None (default), automatically
            computed based on mesh size.
        cavity_cameras: Number of secondary hemisphere cameras for improved cavity
            coverage. Set to 0 (default) to disable.
        depth: Octree depth for Poisson reconstruction (higher = more detail, slower).
            Default is 10.
        density_threshold_quantile: Quantile for removing low-density vertices
            (boundary artifacts). Default 0.0 keeps all vertices.
        simplify_tolerance: Maximum geometric error allowed during simplification,
            as a fraction of the mesh bounding box diagonal. Default is 1e-7.
            Set to None to disable simplification.
        simplify_ratio: Target ratio to reduce triangle count (e.g., 0.1 = keep 10%).
            Only used if simplify_tolerance is None.
        target_triangles: Target number of triangles after simplification.
            Only used if simplify_tolerance and simplify_ratio are None.
        fast_simplification: If True (default), use pyfqmr for fast mesh simplification.
            If False, use Open3D (slower but potentially higher quality).
        keep_largest_island: If True (default), keep only the largest connected component
            after reconstruction. This removes small floating fragments that can occur
            in areas with sparse point coverage.
        device: Warp device for GPU computation. Default uses the default Warp device.
        seed: Random seed for reproducibility. Default is 42.
        verbose: Print progress information. Default is False.

    Returns:
        A tuple (vertices, faces) containing the remeshed mesh:
        - vertices: A numpy array of shape (K, 3) with vertex positions.
        - faces: A numpy array of shape (L, 3) with triangle indices.

    Example:
        >>> import numpy as np
        >>> from newton._src.geometry.remesh import remesh_poisson
        >>> # Remesh with default settings
        >>> new_verts, new_faces = remesh_poisson(vertices, faces)
        >>> # Remesh with higher quality (more views, finer resolution)
        >>> new_verts, new_faces = remesh_poisson(vertices, faces, edge_segments=4, resolution=2000)
    """
    # Extract point cloud
    extractor = PointCloudExtractor(
        edge_segments=edge_segments,
        resolution=resolution,
        voxel_size=voxel_size,
        device=device,
        seed=seed,
        cavity_cameras=cavity_cameras,
    )
    points, normals = extractor.extract(vertices, faces.flatten())

    if verbose:
        print(f"Extracted {len(points)} points")

    # Reconstruct mesh
    reconstructor = SurfaceReconstructor(
        depth=depth,
        density_threshold_quantile=density_threshold_quantile,
        simplify_tolerance=simplify_tolerance,
        simplify_ratio=simplify_ratio,
        target_triangles=target_triangles,
        fast_simplification=fast_simplification,
    )
    mesh = reconstructor.reconstruct(points, normals, verbose=verbose)

    # Get vertices and faces from reconstructed mesh
    new_vertices = mesh.vertices
    new_faces = mesh.indices.reshape(-1, 3)

    # Post-processing: keep only the largest connected component
    if keep_largest_island:
        new_vertices, new_indices = extract_largest_island(new_vertices, new_faces, verbose=verbose)
        new_faces = new_indices.reshape(-1, 3)

    return new_vertices, new_faces
