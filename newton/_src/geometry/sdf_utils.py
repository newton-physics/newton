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

from collections.abc import Sequence

import numpy as np
import warp as wp

from .types import Mesh


@wp.struct
class SDFData:
    """Encapsulates all data needed for SDF-based collision detection.

    Contains both sparse (narrow band) and coarse (background) SDF volumes
    with the same spatial extents but different resolutions.
    """

    # Sparse (narrow band) SDF - high resolution near surface
    sparse_sdf_ptr: wp.uint64
    sparse_voxel_size: wp.vec3

    # Coarse (background) SDF - 8x8x8 covering entire volume
    coarse_sdf_ptr: wp.uint64
    coarse_voxel_size: wp.vec3

    # Shared extents (same for both volumes)
    center: wp.vec3
    half_extents: wp.vec3

    # Background value used for unallocated voxels in the sparse SDF
    background_value: wp.float32


# Default background value for unallocated voxels in sparse SDF
SDF_BACKGROUND_VALUE = 1000.0


def create_empty_sdf_data() -> SDFData:
    """Create an empty SDFData struct for shapes that don't need SDF collision.

    Returns:
        An SDFData struct with zeroed pointers and extents.
    """
    sdf_data = SDFData()
    sdf_data.sparse_sdf_ptr = wp.uint64(0)
    sdf_data.sparse_voxel_size = wp.vec3(0.0, 0.0, 0.0)
    sdf_data.coarse_sdf_ptr = wp.uint64(0)
    sdf_data.coarse_voxel_size = wp.vec3(0.0, 0.0, 0.0)
    sdf_data.center = wp.vec3(0.0, 0.0, 0.0)
    sdf_data.half_extents = wp.vec3(0.0, 0.0, 0.0)
    sdf_data.background_value = SDF_BACKGROUND_VALUE
    return sdf_data


@wp.func
def int_to_vec3f(x: wp.int32, y: wp.int32, z: wp.int32):
    return wp.vec3f(float(x), float(y), float(z))


@wp.func
def get_distance_to_mesh(mesh: wp.uint64, point: wp.vec3, max_dist: wp.float32):
    res = wp.mesh_query_point_sign_winding_number(mesh, point, max_dist)
    if res.result:
        closest = wp.mesh_eval_position(mesh, res.face, res.u, res.v)
        vec_to_surface = closest - point
        return res.sign * wp.length(vec_to_surface)
    return max_dist


@wp.kernel
def sdf_from_mesh_kernel(
    mesh: wp.uint64,
    sdf: wp.uint64,
    tile_points: wp.array(dtype=wp.vec3i),
    thickness: wp.float32,
):
    """
    Populate SDF grid from triangle mesh.
    Only processes specified tiles. Launch with dim=(num_tiles, 8, 8, 8).
    """
    tile_idx, local_x, local_y, local_z = wp.tid()

    # Get the tile origin and compute global voxel coordinates
    tile_origin = tile_points[tile_idx]
    x_id = tile_origin[0] + local_x
    y_id = tile_origin[1] + local_y
    z_id = tile_origin[2] + local_z

    sample_pos = wp.volume_index_to_world(sdf, int_to_vec3f(x_id, y_id, z_id))
    signed_distance = get_distance_to_mesh(mesh, sample_pos, 10000.0)
    signed_distance -= thickness
    wp.volume_store(sdf, x_id, y_id, z_id, signed_distance)


@wp.kernel
def check_tile_occupied_mesh_kernel(
    mesh: wp.uint64,
    tile_points: wp.array(dtype=wp.vec3f),
    threshold: wp.vec2f,
    tile_occupied: wp.array(dtype=bool),
):
    tid = wp.tid()
    sample_pos = tile_points[tid]

    signed_distance = get_distance_to_mesh(mesh, sample_pos, 10000.0)
    is_occupied = wp.bool(False)
    if wp.sign(signed_distance) > 0.0:
        is_occupied = signed_distance < threshold[1]
    else:
        is_occupied = signed_distance > threshold[0]
    tile_occupied[tid] = is_occupied


def compute_sdf(
    mesh_src: Mesh,
    shape_scale: Sequence[float] = (1.0, 1.0, 1.0),
    shape_thickness: float = 0.0,
    narrow_band_distance: Sequence[float] = (-0.1, 0.1),
    margin: float = 0.05,
    target_voxel_size: float | None = None,
    max_dims: int = 64,
    verbose: bool = False,
) -> tuple[SDFData, wp.Volume | None, wp.Volume | None]:
    """Compute sparse and coarse SDF volumes for a mesh.

    Args:
        mesh_src: Mesh source with vertices and indices.
        shape_scale: Scale factors for the mesh. Default (1.0, 1.0, 1.0).
        shape_thickness: Thickness offset to subtract from SDF values.
        narrow_band_distance: Tuple of (inner, outer) distances for narrow band.
        margin: Margin to add to bounding box.
        target_voxel_size: Target voxel size for sparse SDF grid. If None, computed as max_extent/max_dims.
        max_dims: Maximum dimension for sparse SDF grid when target_voxel_size is None. Default 64.
        verbose: Print debug info.

    Returns:
        Tuple of (sdf_data, sparse_volume, coarse_volume) where:
        - sdf_data: SDFData struct with pointers and extents
        - sparse_volume: wp.Volume object for sparse SDF (keep alive for reference counting)
        - coarse_volume: wp.Volume object for coarse SDF (keep alive for reference counting)

    Raises:
        RuntimeError: If CUDA is not available.
    """
    if not wp.is_cuda_available():
        raise RuntimeError("compute_sdf requires CUDA but no CUDA device is available")

    assert isinstance(narrow_band_distance, Sequence), "narrow_band_distance must be a tuple of two floats"
    assert len(narrow_band_distance) == 2, "narrow_band_distance must be a tuple of two floats"
    assert narrow_band_distance[0] < 0.0 < narrow_band_distance[1], (
        "narrow_band_distance[0] must be less than 0.0 and narrow_band_distance[1] must be greater than 0.0"
    )
    assert margin > 0, "margin must be > 0"

    offset = margin + shape_thickness
    # bake scale into SDF
    verts = mesh_src.vertices * np.array(shape_scale)[None, :]
    pos = wp.array(verts, dtype=wp.vec3)
    indices = wp.array(mesh_src.indices, dtype=wp.int32)

    mesh = wp.Mesh(points=pos, indices=indices, support_winding_number=True)
    m_id = mesh.id

    min_ext = np.min(verts, axis=0).tolist()
    max_ext = np.max(verts, axis=0).tolist()

    min_ext = np.array(min_ext) - offset
    max_ext = np.array(max_ext) + offset
    ext = max_ext - min_ext

    # Compute center and half_extents for oriented bounding box collision detection
    center = (min_ext + max_ext) * 0.5
    half_extents = (max_ext - min_ext) * 0.5

    # Calculate uniform voxel size based on the longest dimension
    max_extent = np.max(ext)
    # If target_voxel_size not specified, compute from max_dims
    if target_voxel_size is None:
        # Warp volumes are allocated in tiles of 8 voxels
        assert max_dims % 8 == 0, "max_dims must be divisible by 8 for SDF volume allocation"
        # we store coords as uint16
        assert max_dims < 1 << 16, f"max_dims must be less than {1 << 16}"
        target_voxel_size = max_extent / max_dims
    voxel_size_max_ext = target_voxel_size
    grid_tile_nums = (ext / voxel_size_max_ext).astype(int) // 8
    grid_tile_nums = np.maximum(grid_tile_nums, 1)
    grid_dims = grid_tile_nums * 8

    actual_voxel_size = ext / (grid_dims - 1)

    if verbose:
        print(
            f"Extent: {ext}, Grid dims: {grid_dims}, voxel size: {actual_voxel_size} target_voxel_size: {target_voxel_size}"
        )

    tile_max = np.around((max_ext - min_ext) / actual_voxel_size).astype(np.int32) // 8
    tiles = np.array(
        [[i, j, k] for i in range(tile_max[0] + 1) for j in range(tile_max[1] + 1) for k in range(tile_max[2] + 1)],
        dtype=np.int32,
    )

    tile_points = tiles * 8

    tile_center_points_world = (tile_points + 4) * actual_voxel_size + min_ext
    tile_center_points_world = wp.array(tile_center_points_world, dtype=wp.vec3f)
    tile_occupied = wp.zeros(len(tile_points), dtype=bool)

    # for each tile point, check if it should be marked as occupied
    tile_radius = np.linalg.norm(4 * actual_voxel_size)
    threshold = wp.vec2f(narrow_band_distance[0] - tile_radius, narrow_band_distance[1] + tile_radius)

    wp.launch(
        check_tile_occupied_mesh_kernel,
        dim=(len(tile_points)),
        inputs=[m_id, tile_center_points_world, threshold],
        outputs=[tile_occupied],
    )

    if verbose:
        print("Occupancy: ", tile_occupied.numpy().sum() / len(tile_points))

    tile_points = tile_points[tile_occupied.numpy()]
    tile_points_wp = wp.array(tile_points, dtype=wp.vec3i)

    sparse_volume = wp.Volume.allocate_by_tiles(
        tile_points=tile_points_wp,
        voxel_size=wp.vec3(actual_voxel_size),
        translation=wp.vec3(min_ext),
        bg_value=SDF_BACKGROUND_VALUE,
    )

    # populate the sparse volume with the sdf values
    # Only process allocated tiles (num_tiles x 8x8x8)
    num_allocated_tiles = len(tile_points)
    wp.launch(
        sdf_from_mesh_kernel,
        dim=(num_allocated_tiles, 8, 8, 8),
        inputs=[m_id, sparse_volume.id, tile_points_wp, shape_thickness],
    )

    # Create coarse background SDF (8x8x8 voxels = one tile) with same extents
    coarse_dims = 8
    coarse_voxel_size = ext / (coarse_dims - 1)
    coarse_tile_points = np.array([[0, 0, 0]], dtype=np.int32)

    coarse_tile_points_wp = wp.array(coarse_tile_points, dtype=wp.vec3i)
    coarse_volume = wp.Volume.allocate_by_tiles(
        tile_points=coarse_tile_points_wp,
        voxel_size=wp.vec3(coarse_voxel_size),
        translation=wp.vec3(min_ext),
        bg_value=SDF_BACKGROUND_VALUE,
    )

    # Populate the coarse volume with SDF values (single tile)
    wp.launch(
        sdf_from_mesh_kernel,
        dim=(1, 8, 8, 8),
        inputs=[m_id, coarse_volume.id, coarse_tile_points_wp, shape_thickness],
    )

    if verbose:
        print(f"Coarse SDF: dims={coarse_dims}x{coarse_dims}x{coarse_dims}, voxel size: {coarse_voxel_size}")

    # Create and populate SDFData struct
    sdf_data = SDFData()
    sdf_data.sparse_sdf_ptr = sparse_volume.id
    sdf_data.sparse_voxel_size = wp.vec3(actual_voxel_size)
    sdf_data.coarse_sdf_ptr = coarse_volume.id
    sdf_data.coarse_voxel_size = wp.vec3(coarse_voxel_size)
    sdf_data.center = wp.vec3(center)
    sdf_data.half_extents = wp.vec3(half_extents)
    sdf_data.background_value = SDF_BACKGROUND_VALUE

    return sdf_data, sparse_volume, coarse_volume
