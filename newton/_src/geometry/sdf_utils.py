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

from ..core.types import MAXVAL, nparray
from ..geometry.kernels import box_sdf, capsule_sdf, cone_sdf, cylinder_sdf, ellipsoid_sdf, sphere_sdf
from .sdf_mc import get_mc_tables, int_to_vec3f, mc_calc_face, vec8f
from .types import GeoType, Mesh


@wp.struct
class SDFData:
    """Encapsulates all data needed for SDF-based collision detection.

    Contains both sparse (narrow band) and coarse (background) SDF volumes
    with the same spatial extents but different resolutions.
    """

    # Sparse (narrow band) SDF - high resolution near surface
    sparse_sdf_ptr: wp.uint64
    sparse_voxel_size: wp.vec3
    sparse_voxel_radius: wp.float32

    # Coarse (background) SDF - 8x8x8 covering entire volume
    coarse_sdf_ptr: wp.uint64
    coarse_voxel_size: wp.vec3

    # Shared extents (same for both volumes)
    center: wp.vec3
    half_extents: wp.vec3

    # Background value used for unallocated voxels in the sparse SDF
    background_value: wp.float32

    # Whether shape_scale was baked into the SDF
    scale_baked: wp.bool


class SDF:
    """Opaque SDF container owning kernel payload and runtime references."""

    def __init__(
        self,
        *,
        data: SDFData,
        sparse_volume: wp.Volume | None = None,
        coarse_volume: wp.Volume | None = None,
        block_coords: nparray | Sequence[wp.vec3us] | None = None,
        _internal: bool = False,
    ):
        if not _internal:
            raise RuntimeError(
                "SDF objects are created via mesh.build_sdf(), SDF.create_from_mesh(), or SDF.create_from_data()."
            )
        self.data = data
        self.sparse_volume = sparse_volume
        self.coarse_volume = coarse_volume
        self.block_coords = block_coords

    def to_kernel_data(self) -> SDFData:
        """Return kernel-facing SDF payload."""
        return self.data

    def is_empty(self) -> bool:
        """Return True when this SDF has no sparse/coarse payload."""
        return int(self.data.sparse_sdf_ptr) == 0 and int(self.data.coarse_sdf_ptr) == 0

    def validate(self) -> None:
        """Validate consistency of kernel pointers and owned volumes."""
        if int(self.data.sparse_sdf_ptr) == 0 and self.sparse_volume is not None:
            raise ValueError("SDFData sparse pointer is empty but sparse_volume is set.")
        if int(self.data.coarse_sdf_ptr) == 0 and self.coarse_volume is not None:
            raise ValueError("SDFData coarse pointer is empty but coarse_volume is set.")

    def __copy__(self) -> "SDF":
        """Return self; SDF runtime handles are immutable and shared."""
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "SDF":
        """Keep deep-copy stable by reusing this instance.

        `wp.Volume` instances inside SDF are ctypes-backed and not picklable.
        Treating SDF as an immutable runtime handle keeps builder deepcopy usable.
        """
        memo[id(self)] = self
        return self

    @staticmethod
    def create_from_points(
        points: nparray | Sequence[Sequence[float]],
        indices: nparray | Sequence[int],
        *,
        narrow_band_range: tuple[float, float] = (-0.1, 0.1),
        target_voxel_size: float | None = None,
        max_resolution: int | None = None,
        margin: float = 0.05,
        thickness: float = 0.0,
        scale: tuple[float, float, float] | None = None,
    ) -> "SDF":
        """Create an SDF from triangle mesh points and indices.

        Args:
            points: Vertex positions [m], shape ``(N, 3)``.
            indices: Triangle vertex indices [index], flattened or shape ``(M, 3)``.
            narrow_band_range: Signed narrow-band distance range [m] as ``(inner, outer)``.
            target_voxel_size: Target sparse-grid voxel size [m]. If provided, takes
                precedence over ``max_resolution``.
            max_resolution: Maximum sparse-grid dimension [voxel]. Used when
                ``target_voxel_size`` is not provided.
            margin: Extra AABB padding [m] added before discretization.
            thickness: Thickness offset [m] to subtract from SDF values.
            scale: Scale factors ``(sx, sy, sz)`` to bake into the SDF.

        Returns:
            A validated :class:`SDF` runtime handle with sparse/coarse volumes.
        """
        mesh = Mesh(points, indices, compute_inertia=False)
        return SDF.create_from_mesh(
            mesh,
            narrow_band_range=narrow_band_range,
            target_voxel_size=target_voxel_size,
            max_resolution=max_resolution,
            margin=margin,
            thickness=thickness,
            scale=scale,
        )

    @staticmethod
    def create_from_mesh(
        mesh: Mesh,
        *,
        narrow_band_range: tuple[float, float] = (-0.1, 0.1),
        target_voxel_size: float | None = None,
        max_resolution: int | None = None,
        margin: float = 0.05,
        thickness: float = 0.0,
        scale: tuple[float, float, float] | None = None,
    ) -> "SDF":
        """Create an SDF from a mesh in local mesh coordinates.

        Args:
            mesh: Source mesh geometry.
            narrow_band_range: Signed narrow-band distance range [m] as
                ``(inner, outer)``.
            target_voxel_size: Target sparse-grid voxel size [m]. If provided,
                takes precedence over ``max_resolution``.
            max_resolution: Maximum sparse-grid dimension [voxel]. Used when
                ``target_voxel_size`` is not provided.
            margin: Extra AABB padding [m] added before discretization.
            thickness: Thickness offset [m] to subtract from SDF values. When
                non-zero, the SDF surface is effectively shrunk inward by this
                amount. Useful for modeling compliant layers in hydroelastic
                collision. Defaults to ``0.0`` (no offset, thickness applied
                at runtime).
            scale: Scale factors ``(sx, sy, sz)`` [unitless] to bake into the
                SDF. When provided, mesh vertices are scaled before SDF
                generation and ``scale_baked`` is set to ``True`` in the
                resulting SDF. Required for hydroelastic collision with
                non-unit shape scale. Defaults to ``None`` (no scale baking;
                scale applied at runtime).

        Returns:
            A validated :class:`SDF` runtime handle with sparse/coarse volumes.
        """
        effective_max_resolution = 64 if max_resolution is None and target_voxel_size is None else max_resolution
        bake_scale = scale is not None
        effective_scale = scale if scale is not None else (1.0, 1.0, 1.0)
        sdf_data, sparse_volume, coarse_volume, block_coords = _compute_sdf_from_shape_impl(
            shape_type=GeoType.MESH,
            shape_geo=mesh,
            shape_scale=effective_scale,
            shape_thickness=thickness,
            narrow_band_distance=narrow_band_range,
            margin=margin,
            target_voxel_size=target_voxel_size,
            max_resolution=effective_max_resolution if effective_max_resolution is not None else 64,
            bake_scale=bake_scale,
        )
        sdf = SDF(
            data=sdf_data,
            sparse_volume=sparse_volume,
            coarse_volume=coarse_volume,
            block_coords=block_coords,
            _internal=True,
        )
        sdf.validate()
        return sdf

    @staticmethod
    def create_from_data(
        *,
        sparse_volume: wp.Volume | None = None,
        coarse_volume: wp.Volume | None = None,
        block_coords: nparray | Sequence[wp.vec3us] | None = None,
        center: Sequence[float] | None = None,
        half_extents: Sequence[float] | None = None,
        background_value: float = MAXVAL,
        scale_baked: bool = False,
    ) -> "SDF":
        """Create an SDF from precomputed runtime resources."""
        sdf_data = create_empty_sdf_data()
        if sparse_volume is not None:
            sdf_data.sparse_sdf_ptr = sparse_volume.id
            sparse_voxel_size = np.asarray(sparse_volume.get_voxel_size(), dtype=np.float32)
            sdf_data.sparse_voxel_size = wp.vec3(sparse_voxel_size)
            sdf_data.sparse_voxel_radius = 0.5 * float(np.linalg.norm(sparse_voxel_size))
        if coarse_volume is not None:
            sdf_data.coarse_sdf_ptr = coarse_volume.id
            coarse_voxel_size = np.asarray(coarse_volume.get_voxel_size(), dtype=np.float32)
            sdf_data.coarse_voxel_size = wp.vec3(coarse_voxel_size)

        sdf_data.center = wp.vec3(center) if center is not None else wp.vec3(0.0, 0.0, 0.0)
        sdf_data.half_extents = wp.vec3(half_extents) if half_extents is not None else wp.vec3(0.0, 0.0, 0.0)
        sdf_data.background_value = background_value
        sdf_data.scale_baked = scale_baked

        sdf = SDF(
            data=sdf_data,
            sparse_volume=sparse_volume,
            coarse_volume=coarse_volume,
            block_coords=block_coords,
            _internal=True,
        )
        sdf.validate()
        return sdf


# Default background value for unallocated voxels in sparse SDF.
# Using MAXVAL ensures trilinear interpolation with unallocated voxels produces values >= MAXVAL * 0.99,
# allowing detection of unallocated voxels without triggering verify_fp false positives.
SDF_BACKGROUND_VALUE = MAXVAL


def create_empty_sdf_data() -> SDFData:
    """Create an empty SDFData struct for shapes that don't need SDF collision.

    Returns:
        An SDFData struct with zeroed pointers and extents.
    """
    sdf_data = SDFData()
    sdf_data.sparse_sdf_ptr = wp.uint64(0)
    sdf_data.sparse_voxel_size = wp.vec3(0.0, 0.0, 0.0)
    sdf_data.sparse_voxel_radius = 0.0
    sdf_data.coarse_sdf_ptr = wp.uint64(0)
    sdf_data.coarse_voxel_size = wp.vec3(0.0, 0.0, 0.0)
    sdf_data.center = wp.vec3(0.0, 0.0, 0.0)
    sdf_data.half_extents = wp.vec3(0.0, 0.0, 0.0)
    sdf_data.background_value = SDF_BACKGROUND_VALUE
    sdf_data.scale_baked = False
    return sdf_data


@wp.kernel
def compute_mesh_signed_volume_kernel(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    volume_sum: wp.array(dtype=wp.float32),
):
    """Compute signed volume contribution from each triangle."""
    tri_idx = wp.tid()
    v0 = points[indices[tri_idx * 3 + 0]]
    v1 = points[indices[tri_idx * 3 + 1]]
    v2 = points[indices[tri_idx * 3 + 2]]
    wp.atomic_add(volume_sum, 0, wp.dot(v0, wp.cross(v1, v2)) / 6.0)


def compute_mesh_signed_volume(points: wp.array, indices: wp.array) -> float:
    """Compute signed volume of a mesh on GPU. Positive = correct winding, negative = inverted."""
    num_tris = indices.shape[0] // 3
    volume_sum = wp.zeros(1, dtype=wp.float32)
    wp.launch(compute_mesh_signed_volume_kernel, dim=num_tris, inputs=[points, indices, volume_sum])
    return float(volume_sum.numpy()[0])


@wp.func
def get_distance_to_mesh(mesh: wp.uint64, point: wp.vec3, max_dist: wp.float32, winding_threshold: wp.float32):
    res = wp.mesh_query_point_sign_winding_number(mesh, point, max_dist, 2.0, winding_threshold)
    if res.result:
        closest = wp.mesh_eval_position(mesh, res.face, res.u, res.v)
        vec_to_surface = closest - point
        sign = res.sign
        # For inverted meshes (threshold < 0), the winding > threshold comparison
        # gives inverted signs, so we flip them back
        if winding_threshold < 0.0:
            sign = -sign
        return sign * wp.length(vec_to_surface)
    return max_dist


@wp.kernel
def sdf_from_mesh_kernel(
    mesh: wp.uint64,
    sdf: wp.uint64,
    tile_points: wp.array(dtype=wp.vec3i),
    thickness: wp.float32,
    winding_threshold: wp.float32,
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
    signed_distance = get_distance_to_mesh(mesh, sample_pos, 10000.0, winding_threshold)
    signed_distance -= thickness
    wp.volume_store(sdf, x_id, y_id, z_id, signed_distance)


@wp.kernel(enable_backward=False)
def sdf_from_primitive_kernel(
    shape_type: wp.int32,
    shape_scale: wp.vec3,
    sdf: wp.uint64,
    tile_points: wp.array(dtype=wp.vec3i),
    thickness: wp.float32,
):
    """
    Populate SDF grid from primitive shape.
    Only processes specified tiles. Launch with dim=(num_tiles, 8, 8, 8).
    """
    tile_idx, local_x, local_y, local_z = wp.tid()

    tile_origin = tile_points[tile_idx]
    x_id = tile_origin[0] + local_x
    y_id = tile_origin[1] + local_y
    z_id = tile_origin[2] + local_z

    sample_pos = wp.volume_index_to_world(sdf, int_to_vec3f(x_id, y_id, z_id))
    signed_distance = float(1.0e6)
    if shape_type == GeoType.SPHERE:
        signed_distance = sphere_sdf(wp.vec3(0.0, 0.0, 0.0), shape_scale[0], sample_pos)
    elif shape_type == GeoType.BOX:
        signed_distance = box_sdf(shape_scale, sample_pos)
    elif shape_type == GeoType.CAPSULE:
        signed_distance = capsule_sdf(shape_scale[0], shape_scale[1], sample_pos)
    elif shape_type == GeoType.CYLINDER:
        signed_distance = cylinder_sdf(shape_scale[0], shape_scale[1], sample_pos)
    elif shape_type == GeoType.ELLIPSOID:
        signed_distance = ellipsoid_sdf(shape_scale, sample_pos)
    elif shape_type == GeoType.CONE:
        signed_distance = cone_sdf(shape_scale[0], shape_scale[1], sample_pos)
    signed_distance -= thickness
    wp.volume_store(sdf, x_id, y_id, z_id, signed_distance)


@wp.kernel
def check_tile_occupied_mesh_kernel(
    mesh: wp.uint64,
    tile_points: wp.array(dtype=wp.vec3f),
    threshold: wp.vec2f,
    winding_threshold: wp.float32,
    tile_occupied: wp.array(dtype=bool),
):
    tid = wp.tid()
    sample_pos = tile_points[tid]

    signed_distance = get_distance_to_mesh(mesh, sample_pos, 10000.0, winding_threshold)
    is_occupied = wp.bool(False)
    if wp.sign(signed_distance) > 0.0:
        is_occupied = signed_distance < threshold[1]
    else:
        is_occupied = signed_distance > threshold[0]
    tile_occupied[tid] = is_occupied


@wp.kernel(enable_backward=False)
def check_tile_occupied_primitive_kernel(
    shape_type: wp.int32,
    shape_scale: wp.vec3,
    tile_points: wp.array(dtype=wp.vec3f),
    threshold: wp.vec2f,
    tile_occupied: wp.array(dtype=bool),
):
    tid = wp.tid()
    sample_pos = tile_points[tid]

    signed_distance = float(1.0e6)
    if shape_type == GeoType.SPHERE:
        signed_distance = sphere_sdf(wp.vec3(0.0, 0.0, 0.0), shape_scale[0], sample_pos)
    elif shape_type == GeoType.BOX:
        signed_distance = box_sdf(shape_scale, sample_pos)
    elif shape_type == GeoType.CAPSULE:
        signed_distance = capsule_sdf(shape_scale[0], shape_scale[1], sample_pos)
    elif shape_type == GeoType.CYLINDER:
        signed_distance = cylinder_sdf(shape_scale[0], shape_scale[1], sample_pos)
    elif shape_type == GeoType.ELLIPSOID:
        signed_distance = ellipsoid_sdf(shape_scale, sample_pos)
    elif shape_type == GeoType.CONE:
        signed_distance = cone_sdf(shape_scale[0], shape_scale[1], sample_pos)

    is_occupied = wp.bool(False)
    if wp.sign(signed_distance) > 0.0:
        is_occupied = signed_distance < threshold[1]
    else:
        is_occupied = signed_distance > threshold[0]
    tile_occupied[tid] = is_occupied


def get_primitive_extents(shape_type: int, shape_scale: Sequence[float]) -> tuple[list[float], list[float]]:
    """Get the bounding box extents for a primitive shape.

    Args:
        shape_type: Type of the primitive shape (from GeoType).
        shape_scale: Scale factors for the shape.

    Returns:
        Tuple of (min_ext, max_ext) as lists of [x, y, z] coordinates.

    Raises:
        NotImplementedError: If shape_type is not a supported primitive.
    """
    if shape_type == GeoType.SPHERE:
        min_ext = [-shape_scale[0], -shape_scale[0], -shape_scale[0]]
        max_ext = [shape_scale[0], shape_scale[0], shape_scale[0]]
    elif shape_type == GeoType.BOX:
        min_ext = [-shape_scale[0], -shape_scale[1], -shape_scale[2]]
        max_ext = [shape_scale[0], shape_scale[1], shape_scale[2]]
    elif shape_type == GeoType.CAPSULE:
        min_ext = [-shape_scale[0], -shape_scale[0], -shape_scale[1] - shape_scale[0]]
        max_ext = [shape_scale[0], shape_scale[0], shape_scale[1] + shape_scale[0]]
    elif shape_type == GeoType.CYLINDER:
        min_ext = [-shape_scale[0], -shape_scale[0], -shape_scale[1]]
        max_ext = [shape_scale[0], shape_scale[0], shape_scale[1]]
    elif shape_type == GeoType.ELLIPSOID:
        min_ext = [-shape_scale[0], -shape_scale[1], -shape_scale[2]]
        max_ext = [shape_scale[0], shape_scale[1], shape_scale[2]]
    elif shape_type == GeoType.CONE:
        min_ext = [-shape_scale[0], -shape_scale[0], -shape_scale[1]]
        max_ext = [shape_scale[0], shape_scale[0], shape_scale[1]]
    else:
        raise NotImplementedError(f"Extents not implemented for shape type: {shape_type}")
    return min_ext, max_ext


def _compute_sdf_from_shape_impl(
    shape_type: int,
    shape_geo: Mesh | None = None,
    shape_scale: Sequence[float] = (1.0, 1.0, 1.0),
    shape_thickness: float = 0.0,
    narrow_band_distance: Sequence[float] = (-0.1, 0.1),
    margin: float = 0.05,
    target_voxel_size: float | None = None,
    max_resolution: int = 64,
    bake_scale: bool = False,
    verbose: bool = False,
) -> tuple[SDFData, wp.Volume | None, wp.Volume | None, Sequence[wp.vec3us]]:
    """Compute sparse and coarse SDF volumes for a shape.

    The SDF is computed in the mesh's unscaled local space. Scale is intentionally
    NOT a parameter - the collision system handles scaling at runtime, ensuring
    the SDF and mesh BVH stay consistent and allowing dynamic scale changes.

    Args:
        shape_type: Type of the shape.
        shape_geo: Optional source geometry. Required for mesh shapes.
        shape_scale: Scale factors for the mesh. Applied before SDF generation if bake_scale is True.
        shape_thickness: Thickness offset to subtract from SDF values.
        narrow_band_distance: Tuple of (inner, outer) distances for narrow band.
        margin: Margin to add to bounding box. Must be > 0.
        target_voxel_size: Target voxel size for sparse SDF grid. If None, computed as max_extent/max_resolution.
        max_resolution: Maximum dimension for sparse SDF grid when target_voxel_size is None. Must be divisible by 8.
        bake_scale: If True, bake shape_scale into the SDF. If False, use (1,1,1) scale.
        verbose: Print debug info.

    Returns:
        Tuple of (sdf_data, sparse_volume, coarse_volume, block_coords) where:
        - sdf_data: SDFData struct with pointers and extents
        - sparse_volume: wp.Volume object for sparse SDF (keep alive for reference counting)
        - coarse_volume: wp.Volume object for coarse SDF (keep alive for reference counting)
        - block_coords: List of wp.vec3us tile coordinates for allocated blocks in the sparse volume

    Raises:
        RuntimeError: If CUDA is not available.
    """
    if not wp.is_cuda_available():
        raise RuntimeError("compute_sdf_from_shape requires CUDA but no CUDA device is available")

    if shape_type == GeoType.PLANE or shape_type == GeoType.HFIELD:
        # SDF collisions are not supported for Plane or HField shapes, falling back to mesh collisions
        return create_empty_sdf_data(), None, None, []

    assert isinstance(narrow_band_distance, Sequence), "narrow_band_distance must be a tuple of two floats"
    assert len(narrow_band_distance) == 2, "narrow_band_distance must be a tuple of two floats"
    assert narrow_band_distance[0] < 0.0 < narrow_band_distance[1], (
        "narrow_band_distance[0] must be less than 0.0 and narrow_band_distance[1] must be greater than 0.0"
    )
    assert margin > 0, "margin must be > 0"

    # Determine effective scale based on bake_scale flag
    effective_scale = tuple(shape_scale) if bake_scale else (1.0, 1.0, 1.0)

    offset = margin + shape_thickness

    if shape_type == GeoType.MESH:
        if shape_geo is None:
            raise ValueError("shape_geo must be provided for GeoType.MESH.")
        verts = shape_geo.vertices * np.array(effective_scale)[None, :]
        pos = wp.array(verts, dtype=wp.vec3)
        indices = wp.array(shape_geo.indices, dtype=wp.int32)

        mesh = wp.Mesh(points=pos, indices=indices, support_winding_number=True)
        m_id = mesh.id

        # Compute winding threshold based on mesh volume sign
        # Positive volume = correct winding (threshold 0.5), negative = inverted (threshold -0.5)
        signed_volume = compute_mesh_signed_volume(pos, indices)
        winding_threshold = 0.5 if signed_volume >= 0.0 else -0.5
        if verbose and signed_volume < 0:
            print("Mesh has inverted winding (negative volume), using threshold -0.5")

        min_ext = np.min(verts, axis=0).tolist()
        max_ext = np.max(verts, axis=0).tolist()
    else:
        min_ext, max_ext = get_primitive_extents(shape_type, effective_scale)

    min_ext = np.array(min_ext) - offset
    max_ext = np.array(max_ext) + offset
    ext = max_ext - min_ext

    # Compute center and half_extents for oriented bounding box collision detection
    center = (min_ext + max_ext) * 0.5
    half_extents = (max_ext - min_ext) * 0.5

    # Calculate uniform voxel size based on the longest dimension
    max_extent = np.max(ext)
    # If target_voxel_size not specified, compute from max_resolution
    if target_voxel_size is None:
        # Warp volumes are allocated in tiles of 8 voxels
        assert max_resolution % 8 == 0, "max_resolution must be divisible by 8 for SDF volume allocation"
        # we store coords as uint16
        assert max_resolution < 1 << 16, f"max_resolution must be less than {1 << 16}"
        target_voxel_size = max_extent / max_resolution
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

    if shape_type == GeoType.MESH:
        wp.launch(
            check_tile_occupied_mesh_kernel,
            dim=(len(tile_points)),
            inputs=[m_id, tile_center_points_world, threshold, winding_threshold],
            outputs=[tile_occupied],
        )
    else:
        wp.launch(
            check_tile_occupied_primitive_kernel,
            dim=(len(tile_points)),
            inputs=[shape_type, effective_scale, tile_center_points_world, threshold],
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
    if shape_type == GeoType.MESH:
        wp.launch(
            sdf_from_mesh_kernel,
            dim=(num_allocated_tiles, 8, 8, 8),
            inputs=[m_id, sparse_volume.id, tile_points_wp, shape_thickness, winding_threshold],
        )
    else:
        wp.launch(
            sdf_from_primitive_kernel,
            dim=(num_allocated_tiles, 8, 8, 8),
            inputs=[shape_type, effective_scale, sparse_volume.id, tile_points_wp, shape_thickness],
        )

    tiles = sparse_volume.get_tiles().numpy()
    block_coords = [wp.vec3us(t_coords) for t_coords in tiles]

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
    if shape_type == GeoType.MESH:
        wp.launch(
            sdf_from_mesh_kernel,
            dim=(1, 8, 8, 8),
            inputs=[m_id, coarse_volume.id, coarse_tile_points_wp, shape_thickness, winding_threshold],
        )
    else:
        wp.launch(
            sdf_from_primitive_kernel,
            dim=(1, 8, 8, 8),
            inputs=[shape_type, effective_scale, coarse_volume.id, coarse_tile_points_wp, shape_thickness],
        )

    if verbose:
        print(f"Coarse SDF: dims={coarse_dims}x{coarse_dims}x{coarse_dims}, voxel size: {coarse_voxel_size}")

    # Create and populate SDFData struct
    sdf_data = SDFData()
    sdf_data.sparse_sdf_ptr = sparse_volume.id
    sdf_data.sparse_voxel_size = wp.vec3(actual_voxel_size)
    sdf_data.sparse_voxel_radius = 0.5 * float(np.linalg.norm(actual_voxel_size))
    sdf_data.coarse_sdf_ptr = coarse_volume.id
    sdf_data.coarse_voxel_size = wp.vec3(coarse_voxel_size)
    sdf_data.center = wp.vec3(center)
    sdf_data.half_extents = wp.vec3(half_extents)
    sdf_data.background_value = SDF_BACKGROUND_VALUE
    sdf_data.scale_baked = bake_scale

    return sdf_data, sparse_volume, coarse_volume, block_coords


def compute_sdf_from_shape(
    shape_type: int,
    shape_geo: Mesh | None = None,
    shape_scale: Sequence[float] = (1.0, 1.0, 1.0),
    shape_thickness: float = 0.0,
    narrow_band_distance: Sequence[float] = (-0.1, 0.1),
    margin: float = 0.05,
    target_voxel_size: float | None = None,
    max_resolution: int = 64,
    bake_scale: bool = False,
    verbose: bool = False,
) -> tuple[SDFData, wp.Volume | None, wp.Volume | None, Sequence[wp.vec3us]]:
    """Compute sparse and coarse SDF volumes for a shape.

    Mesh shape dispatches through :meth:`SDF.create_from_mesh` to keep that path canonical.

    Args:
        shape_type: Geometry type identifier from :class:`GeoType`.
        shape_geo: Source mesh geometry when ``shape_type`` is ``GeoType.MESH``.
        shape_scale: Shape scale [unitless].
        shape_thickness: Shape thickness offset [m] subtracted from sampled SDF.
        narrow_band_distance: Signed narrow-band distance range [m] as ``(inner, outer)``.
        margin: Extra AABB padding [m] added before discretization.
        target_voxel_size: Target sparse-grid voxel size [m]. If provided, takes
            precedence over ``max_resolution``.
        max_resolution: Maximum sparse-grid dimension [voxel] when
            ``target_voxel_size`` is not provided.
        bake_scale: If ``True``, bake ``shape_scale`` into generated SDF data.
        verbose: If ``True``, print debug information during SDF construction.

    Returns:
        Tuple ``(sdf_data, sparse_volume, coarse_volume, block_coords)``.
    """
    if shape_type == GeoType.MESH:
        if shape_geo is None:
            raise ValueError("shape_geo must be provided for GeoType.MESH.")
        # Canonical mesh path: use SDF.create_from_mesh for all mesh SDF generation.
        sdf = SDF.create_from_mesh(
            shape_geo,
            narrow_band_range=tuple(narrow_band_distance),
            target_voxel_size=target_voxel_size,
            max_resolution=max_resolution,
            margin=margin,
            thickness=shape_thickness,
            scale=tuple(shape_scale) if bake_scale else None,
        )
        return sdf.to_kernel_data(), sdf.sparse_volume, sdf.coarse_volume, (sdf.block_coords or [])

    return _compute_sdf_from_shape_impl(
        shape_type=shape_type,
        shape_geo=shape_geo,
        shape_scale=shape_scale,
        shape_thickness=shape_thickness,
        narrow_band_distance=narrow_band_distance,
        margin=margin,
        target_voxel_size=target_voxel_size,
        max_resolution=max_resolution,
        bake_scale=bake_scale,
        verbose=verbose,
    )


def compute_isomesh(volume: wp.Volume) -> Mesh | None:
    """Compute an isosurface mesh from an SDFData struct.

    Uses a two-pass approach to minimize memory allocation:
    1. First pass: count actual triangles produced
    2. Allocate exact memory needed
    3. Second pass: generate vertices

    Args:
        volume: The SDF volume.

    Returns:
        Mesh object containing the isosurface mesh.
    """
    device = wp.get_device()
    mc_tables = get_mc_tables(device)

    # Get allocated tile points from the sparse volume
    tile_points = volume.get_tiles()
    tile_points_wp = wp.array(tile_points, dtype=wp.vec3i, device=device)
    num_tiles = tile_points.shape[0]

    if num_tiles == 0:
        return None

    # Pass 1: Count faces (no vertex allocation needed)
    face_count = wp.zeros((1,), dtype=int, device=device)
    wp.launch(
        count_isomesh_faces_kernel,
        dim=(num_tiles, 8, 8, 8),
        inputs=[volume.id, tile_points_wp, mc_tables[0], mc_tables[3]],
        outputs=[face_count],
        device=device,
    )

    num_faces = int(face_count.numpy()[0])
    if num_faces == 0:
        return None

    # Allocate exact memory needed (not worst-case 5*voxels)
    max_verts = 3 * num_faces
    verts = wp.empty((max_verts,), dtype=wp.vec3, device=device)
    face_normals = wp.empty((num_faces,), dtype=wp.vec3, device=device)

    # Pass 2: Generate vertices with exact allocation
    face_count.zero_()
    wp.launch(
        generate_isomesh_kernel,
        dim=(num_tiles, 8, 8, 8),
        inputs=[volume.id, tile_points_wp, mc_tables[0], mc_tables[4], mc_tables[3]],
        outputs=[face_count, verts, face_normals],
        device=device,
    )

    verts_np = verts.numpy()
    faces_np = np.arange(3 * num_faces).reshape(-1, 3)

    # reverse order of triangles indices for correctly displayed normals
    faces_np = faces_np[:, ::-1]
    return Mesh(verts_np, faces_np)


@wp.kernel(enable_backward=False)
def count_isomesh_faces_kernel(
    sdf: wp.uint64,
    tile_points: wp.array(dtype=wp.vec3i),
    tri_range_table: wp.array(dtype=wp.int32),
    corner_offsets_table: wp.array(dtype=wp.vec3ub),
    face_count: wp.array(dtype=int),
):
    """Count isosurface faces without generating vertices (first pass of two-pass approach).
    Only processes specified tiles. Launch with dim=(num_tiles, 8, 8, 8).
    """
    tile_idx, local_x, local_y, local_z = wp.tid()

    # Get the tile origin and compute global voxel coordinates
    tile_origin = tile_points[tile_idx]
    x_id = tile_origin[0] + local_x
    y_id = tile_origin[1] + local_y
    z_id = tile_origin[2] + local_z

    isovalue = 0.0
    cube_idx = wp.int32(0)
    for i in range(8):
        corner_offset = wp.vec3i(corner_offsets_table[i])
        x = x_id + corner_offset.x
        y = y_id + corner_offset.y
        z = z_id + corner_offset.z
        v = wp.volume_lookup_f(sdf, x, y, z)
        if v >= wp.static(MAXVAL * 0.99):
            return
        if v < isovalue:
            cube_idx |= 1 << i

    # look up the tri range for the cube index
    tri_range_start = tri_range_table[cube_idx]
    tri_range_end = tri_range_table[cube_idx + 1]
    num_verts = tri_range_end - tri_range_start

    num_faces = num_verts // 3
    if num_faces > 0:
        wp.atomic_add(face_count, 0, num_faces)


@wp.kernel(enable_backward=False)
def generate_isomesh_kernel(
    sdf: wp.uint64,
    tile_points: wp.array(dtype=wp.vec3i),
    tri_range_table: wp.array(dtype=wp.int32),
    flat_edge_verts_table: wp.array(dtype=wp.vec2ub),
    corner_offsets_table: wp.array(dtype=wp.vec3ub),
    face_count: wp.array(dtype=int),
    vertices: wp.array(dtype=wp.vec3),
    face_normals: wp.array(dtype=wp.vec3),
):
    """Generate isosurface mesh vertices and normals using marching cubes.
    Only processes specified tiles. Launch with dim=(num_tiles, 8, 8, 8).
    """
    tile_idx, local_x, local_y, local_z = wp.tid()

    # Get the tile origin and compute global voxel coordinates
    tile_origin = tile_points[tile_idx]
    x_id = tile_origin[0] + local_x
    y_id = tile_origin[1] + local_y
    z_id = tile_origin[2] + local_z

    isovalue = 0.0
    cube_idx = wp.int32(0)
    corner_vals = vec8f()
    for i in range(8):
        corner_offset = wp.vec3i(corner_offsets_table[i])
        x = x_id + corner_offset.x
        y = y_id + corner_offset.y
        z = z_id + corner_offset.z
        v = wp.volume_lookup_f(sdf, x, y, z)
        if v >= wp.static(MAXVAL * 0.99):
            return
        corner_vals[i] = v

        if v < isovalue:
            cube_idx |= 1 << i

    # look up the tri range for the cube index
    tri_range_start = tri_range_table[cube_idx]
    tri_range_end = tri_range_table[cube_idx + 1]
    num_verts = tri_range_end - tri_range_start  # number of intersected edges

    num_faces = num_verts // 3
    out_idx_faces = wp.atomic_add(face_count, 0, num_faces)

    if num_verts == 0:
        return

    for fi in range(5):
        if fi >= num_faces:
            return
        _area, normal, _face_center, _pen_depth, face_verts = mc_calc_face(
            flat_edge_verts_table,
            corner_offsets_table,
            tri_range_start + 3 * fi,
            corner_vals,
            sdf,
            x_id,
            y_id,
            z_id,
        )
        vertices[3 * out_idx_faces + 3 * fi + 0] = wp.vec3(face_verts[0])
        vertices[3 * out_idx_faces + 3 * fi + 1] = wp.vec3(face_verts[1])
        vertices[3 * out_idx_faces + 3 * fi + 2] = wp.vec3(face_verts[2])
        face_normals[out_idx_faces + fi] = normal
