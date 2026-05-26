# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import warp as wp

from ..core.types import MAXVAL, Axis, Devicelike
from .kernels import sdf_box, sdf_capsule, sdf_cone, sdf_cylinder, sdf_ellipsoid, sdf_sphere
from .sdf_mc import (
    MC_DEGENERATE_N_SQ_EPS,
    MC_EDGE_CLAMP_MAX,
    MC_EDGE_CLAMP_MIN,
    MC_EDGE_VAL_DIFF_EPS,
    get_mc_tables,
    int_to_vec3f,
    mc_calc_face,
    vec8f,
)
from .types import GeoType, Mesh

logger = logging.getLogger(__name__)

SignMethod = Literal["auto", "parity", "winding"]

if TYPE_CHECKING:
    from .sdf_texture import TextureSDFData


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


@wp.func
def sample_sdf_extrapolated(
    sdf_data: SDFData,
    sdf_pos: wp.vec3,
) -> float:
    """Sample NanoVDB SDF with extrapolation for points outside the narrow band or extent.

    Handles three cases:

    1. Point in narrow band: returns sparse grid value directly.
    2. Point inside extent but outside narrow band: returns coarse grid value.
    3. Point outside extent: projects to boundary, returns value at boundary + distance to boundary.

    Args:
        sdf_data: SDFData struct containing sparse/coarse volumes and extent info.
        sdf_pos: Query position in the SDF's local coordinate space [m].

    Returns:
        The signed distance value [m], extrapolated if necessary.
    """
    lower = sdf_data.center - sdf_data.half_extents
    upper = sdf_data.center + sdf_data.half_extents

    inside_extent = (
        sdf_pos[0] >= lower[0]
        and sdf_pos[0] <= upper[0]
        and sdf_pos[1] >= lower[1]
        and sdf_pos[1] <= upper[1]
        and sdf_pos[2] >= lower[2]
        and sdf_pos[2] <= upper[2]
    )

    if inside_extent:
        sparse_idx = wp.volume_world_to_index(sdf_data.sparse_sdf_ptr, sdf_pos)
        sparse_dist = wp.volume_sample_f(sdf_data.sparse_sdf_ptr, sparse_idx, wp.Volume.LINEAR)

        if sparse_dist >= sdf_data.background_value * 0.99 or wp.isnan(sparse_dist):
            coarse_idx = wp.volume_world_to_index(sdf_data.coarse_sdf_ptr, sdf_pos)
            return wp.volume_sample_f(sdf_data.coarse_sdf_ptr, coarse_idx, wp.Volume.LINEAR)
        else:
            return sparse_dist
    else:
        eps = 1e-2 * sdf_data.sparse_voxel_size
        clamped_pos = wp.min(wp.max(sdf_pos, lower + eps), upper - eps)
        dist_to_boundary = wp.length(sdf_pos - clamped_pos)

        coarse_idx = wp.volume_world_to_index(sdf_data.coarse_sdf_ptr, clamped_pos)
        boundary_dist = wp.volume_sample_f(sdf_data.coarse_sdf_ptr, coarse_idx, wp.Volume.LINEAR)

        return boundary_dist + dist_to_boundary


@wp.func
def sample_sdf_grad_extrapolated(
    sdf_data: SDFData,
    sdf_pos: wp.vec3,
) -> tuple[float, wp.vec3]:
    """Sample NanoVDB SDF with gradient, with extrapolation for points outside narrow band or extent.

    Handles three cases:

    1. Point in narrow band: returns sparse grid value and gradient directly.
    2. Point inside extent but outside narrow band: returns coarse grid value and gradient.
    3. Point outside extent: returns extrapolated distance and direction toward boundary.

    Args:
        sdf_data: SDFData struct containing sparse/coarse volumes and extent info.
        sdf_pos: Query position in the SDF's local coordinate space [m].

    Returns:
        Tuple of (distance [m], gradient [unitless]) where gradient points toward increasing distance.
    """
    lower = sdf_data.center - sdf_data.half_extents
    upper = sdf_data.center + sdf_data.half_extents

    gradient = wp.vec3(0.0, 0.0, 0.0)

    inside_extent = (
        sdf_pos[0] >= lower[0]
        and sdf_pos[0] <= upper[0]
        and sdf_pos[1] >= lower[1]
        and sdf_pos[1] <= upper[1]
        and sdf_pos[2] >= lower[2]
        and sdf_pos[2] <= upper[2]
    )

    if inside_extent:
        sparse_idx = wp.volume_world_to_index(sdf_data.sparse_sdf_ptr, sdf_pos)
        sparse_dist = wp.volume_sample_grad_f(sdf_data.sparse_sdf_ptr, sparse_idx, wp.Volume.LINEAR, gradient)

        if sparse_dist >= sdf_data.background_value * 0.99 or wp.isnan(sparse_dist):
            coarse_idx = wp.volume_world_to_index(sdf_data.coarse_sdf_ptr, sdf_pos)
            coarse_dist = wp.volume_sample_grad_f(sdf_data.coarse_sdf_ptr, coarse_idx, wp.Volume.LINEAR, gradient)
            return coarse_dist, gradient
        else:
            return sparse_dist, gradient
    else:
        eps = 1e-2 * sdf_data.sparse_voxel_size
        clamped_pos = wp.min(wp.max(sdf_pos, lower + eps), upper - eps)
        diff = sdf_pos - clamped_pos
        dist_to_boundary = wp.length(diff)

        coarse_idx = wp.volume_world_to_index(sdf_data.coarse_sdf_ptr, clamped_pos)
        boundary_dist = wp.volume_sample_f(sdf_data.coarse_sdf_ptr, coarse_idx, wp.Volume.LINEAR)

        extrapolated_dist = boundary_dist + dist_to_boundary

        if dist_to_boundary > 0.0:
            gradient = diff / dist_to_boundary
        else:
            wp.volume_sample_grad_f(sdf_data.coarse_sdf_ptr, coarse_idx, wp.Volume.LINEAR, gradient)

        return extrapolated_dist, gradient


class SDF:
    """Opaque SDF container owning kernel payload and runtime references."""

    def __init__(
        self,
        *,
        data: SDFData,
        sparse_volume: wp.Volume | None = None,
        coarse_volume: wp.Volume | None = None,
        block_coords: np.ndarray | Sequence[wp.vec3us] | None = None,
        texture_block_coords: Sequence[wp.vec3us] | None = None,
        texture_data: "TextureSDFData | None" = None,
        shape_margin: float = 0.0,
        _coarse_texture: wp.Texture3D | None = None,
        _subgrid_texture: wp.Texture3D | None = None,
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
        self.texture_block_coords = texture_block_coords
        self.texture_data = texture_data
        self.shape_margin = shape_margin
        # Keep texture references alive to prevent GC
        self._coarse_texture = _coarse_texture
        self._subgrid_texture = _subgrid_texture

    def to_kernel_data(self) -> SDFData:
        """Return kernel-facing SDF payload."""
        return self.data

    def to_texture_kernel_data(self) -> "TextureSDFData | None":
        """Return texture SDF kernel payload, or ``None`` if unavailable."""
        return self.texture_data

    def is_empty(self) -> bool:
        """Return True when this SDF has no sparse/coarse or texture payload."""
        return int(self.data.sparse_sdf_ptr) == 0 and int(self.data.coarse_sdf_ptr) == 0 and self.texture_data is None

    def validate(self) -> None:
        """Validate consistency of kernel pointers and owned volumes."""
        if int(self.data.sparse_sdf_ptr) == 0 and self.sparse_volume is not None:
            raise ValueError("SDFData sparse pointer is empty but sparse_volume is set.")
        if int(self.data.coarse_sdf_ptr) == 0 and self.coarse_volume is not None:
            raise ValueError("SDFData coarse pointer is empty but coarse_volume is set.")

    def extract_isomesh(self, isovalue: float = 0.0, device: "Devicelike | None" = None) -> "Mesh | None":
        """Extract an isosurface mesh at the requested isovalue.

        Uses the texture SDF path for mesh-generated SDFs (the only path
        populated by :meth:`create_from_mesh`).  For primitive SDFs built
        via :meth:`create_from_data` with a NanoVDB ``sparse_volume``, the
        fallback branch extracts from the sparse volume instead.

        The ``isovalue`` argument is always interpreted in raw mesh-distance
        units: ``0.0`` yields the base geometry surface regardless of how
        the SDF was constructed, and positive values give an outward
        offset.  Both storage backends are normalized to this convention:

        * Texture SDFs store unmodified signed distance ``d`` (the texture
          builder does not bake ``shape_margin``), so the requested
          isovalue is forwarded as-is to the isomesh extractor.
        * NanoVDB sparse volumes built via :class:`_compute_sdf_from_shape_impl`
          store ``d - shape_margin``.  The extractor compensates with
          ``corrected_isovalue = isovalue - shape_margin`` so external
          behavior matches the texture path.

        As a consequence, :attr:`shape_margin` is only consulted on the
        legacy sparse-volume branch; on the texture branch it is stored
        for backward compatibility with callers that introspect the SDF
        but has no effect on the extracted surface.

        Args:
            isovalue: Surface level to extract [m] in base geometry
                distance units.  ``0.0`` gives the original surface;
                positive values give an outward offset.
            device: CUDA device.  When ``None`` uses the current device.

        Returns:
            :class:`Mesh` or ``None`` when the SDF has no data or the
            isovalue falls outside the stored narrow band.
        """
        if self.texture_data is not None and self._coarse_texture is not None:
            from .sdf_texture import TextureSDFData, compute_isomesh_from_texture_sdf  # noqa: PLC0415

            with wp.ScopedDevice(device):
                tex_arr = wp.array([self.texture_data], dtype=TextureSDFData, device=device)
                ct = self._coarse_texture
                coarse_dims = (ct.width - 1, ct.height - 1, ct.depth - 1)
                slots = self.texture_data.subgrid_start_slots
                # Texture SDF stores raw mesh distance (shape_margin is not
                # baked in), so forward isovalue directly.  See class docstring.
                result = compute_isomesh_from_texture_sdf(
                    tex_arr, 0, slots, coarse_dims, device=device, isovalue=isovalue
                )
                if result is not None:
                    return result

        if self.sparse_volume is not None:
            # Legacy NanoVDB sparse volumes (produced by
            # _compute_sdf_from_shape_impl) store d - shape_margin, so
            # compensate to match the texture branch semantics above.
            corrected_isovalue = isovalue - self.shape_margin if self.shape_margin else isovalue
            return compute_isomesh(self.sparse_volume, isovalue=corrected_isovalue, device=device)

        return None

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
        points: np.ndarray | Sequence[Sequence[float]],
        indices: np.ndarray | Sequence[int],
        *,
        device: Devicelike | None = None,
        narrow_band_range: tuple[float, float] = (-0.1, 0.1),
        target_voxel_size: float | None = None,
        max_resolution: int | None = None,
        margin: float = 0.05,
        shape_margin: float = 0.0,
        scale: tuple[float, float, float] | None = None,
    ) -> "SDF":
        """Create an SDF from triangle mesh points and indices.

        Args:
            points: Vertex positions [m], shape ``(N, 3)``.
            indices: Triangle vertex indices [index], flattened or shape ``(M, 3)``.
            device: CUDA device for SDF allocation. When ``None``, uses the
                current :class:`wp.ScopedDevice` or the Warp default device.
            narrow_band_range: Signed narrow-band distance range [m] as ``(inner, outer)``.
            target_voxel_size: Target sparse-grid voxel size [m]. If provided, takes
                precedence over ``max_resolution``.
            max_resolution: Maximum sparse-grid dimension [voxel]. Used when
                ``target_voxel_size`` is not provided.
            margin: Extra AABB padding [m] added before discretization.
            shape_margin: Shape margin offset [m] to subtract from SDF values.
            scale: Scale factors ``(sx, sy, sz)`` to bake into the SDF.

        Returns:
            A validated :class:`SDF` runtime handle with sparse/coarse volumes.
        """
        mesh = Mesh(points, indices, compute_inertia=False)
        return SDF.create_from_mesh(
            mesh,
            device=device,
            narrow_band_range=narrow_band_range,
            target_voxel_size=target_voxel_size,
            max_resolution=max_resolution,
            margin=margin,
            shape_margin=shape_margin,
            scale=scale,
        )

    @staticmethod
    def create_from_mesh(
        mesh: Mesh,
        *,
        device: Devicelike | None = None,
        narrow_band_range: tuple[float, float] = (-0.1, 0.1),
        target_voxel_size: float | None = None,
        max_resolution: int | None = None,
        margin: float = 0.05,
        shape_margin: float = 0.0,
        scale: tuple[float, float, float] | None = None,
        texture_format: str = "uint16",
        sign_method: SignMethod = "auto",
        cache_dir: str | os.PathLike[str] | None = None,
    ) -> "SDF":
        """Create an SDF from a mesh in local mesh coordinates.

        The SDF is built entirely via the texture-based sparse construction
        path.  NanoVDB volumes are **not** created; all downstream collision
        and simulation code uses the texture SDF.

        Args:
            mesh: Source mesh geometry.
            device: CUDA device for SDF allocation. When ``None``, uses the
                current :class:`wp.ScopedDevice` or the Warp default device.
            narrow_band_range: Signed narrow-band distance range [m] as
                ``(inner, outer)``.
            target_voxel_size: Target sparse-grid voxel size [m]. If provided,
                takes precedence over ``max_resolution``.
            max_resolution: Maximum sparse-grid dimension [voxel]. Used when
                ``target_voxel_size`` is not provided.
            margin: Extra AABB padding [m] added before discretization.
            shape_margin: Shape margin offset [m] to subtract from SDF values.
                When non-zero, the SDF surface is effectively shrunk inward by
                this amount. Useful for modeling compliant layers in hydroelastic
                collision. Defaults to ``0.0``.
            scale: Scale factors ``(sx, sy, sz)`` [unitless] to bake into the
                SDF. When provided, mesh vertices are scaled before SDF
                generation and ``scale_baked`` is set to ``True`` in the
                resulting SDF. Required for hydroelastic collision with
                non-unit shape scale. Defaults to ``None`` (no scale baking;
                scale applied at runtime).
            texture_format: Subgrid texture storage format. ``"uint16"``
                (default) uses 16-bit normalized textures for half the memory
                of ``"float32"`` with negligible precision loss. ``"uint8"``
                uses 8-bit textures for minimum memory.
            sign_method: Inside/outside sign query strategy.

                * ``"auto"`` (default): use parity rays if
                  :attr:`Mesh.is_watertight` is ``True``, else fall back to
                  winding numbers.
                * ``"parity"``: always use ``wp.mesh_query_point_sign_parity``.
                  Cheaper per sample but requires a watertight mesh; results on
                  open meshes are undefined.
                * ``"winding"``: always use
                  ``wp.mesh_query_point_sign_winding_number``. Robust for
                  general (possibly open or non-manifold) meshes but more
                  expensive to build and query.
            cache_dir: Optional directory holding cached cooked SDFs. When
                provided, the cooked SDF data (everything that backs the
                GPU 3D textures) is keyed by mesh content + build
                parameters and persisted as a single ``{hash}.sdf.npz``
                file (an uncompressed ``np.savez`` bundle of typed
                numpy arrays). A subsequent call with the same inputs
                reloads from disk and skips the expensive mesh-SDF
                build. ``shape_margin`` is applied at sample time and
                is *not* part of the cache key. Defaults to ``None``
                (cache disabled).

        Returns:
            A validated :class:`SDF` runtime handle.

        Raises:
            RuntimeError: if no CUDA device is available. The texture SDF build
                pipeline requires CUDA kernels and 3D textures.
            ValueError: if ``texture_format`` or ``sign_method`` is not one of
                the supported values.
        """
        if not wp.is_cuda_available():
            raise RuntimeError(
                "SDF.create_from_mesh requires a CUDA device: the texture SDF "
                "build pipeline uses CUDA kernels and wp.Texture3D. "
                "No CUDA-capable device was detected."
            )

        valid_sign_methods: tuple[SignMethod, ...] = ("auto", "parity", "winding")
        if sign_method not in valid_sign_methods:
            raise ValueError(f"Unknown sign_method {sign_method!r}. Expected one of {list(valid_sign_methods)}.")

        effective_max_resolution = 64 if max_resolution is None and target_voxel_size is None else max_resolution
        bake_scale = scale is not None
        effective_scale = scale if scale is not None else (1.0, 1.0, 1.0)
        is_watertight = mesh.is_watertight

        if sign_method == "auto":
            use_parity = is_watertight
        else:
            use_parity = sign_method == "parity"

        sign_method_resolved = "parity" if use_parity else "winding"

        from .sdf_texture import (  # noqa: PLC0415
            QuantizationMode,
            block_coords_from_subgrid_required,
            create_sparse_sdf_textures,
            create_texture_sdf_from_mesh,
        )

        _tex_fmt_map = {
            "float32": QuantizationMode.FLOAT32,
            "uint16": QuantizationMode.UINT16,
            "uint8": QuantizationMode.UINT8,
        }
        if texture_format not in _tex_fmt_map:
            raise ValueError(f"Unknown texture_format {texture_format!r}. Expected one of {list(_tex_fmt_map)}.")
        qmode = _tex_fmt_map[texture_format]

        cache_hash: str | None = None
        loaded_sparse_data = None
        if cache_dir is not None:
            from . import _sdf_cache  # noqa: PLC0415

            verts_for_hash = np.asarray(mesh.vertices, dtype=np.float32) * np.array(effective_scale, dtype=np.float32)
            indices_for_hash = np.asarray(mesh.indices, dtype=np.int32).reshape(-1)
            cache_hash = _sdf_cache.hash_inputs(
                vertices=verts_for_hash,
                indices=indices_for_hash,
                is_solid=bool(getattr(mesh, "is_solid", True)),
                narrow_band_range=narrow_band_range,
                target_voxel_size=target_voxel_size,
                max_resolution=effective_max_resolution,
                margin=margin,
                texture_format=texture_format,
                sign_method_resolved=sign_method_resolved,
                # winding_threshold's actual value is set post-cook, but
                # the cooked output's equivalence class only depends on
                # its sign. +0.5 is the canonical positive case.
                winding_threshold=0.5,
                scale=scale,
            )
            loaded_sparse_data = _sdf_cache.try_load_sparse_data(cache_dir, cache_hash)

        with wp.ScopedDevice(device):
            if loaded_sparse_data is not None:
                sdf_device = str(wp.get_device())
                sdf_params, coarse_texture, subgrid_texture = create_sparse_sdf_textures(loaded_sparse_data, sdf_device)
                sdf_params.scale_baked = bake_scale
                tex_block_coords = block_coords_from_subgrid_required(
                    loaded_sparse_data["subgrid_required"],
                    loaded_sparse_data["coarse_dims"],
                    loaded_sparse_data["subgrid_size"],
                    subgrid_occupied=loaded_sparse_data["subgrid_occupied"],
                )
                texture_data = sdf_params
            else:
                verts = mesh.vertices * np.array(effective_scale)[None, :]
                pos = wp.array(verts, dtype=wp.vec3)
                indices = wp.array(mesh.indices, dtype=wp.int32)

                winding_threshold = 0.5
                if use_parity:
                    tex_mesh = wp.Mesh(points=pos, indices=indices)
                else:
                    tex_mesh = wp.Mesh(points=pos, indices=indices, support_winding_number=True)
                    signed_volume = compute_mesh_signed_volume(pos, indices)
                    winding_threshold = 0.5 if signed_volume >= 0.0 else -0.5

                want_sparse = cache_dir is not None
                res = effective_max_resolution if effective_max_resolution is not None else 64
                result = create_texture_sdf_from_mesh(
                    tex_mesh,
                    margin=margin,
                    narrow_band_range=narrow_band_range,
                    max_resolution=res,
                    target_voxel_size=target_voxel_size,
                    quantization_mode=qmode,
                    winding_threshold=winding_threshold,
                    scale_baked=bake_scale,
                    use_parity=use_parity,
                    return_sparse_data=want_sparse,
                )
                if want_sparse:
                    texture_data, coarse_texture, subgrid_texture, tex_block_coords, sparse_data = result
                    if sparse_data is not None:
                        _sdf_cache.write(cache_dir, cache_hash, sparse_data)
                else:
                    texture_data, coarse_texture, subgrid_texture, tex_block_coords = result

        sdf = SDF(
            data=create_empty_sdf_data(),
            sparse_volume=None,
            coarse_volume=None,
            block_coords=[],
            texture_block_coords=tex_block_coords,
            texture_data=texture_data,
            shape_margin=shape_margin,
            _coarse_texture=coarse_texture,
            _subgrid_texture=subgrid_texture,
            _internal=True,
        )
        sdf.validate()
        return sdf

    @staticmethod
    def create_from_data(
        *,
        sparse_volume: wp.Volume | None = None,
        coarse_volume: wp.Volume | None = None,
        block_coords: np.ndarray | Sequence[wp.vec3us] | None = None,
        center: Sequence[float] | None = None,
        half_extents: Sequence[float] | None = None,
        background_value: float = MAXVAL,
        scale_baked: bool = False,
        shape_margin: float = 0.0,
        texture_data: "TextureSDFData | None" = None,
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
            shape_margin=shape_margin,
            texture_data=texture_data,
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
    points: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    volume_sum: wp.array[wp.float32],
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


@wp.func
def get_distance_to_mesh_parity(mesh: wp.uint64, point: wp.vec3, max_dist: wp.float32):
    """Signed distance using parity-based ray-cast for inside/outside classification.

    Cheaper than :func:`get_distance_to_mesh` (no winding-number accumulation)
    but requires a watertight mesh for correct results.
    """
    res = wp.mesh_query_point_sign_parity(mesh, point, max_dist)
    if res.result:
        closest = wp.mesh_eval_position(mesh, res.face, res.u, res.v)
        return res.sign * wp.length(closest - point)
    return max_dist


@wp.kernel
def sdf_from_mesh_kernel(
    mesh: wp.uint64,
    sdf: wp.uint64,
    tile_points: wp.array[wp.vec3i],
    shape_margin: wp.float32,
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
    signed_distance -= shape_margin
    wp.volume_store(sdf, x_id, y_id, z_id, signed_distance)


@wp.kernel(enable_backward=False)
def sdf_from_primitive_kernel(
    shape_type: wp.int32,
    shape_scale: wp.vec3,
    sdf: wp.uint64,
    tile_points: wp.array[wp.vec3i],
    shape_margin: wp.float32,
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
        signed_distance = sdf_sphere(sample_pos, shape_scale[0])
    elif shape_type == GeoType.BOX:
        signed_distance = sdf_box(sample_pos, shape_scale[0], shape_scale[1], shape_scale[2])
    elif shape_type == GeoType.CAPSULE:
        signed_distance = sdf_capsule(sample_pos, shape_scale[0], shape_scale[1], int(Axis.Z))
    elif shape_type == GeoType.CYLINDER:
        signed_distance = sdf_cylinder(sample_pos, shape_scale[0], shape_scale[1], int(Axis.Z))
    elif shape_type == GeoType.ELLIPSOID:
        signed_distance = sdf_ellipsoid(sample_pos, shape_scale)
    elif shape_type == GeoType.CONE:
        signed_distance = sdf_cone(sample_pos, shape_scale[0], shape_scale[1], int(Axis.Z))
    signed_distance -= shape_margin
    wp.volume_store(sdf, x_id, y_id, z_id, signed_distance)


@wp.kernel
def check_tile_occupied_mesh_kernel(
    mesh: wp.uint64,
    tile_points: wp.array[wp.vec3f],
    threshold: wp.vec2f,
    winding_threshold: wp.float32,
    tile_occupied: wp.array[bool],
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
    tile_points: wp.array[wp.vec3f],
    threshold: wp.vec2f,
    tile_occupied: wp.array[bool],
):
    tid = wp.tid()
    sample_pos = tile_points[tid]

    signed_distance = float(1.0e6)
    if shape_type == GeoType.SPHERE:
        signed_distance = sdf_sphere(sample_pos, shape_scale[0])
    elif shape_type == GeoType.BOX:
        signed_distance = sdf_box(sample_pos, shape_scale[0], shape_scale[1], shape_scale[2])
    elif shape_type == GeoType.CAPSULE:
        signed_distance = sdf_capsule(sample_pos, shape_scale[0], shape_scale[1], int(Axis.Z))
    elif shape_type == GeoType.CYLINDER:
        signed_distance = sdf_cylinder(sample_pos, shape_scale[0], shape_scale[1], int(Axis.Z))
    elif shape_type == GeoType.ELLIPSOID:
        signed_distance = sdf_ellipsoid(sample_pos, shape_scale)
    elif shape_type == GeoType.CONE:
        signed_distance = sdf_cone(sample_pos, shape_scale[0], shape_scale[1], int(Axis.Z))

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
    shape_margin: float = 0.0,
    narrow_band_distance: Sequence[float] = (-0.1, 0.1),
    margin: float = 0.05,
    target_voxel_size: float | None = None,
    max_resolution: int = 64,
    bake_scale: bool = False,
    verbose: bool = False,
    device: Devicelike | None = None,
) -> tuple[SDFData, wp.Volume | None, wp.Volume | None, Sequence[wp.vec3us]]:
    """Compute sparse and coarse SDF volumes for a shape.

    The SDF is computed in the mesh's unscaled local space. Scale is intentionally
    NOT a parameter - the collision system handles scaling at runtime, ensuring
    the SDF and mesh BVH stay consistent and allowing dynamic scale changes.

    Args:
        shape_type: Type of the shape.
        shape_geo: Optional source geometry. Required for mesh shapes.
        shape_scale: Scale factors for the mesh. Applied before SDF generation if bake_scale is True.
        shape_margin: Margin offset to subtract from SDF values.
        narrow_band_distance: Tuple of (inner, outer) distances for narrow band.
        margin: Margin to add to bounding box. Must be > 0.
        target_voxel_size: Target voxel size for sparse SDF grid. If None, computed as max_extent/max_resolution.
        max_resolution: Maximum dimension for sparse SDF grid when target_voxel_size is None. Must be divisible by 8.
        bake_scale: If True, bake shape_scale into the SDF. If False, use (1,1,1) scale.
        verbose: Print debug info.
        device: CUDA device for all GPU allocations. When ``None``, uses the
            current :class:`wp.ScopedDevice` or the Warp default device.

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

    with wp.ScopedDevice(device):
        assert isinstance(narrow_band_distance, Sequence), "narrow_band_distance must be a tuple of two floats"
        assert len(narrow_band_distance) == 2, "narrow_band_distance must be a tuple of two floats"
        assert narrow_band_distance[0] < 0.0 < narrow_band_distance[1], (
            "narrow_band_distance[0] must be less than 0.0 and narrow_band_distance[1] must be greater than 0.0"
        )
        assert margin > 0, "margin must be > 0"

        # Determine effective scale based on bake_scale flag
        effective_scale = tuple(shape_scale) if bake_scale else (1.0, 1.0, 1.0)

        offset = margin + shape_margin

        if shape_type == GeoType.MESH:
            if shape_geo is None:
                raise ValueError("shape_geo must be provided for GeoType.MESH.")
            verts = shape_geo.vertices * np.array(effective_scale)[None, :]
            pos = wp.array(verts, dtype=wp.vec3)
            indices = wp.array(shape_geo.indices, dtype=wp.int32)

            winding_threshold = 0.5
            mesh = wp.Mesh(points=pos, indices=indices, support_winding_number=True)
            signed_volume = compute_mesh_signed_volume(pos, indices)
            winding_threshold = 0.5 if signed_volume >= 0.0 else -0.5
            if verbose and signed_volume < 0:
                print("Mesh has inverted winding (negative volume), using threshold -0.5")
            m_id = mesh.id

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

        num_allocated_tiles = len(tile_points)
        if shape_type == GeoType.MESH:
            wp.launch(
                sdf_from_mesh_kernel,
                dim=(num_allocated_tiles, 8, 8, 8),
                inputs=[m_id, sparse_volume.id, tile_points_wp, shape_margin, winding_threshold],
            )
        else:
            wp.launch(
                sdf_from_primitive_kernel,
                dim=(num_allocated_tiles, 8, 8, 8),
                inputs=[shape_type, effective_scale, sparse_volume.id, tile_points_wp, shape_margin],
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

        if shape_type == GeoType.MESH:
            wp.launch(
                sdf_from_mesh_kernel,
                dim=(1, 8, 8, 8),
                inputs=[m_id, coarse_volume.id, coarse_tile_points_wp, shape_margin, winding_threshold],
            )
        else:
            wp.launch(
                sdf_from_primitive_kernel,
                dim=(1, 8, 8, 8),
                inputs=[shape_type, effective_scale, coarse_volume.id, coarse_tile_points_wp, shape_margin],
            )

        if shape_type == GeoType.MESH:
            # Synchronize to ensure all kernels reading from the temporary wp.Mesh
            # (created above for SDF construction) have completed before it goes
            # out of scope.  Without this, wp.Mesh.__del__ can free the BVH / winding-
            # number data while an asynchronous kernel is still reading it, corrupting
            # the CUDA context on some driver/GPU combinations (#1616).
            wp.synchronize()

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
    shape_margin: float = 0.0,
    narrow_band_distance: Sequence[float] = (-0.1, 0.1),
    margin: float = 0.05,
    target_voxel_size: float | None = None,
    max_resolution: int = 64,
    bake_scale: bool = False,
    verbose: bool = False,
    device: Devicelike | None = None,
) -> tuple[SDFData, wp.Volume | None, wp.Volume | None, Sequence[wp.vec3us]]:
    """Compute sparse and coarse SDF volumes for a shape.

    Mesh shape dispatches through :meth:`SDF.create_from_mesh` to keep that path canonical.

    Args:
        shape_type: Geometry type identifier from :class:`GeoType`.
        shape_geo: Source mesh geometry when ``shape_type`` is ``GeoType.MESH``.
        shape_scale: Shape scale [unitless].
        shape_margin: Shape margin offset [m] subtracted from sampled SDF.
        narrow_band_distance: Signed narrow-band distance range [m] as ``(inner, outer)``.
        margin: Extra AABB padding [m] added before discretization.
        target_voxel_size: Target sparse-grid voxel size [m]. If provided, takes
            precedence over ``max_resolution``.
        max_resolution: Maximum sparse-grid dimension [voxel] when
            ``target_voxel_size`` is not provided.
        bake_scale: If ``True``, bake ``shape_scale`` into generated SDF data.
        verbose: If ``True``, print debug information during SDF construction.
        device: CUDA device for SDF allocation. When ``None``, uses the
            current :class:`wp.ScopedDevice` or the Warp default device.

    Returns:
        Tuple ``(sdf_data, sparse_volume, coarse_volume, block_coords)``.
    """
    if shape_type == GeoType.MESH:
        if shape_geo is None:
            raise ValueError("shape_geo must be provided for GeoType.MESH.")
        # Canonical mesh path: use SDF.create_from_mesh for all mesh SDF generation.
        sdf = SDF.create_from_mesh(
            shape_geo,
            device=device,
            narrow_band_range=tuple(narrow_band_distance),
            target_voxel_size=target_voxel_size,
            max_resolution=max_resolution,
            margin=margin,
            shape_margin=shape_margin,
            scale=tuple(shape_scale) if bake_scale else None,
        )
        return sdf.to_kernel_data(), sdf.sparse_volume, sdf.coarse_volume, (sdf.block_coords or [])

    return _compute_sdf_from_shape_impl(
        shape_type=shape_type,
        shape_geo=shape_geo,
        shape_scale=shape_scale,
        shape_margin=shape_margin,
        narrow_band_distance=narrow_band_distance,
        margin=margin,
        target_voxel_size=target_voxel_size,
        max_resolution=max_resolution,
        bake_scale=bake_scale,
        verbose=verbose,
        device=device,
    )


def compute_isomesh(volume: wp.Volume, isovalue: float = 0.0, device: Devicelike | None = None) -> Mesh | None:
    """Compute an isosurface mesh from a sparse SDF volume.

    Uses a two-pass approach to minimize memory allocation:
    1. First pass: count actual triangles produced
    2. Allocate exact memory needed
    3. Second pass: generate vertices

    Args:
        volume: The SDF volume.
        isovalue: Surface level to extract [m].  ``0.0`` gives the
            zero-isosurface; positive values extract an outward offset surface.
        device: CUDA device for GPU allocations.  When ``None``, uses the
            current :class:`wp.ScopedDevice` or the Warp default device.

    Returns:
        Mesh object containing the isosurface mesh.
    """
    if device is not None:
        device = wp.get_device(device)
    else:
        device = wp.get_device()
    mc_tables = get_mc_tables(device)

    tile_points = volume.get_tiles()
    tile_points_wp = wp.array(tile_points, dtype=wp.vec3i, device=device)
    num_tiles = tile_points.shape[0]

    if num_tiles == 0:
        return None

    face_count = wp.zeros((1,), dtype=int, device=device)
    wp.launch(
        count_isomesh_faces_kernel,
        dim=(num_tiles, 8, 8, 8),
        inputs=[volume.id, tile_points_wp, mc_tables[0], mc_tables[3], float(isovalue)],
        outputs=[face_count],
        device=device,
    )

    num_faces = int(face_count.numpy()[0])
    if num_faces == 0:
        return None

    max_verts = 3 * num_faces
    verts = wp.empty((max_verts,), dtype=wp.vec3, device=device)
    face_normals = wp.empty((num_faces,), dtype=wp.vec3, device=device)

    face_count.zero_()
    wp.launch(
        generate_isomesh_kernel,
        dim=(num_tiles, 8, 8, 8),
        inputs=[volume.id, tile_points_wp, mc_tables[0], mc_tables[4], mc_tables[3], float(isovalue)],
        outputs=[face_count, verts, face_normals],
        device=device,
    )

    verts_np = verts.numpy()
    faces_np = np.arange(3 * num_faces).reshape(-1, 3)

    faces_np = faces_np[:, ::-1]
    return Mesh(verts_np, faces_np)


def compute_offset_mesh(
    shape_type: int,
    shape_geo: Mesh | None = None,
    shape_scale: Sequence[float] = (1.0, 1.0, 1.0),
    offset: float = 0.0,
    max_resolution: int = 48,
    device: Devicelike | None = None,
) -> Mesh | None:
    """Compute the offset (Minkowski-inflated) isosurface mesh of a shape.

    For primitive shapes with analytical SDFs (sphere, box, capsule, cylinder,
    ellipsoid, cone) this evaluates the SDF directly on a dense grid, avoiding
    NanoVDB volume construction.  For mesh / convex-mesh shapes with a
    pre-built :class:`SDF` (via :meth:`Mesh.build_sdf`), the existing volume
    or texture SDF is reused.  Only when no pre-built SDF is available does
    this fall back to constructing a temporary NanoVDB volume.

    Args:
        shape_type: Geometry type identifier from :class:`GeoType`.
        shape_geo: Source mesh geometry when *shape_type* is
            :attr:`GeoType.MESH` or :attr:`GeoType.CONVEX_MESH`.
        shape_scale: Shape scale factors [unitless].
        offset: Outward surface offset [m].  Use ``0`` for the original surface.
        max_resolution: Maximum grid dimension [voxels].
        device: CUDA device for GPU allocations.

    Returns:
        A :class:`Mesh` representing the offset isosurface, or ``None`` when
        the shape type is unsupported (plane, heightfield) or the resulting
        mesh would be empty.
    """
    if shape_type in (GeoType.PLANE, GeoType.HFIELD):
        return None

    if shape_type in _ANALYTICAL_SDF_TYPES:
        return compute_offset_mesh_analytical(
            shape_type=shape_type,
            shape_scale=shape_scale,
            offset=offset,
            max_resolution=max_resolution,
            device=device,
        )

    # Reuse existing SDF on the mesh when available (avoids building a
    # NanoVDB volume from scratch).  This assumes the stored SDF was built
    # with shape_margin=0 (default) so that extracting at isovalue=offset
    # yields the correct inflated surface.  The fallback path below uses
    # bake_scale=True, so we skip the shortcut when scale hasn't been baked
    # AND the caller requests non-unit scale — otherwise the extracted
    # vertices would be in unscaled mesh-local space.
    if shape_geo is not None:
        existing_sdf = getattr(shape_geo, "sdf", None)
        if existing_sdf is not None:
            scale_ok = existing_sdf.data.scale_baked or all(abs(s - 1.0) < 1e-6 for s in shape_scale)
            if scale_ok:
                result = existing_sdf.extract_isomesh(isovalue=offset, device=device)
                if result is not None:
                    return result

    if shape_type not in (GeoType.MESH, GeoType.CONVEX_MESH):
        raise ValueError(
            f"compute_offset_mesh: unsupported shape type {shape_type} "
            f"without an analytical SDF or a pre-built SDF on the geometry."
        )

    if shape_geo is None:
        raise ValueError("shape_geo must be provided for mesh/convex-mesh offset meshing.")

    padding = max(abs(offset) * 0.5, 0.02)
    narrow_band = (-abs(offset) - padding, abs(offset) + padding)
    margin = max(padding, 0.05)

    _sdf_data, sparse_volume, _coarse_volume, _block_coords = _compute_sdf_from_shape_impl(
        shape_type=GeoType.MESH,
        shape_geo=shape_geo,
        shape_scale=shape_scale,
        shape_margin=offset,
        narrow_band_distance=narrow_band,
        margin=margin,
        max_resolution=max_resolution,
        bake_scale=True,
        device=device,
    )

    if sparse_volume is None:
        return None

    return compute_isomesh(sparse_volume, device=device)


@wp.kernel(enable_backward=False)
def count_isomesh_faces_kernel(
    sdf: wp.uint64,
    tile_points: wp.array[wp.vec3i],
    tri_range_table: wp.array[wp.int32],
    corner_offsets_table: wp.array[wp.vec3ub],
    isovalue: wp.float32,
    face_count: wp.array[int],
):
    """Count isosurface faces without generating vertices (first pass of two-pass approach).
    Only processes specified tiles. Launch with dim=(num_tiles, 8, 8, 8).
    """
    tile_idx, local_x, local_y, local_z = wp.tid()

    tile_origin = tile_points[tile_idx]
    x_id = tile_origin[0] + local_x
    y_id = tile_origin[1] + local_y
    z_id = tile_origin[2] + local_z

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
    tile_points: wp.array[wp.vec3i],
    tri_range_table: wp.array[wp.int32],
    flat_edge_verts_table: wp.array[wp.vec2ub],
    corner_offsets_table: wp.array[wp.vec3ub],
    isovalue: wp.float32,
    face_count: wp.array[int],
    vertices: wp.array[wp.vec3],
    face_normals: wp.array[wp.vec3],
):
    """Generate isosurface mesh vertices and normals using marching cubes.
    Only processes specified tiles. Launch with dim=(num_tiles, 8, 8, 8).
    """
    tile_idx, local_x, local_y, local_z = wp.tid()

    tile_origin = tile_points[tile_idx]
    x_id = tile_origin[0] + local_x
    y_id = tile_origin[1] + local_y
    z_id = tile_origin[2] + local_z

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

    tri_range_start = tri_range_table[cube_idx]
    tri_range_end = tri_range_table[cube_idx + 1]
    num_verts = tri_range_end - tri_range_start

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
            isovalue,
        )
        vertices[3 * out_idx_faces + 3 * fi + 0] = wp.vec3(face_verts[0])
        vertices[3 * out_idx_faces + 3 * fi + 1] = wp.vec3(face_verts[1])
        vertices[3 * out_idx_faces + 3 * fi + 2] = wp.vec3(face_verts[2])
        face_normals[out_idx_faces + fi] = normal


# ---------------------------------------------------------------------------
# Dense-grid analytical marching cubes for primitive shapes
# ---------------------------------------------------------------------------
# These kernels skip NanoVDB volume construction entirely and evaluate the
# analytical SDF on a flat dense grid, which is significantly faster for
# primitives (sphere, box, capsule, cylinder, ellipsoid, cone).
# ---------------------------------------------------------------------------

_ANALYTICAL_SDF_TYPES = frozenset(
    {
        GeoType.SPHERE,
        GeoType.BOX,
        GeoType.CAPSULE,
        GeoType.CYLINDER,
        GeoType.ELLIPSOID,
        GeoType.CONE,
    }
)


@wp.kernel(enable_backward=False)
def _populate_dense_sdf_kernel(
    shape_type: wp.int32,
    shape_scale: wp.vec3,
    origin: wp.vec3,
    voxel_size: wp.vec3,
    ny: wp.int32,
    nz: wp.int32,
    shape_offset: wp.float32,
    sdf_values: wp.array[wp.float32],
):
    """Evaluate analytical SDF at every point of a dense regular grid."""
    x, y, z = wp.tid()
    pos = wp.vec3(
        origin[0] + float(x) * voxel_size[0],
        origin[1] + float(y) * voxel_size[1],
        origin[2] + float(z) * voxel_size[2],
    )
    d = float(1.0e6)
    if shape_type == GeoType.SPHERE:
        d = sdf_sphere(pos, shape_scale[0])
    elif shape_type == GeoType.BOX:
        d = sdf_box(pos, shape_scale[0], shape_scale[1], shape_scale[2])
    elif shape_type == GeoType.CAPSULE:
        d = sdf_capsule(pos, shape_scale[0], shape_scale[1], int(Axis.Z))
    elif shape_type == GeoType.CYLINDER:
        d = sdf_cylinder(pos, shape_scale[0], shape_scale[1], int(Axis.Z))
    elif shape_type == GeoType.ELLIPSOID:
        d = sdf_ellipsoid(pos, shape_scale)
    elif shape_type == GeoType.CONE:
        d = sdf_cone(pos, shape_scale[0], shape_scale[1], int(Axis.Z))
    sdf_values[x * ny * nz + y * nz + z] = d - shape_offset


@wp.kernel(enable_backward=False)
def _count_dense_mc_faces_kernel(
    sdf_values: wp.array[wp.float32],
    ny: wp.int32,
    nz: wp.int32,
    tri_range_table: wp.array[wp.int32],
    corner_offsets_table: wp.array[wp.vec3ub],
    face_count: wp.array[int],
):
    """Count marching-cubes faces on a dense SDF grid (first MC pass)."""
    x, y, z = wp.tid()
    cube_idx = wp.int32(0)
    for i in range(8):
        co = wp.vec3i(corner_offsets_table[i])
        v = sdf_values[(x + co[0]) * ny * nz + (y + co[1]) * nz + (z + co[2])]
        if v < 0.0:
            cube_idx |= 1 << i
    tri_start = tri_range_table[cube_idx]
    tri_end = tri_range_table[cube_idx + 1]
    num_faces = (tri_end - tri_start) // 3
    if num_faces > 0:
        wp.atomic_add(face_count, 0, num_faces)


@wp.kernel(enable_backward=False)
def _generate_dense_mc_kernel(
    sdf_values: wp.array[wp.float32],
    ny: wp.int32,
    nz: wp.int32,
    origin: wp.vec3,
    voxel_size: wp.vec3,
    tri_range_table: wp.array[wp.int32],
    flat_edge_verts_table: wp.array[wp.vec2ub],
    corner_offsets_table: wp.array[wp.vec3ub],
    face_count: wp.array[int],
    vertices: wp.array[wp.vec3],
    face_normals: wp.array[wp.vec3],
):
    """Generate marching-cubes vertices on a dense SDF grid (second MC pass)."""
    x, y, z = wp.tid()
    cube_idx = wp.int32(0)
    corner_vals = vec8f()
    for i in range(8):
        co = wp.vec3i(corner_offsets_table[i])
        v = sdf_values[(x + co[0]) * ny * nz + (y + co[1]) * nz + (z + co[2])]
        corner_vals[i] = v
        if v < 0.0:
            cube_idx |= 1 << i

    tri_start = tri_range_table[cube_idx]
    tri_end = tri_range_table[cube_idx + 1]
    num_verts = tri_end - tri_start
    num_faces = num_verts // 3
    out_idx = wp.atomic_add(face_count, 0, num_faces)
    if num_verts == 0:
        return

    base = wp.vec3(float(x), float(y), float(z))
    for fi in range(5):
        if fi >= num_faces:
            return
        face_verts = wp.mat33f()
        for vi in range(3):
            ev = wp.vec2i(flat_edge_verts_table[tri_start + 3 * fi + vi])
            val_0 = wp.float32(corner_vals[ev[0]])
            val_1 = wp.float32(corner_vals[ev[1]])
            p_0 = wp.vec3f(corner_offsets_table[ev[0]])
            p_1 = wp.vec3f(corner_offsets_table[ev[1]])
            val_diff = val_1 - val_0
            if wp.abs(val_diff) < wp.static(MC_EDGE_VAL_DIFF_EPS):
                p = 0.5 * (p_0 + p_1)
            else:
                t = wp.clamp((0.0 - val_0) / val_diff, wp.static(MC_EDGE_CLAMP_MIN), wp.static(MC_EDGE_CLAMP_MAX))
                p = p_0 + t * (p_1 - p_0)
            local = base + p
            face_verts[vi] = wp.vec3(
                origin[0] + local[0] * voxel_size[0],
                origin[1] + local[1] * voxel_size[1],
                origin[2] + local[2] * voxel_size[2],
            )
        n = wp.cross(face_verts[1] - face_verts[0], face_verts[2] - face_verts[0])
        n_sq = wp.dot(n, n)
        if n_sq < wp.static(MC_DEGENERATE_N_SQ_EPS):
            normal = wp.vec3(0.0, 0.0, 1.0)
        else:
            normal = n / wp.sqrt(n_sq)
        vertices[3 * out_idx + 3 * fi + 0] = wp.vec3(face_verts[0])
        vertices[3 * out_idx + 3 * fi + 1] = wp.vec3(face_verts[1])
        vertices[3 * out_idx + 3 * fi + 2] = wp.vec3(face_verts[2])
        face_normals[out_idx + fi] = normal


def compute_offset_mesh_analytical(
    shape_type: int,
    shape_scale: Sequence[float] = (1.0, 1.0, 1.0),
    offset: float = 0.0,
    max_resolution: int = 48,
    device: Devicelike | None = None,
) -> Mesh | None:
    """Compute the offset isosurface mesh for a primitive via direct analytical SDF evaluation.

    Unlike :func:`compute_offset_mesh` this skips NanoVDB volume construction
    and evaluates the analytical SDF on a dense regular grid before running
    marching cubes.  This is faster for primitive shapes.

    Args:
        shape_type: Geometry type identifier from :class:`GeoType`.  Must be a
            primitive with an analytical SDF (sphere, box, capsule, cylinder,
            ellipsoid, or cone).
        shape_scale: Shape scale factors [unitless].
        offset: Outward surface offset [m].  Use ``0`` for the original surface.
        max_resolution: Maximum grid dimension [voxels].
        device: CUDA device for GPU allocations.

    Returns:
        A :class:`Mesh` representing the offset isosurface, or ``None`` when
        the shape type is unsupported or the mesh would be empty.
    """
    if shape_type not in _ANALYTICAL_SDF_TYPES:
        return None

    if device is None:
        cur = wp.get_device()
        device = cur if cur.is_cuda else "cuda:0"

    with wp.ScopedDevice(device):
        min_ext_list, max_ext_list = get_primitive_extents(shape_type, shape_scale)

        padding = max(abs(offset) * 0.5, 0.02)
        total_expansion = max(abs(offset) + padding, 0.05)

        min_ext = np.array(min_ext_list, dtype=np.float64) - total_expansion
        max_ext = np.array(max_ext_list, dtype=np.float64) + total_expansion
        ext = max_ext - min_ext

        # Adaptively increase resolution when the expansion dominates the
        # shape extents (e.g. a 1 mm sphere with 0.05 m expansion).  This
        # ensures at least ~4 voxels span the smallest shape dimension while
        # capping total memory via a voxel budget (default ~4M voxels ≈ 16 MB
        # of float32) so thin/flat shapes don't cause OOM.
        max_voxel_budget = 4_000_000
        shape_ext = np.array(max_ext_list, dtype=np.float64) - np.array(min_ext_list, dtype=np.float64)
        min_shape_dim = float(np.min(shape_ext))
        if min_shape_dim > 0.0:
            effective_resolution = max(max_resolution, int(np.ceil(float(np.max(ext)) / min_shape_dim * 4)))
        else:
            effective_resolution = max_resolution

        max_extent = float(np.max(ext))
        voxel_target = max_extent / effective_resolution
        grid_dims = np.maximum(np.round(ext / voxel_target).astype(int), 2)

        total_voxels = int(np.prod(grid_dims))
        if total_voxels > max_voxel_budget:
            scale_factor = (max_voxel_budget / total_voxels) ** (1.0 / 3.0)
            grid_dims = np.maximum(np.round(grid_dims * scale_factor).astype(int), 2)
            logger.warning(
                "compute_offset_mesh_analytical: clamped grid from %d voxels to %dx%dx%d (%d voxels) "
                "for shape type %d with scale %s. Visualization may be lower-fidelity for this shape.",
                total_voxels,
                grid_dims[0],
                grid_dims[1],
                grid_dims[2],
                int(np.prod(grid_dims)),
                shape_type,
                shape_scale,
            )

        actual_voxel_size = ext / (grid_dims - 1)

        nx, ny, nz = int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2])

        sdf_values = wp.empty((nx * ny * nz,), dtype=wp.float32, device=device)
        wp.launch(
            _populate_dense_sdf_kernel,
            dim=(nx, ny, nz),
            inputs=[
                int(shape_type),
                wp.vec3(float(shape_scale[0]), float(shape_scale[1]), float(shape_scale[2])),
                wp.vec3(float(min_ext[0]), float(min_ext[1]), float(min_ext[2])),
                wp.vec3(float(actual_voxel_size[0]), float(actual_voxel_size[1]), float(actual_voxel_size[2])),
                ny,
                nz,
                float(offset),
            ],
            outputs=[sdf_values],
            device=device,
        )

        mc_tables = get_mc_tables(device)

        face_count = wp.zeros((1,), dtype=int, device=device)
        wp.launch(
            _count_dense_mc_faces_kernel,
            dim=(nx - 1, ny - 1, nz - 1),
            inputs=[sdf_values, ny, nz, mc_tables[0], mc_tables[3]],
            outputs=[face_count],
            device=device,
        )

        num_faces = int(face_count.numpy()[0])
        if num_faces == 0:
            logger.warning(
                "compute_offset_mesh_analytical: marching cubes produced no faces for shape type %d "
                "with scale %s and offset %.4g (grid %dx%dx%d). "
                "The shape may be too small for the grid resolution.",
                shape_type,
                shape_scale,
                offset,
                nx,
                ny,
                nz,
            )
            return None

        verts = wp.empty((3 * num_faces,), dtype=wp.vec3, device=device)
        face_normals_out = wp.empty((num_faces,), dtype=wp.vec3, device=device)

        face_count.zero_()
        wp.launch(
            _generate_dense_mc_kernel,
            dim=(nx - 1, ny - 1, nz - 1),
            inputs=[
                sdf_values,
                ny,
                nz,
                wp.vec3(float(min_ext[0]), float(min_ext[1]), float(min_ext[2])),
                wp.vec3(float(actual_voxel_size[0]), float(actual_voxel_size[1]), float(actual_voxel_size[2])),
                mc_tables[0],
                mc_tables[4],
                mc_tables[3],
            ],
            outputs=[face_count, verts, face_normals_out],
            device=device,
        )

        verts_np = verts.numpy()
        faces_np = np.arange(3 * num_faces).reshape(-1, 3)
        faces_np = faces_np[:, ::-1]
        return Mesh(verts_np, faces_np)
