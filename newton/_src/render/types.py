# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import enum
from dataclasses import dataclass

import warp as wp

from ..utils.color import ColorSpace


class LightType(enum.IntEnum):
    """Light types supported by the Warp raytracer."""

    SPOTLIGHT = 0
    """Spotlight."""

    DIRECTIONAL = 1
    """Directional Light."""


class RenderOrder(enum.IntEnum):
    """Render Order"""

    PIXEL_PRIORITY = 0
    """Render the same pixel of every view before continuing to the next one"""
    VIEW_PRIORITY = 1
    """Render all pixels of a whole view before continuing to the next one"""
    TILED = 2
    """Render pixels in tiles, defined by tile_width x tile_height"""


class WorldRenderFlag(enum.IntEnum):
    """Negative disable sentinels for the per-view ``world_indices`` array.

    Each entry of ``world_indices`` is either a non-negative world index to
    render for that view, or one of these negative sentinels to skip the view.
    """

    DISABLE_PRESERVE = -101
    """Skip rendering and leave output pixels unchanged."""

    DISABLE_CLEAR = -102
    """Skip rendering and write clear values to output pixels."""


class GaussianRenderMode(enum.IntEnum):
    """Gaussian Render Mode"""

    FAST = 0
    """Fast Render Mode"""

    QUALITY = 1
    """Quality Render Mode, collect hits until minimum transmittance is reached"""


class AntiAliasing(enum.IntEnum):
    """Anti-aliasing mode applied to multisampled camera-ray bundles."""

    NONE = 0
    """Resolve a single ray per pixel. Multisample ray bundles are rejected."""

    SSAA = 1
    """Supersample: trace and shade every ray in the bundle, then average the results.

    Highest quality and highest cost; shading runs once per sample.
    """

    MSAA = 2
    """Multisample: resolve every ray against its own nearest hit and reuse shading
    for subsamples covering the same surface.

    Anti-aliases silhouettes against both the background and other objects at a fraction of
    the :attr:`SSAA` cost when shading is expensive. Each surface is shaded using
    the first ray in bundle order that hits it, which may not be the covered ray
    nearest the pixel center. The built-in ray helpers put a center ray first for
    three or more samples, but surfaces missed by that ray can still be shaded
    off-center. Interior shading is not supersampled.
    Standalone particles and deformable triangle meshes use the particle or face index
    alongside their shared hit IDs, so distinct surfaces are shaded separately.
    """


class DeformableSurfaceMode(enum.IntEnum):
    """Which deformable surfaces a :class:`~newton.sensors.SensorCamera` uploads and renders.

    Visual meshes can overlap the simulation surface or lie inside it. Drawing both may
    hide parts of the visual mesh. The choice is made when the sensor is constructed.

    .. experimental::

        This option may change without a formal deprecation cycle while deformable visual
        support is experimental.
    """

    BOTH = 0
    """Render the simulation triangles (cloth and soft-body surfaces) and the visual meshes, as the viewers do."""
    SIMULATION_ONLY = 1
    """Render the simulation triangles only; the model's visual meshes are left out."""
    VISUAL_ONLY = 2
    """Render the visual meshes only; every simulation triangle is left out, including surfaces without a visual mesh."""


class TextureProjectionMode(enum.IntEnum):
    """Projection mode for texture-mapped shapes without authored UVs."""

    CUBIC = 0
    """Project from the dominant local axis and sample once."""

    TRIPLANAR = 1
    """Blend samples from all three local axes using normal-based weights."""


@dataclass(unsafe_hash=True)
class RenderConfig:
    """Raytrace render settings shared across all worlds."""

    enable_global_world: bool = True
    """Include shapes that belong to no specific world."""

    enable_textures: bool = False
    """Enable texture-mapped rendering for shapes and skinned meshes."""

    texture_projection_mode: TextureProjectionMode = TextureProjectionMode.CUBIC
    """Projection mode for texture-mapped shapes without UVs."""

    enable_shadows: bool = False
    """Enable shadow rays for directional lights.

    Shadows are only rendered when a light also casts them (e.g. a light created
    via :meth:`~newton.sensors.SensorCamera.create_default_light` with
    ``enable_shadows=True``); with no shadow-casting light this setting has no
    effect.
    """

    enable_ambient_lighting: bool = True
    """Enable ambient lighting for the scene."""

    enable_particles: bool = True
    """Enable standalone particle rendering.

    When the model has simulation triangles, particles referenced by them or by
    tetrahedra are not emitted as spheres, even when that surface is hidden.
    """

    enable_backface_culling: bool = True
    """Cull back-facing triangles of rigid meshes and deformable surfaces.

    When disabled, deformable triangle surfaces render from both sides with
    normals oriented toward the camera.
    """

    enable_fast_math: bool = True
    """Compile render kernels with CUDA fast math."""

    output_color_space: ColorSpace = ColorSpace.SRGB
    """Color space for packed color and albedo outputs.

    Use ``ColorSpace.SRGB`` for display-encoded bytes or
    ``ColorSpace.LINEAR`` for linear RGB bytes.
    """

    render_order: RenderOrder = RenderOrder.PIXEL_PRIORITY
    """Render traversal order (see :class:`RenderOrder`)."""

    anti_aliasing: AntiAliasing = AntiAliasing.NONE
    """Anti-aliasing mode applied to multisampled ray bundles (see :class:`AntiAliasing`).

    Has no effect on single-sample ray bundles. Multisampled bundles require
    :attr:`AntiAliasing.SSAA` or :attr:`AntiAliasing.MSAA`.
    """

    tile_width: int = 16
    """Tile width [px] for ``RenderOrder.TILED`` traversal."""

    tile_height: int = 8
    """Tile height [px] for ``RenderOrder.TILED`` traversal."""

    block_dim: int = 64
    """Thread block dimension forwarded to ``wp.launch`` for the render megakernel.

    Launch-time tuning only: it does not affect kernel codegen, so the renderer
    excludes it from the kernel cache key and changing it never triggers a
    recompilation. The OptiX backend ignores it.
    """

    max_distance: float = 1000.0
    """Maximum ray distance [m]."""

    gaussians_mode: GaussianRenderMode = GaussianRenderMode.FAST
    """Gaussian splatting render mode (see :class:`GaussianRenderMode`)."""

    gaussians_min_transmittance: float = 0.49
    """Minimum transmittance before early-out during Gaussian rendering."""

    gaussians_max_num_hits: int = 20
    """Maximum Gaussian hits accumulated per ray."""


@dataclass(unsafe_hash=True)
class RenderState:
    """Mutable flags tracking which render outputs are active."""

    gaussian_shapes_per_ray: int = 0
    """Most Gaussian shapes one ray can cross: those of the world with the most, plus the global ones."""
    has_particles: bool = False
    """Whether the scene has standalone particle items."""
    has_triangle_mesh: bool = False
    """Whether the scene has deformable triangle items."""
    has_world_indices: bool = False
    has_triangle_mesh_textures: bool = False
    render_color: bool = False
    render_depth: bool = False
    render_forward_depth: bool = False
    render_shape_index: bool = False
    render_normal: bool = False
    render_albedo: bool = False
    render_hdr_color: bool = False


@dataclass(unsafe_hash=True)
class ClearData:
    """Default values written to output images before rendering."""

    clear_color: int = 0
    """Packed RGBA value written to the color output."""
    clear_depth: float = 0.0
    """Depth value written to the depth and forward-depth outputs [m]."""
    clear_shape_index: int = 0xFFFFFFFF
    """Shape-index sentinel written to the shape-index output."""
    clear_normal: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Normal vector written to the normal output."""
    clear_albedo: int = 0
    """Packed RGBA value written to the albedo output."""


@wp.struct
class SceneData:
    """What a camera ray traverses: one acceleration structure over *scene items*.

    Items are numbered shapes first, then standalone particles, then deformable triangles, so the
    range of a hit item tells what was hit and its offset in the range which one. With the Warp
    backend, :attr:`id` is a ``wp.Bvh`` whose leaves are the items, grouped by world. With OptiX it
    is an instance structure: shape instances report their shape as instance ID, and one instance
    each holds every world's particles and deformable triangles, packed into world cells.
    """

    id: wp.uint64
    """Warp ``wp.Bvh`` or OptiX instance structure over the items, or 0 without items."""
    world_roots: wp.array[wp.int32]
    """Warp: BVH group root of each world, then of the global world. OptiX: the planes of each world,
    flattened ``[world_count, planes per world]``."""
    shape_item_count: wp.int32
    """Items ``[0, shape_item_count)`` are shapes."""
    item_shapes: wp.array[wp.uint32]
    """Warp: shape of each shape item, or ``BVH_SHAPE_INACTIVE`` to skip it."""
    particle_item_count: wp.int32
    """The next ``particle_item_count`` items are standalone particles."""
    item_particles: wp.array[wp.int32]
    """Particle of each particle item; with OptiX, of each particle primitive."""
    particle_positions: wp.array[wp.vec3f]
    """Particle positions [m]."""
    particle_radii: wp.array[wp.float32]
    """Particle radii [m]."""
    triangle_points: wp.array[wp.vec3f]
    """Points of the deformable triangles [m]; the remaining items are their faces, in order."""
    triangle_indices: wp.array[wp.int32]
    """Vertex triplets of the deformable triangles."""
    cluster_triangle_start: wp.array[wp.int32]
    """OptiX clusters: first entry of each cluster in :attr:`cluster_triangles`; empty without clusters."""
    cluster_triangles: wp.array[wp.int32]
    """OptiX clusters: face of each clustered triangle, in cluster order."""
    slot_gaussians: wp.array2d[wp.int32]
    """OptiX: Gaussian shape in each slot of each world's rays, shape ``[world_count, slots]``."""


@wp.struct
class MeshData:
    """Per-mesh auxiliary vertex data for texture mapping and smooth shading.

    Attributes:
        uvs: Per-vertex UV coordinates, shape ``[vertex_count, 2]``, dtype ``vec2f``.
        normals: Per-vertex normals for smooth shading, shape ``[vertex_count, 3]``, dtype ``vec3f``.
    """

    uvs: wp.array[wp.vec2f]
    normals: wp.array[wp.vec3f]


@wp.struct
class TextureData:
    """Texture image data for surface shading during raytracing.

    Uses a hardware-accelerated ``wp.Texture2D`` with bilinear filtering.

    Attributes:
        texture: 2D Texture as ``wp.Texture2D``.
        repeat: UV tiling factors along U and V axes.
    """

    texture: wp.Texture2D
    repeat: wp.vec2f
