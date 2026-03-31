# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import enum
from dataclasses import dataclass

import warp as wp


class RenderLightType(enum.IntEnum):
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


class GaussianRenderMode(enum.IntEnum):
    """Gaussian Render Mode"""

    FAST = 0
    """Fast Render Mode"""

    QUALITY = 1
    """Quality Render Mode, collect hits until minimum transmittance is reached"""


@dataclass(unsafe_hash=True)
class RenderConfig:
    """Raytrace render settings shared across all worlds."""

    enable_global_world: bool = True
    """Include shapes that belong to no specific world."""

    enable_textures: bool = False
    """Enable texture-mapped rendering for meshes."""

    enable_shadows: bool = False
    """Enable shadow rays for directional lights."""

    enable_ambient_lighting: bool = True
    """Enable ambient lighting for the scene."""

    enable_particles: bool = True
    """Enable particle rendering."""

    enable_backface_culling: bool = True
    """Cull back-facing triangles."""

    render_order: int = RenderOrder.PIXEL_PRIORITY
    """Render traversal order (see :class:`RenderOrder`)."""

    tile_width: int = 16
    """Tile width [px] for ``RenderOrder.TILED`` traversal."""

    tile_height: int = 8
    """Tile height [px] for ``RenderOrder.TILED`` traversal."""

    max_distance: float = 1000.0
    """Maximum ray distance [m]."""

    gaussians_mode: int = GaussianRenderMode.FAST
    """Gaussian splatting render mode (see :class:`GaussianRenderMode`)."""

    gaussians_min_transmittance: float = 0.49
    """Minimum transmittance before early-out during Gaussian rendering."""

    gaussians_max_num_hits: int = 20
    """Maximum Gaussian hits accumulated per ray."""


@dataclass(unsafe_hash=True)
class ClearData:
    """Default values written to output images before rendering."""

    clear_color: int = 0
    clear_depth: float = 0.0
    clear_shape_index: int = 0xFFFFFFFF
    clear_normal: tuple[float, float, float] = (0.0, 0.0, 0.0)
    clear_albedo: int = 0


@wp.struct
class MeshData:
    """Per-mesh auxiliary vertex data for texture mapping and smooth shading.

    Attributes:
        uvs: Per-vertex UV coordinates, shape ``[vertex_count, 2]``, dtype ``vec2f``.
        normals: Per-vertex normals for smooth shading, shape ``[vertex_count, 3]``, dtype ``vec3f``.
    """

    uvs: wp.array(dtype=wp.vec2f)
    normals: wp.array(dtype=wp.vec3f)


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
