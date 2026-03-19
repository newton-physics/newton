# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import enum

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
