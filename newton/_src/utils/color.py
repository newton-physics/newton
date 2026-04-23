# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import warp as wp

TEXTURE_COLOR_SPACE_AUTO = "auto"
TEXTURE_COLOR_SPACE_RAW = "raw"
TEXTURE_COLOR_SPACE_SRGB = "srgb"

TEXTURE_COLOR_SPACE_RAW_ID = 0
TEXTURE_COLOR_SPACE_SRGB_ID = 1


def _to_rgb_array(color: Sequence[float] | np.ndarray) -> np.ndarray:
    rgb = np.asarray(color, dtype=np.float32).reshape(-1)
    if rgb.size < 3:
        raise ValueError("RGB colors require at least three components.")
    return rgb[:3]


def color_srgb_to_linear(color: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    """Convert an sRGB/display RGB triple to linear Rec.709."""

    rgb = np.clip(_to_rgb_array(color), 0.0, None)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, np.power((rgb + 0.055) / 1.055, 2.4))
    return (float(linear[0]), float(linear[1]), float(linear[2]))


def color_linear_to_srgb(color: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    """Convert a linear RGB triple to sRGB/display encoding."""

    rgb = np.clip(_to_rgb_array(color), 0.0, None)
    srgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055)
    return (float(srgb[0]), float(srgb[1]), float(srgb[2]))


def srgb_to_linear_rgb(color: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    """.. deprecated:: 1.1 Use ``color_srgb_to_linear()`` instead."""

    warnings.warn(
        "`newton.utils.srgb_to_linear_rgb()` is deprecated and will be removed in a future release. "
        "Use `newton.utils.color_srgb_to_linear()` instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return color_srgb_to_linear(color)


def linear_to_srgb_rgb(color: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    """.. deprecated:: 1.1 Use ``color_linear_to_srgb()`` instead."""

    warnings.warn(
        "`newton.utils.linear_to_srgb_rgb()` is deprecated and will be removed in a future release. "
        "Use `newton.utils.color_linear_to_srgb()` instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return color_linear_to_srgb(color)


def linear_rgb_to_srgb_uint8(color: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert linear RGB floats to uint8 sRGB bytes."""

    srgb = np.asarray(color_linear_to_srgb(color), dtype=np.float32)
    return np.clip(np.round(srgb * 255.0), 0.0, 255.0).astype(np.uint8)


def srgb_rgb_to_uint8(color: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert sRGB/display RGB floats to uint8 bytes without re-encoding."""

    rgb = np.clip(_to_rgb_array(color), 0.0, 1.0)
    return np.clip(np.round(rgb * 255.0), 0.0, 255.0).astype(np.uint8)


def linear_image_to_srgb_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a linear RGB/RGBA array to uint8 sRGB."""

    img = np.asarray(image, dtype=np.float32)
    if img.ndim < 2 or img.shape[-1] not in (3, 4):
        raise ValueError("Expected an array with RGB or RGBA channels on the last axis.")

    out = img.copy()
    rgb = np.clip(out[..., :3], 0.0, None)
    out[..., :3] = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055)
    return np.clip(np.round(out * 255.0), 0.0, 255.0).astype(np.uint8)


def srgb_image_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert sRGB/display RGB/RGBA floats to uint8 bytes without re-encoding."""

    img = np.asarray(image, dtype=np.float32)
    if img.ndim < 2 or img.shape[-1] not in (3, 4):
        raise ValueError("Expected an array with RGB or RGBA channels on the last axis.")

    return np.clip(np.round(np.clip(img, 0.0, 1.0) * 255.0), 0.0, 255.0).astype(np.uint8)


def normalize_texture_color_space(color_space: str | None) -> str:
    """Normalize texture color-space metadata to ``raw``, ``srgb``, or ``auto``."""

    if color_space is None:
        return TEXTURE_COLOR_SPACE_AUTO

    token = str(color_space).strip().lower()
    if token in ("", TEXTURE_COLOR_SPACE_AUTO, "unknown"):
        return TEXTURE_COLOR_SPACE_AUTO
    if token in ("identity", TEXTURE_COLOR_SPACE_RAW, "data", "lin_rec709_scene") or token.startswith("lin_"):
        return TEXTURE_COLOR_SPACE_RAW
    if token in (TEXTURE_COLOR_SPACE_SRGB, "srgb_rec709_scene", "g22_rec709_scene") or token.startswith("srgb_"):
        return TEXTURE_COLOR_SPACE_SRGB
    return TEXTURE_COLOR_SPACE_AUTO


def texture_color_space_to_id(color_space: str | None) -> int:
    """Map normalized texture color-space metadata to the raytracer enum."""

    return (
        TEXTURE_COLOR_SPACE_RAW_ID
        if normalize_texture_color_space(color_space) == TEXTURE_COLOR_SPACE_RAW
        else TEXTURE_COLOR_SPACE_SRGB_ID
    )


@wp.func
def srgb_channel_to_linear_wp(value: float):
    clamped = wp.max(value, 0.0)
    if clamped <= 0.04045:
        return clamped / 12.92
    return wp.pow((clamped + 0.055) / 1.055, 2.4)


@wp.func
def linear_channel_to_srgb_wp(value: float):
    clamped = wp.max(value, 0.0)
    if clamped <= 0.0031308:
        return clamped * 12.92
    return 1.055 * wp.pow(clamped, 1.0 / 2.4) - 0.055


@wp.func
def srgb_to_linear_wp(rgb: wp.vec3f):
    return wp.vec3f(
        srgb_channel_to_linear_wp(rgb[0]),
        srgb_channel_to_linear_wp(rgb[1]),
        srgb_channel_to_linear_wp(rgb[2]),
    )


@wp.func
def linear_to_srgb_wp(rgb: wp.vec3f):
    return wp.vec3f(
        linear_channel_to_srgb_wp(rgb[0]),
        linear_channel_to_srgb_wp(rgb[1]),
        linear_channel_to_srgb_wp(rgb[2]),
    )
