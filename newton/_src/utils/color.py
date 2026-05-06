# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
from collections.abc import Sequence
from typing import Literal

import numpy as np
import warp as wp


class ColorSpace(enum.IntEnum):
    """RGB color spaces used at Newton rendering boundaries."""

    LINEAR = 0
    """Linear-light RGB."""

    SRGB = 1
    """sRGB/display-encoded RGB."""


TEXTURE_COLOR_SPACE_AUTO = "auto"


def _to_rgb_array(color: Sequence[float] | np.ndarray) -> np.ndarray:
    rgb = np.asarray(color, dtype=np.float32).reshape(-1)
    if rgb.size < 3:
        raise ValueError("RGB colors require at least three components.")
    return rgb[:3]


def color_srgb_to_linear(color: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    """Convert an sRGB/display RGB triple to linear Rec.709.

    Args:
        color: RGB values in sRGB/display encoding. Negative components are
            clamped to zero before conversion.

    Returns:
        Linear RGB triple.
    """

    rgb = np.clip(_to_rgb_array(color), 0.0, None)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, np.power((rgb + 0.055) / 1.055, 2.4))
    return (float(linear[0]), float(linear[1]), float(linear[2]))


def color_linear_to_srgb(color: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    """Convert a linear RGB triple to sRGB/display encoding.

    Args:
        color: Linear RGB values. Negative components are clamped to zero
            before conversion.

    Returns:
        sRGB/display-encoded RGB triple.
    """

    rgb = np.clip(_to_rgb_array(color), 0.0, None)
    srgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055)
    return (float(srgb[0]), float(srgb[1]), float(srgb[2]))


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


def normalize_color_space(
    color_space: ColorSpace | str | int | None,
    *,
    default: ColorSpace = ColorSpace.SRGB,
) -> ColorSpace:
    """Normalize color-space metadata to :class:`ColorSpace`.

    Args:
        color_space: Color-space enum, string token, integer value, or ``None``.
        default: Color space to use for ``None``, ``"auto"``, or unknown
            tokens.

    Returns:
        Normalized color-space enum.
    """

    if color_space is None:
        return default
    if isinstance(color_space, ColorSpace):
        return color_space

    token = str(color_space).strip().lower()
    if token in ("", TEXTURE_COLOR_SPACE_AUTO, "unknown"):
        return default
    if token in ("identity", "raw", "data", "linear", "lin_rec709_scene") or token.startswith("lin_"):
        return ColorSpace.LINEAR
    if token in ("display", "srgb", "srgb_rec709_scene", "g22_rec709_scene") or token.startswith("srgb_"):
        return ColorSpace.SRGB

    try:
        return ColorSpace(color_space)
    except (TypeError, ValueError):
        return default


def normalize_texture_color_space(color_space: ColorSpace | str | int | None) -> ColorSpace | Literal["auto"]:
    """Normalize texture color-space metadata to :class:`ColorSpace` or ``"auto"``."""

    if color_space is None:
        return TEXTURE_COLOR_SPACE_AUTO
    if isinstance(color_space, ColorSpace):
        return color_space

    token = str(color_space).strip().lower()
    if token in ("", TEXTURE_COLOR_SPACE_AUTO, "unknown"):
        return TEXTURE_COLOR_SPACE_AUTO
    try:
        enum_value = ColorSpace(color_space)
    except (TypeError, ValueError):
        enum_value = None
    if enum_value is not None:
        return enum_value
    if token in ("identity", "raw", "data", "linear", "lin_rec709_scene") or token.startswith("lin_"):
        return ColorSpace.LINEAR
    if token in ("display", "srgb", "srgb_rec709_scene", "g22_rec709_scene") or token.startswith("srgb_"):
        return ColorSpace.SRGB
    return TEXTURE_COLOR_SPACE_AUTO


def texture_color_space_to_color_space(color_space: ColorSpace | str | int | None) -> ColorSpace:
    """Map texture color-space metadata to a shading color space."""

    normalized = normalize_texture_color_space(color_space)
    return ColorSpace.SRGB if normalized == TEXTURE_COLOR_SPACE_AUTO else normalized


def texture_color_space_to_id(color_space: ColorSpace | str | int | None) -> int:
    """Map normalized texture color-space metadata to the raytracer enum."""

    return int(texture_color_space_to_color_space(color_space))


@wp.func
def srgb_channel_to_linear_wp(value: float) -> float:
    clamped = wp.max(value, 0.0)
    if clamped <= 0.04045:
        return clamped / 12.92
    return wp.pow((clamped + 0.055) / 1.055, 2.4)


@wp.func
def linear_channel_to_srgb_wp(value: float) -> float:
    clamped = wp.max(value, 0.0)
    if clamped <= 0.0031308:
        return clamped * 12.92
    return 1.055 * wp.pow(clamped, 1.0 / 2.4) - 0.055


@wp.func
def srgb_to_linear_wp(rgb: wp.vec3f) -> wp.vec3f:
    return wp.vec3f(
        srgb_channel_to_linear_wp(rgb[0]),
        srgb_channel_to_linear_wp(rgb[1]),
        srgb_channel_to_linear_wp(rgb[2]),
    )


@wp.func
def linear_to_srgb_wp(rgb: wp.vec3f) -> wp.vec3f:
    return wp.vec3f(
        linear_channel_to_srgb_wp(rgb[0]),
        linear_channel_to_srgb_wp(rgb[1]),
        linear_channel_to_srgb_wp(rgb[2]),
    )
