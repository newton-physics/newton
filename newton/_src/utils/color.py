# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
from collections.abc import Sequence

import numpy as np
import warp as wp


class ColorSpace(enum.IntEnum):
    """RGB color spaces used at Newton rendering boundaries."""

    LINEAR = 0
    """Linear-light RGB."""

    SRGB = 1
    """sRGB/display-encoded RGB."""


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
