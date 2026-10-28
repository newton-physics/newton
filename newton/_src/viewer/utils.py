# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import numpy as np

from ..geometry.types import SH_C0
from ..utils.color import ColorSpace
from ..utils.texture import _linear_to_srgb, load_texture, normalize_texture

OPAQUE_OPACITY_THRESHOLD = 0.999


def to_numpy(x: Any) -> np.ndarray | None:
    """Convert Warp arrays or other array-like inputs to NumPy arrays."""
    if x is None:
        return None
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def prepare_viewer_texture(texture: np.ndarray | str | None) -> np.ndarray | None:
    """Load and normalize texture data for web and recording viewers."""
    return normalize_texture(
        load_texture(texture),
        flip_vertical=False,
        require_channels=True,
        scale_unit_range=True,
    )


def promote_to_clamped_float_array(
    values: Any,
    num_items: int,
    *,
    min_value: float = 0.0,
    max_value: float = 1.0,
    value_name: str = "Values",
) -> np.ndarray | None:
    """Promote scalar or per-item inputs to a clamped NumPy float array."""
    if values is None:
        return None
    if hasattr(values, "numpy"):
        values = values.numpy()
    elif np.isscalar(values):
        values = np.full(num_items, float(values), dtype=np.float32)
    values_np = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values_np) == 1 and num_items > 1:
        values_np = np.full(num_items, float(values_np[0]), dtype=np.float32)
    if len(values_np) != num_items:
        raise ValueError(
            f"{value_name} arrays must contain one value or exactly {num_items} values, got {len(values_np)}."
        )
    return np.clip(values_np, min_value, max_value)


def gaussian_display_colors(sh_coeffs: np.ndarray, color_space: ColorSpace) -> np.ndarray:
    """Convert the DC term of Gaussian SH coefficients to display-encoded RGB in [0, 1].

    Args:
        sh_coeffs: SH coefficients with at least three columns per Gaussian.
        color_space: Color space of the SH radiance. Linear radiance is
            encoded to sRGB for display.

    Returns:
        Float32 array of shape ``(N, 3)``.
    """
    rgb = np.clip(SH_C0 * np.asarray(sh_coeffs)[:, :3] + 0.5, 0.0, None)
    if color_space == ColorSpace.LINEAR:
        rgb = _linear_to_srgb(rgb)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)
