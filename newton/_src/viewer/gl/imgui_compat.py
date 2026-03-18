# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compatibility helpers for imgui_bundle API drift."""

from __future__ import annotations

from typing import Any


def _color_components(color: Any) -> tuple[float, float, float, float]:
    """Normalize RGB/RGBA tuple-like or ImVec-like values to four floats."""
    if all(hasattr(color, attr) for attr in ("x", "y", "z")):
        return (
            float(color.x),
            float(color.y),
            float(color.z),
            float(getattr(color, "w", 1.0)),
        )

    try:
        values = tuple(color)
    except TypeError as exc:
        raise TypeError(f"Unsupported color value {color!r}") from exc

    if len(values) == 3:
        return (float(values[0]), float(values[1]), float(values[2]), 1.0)
    if len(values) == 4:
        return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
    raise ValueError(f"Expected 3 or 4 color components, got {len(values)} from {color!r}")


def to_imgui_color4(imgui: Any, color: Any):
    """Build an ``ImVec4`` from a tuple-like or ImVec-like color."""
    return imgui.ImVec4(*_color_components(color))


def to_rgb_tuple(color: Any) -> tuple[float, float, float]:
    """Convert an imgui color or tuple-like value back to an RGB tuple."""
    r, g, b, _a = _color_components(color)
    return (r, g, b)


def color_edit3_tuple(imgui: Any, label: str, color: Any, *, flags: int = 0) -> tuple[bool, tuple[float, float, float]]:
    """Run ``color_edit3`` while storing renderer colors as plain RGB tuples."""
    changed, edited = imgui.color_edit3(label, to_imgui_color4(imgui, color), flags)
    return changed, to_rgb_tuple(edited)
