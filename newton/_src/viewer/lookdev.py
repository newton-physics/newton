# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Minimal lookdev for the Newton GL viewer.

Newton's GL viewer offers an optional, recognizable look — not a customizable
studio. Lookdev is off by default (``lookdev=None``), leaving the viewer's
plain default rendering untouched. Two fixed modes can be enabled (``LIGHT``
and ``DARK``), each defining the sky gradient, three directional lights, ground
albedo, hemispherical ambient, and tone-map exposure. Nothing here carries a
world-space position, distance, or extent: the look renders correctly across
scene scales.

Example:
    >>> from newton.viewer import LookdevMode, ViewerGL
    >>> viewer = ViewerGL(lookdev=LookdevMode.DARK)
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

__all__ = ["LookdevMode"]

# Number of directional lights driven by every mode (key, fill, rim).
_LIGHT_COUNT = 3


# IntEnum matches Newton's other public enums (e.g. ``GeoType``) and lets the
# viewer UI use the value directly as a combo index.
class LookdevMode(IntEnum):
    """Built-in Newton GL viewer look.

    Two fixed presets are provided. Switch at runtime via
    :meth:`~newton.viewer.ViewerGL.set_lookdev` or by pressing
    ``L`` while the viewer window is focused.
    """

    LIGHT = 0
    """Bright studio look over a light backdrop."""

    DARK = 1
    """Dark studio look over a near-black backdrop."""


# Neutral pass-through used when lookdev is disabled (``lookdev=None``, the
# default). Plain white key/fill/rim so PBR shading reads cleanly without a
# stylized tint; the sky stays the renderer's ``background_color`` and both the
# analytical shadow-catcher and plane-hiding are disabled by the viewer. These
# values are deliberately un-tuned — this is "no look", not another preset.
_NEUTRAL: dict = {
    "lights": [
        # Lookdev-off matches the pre-lookdev viewer: a single sun (main's
        # per-up-axis direction), with fill/rim disabled so the scene is not lit
        # by the lookdev 3-point rig. The spotlight cones this sun's direct light.
        ((0.2, -0.3, 0.8), (1.0, 1.0, 1.0), 2.0),  # Single sun (key)
        ((0.5, -0.6, 0.4), (1.0, 1.0, 1.0), 0.0),  # Fill disabled
        ((0.0, 0.7, 0.4), (1.0, 1.0, 1.0), 0.0),  # Rim disabled
    ],
    "exposure": 1.0,
    # The pre-lookdev viewer's hemispherical ambient, unchanged.
    "ambient_sky": (0.8, 0.8, 0.85),
    "ambient_ground": (0.3, 0.3, 0.35),
    "ground_color": (0.5, 0.5, 0.5),
    "ground_roughness": 0.7,
}


# Per-look parameter set. Values are linear-RGB unless noted otherwise. Light
# directions point TOWARD the light source and are expressed in a Z-up
# canonical world; ``_lights_for_up_axis`` rotates them into the renderer's
# active up-axis at apply time. Each light tuple is
# ``(direction, color, intensity)``.
_PARAMS: dict[LookdevMode, dict] = {
    LookdevMode.LIGHT: {
        # ``sky_upper`` is the zenith colour, ``sky_lower`` the horizon/ground:
        # the GL sky shader mixes toward ``sky_upper`` at the top of the sphere.
        # Authored as the display-sRGB values shown in the viewer's colour
        # pickers: #C4C4C8 zenith over a #626264 horizon/ground.
        "sky_upper": (0.7686, 0.7686, 0.7843),
        "sky_lower": (0.3843, 0.3843, 0.3922),
        "lights": [
            ((0.4, -0.3, 0.8), (1.0, 1.0, 1.0), 3.0),  # Key (front-upper, camera-left)
            ((0.45, 0.6, 0.35), (0.8, 0.82, 0.88), 1.2),  # Fill (cool, front-right, low)
            ((-0.6, 0.25, 0.5), (1.0, 0.98, 0.92), 0.6),  # Rim (warm, behind-upper)
        ],
        "exposure": 1.2,
        # Bright, slightly cool bounce for the white studio.
        "ambient_sky": (0.85, 0.85, 0.88),
        "ambient_ground": (0.45, 0.45, 0.47),
        "ground_color": (0.6, 0.6, 0.62),
        "ground_roughness": 0.7,
    },
    LookdevMode.DARK: {
        # Signature dark backdrop, authored as the display-sRGB values shown in
        # the viewer's colour pickers: #1C1C1E zenith over a #353537
        # horizon/ground. After the shader's sRGB→linear, exposure (1.4), ACES and
        # sRGB encoding these render as #151517 and #39393C respectively (pinned
        # by ``test_dark_sky_tuning_matches_reference_backdrop``). The GL sky
        # shader mixes toward ``sky_upper`` at the top.
        "sky_upper": (0.1098, 0.1098, 0.1176),
        "sky_lower": (0.2078, 0.2078, 0.2157),
        "lights": [
            ((0.4, -0.3, 0.8), (1.0, 0.98, 0.95), 2.5),  # Key (warm, front-upper, camera-left)
            ((0.45, 0.6, 0.35), (0.6, 0.65, 0.75), 0.8),  # Fill (cool, front-right, low)
            ((-0.6, 0.25, 0.5), (1.0, 0.95, 0.85), 0.5),  # Rim (warm, soft, behind-upper)
        ],
        "exposure": 1.4,
        # Dim ambient so the key light does the modelling.
        "ambient_sky": (0.22, 0.22, 0.25),
        "ambient_ground": (0.08, 0.08, 0.10),
        "ground_color": (0.12, 0.12, 0.14),
        "ground_roughness": 0.5,
    },
}


def _params(mode: LookdevMode | None) -> dict:
    """Return the parameter dict for the given lookdev mode.

    Args:
        mode: The lookdev mode to look up, or ``None`` for the neutral
            (lookdev-disabled) pass-through.

    Returns:
        The mode's parameter dict (do not mutate).
    """
    if mode is None:
        return _NEUTRAL
    return _PARAMS[LookdevMode(mode)]


def _coerce_mode(mode: LookdevMode | int | None) -> LookdevMode | None:
    """Validate a user-supplied lookdev mode and coerce it to the enum.

    ``None`` selects the neutral pass-through (lookdev disabled) and is
    returned as-is. Rejects ``bool`` explicitly: it is an ``int`` subclass, so
    ``LookdevMode(True)`` would otherwise silently resolve to ``DARK``.

    Args:
        mode: The value to validate.

    Returns:
        The corresponding :class:`LookdevMode` member, or ``None``.
    """
    if mode is None:
        return None
    if isinstance(mode, bool) or not isinstance(mode, (int, LookdevMode)):
        raise TypeError("lookdev mode must be a LookdevMode, int, or None")
    return LookdevMode(mode)


def _rotate_z_up_to(up_axis: int, vec: tuple[float, float, float]) -> tuple[float, float, float]:
    """Rotate a Z-up vector into the renderer's active up-axis world frame.

    The lookdev light directions are authored once in Z-up (Newton's default).
    This helper preserves the relative geometry across the three supported
    axes so the look is identical regardless of scene convention or scale.
    """
    x, y, z = vec
    if up_axis == 2:  # Z-up: identity
        return (x, y, z)
    if up_axis == 1:  # Y-up: swap (z -> y, y -> -z)
        return (x, z, -y)
    if up_axis == 0:  # X-up: swap (z -> x, x -> -z)
        return (z, y, -x)
    return (x, y, z)


def _lights_for_up_axis(mode: LookdevMode | None, up_axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the per-frame light arrays for ``ShaderShape``.

    Args:
        mode: The active lookdev mode, or ``None`` for the neutral lights.
        up_axis: Active up-axis index (0=X, 1=Y, 2=Z).

    Returns:
        ``(light_dirs, light_colors)`` — both float32 arrays of shape
        ``(3, 3)``. ``light_colors`` is pre-multiplied by per-light intensity
        so the shader needs no separate intensity uniform.
    """
    lights = _params(mode)["lights"]
    if len(lights) != _LIGHT_COUNT:
        raise ValueError(f"lookdev mode {mode!r} must define {_LIGHT_COUNT} lights, got {len(lights)}")
    dirs = np.empty((_LIGHT_COUNT, 3), dtype=np.float32)
    colors = np.empty((_LIGHT_COUNT, 3), dtype=np.float32)
    for i, (direction, color, intensity) in enumerate(lights):
        d = _rotate_z_up_to(up_axis, direction)
        n = max((d[0] ** 2 + d[1] ** 2 + d[2] ** 2) ** 0.5, 1e-12)
        dirs[i] = (d[0] / n, d[1] / n, d[2] / n)
        colors[i] = (color[0] * intensity, color[1] * intensity, color[2] * intensity)
    return dirs, colors
