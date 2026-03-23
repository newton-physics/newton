# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# Python
import dataclasses

Color3 = tuple[float, float, float]


class MeshColors:
    LIGHT = (0.8, 0.8, 0.8)
    GREY = (0.45, 0.45, 0.45)
    DARK = (0.2, 0.2, 0.2)
    BONE = (191 / 255, 190 / 255, 178 / 255)
    ORANGE = (206 / 255, 117 / 255, 52 / 255)
    RED = (200 / 255, 30 / 255, 30 / 255)
    BLUE = (88 / 255, 135 / 255, 171 / 255)
    BLUEGREY = (100 / 255, 100 / 255, 130 / 255)
    SAGEGREY = (145 / 255, 157 / 255, 132 / 255)
    GREEN = (120 / 255, 183.6 / 255, 48 / 255)
    YELLOW = (183 / 255, 146 / 255, 76 / 255)
    SOFTGREEN = (150 / 255, 200 / 255, 150 / 255)
    LIGHTGREEN = (100 / 255, 220 / 255, 100 / 255)
    SOFTPINK = (200 / 255, 182 / 255, 203 / 255)
    PINK = (200 / 255, 150 / 255, 200 / 255)


@dataclasses.dataclass
class ViewerConfig:
    """Viewer appearance settings for :class:`RigidBodySim`.

    All ``None`` defaults leave the standard Newton viewer appearance unchanged.
    """

    robot_color: Color3 | None = None
    """Override color for all robot shapes.  ``None`` keeps USD material colors."""

    diffuse_scale: float | None = None
    """Diffuse light scale (multiplied on top of the base ``* 3.0``).  ``None`` keeps default (1.0)."""

    specular_scale: float | None = None
    """Specular highlight scale.  ``None`` keeps default (1.0)."""

    shadow_radius: float | None = None
    """PCF shadow softness radius.  Larger = softer edges.  ``None`` keeps default (3.0)."""

    shadow_extents: float | None = None
    """Shadow map half-size in world units.  ``None`` keeps default (10.0)."""

    spotlight_enabled: bool | None = None
    """Use cone spotlight (True) or uniform directional light (False).  ``None`` keeps default (True)."""

    background_brightness_scale: float | None = None
    """Scale factor for ground color and ground plane shape brightness.  ``None`` keeps default (1.0)."""

    sky_color: Color3 | None = None
    """Override sky color (ambient + background upper).  ``None`` keeps default blue."""

    light_color: Color3 | None = None
    """Override directional (sun) light color.  ``None`` keeps default white ``(1, 1, 1)``."""

    render_width: int = 1920
    """Viewer / recording width in pixels."""

    render_height: int = 1080
    """Viewer / recording height in pixels."""
