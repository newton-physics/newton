# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Vector sources for the generated GUI icon atlas and uncommon-size fallback.

After changing these definitions, run ``uv run --extra dev
scripts/generate_viewer_icons.py`` to regenerate ``icons.png``.
"""

from __future__ import annotations

from math import cos, pi, sin
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

    from imgui_bundle import imgui
    from numpy.typing import ArrayLike

ICON_NAMES = (
    "articulation",
    "body",
    "box",
    "bvh",
    "camera",
    "capsule",
    "clear",
    "close",
    "cloth",
    "collision",
    "cone",
    "console",
    "contact",
    "convex",
    "copy",
    "custom",
    "cylinder",
    "deselect",
    "documents",
    "dynamic",
    "ellipsoid",
    "empty",
    "eye",
    "eye_off",
    "first",
    "focus",
    "folder",
    "follow",
    "gaussian",
    "grid",
    "hydro",
    "inspector",
    "joint",
    "joint_ball",
    "joint_cable",
    "joint_d6",
    "joint_distance",
    "joint_fixed",
    "joint_free",
    "joint_prismatic",
    "joint_revolute",
    "kinematic",
    "last",
    "link",
    "lock",
    "mesh",
    "particle",
    "particles",
    "pause",
    "plane",
    "play",
    "previous",
    "reset",
    "sdf",
    "shape",
    "site",
    "sliders",
    "sphere",
    "surface",
    "terrain",
    "tet",
    "texture",
    "tree",
    "tri",
    "undo",
    "volume",
    "world",
    "xray",
)

# Logical sizes preserve the original stroke-to-size ratio for small locks,
# regular controls/tree rows, and the larger transport button.
ICON_SPRITES = (
    *((name, 16, solid) for solid in (False, True) for name in ICON_NAMES),
    ("lock", 8, False),
    ("lock", 11, False),
    ("play", 18, False),
    ("pause", 18, False),
)
ICON_PIXELS = 96
ICON_TILE = 128
ICON_COLUMNS = 8


def draw_icon(
    ui: ModuleType,
    draw: imgui.ImDrawList,
    name: str,
    position: imgui.ImVec2 | tuple[float, float],
    size: float,
    *,
    scale: float,
    rgba: ArrayLike,
    solid: bool = False,
) -> None:
    """Emit the original vector primitives for baking or unsupported atlas sizes."""
    x, y = position
    color = ui.get_color_u32(ui.ImVec4(*rgba))
    width = max(1, scale * (1.5 if solid else 1.15))

    def point(a: float, b: float) -> imgui.ImVec2:
        return ui.ImVec2(round(x + a * size) + 0.5, round(y + b * size) + 0.5)

    def lines(points: ArrayLike, closed: bool = False) -> None:
        for a, b in zip(points, (*points[1:], points[0]) if closed else points[1:], strict=False):
            draw.add_line(point(*a), point(*b), color, width)

    def circle(a: float, b: float, radius: float, fill: bool = False) -> None:
        if fill:
            draw.add_circle_filled(point(a, b), radius * size, color, 16)
        else:
            draw.add_circle(point(a, b), radius * size, color, 16, width)

    def shade(factor: float) -> int:
        return ui.get_color_u32(ui.ImVec4(*(channel * factor for channel in rgba[:3]), rgba[3]))

    def polygon(points: ArrayLike, factor: float = 1) -> None:
        draw.add_convex_poly_filled([point(*p) for p in points], shade(factor))

    if solid and name == "folder":
        draw.add_rect_filled(point(0.10, 0.16), point(0.90, 0.32), color, size * 0.04)
        draw.add_rect_filled(point(0.17, 0.40), point(0.83, 0.87), color, size * 0.05)
        draw.add_rect_filled(point(0.39, 0.50), point(0.61, 0.61), shade(0.35), size * 0.025)
    elif solid and name in ("body", "box", "shape"):
        polygon(((0.14, 0.28), (0.5, 0.08), (0.86, 0.28), (0.86, 0.72), (0.5, 0.92), (0.14, 0.72)))
        polygon(((0.14, 0.28), (0.5, 0.48), (0.5, 0.92), (0.14, 0.72)), 0.80)
        polygon(((0.5, 0.48), (0.86, 0.28), (0.86, 0.72), (0.5, 0.92)), 0.60)
    elif solid and name in ("mesh", "tri"):
        vertices = ((0.5, 0.15), (0.12, 0.82), (0.88, 0.82))
        lines(vertices, True)
        if name == "mesh":
            for a, b in vertices:
                draw.add_rect_filled(point(a - 0.09, b - 0.09), point(a + 0.09, b + 0.09), color)
    elif solid and name in ("sphere", "ellipsoid"):
        if name == "sphere":
            circle(0.5, 0.5, 0.40, True)
            draw.add_circle_filled(point(0.38, 0.37), size * 0.10, shade(0.60), 12)
        else:
            polygon(tuple((0.5 + 0.43 * cos(t), 0.5 + 0.28 * sin(t)) for t in np.linspace(0, 2 * pi, 16)))
    elif solid and name == "capsule":
        draw.add_rect_filled(point(0.28, 0.10), point(0.72, 0.90), color, size * 0.22)
    elif solid and name == "cylinder":
        polygon(((0.23, 0.23), (0.77, 0.23), (0.77, 0.77), (0.64, 0.88), (0.36, 0.88), (0.23, 0.77)), 0.78)
        polygon(tuple((0.5 + 0.27 * cos(t), 0.23 + 0.11 * sin(t)) for t in np.linspace(0, 2 * pi, 16)))
    elif solid and name == "cone":
        polygon(((0.5, 0.10), (0.90, 0.80), (0.10, 0.80)))
        polygon(((0.5, 0.10), (0.90, 0.80), (0.5, 0.80)), 0.65)
    elif solid and name in ("surface", "cloth"):
        polygon(((0.12, 0.15), (0.72, 0.15), (0.88, 0.83), (0.28, 0.72)), 0.65)
        lines(((0.12, 0.15), (0.72, 0.15), (0.88, 0.83), (0.28, 0.72)), True)
        lines(((0.42, 0.15), (0.58, 0.77)))
        lines(((0.20, 0.45), (0.80, 0.49)))
    elif solid and name in ("volume", "tet"):
        polygon(((0.5, 0.08), (0.90, 0.84), (0.08, 0.77)))
        polygon(((0.5, 0.08), (0.90, 0.84), (0.5, 0.60)), 0.65)
        polygon(((0.08, 0.77), (0.5, 0.60), (0.90, 0.84)), 0.80)
    elif solid and name == "joint_fixed":
        draw.add_rect_filled(point(0.12, 0.20), point(0.38, 0.80), color, size * 0.04)
        draw.add_rect_filled(point(0.62, 0.20), point(0.88, 0.80), color, size * 0.04)
        for b in (0.35, 0.65):
            lines(((0.38, b), (0.62, b)))
    elif name in ("body", "box", "shape"):
        lines(((0.16, 0.28), (0.50, 0.10), (0.85, 0.28), (0.85, 0.72), (0.50, 0.91), (0.16, 0.72)), True)
        lines(((0.16, 0.28), (0.50, 0.47), (0.85, 0.28)))
        lines(((0.50, 0.47), (0.50, 0.91)))
    elif name == "mesh":
        lines(((0.10, 0.77), (0.36, 0.12), (0.87, 0.34), (0.78, 0.89)), True)
        lines(((0.10, 0.77), (0.87, 0.34), (0.36, 0.12), (0.78, 0.89), (0.87, 0.34)))
    elif name == "convex":
        lines(((0.10, 0.40), (0.39, 0.08), (0.81, 0.19), (0.91, 0.65), (0.58, 0.92), (0.18, 0.79)), True)
        lines(((0.10, 0.40), (0.54, 0.42), (0.81, 0.19)))
        lines(((0.54, 0.42), (0.58, 0.92)))
    elif name == "ellipsoid":
        lines(tuple((0.5 + 0.44 * cos(t), 0.5 + 0.26 * sin(t)) for t in np.linspace(0, 2 * pi, 20)), True)
        lines(tuple((0.5 + 0.16 * cos(t), 0.5 + 0.26 * sin(t)) for t in np.linspace(0, 2 * pi, 16)), True)
    elif name == "cone":
        lines(((0.12, 0.76), (0.50, 0.10), (0.88, 0.76)))
        lines(tuple((0.5 + 0.38 * cos(t), 0.76 + 0.12 * sin(t)) for t in np.linspace(0, 2 * pi, 16)), True)
    elif name == "terrain":
        lines(((0.08, 0.72), (0.3, 0.32), (0.46, 0.52), (0.64, 0.16), (0.92, 0.72), (0.56, 0.91)), True)
        lines(((0.3, 0.32), (0.56, 0.91), (0.64, 0.16)))
    elif name == "gaussian":
        for radius in (0.12, 0.26, 0.40):
            circle(0.5, 0.5, radius, fill=radius == 0.12)
    elif name == "empty":
        circle(0.5, 0.5, 0.37)
        lines(((0.24, 0.76), (0.76, 0.24)))
    elif name in ("sphere", "world"):
        circle(0.5, 0.5, 0.39)
        lines(((0.1, 0.5), (0.9, 0.5)))
        lines(((0.5, 0.11), (0.34, 0.35), (0.34, 0.65), (0.5, 0.89), (0.66, 0.65), (0.66, 0.35)), True)
    elif name in ("capsule", "cylinder"):
        lines(
            (
                (0.27, 0.25),
                (0.27, 0.75),
                (0.4, 0.88),
                (0.6, 0.88),
                (0.73, 0.75),
                (0.73, 0.25),
                (0.6, 0.12),
                (0.4, 0.12),
            ),
            True,
        )
        if name == "cylinder":
            lines(((0.27, 0.25), (0.5, 0.36), (0.73, 0.25)))
    elif name in ("articulation", "tree"):
        lines(((0.5, 0.19), (0.5, 0.5), (0.2, 0.5), (0.2, 0.78)))
        lines(((0.5, 0.5), (0.8, 0.5), (0.8, 0.78)))
        for a, b in ((0.5, 0.15), (0.2, 0.82), (0.8, 0.82)):
            circle(a, b, 0.13 if solid else 0.12, solid)
    elif name == "custom":
        lines(((0.18, 0.14), (0.82, 0.14), (0.82, 0.86), (0.18, 0.86)), True)
        for b, a in ((0.32, 0.43), (0.53, 0.63), (0.74, 0.38)):
            lines(((0.29, b), (0.71, b)))
            circle(a, b, 0.065, True)
    elif name in ("joint", "link"):
        circle(0.27, 0.7, 0.19)
        circle(0.73, 0.3, 0.19)
        lines(((0.34, 0.55), (0.59, 0.34)))
        lines(((0.41, 0.66), (0.66, 0.45)))
    elif name == "joint_revolute":
        circle(0.5, 0.5, 0.12, True)
        lines(tuple((0.5 + 0.36 * cos(t), 0.5 + 0.36 * sin(t)) for t in np.linspace(-2.8, 2.1, 14)))
        lines(((0.12, 0.12), (0.15, 0.39), (0.4, 0.30)))
    elif name == "joint_prismatic":
        lines(((0.08, 0.27), (0.92, 0.27)))
        lines(((0.08, 0.73), (0.92, 0.73)))
        lines(((0.35, 0.18), (0.65, 0.18), (0.65, 0.82), (0.35, 0.82)), True)
        lines(((0.08, 0.50), (0.25, 0.50)))
        lines(((0.79, 0.37), (0.92, 0.50), (0.79, 0.63)))
    elif name == "joint_ball":
        circle(0.5, 0.39, 0.23, solid)
        lines(((0.15, 0.32), (0.15, 0.59), (0.31, 0.74), (0.69, 0.74), (0.85, 0.59), (0.85, 0.32)))
        lines(((0.5, 0.74), (0.5, 0.94)))
    elif name == "joint_fixed":
        lines(((0.14, 0.20), (0.38, 0.20), (0.38, 0.80), (0.14, 0.80)), True)
        lines(((0.62, 0.20), (0.86, 0.20), (0.86, 0.80), (0.62, 0.80)), True)
        for b in (0.35, 0.65):
            lines(((0.38, b), (0.62, b)))
    elif name in ("joint_free", "joint_d6"):
        circle(0.5, 0.5, 0.12, name == "joint_free")
        for dx, dy in ((0.38, 0), (-0.38, 0), (0, 0.38), (0, -0.38)):
            lines(((0.5 + dx * 0.35, 0.5 + dy * 0.35), (0.5 + dx, 0.5 + dy)))
            lines(
                (
                    (0.5 + dx * 0.7 - dy * 0.25, 0.5 + dy * 0.7 + dx * 0.25),
                    (0.5 + dx, 0.5 + dy),
                    (0.5 + dx * 0.7 + dy * 0.25, 0.5 + dy * 0.7 - dx * 0.25),
                )
            )
        if name == "joint_d6":
            lines(((0.2, 0.2), (0.8, 0.8)))
            lines(((0.8, 0.2), (0.2, 0.8)))
    elif name == "joint_distance":
        circle(0.12, 0.5, 0.09)
        circle(0.88, 0.5, 0.09)
        lines(((0.21, 0.5), (0.79, 0.5)))
        for a in (0.3, 0.5, 0.7):
            lines(((a, 0.35), (a, 0.65)))
    elif name == "joint_cable":
        lines(((0.10, 0.28), (0.25, 0.68), (0.5, 0.82), (0.75, 0.68), (0.9, 0.28)))
        circle(0.10, 0.20, 0.08, True)
        circle(0.90, 0.20, 0.08, True)
    elif name == "bvh":
        lines(((0.34, 0.08), (0.66, 0.08), (0.66, 0.32), (0.34, 0.32)), True)
        lines(((0.5, 0.32), (0.5, 0.5), (0.18, 0.5), (0.18, 0.65)))
        lines(((0.5, 0.5), (0.82, 0.5), (0.82, 0.65)))
        for dx in (0.04, 0.68):
            lines(((dx, 0.65), (dx + 0.28, 0.65), (dx + 0.28, 0.9), (dx, 0.9)), True)
    elif name == "sdf":
        lines(((0.12, 0.25), (0.5, 0.08), (0.88, 0.25), (0.88, 0.75), (0.5, 0.92), (0.12, 0.75)), True)
        lines(((0.12, 0.25), (0.5, 0.43), (0.88, 0.25)))
        lines(((0.5, 0.43), (0.5, 0.92)))
        lines(((0.23, 0.42), (0.23, 0.64), (0.39, 0.72)))
        lines(((0.62, 0.55), (0.77, 0.48), (0.77, 0.67), (0.62, 0.74)))
    elif name == "texture":
        lines(((0.10, 0.10), (0.90, 0.10), (0.90, 0.90), (0.10, 0.90)), True)
        for dx, dy in ((0.10, 0.10), (0.5, 0.5)):
            draw.add_rect_filled(point(dx, dy), point(dx + 0.4, dy + 0.4), color)
    elif name == "follow":
        circle(0.5, 0.5, 0.23)
        circle(0.5, 0.5, 0.055, True)
        for a, b in (
            ((0.5, 0.04), (0.5, 0.23)),
            ((0.5, 0.77), (0.5, 0.96)),
            ((0.04, 0.5), (0.23, 0.5)),
            ((0.77, 0.5), (0.96, 0.5)),
        ):
            lines((a, b))
    elif name == "camera":
        lines(((0.10, 0.3), (0.64, 0.3), (0.64, 0.8), (0.10, 0.8)), True)
        lines(((0.64, 0.45), (0.9, 0.28), (0.9, 0.82), (0.64, 0.65)))
        circle(0.28, 0.22, 0.12)
        circle(0.52, 0.22, 0.12)
    elif name == "xray":
        lines(((0.12, 0.12), (0.88, 0.12), (0.88, 0.88), (0.12, 0.88)), True)
        circle(0.5, 0.5, 0.22)
        lines(((0.23, 0.77), (0.77, 0.23)))
    elif name == "folder":
        lines(((0.10, 0.25), (0.40, 0.25), (0.48, 0.36), (0.9, 0.36), (0.9, 0.82), (0.10, 0.82)), True)
    elif name in ("eye", "eye_off"):
        lines(
            (
                (0.04, 0.50),
                (0.25, 0.28),
                (0.5, 0.21),
                (0.75, 0.28),
                (0.96, 0.5),
                (0.75, 0.72),
                (0.5, 0.79),
                (0.25, 0.72),
            ),
            True,
        )
        circle(0.5, 0.5, 0.15)
        if name == "eye_off":
            lines(((0.10, 0.10), (0.9, 0.9)))
    elif name == "lock":
        lines(((0.22, 0.46), (0.78, 0.46), (0.78, 0.9), (0.22, 0.9)), True)
        lines(((0.32, 0.46), (0.32, 0.24), (0.42, 0.13), (0.58, 0.13), (0.68, 0.24), (0.68, 0.46)))
        circle(0.5, 0.66, 0.04, True)
    elif name == "dynamic":
        lines(((0.62, 0.08), (0.22, 0.56), (0.48, 0.56), (0.38, 0.92), (0.82, 0.41), (0.56, 0.41)), True)
    elif name == "kinematic":
        circle(0.5, 0.2, 0.12)
        lines(((0.5, 0.32), (0.5, 0.86)))
        lines(((0.22, 0.52), (0.78, 0.52)))
        lines(((0.16, 0.68), (0.3, 0.85), (0.5, 0.9), (0.7, 0.85), (0.84, 0.68)))
    elif name == "collision":
        lines(((0.08, 0.35), (0.40, 0.35), (0.40, 0.75), (0.08, 0.75)), True)
        circle(0.73, 0.55, 0.22)
        lines(((0.44, 0.12), (0.5, 0.29), (0.62, 0.16)))
    elif name == "contact":
        lines(((0.08, 0.80), (0.92, 0.80)))
        circle(0.5, 0.65, 0.13, True)
        lines(((0.5, 0.48), (0.5, 0.10)))
        lines(((0.32, 0.28), (0.5, 0.10), (0.68, 0.28)))
    elif name == "particles":
        for a, b in ((0.22, 0.26), (0.73, 0.26), (0.48, 0.53), (0.22, 0.8), (0.75, 0.8)):
            circle(a, b, 0.085, True)
    elif name in ("site", "focus"):
        for points in (
            ((0.12, 0.35), (0.12, 0.12), (0.35, 0.12)),
            ((0.65, 0.12), (0.88, 0.12), (0.88, 0.35)),
            ((0.88, 0.65), (0.88, 0.88), (0.65, 0.88)),
            ((0.35, 0.88), (0.12, 0.88), (0.12, 0.65)),
        ):
            lines(points)
        circle(0.5, 0.5, 0.12)
    elif name == "hydro":
        for row in (0.25, 0.5, 0.75):
            lines(tuple((i / 8, row + 0.07 * sin(i * pi / 2)) for i in range(1, 8)))
    elif name in ("plane", "grid"):
        lines(((0.1, 0.68), (0.43, 0.25), (0.91, 0.35), (0.58, 0.79)), True)
        lines(((0.27, 0.46), (0.75, 0.57)))
        lines(((0.26, 0.72), (0.60, 0.29)))
    elif name == "documents":
        lines(((0.08, 0.18), (0.92, 0.18), (0.92, 0.84), (0.08, 0.84)), True)
        lines(((0.08, 0.35), (0.92, 0.35)))
        lines(((0.43, 0.18), (0.43, 0.35)))
        lines(((0.64, 0.35), (0.64, 0.84)))
    elif name == "console":
        lines(((0.08, 0.18), (0.92, 0.18), (0.92, 0.84), (0.08, 0.84)), True)
        lines(((0.22, 0.36), (0.39, 0.51), (0.22, 0.66)))
        lines(((0.51, 0.67), (0.77, 0.67)))
    elif name == "copy":
        lines(((0.32, 0.28), (0.83, 0.28), (0.83, 0.90), (0.32, 0.90)), True)
        lines(((0.64, 0.28), (0.64, 0.10), (0.13, 0.10), (0.13, 0.71), (0.32, 0.71)))
    elif name == "clear":
        lines(((0.18, 0.28), (0.82, 0.28)))
        lines(((0.38, 0.28), (0.38, 0.12), (0.62, 0.12), (0.62, 0.28)))
        lines(((0.27, 0.28), (0.31, 0.88), (0.69, 0.88), (0.73, 0.28)))
    elif name == "close":
        lines(((0.25, 0.25), (0.75, 0.75)))
        lines(((0.25, 0.75), (0.75, 0.25)))
    elif name == "deselect":
        for a, b, c in (
            ((0.35, 0.12), (0.12, 0.12), (0.12, 0.35)),
            ((0.65, 0.12), (0.88, 0.12), (0.88, 0.35)),
            ((0.12, 0.65), (0.12, 0.88), (0.35, 0.88)),
            ((0.88, 0.65), (0.88, 0.88), (0.65, 0.88)),
        ):
            lines((a, b, c))
        lines(((0.34, 0.34), (0.66, 0.66)))
        lines(((0.34, 0.66), (0.66, 0.34)))
    elif name in ("play", "previous"):
        tip, base = (0.80, 0.20) if name == "play" else (0.20, 0.80)
        draw.add_triangle_filled(point(base, 0.16), point(tip, 0.5), point(base, 0.84), color)
    elif name in ("first", "last"):
        tip, base = (0.28, 0.78) if name == "first" else (0.72, 0.22)
        draw.add_triangle_filled(point(tip, 0.5), point(base, 0.18), point(base, 0.82), color)
        bar = 0.15 if name == "first" else 0.85
        lines(((bar, 0.15), (bar, 0.85)))
    elif name in ("surface", "cloth"):
        lines(((0.12, 0.15), (0.72, 0.15), (0.88, 0.83), (0.28, 0.72)), True)
        lines(((0.32, 0.15), (0.48, 0.76)))
        lines(((0.52, 0.15), (0.68, 0.8)))
        lines(((0.18, 0.37), (0.78, 0.38)))
        lines(((0.23, 0.55), (0.83, 0.61)))
    elif name in ("volume", "tet"):
        lines(((0.5, 0.08), (0.08, 0.77), (0.9, 0.84)), True)
        lines(((0.5, 0.08), (0.52, 0.57), (0.08, 0.77)))
        lines(((0.52, 0.57), (0.9, 0.84)))
    elif name == "tri":
        lines(((0.5, 0.12), (0.12, 0.85), (0.88, 0.85)), True)
    elif name == "particle":
        circle(0.5, 0.5, 0.25, True)
    elif name == "pause":
        for a in (0.25, 0.61):
            draw.add_rect_filled(point(a, 0.14), point(a + 0.15, 0.86), color)
    elif name in ("reset", "undo"):
        lines(tuple((0.5 + 0.34 * cos(t), 0.5 + 0.34 * sin(t)) for t in np.linspace(-2.8, 2.2, 11)))
        lines(((0.08, 0.11), (0.12, 0.42), (0.43, 0.35)))
    elif name in ("inspector", "sliders"):
        for a, b in ((0.23, 0.33), (0.50, 0.65), (0.77, 0.42)):
            lines(((a, 0.1), (a, 0.9)))
            draw.add_rect_filled(point(a - 0.11, b - 0.08), point(a + 0.11, b + 0.08), color)
    else:
        lines(((0.14, 0.14), (0.86, 0.14), (0.86, 0.86), (0.14, 0.86)), True)
