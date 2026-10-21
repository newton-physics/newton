# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compact, typed property controls and resolution-independent vector icons."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from math import cos, pi, sin
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
    from enum import IntEnum
    from types import ModuleType
    from typing import Any

    from imgui_bundle import imgui
    from numpy.typing import ArrayLike

    from newton._src.viewer.gui.viewer_gui import ViewerGui


@dataclass
class PropertyEdit:
    """Proposed widget value and gesture boundaries; never a device-side mutation."""

    value: Any
    """Value to apply only when ``changed`` is True."""
    changed: bool = False
    """Whether this draw produced an accepted editable value."""
    activated: bool = False
    """Whether a gesture started, allowing the caller to capture one undo baseline."""
    finished: bool = False
    """Whether ImGui reports that an edited item was deactivated."""


class PhysicsWidgets:
    """Render physics values without knowing their simulation owner or frequency.

    Callers supply host values, units, and edit permissions, then consume a
    PropertyEdit to update storage. Widgets own only presentation state such as
    quaternion display mode. Repeated fields require distinct ImGui ID scopes;
    labels alone are not unique across objects or custom namespaces.
    """

    accent = (0.96, 0.61, 0.32, 1)
    muted = (0.65, 0.65, 0.65, 1)
    axes = ((0.87, 0.39, 0.38, 1), (0.44, 0.75, 0.47, 1), (0.38, 0.62, 0.91, 1), (0.71, 0.60, 0.88, 1))
    explorer_colors: ClassVar[dict[str, tuple[float, float, float, float]]] = {
        "root": (0.74, 0.74, 0.74, 1),
        "group": (0.74, 0.74, 0.74, 1),
        "elements": (0.68, 0.68, 0.68, 1),
        "articulation": (0.82, 0.61, 0.40, 1),
        "body": (0.88, 0.64, 0.38, 1),
        "shape": (0.43, 0.73, 0.55, 1),
        "joint": (0.48, 0.69, 0.80, 1),
        "surface": (0.40, 0.73, 0.63, 1),
        "volume": (0.76, 0.55, 0.68, 1),
        "particle_set": (0.84, 0.71, 0.43, 1),
        "particle": (0.84, 0.71, 0.43, 1),
        "tri": (0.40, 0.73, 0.63, 1),
        "tet": (0.76, 0.55, 0.68, 1),
        "custom": (0.69, 0.61, 0.81, 1),
        "sdf": (0.40, 0.74, 0.79, 1),
        "bvh": (0.40, 0.74, 0.79, 1),
        "texture": (0.76, 0.57, 0.82, 1),
    }

    def __init__(self, gui: ViewerGui) -> None:
        self.gui = gui
        self._text_centers: dict[tuple[int, float, str], float] = {}
        self._property_summary: Callable[[], str] | None = None

    @contextmanager
    def property_tooltip(self, summary: Callable[[], str] | None) -> Iterator[None]:
        """Attach lazy storage metadata to property labels and group headers."""
        previous = self._property_summary
        self._property_summary = summary
        try:
            yield
        finally:
            self._property_summary = previous

    def _tooltip(self, detail: str = "") -> None:
        """Keep editor help alongside attribute metadata, including disabled fields."""
        ui = self.ui
        if not ui.is_item_hovered(ui.HoveredFlags_.delay_normal | ui.HoveredFlags_.allow_when_disabled):
            return
        summary = self._property_summary() if self._property_summary is not None else ""
        text = "\n\n".join(part for part in (summary, detail) if part)
        if text:
            ui.set_tooltip(text)

    @property
    def ui(self) -> ModuleType:
        return self.gui.ui.imgui

    @property
    def scale(self) -> float:
        return self.gui.ui.dpi_scale

    def color(self, rgba: ArrayLike) -> int:
        return self.ui.get_color_u32(self.ui.ImVec4(*rgba))

    def text_top(self, y: float, height: float, *, numeric: bool = False) -> int:
        """Center visible glyphs instead of the font's unequal ascent/descent box."""
        ui = self.ui
        sample = "0123456789.-" if numeric else "Xx0123456789"
        key = (ui.get_font().font_id, ui.get_font_size(), sample)
        if key not in self._text_centers:
            baked = ui.get_font_baked()
            glyphs = [baked.find_glyph(ord(c)) for c in sample]
            glyphs = [g for g in glyphs if g is not None]
            self._text_centers[key] = (min(g.y0 for g in glyphs) + max(g.y1 for g in glyphs)) / 2 if glyphs else 0
        return round(y + height / 2 - self._text_centers[key])

    def label(self, text: str, *, height: float | None = None, width: float | None = None, muted: bool = False) -> None:
        ui = self.ui
        pos = ui.get_cursor_screen_pos()
        height = ui.get_frame_height() if height is None else height
        width = ui.calc_text_size(text).x if width is None else width
        ui.get_window_draw_list().add_text(
            ui.ImVec2(pos.x, self.text_top(pos.y, height)),
            ui.get_color_u32(ui.Col_.text_disabled if muted else ui.Col_.text),
            text,
        )
        ui.dummy(ui.ImVec2(width, height))

    def button(self, label: str, size: imgui.ImVec2 | None = None, *, small: bool = False) -> bool:
        return self.ui.small_button(label) if small else self.ui.button(label, size or self.ui.ImVec2(0, 0))

    def tab_item(self, label: str, *, flags: int = 0, accent: bool = False) -> tuple[bool, bool | None]:
        ui = self.ui
        if not accent:
            return ui.begin_tab_item(label, flags=flags)
        ui.push_style_color(ui.Col_.tab, ui.ImVec4(0.29, 0.23, 0.18, 1))
        ui.push_style_color(ui.Col_.tab_selected, ui.ImVec4(0.40, 0.29, 0.20, 1))
        ui.push_style_color(ui.Col_.tab_hovered, ui.ImVec4(0.44, 0.32, 0.22, 1))
        ui.push_style_color(ui.Col_.text, ui.ImVec4(*self.accent))
        try:
            return ui.begin_tab_item(label, flags=flags)
        finally:
            ui.pop_style_color(4)

    def icon(
        self,
        name: str,
        position: imgui.ImVec2 | tuple[float, float],
        size: float = 16,
        *,
        active: bool = False,
        dim: bool = False,
        tint: Sequence[float] | None = None,
        solid: bool = False,
    ) -> None:
        """Draw vector icons without an icon font, asset loading, or textures."""
        ui = self.ui
        draw = ui.get_window_draw_list()
        x, y = position
        if tint is not None and active:
            tint = (*(channel + (1 - channel) * 0.20 for channel in tint[:3]), tint[3])
        rgba = (0.40, 0.40, 0.40, 1) if dim else tint or (self.accent if active else self.muted)
        color = self.color(rgba)
        width = max(1, self.scale * (1.5 if solid else 1.15))

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
            return self.color((*(channel * factor for channel in rgba[:3]), rgba[3]))

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

    def icon_button(
        self,
        key: str,
        icon: str,
        tooltip: str,
        *,
        active: bool = False,
        enabled: bool = True,
        size: float = 24,
        label: str = "",
        prominent: bool = False,
        show_lock: bool = True,
        flat: bool = False,
    ) -> bool:
        ui, s = self.ui, self.scale
        pos = ui.get_cursor_screen_pos()
        button_width = size + (14 if prominent else 0)
        width = button_width * s + (ui.calc_text_size(label).x + 5 * s if label else 0)
        ui.begin_disabled(not enabled)
        clicked = ui.invisible_button(key, ui.ImVec2(width, size * s))
        hovered = ui.is_item_hovered(ui.HoveredFlags_.allow_when_disabled)
        ui.end_disabled()
        # A disabled action can still represent an enabled physics flag.
        # Keep its state color; the lock communicates that it is read-only.
        if prominent:
            draw = ui.get_window_draw_list()
            pressed = enabled and ui.is_item_active()
            fill = (0.24, 0.24, 0.24, 1) if pressed else (0.38, 0.38, 0.38, 1) if hovered else (0.30, 0.30, 0.30, 1)
            end = ui.ImVec2(pos.x + width, pos.y + size * s)
            draw.add_rect_filled(pos, end, self.color(fill), 3 * s)
        elif hovered or (not flat and (active or size >= 22)):
            ui.get_window_draw_list().add_rect_filled(
                pos,
                ui.ImVec2(pos.x + width, pos.y + size * s),
                self.color(
                    (0.30, 0.30, 0.30, 1) if active else (0.25, 0.25, 0.25, 1) if hovered else (0.19, 0.19, 0.19, 1)
                ),
                3 * s,
            )
        icon_size = 18 if prominent else 16
        self.icon(
            icon,
            (pos.x + (button_width - icon_size) * s / 2, pos.y + (size - icon_size) * s / 2),
            icon_size * s,
            active=active or (prominent and enabled),
            dim=not enabled and not active,
        )
        if label:
            ui.get_window_draw_list().add_text(
                ui.ImVec2(pos.x + button_width * s, self.text_top(pos.y, size * s)),
                self.color(self.accent if active else self.muted),
                label,
            )
        if not enabled and show_lock:
            self.icon("lock", (pos.x + (size - 8) * s, pos.y + (size - 9) * s), 8 * s)
        if (
            tooltip
            and tooltip != label
            and ui.is_item_hovered(ui.HoveredFlags_.allow_when_disabled | ui.HoveredFlags_.delay_normal)
        ):
            ui.set_tooltip(tooltip)
        return clicked and enabled

    def _fixed_label(
        self,
        text: str,
        width: float,
        *,
        color: Sequence[float] | None = None,
        locked: bool = False,
        reason: str = "",
        unit: str = "",
    ) -> None:
        ui, scale = self.ui, self.scale
        pos = ui.get_cursor_screen_pos()
        height = ui.get_frame_height()
        lock_width = 15 * scale if locked or unit else 0
        unit_width = ui.calc_text_size(unit).x + 5 * scale if unit else 0
        draw = ui.get_window_draw_list()
        draw.push_clip_rect(pos, ui.ImVec2(pos.x + max(1, width - lock_width - unit_width), pos.y + height), True)
        draw.add_text(ui.ImVec2(pos.x, self.text_top(pos.y, height)), self.color(color or (0.88, 0.88, 0.88, 1)), text)
        draw.pop_clip_rect()
        if unit:
            draw.add_text(
                ui.ImVec2(pos.x + width - lock_width - unit_width + 5 * scale, self.text_top(pos.y, height)),
                self.color(self.muted),
                unit,
            )
        if locked:
            self.icon("lock", (pos.x + width - 13 * scale, pos.y + (height - 11 * scale) / 2), 11 * scale)
        ui.dummy(ui.ImVec2(width, height))
        if ui.is_item_hovered(ui.HoveredFlags_.delay_normal):
            details = []
            if ui.calc_text_size(text).x > width - lock_width - unit_width:
                details.append(text)
            if locked and reason:
                details.append(reason)
            self._tooltip("\n".join(details))

    @contextmanager
    def group(self, label: str, *, default_open: bool = True) -> Iterator[bool]:
        """A flat, softly rounded card with a disclosure header and inset fields."""
        ui, scale = self.ui, self.scale
        ui.push_style_var(ui.StyleVar_.child_rounding, 5 * scale)
        ui.push_style_var(ui.StyleVar_.child_border_size, 0)
        ui.push_style_var(ui.StyleVar_.window_padding, ui.ImVec2(8 * scale, 4 * scale))
        ui.push_style_var(ui.StyleVar_.item_spacing, ui.ImVec2(6 * scale, 5 * scale))
        ui.push_style_color(ui.Col_.child_bg, ui.ImVec4(0.215, 0.215, 0.215, 1))
        ui.push_style_color(ui.Col_.header, ui.ImVec4(0.215, 0.215, 0.215, 1))
        ui.push_style_color(ui.Col_.header_hovered, ui.ImVec4(0.25, 0.25, 0.25, 1))
        ui.push_style_color(ui.Col_.header_active, ui.ImVec4(0.28, 0.28, 0.28, 1))
        visible = ui.begin_child(
            "##group_" + label,
            ui.ImVec2(0, 0),
            ui.ChildFlags_.auto_resize_y | ui.ChildFlags_.always_use_window_padding,
        )
        ui.push_style_var(ui.StyleVar_.frame_border_size, 0)
        ui.push_style_var(ui.StyleVar_.frame_rounding, 2 * scale)
        opened = visible and ui.collapsing_header(label, ui.TreeNodeFlags_.default_open if default_open else 0)
        if visible:
            self._tooltip()
        ui.pop_style_var(2)
        try:
            yield opened
        finally:
            ui.end_child()
            ui.pop_style_color(4)
            ui.pop_style_var(4)

    @contextmanager
    def disclosure(self, label: str, *, default_open: bool = True) -> Iterator[bool]:
        """Group paged attributes without adding another panel or inset."""
        ui = self.ui
        opened = ui.collapsing_header(label, ui.TreeNodeFlags_.default_open if default_open else 0)
        self._tooltip()
        yield opened

    @contextmanager
    def _child_rows(self) -> Iterator[None]:
        """Keep a shallow, balanced indent for compound property children."""
        indent = 12 * self.scale
        self.ui.indent(indent)
        try:
            yield
        finally:
            self.ui.unindent(indent)

    @contextmanager
    def compound(self, label: str, *, editable: bool = False, unit: str = "", reason: str = "") -> Iterator[None]:
        """Group compound attribute children using indentation and a regular label."""
        self._label(label, editable, unit, reason)
        with self._child_rows():
            yield

    def _label(
        self,
        label: str,
        editable: bool,
        unit: str = "",
        reason: str = "",
        *,
        inline: bool = False,
        width: float | None = None,
    ) -> None:
        ui, s = self.ui, self.scale
        if not label or label.startswith("##"):
            return
        if inline:
            width = width if width is not None else ui.get_content_region_avail().x * 0.46
            self._fixed_label(label, width, locked=not editable, reason=reason, unit=unit)
            ui.same_line(spacing=6 * s)
            return
        ui.text(label)
        self._tooltip()
        if unit:
            ui.same_line(spacing=5 * s)
            ui.text_disabled(unit)
        if not editable:
            ui.same_line(spacing=6 * s)
            pos = ui.get_cursor_screen_pos()
            self.icon("lock", (pos.x, pos.y + 2 * s), 11 * s)
            ui.dummy(ui.ImVec2(13 * s, ui.get_text_line_height()))
            self._tooltip(reason)

    def text_value(self, label: str, value: Any, *, reason: str = "") -> None:
        """Render read-only text with explicit missing and empty-string values."""
        missing = value is None
        empty = isinstance(value, str) and not value
        if missing:
            text = "N/A"
        elif empty:
            text = "Empty string"
        elif isinstance(value, (str, int, float, bool, np.generic)):
            text = str(value)
        else:
            text = type(value).__name__
        ui = self.ui
        self._label(label, False, reason=reason, inline=True)
        ui.align_text_to_frame_padding()
        if missing or empty:
            ui.push_style_color(ui.Col_.text, ui.ImVec4(*self.muted))
        ui.text_wrapped(text)
        if missing or empty:
            ui.pop_style_color()
        self._tooltip(reason)

    def boolean(self, label: str, value: bool, *, editable: bool = False, reason: str = "") -> PropertyEdit:
        """Show explicit boolean values; reserve checkbox controls for editable fields."""
        if not editable:
            self.text_value(label, bool(value), reason=reason)
            return PropertyEdit(bool(value))
        ui = self.ui
        self._label(label, True, inline=True)
        ui.push_id(label)
        changed, value = ui.checkbox("##bool", bool(value))
        edit = PropertyEdit(value, changed, ui.is_item_activated(), ui.is_item_deactivated_after_edit())
        self._tooltip()
        ui.same_line(spacing=6 * self.scale)
        self.label("True" if value else "False")
        self._tooltip()
        ui.pop_id()
        return edit

    def number(
        self,
        label: str,
        value: float | int,
        *,
        editable: bool = False,
        integer: bool = False,
        speed: float = 0.01,
        bounds: tuple[float, float] | None = None,
        unit: str = "",
        reason: str = "",
        width: float | None = None,
        label_width: float | None = None,
        format_spec: str | None = None,
        slider: bool = False,
    ) -> PropertyEdit:
        """Render a bounded slider or constant-speed numeric drag with text entry.

        ``bounds=None`` leaves the value unbounded; it does not create an
        artificial slider range. ``speed`` is the value change per drag pixel,
        independent of the current value's magnitude. Formatting affects display
        precision only, preserving stored values until an edit is accepted.
        """
        ui = self.ui
        self._label(label, editable, unit, reason, inline=True, width=label_width)
        if width is not None:
            ui.set_next_item_width(width)
        else:
            ui.set_next_item_width(-1)
        ui.push_id(label)
        ui.push_style_color(
            ui.Col_.frame_bg, ui.ImVec4(*(0.31, 0.31, 0.31, 1) if editable else (0.265, 0.265, 0.265, 1))
        )
        ui.push_style_color(ui.Col_.text, ui.ImVec4(*(0.92, 0.92, 0.92, 1) if editable else (0.65, 0.65, 0.65, 1)))
        ui.push_style_var(ui.StyleVar_.disabled_alpha, 1.0)
        ui.begin_disabled(not editable)
        font = getattr(self.gui.ui, "number_font", None)
        text_height = ui.get_font_size()
        if font:
            ui.push_font(font, 13.0)
        padding = ui.get_style().frame_padding
        # The numeric font is slightly smaller. Keep fields and their labels on
        # the same row instead of letting the font change shorten the control.
        ui.push_style_var(
            ui.StyleVar_.frame_padding,
            ui.ImVec2(padding.x, padding.y + max(0, text_height - ui.get_font_size()) / 2),
        )
        flags = ui.SliderFlags_.no_round_to_format
        if bounds is not None:
            flags |= ui.SliderFlags_.always_clamp
        low, high = bounds or (0, 0)
        precision = 6
        if not integer:
            available = (
                width if width is not None else ui.get_content_region_avail().x
            ) - 2 * ui.get_style().frame_padding.x
            while precision > 1 and ui.calc_text_size(f"{float(value):.{precision}g}").x > available:
                precision -= 1
        format_spec = format_spec or f".{precision}g"
        # Keep native dragging, focus, keyboard entry and selection. Only replace
        # the centered display text when the native text editor is inactive.
        item_id = ui.get_id("##value")
        typing = ui.internal.temp_input_is_active(item_id)
        if not typing:
            ui.push_style_color(ui.Col_.text, ui.ImVec4(0, 0, 0, 0))
        if slider and bounds is not None and not integer:
            changed, result = ui.slider_float(
                "##value", float(value), float(low), float(high), "%" + format_spec, flags
            )
        elif integer:
            changed, result = ui.drag_int("##value", int(value), speed, int(low), int(high), "%d", flags)
        else:
            changed, result = ui.drag_float(
                "##value", float(value), speed, float(low), float(high), "%" + format_spec, flags
            )
        # Just a soft inset top edge; no bevel or bright outline around fields.
        lo, hi = ui.get_item_rect_min(), ui.get_item_rect_max()
        draw, scale = ui.get_window_draw_list(), self.scale
        inset = 3 * scale
        for offset, alpha in ((0.5, 0.15), (1.5, 0.05)):
            draw.add_line(
                ui.ImVec2(lo.x + inset, lo.y + offset * scale),
                ui.ImVec2(hi.x - inset, lo.y + offset * scale),
                self.color((0, 0, 0, alpha)),
                scale,
            )
        if not typing:
            ui.pop_style_color()
            if not ui.internal.temp_input_is_active(item_id):
                lo, hi = ui.get_item_rect_min(), ui.get_item_rect_max()
                padding = ui.get_style().frame_padding
                draw = ui.get_window_draw_list()
                draw.push_clip_rect(lo, hi, True)
                text = str(int(result)) if integer else format(float(result), format_spec)
                draw.add_text(
                    ui.ImVec2(lo.x + padding.x, self.text_top(lo.y, hi.y - lo.y, numeric=True)),
                    self.color((0.92, 0.92, 0.92, 1) if editable else (0.65, 0.65, 0.65, 1)),
                    text,
                )
                draw.pop_clip_rect()
        edit = PropertyEdit(result, changed and editable, ui.is_item_activated(), ui.is_item_deactivated_after_edit())
        if ui.is_item_hovered(ui.HoveredFlags_.allow_when_disabled):
            if editable and not ui.is_item_active():
                ui.set_mouse_cursor(ui.MouseCursor_.resize_ew)
        if font:
            ui.pop_font()
        ui.pop_style_var()
        ui.end_disabled()
        ui.pop_style_var()
        ui.pop_style_color(2)
        ui.pop_id()
        return edit

    def reference(
        self, label: str, value: int, *, target: str | None = None, caption: str | None = None, reason: str = ""
    ) -> bool:
        """Pair a compact index with a wider, named navigation button."""
        ui, scale = self.ui, self.scale
        available = ui.get_content_region_avail().x
        spacing = 6 * scale
        padding = ui.get_style().frame_padding.x
        font = getattr(self.gui.ui, "number_font", None)
        if font:
            ui.push_font(font, 13.0)
        index_width = max(ui.calc_text_size("00000").x, ui.calc_text_size(str(value)).x) + 2 * padding
        if font:
            ui.pop_font()
        # Keep the normal label column whenever possible. The index stays at
        # five digits; extra panel width belongs to the name and attribute label.
        label_width = min(available * 0.46, max(0, available - 2.5 * index_width - 2 * spacing))
        fields_width = max(1, available - label_width - 2 * spacing)
        index_width = min(index_width, fields_width * 0.4)
        button_width = max(1, fields_width - index_width)
        self.number(
            label,
            value,
            integer=True,
            reason=reason,
            label_width=label_width,
            width=index_width,
        )
        ui.same_line(spacing=spacing)
        if target is None:
            caption = "N/A"
        elif caption is None:
            caption = target.rstrip("/").rsplit("/", 1)[-1]
        text_width = max(0, button_width - 2 * padding)
        if ui.calc_text_size(caption).x > text_width:
            low, high = 0, len(caption)
            while low < high:
                middle = (low + high + 1) // 2
                if ui.calc_text_size(caption[:middle] + "…").x <= text_width:
                    low = middle
                else:
                    high = middle - 1
            caption = caption[:low] + "…"
        ui.begin_disabled(target is None)
        # Stable IDs depend on the property, not on a changing target name.
        # Draw the literal caption so names containing ImGui's ## remain intact.
        clicked = self.button("##reference:" + label, ui.ImVec2(button_width, ui.get_frame_height()))
        lo, hi = ui.get_item_rect_min(), ui.get_item_rect_max()
        draw = ui.get_window_draw_list()
        draw.push_clip_rect(lo, hi, True)
        draw.add_text(
            ui.ImVec2(
                lo.x + max(padding, (button_width - ui.calc_text_size(caption).x) / 2), self.text_top(lo.y, hi.y - lo.y)
            ),
            ui.get_color_u32(ui.Col_.text),
            caption,
        )
        draw.pop_clip_rect()
        ui.end_disabled()
        if target is not None and caption != target and ui.is_item_hovered(ui.HoveredFlags_.delay_normal):
            ui.set_tooltip(target)
        return clicked and target is not None

    def array(
        self,
        label: str,
        value: ArrayLike,
        *,
        editable: bool = False,
        unit: str = "",
        labels: Sequence[str] | None = None,
        component_units: Sequence[str] | None = None,
        speed: float = 0.01,
        reason: str = "",
        integer: bool = False,
        bounds: tuple[float, float] | None = None,
        header: bool = True,
    ) -> PropertyEdit:
        """Use a stable grid and bound rendering work for arrays of any length."""
        ui, scale = self.ui, self.scale
        values = np.asarray(value).reshape(-1)
        count = len(values)
        heading = header and bool(label) and not label.startswith("##")
        with self.compound(label, editable=editable, unit=unit, reason=reason) if heading else nullcontext():
            result = PropertyEdit(values)
            ui.push_id(label)
            columns = max(1, min(3, int(ui.get_content_region_avail().x / (92 * scale))))
            page_size = columns * 8
            start, stop = 0, count
            if count > page_size:
                storage = ui.get_state_storage()
                key = ui.get_id("##page")
                pages = (count + page_size - 1) // page_size
                page = min(pages - 1, storage.get_int(key))
                if self.icon_button(
                    "##previous_page", "previous", "Previous components", enabled=page > 0, show_lock=False, size=22
                ):
                    page = max(0, page - 1)
                ui.same_line()
                if self.icon_button(
                    "##next_page", "play", "Next components", enabled=page < pages - 1, show_lock=False, size=22
                ):
                    page = min(pages - 1, page + 1)
                ui.same_line()
                edit = self.number(
                    "##page_number", page + 1, editable=True, integer=True, speed=1, bounds=(1, pages), width=65 * scale
                )
                if edit.changed:
                    page = edit.value - 1
                storage.set_int(key, page)
                start, stop = page * page_size, min(count, (page + 1) * page_size)
                ui.same_line()
                self.label(f"{start + 1}-{stop} / {count}", muted=True)
            ui.push_style_var(ui.StyleVar_.cell_padding, ui.ImVec2(4 * scale, 2 * scale))
            ui.push_style_var(ui.StyleVar_.frame_padding, ui.ImVec2(4 * scale, ui.get_style().frame_padding.y))
            if count and ui.begin_table("##components", columns, ui.TableFlags_.sizing_stretch_same):
                for i in range(start, stop):
                    ui.table_next_column()
                    name = labels[i] if labels is not None else str(i)
                    self._fixed_label(
                        name,
                        24 * scale,
                        color=self.axes["XYZW".index(name[0])] if name and name[0] in "XYZW" else self.muted,
                    )
                    ui.same_line(spacing=4 * scale)
                    ui.push_id(i)
                    edit = self.number(
                        "",
                        values[i],
                        editable=editable,
                        integer=integer,
                        bounds=bounds,
                        speed=speed,
                        reason=reason,
                        unit=component_units[i] if component_units is not None else unit,
                    )
                    if edit.changed:
                        if not result.changed:
                            result.value = values.copy()
                        result.value[i] = edit.value
                    result.changed |= edit.changed
                    result.activated |= edit.activated
                    result.finished |= edit.finished
                    ui.pop_id()
                ui.end_table()
            ui.pop_style_var(2)
            ui.pop_id()
            return result

    def vector(
        self,
        label: str,
        value: ArrayLike,
        *,
        editable: bool = False,
        unit: str = "",
        labels: Sequence[str] | None = None,
        speed: float = 0.01,
        reason: str = "",
        header: bool = True,
    ) -> PropertyEdit:
        values = np.asarray(value, dtype=np.float32).reshape(-1)
        if labels is None and len(values) <= 4:
            labels = tuple("XYZW"[: len(values)])
        return self.array(
            label, values, editable=editable, unit=unit, labels=labels, speed=speed, reason=reason, header=header
        )

    def vec2(self, label: str, value: ArrayLike, **kwargs: Any) -> PropertyEdit:
        return self.vector(label, np.asarray(value).reshape(2), **kwargs)

    def vec3(self, label: str, value: ArrayLike, **kwargs: Any) -> PropertyEdit:
        return self.vector(label, np.asarray(value).reshape(3), **kwargs)

    rotation_modes = (
        "Quaternion",
        "Euler XYZ",
        "Euler XZY",
        "Euler YXZ",
        "Euler YZX",
        "Euler ZXY",
        "Euler ZYX",
        "Axis angle",
        "Rotation vector",
    )

    @staticmethod
    def rotation_values(value: ArrayLike, mode: str, degrees: bool = True) -> np.ndarray:
        """Convert display values only; the simulation continues to store XYZW."""
        q = np.asarray(value, dtype=np.float64)
        if mode == "Quaternion":
            return q.copy()
        length = np.linalg.norm(q)
        if length < 1e-12 or not np.isfinite(length):
            raise ValueError("Quaternion must be finite and have nonzero length.")
        q = q / length
        if mode.startswith("Euler"):
            axes = tuple("XYZ".index(c) for c in mode[-3:])
            values = np.array(wp.quat_to_euler(wp.quat(*q), *axes), dtype=np.float64)
            values[np.abs(values) < 1e-7] = 0
            return np.rad2deg(values) if degrees else values
        if q[3] < 0:
            q = -q
        length = np.linalg.norm(q[:3])
        axis = q[:3] / length if length > 1e-12 else np.array([1.0, 0.0, 0.0])
        angle = 2 * np.arctan2(length, q[3])
        if degrees:
            angle = np.rad2deg(angle)
        return np.r_[axis, angle] if mode == "Axis angle" else axis * angle

    @staticmethod
    def rotation_quaternion(values: ArrayLike, mode: str, degrees: bool = True) -> np.ndarray:
        """Convert an editor representation to a normalized XYZW quaternion.

        Euler modes use the fixed-axis order encoded in the mode name. Axis-angle
        uses XYZ plus angle; rotation-vector magnitude is the angle. ``degrees``
        applies only to angles, not axis components or quaternion coefficients.

        Raises:
            ValueError: Values are nonfinite, a quaternion has zero length, or a
                nonzero angle is paired with a zero axis.
        """
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Enter finite rotation values.")
        if mode == "Quaternion":
            q = values.copy()
        elif mode.startswith("Euler"):
            axes = tuple("XYZ".index(c) for c in mode[-3:])
            q = np.array(wp.quat_from_euler(wp.vec3(*(np.deg2rad(values) if degrees else values)), *axes))
        else:
            length = np.linalg.norm(values[:3])
            angle = float(values[3]) if mode == "Axis angle" else length
            if length < 1e-12:
                if abs(angle) > 1e-12:
                    raise ValueError("A nonzero angle needs a nonzero axis.")
                return np.array([0, 0, 0, 1], dtype=np.float32)
            axis = values[:3] / length
            angle = np.deg2rad(angle) if degrees else angle
            q = np.array(wp.quat_from_axis_angle(wp.vec3(*axis), float(angle)))
        length = np.linalg.norm(q)
        if length < 1e-12:
            raise ValueError("Quaternion must have nonzero length.")
        return (q / length).astype(np.float32)

    def quat(self, label: str, value: ArrayLike, *, editable: bool = False, reason: str = "") -> PropertyEdit:
        ui, scale = self.ui, self.scale
        original = np.asarray(value, dtype=np.float32).reshape(4)
        cache = getattr(self.gui, "_physics_rotations", None)
        if cache is None:
            cache = self.gui._physics_rotations = OrderedDict()
        key = ui.get_id(label)
        state = cache.setdefault(key, {"mode": 0, "degrees": True, "q": None, "values": None, "error": ""})
        cache.move_to_end(key)
        while len(cache) > 128:
            cache.popitem(last=False)
        title = "Rotation" if label.startswith("##") else label
        self._label(title, editable, reason=reason, inline=True)
        ui.push_id(label)
        available = ui.get_content_region_avail().x
        show_units = self.rotation_modes[state["mode"]] != "Quaternion"
        ui.set_next_item_width(max(50 * scale, available - (40 * scale if show_units else 0)))
        changed_mode, mode_index = ui.combo(
            "##rotation_mode", state["mode"], list(self.rotation_modes), len(self.rotation_modes)
        )
        state["mode"] = mode_index
        mode = self.rotation_modes[mode_index]
        if ui.is_item_hovered(ui.HoveredFlags_.delay_normal):
            tip = "Display representation; storage remains a quaternion."
            if mode.startswith("Euler"):
                tip += "\nFixed axes (extrinsic); gimbal lock at a middle angle of ±90°."
            ui.set_tooltip(tip)
        units_changed = False
        if mode != "Quaternion" and show_units:
            ui.same_line(spacing=5 * scale)
            if self.button("deg" if state["degrees"] else "rad", ui.ImVec2(35 * scale, 0)):
                state["degrees"] = not state["degrees"]
                units_changed = True
        ui.pop_id()
        same_q = state["q"] is not None and (
            np.allclose(original, state["q"], atol=1e-7, rtol=0)
            or (mode != "Quaternion" and np.allclose(-original, state["q"], atol=1e-7, rtol=0))
        )
        # Equivalent quaternions can have opposite signs, and Euler angles have
        # many equivalent branches. Keep the user's representation through a
        # drag instead of repeatedly converting it back to a canonical branch.
        if changed_mode or units_changed or not same_q:
            try:
                values = self.rotation_values(original, mode, state["degrees"])
            except ValueError as error:
                state["mode"], mode = 0, "Quaternion"
                values = original.copy()
                state["error"] = str(error)
            else:
                state["error"] = ""
            if units_changed and not changed_mode and same_q and state["values"] is not None:
                values = state["values"].copy()
                angles = slice(3, 4) if mode == "Axis angle" else slice(None)
                values[angles] = np.rad2deg(values[angles]) if state["degrees"] else np.deg2rad(values[angles])
            if mode.startswith("Euler") and not changed_mode and not units_changed and state["values"] is not None:
                period = 360 if state["degrees"] else 2 * np.pi
                values += period * np.round((state["values"] - values) / period)
            state["values"], state["q"] = values, original.copy()
        labels = (
            tuple("XYZW") if mode == "Quaternion" else ("X", "Y", "Z", "A") if mode == "Axis angle" else tuple("XYZ")
        )
        unit = "" if mode == "Quaternion" else "deg" if state["degrees"] else "rad"
        with self._child_rows():
            edit = self.array(
                label,
                state["values"],
                labels=labels,
                editable=editable,
                speed=0.01 if mode == "Quaternion" else 0.1 if state["degrees"] else 0.001,
                reason=reason,
                unit=unit,
                component_units=("", "", "", unit) if mode == "Axis angle" else None,
                header=False,
            )
        result = PropertyEdit(original.copy(), False, edit.activated, edit.finished)
        if edit.changed:
            try:
                q = self.rotation_quaternion(edit.value, mode, state["degrees"])
                if mode != "Quaternion" and np.dot(q, original) < 0:
                    q = -q
                state["q"] = q.copy()
                state["values"] = q.copy() if mode == "Quaternion" else edit.value.copy()
                state["error"] = ""
                result.value, result.changed = q, True
            except ValueError as error:
                state["error"] = str(error)
        if state["error"]:
            ui.text_wrapped(state["error"])
        return result

    def transform(self, label: str, value: ArrayLike, *, editable: bool = False, reason: str = "") -> PropertyEdit:
        """Group position and quaternion editors beneath their transform attribute."""
        value = np.asarray(value).reshape(7)
        self.ui.push_id(label)
        with self.compound(label, editable=editable, reason=reason):
            p = self.vec3("Position", value[:3], editable=editable, unit="m", speed=0.001, reason=reason)
            q = self.quat("Rotation", value[3:], editable=editable, reason=reason)
        self.ui.pop_id()
        return PropertyEdit(
            np.concatenate((p.value, q.value)),
            p.changed or q.changed,
            p.activated or q.activated,
            p.finished or q.finished,
        )

    def spatial_vector(
        self,
        label: str,
        value: ArrayLike,
        *,
        editable: bool = False,
        force: bool = False,
        position: bool = False,
        reason: str = "",
    ) -> PropertyEdit:
        """Group the linear and angular parts beneath their spatial-vector attribute."""
        value = np.asarray(value).reshape(6)
        self.ui.push_id(label)
        with self.compound(label, editable=editable, reason=reason):
            a = self.vec3(
                "Force" if force else "Linear",
                value[:3],
                editable=editable,
                unit="N" if force else "m" if position else "m/s",
                reason=reason,
            )
            b = self.vec3(
                "Torque" if force else "Angular",
                value[3:],
                editable=editable,
                unit="N m" if force else "rad" if position else "rad/s",
                reason=reason,
            )
        self.ui.pop_id()
        return PropertyEdit(
            np.concatenate((a.value, b.value)),
            a.changed or b.changed,
            a.activated or b.activated,
            a.finished or b.finished,
        )

    def matrix3(
        self, label: str, value: ArrayLike, *, unit: str = "", editable: bool = False, reason: str = ""
    ) -> PropertyEdit:
        values = np.asarray(value).reshape(3, 3).copy()
        result = PropertyEdit(values)
        self.ui.push_id(label)
        with self.compound(label, editable=editable, unit=unit, reason=reason):
            for i, axis in enumerate("XYZ"):
                edit = self.vector(
                    "##" + axis,
                    values[i],
                    labels=tuple(axis + c for c in "XYZ"),
                    editable=editable,
                    reason=reason,
                    header=False,
                )
                values[i] = edit.value
                result.changed |= edit.changed
                result.activated |= edit.activated
                result.finished |= edit.finished
        self.ui.pop_id()
        return result

    def axis(self, label: str, value: int | ArrayLike, *, editable: bool = False, reason: str = "") -> PropertyEdit:
        ui = self.ui
        self._label(label, editable, reason=reason, inline=True)
        ui.push_id(label)
        result = PropertyEdit(int(value))
        for i, name in enumerate("XYZ"):
            if i:
                ui.same_line()
            ui.push_style_color(
                ui.Col_.button, ui.ImVec4(*(*self.axes[i][:3], 0.35) if i == int(value) else (0.18, 0.18, 0.18, 1))
            )
            ui.begin_disabled(not editable)
            if self.button(name, ui.ImVec2(44 * self.scale, 0)):
                result = PropertyEdit(i, i != int(value), True, True)
            ui.end_disabled()
            ui.pop_style_color()
        ui.pop_id()
        return result

    def choice(
        self, label: str, value: int, choices: Mapping[int, str], *, editable: bool = False, reason: str = ""
    ) -> PropertyEdit:
        ui = self.ui
        self._label(label, editable, reason=reason, inline=True)
        values, names = list(choices), list(choices.values())
        if value not in values:
            values.append(value)
            names.append(str(value))
        ui.set_next_item_width(-1)
        ui.begin_disabled(not editable)
        changed, index = ui.combo("##" + label, values.index(value), names)
        ui.end_disabled()
        return PropertyEdit(values[index], changed and editable, changed, changed)

    def enum(
        self, label: str, value: int, options: Iterable[IntEnum], *, editable: bool = False, reason: str = ""
    ) -> PropertyEdit:
        return self.choice(
            label,
            int(value),
            {int(item): item.name.replace("_", " ").title() for item in options},
            editable=editable,
            reason=reason,
        )
