# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compact, typed property controls and DPI-aware cached icons."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import warp as wp

from . import theme
from .icons import draw_icon

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

    accent = theme.ACCENT
    muted = theme.MUTED
    axes = theme.AXES
    explorer_colors: ClassVar[dict[str, tuple[float, float, float, float]]] = theme.EXPLORER_COLORS

    def __init__(self, gui: ViewerGui) -> None:
        """Bind the GUI and initialize text-alignment and lazy property-tooltip state."""
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
        """Return the ImGui binding associated with this GUI instance."""
        return self.gui.ui.imgui

    @property
    def scale(self) -> float:
        """Return the GUI DPI scale used to convert logical widget dimensions to pixels."""
        return self.gui.ui.dpi_scale

    def color(self, rgba: ArrayLike) -> int:
        """Pack a float RGBA color using the current ImGui style alpha."""
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
        """Draw vertically centered text and reserve its requested pixel dimensions in the layout."""
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
        """Draw a standard or compact button and report a click."""
        return self.ui.small_button(label) if small else self.ui.button(label, size or self.ui.ImVec2(0, 0))

    def tab_item(self, label: str, *, flags: int = 0, accent: bool = False) -> tuple[bool, bool | None]:
        """Begin a tab with optional accent styling; the caller must end the tab when open."""
        ui = self.ui
        if not accent:
            return ui.begin_tab_item(label, flags=flags)
        ui.push_style_color(ui.Col_.tab, ui.ImVec4(*theme.ACCENT_TAB))
        ui.push_style_color(ui.Col_.tab_selected, ui.ImVec4(*theme.ACCENT_TAB_SELECTED))
        ui.push_style_color(ui.Col_.tab_hovered, ui.ImVec4(*theme.ACCENT_TAB_HOVERED))
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
        """Draw a tinted cached icon, retaining vector rendering for unusual sizes."""
        ui = self.ui
        draw = ui.get_window_draw_list()
        if tint is not None and active:
            tint = (*(channel + (1 - channel) * 0.20 for channel in tint[:3]), tint[3])
        rgba = theme.ICON_DISABLED if dim else tint or (self.accent if active else self.muted)
        atlas = getattr(self.gui.ui, "icon_atlas", None)
        if atlas is not None and atlas.draw(
            draw, name, position, size, scale=self.scale, color=self.color(rgba), solid=solid
        ):
            return
        draw_icon(ui, draw, name, position, size, scale=self.scale, rgba=rgba, solid=solid)

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
        """Draw an icon action and return whether it was clicked while enabled.

        Size is expressed in logical pixels. Active state and editability are
        independent: read-only active flags retain their accent and can show a
        lock. Tooltip text is displayed only when it adds to the caption.
        """
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
            fill = (
                theme.TRANSPORT_BUTTON_PRESSED
                if pressed
                else theme.TRANSPORT_BUTTON_HOVERED
                if hovered
                else theme.TRANSPORT_BUTTON
            )
            end = ui.ImVec2(pos.x + width, pos.y + size * s)
            draw.add_rect_filled(pos, end, self.color(fill), 3 * s)
        elif hovered or (not flat and (active or size >= 22)):
            ui.get_window_draw_list().add_rect_filled(
                pos,
                ui.ImVec2(pos.x + width, pos.y + size * s),
                self.color(
                    theme.TOOL_BUTTON_ACTIVE if active else theme.TOOL_BUTTON_HOVERED if hovered else theme.TOOL_BUTTON
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
        """Reserve a label column with clipped text, units, lock state, and metadata tooltip on hover."""
        ui, scale = self.ui, self.scale
        pos = ui.get_cursor_screen_pos()
        height = ui.get_frame_height()
        # Reserve the same layout even outside the scroll region, but avoid
        # emitting text and icon quads that would only be discarded by clipping.
        ui.dummy(ui.ImVec2(width, height))
        if not ui.is_item_visible():
            return
        lock_width = 15 * scale if locked or unit else 0
        unit_width = ui.calc_text_size(unit).x + 5 * scale if unit else 0
        draw = ui.get_window_draw_list()
        # Clip glyph geometry instead of changing the GPU scissor for each
        # label. Neighboring labels can then share ImGui's text draw command.
        draw.add_text(
            ui.get_font(),
            ui.get_font_size(),
            ui.ImVec2(pos.x, self.text_top(pos.y, height)),
            self.color(color or theme.TEXT),
            text,
            cpu_fine_clip_rect=ui.ImVec4(pos.x, pos.y, pos.x + max(1, width - lock_width - unit_width), pos.y + height),
        )
        if unit:
            draw.add_text(
                ui.ImVec2(pos.x + width - lock_width - unit_width + 5 * scale, self.text_top(pos.y, height)),
                self.color(self.muted),
                unit,
            )
        if locked:
            self.icon("lock", (pos.x + width - 13 * scale, pos.y + (height - 11 * scale) / 2), 11 * scale)
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
        ui.push_style_color(ui.Col_.child_bg, ui.ImVec4(*theme.GROUP_BACKGROUND))
        ui.push_style_color(ui.Col_.header, ui.ImVec4(*theme.GROUP_BACKGROUND))
        ui.push_style_color(ui.Col_.header_hovered, ui.ImVec4(*theme.GROUP_HOVERED))
        ui.push_style_color(ui.Col_.header_active, ui.ImVec4(*theme.GROUP_ACTIVE))
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
        """Draw an inline or stacked property label, skipping hidden ImGui labels and marking read-only values."""
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
            ui.Col_.frame_bg, ui.ImVec4(*theme.EDITOR_BACKGROUND if editable else theme.EDITOR_READONLY_BACKGROUND)
        )
        ui.push_style_color(ui.Col_.text, ui.ImVec4(*theme.EDITOR_TEXT if editable else theme.EDITOR_READONLY_TEXT))
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
            ui.push_style_color(ui.Col_.text, ui.ImVec4(*theme.TRANSPARENT))
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
        # Submit the native control even offscreen to preserve active edits and
        # keyboard navigation. Only its custom decoration/text can be skipped.
        visible = ui.is_item_visible()
        if visible:
            lo, hi = ui.get_item_rect_min(), ui.get_item_rect_max()
            draw, scale = ui.get_window_draw_list(), self.scale
            inset = 3 * scale
            for offset, alpha in ((0.5, 0.15), (1.5, 0.05)):
                draw.add_line(
                    ui.ImVec2(lo.x + inset, lo.y + offset * scale),
                    ui.ImVec2(hi.x - inset, lo.y + offset * scale),
                    self.color((*theme.BLACK[:3], alpha)),
                    scale,
                )
        if not typing:
            ui.pop_style_color()
            if visible and not ui.internal.temp_input_is_active(item_id):
                padding = ui.get_style().frame_padding
                text = str(int(result)) if integer else format(float(result), format_spec)
                draw.add_text(
                    ui.get_font(),
                    ui.get_font_size(),
                    ui.ImVec2(lo.x + padding.x, self.text_top(lo.y, hi.y - lo.y, numeric=True)),
                    self.color(theme.EDITOR_TEXT if editable else theme.EDITOR_READONLY_TEXT),
                    text,
                    cpu_fine_clip_rect=ui.ImVec4(lo.x, lo.y, hi.x, hi.y),
                )
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
        draw.add_text(
            ui.get_font(),
            ui.get_font_size(),
            ui.ImVec2(
                lo.x + max(padding, (button_width - ui.calc_text_size(caption).x) / 2), self.text_top(lo.y, hi.y - lo.y)
            ),
            ui.get_color_u32(ui.Col_.text),
            caption,
            cpu_fine_clip_rect=ui.ImVec4(lo.x, lo.y, hi.x, hi.y),
        )
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
        """Render a flat float vector, defaulting to XYZW component labels when it has at most four elements."""
        values = np.asarray(value, dtype=np.float32).reshape(-1)
        if labels is None and len(values) <= 4:
            labels = tuple("XYZW"[: len(values)])
        return self.array(
            label, values, editable=editable, unit=unit, labels=labels, speed=speed, reason=reason, header=header
        )

    def vec2(self, label: str, value: ArrayLike, **kwargs: Any) -> PropertyEdit:
        """Validate two components and delegate to the shared vector editor."""
        return self.vector(label, np.asarray(value).reshape(2), **kwargs)

    def vec3(self, label: str, value: ArrayLike, **kwargs: Any) -> PropertyEdit:
        """Validate three components and delegate to the shared vector editor."""
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
        """Edit an XYZW quaternion through a cached display representation.

        Retain representation, angular units, and drag continuity by widget ID;
        alternate representations still return quaternion storage. Reject
        invalid conversions and preserve quaternion sign continuity where possible.
        """
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
        """Render a 3-by-3 matrix in labeled rows and aggregate component edits without mutating the input."""
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
        """Render an integer X/Y/Z axis choice and return the selected value as a PropertyEdit."""
        ui = self.ui
        self._label(label, editable, reason=reason, inline=True)
        ui.push_id(label)
        result = PropertyEdit(int(value))
        for i, name in enumerate("XYZ"):
            if i:
                ui.same_line()
            ui.push_style_color(
                ui.Col_.button,
                ui.ImVec4(*(*self.axes[i][:3], 0.35) if i == int(value) else theme.AXIS_BUTTON_BACKGROUND),
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
        """Render integer-backed choices, retaining an unknown current value as an explicit option."""
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
        """Render integer enum values using readable names through the shared choice editor."""
        return self.choice(
            label,
            int(value),
            {int(item): item.name.replace("_", " ").title() for item in options},
            editable=editable,
            reason=reason,
        )
