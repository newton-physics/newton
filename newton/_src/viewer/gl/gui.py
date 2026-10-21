# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from imgui_bundle import imgui


logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class UI:
    def __init__(self, window):
        self.key_press_filter = None
        self._file_dialog_result: str | None = None
        self._pending_file_dialog = None
        self._frame_time = None
        self._frame_recovery_state = None
        self.section_errors = {}
        # DPI scale applied to ImGui style and fonts. Updated by ``_apply_dpi_scaling``
        # whenever the framebuffer-to-window ratio changes (e.g. moving the window
        # between displays with different DPI).
        self.dpi_scale: float = 1.0

        try:
            from imgui_bundle import (
                imgui,
                imguizmo,
            )
            from imgui_bundle.python_backends import pyglet_backend

            self.imgui = imgui
            self.giz = imguizmo.im_guizmo
            self.is_available = True
        except ImportError:
            self.is_available = False
            logger.warning("imgui_bundle not found. Install with: pip install imgui-bundle")
            return

        self.window = window
        self.imgui.create_context()
        try:
            # Create without callbacks so we can fix the scroll handler first
            self.impl = pyglet_backend.create_renderer(self.window, attach_callbacks=False)
        except Exception as e:
            # Unlikely to happen since RendererGL already sets PYOPENGL_PLATFORM=glx
            # on Wayland, but just in case the auto-detection missed the session type.
            if "no valid context" in str(e).lower() or "no current context" in str(e).lower():
                raise RuntimeError(
                    "Failed to initialize the OpenGL UI renderer. "
                    "If you are on Wayland, try setting the environment variable:\n\n"
                    "  PYOPENGL_PLATFORM=glx uv run -m newton.examples <example>\n"
                ) from e
            raise

        self.io = self.imgui.get_io()
        # Reuse a local UI font; no network, icon atlas, or extra dependency.
        for font_path in (
            "C:/Windows/Fonts/segoeui.ttf",
            "/System/Library/Fonts/SFNS.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ):
            if os.path.isfile(font_path):
                self.io.font_default = self.io.fonts.add_font_from_file_ttf(font_path, 14.0)
                break
        self.heading_font = None
        for font_path in (
            "C:/Windows/Fonts/seguisb.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ):
            if os.path.isfile(font_path):
                self.heading_font = self.io.fonts.add_font_from_file_ttf(font_path, 14.0)
                break
        self.number_font = None
        for font_path in (
            "C:/Windows/Fonts/consola.ttf",
            "/System/Library/Fonts/Menlo.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ):
            if os.path.isfile(font_path):
                self.number_font = self.io.fonts.add_font_from_file_ttf(font_path, 13.0)
                break
        # Menus, tabs and property editors should support Tab/arrow navigation
        # and Escape dismissal just like their native desktop counterparts.
        self.io.config_flags |= self.imgui.ConfigFlags_.nav_enable_keyboard

        # Fix inverted scroll direction in the pyglet imgui backend before
        # attaching callbacks so pyglet captures the corrected handler.
        # The replacement must be named "on_mouse_scroll" because pyglet
        # matches handlers by __name__.
        io = self.io

        def on_mouse_scroll(x, y, scroll_x, scroll_y):
            io.add_mouse_wheel_event(scroll_x, scroll_y)

        self.impl.on_mouse_scroll = on_mouse_scroll
        backend_key_press = self.impl.on_key_press

        def on_key_press(symbol: int, modifiers: int) -> bool | None:
            from pyglet import event
            from pyglet.window import key

            if self.key_press_filter is None or not self.key_press_filter(symbol, modifiers):
                backend_key_press(symbol, modifiers)
            # Pyglet's default Escape handler closes the window even when the
            # viewer callback defers to an editor or popup. Stop propagation
            # after forwarding the cancellation key to ImGui.
            if symbol == key.ESCAPE and (
                io.want_text_input or self.imgui.is_popup_open("", self.imgui.PopupFlags_.any_popup)
            ):
                return event.EVENT_HANDLED

        self.impl.on_key_press = on_key_press
        self.impl._attach_callbacks(self.window)

        # Set up proper DPI scaling for high-DPI displays.
        #
        # We can't rely solely on the framebuffer/window-size ratio: on macOS
        # with the default ``pyglet.options.dpi_scaling = "platform"`` both
        # ``get_size()`` and ``get_framebuffer_size()`` return *physical*
        # pixels, so the ratio is always 1.0 even on Retina (2x). The
        # documented pyglet API for HiDPI is ``window.scale`` which returns
        # ``backingScaleFactor()`` on macOS and ``dpi/96`` elsewhere. Combine
        # both so we always pick the larger non-trivial factor.
        self._refresh_io_metrics()
        self.dpi_scale = self._detect_dpi_scale()

        # ``_apply_dpi_scaling`` resets to the base dark style and then scales
        # it; no need to call ``_setup_dark_style`` separately here.
        self._apply_dpi_scaling()

    def _refresh_io_metrics(self) -> None:
        """Sync ``io.display_size`` and ``io.display_framebuffer_scale`` with
        the current pyglet window.

        Called on initial setup, resize, and DPI refreshes so the ImGui IO
        never lags behind pyglet's view of the window, including scale-only
        display transitions where ``on_resize`` isn't fired.
        """
        window_width, window_height = self.window.get_size()
        fb_width, fb_height = self.window.get_framebuffer_size()
        if window_width > 0 and window_height > 0:
            self.io.display_framebuffer_scale = (fb_width / window_width, fb_height / window_height)
        self.io.display_size = (fb_width, fb_height)

    def refresh_dpi(self, dpi_scale: float | None = None) -> float:
        """Refresh DPI-dependent ImGui state and return the resolved scale.

        Args:
            dpi_scale: Already-resolved DPI scale to apply. If omitted, the
                scale is detected from the window.
        """
        if not self.is_available:
            return self.dpi_scale
        # Keep ImGui IO metrics in sync even when ``on_resize`` is skipped —
        # ``display_framebuffer_scale`` in particular can change without a
        # framebuffer-size change when the backing scale factor flips.
        self._refresh_io_metrics()
        new_scale = self._coerce_dpi_scale(dpi_scale) if dpi_scale is not None else self._detect_dpi_scale()
        if abs(new_scale - self.dpi_scale) > 1e-3:
            self.dpi_scale = new_scale
            self._apply_dpi_scaling()
        return self.dpi_scale

    def _detect_dpi_scale(self) -> float:
        """Return the current DPI factor for the window.

        Prefers pyglet's documented ``window.scale`` (which works on macOS
        Retina where ``framebuffer_size == window_size``), also considering
        the framebuffer/window-size ratio when that is the larger signal.
        """
        scale = self._window_scale()
        try:
            get_size = self.window.get_size
            get_framebuffer_size = self.window.get_framebuffer_size
        except AttributeError:
            return max(1.0, scale)

        try:
            ww, wh = get_size()
            fw, fh = get_framebuffer_size()
            if ww > 0 and wh > 0:
                scale = max(scale, fw / ww, fh / wh)
        except AttributeError:
            return max(1.0, scale)
        return max(1.0, scale)

    def _window_scale(self) -> float:
        try:
            window_scale = self.window.scale
        except AttributeError:
            return 1.0

        return self._coerce_dpi_scale(window_scale)

    @staticmethod
    def _coerce_dpi_scale(value: float) -> float:
        try:
            return max(1.0, float(value))
        except (TypeError, ValueError):
            return 1.0

    def _apply_dpi_scaling(self) -> None:
        """Scale the active ImGui style and fonts to ``self.dpi_scale``.

        ``display_size`` is set to the framebuffer size, so ImGui works in
        framebuffer pixels. Without this rescale every widget would render at
        ~half size on Retina/4K displays. Re-applying first restores the base
        style (``_setup_dark_style``) so repeated calls do not compound.
        """
        if not self.is_available:
            return

        # Reset every size before scaling, including newer ImGui properties
        # that the palette setup does not customize (e.g. separator text).
        self.imgui.get_current_context().style = self.imgui.Style()
        self._setup_dark_style()
        style = self.imgui.get_style()
        style.window_padding = (10, 6)
        style.frame_padding = (6, 4)
        style.item_spacing = (6, 4)
        style.cell_padding = (4, 2)
        style.window_border_size = 1
        style.frame_rounding = 3
        style.frame_border_size = 0
        style.popup_rounding = 6
        style.grab_rounding = 3
        style.tab_rounding = 4
        style.scrollbar_size = 9
        style.scrollbar_rounding = 4
        palette = {
            "text": (0.88, 0.88, 0.88, 1),
            "text_disabled": (0.58, 0.58, 0.58, 1),
            "window_bg": (0.145, 0.145, 0.145, 1),
            "child_bg": (0.145, 0.145, 0.145, 1),
            "popup_bg": (0.18, 0.18, 0.18, 1),
            "border": (0.34, 0.34, 0.34, 1),
            "frame_bg": (0.29, 0.29, 0.29, 1),
            "frame_bg_hovered": (0.34, 0.34, 0.34, 1),
            "frame_bg_active": (0.37, 0.37, 0.37, 1),
            "button": (0.24, 0.24, 0.24, 1),
            "button_hovered": (0.30, 0.30, 0.30, 1),
            "button_active": (0.36, 0.36, 0.36, 1),
            "header": (0.29, 0.29, 0.29, 1),
            "header_hovered": (0.32, 0.32, 0.32, 1),
            "header_active": (0.37, 0.37, 0.37, 1),
            "check_mark": (0.96, 0.61, 0.32, 1),
            "checkbox_selected_bg": (0.32, 0.32, 0.32, 1),
            "slider_grab": (0.70, 0.70, 0.70, 1),
            "slider_grab_active": (0.96, 0.61, 0.32, 1),
            "tab": (0.18, 0.18, 0.18, 1),
            "tab_selected": (0.26, 0.26, 0.26, 1),
            "tab_hovered": (0.30, 0.30, 0.30, 1),
            "tab_selected_overline": (0.96, 0.61, 0.32, 1),
            "tab_dimmed": (0.18, 0.18, 0.18, 1),
            "tab_dimmed_selected": (0.26, 0.26, 0.26, 1),
            "tab_dimmed_selected_overline": (0.50, 0.50, 0.50, 1),
            "title_bg": (0.13, 0.13, 0.13, 1),
            "title_bg_active": (0.19, 0.19, 0.19, 1),
            "title_bg_collapsed": (0.13, 0.13, 0.13, 1),
            "menu_bar_bg": (0.18, 0.18, 0.18, 1),
            "scrollbar_bg": (0.12, 0.12, 0.12, 1),
            "scrollbar_grab": (0.31, 0.31, 0.31, 1),
            "scrollbar_grab_hovered": (0.43, 0.43, 0.43, 1),
            "scrollbar_grab_active": (0.53, 0.53, 0.53, 1),
            "separator": (0.32, 0.32, 0.32, 1),
            "separator_hovered": (0.55, 0.55, 0.55, 1),
            "separator_active": (0.96, 0.61, 0.32, 1),
            "table_header_bg": (0.18, 0.18, 0.18, 1),
            "table_border_strong": (0.25, 0.25, 0.25, 1),
            "table_border_light": (0.20, 0.20, 0.20, 1),
            "table_row_bg": (0, 0, 0, 0),
            "table_row_bg_alt": (1, 1, 1, 0.015),
            "text_selected_bg": (0.55, 0.55, 0.55, 0.45),
            "resize_grip": (0.55, 0.55, 0.55, 0.20),
            "resize_grip_hovered": (0.70, 0.70, 0.70, 0.60),
            "resize_grip_active": (0.96, 0.61, 0.32, 0.90),
            "nav_cursor": (0.96, 0.61, 0.32, 1),
            "input_text_cursor": (0.96, 0.61, 0.32, 1),
            "tree_lines": (0.35, 0.35, 0.35, 1),
            "docking_empty_bg": (0.13, 0.13, 0.13, 1),
            "docking_preview": (0.65, 0.65, 0.65, 0.45),
            "plot_lines": (0.72, 0.72, 0.72, 1),
        }
        for name, color in palette.items():
            style.set_color_(getattr(self.imgui.Col_, name), self.imgui.ImVec4(*color))
        if self.dpi_scale != 1.0:
            style.scale_all_sizes(self.dpi_scale)
        style.font_scale_dpi = self.dpi_scale

    def _setup_grey_style(self):
        if not self.is_available:
            return

        style = self.imgui.get_style()

        # Style properties
        style.alpha = 1.0
        # style.disabled_alpha = 0.5
        style.window_padding = (13.0, 10.0)
        style.window_rounding = 0.0
        style.window_border_size = 1.0
        style.window_min_size = (32.0, 32.0)
        style.window_title_align = (0.5, 0.5)
        style.window_menu_button_position = self.imgui.Dir_.right
        style.child_rounding = 3.0
        style.child_border_size = 1.0
        style.popup_rounding = 5.0
        style.popup_border_size = 1.0
        style.frame_padding = (20.0, 8.100000381469727)
        style.frame_rounding = 2.0
        style.frame_border_size = 0.0
        style.item_spacing = (3.0, 3.0)
        style.item_inner_spacing = (3.0, 8.0)
        style.cell_padding = (6.0, 14.10000038146973)
        style.indent_spacing = 0.0
        style.columns_min_spacing = 10.0
        style.scrollbar_size = 10.0
        style.scrollbar_rounding = 2.0
        style.grab_min_size = 12.10000038146973
        style.grab_rounding = 1.0
        style.tab_rounding = 2.0
        style.tab_border_size = 0.0
        style.color_button_position = self.imgui.Dir_.right
        style.button_text_align = (0.5, 0.5)
        style.selectable_text_align = (0.0, 0.0)

        # fmt: off
        # Colors
        style.set_color_(self.imgui.Col_.text, self.imgui.ImVec4(0.9803921580314636, 0.9803921580314636, 0.9803921580314636, 1.0))
        style.set_color_(self.imgui.Col_.text_disabled, self.imgui.ImVec4(0.4980392158031464, 0.4980392158031464, 0.4980392158031464, 1.0))
        style.set_color_(self.imgui.Col_.window_bg, self.imgui.ImVec4(0.09411764889955521, 0.09411764889955521, 0.09411764889955521, 1.0))
        style.set_color_(self.imgui.Col_.child_bg, self.imgui.ImVec4(0.1568627506494522, 0.1568627506494522, 0.1568627506494522, 1.0))
        style.set_color_(self.imgui.Col_.popup_bg, self.imgui.ImVec4(0.09411764889955521, 0.09411764889955521, 0.09411764889955521, 1.0))
        style.set_color_(self.imgui.Col_.border, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.09803921729326248))
        style.set_color_(self.imgui.Col_.border_shadow, self.imgui.ImVec4(0.0, 0.0, 0.0, 0.0))
        style.set_color_(self.imgui.Col_.frame_bg, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.09803921729326248))
        style.set_color_(self.imgui.Col_.frame_bg_hovered, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.1568627506494522))
        style.set_color_(self.imgui.Col_.frame_bg_active, self.imgui.ImVec4(0.0, 0.0, 0.0, 0.0470588244497776))
        style.set_color_(self.imgui.Col_.title_bg, self.imgui.ImVec4(0.1176470592617989, 0.1176470592617989, 0.1176470592617989, 1.0))
        style.set_color_(self.imgui.Col_.title_bg_active, self.imgui.ImVec4(0.1568627506494522, 0.1568627506494522, 0.1568627506494522, 1.0))
        style.set_color_(self.imgui.Col_.title_bg_collapsed, self.imgui.ImVec4(0.1176470592617989, 0.1176470592617989, 0.1176470592617989, 1.0))
        style.set_color_(self.imgui.Col_.menu_bar_bg, self.imgui.ImVec4(0.0, 0.0, 0.0, 0.0))
        style.set_color_(self.imgui.Col_.scrollbar_bg, self.imgui.ImVec4(0.0, 0.0, 0.0, 0.1098039224743843))
        style.set_color_(self.imgui.Col_.scrollbar_grab, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.3921568691730499))
        style.set_color_(self.imgui.Col_.scrollbar_grab_hovered, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.4705882370471954))
        style.set_color_(self.imgui.Col_.scrollbar_grab_active, self.imgui.ImVec4(0.0, 0.0, 0.0, 0.09803921729326248))
        style.set_color_(self.imgui.Col_.check_mark, self.imgui.ImVec4(1.0, 1.0, 1.0, 1.0))
        style.set_color_(self.imgui.Col_.slider_grab, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.3921568691730499))
        style.set_color_(self.imgui.Col_.slider_grab_active, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.3137255012989044))
        style.set_color_(self.imgui.Col_.button, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.09803921729326248))
        style.set_color_(self.imgui.Col_.button_hovered, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.1568627506494522))
        style.set_color_(self.imgui.Col_.button_active, self.imgui.ImVec4(0.0, 0.0, 0.0, 0.0470588244497776))
        style.set_color_(self.imgui.Col_.header, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.09803921729326248))
        style.set_color_(self.imgui.Col_.header_hovered, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.1568627506494522))
        style.set_color_(self.imgui.Col_.header_active, self.imgui.ImVec4(0.0, 0.0, 0.0, 0.0470588244497776))
        style.set_color_(self.imgui.Col_.separator, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.1568627506494522))
        style.set_color_(self.imgui.Col_.separator_hovered, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.2352941185235977))
        style.set_color_(self.imgui.Col_.separator_active, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.2352941185235977))
        style.set_color_(self.imgui.Col_.resize_grip, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.1568627506494522))
        style.set_color_(self.imgui.Col_.resize_grip_hovered, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.2352941185235977))
        style.set_color_(self.imgui.Col_.resize_grip_active, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.2352941185235977))
        style.set_color_(self.imgui.Col_.tab, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.09803921729326248))
        style.set_color_(self.imgui.Col_.tab_hovered, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.1568627506494522))
        style.set_color_(self.imgui.Col_.tab_selected, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.3137255012989044))
        style.set_color_(self.imgui.Col_.tab_dimmed, self.imgui.ImVec4(0.0, 0.0, 0.0, 0.1568627506494522))
        style.set_color_(self.imgui.Col_.tab_dimmed_selected, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.2352941185235977))
        style.set_color_(self.imgui.Col_.plot_lines, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.3529411852359772))
        style.set_color_(self.imgui.Col_.plot_lines_hovered, self.imgui.ImVec4(1.0, 1.0, 1.0, 1.0))
        style.set_color_(self.imgui.Col_.plot_histogram, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.3529411852359772))
        style.set_color_(self.imgui.Col_.plot_histogram_hovered, self.imgui.ImVec4(1.0, 1.0, 1.0, 1.0))
        style.set_color_(self.imgui.Col_.table_header_bg, self.imgui.ImVec4(0.1568627506494522, 0.1568627506494522, 0.1568627506494522, 1.0))
        style.set_color_(self.imgui.Col_.table_border_strong, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.3137255012989044))
        style.set_color_(self.imgui.Col_.table_border_light, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.196078434586525))
        style.set_color_(self.imgui.Col_.table_row_bg, self.imgui.ImVec4(0.0, 0.0, 0.0, 0.0))
        style.set_color_(self.imgui.Col_.table_row_bg_alt, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.01960784383118153))
        style.set_color_(self.imgui.Col_.text_selected_bg, self.imgui.ImVec4(0.0, 0.0, 0.0, 1.0))
        style.set_color_(self.imgui.Col_.drag_drop_target, self.imgui.ImVec4(0.168627455830574, 0.2313725501298904, 0.5372549295425415, 1.0))
        style.set_color_(self.imgui.Col_.nav_cursor, self.imgui.ImVec4(1.0, 1.0, 1.0, 1.0))
        style.set_color_(self.imgui.Col_.nav_windowing_highlight, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.699999988079071))
        style.set_color_(self.imgui.Col_.nav_windowing_dim_bg, self.imgui.ImVec4(0.800000011920929, 0.800000011920929, 0.800000011920929, 0.2000000029802322))
        style.set_color_(self.imgui.Col_.modal_window_dim_bg, self.imgui.ImVec4(0.0, 0.0, 0.0, 0.5647059082984924))
        # fmt: on

    def _setup_dark_style(self):
        if not self.is_available:
            return

        style = self.imgui.get_style()

        # Style properties
        style.alpha = 1.0
        # style.disabled_alpha = 1.0
        style.window_padding = (12.0, 12.0)
        style.window_rounding = 0.0
        style.window_border_size = 0.0
        style.window_min_size = (20.0, 20.0)
        style.window_title_align = (0.5, 0.5)
        style.window_menu_button_position = self.imgui.Dir_.none
        style.child_rounding = 0.0
        style.child_border_size = 1.0
        style.popup_rounding = 0.0
        style.popup_border_size = 1.0
        style.frame_padding = (6.0, 6.0)
        style.frame_rounding = 0.0
        style.frame_border_size = 0.0
        style.item_spacing = (12.0, 6.0)
        style.item_inner_spacing = (6.0, 3.0)
        style.cell_padding = (12.0, 6.0)
        style.indent_spacing = 20.0
        style.columns_min_spacing = 6.0
        style.scrollbar_size = 12.0
        style.scrollbar_rounding = 0.0
        style.grab_min_size = 12.0
        style.grab_rounding = 0.0
        style.tab_rounding = 0.0
        style.tab_border_size = 0.0
        # style.tab_min_width_for_close_button = 0.0  # Not available in imgui_bundle
        style.color_button_position = self.imgui.Dir_.right
        style.button_text_align = (0.5, 0.5)
        style.selectable_text_align = (0.0, 0.0)

        # fmt: off

        # Colors
        style.set_color_(self.imgui.Col_.text, self.imgui.ImVec4(1.0, 1.0, 1.0, 1.0))
        style.set_color_(self.imgui.Col_.text_disabled, self.imgui.ImVec4(0.2745098173618317, 0.3176470696926117, 0.4509803950786591, 1.0))
        style.set_color_(self.imgui.Col_.window_bg, self.imgui.ImVec4(0.0784313753247261, 0.08627451211214066, 0.1019607856869698, 1.0))
        style.set_color_(self.imgui.Col_.child_bg, self.imgui.ImVec4(0.0784313753247261, 0.08627451211214066, 0.1019607856869698, 1.0))
        style.set_color_(self.imgui.Col_.popup_bg, self.imgui.ImVec4(0.0784313753247261, 0.08627451211214066, 0.1019607856869698, 1.0))
        style.set_color_(self.imgui.Col_.border, self.imgui.ImVec4(0.1568627506494522, 0.168627455830574, 0.1921568661928177, 1.0))
        style.set_color_(self.imgui.Col_.border_shadow, self.imgui.ImVec4(0.0784313753247261, 0.08627451211214066, 0.1019607856869698, 1.0))
        style.set_color_(self.imgui.Col_.frame_bg, self.imgui.ImVec4(0.1176470592617989, 0.1333333402872086, 0.1490196138620377, 1.0))
        style.set_color_(self.imgui.Col_.frame_bg_hovered, self.imgui.ImVec4(0.1568627506494522, 0.168627455830574, 0.1921568661928177, 1.0))
        style.set_color_(self.imgui.Col_.frame_bg_active, self.imgui.ImVec4(0.2352941185235977, 0.2156862765550613, 0.5960784554481506, 1.0))
        style.set_color_(self.imgui.Col_.title_bg, self.imgui.ImVec4(0.0470588244497776, 0.05490196123719215, 0.07058823853731155, 1.0))
        style.set_color_(self.imgui.Col_.title_bg_active, self.imgui.ImVec4(0.0470588244497776, 0.05490196123719215, 0.07058823853731155, 1.0))
        style.set_color_(self.imgui.Col_.title_bg_collapsed, self.imgui.ImVec4(0.0784313753247261, 0.08627451211214066, 0.1019607856869698, 1.0))
        style.set_color_(self.imgui.Col_.menu_bar_bg, self.imgui.ImVec4(0.09803921729326248, 0.105882354080677, 0.1215686276555061, 1.0))
        style.set_color_(self.imgui.Col_.scrollbar_bg, self.imgui.ImVec4(0.0470588244497776, 0.05490196123719215, 0.07058823853731155, 1.0))
        style.set_color_(self.imgui.Col_.scrollbar_grab, self.imgui.ImVec4(0.1176470592617989, 0.1333333402872086, 0.1490196138620377, 1.0))
        style.set_color_(self.imgui.Col_.scrollbar_grab_hovered, self.imgui.ImVec4(0.1568627506494522, 0.168627455830574, 0.1921568661928177, 1.0))
        style.set_color_(self.imgui.Col_.scrollbar_grab_active, self.imgui.ImVec4(0.1176470592617989, 0.1333333402872086, 0.1490196138620377, 1.0))
        style.set_color_(self.imgui.Col_.check_mark, self.imgui.ImVec4(0.4980392158031464, 0.5137255191802979, 1.0, 1.0))
        style.set_color_(self.imgui.Col_.slider_grab, self.imgui.ImVec4(0.4980392158031464, 0.5137255191802979, 1.0, 1.0))
        style.set_color_(self.imgui.Col_.slider_grab_active, self.imgui.ImVec4(0.5372549295425415, 0.5529412031173706, 1.0, 1.0))
        style.set_color_(self.imgui.Col_.button, self.imgui.ImVec4(0.1176470592617989, 0.1333333402872086, 0.1490196138620377, 1.0))
        style.set_color_(self.imgui.Col_.button_hovered, self.imgui.ImVec4(0.196078434586525, 0.1764705926179886, 0.5450980663299561, 1.0))
        style.set_color_(self.imgui.Col_.button_active, self.imgui.ImVec4(0.2352941185235977, 0.2156862765550613, 0.5960784554481506, 1.0))
        style.set_color_(self.imgui.Col_.header, self.imgui.ImVec4(0.1176470592617989, 0.1333333402872086, 0.1490196138620377, 1.0))
        style.set_color_(self.imgui.Col_.header_hovered, self.imgui.ImVec4(0.196078434586525, 0.1764705926179886, 0.5450980663299561, 1.0))
        style.set_color_(self.imgui.Col_.header_active, self.imgui.ImVec4(0.2352941185235977, 0.2156862765550613, 0.5960784554481506, 1.0))
        style.set_color_(self.imgui.Col_.separator, self.imgui.ImVec4(0.1568627506494522, 0.1843137294054031, 0.250980406999588, 1.0))
        style.set_color_(self.imgui.Col_.separator_hovered, self.imgui.ImVec4(0.1568627506494522, 0.1843137294054031, 0.250980406999588, 1.0))
        style.set_color_(self.imgui.Col_.separator_active, self.imgui.ImVec4(0.1568627506494522, 0.1843137294054031, 0.250980406999588, 1.0))
        # Resize grip uses a translucent white instead of a near-window-bg
        # color so the corner handle is actually discoverable at rest. Hover
        # / active states keep the accent-blue look.
        style.set_color_(self.imgui.Col_.resize_grip, self.imgui.ImVec4(1.0, 1.0, 1.0, 0.20))
        style.set_color_(self.imgui.Col_.resize_grip_hovered, self.imgui.ImVec4(0.4980392158031464, 0.5137255191802979, 1.0, 0.85))
        style.set_color_(self.imgui.Col_.resize_grip_active, self.imgui.ImVec4(0.5372549295425415, 0.5529412031173706, 1.0, 1.0))
        style.set_color_(self.imgui.Col_.tab, self.imgui.ImVec4(0.0470588244497776, 0.05490196123719215, 0.07058823853731155, 1.0))
        style.set_color_(self.imgui.Col_.tab_hovered, self.imgui.ImVec4(0.1176470592617989, 0.1333333402872086, 0.1490196138620377, 1.0))
        style.set_color_(self.imgui.Col_.tab_selected, self.imgui.ImVec4(0.09803921729326248, 0.105882354080677, 0.1215686276555061, 1.0))
        style.set_color_(self.imgui.Col_.tab_dimmed, self.imgui.ImVec4(0.0470588244497776, 0.05490196123719215, 0.07058823853731155, 1.0))
        style.set_color_(self.imgui.Col_.tab_dimmed_selected, self.imgui.ImVec4(0.0784313753247261, 0.08627451211214066, 0.1019607856869698, 1.0))
        style.set_color_(self.imgui.Col_.plot_lines, self.imgui.ImVec4(0.5215686559677124, 0.6000000238418579, 0.7019608020782471, 1.0))
        style.set_color_(self.imgui.Col_.plot_lines_hovered, self.imgui.ImVec4(0.03921568766236305, 0.9803921580314636, 0.9803921580314636, 1.0))
        style.set_color_(self.imgui.Col_.plot_histogram, self.imgui.ImVec4(1.0, 0.2901960909366608, 0.5960784554481506, 1.0))
        style.set_color_(self.imgui.Col_.plot_histogram_hovered, self.imgui.ImVec4(0.9960784316062927, 0.4745098054409027, 0.6980392336845398, 1.0))
        style.set_color_(self.imgui.Col_.table_header_bg, self.imgui.ImVec4(0.0470588244497776, 0.05490196123719215, 0.07058823853731155, 1.0))
        style.set_color_(self.imgui.Col_.table_border_strong, self.imgui.ImVec4(0.0470588244497776, 0.05490196123719215, 0.07058823853731155, 1.0))
        style.set_color_(self.imgui.Col_.table_border_light, self.imgui.ImVec4(0.0, 0.0, 0.0, 1.0))
        style.set_color_(self.imgui.Col_.table_row_bg, self.imgui.ImVec4(0.1176470592617989, 0.1333333402872086, 0.1490196138620377, 1.0))
        style.set_color_(self.imgui.Col_.table_row_bg_alt, self.imgui.ImVec4(0.09803921729326248, 0.105882354080677, 0.1215686276555061, 1.0))
        style.set_color_(self.imgui.Col_.text_selected_bg, self.imgui.ImVec4(0.2352941185235977, 0.2156862765550613, 0.5960784554481506, 1.0))
        style.set_color_(self.imgui.Col_.drag_drop_target, self.imgui.ImVec4(0.4980392158031464, 0.5137255191802979, 1.0, 1.0))
        style.set_color_(self.imgui.Col_.nav_cursor, self.imgui.ImVec4(0.4980392158031464, 0.5137255191802979, 1.0, 1.0))
        style.set_color_(self.imgui.Col_.nav_windowing_highlight, self.imgui.ImVec4(0.4980392158031464, 0.5137255191802979, 1.0, 1.0))
        style.set_color_(self.imgui.Col_.nav_windowing_dim_bg, self.imgui.ImVec4(0.196078434586525, 0.1764705926179886, 0.5450980663299561, 0.501960813999176))
        style.set_color_(self.imgui.Col_.modal_window_dim_bg, self.imgui.ImVec4(0.196078434586525, 0.1764705926179886, 0.5450980663299561, 0.501960813999176))
        # fmt: on

    def begin_frame(self):
        """Renders a single frame of the UI. This should be called from the main render loop."""
        if not self.is_available:
            return

        try:
            self.impl.process_inputs()
        except AttributeError:
            # Older integrations may not require this
            pass

        # Some Pyglet integrations treat clock.tick() (an elapsed duration) as
        # an absolute timestamp. Supply a monotonic frame delta so hover delays,
        # key repeat and UI animations track real time at steady frame rates.
        now = perf_counter()
        self.io.delta_time = max(now - self._frame_time, 1e-6) if self._frame_time is not None else 1.0 / 60.0
        self._frame_time = now
        self.imgui.new_frame()
        self._frame_recovery_state = self.imgui.internal.ErrorRecoveryState()
        self.imgui.internal.error_recovery_store_state(self._frame_recovery_state)
        self.giz.begin_frame()

    def _recover_state(self, state: imgui.internal.ErrorRecoveryState) -> None:
        """Unwind an interrupted UI callback without obscuring its exception."""
        ui = self.imgui
        io = ui.get_io()
        options = (
            "config_error_recovery_enable_assert",
            "config_error_recovery_enable_debug_log",
            "config_error_recovery_enable_tooltip",
        )
        saved = [getattr(io, option) for option in options]
        try:
            for option in options:
                setattr(io, option, False)
            ui.internal.error_recovery_try_to_recover_state(state)
        finally:
            for option, value in zip(options, saved, strict=True):
                setattr(io, option, value)

    def abort_frame(self) -> None:
        """Release unfinished windows and stacks before the next viewer frame."""
        if self._frame_recovery_state is not None:
            self._recover_state(self._frame_recovery_state)
            self.imgui.end_frame()
            self._frame_recovery_state = None

    def draw_section(self, name: str, callback: Callable[[], None]) -> None:
        """Isolate Python UI failures and restore ImGui stacks before continuing."""
        ui = self.imgui
        if name not in self.section_errors:
            state = ui.internal.ErrorRecoveryState()
            ui.internal.error_recovery_store_state(state)
            try:
                callback()
            except (Exception, SystemExit) as error:
                self.section_errors[name] = f"{type(error).__name__}: {error}"
                logger.exception("Failed to draw %s", name)
                # Report the original exception once. Stack recovery would
                # otherwise assert on the unclosed widgets left by that error.
                self._recover_state(state)
        if name in self.section_errors:
            ui.text_wrapped(f"{name} unavailable\n{self.section_errors[name]}")
            if ui.button("Retry##section:" + name):
                del self.section_errors[name]

    def end_frame(self):
        if not self.is_available:
            return

        self._poll_file_dialog()
        self.imgui.render()
        self.imgui.end_frame()
        self._frame_recovery_state = None

    def render(self):
        if not self.is_available:
            return

        self.impl.render(self.imgui.get_draw_data())

    def is_capturing_mouse(self):
        if not self.is_available:
            return False

        return self.io.want_capture_mouse

    def is_capturing_keyboard(self):
        if not self.is_available:
            return False

        return self.io.want_capture_keyboard

    def is_capturing(self):
        if not self.is_available:
            return False

        return self.is_capturing_mouse() or self.is_capturing_keyboard()

    def resize(self, width, height):
        if not self.is_available:
            return

        self._refresh_io_metrics()

        # Reapply style/font scaling only when the DPI actually changes —
        # e.g. the window moved to a display with different scaling.
        new_dpi_scale = self._detect_dpi_scale()
        if abs(new_dpi_scale - self.dpi_scale) > 1e-3:
            self.dpi_scale = new_dpi_scale
            self._apply_dpi_scaling()

    def get_theme_color(self, color_id, fallback_color=(1.0, 1.0, 1.0, 1.0)):
        """Get a color from the current theme with fallback.

        Args:
            color_id: ImGui color constant (e.g., self.imgui.Col_.text_disabled)
            fallback_color: RGBA tuple to use if color not available

        Returns:
            RGBA tuple of the theme color or fallback
        """
        if not self.is_available:
            return fallback_color

        try:
            style = self.imgui.get_style()
            color = style.color_(color_id)
            return (color.x, color.y, color.z, color.w)
        except (AttributeError, KeyError, IndexError):
            return fallback_color

    def consume_file_dialog_result(self) -> str | None:
        """Return the latest completed file dialog path once.

        File dialogs are asynchronous: `open_load_file_dialog()` and
        `open_save_file_dialog()` queue a native dialog and return immediately.
        Poll this method from the render loop to retrieve the selected path.
        """
        if not self.is_available:
            return None

        result = self._file_dialog_result
        self._file_dialog_result = None
        return result

    def _poll_file_dialog(self):
        """Check if pending file dialog has completed."""
        if not self.is_available:
            return

        if self._pending_file_dialog is None:
            return
        if self._pending_file_dialog.ready():
            result = self._pending_file_dialog.result()
            if result:
                if isinstance(result, list):
                    if len(result) == 1:
                        self._file_dialog_result = result[0]
                    elif len(result) > 1:
                        logger.warning("multiple files selected; expected a single file.")
                else:
                    self._file_dialog_result = result
            self._pending_file_dialog = None

    def open_save_file_dialog(self, title: str = "Save File") -> None:
        """Start an asynchronous native OS save-file dialog.

        Use `consume_file_dialog_result()` to retrieve the selected path.
        """
        if not self.is_available:
            return

        try:
            from imgui_bundle import portable_file_dialogs as pfd

            self._pending_file_dialog = pfd.save_file(title, os.getcwd())
        except ImportError:
            logger.warning("portable_file_dialogs not available")

    def open_load_file_dialog(self, title: str = "Open File") -> None:
        """Start an asynchronous native OS open-file dialog.

        Use `consume_file_dialog_result()` to retrieve the selected path.
        """
        if not self.is_available:
            return

        try:
            from imgui_bundle import portable_file_dialogs as pfd

            self._pending_file_dialog = pfd.open_file(title, os.getcwd())
        except ImportError:
            logger.warning("portable_file_dialogs not available")

    def shutdown(self):
        if not self.is_available:
            return

        self.impl.shutdown()
