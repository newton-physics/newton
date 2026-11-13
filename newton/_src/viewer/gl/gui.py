# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import string
from time import perf_counter

logger = logging.getLogger(__name__)

# pyglet key name -> Dear ImGui key name
_KEY_NAMES = {
    "TAB": "tab",
    "BACKSPACE": "backspace",
    "RETURN": "enter",
    "ESCAPE": "escape",
    "INSERT": "insert",
    "DELETE": "delete",
    "SPACE": "space",
    "LEFT": "left_arrow",
    "RIGHT": "right_arrow",
    "UP": "up_arrow",
    "DOWN": "down_arrow",
    "PAGEUP": "page_up",
    "PAGEDOWN": "page_down",
    "HOME": "home",
    "END": "end",
    "NUM_ENTER": "keypad_enter",
    "NUM_DECIMAL": "keypad_decimal",
    "NUM_DIVIDE": "keypad_divide",
    "NUM_MULTIPLY": "keypad_multiply",
    "NUM_SUBTRACT": "keypad_subtract",
    "NUM_ADD": "keypad_add",
    "LCTRL": "left_ctrl",
    "RCTRL": "right_ctrl",
    "LSHIFT": "left_shift",
    "RSHIFT": "right_shift",
    "LALT": "left_alt",
    "RALT": "right_alt",
    "LCOMMAND": "left_super",
    "RCOMMAND": "right_super",
    "GRAVE": "grave_accent",
    "MINUS": "minus",
    "EQUAL": "equal",
    "BRACKETLEFT": "left_bracket",
    "BRACKETRIGHT": "right_bracket",
    "BACKSLASH": "backslash",
    "SEMICOLON": "semicolon",
    "APOSTROPHE": "apostrophe",
    "COMMA": "comma",
    "PERIOD": "period",
    "SLASH": "slash",
    "PRINT": "print_screen",
    "PAUSE": "pause",
    "CAPSLOCK": "caps_lock",
    "NUMLOCK": "num_lock",
    "SCROLLLOCK": "scroll_lock",
    **{f"F{i}": f"f{i}" for i in range(1, 13)},
    **{f"_{i}": f"_{i}" for i in range(10)},
    **{f"NUM_{i}": f"keypad{i}" for i in range(10)},
    **{letter: letter.lower() for letter in string.ascii_uppercase},
}
_MODIFIER_NAMES = {"ctrl": "mod_ctrl", "shift": "mod_shift", "alt": "mod_alt", "super": "mod_super"}
# Dear ImGui cursor name -> pyglet window cursor name
_CURSOR_NAMES = {
    "arrow": "CURSOR_DEFAULT",
    "text_input": "CURSOR_TEXT",
    "resize_all": "CURSOR_SIZE",
    "resize_ns": "CURSOR_SIZE_UP_DOWN",
    "resize_ew": "CURSOR_SIZE_LEFT_RIGHT",
    "resize_nesw": "CURSOR_SIZE_DOWN_LEFT",
    "resize_nwse": "CURSOR_SIZE_DOWN_RIGHT",
    "hand": "CURSOR_HAND",
}


class UI:
    def __init__(self, window):
        self._file_dialog_result: str | None = None
        self._pending_file_dialog = None
        self._frame_time: float | None = None
        # DPI scale applied to ImGui style and fonts. Updated by ``_apply_dpi_scaling``
        # whenever the framebuffer-to-window ratio changes (e.g. moving the window
        # between displays with different DPI).
        self.dpi_scale: float = 1.0

        try:
            from imgui_bundle import (
                imgui,
                imguizmo,
            )

            self.imgui = imgui
            self.giz = imguizmo.im_guizmo
            self.is_available = True
        except ImportError as e:
            self.is_available = False
            logger.warning(
                'Viewer GUI is unavailable: %s. Install its dependencies with `pip install "newton[examples]"`.', e
            )
            return

        self.window = window
        # The native renderer binds to one context, and create_context() keeps an earlier viewer's current.
        self._context = self.imgui.create_context()
        self.imgui.set_current_context(self._context)
        self.io = self.imgui.get_io()
        # Dear ImGui's native renderer, so the GUI needs no Python OpenGL bindings besides pyglet's.
        if not self.imgui.backends.opengl3_init("#version 330"):
            raise RuntimeError("Failed to initialize the OpenGL UI renderer.")
        self._attach_input()

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

        # Reset to ImGui defaults, apply the dark theme, then scale style and fonts.
        self._apply_dpi_scaling()

    def _attach_input(self) -> None:
        """Forward pyglet window input to Dear ImGui."""
        from pyglet.window import key, mouse

        imgui, io, window = self.imgui, self.io, self.window
        io.backend_flags |= imgui.BackendFlags_.has_mouse_cursors
        keys = {getattr(key, name): getattr(imgui.Key, value) for name, value in _KEY_NAMES.items()}
        modifiers = {
            getattr(imgui.Key, f"{side}_{name}"): getattr(imgui.Key, modifier)
            for name, modifier in _MODIFIER_NAMES.items()
            for side in ("left", "right")
        }
        buttons = {mouse.LEFT: 0, mouse.RIGHT: 1, mouse.MIDDLE: 2}
        self._cursors = {
            int(getattr(imgui.MouseCursor_, name)): getattr(window, cursor) for name, cursor in _CURSOR_NAMES.items()
        }
        self._cursor: int | None = None
        held_modifiers = set()

        def on_key(symbol, down):
            imgui_key = keys.get(symbol)
            if imgui_key is not None:
                io.add_key_event(imgui_key, down)
                modifier = modifiers.get(imgui_key)
                if modifier is not None:
                    # The modifier stays down while either of its keys is held.
                    (held_modifiers.add if down else held_modifiers.discard)(imgui_key)
                    io.add_key_event(modifier, any(modifiers[held] == modifier for held in held_modifiers))

        def on_mouse_button(button, down):
            if button in buttons:
                io.add_mouse_button_event(buttons[button], down)

        def on_text(text):
            for char in text:
                io.add_input_character(ord(char))

        def on_mouse_position(x, y, *_):
            io.add_mouse_pos_event(x, io.display_size.y - y)

        window.push_handlers(
            on_key_press=lambda symbol, _modifiers: on_key(symbol, True),
            on_key_release=lambda symbol, _modifiers: on_key(symbol, False),
            on_text=on_text,
            on_mouse_motion=on_mouse_position,
            # Button transitions belong to press/release; a press on every drag motion restarts cancelled drags.
            on_mouse_drag=on_mouse_position,
            on_mouse_press=lambda _x, _y, button, _modifiers: on_mouse_button(button, True),
            on_mouse_release=lambda _x, _y, button, _modifiers: on_mouse_button(button, False),
            on_mouse_scroll=lambda _x, _y, scroll_x, scroll_y: io.add_mouse_wheel_event(scroll_x, scroll_y),
            on_resize=lambda _width, _height: self._refresh_io_metrics(),
            # Release the renderer while the window still has its GL context.
            on_close=self.shutdown,
        )

    def _update_mouse_cursor(self) -> None:
        """Show the cursor shape Dear ImGui asks for, such as a resize arrow."""
        if self.io.config_flags & self.imgui.ConfigFlags_.no_mouse_cursor_change:
            return
        cursor = int(self.imgui.get_mouse_cursor())
        if cursor == self._cursor:
            return
        self._cursor = cursor
        visible = cursor != int(self.imgui.MouseCursor_.none)
        self.window.set_mouse_visible(visible)
        if visible:
            shape = self._cursors.get(cursor, self.window.CURSOR_DEFAULT)
            self.window.set_mouse_cursor(self.window.get_system_mouse_cursor(shape))

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
        ~half size on Retina/4K displays. Re-applying first resets to ImGui
        defaults, then applies the dark theme (``_setup_dark_style``) before
        scaling, so fields outside the theme do not compound either.
        """
        if not self.is_available:
            return

        self.imgui.set_current_context(self._context)
        self._context.style = self.imgui.Style()
        self._setup_dark_style()
        style = self.imgui.get_style()
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

    def cancel_mouse_interaction(self) -> None:
        """Discard pending input and release widgets without completing a click."""
        self.io.clear_events_queue()
        self.io.clear_input_mouse()
        self.imgui.set_current_context(self._context)
        self.imgui.internal.clear_active_id()
        self._context.want_capture_mouse_next_frame = -1
        self.io.want_capture_mouse = False

    def begin_frame(self) -> None:
        """Renders a single frame of the UI. This should be called from the main render loop."""
        if not self.is_available:
            return

        now = perf_counter()
        self.io.delta_time = max(now - self._frame_time, 1e-6) if self._frame_time is not None else 1.0 / 60.0
        self._frame_time = now

        self.imgui.set_current_context(self._context)
        self.imgui.new_frame()
        self.giz.begin_frame()

    def end_frame(self):
        if not self.is_available:
            return

        self._poll_file_dialog()
        self.imgui.render()
        self.imgui.end_frame()

    def render(self):
        if not self.is_available:
            return

        self.imgui.set_current_context(self._context)
        self.imgui.backends.opengl3_new_frame()
        self.imgui.backends.opengl3_render_draw_data(self.imgui.get_draw_data())
        self._update_mouse_cursor()

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

        self.is_available = False
        if getattr(self.window, "context", None) is None:
            return  # the window is gone, and its GL objects with it
        self.window.switch_to()
        self.imgui.set_current_context(self._context)
        self.imgui.backends.opengl3_shutdown()
        self.imgui.destroy_context(self._context)
