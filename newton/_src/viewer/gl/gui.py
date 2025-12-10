# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations


def _patch_pyopengl_context(window):
    """
    Patch PyOpenGL's context management to work with Pyglet.

    This is needed because PyOpenGL's context detection doesn't work
    with Pyglet's OpenGL context management out of the box.
    """
    try:
        from OpenGL import contextdata  # noqa: PLC0415

        # Create a context wrapper that PyOpenGL can use
        class PygletContext:
            def __init__(self, pyglet_window):
                self.window = pyglet_window
                self._context_id = id(pyglet_window.context)

            def __hash__(self):
                return self._context_id

            def __eq__(self, other):
                return isinstance(other, PygletContext) and self._context_id == other._context_id

        # Store the original getContext function
        if not hasattr(contextdata, "_original_getContext"):
            contextdata._original_getContext = contextdata.getContext

            # Monkey patch getContext to return our Pyglet-compatible context
            def patched_getContext(context=None):
                if context is None:
                    # Return a context based on the current Pyglet window
                    return PygletContext(window)
                return contextdata._original_getContext(context)

            contextdata.getContext = patched_getContext

    except ImportError:
        # If OpenGL is not installed, no need to patch
        pass


class UI:
    def __init__(self, window):
        try:
            from imgui_bundle import (  # noqa: PLC0415
                imgui,
                imguizmo,
            )
            from imgui_bundle.python_backends import pyglet_backend  # noqa: PLC0415

            self.imgui = imgui
            self.giz = imguizmo.im_guizmo
            self.is_available = True
        except ImportError:
            self.is_available = False
            print("Warning: imgui_bundle not found. Install with: pip install imgui-bundle")
            return

        self.window = window

        # Patch PyOpenGL context management before creating the renderer
        _patch_pyopengl_context(window)

        self.imgui.create_context()
        self.impl = pyglet_backend.create_renderer(self.window)

        self.io = self.imgui.get_io()

        # Set up proper DPI scaling for high-DPI displays
        window_width, window_height = self.window.get_size()
        fb_width, fb_height = self.window.get_framebuffer_size()
        if window_width > 0 and window_height > 0:
            scale_x = fb_width / window_width
            scale_y = fb_height / window_height
            self.io.display_framebuffer_scale = (scale_x, scale_y)
            self.io.display_size = (fb_width, fb_height)

        self._setup_dark_style()

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
        style.set_color_(self.imgui.Col_.resize_grip, self.imgui.ImVec4(0.1176470592617989, 0.1333333402872086, 0.1490196138620377, 1.0))
        style.set_color_(self.imgui.Col_.resize_grip_hovered, self.imgui.ImVec4(0.196078434586525, 0.1764705926179886, 0.5450980663299561, 1.0))
        style.set_color_(self.imgui.Col_.resize_grip_active, self.imgui.ImVec4(0.2352941185235977, 0.2156862765550613, 0.5960784554481506, 1.0))
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

        self.imgui.new_frame()
        self.giz.begin_frame()

    def end_frame(self):
        if not self.is_available:
            return

        self.imgui.render()
        self.imgui.end_frame()

    def render(self):
        if not self.is_available:
            return

        self.impl.render(self.imgui.get_draw_data())

    def is_capturing(self):
        if not self.is_available:
            return False

        return self.io.want_capture_mouse or self.io.want_capture_keyboard

    def resize(self, width, height):
        if not self.is_available:
            return

        # Get framebuffer size for proper DPI scaling
        fb_width, fb_height = self.window.get_framebuffer_size()

        # Update display framebuffer scale
        if width > 0 and height > 0:
            scale_x = fb_width / width
            scale_y = fb_height / height
            self.io.display_framebuffer_scale = (scale_x, scale_y)

        # Use framebuffer size for display size
        self.io.display_size = fb_width, fb_height

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

    def open_save_file_dialog(
        self,
        title: str = "Save File",
        defaultextension: str = "",
        filetypes: list[tuple[str, str]] | None = None,
    ) -> str | None:
        """Opens a file dialog for saving a file and returns the selected path."""
        try:
            import tkinter as tk  # noqa: PLC0415
            from tkinter import filedialog  # noqa: PLC0415
        except ImportError:
            print("Warning: tkinter not found. To use the file dialog, please install it.")
            return None

        try:
            root = tk.Tk()
        except tk.TclError:
            print("Warning: no display found - cannot open file dialog.")
            return None

        root.withdraw()  # Hide the main window
        file_path = filedialog.asksaveasfilename(
            defaultextension=defaultextension,
            filetypes=filetypes or [("All Files", "*.*")],
            title=title,
        )
        root.destroy()
        return file_path

    def open_load_file_dialog(
        self, title: str = "Open File", filetypes: list[tuple[str, str]] | None = None
    ) -> str | None:
        """Opens a file dialog for loading a file and returns the selected path."""
        try:
            import tkinter as tk  # noqa: PLC0415
            from tkinter import filedialog  # noqa: PLC0415
        except ImportError:
            print("Warning: tkinter not found. To use the file dialog, please install it.")
            return None

        try:
            root = tk.Tk()
        except tk.TclError:
            print("Warning: no display found - cannot open file dialog.")
            return None

        root.withdraw()  # Hide the main window
        file_path = filedialog.askopenfilename(
            filetypes=filetypes or [("All Files", "*.*")],
            title=title,
        )
        root.destroy()
        return file_path

    def shutdown(self):
        if not self.is_available:
            return

        self.impl.shutdown()
