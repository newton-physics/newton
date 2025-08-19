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


class UI:
    def __init__(self, window):
        try:
            import imgui  # noqa: PLC0415
            from imgui.integrations.pyglet import PygletProgrammablePipelineRenderer  # noqa: PLC0415

            self.imgui = imgui
            self.is_available = True
        except ImportError:
            self.is_available = False
            print('Warning: imgui not found. To use the UI, please install it with: pip install "imgui[pyglet]"')
            return

        self.window = window
        self.imgui.create_context()
        self.impl = PygletProgrammablePipelineRenderer(self.window)

        self.io = self.imgui.get_io()

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
        style.window_menu_button_position = self.imgui.DIRECTION_RIGHT
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
        style.tab_min_width_for_close_button = 5.0
        style.color_button_position = self.imgui.DIRECTION_RIGHT
        style.button_text_align = (0.5, 0.5)
        style.selectable_text_align = (0.0, 0.0)

        # fmt: off
        # Colors
        colors = style.colors
        colors[self.imgui.COLOR_TEXT] = (0.9803921580314636, 0.9803921580314636, 0.9803921580314636, 1.0)
        colors[self.imgui.COLOR_TEXT_DISABLED] = (0.4980392158031464, 0.4980392158031464, 0.4980392158031464, 1.0)
        colors[self.imgui.COLOR_WINDOW_BACKGROUND] = (0.09411764889955521, 0.09411764889955521, 0.09411764889955521, 1.0)
        colors[self.imgui.COLOR_CHILD_BACKGROUND] = (0.1568627506494522, 0.1568627506494522, 0.1568627506494522, 1.0)
        colors[self.imgui.COLOR_POPUP_BACKGROUND] = (0.09411764889955521, 0.09411764889955521, 0.09411764889955521, 1.0)
        colors[self.imgui.COLOR_BORDER] = (1.0, 1.0, 1.0, 0.09803921729326248)
        colors[self.imgui.COLOR_BORDER_SHADOW] = (0.0, 0.0, 0.0, 0.0)
        colors[self.imgui.COLOR_FRAME_BACKGROUND] = (1.0, 1.0, 1.0, 0.09803921729326248)
        colors[self.imgui.COLOR_FRAME_BACKGROUND_HOVERED] = (1.0, 1.0, 1.0, 0.1568627506494522)
        colors[self.imgui.COLOR_FRAME_BACKGROUND_ACTIVE] = (0.0, 0.0, 0.0, 0.0470588244497776)
        colors[self.imgui.COLOR_TITLE_BACKGROUND] = (0.1176470592617989, 0.1176470592617989, 0.1176470592617989, 1.0)
        colors[self.imgui.COLOR_TITLE_BACKGROUND_ACTIVE] = (0.1568627506494522, 0.1568627506494522, 0.1568627506494522, 1.0)
        colors[self.imgui.COLOR_TITLE_BACKGROUND_COLLAPSED] = (0.1176470592617989, 0.1176470592617989, 0.1176470592617989, 1.0)
        colors[self.imgui.COLOR_MENUBAR_BACKGROUND] = (0.0, 0.0, 0.0, 0.0)
        colors[self.imgui.COLOR_SCROLLBAR_BACKGROUND] = (0.0, 0.0, 0.0, 0.1098039224743843)
        colors[self.imgui.COLOR_SCROLLBAR_GRAB] = (1.0, 1.0, 1.0, 0.3921568691730499)
        colors[self.imgui.COLOR_SCROLLBAR_GRAB_HOVERED] = (1.0, 1.0, 1.0, 0.4705882370471954)
        colors[self.imgui.COLOR_SCROLLBAR_GRAB_ACTIVE] = (0.0, 0.0, 0.0, 0.09803921729326248)
        colors[self.imgui.COLOR_CHECK_MARK] = (1.0, 1.0, 1.0, 1.0)
        colors[self.imgui.COLOR_SLIDER_GRAB] = (1.0, 1.0, 1.0, 0.3921568691730499)
        colors[self.imgui.COLOR_SLIDER_GRAB_ACTIVE] = (1.0, 1.0, 1.0, 0.3137255012989044)
        colors[self.imgui.COLOR_BUTTON] = (1.0, 1.0, 1.0, 0.09803921729326248)
        colors[self.imgui.COLOR_BUTTON_HOVERED] = (1.0, 1.0, 1.0, 0.1568627506494522)
        colors[self.imgui.COLOR_BUTTON_ACTIVE] = (0.0, 0.0, 0.0, 0.0470588244497776)
        colors[self.imgui.COLOR_HEADER] = (1.0, 1.0, 1.0, 0.09803921729326248)
        colors[self.imgui.COLOR_HEADER_HOVERED] = (1.0, 1.0, 1.0, 0.1568627506494522)
        colors[self.imgui.COLOR_HEADER_ACTIVE] = (0.0, 0.0, 0.0, 0.0470588244497776)
        colors[self.imgui.COLOR_SEPARATOR] = (1.0, 1.0, 1.0, 0.1568627506494522)
        colors[self.imgui.COLOR_SEPARATOR_HOVERED] = (1.0, 1.0, 1.0, 0.2352941185235977)
        colors[self.imgui.COLOR_SEPARATOR_ACTIVE] = (1.0, 1.0, 1.0, 0.2352941185235977)
        colors[self.imgui.COLOR_RESIZE_GRIP] = (1.0, 1.0, 1.0, 0.1568627506494522)
        colors[self.imgui.COLOR_RESIZE_GRIP_HOVERED] = (1.0, 1.0, 1.0, 0.2352941185235977)
        colors[self.imgui.COLOR_RESIZE_GRIP_ACTIVE] = (1.0, 1.0, 1.0, 0.2352941185235977)
        colors[self.imgui.COLOR_TAB] = (1.0, 1.0, 1.0, 0.09803921729326248)
        colors[self.imgui.COLOR_TAB_HOVERED] = (1.0, 1.0, 1.0, 0.1568627506494522)
        colors[self.imgui.COLOR_TAB_ACTIVE] = (1.0, 1.0, 1.0, 0.3137255012989044)
        colors[self.imgui.COLOR_TAB_UNFOCUSED] = (0.0, 0.0, 0.0, 0.1568627506494522)
        colors[self.imgui.COLOR_TAB_UNFOCUSED_ACTIVE] = (1.0, 1.0, 1.0, 0.2352941185235977)
        colors[self.imgui.COLOR_PLOT_LINES] = (1.0, 1.0, 1.0, 0.3529411852359772)
        colors[self.imgui.COLOR_PLOT_LINES_HOVERED] = (1.0, 1.0, 1.0, 1.0)
        colors[self.imgui.COLOR_PLOT_HISTOGRAM] = (1.0, 1.0, 1.0, 0.3529411852359772)
        colors[self.imgui.COLOR_PLOT_HISTOGRAM_HOVERED] = (1.0, 1.0, 1.0, 1.0)
        colors[self.imgui.COLOR_TABLE_HEADER_BACKGROUND] = (0.1568627506494522, 0.1568627506494522, 0.1568627506494522, 1.0)
        colors[self.imgui.COLOR_TABLE_BORDER_STRONG] = (1.0, 1.0, 1.0, 0.3137255012989044)
        colors[self.imgui.COLOR_TABLE_BORDER_LIGHT] = (1.0, 1.0, 1.0, 0.196078434586525)
        colors[self.imgui.COLOR_TABLE_ROW_BACKGROUND] = (0.0, 0.0, 0.0, 0.0)
        colors[self.imgui.COLOR_TABLE_ROW_BACKGROUND_ALT] = (1.0, 1.0, 1.0, 0.01960784383118153)
        colors[self.imgui.COLOR_TEXT_SELECTED_BACKGROUND] = (0.0, 0.0, 0.0, 1.0)
        colors[self.imgui.COLOR_DRAG_DROP_TARGET] = (0.168627455830574, 0.2313725501298904, 0.5372549295425415, 1.0)
        colors[self.imgui.COLOR_NAV_HIGHLIGHT] = (1.0, 1.0, 1.0, 1.0)
        colors[self.imgui.COLOR_NAV_WINDOWING_HIGHLIGHT] = (1.0, 1.0, 1.0, 0.699999988079071)
        colors[self.imgui.COLOR_NAV_WINDOWING_DIM_BACKGROUND] = (0.800000011920929, 0.800000011920929, 0.800000011920929, 0.2000000029802322)
        colors[self.imgui.COLOR_MODAL_WINDOW_DIM_BACKGROUND] = (0.0, 0.0, 0.0, 0.5647059082984924)
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
        style.window_menu_button_position = self.imgui.DIRECTION_NONE
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
        style.tab_min_width_for_close_button = 0.0
        style.color_button_position = self.imgui.DIRECTION_RIGHT
        style.button_text_align = (0.5, 0.5)
        style.selectable_text_align = (0.0, 0.0)

        # fmt: off

        # Colors
        colors = style.colors
        colors[self.imgui.COLOR_TEXT] = (1.0, 1.0, 1.0, 1.0)
        colors[self.imgui.COLOR_TEXT_DISABLED] = (0.2745098173618317, 0.3176470696926117, 0.4509803950786591, 1.0)
        colors[self.imgui.COLOR_WINDOW_BACKGROUND] = (0.0784313753247261, 0.08627451211214066, 0.1019607856869698, 1.0)
        colors[self.imgui.COLOR_CHILD_BACKGROUND] = (0.0784313753247261, 0.08627451211214066, 0.1019607856869698, 1.0)
        colors[self.imgui.COLOR_POPUP_BACKGROUND] = (0.0784313753247261, 0.08627451211214066, 0.1019607856869698, 1.0)
        colors[self.imgui.COLOR_BORDER] = (0.1568627506494522, 0.168627455830574, 0.1921568661928177, 1.0)
        colors[self.imgui.COLOR_BORDER_SHADOW] = (0.0784313753247261, 0.08627451211214066, 0.1019607856869698, 1.0)
        colors[self.imgui.COLOR_FRAME_BACKGROUND] = (0.1176470592617989, 0.1333333402872086, 0.1490196138620377, 1.0)
        colors[self.imgui.COLOR_FRAME_BACKGROUND_HOVERED] = (0.1568627506494522, 0.168627455830574, 0.1921568661928177, 1.0)
        colors[self.imgui.COLOR_FRAME_BACKGROUND_ACTIVE] = (0.2352941185235977, 0.2156862765550613, 0.5960784554481506, 1.0)
        colors[self.imgui.COLOR_TITLE_BACKGROUND] = (0.0470588244497776, 0.05490196123719215, 0.07058823853731155, 1.0)
        colors[self.imgui.COLOR_TITLE_BACKGROUND_ACTIVE] = (0.0470588244497776, 0.05490196123719215, 0.07058823853731155, 1.0)
        colors[self.imgui.COLOR_TITLE_BACKGROUND_COLLAPSED] = (0.0784313753247261, 0.08627451211214066, 0.1019607856869698, 1.0)
        colors[self.imgui.COLOR_MENUBAR_BACKGROUND] = (0.09803921729326248, 0.105882354080677, 0.1215686276555061, 1.0)
        colors[self.imgui.COLOR_SCROLLBAR_BACKGROUND] = (0.0470588244497776, 0.05490196123719215, 0.07058823853731155, 1.0)
        colors[self.imgui.COLOR_SCROLLBAR_GRAB] = (0.1176470592617989, 0.1333333402872086, 0.1490196138620377, 1.0)
        colors[self.imgui.COLOR_SCROLLBAR_GRAB_HOVERED] = (0.1568627506494522, 0.168627455830574, 0.1921568661928177, 1.0)
        colors[self.imgui.COLOR_SCROLLBAR_GRAB_ACTIVE] = (0.1176470592617989, 0.1333333402872086, 0.1490196138620377, 1.0)
        colors[self.imgui.COLOR_CHECK_MARK] = (0.4980392158031464, 0.5137255191802979, 1.0, 1.0)
        colors[self.imgui.COLOR_SLIDER_GRAB] = (0.4980392158031464, 0.5137255191802979, 1.0, 1.0)
        colors[self.imgui.COLOR_SLIDER_GRAB_ACTIVE] = (0.5372549295425415, 0.5529412031173706, 1.0, 1.0)
        colors[self.imgui.COLOR_BUTTON] = (0.1176470592617989, 0.1333333402872086, 0.1490196138620377, 1.0)
        colors[self.imgui.COLOR_BUTTON_HOVERED] = (0.196078434586525, 0.1764705926179886, 0.5450980663299561, 1.0)
        colors[self.imgui.COLOR_BUTTON_ACTIVE] = (0.2352941185235977, 0.2156862765550613, 0.5960784554481506, 1.0)
        colors[self.imgui.COLOR_HEADER] = (0.1176470592617989, 0.1333333402872086, 0.1490196138620377, 1.0)
        colors[self.imgui.COLOR_HEADER_HOVERED] = (0.196078434586525, 0.1764705926179886, 0.5450980663299561, 1.0)
        colors[self.imgui.COLOR_HEADER_ACTIVE] = (0.2352941185235977, 0.2156862765550613, 0.5960784554481506, 1.0)
        colors[self.imgui.COLOR_SEPARATOR] = (0.1568627506494522, 0.1843137294054031, 0.250980406999588, 1.0)
        colors[self.imgui.COLOR_SEPARATOR_HOVERED] = (0.1568627506494522, 0.1843137294054031, 0.250980406999588, 1.0)
        colors[self.imgui.COLOR_SEPARATOR_ACTIVE] = (0.1568627506494522, 0.1843137294054031, 0.250980406999588, 1.0)
        colors[self.imgui.COLOR_RESIZE_GRIP] = (0.1176470592617989, 0.1333333402872086, 0.1490196138620377, 1.0)
        colors[self.imgui.COLOR_RESIZE_GRIP_HOVERED] = (0.196078434586525, 0.1764705926179886, 0.5450980663299561, 1.0)
        colors[self.imgui.COLOR_RESIZE_GRIP_ACTIVE] = (0.2352941185235977, 0.2156862765550613, 0.5960784554481506, 1.0)
        colors[self.imgui.COLOR_TAB] = (0.0470588244497776, 0.05490196123719215, 0.07058823853731155, 1.0)
        colors[self.imgui.COLOR_TAB_HOVERED] = (0.1176470592617989, 0.1333333402872086, 0.1490196138620377, 1.0)
        colors[self.imgui.COLOR_TAB_ACTIVE] = (0.09803921729326248, 0.105882354080677, 0.1215686276555061, 1.0)
        colors[self.imgui.COLOR_TAB_UNFOCUSED] = (0.0470588244497776, 0.05490196123719215, 0.07058823853731155, 1.0)
        colors[self.imgui.COLOR_TAB_UNFOCUSED_ACTIVE] = (0.0784313753247261, 0.08627451211214066, 0.1019607856869698, 1.0)
        colors[self.imgui.COLOR_PLOT_LINES] = (0.5215686559677124, 0.6000000238418579, 0.7019608020782471, 1.0)
        colors[self.imgui.COLOR_PLOT_LINES_HOVERED] = (0.03921568766236305, 0.9803921580314636, 0.9803921580314636, 1.0)
        colors[self.imgui.COLOR_PLOT_HISTOGRAM] = (1.0, 0.2901960909366608, 0.5960784554481506, 1.0)
        colors[self.imgui.COLOR_PLOT_HISTOGRAM_HOVERED] = (0.9960784316062927, 0.4745098054409027, 0.6980392336845398, 1.0)
        colors[self.imgui.COLOR_TABLE_HEADER_BACKGROUND] = (0.0470588244497776, 0.05490196123719215, 0.07058823853731155, 1.0)
        colors[self.imgui.COLOR_TABLE_BORDER_STRONG] = (0.0470588244497776, 0.05490196123719215, 0.07058823853731155, 1.0)
        colors[self.imgui.COLOR_TABLE_BORDER_LIGHT] = (0.0, 0.0, 0.0, 1.0)
        colors[self.imgui.COLOR_TABLE_ROW_BACKGROUND] = (0.1176470592617989, 0.1333333402872086, 0.1490196138620377, 1.0)
        colors[self.imgui.COLOR_TABLE_ROW_BACKGROUND_ALT] = (0.09803921729326248, 0.105882354080677, 0.1215686276555061, 1.0)
        colors[self.imgui.COLOR_TEXT_SELECTED_BACKGROUND] = (0.2352941185235977, 0.2156862765550613, 0.5960784554481506, 1.0)
        colors[self.imgui.COLOR_DRAG_DROP_TARGET] = (0.4980392158031464, 0.5137255191802979, 1.0, 1.0)
        colors[self.imgui.COLOR_NAV_HIGHLIGHT] = (0.4980392158031464, 0.5137255191802979, 1.0, 1.0)
        colors[self.imgui.COLOR_NAV_WINDOWING_HIGHLIGHT] = (0.4980392158031464, 0.5137255191802979, 1.0, 1.0)
        colors[self.imgui.COLOR_NAV_WINDOWING_DIM_BACKGROUND] = (0.196078434586525, 0.1764705926179886, 0.5450980663299561, 0.501960813999176)
        colors[self.imgui.COLOR_MODAL_WINDOW_DIM_BACKGROUND] = (0.196078434586525, 0.1764705926179886, 0.5450980663299561, 0.501960813999176)
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

        self.io.display_size = width, height

    def get_theme_color(self, color_id, fallback_color=(1.0, 1.0, 1.0, 1.0)):
        """Get a color from the current theme with fallback.

        Args:
            color_id: ImGui color constant (e.g., self.imgui.COLOR_TEXT_DISABLED)
            fallback_color: RGBA tuple to use if color not available

        Returns:
            RGBA tuple of the theme color or fallback
        """
        if not self.is_available:
            return fallback_color

        try:
            style = self.imgui.get_style()
            return style.colors[color_id]
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
