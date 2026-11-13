# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Color defaults shared by ImGui styles and custom viewer controls.

Edit GUI colors here; widget/layout code should contain no color literals.
Aliases intentionally share defaults for repeated roles. These are startup
defaults, not a public runtime theming API. Per-widget accent/tint overrides
remain supported, including tinting the grayscale icon atlas without uploads.
Legacy and alternate palettes are separate to preserve their appearance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import ModuleType

Color = tuple[float, float, float, float]
"""GUI RGBA components in the range zero to one."""

# Neutral shades shared by independent semantic roles below.
_GRAY_090: Color = (0.09, 0.09, 0.09, 1)
_GRAY_120: Color = (0.12, 0.12, 0.12, 1)
_GRAY_130: Color = (0.13, 0.13, 0.13, 1)
_GRAY_145: Color = (0.145, 0.145, 0.145, 1)
_GRAY_170: Color = (0.17, 0.17, 0.17, 1)
_GRAY_180: Color = (0.18, 0.18, 0.18, 1)
_GRAY_190: Color = (0.19, 0.19, 0.19, 1)
_GRAY_200: Color = (0.2, 0.2, 0.2, 1)
_GRAY_215: Color = (0.215, 0.215, 0.215, 1)
_GRAY_220: Color = (0.22, 0.22, 0.22, 1)
_GRAY_240: Color = (0.24, 0.24, 0.24, 1)
_GRAY_250: Color = (0.25, 0.25, 0.25, 1)
_GRAY_260: Color = (0.26, 0.26, 0.26, 1)
_GRAY_265: Color = (0.265, 0.265, 0.265, 1)
_GRAY_280: Color = (0.28, 0.28, 0.28, 1)
_GRAY_290: Color = (0.29, 0.29, 0.29, 1)
_GRAY_300: Color = (0.3, 0.3, 0.3, 1)
_GRAY_310: Color = (0.31, 0.31, 0.31, 1)
_GRAY_320: Color = (0.32, 0.32, 0.32, 1)
_GRAY_340: Color = (0.34, 0.34, 0.34, 1)
_GRAY_350: Color = (0.35, 0.35, 0.35, 1)
_GRAY_360: Color = (0.36, 0.36, 0.36, 1)
_GRAY_370: Color = (0.37, 0.37, 0.37, 1)
_GRAY_380: Color = (0.38, 0.38, 0.38, 1)
_GRAY_400: Color = (0.4, 0.4, 0.4, 1)
_GRAY_420: Color = (0.42, 0.42, 0.42, 1)
_GRAY_430: Color = (0.43, 0.43, 0.43, 1)
_GRAY_460: Color = (0.46, 0.46, 0.46, 1)
_GRAY_480: Color = (0.48, 0.48, 0.48, 1)
_GRAY_500: Color = (0.5, 0.5, 0.5, 1)
_GRAY_530: Color = (0.53, 0.53, 0.53, 1)
_GRAY_550: Color = (0.55, 0.55, 0.55, 1)
_GRAY_580: Color = (0.58, 0.58, 0.58, 1)
_GRAY_630: Color = (0.63, 0.63, 0.63, 1)
_GRAY_640: Color = (0.64, 0.64, 0.64, 1)
_GRAY_650: Color = (0.65, 0.65, 0.65, 1)
_GRAY_680: Color = (0.68, 0.68, 0.68, 1)
_GRAY_700: Color = (0.7, 0.7, 0.7, 1)
_GRAY_720: Color = (0.72, 0.72, 0.72, 1)
_GRAY_740: Color = (0.74, 0.74, 0.74, 1)
_GRAY_820: Color = (0.82, 0.82, 0.82, 1)
_GRAY_850: Color = (0.85, 0.85, 0.85, 1)
_GRAY_880: Color = (0.88, 0.88, 0.88, 1)
_GRAY_900: Color = (0.9, 0.9, 0.9, 1)
_GRAY_920: Color = (0.92, 0.92, 0.92, 1)
_GRAY_950: Color = (0.95, 0.95, 0.95, 1)

ACCENT: Color = (0.96, 0.61, 0.32, 1)
MUTED = _GRAY_650
WHITE: Color = (1, 1, 1, 1)
BLACK: Color = (0, 0, 0, 1)
TRANSPARENT: Color = (*BLACK[:3], 0)
TEXT = _GRAY_880
TEXT_DISABLED = _GRAY_580
ICON_DISABLED = _GRAY_400
ACCENT_TAB: Color = (0.29, 0.23, 0.18, 1)
ACCENT_TAB_SELECTED: Color = (0.4, 0.29, 0.2, 1)
ACCENT_TAB_HOVERED: Color = (0.44, 0.32, 0.22, 1)
TOOL_BUTTON = _GRAY_190
TOOL_BUTTON_HOVERED = _GRAY_250
TOOL_BUTTON_ACTIVE = _GRAY_300
TRANSPORT_BUTTON = _GRAY_300
TRANSPORT_BUTTON_HOVERED = _GRAY_380
TRANSPORT_BUTTON_PRESSED = _GRAY_240
GROUP_BACKGROUND = _GRAY_215
GROUP_HOVERED = _GRAY_250
GROUP_ACTIVE = _GRAY_280
EDITOR_BACKGROUND = _GRAY_310
EDITOR_READONLY_BACKGROUND = _GRAY_265
EDITOR_TEXT = _GRAY_920
EDITOR_READONLY_TEXT = _GRAY_650
AXIS_BUTTON_BACKGROUND = _GRAY_180
CONSOLE_ERROR: Color = (1, 0.43, 0.43, 1)
CONSOLE_WARNING: Color = (0.96, 0.74, 0.35, 1)
CONSOLE_INFO = _GRAY_850
CONSOLE_DEBUG = _GRAY_580
CONSOLE_BUTTON_HOVERED = _GRAY_220
CONTACT_RIGID: Color = (0.4, 0.84, 0.78, 1)
CONTACT_SOFT_RIGID: Color = (0.36, 0.65, 1, 1)
CONTACT_SOFT: Color = (1, 0.5, 0.64, 1)
BODY_SPEED: Color = (0.7, 0.61, 1, 1)
PARTICLE_SPEED: Color = (1, 0.72, 0.36, 1)
TIMELINE_TEXT = _GRAY_630
TIMELINE_GRID = _GRAY_290
TIMELINE_BACKGROUND = _GRAY_200
TIMELINE_HOVERED = _GRAY_240
TIMELINE_BORDER = _GRAY_300
TIMELINE_BORDER_HOVERED = _GRAY_480
TIMELINE_CURSOR: Color = (1, 0.73, 0.3, 1)
TIMELINE_SEPARATOR = _GRAY_340
STATUS_VALUE = _GRAY_820
STATUS_DETAIL = _GRAY_700
VERSION_TEXT = _GRAY_460
TREE_SELECTED: Color = (0.23, 0.29, 0.42, 1)
BREADCRUMB_SELECTED: Color = (0.32, 0.26, 0.2, 1)
BREADCRUMB_BACKGROUND = _GRAY_220
VIEWPORT_BACKGROUND = _GRAY_120
VIEWPORT_BORDER = _GRAY_420
ORIENTATION_FACE = _GRAY_170
ORIENTATION_HIGHLIGHT: Color = (1, 0.74, 0.47, 1)
ORIENTATION_EDGE = _GRAY_680
ORIENTATION_TEXT = _GRAY_950
ORIENTATION_RING = _GRAY_640
ORIENTATION_TICK = _GRAY_700
ORIENTATION_HANDLE_RIM = _GRAY_120
ORIENTATION_AXIS_TEXT = _GRAY_090
LOADING_DIM: Color = (*BLACK[:3], 0.55)
LOADING_BALL: Color = (0.88, 0.88, 0.92, 1)
LOADING_STRING: Color = (0.55, 0.55, 0.6, 1)
LOADING_BAR: Color = (0.45, 0.45, 0.5, 1)
LOADING_TEXT = _GRAY_900
ERROR_TEXT: Color = (1, 0.3, 0.3, 1)

# Component axes and scene-tree object categories.
AXIS_X: Color = (0.87, 0.39, 0.38, 1)
AXIS_Y: Color = (0.44, 0.75, 0.47, 1)
AXIS_Z: Color = (0.38, 0.62, 0.91, 1)
AXIS_W: Color = (0.71, 0.6, 0.88, 1)
AXES = (AXIS_X, AXIS_Y, AXIS_Z, AXIS_W)
ORIENTATION_AXIS_X: Color = (0.94, 0.43, 0.4, 1)
ORIENTATION_AXIS_Y: Color = (0.49, 0.84, 0.55, 1)
ORIENTATION_AXIS_Z: Color = (0.43, 0.69, 1, 1)
ORIENTATION_AXES = (ORIENTATION_AXIS_X, ORIENTATION_AXIS_Y, ORIENTATION_AXIS_Z)
EXPLORER_ROOT = _GRAY_740
EXPLORER_GROUP = _GRAY_740
EXPLORER_ELEMENTS = _GRAY_680
EXPLORER_ARTICULATION: Color = (0.82, 0.61, 0.4, 1)
EXPLORER_BODY: Color = (0.88, 0.64, 0.38, 1)
EXPLORER_SHAPE: Color = (0.43, 0.73, 0.55, 1)
EXPLORER_JOINT: Color = (0.48, 0.69, 0.8, 1)
EXPLORER_SURFACE: Color = (0.4, 0.73, 0.63, 1)
EXPLORER_VOLUME: Color = (0.76, 0.55, 0.68, 1)
EXPLORER_PARTICLE_SET: Color = (0.84, 0.71, 0.43, 1)
EXPLORER_PARTICLE = EXPLORER_PARTICLE_SET
EXPLORER_TRI = EXPLORER_SURFACE
EXPLORER_TET = EXPLORER_VOLUME
EXPLORER_CUSTOM: Color = (0.69, 0.61, 0.81, 1)
EXPLORER_SDF: Color = (0.4, 0.74, 0.79, 1)
EXPLORER_BVH = EXPLORER_SDF
EXPLORER_TEXTURE: Color = (0.76, 0.57, 0.82, 1)
EXPLORER_COLORS = {
    "root": EXPLORER_ROOT,
    "group": EXPLORER_GROUP,
    "elements": EXPLORER_ELEMENTS,
    "articulation": EXPLORER_ARTICULATION,
    "body": EXPLORER_BODY,
    "shape": EXPLORER_SHAPE,
    "joint": EXPLORER_JOINT,
    "surface": EXPLORER_SURFACE,
    "volume": EXPLORER_VOLUME,
    "particle_set": EXPLORER_PARTICLE_SET,
    "particle": EXPLORER_PARTICLE,
    "tri": EXPLORER_TRI,
    "tet": EXPLORER_TET,
    "custom": EXPLORER_CUSTOM,
    "sdf": EXPLORER_SDF,
    "bvh": EXPLORER_BVH,
    "texture": EXPLORER_TEXTURE,
}


# Grey ImGui slots; repeated values reuse the named defaults above.
_GREY_TEXT: Color = (0.9803921580314636, 0.9803921580314636, 0.9803921580314636, 1.0)
_GREY_TEXT_DISABLED: Color = (0.4980392158031464, 0.4980392158031464, 0.4980392158031464, 1.0)
_GREY_WINDOW_BG: Color = (0.0941176488995552, 0.0941176488995552, 0.0941176488995552, 1.0)
_GREY_CHILD_BG: Color = (0.1568627506494522, 0.1568627506494522, 0.1568627506494522, 1.0)
_GREY_BORDER: Color = (*WHITE[:3], 0.09803921729326248)
_GREY_FRAME_BG_HOVERED: Color = (*WHITE[:3], 0.1568627506494522)
_GREY_FRAME_BG_ACTIVE: Color = (*BLACK[:3], 0.0470588244497776)
_GREY_TITLE_BG: Color = (0.1176470592617989, 0.1176470592617989, 0.1176470592617989, 1.0)
_GREY_SCROLLBAR_BG: Color = (*BLACK[:3], 0.1098039224743843)
_GREY_SCROLLBAR_GRAB: Color = (*WHITE[:3], 0.3921568691730499)
_GREY_SCROLLBAR_GRAB_HOVERED: Color = (*WHITE[:3], 0.4705882370471954)
_GREY_SCROLLBAR_GRAB_ACTIVE: Color = (*BLACK[:3], 0.09803921729326248)
_GREY_SLIDER_GRAB_ACTIVE: Color = (*WHITE[:3], 0.3137255012989044)
_GREY_SEPARATOR_HOVERED: Color = (*WHITE[:3], 0.2352941185235977)
_GREY_TAB_DIMMED: Color = (*BLACK[:3], 0.1568627506494522)
_GREY_PLOT_LINES: Color = (*WHITE[:3], 0.3529411852359772)
_GREY_TABLE_BORDER_LIGHT: Color = (*WHITE[:3], 0.196078434586525)
_GREY_TABLE_ROW_BG_ALT: Color = (*WHITE[:3], 0.01960784383118153)
_GREY_DRAG_DROP_TARGET: Color = (0.168627455830574, 0.2313725501298904, 0.5372549295425415, 1.0)
_GREY_NAV_WINDOWING_HIGHLIGHT: Color = (*WHITE[:3], 0.699999988079071)
_GREY_NAV_WINDOWING_DIM_BG: Color = (0.800000011920929, 0.800000011920929, 0.800000011920929, 0.2000000029802322)
_GREY_MODAL_WINDOW_DIM_BG: Color = (*BLACK[:3], 0.5647059082984924)
GREY_PALETTE: dict[str, Color] = {
    "text": _GREY_TEXT,
    "text_disabled": _GREY_TEXT_DISABLED,
    "window_bg": _GREY_WINDOW_BG,
    "child_bg": _GREY_CHILD_BG,
    "popup_bg": _GREY_WINDOW_BG,
    "border": _GREY_BORDER,
    "border_shadow": TRANSPARENT,
    "frame_bg": _GREY_BORDER,
    "frame_bg_hovered": _GREY_FRAME_BG_HOVERED,
    "frame_bg_active": _GREY_FRAME_BG_ACTIVE,
    "title_bg": _GREY_TITLE_BG,
    "title_bg_active": _GREY_CHILD_BG,
    "title_bg_collapsed": _GREY_TITLE_BG,
    "menu_bar_bg": TRANSPARENT,
    "scrollbar_bg": _GREY_SCROLLBAR_BG,
    "scrollbar_grab": _GREY_SCROLLBAR_GRAB,
    "scrollbar_grab_hovered": _GREY_SCROLLBAR_GRAB_HOVERED,
    "scrollbar_grab_active": _GREY_SCROLLBAR_GRAB_ACTIVE,
    "check_mark": WHITE,
    "slider_grab": _GREY_SCROLLBAR_GRAB,
    "slider_grab_active": _GREY_SLIDER_GRAB_ACTIVE,
    "button": _GREY_BORDER,
    "button_hovered": _GREY_FRAME_BG_HOVERED,
    "button_active": _GREY_FRAME_BG_ACTIVE,
    "header": _GREY_BORDER,
    "header_hovered": _GREY_FRAME_BG_HOVERED,
    "header_active": _GREY_FRAME_BG_ACTIVE,
    "separator": _GREY_FRAME_BG_HOVERED,
    "separator_hovered": _GREY_SEPARATOR_HOVERED,
    "separator_active": _GREY_SEPARATOR_HOVERED,
    "resize_grip": _GREY_FRAME_BG_HOVERED,
    "resize_grip_hovered": _GREY_SEPARATOR_HOVERED,
    "resize_grip_active": _GREY_SEPARATOR_HOVERED,
    "tab": _GREY_BORDER,
    "tab_hovered": _GREY_FRAME_BG_HOVERED,
    "tab_selected": _GREY_SLIDER_GRAB_ACTIVE,
    "tab_dimmed": _GREY_TAB_DIMMED,
    "tab_dimmed_selected": _GREY_SEPARATOR_HOVERED,
    "plot_lines": _GREY_PLOT_LINES,
    "plot_lines_hovered": WHITE,
    "plot_histogram": _GREY_PLOT_LINES,
    "plot_histogram_hovered": WHITE,
    "table_header_bg": _GREY_CHILD_BG,
    "table_border_strong": _GREY_SLIDER_GRAB_ACTIVE,
    "table_border_light": _GREY_TABLE_BORDER_LIGHT,
    "table_row_bg": TRANSPARENT,
    "table_row_bg_alt": _GREY_TABLE_ROW_BG_ALT,
    "text_selected_bg": BLACK,
    "drag_drop_target": _GREY_DRAG_DROP_TARGET,
    "nav_cursor": WHITE,
    "nav_windowing_highlight": _GREY_NAV_WINDOWING_HIGHLIGHT,
    "nav_windowing_dim_bg": _GREY_NAV_WINDOWING_DIM_BG,
    "modal_window_dim_bg": _GREY_MODAL_WINDOW_DIM_BG,
}

# Legacy ImGui slots; repeated values reuse the named defaults above.
_LEGACY_TEXT_DISABLED: Color = (0.2745098173618317, 0.3176470696926117, 0.4509803950786591, 1.0)
_LEGACY_WINDOW_BG: Color = (0.0784313753247261, 0.08627451211214066, 0.1019607856869698, 1.0)
_LEGACY_BORDER: Color = (0.1568627506494522, 0.168627455830574, 0.1921568661928177, 1.0)
_LEGACY_FRAME_BG: Color = (0.1176470592617989, 0.1333333402872086, 0.1490196138620377, 1.0)
_LEGACY_FRAME_BG_ACTIVE: Color = (0.2352941185235977, 0.2156862765550613, 0.5960784554481506, 1.0)
_LEGACY_TITLE_BG: Color = (0.0470588244497776, 0.05490196123719215, 0.07058823853731155, 1.0)
_LEGACY_MENU_BAR_BG: Color = (0.09803921729326248, 0.105882354080677, 0.1215686276555061, 1.0)
_LEGACY_CHECK_MARK: Color = (0.4980392158031464, 0.5137255191802979, 1.0, 1.0)
_LEGACY_SLIDER_GRAB_ACTIVE: Color = (0.5372549295425415, 0.5529412031173706, 1.0, 1.0)
_LEGACY_BUTTON_HOVERED: Color = (0.196078434586525, 0.1764705926179886, 0.545098066329956, 1.0)
_LEGACY_SEPARATOR: Color = (0.1568627506494522, 0.1843137294054031, 0.250980406999588, 1.0)
_LEGACY_RESIZE_GRIP: Color = (*WHITE[:3], 0.2)
_LEGACY_RESIZE_GRIP_HOVERED: Color = (*_LEGACY_CHECK_MARK[:3], 0.85)
_LEGACY_PLOT_LINES: Color = (0.5215686559677124, 0.6000000238418579, 0.7019608020782471, 1.0)
_LEGACY_PLOT_LINES_HOVERED: Color = (0.03921568766236305, 0.9803921580314636, 0.9803921580314636, 1.0)
_LEGACY_PLOT_HISTOGRAM: Color = (1.0, 0.2901960909366608, 0.5960784554481506, 1.0)
_LEGACY_PLOT_HISTOGRAM_HOVERED: Color = (0.9960784316062927, 0.4745098054409027, 0.6980392336845398, 1.0)
_LEGACY_NAV_WINDOWING_DIM_BG: Color = (*_LEGACY_BUTTON_HOVERED[:3], 0.501960813999176)
LEGACY_PALETTE: dict[str, Color] = {
    "text": WHITE,
    "text_disabled": _LEGACY_TEXT_DISABLED,
    "window_bg": _LEGACY_WINDOW_BG,
    "child_bg": _LEGACY_WINDOW_BG,
    "popup_bg": _LEGACY_WINDOW_BG,
    "border": _LEGACY_BORDER,
    "border_shadow": _LEGACY_WINDOW_BG,
    "frame_bg": _LEGACY_FRAME_BG,
    "frame_bg_hovered": _LEGACY_BORDER,
    "frame_bg_active": _LEGACY_FRAME_BG_ACTIVE,
    "title_bg": _LEGACY_TITLE_BG,
    "title_bg_active": _LEGACY_TITLE_BG,
    "title_bg_collapsed": _LEGACY_WINDOW_BG,
    "menu_bar_bg": _LEGACY_MENU_BAR_BG,
    "scrollbar_bg": _LEGACY_TITLE_BG,
    "scrollbar_grab": _LEGACY_FRAME_BG,
    "scrollbar_grab_hovered": _LEGACY_BORDER,
    "scrollbar_grab_active": _LEGACY_FRAME_BG,
    "check_mark": _LEGACY_CHECK_MARK,
    "slider_grab": _LEGACY_CHECK_MARK,
    "slider_grab_active": _LEGACY_SLIDER_GRAB_ACTIVE,
    "button": _LEGACY_FRAME_BG,
    "button_hovered": _LEGACY_BUTTON_HOVERED,
    "button_active": _LEGACY_FRAME_BG_ACTIVE,
    "header": _LEGACY_FRAME_BG,
    "header_hovered": _LEGACY_BUTTON_HOVERED,
    "header_active": _LEGACY_FRAME_BG_ACTIVE,
    "separator": _LEGACY_SEPARATOR,
    "separator_hovered": _LEGACY_SEPARATOR,
    "separator_active": _LEGACY_SEPARATOR,
    "resize_grip": _LEGACY_RESIZE_GRIP,
    "resize_grip_hovered": _LEGACY_RESIZE_GRIP_HOVERED,
    "resize_grip_active": _LEGACY_SLIDER_GRAB_ACTIVE,
    "tab": _LEGACY_TITLE_BG,
    "tab_hovered": _LEGACY_FRAME_BG,
    "tab_selected": _LEGACY_MENU_BAR_BG,
    "tab_dimmed": _LEGACY_TITLE_BG,
    "tab_dimmed_selected": _LEGACY_WINDOW_BG,
    "plot_lines": _LEGACY_PLOT_LINES,
    "plot_lines_hovered": _LEGACY_PLOT_LINES_HOVERED,
    "plot_histogram": _LEGACY_PLOT_HISTOGRAM,
    "plot_histogram_hovered": _LEGACY_PLOT_HISTOGRAM_HOVERED,
    "table_header_bg": _LEGACY_TITLE_BG,
    "table_border_strong": _LEGACY_TITLE_BG,
    "table_border_light": BLACK,
    "table_row_bg": _LEGACY_FRAME_BG,
    "table_row_bg_alt": _LEGACY_MENU_BAR_BG,
    "text_selected_bg": _LEGACY_FRAME_BG_ACTIVE,
    "drag_drop_target": _LEGACY_CHECK_MARK,
    "nav_cursor": _LEGACY_CHECK_MARK,
    "nav_windowing_highlight": _LEGACY_CHECK_MARK,
    "nav_windowing_dim_bg": _LEGACY_NAV_WINDOWING_DIM_BG,
    "modal_window_dim_bg": _LEGACY_NAV_WINDOWING_DIM_BG,
}

# Workspace ImGui slots; repeated values reuse the named defaults above.
_WORKSPACE_TABLE_ROW_BG_ALT: Color = (*WHITE[:3], 0.015)
_WORKSPACE_TEXT_SELECTED_BG: Color = (*_GRAY_550[:3], 0.45)
_WORKSPACE_RESIZE_GRIP: Color = (*_GRAY_550[:3], 0.2)
_WORKSPACE_RESIZE_GRIP_HOVERED: Color = (*_GRAY_700[:3], 0.6)
_WORKSPACE_RESIZE_GRIP_ACTIVE: Color = (*ACCENT[:3], 0.9)
_WORKSPACE_DOCKING_PREVIEW: Color = (*_GRAY_650[:3], 0.45)
WORKSPACE_PALETTE: dict[str, Color] = {
    "border_shadow": _LEGACY_WINDOW_BG,
    "plot_lines_hovered": _LEGACY_PLOT_LINES_HOVERED,
    "plot_histogram": _LEGACY_PLOT_HISTOGRAM,
    "plot_histogram_hovered": _LEGACY_PLOT_HISTOGRAM_HOVERED,
    "drag_drop_target": _LEGACY_CHECK_MARK,
    "nav_windowing_highlight": _LEGACY_CHECK_MARK,
    "nav_windowing_dim_bg": _LEGACY_NAV_WINDOWING_DIM_BG,
    "modal_window_dim_bg": _LEGACY_NAV_WINDOWING_DIM_BG,
    "text": TEXT,
    "text_disabled": TEXT_DISABLED,
    "window_bg": _GRAY_145,
    "child_bg": _GRAY_145,
    "popup_bg": _GRAY_180,
    "border": _GRAY_340,
    "frame_bg": _GRAY_290,
    "frame_bg_hovered": _GRAY_340,
    "frame_bg_active": _GRAY_370,
    "button": _GRAY_240,
    "button_hovered": _GRAY_300,
    "button_active": _GRAY_360,
    "header": _GRAY_290,
    "header_hovered": _GRAY_320,
    "header_active": _GRAY_370,
    "check_mark": ACCENT,
    "checkbox_selected_bg": _GRAY_320,
    "slider_grab": _GRAY_700,
    "slider_grab_active": ACCENT,
    "tab": _GRAY_180,
    "tab_selected": _GRAY_260,
    "tab_hovered": _GRAY_300,
    "tab_selected_overline": ACCENT,
    "tab_dimmed": _GRAY_180,
    "tab_dimmed_selected": _GRAY_260,
    "tab_dimmed_selected_overline": _GRAY_500,
    "title_bg": _GRAY_130,
    "title_bg_active": _GRAY_190,
    "title_bg_collapsed": _GRAY_130,
    "menu_bar_bg": _GRAY_180,
    "scrollbar_bg": _GRAY_120,
    "scrollbar_grab": _GRAY_310,
    "scrollbar_grab_hovered": _GRAY_430,
    "scrollbar_grab_active": _GRAY_530,
    "separator": _GRAY_320,
    "separator_hovered": _GRAY_550,
    "separator_active": ACCENT,
    "table_header_bg": _GRAY_180,
    "table_border_strong": _GRAY_250,
    "table_border_light": _GRAY_200,
    "table_row_bg": TRANSPARENT,
    "table_row_bg_alt": _WORKSPACE_TABLE_ROW_BG_ALT,
    "text_selected_bg": _WORKSPACE_TEXT_SELECTED_BG,
    "resize_grip": _WORKSPACE_RESIZE_GRIP,
    "resize_grip_hovered": _WORKSPACE_RESIZE_GRIP_HOVERED,
    "resize_grip_active": _WORKSPACE_RESIZE_GRIP_ACTIVE,
    "nav_cursor": ACCENT,
    "input_text_cursor": ACCENT,
    "tree_lines": _GRAY_350,
    "docking_empty_bg": _GRAY_130,
    "docking_preview": _WORKSPACE_DOCKING_PREVIEW,
    "plot_lines": _GRAY_720,
}


def apply_palette(ui: ModuleType, palette: Mapping[str, Color]) -> None:
    """Apply supported slots, retaining defaults for older optional ImGui bindings."""
    style = ui.get_style()
    for name, color in palette.items():
        if (slot := getattr(ui.Col_, name, None)) is not None:
            style.set_color_(slot, ui.ImVec4(*color))
