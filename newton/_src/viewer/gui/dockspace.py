# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dockable viewport, images, and plots inside the simulator window."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    from newton._src.viewer.gui.workspace import ViewerWorkspace


class ViewerDockspace:
    """Arrange the viewport and logged outputs within one application window.

    Stable ImGui window IDs preserve placement as examples and logger contents
    change. This object owns panel layout, not images, plots, or GL textures.
    Floating panels stay inside the same native window and GL context.

    The viewport samples the renderer's existing framebuffer. Its ImGui
    content rectangle drives scene rendering, camera projection, and input hit
    testing; it does not request a second scene render or a host image copy.
    """

    viewport_name = "Viewport###workspace_viewport"
    plots_name = "Plots###workspace_plots"

    def __init__(self, workspace: ViewerWorkspace) -> None:
        """Initialize viewport and output-window placement state without creating docking nodes."""
        self.workspace = workspace
        self.viewport_visible = True
        self.viewport_hovered = False
        self.viewport_rect = (0, 0, 1, 1)
        self._root = 0
        self._output_node = 0
        self._reset = False
        self._placed = set()
        self._open = {}
        self._available = {}
        self._class = None
        self._focus = None

    def reset(self) -> None:
        """Schedule layout reconstruction for the next draw rather than mutating docking state immediately."""
        self._reset = True

    def menu(self) -> None:
        """Reopen closed panels without waiting for another logging call."""
        ws = self.workspace
        ui = ws.gui.ui.imgui
        if ws.widgets.icon_button("##documents", "documents", "Viewport, images and plots", size=22):
            ui.open_popup("##documents_menu")
        if ui.begin_popup("##documents_menu"):
            if ui.menu_item("Viewport", "", False)[0]:
                self._focus = self.viewport_name
            for name, label in self._available.items():
                clicked, opened = ui.menu_item(label, "", self._open.get(name, True))
                if clicked:
                    self._open[name] = opened
                    if opened:
                        self._focus = name
            ui.separator()
            if ui.menu_item("Reset panel layout", "", False)[0]:
                self.reset()
            ui.end_popup()

    def _place_output(self, ui: ModuleType, name: str) -> None:
        """Dock a newly seen output beside the viewport once, preserving an already floating viewport."""
        if name in self._placed:
            return
        node = ui.internal.dock_builder_get_node(self._output_node) if self._output_node else None
        if node is None or not node.is_leaf_node():
            viewport = ui.internal.find_window_by_name(self.viewport_name)
            target = viewport.dock_node if viewport is not None else None
            # A floating viewport must remain floating when a new output arrives.
            if target is not None:
                root = target
                while root.parent_node is not None:
                    root = root.parent_node
                if root.id_ != self._root:
                    target = None
            target_id = target.id_ if target is not None else self._root
            target = ui.internal.dock_builder_get_node(target_id)
            if target is not None and target.is_leaf_node():
                split = ui.internal.dock_builder_split_node(target_id, ui.Dir.right, 0.40)
                self._output_node = split.id_at_dir
            else:
                self._output_node = self._root
        ui.internal.dock_builder_dock_window(name, self._output_node)
        ui.internal.dock_builder_finish(self._root)
        self._placed.add(name)

    def draw(self, rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Lay out panels and return the visible viewport rectangle in ImGui pixels.

        Coordinates use a top-left origin. The workspace maps them to the GL
        framebuffer's bottom-left origin using the current display size. A
        hidden viewport returns a minimal rectangle and clears viewport_visible.
        """
        ws = self.workspace
        ui, scale = ws.gui.ui.imgui, ws.widgets.scale
        images, plots = ws.viewer._image_logger, ws.viewer._plot_logger
        self._available = {self.image_name(name): f"Image · {name}" for name in images._images}
        if plots._scalar_buffers or plots._array_buffers:
            self._available[self.plots_name] = "Plots"
        self.viewport_visible = self.viewport_hovered = False
        x, y, width, height = rect
        ui.set_next_window_pos(ui.ImVec2(x, y))
        ui.set_next_window_size(ui.ImVec2(width, height))
        ui.push_style_var(ui.StyleVar_.window_padding, ui.ImVec2(0, 0))
        ui.push_style_var(ui.StyleVar_.window_rounding, 0)
        ui.push_style_var(ui.StyleVar_.window_border_size, 0)
        flags = (
            ui.WindowFlags_.no_decoration
            | ui.WindowFlags_.no_move
            | ui.WindowFlags_.no_saved_settings
            | ui.WindowFlags_.no_docking
            | ui.WindowFlags_.no_bring_to_front_on_focus
            | ui.WindowFlags_.no_nav_focus
        )
        ui.begin("##workspace_documents", flags=flags)
        self._root = ui.get_id("document_dockspace")
        if self._class is None:
            self._class = ui.WindowClass()
            self._class.class_id = ui.get_id("document_class")
            self._class.docking_allow_unclassed = False
            self._class.docking_always_tab_bar = True
        if self._reset or ui.internal.dock_builder_get_node(self._root) is None:
            ui.internal.dock_builder_remove_node(self._root)
            ui.internal.dock_builder_add_node(self._root, ui.internal.DockNodeFlagsPrivate_.dock_space)
            ui.internal.dock_builder_set_node_size(self._root, ui.ImVec2(width, height))
            ui.internal.dock_builder_dock_window(self.viewport_name, self._root)
            ui.internal.dock_builder_finish(self._root)
            self._output_node = 0
            self._placed.clear()
            self._reset = False
        ui.dock_space(self._root, ui.ImVec2(0, 0), window_class=self._class)
        ui.end()
        ui.pop_style_var(3)

        # Sampling the scene framebuffer adds one GPU quad, without readback or
        # a second scene pass. UVs and the render viewport use the same rectangle.
        ui.set_next_window_class(self._class)
        if self._focus == self.viewport_name:
            ui.set_next_window_focus()
            self._focus = None
        ui.push_style_var(ui.StyleVar_.window_padding, ui.ImVec2(0, 0))
        visible, _ = ui.begin(
            self.viewport_name, flags=ui.WindowFlags_.no_scrollbar | ui.WindowFlags_.no_scroll_with_mouse
        )
        if visible:
            pos, available = ui.get_cursor_screen_pos(), ui.get_content_region_avail()
            display = ui.get_io().display_size
            # Restrict the render target to visible pixels if a floating window
            # is partially outside the application window.
            x0, y0 = max(0, round(pos.x)), max(0, round(pos.y))
            x1 = min(round(display.x), round(pos.x + available.x))
            y1 = min(round(display.y), round(pos.y + available.y))
            if x1 > x0 and y1 > y0:
                self.viewport_rect = (x0, y0, x1 - x0, y1 - y0)
                self.viewport_visible = True
                ui.set_cursor_screen_pos(ui.ImVec2(x0, y0))
                ui.image(
                    ui.ImTextureRef(ws.viewer.renderer._frame_texture.value),
                    ui.ImVec2(x1 - x0, y1 - y0),
                    uv0=ui.ImVec2(x0 / display.x, 1.0 - y0 / display.y),
                    uv1=ui.ImVec2(x1 / display.x, 1.0 - y1 / display.y),
                )
                self.viewport_hovered = ui.is_item_hovered()
        ui.end()
        ui.pop_style_var()

        for name, label in self._available.items():
            if not self._open.get(name, True):
                continue
            self._place_output(ui, name)
            ui.set_next_window_class(self._class)
            if self._focus == name:
                ui.set_next_window_focus()
                self._focus = None
            ui.set_next_window_size(ui.ImVec2(480 * scale, 360 * scale), ui.Cond_.first_use_ever)
            flags = (
                ui.WindowFlags_.no_scrollbar | ui.WindowFlags_.no_scroll_with_mouse if name != self.plots_name else 0
            )
            visible, self._open[name] = ui.begin(name, True, flags)
            if visible:
                if name == self.plots_name:
                    ws.gui.ui.draw_section("Document: plots", lambda: plots.draw_content(ws.gui.ui))
                else:
                    image_name = label.removeprefix("Image · ")
                    ws.gui.ui.draw_section(
                        f"Document: image {image_name}",
                        lambda image_name=image_name: images.draw_content(ws.gui.ui, image_name),
                    )
            ui.end()
        return self.viewport_rect if self.viewport_visible else (0, 0, 1, 1)

    @staticmethod
    def image_name(name: str) -> str:
        """Build an ImGui window name with a literal caption and stable image-specific identity."""
        return f"{name.replace('##', '# #')}###workspace_image/{name}"
