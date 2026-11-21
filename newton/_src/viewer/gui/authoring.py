# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scene-authoring presentation for an explicitly application-owned editor.

The presenter sends commands to SceneEditor. Downloads run in its asset
library, while thumbnails and native-window callbacks stay on the GL thread.
Ordinary inspection workspaces never import this module.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ...core.host_transform import transform_assign
from ...geometry.raycast import intersect_ray
from . import theme
from .widgets import PropertyEdit

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from pyglet.image import Texture

    from newton.viewer import SceneEditor

    from ..viewer_assets import AssetLibrary, AssetRecord
    from .workspace import ViewerWorkspace


class AuthoringPanel:
    """Present an authored scene without taking ownership of its physics runtime."""

    payload_type = "NEWTON_SCENE_ASSET"
    gizmo_name = "/authoring/transform"
    thumbnail_limit = 48
    primitives = (
        ("box", "Box", (1.0, 1.0, 1.0)),
        ("sphere", "Sphere", (0.8, 0.8, 0.8)),
        ("capsule", "Capsule", (0.5, 0.5, 1.4)),
        ("cylinder", "Cylinder", (0.8, 0.8, 1.0)),
    )

    def __init__(self, workspace: ViewerWorkspace, editor: SceneEditor) -> None:
        self.workspace, self.editor = workspace, editor
        self.assets: AssetLibrary | None = None
        self.query = ""
        self.scene_query = ""
        self.message = ""
        self._query_time = 0.0
        self._submitted_query = ""
        self._selected_asset: AssetRecord | None = None
        self._pending_imports: list[tuple[AssetRecord, tuple[float, float, float], int]] = []
        self._asset_activity: tuple[AssetRecord, str | None, int] | None = None
        self._asset_activity_rect: tuple[float, float, float, float] | None = None
        self._asset_activity_phase = ""
        self._textures: OrderedDict[Path, tuple[Texture, int]] = OrderedDict()
        self._bad_thumbnails: set[Path] = set()
        self._thumbnail_frame = -1
        self._thumbnail_uploads = 0
        self._payloads: dict[int, tuple[str, Any]] = {
            i + 1: ("primitive", kind) for i, (kind, _, _) in enumerate(self.primitives)
        }
        self._asset_payloads: dict[str, int] = {}
        self._double_click_payload: int | None = None
        self._card_double_clicked = False
        self._dialog_action: str | None = None
        self._dialog_request = None
        self._scale = 1.0
        self._dynamic = True
        self._color = theme.ACCENT[:3]
        self._show_gizmo = True
        self._gizmo_mode = 0
        self._gizmo_transform = wp.transform_identity()
        self._gizmo_before = None
        self._gizmo_active = False
        self._gizmo_pause = False
        self._field_edit = False
        self._drag_preview = False
        self._drag_asset_id: str | None = None
        self._pose_frame = -1
        self._pose_cache = {}
        self._placement_ray = None
        self._placement_query = None
        self._body_map_key = None
        self._body_objects = {}
        self._shape_objects = {}
        self._last_selection = None
        self._last_editor_selection = None
        self._window = workspace.viewer.renderer.window
        self._file_drop_handler = None
        if "on_file_drop" in getattr(self._window, "event_types", ()):
            self._file_drop_handler = self._on_file_drop
            self._window.push_handlers(on_file_drop=self._file_drop_handler)

    @property
    def ui(self):
        return self.workspace.gui.ui.imgui

    @property
    def widgets(self):
        return self.workspace.widgets

    @property
    def editable(self) -> bool:
        history = self.workspace.history
        return (
            not getattr(self.workspace.gui, "_loading_splash_active", False)
            and not self.editor.busy
            and not (history is not None and history.replaying)
        )

    @property
    def interacting(self) -> bool:
        return self._gizmo_active or self._field_edit or self._drag_preview

    def _run(self, action: Callable, *args, **kwargs):
        """Keep authoring errors visible without tearing down the simulation UI."""
        try:
            result = action(*args, **kwargs)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            self.message = str(error)
            return None
        self.message = ""
        return result

    def _pose(self, object_id: str):
        frame = self.ui.get_frame_count()
        if frame != self._pose_frame:
            self._pose_cache.clear()
            self._pose_frame = frame
        if object_id not in self._pose_cache:
            self._pose_cache[object_id] = self.editor.get_pose(object_id)
        return self._pose_cache[object_id]

    def poll(self) -> None:
        """Publish worker results and completed dialogs at the owning frame boundary."""
        if self.assets is not None:
            self.assets.poll()
            if self._pending_imports:
                waiting = []
                can_import = self.editable and not self.interacting and self.editor._edit_before is None
                for record, position, epoch in self._pending_imports:
                    if epoch != self.editor._document_epoch:
                        continue
                    path = self.assets.ready(record)
                    if path is not None:
                        if can_import:
                            object_id = self._import_path(path, position)
                            self._asset_activity = (record, object_id, epoch) if object_id is not None else None
                        else:
                            waiting.append((record, position, epoch))
                    elif error := self.assets.download_error(record):
                        self.message = error
                    else:
                        waiting.append((record, position, epoch))
                self._pending_imports = waiting
        if self._dialog_action and not getattr(self.workspace.gui, "_loading_splash_active", False):
            path = self.workspace.gui.ui.consume_file_dialog_result()
            if path:
                action, self._dialog_action = self._dialog_action, None
                self._dialog_request = None
                if action == "import":
                    self._import_path(path, self.world_position())
                elif action == "save":
                    path = str(Path(path).with_suffix(".usda")) if not Path(path).suffix else path
                    self._run(self.editor.save_usd, path)
                else:
                    self._run(self.editor.load_usd, path)
            elif self.workspace.gui.ui._pending_file_dialog is None:
                self._dialog_action = None
                self._dialog_request = None
        self._sync_selection()

    def _sync_selection(self) -> None:
        objects = self.editor.objects
        runtime = self.editor._runtime
        bindings = runtime.bindings if runtime is not None else {}
        key = (id(self.workspace.model), tuple(obj.id for obj in objects))
        rebuilt = key != self._body_map_key
        if rebuilt:
            self._placement_query = None
            self._body_objects = {body: obj.id for obj in objects for body in self.editor.body_indices(obj.id)}
            self._shape_objects = {
                shape: obj.id for obj in objects if obj.id in bindings for shape in range(*bindings[obj.id].shapes)
            }
            self._body_map_key = key
        if rebuilt or self.editor.selected_id != self._last_editor_selection:
            indices = self.editor.body_indices(self.editor.selected_id) if self.editor.selected_id else ()
            keys = [("body", index) for index in indices]
            if not keys and self.editor.selected_id in bindings:
                keys = [("shape", index) for index in range(*bindings[self.editor.selected_id].shapes)]
            self.workspace.select(keys[0] if keys else None)
            for selected in keys[1:]:
                self.workspace.select(selected, toggle=True)
            self._last_editor_selection = self.editor.selected_id
            self._last_selection = self.workspace.selected
        elif self.workspace.selected != self._last_selection:
            selected = self.workspace.selected
            object_id = None
            if selected is not None:
                if selected[0] == "shape":
                    object_id = self._shape_objects.get(selected[1])
                    if object_id is None and self.workspace.tree is not None:
                        selected = self.workspace.tree.parents.get(selected)
                if object_id is None and selected is not None and selected[0] == "body":
                    object_id = self._body_objects.get(selected[1])
            self.editor.select(object_id)
            self._last_editor_selection = object_id
            self._last_selection = self.workspace.selected
            if object_id is not None:
                self._selected_asset = None

    def select(self, object_id: str | None) -> None:
        self.editor.select(object_id)
        self._selected_asset = None
        self.workspace.show_right = True
        self._sync_selection()

    def world_position(self, mouse: tuple[float, float] | None = None, *, lift: float = 0.0):
        """Aim placement at the current visible surface, falling back to the ground."""
        x, y, width, height = self.workspace.rect
        mx, my = mouse if mouse is not None else (x + width * 0.5, y + height * 0.5)
        camera = self.workspace.viewer.camera
        px = min(max((mx - x) / max(width, 1), 0.0), 1.0) * camera.width
        py = (1.0 - min(max((my - y) / max(height, 1), 0.0), 1.0)) * camera.height
        origin, direction = camera.get_world_ray(px, py)
        origin, direction = np.asarray(origin, dtype=float), np.asarray(direction, dtype=float)
        axis = int(getattr(self.workspace.model, "up_axis", camera.up_axis))
        model, state = self.editor.model, self.editor.state_0
        runtime = self.editor._runtime
        key = (
            model,
            state.body_q if state is not None else None,
            self.editor.sim_time,
            tuple(runtime.objects.values()) if runtime is not None else (),
            tuple(origin),
            tuple(direction),
            axis,
            lift,
        )
        if self._placement_query is not None and self._placement_query[0] == key:
            return self._placement_query[1]
        distance = -1.0
        if model is not None and state is not None:
            if self._placement_ray is None:
                self._placement_ray = (
                    wp.empty(1, dtype=wp.vec3, device=model.device),
                    wp.empty(1, dtype=wp.vec3, device=model.device),
                    wp.full(1, -1, dtype=int, device=model.device),
                    wp.empty(1, dtype=float, device=model.device),
                )
            origins, directions, worlds, distances = self._placement_ray
            origins.assign(np.asarray([origin], dtype=np.float32))
            directions.assign(np.asarray([direction], dtype=np.float32))
            model.bvh_refit_shapes(state)
            intersect_ray(model, ray_origins=origins, ray_directions=directions, ray_worlds=worlds, out_dist=distances)
            distance = float(distances.numpy()[0])
        if distance >= 0:
            point = origin + direction * distance
            point[axis] += lift
        else:
            distance = -origin[axis] / direction[axis] if abs(direction[axis]) > 1.0e-6 else -1.0
            point = origin + direction * (distance if 0 < distance < 1000 else 3.0)
            point[axis] = lift
        position = tuple(float(value) for value in point)
        self._placement_query = key, position
        return position

    def _primitive_size(self, kind: str) -> tuple[float, float, float]:
        return next(tuple(value * self._scale for value in size) for key, _, size in self.primitives if key == kind)

    def _add_primitive(self, kind: str, position=None) -> None:
        if not self.editable:
            return
        size = self._primitive_size(kind)
        axis = int(self.workspace.model.up_axis)
        position = position or self.world_position(lift=size[axis] * 0.5)
        object_id = self._run(
            self.editor.add_primitive,
            kind,
            position=position,
            size=size,
            color=self._color,
            dynamic=self._dynamic,
        )
        if object_id is not None:
            self._run(self.editor.place_object, object_id)
            self.select(object_id)

    def _import_path(self, path, position) -> str | None:
        object_id = self._run(self.editor.add_usd, str(path), position=position)
        if object_id is not None:
            self._run(self.editor.place_object, object_id)
            self.select(object_id)
        return object_id

    def _place_asset(self, record: AssetRecord, position=None) -> None:
        if not self.editable or self.assets is None:
            return
        position = position or self.world_position()
        epoch = self.editor._document_epoch
        path = self.assets.ready(record)
        if path is not None:
            object_id = self._import_path(path, position)
            self._asset_activity = (record, object_id, epoch) if object_id is not None else None
        else:
            self.assets.download(record)
            self._pending_imports.append((record, position, epoch))
            self._asset_activity = record, None, epoch
            self.message = f"Downloading {record.name}…"

    def _open_dialog(self, action: str) -> None:
        backend = self.workspace.gui.ui
        if backend._pending_file_dialog is not None:
            return
        backend.consume_file_dialog_result()
        self._dialog_action = action
        if action == "save":
            backend.open_save_file_dialog("Save authored USD scene")
        else:
            backend.open_load_file_dialog("Import USD asset" if action == "import" else "Open authored USD scene")
        self._dialog_request = backend._pending_file_dialog

    def _on_file_drop(self, x: int, y: int, paths: list[str]) -> bool | None:
        if not self.editable:
            return None
        position = self.world_position((x, self.ui.get_io().display_size.y - y))
        accepted = False
        for path in paths:
            if Path(path).suffix.lower() in (".usd", ".usda", ".usdc", ".usdz"):
                self._import_path(path, position)
                accepted = True
        return True if accepted else None

    def draw(self) -> None:
        """Draw the asset palette in the existing left workspace panel."""
        ui, w, s = self.ui, self.widgets, self.widgets.scale
        ui.text_colored(ui.ImVec4(*theme.ACCENT), "SCENE AUTHORING")
        engine = "XPBD"
        if self.editor.solver_type.value == "mujoco":
            engine = "MuJoCo CPU" if self.editor.device.is_cpu else "MuJoCo Warp"
        ui.text_disabled(f"Solver: {engine}")
        ui.spacing()
        if w.icon_button("##author_open", "folder", "Open authored scene", size=24):
            self._open_dialog("open")
        ui.same_line(spacing=4 * s)
        if w.icon_button("##author_save", "documents", "Save scene as USD", size=24):
            self._open_dialog("save")
        ui.same_line(spacing=12 * s)
        if w.icon_button("##author_undo", "undo", "Undo", enabled=self.editable and self.editor.can_undo, size=24):
            self._run(self.editor.undo)
        ui.same_line(spacing=4 * s)
        if w.icon_button("##author_redo", "last", "Redo", enabled=self.editable and self.editor.can_redo, size=24):
            self._run(self.editor.redo)
        ui.same_line(spacing=12 * s)
        if w.icon_button("##author_reset", "reset", "Reset object poses", enabled=self.editable, size=24):
            self._run(self.editor.reset_scene)
        ui.spacing()
        if ui.begin_tab_bar("##authoring_tabs"):
            for title, content in (
                ("Create", self._draw_create),
                ("Assets", self._draw_assets),
                ("Scene", self._draw_scene),
            ):
                if w.tab_item(title)[0]:
                    ui.begin_child("##authoring_" + title, ui.ImVec2(0, 0))
                    try:
                        content()
                    finally:
                        ui.end_child()
                    ui.end_tab_item()
            ui.end_tab_bar()

    def _draw_create(self) -> None:
        ui, w, s = self.ui, self.widgets, self.widgets.scale
        ui.text_disabled("Drag a shape into the viewport.")
        ui.spacing()
        columns = 2 if ui.get_content_region_avail().x >= 210 * s else 1
        width = (ui.get_content_region_avail().x - (columns - 1) * 8 * s) / columns
        ui.begin_disabled(not self.editable)
        for index, (kind, title, _) in enumerate(self.primitives):
            if index % columns:
                ui.same_line(spacing=8 * s)
            if self._card(title, kind, width, 102 * s, index + 1):
                self._add_primitive(kind)
        ui.end_disabled()
        ui.spacing()
        with w.group("New objects") as opened:
            if opened:
                edit = w.number("Scale", self._scale, editable=True, bounds=(0.02, 100), speed=0.01)
                if edit.changed:
                    self._scale = float(edit.value)
                _, self._dynamic = ui.checkbox("Dynamic", self._dynamic)
                if ui.is_item_hovered():
                    ui.set_tooltip("Dynamic objects fall and collide. Fixed objects stay in place.")
                _, color = ui.color_edit3("Color", list(self._color))
                self._color = tuple(color)
        ui.spacing()
        if w.button("Import USD…", ui.ImVec2(-1, 30 * s)):
            self._open_dialog("import")
        ui.text_wrapped("Drop a local USD file anywhere in the window to import it.")
        self._draw_status()

    def _ensure_assets(self) -> AssetLibrary:
        if self.assets is None:
            from ..viewer_assets import AssetLibrary  # noqa: PLC0415

            self.assets = AssetLibrary()
            self.assets.search(self.query)
        return self.assets

    def _draw_assets(self) -> None:
        ui, w, s = self.ui, self.widgets, self.widgets.scale
        library = self._ensure_assets()
        ui.text_colored(ui.ImVec4(*theme.TEXT), "SimReady")
        ui.same_line()
        ui.text_disabled("NVIDIA · Hugging Face")
        available = ui.get_content_region_avail().x
        ui.set_next_item_width(max(30, available - 31 * s))
        changed, self.query = ui.input_text_with_hint("##simready_search", "Search assets…", self.query)
        if changed:
            self._query_time = perf_counter()
        ui.same_line(spacing=5 * s)
        if w.icon_button("##refresh_assets", "reset", "Refresh SimReady catalog", size=24):
            library.refresh()
        if self.query != self._submitted_query and perf_counter() - self._query_time > 0.2:
            library.search(self.query)
            self._submitted_query = self.query
        ui.text_disabled(f"{len(library.records):,} assets")
        # Reserve one status row: thumbnail jobs must not move the cards under
        # a held pointer when their loading state or metadata changes.
        status = library.error or ((library.progress or "Loading the catalog…") if library.loading else "")
        ui.text_colored(ui.ImVec4(*(theme.CONSOLE_WARNING if library.error else theme.TEXT_DISABLED)), status or " ")
        if status and ui.is_item_hovered():
            ui.set_tooltip(status)
        records = library.records
        columns = 2 if ui.get_content_region_avail().x >= 210 * s else 1
        width = (ui.get_content_region_avail().x - (columns - 1) * 8 * s) / columns
        clipper = ui.ListClipper()
        clipper.begin((len(records) + columns - 1) // columns, width + 30 * s + ui.get_style().item_spacing.y)
        while clipper.step():
            for row in range(clipper.display_start, clipper.display_end):
                for column in range(columns):
                    index = row * columns + column
                    if index >= len(records):
                        break
                    record = records[index]
                    if column:
                        ui.same_line(spacing=8 * s)
                    payload = self._asset_payload(record)
                    selected = self._selected_asset is not None and self._selected_asset.id == record.id
                    if self._card(
                        record.name, "mesh", width, width + 30 * s, payload, record=record, selected=selected
                    ):
                        self.select(None)
                        self._selected_asset = record
                        if self._card_double_clicked:
                            self._place_asset(record)
        clipper.end()
        if not records and not library.loading:
            ui.text_wrapped("No matching assets. Try a shorter search or refresh the catalog.")
        self._draw_status()

    def _asset_payload(self, record: AssetRecord) -> int:
        """Share one pinned asset identity between its card and inspector preview."""
        payload = self._asset_payloads.get(record.id)
        if payload is None:
            payload = len(self._asset_payloads) + len(self.primitives) + 1
            self._asset_payloads[record.id] = payload
        self._payloads[payload] = ("asset", record)
        return payload

    def _card(self, title, icon, width, height, payload, *, record=None, selected=False) -> bool:
        ui, w, s = self.ui, self.widgets, self.widgets.scale
        position = ui.get_cursor_screen_pos()
        clicked = ui.invisible_button(f"##asset_card_{payload}", ui.ImVec2(width, height))
        hovered = ui.is_item_hovered()
        self._card_double_clicked = clicked and self._double_click_payload == payload
        if hovered and ui.is_mouse_clicked(0):
            self._double_click_payload = payload if ui.is_mouse_double_clicked(0) else None
        elif clicked:
            self._double_click_payload = None
        if ui.is_item_visible():
            draw = ui.get_window_draw_list()
            end = ui.ImVec2(position.x + width, position.y + height)
            background = theme.GROUP_ACTIVE if selected else theme.GROUP_HOVERED if hovered else theme.GROUP_BACKGROUND
            draw.add_rect_filled(position, end, w.color(background), 5 * s)
            if selected:
                draw.add_rect(position, end, w.color(theme.ACCENT), 5 * s)
            image_height = height - 27 * s
            reference = self._thumbnail(record) if record is not None else None
            if reference is not None:
                texture, texture_ref = reference
                factor = min((width - 8 * s) / texture.width, (image_height - 8 * s) / texture.height)
                image_width, image_height = texture.width * factor, texture.height * factor
                start = ui.ImVec2(position.x + (width - image_width) / 2, position.y + 4 * s)
                draw.add_image(
                    texture_ref,
                    start,
                    ui.ImVec2(start.x + image_width, start.y + image_height),
                    ui.ImVec2(0, 1),
                    ui.ImVec2(1, 0),
                )
            else:
                size = min(36 * s, width - 20 * s)
                w.icon(
                    icon,
                    (position.x + (width - size) / 2, position.y + (image_height - size) / 2),
                    size,
                    active=hovered,
                )
            label = title.replace("_", " ")
            while len(label) > 1 and ui.calc_text_size(label).x > width - 14 * s:
                label = label[:-2] + "…"
            draw.add_text(
                ui.ImVec2(position.x + 7 * s, position.y + height - 22 * s),
                w.color(theme.TEXT),
                label,
            )
        if hovered:
            ui.set_tooltip(title + "\nDrag into the viewport to place" + (" · Double-click to add" if record else ""))
        self._drag_source(title, payload)
        return clicked

    def _drag_source(self, title: str, payload: int) -> None:
        """Make the last submitted widget a source for viewport asset placement."""
        ui = self.ui
        if self.editable and ui.begin_drag_drop_source():
            # A quick second press may become a drag. Only a completed click
            # can place on double-click; an early import would cancel this drag.
            self._double_click_payload = None
            ui.set_drag_drop_payload_py_id(self.payload_type, payload)
            ui.text_unformatted(title)
            ui.text_disabled("Drop in the viewport")
            ui.end_drag_drop_source()

    def _thumbnail(self, record):
        path = self.assets.thumbnail(record)
        if path is None or path in self._bad_thumbnails:
            return None
        frame = self.ui.get_frame_count()
        if frame != self._thumbnail_frame:
            self._thumbnail_frame, self._thumbnail_uploads = frame, 0
        cached = self._textures.get(path)
        if cached is None:
            # Decode/upload at most one visible thumbnail per frame. A warm
            # library can contain thousands of cached, high-resolution images.
            if self._thumbnail_uploads:
                return None
            self._thumbnail_uploads += 1
            if len(self._textures) >= self.thumbnail_limit:
                victim = next((key for key, (_, used) in self._textures.items() if used != frame), None)
                if victim is None:
                    return None
                texture, _ = self._textures.pop(victim)
                texture.delete()
            from pyglet import gl, image

            previous = gl.GLint()
            gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D, previous)
            try:
                texture = image.load(str(path)).get_texture()
                gl.glBindTexture(texture.target, texture.id)
                gl.glTexParameteri(texture.target, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
                gl.glTexParameteri(texture.target, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            except (OSError, ValueError, image.codecs.ImageDecodeException):
                self._bad_thumbnails.add(path)
                return None
            finally:
                gl.glBindTexture(gl.GL_TEXTURE_2D, previous.value)
            cached = texture, frame
            self._textures[path] = cached
        texture = cached[0]
        self._textures[path] = texture, frame
        self._textures.move_to_end(path)
        return texture, self.ui.ImTextureRef(texture.id)

    def _draw_scene(self) -> None:
        ui = self.ui
        ui.set_next_item_width(-1)
        _, self.scene_query = ui.input_text_with_hint("##scene_search", "Find an object…", self.scene_query)
        ui.spacing()
        query = self.scene_query.casefold()
        objects = [obj for obj in self.editor.objects if query in obj.name.casefold()]
        for obj in objects:
            ui.push_id(obj.id)
            selected = obj.id == self.editor.selected_id
            if ui.selectable(obj.name, selected)[0]:
                self.select(obj.id)
            if ui.begin_popup_context_item("##object_menu"):
                if ui.menu_item("Duplicate", "", False, self.editable)[0]:
                    self._duplicate(obj.id)
                if ui.menu_item("Delete", "", False, self.editable)[0]:
                    self._run(self.editor.remove_object, obj.id)
                ui.end_popup()
            ui.pop_id()
        if not objects:
            ui.text_wrapped("Your scene is empty. Add a shape or drag in a SimReady asset.")
        ui.spacing()
        ui.text_disabled(f"{len(self.editor.objects)} objects")
        self._draw_status()

    def _draw_status(self) -> None:
        ui = self.ui
        ui.spacing()
        if self.editor.busy:
            ui.text_disabled("Preparing USD geometry…")
        if self.editor.error or self.message:
            ui.push_text_wrap_pos()
            ui.text_colored(ui.ImVec4(*theme.CONSOLE_WARNING), self.editor.error or self.message)
            ui.pop_text_wrap_pos()
        elif self.editor.status:
            ui.text_wrapped(self.editor.status)
        if self.editor.pending and not self.editor.auto_apply:
            if self.widgets.button("Apply changes", ui.ImVec2(-1, 0)):
                self._run(self.editor.request_commit)

    def _apply_edit(self, object_id: str, field: str, edit: PropertyEdit) -> None:
        if edit.changed:
            if not self._field_edit:
                self.editor.begin_edit()
                self._field_edit = True
            value = edit.value.tolist() if isinstance(edit.value, np.ndarray) else edit.value
            self._run(self.editor.update_object, object_id, **{field: value})
            if field == "size" and not self.message:
                obj = self.editor.get_object(object_id)
                if obj.source is not None:
                    self._run(
                        self.editor.preview_usd,
                        obj.source,
                        position=self._pose(object_id)[0],
                        scale=value,
                        color=obj.color,
                    )
                else:
                    self.editor.preview_primitive(
                        obj.kind.value, position=self._pose(object_id)[0], size=value, color=obj.color
                    )
        if self._field_edit and (edit.finished or not self.ui.is_any_item_active()):
            self.editor.end_edit()
            self.editor.clear_preview()
            self._field_edit = False

    def _duplicate(self, object_id: str) -> None:
        new_id = self._run(self.editor.duplicate_object, object_id)
        if new_id is not None:
            self._run(self.editor.place_object, new_id)
            self.select(new_id)

    def draw_inspector(self) -> None:
        """Present authored object values, leaving mutation and rebuild decisions to SceneEditor."""
        ui, w, s = self.ui, self.widgets, self.widgets.scale
        object_id = self.editor.selected_id
        if object_id is None:
            if self._selected_asset is not None:
                self._draw_asset_details()
            else:
                ui.text_colored(ui.ImVec4(*theme.ACCENT), "OBJECT PROPERTIES")
                ui.text_wrapped("Select an object to move, rotate, resize, and change its material.")
                ui.spacing()
                ui.text_disabled("Move and rotate directly in the viewport.")
            return
        obj = self.editor.get_object(object_id)
        ui.text_colored(ui.ImVec4(*theme.ACCENT), "OBJECT PROPERTIES")
        ui.push_id(object_id)
        ui.begin_disabled(not self.editable)
        ui.set_next_item_width(-1)
        changed, name = ui.input_text("##object_name", obj.name)
        self._apply_edit(
            object_id, "name", PropertyEdit(name, changed, ui.is_item_activated(), ui.is_item_deactivated_after_edit())
        )
        ui.text_disabled(obj.kind.value.upper())
        ui.spacing()
        if w.icon_button("##duplicate_object", "copy", "Duplicate object", label="Duplicate", size=24):
            self._duplicate(object_id)
        ui.same_line(spacing=12 * s)
        if w.icon_button("##delete_object", "clear", "Delete object", label="Delete", size=24):
            self._run(self.editor.remove_object, object_id)
            ui.end_disabled()
            ui.pop_id()
            return
        position, rotation = self._pose(object_id)
        with w.group("Transform") as opened:
            if opened:
                _, self._show_gizmo = ui.checkbox("Viewport handles", self._show_gizmo)
                if self._show_gizmo:
                    ui.same_line()
                    _, self._gizmo_mode = ui.combo("##gizmo_mode", self._gizmo_mode, ["Move", "Rotate"])
                self._apply_edit(object_id, "position", w.vec3("Position", position, editable=True, unit="m"))
                self._apply_edit(object_id, "rotation", w.quat("Rotation", rotation, editable=True))
                size = np.asarray(obj.size)
                if obj.kind.value == "sphere":
                    edit = w.number("Diameter", float(size[0]), editable=True, bounds=(0.002, 1000), unit="m")
                    edit.value = (float(edit.value),) * 3
                elif obj.kind.value in ("cylinder", "capsule"):
                    edit = w.vec2("Diameter / height", (size[0], size[2]), editable=True, labels=("D", "H"), unit="m")
                    edit.value = (float(edit.value[0]), float(edit.value[0]), float(edit.value[1]))
                else:
                    edit = w.vec3(
                        "Scale" if obj.kind.value == "usd" else "Size",
                        size,
                        editable=True,
                        unit="" if obj.kind.value == "usd" else "m",
                    )
                self._apply_edit(object_id, "size", edit)
        with w.group("Material") as opened:
            if opened:
                changed, color = ui.color_edit3("Color", list(obj.color))
                self._apply_edit(
                    object_id,
                    "color",
                    PropertyEdit(color, changed, ui.is_item_activated(), ui.is_item_deactivated_after_edit()),
                )
                changed, visible = ui.checkbox("Visible", obj.visible)
                if changed:
                    self._run(self.editor.update_object, object_id, visible=visible)
                self._apply_edit(
                    object_id, "friction", w.number("Friction", obj.friction, editable=True, bounds=(0, 10))
                )
        with w.group("Physics") as opened:
            if opened:
                changed, dynamic = ui.checkbox("Dynamic", obj.dynamic)
                if changed:
                    self._run(self.editor.update_object, object_id, dynamic=dynamic)
                if obj.dynamic:
                    self._apply_edit(
                        object_id, "mass", w.number("Mass", obj.mass, editable=True, bounds=(0.001, 100000), unit="kg")
                    )
                else:
                    ui.text_disabled("Fixed in the scene")
        ui.end_disabled()
        ui.pop_id()
        if not self.editable and self.workspace.history is not None and self.workspace.history.replaying:
            ui.text_wrapped("Return to the live frame to edit this object.")
        self._draw_status()

    def _draw_asset_details(self) -> None:
        ui, s, record = self.ui, self.widgets.scale, self._selected_asset
        if self.assets is not None:
            record = next((current for current in self.assets.records if current.id == record.id), record)
            self._selected_asset = record
        ui.text_colored(ui.ImVec4(*theme.ACCENT), "SIMREADY ASSET")
        ui.text_wrapped(record.name)
        reference = self._thumbnail(record)
        if reference is not None:
            texture, texture_ref = reference
            width = min(ui.get_content_region_avail().x, 280 * s)
            ui.push_style_var(ui.StyleVar_.frame_padding, ui.ImVec2(0, 0))
            ui.image_button(
                f"##asset_preview_{record.id}",
                texture_ref,
                ui.ImVec2(width, width * texture.height / texture.width),
                ui.ImVec2(0, 1),
                ui.ImVec2(1, 0),
            )
            ui.pop_style_var()
            self._drag_source(record.name, self._asset_payload(record))
        ui.text_wrapped(record.repo_id)
        if record.license:
            ui.text_disabled(record.license)
        ui.spacing()
        ui.begin_disabled(not self.editable)
        if self.widgets.button("Place in scene", ui.ImVec2(-1, 32 * s)):
            self._place_asset(record)
        ui.end_disabled()
        ui.text_wrapped("Drag this asset into the viewport to choose its position.")
        self._draw_status()

    def drop_target(self) -> None:
        """Accept our palette payload on the just-submitted viewport image."""
        ui = self.ui
        payload = ui.get_drag_drop_payload_py_id()
        if payload is None or payload.type != self.payload_type or payload.data_id not in self._payloads:
            self._drag_asset_id = None
            if self._drag_preview:
                self.editor.clear_preview()
                self._drag_preview = False
            self._draw_asset_activity()
            return
        kind, value = self._payloads[payload.data_id]
        mouse = ui.get_io().mouse_pos
        hovered = ui.is_item_hovered(ui.HoveredFlags_.allow_when_blocked_by_active_item)
        point = None
        if hovered and self.editable:
            self._drag_preview = True
            aim = self.world_position((mouse.x, mouse.y))
            if kind == "primitive":
                size = self._primitive_size(value)
                point = self._run(
                    self.editor.preview_primitive, value, position=aim, size=size, color=self._color, placement=True
                )
            else:
                self._asset_activity = value, None, self.editor._document_epoch
                path = self.assets.ready(value) if self.assets is not None else None
                if path is not None:
                    point = self._run(
                        self.editor.preview_usd,
                        path,
                        position=aim,
                        color=theme.CONTACT_SOFT_RIGID[:3],
                        placement=True,
                    )
                    if point is None:
                        if self.message:
                            self._asset_activity = None
                        else:
                            ui.set_tooltip("Preparing collision preview…")
                else:
                    self.editor.clear_preview()
                    if self.assets is not None:
                        if self._drag_asset_id != value.id:
                            self._drag_asset_id = value.id
                            self.assets.download(value)
                        if error := self.assets.download_error(value):
                            self.message = error
                    ui.set_tooltip(self.message or "Downloading geometry for the collision preview…")
        else:
            self.editor.clear_preview()
            self._drag_preview = False
        if ui.begin_drag_drop_target():
            try:
                delivered = ui.accept_drag_drop_payload_py_id(self.payload_type)
                if delivered is not None and self.editable:
                    mouse_position = (mouse.x, mouse.y)
                    if kind == "primitive":
                        if point is not None:
                            self._add_primitive(value, point)
                    else:
                        if point is not None or not self.message:
                            self._place_asset(value, point or self.world_position(mouse_position))
                    self.editor.clear_preview()
                    self._drag_preview = False
            finally:
                ui.end_drag_drop_target()
        self._draw_asset_activity()

    def _draw_asset_activity(self) -> None:
        """Show asynchronous asset work without capturing input or interrupting a drag."""
        self._asset_activity_rect = None
        self._asset_activity_phase = ""
        if self._asset_activity is None and self._pending_imports:
            record, _, epoch = self._pending_imports[0]
            self._asset_activity = record, None, epoch
        if self._asset_activity is None or self.assets is None:
            return
        record, object_id, epoch = self._asset_activity
        editor, ui = self.editor, self.ui
        payload = ui.get_drag_drop_payload_py_id()
        held = (
            self._payloads.get(payload.data_id) if payload is not None and payload.type == self.payload_type else None
        )
        dragging = self._drag_preview and held is not None and held[0] == "asset" and held[1].id == record.id
        queued = any(item.id == record.id and version == epoch for item, _, version in self._pending_imports)
        published = editor._runtime is not None and object_id in editor._runtime.bindings
        if (
            epoch != editor._document_epoch
            or published
            or self.assets.download_error(record)
            or (object_id is not None and (editor.error or not any(obj.id == object_id for obj in editor.objects)))
            or (object_id is None and not dragging and not queued)
        ):
            self._asset_activity = None
            return
        if object_id is not None:
            phase = (
                "Preparing geometry" if editor.busy else "Adding to scene" if editor.auto_apply else "Ready to apply"
            )
        elif self.assets.ready(record) is None:
            phase = "Downloading"
        elif dragging:
            phase = "Ready to place" if editor._preview is not None else "Preparing preview"
        else:
            phase = "Waiting to add"
        self._asset_activity_phase = phase
        if self.workspace.gui._loading_splash_active:
            return
        s = self.widgets.scale
        vx, vy, vw, vh = self.workspace.dockspace.viewport_rect
        width, height = min(360 * s, vw - 24 * s), 68 * s
        if width < 150 * s or vh < height + 24 * s:
            return
        x, y = vx + (vw - width) * 0.5, vy + vh - height - 12 * s
        self._asset_activity_rect = x, y, width, height
        draw, color = ui.get_window_draw_list(), self.widgets.color
        start, end = ui.ImVec2(x, y), ui.ImVec2(x + width, y + height)
        draw.push_clip_rect(start, end, True)
        draw.add_rect_filled(start, end, color((*theme.VIEWPORT_BACKGROUND[:3], 0.96)), 9 * s)
        draw.add_rect(start, end, color(theme.VIEWPORT_BORDER), 9 * s)
        center = ui.ImVec2(x + 26 * s, y + height * 0.5)
        if phase in ("Ready to place", "Ready to apply", "Waiting to add"):
            self.widgets.icon("pause", (center.x - 10 * s, center.y - 10 * s), 20 * s, tint=theme.ACCENT)
        else:
            draw.add_circle(center, 9 * s, color((*theme.MUTED[:3], 0.25)), 24, 2 * s)
            angle = (ui.get_time() * 4.0) % (2 * np.pi)
            draw.path_arc_to(center, 9 * s, angle, angle + 4.5, 24)
            draw.path_stroke(color(theme.ACCENT), 2 * s)
        queued_count = len(self._pending_imports)
        label = f"{phase} · {queued_count} queued" if queued_count > 1 else phase
        draw.add_text(ui.ImVec2(x + 48 * s, y + 13 * s), color(theme.MUTED), label)
        name = record.name.replace("_", " ")
        while len(name) > 1 and ui.calc_text_size(name).x > width - 60 * s:
            name = name[:-2] + "…"
        draw.add_text(ui.ImVec2(x + 48 * s, y + 34 * s), color(theme.TEXT), name)
        draw.pop_clip_rect()

    def prepare_gizmo(self) -> None:
        """Log a persistent mutable transform through the existing gizmo integration."""
        object_id = self.editor.selected_id
        runtime = self.editor._runtime
        binding = runtime.bindings.get(object_id) if runtime is not None else None
        if (
            not self._show_gizmo
            or not self.editable
            or object_id is None
            or binding is None
            or self.ui.get_drag_drop_payload_py_id() is not None
        ):
            self._finish_gizmo()
            return
        if not self._gizmo_active:
            position, rotation = self._pose(object_id)
            transform_assign(self._gizmo_transform, (*position, *rotation))
        self._gizmo_before = np.asarray(self._gizmo_transform).copy()
        axes = (0, 1, 2)
        self.workspace.viewer.log_gizmo(
            self.gizmo_name,
            self._gizmo_transform,
            translate=axes if self._gizmo_mode == 0 else (),
            rotate=axes if self._gizmo_mode == 1 else (),
        )

    def finish_gizmo(self) -> None:
        """Apply only actual handle edits after ImGuizmo updates its transform."""
        active = bool(self.workspace.gui._gizmo_active.get(self.gizmo_name))
        if active and not self._gizmo_active:
            self._gizmo_pause = self.workspace.viewer.is_paused()
            self.workspace.viewer._paused = True
            self.editor.begin_edit()
            self._gizmo_active = True
        values = np.asarray(self._gizmo_transform)
        if self._gizmo_active and self._gizmo_before is not None and not np.allclose(values, self._gizmo_before):
            self._run(
                self.editor.update_object,
                self.editor.selected_id,
                position=tuple(values[:3]),
                rotation=tuple(values[3:]),
            )
        if self._gizmo_active and not active:
            self._finish_gizmo()

    def _finish_gizmo(self) -> None:
        if self._gizmo_active:
            self.editor.end_edit()
            self.workspace.viewer._paused = self._gizmo_pause
            self._gizmo_active = False
        self._gizmo_before = None

    def cancel_interaction(self) -> None:
        """Finish the current edit when the window or its authoring panels lose focus."""
        self._finish_gizmo()
        if self._field_edit:
            self.editor.end_edit()
            self._field_edit = False
        self.editor.clear_preview()
        self._drag_preview = False

    def close(self) -> None:
        """Release worker and GL resources before the native window is destroyed."""
        self.cancel_interaction()
        if self._dialog_action is not None:
            backend = self.workspace.gui.ui
            if self._dialog_request is backend._pending_file_dialog and self._dialog_request is not None:
                self._dialog_request.kill()
                backend._pending_file_dialog = None
            if backend._pending_file_dialog is None:
                backend.consume_file_dialog_result()
            self._dialog_action = self._dialog_request = None
        if self._file_drop_handler is not None:
            self._window.remove_handlers(on_file_drop=self._file_drop_handler)
            self._file_drop_handler = None
        if self.assets is not None:
            self.assets.close()
            self.assets = None
        if self._textures and self._window.context is not None:
            self._window.switch_to()
            for texture, _ in self._textures.values():
                texture.delete()
        self._textures.clear()
        self._pending_imports.clear()
        self._asset_activity = self._asset_activity_rect = None
        self._asset_activity_phase = ""
        self._placement_query = None
