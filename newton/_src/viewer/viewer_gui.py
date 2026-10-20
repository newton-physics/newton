# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

import math
from collections.abc import Callable
from time import perf_counter
from typing import Any, Literal

import numpy as np

import newton as nt
from newton.selection import ArticulationView

from ..core.types import Axis
from .gl.gui import UI
from .transform import transform_assign, transform_assign_matrix, transform_to_matrix

# Width of the left sidebar in logical (96-DPI) pixels.
_SIDEBAR_WIDTH_PX = 300.0


class ViewerGui:
    """Shared ImGui rendering for concrete viewers (GL / RTX)."""

    def __init__(self, viewer, window):
        self._viewer = viewer
        self.ui = UI(window)
        self.ui.key_press_filter = self._filter_gui_key_press

        # UI callback registry. Backend viewers (GL/RTX) register their
        # ``_ui_populate_rendering_panel`` here, examples register their
        # ``side`` / ``free`` / ``panel`` / ``stats`` callbacks.
        self._ui_callbacks: dict[str, list] = {"side": [], "stats": [], "free": [], "panel": [], "rendering": []}

        # Loading-splash overlay state.
        self._loading_splash_active: bool = False
        self._loading_splash_text: str | None = None

        # Camera keyboard movement (shared with GL/RTX)
        self._cam_vel = np.zeros(3, dtype=np.float32)
        self._cam_damp_tau = 0.083
        self._camera_orbit_sensitivity = 0.1
        self._camera_dolly_scroll_sensitivity = 0.15
        self._camera_dolly_drag_sensitivity = 0.01
        self._frame_key_down = False
        self._follow_held = False
        self._follow_latched = False
        self._follow_center = None
        window.push_handlers(on_deactivate=self._release_camera_keys)

        # Gizmo active-frame tracking (handles snap_to on release)
        self._gizmo_active = {}
        self._gizmo_hovered = False
        self._frame_prepared = False
        self.workspace = None
        # Created on the first UI frame; non-GL viewers keep their existing UI.
        self.workspace_enabled = type(viewer).__name__ == "ViewerGL"
        self.console_capture = None
        if self.workspace_enabled and self.is_available:
            from .viewer_console import ConsoleCapture  # noqa: PLC0415

            self.ui.io.config_flags |= self.ui.imgui.ConfigFlags_.docking_enable
            self.ui.io.config_windows_move_from_title_bar_only = True
            # Attach before constructing an example, so importer diagnostics survive loading.
            self.console_capture = ConsoleCapture()
            window.push_handlers(on_close=self.close_console)
        self._workspace_example = None
        self._frame_camera_pending = False
        self._selection_press = None

        # FPS tracking
        self._fps_history: list[float] = []
        self._last_fps_time: float = perf_counter()
        self._fps_frame_count: int = 0
        self._current_fps: float = 0.0
        # Visual / collision shape counts for the stats panel, refreshed by
        # update_shape_counts() when the viewer is given a model.
        self._shape_visual_count: int | None = None
        self._shape_collision_count: int | None = None

        # Selection panel state (UI-local, not simulation state).
        self._selection_ui_state = {
            "selected_articulation_pattern": "*",
            "selected_articulation_view": None,
            "selected_attribute": "joint_q",
            "attribute_options": ["joint_q", "joint_qd", "joint_f", "body_q", "body_qd"],
            "include_joints": "",
            "exclude_joints": "",
            "include_links": "",
            "exclude_links": "",
            "show_values": False,
            "selected_batch_idx": 0,
            "error_message": "",
        }

    @property
    def is_available(self) -> bool:
        return bool(self.ui and self.ui.is_available)

    @property
    def show_ui(self) -> bool:
        return bool(getattr(self._viewer, "show_ui", True))

    @show_ui.setter
    def show_ui(self, value: bool):
        self._viewer.show_ui = bool(value)

    def is_capturing(self) -> bool:
        if not self.is_available:
            return False
        return self.ui.is_capturing()

    def is_mouse_capturing(self) -> bool:
        if not self.is_available:
            return False
        if (
            getattr(self, "_loading_splash_active", False)
            or (self.show_ui and getattr(self, "_gizmo_hovered", False))
            or (self.show_ui and getattr(self._viewer, "gizmo_is_using", False))
        ):
            return True
        workspace = getattr(self, "workspace", None)
        documents = getattr(workspace, "documents", None)
        if self.show_ui and documents is not None and documents.viewport_hovered:
            return False
        return bool(self.ui.io.want_capture_mouse)

    def is_keyboard_capturing(self) -> bool:
        if not self.is_available:
            return False
        return bool(self.ui.io.want_capture_keyboard)

    def on_rendering_paused(self) -> None:
        """Release camera motion and gizmos when the displayed image freezes."""
        self._cam_vel.fill(0.0)
        self._gizmo_active.clear()
        self._gizmo_hovered = False
        self._selection_press = None
        self._viewer.gizmo_is_using = False
        self._release_camera_keys()

    def _release_camera_keys(self):
        """Drop momentary navigation on focus loss without releasing a pinned follow."""
        self._frame_key_down = False
        self._follow_held = False
        self._cam_vel.fill(0.0)

    @property
    def following_selection(self):
        return getattr(self, "_follow_held", False) or getattr(self, "_follow_latched", False)

    def stop_following_selection(self):
        """A changed selection or scene must not inherit another object's follow mode."""
        self._follow_held = self._follow_latched = False
        self._follow_center = None

    def _start_follow_selection(self):
        bounds = self.workspace.selection_bounds(include_contacts=False)
        self._follow_center = bounds.mean(axis=0) if bounds is not None else None
        return self._follow_center is not None

    def toggle_follow_selection(self):
        if getattr(self, "_follow_latched", False):
            self.stop_following_selection()
        elif self.workspace is not None and self.workspace.can_follow_selection:
            self.frame_camera_on_model(contacts=False)
            self._follow_latched = self._start_follow_selection()

    def update_camera_follow(self, is_key_down):
        """Transport the orbit pivot with the displayed selection, without refitting it."""
        if getattr(self, "_frame_key_down", False):
            from pyglet.window import key

            if not is_key_down(key.F):
                self._frame_key_down = self._follow_held = False
        if not self.following_selection or self._viewer.is_rendering_paused():
            return
        if self.workspace is None or not self.workspace.can_follow_selection:
            self.stop_following_selection()
            return
        if not getattr(self, "_follow_latched", False) and self.should_ignore_keyboard_input():
            self._follow_held = False
            return
        bounds = self.workspace.selection_bounds(include_contacts=False, sync_selection=not self.show_ui)
        if bounds is not None and np.isfinite(bounds).all():
            center = bounds.mean(axis=0)
            if self._follow_center is not None:
                # Preserve a panned view or a close-up contact on the body. Only
                # target translation is inherited; orbit and zoom stay user-controlled.
                camera = self._viewer.camera
                camera.follow(np.asarray(camera.pivot) + center - self._follow_center)
            self._follow_center = center
            if hasattr(self._viewer, "_camera_dirty"):
                self._viewer._camera_dirty = True

    def should_ignore_mouse_input(self, allow_active_pick_drag: bool = False) -> bool:
        if self._viewer.is_rendering_paused():
            return True
        if allow_active_pick_drag and self.is_pick_active():
            return False
        return self.is_mouse_capturing()

    def _numeric_text_input_active(self) -> bool:
        """Recognize native numeric editors without naming particular GUI fields."""
        if not self.is_available or not self.show_ui or not self.ui.io.want_text_input:
            return False
        ui = self.ui.imgui
        active = ui.internal.get_active_id()
        if not active:
            return False
        if ui.internal.temp_input_is_active(active):
            return True
        state = ui.internal.get_input_text_state(active)
        numeric = (
            ui.InputTextFlags_.chars_decimal
            | ui.InputTextFlags_.chars_scientific
            | ui.InputTextFlags_.chars_hexadecimal
        )
        return state is not None and bool(state.flags & numeric)

    def should_ignore_keyboard_input(self) -> bool:
        if getattr(self, "_loading_splash_active", False):
            return True
        if not self.is_available or not self.show_ui:
            return False
        ui, io = self.ui.imgui, self.ui.io
        if ui.is_popup_open("", ui.PopupFlags_.any_popup):
            return True
        return io.want_text_input and not self._numeric_text_input_active()

    def _filter_gui_key_press(self, symbol, modifiers):
        """Prevent Space from both toggling playback and activating a focused widget."""
        from pyglet.window import key

        return symbol == key.SPACE and not self.should_ignore_keyboard_input()

    def is_pick_active(self) -> bool:
        viewer = self._viewer
        if not getattr(viewer, "picking_enabled", False):
            return False
        picking = getattr(viewer, "picking", None)
        if picking is None or not hasattr(picking, "is_picking"):
            return False
        return bool(picking.is_picking())

    def rotate_camera_from_drag(self, dx: float, dy: float, sensitivity: float = 0.1):
        camera = getattr(self._viewer, "camera", None)
        if camera is None:
            return
        camera.yaw -= dx * sensitivity
        camera.pitch += dy * sensitivity
        camera.pitch = max(-89.0, min(89.0, camera.pitch))
        if hasattr(self._viewer, "_camera_dirty"):
            self._viewer._camera_dirty = True

    def adjust_camera_fov_from_scroll(self, scroll_y: float, scale: float = 2.0):
        camera = getattr(self._viewer, "camera", None)
        if camera is None:
            return
        camera.fov = max(15.0, min(90.0, camera.fov - scroll_y * scale))
        if hasattr(self._viewer, "_camera_dirty"):
            self._viewer._camera_dirty = True

    def update_camera_from_keys(self, dt: float, is_key_down):
        """Update camera position from WASD/QE keys. Uses same speed and damping as ViewerGL."""
        if self.should_ignore_keyboard_input() or self._viewer.is_rendering_paused():
            return
        camera = getattr(self._viewer, "camera", None)
        if camera is None:
            return

        import pyglet

        key = pyglet.window.key
        numeric_text = self._numeric_text_input_active()
        navigation_keys = not (self.is_available and self.show_ui and self.ui.io.want_capture_keyboard)
        if self.is_available and self.show_ui and (self.ui.io.key_ctrl or self.ui.io.key_super):
            return
        forward = np.array(camera.get_front(), dtype=np.float32)
        right = np.array(camera.get_right(), dtype=np.float32)
        up = np.array(camera.get_up(), dtype=np.float32)

        # Keep motion in the horizontal plane
        forward -= up * float(np.dot(forward, up))
        right -= up * float(np.dot(right, up))
        fn = float(np.linalg.norm(forward))
        ln = float(np.linalg.norm(right))
        if fn > 1.0e-6:
            forward /= fn
        if ln > 1.0e-6:
            right /= ln

        desired = np.zeros(3, dtype=np.float32)
        if is_key_down(key.W) or (navigation_keys and is_key_down(key.UP)):
            desired += forward
        if is_key_down(key.S) or (navigation_keys and is_key_down(key.DOWN)):
            desired -= forward
        if is_key_down(key.A) or (navigation_keys and is_key_down(key.LEFT)):
            desired -= right
        if is_key_down(key.D) or (navigation_keys and is_key_down(key.RIGHT)):
            desired += right
        if is_key_down(key.Q):
            desired -= up
        if not numeric_text and is_key_down(key.E):
            desired += up

        dn = float(np.linalg.norm(desired))
        if dn > 1.0e-6:
            desired = desired / dn * self._viewer.camera_speed
        else:
            desired[:] = 0.0

        tau = max(1.0e-4, float(self._cam_damp_tau))
        self._cam_vel += (desired - self._cam_vel) * min(1.0, dt / tau)

        camera._navigation_speed_reference = max(1.0e-6, self._viewer.camera_speed * 0.6)
        camera.translate(self._cam_vel * dt)
        if hasattr(self._viewer, "_camera_dirty"):
            self._viewer._camera_dirty = True

    def frame_camera_on_model(self, *, selection=True, contacts=True):
        """Frame the selection, or all visible objects when nothing is selected."""
        viewer = self._viewer
        if viewer.is_rendering_paused():
            return
        if getattr(viewer, "model", None) is None:
            return
        from pyglet.math import Vec3 as PyVec3

        camera = getattr(viewer, "camera", None)
        if camera is None:
            return

        workspace = getattr(self, "workspace", None)
        if selection and workspace is not None:
            points = workspace.collision.focus_points() if contacts else []
            if points:
                self._frame_contact(points)
                return
            bounds = workspace.selection_bounds(include_contacts=contacts)
            if bounds is not None:
                center = bounds.mean(axis=0)
                half = (bounds[1] - bounds[0]) * 0.5
                # Fit all corners to the actual viewport, retaining orientation.
                corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
                front, right, up = map(np.asarray, (camera.get_front(), camera.get_right(), camera.get_up()))
                tangent = np.tan(np.radians(camera.fov) * 0.5)
                aspect = camera.width / max(camera.height, 1)
                distance = max(
                    np.max(np.abs(corners @ right) * 1.15 / (tangent * aspect) - corners @ front),
                    np.max(np.abs(corners @ up) * 1.15 / tangent - corners @ front),
                    camera.near * 2 + np.max(-corners @ front),
                    camera._min_pivot_distance * 1.01,
                )
                camera.pos = PyVec3(*(center - front * distance))
                camera.set_pivot(center)
                self._cam_vel[:] = 0.0
                if hasattr(viewer, "_camera_dirty"):
                    viewer._camera_dirty = True
                return

            tree = getattr(workspace, "tree", None)
            if tree is not None and tree.resources.bvhs.handles(workspace.selected):
                # Empty filters and unrecorded BVHs have no focus target. Keep
                # the camera in place instead of unexpectedly framing the scene.
                return

        min_bounds = np.array([float("inf")] * 3)
        max_bounds = np.array([float("-inf")] * 3)
        found_objects = False

        state = getattr(viewer, "_last_state", None)
        if state is not None:
            offsets = viewer.world_offsets.numpy() if viewer.world_offsets is not None else None
            for kind in ("body", "particle"):
                q = getattr(state, f"{kind}_q", None)
                if q is None or len(q) == 0:
                    continue
                positions = q.numpy()[:, :3].copy()
                worlds = getattr(viewer.model, f"{kind}_world").numpy()
                if offsets is not None:
                    valid = worlds >= 0
                    positions[valid] += offsets[worlds[valid]]
                visible = getattr(viewer, "_visible_worlds", None)
                if visible is not None:
                    positions = positions[(worlds < 0) | np.isin(worlds, list(visible))]
                if len(positions):
                    min_bounds = np.minimum(min_bounds, positions.min(axis=0))
                    max_bounds = np.maximum(max_bounds, positions.max(axis=0))
                    found_objects = True

        if workspace is not None and workspace.tree is not None and len(workspace.tree.deformables.sources) > 1:
            bounds = workspace.tree.deformables.external_bounds()
            if np.isfinite(bounds).all() and (bounds[1] >= bounds[0]).all():
                min_bounds = np.minimum(min_bounds, bounds[0])
                max_bounds = np.maximum(max_bounds, bounds[1])
                found_objects = True

        if not found_objects:
            min_bounds = np.array([-5.0, -5.0, -5.0])
            max_bounds = np.array([5.0, 5.0, 5.0])

        center = (min_bounds + max_bounds) * 0.5
        size = max_bounds - min_bounds
        max_extent = float(np.max(size))
        if max_extent < 1.0:
            max_extent = 1.0

        fov_rad = np.radians(camera.fov)
        padding = 1.5
        half_fov = np.arctan(np.tan(fov_rad / 2.0) * min(1.0, camera.width / max(1, camera.height)))
        distance = max_extent / (2.0 * np.tan(half_fov)) * padding
        front = camera.get_front()
        camera.pos = PyVec3(
            center[0] - front.x * distance,
            center[1] - front.y * distance,
            center[2] - front.z * distance,
        )
        camera.set_pivot(center)
        if hasattr(viewer, "_camera_dirty"):
            viewer._camera_dirty = True

    def _frame_contact(self, points):
        """Frame contact witnesses tightly, keeping both beyond the near plane."""
        viewer, camera = self._viewer, self._viewer.camera
        points = np.asarray(points, dtype=float)
        center = points.mean(axis=0)
        front = np.asarray(camera.get_front())
        if len(points) == 2:
            delta = points[1] - points[0]
            gap = np.linalg.norm(delta)
            if gap > max(1.0e-7, camera.near * 0.01):
                normal = delta / gap
                alignment = float(np.dot(front, normal))
                if abs(alignment) > 0.6:
                    # Avoid looking straight through both witnesses. Retain the
                    # viewing side and azimuth while exposing the separation.
                    tangent = front - alignment * normal
                    if np.linalg.norm(tangent) < 1.0e-6:
                        tangent = np.asarray(camera.get_right())
                        tangent -= np.dot(tangent, normal) * normal
                    front = tangent / np.linalg.norm(tangent) * np.sqrt(1.0 - 0.35**2)
                    front += normal * np.copysign(0.35, alignment)
                    camera._set_orientation_from_direction(front)
        front, right, up = map(np.asarray, (camera.get_front(), camera.get_right(), camera.get_up()))
        offsets = points - center
        depth = offsets @ front
        tangent = np.tan(np.radians(camera.fov) * 0.5)
        aspect = camera.width / max(camera.height, 1)
        # Fill about a third of the viewport with a resolvable gap; touching
        # witnesses use the near-plane clearance instead of fixed world padding.
        distance = max(
            np.max(np.abs(offsets @ right) / (tangent * aspect * 0.35) - depth),
            np.max(np.abs(offsets @ up) / (tangent * 0.35) - depth),
            camera.near * 2.0 - np.min(depth),
            camera._min_pivot_distance * 1.01,
        )
        camera.pos = camera._as_vec3(center - front * distance)
        camera.set_pivot(center)
        self._cam_vel[:] = 0.0
        if hasattr(viewer, "_camera_dirty"):
            viewer._camera_dirty = True

    def map_window_to_target_coords(self, x: float, y: float, window, target_size: tuple[int, int] | None = None):
        if window is None:
            return float(x), float(y)
        win_w, win_h = window.get_size()
        if win_w <= 0 or win_h <= 0:
            return float(x), float(y)
        if target_size is None:
            tgt_w, tgt_h = window.get_framebuffer_size()
        else:
            tgt_w, tgt_h = target_size
        scale_x = tgt_w / win_w
        scale_y = tgt_h / win_h
        return float(x) * scale_x, float(y) * scale_y

    def start_picking_from_screen(self, x: float, y: float, to_framebuffer_coords) -> bool:
        viewer = self._viewer
        if not getattr(viewer, "picking_enabled", False):
            return False
        picking = getattr(viewer, "picking", None)
        camera = getattr(viewer, "camera", None)
        if picking is None or camera is None or viewer._last_state is None:
            return False
        fb_x, fb_y = to_framebuffer_coords(x, y)
        rect = getattr(getattr(viewer, "renderer", None), "_workspace_rect", None)
        if rect is not None:
            fb_x, fb_y = fb_x - rect[0], fb_y - rect[1]
        ray_start, ray_dir = camera.get_world_ray(fb_x, fb_y)
        picking.pick(viewer._last_state, ray_start, ray_dir)
        return True

    def update_picking_from_screen(self, x: float, y: float, to_framebuffer_coords) -> bool:
        viewer = self._viewer
        if not self.is_pick_active():
            return False
        picking = getattr(viewer, "picking", None)
        camera = getattr(viewer, "camera", None)
        if picking is None or camera is None:
            return False
        fb_x, fb_y = to_framebuffer_coords(x, y)
        rect = getattr(getattr(viewer, "renderer", None), "_workspace_rect", None)
        if rect is not None:
            fb_x, fb_y = fb_x - rect[0], fb_y - rect[1]
        ray_start, ray_dir = camera.get_world_ray(fb_x, fb_y)
        picking.update(ray_start, ray_dir)
        return True

    def release_picking(self):
        picking = getattr(self._viewer, "picking", None)
        if picking is not None:
            picking.release()

    def handle_mouse_scroll(self, scroll_y: float, is_ctrl_down: bool = False) -> None:
        """Handle scroll wheel: dolly camera; Ctrl+scroll adjusts FOV."""
        if self.should_ignore_mouse_input():
            return
        camera = getattr(self._viewer, "camera", None)
        if camera is None:
            return
        if is_ctrl_down:
            camera.fov = max(15.0, min(90.0, camera.fov - scroll_y * 2.0))
        else:
            camera.dolly(scroll_y * self._camera_dolly_scroll_sensitivity)
        if hasattr(self._viewer, "_camera_dirty"):
            self._viewer._camera_dirty = True

    def handle_mouse_press(self, x: float, y: float, button: int, to_framebuffer_coords, modifiers: int = 0) -> None:
        """Handle mouse button press: start picking on right-click."""
        if self.should_ignore_mouse_input():
            return
        import pyglet

        viewer = self._viewer
        if button == pyglet.window.mouse.LEFT and getattr(self, "workspace", None) is not None:
            self._selection_press = (
                x,
                y,
                to_framebuffer_coords,
                bool(modifiers & pyglet.window.key.MOD_ALT),
                bool(modifiers & (pyglet.window.key.MOD_CTRL | pyglet.window.key.MOD_SHIFT)),
            )
        if (
            button == pyglet.window.mouse.RIGHT
            and getattr(viewer, "picking_enabled", False)
            and getattr(viewer, "picking", None) is not None
        ):
            self.start_picking_from_screen(x, y, to_framebuffer_coords)

    def handle_mouse_release(self, x: float, y: float, button: int) -> None:
        """Handle mouse button release: end picking on right-click."""
        import pyglet

        if button == pyglet.window.mouse.LEFT:
            press = getattr(self, "_selection_press", None)
            self._selection_press = None
            if press is not None and not self.should_ignore_mouse_input():
                px, py, convert, shape, toggle = press
                if (x - px) ** 2 + (y - py) ** 2 <= 16:
                    self.workspace.select_from_screen(x, y, convert, shape=shape, toggle=toggle)
        if button == pyglet.window.mouse.RIGHT and getattr(self._viewer, "picking", None) is not None:
            self.release_picking()

    def _camera_pan_scale(self) -> float:
        """World-space meters per window pixel for screen-plane camera panning."""
        viewer = self._viewer
        camera = getattr(viewer, "camera", None)
        if camera is None:
            return 0.01
        height = max(float(getattr(camera, "height", 1.0)), 1.0)
        renderer = getattr(viewer, "renderer", None)
        if renderer is not None:
            window = getattr(renderer, "window", None)
            if window is not None and hasattr(window, "get_size"):
                _, h = window.get_size()
                viewport = getattr(renderer, "_workspace_rect", None)
                if viewport is not None:
                    framebuffer_height = max(float(renderer._screen_height), 1.0)
                    height = max(viewport[3] * h / framebuffer_height, 1.0)
                else:
                    height = max(float(h), 1.0)
        distance = max(camera.pivot_distance, camera.MIN_PIVOT_DISTANCE)
        return 2.0 * distance * np.tan(np.radians(camera.fov) * 0.5) / height

    def handle_mouse_drag(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
        buttons: int,
        to_framebuffer_coords,
        modifiers: int = 0,
    ) -> None:
        """Handle mouse drag: middle-click orbit/pan/dolly, left-click look, right-click pick."""
        import pyglet

        press = getattr(self, "_selection_press", None)
        if press is not None:
            if buttons & pyglet.window.mouse.LEFT and (x - press[0]) ** 2 + (y - press[1]) ** 2 <= 16:
                return
            self._selection_press = None
        allow_active_pick_drag = (
            bool(buttons & pyglet.window.mouse.RIGHT)
            and getattr(self._viewer, "picking_enabled", False)
            and self.is_pick_active()
        )
        if self.should_ignore_mouse_input(allow_active_pick_drag=allow_active_pick_drag):
            return
        viewer = self._viewer
        camera = getattr(viewer, "camera", None)

        if buttons & pyglet.window.mouse.MIDDLE and camera is not None:
            if modifiers & pyglet.window.key.MOD_CTRL:
                camera.dolly(dy * self._camera_dolly_drag_sensitivity)
            elif modifiers & pyglet.window.key.MOD_SHIFT:
                scale = self._camera_pan_scale()
                camera.pan(-dx * scale, -dy * scale)
            else:
                camera.orbit(
                    delta_yaw=-dx * self._camera_orbit_sensitivity,
                    delta_pitch=dy * self._camera_orbit_sensitivity,
                )
            if hasattr(viewer, "_camera_dirty"):
                viewer._camera_dirty = True
            return

        if buttons & pyglet.window.mouse.LEFT:
            if camera is not None and self.following_selection:
                camera.orbit(-dx * self._camera_orbit_sensitivity, dy * self._camera_orbit_sensitivity)
                if hasattr(viewer, "_camera_dirty"):
                    viewer._camera_dirty = True
            else:
                self.rotate_camera_from_drag(dx, dy)
                if camera is not None:
                    camera.sync_pivot_to_view()
        if (
            buttons & pyglet.window.mouse.RIGHT
            and getattr(viewer, "picking_enabled", False)
            and getattr(viewer, "picking", None) is not None
        ):
            self.update_picking_from_screen(x, y, to_framebuffer_coords)

    def close_console(self):
        """Release logging routes on window close, including close events outside run()."""
        if self.console_capture is not None:
            self.console_capture.close()

    def handle_key_press(self, symbol: int, close_fn=None, modifiers: int = 0) -> None:
        """Handle common key bindings shared by all viewer backends.

        Args:
            symbol: Pyglet key symbol.
            close_fn: Callable that closes the viewer window, or None.
            modifiers: Pyglet keyboard modifier flags.
        """
        import pyglet

        if symbol == pyglet.window.key.GRAVE and modifiers & pyglet.window.key.MOD_CTRL and self.workspace is not None:
            self.workspace.console.toggle()
            return
        # Mouse-opened popups need not set WantCaptureKeyboard when keyboard
        # navigation is disabled. Let ImGui dismiss them before handling viewer
        # shortcuts, especially Escape (which otherwise closes the window).
        if self.is_available and self.ui.imgui.is_popup_open("", self.ui.imgui.PopupFlags_.any_popup):
            return
        if self.should_ignore_keyboard_input():
            return
        # Numeric entry keeps its decimal point and Escape cancellation. Space
        # and camera keys still work without first clicking the viewport.
        if self._numeric_text_input_active() and symbol in (pyglet.window.key.PERIOD, pyglet.window.key.ESCAPE):
            return
        if symbol == pyglet.window.key.SPACE:
            self._viewer._paused = not self._viewer._paused
        elif symbol == pyglet.window.key.PERIOD and getattr(self._viewer, "_paused", False):
            self._viewer._step_requested = True
        elif symbol == pyglet.window.key.H:
            self.show_ui = not self.show_ui
        elif symbol == pyglet.window.key.F:
            if not getattr(self, "_frame_key_down", False):
                self._frame_key_down = True
                self.frame_camera_on_model()
                # Follow the spatial owner even when F framed a contact on it.
                # Contact row indices themselves are transient across steps.
                self._follow_held = bool(
                    self.workspace is not None
                    and self.workspace.can_follow_selection
                    and self._start_follow_selection()
                )
        elif symbol == pyglet.window.key.ESCAPE and close_fn is not None:
            self.close_console()
            close_fn()

    def handle_key_release(self, symbol: int):
        from pyglet.window import key

        if symbol == key.F:
            self._frame_key_down = self._follow_held = False

    def _update_fps(self):
        """Update FPS counter; called once per rendered frame."""
        current_time = perf_counter()
        self._fps_frame_count += 1
        if current_time - self._last_fps_time >= 1.0:
            self._current_fps = self._fps_frame_count / (current_time - self._last_fps_time)
            self._fps_history.append(self._current_fps)
            if len(self._fps_history) > 60:
                self._fps_history.pop(0)
            self._last_fps_time = current_time
            self._fps_frame_count = 0

    def render_frame(self, update_fps: bool = True):
        """Render GUI into the active OpenGL framebuffer."""
        self.prepare_frame(update_fps=update_fps)
        self.render_prepared_frame()

    def prepare_frame(self, update_fps: bool = True):
        """Process UI actions before the viewer chooses its next scene image."""
        self._frame_prepared = False
        if update_fps:
            self._update_fps()
        if not self.is_available:
            return
        if self.workspace is not None and self.workspace.console.poll_warnings():
            self.show_ui = True
        if not self.show_ui and self.workspace is not None:
            self.workspace.collision.contacts_tab = False
            self.workspace.collision.overlays()
        if not self.show_ui and not self._loading_splash_active:
            return
        self.ui.begin_frame()
        try:
            if self.show_ui:
                loading = self._loading_splash_active
                if loading:
                    # Block interaction without making panel backgrounds transparent;
                    # the foreground overlay dims the whole composed GUI uniformly.
                    self.ui.imgui.push_style_var(self.ui.imgui.StyleVar_.disabled_alpha, 1.0)
                    self.ui.imgui.begin_disabled()
                try:
                    self._render_ui()
                finally:
                    if loading:
                        self.ui.imgui.end_disabled()
                        self.ui.imgui.pop_style_var()
            if self._loading_splash_active:
                self._render_loading_splash()
            self.ui.end_frame()
        except (Exception, SystemExit):
            self.ui.abort_frame()
            raise
        self._frame_prepared = True

    def render_prepared_frame(self):
        """Draw the prepared UI over the scene without processing actions twice."""
        if self._frame_prepared:
            self.ui.render()
            self._frame_prepared = False

    def register_ui_callback(
        self,
        callback: Callable[[Any], None],
        position: Literal["side", "stats", "free", "panel", "rendering"] = "side",
    ):
        """Register a UI callback to be rendered during the UI phase.

        Args:
            callback: Function called during UI rendering, receiving the active
                ImGui context as its only argument.
            position: One of ``"side"``, ``"stats"``, ``"free"``, ``"panel"``,
                ``"rendering"``.
        """
        if not callable(callback):
            raise TypeError("callback must be callable")
        if position not in self._ui_callbacks:
            valid_positions = list(self._ui_callbacks.keys())
            raise ValueError(f"Invalid position '{position}'. Must be one of: {valid_positions}")
        self._ui_callbacks[position].append(callback)

    def clear_example_callbacks(self) -> None:
        """Drop example-registered ``side`` / ``free`` callbacks across a model switch.

        Backend rendering options (``rendering``) and persistent ``panel`` /
        ``stats`` callbacks survive.
        """
        self._ui_callbacks["side"] = []
        self._ui_callbacks["free"] = []
        self._workspace_example = None
        self._frame_camera_pending = False
        if self.workspace is not None:
            self.workspace.detach_example()

    def show_loading_splash(self, text: str | None = None) -> None:
        """Display a centered Newton's-cradle loading splash.

        Args:
            text: Optional sub-label drawn below the cradle.
        """
        self._loading_splash_active = True
        self._loading_splash_text = text

    def hide_loading_splash(self) -> None:
        """Remove the splash set by :meth:`show_loading_splash`."""
        self._loading_splash_active = False
        self._loading_splash_text = None

    def _render_gizmos(self):
        viewer = self._viewer
        self._gizmo_hovered = False
        if not self.is_available:
            return
        if viewer.is_rendering_paused() or self._loading_splash_active:
            return
        if self.workspace_enabled and self.workspace is not None and not self.workspace.documents.viewport_visible:
            return
        if not hasattr(viewer, "_gizmo_log") or not viewer._gizmo_log:
            self._gizmo_active.clear()
            if hasattr(viewer, "gizmo_is_using"):
                viewer.gizmo_is_using = False
            return
        if not hasattr(viewer, "camera") or viewer.camera is None:
            return

        giz = self.ui.giz
        io = self.ui.io

        # Setup ImGuizmo viewport
        giz.set_orthographic(False)
        rect = (
            self.workspace.rect
            if self.workspace_enabled and self.workspace
            else (0, 0, io.display_size[0], io.display_size[1])
        )
        giz.set_rect(*map(float, rect))
        draw_list = None
        if self.workspace_enabled and self.workspace is not None:
            # ImGuizmo's default full-window layer lies behind docked documents.
            # Use the viewport for both draw order and hover ownership.
            imgui = self.ui.imgui
            window = imgui.internal.find_window_by_name(self.workspace.documents.viewport_name)
            giz.set_alternative_window(window)
            draw_list = window.draw_list
            giz.set_drawlist(draw_list)
            x, y, width, height = rect
            draw_list.push_clip_rect(imgui.ImVec2(x, y), imgui.ImVec2(x + width, y + height), True)
        giz.set_gizmo_size_clip_space(0.07)
        giz.set_axis_limit(0.0)
        giz.set_plane_limit(0.0)
        try:
            giz.allow_axis_flip(False)
        except AttributeError:
            pass

        # Camera matrices
        view = viewer.camera.get_view_matrix().reshape(4, 4).transpose()
        proj = viewer.camera.get_projection_matrix().reshape(4, 4).transpose()

        axis_translate = {
            Axis.X: giz.OPERATION.translate_x,
            Axis.Y: giz.OPERATION.translate_y,
            Axis.Z: giz.OPERATION.translate_z,
        }
        axis_rotate = {
            Axis.X: giz.OPERATION.rotate_x,
            Axis.Y: giz.OPERATION.rotate_y,
            Axis.Z: giz.OPERATION.rotate_z,
        }

        def m44_to_mat16(m):
            """Row-major 4x4 -> giz.Matrix16 (column-major, 16 floats)."""
            m = np.asarray(m, dtype=np.float32).reshape(4, 4)
            return giz.Matrix16(m.flatten(order="F").tolist())

        def safe_bool(value) -> bool:
            try:
                return bool(value)
            except Exception:
                return False

        view_ = m44_to_mat16(view)
        proj_ = m44_to_mat16(proj)

        logged_ids = set()
        for gid, entry in viewer._gizmo_log.items():
            logged_ids.add(gid)

            # Support both the rich dict format {transform, snap_to, translate, rotate}
            # and the legacy raw-transform format used by ViewerRTX.
            if isinstance(entry, dict):
                transform = entry["transform"]
                snap_to = entry.get("snap_to")
                translate = entry.get("translate", (Axis.X, Axis.Y, Axis.Z))
                rotate = entry.get("rotate", (Axis.X, Axis.Y, Axis.Z))
            else:
                transform = entry
                snap_to = None
                translate = (Axis.X, Axis.Y, Axis.Z)
                rotate = (Axis.X, Axis.Y, Axis.Z)

            # Build combined operation list
            if len(translate) == 3:
                t_ops = (giz.OPERATION.translate,)
            else:
                t_ops = tuple(axis_translate[a] for a in translate)

            if len(rotate) == 3:
                r_ops = (giz.OPERATION.rotate,)
            else:
                r_ops = tuple(axis_rotate[a] for a in rotate)

            ops = t_ops + r_ops
            was_active = self._gizmo_active.get(gid, False)
            if not ops:
                if was_active and snap_to is not None:
                    transform_assign(transform, snap_to)
                self._gizmo_active[gid] = False
                continue

            giz.push_id(str(gid))

            M = transform_to_matrix(transform)
            M_ = m44_to_mat16(M)

            # The binding accepts named operations, not arbitrary bitmasks.
            # Give each operation its own ID and retain activity across all of
            # them so the last (idle) handle cannot cancel a translation drag.
            modified = is_active = False
            for op in ops:
                giz.push_id(int(op))
                modified |= safe_bool(giz.manipulate(view_, proj_, op, giz.MODE.world, M_, None, None))
                self._gizmo_hovered |= safe_bool(giz.is_over())
                is_active |= safe_bool(giz.is_using())
                giz.pop_id()

            if was_active and not is_active and snap_to is not None:
                transform_assign(transform, snap_to)
            elif modified:
                M[:] = M_.values.reshape(4, 4, order="F")
                transform_assign_matrix(transform, M)

            self._gizmo_active[gid] = is_active

            giz.pop_id()

        # Drop stale interaction state for gizmos that are no longer logged.
        for gid in tuple(self._gizmo_active):
            if gid not in logged_ids:
                del self._gizmo_active[gid]

        if hasattr(viewer, "gizmo_is_using"):
            viewer.gizmo_is_using = giz.is_using_any()
        if self._gizmo_hovered or getattr(viewer, "gizmo_is_using", False):
            self._selection_press = None
        if draw_list is not None:
            draw_list.pop_clip_rect()

    def _render_loading_splash(self):
        """Render a stylized Newton's-cradle loading splash, optionally with a sub-label.

        The cradle is drawn statically with the leftmost ball lifted; this is
        a one-frame snapshot, not an animation.  Sizes scale with the current
        ImGui font size so the splash stays legible across DPI settings.
        """
        if not self.is_available:
            return
        text = self._loading_splash_text
        imgui = self.ui.imgui
        viewport = imgui.get_main_viewport()

        # Scale relative to the default 13 px ImGui font so the splash
        # respects user/DPI font scaling.
        scale = imgui.get_font_size() / 13.0
        ball_radius = 16.0 * scale
        # 2.05 (vs 2.0) leaves a hairline gap between balls so adjacent
        # rest-position balls remain visually distinguishable.
        ball_spacing = ball_radius * 2.05
        string_length = 80.0 * scale
        bar_thickness = 5.0 * scale
        text_gap = 18.0 * scale
        bar_overhang = 8.0 * scale
        string_thickness = 1.5 * scale
        n_balls = 5

        # Center the cradle's full bounding box (bar -> deepest ball) at the
        # viewport center.  ``pivot_y`` is the bar's *bottom* edge (where
        # strings attach), not the bar centerline — hence the
        # ``+ bar_thickness`` after positioning the bbox top.
        cradle_height = bar_thickness + string_length + ball_radius
        cx = viewport.pos.x + viewport.size.x * 0.5
        cy = viewport.pos.y + viewport.size.y * 0.5
        pivot_y = cy - cradle_height * 0.5 + bar_thickness

        # Foreground covers docked/floating documents, popups and all panels.
        draw_list = imgui.get_foreground_draw_list(viewport)

        dim_col = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.0, 0.0, 0.0, 0.55))
        ball_col = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.88, 0.88, 0.92, 1.0))
        string_col = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.55, 0.55, 0.6, 1.0))
        bar_col = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.45, 0.45, 0.5, 1.0))
        text_col = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.9, 0.9, 0.9, 1.0))

        # Dim the underlying scene.  Drawn manually rather than via
        # ``set_next_window_bg_alpha`` so the dim color is independent
        # of the active ImGui style.
        draw_list.add_rect_filled(
            imgui.ImVec2(viewport.pos.x, viewport.pos.y),
            imgui.ImVec2(viewport.pos.x + viewport.size.x, viewport.pos.y + viewport.size.y),
            dim_col,
        )

        first_pivot_x = cx - (n_balls - 1) * ball_spacing * 0.5
        bar_half = (n_balls - 1) * ball_spacing * 0.5 + ball_radius + bar_overhang
        draw_list.add_rect_filled(
            imgui.ImVec2(cx - bar_half, pivot_y - bar_thickness),
            imgui.ImVec2(cx + bar_half, pivot_y),
            bar_col,
        )

        swing_angle = math.radians(32.0)
        for i in range(n_balls):
            pivot_x = first_pivot_x + i * ball_spacing
            if i == 0:
                ball_x = pivot_x - math.sin(swing_angle) * string_length
                ball_y = pivot_y + math.cos(swing_angle) * string_length
            else:
                ball_x = pivot_x
                ball_y = pivot_y + string_length

            draw_list.add_line(
                imgui.ImVec2(pivot_x, pivot_y),
                imgui.ImVec2(ball_x, ball_y),
                string_col,
                string_thickness,
            )
            draw_list.add_circle_filled(
                imgui.ImVec2(ball_x, ball_y),
                ball_radius,
                ball_col,
            )

        if text:
            text_size = imgui.calc_text_size(text)
            text_x = cx - text_size.x * 0.5
            text_y = pivot_y + string_length + ball_radius + text_gap
            draw_list.add_text(imgui.ImVec2(text_x, text_y), text_col, text)

    def _render_ui(self):
        """Render the complete ImGui interface."""
        if not self.is_available:
            return

        if self.workspace_enabled:
            if self.workspace is None:
                from .viewer_workspace import ViewerWorkspace  # noqa: PLC0415

                self.workspace = ViewerWorkspace(self)
            self.workspace.bind(self._workspace_example)
            self.workspace.draw()
            self._render_gizmos()
            for callback in self._ui_callbacks["free"]:
                if getattr(callback, "__name__", "") != "timeline_gui":
                    name = getattr(callback, "__qualname__", type(callback).__name__)
                    self.ui.draw_section(f"Example overlay {name}", lambda cb=callback: cb(self.ui.imgui))
            return

        self._render_left_panel()
        self._render_gizmos()
        self._render_stats_overlay()
        self._render_scalar_plots()
        self._render_logged_images()

        for callback in self._ui_callbacks["free"]:
            name = getattr(callback, "__qualname__", type(callback).__name__)
            self.ui.draw_section(f"Example overlay {name}", lambda cb=callback: cb(self.ui.imgui))

    def _render_left_panel(self):
        """Render left panel with model details and visualization controls."""
        if not self.is_available:
            return

        viewer = self._viewer
        imgui = self.ui.imgui

        nav_highlight_color = self.ui.get_theme_color(imgui.Col_.nav_cursor, (1.0, 1.0, 1.0, 1.0))

        io = self.ui.io
        # ``dpi_scale`` keeps the panel at a constant visual size on HiDPI
        # displays, where ``display_size`` is in framebuffer pixels.
        s = self.ui.dpi_scale
        # Initial position/size only — ``first_use_ever`` lets the user drag
        # the title bar and resize via the bottom-right corner without
        # snapping back on every appearance.
        imgui.set_next_window_pos(imgui.ImVec2(10 * s, 10 * s), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(
            imgui.ImVec2(_SIDEBAR_WIDTH_PX * s, io.display_size[1] - 20 * s),
            imgui.Cond_.first_use_ever,
        )
        # Allow generous downsizing while keeping at least one button row plus
        # the title bar visible.
        imgui.set_next_window_size_constraints(
            imgui.ImVec2(160 * s, 80 * s),
            imgui.ImVec2(io.display_size[0], io.display_size[1]),
        )

        flags = 0

        if imgui.begin(f"Newton Viewer v{nt.__version__}", flags=flags):
            imgui.separator()
            header_flags = 0

            changed, paused = imgui.checkbox("Pause Rendering", viewer.is_rendering_paused())
            if changed:
                viewer.set_rendering_paused(paused)
            if paused:
                imgui.text_disabled("Image frozen")
            imgui.separator()

            # Run controls — shown once a model is loaded
            if viewer.model is not None:
                _changed, viewer._paused = imgui.checkbox("Pause", viewer._paused)
                imgui.same_line()
                imgui.begin_disabled(not viewer._paused)
                if imgui.button("Step"):
                    viewer._step_requested = True
                imgui.end_disabled()
                reset_cb = getattr(viewer, "_reset_callback", None)
                if reset_cb is not None:
                    imgui.same_line()
                    if imgui.button("Reset"):
                        reset_cb()
                imgui.separator()

            # Top-level collapsing headers injected by the viewer (e.g. example browser)
            for callback in self._ui_callbacks.get("panel", []):
                callback(self.ui.imgui)

            if viewer.model is not None:
                imgui.set_next_item_open(True, imgui.Cond_.appearing)
                if imgui.collapsing_header("Model Information", flags=header_flags):
                    imgui.separator()
                    axis_names = ["X", "Y", "Z"]
                    imgui.text(f"Up Axis: {axis_names[viewer.model.up_axis]}")
                    gravity = viewer.model.gravity.numpy()
                    world_gravity = gravity[0]
                    global_gravity = gravity[-1]
                    imgui.text(
                        f"World 0 Gravity: ({world_gravity[0]:.2f}, {world_gravity[1]:.2f}, {world_gravity[2]:.2f})"
                    )
                    imgui.text(
                        f"Global Gravity: ({global_gravity[0]:.2f}, {global_gravity[1]:.2f}, {global_gravity[2]:.2f})"
                    )

                imgui.set_next_item_open(True, imgui.Cond_.appearing)
                if imgui.collapsing_header("Visualization", flags=header_flags):
                    imgui.separator()
                    renderer = getattr(viewer, "renderer", None)
                    _changed, viewer.show_joints = imgui.checkbox("Show Joints", viewer.show_joints)
                    if viewer.show_joints and renderer is not None and hasattr(renderer, "joint_scale"):
                        _, renderer.joint_scale = imgui.slider_float("Joint Scale", renderer.joint_scale, 0.25, 5.0)
                    _changed, viewer.show_contacts = imgui.checkbox("Show Contacts", viewer.show_contacts)
                    if viewer.show_contacts and renderer is not None:
                        imgui.indent()
                        _, viewer.show_contact_normals = imgui.checkbox("Normal", bool(viewer.show_contact_normals))
                        _, viewer.show_contact_disks = imgui.checkbox("Contact Mode", bool(viewer.show_contact_disks))
                        _, viewer.show_contact_forces = imgui.checkbox("Force", bool(viewer.show_contact_forces))
                        imgui.unindent()

                        log_flag = imgui.SliderFlags_.logarithmic.value
                        base = float(viewer._contact_viz_scale_default) or 1.0
                        _, viewer.contact_viz_scale = imgui.slider_float(
                            "Contact Relative Scale",
                            float(viewer.contact_viz_scale),
                            base * 0.01,
                            base * 100.0,
                            "%.4g",
                            log_flag,
                        )
                        if viewer.show_contact_forces:
                            base = float(viewer._contact_force_scale_default) or 0.5
                            _, viewer.contact_force_scale = imgui.slider_float(
                                "Force Relative Scale",
                                float(viewer.contact_force_scale),
                                base * 0.01,
                                base * 100.0,
                                "%.4g",
                                log_flag,
                            )
                        if hasattr(renderer, "arrow_scale") and (
                            viewer.show_contact_normals or viewer.show_contact_forces
                        ):
                            _, renderer.arrow_scale = imgui.slider_float(
                                "Arrow Thickness", renderer.arrow_scale, 0.25, 5.0
                            )
                    _changed, viewer.show_particles = imgui.checkbox("Show Particles", viewer.show_particles)
                    if _changed:
                        viewer._particle_visibility_user = viewer.show_particles
                    _changed, viewer.show_springs = imgui.checkbox("Show Springs", viewer.show_springs)
                    _changed, viewer.show_com = imgui.checkbox("Show Center of Mass", viewer.show_com)
                    if viewer.show_com and renderer is not None and hasattr(renderer, "com_scale"):
                        _, renderer.com_scale = imgui.slider_float("COM Scale", renderer.com_scale, 0.25, 5.0)
                    _changed, viewer.show_triangles = imgui.checkbox("Show Cloth", viewer.show_triangles)
                    _changed, viewer.show_collision = imgui.checkbox("Show Collision", viewer.show_collision)
                    if renderer is not None and hasattr(renderer, "draw_edges"):
                        _changed, renderer.draw_edges = imgui.checkbox("Show Edges", renderer.draw_edges)
                    sdf_margin_mode = getattr(viewer, "sdf_margin_mode", None)
                    SDFMarginMode = getattr(type(viewer), "SDFMarginMode", None)
                    if sdf_margin_mode is not None and SDFMarginMode is not None:
                        _sdf_margin_labels = ["Off", "Margin", "Margin + Gap"]
                        _, new_sdf_idx = imgui.combo("Gap + Margin", int(sdf_margin_mode), _sdf_margin_labels)
                        viewer.sdf_margin_mode = SDFMarginMode(new_sdf_idx)
                        if viewer.sdf_margin_mode != SDFMarginMode.OFF and renderer is not None:
                            _, renderer.wireframe_line_width = imgui.slider_float(
                                "Wireframe Width (px)", renderer.wireframe_line_width, 0.5, 5.0
                            )
                    _changed, viewer.show_visual = imgui.checkbox("Show Visual", viewer.show_visual)
                    _changed, viewer.show_ground = imgui.checkbox("Show Ground", viewer.show_ground)
                    _changed, viewer.show_inertia_boxes = imgui.checkbox(
                        "Show Inertia Boxes", viewer.show_inertia_boxes
                    )

            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Example Options"):
                for callback in self._ui_callbacks["side"]:
                    callback(self.ui.imgui)

            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Rendering Options"):
                imgui.separator()
                _changed, viewer.vsync = imgui.checkbox("VSync", viewer.vsync)
                # Viewer-specific rendering options (e.g. GL sky/shadows/wireframe)
                for callback in self._ui_callbacks.get("rendering", []):
                    callback(self.ui.imgui)
                image_logger = getattr(viewer, "_image_logger", None)
                if image_logger is not None:
                    image_logger.draw_controls(imgui)

            wind = getattr(viewer, "wind", None)
            if wind is not None:
                imgui.set_next_item_open(False, imgui.Cond_.once)
                if imgui.collapsing_header("Wind"):
                    imgui.separator()
                    changed, amplitude = imgui.slider_float("Wind Amplitude", wind.amplitude, -2.0, 2.0, "%.2f")
                    if changed:
                        wind.amplitude = amplitude
                    changed, period = imgui.slider_float("Wind Period", wind.period, 1.0, 30.0, "%.2f")
                    if changed:
                        wind.period = period
                    changed, frequency = imgui.slider_float("Wind Frequency", wind.frequency, 0.1, 5.0, "%.2f")
                    if changed:
                        wind.frequency = frequency
                    direction = [wind.direction[0], wind.direction[1], wind.direction[2]]
                    changed, direction = imgui.slider_float3("Wind Direction", direction, -1.0, 1.0, "%.2f")
                    if changed:
                        wind.direction = direction

            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Controls"):
                imgui.separator()
                self._render_camera_info()

                imgui.separator()
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*nav_highlight_color))
                imgui.text("Controls:")
                imgui.pop_style_color()
                imgui.text("WASD - Move camera")
                imgui.text("QE - Pan up/down")
                imgui.text("Left Click - Look around")
                imgui.text("Right Click - Pick objects")
                imgui.text("Middle Click - Orbit")
                imgui.text("Shift + Middle Click - Pan")
                imgui.text("Ctrl + Middle Click - Dolly")
                imgui.text("Scroll - Dolly")
                imgui.text("Ctrl + Scroll - FOV zoom")
                imgui.text("Space - Pause/Resume")
                imgui.text(". - Step one frame (when paused)")
                imgui.text("H - Toggle UI")
                imgui.text("F - Frame camera around model")

            self._render_selection_panel()

        imgui.end()

    def _render_camera_info(self):
        imgui = self.ui.imgui
        cam = getattr(self._viewer, "camera", None)
        if cam is None:
            imgui.text("Camera information not available.")
            return

        pos = cam.pos
        imgui.text(f"Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        imgui.text(f"FOV: {cam.fov:.1f} deg")
        imgui.text(f"Pitch: {cam.pitch:.1f} deg")
        imgui.text(f"Yaw: {cam.yaw:.1f} deg")

    def update_shape_counts(self, model) -> None:
        """Recompute the visual / collision shape counts shown in the stats overlay.

        Called when the viewer is given a model. ``shape_flags`` is a device array, so
        this is done once per model rather than while rendering the overlay.
        """
        flags_array = getattr(model, "shape_flags", None) if model is not None else None
        if flags_array is None or len(flags_array) == 0:
            self._shape_visual_count = None
            self._shape_collision_count = None
            return
        flags = flags_array.numpy()
        self._shape_visual_count = int(np.count_nonzero(flags & int(nt.ShapeFlags.VISIBLE)))
        self._shape_collision_count = int(np.count_nonzero(flags & int(nt.ShapeFlags.COLLIDE_SHAPES)))

    def _render_stats_overlay(self):
        """Render performance overlay in the top-right corner."""
        if not self.is_available:
            return

        viewer = self._viewer
        imgui = self.ui.imgui
        io = self.ui.io
        s = self.ui.dpi_scale
        fps_color = (1.0, 1.0, 1.0, 1.0)

        window_pos = (io.display_size[0] - 10 * s, 10 * s)
        imgui.set_next_window_pos(imgui.ImVec2(window_pos[0], window_pos[1]), pivot=imgui.ImVec2(1.0, 0.0))

        flags: imgui.WindowFlags = (
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.always_auto_resize.value
            | imgui.WindowFlags_.no_resize.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.no_focus_on_appearing.value
            | imgui.WindowFlags_.no_nav.value
            | imgui.WindowFlags_.no_move.value
        )

        pushed_window_bg = False
        try:
            imgui.set_next_window_bg_alpha(0.7)
        except AttributeError:
            try:
                style = imgui.get_style()
                bg = style.color_(imgui.Col_.window_bg)
                r, g, b = bg.x, bg.y, bg.z
            except Exception:
                r, g, b = 0.094, 0.094, 0.094
            imgui.push_style_color(imgui.Col_.window_bg, imgui.ImVec4(r, g, b, 0.7))
            pushed_window_bg = True

        if imgui.begin("Performance Stats", flags=flags):
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*fps_color))
            imgui.text(f"FPS: {self._current_fps:.1f}")
            imgui.pop_style_color()

            if viewer.model is not None:
                imgui.separator()
                imgui.text(f"Worlds: {viewer.model.world_count}")
                imgui.text(f"Bodies: {viewer.model.body_count}")
                imgui.text(f"Shapes: {viewer.model.shape_count}")
                if self._shape_visual_count is not None:
                    # Categories overlap when a shape carries both flags.
                    imgui.indent()
                    imgui.text(f"visual: {self._shape_visual_count}")
                    imgui.text(f"collision: {self._shape_collision_count}")
                    imgui.unindent()
                imgui.text(f"Joints: {viewer.model.joint_count}")
                imgui.text(f"Particles: {viewer.model.particle_count}")
                imgui.text(f"Springs: {viewer.model.spring_count}")
                imgui.text(f"Triangles: {viewer.model.tri_count}")
                imgui.text(f"Edges: {viewer.model.edge_count}")
                imgui.text(f"Tetrahedra: {viewer.model.tet_count}")

            objects = getattr(viewer, "objects", None)
            if objects is not None:
                imgui.separator()
                imgui.text(f"Unique Objects: {len(objects)}")

        for callback in self._ui_callbacks["stats"]:
            callback(self.ui.imgui)

        imgui.end()
        if pushed_window_bg:
            imgui.pop_style_color()

    def _render_scalar_plots(self):
        """Render live scalar plots and array heatmaps."""
        plot_logger = getattr(self._viewer, "_plot_logger", None)
        if plot_logger is not None:
            plot_logger.draw(self.ui)

    def _render_logged_images(self):
        """Render the selected :meth:`~newton.viewer.ViewerBase.log_image` window."""
        image_logger = getattr(self._viewer, "_image_logger", None)
        if image_logger is not None:
            image_logger.draw(self.ui, sidebar_width_px=_SIDEBAR_WIDTH_PX)

    def _render_selection_panel(self):
        """Render the articulation selection panel."""
        if not self.is_available:
            return

        viewer = self._viewer
        imgui = self.ui.imgui
        header_flags = 0
        imgui.set_next_item_open(False, imgui.Cond_.appearing)
        if not imgui.collapsing_header("Selection API", flags=header_flags):
            return

        imgui.separator()
        if viewer._last_state is None:
            imgui.text("No state data available.")
            imgui.text("Start simulation to enable selection.")
            return

        state = self._selection_ui_state

        if state["error_message"]:
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 0.3, 0.3, 1.0))
            imgui.text(f"Error: {state['error_message']}")
            imgui.pop_style_color()
            imgui.separator()

        imgui.text("Articulation Pattern:")
        imgui.push_item_width(200)
        _changed, state["selected_articulation_pattern"] = imgui.input_text(
            "##pattern", state["selected_articulation_pattern"]
        )
        imgui.pop_item_width()

        imgui.spacing()
        imgui.text("Joint Filters (optional):")
        imgui.push_item_width(150)
        imgui.text("Include:")
        imgui.same_line()
        _changed, state["include_joints"] = imgui.input_text("##inc_joints", state["include_joints"])
        imgui.text("Exclude:")
        imgui.same_line()
        _changed, state["exclude_joints"] = imgui.input_text("##exc_joints", state["exclude_joints"])
        imgui.pop_item_width()

        imgui.spacing()
        imgui.text("Link Filters (optional):")
        imgui.push_item_width(150)
        imgui.text("Include:")
        imgui.same_line()
        _changed, state["include_links"] = imgui.input_text("##inc_links", state["include_links"])
        imgui.text("Exclude:")
        imgui.same_line()
        _changed, state["exclude_links"] = imgui.input_text("##exc_links", state["exclude_links"])
        imgui.pop_item_width()

        imgui.spacing()
        if imgui.button("Create Articulation View"):
            self._create_articulation_view()

        if state["selected_articulation_view"] is None:
            return

        view = state["selected_articulation_view"]

        def text(value):
            return "varies" if value is None else str(value)

        imgui.separator()
        imgui.text(f"  Count: {view.count}")
        imgui.text(f"  Joints: {text(view.joint_count)}")
        imgui.text(f"  Links: {text(view.link_count)}")
        imgui.text(f"  DOFs: {text(view.joint_dof_count)}")
        imgui.text(f"  Fixed base: {text(view.is_fixed_base)}")
        imgui.text(f"  Floating base: {text(view.is_floating_base)}")

        imgui.spacing()
        imgui.text("Select Attribute:")
        imgui.push_item_width(150)
        if state["selected_attribute"] in state["attribute_options"]:
            current_attr_idx = state["attribute_options"].index(state["selected_attribute"])
        else:
            current_attr_idx = 0
        _changed, new_attr_idx = imgui.combo("##attribute", current_attr_idx, state["attribute_options"])
        state["selected_attribute"] = state["attribute_options"][new_attr_idx]
        imgui.pop_item_width()

        _changed, state["show_values"] = imgui.checkbox("Show Values", state["show_values"])
        if state["show_values"]:
            self._render_attribute_values(view, state["selected_attribute"])

    def _create_articulation_view(self):
        state = self._selection_ui_state
        viewer = self._viewer
        try:
            state["error_message"] = ""

            include_joints = [joint.strip() for joint in state["include_joints"].split(",") if joint.strip()] or None
            exclude_joints = [joint.strip() for joint in state["exclude_joints"].split(",") if joint.strip()] or None
            include_links = [link.strip() for link in state["include_links"].split(",") if link.strip()] or None
            exclude_links = [link.strip() for link in state["exclude_links"].split(",") if link.strip()] or None

            state["selected_articulation_view"] = ArticulationView(
                model=viewer.model,
                pattern=state["selected_articulation_pattern"],
                include_joints=include_joints,
                exclude_joints=exclude_joints,
                include_links=include_links,
                exclude_links=exclude_links,
                allow_partial_layouts=True,
                verbose=False,
            )
        except Exception as e:
            state["error_message"] = str(e)
            state["selected_articulation_view"] = None

    def _render_attribute_values(self, view: ArticulationView, attribute_name: str):
        imgui = self.ui.imgui
        viewer = self._viewer
        state = self._selection_ui_state

        try:
            if attribute_name.startswith("joint_f"):
                if viewer._last_control is not None:
                    source = viewer._last_control
                else:
                    imgui.text("No control data available for forces")
                    return
            else:
                source = viewer._last_state

            values = view.get_attribute(attribute_name, source).numpy()

            imgui.separator()
            imgui.text(f"Attribute: {attribute_name}")
            imgui.text(f"Shape: {values.shape}")
            imgui.text(f"Dtype: {values.dtype}")

            if len(values.shape) == 2:
                batch_size = values.shape[0]
                imgui.spacing()
                imgui.text("Batch/World Selection:")
                imgui.push_item_width(100)
                state["selected_batch_idx"] = max(0, min(state["selected_batch_idx"], batch_size - 1))
                _changed, state["selected_batch_idx"] = imgui.slider_int(
                    "##batch", state["selected_batch_idx"], 0, batch_size - 1
                )
                imgui.pop_item_width()
                imgui.same_line()
                imgui.text(f"World {state['selected_batch_idx']} / {batch_size}")

            imgui.spacing()
            imgui.text("Values:")
            if imgui.begin_child("values_scroll", 0, 300 * self.ui.dpi_scale, border=True):
                if len(values.shape) == 1:
                    names = self._get_attribute_names(view, attribute_name)
                    self._render_value_sliders(values, names, attribute_name, state)
                elif len(values.shape) == 2:
                    batch_idx = state["selected_batch_idx"]
                    selected_batch = values[batch_idx]
                    names = self._get_attribute_names(view, attribute_name)
                    self._render_value_sliders(selected_batch, names, attribute_name, state)
                else:
                    imgui.text(f"Multi-dimensional array with shape {values.shape}")
            imgui.end_child()

            if values.dtype.kind in "biufc":
                imgui.spacing()
                if len(values.shape) == 2:
                    batch_idx = state["selected_batch_idx"]
                    stats_data = values[batch_idx]
                    imgui.text(f"Statistics for World {batch_idx}:")
                else:
                    stats_data = values
                    imgui.text("Statistics:")
                imgui.text(f"  Min: {np.min(stats_data):.6f}")
                imgui.text(f"  Max: {np.max(stats_data):.6f}")
                imgui.text(f"  Mean: {np.mean(stats_data):.6f}")
                if stats_data.size > 1:
                    imgui.text(f"  Std: {np.std(stats_data):.6f}")

        except Exception as e:
            imgui.text(f"Error getting attribute: {e!s}")

    def _get_attribute_names(self, view: ArticulationView, attribute_name: str):
        try:
            if attribute_name.startswith("joint_q") or attribute_name.startswith("joint_f"):
                if attribute_name == "joint_q":
                    return view.joint_coord_names
                return view.joint_dof_names
            if attribute_name.startswith("body_"):
                return view.body_names
            return None
        except Exception:
            return None

    def _render_value_sliders(self, values, names, attribute_name: str, state):
        imgui = self.ui.imgui

        if attribute_name.startswith("joint_q"):
            slider_min, slider_max = -3.14159, 3.14159
        elif attribute_name.startswith("joint_qd"):
            slider_min, slider_max = -10.0, 10.0
        elif attribute_name.startswith("joint_f"):
            slider_min, slider_max = -100.0, 100.0
        else:
            if len(values) > 0 and values.dtype.kind in "biufc":
                val_min, val_max = float(np.min(values)), float(np.max(values))
                val_range = val_max - val_min
                if val_range < 1e-6:
                    slider_min = val_min - 1.0
                    slider_max = val_max + 1.0
                else:
                    padding = val_range * 0.2
                    slider_min = val_min - padding
                    slider_max = val_max + padding
            else:
                slider_min, slider_max = -1.0, 1.0

        if "slider_values" not in state:
            state["slider_values"] = {}

        slider_key = f"{attribute_name}_sliders"
        if slider_key not in state["slider_values"]:
            state["slider_values"][slider_key] = [float(v) for v in values]

        current_sliders = state["slider_values"][slider_key]
        while len(current_sliders) < len(values):
            current_sliders.append(0.0)
        while len(current_sliders) > len(values):
            current_sliders.pop()

        for i, val in enumerate(values):
            if i < len(current_sliders):
                current_sliders[i] = float(val)

        for i, val in enumerate(values):
            name = names[i] if names and i < len(names) else f"[{i}]"

            if isinstance(val, int | float) or hasattr(val, "dtype"):
                if name.startswith("floating_base"):
                    name = "base"

                display_name = name[:8] + "..." if len(name) > 8 else name
                display_name = f"{display_name:<11}"
                imgui.text(display_name)
                if imgui.is_item_hovered() and len(name) > 8:
                    imgui.set_tooltip(name)
                imgui.same_line()

                imgui.push_item_width(150)
                slider_id = f"##{attribute_name}_{i}"
                _changed, _new_val = imgui.slider_float(slider_id, current_sliders[i], slider_min, slider_max, "%.6f")
                imgui.pop_item_width()
            else:
                imgui.text(f"{name}: {val}")
