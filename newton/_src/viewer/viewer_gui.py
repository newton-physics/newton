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

import numpy as np
import warp as wp

import newton as nt
from newton.selection import ArticulationView

from .gl.gui import UI


class ViewerGui:
    """Shared ImGui rendering for concrete viewers (GL / RTX)."""

    def __init__(self, viewer, window):
        self._viewer = viewer
        self.ui = UI(window)

        # Camera keyboard movement (shared with GL/RTX)
        self._cam_vel = np.zeros(3, dtype=np.float32)
        self._cam_speed = 4.0
        self._cam_damp_tau = 0.083

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
        return bool(self.ui.io.want_capture_mouse)

    def is_keyboard_capturing(self) -> bool:
        if not self.is_available:
            return False
        return bool(self.ui.io.want_capture_keyboard)

    def should_ignore_mouse_input(self, allow_active_pick_drag: bool = False) -> bool:
        if allow_active_pick_drag and self.is_pick_active():
            return False
        return self.is_mouse_capturing()

    def should_ignore_keyboard_input(self) -> bool:
        return self.is_keyboard_capturing()

    def is_pick_active(self) -> bool:
        viewer = self._viewer
        if not getattr(viewer, "picking_enabled", False):
            return False
        picking = getattr(viewer, "picking", None)
        if picking is None or not hasattr(picking, "is_picking"):
            return False
        try:
            return bool(picking.is_picking())
        except Exception:
            return False

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
        if self.is_capturing():
            return
        camera = getattr(self._viewer, "camera", None)
        if camera is None:
            return

        import pyglet

        key = pyglet.window.key
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
        if is_key_down(key.W) or is_key_down(key.UP):
            desired += forward
        if is_key_down(key.S) or is_key_down(key.DOWN):
            desired -= forward
        if is_key_down(key.A) or is_key_down(key.LEFT):
            desired -= right
        if is_key_down(key.D) or is_key_down(key.RIGHT):
            desired += right
        if is_key_down(key.Q):
            desired -= up
        if is_key_down(key.E):
            desired += up

        dn = float(np.linalg.norm(desired))
        if dn > 1.0e-6:
            desired = desired / dn * self._cam_speed
        else:
            desired[:] = 0.0

        tau = max(1.0e-4, float(self._cam_damp_tau))
        self._cam_vel += (desired - self._cam_vel) * (dt / tau)

        pos = camera.pos
        camera.pos = type(pos)(
            pos.x + self._cam_vel[0] * dt, pos.y + self._cam_vel[1] * dt, pos.z + self._cam_vel[2] * dt
        )
        if hasattr(self._viewer, "_camera_dirty"):
            self._viewer._camera_dirty = True

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
        ray_start, ray_dir = camera.get_world_ray(fb_x, fb_y)
        picking.update(ray_start, ray_dir)
        return True

    def release_picking(self):
        picking = getattr(self._viewer, "picking", None)
        if picking is not None:
            picking.release()

    def render_frame(self, update_fps: bool = True):
        """Render GUI into the active OpenGL framebuffer."""
        if update_fps:
            self._viewer._update_fps()
        if not self.is_available or not self.show_ui:
            return
        self.ui.begin_frame()
        self._render_ui()
        self.ui.end_frame()
        self.ui.render()

    def register_ui_callback(self, callback, position="side"):
        self._viewer.register_ui_callback(callback, position=position)

    def _render_gizmos(self):
        if not self.is_available:
            return
        if not hasattr(self._viewer, "_gizmo_log") or not self._viewer._gizmo_log:
            return
        if not hasattr(self._viewer, "camera") or self._viewer.camera is None:
            return

        giz = self.ui.giz
        io = self.ui.io

        # Setup ImGuizmo viewport
        giz.set_orthographic(False)
        giz.set_rect(0.0, 0.0, float(io.display_size[0]), float(io.display_size[1]))
        giz.set_gizmo_size_clip_space(0.07)
        giz.set_axis_limit(0.0)
        giz.set_plane_limit(0.0)

        # Camera matrices
        view = self._viewer.camera.get_view_matrix().reshape(4, 4).transpose()
        proj = self._viewer.camera.get_projection_matrix().reshape(4, 4).transpose()

        # Draw & mutate each gizmo
        for gid, transform in self._viewer._gizmo_log.items():
            giz.push_id(str(gid))

            M = wp.transform_to_matrix(transform)

            def m44_to_mat16(m):
                """Row-major 4x4 -> giz.Matrix16 (column-major, 16 floats)."""
                m = np.asarray(m, dtype=np.float32).reshape(4, 4)
                return giz.Matrix16(m.flatten(order="F").tolist())

            view_ = m44_to_mat16(view)
            proj_ = m44_to_mat16(proj)
            M_ = m44_to_mat16(M)

            giz.manipulate(view_, proj_, giz.OPERATION.rotate, giz.MODE.world, M_, None, None)
            giz.manipulate(view_, proj_, giz.OPERATION.translate, giz.MODE.world, M_, None, None)

            M[:] = M_.values.reshape(4, 4, order="F")
            transform[:] = wp.transform_from_matrix(M)

            giz.pop_id()

    def _render_ui(self):
        """Render the complete ImGui interface."""
        if not self.is_available:
            return

        self._render_gizmos()
        self._render_left_panel()
        self._render_stats_overlay()

        for callback in self._viewer._ui_callbacks["free"]:
            callback(self.ui.imgui)

    def _render_left_panel(self):
        """Render left panel with model details and visualization controls."""
        if not self.is_available:
            return

        viewer = self._viewer
        imgui = self.ui.imgui

        nav_highlight_color = self.ui.get_theme_color(imgui.Col_.nav_cursor, (1.0, 1.0, 1.0, 1.0))

        io = self.ui.io
        imgui.set_next_window_pos(imgui.ImVec2(10, 10))
        imgui.set_next_window_size(imgui.ImVec2(300, io.display_size[1] - 20))

        flags = imgui.WindowFlags_.no_resize.value

        if imgui.begin(f"Newton Viewer v{nt.__version__}", flags=flags):
            imgui.separator()
            header_flags = 0

            if viewer.model is not None:
                imgui.set_next_item_open(True, imgui.Cond_.appearing)
                if imgui.collapsing_header("Model Information", flags=header_flags):
                    imgui.separator()
                    imgui.text(f"Worlds: {viewer.model.world_count}")
                    axis_names = ["X", "Y", "Z"]
                    imgui.text(f"Up Axis: {axis_names[viewer.model.up_axis]}")
                    gravity = viewer.model.gravity.numpy()[0]
                    imgui.text(f"Gravity: ({gravity[0]:.2f}, {gravity[1]:.2f}, {gravity[2]:.2f})")
                    _changed, viewer._paused = imgui.checkbox("Pause", viewer._paused)

                imgui.set_next_item_open(True, imgui.Cond_.appearing)
                if imgui.collapsing_header("Visualization", flags=header_flags):
                    imgui.separator()
                    _changed, viewer.show_joints = imgui.checkbox("Show Joints", viewer.show_joints)
                    _changed, viewer.show_contacts = imgui.checkbox("Show Contacts", viewer.show_contacts)
                    _changed, viewer.show_particles = imgui.checkbox("Show Particles", viewer.show_particles)
                    _changed, viewer.show_springs = imgui.checkbox("Show Springs", viewer.show_springs)
                    _changed, viewer.show_com = imgui.checkbox("Show Center of Mass", viewer.show_com)
                    _changed, viewer.show_triangles = imgui.checkbox("Show Cloth", viewer.show_triangles)
                    _changed, viewer.show_collision = imgui.checkbox("Show Collision", viewer.show_collision)
                    _changed, viewer.show_visual = imgui.checkbox("Show Visual", viewer.show_visual)
                    _changed, viewer.show_inertia_boxes = imgui.checkbox(
                        "Show Inertia Boxes", viewer.show_inertia_boxes
                    )

            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Rendering Options"):
                imgui.separator()
                _changed, viewer.vsync = imgui.checkbox("VSync", viewer.vsync)

            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Example Options"):
                for callback in viewer._ui_callbacks["side"]:
                    callback(self.ui.imgui)

            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Camera"):
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
                imgui.text("Scroll - Zoom")
                imgui.text("Space - Pause/Resume")
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

    def _render_stats_overlay(self):
        """Render performance overlay in the top-right corner."""
        if not self.is_available:
            return

        viewer = self._viewer
        imgui = self.ui.imgui
        io = self.ui.io
        fps_color = (1.0, 1.0, 1.0, 1.0)

        window_pos = (io.display_size[0] - 10, 10)
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
            imgui.text(f"FPS: {viewer._current_fps:.1f}")
            imgui.pop_style_color()

            if viewer.model is not None:
                imgui.separator()
                imgui.text(f"Bodies: {viewer.model.body_count}")
                imgui.text(f"Shapes: {viewer.model.shape_count}")
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

        for callback in viewer._ui_callbacks["stats"]:
            callback(self.ui.imgui)

        imgui.end()
        if pushed_window_bg:
            imgui.pop_style_color()

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
        if imgui.is_item_hovered():
            imgui.set_tooltip("Pattern to match articulations (e.g., '*', 'robot*', 'cartpole')")

        imgui.spacing()
        imgui.text("Joint Filters (optional):")
        imgui.push_item_width(150)
        imgui.text("Include:")
        imgui.same_line()
        _changed, state["include_joints"] = imgui.input_text("##inc_joints", state["include_joints"])
        if imgui.is_item_hovered():
            imgui.set_tooltip("Comma-separated joint names/patterns")
        imgui.text("Exclude:")
        imgui.same_line()
        _changed, state["exclude_joints"] = imgui.input_text("##exc_joints", state["exclude_joints"])
        if imgui.is_item_hovered():
            imgui.set_tooltip("Comma-separated joint names/patterns")
        imgui.pop_item_width()

        imgui.spacing()
        imgui.text("Link Filters (optional):")
        imgui.push_item_width(150)
        imgui.text("Include:")
        imgui.same_line()
        _changed, state["include_links"] = imgui.input_text("##inc_links", state["include_links"])
        if imgui.is_item_hovered():
            imgui.set_tooltip("Comma-separated link names/patterns")
        imgui.text("Exclude:")
        imgui.same_line()
        _changed, state["exclude_links"] = imgui.input_text("##exc_links", state["exclude_links"])
        if imgui.is_item_hovered():
            imgui.set_tooltip("Comma-separated link names/patterns")
        imgui.pop_item_width()

        imgui.spacing()
        if imgui.button("Create Articulation View"):
            self._create_articulation_view()

        if state["selected_articulation_view"] is None:
            return

        view = state["selected_articulation_view"]
        imgui.separator()
        imgui.text(f"  Count: {view.count}")
        imgui.text(f"  Joints: {view.joint_count}")
        imgui.text(f"  Links: {view.link_count}")
        imgui.text(f"  DOFs: {view.joint_dof_count}")
        imgui.text(f"  Fixed base: {view.is_fixed_base}")
        imgui.text(f"  Floating base: {view.is_floating_base}")

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
            if imgui.begin_child("values_scroll", 0, 300, border=True):
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
