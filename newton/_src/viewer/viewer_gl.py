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

import time

import numpy as np
import warp as wp

import newton as nt
from newton.selection import ArticulationView

from .camera import Camera
from .gl.gui import UI
from .gl.opengl import LinesGL, MeshGL, MeshInstancerGL, RendererGL
from .picking import Picking
from .viewer import ViewerBase


class ViewerGL(ViewerBase):
    def __init__(self, model):
        super().__init__(model)

        # map from path to any object type
        self.objects = {}
        self.lines = {}
        self.renderer = RendererGL()
        self.renderer.set_title("Newton Viewer")

        self._paused = False

        # State caching for selection panel
        self._last_state = None
        self._last_control = None

        # Selection panel state
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

        self.picking = Picking(model, pick_stiffness=10000.0, pick_damping=1000.0)

        self.renderer.register_key_press(self.on_key_press)
        self.renderer.register_key_release(self.on_key_release)
        self.renderer.register_mouse_press(self.on_mouse_press)
        self.renderer.register_mouse_release(self.on_mouse_release)
        self.renderer.register_mouse_drag(self.on_mouse_drag)
        self.renderer.register_mouse_scroll(self.on_mouse_scroll)
        self.renderer.register_resize(self.on_resize)

        # Camera movement settings
        self._camera_speed = 0.04
        self._cam_vel = np.zeros(3, dtype=np.float32)
        self._cam_speed = 4.0  # m/s
        self._cam_damp_tau = 0.083  # s

        fb_w, fb_h = self.renderer.window.get_framebuffer_size()
        self.camera = Camera(up_axis=model.up_axis, width=fb_w, height=fb_h)

        self._populate(model)

        # initialize viewer-local timer for per-frame integration
        self._last_time = time.perf_counter()

        self.ui = UI(self.renderer.window)

        # Performance tracking
        self._fps_history = []
        self._last_fps_time = time.perf_counter()
        self._frame_count = 0
        self._current_fps = 0.0

        # UI visibility toggle
        self.show_ui = True

    def log_mesh(
        self,
        name,
        points: wp.array,
        indices: wp.array,
        normals: wp.array = None,
        uvs: wp.array = None,
        hidden=False,
        backface_culling=True,
    ):
        assert isinstance(points, wp.array)
        assert isinstance(indices, wp.array)
        assert normals is None or isinstance(normals, wp.array)
        assert uvs is None or isinstance(uvs, wp.array)

        if name not in self.objects:
            self.objects[name] = MeshGL(
                len(points), len(indices), self.device, hidden=hidden, backface_culling=backface_culling
            )

        self.objects[name].update(points, indices, normals, uvs)

    def log_instances(self, name, mesh, xforms, scales, colors, materials):
        # check that mesh exists
        if mesh not in self.objects:
            raise RuntimeError(f"Path {mesh} not found")

        # check it is a mesh object
        if not isinstance(self.objects[mesh], MeshGL):
            raise RuntimeError(f"Path {mesh} is not a Mesh object")

        if name not in self.objects:
            self.objects[name] = MeshInstancerGL(len(xforms), self.objects[mesh])

        self.objects[name].update(xforms, scales, colors, materials)

    def log_lines(
        self,
        name,
        line_begins: wp.array,
        line_ends: wp.array,
        line_colors,
        hidden=False,
    ):
        """Log line data for rendering.

        Args:
            name: Unique identifier for the line batch
            line_begins: Array of line start positions (shape: [N, 3]) or None for empty
            line_ends: Array of line end positions (shape: [N, 3]) or None for empty
            line_colors: Array of line colors (shape: [N, 3]) or tuple/list of RGB or None for empty
            hidden: Whether the lines are initially hidden
        """
        # Handle empty logs by resetting the LinesGL object
        if line_begins is None or line_ends is None or line_colors is None:
            if name in self.lines:
                self.lines[name].update(None, None, None)
            return

        assert isinstance(line_begins, wp.array)
        assert isinstance(line_ends, wp.array)
        num_lines = len(line_begins)
        assert len(line_ends) == num_lines, "Number of line ends must match line begins"

        # Handle tuple/list colors by expanding to array (only if not already converted above)
        if isinstance(line_colors, (tuple, list)):
            if num_lines > 0:
                color_vec = wp.vec3(*line_colors)
                line_colors = wp.zeros(num_lines, dtype=wp.vec3, device=self.device)
                line_colors.fill_(color_vec)  # Efficiently fill on GPU
            else:
                # Handle zero lines case
                line_colors = wp.array([], dtype=wp.vec3, device=self.device)

        assert isinstance(line_colors, wp.array)
        assert len(line_colors) == num_lines, "Number of line colors must match line begins"

        # Create or resize LinesGL object based on current requirements
        if name not in self.lines:
            # Start with reasonable default size, will expand as needed
            max_lines = max(num_lines, 1000)  # Reasonable default
            self.lines[name] = LinesGL(max_lines, self.device, hidden=hidden)
        elif num_lines > self.lines[name].max_lines:
            # Need to recreate with larger capacity
            self.lines[name].destroy()
            max_lines = max(num_lines, self.lines[name].max_lines * 2)
            self.lines[name] = LinesGL(max_lines, self.device, hidden=hidden)

        self.lines[name].update(line_begins, line_ends, line_colors)

    def log_points(self, name, state):
        pass

    def log_array(self, name, array):
        pass

    def log_scalar(self, name, value):
        pass

    def log_state(self, state):
        """Override log_state to cache the state for the selection panel."""
        # Cache the state for selection panel use
        self._last_state = state
        # Call parent implementation
        super().log_state(state)

        self._render_picking_line(state)

    def _render_picking_line(self, state):
        """Render a line from the mouse cursor to the COM of the picked body."""
        if not self.picking.is_picking():
            # Clear the picking line if not picking
            self.log_lines("picking_line", None, None, None)
            return

        # Get the picked body index
        pick_body_idx = self.picking.pick_body.numpy()[0]
        if pick_body_idx < 0:
            self.log_lines("picking_line", None, None, None)
            return

        # Get the pick target
        pick_state = self.picking.pick_state.numpy()
        pick_target = wp.vec3(pick_state[3], pick_state[4], pick_state[5])

        # Get the body's COM position
        body_transforms = state.body_q.numpy()
        if pick_body_idx >= len(body_transforms):
            self.log_lines("picking_line", None, None, None)
            return
        body_transform = body_transforms[pick_body_idx]
        com_position = wp.vec3(body_transform[0], body_transform[1], body_transform[2])

        # Create line data
        line_begins = wp.array([com_position], dtype=wp.vec3, device=self.device)
        line_ends = wp.array([pick_target], dtype=wp.vec3, device=self.device)
        line_colors = wp.array([wp.vec3(0.0, 1.0, 1.0)], dtype=wp.vec3, device=self.device)

        # Render the line
        self.log_lines("picking_line", line_begins, line_ends, line_colors, hidden=False)

    def begin_frame(self, time):
        pass

    def end_frame(self):
        """Finish rendering the current frame.

        This method first updates the renderer which will poll and process
        window events.  It is possible that the user closes the window during
        this event processing step, which would invalidate the underlying
        OpenGL context.  Trying to issue GL calls after the context has been
        destroyed results in a crash (access violation).  Therefore we check
        whether an exit was requested and early-out before touching GL if so.
        """
        self._update()

        # continue running the viewer update while the simulation is paused
        while self.is_paused():
            self._update()

    def apply_forces(self, state):
        """Apply viewer-driven forces to the model."""
        self.picking._apply_picking_force(state)

    def _update(self):
        # Process events and update the internal state
        self.renderer.update()

        # Integrate camera motion with viewer-owned timing
        now = time.perf_counter()
        dt = max(0.0, min(0.1, now - self._last_time))
        self._last_time = now
        self._update_camera(dt)

        # If the window was closed during event processing, skip rendering
        if self.renderer.has_exit():
            return

        # Render the scene and present it
        self.renderer.render(self.camera, self.objects, self.lines)

        # Always update FPS tracking, even if UI is hidden
        self._update_fps()

        if self.ui.is_available and self.show_ui:
            self.ui.begin_frame()

            # Render the UI
            self._render_ui()

            self.ui.end_frame()
            self.ui.render()

        self.renderer.present()

    def is_running(self) -> bool:
        """Check if the viewer is still running."""
        return not self.renderer.has_exit()

    def is_paused(self) -> bool:
        """Check if the simulation is paused."""
        return self._paused

    def close(self):
        """Close the viewer and clean up resources."""
        self.renderer.close()

    def is_key_down(self, key):
        """Check if a key is currently pressed.

        Args:
            key: Either a string representing a character/key name, or an int
                 representing a pyglet key constant.

                 String examples: 'w', 'a', 's', 'd', 'space', 'escape', 'ent
                 Int examples: pyglet.window.key.W, pyglet.window.key.SPACE

        Returns:
            bool: True if the key is currently pressed, False otherwise.
        """
        try:
            import pyglet  # noqa: PLC0415
        except Exception:
            return False

        if isinstance(key, str):
            # Convert string to pyglet key constant
            key = key.lower()

            # Handle single characters
            if len(key) == 1 and key.isalpha():
                key_code = getattr(pyglet.window.key, key.upper(), None)
            elif len(key) == 1 and key.isdigit():
                key_code = getattr(pyglet.window.key, f"_{key}", None)
            else:
                # Handle special key names
                special_keys = {
                    "space": pyglet.window.key.SPACE,
                    "escape": pyglet.window.key.ESCAPE,
                    "esc": pyglet.window.key.ESCAPE,
                    "enter": pyglet.window.key.ENTER,
                    "return": pyglet.window.key.ENTER,
                    "tab": pyglet.window.key.TAB,
                    "shift": pyglet.window.key.LSHIFT,
                    "ctrl": pyglet.window.key.LCTRL,
                    "alt": pyglet.window.key.LALT,
                    "up": pyglet.window.key.UP,
                    "down": pyglet.window.key.DOWN,
                    "left": pyglet.window.key.LEFT,
                    "right": pyglet.window.key.RIGHT,
                    "backspace": pyglet.window.key.BACKSPACE,
                    "delete": pyglet.window.key.DELETE,
                }
                key_code = special_keys.get(key, None)

            if key_code is None:
                return False
        else:
            # Assume it's already a pyglet key constant
            key_code = key

        return self.renderer.is_key_down(key_code)

    # events

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        """Handle mouse scroll for zooming (FOV adjustment)."""
        if self.ui.is_capturing():
            return

        fov_delta = scroll_y * 2.0
        self.camera.fov -= fov_delta
        self.camera.fov = max(min(self.camera.fov, 90.0), 15.0)

    def on_mouse_press(self, x, y, button, modifiers):
        if self.ui.is_capturing():
            return

        import pyglet  # noqa: PLC0415

        # Handle right-click for picking
        if button == pyglet.window.mouse.RIGHT:
            ray_start, ray_dir = self.camera.get_world_ray(x, y)
            print(f"ray: start {ray_start}, dir {ray_dir}")
            self.picking.pick(self._last_state, ray_start, ray_dir)

    def on_mouse_release(self, x, y, button, modifiers):
        """Handle mouse release events to stop dragging."""
        self.picking.release()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self.ui.is_capturing():
            return

        import pyglet  # noqa: PLC0415

        if buttons & pyglet.window.mouse.LEFT:
            sensitivity = 0.1
            dx *= sensitivity
            dy *= sensitivity

            # Map screen-space right drag to a right turn (clockwise),
            # independent of world up-axis convention.
            self.camera.yaw -= dx
            self.camera.pitch += dy

        if buttons & pyglet.window.mouse.RIGHT:
            ray_start, ray_dir = self.camera.get_world_ray(x, y)

            if self.picking.is_picking():
                self.picking.update(ray_start, ray_dir)

    def on_mouse_motion(self, x, y, dx, dy):
        pass

    def on_key_press(self, symbol, modifiers):
        if self.ui.is_capturing():
            return

        try:
            import pyglet  # noqa: PLC0415
        except Exception:
            return

        if symbol == pyglet.window.key.H:
            self.show_ui = not self.show_ui
        elif symbol == pyglet.window.key.SPACE:
            # Toggle pause with space key
            self._paused = not self._paused
        elif symbol == pyglet.window.key.ESCAPE or symbol == pyglet.window.key.Q:
            # Exit with Escape or Q key
            self.renderer.close()

    def on_key_release(self, symbol, modifiers):
        pass

    def _update_camera(self, dt: float):
        if self.ui.is_capturing():
            return

        # camera-relative basis
        forward = np.array(self.camera.get_front(), dtype=np.float32)
        right = np.array(self.camera.get_right(), dtype=np.float32)
        up = np.array(self.camera.get_up(), dtype=np.float32)

        # keep motion in the horizontal plane
        forward -= up * float(np.dot(forward, up))
        right -= up * float(np.dot(right, up))
        # renormalize
        fn = float(np.linalg.norm(forward))
        ln = float(np.linalg.norm(right))
        if fn > 1.0e-6:
            forward /= fn
        if ln > 1.0e-6:
            right /= ln

        import pyglet  # noqa: PLC0415

        desired = np.zeros(3, dtype=np.float32)
        if self.renderer.is_key_down(pyglet.window.key.W) or self.renderer.is_key_down(pyglet.window.key.UP):
            desired += forward
        if self.renderer.is_key_down(pyglet.window.key.S) or self.renderer.is_key_down(pyglet.window.key.DOWN):
            desired -= forward
        if self.renderer.is_key_down(pyglet.window.key.A) or self.renderer.is_key_down(pyglet.window.key.LEFT):
            desired -= right  # strafe left
        if self.renderer.is_key_down(pyglet.window.key.D) or self.renderer.is_key_down(pyglet.window.key.RIGHT):
            desired += right  # strafe right

        dn = float(np.linalg.norm(desired))
        if dn > 1.0e-6:
            desired = desired / dn * self._cam_speed
        else:
            desired[:] = 0.0

        tau = max(1.0e-4, float(self._cam_damp_tau))
        self._cam_vel += (desired - self._cam_vel) * (dt / tau)

        # integrate position
        dv = type(self.camera.pos)(*self._cam_vel)
        self.camera.pos += dv * dt

    def on_resize(self, width, height):
        fb_w, fb_h = self.renderer.window.get_framebuffer_size()
        self.camera.update_screen_size(fb_w, fb_h)

        self.ui.resize(width, height)

    def _update_fps(self):
        """Update FPS calculation."""
        current_time = time.perf_counter()
        self._frame_count += 1

        # Update FPS every second
        if current_time - self._last_fps_time >= 1.0:
            time_delta = current_time - self._last_fps_time
            self._current_fps = self._frame_count / time_delta
            self._fps_history.append(self._current_fps)

            # Keep only last 60 FPS readings
            if len(self._fps_history) > 60:
                self._fps_history.pop(0)

            self._last_fps_time = current_time
            self._frame_count = 0

    def _render_ui(self):
        """Render the complete ImGui interface."""

        if not self.ui.is_available:
            return

        # Render left panel
        self._render_left_panel()

        # Render top-right stats overlay
        self._render_stats_overlay()

    def _render_left_panel(self):
        """Render the left panel with model info and visualization controls."""
        imgui = self.ui.imgui

        # Use theme colors directly
        nav_highlight_color = self.ui.get_theme_color(imgui.COLOR_NAV_HIGHLIGHT, (1.0, 1.0, 1.0, 1.0))

        # Position the window on the left side
        io = self.ui.io
        imgui.set_next_window_position(10, 10)
        imgui.set_next_window_size(300, io.display_size[1] - 20)

        # Main control panel window - use safe flag values
        flags = imgui.WINDOW_NO_RESIZE

        if imgui.begin(f"Newton Viewer v{nt.__version__}", flags=flags):
            imgui.separator()

            # Collapsing headers default-open handling (first frame only)
            header_flags = 0

            cond = getattr(imgui, "COND_APPEARING", getattr(imgui, "COND_FIRST_USE_EVER", 0))

            # Model Information section
            if hasattr(imgui, "set_next_item_open") and not getattr(self, "_ui_headers_init", False):
                imgui.set_next_item_open(True, cond)
            _open = imgui.collapsing_header("Model Information", flags=header_flags)
            if isinstance(_open, tuple):
                _open = _open[0]
            if _open:
                imgui.separator()
                imgui.text(f"Environments: {self.model.num_envs}")
                axis_names = ["X", "Y", "Z"]
                imgui.text(f"Up Axis: {axis_names[self.model.up_axis]}")
                gravity = self.model.gravity
                gravity_text = f"Gravity: ({gravity[0]:.2f}, {gravity[1]:.2f}, {gravity[2]:.2f})"
                imgui.text(gravity_text)

                # Pause simulation checkbox
                changed, self._paused = imgui.checkbox("Pause", self._paused)

            # Visualization Controls section
            if hasattr(imgui, "set_next_item_open") and not getattr(self, "_ui_headers_init", False):
                imgui.set_next_item_open(True, cond)
            _open = imgui.collapsing_header("Visualization", flags=header_flags)
            if isinstance(_open, tuple):
                _open = _open[0]
            if _open:
                imgui.separator()

                # Joint visualization
                show_joints = self.options["show_joints"]
                changed, self.options["show_joints"] = imgui.checkbox("Show Joints", show_joints)

                # Contact visualization
                show_contacts = self.options["show_contacts"]
                changed, self.options["show_contacts"] = imgui.checkbox("Show Contacts", show_contacts)

                # Particle visualization
                show_particles = self.options["show_particles"]
                changed, self.options["show_particles"] = imgui.checkbox("Show Particles", show_particles)

                # Spring visualization
                show_springs = self.options["show_springs"]
                changed, self.options["show_springs"] = imgui.checkbox("Show Springs", show_springs)

                # Center of mass visualization
                show_com = self.options["show_com"]
                changed, self.options["show_com"] = imgui.checkbox("Show Center of Mass", show_com)

                # Triangle mesh visualization
                show_triangles = self.options["show_triangles"]
                changed, self.options["show_triangles"] = imgui.checkbox("Show Triangles", show_triangles)

            # Rendering Options section
            if hasattr(imgui, "set_next_item_open") and not getattr(self, "_ui_headers_init", False):
                imgui.set_next_item_open(True, cond)
            _open = imgui.collapsing_header("Rendering Options")
            if isinstance(_open, tuple):
                _open = _open[0]
            if _open:
                imgui.separator()

                # Sky rendering
                changed, self.renderer.draw_sky = imgui.checkbox("Sky", self.renderer.draw_sky)

                # Shadow rendering
                changed, self.renderer.draw_shadows = imgui.checkbox("Shadows", self.renderer.draw_shadows)

                # Wireframe mode
                changed, self.renderer.draw_wireframe = imgui.checkbox("Wireframe", self.renderer.draw_wireframe)

                # Light color
                changed, self.renderer._light_color = imgui.color_edit3("Light Color", *self.renderer._light_color)
                # Sky color
                changed, self.renderer.sky_upper = imgui.color_edit3("Sky Color", *self.renderer.sky_upper)
                # Ground color
                changed, self.renderer.sky_lower = imgui.color_edit3("Ground Color", *self.renderer.sky_lower)

            # Camera Information section
            if hasattr(imgui, "set_next_item_open") and not getattr(self, "_ui_headers_init", False):
                imgui.set_next_item_open(True, cond)
            _open = imgui.collapsing_header("Camera")
            if isinstance(_open, tuple):
                _open = _open[0]
            if _open:
                imgui.separator()

                pos = self.camera.pos
                pos_text = f"Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
                imgui.text(pos_text)
                imgui.text(f"FOV: {self.camera.fov:.1f}°")
                imgui.text(f"Yaw: {self.camera.yaw:.1f}°")
                imgui.text(f"Pitch: {self.camera.pitch:.1f}°")

                # Camera controls hint
                imgui.separator()
                imgui.push_style_color(imgui.COLOR_TEXT, *nav_highlight_color)
                imgui.text("Controls:")
                imgui.pop_style_color()
                imgui.text("WASD - Move camera")
                imgui.text("Mouse - Look around")
                imgui.text("Scroll - Zoom")

            # Selection API section
            self._render_selection_panel()

        imgui.end()

        # Mark headers as initialized so user can collapse in subsequent frames
        if not getattr(self, "_ui_headers_init", False):
            self._ui_headers_init = True

    def _render_stats_overlay(self):
        """Render performance stats overlay in the top-right corner."""
        imgui = self.ui.imgui
        io = self.ui.io

        # Use fallback color for FPS display
        fps_color = (1.0, 1.0, 1.0, 1.0)  # Bright white

        # Position in top-right corner
        window_pos = (io.display_size[0] - 10, 10)
        imgui.set_next_window_position(window_pos[0], window_pos[1], pivot_x=1.0, pivot_y=0.0)

        # Transparent background, auto-sized, non-resizable/movable - use safe flags
        #        try:
        flags = (
            imgui.WINDOW_NO_DECORATION
            | imgui.WINDOW_ALWAYS_AUTO_RESIZE
            | imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_NO_SAVED_SETTINGS
            | imgui.WINDOW_NO_FOCUS_ON_APPEARING
            | imgui.WINDOW_NO_NAV
            | imgui.WINDOW_NO_MOVE
        )

        # Set semi-transparent background for the overlay window
        pushed_window_bg = False
        try:
            # Preferred API name in pyimgui
            imgui.set_next_window_bg_alpha(0.7)
        except AttributeError:
            # Fallback: temporarily override window bg color alpha
            try:
                style = imgui.get_style()
                bg = style.colors[imgui.COLOR_WINDOW_BACKGROUND]
                r, g, b = bg[0], bg[1], bg[2]
            except Exception:
                # Reasonable dark default
                r, g, b = 0.094, 0.094, 0.094
            imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, r, g, b, 0.7)
            pushed_window_bg = True

        if imgui.begin("Performance Stats", flags=flags):
            # FPS display
            fps_text = f"FPS: {self._current_fps:.1f}"
            imgui.push_style_color(imgui.COLOR_TEXT, *fps_color)
            imgui.text(fps_text)
            imgui.pop_style_color()

            # Model stats
            imgui.separator()
            imgui.text(f"Bodies: {self.model.body_count}")
            imgui.text(f"Shapes: {self.model.shape_count}")
            imgui.text(f"Joints: {self.model.joint_count}")
            imgui.text(f"Particles: {self.model.particle_count}")
            imgui.text(f"Springs: {self.model.spring_count}")
            imgui.text(f"Triangles: {self.model.tri_count}")
            imgui.text(f"Edges: {self.model.edge_count}")
            imgui.text(f"Tetrahedra: {self.model.tet_count}")

            # Rendered objects count
            imgui.separator()
            imgui.text(f"Unique Objects: {len(self.objects)}")

        imgui.end()

        # Restore bg color if we pushed it
        if pushed_window_bg:
            imgui.pop_style_color()

    def _render_selection_panel(self):
        """Render the selection panel for Newton Model introspection."""
        imgui = self.ui.imgui

        # Selection Panel section
        header_flags = 0
        cond = getattr(imgui, "COND_APPEARING", getattr(imgui, "COND_FIRST_USE_EVER", 0))
        if hasattr(imgui, "set_next_item_open") and not getattr(self, "_ui_headers_init", False):
            imgui.set_next_item_open(False, cond)  # Default to closed
        _open = imgui.collapsing_header("Selection API", flags=header_flags)
        if isinstance(_open, tuple):
            _open = _open[0]
        if _open:
            imgui.separator()

            # Check if we have state data available
            if self._last_state is None:
                imgui.text("No state data available.")
                imgui.text("Start simulation to enable selection.")
                return

            state = self._selection_ui_state

            # Display error message if any
            if state["error_message"]:
                imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.3, 0.3, 1.0)
                imgui.text(f"Error: {state['error_message']}")
                imgui.pop_style_color()
                imgui.separator()

            # Articulation Pattern Input
            imgui.text("Articulation Pattern:")
            imgui.push_item_width(200)
            changed, state["selected_articulation_pattern"] = imgui.input_text(
                "##pattern", state["selected_articulation_pattern"], 256
            )
            imgui.pop_item_width()
            if imgui.is_item_hovered():
                tooltip = "Pattern to match articulations (e.g., '*', 'robot*', 'cartpole')"
                imgui.set_tooltip(tooltip)

            # Joint filtering
            imgui.spacing()
            imgui.text("Joint Filters (optional):")
            imgui.push_item_width(150)
            imgui.text("Include:")
            imgui.same_line()
            _, state["include_joints"] = imgui.input_text("##inc_joints", state["include_joints"], 256)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Comma-separated joint names/patterns")

            imgui.text("Exclude:")
            imgui.same_line()
            _, state["exclude_joints"] = imgui.input_text("##exc_joints", state["exclude_joints"], 256)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Comma-separated joint names/patterns")
            imgui.pop_item_width()

            # Link filtering
            imgui.spacing()
            imgui.text("Link Filters (optional):")
            imgui.push_item_width(150)
            imgui.text("Include:")
            imgui.same_line()
            _, state["include_links"] = imgui.input_text("##inc_links", state["include_links"], 256)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Comma-separated link names/patterns")

            imgui.text("Exclude:")
            imgui.same_line()
            _, state["exclude_links"] = imgui.input_text("##exc_links", state["exclude_links"], 256)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Comma-separated link names/patterns")
            imgui.pop_item_width()

            # Create View Button
            imgui.spacing()
            if imgui.button("Create Articulation View"):
                self._create_articulation_view()

            # Show view info if created
            if state["selected_articulation_view"] is not None:
                view = state["selected_articulation_view"]
                imgui.separator()
                imgui.text(f"  Count: {view.count}")
                imgui.text(f"  Joints: {view.joint_count}")
                imgui.text(f"  Links: {view.link_count}")
                imgui.text(f"  DOFs: {view.joint_dof_count}")
                imgui.text(f"  Fixed base: {view.is_fixed_base}")
                imgui.text(f"  Floating base: {view.is_floating_base}")

                # Attribute selector
                imgui.spacing()
                imgui.text("Select Attribute:")
                imgui.push_item_width(150)
                if state["selected_attribute"] in state["attribute_options"]:
                    current_attr_idx = state["attribute_options"].index(state["selected_attribute"])
                else:
                    current_attr_idx = 0
                _, new_attr_idx = imgui.combo("##attribute", current_attr_idx, state["attribute_options"])
                state["selected_attribute"] = state["attribute_options"][new_attr_idx]
                imgui.pop_item_width()

                # Toggle values display
                _, state["show_values"] = imgui.checkbox("Show Values", state["show_values"])

                # Display attribute values if requested
                if state["show_values"]:
                    self._render_attribute_values(view, state["selected_attribute"])

    def _create_articulation_view(self):
        """Create an ArticulationView based on current UI state."""
        state = self._selection_ui_state

        try:
            # Clear any previous error
            state["error_message"] = ""

            # Parse filter strings
            if state["include_joints"]:
                include_joints = [j.strip() for j in state["include_joints"].split(",") if j.strip()]
            else:
                include_joints = None

            if state["exclude_joints"]:
                exclude_joints = [j.strip() for j in state["exclude_joints"].split(",") if j.strip()]
            else:
                exclude_joints = None

            if state["include_links"]:
                include_links = [link.strip() for link in state["include_links"].split(",") if link.strip()]
            else:
                include_links = None

            if state["exclude_links"]:
                exclude_links = [link.strip() for link in state["exclude_links"].split(",") if link.strip()]
            else:
                exclude_links = None

            # Create ArticulationView
            state["selected_articulation_view"] = ArticulationView(
                model=self.model,
                pattern=state["selected_articulation_pattern"],
                include_joints=include_joints,
                exclude_joints=exclude_joints,
                include_links=include_links,
                exclude_links=exclude_links,
                verbose=False,  # Don't print to console in UI
            )

        except Exception as e:
            state["error_message"] = str(e)
            state["selected_articulation_view"] = None

    def _render_attribute_values(self, view: ArticulationView, attribute_name: str):
        """Render the values of the selected attribute."""
        imgui = self.ui.imgui
        state = self._selection_ui_state

        try:
            # Determine source based on attribute
            if attribute_name.startswith("joint_f"):
                # Forces come from control
                if hasattr(self, "_last_control") and self._last_control is not None:
                    source = self._last_control
                else:
                    imgui.text("No control data available for forces")
                    return
            else:
                # Other attributes come from state or model
                source = self._last_state

            # Get the attribute values
            values = view.get_attribute(attribute_name, source).numpy()

            imgui.separator()
            imgui.text(f"Attribute: {attribute_name}")
            imgui.text(f"Shape: {values.shape}")
            imgui.text(f"Dtype: {values.dtype}")

            # Handle batch dimension selection for 2D arrays
            if len(values.shape) == 2:
                batch_size = values.shape[0]
                imgui.spacing()
                imgui.text("Batch/Environment Selection:")
                imgui.push_item_width(100)

                # Ensure selected batch index is valid
                state["selected_batch_idx"] = max(0, min(state["selected_batch_idx"], batch_size - 1))

                _, state["selected_batch_idx"] = imgui.slider_int(
                    "##batch", state["selected_batch_idx"], 0, batch_size - 1
                )
                imgui.pop_item_width()
                imgui.same_line()
                text = f"Environment {state['selected_batch_idx']} / {batch_size}"
                imgui.text(text)

            # Display values as sliders in a scrollable region
            imgui.spacing()
            imgui.text("Values:")

            # Create a child window for scrollable content
            if imgui.begin_child("values_scroll", 0, 300, border=True):
                if len(values.shape) == 1:
                    # 1D array - show as sliders with names if available
                    names = self._get_attribute_names(view, attribute_name)
                    self._render_value_sliders(values, names, attribute_name, state)

                elif len(values.shape) == 2:
                    # 2D array - show selected batch with joint names
                    batch_idx = state["selected_batch_idx"]
                    selected_batch = values[batch_idx]
                    names = self._get_attribute_names(view, attribute_name)
                    self._render_value_sliders(selected_batch, names, attribute_name, state)

                else:
                    # Higher dimensional - just show summary
                    shape_str = str(values.shape)
                    imgui.text(f"Multi-dimensional array with shape {shape_str}")

            imgui.end_child()

            # Show some statistics for numeric data
            if values.dtype.kind in "biufc":  # numeric types
                imgui.spacing()

                # Calculate stats on the selected batch for 2D arrays
                if len(values.shape) == 2:
                    batch_idx = state["selected_batch_idx"]
                    stats_data = values[batch_idx]
                    imgui.text(f"Statistics for Environment {batch_idx}:")
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
        """Get the names associated with an attribute (joint names, link names, etc.)."""
        try:
            if attribute_name.startswith("joint_q") or attribute_name.startswith("joint_f"):
                # For joint positions/velocities/forces, return DOF names or coord names
                if attribute_name == "joint_q":
                    return view.joint_coord_names
                else:  # joint_qd, joint_f
                    return view.joint_dof_names
            elif attribute_name.startswith("body_"):
                # For body attributes, return body/link names
                return view.body_names
            else:
                return None
        except Exception:
            return None

    def _render_value_sliders(self, values, names, attribute_name: str, state):
        """Render values as individual sliders for each DOF."""
        imgui = self.ui.imgui

        # Determine appropriate slider ranges based on attribute type
        if attribute_name.startswith("joint_q"):
            # Joint positions - use reasonable angle/position ranges
            slider_min, slider_max = -3.14159, 3.14159  # Default to ±π
        elif attribute_name.startswith("joint_qd"):
            # Joint velocities - use reasonable velocity ranges
            slider_min, slider_max = -10.0, 10.0
        elif attribute_name.startswith("joint_f"):
            # Joint forces - use reasonable force ranges
            slider_min, slider_max = -100.0, 100.0
        else:
            # For other attributes, use data-driven ranges
            if len(values) > 0 and values.dtype.kind in "biufc":  # numeric
                val_min, val_max = float(np.min(values)), float(np.max(values))
                val_range = val_max - val_min
                if val_range < 1e-6:  # Nearly constant values
                    slider_min = val_min - 1.0
                    slider_max = val_max + 1.0
                else:
                    # Add 20% padding
                    padding = val_range * 0.2
                    slider_min = val_min - padding
                    slider_max = val_max + padding
            else:
                slider_min, slider_max = -1.0, 1.0

        # Initialize slider state if needed
        if "slider_values" not in state:
            state["slider_values"] = {}

        slider_key = f"{attribute_name}_sliders"
        if slider_key not in state["slider_values"]:
            state["slider_values"][slider_key] = [float(v) for v in values]

        # Ensure slider values array has correct length
        current_sliders = state["slider_values"][slider_key]
        while len(current_sliders) < len(values):
            current_sliders.append(0.0)
        while len(current_sliders) > len(values):
            current_sliders.pop()

        # Update slider values to match current data
        for i, val in enumerate(values):
            if i < len(current_sliders):
                current_sliders[i] = float(val)

        # Render sliders
        for i, val in enumerate(values):
            name = names[i] if names and i < len(names) else f"[{i}]"

            if isinstance(val, (int, float)) or hasattr(val, "dtype"):
                # shorten floating base key for ui
                # todo: consider doing this in the importers
                if name.startswith("floating_base"):
                    name = "base"

                # Truncate name for display but keep full name for tooltip
                display_name = name[:8] + "..." if len(name) > 8 else name
                # Pad display name to ensure consistent width
                display_name = f"{display_name:<11}"

                # Show truncated name with tooltip
                imgui.text(display_name)
                if imgui.is_item_hovered() and len(name) > 8:
                    imgui.set_tooltip(name)
                imgui.same_line()

                # Use slider for numeric values with fixed width
                imgui.push_item_width(150)
                slider_id = f"##{attribute_name}_{i}"
                changed, new_val = imgui.slider_float(slider_id, current_sliders[i], slider_min, slider_max, "%.6f")
                imgui.pop_item_width()
                # if changed:
                #     current_sliders[i] = new_val

            else:
                # For non-numeric values, just show as text
                imgui.text(f"{name}: {val}")
