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

import math
import time

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import warp as wp

from newton.sim import Model, State

########################################################################################################################
###############################################    Polyscope Renderer    ###############################################
########################################################################################################################


class PolyscopeRenderer:
    def __init__(
        self,
        model: Model = None,
        window_size: tuple[int, int] = (1920, 1080),
        vsync=False,
    ):
        """Initialize a 3D renderer with customizable window properties.
        Args:
            model (newton.Model): The Newton physics model to render
            window_size (Tuple[int, int]): Window dimensions (width, height)
            vsync (bool): Enable vertical synchronization (default: False)
        """
        self.vsync = vsync
        self.model = model
        self.paused = True
        self.sim_time = 0.0
        self.sim_frames = 0
        self._tri_mesh = None
        self._tri_indices = None
        self._coord_axes = None
        self.user_update = None
        self._pick_result = None
        self._body_meshes = None
        self.ground_plane_mode = "tile_reflection"

        # FPS counting
        self._sim_fps = 0.0
        self._render_fps = 1.0
        self._redner_fps_count = 0
        self._render_fps_last_time = 0.0
        self._sim_fps_time_cost = []

        # Drag info
        self._drag_dist = 0.0
        self.drag_index = -1
        self.drag_position = wp.vec3(0, 0, 0)
        self.drag_bary_coord = wp.vec3(0, 0, 0)
        self.drag_info_chg = False

        # Setup camera
        self._last_mouse_pos = (0, 0)
        self._camera_origin = [0, 1, 0]
        self._camera_radius = 2.0
        self._camera_theta = 80.0
        self._camera_phi = 0.0

        # Setup polyscope scene parameters
        ps.init()
        ps.set_SSAA_factor(4)
        ps.set_enable_vsync(vsync)
        ps.set_ground_plane_height(0)
        ps.set_ground_plane_mode(self.ground_plane_mode)
        ps.set_automatically_compute_scene_extents(False)
        ps.set_window_size(window_size[0], window_size[1])
        ps.set_background_color((0.015, 0.015, 0.015))
        ps.set_do_default_mouse_interaction(False)
        ps.set_user_callback(self._update)
        ps.set_max_fps(200)

        # Inner group (hide children)
        self._inner_group = ps.create_group("Inner")
        self._inner_group.set_show_child_details(False)
        self._inner_group.set_hide_descendants_from_structure_lists(True)

        # Add coordinate axes
        self._set_up_coord_axes()

        # Add look-at point
        self._look_at_point = ps.register_point_cloud(
            name="Look-At",
            points=np.array([0.0, 0.0, 0.0]).reshape(-1, 3),
            color=(1, 0.1, 0.1),
            material="candy",
            enabled=False,
        )
        self._look_at_point.add_to_group(self._inner_group)

        # Add drag point
        self._drag_point = ps.register_point_cloud(
            name="Ray-hit",
            points=np.array([0.0, 0.0, 0.0]).reshape(-1, 3),
            color=(72 / 255.0, 167 / 255.0, 1),
            material="flat",
            enabled=False,
        )
        self._drag_point.add_to_group(self._inner_group)

        # Init camera
        self._update_camera()

        # Add meshes
        if model is not None:
            self._tri_indices = model.tri_indices.numpy()
            self._tri_mesh = ps.register_surface_mesh(
                name="Triangles",
                vertices=model.particle_q.numpy().reshape(model.particle_count, 3),
                faces=model.tri_indices.numpy().reshape(model.tri_count, 3),
                color=(184 / 255.0, 67 / 255.0, 1),
                back_face_policy="custom",
                edge_color=(0, 0, 0),
                smooth_shade=False,
                edge_width=0.3,
            )
            self._tri_mesh.set_selection_mode("faces_only")

            for i in range(model.body_count):
                shape_indices = model.body_shapes[i]
                for shape_idx in shape_indices:
                    ps.register_surface_mesh(
                        name=model.shape_key[shape_idx],
                        vertices=model.shape_geo_src[shape_idx].vertices,
                        faces=model.shape_geo_src[shape_idx].indices.reshape(-1, 3),
                        back_face_policy="cull",
                        edge_color=(1, 1, 1),
                        smooth_shade=False,
                        edge_width=0.2,
                        color=(0, 0, 0),
                        material="wax",
                    )

    def set_user_update(self, callback):
        self.user_update = callback

    def update_state(self, state: State):
        if self._tri_mesh is not None:
            vertices = state.particle_q.numpy().reshape(state.particle_count, 3)
            self._tri_mesh.update_vertex_positions(vertices)

            if self._pick_result is not None:
                if self._pick_result.structure_name == self._tri_mesh.get_name():
                    bary_coord = self.drag_bary_coord
                    index = self._pick_result.structure_data["index"]
                    face = self._tri_indices[index, 0:3]
                    x0 = wp.vec3(vertices[face[0], 0:3])
                    x1 = wp.vec3(vertices[face[1], 0:3])
                    x2 = wp.vec3(vertices[face[2], 0:3])
                    self._drag_point.set_position(x0 * bary_coord[0] + x1 * bary_coord[1] + x2 * bary_coord[2])

            ps.request_redraw()

    def _set_up_coord_axes(self, scale: float = 0.2, radius: float = 3e-3):
        edges = np.array([[0, 1], [2, 3], [4, 5]])
        nodes = np.array([[0, 0, 0], [scale, 0, 0], [0, 0, 0], [0, scale, 0], [0, 0, 0], [0, 0, scale]])
        colors = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
        axes = ps.register_curve_network("Axes", nodes, edges, True, radius)
        axes.add_color_quantity("Axes color", colors, "nodes", enabled=True)
        axes.add_to_group(self._inner_group)
        self._coord_axes = axes

    def _process_key_inputs(self):
        if psim.IsKeyPressed(psim.ImGuiKey_Space):
            self.paused = not self.paused  # Run/pause
        elif psim.IsKeyPressed(psim.ImGuiKey_Escape):
            ps.unshow()  # Exit
        elif psim.IsKeyPressed(psim.ImGuiKey_X):  # Show/hide edges
            if self._body_meshes is not None:
                for mesh in self._body_meshes:
                    mesh.set_edge_width(0 if mesh.get_edge_width() != 0 else 0.3)
            if self._tri_mesh is not None:
                self._tri_mesh.set_edge_width(0 if self._tri_mesh.get_edge_width() != 0 else 0.3)
        elif psim.IsKeyPressed(psim.ImGuiKey_C):
            # Show/hide coordinate axes
            self._coord_axes.set_enabled(not self._coord_axes.is_enabled())
        elif psim.IsKeyPressed(psim.ImGuiKey_V):
            self.vsync = not self.vsync
            ps.set_enable_vsync(self.vsync)
        elif psim.IsKeyPressed(psim.ImGuiKey_G):
            # Rolling ground plane mode
            if self.ground_plane_mode == "none":
                self.ground_plane_mode = "tile"
            elif self.ground_plane_mode == "tile":
                self.ground_plane_mode = "tile_reflection"
            elif self.ground_plane_mode == "tile_reflection":
                self.ground_plane_mode = "none"
            ps.set_ground_plane_mode(self.ground_plane_mode)

    def _process_mouse_inputs(self):
        # Mouse delta Pos
        mouse_pos = psim.GetMousePos()
        dx = mouse_pos[0] - self._last_mouse_pos[0]
        dy = mouse_pos[1] - self._last_mouse_pos[1]
        self._last_mouse_pos = mouse_pos

        # Button key
        LeftButton, RightButton, MiddleButton = 0, 1, 2

        # Click evnet
        if psim.IsMouseClicked(LeftButton):
            pick_result = ps.pick(screen_coords=mouse_pos)
            if pick_result.is_hit:
                self._pick_result = pick_result
                self._drag_point.set_enabled(True)
                self._drag_point.set_position(pick_result.position)
                if (pick_result.structure_name == self._tri_mesh.get_name()) and (
                    self._pick_result.structure_data["element_type"] == "face"
                ):
                    self.drag_index = pick_result.structure_data["index"]
                    self.drag_bary_coord = wp.vec3(pick_result.structure_data["bary_coords"])
                    self.drag_position = wp.vec3(pick_result.position)
                    eye_pos = wp.vec3(ps.get_view_camera_parameters().get_position())
                    self._drag_dist = wp.length(self.drag_position - eye_pos)
            else:
                self._drag_point.set_enabled(False)
                self._pick_result = None
                self.drag_index = -1
            self.drag_info_chg = True
        if psim.IsMouseClicked(MiddleButton):
            pass
        if psim.IsMouseClicked(RightButton):
            pass

        # Show/hide look-at point
        self._look_at_point.set_enabled(psim.IsMouseDown(MiddleButton) or psim.IsMouseDown(RightButton))

        # Dragging
        if (self.drag_index != -1) and psim.IsMouseDown(LeftButton) and ((dx != 0) or (dy != 0)):
            camera_params = ps.get_view_camera_parameters()
            aspect = camera_params.get_aspect()
            view_mat = camera_params.get_view_mat()
            fov = wp.radians(camera_params.get_fov_vertical_deg())
            axis_x = wp.vec3(view_mat[0, 0], view_mat[0, 1], view_mat[0, 2])
            axis_y = wp.vec3(view_mat[1, 0], view_mat[1, 1], view_mat[1, 2])
            axis_z = wp.vec3(view_mat[2, 0], view_mat[2, 1], view_mat[2, 2])
            origin = wp.vec3(camera_params.get_position())
            (width, height) = ps.get_window_size()
            nx = 2.0 * (mouse_pos[0] + 0.5) / width - 1.0
            ny = 2.0 * (mouse_pos[1] + 0.5) / height - 1.0
            u = nx * wp.tan(fov / 2.0) * aspect
            v = ny * wp.tan(fov / 2.0)
            ray_dir = wp.normalize(axis_x * u - axis_y * v - axis_z)
            self.drag_position = origin + ray_dir * self._drag_dist
            self.drag_info_chg = True
        elif not psim.IsMouseDown(LeftButton):
            self.drag_info_chg = True
            self.drag_index = -1

        should_update_camera = False

        # Rotate camera
        if psim.IsMouseDown(RightButton) and ((dx != 0) or (dy != 0)):
            self._camera_phi -= dx / 2.0
            self._camera_theta -= dy / 4.0
            self._camera_theta = wp.clamp(self._camera_theta, 1.0, 179.0)
            should_update_camera = True

        # Translate camera
        if psim.IsMouseDown(MiddleButton) and ((dx != 0) or (dy != 0)):
            camera_params = ps.get_view_camera_parameters()
            fov = camera_params.get_fov_vertical_deg()
            window_height = ps.get_window_size()[1]
            up_dir = camera_params.get_up_dir()
            right_dir = camera_params.get_right_dir()
            delta = up_dir * dy * 2.0 / window_height
            delta -= right_dir * dx * 2.0 / window_height
            delta *= wp.tan(wp.radians(fov) / 2.0)
            delta *= self._camera_radius
            delta[0] += self._camera_origin[0]
            delta[1] += self._camera_origin[1]
            delta[2] += self._camera_origin[2]
            self._camera_origin = (delta[0], delta[1], delta[2])
            should_update_camera = True

        # Zoom camera
        io = psim.GetIO()
        if io.MouseWheel != 0.0:
            ratio = 0.9 if (io.MouseWheel < 0.0) else (1.0 / 0.9)
            self._camera_radius = wp.clamp(self._camera_radius * ratio, 2e-2, 1e2)
            should_update_camera = True

        if should_update_camera:
            self._update_camera()

    def _update_camera(self):
        r = wp.sin(wp.radians(self._camera_theta))
        x = r * wp.sin(wp.radians(self._camera_phi))
        z = r * wp.cos(wp.radians(self._camera_phi))
        y = wp.cos(wp.radians(self._camera_theta))
        x = x * self._camera_radius + self._camera_origin[0]
        y = y * self._camera_radius + self._camera_origin[1]
        z = z * self._camera_radius + self._camera_origin[2]
        ps.look_at_dir(camera_location=(x, y, z), target=self._camera_origin, up_dir=(0, 1, 0))
        self._drag_point.set_radius(self._camera_radius * 2e-3, False)
        self._look_at_point.set_radius(self._camera_radius * 5e-3, False)
        self._look_at_point.set_position(self._camera_origin)

    def _update_gui(self):
        # update render fps
        curr_time = time.time()
        if (self._redner_fps_count > 0) and (curr_time - self._render_fps_last_time > 0.1):
            self._render_fps = self._redner_fps_count / (curr_time - self._render_fps_last_time)
            self._render_fps_last_time = curr_time
            self._redner_fps_count = 0
        self._redner_fps_count += 1

        psim.Text("State: ")
        psim.SameLine()
        if self.paused:
            psim.TextColored([1, 0, 0, 1], "Paused")
        else:
            psim.TextColored([0, 1, 0, 1], "Running")
        psim.Text(f"Sim Time: {self.sim_time:.1f} s")
        psim.Text(f"Frame Count: {self.sim_frames}")
        if self._sim_fps != 0.0:
            psim.Text(f"Update FPS: {self._sim_fps:.1f} / {1e3 / self._sim_fps:.1f}ms")
        else:
            psim.Text("Update FPS: 0.0 / Inf.ms")
        psim.Text(f"Render FPS: {self._render_fps:.1f} / {1e3 / self._render_fps:.1f}ms")

        if self.model is not None:
            psim.Separator()
            psim.Text("Statistics:")
            psim.Text(f" - Body Count: {self.model.body_count}")
            psim.Text(f" - Particle Count: {self.model.particle_count}")
            psim.Text(f" - Triangle Count: {self.model.triangle_count}")
            psim.Text(f" - Tetrahedra Count: {self.model.tetrahedra_count}")

        if self._pick_result is not None:
            psim.Separator()
            psim.Text("Pick Result:")
            psim.Text(f" - Structure Name: {self._pick_result.structure_name}")
            psim.Text(f" - Structure Type: {self._pick_result.structure_type_name}")
            psim.Text(f" - Screen Coordinate: {self._pick_result.screen_coords}")
            psim.Text(f" - World Position: {self._pick_result.position}")
            if self._pick_result.structure_type_name != "Point Cloud":
                if self._pick_result.structure_data["element_type"] == "face":
                    psim.Text(f" - Bary Coordinate: {self._pick_result.structure_data['bary_coords']}")
                psim.Text(f" - Element Type: {self._pick_result.structure_data['element_type']}")
                psim.Text(f" - Index: {self._pick_result.structure_data['index']}")

    def _update(self):
        self._update_gui()
        self._process_key_inputs()
        self._process_mouse_inputs()
        if self.user_update is not None:
            if not self.paused:
                time_begin = time.time()
                self.user_update()
                time_end = time.time()
                # update simulation fps
                self._sim_fps_time_cost.append(time_end - time_begin)
                if len(self._sim_fps_time_cost) > 5:
                    self._sim_fps_time_cost.pop(0)
                    self._sim_fps = len(self._sim_fps_time_cost) / math.fsum(self._sim_fps_time_cost)

    def run(self):
        ps.show()


if __name__ == "__main__":
    renderer = PolyscopeRenderer()
    renderer.run()
