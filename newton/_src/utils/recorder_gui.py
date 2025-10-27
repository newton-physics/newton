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


import warp as wp
from warp.render.imgui_manager import ImGuiManager


class RecorderImGuiManager(ImGuiManager):
    """
    An ImGui manager for controlling simulation playback with a recorder.

    This class provides a graphical user interface for controlling simulation playback,
    including pausing, resuming, scrubbing through frames, and saving/loading recordings.
    It also manages the visualization of contact points and updates the viewer
    according to the selected frame.
    """

    def __init__(self, viewer, recorder, example, window_pos=(10, 10), window_size=(300, 120)):
        """
        Initialize the RecorderImGuiManager.

        Args:
            viewer: The viewer instance (must have a renderer attribute).
            recorder: The recorder object that stores simulation history.
            example: The simulation example object (must have .paused and .frame_dt attributes).
            window_pos (tuple, optional): The (x, y) position of the ImGui window. Defaults to (10, 10).
            window_size (tuple, optional): The (width, height) of the ImGui window. Defaults to (300, 120).
        """
        super().__init__(viewer.renderer)
        self.viewer = viewer
        if not self.is_available:
            return

        self.window_pos = window_pos
        self.window_size = window_size
        self.recorder = recorder
        self.example = example
        self.selected_frame = 0
        self.num_point_clouds_rendered = 0

    def _clear_contact_points(self):
        """
        Clears all rendered contact points from the viewer.

        This method removes all previously rendered contact point clouds by rendering
        empty point clouds in their place.
        """
        for i in range(self.num_point_clouds_rendered):
            # use size 1 as size 0 seems to do nothing
            self.viewer.renderer.render_points(f"contact_points{i}", wp.empty(1, dtype=wp.vec3), radius=1e-2)
        self.num_point_clouds_rendered = 0

    def _update_frame(self, frame_id):
        """
        Update the selected frame and renderer transforms if paused.

        Args:
            frame_id (int): The frame index to display.
        """
        self.selected_frame = frame_id
        if self.example.paused:
            transforms, point_clouds = self.recorder.playback(self.selected_frame)
            if transforms:
                self.viewer.renderer.update_body_transforms(transforms)

            self._clear_contact_points()
            if point_clouds:
                for i, pc in enumerate(point_clouds):
                    self.viewer.renderer.render_points(
                        f"contact_points{i}", pc, radius=1e-2, colors=self.viewer.renderer.get_new_color(i)
                    )
                self.num_point_clouds_rendered = len(point_clouds)

    def draw_ui(self):
        """
        Draw the ImGui user interface for controlling playback and managing recordings.

        This includes controls for pausing/resuming, frame navigation, saving, and loading.
        """
        total_frames = len(self.recorder.transforms_history)
        if not self.example.paused and total_frames > 0:
            self.selected_frame = total_frames - 1

        self.imgui.set_next_window_size(self.window_size[0], self.window_size[1], self.imgui.ONCE)
        self.imgui.set_next_window_position(self.window_pos[0], self.window_pos[1], self.imgui.ONCE)

        self.imgui.begin("Recorder Controls")

        # Start/Stop button
        if self.example.paused:
            if self.imgui.button("Resume"):
                self.example.paused = False
        else:
            if self.imgui.button("Pause"):
                self.example.paused = True

        self.imgui.same_line()
        # total frames
        frame_time = self.selected_frame * self.example.frame_dt
        self.imgui.text(
            f"Frame: {self.selected_frame}/{total_frames - 1 if total_frames > 0 else 0} ({frame_time:.2f}s)"
        )

        # Frame slider
        if total_frames > 0:
            changed, self.selected_frame = self.imgui.slider_int("Timeline", self.selected_frame, 0, total_frames - 1)
            if changed and self.example.paused:
                self._update_frame(self.selected_frame)

            # Back/Forward buttons
            if self.imgui.button(" < "):
                self._update_frame(max(0, self.selected_frame - 1))

            self.imgui.same_line()

            if self.imgui.button(" > "):
                self._update_frame(min(total_frames - 1, self.selected_frame + 1))

        self.imgui.separator()

        if self.imgui.button("Save"):
            file_path = self.open_save_file_dialog(
                defaultextension=".npz",
                filetypes=[("Numpy Archives", "*.npz"), ("All files", "*.*")],
                title="Save Recording",
            )
            if file_path:
                self.recorder.save_to_file(file_path)

        self.imgui.same_line()

        if self.imgui.button("Load"):
            file_path = self.open_load_file_dialog(
                filetypes=[("Numpy Archives", "*.npz"), ("All files", "*.*")],
                title="Load Recording",
            )
            if file_path:
                self.recorder.load_from_file(file_path, device=wp.get_device())
                # When loading, pause the simulation and go to the first frame
                self.example.paused = True
                self.selected_frame = 0
                if len(self.recorder.transforms_history) > 0:
                    self._update_frame(self.selected_frame)

        self.imgui.end()
