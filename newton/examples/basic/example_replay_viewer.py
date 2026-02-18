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

###########################################################################
# Example Integrated Viewer
#
# Shows how to use the replay UI with ViewerGL to load and
# display previously recorded simulation data.
#
# Recording is done automatically using ViewerFile (like ViewerUSD):
#   viewer = newton.viewer.ViewerFile("my_recording.json")
#   viewer.set_model(model)
#   viewer.log_state(state)  # Records automatically
#   viewer.close()  # Saves automatically
#
# Command: python -m newton.examples.example_replay_viewer
#
###########################################################################

import os
import traceback

import newton
import newton.examples


class ReplayUI:
    """
    A UI extension for ViewerGL that adds replay capabilities.

    This class can be added to any ViewerGL instance to provide:
    - Loading and replaying recorded data
    - Timeline scrubbing and playback controls

    Usage:
        viewer = newton.viewer.ViewerGL()
        replay_ui = ReplayUI(viewer)
        viewer.register_ui_callback(replay_ui.render, "free")
    """

    def __init__(self, viewer=None):
        """Initialize the ReplayUI extension.

        Args:
            viewer: The ViewerGL instance this UI will be attached to (optional)
        """
        # Store reference to viewer for accessing viewer functionality
        self.viewer = viewer

        # Playback state
        self.current_frame = 0
        self.total_frames = 0

        # UI state
        self.selected_file = ""
        self.status_message = ""
        self.status_color = (1.0, 1.0, 1.0, 1.0)  # White by default

    def set_viewer(self, viewer):
        """Set the viewer reference after initialization."""
        self.viewer = viewer

    def render(self, imgui):
        """
        Render the replay UI controls.

        Args:
            imgui: The ImGui object passed by the ViewerGL callback system
        """
        if not self.viewer or not self.viewer.ui.is_available:
            return

        io = self.viewer.ui.io

        # Position the replay controls window
        window_width = 400
        window_height = 350
        imgui.set_next_window_pos(
            imgui.ImVec2(io.display_size[0] - window_width - 10, io.display_size[1] - window_height - 10)
        )
        imgui.set_next_window_size(imgui.ImVec2(window_width, window_height))

        flags = imgui.WindowFlags_.no_resize.value

        if imgui.begin("Replay Controls", flags=flags):
            # Show status message if any
            if self.status_message:
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*self.status_color))
                imgui.text(self.status_message)
                imgui.pop_style_color()
                imgui.separator()

            self._render_playback_controls(imgui)

        imgui.end()

    def _render_playback_controls(self, imgui):
        """Render playback controls section."""

        # File loading
        imgui.text("Recording File:")
        imgui.text(self.selected_file if self.selected_file else "No file loaded")

        if imgui.button("Load Recording..."):
            file_path = self.viewer.ui.open_load_file_dialog(
                filetypes=[
                    ("Recording files", ("*.json", "*.bin")),
                    ("JSON files", "*.json"),
                    ("Binary files", "*.bin"),
                    ("All files", "*.*"),
                ],
                title="Select Recording File",
            )
            if file_path:
                self._clear_status()
                self._load_recording(file_path)

        # Playback controls (only if recording is loaded)
        if self.total_frames > 0:
            imgui.separator()
            imgui.text(f"Total frames: {self.total_frames}")

            # Frame slider
            changed, new_frame = imgui.slider_int("Frame", self.current_frame, 0, self.total_frames - 1)
            if changed:
                self.current_frame = new_frame
                self._load_frame()

            # Playback buttons
            if imgui.button("First"):
                self.current_frame = 0
                self._load_frame()

            imgui.same_line()
            if imgui.button("Prev") and self.current_frame > 0:
                self.current_frame -= 1
                self._load_frame()

            imgui.same_line()
            if imgui.button("Next") and self.current_frame < self.total_frames - 1:
                self.current_frame += 1
                self._load_frame()

            imgui.same_line()
            if imgui.button("Last"):
                self.current_frame = self.total_frames - 1
                self._load_frame()
        else:
            imgui.text("Load a recording to enable playback")

    def _clear_status(self):
        """Clear status messages."""
        self.status_message = ""
        self.status_color = (1.0, 1.0, 1.0, 1.0)

    def _load_recording(self, file_path):
        """Load a recording file for playback using ViewerFile."""
        try:
            viewer_file = newton.viewer.ViewerFile(file_path)
            viewer_file.load_recording()

            self.total_frames = viewer_file.get_frame_count()
            self.selected_file = os.path.basename(file_path)

            if viewer_file.has_model() and self.total_frames > 0:
                model = newton.Model()
                viewer_file.load_model(model)

                self.viewer.set_model(model)
                self._viewer_file = viewer_file
                self.current_frame = 0

                state = model.state()
                viewer_file.load_state(state, 0)
                self.viewer.log_state(state)

                self.status_message = f"Loaded {self.selected_file} ({self.total_frames} frames)"
                self.status_color = (0.3, 1.0, 0.3, 1.0)  # Green
            else:
                self.status_message = "Warning: No model data or frames found in recording"
                self.status_color = (1.0, 1.0, 0.3, 1.0)  # Yellow

        except FileNotFoundError:
            self.status_message = f"File not found: {file_path}"
            self.status_color = (1.0, 0.3, 0.3, 1.0)  # Red
            print(f"[ReplayUI] File not found: {file_path}")
        except Exception as e:
            self.status_message = f"Error loading recording: {str(e)[:50]}..."
            self.status_color = (1.0, 0.3, 0.3, 1.0)  # Red
            print(f"[ReplayUI] Error loading recording: {file_path}")
            print(f"[ReplayUI] Full error: {e}")
            traceback.print_exc()

    def _load_frame(self):
        """Load a specific frame for display."""
        if hasattr(self, "_viewer_file") and 0 <= self.current_frame < self.total_frames:
            state = self.viewer.model.state()
            self._viewer_file.load_state(state, self.current_frame)
            self.viewer.log_state(state)


class Example:
    def __init__(self, viewer):
        """Initialize the integrated viewer example with replay UI."""
        self.viewer = viewer

        # Add replay UI extension to the viewer
        self.replay_ui = ReplayUI(viewer)
        self.viewer.register_ui_callback(self.replay_ui.render, "free")

        # No simulation - this example is purely for replay
        self.sim_time = 0.0

    def step(self):
        """No simulation step needed - replay is handled by UI."""
        pass

    def render(self):
        """Render the current state (managed by replay UI)."""
        self.viewer.begin_frame(self.sim_time)
        # Current state is logged by the replay UI when frames are loaded
        # No need to call viewer.log_state() here
        self.viewer.end_frame()

    def test_final(self):
        pass


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create example and run
    example = Example(viewer)

    newton.examples.run(example, args)
