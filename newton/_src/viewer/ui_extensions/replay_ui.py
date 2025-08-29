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

import os

import newton

from ...utils.recorder import ModelAndStateRecorder


class ReplayUI:
    """
    A UI extension for ViewerGL that adds replay capabilities.

    This class can be added to any ViewerGL instance to provide:
    - Loading and replaying recorded data
    - Timeline scrubbing and playback controls

    Usage:
        viewer = newton.viewer.ViewerGL()
        replay_ui = ReplayUI()
        viewer.add_custom_ui(replay_ui.render, "ReplayControls")
    """

    def __init__(self):
        """Initialize the ReplayUI extension."""
        # Playback state
        self.current_frame = 0
        self.total_frames = 0

        # UI state
        self.selected_file = ""
        self.status_message = ""
        self.status_color = (1.0, 1.0, 1.0, 1.0)  # White by default

    def render(self, viewer):
        """
        Render the replay UI controls.

        Args:
            viewer: The ViewerGL instance this UI is attached to
        """
        if not viewer.ui.is_available:
            return

        imgui = viewer.ui.imgui
        io = viewer.ui.io

        # Position the replay controls window
        window_width = 400
        window_height = 350
        imgui.set_next_window_position(io.display_size[0] - window_width - 10, io.display_size[1] - window_height - 10)
        imgui.set_next_window_size(window_width, window_height)

        flags = imgui.WINDOW_NO_RESIZE

        if imgui.begin("Replay Controls", flags=flags):
            # Show status message if any
            if self.status_message:
                imgui.push_style_color(imgui.COLOR_TEXT, *self.status_color)
                imgui.text(self.status_message)
                imgui.pop_style_color()
                imgui.separator()

            self._render_playback_controls(viewer)

        imgui.end()

    def _render_playback_controls(self, viewer):
        """Render playback controls section."""
        imgui = viewer.ui.imgui

        # File loading
        imgui.text("Recording File:")
        imgui.text(self.selected_file if self.selected_file else "No file loaded")

        if imgui.button("Load Recording..."):
            file_path = viewer.ui.open_load_file_dialog(
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
                self._load_recording(file_path, viewer)

        # Playback controls (only if recording is loaded)
        if self.total_frames > 0:
            imgui.separator()
            imgui.text(f"Total frames: {self.total_frames}")

            # Frame slider
            changed, new_frame = imgui.slider_int("Frame", self.current_frame, 0, self.total_frames - 1)
            if changed:
                self.current_frame = new_frame
                self._load_frame(viewer)

            # Playback buttons
            if imgui.button("First"):
                self.current_frame = 0
                self._load_frame(viewer)

            imgui.same_line()
            if imgui.button("Prev") and self.current_frame > 0:
                self.current_frame -= 1
                self._load_frame(viewer)

            imgui.same_line()
            if imgui.button("Next") and self.current_frame < self.total_frames - 1:
                self.current_frame += 1
                self._load_frame(viewer)

            imgui.same_line()
            if imgui.button("Last"):
                self.current_frame = self.total_frames - 1
                self._load_frame(viewer)
        else:
            imgui.text("Load a recording to enable playback")

    def _clear_status(self):
        """Clear status messages."""
        self.status_message = ""
        self.status_color = (1.0, 1.0, 1.0, 1.0)

    def _load_recording(self, file_path, viewer):
        """Load a recording file for playback (same approach as example_replay_viewer.py)."""
        try:
            # Create a new recorder for playback
            playback_recorder = ModelAndStateRecorder()
            playback_recorder.load_from_file(file_path)

            self.total_frames = len(playback_recorder.history)
            self.selected_file = os.path.basename(file_path)

            # Create new model and state objects (like example_replay_viewer.py)
            if playback_recorder.deserialized_model:
                model = newton.Model()
                state = newton.State()

                # Restore the model from the recording
                playback_recorder.playback_model(model)

                # Set the model in the viewer (this will trigger setup)
                viewer.set_model(model)

                # Store the playback recorder
                self.playback_recorder = playback_recorder
                self.current_frame = 0

                # Restore the first frame's state (like example_replay_viewer.py)
                if len(playback_recorder.history) > 0:
                    playback_recorder.playback(state, 0)
                    viewer.log_state(state)

                self.status_message = f"Loaded {self.selected_file} ({self.total_frames} frames)"
                self.status_color = (0.3, 1.0, 0.3, 1.0)  # Green
            else:
                self.status_message = "Warning: No model data found in recording"
                self.status_color = (1.0, 1.0, 0.3, 1.0)  # Yellow

        except FileNotFoundError:
            self.status_message = f"File not found: {file_path}"
            self.status_color = (1.0, 0.3, 0.3, 1.0)  # Red
        except Exception as e:
            self.status_message = f"Error loading recording: {str(e)[:50]}..."
            self.status_color = (1.0, 0.3, 0.3, 1.0)  # Red

    def _load_frame(self, viewer):
        """Load a specific frame for display."""
        if hasattr(self, "playback_recorder") and 0 <= self.current_frame < self.total_frames:
            state = newton.State()
            self.playback_recorder.playback(state, self.current_frame)
            viewer.log_state(state)
