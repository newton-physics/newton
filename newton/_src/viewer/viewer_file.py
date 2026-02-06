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

from pathlib import Path

import warp as wp

from ..core.types import override
from ..utils.recorder import RecorderModelAndState
from .viewer import ViewerBase


class ViewerFile(ViewerBase):
    """
    File-based viewer backend for Newton physics simulations.

    This backend records simulation data to JSON or binary files using the same
    ViewerBase API as other viewers. It captures model structure and state data
    during simulation for later replay or analysis.

    Format is determined by file extension:
    - .json: Human-readable JSON format
    - .bin: Binary CBOR2 format (more efficient)
    """

    def __init__(self, output_path: str, auto_save: bool = True, save_interval: int = 100):
        """
        Initialize the File viewer backend for Newton physics simulations.

        Args:
            output_path (str): Path to the output file (.json or .bin)
            auto_save (bool): If True, automatically save periodically during recording
            save_interval (int): Number of frames between auto-saves (when auto_save=True)
        """
        super().__init__()

        self.output_path = Path(output_path)
        self.auto_save = auto_save
        self.save_interval = save_interval

        # Initialize the recorder
        self._recorder = RecorderModelAndState()
        self._frame_count = 0
        self._model_recorded = False

    @override
    def set_model(self, model, max_worlds: int | None = None):
        """Override set_model to record the model when it's set."""
        super().set_model(model, max_worlds=max_worlds)

        if model is not None and not self._model_recorded:
            self._recorder.record_model(model)
            self._model_recorded = True

    @override
    def log_state(self, state):
        """Override log_state to record the state in addition to standard processing."""
        super().log_state(state)

        # Record the state
        self._recorder.record(state)
        self._frame_count += 1

        # Auto-save if enabled
        if self.auto_save and self._frame_count % self.save_interval == 0:
            self._save_recording()

    def save_recording(self):
        """Save the recorded data to file."""
        self._save_recording()

    def _save_recording(self):
        """Internal method to save recording."""
        try:
            self._recorder.save_to_file(str(self.output_path))
            print(f"Recording saved to {self.output_path} ({self._frame_count} frames)")
        except Exception as e:
            print(f"Error saving recording: {e}")

    # Abstract method implementations (no-ops for file recording)

    @override
    def log_mesh(
        self,
        name,
        points: wp.array,
        indices: wp.array,
        normals: wp.array | None = None,
        uvs: wp.array | None = None,
        texture=None,
        hidden=False,
        backface_culling=True,
    ):
        """File viewer doesn't render meshes, so this is a no-op."""
        pass

    @override
    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        """File viewer doesn't render instances, so this is a no-op."""
        pass

    @override
    def log_lines(self, name, starts, ends, colors, width: float = 0.01, hidden=False):
        """File viewer doesn't render lines, so this is a no-op."""
        pass

    @override
    def log_points(self, name, points, radii, colors, hidden=False):
        """File viewer doesn't render points, so this is a no-op."""
        pass

    @override
    def log_array(self, name, array):
        """File viewer doesn't log arrays visually, so this is a no-op."""
        pass

    @override
    def log_scalar(self, name, value):
        """File viewer doesn't log scalars visually, so this is a no-op."""
        pass

    @override
    def end_frame(self):
        """No frame rendering needed for file viewer."""
        pass

    @override
    def close(self):
        """Save final recording and cleanup."""
        if self._frame_count > 0:
            self._save_recording()
        print(f"ViewerFile closed. Total frames recorded: {self._frame_count}")

    def load_recording(self, file_path: str):
        """Load a previously recorded file."""
        self._recorder.load_from_file(file_path)
        self._frame_count = len(self._recorder.history)
        print(f"Loaded recording with {self._frame_count} frames from {file_path}")

    def get_frame_count(self) -> int:
        """Get the number of recorded frames."""
        return self._frame_count

    def get_recorder(self) -> RecorderModelAndState:
        """Get access to the underlying recorder for advanced operations."""
        return self._recorder
