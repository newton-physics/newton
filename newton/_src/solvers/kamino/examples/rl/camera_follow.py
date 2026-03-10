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


class CameraFollowRobot:
    """Smoothly follow the robot root body with the viewer camera.

    The camera maintains a fixed offset from the robot position and uses
    exponential smoothing to avoid jerky motion.  Call :meth:`update` once
    per render frame.

    Args:
        viewer: Newton viewer instance (must support ``set_camera``).
        camera_offset: ``(x, y, z)`` offset from robot root to camera position.
        pitch: Camera pitch angle in degrees.
        yaw: Camera yaw angle in degrees.
        filter_coeff: Exponential smoothing factor in ``(0, 1]``.
            Smaller = smoother/slower tracking, larger = snappier.
    """

    def __init__(
        self,
        viewer,
        camera_offset: tuple[float, float, float] = (1.5, 1.5, 0.5),
        pitch: float = -10.0,
        yaw: float = 225.0,
        filter_coeff: float = 0.1,
    ):
        self._viewer = viewer
        self._offset = np.array(camera_offset, dtype=np.float32)
        self._pitch = pitch
        self._yaw = yaw
        self._filter = filter_coeff
        self._cam_pos: np.ndarray | None = None

    def update(self, root_pos: np.ndarray):
        """Update camera to follow the given root position.

        Args:
            root_pos: Robot root position as ``(3,)`` numpy array or similar.
        """
        target = np.asarray(root_pos, dtype=np.float32).ravel()[:3]
        desired = target + self._offset

        if self._cam_pos is None:
            self._cam_pos = desired.copy()
        else:
            self._cam_pos += self._filter * (desired - self._cam_pos)

        self._viewer.set_camera(wp.vec3(*self._cam_pos.tolist()), self._pitch, self._yaw)

    def set_offset(self, offset: tuple[float, float, float]):
        """Change the camera offset from the robot."""
        self._offset = np.array(offset, dtype=np.float32)

    def set_pitch(self, pitch: float):
        """Change the camera pitch angle in degrees."""
        self._pitch = pitch

    def set_yaw(self, yaw: float):
        """Change the camera yaw angle in degrees."""
        self._yaw = yaw

    def reset(self):
        """Reset smoothing state (e.g. after a simulation reset)."""
        self._cam_pos = None
