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

from ..core.types import override
from .viewer import ViewerBase


class ViewerNull(ViewerBase):
    """
    A no-operation (no-op) viewer implementation for Newton.

    This class provides a minimal, non-interactive viewer that does not perform any rendering
    or visualization. It is intended for use in headless or automated worlds where
    visualization is not required. The viewer runs for a fixed number of frames and provides
    stub implementations for all logging and frame management methods.
    """

    def __init__(self, num_frames=1000):
        """
        Initialize a no-op Viewer that runs for a fixed number of frames.

        Args:
            num_frames (int): The number of frames to run before stopping.
        """
        super().__init__()

        self.num_frames = num_frames
        self.frame_count = 0

    @override
    def log_mesh(
        self,
        name,
        points: wp.array,
        indices: wp.array,
        normals: wp.array | None = None,
        uvs: wp.array | None = None,
        hidden=False,
        backface_culling=True,
        face_varying_uv=False,
    ):
        """
        No-op implementation for logging a mesh.

        Args:
            name: Name of the mesh.
            points: Vertex positions.
            indices: Mesh indices.
            normals: Vertex normals (optional).
            uvs: Texture coordinates (optional).
            hidden: Whether the mesh is hidden.
            backface_culling: Whether to enable backface culling.
            face_varying_uv: Whether the mesh UVs are face-varying (optional).
        """
        pass

    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        """
        No-op implementation for logging mesh instances.

        Args:
            name: Name of the instance batch.
            mesh: Mesh object.
            xforms: Instance transforms.
            scales: Instance scales.
            colors: Instance colors.
            materials: Instance materials.
            hidden: Whether the instances are hidden.
        """
        pass

    def begin_frame(self, time):
        """
        No-op implementation for beginning a frame.

        Args:
            time: The current simulation time.
        """
        pass

    def end_frame(self):
        """
        Increment the frame count at the end of each frame.
        """
        self.frame_count += 1

    def is_running(self) -> bool:
        """
        Check if the viewer should continue running.

        Returns:
            bool: True if the frame count is less than the maximum number of frames.
        """
        return self.frame_count < self.num_frames

    def close(self):
        """
        No-op implementation for closing the viewer.
        """
        pass

    # Not implemented yet - placeholder methods from ViewerBase
    def log_lines(self, name, starts, ends, colors, width: float = 0.01, hidden=False):
        """
        No-op implementation for logging lines.

        Args:
            name: Name of the line batch.
            starts: Line start points.
            ends: Line end points.
            colors: Line colors.
            hidden: Whether the lines are hidden.
        """
        pass

    def log_points(self, name, points, radii, colors, width: float = 0.01, hidden=False):
        """
        No-op implementation for logging points.

        Args:
            name: Name of the point batch.
            points: Point positions.
            radii: Point radii.
            colors: Point colors.
            hidden: Whether the points are hidden.
        """
        pass

    def log_array(self, name, array):
        """
        No-op implementation for logging a generic array.

        Args:
            name: Name of the array.
            array: The array data.
        """
        pass

    def log_scalar(self, name, value):
        """
        No-op implementation for logging a scalar value.

        Args:
            name: Name of the scalar.
            value: The scalar value.
        """
        pass
