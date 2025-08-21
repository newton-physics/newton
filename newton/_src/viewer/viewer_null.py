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

from .viewer import ViewerBase


class ViewerNull(ViewerBase):
    def __init__(self, num_frames=1000):
        """Provides a nop Viewer implementation that will just run for a set number of frames"""

        super().__init__()

        self.num_frames = num_frames
        self.frame_count = 0

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
        pass

    def log_instances(self, name, mesh, xforms, scales, colors, materials):
        pass

    def begin_frame(self, time):
        pass

    def end_frame(self):
        self.frame_count += 1

    def is_running(self) -> bool:
        return self.frame_count < self.num_frames

    def close(self):
        pass

    # Not implemented yet - placeholder methods from ViewerBase
    def log_lines(self, name, line_begins, line_ends, line_colors, hidden=False):
        pass

    def log_points(self, name, points, widths, colors, hidden=False):
        pass

    def log_array(self, name, array):
        pass

    def log_scalar(self, name, value):
        pass
