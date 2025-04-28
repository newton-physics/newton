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

import warp as wp


class Contact:
    """Provides contact information to be consumed by a solver.
    Stores the contact distance, position, frame, and geometry for each contact point.
    """

    def __init__(self):
        self.dist: wp.array(dtype=wp.float32) | None = None
        self.pos: wp.array(dtype=wp.vec3f) | None = None
        self.frame: wp.array(dtype=wp.mat33f) | None = None
        self.dim: wp.array(dtype=wp.int32) | None = None
        self.geom: wp.array(dtype=wp.vec2i) | None = None
