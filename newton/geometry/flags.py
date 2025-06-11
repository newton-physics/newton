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


# Particle flags
PARTICLE_FLAG_ACTIVE = wp.constant(wp.uint32(1 << 0))
"""Indicates that the particle is active."""

# Shape flags
SHAPE_FLAG_VISIBLE = wp.constant(wp.uint32(1 << 0))
"""Indicates that the shape is visible."""

SHAPE_FLAG_COLLIDE_SHAPES = wp.constant(wp.uint32(1 << 1))
"""Indicates that the shape collides with other shapes."""

SHAPE_FLAG_COLLIDE_PARTICLES = wp.constant(wp.uint32(1 << 2))
"""Indicates that the shape collides with particles."""

__all__ = [
    "PARTICLE_FLAG_ACTIVE",
    "SHAPE_FLAG_COLLIDE_PARTICLES",
    "SHAPE_FLAG_COLLIDE_SHAPES",
    "SHAPE_FLAG_VISIBLE",
]
