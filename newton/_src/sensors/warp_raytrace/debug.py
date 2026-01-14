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

import numpy as np
import warp as wp

from .render_context import RenderContext
from .types import RenderLightType


class RenderDebugUtils:
    @staticmethod
    def assign_random_colors_per_world(render_context: RenderContext, seed: int = 100):
        colors = np.random.default_rng(seed).random((render_context.num_shapes_total, 4)) * 0.5 + 0.5
        colors[:, -1] = 1.0
        render_context.shape_colors = wp.array(
            colors[render_context.shape_world_index.numpy() % len(colors)], dtype=wp.vec4f
        )

    @staticmethod
    def assign_random_colors_per_shape(render_context: RenderContext, seed: int = 100):
        colors = np.random.default_rng(seed).random((render_context.num_shapes_total, 4)) * 0.5 + 0.5
        colors[:, -1] = 1.0
        render_context.shape_colors = wp.array(colors, dtype=wp.vec4f)

    @staticmethod
    def create_default_light(
        render_context: RenderContext, enable_shadows: bool = True, direction: wp.vec3f | None = None
    ):
        render_context.enable_shadows = enable_shadows
        render_context.lights_active = wp.array([True], dtype=wp.bool)
        render_context.lights_type = wp.array([RenderLightType.DIRECTIONAL], dtype=wp.int32)
        render_context.lights_cast_shadow = wp.array([True], dtype=wp.bool)
        render_context.lights_position = wp.array([wp.vec3f(0.0)], dtype=wp.vec3f)
        render_context.lights_orientation = wp.array(
            [direction if direction is not None else wp.vec3f(-0.57735026, 0.57735026, -0.57735026)], dtype=wp.vec3f
        )

    @staticmethod
    def assign_checkerboard_material_to_all_shapes(
        render_context: RenderContext, resolution: int = 64, checker_size: int = 32
    ):
        checkerboard = (
            (np.arange(resolution) // checker_size)[:, None] + (np.arange(resolution) // checker_size)
        ) % 2 == 0
        pixels = np.where(checkerboard, 0xFF808080, 0xFFBFBFBF).astype(np.uint32).flatten()

        render_context.enable_textures = True
        render_context.texture_data = wp.array(pixels, dtype=wp.uint32)
        render_context.texture_offsets = wp.array([0], dtype=wp.int32)
        render_context.texture_width = wp.array([resolution], dtype=wp.int32)
        render_context.texture_height = wp.array([resolution], dtype=wp.int32)

        render_context.material_texture_ids = wp.array([0], dtype=wp.int32)
        render_context.material_texture_repeat = wp.array([wp.vec2f(1.0)], dtype=wp.vec2f)
        render_context.material_rgba = wp.array([wp.vec4f(1.0)], dtype=wp.vec4f)

        render_context.shape_materials = wp.array(
            np.full(render_context.num_shapes_total, fill_value=0, dtype=np.int32), dtype=wp.int32
        )
