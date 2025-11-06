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

from .bvh import compute_bvh_bounds, compute_bvh_group_roots
from .render import render_megakernel


class RenderContext:
    class Output:
        def __init__(self, num_worlds: int, num_cameras: int, width: int, height: int):
            self.color_image = wp.zeros((num_worlds, num_cameras, width * height), dtype=wp.uint32)
            self.depth_image = wp.zeros((num_worlds, num_cameras, width * height), dtype=wp.float32)

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        render_rgb: bool = True,
        render_depth: bool = True,
        use_textures: bool = True,
        use_shadows: bool = True,
        num_worlds: int = 1,
        has_global_world: bool = False,
    ):
        self.width = width
        self.height = height
        self.render_rgb = render_rgb
        self.render_depth = render_depth
        self.use_textures = use_textures
        self.use_shadows = use_shadows
        self.num_worlds = num_worlds
        self.has_global_world = has_global_world

        self.bvh: wp.Bvh = None
        self.num_geom_in_bvh = 0

        self.mesh_bounds: wp.array2d(dtype=wp.vec3f) = None
        self.mesh_texcoord: wp.array(dtype=wp.vec2f) = None
        self.mesh_texcoord_offsets: wp.array(dtype=wp.int32) = None
        self.mesh_face_offsets: wp.array(dtype=wp.int32) = None
        self.mesh_face_vertices: wp.array(dtype=wp.vec3i) = None
        self.mesh_ids: wp.array(dtype=wp.uint64) = None

        self.geom_enabled: wp.array(dtype=wp.int32) = None
        self.geom_types: wp.array(dtype=wp.int32) = None
        self.geom_mesh_indices: wp.array(dtype=wp.int32) = None
        self.geom_sizes: wp.array(dtype=wp.vec3f) = None
        self.geom_positions: wp.array(dtype=wp.vec3f) = None
        self.geom_orientations: wp.array(dtype=wp.mat33f) = None
        self.geom_materials: wp.array(dtype=wp.int32) = None
        self.geom_colors: wp.array(dtype=wp.vec4f) = None
        self.geom_world_index: wp.array(dtype=wp.int32) = None

        self.tex_adr: wp.array(dtype=wp.int32) = None
        self.tex_data: wp.array(dtype=wp.uint32) = None
        self.tex_height: wp.array(dtype=wp.int32) = None
        self.tex_width: wp.array(dtype=wp.int32) = None

        self.num_cameras = 0
        self.camera_rays: wp.array(dtype=wp.vec3f, ndim=4) = None
        self.camera_positions: wp.array(dtype=wp.vec3f) = None
        self.camera_orientations: wp.array(dtype=wp.mat33f) = None

        self.material_texture_ids: wp.array(dtype=wp.int32) = None
        self.material_texture_repeat: wp.array(dtype=wp.vec2f) = None
        self.material_rgba: wp.array(dtype=wp.vec4f) = None

        self.num_lights = 0
        self.lights_active: wp.array(dtype=wp.bool) = None
        self.lights_type: wp.array(dtype=wp.int32) = None
        self.lights_cast_shadow: wp.array(dtype=wp.bool) = None
        self.lights_position: wp.array(dtype=wp.vec3f) = None
        self.lights_orientation: wp.array(dtype=wp.vec3f) = None

    def init_outputs(self):
        self.bvh_lowers = wp.zeros((self.num_worlds_total * self.num_geom_in_bvh), dtype=wp.vec3f)
        self.bvh_uppers = wp.zeros((self.num_worlds_total * self.num_geom_in_bvh), dtype=wp.vec3f)
        self.bvh_groups = wp.zeros((self.num_worlds_total * self.num_geom_in_bvh), dtype=wp.int32)
        self.bvh_group_roots = wp.zeros((self.num_worlds_total), dtype=wp.int32)
        self.output = RenderContext.Output(self.num_worlds, self.num_cameras, self.width, self.height)

    def init_camera_rays(self):
        self.camera_rays = wp.empty((self.num_cameras, self.height, self.width, 2), dtype=wp.vec3f)

    @property
    def num_worlds_total(self) -> int:
        if self.has_global_world:
            return self.num_worlds + 1
        return self.num_worlds

    def build_bvh(self):
        self.__compute_bvh_bounds()
        self.bvh = wp.Bvh(self.bvh_lowers, self.bvh_uppers, groups=self.bvh_groups)
        wp.launch(kernel=compute_bvh_group_roots, dim=self.num_worlds_total, inputs=[self.bvh.id, self.bvh_group_roots])

    def refit_bvh(self):
        if self.bvh is None:
            self.build_bvh()
            return
        self.__compute_bvh_bounds()
        self.bvh.refit()

    def render(self):
        self.refit_bvh()
        render_megakernel(self)

    def __compute_bvh_bounds(self):
        wp.launch(
            kernel=compute_bvh_bounds,
            dim=(self.num_worlds_total * self.num_geom_in_bvh),
            inputs=[
                self.num_geom_in_bvh,
                self.num_worlds_total,
                self.geom_world_index,
                self.geom_enabled,
                self.geom_types,
                self.geom_mesh_indices,
                self.geom_sizes,
                self.geom_positions,
                self.geom_orientations,
                self.mesh_bounds,
                self.bvh_lowers,
                self.bvh_uppers,
                self.bvh_groups,
            ],
        )
