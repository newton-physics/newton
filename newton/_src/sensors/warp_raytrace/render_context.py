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

from .bvh import compute_bvh_group_roots, compute_geom_bvh_bounds, compute_particle_bvh_bounds
from .render import render_megakernel


class RenderContext:
    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        enable_textures: bool = True,
        enable_shadows: bool = True,
        enable_ambient_lighting: bool = True,
        enable_particles: bool = True,
        num_worlds: int = 1,
        num_cameras: int = 1,
        has_global_world: bool = False,
        tile_rendering: bool = False,
        tile_size: int = 8,
    ):
        self.width = width
        self.height = height
        self.tile_rendering = tile_rendering
        self.tile_size = tile_size
        self.enable_textures = enable_textures
        self.enable_shadows = enable_shadows
        self.enable_ambient_lighting = enable_ambient_lighting
        self.num_worlds = num_worlds
        self.has_global_world = has_global_world
        self.enable_particles = enable_particles
        self.max_distance = 1000.0

        self.bvh_geom: wp.Bvh = None
        self.bvh_particles: wp.Bvh = None
        self.triangle_mesh: wp.Mesh = None
        self.num_geoms = 0

        self.mesh_bounds: wp.array2d(dtype=wp.vec3f) = None
        self.mesh_texcoord: wp.array(dtype=wp.vec2f) = None
        self.mesh_texcoord_offsets: wp.array(dtype=wp.int32) = None
        self.mesh_face_offsets: wp.array(dtype=wp.int32) = None
        self.mesh_face_vertices: wp.array(dtype=wp.vec3i) = None
        self.mesh_ids: wp.array(dtype=wp.uint64) = None

        self.__triangle_points: wp.array(dtype=wp.vec3f) = None
        self.__triangle_indices: wp.array(dtype=wp.int32) = None

        self.__particles_position: wp.array(dtype=wp.vec3f) = None
        self.__particles_radius: wp.array(dtype=wp.float32) = None
        self.__particles_world_index: wp.array(dtype=wp.int32) = None

        self.geom_enabled: wp.array(dtype=wp.int32) = None
        self.geom_types: wp.array(dtype=wp.int32) = None
        self.geom_mesh_indices: wp.array(dtype=wp.int32) = None
        self.geom_sizes: wp.array(dtype=wp.vec3f) = None
        self.geom_positions: wp.array(dtype=wp.vec3f) = None
        self.geom_orientations: wp.array(dtype=wp.mat33f) = None
        self.geom_materials: wp.array(dtype=wp.int32) = None
        self.geom_colors: wp.array(dtype=wp.vec4f) = None
        self.geom_world_index: wp.array(dtype=wp.int32) = None

        self.texture_offsets: wp.array(dtype=wp.int32) = None
        self.texture_data: wp.array(dtype=wp.uint32) = None
        self.texture_width: wp.array(dtype=wp.int32) = None
        self.texture_height: wp.array(dtype=wp.int32) = None

        self.num_cameras = num_cameras

        self.material_texture_ids: wp.array(dtype=wp.int32) = None
        self.material_texture_repeat: wp.array(dtype=wp.vec2f) = None
        self.material_rgba: wp.array(dtype=wp.vec4f) = None

        self.lights_active: wp.array(dtype=wp.bool) = None
        self.lights_type: wp.array(dtype=wp.int32) = None
        self.lights_cast_shadow: wp.array(dtype=wp.bool) = None
        self.lights_position: wp.array(dtype=wp.vec3f) = None
        self.lights_orientation: wp.array(dtype=wp.vec3f) = None

        self.bvh_geom_lowers: wp.array(dtype=wp.vec3f) = None
        self.bvh_geom_uppers: wp.array(dtype=wp.vec3f) = None
        self.bvh_geom_groups: wp.array(dtype=wp.int32) = None
        self.bvh_geom_group_roots: wp.array(dtype=wp.int32) = None
        self.bvh_particles_lowers: wp.array(dtype=wp.vec3f) = None
        self.bvh_particles_uppers: wp.array(dtype=wp.vec3f) = None
        self.bvh_particles_groups: wp.array(dtype=wp.int32) = None
        self.bvh_particles_group_roots: wp.array(dtype=wp.int32) = None

    def __init_geom_outputs(self):
        if self.bvh_geom_lowers is None:
            self.bvh_geom_lowers = wp.zeros(self.num_geoms_total, dtype=wp.vec3f)
        if self.bvh_geom_uppers is None:
            self.bvh_geom_uppers = wp.zeros(self.num_geoms_total, dtype=wp.vec3f)
        if self.bvh_geom_groups is None:
            self.bvh_geom_groups = wp.zeros(self.num_geoms_total, dtype=wp.int32)
        if self.bvh_geom_group_roots is None:
            self.bvh_geom_group_roots = wp.zeros((self.num_worlds_total), dtype=wp.int32)

    def __init_particle_outputs(self):
        if self.bvh_particles_lowers is None:
            self.bvh_particles_lowers = wp.zeros(self.num_particles_total, dtype=wp.vec3f)
        if self.bvh_particles_uppers is None:
            self.bvh_particles_uppers = wp.zeros(self.num_particles_total, dtype=wp.vec3f)
        if self.bvh_particles_groups is None:
            self.bvh_particles_groups = wp.zeros(self.num_particles_total, dtype=wp.int32)
        if self.bvh_particles_group_roots is None:
            self.bvh_particles_group_roots = wp.zeros((self.num_worlds_total), dtype=wp.int32)

    def create_color_image_output(self):
        return wp.zeros((self.num_worlds, self.num_cameras, self.width * self.height), dtype=wp.uint32)

    def create_depth_image_output(self):
        return wp.zeros((self.num_worlds, self.num_cameras, self.width * self.height), dtype=wp.float32)

    def refit_bvh(self):
        if self.num_geoms_total:
            self.__init_geom_outputs()
            self.__compute_bvh_geom_bounds()
            if self.bvh_geom is None:
                self.bvh_geom = wp.Bvh(self.bvh_geom_lowers, self.bvh_geom_uppers, groups=self.bvh_geom_groups)
                wp.launch(
                    kernel=compute_bvh_group_roots,
                    dim=self.num_worlds_total,
                    inputs=[self.bvh_geom.id, self.bvh_geom_group_roots],
                )
            else:
                self.bvh_geom.refit()

        if self.num_particles_total:
            self.__init_particle_outputs()
            self.__compute_bvh_particle_bounds()
            if self.bvh_particles is None:
                self.bvh_particles = wp.Bvh(
                    self.bvh_particles_lowers,
                    self.bvh_particles_uppers,
                    groups=self.bvh_particles_groups,
                )
                wp.launch(
                    kernel=compute_bvh_group_roots,
                    dim=self.num_worlds_total,
                    inputs=[self.bvh_particles.id, self.bvh_particles_group_roots],
                )
            else:
                self.bvh_particles.refit()

        if self.has_triangle_mesh:
            if self.triangle_mesh is None:
                self.triangle_mesh = wp.Mesh(self.triangle_points, self.triangle_indices)
            else:
                self.triangle_mesh.refit()

    def render(
        self,
        camera_transforms: wp.array(dtype=wp.transformf, ndim=2),
        camera_rays: wp.array(dtype=wp.vec3f, ndim=4),
        color_image: wp.array(dtype=wp.uint32, ndim=3) | None = None,
        depth_image: wp.array(dtype=wp.float32, ndim=3) | None = None,
        refit_bvh: bool = True,
        clear_images: bool = True,
    ):
        if self.has_geometries or self.has_particles or self.has_triangle_mesh:
            if refit_bvh:
                self.refit_bvh()
            render_megakernel(self, camera_transforms, camera_rays, color_image, depth_image, clear_images)

    def __compute_bvh_geom_bounds(self):
        wp.launch(
            kernel=compute_geom_bvh_bounds,
            dim=self.num_geoms_total,
            inputs=[
                self.num_geoms_total,
                self.num_worlds_total,
                self.geom_world_index,
                self.geom_enabled,
                self.geom_types,
                self.geom_mesh_indices,
                self.geom_sizes,
                self.geom_positions,
                self.geom_orientations,
                self.mesh_bounds,
                self.bvh_geom_lowers,
                self.bvh_geom_uppers,
                self.bvh_geom_groups,
            ],
        )

    def __compute_bvh_particle_bounds(self):
        wp.launch(
            kernel=compute_particle_bvh_bounds,
            dim=self.num_particles_total,
            inputs=[
                self.particles_position.shape[0],
                self.num_worlds_total,
                self.particles_world_index,
                self.particles_position,
                self.particles_radius,
                self.bvh_particles_lowers,
                self.bvh_particles_uppers,
                self.bvh_particles_groups,
            ],
        )

    @property
    def num_worlds_total(self) -> int:
        if self.has_global_world:
            return self.num_worlds + 1
        return self.num_worlds

    @property
    def num_geoms_total(self) -> int:
        return self.num_geoms

    @property
    def num_particles_total(self) -> int:
        if self.particles_position is not None:
            return self.particles_position.shape[0]
        return 0

    @property
    def num_lights(self) -> int:
        if self.lights_active is not None:
            return self.lights_active.shape[0]
        return 0

    @property
    def has_geometries(self) -> bool:
        return self.num_geoms_total > 0

    @property
    def has_particles(self) -> bool:
        return self.particles_position is not None

    @property
    def has_triangle_mesh(self) -> bool:
        return self.triangle_points is not None

    @property
    def triangle_points(self) -> wp.array(dtype=wp.vec3f):
        return self.__triangle_points

    @triangle_points.setter
    def triangle_points(self, triangle_points: wp.array(dtype=wp.vec3f)):
        if self.__triangle_points is None or self.__triangle_points.ptr != triangle_points.ptr:
            self.triangle_mesh = None
        self.__triangle_points = triangle_points

    @property
    def triangle_indices(self) -> wp.array(dtype=wp.int32):
        return self.__triangle_indices

    @triangle_indices.setter
    def triangle_indices(self, triangle_indices: wp.array(dtype=wp.int32)):
        if self.__triangle_indices is None or self.__triangle_indices.ptr != triangle_indices.ptr:
            self.triangle_mesh = None
        self.__triangle_indices = triangle_indices

    @property
    def particles_position(self) -> wp.array(dtype=wp.vec3f):
        return self.__particles_position

    @particles_position.setter
    def particles_position(self, particles_position: wp.array(dtype=wp.vec3f)):
        if self.__particles_position is None or self.__particles_position.ptr != particles_position.ptr:
            self.bvh_particles = None
        self.__particles_position = particles_position

    @property
    def particles_radius(self) -> wp.array(dtype=wp.float32):
        return self.__particles_radius

    @particles_radius.setter
    def particles_radius(self, particles_radius: wp.array(dtype=wp.float32)):
        if self.__particles_radius is None or self.__particles_radius.ptr != particles_radius.ptr:
            self.bvh_particles = None
        self.__particles_radius = particles_radius

    @property
    def particles_world_index(self) -> wp.array(dtype=wp.int32):
        return self.__particles_world_index

    @particles_world_index.setter
    def particles_world_index(self, particles_world_index: wp.array(dtype=wp.int32)):
        if self.__particles_world_index is None or self.__particles_world_index.ptr != particles_world_index.ptr:
            self.bvh_particles = None
        self.__particles_world_index = particles_world_index
