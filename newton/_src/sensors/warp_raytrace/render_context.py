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
        self.__width = width
        self.__height = height
        self.__tile_rendering = tile_rendering
        self.__tile_size = tile_size
        self.__enable_textures = enable_textures
        self.__enable_shadows = enable_shadows
        self.__enable_ambient_lighting = enable_ambient_lighting
        self.__num_worlds = num_worlds
        self.__has_global_world = has_global_world
        self.__enable_particles = enable_particles
        self.__max_distance = 1000.0

        self.__bvh_geom: wp.Bvh = None
        self.__bvh_particles: wp.Bvh = None
        self.__triangle_mesh: wp.Mesh = None
        self.__num_geom_in_bvh = 0

        self.__mesh_bounds: wp.array2d(dtype=wp.vec3f) = None
        self.__mesh_texcoord: wp.array(dtype=wp.vec2f) = None
        self.__mesh_texcoord_offsets: wp.array(dtype=wp.int32) = None
        self.__mesh_face_offsets: wp.array(dtype=wp.int32) = None
        self.__mesh_face_vertices: wp.array(dtype=wp.vec3i) = None
        self.__mesh_ids: wp.array(dtype=wp.uint64) = None

        self.__triangle_points: wp.array(dtype=wp.vec3f) = None
        self.__triangle_indices: wp.array(dtype=wp.int32) = None

        self.__particles_position: wp.array(dtype=wp.vec3f) = None
        self.__particles_radius: wp.array(dtype=wp.float32) = None
        self.__particles_world_index: wp.array(dtype=wp.int32) = None

        self.__geom_enabled: wp.array(dtype=wp.int32) = None
        self.__geom_types: wp.array(dtype=wp.int32) = None
        self.__geom_mesh_indices: wp.array(dtype=wp.int32) = None
        self.__geom_sizes: wp.array(dtype=wp.vec3f) = None
        self.__geom_positions: wp.array(dtype=wp.vec3f) = None
        self.__geom_orientations: wp.array(dtype=wp.mat33f) = None
        self.__geom_materials: wp.array(dtype=wp.int32) = None
        self.__geom_colors: wp.array(dtype=wp.vec4f) = None
        self.__geom_world_index: wp.array(dtype=wp.int32) = None

        self.__texture_offsets: wp.array(dtype=wp.int32) = None
        self.__texture_data: wp.array(dtype=wp.uint32) = None
        self.__texture_width: wp.array(dtype=wp.int32) = None
        self.__texture_height: wp.array(dtype=wp.int32) = None

        self.__num_cameras = num_cameras
        self.__camera_rays: wp.array(dtype=wp.vec3f, ndim=4) = None
        self.__camera_positions: wp.array(dtype=wp.vec3f) = None
        self.__camera_orientations: wp.array(dtype=wp.mat33f) = None

        self.__material_texture_ids: wp.array(dtype=wp.int32) = None
        self.__material_texture_repeat: wp.array(dtype=wp.vec2f) = None
        self.__material_rgba: wp.array(dtype=wp.vec4f) = None

        self.__lights_active: wp.array(dtype=wp.bool) = None
        self.__lights_type: wp.array(dtype=wp.int32) = None
        self.__lights_cast_shadow: wp.array(dtype=wp.bool) = None
        self.__lights_position: wp.array(dtype=wp.vec3f) = None
        self.__lights_orientation: wp.array(dtype=wp.vec3f) = None

        self.__bvh_geom_lowers: wp.array(dtype=wp.vec3f) = None
        self.__bvh_geom_uppers: wp.array(dtype=wp.vec3f) = None
        self.__bvh_geom_groups: wp.array(dtype=wp.int32) = None
        self.__bvh_geom_group_roots: wp.array(dtype=wp.int32) = None
        self.__bvh_particles_lowers: wp.array(dtype=wp.vec3f) = None
        self.__bvh_particles_uppers: wp.array(dtype=wp.vec3f) = None
        self.__bvh_particles_groups: wp.array(dtype=wp.int32) = None
        self.__bvh_particles_group_roots: wp.array(dtype=wp.int32) = None

    def __init_geom_outputs(self):
        if self.__bvh_geom_lowers is None:
            self.__bvh_geom_lowers = wp.zeros(self.num_geom_in_bvh_total, dtype=wp.vec3f)
        if self.__bvh_geom_uppers is None:
            self.__bvh_geom_uppers = wp.zeros(self.num_geom_in_bvh_total, dtype=wp.vec3f)
        if self.__bvh_geom_groups is None:
            self.__bvh_geom_groups = wp.zeros(self.num_geom_in_bvh_total, dtype=wp.int32)
        if self.__bvh_geom_group_roots is None:
            self.__bvh_geom_group_roots = wp.zeros((self.num_worlds_total), dtype=wp.int32)

    def __init_particle_outputs(self):
        if self.__bvh_particles_lowers is None:
            self.__bvh_particles_lowers = wp.zeros(self.num_particle_in_bvh_total, dtype=wp.vec3f)
        if self.__bvh_particles_uppers is None:
            self.__bvh_particles_uppers = wp.zeros(self.num_particle_in_bvh_total, dtype=wp.vec3f)
        if self.__bvh_particles_groups is None:
            self.__bvh_particles_groups = wp.zeros(self.num_particle_in_bvh_total, dtype=wp.int32)
        if self.__bvh_particles_group_roots is None:
            self.__bvh_particles_group_roots = wp.zeros((self.num_worlds_total), dtype=wp.int32)

    def init_camera_rays(self):
        self.camera_rays = wp.empty((self.num_cameras, self.height, self.width, 2), dtype=wp.vec3f)

    def create_color_image_output(self):
        return wp.zeros((self.num_worlds, self.num_cameras, self.width * self.height), dtype=wp.uint32)

    def create_depth_image_output(self):
        return wp.zeros((self.num_worlds, self.num_cameras, self.width * self.height), dtype=wp.float32)

    def refit_bvh(self):
        if self.num_geom_in_bvh_total:
            self.__init_geom_outputs()
            self.__compute_bvh_geom_bounds()
            if self.bvh_geom is None:
                self.__bvh_geom = wp.Bvh(self.bvh_geom_lowers, self.bvh_geom_uppers, groups=self.bvh_geom_groups)
                wp.launch(
                    kernel=compute_bvh_group_roots,
                    dim=self.num_worlds_total,
                    inputs=[self.bvh_geom.id, self.bvh_geom_group_roots],
                )
            else:
                self.bvh_geom.refit()

        if self.num_particle_in_bvh_total:
            self.__init_particle_outputs()
            self.__compute_bvh_particle_bounds()
            if self.bvh_particles is None:
                self.__bvh_particles = wp.Bvh(
                    self.bvh_particles_lowers, self.bvh_particles_uppers, groups=self.bvh_particles_groups
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
                self.__triangle_mesh = wp.Mesh(self.triangle_points, self.triangle_indices)
            else:
                self.triangle_mesh.refit()

    def render(
        self,
        color_image: wp.array(dtype=wp.uint32, ndim=3) | None = None,
        depth_image: wp.array(dtype=wp.float32, ndim=3) | None = None,
        refit_bvh: bool = True,
        clear_images: bool = True,
    ):
        if self.has_geometries or self.has_particles or self.has_triangle_mesh:
            if refit_bvh:
                self.refit_bvh()
            render_megakernel(self, color_image, depth_image, clear_images)

    def __compute_bvh_geom_bounds(self):
        wp.launch(
            kernel=compute_geom_bvh_bounds,
            dim=self.num_geom_in_bvh_total,
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
                self.bvh_geom_lowers,
                self.bvh_geom_uppers,
                self.bvh_geom_groups,
            ],
        )

    def __compute_bvh_particle_bounds(self):
        wp.launch(
            kernel=compute_particle_bvh_bounds,
            dim=self.num_particle_in_bvh_total,
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
    def num_geom_in_bvh(self) -> int:
        return self.__num_geom_in_bvh

    @num_geom_in_bvh.setter
    def num_geom_in_bvh(self, num_geom_in_bvh: int):
        self.__num_geom_in_bvh = num_geom_in_bvh

    @property
    def num_geom_in_bvh_total(self) -> int:
        return self.num_geom_in_bvh

    @property
    def num_particle_in_bvh_total(self) -> int:
        if self.particles_position is not None:
            return self.particles_position.shape[0]
        return 0

    @property
    def has_geometries(self) -> bool:
        return self.num_geom_in_bvh > 0

    @property
    def has_particles(self) -> bool:
        return self.particles_position is not None

    @property
    def has_triangle_mesh(self) -> bool:
        return self.triangle_points is not None

    @property
    def width(self) -> int:
        return self.__width

    @property
    def height(self) -> int:
        return self.__height

    @property
    def tile_size(self) -> int:
        return self.__tile_size

    @tile_size.setter
    def tile_size(self, tile_size: int) -> int:
        self.__tile_size = tile_size

    @property
    def tile_rendering(self) -> bool:
        return self.__tile_rendering

    @tile_rendering.setter
    def tile_rendering(self, tile_rendering: bool) -> bool:
        self.__tile_rendering = tile_rendering

    @property
    def enable_textures(self) -> bool:
        return self.__enable_textures

    @enable_textures.setter
    def enable_textures(self, enable_textures: bool):
        self.__enable_textures = enable_textures

    @property
    def enable_shadows(self) -> bool:
        return self.__enable_shadows

    @enable_shadows.setter
    def enable_shadows(self, enable_shadows: bool):
        self.__enable_shadows = enable_shadows

    @property
    def enable_ambient_lighting(self) -> bool:
        return self.__enable_ambient_lighting

    @enable_ambient_lighting.setter
    def enable_ambient_lighting(self, enable_ambient_lighting: bool):
        self.__enable_ambient_lighting = enable_ambient_lighting

    @property
    def enable_particles(self) -> bool:
        return self.__enable_particles

    @enable_particles.setter
    def enable_particles(self, enable_particles: bool):
        self.__enable_particles = enable_particles

    @property
    def num_worlds(self) -> int:
        return self.__num_worlds

    @property
    def has_global_world(self) -> bool:
        return self.__has_global_world

    @property
    def max_distance(self) -> float:
        return self.__max_distance

    @property
    def num_cameras(self) -> int:
        return self.__num_cameras

    @property
    def num_lights(self) -> int:
        if self.__lights_active is not None:
            return self.__lights_active.shape[0]
        return 0

    @property
    def mesh_bounds(self) -> wp.array2d(dtype=wp.vec3f):
        return self.__mesh_bounds

    @mesh_bounds.setter
    def mesh_bounds(self, mesh_bounds: wp.array2d(dtype=wp.vec3f)):
        self.__mesh_bounds = mesh_bounds

    @property
    def mesh_texcoord(self) -> wp.array(dtype=wp.vec2f):
        return self.__mesh_texcoord

    @mesh_texcoord.setter
    def mesh_texcoord(self, mesh_texcoord: wp.array(dtype=wp.vec2f)):
        self.__mesh_texcoord = mesh_texcoord

    @property
    def mesh_texcoord_offsets(self) -> wp.array(dtype=wp.int32):
        return self.__mesh_texcoord_offsets

    @mesh_texcoord_offsets.setter
    def mesh_texcoord_offsets(self, mesh_texcoord_offsets: wp.array(dtype=wp.int32)):
        self.__mesh_texcoord_offsets = mesh_texcoord_offsets

    @property
    def mesh_face_offsets(self) -> wp.array(dtype=wp.int32):
        return self.__mesh_face_offsets

    @mesh_face_offsets.setter
    def mesh_face_offsets(self, mesh_face_offsets: wp.array(dtype=wp.int32)):
        self.__mesh_face_offsets = mesh_face_offsets

    @property
    def mesh_face_vertices(self) -> wp.array(dtype=wp.vec3i):
        return self.__mesh_face_vertices

    @mesh_face_vertices.setter
    def mesh_face_vertices(self, mesh_face_vertices: wp.array(dtype=wp.vec3i)):
        self.__mesh_face_vertices = mesh_face_vertices

    @property
    def mesh_ids(self) -> wp.array(dtype=wp.uint64):
        return self.__mesh_ids

    @mesh_ids.setter
    def mesh_ids(self, mesh_ids: wp.array(dtype=wp.uint64)):
        self.__mesh_ids = mesh_ids

    @property
    def triangle_points(self) -> wp.array(dtype=wp.vec3f):
        return self.__triangle_points

    @triangle_points.setter
    def triangle_points(self, triangle_points: wp.array(dtype=wp.vec3f)):
        if self.__triangle_points is None or self.__triangle_points.ptr != triangle_points.ptr:
            self.__triangle_mesh = None
        self.__triangle_points = triangle_points

    @property
    def triangle_indices(self) -> wp.array(dtype=wp.int32):
        return self.__triangle_indices

    @triangle_indices.setter
    def triangle_indices(self, triangle_indices: wp.array(dtype=wp.int32)):
        if self.__triangle_indices is None or self.__triangle_indices.ptr != triangle_indices.ptr:
            self.__triangle_mesh = None
        self.__triangle_indices = triangle_indices

    @property
    def particles_position(self) -> wp.array(dtype=wp.vec3f):
        return self.__particles_position

    @particles_position.setter
    def particles_position(self, particles_position: wp.array(dtype=wp.vec3f)):
        if self.__particles_position is None or self.__particles_position.ptr != particles_position.ptr:
            self.__bvh_particles = None
        self.__particles_position = particles_position

    @property
    def particles_radius(self) -> wp.array(dtype=wp.float32):
        return self.__particles_radius

    @particles_radius.setter
    def particles_radius(self, particles_radius: wp.array(dtype=wp.float32)):
        if self.__particles_radius is None or self.__particles_radius.ptr != particles_radius.ptr:
            self.__bvh_particles = None
        self.__particles_radius = particles_radius

    @property
    def particles_world_index(self) -> wp.array(dtype=wp.int32):
        return self.__particles_world_index

    @particles_world_index.setter
    def particles_world_index(self, particles_world_index: wp.array(dtype=wp.int32)):
        if self.__particles_world_index is None or self.__particles_world_index.ptr != particles_world_index.ptr:
            self.__bvh_particles = None
        self.__particles_world_index = particles_world_index

    @property
    def geom_enabled(self) -> wp.array(dtype=wp.int32):
        return self.__geom_enabled

    @geom_enabled.setter
    def geom_enabled(self, geom_enabled: wp.array(dtype=wp.int32)):
        self.__geom_enabled = geom_enabled

    @property
    def geom_types(self) -> wp.array(dtype=wp.int32):
        return self.__geom_types

    @geom_types.setter
    def geom_types(self, geom_types: wp.array(dtype=wp.int32)):
        self.__geom_types = geom_types

    @property
    def geom_mesh_indices(self) -> wp.array(dtype=wp.int32):
        return self.__geom_mesh_indices

    @geom_mesh_indices.setter
    def geom_mesh_indices(self, geom_mesh_indices: wp.array(dtype=wp.int32)):
        self.__geom_mesh_indices = geom_mesh_indices

    @property
    def geom_sizes(self) -> wp.array(dtype=wp.vec3f):
        return self.__geom_sizes

    @geom_sizes.setter
    def geom_sizes(self, geom_sizes: wp.array(dtype=wp.vec3f)):
        self.__geom_sizes = geom_sizes

    @property
    def geom_positions(self) -> wp.array(dtype=wp.vec3f):
        return self.__geom_positions

    @geom_positions.setter
    def geom_positions(self, geom_positions: wp.array(dtype=wp.vec3f)):
        self.__geom_positions = geom_positions

    @property
    def geom_orientations(self) -> wp.array(dtype=wp.mat33f):
        return self.__geom_orientations

    @geom_orientations.setter
    def geom_orientations(self, geom_orientations: wp.array(dtype=wp.mat33f)):
        self.__geom_orientations = geom_orientations

    @property
    def geom_materials(self) -> wp.array(dtype=wp.int32):
        return self.__geom_materials

    @geom_materials.setter
    def geom_materials(self, geom_materials: wp.array(dtype=wp.int32)):
        self.__geom_materials = geom_materials

    @property
    def geom_colors(self) -> wp.array(dtype=wp.vec4f):
        return self.__geom_colors

    @geom_colors.setter
    def geom_colors(self, geom_colors: wp.array(dtype=wp.vec4f)):
        self.__geom_colors = geom_colors

    @property
    def geom_world_index(self) -> wp.array(dtype=wp.int32):
        return self.__geom_world_index

    @geom_world_index.setter
    def geom_world_index(self, geom_world_index: wp.array(dtype=wp.int32)):
        self.__geom_world_index = geom_world_index

    @property
    def texture_offsets(self) -> wp.array(dtype=wp.int32):
        return self.__texture_offsets

    @texture_offsets.setter
    def texture_offsets(self, texture_offsets: wp.array(dtype=wp.int32)):
        self.__texture_offsets = texture_offsets

    @property
    def texture_data(self) -> wp.array(dtype=wp.uint32):
        return self.__texture_data

    @texture_data.setter
    def texture_data(self, texture_data: wp.array(dtype=wp.uint32)):
        self.__texture_data = texture_data

    @property
    def texture_width(self) -> wp.array(dtype=wp.int32):
        return self.__texture_width

    @texture_width.setter
    def texture_width(self, texture_width: wp.array(dtype=wp.int32)):
        self.__texture_width = texture_width

    @property
    def texture_height(self) -> wp.array(dtype=wp.int32):
        return self.__texture_height

    @texture_height.setter
    def texture_height(self, texture_height: wp.array(dtype=wp.int32)):
        self.__texture_height = texture_height

    @property
    def camera_rays(self) -> wp.array(dtype=wp.vec3f, ndim=4):
        return self.__camera_rays

    @camera_rays.setter
    def camera_rays(self, camera_rays: wp.array(dtype=wp.vec3f, ndim=4)):
        self.__camera_rays = camera_rays

    @property
    def camera_positions(self) -> wp.array(dtype=wp.vec3f):
        return self.__camera_positions

    @camera_positions.setter
    def camera_positions(self, camera_positions: wp.array(dtype=wp.vec3f)):
        self.__camera_positions = camera_positions

    @property
    def camera_orientations(self) -> wp.array(dtype=wp.mat33f):
        return self.__camera_orientations

    @camera_orientations.setter
    def camera_orientations(self, camera_orientations: wp.array(dtype=wp.mat33f)):
        self.__camera_orientations = camera_orientations

    @property
    def material_texture_ids(self) -> wp.array(dtype=wp.int32):
        return self.__material_texture_ids

    @material_texture_ids.setter
    def material_texture_ids(self, material_texture_ids: wp.array(dtype=wp.int32)):
        self.__material_texture_ids = material_texture_ids

    @property
    def material_texture_repeat(self) -> wp.array(dtype=wp.vec2f):
        return self.__material_texture_repeat

    @material_texture_repeat.setter
    def material_texture_repeat(self, material_texture_repeat: wp.array(dtype=wp.vec2f)):
        self.__material_texture_repeat = material_texture_repeat

    @property
    def material_rgba(self) -> wp.array(dtype=wp.vec4f):
        return self.__material_rgba

    @material_rgba.setter
    def material_rgba(self, material_rgba: wp.array(dtype=wp.vec4f)):
        self.__material_rgba = material_rgba

    @property
    def lights_active(self) -> wp.array(dtype=wp.bool):
        return self.__lights_active

    @lights_active.setter
    def lights_active(self, lights_active: wp.array(dtype=wp.bool)):
        self.__lights_active = lights_active

    @property
    def lights_type(self) -> wp.array(dtype=wp.int32):
        return self.__lights_type

    @lights_type.setter
    def lights_type(self, lights_type: wp.array(dtype=wp.int32)):
        self.__lights_type = lights_type

    @property
    def lights_cast_shadow(self) -> wp.array(dtype=wp.bool):
        return self.__lights_cast_shadow

    @lights_cast_shadow.setter
    def lights_cast_shadow(self, lights_cast_shadow: wp.array(dtype=wp.bool)):
        self.__lights_cast_shadow = lights_cast_shadow

    @property
    def lights_position(self) -> wp.array(dtype=wp.vec3f):
        return self.__lights_position

    @lights_position.setter
    def lights_position(self, lights_position: wp.array(dtype=wp.vec3f)):
        self.__lights_position = lights_position

    @property
    def lights_orientation(self) -> wp.array(dtype=wp.vec3f):
        return self.__lights_orientation

    @lights_orientation.setter
    def lights_orientation(self, lights_orientation: wp.array(dtype=wp.vec3f)):
        self.__lights_orientation = lights_orientation

    @property
    def triangle_mesh(self) -> wp.Mesh:
        return self.__triangle_mesh

    @property
    def bvh_geom_lowers(self) -> wp.array(dtype=wp.vec3f):
        return self.__bvh_geom_lowers

    @property
    def bvh_geom_uppers(self) -> wp.array(dtype=wp.vec3f):
        return self.__bvh_geom_uppers

    @property
    def bvh_geom_groups(self) -> wp.array(dtype=wp.int32):
        return self.__bvh_geom_groups

    @property
    def bvh_geom_group_roots(self) -> wp.array(dtype=wp.int32):
        return self.__bvh_geom_group_roots

    @property
    def bvh_particles_lowers(self) -> wp.array(dtype=wp.vec3f):
        return self.__bvh_particles_lowers

    @property
    def bvh_particles_uppers(self) -> wp.array(dtype=wp.vec3f):
        return self.__bvh_particles_uppers

    @property
    def bvh_particles_groups(self) -> wp.array(dtype=wp.int32):
        return self.__bvh_particles_groups

    @property
    def bvh_particles_group_roots(self) -> wp.array(dtype=wp.int32):
        return self.__bvh_particles_group_roots

    @property
    def bvh_geom(self) -> wp.Bvh:
        return self.__bvh_geom

    @property
    def bvh_particles(self) -> wp.Bvh:
        return self.__bvh_particles
