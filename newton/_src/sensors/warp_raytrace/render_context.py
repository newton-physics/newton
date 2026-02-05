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

from dataclasses import dataclass, field

import warp as wp

from .bvh import (
    compute_bvh_group_roots,
    compute_particle_bvh_bounds,
    compute_shape_bvh_bounds,
)
from .render import render_megakernel
from .types import RenderOrder
from .utils import Utils


@dataclass
class ClearData:
    clear_color: int | wp.int32 | None = field(default_factory=lambda: wp.int32(0))
    clear_depth: float | wp.float32 | None = field(default_factory=lambda: wp.float32(0.0))
    clear_shape_index: int | wp.uint32 | None = field(default_factory=lambda: wp.uint32(0xFFFFFFFF))
    clear_normal: wp.vec3f | None = field(default_factory=lambda: wp.vec3f(0.0))
    clear_albedo: int | wp.int32 | None = field(default_factory=lambda: wp.int32(0))


DEFAULT_CLEAR_DATA = ClearData()


class RenderContext:
    @dataclass
    class Options:
        enable_global_world: bool = True
        enable_textures: bool = True
        enable_shadows: bool = True
        enable_ambient_lighting: bool = True
        enable_particles: bool = True
        enable_backface_culling: bool = True
        render_order: int = RenderOrder.PIXEL_PRIORITY
        tile_width: int = 16
        tile_height: int = 8
        max_distance: float = 1000.0

    def __init__(self, num_worlds: int = 1, options: Options | None = None, device: str | None = None):
        self.device = device
        self.utils = Utils(self)
        self.options = options if options else RenderContext.Options()

        self.num_worlds = num_worlds

        self.bvh_shapes: wp.Bvh = None
        self.bvh_particles: wp.Bvh = None
        self.triangle_mesh: wp.Mesh = None
        self.num_shapes_enabled = 0
        self.num_shapes_total = 0

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

        self.shape_enabled: wp.array(dtype=wp.uint32) = None
        self.shape_types: wp.array(dtype=wp.int32) = None
        self.shape_mesh_indices: wp.array(dtype=wp.int32) = None
        self.shape_sizes: wp.array(dtype=wp.vec3f) = None
        self.shape_transforms: wp.array(dtype=wp.transformf) = None
        self.shape_materials: wp.array(dtype=wp.int32) = None
        self.shape_colors: wp.array(dtype=wp.vec4f) = None
        self.shape_world_index: wp.array(dtype=wp.int32) = None

        self.texture_offsets: wp.array(dtype=wp.int32) = None
        self.texture_data: wp.array(dtype=wp.uint32) = None
        self.texture_width: wp.array(dtype=wp.int32) = None
        self.texture_height: wp.array(dtype=wp.int32) = None

        self.material_texture_ids: wp.array(dtype=wp.int32) = None
        self.material_texture_repeat: wp.array(dtype=wp.vec2f) = None
        self.material_rgba: wp.array(dtype=wp.vec4f) = None

        self.lights_active: wp.array(dtype=wp.bool) = None
        self.lights_type: wp.array(dtype=wp.int32) = None
        self.lights_cast_shadow: wp.array(dtype=wp.bool) = None
        self.lights_position: wp.array(dtype=wp.vec3f) = None
        self.lights_orientation: wp.array(dtype=wp.vec3f) = None

        self.bvh_shapes_lowers: wp.array(dtype=wp.vec3f) = None
        self.bvh_shapes_uppers: wp.array(dtype=wp.vec3f) = None
        self.bvh_shapes_groups: wp.array(dtype=wp.int32) = None
        self.bvh_shapes_group_roots: wp.array(dtype=wp.int32) = None
        self.bvh_particles_lowers: wp.array(dtype=wp.vec3f) = None
        self.bvh_particles_uppers: wp.array(dtype=wp.vec3f) = None
        self.bvh_particles_groups: wp.array(dtype=wp.int32) = None
        self.bvh_particles_group_roots: wp.array(dtype=wp.int32) = None

    def __init_shape_outputs(self):
        if self.bvh_shapes_lowers is None:
            self.bvh_shapes_lowers = wp.zeros(self.num_shapes_enabled, dtype=wp.vec3f, device=self.device)
        if self.bvh_shapes_uppers is None:
            self.bvh_shapes_uppers = wp.zeros(self.num_shapes_enabled, dtype=wp.vec3f, device=self.device)
        if self.bvh_shapes_groups is None:
            self.bvh_shapes_groups = wp.zeros(self.num_shapes_enabled, dtype=wp.int32, device=self.device)
        if self.bvh_shapes_group_roots is None:
            self.bvh_shapes_group_roots = wp.zeros((self.num_worlds_total), dtype=wp.int32, device=self.device)

    def __init_particle_outputs(self):
        if self.bvh_particles_lowers is None:
            self.bvh_particles_lowers = wp.zeros(self.num_particles_total, dtype=wp.vec3f, device=self.device)
        if self.bvh_particles_uppers is None:
            self.bvh_particles_uppers = wp.zeros(self.num_particles_total, dtype=wp.vec3f, device=self.device)
        if self.bvh_particles_groups is None:
            self.bvh_particles_groups = wp.zeros(self.num_particles_total, dtype=wp.int32, device=self.device)
        if self.bvh_particles_group_roots is None:
            self.bvh_particles_group_roots = wp.zeros((self.num_worlds_total), dtype=wp.int32, device=self.device)

    def create_color_image_output(self, width: int, height: int, num_cameras: int = 1) -> wp.array(
        dtype=wp.uint32, ndim=4
    ):
        return wp.zeros((self.num_worlds, num_cameras, height, width), dtype=wp.uint32, device=self.device)

    def create_depth_image_output(self, width: int, height: int, num_cameras: int = 1) -> wp.array(
        dtype=wp.float32, ndim=4
    ):
        return wp.zeros((self.num_worlds, num_cameras, height, width), dtype=wp.float32, device=self.device)

    def create_shape_index_image_output(self, width: int, height: int, num_cameras: int = 1) -> wp.array(
        dtype=wp.uint32, ndim=4
    ):
        return wp.zeros((self.num_worlds, num_cameras, height, width), dtype=wp.uint32, device=self.device)

    def create_normal_image_output(self, width: int, height: int, num_cameras: int = 1) -> wp.array(
        dtype=wp.vec3f, ndim=4
    ):
        return wp.zeros((self.num_worlds, num_cameras, height, width), dtype=wp.vec3f, device=self.device)

    def create_albedo_image_output(self, width: int, height: int, num_cameras: int = 1) -> wp.array(
        dtype=wp.uint32, ndim=4
    ):
        return wp.zeros((self.num_worlds, num_cameras, height, width), dtype=wp.uint32, device=self.device)

    def refit_bvh(self):
        if self.num_shapes_enabled:
            self.__init_shape_outputs()
            self.__compute_bvh_shape_bounds()
            if self.bvh_shapes is None:
                self.bvh_shapes = wp.Bvh(self.bvh_shapes_lowers, self.bvh_shapes_uppers, groups=self.bvh_shapes_groups)
                wp.launch(
                    kernel=compute_bvh_group_roots,
                    dim=self.num_worlds_total,
                    inputs=[self.bvh_shapes.id, self.bvh_shapes_group_roots],
                    device=self.device,
                )
            else:
                self.bvh_shapes.refit()

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
                    device=self.device,
                )
            else:
                self.bvh_particles.refit()

        if self.has_triangle_mesh:
            if self.triangle_mesh is None:
                self.triangle_mesh = wp.Mesh(self.triangle_points, self.triangle_indices, device=self.device)
            else:
                self.triangle_mesh.refit()

    def render(
        self,
        camera_transforms: wp.array(dtype=wp.transformf, ndim=2),
        camera_rays: wp.array(dtype=wp.vec3f, ndim=4),
        color_image: wp.array(dtype=wp.uint32, ndim=4) | None = None,
        depth_image: wp.array(dtype=wp.float32, ndim=4) | None = None,
        shape_index_image: wp.array(dtype=wp.uint32, ndim=4) | None = None,
        normal_image: wp.array(dtype=wp.vec3f, ndim=4) | None = None,
        albedo_image: wp.array(dtype=wp.uint32, ndim=4) | None = None,
        refit_bvh: bool = True,
        clear_data: ClearData | None = DEFAULT_CLEAR_DATA,
    ):
        if self.has_shapes or self.has_particles or self.has_triangle_mesh:
            if refit_bvh:
                self.refit_bvh()
            width = camera_rays.shape[2]
            height = camera_rays.shape[1]
            num_cameras = camera_rays.shape[0]

            assert camera_transforms.shape == (num_cameras, self.num_worlds), (
                f"camera_transforms size must match {num_cameras} x {self.num_worlds}"
            )

            assert camera_rays.shape == (num_cameras, height, width, 2), (
                f"camera_rays size must match {num_cameras} x {height} x {width} x 2"
            )

            if color_image is not None:
                assert color_image.shape == (self.num_worlds, num_cameras, height, width), (
                    f"color_image size must match {self.num_worlds} x {num_cameras} x {height} x {width}"
                )
                if clear_data is not None and clear_data.clear_color is not None:
                    color_image.fill_(wp.uint32(clear_data.clear_color))

            if depth_image is not None:
                assert depth_image.shape == (self.num_worlds, num_cameras, height, width), (
                    f"depth_image size must match {self.num_worlds} x {num_cameras} x {height} x {width}"
                )
                if clear_data is not None and clear_data.clear_depth is not None:
                    depth_image.fill_(wp.float32(clear_data.clear_depth))

            if shape_index_image is not None:
                assert shape_index_image.shape == (self.num_worlds, num_cameras, height, width), (
                    f"shape_index_image size must match {self.num_worlds} x {num_cameras} x {height} x {width}"
                )
                if clear_data is not None and clear_data.clear_shape_index is not None:
                    shape_index_image.fill_(wp.uint32(clear_data.clear_shape_index))

            if normal_image is not None:
                assert normal_image.shape == (self.num_worlds, num_cameras, height, width), (
                    f"normal_image size must match {self.num_worlds} x {num_cameras} x {height} x {width}"
                )
                if clear_data is not None and clear_data.clear_normal is not None:
                    normal_image.fill_(clear_data.clear_normal)

            if albedo_image is not None:
                assert albedo_image.shape == (self.num_worlds, num_cameras, height, width), (
                    f"albedo_image size must match {self.num_worlds} x {num_cameras} x {height} x {width}"
                )
                if clear_data is not None and clear_data.clear_albedo is not None:
                    albedo_image.fill_(wp.uint32(clear_data.clear_albedo))

            if self.options.render_order == RenderOrder.TILED:
                assert width % self.options.tile_width == 0, "render width must be a multiple of tile_width"
                assert height % self.options.tile_height == 0, "render height must be a multiple of tile_height"

            # Reshaping output images to one dimension, slightly improves performance in the Kernel.
            if color_image is not None:
                color_image = color_image.reshape(self.num_worlds * num_cameras * width * height)
            if depth_image is not None:
                depth_image = depth_image.reshape(self.num_worlds * num_cameras * width * height)
            if shape_index_image is not None:
                shape_index_image = shape_index_image.reshape(self.num_worlds * num_cameras * width * height)
            if normal_image is not None:
                normal_image = normal_image.reshape(self.num_worlds * num_cameras * width * height)
            if albedo_image is not None:
                albedo_image = albedo_image.reshape(self.num_worlds * num_cameras * width * height)

            wp.launch(
                kernel=render_megakernel,
                dim=(self.num_worlds * num_cameras * width * height),
                inputs=[
                    # Model and Options
                    self.num_worlds,
                    num_cameras,
                    self.num_lights,
                    width,
                    height,
                    self.options.render_order,
                    self.options.tile_width,
                    self.options.tile_height,
                    self.options.enable_shadows,
                    self.options.enable_textures,
                    self.options.enable_ambient_lighting,
                    self.options.enable_particles and self.has_particles,
                    self.options.enable_backface_culling,
                    self.options.enable_global_world,
                    self.options.max_distance,
                    # Camera
                    camera_rays,
                    camera_transforms,
                    # Shape BVH
                    self.num_shapes_enabled,
                    self.bvh_shapes.id if self.bvh_shapes else 0,
                    self.bvh_shapes_group_roots,
                    # Shapes
                    self.shape_enabled,
                    self.shape_types,
                    self.shape_mesh_indices,
                    self.shape_materials,
                    self.shape_sizes,
                    self.shape_colors,
                    self.shape_transforms,
                    # Meshes
                    self.mesh_ids,
                    self.mesh_face_offsets,
                    self.mesh_face_vertices,
                    self.mesh_texcoord,
                    self.mesh_texcoord_offsets,
                    # Particle BVH
                    self.num_particles_total,
                    self.bvh_particles.id if self.bvh_particles else 0,
                    self.bvh_particles_group_roots,
                    # Particles
                    self.particles_position,
                    self.particles_radius,
                    # Triangle Mesh
                    self.triangle_mesh.id if self.triangle_mesh is not None else 0,
                    # Textures
                    self.material_texture_ids,
                    self.material_texture_repeat,
                    self.material_rgba,
                    self.texture_offsets,
                    self.texture_data,
                    self.texture_height,
                    self.texture_width,
                    # Lights
                    self.lights_active,
                    self.lights_type,
                    self.lights_cast_shadow,
                    self.lights_position,
                    self.lights_orientation,
                    # Outputs
                    color_image is not None,
                    depth_image is not None,
                    shape_index_image is not None,
                    normal_image is not None,
                    albedo_image is not None,
                    color_image,
                    depth_image,
                    shape_index_image,
                    normal_image,
                    albedo_image,
                ],
                device=self.device,
            )

    def __compute_bvh_shape_bounds(self):
        wp.launch(
            kernel=compute_shape_bvh_bounds,
            dim=self.num_shapes_enabled,
            inputs=[
                self.num_shapes_enabled,
                self.num_worlds_total,
                self.shape_world_index,
                self.shape_enabled,
                self.shape_types,
                self.shape_mesh_indices,
                self.shape_sizes,
                self.shape_transforms,
                self.mesh_bounds,
                self.bvh_shapes_lowers,
                self.bvh_shapes_uppers,
                self.bvh_shapes_groups,
            ],
            device=self.device,
        )

    def __compute_bvh_particle_bounds(self):
        wp.launch(
            kernel=compute_particle_bvh_bounds,
            dim=self.num_particles_total,
            inputs=[
                self.num_particles_total,
                self.num_worlds_total,
                self.particles_world_index,
                self.particles_position,
                self.particles_radius,
                self.bvh_particles_lowers,
                self.bvh_particles_uppers,
                self.bvh_particles_groups,
            ],
            device=self.device,
        )

    @property
    def num_worlds_total(self) -> int:
        if self.options.enable_global_world:
            return self.num_worlds + 1
        return self.num_worlds

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
    def has_shapes(self) -> bool:
        return self.num_shapes_enabled > 0

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
