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

from typing import TYPE_CHECKING

import warp as wp

from ...geometry import Gaussian, GeoType
from . import lighting, raytrace, textures, tiling
from .types import MeshData, RenderOrder, TextureData

if TYPE_CHECKING:
    from .render_context import RenderContext


def create_kernel(
    config: RenderContext.Config, state: RenderContext.State, clear_data: RenderContext.ClearData
) -> wp.kernel:
    compute_lighting = lighting.create_compute_lighting_function(config, state)

    if state.render_color or state.render_normal:
        raytrace_closest_hit = raytrace.create_closest_hit_function(config, state)
    else:
        raytrace_closest_hit = raytrace.create_closest_hit_depth_only_function(config, state)

    @wp.kernel(enable_backward=False)
    def render_megakernel(
        # Model and Config
        world_count: wp.int32,
        camera_count: wp.int32,
        light_count: wp.int32,
        img_width: wp.int32,
        img_height: wp.int32,
        # Camera
        camera_rays: wp.array(dtype=wp.vec3f, ndim=4),
        camera_transforms: wp.array(dtype=wp.transformf, ndim=2),
        # Shapes BVH
        bvh_shapes_size: wp.int32,
        bvh_shapes_id: wp.uint64,
        bvh_shapes_group_roots: wp.array(dtype=wp.int32),
        # Shapes
        shape_enabled: wp.array(dtype=wp.uint32),
        shape_types: wp.array(dtype=wp.int32),
        shape_sizes: wp.array(dtype=wp.vec3f),
        shape_colors: wp.array(dtype=wp.vec4f),
        shape_transforms: wp.array(dtype=wp.transformf),
        shape_source_ptr: wp.array(dtype=wp.uint64),
        shape_texture_ids: wp.array(dtype=wp.int32),
        shape_mesh_data_ids: wp.array(dtype=wp.int32),
        # Particle BVH
        bvh_particles_size: wp.int32,
        bvh_particles_id: wp.uint64,
        bvh_particles_group_roots: wp.array(dtype=wp.int32),
        # Particles
        particles_position: wp.array(dtype=wp.vec3f),
        particles_radius: wp.array(dtype=wp.float32),
        # Triangle Mesh:
        triangle_mesh_id: wp.uint64,
        # Meshes
        mesh_data: wp.array(dtype=MeshData),
        # Gaussians
        gaussians_data: wp.array(dtype=Gaussian.Data),
        # Textures
        texture_data: wp.array(dtype=TextureData),
        # Lights
        light_active: wp.array(dtype=wp.bool),
        light_type: wp.array(dtype=wp.int32),
        light_cast_shadow: wp.array(dtype=wp.bool),
        light_positions: wp.array(dtype=wp.vec3f),
        light_orientations: wp.array(dtype=wp.vec3f),
        # Outputs
        out_color: wp.array(dtype=wp.uint32),
        out_depth: wp.array(dtype=wp.float32),
        out_shape_index: wp.array(dtype=wp.uint32),
        out_normal: wp.array(dtype=wp.vec3f),
        out_albedo: wp.array(dtype=wp.uint32),
    ):
        tid = wp.tid()

        if wp.static(config.render_order == RenderOrder.PIXEL_PRIORITY):
            world_index, camera_index, py, px = tiling.tid_to_coord_pixel_priority(
                tid, world_count, camera_count, img_width
            )
        elif wp.static(config.render_order == RenderOrder.VIEW_PRIORITY):
            world_index, camera_index, py, px = tiling.tid_to_coord_view_priority(
                tid, camera_count, img_width, img_height
            )
        elif wp.static(config.render_order == RenderOrder.TILED):
            world_index, camera_index, py, px = tiling.tid_to_coord_tiled(
                tid, camera_count, img_width, img_height, wp.static(config.tile_width), wp.static(config.tile_height)
            )
        else:
            return

        if px >= img_width or py >= img_height:
            return

        pixels_per_camera = img_width * img_height
        pixels_per_world = camera_count * pixels_per_camera
        out_index = world_index * pixels_per_world + camera_index * pixels_per_camera + py * img_width + px

        camera_transform = camera_transforms[camera_index, world_index]
        ray_origin_world = wp.transform_point(camera_transform, camera_rays[camera_index, py, px, 0])
        ray_dir_world = wp.transform_vector(camera_transform, camera_rays[camera_index, py, px, 1])

        closest_hit = raytrace_closest_hit(
            bvh_shapes_size,
            bvh_shapes_id,
            bvh_shapes_group_roots,
            bvh_particles_size,
            bvh_particles_id,
            bvh_particles_group_roots,
            world_index,
            wp.static(config.max_distance),
            shape_enabled,
            shape_types,
            shape_sizes,
            shape_transforms,
            shape_source_ptr,
            shape_mesh_data_ids,
            mesh_data,
            particles_position,
            particles_radius,
            triangle_mesh_id,
            gaussians_data,
            ray_origin_world,
            ray_dir_world,
        )

        if closest_hit.shape_index == raytrace.NO_HIT_SHAPE_ID:
            if wp.static(state.render_color):
                out_color[out_index] = wp.uint32(wp.static(clear_data.clear_color))
            if wp.static(state.render_albedo):
                out_albedo[out_index] = wp.uint32(wp.static(clear_data.clear_albedo))
            if wp.static(state.render_depth):
                out_depth[out_index] = wp.float32(wp.static(clear_data.clear_depth))
            if wp.static(state.render_normal):
                out_normal[out_index] = wp.vec3f(
                    wp.static(clear_data.clear_normal[0]),
                    wp.static(clear_data.clear_normal[1]),
                    wp.static(clear_data.clear_normal[2]),
                )
            if wp.static(state.render_shape_index):
                out_shape_index[out_index] = wp.uint32(wp.static(clear_data.clear_shape_index))
            return

        if wp.static(state.render_depth):
            out_depth[out_index] = closest_hit.distance

        if wp.static(state.render_normal):
            out_normal[out_index] = closest_hit.normal

        if wp.static(state.render_shape_index):
            out_shape_index[out_index] = closest_hit.shape_index

        if not wp.static(state.render_color) and not wp.static(state.render_albedo):
            return

        is_gaussian = wp.bool(False)
        if closest_hit.shape_index < raytrace.MAX_SHAPE_ID:
            if shape_types[closest_hit.shape_index] == GeoType.GAUSSIAN:
                is_gaussian = wp.bool(True)

        albedo_color = wp.vec3f(0.0)

        if not is_gaussian:
            hit_point = ray_origin_world + ray_dir_world * closest_hit.distance

            color = wp.vec4f(1.0)
            if closest_hit.shape_index < raytrace.MAX_SHAPE_ID:
                color = shape_colors[closest_hit.shape_index]

            albedo_color = wp.vec3f(color[0], color[1], color[2])

            if wp.static(config.enable_textures) and closest_hit.shape_index < raytrace.MAX_SHAPE_ID:
                texture_index = shape_texture_ids[closest_hit.shape_index]
                if texture_index > -1:
                    tex_color = textures.sample_texture(
                        shape_types[closest_hit.shape_index],
                        shape_transforms[closest_hit.shape_index],
                        texture_data,
                        texture_index,
                        shape_source_ptr[closest_hit.shape_index],
                        mesh_data,
                        shape_mesh_data_ids[closest_hit.shape_index],
                        hit_point,
                        closest_hit.bary_u,
                        closest_hit.bary_v,
                        closest_hit.face_idx,
                    )

                    albedo_color = wp.cw_mul(albedo_color, tex_color)

        if wp.static(state.render_albedo):
            out_albedo[out_index] = tiling.pack_rgba_to_uint32(albedo_color, 1.0)

        if not wp.static(state.render_color):
            return

        shaded_color = closest_hit.color

        if not is_gaussian:
            if wp.static(config.enable_ambient_lighting):
                up = wp.vec3f(0.0, 0.0, 1.0)
                len_n = wp.length(closest_hit.normal)
                n = closest_hit.normal if len_n > 0.0 else up
                n = wp.normalize(n)
                hemispheric = 0.5 * (wp.dot(n, up) + 1.0)
                sky = wp.vec3f(0.4, 0.4, 0.45)
                ground = wp.vec3f(0.1, 0.1, 0.12)
                ambient_color = sky * hemispheric + ground * (1.0 - hemispheric)
                ambient_intensity = 0.5

                shaded_color = wp.cw_mul(albedo_color, ambient_color * ambient_intensity)

            # Apply lighting and shadows
            for light_index in range(light_count):
                light_contribution = compute_lighting(
                    world_index,
                    bvh_shapes_size,
                    bvh_shapes_id,
                    bvh_shapes_group_roots,
                    bvh_particles_size,
                    bvh_particles_id,
                    bvh_particles_group_roots,
                    shape_enabled,
                    shape_types,
                    shape_sizes,
                    shape_transforms,
                    shape_source_ptr,
                    light_active[light_index],
                    light_type[light_index],
                    light_cast_shadow[light_index],
                    light_positions[light_index],
                    light_orientations[light_index],
                    particles_position,
                    particles_radius,
                    triangle_mesh_id,
                    closest_hit.normal,
                    hit_point,
                )
                shaded_color = shaded_color + albedo_color * light_contribution

        out_color[out_index] = tiling.pack_rgba_to_uint32(shaded_color, 1.0)

    return render_megakernel
