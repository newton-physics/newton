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

from . import lighting, ray_cast, textures

if TYPE_CHECKING:
    from .render_context import ClearData, RenderContext


@wp.func
def tid_to_tile_coord(tid: wp.int32, num_worlds: wp.int32, num_cameras: wp.int32, tile_size: wp.int32, width: wp.int32):
    num_views_per_pixel = num_worlds * num_cameras
    num_pixels_per_tile = tile_size * tile_size
    num_tiles_per_row = width // tile_size

    pixel_idx = tid // num_views_per_pixel
    view_idx = tid % num_views_per_pixel

    world_index = view_idx % num_worlds
    camera_index = view_idx // num_worlds

    tile_local = pixel_idx % num_pixels_per_tile
    tile_offset = pixel_idx // num_pixels_per_tile

    tile_offset_x = tile_offset % num_tiles_per_row
    tile_offset_y = tile_offset // num_tiles_per_row

    py = tile_local % tile_size + tile_offset_y * tile_size
    px = tile_local // tile_size + tile_offset_x * tile_size

    return world_index, camera_index, py, px


@wp.func
def tid_to_pixel_coord(tid: wp.int32, num_worlds: wp.int32, num_cameras: wp.int32, width: wp.int32):
    num_views_per_pixel = num_worlds * num_cameras

    pixel_idx = tid // num_views_per_pixel
    view_idx = tid % num_views_per_pixel

    world_index = view_idx % num_worlds
    camera_index = view_idx // num_worlds

    py = pixel_idx // width
    px = pixel_idx % width

    return world_index, camera_index, py, px


@wp.func
def pack_rgba_to_uint32(r: wp.float32, g: wp.float32, b: wp.float32, a: wp.float32) -> wp.uint32:
    """Pack RGBA values into a single uint32 for efficient memory access."""
    return (
        (wp.uint32(a) << wp.uint32(24))
        | (wp.uint32(b) << wp.uint32(16))
        | (wp.uint32(g) << wp.uint32(8))
        | wp.uint32(r)
    )


@wp.kernel(enable_backward=False)
def _render_megakernel(
    # Model and Options
    num_worlds: wp.int32,
    num_cameras: wp.int32,
    num_lights: wp.int32,
    img_width: wp.int32,
    img_height: wp.int32,
    tile_size: wp.int32,
    tile_rendering: wp.bool,
    enable_shadows: wp.bool,
    enable_textures: wp.bool,
    enable_ambient_lighting: wp.bool,
    enable_particles: wp.bool,
    enable_backface_culling: wp.bool,
    enable_global_world: wp.bool,
    max_distance: wp.float32,
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
    shape_mesh_indices: wp.array(dtype=wp.int32),
    shape_materials: wp.array(dtype=wp.int32),
    shape_sizes: wp.array(dtype=wp.vec3f),
    shape_colors: wp.array(dtype=wp.vec4f),
    shape_transforms: wp.array(dtype=wp.transformf),
    # Meshes
    mesh_ids: wp.array(dtype=wp.uint64),
    mesh_face_offsets: wp.array(dtype=wp.int32),
    mesh_face_vertices: wp.array(dtype=wp.vec3i),
    mesh_texcoord: wp.array(dtype=wp.vec2f),
    mesh_texcoord_offsets: wp.array(dtype=wp.int32),
    # Particle BVH
    bvh_particles_size: wp.int32,
    bvh_particles_id: wp.uint64,
    bvh_particles_group_roots: wp.array(dtype=wp.int32),
    # Particles
    particles_position: wp.array(dtype=wp.vec3f),
    particles_radius: wp.array(dtype=wp.float32),
    # Triangle Mesh:
    triangle_mesh_id: wp.uint64,
    # Materials
    material_texture_ids: wp.array(dtype=wp.int32),
    material_texture_repeat: wp.array(dtype=wp.vec2f),
    material_rgba: wp.array(dtype=wp.vec4f),
    # Textures
    texture_offsets: wp.array(dtype=wp.int32),
    texture_data: wp.array(dtype=wp.uint32),
    texture_height: wp.array(dtype=wp.int32),
    texture_width: wp.array(dtype=wp.int32),
    # Lights
    light_active: wp.array(dtype=wp.bool),
    light_type: wp.array(dtype=wp.int32),
    light_cast_shadow: wp.array(dtype=wp.bool),
    light_positions: wp.array(dtype=wp.vec3f),
    light_orientations: wp.array(dtype=wp.vec3f),
    # Enabled Output
    render_color: wp.bool,
    render_depth: wp.bool,
    render_shape_index: wp.bool,
    render_normal: wp.bool,
    # Outputs
    out_pixels: wp.array3d(dtype=wp.uint32),
    out_depth: wp.array3d(dtype=wp.float32),
    out_shape_index: wp.array3d(dtype=wp.uint32),
    out_normal: wp.array3d(dtype=wp.vec3f),
):
    tid = wp.tid()

    if tile_rendering:
        world_index, camera_index, py, px = tid_to_tile_coord(tid, num_worlds, num_cameras, tile_size, img_width)
    else:
        world_index, camera_index, py, px = tid_to_pixel_coord(tid, num_worlds, num_cameras, img_width)

    if px >= img_width or py >= img_height:
        return

    out_index = py * img_width + px

    ray_origin_world = wp.transform_point(
        camera_transforms[camera_index, world_index], camera_rays[camera_index, py, px, 0]
    )
    ray_dir_world = wp.transform_vector(
        camera_transforms[camera_index, world_index], camera_rays[camera_index, py, px, 1]
    )

    closest_hit = ray_cast.closest_hit(
        bvh_shapes_size,
        bvh_shapes_id,
        bvh_shapes_group_roots,
        bvh_particles_size,
        bvh_particles_id,
        bvh_particles_group_roots,
        world_index,
        enable_global_world,
        enable_particles,
        enable_backface_culling,
        max_distance,
        shape_enabled,
        shape_types,
        shape_mesh_indices,
        shape_sizes,
        shape_transforms,
        mesh_ids,
        particles_position,
        particles_radius,
        triangle_mesh_id,
        ray_origin_world,
        ray_dir_world,
    )

    if closest_hit.shape_index == ray_cast.NO_HIT_SHAPE_ID:
        return

    if render_depth:
        out_depth[world_index, camera_index, out_index] = closest_hit.distance

    if render_normal:
        out_normal[world_index, camera_index, out_index] = closest_hit.normal

    if render_shape_index:
        out_shape_index[world_index, camera_index, out_index] = closest_hit.shape_index

    if not render_color:
        return

    # Shade the pixel
    hit_point = ray_origin_world + ray_dir_world * closest_hit.distance

    color = wp.vec4f(1.0)
    if closest_hit.shape_index < ray_cast.MAX_SHAPE_ID:
        color = shape_colors[closest_hit.shape_index]
        if shape_materials[closest_hit.shape_index] > -1:
            color = wp.cw_mul(color, material_rgba[shape_materials[closest_hit.shape_index]])

    base_color = wp.vec3f(color[0], color[1], color[2])
    out_color = wp.vec3f(0.0)

    if enable_textures and closest_hit.shape_index < ray_cast.MAX_SHAPE_ID:
        material_index = shape_materials[closest_hit.shape_index]
        if material_index > -1:
            texture_index = material_texture_ids[material_index]
            if texture_index > -1:
                tex_color = textures.sample_texture(
                    shape_types[closest_hit.shape_index],
                    shape_transforms[closest_hit.shape_index],
                    material_index,
                    texture_index,
                    material_texture_repeat[material_index],
                    texture_offsets[texture_index],
                    texture_data,
                    texture_height[texture_index],
                    texture_width[texture_index],
                    mesh_face_offsets,
                    mesh_face_vertices,
                    mesh_texcoord,
                    mesh_texcoord_offsets,
                    hit_point,
                    closest_hit.bary_u,
                    closest_hit.bary_v,
                    closest_hit.face_idx,
                    closest_hit.shape_mesh_index,
                )

                base_color = wp.vec3f(
                    base_color[0] * tex_color[0],
                    base_color[1] * tex_color[1],
                    base_color[2] * tex_color[2],
                )

    if enable_ambient_lighting:
        up = wp.vec3f(0.0, 0.0, 1.0)
        len_n = wp.length(closest_hit.normal)
        n = closest_hit.normal if len_n > 0.0 else up
        n = wp.normalize(n)
        hemispheric = 0.5 * (wp.dot(n, up) + 1.0)
        sky = wp.vec3f(0.4, 0.4, 0.45)
        ground = wp.vec3f(0.1, 0.1, 0.12)
        ambient_color = sky * hemispheric + ground * (1.0 - hemispheric)
        ambient_intensity = 0.5
        out_color = wp.vec3f(
            base_color[0] * (ambient_color[0] * ambient_intensity),
            base_color[1] * (ambient_color[1] * ambient_intensity),
            base_color[2] * (ambient_color[2] * ambient_intensity),
        )

    # Apply lighting and shadows
    for light_index in range(num_lights):
        light_contribution = lighting.compute_lighting(
            enable_shadows,
            enable_particles,
            enable_backface_culling,
            world_index,
            enable_global_world,
            bvh_shapes_size,
            bvh_shapes_id,
            bvh_shapes_group_roots,
            bvh_particles_size,
            bvh_particles_id,
            bvh_particles_group_roots,
            shape_enabled,
            shape_types,
            shape_mesh_indices,
            shape_sizes,
            shape_transforms,
            mesh_ids,
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
        out_color = out_color + base_color * light_contribution

    out_color = wp.min(wp.max(out_color, wp.vec3f(0.0)), wp.vec3f(1.0))

    out_pixels[world_index, camera_index, out_index] = pack_rgba_to_uint32(
        out_color[0] * 255.0,
        out_color[1] * 255.0,
        out_color[2] * 255.0,
        255.0,
    )


def render_megakernel(
    rc: RenderContext,
    camera_transforms: wp.array(dtype=wp.transformf, ndim=2),
    camera_rays: wp.array(dtype=wp.vec3f, ndim=4),
    color_image: wp.array(dtype=wp.uint32, ndim=3) | None,
    depth_image: wp.array(dtype=wp.float32, ndim=3) | None,
    shape_index_image: wp.array(dtype=wp.uint32, ndim=3) | None,
    normal_image: wp.array(dtype=wp.vec3f, ndim=3) | None,
    clear_data: ClearData | None,
):
    if rc.options.tile_rendering:
        assert rc.width % rc.options.tile_size == 0, "render width must be a multiple of tile_size"
        assert rc.height % rc.options.tile_size == 0, "render height must be a multiple of tile_size"

    if clear_data is not None and clear_data.clear_color is not None and color_image is not None:
        color_image.fill_(wp.uint32(clear_data.clear_color))

    if clear_data is not None and clear_data.clear_depth is not None and depth_image is not None:
        depth_image.fill_(wp.float32(clear_data.clear_depth))

    if clear_data is not None and clear_data.clear_shape_index is not None and shape_index_image is not None:
        shape_index_image.fill_(wp.uint32(clear_data.clear_shape_index))

    if clear_data is not None and clear_data.clear_normal is not None and normal_image is not None:
        normal_image.fill_(clear_data.clear_normal)

    wp.launch(
        kernel=_render_megakernel,
        dim=rc.num_worlds * rc.num_cameras * rc.width * rc.height,
        inputs=[
            # Model and Options
            rc.num_worlds,
            rc.num_cameras,
            rc.num_lights,
            rc.width,
            rc.height,
            rc.options.tile_size,
            rc.options.tile_rendering,
            rc.options.enable_shadows,
            rc.options.enable_textures,
            rc.options.enable_ambient_lighting,
            rc.options.enable_particles and rc.has_particles,
            rc.options.enable_backface_culling,
            rc.options.enable_global_world,
            rc.options.max_distance,
            # Camera
            camera_rays,
            camera_transforms,
            # Shape BVH
            rc.num_shapes_enabled,
            rc.bvh_shapes.id if rc.bvh_shapes else 0,
            rc.bvh_shapes_group_roots,
            # Shapes
            rc.shape_enabled,
            rc.shape_types,
            rc.shape_mesh_indices,
            rc.shape_materials,
            rc.shape_sizes,
            rc.shape_colors,
            rc.shape_transforms,
            # Meshes
            rc.mesh_ids,
            rc.mesh_face_offsets,
            rc.mesh_face_vertices,
            rc.mesh_texcoord,
            rc.mesh_texcoord_offsets,
            # Particle BVH
            rc.num_particles_total,
            rc.bvh_particles.id if rc.bvh_particles else 0,
            rc.bvh_particles_group_roots,
            # Particles
            rc.particles_position,
            rc.particles_radius,
            # Triangle Mesh
            rc.triangle_mesh.id if rc.triangle_mesh is not None else 0,
            # Textures
            rc.material_texture_ids,
            rc.material_texture_repeat,
            rc.material_rgba,
            rc.texture_offsets,
            rc.texture_data,
            rc.texture_height,
            rc.texture_width,
            # Lights
            rc.lights_active,
            rc.lights_type,
            rc.lights_cast_shadow,
            rc.lights_position,
            rc.lights_orientation,
            # Outputs
            color_image is not None,
            depth_image is not None,
            shape_index_image is not None,
            normal_image is not None,
            color_image,
            depth_image,
            shape_index_image,
            normal_image,
        ],
    )
