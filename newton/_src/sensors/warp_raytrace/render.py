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
    from .render_context import RenderContext


@wp.func
def tid_to_tile_coord(tid: wp.int32, num_worlds: wp.int32, num_cameras: wp.int32, tile_size: wp.int32, width: wp.int32):
    num_views_per_pixel = num_worlds * num_cameras
    num_pixels_per_tile = tile_size * tile_size
    num_tiles_per_row = width // tile_size

    pixel_idx = tid // num_views_per_pixel
    view_idx = tid % num_views_per_pixel

    world_id = view_idx % num_worlds
    camera_id = view_idx // num_worlds

    tile_local = pixel_idx % num_pixels_per_tile
    tile_offset = pixel_idx // num_pixels_per_tile

    tile_offset_x = tile_offset % num_tiles_per_row
    tile_offset_y = tile_offset // num_tiles_per_row

    py = tile_local % tile_size + tile_offset_y * tile_size
    px = tile_local // tile_size + tile_offset_x * tile_size

    return world_id, camera_id, py, px


@wp.func
def tid_to_pixel_coord(tid: wp.int32, num_worlds: wp.int32, num_cameras: wp.int32, width: wp.int32):
    num_views_per_pixel = num_worlds * num_cameras

    pixel_idx = tid // num_views_per_pixel
    view_idx = tid % num_views_per_pixel

    world_id = view_idx % num_worlds
    camera_id = view_idx // num_worlds

    py = pixel_idx // width
    px = pixel_idx % width

    return world_id, camera_id, py, px


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
    has_global_world: wp.bool,
    max_distance: wp.float32,
    # Camera
    camera_rays: wp.array(dtype=wp.vec3f, ndim=4),
    camera_transforms: wp.array(dtype=wp.transformf, ndim=2),
    # Geometry BVH
    bvh_geom_size: wp.int32,
    bvh_geom_id: wp.uint64,
    bvh_geom_group_roots: wp.array(dtype=wp.int32),
    # Geometry
    geom_enabled: wp.array(dtype=wp.uint32),
    geom_types: wp.array(dtype=wp.int32),
    geom_mesh_indices: wp.array(dtype=wp.int32),
    geom_materials: wp.array(dtype=wp.int32),
    geom_sizes: wp.array(dtype=wp.vec3f),
    geom_colors: wp.array(dtype=wp.vec4f),
    mesh_ids: wp.array(dtype=wp.uint64),
    mesh_face_offsets: wp.array(dtype=wp.int32),
    mesh_face_vertices: wp.array(dtype=wp.vec3i),
    mesh_texcoord: wp.array(dtype=wp.vec2f),
    mesh_texcoord_offsets: wp.array(dtype=wp.int32),
    # Geometry BVH
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
    # Data
    geom_transforms: wp.array(dtype=wp.transformf),
    # Enabled Output
    render_color: wp.bool,
    render_depth: wp.bool,
    render_geom_id: wp.bool,
    render_normal: wp.bool,
    # Outputs
    out_pixels: wp.array3d(dtype=wp.uint32),
    out_depth: wp.array3d(dtype=wp.float32),
    out_geom_id: wp.array3d(dtype=wp.uint32),
    out_normal: wp.array3d(dtype=wp.vec3f),
):
    tid = wp.tid()

    if tile_rendering:
        world_id, camera_id, py, px = tid_to_tile_coord(tid, num_worlds, num_cameras, tile_size, img_width)
    else:
        world_id, camera_id, py, px = tid_to_pixel_coord(tid, num_worlds, num_cameras, img_width)

    if px >= img_width or py >= img_height:
        return

    out_index = py * img_width + px

    ray_origin_world = wp.transform_point(camera_transforms[camera_id, world_id], camera_rays[camera_id, py, px, 0])
    ray_dir_world = wp.transform_vector(camera_transforms[camera_id, world_id], camera_rays[camera_id, py, px, 1])

    closest_hit = ray_cast.closest_hit(
        bvh_geom_size,
        bvh_geom_id,
        bvh_geom_group_roots,
        bvh_particles_size,
        bvh_particles_id,
        bvh_particles_group_roots,
        world_id,
        has_global_world,
        enable_particles,
        max_distance,
        geom_enabled,
        geom_types,
        geom_mesh_indices,
        geom_sizes,
        mesh_ids,
        geom_transforms,
        particles_position,
        particles_radius,
        triangle_mesh_id,
        ray_origin_world,
        ray_dir_world,
    )

    # Early Out
    if closest_hit.geom_id == ray_cast.NO_HIT_GEOM_ID:
        return

    if render_depth:
        out_depth[world_id, camera_id, out_index] = closest_hit.distance

    if render_normal:
        out_normal[world_id, camera_id, out_index] = closest_hit.normal

    if render_geom_id:
        out_geom_id[world_id, camera_id, out_index] = closest_hit.geom_id

    if not render_color:
        return

    # Shade the pixel
    hit_point = ray_origin_world + ray_dir_world * closest_hit.distance

    color = wp.vec4f(1.0)
    if closest_hit.geom_id < ray_cast.MAX_GEOM_ID:
        color = geom_colors[closest_hit.geom_id]
        if geom_materials[closest_hit.geom_id] > -1:
            color = wp.cw_mul(color, material_rgba[geom_materials[closest_hit.geom_id]])

    base_color = wp.vec3f(color[0], color[1], color[2])
    out_color = wp.vec3f(0.0)

    if enable_textures and closest_hit.geom_id < ray_cast.MAX_GEOM_ID:
        mat_id = geom_materials[closest_hit.geom_id]
        if mat_id > -1:
            tex_id = material_texture_ids[mat_id]
            if tex_id > -1:
                tex_color = textures.sample_texture(
                    world_id,
                    closest_hit.geom_id,
                    geom_types,
                    mat_id,
                    tex_id,
                    material_texture_repeat[mat_id],
                    texture_offsets[tex_id],
                    texture_data,
                    texture_height[tex_id],
                    texture_width[tex_id],
                    geom_transforms[closest_hit.geom_id],
                    mesh_face_offsets,
                    mesh_face_vertices,
                    mesh_texcoord,
                    mesh_texcoord_offsets,
                    hit_point,
                    closest_hit.bary_u,
                    closest_hit.bary_v,
                    closest_hit.face_idx,
                    closest_hit.geom_mesh_id,
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
    for light_idx in range(num_lights):
        light_contribution = lighting.compute_lighting(
            enable_shadows,
            bvh_geom_size,
            bvh_geom_id,
            bvh_geom_group_roots,
            bvh_particles_size,
            bvh_particles_id,
            bvh_particles_group_roots,
            geom_enabled,
            world_id,
            has_global_world,
            enable_particles,
            light_active[light_idx],
            light_type[light_idx],
            light_cast_shadow[light_idx],
            light_positions[light_idx],
            light_orientations[light_idx],
            closest_hit.normal,
            geom_types,
            geom_mesh_indices,
            geom_sizes,
            mesh_ids,
            geom_transforms,
            particles_position,
            particles_radius,
            triangle_mesh_id,
            hit_point,
        )
        out_color = out_color + base_color * light_contribution

    out_color = wp.min(wp.max(out_color, wp.vec3f(0.0)), wp.vec3f(1.0))

    out_pixels[world_id, camera_id, out_index] = pack_rgba_to_uint32(
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
    geom_id_image: wp.array(dtype=wp.uint32, ndim=3) | None,
    normal_image: wp.array(dtype=wp.vec3f, ndim=3) | None,
    clear_color: int | None,
    clear_depth: float | None,
    clear_geom_id: int | None,
    clear_normal: wp.vec3f | None,
):
    if rc.tile_rendering:
        assert rc.width % rc.tile_size == 0, "render width must be a multiple of tile_size"
        assert rc.height % rc.tile_size == 0, "render height must be a multiple of tile_size"

    if clear_color is not None and color_image is not None:
        color_image.fill_(wp.uint32(clear_color))

    if clear_depth is not None and depth_image is not None:
        depth_image.fill_(wp.float32(clear_depth))

    if clear_geom_id is not None and geom_id_image is not None:
        geom_id_image.fill_(wp.uint32(clear_geom_id))

    if clear_normal is not None and normal_image is not None:
        normal_image.fill_(clear_normal)

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
            rc.tile_size,
            rc.tile_rendering,
            rc.enable_shadows,
            rc.enable_textures,
            rc.enable_ambient_lighting,
            rc.enable_particles,
            rc.has_global_world,
            rc.max_distance,
            # Camera
            camera_rays,
            camera_transforms,
            # Geometry BVH
            rc.num_geoms,
            rc.bvh_geom.id if rc.bvh_geom else 0,
            rc.bvh_geom_group_roots,
            # Geometry
            rc.geom_enabled,
            rc.geom_types,
            rc.geom_mesh_indices,
            rc.geom_materials,
            rc.geom_sizes,
            rc.geom_colors,
            rc.mesh_ids,
            rc.mesh_face_offsets,
            rc.mesh_face_vertices,
            rc.mesh_texcoord,
            rc.mesh_texcoord_offsets,
            # Particle BVH
            rc.particles_position.shape[0] if rc.particles_position else 0,
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
            # Data
            rc.geom_transforms,
            # Outputs
            color_image is not None,
            depth_image is not None,
            geom_id_image is not None,
            normal_image is not None,
            color_image,
            depth_image,
            geom_id_image,
            normal_image,
        ],
    )
