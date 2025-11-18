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


MAX_NUM_VIEWS_PER_THREAD = 8

BACKGROUND_COLOR = 255 << 24 | int(0.4 * 255.0) << 16 | int(0.4 * 255.0) << 8 | int(0.4 * 255.0)

TILE_W: wp.int32 = 8
TILE_H: wp.int32 = 8
THREADS_PER_TILE: wp.int32 = TILE_W * TILE_H


@wp.func
def ceil_div(a: wp.int32, b: wp.int32):
    return (a + b - 1) // b


# Map linear thread id (per image) -> (px, py) using TILE_W x TILE_H tiles
@wp.func
def tile_coords(tid: wp.int32, width: wp.int32):
    tile_id = tid // THREADS_PER_TILE
    local = tid - tile_id * THREADS_PER_TILE

    u = local % TILE_W
    v = local // TILE_W

    tiles_x = ceil_div(width, TILE_W)
    tile_x = (tile_id % tiles_x) * TILE_W
    tile_y = (tile_id // tiles_x) * TILE_H

    i = tile_x + u
    j = tile_y + v
    return i, j


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
    enable_shadows: wp.bool,
    enable_textures: wp.bool,
    enable_ambient_lighting: wp.bool,
    enable_particles: wp.bool,
    has_global_world: wp.bool,
    max_distance: wp.float32,
    # Camera
    camera_rays: wp.array(dtype=wp.vec3f, ndim=4),
    camera_positions: wp.array(dtype=wp.vec3f),
    camera_orientations: wp.array(dtype=wp.mat33f),
    # Geometry BVH
    bvh_geom_size: wp.int32,
    bvh_geom_id: wp.uint64,
    bvh_geom_group_roots: wp.array(dtype=wp.int32),
    # Geometry
    geom_enabled: wp.array(dtype=wp.int32),
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
    geom_positions: wp.array(dtype=wp.vec3f),
    geom_orientations: wp.array(dtype=wp.mat33f),
    # Output
    render_color: wp.bool,
    render_depth: wp.bool,
    out_pixels: wp.array3d(dtype=wp.uint32),
    out_depth: wp.array3d(dtype=wp.float32),
):
    tid = wp.tid()

    if tid >= num_worlds * num_cameras * img_width * img_height:
        return

    total_views = num_worlds * num_cameras
    pixels_per_image = img_width * img_height
    num_view_groups = (total_views + MAX_NUM_VIEWS_PER_THREAD - 1) // MAX_NUM_VIEWS_PER_THREAD

    group_idx = tid // pixels_per_image
    pixel_idx = tid % pixels_per_image

    if group_idx >= num_view_groups:
        return

    px, py = tile_coords(pixel_idx, img_width)
    if px >= img_width or py >= img_height:
        return
    mapped_idx = py * img_width + px

    base_view = group_idx * MAX_NUM_VIEWS_PER_THREAD

    for i in range(MAX_NUM_VIEWS_PER_THREAD):
        view = base_view + i
        if view >= total_views:
            break

        world_id = view // num_cameras
        camera_id = view % num_cameras

        ray_origin_world = camera_positions[camera_id] + camera_rays[camera_id, py, px, 0]
        ray_dir_world = camera_orientations[camera_id] @ camera_rays[camera_id, py, px, 1]

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
            geom_positions,
            geom_orientations,
            particles_position,
            particles_radius,
            triangle_mesh_id,
            ray_origin_world,
            ray_dir_world,
        )

        # Early Out
        if closest_hit.geom_id == -1:
            continue

        if render_depth:
            out_depth[world_id, camera_id, mapped_idx] = closest_hit.distance

        if not render_color:
            continue

        # Shade the pixel
        hit_point = ray_origin_world + ray_dir_world * closest_hit.distance

        color = wp.vec4f(1.0)
        if closest_hit.geom_id > -1:
            color = geom_colors[closest_hit.geom_id]
            if geom_materials[closest_hit.geom_id] > -1:
                color = wp.cw_mul(color, material_rgba[geom_materials[closest_hit.geom_id]])

        base_color = wp.vec3f(color[0], color[1], color[2])
        out_color = wp.vec3f(0.0)

        if enable_textures and closest_hit.geom_id > -1:
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
                        geom_positions[closest_hit.geom_id],
                        geom_orientations[closest_hit.geom_id],
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
                geom_positions,
                geom_orientations,
                particles_position,
                particles_radius,
                triangle_mesh_id,
                hit_point,
            )
            out_color = out_color + base_color * light_contribution

        out_color = wp.min(wp.max(out_color, wp.vec3f(0.0)), wp.vec3f(1.0))

        out_pixels[world_id, camera_id, mapped_idx] = pack_rgba_to_uint32(
            out_color[0] * 255.0,
            out_color[1] * 255.0,
            out_color[2] * 255.0,
            255.0,
        )


def render_megakernel(
    rc: RenderContext,
    color_image: wp.array(dtype=wp.uint32, ndim=3) | None = None,
    depth_image: wp.array(dtype=wp.float32, ndim=3) | None = None,
    clear_images: bool = True,
):
    total_views = rc.num_worlds * rc.num_cameras
    total_pixels = rc.width * rc.height
    num_view_groups = (total_views + MAX_NUM_VIEWS_PER_THREAD - 1) // MAX_NUM_VIEWS_PER_THREAD
    if num_view_groups == 0:
        return

    if clear_images:
        if color_image is not None:
            color_image.fill_(wp.uint32(BACKGROUND_COLOR))

        if depth_image is not None:
            depth_image.fill_(wp.float32(0.0))

    wp.launch(
        kernel=_render_megakernel,
        dim=(num_view_groups * total_pixels),
        inputs=[
            # Model and Options
            rc.num_worlds,
            rc.num_cameras,
            rc.num_lights,
            rc.width,
            rc.height,
            rc.enable_shadows,
            rc.enable_textures,
            rc.enable_ambient_lighting,
            rc.enable_particles,
            rc.has_global_world,
            rc.max_distance,
            # Camera
            rc.camera_rays,
            rc.camera_positions,
            rc.camera_orientations,
            # Geometry BVH
            rc.num_geom_in_bvh,
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
            rc.geom_positions,
            rc.geom_orientations,
            # Outputs
            color_image is not None,
            depth_image is not None,
            color_image,
            depth_image,
        ],
        block_dim=THREADS_PER_TILE,
    )
