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

TILE_W: wp.int32 = 32
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


def render_megakernel(rc: RenderContext):
    total_views = rc.num_worlds * rc.num_cameras
    total_pixels = rc.width * rc.height
    num_view_groups = (total_views + MAX_NUM_VIEWS_PER_THREAD - 1) // MAX_NUM_VIEWS_PER_THREAD
    if num_view_groups == 0:
        return

    if rc.render_rgb:
        rc.output.color_image.fill_(wp.uint32(BACKGROUND_COLOR))

    if rc.render_depth:
        rc.output.depth_image.fill_(wp.float32(0.0))

    @wp.kernel(enable_backward=False)
    def _render_megakernel(
        # Model and Options
        num_worlds: wp.int32,
        num_cameras: wp.int32,
        num_geom_in_bvh: wp.int32,
        img_width: wp.int32,
        img_height: wp.int32,
        use_shadows: wp.bool,
        # Camera
        camera_rays: wp.array(dtype=wp.vec3f, ndim=4),
        camera_positions: wp.array(dtype=wp.vec3f),
        camera_orientations: wp.array(dtype=wp.mat33f),
        # BVH
        bvh_id: wp.uint64,
        group_roots: wp.array(dtype=wp.int32),
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
        # Textures
        mat_texid: wp.array(dtype=wp.int32),
        mat_texrepeat: wp.array(dtype=wp.vec2f),
        mat_rgba: wp.array(dtype=wp.vec4f),
        tex_adr: wp.array(dtype=wp.int32),
        tex_data: wp.array(dtype=wp.uint32),
        tex_height: wp.array(dtype=wp.int32),
        tex_width: wp.array(dtype=wp.int32),
        # Lights
        light_active: wp.array(dtype=wp.bool),
        light_type: wp.array(dtype=wp.int32),
        light_cast_shadow: wp.array(dtype=wp.bool),
        light_positions: wp.array(dtype=wp.vec3f),
        light_orientations: wp.array(dtype=wp.vec3f),
        # Data
        geom_positions: wp.array(dtype=wp.vec3f),
        geom_orientations: wp.array(dtype=wp.mat33),
        # Output
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
            cam_idx = view % num_cameras

            ray_origin_world = camera_positions[cam_idx] + camera_rays[cam_idx, py, px, 0]
            ray_dir_world = camera_orientations[cam_idx] @ camera_rays[cam_idx, py, px, 1]

            geom_id, dist, normal, u, v, f, mesh_id = ray_cast.closest_hit(
                bvh_id,
                group_roots,
                world_id,
                num_geom_in_bvh,
                geom_enabled,
                geom_types,
                geom_mesh_indices,
                geom_sizes,
                mesh_ids,
                geom_positions,
                geom_orientations,
                ray_origin_world,
                ray_dir_world,
            )

            # Early Out
            if geom_id == -1:
                continue

            if wp.static(rc.render_depth):
                out_depth[world_id, cam_idx, mapped_idx] = dist

            if not wp.static(rc.render_rgb):
                continue

            # Shade the pixel
            hit_point = ray_origin_world + ray_dir_world * dist

            if geom_materials[geom_id] == -1:
                color = geom_colors[geom_id]
            else:
                color = mat_rgba[geom_materials[geom_id]]

            base_color = wp.vec3f(color[0], color[1], color[2])
            out_color = wp.vec3f()

            if wp.static(rc.use_textures):
                mat_id = geom_materials[geom_id]
                if mat_id > -1:
                    tex_id = mat_texid[mat_id]
                    if tex_id > -1:
                        tex_color = textures.sample_texture(
                            world_id,
                            geom_id,
                            geom_types,
                            mat_id,
                            tex_id,
                            mat_texrepeat[mat_id],
                            tex_adr[tex_id],
                            tex_data,
                            tex_height[tex_id],
                            tex_width[tex_id],
                            geom_positions[geom_id],
                            geom_orientations[geom_id],
                            mesh_face_offsets,
                            mesh_face_vertices,
                            mesh_texcoord,
                            mesh_texcoord_offsets,
                            hit_point,
                            u,
                            v,
                            f,
                            mesh_id,
                        )

                        base_color = wp.vec3f(
                            base_color[0] * tex_color[0],
                            base_color[1] * tex_color[1],
                            base_color[2] * tex_color[2],
                        )

            if wp.static(rc.use_ambient_lighting):
                up = wp.vec3f(0.0, 0.0, 1.0)
                len_n = wp.length(normal)
                n = normal if len_n > 0.0 else up
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
            for light_idx in range(wp.static(rc.num_lights)):
                light_contribution = lighting.compute_lighting(
                    use_shadows,
                    bvh_id,
                    group_roots,
                    num_geom_in_bvh,
                    geom_enabled,
                    world_id,
                    light_active[light_idx],
                    light_type[light_idx],
                    light_cast_shadow[light_idx],
                    light_positions[light_idx],
                    light_orientations[light_idx],
                    normal,
                    geom_types,
                    geom_mesh_indices,
                    geom_sizes,
                    mesh_ids,
                    geom_positions,
                    geom_orientations,
                    hit_point,
                )
                out_color = out_color + base_color * light_contribution

            out_color = wp.min(wp.max(out_color, wp.vec3f(0.0)), wp.vec3f(1.0))

            if wp.static(rc.render_rgb):
                out_pixels[world_id, cam_idx, mapped_idx] = pack_rgba_to_uint32(
                    out_color[0] * 255.0,
                    out_color[1] * 255.0,
                    out_color[2] * 255.0,
                    255.0,
                )

    wp.launch(
        kernel=_render_megakernel,
        dim=(num_view_groups * total_pixels),
        inputs=[
            # Model and Options
            rc.num_worlds,
            rc.num_cameras,
            rc.num_geom_in_bvh,
            rc.width,
            rc.height,
            rc.use_shadows,
            # Camera
            rc.camera_rays,
            rc.camera_positions,
            rc.camera_orientations,
            # BVH
            rc.bvh.id,
            rc.bvh_group_roots,
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
            # Textures
            rc.material_texture_ids,
            rc.material_texture_repeat,
            rc.material_rgba,
            rc.tex_adr,
            rc.tex_data,
            rc.tex_height,
            rc.tex_width,
            # Lights
            rc.lights_active,
            rc.lights_type,
            rc.lights_cast_shadow,
            rc.lights_position,
            rc.lights_orientation,
            # Data
            rc.geom_positions,
            rc.geom_orientations,
        ],
        outputs=[
            rc.output.color_image,
            rc.output.depth_image,
        ],
        block_dim=THREADS_PER_TILE,
    )
