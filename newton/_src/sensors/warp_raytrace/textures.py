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

from .types import GeomType


@wp.func
def sample_texture_2d(
    uv: wp.vec2f, width: wp.int32, height: wp.int32, texture_offsets: wp.int32, texture_data: wp.array(dtype=wp.uint32)
) -> wp.vec3f:
    ix = wp.min(width - 1, wp.int32(uv[0] * wp.float32(width)))
    it = wp.min(height - 1, wp.int32(uv[1] * wp.float32(height)))
    linear_idx = texture_offsets + (it * width + ix)
    packed_rgba = texture_data[linear_idx]
    r = wp.float32((packed_rgba >> wp.uint32(16)) & wp.uint32(0xFF)) / 255.0
    g = wp.float32((packed_rgba >> wp.uint32(8)) & wp.uint32(0xFF)) / 255.0
    b = wp.float32(packed_rgba & wp.uint32(0xFF)) / 255.0
    return wp.vec3f(r, g, b)


@wp.func
def sample_texture_plane(
    hit_point: wp.vec3f,
    geom_transform: wp.transformf,
    material_texture_repeat: wp.vec2f,
    texture_offsets: wp.int32,
    texture_data: wp.array(dtype=wp.uint32),
    texture_height: wp.int32,
    texture_width: wp.int32,
) -> wp.vec3f:
    inv_transform = wp.transform_inverse(geom_transform)
    local = wp.transform_point(inv_transform, hit_point)
    u = local[0] * material_texture_repeat[0]
    v = local[1] * material_texture_repeat[1]
    u = u - wp.floor(u)
    v = v - wp.floor(v)
    v = 1.0 - v
    return sample_texture_2d(wp.vec2f(u, v), texture_width, texture_height, texture_offsets, texture_data)


@wp.func
def sample_texture_mesh(
    bary_u: wp.float32,
    bary_v: wp.float32,
    uv_baseadr: wp.int32,
    v_idx: wp.vec3i,
    mesh_texcoord: wp.array(dtype=wp.vec2f),
    material_texture_repeat: wp.vec2f,
    texture_offsets: wp.int32,
    texture_data: wp.array(dtype=wp.uint32),
    texture_height: wp.int32,
    texture_width: wp.int32,
) -> wp.vec3f:
    bw = 1.0 - bary_u - bary_v
    uv0 = mesh_texcoord[uv_baseadr + v_idx.x]
    uv1 = mesh_texcoord[uv_baseadr + v_idx.y]
    uv2 = mesh_texcoord[uv_baseadr + v_idx.z]
    uv = uv0 * bw + uv1 * bary_u + uv2 * bary_v
    u = uv[0] * material_texture_repeat[0]
    v = uv[1] * material_texture_repeat[1]
    u = u - wp.floor(u)
    v = v - wp.floor(v)
    v = 1.0 - v
    return sample_texture_2d(
        wp.vec2f(u, v),
        texture_width,
        texture_height,
        texture_offsets,
        texture_data,
    )


@wp.func
def sample_texture(
    world_id: wp.int32,
    geom_id: wp.uint32,
    geom_type: wp.array(dtype=wp.int32),
    geom_material_id: wp.int32,
    material_texture_id: wp.int32,
    material_texture_repeat: wp.vec2f,
    texture_offsets: wp.int32,
    texture_data: wp.array(dtype=wp.uint32),
    texture_height: wp.int32,
    texture_width: wp.int32,
    geom_transforms: wp.transformf,
    mesh_face_offsets: wp.array(dtype=wp.int32),
    mesh_face_vertices: wp.array(dtype=wp.vec3i),
    mesh_texcoord: wp.array(dtype=wp.vec2f),
    mesh_texcoord_offsets: wp.array(dtype=wp.int32),
    hit_point: wp.vec3f,
    u: wp.float32,
    v: wp.float32,
    f: wp.int32,
    mesh_id: wp.int32,
) -> wp.vec3f:
    tex_color = wp.vec3f(1.0, 1.0, 1.0)

    if geom_material_id == -1 or material_texture_id == -1:
        return tex_color

    if geom_type[geom_id] == GeomType.PLANE:
        tex_color = sample_texture_plane(
            hit_point,
            geom_transforms,
            material_texture_repeat,
            texture_offsets,
            texture_data,
            texture_height,
            texture_width,
        )

    if geom_type[geom_id] == GeomType.MESH:
        if f < 0 or mesh_id < 0 or not mesh_texcoord_offsets.shape[0]:
            return tex_color

        uv_base = mesh_texcoord_offsets[mesh_id]

        if mesh_texcoord.shape[0] <= uv_base:
            return tex_color

        tex_color = sample_texture_mesh(
            u,
            v,
            uv_base,
            # mesh_face[mesh_faceadr[mesh_id] + base_face + f],
            wp.vec3i(f * 3 + 2, f * 3 + 0, f * 3 + 1),
            mesh_texcoord,
            material_texture_repeat,
            texture_offsets,
            texture_data,
            texture_height,
            texture_width,
        )

    return tex_color
