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

from ...geometry import GeoType


@wp.func
def sample_texture_2d(
    uv: wp.vec2f, width: wp.int32, height: wp.int32, texture_offsets: wp.int32, texture_data: wp.array(dtype=wp.uint32)
) -> wp.vec3f:
    px = wp.min(width - 1, wp.int32(uv[0] * wp.float32(width)))
    py = wp.min(height - 1, wp.int32(uv[1] * wp.float32(height)))
    linear_idx = texture_offsets + (py * width + px)
    packed_rgba = texture_data[linear_idx]
    b = wp.float32((packed_rgba >> wp.uint32(16)) & wp.uint32(0xFF)) / 255.0
    g = wp.float32((packed_rgba >> wp.uint32(8)) & wp.uint32(0xFF)) / 255.0
    r = wp.float32(packed_rgba & wp.uint32(0xFF)) / 255.0
    return wp.vec3f(r, g, b)


@wp.func
def sample_texture_plane(
    hit_point: wp.vec3f,
    shape_transform: wp.transformf,
    texture_repeat: wp.vec2f,
    texture_offsets: wp.int32,
    texture_data: wp.array(dtype=wp.uint32),
    texture_height: wp.int32,
    texture_width: wp.int32,
) -> wp.vec3f:
    inv_transform = wp.transform_inverse(shape_transform)
    local = wp.transform_point(inv_transform, hit_point)
    u = local[0] * texture_repeat[0]
    v = local[1] * texture_repeat[1]
    u = u - wp.floor(u)
    v = v - wp.floor(v)
    v = 1.0 - v
    return sample_texture_2d(wp.vec2f(u, v), texture_width, texture_height, texture_offsets, texture_data)


@wp.func
def sample_texture_mesh(
    bary_u: wp.float32,
    bary_v: wp.float32,
    uv_offset: wp.int32,
    face_id: wp.int32,
    mesh_texcoord: wp.array(dtype=wp.vec2f),
    texture_repeat: wp.vec2f,
    texture_offsets: wp.int32,
    texture_data: wp.array(dtype=wp.uint32),
    texture_height: wp.int32,
    texture_width: wp.int32,
) -> wp.vec3f:
    bw = 1.0 - bary_u - bary_v
    uv0 = mesh_texcoord[uv_offset + face_id * 3 + 2]
    uv1 = mesh_texcoord[uv_offset + face_id * 3 + 0]
    uv2 = mesh_texcoord[uv_offset + face_id * 3 + 1]
    uv = uv0 * bw + uv1 * bary_u + uv2 * bary_v
    u = uv[0] * texture_repeat[0]
    v = uv[1] * texture_repeat[1]
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
    shape_type: wp.int32,
    shape_transform: wp.transformf,
    texture_index: wp.int32,
    texture_repeat: wp.vec2f,
    texture_offsets: wp.int32,
    texture_data: wp.array(dtype=wp.uint32),
    texture_height: wp.int32,
    texture_width: wp.int32,
    mesh_texcoord: wp.array(dtype=wp.vec2f),
    mesh_texcoord_offsets: wp.array(dtype=wp.int32),
    hit_point: wp.vec3f,
    u: wp.float32,
    v: wp.float32,
    face_id: wp.int32,
    mesh_id: wp.int32,
) -> wp.vec3f:
    tex_color = wp.vec3f(1.0, 1.0, 1.0)

    if texture_index == -1:
        return tex_color

    if shape_type == GeoType.PLANE:
        tex_color = sample_texture_plane(
            hit_point,
            shape_transform,
            texture_repeat,
            texture_offsets,
            texture_data,
            texture_height,
            texture_width,
        )

    if shape_type == GeoType.MESH:
        if face_id < 0 or mesh_id < 0 or not mesh_texcoord_offsets.shape[0]:
            return tex_color

        uv_offset = mesh_texcoord_offsets[mesh_id]
        if uv_offset < 0:
            return tex_color

        if mesh_texcoord.shape[0] <= uv_offset:
            return tex_color

        tex_color = sample_texture_mesh(
            u,
            v,
            uv_offset,
            face_id,
            mesh_texcoord,
            texture_repeat,
            texture_offsets,
            texture_data,
            texture_height,
            texture_width,
        )

    return tex_color
