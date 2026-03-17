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
from .types import MeshData, TextureData


@wp.func
def sample_texture_2d(uv: wp.vec2f, texture_data: TextureData) -> wp.vec3f:
    px = wp.clamp(wp.int32(uv[0] * wp.float32(texture_data.width)), 0, texture_data.width - 1)
    py = wp.clamp(wp.int32(uv[1] * wp.float32(texture_data.height)), 0, texture_data.height - 1)

    packed_rgba = texture_data.pixels[py * texture_data.width + px]
    r = wp.float32(packed_rgba & wp.uint32(0xFF)) / 255.0
    g = wp.float32((packed_rgba >> wp.uint32(8)) & wp.uint32(0xFF)) / 255.0
    b = wp.float32((packed_rgba >> wp.uint32(16)) & wp.uint32(0xFF)) / 255.0
    return wp.vec3f(r, g, b)


@wp.func
def sample_texture_plane(
    hit_point: wp.vec3f,
    shape_transform: wp.transformf,
    texture_data: TextureData,
) -> wp.vec3f:
    inv_transform = wp.transform_inverse(shape_transform)
    local = wp.transform_point(inv_transform, hit_point)
    u = local[0] * texture_data.repeat[0]
    v = local[1] * texture_data.repeat[1]
    u = u - wp.floor(u)
    v = v - wp.floor(v)
    v = 1.0 - v
    return sample_texture_2d(wp.vec2f(u, v), texture_data)


@wp.func
def sample_texture_mesh(
    bary_u: wp.float32,
    bary_v: wp.float32,
    face_id: wp.int32,
    mesh_data: MeshData,
    texture_data: TextureData,
) -> wp.vec3f:
    bw = 1.0 - bary_u - bary_v
    uv0 = mesh_data.uvs[face_id * 3 + 2]
    uv1 = mesh_data.uvs[face_id * 3 + 0]
    uv2 = mesh_data.uvs[face_id * 3 + 1]
    uv = uv0 * bw + uv1 * bary_u + uv2 * bary_v
    u = uv[0] * texture_data.repeat[0]
    v = uv[1] * texture_data.repeat[1]
    u = u - wp.floor(u)
    v = v - wp.floor(v)
    v = 1.0 - v
    return sample_texture_2d(wp.vec2f(u, v), texture_data)


@wp.func
def sample_texture(
    shape_type: wp.int32,
    shape_transform: wp.transformf,
    texture_data: wp.array(dtype=TextureData),
    texture_index: wp.int32,
    mesh_data: wp.array(dtype=MeshData),
    mesh_data_index: wp.int32,
    hit_point: wp.vec3f,
    u: wp.float32,
    v: wp.float32,
    face_id: wp.int32,
) -> wp.vec3f:
    tex_color = wp.vec3f(1.0, 1.0, 1.0)

    if texture_index == -1:
        return tex_color

    if shape_type == GeoType.PLANE:
        tex_color = sample_texture_plane(hit_point, shape_transform, texture_data[texture_index])

    if shape_type == GeoType.MESH:
        if face_id < 0 or mesh_data_index < 0:
            return tex_color

        if mesh_data[mesh_data_index].uvs.shape[0] == 0:
            return tex_color

        tex_color = sample_texture_mesh(u, v, face_id, mesh_data[mesh_data_index], texture_data[texture_index])

    return tex_color
