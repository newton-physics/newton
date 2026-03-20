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
def flip_v(uv: wp.vec2f) -> wp.vec2f:
    return wp.vec2f(uv[0], 1.0 - uv[1])


@wp.func
def sample_texture_2d(uv: wp.vec2f, texture_data: TextureData) -> wp.vec3f:
    color = wp.texture_sample(texture_data.texture, uv, dtype=wp.vec4f)
    return wp.vec3f(color[0], color[1], color[2])


@wp.func
def sample_texture_plane(
    hit_point: wp.vec3f,
    shape_transform: wp.transformf,
    texture_data: TextureData,
) -> wp.vec3f:
    inv_transform = wp.transform_inverse(shape_transform)
    local = wp.transform_point(inv_transform, hit_point)
    uv = wp.vec2f(local[0], local[1])
    return sample_texture_2d(flip_v(wp.cw_mul(uv, texture_data.repeat)), texture_data)


@wp.func
def sample_texture_mesh(
    bary_u: wp.float32,
    bary_v: wp.float32,
    face_id: wp.int32,
    mesh_id: wp.uint64,
    mesh_data: MeshData,
    texture_data: TextureData,
) -> wp.vec3f:
    bary_w = 1.0 - bary_u - bary_v
    uv0 = wp.mesh_get_index(mesh_id, face_id * 3 + 0)
    uv1 = wp.mesh_get_index(mesh_id, face_id * 3 + 1)
    uv2 = wp.mesh_get_index(mesh_id, face_id * 3 + 2)
    uv = mesh_data.uvs[uv0] * bary_u + mesh_data.uvs[uv1] * bary_v + mesh_data.uvs[uv2] * bary_w
    return sample_texture_2d(flip_v(wp.cw_mul(uv, texture_data.repeat)), texture_data)


@wp.func
def sample_texture(
    shape_type: wp.int32,
    shape_transform: wp.transformf,
    texture_data: wp.array(dtype=TextureData),
    texture_index: wp.int32,
    mesh_id: wp.uint64,
    mesh_data: wp.array(dtype=MeshData),
    mesh_data_index: wp.int32,
    hit_point: wp.vec3f,
    bary_u: wp.float32,
    bary_v: wp.float32,
    face_id: wp.int32,
) -> wp.vec3f:
    DEFAULT_RETURN = wp.vec3f(1.0, 1.0, 1.0)

    if texture_index == -1:
        return DEFAULT_RETURN

    if shape_type == GeoType.PLANE:
        return sample_texture_plane(hit_point, shape_transform, texture_data[texture_index])

    if shape_type == GeoType.MESH:
        if face_id < 0 or mesh_data_index < 0:
            return DEFAULT_RETURN

        if mesh_data[mesh_data_index].uvs.shape[0] == 0:
            return DEFAULT_RETURN

        return sample_texture_mesh(
            bary_u, bary_v, face_id, mesh_id, mesh_data[mesh_data_index], texture_data[texture_index]
        )

    return DEFAULT_RETURN
