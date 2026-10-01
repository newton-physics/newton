# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ...geometry import GeoType
from .types import MeshData, TextureData, TextureProjectionMode

if TYPE_CHECKING:
    from .render_context import RenderContext


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
def sample_texture_cubic_projection_local(
    local: wp.vec3f,
    normal: wp.vec3f,
    texture_data: TextureData,
) -> wp.vec3f:
    """Cubic projection for local coordinates without UVs, similar to OmniPBR project_uvw."""
    repeat = texture_data.repeat
    abs_normal = wp.vec3f(wp.abs(normal[0]), wp.abs(normal[1]), wp.abs(normal[2]))

    uv = wp.vec2f(local[0], local[1])
    if abs_normal[0] >= abs_normal[1] and abs_normal[0] >= abs_normal[2]:
        uv = wp.vec2f(local[1], local[2])
    elif abs_normal[1] >= abs_normal[2]:
        uv = wp.vec2f(local[0], local[2])

    return sample_texture_2d(flip_v(wp.cw_mul(uv, repeat)), texture_data)


@wp.func
def sample_texture_cubic_projection_shape(
    hit_point: wp.vec3f,
    hit_normal: wp.vec3f,
    shape_transform: wp.transformf,
    texture_data: TextureData,
) -> wp.vec3f:
    inv_transform = wp.transform_inverse(shape_transform)
    local = wp.transform_point(inv_transform, hit_point)
    normal = wp.transform_vector(inv_transform, hit_normal)

    return sample_texture_cubic_projection_local(local, normal, texture_data)


@wp.func
def sample_texture_triplanar_local(
    local: wp.vec3f,
    normal: wp.vec3f,
    texture_data: TextureData,
) -> wp.vec3f:
    """Weighted triplanar projection for local coordinates without UVs."""
    w = wp.vec3f(wp.abs(normal[0]), wp.abs(normal[1]), wp.abs(normal[2]))
    w_sum = w[0] + w[1] + w[2]
    if w_sum > 0.0:
        w = w / w_sum
    else:
        w = wp.vec3f(0.0, 0.0, 1.0)

    repeat = texture_data.repeat
    c_x = sample_texture_2d(flip_v(wp.cw_mul(wp.vec2f(local[1], local[2]), repeat)), texture_data)
    c_y = sample_texture_2d(flip_v(wp.cw_mul(wp.vec2f(local[0], local[2]), repeat)), texture_data)
    c_z = sample_texture_2d(flip_v(wp.cw_mul(wp.vec2f(local[0], local[1]), repeat)), texture_data)

    return c_x * w[0] + c_y * w[1] + c_z * w[2]


@wp.func
def sample_texture_triplanar_shape(
    hit_point: wp.vec3f,
    hit_normal: wp.vec3f,
    shape_transform: wp.transformf,
    texture_data: TextureData,
) -> wp.vec3f:
    inv_transform = wp.transform_inverse(shape_transform)
    local = wp.transform_point(inv_transform, hit_point)
    normal = wp.transform_vector(inv_transform, hit_normal)

    return sample_texture_triplanar_local(local, normal, texture_data)


def create_sample_texture_function(config: RenderContext.Config) -> wp.Function:
    @wp.func
    def sample_texture(
        shape_type: wp.int32,
        shape_transform: wp.transformf,
        texture_data: wp.array[TextureData],
        texture_index: wp.int32,
        mesh_id: wp.uint64,
        mesh_data: wp.array[MeshData],
        mesh_data_index: wp.int32,
        hit_point: wp.vec3f,
        hit_normal: wp.vec3f,
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
            if face_id < 0:
                return DEFAULT_RETURN

            if mesh_data_index >= 0:
                if mesh_data[mesh_data_index].uvs.shape[0] > 0:
                    return sample_texture_mesh(
                        bary_u, bary_v, face_id, mesh_id, mesh_data[mesh_data_index], texture_data[texture_index]
                    )

            if wp.static(config.texture_projection_mode == TextureProjectionMode.TRIPLANAR):
                return sample_texture_triplanar_shape(
                    hit_point, hit_normal, shape_transform, texture_data[texture_index]
                )

            return sample_texture_cubic_projection_shape(
                hit_point, hit_normal, shape_transform, texture_data[texture_index]
            )

        if shape_type == GeoType.GAUSSIAN:
            return DEFAULT_RETURN

        if wp.static(config.texture_projection_mode == TextureProjectionMode.TRIPLANAR):
            return sample_texture_triplanar_shape(hit_point, hit_normal, shape_transform, texture_data[texture_index])

        return sample_texture_cubic_projection_shape(
            hit_point, hit_normal, shape_transform, texture_data[texture_index]
        )

    return sample_texture
