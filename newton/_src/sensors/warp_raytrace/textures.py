import warp as wp

from .types import GeomType


@wp.func
def sample_texture_2d(
    uv: wp.vec2, width: int, height: int, tex_adr: int, tex_data: wp.array(dtype=wp.uint32)
) -> wp.vec3:
    ix = wp.min(width - 1, wp.int32(uv[0] * wp.float32(width)))
    it = wp.min(height - 1, wp.int32(uv[1] * wp.float32(height)))
    linear_idx = tex_adr + (it * width + ix)
    packed_rgba = tex_data[linear_idx]
    r = wp.float32((packed_rgba >> wp.uint32(16)) & wp.uint32(0xFF)) / 255.0
    g = wp.float32((packed_rgba >> wp.uint32(8)) & wp.uint32(0xFF)) / 255.0
    b = wp.float32(packed_rgba & wp.uint32(0xFF)) / 255.0
    return wp.vec3(r, g, b)


@wp.func
def sample_texture_plane(
    hit_point: wp.vec3,
    geom_pos: wp.vec3,
    geom_rot: wp.mat33,
    mat_texrepeat: wp.vec2,
    tex_adr: int,
    tex_data: wp.array(dtype=wp.uint32),
    tex_height: int,
    tex_width: int,
) -> wp.vec3:
    local = wp.transpose(geom_rot) @ (hit_point - geom_pos)
    u = local[0] * mat_texrepeat[0]
    v = local[1] * mat_texrepeat[1]
    u = u - wp.floor(u)
    v = v - wp.floor(v)
    v = 1.0 - v
    return sample_texture_2d(wp.vec2(u, v), tex_width, tex_height, tex_adr, tex_data)


@wp.func
def sample_texture_mesh(
    bary_u: wp.float32,
    bary_v: wp.float32,
    uv_baseadr: int,
    v_idx: wp.vec3i,
    mesh_texcoord: wp.array(dtype=wp.vec2),
    mat_texrepeat: wp.vec2,
    tex_adr: int,
    tex_data: wp.array(dtype=wp.uint32),
    tex_height: int,
    tex_width: int,
) -> wp.vec3:
    bw = 1.0 - bary_u - bary_v
    uv0 = mesh_texcoord[uv_baseadr + v_idx.x]
    uv1 = mesh_texcoord[uv_baseadr + v_idx.y]
    uv2 = mesh_texcoord[uv_baseadr + v_idx.z]
    uv = uv0 * bw + uv1 * bary_u + uv2 * bary_v
    u = uv[0] * mat_texrepeat[0]
    v = uv[1] * mat_texrepeat[1]
    u = u - wp.floor(u)
    v = v - wp.floor(v)
    v = 1.0 - v
    return sample_texture_2d(
        wp.vec2(u, v),
        tex_width,
        tex_height,
        tex_adr,
        tex_data,
    )


@wp.func
def sample_texture(
    world_id: int,
    geom_id: int,
    geom_type: wp.array(dtype=int),
    geom_matid: int,
    mat_texid: int,
    mat_texrepeat: wp.vec2,
    tex_adr: int,
    tex_data: wp.array(dtype=wp.uint32),
    tex_height: int,
    tex_width: int,
    geom_position: wp.vec3,
    geom_orientation: wp.mat33,
    mesh_faceadr: wp.array(dtype=int),
    mesh_face: wp.array(dtype=wp.vec3i),
    mesh_texcoord: wp.array(dtype=wp.vec2),
    mesh_texcoord_offsets: wp.array(dtype=int),
    hit_point: wp.vec3,
    u: wp.float32,
    v: wp.float32,
    f: wp.int32,
    mesh_id: wp.int32,
) -> wp.vec3:
    tex_color = wp.vec3(1.0, 1.0, 1.0)

    if geom_matid == -1 or mat_texid == -1:
        return tex_color

    if geom_type[geom_id] == int(GeomType.PLANE.value):
        tex_color = sample_texture_plane(
            hit_point,
            geom_position,
            geom_orientation,
            mat_texrepeat,
            tex_adr,
            tex_data,
            tex_height,
            tex_width,
        )

    if geom_type[geom_id] == int(GeomType.MESH.value):
        if f < 0 or mesh_id < 0:
            return tex_color

        uv_base = mesh_texcoord_offsets[mesh_id]
        tex_color = sample_texture_mesh(
            u,
            v,
            uv_base,
            # mesh_face[mesh_faceadr[mesh_id] + base_face + f],
            wp.vec3i(f * 3 + 2, f * 3 + 0, f * 3 + 1),
            mesh_texcoord,
            mat_texrepeat,
            tex_adr,
            tex_data,
            tex_height,
            tex_width,
        )

    return tex_color
