# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ...geometry import Gaussian, GeoType
from ...utils.color import ColorSpace, color_srgb_to_linear, linear_to_srgb_wp, srgb_to_linear_wp
from . import lighting, raytrace, textures, tiling
from .types import AntiAliasing, ClearData, MeshData, RenderConfig, RenderOrder, TextureData, WorldRenderFlag

if TYPE_CHECKING:
    from .render_context import RenderContext


_MSAA_SURFACE_SLOTS = 4


@wp.struct
class RenderSample:
    """Outputs produced by tracing and shading one camera ray."""

    hit: wp.bool
    distance: wp.float32
    forward_depth: wp.float32
    shape_index: wp.uint32
    normal: wp.vec3f
    albedo: wp.vec3f
    color: wp.vec3f


@wp.struct
class TraceResult:
    """Geometry recovered by tracing one camera ray, retained so shading can be deferred.

    ``MSAA`` traces every subsample first and shades distinct surfaces afterwards, so the
    full :class:`~newton._src.sensors.sensor_camera_render.raytrace.ClosestHit` and the
    world-space ray are kept to shade later.
    """

    hit: wp.bool
    forward_depth: wp.float32
    ray_origin_world: wp.vec3f
    ray_dir_world: wp.vec3f
    closest_hit: raytrace.ClosestHit


@wp.struct
class ShadeResult:
    """Linear albedo and shaded color produced from a traced hit."""

    albedo: wp.vec3f
    color: wp.vec3f


def _srgb_packed_rgba_to_linear(packed: int) -> int:
    r = packed & 0xFF
    g = (packed >> 8) & 0xFF
    b = (packed >> 16) & 0xFF
    a = (packed >> 24) & 0xFF
    linear = color_srgb_to_linear((r / 255.0, g / 255.0, b / 255.0))
    lr = min(max(int(linear[0] * 255.0), 0), 255)
    lg = min(max(int(linear[1] * 255.0), 0), 255)
    lb = min(max(int(linear[2] * 255.0), 0), 255)
    return (a << 24) | (lb << 16) | (lg << 8) | lr


@wp.func
def _unpack_rgba(packed: wp.uint32):
    scale = 1.0 / 255.0
    return wp.vec4f(
        wp.float32(packed & wp.uint32(0xFF)) * scale,
        wp.float32((packed >> wp.uint32(8)) & wp.uint32(0xFF)) * scale,
        wp.float32((packed >> wp.uint32(16)) & wp.uint32(0xFF)) * scale,
        wp.float32((packed >> wp.uint32(24)) & wp.uint32(0xFF)) * scale,
    )


def create_kernel(config: RenderConfig, state: RenderContext.RenderState, clear_data: ClearData) -> wp.kernel:
    compute_lighting = lighting.create_compute_lighting_function(config, state)
    sample_texture = textures.create_sample_texture_function(config)

    if (
        state.render_color
        or state.render_hdr_color
        or state.render_normal
        or (state.render_albedo and config.enable_textures)
    ):
        raytrace_closest_hit = raytrace.create_closest_hit_function(config, state)
    else:
        raytrace_closest_hit = raytrace.create_closest_hit_depth_only_function(config, state)

    if config.output_color_space == ColorSpace.LINEAR:
        clear_data = ClearData(
            clear_color=_srgb_packed_rgba_to_linear(clear_data.clear_color),
            clear_depth=clear_data.clear_depth,
            clear_shape_index=clear_data.clear_shape_index,
            clear_normal=clear_data.clear_normal,
            clear_albedo=_srgb_packed_rgba_to_linear(clear_data.clear_albedo),
        )

    @wp.func
    def write_clear_outputs(
        out_index: wp.int32,
        out_color: wp.array[wp.uint32],
        out_depth: wp.array[wp.float32],
        out_forward_depth: wp.array[wp.float32],
        out_shape_index: wp.array[wp.uint32],
        out_normal: wp.array[wp.vec3f],
        out_albedo: wp.array[wp.uint32],
        out_hdr_color: wp.array[wp.vec3f],
    ):
        if wp.static(state.render_color):
            out_color[out_index] = wp.static(wp.uint32(clear_data.clear_color))
        if wp.static(state.render_albedo):
            out_albedo[out_index] = wp.static(wp.uint32(clear_data.clear_albedo))
        if wp.static(state.render_hdr_color):
            out_hdr_color[out_index] = wp.vec3f(0.0)
        if wp.static(state.render_depth):
            out_depth[out_index] = wp.float32(wp.static(clear_data.clear_depth))
        if wp.static(state.render_forward_depth):
            out_forward_depth[out_index] = wp.float32(wp.static(clear_data.clear_depth))
        if wp.static(state.render_normal):
            out_normal[out_index] = wp.vec3f(
                wp.static(clear_data.clear_normal[0]),
                wp.static(clear_data.clear_normal[1]),
                wp.static(clear_data.clear_normal[2]),
            )
        if wp.static(state.render_shape_index):
            out_shape_index[out_index] = wp.static(wp.uint32(clear_data.clear_shape_index))

    @wp.func
    def trace_sample(
        world_index: wp.int32,
        camera_transform: wp.transformf,
        camera_forward: wp.vec3f,
        forward_axis: wp.vec3f,
        camera_ray_origin: wp.vec3f,
        camera_ray_direction: wp.vec3f,
        bvh_shapes_size: wp.int32,
        bvh_shapes_id: wp.uint64,
        bvh_shapes_group_roots: wp.array[wp.int32],
        shape_enabled: wp.array[wp.uint32],
        shape_types: wp.array[wp.int32],
        shape_sizes: wp.array[wp.vec3f],
        shape_transforms: wp.array[wp.transformf],
        shape_source_ptr: wp.array[wp.uint64],
        shape_mesh_data_ids: wp.array[wp.int32],
        bvh_particles_size: wp.int32,
        bvh_particles_id: wp.uint64,
        bvh_particles_group_roots: wp.array[wp.int32],
        particles_position: wp.array[wp.vec3f],
        particles_radius: wp.array[wp.float32],
        topology_particle_mask: wp.array[wp.bool],
        triangle_mesh_id: wp.uint64,
        triangle_mesh_group_roots: wp.array[wp.int32],
        mesh_data: wp.array[MeshData],
        gaussians_data: wp.array[Gaussian.Data],
    ) -> TraceResult:
        result = TraceResult()
        result.hit = wp.bool(False)
        result.forward_depth = wp.float32(0.0)

        ray_origin_world = wp.transform_point(camera_transform, camera_ray_origin)
        ray_dir_world = wp.transform_vector(camera_transform, camera_ray_direction)
        result.ray_origin_world = ray_origin_world
        result.ray_dir_world = ray_dir_world
        if wp.dot(ray_dir_world, ray_dir_world) <= 1.0e-12:
            return result

        closest_hit = raytrace_closest_hit(
            bvh_shapes_size,
            bvh_shapes_id,
            bvh_shapes_group_roots,
            bvh_particles_size,
            bvh_particles_id,
            bvh_particles_group_roots,
            world_index,
            wp.static(config.max_distance),
            shape_enabled,
            shape_types,
            shape_sizes,
            shape_transforms,
            shape_source_ptr,
            shape_mesh_data_ids,
            mesh_data,
            particles_position,
            particles_radius,
            topology_particle_mask,
            triangle_mesh_id,
            triangle_mesh_group_roots,
            gaussians_data,
            ray_origin_world,
            ray_dir_world,
            camera_forward,
        )
        result.closest_hit = closest_hit
        if closest_hit.shape_index == raytrace.NO_HIT_SHAPE_ID:
            return result

        result.hit = wp.bool(True)
        if wp.static(state.render_forward_depth):
            result.forward_depth = closest_hit.distance * wp.dot(ray_dir_world, forward_axis)
        return result

    @wp.func
    def shade_trace(
        trace: TraceResult,
        world_index: wp.int32,
        light_count: wp.int32,
        bvh_shapes_size: wp.int32,
        bvh_shapes_id: wp.uint64,
        bvh_shapes_group_roots: wp.array[wp.int32],
        shape_enabled: wp.array[wp.uint32],
        shape_types: wp.array[wp.int32],
        shape_sizes: wp.array[wp.vec3f],
        shape_colors: wp.array[wp.vec3f],
        shape_transforms: wp.array[wp.transformf],
        shape_source_ptr: wp.array[wp.uint64],
        shape_texture_ids: wp.array[wp.int32],
        shape_mesh_data_ids: wp.array[wp.int32],
        bvh_particles_size: wp.int32,
        bvh_particles_id: wp.uint64,
        bvh_particles_group_roots: wp.array[wp.int32],
        particles_position: wp.array[wp.vec3f],
        particles_radius: wp.array[wp.float32],
        topology_particle_mask: wp.array[wp.bool],
        triangle_mesh_id: wp.uint64,
        triangle_mesh_group_roots: wp.array[wp.int32],
        triangle_colors: wp.array[wp.vec3f],
        mesh_data: wp.array[MeshData],
        texture_data: wp.array[TextureData],
        light_active: wp.array[wp.bool],
        light_type: wp.array[wp.int32],
        light_cast_shadow: wp.array[wp.bool],
        light_positions: wp.array[wp.vec3f],
        light_orientations: wp.array[wp.vec3f],
    ) -> ShadeResult:
        result = ShadeResult()
        result.albedo = wp.vec3f(0.0)
        result.color = wp.vec3f(0.0)

        closest_hit = trace.closest_hit
        is_gaussian = wp.bool(False)
        if wp.static(state.num_gaussians > 0):
            if closest_hit.shape_index < raytrace.MAX_SHAPE_ID:
                if shape_types[closest_hit.shape_index] == GeoType.GAUSSIAN:
                    is_gaussian = wp.bool(True)

        albedo_color = wp.vec3f(0.0)
        hit_point = trace.ray_origin_world + trace.ray_dir_world * closest_hit.distance
        if not is_gaussian:
            albedo_color = wp.vec3f(1.0)
            if closest_hit.shape_index < raytrace.MAX_SHAPE_ID:
                albedo_color = srgb_to_linear_wp(shape_colors[closest_hit.shape_index])
            elif closest_hit.shape_index == raytrace.TRIANGLE_MESH_SHAPE_ID:
                if closest_hit.face_idx >= 0 and closest_hit.face_idx < triangle_colors.shape[0]:
                    albedo_color = srgb_to_linear_wp(triangle_colors[closest_hit.face_idx])

            if wp.static(config.enable_textures) and closest_hit.shape_index < raytrace.MAX_SHAPE_ID:
                texture_index = shape_texture_ids[closest_hit.shape_index]
                if texture_index > -1:
                    tex_color = sample_texture(
                        shape_types[closest_hit.shape_index],
                        shape_transforms[closest_hit.shape_index],
                        texture_data,
                        texture_index,
                        shape_source_ptr[closest_hit.shape_index],
                        mesh_data,
                        shape_mesh_data_ids[closest_hit.shape_index],
                        hit_point,
                        closest_hit.normal,
                        closest_hit.bary_u,
                        closest_hit.bary_v,
                        closest_hit.face_idx,
                    )
                    albedo_color = wp.cw_mul(albedo_color, srgb_to_linear_wp(tex_color))

        result.albedo = albedo_color
        if not wp.static(state.render_color) and not wp.static(state.render_hdr_color):
            return result

        shaded_color = closest_hit.color
        if not is_gaussian:
            if wp.static(config.enable_ambient_lighting):
                up = wp.vec3f(0.0, 0.0, 1.0)
                len_n = wp.length(closest_hit.normal)
                n = closest_hit.normal if len_n > 0.0 else up
                n = wp.normalize(n)
                hemispheric = 0.5 * (wp.dot(n, up) + 1.0)
                sky = wp.vec3f(0.4, 0.4, 0.45)
                ground = wp.vec3f(0.1, 0.1, 0.12)
                ambient_color = sky * hemispheric + ground * (1.0 - hemispheric)
                shaded_color = wp.cw_mul(albedo_color, ambient_color * 0.5)

            for light_index in range(light_count):
                light_contribution = compute_lighting(
                    world_index,
                    bvh_shapes_size,
                    bvh_shapes_id,
                    bvh_shapes_group_roots,
                    bvh_particles_size,
                    bvh_particles_id,
                    bvh_particles_group_roots,
                    shape_enabled,
                    shape_types,
                    shape_sizes,
                    shape_transforms,
                    shape_source_ptr,
                    light_active[light_index],
                    light_type[light_index],
                    light_cast_shadow[light_index],
                    light_positions[light_index],
                    light_orientations[light_index],
                    particles_position,
                    particles_radius,
                    topology_particle_mask,
                    triangle_mesh_id,
                    triangle_mesh_group_roots,
                    closest_hit.normal,
                    hit_point,
                )
                shaded_color = shaded_color + albedo_color * light_contribution

        result.color = shaded_color
        return result

    @wp.func
    def render_sample(
        light_count: wp.int32,
        world_index: wp.int32,
        camera_transform: wp.transformf,
        camera_forward: wp.vec3f,
        forward_axis: wp.vec3f,
        camera_ray_origin: wp.vec3f,
        camera_ray_direction: wp.vec3f,
        bvh_shapes_size: wp.int32,
        bvh_shapes_id: wp.uint64,
        bvh_shapes_group_roots: wp.array[wp.int32],
        shape_enabled: wp.array[wp.uint32],
        shape_types: wp.array[wp.int32],
        shape_sizes: wp.array[wp.vec3f],
        shape_colors: wp.array[wp.vec3f],
        shape_transforms: wp.array[wp.transformf],
        shape_source_ptr: wp.array[wp.uint64],
        shape_texture_ids: wp.array[wp.int32],
        shape_mesh_data_ids: wp.array[wp.int32],
        bvh_particles_size: wp.int32,
        bvh_particles_id: wp.uint64,
        bvh_particles_group_roots: wp.array[wp.int32],
        particles_position: wp.array[wp.vec3f],
        particles_radius: wp.array[wp.float32],
        topology_particle_mask: wp.array[wp.bool],
        triangle_mesh_id: wp.uint64,
        triangle_mesh_group_roots: wp.array[wp.int32],
        triangle_colors: wp.array[wp.vec3f],
        mesh_data: wp.array[MeshData],
        gaussians_data: wp.array[Gaussian.Data],
        texture_data: wp.array[TextureData],
        light_active: wp.array[wp.bool],
        light_type: wp.array[wp.int32],
        light_cast_shadow: wp.array[wp.bool],
        light_positions: wp.array[wp.vec3f],
        light_orientations: wp.array[wp.vec3f],
    ) -> RenderSample:
        result = RenderSample()
        result.hit = wp.bool(False)
        result.distance = wp.float32(0.0)
        result.forward_depth = wp.float32(0.0)
        result.shape_index = raytrace.NO_HIT_SHAPE_ID
        result.normal = wp.vec3f(0.0)
        result.albedo = wp.vec3f(0.0)
        result.color = wp.vec3f(0.0)

        trace = trace_sample(
            world_index,
            camera_transform,
            camera_forward,
            forward_axis,
            camera_ray_origin,
            camera_ray_direction,
            bvh_shapes_size,
            bvh_shapes_id,
            bvh_shapes_group_roots,
            shape_enabled,
            shape_types,
            shape_sizes,
            shape_transforms,
            shape_source_ptr,
            shape_mesh_data_ids,
            bvh_particles_size,
            bvh_particles_id,
            bvh_particles_group_roots,
            particles_position,
            particles_radius,
            topology_particle_mask,
            triangle_mesh_id,
            triangle_mesh_group_roots,
            mesh_data,
            gaussians_data,
        )
        if not trace.hit:
            return result

        closest_hit = trace.closest_hit
        result.hit = wp.bool(True)
        result.distance = closest_hit.distance
        result.shape_index = closest_hit.shape_index
        result.normal = closest_hit.normal
        result.forward_depth = trace.forward_depth

        if (
            not wp.static(state.render_color)
            and not wp.static(state.render_albedo)
            and not wp.static(state.render_hdr_color)
        ):
            return result

        shade = shade_trace(
            trace,
            world_index,
            light_count,
            bvh_shapes_size,
            bvh_shapes_id,
            bvh_shapes_group_roots,
            shape_enabled,
            shape_types,
            shape_sizes,
            shape_colors,
            shape_transforms,
            shape_source_ptr,
            shape_texture_ids,
            shape_mesh_data_ids,
            bvh_particles_size,
            bvh_particles_id,
            bvh_particles_group_roots,
            particles_position,
            particles_radius,
            topology_particle_mask,
            triangle_mesh_id,
            triangle_mesh_group_roots,
            triangle_colors,
            mesh_data,
            texture_data,
            light_active,
            light_type,
            light_cast_shadow,
            light_positions,
            light_orientations,
        )
        result.albedo = shade.albedo
        result.color = shade.color
        return result

    @wp.kernel(enable_backward=False, module="unique", module_options={"fast_math": config.enable_fast_math})
    def render_megakernel(
        # Model and Config
        view_count: wp.int32,
        world_count: wp.int32,
        light_count: wp.int32,
        img_width: wp.int32,
        img_height: wp.int32,
        # Camera
        camera_rays: wp.array4d[wp.vec3f],
        camera_transforms: wp.array[wp.transformf],
        world_indices: wp.array[wp.int32],
        # Shapes BVH
        bvh_shapes_size: wp.int32,
        bvh_shapes_id: wp.uint64,
        bvh_shapes_group_roots: wp.array[wp.int32],
        # Shapes
        shape_enabled: wp.array[wp.uint32],
        shape_types: wp.array[wp.int32],
        shape_sizes: wp.array[wp.vec3f],
        shape_colors: wp.array[wp.vec3f],
        shape_transforms: wp.array[wp.transformf],
        shape_source_ptr: wp.array[wp.uint64],
        shape_texture_ids: wp.array[wp.int32],
        shape_mesh_data_ids: wp.array[wp.int32],
        # Particle BVH
        bvh_particles_size: wp.int32,
        bvh_particles_id: wp.uint64,
        bvh_particles_group_roots: wp.array[wp.int32],
        # Particles
        particles_position: wp.array[wp.vec3f],
        particles_radius: wp.array[wp.float32],
        topology_particle_mask: wp.array[wp.bool],
        # Triangle Mesh:
        triangle_mesh_id: wp.uint64,
        triangle_mesh_group_roots: wp.array[wp.int32],
        triangle_colors: wp.array[wp.vec3f],
        # Meshes
        mesh_data: wp.array[MeshData],
        # Gaussians
        gaussians_data: wp.array[Gaussian.Data],
        # Textures
        texture_data: wp.array[TextureData],
        # Lights
        light_active: wp.array[wp.bool],
        light_type: wp.array[wp.int32],
        light_cast_shadow: wp.array[wp.bool],
        light_positions: wp.array[wp.vec3f],
        light_orientations: wp.array[wp.vec3f],
        # Outputs
        out_color: wp.array[wp.uint32],
        out_depth: wp.array[wp.float32],
        out_forward_depth: wp.array[wp.float32],
        out_shape_index: wp.array[wp.uint32],
        out_normal: wp.array[wp.vec3f],
        out_albedo: wp.array[wp.uint32],
        out_hdr_color: wp.array[wp.vec3f],
    ):
        tid = wp.tid()

        if wp.static(config.render_order == RenderOrder.PIXEL_PRIORITY):
            view_index, py, px = tiling.tid_to_coord_pixel_priority(tid, view_count, img_width)
        elif wp.static(config.render_order == RenderOrder.VIEW_PRIORITY):
            view_index, py, px = tiling.tid_to_coord_view_priority(tid, img_width, img_height)
        elif wp.static(config.render_order == RenderOrder.TILED):
            view_index, py, px = tiling.tid_to_coord_tiled(
                tid, img_width, img_height, wp.static(config.tile_width), wp.static(config.tile_height)
            )
        else:
            return

        if px >= img_width or py >= img_height:
            return

        pixels_per_view = img_width * img_height
        out_index = view_index * pixels_per_view + py * img_width + px

        # With an explicit mapping, a valid entry is a world index in
        # ``[0, world_count)``. ``DISABLE_PRESERVE`` leaves the outputs untouched;
        # every other out-of-range value -- ``DISABLE_CLEAR``, ``-1`` (reserved
        # for future global-world rendering), or an index ``>= world_count`` --
        # clears the outputs, which also avoids reading past the group-root arrays.
        # Without a mapping (``world_indices is None``), each view renders its own
        # world (``world_index == view_index``).
        if wp.static(state.has_world_indices):
            world_index = world_indices[view_index]
            if world_index == wp.static(int(WorldRenderFlag.DISABLE_PRESERVE)):
                return
            if world_index < 0 or world_index >= world_count:
                write_clear_outputs(
                    out_index,
                    out_color,
                    out_depth,
                    out_forward_depth,
                    out_shape_index,
                    out_normal,
                    out_albedo,
                    out_hdr_color,
                )
                return
        else:
            world_index = view_index

        camera_transform = camera_transforms[view_index]
        camera_forward = wp.vec3f(0.0)
        if wp.static(state.num_gaussians > 0):
            camera_forward = wp.transform_vector(camera_transform, wp.vec3f(0.0, 0.0, -1.0))

        forward_axis = wp.vec3f(0.0)
        if wp.static(state.render_forward_depth):
            forward_axis = wp.normalize(wp.transform_vector(camera_transform, wp.vec3f(0.0, 0.0, -1.0)))

        if wp.static(config.anti_aliasing == AntiAliasing.NONE):
            sample = render_sample(
                light_count,
                world_index,
                camera_transform,
                camera_forward,
                forward_axis,
                camera_rays[py, px, 0, 0],
                camera_rays[py, px, 0, 1],
                bvh_shapes_size,
                bvh_shapes_id,
                bvh_shapes_group_roots,
                shape_enabled,
                shape_types,
                shape_sizes,
                shape_colors,
                shape_transforms,
                shape_source_ptr,
                shape_texture_ids,
                shape_mesh_data_ids,
                bvh_particles_size,
                bvh_particles_id,
                bvh_particles_group_roots,
                particles_position,
                particles_radius,
                topology_particle_mask,
                triangle_mesh_id,
                triangle_mesh_group_roots,
                triangle_colors,
                mesh_data,
                gaussians_data,
                texture_data,
                light_active,
                light_type,
                light_cast_shadow,
                light_positions,
                light_orientations,
            )
            if not sample.hit:
                write_clear_outputs(
                    out_index,
                    out_color,
                    out_depth,
                    out_forward_depth,
                    out_shape_index,
                    out_normal,
                    out_albedo,
                    out_hdr_color,
                )
                return

            if wp.static(state.render_depth):
                out_depth[out_index] = sample.distance
            if wp.static(state.render_forward_depth):
                out_forward_depth[out_index] = sample.forward_depth
            if wp.static(state.render_normal):
                out_normal[out_index] = sample.normal
            if wp.static(state.render_shape_index):
                out_shape_index[out_index] = sample.shape_index
            if wp.static(state.render_hdr_color):
                out_hdr_color[out_index] = sample.color
            if wp.static(state.render_albedo):
                albedo = sample.albedo
                if wp.static(config.output_color_space == ColorSpace.SRGB):
                    albedo = linear_to_srgb_wp(albedo)
                out_albedo[out_index] = tiling.pack_rgba_to_uint32(albedo, 1.0)
            if wp.static(state.render_color):
                color = sample.color
                if wp.static(config.output_color_space == ColorSpace.SRGB):
                    color = linear_to_srgb_wp(color)
                out_color[out_index] = tiling.pack_rgba_to_uint32(color, 1.0)
            return

        clear_color = wp.vec4f(0.0)
        if wp.static(state.render_color):
            clear_color = _unpack_rgba(wp.uint32(wp.static(clear_data.clear_color)))
            if wp.static(config.output_color_space == ColorSpace.SRGB):
                clear_color_rgb = srgb_to_linear_wp(wp.vec3f(clear_color[0], clear_color[1], clear_color[2]))
                clear_color = wp.vec4f(clear_color_rgb[0], clear_color_rgb[1], clear_color_rgb[2], clear_color[3])

        clear_albedo = wp.vec4f(0.0)
        if wp.static(state.render_albedo):
            clear_albedo = _unpack_rgba(wp.uint32(wp.static(clear_data.clear_albedo)))
            if wp.static(config.output_color_space == ColorSpace.SRGB):
                clear_albedo_rgb = srgb_to_linear_wp(wp.vec3f(clear_albedo[0], clear_albedo[1], clear_albedo[2]))
                clear_albedo = wp.vec4f(clear_albedo_rgb[0], clear_albedo_rgb[1], clear_albedo_rgb[2], clear_albedo[3])

        sample_count = camera_rays.shape[2]

        if wp.static(config.anti_aliasing == AntiAliasing.SSAA):
            color_sum = wp.vec4f(0.0)
            albedo_sum = wp.vec4f(0.0)
            hdr_color_sum = wp.vec3f(0.0)
            nearest_distance = wp.float32(wp.static(config.max_distance)) + 1.0
            nearest_forward_depth = wp.float32(0.0)
            nearest_normal = wp.vec3f(0.0)
            nearest_shape_index = wp.uint32(raytrace.NO_HIT_SHAPE_ID)
            has_hit = wp.bool(False)

            for sample_index in range(sample_count):
                sample = render_sample(
                    light_count,
                    world_index,
                    camera_transform,
                    camera_forward,
                    forward_axis,
                    camera_rays[py, px, sample_index, 0],
                    camera_rays[py, px, sample_index, 1],
                    bvh_shapes_size,
                    bvh_shapes_id,
                    bvh_shapes_group_roots,
                    shape_enabled,
                    shape_types,
                    shape_sizes,
                    shape_colors,
                    shape_transforms,
                    shape_source_ptr,
                    shape_texture_ids,
                    shape_mesh_data_ids,
                    bvh_particles_size,
                    bvh_particles_id,
                    bvh_particles_group_roots,
                    particles_position,
                    particles_radius,
                    topology_particle_mask,
                    triangle_mesh_id,
                    triangle_mesh_group_roots,
                    triangle_colors,
                    mesh_data,
                    gaussians_data,
                    texture_data,
                    light_active,
                    light_type,
                    light_cast_shadow,
                    light_positions,
                    light_orientations,
                )
                if not sample.hit:
                    if wp.static(state.render_color):
                        color_sum += clear_color
                    if wp.static(state.render_albedo):
                        albedo_sum += clear_albedo
                    continue

                has_hit = wp.bool(True)
                if sample.distance < nearest_distance:
                    nearest_distance = sample.distance
                    if wp.static(state.render_forward_depth):
                        nearest_forward_depth = sample.forward_depth
                    if wp.static(state.render_normal):
                        nearest_normal = sample.normal
                    if wp.static(state.render_shape_index):
                        nearest_shape_index = sample.shape_index

                if wp.static(state.render_albedo):
                    albedo_sum += wp.vec4f(sample.albedo[0], sample.albedo[1], sample.albedo[2], 1.0)
                if wp.static(state.render_color):
                    color_sum += wp.vec4f(sample.color[0], sample.color[1], sample.color[2], 1.0)
                if wp.static(state.render_hdr_color):
                    hdr_color_sum += sample.color

            if not has_hit:
                write_clear_outputs(
                    out_index,
                    out_color,
                    out_depth,
                    out_forward_depth,
                    out_shape_index,
                    out_normal,
                    out_albedo,
                    out_hdr_color,
                )
                return

            if wp.static(state.render_depth):
                out_depth[out_index] = nearest_distance
            if wp.static(state.render_forward_depth):
                out_forward_depth[out_index] = nearest_forward_depth
            if wp.static(state.render_normal):
                out_normal[out_index] = nearest_normal
            if wp.static(state.render_shape_index):
                out_shape_index[out_index] = nearest_shape_index

            sample_scale = 1.0 / float(sample_count)
            if wp.static(state.render_hdr_color):
                out_hdr_color[out_index] = hdr_color_sum * sample_scale
            if wp.static(state.render_albedo):
                averaged_albedo = albedo_sum * sample_scale
                albedo_rgb = wp.vec3f(averaged_albedo[0], averaged_albedo[1], averaged_albedo[2])
                if wp.static(config.output_color_space == ColorSpace.SRGB):
                    albedo_rgb = linear_to_srgb_wp(albedo_rgb)
                out_albedo[out_index] = tiling.pack_rgba_to_uint32(albedo_rgb, averaged_albedo[3])
            if wp.static(state.render_color):
                averaged_color = color_sum * sample_scale
                color_rgb = wp.vec3f(averaged_color[0], averaged_color[1], averaged_color[2])
                if wp.static(config.output_color_space == ColorSpace.SRGB):
                    color_rgb = linear_to_srgb_wp(color_rgb)
                out_color[out_index] = tiling.pack_rgba_to_uint32(color_rgb, averaged_color[3])
            return

        if wp.static(config.anti_aliasing == AntiAliasing.MSAA):
            nearest_distance = wp.float32(wp.static(config.max_distance)) + 1.0
            nearest_forward_depth = wp.float32(0.0)
            nearest_normal = wp.vec3f(0.0)
            nearest_shape_index = wp.uint32(raytrace.NO_HIT_SHAPE_ID)
            hit_count = wp.int32(0)
            miss_count = wp.int32(0)

            shaded_sum = wp.vec3f(0.0)
            albedo_accum = wp.vec3f(0.0)

            slot_shape = wp.vector(length=_MSAA_SURFACE_SLOTS, dtype=wp.uint32)
            slot_shape_sub_index = wp.vector(length=_MSAA_SURFACE_SLOTS, dtype=wp.int32)
            slot_color = wp.matrix(shape=(_MSAA_SURFACE_SLOTS, 3), dtype=wp.float32)
            slot_albedo = wp.matrix(shape=(_MSAA_SURFACE_SLOTS, 3), dtype=wp.float32)
            if wp.static(state.render_color or state.render_albedo or state.render_hdr_color):
                for slot in range(_MSAA_SURFACE_SLOTS):
                    slot_shape[slot] = raytrace.NO_HIT_SHAPE_ID
                    slot_shape_sub_index[slot] = -1

            for sample_index in range(sample_count):
                trace = trace_sample(
                    world_index,
                    camera_transform,
                    camera_forward,
                    forward_axis,
                    camera_rays[py, px, sample_index, 0],
                    camera_rays[py, px, sample_index, 1],
                    bvh_shapes_size,
                    bvh_shapes_id,
                    bvh_shapes_group_roots,
                    shape_enabled,
                    shape_types,
                    shape_sizes,
                    shape_transforms,
                    shape_source_ptr,
                    shape_mesh_data_ids,
                    bvh_particles_size,
                    bvh_particles_id,
                    bvh_particles_group_roots,
                    particles_position,
                    particles_radius,
                    topology_particle_mask,
                    triangle_mesh_id,
                    triangle_mesh_group_roots,
                    mesh_data,
                    gaussians_data,
                )
                if not trace.hit:
                    miss_count += 1
                    continue

                hit_count += 1
                if trace.closest_hit.distance < nearest_distance:
                    nearest_distance = trace.closest_hit.distance
                    if wp.static(state.render_forward_depth):
                        nearest_forward_depth = trace.forward_depth
                    if wp.static(state.render_normal):
                        nearest_normal = trace.closest_hit.normal
                    if wp.static(state.render_shape_index):
                        nearest_shape_index = trace.closest_hit.shape_index

                if wp.static(state.render_color or state.render_albedo or state.render_hdr_color):
                    surface_id = trace.closest_hit.shape_index
                    shape_sub_index = wp.int32(-1)
                    if surface_id == raytrace.PARTICLES_SHAPE_ID or surface_id == raytrace.TRIANGLE_MESH_SHAPE_ID:
                        shape_sub_index = trace.closest_hit.face_idx
                    sample_color = wp.vec3f(0.0)
                    sample_albedo = wp.vec3f(0.0)
                    found = wp.bool(False)
                    # Sentinel shape IDs need a particle or face index to distinguish hits.
                    if surface_id < raytrace.MAX_SHAPE_ID or shape_sub_index >= 0:
                        for slot in range(_MSAA_SURFACE_SLOTS):
                            if (
                                not found
                                and slot_shape[slot] == surface_id
                                and slot_shape_sub_index[slot] == shape_sub_index
                            ):
                                sample_color = wp.vec3f(slot_color[slot, 0], slot_color[slot, 1], slot_color[slot, 2])
                                sample_albedo = wp.vec3f(
                                    slot_albedo[slot, 0], slot_albedo[slot, 1], slot_albedo[slot, 2]
                                )
                                found = wp.bool(True)

                    if not found:
                        shade = shade_trace(
                            trace,
                            world_index,
                            light_count,
                            bvh_shapes_size,
                            bvh_shapes_id,
                            bvh_shapes_group_roots,
                            shape_enabled,
                            shape_types,
                            shape_sizes,
                            shape_colors,
                            shape_transforms,
                            shape_source_ptr,
                            shape_texture_ids,
                            shape_mesh_data_ids,
                            bvh_particles_size,
                            bvh_particles_id,
                            bvh_particles_group_roots,
                            particles_position,
                            particles_radius,
                            topology_particle_mask,
                            triangle_mesh_id,
                            triangle_mesh_group_roots,
                            triangle_colors,
                            mesh_data,
                            texture_data,
                            light_active,
                            light_type,
                            light_cast_shadow,
                            light_positions,
                            light_orientations,
                        )
                        sample_color = shade.color
                        sample_albedo = shade.albedo
                        if surface_id < raytrace.MAX_SHAPE_ID or shape_sub_index >= 0:
                            inserted = wp.bool(False)
                            for slot in range(_MSAA_SURFACE_SLOTS):
                                if not inserted and slot_shape[slot] == raytrace.NO_HIT_SHAPE_ID:
                                    slot_shape[slot] = surface_id
                                    slot_shape_sub_index[slot] = shape_sub_index
                                    slot_color[slot, 0] = sample_color[0]
                                    slot_color[slot, 1] = sample_color[1]
                                    slot_color[slot, 2] = sample_color[2]
                                    slot_albedo[slot, 0] = sample_albedo[0]
                                    slot_albedo[slot, 1] = sample_albedo[1]
                                    slot_albedo[slot, 2] = sample_albedo[2]
                                    inserted = wp.bool(True)

                    if wp.static(state.render_color or state.render_hdr_color):
                        shaded_sum += sample_color
                    if wp.static(state.render_albedo):
                        albedo_accum += sample_albedo

            if hit_count == 0:
                write_clear_outputs(
                    out_index,
                    out_color,
                    out_depth,
                    out_forward_depth,
                    out_shape_index,
                    out_normal,
                    out_albedo,
                    out_hdr_color,
                )
                return

            if wp.static(state.render_depth):
                out_depth[out_index] = nearest_distance
            if wp.static(state.render_forward_depth):
                out_forward_depth[out_index] = nearest_forward_depth
            if wp.static(state.render_normal):
                out_normal[out_index] = nearest_normal
            if wp.static(state.render_shape_index):
                out_shape_index[out_index] = nearest_shape_index

            sample_scale = 1.0 / float(sample_count)
            miss_weight = float(miss_count)
            if wp.static(state.render_hdr_color):
                out_hdr_color[out_index] = shaded_sum * sample_scale
            if wp.static(state.render_albedo):
                albedo_rgb = (
                    albedo_accum + wp.vec3f(clear_albedo[0], clear_albedo[1], clear_albedo[2]) * miss_weight
                ) * sample_scale
                albedo_alpha = (float(hit_count) + clear_albedo[3] * miss_weight) * sample_scale
                if wp.static(config.output_color_space == ColorSpace.SRGB):
                    albedo_rgb = linear_to_srgb_wp(albedo_rgb)
                out_albedo[out_index] = tiling.pack_rgba_to_uint32(albedo_rgb, albedo_alpha)
            if wp.static(state.render_color):
                color_rgb = (
                    shaded_sum + wp.vec3f(clear_color[0], clear_color[1], clear_color[2]) * miss_weight
                ) * sample_scale
                color_alpha = (float(hit_count) + clear_color[3] * miss_weight) * sample_scale
                if wp.static(config.output_color_space == ColorSpace.SRGB):
                    color_rgb = linear_to_srgb_wp(color_rgb)
                out_color[out_index] = tiling.pack_rgba_to_uint32(color_rgb, color_alpha)

    return render_megakernel
