# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ...core import MAXVAL
from . import raytrace

if TYPE_CHECKING:
    from .render_context import RenderContext


def create_compute_lighting_function(config: RenderContext.Config, state: RenderContext.State) -> wp.Function:
    raytrace_first_hit = raytrace.create_first_hit_function(config, state)

    @wp.func
    def compute_lighting(
        world_index: wp.int32,
        bvh_shapes_size: wp.int32,
        bvh_shapes_id: wp.uint64,
        bvh_shapes_group_roots: wp.array(dtype=wp.int32),
        bvh_particles_size: wp.int32,
        bvh_particles_id: wp.uint64,
        bvh_particles_group_roots: wp.array(dtype=wp.int32),
        shape_enabled: wp.array(dtype=wp.uint32),
        shape_types: wp.array(dtype=wp.int32),
        shape_sizes: wp.array(dtype=wp.vec3f),
        shape_transforms: wp.array(dtype=wp.transformf),
        shape_source_ptr: wp.array(dtype=wp.uint64),
        light_active: wp.bool,
        light_type: wp.int32,
        light_cast_shadow: wp.bool,
        light_position: wp.vec3f,
        light_orientation: wp.vec3f,
        particles_position: wp.array(dtype=wp.vec3f),
        particles_radius: wp.array(dtype=wp.float32),
        triangle_mesh_id: wp.uint64,
        normal: wp.vec3f,
        hit_point: wp.vec3f,
    ) -> wp.float32:
        light_contribution = wp.float32(0.0)

        if not light_active:
            return light_contribution

        L = wp.vec3f(0.0, 0.0, 0.0)
        dist_to_light = wp.float32(MAXVAL)
        attenuation = wp.float32(1.0)

        if light_type == 1:  # directional light
            L = wp.normalize(-light_orientation)
        else:
            to_light = light_position - hit_point
            dist_to_light = wp.length(to_light)
            L = wp.normalize(to_light)
            attenuation = 1.0 / (1.0 + 0.02 * dist_to_light * dist_to_light)
            if light_type == 0:  # spot light
                spot_dir = wp.normalize(light_orientation)
                cos_theta = wp.dot(-L, spot_dir)
                inner = 0.95
                outer = 0.85
                spot_factor = wp.min(1.0, wp.max(0.0, (cos_theta - outer) / (inner - outer)))
                attenuation = attenuation * spot_factor

        ndotl = wp.max(0.0, wp.dot(normal, L))

        if ndotl == 0.0:
            return light_contribution

        visible = wp.float32(1.0)
        shadow_min_visibility = wp.float32(0.3)  # reduce shadow darkness (0: full black, 1: no shadow)

        if wp.static(config.enable_shadows) and light_cast_shadow:
            # Nudge the origin slightly along the surface normal to avoid
            # self-intersection when casting shadow rays
            eps = 1.0e-4
            shadow_origin = hit_point + normal * eps
            # Distance-limited shadows: cap by dist_to_light (for non-directional)
            max_t = wp.max(wp.float32(1.0e-4), wp.float32(dist_to_light - 1.0e-3))
            if light_type == 1:  # directional light
                max_t = wp.float32(1.0e8)

            shadow_hit = raytrace_first_hit(
                bvh_shapes_size,
                bvh_shapes_id,
                bvh_shapes_group_roots,
                bvh_particles_size,
                bvh_particles_id,
                bvh_particles_group_roots,
                world_index,
                shape_enabled,
                shape_types,
                shape_sizes,
                shape_transforms,
                shape_source_ptr,
                particles_position,
                particles_radius,
                triangle_mesh_id,
                shadow_origin,
                L,
                max_t,
            )

            if shadow_hit:
                visible = shadow_min_visibility

        return ndotl * attenuation * visible

    return compute_lighting
