# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import warp as wp

from ...core import MAXVAL
from ...sim.light import LightType
from . import raytrace
from .types import RenderConfig, RenderLight

if TYPE_CHECKING:
    from .render_context import RenderContext


def create_compute_lighting_function(
    config: RenderConfig, state: "RenderContext.RenderState", optix=None
) -> wp.Function:
    raytrace_first_hit = raytrace.create_first_hit_function(config, state, optix)

    @wp.func
    def compute_lighting(
        world_index: wp.int32,
        bvh_shapes_size: wp.int32,
        bvh_shapes_id: wp.uint64,
        bvh_shapes_group_roots: wp.array[wp.int32],
        bvh_particles_size: wp.int32,
        bvh_particles_id: wp.uint64,
        bvh_particles_group_roots: wp.array[wp.int32],
        shape_enabled: wp.array[wp.uint32],
        shape_types: wp.array[wp.int32],
        shape_sizes: wp.array[wp.vec3f],
        shape_transforms: wp.array[wp.transformf],
        shape_source_ptr: wp.array[wp.uint64],
        light: RenderLight,
        particles_position: wp.array[wp.vec3f],
        particles_radius: wp.array[wp.float32],
        topology_particle_mask: wp.array[wp.bool],
        triangle_mesh_id: wp.uint64,
        triangle_mesh_group_roots: wp.array[wp.int32],
        normal: wp.vec3f,
        hit_point: wp.vec3f,
    ) -> wp.float32:
        """Return the fraction of the light's color that reaches ``hit_point``, including ``n · l``."""
        light_contribution = wp.float32(0.0)

        if not light.enabled:
            return light_contribution

        L = wp.vec3f(0.0, 0.0, 0.0)
        dist_to_light = wp.float32(MAXVAL)
        attenuation = wp.float32(1.0)

        if light.type == wp.static(int(LightType.DIRECTIONAL)):
            L = -light.direction
        else:
            to_light = light.position - hit_point
            dist_to_light = wp.length(to_light)
            if dist_to_light == 0.0:
                return light_contribution
            L = to_light / dist_to_light
            c = light.attenuation
            falloff = c[0] + c[1] * dist_to_light + c[2] * dist_to_light * dist_to_light
            attenuation = 1.0 / wp.max(falloff, 1.0e-6)
            if light.type == wp.static(int(LightType.SPOT)):
                cos_theta = wp.dot(-L, light.direction)
                if cos_theta < light.cos_cone_outer:
                    return light_contribution
                if light.cos_cone_inner > light.cos_cone_outer:
                    attenuation = attenuation * wp.smoothstep(light.cos_cone_outer, light.cos_cone_inner, cos_theta)
                if light.focus > 0.0:
                    attenuation = attenuation * wp.pow(wp.max(cos_theta, 0.0), light.focus)

        ndotl = wp.max(0.0, wp.dot(normal, L))

        if ndotl == 0.0 or attenuation == 0.0:
            return light_contribution

        visible = wp.float32(1.0)
        shadow_min_visibility = wp.float32(0.3)  # reduce shadow darkness (0: full black, 1: no shadow)

        if wp.static(config.enable_shadows) and light.cast_shadow:
            # Nudge the origin slightly along the surface normal to avoid
            # self-intersection when casting shadow rays
            eps = 1.0e-4
            shadow_origin = hit_point + normal * eps
            # Distance-limited shadows: cap by dist_to_light (for non-directional)
            max_t = wp.max(wp.float32(1.0e-4), wp.float32(dist_to_light - 1.0e-3))
            if light.type == wp.static(int(LightType.DIRECTIONAL)):
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
                topology_particle_mask,
                triangle_mesh_id,
                triangle_mesh_group_roots,
                shadow_origin,
                L,
                max_t,
            )

            if shadow_hit:
                visible = shadow_min_visibility

        return ndotl * attenuation * visible

    return compute_lighting
