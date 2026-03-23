# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ...geometry import Gaussian
from ...math import safe_div
from . import bvh
from .ray_intersect import GeomHit, map_ray_to_local_scaled
from .types import GaussianRenderMode

if TYPE_CHECKING:
    from .render_context import RenderContext


SH_C0 = wp.float32(0.28209479177387814)


@wp.func
def compute_gaussian_bounds(gaussians_data: Gaussian.Data, tid: wp.int32) -> tuple[wp.vec3f, wp.vec3f]:
    transform = gaussians_data.transforms[tid]
    scale = gaussians_data.scales[tid]

    mod = gaussians_data.min_response / wp.max(gaussians_data.opacities[tid], wp.float32(1e-6))
    min_response = wp.clamp(mod, wp.float32(1e-6), wp.float32(0.97))
    ks = wp.sqrt(wp.log(min_response) / wp.float32(-0.5))
    scale = wp.vec3f(scale[0] * ks, scale[1] * ks, scale[2] * ks)

    return bvh.compute_ellipsoid_bounds(transform, scale)


@wp.kernel
def compute_gaussian_bvh_bounds(
    gaussians_data: Gaussian.Data,
    lowers: wp.array(dtype=wp.vec3f),
    uppers: wp.array(dtype=wp.vec3f),
):
    tid = wp.tid()
    lower, upper = compute_gaussian_bounds(gaussians_data, tid)
    lowers[tid] = lower
    uppers[tid] = upper


@wp.func
def canonical_ray_hit_distance(ray_origin: wp.vec3f, ray_direction: wp.vec3f) -> wp.float32:
    numerator = -wp.dot(ray_origin, ray_direction)
    return safe_div(numerator, wp.dot(ray_direction, ray_direction))


@wp.func
def canonical_ray_min_squared_distance(ray_origin: wp.vec3f, ray_direction: wp.vec3f) -> wp.float32:
    gcrod = wp.cross(ray_direction, ray_origin)
    return wp.dot(gcrod, gcrod)


@wp.func
def canonical_ray_max_kernel_response(ray_origin: wp.vec3f, ray_direction: wp.vec3f) -> wp.float32:
    return wp.exp(-0.5 * canonical_ray_min_squared_distance(ray_origin, ray_direction))


@wp.func
def ray_gsplat_hit_response(
    transform: wp.transformf,
    scale: wp.vec3f,
    opacity: wp.float32,
    min_response: wp.float32,
    ray_origin_world: wp.vec3f,
    ray_direction_world: wp.vec3f,
    max_distance: wp.float32,
) -> tuple[wp.float32, wp.float32]:
    ray_origin_local, ray_direction_local = map_ray_to_local_scaled(
        transform, scale, ray_origin_world, ray_direction_world
    )

    hit_distance = canonical_ray_hit_distance(ray_origin_local, ray_direction_local)

    if hit_distance > 0.0 and hit_distance < max_distance:
        max_response = canonical_ray_max_kernel_response(ray_origin_local, wp.normalize(ray_direction_local))

        alpha = wp.min(wp.float32(1.0), max_response * opacity)
        if max_response > min_response and alpha > wp.static(1.0 / 255.0):
            return alpha, hit_distance
    return 0.0, -1.0


def create_shade_function(config: RenderContext.Config, state: RenderContext.State) -> wp.Function:
    @wp.func
    def shade(
        transform: wp.transformf,
        scale: wp.vec3f,
        ray_origin: wp.vec3f,
        ray_direction: wp.vec3f,
        gaussian_data: Gaussian.Data,
        max_distance: wp.float32,
    ) -> tuple[GeomHit, wp.vec3f]:
        result_hit = GeomHit()
        result_hit.hit = False
        result_hit.normal = wp.vec3f(0.0)
        result_hit.distance = max_distance
        result_color = wp.vec3f(0.0)

        ray_origin_local, ray_direction_local = map_ray_to_local_scaled(transform, scale, ray_origin, ray_direction)

        hit_index = wp.int32(0)
        min_distance = wp.float32(0.0)
        ray_transmittance = wp.float32(1.0)

        hit_distances = wp.vector(max_distance, length=wp.static(config.gaussians_max_num_hits), dtype=wp.float32)
        hit_indices = wp.vector(-1, length=wp.static(config.gaussians_max_num_hits), dtype=wp.int32)
        hit_alphas = wp.vector(0.0, length=wp.static(config.gaussians_max_num_hits), dtype=wp.float32)

        while ray_transmittance > 0.003:
            num_hits = wp.int32(0)

            for i in range(wp.static(config.gaussians_max_num_hits)):
                hit_distances[i] = max_distance - min_distance

            query = wp.bvh_query_ray(
                gaussian_data.bvh_id, ray_origin_local + ray_direction_local * min_distance, ray_direction_local
            )

            while wp.bvh_query_next(query, hit_index, hit_distances[-1]):
                hit_alpha, hit_distance = ray_gsplat_hit_response(
                    gaussian_data.transforms[hit_index],
                    gaussian_data.scales[hit_index],
                    gaussian_data.opacities[hit_index],
                    gaussian_data.min_response,
                    ray_origin_local,
                    ray_direction_local,
                    hit_distances[-1],
                )

                if hit_distance > -1:
                    if num_hits < wp.static(config.gaussians_max_num_hits):
                        num_hits += 1

                    for h in range(num_hits):
                        if hit_distance < hit_distances[h]:
                            for hh in range(num_hits - 1, h, -1):
                                hit_distances[hh] = hit_distances[hh - 1]
                                hit_indices[hh] = hit_indices[hh - 1]
                                hit_alphas[hh] = hit_alphas[hh - 1]
                            hit_distances[h] = hit_distance
                            hit_indices[h] = hit_index
                            hit_alphas[h] = hit_alpha
                            break

            if num_hits == 0:
                break

            for hit in range(num_hits):
                hit_index = hit_indices[hit]

                color = SH_C0 * wp.vec3f(
                    gaussian_data.sh_coeffs[hit_index][0],
                    gaussian_data.sh_coeffs[hit_index][1],
                    gaussian_data.sh_coeffs[hit_index][2],
                ) + wp.vec3f(0.5)

                opacity = hit_alphas[hit]
                result_color += color * opacity * ray_transmittance
                ray_transmittance *= 1.0 - opacity

                if ray_transmittance < wp.static(config.gaussians_min_transmittance):
                    result_hit.distance = wp.min(hit_distances[hit], result_hit.distance)

            min_distance = hit_distances[-1] + wp.float32(1e-06)

            if wp.static(config.gaussians_mode) == GaussianRenderMode.FAST:
                break

        if ray_transmittance < wp.static(config.gaussians_min_transmittance):
            result_hit.hit = True
            result_color /= 1.0 - ray_transmittance

        return result_hit, result_color

    return shade
