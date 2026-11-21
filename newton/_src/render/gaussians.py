# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from typing import Any

import warp as wp

from ..core import MAXVAL
from ..geometry import Gaussian
from ..geometry.bvh import compute_ellipsoid_bounds
from ..geometry.raycast import map_ray_to_local
from ..geometry.types import SH_C0
from ..math import safe_div
from ..utils.color import ColorSpace, srgb_to_linear_wp
from .types import GaussianRenderMode, RenderConfig, RenderState


@wp.func
def compute_gaussian_bounds(gaussians_data: Gaussian.Data, tid: wp.int32) -> tuple[wp.vec3f, wp.vec3f]:
    transform = gaussians_data.transforms[tid]
    scale = gaussians_data.scales[tid]

    mod = gaussians_data.min_response / wp.max(gaussians_data.opacities[tid], wp.float32(1e-6))
    min_response = wp.clamp(mod, wp.float32(1e-6), wp.float32(0.97))
    ks = wp.sqrt(wp.log(min_response) / wp.float32(-0.5))
    scale = wp.vec3f(scale[0] * ks, scale[1] * ks, scale[2] * ks)

    return compute_ellipsoid_bounds(transform, scale)


@wp.kernel
def compute_gaussian_bvh_bounds(
    gaussians_data: Gaussian.Data,
    lowers: wp.array[wp.vec3f],
    uppers: wp.array[wp.vec3f],
):
    tid = wp.tid()
    lower, upper = compute_gaussian_bounds(gaussians_data, tid)
    lowers[tid] = lower
    uppers[tid] = upper


@wp.kernel(enable_backward=False)
def compute_asset_splat_bounds(
    gaussians_data: wp.array[Gaussian.Data],
    asset: wp.int32,
    lowers: wp.array[wp.vec3f],
    uppers: wp.array[wp.vec3f],
):
    """Bounds of each splat of a finalized asset, like :func:`compute_gaussian_bvh_bounds`."""
    tid = wp.tid()
    lower, upper = compute_gaussian_bounds(gaussians_data[asset], tid)
    lowers[tid] = lower
    uppers[tid] = upper


@wp.func
def sorting_mode_ray_hit_distance(ray_origin: wp.vec3f, ray_direction: wp.vec3f) -> wp.float32:
    numerator = -wp.dot(ray_origin, ray_direction)
    return safe_div(numerator, wp.dot(ray_direction, ray_direction))


@wp.func
def sorting_mode_camera_distance(
    gaussian_center: wp.vec3f,
    ray_origin: wp.vec3f,
    ray_direction: wp.vec3f,
) -> wp.float32:
    """Parametric *t* where the Gaussian center projects onto the ray."""
    to_center = gaussian_center - ray_origin
    return safe_div(wp.dot(to_center, ray_direction), wp.dot(ray_direction, ray_direction))


@wp.func
def sorting_mode_z_depth(
    gaussian_center_world: wp.vec3f,
    camera_forward: wp.vec3f,
    ray_origin_world: wp.vec3f,
    ray_direction_world: wp.vec3f,
) -> wp.float32:
    """Parametric *t* at which the ray reaches the Gaussian center's depth plane."""
    return safe_div(
        wp.dot(camera_forward, gaussian_center_world - ray_origin_world),
        wp.dot(camera_forward, ray_direction_world),
    )


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
    sorting_mode: wp.int32,
    ray_origin_world: wp.vec3f,
    ray_direction_world: wp.vec3f,
    camera_forward: wp.vec3f,
    max_distance: wp.float32,
) -> tuple[wp.float32, wp.float32]:
    ray_origin_local, ray_direction_local = map_ray_to_local(transform, ray_origin_world, ray_direction_world, scale)

    hit_distance = 0.0
    if sorting_mode == wp.static(Gaussian.SortingMode.CAMERA_DISTANCE):
        gaussian_center = wp.transform_get_translation(transform)
        hit_distance = sorting_mode_camera_distance(gaussian_center, ray_origin_local, ray_direction_local)
    elif sorting_mode == wp.static(Gaussian.SortingMode.Z_DEPTH):
        gaussian_center = wp.transform_get_translation(transform)
        gaussian_center_world = wp.transform_point(transform, wp.cw_mul(gaussian_center, scale))
        hit_distance = sorting_mode_z_depth(
            gaussian_center_world, camera_forward, ray_origin_world, ray_direction_world
        )
    else:
        hit_distance = sorting_mode_ray_hit_distance(ray_origin_local, ray_direction_local)

    if hit_distance > 0.0 and hit_distance < max_distance:
        max_response = canonical_ray_max_kernel_response(ray_origin_local, wp.normalize(ray_direction_local))

        alpha = wp.min(wp.float32(1.0), max_response * opacity)
        if max_response > min_response and alpha > wp.static(1.0 / 255.0):
            return alpha, hit_distance
    return 0.0, -1.0


SPLAT_BITS = 24
"""Bits of a splat hit's id that hold the splat; the bits above hold the slot of its Gaussian shape."""
MAX_SPLATS = 1 << SPLAT_BITS
"""Splats a Gaussian asset may hold."""
MAX_SLOTS = 1 << (32 - SPLAT_BITS)
"""Gaussian shapes one ray may cross."""


@wp.func
def splat_hit_id(slot: wp.int32, splat: wp.int32) -> wp.uint32:
    """Id of a hit on ``splat`` of the Gaussian shape in ``slot``."""
    return (wp.uint32(slot) << wp.uint32(SPLAT_BITS)) | wp.uint32(splat)


@wp.func
def hit_before(t0: wp.float32, id0: wp.uint32, t1: wp.float32, id1: wp.uint32) -> wp.bool:
    """Whether splat hit ``(t0, id0)`` sorts before ``(t1, id1)``; ids break distance ties, so hits
    have one order however they are gathered."""
    if t0 != t1:
        return t0 < t1
    return id0 < id1


@functools.cache
def create_hit_buffer(capacity: int) -> tuple[type, wp.Function, wp.Function]:
    """Sorted buffer of the nearest ``capacity`` splat hits along a ray, by distance, then id.

    Returns the buffer struct, a function returning an empty buffer, and a function inserting a hit.
    Inserting a hit already in the buffer leaves it unchanged, as OptiX may report a primitive twice.
    """
    distances = wp.types.vector(capacity, wp.float32)
    ids = wp.types.vector(capacity, wp.uint32)
    # Built from annotation objects: postponed annotations cannot name the capacity.
    SplatHits = wp.struct(type("SplatHits", (), {"__annotations__": {"t": distances, "id": ids}}))

    @wp.func
    def empty_hits():
        hits = SplatHits()
        hits.t = distances(MAXVAL)
        hits.id = ids(wp.uint32(0xFFFFFFFF))
        return hits

    @wp.func
    def insert_hit(hits: Any, t: wp.float32, id: wp.uint32):
        carry_t = t
        carry_id = id
        for i in range(wp.static(capacity)):
            if carry_t == hits.t[i] and carry_id == hits.id[i]:
                return hits
            if hit_before(carry_t, carry_id, hits.t[i], hits.id[i]):
                # Bubble the new hit into place, carrying the displaced one onward.
                t_i = hits.t[i]
                id_i = hits.id[i]
                hits.t[i] = carry_t
                hits.id[i] = carry_id
                carry_t = t_i
                carry_id = id_i
        return hits

    return SplatHits, empty_hits, insert_hit


@wp.struct
class SplatBlend:
    """Front-to-back blend of the splats along a ray."""

    transmittance: wp.float32
    srgb_color: wp.vec3f
    """Weighted colors of splats authored in sRGB, blended in sRGB as 3DGS is trained."""
    srgb_weight: wp.float32
    linear_color: wp.vec3f
    """Weighted colors of splats authored in linear color."""
    distance: wp.float32
    """Distance at which the transmittance first fell below the opacity threshold, or ``MAXVAL``."""
    shape: wp.int32
    """Gaussian shape of the splat at :attr:`distance`."""


@wp.func
def start_blend() -> SplatBlend:
    blend = SplatBlend()
    blend.transmittance = 1.0
    blend.distance = MAXVAL
    blend.shape = -1
    return blend


def create_blend_function(config: RenderConfig) -> wp.Function:
    @wp.func
    def blend_splat(
        blend: SplatBlend,
        t: wp.float32,
        si: wp.int32,
        splat: wp.int32,
        shape_transforms: wp.array[wp.transformf],
        shape_sizes: wp.array[wp.vec3f],
        shape_source_ptr: wp.array[wp.uint64],
        gaussians_data: wp.array[Gaussian.Data],
        ray_origin: wp.vec3f,
        ray_direction: wp.vec3f,
        camera_forward: wp.vec3f,
    ) -> SplatBlend:
        """Blend one splat hit behind the splats already blended."""
        data = gaussians_data[int(shape_source_ptr[si])]
        origin, direction = map_ray_to_local(shape_transforms[si], ray_origin, ray_direction, shape_sizes[si])
        alpha, _t = ray_gsplat_hit_response(
            data.transforms[splat],
            data.scales[splat],
            data.opacities[splat],
            data.min_response,
            data.sorting_mode,
            origin,
            direction,
            camera_forward,
            MAXVAL,
        )
        color = wp.max(
            SH_C0 * wp.vec3f(data.sh_coeffs[splat][0], data.sh_coeffs[splat][1], data.sh_coeffs[splat][2])
            + wp.vec3f(0.5),
            wp.vec3f(0.0),
        )
        weight = alpha * blend.transmittance
        if data.sh_color_space == ColorSpace.SRGB:
            blend.srgb_color += color * weight
            blend.srgb_weight += weight
        else:
            blend.linear_color += color * weight
        blend.transmittance *= 1.0 - alpha
        if blend.transmittance < wp.static(config.gaussians_min_transmittance) and blend.distance == MAXVAL:
            blend.distance = t
            blend.shape = si
        return blend

    return blend_splat


@wp.func
def finish_blend(blend: SplatBlend) -> tuple[wp.float32, wp.vec3f, wp.int32]:
    """Distance, linear color and shape of the blended splats; distance is -1 unless they are opaque enough."""
    if blend.distance == MAXVAL:
        return -1.0, wp.vec3f(0.0), -1
    opacity = 1.0 - blend.transmittance
    color = blend.linear_color / opacity
    if blend.srgb_weight > 0.0:
        # sRGB is decoded once per ray, after blending.
        color += srgb_to_linear_wp(blend.srgb_color / blend.srgb_weight) * (blend.srgb_weight / opacity)
    return blend.distance, color, blend.shape


def create_shade_function(config: RenderConfig, state: RenderState) -> wp.Function:
    """Blend the splats of the Gaussian shapes a ray crosses, nearest first, in front of ``max_distance``.

    Each round gathers the nearest :attr:`RenderConfig.gaussians_max_num_hits` splat hits after those
    already blended, across all shapes, and blends them; ``FAST`` renders stop after one round. A hit's
    slot is its shape's position in ``shapes_hit``.
    """
    capacity = config.gaussians_max_num_hits
    _SplatHits, empty_hits, insert_hit = create_hit_buffer(capacity)
    blend_splat = create_blend_function(config)

    @wp.func
    def shade(
        shapes_hit: Any,
        shape_count: wp.int32,
        shape_transforms: wp.array[wp.transformf],
        shape_sizes: wp.array[wp.vec3f],
        shape_source_ptr: wp.array[wp.uint64],
        gaussians_data: wp.array[Gaussian.Data],
        ray_origin: wp.vec3f,
        ray_direction: wp.vec3f,
        camera_forward: wp.vec3f,
        max_distance: wp.float32,
    ) -> tuple[wp.float32, wp.vec3f, wp.int32]:
        blend = start_blend()
        # Hits up to (last_t, last_id) are blended.
        last_t = wp.float32(-MAXVAL)
        last_id = wp.uint32(0)
        while blend.transmittance > 0.003:
            hits = empty_hits()
            for slot in range(shape_count):
                si = shapes_hit[slot]
                data = gaussians_data[int(shape_source_ptr[si])]
                origin, direction = map_ray_to_local(shape_transforms[si], ray_origin, ray_direction, shape_sizes[si])
                query = wp.bvh_query_ray(data.bvh_id, origin, direction)
                splat = wp.int32(0)
                while wp.bvh_query_next(query, splat, wp.min(hits.t[-1], max_distance)):
                    alpha, t = ray_gsplat_hit_response(
                        data.transforms[splat],
                        data.scales[splat],
                        data.opacities[splat],
                        data.min_response,
                        data.sorting_mode,
                        origin,
                        direction,
                        camera_forward,
                        wp.min(hits.t[-1], max_distance),
                    )
                    id = splat_hit_id(slot, splat)
                    if alpha > 0.0 and hit_before(last_t, last_id, t, id):
                        hits = insert_hit(hits, t, id)
            count = int(0)
            for i in range(wp.static(capacity)):
                if hits.t[i] < MAXVAL:
                    id = hits.id[i]
                    blend = blend_splat(
                        blend,
                        hits.t[i],
                        shapes_hit[int(id >> wp.uint32(SPLAT_BITS))],
                        int(id & wp.uint32(MAX_SPLATS - 1)),
                        shape_transforms,
                        shape_sizes,
                        shape_source_ptr,
                        gaussians_data,
                        ray_origin,
                        ray_direction,
                        camera_forward,
                    )
                    count += 1
                    last_t = hits.t[i]
                    last_id = id
            if count == 0 or wp.static(config.gaussians_mode == GaussianRenderMode.FAST):
                break
        return finish_blend(blend)

    return shade
