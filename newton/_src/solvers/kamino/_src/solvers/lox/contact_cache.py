# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Match angular contact reactions across timesteps without host readback."""

from __future__ import annotations

import warp as wp

from ...geometry.keying import KeySorter, binary_search_find_range_start

wp.set_module_options({"enable_backward": False})


@wp.struct
class _CacheData:
    count: wp.array[wp.int32]
    key: wp.array[wp.uint64]
    world: wp.array[wp.int32]
    body: wp.array[wp.int32]
    position: wp.array[wp.vec3f]
    torque: wp.array[wp.vec3f]
    current_position: wp.array[wp.vec3f]


@wp.kernel
def _import_reactions(
    source_count: wp.array[wp.int32],
    source_map: wp.array[wp.int32],
    source_key: wp.array[wp.uint64],
    source_world: wp.array[wp.int32],
    source_body: wp.array[wp.vec2i],
    source_position: wp.array[wp.vec3f],
    source_frame: wp.array[wp.quatf],
    body_pose: wp.array[wp.transformf],
    time_step: wp.array[wp.float32],
    sorted_keys: wp.array[wp.uint64],
    sorted_to_source: wp.array[wp.int32],
    tolerance_squared: float,
    cache: _CacheData,
    angular_reaction: wp.array[wp.vec3f],
):
    source = wp.tid()
    if source >= source_count[0]:
        return
    internal = source_map[source]
    if internal < 0:
        return
    body = source_body[source][1]
    position = source_position[source]
    if body >= 0:
        position = wp.transform_point(wp.transform_inverse(body_pose[body]), position)
    cache.current_position[source] = position

    key = source_key[source]
    index = binary_search_find_range_start(0, cache.count[0], key, sorted_keys)
    best = int(-1)
    distance_squared = tolerance_squared
    while index >= 0 and index < cache.count[0]:
        if sorted_keys[index] != key:
            break
        previous = sorted_to_source[index]
        if cache.world[previous] == source_world[source] and cache.body[previous] == body:
            delta = cache.position[previous] - position
            candidate_distance = wp.dot(delta, delta)
            if candidate_distance < distance_squared:
                distance_squared = candidate_distance
                best = previous
        index += 1
    if best >= 0:
        local = wp.quat_rotate_inv(source_frame[source], cache.torque[best])
        angular_reaction[internal] = time_step[source_world[source]] * wp.vec3f(local[2], local[0], local[1])


@wp.kernel
def _export_reactions(
    source_count: wp.array[wp.int32],
    source_map: wp.array[wp.int32],
    source_key: wp.array[wp.uint64],
    source_world: wp.array[wp.int32],
    source_body: wp.array[wp.vec2i],
    source_frame: wp.array[wp.quatf],
    inverse_time_step: wp.array[wp.float32],
    angular_reaction: wp.array[wp.vec3f],
    cache: _CacheData,
):
    source = wp.tid()
    cache.world[source] = -1
    if source >= source_count[0]:
        return
    cache.key[source] = source_key[source]
    internal = source_map[source]
    if internal < 0:
        return
    world = source_world[source]
    angular = angular_reaction[internal]
    local = wp.vec3f(angular[1], angular[2], angular[0])
    cache.torque[source] = inverse_time_step[world] * wp.quat_rotate(source_frame[source], local)
    cache.position[source] = cache.current_position[source]
    cache.world[source] = world
    cache.body[source] = source_body[source][1]


@wp.kernel
def _copy_bounded_count(source: wp.array[wp.int32], capacity: int, target: wp.array[wp.int32]):
    target[0] = wp.clamp(source[0], 0, capacity)


@wp.kernel
def _reset_cache(world_mask: wp.array[wp.bool], world: wp.array[wp.int32]):
    source = wp.tid()
    wid = world[source]
    if wid >= 0:
        if not world_mask or world_mask[wid]:
            world[source] = -1


class AngularContactCache:
    """Cache world torques by geometry key and body-local contact position.

    Import once after preparing contacts at the beginning of the timestep;
    export after solving using the same source contact ordering. Matching
    searches only the cached geometry pair and selects its closest point.
    The common spatial warm-start projection applies the current material cap.
    """

    def __init__(self, capacity: int, device: wp.DeviceLike, *, match_tolerance: float = 1.0e-3):
        self.capacity = capacity
        self.device = wp.get_device(device)
        self.match_tolerance = match_tolerance
        self.data = _CacheData()
        self.data.count = wp.zeros(1, dtype=wp.int32, device=self.device)
        for name, dtype in (
            ("key", wp.uint64),
            ("body", wp.int32),
            ("position", wp.vec3f),
            ("torque", wp.vec3f),
            ("current_position", wp.vec3f),
        ):
            setattr(self.data, name, wp.zeros(capacity, dtype=dtype, device=self.device))
        self.data.world = wp.full(capacity, -1, dtype=wp.int32, device=self.device)
        self.sorter = KeySorter(capacity, self.device) if capacity else None

    def import_reactions(self, contacts, source_to_internal, body_pose, time_step, angular_reaction):
        """Restore angular impulses and capture contact positions before integration."""
        angular_reaction.zero_()
        if not self.capacity:
            return
        wp.launch(
            _import_reactions,
            dim=self.capacity,
            inputs=[
                contacts.model_active_contacts,
                source_to_internal,
                contacts.key,
                contacts.wid,
                contacts.bid_AB,
                contacts.position_B,
                contacts.frame,
                body_pose,
                time_step,
                self.sorter.sorted_keys,
                self.sorter.sorted_to_unsorted_map,
                self.match_tolerance**2,
                self.data,
            ],
            outputs=[angular_reaction],
            device=self.device,
        )

    def export_reactions(self, contacts, source_to_internal, inverse_time_step, angular_reaction):
        """Replace the previous cache, including removal of disappeared contacts."""
        if not self.capacity:
            return
        wp.launch(
            _export_reactions,
            dim=self.capacity,
            inputs=[
                contacts.model_active_contacts,
                source_to_internal,
                contacts.key,
                contacts.wid,
                contacts.bid_AB,
                contacts.frame,
                inverse_time_step,
                angular_reaction,
                self.data,
            ],
            device=self.device,
        )
        wp.launch(
            _copy_bounded_count,
            dim=1,
            inputs=[contacts.model_active_contacts, self.capacity],
            outputs=[self.data.count],
            device=self.device,
        )
        self.sorter.sort(self.data.count, self.data.key)

    def reset(self, world_mask: wp.array[wp.bool] | None = None):
        """Invalidate all cached contacts or only those in selected worlds."""
        if not self.capacity:
            return
        wp.launch(_reset_cache, dim=self.capacity, inputs=[world_mask, self.data.world], device=self.device)
        if world_mask is None:
            self.data.count.zero_()
