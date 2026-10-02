# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warm start of the angular reactions of LOX spatial contacts.

Kamino's :class:`~newton._src.solvers.kamino._src.solvers.warmstart.WarmstarterContacts`
carries the linear contact reactions across time steps through the contacts
container. That container stores the linear reactions only, so
:class:`WarmstarterAngularContacts` plays the same role for the spin and rolling
reactions of LOX spatial contacts: it matches the detected contacts of a time step
to those of the previous step by geometry pair and body-local position, on the device.
"""

from __future__ import annotations

import warp as wp

from ...geometry.keying import KeySorter, binary_search_find_range_start

###
# Module interface
###

__all__ = ["WarmstarterAngularContacts"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


@wp.kernel
def _warmstart_angular_reactions(
    # Inputs:
    contacts_model_num: wp.array[wp.int32],
    source_map: wp.array[wp.int32],
    contacts_key: wp.array[wp.uint64],
    contacts_wid: wp.array[wp.int32],
    contacts_bid_AB: wp.array[wp.vec2i],
    contacts_position_B: wp.array[wp.vec3f],
    contacts_frame: wp.array[wp.quatf],
    data_bodies_q_i: wp.array[wp.transformf],
    model_time_dt: wp.array[wp.float32],
    sorted_keys: wp.array[wp.uint64],
    sorted_to_source: wp.array[wp.int32],
    cache_count: wp.array[wp.int32],
    cache_world: wp.array[wp.int32],
    cache_body: wp.array[wp.int32],
    cache_position: wp.array[wp.vec3f],
    cache_torque: wp.array[wp.vec3f],
    # Outputs:
    current_position: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
):
    cid = wp.tid()
    if cid >= contacts_model_num[0]:
        return
    internal = source_map[cid]
    if internal < 0:
        return
    wid = contacts_wid[cid]
    bid = contacts_bid_AB[cid][1]
    position = contacts_position_B[cid]
    if bid >= 0:
        position = wp.transform_point(wp.transform_inverse(data_bodies_q_i[bid]), position)
    current_position[cid] = position

    key = contacts_key[cid]
    index = binary_search_find_range_start(0, cache_count[0], key, sorted_keys)
    # Rolling moves the contact point across body B by about v dt per step, so the closest cached
    # point of the same geometry pair is kept whatever its distance. The warm-start projection then
    # applies the current friction budget.
    best = int(-1)
    distance_squared = float(0.0)
    while index >= 0 and index < cache_count[0]:
        if sorted_keys[index] != key:
            break
        previous = sorted_to_source[index]
        if cache_world[previous] == wid and cache_body[previous] == bid:
            delta = cache_position[previous] - position
            candidate_distance = wp.dot(delta, delta)
            if best < 0 or candidate_distance < distance_squared:
                distance_squared = candidate_distance
                best = previous
        index += 1
    if best >= 0:
        dt = model_time_dt[wid]
        local = wp.quat_rotate_inv(contacts_frame[cid], cache_torque[best])
        angular_reaction[internal] = dt * wp.vec3f(local[2], local[0], local[1])


@wp.kernel
def _update_angular_reactions(
    # Inputs:
    contacts_model_num: wp.array[wp.int32],
    source_map: wp.array[wp.int32],
    contacts_key: wp.array[wp.uint64],
    contacts_wid: wp.array[wp.int32],
    contacts_bid_AB: wp.array[wp.vec2i],
    contacts_frame: wp.array[wp.quatf],
    model_time_inv_dt: wp.array[wp.float32],
    angular_reaction: wp.array[wp.vec3f],
    current_position: wp.array[wp.vec3f],
    # Outputs:
    cache_key: wp.array[wp.uint64],
    cache_world: wp.array[wp.int32],
    cache_body: wp.array[wp.int32],
    cache_position: wp.array[wp.vec3f],
    cache_torque: wp.array[wp.vec3f],
):
    cid = wp.tid()
    cache_world[cid] = -1
    if cid >= contacts_model_num[0]:
        return
    cache_key[cid] = contacts_key[cid]
    internal = source_map[cid]
    if internal < 0:
        return
    wid = contacts_wid[cid]
    angular = angular_reaction[internal]
    local = wp.vec3f(angular[1], angular[2], angular[0])
    cache_torque[cid] = model_time_inv_dt[wid] * wp.quat_rotate(contacts_frame[cid], local)
    cache_position[cid] = current_position[cid]
    cache_world[cid] = wid
    cache_body[cid] = contacts_bid_AB[cid][1]


@wp.kernel
def _copy_bounded_count(
    # Inputs:
    contacts_model_num: wp.array[wp.int32],
    capacity: int,
    # Outputs:
    target: wp.array[wp.int32],
):
    target[0] = wp.clamp(contacts_model_num[0], 0, capacity)


@wp.kernel
def _reset_cache(
    # Inputs:
    world_mask: wp.array[wp.bool],
    # Outputs:
    cache_world: wp.array[wp.int32],
):
    tid = wp.tid()
    wid = cache_world[tid]
    if wid >= 0:
        if not world_mask or world_mask[wid]:
            cache_world[tid] = -1


###
# Interfaces
###


class WarmstarterAngularContacts:
    """Warm-start the spin and rolling reactions of spatial contacts, alongside Kamino's contact warm starter.

    Mirrors the ``warmstart``/``update``/``reset`` interface of Kamino's
    ``WarmstarterContacts``: :meth:`warmstart` restores the angular impulses of the
    current contacts after they are prepared at the beginning of a time step, and
    :meth:`update` stores the solved angular reactions, as world torques, for the
    next step. Matching searches only the cached geometry pair and selects its
    closest point in the frame of body B, at any distance, so that rolling contacts
    keep their warm start. The spatial warm-start projection then applies the
    current friction budget.
    """

    def __init__(self, capacity: int, device: wp.DeviceLike):
        self.capacity = capacity
        self.device = wp.get_device(device)
        self.count = wp.zeros(1, dtype=wp.int32, device=self.device)
        """Number of cached contacts."""
        self.key = wp.zeros(capacity, dtype=wp.uint64, device=self.device)
        """Geometry-pair key of each cached contact."""
        self.world = wp.full(capacity, -1, dtype=wp.int32, device=self.device)
        """World of each cached contact, or ``-1`` for an empty slot."""
        self.body = wp.zeros(capacity, dtype=wp.int32, device=self.device)
        """Second body of each cached contact."""
        self.position = wp.zeros(capacity, dtype=wp.vec3f, device=self.device)
        """Contact point in the frame of body B."""
        self.torque = wp.zeros(capacity, dtype=wp.vec3f, device=self.device)
        """Angular contact torque in world axes."""
        self.current_position = wp.zeros(capacity, dtype=wp.vec3f, device=self.device)
        """Body-frame contact point of the current step, cached at export."""
        self.sorter = KeySorter(capacity, self.device) if capacity else None

    ###
    # Public API
    ###

    def warmstart(self, contacts, source_to_internal, body_pose, time_step, angular_reaction):
        """Restore the angular impulses of the current contacts, and capture their positions before integration."""
        angular_reaction.zero_()
        if not self.capacity:
            return
        wp.launch(
            _warmstart_angular_reactions,
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
                self.count,
                self.world,
                self.body,
                self.position,
                self.torque,
            ],
            outputs=[self.current_position, angular_reaction],
            device=self.device,
        )

    def update(self, contacts, source_to_internal, inverse_time_step, angular_reaction):
        """Store the solved angular reactions of the current contacts, dropping disappeared contacts."""
        if not self.capacity:
            return
        wp.launch(
            _update_angular_reactions,
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
                self.current_position,
            ],
            outputs=[self.key, self.world, self.body, self.position, self.torque],
            device=self.device,
        )
        wp.launch(
            _copy_bounded_count,
            dim=1,
            inputs=[contacts.model_active_contacts, self.capacity],
            outputs=[self.count],
            device=self.device,
        )
        self.sorter.sort(self.count, self.key)

    def reset(self, world_mask: wp.array[wp.bool] | None = None):
        """Invalidate all cached contacts or only those in selected worlds."""
        if not self.capacity:
            return
        wp.launch(_reset_cache, dim=self.capacity, inputs=[world_mask], outputs=[self.world], device=self.device)
        if world_mask is None:
            self.count.zero_()
