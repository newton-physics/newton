# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Carry LOX spin and rolling reactions across pipeline-matched contacts."""

from __future__ import annotations

import warp as wp

from ..warmstart import invalidate_contact_worlds

__all__ = ["WarmstarterAngularContacts"]

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _restore_angular_reactions(
    count: wp.array[wp.int32],
    source_to_internal: wp.array[wp.int32],
    matched_slots: wp.array[wp.int32],
    world: wp.array[wp.int32],
    frame: wp.array[wp.quatf],
    dt: wp.array[wp.float32],
    cached_world: wp.array[wp.int32],
    cached_torque: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
):
    cid = wp.tid()
    if cid >= count[0]:
        return
    internal = source_to_internal[cid]
    previous = matched_slots[cid]
    wid = world[cid]
    if internal >= 0 and previous >= 0 and cached_world[previous] == wid:
        local = wp.quat_rotate_inv(frame[cid], cached_torque[previous])
        angular_reaction[internal] = dt[wid] * wp.vec3f(local[2], local[0], local[1])


@wp.kernel
def _store_angular_reactions(
    count: wp.array[wp.int32],
    source_to_internal: wp.array[wp.int32],
    world: wp.array[wp.int32],
    frame: wp.array[wp.quatf],
    inv_dt: wp.array[wp.float32],
    angular_reaction: wp.array[wp.vec3f],
    cached_world: wp.array[wp.int32],
    cached_torque: wp.array[wp.vec3f],
):
    cid = wp.tid()
    cached_world[cid] = -1
    if cid >= count[0]:
        return
    internal = source_to_internal[cid]
    if internal >= 0:
        wid = world[cid]
        angular = angular_reaction[internal]
        local = wp.vec3f(angular[1], angular[2], angular[0])
        cached_torque[cid] = inv_dt[wid] * wp.quat_rotate(frame[cid], local)
        cached_world[cid] = wid


class WarmstarterAngularContacts:
    """Cache spatial contact torques in world axes."""

    def __init__(self, capacity: int, device: wp.DeviceLike):
        self.capacity = capacity
        self.device = wp.get_device(device)
        self.world = wp.full(capacity, -1, dtype=wp.int32, device=device)
        self.torque = wp.empty(capacity, dtype=wp.vec3f, device=device)

    def warmstart(self, contacts, source_to_internal, time_step, angular_reaction):
        """Restore matched spin and rolling impulses in their current frames."""
        angular_reaction.zero_()
        if not self.capacity:
            return
        wp.launch(
            _restore_angular_reactions,
            dim=self.capacity,
            inputs=[
                contacts.model_active_contacts,
                source_to_internal,
                contacts.data.matched_slots,
                contacts.wid,
                contacts.frame,
                time_step,
                self.world,
                self.torque,
                angular_reaction,
            ],
            device=self.device,
        )

    def update(self, contacts, source_to_internal, inverse_time_step, angular_reaction):
        """Store the solved spin and rolling reactions."""
        if not self.capacity:
            return
        wp.launch(
            _store_angular_reactions,
            dim=self.capacity,
            inputs=[
                contacts.model_active_contacts,
                source_to_internal,
                contacts.wid,
                contacts.frame,
                inverse_time_step,
                angular_reaction,
                self.world,
                self.torque,
            ],
            device=self.device,
        )

    def reset(self, world_mask: wp.array[wp.bool] | None = None):
        """Invalidate cached torques in the selected worlds."""
        if self.capacity:
            wp.launch(invalidate_contact_worlds, dim=self.capacity, inputs=[world_mask, self.world], device=self.device)
