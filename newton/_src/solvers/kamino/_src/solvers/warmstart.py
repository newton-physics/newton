# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warm-start joint limits and transfer cached reactions using pipeline contact matches."""

from __future__ import annotations

from weakref import WeakKeyDictionary

import warp as wp

from .....geometry.contact_match import ContactMatcher
from .....sim.contacts import GENERATION_SENTINEL, Contacts
from ..geometry.contacts import ContactsKamino
from ..geometry.keying import KeySorter, binary_search_find_range_start, make_bitmask, uint64_sentinel_value
from ..kinematics.limits import LimitsKamino, LimitsKaminoData

###
# Module interface
###

__all__ = [
    "ContactMatchMap",
    "WarmstarterContacts",
    "WarmstarterLimits",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


###
# Kernels
###


@wp.kernel
def _invalidate_keys_from_select_worlds(
    world_id: wp.array[wp.int32],
    keys: wp.array[wp.uint64],
    world_mask: wp.array[wp.bool],
):
    """Reset limit keys to the sentinel value in selected worlds."""
    tid = wp.tid()
    wid = world_id[tid]
    if world_mask[wid]:
        keys[tid] = uint64_sentinel_value()


@wp.kernel
def _warmstart_limits_by_matched_jid_dof_key(
    # Inputs - Previous:
    sorted_limit_keys_old: wp.array[wp.uint64],
    sorted_to_unsorted_map_old: wp.array[wp.int32],
    num_active_limits_old: wp.array[wp.int32],
    limit_velocity_old: wp.array[wp.float32],
    limit_reaction_old: wp.array[wp.float32],
    # Inputs - Next:
    num_active_limits_new: wp.array[wp.int32],
    limit_key_new: wp.array[wp.uint64],
    # Outputs:
    limit_reaction_new: wp.array[wp.float32],
    limit_velocity_new: wp.array[wp.float32],
):
    """
    Match current limits to previous timestep limits using joint-DoF index pair keys.

    For each current limit, finds a matching limit from the
    previous step using a binary-search on sorted keys (O(log n))
    """
    # Retrieve the limit index as the thread index
    lid = wp.tid()

    # Perform early exit if out of active bounds
    if lid >= num_active_limits_new[0]:
        return

    # Retrieve number of active old limits and the target key to search for
    num_active_old = num_active_limits_old[0]
    target_key = limit_key_new[lid]

    # Perform binary search to find the start index of the
    # target key - i.e. assuming a joint-DoF index pair key
    start = binary_search_find_range_start(0, num_active_old, target_key, sorted_limit_keys_old)

    # If key not found, then mark as a new limit and skip further processing
    # NOTE: This means that a new limit has become active
    if start == -1:
        limit_reaction_new[lid] = 0.0
        limit_velocity_new[lid] = 0.0
    else:
        # Retrieve the old limit index from the sorted->unsorted map
        lid_old = sorted_to_unsorted_map_old[start]
        reaction_old = limit_reaction_old[lid_old]
        velocity_old = limit_velocity_old[lid_old]

        # Retrieve the matched limit reaction and velocity from
        # the old limit and store them to the new limit
        limit_reaction_new[lid] = wp.max(reaction_old, 0.0)
        limit_velocity_new[lid] = wp.max(velocity_old, 0.0)


def warmstart_limits_by_matched_jid_dof_key(
    sorter: KeySorter,
    cache: LimitsKaminoData,
    limits: LimitsKaminoData,
):
    """
    Warm-starts limits by matching joint-DoF index-pair keys.

    Args:
        sorter: The key sorter used to sort cached limit keys.
        cache: The cached limits data from the previous simulation step.
        limits: The current limits data to be warm-started.
    """
    # First sort the keys of cached limits to facilitate binary search
    sorter.sort(num_active_keys=cache.model_active_limits, keys=cache.key)

    # Launch kernel to warmstart limits by matching jid keys
    wp.launch(
        kernel=_warmstart_limits_by_matched_jid_dof_key,
        dim=limits.model_max_limits_host,
        inputs=[
            # Inputs - Previous:
            sorter.sorted_keys,
            sorter.sorted_to_unsorted_map,
            cache.model_active_limits,
            cache.velocity,
            cache.reaction,
            # Inputs - Next:
            limits.model_active_limits,
            limits.key,
            # Outputs:
            limits.reaction,
            limits.velocity,
        ],
        device=sorter.device,
    )


class WarmstarterLimits:
    """
    Provides a unified mechanism for warm-starting unilateral limit constraints.
    """

    def __init__(self, limits: LimitsKamino | None = None):
        """
        Initializes the limits warmstarter using the allocations of the provided limits container.

        Args:
            limits: The limits container whose allocations are used to initialize the warmstarter.
        """
        # Store the device of the provided contacts container
        self._device: wp.DeviceLike = limits.device if limits is not None else None

        # Declare the internal limits cache
        self._cache: LimitsKaminoData | None = None

        # Check if the limits container has allocations and skip cache allocations if not
        if limits is None or (limits is not None and limits.model_max_limits_host <= 0):
            return

        # Allocate contact data cache based on the those of the provided contacts container
        with wp.ScopedDevice(self._device):
            self._cache = LimitsKaminoData(
                model_max_limits_host=limits.model_max_limits_host,
                world_max_limits_host=limits.world_max_limits_host,
                model_active_limits=wp.zeros_like(limits.model_active_limits),
                key=wp.zeros_like(limits.key),
                velocity=wp.zeros_like(limits.velocity),
                reaction=wp.zeros_like(limits.reaction),
                wid=wp.zeros_like(limits.wid),
            )

        # Create a key sorter that can handle the maximum number of limits
        self._sorter = KeySorter(max_num_keys=limits.model_max_limits_host, device=self._device)

    @property
    def device(self) -> wp.DeviceLike:
        """
        Returns the device on which the warmstarter operates.
        """
        return self._device

    @property
    def cache(self) -> LimitsKaminoData | None:
        """
        Returns the internal limits cache data.
        """
        return self._cache

    def warmstart(self, limits: LimitsKamino):
        """
        Warm-starts the provided contacts container using the internal cache.

        Limits are matched by joint and degree of freedom.

        Args:
            model: The model containing simulation parameters.
            data: The model data containing body states.
            limits: The limits container to warm-start.
        """
        # Early exit if no cache is allocated
        if self._cache is None:
            return

        # Otherwise, perform warm-starting using matched jid-dof keys
        warmstart_limits_by_matched_jid_dof_key(
            sorter=self._sorter,
            cache=self._cache,
            limits=limits.data,
        )

    def update(self, limits: LimitsKamino | None = None):
        """
        Updates the warmstarter's internal cache with the provided limits data.

        Args:
            limits: The limits container from which to update the cache.
        """
        # Early exit if no cache is allocated or no limits data is provided
        if self._cache is None or limits is None:
            return

        # Otherwise, copy over the limits data to the internal cache
        wp.copy(self._cache.model_active_limits, limits.model_active_limits)
        wp.copy(self._cache.key, limits.key)
        wp.copy(self._cache.velocity, limits.velocity)
        wp.copy(self._cache.reaction, limits.reaction)
        wp.copy(self._cache.wid, limits.wid)

    def reset(self, world_mask: wp.array[wp.bool] | None = None):
        """
        Resets the warmstarter's internal cache, for all or a subset of the worlds.

        Args:
            world_mask: If provided, cached data is invalidated for designated worlds (`True` in the mask),
                so that new active limits in these worlds won't match any cached active limit.
                Otherwise, the entire cache is zeroed out.
        """
        if self._cache is None:
            return
        if world_mask is not None:
            wp.launch(
                _invalidate_keys_from_select_worlds,
                dim=self._cache.model_max_limits_host,
                inputs=[self._cache.wid, self._cache.key, world_mask],
                device=self.device,
            )
        else:
            self._cache.model_active_limits.zero_()
            self._cache.key.fill_(make_bitmask(63))
            self._cache.reaction.zero_()
            self._cache.velocity.zero_()


@wp.kernel
def _resolve_contact_matches(
    count: wp.array[wp.int32],
    source_index: wp.array[wp.int32],
    match_index: wp.array[wp.int32],
    generation: wp.array[wp.int32],
    match_generation: wp.array[wp.int32],
    stream: int,
    previous_generation: wp.array[wp.int32],
    previous_stream: wp.array[wp.int32],
    previous_slots: wp.array[wp.int32],
    matched_slots: wp.array[wp.int32],
):
    cid = wp.tid()
    matched_slots[cid] = -1
    if cid >= count[0] or previous_stream[0] != stream:
        return
    source = source_index[cid]
    previous = int(-1)
    if generation[0] == previous_generation[0]:
        previous = source
    elif match_generation[0] == previous_generation[0]:
        previous = match_index[source]
    if previous >= 0 and previous < previous_slots.shape[0]:
        matched_slots[cid] = previous_slots[previous]


@wp.kernel
def _record_contact_slots(
    count: wp.array[wp.int32],
    source_index: wp.array[wp.int32],
    generation: wp.array[wp.int32],
    stream: int,
    previous_generation: wp.array[wp.int32],
    previous_stream: wp.array[wp.int32],
    previous_slots: wp.array[wp.int32],
):
    cid = wp.tid()
    if cid == 0:
        previous_generation[0] = generation[0]
        previous_stream[0] = stream
    if cid < count[0]:
        previous_slots[source_index[cid]] = cid


class ContactMatchMap:
    """Translate pipeline contact indices into the previous compacted Kamino slots."""

    def __init__(self, device: wp.DeviceLike, capacity: int = 0):
        self._device = device
        self._streams: WeakKeyDictionary[Contacts, int] = WeakKeyDictionary()
        self._next_stream = 1
        self._previous_generation = wp.full(1, GENERATION_SENTINEL, dtype=wp.int32, device=device)
        self._previous_stream = wp.zeros(1, dtype=wp.int32, device=device)
        # Sized up front for the model's contact buffers so CUDA graph capture does not allocate.
        self._previous_slots = wp.full(capacity, -1, dtype=wp.int32, device=device)

    def resolve(self, source: Contacts | None, contacts: ContactsKamino) -> int:
        """Resolve pipeline matches into ``contacts.data.matched_slots``.

        Returns:
            The ID of the source buffer, or ``0`` when the source carries no matches.
            In that case ``matched_slots`` is left for :class:`WarmstarterContacts` to fill.
        """
        matched_slots = contacts.data.matched_slots
        if source is None or source.rigid_contact_match_index is None or source.rigid_contact_max == 0:
            return 0
        stream = self._streams.get(source)
        if stream is None:
            stream = self._next_stream
            self._next_stream += 1
            self._streams[source] = stream
        if self._previous_slots.size < source.rigid_contact_max:
            self._previous_slots = wp.full(source.rigid_contact_max, -1, dtype=wp.int32, device=self._device)
        wp.launch(
            _resolve_contact_matches,
            dim=contacts.model_max_contacts_host,
            inputs=[
                contacts.model_active_contacts,
                contacts.remap,
                source.rigid_contact_match_index,
                source.contact_generation,
                source.rigid_contact_match_generation,
                stream,
                self._previous_generation,
                self._previous_stream,
                self._previous_slots,
                matched_slots,
            ],
            device=self._device,
        )
        return stream

    def update(self, source: Contacts | None, contacts: ContactsKamino, stream: int) -> None:
        """Record the source slots for the newly solved contact set."""
        if stream == 0:
            self._previous_stream.zero_()
            return
        # Slots omitted by Kamino's conversion must remain unmapped for the next solve.
        self._previous_slots.fill_(-1)
        wp.launch(
            _record_contact_slots,
            dim=contacts.model_max_contacts_host,
            inputs=[
                contacts.model_active_contacts,
                contacts.remap,
                source.contact_generation,
                stream,
                self._previous_generation,
                self._previous_stream,
                self._previous_slots,
            ],
            device=self._device,
        )


@wp.kernel
def _restore_contact_reactions(
    count: wp.array[wp.int32],
    matched_slots: wp.array[wp.int32],
    frame: wp.array[wp.quatf],
    world: wp.array[wp.int32],
    previous_world: wp.array[wp.int32],
    previous_frame: wp.array[wp.quatf],
    previous_reaction: wp.array[wp.vec3f],
    previous_velocity: wp.array[wp.vec3f],
    reaction: wp.array[wp.vec3f],
    velocity: wp.array[wp.vec3f],
):
    cid = wp.tid()
    if cid >= count[0]:
        return
    reaction[cid] = wp.vec3f(0.0)
    velocity[cid] = wp.vec3f(0.0)
    previous = matched_slots[cid]
    if previous >= 0 and previous_world[previous] == world[cid]:
        rotation = wp.quat_inverse(frame[cid]) * previous_frame[previous]
        reaction[cid] = wp.quat_rotate(rotation, previous_reaction[previous])
        velocity[cid] = wp.quat_rotate(rotation, previous_velocity[previous])


@wp.kernel
def invalidate_contact_worlds(world_mask: wp.array[wp.bool], world: wp.array[wp.int32]):
    cid = wp.tid()
    wid = world[cid]
    if wid >= 0 and (not world_mask or world_mask[wid]):
        world[cid] = -1


@wp.kernel
def _gather_sorted_contacts(
    count: wp.array[wp.int32],
    sorted_to_slot: wp.array[wp.int32],
    gid_AB: wp.array[wp.vec2i],
    position_A: wp.array[wp.vec3f],
    position_B: wp.array[wp.vec3f],
    frame: wp.array[wp.quatf],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
):
    row = wp.tid()
    if row >= count[0]:
        return
    slot = sorted_to_slot[row]
    gid = gid_AB[slot]
    shape0[row] = gid[0]
    shape1[row] = gid[1]
    point0[row] = position_A[slot]
    point1[row] = position_B[slot]
    # Kamino contact frames carry the normal on their z-axis.
    normal[row] = wp.quat_rotate(frame[slot], wp.vec3f(0.0, 0.0, 1.0))


@wp.kernel
def _matched_slots_from_sorted_rows(
    count: wp.array[wp.int32],
    sorted_to_slot: wp.array[wp.int32],
    match_index: wp.array[wp.int32],
    previous_sorted_to_slot: wp.array[wp.int32],
    matched_slots: wp.array[wp.int32],
):
    row = wp.tid()
    if row >= count[0]:
        return
    previous = match_index[row]
    slot = int(-1)
    if previous >= 0:
        slot = previous_sorted_to_slot[previous]
    matched_slots[sorted_to_slot[row]] = slot


class WarmstarterContacts:
    """Carry linear reactions and velocities across matched contacts.

    Matches come from the Newton collision pipeline when it provides them.
    Otherwise this warm starter runs the same :class:`ContactMatcher` on the
    Kamino contacts, for example those of Kamino's own collision detector.
    """

    def __init__(self, contacts: ContactsKamino, geom_world: wp.array[wp.int32], world_count: int):
        self.world = wp.full_like(contacts.wid, -1)
        self.frame = wp.empty_like(contacts.frame)
        self.reaction = wp.empty_like(contacts.reaction)
        self.velocity = wp.empty_like(contacts.velocity)

        # Matcher state for contacts that arrive without pipeline matches.
        capacity = contacts.model_max_contacts_host
        device = contacts.device
        with wp.ScopedDevice(device):
            self._sorter = KeySorter(max_num_keys=capacity, device=device)
            self._matcher = ContactMatcher(
                capacity,
                sorter=None,
                shape_world=geom_world,
                world_count=world_count,
                shape_index_bits=31,
                # Kamino keys are bare geometry-pair keys.
                sub_key_bits=0,
                device=device,
            )
            self._buffer_id = self._matcher.buffer_id(self)
            self._generation = wp.zeros(1, dtype=wp.int32)
            self._match_generation = wp.zeros(1, dtype=wp.int32)
            # Contact points are world positions, so no shape has a body here.
            self._shape_body = wp.full(geom_world.shape[0], -1, dtype=wp.int32)
            self._body_q = wp.empty(1, dtype=wp.transform)
            self._shape0 = wp.empty(capacity, dtype=wp.int32)
            self._shape1 = wp.empty(capacity, dtype=wp.int32)
            self._point0 = wp.empty(capacity, dtype=wp.vec3)
            self._point1 = wp.empty(capacity, dtype=wp.vec3)
            self._normal = wp.empty(capacity, dtype=wp.vec3)
            self._match_index = wp.empty(capacity, dtype=wp.int32)
            self._previous_sorted_to_slot = wp.empty(capacity, dtype=wp.int32)

    def match(self, contacts: ContactsKamino) -> None:
        """Match the contacts against the previous solve and write ``contacts.data.matched_slots``."""
        count = contacts.model_active_contacts
        self._sorter.sort(num_active_keys=count, keys=contacts.key)
        sorted_keys = self._sorter.sorted_keys_int64
        sorted_to_slot = self._sorter.sorted_to_unsorted_map
        capacity = contacts.model_max_contacts_host
        wp.launch(
            _gather_sorted_contacts,
            dim=capacity,
            inputs=[
                count,
                sorted_to_slot,
                contacts.gid_AB,
                contacts.position_A,
                contacts.position_B,
                contacts.frame,
                self._shape0,
                self._shape1,
                self._point0,
                self._point1,
                self._normal,
            ],
            device=contacts.device,
        )
        sorted_args = (
            sorted_keys,
            count,
            self._point0,
            self._point1,
            self._shape0,
            self._shape1,
            self._normal,
            self._body_q,
            self._shape_body,
        )
        self._matcher.match(
            *sorted_args,
            self._match_index,
            buffer_id=self._buffer_id,
            match_generation_out=self._match_generation,
            device=contacts.device,
        )
        wp.launch(
            _matched_slots_from_sorted_rows,
            dim=capacity,
            inputs=[
                count,
                sorted_to_slot,
                self._match_index,
                self._previous_sorted_to_slot,
                contacts.data.matched_slots,
            ],
            device=contacts.device,
        )
        self._matcher.save_sorted_state(
            *sorted_args,
            buffer_id=self._buffer_id,
            contact_generation=self._generation,
            device=contacts.device,
        )
        wp.copy(self._previous_sorted_to_slot, sorted_to_slot, count=capacity)

    def warmstart(self, contacts: ContactsKamino, matched: bool = False) -> None:
        """Restore cached reactions on matched contacts.

        Args:
            contacts: The contacts to warm-start.
            matched: Whether ``contacts.data.matched_slots`` already holds the
                collision pipeline's matches for this step.
        """
        if matched:
            # The pipeline owns this step's history, so a later fallback must not match against an older one.
            self._matcher.reset()
        else:
            self.match(contacts)
        wp.launch(
            _restore_contact_reactions,
            dim=contacts.model_max_contacts_host,
            inputs=[
                contacts.model_active_contacts,
                contacts.data.matched_slots,
                contacts.frame,
                contacts.wid,
                self.world,
                self.frame,
                self.reaction,
                self.velocity,
                contacts.reaction,
                contacts.velocity,
            ],
            device=contacts.device,
        )

    def update(self, contacts: ContactsKamino) -> None:
        wp.copy(self.world, contacts.wid)
        wp.copy(self.frame, contacts.frame)
        wp.copy(self.reaction, contacts.reaction)
        wp.copy(self.velocity, contacts.velocity)

    def reset(self, world_mask: wp.array[wp.bool] | None = None) -> None:
        wp.launch(
            invalidate_contact_worlds, dim=self.world.size, inputs=[world_mask, self.world], device=self.world.device
        )
