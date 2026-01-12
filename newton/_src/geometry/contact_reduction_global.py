# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Global GPU contact reduction using hashtable-based tracking.

This module provides a global contact reduction system that uses a hashtable
to track the best contacts across shape pairs, normal bins, and scan directions.
Unlike the shared-memory based approach in contact_reduction.py, this works
across the entire GPU without block-level synchronization constraints.

Key Design:
- Contacts are stored in a global buffer (struct of arrays, packed into vec4)
- A hashtable tracks the best contact per (shape_pair, normal_bin, scan_direction)
- Each contact is registered 6 times (once per scan direction)
- Atomic max selects the best contact based on spatial projection score
"""

from __future__ import annotations

from typing import Any

import warp as wp

from newton._src.geometry.hashtable import (
    HASHTABLE_EMPTY_KEY,
    HashTable,
    hashtable_find_or_insert,
)

from .collision_core import (
    build_pair_key3,
    create_compute_gjk_mpr_contacts,
    get_triangle_shape_from_mesh,
)
from .contact_data import ContactData
from .contact_reduction import (
    NUM_SPATIAL_DIRECTIONS,
    float_flip,
    get_slot,
    get_spatial_direction_2d,
    project_point_to_plane,
)
from .support_function import extract_shape_data

# =============================================================================
# Reduction slot functions (specific to contact reduction)
# =============================================================================
# These functions handle the slot-major value storage used for contact reduction.
# Memory layout is slot-major (SoA) for coalesced GPU access:
# [slot0_entry0, slot0_entry1, ..., slot0_entryN, slot1_entry0, ...]


@wp.func
def reduction_update_slot(
    entry_idx: int,
    slot_id: int,
    value: wp.uint64,
    values: wp.array(dtype=wp.uint64),
    capacity: int,
):
    """Update a reduction slot using atomic max.

    Use this after hashtable_find_or_insert() to write multiple values
    to the same entry without repeated hash lookups.

    Args:
        entry_idx: Entry index from hashtable_find_or_insert()
        slot_id: Which value slot to write to (0 to values_per_key-1)
        value: The uint64 value to max with existing value
        values: Values array in slot-major layout
        capacity: Hashtable capacity (number of entries)
    """
    value_idx = slot_id * capacity + entry_idx
    # Check before atomic to reduce contention
    if values[value_idx] < value:
        wp.atomic_max(values, value_idx, value)


@wp.func
def reduction_insert_slot(
    key: wp.uint64,
    slot_id: int,
    value: wp.uint64,
    keys: wp.array(dtype=wp.uint64),
    values: wp.array(dtype=wp.uint64),
    active_slots: wp.array(dtype=wp.int32),
) -> bool:
    """Insert or update a value in a specific reduction slot.

    Convenience function that combines hashtable_find_or_insert()
    and reduction_update_slot(). For inserting multiple values to
    the same key, prefer using those functions separately.

    Args:
        key: The uint64 key to insert
        slot_id: Which value slot to write to (0 to values_per_key-1)
        value: The uint64 value to insert or max with
        keys: The hash table keys array (length must be power of two)
        values: Values array in slot-major layout
        active_slots: Array of size (capacity + 1) tracking active entry indices.

    Returns:
        True if insertion/update succeeded, False if the table is full
    """
    capacity = keys.shape[0]
    entry_idx = hashtable_find_or_insert(key, keys, active_slots)
    if entry_idx < 0:
        return False
    reduction_update_slot(entry_idx, slot_id, value, values, capacity)
    return True


# =============================================================================
# Contact key/value packing
# =============================================================================

# Bit layout for hashtable key (64 bits total):
# Key is (shape_a, shape_b, bin_id) - NO slot_id (slots are handled via values_per_key)
# - Bits 0-28:   shape_a (29 bits, up to ~537M shapes)
# - Bits 29-57:  shape_b (29 bits, up to ~537M shapes)
# - Bits 58-62:  icosahedron_bin (5 bits, 0-19)
# - Bit 63:      unused (could be used for flags)
# Total: 63 bits used

SHAPE_ID_BITS = wp.constant(wp.uint64(29))
SHAPE_ID_MASK = wp.constant(wp.uint64((1 << 29) - 1))
BIN_BITS = wp.constant(wp.uint64(5))
BIN_MASK = wp.constant(wp.uint64((1 << 5) - 1))


@wp.func
def make_contact_key(shape_a: int, shape_b: int, bin_id: int) -> wp.uint64:
    """Create a hashtable key from shape pair and normal bin.

    Args:
        shape_a: First shape index
        shape_b: Second shape index
        bin_id: Icosahedron bin index (0-19)

    Returns:
        64-bit key for hashtable lookup
    """
    key = wp.uint64(shape_a) & SHAPE_ID_MASK
    key = key | ((wp.uint64(shape_b) & SHAPE_ID_MASK) << SHAPE_ID_BITS)
    key = key | ((wp.uint64(bin_id) & BIN_MASK) << wp.uint64(58))
    return key


@wp.func
def make_contact_value(score: float, contact_id: int) -> wp.uint64:
    """Pack score and contact_id into hashtable value for atomic max.

    High 32 bits: float_flip(score) - makes floats comparable as unsigned ints
    Low 32 bits: contact_id - identifies which contact in the buffer

    Args:
        score: Spatial projection score (higher is better)
        contact_id: Index into the contact buffer

    Returns:
        64-bit value for hashtable (atomic max will select highest score)
    """
    return (wp.uint64(float_flip(score)) << wp.uint64(32)) | wp.uint64(contact_id)


@wp.func_native("""
return static_cast<int32_t>(packed & 0xFFFFFFFFull);
""")
def unpack_contact_id(packed: wp.uint64) -> int:
    """Extract contact_id from packed value."""
    ...


@wp.func_native("""
return reinterpret_cast<float&>(i);
""")
def int_as_float(i: wp.int32) -> float:
    """Reinterpret int32 bits as float32."""
    ...


@wp.func_native("""
return reinterpret_cast<int&>(f);
""")
def float_as_int(f: float) -> wp.int32:
    """Reinterpret float32 bits as int32."""
    ...


@wp.struct
class GlobalContactReducerData:
    """Struct for passing GlobalContactReducer arrays to kernels.

    This struct bundles all the arrays needed for global contact reduction
    so they can be passed as a single argument to warp kernels/functions.
    """

    # Contact buffer arrays
    position_depth: wp.array(dtype=wp.vec4)
    normal_feature: wp.array(dtype=wp.vec4)
    shape_pairs: wp.array(dtype=wp.vec2i)
    contact_count: wp.array(dtype=wp.int32)
    capacity: int

    # Hashtable arrays
    ht_keys: wp.array(dtype=wp.uint64)
    ht_values: wp.array(dtype=wp.uint64)
    ht_active_slots: wp.array(dtype=wp.int32)
    ht_capacity: int
    ht_values_per_key: int


@wp.kernel
def _clear_active_kernel(
    # Hashtable arrays
    ht_keys: wp.array(dtype=wp.uint64),
    ht_values: wp.array(dtype=wp.uint64),
    ht_active_slots: wp.array(dtype=wp.int32),
    ht_capacity: int,
    values_per_key: int,
    num_threads: int,
):
    """Kernel to clear active hashtable entries (keys and values).

    Uses grid-stride loop for efficient thread utilization.
    Each thread handles one value slot, with key clearing done once per entry.

    Memory layout for values is slot-major (SoA):
    [slot0_entry0, slot0_entry1, ..., slot0_entryN, slot1_entry0, ...]
    """
    tid = wp.tid()

    # Read count from GPU - stored at active_slots[capacity]
    count = ht_active_slots[ht_capacity]

    # Total work items: count entries * values_per_key slots per entry
    total_work = count * values_per_key

    # Grid-stride loop: each thread processes one value slot
    i = tid
    while i < total_work:
        # Compute which entry and which slot within that entry
        active_idx = i / values_per_key
        local_idx = i % values_per_key
        entry_idx = ht_active_slots[active_idx]

        # Clear the key only once per entry (when processing slot 0)
        if local_idx == 0:
            ht_keys[entry_idx] = HASHTABLE_EMPTY_KEY

        # Clear this value slot (slot-major layout)
        value_idx = local_idx * ht_capacity + entry_idx
        ht_values[value_idx] = wp.uint64(0)
        i += num_threads


@wp.kernel
def _zero_count_and_contacts_kernel(
    ht_active_slots: wp.array(dtype=wp.int32),
    contact_count: wp.array(dtype=wp.int32),
    ht_capacity: int,
):
    """Zero the active slots count and contact count."""
    ht_active_slots[ht_capacity] = 0
    contact_count[0] = 0


class GlobalContactReducer:
    """Global contact reduction using hashtable-based tracking.

    This class manages:
    1. A global contact buffer storing contact data (struct of arrays)
    2. A hashtable tracking the best contact per (shape_pair, bin, slot)

    The hashtable key is (shape_a, shape_b, bin_id). Each key has multiple
    values (one per slot = direction x beta + deepest). This allows one thread
    to process all slots for a bin and deduplicate locally.

    Contact data is packed into vec4 for efficient memory access:
    - position_depth: vec4(position.x, position.y, position.z, depth)
    - normal_feature: vec4(normal.x, normal.y, normal.z, float_bits(feature))

    Attributes:
        capacity: Maximum number of contacts that can be stored
        values_per_key: Number of value slots per hashtable entry (13 for 2 betas)
        position_depth: vec4 array storing position.xyz and depth
        normal_feature: vec4 array storing normal.xyz and feature
        shape_pairs: vec2i array storing (shape_a, shape_b) per contact
        contact_count: Atomic counter for allocated contacts
        hashtable: HashTable for tracking best contacts (keys only)
        ht_values: Values array for hashtable (managed here, not by HashTable)
    """

    def __init__(
        self,
        capacity: int,
        device: str | None = None,
        num_betas: int = 2,
    ):
        """Initialize the global contact reducer.

        Args:
            capacity: Maximum number of contacts to store
            device: Warp device (e.g., "cuda:0", "cpu")
            num_betas: Number of depth thresholds for contact reduction.
                       Total slots per bin = 6 directions * num_betas + 1 deepest.
                       Default 2 gives 13 slots per bin.
        """
        self.capacity = capacity
        self.device = device
        self.num_betas = num_betas

        # Values per key: 6 directions x num_betas + 1 deepest
        self.values_per_key = NUM_SPATIAL_DIRECTIONS * num_betas + 1

        # Contact buffer (struct of arrays with vec4 packing)
        self.position_depth = wp.zeros(capacity, dtype=wp.vec4, device=device)
        self.normal_feature = wp.zeros(capacity, dtype=wp.vec4, device=device)
        self.shape_pairs = wp.zeros(capacity, dtype=wp.vec2i, device=device)

        # Atomic counter for contact allocation
        self.contact_count = wp.zeros(1, dtype=wp.int32, device=device)

        # Hashtable: sized for worst case where each contact is a unique (shape_pair, bin)
        # Each contact goes into exactly ONE normal bin, so max unique keys = num_contacts
        # Use 2x for load factor to reduce hash collisions
        hashtable_size = capacity * 2
        self.hashtable = HashTable(hashtable_size, device=device)

        # Values array for hashtable - managed here, not by HashTable
        # This is contact-reduction-specific (slot-major layout with values_per_key slots)
        self.ht_values = wp.zeros(self.hashtable.capacity * self.values_per_key, dtype=wp.uint64, device=device)

    def clear(self):
        """Clear all contacts and reset the reducer (full clear)."""
        self.contact_count.zero_()
        self.hashtable.clear()
        self.ht_values.zero_()

    def clear_active(self):
        """Clear only the active entries (efficient for sparse usage).

        Uses a combined kernel that clears both hashtable keys and values,
        followed by a small kernel to zero the counters.
        """
        # Use fixed thread count for efficient GPU utilization
        num_threads = min(1024, self.hashtable.capacity)

        # Single kernel clears both keys and values for active entries
        wp.launch(
            _clear_active_kernel,
            dim=num_threads,
            inputs=[
                self.hashtable.keys,
                self.ht_values,
                self.hashtable.active_slots,
                self.hashtable.capacity,
                self.values_per_key,
                num_threads,
            ],
            device=self.device,
        )

        # Zero the counts in a separate kernel
        wp.launch(
            _zero_count_and_contacts_kernel,
            dim=1,
            inputs=[
                self.hashtable.active_slots,
                self.contact_count,
                self.hashtable.capacity,
            ],
            device=self.device,
        )

    def get_data_struct(self) -> GlobalContactReducerData:
        """Get a GlobalContactReducerData struct for passing to kernels.

        Returns:
            A GlobalContactReducerData struct containing all arrays.
        """
        data = GlobalContactReducerData()
        data.position_depth = self.position_depth
        data.normal_feature = self.normal_feature
        data.shape_pairs = self.shape_pairs
        data.contact_count = self.contact_count
        data.capacity = self.capacity
        data.ht_keys = self.hashtable.keys
        data.ht_values = self.ht_values
        data.ht_active_slots = self.hashtable.active_slots
        data.ht_capacity = self.hashtable.capacity
        data.ht_values_per_key = self.values_per_key
        return data


@wp.func
def export_contact_to_buffer(
    shape_a: int,
    shape_b: int,
    position: wp.vec3,
    normal: wp.vec3,
    depth: float,
    feature: int,
    reducer_data: GlobalContactReducerData,
) -> int:
    """Store a contact in the buffer without reduction.

    Args:
        shape_a: First shape index
        shape_b: Second shape index
        position: Contact position in world space
        normal: Contact normal
        depth: Penetration depth (negative = penetrating)
        feature: Feature identifier for deduplication
        reducer_data: GlobalContactReducerData with all arrays

    Returns:
        Contact ID if successfully stored, -1 if buffer full
    """
    # Allocate contact slot
    contact_id = wp.atomic_add(reducer_data.contact_count, 0, 1)
    if contact_id >= reducer_data.capacity:
        return -1

    # Store contact data (packed into vec4)
    reducer_data.position_depth[contact_id] = wp.vec4(position[0], position[1], position[2], depth)
    reducer_data.normal_feature[contact_id] = wp.vec4(normal[0], normal[1], normal[2], int_as_float(wp.int32(feature)))
    reducer_data.shape_pairs[contact_id] = wp.vec2i(shape_a, shape_b)

    return contact_id


@wp.func
def reduce_contact_in_hashtable(
    contact_id: int,
    reducer_data: GlobalContactReducerData,
    beta0: float,
    beta1: float,
):
    """Register a buffered contact in the reduction hashtable.

    Args:
        contact_id: Index of contact in buffer
        reducer_data: Reducer data
        beta0: First depth threshold
        beta1: Second depth threshold
    """
    # Read contact data from buffer
    pd = reducer_data.position_depth[contact_id]
    nf = reducer_data.normal_feature[contact_id]
    pair = reducer_data.shape_pairs[contact_id]

    position = wp.vec3(pd[0], pd[1], pd[2])
    depth = pd[3]
    normal = wp.vec3(nf[0], nf[1], nf[2])
    shape_a = pair[0]
    shape_b = pair[1]

    # Get icosahedron bin from normal
    bin_id = get_slot(normal)

    # Project position to 2D plane of the icosahedron face
    pos_2d = project_point_to_plane(bin_id, position)

    # Key is (shape_a, shape_b, bin_id) - NO slot in key
    key = make_contact_key(shape_a, shape_b, bin_id)

    # Find or create the hashtable entry ONCE, then write directly to slots
    entry_idx = hashtable_find_or_insert(key, reducer_data.ht_keys, reducer_data.ht_active_slots)
    if entry_idx < 0:
        return

    ht_capacity = reducer_data.ht_capacity

    use_beta0 = depth < beta0
    use_beta1 = depth < beta1

    if beta0 < beta1 and use_beta0:
        use_beta1 = False

    if beta0 > beta1 and use_beta1:
        use_beta0 = False

    # Register in hashtable for all 6 spatial directions x 2 betas
    # Using direct slot access (no repeated hash lookups)
    # Memory layout is slot-major for coalesced access
    for dir_i in range(NUM_SPATIAL_DIRECTIONS):
        dir_2d = get_spatial_direction_2d(dir_i)
        score = wp.dot(pos_2d, dir_2d)
        value = make_contact_value(score, contact_id)

        # Beta 0 slot (even indices: 0, 2, 4, 6, 8, 10)
        if use_beta0:
            slot_id = dir_i * 2
            reduction_update_slot(entry_idx, slot_id, value, reducer_data.ht_values, ht_capacity)

        # Beta 1 slot (odd indices: 1, 3, 5, 7, 9, 11)
        if use_beta1:
            slot_id = dir_i * 2 + 1
            reduction_update_slot(entry_idx, slot_id, value, reducer_data.ht_values, ht_capacity)

    # Also register for max-depth slot (last slot = 12)
    # Use -depth as score so atomic_max selects the deepest (most negative depth)
    max_depth_slot_id = NUM_SPATIAL_DIRECTIONS * 2  # = 12
    max_depth_value = make_contact_value(-depth, contact_id)
    reduction_update_slot(entry_idx, max_depth_slot_id, max_depth_value, reducer_data.ht_values, ht_capacity)


@wp.func
def export_and_reduce_contact(
    shape_a: int,
    shape_b: int,
    position: wp.vec3,
    normal: wp.vec3,
    depth: float,
    feature: int,
    reducer_data: GlobalContactReducerData,
    beta0: float,
    beta1: float,
) -> int:
    """Legacy wrapper for backward compatibility."""
    contact_id = export_contact_to_buffer(shape_a, shape_b, position, normal, depth, feature, reducer_data)

    if contact_id >= 0:
        reduce_contact_in_hashtable(contact_id, reducer_data, beta0, beta1)

    return contact_id


def create_reduce_buffered_contacts_kernel(beta0: float, beta1: float):
    """Create a kernel that registers buffered contacts to the hashtable.

    This splits the contact reduction process into two steps:
    1. Writing contacts to buffer (done by contact generation kernel)
    2. Registering contacts to hashtable (done by this kernel)

    This reduces register pressure on the contact generation kernel.
    """

    @wp.kernel(enable_backward=False)
    def reduce_buffered_contacts_kernel(
        reducer_data: GlobalContactReducerData,
        total_num_threads: int,
    ):
        """Iterate over buffered contacts and register them in the hashtable."""
        tid = wp.tid()

        # Get total number of contacts written
        num_contacts = reducer_data.contact_count[0]

        # Cap at capacity
        num_contacts = wp.min(num_contacts, reducer_data.capacity)

        # Grid stride loop over contacts
        for i in range(tid, num_contacts, total_num_threads):
            reduce_contact_in_hashtable(i, reducer_data, beta0, beta1)

    return reduce_buffered_contacts_kernel


@wp.func
def unpack_contact(
    contact_id: int,
    position_depth: wp.array(dtype=wp.vec4),
    normal_feature: wp.array(dtype=wp.vec4),
):
    """Unpack contact data from the buffer.

    Args:
        contact_id: Index into the contact buffer
        position_depth: Contact buffer for position.xyz + depth
        normal_feature: Contact buffer for normal.xyz + feature

    Returns:
        Tuple of (position, normal, depth, feature)
    """
    pd = position_depth[contact_id]
    nf = normal_feature[contact_id]

    position = wp.vec3(pd[0], pd[1], pd[2])
    depth = pd[3]
    normal = wp.vec3(nf[0], nf[1], nf[2])
    feature = float_as_int(nf[3])

    return position, normal, depth, feature


@wp.func
def write_contact_to_reducer(
    contact_data: Any,  # ContactData struct
    reducer_data: GlobalContactReducerData,
    output_index: int,  # Unused, kept for API compatibility with write_contact_simple
    beta0: float,
    beta1: float,
):
    """Writer function that stores contacts in GlobalContactReducer for reduction.

    This follows the same signature as write_contact_simple in narrow_phase.py,
    so it can be used with create_compute_gjk_mpr_contacts and other contact
    generation functions.

    Args:
        contact_data: ContactData struct from contact computation
        reducer_data: GlobalContactReducerData struct with all reducer arrays
        beta0: First depth threshold (typically large, e.g., 1000000.0)
        beta1: Second depth threshold (typically small, e.g., 0.0001)
    """
    # Extract contact info from ContactData
    position = contact_data.contact_point_center
    normal = contact_data.contact_normal_a_to_b
    depth = contact_data.contact_distance
    shape_a = contact_data.shape_a
    shape_b = contact_data.shape_b
    feature = int(contact_data.feature)

    # Store contact ONLY (registration to hashtable happens in a separate kernel)
    # This reduces register pressure on the contact generation kernel
    export_contact_to_buffer(
        shape_a=shape_a,
        shape_b=shape_b,
        position=position,
        normal=normal,
        depth=depth,
        feature=feature,
        reducer_data=reducer_data,
    )


def create_export_reduced_contacts_kernel(writer_func: Any, values_per_key: int = 13):
    """Create a kernel that exports reduced contacts using a custom writer function.

    The kernel processes one hashtable ENTRY per thread (not one value slot).
    Each entry has values_per_key value slots. The thread reads all slots,
    collects unique contact IDs, and exports each unique contact once.

    This naturally deduplicates: one thread handles one (shape_pair, bin) entry
    and can locally track which contact IDs it has already exported.

    Args:
        writer_func: A warp function with signature (ContactData, writer_data) -> None
                     This follows the same pattern as narrow_phase.py's write_contact_simple.
        values_per_key: Number of value slots per hashtable entry (default 13 for 2 betas)

    Returns:
        A warp kernel that can be launched to export reduced contacts.
    """
    # Define vector type for tracking exported contact IDs
    exported_ids_vec = wp.types.vector(length=values_per_key, dtype=wp.int32)

    @wp.kernel(enable_backward=False)
    def export_reduced_contacts_kernel(
        # Hashtable arrays
        ht_keys: wp.array(dtype=wp.uint64),
        ht_values: wp.array(dtype=wp.uint64),
        ht_active_slots: wp.array(dtype=wp.int32),
        # Contact buffer arrays
        position_depth: wp.array(dtype=wp.vec4),
        normal_feature: wp.array(dtype=wp.vec4),
        shape_pairs: wp.array(dtype=wp.vec2i),
        # Shape data for extracting thickness
        shape_data: wp.array(dtype=wp.vec4),
        # Per-shape contact margins
        shape_contact_margin: wp.array(dtype=float),
        # Writer data (custom struct)
        writer_data: Any,
        # Grid stride parameters
        total_num_threads: int,
    ):
        """Export reduced contacts to the writer.

        Uses grid stride loop to iterate over active hashtable ENTRIES.
        For each entry, reads all value slots, collects unique contact IDs,
        and exports each unique contact once.
        """
        tid = wp.tid()

        # Get number of active entries (stored at index = ht_capacity)
        ht_capacity = ht_keys.shape[0]
        num_active = ht_active_slots[ht_capacity]

        # Grid stride loop over active entries
        for i in range(tid, num_active, total_num_threads):
            # Get the hashtable entry index
            entry_idx = ht_active_slots[i]

            # Track exported contact IDs for this entry
            exported_ids = exported_ids_vec()
            num_exported = int(0)

            # Read all value slots for this entry (slot-major layout)
            for slot in range(values_per_key):
                value = ht_values[slot * ht_capacity + entry_idx]

                # Skip empty slots (value = 0)
                if value == wp.uint64(0):
                    continue

                # Extract contact ID from low 32 bits
                contact_id = unpack_contact_id(value)

                # Skip if already exported
                already_exported = False
                for j in range(values_per_key):
                    if j < num_exported and exported_ids[j] == contact_id:
                        already_exported = True
                if already_exported:
                    continue

                # Record this contact ID as exported
                exported_ids[num_exported] = contact_id
                num_exported = num_exported + 1

                # Unpack contact data
                position, normal, depth, feature = unpack_contact(contact_id, position_depth, normal_feature)

                # Get shape pair
                pair = shape_pairs[contact_id]
                shape_a = pair[0]
                shape_b = pair[1]

                # Extract thickness from shape_data (stored in w component)
                thickness_a = shape_data[shape_a][3]
                thickness_b = shape_data[shape_b][3]

                # Use per-shape contact margin (max of both shapes, matching other kernels)
                margin_a = shape_contact_margin[shape_a]
                margin_b = shape_contact_margin[shape_b]
                margin = wp.max(margin_a, margin_b)

                # Create ContactData struct
                contact_data = ContactData()
                contact_data.contact_point_center = position
                contact_data.contact_normal_a_to_b = normal
                contact_data.contact_distance = depth
                contact_data.radius_eff_a = 0.0
                contact_data.radius_eff_b = 0.0
                contact_data.thickness_a = thickness_a
                contact_data.thickness_b = thickness_b
                contact_data.shape_a = shape_a
                contact_data.shape_b = shape_b
                contact_data.margin = margin
                contact_data.feature = wp.uint32(feature)
                contact_data.feature_pair_key = wp.uint64(0)

                # Call the writer function
                writer_func(contact_data, writer_data, -1)

    return export_reduced_contacts_kernel


def create_mesh_triangle_contacts_to_reducer_kernel(beta0: float, beta1: float):
    """Create a kernel that processes mesh-triangle contacts and stores them in GlobalContactReducer.

    This kernel processes triangle pairs (mesh-shape, convex-shape, triangle_index) and
    computes contacts using GJK/MPR, storing results in the GlobalContactReducer for
    subsequent reduction and export.

    Args:
        beta0: First depth threshold for contact reduction
        beta1: Second depth threshold for contact reduction

    Returns:
        A warp kernel for processing mesh-triangle contacts with global reduction.
    """

    # Create a writer function that captures beta0 and beta1
    @wp.func
    def write_to_reducer_with_betas(
        contact_data: Any,  # ContactData struct
        reducer_data: GlobalContactReducerData,
        output_index: int,  # Unused, kept for API compatibility with write_contact_simple
    ):
        write_contact_to_reducer(contact_data, reducer_data, output_index, beta0, beta1)

    @wp.kernel(enable_backward=False)
    def mesh_triangle_contacts_to_reducer_kernel(
        shape_types: wp.array(dtype=int),
        shape_data: wp.array(dtype=wp.vec4),
        shape_transform: wp.array(dtype=wp.transform),
        shape_source: wp.array(dtype=wp.uint64),
        shape_contact_margin: wp.array(dtype=float),
        triangle_pairs: wp.array(dtype=wp.vec3i),
        triangle_pairs_count: wp.array(dtype=int),
        reducer_data: GlobalContactReducerData,
        total_num_threads: int,
    ):
        """Process triangle pairs and store contacts in GlobalContactReducer.

        Uses grid stride loop over triangle pairs.
        """
        tid = wp.tid()

        num_triangle_pairs = triangle_pairs_count[0]

        for i in range(tid, num_triangle_pairs, total_num_threads):
            if i >= triangle_pairs.shape[0]:
                break

            triple = triangle_pairs[i]
            shape_a = triple[0]  # Mesh shape
            shape_b = triple[1]  # Convex shape
            tri_idx = triple[2]

            # Get mesh data for shape A
            mesh_id_a = shape_source[shape_a]
            if mesh_id_a == wp.uint64(0):
                continue

            scale_data_a = shape_data[shape_a]
            mesh_scale_a = wp.vec3(scale_data_a[0], scale_data_a[1], scale_data_a[2])

            # Get mesh world transform
            X_mesh_ws_a = shape_transform[shape_a]

            # Extract triangle shape data from mesh
            shape_data_a, v0_world = get_triangle_shape_from_mesh(mesh_id_a, mesh_scale_a, X_mesh_ws_a, tri_idx)

            # Extract shape B data
            pos_b, quat_b, shape_data_b, _scale_b, thickness_b = extract_shape_data(
                shape_b,
                shape_transform,
                shape_types,
                shape_data,
                shape_source,
            )

            # Set pos_a to be vertex A (origin of triangle in local frame)
            pos_a = v0_world
            quat_a = wp.quat_identity()  # Triangle has no orientation

            # Extract thickness for shape A
            thickness_a = shape_data[shape_a][3]

            # Use per-shape contact margin
            margin_a = shape_contact_margin[shape_a]
            margin_b = shape_contact_margin[shape_b]
            margin = wp.max(margin_a, margin_b)

            # Build pair key including triangle index
            pair_key = build_pair_key3(wp.uint32(shape_a), wp.uint32(shape_b), wp.uint32(tri_idx))

            # Compute and write contacts using GJK/MPR
            wp.static(create_compute_gjk_mpr_contacts(write_to_reducer_with_betas))(
                shape_data_a,
                shape_data_b,
                quat_a,
                quat_b,
                pos_a,
                pos_b,
                margin,
                shape_a,
                shape_b,
                thickness_a,
                thickness_b,
                reducer_data,
                pair_key,
            )

    return mesh_triangle_contacts_to_reducer_kernel
