# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Global GPU contact reduction using hashtable-based tracking.

This module provides a global contact reduction system that uses a hashtable
to track the best contacts across shape pairs, normal bins, and scan directions.
Unlike the shared-memory based approach in ``contact_reduction.py``, this works
across the entire GPU without block-level synchronization constraints.

**When to Use:**

- Used for mesh-mesh (SDF) collisions where contacts span multiple GPU blocks
- The shared-memory approach in ``contact_reduction.py`` is used for mesh-plane
  and mesh-convex where all contacts for a pair fit in one block

**Contact Reduction Strategy:**

The same three-strategy approach as the shared-memory reduction in
``contact_reduction.py``.  Slot counts depend on the configuration in that
module (``NUM_NORMAL_BINS``, ``NUM_SPATIAL_DIRECTIONS``,
``NUM_VOXEL_DEPTH_SLOTS``, ``MAX_CONTACTS_PER_PAIR``).

1. **Spatial Extreme Slots** (``NUM_SPATIAL_DIRECTIONS`` per normal bin)
   - Builds support polygon boundary for stable stacking
   - Only contacts with depth < beta participate

2. **Per-Bin Max-Depth Slots** (1 per normal bin)
   - Tracks deepest contact per normal direction
   - Critical for gear-like contacts with varied normal orientations
   - Participates unconditionally (not gated by beta)

3. **Voxel-Based Depth Slots** (``NUM_VOXEL_DEPTH_SLOTS`` total per pair)
   - Tracks deepest contact per mesh-local voxel region
   - Ensures early detection of contacts at mesh centers
   - Prevents sudden contact jumps between frames

**Implementation Details:**

- Contacts stored in global buffer (struct of arrays: position_depth, normal, shape_pairs)
- Hashtable key: ``(shape_a, shape_b, bin_id)`` where ``bin_id`` is
  ``0..NUM_NORMAL_BINS-1`` for normal bins and higher indices for voxel groups
- Each normal bin entry has ``NUM_SPATIAL_DIRECTIONS + 1`` value slots
- Voxels are grouped by ``NUM_SPATIAL_DIRECTIONS + 1``:
  ``bin_id = NUM_NORMAL_BINS + (voxel_idx // group_size)``
- Atomic max on packed (score, contact_id) selects winners

See Also:
    ``contact_reduction.py`` for shared utility functions, configuration
    constants, and detailed algorithm documentation.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from newton._src.geometry.hashtable import (
    HASHTABLE_EMPTY_KEY,
    HashTable,
    hashtable_find_or_insert,
)

from ..utils.heightfield import HeightfieldData, get_triangle_shape_from_heightfield
from .collision_core import (
    create_compute_gjk_mpr_contacts,
    get_triangle_shape_from_mesh,
)
from .contact_data import ContactData
from .contact_reduction import (
    NUM_NORMAL_BINS,
    NUM_SPATIAL_DIRECTIONS,
    NUM_VOXEL_DEPTH_SLOTS,
    compute_voxel_index,
    float_flip,
    get_slot,
    get_spatial_direction_2d,
    project_point_to_plane,
)
from .support_function import GeoTypeEx, extract_shape_data
from .types import GeoType

# Fixed beta threshold for contact reduction - small positive value to avoid flickering
# from numerical noise while effectively selecting only near-penetrating contacts for
# the support polygon.
BETA_THRESHOLD = 0.0001  # 0.1mm

VALUES_PER_KEY = NUM_SPATIAL_DIRECTIONS + 1

# Vector type for tracking exported contact IDs (used in export kernels)
exported_ids_vec_type = wp.types.vector(length=VALUES_PER_KEY, dtype=wp.int32)


@wp.func
def is_contact_already_exported(
    contact_id: int,
    exported_ids: wp.types.vector(length=VALUES_PER_KEY, dtype=wp.int32),
    num_exported: int,
) -> bool:
    """Check if a contact_id is already in the exported list.

    Args:
        contact_id: The contact ID to check
        exported_ids: Vector of already exported contact IDs
        num_exported: Number of valid entries in exported_ids

    Returns:
        True if contact_id is already in the list, False otherwise
    """
    j = int(0)
    while j < num_exported:
        if exported_ids[j] == contact_id:
            return True
        j = j + 1
    return False


@wp.func
def compute_effective_radius(shape_type: int, shape_scale: wp.vec4) -> float:
    """Compute effective radius for a shape based on its type.

    For shapes that can be represented as Minkowski sums with a sphere (sphere, capsule),
    the effective radius is the sphere radius component. For other shapes, it's 0.

    Args:
        shape_type: The GeoType of the shape
        shape_scale: Shape scale data (vec4, xyz are scale components)

    Returns:
        Effective radius (scale[0] for sphere/capsule, 0 otherwise)
    """
    if shape_type == GeoType.SPHERE or shape_type == GeoType.CAPSULE:
        return shape_scale[0]
    return 0.0


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
    values: wp.array[wp.uint64],
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
    keys: wp.array[wp.uint64],
    values: wp.array[wp.uint64],
    active_slots: wp.array[wp.int32],
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

# Bit layout for hashtable key (63 bits used, bit 63 kept 0 for signed/unsigned safety):
# Key is (shape_a, shape_b, bin_id) - NO slot_id (slots are handled via values_per_key)
# - Bits 0-26:   shape_a (27 bits, up to ~134M shapes)
# - Bits 27-54:  shape_b (28 bits, up to ~268M shapes)
# - Bits 55-62:  bin_id (8 bits, 0-255, supports normal bins + voxel groups)
# - Bit 63:      unused (kept 0 for signed/unsigned compatibility)
# Total: 63 bits used

SHAPE_A_BITS = wp.constant(wp.uint64(27))
SHAPE_A_MASK = wp.constant(wp.uint64((1 << 27) - 1))
SHAPE_B_BITS = wp.constant(wp.uint64(28))
SHAPE_B_MASK = wp.constant(wp.uint64((1 << 28) - 1))
BIN_BITS = wp.constant(wp.uint64(8))
BIN_MASK = wp.constant(wp.uint64((1 << 8) - 1))


@wp.func
def make_contact_key(shape_a: int, shape_b: int, bin_id: int) -> wp.uint64:
    """Create a hashtable key from shape pair and bin.

    Args:
        shape_a: First shape index
        shape_b: Second shape index
        bin_id: Bin index (``0..NUM_NORMAL_BINS-1`` for normal bins, higher for voxel groups)

    Returns:
        64-bit key for hashtable lookup (only 63 bits used)
    """
    key = wp.uint64(shape_a) & SHAPE_A_MASK
    key = key | ((wp.uint64(shape_b) & SHAPE_B_MASK) << SHAPE_A_BITS)
    # bin_id goes at bits 55-62 (after 27 + 28 = 55 bits for shape IDs)
    key = key | ((wp.uint64(bin_id) & BIN_MASK) << wp.uint64(55))
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


@wp.func
def encode_oct(n: wp.vec3) -> wp.vec2:
    """Encode a unit normal into octahedral 2D representation.

    Projects the unit vector onto an octahedron and flattens to 2D.
    Near-uniform precision, stable numerics, no trig needed.
    """
    l1 = wp.abs(n[0]) + wp.abs(n[1]) + wp.abs(n[2])
    if l1 < 1.0e-20:
        return wp.vec2(0.0, 0.0)
    inv_l1 = 1.0 / l1
    ox = n[0] * inv_l1
    oy = n[1] * inv_l1
    oz = n[2] * inv_l1

    if oz < 0.0:
        sign_x = 1.0
        if ox < 0.0:
            sign_x = -1.0
        sign_y = 1.0
        if oy < 0.0:
            sign_y = -1.0
        new_x = (1.0 - wp.abs(oy)) * sign_x
        new_y = (1.0 - wp.abs(ox)) * sign_y
        ox = new_x
        oy = new_y

    return wp.vec2(ox, oy)


@wp.func
def decode_oct(e: wp.vec2) -> wp.vec3:
    """Decode octahedral 2D representation back to a unit normal.

    Inverse of encode_oct.  Lossless within float precision.
    """
    nz = 1.0 - wp.abs(e[0]) - wp.abs(e[1])
    nx = e[0]
    ny = e[1]

    if nz < 0.0:
        sign_x = 1.0
        if nx < 0.0:
            sign_x = -1.0
        sign_y = 1.0
        if ny < 0.0:
            sign_y = -1.0
        new_x = (1.0 - wp.abs(ny)) * sign_x
        new_y = (1.0 - wp.abs(nx)) * sign_y
        nx = new_x
        ny = new_y

    return wp.normalize(wp.vec3(nx, ny, nz))


@wp.struct
class GlobalContactReducerData:
    """Struct for passing GlobalContactReducer arrays to kernels.

    This struct bundles all the arrays needed for global contact reduction
    so they can be passed as a single argument to warp kernels/functions.
    """

    # Contact buffer arrays
    position_depth: wp.array[wp.vec4]
    normal: wp.array[wp.vec2]  # Octahedral-encoded unit normal (see encode_oct/decode_oct)
    shape_pairs: wp.array[wp.vec2i]
    contact_count: wp.array[wp.int32]
    capacity: int

    # Optional hydroelastic data
    # contact_area: area of contact surface element (per contact)
    contact_area: wp.array[wp.float32]

    # Cached normal-bin hashtable entry index per contact
    contact_nbin_entry: wp.array[wp.int32]

    # Effective stiffness coefficient k_a*k_b/(k_a+k_b) per hashtable entry
    # Constant for a given shape pair, stored once per entry instead of per contact
    entry_k_eff: wp.array[wp.float32]

    # Aggregate force per hashtable entry (indexed by ht_capacity)
    # Used for hydroelastic stiffness calculation: c_stiffness = k_eff * |agg_force| / total_depth
    # Accumulates sum(area * depth * normal) for all penetrating contacts per entry
    agg_force: wp.array[wp.vec3]

    # Weighted position sum per hashtable entry (for anchor contact computation)
    # Accumulates sum(area * depth * position) for penetrating contacts
    # Divide by weight_sum to get center of pressure (anchor position)
    weighted_pos_sum: wp.array[wp.vec3]

    # Weight sum per hashtable entry (for anchor contact normalization)
    # Accumulates sum(area * depth) for penetrating contacts
    weight_sum: wp.array[wp.float32]

    # Total depth of reduced (winning) contacts per normal bin entry.
    total_depth_reduced: wp.array[wp.float32]

    # Total depth-weighted normal of reduced (winning) contacts per normal bin entry.
    total_normal_reduced: wp.array[wp.vec3]

    # Hashtable arrays
    ht_keys: wp.array[wp.uint64]
    ht_values: wp.array[wp.uint64]
    ht_active_slots: wp.array[wp.int32]
    ht_insert_failures: wp.array[wp.int32]
    ht_capacity: int
    ht_values_per_key: int


@wp.kernel(enable_backward=False)
def _clear_active_kernel(
    # Hashtable arrays
    ht_keys: wp.array[wp.uint64],
    ht_values: wp.array[wp.uint64],
    ht_active_slots: wp.array[wp.int32],
    # Hydroelastic per-entry arrays
    agg_force: wp.array[wp.vec3],
    weighted_pos_sum: wp.array[wp.vec3],
    weight_sum: wp.array[wp.float32],
    entry_k_eff: wp.array[wp.float32],
    total_depth_reduced: wp.array[wp.float32],
    total_normal_reduced: wp.array[wp.vec3],
    agg_moment_unreduced: wp.array[wp.float32],
    agg_moment_reduced: wp.array[wp.float32],
    agg_moment2_reduced: wp.array[wp.float32],
    ht_capacity: int,
    values_per_key: int,
    num_threads: int,
):
    """Kernel to clear active hashtable entries (keys, values, and hydroelastic aggregates).

    Uses grid-stride loop for efficient thread utilization.
    Each thread handles one value slot, with key and aggregate clearing done once per entry.

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

        # Clear keys and hydroelastic aggregates only once per entry (when processing slot 0)
        if local_idx == 0:
            ht_keys[entry_idx] = HASHTABLE_EMPTY_KEY
            # Clear hydroelastic aggregates if arrays are not empty
            if agg_force.shape[0] > 0:
                agg_force[entry_idx] = wp.vec3(0.0, 0.0, 0.0)
                weighted_pos_sum[entry_idx] = wp.vec3(0.0, 0.0, 0.0)
                weight_sum[entry_idx] = 0.0
                entry_k_eff[entry_idx] = 0.0
                total_depth_reduced[entry_idx] = 0.0
                total_normal_reduced[entry_idx] = wp.vec3(0.0, 0.0, 0.0)
                if agg_moment_unreduced.shape[0] > 0:
                    agg_moment_unreduced[entry_idx] = 0.0
                    agg_moment_reduced[entry_idx] = 0.0
                    agg_moment2_reduced[entry_idx] = 0.0

        # Clear this value slot (slot-major layout)
        value_idx = local_idx * ht_capacity + entry_idx
        ht_values[value_idx] = wp.uint64(0)
        i += num_threads


@wp.kernel(enable_backward=False)
def _zero_count_and_contacts_kernel(
    ht_active_slots: wp.array[wp.int32],
    contact_count: wp.array[wp.int32],
    ht_insert_failures: wp.array[wp.int32],
    ht_capacity: int,
):
    """Zero the active slots count and contact count."""
    ht_active_slots[ht_capacity] = 0
    contact_count[0] = 0
    ht_insert_failures[0] = 0


class GlobalContactReducer:
    """Global contact reduction using hashtable-based tracking.

    This class manages:

    1. A global contact buffer storing contact data (struct of arrays)
    2. A hashtable tracking the best contact per (shape_pair, bin, slot)

    Slot counts depend on the configuration in ``contact_reduction.py``.

    **Hashtable Structure:**

    - Key: ``(shape_a, shape_b, bin_id)`` packed into 64 bits
    - bin_id ``0..NUM_NORMAL_BINS-1``: Normal bins (polyhedron faces)
    - Higher bin_ids: Voxel groups

    **Slot Layout per Normal Bin Entry** (``NUM_SPATIAL_DIRECTIONS + 1`` slots):

    - Slots ``0..NUM_SPATIAL_DIRECTIONS-1``: Spatial direction extremes (depth < beta)
    - Last slot: Maximum depth contact for the bin (unconditional)

    **Slot Layout per Voxel Group Entry** (``NUM_SPATIAL_DIRECTIONS + 1`` slots):

    - Each slot tracks the deepest contact for one voxel in the group
    - ``bin_id = NUM_NORMAL_BINS + (voxel_idx // group_size)``

    **Contact Data Storage:**

    Packed for efficient memory access:

    - position_depth: vec4(position.x, position.y, position.z, depth)
    - normal: vec2(octahedral-encoded unit normal)
    - shape_pairs: vec2i(shape_a, shape_b)
    - contact_area: float (optional, per contact, for hydroelastic contacts)

    Attributes:
        capacity: Maximum number of contacts that can be stored
        values_per_key: Number of value slots per hashtable entry (``NUM_SPATIAL_DIRECTIONS + 1``)
        position_depth: vec4 array storing position.xyz and depth
        normal: vec2 array storing octahedral-encoded contact normal
        shape_pairs: vec2i array storing (shape_a, shape_b) per contact
        contact_area: float array storing contact area per contact (for hydroelastic)
        entry_k_eff: float array storing effective stiffness per hashtable entry (for hydroelastic)
        contact_count: Atomic counter for allocated contacts
        hashtable: HashTable for tracking best contacts (keys only)
        ht_values: Values array for hashtable (managed here, not by HashTable)
    """

    def __init__(
        self,
        capacity: int,
        device: str | None = None,
        store_hydroelastic_data: bool = False,
        store_moment_data: bool = False,
    ):
        """Initialize the global contact reducer.

        Args:
            capacity: Maximum number of contacts to store
            device: Warp device (e.g., "cuda:0", "cpu")
            store_hydroelastic_data: If True, allocate arrays for contact_area and entry_k_eff
            store_moment_data: If True, allocate moment accumulator arrays for friction
                moment matching. Only needed when ``moment_matching=True``.
        """
        self.capacity = capacity
        self.device = device
        self.store_hydroelastic_data = store_hydroelastic_data

        self.values_per_key = NUM_SPATIAL_DIRECTIONS + 1

        # Contact buffer (struct of arrays)
        self.position_depth = wp.zeros(capacity, dtype=wp.vec4, device=device)
        self.normal = wp.zeros(capacity, dtype=wp.vec2, device=device)  # Octahedral-encoded normals
        self.shape_pairs = wp.zeros(capacity, dtype=wp.vec2i, device=device)

        # Optional hydroelastic data arrays
        if store_hydroelastic_data:
            self.contact_area = wp.zeros(capacity, dtype=wp.float32, device=device)
            self.contact_nbin_entry = wp.zeros(capacity, dtype=wp.int32, device=device)
        else:
            self.contact_area = wp.zeros(0, dtype=wp.float32, device=device)
            self.contact_nbin_entry = wp.zeros(0, dtype=wp.int32, device=device)

        # Per-contact dedup flags for cross-entry deduplication during export
        self.exported_flags = wp.zeros(capacity, dtype=wp.int32, device=device)

        # Atomic counter for contact allocation
        self.contact_count = wp.zeros(1, dtype=wp.int32, device=device)
        # Count failed hashtable inserts (e.g., table full)
        self.ht_insert_failures = wp.zeros(1, dtype=wp.int32, device=device)

        # Hashtable sizing: estimate unique (shape_pair, bin) keys needed
        # - NUM_NORMAL_BINS + ceil(NUM_VOXEL_DEPTH_SLOTS / values_per_key) bins per pair
        # - Dense hydroelastic contacts: many contacts share the same bin
        # - Assume ~8 contacts per unique key on average (conservative for dense contacts)
        # - Provides 2x load factor headroom within the /4 estimate
        # - If table fills, contacts gracefully skip reduction (still in buffer)
        hashtable_size = max(capacity // 4, 1024)  # minimum 1024 for small scenes
        self.hashtable = HashTable(hashtable_size, device=device)

        # Values array for hashtable - managed here, not by HashTable
        # This is contact-reduction-specific (slot-major layout with values_per_key slots)
        self.ht_values = wp.zeros(self.hashtable.capacity * self.values_per_key, dtype=wp.uint64, device=device)

        # Aggregate force per hashtable entry (for hydroelastic stiffness calculation)
        # Accumulates sum(area * depth * normal) for all penetrating contacts per entry
        if store_hydroelastic_data:
            self.agg_force = wp.zeros(self.hashtable.capacity, dtype=wp.vec3, device=device)
            self.weighted_pos_sum = wp.zeros(self.hashtable.capacity, dtype=wp.vec3, device=device)
            self.weight_sum = wp.zeros(self.hashtable.capacity, dtype=wp.float32, device=device)
            # k_eff per entry (constant per shape pair, set once on first insert)
            self.entry_k_eff = wp.zeros(self.hashtable.capacity, dtype=wp.float32, device=device)
            # Total depth of reduced contacts per normal bin (accumulated from all winning contacts)
            self.total_depth_reduced = wp.zeros(self.hashtable.capacity, dtype=wp.float32, device=device)
            # Total depth-weighted normal of reduced contacts per normal bin
            self.total_normal_reduced = wp.zeros(self.hashtable.capacity, dtype=wp.vec3, device=device)
            # Moment accumulators for moment matching (friction scale adjustment)
            if store_moment_data:
                self.agg_moment_unreduced = wp.zeros(self.hashtable.capacity, dtype=wp.float32, device=device)
                self.agg_moment_reduced = wp.zeros(self.hashtable.capacity, dtype=wp.float32, device=device)
                self.agg_moment2_reduced = wp.zeros(self.hashtable.capacity, dtype=wp.float32, device=device)
            else:
                self.agg_moment_unreduced = wp.zeros(0, dtype=wp.float32, device=device)
                self.agg_moment_reduced = wp.zeros(0, dtype=wp.float32, device=device)
                self.agg_moment2_reduced = wp.zeros(0, dtype=wp.float32, device=device)
        else:
            self.agg_force = wp.zeros(0, dtype=wp.vec3, device=device)
            self.weighted_pos_sum = wp.zeros(0, dtype=wp.vec3, device=device)
            self.weight_sum = wp.zeros(0, dtype=wp.float32, device=device)
            self.entry_k_eff = wp.zeros(0, dtype=wp.float32, device=device)
            self.total_depth_reduced = wp.zeros(0, dtype=wp.float32, device=device)
            self.total_normal_reduced = wp.zeros(0, dtype=wp.vec3, device=device)
            self.agg_moment_unreduced = wp.zeros(0, dtype=wp.float32, device=device)
            self.agg_moment_reduced = wp.zeros(0, dtype=wp.float32, device=device)
            self.agg_moment2_reduced = wp.zeros(0, dtype=wp.float32, device=device)

    def clear(self):
        """Clear all contacts and reset the reducer (full clear)."""
        self.contact_count.zero_()
        self.ht_insert_failures.zero_()
        self.hashtable.clear()
        self.ht_values.zero_()

    def clear_active(self):
        """Clear only the active entries (efficient for sparse usage).

        Uses a combined kernel that clears both hashtable keys, values, and aggregate force,
        followed by a small kernel to zero the counters.
        """
        # Use fixed thread count for efficient GPU utilization
        num_threads = min(1024, self.hashtable.capacity)

        # Single kernel clears keys, values, and hydroelastic aggregates for active entries
        wp.launch(
            _clear_active_kernel,
            dim=num_threads,
            inputs=[
                self.hashtable.keys,
                self.ht_values,
                self.hashtable.active_slots,
                self.agg_force,
                self.weighted_pos_sum,
                self.weight_sum,
                self.entry_k_eff,
                self.total_depth_reduced,
                self.total_normal_reduced,
                self.agg_moment_unreduced,
                self.agg_moment_reduced,
                self.agg_moment2_reduced,
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
                self.ht_insert_failures,
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
        data.normal = self.normal
        data.shape_pairs = self.shape_pairs
        data.contact_count = self.contact_count
        data.capacity = self.capacity
        data.contact_area = self.contact_area
        data.contact_nbin_entry = self.contact_nbin_entry
        data.entry_k_eff = self.entry_k_eff
        data.agg_force = self.agg_force
        data.weighted_pos_sum = self.weighted_pos_sum
        data.weight_sum = self.weight_sum
        data.total_depth_reduced = self.total_depth_reduced
        data.total_normal_reduced = self.total_normal_reduced
        data.ht_keys = self.hashtable.keys
        data.ht_values = self.ht_values
        data.ht_active_slots = self.hashtable.active_slots
        data.ht_insert_failures = self.ht_insert_failures
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
    reducer_data: GlobalContactReducerData,
) -> int:
    """Store a contact in the buffer without reduction.

    Args:
        shape_a: First shape index
        shape_b: Second shape index
        position: Contact position in world space
        normal: Contact normal
        depth: Penetration depth (negative = penetrating)
        reducer_data: GlobalContactReducerData with all arrays

    Returns:
        Contact ID if successfully stored, -1 if buffer full
    """
    # Allocate contact slot.  On overflow, contact_count keeps incrementing
    # past capacity so (contact_count - capacity) gives the drop count.
    contact_id = wp.atomic_add(reducer_data.contact_count, 0, 1)
    if contact_id >= reducer_data.capacity:
        return -1

    # Store contact data (packed into vec4, normal octahedral-encoded into vec2)
    reducer_data.position_depth[contact_id] = wp.vec4(position[0], position[1], position[2], depth)
    reducer_data.normal[contact_id] = encode_oct(normal)
    reducer_data.shape_pairs[contact_id] = wp.vec2i(shape_a, shape_b)

    return contact_id


@wp.func
def reduce_contact_in_hashtable(
    contact_id: int,
    reducer_data: GlobalContactReducerData,
    beta: float,
    shape_transform: wp.array[wp.transform],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    shape_voxel_resolution: wp.array[wp.vec3i],
):
    """Register a buffered contact in the reduction hashtable.

    Uses single beta threshold for contact reduction with two strategies:

    1. **Normal-binned slots** (``NUM_NORMAL_BINS`` x ``NUM_SPATIAL_DIRECTIONS + 1``):
       - Spatial direction slots for contacts with depth < beta
       - 1 max-depth slot per normal bin (always participates)

    2. **Voxel-based depth slots** (``NUM_VOXEL_DEPTH_SLOTS`` voxels, grouped):
       - Voxels are grouped by ``NUM_SPATIAL_DIRECTIONS + 1``
       - Each slot tracks the deepest contact in that voxel region
       - Provides spatial coverage independent of contact normal

    Args:
        contact_id: Index of contact in buffer
        reducer_data: Reducer data
        beta: Depth threshold (contacts with depth < beta participate in spatial competition)
        shape_transform: Per-shape world transforms (for transforming position to local space)
        shape_collision_aabb_lower: Per-shape local AABB lower bounds
        shape_collision_aabb_upper: Per-shape local AABB upper bounds
        shape_voxel_resolution: Per-shape voxel grid resolution
    """
    # Read contact data from buffer (normal is octahedral-encoded)
    pd = reducer_data.position_depth[contact_id]
    normal = decode_oct(reducer_data.normal[contact_id])
    pair = reducer_data.shape_pairs[contact_id]

    position = wp.vec3(pd[0], pd[1], pd[2])
    depth = pd[3]
    shape_a = pair[0]  # Mesh shape
    shape_b = pair[1]  # Convex shape

    aabb_lower = shape_collision_aabb_lower[shape_a]
    aabb_upper = shape_collision_aabb_upper[shape_a]

    ht_capacity = reducer_data.ht_capacity

    # === Part 1: Normal-binned reduction (spatial extremes + max-depth per bin) ===
    # Get normal bin from polyhedron face matching
    bin_id = get_slot(normal)

    # Project position to 2D plane of the polyhedron face
    pos_2d = project_point_to_plane(bin_id, position)

    # Key is (shape_a, shape_b, bin_id)
    key = make_contact_key(shape_a, shape_b, bin_id)

    # Find or create the hashtable entry ONCE, then write directly to slots
    entry_idx = hashtable_find_or_insert(key, reducer_data.ht_keys, reducer_data.ht_active_slots)
    if entry_idx >= 0:
        use_beta = depth < beta * wp.length(aabb_upper - aabb_lower)
        for dir_i in range(wp.static(NUM_SPATIAL_DIRECTIONS)):
            if use_beta:
                dir_2d = get_spatial_direction_2d(dir_i)
                score = wp.dot(pos_2d, dir_2d)
                value = make_contact_value(score, contact_id)
                slot_id = dir_i
                reduction_update_slot(entry_idx, slot_id, value, reducer_data.ht_values, ht_capacity)

        max_depth_value = make_contact_value(-depth, contact_id)
        reduction_update_slot(
            entry_idx, wp.static(NUM_SPATIAL_DIRECTIONS), max_depth_value, reducer_data.ht_values, ht_capacity
        )
    else:
        wp.atomic_add(reducer_data.ht_insert_failures, 0, 1)

    # === Part 2: Voxel-based reduction (deepest contact per voxel) ===
    # Transform contact position from world space to shape_a's local space
    X_shape_ws = shape_transform[shape_a]
    X_ws_shape = wp.transform_inverse(X_shape_ws)
    position_local = wp.transform_point(X_ws_shape, position)

    # Compute voxel index using shape_a's local AABB
    voxel_res = shape_voxel_resolution[shape_a]
    voxel_idx = compute_voxel_index(position_local, aabb_lower, aabb_upper, voxel_res)

    # Clamp voxel index to valid range
    voxel_idx = wp.clamp(voxel_idx, 0, wp.static(NUM_VOXEL_DEPTH_SLOTS - 1))

    voxels_per_group = wp.static(NUM_SPATIAL_DIRECTIONS + 1)
    voxel_group = voxel_idx // voxels_per_group
    voxel_local_slot = voxel_idx % voxels_per_group

    voxel_bin_id = wp.static(NUM_NORMAL_BINS) + voxel_group
    voxel_key = make_contact_key(shape_a, shape_b, voxel_bin_id)

    voxel_entry_idx = hashtable_find_or_insert(voxel_key, reducer_data.ht_keys, reducer_data.ht_active_slots)
    if voxel_entry_idx >= 0:
        # Use -depth so atomic_max selects most penetrating (most negative depth)
        voxel_value = make_contact_value(-depth, contact_id)
        reduction_update_slot(voxel_entry_idx, voxel_local_slot, voxel_value, reducer_data.ht_values, ht_capacity)
    else:
        wp.atomic_add(reducer_data.ht_insert_failures, 0, 1)


@wp.func
def export_and_reduce_contact(
    shape_a: int,
    shape_b: int,
    position: wp.vec3,
    normal: wp.vec3,
    depth: float,
    reducer_data: GlobalContactReducerData,
    beta: float,
    shape_transform: wp.array[wp.transform],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    shape_voxel_resolution: wp.array[wp.vec3i],
) -> int:
    """Export contact to buffer and register in hashtable for reduction."""
    contact_id = export_contact_to_buffer(shape_a, shape_b, position, normal, depth, reducer_data)

    if contact_id >= 0:
        reduce_contact_in_hashtable(
            contact_id,
            reducer_data,
            beta,
            shape_transform,
            shape_collision_aabb_lower,
            shape_collision_aabb_upper,
            shape_voxel_resolution,
        )

    return contact_id


@wp.func
def export_and_reduce_contact_centered(
    shape_a: int,
    shape_b: int,
    position: wp.vec3,
    normal: wp.vec3,
    depth: float,
    centered_position: wp.vec3,
    X_ws_voxel_shape: wp.transform,
    aabb_lower_voxel: wp.vec3,
    aabb_upper_voxel: wp.vec3,
    voxel_res: wp.vec3i,
    reducer_data: GlobalContactReducerData,
) -> int:
    """Export contact to buffer and register in hashtable matching thread-block behavior.

    Differs from :func:`export_and_reduce_contact` in three ways that match
    the shared-memory reduction used for mesh-mesh contacts:

    - Spatial projection uses *centered_position* (midpoint-centered)
    - Beta threshold is fixed at 0.0001 m (not scale-relative)
    - Voxel grid uses the caller-specified AABB/transform (tri_shape's)

    Pre-prunes contacts by non-atomically reading current slot values before
    allocating a buffer slot. Contacts that cannot beat any existing winner
    are skipped entirely, reducing buffer pressure.

    Args:
        shape_a: First shape index (for hashtable key)
        shape_b: Second shape index (for hashtable key)
        position: World-space contact position (stored in buffer)
        normal: Contact normal (a-to-b)
        depth: Penetration depth
        centered_position: Midpoint-centered position for spatial projection
        X_ws_voxel_shape: World-to-local transform for voxel computation
        aabb_lower_voxel: Local AABB lower for voxel grid
        aabb_upper_voxel: Local AABB upper for voxel grid
        voxel_res: Voxel grid resolution
        reducer_data: Global reducer data
    """
    ht_capacity = reducer_data.ht_capacity
    use_beta = depth < wp.static(BETA_THRESHOLD)

    # === Normal bin: find/create hashtable entry ===
    bin_id = get_slot(normal)
    pos_2d = project_point_to_plane(bin_id, centered_position)
    key = make_contact_key(shape_a, shape_b, bin_id)

    entry_idx = hashtable_find_or_insert(key, reducer_data.ht_keys, reducer_data.ht_active_slots)

    # === Pre-prune normal bin: non-atomic reads ===
    might_win = False

    if entry_idx >= 0:
        # Check max-depth slot first (cheapest — no direction computation)
        max_depth_probe = make_contact_value(-depth, 0)
        if reducer_data.ht_values[wp.static(NUM_SPATIAL_DIRECTIONS) * ht_capacity + entry_idx] < max_depth_probe:
            might_win = True

        if not might_win and use_beta:
            for dir_i in range(wp.static(NUM_SPATIAL_DIRECTIONS)):
                if not might_win:
                    dir_2d = get_spatial_direction_2d(dir_i)
                    score = wp.dot(pos_2d, dir_2d)
                    probe = make_contact_value(score, 0)
                    if reducer_data.ht_values[dir_i * ht_capacity + entry_idx] < probe:
                        might_win = True

    # === Voxel bin: only look up if normal bin didn't already show a win ===
    position_local = wp.transform_point(X_ws_voxel_shape, position)
    voxel_idx = compute_voxel_index(position_local, aabb_lower_voxel, aabb_upper_voxel, voxel_res)
    voxel_idx = wp.clamp(voxel_idx, 0, wp.static(NUM_VOXEL_DEPTH_SLOTS - 1))

    voxels_per_group = wp.static(NUM_SPATIAL_DIRECTIONS + 1)
    voxel_group = voxel_idx // voxels_per_group
    voxel_local_slot = voxel_idx % voxels_per_group
    voxel_bin_id = wp.static(NUM_NORMAL_BINS) + voxel_group
    voxel_key = make_contact_key(shape_a, shape_b, voxel_bin_id)

    voxel_entry_idx = -1
    if not might_win:
        voxel_entry_idx = hashtable_find_or_insert(voxel_key, reducer_data.ht_keys, reducer_data.ht_active_slots)
        if voxel_entry_idx >= 0:
            voxel_probe = make_contact_value(-depth, 0)
            if reducer_data.ht_values[voxel_local_slot * ht_capacity + voxel_entry_idx] < voxel_probe:
                might_win = True

    if not might_win:
        return -1

    # === Allocate buffer slot (only for contacts that might win) ===
    contact_id = export_contact_to_buffer(shape_a, shape_b, position, normal, depth, reducer_data)
    if contact_id < 0:
        return -1

    # === Register in hashtable with real contact_id ===
    if entry_idx >= 0:
        for dir_i in range(wp.static(NUM_SPATIAL_DIRECTIONS)):
            if use_beta:
                dir_2d = get_spatial_direction_2d(dir_i)
                score = wp.dot(pos_2d, dir_2d)
                value = make_contact_value(score, contact_id)
                reduction_update_slot(entry_idx, dir_i, value, reducer_data.ht_values, ht_capacity)

        max_depth_value = make_contact_value(-depth, contact_id)
        reduction_update_slot(
            entry_idx, wp.static(NUM_SPATIAL_DIRECTIONS), max_depth_value, reducer_data.ht_values, ht_capacity
        )
    else:
        wp.atomic_add(reducer_data.ht_insert_failures, 0, 1)

    # Deferred voxel entry lookup for contacts that won via normal bin
    if voxel_entry_idx < 0:
        voxel_entry_idx = hashtable_find_or_insert(voxel_key, reducer_data.ht_keys, reducer_data.ht_active_slots)
    if voxel_entry_idx >= 0:
        voxel_value = make_contact_value(-depth, contact_id)
        reduction_update_slot(voxel_entry_idx, voxel_local_slot, voxel_value, reducer_data.ht_values, ht_capacity)
    else:
        wp.atomic_add(reducer_data.ht_insert_failures, 0, 1)

    return contact_id


@wp.kernel(enable_backward=False)
def reduce_buffered_contacts_kernel(
    reducer_data: GlobalContactReducerData,
    shape_transform: wp.array[wp.transform],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    shape_voxel_resolution: wp.array[wp.vec3i],
    total_num_threads: int,
):
    """Register buffered contacts in the hashtable for reduction.

    Uses the fixed BETA_THRESHOLD (0.1mm) for spatial competition.
    Contacts with depth < beta participate in spatial extreme competition.
    """
    tid = wp.tid()

    # Get total number of contacts written
    num_contacts = reducer_data.contact_count[0]

    # Early exit if no contacts (fast path for empty work)
    if num_contacts == 0:
        return

    # Cap at capacity
    num_contacts = wp.min(num_contacts, reducer_data.capacity)

    # Grid stride loop over contacts
    for i in range(tid, num_contacts, total_num_threads):
        reduce_contact_in_hashtable(
            i,
            reducer_data,
            wp.static(BETA_THRESHOLD),
            shape_transform,
            shape_collision_aabb_lower,
            shape_collision_aabb_upper,
            shape_voxel_resolution,
        )


# =============================================================================
# Helper functions for contact unpacking and writing
# =============================================================================


@wp.func
def unpack_contact(
    contact_id: int,
    position_depth: wp.array[wp.vec4],
    normal: wp.array[wp.vec2],
):
    """Unpack contact data from the buffer.

    Normal is stored as octahedral-encoded vec2 and decoded back to vec3.

    Args:
        contact_id: Index into the contact buffer
        position_depth: Contact buffer for position.xyz + depth
        normal: Contact buffer for octahedral-encoded normal

    Returns:
        Tuple of (position, normal, depth)
    """
    pd = position_depth[contact_id]
    n = decode_oct(normal[contact_id])

    position = wp.vec3(pd[0], pd[1], pd[2])
    depth = pd[3]

    return position, n, depth


@wp.func
def write_contact_to_reducer(
    contact_data: Any,  # ContactData struct
    reducer_data: GlobalContactReducerData,
    output_index: int,  # Unused, kept for API compatibility with write_contact_simple
):
    """Writer function that stores contacts in GlobalContactReducer for reduction.

    This follows the same signature as write_contact_simple in narrow_phase.py,
    so it can be used with create_compute_gjk_mpr_contacts and other contact
    generation functions.

    Note: Beta threshold is applied later in create_reduce_buffered_contacts_kernel,
    not at write time. This reduces register pressure on contact generation kernels.

    Args:
        contact_data: ContactData struct from contact computation
        reducer_data: GlobalContactReducerData struct with all reducer arrays
        output_index: Unused, kept for API compatibility
    """
    # Extract contact info from ContactData
    position = contact_data.contact_point_center
    normal = contact_data.contact_normal_a_to_b
    depth = contact_data.contact_distance
    shape_a = contact_data.shape_a
    shape_b = contact_data.shape_b

    # Store contact ONLY (registration to hashtable happens in a separate kernel)
    # This reduces register pressure on the contact generation kernel
    export_contact_to_buffer(
        shape_a=shape_a,
        shape_b=shape_b,
        position=position,
        normal=normal,
        depth=depth,
        reducer_data=reducer_data,
    )


def create_export_reduced_contacts_kernel(writer_func: Any):
    """Create a kernel that exports reduced contacts using a custom writer function.

    The kernel processes one hashtable ENTRY per thread (not one value slot).
    Each entry has VALUES_PER_KEY value slots (``NUM_SPATIAL_DIRECTIONS`` spatial + 1 max-depth).
    The thread reads all slots, collects unique contact IDs, and exports each
    unique contact once.

    This naturally deduplicates: one thread handles one (shape_pair, bin) entry
    and can locally track which contact IDs it has already exported.

    Args:
        writer_func: A warp function with signature (ContactData, writer_data, int) -> None.
            The third argument is an output_index (-1 indicates the writer should allocate
            a new slot). This follows the same pattern as narrow_phase.py's write_contact_simple.

    Returns:
        A warp kernel that can be launched to export reduced contacts.
    """
    # Define vector type for tracking exported contact IDs
    exported_ids_vec = wp.types.vector(length=VALUES_PER_KEY, dtype=wp.int32)

    @wp.kernel(enable_backward=False, module="unique")
    def export_reduced_contacts_kernel(
        # Hashtable arrays
        ht_keys: wp.array[wp.uint64],
        ht_values: wp.array[wp.uint64],
        ht_active_slots: wp.array[wp.int32],
        # Contact buffer arrays
        position_depth: wp.array[wp.vec4],
        normal: wp.array[wp.vec2],  # Octahedral-encoded
        shape_pairs: wp.array[wp.vec2i],
        # Global dedup flags: one int per buffer contact, for cross-entry deduplication
        exported_flags: wp.array[wp.int32],
        # Shape data for extracting margin and effective radius
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        # Per-shape contact gaps
        shape_gap: wp.array[float],
        # Writer data (custom struct)
        writer_data: Any,
        # Grid stride parameters
        total_num_threads: int,
    ):
        """Export reduced contacts to the writer.

        Uses grid stride loop to iterate over active hashtable ENTRIES.
        For each entry, reads all value slots, collects unique contact IDs,
        and exports each unique contact once. Uses atomic flags per contact_id
        for cross-entry deduplication (same contact winning multiple entries).
        """
        tid = wp.tid()

        # Get number of active entries (stored at index = ht_capacity)
        ht_capacity = ht_keys.shape[0]
        num_active = ht_active_slots[ht_capacity]

        # Early exit if no active entries (fast path for empty work)
        if num_active == 0:
            return

        # Grid stride loop over active entries
        for i in range(tid, num_active, total_num_threads):
            # Get the hashtable entry index
            entry_idx = ht_active_slots[i]

            # Track exported contact IDs for this entry (intra-entry dedup)
            exported_ids = exported_ids_vec()
            num_exported = int(0)

            # Read all value slots for this entry (slot-major layout)
            for slot in range(wp.static(VALUES_PER_KEY)):
                value = ht_values[slot * ht_capacity + entry_idx]

                # Skip empty slots (value = 0)
                if value == wp.uint64(0):
                    continue

                # Extract contact ID from low 32 bits
                contact_id = unpack_contact_id(value)

                # Skip if already exported within this entry
                if is_contact_already_exported(contact_id, exported_ids, num_exported):
                    continue

                # Record this contact ID for intra-entry dedup
                exported_ids[num_exported] = contact_id
                num_exported = num_exported + 1

                # Cross-entry dedup: same contact can win slots in different entries
                # (e.g., normal-bin AND voxel entry). Atomic flag per contact_id.
                old_flag = wp.atomic_add(exported_flags, contact_id, 1)
                if old_flag > 0:
                    continue

                # Unpack contact data
                position, contact_normal, depth = unpack_contact(contact_id, position_depth, normal)

                # Get shape pair
                pair = shape_pairs[contact_id]
                shape_a = pair[0]
                shape_b = pair[1]

                # Extract margin offsets from shape_data (stored in w component)
                margin_offset_a = shape_data[shape_a][3]
                margin_offset_b = shape_data[shape_b][3]

                # Compute effective radius for spheres, capsules, and cones
                radius_eff_a = compute_effective_radius(shape_types[shape_a], shape_data[shape_a])
                radius_eff_b = compute_effective_radius(shape_types[shape_b], shape_data[shape_b])

                # Use additive per-shape contact gap (matching broad/narrow phase)
                gap_a = shape_gap[shape_a]
                gap_b = shape_gap[shape_b]
                gap_sum = gap_a + gap_b

                # Create ContactData struct
                contact_data = ContactData()
                contact_data.contact_point_center = position
                contact_data.contact_normal_a_to_b = contact_normal
                contact_data.contact_distance = depth
                contact_data.radius_eff_a = radius_eff_a
                contact_data.radius_eff_b = radius_eff_b
                contact_data.margin_a = margin_offset_a
                contact_data.margin_b = margin_offset_b
                contact_data.shape_a = shape_a
                contact_data.shape_b = shape_b
                contact_data.gap_sum = gap_sum

                # Call the writer function
                writer_func(contact_data, writer_data, -1)

    return export_reduced_contacts_kernel


@wp.kernel(enable_backward=False, module="unique")
def mesh_triangle_contacts_to_reducer_kernel(
    shape_types: wp.array[int],
    shape_data: wp.array[wp.vec4],
    shape_transform: wp.array[wp.transform],
    shape_source: wp.array[wp.uint64],
    shape_gap: wp.array[float],
    shape_heightfield_index: wp.array[wp.int32],
    heightfield_data: wp.array[HeightfieldData],
    heightfield_elevations: wp.array[wp.float32],
    triangle_pairs: wp.array[wp.vec3i],
    triangle_pairs_count: wp.array[int],
    reducer_data: GlobalContactReducerData,
    total_num_threads: int,
):
    """Process mesh/heightfield-triangle contacts and store them in GlobalContactReducer.

    This kernel processes triangle pairs (mesh-or-hfield shape, convex-shape, triangle_index)
    and computes contacts using GJK/MPR, storing results in the GlobalContactReducer for
    subsequent reduction and export.

    Uses grid stride loop over triangle pairs.
    """
    tid = wp.tid()

    num_triangle_pairs = triangle_pairs_count[0]

    for i in range(tid, num_triangle_pairs, total_num_threads):
        if i >= triangle_pairs.shape[0]:
            break

        triple = triangle_pairs[i]
        shape_a = triple[0]  # Mesh or heightfield shape
        shape_b = triple[1]  # Convex shape
        tri_idx = triple[2]

        type_a = shape_types[shape_a]

        if type_a == GeoType.HFIELD:
            # Heightfield triangle
            hfd = heightfield_data[shape_heightfield_index[shape_a]]
            X_ws_a = shape_transform[shape_a]
            shape_data_a, v0_world = get_triangle_shape_from_heightfield(hfd, heightfield_elevations, X_ws_a, tri_idx)
        else:
            # Mesh triangle (mesh_id already validated by midphase)
            mesh_id_a = shape_source[shape_a]
            scale_data_a = shape_data[shape_a]
            mesh_scale_a = wp.vec3(scale_data_a[0], scale_data_a[1], scale_data_a[2])
            X_ws_a = shape_transform[shape_a]
            shape_data_a, v0_world = get_triangle_shape_from_mesh(mesh_id_a, mesh_scale_a, X_ws_a, tri_idx)

        # Extract shape B data
        pos_b, quat_b, shape_data_b, _scale_b, margin_offset_b = extract_shape_data(
            shape_b,
            shape_transform,
            shape_types,
            shape_data,
            shape_source,
        )

        # Triangle position is vertex A in world space.
        # For heightfield prisms, edges are in heightfield-local space
        # so we pass the heightfield rotation to let MPR/GJK work in
        # that frame (where -Z is always the down axis).
        pos_a = v0_world
        if type_a == GeoType.HFIELD:
            quat_a = wp.transform_get_rotation(shape_transform[shape_a])
        else:
            quat_a = wp.quat_identity()

        # Back-face culling: skip when the convex center is behind the
        # triangle face.  TRIANGLE_PRISM (heightfields) handles this
        # via its extruded support function.
        if shape_data_a.shape_type == int(GeoTypeEx.TRIANGLE):
            face_normal = wp.cross(shape_data_a.scale, shape_data_a.auxiliary)
            center_dist = wp.dot(face_normal, pos_b - pos_a)
            if center_dist < 0.0:
                continue

        # Extract margin offset for shape A (signed distance padding)
        margin_offset_a = shape_data[shape_a][3]

        # Use additive per-shape contact gap for detection threshold
        gap_a = shape_gap[shape_a]
        gap_b = shape_gap[shape_b]
        gap_sum = gap_a + gap_b

        # Compute and write contacts using GJK/MPR
        wp.static(create_compute_gjk_mpr_contacts(write_contact_to_reducer))(
            shape_data_a,
            shape_data_b,
            quat_a,
            quat_b,
            pos_a,
            pos_b,
            gap_sum,
            shape_a,
            shape_b,
            margin_offset_a,
            margin_offset_b,
            reducer_data,
        )
