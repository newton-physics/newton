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

"""GPU-friendly hash table for concurrent key-to-index mapping.

This module provides a generic hash table that maps keys to entry indices.
It is designed for GPU kernels where many threads insert concurrently.

Key features:
- Thread-safe insertion using atomic compare-and-swap
- Active entry tracking for efficient clearing
- Power-of-two capacity for fast modulo via bitwise AND

The hash table does NOT store values - it only maps keys to entry indices.
Callers can use these indices to access their own value storage.
"""

from __future__ import annotations

import warp as wp

# Sentinel value for empty slots
_HASHTABLE_EMPTY_KEY_VALUE = 0xFFFFFFFFFFFFFFFF
HASHTABLE_EMPTY_KEY = wp.constant(wp.uint64(_HASHTABLE_EMPTY_KEY_VALUE))


def _next_power_of_two(n: int) -> int:
    """Round up to the next power of two."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


@wp.func
def _hashtable_hash(key: wp.uint64, capacity_mask: int) -> int:
    """Compute hash index using a simplified mixer."""
    h = key
    h = h ^ (h >> wp.uint64(33))
    h = h * wp.uint64(0xFF51AFD7ED558CCD)
    h = h ^ (h >> wp.uint64(33))
    return int(h) & capacity_mask


@wp.func
def hashtable_find_or_insert(
    key: wp.uint64,
    keys: wp.array(dtype=wp.uint64),
    active_slots: wp.array(dtype=wp.int32),
) -> int:
    """Find or insert a key and return the entry index.

    This function locates an existing entry or creates a new one for the key.
    The returned entry index can be used to access caller-managed value storage.

    Args:
        key: The uint64 key to find or insert
        keys: The hash table keys array (length must be power of two)
        active_slots: Array of size (capacity + 1) tracking active entry indices.
                      active_slots[capacity] is the count of active entries.

    Returns:
        Entry index (>= 0) if successful, -1 if the table is full
    """
    capacity = keys.shape[0]
    capacity_mask = capacity - 1
    idx = _hashtable_hash(key, capacity_mask)

    # Linear probing with a maximum of 'capacity' attempts
    for _i in range(capacity):
        # Read first to check if key exists (keys only transition EMPTY -> KEY)
        stored_key = keys[idx]

        if stored_key == key:
            # Key already exists - return its index
            return idx

        if stored_key == HASHTABLE_EMPTY_KEY:
            # Try to claim this slot
            old_key = wp.atomic_cas(keys, idx, HASHTABLE_EMPTY_KEY, key)

            if old_key == HASHTABLE_EMPTY_KEY:
                # We claimed an empty slot - this is a NEW entry
                # Add to active slots list
                active_idx = wp.atomic_add(active_slots, capacity, 1)
                if active_idx < capacity:
                    active_slots[active_idx] = idx
                return idx
            elif old_key == key:
                # Another thread just inserted the same key - use it
                return idx
            # else: Another thread claimed with different key - continue probing

        # Collision with different key - linear probe to next slot
        idx = (idx + 1) & capacity_mask

    # Table is full
    return -1


@wp.kernel
def _hashtable_clear_keys_kernel(
    keys: wp.array(dtype=wp.uint64),
    active_slots: wp.array(dtype=wp.int32),
    capacity: int,
    num_threads: int,
):
    """Kernel to clear only the active keys in the hash table.

    Uses grid-stride loop for efficient thread utilization.
    Reads count from GPU memory - works because all threads read before any writes.
    """
    tid = wp.tid()

    # Read count from GPU - stored at active_slots[capacity]
    # All threads read this value before any modifications happen
    count = active_slots[capacity]

    # Grid-stride loop: each thread processes multiple entries if needed
    i = tid
    while i < count:
        entry_idx = active_slots[i]
        keys[entry_idx] = HASHTABLE_EMPTY_KEY
        i += num_threads


@wp.kernel
def _zero_count_kernel(
    active_slots: wp.array(dtype=wp.int32),
    capacity: int,
):
    """Zero the count element after clearing."""
    active_slots[capacity] = 0


class HashTable:
    """Generic hash table for concurrent key-to-index mapping.

    Uses open addressing with linear probing. Designed for GPU kernels
    where many threads insert concurrently.

    This hash table does NOT store values - it only maps keys to entry indices.
    Callers can use the entry indices to access their own value storage with
    whatever layout they prefer.

    Attributes:
        capacity: Maximum number of unique keys (power of two)
        keys: Warp array storing the keys
        active_slots: Array tracking active slot indices (size = capacity + 1)
        device: The device where the table is allocated
    """

    def __init__(self, capacity: int, device: str | None = None):
        """Initialize an empty hash table.

        Args:
            capacity: Maximum number of unique keys. Rounded up to power of two.
            device: Warp device (e.g., "cuda:0", "cpu").
        """
        self.capacity = _next_power_of_two(capacity)
        self.device = device

        # Allocate arrays
        self.keys = wp.zeros(self.capacity, dtype=wp.uint64, device=device)
        self.active_slots = wp.zeros(self.capacity + 1, dtype=wp.int32, device=device)

        self.clear()

    def clear(self):
        """Clear all entries in the hash table."""
        self.keys.fill_(_HASHTABLE_EMPTY_KEY_VALUE)
        self.active_slots.zero_()

    def clear_active(self):
        """Clear only the active entries. CUDA graph capture compatible.

        Uses two kernel launches:
        1. Clear all active hashtable keys using grid-stride loop
        2. Zero the count element

        The two-kernel approach is needed to avoid race conditions on CPU where
        threads execute sequentially.
        """
        # Use fixed thread count to cover the GPU (65536 = 256 blocks x 256 threads)
        # Grid-stride loop handles any number of active entries
        num_threads = min(65536, self.capacity)
        wp.launch(
            _hashtable_clear_keys_kernel,
            dim=num_threads,
            inputs=[self.keys, self.active_slots, self.capacity, num_threads],
            device=self.device,
        )
        # Zero the count in a separate kernel to avoid CPU race condition
        wp.launch(
            _zero_count_kernel,
            dim=1,
            inputs=[self.active_slots, self.capacity],
            device=self.device,
        )
