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

"""
Contact matching across simulation timesteps for warm-starting physics solvers.

Tracks contacts between timesteps by matching (key, payload) pairs, enabling:
- Contact persistence for stable simulation
- Warm-starting constraint solvers
- Smooth contact forces over time

The matcher uses sorted keys with binary search for O(log n) lookups per contact.
"""

import warp as wp


@wp.func
def binary_search_find_range_start(
    keys: wp.array(dtype=wp.uint64, ndim=1),
    key_to_find: wp.uint64,
    lower: wp.int32,
    upper: wp.int32,
) -> wp.int32:
    """
    Binary search to find the start index of the first occurrence of key_to_find.
    Returns -1 if key is not found.
    """
    # Find lower bound: first position where keys[i] >= key_to_find
    left = lower
    right = upper

    while left < right:
        mid = left + (right - left) // 2
        if keys[mid] < key_to_find:
            left = mid + 1
        else:
            right = mid

    # Validate that the key was actually found
    if left >= upper or keys[left] != key_to_find:
        return wp.int32(-1)

    return left


@wp.kernel(enable_backward=False)
def find_keys_in_buffer_and_update_map(
    # Prev step data
    sorted_keys_prev_step: wp.array(dtype=wp.uint64, ndim=1),
    num_keys_prev_step: wp.array(dtype=wp.int32, ndim=1),
    payloads_prev_step: wp.array(dtype=wp.uint32, ndim=1),
    sorted_to_unsorted_map_prev_step: wp.array(dtype=wp.int32, ndim=1),
    # Current step data
    raw_keys_current_step: wp.array(dtype=wp.uint64, ndim=1),
    num_keys_current_step: wp.array(dtype=wp.int32, ndim=1),
    payloads_current_step: wp.array(dtype=wp.uint32, ndim=1),
    # Output
    result_index_map_new_to_old: wp.array(dtype=wp.int32, ndim=1),
):
    """
    Match current contacts to previous timestep contacts using (key, payload) pairs.

    For each current contact, finds matching contact from previous step by:
    1. Binary search on sorted keys (O(log n))
    2. Linear scan through matching keys to find matching payload

    Outputs mapping: current_index -> previous_original_index (or -1 if new contact)
    """
    tid = wp.tid()
    if tid >= num_keys_current_step[0]:
        return

    key_to_find = raw_keys_current_step[tid]
    payload_to_find = payloads_current_step[tid]
    count = num_keys_prev_step[0]

    start = binary_search_find_range_start(sorted_keys_prev_step, key_to_find, 0, count)
    if start == -1:
        result_index_map_new_to_old[tid] = -1
        return

    result = wp.int32(-1)

    while start < count:
        if sorted_keys_prev_step[start] != key_to_find:
            break

        if payloads_prev_step[start] == payload_to_find:
            result = sorted_to_unsorted_map_prev_step[start]
            break

        start += 1

    result_index_map_new_to_old[tid] = result


@wp.func_native("""
    return 0x7FFFFFFFFFFFFFFFull;
""")
def uint64_sentinel_value() -> wp.uint64: ...


@wp.kernel(enable_backward=False)
def prepare_sort(
    key_source: wp.array(dtype=wp.uint64, ndim=1),
    keys: wp.array(dtype=wp.uint64, ndim=1),
    sorted_to_unsorted_map: wp.array(dtype=wp.int32, ndim=1),
    count: wp.array(dtype=wp.int32, ndim=1),
):
    """Prepare data for radix sort by copying keys and initializing index map."""
    tid = wp.tid()
    if tid < count[0]:
        keys[tid] = key_source[tid]
        sorted_to_unsorted_map[tid] = tid
    else:
        keys[tid] = (
            uint64_sentinel_value()
        )  # Fill unused slots with sentinel value (sorts to end when treated as signed int64)


@wp.kernel(enable_backward=False)
def reorder_payloads(
    payload_source: wp.array(dtype=wp.uint32, ndim=1),
    payloads: wp.array(dtype=wp.uint32, ndim=1),
    sorted_to_unsorted_map: wp.array(dtype=wp.int32, ndim=1),
    count: wp.array(dtype=wp.int32, ndim=1),
    count_copy: wp.array(dtype=wp.int32, ndim=1),
):
    """Reorder payloads to match sorted key order and save count for next timestep."""
    tid = wp.tid()
    if tid == 0:
        count_copy[0] = count[0]  # Save current count for next timestep

    if tid < count[0]:
        payloads[tid] = payload_source[sorted_to_unsorted_map[tid]]


class ContactMatcher:
    """
    Matches contacts across simulation timesteps for warm-starting physics solvers.

    Maintains sorted contact data from previous timestep and matches current contacts
    to find persistent contacts. Uses binary search for efficient O(log n) lookups.

    Args:
        max_num_contacts: Maximum number of contacts to track
        device: Device to allocate buffers on (None for default)
    """

    def __init__(self, max_num_contacts: int, device=None):
        self.max_num_contacts = max_num_contacts
        # Factor of 2 because that is a requirement of the sort algorithm
        self.sorted_keys_prev_step = wp.zeros(2 * max_num_contacts, dtype=wp.uint64, device=device)
        self.sorted_to_unsorted_map_prev_step = wp.zeros(2 * max_num_contacts, dtype=wp.int32, device=device)
        self.payloads_prev_step = wp.zeros(max_num_contacts, dtype=wp.uint32, device=device)
        self.num_keys_prev_step = wp.zeros(1, dtype=wp.int32, device=device)

    def launch(
        self,
        keys: wp.array(dtype=wp.uint64, ndim=1),
        num_keys: wp.array(dtype=wp.int32, ndim=1),
        payloads: wp.array(dtype=wp.uint32, ndim=1),
        result_index_map_new_to_old: wp.array(dtype=wp.int32, ndim=1),
        device=None,
    ):
        """
        Match current contacts to previous timestep and prepare for next timestep.

        Args:
            keys: Contact pair keys (shape_a, shape_b, optional triangle_idx)
            num_keys: Single-element array with number of active contacts
            payloads: Contact feature IDs for disambiguation
            result_index_map_new_to_old: Output mapping from current to previous indices (-1 for new contacts)
            device: Device to launch on (None for default)
        """
        wp.launch(
            kernel=find_keys_in_buffer_and_update_map,
            dim=self.max_num_contacts,
            inputs=[
                self.sorted_keys_prev_step,
                self.num_keys_prev_step,
                self.payloads_prev_step,
                self.sorted_to_unsorted_map_prev_step,
                keys,
                num_keys,
                payloads,
                result_index_map_new_to_old,
            ],
            device=device,
        )

        wp.launch(
            kernel=prepare_sort,
            dim=self.max_num_contacts,
            inputs=[keys, self.sorted_keys_prev_step, self.sorted_to_unsorted_map_prev_step, num_keys],
            device=device,
        )

        # Cast uint64 keys to int64 for radix_sort_pairs (which supports int64 + int32)
        keys_as_int64 = wp.array(
            ptr=self.sorted_keys_prev_step.ptr,
            dtype=wp.int64,
            shape=self.sorted_keys_prev_step.shape,
            device=self.sorted_keys_prev_step.device,
            copy=False,
        )

        wp.utils.radix_sort_pairs(keys_as_int64, self.sorted_to_unsorted_map_prev_step, self.max_num_contacts)

        wp.launch(
            kernel=reorder_payloads,
            dim=self.max_num_contacts,
            inputs=[
                payloads,
                self.payloads_prev_step,
                self.sorted_to_unsorted_map_prev_step,
                num_keys,
                self.num_keys_prev_step,
            ],
            device=device,
        )
