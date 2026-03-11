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
Provides functions for generating and searching for unique keys for pairs and triplets of indices.

TODO: Add more detailed description and documentation.
"""

import warp as wp

from ..core.types import int32, int64, uint32, uint64, vec2i

###
# Module interface
###

__all__ = [
    "binary_search_find_pair",
    "binary_search_find_range_start",
    "build_pair_key2",
    "make_build_pair_key3_func",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###


def make_bitmask(num_bits: int) -> int:
    """
    Returns an all-ones mask for the requested number of bits.

    Examples:
        num_bits=20 -> 0x00000000000FFFFF
        num_bits=23 -> 0x00000000007FFFFF

    Args:
        num_bits (int): Number of bits to set in the mask.

    Returns:
        int: Bitmask with the specified number of lower bits set to `1`.
    """
    # Ensure the number of bits is valid
    if num_bits <= 0 or num_bits > 64:
        raise ValueError(f"`num_bits` was {num_bits}, but must be positive integer in the range [1, 64].")
    # Handle the special case for 64 bits
    if num_bits == 64:
        return 0xFFFFFFFFFFFFFFFF
    # General case
    return (1 << num_bits) - 1


@wp.func
def build_pair_key2(index_A: wp.uint32, index_B: wp.uint32) -> wp.uint64:
    """
    Build a 63-bit key from two indices with the following layout:
    - The highest bit is always `0`, reserved as a sign bit to support conversion to signed int64.
    - Upper 31 bits: lower 31 bits of index_A
    - Lower 32 bits: all 32 bits of index_B

    Args:
        index_A (wp.uint32): First index.
        index_B (wp.uint32): Second index.

    Returns:
        wp.uint64: Combined 64-bit key.
    """
    key = wp.uint64(index_A & wp.uint32(wp.static(make_bitmask(31))))
    key = key << wp.uint64(32)
    key = key | wp.uint64(index_B)
    return key


def make_build_pair_key3_func(main_key_bits: int, aux_key_bits: int | None = None):
    """
    Generates a function that builds a 63-bit key from three indices with the following layout:
    - The highest bit is always `0`, reserved as a sign bit to support conversion to signed int64.
    - Upper `main_key_bits` bits: lower `main_key_bits` bits of index_A
    - Middle `main_key_bits` bits: lower `main_key_bits` bits of index_B
    - Lower `aux_key_bits` bits: lower `aux_key_bits` bits of index_C

    Note:
    - The total number of bits used is `2 * main_key_bits + aux_key_bits`, which must be less than or equal to 63.

    Args:
        main_key_bits (int): Number of bits to allocate for index_A and index_B.
        aux_key_bits (int, optional): Number of bits to allocate for index_C.
            If `None`, it will be set to `63 - 2 * main_key_bits`.

    Returns:
        function: A Warp function that takes three `wp.uint32` indices and returns a combined `wp.uint64` key.
    """
    # Ensure the number of bits is valid
    if main_key_bits <= 0 or main_key_bits > 32:
        raise ValueError(f"`main_key_bits` was {main_key_bits}, but must be positive integer in the range [1, 32].")
    if aux_key_bits is None:
        aux_key_bits = 63 - 2 * main_key_bits
    if aux_key_bits <= 0 or aux_key_bits > 32:
        raise ValueError(f"`aux_key_bits` was {aux_key_bits}, but must be positive integer in the range [1, 32].")
    if 2 * main_key_bits + aux_key_bits != 63:
        raise ValueError(
            f"`2 * main_key_bits + aux_key_bits` was {2 * main_key_bits + aux_key_bits}, but must be equal to 63 bits."
        )

    # Precompute bitmasks for the specified bit widths
    MAIN_BITMASK = make_bitmask(main_key_bits)
    AUX_BITMASK = make_bitmask(aux_key_bits)

    # Define the function
    @wp.func
    def _build_pair_key3(index_A: uint32, index_B: uint32, index_C: uint32) -> uint64:
        key = wp.uint64(index_A & wp.uint32(MAIN_BITMASK))
        key = key << wp.uint64(main_key_bits)
        key = key | wp.uint64(index_B & wp.uint32(MAIN_BITMASK))
        key = key << wp.uint64(aux_key_bits)
        key = key | wp.uint64(index_C & wp.uint32(AUX_BITMASK))
        return key

    # Return the generated function
    return _build_pair_key3


@wp.func
def binary_search_find_pair(
    num_pairs: int32,
    target: vec2i,
    pairs: wp.array(dtype=vec2i),
) -> int32:
    """
    Performs binary-search over a sorted array of pairs to find the index of a target pair.

    Assumes that pairs are sorted in ascending lexicographical
    order, i.e. first by the first element, then by the second.

    Args:
        num_pairs (int32): Number of "active" pairs in the array.\n
            This is required because not all elements may be active.
        target (vec2i): The target pair to search for.
        pairs (wp.array(dtype=vec2i)): Sorted array of pairs to search within.

    Returns:
        Index of the target pair if found, otherwise `-1`.
    """
    lower = int32(0)
    upper = num_pairs
    while lower < upper:
        mid = (lower + upper) >> 1
        mid_pair = pairs[mid]
        # Compare pairs lexicographically (first by the first element, then by the second)
        if mid_pair[0] < target[0] or (mid_pair[0] == target[0] and mid_pair[1] < target[1]):
            lower = mid + 1
        elif mid_pair[0] > target[0] or (mid_pair[0] == target[0] and mid_pair[1] > target[1]):
            upper = mid
        else:
            # Found exact match
            return mid
    # Not found
    return -1


@wp.func
def binary_search_find_range_start(
    lower: int32,
    upper: int32,
    target: uint64,
    keys: wp.array(dtype=uint64),
) -> int32:
    """
    Performs binary-search over a sorted array of integer keys
    to find the start index of the first occurrence of target.

    Assumes that keys are sorted in ascending order.

    Args:
        lower (wp.int32): Lower bound index for the search (inclusive).
        upper (wp.int32): Upper bound index for the search (exclusive).
        target (wp.uint64): The target key to search for.
        keys (wp.array(dtype=wp.uint64)): Sorted array of keys to search within.

    Returns:
        Index of the first occurrence of target if found, otherwise `-1`.
    """
    # Find lower bound: first position where keys[i] >= target
    left = lower
    right = upper
    while left < right:
        mid = left + (right - left) // 2
        if keys[mid] < target:
            left = mid + 1
        else:
            right = mid
    # Check if the key was actually found
    if left >= upper or keys[left] != target:
        return -1
    # Return the index of the first occurrence of the target key
    return left


@wp.func_native("""return 0x7FFFFFFFFFFFFFFFull;""")
def uint64_sentinel_value() -> wp.uint64: ...


###
# Kernels
###


@wp.kernel
def _prepare_key_sort(
    # Inputs:
    num_active_keys: wp.array(dtype=wp.int32),
    keys_source: wp.array(dtype=wp.uint64),
    # Outputs:
    keys: wp.array(dtype=wp.uint64),
    sorted_to_unsorted_map: wp.array(dtype=wp.int32),
):
    """
    Prepares keys and sorting-maps for radix sort.

    Args:
        num_active_keys (wp.array(dtype=wp.int32)): Number of active keys to copy
        keys_source (wp.array(dtype=wp.uint64)): Source array of keys
        keys (wp.array(dtype=wp.uint64)): Destination array of keys for sorting
        sorted_to_unsorted_map (wp.array(dtype=wp.int32)): Map from sorted indices to original unsorted indices
    """
    # Retrieve the thread index
    tid = wp.tid()

    # Copy active keys and initialize the sorted-to-unsorted index map
    if tid < num_active_keys[0]:
        keys[tid] = keys_source[tid]
        sorted_to_unsorted_map[tid] = tid

    # Otherwise fill unused slots with the sentinel value
    # NOTE: This ensures that these entries sort to the end when treated as signed int64
    else:
        # keys[tid] = wp.static(make_bitmask(63))
        keys[tid] = uint64_sentinel_value()


###
# Launchers
###


def prepare_key_sort(
    num_active: wp.array,
    unsorted: wp.array,
    sorted: wp.array,
    sorted_to_unsorted_map: wp.array,
):
    """
    Prepares keys and sorting-maps for radix sort.

    Args:
        num_active (wp.array):
            An array containing the number of active keys to be sorted.
        unsorted (wp.array):
            The source array of keys to be sorted.
        sorted (wp.array):
            The destination array where sorted keys will be stored.
        sorted_to_unsorted_map (wp.array):
            An array of index-mappings from sorted to source key indices
    """
    wp.launch(
        kernel=_prepare_key_sort,
        dim=sorted.size,
        inputs=[num_active, unsorted, sorted, sorted_to_unsorted_map],
        device=sorted.device,
    )


###
# Interfaces
###


class KeySorter:
    """
    A utility class for sorting integer keys using radix sort.
    """

    def __init__(self, max_num_keys: int, device: wp.DeviceLike = None):
        """
        Creates a KeySorter instance to sort keys using radix sort.

        Args:
            max_num_keys (int): Maximum number of keys to sort.
            device (wp.DeviceLike, optional): Device to allocate buffers on (None for default).
        """
        # Declare and initialize the maximum number of keys
        # NOTE: This is used set dimensions of all kernel launches
        self._max_num_keys = max_num_keys

        # Cache the Warp device on which data will be
        # allocated and all kernels will be executed
        self._device = device

        # Allocate data buffers for key sorting
        # NOTE: Allocations are multiplied by a factor of
        # 2 as required by the Warp radix sort algorithm
        with wp.ScopedDevice(device):
            self._sorted_keys = wp.zeros(2 * self._max_num_keys, dtype=uint64)
            self._sorted_to_unsorted_map = wp.zeros(2 * self._max_num_keys, dtype=int32)

        # Define a view of the sorted keys as int64
        # NOTE: This required in order to use Warp's radix_sort_pairs, which only supports signed integers
        self._sorted_keys_int64 = wp.array(
            ptr=self._sorted_keys.ptr,
            shape=self._sorted_keys.shape,
            device=self._sorted_keys.device,
            dtype=int64,
            copy=False,
        )

    @property
    def device(self) -> wp.DeviceLike:
        """Returns the device on which the KeySorter is allocated."""
        return self._device

    @property
    def sorted_keys(self) -> wp.array:
        """Returns the sorted keys array."""
        return self._sorted_keys

    @property
    def sorted_keys_int64(self) -> wp.array:
        """Returns the sorted keys array as an int64 view."""
        return self._sorted_keys_int64

    @property
    def sorted_to_unsorted_map(self) -> wp.array:
        """Returns the sorted-to-unsorted index map array."""
        return self._sorted_to_unsorted_map

    def sort(self, num_active_keys: wp.array, keys: wp.array):
        """
        Sorts the provided keys using radix sort.

        Args:
            num_active_keys (wp.array(dtype=int32)): Number of active keys to sort.
            keys (wp.array(dtype=uint64)): The source keys to be sorted.
        """
        # Check compatibility of input sizes
        if num_active_keys.device != self._device:
            raise ValueError("`num_active_keys` device does not match the KeySorter device.")
        if keys.device != self._device:
            raise ValueError("`keys` device does not match the KeySorter device.")
        if keys.dtype != self._sorted_keys.dtype:
            raise ValueError(
                f"`keys` dtype ({keys.dtype}) does not match the sorted_keys dtype ({self._sorted_keys.dtype})"
            )
        if keys.size > 2 * self._max_num_keys:
            raise ValueError(
                "`keys` size exceeds the supported maximum number of keys."
                f"keys must be at most {2 * self._max_num_keys}, but is {keys.size}."
            )

        # First prepare keys and sorting maps for radix sort
        wp.launch(
            kernel=_prepare_key_sort,
            dim=self._max_num_keys,
            inputs=[num_active_keys, keys, self._sorted_keys, self._sorted_to_unsorted_map],
            device=self._device,
        )

        # Perform the radix sort on the key-index pairs
        wp.utils.radix_sort_pairs(self._sorted_keys_int64, self._sorted_to_unsorted_map, self._max_num_keys)
