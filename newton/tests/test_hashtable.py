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

"""Tests for the hash table."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.contact_reduction_global import reduction_insert_slot
from newton._src.geometry.hashtable import HashTable
from newton.tests.unittest_utils import add_function_test, get_test_devices

# =============================================================================
# Test class
# =============================================================================


class TestHashTable(unittest.TestCase):
    """Test cases for the HashTable class."""

    pass


# =============================================================================
# Test functions
# =============================================================================


def test_basic_creation(test, device):
    """Test creating an empty hash table."""
    ht = HashTable(capacity=64, device=device)
    test.assertGreaterEqual(ht.capacity, 64)
    # Check that keys are initialized to empty
    keys_np = ht.keys.numpy()
    test.assertTrue(np.all(keys_np == 0xFFFFFFFFFFFFFFFF))


def test_power_of_two_rounding(test, device):
    """Test that capacity is rounded to power of two."""
    ht1 = HashTable(capacity=100, device=device)
    test.assertEqual(ht1.capacity, 128)  # Next power of 2

    ht2 = HashTable(capacity=64, device=device)
    test.assertEqual(ht2.capacity, 64)  # Already power of 2

    ht3 = HashTable(capacity=1, device=device)
    test.assertEqual(ht3.capacity, 1)


def test_insert_single_slot(test, device):
    """Test inserting values into different slots of the same key."""
    values_per_key = 13

    @wp.kernel
    def insert_test_kernel(
        keys: wp.array(dtype=wp.uint64),
        values: wp.array(dtype=wp.uint64),
        active_slots: wp.array(dtype=wp.int32),
    ):
        # Insert into slot 0
        reduction_insert_slot(wp.uint64(123), 0, wp.uint64(100), keys, values, active_slots)
        # Insert into slot 5
        reduction_insert_slot(wp.uint64(123), 5, wp.uint64(200), keys, values, active_slots)
        # Insert into slot 12
        reduction_insert_slot(wp.uint64(123), 12, wp.uint64(300), keys, values, active_slots)

    ht = HashTable(capacity=64, device=device)
    # Allocate values array externally (caller-managed)
    values = wp.zeros(ht.capacity * values_per_key, dtype=wp.uint64, device=device)

    wp.launch(
        insert_test_kernel,
        dim=1,
        inputs=[ht.keys, values, ht.active_slots],
        device=device,
    )
    wp.synchronize()

    # Find the entry
    keys_np = ht.keys.numpy()
    values_np = values.numpy()

    entry_idx = np.where(keys_np == 123)[0]
    test.assertEqual(len(entry_idx), 1)
    idx = entry_idx[0]

    # Check values at each slot (slot-major layout: slot * capacity + entry_idx)
    test.assertEqual(values_np[0 * ht.capacity + idx], 100)
    test.assertEqual(values_np[5 * ht.capacity + idx], 200)
    test.assertEqual(values_np[12 * ht.capacity + idx], 300)


def test_atomic_max_behavior(test, device):
    """Test that atomic max correctly keeps the maximum value."""
    values_per_key = 13

    @wp.kernel
    def atomic_max_test_kernel(
        keys: wp.array(dtype=wp.uint64),
        values: wp.array(dtype=wp.uint64),
        active_slots: wp.array(dtype=wp.int32),
    ):
        tid = wp.tid()
        # All threads try to write to same key and slot
        # Values are 1, 2, 3, ..., 100
        reduction_insert_slot(wp.uint64(999), 0, wp.uint64(tid + 1), keys, values, active_slots)

    ht = HashTable(capacity=64, device=device)
    values = wp.zeros(ht.capacity * values_per_key, dtype=wp.uint64, device=device)

    wp.launch(
        atomic_max_test_kernel,
        dim=100,
        inputs=[ht.keys, values, ht.active_slots],
        device=device,
    )
    wp.synchronize()

    # Find the entry
    keys_np = ht.keys.numpy()
    values_np = values.numpy()

    entry_idx = np.where(keys_np == 999)[0]
    test.assertEqual(len(entry_idx), 1)
    idx = entry_idx[0]

    # The maximum value should be 100 (slot-major layout)
    test.assertEqual(values_np[0 * ht.capacity + idx], 100)


def test_multiple_keys(test, device):
    """Test inserting multiple different keys."""
    values_per_key = 1

    @wp.kernel
    def multi_key_kernel(
        keys: wp.array(dtype=wp.uint64),
        values: wp.array(dtype=wp.uint64),
        active_slots: wp.array(dtype=wp.int32),
    ):
        tid = wp.tid()
        key = wp.uint64(tid + 1)  # Keys 1, 2, 3, ...
        value = wp.uint64((tid + 1) * 10)  # Values 10, 20, 30, ...
        reduction_insert_slot(key, 0, value, keys, values, active_slots)

    ht = HashTable(capacity=256, device=device)
    values = wp.zeros(ht.capacity * values_per_key, dtype=wp.uint64, device=device)

    wp.launch(
        multi_key_kernel,
        dim=100,
        inputs=[ht.keys, values, ht.active_slots],
        device=device,
    )
    wp.synchronize()

    # Check that we have 100 entries
    keys_np = ht.keys.numpy()
    non_empty = keys_np != 0xFFFFFFFFFFFFFFFF
    test.assertEqual(np.sum(non_empty), 100)

    # Check active slots count
    active_count = ht.active_slots.numpy()[ht.capacity]
    test.assertEqual(active_count, 100)


def test_clear(test, device):
    """Test clearing the hash table."""
    values_per_key = 13

    @wp.kernel
    def insert_kernel(
        keys: wp.array(dtype=wp.uint64),
        values: wp.array(dtype=wp.uint64),
        active_slots: wp.array(dtype=wp.int32),
    ):
        tid = wp.tid()
        reduction_insert_slot(wp.uint64(tid + 1), 0, wp.uint64(tid * 10), keys, values, active_slots)

    ht = HashTable(capacity=64, device=device)
    values = wp.zeros(ht.capacity * values_per_key, dtype=wp.uint64, device=device)

    # Insert some data
    wp.launch(
        insert_kernel,
        dim=50,
        inputs=[ht.keys, values, ht.active_slots],
        device=device,
    )
    wp.synchronize()

    # Verify data exists
    keys_np = ht.keys.numpy()
    non_empty = keys_np != 0xFFFFFFFFFFFFFFFF
    test.assertEqual(np.sum(non_empty), 50)

    # Clear
    ht.clear()
    values.zero_()  # Caller must clear their own values

    # Verify table is empty
    keys_np = ht.keys.numpy()
    test.assertTrue(np.all(keys_np == 0xFFFFFFFFFFFFFFFF))
    test.assertTrue(np.all(values.numpy() == 0))
    test.assertTrue(np.all(ht.active_slots.numpy() == 0))


def test_clear_active(test, device):
    """Test clearing only active entries (keys only, not values)."""
    values_per_key = 13

    @wp.kernel
    def insert_kernel(
        keys: wp.array(dtype=wp.uint64),
        values: wp.array(dtype=wp.uint64),
        active_slots: wp.array(dtype=wp.int32),
    ):
        tid = wp.tid()
        reduction_insert_slot(wp.uint64(tid + 1), 0, wp.uint64(tid * 10), keys, values, active_slots)

    ht = HashTable(capacity=256, device=device)
    values = wp.zeros(ht.capacity * values_per_key, dtype=wp.uint64, device=device)

    # Insert some data (sparse - only 20 entries in a 256-capacity table)
    wp.launch(
        insert_kernel,
        dim=20,
        inputs=[ht.keys, values, ht.active_slots],
        device=device,
    )
    wp.synchronize()

    # Verify data exists
    active_count = ht.active_slots.numpy()[ht.capacity]
    test.assertEqual(active_count, 20)

    # Clear active (keys only)
    ht.clear_active()

    # Verify keys are empty
    keys_np = ht.keys.numpy()
    non_empty = keys_np != 0xFFFFFFFFFFFFFFFF
    test.assertEqual(np.sum(non_empty), 0)

    # Active count should be 0
    active_count = ht.active_slots.numpy()[ht.capacity]
    test.assertEqual(active_count, 0)

    # Note: values are NOT cleared by clear_active - caller is responsible
    # This test only verifies the HashTable clears its own keys


def test_high_collision(test, device):
    """Test with many threads competing for same keys."""
    values_per_key = 13

    @wp.kernel
    def collision_kernel(
        keys: wp.array(dtype=wp.uint64),
        values: wp.array(dtype=wp.uint64),
        active_slots: wp.array(dtype=wp.int32),
    ):
        tid = wp.tid()
        # Only 10 unique keys, but 1000 threads
        key = wp.uint64(tid % 10)
        slot = tid % 13
        value = wp.uint64(tid)
        reduction_insert_slot(key, slot, value, keys, values, active_slots)

    ht = HashTable(capacity=64, device=device)
    values = wp.zeros(ht.capacity * values_per_key, dtype=wp.uint64, device=device)

    wp.launch(
        collision_kernel,
        dim=1000,
        inputs=[ht.keys, values, ht.active_slots],
        device=device,
    )
    wp.synchronize()

    # Should have exactly 10 unique keys
    keys_np = ht.keys.numpy()
    non_empty = keys_np != 0xFFFFFFFFFFFFFFFF
    test.assertEqual(np.sum(non_empty), 10)

    # Active count should be 10
    active_count = ht.active_slots.numpy()[ht.capacity]
    test.assertEqual(active_count, 10)


def test_early_exit_optimization(test, device):
    """Test that the early exit optimization works correctly.

    When a smaller value tries to update a slot that already has a larger value,
    it should skip the atomic operation but still return True.
    """
    values_per_key = 13

    @wp.kernel
    def insert_descending_kernel(
        keys: wp.array(dtype=wp.uint64),
        values: wp.array(dtype=wp.uint64),
        active_slots: wp.array(dtype=wp.int32),
    ):
        tid = wp.tid()
        # Insert values in descending order: 999, 998, 997, ...
        value = wp.uint64(999 - tid)
        reduction_insert_slot(wp.uint64(1), 0, value, keys, values, active_slots)

    ht = HashTable(capacity=64, device=device)
    values = wp.zeros(ht.capacity * values_per_key, dtype=wp.uint64, device=device)

    wp.launch(
        insert_descending_kernel,
        dim=1000,
        inputs=[ht.keys, values, ht.active_slots],
        device=device,
    )
    wp.synchronize()

    # Find the entry
    keys_np = ht.keys.numpy()
    values_np = values.numpy()

    entry_idx = np.where(keys_np == 1)[0]
    test.assertEqual(len(entry_idx), 1)
    idx = entry_idx[0]

    # The maximum value should be 999 (first insertion), slot-major layout
    test.assertEqual(values_np[0 * ht.capacity + idx], 999)


# =============================================================================
# Test registration
# =============================================================================

devices = get_test_devices()

# Register tests for all devices (CPU and CUDA)
add_function_test(TestHashTable, "test_basic_creation", test_basic_creation, devices=devices)
add_function_test(TestHashTable, "test_power_of_two_rounding", test_power_of_two_rounding, devices=devices)
add_function_test(TestHashTable, "test_insert_single_slot", test_insert_single_slot, devices=devices)
add_function_test(TestHashTable, "test_atomic_max_behavior", test_atomic_max_behavior, devices=devices)
add_function_test(TestHashTable, "test_multiple_keys", test_multiple_keys, devices=devices)
add_function_test(TestHashTable, "test_clear", test_clear, devices=devices)
add_function_test(TestHashTable, "test_clear_active", test_clear_active, devices=devices)
add_function_test(TestHashTable, "test_high_collision", test_high_collision, devices=devices)
add_function_test(TestHashTable, "test_early_exit_optimization", test_early_exit_optimization, devices=devices)


if __name__ == "__main__":
    wp.init()
    unittest.main(verbosity=2)
