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

"""Tests for the global contact reduction module."""

import unittest

import warp as wp

from newton._src.geometry.contact_data import ContactData
from newton._src.geometry.contact_reduction_global import (
    GlobalContactReducer,
    GlobalContactReducerData,
    create_export_reduced_contacts_kernel,
    export_and_reduce_contact,
    make_contact_key,
)
from newton._src.geometry.narrow_phase import ContactWriterData
from newton.tests.unittest_utils import add_function_test, get_test_devices

# =============================================================================
# Test helper functions
# =============================================================================


def get_contact_count(reducer: GlobalContactReducer) -> int:
    """Get the current number of stored contacts (test helper)."""
    return int(reducer.contact_count.numpy()[0])


def get_active_slot_count(reducer: GlobalContactReducer) -> int:
    """Get the number of active hashtable slots (test helper)."""
    return int(reducer.hashtable.active_slots.numpy()[reducer.hashtable.capacity])


def get_winning_contacts(reducer: GlobalContactReducer) -> list[int]:
    """Extract the winning contact IDs from the hashtable (test helper)."""
    values = reducer.ht_values.numpy()
    capacity = reducer.hashtable.capacity
    values_per_key = reducer.values_per_key

    contact_ids = set()

    # Iterate over active slots
    active_slots_np = reducer.hashtable.active_slots.numpy()
    count = active_slots_np[capacity]

    for i in range(count):
        entry_idx = active_slots_np[i]
        # Slot-major layout: slot * capacity + entry_idx
        for slot in range(values_per_key):
            val = values[slot * capacity + entry_idx]
            if val != 0:
                contact_id = val & 0xFFFFFFFF
                contact_ids.add(int(contact_id))

    return sorted(contact_ids)


# =============================================================================
# Test class
# =============================================================================


class TestGlobalContactReducer(unittest.TestCase):
    """Test cases for GlobalContactReducer."""

    pass


class TestKeyConstruction(unittest.TestCase):
    """Test the key construction function."""

    pass


# =============================================================================
# Test functions
# =============================================================================


def test_basic_contact_storage(test, device):
    """Test basic contact storage and retrieval."""
    reducer = GlobalContactReducer(capacity=100, device=device)

    @wp.kernel
    def store_contact_kernel(
        reducer_data: GlobalContactReducerData,
    ):
        _ = export_and_reduce_contact(
            shape_a=0,
            shape_b=1,
            position=wp.vec3(1.0, 2.0, 3.0),
            normal=wp.vec3(0.0, 1.0, 0.0),
            depth=-0.01,
            reducer_data=reducer_data,
            beta0=1000.0,
            beta1=0.001,
        )

    reducer_data = reducer.get_data_struct()
    wp.launch(
        store_contact_kernel,
        dim=1,
        inputs=[reducer_data],
        device=device,
    )

    test.assertEqual(get_contact_count(reducer), 1)

    # Check stored data
    pd = reducer.position_depth.numpy()[0]
    test.assertAlmostEqual(pd[0], 1.0)
    test.assertAlmostEqual(pd[1], 2.0)
    test.assertAlmostEqual(pd[2], 3.0)
    test.assertAlmostEqual(pd[3], -0.01, places=5)


def test_multiple_contacts_same_pair(test, device):
    """Test that multiple contacts for same shape pair get reduced."""
    reducer = GlobalContactReducer(capacity=100, device=device)

    @wp.kernel
    def store_multiple_contacts_kernel(
        reducer_data: GlobalContactReducerData,
    ):
        tid = wp.tid()
        # All contacts have same shape pair and similar normal (pointing up)
        # But different positions - reduction should pick spatial extremes
        x = float(tid) - 5.0  # Range from -5 to +4
        export_and_reduce_contact(
            shape_a=0,
            shape_b=1,
            position=wp.vec3(x, 0.0, 0.0),
            normal=wp.vec3(0.0, 1.0, 0.0),
            depth=-0.01,
            reducer_data=reducer_data,
            beta0=1000.0,
            beta1=0.001,
        )

    reducer_data = reducer.get_data_struct()
    wp.launch(
        store_multiple_contacts_kernel,
        dim=10,
        inputs=[reducer_data],
        device=device,
    )

    # All 10 contacts should be stored in buffer
    test.assertEqual(get_contact_count(reducer), 10)

    # But only a few should win hashtable slots (spatial extremes)
    winners = get_winning_contacts(reducer)
    # Should have fewer winners than total contacts due to reduction
    test.assertLess(len(winners), 10)
    test.assertGreater(len(winners), 0)


def test_different_shape_pairs(test, device):
    """Test that different shape pairs are tracked separately."""
    reducer = GlobalContactReducer(capacity=100, device=device)

    @wp.kernel
    def store_different_pairs_kernel(
        reducer_data: GlobalContactReducerData,
    ):
        tid = wp.tid()
        # Each thread represents a different shape pair
        export_and_reduce_contact(
            shape_a=tid,
            shape_b=tid + 100,
            position=wp.vec3(0.0, 0.0, 0.0),
            normal=wp.vec3(0.0, 1.0, 0.0),
            depth=-0.01,
            reducer_data=reducer_data,
            beta0=1000.0,
            beta1=0.001,
        )

    reducer_data = reducer.get_data_struct()
    wp.launch(
        store_different_pairs_kernel,
        dim=5,
        inputs=[reducer_data],
        device=device,
    )

    # All 5 contacts stored
    test.assertEqual(get_contact_count(reducer), 5)

    # Each shape pair should have its own winners
    winners = get_winning_contacts(reducer)
    # All 5 should win (different pairs, no competition)
    test.assertEqual(len(winners), 5)


def test_clear(test, device):
    """Test that clear resets the reducer."""
    reducer = GlobalContactReducer(capacity=100, device=device)

    @wp.kernel
    def store_one_contact_kernel(
        reducer_data: GlobalContactReducerData,
    ):
        export_and_reduce_contact(
            shape_a=0,
            shape_b=1,
            position=wp.vec3(0.0, 0.0, 0.0),
            normal=wp.vec3(0.0, 1.0, 0.0),
            depth=-0.01,
            reducer_data=reducer_data,
            beta0=1000.0,
            beta1=0.001,
        )

    reducer_data = reducer.get_data_struct()
    wp.launch(
        store_one_contact_kernel,
        dim=1,
        inputs=[reducer_data],
        device=device,
    )

    test.assertEqual(get_contact_count(reducer), 1)
    test.assertGreater(len(get_winning_contacts(reducer)), 0)

    reducer.clear()

    test.assertEqual(get_contact_count(reducer), 0)
    test.assertEqual(len(get_winning_contacts(reducer)), 0)


def test_stress_many_contacts(test, device):
    """Stress test with many contacts from many shape pairs."""
    reducer = GlobalContactReducer(capacity=10000, device=device)

    @wp.kernel
    def stress_kernel(
        reducer_data: GlobalContactReducerData,
    ):
        tid = wp.tid()
        # 100 shape pairs, 50 contacts each = 5000 total
        pair_id = tid // 50
        contact_in_pair = tid % 50

        shape_a = pair_id
        shape_b = pair_id + 1000

        # Vary positions within each pair
        x = float(contact_in_pair) - 25.0
        y = float(contact_in_pair % 10) - 5.0

        # Vary normals slightly
        nx = 0.1 * float(contact_in_pair % 3)
        ny = 1.0
        nz = 0.1 * float(contact_in_pair % 5)
        n_len = wp.sqrt(nx * nx + ny * ny + nz * nz)

        export_and_reduce_contact(
            shape_a=shape_a,
            shape_b=shape_b,
            position=wp.vec3(x, y, 0.0),
            normal=wp.vec3(nx / n_len, ny / n_len, nz / n_len),
            depth=-0.01,
            reducer_data=reducer_data,
            beta0=1000.0,
            beta1=0.001,
        )

    reducer_data = reducer.get_data_struct()
    wp.launch(
        stress_kernel,
        dim=5000,
        inputs=[reducer_data],
        device=device,
    )

    test.assertEqual(get_contact_count(reducer), 5000)

    winners = get_winning_contacts(reducer)
    # Should have significant reduction
    test.assertLess(len(winners), 5000)
    # But at least some winners per pair (100 pairs * some contacts)
    test.assertGreater(len(winners), 100)


def test_clear_active(test, device):
    """Test that clear_active only clears used slots."""
    reducer = GlobalContactReducer(capacity=100, device=device)

    @wp.kernel
    def store_contact_kernel(
        reducer_data: GlobalContactReducerData,
    ):
        export_and_reduce_contact(
            shape_a=0,
            shape_b=1,
            position=wp.vec3(1.0, 2.0, 3.0),
            normal=wp.vec3(0.0, 1.0, 0.0),
            depth=-0.01,
            reducer_data=reducer_data,
            beta0=1000.0,
            beta1=0.001,
        )

    reducer_data = reducer.get_data_struct()
    # Store one contact
    wp.launch(
        store_contact_kernel,
        dim=1,
        inputs=[reducer_data],
        device=device,
    )

    test.assertEqual(get_contact_count(reducer), 1)
    test.assertGreater(get_active_slot_count(reducer), 0)

    # Clear active and verify
    reducer.clear_active()
    test.assertEqual(get_contact_count(reducer), 0)
    test.assertEqual(get_active_slot_count(reducer), 0)

    # Store again should work
    wp.launch(
        store_contact_kernel,
        dim=1,
        inputs=[reducer_data],
        device=device,
    )

    test.assertEqual(get_contact_count(reducer), 1)


def test_export_reduced_contacts_kernel(test, device):
    """Test the export_reduced_contacts_kernel with a custom writer."""
    reducer = GlobalContactReducer(capacity=100, device=device)

    # Define a simple writer function
    @wp.func
    def test_writer(contact_data: ContactData, writer_data: ContactWriterData, output_index: int):
        idx = wp.atomic_add(writer_data.contact_count, 0, 1)
        if idx < writer_data.contact_max:
            writer_data.contact_pair[idx] = wp.vec2i(contact_data.shape_a, contact_data.shape_b)
            writer_data.contact_position[idx] = contact_data.contact_point_center
            writer_data.contact_normal[idx] = contact_data.contact_normal_a_to_b
            writer_data.contact_penetration[idx] = contact_data.contact_distance

    # Create the export kernel
    export_kernel = create_export_reduced_contacts_kernel(test_writer)

    # Store some contacts
    @wp.kernel
    def store_contacts_kernel(
        reducer_data: GlobalContactReducerData,
    ):
        tid = wp.tid()
        # Different shape pairs so all contacts win
        export_and_reduce_contact(
            shape_a=tid,
            shape_b=tid + 100,
            position=wp.vec3(float(tid), 0.0, 0.0),
            normal=wp.vec3(0.0, 1.0, 0.0),
            depth=-0.01,
            reducer_data=reducer_data,
            beta0=1000.0,
            beta1=0.001,
        )

    reducer_data = reducer.get_data_struct()
    wp.launch(
        store_contacts_kernel,
        dim=5,
        inputs=[reducer_data],
        device=device,
    )

    # Prepare output buffers
    max_output = 100
    contact_pair_out = wp.zeros(max_output, dtype=wp.vec2i, device=device)
    contact_position_out = wp.zeros(max_output, dtype=wp.vec3, device=device)
    contact_normal_out = wp.zeros(max_output, dtype=wp.vec3, device=device)
    contact_penetration_out = wp.zeros(max_output, dtype=float, device=device)
    contact_count_out = wp.zeros(1, dtype=int, device=device)
    contact_tangent_out = wp.zeros(0, dtype=wp.vec3, device=device)

    # Create dummy shape_data for thickness lookup
    num_shapes = 200
    shape_data = wp.zeros(num_shapes, dtype=wp.vec4, device=device)
    shape_data_np = shape_data.numpy()
    for i in range(num_shapes):
        shape_data_np[i] = [1.0, 1.0, 1.0, 0.01]  # scale xyz, thickness
    shape_data = wp.array(shape_data_np, dtype=wp.vec4, device=device)

    # Create per-shape contact margins
    shape_contact_margin = wp.full(num_shapes, 0.01, dtype=wp.float32, device=device)

    writer_data = ContactWriterData()
    writer_data.contact_max = max_output
    writer_data.contact_count = contact_count_out
    writer_data.contact_pair = contact_pair_out
    writer_data.contact_position = contact_position_out
    writer_data.contact_normal = contact_normal_out
    writer_data.contact_penetration = contact_penetration_out
    writer_data.contact_tangent = contact_tangent_out

    # Launch export kernel
    total_threads = 128  # Grid stride threads
    wp.launch(
        export_kernel,
        dim=total_threads,
        inputs=[
            reducer.hashtable.keys,
            reducer.ht_values,  # Values are now managed by GlobalContactReducer
            reducer.hashtable.active_slots,
            reducer.position_depth,
            reducer.normal,
            reducer.shape_pairs,
            shape_data,
            shape_contact_margin,
            writer_data,
            total_threads,
        ],
        device=device,
    )

    # Verify output - should have exported all unique winners
    num_exported = int(contact_count_out.numpy()[0])

    test.assertGreater(num_exported, 0)


def test_key_uniqueness(test, device):
    """Test that different inputs produce different keys."""

    @wp.kernel
    def compute_keys_kernel(
        keys_out: wp.array(dtype=wp.uint64),
    ):
        # Test various combinations
        keys_out[0] = make_contact_key(0, 1, 0)
        keys_out[1] = make_contact_key(1, 0, 0)  # Swapped shapes
        keys_out[2] = make_contact_key(0, 1, 1)  # Different bin
        keys_out[3] = make_contact_key(100, 200, 10)  # Larger values
        keys_out[4] = make_contact_key(0, 1, 0)  # Duplicate

    keys = wp.zeros(5, dtype=wp.uint64, device=device)
    wp.launch(compute_keys_kernel, dim=1, inputs=[keys], device=device)

    keys_np = keys.numpy()
    # First 4 keys should be unique
    test.assertEqual(len(set(keys_np[:4])), 4)
    # 5th key is duplicate of 1st
    test.assertEqual(keys_np[0], keys_np[4])


# =============================================================================
# Test registration
# =============================================================================

devices = get_test_devices()

# Register tests for all devices (CPU and CUDA)
add_function_test(TestGlobalContactReducer, "test_basic_contact_storage", test_basic_contact_storage, devices=devices)
add_function_test(
    TestGlobalContactReducer, "test_multiple_contacts_same_pair", test_multiple_contacts_same_pair, devices=devices
)
add_function_test(TestGlobalContactReducer, "test_different_shape_pairs", test_different_shape_pairs, devices=devices)
add_function_test(TestGlobalContactReducer, "test_clear", test_clear, devices=devices)
add_function_test(TestGlobalContactReducer, "test_stress_many_contacts", test_stress_many_contacts, devices=devices)
add_function_test(TestGlobalContactReducer, "test_clear_active", test_clear_active, devices=devices)
add_function_test(
    TestGlobalContactReducer,
    "test_export_reduced_contacts_kernel",
    test_export_reduced_contacts_kernel,
    devices=devices,
)
add_function_test(TestKeyConstruction, "test_key_uniqueness", test_key_uniqueness, devices=devices)


if __name__ == "__main__":
    wp.init()
    unittest.main(verbosity=2)
