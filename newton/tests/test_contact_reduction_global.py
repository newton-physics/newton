# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the global contact reduction module."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.contact_data import ContactData, make_contact_sort_key
from newton._src.geometry.contact_reduction import float_flip
from newton._src.geometry.contact_reduction_global import (
    SCORE_SHIFT,
    GlobalContactReducer,
    GlobalContactReducerData,
    _make_contact_value_det,
    _make_preprune_probe_det,
    _unpack_contact_id_det,
    create_export_reduced_contacts_kernel,
    decode_oct,
    encode_oct,
    export_and_reduce_contact,
    export_and_reduce_contact_centered,
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

    # Create dummy arrays for the required parameters
    num_shapes = 200
    shape_transform = wp.zeros(num_shapes, dtype=wp.transform, device=device)
    shape_collision_aabb_lower = wp.zeros(num_shapes, dtype=wp.vec3, device=device)
    shape_collision_aabb_upper = wp.ones(num_shapes, dtype=wp.vec3, device=device)
    shape_voxel_resolution = wp.full(num_shapes, wp.vec3i(4, 4, 4), dtype=wp.vec3i, device=device)

    @wp.kernel
    def store_contact_kernel(
        reducer_data: GlobalContactReducerData,
        xform: wp.array[wp.transform],
        aabb_lower: wp.array[wp.vec3],
        aabb_upper: wp.array[wp.vec3],
        voxel_res: wp.array[wp.vec3i],
    ):
        _ = export_and_reduce_contact(
            shape_a=0,
            shape_b=1,
            position=wp.vec3(1.0, 2.0, 3.0),
            normal=wp.vec3(0.0, 1.0, 0.0),
            depth=-0.01,
            fingerprint=0,
            reducer_data=reducer_data,
            beta=0.001,
            shape_transform=xform,
            shape_collision_aabb_lower=aabb_lower,
            shape_collision_aabb_upper=aabb_upper,
            shape_voxel_resolution=voxel_res,
        )

    reducer_data = reducer.get_data_struct()
    wp.launch(
        store_contact_kernel,
        dim=1,
        inputs=[
            reducer_data,
            shape_transform,
            shape_collision_aabb_lower,
            shape_collision_aabb_upper,
            shape_voxel_resolution,
        ],
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

    # Create dummy arrays for the required parameters
    num_shapes = 200
    shape_transform = wp.zeros(num_shapes, dtype=wp.transform, device=device)
    shape_collision_aabb_lower = wp.zeros(num_shapes, dtype=wp.vec3, device=device)
    shape_collision_aabb_upper = wp.ones(num_shapes, dtype=wp.vec3, device=device)
    shape_voxel_resolution = wp.full(num_shapes, wp.vec3i(4, 4, 4), dtype=wp.vec3i, device=device)

    @wp.kernel
    def store_multiple_contacts_kernel(
        reducer_data: GlobalContactReducerData,
        xform: wp.array[wp.transform],
        aabb_lower: wp.array[wp.vec3],
        aabb_upper: wp.array[wp.vec3],
        voxel_res: wp.array[wp.vec3i],
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
            fingerprint=0,
            reducer_data=reducer_data,
            beta=0.001,
            shape_transform=xform,
            shape_collision_aabb_lower=aabb_lower,
            shape_collision_aabb_upper=aabb_upper,
            shape_voxel_resolution=voxel_res,
        )

    reducer_data = reducer.get_data_struct()
    wp.launch(
        store_multiple_contacts_kernel,
        dim=10,
        inputs=[
            reducer_data,
            shape_transform,
            shape_collision_aabb_lower,
            shape_collision_aabb_upper,
            shape_voxel_resolution,
        ],
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

    # Create dummy arrays for the required parameters
    num_shapes = 200
    shape_transform = wp.zeros(num_shapes, dtype=wp.transform, device=device)
    shape_collision_aabb_lower = wp.zeros(num_shapes, dtype=wp.vec3, device=device)
    shape_collision_aabb_upper = wp.ones(num_shapes, dtype=wp.vec3, device=device)
    shape_voxel_resolution = wp.full(num_shapes, wp.vec3i(4, 4, 4), dtype=wp.vec3i, device=device)

    @wp.kernel
    def store_different_pairs_kernel(
        reducer_data: GlobalContactReducerData,
        xform: wp.array[wp.transform],
        aabb_lower: wp.array[wp.vec3],
        aabb_upper: wp.array[wp.vec3],
        voxel_res: wp.array[wp.vec3i],
    ):
        tid = wp.tid()
        # Each thread represents a different shape pair
        export_and_reduce_contact(
            shape_a=tid,
            shape_b=tid + 100,
            position=wp.vec3(0.0, 0.0, 0.0),
            normal=wp.vec3(0.0, 1.0, 0.0),
            depth=-0.01,
            fingerprint=0,
            reducer_data=reducer_data,
            beta=0.001,
            shape_transform=xform,
            shape_collision_aabb_lower=aabb_lower,
            shape_collision_aabb_upper=aabb_upper,
            shape_voxel_resolution=voxel_res,
        )

    reducer_data = reducer.get_data_struct()
    wp.launch(
        store_different_pairs_kernel,
        dim=5,
        inputs=[
            reducer_data,
            shape_transform,
            shape_collision_aabb_lower,
            shape_collision_aabb_upper,
            shape_voxel_resolution,
        ],
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

    # Create dummy arrays for the required parameters
    num_shapes = 200
    shape_transform = wp.zeros(num_shapes, dtype=wp.transform, device=device)
    shape_collision_aabb_lower = wp.zeros(num_shapes, dtype=wp.vec3, device=device)
    shape_collision_aabb_upper = wp.ones(num_shapes, dtype=wp.vec3, device=device)
    shape_voxel_resolution = wp.full(num_shapes, wp.vec3i(4, 4, 4), dtype=wp.vec3i, device=device)

    @wp.kernel
    def store_one_contact_kernel(
        reducer_data: GlobalContactReducerData,
        xform: wp.array[wp.transform],
        aabb_lower: wp.array[wp.vec3],
        aabb_upper: wp.array[wp.vec3],
        voxel_res: wp.array[wp.vec3i],
    ):
        export_and_reduce_contact(
            shape_a=0,
            shape_b=1,
            position=wp.vec3(0.0, 0.0, 0.0),
            normal=wp.vec3(0.0, 1.0, 0.0),
            depth=-0.01,
            fingerprint=0,
            reducer_data=reducer_data,
            beta=0.001,
            shape_transform=xform,
            shape_collision_aabb_lower=aabb_lower,
            shape_collision_aabb_upper=aabb_upper,
            shape_voxel_resolution=voxel_res,
        )

    reducer_data = reducer.get_data_struct()
    wp.launch(
        store_one_contact_kernel,
        dim=1,
        inputs=[
            reducer_data,
            shape_transform,
            shape_collision_aabb_lower,
            shape_collision_aabb_upper,
            shape_voxel_resolution,
        ],
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

    # Create dummy arrays for the required parameters
    num_shapes = 2000
    shape_transform = wp.zeros(num_shapes, dtype=wp.transform, device=device)
    shape_collision_aabb_lower = wp.zeros(num_shapes, dtype=wp.vec3, device=device)
    shape_collision_aabb_upper = wp.ones(num_shapes, dtype=wp.vec3, device=device)
    shape_voxel_resolution = wp.full(num_shapes, wp.vec3i(4, 4, 4), dtype=wp.vec3i, device=device)

    @wp.kernel
    def stress_kernel(
        reducer_data: GlobalContactReducerData,
        xform: wp.array[wp.transform],
        aabb_lower: wp.array[wp.vec3],
        aabb_upper: wp.array[wp.vec3],
        voxel_res: wp.array[wp.vec3i],
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
            fingerprint=0,
            reducer_data=reducer_data,
            beta=0.001,
            shape_transform=xform,
            shape_collision_aabb_lower=aabb_lower,
            shape_collision_aabb_upper=aabb_upper,
            shape_voxel_resolution=voxel_res,
        )

    reducer_data = reducer.get_data_struct()
    wp.launch(
        stress_kernel,
        dim=5000,
        inputs=[
            reducer_data,
            shape_transform,
            shape_collision_aabb_lower,
            shape_collision_aabb_upper,
            shape_voxel_resolution,
        ],
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

    # Create dummy arrays for the required parameters
    num_shapes = 200
    shape_transform = wp.zeros(num_shapes, dtype=wp.transform, device=device)
    shape_collision_aabb_lower = wp.zeros(num_shapes, dtype=wp.vec3, device=device)
    shape_collision_aabb_upper = wp.ones(num_shapes, dtype=wp.vec3, device=device)
    shape_voxel_resolution = wp.full(num_shapes, wp.vec3i(4, 4, 4), dtype=wp.vec3i, device=device)

    @wp.kernel
    def store_contact_kernel(
        reducer_data: GlobalContactReducerData,
        xform: wp.array[wp.transform],
        aabb_lower: wp.array[wp.vec3],
        aabb_upper: wp.array[wp.vec3],
        voxel_res: wp.array[wp.vec3i],
    ):
        export_and_reduce_contact(
            shape_a=0,
            shape_b=1,
            position=wp.vec3(1.0, 2.0, 3.0),
            normal=wp.vec3(0.0, 1.0, 0.0),
            depth=-0.01,
            fingerprint=0,
            reducer_data=reducer_data,
            beta=0.001,
            shape_transform=xform,
            shape_collision_aabb_lower=aabb_lower,
            shape_collision_aabb_upper=aabb_upper,
            shape_voxel_resolution=voxel_res,
        )

    reducer_data = reducer.get_data_struct()
    # Store one contact
    wp.launch(
        store_contact_kernel,
        dim=1,
        inputs=[
            reducer_data,
            shape_transform,
            shape_collision_aabb_lower,
            shape_collision_aabb_upper,
            shape_voxel_resolution,
        ],
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
        inputs=[
            reducer_data,
            shape_transform,
            shape_collision_aabb_lower,
            shape_collision_aabb_upper,
            shape_voxel_resolution,
        ],
        device=device,
    )

    test.assertEqual(get_contact_count(reducer), 1)


def test_export_reduced_contacts_kernel(test, device):
    """Test the export_reduced_contacts_kernel with a custom writer."""
    reducer = GlobalContactReducer(capacity=100, device=device)

    # Create dummy arrays for the required parameters
    num_shapes = 200
    shape_transform = wp.zeros(num_shapes, dtype=wp.transform, device=device)
    shape_collision_aabb_lower = wp.zeros(num_shapes, dtype=wp.vec3, device=device)
    shape_collision_aabb_upper = wp.ones(num_shapes, dtype=wp.vec3, device=device)
    shape_voxel_resolution = wp.full(num_shapes, wp.vec3i(4, 4, 4), dtype=wp.vec3i, device=device)

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
        xform: wp.array[wp.transform],
        aabb_lower: wp.array[wp.vec3],
        aabb_upper: wp.array[wp.vec3],
        voxel_res: wp.array[wp.vec3i],
    ):
        tid = wp.tid()
        # Different shape pairs so all contacts win
        export_and_reduce_contact(
            shape_a=tid,
            shape_b=tid + 100,
            position=wp.vec3(float(tid), 0.0, 0.0),
            normal=wp.vec3(0.0, 1.0, 0.0),
            depth=-0.01,
            fingerprint=0,
            reducer_data=reducer_data,
            beta=0.001,
            shape_transform=xform,
            shape_collision_aabb_lower=aabb_lower,
            shape_collision_aabb_upper=aabb_upper,
            shape_voxel_resolution=voxel_res,
        )

    reducer_data = reducer.get_data_struct()
    wp.launch(
        store_contacts_kernel,
        dim=5,
        inputs=[
            reducer_data,
            shape_transform,
            shape_collision_aabb_lower,
            shape_collision_aabb_upper,
            shape_voxel_resolution,
        ],
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
    shape_types = wp.zeros(num_shapes, dtype=int, device=device)  # Shape types (0 = PLANE, doesn't affect test)
    shape_data = wp.zeros(num_shapes, dtype=wp.vec4, device=device)
    shape_data_np = shape_data.numpy()
    for i in range(num_shapes):
        shape_data_np[i] = [1.0, 1.0, 1.0, 0.01]  # scale xyz, thickness
    shape_data = wp.array(shape_data_np, dtype=wp.vec4, device=device)

    # Create per-shape contact margins
    shape_gap = wp.full(num_shapes, 0.01, dtype=wp.float32, device=device)

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
    reducer.exported_flags.zero_()
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
            reducer.contact_fingerprints,
            reducer.exported_flags,
            shape_types,
            shape_data,
            shape_gap,
            writer_data,
            total_threads,
            0,  # deterministic=0 (fast packing)
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
        keys_out: wp.array[wp.uint64],
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


def test_oct_encode_decode_roundtrip(test, device):
    """Validate octahedral normal encode/decode round-trip accuracy.

    Args:
        test: Unittest-style assertion helper.
        device: Warp device under test.
    """

    @wp.kernel
    def roundtrip_error_kernel(normals: wp.array[wp.vec3], errors: wp.array[wp.float32]):
        tid = wp.tid()
        n = wp.normalize(normals[tid])
        decoded = decode_oct(encode_oct(n))
        errors[tid] = wp.length(decoded - n)

    normals_np = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 0.5],
            [0.2, -0.7, -0.68],
            [-0.35, -0.12, -0.93],
            [0.0001, 1.0, -0.0002],
            [-0.9, 0.3, -0.3],
        ],
        dtype=np.float32,
    )

    normals = wp.array(normals_np, dtype=wp.vec3, device=device)
    errors = wp.empty(normals.shape[0], dtype=wp.float32, device=device)
    wp.launch(roundtrip_error_kernel, dim=normals.shape[0], inputs=[normals, errors], device=device)

    max_error = float(np.max(errors.numpy()))
    test.assertLess(max_error, 1.0e-5, f"Expected oct encode/decode max error < 1e-5, got {max_error:.3e}")


def test_centered_basic_storage_and_reduction(test, device):
    """Test that export_and_reduce_contact_centered stores and reduces contacts correctly."""
    reducer = GlobalContactReducer(capacity=200, device=device)

    @wp.kernel
    def store_centered_contacts_kernel(reducer_data: GlobalContactReducerData):
        tid = wp.tid()
        x = float(tid) - 10.0
        position = wp.vec3(x, 0.0, 0.0)
        normal = wp.vec3(0.0, 1.0, 0.0)
        depth = -0.01

        centered_position = position
        X_ws_shape = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        aabb_lower = wp.vec3(-15.0, -5.0, -5.0)
        aabb_upper = wp.vec3(15.0, 5.0, 5.0)
        voxel_res = wp.vec3i(4, 4, 4)

        export_and_reduce_contact_centered(
            shape_a=0,
            shape_b=1,
            position=position,
            normal=normal,
            depth=depth,
            fingerprint=0,
            centered_position=centered_position,
            X_ws_voxel_shape=X_ws_shape,
            aabb_lower_voxel=aabb_lower,
            aabb_upper_voxel=aabb_upper,
            voxel_res=voxel_res,
            reducer_data=reducer_data,
        )

    reducer_data = reducer.get_data_struct()
    wp.launch(store_centered_contacts_kernel, dim=20, inputs=[reducer_data], device=device)

    contact_count = get_contact_count(reducer)
    test.assertGreater(contact_count, 0, "At least one contact should be stored")

    winners = get_winning_contacts(reducer)
    test.assertGreater(len(winners), 0, "At least one contact should win a slot")
    test.assertLess(len(winners), 20, "Reduction should produce fewer winners than inputs")


def test_centered_different_pairs_independent(test, device):
    """Test that different shape pairs are tracked independently in centered reduction."""
    reducer = GlobalContactReducer(capacity=200, device=device)

    @wp.kernel
    def store_different_pairs_centered_kernel(reducer_data: GlobalContactReducerData):
        tid = wp.tid()
        X_ws_shape = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        aabb_lower = wp.vec3(-5.0, -5.0, -5.0)
        aabb_upper = wp.vec3(5.0, 5.0, 5.0)
        voxel_res = wp.vec3i(4, 4, 4)

        export_and_reduce_contact_centered(
            shape_a=tid,
            shape_b=tid + 100,
            position=wp.vec3(0.0, 0.0, 0.0),
            normal=wp.vec3(0.0, 1.0, 0.0),
            depth=-0.01,
            fingerprint=0,
            centered_position=wp.vec3(0.0, 0.0, 0.0),
            X_ws_voxel_shape=X_ws_shape,
            aabb_lower_voxel=aabb_lower,
            aabb_upper_voxel=aabb_upper,
            voxel_res=voxel_res,
            reducer_data=reducer_data,
        )

    reducer_data = reducer.get_data_struct()
    wp.launch(store_different_pairs_centered_kernel, dim=5, inputs=[reducer_data], device=device)

    test.assertEqual(get_contact_count(reducer), 5)
    winners = get_winning_contacts(reducer)
    test.assertEqual(len(winners), 5, "Each unique shape pair should have its own winner")


def test_centered_deepest_wins_max_depth_slot(test, device):
    """Test that the deepest contact always wins the max-depth slot."""
    reducer = GlobalContactReducer(capacity=200, device=device)

    @wp.kernel
    def store_varying_depth_kernel(reducer_data: GlobalContactReducerData):
        tid = wp.tid()
        depth = -0.001 * float(tid + 1)
        X_ws_shape = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        aabb_lower = wp.vec3(-5.0, -5.0, -5.0)
        aabb_upper = wp.vec3(5.0, 5.0, 5.0)

        export_and_reduce_contact_centered(
            shape_a=0,
            shape_b=1,
            position=wp.vec3(0.0, 0.0, 0.0),
            normal=wp.vec3(0.0, 1.0, 0.0),
            depth=depth,
            fingerprint=0,
            centered_position=wp.vec3(0.0, 0.0, 0.0),
            X_ws_voxel_shape=X_ws_shape,
            aabb_lower_voxel=aabb_lower,
            aabb_upper_voxel=aabb_upper,
            voxel_res=wp.vec3i(4, 4, 4),
            reducer_data=reducer_data,
        )

    reducer_data = reducer.get_data_struct()
    wp.launch(store_varying_depth_kernel, dim=10, inputs=[reducer_data], device=device)

    deepest_depth = -0.01  # tid=9 → depth = -0.001 * 10 = -0.01

    winners = get_winning_contacts(reducer)
    test.assertGreater(len(winners), 0)

    best_depth = 0.0
    for cid in winners:
        pd = reducer.position_depth.numpy()[cid]
        if pd[3] < best_depth:
            best_depth = pd[3]
    test.assertAlmostEqual(best_depth, deepest_depth, places=5, msg="Deepest contact should be among winners")


def test_centered_pre_pruning_reduces_buffer_usage(test, device):
    """Verify pre-pruning skips dominated contacts, reducing buffer allocations.

    First stores strong contacts at spatial extremes, then stores many weak
    dominated contacts in small sequential batches (with synchronize between
    them so earlier writes are visible to later pre-prune reads).
    """
    reducer = GlobalContactReducer(capacity=1000, device=device)
    reducer_data = reducer.get_data_struct()

    @wp.kernel
    def store_extreme_contacts_kernel(reducer_data: GlobalContactReducerData):
        tid = wp.tid()
        positions = wp.vec3(0.0, 0.0, 0.0)
        if tid == 0:
            positions = wp.vec3(-10.0, 0.0, 0.0)
        elif tid == 1:
            positions = wp.vec3(10.0, 0.0, 0.0)
        elif tid == 2:
            positions = wp.vec3(0.0, 0.0, -10.0)
        else:
            positions = wp.vec3(0.0, 0.0, 10.0)

        X_ws = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        export_and_reduce_contact_centered(
            shape_a=0,
            shape_b=1,
            position=positions,
            normal=wp.vec3(0.0, 1.0, 0.0),
            depth=-0.5,
            fingerprint=0,
            centered_position=positions,
            X_ws_voxel_shape=X_ws,
            aabb_lower_voxel=wp.vec3(-15.0, -5.0, -15.0),
            aabb_upper_voxel=wp.vec3(15.0, 5.0, 15.0),
            voxel_res=wp.vec3i(4, 4, 4),
            reducer_data=reducer_data,
        )

    wp.launch(store_extreme_contacts_kernel, dim=4, inputs=[reducer_data], device=device)
    count_after_extremes = get_contact_count(reducer)
    test.assertEqual(count_after_extremes, 4)

    @wp.kernel
    def store_one_dominated_contact_kernel(
        reducer_data: GlobalContactReducerData,
        x_offset: float,
    ):
        X_ws = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        export_and_reduce_contact_centered(
            shape_a=0,
            shape_b=1,
            position=wp.vec3(x_offset, 0.0, 0.0),
            normal=wp.vec3(0.0, 1.0, 0.0),
            depth=-0.001,
            fingerprint=0,
            centered_position=wp.vec3(x_offset, 0.0, 0.0),
            X_ws_voxel_shape=X_ws,
            aabb_lower_voxel=wp.vec3(-15.0, -5.0, -15.0),
            aabb_upper_voxel=wp.vec3(15.0, 5.0, 15.0),
            voxel_res=wp.vec3i(4, 4, 4),
            reducer_data=reducer_data,
        )

    # Launch dominated contacts one at a time with synchronize so pre-prune
    # reads see prior writes (avoids GPU warp-level race conditions).
    total_dominated = 20
    for i in range(total_dominated):
        x = float(i) * 0.1 - 1.0
        wp.launch(
            store_one_dominated_contact_kernel,
            dim=1,
            inputs=[reducer_data, x],
            device=device,
        )
        wp.synchronize()

    count_after_dominated = get_contact_count(reducer)
    new_allocations = count_after_dominated - count_after_extremes

    # Sequential dominated contacts should mostly be pruned: shallower depth
    # (-0.001 vs -0.5) and interior positions ([-1, 1] vs ±10).
    test.assertLess(
        new_allocations,
        total_dominated,
        f"Pre-pruning should skip some dominated contacts, but {new_allocations}/{total_dominated} were allocated",
    )


# =============================================================================
# Float-flip precision tests
# =============================================================================


@wp.func_native("""
uint32_t mask = ((u >> 31) - 1) | 0x80000000;
uint32_t i = u ^ mask;
return reinterpret_cast<float&>(i);
""")
def ifloat_flip(u: wp.uint32) -> float: ...


@wp.kernel(enable_backward=False)
def float_flip_roundtrip_kernel(
    values_in: wp.array[float],
    values_out: wp.array[float],
    truncated_out: wp.array[wp.uint32],
    score_shift: int,
):
    """Apply float_flip, clear low bits, then ifloat_flip to measure precision loss."""
    tid = wp.tid()
    f = values_in[tid]
    flipped = float_flip(f)
    truncated = (flipped >> wp.uint32(score_shift)) << wp.uint32(score_shift)
    reconstructed = ifloat_flip(truncated)
    values_out[tid] = reconstructed
    truncated_out[tid] = truncated


class TestFloatFlipPrecision(unittest.TestCase):
    """Test float_flip / ifloat_flip roundtrip precision with the SCORE_SHIFT used in contact reduction."""

    pass


def test_float_flip_roundtrip_precision(test, device):
    """Validate that ifloat_flip((float_flip(f) >> SCORE_SHIFT) << SCORE_SHIFT) preserves
    order and maintains relative precision ~2^-(23-SCORE_SHIFT)."""
    score_shift = SCORE_SHIFT
    mantissa_bits_kept = 23 - score_shift
    max_relative_error = 2.0 ** (-mantissa_bits_kept)  # ~1.22e-4

    # --- Build test values spanning actual score ranges ---
    # Depth scores: -depth in [0.0001, 0.1]
    depth_scores = np.linspace(0.0001, 0.1, 200, dtype=np.float32)
    # Spatial scores: dot(pos_2d, dir_2d) in [-10, +10]
    spatial_scores = np.linspace(-10.0, 10.0, 600, dtype=np.float32)
    # Dense sweep around 1.0 to probe distinguishability
    dense_sweep = np.linspace(0.999, 1.001, 100, dtype=np.float32)
    # Edge cases
    edge_cases = np.array(
        [0.0, -0.0, 1e-6, -1e-6, 1e-4, -1e-4, 1.0, -1.0, 100.0, -100.0],
        dtype=np.float32,
    )

    all_values = np.concatenate([depth_scores, spatial_scores, dense_sweep, edge_cases])
    n = len(all_values)

    values_in = wp.array(all_values, dtype=float, device=device)
    values_out = wp.zeros(n, dtype=float, device=device)
    truncated_out = wp.zeros(n, dtype=wp.uint32, device=device)

    wp.launch(
        float_flip_roundtrip_kernel,
        dim=n,
        inputs=[values_in, values_out, truncated_out, score_shift],
        device=device,
    )

    orig = all_values
    recon = values_out.numpy()
    trunc = truncated_out.numpy()

    # --- 1. Roundtrip for +0.0 (positive zero roundtrips exactly;
    #     -0.0 maps to a tiny negative via float_flip, which is fine) ---
    pos_zero_mask = (orig == 0.0) & ~np.signbit(orig)
    if np.any(pos_zero_mask):
        test.assertTrue(
            np.all(recon[pos_zero_mask] == 0.0),
            "Positive zero must roundtrip exactly",
        )

    # --- 2. Relative error bound ---
    nonzero_mask = np.abs(orig) > 1e-30
    rel_err = np.abs(recon[nonzero_mask] - orig[nonzero_mask]) / np.abs(orig[nonzero_mask])
    worst_rel = float(np.max(rel_err))
    test.assertLessEqual(
        worst_rel,
        max_relative_error,
        f"Worst relative error {worst_rel:.6e} exceeds 2^-{mantissa_bits_kept} = {max_relative_error:.6e}",
    )

    # --- 3. Order preservation (monotonicity) ---
    sorted_idx = np.argsort(orig)
    sorted_orig = orig[sorted_idx]
    sorted_trunc = trunc[sorted_idx]
    violations = 0
    for i in range(len(sorted_orig) - 1):
        if sorted_orig[i] < sorted_orig[i + 1] and sorted_trunc[i] > sorted_trunc[i + 1]:
            violations += 1
    test.assertEqual(violations, 0, f"Order violations: {violations}")

    # --- 4. Distinguishability at the precision floor ---
    # Two values separated by more than 2^-(mantissa_bits_kept) relative
    # to their magnitude must produce different truncated values.
    base = np.float32(1.0)
    eps_distinguishable = np.float32(2.0 ** (-mantissa_bits_kept + 1))  # 2x the LSB
    pair = np.array([base, base + eps_distinguishable], dtype=np.float32)
    pair_in = wp.array(pair, dtype=float, device=device)
    pair_out = wp.zeros(2, dtype=float, device=device)
    pair_trunc = wp.zeros(2, dtype=wp.uint32, device=device)
    wp.launch(
        float_flip_roundtrip_kernel,
        dim=2,
        inputs=[pair_in, pair_out, pair_trunc, score_shift],
        device=device,
    )
    t = pair_trunc.numpy()
    test.assertNotEqual(
        int(t[0]),
        int(t[1]),
        f"Values {pair[0]} and {pair[1]} (eps={eps_distinguishable:.6e}) must produce different truncated values",
    )

    # --- 5. Sign correctness: all negatives map below all positives ---
    neg_mask = orig < 0
    pos_mask = orig > 0
    if np.any(neg_mask) and np.any(pos_mask):
        max_neg_trunc = int(np.max(trunc[neg_mask]))
        min_pos_trunc = int(np.min(trunc[pos_mask]))
        test.assertLess(
            max_neg_trunc,
            min_pos_trunc,
            "All negative truncated values must be less than all positive truncated values",
        )


def test_float_flip_exact_inverse(test, device):
    """Verify ifloat_flip(float_flip(f)) == f exactly (no truncation)."""
    values = np.array(
        [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 3.14159, -2.71828, 1e-6, -1e-6, 1e6, -1e6, 42.0],
        dtype=np.float32,
    )
    n = len(values)
    values_in = wp.array(values, dtype=float, device=device)
    values_out = wp.zeros(n, dtype=float, device=device)
    truncated_out = wp.zeros(n, dtype=wp.uint32, device=device)

    # Use score_shift=0 for exact roundtrip (no bits cleared)
    wp.launch(
        float_flip_roundtrip_kernel,
        dim=n,
        inputs=[values_in, values_out, truncated_out, 0],
        device=device,
    )

    orig = values
    recon = values_out.numpy()
    for i in range(n):
        test.assertEqual(
            orig[i],
            recon[i],
            f"Exact roundtrip failed for {orig[i]}: got {recon[i]}",
        )


# =============================================================================
# make_contact_sort_key tests
# =============================================================================


class TestMakeContactSortKey(unittest.TestCase):
    """Test make_contact_sort_key bit layout and ordering."""

    pass


@wp.kernel(enable_backward=False)
def _sort_key_kernel(
    shape_a: wp.array[int],
    shape_b: wp.array[int],
    sub_key: wp.array[int],
    keys_out: wp.array[wp.int64],
):
    tid = wp.tid()
    keys_out[tid] = make_contact_sort_key(shape_a[tid], shape_b[tid], sub_key[tid])


def test_sort_key_bit_layout(test, device):
    """Verify that make_contact_sort_key produces correct lexicographic ordering."""
    # Pairs ordered lexicographically: (shape_a, shape_b, sub_key)
    # Each successive entry should produce a strictly larger key.
    sa = wp.array([0, 0, 0, 1, 1], dtype=int, device=device)
    sb = wp.array([0, 0, 1, 0, 0], dtype=int, device=device)
    sk = wp.array([0, 1, 0, 0, 1], dtype=int, device=device)
    keys = wp.zeros(5, dtype=wp.int64, device=device)
    wp.launch(_sort_key_kernel, dim=5, inputs=[sa, sb, sk, keys], device=device)

    keys_np = keys.numpy()
    for i in range(len(keys_np) - 1):
        test.assertLess(
            keys_np[i],
            keys_np[i + 1],
            f"Key[{i}]={keys_np[i]} should be < Key[{i + 1}]={keys_np[i + 1]}",
        )


def test_sort_key_overflow_masking(test, device):
    """Verify that values exceeding bit widths are masked (not corrupting other fields)."""
    # shape_a with 21 bits set (exceeds 20-bit field) — high bit should be masked
    large_a = (1 << 21) | 5  # bit 20 set + low bits
    sa = wp.array([large_a, 5], dtype=int, device=device)
    sb = wp.array([0, 0], dtype=int, device=device)
    sk = wp.array([0, 0], dtype=int, device=device)
    keys = wp.zeros(2, dtype=wp.int64, device=device)
    wp.launch(_sort_key_kernel, dim=2, inputs=[sa, sb, sk, keys], device=device)

    keys_np = keys.numpy()
    # After masking to 20 bits, large_a & 0xFFFFF == 5, so both keys should be equal
    test.assertEqual(keys_np[0], keys_np[1], "Overflow bits should be masked away")


# =============================================================================
# Deterministic packing function tests
# =============================================================================


class TestDeterministicPacking(unittest.TestCase):
    """Test deterministic contact value packing and unpacking."""

    pass


@wp.kernel(enable_backward=False)
def _det_pack_unpack_kernel(
    scores: wp.array[float],
    fingerprints: wp.array[int],
    contact_ids: wp.array[int],
    packed_out: wp.array[wp.uint64],
    unpacked_ids_out: wp.array[int],
):
    tid = wp.tid()
    packed = _make_contact_value_det(scores[tid], fingerprints[tid], contact_ids[tid])
    packed_out[tid] = packed
    unpacked_ids_out[tid] = _unpack_contact_id_det(packed)


def test_det_pack_unpack_roundtrip(test, device):
    """Verify contact_id survives pack/unpack roundtrip in deterministic mode."""
    n = 5
    scores = wp.array([0.1, 0.5, -0.01, 1.0, 0.0], dtype=float, device=device)
    fps = wp.array([0, 100, 999999, 42, 0], dtype=int, device=device)
    ids = wp.array([0, 1, 1048575, 500, 12345], dtype=int, device=device)
    packed = wp.zeros(n, dtype=wp.uint64, device=device)
    unpacked = wp.zeros(n, dtype=int, device=device)

    wp.launch(_det_pack_unpack_kernel, dim=n, inputs=[scores, fps, ids, packed, unpacked], device=device)

    ids_np = ids.numpy()
    unpacked_np = unpacked.numpy()
    for i in range(n):
        expected = ids_np[i] & ((1 << 20) - 1)
        test.assertEqual(
            unpacked_np[i],
            expected,
            f"Roundtrip failed for contact_id={ids_np[i]}: got {unpacked_np[i]}, expected {expected}",
        )


def test_det_packing_score_dominates(test, device):
    """Verify that higher score always produces a larger packed value."""
    n = 2
    # Same fingerprint and contact_id, different scores
    scores = wp.array([0.1, 0.5], dtype=float, device=device)
    fps = wp.array([42, 42], dtype=int, device=device)
    ids = wp.array([10, 10], dtype=int, device=device)
    packed = wp.zeros(n, dtype=wp.uint64, device=device)
    unpacked = wp.zeros(n, dtype=int, device=device)

    wp.launch(_det_pack_unpack_kernel, dim=n, inputs=[scores, fps, ids, packed, unpacked], device=device)

    packed_np = packed.numpy()
    test.assertLess(packed_np[0], packed_np[1], "Higher score should produce larger packed value")


def test_det_packing_fingerprint_breaks_tie(test, device):
    """Verify that fingerprint breaks ties when scores are equal."""
    n = 2
    # Same score and contact_id, different fingerprints
    scores = wp.array([0.5, 0.5], dtype=float, device=device)
    fps = wp.array([10, 20], dtype=int, device=device)
    ids = wp.array([5, 5], dtype=int, device=device)
    packed = wp.zeros(n, dtype=wp.uint64, device=device)
    unpacked = wp.zeros(n, dtype=int, device=device)

    wp.launch(_det_pack_unpack_kernel, dim=n, inputs=[scores, fps, ids, packed, unpacked], device=device)

    packed_np = packed.numpy()
    test.assertNotEqual(packed_np[0], packed_np[1], "Different fingerprints should produce different packed values")
    test.assertLess(packed_np[0], packed_np[1], "Higher fingerprint should produce larger packed value")


@wp.kernel(enable_backward=False)
def _det_preprune_probe_kernel(
    scores: wp.array[float],
    fingerprints: wp.array[int],
    probes_out: wp.array[wp.uint64],
):
    tid = wp.tid()
    probes_out[tid] = _make_preprune_probe_det(scores[tid], fingerprints[tid])


def test_det_preprune_probe_is_ceiling(test, device):
    """Verify that the preprune probe is >= any packed value with the same score and fingerprint."""
    n = 1
    score = wp.array([0.5], dtype=float, device=device)
    fp = wp.array([42], dtype=int, device=device)
    probe = wp.zeros(n, dtype=wp.uint64, device=device)
    wp.launch(_det_preprune_probe_kernel, dim=n, inputs=[score, fp, probe], device=device)

    # Pack with same score/fp but various contact_ids
    ids_to_test = [0, 1, 100, 500000, (1 << 20) - 1]
    for cid in ids_to_test:
        ids_arr = wp.array([cid], dtype=int, device=device)
        packed = wp.zeros(1, dtype=wp.uint64, device=device)
        unpacked = wp.zeros(1, dtype=int, device=device)
        wp.launch(_det_pack_unpack_kernel, dim=1, inputs=[score, fp, ids_arr, packed, unpacked], device=device)

        probe_val = int(probe.numpy()[0])
        packed_val = int(packed.numpy()[0])
        test.assertGreaterEqual(
            probe_val,
            packed_val,
            f"Preprune probe should be >= packed value for contact_id={cid}",
        )


# =============================================================================
# End-to-end deterministic test
# =============================================================================


class TestDeterministicEndToEnd(unittest.TestCase):
    """End-to-end test that deterministic mode produces identical contacts across runs."""

    pass


def test_deterministic_identical_across_runs(test, device):
    """Create a deterministic reducer, collide the same scene N times, assert identical winners.

    Compares winning contacts by their geometric content (position, depth,
    fingerprint) rather than buffer contact IDs, since IDs are assigned by
    atomic_add and vary between runs.
    """
    if str(device) == "cpu":
        return  # Deterministic mode is primarily for GPU

    num_runs = 5
    all_contact_sets = []

    for _ in range(num_runs):
        reducer = GlobalContactReducer(capacity=500, device=device, deterministic=True)
        reducer_data = reducer.get_data_struct()

        @wp.kernel(enable_backward=False)
        def store_contacts_det_kernel(reducer_data: GlobalContactReducerData):
            tid = wp.tid()
            x = float(tid % 10) - 5.0
            z = float(tid // 10) - 5.0
            depth = -0.001 * float((tid * 7 + 3) % 20 + 1)
            fingerprint = tid * 3 + 1

            X_ws = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
            export_and_reduce_contact_centered(
                shape_a=0,
                shape_b=1,
                position=wp.vec3(x, 0.0, z),
                normal=wp.vec3(0.0, 1.0, 0.0),
                depth=depth,
                fingerprint=fingerprint,
                centered_position=wp.vec3(x, 0.0, z),
                X_ws_voxel_shape=X_ws,
                aabb_lower_voxel=wp.vec3(-10.0, -5.0, -10.0),
                aabb_upper_voxel=wp.vec3(10.0, 5.0, 10.0),
                voxel_res=wp.vec3i(4, 4, 4),
                reducer_data=reducer_data,
            )

        wp.launch(store_contacts_det_kernel, dim=100, inputs=[reducer_data], device=device)

        # Extract winning contacts by geometric content (not by contact ID)
        values = reducer.ht_values.numpy()
        capacity = reducer.hashtable.capacity
        active_slots_np = reducer.hashtable.active_slots.numpy()
        count = active_slots_np[capacity]
        pd_np = reducer.position_depth.numpy()
        fp_np = reducer.contact_fingerprints.numpy()

        seen_ids = set()
        contact_set = set()
        for i in range(count):
            entry_idx = active_slots_np[i]
            for slot in range(reducer.values_per_key):
                val = values[slot * capacity + entry_idx]
                if val != 0:
                    contact_id = int(val & ((1 << 20) - 1))
                    if contact_id not in seen_ids:
                        seen_ids.add(contact_id)
                        pd = pd_np[contact_id]
                        fp = int(fp_np[contact_id])
                        # Round to avoid floating-point noise from storage
                        key = (
                            round(float(pd[0]), 5),
                            round(float(pd[1]), 5),
                            round(float(pd[2]), 5),
                            round(float(pd[3]), 5),
                            fp,
                        )
                        contact_set.add(key)

        all_contact_sets.append(contact_set)

    # All runs should produce identical geometric contact sets
    for run_idx in range(1, num_runs):
        test.assertEqual(
            all_contact_sets[0],
            all_contact_sets[run_idx],
            f"Run 0 and run {run_idx} produced different winning contacts",
        )


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
add_function_test(
    TestGlobalContactReducer,
    "test_centered_basic_storage_and_reduction",
    test_centered_basic_storage_and_reduction,
    devices=devices,
)
add_function_test(
    TestGlobalContactReducer,
    "test_centered_different_pairs_independent",
    test_centered_different_pairs_independent,
    devices=devices,
)
add_function_test(
    TestGlobalContactReducer,
    "test_centered_deepest_wins_max_depth_slot",
    test_centered_deepest_wins_max_depth_slot,
    devices=devices,
)
add_function_test(
    TestGlobalContactReducer,
    "test_centered_pre_pruning_reduces_buffer_usage",
    test_centered_pre_pruning_reduces_buffer_usage,
    devices=devices,
)
add_function_test(TestKeyConstruction, "test_key_uniqueness", test_key_uniqueness, devices=devices)
add_function_test(
    TestKeyConstruction,
    "test_oct_encode_decode_roundtrip",
    test_oct_encode_decode_roundtrip,
    devices=devices,
)
add_function_test(
    TestFloatFlipPrecision,
    "test_float_flip_roundtrip_precision",
    test_float_flip_roundtrip_precision,
    devices=devices,
)
add_function_test(
    TestFloatFlipPrecision,
    "test_float_flip_exact_inverse",
    test_float_flip_exact_inverse,
    devices=devices,
)

# make_contact_sort_key tests
add_function_test(TestMakeContactSortKey, "test_sort_key_bit_layout", test_sort_key_bit_layout, devices=devices)
add_function_test(
    TestMakeContactSortKey, "test_sort_key_overflow_masking", test_sort_key_overflow_masking, devices=devices
)

# Deterministic packing tests
add_function_test(
    TestDeterministicPacking, "test_det_pack_unpack_roundtrip", test_det_pack_unpack_roundtrip, devices=devices
)
add_function_test(
    TestDeterministicPacking, "test_det_packing_score_dominates", test_det_packing_score_dominates, devices=devices
)
add_function_test(
    TestDeterministicPacking,
    "test_det_packing_fingerprint_breaks_tie",
    test_det_packing_fingerprint_breaks_tie,
    devices=devices,
)
add_function_test(
    TestDeterministicPacking,
    "test_det_preprune_probe_is_ceiling",
    test_det_preprune_probe_is_ceiling,
    devices=devices,
)

# End-to-end deterministic test
add_function_test(
    TestDeterministicEndToEnd,
    "test_deterministic_identical_across_runs",
    test_deterministic_identical_across_runs,
    devices=devices,
)


if __name__ == "__main__":
    wp.init()
    unittest.main(verbosity=2)
