# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for contact reduction functionality.

All tests are written against the *configured* polyhedron (set via
``NORMAL_BINNING_POLYHEDRON`` in ``contact_reduction.py``) so they pass
regardless of which polyhedron or slot counts are selected.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.contact_reduction import (
    _POLYHEDRON_BINS,
    FACE_NORMALS,
    MAX_CONTACTS_PER_PAIR,
    NORMAL_BINNING_POLYHEDRON,
    NUM_NORMAL_BINS,
    NUM_SPATIAL_DIRECTIONS,
    NUM_VOXEL_DEPTH_SLOTS,
    compute_num_reduction_slots,
    get_slot,
)
from newton._src.geometry.contact_reduction_hydroelastic import compute_pressure_memory_scales_kernel
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _get_slot_kernel(
    normals: wp.array[wp.vec3],
    slots: wp.array[int],
):
    """Kernel to test get_slot function."""
    tid = wp.tid()
    slots[tid] = get_slot(normals[tid])


class TestContactReduction(unittest.TestCase):
    """Tests for contact reduction functionality."""

    pass


# =============================================================================
# Face-normal geometry tests
# =============================================================================


def test_face_normals_are_unit_vectors(test, device):
    """Verify all face normals of the configured polyhedron are unit vectors."""
    for i in range(NUM_NORMAL_BINS):
        normal = np.array([FACE_NORMALS[i, 0], FACE_NORMALS[i, 1], FACE_NORMALS[i, 2]])
        length = np.linalg.norm(normal)
        test.assertAlmostEqual(length, 1.0, places=5, msg=f"Face normal {i} is not a unit vector")


def test_face_normals_cover_sphere(test, device):
    """Test that face normals roughly cover the sphere (no hemisphere is empty)."""
    normals = np.array([[FACE_NORMALS[i, j] for j in range(3)] for i in range(NUM_NORMAL_BINS)])

    test.assertTrue(np.any(normals[:, 0] > 0.3), "No face normals point in +X direction")
    test.assertTrue(np.any(normals[:, 0] < -0.3), "No face normals point in -X direction")
    test.assertTrue(np.any(normals[:, 1] > 0.3), "No face normals point in +Y direction")
    test.assertTrue(np.any(normals[:, 1] < -0.3), "No face normals point in -Y direction")
    test.assertTrue(np.any(normals[:, 2] > 0.3), "No face normals point in +Z direction")
    test.assertTrue(np.any(normals[:, 2] < -0.3), "No face normals point in -Z direction")


def test_constants(test, device):
    """Verify invariants that must hold for any valid configuration."""
    test.assertEqual(NUM_NORMAL_BINS, _POLYHEDRON_BINS[NORMAL_BINNING_POLYHEDRON])
    test.assertGreaterEqual(NUM_NORMAL_BINS, 6)
    test.assertGreaterEqual(NUM_SPATIAL_DIRECTIONS, 3)
    test.assertGreaterEqual(NUM_VOXEL_DEPTH_SLOTS, 1)
    test.assertLessEqual(compute_num_reduction_slots(), MAX_CONTACTS_PER_PAIR)


def test_compute_num_reduction_slots(test, device):
    """Test compute_num_reduction_slots formula and MAX_CONTACTS_PER_PAIR ceiling."""
    expected = NUM_NORMAL_BINS * (NUM_SPATIAL_DIRECTIONS + 1) + NUM_VOXEL_DEPTH_SLOTS
    test.assertEqual(compute_num_reduction_slots(), expected)
    test.assertLessEqual(compute_num_reduction_slots(), MAX_CONTACTS_PER_PAIR)


# =============================================================================
# Tests for get_slot function (works on CPU and GPU)
# =============================================================================


def test_get_slot_axis_aligned_normals(test, device):
    """Test get_slot with axis-aligned normals."""
    test_normals = [
        wp.vec3(0.0, 1.0, 0.0),  # +Y (top)
        wp.vec3(0.0, -1.0, 0.0),  # -Y (bottom)
        wp.vec3(1.0, 0.0, 0.0),  # +X
        wp.vec3(-1.0, 0.0, 0.0),  # -X
        wp.vec3(0.0, 0.0, 1.0),  # +Z
        wp.vec3(0.0, 0.0, -1.0),  # -Z
    ]

    normals = wp.array(test_normals, dtype=wp.vec3, device=device)
    slots = wp.zeros(len(test_normals), dtype=int, device=device)

    wp.launch(_get_slot_kernel, dim=len(test_normals), inputs=[normals, slots], device=device)

    slots_np = slots.numpy()

    for i, slot in enumerate(slots_np):
        test.assertGreaterEqual(slot, 0, f"Slot {i} is negative")
        test.assertLess(slot, NUM_NORMAL_BINS, f"Slot {i} exceeds max ({NUM_NORMAL_BINS})")


def test_get_slot_matches_best_face_normal(test, device):
    """Test that get_slot returns the face with highest dot product."""
    rng = np.random.default_rng(42)
    test_normals_np = rng.standard_normal((50, 3)).astype(np.float32)
    test_normals_np /= np.linalg.norm(test_normals_np, axis=1, keepdims=True)

    test_normals = [wp.vec3(n[0], n[1], n[2]) for n in test_normals_np]
    normals = wp.array(test_normals, dtype=wp.vec3, device=device)
    slots = wp.zeros(len(test_normals), dtype=int, device=device)

    wp.launch(_get_slot_kernel, dim=len(test_normals), inputs=[normals, slots], device=device)

    slots_np = slots.numpy()

    face_normals = np.array([[FACE_NORMALS[i, j] for j in range(3)] for i in range(NUM_NORMAL_BINS)])

    for i in range(len(test_normals_np)):
        normal = test_normals_np[i]
        result_slot = slots_np[i]

        dots = face_normals @ normal
        cpu_best_slot = np.argmax(dots)

        test.assertEqual(
            result_slot, cpu_best_slot, f"Normal {i}: result slot {result_slot} != expected slot {cpu_best_slot}"
        )


def test_pressure_memory_scales_unloading_force(test, device):
    """Pressure memory should remember max compression and attenuate unloading."""
    capacity = 4
    shape_a = 1
    shape_b = 2
    bin_id = 3
    key = np.uint64(shape_a | (shape_b << 27) | (bin_id << 55))

    keys = wp.array(np.array([key, 0, 0, 0], dtype=np.uint64), dtype=wp.uint64, device=device)
    active_np = np.zeros(capacity + 1, dtype=np.int32)
    active_np[0] = 0
    active_np[capacity] = 1
    active_slots = wp.array(active_np, dtype=wp.int32, device=device)
    weighted_pos_sum = wp.array(
        np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )
    weight_sum = wp.array(np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32), dtype=wp.float32, device=device)
    aabb_lower = wp.array(np.zeros((3, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    aabb_upper = wp.array(np.ones((3, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    pressure_memory = wp.zeros(3 * 3 * NUM_NORMAL_BINS * 2 * 2, dtype=wp.float32, device=device)
    entry_scale = wp.ones(capacity, dtype=wp.float32, device=device)

    agg_loading = wp.array(
        np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )
    wp.launch(
        compute_pressure_memory_scales_kernel,
        dim=1,
        inputs=[
            keys,
            active_slots,
            agg_loading,
            weighted_pos_sum,
            weight_sum,
            aabb_lower,
            aabb_upper,
            pressure_memory,
            entry_scale,
            3,
            2,
            2,
            1,
            0.5,
            0.25,
            1.0 / 240.0,
            1,
        ],
        device=device,
    )
    test.assertAlmostEqual(float(entry_scale.numpy()[0]), 1.0, places=6)

    agg_unloading = wp.array(
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )
    wp.launch(
        compute_pressure_memory_scales_kernel,
        dim=1,
        inputs=[
            keys,
            active_slots,
            agg_unloading,
            weighted_pos_sum,
            weight_sum,
            aabb_lower,
            aabb_upper,
            pressure_memory,
            entry_scale,
            3,
            2,
            2,
            1,
            0.5,
            0.25,
            1.0 / 240.0,
            1,
        ],
        device=device,
    )
    test.assertLess(float(entry_scale.numpy()[0]), 1.0)
    test.assertGreater(float(pressure_memory.numpy().max()), 1.0)


# =============================================================================
# Test registration
# =============================================================================

devices = get_test_devices()

for device in devices:
    add_function_test(
        TestContactReduction, "test_face_normals_are_unit_vectors", test_face_normals_are_unit_vectors, devices=[device]
    )
    add_function_test(
        TestContactReduction, "test_face_normals_cover_sphere", test_face_normals_cover_sphere, devices=[device]
    )
    add_function_test(TestContactReduction, "test_constants", test_constants, devices=[device])
    add_function_test(
        TestContactReduction, "test_compute_num_reduction_slots", test_compute_num_reduction_slots, devices=[device]
    )
    add_function_test(
        TestContactReduction, "test_get_slot_axis_aligned_normals", test_get_slot_axis_aligned_normals, devices=[device]
    )
    add_function_test(
        TestContactReduction,
        "test_get_slot_matches_best_face_normal",
        test_get_slot_matches_best_face_normal,
        devices=[device],
    )

add_function_test(
    TestContactReduction,
    "test_pressure_memory_scales_unloading_force",
    test_pressure_memory_scales_unloading_force,
    devices=[wp.get_device("cpu")],
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
