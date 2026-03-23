# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for contact reduction functionality.

This test suite validates:
1. Icosahedron face normals are unit vectors
2. get_slot returns correct face indices for different normals
3. Contact reduction utility functions work correctly
"""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.contact_reduction import (
    ICOSAHEDRON_FACE_NORMALS,
    NUM_NORMAL_BINS,
    NUM_SPATIAL_DIRECTIONS,
    compute_num_reduction_slots,
    get_slot,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _get_slot_kernel(
    normals: wp.array(dtype=wp.vec3),
    slots: wp.array(dtype=int),
):
    """Kernel to test get_slot function."""
    tid = wp.tid()
    slots[tid] = get_slot(normals[tid])


class TestContactReduction(unittest.TestCase):
    """Tests for contact reduction functionality."""

    pass


# =============================================================================
# Tests for icosahedron geometry (no device needed, pure Python/NumPy)
# =============================================================================


def test_face_normals_are_unit_vectors(test, device):
    """Verify all 20 icosahedron face normals are unit vectors."""
    for i in range(NUM_NORMAL_BINS):
        normal = np.array(
            [
                ICOSAHEDRON_FACE_NORMALS[i, 0],
                ICOSAHEDRON_FACE_NORMALS[i, 1],
                ICOSAHEDRON_FACE_NORMALS[i, 2],
            ]
        )
        length = np.linalg.norm(normal)
        test.assertAlmostEqual(length, 1.0, places=5, msg=f"Face normal {i} is not a unit vector")


def test_face_normals_cover_sphere(test, device):
    """Test that face normals roughly cover the sphere (no hemisphere is empty)."""
    normals = []
    for i in range(NUM_NORMAL_BINS):
        normals.append(
            [
                ICOSAHEDRON_FACE_NORMALS[i, 0],
                ICOSAHEDRON_FACE_NORMALS[i, 1],
                ICOSAHEDRON_FACE_NORMALS[i, 2],
            ]
        )
    normals = np.array(normals)

    # Check there are normals with positive and negative components in each axis
    test.assertTrue(np.any(normals[:, 0] > 0.3), "No face normals point in +X direction")
    test.assertTrue(np.any(normals[:, 0] < -0.3), "No face normals point in -X direction")
    test.assertTrue(np.any(normals[:, 1] > 0.3), "No face normals point in +Y direction")
    test.assertTrue(np.any(normals[:, 1] < -0.3), "No face normals point in -Y direction")
    test.assertTrue(np.any(normals[:, 2] > 0.3), "No face normals point in +Z direction")
    test.assertTrue(np.any(normals[:, 2] < -0.3), "No face normals point in -Z direction")


def test_constants(test, device):
    """Test NUM_NORMAL_BINS and NUM_SPATIAL_DIRECTIONS constants."""
    test.assertEqual(NUM_NORMAL_BINS, 20)  # icosahedron faces
    test.assertEqual(NUM_SPATIAL_DIRECTIONS, 6)  # 3 edges + 3 negated


def test_compute_num_reduction_slots(test, device):
    """Test compute_num_reduction_slots calculation."""
    # Formula: 20 bins * (6 directions + 1 max-depth) + 100 voxel slots
    # 20 * 7 + 100 = 140 + 100 = 240
    test.assertEqual(compute_num_reduction_slots(), 240)


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

    # All slots should be valid (0-19)
    for i, slot in enumerate(slots_np):
        test.assertGreaterEqual(slot, 0, f"Slot {i} is negative")
        test.assertLess(slot, NUM_NORMAL_BINS, f"Slot {i} exceeds max ({NUM_NORMAL_BINS})")


def test_get_slot_matches_best_face_normal(test, device):
    """Test that get_slot returns the face with highest dot product."""
    # Use a random set of normals and verify result matches CPU reference
    rng = np.random.default_rng(42)
    test_normals_np = rng.standard_normal((50, 3)).astype(np.float32)
    # Normalize
    test_normals_np /= np.linalg.norm(test_normals_np, axis=1, keepdims=True)

    test_normals = [wp.vec3(n[0], n[1], n[2]) for n in test_normals_np]
    normals = wp.array(test_normals, dtype=wp.vec3, device=device)
    slots = wp.zeros(len(test_normals), dtype=int, device=device)

    wp.launch(_get_slot_kernel, dim=len(test_normals), inputs=[normals, slots], device=device)

    slots_np = slots.numpy()

    # Build face normals array for CPU reference
    face_normals = np.array([[ICOSAHEDRON_FACE_NORMALS[i, j] for j in range(3)] for i in range(NUM_NORMAL_BINS)])

    # Verify each slot
    for i in range(len(test_normals_np)):
        normal = test_normals_np[i]
        result_slot = slots_np[i]

        # Compute dot products with all face normals
        dots = face_normals @ normal
        cpu_best_slot = np.argmax(dots)

        test.assertEqual(
            result_slot, cpu_best_slot, f"Normal {i}: result slot {result_slot} != expected slot {cpu_best_slot}"
        )


# =============================================================================
# Test registration
# =============================================================================

devices = get_test_devices()

# Register tests that work on all devices (CPU and CUDA)
for device in devices:
    # Icosahedron geometry tests (pure NumPy, but registered per device for consistency)
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

    # get_slot tests
    add_function_test(
        TestContactReduction, "test_get_slot_axis_aligned_normals", test_get_slot_axis_aligned_normals, devices=[device]
    )
    add_function_test(
        TestContactReduction,
        "test_get_slot_matches_best_face_normal",
        test_get_slot_matches_best_face_normal,
        devices=[device],
    )


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
