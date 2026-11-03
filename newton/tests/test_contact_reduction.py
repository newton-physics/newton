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
    _POLYHEDRON_NORMALS_DATA,
    FACE_NORMALS,
    MAX_CONTACTS_PER_PAIR,
    NORMAL_BINNING_POLYHEDRON,
    NUM_NORMAL_BINS,
    NUM_SPATIAL_DIRECTIONS,
    NUM_VOXEL_DEPTH_SLOTS,
    compute_num_reduction_slots,
    get_face_normal,
    get_slot,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _get_slot_kernel(
    normals: wp.array[wp.vec3],
    slots: wp.array[int],
):
    """Kernel to test get_slot function."""
    tid = wp.tid()
    slots[tid] = get_slot(normals[tid])


@wp.kernel
def _get_face_normal_kernel(normals: wp.array[wp.vec3]):
    """Fetch every configured face normal through the device lookup."""
    tid = wp.tid()
    normals[tid] = get_face_normal(tid)


class TestContactReduction(unittest.TestCase):
    """Tests for contact reduction functionality."""

    pass


# =============================================================================
# Face-normal geometry tests
# =============================================================================


def test_face_normal_lookup_matches_matrix(test, device):
    """Verify the device lookup exactly reproduces the configured face-normal matrix."""
    normals = wp.empty(NUM_NORMAL_BINS, dtype=wp.vec3, device=device)
    wp.launch(_get_face_normal_kernel, dim=NUM_NORMAL_BINS, inputs=[normals], device=device)

    expected = np.array([[FACE_NORMALS[i, j] for j in range(3)] for i in range(NUM_NORMAL_BINS)], dtype=np.float32)
    np.testing.assert_array_equal(normals.numpy(), expected)


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
    """Keep axis-aligned normals and nearby ground and wall normals in valid, stable bins."""
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

    # Require the same clearance around every coordinate axis. Keep thresholds
    # just inside the geometric margins to allow for the rounded table entries.
    protected_degrees = {"hexahedron": 44.0, "octahedron": 19.0, "dodecahedron": 14.0, "icosahedron": 14.0}
    axes = np.asarray(test_normals)

    # Check every table on CI even when only one configuration is compiled.
    for polyhedron, values in _POLYHEDRON_NORMALS_DATA.items():
        with test.subTest(polyhedron=polyhedron):
            tilt = np.deg2rad(protected_degrees[polyhedron])
            tilted_normals = []
            for axis in axes:
                tangent_axes = np.eye(3)[axis == 0.0]
                for angle in np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False):
                    tangent = np.cos(angle) * tangent_axes[0] + np.sin(angle) * tangent_axes[1]
                    tilted_normals.append(np.cos(tilt) * axis + np.sin(tilt) * tangent)
            tilted_normals = np.asarray(tilted_normals, dtype=np.float32)
            if polyhedron == NORMAL_BINNING_POLYHEDRON:
                configured_tilted_normals = tilted_normals
            faces = np.asarray(values, dtype=np.float32).reshape(-1, 3)
            best_faces = (axes @ faces.T).argmax(axis=1)
            # Bound the whole cone, including directions between the samples:
            # a bin boundary is perpendicular to the difference of its faces.
            margins = []
            for axis, best in zip(axes, best_faces, strict=True):
                differences = faces[best] - np.delete(faces, best, axis=0)
                distances = differences @ axis / np.linalg.norm(differences, axis=1)
                margin = np.rad2deg(np.arcsin(np.clip(distances, -1.0, 1.0))).min()
                test.assertGreater(margin, protected_degrees[polyhedron])
                margins.append(margin)
            test.assertLess(np.ptp(margins), 1.0e-4, "Axis clearances must agree within table rounding error")
            expected = np.repeat(best_faces[:, None], 32, axis=1)
            np.testing.assert_array_equal((tilted_normals @ faces.T).argmax(axis=1).reshape(6, 32), expected)

    normals = wp.array(configured_tilted_normals, dtype=wp.vec3, device=device)
    slots = wp.empty(len(configured_tilted_normals), dtype=int, device=device)
    wp.launch(_get_slot_kernel, dim=len(configured_tilted_normals), inputs=[normals, slots], device=device)
    np.testing.assert_array_equal(slots.numpy().reshape(6, 32), np.repeat(slots_np[:, None], 32, axis=1))


def test_get_slot_matches_best_face_normal(test, device):
    """Test that get_slot returns the face with highest dot product."""
    rng = np.random.default_rng(42)
    test_normals_np = rng.standard_normal((2048, 3)).astype(np.float32)
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


# =============================================================================
# Test registration
# =============================================================================

devices = get_test_devices()

for device in devices:
    add_function_test(
        TestContactReduction, "test_face_normals_are_unit_vectors", test_face_normals_are_unit_vectors, devices=[device]
    )
    add_function_test(
        TestContactReduction,
        "test_face_normal_lookup_matches_matrix",
        test_face_normal_lookup_matches_matrix,
        devices=[device],
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


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
