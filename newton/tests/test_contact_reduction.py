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

"""Tests for contact reduction functionality.

This test suite validates:
1. Icosahedron face normals are unit vectors
2. get_slot returns correct face indices for different normals
3. Contact reduction utility functions work correctly

Note: Tests that use shared memory (ContactReductionFunctions) require CUDA.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.contact_reduction import (
    ICOSAHEDRON_FACE_NORMALS,
    NUM_NORMAL_BINS,
    NUM_SPATIAL_DIRECTIONS,
    ContactReductionFunctions,
    ContactStruct,
    compute_num_reduction_slots,
    create_betas_array,
    get_slot,
    synchronize,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


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
    # Formula: 20 + (20 bins * 6 directions * num_betas) = 20 + 120 * num_betas
    # With 1 beta: 20 + 120 = 140
    test.assertEqual(compute_num_reduction_slots(1), 140)
    # With default 2 betas: 20 + 240 = 260
    test.assertEqual(compute_num_reduction_slots(2), 260)
    # With 3 betas: 20 + 360 = 380
    test.assertEqual(compute_num_reduction_slots(3), 380)


def test_create_betas_array(test, device):
    """Test create_betas_array creates correct array."""
    betas = (10.0, 1000000.0)
    arr = create_betas_array(betas, device=device)

    test.assertEqual(arr.shape, (2,))
    test.assertEqual(arr.dtype, wp.float32)

    arr_np = arr.numpy()
    test.assertAlmostEqual(arr_np[0], 10.0, places=5)
    test.assertAlmostEqual(arr_np[1], 1000000.0, places=1)


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
# Tests for ContactReductionFunctions (requires CUDA for shared memory)
# =============================================================================


@wp.func
def _generate_test_contact(t: int) -> ContactStruct:
    """Generate deterministic test contact data."""
    c = ContactStruct()
    ft = float(t)
    c.position = wp.vec3(wp.sin(ft * 0.1) * ft * 0.01, 0.0, wp.cos(ft * 0.1) * ft * 0.01)
    c.normal = wp.vec3(0.0, 1.0, 0.0)
    c.depth = 0.1
    c.mode = t % 10  # Assign features 0-9 cyclically
    c.projection = 0.0
    return c


def _create_reduction_test_kernel(reduction_funcs: ContactReductionFunctions):
    """Create a test kernel for contact reduction with shared memory."""
    num_slots = reduction_funcs.num_reduction_slots
    store_reduced_contact = reduction_funcs.store_reduced_contact
    collect_active_contacts = reduction_funcs.collect_active_contacts

    @wp.kernel(enable_backward=False)
    def reduction_test_kernel(
        out_contacts: wp.array(dtype=ContactStruct),
        out_count: wp.array(dtype=int),
        betas_arr: wp.array(dtype=wp.float32),
    ):
        _block_id, t = wp.tid()
        empty_marker = -1000000000.0

        # Initialize shared memory buffer
        buffer = wp.array(
            ptr=reduction_funcs.get_smem_slots_contacts(),
            shape=(wp.static(num_slots),),
            dtype=ContactStruct,
        )
        active_ids = wp.array(
            ptr=reduction_funcs.get_smem_slots_plus_1(),
            shape=(wp.static(num_slots + 1),),
            dtype=wp.int32,
        )

        # Initialize buffer
        for i in range(t, wp.static(num_slots), wp.block_dim()):
            buffer[i].projection = empty_marker
        if t == 0:
            active_ids[wp.static(num_slots)] = 0
        synchronize()

        # Generate and store contacts (every other thread has a contact)
        has_contact = t % 2 == 0
        c = _generate_test_contact(t)

        store_reduced_contact(t, has_contact, c, buffer, active_ids, betas_arr, empty_marker)

        # Collect active contacts
        collect_active_contacts(t, buffer, active_ids, empty_marker)

        # Write output
        num_contacts = active_ids[wp.static(num_slots)]
        if t == 0:
            out_count[0] = num_contacts

        for i in range(t, num_contacts, wp.block_dim()):
            contact_id = active_ids[i]
            out_contacts[i] = buffer[contact_id]

    return reduction_test_kernel


def test_reduction_functions_initialization(test, device):
    """Test that ContactReductionFunctions initializes correctly."""
    funcs = ContactReductionFunctions(betas=(10.0, 1000000.0))

    test.assertEqual(funcs.num_betas, 2)
    test.assertEqual(funcs.betas, (10.0, 1000000.0))
    # 20 + (20 bins * 6 directions * 2 betas) = 260
    test.assertEqual(funcs.num_reduction_slots, 260)


def test_reduction_functions_single_beta(test, device):
    """Test ContactReductionFunctions with single beta value."""
    funcs = ContactReductionFunctions(betas=(100.0,))

    test.assertEqual(funcs.num_betas, 1)
    # 20 + (20 bins * 6 directions * 1 beta) = 140
    test.assertEqual(funcs.num_reduction_slots, 140)


def test_contact_reduction_produces_valid_output(test, device):
    """Test that contact reduction kernel produces valid output."""
    reduction_funcs = ContactReductionFunctions(betas=(10.0, 1000000.0))
    num_slots = reduction_funcs.num_reduction_slots

    # Create test kernel
    kernel = _create_reduction_test_kernel(reduction_funcs)

    # Allocate arrays on GPU
    out_contacts = wp.zeros(num_slots, dtype=ContactStruct, device=device)
    out_count = wp.zeros(1, dtype=int, device=device)
    betas_arr = reduction_funcs.create_betas_array(device=device)

    # Launch kernel with tiled launch (for shared memory)
    wp.launch_tiled(
        kernel=kernel,
        dim=1,
        inputs=[out_contacts, out_count, betas_arr],
        block_dim=128,
        device=device,
    )
    wp.synchronize_device(device)

    # Verify output
    count = out_count.numpy()[0]
    test.assertGreater(count, 0, "No contacts were reduced")
    test.assertLessEqual(count, num_slots, "Too many contacts")

    # Verify contact data is valid
    contacts = out_contacts.numpy()
    for i in range(count):
        c = contacts[i]
        # Projection should be set (not the empty marker)
        test.assertGreater(c["projection"], -1e9, f"Contact {i} has invalid projection")
        # Normal should be unit-ish (we set it to (0,1,0))
        normal = c["normal"]
        test.assertAlmostEqual(normal[1], 1.0, places=5)


def test_contact_reduction_reduces_count(test, device):
    """Test that contact reduction reduces the number of contacts."""
    reduction_funcs = ContactReductionFunctions(betas=(10.0,))
    num_slots = reduction_funcs.num_reduction_slots

    kernel = _create_reduction_test_kernel(reduction_funcs)

    out_contacts = wp.zeros(num_slots, dtype=ContactStruct, device=device)
    out_count = wp.zeros(1, dtype=int, device=device)
    betas_arr = reduction_funcs.create_betas_array(device=device)

    wp.launch_tiled(
        kernel=kernel,
        dim=1,
        inputs=[out_contacts, out_count, betas_arr],
        block_dim=128,
        device=device,
    )
    wp.synchronize_device(device)

    count = out_count.numpy()[0]

    # With 64 active contacts (128 threads, every other one active),
    # reduction should produce fewer contacts due to keeping only best contact per (bin, direction) slot
    test.assertGreater(count, 0, "Should have at least one contact")
    test.assertLess(count, 64, "Reduction should reduce contact count")


# =============================================================================
# Test registration
# =============================================================================

devices = get_test_devices()
cuda_devices = get_cuda_test_devices()

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
    add_function_test(TestContactReduction, "test_create_betas_array", test_create_betas_array, devices=[device])

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

# ContactReductionFunctions tests (CUDA only - uses shared memory)
for device in cuda_devices:
    add_function_test(
        TestContactReduction,
        "test_reduction_functions_initialization",
        test_reduction_functions_initialization,
        devices=[device],
    )
    add_function_test(
        TestContactReduction,
        "test_reduction_functions_single_beta",
        test_reduction_functions_single_beta,
        devices=[device],
    )
    add_function_test(
        TestContactReduction,
        "test_contact_reduction_produces_valid_output",
        test_contact_reduction_produces_valid_output,
        devices=[device],
    )
    add_function_test(
        TestContactReduction,
        "test_contact_reduction_reduces_count",
        test_contact_reduction_reduces_count,
        devices=[device],
    )


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
