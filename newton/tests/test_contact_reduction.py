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

import newton
from newton._src.geometry.contact_reduction import (
    _POLYHEDRON_BINS,
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


def _slots(normals, device):
    normals = np.asarray(normals, dtype=np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    out = wp.zeros(len(normals), dtype=int, device=device)
    wp.launch(
        _get_slot_kernel, dim=len(normals), inputs=[wp.array(normals, dtype=wp.vec3, device=device), out], device=device
    )
    return normals, out.numpy()


def _tilted(axis, angle, count=16):
    """Unit normals at ``angle`` [rad] from ``axis``, spread around it."""
    axis = np.asarray(axis, dtype=np.float64)
    tangent = np.cross(axis, [0.0, 0.0, 1.0] if abs(axis[2]) < 0.9 else [1.0, 0.0, 0.0])
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(axis, tangent)
    phi = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)[:, None]
    return np.cos(angle) * axis + np.sin(angle) * (np.cos(phi) * tangent + np.sin(phi) * bitangent)


def test_get_slot_keeps_world_axes_inside_one_bin(test, device):
    """Normals near a world axis share the axis' bin, so rounding noise can't move flat contacts between bins."""
    for axis, angle in (
        ((0.0, 0.0, 1.0), np.radians(15.0)),
        ((0.0, 0.0, -1.0), np.radians(15.0)),
        ((1.0, 0.0, 0.0), np.radians(3.0)),
        ((-1.0, 0.0, 0.0), np.radians(3.0)),
        ((0.0, 1.0, 0.0), np.radians(3.0)),
        ((0.0, -1.0, 0.0), np.radians(3.0)),
    ):
        normals = np.concatenate([[axis], _tilted(axis, 1.0e-4), _tilted(axis, angle)])
        _, slots = _slots(normals, device)
        test.assertEqual(len(set(slots.tolist())), 1, f"axis {axis} spans bins {sorted(set(slots.tolist()))}")


def test_get_slot_matches_best_face_normal_near_range_bounds(test, device):
    """Match a full search where get_slot() switches ranges and where bins meet."""
    rng = np.random.default_rng(7)
    face_normals = np.array([[FACE_NORMALS[i, j] for j in range(3)] for i in range(NUM_NORMAL_BINS)])
    bands = []
    for z in (1.0, 0.85, 0.79, 0.65, 0.25, 0.19, 0.0, -0.19, -0.25, -0.65, -0.79, -0.85, -1.0):
        zz = np.clip(z + rng.uniform(-1.0e-3, 1.0e-3, 400), -1.0, 1.0)
        phi = rng.uniform(0.0, 2.0 * np.pi, 400)
        r = np.sqrt(1.0 - zz**2)
        bands.append(np.stack([r * np.cos(phi), r * np.sin(phi), zz], axis=1))
    for axis in np.eye(3):
        bands.append(np.concatenate([[axis], [-axis], _tilted(axis, 1.0e-4), _tilted(-axis, 1.0e-4)]))
    normals, slots = _slots(np.concatenate(bands), device)
    # Float32 dot products can't separate faces closer than this.
    dots = normals @ face_normals.T
    np.testing.assert_array_less(dots.max(axis=1) - 1.0e-6, dots[np.arange(len(normals)), slots])


def test_mesh_ground_contacts_near_up_share_one_bin(test, device):
    """Box-on-mesh contacts whose normals are within a few degrees of +Z all bin together."""
    rng = np.random.default_rng(0)
    xs = np.linspace(-0.5, 0.5, 9)
    x, y = np.meshgrid(xs, xs, indexing="ij")
    vertices = np.stack([x.ravel(), y.ravel(), rng.uniform(-1.0e-5, 1.0e-5, x.size)], axis=1)
    indices = []
    for i in range(8):
        for j in range(8):
            a, b, c, d = i * 9 + j, (i + 1) * 9 + j, (i + 1) * 9 + j + 1, i * 9 + j + 1
            indices += [a, b, c, a, c, d]
    builder = newton.ModelBuilder()
    builder.add_shape_mesh(-1, mesh=newton.Mesh(vertices.astype(np.float32), np.array(indices, dtype=np.int32)))
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0495), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(body, hx=0.2, hy=0.2, hz=0.05)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    normals = contacts.rigid_contact_normal.numpy()[:count]
    up = normals[np.abs(normals @ np.array([0.0, 0.0, 1.0])) > np.cos(np.radians(5.0))]
    test.assertGreater(len(up), 4, "the box does not rest on the mesh")
    _, slots = _slots(up, device)
    test.assertEqual(
        len(set(slots.tolist())), 1, f"near-vertical mesh contacts span bins {sorted(set(slots.tolist()))}"
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

    if NORMAL_BINNING_POLYHEDRON == "icosahedron":
        add_function_test(
            TestContactReduction,
            "test_get_slot_keeps_world_axes_inside_one_bin",
            test_get_slot_keeps_world_axes_inside_one_bin,
            devices=[device],
        )
    add_function_test(
        TestContactReduction,
        "test_get_slot_matches_best_face_normal_near_range_bounds",
        test_get_slot_matches_best_face_normal_near_range_bounds,
        devices=[device],
    )
    add_function_test(
        TestContactReduction,
        "test_mesh_ground_contacts_near_up_share_one_bin",
        test_mesh_ground_contacts_near_up_share_one_bin,
        devices=[device],
    )


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
