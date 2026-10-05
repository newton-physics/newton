# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test GJK distances, normals and witnesses for near-contact convex pairs."""

import unittest

import numpy as np
import warp as wp

from newton import GeoType
from newton._src.geometry.simplex_solver import create_solve_closest_distance
from newton._src.geometry.support_function import GenericShapeData, SupportMapDataProvider, support_map
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _query_box_gap(gap: float, direction: float, output: wp.array[float]):
    """Query two aligned boxes with a known signed surface gap."""
    a = GenericShapeData()
    a.shape_type = int(GeoType.BOX)
    a.scale = wp.vec3(0.01, 0.02, 0.03)
    b = a
    separated, pa, pb, normal, distance = wp.static(create_solve_closest_distance(support_map).core)(
        a,
        b,
        wp.quat_identity(),
        wp.vec3(direction * (0.02 + gap), 0.0, 0.0),
        0.0,
        SupportMapDataProvider(),
    )
    output[0] = float(separated)
    output[1] = distance
    for axis in range(3):
        output[2 + axis] = normal[axis]
        output[5 + axis] = pb[axis] - pa[axis]


@wp.kernel
def _query_tiny_gap(shape_type: int, scale: wp.vec3, positions: wp.array[wp.vec3], output: wp.array2d[float]):
    """Query pairs with representable nanometer gaps and retain both witnesses."""
    i = wp.tid()
    a = GenericShapeData()
    a.shape_type = shape_type
    a.scale = scale
    separated, pa, pb, normal, distance = wp.static(create_solve_closest_distance(support_map).core)(
        a, a, wp.quat_identity(), positions[i], 0.0, SupportMapDataProvider()
    )
    output[i, 0] = float(separated)
    output[i, 1] = distance
    for axis in range(3):
        output[i, 2 + axis] = normal[axis]
        output[i, 5 + axis] = pa[axis]
        output[i, 8 + axis] = pb[axis]


@wp.kernel
def _query_rotated_box(output: wp.array2d[float]):
    """Place a small rotated box at a known gap above a much larger box."""
    i = wp.tid()
    a = GenericShapeData()
    a.shape_type = int(GeoType.BOX)
    a.scale = wp.vec3(0.8, 0.6, 0.02)
    b = GenericShapeData()
    b.shape_type = int(GeoType.BOX)
    b.scale = wp.vec3(0.03, 0.04, 0.07)
    rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.7, 0.2)), float(i % 100) * 0.03)
    height = (
        wp.abs(wp.quat_rotate(rotation, wp.vec3(0.03, 0.0, 0.0))[2])
        + wp.abs(wp.quat_rotate(rotation, wp.vec3(0.0, 0.04, 0.0))[2])
        + wp.abs(wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 0.07))[2])
    )
    gap = float(i // 100 + 1) * 1e-5
    separated, _, _, normal, distance = wp.static(create_solve_closest_distance(support_map).core)(
        a, b, rotation, wp.vec3(0.17, 0.13, 0.02 + height + gap), 0.0, SupportMapDataProvider()
    )
    output[i, 0] = float(separated)
    output[i, 1] = distance
    output[i, 2] = gap
    for axis in range(3):
        output[i, 3 + axis] = normal[axis]


def test_positive_sub_tolerance_gap(test, device):
    """Return positive gaps and oriented unit normals below the 0.1 mm convergence tolerance."""
    for gap in (1e-3, 1.1e-4, 9.5e-5, 5e-5, 1e-5, 1e-6, 1e-7, 5e-9):
        for direction in (-1.0, 1.0):
            with test.subTest(gap=gap, direction=direction):
                out = wp.zeros(8, dtype=float, device=device)
                wp.launch(_query_box_gap, dim=1, inputs=[gap, direction], outputs=[out], device=device)
                actual = out.numpy()
                test.assertEqual(actual[0], 1.0)
                test.assertAlmostEqual(float(actual[1]), gap, delta=3e-9)
                np.testing.assert_allclose(actual[2:5], [direction, 0.0, 0.0], atol=1e-6)
                np.testing.assert_allclose(actual[5:8], [direction * gap, 0.0, 0.0], atol=3e-9)


def test_tiny_gap_with_tangential_offsets(test, device):
    """Preserve signed nanometer gaps when offset boxes need edge or face refinement."""
    for gap in (-5e-9, 0.0, 5e-9):
        positions = np.array(
            [
                [direction * (np.float32(0.02) + np.float32(gap)), y, z]
                for direction in (-1.0, 1.0)
                for y in (-0.035, -0.0315, 0.0, 0.014, 0.035)
                for z in (-0.055, -0.022, 0.0, 0.033, 0.055)
            ],
            dtype=np.float32,
        )
        output = wp.zeros((len(positions), 11), dtype=float, device=device)
        wp.launch(
            _query_tiny_gap,
            dim=len(positions),
            inputs=[int(GeoType.BOX), wp.vec3(0.01, 0.02, 0.03), wp.array(positions, dtype=wp.vec3, device=device)],
            outputs=[output],
            device=device,
        )
        actual = output.numpy()
        test.assertTrue(np.isfinite(actual).all())
        np.testing.assert_array_equal(actual[:, 0], float(gap > 0.0))
        if gap > 0.0:
            expected_gap = np.abs(positions[:, 0]) - np.float32(0.02)
            np.testing.assert_allclose(actual[:, 1], expected_gap, atol=1e-12, rtol=1e-5)
            normals = np.zeros_like(positions)
            normals[:, 0] = np.sign(positions[:, 0])
            np.testing.assert_allclose(actual[:, 2:5], normals, atol=1e-6)
        else:
            np.testing.assert_array_equal(actual[:, 1], 0.0)


def test_tiny_initial_center_displacement(test, device):
    """Build a simplex for separated points whose initial displacement is below epsilon."""
    positions = np.array(
        [
            [direction * gap if axis == coordinate else 0.0 for coordinate in range(3)]
            for gap in (1e-9, 5e-9, 1e-8)
            for direction in (-1.0, 1.0)
            for axis in range(3)
        ],
        dtype=np.float32,
    )
    output = wp.zeros((len(positions), 11), dtype=float, device=device)
    wp.launch(
        _query_tiny_gap,
        dim=len(positions),
        inputs=[int(GeoType.SPHERE), wp.vec3(0.0), wp.array(positions, dtype=wp.vec3, device=device)],
        outputs=[output],
        device=device,
    )
    actual = output.numpy()
    distances = np.linalg.norm(positions, axis=1)
    np.testing.assert_array_equal(actual[:, 0], 1.0)
    np.testing.assert_allclose(actual[:, 1], distances, atol=1e-15, rtol=1e-6)
    np.testing.assert_allclose(actual[:, 2:5], positions / distances[:, None], atol=1e-6)
    np.testing.assert_array_equal(actual[:, 5:8], 0.0)
    np.testing.assert_allclose(actual[:, 8:11], positions, atol=1e-15, rtol=1e-6)


def test_true_overlap(test, device):
    """Keep the overlap classification for touching and penetrating boxes."""
    for gap in (-1e-3, -1e-5, -5e-9, 0.0):
        with test.subTest(gap=gap):
            out = wp.zeros(8, dtype=float, device=device)
            wp.launch(_query_box_gap, dim=1, inputs=[gap, 1.0], outputs=[out], device=device)
            actual = out.numpy()
            test.assertEqual(actual[0], 0.0)
            test.assertEqual(actual[1], 0.0)


def test_rotated_box_above_large_face(test, device):
    """Resolve sub-tolerance gaps above a large face without cancellation error."""
    output = wp.zeros((1000, 6), dtype=float, device=device)
    wp.launch(_query_rotated_box, dim=1000, outputs=[output], device=device)
    actual = output.numpy()
    np.testing.assert_array_equal(actual[:, 0], 1.0)
    np.testing.assert_allclose(actual[:, 1], actual[:, 2], atol=2e-7, rtol=0.0)
    np.testing.assert_allclose(np.linalg.norm(actual[:, 3:6], axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(actual[:900, 3:6], np.tile([0.0, 0.0, 1.0], (900, 1)), atol=1e-5)
    # The 0.1 mm boundary can round outside the near-contact branch and
    # keeps the witness-difference normal.
    np.testing.assert_allclose(actual[900:, 3:6], np.tile([0.0, 0.0, 1.0], (100, 1)), atol=2e-3)


class TestGJKNearContact(unittest.TestCase):
    """Preserve geometric separation independently of the convergence tolerance."""


devices = get_test_devices()
for _test in (
    test_positive_sub_tolerance_gap,
    test_tiny_gap_with_tangential_offsets,
    test_tiny_initial_center_displacement,
    test_true_overlap,
    test_rotated_box_above_large_face,
):
    add_function_test(TestGJKNearContact, _test.__name__, _test, devices=devices)


if __name__ == "__main__":
    unittest.main()
