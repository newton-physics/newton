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


@wp.kernel(module="unique")
def _query_separated_spheres_with_cutoff(cutoff: float, output: wp.array[float]):
    """Query tiny spheres separated by much more than their radius, with a separation cutoff."""
    a = GenericShapeData()
    a.shape_type = int(GeoType.SPHERE)
    a.scale = wp.vec3(0.001, 0.0, 0.0)
    separated, point_a, point_b, normal, distance = wp.static(create_solve_closest_distance(support_map).core)(
        a, a, wp.quat_identity(), wp.vec3(100.0, 0.0, 0.0), 0.0, SupportMapDataProvider(), max_dist=cutoff
    )
    output[0] = float(separated)
    for axis in range(3):
        output[1 + axis] = point_a[axis]
        output[4 + axis] = point_b[axis]
        output[7 + axis] = normal[axis]
    output[10] = distance


@wp.func
def _write_query(
    output: wp.array2d[float],
    row: int,
    separated: bool,
    point_a: wp.vec3,
    point_b: wp.vec3,
    normal: wp.vec3,
    distance: float,
):
    output[row, 0] = float(separated)
    output[row, 1] = distance
    for axis in range(3):
        output[row, 2 + axis] = point_a[axis]
        output[row, 5 + axis] = point_b[axis]
        output[row, 8 + axis] = normal[axis]


@wp.kernel(module="unique")
def _query_hull_pairs_with_cutoff(
    shape_types: wp.array[int],
    scales: wp.array[wp.vec3],
    rotations: wp.array[wp.quat],
    positions: wp.array[wp.vec3],
    cutoff: float,
    exact: wp.array2d[float],
    cut: wp.array2d[float],
):
    """Query each pair once without and once with the separation cutoff."""
    i = wp.tid()
    a = GenericShapeData()
    a.shape_type = shape_types[2 * i]
    a.scale = scales[2 * i]
    b = GenericShapeData()
    b.shape_type = shape_types[2 * i + 1]
    b.scale = scales[2 * i + 1]
    gjk = wp.static(create_solve_closest_distance(support_map).core)
    separated, point_a, point_b, normal, distance = gjk(a, b, rotations[i], positions[i], 0.0, SupportMapDataProvider())
    _write_query(exact, i, separated, point_a, point_b, normal, distance)
    separated, point_a, point_b, normal, distance = gjk(
        a, b, rotations[i], positions[i], 0.0, SupportMapDataProvider(), max_dist=cutoff
    )
    _write_query(cut, i, separated, point_a, point_b, normal, distance)


@wp.kernel(module="unique")
def _query_positional_arguments(cutoff: float, output: wp.array2d[float]):
    """Bind the trailing optional arguments positionally and by keyword."""
    a = GenericShapeData()
    a.shape_type = int(GeoType.BOX)
    a.scale = wp.vec3(0.1, 0.2, 0.3)
    b = GenericShapeData()
    b.shape_type = int(GeoType.BOX)
    b.scale = wp.vec3(0.3, 0.1, 0.2)
    rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 1.0, 0.5)), 0.7)
    position = wp.vec3(0.9, 0.4, 0.2)
    provider = SupportMapDataProvider()
    gjk = wp.static(create_solve_closest_distance(support_map).core)
    separated, point_a, point_b, normal, distance = gjk(a, b, rotation, position, 0.0, provider)
    _write_query(output, 0, separated, point_a, point_b, normal, distance)
    separated, point_a, point_b, normal, distance = gjk(a, b, rotation, position, 0.0, provider, 30, 1e-4)
    _write_query(output, 1, separated, point_a, point_b, normal, distance)
    separated, point_a, point_b, normal, distance = gjk(a, b, rotation, position, 0.0, provider, 1, 1e-4)
    _write_query(output, 2, separated, point_a, point_b, normal, distance)
    separated, point_a, point_b, normal, distance = gjk(
        a, b, rotation, position, 0.0, provider, MAX_ITER=1, COLLIDE_EPSILON=1e-4
    )
    _write_query(output, 3, separated, point_a, point_b, normal, distance)
    separated, point_a, point_b, normal, distance = gjk(a, b, rotation, position, 0.0, provider, 30, 1e-4, cutoff)
    _write_query(output, 4, separated, point_a, point_b, normal, distance)
    separated, point_a, point_b, normal, distance = gjk(a, b, rotation, position, 0.0, provider, max_dist=cutoff)
    _write_query(output, 5, separated, point_a, point_b, normal, distance)


def test_positive_sub_tolerance_gap(test, device):
    """Return positive gaps and oriented unit normals below the 0.1 mm convergence tolerance."""
    for gap in (1e-3, 1.1e-4, 9.5e-5, 5e-5, 1e-5, 1e-6, 1e-7):
        for direction in (-1.0, 1.0):
            with test.subTest(gap=gap, direction=direction):
                out = wp.zeros(8, dtype=float, device=device)
                wp.launch(_query_box_gap, dim=1, inputs=[gap, direction], outputs=[out], device=device)
                actual = out.numpy()
                test.assertEqual(actual[0], 1.0)
                test.assertAlmostEqual(float(actual[1]), gap, delta=3e-9)
                np.testing.assert_allclose(actual[2:5], [direction, 0.0, 0.0], atol=1e-6)
                np.testing.assert_allclose(actual[5:8], [direction * gap, 0.0, 0.0], atol=3e-9)


def test_true_overlap(test, device):
    """Keep the overlap classification for touching and penetrating boxes."""
    for gap in (-1e-3, -1e-5, 0.0):
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


def test_early_exit_preserves_separated_sphere_witnesses(test, device):
    """Return surface witnesses whether or not the separation cutoff stops GJK."""
    for cutoff in (0.0, 1.0):
        with test.subTest(cutoff=cutoff):
            output = wp.zeros(11, dtype=float, device=device)
            wp.launch(_query_separated_spheres_with_cutoff, dim=1, inputs=[cutoff], outputs=[output], device=device)
            actual = output.numpy()
            test.assertEqual(actual[0], 1.0)
            np.testing.assert_allclose(actual[1:4], [0.001, 0.0, 0.0], atol=8e-6, rtol=0.0)
            np.testing.assert_allclose(actual[4:7], [99.999, 0.0, 0.0], atol=8e-6, rtol=0.0)
            np.testing.assert_allclose(actual[7:10], [1.0, 0.0, 0.0], atol=1e-6)
            test.assertAlmostEqual(float(actual[10]), 99.998, delta=8e-6)
            test.assertAlmostEqual(float(actual[10]), float(np.linalg.norm(actual[4:7] - actual[1:4])), delta=8e-6)


def test_separation_cutoff_matches_exact_query_within_cutoff(test, device):
    """Keep results within the cutoff bit-identical and report pairs beyond it as farther than the cutoff."""
    rng = np.random.default_rng(11)
    count, cutoff = 4096, 0.02
    types = rng.choice([int(GeoType.BOX), int(GeoType.CYLINDER), int(GeoType.CAPSULE), int(GeoType.CONE)], 2 * count)
    scales = rng.uniform(0.02, 0.2, size=(2 * count, 3))
    # Only boxes use a third extent; a nonzero cylinder scale[2] would select a barrel profile.
    scales[types != int(GeoType.BOX), 2] = 0.0
    axes = rng.normal(size=(count, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = rng.uniform(0.0, np.pi, count)
    rotations = np.concatenate([axes * np.sin(0.5 * angles)[:, None], np.cos(0.5 * angles)[:, None]], axis=1)
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    positions = directions * rng.uniform(0.0, 0.6, count)[:, None]
    exact = wp.zeros((count, 11), dtype=float, device=device)
    cut = wp.zeros((count, 11), dtype=float, device=device)
    wp.launch(
        _query_hull_pairs_with_cutoff,
        dim=count,
        inputs=[
            wp.array(types, dtype=int, device=device),
            wp.array(scales, dtype=wp.vec3, device=device),
            wp.array(rotations, dtype=wp.quat, device=device),
            wp.array(positions, dtype=wp.vec3, device=device),
            cutoff,
        ],
        outputs=[exact, cut],
        device=device,
    )
    exact, cut = exact.numpy(), cut.numpy()
    within = exact[:, 1] <= cutoff
    beyond = ~within
    # Guard the sample: both sides of the cutoff and overlapping pairs are present.
    test.assertGreater(int(np.count_nonzero(exact[:, 0] == 0.0)), 100)
    test.assertGreater(int(np.count_nonzero(within & (exact[:, 0] == 1.0))), 100)
    test.assertGreater(int(np.count_nonzero(beyond)), 1000)
    np.testing.assert_array_equal(cut[within], exact[within])
    np.testing.assert_array_equal(cut[beyond, 0], 1.0)
    test.assertTrue(np.all(cut[beyond, 1] > cutoff))
    # The cutoff distance is the current simplex distance, an upper bound on the exact one.
    np.testing.assert_array_less(exact[beyond, 1] - 1e-5, cut[beyond, 1])
    # The cutoff does stop early for some pairs: their distance is not yet refined.
    test.assertGreater(int(np.count_nonzero(cut[beyond, 1] > exact[beyond, 1] + 1e-4)), 10)


def test_positional_iteration_arguments_keep_their_meaning(test, device):
    """Bind positional MAX_ITER and COLLIDE_EPSILON as before the cutoff argument was added."""
    output = wp.zeros((6, 11), dtype=float, device=device)
    wp.launch(_query_positional_arguments, dim=1, inputs=[0.05], outputs=[output], device=device)
    default, positional, one_iter, one_iter_keyword, positional_cutoff, keyword_cutoff = output.numpy()
    np.testing.assert_array_equal(positional, default)
    np.testing.assert_array_equal(one_iter, one_iter_keyword)
    np.testing.assert_array_equal(positional_cutoff, keyword_cutoff)
    # One iteration leaves an unrefined, larger distance; the default converges.
    test.assertGreater(float(one_iter[1]), float(default[1]) + 1e-3)


class TestGJKNearContact(unittest.TestCase):
    """Preserve geometric separation independently of the convergence tolerance."""


devices = get_test_devices()
for _test in (
    test_positive_sub_tolerance_gap,
    test_true_overlap,
    test_rotated_box_above_large_face,
    test_early_exit_preserves_separated_sphere_witnesses,
    test_separation_cutoff_matches_exact_query_within_cutoff,
    test_positional_iteration_arguments_keep_their_meaning,
):
    add_function_test(TestGJKNearContact, _test.__name__, _test, devices=devices)


if __name__ == "__main__":
    unittest.main()
