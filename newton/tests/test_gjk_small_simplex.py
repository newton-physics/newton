# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test scale-relative degeneracy checks for small GJK simplex faces."""

import unittest

import numpy as np
import warp as wp

from newton import GeoType
from newton._src.geometry.simplex_solver import create_solve_closest_distance
from newton._src.geometry.support_function import (
    GenericShapeData,
    GeoTypeEx,
    SupportMapDataProvider,
    support_map,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _query_triangle_point(a: wp.vec3, b: wp.vec3, c: wp.vec3, out: wp.array[float]):
    """Query a triangle against a zero-radius sphere at the origin."""
    triangle = GenericShapeData()
    triangle.shape_type = int(GeoTypeEx.TRIANGLE)
    triangle.scale = b - a
    triangle.auxiliary = c - a
    triangle.center = (b + c - 2.0 * a) / 3.0
    point = GenericShapeData()
    point.shape_type = int(GeoType.SPHERE)
    point.scale = wp.vec3(0.0)
    # Keep the overlap and convergence tolerance below the face gap so this
    # query isolates simplex degeneracy rather than near-contact termination.
    separated, _, _, normal, distance = wp.static(create_solve_closest_distance(support_map).core)(
        triangle, point, wp.quat_identity(), -a, 0.0, SupportMapDataProvider(), 30, 1e-6
    )
    out[0] = float(separated)
    out[1] = distance
    for axis in range(3):
        out[2 + axis] = normal[axis]


def test_triangle_interior_distance_is_scale_relative(test, device):
    """Retain the interior face distance and normal across triangle scales."""
    # The origin projects inside this 6 mm by 7 mm right triangle, 40 um away.
    # Its squared area is 1.764e-9 m^4, but its squared sine is 1.
    vertices = np.array(
        [[-0.002, -0.003, 0.00004], [0.004, -0.003, 0.00004], [-0.002, 0.004, 0.00004]],
        dtype=np.float64,
    )
    for scale in (0.5, 1.0, 2.0):
        for axes in ((0, 1, 2), (2, 0, 1), (1, 2, 0)):
            with test.subTest(scale=scale, axes=axes):
                a, b, c = vertices[:, axes] * scale
                normal = np.cross(b - a, c - a)
                normal /= np.linalg.norm(normal)
                signed_distance = float(normal @ a)
                closest = signed_distance * normal
                weights = np.linalg.lstsq(np.stack([b - a, c - a], axis=1), closest - a, rcond=None)[0]
                test.assertGreater(float(weights.min()), 0.0)
                test.assertLess(float(weights.sum()), 1.0)
                expected_normal = -closest / np.linalg.norm(closest)
                output = wp.zeros(5, dtype=float, device=device)
                wp.launch(
                    _query_triangle_point,
                    dim=1,
                    inputs=[wp.vec3(*vertex) for vertex in (a, b, c)],
                    outputs=[output],
                    device=device,
                )
                actual = output.numpy()
                test.assertEqual(actual[0], 1.0)
                test.assertAlmostEqual(float(actual[1]), abs(signed_distance), delta=1e-7 * scale)
                np.testing.assert_allclose(actual[2:5], expected_normal, atol=2e-5)


def test_short_segment_distance(test, device):
    """Resolve the interior distance of simplex edges shorter than 0.1 mm."""
    vertices = np.array([[-2e-5, 0.0, 4e-5], [2e-5, 0.0, 4e-5], [0.0, 3e-5, 4e-5]])
    for scale in (0.5, 1.0, 2.0):
        with test.subTest(scale=scale):
            output = wp.zeros(5, dtype=float, device=device)
            wp.launch(
                _query_triangle_point,
                dim=1,
                inputs=[wp.vec3(*vertex) for vertex in vertices * scale],
                outputs=[output],
                device=device,
            )
            actual = output.numpy()
            test.assertEqual(actual[0], 1.0)
            test.assertAlmostEqual(float(actual[1]), 4e-5 * scale, delta=1e-9 * scale)
            np.testing.assert_allclose(actual[2:5], [0.0, 0.0, -1.0], atol=1e-6)


class TestGJKSmallSimplex(unittest.TestCase):
    """Preserve small, nondegenerate simplex faces during distance queries."""


add_function_test(
    TestGJKSmallSimplex,
    "test_triangle_interior_distance_is_scale_relative",
    test_triangle_interior_distance_is_scale_relative,
    devices=get_test_devices(),
)
add_function_test(
    TestGJKSmallSimplex,
    "test_short_segment_distance",
    test_short_segment_distance,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
