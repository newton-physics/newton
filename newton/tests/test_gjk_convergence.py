# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test GJK distance convergence and surface witnesses."""

import unittest

import numpy as np
import warp as wp

from newton import GeoType
from newton._src.geometry.simplex_solver import create_solve_closest_distance
from newton._src.geometry.support_function import GenericShapeData, SupportMapDataProvider, support_map
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _query_sphere_box(scale: float, center: wp.vec3, output: wp.array[float]):
    """Query the distance and normal from a box to a sphere."""
    box = GenericShapeData()
    box.shape_type = int(GeoType.BOX)
    box.scale = scale * wp.vec3(0.02, 0.015, 0.01)
    sphere = GenericShapeData()
    sphere.shape_type = int(GeoType.SPHERE)
    sphere.scale = wp.vec3(scale * 0.01, 0.0, 0.0)
    separated, _, _, normal, distance = wp.static(create_solve_closest_distance(support_map).core)(
        box, sphere, wp.quat_identity(), center, 0.0, SupportMapDataProvider()
    )
    output[0] = float(separated)
    output[1] = distance
    for axis in range(3):
        output[2 + axis] = normal[axis]


@wp.kernel
def _query_distant_spheres(radius: float, center: wp.vec3, output: wp.array[float]):
    """Return surface witnesses for two small spheres separated by a large distance."""
    sphere = GenericShapeData()
    sphere.shape_type = int(GeoType.SPHERE)
    sphere.scale = wp.vec3(radius, 0.0, 0.0)
    separated, point_a, point_b, _, distance = wp.static(create_solve_closest_distance(support_map).core)(
        sphere, sphere, wp.quat_identity(), center, 0.0, SupportMapDataProvider()
    )
    output[0] = float(separated)
    output[1] = distance
    for axis in range(3):
        output[2 + axis] = point_a[axis]
        output[5 + axis] = point_b[axis]


def test_sphere_box_distance_convergence(test, device):
    """Resolve small gaps at box faces, edges, and corners against analytic distances."""
    features = (
        ((0.02, 0.007, -0.003), (1.0, 0.0, 0.0)),
        ((0.02, 0.015, 0.004), (0.6, 0.8, 0.0)),
        ((0.02, -0.015, 0.01), (0.4, -0.7, 0.5)),
    )
    for feature, direction in features:
        normal = np.asarray(direction, dtype=np.float64)
        normal /= np.linalg.norm(normal)
        for scale in (0.5, 1.0, 2.0):
            for gap in (0.0004, 0.001, 0.003):
                with test.subTest(feature=feature, scale=scale, gap=gap):
                    center = scale * (np.asarray(feature) + (0.01 + gap) * normal)
                    output = wp.zeros(5, dtype=float, device=device)
                    wp.launch(_query_sphere_box, dim=1, inputs=[scale, wp.vec3(*center), output], device=device)
                    actual = output.numpy()
                    test.assertEqual(actual[0], 1.0)
                    test.assertAlmostEqual(float(actual[1]), scale * gap, delta=2e-6 * scale)
                    cosine = np.clip(np.dot(actual[2:5], normal), -1.0, 1.0)
                    test.assertLess(float(np.degrees(np.arccos(cosine))), 1.0)


def test_distant_spheres_have_surface_witnesses(test, device):
    """Populate the simplex before accepting convergence for distant small shapes."""
    radius = 0.001
    center = np.array([100.0, 0.0, 0.0])
    output = wp.zeros(8, dtype=float, device=device)
    wp.launch(_query_distant_spheres, dim=1, inputs=[radius, wp.vec3(*center), output], device=device)
    actual = output.numpy()
    test.assertEqual(actual[0], 1.0)
    test.assertAlmostEqual(float(actual[1]), 100.0 - 2.0 * radius, delta=2e-5)
    test.assertAlmostEqual(float(np.linalg.norm(actual[2:5])), radius, delta=1e-5)
    test.assertAlmostEqual(float(np.linalg.norm(actual[5:8] - center)), radius, delta=1e-5)


class TestGJKConvergence(unittest.TestCase):
    """Validate convergence independently of simplex degeneracy and overlap thresholds."""


devices = get_test_devices()
add_function_test(
    TestGJKConvergence, "test_sphere_box_distance_convergence", test_sphere_box_distance_convergence, devices
)
add_function_test(
    TestGJKConvergence,
    "test_distant_spheres_have_surface_witnesses",
    test_distant_spheres_have_surface_witnesses,
    devices,
)


if __name__ == "__main__":
    unittest.main()
