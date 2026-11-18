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

_F32_EPS = float(np.finfo(np.float32).eps)


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


@wp.kernel
def _query_capsule_pairs(
    scales: wp.array[wp.vec3],
    rotations: wp.array[wp.quat],
    positions: wp.array[wp.vec3],
    distances: wp.array[float],
):
    """Query the distance between each pair of capsules."""
    i = wp.tid()
    a = GenericShapeData()
    a.shape_type = int(GeoType.CAPSULE)
    a.scale = scales[2 * i]
    b = GenericShapeData()
    b.shape_type = int(GeoType.CAPSULE)
    b.scale = scales[2 * i + 1]
    _separated, _point_a, _point_b, _normal, distance = wp.static(create_solve_closest_distance(support_map).core)(
        a, b, rotations[i], positions[i], 0.0, SupportMapDataProvider()
    )
    distances[i] = distance


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
            for gap in (1e-5, 5e-5, 9.5e-5, 0.0004, 0.001, 0.003):
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


def _segment_distances(p1, q1, p2, q2):
    """Return float64 distances between non-degenerate segment pairs (Ericson, RTCD 5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = np.einsum("ij,ij->i", d1, d1), np.einsum("ij,ij->i", d2, d2)
    b, c, f = np.einsum("ij,ij->i", d1, d2), np.einsum("ij,ij->i", d1, r), np.einsum("ij,ij->i", d2, r)
    denom = a * e - b * b
    s = np.where(denom > 1e-30, np.clip((b * f - c * e) / np.where(denom > 1e-30, denom, 1.0), 0.0, 1.0), 0.0)
    t = (b * s + f) / e
    s = np.where(t < 0.0, np.clip(-c / a, 0.0, 1.0), np.where(t > 1.0, np.clip((b - c) / a, 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm(p1 + d1 * s[:, None] - p2 - d2 * t[:, None], axis=1)


def test_distance_converges_to_the_analytic_capsule_distance(test, device):
    """Converge separated capsule distances to float32 rounding of the coordinates involved."""
    rng = np.random.default_rng(3)
    count = 1024
    scales = np.zeros((2 * count, 3), dtype=np.float32)
    scales[:, 0] = rng.uniform(0.02, 0.3, 2 * count)
    scales[:, 1] = rng.uniform(0.02, 0.3, 2 * count)
    rotations = rng.normal(size=(count, 4))
    rotations = (rotations / np.linalg.norm(rotations, axis=1, keepdims=True)).astype(np.float32)
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    positions = (directions * rng.uniform(0.3, 1.2, count)[:, None]).astype(np.float32)

    # Analytic distances of the float32 inputs; both capsule axes are local +Z.
    radius = scales[:, 0].astype(np.float64).reshape(count, 2)
    half = scales[:, 1].astype(np.float64).reshape(count, 2, 1)
    x, y, z, w = rotations.astype(np.float64).T
    axis_b = np.stack([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)], axis=1)
    axis_a = np.array([0.0, 0.0, 1.0])
    p = positions.astype(np.float64)
    distance = _segment_distances(
        -half[:, 0] * axis_a, half[:, 0] * axis_a, p - half[:, 1] * axis_b, p + half[:, 1] * axis_b
    )
    distance -= radius.sum(axis=1)
    separated = distance > 1e-3
    test.assertGreater(int(np.count_nonzero(separated)), 200)

    output = wp.zeros(count, dtype=float, device=device)
    wp.launch(
        _query_capsule_pairs,
        dim=count,
        inputs=[
            wp.array(scales, dtype=wp.vec3, device=device),
            wp.array(rotations, dtype=wp.quat, device=device),
            wp.array(positions, dtype=wp.vec3, device=device),
        ],
        outputs=[output],
        device=device,
    )
    # A relative duality gap of 1e-4 left errors of several hundred eps * scale here.
    tolerance = 32.0 * _F32_EPS * (np.abs(p).sum(axis=1) + np.abs(scales).reshape(count, 6).sum(axis=1))
    error = np.abs(output.numpy()[separated] - distance[separated])
    np.testing.assert_array_less(error, tolerance[separated])


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
add_function_test(
    TestGJKConvergence,
    "test_distance_converges_to_the_analytic_capsule_distance",
    test_distance_converges_to_the_analytic_capsule_distance,
    devices,
)


if __name__ == "__main__":
    unittest.main()
