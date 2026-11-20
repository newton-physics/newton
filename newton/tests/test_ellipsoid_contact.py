# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytic reference contacts for the fixed-storage ellipsoid query."""

import itertools
import unittest

import numpy as np
import warp as wp

from newton._src.geometry.ellipsoid_contact import collide_ellipsoids
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel(enable_backward=False)
def _query_pairs(
    radii_a: wp.array[wp.vec3],
    radii_b: wp.array[wp.vec3],
    rotations: wp.array[wp.quat],
    positions: wp.array[wp.vec3],
    solved: wp.array[bool],
    distances: wp.array[float],
    centers: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
):
    i = wp.tid()
    success, distance, center, normal = collide_ellipsoids(radii_a[i], radii_b[i], rotations[i], positions[i])
    solved[i] = success
    distances[i] = distance
    centers[i] = center
    normals[i] = normal


def test_constructed_contacts(test, device):
    """Verify signed distances and reject uncertified local penetration minima."""
    inputs = []
    references = []
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, -2.0, 3.0]])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    cases = itertools.product(
        (False, True), (0.0, 0.63, 1.27), directions, (0.001, 1.0, 1000.0), (-0.01, 0.0001, 0.025)
    )

    def add_case(a, b, angle, normal, scale, depth, allow_fallback):
        a = np.asarray(a)
        b = np.asarray(b)
        normal = np.asarray(normal) / np.linalg.norm(normal)
        rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 3.0)), angle)
        matrix = np.asarray(wp.quat_to_matrix(rotation), dtype=np.float64).reshape(3, 3)
        point_a = a * a * normal / np.linalg.norm(a * normal)
        local_normal_b = -matrix.T @ normal
        point_b = b * b * local_normal_b / np.linalg.norm(b * local_normal_b)
        position = (point_a - matrix @ point_b - depth * normal) * scale
        inputs.append((a * scale, b * scale, rotation, position))
        references.append((normal, scale, depth, matrix, allow_fallback))

    for sphere, angle, normal, scale, depth in cases:
        a = np.array([0.6, 0.6, 0.6] if sphere else [0.4, 0.7, 1.0])
        b = np.array([0.8, 0.5, 0.6])
        add_case(a, b, angle, normal, scale, depth, allow_fallback=depth < 0.0)
    # The measured rotated near-gap workload must avoid a second convex query.
    add_case([0.5] * 3, [1.0, 0.6, 0.3], 0.71, [1.0, 1.7, 0.4], 1.0, -0.005, allow_fallback=False)
    # This pair has multiple local minima. The reference direction gives a
    # shallower separating plane than the first local solution (about 0.72966).
    # Rejecting the pair is valid; accepting that deeper solution is not.
    rotation = wp.quat(0.84911776, 0.18545088, -0.20776686, 0.44882065)
    a = np.array([0.81310971, 0.42758673, 0.31684289])
    b = np.array([0.81929639, 0.24356250, 0.73067252])
    position = np.array([0.08802737, -0.02949503, -0.08029378])
    normal = np.array([-0.27924614, 0.50044898, -0.81949522])
    normal /= np.linalg.norm(normal)
    matrix = np.asarray(wp.quat_to_matrix(rotation), dtype=np.float64).reshape(3, 3)
    depth = np.linalg.norm(a * normal) + np.linalg.norm(b * (matrix.T @ normal)) - np.dot(position, normal)
    inputs.append((a, b, rotation, position))
    references.append((normal, 1.0, depth, matrix, True))
    a, b, rotations, positions = zip(*inputs, strict=True)
    count = len(inputs)
    solved = wp.zeros(count, dtype=bool, device=device)
    distances = wp.zeros(count, device=device)
    centers = wp.zeros(count, dtype=wp.vec3, device=device)
    normals = wp.zeros(count, dtype=wp.vec3, device=device)
    wp.launch(
        _query_pairs,
        count,
        [
            wp.array(a, dtype=wp.vec3, device=device),
            wp.array(b, dtype=wp.vec3, device=device),
            wp.array(rotations, dtype=wp.quat, device=device),
            wp.array(positions, dtype=wp.vec3, device=device),
            solved,
            distances,
            centers,
            normals,
        ],
        device=device,
    )
    successes, distances, centers, normals = solved.numpy(), distances.numpy(), centers.numpy(), normals.numpy()
    for i, (normal, scale, depth, matrix, allow_fallback) in enumerate(references):
        with test.subTest(case=i, scale=scale, depth=depth):
            if allow_fallback and not successes[i]:
                continue
            test.assertTrue(successes[i])
            test.assertAlmostEqual(distances[i] / scale, -depth, delta=3e-6)
            np.testing.assert_allclose(normals[i], normal, atol=3e-3)
            np.testing.assert_allclose(np.linalg.norm(normals[i]), 1.0, atol=1e-6)
            pa = centers[i] - 0.5 * distances[i] * normals[i]
            pb = matrix.T @ (centers[i] + 0.5 * distances[i] * normals[i] - positions[i])
            test.assertAlmostEqual(np.linalg.norm(pa / a[i]), 1.0, delta=1e-5)
            test.assertAlmostEqual(np.linalg.norm(pb / b[i]), 1.0, delta=1e-5)


class TestEllipsoidContact(unittest.TestCase):
    pass


add_function_test(
    TestEllipsoidContact, "test_constructed_contacts", test_constructed_contacts, devices=get_test_devices()
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
