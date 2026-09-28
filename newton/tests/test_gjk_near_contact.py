# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regress positive convex gaps smaller than the distance convergence tolerance."""

import unittest

import numpy as np
import warp as wp

from newton import GeoType
from newton._src.geometry.simplex_solver import create_solve_closest_distance
from newton._src.geometry.support_function import GenericShapeData, SupportMapDataProvider, support_map


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


class TestGJKNearContact(unittest.TestCase):
    """Preserve geometric separation independently of convergence tolerance."""

    def test_positive_sub_tolerance_gap(self):
        """Return actual positive gaps and oriented unit normals below 100 micrometers."""
        for device in wp.get_devices():
            for gap in (1e-3, 1.1e-4, 9.5e-5, 5e-5, 1e-5, 1e-6, 1e-7):
                for direction in (-1.0, 1.0):
                    with self.subTest(device=str(device), gap=gap, direction=direction):
                        out = wp.zeros(8, dtype=float, device=device)
                        wp.launch(_query_box_gap, dim=1, inputs=[gap, direction], outputs=[out], device=device)
                        actual = out.numpy()
                        self.assertEqual(actual[0], 1.0)
                        self.assertAlmostEqual(float(actual[1]), gap, delta=3e-9)
                        np.testing.assert_allclose(actual[2:5], [direction, 0.0, 0.0], atol=1e-6)
                        np.testing.assert_allclose(actual[5:8], [direction * gap, 0.0, 0.0], atol=3e-9)

    def test_true_overlap(self):
        """Keep the distance query's overlap classification for penetrating boxes."""
        for device in wp.get_devices():
            for gap in (-1e-3, -1e-5, 0.0):
                with self.subTest(device=str(device), gap=gap):
                    out = wp.zeros(8, dtype=float, device=device)
                    wp.launch(_query_box_gap, dim=1, inputs=[gap, 1.0], outputs=[out], device=device)
                    actual = out.numpy()
                    self.assertEqual(actual[0], 0.0)
                    self.assertEqual(actual[1], 0.0)

    def test_rotated_box_above_large_face(self):
        """Refine sub-tolerance gaps without cancellation along a large face."""
        for device in wp.get_devices():
            output = wp.zeros((1000, 6), dtype=float, device=device)
            wp.launch(_query_rotated_box, dim=1000, outputs=[output], device=device)
            actual = output.numpy()
            np.testing.assert_array_equal(actual[:, 0], 1.0)
            np.testing.assert_allclose(actual[:, 1], actual[:, 2], atol=2e-7, rtol=0.0)
            np.testing.assert_allclose(np.linalg.norm(actual[:, 3:6], axis=1), 1.0, atol=1e-6)
            np.testing.assert_allclose(actual[:900, 3:6], np.tile([0.0, 0.0, 1.0], (900, 1)), atol=1e-5)
            # The 100-micrometer boundary can round outside the near-contact
            # branch and retains the pre-existing witness-difference normal.
            np.testing.assert_allclose(actual[900:, 3:6], np.tile([0.0, 0.0, 1.0], (100, 1)), atol=2e-3)

    def test_millimeter_gap_converges_relative_to_distance(self):
        """Converge millimeter gaps to the relative tolerance, not an absolute 0.1 mm."""
        radius, half = 0.01, np.array([0.02, 0.015, 0.01])
        rng = np.random.default_rng(3)
        directions = rng.normal(size=(500, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        centers, expected = [], []
        for direction, gap in zip(directions, 10.0 ** rng.uniform(-4.0, -2.5, 500), strict=True):
            lo, hi = 0.0, 1.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                offset = direction * mid - np.clip(direction * mid, -half, half)
                lo, hi = (mid, hi) if np.linalg.norm(offset) < radius + gap else (lo, mid)
            offset = direction * hi - np.clip(direction * hi, -half, half)
            centers.append(direction * hi)
            expected.append([np.linalg.norm(offset) - radius, *(offset / np.linalg.norm(offset))])
        expected = np.array(expected)
        for device in wp.get_devices():
            with self.subTest(device=str(device)):
                output = wp.zeros((500, 4), dtype=float, device=device)
                wp.launch(
                    _query_sphere_near_box,
                    dim=500,
                    inputs=[wp.array(np.array(centers), dtype=wp.vec3, device=device)],
                    outputs=[output],
                    device=device,
                )
                actual = output.numpy()
                np.testing.assert_allclose(actual[:, 0], expected[:, 0], atol=2e-6, rtol=0.0)
                cosine = np.clip(np.sum(actual[:, 1:] * expected[:, 1:], axis=1), -1.0, 1.0)
                self.assertLess(np.degrees(np.arccos(cosine)).max(), 1.0)

    def test_distant_small_shapes_return_witnesses_on_both_shapes(self):
        """Populate the simplex before accepting convergence, so far-apart witnesses lie on the shapes."""
        radius, center_b = 0.001, np.array([100.0, 0.0, 0.0])
        for device in wp.get_devices():
            with self.subTest(device=str(device)):
                output = wp.zeros(7, dtype=float, device=device)
                wp.launch(
                    _query_distant_spheres,
                    dim=1,
                    inputs=[radius, wp.vec3(*center_b)],
                    outputs=[output],
                    device=device,
                )
                actual = output.numpy()
                self.assertAlmostEqual(float(actual[0]), 100.0 - 2.0 * radius, delta=2e-5)
                self.assertAlmostEqual(float(np.linalg.norm(actual[1:4])), radius, delta=1e-5)
                self.assertAlmostEqual(float(np.linalg.norm(actual[4:7] - center_b)), radius, delta=1e-5)


@wp.kernel
def _query_distant_spheres(radius: float, center_b: wp.vec3, output: wp.array[float]):
    """Query two small spheres far enough apart that the center offset alone satisfies the relative gap."""
    a = GenericShapeData()
    a.shape_type = int(GeoType.SPHERE)
    a.scale = wp.vec3(radius, 0.0, 0.0)
    b = GenericShapeData()
    b.shape_type = int(GeoType.SPHERE)
    b.scale = wp.vec3(radius, 0.0, 0.0)
    _separated, point_a, point_b, _normal, distance = wp.static(create_solve_closest_distance(support_map).core)(
        a, b, wp.quat_identity(), center_b, 0.0, SupportMapDataProvider()
    )
    output[0] = distance
    for axis in range(3):
        output[1 + axis] = point_a[axis]
        output[4 + axis] = point_b[axis]


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


@wp.kernel
def _query_sphere_near_box(center: wp.array[wp.vec3], output: wp.array2d[float]):
    """Query a sphere whose closest box feature is a face, edge, or corner."""
    i = wp.tid()
    a = GenericShapeData()
    a.shape_type = int(GeoType.BOX)
    a.scale = wp.vec3(0.02, 0.015, 0.01)
    b = GenericShapeData()
    b.shape_type = int(GeoType.SPHERE)
    b.scale = wp.vec3(0.01, 0.0, 0.0)
    _separated, _point_a, _point_b, normal, distance = wp.static(create_solve_closest_distance(support_map).core)(
        a, b, wp.quat_identity(), center[i], 0.0, SupportMapDataProvider()
    )
    output[i, 0] = distance
    for axis in range(3):
        output[i, 1 + axis] = normal[axis]


if __name__ == "__main__":
    unittest.main()
