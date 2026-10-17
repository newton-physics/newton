# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate procedural Hoberman geometry against the original sphere."""

import contextlib
import io
import unittest
from collections import Counter
from itertools import product

import numpy as np
import warp as wp

from newton.examples.kamino._hoberman_sphere import generate_sphere
from newton.examples.kamino.example_kamino_stewart_platform import Example


class TestHobermanGeometry(unittest.TestCase):
    def test_reference_sphere(self):
        """Reproduce the authored sphere's dimensions, mass properties and strut boxes."""
        sphere = generate_sphere()
        self.assertEqual((len(sphere.bodies), len(sphere.joints), len(sphere.tiles)), (240, 421, 480))
        self.assertEqual(Counter(j[0] for j in sphere.joints), {"FREE": 1, "REVOLUTE": 360, "BALL": 60})
        # Independent measurements from the former USD-derived table, covering
        # all six ring planes and the supported outer-crossing strut.
        positions = {
            "circle0_rb0": (0.24441300332546234, 1.7621614933013916, 0.02500000037252903),
            "circle0_rb29": (-1.7621614933013916, -0.24441300332546234, -0.02500000037252903),
            "circle1_rb2": (-0.1428077220916748, 1.6363458633422852, -0.683779239654541),
            "circle2_rb0": (0.6272266507148743, 1.6490479707717896, 0.22978997230529785),
            "circle3_rb5": (-0.6478025913238525, 1.206943392753601, 1.1354361772537231),
            "circle4_rb0": (1.5030096769332886, 0.9292529225349426, -0.20742928981781006),
            "circle5_rb39": (-1.7621614933013916, -0.13166549801826477, 0.20742928981781006),
        }
        by_name = {b[0].rsplit("/", 1)[-1]: b for b in sphere.bodies}
        for label, position in positions.items():
            np.testing.assert_allclose(by_name[label][1][:3], position, rtol=0, atol=2e-7)
        body = by_name["circle0_rb0"]
        np.testing.assert_allclose(
            body[3],
            (
                (0.20170384645462036, 0.09209958463907242, 0),
                (0.09209957718849182, 0.04528077319264412, 0),
                (0, 0, 0.24586963653564453),
            ),
            rtol=0,
            atol=1e-8,
        )
        self.assertAlmostEqual(sum(b[4] for b in sphere.bodies), 642.2305870056152, places=8)
        self.assertEqual(sphere.opening_label, "/hoberman_sphere/Joints/circle0_outer14")
        self.assertTrue(sphere.bodies[sphere.mount_body][0].endswith("/circle0_rb29"))

        def corners(pose, extents):
            transform = wp.transform(*pose)
            return np.array(
                [
                    tuple(wp.transform_point(transform, wp.vec3(*(np.array(extents) * signs))))
                    for signs in product((-1, 1), repeat=3)
                ]
            )

        reference_tiles = (
            (0.10988911986351013, -0.23822249472141266, 0, 0, 0, -0.6029926538467407, 0.7977467775344849),
            (-0.11045127362012863, 0.23796238005161285, 0, 0, 0, 0.8822540044784546, 0.47077369689941406),
        )
        expected = np.concatenate([corners(pose, (0.266, 0.025, 0.025)) for pose in reference_tiles])
        actual = np.concatenate([corners(pose, extents) for index, pose, extents, _ in sphere.tiles if index == 0])
        distances = np.linalg.norm(expected[:, None] - actual[None, :], axis=2)
        self.assertLess(float(distances.min(axis=1).max()), 3e-6)

    def test_resolution_preserves_connected_mechanism(self):
        """Keep tree topology, three-pin struts and shared ring crossings at varied resolutions."""
        for resolution in (1, 3, 4, 7):
            with self.subTest(resolution=resolution):
                sphere = generate_sphere(resolution)
                self.assertEqual(len(sphere.bodies), 120 * resolution)
                self.assertEqual(len(sphere.tiles), 240 * resolution)
                self.assertEqual(
                    Counter(j[0] for j in sphere.joints), {"FREE": 1, "REVOLUTE": 180 * resolution, "BALL": 60}
                )
                visited, degree, crossings = set(), Counter(), []
                pins = [[] for _ in sphere.bodies]
                for index, (kind, _, parent, child, xp, xc, _, _) in enumerate(sphere.joints):
                    if index < sphere.tree_joint_count:
                        self.assertNotIn(child, visited)
                        if parent >= 0:
                            self.assertIn(parent, visited)
                        visited.add(child)
                    if kind == "FREE":
                        continue
                    a = np.array(sphere.bodies[parent][1][:3]) + xp[:3]
                    b = np.array(sphere.bodies[child][1][:3]) + xc[:3]
                    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
                    if kind == "REVOLUTE":
                        degree.update((parent, child))
                        pins[parent].append(a)
                        pins[child].append(b)
                    else:
                        crossings.append((parent, child, a))
                self.assertEqual(len(visited), len(sphere.bodies))
                self.assertEqual(set(degree.values()), {3})
                for parent, child, point in crossings:
                    for body in (parent, child):
                        distances = np.linalg.norm(np.array(pins[body]) - point, axis=1)
                        self.assertLess(float(distances.min()), 1e-12)
                for body in sphere.bodies:
                    self.assertGreater(float(np.linalg.eigvalsh(body[3]).min()), 0)
                for _, _, extents, _ in sphere.tiles:
                    self.assertGreater(min(extents), 0)
                self.assertGreater(sphere.opening_slope, 0)
        for resolution in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                generate_sphere(resolution)
        parser = Example.create_parser()
        self.assertEqual(parser.parse_args([]).flower_resolution, 2)
        self.assertEqual(parser.parse_args(["--flower-resolution", "3"]).flower_resolution, 3)
        for value in ("0", "-1", "1.5"):
            with self.subTest(cli_resolution=value):
                errors = io.StringIO()
                with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as raised:
                    parser.parse_args(["--flower-resolution", value])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("--flower-resolution", errors.getvalue())
                self.assertIn("positive integer", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
