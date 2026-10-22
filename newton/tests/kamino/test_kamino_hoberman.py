# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate procedural Hoberman geometry against the original sphere."""

import contextlib
import gc
import io
import time
import unittest
import weakref
from collections import Counter
from itertools import product

import numpy as np
import warp as wp

from newton.examples.kamino.example_kamino_earworm_flowers import (
    SONG_LOOP_BAR,
    SONG_ROW,
    Example,
    play_song,
)
from newton.examples.kamino.utils.hoberman import generate_sphere
from newton.tests.test_viewer_get_frame import _make_headless_viewer_gl_or_skip


def _box_corners(pose, extents):
    """Transform a box's eight local corners through its pose."""
    transform = wp.transform(*pose)
    return np.array(
        [
            tuple(wp.transform_point(transform, wp.vec3(*(np.array(extents) * signs))))
            for signs in product((-1, 1), repeat=3)
        ]
    )


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
        self.assertEqual(sphere.reference_opening_slope, sphere.opening_slope)
        self.assertAlmostEqual(sphere.diameter_rest, 4.4711468, places=6)

        reference_tiles = (
            (0.10988911986351013, -0.23822249472141266, 0, 0, 0, -0.6029926538467407, 0.7977467775344849),
            (-0.11045127362012863, 0.23796238005161285, 0, 0, 0, 0.8822540044784546, 0.47077369689941406),
        )
        expected = np.concatenate([_box_corners(pose, (0.266, 0.025, 0.025)) for pose in reference_tiles])
        actual = np.concatenate([_box_corners(pose, extents) for index, pose, extents in sphere.tiles if index == 0])
        distances = np.linalg.norm(expected[:, None] - actual[None, :], axis=2)
        self.assertLess(float(distances.min(axis=1).max()), 3e-6)

    def test_resolution_preserves_connected_mechanism(self):
        """Keep tree topology, three-pin struts and shared ring crossings at varied resolutions."""
        reference_slope = generate_sphere().opening_slope
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
                radius_bounds = 0.0
                for index, pose, extents in sphere.tiles:
                    self.assertGreater(min(extents), 0)
                    vertices = _box_corners(pose, extents) + sphere.bodies[index][1][:3]
                    radius_bounds = max(radius_bounds, float(np.linalg.norm(vertices, axis=1).max()))
                self.assertAlmostEqual(radius_bounds / sphere.diameter_rest, 0.5, places=6)
                self.assertEqual(sphere.reference_opening_slope, reference_slope)
                self.assertGreater(sphere.opening_slope, 0)
        for resolution in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                generate_sphere(resolution)

    def test_cli_arguments(self):
        """Parse flower and tower sizes and reject nonpositive or fractional values."""
        parser = Example.create_parser()
        self.assertEqual(parser.parse_args([]).flower_resolution, 2)
        self.assertEqual(parser.parse_args(["--flower-resolution", "3"]).flower_resolution, 3)
        self.assertEqual(parser.parse_args([]).stage_count, 20)
        self.assertEqual(parser.parse_args(["--stage-count", "3"]).stage_count, 3)
        self.assertFalse(parser.parse_args([]).mute)
        self.assertTrue(parser.parse_args(["--mute"]).mute)
        self.assertIn("--mute", parser.format_help())
        for flag in ("--flower-resolution", "--stage-count"):
            for value in ("0", "-1", "1.5"):
                with self.subTest(flag=flag, value=value):
                    errors = io.StringIO()
                    with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as raised:
                        parser.parse_args([flag, value])
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn(flag, errors.getvalue())
                    self.assertIn("positive integer", errors.getvalue())


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA for Kamino")
class TestEarwormPlayback(unittest.TestCase):
    def test_pause_resume_and_discard(self):
        """Follow real playback, pause without clock jumps, and release a discarded example."""
        viewer = _make_headless_viewer_gl_or_skip(self)
        self.addCleanup(viewer.close)
        args = Example.create_parser().parse_args(
            ["--mute", "--world-count", "1", "--stage-count", "1", "--flower-resolution", "1"]
        )
        example = Example(viewer, args)
        # Exercise the real streaming source and player without emitting sound
        # or synthesizing a full song. Include the intro and one loop bar.
        player = play_song(bytes(4 * 16 * (SONG_LOOP_BAR + 1) * SONG_ROW))
        if player is None:
            self.skipTest("Audio playback unavailable")
        self.addCleanup(player.delete)
        self.addCleanup(player.pause)
        example.player = player

        # A slow frame must advance the dance with the audio, rather than
        # leave it on a separate clock that drifts at any rate other than 60 Hz.
        time.sleep(0.04)
        audio_before = player.time
        example.step()
        dance_time = float(example.time.numpy()[0]) - example.frame_dt
        self.assertGreaterEqual(dance_time, audio_before - 1e-5)
        self.assertLessEqual(dance_time, player.time + 1e-5)
        example.render()
        self.assertTrue(player.playing)

        viewer._paused = True
        example.render()
        self.assertFalse(player.playing)
        paused_time = player.time
        before = float(example.time.numpy()[0])
        time.sleep(0.1)
        example.render()
        self.assertEqual(player.time, paused_time)
        self.assertEqual(float(example.time.numpy()[0]), before)

        # A single step while paused must leave playback paused too.
        example.step()
        example.render()
        self.assertFalse(player.playing)
        self.assertAlmostEqual(float(example.time.numpy()[0]), player.time, places=5)
        self.assertAlmostEqual(player.time - paused_time, example.frame_dt, places=5)
        for _ in range(3):
            example.step()
        self.assertFalse(player.playing)
        self.assertAlmostEqual(player.time - paused_time, 4 * example.frame_dt, places=5)
        self.assertAlmostEqual(float(example.time.numpy()[0]), player.time, places=5)

        viewer._paused = False
        audio_before = player.time
        example.step()
        example.render()
        self.assertTrue(player.playing)
        dance_time = float(example.time.numpy()[0]) - example.frame_dt
        self.assertGreaterEqual(dance_time, audio_before - 1e-5)
        self.assertLessEqual(dance_time, player.time + 1e-5)

        # Reset and browser switching discard the previous example. Retain the
        # player to verify that its worker no longer owns playback afterward.
        reference = weakref.ref(example)
        del example
        gc.collect()
        self.assertIsNone(reference())
        self.assertFalse(player.playing)
        self.assertIsNone(player._audio_player)


if __name__ == "__main__":
    unittest.main()
