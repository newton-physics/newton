# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``--render-fps`` example CLI option."""

import contextlib
import io
import unittest

from newton.examples import _throttle_render_fps, create_parser


class TestRenderFPSCLI(unittest.TestCase):
    """Tests for render FPS parsing and throttling."""

    def test_parser_has_render_fps_arg(self):
        """The base parser should include --render-fps."""
        parser = create_parser()
        args = parser.parse_known_args(["--render-fps", "30"])[0]
        self.assertEqual(args.render_fps, 30.0)

    def test_default_render_fps_none(self):
        """Render FPS should be uncapped by default."""
        parser = create_parser()
        args = parser.parse_known_args([])[0]
        self.assertIsNone(args.render_fps)

    def test_render_fps_rejects_non_positive_values(self):
        """Non-positive render FPS limits should be rejected."""
        parser = create_parser()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                parser.parse_known_args(["--render-fps", "0"])
            with self.assertRaises(SystemExit):
                parser.parse_known_args(["--render-fps", "-10"])
            with self.assertRaises(SystemExit):
                parser.parse_known_args(["--render-fps", "nan"])
        self.assertIn("must be a finite value greater than 0", stderr.getvalue())

    def test_throttle_sleeps_for_remaining_frame_time(self):
        """Throttle should sleep for the remaining frame period."""
        sleeps = []

        slept = _throttle_render_fps(
            frame_start_time=10.0,
            render_fps=20.0,
            time_fn=lambda: 10.02,
            sleep_fn=sleeps.append,
        )

        self.assertAlmostEqual(slept, 0.03)
        self.assertEqual(sleeps, [slept])

    def test_throttle_skips_sleep_when_frame_is_slow(self):
        """Throttle should not sleep once a frame exceeds the target period."""
        sleeps = []

        slept = _throttle_render_fps(
            frame_start_time=10.0,
            render_fps=20.0,
            time_fn=lambda: 10.07,
            sleep_fn=sleeps.append,
        )

        self.assertEqual(slept, 0.0)
        self.assertEqual(sleeps, [])

    def test_throttle_skips_sleep_without_render_fps(self):
        """No cap should be applied when render FPS is None."""
        sleeps = []

        slept = _throttle_render_fps(
            frame_start_time=10.0,
            render_fps=None,
            time_fn=lambda: 10.0,
            sleep_fn=sleeps.append,
        )

        self.assertEqual(slept, 0.0)
        self.assertEqual(sleeps, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
