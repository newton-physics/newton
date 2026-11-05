# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test that examples reported empty by QA render meaningful scene content."""

import json
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

_FRAME_SIZE = 320
# This is a coarse visibility floor, not an image-fidelity threshold. The fixed
# examples render well over twice this many subject pixels on the reference GPU.
_MIN_SUBJECT_PIXELS = 500


class _ViewerGLUnavailable(Exception):
    """Signal that the headless GL viewer could not be constructed."""


def _is_gl_unavailable(exc: Exception) -> bool:
    """Return whether pyglet reported an unavailable GL display or context."""
    return type(exc).__module__.startswith("pyglet.") and type(exc).__name__ in {
        "ConfigException",
        "ContextException",
        "MissingFunctionException",
        "NoSuchConfigException",
        "NoSuchDisplayException",
    }


def _make_viewer():
    """Construct a headless GL viewer or report that its backend is unavailable."""
    import newton.viewer  # noqa: PLC0415

    try:
        return newton.viewer.ViewerGL(width=_FRAME_SIZE, height=_FRAME_SIZE, headless=True, num_frames=2)
    except Exception as exc:
        if not _is_gl_unavailable(exc):
            raise
        raise _ViewerGLUnavailable(str(exc)) from exc


def _capture_example(name: str) -> dict:
    """Render one example and return objective scene and frame metrics."""
    viewer = _make_viewer()
    try:
        if name == "basic_dzhanibekov":
            from newton.examples.basic.example_basic_dzhanibekov import Example  # noqa: PLC0415

            example = Example(viewer, SimpleNamespace(solver="vbd"))
        elif name == "basic_multi_solver_overlay":
            from newton.examples.basic.example_basic_multi_solver_overlay import Example  # noqa: PLC0415

            example = Example(viewer, SimpleNamespace())
        else:
            raise ValueError(f"Unknown example: {name}")

        viewer.renderer.draw_sky = False
        for _ in range(2):
            example.render()

        frame = viewer.get_frame().numpy().astype(np.float32)
        red, green, blue = np.moveaxis(frame, -1, 0)
        subject = (red > 1.5 * green) & (red > 1.5 * blue) & (red > 60.0)
        rows, columns = np.nonzero(subject)
        return {
            "frame_shape": list(frame.shape),
            "subject_pixels": int(subject.sum()),
            "touches_edge": bool(
                len(rows) == 0
                or rows.min() == 0
                or rows.max() == _FRAME_SIZE - 1
                or columns.min() == 0
                or columns.max() == _FRAME_SIZE - 1
            ),
        }
    finally:
        viewer.close()


class TestExampleRendering(unittest.TestCase):
    def test_qa_examples_render_meaningful_content(self):
        """Keep each QA example's subject visibly framed in a 320-pixel image."""
        if not wp.get_cuda_devices():
            self.skipTest("Headless example rendering requires a CUDA device")

        env = os.environ.copy()
        env["PYGLET_HEADLESS"] = "1"

        for name in ("basic_dzhanibekov", "basic_multi_solver_overlay"):
            with self.subTest(example=name):
                result = subprocess.run(
                    [sys.executable, __file__, "--capture", name],
                    capture_output=True,
                    text=True,
                    env=env,
                    # Match the example harness's allowance for cold Warp compilation.
                    timeout=600,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
                metrics_line = next(
                    (line for line in result.stdout.splitlines() if line.startswith("FRAME_METRICS ")),
                    None,
                )
                unavailable_line = next(
                    (line for line in result.stdout.splitlines() if line.startswith("FRAME_UNAVAILABLE ")),
                    None,
                )
                if unavailable_line is not None:
                    self.skipTest(unavailable_line.removeprefix("FRAME_UNAVAILABLE "))
                self.assertIsNotNone(metrics_line, msg=result.stdout + result.stderr)
                metrics = json.loads(metrics_line.removeprefix("FRAME_METRICS "))
                self.assertEqual(metrics["frame_shape"], [_FRAME_SIZE, _FRAME_SIZE, 3])
                self.assertGreaterEqual(metrics["subject_pixels"], _MIN_SUBJECT_PIXELS, msg=metrics)
                self.assertFalse(metrics["touches_edge"], msg=metrics)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--capture":
        wp.config.log_level = wp.LOG_WARNING
        wp.set_device("cuda:0")
        try:
            print("FRAME_METRICS " + json.dumps(_capture_example(sys.argv[2]), sort_keys=True))
        except _ViewerGLUnavailable as exc:
            print(f"FRAME_UNAVAILABLE {exc}")
    else:
        unittest.main(verbosity=2)
