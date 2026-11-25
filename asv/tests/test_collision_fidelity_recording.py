# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check MP4 capture without constructing or advancing a physics scene."""

import hashlib
import importlib
import importlib.util
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.append(str(Path(__file__).parents[1]))


class _Viewer:
    """Supply changing pixels at the public viewer boundary."""

    def __init__(self, **kwargs):
        self.width, self.height = kwargs["width"], kwargs["height"]
        self.camera = SimpleNamespace(fov=None, look_at=lambda target: None)
        self.closed = False

    def set_model(self, model):
        self.model = model

    def set_camera(self, pos):
        self.position = pos

    def begin_frame(self, time_s):
        self.time = time_s

    def log_state(self, state):
        self.state = state

    def end_frame(self):
        pass

    def get_frame(self):
        pixels = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        pixels[:, :, self.state] = 255
        return SimpleNamespace(numpy=lambda: pixels)

    def close(self):
        self.closed = True


class TestFidelityRecording(unittest.TestCase):
    def setUp(self):
        """Load the optional recorder without initializing a physics backend."""
        self.assertIsNotNone(importlib.util.find_spec("collision_fidelity.recording"))
        self.module = importlib.import_module("collision_fidelity.recording")
        self.scene = SimpleNamespace(model=object(), camera_pos=(1, -2, 3), camera_target=(0, 0, 0))
        self.viewer_modules = {
            "newton.viewer": SimpleNamespace(ViewerGL=_Viewer),
            "pyglet": SimpleNamespace(options={}),
        }

    def test_real_encoder_records_live_frames_and_timestamps(self):
        """Encode and decode changing CPU frames with exact capture metadata."""
        try:
            encoder = self.module._find_ffmpeg()
        except RuntimeError as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, self.viewer_modules):
            path = Path(temporary) / "clip.mp4"
            recorder = self.module.VideoRecorder(self.scene, path, fps=30.0)
            recorder.record(0, 0.0)
            recorder.record(1, 1.0 / 30)
            with self.assertRaises(ValueError):
                recorder.record(0, 0.0)
            self.assertNotIn("sha256", recorder.metadata)
            recorder.close()
            recorder.close()
            self.assertEqual(recorder.metadata["status"], "completed")
            self.assertEqual(recorder.metadata["path"], "clip.mp4")
            self.assertEqual(recorder.metadata["frame_times_s"], [0.0, 1.0 / 30])
            self.assertEqual(recorder.metadata["physical_end_time_s"], 1.0 / 30)
            self.assertEqual(recorder.metadata["encoded_duration_s"], 2.0 / 30)
            self.assertEqual(recorder.metadata["camera"]["fov"], 45.0)
            self.assertEqual(recorder.metadata["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            decoded = subprocess.run(
                [encoder, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                capture_output=True,
                check=True,
                timeout=15,
            )
            frames = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(2, 720, 960, 3)
            self.assertGreater(frames[0, :, :, 0].mean(), 240)
            self.assertGreater(frames[1, :, :, 1].mean(), 240)

    def test_missing_dependencies_report_install_guidance(self):
        """Reject missing viewer dependencies before an expensive run starts."""
        with patch.object(self.module.importlib.util, "find_spec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "examples"):
                self.module.recording_preflight()

    def test_output_and_invalid_inputs_are_preserved(self):
        """Reject invalid FPS and occupied paths without overwriting files."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "clip.mp4"
            path.write_bytes(b"existing")
            for fps in (0, float("nan"), float("inf")):
                with self.subTest(fps=fps), self.assertRaises(ValueError):
                    self.module.VideoRecorder(self.scene, path, fps=fps)
            with self.assertRaises(FileExistsError):
                self.module.VideoRecorder(self.scene, path, fps=30)
            self.assertEqual(path.read_bytes(), b"existing")

    def test_encoder_failure_retains_error_without_success_hash(self):
        """Surface encoder failure while closing the public viewer and pipes."""
        real_popen = subprocess.Popen

        def launch(*args, **kwargs):
            return real_popen(
                [sys.executable, "-c", "import sys; sys.stderr.write('encoder failed'); sys.exit(7)"], **kwargs
            )

        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, self.viewer_modules):
            with (
                patch.object(self.module, "_find_ffmpeg", return_value="unused"),
                patch.object(self.module.subprocess, "Popen", side_effect=launch),
            ):
                recorder = self.module.VideoRecorder(self.scene, Path(temporary) / "clip.mp4", fps=30)
                with self.assertRaisesRegex(RuntimeError, "encoder failed|FFmpeg"):
                    try:
                        recorder.record(0, 0.0)
                    finally:
                        recorder.close()
                self.assertEqual(recorder.metadata["status"], "error")
                self.assertNotIn("sha256", recorder.metadata)
                self.assertTrue(recorder._viewer.closed)
                self.assertIsNotNone(recorder._process.poll())

    def test_stalled_encoder_write_has_bounded_cleanup(self):
        """Terminate a child that never consumes its input rather than hanging."""
        real_popen = subprocess.Popen

        def launch(*args, **kwargs):
            return real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)

        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, self.viewer_modules):
            with (
                patch.object(self.module, "_find_ffmpeg", return_value="unused"),
                patch.object(self.module.subprocess, "Popen", side_effect=launch),
                patch.object(self.module, "_WRITE_TIMEOUT_S", 0.2),
            ):
                recorder = self.module.VideoRecorder(self.scene, Path(temporary) / "clip.mp4", fps=30)
                started = time.monotonic()
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    recorder.record(0, 0.0)
                with self.assertRaises(RuntimeError):
                    recorder.close()
                self.assertLess(time.monotonic() - started, 5)
                self.assertIsNotNone(recorder._process.poll())
                self.assertFalse(recorder._writer.is_alive())

    def test_encoder_finalization_timeout_is_bounded(self):
        """Stop a child that consumes frames but hangs after its input closes."""
        real_popen = subprocess.Popen

        def launch(*args, **kwargs):
            return real_popen(
                [sys.executable, "-c", "import sys,time; sys.stdin.buffer.read(); time.sleep(60)"], **kwargs
            )

        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, self.viewer_modules):
            with (
                patch.object(self.module, "_find_ffmpeg", return_value="unused"),
                patch.object(self.module.subprocess, "Popen", side_effect=launch),
                patch.object(self.module, "_CLOSE_TIMEOUT_S", 0.2),
            ):
                recorder = self.module.VideoRecorder(self.scene, Path(temporary) / "clip.mp4", fps=30)
                recorder.record(0, 0.0)
                started = time.monotonic()
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    recorder.close()
                self.assertLess(time.monotonic() - started, 5)
                self.assertEqual(recorder.metadata["status"], "error")
                self.assertNotIn("sha256", recorder.metadata)
                self.assertIsNotNone(recorder._process.poll())
                self.assertTrue(recorder._viewer.closed)

    def test_invalid_timestamps_and_empty_recording_do_not_claim_success(self):
        """Reject nonfinite timestamps and an empty output without a video hash."""
        try:
            self.module._find_ffmpeg()
        except RuntimeError as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, self.viewer_modules):
            recorder = self.module.VideoRecorder(self.scene, Path(temporary) / "clip.mp4", fps=30)
            with self.assertRaises(ValueError):
                recorder.record(0, float("nan"))
            with self.assertRaisesRegex(RuntimeError, "frames"):
                recorder.close()
            self.assertNotIn("sha256", recorder.metadata)


if __name__ == "__main__":
    unittest.main()
