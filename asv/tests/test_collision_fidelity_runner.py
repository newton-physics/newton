# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check runner validation and failure reporting without advancing physics."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1]))

from collision_fidelity import runner
from collision_fidelity.catalog import get_case
from collision_fidelity.protocol import Timeline


class TestFidelityRunner(unittest.TestCase):
    def test_recording_reads_live_states_without_changing_steps(self):
        """Observe a real two-frame free fall without adding steps or stale buffers."""
        from collision_fidelity import recording  # noqa: PLC0415

        samples = []

        class Recorder:
            def __init__(self, scene, path, *, fps):
                self.metadata = {"status": "recording"}

            def record(self, state, time_s):
                samples.append((time_s, float(state.particle_q.numpy()[:, 2].mean())))

            def close(self):
                self.metadata["status"] = "completed"

        case_id = "s3_pads_10substeps"
        duration = 2 * get_case(case_id)["frame_dt_s"]
        baseline = runner.run_case(case_id, device="cpu", duration=duration)
        with patch.object(recording, "VideoRecorder", Recorder):
            result = runner.run_case(case_id, device="cpu", duration=duration, video_path="not-written.mp4")
        self.assertEqual(result["execution"], baseline["execution"])
        self.assertEqual(result["measurements"], baseline["measurements"])
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0][0], 0.0)
        self.assertAlmostEqual(samples[-1][0], duration)
        self.assertGreater(samples[0][1], samples[1][1])
        self.assertGreater(samples[1][1], samples[2][1])
        self.assertEqual(result["recording"]["status"], "completed")

    def test_unsupported_video_does_not_import_renderer(self):
        """Leave an explicit video outcome when a case cannot run with VBD."""
        result = runner.run_case("r2_franka_duck", video_path="duck.mp4")
        self.assertEqual(result["recording"]["status"], "unsupported")
        self.assertEqual(result["execution"]["steps"], 0)

    def test_capture_failure_does_not_change_physics_outcome(self):
        """Preserve the completed physical timeline after a frame capture fails."""
        recording = {"status": "recording", "frame_count": 0}
        recorder = SimpleNamespace(record=lambda state, time: (_ for _ in ()).throw(RuntimeError("capture failed")))
        runner._record_video_frame(recorder, recording, object(), 0.0)
        self.assertEqual(recording["status"], "error")
        self.assertIn("capture failed", recording["error"])
        execution = {"status": "completed", "completed": True, "steps": 10}
        recorder.close = lambda: None
        runner._close_video(recorder, recording, execution)
        self.assertEqual(execution, {"status": "completed", "completed": True, "steps": 10})

    def test_video_close_failure_keeps_original_physics_failure(self):
        """Keep the two error channels separate even if finalizing the MP4 fails."""
        recording = {"status": "recording", "frame_count": 2}
        recorder = SimpleNamespace(close=lambda: (_ for _ in ()).throw(RuntimeError("encoder failed")))
        execution = {"status": "nonfinite", "reason": "Nonfinite state observed", "steps": 10}
        runner._close_video(recorder, recording, execution)
        self.assertEqual(recording["status"], "error")
        self.assertIn("encoder failed", recording["error"])
        self.assertEqual(execution["status"], "nonfinite")
        self.assertEqual(execution["reason"], "Nonfinite state observed")

    def test_duration_shorter_than_substep_is_rejected(self):
        """Reject a zero-step or overshooting duration before scene construction."""
        dt = Timeline(get_case("r2_franka_duck")).dt
        for duration in (1e-20, 0.5 * dt):
            with self.subTest(duration=duration), self.assertRaisesRegex(ValueError, "substep"):
                runner.run_case("r2_franka_duck", duration=duration)

    def test_revision_is_optional_when_git_is_not_installed(self):
        """Keep optional source provenance from preventing execution without git."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".git").mkdir()
            module = SimpleNamespace(__file__=str(root / "package" / "__init__.py"))
            with patch.dict(os.environ, {"PATH": temporary}):
                self.assertIsNone(runner._revision(module))

    def test_usd_close_error_keeps_execution_evidence(self):
        """Contain a real USD save failure and preserve a prior physics error."""
        try:
            from pxr import Usd  # noqa: PLC0415, F401

            from newton.viewer import ViewerUSD  # noqa: PLC0415
        except ImportError:
            self.skipTest("USD recording requires the optional usd-core package")
        close = getattr(runner, "_close_viewer", None)
        self.assertTrue(callable(close), "The runner must contain viewer.close failures")
        with tempfile.TemporaryDirectory() as temporary:
            for status in ("completed", "nonfinite"):
                with self.subTest(status=status):
                    viewer = ViewerUSD(str(Path(temporary) / f"{status}.usda"))
                    viewer.stage.GetRootLayer().SetPermissionToSave(False)
                    execution = {"status": status, "completed": status == "completed", "steps": 7}
                    if status == "nonfinite":
                        execution["reason"] = "Nonfinite state observed"
                    close(viewer, execution)
                    self.assertEqual(execution["steps"], 7)
                    self.assertIn("usd_export_error", execution)
                    self.assertEqual(execution["status"], "error" if status == "completed" else "nonfinite")
                    if status == "nonfinite":
                        self.assertEqual(execution["reason"], "Nonfinite state observed")
                    json.dumps(execution, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
