# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: UNIT TESTS: VIEWER RECORDING
"""

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

import warp as wp

from newton._src.solvers.kamino._src.utils.sim.viewer_recording import enable_recording
from newton.tests.kamino import setup_tests
from newton.tests.test_viewer_get_frame import _make_box_model, _make_headless_viewer_gl_or_skip


@unittest.skipUnless(importlib.util.find_spec("imageio_ffmpeg") is not None, "Requires imageio-ffmpeg")
class TestKaminoViewerRecording(unittest.TestCase):
    def test_headless_video_recording(self):
        """Encode and decode a real recording driven by a headless ``ViewerGL``.

        Exercises both clip modes end-to-end on a CPU-only viewer so the test
        runs on standard CI runners (no OVRTX required): a finite clip that
        auto-finishes at ``max_frames``, then an open-ended clip finalized by
        ``viewer.close()`` with a stale PNG dropped into the frame folder to
        confirm ``_generate_video`` enumerates only this clip's frames.
        """
        import imageio_ffmpeg as ffmpeg  # noqa: PLC0415
        from PIL import Image

        def decode_frames(video_path: Path, out_dir: Path, prefix: str) -> list[Path]:
            """Decode ``video_path`` into PNGs under ``out_dir`` and return them sorted."""
            subprocess.run(
                [
                    ffmpeg.get_ffmpeg_exe(),
                    "-v",
                    "error",
                    "-i",
                    str(video_path),
                    str(out_dir / f"{prefix}-%02d.png"),
                ],
                check=True,
                capture_output=True,
            )
            return sorted(out_dir.glob(f"{prefix}-*.png"))

        # Build a minimal CPU scene so get_frame() returns a real rendered image.
        with wp.ScopedDevice("cpu"):
            viewer = _make_headless_viewer_gl_or_skip(self)
        model = _make_box_model("cpu")
        viewer.set_model(model)
        state = model.state()

        closed = False
        try:
            self.assertTrue(enable_recording(viewer))
            with tempfile.TemporaryDirectory() as directory:
                # Finite clip: auto-finishes at max_frames and decodes back to the same count.
                path = Path(directory) / "recording.mp4"
                frames_dir = Path(directory) / "frames"
                viewer.start_clip(str(path), max_frames=2, video_folder=str(frames_dir))
                for frame_index in range(2):
                    viewer.begin_frame(frame_index / 60)
                    self.assertTrue(viewer.should_step())
                    viewer.log_state(state)
                    viewer.end_frame()

                self.assertTrue(path.is_file())
                decoded = decode_frames(path, Path(directory), "decoded")
                self.assertEqual(len(decoded), 2)
                for frame_path in decoded:
                    with Image.open(frame_path) as frame:
                        self.assertEqual(frame.size, (64, 48))

                # Open-ended clip: no max_frames, finalized by viewer.close().
                # Drop a stale PNG into the frame folder beforehand to confirm
                # _generate_video enumerates only this clip's frames (fixed set from img_idx)
                # and leaves unrelated files alone.
                open_path = Path(directory) / "open_ended.mp4"
                open_frames_dir = Path(directory) / "open_frames"
                viewer.start_clip(str(open_path), max_frames=None, video_folder=str(open_frames_dir))
                stale_frame = open_frames_dir / "stale.png"
                Image.new("RGB", (64, 48), color=(255, 0, 0)).save(stale_frame)
                for frame_index in range(2):
                    viewer.begin_frame((frame_index + 10) / 60)
                    self.assertTrue(viewer.should_step())
                    viewer.log_state(state)
                    viewer.end_frame()
                self.assertFalse(open_path.is_file())  # Finalized only on close.

                viewer.close()
                closed = True

                self.assertTrue(open_path.is_file())
                decoded_open = decode_frames(open_path, Path(directory), "decoded-open")
                self.assertEqual(len(decoded_open), 2)
                for frame_path in decoded_open:
                    with Image.open(frame_path) as frame:
                        self.assertEqual(frame.size, (64, 48))
                self.assertTrue(stale_frame.is_file())  # Untouched: not part of the explicit frame list.
        finally:
            if not closed:
                viewer.close()


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
