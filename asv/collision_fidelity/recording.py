# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optional fixed-camera MP4 capture using public viewer APIs only.

Run one recorder per child process. Recording reads the supplied live state;
it never advances physics or applies viewer forces. The caller owns physics
status and handles recording failures separately.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import queue
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

__all__ = ["VideoRecorder", "recording_preflight"]

_WRITE_TIMEOUT_S = 30.0
_CLOSE_TIMEOUT_S = 30.0
_STOP_TIMEOUT_S = 2.0


def _find_ffmpeg() -> str:
    """Prefer a system encoder, then the explicitly installed optional wheel."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg  # noqa: PLC0415

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError) as error:
        raise RuntimeError(
            "MP4 recording requires FFmpeg on PATH or the optional imageio-ffmpeg package "
            "(for example, uv run --with imageio-ffmpeg ...)."
        ) from error


def recording_preflight() -> None:
    """Check optional packages and H.264 support without creating a GL context.

    A working OpenGL/EGL context and a rendering GPU compatible with the model
    are still required when the recorder is constructed.
    """
    missing = [name for name in ("pyglet", "imgui_bundle") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Video recording requires Newton's examples extra; missing {', '.join(missing)}.")
    executable = _find_ffmpeg()
    try:
        result = subprocess.run(
            [executable, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Cannot run FFmpeg for MP4 recording: {error}") from error
    if result.returncode or "libx264" not in result.stdout:
        raise RuntimeError("MP4 recording requires a working FFmpeg with the libx264 H.264 encoder.")


class VideoRecorder:
    """Stream public ViewerGL RGB frames to FFmpeg without changing physics.

    The mutable ``metadata`` dictionary remains available after errors.
    ``frame_times_s`` contains physical capture times; encoded duration includes
    every frame, including the initial frame at zero. A hash is added only after
    a successful close. Headless GPU selection follows pyglet's documented
    ``PYGLET_HEADLESS_DEVICE`` setting.
    """

    def __init__(self, scene, path: Path, *, fps: float):
        fps = float(fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Recording FPS must be finite and positive")
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"Recording already exists: {path}")
        position = [float(value) for value in scene.camera_pos]
        target = [float(value) for value in scene.camera_target]
        fov = float(getattr(scene, "camera_fov", 45.0))
        if (
            len(position) != 3
            or len(target) != 3
            or not all(math.isfinite(value) for value in (*position, *target, fov))
            or not 0 < fov < 180
            or position == target
        ):
            raise ValueError("Recording requires a finite fixed camera and a distinct target")
        self.metadata = {
            "status": "recording",
            "path": path.name,
            "fps": fps,
            "width": 960,
            "height": 720,
            "camera": {"position": position, "target": target, "fov": fov},
            "frame_count": 0,
            "frame_times_s": [],
            "physical_end_time_s": None,
            "encoded_duration_s": 0.0,
        }
        self._path = path
        self._viewer = self._process = self._stderr = self._writer = None
        self._queue = queue.SimpleQueue()
        self._error = None
        self._closed = False
        try:
            executable = _find_ffmpeg()
            import pyglet  # noqa: PLC0415

            pyglet.options["headless"] = True
            from newton.viewer import ViewerGL  # noqa: PLC0415

            self._viewer = ViewerGL(width=960, height=720, headless=True, num_frames=None)
            self._viewer.set_model(scene.model)
            self._viewer.set_camera(position)
            self._viewer.camera.look_at(target)
            self._viewer.camera.fov = fov
            path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-n",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "960x720",
                "-r",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path.resolve()),
            ]
            self.metadata["encoder_command"] = command
            # A file prevents stderr backpressure without an extra reader thread.
            self._stderr = tempfile.TemporaryFile()
            self._process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr, bufsize=0
            )
            self._writer = threading.Thread(target=self._write_frames, name="fidelity-video-writer", daemon=True)
            self._writer.start()
        except Exception as error:
            self._set_error(error)
            self._stop_encoder()
            if self._viewer is not None:
                self._viewer.close()
            if self._stderr is not None:
                self._stderr.close()
            self._closed = True
            raise

    def _set_error(self, error):
        """Retain the first failure without claiming an encoded video."""
        if self._error is None:
            self._error = str(error)
        self.metadata.update(status="error", error=self._error)

    def _write_frames(self):
        """Write one acknowledged frame at a time; bounded waits live in the caller."""
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                data, done = item
                try:
                    view = memoryview(data)
                    while view:
                        count = self._process.stdin.write(view)
                        if not count:
                            raise BrokenPipeError("FFmpeg closed its frame input")
                        view = view[count:]
                except Exception as error:
                    self._set_error(f"FFmpeg frame write failed: {error}")
                    return
                finally:
                    done.set()
        finally:
            self._process.stdin.close()

    def _stop_encoder(self):
        """Bound child cleanup even when it stops consuming frame input."""
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=_STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=_STOP_TIMEOUT_S)
        self._queue.put(None)
        if self._writer is not None:
            self._writer.join(timeout=_STOP_TIMEOUT_S)

    def record(self, state, time_s: float) -> None:
        """Read the supplied current state and capture one physical timestamp."""
        if self._closed or self._error:
            raise RuntimeError(self._error or "Recording is closed")
        time_s = float(time_s)
        times = self.metadata["frame_times_s"]
        if not math.isfinite(time_s) or time_s < 0 or (times and time_s <= times[-1]):
            raise ValueError("Frame timestamps must be finite, nonnegative, and strictly increasing")
        try:
            self._viewer.begin_frame(time_s)
            self._viewer.log_state(state)
            self._viewer.end_frame()
            pixels = self._viewer.get_frame().numpy()
            if pixels.shape != (720, 960, 3) or str(pixels.dtype) != "uint8":
                raise RuntimeError("Viewer capture must return a 720 x 960 RGB uint8 image")
            done = threading.Event()
            self._queue.put((pixels.tobytes(), done))
            if not done.wait(_WRITE_TIMEOUT_S):
                raise RuntimeError("FFmpeg frame write timed out")
            if self._error:
                raise RuntimeError(self._error)
        except Exception as error:
            self._set_error(error)
            self._stop_encoder()
            raise
        times.append(time_s)
        self.metadata.update(
            frame_count=len(times), physical_end_time_s=time_s, encoded_duration_s=len(times) / self.metadata["fps"]
        )

    def close(self) -> None:
        """Finalize MP4 output, release resources, and hash successful output only."""
        if self._closed:
            if self._error:
                raise RuntimeError(self._error)
            return
        try:
            self._queue.put(None)
            self._writer.join(timeout=_WRITE_TIMEOUT_S)
            if self._writer.is_alive():
                raise RuntimeError("FFmpeg frame write timed out during close")
            code = self._process.wait(timeout=_CLOSE_TIMEOUT_S)
            if self._error:
                raise RuntimeError(self._error)
            if code:
                self._stderr.seek(0, 2)
                self._stderr.seek(max(0, self._stderr.tell() - 8192))
                detail = self._stderr.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"FFmpeg exited {code}: {detail}")
            if not self.metadata["frame_count"] or not self._path.is_file() or not self._path.stat().st_size:
                raise RuntimeError("Recording produced no video frames")
        except Exception as error:
            self._set_error(error)
            self._stop_encoder()
        finally:
            try:
                self._viewer.close()
            except Exception as error:
                self._set_error(error)
            self._stderr.close()
            self._closed = True
        if self._error:
            raise RuntimeError(self._error)
        digest = hashlib.sha256()
        with self._path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        self.metadata.update(status="completed", sha256=digest.hexdigest())
