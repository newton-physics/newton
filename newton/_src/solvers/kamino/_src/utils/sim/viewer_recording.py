# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Ad-hoc viewer recording helper.

This monkey-patches a fully-initialized Newton viewer (as returned by
``newton.examples.init(parser)``) so the user can record per-frame PNGs
and stitch them into a video from inside the running viewer session.

After ``enable_recording`` the viewer is wired up but inactive: nothing is
written to disk until ``viewer.start_clip(...)`` is called, or the
``start_clip=True`` is passed to ``enable_recording``. Each call to
``viewer.start_clip()`` clears the target folder, resets counters, and records
up to ``max_frames`` frames before automatically writing the video file.

Example usage: (multi-clip mode, one video file per backend triggered by the UI):

    viewer, args = newton.examples.init(parser)
    record_video = enable_recording(viewer, default_video_folder=base_dir)
    ...
    # inside the GUI button handler:
    viewer.start_clip(
        output_path=os.path.join(base_dir, f"recording_{solver_name}.mp4"),
        max_frames=num_frames,
        video_folder=os.path.join(base_dir, f"frames_{solver_name}"),
        fps=viewer_fps,
        keep_frames=False,
    )

Exactly one PNG is captured per ``viewer.should_step()`` call that returns
True (i.e. per ``simulate()`` step). No PNGs are produced while paused or
when the render loop runs faster than the simulation.
"""

from __future__ import annotations

import glob
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from types import MethodType

import warp as wp

from newton._src.solvers.kamino._src.utils import logger as msg

__all__ = ["enable_recording"]


@dataclass
class _VideoRecording:
    """Helper class to store video recording settings and state in the viewer."""

    ###
    # Recording settings
    ###
    record_video: bool = True
    default_video_folder: str = "./video"
    async_save: bool = False
    num_skipped_frames: int = 0

    ###
    # Clip settings
    ###
    clip_max_frames: int | None = None
    clip_output_path: str | None = None
    clip_keep_frames: bool = False
    clip_on_done: Callable[[], None] | None = None
    clip_fps: int = 60

    ###
    # State
    ###
    img_idx: int = 0
    frame_buffer: wp.array3d[wp.uint8] | None = None
    recording_active: bool = False
    recording_stopped: bool = False
    capture_pending: bool = False
    created_video_folder: bool = False
    save_threads: list[threading.Thread] = field(default_factory=list)

    ###
    # Original viewer function refs
    ###
    original_should_step: Callable | None = None
    original_end_frame: Callable | None = None


def enable_recording(
    viewer,
    record_video: bool = True,
    default_video_folder: str = "./video",
    num_skipped_frames: int = 0,
    async_save: bool = False,
    start_clip: bool = False,
    **kwargs,
) -> bool:
    """Monkey-patch ``viewer`` with PNG recording and video file generation.

    Wires the recording machinery onto ``viewer`` but does **not** start
    recording. Call ``viewer.start_clip(...)`` to actually capture frames.
    No directories are created until then.

    Args:
        viewer: A Newton viewer instance (typically a ``ViewerGL`` returned by
            ``newton.examples.init``). Must expose ``get_frame()`` and a
            ``renderer`` with ``_screen_width`` / ``_screen_height``.
        record_video: If False, this is a no-op.
        default_video_folder: Default output directory used by clips that do not
            pass an explicit ``video_folder`` to ``start_clip``. Not created
            on disk until a clip starts.
        num_skipped_frames: Number of leading captured frames to drop before saving.
        async_save: If True, save each PNG on a background thread.
        start_clip: If True, will directly start a new clip.

    Returns:
        Flag indicating whether enabling the recording was successful.
    """
    if not record_video:
        return False

    if not hasattr(viewer, "get_frame"):
        msg.warning(f"enable_recording: viewer {type(viewer).__name__} has no get_frame(); recording disabled.")
        return False

    if getattr(viewer, "_recording", False):
        msg.warning("enable_recording: viewer already has recording enabled; skipping re-config.")
        return False

    viewer._recording = _VideoRecording(
        default_video_folder=default_video_folder,
        async_save=async_save,
        num_skipped_frames=num_skipped_frames,
    )

    # Recording is inactive until start_clip() is called; this keeps the
    # output folder empty during plain viewer sessions and avoids capturing
    # frames before the user explicitly asks for it.

    # Wrap viewer functions
    viewer._recording.original_should_step = viewer.should_step
    viewer._recording.original_end_frame = viewer.end_frame
    viewer.should_step = MethodType(_should_step_with_record, viewer)
    viewer.end_frame = MethodType(_end_frame_with_record, viewer)

    # Monkey-patch recording functions into object
    viewer.generate_video = MethodType(_generate_video, viewer)
    viewer.reset_recording = MethodType(_reset_recording, viewer)
    viewer.start_clip = MethodType(_start_clip, viewer)
    viewer.finish_clip = MethodType(_finish_clip, viewer)

    if start_clip:
        viewer.start_clip(**kwargs)

    return True


def _clear_pngs(folder: str) -> int:
    """Delete all ``*.png`` files in ``folder`` and return the count removed."""
    if not os.path.isdir(folder):
        return 0
    files = glob.glob(os.path.join(folder, "*.png"))
    for f in files:
        try:
            os.remove(f)
        except OSError:
            pass
    return len(files)


def _should_step_with_record(self):
    recording: _VideoRecording = self._recording

    do_step = recording.original_should_step()
    if do_step:
        recording.capture_pending = True
    return do_step


def _end_frame_with_record(self):
    recording: _VideoRecording = self._recording

    recording.original_end_frame()

    if not (recording.record_video and recording.recording_active and not recording.recording_stopped):
        # Even when not actively recording, clear any stale pending flag so we
        # do not capture a leftover frame on the next start_clip.
        recording.capture_pending = False
        return

    if not recording.capture_pending:
        return

    recording.capture_pending = False
    _capture_frame(self, recording)

    if recording.clip_max_frames is not None:
        captured = recording.img_idx - recording.num_skipped_frames
        if captured >= recording.clip_max_frames:
            self.finish_clip()


def _finish_clip(self):
    """Stop the current clip, flush async saves, and write the video file."""
    recording = self._recording

    if not recording.recording_active:
        return

    captured = recording.img_idx - recording.num_skipped_frames
    recording.recording_active = False
    recording.recording_stopped = True
    msg.notif(f"Clip captured: {captured} frames")

    for t in recording.save_threads:
        t.join()
    recording.save_threads.clear()

    out_path = recording.clip_output_path
    self.generate_video(
        output_filename=out_path,
        fps=recording.clip_fps,
        keep_frames=recording.clip_keep_frames,
    )
    msg.notif(f"Video saved: {out_path}")

    on_done = recording.clip_on_done
    recording.clip_max_frames = None
    recording.clip_output_path = None
    recording.clip_on_done = None

    if on_done is not None:
        on_done()


def _reset_recording(self, video_folder: str | None = None) -> None:
    """Clear recorded PNGs and reset counters. Optionally switch folder.

    Joins any pending async-save threads so the folder is safe to clear.

    Args:
        video_folder: If given and different from the current one, switches
            the active output directory.
    """
    recording = self._recording

    for t in recording.save_threads:
        t.join()
    recording.save_threads.clear()

    if video_folder is not None and video_folder != recording.default_video_folder:
        recording.default_video_folder = video_folder
        recording.created_video_folder = not os.path.exists(video_folder)
    elif not os.path.exists(recording.default_video_folder):
        recording.created_video_folder = True
    os.makedirs(recording.default_video_folder, exist_ok=True)

    removed = _clear_pngs(recording.default_video_folder)
    if removed:
        msg.info(f"reset_recording: cleared {removed} PNG frames in {recording.default_video_folder}")

    recording.img_idx = 0
    recording.recording_stopped = False
    recording.capture_pending = False
    recording.frame_buffer = None


def _start_clip(
    self,
    output_path: str,
    max_frames: int,
    video_folder: str | None = None,
    fps: int = 60,
    keep_frames: bool = False,
    on_done: Callable[[], None] | None = None,
) -> None:
    """Begin a finite-length recording clip with automatic video file generation.

    Clears any existing PNGs in the output folder, resets the frame counter,
    and arms an auto-stop trigger at ``max_frames`` captured frames. When the
    target is hit, the video file is written, frames are optionally deleted, and
    ``on_done`` (if provided) is called.

    Args:
        output_path: Destination video file path.
        max_frames: Number of frames to capture before auto-stopping.
        video_folder: Optional override for the PNG output directory.
        fps: Frame rate to embed in the video file (should match ``viewer_fps``).
        keep_frames: If False, delete the per-frame PNGs after the video file is
            written.
        on_done: Optional callback invoked once the video file has been written.
    """
    recording = self._recording

    self.reset_recording(video_folder=video_folder)
    recording.clip_max_frames = max_frames
    recording.clip_output_path = output_path
    recording.clip_keep_frames = keep_frames
    recording.clip_on_done = on_done
    recording.clip_fps = fps
    recording.recording_active = True
    msg.notif(f"Recording started: {output_path} ({max_frames} frames -> {recording.default_video_folder})")


def _capture_frame(viewer, recording: _VideoRecording) -> bool:
    """Save the latest rendered frame as a PNG."""
    try:
        from PIL import Image
    except ImportError:
        msg.warning("PIL not installed. Frames cannot be saved as images.")
        msg.info("Install with: pip install pillow")
        return False

    if recording.img_idx >= recording.num_skipped_frames:
        frame = viewer.get_frame(target_image=recording.frame_buffer)
        if recording.frame_buffer is None:
            recording.frame_buffer = frame

        frame_np = frame.numpy()
        image = Image.fromarray(frame_np, mode="RGB")

        filename = os.path.join(
            recording.default_video_folder, f"{recording.img_idx - recording.num_skipped_frames:05d}.png"
        )

        if recording.async_save:
            t = threading.Thread(
                target=image.save,
                args=(filename,),
                daemon=False,
            )
            t.start()
            recording.save_threads.append(t)
            recording.save_threads = [s for s in recording.save_threads if s.is_alive()]
        else:
            image.save(filename)

    recording.img_idx += 1
    return True


def _generate_video(
    self,
    output_filename: str = "recording.mp4",
    fps: int = 60,
    keep_frames: bool = False,
    quality: int = 8,
    codec: str = "libx264",
) -> bool:
    """Stitch the recorded PNGs in ``default_video_folder`` into a video file."""
    try:
        import imageio_ffmpeg as ffmpeg  # noqa: PLC0415
    except ImportError:
        msg.warning("imageio-ffmpeg not installed. Frames saved but video not generated.")
        msg.info("Install with: pip install imageio-ffmpeg")
        return False
    try:
        from PIL import Image
    except ImportError:
        msg.warning("PIL not installed. Frames saved but video not generated.")
        msg.info("Install with: pip install pillow")
        return False
    import numpy as np  # noqa: PLC0415

    recording = self._recording

    if not recording.record_video or recording.img_idx <= recording.num_skipped_frames:
        msg.warning("No frames recorded, cannot generate video")
        return False

    for t in recording.save_threads:
        t.join()
    recording.save_threads.clear()

    frame_files = sorted(glob.glob(os.path.join(recording.default_video_folder, "*.png")))
    if not frame_files:
        msg.warning(f"No png frames found in {recording.default_video_folder}")
        return False

    msg.info(f"Generating video from {len(frame_files)} frames...")
    try:
        writer = ffmpeg.write_frames(
            output_filename,
            size=(self.renderer._screen_width, self.renderer._screen_height),
            fps=fps,
            codec=codec,
            macro_block_size=8,
            quality=quality,
        )
        writer.send(None)

        for frame_path in frame_files:
            img = Image.open(frame_path)
            frame_array = np.array(img)
            writer.send(frame_array)

        writer.close()
        msg.info(f"Video generated successfully: {output_filename}")

        if not keep_frames:
            msg.info("Deleting png frames...")
            for frame_path in frame_files:
                os.remove(frame_path)
            if recording.created_video_folder:
                try:
                    os.rmdir(recording.default_video_folder)
                except OSError:
                    pass
            msg.info("Frames deleted")

        return True

    except Exception as e:
        msg.warning(f"Failed to generate video: {e}")
        return False
