# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cable Calibration
#
# Fits the rest shape and bend stiffness of a simulated cable to camera
# recordings of a cable. While the search runs, the viewer plays the best
# cable so far through each recorded motion, in front of the recorded video.
# The recordings come from an evidence bundle: masks, camera poses and the
# gripper trajectory.
#
# Command: uv run --extra calibration -m newton.examples cable_calibration
#
###########################################################################

"""Calibrate a cable from an evidence bundle and show the fit while it runs.

Without --bundle, the example downloads its bundle from newton-assets. Install
newton[calibration]. The search runs in a child process, so the viewer stays
responsive. The example drives the search with ask() and tell(), so it can
send the poses of each new best cable to the viewer. calibrate() runs the same
loop in one call and writes the result to files.

The setup, the searched parameters and the solver settings in make_problem()
are illustrative; adapt them to your cable.
"""

from __future__ import annotations

import importlib
import logging
import logging.handlers
import multiprocessing
import queue
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils

# The calibration module is private until its public PR; import it from _src for now.
from newton._src.calibration.cable_setup import CableSetup
from newton._src.calibration.evaluate import CableParameters
from newton._src.calibration.evidence import CableEvidenceBundle
from newton._src.calibration.optimizer import CMASettings
from newton._src.calibration.problem import CableCalibrationProblem
from newton._src.calibration.result import CableCalibrationResult
from newton._src.calibration.search_space import CableSearchSpace

NUM_ELEMENTS = 8

# Chamfer loss of the search start on the demo bundle.
START_LOSS = 0.189


def make_problem(bundle_dir: str | Path) -> CableCalibrationProblem:
    """Build the calibration problem: the cable setup, the search space and the solver settings."""
    # The physical cable. The attachment is at the TCP with no offset, and the cable leaves it along +Z.
    setup = CableSetup(
        num_elements=NUM_ELEMENTS,
        segment_length=0.05,
        cable_radius=0.007,
        cable_mass=0.05,
        angle_parametrization="exp_map",
        attachment_transform=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        clamp_position=0.0,
        stretch_stiffness=1000.0,
    )

    # The search start: a straight cable. Fields that are not searched keep these values.
    start = CableParameters(
        angles=[(0.0, 0.0)] * NUM_ELEMENTS,
        bend_stiffness=5.0,
        twist_stiffness=5.0,
        bend_damping=0.04,
        twist_damping=0.04,
    )
    search_space = CableSearchSpace(
        start,
        search=["angles", "bend_stiffness"],
        bounds={"bend_stiffness": (0.01, 1000.0)},
        couple_twist=True,
    )

    # The damping is held fixed and the twist follows the bend.
    return CableCalibrationProblem.from_bundle(
        CableEvidenceBundle.load(bundle_dir),
        bundle_dir,
        setup,
        search_space,
        settle_mode="dynamic",
        settle_frames=500,
        sim_iterations=20,
    )


def _read_rgb(path: Path) -> np.ndarray:
    """Read one recorded frame as an ``(H, W, 3)`` uint8 array."""
    from PIL import Image  # noqa: PLC0415 -- part of newton[calibration]

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _camera_frame(goal):
    """Return the camera position, its rotation matrix and ``(width, height, fx, fy, cx, cy)``."""
    position = np.asarray(goal.sensor_pos, dtype=float)
    rotation = np.asarray(wp.quat_to_matrix(wp.quat(*goal.sensor_quat)), dtype=float).reshape(3, 3)
    if goal.camera_intrinsics is not None:
        intrinsics = tuple(float(v) for v in goal.camera_intrinsics)
    else:
        width, height = goal.render_size
        focal = 0.5 * height / np.tan(0.5 * np.radians(goal.fov_deg))
        intrinsics = (width, height, focal, focal, 0.5 * (width - 1), 0.5 * (height - 1))
    return position, rotation, intrinsics


def _video_quad(goal, setup: CableSetup):
    """Return the points, triangle indices and UVs of the picture behind the cable.

    The picture fills the camera's view at a plane past the cable, so from the
    camera the recording and the simulated cable line up. The camera looks along
    its local -z axis with +y up; pixel centres are at whole pixel coordinates.
    """
    position, rotation, (width, height, fx, fy, cx, cy) = _camera_frame(goal)
    cable_depth = -(rotation.T @ (np.asarray(goal.cable_start, dtype=float) - position))[2]
    depth = cable_depth + setup.num_elements * setup.segment_length
    corners = []
    for u, v in ((-0.5, -0.5), (width - 0.5, -0.5), (width - 0.5, height - 0.5), (-0.5, height - 0.5)):
        in_camera = np.array([(u - cx) / fx * depth, -(v - cy) / fy * depth, -depth])
        corners.append(position + rotation @ in_camera)
    points = wp.array(np.asarray(corners, dtype=np.float32), dtype=wp.vec3)
    indices = wp.array([0, 1, 2, 0, 2, 3], dtype=wp.int32)
    # The viewer flips images vertically on upload, so the top image row is at v = 1.
    uvs = wp.array([(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)], dtype=wp.vec2)
    return points, indices, uvs


def _camera_view(goal) -> dict:
    """Return the viewer camera pose of the recording camera, as ``set_camera()`` arguments."""
    position, rotation, _ = _camera_frame(goal)
    forward = rotation @ np.array([0.0, 0.0, -1.0])
    return {
        "pos": wp.vec3(*position),
        "pitch": float(np.degrees(np.arcsin(np.clip(forward[2], -1.0, 1.0)))),
        "yaw": float(np.degrees(np.arctan2(forward[1], forward[0]))),
    }


def fit(bundle_dir, device, warp_log_level, test, messages) -> None:
    """Run the search and send its progress and its result to ``messages``.

    Runs in a child process, so it has its own CUDA context: the search captures
    CUDA graphs, which cannot share a stream with the viewer. ``device`` and
    ``warp_log_level`` repeat the Warp settings of the parent process.
    """
    wp.config.log_level = warp_log_level
    if device is not None:
        wp.set_device(device)
    try:
        problem = make_problem(bundle_dir)
        progress = logging.getLogger("newton._src.calibration")
        progress.addHandler(logging.handlers.QueueHandler(messages))
        progress.setLevel(logging.INFO)
        progress.propagate = False

        # A short search in test mode keeps the example test fast.
        optimizer = CMASettings(seed=42, sigma0=0.1, maxiter=3 if test else 120, popsize=8 if test else 24)
        search = optimizer.start(problem.search_space)
        while search.stop_reason is None:
            best_loss = search.best_loss
            losses, poses = problem.evaluate(search.ask(), return_poses=True)
            search.tell(losses)
            # Send the poses of a new best candidate, one array per recorded motion.
            if search.best_loss < best_loss:
                best = losses.index(search.best_loss)
                messages.put(("fit", search.iteration, search.best_loss, [recording[best] for recording in poses]))
        result = problem.build_result(search.outcome(), optimizer=optimizer, artifacts={})
        messages.put(("result", result.to_dict()))
    except Exception as exc:
        messages.put(("error", f"{type(exc).__name__}: {exc}"))


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.test = args.test

        bundle_dir = args.bundle or newton.utils.download_asset("cable_calibration_demo")

        # The child process imports fit() by module name, not from the launcher's __main__.
        module = importlib.import_module("newton.examples.cable.example_cable_calibration")
        context = multiprocessing.get_context("spawn")
        self.messages = context.Queue()
        self.process = context.Process(
            target=module.fit,
            args=(
                str(bundle_dir),
                args.device,
                wp.config.log_level,
                args.test,
                self.messages,
            ),
            daemon=True,
        )
        self.process.start()

        self.status = "Fitting..."
        self.result = None
        # Iterations and best losses of the search so far, for the plot.
        self.loss_history = ([], [])

        # This process shows the best cable so far, playing the recorded motions one after another in a
        # loop. The search process sends the poses of each new best cable, so this process only plays
        # them back. It works on the CPU: a CUDA context here would wait for the search's GPU time slices.
        with wp.ScopedDevice("cpu"):
            problem = make_problem(bundle_dir)
            start = problem.candidate(problem.search_space.x0())
            self.display = problem.build_simulation(start, recording=0)
            self.viewer.set_model(self.display.model)
            self.poses = None
            self.recording = 0
            self.frame = 0
            self.motion_start = 0.0
            # (iteration, loss, poses) of a new best fit waiting for the next motion, and
            # (iteration, loss) of the fit on screen.
            self.next_fit = None
            self.shown_fit = None
            # (motion, video frame) of the recorded frame on the picture.
            self.video_shown = None

            # Per recorded motion, in the order of the poses: its first camera view, shown as a
            # picture behind the cable, and the recording camera's viewer pose.
            self.motions = []
            for goal in problem.goals:
                if any(motion["key"] == goal.recording_key for motion in self.motions):
                    continue
                recording = next(r for r in problem.bundle.recordings if r.label == goal.label)
                self.motions.append(
                    {
                        "key": goal.recording_key,
                        "video": [_read_rgb(Path(bundle_dir) / path) for path in recording.rgb_frames],
                        "times": np.asarray(goal.frame_times),
                        "quad": _video_quad(goal, problem.setup),
                        "camera": _camera_view(goal),
                    }
                )
            # One untransformed white instance; material (roughness, metallic, checker, texture_enable).
            self.video_instance = (
                wp.array([wp.transform_identity()], dtype=wp.transform),
                wp.array([wp.vec3(1.0, 1.0, 1.0)], dtype=wp.vec3),
                wp.array([wp.vec3(1.0, 1.0, 1.0)], dtype=wp.vec3),
                wp.array([wp.vec4(1.0, 0.0, 0.0, 1.0)], dtype=wp.vec4),
            )
        self.viewer.set_camera(**self.motions[0]["camera"])
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self._free_windows, position="free")
        # The replay has no more frames per second than this; rendering faster would only take GPU time from the search.
        if args.render_fps is None and not self.test:
            args.render_fps = self.display.fps

        # The example test checks the fit, so it waits for the search before its frames run.
        if self.test:
            while self.result is None:
                self._receive(timeout=1.0)

    def _receive(self, timeout: float | None = None):
        """Handle the messages from the search process. Wait up to ``timeout`` [s] for the first one."""
        while True:
            try:
                message = self.messages.get(timeout=timeout) if timeout else self.messages.get_nowait()
            except queue.Empty:
                if not self.process.is_alive() and self.result is None:
                    self._fail("the search process stopped without a result.")
                return
            timeout = None
            if isinstance(message, logging.LogRecord):
                self.status = message.getMessage()
                print(self.status)
                if hasattr(message, "vector"):
                    self.loss_history[0].append(float(message.iteration))
                    self.loss_history[1].append(message.loss)
            elif message[0] == "fit":
                # A new best fit starts with the next motion, so the playback does not jump mid-motion.
                self.next_fit = message[1:]
                if self.poses is None:
                    self._show_next_fit()
            elif message[0] == "error":
                self._fail(message[1])
                return
            else:
                self.result = result = CableCalibrationResult.from_dict(message[1])
                self.status = (
                    f"Fit status: {result.status} ({result.diagnostics['stop_reason']}); objective: "
                    f"{result.metrics['loss']:.6g}; bend stiffness: {result.fit['bend_stiffness']:.4g} N·m/rad"
                )
                print(self.status)

    def _fail(self, reason: str):
        """Report a failed search. The test stops; the viewer keeps running and shows the reason."""
        self.status = f"The search failed: {reason}"
        print(self.status)
        if self.test:
            raise RuntimeError(self.status)
        self.process = None

    def step(self):
        if self.result is None and self.process is not None:
            self._receive()
        # Until the search sends the poses of its start, the built cable stays as it is.
        if self.poses is None:
            return
        # The playback follows the wall clock, so a slow viewer frame skips frames instead of slowing the motion.
        self.frame = int((time.perf_counter() - self.motion_start) * self.display.fps)
        if self.frame >= len(self.poses[self.recording]):
            # Go on with the next recorded motion, seen from its recording camera, with a new best fit if one came in.
            self.recording = (self.recording + 1) % len(self.poses)
            self.viewer.set_camera(**self.motions[self.recording]["camera"])
            if self.next_fit is not None:
                self._show_next_fit()
            self.motion_start, self.frame = time.perf_counter(), 0
        self.display.state_0.body_q.assign(self.poses[self.recording][self.frame])

    def _show_next_fit(self):
        """Play the waiting new best fit from the start of the current motion."""
        iteration, loss, self.poses = self.next_fit
        self.next_fit = None
        self.shown_fit = (iteration, loss)
        self.motion_start = time.perf_counter()

    def render(self):
        motion = self.motions[self.recording]
        sim_time = self.frame / self.display.fps
        video_index = int(np.argmin(np.abs(motion["times"] - sim_time)))
        self.viewer.begin_frame(sim_time)
        self.viewer.log_state(self.display.state_0)
        # Upload the recorded frame only when it changes; the viewer keeps the mesh between frames.
        if (self.recording, video_index) != self.video_shown:
            self.video_shown = (self.recording, video_index)
            points, indices, uvs = motion["quad"]
            # The viewer applies a texture through an instance material, so the quad is one instance of a hidden mesh.
            self.viewer.log_mesh(
                "recording/frame",
                points,
                indices,
                uvs=uvs,
                texture=motion["video"][video_index],
                hidden=True,
                backface_culling=False,
            )
            self.viewer.log_instances("recording", "recording/frame", *self.video_instance)
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("Cable calibration")
        ui.separator()
        ui.text(self.status)
        ui.text("The replay shows the best fit so far in front of the recording.")

    def _free_windows(self, ui):
        """Draw the label of the fit on screen over the 3D view, and the loss per iteration in a window of its own."""
        from imgui_bundle import imgui, implot  # noqa: PLC0415 -- installed with the GL viewer

        if self.shown_fit is not None:
            iteration, loss = self.shown_fit
            viewport = imgui.get_main_viewport()
            imgui.set_next_window_pos(
                imgui.ImVec2(viewport.work_pos.x + viewport.work_size.x * 0.5, viewport.work_pos.y + 12.0),
                imgui.Cond_.always,
                imgui.ImVec2(0.5, 0.0),
            )
            imgui.set_next_window_bg_alpha(0.5)
            flags = (
                imgui.WindowFlags_.no_decoration
                | imgui.WindowFlags_.always_auto_resize
                | imgui.WindowFlags_.no_inputs
                | imgui.WindowFlags_.no_saved_settings
                | imgui.WindowFlags_.no_focus_on_appearing
                | imgui.WindowFlags_.no_nav
            )
            if imgui.begin("##fit on screen", None, flags)[0]:
                motion = self.motions[self.recording]["key"]
                imgui.text(f"Best fit (iteration {iteration})  ·  loss {loss:.4g}  ·  {motion}")
            imgui.end()

        # The viewer does not create an ImPlot context, so create it on first use.
        if implot.get_current_context() is None:
            implot.create_context()
        imgui.set_next_window_size(imgui.ImVec2(480, 300), imgui.Cond_.first_use_ever)
        if imgui.begin("Search progress (best so far)")[0]:
            iterations, losses = self.loss_history
            if iterations and implot.begin_plot("##best loss", imgui.ImVec2(-1, -1)):
                implot.setup_axes("iteration", "loss", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
                implot.plot_line("loss", np.asarray(iterations), np.asarray(losses))
                implot.end_plot()
        imgui.end()

    def test_final(self):
        self.result.validate()
        if self.result.status != "ok":
            raise AssertionError(f"the fit ended with status {self.result.status!r}")
        # Even the short search of test mode lowers the loss well below the start.
        if not self.result.metrics["loss"] < 0.9 * START_LOSS:
            raise AssertionError(f"the fit did not improve: loss {self.result.metrics['loss']:.4g}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--bundle",
            type=Path,
            default=None,
            help="Local directory containing bundle.json. If omitted, the bundle is downloaded from newton-assets.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
