# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Deformable Visual Gaussian
#
# Loads a soft Gaussian bear from USD, drops its tetrahedral simulation
# proxy onto a ground plane, and displays the evaluated Gaussian field
# through SensorTiledCamera.
#
# Command: python -m newton.examples deformable_visual_gaussian
#
###########################################################################

import atexit
import math
import subprocess
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.sensors import SensorTiledCamera


class _CameraRecorder:
    """Stream optional RGB and depth recordings to ffmpeg."""

    def __init__(self, output_path: str | None, mode: str, width: int, height: int, fps: int):
        self._processes = {}
        self._closed = False
        if mode == "none":
            return
        if not output_path:
            raise ValueError("--camera-record-output is required when recording is enabled")

        output = Path(output_path)
        if not output.suffix:
            output = output.with_suffix(".mp4")
        modes = ("rgb", "depth") if mode == "both" else (mode,)
        for stream_mode in modes:
            stream_output = (
                output if len(modes) == 1 else output.with_name(f"{output.stem}_{stream_mode}{output.suffix}")
            )
            stream_output.parent.mkdir(parents=True, exist_ok=True)
            self._processes[stream_mode] = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgba",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    str(fps),
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    str(stream_output),
                ],
                stdin=subprocess.PIPE,
            )
            print(f"Recording {stream_mode} camera view to {stream_output}")
        atexit.register(lambda: self.close(raise_on_error=False))

    @property
    def is_active(self) -> bool:
        """Whether any recording stream is open."""
        return bool(self._processes)

    def write(self, mode: str, rgba: wp.array[wp.uint8]) -> None:
        """Write one RGBA frame to a recording stream."""
        process = self._processes.get(mode)
        if process is not None and process.stdin is not None:
            process.stdin.write(np.ascontiguousarray(rgba.numpy()).tobytes())

    def close(self, raise_on_error: bool = True) -> None:
        """Close every recording stream."""
        if self._closed:
            return
        self._closed = True
        errors = []
        for mode, process in self._processes.items():
            if process.stdin is not None:
                process.stdin.close()
            return_code = process.wait()
            if return_code:
                errors.append(f"{mode}: status {return_code}")
        self._processes.clear()
        if errors and raise_on_error:
            raise RuntimeError(f"ffmpeg failed while recording the camera ({'; '.join(errors)})")


def _look_at_transform(position, target):
    position = np.asarray(position, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.column_stack((right, up, -forward)).astype(np.float32)
    return wp.transformf(wp.vec3f(*position), wp.quat_from_matrix(wp.mat33f(rotation)))


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder()
        # Tet damping has no AOUSD proposal attribute yet, so the importer
        # intentionally falls back to this builder default.
        builder.default_tet_k_damp = 10.0
        builder.add_usd(newton.examples.get_asset("bear_gaussian.usda"))
        builder.add_ground_plane(color=(0.22, 0.24, 0.28))
        builder.color()

        self.model = builder.finalize()
        if self.model.tet_count == 0 or self.model.deformable_visual_gaussian_count != 1:
            raise RuntimeError("bear_gaussian.usda must contain one volume and one Gaussian visual")
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 0.0
        self.model.soft_contact_mu = 0.8
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=10,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=False,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.visuals = self.model.deformable_visuals()
        self.initial_center_height = float(np.mean(self.state_0.particle_q.numpy()[:, 2]))

        self.viewer.set_model(self.model)
        self.viewer.set_deformable_visuals(self.visuals)
        self.viewer.show_triangles = False
        self.viewer.show_gaussians = False
        self.viewer.set_camera(pos=wp.vec3(1.4, -3.2, 1.7), pitch=-7.0, yaw=66.0)

        render_config = SensorTiledCamera.RenderConfig(
            gaussians_mode=SensorTiledCamera.GaussianRenderMode.QUALITY,
            gaussians_max_num_hits=32,
            enable_simulation_triangles=False,
        )
        self.sensor = SensorTiledCamera(self.model, default_render_config=render_config)
        self.sensor.utils.create_default_light(enable_shadows=False)
        self.camera_width = args.camera_width
        self.camera_height = args.camera_height
        self.camera_view = args.camera_view
        self.camera_rays = self.sensor.utils.compute_camera_rays_pinhole(
            self.camera_width,
            self.camera_height,
            camera_fovs=math.radians(args.camera_fov),
        )
        camera = _look_at_transform((1.45, -3.1, 1.35), (0.0, 0.0, 0.95))
        self.camera_transforms = wp.array([[camera]], dtype=wp.transformf, device=self.model.device)
        self.color_image = self.sensor.utils.create_color_image_output(self.camera_width, self.camera_height, 1)
        self.depth_image = self.sensor.utils.create_depth_image_output(self.camera_width, self.camera_height, 1)
        self.depth_rgba = wp.empty(
            (1, self.camera_height, self.camera_width, 4), dtype=wp.uint8, device=self.model.device
        )
        self.depth_range = wp.array([0.0, 5.0], dtype=wp.float32, device=self.model.device)
        self.color_rgba_tiled = None
        self.depth_rgba_tiled = None
        self.recorder = _CameraRecorder(
            args.camera_record_output,
            args.camera_record_mode,
            self.camera_width,
            self.camera_height,
            args.camera_record_fps,
        )
        self.capture()

    def capture(self):
        if self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def _render_camera(self):
        self.model.update_deformable_visuals(self.state_0, self.visuals)
        self.model.bvh_refit_shapes(self.state_0)
        self.model.bvh_refit_particles(self.state_0)
        self.sensor.update(
            self.state_0,
            self.camera_transforms,
            self.camera_rays,
            color_image=self.color_image,
            depth_image=self.depth_image,
            clear_data=SensorTiledCamera.GRAY_CLEAR_DATA,
            deformable_visuals=self.visuals,
        )
        color_rgba = self.sensor.utils.to_rgba_from_color(self.color_image)
        self.sensor.utils.to_rgba_from_depth(self.depth_image, depth_range=(0.0, 5.0), out_buffer=self.depth_rgba)
        return color_rgba

    def render(self):
        color_rgba = self._render_camera()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        if self.camera_view in ("rgb", "both"):
            self.viewer.log_image("camera/rgb", color_rgba, fullscreen=self.camera_view == "rgb")
        if self.camera_view in ("depth", "both"):
            self.viewer.log_image("camera/depth", self.depth_rgba, fullscreen=self.camera_view == "depth")
        self.viewer.end_frame()

        if self.recorder.is_active:
            self.color_rgba_tiled = self.sensor.utils.flatten_color_image_to_rgba(
                self.color_image, out_buffer=self.color_rgba_tiled
            )
            self.depth_rgba_tiled = self.sensor.utils.flatten_depth_image_to_rgba(
                self.depth_image, out_buffer=self.depth_rgba_tiled, depth_range=self.depth_range
            )
            self.recorder.write("rgb", self.color_rgba_tiled)
            self.recorder.write("depth", self.depth_rgba_tiled)

    def test_final(self):
        self._render_camera()
        self.recorder.close()

        positions = self.state_0.particle_q.numpy()
        assert np.all(np.isfinite(positions))
        assert float(np.mean(positions[:, 2])) < self.initial_center_height - 0.05
        assert float(np.min(positions[:, 2])) > -0.2

        tet_indices = self.model.tet_indices.numpy()
        tet_points = positions[tet_indices]
        signed_six_volumes = np.linalg.det(
            np.stack(
                [
                    tet_points[:, 1] - tet_points[:, 0],
                    tet_points[:, 2] - tet_points[:, 0],
                    tet_points[:, 3] - tet_points[:, 0],
                ],
                axis=2,
            )
        )
        assert np.all(signed_six_volumes > 0.0)

        color = self.color_image.numpy()
        depth = self.depth_image.numpy()
        assert color.shape == (1, 1, self.camera_height, self.camera_width)
        assert depth.shape == color.shape
        assert color.min() < color.max()
        assert depth.min() < depth.max()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--camera-view", choices=["none", "rgb", "depth", "both"], default="rgb")
        parser.add_argument("--camera-width", type=int, default=320)
        parser.add_argument("--camera-height", type=int, default=320)
        parser.add_argument("--camera-fov", type=float, default=48.0)
        parser.add_argument("--camera-record-mode", choices=["none", "rgb", "depth", "both"], default="none")
        parser.add_argument("--camera-record-output", type=str, default=None)
        parser.add_argument("--camera-record-fps", type=int, default=60)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
