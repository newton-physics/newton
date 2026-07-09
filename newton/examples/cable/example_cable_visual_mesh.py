# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cable Visual Mesh
#
# A cable simulated as a chain of capsule rigid bodies (add_rod) drives a
# smooth, textured visual tube. Each visual vertex is rigidly bound to its
# nearest cable segment and follows that body's pose, so the checkerboard
# tube bends and swings with the cable. Demonstrates the rigid-body
# (cable/rod) path of the deformable visual-mesh workflow
# (https://github.com/newton-physics/newton/issues/3223).
#
# Command: uv run -m newton.examples cable_visual_mesh
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
from newton.viewer import ViewerGL


class _CameraRecorder:
    def __init__(self, output_path: str | None, mode: str, width: int, height: int, fps: int):
        self._processes = {}
        self._closed = False
        if mode == "none":
            return
        if not output_path:
            raise ValueError("--camera-record-output is required when --camera-record-mode is not 'none'")

        output = Path(output_path)
        if not output.suffix:
            output = output.with_suffix(".mp4")
        modes = ("rgb", "depth") if mode == "both" else (mode,)
        for stream_mode in modes:
            suffix = output.suffix or ".mp4"
            stream_output = output if len(modes) == 1 else output.with_name(f"{output.stem}_{stream_mode}{suffix}")
            stream_output.parent.mkdir(parents=True, exist_ok=True)
            command = [
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
            ]
            self._processes[stream_mode] = subprocess.Popen(command, stdin=subprocess.PIPE)
            print(f"Recording {stream_mode} camera view to {stream_output}")

        atexit.register(lambda: self.close(raise_on_error=False))

    def write(self, mode: str, rgba: wp.array):
        process = self._processes.get(mode)
        if process is None or process.stdin is None:
            return
        frame = np.ascontiguousarray(rgba.numpy()[0])
        process.stdin.write(frame.tobytes())

    def close(self, raise_on_error: bool = True):
        if self._closed:
            return
        self._closed = True
        for mode, process in self._processes.items():
            if process.stdin is not None:
                process.stdin.close()
            return_code = process.wait()
            if return_code != 0 and raise_on_error:
                raise RuntimeError(f"ffmpeg exited with status {return_code} while writing {mode} camera video")
        self._processes.clear()


def _look_at_transform(pos, target):
    pos_np = np.asarray(pos, dtype=np.float32)
    target_np = np.asarray(target, dtype=np.float32)
    forward = target_np - pos_np
    forward /= np.linalg.norm(forward) + 1.0e-12
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(forward, world_up))) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1.0e-12
    up = np.cross(right, forward)
    rotation = np.column_stack((right, up, -forward)).astype(np.float32)
    return wp.transformf(wp.vec3f(*pos_np), wp.quat_from_matrix(wp.mat33f(rotation)))


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        # Cable centerline: a straight horizontal rod, pinned at the left end so
        # it swings down and bends under gravity.
        num_elements = 40
        length = 2.0
        radius = 0.04
        z0 = 1.6
        nodes = np.stack(
            [np.linspace(0.0, length, num_elements + 1), np.zeros(num_elements + 1), np.full(num_elements + 1, z0)],
            axis=1,
        ).astype(np.float32)

        rod_bodies, _ = builder.add_rod(
            positions=[wp.vec3(*p) for p in nodes],
            radius=radius,
            stretch_stiffness=1.0e5,
            bend_stiffness=5.0e1,
            bend_damping=1.0e0,
            label="cable",
            body_frame_origin="com",
        )

        # Pin the first segment.
        first = rod_bodies[0]
        builder.body_mass[first] = 0.0
        builder.body_inv_mass[first] = 0.0
        builder.body_inertia[first] = wp.mat33(0.0)
        builder.body_inv_inertia[first] = wp.mat33(0.0)

        # High-resolution textured tube around the centerline, bound rigidly to
        # the cable segments. A larger radius hides the underlying capsules.
        verts, indices, uvs = self._tube_mesh(nodes, radius=radius * 1.3, segments=20)
        builder.add_deformable_visual_mesh(
            verts,
            indices,
            kind="body",
            bodies=rod_bodies,
            uvs=uvs,
            texture=self._checker_texture(),
            label="cable_skin",
        )

        builder.color()

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverVBD(self.model, iterations=self.iterations)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(1.0, -2.6, 1.2), pitch=-12.0, yaw=100.0)
        self._init_camera_sensor(args, camera_pos=(1.0, -2.6, 1.2), camera_target=(1.0, 0.0, 1.0))
        self.capture()

    @staticmethod
    def _tube_mesh(centerline: np.ndarray, radius: float, segments: int):
        """Build a tube (vertices, flat triangle indices, UVs) around a polyline."""
        n = len(centerline)
        tangents = np.gradient(centerline, axis=0)
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12

        verts, uvs = [], []
        for i, (p, t) in enumerate(zip(centerline, tangents, strict=True)):
            up = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            nrm = np.cross(t, up)
            nrm /= np.linalg.norm(nrm) + 1e-12
            binrm = np.cross(t, nrm)
            for j in range(segments):
                a = 2.0 * np.pi * j / segments
                offset = np.cos(a) * nrm + np.sin(a) * binrm
                verts.append(p + radius * offset)
                uvs.append([4.0 * i / (n - 1), j / segments])

        faces = []
        for i in range(n - 1):
            for j in range(segments):
                a = i * segments + j
                b = i * segments + (j + 1) % segments
                c = (i + 1) * segments + j
                d = (i + 1) * segments + (j + 1) % segments
                faces += [a, b, c, b, d, c]

        return (
            np.asarray(verts, dtype=np.float32),
            np.asarray(faces, dtype=np.int32),
            np.asarray(uvs, dtype=np.float32),
        )

    @staticmethod
    def _checker_texture(tiles: int = 8, size: int = 512) -> np.ndarray:
        """Build an RGB checkerboard texture (H, W, 3) uint8."""
        image = np.zeros((size, size, 3), dtype=np.uint8)
        step = size // tiles
        for i in range(tiles):
            for j in range(tiles):
                color = (235, 90, 40) if (i + j) % 2 else (40, 120, 255)
                image[i * step : (i + 1) * step, j * step : (j + 1) * step] = color
        return image

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        self._close_camera_recorder()
        p_lower = wp.vec3(-3.0, -3.0, -0.5)
        p_upper = wp.vec3(3.0, 3.0, 3.0)
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cable bodies are within a reasonable volume",
            lambda q, _qd: newton.math.vec_inside_limits(wp.transform_get_translation(q), p_lower, p_upper),
        )

    def render(self):
        self._render_camera_sensor()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def _init_camera_sensor(self, args, camera_pos, camera_target):
        self.camera_view = getattr(args, "camera_view", "none")
        self.camera_record_mode = getattr(args, "camera_record_mode", "none")
        self.camera_enabled = self.camera_view != "none" or self.camera_record_mode != "none"
        self.camera_recorder = None
        self.camera_frame = 0
        self.camera_num_frames = int(getattr(args, "num_frames", 0) or 0)
        if not self.camera_enabled:
            return

        self.camera_count = 1
        self.camera_width = int(getattr(args, "camera_width", 320))
        self.camera_height = int(getattr(args, "camera_height", 320))
        self.camera_depth_range = (
            float(getattr(args, "camera_depth_near", 0.0)),
            float(getattr(args, "camera_depth_far", 5.0)),
        )
        self.camera_fixed_transform = _look_at_transform(camera_pos, camera_target)

        self.tiled_camera_sensor = SensorTiledCamera(
            model=self.model,
            config=SensorTiledCamera.RenderConfig(
                enable_particles=False,
                enable_simulation_triangles=False,
                enable_textures=True,
                max_distance=self.camera_depth_range[1],
            ),
        )
        self.tiled_camera_sensor.utils.create_default_light(enable_shadows=True)

        fov = self.viewer.camera.fov if isinstance(self.viewer, ViewerGL) else float(getattr(args, "camera_fov", 45.0))
        self.camera_rays = self.tiled_camera_sensor.utils.compute_camera_rays_pinhole(
            self.camera_width, self.camera_height, camera_fovs=math.radians(fov)
        )
        self.camera_color_image = self.tiled_camera_sensor.utils.create_color_image_output(
            self.camera_width, self.camera_height, self.camera_count
        )
        self.camera_depth_image = self.tiled_camera_sensor.utils.create_depth_image_output(
            self.camera_width, self.camera_height, self.camera_count
        )
        self.camera_depth_rgba = wp.empty(
            (self.model.world_count * self.camera_count, self.camera_height, self.camera_width, 4),
            dtype=wp.uint8,
            device=self.camera_depth_image.device,
        )
        self.camera_recorder = _CameraRecorder(
            getattr(args, "camera_record_output", None),
            self.camera_record_mode,
            self.camera_width,
            self.camera_height,
            int(getattr(args, "camera_record_fps", self.fps)),
        )

    def _camera_transforms(self):
        if isinstance(self.viewer, ViewerGL):
            transform = wp.transformf(
                self.viewer.camera.pos,
                wp.quat_from_matrix(wp.mat33f(self.viewer.camera.get_view_matrix().reshape(4, 4)[:3, :3])),
            )
        else:
            transform = self.camera_fixed_transform
        return wp.array([[transform] * self.model.world_count], dtype=wp.transformf, device=self.model.device)

    def _render_camera_sensor(self):
        if not self.camera_enabled:
            return

        self.model.bvh_refit_shapes(self.state_0)
        self.model.bvh_refit_particles(self.state_0)
        self.tiled_camera_sensor.update(
            self.state_0,
            self._camera_transforms(),
            self.camera_rays,
            color_image=self.camera_color_image,
            depth_image=self.camera_depth_image,
            clear_data=SensorTiledCamera.GRAY_CLEAR_DATA,
        )
        utils = self.tiled_camera_sensor.utils
        color_rgba = utils.to_rgba_from_color(self.camera_color_image)
        utils.to_rgba_from_depth(
            self.camera_depth_image, depth_range=self.camera_depth_range, out_buffer=self.camera_depth_rgba
        )

        if self.camera_view in ("rgb", "both"):
            self.viewer.log_image("camera/rgb", color_rgba)
        if self.camera_view in ("depth", "both"):
            self.viewer.log_image("camera/depth", self.camera_depth_rgba)

        if self.camera_recorder is not None:
            self.camera_recorder.write("rgb", color_rgba)
            self.camera_recorder.write("depth", self.camera_depth_rgba)
            self.camera_frame += 1
            if self.camera_num_frames > 0 and self.camera_frame >= self.camera_num_frames:
                self._close_camera_recorder()

    def _close_camera_recorder(self):
        if self.camera_recorder is not None:
            self.camera_recorder.close()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--camera-view", choices=["none", "rgb", "depth", "both"], default="none")
        parser.add_argument("--camera-width", type=int, default=320)
        parser.add_argument("--camera-height", type=int, default=320)
        parser.add_argument("--camera-fov", type=float, default=45.0)
        parser.add_argument("--camera-depth-near", type=float, default=0.0)
        parser.add_argument("--camera-depth-far", type=float, default=5.0)
        parser.add_argument("--camera-record-mode", choices=["none", "rgb", "depth", "both"], default="none")
        parser.add_argument("--camera-record-output", type=str, default=None)
        parser.add_argument("--camera-record-fps", type=int, default=60)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
