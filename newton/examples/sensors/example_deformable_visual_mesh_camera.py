# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Deformable Visual Mesh Camera
#
# Shows one scene with a cable, cloth, and soft volume. Each object drives a
# textured deformable visual mesh, and three tiled cameras focus on the three
# skinned meshes. Use this example to check that viewer images, camera RGB,
# depth, and optional recordings see the same skinned visual geometry.
#
# Command: uv run -m newton.examples deformable_visual_mesh_camera
#
###########################################################################

import argparse
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
        frame = np.ascontiguousarray(rgba.numpy())
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

        builder = self._build_model_builder(args)

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 1.0e2
        self.model.soft_contact_mu = 0.5

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        self.viewer.show_triangles = False
        self.viewer.set_camera(pos=wp.vec3(0.0, -4.4, 1.8), pitch=-10.0, yaw=90.0)

        self._init_camera_sensor(args)
        self.capture()

    @staticmethod
    def _checker_texture() -> str:
        """Return the shared procedural/USD checker texture."""
        return newton.examples.get_asset("deformable_visual_checker.ppm")

    @staticmethod
    def _tube_mesh(centerline: np.ndarray, radius: float, segments: int):
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

    def _build_model_builder(self, args):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        if getattr(args, "load_from_usd", False):
            import_result = builder.add_usd(
                newton.examples.get_asset("deformable_visual_mesh_camera.usda"),
                return_deformable_results=True,
            )
            self._apply_usd_procedural_parity_overrides(builder, import_result)
            builder.color()
            return builder

        self._add_cloth_obstacle(builder, x_offset=0.0)
        self._add_cable(builder, x_offset=-2.4)
        self._add_cloth(builder, x_offset=0.0)
        self._add_volume(builder, x_offset=2.4)
        builder.color()
        return builder

    @staticmethod
    def _apply_usd_procedural_parity_overrides(builder: newton.ModelBuilder, import_result: dict):
        """Patch values the current USD importer cannot express directly yet."""

        cable_bodies, cable_joints = import_result["path_cable_map"]["/World/Cable/Sim"]
        first_body = cable_bodies[0]
        builder.body_mass[first_body] = 0.0
        builder.body_inv_mass[first_body] = 0.0
        builder.body_inertia[first_body] = wp.mat33(0.0)
        builder.body_inv_inertia[first_body] = wp.mat33(0.0)

        # Gap: AOUSD curve material import handles stiffness but not cable bend damping yet.
        # The cable joint stores stretch first and bend/twist second, so patch only the angular slot.
        for joint in cable_joints:
            dof_start = builder.joint_qd_start[joint]
            linear_dim, angular_dim = builder.joint_dof_dim[joint]
            for dof in range(dof_start + linear_dim, dof_start + linear_dim + angular_dim):
                builder.joint_target_kd[dof] = 1.0

        cloth = import_result["path_cloth_map"]["/World/Cloth/Sim"]
        tri_start, tri_end = cloth["tri"]
        for tri in range(tri_start, tri_end):
            tri_ke, _tri_ka, _tri_kd, tri_drag, tri_lift = builder.tri_materials[tri]
            # Gap: AOUSD surface material import has no direct area stiffness or damping field.
            builder.tri_materials[tri] = (tri_ke, 1.0e3, 2.0e1, tri_drag, tri_lift)

        soft = import_result["path_soft_map"]["/World/Volume/Sim"]
        tet_start, tet_end = soft["tet"]
        for tet in range(tet_start, tet_end):
            k_mu, k_lambda, _k_damp = builder.tet_materials[tet]
            # Gap: AOUSD volume material import maps Young/Poisson to k_mu/k_lambda,
            # but does not carry Newton's tet damping coefficient yet.
            builder.tet_materials[tet] = (k_mu, k_lambda, 1.0e1)

    def _add_cable(self, builder: newton.ModelBuilder, x_offset: float):
        num_elements = 36
        length = 1.8
        radius = 0.035
        z0 = 1.55
        nodes = np.stack(
            [
                x_offset + np.linspace(-0.5 * length, 0.5 * length, num_elements + 1),
                np.zeros(num_elements + 1),
                np.full(num_elements + 1, z0),
            ],
            axis=1,
        ).astype(np.float32)

        rod_bodies, _ = builder.add_rod(
            positions=[wp.vec3(*p) for p in nodes],
            radius=radius,
            stretch_stiffness=1.0e5,
            bend_stiffness=5.0e1,
            bend_damping=1.0e0,
            label="camera_cable",
            body_frame_origin="com",
        )

        first = rod_bodies[0]
        builder.body_mass[first] = 0.0
        builder.body_inv_mass[first] = 0.0
        builder.body_inertia[first] = wp.mat33(0.0)
        builder.body_inv_inertia[first] = wp.mat33(0.0)

        verts, indices, uvs = self._tube_mesh(nodes, radius=radius * 1.3, segments=20)
        builder.add_deformable_visual_mesh(
            verts,
            indices,
            kind="body",
            bodies=rod_bodies,
            uvs=uvs,
            texture=self._checker_texture(),
            label="camera_cable_skin",
        )

    @staticmethod
    def _add_cloth_obstacle(builder: newton.ModelBuilder, x_offset: float):
        sphere_cfg = newton.ModelBuilder.ShapeConfig()
        sphere_cfg.density = 0.0
        sphere_cfg.ke = 1.0e5
        sphere_cfg.kd = 1.0e1
        sphere_cfg.mu = 0.5
        builder.add_shape_sphere(
            -1,
            xform=wp.transform(wp.vec3(x_offset, 0.0, 0.5), wp.quat_identity()),
            radius=0.35,
            cfg=sphere_cfg,
            label="cloth_obstacle",
            color=(0.35, 0.55, 0.8),
        )

    def _add_cloth(self, builder: newton.ModelBuilder, x_offset: float):
        dim = 34
        cell = 0.035
        span = dim * cell
        particle_start = builder.particle_count
        tri_start = len(builder.tri_indices)
        builder.add_cloth_grid(
            pos=wp.vec3(x_offset - 0.5 * span, -0.5 * span, 1.05),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim,
            dim_y=dim,
            cell_x=cell,
            cell_y=cell,
            mass=0.1,
            particle_radius=0.01,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=2.0e1,
        )
        particle_count = builder.particle_count - particle_start

        rest = np.asarray(builder.particle_q[particle_start:], dtype=np.float32)
        indices = (np.asarray(builder.tri_indices[tri_start:], dtype=np.int32) - particle_start).reshape(-1)
        spans = rest.max(axis=0) - rest.min(axis=0)
        u_axis, v_axis = np.argsort(spans)[-2:]
        uvs = (rest[:, [u_axis, v_axis]] - rest[:, [u_axis, v_axis]].min(0)) / spans[[u_axis, v_axis]]

        builder.add_deformable_visual_mesh(
            rest,
            indices,
            kind="particle",
            particles=np.arange(particle_start, particle_start + particle_count, dtype=np.int32),
            uvs=uvs.astype(np.float32),
            texture=self._checker_texture(),
            label="camera_cloth_skin",
        )

    def _add_volume(self, builder: newton.ModelBuilder, x_offset: float):
        dim = 3
        cell = 0.2
        length = dim * cell
        origin = wp.vec3(x_offset - 0.5 * length, -0.1, 1.0)
        tet_start = builder.tet_count
        builder.add_soft_grid(
            pos=origin,
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim,
            dim_y=dim,
            dim_z=dim,
            cell_x=cell,
            cell_y=cell,
            cell_z=cell,
            density=1.0e3,
            k_mu=2.0e4,
            k_lambda=2.0e4,
            k_damp=1.0e1,
            fix_left=True,
        )

        center = np.array([origin[0], origin[1], origin[2]], dtype=np.float32) + 0.5 * length
        sphere = newton.Mesh.create_sphere(radius=0.45 * length, num_latitudes=48, num_longitudes=48)
        visual_verts = np.asarray(sphere.vertices, dtype=np.float32) + center
        visual_indices = np.asarray(sphere.indices, dtype=np.int32)

        builder.add_deformable_visual_mesh(
            visual_verts,
            visual_indices,
            kind="tet",
            tet_range=(tet_start, builder.tet_count),
            uvs=sphere.uvs,
            texture=self._checker_texture(),
            label="camera_volume_skin",
        )

    def _init_camera_sensor(self, args):
        self.camera_view = getattr(args, "camera_view", "both")
        self.camera_record_mode = getattr(args, "camera_record_mode", "none")
        self.camera_count = 3
        self.camera_width = int(getattr(args, "camera_width", 320))
        self.camera_height = int(getattr(args, "camera_height", 320))
        self.camera_depth_range = (
            float(getattr(args, "camera_depth_near", 0.0)),
            float(getattr(args, "camera_depth_far", 5.0)),
        )

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

        fov = math.radians(float(getattr(args, "camera_fov", 45.0)))
        self.camera_rays = self.tiled_camera_sensor.utils.compute_camera_rays_pinhole(
            self.camera_width,
            self.camera_height,
            camera_fovs=np.full(self.camera_count, fov, dtype=np.float32),
        )
        self.camera_transforms = wp.array(
            [
                [_look_at_transform((-2.4, -2.5, 1.3), (-2.4, 0.0, 1.1))],
                [_look_at_transform((0.0, -1.8, 1.2), (0.0, 0.0, 0.55))],
                [_look_at_transform((2.4, -1.4, 1.45), (2.4, 0.2, 1.25))],
            ],
            dtype=wp.transformf,
            device=self.model.device,
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
        self.camera_depth_range_wp = wp.array(
            [self.camera_depth_range[0], self.camera_depth_range[1]],
            dtype=wp.float32,
            device=self.camera_depth_image.device,
        )
        self.camera_color_rgba_tiled = None
        self.camera_depth_rgba_tiled = None
        self.camera_frame = 0
        self.camera_num_frames = int(getattr(args, "num_frames", 0) or 0)
        self.camera_recorder = _CameraRecorder(
            getattr(args, "camera_record_output", None),
            self.camera_record_mode,
            self.camera_width * self.camera_count,
            self.camera_height,
            int(getattr(args, "camera_record_fps", self.fps)),
        )

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

    def render(self):
        self._render_camera_sensor()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def _render_camera_sensor(self):
        self.model.bvh_refit_shapes(self.state_0)
        self.model.bvh_refit_particles(self.state_0)
        self.tiled_camera_sensor.update(
            self.state_0,
            self.camera_transforms,
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
            self.camera_color_rgba_tiled = utils.flatten_color_image_to_rgba(
                self.camera_color_image, out_buffer=self.camera_color_rgba_tiled, worlds_per_row=self.camera_count
            )
            self.camera_depth_rgba_tiled = utils.flatten_depth_image_to_rgba(
                self.camera_depth_image,
                out_buffer=self.camera_depth_rgba_tiled,
                worlds_per_row=self.camera_count,
                depth_range=self.camera_depth_range_wp,
            )
            self.camera_recorder.write("rgb", self.camera_color_rgba_tiled)
            self.camera_recorder.write("depth", self.camera_depth_rgba_tiled)
            self.camera_frame += 1
            if self.camera_num_frames > 0 and self.camera_frame >= self.camera_num_frames:
                self._close_camera_recorder()

    def _close_camera_recorder(self):
        if self.camera_recorder is not None:
            self.camera_recorder.close()

    def test_final(self):
        self._render_camera_sensor()
        self._close_camera_recorder()

        expected_shape = (1, self.camera_count, self.camera_height, self.camera_width)
        color_image = self.camera_color_image.numpy()
        assert color_image.shape == expected_shape
        for camera_index in range(self.camera_count):
            assert color_image[0, camera_index].min() < color_image[0, camera_index].max()

        depth_image = self.camera_depth_image.numpy()
        assert depth_image.shape == expected_shape
        for camera_index in range(self.camera_count):
            assert depth_image[0, camera_index].min() < depth_image[0, camera_index].max()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--load-from-usd",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Load the deformable simulation and graphics meshes from a checked-in AOUSD asset.",
        )
        parser.add_argument("--camera-view", choices=["none", "rgb", "depth", "both"], default="both")
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
