# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tiled Camera Sensor
#
# Shows how to use the SensorTiledCamera class and display its output
# via Viewer.log_image.
#
# Command: python -m newton.examples sensor_tiled_camera
#
###########################################################################

import math
import random

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
from newton.sensors import SensorTiledCamera
from newton.viewer import ViewerGL

SEMANTIC_COLOR_CYLINDER = (255, 0, 0)
SEMANTIC_COLOR_SPHERE = (255, 255, 0)
SEMANTIC_COLOR_CAPSULE = (0, 255, 255)
SEMANTIC_COLOR_BOX = (0, 0, 255)
SEMANTIC_COLOR_MESH = (0, 255, 0)
SEMANTIC_COLOR_ROBOT = (255, 0, 255)
SEMANTIC_COLOR_GAUSSIAN = (255, 153, 0)
SEMANTIC_COLOR_GROUND_PLANE = (68, 68, 68)


@wp.kernel(enable_backward=False)
def animate_franka(
    time: wp.float32,
    joint_type: wp.array[wp.int32],
    joint_dof_dim: wp.array2d[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
    joint_q: wp.array[wp.float32],
):
    tid = wp.tid()

    if joint_type[tid] == newton.JointType.FREE:
        return

    rng = wp.rand_init(1234, tid)
    num_linear_dofs = joint_dof_dim[tid, 0]
    num_angular_dofs = joint_dof_dim[tid, 1]
    q_start = joint_q_start[tid]
    qd_start = joint_qd_start[tid]
    for i in range(num_linear_dofs + num_angular_dofs):
        joint_q[q_start + i] = joint_limit_lower[qd_start + i] + (
            joint_limit_upper[qd_start + i] - joint_limit_lower[qd_start + i]
        ) * ((wp.sin(time + wp.randf(rng)) + 1.0) * 0.5)


class Example:
    def __init__(self, viewer: ViewerGL, args):
        self.worlds_per_row = 6
        self.worlds_per_col = 4
        self.world_count_total = self.worlds_per_row * self.worlds_per_col

        self.time = 0.0
        self.time_delta = 0.005

        self.viewer = viewer

        usd_stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
        bunny_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        robot_asset = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"
        robot_builder = newton.ModelBuilder()
        robot_builder.add_urdf(robot_asset, floating=False)

        gaussian = None
        if args.ply:
            gaussian = newton.Gaussian.create_from_ply(args.ply, args.min_response)

        builder = newton.ModelBuilder()

        semantic_colors = []

        rng = random.Random(1234)
        for _ in range(self.world_count_total):
            builder.begin_world()
            if rng.random() < 0.5:
                builder.add_shape_cylinder(
                    builder.add_body(xform=wp.transform(p=wp.vec3(0.0, -4.0, 0.5), q=wp.quat_identity())),
                    radius=0.4,
                    half_height=0.5,
                    color=(0.27, 0.47, 0.67),
                )
                semantic_colors.append(SEMANTIC_COLOR_CYLINDER)
            if rng.random() < 0.5:
                builder.add_shape_sphere(
                    builder.add_body(xform=wp.transform(p=wp.vec3(-2.0, -2.0, 0.5), q=wp.quat_identity())),
                    radius=0.5,
                    color=(0.40, 0.80, 0.93),
                )
                semantic_colors.append(SEMANTIC_COLOR_SPHERE)
            if rng.random() < 0.5:
                builder.add_shape_capsule(
                    builder.add_body(xform=wp.transform(p=wp.vec3(-4.0, 0.0, 0.75), q=wp.quat_identity())),
                    radius=0.25,
                    half_height=0.5,
                    color=(0.13, 0.53, 0.20),
                )
                semantic_colors.append(SEMANTIC_COLOR_CAPSULE)
            if rng.random() < 0.5:
                builder.add_shape_box(
                    builder.add_body(xform=wp.transform(p=wp.vec3(-2.0, 2.0, 0.5), q=wp.quat_identity())),
                    hx=0.5,
                    hy=0.35,
                    hz=0.5,
                    color=(0.80, 0.73, 0.27),
                )
                semantic_colors.append(SEMANTIC_COLOR_BOX)
            if rng.random() < 0.5:
                builder.add_shape_mesh(
                    builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 4.0, 0.0), q=wp.quat(0.5, 0.5, 0.5, 0.5))),
                    mesh=bunny_mesh,
                    scale=(0.5, 0.5, 0.5),
                    color=(0.93, 0.40, 0.47),
                )
                semantic_colors.append(SEMANTIC_COLOR_MESH)

            if gaussian is not None:
                builder.add_shape_gaussian(
                    body=builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.4), q=wp.quat_identity())),
                    gaussian=gaussian,
                )
                semantic_colors.append(SEMANTIC_COLOR_GAUSSIAN)

            builder.add_builder(robot_builder, xform=wp.transform(p=wp.vec3(2.0, 0.0, 0.0), q=wp.quat_identity()))
            semantic_colors.extend([SEMANTIC_COLOR_ROBOT] * robot_builder.shape_count)
            builder.end_world()

        builder.add_ground_plane(color=(0.6, 0.6, 0.6))
        semantic_colors.append(SEMANTIC_COLOR_GROUND_PLANE)

        self.model = builder.finalize()
        self.state = self.model.state()

        self.viewer.set_model(self.model)

        self.camera_count = 1
        self.sensor_render_width = 256
        self.sensor_render_height = 256

        # Setup Tiled Camera Sensor
        self.tiled_camera_sensor = SensorTiledCamera(model=self.model)
        self.tiled_camera_sensor.utils.create_default_light(enable_shadows=True)
        self.tiled_camera_sensor.utils.assign_checkerboard_material_to_all_shapes()

        fov = 45.0
        if isinstance(self.viewer, ViewerGL):
            fov = self.viewer.camera.fov

        self.camera_rays = self.tiled_camera_sensor.utils.compute_pinhole_camera_rays(
            self.sensor_render_width, self.sensor_render_height, math.radians(fov)
        )
        self.tiled_camera_sensor_color_image = self.tiled_camera_sensor.utils.create_color_image_output(
            self.sensor_render_width, self.sensor_render_height, self.camera_count
        )
        self.tiled_camera_sensor_albedo_image = self.tiled_camera_sensor.utils.create_albedo_image_output(
            self.sensor_render_width, self.sensor_render_height, self.camera_count
        )
        self.tiled_camera_sensor_depth_image = self.tiled_camera_sensor.utils.create_depth_image_output(
            self.sensor_render_width, self.sensor_render_height, self.camera_count
        )
        self.tiled_camera_sensor_normal_image = self.tiled_camera_sensor.utils.create_normal_image_output(
            self.sensor_render_width, self.sensor_render_height, self.camera_count
        )
        self.tiled_camera_sensor_shape_index_image = self.tiled_camera_sensor.utils.create_shape_index_image_output(
            self.sensor_render_width, self.sensor_render_height, self.camera_count
        )

        # Palette for the "semantic" debug view: looked up by shape index.
        # Indices written into shape_index_image come from builder shape order,
        # so the palette must be one entry per shape in that same order.
        assert len(semantic_colors) == self.model.shape_count, (
            f"semantic_colors out of sync with model: {len(semantic_colors)} vs {self.model.shape_count}"
        )
        self.semantic_palette = wp.array(
            np.asarray(semantic_colors, dtype=np.uint8),
            dtype=wp.uint8,
            device=self.tiled_camera_sensor_color_image.device,
        )

        device = self.tiled_camera_sensor_color_image.device
        n = self.world_count_total * self.camera_count
        H = self.sensor_render_height
        W = self.sensor_render_width
        self.depth_rgba = wp.empty((n, H, W, 4), dtype=wp.uint8, device=device)
        self.normal_rgba = wp.empty((n, H, W, 4), dtype=wp.uint8, device=device)
        self.shape_rgba = wp.empty((n, H, W, 4), dtype=wp.uint8, device=device)
        self.semantic_rgba = wp.empty((n, H, W, 4), dtype=wp.uint8, device=device)

        newton.geometry.build_bvh_shape(self.model, self.state)
        newton.geometry.build_bvh_particle(self.model, self.state)

    def step(self):
        wp.launch(
            animate_franka,
            self.model.joint_count,
            [
                self.time,
                self.model.joint_type,
                self.model.joint_dof_dim,
                self.model.joint_q_start,
                self.model.joint_qd_start,
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
            ],
            outputs=[self.state.joint_q],
        )
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.time += self.time_delta

    def render(self):
        self.render_sensors()

        self.viewer.begin_frame(0.0)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def render_sensors(self):
        newton.geometry.refit_bvh_shape(self.model, self.state)
        newton.geometry.refit_bvh_particle(self.model, self.state)
        self.tiled_camera_sensor.update(
            self.state,
            self.get_camera_transforms(),
            self.camera_rays,
            color_image=self.tiled_camera_sensor_color_image,
            albedo_image=self.tiled_camera_sensor_albedo_image,
            depth_image=self.tiled_camera_sensor_depth_image,
            normal_image=self.tiled_camera_sensor_normal_image,
            shape_index_image=self.tiled_camera_sensor_shape_index_image,
            clear_data=SensorTiledCamera.GRAY_CLEAR_DATA,
        )
        utils = self.tiled_camera_sensor.utils
        color_rgba = utils.to_rgba_from_color(self.tiled_camera_sensor_color_image)
        albedo_rgba = utils.to_rgba_from_color(self.tiled_camera_sensor_albedo_image)
        utils.to_rgba_from_depth(
            self.tiled_camera_sensor_depth_image, depth_range=(0.0, 10.0), out_buffer=self.depth_rgba
        )
        utils.to_rgba_from_normal(self.tiled_camera_sensor_normal_image, out_buffer=self.normal_rgba)
        utils.to_rgba_from_shape_index(self.tiled_camera_sensor_shape_index_image, out_buffer=self.shape_rgba)
        utils.to_rgba_from_shape_index(
            self.tiled_camera_sensor_shape_index_image, colors=self.semantic_palette, out_buffer=self.semantic_rgba
        )

        self.viewer.log_image("color", color_rgba)
        self.viewer.log_image("albedo", albedo_rgba)
        self.viewer.log_image("depth", self.depth_rgba)
        self.viewer.log_image("normal", self.normal_rgba)
        self.viewer.log_image("shape_index", self.shape_rgba)
        self.viewer.log_image("semantic", self.semantic_rgba)

    def get_camera_transforms(self) -> wp.array[wp.transformf]:
        if isinstance(self.viewer, ViewerGL):
            return wp.array(
                [
                    [
                        wp.transformf(
                            self.viewer.camera.pos,
                            wp.quat_from_matrix(wp.mat33f(self.viewer.camera.get_view_matrix().reshape(4, 4)[:3, :3])),
                        )
                    ]
                    * self.world_count_total
                ],
                dtype=wp.transformf,
            )
        return wp.array(
            [[wp.transformf(wp.vec3f(10.0, 0.0, 2.0), wp.quatf(0.5, 0.5, 0.5, 0.5))] * self.world_count_total],
            dtype=wp.transformf,
        )

    def test_final(self):
        self.render_sensors()

        expected_shape = (24, 1, self.sensor_render_height, self.sensor_render_width)

        color_image = self.tiled_camera_sensor_color_image.numpy()
        assert color_image.shape == expected_shape
        assert color_image.min() < color_image.max()

        depth_image = self.tiled_camera_sensor_depth_image.numpy()
        assert depth_image.shape == expected_shape
        assert depth_image.min() < depth_image.max()

        # Loose allocation-regression checks on the other outputs: just
        # verify the sensor wrote into arrays with the right shapes/dtypes.
        albedo_image = self.tiled_camera_sensor_albedo_image.numpy()
        assert albedo_image.shape == expected_shape
        assert albedo_image.dtype == np.uint32

        normal_image = self.tiled_camera_sensor_normal_image.numpy()
        assert normal_image.shape == (24, 1, self.sensor_render_height, self.sensor_render_width, 3)
        assert normal_image.dtype == np.float32

        shape_index_image = self.tiled_camera_sensor_shape_index_image.numpy()
        assert shape_index_image.shape == expected_shape
        assert shape_index_image.dtype == np.uint32

    def gui(self, ui):
        show_compile_kernel_info = False

        if ui.radio_button(
            "Gaussians: Fast",
            self.tiled_camera_sensor.render_config.gaussians_mode == SensorTiledCamera.GaussianRenderMode.FAST,
        ):
            if self.tiled_camera_sensor.render_config.gaussians_mode != SensorTiledCamera.GaussianRenderMode.FAST:
                self.tiled_camera_sensor.render_config.gaussians_mode = SensorTiledCamera.GaussianRenderMode.FAST
                show_compile_kernel_info = True

        if ui.radio_button(
            "Gaussians: Quality",
            self.tiled_camera_sensor.render_config.gaussians_mode == SensorTiledCamera.GaussianRenderMode.QUALITY,
        ):
            if self.tiled_camera_sensor.render_config.gaussians_mode != SensorTiledCamera.GaussianRenderMode.QUALITY:
                self.tiled_camera_sensor.render_config.gaussians_mode = SensorTiledCamera.GaussianRenderMode.QUALITY
                show_compile_kernel_info = True

        changed, value = ui.slider_float(
            "Min Transmittance",
            self.tiled_camera_sensor.render_config.gaussians_min_transmittance,
            0.0,
            1.0,
            "%.2f",
        )
        if changed:
            self.tiled_camera_sensor.render_config.gaussians_min_transmittance = value
            show_compile_kernel_info = True

        changed, value = ui.slider_int(
            "Max Num Hits",
            self.tiled_camera_sensor.render_config.gaussians_max_num_hits,
            1,
            40,
            "%d",
        )
        if changed:
            self.tiled_camera_sensor.render_config.gaussians_max_num_hits = value
            show_compile_kernel_info = True

        if show_compile_kernel_info:
            display_width = self.viewer.renderer.window.width
            display_height = self.viewer.renderer.window.height

            overlay_width = 200
            overlay_height = 100

            text_width, text_height = ui.calc_text_size("Rebuilding Kernels")

            ui.set_next_window_pos(
                ui.ImVec2((display_width - overlay_width) * 0.5, (display_height - overlay_height) * 0.5)
            )
            ui.set_next_window_size(ui.ImVec2(overlay_width, overlay_height))

            if ui.begin(
                "Message",
                flags=(
                    ui.WindowFlags_.no_title_bar.value
                    | ui.WindowFlags_.no_mouse_inputs.value
                    | ui.WindowFlags_.no_scrollbar.value
                ),
            ):
                ui.set_cursor_pos(ui.ImVec2((overlay_width - text_width) * 0.5, (overlay_height - text_height) * 0.5))
                ui.text("Rebuilding Kernels")
            ui.end()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--ply",
            help="Gaussian filename.",
        )
        parser.add_argument(
            "-min",
            "--min-response",
            type=float,
            default=0.1,
            help="Gaussian min response.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create viewer and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
