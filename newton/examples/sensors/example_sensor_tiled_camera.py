# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tiled Camera Sensor
#
# Shows how to use the SensorTiledCamera class.
# The current view will be rendered using the Tiled Camera Sensor
# upon pressing ENTER and displayed in the side panel.
#
# Command: python -m newton.examples sensor_tiled_camera
#
###########################################################################

import ctypes
import math
import random

import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
from newton.sensors import SensorTiledCamera
from newton.viewer import ViewerGL

SEMANTIC_COLOR_CYLINDER = 0xFFFF0000
SEMANTIC_COLOR_SPHERE = 0xFFFFFF00
SEMANTIC_COLOR_CAPSULE = 0xFF00FFFF
SEMANTIC_COLOR_BOX = 0xFF0000FF
SEMANTIC_COLOR_MESH = 0xFF00FF00
SEMANTIC_COLOR_ROBOT = 0xFFFF00FF
SEMANTIC_COLOR_GAUSSIAN = 0xFFFF9900
SEMANTIC_COLOR_GROUND_PLANE = 0xFF444444


@wp.kernel(enable_backward=False)
def animate_franka(
    time: wp.float32,
    joint_type: wp.array(dtype=wp.int32),
    joint_dof_dim: wp.array(dtype=wp.int32, ndim=2),
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_limit_lower: wp.array(dtype=wp.float32),
    joint_limit_upper: wp.array(dtype=wp.float32),
    joint_q: wp.array(dtype=wp.float32),
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


@wp.kernel
def shape_index_to_semantic_rgb(
    shape_indices: wp.array(dtype=wp.uint32, ndim=4),
    colors: wp.array(dtype=wp.uint32),
    rgba: wp.array(dtype=wp.uint32, ndim=4),
):
    world_id, camera_id, y, x = wp.tid()
    shape_index = shape_indices[world_id, camera_id, y, x]
    if shape_index < colors.shape[0]:
        rgba[world_id, camera_id, y, x] = colors[shape_index]
    else:
        rgba[world_id, camera_id, y, x] = wp.uint32(0xFF000000)


@wp.kernel
def shape_index_to_random_rgb(
    shape_indices: wp.array(dtype=wp.uint32, ndim=4),
    rgba: wp.array(dtype=wp.uint32, ndim=4),
):
    world_id, camera_id, y, x = wp.tid()
    shape_index = shape_indices[world_id, camera_id, y, x]
    random_color = wp.randi(wp.rand_init(12345, wp.int32(shape_index)))
    rgba[world_id, camera_id, y, x] = wp.uint32(random_color) | wp.uint32(0xFF000000)


class Example:
    def __init__(self, viewer: ViewerGL, args):
        self.worlds_per_row = 6
        self.worlds_per_col = 4
        self.world_count_total = self.worlds_per_row * self.worlds_per_col

        self.time = 0.0
        self.time_delta = 0.005
        self.image_output = 0
        self.texture_id = 0
        self.show_sensor_output = True

        self.viewer = viewer
        if isinstance(self.viewer, ViewerGL):
            self.show_sensor_output = viewer.ui.is_available
            self.viewer.register_ui_callback(self.display, "free")

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
                )
                semantic_colors.append(SEMANTIC_COLOR_CYLINDER)
            if rng.random() < 0.5:
                builder.add_shape_sphere(
                    builder.add_body(xform=wp.transform(p=wp.vec3(-2.0, -2.0, 0.5), q=wp.quat_identity())), radius=0.5
                )
                semantic_colors.append(SEMANTIC_COLOR_SPHERE)
            if rng.random() < 0.5:
                builder.add_shape_capsule(
                    builder.add_body(xform=wp.transform(p=wp.vec3(-4.0, 0.0, 0.75), q=wp.quat_identity())),
                    radius=0.25,
                    half_height=0.5,
                )
                semantic_colors.append(SEMANTIC_COLOR_CAPSULE)
            if rng.random() < 0.5:
                builder.add_shape_box(
                    builder.add_body(xform=wp.transform(p=wp.vec3(-2.0, 2.0, 0.5), q=wp.quat_identity())),
                    hx=0.5,
                    hy=0.35,
                    hz=0.5,
                )
                semantic_colors.append(SEMANTIC_COLOR_BOX)
            if rng.random() < 0.5:
                builder.add_shape_mesh(
                    builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 4.0, 0.0), q=wp.quat(0.5, 0.5, 0.5, 0.5))),
                    mesh=bunny_mesh,
                    scale=(0.5, 0.5, 0.5),
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

        builder.add_ground_plane()
        semantic_colors.append(SEMANTIC_COLOR_GROUND_PLANE)

        self.model = builder.finalize()
        self.state = self.model.state()

        self.semantic_colors = wp.array(semantic_colors, dtype=wp.uint32)

        self.viewer.set_model(self.model)

        self.ui_padding = 10
        self.ui_side_panel_width = 300

        self.camera_count = 1
        self.sensor_render_width = 64
        self.sensor_render_height = 64

        if isinstance(self.viewer, ViewerGL):
            display_width = self.viewer.renderer.window.width - self.ui_side_panel_width - self.ui_padding * 4
            display_height = self.viewer.renderer.window.height - self.ui_padding * 2

            self.sensor_render_width = int(display_width // self.worlds_per_row)
            self.sensor_render_height = int(display_height // self.worlds_per_col)

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
        self.tiled_camera_sensor_depth_image = self.tiled_camera_sensor.utils.create_depth_image_output(
            self.sensor_render_width, self.sensor_render_height, self.camera_count
        )
        self.tiled_camera_sensor_normal_image = self.tiled_camera_sensor.utils.create_normal_image_output(
            self.sensor_render_width, self.sensor_render_height, self.camera_count
        )
        self.tiled_camera_sensor_shape_index_image = self.tiled_camera_sensor.utils.create_shape_index_image_output(
            self.sensor_render_width, self.sensor_render_height, self.camera_count
        )
        self.tiled_camera_sensor_albedo_image = self.tiled_camera_sensor.utils.create_albedo_image_output(
            self.sensor_render_width, self.sensor_render_height, self.camera_count
        )
        self.depth_range = wp.array([1.0, 100.0], dtype=wp.float32)

        if isinstance(self.viewer, ViewerGL):
            self.create_texture()

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
        if self.show_sensor_output:
            self.render_sensors()

        self.viewer.begin_frame(0.0)
        if not self.show_sensor_output:
            self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def render_sensors(self):
        self.tiled_camera_sensor.update(
            self.state,
            self.get_camera_transforms(),
            self.camera_rays,
            color_image=self.tiled_camera_sensor_color_image,
            depth_image=self.tiled_camera_sensor_depth_image,
            normal_image=self.tiled_camera_sensor_normal_image,
            shape_index_image=self.tiled_camera_sensor_shape_index_image,
            albedo_image=self.tiled_camera_sensor_albedo_image,
            clear_data=SensorTiledCamera.GRAY_CLEAR_DATA,
        )
        self.update_texture()

    def get_camera_transforms(self) -> wp.array(dtype=wp.transformf):
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

    def create_texture(self):
        from pyglet import gl  # noqa: PLC0415

        width = self.sensor_render_width * self.worlds_per_row
        height = self.sensor_render_height * self.worlds_per_col

        texture_id = gl.GLuint()
        gl.glGenTextures(1, texture_id)
        self.texture_id = texture_id.value

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, width, height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        pixel_buffer = gl.GLuint()
        gl.glGenBuffers(1, pixel_buffer)
        self.pixel_buffer = pixel_buffer.value
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pixel_buffer)
        gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, width * height * 4, None, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

        self.texture_buffer = wp.RegisteredGLBuffer(self.pixel_buffer)

    def update_texture(self):
        if not self.texture_id:
            return

        from pyglet import gl  # noqa: PLC0415

        texture_buffer = self.texture_buffer.map(
            dtype=wp.uint8,
            shape=(
                self.worlds_per_col * self.sensor_render_height,
                self.worlds_per_row * self.sensor_render_width,
                4,
            ),
        )
        if self.image_output == 0:
            self.tiled_camera_sensor.utils.flatten_color_image_to_rgba(
                self.tiled_camera_sensor_color_image,
                texture_buffer,
                self.worlds_per_row,
            )
        elif self.image_output == 1:
            self.tiled_camera_sensor.utils.flatten_color_image_to_rgba(
                self.tiled_camera_sensor_albedo_image,
                texture_buffer,
                self.worlds_per_row,
            )
        elif self.image_output == 2:
            self.tiled_camera_sensor.utils.flatten_depth_image_to_rgba(
                self.tiled_camera_sensor_depth_image,
                texture_buffer,
                self.worlds_per_row,
                self.depth_range,
            )
        elif self.image_output == 3:
            self.tiled_camera_sensor.utils.flatten_normal_image_to_rgba(
                self.tiled_camera_sensor_normal_image,
                texture_buffer,
                self.worlds_per_row,
            )
        elif self.image_output == 4:
            wp.launch(
                shape_index_to_semantic_rgb,
                self.tiled_camera_sensor_shape_index_image.shape,
                [self.tiled_camera_sensor_shape_index_image, self.semantic_colors],
                [self.tiled_camera_sensor_shape_index_image],
            )
            self.tiled_camera_sensor.utils.flatten_color_image_to_rgba(
                self.tiled_camera_sensor_shape_index_image,
                texture_buffer,
                self.worlds_per_row,
            )
        elif self.image_output == 5:
            wp.launch(
                shape_index_to_random_rgb,
                self.tiled_camera_sensor_shape_index_image.shape,
                [self.tiled_camera_sensor_shape_index_image],
                [self.tiled_camera_sensor_shape_index_image],
            )
            self.tiled_camera_sensor.utils.flatten_color_image_to_rgba(
                self.tiled_camera_sensor_shape_index_image,
                texture_buffer,
                self.worlds_per_row,
            )
        self.texture_buffer.unmap()

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pixel_buffer)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            self.sensor_render_width * self.worlds_per_row,
            self.sensor_render_height * self.worlds_per_col,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            ctypes.c_void_p(0),
        )
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def test_final(self):
        self.render_sensors()

        color_image = self.tiled_camera_sensor_color_image.numpy()
        assert color_image.shape == (24, 1, self.sensor_render_height, self.sensor_render_width)
        assert color_image.min() < color_image.max()

        depth_image = self.tiled_camera_sensor_depth_image.numpy()
        assert depth_image.shape == (24, 1, self.sensor_render_height, self.sensor_render_width)
        assert depth_image.min() < depth_image.max()

    def gui(self, ui):
        show_compile_kernel_info = False

        _, self.show_sensor_output = ui.checkbox("Show Sensor Output", self.show_sensor_output)
        self.viewer.show_gaussians = not self.show_sensor_output

        ui.separator()

        if ui.radio_button("Show Color Output", self.image_output == 0):
            self.image_output = 0
        if ui.radio_button("Show Albedo Output", self.image_output == 1):
            self.image_output = 1
        if ui.radio_button("Show Depth Output", self.image_output == 2):
            self.image_output = 2
        if ui.radio_button("Show Normal Output", self.image_output == 3):
            self.image_output = 3
        if ui.radio_button("Show Semantic Output", self.image_output == 4):
            self.image_output = 4
        if ui.radio_button("Show Shape Index Output", self.image_output == 5):
            self.image_output = 5

        ui.separator()
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

    def display(self, imgui):
        if not self.show_sensor_output:
            return

        line_color = imgui.get_color_u32(imgui.Col_.window_bg)

        width = self.viewer.renderer.window.width - self.ui_side_panel_width - self.ui_padding * 4
        height = self.viewer.renderer.window.height - self.ui_padding * 2

        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_size(self.viewer.ui.io.display_size)

        flags = (
            imgui.WindowFlags_.no_title_bar.value
            | imgui.WindowFlags_.no_mouse_inputs.value
            | imgui.WindowFlags_.no_bring_to_front_on_focus.value
            | imgui.WindowFlags_.no_scrollbar.value
        )

        if imgui.begin("Sensors", flags=flags):
            pos_x = self.ui_side_panel_width + self.ui_padding * 2
            pos_y = self.ui_padding

            if self.texture_id > 0:
                imgui.set_cursor_pos(imgui.ImVec2(pos_x, pos_y))
                imgui.image(imgui.ImTextureRef(self.texture_id), imgui.ImVec2(width, height))

            draw_list = imgui.get_window_draw_list()
            for x in range(1, self.worlds_per_row):
                draw_list.add_line(
                    imgui.ImVec2(pos_x + x * (width / self.worlds_per_row), pos_y),
                    imgui.ImVec2(pos_x + x * (width / self.worlds_per_row), pos_y + height),
                    line_color,
                    2.0,
                )
            for y in range(1, self.worlds_per_col):
                draw_list.add_line(
                    imgui.ImVec2(pos_x, pos_y + y * (height / self.worlds_per_col)),
                    imgui.ImVec2(pos_x + width, pos_y + y * (height / self.worlds_per_col)),
                    line_color,
                    2.0,
                )

        imgui.end()

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
