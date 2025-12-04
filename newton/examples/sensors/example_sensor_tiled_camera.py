# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Tiled Camera Sensor
#
# Shows how to use the TiledCameraSensor class.
# The current view will be rendered using the Tiled Camera Sensor
# upon pressing ENTER and displayed in the side panel.
#
# Command: python -m newton.examples sensor_tiled_camera
#
###########################################################################

import math

import numpy as np
import OpenGL.GL as gl
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
from newton.sensors import TiledCameraSensor
from ...viewer import ViewerGL


@wp.kernel
def animate_franka(time: wp.float32, joint_dof_dim: wp.array(dtype=wp.int32, ndim=2), joint_q_start: wp.array(dtype=wp.int32), joint_qd_start: wp.array(dtype=wp.int32), joint_q: wp.array(dtype=wp.float32), joint_limit_lower: wp.array(dtype=wp.float32), joint_limit_upper: wp.array(dtype=wp.float32)):
    tid = wp.tid()

    rng = wp.rand_init(1234, tid)

    num_linear_dofs = joint_dof_dim[tid, 0]
    num_angular_dofs = joint_dof_dim[tid, 1]
    q_start = joint_q_start[tid]
    qd_start = joint_qd_start[tid]
    for i in range(num_linear_dofs + num_angular_dofs):
        joint_q[q_start + i] = joint_limit_lower[qd_start + i] + (joint_limit_upper[qd_start + i] - joint_limit_lower[qd_start + i]) * wp.sin(time + wp.randf(rng))


class Example:
    def __init__(self, viewer: ViewerGL):
        self.num_worlds_per_row = 4
        self.num_worlds_per_col = 4

        self.time = 0.0
        self.time_delta = 0.005

        self.color_image_texture = 0
        self.depth_image_texture = 0
        self.show_rgb_image = True

        self.viewer = viewer
        self.viewer.register_ui_callback(self.display, "free")

        builder = newton.ModelBuilder()

        usd_stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))
        bunny_mesh = newton.Mesh(np.array(usd_geom.GetPointsAttr().Get()), np.array(usd_geom.GetFaceVertexIndicesAttr().Get()))

        # builder.add_shape_cylinder(builder.add_body(xform=wp.transform(p=wp.vec3(0.0, -4.0, 0.5), q=wp.quat_identity())), radius=0.4, half_height=0.5)
        # builder.add_shape_sphere(builder.add_body(xform=wp.transform(p=wp.vec3(-2.0, -2.0, 0.5), q=wp.quat_identity())), radius=0.5)
        # builder.add_shape_capsule(builder.add_body(xform=wp.transform(p=wp.vec3(-4.0, 0.0, 0.75), q=wp.quat_identity())), radius=0.25, half_height=0.5)
        # builder.add_shape_box(builder.add_body(xform=wp.transform(p=wp.vec3(-2.0, 2.0, 0.5), q=wp.quat_identity())), hx=0.5, hy=0.35, hz=0.5)
        # builder.add_shape_mesh(builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 4.0, 0.0), q=wp.quat(0.5, 0.5, 0.5, 0.5))), mesh=bunny_mesh)
        builder.add_urdf(newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf", floating=False)

        scene = newton.ModelBuilder()
        scene.replicate(builder, self.num_worlds_per_row * self.num_worlds_per_col)
        scene.add_ground_plane()

        self.model = scene.finalize()
        self.state = self.model.state()

        self.viewer.set_model(self.model)

        sensor_render_width = int(self.viewer.ui.io.display_size[0] // self.num_worlds_per_col)
        sensor_render_height = int(self.viewer.ui.io.display_size[1] // self.num_worlds_per_row)

        # Setup Tiled Camera Sensor
        self.tiled_camera_sensor = TiledCameraSensor(
            model=self.model,
            num_cameras=1,
            width=sensor_render_width,
            height=sensor_render_height,
            options=TiledCameraSensor.Options(
                default_light=True, default_light_shadows=True, colors_per_shape=True, checkerboard_texture=True
            ),
        )

        fov = 45.0
        if isinstance(self.viewer, ViewerGL):
            fov = self.viewer.camera.fov

        self.camera_rays = self.tiled_camera_sensor.compute_pinhole_camera_rays(math.radians(fov))
        self.tiled_camera_sensor_color_image = self.tiled_camera_sensor.create_color_image_output()
        self.tiled_camera_sensor_depth_image = self.tiled_camera_sensor.create_depth_image_output()
        self.create_textures()

    def step(self):
        wp.launch(animate_franka, self.model.joint_count, [self.time, self.model.joint_dof_dim, self.model.joint_q_start, self.model.joint_qd_start, self.model.joint_q, self.model.joint_limit_lower, self.model.joint_limit_upper])
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.time += self.time_delta

    def render(self):
        self.render_sensors()
        self.viewer.begin_frame(0.0)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def render_sensors(self):
        self.tiled_camera_sensor.render(
            self.state,
            self.get_camera_transforms(),
            self.camera_rays,
            self.tiled_camera_sensor_color_image,
            self.tiled_camera_sensor_depth_image,
        )
        self.update_textures()

    def get_camera_transforms(self) -> wp.array(dtype=wp.transformf):
        if isinstance(self.viewer, ViewerGL):
            return wp.array(
                [
                    [
                        wp.transformf(
                            self.viewer.camera.pos,
                            wp.quat_from_matrix(wp.mat33f(self.viewer.camera.get_view_matrix().reshape(4, 4)[:3, :3])),
                        )
                    ] * (self.num_worlds_per_row * self.num_worlds_per_col)
                ],
                dtype=wp.transformf,
            )

        camera_position = wp.vec3f(10.0, 0.0, 2.0)
        camera_orientation = wp.mat33f(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        return wp.array(
            [[wp.transformf(camera_position, wp.quat_from_matrix(camera_orientation))] * (self.num_worlds_per_row * self.num_worlds_per_col)], dtype=wp.transformf
        )

    def create_textures(self):
        width = self.tiled_camera_sensor.render_context.width * self.num_worlds_per_col
        height = self.tiled_camera_sensor.render_context.height * self.num_worlds_per_row

        self.color_image_texture, self.depth_image_texture = gl.glGenTextures(2)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.color_image_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_image_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def update_textures(self):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.color_image_texture)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            self.tiled_camera_sensor.render_context.width * self.num_worlds_per_col,
            self.tiled_camera_sensor.render_context.height * self.num_worlds_per_row,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            self.tiled_camera_sensor.flatten_color_image(self.tiled_camera_sensor_color_image, num_rows=self.num_worlds_per_row).tobytes(),
        )

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_image_texture)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            self.tiled_camera_sensor.render_context.width * self.num_worlds_per_col,
            self.tiled_camera_sensor.render_context.height * self.num_worlds_per_row,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            np.dstack(
                [self.tiled_camera_sensor.flatten_depth_image(self.tiled_camera_sensor_depth_image, num_rows=self.num_worlds_per_row)] * 3
            ).tobytes(),
        )

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def test_final(self):
        self.render_sensors()
        color_image = self.tiled_camera_sensor_color_image.numpy()
        assert color_image.shape == (1, 1, 640 * 360)
        assert color_image.min() < color_image.max()

        depth_image = self.tiled_camera_sensor_depth_image.numpy()
        assert depth_image.shape == (1, 1, 640 * 360)
        assert depth_image.min() < depth_image.max()

    def gui(self, ui):
        if ui.button("Toggle RGB / Depth Image", ui.ImVec2(280, 30)):
            self.show_rgb_image = not self.show_rgb_image

    def display(self, imgui):
        side_panel_width = 300
        padding = 10

        io = self.viewer.ui.io

        width = io.display_size[0] - side_panel_width - padding * 4
        height = io.display_size[1] - padding * 2

        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_size(io.display_size)

        flags = imgui.WindowFlags_.no_title_bar.value | imgui.WindowFlags_.no_mouse_inputs.value | imgui.WindowFlags_.no_bring_to_front_on_focus.value | imgui.WindowFlags_.no_scrollbar.value

        if imgui.begin("Sensors", flags=flags):
            if self.color_image_texture > 0:
                imgui.set_cursor_pos(imgui.ImVec2(side_panel_width + padding * 2, padding))
                if self.show_rgb_image:
                    imgui.image(imgui.ImTextureRef(self.color_image_texture), imgui.ImVec2(width, height))
                else:
                    imgui.image(imgui.ImTextureRef(self.depth_image_texture), imgui.ImVec2(width, height))
        imgui.end()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create viewer and run
    example = Example(viewer)

    newton.examples.run(example, args)
