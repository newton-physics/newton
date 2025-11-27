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


class Example:
    def __init__(self, viewer):
        self.enable_rendering = True
        self.color_image_texture = 0
        self.depth_image_texture = 0

        self.viewer = viewer

        builder = newton.ModelBuilder()

        # add ground plane
        builder.add_ground_plane()

        # SPHERE
        self.sphere_pos = wp.vec3(0.0, -2.0, 0.5)
        body_sphere = builder.add_body(xform=wp.transform(p=self.sphere_pos, q=wp.quat_identity()), key="sphere")
        builder.add_shape_sphere(body_sphere, radius=0.5)

        # CAPSULE
        self.capsule_pos = wp.vec3(0.0, 0.0, 0.75)
        body_capsule = builder.add_body(xform=wp.transform(p=self.capsule_pos, q=wp.quat_identity()), key="capsule")
        builder.add_shape_capsule(body_capsule, radius=0.25, half_height=0.5)

        # CYLINDER
        self.cylinder_pos = wp.vec3(0.0, -4.0, 0.5)
        body_cylinder = builder.add_body(xform=wp.transform(p=self.cylinder_pos, q=wp.quat_identity()), key="cylinder")
        builder.add_shape_cylinder(body_cylinder, radius=0.4, half_height=0.5)

        # BOX
        self.box_pos = wp.vec3(0.0, 2.0, 0.5)
        body_box = builder.add_body(xform=wp.transform(p=self.box_pos, q=wp.quat_identity()), key="box")
        builder.add_shape_box(body_box, hx=0.5, hy=0.35, hz=0.5)

        # MESH (bunny)
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        mesh_vertices = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        demo_mesh = newton.Mesh(mesh_vertices, mesh_indices)

        self.mesh_pos = wp.vec3(0.0, 4.0, 0.0)
        body_mesh = builder.add_body(xform=wp.transform(p=self.mesh_pos, q=wp.quat(0.5, 0.5, 0.5, 0.5)), key="mesh")
        builder.add_shape_mesh(body_mesh, mesh=demo_mesh)

        # finalize model
        self.model = builder.finalize()
        self.state = self.model.state()

        self.viewer.set_model(self.model)

        # Setup Tiled Camera Sensor
        self.tiled_camera_sensor = TiledCameraSensor(
            model=self.model,
            num_cameras=1,
            width=640,
            height=360,
            options=TiledCameraSensor.Options(default_light=True, default_light_shadows=True, colors_per_shape=True, checkerboard_texture=True),
        )
        if isinstance(self.viewer, ViewerGL):
            self.tiled_camera_sensor.compute_camera_rays(
                wp.array([math.radians(self.viewer.camera.fov)], dtype=wp.float32)
            )
        else:
            self.tiled_camera_sensor.compute_camera_rays(wp.array([math.radians(45.0)], dtype=wp.float32))
        self.tiled_camera_sensor_color_image = self.tiled_camera_sensor.create_color_image_output()
        self.tiled_camera_sensor_depth_image = self.tiled_camera_sensor.create_depth_image_output()
        self.create_textures()

    def step(self):
        pass

    def render(self):
        if self.enable_rendering:
            self.render_sensors()
        self.viewer.begin_frame(0.0)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def render_sensors(self):
        camera_transforms = None
        if isinstance(self.viewer, ViewerGL):
            camera_transforms = wp.array(
                [
                    wp.transformf(
                        self.viewer.camera.pos,
                        wp.quat_from_matrix(wp.mat33f(self.viewer.camera.get_view_matrix().reshape(4, 4)[:3, :3])),
                    )
                ],
                dtype=wp.transformf,
            )

        else:
            camera_position = wp.vec3f(10.0, 0.0, 2.0)
            camera_orientation = wp.mat33f(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
            camera_transforms = wp.array(
                [wp.transformf(camera_position, wp.quat_from_matrix(camera_orientation))], dtype=wp.transformf
            )

        self.tiled_camera_sensor.render(
            self.state, camera_transforms, self.tiled_camera_sensor_color_image, self.tiled_camera_sensor_depth_image
        )
        self.update_textures()

    def create_textures(self):
        checker_size = 64
        width = self.tiled_camera_sensor.render_context.width
        height = self.tiled_camera_sensor.render_context.height

        pattern = ((np.arange(height) // checker_size)[:, None] + (np.arange(width) // checker_size)) % 2 == 0
        pixels = np.where(pattern, 0x22, 0x33).astype(np.uint8)
        pixels = np.dstack([pixels, pixels, pixels])

        self.color_image_texture, self.depth_image_texture = gl.glGenTextures(2)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.color_image_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, pixels.tobytes()
        )

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_image_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, pixels.tobytes()
        )

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def update_textures(self):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.color_image_texture)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            self.tiled_camera_sensor.render_context.width,
            self.tiled_camera_sensor.render_context.height,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            self.tiled_camera_sensor.flatten_color_image(self.tiled_camera_sensor_color_image).tobytes(),
        )

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_image_texture)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            self.tiled_camera_sensor.render_context.width,
            self.tiled_camera_sensor.render_context.height,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            np.dstack(
                [self.tiled_camera_sensor.flatten_depth_image(self.tiled_camera_sensor_depth_image)] * 3
            ).tobytes(),
        )

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def test(self):
        self.render_sensors()
        color_image = self.tiled_camera_sensor_color_image.numpy()
        assert color_image.shape == (1, 1, 640 * 360)
        assert color_image.min() < color_image.max()

        depth_image = self.tiled_camera_sensor_depth_image.numpy()
        assert depth_image.shape == (1, 1, 640 * 360)
        assert depth_image.min() < depth_image.max()

    def gui(self, ui):
        width = 270
        height = width / self.tiled_camera_sensor.render_context.width * self.tiled_camera_sensor.render_context.height

        if ui.button("Pause Rendering" if self.enable_rendering else "Resume Rendering", ui.ImVec2(width, 30)):
            self.enable_rendering = not self.enable_rendering

        ui.text("Color Image")
        if self.color_image_texture > 0:
            ui.image(ui.ImTextureRef(self.color_image_texture), ui.ImVec2(width, height))

        ui.text("Depth Image")
        if self.depth_image_texture > 0:
            ui.image(ui.ImTextureRef(self.depth_image_texture), ui.ImVec2(width, height))


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create viewer and run
    example = Example(viewer)

    newton.examples.run(example, args)
