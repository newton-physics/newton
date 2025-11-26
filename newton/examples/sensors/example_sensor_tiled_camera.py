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
# upon pressing ENTER and saved to example_color.png and example_depth.png.
#
# Command: python -m newton.examples sensor_tiled_camera
#
###########################################################################

import math

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
from newton.sensors import TiledCameraSensor

from ...viewer import ViewerGL


class Example:
    def __init__(self, viewer):
        self.render_key_is_pressed = False

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
        self.tiled_camera_sensor = TiledCameraSensor(model=self.model, num_cameras=1, width=1280, height=720)
        self.tiled_camera_sensor.create_default_light()
        self.tiled_camera_sensor.assign_debug_colors_per_shape()
        self.tiled_camera_sensor.assign_default_checkerboard_material()
        if isinstance(self.viewer, ViewerGL):
            self.tiled_camera_sensor.compute_camera_rays(
                wp.array([math.radians(self.viewer.camera.fov)], dtype=wp.float32)
            )
        else:
            self.tiled_camera_sensor.compute_camera_rays(wp.array([math.radians(45.0)], dtype=wp.float32))
        self.tiled_camera_sensor_color_image = self.tiled_camera_sensor.create_color_image_output()
        self.tiled_camera_sensor_depth_image = self.tiled_camera_sensor.create_depth_image_output()

    def step(self):
        pass

    def render(self):
        if self.viewer.is_key_down("enter"):
            if not self.render_key_is_pressed:
                self.render_sensors()
            self.render_key_is_pressed = True
        else:
            self.render_key_is_pressed = False

        self.viewer.begin_frame(0.0)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def render_sensors(self):
        print("Rendering Tiled Camera Sensor")

        camera_transforms = None
        if isinstance(self.viewer, ViewerGL):
            camera_transforms = self.tiled_camera_sensor.convert_camera_to_warp_arrays([self.viewer.camera])
        else:
            camera_position = wp.vec3f(10.0, 0.0, 2.0)
            camera_orientation = wp.mat33f(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
            camera_transforms = wp.array(
                wp.transformf(camera_position, wp.quat_from_matrix(camera_orientation)), dtype=wp.transformf
            )

        self.tiled_camera_sensor.render(
            self.state, camera_transforms, self.tiled_camera_sensor_color_image, self.tiled_camera_sensor_depth_image
        )
        self.tiled_camera_sensor.save_color_image(self.tiled_camera_sensor_color_image, "example_color.png")
        self.tiled_camera_sensor.save_depth_image(self.tiled_camera_sensor_depth_image, "example_depth.png")

    def test(self):
        self.render_sensors()
        color_image = self.tiled_camera_sensor_color_image.numpy()
        assert color_image.shape == (1, 1, 1280 * 720)
        assert color_image.min() < color_image.max()

        depth_image = self.tiled_camera_sensor_depth_image.numpy()
        assert depth_image.shape == (1, 1, 1280 * 720)
        assert depth_image.min() < depth_image.max()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create viewer and run
    example = Example(viewer)

    newton.examples.run(example, args)
