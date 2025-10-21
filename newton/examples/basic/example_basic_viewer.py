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
# Example Viewer
#
# Shows how to use the Newton Viewer class to visualize various shapes
# and line instances without a Newton model.
#
# Command: python -m newton.examples basic_viewer
#
###########################################################################

import math

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples


class Example:
    def __init__(self, viewer):
        self.viewer = viewer

        # self.colors and materials per instance
        self.col_sphere = wp.array([wp.vec3(0.9, 0.1, 0.1)], dtype=wp.vec3)
        self.col_box = wp.array([wp.vec3(0.1, 0.9, 0.1)], dtype=wp.vec3)
        self.col_cone = wp.array([wp.vec3(0.1, 0.4, 0.9)], dtype=wp.vec3)
        self.col_capsule = wp.array([wp.vec3(0.9, 0.9, 0.1)], dtype=wp.vec3)
        self.col_cylinder = wp.array([wp.vec3(0.8, 0.5, 0.2)], dtype=wp.vec3)
        self.col_bunny = wp.array([wp.vec3(0.5, 0.2, 0.8)], dtype=wp.vec3)
        self.col_plane = wp.array([wp.vec3(0.125, 0.125, 0.15)], dtype=wp.vec3)

        # material = (metallic, roughness, checker, unused)
        self.mat_default = wp.array([wp.vec4(0.0, 0.7, 0.0, 0.0)], dtype=wp.vec4)
        self.mat_plane = wp.array([wp.vec4(0.5, 0.5, 1.0, 0.0)], dtype=wp.vec4)

        # MESH (bunny)
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))
        mesh_vertices = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())
        self.bunny_mesh = newton.Mesh(mesh_vertices, mesh_indices)
        self.bunny_mesh.finalize()

        # Demonstrate log_lines() with animated debug/visualization lines
        axis_eps = 0.01
        axis_length = 2.0
        self.axes_begins = wp.array(
            [
                wp.vec3(0.0, 0.0, axis_eps),  # X axis start
                wp.vec3(0.0, 0.0, axis_eps),  # Y axis start
                wp.vec3(0.0, 0.0, axis_eps),  # Z axis start
            ],
            dtype=wp.vec3,
        )

        self.axes_ends = wp.array(
            [
                wp.vec3(axis_length, 0.0, axis_eps),  # X axis end
                wp.vec3(0.0, axis_length, axis_eps),  # Y axis end
                wp.vec3(0.0, 0.0, axis_length + axis_eps),  # Z axis end
            ],
            dtype=wp.vec3,
        )

        self.axes_colors = wp.array(
            [
                wp.vec3(1.0, 0.0, 0.0),  # Red X
                wp.vec3(0.0, 1.0, 0.0),  # Green Y
                wp.vec3(0.0, 0.0, 1.0),  # Blue Z
            ],
            dtype=wp.vec3,
        )

        self.time = 0.0
        self.spacing = 2.0

    def gui(self, ui):
        ui.text("Custom UI text")
        _changed, self.time = ui.slider_float("Time", self.time, 0.0, 100.0)
        _changed, self.spacing = ui.slider_float("Spacing", self.spacing, 0.0, 10.0)

    def step(self):
        pass

    def render(self):
        # Begin frame with time
        self.viewer.begin_frame(self.time)

        # Clean layout: arrange objects in a line along X-axis
        # All objects at same height to avoid ground intersection
        base_height = 2.0
        base_left = -4.0

        # Simple rotation animations
        qy_slow = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.3 * self.time)
        qx_slow = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.2 * self.time)
        qz_slow = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.4 * self.time)

        # Sphere: gentle bounce at y = -4
        sphere_pos = wp.vec3(0.0, base_left, base_height + 0.3 * abs(math.sin(1.2 * self.time)))
        x_sphere_anim = wp.array([wp.transform(sphere_pos, qy_slow)], dtype=wp.transform)
        base_left += self.spacing

        # Box: rocking rotation at y = -2
        x_box_anim = wp.array([wp.transform([0.0, base_left, base_height], qx_slow)], dtype=wp.transform)
        base_left += self.spacing

        # Cone: spinning at origin (y = 0)
        x_cone_anim = wp.array([wp.transform([0.0, base_left, base_height], qy_slow)], dtype=wp.transform)
        base_left += self.spacing

        # Cylinder: spinning on different axis at y = 2
        x_cyl_anim = wp.array([wp.transform([0.0, base_left, base_height], qz_slow)], dtype=wp.transform)
        base_left += self.spacing

        # Capsule: gentle sway at y = 4
        capsule_pos = wp.vec3(0.3 * math.sin(0.8 * self.time), base_left, base_height)
        x_cap_anim = wp.array([wp.transform(capsule_pos, qy_slow)], dtype=wp.transform)
        base_left += self.spacing

        # Bunny: spinning at y = 6
        x_bunny_anim = wp.array([wp.transform([0.0, base_left, base_height], qz_slow)], dtype=wp.transform)
        base_left += self.spacing

        # Update instances via log_shapes
        self.viewer.log_shapes(
            "/sphere_instance",
            newton.GeoType.SPHERE,
            0.5,
            x_sphere_anim,
            self.col_sphere,
            self.mat_default,
        )
        self.viewer.log_shapes(
            "/box_instance",
            newton.GeoType.BOX,
            (0.5, 0.3, 0.8),
            x_box_anim,
            self.col_box,
            self.mat_default,
        )
        self.viewer.log_shapes(
            "/cone_instance",
            newton.GeoType.CONE,
            (0.4, 1.2),
            x_cone_anim,
            self.col_cone,
            self.mat_default,
        )
        self.viewer.log_shapes(
            "/cylinder_instance",
            newton.GeoType.CYLINDER,
            (0.35, 1.0),
            x_cyl_anim,
            self.col_cylinder,
            self.mat_default,
        )
        self.viewer.log_shapes(
            "/capsule_instance",
            newton.GeoType.CAPSULE,
            (0.3, 1.0),
            x_cap_anim,
            self.col_capsule,
            self.mat_default,
        )
        self.viewer.log_shapes(
            "/bunny_instance",
            newton.GeoType.MESH,
            (1.0, 1.0, 1.0),
            x_bunny_anim,
            self.col_bunny,
            self.mat_default,
            geo_src=self.bunny_mesh,
        )

        self.viewer.log_shapes(
            "/plane_instance",
            newton.GeoType.PLANE,
            (50.0, 50.0),
            wp.array([wp.transform_identity()], dtype=wp.transform),
            self.col_plane,
            self.mat_plane,
        )

        self.viewer.log_lines("/coordinate_axes", self.axes_begins, self.axes_ends, self.axes_colors)

        # End frame (process events, render, present)
        self.viewer.end_frame()

        self.time += 1.0 / 60.0

    def test(self):
        pass


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create viewer and run
    example = Example(viewer)
    newton.examples.run(example, args)
