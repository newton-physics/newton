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
# and line instances.
#
###########################################################################


from __future__ import annotations

import argparse
import math
import time

import warp as wp

import newton


def create_model() -> newton.Model:
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    return builder.finalize()


class Example:
    def __init__(self, viewer_type: str):
        # Create a minimal model and viewer
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        self.model = builder.finalize()

        if viewer_type == "usd":
            from newton.viewer import ViewerUSD  # noqa: PLC0415

            self.viewer = ViewerUSD(self.model, output_path="example_viewer.usd", num_frames=600)
        elif viewer_type == "rerun":
            from newton.viewer import ViewerRerun  # noqa: PLC0415

            self.viewer = ViewerRerun(self.model, server=True, launch_viewer=True)
        else:
            from newton.viewer import ViewerGL  # noqa: PLC0415

            self.viewer = ViewerGL(self.model)

        # No explicit mesh creation; we'll use viewer.log_shapes() below

        # self.colors and materials per instance
        self.col_sphere = wp.array([wp.vec3(1.0, 0.1, 0.1)], dtype=wp.vec3)
        self.col_box = wp.array([wp.vec3(0.1, 1.0, 0.1)], dtype=wp.vec3)
        self.col_cone = wp.array([wp.vec3(0.1, 0.4, 1.0)], dtype=wp.vec3)
        self.col_capsule = wp.array([wp.vec3(1.0, 1.0, 0.1)], dtype=wp.vec3)
        self.col_cylinder = wp.array([wp.vec3(0.8, 0.5, 0.2)], dtype=wp.vec3)

        # material = (metallic, roughness, checker, unused)
        self.mat_default = wp.array([wp.vec4(0.0, 0.7, 0.0, 0.0)], dtype=wp.vec4)

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

        if viewer_type == "gl":
            print("Viewer running. WASD/Arrow keys to move, drag to orbit, scroll to zoom. Close window to exit.")

        self.start = time.time()
        self.frame = 0

    def step(self):
        pass

    def render(self):
        t = time.time() - self.start

        # Begin frame with time
        self.viewer.begin_frame(t)

        # Render model-driven content (ground plane)
        self.viewer.log_state(self.model.state())

        # Clean layout: arrange objects in a line along X-axis
        # All objects at same height to avoid ground intersection
        base_height = 2.0
        base_left = -4.0
        spacing = 2.0

        # Simple rotation animations
        qy_slow = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.3 * t)
        qx_slow = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.2 * t)
        qz_slow = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.4 * t)

        # Sphere: gentle bounce at x = -6
        sphere_pos = wp.vec3(0.0, base_left, base_height + 0.3 * abs(math.sin(1.2 * t)))
        x_sphere_anim = wp.array([wp.transform(sphere_pos, qy_slow)], dtype=wp.transform)

        base_left += spacing

        # Box: rocking rotation at x = -3
        x_box_anim = wp.array([wp.transform([0.0, base_left, base_height], qx_slow)], dtype=wp.transform)
        base_left += spacing

        # Cone: spinning at origin (x = 0)
        x_cone_anim = wp.array([wp.transform([0.0, base_left, base_height], qy_slow)], dtype=wp.transform)
        base_left += spacing

        # Cylinder: spinning on different axis at x = 3
        x_cyl_anim = wp.array([wp.transform([0.0, base_left, base_height], qz_slow)], dtype=wp.transform)
        base_left += spacing

        # Capsule: gentle sway at x = 6
        capsule_pos = wp.vec3(0.3 * math.sin(0.8 * t), base_left, base_height)
        x_cap_anim = wp.array([wp.transform(capsule_pos, qy_slow)], dtype=wp.transform)
        base_left += spacing

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

        self.viewer.log_lines("coordinate_self.axes", self.axes_begins, self.axes_ends, self.axes_colors)

        # End frame (process events, render, present)
        self.viewer.end_frame()

        self.frame += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--viewer", choices=["gl", "usd", "rerun"], default="gl", help="Viewer backend to use.")
    args = parser.parse_args()

    example = Example(args.viewer)

    while example.viewer.is_running():
        example.step()
        example.render()

    example.viewer.close()
