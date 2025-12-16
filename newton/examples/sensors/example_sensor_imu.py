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

import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd


@wp.kernel
def acc_to_color(
    alpha: float,
    imu_acc: wp.array(dtype=wp.vec3),
    buffer: wp.array(dtype=wp.vec3),
    color: wp.array(dtype=wp.vec3),
):
    """Kernel mapping an acceleration to a color, with exponential smoothing."""
    idx = wp.tid()
    if idx >= len(imu_acc):
        return

    stored = buffer[idx]

    limit = wp.vec3(40.0)
    acc = wp.max(wp.min(imu_acc[idx], limit), -limit)

    smoothed = (1.0 - alpha) * stored + alpha * acc
    buffer[idx] = smoothed

    c = wp.vec3(0.5) + 0.5 * (0.1 * wp.min(wp.abs(smoothed), wp.vec3(10.0)) - wp.vec3(0.5))
    color[idx] = wp.max(wp.min(c, wp.vec3(1.0)), wp.vec3(0.0))


class Example:
    def __init__(self, viewer):
        # setup simulation parameters first
        self.fps = 200
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        builder = newton.ModelBuilder()

        # add ground plane
        builder.add_ground_plane()

        # pendulum
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("axis_cube.usda"))
        axis_cube_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/AxisCube/VisualCube"))

        body = builder.add_body(key="mesh", xform=wp.transform(wp.vec3(0, 0, 1)))
        scale = 0.2

        self.visual_cube = builder.add_shape_mesh(
            body,
            scale=wp.vec3(scale),
            mesh=axis_cube_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False, density=0),
        )

        scale_filler = scale * 0.98

        visual_cube_filler = builder.add_shape_box(
            body,
            hx=scale_filler,
            hy=scale_filler,
            hz=scale_filler,
            cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False, density=0),
        )
        builder.add_shape_box(
            body, hx=scale, hy=scale, hz=scale, cfg=newton.ModelBuilder.ShapeConfig(is_visible=False, density=200)
        )
        imu_site = builder.add_site(body, key="imu_site")

        # finalize model
        self.model = builder.finalize()

        self.imu = newton.sensors.SensorIMU(self.model, [imu_site])

        self.solver = newton.solvers.SolverMuJoCo(self.model, njmax=100)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.buffer = wp.zeros(1, dtype=wp.vec3)
        self.colors = wp.zeros(1, dtype=wp.vec3)

        self.viewer.set_model(self.model)

        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.camera.pos = type(self.viewer.camera.pos)(3.0, 0.0, 2.0)
            self.viewer.camera.pitch = type(self.viewer.camera.pitch)(-20)

        self.viewer.update_shape_colors({visual_cube_filler: (0.1, 0.1, 0.1)})

        self.capture()

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

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

            # read IMU acceleration
            self.imu.update(self.state_0)
            # average and compute color
            wp.launch(acc_to_color, dim=1, inputs=[0.01, self.imu.accelerometer, self.buffer, self.colors])

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        self.viewer.update_shape_colors({self.visual_cube: self.colors.numpy()[0]})

    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create viewer and run
    example = Example(viewer)

    newton.examples.run(example, args)
