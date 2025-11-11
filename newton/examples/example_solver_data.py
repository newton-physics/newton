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

import numpy as np
import warp as wp

import newton
import newton.examples


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
        body_sphere = builder.add_body(key="pendulum")
        builder.add_shape_sphere(body_sphere, radius=0.2)
        builder.add_shape_capsule(
            body_sphere, xform=wp.transform(p=(0, 0, 0.5)), radius=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0)
        )
        builder.add_joint_revolute(
            -1, body_sphere, parent_xform=wp.transform(p=(0.0, 0.0, 2.0)), child_xform=wp.transform(p=(0.0, 0.0, 1.0))
        )
        builder.joint_qd[0] = 3

        # finalize model
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverMuJoCo(self.model, njmax=100)

        self.solver.require_data("body_acceleration")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        for state in (self.state_0, self.state_1):
            self.solver.allocate_data(state)
        self.control = self.model.control()

        if viewer is not None:
            self.viewer.set_model(self.model)
            self.init_plot(scale_min=0, graph_size=(400, 200))
            self.viewer.register_ui_callback(self.render_plot, "side")

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

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        pendulum_acc = self.state_0.data.body_acceleration.numpy()[0]
        self.add_plot_point(pendulum_acc[2])
        print(f"Pendulum acceleration: {pendulum_acc}")

        self.sim_time += self.frame_dt

    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def init_plot(self, **plot_kwargs):
        self.cache = []
        self.avg = 4
        n_points = 200
        self.plot_kwargs = plot_kwargs
        self.data = np.zeros(n_points, dtype=np.float32)

    def add_plot_point(self, point):
        self.cache.append(point)
        if len(self.cache) == self.avg:
            self.data[0] = sum(self.cache) / self.avg
            self.data = np.roll(self.data, -1)
            self.cache.clear()

    def render_plot(self, imgui):
        if not self.viewer or not self.viewer.ui.is_available:
            return

        window_shape = (400, 250)
        io = self.viewer.ui.io
        imgui.set_next_window_pos(
            imgui.ImVec2(io.display_size[0] - window_shape[0] - 10, io.display_size[1] - window_shape[1] - 10)
        )
        imgui.set_next_window_size(imgui.ImVec2(*window_shape))

        if imgui.begin("Body Acceleration (World Z, clamped to 0)"):
            imgui.plot_lines("Body acceleration", self.data, **self.plot_kwargs)
        imgui.end()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create viewer and run
    example = Example(viewer)

    newton.examples.run(example, args)
