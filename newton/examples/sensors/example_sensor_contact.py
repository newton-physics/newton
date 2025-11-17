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
# Example Contact Sensor
#
# Shows how to use the ContactSensor class to evaluate both net contact
# forces and contact forces between individual objects.
# The flap has a contact sensor registering the net contact force of the
# objects on top. The upper and lower plates' sensors will only register
# contacts with the cube and with the ball, respectively.
#
# Command: python -m newton.examples sensor_contact
#
###########################################################################

import re

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import Contacts
from newton.sensors import ContactSensor, populate_contacts
from newton.tests.unittest_utils import find_nonfinite_members


class Example:
    def __init__(self, viewer, num_worlds=1):
        # setup simulation parameters first
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.reset_interval = 5

        self.num_worlds = num_worlds

        self.viewer = viewer
        self.plot_window = ViewerPlot(viewer, "Flap Contact Force", scale_min=0, graph_size=(400, 200))
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.register_ui_callback(self.plot_window.render, "free")

        world_builder = newton.ModelBuilder()
        world_builder.add_usd(newton.examples.get_asset("contact_sensor_scene.usda"))
        newton.solvers.SolverMuJoCo.register_custom_attributes(world_builder)

        builder = newton.ModelBuilder()
        builder.replicate(world_builder, self.num_worlds, spacing=(1.0, 1.0, 0.0))

        builder.add_ground_plane()
        # stores contact info required by contact sensors
        self.contacts = Contacts(0, 0)

        # finalize model
        self.model = builder.finalize()

        self.flap_contact_sensor = ContactSensor(self.model, sensing_obj_shapes="*Flap", verbose=True)

        self.plate_contact_sensor = ContactSensor(
            self.model,
            sensing_obj_shapes=".*Plate.*",
            counterpart_shapes=".*Cube.*|.*Sphere.*",
            match_fn=lambda string, pat: re.match(pat, string),
            include_total=False,
            verbose=True,
        )

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            njmax=100,
            nconmax=100,
            cone="pyramidal",
            impratio=1,
        )

        self.viewer.set_model(self.model)

        self.plates_touched = 2 * [False]
        self.shape_colors = {
            "/env/Plate1": 3 * [0.4],
            "/env/Plate2": 3 * [0.4],
            "/env/Sphere": [1.0, 0.4, 0.2],
            "/env/Cube": [0.2, 0.4, 0.8],
            "/env/Flap": 3 * [0.8],
        }
        self.shape_map = {key: s for s, key in enumerate(self.model.shape_key)}

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.control = self.model.control()
        hinge_joint_idx = self.model.joint_key.index("/env/Hinge")
        self.hinge_joint_q_start = int(self.model.joint_q_start.numpy()[hinge_joint_idx])

        self.next_reset = 0.0

        # store initial state for reset
        self.initial_joint_q = wp.clone(self.state_0.joint_q)
        self.initial_joint_qd = wp.clone(self.state_0.joint_qd)

        if wp.get_device().is_cuda:
            self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.sim_time >= self.next_reset:
            self.reset()

        hinge_angle = min(self.sim_time / 3, 1.6)
        self.control.joint_target_pos[self.hinge_joint_q_start : self.hinge_joint_q_start + 1].fill_(hinge_angle)

        with wp.ScopedTimer("step", active=False):
            if self.graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

        populate_contacts(self.contacts, self.solver)
        self.plate_contact_sensor.eval(self.contacts)

        net_force = self.plate_contact_sensor.net_force.numpy()
        for i in range(2):
            if np.abs(net_force[i, i]).max() == 0:
                continue
            if self.plates_touched[i]:
                continue

            # color newly touched plate
            plate = self.plate_contact_sensor.sensing_objs[i][0]
            obj = self.plate_contact_sensor.counterparts[i][0]
            obj_key = self.model.shape_key[obj]
            self.plates_touched[i] = True
            print(f"Plate {self.model.shape_key[plate]} was touched by counterpart {obj_key}")
            self.viewer.update_shape_colors({plate: self.shape_colors[obj_key]})

        self.flap_contact_sensor.eval(self.contacts)
        self.plot_window.add_point(np.abs(self.flap_contact_sensor.net_force.numpy()[0, 0, 2]))
        self.sim_time += self.frame_dt

    def reset(self):
        self.sim_time = 0
        self.next_reset = self.sim_time + self.reset_interval
        self.viewer.update_shape_colors({self.shape_map[s]: v for s, v in self.shape_colors.items()})
        self.plates_touched = 2 * [False]

        print("Resetting")
        # Restore initial joint positions and velocities in-place.
        self.state_0.joint_q.assign(self.initial_joint_q)
        self.state_0.joint_qd.assign(self.initial_joint_qd)
        # Recompute forward kinematics to refresh derived state.
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        # pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.0,
        )
        assert len(find_nonfinite_members(self.flap_contact_sensor)) == 0
        assert len(find_nonfinite_members(self.plate_contact_sensor)) == 0


class ViewerPlot:
    """ImGui plot window"""

    def __init__(self, viewer=None, title="Plot", n_points=200, avg=4, **kwargs):
        self.viewer = viewer
        self.avg = avg
        self.title = title
        self.data = np.zeros(n_points, dtype=np.float32)
        self.plot_kwargs = kwargs
        self.cache = []

    def add_point(self, point):
        self.cache.append(point)
        if len(self.cache) == self.avg:
            self.data[0] = sum(self.cache) / self.avg
            self.data = np.roll(self.data, -1)
            self.cache.clear()

    def render(self, imgui):
        """
        Render the replay UI controls.

        Args:
            imgui: The ImGui object passed by the ViewerGL callback system
        """
        if not self.viewer or not self.viewer.ui.is_available:
            return

        io = self.viewer.ui.io

        # Position the replay controls window
        window_shape = (400, 350)
        imgui.set_next_window_pos(
            imgui.ImVec2(io.display_size[0] - window_shape[0] - 10, io.display_size[1] - window_shape[1] - 10)
        )
        imgui.set_next_window_size(imgui.ImVec2(*window_shape))

        flags = imgui.WindowFlags_.no_resize.value

        if imgui.begin(self.title, flags=flags):
            imgui.text("Flap contact force")
            imgui.plot_lines("Force", self.data, **self.plot_kwargs)
        imgui.end()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=1, help="Total number of simulated worlds.")

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args.num_worlds)

    newton.examples.run(example, args)
