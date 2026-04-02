# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Contact Sensor
#
# Shows how to use the SensorContact class to evaluate total contact
# forces and per-counterpart force breakdowns.
# The flap has a contact sensor registering the total contact force of
# the objects on top. The plates' sensors register per-counterpart forces
# for the cube and the ball to detect which object touched which plate. Each
# plate will light up when touched by the matching object.
#
#
# Command: python -m newton.examples sensor_contact
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import Contacts
from newton.sensors import SensorContact
from newton.tests.unittest_utils import find_nonfinite_members


class Example:
    def __init__(self, viewer, args):
        # setup simulation parameters first
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_dt = self.frame_dt
        self.reset_interval = 8.0

        self.viewer = viewer
        self.plot_window = ViewerPlot(
            viewer, "Flap Contact Force", n_points=100, avg=10, scale_min=0, graph_size=(400, 200)
        )
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.register_ui_callback(self.plot_window.render, "free")

        builder = newton.ModelBuilder()
        builder.add_usd(newton.examples.get_asset("sensor_contact_scene.usda"))
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        builder.add_ground_plane()

        # finalize model
        self.model = builder.finalize()

        self.flap_contact_sensor = SensorContact(self.model, sensing_obj_shapes="*Flap", verbose=True)

        # String patterns return matches in ascending shape index order.
        # Plate1 has a lower index than Plate2 (added first), so row 0 → Plate1, row 1 → Plate2.
        plate_labels = ["*Plate1", "*Plate2"]
        counterpart_labels = ["*Cube*", "*Sphere*"]
        self.plate_contact_sensor = SensorContact(
            self.model,
            sensing_obj_shapes=plate_labels,
            counterpart_shapes=counterpart_labels,
            measure_total=False,
            verbose=True,
        )
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            njmax=100,
            nconmax=100,
            cone="pyramidal",
            impratio=1,
        )

        # used for storing contact info required by contact sensor
        self.contacts = Contacts(
            self.solver.get_max_contact_count(),
            0,
            requested_attributes=self.model.get_requested_contact_attributes(),
        )

        self.viewer.set_model(self.model)

        self.shape_map = {key: s for s, key in enumerate(self.model.shape_label)}
        self.plates_touched = 2 * [False]
        # Each plate watches one counterpart — Plate1 watches Cube, Plate2 watches Sphere.
        # Look up the counterpart column for each plate's target.
        cube_shape = self.shape_map["/env/Cube"]
        sphere_shape = self.shape_map["/env/Sphere"]
        self.counterpart_col = [
            self.plate_contact_sensor.counterpart_indices[0].index(cube_shape),
            self.plate_contact_sensor.counterpart_indices[1].index(sphere_shape),
        ]
        self.shape_colors = {
            "/env/Plate1": 3 * [0.4],
            "/env/Plate2": 3 * [0.4],
            "/env/Sphere": [1.0, 0.4, 0.2],
            "/env/Cube": [0.2, 0.4, 0.8],
            "/env/Flap": 3 * [0.8],
        }

        self.state_0 = self.model.state()

        self.control = self.model.control()
        hinge_joint_idx = self.model.joint_label.index("/env/Hinge")
        self.hinge_joint_q_start = int(self.model.joint_q_start.numpy()[hinge_joint_idx])

        self.next_reset = 0.0

        # store initial state for reset
        self.initial_joint_q = wp.clone(self.state_0.joint_q)
        self.initial_joint_qd = wp.clone(self.state_0.joint_qd)

        self.capture()

    def _set_shape_colors(self, shape_colors: dict[int, list[float] | tuple[float, float, float]]):
        for shape_idx, color in shape_colors.items():
            self.model.shape_color[shape_idx : shape_idx + 1].fill_(wp.vec3(color))

    def capture(self):
        self.graph = None

        if not wp.get_device().is_cuda:
            return

        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_0, self.control, None, self.sim_dt)
        self.solver.update_contacts(self.contacts, self.state_0)

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
        self.plate_contact_sensor.update(self.state_0, self.contacts)

        # Check if any object touched the matching plate by looking up per-counterpart forces.
        net_force = self.plate_contact_sensor.force_matrix.numpy()
        for i in range(2):
            if self.plates_touched[i]:
                continue
            if np.abs(net_force[i, self.counterpart_col[i]]).max() == 0:
                continue
            plate_shape = self.plate_contact_sensor.sensing_obj_idx[i]
            counterpart_shape = self.plate_contact_sensor.counterpart_indices[i][self.counterpart_col[i]]
            self.plates_touched[i] = True
            plate_label = self.model.shape_label[plate_shape]
            counterpart_label = self.model.shape_label[counterpart_shape]
            print(f"Plate {plate_label} was touched by counterpart {counterpart_label}")
            self._set_shape_colors({plate_shape: self.shape_colors[counterpart_label]})

        self.flap_contact_sensor.update(self.state_0, self.contacts)
        self.plot_window.add_point(np.abs(self.flap_contact_sensor.total_force.numpy()[0, 2]))
        self.sim_time += self.frame_dt

    def reset(self):
        self.sim_time = 0
        self.next_reset = self.sim_time + self.reset_interval
        self._set_shape_colors({self.shape_map[s]: v for s, v in self.shape_colors.items()})
        self.plates_touched = 2 * [False]
        self.plot_window.reset()

        print("Resetting")
        # Restore initial joint positions and velocities in-place.
        self.state_0.joint_q.assign(self.initial_joint_q)
        self.state_0.joint_qd.assign(self.initial_joint_qd)
        # Recompute forward kinematics to refresh derived state.
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        assert not self.plates_touched[1] or self.plates_touched[0]  # plate 0 always touched first
        assert len(find_nonfinite_members(self.flap_contact_sensor)) == 0
        assert len(find_nonfinite_members(self.plate_contact_sensor)) == 0
        # first plate touched by 1.4s, second by 4s, flap left by 2.8s
        if self.sim_time > 1.4:
            assert self.plates_touched[0]
        if self.sim_time > 2.8:
            assert self.flap_contact_sensor.total_force.numpy().sum() == 0
        # if self.sim_time > 4.0: assert self.plates_touched[1]   # unreliable due to jerky cube motion

    def test_final(self):
        self.test_post_step()
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.0,
        )
        assert len(find_nonfinite_members(self.flap_contact_sensor)) == 0
        assert len(find_nonfinite_members(self.plate_contact_sensor)) == 0
        # sensing_obj_idx preserves the input order given to the sensor.
        assert self.model.shape_label[self.plate_contact_sensor.sensing_obj_idx[0]] == "/env/Plate1"
        assert self.model.shape_label[self.plate_contact_sensor.sensing_obj_idx[1]] == "/env/Plate2"


class ViewerPlot:
    """ImGui plot window"""

    def __init__(self, viewer=None, title="Plot", n_points=200, avg=1, **kwargs):
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

    def reset(self):
        self.data.fill(0)
        self.cache.clear()

    def render(self, imgui):
        """Render the force plot window.

        Args:
            imgui: The ImGui object passed by the ViewerGL callback system.
        """
        if not self.viewer or not self.viewer.ui.is_available:
            return

        io = self.viewer.ui.io

        # Position the plot window
        window_shape = (400, 350)
        imgui.set_next_window_pos(
            imgui.ImVec2(io.display_size[0] - window_shape[0] - 10, io.display_size[1] - window_shape[1] - 10)
        )
        imgui.set_next_window_size(imgui.ImVec2(*window_shape))

        flags = imgui.WindowFlags_.no_resize.value

        if imgui.begin(self.title, flags=flags):
            imgui.text("Flap contact force")
            avail = imgui.get_content_region_avail()
            plot_kwargs = dict(self.plot_kwargs)
            plot_kwargs["graph_size"] = (avail.x, plot_kwargs.get("graph_size", (0, 0))[1])
            imgui.plot_lines("##force", self.data, **plot_kwargs)
        imgui.end()


if __name__ == "__main__":
    parser = newton.examples.create_parser()

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
