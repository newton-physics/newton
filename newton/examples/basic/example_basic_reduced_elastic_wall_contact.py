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
# Example Basic Reduced Elastic Wall Contact
#
# Demonstrates a kinematic reduced elastic rubber pad pressed into a rigid wall.
#
# Command: python -m newton.examples basic_reduced_elastic_wall_contact
#
###########################################################################

import math

import warp as wp

import newton
import newton.examples
from newton.examples.basic._reduced_elastic import (
    init_elastic_solver_metric_tracking,
    update_elastic_solver_metric_tracking,
)
from newton.examples.basic._reduced_elastic_contact import (
    apply_kinematic_targets,
    contact_shape_config,
    finite_difference_target_velocities,
    identity_inertia,
    owner_q_starts,
    owner_qd_starts,
    rubber_contact_modes,
    run_example_test,
    validate_elastic_vertices,
    visual_shape_config,
)


class Example:
    solver_iterations = 24

    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.args = args
        self.show_elastic_strain = True

        contact_cfg = contact_shape_config()
        contact_cfg.gap = 0.001
        visual_cfg = visual_shape_config()

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis="Z")
        builder.num_rigid_contacts_per_world = 2048

        self.pad_hx = 0.14
        self.wall_x = 0.36
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(self.wall_x, 0.0, 0.28), wp.quat_identity()),
            hx=0.025,
            hy=0.32,
            hz=0.28,
            cfg=contact_cfg,
            label="rigid_wall",
        )
        self.wall_pad = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(0.08, 0.0, 0.26), wp.quat_identity()),
            mass=0.8,
            inertia=identity_inertia(),
            mode_count=2,
            mode_mass=[0.05, 0.035],
            mode_stiffness=[220.0, 900.0],
            mode_damping=[0.55, 0.8],
            mode_shape_fn=rubber_contact_modes(self.pad_hx, contact_side=1.0),
            is_kinematic=True,
            label="rubber_wall_pad",
        )
        builder.add_shape_box(self.wall_pad, hx=self.pad_hx, hy=0.12, hz=0.10, cfg=contact_cfg)
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.18, 0.0, 0.035), wp.quat_identity()),
            hx=0.42,
            hy=0.015,
            hz=0.015,
            cfg=visual_cfg,
            label="wall_pad_drive_rail",
        )

        builder.color()
        self.model = builder.finalize()
        self.model.rigid_contact_max = 2048
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.solver_iterations,
            rigid_compliant_alm=True,
            rigid_contact_k_start=8.0e4,
            rigid_body_contact_buffer_size=64,
            friction_epsilon=2.0e-3,
            elastic_body_relaxation=0.30,
            elastic_solver_metrics=True,
        )

        self._owner_q_starts = owner_q_starts(self.model, [self.wall_pad])
        self._owner_qd_starts = owner_qd_starts(self.model, [self.wall_pad])
        self.max_compression = 0.0
        self.max_contact_count = 0
        self.settled_contact_dropouts = 0
        self.settled_modal_step_max = 0.0
        self.settled_modal_accel_max = 0.0
        self._settled_prev_q = None
        self._settled_prev_prev_q = None
        init_elastic_solver_metric_tracking(self)

        self.viewer.set_model(self.model)
        self.viewer.show_elastic_strain = True
        self.viewer.set_camera(pos=wp.vec3(0.55, -1.15, 0.62), pitch=-21.0, yaw=88.0)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 42.0

    def _drive_targets(self, t: float):
        press = 0.5 * (1.0 - math.cos(min(t / 0.75, 1.0) * math.pi))
        wall_x = 0.08 + 0.18 * press
        return {self.wall_pad: (wp.vec3(wall_x, 0.0, 0.26), wp.quat_identity())}

    def _update_metrics(self):
        update_elastic_solver_metric_tracking(self)
        q = self.state_0.joint_q.numpy()
        start = self._owner_q_starts[self.wall_pad]
        modal_q = q[start + 7 : start + 9].copy()
        self.max_compression = max(self.max_compression, abs(float(modal_q[0])))
        contact_count = int(self.contacts.rigid_contact_count.numpy()[0])
        self.max_contact_count = max(self.max_contact_count, contact_count)

        if self.sim_time > 1.0:
            if contact_count == 0:
                self.settled_contact_dropouts += 1
            if self._settled_prev_q is not None:
                step = modal_q - self._settled_prev_q
                self.settled_modal_step_max = max(self.settled_modal_step_max, float(max(abs(step))))
            if self._settled_prev_q is not None and self._settled_prev_prev_q is not None:
                accel = modal_q - 2.0 * self._settled_prev_q + self._settled_prev_prev_q
                self.settled_modal_accel_max = max(self.settled_modal_accel_max, float(max(abs(accel))))
            self._settled_prev_prev_q = self._settled_prev_q
            self._settled_prev_q = modal_q

    def simulate(self):
        for substep in range(self.sim_substeps):
            t = self.sim_time + substep * self.sim_dt
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            targets = self._drive_targets(t)
            previous_targets = self._drive_targets(max(t - self.sim_dt, 0.0))
            velocities = finite_difference_target_velocities(targets, previous_targets, self.sim_dt)
            apply_kinematic_targets(self.state_0, self._owner_q_starts, targets, velocities, self._owner_qd_starts)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self._update_metrics()

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if self.max_contact_count == 0:
            raise AssertionError("wall contact example did not generate contacts")
        if self.max_compression < 0.01:
            raise AssertionError(f"wall pad compression too small: {self.max_compression}")
        if self.settled_contact_dropouts != 0:
            raise AssertionError(f"wall pad contact dropped out after settling: {self.settled_contact_dropouts}")
        if self.settled_modal_step_max > 3.0e-5 or self.settled_modal_accel_max > 3.0e-5:
            raise AssertionError(
                "wall pad settled contact jitter too high: "
                f"step={self.settled_modal_step_max}, accel={self.settled_modal_accel_max}"
            )
        if self.final_modal_solve_residual_ratio > 0.2:
            raise AssertionError(
                f"wall pad modal solve residual ratio too high: {self.final_modal_solve_residual_ratio}"
            )
        if self.final_modal_update_norm > 2.0e-4 or self.max_modal_update_norm > 3.0e-3:
            raise AssertionError(
                "wall pad modal update too large: "
                f"final={self.final_modal_update_norm}, max={self.max_modal_update_norm}"
            )
        validate_elastic_vertices(self.model, self.state_0)


def test(device=None, frame_count: int = 90):
    return run_example_test(Example, frame_count, device)


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
