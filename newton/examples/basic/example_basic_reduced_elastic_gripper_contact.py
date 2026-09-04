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
# Example Basic Reduced Elastic Gripper Contact
#
# Demonstrates two reduced elastic rubber pads squeezing and lifting a dynamic
# rigid object through frictional contact.
#
# Command: python -m newton.examples basic_reduced_elastic_gripper_contact
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
    contact_ke = 8.0e4
    contact_kd = 0.35
    contact_mu = 8.0
    part_mass = 0.06
    grip_initial_x = 0.097
    grip_closed_x = 0.080
    contact_gap = 0.002
    close_duration = 0.1
    lift_start_time = 0.35
    lift_duration = 1.0
    settle_time = 1.45
    solver_iterations = 12

    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.args = args
        self.show_elastic_strain = True

        contact_cfg = contact_shape_config(ke=self.contact_ke, kd=self.contact_kd, mu=self.contact_mu)
        contact_cfg.gap = self.contact_gap
        visual_cfg = visual_shape_config()

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis="Z")
        builder.num_rigid_contacts_per_world = 4096
        builder.add_ground_plane(cfg=contact_cfg)

        self.part_initial_x = 0.0
        self.part_initial_z = 0.28
        self.part_radius = 0.055
        part_inertia = wp.mat33(0.002, 0.0, 0.0, 0.0, 0.002, 0.0, 0.0, 0.0, 0.002)
        self.part = builder.add_body(
            xform=wp.transform(wp.vec3(self.part_initial_x, 0.0, self.part_initial_z), wp.quat_identity()),
            mass=self.part_mass,
            inertia=part_inertia,
            label="grasped_rigid_part",
        )
        builder.add_shape_sphere(self.part, radius=self.part_radius, cfg=contact_cfg)
        self.grip_hx = 0.04
        self.grip_initial_x = self.__class__.grip_initial_x
        self.left_grip_closed_x = self.grip_closed_x
        self.right_grip_closed_x = self.grip_closed_x
        self.left_grip = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(-self.grip_initial_x, 0.0, self.part_initial_z), wp.quat_identity()),
            mass=0.4,
            inertia=identity_inertia(0.006),
            mode_count=2,
            mode_mass=[0.03, 0.02],
            mode_stiffness=[110.0, 420.0],
            mode_damping=[1.1, 1.35],
            mode_shape_fn=rubber_contact_modes(self.grip_hx, contact_side=1.0, axis=0),
            is_kinematic=True,
            label="left_rubber_gripper",
        )
        self.right_grip = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(self.grip_initial_x, 0.0, self.part_initial_z), wp.quat_identity()),
            mass=0.4,
            inertia=identity_inertia(0.006),
            mode_count=2,
            mode_mass=[0.03, 0.02],
            mode_stiffness=[110.0, 420.0],
            mode_damping=[1.1, 1.35],
            mode_shape_fn=rubber_contact_modes(self.grip_hx, contact_side=-1.0, axis=0),
            is_kinematic=True,
            label="right_rubber_gripper",
        )
        for grip in (self.left_grip, self.right_grip):
            builder.add_shape_box(grip, hx=self.grip_hx, hy=0.085, hz=0.09, cfg=contact_cfg)

        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.12), wp.quat_identity()),
            hx=0.34,
            hy=0.015,
            hz=0.015,
            cfg=visual_cfg,
            label="gripper_drive_rail",
        )

        builder.color()
        self.model = builder.finalize()
        self.model.rigid_contact_max = 4096
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.solver_iterations,
            rigid_compliant_alm=True,
            rigid_contact_k_start=self.contact_ke,
            rigid_body_contact_buffer_size=64,
            friction_epsilon=3.0e-3,
            elastic_body_relaxation=0.30,
            elastic_solver_metrics=True,
        )

        self._owner_q_starts = owner_q_starts(self.model, [self.left_grip, self.right_grip])
        self._owner_qd_starts = owner_qd_starts(self.model, [self.left_grip, self.right_grip])
        self.max_left_compression = 0.0
        self.max_right_compression = 0.0
        self.max_part_z = self.part_initial_z
        self.min_part_z_after_lift = 1.0e6
        self.max_contact_count = 0
        self.max_left_contact_count = 0
        self.max_right_contact_count = 0
        self.settled_sample_count = 0
        self.settled_contact_dropouts = 0
        self.settled_part_x_min = float("inf")
        self.settled_part_x_max = float("-inf")
        self.settled_rel_z_min = float("inf")
        self.settled_rel_z_max = float("-inf")
        self.max_settled_horizontal_speed = 0.0
        self.max_settled_vertical_speed = 0.0
        self.max_lift_lag = 0.0
        self.max_lift_lead = 0.0
        self.max_horizontal_drift = 0.0
        init_elastic_solver_metric_tracking(self)

        self.viewer.set_model(self.model)
        self.viewer.show_elastic_strain = True
        self.viewer.set_camera(pos=wp.vec3(0.38, -1.35, 0.78), pitch=-22.0, yaw=90.0)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 42.0

    def _drive_targets(self, t: float):
        close = 0.5 * (1.0 - math.cos(min(t / self.close_duration, 1.0) * math.pi))
        lift = 0.0
        if t > self.lift_start_time:
            lift = 0.22 * 0.5 * (1.0 - math.cos(min((t - self.lift_start_time) / self.lift_duration, 1.0) * math.pi))

        left_grip_x = self.grip_initial_x + (self.left_grip_closed_x - self.grip_initial_x) * close
        right_grip_x = self.grip_initial_x + (self.right_grip_closed_x - self.grip_initial_x) * close
        grip_z = self.part_initial_z + lift
        return {
            self.left_grip: (wp.vec3(-left_grip_x, 0.0, grip_z), wp.quat_identity()),
            self.right_grip: (wp.vec3(right_grip_x, 0.0, grip_z), wp.quat_identity()),
        }

    def _update_metrics(self):
        update_elastic_solver_metric_tracking(self)
        q = self.state_0.joint_q.numpy()
        left_start = self._owner_q_starts[self.left_grip]
        right_start = self._owner_q_starts[self.right_grip]
        self.max_left_compression = max(self.max_left_compression, abs(float(q[left_start + 7])))
        self.max_right_compression = max(self.max_right_compression, abs(float(q[right_start + 7])))

        part_q = self.state_0.body_q.numpy()[self.part]
        part_qd = self.state_0.body_qd.numpy()[self.part]
        part_z = float(part_q[2])
        self.max_part_z = max(self.max_part_z, part_z)
        self.max_horizontal_drift = max(self.max_horizontal_drift, abs(float(part_q[0] - self.part_initial_x)))
        target_z = float(self._drive_targets(self.sim_time)[self.left_grip][0][2])
        if self.sim_time > 0.8:
            self.min_part_z_after_lift = min(self.min_part_z_after_lift, part_z)
            relative_z = target_z - part_z
            self.max_lift_lag = max(self.max_lift_lag, relative_z)
            self.max_lift_lead = max(self.max_lift_lead, -relative_z)
        contact_count = min(int(self.contacts.rigid_contact_count.numpy()[0]), int(self.model.rigid_contact_max))
        self.max_contact_count = max(self.max_contact_count, contact_count)
        left_contact_count = 0
        right_contact_count = 0
        if contact_count > 0:
            shape_body = self.model.shape_body.numpy()
            shape0 = self.contacts.rigid_contact_shape0.numpy()[:contact_count]
            shape1 = self.contacts.rigid_contact_shape1.numpy()[:contact_count]
            active = (shape0 >= 0) & (shape1 >= 0)
            bodies0 = shape_body[shape0[active]]
            bodies1 = shape_body[shape1[active]]
            left_contact_count = int(((bodies0 == self.left_grip) | (bodies1 == self.left_grip)).sum())
            right_contact_count = int(((bodies0 == self.right_grip) | (bodies1 == self.right_grip)).sum())
            self.max_left_contact_count = max(self.max_left_contact_count, left_contact_count)
            self.max_right_contact_count = max(self.max_right_contact_count, right_contact_count)
        if self.sim_time > self.settle_time:
            self.settled_sample_count += 1
            part_x = float(part_q[0] - self.part_initial_x)
            relative_z = target_z - part_z
            self.settled_part_x_min = min(self.settled_part_x_min, part_x)
            self.settled_part_x_max = max(self.settled_part_x_max, part_x)
            self.settled_rel_z_min = min(self.settled_rel_z_min, relative_z)
            self.settled_rel_z_max = max(self.settled_rel_z_max, relative_z)
            horizontal_speed = math.hypot(float(part_qd[0]), float(part_qd[1]))
            vertical_speed = abs(float(part_qd[2]))
            self.max_settled_horizontal_speed = max(self.max_settled_horizontal_speed, horizontal_speed)
            self.max_settled_vertical_speed = max(self.max_settled_vertical_speed, vertical_speed)
            if left_contact_count == 0 or right_contact_count == 0:
                self.settled_contact_dropouts += 1

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
            raise AssertionError("gripper contact example did not generate contacts")
        if self.max_left_contact_count == 0 or self.max_right_contact_count == 0:
            raise AssertionError(
                "rigid part did not contact both grippers: "
                f"left={self.max_left_contact_count}, right={self.max_right_contact_count}"
            )
        if min(self.max_left_compression, self.max_right_compression) < 0.0115:
            raise AssertionError(
                "gripper modal compression too small: "
                f"left={self.max_left_compression}, right={self.max_right_compression}"
            )
        if self.max_part_z < self.part_initial_z + 0.12:
            raise AssertionError(f"grasped part did not lift: max z={self.max_part_z}")
        if self.min_part_z_after_lift < self.part_initial_z + 0.04:
            raise AssertionError(f"grasped part slipped out: min lifted z={self.min_part_z_after_lift}")
        if self.max_lift_lag > 0.035:
            raise AssertionError(f"grasped part slipped downward relative to the pads: lag={self.max_lift_lag}")
        if self.max_lift_lead > 0.035:
            raise AssertionError(f"grasped part separated upward relative to the pads: lead={self.max_lift_lead}")
        part_q = self.state_0.body_q.numpy()[self.part]
        if self.max_horizontal_drift > 0.05:
            raise AssertionError(f"grasped part left the gripper gap: x={part_q[0]}")
        if self.settled_sample_count == 0:
            raise AssertionError("gripper contact example did not reach the settled measurement window")
        settled_x_range = self.settled_part_x_max - self.settled_part_x_min
        settled_rel_z_range = self.settled_rel_z_max - self.settled_rel_z_min
        if self.settled_contact_dropouts > 0:
            raise AssertionError(
                "gripper lost settled contact on at least one side: "
                f"dropouts={self.settled_contact_dropouts}, left={self.max_left_contact_count}, "
                f"right={self.max_right_contact_count}"
            )
        if settled_x_range > 0.004:
            raise AssertionError(f"settled grasp vibrated horizontally: range={settled_x_range}")
        if settled_rel_z_range > 0.004:
            raise AssertionError(f"settled grasp vibrated vertically relative to pads: range={settled_rel_z_range}")
        if self.max_settled_horizontal_speed > 0.02:
            raise AssertionError(f"settled grasp horizontal speed too high: speed={self.max_settled_horizontal_speed}")
        if self.max_settled_vertical_speed > 0.02:
            raise AssertionError(f"settled grasp vertical speed too high: speed={self.max_settled_vertical_speed}")
        if self.final_modal_solve_residual_ratio > 0.05:
            raise AssertionError(
                f"gripper modal solve residual ratio too high: {self.final_modal_solve_residual_ratio}"
            )
        if self.final_modal_update_norm > 4.0e-5 or self.max_modal_update_norm > 5.0e-4:
            raise AssertionError(
                "gripper modal update too large: "
                f"final={self.final_modal_update_norm}, max={self.max_modal_update_norm}"
            )
        validate_elastic_vertices(self.model, self.state_0)


def test(device=None, frame_count: int = 120):
    return run_example_test(Example, frame_count, device)


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
