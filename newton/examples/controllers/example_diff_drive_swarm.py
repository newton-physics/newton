# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Controllers — A swarm of differential-drive robots
#
# A single ControllerDifferentialDrive drives the wheels of N two-wheel
# differential-drive robots. Each robot receives a constant
# (linear_speed, angular_speed) command — most go straight at different
# speeds, the last few spin in place. MuJoCo's velocity actuators track
# the wheel-velocity targets the controller writes each substep.
#
# Command: python -m newton.examples diff_drive_swarm
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import JointTargetMode
from newton.controllers import ControllerDifferentialDrive

ROBOT_COUNT = 64
ROBOT_SPACING = 0.6
WHEEL_RADIUS = 0.05
WHEEL_BASE = 0.2


def _build_robot_template():
    """One chassis fixed to the world + two revolute wheels."""
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    chassis = builder.add_link()
    builder.add_shape_box(chassis, hx=0.15, hy=0.10, hz=0.04)
    # add_shape_cylinder is fixed along Z; rotate the shape so its length
    # aligns with the wheel rotation axis (Y).
    wheel_xform = wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -wp.pi / 2))
    left = builder.add_link()
    builder.add_shape_cylinder(left, xform=wheel_xform, radius=WHEEL_RADIUS, half_height=0.01)
    right = builder.add_link()
    builder.add_shape_cylinder(right, xform=wheel_xform, radius=WHEEL_RADIUS, half_height=0.01)

    j_base = builder.add_joint_fixed(parent=-1, child=chassis)
    j_left = builder.add_joint_revolute(
        parent=chassis,
        child=left,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(0.0, -WHEEL_BASE / 2, -0.05)),
    )
    j_right = builder.add_joint_revolute(
        parent=chassis,
        child=right,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(0.0, +WHEEL_BASE / 2, -0.05)),
    )
    builder.add_articulation([j_base, j_left, j_right], label="diff_drive_robot")

    # MuJoCo velocity actuators on the two wheel DOFs (the fixed joint has no
    # actuator). ke = 0 means no position term; kd is the velocity gain.
    for i in range(len(builder.joint_target_kd)):
        builder.joint_target_ke[i] = 0.0
        builder.joint_target_kd[i] = 20.0
        builder.joint_target_mode[i] = int(JointTargetMode.VELOCITY)
    return builder


class Example:
    def __init__(self, viewer, args):
        # joint_target_qd is the canonical coord-layout array MuJoCo's
        # velocity actuators read from.
        newton.use_coord_layout_targets = True

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.device = wp.get_device()

        template = _build_robot_template()
        scene = newton.ModelBuilder()
        scene.replicate(template, ROBOT_COUNT, spacing=(ROBOT_SPACING, 0.0, 0.0))
        self.model = scene.finalize()

        self.solver = newton.solvers.SolverMuJoCo(self.model, disable_contacts=True)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # 2 wheel DOFs per robot, laid out [r0_L, r0_R, r1_L, r1_R, ...].
        default_dof_indices = wp.array(np.arange(2 * ROBOT_COUNT, dtype=np.uint32), device=self.device)
        self.controller = ControllerDifferentialDrive(
            num_robots=ROBOT_COUNT,
            wheel_radius=wp.full(ROBOT_COUNT, WHEEL_RADIUS, dtype=wp.float32, device=self.device),
            wheel_base=wp.full(ROBOT_COUNT, WHEEL_BASE, dtype=wp.float32, device=self.device),
            default_dof_indices=default_dof_indices,
            device=self.device,
        )

        # Constant per-robot commands: forward-speed sweep across most of the
        # swarm; the last 8 robots spin in place.
        linear_cmd = np.linspace(0.1, 1.0, ROBOT_COUNT, dtype=np.float32)
        angular_cmd = np.zeros(ROBOT_COUNT, dtype=np.float32)
        linear_cmd[-8:] = 0.0
        angular_cmd[-8:] = 2.0
        self._input = self.controller.input_struct()
        self._input.linear_speed_command.assign(linear_cmd)
        self._input.angular_speed_command.assign(angular_cmd)
        self._linear_cmd = linear_cmd
        self._angular_cmd = angular_cmd

        # Controller writes directly into control.joint_target_qd so MuJoCo's
        # velocity actuators read the freshest target each substep.
        self._output = self.controller.output_struct()
        self._output.joint_target_qd = self.control.joint_target_qd

        # One compute() up front — commands are constant, so the wheel targets
        # never change. Skipping the per-substep call keeps the loop tight.
        self.controller.compute(self._input, self._output, None, None, time_step=self.sim_dt)

        self.viewer.set_model(self.model)
        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate()
            self.graph = capture.graph

    def _simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        # Verify the controller produced the analytical wheel velocities.
        # omega_L = (2v - omega*b) / (2r); omega_R = (2v + omega*b) / (2r).
        expected_left = (2.0 * self._linear_cmd - self._angular_cmd * WHEEL_BASE) / (2.0 * WHEEL_RADIUS)
        expected_right = (2.0 * self._linear_cmd + self._angular_cmd * WHEEL_BASE) / (2.0 * WHEEL_RADIUS)
        out = self.control.joint_target_qd.numpy()
        np.testing.assert_allclose(out[0::2], expected_left, atol=1e-4)
        np.testing.assert_allclose(out[1::2], expected_right, atol=1e-4)


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
