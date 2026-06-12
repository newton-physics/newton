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

ROBOT_COUNT = 16
ROBOT_SPACING = 0.3
WHEEL_RADIUS = 0.05
WHEEL_BASE = 0.2


def dof_starts_per_world(
    model: newton.Model,
    template: newton.ModelBuilder,
    *joint_ids: int,
) -> list[int]:
    """Global ``joint_qd`` indices for template joint IDs across replicated worlds.

    Each world is laid out like the unfinalized *template*. Use
    ``template.joint_qd_start[joint_id]`` for the DOF offset within a world
    (not the raw joint index returned by ``add_joint_*``).
    """
    dof_world_start = model.joint_dof_world_start.numpy()
    flat: list[int] = []
    for world in range(model.world_count):
        world_dof_base = int(dof_world_start[world])
        flat += [world_dof_base + template.joint_qd_start[j] for j in joint_ids]
    return flat


def _build_robot_template():
    """One chassis fixed to the world + two revolute wheels."""
    builder = newton.ModelBuilder()

    chassis = builder.add_link(
        xform=wp.transform(p=wp.vec3(0., 0., 0.2))
    )
    builder.add_shape_box(chassis, hx=0.15, hy=0.10, hz=0.04)
    # add_shape_cylinder is fixed along Z; rotate the shape so its length
    # aligns with the wheel rotation axis (Y).
    wheel_xform = wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -wp.pi / 2))
    left = builder.add_link()
    builder.add_shape_cylinder(left, xform=wheel_xform, radius=WHEEL_RADIUS, half_height=0.01)
    right = builder.add_link()
    builder.add_shape_cylinder(right, xform=wheel_xform, radius=WHEEL_RADIUS, half_height=0.01)

    j_base = builder.add_joint_free(
        parent=-1, 
        child=chassis,
    )
    j_left = builder.add_joint_revolute(
        parent=chassis,
        child=left,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(0.0, -WHEEL_BASE / 2, -0.05)),
        label="wheel_left",
    )
    j_right = builder.add_joint_revolute(
        parent=chassis,
        child=right,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(0.0, +WHEEL_BASE / 2, -0.05)),
        label="wheel_right",
    )
    builder.add_articulation([j_base, j_left, j_right], label="diff_drive_robot")

    # MuJoCo velocity actuators on the two wheel DOFs (the fixed joint has no
    # actuator). ke = 0 means no position term; kd is the velocity gain.
    for i in range(len(builder.joint_target_kd)):
        builder.joint_target_ke[i] = 0.0
        builder.joint_target_kd[i] = 20.0
        builder.joint_target_mode[i] = int(JointTargetMode.VELOCITY)
    return builder, j_left, j_right


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

        template, j_left, j_right = _build_robot_template()
        scene = newton.ModelBuilder()
        scene.replicate(template, ROBOT_COUNT)
        scene.add_ground_plane()
        self.model = scene.finalize()

        self.solver = newton.solvers.SolverMuJoCo(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # 2 wheel DOFs per robot, laid out [r0_L, r0_R, r1_L, r1_R, ...].
        wheel_dof_idx = dof_starts_per_world(self.model, template, j_left, j_right)
        self._wheel_dof_idx = np.array(wheel_dof_idx, dtype=np.int32)
        default_dof_indices = wp.array(wheel_dof_idx, dtype=wp.uint32, device=self.device)

        self.controller = ControllerDifferentialDrive(
            num_robots=ROBOT_COUNT,
            wheel_radius=wp.full(ROBOT_COUNT, WHEEL_RADIUS, dtype=wp.float32, device=self.device),
            wheel_base=wp.full(ROBOT_COUNT, WHEEL_BASE, dtype=wp.float32, device=self.device),
            default_dof_indices=default_dof_indices,
            device=self.device,
        )

        # robots all go in a circle:
        omega = 0.05 # speed at which they will circle.
        linear_cmd = np.linspace(ROBOT_SPACING*omega, ROBOT_COUNT*ROBOT_SPACING*omega, ROBOT_COUNT, dtype=np.float32)
        angular_cmd = wp.full(shape=ROBOT_COUNT, value=omega, dtype=wp.float32)
        self._input = self.controller.input_struct()
        self._input.linear_speed_command.assign(linear_cmd)
        self._input.angular_speed_command.assign(angular_cmd)
        self._linear_cmd = linear_cmd
        self._angular_cmd = angular_cmd

        # Controller writes directly into control.joint_target_qd so MuJoCo's
        # velocity actuators read the freshest target each substep.
        self._output = self.controller.output_struct()
        self._output.joint_target_qd = self.control.joint_target_qd

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((0., ROBOT_SPACING, 0.))
        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate()
            self.graph = capture.graph

    def _simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.model.collide(
                state=self.state_0,
                contacts=self.contacts,
            )
            self.controller.compute(self._input, self._output, None, None, time_step=self.sim_dt)
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
        out = self.control.joint_target_qd.numpy()[self._wheel_dof_idx]
        np.testing.assert_allclose(out[0::2], expected_left, atol=1e-4)
        np.testing.assert_allclose(out[1::2], expected_right, atol=1e-4)


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
