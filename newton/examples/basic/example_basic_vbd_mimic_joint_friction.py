# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic VBD Mimic Joint Friction
#
# A driven prismatic joint closes one gripper jaw. The opposite jaw mimics it
# with a negative ratio. Both slider joints use Coulomb friction, and each jaw
# carries a passive frictional hinge downstream.
#
# Command: python -m newton.examples basic_vbd_mimic_joint_friction
#
###########################################################################

import math

import warp as wp

import newton
import newton.examples


class Example:
    FPS = 60
    SIM_SUBSTEPS = 2
    CYCLE_TIME = 5.0
    MAX_TRAVEL = 0.34

    def __init__(self, viewer, args):
        """Build a VBD parallel gripper with frictional mimic and passive joints."""
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.frame_dt = 1.0 / self.FPS
        self.sim_dt = self.frame_dt / self.SIM_SUBSTEPS
        self.sim_time = 0.0
        self.max_observed_travel = 0.0
        self.max_mimic_error = 0.0

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        dynamic_cfg = newton.ModelBuilder.ShapeConfig(
            density=350.0,
            collision_group=0,
            has_shape_collision=False,
            has_particle_collision=False,
        )
        visual_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            collision_group=0,
            has_shape_collision=False,
            has_particle_collision=False,
        )

        jaw_x = 0.65
        jaw_z = 1.45
        jaw_bodies = []
        for side, color in ((-1.0, (0.95, 0.48, 0.18)), (1.0, (0.22, 0.58, 0.95))):
            body = builder.add_link(
                xform=wp.transform(p=(side * jaw_x, 0.0, jaw_z), q=wp.quat_identity()),
                label="leader_jaw" if side < 0.0 else "mimic_jaw",
            )
            builder.add_shape_box(
                body,
                hx=0.13,
                hy=0.32,
                hz=0.16,
                cfg=dynamic_cfg,
                color=color,
            )
            jaw_bodies.append(body)

        self.leader_joint = builder.add_joint_prismatic(
            parent=-1,
            child=jaw_bodies[0],
            parent_xform=wp.transform(p=(-jaw_x, 0.0, jaw_z), q=wp.quat_identity()),
            axis=newton.Axis.X,
            target_ke=4.0e3,
            target_kd=120.0,
            limit_lower=-0.02,
            limit_upper=0.38,
            limit_ke=2.0e4,
            limit_kd=200.0,
            friction=8.0,
            label="actuated_slider",
        )
        self.follower_joint = builder.add_joint_prismatic(
            parent=-1,
            child=jaw_bodies[1],
            parent_xform=wp.transform(p=(jaw_x, 0.0, jaw_z), q=wp.quat_identity()),
            axis=newton.Axis.X,
            limit_lower=-0.38,
            limit_upper=0.02,
            limit_ke=2.0e4,
            limit_kd=200.0,
            friction=12.0,
            label="mimic_slider",
        )

        passive_joints = []
        finger_length = 0.42
        for side, jaw_body, color in (
            (-1.0, jaw_bodies[0], (0.98, 0.72, 0.28)),
            (1.0, jaw_bodies[1], (0.36, 0.78, 1.0)),
        ):
            finger_body = builder.add_link(
                xform=wp.transform(
                    p=(side * jaw_x, 0.0, jaw_z - 0.16 - finger_length),
                    q=wp.quat_identity(),
                ),
                label="passive_finger",
            )
            builder.add_shape_box(
                finger_body,
                hx=0.075,
                hy=0.24,
                hz=finger_length,
                cfg=dynamic_cfg,
                color=color,
            )
            passive_joint = builder.add_joint_revolute(
                parent=jaw_body,
                child=finger_body,
                parent_xform=wp.transform(p=(0.0, 0.0, -0.16), q=wp.quat_identity()),
                child_xform=wp.transform(p=(0.0, 0.0, finger_length), q=wp.quat_identity()),
                axis=newton.Axis.Y,
                limit_lower=-0.9,
                limit_upper=0.9,
                limit_ke=5.0e3,
                limit_kd=50.0,
                friction=0.8,
                label="passive_finger_hinge",
            )
            passive_joints.append(passive_joint)

        builder.add_articulation(
            [self.leader_joint, self.follower_joint, *passive_joints],
            label="frictional_parallel_gripper",
        )
        builder.set_joint_mimic(self.follower_joint, self.leader_joint, coeffs=(0.0, -1.0))

        builder.joint_q[builder.joint_q_start[passive_joints[0]]] = -0.55
        builder.joint_q[builder.joint_q_start[passive_joints[1]]] = 0.55

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(p=(0.0, 0.0, 1.72), q=wp.quat_identity()),
            hx=0.95,
            hy=0.38,
            hz=0.06,
            cfg=visual_cfg,
            color=(0.2, 0.22, 0.26),
            label="gripper_rail",
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(p=(0.0, 0.0, 0.42), q=wp.quat_identity()),
            hx=0.06,
            hy=0.38,
            hz=0.65,
            cfg=visual_cfg,
            color=(0.2, 0.22, 0.26),
            label="gripper_mount",
        )

        builder.color()
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=6,
            rigid_compliant_alm=True,
            rigid_joint_linear_ke=1.0e6,
            rigid_joint_angular_ke=1.0e6,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        joint_q_start = self.model.joint_q_start.numpy()
        joint_target_q_start = self.model.joint_target_q_start.numpy()
        self.leader_q_index = int(joint_q_start[self.leader_joint])
        self.follower_q_index = int(joint_q_start[self.follower_joint])
        self.leader_target_index = int(joint_target_q_start[self.leader_joint])
        self.joint_q = wp.empty_like(self.model.joint_q)
        self.joint_qd = wp.empty_like(self.model.joint_qd)
        self.target_travel = 0.0
        self.leader_travel = 0.0
        self.follower_travel = 0.0

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(3.2, -4.2, 2.5), pitch=-9.0, yaw=128.0)

    def step(self):
        """Drive the leader while VBD enforces the frictional mimic pair."""
        next_time = self.sim_time + self.frame_dt
        phase = 2.0 * math.pi * next_time / self.CYCLE_TIME
        self.target_travel = 0.5 * self.MAX_TRAVEL * (1.0 - math.cos(phase))
        self.control.joint_target_q[self.leader_target_index : self.leader_target_index + 1].fill_(
            self.target_travel
        )

        for _ in range(self.SIM_SUBSTEPS):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time = next_time
        newton.eval_ik(self.model, self.state_0, self.joint_q, self.joint_qd)
        joint_q = self.joint_q.numpy()
        self.leader_travel = float(joint_q[self.leader_q_index])
        self.follower_travel = float(joint_q[self.follower_q_index])
        mimic_error = self.follower_travel + self.leader_travel
        self.max_observed_travel = max(self.max_observed_travel, abs(self.leader_travel))
        self.max_mimic_error = max(self.max_mimic_error, abs(mimic_error))

        self.viewer.log_scalar("Target travel [m]", self.target_travel)
        self.viewer.log_scalar("Leader travel [m]", self.leader_travel)
        self.viewer.log_scalar("Follower travel [m]", self.follower_travel)
        self.viewer.log_scalar("Mimic error [mm]", 1000.0 * mimic_error)

    def render(self):
        """Render the gripper."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify the driven joint moves and its frictional follower tracks it."""
        if self.max_observed_travel < 0.1:
            raise ValueError("The actuated gripper jaw did not move")
        if self.max_mimic_error > 0.01:
            raise ValueError(f"Mimic error exceeded 10 mm: {1000.0 * self.max_mimic_error:.3f} mm")

    @staticmethod
    def create_parser():
        """Create the example command-line parser."""
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=300)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
