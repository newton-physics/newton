# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic VBD Mimic Joint Friction
#
# Two identically driven parallel grippers show how friction on a mimic
# follower contributes to the coupled mechanism. The upper gripper has 4 N of
# friction on both sliders. The lower gripper has 4 N on its leader and 16 N
# on its follower, so it sticks longer and lags farther behind the same target.
# Each jaw also carries a passive frictional hinge downstream.
#
# Command: python -m newton.examples basic_vbd_mimic_joint_friction
#
###########################################################################

import math

import warp as wp

import newton
import newton.examples


def _add_gripper(builder, *, center_z, follower_friction, label, colors, dynamic_cfg, visual_cfg):
    """Add one frictional mimic gripper and return its leader and follower joints."""
    jaw_x = 0.65
    jaw_bodies = []
    for side, color in zip((-1.0, 1.0), colors, strict=True):
        body = builder.add_link(
            xform=wp.transform(p=(side * jaw_x, 0.0, center_z), q=wp.quat_identity()),
            label=f"{label}_{'leader' if side < 0.0 else 'follower'}_jaw",
        )
        builder.add_shape_box(
            body,
            hx=0.13,
            hy=0.3,
            hz=0.15,
            cfg=dynamic_cfg,
            color=color,
        )
        jaw_bodies.append(body)

    leader_joint = builder.add_joint_prismatic(
        parent=-1,
        child=jaw_bodies[0],
        parent_xform=wp.transform(p=(-jaw_x, 0.0, center_z), q=wp.quat_identity()),
        axis=newton.Axis.X,
        target_ke=120.0,
        target_kd=12.0,
        limit_lower=-0.02,
        limit_upper=0.38,
        limit_ke=2.0e4,
        limit_kd=200.0,
        friction=4.0,
        label=f"{label}_actuated_slider",
    )
    follower_joint = builder.add_joint_prismatic(
        parent=-1,
        child=jaw_bodies[1],
        parent_xform=wp.transform(p=(jaw_x, 0.0, center_z), q=wp.quat_identity()),
        axis=newton.Axis.X,
        limit_lower=-0.38,
        limit_upper=0.02,
        limit_ke=2.0e4,
        limit_kd=200.0,
        friction=follower_friction,
        label=f"{label}_mimic_slider",
    )

    passive_joints = []
    finger_length = 0.32
    for side, jaw_body, color in zip((-1.0, 1.0), jaw_bodies, colors, strict=True):
        finger_body = builder.add_link(
            xform=wp.transform(
                p=(side * jaw_x, 0.0, center_z - 0.15 - finger_length),
                q=wp.quat_identity(),
            ),
            label=f"{label}_passive_finger",
        )
        builder.add_shape_box(
            finger_body,
            hx=0.07,
            hy=0.22,
            hz=finger_length,
            cfg=dynamic_cfg,
            color=color,
        )
        passive_joint = builder.add_joint_revolute(
            parent=jaw_body,
            child=finger_body,
            parent_xform=wp.transform(p=(0.0, 0.0, -0.15), q=wp.quat_identity()),
            child_xform=wp.transform(p=(0.0, 0.0, finger_length), q=wp.quat_identity()),
            axis=newton.Axis.Y,
            limit_lower=-0.9,
            limit_upper=0.9,
            limit_ke=5.0e3,
            limit_kd=50.0,
            friction=0.25,
            label=f"{label}_passive_hinge",
        )
        passive_joints.append(passive_joint)

    builder.add_articulation(
        [leader_joint, follower_joint, *passive_joints],
        label=label,
    )
    builder.set_joint_mimic(follower_joint, leader_joint, coeffs=(0.0, -1.0))
    builder.joint_q[builder.joint_q_start[passive_joints[0]]] = -0.45
    builder.joint_q[builder.joint_q_start[passive_joints[1]]] = 0.45

    builder.add_shape_box(
        body=-1,
        xform=wp.transform(p=(0.0, 0.0, center_z + 0.27), q=wp.quat_identity()),
        hx=0.95,
        hy=0.34,
        hz=0.045,
        cfg=visual_cfg,
        color=(0.2, 0.22, 0.26),
        label=f"{label}_rail",
    )
    return leader_joint, follower_joint


class Example:
    FPS = 60
    SIM_SUBSTEPS = 2
    CYCLE_TIME = 5.0
    MAX_TRAVEL = 0.34

    def __init__(self, viewer, args):
        """Build matched- and different-friction VBD mimic grippers."""
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.frame_dt = 1.0 / self.FPS
        self.sim_dt = self.frame_dt / self.SIM_SUBSTEPS
        self.sim_time = 0.0
        self.max_observed_travel = 0.0
        self.max_mimic_error = 0.0
        self.max_response_separation = 0.0

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        dynamic_cfg = newton.ModelBuilder.ShapeConfig(
            density=50.0,
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

        matched = _add_gripper(
            builder,
            center_z=2.15,
            follower_friction=4.0,
            label="matched_friction_4N_4N",
            colors=((0.42, 0.78, 0.46), (0.42, 0.78, 0.46)),
            dynamic_cfg=dynamic_cfg,
            visual_cfg=visual_cfg,
        )
        different = _add_gripper(
            builder,
            center_z=0.85,
            follower_friction=16.0,
            label="different_friction_4N_16N",
            colors=((0.95, 0.48, 0.18), (0.22, 0.58, 0.95)),
            dynamic_cfg=dynamic_cfg,
            visual_cfg=visual_cfg,
        )
        self.leader_joints = (matched[0], different[0])
        self.follower_joints = (matched[1], different[1])

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
        self.leader_q_indices = tuple(int(joint_q_start[joint]) for joint in self.leader_joints)
        self.follower_q_indices = tuple(int(joint_q_start[joint]) for joint in self.follower_joints)
        self.leader_target_indices = tuple(int(joint_target_q_start[joint]) for joint in self.leader_joints)
        self.joint_q = wp.empty_like(self.model.joint_q)
        self.joint_qd = wp.empty_like(self.model.joint_qd)
        self.target_travel = 0.0
        self.matched_travel = 0.0
        self.different_travel = 0.0
        self.different_follower_travel = 0.0

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(3.5, -5.0, 2.3), pitch=-5.0, yaw=126.0)

    def step(self):
        """Drive both leaders toward the same target and compare their response."""
        next_time = self.sim_time + self.frame_dt
        phase = 2.0 * math.pi * next_time / self.CYCLE_TIME
        self.target_travel = 0.5 * self.MAX_TRAVEL * (1.0 - math.cos(phase))
        for target_index in self.leader_target_indices:
            self.control.joint_target_q[target_index : target_index + 1].fill_(self.target_travel)

        for _ in range(self.SIM_SUBSTEPS):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time = next_time
        newton.eval_ik(self.model, self.state_0, self.joint_q, self.joint_qd)
        joint_q = self.joint_q.numpy()
        leader_travel = tuple(float(joint_q[index]) for index in self.leader_q_indices)
        follower_travel = tuple(float(joint_q[index]) for index in self.follower_q_indices)
        self.matched_travel, self.different_travel = leader_travel
        self.different_follower_travel = -follower_travel[1]
        mimic_errors = tuple(follower + leader for follower, leader in zip(follower_travel, leader_travel, strict=True))
        response_separation = self.matched_travel - self.different_travel

        self.max_observed_travel = max(self.max_observed_travel, abs(self.matched_travel))
        self.max_mimic_error = max(self.max_mimic_error, *(abs(error) for error in mimic_errors))
        self.max_response_separation = max(self.max_response_separation, abs(response_separation))

        self.viewer.log_scalar("Target travel [m]", self.target_travel)
        self.viewer.log_scalar("Matched friction 4N + 4N [m]", self.matched_travel)
        self.viewer.log_scalar("Different friction 4N + 16N [m]", self.different_travel)
        self.viewer.log_scalar("Mirrored 16N follower travel [m]", self.different_follower_travel)
        self.viewer.log_scalar("Friction response separation [mm]", 1000.0 * response_separation)
        self.viewer.log_scalar("Maximum mimic error [mm]", 1000.0 * max(abs(error) for error in mimic_errors))

    def render(self):
        """Render both grippers without an object between their fingers."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        """Explain the visual comparison and report the current responses."""
        ui.text("Upper green gripper: matched friction (4 N + 4 N)")
        ui.text("Lower orange/blue gripper: higher follower friction (4 N + 16 N)")
        ui.text("Both leaders receive the same position target.")
        ui.separator()
        ui.text(f"Target: {self.target_travel:.3f} m")
        ui.text(f"Matched response: {self.matched_travel:.3f} m")
        ui.text(f"Higher-follower-friction response: {self.different_travel:.3f} m")
        ui.text("Each follower mirrors its leader; friction changes the pair response, not the mimic ratio.")

    def test_final(self):
        """Verify follower friction changes response without breaking mimic tracking."""
        if self.max_observed_travel < 0.1:
            raise ValueError("The actuated grippers did not move")
        if self.max_response_separation < 0.03:
            raise ValueError("Higher mimic-follower friction did not visibly change the driven response")
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
