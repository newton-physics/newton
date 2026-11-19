# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example IK Delta Robot (closed loops + hard joint limits)

Inverse kinematics of a grid of delta robots, parallel mechanisms with closed kinematic loops,
solved as one batch: one IK problem per robot.
- Three motor arms hang from a base plate; a rod connects each arm to a moving platform.
- The articulation tree reaches the platform through the first rod; the ball joints of the
  other two rods are loop-closing joints, added outside the articulation.
- The motor joints have limits.
- Each platform tracks a target that circles below its base, partly beyond the workspace; the
  targets are staggered in time, so at any moment some robots reach beyond their workspace.
- ``IKOptimizer.CUNLS`` keeps the loops closed (equality constraints) and the motors inside
  their limits (hard bounds); ``--optimizer lm`` uses Newton's default optimizer, which ignores
  loop-closing joints.

Success criteria:
- A grid of 128 delta robots hangs below base plates; each platform follows its red target.
- All rods stay attached to their platforms.
- Where a target leaves the workspace, the arms stop at their limits and the platform stays
  as close to the target as the mechanism allows.
- With ``--optimizer lm``, the second and third rods of every robot come apart from the platform.

Requires cuNLS (``pip install newton[cunls-cu12]`` or ``newton[cunls-cu13]``).

Runs:
- `python -m newton.examples ik_delta_robot`
- `python -m newton.examples ik_delta_robot --world-count 1`
- `python -m newton.examples ik_delta_robot --optimizer lm`
"""

import math

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik

# Geometry [m]
BASE_HEIGHT = 1.0
BASE_RADIUS = 0.2
ARM_LENGTH = 0.25
ROD_LENGTH = 0.55
PLATFORM_RADIUS = 0.06

ROBOT_SPACING = 1.2

# Motor limits [rad]: arms swing from 40 degrees up to 80 degrees down.
MOTOR_LIMIT_LOWER = -0.7
MOTOR_LIMIT_UPPER = 1.4

# Target path: a circle whose radius swells past the workspace boundary (about 0.33 m) and back.
TARGET_HEIGHT = 0.45
TARGET_ANGULAR_SPEED = 0.8  # [rad/s]
TARGET_SWELL_RATE = 0.6  # [rad/s]


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.world_count = args.world_count

        self.viewer = viewer
        self.viewer.picking_enabled = False

        # One robot is the IK model; the scene holds a copy of it in each world.
        robot = newton.ModelBuilder()
        self.build_delta_robot(robot)
        self.robot_model = robot.finalize()

        scene = newton.ModelBuilder()
        scene.replicate(robot, self.world_count)
        scene.add_ground_plane()
        self.model = scene.finalize()
        self.state = self.model.state()

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((ROBOT_SPACING, ROBOT_SPACING, 0.0))
        self.world_offsets = self.viewer.world_offsets.numpy()  # to draw each target next to its robot
        distance = 1.3 + 0.75 * ROBOT_SPACING * math.ceil(math.sqrt(self.world_count))
        self.viewer.set_camera(pos=wp.vec3(distance, -distance, 0.6 + 0.7 * distance), pitch=-28.0, yaw=135.0)

        # ------------------------------------------------------------------
        # IK: platform position and orientation, motor limits; one problem per robot
        # ------------------------------------------------------------------
        # Stagger the robots over one swell period of the target path.
        self.time_offsets = 2.0 * math.pi / TARGET_SWELL_RATE * np.arange(self.world_count) / self.world_count
        self.targets = np.zeros((self.world_count, 3), dtype=np.float32)
        self.target_colors = wp.full(self.world_count, wp.vec3(1.0, 0.1, 0.1), dtype=wp.vec3)
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.platform,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.zeros(self.world_count, dtype=wp.vec3),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.platform,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(0.0, 0.0, 0.0, 1.0)] * self.world_count, dtype=wp.vec4),  # level
        )
        self.limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.robot_model.joint_limit_lower,
            joint_limit_upper=self.robot_model.joint_limit_upper,
        )

        # Row i of joint_q holds the coordinates of world i.
        self.joint_q = self.model.joint_q.reshape((self.world_count, self.robot_model.joint_coord_count))
        self.ik_iters = 30
        self.solver = ik.IKSolver(
            model=self.robot_model,
            n_problems=self.world_count,
            objectives=[self.pos_obj, self.rot_obj, self.limit_obj],
            optimizer=args.optimizer,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        self.tracking_errors = []

    def build_delta_robot(self, builder):
        """Add the delta robot: one articulation tree plus two loop-closing ball joints."""
        motor_color, rod_color = (0.85, 0.45, 0.1), (0.7, 0.7, 0.75)
        builder.add_shape_cylinder(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, BASE_HEIGHT + 0.02), wp.quat_identity()),
            radius=BASE_RADIUS + 0.06,
            half_height=0.02,
            color=(0.3, 0.3, 0.35),
        )
        self.platform = builder.add_link(label="platform")
        builder.add_shape_cylinder(self.platform, radius=PLATFORM_RADIUS + 0.03, half_height=0.015, color=motor_color)

        tree_joints, rods, self.motor_joints = [], [], []
        for i in range(3):
            angle = 2.0 * math.pi * i / 3.0
            radial = wp.vec3(math.cos(angle), math.sin(angle), 0.0)
            tangent = wp.vec3(-math.sin(angle), math.cos(angle), 0.0)

            arm = builder.add_link(label=f"arm_{i}")
            builder.add_shape_capsule(
                arm,
                xform=wp.transform(
                    radial * (0.5 * ARM_LENGTH), wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), radial)
                ),
                radius=0.02,
                half_height=0.5 * ARM_LENGTH,
                color=motor_color,
            )
            rod = builder.add_link(label=f"rod_{i}")
            builder.add_shape_capsule(
                rod,
                xform=wp.transform(wp.vec3(0.0, 0.0, -0.5 * ROD_LENGTH), wp.quat_identity()),
                radius=0.008,
                half_height=0.5 * ROD_LENGTH,
                color=rod_color,
            )
            rods.append(rod)

            # Motor: the arm swings about the base tangent; positive angles point it down.
            motor = builder.add_joint_revolute(
                parent=-1,
                child=arm,
                parent_xform=wp.transform(radial * BASE_RADIUS + wp.vec3(0.0, 0.0, BASE_HEIGHT), wp.quat_identity()),
                axis=tangent,
                limit_lower=MOTOR_LIMIT_LOWER,
                limit_upper=MOTOR_LIMIT_UPPER,
                label=f"motor_{i}",
            )
            elbow = builder.add_joint_ball(
                parent=arm,
                child=rod,
                parent_xform=wp.transform(radial * ARM_LENGTH, wp.quat_identity()),
                label=f"elbow_{i}",
            )
            self.motor_joints.append(motor)
            tree_joints += [motor, elbow]

        # The tree reaches the platform through the first rod ...
        tree_joints.append(self._add_wrist(builder, rods[0], 0))
        builder.add_articulation(tree_joints, label="delta_robot")
        # ... and the wrists of the other two rods close the loops.
        for i in (1, 2):
            self._add_wrist(builder, rods[i], i)

    def _add_wrist(self, builder, rod, i):
        angle = 2.0 * math.pi * i / 3.0
        anchor = wp.vec3(math.cos(angle), math.sin(angle), 0.0) * PLATFORM_RADIUS
        return builder.add_joint_ball(
            parent=rod,
            child=self.platform,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, -ROD_LENGTH), wp.quat_identity()),
            child_xform=wp.transform(anchor, wp.quat_identity()),
            label=f"wrist_{i}",
        )

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def update_targets(self):
        """Move each target along its circle below the robot's base."""
        t = self.sim_time + self.time_offsets
        phase = TARGET_ANGULAR_SPEED * t
        radius = 0.24 - 0.14 * np.cos(TARGET_SWELL_RATE * t)
        self.targets[:, 0] = radius * np.cos(phase)
        self.targets[:, 1] = radius * np.sin(phase)
        self.targets[:, 2] = TARGET_HEIGHT
        self.pos_obj.set_target_positions(wp.array(self.targets, dtype=wp.vec3))

    def loop_gaps(self):
        """Distance [m] between the two frames of each loop-closing joint, in all worlds."""
        body_q = self.state.body_q.numpy()
        parent, child = self.model.joint_parent.numpy(), self.model.joint_child.numpy()
        X_p, X_c = self.model.joint_X_p.numpy(), self.model.joint_X_c.numpy()
        gaps = []
        for j in np.flatnonzero(self.model.joint_articulation.numpy() == -1):
            p = wp.transform_point(wp.transform(*body_q[parent[j]]), wp.vec3(*X_p[j][:3]))
            c = wp.transform_point(wp.transform(*body_q[child[j]]), wp.vec3(*X_c[j][:3]))
            gaps.append(float(wp.length(p - c)))
        return gaps

    # ----------------------------------------------------------------------
    # Template API
    # ----------------------------------------------------------------------
    def step(self):
        self.update_targets()
        self.solver.step(self.joint_q, self.joint_q, iterations=self.ik_iters)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.sim_time += self.frame_dt

    def test_post_step(self):
        gaps = self.loop_gaps()
        if max(gaps) > 1e-3:
            raise ValueError(f"A loop-closing joint opened by {max(gaps):.4f} m")
        motor_angles = self.joint_q.numpy()[:, self.robot_model.joint_q_start.numpy()[self.motor_joints]]
        if not np.all((motor_angles >= MOTOR_LIMIT_LOWER - 1e-5) & (motor_angles <= MOTOR_LIMIT_UPPER + 1e-5)):
            raise ValueError(
                f"A motor joint left its limits: angles span [{motor_angles.min():.4f}, {motor_angles.max():.4f}] rad"
            )
        platforms = self.state.body_q.numpy()[self.platform :: self.robot_model.body_count, :3]
        self.tracking_errors.extend(np.linalg.norm(platforms - self.targets, axis=1))

    def test_final(self):
        reached = np.mean(np.array(self.tracking_errors) < 1e-3)
        if reached < 0.5:
            raise ValueError(
                f"The platforms reached their targets in {reached:.0%} of the frames, expected 50% or more"
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.log_points(
            "targets",
            wp.array(self.targets + self.world_offsets, dtype=wp.vec3),
            radii=0.02,
            colors=self.target_colors,
        )
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=128)
        parser.add_argument(
            "--optimizer",
            type=str,
            default="cunls",
            choices=[o.value for o in ik.IKOptimizer],
            help="IK optimizer backend.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
