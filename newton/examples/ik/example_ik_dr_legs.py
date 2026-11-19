# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example IK DR Legs (closed-loop biped legs)

Whole-body inverse kinematics of DR Legs, a biped whose legs are serial-parallel linkages with
three closed kinematic loops each, solved for a grid of robots in one batch.
- The asset's articulation tree reaches every body; six revolute joints close the loops.
- The feet stay planted while the pelvis squats, sways and twists; the robots are staggered in
  time.
- ``IKOptimizer.CUNLS`` keeps the loops closed (equality constraints) and the servos inside
  their limits (hard bounds); ``--optimizer lm`` uses Newton's default optimizer, which ignores
  loop-closing joints.

Success criteria:
- A grid of bipeds keeps its feet on the ground while each pelvis squats, sways and twists.
- All leg linkages stay connected.
- With ``--optimizer lm``, the leg linkages come apart.

Requires cuNLS (``pip install newton[cunls-cu12]`` or ``newton[cunls-cu13]``).

Runs:
- `python -m newton.examples ik_dr_legs`
- `python -m newton.examples ik_dr_legs --world-count 1`
- `python -m newton.examples ik_dr_legs --optimizer lm`
"""

import math

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik

ROBOT_SPACING = 0.5  # [m]
FOOT_SOLE = 0.018  # [m], depth of the sole below the foot frame

# Pelvis motion around a crouch, relative to the pose the asset is loaded in (legs almost straight).
CROUCH = 0.04  # [m]
SQUAT_AMPLITUDE = 0.015  # [m]
SWAY_AMPLITUDE = 0.02  # [m], sideways
SURGE_AMPLITUDE = 0.015  # [m], forward and back
TWIST_AMPLITUDE = 0.2  # [rad], about the vertical axis
MOTION_RATE = 1.5  # [rad/s]


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
        # Pinned to a revision whose USD defines the articulation and marks the loop-closing joints.
        asset_path = newton.utils.download_asset("disneyresearch", ref="ac1f172b6f53915f9b81aa9ccd765ae792cc264c")
        robot.add_usd(str(asset_path / "dr_legs" / "usd" / "dr_legs.usda"), enable_self_collisions=False)
        self.robot_model = robot.finalize()
        body_names = [label.split("/")[-1] for label in self.robot_model.body_label]
        self.pelvis = body_names.index("pelvis")
        self.feet = [body_names.index("foot_l"), body_names.index("foot_r")]

        # The loaded pose is closed and stands on the feet: it defines the targets.
        state = self.robot_model.state()
        newton.eval_fk(self.robot_model, self.robot_model.joint_q, self.robot_model.joint_qd, state)
        home = state.body_q.numpy()
        self.pelvis_home = home[self.pelvis]
        foot_poses = home[self.feet]

        scene = newton.ModelBuilder()
        scene.replicate(robot, self.world_count)
        scene.add_ground_plane(height=float(foot_poses[:, 2].min()) - FOOT_SOLE)
        self.model = scene.finalize()
        self.state = self.model.state()

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((ROBOT_SPACING, ROBOT_SPACING, 0.0))
        distance = 0.3 + 0.5 * ROBOT_SPACING * math.ceil(math.sqrt(self.world_count))
        self.viewer.set_camera(pos=wp.vec3(distance, -distance, 0.05 + 0.5 * distance), pitch=-22.0, yaw=135.0)

        # ------------------------------------------------------------------
        # IK: pelvis pose, planted feet, servo limits; one problem per robot
        # ------------------------------------------------------------------
        # Stagger the robots over one period of the motion.
        self.time_offsets = 2.0 * math.pi / MOTION_RATE * np.arange(self.world_count) / self.world_count
        self.pelvis_positions = np.zeros((self.world_count, 3), dtype=np.float32)
        self.pelvis_rotations = np.zeros((self.world_count, 4), dtype=np.float32)
        self.pelvis_pos_obj = ik.IKObjectivePosition(
            link_index=self.pelvis,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.zeros(self.world_count, dtype=wp.vec3),
        )
        self.pelvis_rot_obj = ik.IKObjectiveRotation(
            link_index=self.pelvis,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.zeros(self.world_count, dtype=wp.vec4),
        )
        foot_objectives = []
        for foot, pose in zip(self.feet, foot_poses, strict=True):
            foot_objectives.append(
                ik.IKObjectivePosition(
                    link_index=foot,
                    link_offset=wp.vec3(0.0, 0.0, 0.0),
                    target_positions=wp.array([pose[:3]] * self.world_count, dtype=wp.vec3),
                )
            )
            foot_objectives.append(
                ik.IKObjectiveRotation(
                    link_index=foot,
                    link_offset_rotation=wp.quat_identity(),
                    target_rotations=wp.array([pose[3:]] * self.world_count, dtype=wp.vec4),
                )
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
            objectives=[self.pelvis_pos_obj, self.pelvis_rot_obj, *foot_objectives, self.limit_obj],
            optimizer=args.optimizer,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        self.tracking_errors = []

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def update_targets(self):
        """Squat, sway, surge and twist each pelvis about a crouch above its planted feet."""
        phase = MOTION_RATE * (self.sim_time + self.time_offsets)
        self.pelvis_positions[:] = self.pelvis_home[:3]
        self.pelvis_positions[:, 0] += SURGE_AMPLITUDE * np.sin(2.0 * phase)
        self.pelvis_positions[:, 1] += SWAY_AMPLITUDE * np.sin(phase)
        self.pelvis_positions[:, 2] += -CROUCH + SQUAT_AMPLITUDE * np.cos(2.0 * phase)
        home_rotation = wp.quat(*self.pelvis_home[3:])
        for i, angle in enumerate(TWIST_AMPLITUDE * np.sin(phase)):
            twist = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(angle))
            self.pelvis_rotations[i] = wp.mul(twist, home_rotation)
        self.pelvis_pos_obj.set_target_positions(wp.array(self.pelvis_positions, dtype=wp.vec3))
        self.pelvis_rot_obj.set_target_rotations(wp.array(self.pelvis_rotations, dtype=wp.vec4))

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
        lower, upper = self.robot_model.joint_limit_lower.numpy(), self.robot_model.joint_limit_upper.numpy()
        limited = (lower > -1e6) & (upper < 1e6)
        # Every joint has one coordinate per DOF except the free root (7 coordinates, 6 DOFs),
        # so DOF d is coordinate d + 1.
        q = self.joint_q.numpy()[:, 1:][:, limited]
        if not np.all((q >= lower[limited] - 1e-5) & (q <= upper[limited] + 1e-5)):
            raise ValueError("A servo left its joint limits")
        pelvis = self.state.body_q.numpy()[self.pelvis :: self.robot_model.body_count, :3]
        self.tracking_errors.extend(np.linalg.norm(pelvis - self.pelvis_positions, axis=1))

    def test_final(self):
        worst = max(self.tracking_errors)
        if worst > 1e-3:
            raise ValueError(f"A pelvis missed its target by {worst * 1e3:.2f} mm, expected at most 1 mm")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
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
