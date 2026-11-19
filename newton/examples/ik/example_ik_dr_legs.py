# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example IK DR Legs (closed-loop biped legs)

Whole-body inverse kinematics of DR Legs, a biped from Disney Research whose legs are
serial-parallel linkages with three closed kinematic loops each, solved for many robots in one
batch.
- The asset's articulation tree reaches every body; six revolute joints close the loops.
- The robots walk single file around concentric rings, neighboring rings in opposite
  directions: IK places the pelvis along the ring and each foot on its footsteps, alternately
  planted and swinging.
- ``IKOptimizer.CUNLS`` keeps the loops closed (equality constraints) and the servos inside
  their limits (hard bounds); ``--optimizer lm`` uses Newton's default optimizer, which ignores
  loop-closing joints.

Success criteria:
- Rings of bipeds walk in circles, one behind the other, feet stepping on the ground.
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

FOOT_SOLE = 0.018  # [m], depth of the sole below the foot frame

# Rings: robots walk single file, ROBOT_GAP apart along each ring.
FIRST_RING_RADIUS = 1.0  # [m]
RING_GAP = 0.4  # [m]
ROBOT_GAP = 0.35  # [m]

# Gait, relative to the pose the asset is loaded in (legs almost straight).
CROUCH = 0.04  # [m], pelvis height below the loaded pose
WALK_SPEED = 0.08  # [m/s]
GAIT_PERIOD = 0.8  # [s], each foot swings once per period
STEP_HEIGHT = 0.02  # [m]


def ring_layout(robot_count):
    """Place the robots single file on concentric rings.

    Returns the ring radius [m], starting angle [rad] and walking direction (+1 counterclockwise,
    -1 clockwise) of each robot.
    """
    radius, angle, direction = [], [], []
    ring = 0
    while len(radius) < robot_count:
        r = FIRST_RING_RADIUS + ring * RING_GAP
        count = min(int(2.0 * math.pi * r / ROBOT_GAP), robot_count - len(radius))
        radius += [r] * count
        angle += list(2.0 * math.pi * np.arange(count) / count)
        direction += [1.0 if ring % 2 == 0 else -1.0] * count
        ring += 1
    return np.array(radius), np.array(angle), np.array(direction)


def yaw_quats(yaw):
    """Quaternions (xyzw) of rotations by ``yaw`` [rad] about the vertical axis."""
    q = np.zeros((len(yaw), 4), dtype=np.float32)
    q[:, 2] = np.sin(0.5 * yaw)
    q[:, 3] = np.cos(0.5 * yaw)
    return q


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

        # In the loaded pose the robot faces +x, its loops are closed and its bodies are not
        # rotated: the gait places the pelvis and feet at these offsets, turned along the ring.
        state = self.robot_model.state()
        newton.eval_fk(self.robot_model, self.robot_model.joint_q, self.robot_model.joint_qd, state)
        home = state.body_q.numpy()
        self.pelvis_offset = home[self.pelvis, :3] - np.array([0.0, 0.0, CROUCH])
        self.foot_offsets = home[self.feet, :3]

        scene = newton.ModelBuilder()
        scene.replicate(robot, self.world_count)
        scene.add_ground_plane(height=float(self.foot_offsets[:, 2].min()) - FOOT_SOLE)
        self.model = scene.finalize()
        self.state = self.model.state()

        # All robots share one space; the rings keep them apart.
        self.ring_radius, self.ring_angle, self.ring_direction = ring_layout(self.world_count)
        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        distance = 0.9 * self.ring_radius.max() + 0.8
        self.viewer.set_camera(pos=wp.vec3(distance, -distance, 0.9 * distance), pitch=-35.0, yaw=135.0)

        # ------------------------------------------------------------------
        # IK: pelvis and foot poses, servo limits; one problem per robot
        # ------------------------------------------------------------------
        def pose_objectives(link):
            position = ik.IKObjectivePosition(
                link_index=link,
                link_offset=wp.vec3(0.0, 0.0, 0.0),
                target_positions=wp.zeros(self.world_count, dtype=wp.vec3),
            )
            rotation = ik.IKObjectiveRotation(
                link_index=link,
                link_offset_rotation=wp.quat_identity(),
                target_rotations=wp.zeros(self.world_count, dtype=wp.vec4),
            )
            return position, rotation

        self.pose_objectives = [pose_objectives(link) for link in (self.pelvis, *self.feet)]
        self.limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.robot_model.joint_limit_lower,
            joint_limit_upper=self.robot_model.joint_limit_upper,
        )

        self.ik_iters = 30
        self.solver = ik.IKSolver(
            model=self.robot_model,
            n_problems=self.world_count,
            objectives=[*(obj for pair in self.pose_objectives for obj in pair), self.limit_obj],
            optimizer=args.optimizer,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        # Row i of joint_q holds the coordinates of world i. Start each robot (its free root:
        # position, then rotation) at its place on the ring.
        self.joint_q = self.model.joint_q.reshape((self.world_count, self.robot_model.joint_coord_count))
        self.update_targets()
        q = self.joint_q.numpy()
        q[:, 0:3], q[:, 3:7] = self.target_positions[0], self.target_rotations[0]
        self.joint_q.assign(q)

        self.tracking_errors = []

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def place_on_ring(self, arc, offset):
        """Position [m] and heading [rad] of a point ``offset`` (forward, left, up) from the spot
        ``arc`` [m] along each robot's ring, facing the walking direction."""
        theta = self.ring_angle + self.ring_direction * arc / self.ring_radius
        heading = np.stack([-np.sin(theta), np.cos(theta)], axis=1) * self.ring_direction[:, None]
        left = np.stack([-heading[:, 1], heading[:, 0]], axis=1)
        position = np.empty((self.world_count, 3), dtype=np.float32)
        position[:, :2] = (
            self.ring_radius[:, None] * np.stack([np.cos(theta), np.sin(theta)], axis=1)
            + offset[0] * heading
            + offset[1] * left
        )
        position[:, 2] = offset[2]
        return position, theta + 0.5 * math.pi * self.ring_direction  # continuous: no jumps at +-pi

    def update_targets(self):
        """Advance the pelvis along the ring and step the feet: each foot stays planted for half a
        gait period, then swings one stride forward in an arc."""
        stride = WALK_SPEED * GAIT_PERIOD
        positions, yaws = [], []
        position, yaw = self.place_on_ring(WALK_SPEED * self.sim_time, self.pelvis_offset)
        positions.append(position)
        yaws.append(yaw)
        for side, offset in enumerate(self.foot_offsets):
            cycle = self.sim_time / GAIT_PERIOD + 0.5 * side  # the feet swing half a period apart
            swing = min(max(2.0 * (cycle % 1.0) - 1.0, 0.0), 1.0)  # 0 while planted, 0 -> 1 in the swing
            swing = swing * swing * (3.0 - 2.0 * swing)  # smooth start and stop
            # Footsteps are one stride apart; on average each foot keeps pace with the pelvis.
            arc = stride * (math.floor(cycle) + swing - 0.5 * side + 0.25)
            lifted = offset + np.array([0.0, 0.0, STEP_HEIGHT * math.sin(math.pi * swing)])
            position, yaw = self.place_on_ring(arc, lifted)
            positions.append(position)
            yaws.append(yaw)
        self.target_positions = positions
        self.target_rotations = [yaw_quats(yaw) for yaw in yaws]
        for (position_obj, rotation_obj), position, rotation in zip(
            self.pose_objectives, self.target_positions, self.target_rotations, strict=True
        ):
            position_obj.set_target_positions(wp.array(position, dtype=wp.vec3))
            rotation_obj.set_target_rotations(wp.array(rotation, dtype=wp.vec4))

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
        body_q = self.state.body_q.numpy()
        for link, target in zip((self.pelvis, *self.feet), self.target_positions, strict=True):
            position = body_q[link :: self.robot_model.body_count, :3]
            self.tracking_errors.extend(np.linalg.norm(position - target, axis=1))

    def test_final(self):
        worst = max(self.tracking_errors)
        if worst > 1e-3:
            raise ValueError(f"A pelvis or foot missed its target by {worst * 1e3:.2f} mm, expected at most 1 mm")

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
