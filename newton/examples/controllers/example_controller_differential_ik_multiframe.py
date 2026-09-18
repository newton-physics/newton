# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Controllers — Differential IK, Multi-Frame
#
# Demonstrates ControllerDifferentialIK controlling two independent tool
# frames per robot, on two robots at once. Each is Y-shaped: a shared pan
# joint at the base, splitting into a left and a right sub-arm. tool_sites
# matches both tips ("tool_left", "tool_right") on each robot, so each
# robot's own controller solve stacks its two frames' rows into one
# combined, weighted-least-squares task rather than treating them as
# separate problems. The shared pan joint means dragging one gizmo also
# (softly) moves that robot's other tip -- exactly the coupling a real
# dual-arm/shared-torso task has.
#
# Robot A's sub-arms are 2-DOF each (5 DOFs total driving a 6D combined
# task: under-determined, not redundant). Robot B's are 3-DOF each (7 DOFs
# total: redundant by 1, the same "7 DOF against a 6D task" shape as a
# Franka arm). Null-space posture control isn't available for a multi-frame
# robot (see ControllerDifferentialIKModelFree's frames_per_robot), so
# robot B's extra DOF isn't auto-centered by a posture target -- it's just
# slack the solver is free to use however the combined task allows.
#
# Robot C is a single-tool-site, 4-DOF arm on its own separate
# ControllerDifferentialIK instance (multi-frame robots can't use
# use_joint_limit_avoidance), redundant by 1 against its own 3D position
# task -- demonstrating that same null-space joint-limit avoidance actually
# working: its last joint starts near its upper limit and gets pulled back
# toward mid-range without disturbing its own tool's tracked position.
#
# Kinematics only: the controller's joint targets are applied directly to
# the sim state each frame (no physics solver), keeping the demo focused on
# the IK itself.
#
# Command: python -m newton.examples controller_differential_ik_multiframe
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.controllers import ControllerDifferentialIK, DifferentialIKMethod

ARM_LINK_LENGTH = 0.3
ARM_SIDE_OFFSET = 0.3
POSITION_ONLY_AXIS_WEIGHT = wp.spatial_vector(1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
TOOL_SITE_SCALE = (0.02, 0.02, 0.02)
# A capsule extends along its own local Z by default; these align that with
# local +X (the arm segments' own chaining direction) and local +Y (the pan
# link's shoulder-to-shoulder bar), respectively.
_CAPSULE_ROTATION_X = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2.0)
_CAPSULE_ROTATION_Y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -np.pi / 2.0)

_AXIS_Y = wp.vec3(0.0, 1.0, 0.0)  # bends the arm up/down, in whatever vertical plane precedes it
_AXIS_Z = wp.vec3(0.0, 0.0, 1.0)  # yaws the arm, parallel to the ground plane

# Pan, then 2 joints per arm (not redundant against the 6D combined task).
# Both arm joints are Y (up/down bends only) -- the shared pan joint is the
# only source of ground-plane-parallel motion for this robot.
ROBOT_A_BASE_POSITION = wp.vec3(0.0, 0.0, 0.6)
ROBOT_A_JOINT_AXES = [_AXIS_Y, _AXIS_Y]
ROBOT_A_READY_POSE = [0.0, 0.7, -1.0, 0.7, -1.0]

# Pan, then 3 joints per arm (redundant by 1 against the 6D combined task,
# the same shape as a 7-DOF arm against a single 6D pose task). Each arm's
# own first joint is a Z yaw, so it isn't only the shared pan bar that
# moves things parallel to the ground plane -- each arm can sweep
# horizontally on its own, independent of the other.
ROBOT_B_BASE_POSITION = wp.vec3(0.0, -1.6, 0.6)
ROBOT_B_JOINT_AXES = [_AXIS_Z, _AXIS_Y, _AXIS_Y]
ROBOT_B_READY_POSE = [0.0, 0.5, -0.8, 0.3, 0.5, -0.8, 0.3]

# A single-tool-site, single-frame robot: use_joint_limit_avoidance requires
# every robot on its own controller to have exactly one frame, unlike A and
# B above, so this one gets its own ControllerDifferentialIK instance. 4
# DOFs against a 3D position-only task -- redundant by 1, giving the
# null-space projector a genuine (1D) direction to work in. Its own last
# joint starts deliberately close to its upper limit, so the joint-limit
# avoidance bias visibly pulls it back toward mid-range over the first few
# seconds, even with zero primary-task error the whole time.
ROBOT_C_BASE_POSITION = wp.vec3(0.0, 1.6, 0.6)
ROBOT_C_JOINT_AXES = [_AXIS_Z, _AXIS_Y, _AXIS_Y, _AXIS_Y]
ROBOT_C_READY_POSE = [0.2, 0.4, -0.5, 1.3]
ROBOT_C_JOINT_POS_LOWER = [-1.4, -1.4, -1.4, -1.4]
ROBOT_C_JOINT_POS_UPPER = [1.4, 1.4, 1.4, 1.4]


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.viewer = viewer
        self.device = wp.get_device()

        # ---- Scene: two Y-shaped, dual-tip robots -------------------------
        builder = newton.ModelBuilder()
        _, left_tip_a, left_tf_a, right_tip_a, right_tf_a = self._add_dual_arm_robot(
            builder,
            base_position=ROBOT_A_BASE_POSITION,
            joint_axes=ROBOT_A_JOINT_AXES,
            label="robot_a",
        )
        _, left_tip_b, left_tf_b, right_tip_b, right_tf_b = self._add_dual_arm_robot(
            builder,
            base_position=ROBOT_B_BASE_POSITION,
            joint_axes=ROBOT_B_JOINT_AXES,
            label="robot_b",
        )
        c_joints, tip_c, tf_c = self._add_arm(
            builder, parent=-1, side_offset=ROBOT_C_BASE_POSITION, side_label="c", joint_axes=ROBOT_C_JOINT_AXES
        )
        builder.add_articulation(c_joints, label="robot_c")
        for coord, angle in enumerate(ROBOT_A_READY_POSE + ROBOT_B_READY_POSE + ROBOT_C_READY_POSE):
            builder.joint_q[coord] = angle

        builder.add_ground_plane()
        self.model = builder.finalize(device=self.device)
        self.state_0 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # ---- Differential-kinematics controllers -------------------------
        # One call handles robots A and B, restricted to just those two
        # articulations (robot C has no "tool_left"/"tool_right" site, so
        # leaving articulations at its None default would make tool_sites
        # fail to match on it). frames_per_robot is inferred from how many
        # sites tool_sites matches on each one's own articulation (2 here,
        # for both) -- there's no separate argument for it.
        self.controller = ControllerDifferentialIK(
            self.model,
            articulations=["robot_a", "robot_b"],
            tool_sites=["tool_left", "tool_right"],
            axis_weight=POSITION_ONLY_AXIS_WEIGHT,
            bandwidth=10.0,
            damping=0.1,
            ik_method=DifferentialIKMethod.DAMPED_LEAST_SQUARES,
        )

        self._input = self.controller.input()
        self._output = self.controller.output()
        self._input.joint_q = self.state_0.joint_q
        self._input.joint_qd = self.state_0.joint_qd
        self._output.joint_q_target = self.state_0.joint_q[self.controller.q_start]
        self._output.joint_qd_target = self.state_0.joint_qd[self.controller.qd_start]

        # Robot C: its own controller, single frame, with joint-limit
        # avoidance enabled -- ControllerDifferentialIKModelFree requires
        # every robot on a controller to have exactly one frame whenever
        # use_joint_limit_avoidance/use_null_space_posture_control is set,
        # so this can't be folded into self.controller above.
        joint_pos_lower = wp.array(ROBOT_C_JOINT_POS_LOWER, dtype=wp.float32, device=self.device)
        joint_pos_upper = wp.array(ROBOT_C_JOINT_POS_UPPER, dtype=wp.float32, device=self.device)
        self.controller_c = ControllerDifferentialIK(
            self.model,
            articulations="robot_c",
            tool_sites="tool_c",
            axis_weight=POSITION_ONLY_AXIS_WEIGHT,
            bandwidth=10.0,
            damping=0.1,
            ik_method=DifferentialIKMethod.DAMPED_LEAST_SQUARES,
            use_joint_limit_avoidance=True,
            joint_limit_avoidance_gain=3.0,
            joint_limit_avoidance_margin=0.3,
            joint_pos_lower=joint_pos_lower,
            joint_pos_upper=joint_pos_upper,
            null_space_damping=0.05,
        )
        self._input_c = self.controller_c.input()
        self._output_c = self.controller_c.output()
        self._input_c.joint_q = self.state_0.joint_q
        self._input_c.joint_qd = self.state_0.joint_qd
        self._output_c.joint_q_target = self.state_0.joint_q[self.controller_c.q_start]
        self._output_c.joint_qd_target = self.state_0.joint_qd[self.controller_c.qd_start]

        # One draggable gizmo per frame, seeded at each tip's actual
        # starting world pose. Order matches tool_sites's own resolution:
        # robot 0's frames first (left, then right), then robot 1's; robot
        # C's single frame is tracked separately, appended last.
        body_q_np = self.state_0.body_q.numpy()
        self.gizmo_tfs = [
            wp.transform(*body_q_np[left_tip_a].tolist()) * left_tf_a,
            wp.transform(*body_q_np[right_tip_a].tolist()) * right_tf_a,
            wp.transform(*body_q_np[left_tip_b].tolist()) * left_tf_b,
            wp.transform(*body_q_np[right_tip_b].tolist()) * right_tf_b,
            wp.transform(*body_q_np[tip_c].tolist()) * tf_c,
        ]

        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(2.2, -0.8, 1.4), pitch=-20.0, yaw=180.0)

        self.viewer.set_model(self.model)

        self.graph = None
        if self.controller.is_graphable() and self.controller_c.is_graphable() and self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate()
            self.graph = capture.graph

    @staticmethod
    def _add_dual_arm_robot(builder, *, base_position, joint_axes, label):
        """Build one Y-shaped robot: a shared pan joint splitting into a left/right sub-arm, each with ``joint_axes``.

        Returns:
            Tuple of (this robot's joint indices in build order, left tip
            body, left tool transform, right tip body, right tool
            transform).
        """
        pan = builder.add_link()
        joint_pan = builder.add_joint_revolute(
            parent=-1,
            child=pan,
            axis=_AXIS_Z,
            parent_xform=wp.transform(base_position, wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )
        builder.add_shape_capsule(
            pan,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), _CAPSULE_ROTATION_Y),
            radius=0.02,
            half_height=ARM_SIDE_OFFSET,
        )
        left_joints, left_tip_body, left_tool_transform = Example._add_arm(
            builder,
            parent=pan,
            side_offset=wp.vec3(0.0, ARM_SIDE_OFFSET, 0.0),
            side_label="left",
            joint_axes=joint_axes,
        )
        right_joints, right_tip_body, right_tool_transform = Example._add_arm(
            builder,
            parent=pan,
            side_offset=wp.vec3(0.0, -ARM_SIDE_OFFSET, 0.0),
            side_label="right",
            joint_axes=joint_axes,
        )
        joints = [joint_pan, *left_joints, *right_joints]
        builder.add_articulation(joints, label=label)
        return joints, left_tip_body, left_tool_transform, right_tip_body, right_tool_transform

    @staticmethod
    def _add_arm(builder, *, parent, side_offset, side_label, joint_axes):
        """Add a sub-arm (one joint per entry of ``joint_axes``) hanging off ``parent``, with a tool site at its tip.

        Each joint's own ``parent_xform`` is an identity rotation, so
        ``joint_axes`` entries are expressed in whatever rotation the prior
        joints already accumulated -- e.g. a Z axis after a Y bend yaws
        within that bend's own current plane, not world Z.

        Returns:
            Tuple of (this sub-arm's own joint indices, tip body index,
            tool site's body-local transform).
        """
        joints = []
        link = parent
        parent_xform = wp.transform(side_offset, wp.quat_identity())
        for axis in joint_axes:
            next_link = builder.add_link()
            joint = builder.add_joint_revolute(
                parent=link,
                child=next_link,
                axis=axis,
                parent_xform=parent_xform,
                child_xform=wp.transform_identity(),
            )
            builder.add_shape_capsule(
                next_link,
                xform=wp.transform(wp.vec3(ARM_LINK_LENGTH / 2.0, 0.0, 0.0), _CAPSULE_ROTATION_X),
                radius=0.03,
                half_height=ARM_LINK_LENGTH / 2.0,
            )
            joints.append(joint)
            link = next_link
            parent_xform = wp.transform(wp.vec3(ARM_LINK_LENGTH, 0.0, 0.0), wp.quat_identity())
        tool_site_transform = wp.transform(wp.vec3(ARM_LINK_LENGTH, 0.0, 0.0), wp.quat_identity())
        builder.add_site(
            link, xform=tool_site_transform, label=f"tool_{side_label}", visible=True, scale=TOOL_SITE_SCALE
        )
        return joints, link, tool_site_transform

    def _simulate(self):
        self.controller.step(inputs=self._input, outputs=self._output, dt=self.frame_dt)
        self.controller_c.step(inputs=self._input_c, outputs=self._output_c, dt=self.frame_dt)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def step(self):
        pose = np.zeros((len(self.gizmo_tfs), 7), dtype=np.float32)
        for i, tf in enumerate(self.gizmo_tfs):
            pose[i, :3] = wp.transform_get_translation(tf)
            pose[i, 3:] = wp.transform_get_rotation(tf)
        self._input.desired_tool_pose_world.assign(pose[:4])
        self._input_c.desired_tool_pose_world.assign(pose[4:])

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        tool_pose_world = np.concatenate(
            [self.controller.tool_pose_world.numpy(), self.controller_c.tool_pose_world.numpy()]
        )
        for i, tf in enumerate(self.gizmo_tfs):
            self.viewer.log_gizmo(
                f"target_{i}",
                tf,
                translate=(newton.Axis.X, newton.Axis.Y, newton.Axis.Z),
                snap_to=wp.transform(*tool_pose_world[i].tolist()),
            )
        self.viewer.end_frame()

    def test_final(self):
        """Verify A and B stay near their ready pose, and C's redundant joint moved off its near-limit start."""
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        assert np.all(np.isfinite(joint_q)), f"joint_q has NaN/Inf: {joint_q}"
        assert np.all(np.isfinite(joint_qd)), f"joint_qd has NaN/Inf: {joint_qd}"

        ab_count = len(ROBOT_A_READY_POSE) + len(ROBOT_B_READY_POSE)
        ab_ready_q = np.array(ROBOT_A_READY_POSE + ROBOT_B_READY_POSE, dtype=np.float32)
        assert np.all(np.abs(joint_q[:ab_count] - ab_ready_q) < 0.2), (
            f"Robot A/B joints drifted from ready pose: {joint_q[:ab_count]}"
        )

        # Robot C: the null-space direction is a *combination* of all 4
        # joints that leaves the tool position fixed while joint 4 moves --
        # not "joint 4 moves, the other 3 stay put" -- so the invariant to
        # check is the tool position itself staying near its (unmoved)
        # target, not individual joint values.
        c_q = joint_q[ab_count:]
        tool_pos = np.array(wp.transform_get_translation(self.gizmo_tfs[4]))
        tracked_pos = np.array(
            wp.transform_get_translation(wp.transform(*self.controller_c.tool_pose_world.numpy()[0]))
        )
        assert np.allclose(tracked_pos, tool_pos, atol=0.05), (
            f"Robot C's tool drifted from its target despite zero task error: {tracked_pos} vs {tool_pos}"
        )
        # The redundant 4th joint started at 1.3, within the 0.3 margin of
        # its 1.4 upper limit; joint-limit avoidance should pull it clear
        # of that margin band, to its natural equilibrium right at the
        # margin's edge (1.4 - 0.3 = 1.1) -- the bias ramps to exactly zero
        # once a DOF clears the margin, so it doesn't get pulled any
        # further than that, all the way to mid-range.
        assert c_q[3] < 1.4 - 0.3 + 0.05, f"Robot C's redundant joint did not clear its limit margin: {c_q[3]}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
