# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Controllers — Differential IK, Humanoid WBC
#
# One ControllerDifferentialIK call tracking three frames on one robot --
# left hand, right hand, torso -- the multi-frame WBC case. Each 7-DOF arm
# is redundant by 1 against its hand's full 6D pose task (an "elbow swivel",
# like a real arm); both elbows start near their lower joint limit, and
# joint-limit avoidance visibly bends them back out without disturbing the
# tracked hand pose. The torso frame is position-only and left unprotected
# by null_space_axes, a lower priority than the hands.
#
# Upper body only, kinematics only: fixed pelvis, no legs, no physics
# solver -- joint targets are written directly into the sim state.
#
# Command: python -m newton.examples controller_differential_ik_humanoid_wbc
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.controllers import ControllerDifferentialIK, DifferentialIKMethod

UPPER_ARM_LENGTH = 0.25
FOREARM_LENGTH = 0.25
HAND_LENGTH = 0.08
SHOULDER_OFFSET = wp.vec3(0.05, 0.2, 0.1)  # forward, out to the side, and up from the spine's own attach point
TORSO_SITE_OFFSET = wp.vec3(0.1, 0.0, 0.3)  # a chest-height reference point, forward of the spine
SPINE_BASE_POSITION = wp.vec3(0.0, 0.0, 1.0)  # pelvis height, fixed in place (no legs)

FULL_POSE_AXIS_WEIGHT = wp.spatial_vector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
POSITION_ONLY_AXIS_WEIGHT = wp.spatial_vector(1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
UNPROTECTED_AXES = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
TOOL_SITE_SCALE = (0.02, 0.02, 0.02)

# Spine (yaw, pitch), then each arm: shoulder yaw/pitch/roll, elbow, wrist
# yaw/pitch/roll -- 7 DOF against its own hand's 6D pose task, redundant by
# 1. The elbow (index 3 within an arm's own 7) starts at 0.15, inside the
# 0.35 avoidance margin of its 0.0 lower limit -- nearly straight, like an
# arm that hasn't bent into a natural, relaxed pose yet.
ELBOW_INDEX_IN_ARM = 3
SPINE_READY_POSE = [0.0, 0.0]
ARM_READY_POSE = [0.0, 1.0, 0.0, 0.15, 0.0, 0.0, 0.0]
READY_POSE = SPINE_READY_POSE + ARM_READY_POSE + ARM_READY_POSE

SPINE_POS_LOWER = [-1.0, -0.4]
SPINE_POS_UPPER = [1.0, 0.4]
# shoulder yaw, shoulder pitch, shoulder roll, elbow, wrist yaw, wrist pitch, wrist roll
ARM_POS_LOWER = [-1.6, -0.3, -1.6, 0.0, -1.2, -1.2, -1.5]
ARM_POS_UPPER = [1.6, 3.0, 1.6, 2.6, 1.2, 1.2, 1.5]
JOINT_POS_LOWER = SPINE_POS_LOWER + ARM_POS_LOWER + ARM_POS_LOWER
JOINT_POS_UPPER = SPINE_POS_UPPER + ARM_POS_UPPER + ARM_POS_UPPER
JOINT_LIMIT_AVOIDANCE_GAIN = 3.0
JOINT_LIMIT_AVOIDANCE_MARGIN = 0.35

_AXIS_X = wp.vec3(1.0, 0.0, 0.0)
_AXIS_Y = wp.vec3(0.0, 1.0, 0.0)
_AXIS_Z = wp.vec3(0.0, 0.0, 1.0)
# A capsule extends along its own local Z by default; this aligns that with
# local +X, the direction every segment below chains along.
_CAPSULE_ROTATION_X = wp.quat_from_axis_angle(_AXIS_Y, np.pi / 2.0)


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.viewer = viewer
        self.device = wp.get_device()

        # ---- Scene: one humanoid upper body, three tool frames -----------
        builder = newton.ModelBuilder()
        spine_joints, torso_body, torso_site_transform = self._add_spine(builder)
        left_joints, left_hand_body, left_hand_transform = self._add_arm(
            builder, parent=torso_body, shoulder_offset=SHOULDER_OFFSET, side_label="hand_left"
        )
        right_joints, right_hand_body, right_hand_transform = self._add_arm(
            builder,
            parent=torso_body,
            shoulder_offset=wp.vec3(SHOULDER_OFFSET[0], -SHOULDER_OFFSET[1], SHOULDER_OFFSET[2]),
            side_label="hand_right",
        )
        builder.add_articulation([*spine_joints, *left_joints, *right_joints], label="humanoid")
        for coord, angle in enumerate(READY_POSE):
            builder.joint_q[coord] = angle

        builder.add_ground_plane()
        self.model = builder.finalize(device=self.device)
        self.state_0 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # ---- Differential-kinematics controller, three frames on one robot
        # frames_per_robot is inferred from how many sites tool_sites
        # matches (3 here: both hands and the torso) -- there's no separate
        # argument for it. Hands track full pose; the torso frame stays
        # position-only. null_space_axes leaves the torso frame entirely
        # unprotected (independent of axis_weight), so joint-limit
        # avoidance only has to preserve both hands' pose, not the torso's
        # position too -- a softer priority for the torso, matching the
        # module docstring's note above.
        self.controller = ControllerDifferentialIK(
            self.model,
            tool_sites=["tool_hand_left", "tool_hand_right", "tool_torso"],
            axis_weight=wp.array(
                [FULL_POSE_AXIS_WEIGHT, FULL_POSE_AXIS_WEIGHT, POSITION_ONLY_AXIS_WEIGHT],
                dtype=wp.spatial_vector,
                device=self.device,
            ),
            bandwidth=10.0,
            damping=0.1,
            ik_method=DifferentialIKMethod.DAMPED_LEAST_SQUARES,
            use_joint_limit_avoidance=True,
            joint_limit_avoidance_gain=JOINT_LIMIT_AVOIDANCE_GAIN,
            joint_limit_avoidance_margin=JOINT_LIMIT_AVOIDANCE_MARGIN,
            joint_pos_lower=wp.array(JOINT_POS_LOWER, dtype=wp.float32, device=self.device),
            joint_pos_upper=wp.array(JOINT_POS_UPPER, dtype=wp.float32, device=self.device),
            null_space_axes=wp.array(
                [FULL_POSE_AXIS_WEIGHT, FULL_POSE_AXIS_WEIGHT, UNPROTECTED_AXES],
                dtype=wp.spatial_vector,
                device=self.device,
            ),
        )

        self._input = self.controller.input()
        self._output = self.controller.output()
        self._input.joint_q = self.state_0.joint_q
        self._input.joint_qd = self.state_0.joint_qd
        self._output.joint_q_target = self.state_0.joint_q[self.controller.q_start]
        self._output.joint_qd_target = self.state_0.joint_qd[self.controller.qd_start]

        # One draggable gizmo per frame, seeded at each site's actual
        # starting world pose. tool_sites resolves a robot's frames in its
        # own entry order, so this list matches tool_sites above exactly.
        body_q_np = self.state_0.body_q.numpy()
        self.gizmo_tfs = [
            wp.transform(*body_q_np[left_hand_body].tolist()) * left_hand_transform,
            wp.transform(*body_q_np[right_hand_body].tolist()) * right_hand_transform,
            wp.transform(*body_q_np[torso_body].tolist()) * torso_site_transform,
        ]

        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(1.6, 0.0, 1.0), pitch=-10.0, yaw=180.0)

        self.viewer.set_model(self.model)

        self.graph = None
        if self.controller.is_graphable() and self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate()
            self.graph = capture.graph

    @staticmethod
    def _add_spine(builder):
        """Add a 2-DOF spine (yaw, then pitch) and a torso tool site at chest height.

        Returns:
            Tuple of (spine joint indices, torso body index, tool site's
            body-local transform).
        """
        spine_link = builder.add_link()
        joint_yaw = builder.add_joint_revolute(
            parent=-1,
            child=spine_link,
            axis=_AXIS_Z,
            parent_xform=wp.transform(SPINE_BASE_POSITION, wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )
        torso_body = builder.add_link()
        joint_pitch = builder.add_joint_revolute(
            parent=spine_link,
            child=torso_body,
            axis=_AXIS_Y,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        builder.add_shape_box(torso_body, xform=wp.transform_identity(), hx=0.08, hy=0.22, hz=0.15)
        torso_site_transform = wp.transform(TORSO_SITE_OFFSET, wp.quat_identity())
        builder.add_site(
            torso_body, xform=torso_site_transform, label="tool_torso", visible=True, scale=TOOL_SITE_SCALE
        )
        return [joint_yaw, joint_pitch], torso_body, torso_site_transform

    @staticmethod
    def _add_arm(builder, *, parent, shoulder_offset, side_label):
        """Add a 7-DOF human-like arm hanging off ``parent``, with a tool site at the hand.

        Shoulder (yaw, pitch, roll) and wrist (yaw, pitch, roll) are each 3
        co-located joints -- a spherical joint approximated by 3 revolutes,
        every parent_xform between them an identity rotation, so each
        joint's own axis is expressed in whatever rotation the joints
        before it already accumulated. The elbow sits one upper-arm length
        out from the shoulder; the hand site sits one hand length out from
        the wrist.

        Returns:
            Tuple of (this arm's own 7 joint indices, hand body index, tool
            site's body-local transform).
        """
        shoulder_yaw_body = builder.add_link()
        joint_shoulder_yaw = builder.add_joint_revolute(
            parent=parent,
            child=shoulder_yaw_body,
            axis=_AXIS_Z,
            parent_xform=wp.transform(shoulder_offset, wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )
        shoulder_pitch_body = builder.add_link()
        joint_shoulder_pitch = builder.add_joint_revolute(
            parent=shoulder_yaw_body,
            child=shoulder_pitch_body,
            axis=_AXIS_Y,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        shoulder_roll_body = builder.add_link()
        joint_shoulder_roll = builder.add_joint_revolute(
            parent=shoulder_pitch_body,
            child=shoulder_roll_body,
            axis=_AXIS_X,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        builder.add_shape_capsule(
            shoulder_roll_body,
            xform=wp.transform(wp.vec3(UPPER_ARM_LENGTH / 2.0, 0.0, 0.0), _CAPSULE_ROTATION_X),
            radius=0.03,
            half_height=UPPER_ARM_LENGTH / 2.0,
        )

        elbow_body = builder.add_link()
        joint_elbow = builder.add_joint_revolute(
            parent=shoulder_roll_body,
            child=elbow_body,
            axis=_AXIS_Y,
            parent_xform=wp.transform(wp.vec3(UPPER_ARM_LENGTH, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )
        builder.add_shape_capsule(
            elbow_body,
            xform=wp.transform(wp.vec3(FOREARM_LENGTH / 2.0, 0.0, 0.0), _CAPSULE_ROTATION_X),
            radius=0.025,
            half_height=FOREARM_LENGTH / 2.0,
        )

        wrist_yaw_body = builder.add_link()
        joint_wrist_yaw = builder.add_joint_revolute(
            parent=elbow_body,
            child=wrist_yaw_body,
            axis=_AXIS_Z,
            parent_xform=wp.transform(wp.vec3(FOREARM_LENGTH, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )
        wrist_pitch_body = builder.add_link()
        joint_wrist_pitch = builder.add_joint_revolute(
            parent=wrist_yaw_body,
            child=wrist_pitch_body,
            axis=_AXIS_Y,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        wrist_roll_body = builder.add_link()
        joint_wrist_roll = builder.add_joint_revolute(
            parent=wrist_pitch_body,
            child=wrist_roll_body,
            axis=_AXIS_X,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        builder.add_shape_capsule(
            wrist_roll_body,
            xform=wp.transform(wp.vec3(HAND_LENGTH / 2.0, 0.0, 0.0), _CAPSULE_ROTATION_X),
            radius=0.02,
            half_height=HAND_LENGTH / 2.0,
        )

        hand_transform = wp.transform(wp.vec3(HAND_LENGTH, 0.0, 0.0), wp.quat_identity())
        builder.add_site(
            wrist_roll_body, xform=hand_transform, label=f"tool_{side_label}", visible=True, scale=TOOL_SITE_SCALE
        )
        joints = [
            joint_shoulder_yaw,
            joint_shoulder_pitch,
            joint_shoulder_roll,
            joint_elbow,
            joint_wrist_yaw,
            joint_wrist_pitch,
            joint_wrist_roll,
        ]
        return joints, wrist_roll_body, hand_transform

    def _simulate(self):
        self.controller.step(inputs=self._input, outputs=self._output, dt=self.frame_dt)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def step(self):
        pose = np.zeros((len(self.gizmo_tfs), 7), dtype=np.float32)
        for i, tf in enumerate(self.gizmo_tfs):
            pose[i, :3] = wp.transform_get_translation(tf)
            pose[i, 3:] = wp.transform_get_rotation(tf)
        self._input.desired_tool_pose_world.assign(pose)

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        tool_pose_world = self.controller.tool_pose_world.numpy()
        for i, tf in enumerate(self.gizmo_tfs):
            # Hands (frames 0, 1) track full pose, so their gizmos expose
            # rotation handles too; the torso (frame 2) is position-only.
            rotate = (newton.Axis.X, newton.Axis.Y, newton.Axis.Z) if i < 2 else ()
            self.viewer.log_gizmo(
                f"target_{i}",
                tf,
                translate=(newton.Axis.X, newton.Axis.Y, newton.Axis.Z),
                rotate=rotate,
                snap_to=wp.transform(*tool_pose_world[i].tolist()),
            )
        self.viewer.end_frame()

    def test_final(self):
        """Verify both hands stay at their target pose while joint-limit avoidance clears each elbow's near-limit start."""
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        assert np.all(np.isfinite(joint_q)), f"joint_q has NaN/Inf: {joint_q}"
        assert np.all(np.isfinite(joint_qd)), f"joint_qd has NaN/Inf: {joint_qd}"

        # Zero task error the whole time (gizmos never dragged), so each
        # hand's tracked position must stay at its starting target despite
        # joint-limit avoidance moving several joints -- exactly what
        # "doesn't disturb the primary task" means for the elbow-swivel
        # redundancy to be worth having.
        tool_pose_world = self.controller.tool_pose_world.numpy()
        for i in range(2):
            target_pos = np.array(wp.transform_get_translation(self.gizmo_tfs[i]))
            tracked_pos = np.array(wp.transform_get_translation(wp.transform(*tool_pose_world[i].tolist())))
            assert np.allclose(tracked_pos, target_pos, atol=0.05), (
                f"Hand {i} drifted from its target: {tracked_pos} vs {target_pos}"
            )

        # Both elbows started at 0.15, inside the 0.35 avoidance margin of
        # their 0.0 lower limit; joint-limit avoidance should pull them
        # clear of it.
        elbow_start = ARM_READY_POSE[ELBOW_INDEX_IN_ARM]
        left_elbow_idx = len(SPINE_READY_POSE) + ELBOW_INDEX_IN_ARM
        right_elbow_idx = len(SPINE_READY_POSE) + len(ARM_READY_POSE) + ELBOW_INDEX_IN_ARM
        for idx in (left_elbow_idx, right_elbow_idx):
            assert joint_q[idx] > elbow_start + 0.05, (
                f"Elbow at joint index {idx} did not clear its limit margin: {joint_q[idx]}"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
