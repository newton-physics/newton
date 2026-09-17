# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Controllers — Differential IK, Humanoid WBC
#
# The motivating case for multi-frame ControllerDifferentialIK: a humanoid
# whole-body-control (WBC) task tracking three frames at once on a single
# robot -- left hand, right hand, and torso -- exactly the frame set a
# WBC controller typically has to combine. tool_sites matches all three, so
# one ControllerDifferentialIK call stacks their rows into a single
# weighted-least-squares solve every step, instead of solving each frame
# separately and combining the results afterward (e.g. with an iterative
# alternating-projection scheme, which does not batch onto the GPU).
#
# Upper body only, kinematics only: the pelvis/spine base is fixed in
# place (no legs, no balance) and the controller's joint targets are
# applied directly to the sim state each frame (no physics solver) --
# keeping the demo focused on the multi-frame IK itself, not full humanoid
# control.
#
# Each arm (shoulder yaw, shoulder pitch, elbow) is redundant against its
# own 3D position-only hand task, and the torso frame shares the spine's 2
# DOFs with both arms -- dragging either hand gizmo can (softly) pull the
# torso and the other arm along with it, the same coupling a real WBC has
# to resolve every tick.
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
SHOULDER_OFFSET = wp.vec3(0.05, 0.2, 0.1)  # forward, out to the side, and up from the spine's own attach point
TORSO_SITE_OFFSET = wp.vec3(0.1, 0.0, 0.3)  # a chest-height reference point, forward of the spine
SPINE_BASE_POSITION = wp.vec3(0.0, 0.0, 1.0)  # pelvis height, fixed in place (no legs)

POSITION_ONLY_AXIS_WEIGHT = wp.spatial_vector(1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
TOOL_SITE_SCALE = (0.02, 0.02, 0.02)

# Spine (yaw, pitch), then each arm (shoulder yaw, shoulder pitch, elbow) --
# a natural, symmetric arms-forward pose.
SPINE_READY_POSE = [0.0, 0.0]
ARM_READY_POSE = [0.0, 0.6, -0.9]  # shoulder yaw, shoulder pitch, elbow
READY_POSE = SPINE_READY_POSE + ARM_READY_POSE + ARM_READY_POSE

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
        # argument for it. Every frame is weighted the same, position-only,
        # here -- a real WBC would likely give the torso frame a lower
        # weight than the hands, or a hard priority over them (task
        # priority is a deliberately separate, not-yet-built mechanism;
        # see the design doc).
        self.controller = ControllerDifferentialIK(
            self.model,
            tool_sites=["tool_hand_left", "tool_hand_right", "tool_torso"],
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
        """Add a 3-DOF arm (shoulder yaw, shoulder pitch, elbow) hanging off ``parent``, with a tool site at the hand.

        The shoulder's two joints are co-located (zero-length between
        them); the upper arm and forearm are each one segment.

        Returns:
            Tuple of (this arm's own joint indices, hand body index, tool
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
        upper_arm_body = builder.add_link()
        joint_shoulder_pitch = builder.add_joint_revolute(
            parent=shoulder_yaw_body,
            child=upper_arm_body,
            axis=_AXIS_Y,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        builder.add_shape_capsule(
            upper_arm_body,
            xform=wp.transform(wp.vec3(UPPER_ARM_LENGTH / 2.0, 0.0, 0.0), _CAPSULE_ROTATION_X),
            radius=0.03,
            half_height=UPPER_ARM_LENGTH / 2.0,
        )
        forearm_body = builder.add_link()
        joint_elbow = builder.add_joint_revolute(
            parent=upper_arm_body,
            child=forearm_body,
            axis=_AXIS_Y,
            parent_xform=wp.transform(wp.vec3(UPPER_ARM_LENGTH, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )
        builder.add_shape_capsule(
            forearm_body,
            xform=wp.transform(wp.vec3(FOREARM_LENGTH / 2.0, 0.0, 0.0), _CAPSULE_ROTATION_X),
            radius=0.025,
            half_height=FOREARM_LENGTH / 2.0,
        )
        hand_transform = wp.transform(wp.vec3(FOREARM_LENGTH, 0.0, 0.0), wp.quat_identity())
        builder.add_site(
            forearm_body, xform=hand_transform, label=f"tool_{side_label}", visible=True, scale=TOOL_SITE_SCALE
        )
        return [joint_shoulder_yaw, joint_shoulder_pitch, joint_elbow], forearm_body, hand_transform

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
            self.viewer.log_gizmo(
                f"target_{i}",
                tf,
                translate=(newton.Axis.X, newton.Axis.Y, newton.Axis.Z),
                snap_to=wp.transform(*tool_pose_world[i].tolist()),
            )
        self.viewer.end_frame()

    def test_final(self):
        """Verify the humanoid stays near its ready pose, since gizmos aren't dragged in headless test mode."""
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        assert np.all(np.isfinite(joint_q)), f"joint_q has NaN/Inf: {joint_q}"
        assert np.all(np.isfinite(joint_qd)), f"joint_qd has NaN/Inf: {joint_qd}"
        ready_q = np.array(READY_POSE, dtype=np.float32)
        assert np.all(np.abs(joint_q - ready_q) < 0.2), f"Joints drifted from ready pose: {joint_q}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
