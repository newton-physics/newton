# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Controllers — Differential IK
#
# Demonstrates ControllerDiffIK on two real, heterogeneous robots at once --
# a 7-DOF Franka Panda arm (redundant against the 6D task) and a 6-DOF UR10
# arm (not redundant) -- each independently tracking its own draggable
# 6-DOF gizmo target. One controller call handles both, each robot resolved
# through its own tool site and Jacobian.
#
# ControllerDiffIK outputs one-step-ahead joint position/velocity targets,
# not torques, so every controlled arm DOF is left in MuJoCo's
# POSITION_VELOCITY actuator mode: the controller runs once per frame, and
# MuJoCo's own implicit PD tracks that fixed target across every physics
# substep. The Franka's fingers are left untouched, held at their initial
# target by the same PD.
#
# The Franka's 7th DOF is redundant against the 6D task; null-space posture
# control continuously pulls it back toward its ready pose, so it doesn't
# drift toward a bad internal configuration with nothing to anchor it. The
# UR10 has no redundant DOF, so the same posture control is a no-op for it.
#
# Uses IkMethod.ADAPTIVE_DAMPING: damping ramps up automatically as either
# arm nears a kinematic singularity or the edge of its reach, instead of a
# single fixed damping value -- this stays smooth (no chatter) right up to
# the boundary, unlike a plain fixed-damping or truncated-SVD solve.
#
# Command: python -m newton.examples controller_diff_ik
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.solvers
import newton.utils
from newton import JointTargetMode
from newton.controllers import ControllerDiffIK

IkMethod = ControllerDiffIK.IkMethod

# ---------------------------------------------------------------------------
# Robot configuration
# ---------------------------------------------------------------------------

# Franka's standard "ready" pose.
FRANKA_READY_POSE = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
FRANKA_ARM_DOFS = len(FRANKA_READY_POSE)  # 7; redundant against the 6D task
FRANKA_BASE_POSITION = wp.vec3(0.0, 0.0, 0.0)

# A UR10 configuration reaching forward and down, the same reach scale as
# the Franka's ready pose.
UR10_READY_POSE = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
UR10_ARM_DOFS = len(UR10_READY_POSE)  # 6; no redundant DOF, unlike the Franka
UR10_BASE_POSITION = wp.vec3(0.0, 1.8, 0.0)  # separated from the Franka along Y

TOOL_SITE_SCALE = (0.02, 0.02, 0.02)

# Every controlled arm DOF tracks joint_target_q/qd via MuJoCo's own
# implicit PD, driven once per frame by the controller's one-step-ahead
# targets.
JOINT_TARGET_KE = 3000.0
JOINT_TARGET_KD = 100.0

BANDWIDTH = 20.0
# Empirically tuned against this example's continuous velocity-based control
# loop (see IkMethod.ADAPTIVE_DAMPING): min/max away from vs. at a full
# singularity, threshold set well clear of the arms' reachable workspace so
# damping is already ramped up before either arm gets there, instead of
# still transitioning exactly at the boundary (which chatters).
ADAPTIVE_DAMPING_MIN = 0.02
ADAPTIVE_DAMPING_MAX = 0.5
ADAPTIVE_DAMPING_THRESHOLD = 0.2

# Null-space posture control: pulls every controlled DOF toward its own
# ready-pose entry, projected through the null-space projector so it never
# disturbs the primary 6D task -- only the Franka's redundant 7th DOF has
# any null space to move in.
NULL_SPACE_STIFFNESS = 2.0
NULL_SPACE_DAMPING = 0.05


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


class Example:
    @staticmethod
    def create_parser():
        return newton.examples.create_parser()

    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.device = wp.get_device()

        # ---- Physics scene ---------------------------------------------------
        franka_urdf_path = str(newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf")
        ur10_asset_file = str(newton.utils.download_asset("universal_robots_ur10") / "usd/ur10_instanceable.usda")
        builder = newton.ModelBuilder()

        franka_joints, franka_tool_body, franka_tool_site_transform = self._add_franka(
            builder, franka_urdf_path, FRANKA_BASE_POSITION
        )
        ur10_joints, ur10_tool_body, ur10_tool_site_transform = self._add_ur10(
            builder, ur10_asset_file, UR10_BASE_POSITION
        )
        self._franka_joints = franka_joints
        self._ur10_joints = ur10_joints

        builder.add_ground_plane()

        # Every arm DOF tracks joint_target_q/qd via MuJoCo's implicit PD;
        # this includes the Franka's finger DOFs, so they hold their
        # builder-set home target even though the controller never writes
        # to them.
        for i in range(builder.joint_dof_count):
            builder.joint_target_ke[i] = JOINT_TARGET_KE
            builder.joint_target_kd[i] = JOINT_TARGET_KD
            builder.joint_target_mode[i] = int(JointTargetMode.POSITION_VELOCITY)

        self.model = builder.finalize(device=self.device)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Contacts play no role in this tracking demo.
        self.solver = newton.solvers.SolverMuJoCo(self.model, disable_contacts=True)

        # ---- Differential-kinematics controller -------------------------------
        # One controller call handles both robots; joints lists robot 0's
        # (Franka's) controlled joints first, then robot 1's (UR10's),
        # matching desired_tool_pose_world's per-robot ordering below.
        self.controller = ControllerDiffIK(
            self.model,
            joints=franka_joints + ur10_joints,
            tool_sites="tool_site",
            bandwidth=BANDWIDTH,
            damping=None,
            ik_method=IkMethod.ADAPTIVE_DAMPING,
            adaptive_damping_min=ADAPTIVE_DAMPING_MIN,
            adaptive_damping_max=ADAPTIVE_DAMPING_MAX,
            adaptive_damping_threshold=ADAPTIVE_DAMPING_THRESHOLD,
            use_null_space_posture_control=True,
            null_space_stiffness=NULL_SPACE_STIFFNESS,
            null_space_damping=NULL_SPACE_DAMPING,
        )

        self._input = self.controller.input()
        self._output = self.controller.output()
        # Constant across every step -- the posture target is always each
        # robot's own ready pose, so this is assigned once rather than
        # reassigned in step().
        self._input.q_des_null.assign(np.array(FRANKA_READY_POSE + UR10_READY_POSE, dtype=np.float32))
        # The controller's outputs are compact (one entry per controlled
        # DOF); indexed views scatter them straight into the sim control
        # buffers, in each buffer's own layout (q_start: coordinate space,
        # qd_start: DOF space).
        self._output.joint_q_target = self.control.joint_target_q[self.controller.q_start]
        self._output.joint_qd_target = self.control.joint_target_qd[self.controller.qd_start]

        # Draggable gizmo per robot, seeded at each tool's actual starting
        # world pose -- zero initial error, rather than a sudden snap at
        # startup. Mutated in place by the viewer each render() call.
        body_q_np = self.state_0.body_q.numpy()
        self.gizmo_tfs = [
            wp.transform(*body_q_np[franka_tool_body].tolist()) * franka_tool_site_transform,
            wp.transform(*body_q_np[ur10_tool_body].tolist()) * ur10_tool_site_transform,
        ]

        # Pulled back and to the side so both robots (Franka at y=0, UR10 at
        # y=1.8) are in view together.
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(-2.1, 0.9, 1.8), pitch=-15.0, yaw=15.0)
            if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(wp.vec3(0.4, 0.9, 0.4))

        self.viewer.set_model(self.model)

        # No CUDA graph capture: the gizmo drag updates desired_tool_pose_world
        # from Python each frame, which isn't graph-capturable.

    @staticmethod
    def _add_franka(builder, urdf_path, base_position):
        """Load one Franka at base_position, set its ready pose, and add a tool site at its TCP.

        Returns:
            Tuple of (arm joint indices, fr3_hand_tcp body index, tool
            site's body-local transform).
        """
        joint_count_before = builder.joint_count
        coord_count_before = builder.joint_coord_count
        body_count_before = builder.body_count
        builder.add_urdf(urdf_path, xform=wp.transform(base_position, wp.quat_identity()), floating=False)

        # fr3_joint1..7 are the first 7 non-fixed joints after the (fixed,
        # 0-coordinate) base/mount joints this URDF starts with; the finger
        # joints follow. Indices are relative to this call since add_urdf
        # appends them.
        arm_joints = [joint_count_before + 2 + i for i in range(FRANKA_ARM_DOFS)]
        arm_coords = list(range(coord_count_before, coord_count_before + FRANKA_ARM_DOFS))
        for coord, angle in zip(arm_coords, FRANKA_READY_POSE, strict=True):
            builder.joint_q[coord] = angle

        # Body 11 (0-based) this URDF adds is fr3_hand_tcp, the fixed frame
        # between the fingers -- the tool site sits there directly, with no
        # offset.
        tool_body = body_count_before + 11
        tool_site_transform = wp.transform_identity()
        builder.add_site(tool_body, xform=tool_site_transform, label="tool_site", visible=True, scale=TOOL_SITE_SCALE)

        return arm_joints, tool_body, tool_site_transform

    @staticmethod
    def _add_ur10(builder, asset_file, base_position):
        """Load one UR10 at base_position, set a ready pose, and add a tool site at its wrist flange.

        Returns:
            Tuple of (arm joint indices, tool body index, tool site's
            body-local transform).
        """
        joint_count_before = builder.joint_count
        coord_count_before = builder.joint_coord_count
        body_count_before = builder.body_count
        builder.add_usd(
            asset_file,
            xform=wp.transform(base_position, wp.quat_identity()),
            floating=False,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )

        # shoulder_pan..wrist_3 are the 6 non-fixed joints after the (fixed,
        # 0-coordinate) base mount joint this asset starts with; the fixed
        # ee_joint follows. Indices are relative to this call since add_usd
        # appends them.
        arm_joints = [joint_count_before + 1 + i for i in range(UR10_ARM_DOFS)]
        arm_coords = list(range(coord_count_before, coord_count_before + UR10_ARM_DOFS))
        for coord, angle in zip(arm_coords, UR10_READY_POSE, strict=True):
            builder.joint_q[coord] = angle

        # Body 7 (0-based) this asset adds is ee_link, the fixed wrist-flange
        # frame -- the tool site sits there directly, with no offset.
        tool_body = body_count_before + 7
        tool_site_transform = wp.transform_identity()
        builder.add_site(tool_body, xform=tool_site_transform, label="tool_site", visible=True, scale=TOOL_SITE_SCALE)

        return arm_joints, tool_body, tool_site_transform

    def step(self):
        # Gizmo drag updates are Python-side, so this whole frame is outside
        # any CUDA graph. Rebind joint_q/joint_qd to whichever State buffer
        # the substep swap left at state_0 after the previous frame.
        self._input.joint_q = self.state_0.joint_q
        self._input.joint_qd = self.state_0.joint_qd
        pose = np.zeros((2, 7), dtype=np.float32)
        for i, tf in enumerate(self.gizmo_tfs):
            pose[i, :3] = wp.transform_get_translation(tf)
            pose[i, 3:] = wp.transform_get_rotation(tf)
        self._input.desired_tool_pose_world.assign(pose)

        # One controller step per frame: joint_q_target/joint_qd_target are
        # one frame ahead, tracked by MuJoCo's own PD across every substep
        # below -- refreshing the target every substep instead would chase
        # the drifted current_q with no restoring signal left against
        # gravity.
        self.controller.step(inputs=self._input, outputs=self._output, dt=self.frame_dt)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        # The controller's own resolved tool pose, as of last step()'s FK --
        # dragging the gizmo away and releasing it snaps it back to
        # wherever the tool actually is.
        tool_pose_world = self.controller.tool_pose_world.numpy()
        for i, tf in enumerate(self.gizmo_tfs):
            self.viewer.log_gizmo(f"target_{i}", tf, snap_to=wp.transform(*tool_pose_world[i].tolist()))
        self.viewer.end_frame()

    def test_final(self):
        """Gizmos aren't dragged in headless test mode, so both arms should stay near their ready pose."""
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        assert np.all(np.isfinite(joint_q)), f"joint_q has NaN/Inf: {joint_q}"
        assert np.all(np.isfinite(joint_qd)), f"joint_qd has NaN/Inf: {joint_qd}"

        franka_q = joint_q[:FRANKA_ARM_DOFS]
        franka_ready_q = np.array(FRANKA_READY_POSE, dtype=np.float32)
        assert np.all(np.abs(franka_q - franka_ready_q) < 0.2), (
            f"Franka arm joints drifted from its ready pose: {franka_q}"
        )

        ur10_q_start = self.model.joint_q_start.numpy()[self._ur10_joints[0]]
        ur10_q = joint_q[ur10_q_start : ur10_q_start + UR10_ARM_DOFS]
        ur10_ready_q = np.array(UR10_READY_POSE, dtype=np.float32)
        assert np.all(np.abs(ur10_q - ur10_ready_q) < 0.2), f"UR10 arm joints drifted from its ready pose: {ur10_q}"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
