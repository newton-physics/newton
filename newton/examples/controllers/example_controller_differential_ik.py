# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Controllers — Differential IK
#
# Demonstrates ControllerDifferentialIK on three real, heterogeneous robots at once
# -- a 7-DOF Franka Panda arm tasking the full 6D pose (redundant by 1 DOF),
# a 6-DOF UR10 arm tasking position only (redundant by 3 DOFs), and a 4-DOF
# planar arm restricted to a 3D task (X, Y, yaw) via axis_weight (redundant
# by 1 DOF) -- each independently tracking its own draggable gizmo target.
# One controller call handles all three, each robot resolved through its own
# tool site and Jacobian.
#
# Kinematics only: the controller's joint targets are applied directly to
# the sim state each frame (no physics solver), keeping the demo focused on
# the IK itself.
#
# Every robot here is redundant against its own task, so null-space posture
# control continuously pulls each toward its own ready pose, so none of them
# drifts toward a bad internal configuration with nothing to anchor it.
#
# Each robot's gizmo only exposes the axes its own axis_weight keeps active
# -- via log_gizmo's own translate/rotate axis selection -- so the widget
# itself can't suggest a motion the controller would ignore.
#
# --ik-method picks the inverse-Jacobian solve (dls, pinv, transpose,
# adaptive_damping, truncated_svd) -- see the create_controller_from_*_method
# functions below, one per DifferentialIKMethod.
#
# Command: python -m newton.examples controller_differential_ik --ik-method adaptive_damping
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import Axis
from newton.controllers import ControllerDifferentialIK, DifferentialIKMethod

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
UR10_ARM_DOFS = len(UR10_READY_POSE)  # 6; redundant by 3 DOFs against its position-only 3D task
UR10_BASE_POSITION = wp.vec3(0.0, 1.8, 0.0)  # separated from the Franka along Y

# A 4R planar arm: every joint rotates about world Z, so the tool stays at a
# fixed height and every reachable pose has zero roll/pitch -- exactly the 3
# axes (X, Y, yaw) axis_weight keeps active for it below. Mounted above the
# ground plane so the horizontal arm has room to swing without intersecting it.
PLANAR_LINK_LENGTH = 0.25
PLANAR_READY_POSE = [0.4, 1.6, -1.4, 1.0]
PLANAR_ARM_DOFS = len(PLANAR_READY_POSE)
PLANAR_BASE_POSITION = wp.vec3(0.0, 3.6, 0.5)  # separated from the UR10 along Y

TOOL_SITE_SCALE = (0.02, 0.02, 0.02)

# Franka tasks the full 6D pose; UR10 position only; the planar arm only
# X, Y, and yaw -- each excluded axis is structurally dropped from its own
# solve, not merely weighted toward zero.
FULL_POSE_AXIS_WEIGHT = wp.spatial_vector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
POSITION_ONLY_AXIS_WEIGHT = wp.spatial_vector(1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
PLANAR_AXIS_WEIGHT = wp.spatial_vector(1.0, 1.0, 0.0, 0.0, 0.0, 1.0)

_XYZ_AXES = (Axis.X, Axis.Y, Axis.Z)


def _gizmo_axes_from_weight(axis_weight):
    """log_gizmo's translate/rotate axis lists for a robot's own axis_weight, one gizmo handle per active axis."""
    return {
        "translate": [axis for i, axis in enumerate(_XYZ_AXES) if axis_weight[i] > 0.0],
        "rotate": [axis for i, axis in enumerate(_XYZ_AXES) if axis_weight[3 + i] > 0.0],
    }


# ---------------------------------------------------------------------------
# One constructor per DifferentialIKMethod, each fully self-contained --
# every parameter that method needs (or forbids) lives right here, so
# picking a method from the command line is just picking which function
# runs, and each one is a complete, independent example of that method's
# own ControllerDifferentialIK arguments.
# ---------------------------------------------------------------------------


def create_controller_from_dls_method(model, joints, axis_weight, device):
    """DifferentialIKMethod.DAMPED_LEAST_SQUARES: a single fixed damping λ everywhere."""
    return ControllerDifferentialIK(
        model,
        joints=joints,
        tool_sites="tool_site",
        axis_weight=axis_weight,
        bandwidth=20.0,
        damping=0.1,
        ik_method=DifferentialIKMethod.DAMPED_LEAST_SQUARES,
        use_null_space_posture_control=True,
        null_space_stiffness=2.0,
        null_space_damping=0.05,
    )


def create_controller_from_pinv_method(model, joints, axis_weight, device):
    """DifferentialIKMethod.PSEUDO_INVERSE: exact (λ=0) Moore-Penrose pseudo-inverse, no damping."""
    return ControllerDifferentialIK(
        model,
        joints=joints,
        tool_sites="tool_site",
        axis_weight=axis_weight,
        bandwidth=20.0,
        damping=None,
        ik_method=DifferentialIKMethod.PSEUDO_INVERSE,
        use_null_space_posture_control=True,
        null_space_stiffness=2.0,
        null_space_damping=0.05,
    )


def create_controller_from_transpose_method(model, joints, axis_weight, device):
    """DifferentialIKMethod.TRANSPOSE: qd = bandwidth * Jᵀe, no matrix inversion at all.

    Unlike the inverting methods, there's no damping to keep this loop
    stable at a high gain, so bandwidth has to stay small (well below
    ``1/frame_dt``) or the discrete-time position update overshoots and
    diverges.
    """
    return ControllerDifferentialIK(
        model,
        joints=joints,
        tool_sites="tool_site",
        axis_weight=axis_weight,
        bandwidth=5.0,
        damping=None,
        ik_method=DifferentialIKMethod.TRANSPOSE,
        use_null_space_posture_control=True,
        null_space_stiffness=2.0,
        null_space_damping=0.05,
    )


def create_controller_from_adaptive_damping_method(model, joints, axis_weight, device):
    """DifferentialIKMethod.ADAPTIVE_DAMPING: λ ramps up automatically near a singularity or reach limit."""
    return ControllerDifferentialIK(
        model,
        joints=joints,
        tool_sites="tool_site",
        axis_weight=axis_weight,
        bandwidth=20.0,
        damping=None,
        ik_method=DifferentialIKMethod.ADAPTIVE_DAMPING,
        adaptive_damping_min=1e-2,
        adaptive_damping_max=0.5,
        adaptive_damping_threshold=0.05,
        use_null_space_posture_control=True,
        null_space_stiffness=2.0,
        null_space_damping=0.05,
    )


def create_controller_from_truncated_svd_method(model, joints, axis_weight, device):
    """DifferentialIKMethod.TRUNCATED_SVD: directions below the threshold are dropped, not damped."""
    return ControllerDifferentialIK(
        model,
        joints=joints,
        tool_sites="tool_site",
        axis_weight=axis_weight,
        bandwidth=20.0,
        damping=None,
        ik_method=DifferentialIKMethod.TRUNCATED_SVD,
        truncated_svd_threshold=0.1,
        use_null_space_posture_control=True,
        null_space_stiffness=2.0,
        null_space_damping=0.05,
    )


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


class Example:
    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--ik-method",
            type=str,
            default="adaptive_damping",
            choices=["dls", "pinv", "transpose", "adaptive_damping", "truncated_svd"],
            help="Inverse-Jacobian solve method, a DifferentialIKMethod.",
        )
        return parser

    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.viewer = viewer
        self.device = wp.get_device()

        # ---- Scene -------------------------------------------------------
        franka_urdf_path = str(newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf")
        ur10_asset_file = str(newton.utils.download_asset("universal_robots_ur10") / "usd/ur10_instanceable.usda")
        builder = newton.ModelBuilder()

        franka_joints, franka_tool_body, franka_tool_site_transform = self._add_franka(
            builder, franka_urdf_path, FRANKA_BASE_POSITION
        )
        ur10_joints, ur10_tool_body, ur10_tool_site_transform = self._add_ur10(
            builder, ur10_asset_file, UR10_BASE_POSITION
        )
        planar_joints, planar_tool_body, planar_tool_site_transform = self._add_planar_arm(
            builder, PLANAR_BASE_POSITION
        )
        self._franka_joints = franka_joints
        self._ur10_joints = ur10_joints
        self._planar_joints = planar_joints

        builder.add_ground_plane()

        self.model = builder.finalize(device=self.device)
        self.state_0 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # ---- Differential-kinematics controller -------------------------------
        # One controller call handles all three robots; joints lists robot
        # 0's (Franka's) controlled joints first, then robot 1's (UR10's),
        # then robot 2's (the planar arm's), matching axis_weight's and
        # desired_tool_pose_world's per-robot ordering below.
        joints = franka_joints + ur10_joints + planar_joints
        axis_weight_rows = [FULL_POSE_AXIS_WEIGHT, POSITION_ONLY_AXIS_WEIGHT, PLANAR_AXIS_WEIGHT]
        axis_weight = wp.array(axis_weight_rows, dtype=wp.spatial_vector, device=self.device)
        ik_method = args.ik_method
        if ik_method == "dls":
            self.controller = create_controller_from_dls_method(self.model, joints, axis_weight, self.device)
        elif ik_method == "pinv":
            self.controller = create_controller_from_pinv_method(self.model, joints, axis_weight, self.device)
        elif ik_method == "transpose":
            self.controller = create_controller_from_transpose_method(self.model, joints, axis_weight, self.device)
        elif ik_method == "adaptive_damping":
            self.controller = create_controller_from_adaptive_damping_method(
                self.model, joints, axis_weight, self.device
            )
        elif ik_method == "truncated_svd":
            self.controller = create_controller_from_truncated_svd_method(self.model, joints, axis_weight, self.device)
        else:
            raise ValueError(f"Unknown --ik-method: {ik_method}")

        self._input = self.controller.input()
        self._output = self.controller.output()
        # Constant across every step -- the posture target is always each
        # robot's own ready pose, so this is assigned once rather than
        # reassigned in step().
        self._input.q_des_null.assign(
            np.array(FRANKA_READY_POSE + UR10_READY_POSE + PLANAR_READY_POSE, dtype=np.float32)
        )
        # The controller's outputs are compact (one entry per controlled
        # DOF); indexed views scatter them straight into the sim state, in
        # each buffer's own layout (q_start: coordinate space, qd_start:
        # DOF space).
        self._output.joint_q_target = self.state_0.joint_q[self.controller.q_start]
        self._output.joint_qd_target = self.state_0.joint_qd[self.controller.qd_start]

        # Draggable gizmo per robot, seeded at each tool's actual starting
        # world pose -- zero initial error, rather than a sudden snap at
        # startup. Mutated in place by the viewer each render() call.
        body_q_np = self.state_0.body_q.numpy()
        self.gizmo_tfs = [
            wp.transform(*body_q_np[franka_tool_body].tolist()) * franka_tool_site_transform,
            wp.transform(*body_q_np[ur10_tool_body].tolist()) * ur10_tool_site_transform,
            wp.transform(*body_q_np[planar_tool_body].tolist()) * planar_tool_site_transform,
        ]
        # A zero-weighted axis is excluded from that robot's solve entirely
        # (see axis_weight above), so its gizmo handle is dropped too -- the
        # widget can't suggest a motion the controller would ignore.
        self.gizmo_axes = [_gizmo_axes_from_weight(weight) for weight in axis_weight_rows]

        # Set such that Franka at y=0, UR10 at y=1.8, planar arm at y=3.6 are in view together.
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(3.2, -1.8, 2.4), pitch=-20.0, yaw=15.0)
            if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(wp.vec3(0.4, 1.8, 0.4))

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

    @staticmethod
    def _add_planar_arm(builder, base_position):
        """Build a 4R planar arm at base_position and add a tool site at its tip.

        Every joint rotates about world Z; successive links chain along
        local +X. Returns:
            Tuple of (arm joint indices, last link's body index, tool
            site's body-local transform).
        """
        coord_count_before = builder.joint_coord_count

        # A capsule extends along its own local Z axis by default; this
        # rotation aligns that with the link's own local +X, the direction
        # successive links chain along below.
        capsule_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2.0)

        arm_joints = []
        parent = -1
        parent_xform = wp.transform(base_position, wp.quat_identity())
        link = -1
        for _ in range(PLANAR_ARM_DOFS):
            link = builder.add_link()
            joint = builder.add_joint_revolute(
                parent=parent,
                child=link,
                axis=wp.vec3(0.0, 0.0, 1.0),
                parent_xform=parent_xform,
                child_xform=wp.transform_identity(),
            )
            builder.add_shape_capsule(
                link,
                xform=wp.transform(wp.vec3(PLANAR_LINK_LENGTH / 2.0, 0.0, 0.0), capsule_rotation),
                radius=0.02,
                half_height=PLANAR_LINK_LENGTH / 2.0,
            )
            arm_joints.append(joint)
            parent = link
            parent_xform = wp.transform(wp.vec3(PLANAR_LINK_LENGTH, 0.0, 0.0), wp.quat_identity())
        builder.add_articulation(arm_joints, label="planar_arm")

        arm_coords = list(range(coord_count_before, coord_count_before + PLANAR_ARM_DOFS))
        for coord, angle in zip(arm_coords, PLANAR_READY_POSE, strict=True):
            builder.joint_q[coord] = angle

        # The tool site sits at the last link's tip, one more link length
        # out along its own local +X.
        tool_body = link
        tool_site_transform = wp.transform(wp.vec3(PLANAR_LINK_LENGTH, 0.0, 0.0), wp.quat_identity())
        builder.add_site(tool_body, xform=tool_site_transform, label="tool_site", visible=True, scale=TOOL_SITE_SCALE)

        return arm_joints, tool_body, tool_site_transform

    def step(self):
        # Gizmo drag updates are Python-side, so this whole frame is outside
        # any CUDA graph.
        self._input.joint_q = self.state_0.joint_q
        self._input.joint_qd = self.state_0.joint_qd
        pose = np.zeros((len(self.gizmo_tfs), 7), dtype=np.float32)
        for i, tf in enumerate(self.gizmo_tfs):
            pose[i, :3] = wp.transform_get_translation(tf)
            pose[i, 3:] = wp.transform_get_rotation(tf)
        self._input.desired_tool_pose_world.assign(pose)

        # joint_q_target/joint_qd_target write straight into state_0 (see
        # the output bindings in __init__); eval_fk brings body_q/body_qd
        # back in sync for rendering and the next frame's Jacobian.
        self.controller.step(inputs=self._input, outputs=self._output, dt=self.frame_dt)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        # The controller's own resolved tool pose, as of last step()'s FK --
        # dragging the gizmo away and releasing it snaps it back to
        # wherever the tool actually is.
        tool_pose_world = self.controller.tool_pose_world.numpy()
        for i, tf in enumerate(self.gizmo_tfs):
            self.viewer.log_gizmo(
                f"target_{i}",
                tf,
                translate=self.gizmo_axes[i]["translate"],
                rotate=self.gizmo_axes[i]["rotate"],
                snap_to=wp.transform(*tool_pose_world[i].tolist()),
            )
        self.viewer.end_frame()

    def test_final(self):
        """Gizmos aren't dragged in headless test mode, so all three arms should stay near their ready pose."""
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

        planar_q_start = self.model.joint_q_start.numpy()[self._planar_joints[0]]
        planar_q = joint_q[planar_q_start : planar_q_start + PLANAR_ARM_DOFS]
        planar_ready_q = np.array(PLANAR_READY_POSE, dtype=np.float32)
        assert np.all(np.abs(planar_q - planar_ready_q) < 0.2), (
            f"Planar arm joints drifted from its ready pose: {planar_q}"
        )


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
