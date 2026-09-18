# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Controllers — Differential IK, Methods x Multi-Frame
#
# Combines example_controller_differential_ik.py's --ik-method selection
# (one create_controller_from_*_method function per DifferentialIKMethod)
# with heterogeneous frame counts: robot A has a single tool frame, robot B
# has two. One ControllerDifferentialIK call handles both, each robot's own
# frames_per_robot inferred from how many sites tool_sites matches on it.
#
# Robot A: a redundant 7-DOF arm tracking full 6D pose on one frame.
# Robot B: a Y-shaped, shared-pan arm tracking position on two frames (its
# tips) at once, stacked into one combined task.
#
# Kinematics only: joint targets are written directly into the sim state,
# no physics solver. Every robot is redundant against its own task, so
# null-space posture control pulls each back toward its own ready pose.
#
# --ik-method picks the inverse-Jacobian solve (dls, pinv, transpose,
# adaptive_damping, truncated_svd) -- see the create_controller_from_*_method
# functions below, one per DifferentialIKMethod.
#
# Command: python -m newton.examples controller_differential_ik_methods_multiframe --ik-method dls
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import Axis
from newton.controllers import ControllerDifferentialIK, DifferentialIKMethod

# ---------------------------------------------------------------------------
# Robot configuration
# ---------------------------------------------------------------------------

# Robot A: single tool frame, 7-DOF, alternating Z/Y axes -- redundant by 1
# against its own full 6D pose task.
ROBOT_A_LINK_LENGTH = 0.15
ROBOT_A_AXES = [Axis.Z, Axis.Y, Axis.Z, Axis.Y, Axis.Z, Axis.Y, Axis.Z]
ROBOT_A_READY_POSE = [0.0, 0.6, 0.0, -1.0, 0.0, 0.6, 0.0]
ROBOT_A_BASE_POSITION = wp.vec3(0.0, 1.8, 0.9)

# Robot B: two tool frames (its left/right tips), sharing a pan joint. Each
# arm is 3-DOF (yaw, then two bends) -- redundant by 1 against the 6D
# combined (position-only x2) task across both frames.
ARM_LINK_LENGTH = 0.3
ARM_SIDE_OFFSET = 0.3
ROBOT_B_ARM_AXES = [Axis.Z, Axis.Y, Axis.Y]
ROBOT_B_READY_POSE = [0.0, 0.5, -0.8, 0.3, 0.5, -0.8, 0.3]  # pan, left arm x3, right arm x3
ROBOT_B_BASE_POSITION = wp.vec3(0.0, -1.8, 0.6)

TOOL_SITE_SCALE = (0.02, 0.02, 0.02)
FULL_POSE_AXIS_WEIGHT = wp.spatial_vector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
POSITION_ONLY_AXIS_WEIGHT = wp.spatial_vector(1.0, 1.0, 1.0, 0.0, 0.0, 0.0)

_AXIS_VEC = {Axis.X: wp.vec3(1.0, 0.0, 0.0), Axis.Y: wp.vec3(0.0, 1.0, 0.0), Axis.Z: wp.vec3(0.0, 0.0, 1.0)}
_XYZ_AXES = (Axis.X, Axis.Y, Axis.Z)
# A capsule extends along its own local Z by default; this aligns that with
# local +X, the direction robot B's arm segments chain along (robot A's own
# segments already chain along local Z, so need no extra rotation).
_CAPSULE_ROTATION_X = wp.quat_from_axis_angle(_AXIS_VEC[Axis.Y], np.pi / 2.0)


def _gizmo_axes_from_weight(axis_weight):
    """log_gizmo's translate/rotate axis lists for a frame's own axis_weight, one gizmo handle per active axis."""
    return {
        "translate": [axis for i, axis in enumerate(_XYZ_AXES) if axis_weight[i] > 0.0],
        "rotate": [axis for i, axis in enumerate(_XYZ_AXES) if axis_weight[3 + i] > 0.0],
    }


# ---------------------------------------------------------------------------
# One constructor per DifferentialIKMethod -- see
# example_controller_differential_ik.py for the same pattern with a
# single-frame-per-robot fleet.
# ---------------------------------------------------------------------------


def create_controller_from_dls_method(model, tool_sites, axis_weight):
    """DifferentialIKMethod.DAMPED_LEAST_SQUARES: a single fixed damping λ everywhere."""
    return ControllerDifferentialIK(
        model,
        tool_sites=tool_sites,
        axis_weight=axis_weight,
        bandwidth=15.0,
        damping=0.1,
        ik_method=DifferentialIKMethod.DAMPED_LEAST_SQUARES,
        use_null_space_posture_control=True,
        null_space_stiffness=2.0,
        null_space_damping=0.05,
    )


def create_controller_from_pinv_method(model, tool_sites, axis_weight):
    """DifferentialIKMethod.PSEUDO_INVERSE: exact (λ=0) Moore-Penrose pseudo-inverse, no damping."""
    return ControllerDifferentialIK(
        model,
        tool_sites=tool_sites,
        axis_weight=axis_weight,
        bandwidth=15.0,
        damping=None,
        ik_method=DifferentialIKMethod.PSEUDO_INVERSE,
        use_null_space_posture_control=True,
        null_space_stiffness=2.0,
        null_space_damping=0.05,
    )


def create_controller_from_transpose_method(model, tool_sites, axis_weight):
    """DifferentialIKMethod.TRANSPOSE: qd = bandwidth * Jᵀe, no matrix inversion at all.

    Unlike the inverting methods, there's no damping to keep this loop
    stable at a high gain, so bandwidth has to stay small (well below
    ``1/frame_dt``) or the discrete-time position update overshoots and
    diverges.
    """
    return ControllerDifferentialIK(
        model,
        tool_sites=tool_sites,
        axis_weight=axis_weight,
        bandwidth=5.0,
        damping=None,
        ik_method=DifferentialIKMethod.TRANSPOSE,
        use_null_space_posture_control=True,
        null_space_stiffness=2.0,
        null_space_damping=0.05,
    )


def create_controller_from_adaptive_damping_method(model, tool_sites, axis_weight):
    """DifferentialIKMethod.ADAPTIVE_DAMPING: λ ramps up automatically near a singularity or reach limit."""
    return ControllerDifferentialIK(
        model,
        tool_sites=tool_sites,
        axis_weight=axis_weight,
        bandwidth=15.0,
        damping=None,
        ik_method=DifferentialIKMethod.ADAPTIVE_DAMPING,
        adaptive_damping_min=1e-2,
        adaptive_damping_max=0.5,
        adaptive_damping_threshold=0.05,
        use_null_space_posture_control=True,
        null_space_stiffness=2.0,
        null_space_damping=0.05,
    )


def create_controller_from_truncated_svd_method(model, tool_sites, axis_weight):
    """DifferentialIKMethod.TRUNCATED_SVD: directions below the threshold are dropped, not damped."""
    return ControllerDifferentialIK(
        model,
        tool_sites=tool_sites,
        axis_weight=axis_weight,
        bandwidth=15.0,
        damping=None,
        ik_method=DifferentialIKMethod.TRUNCATED_SVD,
        truncated_svd_threshold=0.1,
        use_null_space_posture_control=True,
        null_space_stiffness=2.0,
        null_space_damping=0.05,
    )


_CONTROLLER_FACTORIES = {
    "dls": create_controller_from_dls_method,
    "pinv": create_controller_from_pinv_method,
    "transpose": create_controller_from_transpose_method,
    "adaptive_damping": create_controller_from_adaptive_damping_method,
    "truncated_svd": create_controller_from_truncated_svd_method,
}


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
            choices=list(_CONTROLLER_FACTORIES.keys()),
            help="Inverse-Jacobian solve method, a DifferentialIKMethod.",
        )
        return parser

    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.viewer = viewer
        self.device = wp.get_device()

        # ---- Scene: robot A (1 frame) and robot B (2 frames) -------------
        builder = newton.ModelBuilder()
        _, tip_a, tf_a = self._add_seven_dof_arm(builder)
        _, left_tip_b, left_tf_b, right_tip_b, right_tf_b = self._add_dual_arm_robot(builder)
        for coord, angle in enumerate(ROBOT_A_READY_POSE + ROBOT_B_READY_POSE):
            builder.joint_q[coord] = angle

        builder.add_ground_plane()
        self.model = builder.finalize(device=self.device)
        self.state_0 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # ---- Differential-kinematics controller, heterogeneous frame counts
        # frames_per_robot is inferred per robot from how many sites
        # tool_sites matches on it (1 for A, 2 for B) -- there's no
        # separate argument for it.
        tool_sites = ["tool_a", "tool_b_left", "tool_b_right"]
        axis_weight = wp.array(
            [FULL_POSE_AXIS_WEIGHT, POSITION_ONLY_AXIS_WEIGHT, POSITION_ONLY_AXIS_WEIGHT],
            dtype=wp.spatial_vector,
            device=self.device,
        )
        self.controller = _CONTROLLER_FACTORIES[args.ik_method](self.model, tool_sites, axis_weight)

        self._input = self.controller.input()
        self._output = self.controller.output()
        self._input.joint_q = self.state_0.joint_q
        self._input.joint_qd = self.state_0.joint_qd
        self._input.q_des_null.assign(np.array(ROBOT_A_READY_POSE + ROBOT_B_READY_POSE, dtype=np.float32))
        self._output.joint_q_target = self.state_0.joint_q[self.controller.q_start]
        self._output.joint_qd_target = self.state_0.joint_qd[self.controller.qd_start]

        # One draggable gizmo per frame, seeded at each site's actual
        # starting world pose. tool_sites resolves each robot's frames in
        # its own entry order, so this list matches tool_sites above exactly.
        body_q_np = self.state_0.body_q.numpy()
        self.gizmo_tfs = [
            wp.transform(*body_q_np[tip_a].tolist()) * tf_a,
            wp.transform(*body_q_np[left_tip_b].tolist()) * left_tf_b,
            wp.transform(*body_q_np[right_tip_b].tolist()) * right_tf_b,
        ]
        # A zero-weighted axis is excluded from that frame's solve entirely,
        # so its gizmo handle is dropped too -- the widget can't suggest a
        # motion the controller would ignore.
        self.gizmo_axes = [
            _gizmo_axes_from_weight(FULL_POSE_AXIS_WEIGHT),
            _gizmo_axes_from_weight(POSITION_ONLY_AXIS_WEIGHT),
            _gizmo_axes_from_weight(POSITION_ONLY_AXIS_WEIGHT),
        ]

        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(2.2, 0.0, 1.4), pitch=-15.0, yaw=180.0)

        self.viewer.set_model(self.model)

        self.graph = None
        if self.controller.is_graphable() and self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate()
            self.graph = capture.graph

    @staticmethod
    def _add_seven_dof_arm(builder):
        """Build robot A: a 7-DOF chain, alternating Z/Y axes, with a tool site at its tip.

        Returns:
            Tuple of (joint indices, tip body index, tool site's body-local
            transform).
        """
        joints = []
        link = -1
        parent_xform = wp.transform(ROBOT_A_BASE_POSITION, wp.quat_identity())
        for axis in ROBOT_A_AXES:
            next_link = builder.add_link()
            joint = builder.add_joint_revolute(
                parent=link,
                child=next_link,
                axis=_AXIS_VEC[axis],
                parent_xform=parent_xform,
                child_xform=wp.transform_identity(),
            )
            builder.add_shape_capsule(
                next_link,
                xform=wp.transform(wp.vec3(0.0, 0.0, ROBOT_A_LINK_LENGTH / 2.0), wp.quat_identity()),
                radius=0.025,
                half_height=ROBOT_A_LINK_LENGTH / 2.0,
            )
            joints.append(joint)
            link = next_link
            parent_xform = wp.transform(wp.vec3(0.0, 0.0, ROBOT_A_LINK_LENGTH), wp.quat_identity())
        builder.add_articulation(joints, label="robot_a")
        tool_transform = wp.transform(wp.vec3(0.0, 0.0, ROBOT_A_LINK_LENGTH), wp.quat_identity())
        builder.add_site(link, xform=tool_transform, label="tool_a", visible=True, scale=TOOL_SITE_SCALE)
        return joints, link, tool_transform

    @staticmethod
    def _add_dual_arm_robot(builder):
        """Build robot B: a shared pan joint splitting into a left/right 3-DOF sub-arm, each with a tool site.

        Returns:
            Tuple of (joint indices, left tip body, left tool transform,
            right tip body, right tool transform).
        """
        pan = builder.add_link()
        joint_pan = builder.add_joint_revolute(
            parent=-1,
            child=pan,
            axis=_AXIS_VEC[Axis.Z],
            parent_xform=wp.transform(ROBOT_B_BASE_POSITION, wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )
        builder.add_shape_capsule(
            pan,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_from_axis_angle(_AXIS_VEC[Axis.X], -np.pi / 2.0)),
            radius=0.02,
            half_height=ARM_SIDE_OFFSET,
        )
        left_joints, left_tip, left_tf = Example._add_arm(
            builder, parent=pan, side_offset=wp.vec3(0.0, ARM_SIDE_OFFSET, 0.0), side_label="b_left"
        )
        right_joints, right_tip, right_tf = Example._add_arm(
            builder, parent=pan, side_offset=wp.vec3(0.0, -ARM_SIDE_OFFSET, 0.0), side_label="b_right"
        )
        joints = [joint_pan, *left_joints, *right_joints]
        builder.add_articulation(joints, label="robot_b")
        return joints, left_tip, left_tf, right_tip, right_tf

    @staticmethod
    def _add_arm(builder, *, parent, side_offset, side_label):
        """Add a 3-DOF sub-arm (``ROBOT_B_ARM_AXES``) hanging off ``parent``, with a tool site at its tip.

        Returns:
            Tuple of (this sub-arm's own joint indices, tip body index,
            tool site's body-local transform).
        """
        joints = []
        link = parent
        parent_xform = wp.transform(side_offset, wp.quat_identity())
        for axis in ROBOT_B_ARM_AXES:
            next_link = builder.add_link()
            joint = builder.add_joint_revolute(
                parent=link,
                child=next_link,
                axis=_AXIS_VEC[axis],
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
        tool_transform = wp.transform(wp.vec3(ARM_LINK_LENGTH, 0.0, 0.0), wp.quat_identity())
        builder.add_site(link, xform=tool_transform, label=f"tool_{side_label}", visible=True, scale=TOOL_SITE_SCALE)
        return joints, link, tool_transform

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
                translate=self.gizmo_axes[i]["translate"],
                rotate=self.gizmo_axes[i]["rotate"],
                snap_to=wp.transform(*tool_pose_world[i].tolist()),
            )
        self.viewer.end_frame()

    def test_final(self):
        """Verify both robots stay near their ready pose, since gizmos aren't dragged in headless test mode."""
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        assert np.all(np.isfinite(joint_q)), f"joint_q has NaN/Inf: {joint_q}"
        assert np.all(np.isfinite(joint_qd)), f"joint_qd has NaN/Inf: {joint_qd}"
        ready_q = np.array(ROBOT_A_READY_POSE + ROBOT_B_READY_POSE, dtype=np.float32)
        assert np.all(np.abs(joint_q - ready_q) < 0.2), f"Joints drifted from ready pose: {joint_q}"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
