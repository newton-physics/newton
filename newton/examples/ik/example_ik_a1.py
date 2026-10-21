# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example IK A1 (shared-object poses)
#
# Inverse kinematics on the OneRobotics A1 bimanual stand with two terminal
# links controlled by one interactive shared-object frame.
#
# - Uses position and rotation objectives for both terminal links
# - Keeps both wrist poses relative to the shared object frame
# - Uses a joint-limit objective to keep the solution valid
#
# Command: python -m newton.examples ik_a1
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils

A1_ASSET_PATH = "usd/a1_bimanual_stand/onerobotics_a1_bimanual_stand.usda"
TERMINAL_LINKS = ("Link_l7", "Link_r7")
TARGET_OFFSETS = np.array([[0.15, -0.06, 0.12], [0.15, 0.06, 0.12]], dtype=np.float32)


def _quat_to_vec4(quaternion: wp.quat) -> wp.vec4:
    """Convert a Warp quaternion to the vector representation IK expects."""
    return wp.vec4(quaternion[0], quaternion[1], quaternion[2], quaternion[3])


def _find_unique_bodies(builder: newton.ModelBuilder, labels: tuple[str, ...]) -> list[int]:
    """Resolve unique body indices from the final segments of USD paths."""
    bodies_by_label: dict[str, int] = {}
    for index, full_label in enumerate(builder.body_label):
        short_label = full_label.rsplit("/", 1)[-1]
        if short_label not in labels:
            continue
        if short_label in bodies_by_label:
            raise ValueError(f"A1 body label is not unique: {short_label}")
        bodies_by_label[short_label] = index

    missing = set(labels) - bodies_by_label.keys()
    if missing:
        raise ValueError(f"A1 terminal links are missing: {sorted(missing)}")
    return [bodies_by_label[label] for label in labels]


class Example:
    """Coordinate both A1 arms around an interactive shared object frame."""

    def __init__(self, viewer, args):
        """Load the fixed bimanual A1 stand and configure paired reach objectives."""
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.viewer = viewer
        self.viewer.picking_enabled = False
        self.device = wp.get_device()

        asset_root = newton.utils.download_asset("onerobotics_a1")
        builder = newton.ModelBuilder()
        builder.add_usd(
            str(asset_root / A1_ASSET_PATH),
            floating=False,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
            skip_mesh_approximation=True,
        )
        self.terminal_bodies = _find_unique_bodies(builder, TERMINAL_LINKS)

        self.model = builder.finalize()
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        initial_body_q = self.state.body_q.numpy()
        self.initial_joint_q = self.model.joint_q.numpy().copy()
        initial_targets = (
            np.asarray([initial_body_q[body, :3] for body in self.terminal_bodies], dtype=np.float32) + TARGET_OFFSETS
        )
        shared_object_position = initial_targets.mean(axis=0)
        self.object_tf = wp.transform(wp.vec3(*shared_object_position), wp.quat_identity())
        self.grasp_offsets = initial_targets - shared_object_position
        self.grasp_rotations = [wp.quat(*initial_body_q[body, 3:]) for body in self.terminal_bodies]
        self.target_positions = initial_targets.copy()
        self.target_rotations = self.grasp_rotations.copy()
        self.target_points = wp.array(self.target_positions, dtype=wp.vec3, device=self.device)
        self.target_colors = wp.array(
            np.tile(np.array([0.95, 0.25, 0.10], dtype=np.float32), (len(self.terminal_bodies), 1)),
            dtype=wp.vec3,
            device=self.device,
        )

        self.position_objectives = [
            ik.IKObjectivePosition(
                link_index=body,
                link_offset=wp.vec3(0.0, 0.0, 0.0),
                target_positions=wp.array([wp.vec3(*target)], dtype=wp.vec3, device=self.device),
            )
            for body, target in zip(self.terminal_bodies, self.target_positions, strict=True)
        ]
        self.rotation_objectives = [
            ik.IKObjectiveRotation(
                link_index=body,
                link_offset_rotation=wp.quat_identity(),
                target_rotations=wp.array([_quat_to_vec4(rotation)], dtype=wp.vec4, device=self.device),
            )
            for body, rotation in zip(self.terminal_bodies, self.target_rotations, strict=True)
        ]
        joint_limit_objective = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )
        self.joint_q = self.model.joint_q.reshape((1, self.model.joint_coord_count))
        self.solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[*self.position_objectives, *self.rotation_objectives, joint_limit_objective],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.5, 0.0, 0.5), pitch=-5.0, yaw=180.0)
        if args.test:
            object_position = wp.transform_get_translation(self.object_tf) + wp.vec3(0.02, -0.01, 0.01)
            object_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.15)
            self.object_tf = wp.transform(object_position, object_rotation)
            self.update_targets_from_object_frame()
        self.capture()

    def update_targets_from_object_frame(self):
        """Transform both local grasp poses from the shared object frame."""
        self.target_positions = np.asarray(
            [wp.transform_point(self.object_tf, wp.vec3(*offset)) for offset in self.grasp_offsets], dtype=np.float32
        )
        object_rotation = wp.transform_get_rotation(self.object_tf)
        self.target_rotations = [object_rotation * rotation for rotation in self.grasp_rotations]
        for objective, target in zip(self.position_objectives, self.target_positions, strict=True):
            objective.set_target_position(0, wp.vec3(*target))
        for objective, rotation in zip(self.rotation_objectives, self.target_rotations, strict=True):
            objective.set_target_rotation(0, _quat_to_vec4(rotation))
        self.target_points.assign(self.target_positions)

    def capture(self):
        """Capture the IK solve as a Warp graph when CUDA supports it."""
        if self.device.is_cuda and not wp.config.verify_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        """Solve both terminal-link pose objectives and update forward kinematics."""
        self.solver.step(self.joint_q, self.joint_q, iterations=24)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

    def step(self):
        """Advance the paired IK task using the current shared-object pose."""
        self.update_targets_from_object_frame()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Render the solved arms, the shared object frame, and its grasp targets."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.log_gizmo("a1_shared_object", self.object_tf)
        self.viewer.log_points(
            "a1_reach_targets",
            self.target_points,
            radii=0.025,
            colors=self.target_colors,
        )
        self.viewer.end_frame()

    def test_final(self):
        """Verify both terminal links reach their target poses and the joints move."""
        joint_q = self.model.joint_q.numpy()
        if not np.isfinite(joint_q).all():
            raise ValueError("A1 joint coordinates contain non-finite values")
        if np.max(np.abs(joint_q - self.initial_joint_q)) < 0.05:
            raise ValueError("A1 inverse kinematics did not move either arm")

        body_q = self.state.body_q.numpy()
        object_inv = wp.transform_inverse(self.object_tf)
        relative_poses = [object_inv * wp.transform(*body_q[body]) for body in self.terminal_bodies]
        relative_positions = np.asarray(
            [wp.transform_get_translation(pose) for pose in relative_poses], dtype=np.float32
        )
        position_errors = np.linalg.norm(relative_positions - self.grasp_offsets, axis=1)
        if np.max(position_errors) > 1.0e-3:
            raise ValueError(f"A1 relative grasp-position errors are {position_errors} m")

        relative_rotations = np.asarray([wp.transform_get_rotation(pose) for pose in relative_poses])
        grasp_rotations = np.asarray(self.grasp_rotations)
        orientation_alignment = np.abs(np.sum(relative_rotations * grasp_rotations, axis=1))
        if np.min(orientation_alignment) < 0.9999:
            raise ValueError(f"A1 relative grasp-orientation alignment is {orientation_alignment}")

if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
