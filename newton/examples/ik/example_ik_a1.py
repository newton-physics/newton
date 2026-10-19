# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solve a shared-object dual-arm reach task on the OneRobotics A1 stand.

One :class:`newton.ik.IKSolver` solves position objectives for both terminal
links simultaneously. A single interactive object frame moves both grasp
targets together, while a joint-limit objective keeps the solution valid.

Command: python -m newton.examples ik_a1
"""

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils

A1_ASSET_PATH = Path("usd/a1_bimanual_stand/onerobotics_a1_bimanual_stand.usda")
TERMINAL_LINKS = ("Link_l7", "Link_r7")
TARGET_OFFSETS = np.array([[0.15, -0.06, 0.12], [0.15, 0.06, 0.12]], dtype=np.float32)


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
        self.target_positions = initial_targets.copy()
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
        joint_limit_objective = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )
        self.joint_q = self.model.joint_q.reshape((1, self.model.joint_coord_count))
        self.solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[*self.position_objectives, joint_limit_objective],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.5, 0.0, 0.5), pitch=-5.0, yaw=180.0)
        self.capture()

    def update_targets_from_object_frame(self):
        """Transform both local grasp offsets from the shared object frame."""
        self.target_positions = np.asarray(
            [wp.transform_point(self.object_tf, wp.vec3(*offset)) for offset in self.grasp_offsets], dtype=np.float32
        )
        for objective, target in zip(self.position_objectives, self.target_positions, strict=True):
            objective.set_target_position(0, wp.vec3(*target))
        self.target_points.assign(self.target_positions)

    def capture(self):
        """Capture the IK solve as a Warp graph when CUDA supports it."""
        if wp.get_device().is_cuda and not wp.config.verify_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        """Solve both terminal-link position objectives and update forward kinematics."""
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
        """Verify both terminal links reach their targets and the joints actually move."""
        joint_q = self.model.joint_q.numpy()
        if not np.isfinite(joint_q).all():
            raise ValueError("A1 joint coordinates contain non-finite values")
        if np.max(np.abs(joint_q - self.initial_joint_q)) < 0.05:
            raise ValueError("A1 inverse kinematics did not move either arm")

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        body_q = self.state.body_q.numpy()
        actual_positions = np.asarray([body_q[body, :3] for body in self.terminal_bodies])
        position_errors = np.linalg.norm(actual_positions - self.target_positions, axis=1)
        if np.max(position_errors) > 1.0e-3:
            raise ValueError(f"A1 terminal-link position errors are {position_errors} m")

    @staticmethod
    def create_parser():
        """Create the common Newton example parser."""
        return newton.examples.create_parser()


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
