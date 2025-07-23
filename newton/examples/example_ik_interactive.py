# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
from typing import Callable

import numpy as np
import warp as wp

wp.config.enable_backward = False

import newton
import newton.examples
import newton.sim.ik as ik
import newton.utils
from newton.sim import eval_fk
from newton.utils.gizmo import GizmoSystem

# -------------------------------------------------------------------------
# Utility classes
# -------------------------------------------------------------------------


class FrameAlignedHandler:
    """Buffers GUI events and forwards the *latest* one once per render frame."""

    def __init__(self, handler: Callable, consume_ret: Callable[..., bool] | None = None):
        self._handler = handler
        self._consume_ret = consume_ret or (lambda *_: False)
        self._pending: tuple | None = None

    def __call__(self, *args):
        self._pending = args
        return self._consume_ret(*args)

    def flush(self):
        if self._pending is None:
            return
        self._handler(*self._pending)
        self._pending = None


# -------------------------------------------------------------------------
# Example
# -------------------------------------------------------------------------


class Example:
    """Interactive inverse-kinematics playground for a batch of H1 robots."""

    END_EFFECTOR_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot")

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------

    def __init__(
        self,
        stage_path: str | None = "example_h1_ik_interactive.usd",
        num_envs: int = 4,
        tie_targets: bool = True,
        ik_iters: int = 20,
    ):
        self.stage_path = stage_path
        self.num_envs = num_envs
        self.tie_targets = tie_targets
        self.ik_iters = ik_iters

        # timings ------------------------------------------------------
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        # model(s) -----------------------------------------------------
        (
            self.model,
            self.num_links,
            self.ee_link_indices,
            self.ee_link_offsets,
            self.num_coords,
            self.env_offsets,
        ) = self._build_model(num_envs)

        # dedicated 1-env model for IK
        self.singleton_model, *_ = self._build_model(1)

        # simulation state --------------------------------------------
        self.state = self.model.state()
        self.joint_q = wp.zeros((num_envs, self.singleton_model.joint_coord_count), dtype=wp.float32)

        # target buffers ----------------------------------------------
        self.target_positions, self.target_rotations = self._initialize_targets()
        (
            self.position_objectives,
            self.rotation_objectives,
            total_residuals,
        ) = self._create_objectives()

        # joint limits -------------------------------------------------
        joint_limit_objective = ik.JointLimitObjective(
            joint_limit_lower=self.singleton_model.joint_limit_lower,
            joint_limit_upper=self.singleton_model.joint_limit_upper,
            n_problems=num_envs,
            total_residuals=total_residuals,
            residual_offset=len(self.END_EFFECTOR_NAMES) * 3 * 2,
            weight=0.1,
        )

        # IK solver ----------------------------------------------------
        self.ik_solver = ik.IKSolver(
            model=self.singleton_model,
            joint_q=self.joint_q,
            objectives=self.position_objectives + self.rotation_objectives + [joint_limit_objective],
            lambda_initial=0.1,
            jacobian_mode=ik.JacobianMode.ANALYTIC,
        )

        # renderer & gizmos -------------------------------------------
        self.renderer = None
        if stage_path:
            self.renderer = newton.utils.SimRendererOpenGL(path=stage_path, model=self.model, scaling=1.0)
            self._setup_gizmos()

        # warm-up + CUDA graph ----------------------------------------
        self.use_cuda_graph = wp.get_device().is_cuda
        self.ik_solver.solve(iterations=ik_iters)  # JIT & cache
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.ik_solver.solve(iterations=ik_iters)
                wp.copy(self.model.joint_q, self.joint_q.flatten())
            self.graph = capture.graph

    # -----------------------------------------------------------------
    # Model construction helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _build_model(num_envs: int):
        """Return (`model`, `num_links`, `ee_indices`, `ee_offsets`, `n_coords`, `env_offsets`)."""
        articulation_builder = newton.ModelBuilder()
        articulation_builder.default_shape_cfg.density = 100.0
        articulation_builder.default_joint_cfg.armature = 0.1
        articulation_builder.default_body_armature = 0.1

        newton.utils.parse_mjcf(
            newton.utils.download_asset("h1_description") / "mjcf/h1_with_hand.xml",
            articulation_builder,
            floating=False,
        )

        # initial joint angles (same as original script) --------------
        initial_joint_positions = [
            0.0,
            0.0,
            -0.3,
            0.6,
            -0.3,  # left leg
            0.0,
            0.0,
            -0.3,
            0.6,
            -0.3,  # right leg
            0.0,  # torso pitch
            0.0,
            0.0,
            0.0,
            -0.5,  # left arm
            0.0,
            -0.3,
            0.0,
            -0.8,  # right arm
        ]
        joint_mapping = [
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            35,
            36,
            37,
            38,
        ]
        for joint_idx, value in zip(joint_mapping, initial_joint_positions):
            articulation_builder.joint_q[joint_idx - 7] = value

        articulation_builder.joint_q[22 - 7] = 0.0  # left_hand_joint
        articulation_builder.joint_q[39 - 7] = 0.0  # right_hand_joint

        # zero all finger joints
        for i in range(23, 35):
            articulation_builder.joint_q[i - 7] = 0.0
        for i in range(40, 52):
            articulation_builder.joint_q[i - 7] = 0.0

        # wrap into batched ModelBuilder ------------------------------
        builder = newton.ModelBuilder()
        builder.num_rigid_contacts_per_env = 0

        env_offsets = newton.examples.compute_env_offsets(num_envs, env_offset=(-1.0, -2.0, 0.0))

        for pos in env_offsets:
            builder.add_builder(articulation_builder, xform=wp.transform(pos, wp.quat_identity()))

        builder.add_ground_plane()
        model = builder.finalize(requires_grad=True)

        ee_link_indices = [16, 33, 5, 10]  # hands & ankles
        ee_link_offsets = [wp.vec3()] * 4

        return (
            model,
            len(articulation_builder.body_q),
            ee_link_indices,
            ee_link_offsets,
            len(articulation_builder.joint_q),
            env_offsets,
        )

    # -----------------------------------------------------------------
    # Targets & Objectives
    # -----------------------------------------------------------------

    def _initialize_targets(self):
        """Compute initial local-frame targets from current FK."""
        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state, None)

        num_ees = len(self.END_EFFECTOR_NAMES)
        tgt_pos = np.zeros((self.num_envs, num_ees, 3), dtype=np.float32)
        tgt_rot = np.zeros((self.num_envs, num_ees, 4), dtype=np.float32)

        body_q_np = self.state.body_q.numpy()
        for env in range(self.num_envs):
            base = env * self.num_links
            for ee_idx, link_idx in enumerate(self.ee_link_indices):
                tf = body_q_np[base + link_idx]
                world_pos = wp.transform_point(wp.transform(tf[:3], wp.quat(*tf[3:])), self.ee_link_offsets[ee_idx])
                tgt_pos[env, ee_idx] = np.array(world_pos) - self.env_offsets[env]  # ← LOCAL!
                quat = tf[3:7] / np.linalg.norm(tf[3:7])
                tgt_rot[env, ee_idx] = quat
        return tgt_pos, tgt_rot

    def _create_objectives(self):
        num_ees = len(self.END_EFFECTOR_NAMES)
        total_residuals = num_ees * 3 * 2 + self.num_coords

        position_objectives, rotation_objectives = [], []
        self.position_target_arrays, self.rotation_target_arrays = [], []

        for ee_idx in range(num_ees):
            pos_wp = wp.array(self.target_positions[:, ee_idx], dtype=wp.vec3)
            rot_wp = wp.array(self.target_rotations[:, ee_idx], dtype=wp.vec4)
            self.position_target_arrays.append(pos_wp)
            self.rotation_target_arrays.append(rot_wp)

        # position objectives -----------------------------------------
        for ee_idx, (link_idx, offset) in enumerate(zip(self.ee_link_indices, self.ee_link_offsets)):
            obj = ik.PositionObjective(
                link_index=link_idx,
                link_offset=offset,
                target_positions=self.position_target_arrays[ee_idx],
                n_problems=self.num_envs,
                total_residuals=total_residuals,
                residual_offset=ee_idx * 3,
            )
            position_objectives.append(obj)

        # rotation objectives -----------------------------------------
        for ee_idx, link_idx in enumerate(self.ee_link_indices):
            obj = ik.RotationObjective(
                link_index=link_idx,
                link_offset_rotation=wp.quat_identity(),
                target_rotations=self.rotation_target_arrays[ee_idx],
                n_problems=self.num_envs,
                total_residuals=total_residuals,
                residual_offset=num_ees * 3 + ee_idx * 3,
            )
            rotation_objectives.append(obj)

        return position_objectives, rotation_objectives, total_residuals

    # -----------------------------------------------------------------
    # Gizmos & mouse interaction
    # -----------------------------------------------------------------

    GIZMO_OFFSET_DISTANCE = 0.3
    GIZMO_OFFSETS = (
        np.array([0.0, GIZMO_OFFSET_DISTANCE, 0.0], dtype=np.float32),
        np.array([0.0, -GIZMO_OFFSET_DISTANCE, 0.0], dtype=np.float32),
        np.array([0.0, GIZMO_OFFSET_DISTANCE, 0.0], dtype=np.float32),
        np.array([0.0, -GIZMO_OFFSET_DISTANCE, 0.0], dtype=np.float32),
    )

    def _setup_gizmos(self):
        self.gizmo_system = GizmoSystem(self.renderer, scale_factor=0.15, rotation_sensitivity=1.0)
        self.gizmo_system.set_callbacks(
            position_callback=self._on_position_dragged,
            rotation_callback=self._on_rotation_dragged,
        )

        if self.tie_targets:
            for ee_idx in range(len(self.END_EFFECTOR_NAMES)):
                world_pos = self._local_to_world(0, self.target_positions[0, ee_idx])
                self.gizmo_system.create_target(
                    ee_idx,
                    world_pos,
                    self.target_rotations[0, ee_idx],
                    self.GIZMO_OFFSETS[ee_idx],
                )
        else:
            for env in range(self.num_envs):
                for ee_idx in range(len(self.END_EFFECTOR_NAMES)):
                    gid = env * len(self.END_EFFECTOR_NAMES) + ee_idx
                    world_pos = self._local_to_world(env, self.target_positions[env, ee_idx])
                    self.gizmo_system.create_target(
                        gid,
                        world_pos,
                        self.target_rotations[env, ee_idx],
                        self.GIZMO_OFFSETS[ee_idx],
                    )
        self.gizmo_system.finalize()

        # frame-aligned wrappers --------------------------------------
        self._mouse_press_handler = FrameAlignedHandler(self.gizmo_system.on_mouse_press)
        self._mouse_drag_handler = FrameAlignedHandler(
            self.gizmo_system.on_mouse_drag,
            consume_ret=lambda *_: self.gizmo_system.drag_state is not None,
        )
        self._mouse_release_handler = FrameAlignedHandler(self.gizmo_system.on_mouse_release)

        # register pyglet callbacks
        self.renderer.window.push_handlers(on_mouse_press=self._mouse_press_handler)
        self.renderer.window.push_handlers(on_mouse_drag=self._mouse_drag_handler)
        self.renderer.window.push_handlers(on_mouse_release=self._mouse_release_handler)

        # tied-target drag bookkeeping
        if self.tie_targets:
            self._drag_start_positions = None
            self._drag_start_rotations = None
            self._is_dragging_pos = False
            self._is_dragging_rot = False
            self._last_drag_id: int | None = None

    # coordinate helpers ----------------------------------------------

    def _local_to_world(self, env_idx: int, local_pos: np.ndarray):
        return local_pos + self.env_offsets[env_idx]

    # snapshot helpers for tied drag ----------------------------------

    def _capture_drag_start(self):
        self._drag_start_positions = np.copy(self.target_positions)
        self._drag_start_rotations = np.copy(self.target_rotations)

    # callbacks -------------------------------------------------------

    def _on_position_dragged(self, global_id: int, new_world_pos: np.ndarray):
        env = global_id // len(self.END_EFFECTOR_NAMES)
        ee = global_id % len(self.END_EFFECTOR_NAMES)
        local_pos = new_world_pos - self.env_offsets[env]

        if self.tie_targets:
            if not self._is_dragging_pos or self._last_drag_id != global_id:
                self._capture_drag_start()
                self._is_dragging_pos = True
                self._last_drag_id = global_id

            delta = local_pos - self._drag_start_positions[env, ee]
            new_local_targets = self._drag_start_positions[:, ee] + delta
            self.target_positions[:, ee] = new_local_targets
            self.position_objectives[ee].set_target_positions(wp.array(new_local_targets, dtype=wp.vec3))
            self._is_dragging_pos = False
        else:
            self.target_positions[env, ee] = local_pos
            self.position_objectives[ee].set_target_position(env, wp.vec3(*local_pos))

        self._solve()

    def _on_rotation_dragged(self, global_id: int, new_q: np.ndarray):
        num_ees = len(self.END_EFFECTOR_NAMES)
        env = global_id // num_ees
        ee = global_id % num_ees

        new_q = new_q / np.linalg.norm(new_q)

        if self.tie_targets:
            # start of a new drag?
            if (not self._is_dragging_rot) or (self._last_drag_id != global_id):
                self._drag_start_rotations = np.copy(self.target_rotations)
                self._is_dragging_rot = True
                self._last_drag_id = global_id

            # delta = new * conj(initial)
            q0 = self._drag_start_rotations[env, ee]
            conj = np.array([-q0[0], -q0[1], -q0[2], q0[3]], dtype=np.float32)

            delta = np.array(
                [
                    new_q[3] * conj[0] + new_q[0] * conj[3] + new_q[1] * conj[2] - new_q[2] * conj[1],
                    new_q[3] * conj[1] - new_q[0] * conj[2] + new_q[1] * conj[3] + new_q[2] * conj[0],
                    new_q[3] * conj[2] + new_q[0] * conj[1] - new_q[1] * conj[0] + new_q[2] * conj[3],
                    new_q[3] * conj[3] - new_q[0] * conj[0] - new_q[1] * conj[1] - new_q[2] * conj[2],
                ],
                dtype=np.float32,
            )
            delta /= np.linalg.norm(delta)

            # apply the same delta to every env's stored initial rotation
            initial = self._drag_start_rotations[:, ee]  # shape (num_envs, 4)
            q1, q2 = delta, initial  # aliases to match original math

            updated = np.zeros_like(initial)
            updated[:, 0] = q1[3] * q2[:, 0] + q1[0] * q2[:, 3] + q1[1] * q2[:, 2] - q1[2] * q2[:, 1]
            updated[:, 1] = q1[3] * q2[:, 1] - q1[0] * q2[:, 2] + q1[1] * q2[:, 3] + q1[2] * q2[:, 0]
            updated[:, 2] = q1[3] * q2[:, 2] + q1[0] * q2[:, 1] - q1[1] * q2[:, 0] + q1[2] * q2[:, 3]
            updated[:, 3] = q1[3] * q2[:, 3] - q1[0] * q2[:, 0] - q1[1] * q2[:, 1] - q1[2] * q2[:, 2]

            # normalise all rows (protects against numeric drift)
            updated /= np.linalg.norm(updated, axis=1, keepdims=True)

            self.target_rotations[:, ee] = updated
            self.rotation_objectives[ee].set_target_rotations(wp.array(updated, dtype=wp.vec4))

            self._is_dragging_rot = False
        else:
            # untied mode: update only this environment
            self.target_rotations[env, ee] = new_q
            self.rotation_objectives[ee].set_target_rotation(env, wp.vec4(*new_q))

        # re-solve IK
        self._solve()

    # -----------------------------------------------------------------
    # Solve / render / run
    # -----------------------------------------------------------------

    def _solve(self):
        with wp.ScopedTimer("solve", synchronize=True):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.ik_solver.solve(iterations=self.ik_iters)
                wp.copy(self.model.joint_q, self.joint_q.flatten())

    def _render_frame(self):
        if self.renderer is None:
            return

        self.renderer.begin_frame(self.sim_time)
        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state, None)
        self.renderer.render(self.state)
        self.renderer.end_frame()

    def run(self):
        # initial solve so joints match targets before first frame
        if not self.use_cuda_graph:
            self.ik_solver.solve(iterations=self.ik_iters)

        while self.renderer.is_running():
            self.sim_time += self.frame_dt

            # process any pending GUI events
            self._mouse_press_handler.flush()
            self._mouse_drag_handler.flush()
            self._mouse_release_handler.flush()

            self._render_frame()

        self.renderer.close()


# -------------------------------------------------------------------------
# main()
# -------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_h1_ik_interactive.usd",
        help="Path of the output USD.",
    )
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments.")
    parser.add_argument(
        "--tie-targets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Tie all envs together so dragging one EE moves the others.",
    )
    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            stage_path=args.stage_path,
            num_envs=args.num_envs,
            tie_targets=args.tie_targets,
        )
        example.run()


if __name__ == "__main__":
    main()
