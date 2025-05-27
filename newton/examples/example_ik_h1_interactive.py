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

import time

import numpy as np
import warp as wp
from gizmo_utils import GizmoSystem

import newton
import newton.core.articulation
import newton.core.ik as ik
import newton.examples
import newton.utils


class FrameAlignedHandlerWrapper:
    def __init__(self, real_handler, ret_fn=lambda *params: False):
        self.real_handler = real_handler
        self.pending_params = None
        self.last_frame_time = 0
        self.ret_fn = ret_fn

    def wrapped_handler(self, *params):
        # Just store the latest params, don't process yet
        self.pending_params = params
        return self.ret_fn()

    def process_frame(self):
        """Call this once per render frame"""
        now = time.perf_counter()

        # Process pending update if any
        if self.pending_params is not None:
            self.real_handler(*self.pending_params)
            self.pending_params = None

        self.last_frame_time = now


class Example:
    def __init__(
        self,
        stage_path="example_h1_ik_interactive.usd",
        num_envs=4,
        randomize_targets=False,
        tie_targets=False,
        scale=1.0,
        use_step_api=False,
        ik_iters=20,
    ):
        self.rng = np.random.default_rng()

        self.num_envs = num_envs
        self.frame_dt = 1.0 / 60
        self.sim_time = 0.0
        self.randomize_targets = randomize_targets
        self.tie_targets = tie_targets
        self.robot_scale = scale
        self.use_step_api = use_step_api
        self.ik_iters = ik_iters

        self.ee_names = ["left_hand", "right_hand", "left_foot", "right_foot"]
        self.num_ees = len(self.ee_names)

        self.gizmo_offset_distance = 0.3 * self.robot_scale
        self.gizmo_offsets = [
            np.array([0.0, 0.0, -self.gizmo_offset_distance], dtype=np.float32),
            np.array([0.0, 0.0, self.gizmo_offset_distance], dtype=np.float32),
            np.array([0.0, 0.0, -self.gizmo_offset_distance], dtype=np.float32),
            np.array([0.0, 0.0, self.gizmo_offset_distance], dtype=np.float32),
        ]

        # Drag state tracking for tied targets
        if self.tie_targets:
            self.drag_initial_positions = None
            self.drag_initial_rotations = None
            self.is_dragging_position = False
            self.is_dragging_rotation = False
            self.last_dragged_id = None

        self.model, self.num_links, self.ee_link_indices, self.ee_link_offsets, self.coords = self._build_model(
            num_envs
        )

        self.target_positions, self.target_rotations = self._initialize_targets(num_envs)

        self.objectives = []
        self.position_objectives = []
        self.rotation_objectives = []

        # - position: 3 per end-effector
        # - rotation: 3 per end-effector
        # - joint limits: 1 per coordinate
        total_residuals = self.num_ees * 3 + self.num_ees * 3 + self.coords

        self.position_target_arrays = []
        for ee_idx in range(self.num_ees):
            targets = np.array([self.target_positions[env_idx, ee_idx] for env_idx in range(num_envs)])
            target_array = wp.array(targets, dtype=wp.vec3)
            self.position_target_arrays.append(target_array)

        self.rotation_target_arrays = []
        for ee_idx in range(self.num_ees):
            targets = np.array([self.target_rotations[env_idx, ee_idx] for env_idx in range(num_envs)])
            target_array = wp.array(targets, dtype=wp.vec4)
            self.rotation_target_arrays.append(target_array)

        for ee_idx in range(self.num_ees):
            pos_obj = ik.PositionObjective(
                link_index=self.ee_link_indices[ee_idx],
                link_offset=self.ee_link_offsets[ee_idx],
                target_positions=self.position_target_arrays[ee_idx],
                num_links=self.num_links,
                num_envs=self.num_envs,
                total_residuals=total_residuals,
                residual_offset=ee_idx * 3,
            )
            self.objectives.append(pos_obj)
            self.position_objectives.append(pos_obj)

        for ee_idx in range(self.num_ees):
            rot_obj = ik.RotationObjective(
                link_index=self.ee_link_indices[ee_idx],
                link_offset_rotation=wp.quat_identity(),
                target_rotations=self.rotation_target_arrays[ee_idx],
                num_links=self.num_links,
                num_envs=self.num_envs,
                total_residuals=total_residuals,
                residual_offset=self.num_ees * 3 + ee_idx * 3,
            )
            self.objectives.append(rot_obj)
            self.rotation_objectives.append(rot_obj)

        joint_limit_obj = ik.JointLimitObjective(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=0.1,
            num_envs=self.num_envs,
            total_residuals=total_residuals,
            residual_offset=self.num_ees * 3 + self.num_ees * 3,
        )
        self.objectives.append(joint_limit_obj)

        self.ik_sys = ik.create_ik(
            model=self.model,
            num_envs=self.num_envs,
            objectives=self.objectives,
            damping=1.0,
            jacobian_mode=ik.JacobianMode.ANALYTIC,
        )

        # Create separate state objects for step-based solving if using step API
        if self.use_step_api:
            self.state_current = self.model.state()
            self.state_next = self.model.state()
            # Initialize current state with model's joint positions
            wp.copy(self.state_current.joint_q, self.model.joint_q)

        if stage_path:
            self.renderer = newton.utils.SimRendererOpenGL(
                path=stage_path, model=self.model, scaling=1.0 / self.robot_scale
            )
        else:
            self.renderer = None

        if self.renderer:
            self.gizmo_system = GizmoSystem(
                self.renderer, scale_factor=0.15 * self.robot_scale, rotation_sensitivity=1.0
            )

            self.gizmo_system.set_callbacks(
                position_callback=self._on_position_dragged, rotation_callback=self._on_rotation_dragged
            )

            # Create gizmos - only for first env if targets are tied, all envs otherwise
            if self.tie_targets:
                # Only create gizmos for the first environment
                for ee_idx in range(self.num_ees):
                    global_id = 0 * self.num_ees + ee_idx
                    target_position = self.target_positions[0, ee_idx]
                    target_rotation = self.target_rotations[0, ee_idx]
                    world_offset = self.gizmo_offsets[ee_idx]

                    self.gizmo_system.create_target(global_id, target_position, target_rotation, world_offset)
            else:
                # Create gizmos for all environments
                for env_idx in range(num_envs):
                    for ee_idx in range(self.num_ees):
                        global_id = env_idx * self.num_ees + ee_idx
                        target_position = self.target_positions[env_idx, ee_idx]
                        target_rotation = self.target_rotations[env_idx, ee_idx]
                        world_offset = self.gizmo_offsets[ee_idx]

                        self.gizmo_system.create_target(global_id, target_position, target_rotation, world_offset)

            self.frame_aligned_press = FrameAlignedHandlerWrapper(self.gizmo_system.on_mouse_press)
            # ret_fn returns true to avoid viewpoint adjustment while dragging
            self.frame_aligned_drag = FrameAlignedHandlerWrapper(
                self.gizmo_system.on_mouse_drag, ret_fn=lambda *params: self.gizmo_system.drag_state is not None
            )
            self.frame_aligned_release = FrameAlignedHandlerWrapper(self.gizmo_system.on_mouse_release)
            self.renderer.window.push_handlers(on_mouse_press=self.frame_aligned_press.wrapped_handler)
            self.renderer.window.push_handlers(on_mouse_drag=self.frame_aligned_drag.wrapped_handler)
            self.renderer.window.push_handlers(on_mouse_release=self.frame_aligned_release.wrapped_handler)

        self.use_cuda_graph = wp.get_device().is_cuda

        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.solve_ik()
            self.graph = capture.graph

    def _build_model(self, num_envs):
        articulation_builder = newton.ModelBuilder()
        # Adjust density inversely with scale cubed to maintain mass
        articulation_builder.default_shape_cfg.density = 100.0 / (self.robot_scale**3)
        articulation_builder.default_joint_cfg.armature = 0.1
        articulation_builder.default_body_armature = 0.1

        # Load H1 robot with hands from MJCF
        newton.utils.parse_mjcf(
            newton.utils.download_asset("h1_description") / "mjcf/h1_with_hand.xml",
            articulation_builder,
            floating=False,
            armature_scale=self.robot_scale,
            scale=self.robot_scale,
        )

        initial_joint_positions = [
            0.0,  # left_hip_yaw
            0.0,  # left_hip_roll
            -0.3,  # left_hip_pitch
            0.6,  # left_knee
            -0.3,  # left_ankle
            0.0,  # right_hip_yaw
            0.0,  # right_hip_roll
            -0.3,  # right_hip_pitch
            0.6,  # right_knee
            -0.3,  # right_ankle
            0.0,  # torso
            0.0,  # left_shoulder_pitch
            0.0,  # left_shoulder_roll
            0.0,  # left_shoulder_yaw
            -0.5,  # left_elbow
            0.0,  # right_shoulder_pitch
            -0.3,  # right_shoulder_roll
            0.0,  # right_shoulder_yaw
            -0.8,  # right_elbow
        ]

        # Joint mapping from initial_joint_positions to model joints (after 7 floating base coords)
        joint_mapping = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 35, 36, 37, 38]

        for _, (joint_idx, value) in enumerate(zip(joint_mapping, initial_joint_positions)):
            articulation_builder.joint_q[joint_idx - 7] = value

        articulation_builder.joint_q[22 - 7] = 0.0  # left_hand_joint
        articulation_builder.joint_q[39 - 7] = 0.0  # right_hand_joint

        for i in range(23, 35):  # left hand finger joints
            articulation_builder.joint_q[i - 7] = 0.0
        for i in range(40, 52):  # right hand finger joints
            articulation_builder.joint_q[i - 7] = 0.0

        builder = newton.ModelBuilder()
        builder.num_rigid_contacts_per_env = 0

        num_links = len(articulation_builder.body_q)

        ee_link_indices = [
            16,  # left_hand_link
            33,  # right_hand_link (after 12 left finger bodies)
            5,  # left_ankle_link
            10,  # right_ankle_link
        ]

        hand_offset = wp.vec3(0.0, 0.0, 0.0)
        foot_offset = wp.vec3(0.0, 0.0, 0.0)
        ee_link_offsets = [hand_offset, hand_offset, foot_offset, foot_offset]

        coords = len(articulation_builder.joint_q)

        positions = newton.examples.compute_env_offsets(
            num_envs, env_offset=(-1.0 * self.robot_scale, 0.0, -2.0 * self.robot_scale)
        )

        upright_quat = wp.quat_from_axis_angle(wp.vec3(-1.0, 0.0, 0.0), wp.pi / 2.0)
        for i in range(num_envs):
            builder.add_builder(articulation_builder, xform=wp.transform(positions[i], upright_quat))

        builder.up_axis = "Y"
        model = builder.finalize(requires_grad=True)
        model.ground = True

        return model, num_links, ee_link_indices, ee_link_offsets, coords

    def _initialize_targets(self, num_envs):
        state = self.model.state()
        newton.core.articulation.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state, None)

        target_positions = np.zeros((num_envs, self.num_ees, 3), dtype=np.float32)
        target_rotations = np.zeros((num_envs, self.num_ees, 4), dtype=np.float32)  # x,y,z,w

        # Randomization scale
        position_noise_scale = 0.3 * self.robot_scale
        rotation_noise_scale = 0.2

        for i in range(num_envs):
            for ee_idx in range(self.num_ees):
                body_tf = state.body_q.numpy()[i * self.num_links + self.ee_link_indices[ee_idx]]

                ee_pos = wp.transform_point(
                    wp.transform(body_tf[:3], wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])),
                    self.ee_link_offsets[ee_idx],
                )

                # Add random offset to position if randomization is enabled
                if self.randomize_targets and (not self.tie_targets or i > 0):
                    noise = self.rng.uniform(-position_noise_scale, position_noise_scale, 3)
                    ee_pos = ee_pos + wp.vec3(noise[0], noise[1], noise[2])

                target_positions[i, ee_idx] = ee_pos

                # Get base rotation
                base_quat = body_tf[3:7]

                # Add small random rotation if randomization is enabled
                if self.randomize_targets and (not self.tie_targets or i > 0):
                    # Create small random rotation
                    angle = self.rng.uniform(0, rotation_noise_scale)
                    axis = self.rng.uniform(-1, 1, 3)
                    axis = axis / np.linalg.norm(axis)

                    # Convert to quaternion
                    s = np.sin(angle / 2)
                    c = np.cos(angle / 2)
                    random_quat = np.array([axis[0] * s, axis[1] * s, axis[2] * s, c])

                    # Multiply quaternions
                    q1 = base_quat
                    q2 = random_quat
                    result_quat = np.array(
                        [
                            q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1],
                            q1[3] * q2[1] - q1[0] * q2[2] + q1[1] * q2[3] + q1[2] * q2[0],
                            q1[3] * q2[2] + q1[0] * q2[1] - q1[1] * q2[0] + q1[2] * q2[3],
                            q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2],
                        ]
                    )
                    # Normalize the result
                    result_quat = result_quat / np.linalg.norm(result_quat)
                    target_rotations[i, ee_idx] = result_quat
                else:
                    # Normalize base quaternion
                    normalized_base = base_quat / np.linalg.norm(base_quat)
                    target_rotations[i, ee_idx] = normalized_base

        return target_positions, target_rotations

    def _capture_drag_initial_states(self):
        """Capture current target states at start of drag"""
        self.drag_initial_positions = np.copy(self.target_positions)
        self.drag_initial_rotations = np.copy(self.target_rotations)

        # Normalize all stored rotations to prevent accumulation of errors
        for env in range(self.num_envs):
            for ee in range(self.num_ees):
                norm = np.linalg.norm(self.drag_initial_rotations[env, ee])
                if norm > 0:
                    self.drag_initial_rotations[env, ee] /= norm

    def _on_position_dragged(self, global_target_id, new_position):
        env_idx = global_target_id // self.num_ees
        ee_idx = global_target_id % self.num_ees

        if self.tie_targets:
            if not self.is_dragging_position or self.last_dragged_id != global_target_id:
                self._capture_drag_initial_states()
                self.is_dragging_position = True
                self.last_dragged_id = global_target_id

            initial_pos = self.drag_initial_positions[env_idx, ee_idx]
            delta = new_position - initial_pos

            initial_positions = self.drag_initial_positions[:, ee_idx]  # Shape: (num_envs, 3)

            all_target_positions = initial_positions + delta

            self.target_positions[:, ee_idx] = all_target_positions

            wp_positions = wp.array(all_target_positions, dtype=wp.vec3)
            self.position_objectives[ee_idx].set_target_positions(wp_positions)

            self.is_dragging_position = False
        else:
            self.target_positions[env_idx, ee_idx] = new_position

            self.position_objectives[ee_idx].set_target_position(
                env_idx, wp.vec3(new_position[0], new_position[1], new_position[2])
            )

        with wp.ScopedTimer("solve", synchronize=True):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.solve_ik()

    def _on_rotation_dragged(self, global_target_id, new_rotation):
        env_idx = global_target_id // self.num_ees
        ee_idx = global_target_id % self.num_ees

        if self.tie_targets:
            # Detect new drag
            if not self.is_dragging_rotation or self.last_dragged_id != global_target_id:
                self._capture_drag_initial_states()
                self.is_dragging_rotation = True
                self.last_dragged_id = global_target_id

            # Normalize input rotation
            new_rotation = new_rotation / np.linalg.norm(new_rotation)

            # Calculate rotation delta
            initial_rot = self.drag_initial_rotations[env_idx, ee_idx]

            # Compute quaternion difference: delta = new * conjugate(initial)
            initial_conj = np.array([-initial_rot[0], -initial_rot[1], -initial_rot[2], initial_rot[3]])

            # Quaternion multiplication
            q1 = new_rotation
            q2 = initial_conj
            delta_quat = np.array(
                [
                    q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1],
                    q1[3] * q2[1] - q1[0] * q2[2] + q1[1] * q2[3] + q1[2] * q2[0],
                    q1[3] * q2[2] + q1[0] * q2[1] - q1[1] * q2[0] + q1[2] * q2[3],
                    q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2],
                ]
            )

            delta_quat = delta_quat / np.linalg.norm(delta_quat)

            initial_rotations = self.drag_initial_rotations[:, ee_idx]  # Shape: (num_envs, 4)

            q1 = delta_quat
            q2 = initial_rotations

            # Vectorized quaternion multiplication
            all_target_rotations = np.zeros((self.num_envs, 4), dtype=np.float32)
            all_target_rotations[:, 0] = q1[3] * q2[:, 0] + q1[0] * q2[:, 3] + q1[1] * q2[:, 2] - q1[2] * q2[:, 1]
            all_target_rotations[:, 1] = q1[3] * q2[:, 1] - q1[0] * q2[:, 2] + q1[1] * q2[:, 3] + q1[2] * q2[:, 0]
            all_target_rotations[:, 2] = q1[3] * q2[:, 2] + q1[0] * q2[:, 1] - q1[1] * q2[:, 0] + q1[2] * q2[:, 3]
            all_target_rotations[:, 3] = q1[3] * q2[:, 3] - q1[0] * q2[:, 0] - q1[1] * q2[:, 1] - q1[2] * q2[:, 2]

            norms = np.linalg.norm(all_target_rotations, axis=1, keepdims=True)
            all_target_rotations = all_target_rotations / norms

            self.target_rotations[:, ee_idx] = all_target_rotations

            wp_rotations = wp.array(all_target_rotations, dtype=wp.vec4)
            self.rotation_objectives[ee_idx].set_target_rotations(wp_rotations)

            self.is_dragging_rotation = False
        else:
            self.target_rotations[env_idx, ee_idx] = new_rotation

            self.rotation_objectives[ee_idx].set_target_rotation(
                env_idx, wp.vec4(new_rotation[0], new_rotation[1], new_rotation[2], new_rotation[3])
            )

        with wp.ScopedTimer("solve", synchronize=True):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.solve_ik()

    def render(self):
        if self.renderer is None:
            return

        self.renderer.begin_frame(self.sim_time)

        if self.use_step_api:
            # Use current state for FK evaluation
            newton.core.articulation.eval_fk(
                self.model, self.state_current.joint_q, self.model.joint_qd, self.ik_sys.state
            )
        else:
            newton.core.articulation.eval_fk(
                self.model, self.model.joint_q, self.model.joint_qd, self.ik_sys.state, None
            )

        self.renderer.render(self.ik_sys.state)

        self.renderer.end_frame()

    def solve_ik(self):
        if self.use_step_api:
            self.ik_sys.step(self.state_current, self.state_next, iterations=self.ik_iters)
            self.state_current, self.state_next = self.state_next, self.state_current
        else:
            self.ik_sys.solve(iterations=self.ik_iters)

    def run(self):
        if self.renderer is None:
            return

        if self.use_cuda_graph:
            wp.capture_launch(self.graph)
        else:
            self.solve_ik()

        while self.renderer.is_running():
            self.sim_time += self.frame_dt
            self.frame_aligned_press.process_frame()
            self.frame_aligned_drag.process_frame()
            self.frame_aligned_release.process_frame()
            self.render()

        self.renderer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_h1_ik_interactive.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_envs", type=int, default=4, help="Total number of simulated environments.")
    parser.add_argument(
        "--randomize_targets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Randomize initial target positions for each environment.",
    )
    parser.add_argument(
        "--tie_targets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Tie all environments together - dragging one broadcasts (delta) to all.",
    )
    parser.add_argument(
        "--use_step_api",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Demonstrate step(state_in, state_out) API",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="Model scale (1.0 for meters, 100.0 for cm)")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            stage_path=args.stage_path,
            num_envs=args.num_envs,
            randomize_targets=args.randomize_targets,
            tie_targets=args.tie_targets,
            scale=args.scale,
            use_step_api=args.use_step_api,
        )
        example.run()
