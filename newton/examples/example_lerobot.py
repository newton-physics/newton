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

###########################################################################
# Example H1
#
# This example loads the Unitree H1 humanoid from MJCF and shows how to
# animate the robot with forward kinematics only. It is similar to the
#
###########################################################################

"""Example H1 folding demonstration.

This example loads the Unitree H1 humanoid from MJCF and shows how to
animate the robot based on a loaded trajectory with forward kinematics only.
"""

import math, time
from collections.abc import Sequence
from datasets import load_dataset # Added for LeRobot dataset

import numpy as np
import warp as wp

import newton
import newton.core.articulation
import newton.examples
import newton.utils


class Example:
    def __init__(self, stage_path: str | None = "example_lerobot.usd", num_envs: int = 1):
        self.num_envs = num_envs

        articulation_builder = newton.ModelBuilder()
        articulation_builder.default_shape_cfg.density = 100.0
        articulation_builder.default_joint_cfg.armature = 0.1
        articulation_builder.default_body_armature = 0.1

        newton.utils.parse_mjcf(
            newton.examples.get_asset("h1_description/mjcf/h1_with_hand.xml"),
            articulation_builder,
            floating=True,
            armature_scale=1.0,
            scale=1.0,
        )

        # Load LeRobot dataset
        dataset = load_dataset("lerobot/unitreeh1_fold_clothes", split="train")
        first_episode_states = []
        for sample in dataset:
            if sample['episode_index'] == 0:
                first_episode_states.append(sample['observation.state'])
            elif sample['episode_index'] > 0 and len(first_episode_states) > 0:
                 break
        self.trajectory_q = np.array(first_episode_states, dtype=np.float32)
        
        # Define joint mapping from trajectory (19 DOFs) to h1_with_hand model
        # The trajectory contains 19 values corresponding to the main body joints:
        # [left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle,
        #  right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle,
        #  torso, left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow,
        #  right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow]
        
        # In h1_with_hand.xml, the joint order (after 7 floating base DOFs) is:
        # 0: left_hip_yaw_joint, 1: left_hip_roll_joint, 2: left_hip_pitch_joint,
        # 3: left_knee_joint, 4: left_ankle_joint, 5: right_hip_yaw_joint,
        # 6: right_hip_roll_joint, 7: right_hip_pitch_joint, 8: right_knee_joint,
        # 9: right_ankle_joint, 10: torso_joint, 11: left_shoulder_pitch_joint,
        # 12: left_shoulder_roll_joint, 13: left_shoulder_yaw_joint, 14: left_elbow_joint,
        # 15: left_hand_joint, 16-27: left hand finger joints,
        # 28: right_shoulder_pitch_joint, 29: right_shoulder_roll_joint,
        # 30: right_shoulder_yaw_joint, 31: right_elbow_joint,
        # 32: right_hand_joint, 33-44: right hand finger joints
        
        self.joint_mapping = [
            7,   # left_hip_yaw_joint
            8,   # left_hip_roll_joint  
            9,   # left_hip_pitch_joint
            10,  # left_knee_joint
            11,  # left_ankle_joint
            12,  # right_hip_yaw_joint
            13,  # right_hip_roll_joint
            14,  # right_hip_pitch_joint
            15,  # right_knee_joint
            16,  # right_ankle_joint
            17,  # torso_joint
            18,  # left_shoulder_pitch_joint
            19,  # left_shoulder_roll_joint
            20,  # left_shoulder_yaw_joint
            21,  # left_elbow_joint
            # Skip left_hand_joint (22) and left finger joints (23-34)
            35,  # right_shoulder_pitch_joint
            36,  # right_shoulder_roll_joint
            37,  # right_shoulder_yaw_joint
            38,  # right_elbow_joint
            # Skip right_hand_joint and right finger joints
        ]
        
        # Set initial joint positions using proper mapping
        for i, traj_value in enumerate(self.trajectory_q[0]):
            if i < len(self.joint_mapping):
                articulation_builder.joint_q[self.joint_mapping[i]] = traj_value
        
        # Set hand joints to neutral positions (0.0)
        # Left hand joint (index 22) and right hand joint (index 39)
        articulation_builder.joint_q[22] = 0.0  # left_hand_joint
        articulation_builder.joint_q[39] = 0.0  # right_hand_joint
        
        # Set all finger joints to neutral positions (indices 23-34 for left, 40-51 for right)
        for i in range(23, 35):  # left hand finger joints
            articulation_builder.joint_q[i] = 0.0
        for i in range(40, 52):  # right hand finger joints
            articulation_builder.joint_q[i] = 0.0

        self.env_coord_count = len(articulation_builder.joint_q)
        self.current_frame_index = 0

        builder = newton.ModelBuilder()

        offsets = newton.examples.compute_env_offsets(self.num_envs, env_offset=(1.0, 0.0, 2.0))
        for i in range(self.num_envs):
            builder.add_builder(articulation_builder, xform=wp.transform(offsets[i], wp.quat_identity()))

        fps = 50  # Match dataset FPS
        self.frame_dt = 1.0 / fps
        self.sim_time = 0.0

        self.model = builder.finalize()
        self.model.ground = False

        self.renderer = None
        if stage_path:
            self.renderer = newton.utils.SimRendererOpenGL(path=stage_path, model=self.model, scaling=1.0)

        self.state = self.model.state()
        newton.core.articulation.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state)

    def step(self):
        current_pose_from_trajectory = None
        if self.current_frame_index < self.trajectory_q.shape[0]:
            current_pose_from_trajectory = self.trajectory_q[self.current_frame_index]
        else:
            # End of trajectory: hold the last pose
            current_pose_from_trajectory = self.trajectory_q[-1]

        q_all_envs = self.model.joint_q.numpy()
        
        for e in range(self.num_envs):
            env_q_offset = e * self.env_coord_count
            
            # Apply trajectory values using proper joint mapping
            for i, traj_value in enumerate(current_pose_from_trajectory):
                if i < len(self.joint_mapping):
                    joint_idx = env_q_offset + self.joint_mapping[i]
                    q_all_envs[joint_idx] = traj_value
            
            # Keep hand joints at neutral positions
            q_all_envs[env_q_offset + 22] = 0.0  # left_hand_joint
            q_all_envs[env_q_offset + 39] = 0.0  # right_hand_joint
            
            # Keep all finger joints at neutral positions
            for i in range(23, 35):  # left hand finger joints
                q_all_envs[env_q_offset + i] = 0.0
            for i in range(40, 52):  # right hand finger joints
                q_all_envs[env_q_offset + i] = 0.0
            
        self.model.joint_q.assign(q_all_envs)
        
        self.current_frame_index += 1

        newton.core.articulation.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state)
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return
        self.renderer.begin_frame(self.sim_time)
        self.renderer.render(self.state)
        self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_h1_folding.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=600, help="Total number of frames.")
    parser.add_argument("--num_envs", type=int, default=1, help="Total number of simulated environments.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_envs=args.num_envs)
        time.sleep(1.0)
        for _ in range(args.num_frames):
            example.step()
            example.render()
            time.sleep(example.frame_dt)
        if example.renderer:
            example.renderer.save()
