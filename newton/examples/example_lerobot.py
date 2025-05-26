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
            newton.examples.get_asset("unitree_h1/h1.xml"),
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
        
        # Set initial joint positions from the first frame of the trajectory
        articulation_builder.joint_q[7:] = self.trajectory_q[0].tolist()

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
        num_actuated_joints = 19 # Based on dataset observation.state shape
        
        for e in range(self.num_envs):
            env_q_offset = e * self.env_coord_count
            # The first 7 elements are for the floating base (pos, quat)
            # The next 19 are the actuated joints
            start_idx = env_q_offset + 7
            end_idx = start_idx + num_actuated_joints
            q_all_envs[start_idx:end_idx] = current_pose_from_trajectory
            
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
