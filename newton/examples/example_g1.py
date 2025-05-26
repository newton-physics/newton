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
# Example G1
#
# This example loads the Unitree G1 humanoid from MJCF and shows how to
# animate the robot with forward kinematics only using sinusoidal motion.
#
###########################################################################

"""Example G1 sinusoidal motion demonstration.

This example loads the Unitree G1 humanoid from MJCF and shows how to
animate the robot using sinusoidal motion patterns with forward kinematics only.
"""

import math
import time
from collections.abc import Sequence

import numpy as np
import warp as wp

import newton
import newton.core.articulation
import newton.examples
import newton.utils


class Example:
    def __init__(self, stage_path: str | None = "example_g1.usd", num_envs: int = 1):
        self.num_envs = num_envs

        articulation_builder = newton.ModelBuilder()
        articulation_builder.default_shape_cfg.density = 100.0
        articulation_builder.default_joint_cfg.armature = 0.1
        articulation_builder.default_body_armature = 0.1

        newton.utils.parse_mjcf(
            newton.examples.get_asset("g1/g1_body29_hand14.xml"),
            articulation_builder,
            floating=True,
            armature_scale=1.0,
            scale=1.0,
        )

        # Define joint mapping for sinusoidal motion
        # After 7 floating base DOFs, the joint order in g1_body29_hand14.xml is:
        # 7: left_hip_pitch_joint, 8: left_hip_roll_joint, 9: left_hip_yaw_joint,
        # 10: left_knee_joint, 11: left_ankle_pitch_joint, 12: left_ankle_roll_joint,
        # 13: right_hip_pitch_joint, 14: right_hip_roll_joint, 15: right_hip_yaw_joint,
        # 16: right_knee_joint, 17: right_ankle_pitch_joint, 18: right_ankle_roll_joint,
        # 19: waist_yaw_joint, 20: waist_roll_joint, 21: waist_pitch_joint,
        # 22: left_shoulder_pitch_joint, 23: left_shoulder_roll_joint, 24: left_shoulder_yaw_joint,
        # 25: left_elbow_joint, 26: left_wrist_roll_joint, 27: left_wrist_pitch_joint,
        # 28: left_wrist_yaw_joint, 29-35: left hand joints,
        # 36: right_shoulder_pitch_joint, 37: right_shoulder_roll_joint, 38: right_shoulder_yaw_joint,
        # 39: right_elbow_joint, 40: right_wrist_roll_joint, 41: right_wrist_pitch_joint,
        # 42: right_wrist_yaw_joint, 43-49: right hand joints

        # Define joints to animate with sinusoidal motion
        self.animated_joints = {
            # Waist joints - subtle torso movement
            "waist_yaw": {"index": 19, "amplitude": 0.1, "frequency": 0.5, "phase": 0.0},
            "waist_pitch": {"index": 21, "amplitude": 0.05, "frequency": 0.3, "phase": 0.0},
            
            # Leg joints - walking-like motion
            "left_hip_pitch": {"index": 7, "amplitude": 0.3, "frequency": 1.0, "phase": 0.0},
            "right_hip_pitch": {"index": 13, "amplitude": 0.3, "frequency": 1.0, "phase": math.pi},
            "left_knee": {"index": 10, "amplitude": 0.4, "frequency": 1.0, "phase": 0.5},
            "right_knee": {"index": 16, "amplitude": 0.4, "frequency": 1.0, "phase": 0.5 + math.pi},
            
            # Arm joints - coordinated arm swinging
            "left_shoulder_pitch": {"index": 22, "amplitude": 0.4, "frequency": 0.8, "phase": 0.0},
            "right_shoulder_pitch": {"index": 36, "amplitude": 0.4, "frequency": 0.8, "phase": math.pi},
            "left_elbow": {"index": 25, "amplitude": 0.3, "frequency": 0.8, "phase": 0.2},
            "right_elbow": {"index": 39, "amplitude": 0.3, "frequency": 0.8, "phase": 0.2 + math.pi},
            
            # Hand joints - simple finger movements
            "left_hand_thumb_0": {"index": 29, "amplitude": 0.2, "frequency": 1.5, "phase": 0.0},
            "left_hand_middle_0": {"index": 32, "amplitude": 0.3, "frequency": 1.2, "phase": 0.3},
            "right_hand_thumb_0": {"index": 43, "amplitude": 0.2, "frequency": 1.5, "phase": math.pi},
            "right_hand_middle_0": {"index": 46, "amplitude": 0.3, "frequency": 1.2, "phase": 0.3 + math.pi},
        }

        # Set initial joint positions to neutral
        for joint_info in self.animated_joints.values():
            articulation_builder.joint_q[joint_info["index"]] = 0.0

        self.env_coord_count = len(articulation_builder.joint_q)

        builder = newton.ModelBuilder()

        offsets = newton.examples.compute_env_offsets(self.num_envs, env_offset=(1.0, 0.0, 2.0))
        for i in range(self.num_envs):
            builder.add_builder(articulation_builder, xform=wp.transform(offsets[i], wp.quat_identity()))

        fps = 60
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
        q_all_envs = self.model.joint_q.numpy()
        
        for e in range(self.num_envs):
            env_q_offset = e * self.env_coord_count
            
            # Apply sinusoidal motion to selected joints
            for joint_name, joint_info in self.animated_joints.items():
                joint_idx = env_q_offset + joint_info["index"]
                amplitude = joint_info["amplitude"]
                frequency = joint_info["frequency"]
                phase = joint_info["phase"]
                
                # Calculate sinusoidal position
                sin_value = amplitude * math.sin(2 * math.pi * frequency * self.sim_time + phase)
                q_all_envs[joint_idx] = sin_value
            
        self.model.joint_q.assign(q_all_envs)
        
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
        default="example_g1.usd",
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
