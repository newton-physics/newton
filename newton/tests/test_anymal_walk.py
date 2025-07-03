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

"""Tests that anymal can walk with the provided policy."""

import unittest

import numpy as np
import warp as wp

from newton.examples.example_anymal_c_walk import Example


class TestAnymalCWalk(unittest.TestCase):
    def test_anymal_walk_policy(self):
        example = Example(stage_path=None, render=False)
        num_test_steps = 1000
        for step_num in range(num_test_steps):
            if example.use_cuda_graph:
                wp.capture_launch(example.graph)
            else:
                example.simulate()

            root_pos = wp.to_torch(example.state_0.joint_q[:3]).detach().cpu().numpy()
            root_height = root_pos[1]
            root_quat = wp.to_torch(example.state_0.joint_q[3:7]).detach().cpu().numpy()

            x, y, z, w = root_quat[0], root_quat[1], root_quat[2], root_quat[3]
            R = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                ]
            )

            joint_qd_linear = wp.to_torch(example.state_0.joint_qd[3:6]).detach().cpu().numpy()
            joint_qd_angular = wp.to_torch(example.state_0.joint_qd[0:3]).detach().cpu().numpy()

            joint_qd_linear_corrected = joint_qd_linear - np.cross(root_pos, joint_qd_angular)
            vel_body_joint_qd_corrected = R.T @ joint_qd_linear_corrected
            height_threshold = 0.3
            has_fallen = root_height < height_threshold

            if has_fallen:
                self.fail(f"Robot fallen, Step {step_num} - Height: {root_height:.3f}m (threshold: {height_threshold}m)")

            if step_num % 100 == 0 and step_num != 0:
                self.assertGreater(
                    vel_body_joint_qd_corrected[0],
                    0.5,
                    f"Step {step_num}: Forward velocity too low: {vel_body_joint_qd_corrected[0]:.3f} m/s (expected > 0.5)",
                )

            example.controller.get_control(example.state_0, example.control)
            example.sim_step += 1
            example.sim_time += example.frame_dt


if __name__ == "__main__":
    unittest.main()
