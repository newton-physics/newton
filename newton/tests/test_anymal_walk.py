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


class TestAnymalCWalk(unittest.TestCase):
    def test_anymal_walk_policy(self):
        try:
            from newton.examples.example_anymal_c_walk import Example  # noqa: PLC0415
        except ImportError:
            self.skipTest("Example import failed - skipping test")

        example = Example(stage_path=None, headless=True)
        num_test_steps = 1000
        for step_num in range(num_test_steps):
            if example.use_cuda_graph:
                wp.capture_launch(example.graph)
            else:
                example.simulate()

            root_pos = example.state_0.joint_q[:3].numpy()
            root_height = root_pos[1]

            qd_linear = example.state_0.joint_qd[3:6].numpy()
            qd_angular = example.state_0.joint_qd[0:3].numpy()

            qd_linear_corrected = qd_linear - np.cross(root_pos, qd_angular)
            height_threshold = 0.3
            has_fallen = root_height < height_threshold

            if has_fallen:
                self.fail(f"Robot fell, Step {step_num} - Height: {root_height:.3f}m (threshold: {height_threshold}m)")

            if step_num % 100 == 0 and step_num != 0:
                self.assertGreater(
                    qd_linear_corrected[0],
                    0.5,
                    f"Step {step_num}: Forward velocity too low: {qd_linear_corrected[0]:.3f} m/s (expected > 0.5)",
                )

            example.controller.get_control(example.state_0)
            example.sim_step += 1
            example.sim_time += example.frame_dt


if __name__ == "__main__":
    unittest.main()
