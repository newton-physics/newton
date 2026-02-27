# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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
# Example Robot Analog Digital Clock
#
# Shows how to load a single-world USD asset directly with
# newton.ModelBuilder.add_usd().
#
# Command: python -m newton.examples robot_analog_digital_clock
#
###########################################################################

from typing import Any

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer: Any):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.viewer = viewer

        builder = newton.ModelBuilder()
        builder.add_usd(newton.examples.get_asset("AnalogDigitalClock.usda"), xform=wp.transform(wp.vec3(0, 0.8, 0.5), wp.quat_rpy(0.3, 0.2, 0.6)))

        self.model = builder.finalize(skip_validation_joints=True)
        self.state_0 = self.model.state()

        # Evaluate once so the viewer has body transforms to log.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)

    def step(self):
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if self.model.body_count == 0:
            raise ValueError("Clock USD did not produce any bodies.")
        if self.model.shape_count == 0:
            raise ValueError("Clock USD did not produce any collision or render shapes.")


if __name__ == "__main__":
    viewer, args = newton.examples.init()

    example = Example(viewer)

    newton.examples.run(example, args)
