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

import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

from newton.examples.example_cloth_self_contact import Example as ExampleClothSelfContact
from newton.examples.example_robot_manipulating_cloth import Example as ExampleClothManipulation


class VBDSpeedTestSelfContact:
    number = 5

    def setup(self):
        wp.init()

        if wp.get_cuda_device_count() > 0:
            with wp.ScopedDevice("cuda"):
                self.example = ExampleClothSelfContact(stage_path=None, num_frames=300)
        else:
            self.example = None

    @skip_benchmark_if(wp.get_cuda_device_count() == 0 or wp.context.runtime.driver_version < (12, 3))
    def time_run_example_cloth_self_contact(self):
        with wp.ScopedDevice("cuda"):
            self.example.run()


class VBDSpeedClothManipulation:
    timeout = 180
    number = 3

    def setup(self):
        wp.init()

        if wp.get_cuda_device_count() > 0:
            with wp.ScopedDevice("cuda"):
                self.example = ExampleClothManipulation(stage_path=None, num_frames=300)
        else:
            self.example = None

    @skip_benchmark_if(wp.get_cuda_device_count() == 0 or wp.context.runtime.driver_version < (12, 3))
    def time_run_example_cloth_manipulation(self):
        with wp.ScopedDevice("cuda"):
            for frame_idx in range(self.example.num_frames):
                self.example.step()

                if self.example.cloth_solver and not (frame_idx % 10):
                    self.example.cloth_solver.rebuild_bvh(self.example.state_0)
                    self.example.capture_cuda_graph()
