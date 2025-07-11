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

from newton.examples.example_cloth_self_contact import Example as ExampleClothSelfContact
from newton.examples.example_robot_manipulating_cloth import Example as ExampleClothManipulation


class VBDSpeedTestSelfContact:
    params = [
        60,
    ]

    number = 10

    def setup(self, num_frames):
        wp.init()

        with wp.ScopedDevice("cpu"):
            self.example = ExampleClothSelfContact(stage_path=None, num_frames=num_frames)

    def time_run_example_cloth_self_contact(self, num_frames):
        with wp.ScopedDevice("cpu"):
            self.example.run()


class VBDSpeedClothManipulation:
    params = [
        120,
    ]

    number = 10

    def setup(self, num_frames):
        wp.init()

        with wp.ScopedDevice("cpu"):
            self.example = ExampleClothManipulation(stage_path=None, num_frames=num_frames)

    def time_run_example_cloth_manipulation(self, num_frames):
        with wp.ScopedDevice("cpu"):
            self.example.run()
