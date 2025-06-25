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

from newton.examples.example_cartpole import Example


class CartpoleMemory:
    params = [128, 256]

    def setup(self, num_envs):
        wp.init()

    def peakmem_initialize_model(self, num_envs):
        with wp.ScopedDevice("cpu"):
            _example = Example(stage_path=None, num_envs=num_envs)


class CartpoleModel:
    params = [64, 128]

    number = 10

    def setup(self, num_envs):
        wp.init()

    def time_initialize_model(self, num_envs):
        with wp.ScopedDevice("cpu"):
            _example = Example(stage_path=None, num_envs=num_envs)
