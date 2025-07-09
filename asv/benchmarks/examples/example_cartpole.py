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

import subprocess
import sys

import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

from newton.examples.example_cartpole import Example


class ModelMemory:
    params = [128, 256]

    def setup(self, num_envs):
        wp.init()

    def peakmem_initialize_model_cpu(self, num_envs):
        with wp.ScopedDevice("cpu"):
            _example = Example(stage_path=None, num_envs=num_envs)


class InitializeModel:
    params = [64, 128]

    number = 10

    def setup(self, num_envs):
        wp.init()

    def time_initialize_model(self, num_envs):
        with wp.ScopedDevice("cpu"):
            _example = Example(stage_path=None, num_envs=num_envs)


class ExampleLoad:
    warmup_time = 0
    repeat = 2
    number = 1
    timeout = 600

    def setup(self):
        wp.build.clear_lto_cache()
        wp.build.clear_kernel_cache()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_load(self):
        """Time the amount of time it takes to load and run one frame of the Cartpole example."""

        command = [
            sys.executable,
            "-m",
            "newton.examples.example_cartpole",
            "--stage-path",
            "None",
            "--num_frames",
            "1",
            "--no-use-cuda-graph",
        ]

        # Run the script as a subprocess
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        print(f"Output:\n{result.stdout}\n{result.stderr}")


class MuJoCoSolverSimulate:
    repeat = 10
    number = 1

    def setup(self):
        self.num_frames = 200
        self.example = Example(stage_path=None, num_envs=8, use_cuda_graph=True)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0 or wp.context.runtime.driver_version < (12, 3))
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()
