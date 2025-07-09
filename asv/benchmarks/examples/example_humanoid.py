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

from newton.examples.example_humanoid import Example


class ExampleLoad:
    warmup_time = 0
    repeat = 2
    number = 1
    timeout = 600

    def setup(self):
        wp.build.clear_lto_cache()
        wp.build.clear_kernel_cache()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0 or wp.context.runtime.driver_version < (12, 3))
    def time_load(self):
        command = [
            sys.executable,
            "-m",
            "newton.examples.example_humanoid",
            "--stage-path",
            "None",
            "--num-frames",
            "1",
            "--headless",
            "--no-use-cuda-graph",
        ]

        # Run the script as a subprocess
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        print(f"Output:\n{result.stdout}\n{result.stderr}")


class MuJoCoSolverSimulate:
    repeat = 5
    number = 1

    def setup(self):
        self.num_frames = 100
        self.example = Example(stage_path=None, num_envs=8, use_cuda_graph=True)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0 or wp.context.runtime.driver_version < (12, 3))
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()
