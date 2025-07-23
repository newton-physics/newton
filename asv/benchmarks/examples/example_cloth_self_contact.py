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

from newton.examples.example_cloth_self_contact import Example as ExampleClothSelfContact


class VBDSolverLoad:
    warmup_time = 0
    repeat = 2
    number = 1

    def setup(self):
        wp.build.clear_lto_cache()
        wp.build.clear_kernel_cache()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_load(self):
        """Time the amount of time it takes to load and run one frame of the example."""

        command = [
            sys.executable,
            "-m",
            "newton.examples.example_cloth_self_contact",
            "--stage-path",
            "None",
            "--num-frames",
            "1",
        ]

        # Run the script as a subprocess
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        print(f"Output:\n{result.stdout}\n{result.stderr}")


class VBDSolverSimulate:
    repeat = 5
    number = 1

    def setup(self):
        self.num_frames = 100
        self.example = ExampleClothSelfContact(stage_path=None)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for i in range(self.num_frames):
            self.example.step()

            if (
                i != 0
                and not i % self.example.bvh_rebuild_frames
                and self.example.use_cuda_graph
                and self.example.solver.handle_self_contact
            ):
                self.example.solver.rebuild_bvh(self.example.state_0)
                with wp.ScopedCapture() as capture:
                    self.example.simulate_substeps()
                self.example.cuda_graph = capture.graph
