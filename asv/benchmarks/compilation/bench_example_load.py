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


class SlowExampleRobotAnymal:
    warmup_time = 0
    repeat = 2
    number = 1
    timeout = 600

    def setup(self):
        wp.build.clear_lto_cache()
        wp.build.clear_kernel_cache()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_load(self):
        """Time the amount of time it takes to load and run one frame of the example."""

        command = [
            sys.executable,
            "-m",
            "newton.examples.robot.example_robot_anymal_c_walk",
            "--num-frames",
            "1",
            "--viewer",
            "null",
        ]

        # Run the script as a subprocess
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        print(f"Output:\n{result.stdout}\n{result.stderr}")


class SlowExampleRobotCartpole:
    warmup_time = 0
    repeat = 2
    number = 1
    timeout = 600

    def setup(self):
        wp.build.clear_lto_cache()
        wp.build.clear_kernel_cache()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_load(self):
        """Time the amount of time it takes to load and run one frame of the example."""

        command = [
            sys.executable,
            "-m",
            "newton.examples.robot.example_robot_cartpole",
            "--num-frames",
            "1",
            "--viewer",
            "null",
        ]

        # Run the script as a subprocess
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        print(f"Output:\n{result.stdout}\n{result.stderr}")


class SlowExampleClothFranka:
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
            "newton.examples.cloth.example_cloth_franka",
            "--num-frames",
            "1",
            "--viewer",
            "null",
        ]

        # Run the script as a subprocess
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        print(f"Output:\n{result.stdout}\n{result.stderr}")


class SlowExampleClothTwist:
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
            "newton.examples.cloth.example_cloth_twist",
            "--num-frames",
            "1",
            "--viewer",
            "null",
        ]

        # Run the script as a subprocess
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        print(f"Output:\n{result.stdout}\n{result.stderr}")


class SlowExampleRobotHumanoid:
    warmup_time = 0
    repeat = 2
    number = 1
    timeout = 600

    def setup(self):
        wp.build.clear_lto_cache()
        wp.build.clear_kernel_cache()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_load(self):
        """Time the amount of time it takes to load and run one frame of the example."""

        command = [
            sys.executable,
            "-m",
            "newton.examples.robot.example_robot_humanoid",
            "--num-frames",
            "1",
            "--viewer",
            "null",
        ]

        # Run the script as a subprocess
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        print(f"Output:\n{result.stdout}\n{result.stderr}")


class SlowExampleBasicUrdf:
    warmup_time = 0
    repeat = 2
    number = 1
    timeout = 600

    def setup(self):
        wp.build.clear_lto_cache()
        wp.build.clear_kernel_cache()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_load(self):
        """Time the amount of time it takes to load and run one frame of the example."""

        command = [
            sys.executable,
            "-m",
            "newton.examples.basic.example_basic_urdf",
            "--num-frames",
            "1",
            "--viewer",
            "null",
        ]

        # Run the script as a subprocess
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        print(f"Output:\n{result.stdout}\n{result.stderr}")


if __name__ == "__main__":
    from newton.utils import run_benchmark

    run_benchmark(SlowExampleBasicUrdf)
    run_benchmark(SlowExampleRobotAnymal)
    run_benchmark(SlowExampleRobotCartpole)
    run_benchmark(SlowExampleRobotHumanoid)
    run_benchmark(SlowExampleClothFranka)
    run_benchmark(SlowExampleClothTwist)
