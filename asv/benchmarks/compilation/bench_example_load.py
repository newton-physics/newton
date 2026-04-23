# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

import warp as wp

wp.config.enable_backward = False
wp.config.quiet = True

from asv_runner.benchmarks.mark import skip_benchmark_if


class SlowExampleRobotAnymal:
    warmup_time = 0
    repeat = 2
    number = 1
    timeout = 600

    def setup(self):
        wp.clear_lto_cache()
        wp.clear_kernel_cache()

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
        subprocess.run(command, capture_output=True, text=True, check=True)


class SlowExampleRobotCartpole:
    warmup_time = 0
    repeat = 2
    number = 1
    timeout = 600

    def setup(self):
        wp.clear_lto_cache()
        wp.clear_kernel_cache()

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
        subprocess.run(command, capture_output=True, text=True, check=True)


class SlowExampleClothFranka:
    warmup_time = 0
    repeat = 2
    number = 1

    def setup(self):
        wp.clear_lto_cache()
        wp.clear_kernel_cache()

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
        subprocess.run(command, capture_output=True, text=True, check=True)


class SlowExampleClothTwist:
    warmup_time = 0
    repeat = 2
    number = 1

    def setup(self):
        wp.clear_lto_cache()
        wp.clear_kernel_cache()

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
        subprocess.run(command, capture_output=True, text=True, check=True)


class SlowExampleBasicUrdf:
    warmup_time = 0
    repeat = 2
    number = 1
    timeout = 600

    def setup(self):
        wp.clear_lto_cache()
        wp.clear_kernel_cache()

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
        subprocess.run(command, capture_output=True, text=True, check=True)


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "SlowExampleBasicUrdf": SlowExampleBasicUrdf,
        "SlowExampleRobotAnymal": SlowExampleRobotAnymal,
        "SlowExampleRobotCartpole": SlowExampleRobotCartpole,
        "SlowExampleClothFranka": SlowExampleClothFranka,
        "SlowExampleClothTwist": SlowExampleClothTwist,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b",
        "--bench",
        default=None,
        action="append",
        choices=benchmark_list.keys(),
        help="Run a specific benchmark; may be repeated to run multiple (e.g., --bench A --bench B).",
    )
    args = parser.parse_known_args()[0]

    if args.bench is None:
        benchmarks = benchmark_list.keys()
    else:
        benchmarks = args.bench

    for key in benchmarks:
        benchmark = benchmark_list[key]
        run_benchmark(benchmark)
