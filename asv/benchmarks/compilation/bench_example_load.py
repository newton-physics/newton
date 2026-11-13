# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import sys
import tempfile

import warp as wp

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

from asv_runner.benchmarks.mark import skip_benchmark_if


class _ExampleLoad:
    """Time a cold start of an example: imports, kernel compilation, and one frame."""

    module = None
    warmup_time = 0
    repeat = 2
    number = 1
    timeout = 600

    def setup(self):
        # An empty private cache forces compilation without wiping the shared one.
        self.cache_dir = tempfile.mkdtemp(prefix="newton-asv-warp-cache-")

    def teardown(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_load(self):
        command = [sys.executable, "-m", self.module, "--num-frames", "1", "--viewer", "null"]
        env = {**os.environ, "WARP_CACHE_PATH": self.cache_dir}
        subprocess.run(command, capture_output=True, text=True, check=True, env=env)


class SlowExampleRobotAnymal(_ExampleLoad):
    module = "newton.examples.robot.example_robot_anymal_c_walk"


class SlowExampleRobotCartpole(_ExampleLoad):
    module = "newton.examples.robot.example_robot_cartpole"


class SlowExampleClothFranka(_ExampleLoad):
    module = "newton.examples.cloth.example_cloth_franka"


class SlowExampleClothTwist(_ExampleLoad):
    module = "newton.examples.cloth.example_cloth_twist"


class SlowExampleBasicUrdf(_ExampleLoad):
    module = "newton.examples.basic.example_basic_urdf"


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
