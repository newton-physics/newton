# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from benchmark_metrics import (
    _SimulationMetricTracksUnparameterized,
    collect_simulation_metrics_synchronized,
    compute_gpu_memory_usage,
    validate_simulation_workload,
)

import newton
import newton.examples
from newton.examples.basic.example_basic_urdf import Example


def _create_example(num_frames, world_count):
    if hasattr(newton.examples, "default_args") and hasattr(Example, "create_parser"):
        args = newton.examples.default_args(Example.create_parser())
        args.world_count = world_count
        return Example(newton.viewer.ViewerNull(num_frames=num_frames), args)
    return Example(newton.viewer.ViewerNull(num_frames=num_frames), world_count)


class FastExampleQuadrupedXPBD:
    repeat = 10
    number = 1

    def setup(self):
        self.num_frames = 1000
        self.example = _create_example(self.num_frames, world_count=200)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


class FastMetricsExampleQuadrupedXPBD(_SimulationMetricTracksUnparameterized):
    num_frames = 1000
    samples = 1
    world_count = 200

    def setup_cache(self):
        wp.synchronize_device()
        device = wp.get_device()
        free_memory_before = device.free_memory
        return collect_simulation_metrics_synchronized(
            create_workload=lambda: _create_example(self.num_frames, self.world_count),
            world_count=self.world_count,
            num_frames=self.num_frames,
            samples=self.samples,
            synchronize=wp.synchronize_device,
            memory_usage_bytes=lambda workload: compute_gpu_memory_usage(device, free_memory_before),
            validate=lambda workload: validate_simulation_workload(
                workload,
                max_linear_speed=0.3,
                max_angular_speed=0.3,
            ),
        )


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastExampleQuadrupedXPBD": FastExampleQuadrupedXPBD,
        "FastMetricsExampleQuadrupedXPBD": FastMetricsExampleQuadrupedXPBD,
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
