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
    collect_simulation_metrics,
    validate_simulation_state,
)

_NUM_FRAMES = 50
_MIN_BASE_HEIGHT = 0.4
_MIN_FORWARD_PROGRESS = 0.25


def _create_example(num_frames):
    import newton  # noqa: PLC0415
    import newton.examples  # noqa: PLC0415
    from newton.examples.robot.example_robot_anymal_c_walk import Example  # noqa: PLC0415

    if hasattr(newton.examples, "default_args"):
        args = newton.examples.default_args()
    else:
        args = None
    return Example(newton.viewer.ViewerNull(num_frames=num_frames), args)


def _validate_workload(workload):
    validate_simulation_state(
        workload.state_0,
        max_linear_speed=10.0,
        max_angular_speed=50.0,
    )
    root_position = workload.state_0.joint_q.numpy()[:3]
    if root_position[2] < _MIN_BASE_HEIGHT:
        raise RuntimeError(f"ANYmal base height is too low after {_NUM_FRAMES} frames: {root_position[2]:.3f} m")
    if root_position[1] < _MIN_FORWARD_PROGRESS:
        raise RuntimeError(
            f"ANYmal made insufficient forward progress after {_NUM_FRAMES} frames: {root_position[1]:.3f} m"
        )


class FastExampleAnymalPretrained:
    repeat = 3
    number = 1

    def setup(self):
        self.num_frames = _NUM_FRAMES
        self.example = _create_example(self.num_frames)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


class FastMetricsExampleAnymalPretrained(_SimulationMetricTracksUnparameterized):
    num_frames = _NUM_FRAMES
    samples = 3
    world_count = 1

    def setup_cache(self):
        if wp.get_cuda_device_count() == 0:
            return None
        return collect_simulation_metrics(
            create_workload=lambda: _create_example(self.num_frames),
            world_count=self.world_count,
            num_frames=self.num_frames,
            samples=self.samples,
            synchronize=wp.synchronize_device,
            validate=_validate_workload,
        )


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastExampleAnymalPretrained": FastExampleAnymalPretrained,
        "FastMetricsExampleAnymalPretrained": FastMetricsExampleAnymalPretrained,
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
