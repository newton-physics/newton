# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.enable_backward = False
wp.config.quiet = True

import newton
import newton.examples
from newton.examples.selection.example_selection_cartpole import Example


class FastExampleSelectionCartpoleMuJoCo:
    repeat = 10
    number = 1

    def setup(self):
        self.num_frames = 200
        if hasattr(newton.examples, "default_args") and hasattr(Example, "create_parser"):
            args = newton.examples.default_args(Example.create_parser())
            self.example = Example(newton.viewer.ViewerNull(num_frames=self.num_frames), args)
        else:
            self.example = Example(
                viewer=newton.viewer.ViewerNull(num_frames=self.num_frames), world_count=16, verbose=False
            )

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastExampleSelectionCartpoleMuJoCo": FastExampleSelectionCartpoleMuJoCo,
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
