# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import inspect

import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.enable_backward = False
wp.config.quiet = True

import newton.examples
from newton.examples.cable.example_cable_pile import Example as ExampleCablePile
from newton.viewer import ViewerNull


def _supports_cable_pile_size_args():
    parameters = inspect.signature(ExampleCablePile).parameters
    return "layers" in parameters and "lanes_per_layer" in parameters


class FastExampleCablePile:
    number = 1
    rounds = 2
    repeat = 2

    def setup(self):
        self.num_frames = 30
        if hasattr(newton.examples, "default_args"):
            args = newton.examples.default_args()
        else:
            args = None
        viewer = ViewerNull(num_frames=self.num_frames)
        if _supports_cable_pile_size_args():
            self.example = ExampleCablePile(viewer, args, layers=4, lanes_per_layer=10)
        else:
            self.example = ExampleCablePile(viewer, args)
        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        newton.examples.run(self.example, args=None)

        wp.synchronize_device()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastExampleCablePile": FastExampleCablePile,
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
