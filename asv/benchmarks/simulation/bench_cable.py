# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
import inspect

import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

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


class FastExampleCableViscoelasticRelease:
    """Benchmark VBD cable bending with persistent SLS material state."""

    timeout = 300
    number = 1
    rounds = 2
    repeat = 3

    def setup(self):
        try:
            module = importlib.import_module("newton.examples.cable.example_cable_viscoelastic_release")
        except ModuleNotFoundError as error:
            raise SkipNotImplemented from error
        if not hasattr(newton.examples, "default_args"):
            raise SkipNotImplemented

        self.num_frames = 100
        args = newton.examples.default_args(module.Example.create_parser())
        args.release_time = 0.1
        self.example = module.Example(ViewerNull(num_frames=self.num_frames), args)
        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastExampleCablePile": FastExampleCablePile,
        "FastExampleCableViscoelasticRelease": FastExampleCableViscoelasticRelease,
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
