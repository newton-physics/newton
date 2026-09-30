# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

import newton


class _InitializeArticulationView:
    param_names = ["world_count"]

    rounds = 1
    repeat = 3
    number = 1
    min_run_count = 1

    def setup(self, world_count):
        robot = newton.ModelBuilder()
        root = robot.add_link(label="robot/root")
        tip = robot.add_link(label="robot/tip")
        robot.add_shape_box(root, hx=0.1, hy=0.1, hz=0.1)
        robot.add_shape_sphere(tip, radius=0.1)
        root_joint = robot.add_joint_free(child=root)
        tip_joint = robot.add_joint_revolute(root, tip)
        robot.add_articulation([root_joint, tip_joint], label="robot")

        scene = newton.ModelBuilder()
        scene.replicate(robot, world_count=world_count)
        self.model = scene.finalize()
        self.pattern = "robot"

        newton.selection.ArticulationView(self.model, self.pattern)
        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_initialize_articulation_view(self, world_count):
        _view = newton.selection.ArticulationView(self.model, self.pattern)
        wp.synchronize_device()

    def teardown(self, world_count):
        del self.model


class KpiInitializeArticulationView(_InitializeArticulationView):
    params = ([8192],)


class FastInitializeArticulationView(_InitializeArticulationView):
    params = ([1, 64],)


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "KpiInitializeArticulationView": KpiInitializeArticulationView,
        "FastInitializeArticulationView": FastInitializeArticulationView,
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
