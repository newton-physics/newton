# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import warp as wp

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

from asv_runner.benchmarks.mark import skip_benchmark_if

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from benchmark_mujoco import Example

from newton.utils import EventTracer


class _KpiBenchmark:
    """Utility base class for KPI benchmarks."""

    param_names = ["world_count"]
    num_frames = None
    params = None
    robot = None
    samples = None
    ls_iteration = None
    random_init = None
    environment = "None"

    def setup(self, world_count):
        if not hasattr(self, "builder") or self.builder is None:
            self.builder = {}
        if world_count not in self.builder:
            self.builder[world_count] = Example.create_model_builder(
                self.robot, world_count, randomize=self.random_init, seed=123
            )

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_simulate(self, world_count):
        total_time = 0.0
        for _iter in range(self.samples):
            example = Example(
                stage_path=None,
                robot=self.robot,
                randomize=self.random_init,
                headless=True,
                actuation="random",
                use_cuda_graph=True,
                builder=self.builder[world_count],
                ls_iteration=self.ls_iteration,
                environment=self.environment,
            )

            wp.synchronize_device()
            for _ in range(self.num_frames):
                example.step()
            wp.synchronize_device()
            total_time += example.benchmark_time

        return total_time * 1000 / (self.num_frames * example.sim_substeps * world_count * self.samples)

    track_simulate.unit = "ms/world-step"


class _NewtonOverheadBenchmark:
    """Utility base class for measuring Newton overhead."""

    param_names = ["world_count"]
    num_frames = None
    params = None
    robot = None
    samples = None
    ls_iteration = None
    random_init = None

    def setup(self, world_count):
        if not hasattr(self, "builder") or self.builder is None:
            self.builder = {}
        if world_count not in self.builder:
            self.builder[world_count] = Example.create_model_builder(
                self.robot, world_count, randomize=self.random_init, seed=123
            )

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_simulate(self, world_count):
        trace = {}
        with EventTracer(enabled=True) as tracer:
            for _iter in range(self.samples):
                example = Example(
                    stage_path=None,
                    robot=self.robot,
                    randomize=self.random_init,
                    headless=True,
                    actuation="random",
                    world_count=world_count,
                    use_cuda_graph=True,
                    builder=self.builder[world_count],
                    ls_iteration=self.ls_iteration,
                )

                for _ in range(self.num_frames):
                    example.step()
                    trace = tracer.add_trace(trace, tracer.trace())

        step_time = trace["step"][0]
        step_trace = trace["step"][1]
        mujoco_warp_step_time = step_trace["_mujoco_warp_step"][0]
        overhead = 100.0 * (step_time - mujoco_warp_step_time) / step_time
        return overhead

    track_simulate.unit = "%"


class FastCartpole(_KpiBenchmark):
    params = [[8192]]
    num_frames = 50
    robot = "cartpole"
    samples = 4
    ls_iteration = 3
    random_init = True
    environment = "None"


class FastG1(_KpiBenchmark):
    params = [[8192]]
    num_frames = 50
    robot = "g1"
    timeout = 900
    samples = 2
    ls_iteration = 10
    random_init = True
    environment = "None"


class FastNewtonOverheadG1(_NewtonOverheadBenchmark):
    params = [[8192]]
    num_frames = 50
    robot = "g1"
    timeout = 900
    samples = 2
    ls_iteration = 10
    random_init = True


class FastHumanoid(_KpiBenchmark):
    params = [[8192]]
    num_frames = 100
    robot = "humanoid"
    samples = 4
    ls_iteration = 15
    random_init = True
    environment = "None"


class FastNewtonOverheadHumanoid(_NewtonOverheadBenchmark):
    params = [[8192]]
    num_frames = 100
    robot = "humanoid"
    samples = 4
    ls_iteration = 15
    random_init = True


class FastAllegro(_KpiBenchmark):
    params = [[8192]]
    num_frames = 300
    robot = "allegro"
    timeout = 900
    samples = 2
    ls_iteration = 10
    random_init = False
    environment = "None"


class FastKitchenG1(_KpiBenchmark):
    params = [[512]]
    num_frames = 50
    robot = "g1"
    timeout = 900
    samples = 2
    ls_iteration = 10
    random_init = True
    environment = "kitchen"


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastCartpole": FastCartpole,
        "FastG1": FastG1,
        "FastHumanoid": FastHumanoid,
        "FastAllegro": FastAllegro,
        "FastKitchenG1": FastKitchenG1,
        "FastNewtonOverheadG1": FastNewtonOverheadG1,
        "FastNewtonOverheadHumanoid": FastNewtonOverheadHumanoid,
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
