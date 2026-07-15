# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from typing import ClassVar

import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from benchmark_kamino import DRLegsBenchmarkWorkload


class _FastBenchmark:
    """Utility base class for fast Kamino benchmarks."""

    num_frames = None
    robot = None
    number = 1
    rounds = 2
    repeat = None
    world_count = None

    def setup(self):
        if not hasattr(self, "_builder") or self._builder is None:
            self._builder = DRLegsBenchmarkWorkload.create_model_builder(self.robot, self.world_count)

        self.workload = DRLegsBenchmarkWorkload(
            robot=self.robot,
            world_count=self.world_count,
            use_cuda_graph=True,
            use_policy=False,
            builder=self._builder,
        )

        wp.synchronize_device()

        if self.workload.graph is None or self.workload.reset_graph is None:
            raise SkipNotImplemented("CUDA graph capture unavailable (is the CUDA mempool allocator enabled?)")

    def teardown(self):
        workload = getattr(self, "workload", None)
        if workload is not None:
            workload.test_final()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            for _ in range(self.workload.decimation):
                wp.capture_launch(self.workload.reset_graph)
                wp.capture_launch(self.workload.graph)
        wp.synchronize_device()


class _KpiBenchmark:
    """Utility base class for Kamino KPI benchmarks."""

    param_names: ClassVar[list[str]] = ["world_count"]
    num_frames = None
    params: ClassVar[list[list[int]] | None] = None
    robot = None
    samples = None
    use_policy = True

    def setup(self, world_count):
        if not hasattr(self, "_builder") or self._builder is None:
            self._builder = {}
        if world_count not in self._builder:
            self._builder[world_count] = DRLegsBenchmarkWorkload.create_model_builder(self.robot, world_count)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_simulate(self, world_count):
        total_time = 0.0
        for _iter in range(self.samples):
            workload = DRLegsBenchmarkWorkload(
                robot=self.robot,
                world_count=world_count,
                use_cuda_graph=True,
                use_policy=self.use_policy,
                builder=self._builder[world_count],
            )
            if workload.graph is None or workload.reset_graph is None:
                raise RuntimeError("KPI benchmark requires CUDA graph capture (is the CUDA mempool allocator enabled?)")

            wp.synchronize_device()
            for _ in range(self.num_frames):
                workload.step()
            total_time += workload.benchmark_time
            workload.test_final()

        return total_time * 1000 / (self.num_frames * workload.sim_substeps * world_count * self.samples)

    track_simulate.unit = "ms/world-step"


class FastDRLegs(_FastBenchmark):
    num_frames = 25
    robot = "dr_legs"
    repeat = 2
    world_count = 32


class KpiDRLegs(_KpiBenchmark):
    params: ClassVar[list[list[int]]] = [[4096]]
    num_frames = 25
    robot = "dr_legs"
    samples = 2


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastDRLegs": FastDRLegs,
        "KpiDRLegs": KpiDRLegs,
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
