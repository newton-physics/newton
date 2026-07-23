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

from benchmark_metrics import (
    _SimulationMetricTracks,
    _SimulationMetricTracksUnparameterized,
    collect_simulation_metrics,
)


def _collect_metrics_dr_legs(robot, world_count, num_frames, samples, use_policy):
    if wp.get_cuda_device_count() == 0:
        return None

    from benchmark_kamino import DRLegsBenchmarkWorkload  # noqa: PLC0415

    builder = DRLegsBenchmarkWorkload.create_model_builder(robot, world_count)

    def create_workload():
        workload = DRLegsBenchmarkWorkload(
            robot=robot,
            world_count=world_count,
            use_cuda_graph=True,
            use_policy=use_policy,
            builder=builder,
        )
        if workload.graph is None or workload.reset_graph is None:
            raise RuntimeError("KPI benchmark requires CUDA graph capture (is the CUDA mempool allocator enabled?)")
        wp.synchronize_device()
        return workload

    return collect_simulation_metrics(
        create_workload=create_workload,
        world_count=world_count,
        num_frames=num_frames,
        samples=samples,
        validate=lambda workload: workload.test_final(),
    )


class _FastBenchmark:
    """Utility base class for fast Kamino benchmarks."""

    num_frames = None
    robot = None
    number = 1
    rounds = 2
    repeat = None
    world_count = None

    def setup(self):
        from benchmark_kamino import DRLegsBenchmarkWorkload  # noqa: PLC0415

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


class _KpiBenchmark(_SimulationMetricTracks):
    """Utility base class for Kamino KPI benchmarks."""

    param_names: ClassVar[list[str]] = ["world_count"]
    num_frames = None
    params: ClassVar[list[list[int]] | None] = None
    robot = None
    samples = None
    use_policy = True

    def _collect_metrics(self):
        if wp.get_cuda_device_count() == 0:
            return None

        metrics = {}
        for world_count in self.params[0]:
            metrics[world_count] = _collect_metrics_dr_legs(
                robot=self.robot,
                world_count=world_count,
                num_frames=self.num_frames,
                samples=self.samples,
                use_policy=self.use_policy,
            )
        return metrics


class FastDRLegs(_FastBenchmark):
    num_frames = 25
    robot = "dr_legs"
    repeat = 2
    world_count = 32


class FastMetricsDRLegs(_SimulationMetricTracksUnparameterized):
    num_frames = 25
    robot = "dr_legs"
    samples = 2
    world_count = 32

    def setup_cache(self):
        return _collect_metrics_dr_legs(
            robot=self.robot,
            world_count=self.world_count,
            num_frames=self.num_frames,
            samples=self.samples,
            use_policy=False,
        )


class KpiDRLegs(_KpiBenchmark):
    params: ClassVar[list[list[int]]] = [[4096]]
    num_frames = 25
    robot = "dr_legs"
    samples = 2

    def setup_cache(self):
        return self._collect_metrics()

    setup_cache.timeout = 1200


class NotifyDRLegs:
    """Benchmark Kamino model notifications for 2048 DR Legs worlds."""

    number = 10
    repeat = 7
    rounds = 1
    timeout = 3600
    world_count = 2048

    def setup(self):
        from benchmark_kamino import DRLegsBenchmarkWorkload  # noqa: PLC0415

        import newton  # noqa: PLC0415

        builder = DRLegsBenchmarkWorkload.create_model_builder("dr_legs", self.world_count)
        model = builder.finalize(skip_validation_joints=True)
        self._solver = newton.solvers.SolverKamino(model)
        self._model_flags = newton.ModelFlags
        for flag in (
            self._model_flags.MODEL_PROPERTIES,
            self._model_flags.BODY_PROPERTIES,
            self._model_flags.BODY_INERTIAL_PROPERTIES,
            self._model_flags.SHAPE_PROPERTIES,
            self._model_flags.JOINT_PROPERTIES,
            self._model_flags.JOINT_DOF_PROPERTIES,
            self._model_flags.ACTUATOR_PROPERTIES,
            self._model_flags.ALL,
        ):
            self._solver.notify_model_changed(flag)
        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_notify_actuator_properties(self):
        self._notify(self._model_flags.ACTUATOR_PROPERTIES)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_notify_all(self):
        self._notify(self._model_flags.ALL)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_notify_body_inertial_properties(self):
        self._notify(self._model_flags.BODY_INERTIAL_PROPERTIES)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_notify_body_properties(self):
        self._notify(self._model_flags.BODY_PROPERTIES)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_notify_joint_dof_properties(self):
        self._notify(self._model_flags.JOINT_DOF_PROPERTIES)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_notify_joint_properties(self):
        self._notify(self._model_flags.JOINT_PROPERTIES)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_notify_model_properties(self):
        self._notify(self._model_flags.MODEL_PROPERTIES)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_notify_shape_properties(self):
        self._notify(self._model_flags.SHAPE_PROPERTIES)

    def _notify(self, flag):
        self._solver.notify_model_changed(flag)
        wp.synchronize_device()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastDRLegs": FastDRLegs,
        "FastMetricsDRLegs": FastMetricsDRLegs,
        "KpiDRLegs": KpiDRLegs,
        "NotifyDRLegs": NotifyDRLegs,
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
