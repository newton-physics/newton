# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import warp as wp

wp.config.enable_backward = False
wp.config.quiet = True

from asv_runner.benchmarks.mark import skip_benchmark_if

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from benchmark_mujoco import Example

import newton
from newton._src.utils.selection import match_labels
from newton.sensors import SensorContact

# Per-robot label patterns for sensing objects (extremities that contact the ground).
_SENSING_PATTERN = {
    "ant": "*_leg*",
    "humanoid": ["*thigh*", "*shin*", "*foot*", "*arm*"],
}

# Cache finalized models across benchmark classes and repeats. The model is
# never mutated by any benchmark, so sharing a single instance is safe.
_model_cache: dict[tuple[str, int], newton.Model] = {}


def _get_model(robot: str, world_count: int) -> newton.Model:
    key = (robot, world_count)
    if key not in _model_cache:
        builder = Example.create_model_builder(robot, world_count, randomize=False, seed=123)
        _model_cache[key] = builder.finalize()
    return _model_cache[key]


class InitSensorContact:
    """Benchmark SensorContact construction with pre-resolved index lists."""

    params = (["ant", "humanoid"], [64, 8192])
    param_names = ["robot", "world_count"]
    repeat = 3
    number = 1

    def setup(self, robot, world_count):
        wp.init()
        self._model = _get_model(robot, world_count)
        self._sensing = match_labels(self._model.body_label, _SENSING_PATTERN[robot])
        self._all = match_labels(self._model.body_label, "*")

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_with_counterparts(self, robot, world_count):
        SensorContact(self._model, sensing_obj_bodies=self._sensing, counterpart_bodies=self._all)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_total_only(self, robot, world_count):
        SensorContact(self._model, sensing_obj_bodies=self._sensing)


class InitSensorContactWithMatching:
    """Benchmark SensorContact construction including label pattern matching."""

    params = (["ant", "humanoid"], [8192])
    param_names = ["robot", "world_count"]
    repeat = 3
    number = 1

    def setup(self, robot, world_count):
        wp.init()
        self._model = _get_model(robot, world_count)
        self._pattern = _SENSING_PATTERN[robot]

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_with_counterparts(self, robot, world_count):
        SensorContact(self._model, sensing_obj_bodies=self._pattern, counterpart_bodies="*")

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_total_only(self, robot, world_count):
        SensorContact(self._model, sensing_obj_bodies=self._pattern)


class UpdateSensorContact:
    """Benchmark SensorContact.update() with many worlds, 1000 iterations via CUDA graph."""

    params = (["ant", "humanoid"], [64, 8192])
    param_names = ["robot", "world_count"]
    repeat = 5
    number = 1
    _GRAPH_ITERS = 1000

    def setup(self, robot, world_count):
        wp.init()
        self._model = _get_model(robot, world_count)

        legs = match_labels(self._model.body_label, _SENSING_PATTERN[robot])
        all_bodies = match_labels(self._model.body_label, "*")
        self._sensor_counterparts = SensorContact(self._model, sensing_obj_bodies=legs, counterpart_bodies=all_bodies)
        self._sensor_total = SensorContact(self._model, sensing_obj_bodies=legs)

        solver = newton.solvers.SolverMuJoCo(self._model)
        self._state = self._model.state()
        state_out = self._model.state()
        self._contacts = newton.Contacts(
            solver.get_max_contact_count(),
            0,
            device=self._model.device,
            requested_attributes=self._model.get_requested_contact_attributes(),
        )
        # Step until contacts are established (ant legs reach the ground at ~30 steps).
        control = self._model.control()
        dt = 1.0 / 60.0
        for _ in range(30):
            solver.step(self._state, state_out, control, None, dt)
            self._state, state_out = state_out, self._state
        solver.update_contacts(self._contacts, self._state)

        # Warmup: compile kernels for both sensor variants.
        self._sensor_counterparts.update(self._state, self._contacts)
        self._sensor_total.update(self._state, self._contacts)
        wp.synchronize()

        # Capture CUDA graphs that run update() 1000 times each.
        device = self._model.device
        self._graph_counterparts = None
        self._graph_total = None
        if device.is_cuda and wp.is_mempool_enabled(device):
            with wp.ScopedCapture(device) as cap:
                for _ in range(self._GRAPH_ITERS):
                    self._sensor_counterparts.update(self._state, self._contacts)
            self._graph_counterparts = cap.graph

            with wp.ScopedCapture(device) as cap:
                for _ in range(self._GRAPH_ITERS):
                    self._sensor_total.update(self._state, self._contacts)
            self._graph_total = cap.graph

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_with_counterparts(self, robot, world_count):
        if self._graph_counterparts is not None:
            wp.capture_launch(self._graph_counterparts)
        else:
            for _ in range(self._GRAPH_ITERS):
                self._sensor_counterparts.update(self._state, self._contacts)
        wp.synchronize()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_total_only(self, robot, world_count):
        if self._graph_total is not None:
            wp.capture_launch(self._graph_total)
        else:
            for _ in range(self._GRAPH_ITERS):
                self._sensor_total.update(self._state, self._contacts)
        wp.synchronize()


if __name__ == "__main__":
    from newton.utils import run_benchmark

    run_benchmark(InitSensorContact)
    run_benchmark(InitSensorContactWithMatching)
    run_benchmark(UpdateSensorContact)
