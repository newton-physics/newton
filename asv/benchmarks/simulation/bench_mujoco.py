# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import warp as wp

wp.config.enable_backward = False
wp.config.quiet = True

from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

from newton.examples.example_mujoco import Example
from newton.utils import EventTracer


@wp.kernel
def apply_random_control(state: wp.uint32, joint_target: wp.array(dtype=float)):
    tid = wp.tid()

    joint_target[tid] = wp.randf(state) * 2.0 - 1.0


class _FastBenchmark:
    """Utility base class for fast benchmarks."""

    num_frames = None
    robot = None
    number = 1
    rounds = 2
    repeat = None
    num_worlds = None
    random_init = None
    environment = "None"

    def setup(self):
        if not hasattr(self, "builder") or self.builder is None:
            self.builder = Example.create_model_builder(self.robot, self.num_worlds, randomize=True, seed=123)

        self.example = Example(
            stage_path=None,
            robot=self.robot,
            randomize=self.random_init,
            headless=True,
            actuation="None",
            use_cuda_graph=True,
            builder=self.builder,
            environment=self.environment,
        )

        wp.synchronize_device()

        # Recapture the graph with control application included
        cuda_graph_comp = wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device())
        if not cuda_graph_comp:
            raise SkipNotImplemented
        else:
            state = wp.rand_init(self.example.seed)
            with wp.ScopedCapture() as capture:
                wp.launch(
                    apply_random_control,
                    dim=(self.example.model.joint_dof_count,),
                    inputs=[state],
                    outputs=[self.example.control.joint_target],
                )
                self.example.simulate()
            self.graph = capture.graph

        wp.synchronize_device()

    def time_simulate(self):
        for _ in range(self.num_frames):
            wp.capture_launch(self.graph)
        wp.synchronize_device()


class _KpiBenchmark:
    """Utility base class for KPI benchmarks."""

    param_names = ["num_worlds"]
    num_frames = None
    params = None
    robot = None
    samples = None
    ls_iteration = None
    random_init = None
    environment = "None"

    def setup(self, num_worlds):
        if not hasattr(self, "builder") or self.builder is None:
            self.builder = {}
        if num_worlds not in self.builder:
            self.builder[num_worlds] = Example.create_model_builder(self.robot, num_worlds, randomize=True, seed=123)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_simulate(self, num_worlds):
        total_time = 0.0
        for _iter in range(self.samples):
            example = Example(
                stage_path=None,
                robot=self.robot,
                randomize=self.random_init,
                headless=True,
                actuation="random",
                use_cuda_graph=True,
                builder=self.builder[num_worlds],
                ls_iteration=self.ls_iteration,
                environment=self.environment,
            )

            wp.synchronize_device()
            for _ in range(self.num_frames):
                example.step()
            wp.synchronize_device()
            total_time += example.benchmark_time

        return total_time * 1000 / (self.num_frames * example.sim_substeps * num_worlds * self.samples)

    track_simulate.unit = "ms/world-step"


class _NewtonOverheadBenchmark:
    """Utility base class for measuring Newton overhead."""

    param_names = ["num_worlds"]
    num_frames = None
    params = None
    robot = None
    samples = None
    ls_iteration = None
    random_init = None

    def setup(self, num_worlds):
        if not hasattr(self, "builder") or self.builder is None:
            self.builder = {}
        if num_worlds not in self.builder:
            self.builder[num_worlds] = Example.create_model_builder(self.robot, num_worlds, randomize=True, seed=123)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_simulate(self, num_worlds):
        trace = {}
        with EventTracer(enabled=True) as tracer:
            for _iter in range(self.samples):
                example = Example(
                    stage_path=None,
                    robot=self.robot,
                    randomize=self.random_init,
                    headless=True,
                    actuation="random",
                    num_worlds=num_worlds,
                    use_cuda_graph=True,
                    builder=self.builder[num_worlds],
                    ls_iteration=self.ls_iteration,
                )

                for _ in range(self.num_frames):
                    example.step()
                    trace = example.add_trace(trace, tracer.trace())

        step_time = trace["step"][0]
        mujoco_warp_step_time = trace["step"][1]["mujoco_warp_step"][0]
        overhead = 100.0 * (step_time - mujoco_warp_step_time) / step_time
        return overhead

    track_simulate.unit = "%"


class FastCartpole(_FastBenchmark):
    num_frames = 50
    robot = "cartpole"
    repeat = 8
    num_worlds = 256
    random_init = True
    environment = "None"


class KpiCartpole(_KpiBenchmark):
    params = [[8192]]
    num_frames = 50
    robot = "cartpole"
    samples = 4
    ls_iteration = 3
    random_init = True
    environment = "None"


class FastG1(_FastBenchmark):
    num_frames = 25
    robot = "g1"
    repeat = 2
    num_worlds = 256
    random_init = True
    environment = "None"


class KpiG1(_KpiBenchmark):
    params = [[8192]]
    num_frames = 50
    robot = "g1"
    timeout = 900
    samples = 2
    ls_iteration = 10
    random_init = True
    environment = "None"


class FastNewtonOverheadG1(_NewtonOverheadBenchmark):
    params = [[256]]
    num_frames = 25
    robot = "g1"
    repeat = 2
    samples = 1
    random_init = True


class KpiNewtonOverheadG1(_NewtonOverheadBenchmark):
    params = [[8192]]
    num_frames = 50
    robot = "g1"
    timeout = 900
    samples = 2
    ls_iteration = 10
    random_init = True


class FastHumanoid(_FastBenchmark):
    num_frames = 50
    robot = "humanoid"
    repeat = 8
    num_worlds = 256
    random_init = True
    environment = "None"


class KpiHumanoid(_KpiBenchmark):
    params = [[8192]]
    num_frames = 100
    robot = "humanoid"
    samples = 4
    ls_iteration = 15
    random_init = True
    environment = "None"


class FastNewtonOverheadHumanoid(_NewtonOverheadBenchmark):
    params = [[256]]
    num_frames = 50
    robot = "humanoid"
    repeat = 8
    samples = 1
    random_init = True


class KpiNewtonOverheadHumanoid(_NewtonOverheadBenchmark):
    params = [[8192]]
    num_frames = 100
    robot = "humanoid"
    samples = 4
    ls_iteration = 15
    random_init = True


class FastAllegro(_FastBenchmark):
    num_frames = 100
    robot = "allegro"
    repeat = 2
    num_worlds = 256
    random_init = False
    environment = "None"


class KpiAllegro(_KpiBenchmark):
    params = [[8192]]
    num_frames = 300
    robot = "allegro"
    samples = 2
    ls_iteration = 10
    random_init = False
    environment = "None"


class FastKitchenG1(_FastBenchmark):
    num_frames = 25
    robot = "g1"
    repeat = 2
    num_worlds = 32
    random_init = True
    environment = "kitchen"


class KpiKitchenG1(_KpiBenchmark):
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
        "KpiCartpole": KpiCartpole,
        "KpiG1": KpiG1,
        "KpiHumanoid": KpiHumanoid,
        "KpiAllegro": KpiAllegro,
        "KpiKitchenG1": KpiKitchenG1,
        "KpiNewtonOverheadG1": KpiNewtonOverheadG1,
        "KpiNewtonOverheadHumanoid": KpiNewtonOverheadHumanoid,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b", "--bench", default=None, action="append", choices=benchmark_list.keys(), help="Run a single benchmark."
    )
    args = parser.parse_known_args()[0]

    if args.bench is None:
        benchmarks = benchmark_list.keys()
    else:
        benchmarks = args.bench

    for key in benchmarks:
        benchmark = benchmark_list[key]
        run_benchmark(benchmark)
