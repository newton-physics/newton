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
import time

wp.config.enable_backward = False
wp.config.quiet = True

from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

from newton.examples.example_mujoco import Example


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
    num_envs = None

    def setup(self):
        if not hasattr(self, "builder") or self.builder is None:
            self.builder = Example.create_model_builder(self.robot, self.num_envs, randomize=True, seed=123)

        self.example = Example(
            stage_path=None,
            robot=self.robot,
            randomize=True,
            headless=True,
            actuation="None",
            num_envs=self.num_envs,
            use_cuda_graph=True,
            builder=self.builder,
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

    param_names = ["num_envs"]
    num_frames = None
    params = None
    robot = None
    samples = None
    ls_iteration = None

    def setup(self, num_envs):
        if not hasattr(self, "builder") or self.builder is None:
            self.builder = {}
        if num_envs not in self.builder:
            self.builder[num_envs] = Example.create_model_builder(self.robot, num_envs, randomize=True, seed=123)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_simulate(self, num_envs):
        total_time = 0.0
        for _iter in range(self.samples):
            example = Example(
                stage_path=None,
                robot=self.robot,
                randomize=True,
                headless=True,
                actuation="random",
                num_envs=num_envs,
                use_cuda_graph=True,
                builder=self.builder[num_envs],
                ls_iteration=self.ls_iteration,
            )

            wp.synchronize_device()
            for _ in range(self.num_frames):
                example.step()
            wp.synchronize_device()
            total_time += example.benchmark_time

        return total_time * 1000 / (self.num_frames * example.sim_substeps * num_envs * self.samples)

    track_simulate.unit = "ms/env-step"

class _SwizzleBenchmark:
    """Utility base class for measuring swizzling overhead."""

    param_names = ["num_envs"]
    num_frames = None
    params = None
    robot = None
    samples = None
    ls_iteration = None
    overhead_time = 0.0
    step_time = 0.0


    def setup(self, num_envs):
        if not hasattr(self, "builder") or self.builder is None:
            self.builder = {}
        if num_envs not in self.builder:
            self.builder[num_envs] = Example.create_model_builder(self.robot, num_envs, randomize=True, seed=123)

    def create_graph(self, example):
        self.sim_substeps = example.sim_substeps
        self.frame_dt = example.frame_dt

        # simulate() allocates memory via a clone, so we can't use graph capture if the device does not support mempools
        cuda_graph_comp = wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device())
        if not cuda_graph_comp:
            print("Cannot use graph capture. Graph capture is disabled.")
            raise RuntimeError
        else:
            # Capture the control graph
            with wp.ScopedCapture() as capture:
                joint_target = wp.array(example.rng.uniform(-1.0, 1.0, size=example.model.joint_dof_count), dtype=float)
                wp.copy(example.control.joint_target, joint_target)
            self.graph_control = capture.graph

            # Capture the pre-step graph
            with wp.ScopedCapture() as capture:
                example.state_0.clear_forces()
                example.solver.apply_mjc_control(example.model, example.state_0, example.control, example.solver.mjw_data)
                if example.solver.update_data_interval > 0 and example.solver._step % example.solver.update_data_interval == 0:
                    example.solver.update_mjc_data(example.solver.mjw_data, example.model, example.state_0)
                example.solver.mjw_model.opt.timestep.fill_(example.sim_dt)
            self.graph_prestep = capture.graph

            # Capture the step graph
            with wp.ScopedCapture() as capture:
                example.solver._mujoco_warp.step(example.solver.mjw_model, example.solver.mjw_data)
            self.graph_step = capture.graph

            # Capture the post-step graph
            with wp.ScopedCapture() as capture:
                example.solver.update_newton_state(example.model, example.state_1, example.solver.mjw_data)
                example.solver._step += 1
                example.state_0, example.state_1 = example.state_1, example.state_0
            self.graph_poststep = capture.graph


    def step(self):
        # Reimplement the step function to be able to graph the mujoco_warp step function

        # Setup control input
        wp.synchronize_device()
        start_time = time.time()
        wp.capture_launch(self.graph_control)
        wp.synchronize_device()
        end_time = time.time()
        self.overhead_time += end_time - start_time

        for _ in range(self.sim_substeps):
            # Prepare for the step
            wp.synchronize_device()
            start_time = time.time()
            wp.capture_launch(self.graph_prestep)
            wp.synchronize_device()
            end_time = time.time()
            self.overhead_time += end_time - start_time

            # Step
            wp.synchronize_device()
            start_time = time.time()
            wp.capture_launch(self.graph_step)
            wp.synchronize_device()
            end_time = time.time()
            self.step_time += end_time - start_time

            # Complete the step
            wp.synchronize_device()
            start_time = time.time()
            wp.capture_launch(self.graph_poststep)
            wp.synchronize_device()
            end_time = time.time()
            self.overhead_time += end_time - start_time


    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_simulate(self, num_envs):
        for _iter in range(self.samples):
            example = Example(
                stage_path=None,
                robot=self.robot,
                randomize=True,
                headless=True,
                actuation="random",
                num_envs=num_envs,
                use_cuda_graph=True,
                builder=self.builder[num_envs],
                ls_iteration=self.ls_iteration,
            )

            self.create_graph(example)
            for _ in range(self.num_frames):
                self.step()

            swizzle_overhead = 100.0 * self.overhead_time / (self.overhead_time + self.step_time)

        return swizzle_overhead


class SwizzleAnt(_SwizzleBenchmark):
    params = [256]
    num_frames = 100
    robot = "ant"
    samples = 4
    ls_iteration = 10


class FastAnt(_FastBenchmark):
    num_frames = 50
    robot = "ant"
    repeat = 8
    num_envs = 256


class KpiAnt(_KpiBenchmark):
    params = [4096, 8192, 16384]
    num_frames = 100
    robot = "ant"
    samples = 4
    ls_iteration = 10


class FastCartpole(_FastBenchmark):
    num_frames = 50
    robot = "cartpole"
    repeat = 8
    num_envs = 256


class KpiCartpole(_KpiBenchmark):
    params = [4096, 8192]
    num_frames = 50
    robot = "cartpole"
    samples = 4
    ls_iteration = 3


class FastG1(_FastBenchmark):
    num_frames = 25
    robot = "g1"
    repeat = 2
    num_envs = 256


class KpiG1(_KpiBenchmark):
    params = [4096, 8192]
    num_frames = 50
    robot = "g1"
    timeout = 900
    samples = 2
    ls_iteration = 10


class FastH1(_FastBenchmark):
    num_frames = 25
    robot = "h1"
    repeat = 2
    num_envs = 256


class KpiH1(_KpiBenchmark):
    params = [4096, 8192]
    num_frames = 50
    robot = "h1"
    timeout = 900
    samples = 2
    ls_iteration = 10


class FastHumanoid(_FastBenchmark):
    num_frames = 50
    robot = "humanoid"
    repeat = 8
    num_envs = 256


class KpiHumanoid(_KpiBenchmark):
    params = [4096, 8192]
    num_frames = 100
    robot = "humanoid"
    samples = 4
    ls_iteration = 15


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastAnt": FastAnt,
        "FastCartpole": FastCartpole,
        "FastG1": FastG1,
        "FastH1": FastH1,
        "FastHumanoid": FastHumanoid,
        "KpiAnt": KpiAnt,
        "KpiCartpole": KpiCartpole,
        "KpiG1": KpiG1,
        "KpiH1": KpiH1,
        "KpiHumanoid": KpiHumanoid,
        "SwizzleAnt": SwizzleAnt,
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
