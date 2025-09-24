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

import time

import warp as wp

wp.config.enable_backward = False
wp.config.quiet = True

from asv_runner.benchmarks.mark import skip_benchmark_if

from newton.examples.example_mujoco import Example


class FastAnt:
    num_frames = 50
    robot = "ant"
    number = 1
    repeat = 8
    rounds = 2
    num_envs = 256

    def setup(self):
        if not hasattr(self, "builder") or self.builder is None:
            self.builder = Example.create_model_builder(self.robot, self.num_envs, randomize=True, seed=123)

        self.example = Example(
            stage_path=None,
            robot=self.robot,
            randomize=True,
            headless=True,
            actuation="random",
            num_envs=self.num_envs,
            use_cuda_graph=True,
            builder=self.builder,
        )

        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


class KpiAnt:
    params = [4096, 8192, 16384]
    param_names = ["num_envs"]
    num_frames = 100
    robot = "ant"
    samples = 4

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
                ls_iteration=10,
            )

            wp.synchronize_device()
            start_time = time.time()
            for _ in range(self.num_frames):
                example.step()
            wp.synchronize_device()
            total_time += time.time() - start_time

        return total_time * 1000 / (self.num_frames * example.sim_substeps * num_envs * self.samples)

    track_simulate.unit = "ms/env-step"


class FastCartpole:
    num_frames = 50
    robot = "cartpole"
    number = 1
    repeat = 8
    rounds = 2
    num_envs = 256

    def setup(self):
        if not hasattr(self, "builder") or self.builder is None:
            self.builder = Example.create_model_builder(self.robot, self.num_envs, randomize=True, seed=123)

        self.example = Example(
            stage_path=None,
            robot=self.robot,
            randomize=True,
            headless=True,
            actuation="random",
            num_envs=self.num_envs,
            use_cuda_graph=True,
            builder=self.builder,
        )

        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


class KpiCartpole:
    params = [4096, 8192]
    param_names = ["num_envs"]
    num_frames = 50
    robot = "cartpole"
    samples = 4

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
                ls_iteration=3,
            )

            wp.synchronize_device()
            start_time = time.time()
            for _ in range(self.num_frames):
                example.step()
            wp.synchronize_device()
            total_time += time.time() - start_time

        return total_time * 1000 / (self.num_frames * example.sim_substeps * num_envs * self.samples)

    track_simulate.unit = "ms/env-step"


class FastG1:
    num_frames = 25
    robot = "g1"
    number = 1
    repeat = 2
    rounds = 2
    num_envs = 256

    def setup(self):
        if not hasattr(self, "builder") or self.builder is None:
            self.builder = Example.create_model_builder(self.robot, self.num_envs, randomize=True, seed=123)

        self.example = Example(
            stage_path=None,
            robot=self.robot,
            randomize=True,
            headless=True,
            actuation="random",
            num_envs=self.num_envs,
            use_cuda_graph=True,
            builder=self.builder,
        )

        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


class KpiG1:
    params = [4096, 8192]
    param_names = ["num_envs"]
    num_frames = 50
    robot = "g1"
    timeout = 900
    samples = 2

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
                ls_iteration=10,
            )

            wp.synchronize_device()
            start_time = time.time()
            for _ in range(self.num_frames):
                example.step()
            wp.synchronize_device()
            total_time += time.time() - start_time

        return total_time * 1000 / (self.num_frames * example.sim_substeps * num_envs * self.samples)

    track_simulate.unit = "ms/env-step"


class FastH1:
    num_frames = 25
    robot = "h1"
    number = 1
    repeat = 2
    rounds = 2
    num_envs = 256

    def setup(self):
        if not hasattr(self, "builder") or self.builder is None:
            self.builder = Example.create_model_builder(self.robot, self.num_envs, randomize=True, seed=123)

        self.example = Example(
            stage_path=None,
            robot=self.robot,
            randomize=True,
            headless=True,
            actuation="random",
            num_envs=self.num_envs,
            use_cuda_graph=True,
            builder=self.builder,
        )

        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


class KpiH1:
    params = [4096, 8192]
    param_names = ["num_envs"]
    num_frames = 50
    robot = "h1"
    timeout = 900
    samples = 2

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
                ls_iteration=10,
            )

            wp.synchronize_device()
            start_time = time.time()
            for _ in range(self.num_frames):
                example.step()
            wp.synchronize_device()
            total_time += time.time() - start_time

        return total_time * 1000 / (self.num_frames * example.sim_substeps * num_envs * self.samples)

    track_simulate.unit = "ms/env-step"


class FastHumanoid:
    num_frames = 50
    robot = "humanoid"
    number = 1
    repeat = 8
    rounds = 2
    num_envs = 256

    def setup(self):
        if not hasattr(self, "builder") or self.builder is None:
            self.builder = Example.create_model_builder(self.robot, self.num_envs, randomize=True, seed=123)

        self.example = Example(
            stage_path=None,
            robot=self.robot,
            randomize=True,
            headless=True,
            actuation="random",
            num_envs=self.num_envs,
            use_cuda_graph=True,
            builder=self.builder,
        )

        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


class KpiHumanoid:
    params = [4096, 8192]
    param_names = ["num_envs"]
    num_frames = 100
    robot = "humanoid"
    samples = 4

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
                ls_iteration=15,
            )

            wp.synchronize_device()
            start_time = time.time()
            for _ in range(self.num_frames):
                example.step()
            wp.synchronize_device()
            total_time += time.time() - start_time

        return total_time * 1000 / (self.num_frames * example.sim_substeps * num_envs * self.samples)

    track_simulate.unit = "ms/env-step"


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
