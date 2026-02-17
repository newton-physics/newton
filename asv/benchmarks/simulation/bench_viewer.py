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

import os
import sys

# Force headless mode for CI environments before any pyglet imports
os.environ["PYGLET_HEADLESS"] = "1"

import warp as wp

wp.config.enable_backward = False
wp.config.quiet = True

from asv_runner.benchmarks.mark import skip_benchmark_if

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from benchmark_mujoco import Example

from newton.viewer import ViewerGL


class KpiViewerGL:
    params = (["g1"], [8192])
    param_names = ["robot", "world_count"]

    rounds = 1
    repeat = 3
    number = 1
    min_run_count = 1

    def setup(self, robot, world_count):
        wp.init()
        builder = Example.create_model_builder(robot, world_count, randomize=True, seed=123)

        # finalize model
        self._model = builder.finalize()
        self._state = self._model.state()

        # Setting up the renderer
        self.renderer = ViewerGL(headless=True)
        self.renderer.set_model(self._model)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_rendering_frame(self, robot, world_count):
        # Rendering one frame
        self.renderer.begin_frame(0.0)
        self.renderer.log_state(self._state)
        self.renderer.end_frame()
        wp.synchronize_device()

    def teardown(self, robot, world_count):
        self.renderer.close()
        del self.renderer
        del self._model
        del self._state


class FastViewerGL:
    params = (["g1"], [256])
    param_names = ["robot", "world_count"]

    rounds = 1
    repeat = 3
    number = 1
    min_run_count = 1

    def setup(self, robot, world_count):
        wp.init()
        builder = Example.create_model_builder(robot, world_count, randomize=True, seed=123)

        # finalize model
        self._model = builder.finalize()
        self._state = self._model.state()

        # Setting up the renderer
        self.renderer = ViewerGL(headless=True)
        self.renderer.set_model(self._model)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_rendering_frame(self, robot, world_count):
        # Rendering one frame
        self.renderer.begin_frame(0.0)
        self.renderer.log_state(self._state)
        self.renderer.end_frame()
        wp.synchronize_device()

    def teardown(self, robot, world_count):
        self.renderer.close()
        del self.renderer
        del self._model
        del self._state


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "KpiViewerGL": KpiViewerGL,
        "FastViewerGL": FastViewerGL,
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
