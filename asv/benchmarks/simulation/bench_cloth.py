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
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.quiet = True

import newton.examples
from newton.examples.cloth.example_cloth_franka import Example as ExampleClothManipulation
from newton.examples.cloth.example_cloth_twist import Example as ExampleClothTwist
from newton.viewer import ViewerNull


class FastExampleClothManipulation:
    timeout = 300
    repeat = 3
    number = 1

    def setup(self):
        self.num_frames = 30
        self.example = ExampleClothManipulation(ViewerNull(num_frames=self.num_frames))

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        newton.examples.run(self.example, args=None)

        wp.synchronize_device()


class FastExampleClothTwist:
    repeat = 5
    number = 1

    def setup(self):
        self.num_frames = 100
        self.example = ExampleClothTwist(ViewerNull(num_frames=self.num_frames))

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        newton.examples.run(self.example, None)

        wp.synchronize_device()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastExampleClothManipulation": FastExampleClothManipulation,
        "FastExampleClothTwist": FastExampleClothTwist,
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
