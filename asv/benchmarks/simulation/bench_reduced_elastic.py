# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

from importlib import import_module

import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.enable_backward = False
wp.config.quiet = True

import newton
from newton.viewer import ViewerNull

EXAMPLE_NAMES = ("wall", "dipper", "chair")
WARMUP_FRAMES = 5


def _make_warmed_example(case):
    if not hasattr(newton, "ModalBasis"):
        raise NotImplementedError("Reduced elastic links are unavailable in this revision")

    examples = {
        "wall": ("newton.examples.basic.example_basic_reduced_elastic_wall_contact", 60),
        "dipper": ("newton.examples.basic.example_basic_reduced_elastic_dipper", 60),
        "chair": ("newton.examples.basic.example_basic_reduced_elastic_chair_stick_slip", 60),
    }
    module_name, frame_count = examples[case]
    example_cls = import_module(module_name).Example
    example = example_cls(ViewerNull(num_frames=frame_count + WARMUP_FRAMES), None)

    for _ in range(WARMUP_FRAMES):
        example.step()
    wp.synchronize_device()

    return example, frame_count


class FastReducedElasticRepresentativeExamples:
    """Benchmark representative reduced elastic examples without rendering."""

    params = (EXAMPLE_NAMES,)
    param_names = ("case",)
    repeat = 3
    number = 1

    def setup(self, case):
        self.example, self.frame_count = _make_warmed_example(case)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_step(self, case):
        del case
        for _ in range(self.frame_count):
            self.example.step()
        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_final_modal_solve_residual_ratio(self, case):
        example, frame_count = _make_warmed_example(case)
        for _ in range(frame_count):
            example.step()
        wp.synchronize_device()
        return example.final_modal_solve_residual_ratio


class FastVBDRigidBaseline:
    """Track rigid-only VBD cost beside the reduced-elastic benchmarks."""

    params = ((1, 64),)
    param_names = ("body_count",)
    repeat = 3
    number = 1

    def setup(self, body_count):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        for body_index in range(body_count):
            builder.add_body(
                xform=wp.transform(wp.vec3(float(body_index % 8), float(body_index // 8), 1.0)),
                mass=1.0,
                inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            )
        builder.color()
        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.solver = newton.solvers.SolverVBD(self.model, iterations=10)
        for _ in range(WARMUP_FRAMES):
            self.solver.step(self.state_0, self.state_1, self.control, None, 1.0 / 240.0)
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_step(self, body_count):
        del body_count
        for _ in range(60):
            self.solver.step(self.state_0, self.state_1, self.control, None, 1.0 / 240.0)
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.synchronize_device()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastReducedElasticRepresentativeExamples": FastReducedElasticRepresentativeExamples,
        "FastVBDRigidBaseline": FastVBDRigidBaseline,
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
