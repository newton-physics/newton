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

from __future__ import annotations

import os
import sys

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.quiet = True

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from benchmark_ik import build_ik_solver, create_franka_model, eval_success, fk_targets, random_solutions


class _IKBenchmark:
    """Utility base class for IK benchmarks."""

    params = None
    param_names = ["batch_size"]

    repeat = None
    number = 1

    EE_LINKS = (9,)
    ITERATIONS = 16
    STEP_SIZE = 1.0
    POS_THRESH_M = 5e-3
    ORI_THRESH_RAD = 0.05
    SEED = 123

    def setup(self, batch_size):
        self.model = create_franka_model()
        self.solver, self.pos_obj, self.rot_obj = build_ik_solver(self.model, batch_size, self.EE_LINKS)
        self.n_coords = self.model.joint_coord_count

        rng = np.random.default_rng(self.SEED)
        q_gt = random_solutions(self.model, batch_size, rng)
        self.tgt_p, self.tgt_r = fk_targets(self.model, q_gt, self.EE_LINKS)

        self.winners_d = wp.zeros((batch_size, self.n_coords), dtype=wp.float32)
        self.seeds_d = wp.zeros((batch_size, self.n_coords), dtype=wp.float32)

        # Set targets
        for ee in range(len(self.EE_LINKS)):
            self.pos_obj[ee].set_target_positions(
                wp.array(self.tgt_p[:, ee].astype(np.float32, copy=False), dtype=wp.vec3)
            )
            self.rot_obj[ee].set_target_rotations(
                wp.array(self.tgt_r[:, ee].astype(np.float32, copy=False), dtype=wp.vec4)
            )

        # Capture CUDA graph
        self.solve_graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self.solver.step(self.seeds_d, self.winners_d, iterations=self.ITERATIONS, step_size=self.STEP_SIZE)
            self.solve_graph = cap.graph

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_solve(self, batch_size):
        self.solver.reset()

        if self.solve_graph is not None:
            wp.capture_launch(self.solve_graph)
        else:
            self.solver.step(self.seeds_d, self.winners_d, iterations=self.ITERATIONS, step_size=self.STEP_SIZE)

        wp.synchronize_device()

        # Validate 100% success rate
        q_best = self.winners_d.numpy()
        success = eval_success(
            self.solver, self.model, q_best, self.tgt_p, self.tgt_r,
            self.EE_LINKS, self.POS_THRESH_M, self.ORI_THRESH_RAD,
        )
        success_rate = success.mean()
        if success_rate < 1.0:
            raise RuntimeError(f"IK success rate {success_rate * 100:.1f}% < 100% (batch_size={batch_size})")


class KpiIKSolve(_IKBenchmark):
    params = ([8192],)
    repeat = 3


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "KpiIKSolve": KpiIKSolve,
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
