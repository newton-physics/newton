# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

wp.config.quiet = True
wp.config.enable_backward = False

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from benchmark_ik import build_ik_solver, create_franka_model, eval_success, fk_targets, random_solutions


class _IKBenchmark:
    """Utility base class for IK benchmarks."""

    params = None
    param_names = ["batch_size"]
    repeat = None
    number = 1
    rounds = 2

    EE_LINKS = (9,)
    ITERATIONS = 16
    STEP_SIZE = 1.0
    POS_THRESH_M = 5e-3
    ORI_THRESH_RAD = 0.05
    SEED = 123
    NUM_SOLVES = 50

    def setup(self, batch_size):
        if not (wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device())):
            raise SkipNotImplemented

        self.model = create_franka_model()
        self.solver, self.pos_obj, self.rot_obj = build_ik_solver(self.model, batch_size, self.EE_LINKS)
        self.n_coords = self.model.joint_coord_count

        rng = np.random.default_rng(self.SEED)
        q_gt = random_solutions(self.model, batch_size, rng)
        self.tgt_p, self.tgt_r = fk_targets(self.solver, self.model, q_gt, self.EE_LINKS)

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

        with wp.ScopedCapture() as cap:
            self.solver.step(self.seeds_d, self.winners_d, iterations=self.ITERATIONS, step_size=self.STEP_SIZE)
        self.solve_graph = cap.graph

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_solve(self, batch_size):
        for _ in range(self.NUM_SOLVES):
            wp.capture_launch(self.solve_graph)
        wp.synchronize_device()

    def teardown(self, batch_size):
        q_best = self.winners_d.numpy()
        success = eval_success(
            self.solver,
            self.model,
            q_best,
            self.tgt_p,
            self.tgt_r,
            self.EE_LINKS,
            self.POS_THRESH_M,
            self.ORI_THRESH_RAD,
        )
        if not success.all():
            n_failed = int((~success).sum())
            raise RuntimeError(f"IK failed for {n_failed}/{batch_size} problems")


class FastIKSolve(_IKBenchmark):
    params = ([512],)
    repeat = 6


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastIKSolve": FastIKSolve,
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
