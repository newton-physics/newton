# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU regression benchmarks.

Minimal but broad coverage of Newton's CPU codepath, intended to catch
regressions in the Warp CPU backend (issue #2830). Each benchmark exercises
a different subsystem and runs within ``wp.ScopedDevice("cpu")`` so it
executes without a GPU.
"""

import os
import sys

import numpy as np
import warp as wp

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

import newton
import newton.examples
from newton.examples.basic.example_basic_urdf import Example as XPBDQuadrupedExample
from newton.viewer import ViewerNull

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from benchmark_ik import build_ik_solver, create_franka_model, fk_targets, random_solutions
from benchmark_mujoco import Example as MuJoCoExample


class CpuMuJoCoAnt:
    """MuJoCo (Warp CPU) ant — exercises mujoco_warp + Newton glue with contacts and constraints."""

    repeat = 3
    number = 1
    num_frames = 50

    def setup(self):
        with wp.ScopedDevice("cpu"):
            self.example = MuJoCoExample(
                stage_path=None,
                robot="ant",
                randomize=False,
                headless=True,
                actuation="None",
                use_cuda_graph=False,
                world_count=1,
            )

    def time_simulate(self):
        with wp.ScopedDevice("cpu"):
            for _ in range(self.num_frames):
                self.example.step()


class CpuXPBDQuadruped:
    """XPBD rigid-body quadruped — exercises XPBD solver, contacts, articulations."""

    repeat = 3
    number = 1
    num_frames = 50

    def setup(self):
        with wp.ScopedDevice("cpu"):
            args = newton.examples.default_args()
            args.world_count = 1
            self.example = XPBDQuadrupedExample(ViewerNull(num_frames=self.num_frames), args)

    def time_simulate(self):
        with wp.ScopedDevice("cpu"):
            for _ in range(self.num_frames):
                self.example.step()


class CpuIKFranka:
    """IK on a Franka arm — exercises the IK solver and Jacobian path."""

    repeat = 3
    number = 1
    batch_size = 4
    iterations = 16
    num_solves = 20
    ee_links = (9,)
    seed = 123

    def setup(self):
        with wp.ScopedDevice("cpu"):
            self.model = create_franka_model()
            self.solver, pos_obj, rot_obj = build_ik_solver(self.model, self.batch_size, self.ee_links)
            n_coords = self.model.joint_coord_count

            q_gt = random_solutions(self.model, self.batch_size, np.random.default_rng(self.seed))
            tgt_p, tgt_r = fk_targets(self.solver, self.model, q_gt, self.ee_links)
            for ee in range(len(self.ee_links)):
                pos_obj[ee].set_target_positions(wp.array(tgt_p[:, ee].astype(np.float32), dtype=wp.vec3))
                rot_obj[ee].set_target_rotations(wp.array(tgt_r[:, ee].astype(np.float32), dtype=wp.vec4))

            self.seeds = wp.zeros((self.batch_size, n_coords), dtype=wp.float32)
            self.winners = wp.zeros((self.batch_size, n_coords), dtype=wp.float32)

    def time_solve(self):
        with wp.ScopedDevice("cpu"):
            for _ in range(self.num_solves):
                self.solver.step(self.seeds, self.winners, iterations=self.iterations, step_size=1.0)


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "CpuMuJoCoAnt": CpuMuJoCoAnt,
        "CpuXPBDQuadruped": CpuXPBDQuadruped,
        "CpuIKFranka": CpuIKFranka,
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

    benchmarks = args.bench if args.bench is not None else benchmark_list.keys()
    for key in benchmarks:
        run_benchmark(benchmark_list[key])
