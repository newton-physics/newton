# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

import newton

wp.config.quiet = True
wp.config.enable_backward = False

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from benchmark_inverse_dynamics import create_franka_model, set_default_pose


class _InverseDynamicsBenchmark:
    """Utility base class for inverse-dynamics benchmarks."""

    repeat = None
    number = 1
    rounds = 2

    NUM_EVALS = 500
    NUM_FORCE_EVALS = 40_000
    WORLD_COUNT = 1024

    def setup(self):
        if not (wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device())):
            raise SkipNotImplemented

        self.model = create_franka_model(world_count=self.WORLD_COUNT)
        self.state = self.model.state()
        set_default_pose(self.model, self.state)

        self.inverse_dynamics, self.scratch = self.model.inverse_dynamics()

        # Capture one full M(q) + g(q) + C(q, q_dot)*q_dot evaluation into a
        # CUDA graph so the timed inner loop is just graph replays.
        with wp.ScopedCapture() as cap:
            newton.eval_inverse_dynamics(
                self.model,
                self.state,
                newton.InverseDynamics.EvalType.ALL,
                self.inverse_dynamics,
                self.scratch,
            )
        self.eval_graph = cap.graph

        # Populate the inverse_dynamics buffers so eval_inverse_dynamics_force
        # has a valid M(q) / g(q) / C(q, q_dot)*q_dot to consume. The capture
        # above only records the launches; it does not execute them.
        wp.capture_launch(self.eval_graph)

        # qddot input for eval_inverse_dynamics_force. Any finite values are
        # fine; use a tiled ramp matching set_default_pose's joint_qd pattern.
        n_dofs = self.model.joint_dof_count
        dofs_per_world = n_dofs // max(self.model.world_count, 1)
        qddot_per_world = np.linspace(-0.1, 0.1, dofs_per_world, dtype=np.float32)
        self.qddot = wp.array(
            np.tile(qddot_per_world, max(self.model.world_count, 1)),
            dtype=wp.float32,
            device=self.model.device,
        )

        # Capture eval_inverse_dynamics_force into a second graph so the
        # force-only inner loop is purely replays.
        with wp.ScopedCapture() as cap_force:
            newton.eval_inverse_dynamics_force(
                self.model,
                self.inverse_dynamics.mass_matrix,
                self.qddot,
                self.inverse_dynamics.coriolis_compensation_force,
                self.inverse_dynamics.gravity_compensation_force,
                self.inverse_dynamics.tau,
            )
        self.force_graph = cap_force.graph

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_eval_inverse_dynamics(self):
        for _ in range(self.NUM_EVALS):
            wp.capture_launch(self.eval_graph)
        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_eval_inverse_dynamics_force(self):
        for _ in range(self.NUM_FORCE_EVALS):
            wp.capture_launch(self.force_graph)
        wp.synchronize_device()

    def teardown(self):
        H = self.inverse_dynamics.mass_matrix.numpy()
        g = self.inverse_dynamics.gravity_compensation_force.numpy()
        c = self.inverse_dynamics.coriolis_compensation_force.numpy()
        tau = self.inverse_dynamics.tau.numpy()
        finite = (
            np.all(np.isfinite(H))
            and np.all(np.isfinite(g))
            and np.all(np.isfinite(c))
            and np.all(np.isfinite(tau))
        )
        if not finite:
            raise RuntimeError("Inverse-dynamics output contains non-finite values.")


class FastInverseDynamics(_InverseDynamicsBenchmark):
    """Time ``eval_inverse_dynamics(EvalType.ALL)`` on a single Franka arm."""

    repeat = 6


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastInverseDynamics": FastInverseDynamics,
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
