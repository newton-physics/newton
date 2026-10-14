# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

import newton
from newton.actuators import Actuator, DrivePD, JointSpaceResponse

wp.config.log_level = wp.LOG_WARNING
wp.config.enable_backward = False

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from benchmark_config import pr_gate_repeat
from benchmark_mujoco import _setup_g1

KP = 1000.0
KD = 5.0


def _build_g1(world_count: int, internal_drive: bool) -> tuple[newton.Model, int]:
    g1 = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(g1)
    _setup_g1(g1)
    for i in range(6, g1.joint_dof_count):
        g1.joint_target_ke[i] = KP if internal_drive else 0.0
        g1.joint_target_kd[i] = KD if internal_drive else 0.0
        g1.joint_target_mode[i] = int(newton.JointTargetMode.POSITION)
    builder = newton.ModelBuilder()
    builder.replicate(g1, world_count)
    builder.add_ground_plane()
    return builder.finalize(), g1.joint_dof_count


class _ActuatorBenchmark:
    """Step G1 on SolverMuJoCo with joint PD driven inside the solver or by newton.actuators.

    ``drive`` selects ``internal`` (solver joint drive), ``implicit``
    (response from :meth:`JointSpaceResponse.refresh`, usable with any solver)
    or ``implicit_from_solve`` (response from MuJoCo's factorized inertia).
    Each step is one CUDA graph replay.
    """

    param_names = ["drive", "world_count"]
    params = None

    repeat = None
    number = 1
    rounds = 2
    timeout = 600

    NUM_STEPS = 200
    DT = 0.004

    def setup(self, drive, world_count):
        if not (wp.get_cuda_device_count() and wp.is_mempool_enabled(wp.get_device("cuda:0"))):
            raise SkipNotImplemented

        with wp.ScopedDevice("cuda:0"):
            model, dofs_per_world = _build_g1(world_count, internal_drive=drive == "internal")
            self.model = model
            self.solver = newton.solvers.SolverMuJoCo(
                model, solver="newton", integrator="implicitfast", njmax=210, nconmax=35, cone="pyramidal"
            )
            self.state_0, self.state_1 = model.state(), model.state()
            self.control = model.control()

            self.actuator = None
            self.response = None
            if drive != "internal":
                qd_start = model.joint_qd_start.numpy()
                q_start = model.joint_q_start.numpy()
                art_start = model.articulation_start.numpy()
                dofs, coords = [], []
                for a in range(model.articulation_count):
                    root = int(art_start[a])
                    dofs += range(int(qd_start[root]) + 6, int(qd_start[root]) + dofs_per_world)
                    coords += range(int(q_start[root]) + 7, int(q_start[root]) + dofs_per_world + 1)
                n = len(dofs)
                self.actuator = Actuator(
                    indices=wp.array(dofs, dtype=wp.uint32),
                    pos_indices=wp.array(coords, dtype=wp.uint32),
                    drive=DrivePD(kp=wp.full(n, KP, dtype=float), kd=wp.full(n, KD, dtype=float)),
                    control_target_pos_attr="joint_target_q",
                    control_target_vel_attr="joint_target_qd",
                )
                self.response = JointSpaceResponse(model)
                self.actuator.set_effort_mode_implicit(self.response)
                self.solve_inverse = None
                if drive == "implicit_from_solve":
                    import mujoco_warp  # noqa: PLC0415

                    solver = self.solver

                    def solve_inverse(x, y):
                        mujoco_warp.solve_m(solver.mjw_model, solver.mjw_data, x, y)

                    self.solve_inverse = solve_inverse
                    # Factorize the inertia once so the first captured refresh reads a valid one.
                    self._step(self.state_0, self.state_1)

            for _ in range(10):
                self._step(self.state_0, self.state_1)
                self._step(self.state_1, self.state_0)
            with wp.ScopedCapture() as capture:
                self._step(self.state_0, self.state_1)
                self._step(self.state_1, self.state_0)
            self.graph = capture.graph
            wp.synchronize_device()

    def _step(self, state_in, state_out):
        if self.actuator is not None:
            if self.solve_inverse is not None:
                self.response.refresh_from_solve(self.solve_inverse, dof_map=self.solver.mjc_dof_to_newton_dof)
            elif self.response is not None:
                self.response.refresh(state_in)
            self.control.joint_f.zero_()
            self.actuator.step(state_in, self.control, dt=self.DT)
        self.solver.step(state_in, state_out, self.control, None, self.DT)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_step(self, drive, world_count):
        for _ in range(self.NUM_STEPS // 2):
            wp.capture_launch(self.graph)
        wp.synchronize_device()

    def teardown(self, drive, world_count):
        if not np.all(np.isfinite(self.state_0.joint_q.numpy())):
            raise RuntimeError("Actuator benchmark produced non-finite joint positions.")


class FastActuatorG1(_ActuatorBenchmark):
    """Time 200 G1 steps on SolverMuJoCo for each drive, single world and batched."""

    params = [["internal", "implicit", "implicit_from_solve"], [1, 1024]]
    repeat = pr_gate_repeat(6)


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastActuatorG1": FastActuatorG1,
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
