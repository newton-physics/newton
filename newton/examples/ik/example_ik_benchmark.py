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

###########################################################################
# Example: IK Benchmark
#
# Shows how to benchmark the performance of the Newton IK solver on the
# Franka Emika Panda robot for various batch sizes.
#
# Command: python -m newton.examples ik_benchmark
#
###########################################################################

from __future__ import annotations

import time
from functools import partial
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils


class Example:
    """
    Manages the setup and execution of the IK benchmark.
    """

    def __init__(self, viewer, repeats, batch_sizes, seed: int):
        # Benchmark parameters from parsed arguments
        self.batch_sizes = batch_sizes
        self.repeats = repeats
        self.iterations = 16
        self.step_size = 1.0
        self.pos_thresh_m = 5e-3
        self.ori_thresh_rad = 0.05

        # Hardcoded Franka configuration
        self.robot_name = "franka"
        self.asset_name = "franka_emika_panda"
        self.asset_file = Path("urdf/fr3.urdf")
        self.parser = partial(newton.ModelBuilder.add_urdf, scale=1.0)
        self.ee_names = ("ee",)
        self.ee_links = (9,)
        self.seeds = 64
        self.lambda_factor = 4.0

        self.use_cuda_graph = wp.get_device().is_cuda
        self.model = self._create_robot()
        self.n_coords = self.model.joint_coord_count
        self.results = []
        self.rng = np.random.default_rng(seed)

    def _create_robot(self) -> newton.Model:
        franka = newton.ModelBuilder()
        franka.num_rigid_contacts_per_world = 0
        franka.default_shape_cfg.density = 100.0
        asset_path = newton.utils.download_asset(self.asset_name) / self.asset_file
        self.parser(franka, asset_path, floating=False)
        model = franka.finalize(requires_grad=False)
        return model

    def _random_solutions(self, n: int) -> np.ndarray:
        lower = self.model.joint_limit_lower.numpy()[: self.n_coords]
        upper = self.model.joint_limit_upper.numpy()[: self.n_coords]
        span = upper - lower
        mask = np.abs(span) > 1e5
        span[mask] = 0.0
        q = self.rng.random((n, self.n_coords)) * span + lower
        q[:, mask] = 0.0
        return q.astype(np.float32)

    def _build_ik_solver(self, n_problems: int):
        zero_pos = [wp.zeros(n_problems, dtype=wp.vec3) for _ in self.ee_links]
        zero_rot = [wp.zeros(n_problems, dtype=wp.vec4) for _ in self.ee_links]
        objectives = []
        for ee, link in enumerate(self.ee_links):
            objectives.append(ik.IKObjectivePosition(link, wp.vec3(), zero_pos[ee]))
        for ee, link in enumerate(self.ee_links):
            objectives.append(
                ik.IKObjectiveRotation(
                    link,
                    wp.quat_identity(),
                    zero_rot[ee],
                    canonicalize_quat_err=False,
                )
            )
        objectives.append(
            ik.IKObjectiveJointLimit(
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
                weight=1.0,
            )
        )
        solver = ik.IKSolver(
            self.model,
            n_problems,
            objectives,
            sampler=ik.IKSampler.ROBERTS,
            n_seeds=self.seeds,
            lambda_factor=self.lambda_factor,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        return (
            solver,
            objectives[: len(self.ee_links)],
            objectives[len(self.ee_links) : 2 * len(self.ee_links)],
        )

    def _fk_targets(self, q_batch: np.ndarray):
        state = self.model.state()
        pos, rot = [], []
        for q in q_batch:
            wp.copy(self.model.joint_q, wp.array(q, dtype=wp.float32))
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)
            body_q = state.body_q.numpy()
            pos.append(body_q[self.ee_links, :3])
            rot.append(body_q[self.ee_links, 3:7])
        return np.stack(pos), np.stack(rot)

    def _eval_winners(self, solver, q_best, tgt_pos, tgt_rot):
        batch_size = q_best.shape[0]
        solver._fk_two_pass(
            self.model,
            wp.array(q_best, dtype=wp.float32),
            solver.body_q,
            solver.X_local,
            batch_size,
        )
        wp.synchronize_device()
        bq = solver.body_q.numpy()[:batch_size]
        ee = np.asarray(self.ee_links)
        pos_err = np.linalg.norm(bq[:, ee, :3] - tgt_pos, axis=-1).max(axis=-1)

        def _qmul(a, b):
            w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
            w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
            return np.stack(
                (
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                ),
                axis=-1,
            )

        tgt_conj = np.concatenate([tgt_rot[..., :1], -tgt_rot[..., 1:]], axis=-1)
        rel = _qmul(tgt_conj, bq[:, ee, 3:7])
        rot_err = (2 * np.arctan2(np.linalg.norm(rel[..., 1:], axis=-1), np.abs(rel[..., 0]))).max(axis=-1)
        success = (pos_err < self.pos_thresh_m) & (rot_err < self.ori_thresh_rad)
        return pos_err, rot_err, success

    def _capture_batch_graph(self, solver, seeds_d, winners_d):
        with wp.ScopedCapture() as cap:
            solver.step(seeds_d, winners_d, iterations=self.iterations, step_size=self.step_size)
        return cap.graph

    def run_benchmark(self):
        """
        Executes the main benchmark logic by iterating through batch sizes.
        """
        for batch in self.batch_sizes:
            solver, pos_obj, rot_obj = self._build_ik_solver(batch)

            # Prepare device arrays for the full batch
            winners_d = wp.zeros((batch, self.n_coords), dtype=wp.float32)
            seeds_in_d = wp.zeros((batch, self.n_coords), dtype=wp.float32)

            # Capture CUDA graph for the full batch operation
            if self.use_cuda_graph:
                solve_graph = self._capture_batch_graph(solver, seeds_in_d, winners_d)

            # Prepare host data (ground truth and initial seeds)
            q_gt = self._random_solutions(batch)
            tgt_p, tgt_r = self._fk_targets(q_gt)

            times = []
            for _ in range(self.repeats):
                wp.synchronize_device()
                t0 = time.perf_counter()

                # --- Main benchmark execution ---

                # Set targets for all problems in the batch
                for ee in range(len(self.ee_names)):
                    pos_obj[ee].set_target_positions(
                        wp.array(tgt_p[:, ee].astype(np.float32, copy=False), dtype=wp.vec3)
                    )
                    rot_obj[ee].set_target_rotations(
                        wp.array(tgt_r[:, ee].astype(np.float32, copy=False), dtype=wp.vec4)
                    )

                solver.reset()

                # Run the solver
                if self.use_cuda_graph and solve_graph is not None:
                    wp.capture_launch(solve_graph)
                else:
                    solver.step(seeds_in_d, winners_d, iterations=self.iterations, step_size=self.step_size)

                wp.synchronize_device()
                times.append(time.perf_counter() - t0)

            # Evaluate results
            q_best = winners_d.numpy()
            pos_e, rot_e, succ = self._eval_winners(solver, q_best, tgt_p, tgt_r)

            last_t_ms = times[-1] * 1_000.0
            if succ.any():
                pos_98 = np.percentile(pos_e[succ], 98) * 1_000.0
                rot_98 = np.percentile(rot_e[succ], 98)
            else:
                pos_98 = rot_98 = float("nan")

            self.results.append([self.asset_file.name, batch, last_t_ms, succ.mean() * 100, pos_98, rot_98])

    def print_results(self):
        """
        Formats and prints the benchmark results to the console.
        """

        def _border(widths, sep="+", glyph="-"):
            return sep + sep.join(glyph * w for w in widths) + sep

        def _row(cells, widths, aligns):
            pad = {"l": str.ljust, "r": str.rjust}
            padded = [f" {pad[a](txt, w - 2)} " for txt, w, a in zip(cells, widths, aligns, strict=False)]
            return "|" + "|".join(padded) + "|"

        header = (
            "",
            "robot",
            "batch",
            "newton-time (ms)",
            "newton-success (%)",
            "newton-pos-err (mm)",
            "newton-ori-err (rad)",
        )
        rows = [
            [
                str(i),
                r,
                str(b),
                f"{t:.6g}",
                f"{s:.3f}",
                "nan" if np.isnan(pe) else f"{pe:.6g}",
                "nan" if np.isnan(oe) else f"{oe:.6g}",
            ]
            for i, (r, b, t, s, pe, oe) in enumerate(self.results)
        ]
        widths = [max(len(cell) for cell in col) + 2 for col in zip(*([header, *rows]), strict=False)]

        print("\nReported errors are 98-percentile of successful solves\n")
        print(_border(widths))
        print(_row(header, widths, ["l"] * len(header)))
        print(_border(widths, "+", "="))
        for row in rows:
            print(_row(row, widths, ["r", "l", "r", "r", "r", "r", "r"]))
            print(_border(widths))

    def step(self):
        pass  # Not used in this benchmark

    def render(self):
        pass  # Not used in this benchmark

    def test_final(self):
        pass  # Not used in this benchmark


def main():
    parser = newton.examples.create_parser()
    parser.add_argument("--repeats", type=int, default=3, help="Number of times to run the benchmark.")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 10, 100, 1_000, 2_000],
        help="A list of batch sizes (e.g., --batch-sizes 1 10 100).",
    )
    parser.add_argument("--seed", type=int, default=123, help="RNG seed for reproducibility.")
    # non-visual example, default to null viewer
    parser.set_defaults(viewer="null")

    viewer, args = newton.examples.init(parser)

    with wp.ScopedDevice(args.device):
        example = Example(viewer, repeats=args.repeats, batch_sizes=args.batch_sizes, seed=args.seed)
        example.run_benchmark()
        example.print_results()


if __name__ == "__main__":
    main()
