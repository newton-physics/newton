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

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable

import numpy as np
import warp as wp

import newton
import newton.sim.ik as ik
import newton.utils


@dataclass(frozen=True)
class RobotCfg:
    asset: str
    file: Path
    parser: Callable[..., None]
    ee_names: tuple[str, ...]
    ee_links: tuple[int, ...]
    seeds: int
    sub_batch_size: int
    lambda_factor: float


ROBOTS: dict[str, RobotCfg] = {
    "h1": RobotCfg(
        asset="h1_description",
        file=Path("mjcf/h1.xml"),
        parser=newton.utils.parse_mjcf,
        ee_names=("left_hand", "right_hand", "left_foot", "right_foot"),
        ee_links=(15, 19, 5, 10),
        seeds=256,
        sub_batch_size=25,
        lambda_factor=2.0,
    ),
    "franka": RobotCfg(
        asset="franka_description",
        file=Path("urdfs/fr3.urdf"),
        parser=partial(newton.utils.parse_urdf, scale=1.0),
        ee_names=("ee",),
        ee_links=(9,),
        seeds=64,
        sub_batch_size=1000,
        lambda_factor=4.0,
    ),
}


@lru_cache(maxsize=64)
def _roberts_root(dim: int) -> float:
    x = 1.5
    for _ in range(20):
        f = x ** (dim + 1) - x - 1.0
        df = (dim + 1) * x**dim - 1.0
        x_next = x - f / df
        if abs(x_next - x) < 1.0e-12:
            break
        x = x_next
    return x


@wp.kernel
def _pick_best(costs: wp.array(dtype=float), n_seeds: int, best: wp.array(dtype=int)):
    q = wp.tid()
    base = q * n_seeds
    best_cost = float(1.0e30)
    best_seed = int(0)
    for s in range(n_seeds):
        c = costs[base + s]
        if c < best_cost:
            best_cost, best_seed = c, s
    best[q] = best_seed


@wp.kernel
def _scatter_winner(
    src: wp.array2d(dtype=float),
    winners: wp.array2d(dtype=float),
    best: wp.array(dtype=int),
    offs: wp.array(dtype=int),
    n_seeds: int,
    n_coords: int,
):
    sb_start = offs[0]
    q = wp.tid()
    src_idx = q * n_seeds + best[q]
    for d in range(n_coords):
        winners[sb_start + q, d] = src[src_idx, d]


class Example:
    def __init__(self, robot="h1", batch_sizes=(1, 10, 100, 1_000, 2_000), repeats=3):
        self.robot_name = robot
        self.batch_sizes = batch_sizes
        self.repeats = repeats

        self.iterations = 16
        self.step_size = 1.0
        self.pos_thresh_m = 5e-3
        self.ori_thresh_rad = 0.05

        self.device = "cuda" if wp.get_preferred_device().is_cuda else "cpu"
        wp.set_device(self.device)

        self.cfg = ROBOTS[robot]
        self.model = self._create_robot()
        self.n_coords = self.model.joint_coord_count

        self.results = []
        self.rng = np.random.default_rng()

    def _create_robot(self) -> newton.Model:
        builder = newton.ModelBuilder()
        builder.num_rigid_contacts_per_env = 0
        builder.default_shape_cfg.density = 100.0

        asset_path = newton.utils.download_asset(self.cfg.asset) / self.cfg.file
        self.cfg.parser(asset_path, builder, floating=False)

        model = builder.finalize(requires_grad=False)
        return model

    def _roberts_sequence(self, n: int, dim: int) -> np.ndarray:
        r = _roberts_root(dim)
        basis = 1.0 - 1.0 / r ** (1 + np.arange(dim))
        return ((np.arange(n)[:, None] * basis) % 1.0).astype(np.float32)

    def _random_solutions(self, n: int) -> np.ndarray:
        lower = self.model.joint_limit_lower.numpy()[: self.n_coords]
        upper = self.model.joint_limit_upper.numpy()[: self.n_coords]
        span = upper - lower

        mask = np.abs(span) > 1e5
        span[mask] = 0.0

        q = self.rng.random((n, self.n_coords)) * span + lower
        q[:, mask] = 0.0
        return q.astype(np.float32)

    def _build_ik_solver(self, max_problems: int):
        n_residuals = len(self.cfg.ee_links) * 6 + self.n_coords

        zero_pos = [wp.zeros(max_problems, dtype=wp.vec3) for _ in self.cfg.ee_links]
        zero_rot = [wp.zeros(max_problems, dtype=wp.vec4) for _ in self.cfg.ee_links]

        objectives = []

        for ee, link in enumerate(self.cfg.ee_links):
            objectives.append(ik.PositionObjective(link, wp.vec3(), zero_pos[ee], max_problems, n_residuals, ee * 3))

        for ee, link in enumerate(self.cfg.ee_links):
            objectives.append(
                ik.RotationObjective(
                    link,
                    wp.quat_identity(),
                    zero_rot[ee],
                    max_problems,
                    n_residuals,
                    len(self.cfg.ee_links) * 3 + ee * 3,
                )
            )

        objectives.append(
            ik.JointLimitObjective(
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
                max_problems,
                n_residuals,
                len(self.cfg.ee_links) * 6,
                1.0,
            )
        )

        q0 = wp.zeros((max_problems, self.n_coords), dtype=wp.float32)

        solver = ik.IKSolver(
            self.model,
            q0,
            objectives,
            lambda_factor=self.cfg.lambda_factor,
            jacobian_mode=ik.JacobianMode.ANALYTIC,
        )
        return (
            solver,
            objectives[: len(self.cfg.ee_links)],
            objectives[len(self.cfg.ee_links) : 2 * len(self.cfg.ee_links)],
        )

    def _fk_targets(self, q_batch: np.ndarray):
        state = self.model.state()
        pos, rot = [], []
        for q in q_batch:
            wp.copy(self.model.joint_q, wp.array(q, dtype=wp.float32))
            newton.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)
            body_q = state.body_q.numpy()
            pos.append(body_q[self.cfg.ee_links, :3])
            rot.append(body_q[self.cfg.ee_links, 3:7])
        return np.stack(pos), np.stack(rot)

    def _eval_winners(self, solver, q_best, tgt_pos, tgt_rot):
        sb_size = q_best.shape[0]

        solver._fk_two_pass(
            self.model,
            wp.array(q_best, dtype=wp.float32),
            solver.body_q,
            solver.X_local,
            sb_size,
        )
        wp.synchronize()

        bq = solver.body_q.numpy()[:sb_size]
        ee = np.asarray(self.cfg.ee_links)

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

    def run(self):
        for batch in self.batch_sizes:
            sub = min(self.cfg.sub_batch_size, batch)
            max_problems = sub * self.cfg.seeds

            solver, pos_obj, rot_obj = self._build_ik_solver(max_problems)

            winners_d = wp.zeros((batch, self.n_coords), dtype=wp.float32, device=self.device)
            best_d = wp.zeros(sub, dtype=int, device=self.device)
            off_d = wp.zeros(1, dtype=int, device=self.device)

            solver.solve(self.iterations, self.step_size)

            with wp.ScopedCapture() as cap:
                solver.solve(self.iterations, self.step_size)
                wp.launch(_pick_best, dim=sub, inputs=[solver.costs, self.cfg.seeds, best_d], device=self.device)
                wp.launch(
                    _scatter_winner,
                    dim=sub,
                    inputs=[solver.joint_q, winners_d, best_d, off_d, self.cfg.seeds, self.n_coords],
                    device=self.device,
                )
            solve_graph = cap.graph

            q_gt = self._random_solutions(batch)
            tgt_p, tgt_r = self._fk_targets(q_gt)

            span = (
                self.model.joint_limit_upper.numpy()[: self.n_coords]
                - self.model.joint_limit_lower.numpy()[: self.n_coords]
            )
            base = (
                self._roberts_sequence(self.cfg.seeds, self.n_coords) * span
                + self.model.joint_limit_lower.numpy()[: self.n_coords]
            ).astype(np.float32)
            starts = np.tile(base, (sub, 1))

            scratch_p = np.empty((max_problems, 3), np.float32)
            scratch_r = np.empty((max_problems, 4), np.float32)

            times = []

            for _ in range(self.repeats):
                wp.synchronize()
                t0 = time.perf_counter()

                for sb_start in range(0, batch, sub):
                    sb_size = min(sub, batch - sb_start)
                    active = sb_size * self.cfg.seeds

                    for ee in range(len(self.cfg.ee_names)):
                        scratch_p[:active] = np.repeat(tgt_p[sb_start : sb_start + sb_size, ee], self.cfg.seeds, axis=0)
                        pos_obj[ee].set_target_positions(wp.array(scratch_p, dtype=wp.vec3))

                        scratch_r[:active] = np.repeat(tgt_r[sb_start : sb_start + sb_size, ee], self.cfg.seeds, axis=0)
                        rot_obj[ee].set_target_rotations(wp.array(scratch_r, dtype=wp.vec4))

                    wp.copy(
                        solver.joint_q,
                        wp.array(
                            starts,
                            dtype=wp.float32,
                        ),
                    )

                    off_d.fill_(sb_start)
                    wp.capture_launch(solve_graph)

                wp.synchronize()
                times.append(time.perf_counter() - t0)

            q_best = winners_d.numpy()
            pos_e, rot_e, succ = [], [], []

            for sb_start in range(0, batch, sub):
                sb_size = min(sub, batch - sb_start)
                p, r, s = self._eval_winners(
                    solver,
                    q_best[sb_start : sb_start + sb_size],
                    tgt_p[sb_start : sb_start + sb_size],
                    tgt_r[sb_start : sb_start + sb_size],
                )
                pos_e.append(p)
                rot_e.append(r)
                succ.append(s)

            pos_e = np.concatenate(pos_e)
            rot_e = np.concatenate(rot_e)
            succ = np.concatenate(succ)

            last_t_ms = times[-1] * 1_000.0
            if succ.any():
                pos_98 = np.percentile(pos_e[succ], 98) * 1_000.0
                rot_98 = np.percentile(rot_e[succ], 98)
            else:
                pos_98 = rot_98 = float("nan")

            self.results.append([self.cfg.file.name, batch, last_t_ms, succ.mean() * 100, pos_98, rot_98])

    def print_results(self):
        def _border(widths, sep="+", glyph="-"):
            return sep + sep.join(glyph * w for w in widths) + sep

        def _row(cells, widths, aligns):
            pad = {"l": str.ljust, "r": str.rjust}
            padded = [f" {pad[a](txt, w - 2)} " for txt, w, a in zip(cells, widths, aligns)]
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

        widths = [max(len(cell) for cell in col) + 2 for col in zip(*([header, *rows]))]

        print("\nReported errors are 98-percentile of successful solves\n")
        print(_border(widths))
        print(_row(header, widths, ["l"] * len(header)))
        print(_border(widths, "+", "="))
        for row in rows:
            print(_row(row, widths, ["r", "l", "r", "r", "r", "r", "r"]))
            print(_border(widths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot",
        type=str,
        default="h1",
        choices=ROBOTS.keys(),
        help="Robot model to benchmark",
    )
    args = parser.parse_args()

    example = Example(robot=args.robot, repeats=3)
    example.run()
    example.print_results()


if __name__ == "__main__":
    main()
