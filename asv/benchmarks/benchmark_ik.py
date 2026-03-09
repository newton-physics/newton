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

###########################################################################
# Used for benchmarking the Newton IK solver.
#
# This module provides shared logic for IK benchmarks on the
# Franka Emika Panda robot.
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.ik as ik
import newton.utils


def create_franka_model() -> newton.Model:
    builder = newton.ModelBuilder()
    builder.num_rigid_contacts_per_world = 0
    builder.default_shape_cfg.density = 100.0
    asset_path = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3.urdf"
    builder.add_urdf(asset_path, floating=False, scale=1.0)
    return builder.finalize(requires_grad=False)


def random_solutions(model: newton.Model, n: int, rng: np.random.Generator) -> np.ndarray:
    n_coords = model.joint_coord_count
    lower = model.joint_limit_lower.numpy()[:n_coords]
    upper = model.joint_limit_upper.numpy()[:n_coords]
    span = upper - lower
    mask = np.abs(span) > 1e5
    span[mask] = 0.0
    q = rng.random((n, n_coords)) * span + lower
    q[:, mask] = 0.0
    return q.astype(np.float32)


def build_ik_solver(model: newton.Model, n_problems: int, ee_links: tuple[int, ...]):
    zero_pos = [wp.zeros(n_problems, dtype=wp.vec3) for _ in ee_links]
    zero_rot = [wp.zeros(n_problems, dtype=wp.vec4) for _ in ee_links]
    objectives = []
    for ee, link in enumerate(ee_links):
        objectives.append(ik.IKObjectivePosition(link, wp.vec3(), zero_pos[ee]))
    for ee, link in enumerate(ee_links):
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
            model.joint_limit_lower,
            model.joint_limit_upper,
            weight=1.0,
        )
    )
    solver = ik.IKSolver(
        model,
        n_problems,
        objectives,
        sampler=ik.IKSampler.ROBERTS,
        n_seeds=64,
        lambda_factor=4.0,
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )
    return (
        solver,
        objectives[: len(ee_links)],
        objectives[len(ee_links) : 2 * len(ee_links)],
    )


def fk_targets(solver, model: newton.Model, q_batch: np.ndarray, ee_links: tuple[int, ...]):
    batch_size = q_batch.shape[0]
    solver._fk_two_pass(
        model,
        wp.array(q_batch, dtype=wp.float32),
        solver.body_q,
        solver.X_local,
        batch_size,
    )
    wp.synchronize_device()
    bq = solver.body_q.numpy()[:batch_size]
    ee = np.asarray(ee_links)
    return bq[:, ee, :3].copy(), bq[:, ee, 3:7].copy()


def eval_success(solver, model, q_best, tgt_pos, tgt_rot, ee_links, pos_thresh_m, ori_thresh_rad):
    batch_size = q_best.shape[0]
    solver._fk_two_pass(
        model,
        wp.array(q_best, dtype=wp.float32),
        solver.body_q,
        solver.X_local,
        batch_size,
    )
    wp.synchronize_device()
    bq = solver.body_q.numpy()[:batch_size]
    ee = np.asarray(ee_links)
    pos_err = np.linalg.norm(bq[:, ee, :3] - tgt_pos, axis=-1).max(axis=-1)

    def _qmul(a, b):
        # Quaternions stored as (x, y, z, w) — scalar-last, matching Warp convention.
        x1, y1, z1, w1 = np.moveaxis(a, -1, 0)
        x2, y2, z2, w2 = np.moveaxis(b, -1, 0)
        return np.stack(
            (
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ),
            axis=-1,
        )

    tgt_conj = np.concatenate([-tgt_rot[..., :3], tgt_rot[..., 3:]], axis=-1)
    rel = _qmul(tgt_conj, bq[:, ee, 3:7])
    rot_err = (2 * np.arctan2(np.linalg.norm(rel[..., :3], axis=-1), np.abs(rel[..., 3]))).max(axis=-1)
    success = (pos_err < pos_thresh_m) & (rot_err < ori_thresh_rad)
    return success
