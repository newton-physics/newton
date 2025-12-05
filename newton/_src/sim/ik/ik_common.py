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

"""Common enums and utility kernels shared across IK components."""

from enum import Enum as _Enum

import warp as wp

from ..articulation import eval_single_articulation_fk


class IKJacobianMode(_Enum):
    """
    Specifies the backend used for Jacobian computation in inverse kinematics.
    """

    AUTODIFF = "autodiff"
    """Use Warp's reverse-mode autodiff for every objective."""

    ANALYTIC = "analytic"
    """Use analytic Jacobians for objectives that support them."""

    MIXED = "mixed"
    """Use analytic Jacobians where available, otherwise use autodiff."""


@wp.kernel
def _eval_fk_articulation_batched(
    articulation_start: wp.array1d(dtype=wp.int32),
    joint_q: wp.array2d(dtype=wp.float32),
    joint_qd: wp.array2d(dtype=wp.float32),
    joint_q_start: wp.array1d(dtype=wp.int32),
    joint_qd_start: wp.array1d(dtype=wp.int32),
    joint_type: wp.array1d(dtype=wp.int32),
    joint_parent: wp.array1d(dtype=wp.int32),
    joint_child: wp.array1d(dtype=wp.int32),
    joint_X_p: wp.array1d(dtype=wp.transform),
    joint_X_c: wp.array1d(dtype=wp.transform),
    joint_axis: wp.array1d(dtype=wp.vec3),
    joint_dof_dim: wp.array2d(dtype=wp.int32),
    body_com: wp.array1d(dtype=wp.vec3),
    body_q: wp.array2d(dtype=wp.transform),
    body_qd: wp.array2d(dtype=wp.spatial_vector),
):
    problem_idx, articulation_idx = wp.tid()

    joint_start = articulation_start[articulation_idx]
    joint_end = articulation_start[articulation_idx + 1]

    eval_single_articulation_fk(
        joint_start,
        joint_end,
        joint_q[problem_idx],
        joint_qd[problem_idx],
        joint_q_start,
        joint_qd_start,
        joint_type,
        joint_parent,
        joint_child,
        joint_X_p,
        joint_X_c,
        joint_axis,
        joint_dof_dim,
        body_com,
        False,
        body_q[problem_idx],
        body_qd[problem_idx],
    )


def _eval_fk_batched(model, joint_q, joint_qd, body_q, body_qd):
    """Compute batched forward kinematics for a set of articulations."""
    n_problems = joint_q.shape[0]
    wp.launch(
        kernel=_eval_fk_articulation_batched,
        dim=[n_problems, model.articulation_count],
        inputs=[
            model.articulation_start,
            joint_q,
            joint_qd,
            model.joint_q_start,
            model.joint_qd_start,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            model.joint_dof_dim,
            model.body_com,
        ],
        outputs=[body_q, body_qd],
        device=model.device,
    )


@wp.kernel
def _fk_accum(
    joint_parent: wp.array1d(dtype=wp.int32),
    X_local: wp.array2d(dtype=wp.transform),
    body_q: wp.array2d(dtype=wp.transform),
):
    problem_idx, local_joint_idx = wp.tid()
    Xw = X_local[problem_idx, local_joint_idx]
    parent = joint_parent[local_joint_idx]
    while parent >= 0:
        Xw = X_local[problem_idx, parent] * Xw
        parent = joint_parent[parent]
    body_q[problem_idx, local_joint_idx] = Xw


@wp.kernel
def _compute_costs(
    residuals: wp.array2d(dtype=wp.float32),
    num_residuals: int,
    costs: wp.array1d(dtype=wp.float32),
):
    problem_idx = wp.tid()
    cost = float(0.0)
    for i in range(num_residuals):
        r = residuals[problem_idx, i]
        cost += r * r
    costs[problem_idx] = cost
