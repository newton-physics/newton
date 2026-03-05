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

"""L-BFGS optimizer backend for inverse kinematics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import warp as wp

from .ik_common import IKJacobianType, compute_costs, eval_fk_batched, fk_accum


@wp.kernel
def _scale_negate(
    src: wp.array2d(dtype=wp.float32),  # (n_batch, n_dofs)
    scale: float,
    # outputs
    dst: wp.array2d(dtype=wp.float32),  # (n_batch, n_dofs)
):
    row, dof_idx = wp.tid()
    dst[row, dof_idx] = -scale * src[row, dof_idx]


@wp.kernel
def _fan_out_problem_idx(
    batch_problem_idx: wp.array1d(dtype=wp.int32),
    out_indices: wp.array2d(dtype=wp.int32),
):
    row_idx, candidate_idx = wp.tid()
    out_indices[row_idx, candidate_idx] = batch_problem_idx[row_idx]


@wp.kernel
def _generate_candidates_velocity(
    joint_q: wp.array2d(dtype=wp.float32),  # (n_batch, n_coords)
    search_direction: wp.array2d(dtype=wp.float32),  # (n_batch, n_dofs)
    line_search_alphas: wp.array1d(dtype=wp.float32),  # (n_steps)
    # outputs
    candidate_q: wp.array3d(dtype=wp.float32),  # (n_batch, n_steps, n_coords)
    candidate_dq: wp.array3d(dtype=wp.float32),  # (n_batch, n_steps, n_dofs)
):
    row, step_idx = wp.tid()
    alpha = line_search_alphas[step_idx]

    n_coords = joint_q.shape[1]
    for coord in range(n_coords):
        candidate_q[row, step_idx, coord] = joint_q[row, coord]

    n_dofs = search_direction.shape[1]
    for dof in range(n_dofs):
        candidate_dq[row, step_idx, dof] = alpha * search_direction[row, dof]


@wp.kernel
def _apply_residual_mask(
    residuals: wp.array2d(dtype=wp.float32),
    mask: wp.array2d(dtype=wp.float32),
    seeds_out: wp.array2d(dtype=wp.float32),
):
    row, residual_idx = wp.tid()
    seeds_out[row, residual_idx] = residuals[row, residual_idx] * mask[row, residual_idx]


@wp.kernel
def _accumulate_gradients(
    base_grad: wp.array2d(dtype=wp.float32),
    add_grad: wp.array2d(dtype=wp.float32),
):
    row, dof_idx = wp.tid()
    base_grad[row, dof_idx] += add_grad[row, dof_idx]


@dataclass(slots=True)
class BatchCtx:
    joint_q: wp.array2d
    residuals: wp.array2d
    fk_body_q: wp.array2d
    problem_idx: wp.array1d

    # AUTODIFF and MIXED
    fk_body_qd: wp.array2d | None = None
    dq_dof: wp.array2d | None = None
    joint_q_proposed: wp.array2d | None = None
    joint_qd: wp.array2d | None = None

    # ANALYTIC and MIXED
    jacobian_out: wp.array3d | None = None
    motion_subspace: wp.array2d | None = None
    fk_qd_zero: wp.array2d | None = None
    fk_X_local: wp.array2d | None = None

    # MIXED-only helpers
    gradient_tmp: wp.array2d | None = None
    autodiff_mask: wp.array2d | None = None
    autodiff_seed: wp.array2d | None = None


class IKOptimizerLBFGS:
    """
    L-BFGS inverse-kinematics solver with a parallel Wolfe-conditions line search.

    This solver uses L-BFGS with a parallel line search that seeks to satisfy the
    strong Wolfe conditions for robust step selection. It supports the same three
    Jacobian back-ends as the LM solver:

    * **AUTODIFF**   — Warp's reverse-mode autodiff for every objective
    * **ANALYTIC**   — objective-specific analytic Jacobians only
    * **MIXED**      — analytic where available, autodiff fallback elsewhere

    Parameters
    ----------
    model : newton.Model
        Singleton articulation shared by all problems.
    n_batch : int
        Number of rows processed in parallel (e.g., `n_problems * n_seeds`).
    objectives : Sequence[IKObjective]
        Ordered list of objectives shared by _all_ problems.
    jacobian_mode : IKJacobianType, default IKJacobianType.AUTODIFF
        Backend used in `compute_jacobian`.
    history_len : int, default 10
        L-BFGS memory length (number of {s,y} pairs to remember).
    h0_scale : float, default 1.0
        Initial Hessian approximation scale factor.
    line_search_alphas : list[float], default [0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
        Parallel step sizes to try in line search.
    wolfe_c1 : float, default 1e-4
        Wolfe condition parameter for sufficient decrease (Armijo condition).
    wolfe_c2 : float, default 0.9
        Wolfe condition parameter for curvature condition (strong Wolfe).

    Batch structure
    ---------------
    Same as the LM solver - handles a batch of independent IK problems that all
    reference the same articulation and objectives.
    """

    TILE_N_DOFS = None
    TILE_N_RESIDUALS = None
    TILE_HISTORY_LEN = None
    TILE_N_LINE_STEPS = None
    _cache: ClassVar[dict[tuple[int, int, int, int, str], type]] = {}

    def __new__(cls, model, n_batch, objectives, *a, **kw):
        n_dofs = model.joint_dof_count
        n_residuals = sum(o.residual_dim() for o in objectives)
        history_len = kw.get("history_len", 10)
        alphas = kw.get("line_search_alphas") or [0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
        n_line_search = len(alphas)
        arch = model.device.arch
        key = (n_dofs, n_residuals, history_len, n_line_search, arch)

        spec_cls = cls._cache.get(key)
        if spec_cls is None:
            spec_cls = cls._build_specialized(key)
            cls._cache[key] = spec_cls

        return super().__new__(spec_cls)

    def __init__(
        self,
        model,
        n_batch,
        objectives,
        jacobian_mode=IKJacobianType.AUTODIFF,
        history_len=10,
        h0_scale=1.0,
        line_search_alphas=None,
        wolfe_c1=1e-4,
        wolfe_c2=0.9,
        *,
        problem_idx: wp.array | None = None,
    ):
        if line_search_alphas is None:
            line_search_alphas = [0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]

        self.model = model
        self.device = model.device
        self.n_batch = n_batch
        self.n_coords = model.joint_coord_count
        self.n_dofs = model.joint_dof_count
        self.n_residuals = sum(o.residual_dim() for o in objectives)
        self.history_len = history_len
        self.n_line_search = len(line_search_alphas)
        self.h0_scale = h0_scale
        self.wolfe_c1 = wolfe_c1
        self.wolfe_c2 = wolfe_c2

        self.objectives = objectives
        self.jacobian_mode = jacobian_mode

        if self.TILE_N_DOFS is not None:
            assert self.n_dofs == self.TILE_N_DOFS
        if self.TILE_N_RESIDUALS is not None:
            assert self.n_residuals == self.TILE_N_RESIDUALS
        if self.TILE_HISTORY_LEN is not None:
            assert self.history_len == self.TILE_HISTORY_LEN
        if self.TILE_N_LINE_STEPS is not None:
            assert self.n_line_search == self.TILE_N_LINE_STEPS

        grad = jacobian_mode in (IKJacobianType.AUTODIFF, IKJacobianType.MIXED)

        self.has_analytic_objective = any(o.supports_analytic() for o in objectives)
        self.has_autodiff_objective = any(not o.supports_analytic() for o in objectives)

        self._alloc_solver_buffers(grad)
        self.problem_idx = problem_idx if problem_idx is not None else self.problem_idx_identity
        self._alloc_line_search_buffers(grad, line_search_alphas)

        self.tape = wp.Tape() if grad else None

        self._build_residual_offsets()

        if self.jacobian_mode != IKJacobianType.AUTODIFF:
            self._alloc_line_search_analytic_buffers()
        else:
            self.candidate_jacobians = None
            self.cand_joint_S_s = None
            self.cand_X_local = None

        if self.jacobian_mode == IKJacobianType.MIXED:
            self._alloc_mixed_buffers()
        else:
            self.gradient_tmp = None
            self.candidate_gradient_tmp = None
            self.autodiff_residual_mask = None
            self.autodiff_residual_seed = None
            self.autodiff_residual_mask_candidates = None
            self.candidate_autodiff_residual_grads = None

        self._init_objectives()
        self._init_cuda_streams()

    def _alloc_solver_buffers(self, grad: bool):
        device = self.device
        model = self.model

        self.qd_zero = wp.zeros((self.n_batch, self.n_dofs), dtype=wp.float32, device=device)
        self.body_q = wp.zeros((self.n_batch, model.body_count), dtype=wp.transform, requires_grad=grad, device=device)
        self.body_qd = (
            wp.zeros((self.n_batch, model.body_count), dtype=wp.spatial_vector, device=device) if grad else None
        )
        self.joint_q_proposed = wp.zeros(
            (self.n_batch, self.n_coords), dtype=wp.float32, requires_grad=grad, device=device
        )
        self.residuals = wp.zeros((self.n_batch, self.n_residuals), dtype=wp.float32, requires_grad=grad, device=device)
        self.jacobian = wp.zeros((self.n_batch, self.n_residuals, self.n_dofs), dtype=wp.float32, device=device)
        self.dq_dof = wp.zeros((self.n_batch, self.n_dofs), dtype=wp.float32, requires_grad=grad, device=device)

        self.gradient = wp.zeros((self.n_batch, self.n_dofs), dtype=wp.float32, device=device)
        self.gradient_prev = wp.zeros((self.n_batch, self.n_dofs), dtype=wp.float32, device=device)
        self.search_direction = wp.zeros((self.n_batch, self.n_dofs), dtype=wp.float32, device=device)
        self.last_step_dq = wp.zeros((self.n_batch, self.n_dofs), dtype=wp.float32, device=device)

        self.s_history = wp.zeros((self.n_batch, self.history_len, self.n_dofs), dtype=wp.float32, device=device)
        self.y_history = wp.zeros((self.n_batch, self.history_len, self.n_dofs), dtype=wp.float32, device=device)
        self.rho_history = wp.zeros((self.n_batch, self.history_len), dtype=wp.float32, device=device)
        self.alpha_history = wp.zeros((self.n_batch, self.history_len), dtype=wp.float32, device=device)
        self.history_count = wp.zeros(self.n_batch, dtype=wp.int32, device=device)
        self.history_start = wp.zeros(self.n_batch, dtype=wp.int32, device=device)

        self.costs = wp.zeros(self.n_batch, dtype=wp.float32, device=device)
        self.problem_idx_identity = wp.array(np.arange(self.n_batch, dtype=np.int32), dtype=wp.int32, device=device)
        self.X_local = wp.zeros((self.n_batch, model.joint_count), dtype=wp.transform, device=device)

        if self.jacobian_mode != IKJacobianType.AUTODIFF and self.has_analytic_objective:
            self.joint_S_s = wp.zeros((self.n_batch, self.n_dofs), dtype=wp.spatial_vector, device=device)
        else:
            self.joint_S_s = None

    def _alloc_line_search_buffers(self, grad: bool, line_search_alphas):
        device = self.device
        model = self.model

        self.line_search_alphas = wp.array(line_search_alphas, dtype=wp.float32, device=device)

        self.candidate_q = wp.zeros((self.n_batch, self.n_line_search, self.n_coords), dtype=wp.float32, device=device)
        self.candidate_residuals = wp.zeros(
            (self.n_batch, self.n_line_search, self.n_residuals), dtype=wp.float32, device=device
        )
        if self.n_line_search > 0:
            self.candidate_problem_idx = wp.zeros((self.n_batch, self.n_line_search), dtype=wp.int32, device=device)
            wp.launch(
                _fan_out_problem_idx,
                dim=[self.n_batch, self.n_line_search],
                inputs=[self.problem_idx],
                outputs=[self.candidate_problem_idx],
                device=device,
            )
        else:
            self.candidate_problem_idx = None
        self.candidate_costs = wp.zeros((self.n_batch, self.n_line_search), dtype=wp.float32, device=device)
        self.best_step_idx = wp.zeros(self.n_batch, dtype=wp.int32, device=device)
        self.initial_slope = wp.zeros(self.n_batch, dtype=wp.float32, device=device)
        self.candidate_gradients = wp.zeros(
            (self.n_batch, self.n_line_search, self.n_dofs), dtype=wp.float32, device=device
        )
        self.candidate_slopes = wp.zeros((self.n_batch, self.n_line_search), dtype=wp.float32, device=device)

        body_count = model.body_count
        self.cand_body_q = wp.zeros(
            (self.n_batch, self.n_line_search, body_count),
            dtype=wp.transform,
            requires_grad=grad,
            device=device,
        )
        if grad:
            self.cand_body_qd = wp.zeros(
                (self.n_batch, self.n_line_search, body_count),
                dtype=wp.spatial_vector,
                device=device,
            )
        else:
            self.cand_body_qd = None

        if self.n_line_search > 0:
            self.cand_joint_q_proposed = wp.zeros(
                (self.n_batch, self.n_line_search, self.n_coords),
                dtype=wp.float32,
                requires_grad=grad,
                device=device,
            )
            self.cand_dq_dof = wp.zeros(
                (self.n_batch, self.n_line_search, self.n_dofs),
                dtype=wp.float32,
                requires_grad=grad,
                device=device,
            )
            self.cand_step_dq = wp.zeros(
                (self.n_batch, self.n_line_search, self.n_dofs),
                dtype=wp.float32,
                device=device,
            )
        else:
            self.cand_joint_q_proposed = None
            self.cand_dq_dof = None
            self.cand_step_dq = None

        if self.n_line_search > 0:
            self.cand_qd_zero = wp.zeros(
                (self.n_batch, self.n_line_search, self.n_dofs),
                dtype=wp.float32,
                device=device,
            )
        else:
            self.cand_qd_zero = None

    def _alloc_line_search_analytic_buffers(self):
        device = self.device

        if self.n_line_search > 0:
            self.candidate_jacobians = wp.zeros(
                (self.n_batch, self.n_line_search, self.n_residuals, self.n_dofs),
                dtype=wp.float32,
                device=device,
            )
        else:
            self.candidate_jacobians = None

        if self.has_analytic_objective and self.n_line_search > 0:
            self.cand_joint_S_s = wp.zeros(
                (self.n_batch, self.n_line_search, self.n_dofs),
                dtype=wp.spatial_vector,
                device=device,
            )
        else:
            self.cand_joint_S_s = None

        if self.jacobian_mode == IKJacobianType.ANALYTIC and self.n_line_search > 0:
            self.cand_X_local = wp.zeros(
                (self.n_batch, self.n_line_search, self.model.joint_count),
                dtype=wp.transform,
                device=device,
            )
        else:
            self.cand_X_local = None

    def _alloc_mixed_buffers(self):
        device = self.device

        self.gradient_tmp = wp.zeros((self.n_batch, self.n_dofs), dtype=wp.float32, device=device)
        if self.n_line_search > 0:
            self.candidate_gradient_tmp = wp.zeros(
                (self.n_batch, self.n_line_search, self.n_dofs),
                dtype=wp.float32,
                device=device,
            )
        else:
            self.candidate_gradient_tmp = None

        mask_row = np.ones(self.n_residuals, dtype=np.float32)
        for obj, offset in zip(self.objectives, self.residual_offsets, strict=False):
            width = obj.residual_dim()
            if obj.supports_analytic():
                mask_row[offset : offset + width] = 0.0

        if self.n_batch > 0:
            mask_matrix = np.tile(mask_row, (self.n_batch, 1))
        else:
            mask_matrix = np.zeros((0, self.n_residuals), dtype=np.float32)

        self.autodiff_residual_mask = wp.array(mask_matrix, dtype=wp.float32, device=device)
        self.autodiff_residual_seed = wp.zeros((self.n_batch, self.n_residuals), dtype=wp.float32, device=device)

        if self.n_line_search > 0:
            B = self.n_batch * self.n_line_search
            if B > 0:
                mask_candidates = np.tile(mask_row, (B, 1))
            else:
                mask_candidates = np.zeros((0, self.n_residuals), dtype=np.float32)
            self.autodiff_residual_mask_candidates = wp.array(mask_candidates, dtype=wp.float32, device=device)
            self.candidate_autodiff_residual_grads = wp.zeros(
                (self.n_batch, self.n_line_search, self.n_residuals), dtype=wp.float32, device=device
            )
        else:
            self.autodiff_residual_mask_candidates = None
            self.candidate_autodiff_residual_grads = None

    def _build_residual_offsets(self):
        self.residual_offsets = []
        off = 0
        for obj in self.objectives:
            self.residual_offsets.append(off)
            off += obj.residual_dim()

    def _ctx_solver(self, joint_q, *, residuals=None) -> BatchCtx:
        """Build a context for operations on the solver batch."""
        ctx = BatchCtx(
            joint_q=joint_q,
            residuals=residuals if residuals is not None else self.residuals,
            fk_body_q=self.body_q,
            problem_idx=self.problem_idx,
            fk_body_qd=self.body_qd,
            dq_dof=self.dq_dof,
            joint_q_proposed=self.joint_q_proposed,
            joint_qd=self.qd_zero,
            jacobian_out=self.jacobian,
            motion_subspace=self.joint_S_s,
            fk_qd_zero=self.qd_zero,
            fk_X_local=self.X_local,
            gradient_tmp=self.gradient_tmp,
            autodiff_mask=self.autodiff_residual_mask,
            autodiff_seed=self.autodiff_residual_seed,
        )
        self._validate_ctx(
            ctx,
            label="solver",
            require_autodiff=self.jacobian_mode in (IKJacobianType.AUTODIFF, IKJacobianType.MIXED),
            require_analytic=(
                self.jacobian_mode == IKJacobianType.ANALYTIC
                or (self.jacobian_mode == IKJacobianType.MIXED and self.has_analytic_objective)
            ),
            require_fk_x_local=self.jacobian_mode == IKJacobianType.ANALYTIC,
        )
        return ctx

    def _ctx_candidates(self) -> BatchCtx:
        """Build a context for the flattened line-search candidate batch."""
        if self.n_line_search == 0:
            raise RuntimeError("line-search context requested without candidate buffers")

        P = self.n_batch
        S = self.n_line_search
        B = P * S

        def _reshape2(arr):
            return arr.reshape((B, arr.shape[-1]))

        def _reshape3(arr):
            return arr.reshape((B, arr.shape[-2], arr.shape[-1]))

        cand_body_qd = getattr(self, "cand_body_qd", None)
        cand_dq_dof = getattr(self, "cand_dq_dof", None)
        cand_joint_q_proposed = getattr(self, "cand_joint_q_proposed", None)
        cand_qd_zero = getattr(self, "cand_qd_zero", None)
        candidate_jacobians = getattr(self, "candidate_jacobians", None)
        cand_joint_S_s = getattr(self, "cand_joint_S_s", None)
        cand_X_local = getattr(self, "cand_X_local", None)
        candidate_gradient_tmp = getattr(self, "candidate_gradient_tmp", None)
        autodiff_mask_candidates = getattr(self, "autodiff_residual_mask_candidates", None)
        candidate_autodiff_residual_grads = getattr(self, "candidate_autodiff_residual_grads", None)

        problem_idx_flat = self.candidate_problem_idx.flatten()

        ctx = BatchCtx(
            joint_q=_reshape2(self.candidate_q),
            residuals=_reshape2(self.candidate_residuals),
            fk_body_q=_reshape2(self.cand_body_q),
            problem_idx=problem_idx_flat,
            fk_body_qd=_reshape2(cand_body_qd) if cand_body_qd is not None else None,
            dq_dof=_reshape2(cand_dq_dof) if cand_dq_dof is not None else None,
            joint_q_proposed=_reshape2(cand_joint_q_proposed) if cand_joint_q_proposed is not None else None,
            joint_qd=_reshape2(cand_qd_zero) if cand_qd_zero is not None else None,
            jacobian_out=_reshape3(candidate_jacobians) if candidate_jacobians is not None else None,
            motion_subspace=_reshape2(cand_joint_S_s) if cand_joint_S_s is not None else None,
            fk_qd_zero=_reshape2(cand_qd_zero) if cand_qd_zero is not None else None,
            fk_X_local=_reshape2(cand_X_local) if cand_X_local is not None else None,
            gradient_tmp=_reshape2(candidate_gradient_tmp) if candidate_gradient_tmp is not None else None,
            autodiff_mask=_reshape2(autodiff_mask_candidates) if autodiff_mask_candidates is not None else None,
            autodiff_seed=_reshape2(candidate_autodiff_residual_grads)
            if candidate_autodiff_residual_grads is not None
            else None,
        )
        self._validate_ctx(
            ctx,
            label="candidates",
            require_autodiff=self.jacobian_mode in (IKJacobianType.AUTODIFF, IKJacobianType.MIXED),
            require_analytic=(
                self.jacobian_mode == IKJacobianType.ANALYTIC
                or (self.jacobian_mode == IKJacobianType.MIXED and self.has_analytic_objective)
            ),
            require_fk_x_local=self.jacobian_mode == IKJacobianType.ANALYTIC,
        )
        return ctx

    def _validate_ctx(
        self,
        ctx: BatchCtx,
        *,
        label: str,
        require_autodiff: bool,
        require_analytic: bool,
        require_fk_x_local: bool,
    ) -> None:
        missing: list[str] = []

        if ctx.joint_q is None:
            missing.append("joint_q")
        if ctx.residuals is None:
            missing.append("residuals")
        if ctx.fk_body_q is None:
            missing.append("fk_body_q")
        if ctx.problem_idx is None:
            missing.append("problem_idx")

        if require_autodiff:
            if ctx.fk_body_qd is None:
                missing.append("fk_body_qd")
            if ctx.dq_dof is None:
                missing.append("dq_dof")
            if ctx.joint_q_proposed is None:
                missing.append("joint_q_proposed")
            if ctx.joint_qd is None:
                missing.append("joint_qd")
            if self.jacobian_mode == IKJacobianType.MIXED and ctx.autodiff_mask is None:
                missing.append("autodiff_mask")
            if self.jacobian_mode == IKJacobianType.MIXED and ctx.autodiff_seed is None:
                missing.append("autodiff_seed")

        if require_analytic:
            if ctx.jacobian_out is None:
                missing.append("jacobian_out")
            if ctx.motion_subspace is None:
                missing.append("motion_subspace")
            if ctx.fk_qd_zero is None:
                missing.append("fk_qd_zero")
            if self.jacobian_mode == IKJacobianType.MIXED and self.has_analytic_objective and ctx.gradient_tmp is None:
                missing.append("gradient_tmp")
            if require_fk_x_local and ctx.fk_X_local is None:
                missing.append("fk_X_local")

        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"{label} context missing required buffers: {joined}")

    def _gradient_at(self, ctx: BatchCtx, out_grad: wp.array2d):
        mode = self.jacobian_mode

        if mode in (IKJacobianType.AUTODIFF, IKJacobianType.MIXED):
            self._grad_autodiff(ctx, out_grad)

        if mode == IKJacobianType.ANALYTIC:
            self._grad_analytic(ctx, out_grad, accumulate=False)
        elif mode == IKJacobianType.MIXED and self.has_analytic_objective:
            self._grad_analytic(ctx, out_grad, accumulate=True)

    def _grad_autodiff(self, ctx: BatchCtx, out_grad: wp.array2d):
        batch = ctx.joint_q.shape[0]

        self.tape.reset()
        self.tape.gradients = {}
        ctx.dq_dof.zero_()

        with self.tape:
            self._integrate_dq(
                ctx.joint_q,
                dq_in=ctx.dq_dof,
                joint_q_out=ctx.joint_q_proposed,
                joint_qd_out=ctx.joint_qd,
            )

            res_ctx = BatchCtx(
                joint_q=ctx.joint_q_proposed,
                residuals=ctx.residuals,
                fk_body_q=ctx.fk_body_q,
                problem_idx=ctx.problem_idx,
                fk_body_qd=ctx.fk_body_qd,
                joint_qd=ctx.joint_qd,
            )
            self._residuals_autodiff(res_ctx)
            residuals_2d = ctx.residuals

        self.tape.outputs = [residuals_2d]

        if ctx.autodiff_mask is not None and ctx.autodiff_seed is not None:
            wp.launch(
                _apply_residual_mask,
                dim=[batch, self.n_residuals],
                inputs=[ctx.residuals, ctx.autodiff_mask],
                outputs=[ctx.autodiff_seed],
                device=self.device,
            )
            seed = ctx.autodiff_seed
            self.tape.backward(grads={residuals_2d: seed})
        else:
            self.tape.backward(grads={residuals_2d: residuals_2d})

        wp.copy(out_grad, self.tape.gradients[ctx.dq_dof])
        self.tape.zero()

    def _grad_analytic(self, ctx: BatchCtx, out_grad: wp.array2d, *, accumulate: bool):
        if not accumulate:
            self._residuals_analytic(ctx)

        ctx.fk_qd_zero.zero_()
        ctx.jacobian_out.zero_()

        self._compute_motion_subspace(
            body_q=ctx.fk_body_q,
            joint_S_s_out=ctx.motion_subspace,
            joint_qd_in=ctx.fk_qd_zero,
        )

        def _emit_jac(obj, off, body_q_view, q_view, model, J_view, S_view):
            if obj.supports_analytic():
                obj.compute_jacobian_analytic(body_q_view, q_view, model, J_view, S_view, off)

        self._parallel_for_objectives(
            _emit_jac,
            ctx.fk_body_q,
            ctx.joint_q,
            self.model,
            ctx.jacobian_out,
            ctx.motion_subspace,
        )

        target = ctx.gradient_tmp if accumulate else out_grad
        if target is None:
            target = out_grad
        elif accumulate:
            target.zero_()

        wp.launch_tiled(
            self._compute_gradient_jtr_tiled,
            dim=ctx.joint_q.shape[0],
            inputs=[ctx.jacobian_out, ctx.residuals],
            outputs=[target],
            block_dim=self.TILE_THREADS,
            device=self.device,
        )

        if accumulate and target is not out_grad:
            wp.launch(
                _accumulate_gradients,
                dim=[ctx.joint_q.shape[0], self.n_dofs],
                inputs=[out_grad, target],
                device=self.device,
            )

    def _for_objectives_residuals(self, ctx: BatchCtx):
        def _do(obj, offset, body_q_view, joint_q_view, model, output_residuals, base_idx_array):
            obj.compute_residuals(
                body_q_view,
                joint_q_view,
                model,
                output_residuals,
                offset,
                problem_idx=base_idx_array,
            )

        self._parallel_for_objectives(
            _do,
            ctx.fk_body_q,
            ctx.joint_q,
            self.model,
            ctx.residuals,
            ctx.problem_idx,
        )

    def _residuals_autodiff(self, ctx: BatchCtx):
        eval_fk_batched(
            self.model,
            ctx.joint_q,
            ctx.joint_qd,
            ctx.fk_body_q,
            ctx.fk_body_qd,
        )

        ctx.residuals.zero_()
        self._for_objectives_residuals(ctx)

    def _residuals_analytic(self, ctx: BatchCtx):
        self._fk_two_pass(
            self.model,
            ctx.joint_q,
            ctx.fk_body_q,
            ctx.fk_X_local,
            ctx.joint_q.shape[0],
        )

        ctx.residuals.zero_()
        self._for_objectives_residuals(ctx)

    def _init_objectives(self):
        """Allocate any per-objective buffers that must live on `self.device`."""
        for obj, offset in zip(self.objectives, self.residual_offsets, strict=False):
            obj.set_batch_layout(self.n_residuals, offset, self.n_batch)
            obj.bind_device(self.device)
            if self.jacobian_mode == IKJacobianType.MIXED:
                mode = IKJacobianType.ANALYTIC if obj.supports_analytic() else IKJacobianType.AUTODIFF
            else:
                mode = self.jacobian_mode
            obj.init_buffers(model=self.model, jacobian_mode=mode)

    def _init_cuda_streams(self):
        """Allocate per-objective Warp streams and sync events."""
        self.objective_streams = []
        self.sync_events = []

        if self.device.is_cuda:
            for _ in range(len(self.objectives)):
                stream = wp.Stream(self.device)
                event = wp.Event(self.device)
                self.objective_streams.append(stream)
                self.sync_events.append(event)
        else:
            self.objective_streams = [None] * len(self.objectives)
            self.sync_events = [None] * len(self.objectives)

    def _parallel_for_objectives(self, fn, *extra):
        """Run <fn(obj, offset, *extra)> across objectives on parallel CUDA streams."""
        if self.device.is_cuda:
            main = wp.get_stream(self.device)
            init_evt = main.record_event()
            for obj, offset, obj_stream, sync_event in zip(
                self.objectives, self.residual_offsets, self.objective_streams, self.sync_events, strict=False
            ):
                obj_stream.wait_event(init_evt)
                with wp.ScopedStream(obj_stream):
                    fn(obj, offset, *extra)
                obj_stream.record_event(sync_event)
            for sync_event in self.sync_events:
                main.wait_event(sync_event)
        else:
            for obj, offset in zip(self.objectives, self.residual_offsets, strict=False):
                fn(obj, offset, *extra)

    def step(self, joint_q_in, joint_q_out, iterations=50):
        """Run L-BFGS iterations using the provided joint buffers."""
        if joint_q_in.shape != (self.n_batch, self.n_coords):
            raise ValueError("joint_q_in has incompatible shape")
        if joint_q_out.shape != (self.n_batch, self.n_coords):
            raise ValueError("joint_q_out has incompatible shape")

        if joint_q_in.ptr != joint_q_out.ptr:
            wp.copy(joint_q_out, joint_q_in)

        joint_q = joint_q_out

        for i in range(iterations):
            self._step(joint_q, iteration=i)

    def reset(self):
        self.history_count.zero_()
        self.history_start.zero_()
        self.s_history.zero_()
        self.y_history.zero_()
        self.rho_history.zero_()
        self.alpha_history.zero_()
        self.gradient.zero_()
        self.gradient_prev.zero_()
        self.search_direction.zero_()
        self.last_step_dq.zero_()
        self.best_step_idx.zero_()
        self.costs.zero_()
        if self.cand_step_dq is not None:
            self.cand_step_dq.zero_()

    def compute_costs(self, joint_q):
        self._compute_residuals(joint_q)
        wp.launch(
            compute_costs,
            dim=self.n_batch,
            inputs=[self.residuals, self.n_residuals],
            outputs=[self.costs],
            device=self.device,
        )
        return self.costs

    def _compute_residuals(self, joint_q, residuals_out=None):
        residuals = residuals_out if residuals_out is not None else self.residuals
        ctx = self._ctx_solver(joint_q, residuals=residuals)

        if self.jacobian_mode in (IKJacobianType.AUTODIFF, IKJacobianType.MIXED):
            self._residuals_autodiff(ctx)
        else:
            self._residuals_analytic(ctx)

        return ctx.residuals

    def _compute_motion_subspace(self, *, body_q, joint_S_s_out, joint_qd_in):
        n_joints = self.model.joint_count
        batch = body_q.shape[0]
        wp.launch(
            self._compute_motion_subspace_2d,
            dim=[batch, n_joints],
            inputs=[
                self.model.joint_type,
                self.model.joint_parent,
                self.model.joint_qd_start,
                joint_qd_in,
                self.model.joint_axis,
                self.model.joint_dof_dim,
                body_q,
                self.model.joint_X_p,
            ],
            outputs=[
                joint_S_s_out,
            ],
            device=self.device,
        )

    def _integrate_dq(
        self,
        joint_q,
        *,
        dq_in,
        joint_q_out,
        joint_qd_out,
        step_size=1.0,
    ):
        batch = joint_q.shape[0]

        wp.launch(
            self._integrate_dq_dof,
            dim=[batch, self.model.joint_count],
            inputs=[
                self.model.joint_type,
                self.model.joint_q_start,
                self.model.joint_qd_start,
                self.model.joint_dof_dim,
                joint_q,
                dq_in,
                joint_qd_out,
                step_size,
            ],
            outputs=[
                joint_q_out,
                joint_qd_out,
            ],
            device=self.device,
        )
        joint_qd_out.zero_()

    def _step(self, joint_q, iteration=0):
        """Execute one L-BFGS iteration."""
        self.compute_costs(joint_q)

        ctx = self._ctx_solver(joint_q)
        self._gradient_at(ctx, self.gradient)

        if iteration == 0:
            wp.copy(self.gradient_prev, self.gradient)
            wp.launch(
                _scale_negate,
                dim=[self.n_batch, self.n_dofs],
                inputs=[self.gradient, 1e-2],
                outputs=[self.last_step_dq],
                device=self.device,
            )
            self._integrate_dq(
                joint_q,
                dq_in=self.last_step_dq,
                joint_q_out=self.joint_q_proposed,
                joint_qd_out=self.qd_zero,
                step_size=1.0,
            )
            wp.copy(joint_q, self.joint_q_proposed)
            return

        self._update_history()
        self._compute_search_direction()
        self._compute_initial_slope()

        wp.copy(self.gradient_prev, self.gradient)

        self._line_search(joint_q)
        self._line_search_select_best(joint_q)

    def _compute_initial_slope(self):
        """Compute and store dot(gradient, search_direction) for the current state."""
        wp.launch_tiled(
            self._compute_slope_tiled,
            dim=[self.n_batch],
            inputs=[self.gradient, self.search_direction],
            outputs=[self.initial_slope],
            block_dim=self.TILE_THREADS,
            device=self.device,
        )

    def _compute_search_direction(self):
        """Compute L-BFGS search direction using two-loop recursion."""
        wp.launch_tiled(
            self._compute_search_direction_tiled,
            dim=[self.n_batch],
            inputs=[
                self.gradient,
                self.s_history,
                self.y_history,
                self.rho_history,
                self.alpha_history,
                self.history_count,
                self.history_start,
                self.h0_scale,
            ],
            outputs=[
                self.search_direction,
            ],
            block_dim=self.TILE_THREADS,
            device=self.device,
        )

    def _update_history(self):
        """Update L-BFGS history with new s_k and y_k pairs."""
        # if self.device.is_cuda:
        wp.launch_tiled(
            self._update_history_tiled,
            dim=[self.n_batch],
            inputs=[
                self.last_step_dq,
                self.gradient,
                self.gradient_prev,
                self.history_len,
            ],
            outputs=[
                self.s_history,
                self.y_history,
                self.rho_history,
                self.history_count,
                self.history_start,
            ],
            block_dim=self.TILE_THREADS,
            device=self.device,
        )

    def _line_search(self, joint_q):
        """
        Generate candidate configurations and compute their costs and gradients
        to check the Wolfe conditions.
        """
        P = self.n_batch
        S = self.n_line_search
        B = P * S

        if S == 0:
            return

        wp.launch(
            _generate_candidates_velocity,
            dim=[P, S],
            inputs=[joint_q, self.search_direction, self.line_search_alphas],
            outputs=[self.candidate_q, self.cand_dq_dof],
            device=self.device,
        )

        if self.cand_step_dq is not None:
            wp.copy(self.cand_step_dq, self.cand_dq_dof)

        cand_ctx = self._ctx_candidates()

        self._integrate_dq(
            cand_ctx.joint_q,
            dq_in=cand_ctx.dq_dof,
            joint_q_out=cand_ctx.joint_q_proposed,
            joint_qd_out=cand_ctx.joint_qd,
            step_size=1.0,
        )
        wp.copy(cand_ctx.joint_q, cand_ctx.joint_q_proposed)

        n_candidates = self.n_batch * self.n_line_search
        candidate_gradients_flat = self.candidate_gradients.reshape((n_candidates, -1))

        # NOTE: _gradient_at also computes residuals (needed for costs)
        self._gradient_at(cand_ctx, candidate_gradients_flat)

        wp.launch(
            compute_costs,
            dim=B,
            inputs=[cand_ctx.residuals, self.n_residuals],
            outputs=[self.candidate_costs.flatten()],
            device=self.device,
        )

        wp.launch_tiled(
            self._compute_slope_candidates_tiled,
            dim=[P, S],
            inputs=[
                self.candidate_gradients,
                self.search_direction,
            ],
            outputs=[
                self.candidate_slopes,
            ],
            block_dim=self.TILE_THREADS,
            device=self.device,
        )

    def _line_search_select_best(self, joint_q):
        """Select the best step size based on Wolfe conditions and update joint_q."""
        if self.n_line_search == 0:
            return

        wp.copy(self.joint_q_proposed, joint_q)

        wp.launch_tiled(
            self._select_best_step_tiled,
            dim=[self.n_batch],
            inputs=[
                self.candidate_costs,
                self.cand_step_dq,
                self.costs,
                self.initial_slope,
                self.candidate_slopes,
                self.line_search_alphas,
                self.wolfe_c1,
                self.wolfe_c2,
            ],
            outputs=[
                self.best_step_idx,
                self.last_step_dq,
            ],
            block_dim=self.TILE_THREADS,
            device=self.device,
        )

        self._integrate_dq(
            self.joint_q_proposed,
            dq_in=self.last_step_dq,
            joint_q_out=joint_q,
            joint_qd_out=self.qd_zero,
            step_size=1.0,
        )

    @classmethod
    def _build_specialized(cls, key):
        """Build a specialized IKOptimizerLBFGS subclass with tiled kernels for given dimensions."""
        C, R, M_HIST, N_LINE_SEARCH, _ARCH = key

        def _compute_slope_template(
            # inputs
            gradient: wp.array2d(dtype=wp.float32),  # (n_batch, n_dofs)
            search_direction: wp.array2d(dtype=wp.float32),  # (n_batch, n_dofs)
            # outputs
            slope_out: wp.array1d(dtype=wp.float32),  # (n_batch,)
        ):
            row = wp.tid()
            DOF = _Specialized.TILE_N_DOFS

            g = wp.tile_load(gradient[row], shape=(DOF,))
            p = wp.tile_load(search_direction[row], shape=(DOF,))

            slope = wp.tile_sum(wp.tile_map(wp.mul, g, p))

            slope_out[row] = slope[0]

        def _compute_slope_candidates_template(
            # inputs
            candidate_gradient: wp.array3d(dtype=wp.float32),  # (n_batch, n_line_steps, n_dofs)
            search_direction: wp.array2d(dtype=wp.float32),  # (n_batch, n_dofs)
            # outputs
            slope_out: wp.array2d(dtype=wp.float32),  # (n_batch, n_line_steps)
        ):
            row, step_idx = wp.tid()
            DOF = _Specialized.TILE_N_DOFS

            g = wp.tile_load(candidate_gradient[row, step_idx], shape=(DOF,))
            p = wp.tile_load(search_direction[row], shape=(DOF,))

            slope = wp.tile_sum(wp.tile_map(wp.mul, g, p))

            slope_out[row, step_idx] = slope[0]

        def _compute_gradient_jtr_template(
            # inputs
            jacobian: wp.array3d(dtype=wp.float32),  # (n_batch, n_residuals, n_dofs)
            residuals: wp.array2d(dtype=wp.float32),  # (n_batch, n_residuals)
            # outputs
            gradient: wp.array2d(dtype=wp.float32),  # (n_batch, n_dofs)
        ):
            row = wp.tid()

            RES = _Specialized.TILE_N_RESIDUALS
            DOF = _Specialized.TILE_N_DOFS

            J = wp.tile_load(jacobian[row], shape=(RES, DOF))
            r = wp.tile_load(residuals[row], shape=(RES,))

            Jt = wp.tile_transpose(J)
            r_2d = wp.tile_reshape(r, shape=(RES, 1))
            grad_2d = wp.tile_matmul(Jt, r_2d)
            grad_1d = wp.tile_reshape(grad_2d, shape=(DOF,))

            wp.tile_store(gradient[row], grad_1d)

        def _compute_search_direction_template(
            # inputs
            gradient: wp.array2d(dtype=wp.float32),  # (n_batch, n_dofs)
            s_history: wp.array3d(dtype=wp.float32),  # (n_batch, history_len, n_dofs)
            y_history: wp.array3d(dtype=wp.float32),  # (n_batch, history_len, n_dofs)
            rho_history: wp.array2d(dtype=wp.float32),  # (n_batch, history_len)
            alpha_history: wp.array2d(dtype=wp.float32),  # (n_batch, history_len)
            history_count: wp.array1d(dtype=wp.int32),  # (n_batch)
            history_start: wp.array1d(dtype=wp.int32),  # (n_batch)
            h0_scale: float,  # scalar
            # outputs
            search_direction: wp.array2d(dtype=wp.float32),  # (n_batch, n_dofs)
        ):
            row = wp.tid()
            DOF = _Specialized.TILE_N_DOFS
            M_HIST = _Specialized.TILE_HISTORY_LEN

            q = wp.tile_load(gradient[row], shape=(DOF,), storage="shared")
            count = history_count[row]
            start = history_start[row]

            # First loop: backward through history
            for i in range(count):
                idx = (start + count - 1 - i) % M_HIST
                s_i = wp.tile_load(s_history[row, idx], shape=(DOF,), storage="shared")
                rho_i = rho_history[row, idx]

                s_dot_q = wp.tile_sum(wp.tile_map(wp.mul, s_i, q))
                alpha_i = rho_i * s_dot_q[0]
                alpha_history[row, idx] = alpha_i

                y_i = wp.tile_load(y_history[row, idx], shape=(DOF,), storage="shared")

                for j in range(DOF):
                    q[j] = q[j] - alpha_i * y_i[j]

            # Apply initial Hessian approximation in-place
            for j in range(DOF):
                q[j] = h0_scale * q[j]

            # Second loop: forward through history
            for i in range(count):
                idx = (start + i) % M_HIST
                y_i = wp.tile_load(y_history[row, idx], shape=(DOF,), storage="shared")
                s_i = wp.tile_load(s_history[row, idx], shape=(DOF,), storage="shared")
                rho_i = rho_history[row, idx]
                alpha_i = alpha_history[row, idx]

                y_dot_q = wp.tile_sum(wp.tile_map(wp.mul, y_i, q))
                beta = rho_i * y_dot_q[0]
                diff = alpha_i - beta

                for j in range(DOF):
                    q[j] = q[j] + diff * s_i[j]

            # Store negative gradient (descent direction)
            for j in range(DOF):
                q[j] = -q[j]

            wp.tile_store(search_direction[row], q)

        def _update_history_template(
            # inputs
            last_step: wp.array2d(dtype=wp.float32),  # (n_batch, n_dofs)
            gradient: wp.array2d(dtype=wp.float32),  # (n_batch, n_dofs)
            gradient_prev: wp.array2d(dtype=wp.float32),  # (n_batch, n_dofs)
            history_len: int,
            # outputs
            s_history: wp.array3d(dtype=wp.float32),
            y_history: wp.array3d(dtype=wp.float32),
            rho_history: wp.array2d(dtype=wp.float32),
            history_count: wp.array1d(dtype=wp.int32),
            history_start: wp.array1d(dtype=wp.int32),
        ):
            row = wp.tid()
            DOF = _Specialized.TILE_N_DOFS

            s_k = wp.tile_load(last_step[row], shape=(DOF,))

            g_curr = wp.tile_load(gradient[row], shape=(DOF,))
            g_prev = wp.tile_load(gradient_prev[row], shape=(DOF,))
            y_k = wp.tile_map(wp.sub, g_curr, g_prev)

            y_dot_s_tile = wp.tile_sum(wp.tile_map(wp.mul, y_k, s_k))
            y_dot_s = y_dot_s_tile[0]

            # Check curvature condition to ensure Hessian approximation is positive definite
            if y_dot_s > 1e-8:
                rho_k = 1.0 / y_dot_s

                count = history_count[row]
                start = history_start[row]

                write_idx = (start + count) % history_len
                if count < history_len:
                    history_count[row] = count + 1
                else:
                    history_start[row] = (start + 1) % history_len

                wp.tile_store(s_history[row, write_idx], s_k)
                wp.tile_store(y_history[row, write_idx], y_k)
                rho_history[row, write_idx] = rho_k

        def _select_best_step_template(
            # inputs
            candidate_costs: wp.array2d(dtype=wp.float32),  # (n_batch, n_line_steps)
            candidate_step: wp.array3d(dtype=wp.float32),  # (n_batch, n_line_steps, n_dofs)
            cost_initial: wp.array1d(dtype=wp.float32),  # (n_batch)
            slope_initial: wp.array1d(dtype=wp.float32),  # (n_batch)
            candidate_slopes: wp.array2d(dtype=wp.float32),  # (n_batch, n_line_steps)
            line_search_alphas: wp.array1d(dtype=wp.float32),  # (n_line_steps)
            wolfe_c1: float,  # scalar
            wolfe_c2: float,  # scalar
            # outputs
            best_step_idx_out: wp.array1d(dtype=wp.int32),  # (n_batch)
            last_step_out: wp.array2d(dtype=wp.float32),  # (n_batch, n_dofs)
        ):
            row = wp.tid()
            N_STEPS = _Specialized.TILE_N_LINE_STEPS
            DOF = _Specialized.TILE_N_DOFS

            cost_k = cost_initial[row]
            slope_k = slope_initial[row]

            best_idx = int(-1)

            # Search backwards for the largest step size satisfying Wolfe conditions
            for i in range(N_STEPS - 1, -1, -1):
                cost_new = candidate_costs[row, i]
                alpha = line_search_alphas[i]

                # Armijo (Sufficient Decrease) Condition
                armijo_ok = cost_new <= cost_k + wolfe_c1 * alpha * slope_k

                # Strong Curvature Condition
                slope_new = candidate_slopes[row, i]
                curvature_ok = wp.abs(slope_new) <= wolfe_c2 * wp.abs(slope_k)

                if armijo_ok and curvature_ok:
                    best_idx = i
                    break

            # Fallback: If no step satisfies Wolfe, choose the one with the minimum cost.
            if best_idx == -1:
                costs = wp.tile_load(candidate_costs[row], shape=(N_STEPS,), storage="shared")
                argmin_tile = wp.tile_argmin(costs)
                best_idx = argmin_tile[0]

            accept_idx = best_idx
            if best_idx >= 0:
                cost_best = candidate_costs[row, best_idx]
                if cost_best >= cost_k:
                    accept_idx = -1

            best_step_idx_out[row] = accept_idx

            if accept_idx >= 0:
                best_step_vec = wp.tile_load(candidate_step[row, accept_idx], shape=(DOF,), storage="shared")
                wp.tile_store(last_step_out[row], best_step_vec)
            else:
                zero_vec = wp.tile_zeros(shape=(DOF,), dtype=wp.float32)
                wp.tile_store(last_step_out[row], zero_vec)

        _compute_slope_template.__name__ = f"_compute_slope_tiled_{C}"
        _compute_slope_template.__qualname__ = f"_compute_slope_tiled_{C}"
        _compute_slope_tiled = wp.kernel(enable_backward=False, module="unique")(_compute_slope_template)

        _compute_slope_candidates_template.__name__ = f"_compute_slope_candidates_tiled_{C}_{N_LINE_SEARCH}"
        _compute_slope_candidates_template.__qualname__ = f"_compute_slope_candidates_tiled_{C}_{N_LINE_SEARCH}"
        _compute_slope_candidates_tiled = wp.kernel(enable_backward=False, module="unique")(
            _compute_slope_candidates_template
        )

        _compute_gradient_jtr_template.__name__ = f"_compute_gradient_jtr_tiled_{C}_{R}"
        _compute_gradient_jtr_template.__qualname__ = f"_compute_gradient_jtr_tiled_{C}_{R}"
        _compute_gradient_jtr_tiled = wp.kernel(enable_backward=False, module="unique")(_compute_gradient_jtr_template)

        _compute_search_direction_template.__name__ = f"_compute_search_direction_tiled_{C}_{M_HIST}"
        _compute_search_direction_template.__qualname__ = f"_compute_search_direction_tiled_{C}_{M_HIST}"
        _compute_search_direction_tiled = wp.kernel(enable_backward=False, module="unique")(
            _compute_search_direction_template
        )

        _update_history_template.__name__ = f"_update_history_tiled_{C}_{M_HIST}"
        _update_history_template.__qualname__ = f"_update_history_tiled_{C}_{M_HIST}"
        _update_history_tiled = wp.kernel(enable_backward=False, module="unique")(_update_history_template)

        _select_best_step_template.__name__ = f"_select_best_step_tiled_{C}_{N_LINE_SEARCH}"
        _select_best_step_template.__qualname__ = f"_select_best_step_tiled_{C}_{N_LINE_SEARCH}"
        _select_best_step_tiled = wp.kernel(enable_backward=False, module="unique")(_select_best_step_template)

        # late-import jcalc_motion, jcalc_transform to avoid circular import error
        from ...solvers.featherstone.kernels import (  # noqa: PLC0415
            jcalc_integrate,
            jcalc_motion,
            jcalc_transform,
        )

        @wp.kernel
        def _integrate_dq_dof(
            # model-wide
            joint_type: wp.array1d(dtype=wp.int32),  # (n_joints)
            joint_q_start: wp.array1d(dtype=wp.int32),  # (n_joints + 1)
            joint_qd_start: wp.array1d(dtype=wp.int32),  # (n_joints + 1)
            joint_dof_dim: wp.array2d(dtype=wp.int32),  # (n_joints, 2)  → (lin, ang)
            # per-row
            joint_q_curr: wp.array2d(dtype=wp.float32),  # (n_batch, n_coords)
            joint_qd_curr: wp.array2d(dtype=wp.float32),  # (n_batch, n_dofs)  (typically all-zero)
            dq_dof: wp.array2d(dtype=wp.float32),  # (n_batch, n_dofs)  ← LM update (q̇)
            dt: float,  # LM step (usually 1.0)
            # outputs
            joint_q_out: wp.array2d(dtype=wp.float32),  # (n_batch, n_coords)
            joint_qd_out: wp.array2d(dtype=wp.float32),  # (n_batch, n_dofs)
        ):
            """
            Integrate the candidate update `dq_dof` (interpreted as a joint-space
            velocity times `dt`) into a new configuration.

            q_out  = integrate(q_curr, dq_dof)

            One thread handles one joint of one batch row. All joint types supported by
            `jcalc_integrate` (revolute, prismatic, ball, free, D6, ...) work out of the
            box.
            """
            row, joint_idx = wp.tid()

            # Static joint metadata
            t = joint_type[joint_idx]
            coord_start = joint_q_start[joint_idx]
            dof_start = joint_qd_start[joint_idx]
            lin_axes = joint_dof_dim[joint_idx, 0]
            ang_axes = joint_dof_dim[joint_idx, 1]

            # Views into the current batch row
            q_row = joint_q_curr[row]
            qd_row = joint_qd_curr[row]  # typically zero
            delta_row = dq_dof[row]  # update vector

            q_out_row = joint_q_out[row]
            qd_out_row = joint_qd_out[row]

            # Treat `delta_row` as acceleration with dt=1:
            #   qd_new = 0 + delta           (qd ← delta)
            #   q_new  = q + qd_new * dt     (q ← q + delta)
            jcalc_integrate(
                t,
                q_row,
                qd_row,
                delta_row,  # passed as joint_qdd
                coord_start,
                dof_start,
                lin_axes,
                ang_axes,
                dt,
                q_out_row,
                qd_out_row,
            )

        @wp.kernel(module="unique")
        def _compute_motion_subspace_2d(
            joint_type: wp.array1d(dtype=wp.int32),  # (n_joints)
            joint_parent: wp.array1d(dtype=wp.int32),  # (n_joints)
            joint_qd_start: wp.array1d(dtype=wp.int32),  # (n_joints + 1)
            joint_qd: wp.array2d(dtype=wp.float32),  # (n_batch, n_joint_dof_count)
            joint_axis: wp.array1d(dtype=wp.vec3),  # (n_joint_dof_count)
            joint_dof_dim: wp.array2d(dtype=wp.int32),  # (n_joints, 2)
            body_q: wp.array2d(dtype=wp.transform),  # (n_batch, n_bodies)
            joint_X_p: wp.array1d(dtype=wp.transform),  # (n_joints)
            # outputs
            joint_S_s: wp.array2d(dtype=wp.spatial_vector),  # (n_batch, n_joint_dof_count)
        ):
            row, joint_idx = wp.tid()

            type = joint_type[joint_idx]
            parent = joint_parent[joint_idx]
            qd_start = joint_qd_start[joint_idx]

            X_pj = joint_X_p[joint_idx]
            X_wpj = X_pj
            if parent >= 0:
                X_wpj = body_q[row, parent] * X_pj

            lin_axis_count = joint_dof_dim[joint_idx, 0]
            ang_axis_count = joint_dof_dim[joint_idx, 1]

            joint_qd_1d = joint_qd[row]
            S_s_out = joint_S_s[row]

            jcalc_motion(
                type,
                joint_axis,
                lin_axis_count,
                ang_axis_count,
                X_wpj,
                joint_qd_1d,
                qd_start,
                S_s_out,
            )

        @wp.kernel(module="unique")
        def _fk_local(
            joint_type: wp.array1d(dtype=wp.int32),  # (n_joints)
            joint_q: wp.array2d(dtype=wp.float32),  # (n_batch, n_coords)
            joint_q_start: wp.array1d(dtype=wp.int32),  # (n_joints + 1)
            joint_qd_start: wp.array1d(dtype=wp.int32),  # (n_joints + 1)
            joint_axis: wp.array1d(dtype=wp.vec3),  # (n_axes)
            joint_dof_dim: wp.array2d(dtype=wp.int32),  # (n_joints, 2)  → (lin, ang)
            joint_X_p: wp.array1d(dtype=wp.transform),  # (n_joints)
            joint_X_c: wp.array1d(dtype=wp.transform),  # (n_joints)
            # outputs
            X_local_out: wp.array2d(dtype=wp.transform),  # (n_batch, n_joints)
        ):
            row, local_joint_idx = wp.tid()

            t = joint_type[local_joint_idx]
            q_start = joint_q_start[local_joint_idx]
            axis_start = joint_qd_start[local_joint_idx]
            lin_axes = joint_dof_dim[local_joint_idx, 0]
            ang_axes = joint_dof_dim[local_joint_idx, 1]

            X_j = jcalc_transform(
                t,
                joint_axis,
                axis_start,
                lin_axes,
                ang_axes,
                joint_q[row],  # 1-D row slice
                q_start,
            )

            X_rel = joint_X_p[local_joint_idx] * X_j * wp.transform_inverse(joint_X_c[local_joint_idx])
            X_local_out[row, local_joint_idx] = X_rel

        def _fk_two_pass(model, joint_q, body_q, X_local, n_batch):
            """
            Compute forward kinematics using two-pass algorithm.

            Args:
                model: newton.Model instance
                joint_q: 2D array [n_batch, joint_coord_count]
                body_q: 2D array [n_batch, body_count] (output)
                X_local: 2D array [n_batch, joint_count] (workspace)
                n_batch: Number of rows to process
            """
            wp.launch(
                _fk_local,
                dim=[n_batch, model.joint_count],
                inputs=[
                    model.joint_type,
                    joint_q,
                    model.joint_q_start,
                    model.joint_qd_start,
                    model.joint_axis,
                    model.joint_dof_dim,
                    model.joint_X_p,
                    model.joint_X_c,
                ],
                outputs=[
                    X_local,
                ],
                device=model.device,
            )

            wp.launch(
                fk_accum,
                dim=[n_batch, model.joint_count],
                inputs=[
                    model.joint_parent,
                    X_local,
                ],
                outputs=[
                    body_q,
                ],
                device=model.device,
            )

        class _Specialized(IKOptimizerLBFGS):
            TILE_N_DOFS = wp.constant(C)
            TILE_N_RESIDUALS = wp.constant(R)
            TILE_HISTORY_LEN = wp.constant(M_HIST)
            TILE_N_LINE_STEPS = wp.constant(N_LINE_SEARCH)
            TILE_THREADS = wp.constant(32)

        _Specialized.__name__ = f"LBFGS_Wolfe_{C}x{R}x{M_HIST}x{N_LINE_SEARCH}"
        _Specialized._compute_gradient_jtr_tiled = staticmethod(_compute_gradient_jtr_tiled)
        _Specialized._compute_slope_tiled = staticmethod(_compute_slope_tiled)
        _Specialized._compute_slope_candidates_tiled = staticmethod(_compute_slope_candidates_tiled)
        _Specialized._compute_search_direction_tiled = staticmethod(_compute_search_direction_tiled)
        _Specialized._update_history_tiled = staticmethod(_update_history_tiled)
        _Specialized._select_best_step_tiled = staticmethod(_select_best_step_tiled)
        _Specialized._integrate_dq_dof = staticmethod(_integrate_dq_dof)
        _Specialized._compute_motion_subspace_2d = staticmethod(_compute_motion_subspace_2d)
        _Specialized._fk_two_pass = staticmethod(_fk_two_pass)

        return _Specialized
