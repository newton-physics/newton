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

"""Levenberg-Marquardt optimizer backend for inverse kinematics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import warp as wp

from .ik_common import IKJacobianType, _compute_costs, _eval_fk_batched, _fk_accum


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


@wp.kernel
def _accept_reject(
    cost_curr: wp.array1d(dtype=wp.float32),
    cost_prop: wp.array1d(dtype=wp.float32),
    pred_red: wp.array1d(dtype=wp.float32),
    rho_min: float,
    accept: wp.array1d(dtype=wp.int32),
):
    row = wp.tid()
    rho = (cost_curr[row] - cost_prop[row]) / (pred_red[row] + 1.0e-8)
    accept[row] = wp.int32(1) if rho >= rho_min else wp.int32(0)


@wp.kernel
def _update_lm_state(
    joint_q_proposed: wp.array2d(dtype=wp.float32),
    residuals_proposed: wp.array2d(dtype=wp.float32),
    costs_proposed: wp.array1d(dtype=wp.float32),
    accept_flags: wp.array1d(dtype=wp.int32),
    n_coords: int,
    num_residuals: int,
    lambda_factor: float,
    lambda_min: float,
    lambda_max: float,
    joint_q_current: wp.array2d(dtype=wp.float32),
    residuals_current: wp.array2d(dtype=wp.float32),
    costs: wp.array1d(dtype=wp.float32),
    lambda_values: wp.array1d(dtype=wp.float32),
):
    row = wp.tid()

    if accept_flags[row] == 1:
        for i in range(n_coords):
            joint_q_current[row, i] = joint_q_proposed[row, i]
        for i in range(num_residuals):
            residuals_current[row, i] = residuals_proposed[row, i]
        costs[row] = costs_proposed[row]
        lambda_values[row] = lambda_values[row] / lambda_factor
    else:
        new_lambda = lambda_values[row] * lambda_factor
        lambda_values[row] = wp.clamp(new_lambda, lambda_min, lambda_max)


class IKOptimizerLM:
    """
    Modular inverse-kinematics solver.

    The solver uses an adaptive Levenberg-Marquardt loop and supports
    three Jacobian back-ends:

        * **AUTODIFF**: Warp's reverse-mode autodiff for every objective.
        * **ANALYTIC**: Objective-specific analytic Jacobians only.
        * **MIXED**: Analytic where available, autodiff fallback elsewhere.

    Args:
        model (newton.Model): Singleton articulation shared by all problems.
        n_batch (int): Number of rows processed in parallel (e.g., `n_problems * n_seeds`).
        objectives (Sequence[IKObjective]): Ordered list of objectives shared by all problems. Each `IKObjective` instance can carry arrays of per-problem parameters (sized by the true problem count) and should dereference them via `problem_idx`.
        jacobian_mode (IKJacobianType, optional): Backend used in `compute_jacobian`. Defaults to IKJacobianType.AUTODIFF.
        lambda_initial (float, optional): Initial LM damping per problem. Defaults to 0.1.
        lambda_factor (float, optional): Multiplicative update factor for λ. Defaults to 2.0.
        lambda_min (float, optional): Lower clamp for λ. Defaults to 1e-5.
        lambda_max (float, optional): Upper clamp for λ. Defaults to 1e10.
        rho_min (float, optional): Acceptance threshold on predicted vs. actual reduction. Defaults to 1e-3.

    Batch Structure:
        The solver handles a batch of independent IK problem instances (possibly expanded by sampling) that all reference the same articulation (`model`) and objective list.
        Per-problem parameters (targets, weights, ...) live in problem space and are accessed through the `problem_idx` indirection supplied by `IKSolver`.
    """

    TILE_N_DOFS = None
    TILE_N_RESIDUALS = None
    _cache: ClassVar[dict[tuple[int, int], type]] = {}

    def __new__(cls, model, n_batch, objectives, *a, **kw):
        n_dofs = model.joint_dof_count
        n_residuals = sum(o.residual_dim() for o in objectives)
        arch = model.device.arch
        key = (n_dofs, n_residuals, arch)

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
        lambda_initial=0.1,
        jacobian_mode=IKJacobianType.AUTODIFF,
        lambda_factor=2.0,
        lambda_min=1e-5,
        lambda_max=1e10,
        rho_min=1e-3,
        *,
        problem_idx: wp.array | None = None,
    ):
        """
        Construct a batch IK solver.

        See class doc-string for parameter semantics.
        """

        self.model = model
        self.device = model.device
        self.n_batch = n_batch
        self.n_coords = model.joint_coord_count
        self.n_dofs = model.joint_dof_count
        self.n_residuals = sum(o.residual_dim() for o in objectives)

        self.objectives = objectives
        self.jacobian_mode = jacobian_mode
        self.has_analytic_objective = any(o.supports_analytic() for o in objectives)
        self.has_autodiff_objective = any(not o.supports_analytic() for o in objectives)

        self.lambda_initial = lambda_initial
        self.lambda_factor = lambda_factor
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.rho_min = rho_min

        if self.TILE_N_DOFS is not None:
            assert self.n_dofs == self.TILE_N_DOFS
        if self.TILE_N_RESIDUALS is not None:
            assert self.n_residuals == self.TILE_N_RESIDUALS

        grad = jacobian_mode in (IKJacobianType.AUTODIFF, IKJacobianType.MIXED)

        self._alloc_solver_buffers(grad)
        self.problem_idx = problem_idx if problem_idx is not None else self.problem_idx_identity
        self.tape = wp.Tape() if grad else None

        self._build_residual_offsets()

        self._init_objectives()
        self._init_cuda_streams()

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

    def _alloc_solver_buffers(self, grad: bool) -> None:
        device = self.device
        model = self.model

        self.qd_zero = wp.zeros((self.n_batch, self.n_dofs), dtype=wp.float32, device=device)
        self.body_q = wp.zeros((self.n_batch, model.body_count), dtype=wp.transform, requires_grad=grad, device=device)
        self.body_qd = (
            wp.zeros((self.n_batch, model.body_count), dtype=wp.spatial_vector, device=device) if grad else None
        )

        self.residuals = wp.zeros((self.n_batch, self.n_residuals), dtype=wp.float32, requires_grad=grad, device=device)
        self.residuals_proposed = wp.zeros(
            (self.n_batch, self.n_residuals), dtype=wp.float32, requires_grad=grad, device=device
        )
        self.residuals_3d = wp.zeros((self.n_batch, self.n_residuals, 1), dtype=wp.float32, device=device)

        self.jacobian = wp.zeros((self.n_batch, self.n_residuals, self.n_dofs), dtype=wp.float32, device=device)
        self.dq_dof = wp.zeros((self.n_batch, self.n_dofs), dtype=wp.float32, requires_grad=grad, device=device)

        self.joint_q_proposed = wp.zeros(
            (self.n_batch, self.n_coords), dtype=wp.float32, requires_grad=grad, device=device
        )

        self.costs = wp.zeros(self.n_batch, dtype=wp.float32, device=device)
        self.costs_proposed = wp.zeros(self.n_batch, dtype=wp.float32, device=device)
        self.lambda_values = wp.zeros(self.n_batch, dtype=wp.float32, device=device)
        self.accept_flags = wp.zeros(self.n_batch, dtype=wp.int32, device=device)
        self.pred_reduction = wp.zeros(self.n_batch, dtype=wp.float32, device=device)

        self.problem_idx_identity = wp.array(np.arange(self.n_batch, dtype=np.int32), dtype=wp.int32, device=device)

        self.X_local = wp.zeros((self.n_batch, model.joint_count), dtype=wp.transform, device=device)
        self.joint_S_s = (
            wp.zeros((self.n_batch, self.n_dofs), dtype=wp.spatial_vector, device=device)
            if self.jacobian_mode != IKJacobianType.AUTODIFF and self.has_analytic_objective
            else None
        )

    def _build_residual_offsets(self) -> None:
        offsets: list[int] = []
        offset = 0
        for obj in self.objectives:
            offsets.append(offset)
            offset += obj.residual_dim()
        self.residual_offsets = offsets

    def _ctx_solver(self, joint_q, *, residuals=None, jacobian=None) -> BatchCtx:
        ctx = BatchCtx(
            joint_q=joint_q,
            residuals=residuals if residuals is not None else self.residuals,
            fk_body_q=self.body_q,
            problem_idx=self.problem_idx,
            fk_body_qd=getattr(self, "body_qd", None),
            dq_dof=self.dq_dof,
            joint_q_proposed=self.joint_q_proposed,
            joint_qd=self.qd_zero,
            jacobian_out=jacobian if jacobian is not None else self.jacobian,
            motion_subspace=getattr(self, "joint_S_s", None),
            fk_qd_zero=self.qd_zero,
            fk_X_local=self.X_local,
        )
        self._validate_ctx_for_mode(ctx)
        return ctx

    def _validate_ctx_for_mode(self, ctx: BatchCtx) -> None:
        missing: list[str] = []

        for name in ("joint_q", "residuals", "fk_body_q", "problem_idx"):
            if getattr(ctx, name) is None:
                missing.append(name)

        mode = self.jacobian_mode
        if mode in (IKJacobianType.AUTODIFF, IKJacobianType.MIXED):
            for name in ("fk_body_qd", "dq_dof", "joint_q_proposed", "joint_qd"):
                if getattr(ctx, name) is None:
                    missing.append(name)

        needs_analytic = mode == IKJacobianType.ANALYTIC or (
            mode == IKJacobianType.MIXED and self.has_analytic_objective
        )
        if needs_analytic:
            for name in ("jacobian_out", "motion_subspace", "fk_qd_zero"):
                if getattr(ctx, name) is None:
                    missing.append(name)
            if ctx.fk_X_local is None:
                missing.append("fk_X_local")

        if missing:
            raise RuntimeError(f"solver context missing: {', '.join(missing)}")

    def _for_objectives_residuals(self, ctx: BatchCtx) -> None:
        def _do(obj, offset, body_q_view, joint_q_view, model, output_residuals, problem_idx_array):
            obj.compute_residuals(
                body_q_view,
                joint_q_view,
                model,
                output_residuals,
                offset,
                problem_idx=problem_idx_array,
            )

        self._parallel_for_objectives(
            _do,
            ctx.fk_body_q,
            ctx.joint_q,
            self.model,
            ctx.residuals,
            ctx.problem_idx,
        )

    def _residuals_autodiff(self, ctx: BatchCtx) -> None:
        _eval_fk_batched(
            self.model,
            ctx.joint_q,
            ctx.joint_qd,
            ctx.fk_body_q,
            ctx.fk_body_qd,
        )

        ctx.residuals.zero_()
        self._for_objectives_residuals(ctx)

    def _residuals_analytic(self, ctx: BatchCtx) -> None:
        self._fk_two_pass(
            self.model,
            ctx.joint_q,
            ctx.fk_body_q,
            ctx.fk_X_local,
            ctx.joint_q.shape[0],
        )

        ctx.residuals.zero_()
        self._for_objectives_residuals(ctx)

    def _jacobian_at(self, ctx: BatchCtx) -> wp.array3d:
        mode = self.jacobian_mode

        if mode == IKJacobianType.AUTODIFF:
            self._jacobian_autodiff(ctx)
            return ctx.jacobian_out

        if mode == IKJacobianType.ANALYTIC:
            self._jacobian_analytic(ctx, accumulate=False)
            return ctx.jacobian_out

        # MIXED mode
        if self.has_autodiff_objective:
            self._jacobian_autodiff(ctx)
        else:
            ctx.jacobian_out.zero_()

        if self.has_analytic_objective:
            self._jacobian_analytic(ctx, accumulate=self.has_autodiff_objective)

        return ctx.jacobian_out

    def _jacobian_autodiff(self, ctx: BatchCtx) -> None:
        if self.tape is None:
            raise RuntimeError("Autodiff Jacobian requested but tape is not initialized")

        ctx.jacobian_out.zero_()
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
            residuals_flat = ctx.residuals.flatten()

        self.tape.outputs = [residuals_flat]

        for obj, offset in zip(self.objectives, self.residual_offsets, strict=False):
            if self.jacobian_mode == IKJacobianType.MIXED and obj.supports_analytic():
                continue
            obj.compute_jacobian_autodiff(self.tape, self.model, ctx.jacobian_out, offset, ctx.dq_dof)
            self.tape.zero()

    def _jacobian_analytic(self, ctx: BatchCtx, *, accumulate: bool) -> None:
        if not accumulate:
            ctx.jacobian_out.zero_()

        ctx.fk_qd_zero.zero_()
        self._compute_motion_subspace(
            body_q=ctx.fk_body_q,
            joint_S_s_out=ctx.motion_subspace,
            joint_qd_in=ctx.fk_qd_zero,
        )

        def _emit(obj, off, body_q_view, joint_q_view, model, jac_view, motion_subspace_view):
            if obj.supports_analytic():
                obj.compute_jacobian_analytic(body_q_view, joint_q_view, model, jac_view, motion_subspace_view, off)
            elif not accumulate:
                raise ValueError(f"Objective {type(obj).__name__} does not support analytic Jacobian")

        self._parallel_for_objectives(
            _emit,
            ctx.fk_body_q,
            ctx.joint_q,
            self.model,
            ctx.jacobian_out,
            ctx.motion_subspace,
        )

    def step(self, joint_q_in, joint_q_out, iterations=10, step_size=1.0):
        """Run LM iterations using the provided joint buffers."""
        if joint_q_in.shape != (self.n_batch, self.n_coords):
            raise ValueError("joint_q_in has incompatible shape")
        if joint_q_out.shape != (self.n_batch, self.n_coords):
            raise ValueError("joint_q_out has incompatible shape")

        if joint_q_in.ptr != joint_q_out.ptr:
            wp.copy(joint_q_out, joint_q_in)

        joint_q = joint_q_out

        self.lambda_values.fill_(self.lambda_initial)
        for i in range(iterations):
            self._step(joint_q, step_size=step_size, iteration=i)

    def _compute_residuals(self, joint_q, output_residuals=None):
        buffer = output_residuals or self.residuals
        ctx = self._ctx_solver(joint_q, residuals=buffer)

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

    def _step(self, joint_q, step_size=1.0, iteration=0):
        """Execute one Levenberg-Marquardt iteration with adaptive damping."""

        ctx_curr = self._ctx_solver(joint_q)

        if iteration == 0:
            if self.jacobian_mode in (IKJacobianType.AUTODIFF, IKJacobianType.MIXED):
                self._residuals_autodiff(ctx_curr)
            else:
                self._residuals_analytic(ctx_curr)

        wp.launch(
            _compute_costs,
            dim=self.n_batch,
            inputs=[ctx_curr.residuals, self.n_residuals],
            outputs=[self.costs],
            device=self.device,
        )

        self._jacobian_at(ctx_curr)

        residuals_flat = ctx_curr.residuals.flatten()
        residuals_3d_flat = self.residuals_3d.flatten()
        wp.copy(residuals_3d_flat, residuals_flat)

        self.dq_dof.zero_()
        self._solve_tiled(
            ctx_curr.jacobian_out, self.residuals_3d, self.lambda_values, self.dq_dof, self.pred_reduction
        )

        self._integrate_dq(
            joint_q,
            dq_in=self.dq_dof,
            joint_q_out=self.joint_q_proposed,
            joint_qd_out=self.qd_zero,
            step_size=step_size,
        )

        ctx_prop = self._ctx_solver(self.joint_q_proposed, residuals=self.residuals_proposed)
        if self.jacobian_mode in (IKJacobianType.AUTODIFF, IKJacobianType.MIXED):
            self._residuals_autodiff(ctx_prop)
        else:
            self._residuals_analytic(ctx_prop)

        wp.launch(
            _compute_costs,
            dim=self.n_batch,
            inputs=[self.residuals_proposed, self.n_residuals],
            outputs=[self.costs_proposed],
            device=self.device,
        )

        wp.launch(
            _accept_reject,
            dim=self.n_batch,
            inputs=[self.costs, self.costs_proposed, self.pred_reduction, self.rho_min],
            outputs=[self.accept_flags],
            device=self.device,
        )

        wp.launch(
            _update_lm_state,
            dim=self.n_batch,
            inputs=[
                self.joint_q_proposed,
                self.residuals_proposed,
                self.costs_proposed,
                self.accept_flags,
                self.n_coords,
                self.n_residuals,
                self.lambda_factor,
                self.lambda_min,
                self.lambda_max,
            ],
            outputs=[joint_q, self.residuals, self.costs, self.lambda_values],
            device=self.device,
        )

    def reset(self):
        self.lambda_values.zero_()
        self.accept_flags.zero_()

    def compute_costs(self, joint_q):
        self._compute_residuals(joint_q)
        wp.launch(
            _compute_costs,
            dim=self.n_batch,
            inputs=[self.residuals, self.n_residuals],
            outputs=[self.costs],
            device=self.device,
        )
        return self.costs

    def _solve_tiled(self, jacobian, residuals, lambda_values, dq_dof, pred_reduction):
        raise NotImplementedError("This method should be overridden by specialized solver")

    @classmethod
    def _build_specialized(cls, key):
        """Build a specialized IKOptimizerLM subclass with tiled solver for given dimensions."""
        C, R, _ = key

        def _template(
            jacobians: wp.array3d(dtype=wp.float32),  # (n_batch, n_residuals, n_dofs)
            residuals: wp.array3d(dtype=wp.float32),  # (n_batch, n_residuals, 1)
            lambda_values: wp.array1d(dtype=wp.float32),  # (n_batch)
            # outputs
            dq_dof: wp.array2d(dtype=wp.float32),  # (n_batch, n_dofs)
            pred_reduction_out: wp.array1d(dtype=wp.float32),  # (n_batch)
        ):
            row = wp.tid()

            RES = _Specialized.TILE_N_RESIDUALS
            DOF = _Specialized.TILE_N_DOFS
            J = wp.tile_load(jacobians[row], shape=(RES, DOF))
            r = wp.tile_load(residuals[row], shape=(RES, 1))
            lam = lambda_values[row]

            Jt = wp.tile_transpose(J)
            JtJ = wp.tile_zeros(shape=(DOF, DOF), dtype=wp.float32)
            wp.tile_matmul(Jt, J, JtJ)

            diag = wp.tile_zeros(shape=(DOF,), dtype=wp.float32)
            for i in range(DOF):
                diag[i] = lam
            A = wp.tile_diag_add(JtJ, diag)
            g = wp.tile_zeros(shape=(DOF,), dtype=wp.float32)
            tmp2d = wp.tile_zeros(shape=(DOF, 1), dtype=wp.float32)
            wp.tile_matmul(Jt, r, tmp2d)
            for i in range(DOF):
                g[i] = tmp2d[i, 0]

            rhs = wp.tile_map(wp.neg, g)
            L = wp.tile_cholesky(A)
            delta = wp.tile_cholesky_solve(L, rhs)
            wp.tile_store(dq_dof[row], delta)
            lambda_delta = wp.tile_zeros(shape=(DOF,), dtype=wp.float32)
            for i in range(DOF):
                lambda_delta[i] = lam * delta[i]

            diff = wp.tile_map(wp.sub, lambda_delta, g)
            prod = wp.tile_map(wp.mul, delta, diff)
            red = wp.tile_sum(prod)[0]
            pred_reduction_out[row] = 0.5 * red

        _template.__name__ = f"_lm_solve_tiled_{C}_{R}"
        _template.__qualname__ = f"_lm_solve_tiled_{C}_{R}"
        _lm_solve_tiled = wp.kernel(enable_backward=False, module="unique")(_template)

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
            """Compute forward kinematics using two-pass algorithm.

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
                _fk_accum,
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

        class _Specialized(IKOptimizerLM):
            TILE_N_DOFS = wp.constant(C)
            TILE_N_RESIDUALS = wp.constant(R)
            TILE_THREADS = wp.constant(32)

            def _solve_tiled(self, jac, res, lam, dq, pred):
                wp.launch_tiled(
                    _lm_solve_tiled,
                    dim=[self.n_batch],
                    inputs=[jac, res, lam, dq, pred],
                    block_dim=self.TILE_THREADS,
                    device=self.device,
                )

        _Specialized.__name__ = f"IK_{C}x{R}"
        _Specialized._integrate_dq_dof = staticmethod(_integrate_dq_dof)
        _Specialized._compute_motion_subspace_2d = staticmethod(_compute_motion_subspace_2d)
        _Specialized._fk_two_pass = staticmethod(_fk_two_pass)
        return _Specialized
