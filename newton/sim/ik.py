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

from enum import Enum
from typing import ClassVar

import numpy as np
import warp as wp

from .articulation import eval_single_articulation_fk
from .joints import (
    JOINT_FIXED,
    JOINT_PRISMATIC,
    JOINT_REVOLUTE,
)


class JacobianMode(Enum):
    AUTODIFF = "autodiff"
    ANALYTIC = "analytic"
    MIXED = "mixed"


class IKSolver:
    """
    Modular inverse-kinematics solver.

    The solver uses an adaptive Levenberg-Marquardt loop and supports
    three Jacobian back-ends:

    * **AUTODIFF**   — Warp's reverse-mode autodiff for every objective
    * **ANALYTIC**   — objective-specific analytic Jacobians only
    * **MIXED**      — analytic where available, autodiff fallback elsewhere

    Parameters
    ----------
    model : newton.Model
        Singleton articulation shared by _all_ problems.
    joint_q : wp.array2d[float32] shape (n_problems, model.joint_coord_count)
        Initial joint coordinates, one row per problem.
        **Modified in place.**
    objectives : Sequence[IKObjective]
        Ordered list of objectives shared by _all_ problems.
        Each `IKObjective` instance can carry arrays of per-problem
        parameters (e.g. an array of target positions of length `n_problems`).
    jacobian_mode : JacobianMode, default JacobianMode.AUTODIFF
        Backend used in `compute_jacobian`.
    lambda_initial : float, default 0.1
        Initial LM damping per problem.
    lambda_factor : float, default 2.0
        Multiplicative update factor for λ.
    lambda_min : float, default 1e-5
        Lower clamp for λ.
    lambda_max : float, default 1e10
        Upper clamp for λ.
    rho_min : float, default 1e-3
        Acceptance threshold on predicted vs. actual reduction.

    Batch structure
    ------
    The solver handles a batch of independent IK problems that all reference the same
    articulation (`model`) and the same list of objective objects.
    What varies from problem to problem are (1) the corresponding row in `joint_q`, and
    (2) any per-problem data stored internally by the objectives, e.g., an array
    of target positions. Nothing else is duplicated.

    - Shared across problems: `model`, `objectives`
    - Per-problem data: each row of `joint_q`, objective parameters (e.g. targets)


    Supported joint types
    ------
    ANALYTIC and MIXED modes currently only support models with revolute, prismatic, and fixed joints.
    For more complex joint types (e.g., free, ball, D6), use AUTODIFF mode.
    Velocity-space optimization is planned for broader analytic support in future updates.
    """

    TILE_N_COORDS = None
    TILE_N_RESIDUALS = None
    _cache: ClassVar[dict[tuple[int, int], type]] = {}

    def __new__(cls, model, joint_q, objectives, *a, **kw):
        n_coords = model.joint_coord_count
        n_residuals = sum(o.residual_dim() for o in objectives)
        arch = model.device.arch
        key = (n_coords, n_residuals, arch)

        spec_cls = cls._cache.get(key)
        if spec_cls is None:
            spec_cls = cls._build_specialised(key)
            cls._cache[key] = spec_cls

        return super().__new__(spec_cls)

    def __init__(
        self,
        model,
        joint_q,
        objectives,
        lambda_initial=0.1,
        jacobian_mode=JacobianMode.AUTODIFF,
        lambda_factor=2.0,
        lambda_min=1e-5,
        lambda_max=1e10,
        rho_min=1e-3,
    ):
        """
        Construct a batch IK solver.

        See class doc-string for parameter semantics.
        """

        self.model = model
        self.device = model.device
        self.n_problems, self.n_coords = joint_q.shape
        self.n_dofs = model.joint_dof_count
        self.n_residuals = sum(o.residual_dim() for o in objectives)
        assert self.n_coords == model.joint_coord_count, (
            f"Coordinate count mismatch: expected {model.joint_coord_count}, got {self.n_coords}"
        )

        self.objectives = objectives
        self.jacobian_mode = jacobian_mode

        if self.jacobian_mode in (JacobianMode.ANALYTIC, JacobianMode.MIXED):
            joint_types = set(self.model.joint_type.numpy())
            supported = {JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_FIXED}
            unsupported = joint_types - supported
            if unsupported:
                raise ValueError(
                    f"Analytic/Mixed Jacobians currently only support revolute, prismatic, and fixed joints. "
                    f"Unsupported types found: {unsupported}. Use AUTODIFF mode as a workaround."
                )

        self.lambda_initial = lambda_initial
        self.lambda_factor = lambda_factor
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.rho_min = rho_min

        if self.TILE_N_COORDS is not None:
            assert self.n_coords == self.TILE_N_COORDS
        if self.TILE_N_RESIDUALS is not None:
            assert self.n_residuals == self.TILE_N_RESIDUALS

        grad = jacobian_mode in (JacobianMode.AUTODIFF, JacobianMode.MIXED)

        self.joint_q = joint_q
        self.joint_qd = wp.zeros((self.n_problems, self.n_dofs), dtype=wp.float32, device=self.device)
        self.body_q = wp.zeros(
            (self.n_problems, model.body_count), dtype=wp.transform, requires_grad=grad, device=self.device
        )
        if grad:
            self.body_qd = wp.zeros((self.n_problems, model.body_count), dtype=wp.spatial_vector, device=self.device)

        self.residuals = wp.zeros(
            (self.n_problems, self.n_residuals), dtype=wp.float32, requires_grad=grad, device=self.device
        )
        self.jacobian = wp.zeros(
            (self.n_problems, self.n_residuals, self.n_coords), dtype=wp.float32, device=self.device
        )
        self.delta_q = wp.zeros((self.n_problems, self.n_coords), dtype=wp.float32, device=self.device)
        self.residuals_3d = wp.zeros((self.n_problems, self.n_residuals, 1), dtype=wp.float32, device=self.device)
        self.residuals_proposed = wp.zeros_like(self.residuals, device=self.device)
        self.joint_q_proposed = wp.zeros_like(self.joint_q, device=self.device)
        self.costs = wp.zeros(self.n_problems, dtype=wp.float32, device=self.device)
        self.costs_proposed = wp.zeros_like(self.costs, device=self.device)
        self.lambda_values = wp.zeros(self.n_problems, dtype=wp.float32, device=self.device)
        self.accept_flags = wp.zeros(self.n_problems, dtype=wp.int32, device=self.device)
        self.pred_reduction = wp.zeros(self.n_problems, dtype=wp.float32, device=self.device)

        self.tape = wp.Tape() if grad else None

        if jacobian_mode != JacobianMode.AUTODIFF and any(o.supports_analytic() for o in objectives):
            self.joint_S_s = wp.zeros((self.n_problems, self.n_dofs), dtype=wp.spatial_vector, device=self.device)
        self.X_local = wp.zeros((self.n_problems, model.joint_count), dtype=wp.transform, device=self.device)

        off = 0
        self.residual_offsets = []
        for o in objectives:
            self.residual_offsets.append(off)
            off += o.residual_dim()

        self._init_objectives()
        self._init_cuda_streams()

    def _init_objectives(self):
        """Allocate any per-objective buffers that must live on `self.device`."""
        for obj in self.objectives:
            obj.bind_device(self.device)
            if self.jacobian_mode == JacobianMode.MIXED:
                mode = JacobianMode.ANALYTIC if obj.supports_analytic() else JacobianMode.AUTODIFF
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
                self.objectives, self.residual_offsets, self.objective_streams, self.sync_events
            ):
                obj_stream.wait_event(init_evt)
                with wp.ScopedStream(obj_stream):
                    fn(obj, offset, *extra)
                obj_stream.record_event(sync_event)
            for sync_event in self.sync_events:
                main.wait_event(sync_event)
        else:
            for obj, offset in zip(self.objectives, self.residual_offsets):
                fn(obj, offset, *extra)

    def solve(self, iterations=10, step_size=1.0):
        """
        Run the Levenberg-Marquardt loop.

        Parameters
        ----------
        iterations : int, default 10
            Maximum number of outer iterations.
        step_size : float, default 1.0
            Multiplicative scale on deltaq before applying the proposal.

        Side-effects
        ------------
        Updates `self.joint_q` in-place with the converged coordinates.
        """
        self.lambda_values.fill_(self.lambda_initial)
        for i in range(iterations):
            self._step(step_size=step_size, iteration=i)

    def compute_residuals(self, joint_q=None, output_residuals=None):
        joint_q = joint_q or self.joint_q
        output_residuals = output_residuals or self.residuals

        if self.jacobian_mode in [JacobianMode.AUTODIFF, JacobianMode.MIXED]:
            _eval_fk_batched(self.model, joint_q, self.joint_qd, self.body_q, self.body_qd)
        else:
            self._fk_two_pass(self.model, joint_q, self.body_q, self.X_local, self.n_problems)

        output_residuals.zero_()

        def _do(obj, off, body_q, joint_q, model, output_residuals):
            obj.compute_residuals(body_q, joint_q, model, output_residuals, off)

        self._parallel_for_objectives(_do, self.body_q, joint_q, self.model, output_residuals)

        return output_residuals

    def compute_jacobian(self):
        self.jacobian.zero_()

        if self.jacobian_mode == JacobianMode.AUTODIFF:
            self.tape.reset()
            with self.tape:
                residuals_2d = self.compute_residuals(self.joint_q)
                current_residuals_wp = residuals_2d.flatten()

            self.tape.outputs = [current_residuals_wp]

            for obj, offset in zip(self.objectives, self.residual_offsets):
                obj.compute_jacobian_autodiff(self.tape, self.model, self.jacobian, offset, self.joint_q)
                self.tape.zero()

        elif self.jacobian_mode == JacobianMode.ANALYTIC:
            n_joints = self.model.joint_count

            wp.launch(
                self._compute_motion_subspace_2d,
                dim=self.n_problems * n_joints,
                inputs=[
                    self.n_problems,
                    self.model.joint_type,
                    self.model.joint_parent,
                    self.model.joint_qd_start,
                    self.joint_qd,
                    self.model.joint_axis,
                    self.model.joint_dof_dim,
                    self.body_q,
                    self.model.joint_X_p,
                ],
                outputs=[
                    self.joint_S_s,
                ],
                device=self.device,
            )

            def _do_analytic(obj, off, body_q, q, model, jac, joint_S_s):
                if obj.supports_analytic():
                    obj.compute_jacobian_analytic(body_q, q, model, jac, joint_S_s, off)
                else:
                    raise ValueError(f"Objective {type(obj).__name__} does not support analytic Jacobian")

            self._parallel_for_objectives(
                _do_analytic, self.body_q, self.joint_q, self.model, self.jacobian, self.joint_S_s
            )

        elif self.jacobian_mode == JacobianMode.MIXED:
            self.tape.reset()
            need_autodiff = any(not obj.supports_analytic() for obj in self.objectives)
            need_analytic = any(obj.supports_analytic() for obj in self.objectives)

            if need_autodiff:
                with self.tape:
                    residuals_2d = self.compute_residuals(self.joint_q)
                    current_residuals_wp = residuals_2d.flatten()
                self.tape.outputs = [current_residuals_wp]

            if need_analytic:
                n_joints = self.model.joint_count
                wp.launch(
                    self._compute_motion_subspace_2d,
                    dim=self.n_problems * n_joints,
                    inputs=[
                        self.n_problems,
                        self.model.joint_type,
                        self.model.joint_parent,
                        self.model.joint_qd_start,
                        self.joint_qd,
                        self.model.joint_axis,
                        self.model.joint_dof_dim,
                        self.body_q,
                        self.model.joint_X_p,
                    ],
                    outputs=[
                        self.joint_S_s,
                    ],
                    device=self.device,
                )

            for obj, offset in zip(self.objectives, self.residual_offsets):
                if obj.supports_analytic():
                    obj.compute_jacobian_analytic(
                        self.body_q, self.joint_q, self.model, self.jacobian, self.joint_S_s, offset
                    )
                else:
                    obj.compute_jacobian_autodiff(self.tape, self.model, self.jacobian, offset, self.joint_q)

        return self.jacobian

    def _step(self, step_size=1.0, iteration=0):
        """Execute one Levenberg-Marquardt iteration with adaptive damping."""
        if iteration == 0:
            self.compute_residuals()
        wp.launch(
            _compute_costs,
            dim=self.n_problems,
            inputs=[self.residuals, self.n_residuals],
            outputs=[self.costs],
            device=self.device,
        )

        self.compute_jacobian()
        residuals_flat = self.residuals.flatten()
        flat_3d_view = self.residuals_3d.flatten()
        wp.copy(flat_3d_view, residuals_flat)

        self.delta_q.zero_()

        self._solve_tiled(self.jacobian, self.residuals_3d, self.lambda_values, self.delta_q, self.pred_reduction)
        total_coords = self.n_problems * self.n_coords
        wp.launch(
            _compute_q_proposed,
            dim=total_coords,
            inputs=[self.joint_q, self.delta_q, step_size, self.n_coords],
            outputs=[self.joint_q_proposed],
            device=self.device,
        )

        self.compute_residuals(self.joint_q_proposed, self.residuals_proposed)
        wp.launch(
            _compute_costs,
            dim=self.n_problems,
            inputs=[self.residuals_proposed, self.n_residuals],
            outputs=[self.costs_proposed],
            device=self.device,
        )

        wp.launch(
            _accept_reject,
            dim=self.n_problems,
            inputs=[self.costs, self.costs_proposed, self.pred_reduction, self.rho_min],
            outputs=[self.accept_flags],
            device=self.device,
        )

        wp.launch(
            _update_lm_state,
            dim=self.n_problems,
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
            outputs=[self.joint_q, self.residuals, self.costs, self.lambda_values],
            device=self.device,
        )

    def _solve_tiled(self, jacobian, residuals, lambda_values, delta_q, pred_reduction):
        raise NotImplementedError("This method should be overridden by specialized solver")

    @classmethod
    def _build_specialised(cls, key):
        """Build a specialized IKSolver subclass with tiled solver for given dimensions."""
        C, R, _ = key

        @wp.kernel(enable_backward=False, module="unique")
        def _lm_solve_tiled(
            jacobians: wp.array3d(dtype=wp.float32),  # (n_problems, n_residuals, n_coords)
            residuals: wp.array3d(dtype=wp.float32),  # (n_problems, n_residuals, 1)
            lambda_values: wp.array1d(dtype=wp.float32),  # (n_problems)
            # outputs
            delta_q: wp.array2d(dtype=wp.float32),  # (n_problems, n_coords)
            pred_reduction_out: wp.array1d(dtype=wp.float32),  # (n_problems)
        ):
            problem_idx = wp.tid()

            RES = _Specialised.TILE_N_RESIDUALS
            COORD = _Specialised.TILE_N_COORDS
            J = wp.tile_load(jacobians[problem_idx], shape=(RES, COORD))
            r = wp.tile_load(residuals[problem_idx], shape=(RES, 1))
            lam = lambda_values[problem_idx]
            Jt = wp.tile_transpose(J)
            JtJ = wp.tile_zeros(shape=(COORD, COORD), dtype=wp.float32)
            wp.tile_matmul(Jt, J, JtJ)

            diag = wp.tile_zeros(shape=(COORD,), dtype=wp.float32)
            for i in range(COORD):
                diag[i] = lam
            A = wp.tile_diag_add(JtJ, diag)
            g = wp.tile_zeros(shape=(COORD,), dtype=wp.float32)
            tmp2d = wp.tile_zeros(shape=(COORD, 1), dtype=wp.float32)
            wp.tile_matmul(Jt, r, tmp2d)
            for i in range(COORD):
                g[i] = tmp2d[i, 0]

            rhs = wp.tile_map(wp.neg, g)
            L = wp.tile_cholesky(A)
            delta = wp.tile_cholesky_solve(L, rhs)
            wp.tile_store(delta_q[problem_idx], delta)
            lambda_delta = wp.tile_zeros(shape=(COORD,), dtype=wp.float32)
            for i in range(COORD):
                lambda_delta[i] = lam * delta[i]

            diff = wp.tile_map(wp.sub, lambda_delta, g)
            prod = wp.tile_map(wp.mul, delta, diff)
            red = wp.tile_sum(prod)[0]
            pred_reduction_out[problem_idx] = 0.5 * red

        # late-import jcalc_motion, jcalc_transform to avoid circular import error
        from newton.solvers.featherstone.kernels import (  # noqa: PLC0415
            jcalc_motion,
            jcalc_transform,
        )

        @wp.kernel(module="unique")
        def _compute_motion_subspace_2d(
            n_problems: int,
            joint_type: wp.array1d(dtype=wp.int32),  # (n_joints)
            joint_parent: wp.array1d(dtype=wp.int32),  # (n_joints)
            joint_qd_start: wp.array1d(dtype=wp.int32),  # (n_joints + 1)
            joint_qd: wp.array2d(dtype=wp.float32),  # (n_problems, n_joint_dof_count)
            joint_axis: wp.array1d(dtype=wp.vec3),  # (n_joint_dof_count)
            joint_dof_dim: wp.array2d(dtype=wp.int32),  # (n_joints, 2)
            body_q: wp.array2d(dtype=wp.transform),  # (n_problems, n_bodies)
            joint_X_p: wp.array1d(dtype=wp.transform),  # (n_joints)
            # outputs
            joint_S_s: wp.array2d(dtype=wp.spatial_vector),  # (n_problems, n_joint_dof_count)
        ):
            tid = wp.tid()

            n_joints = len(joint_type)
            problem_idx = tid / n_joints
            joint_idx = tid % n_joints

            if problem_idx >= n_problems:
                return

            type = joint_type[joint_idx]
            parent = joint_parent[joint_idx]
            qd_start = joint_qd_start[joint_idx]

            X_pj = joint_X_p[joint_idx]
            X_wpj = X_pj
            if parent >= 0:
                X_wpj = body_q[problem_idx, parent] * X_pj

            lin_axis_count = joint_dof_dim[joint_idx, 0]
            ang_axis_count = joint_dof_dim[joint_idx, 1]

            joint_qd_1d = joint_qd[problem_idx]
            S_s_out = joint_S_s[problem_idx]

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
            joint_q: wp.array2d(dtype=wp.float32),  # (n_problems, n_coords)
            joint_q_start: wp.array1d(dtype=wp.int32),  # (n_joints + 1)
            joint_qd_start: wp.array1d(dtype=wp.int32),  # (n_joints + 1)
            joint_axis: wp.array1d(dtype=wp.vec3),  # (n_axes)
            joint_dof_dim: wp.array2d(dtype=wp.int32),  # (n_joints, 2)  → (lin, ang)
            joint_X_p: wp.array1d(dtype=wp.transform),  # (n_joints)
            joint_X_c: wp.array1d(dtype=wp.transform),  # (n_joints)
            joint_count: int,
            # outputs
            X_local_out: wp.array2d(dtype=wp.transform),  # (n_problems, n_joints)
        ):
            global_joint_idx = wp.tid()
            problem_idx = global_joint_idx / joint_count
            local_joint_idx = global_joint_idx % joint_count

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
                joint_q[problem_idx],  # 1-D row slice
                q_start,
            )

            X_rel = joint_X_p[local_joint_idx] * X_j * wp.transform_inverse(joint_X_c[local_joint_idx])
            X_local_out[problem_idx, local_joint_idx] = X_rel

        def _fk_two_pass(model, joint_q, body_q, X_local, n_problems):
            """Compute forward kinematics using two-pass algorithm.

            Args:
                model: newton.Model instance
                joint_q: 2D array [n_problems, joint_coord_count]
                body_q: 2D array [n_problems, body_count] (output)
                X_local: 2D array [n_problems, joint_count] (workspace)
                n_problems: Number of problems
            """
            total_joints = n_problems * model.joint_count
            wp.launch(
                _fk_local,
                dim=total_joints,
                inputs=[
                    model.joint_type,
                    joint_q,
                    model.joint_q_start,
                    model.joint_qd_start,
                    model.joint_axis,
                    model.joint_dof_dim,
                    model.joint_X_p,
                    model.joint_X_c,
                    model.joint_count,
                ],
                outputs=[
                    X_local,
                ],
                device=model.device,
            )

            wp.launch(
                _fk_accum,
                dim=total_joints,
                inputs=[
                    model.joint_parent,
                    X_local,
                    model.joint_count,
                ],
                outputs=[
                    body_q,
                ],
                device=model.device,
            )

        class _Specialised(IKSolver):
            TILE_N_COORDS = wp.constant(C)
            TILE_N_RESIDUALS = wp.constant(R)
            TILE_THREADS = wp.constant(32)

            def _solve_tiled(self, jac, res, lam, dq, pred):
                wp.launch_tiled(
                    _lm_solve_tiled,
                    dim=[self.n_problems],
                    inputs=[jac, res, lam, dq, pred],
                    block_dim=self.TILE_THREADS,
                    device=self.device,
                )

        _Specialised.__name__ = f"IK_{C}x{R}"
        _Specialised._compute_motion_subspace_2d = staticmethod(_compute_motion_subspace_2d)
        _Specialised._fk_two_pass = staticmethod(_fk_two_pass)
        return _Specialised


@wp.kernel
def _eval_fk_articulation_batched(
    n_problems: int,
    articulation_start: wp.array1d(dtype=wp.int32),  # (n_articulations + 1)
    joint_q: wp.array2d(dtype=wp.float32),  # (n_problems, n_coords)
    joint_qd: wp.array2d(dtype=wp.float32),  # (n_problems, n_dofs)
    joint_q_start: wp.array1d(dtype=wp.int32),  # (n_joints + 1)
    joint_qd_start: wp.array1d(dtype=wp.int32),  # (n_joints + 1)
    joint_type: wp.array1d(dtype=wp.int32),  # (n_joints)
    joint_parent: wp.array1d(dtype=wp.int32),  # (n_joints)
    joint_child: wp.array1d(dtype=wp.int32),  # (n_joints)
    joint_X_p: wp.array1d(dtype=wp.transform),  # (n_joints)
    joint_X_c: wp.array1d(dtype=wp.transform),  # (n_joints)
    joint_axis: wp.array1d(dtype=wp.vec3),  # (n_dofs)
    joint_dof_dim: wp.array2d(dtype=int),  # (n_joints, 2)
    body_com: wp.array1d(dtype=wp.vec3),  # (n_bodies)
    # outputs
    body_q: wp.array2d(dtype=wp.transform),  # (n_problems, n_bodies)
    body_qd: wp.array2d(dtype=wp.spatial_vector),  # (n_problems, n_bodies)
):
    tid = wp.tid()

    num_articulations = len(articulation_start) - 1
    problem_idx = tid // num_articulations
    articulation_idx = tid % num_articulations

    if problem_idx >= n_problems:
        return

    joint_start = articulation_start[articulation_idx]
    joint_end = articulation_start[articulation_idx + 1]

    joint_q_slice = joint_q[problem_idx]
    joint_qd_slice = joint_qd[problem_idx]
    body_q_slice = body_q[problem_idx]
    body_qd_slice = body_qd[problem_idx]

    eval_single_articulation_fk(
        joint_start,
        joint_end,
        joint_q_slice,
        joint_qd_slice,
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
        body_q_slice,
        body_qd_slice,
    )


def _eval_fk_batched(model, joint_q, joint_qd, body_q, body_qd):
    """Compute batched forward kinematics."""
    n_problems = joint_q.shape[0]

    wp.launch(
        kernel=_eval_fk_articulation_batched,
        dim=n_problems * model.articulation_count,
        inputs=[
            n_problems,
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
        outputs=[
            body_q,
            body_qd,
        ],
        device=model.device,
    )


@wp.kernel
def _fk_accum(
    joint_parent: wp.array1d(dtype=wp.int32),  # (n_joints)
    X_local: wp.array2d(dtype=wp.transform),  # (n_problems, n_joints)
    joint_count: int,
    # outputs
    body_q: wp.array2d(dtype=wp.transform),  # (n_problems, n_bodies)
):
    global_joint_idx = wp.tid()
    problem_idx = global_joint_idx / joint_count
    local_joint_idx = global_joint_idx % joint_count

    Xw = X_local[problem_idx, local_joint_idx]
    parent = joint_parent[local_joint_idx]

    while parent >= 0:
        Xp = X_local[problem_idx, parent]
        Xw = Xp * Xw
        parent = joint_parent[parent]

    body_q[problem_idx, local_joint_idx] = Xw


@wp.kernel
def _compute_costs(
    residuals: wp.array2d(dtype=wp.float32),  # (n_problems, n_residuals)
    num_residuals: int,
    # outputs
    costs: wp.array1d(dtype=wp.float32),  # (n_problems)
):
    problem_idx = wp.tid()

    cost = float(0.0)
    for i in range(num_residuals):
        r = residuals[problem_idx, i]
        cost += r * r

    costs[problem_idx] = cost


@wp.kernel
def _compute_q_proposed(
    joint_q_current: wp.array2d(dtype=wp.float32),  # (n_problems, n_coords)
    delta_q: wp.array2d(dtype=wp.float32),  # (n_problems, n_coords)
    step_size: float,
    n_coords: int,
    # outputs
    joint_q_proposed: wp.array2d(dtype=wp.float32),  # (n_problems, n_coords)
):
    global_coord_idx = wp.tid()

    problem_idx = global_coord_idx / n_coords
    coord_idx = global_coord_idx % n_coords

    joint_q_proposed[problem_idx, coord_idx] = (
        joint_q_current[problem_idx, coord_idx] + step_size * delta_q[problem_idx, coord_idx]
    )


@wp.kernel
def _accept_reject(
    cost_curr: wp.array1d(dtype=wp.float32),  # (n_problems)
    cost_prop: wp.array1d(dtype=wp.float32),  # (n_problems)
    pred_red: wp.array1d(dtype=wp.float32),  # (n_problems)
    rho_min: float,
    # outputs
    accept: wp.array1d(dtype=wp.int32),  # (n_problems)
):
    problem_idx = wp.tid()
    rho = (cost_curr[problem_idx] - cost_prop[problem_idx]) / (pred_red[problem_idx] + 1e-8)
    accept[problem_idx] = wp.int32(1) if rho >= rho_min else wp.int32(0)


@wp.kernel
def _update_lm_state(
    joint_q_proposed: wp.array2d(dtype=wp.float32),  # (n_problems, n_coords)
    residuals_proposed: wp.array2d(dtype=wp.float32),  # (n_problems, n_residuals)
    costs_proposed: wp.array1d(dtype=wp.float32),  # (n_problems)
    accept_flags: wp.array1d(dtype=wp.int32),  # (n_problems)
    n_coords: int,
    num_residuals: int,
    lambda_factor: float,
    lambda_min: float,
    lambda_max: float,
    # outputs
    joint_q_current: wp.array2d(dtype=wp.float32),  # (n_problems, n_coords)
    residuals_current: wp.array2d(dtype=wp.float32),  # (n_problems, n_residuals)
    costs: wp.array1d(dtype=wp.float32),  # (n_problems)
    lambda_values: wp.array1d(dtype=wp.float32),  # (n_problems)
):
    problem_idx = wp.tid()

    if accept_flags[problem_idx] == 1:
        for i in range(n_coords):
            joint_q_current[problem_idx, i] = joint_q_proposed[problem_idx, i]

        for i in range(num_residuals):
            residuals_current[problem_idx, i] = residuals_proposed[problem_idx, i]

        costs[problem_idx] = costs_proposed[problem_idx]

        lambda_values[problem_idx] = lambda_values[problem_idx] / lambda_factor
    else:
        new_lambda = lambda_values[problem_idx] * lambda_factor
        lambda_values[problem_idx] = wp.clamp(new_lambda, lambda_min, lambda_max)


class IKObjective:
    def residual_dim(self):
        raise NotImplementedError

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx):
        raise NotImplementedError

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, joint_q):
        raise NotImplementedError

    def supports_analytic(self):
        return False

    def bind_device(self, device):
        self.device = device

    def init_buffers(self, model, jacobian_mode):
        pass

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        pass


@wp.kernel
def _pos_residuals(
    body_q: wp.array2d(dtype=wp.transform),  # (n_problems, n_bodies)
    target_pos: wp.array1d(dtype=wp.vec3),  # (n_problems)
    link_index: int,
    link_offset: wp.vec3,
    start_idx: int,
    weight: float,
    # outputs
    residuals: wp.array2d(dtype=wp.float32),  # (n_problems, n_residuals)
):
    problem_idx = wp.tid()

    body_tf = body_q[problem_idx, link_index]
    ee_pos = wp.transform_point(body_tf, link_offset)

    error = target_pos[problem_idx] - ee_pos
    residuals[problem_idx, start_idx + 0] = weight * error[0]
    residuals[problem_idx, start_idx + 1] = weight * error[1]
    residuals[problem_idx, start_idx + 2] = weight * error[2]


@wp.kernel
def _pos_jac_fill(
    q_grad: wp.array2d(dtype=wp.float32),  # (n_problems, n_coords)
    n_coords: int,
    start_idx: int,
    component: int,
    # outputs
    jacobian: wp.array3d(dtype=wp.float32),  # (n_problems, n_residuals, n_coords)
):
    problem_idx = wp.tid()

    residual_idx = start_idx + component

    for j in range(n_coords):
        jacobian[problem_idx, residual_idx, j] = q_grad[problem_idx, j]


@wp.kernel
def _update_position_target(
    problem_idx: int,
    new_position: wp.vec3,
    # outputs
    target_array: wp.array1d(dtype=wp.vec3),  # (n_problems)
):
    target_array[problem_idx] = new_position


@wp.kernel
def _update_position_targets(
    new_positions: wp.array1d(dtype=wp.vec3),  # (n_problems)
    # outputs
    target_array: wp.array1d(dtype=wp.vec3),  # (n_problems)
):
    problem_idx = wp.tid()
    target_array[problem_idx] = new_positions[problem_idx]


@wp.kernel
def _pos_jac_analytic(
    link_index: int,
    link_offset: wp.vec3,
    joint_child: wp.array1d(dtype=wp.int32),  # (n_joints)
    coord_to_joint: wp.array1d(dtype=wp.int32),  # (n_coords)
    affects_coord: wp.array1d(dtype=wp.uint8),  # (n_coords)
    joint_qd_start: wp.array1d(dtype=wp.int32),  # (n_joints + 1)
    joint_q_start: wp.array1d(dtype=wp.int32),  # (n_joints + 1)
    joint_type: wp.array1d(dtype=wp.int32),  # (n_joints)
    joint_S_s: wp.array2d(dtype=wp.spatial_vector),  # (n_problems, n_dofs)
    body_q: wp.array2d(dtype=wp.transform),  # (n_problems, n_bodies)
    start_idx: int,
    n_coords: int,
    weight: float,
    # outputs
    jacobian: wp.array3d(dtype=wp.float32),  # (n_problems, n_residuals, n_coords)
):
    global_coord = wp.tid()

    problem_idx = global_coord // n_coords
    coord_idx = global_coord % n_coords

    if affects_coord[coord_idx] == 0:
        return

    joint_idx = coord_to_joint[coord_idx]

    joint_coord_start = joint_q_start[joint_idx]
    joint_coord_end = joint_q_start[joint_idx + 1]
    local_coord = coord_idx - joint_coord_start

    body_tf = body_q[problem_idx, link_index]

    rot_w = wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])
    pos_w = wp.vec3(body_tf[0], body_tf[1], body_tf[2])
    ee_pos_world = pos_w + wp.quat_rotate(rot_w, link_offset)

    col = coord_idx

    # General twist-based Jacobian (supports revolute/prismatic/fixed only)
    dof_start = joint_qd_start[joint_idx]
    dof_local = local_coord
    dof = dof_start + dof_local

    if dof >= dof_start and dof < dof_start + (joint_coord_end - joint_coord_start):
        S = joint_S_s[problem_idx, dof]
        omega = wp.vec3(S[0], S[1], S[2])
        v_orig = wp.vec3(S[3], S[4], S[5])

    v_ee = v_orig + wp.cross(omega, ee_pos_world)

    jacobian[problem_idx, start_idx + 0, col] = -weight * v_ee[0]
    jacobian[problem_idx, start_idx + 1, col] = -weight * v_ee[1]
    jacobian[problem_idx, start_idx + 2, col] = -weight * v_ee[2]


class PositionObjective(IKObjective):
    """
    End-effector positional target for one link.

    Parameters
    ----------
    link_index : int
        Body index whose frame defines the end-effector.
    link_offset : wp.vec3
        Offset from the body frame (local coordinates).
    target_positions : wp.array(dtype=wp.vec3)
        One target position per problem.
    n_problems : int
        Number of parallel IK problems.
    total_residuals : int
        Global residual vector length (for autodiff bookkeeping).
    residual_offset : int
        Starting index of this objective inside the residual vector.
    weight : float, default 1.0
        Scalar weight multiplying both residual and Jacobian rows.
    """

    def __init__(
        self, link_index, link_offset, target_positions, n_problems, total_residuals, residual_offset, weight=1.0
    ):
        self.link_index = link_index
        self.link_offset = link_offset
        self.target_positions = target_positions
        self.n_problems = n_problems
        self.weight = weight
        self.total_residuals = total_residuals
        self.residual_offset = residual_offset

        self.body_to_joint = None
        self.coord_to_joint = None
        self.affects_coord = None

    def init_buffers(self, model, jacobian_mode):
        """Precompute lookup tables for analytic jacobian computation."""
        if jacobian_mode == JacobianMode.ANALYTIC:
            links_per_problem = model.body_count
            n_coords = model.joint_coord_count

            joint_child_np = model.joint_child.numpy()
            body_to_joint_np = np.full(links_per_problem, -1, np.int32)

            for j in range(model.joint_count):
                child = joint_child_np[j]
                if child != -1:
                    body_to_joint_np[child] = j

            self.body_to_joint = wp.array(body_to_joint_np, dtype=wp.int32, device=self.device)

            coord_to_joint_np = np.empty(n_coords, dtype=np.int32)
            joint_q_start_np = model.joint_q_start.numpy()

            for j in range(len(joint_q_start_np) - 1):
                start, end = joint_q_start_np[j : j + 2]
                coord_to_joint_np[start:end] = j

            self.coord_to_joint = wp.array(coord_to_joint_np, dtype=wp.int32, device=self.device)

            ancestors = np.zeros(len(joint_q_start_np) - 1, dtype=bool)
            joint_parent_np = model.joint_parent.numpy()
            body = self.link_index
            while body != -1:
                j = body_to_joint_np[body]
                if j != -1:
                    ancestors[j] = True
                body = joint_parent_np[j] if j != -1 else -1

            affects_coord_np = ancestors[coord_to_joint_np]
            self.affects_coord = wp.array(affects_coord_np.astype(np.uint8), device=self.device)
        elif jacobian_mode == JacobianMode.AUTODIFF:
            self.e_arrays = []
            for component in range(3):
                e = np.zeros((self.n_problems, self.total_residuals), dtype=np.float32)
                for prob_idx in range(self.n_problems):
                    e[prob_idx, self.residual_offset + component] = 1.0
                self.e_arrays.append(wp.array(e.flatten(), dtype=wp.float32, device=self.device))

    def supports_analytic(self):
        return True

    def set_target_position(self, problem_idx, new_position):
        wp.launch(
            _update_position_target,
            dim=1,
            inputs=[problem_idx, new_position],
            outputs=[self.target_positions],
            device=self.device,
        )

    def set_target_positions(self, new_positions):
        wp.launch(
            _update_position_targets,
            dim=self.n_problems,
            inputs=[new_positions],
            outputs=[self.target_positions],
            device=self.device,
        )

    def residual_dim(self):
        return 3

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx):
        wp.launch(
            _pos_residuals,
            dim=self.n_problems,
            inputs=[
                body_q,
                self.target_positions,
                self.link_index,
                self.link_offset,
                start_idx,
                self.weight,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, joint_q):
        for component in range(3):
            tape.backward(grads={tape.outputs[0]: self.e_arrays[component].flatten()})

            q_grad = tape.gradients[joint_q]

            n_coords = model.joint_coord_count

            wp.launch(
                _pos_jac_fill,
                dim=self.n_problems,
                inputs=[
                    q_grad,
                    n_coords,
                    start_idx,
                    component,
                ],
                outputs=[
                    jacobian,
                ],
                device=self.device,
            )

            tape.zero()

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        n_coords = model.joint_coord_count

        wp.launch(
            _pos_jac_analytic,
            dim=self.n_problems * n_coords,
            inputs=[
                self.link_index,
                self.link_offset,
                model.joint_child,
                self.coord_to_joint,
                self.affects_coord,
                model.joint_qd_start,
                model.joint_q_start,
                model.joint_type,
                joint_S_s,
                body_q,
                start_idx,
                n_coords,
                self.weight,
            ],
            outputs=[
                jacobian,
            ],
            device=self.device,
        )


@wp.kernel
def _limit_residuals(
    joint_q: wp.array2d(dtype=wp.float32),  # (n_problems, n_coords)
    joint_limit_lower: wp.array1d(dtype=wp.float32),  # (n_coords)
    joint_limit_upper: wp.array1d(dtype=wp.float32),  # (n_coords)
    n_coords: int,
    weight: float,
    start_idx: int,
    # outputs
    residuals: wp.array2d(dtype=wp.float32),  # (n_problems, n_residuals)
):
    global_coord_idx = wp.tid()
    problem_idx = global_coord_idx / n_coords
    coord_idx = global_coord_idx % n_coords

    q = joint_q[problem_idx, coord_idx]
    lower = joint_limit_lower[coord_idx]
    upper = joint_limit_upper[coord_idx]

    upper_violation = wp.max(0.0, q - upper)
    lower_violation = wp.max(0.0, lower - q)

    residuals[problem_idx, start_idx + coord_idx] = weight * (upper_violation + lower_violation)


@wp.kernel
def _limit_jac_fill(
    q_grad: wp.array2d(dtype=wp.float32),  # (n_problems, n_coords)
    n_coords: int,
    start_idx: int,
    # outputs
    jacobian: wp.array3d(dtype=wp.float32),  # (n_problems, n_residuals, n_coords)
):
    global_coord_idx = wp.tid()
    problem_idx = global_coord_idx / n_coords
    coord_idx = global_coord_idx % n_coords

    residual_idx = start_idx + coord_idx

    jacobian[problem_idx, residual_idx, coord_idx] = q_grad[problem_idx, coord_idx]


@wp.kernel
def _limit_jac_analytic(
    joint_q: wp.array2d(dtype=wp.float32),  # (n_problems, n_coords)
    joint_limit_lower: wp.array1d(dtype=wp.float32),  # (n_coords)
    joint_limit_upper: wp.array1d(dtype=wp.float32),  # (n_coords)
    n_coords: int,
    start_idx: int,
    weight: float,
    # outputs
    jacobian: wp.array3d(dtype=wp.float32),  # (n_problems, n_residuals, n_coords)
):
    global_coord_idx = wp.tid()
    problem_idx = global_coord_idx / n_coords
    coord_idx = global_coord_idx % n_coords

    q = joint_q[problem_idx, coord_idx]
    lower = joint_limit_lower[coord_idx]
    upper = joint_limit_upper[coord_idx]

    grad = float(0.0)
    if q >= upper:
        grad = weight
    elif q <= lower:
        grad = -weight

    residual_idx = start_idx + coord_idx
    jacobian[problem_idx, residual_idx, coord_idx] = grad


class JointLimitObjective(IKObjective):
    """
    Joint limit constraint objective.

    Parameters
    ----------
    joint_limit_lower : wp.array(dtype=float)
        Lower bounds for each joint coordinate.
    joint_limit_upper : wp.array(dtype=float)
        Upper bounds for each joint coordinate.
    n_problems : int, optional
        Number of parallel IK problems.
    total_residuals : int, optional
        Global residual vector length (for autodiff bookkeeping).
    residual_offset : int, optional
        Starting index of this objective inside the residual vector.
    weight : float, default 0.1
        Scalar weight for limit violation penalty.
    """

    def __init__(
        self,
        joint_limit_lower,
        joint_limit_upper,
        n_problems,
        total_residuals,
        residual_offset,
        weight=0.1,
    ):
        self.joint_limit_lower = joint_limit_lower
        self.joint_limit_upper = joint_limit_upper
        self.e_array = None
        self.n_problems = n_problems
        self.total_residuals = total_residuals
        self.residual_offset = residual_offset
        self.weight = weight

        self.n_coords = len(joint_limit_lower)

    def init_buffers(self, model, jacobian_mode):
        if jacobian_mode == JacobianMode.AUTODIFF:
            e = np.zeros((self.n_problems, self.total_residuals), dtype=np.float32)
            for prob_idx in range(self.n_problems):
                for coord_idx in range(self.n_coords):
                    e[prob_idx, self.residual_offset + coord_idx] = 1.0
            self.e_array = wp.array(e.flatten(), dtype=wp.float32, device=self.device)

    def supports_analytic(self):
        return True

    def residual_dim(self):
        return self.n_coords

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx):
        wp.launch(
            _limit_residuals,
            dim=self.n_problems * self.n_coords,
            inputs=[
                joint_q,
                self.joint_limit_lower,
                self.joint_limit_upper,
                self.n_coords,
                self.weight,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, joint_q):
        tape.backward(grads={tape.outputs[0]: self.e_array})

        q_grad = tape.gradients[joint_q]

        wp.launch(
            _limit_jac_fill,
            dim=self.n_problems * self.n_coords,
            inputs=[
                q_grad,
                self.n_coords,
                start_idx,
            ],
            outputs=[jacobian],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        wp.launch(
            _limit_jac_analytic,
            dim=self.n_problems * self.n_coords,
            inputs=[
                joint_q,
                self.joint_limit_lower,
                self.joint_limit_upper,
                self.n_coords,
                start_idx,
                self.weight,
            ],
            outputs=[jacobian],
            device=self.device,
        )


@wp.kernel
def _rot_residuals(
    body_q: wp.array2d(dtype=wp.transform),  # (n_problems, n_bodies)
    target_rot: wp.array1d(dtype=wp.vec4),  # (n_problems)
    link_index: int,
    link_offset_rotation: wp.quat,
    start_idx: int,
    weight: float,
    # outputs
    residuals: wp.array2d(dtype=wp.float32),  # (n_problems, n_residuals)
):
    problem_idx = wp.tid()

    body_tf = body_q[problem_idx, link_index]
    body_rot = wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])

    actual_rot = body_rot * link_offset_rotation

    target_quat_vec = target_rot[problem_idx]
    target_quat = wp.quat(target_quat_vec[0], target_quat_vec[1], target_quat_vec[2], target_quat_vec[3])

    q_err = actual_rot * wp.quat_inverse(target_quat)

    v_norm = wp.sqrt(q_err[0] * q_err[0] + q_err[1] * q_err[1] + q_err[2] * q_err[2])

    angle = 2.0 * wp.atan2(v_norm, q_err[3])

    eps = float(1e-8)
    axis_angle = wp.vec3(0.0, 0.0, 0.0)

    if v_norm > eps:
        axis = wp.vec3(q_err[0] / v_norm, q_err[1] / v_norm, q_err[2] / v_norm)
        axis_angle = axis * angle
    else:
        axis_angle = wp.vec3(2.0 * q_err[0], 2.0 * q_err[1], 2.0 * q_err[2])

    residuals[problem_idx, start_idx + 0] = weight * axis_angle[0]
    residuals[problem_idx, start_idx + 1] = weight * axis_angle[1]
    residuals[problem_idx, start_idx + 2] = weight * axis_angle[2]


@wp.kernel
def _rot_jac_fill(
    q_grad: wp.array2d(dtype=wp.float32),  # (n_problems, n_coords)
    n_coords: int,
    start_idx: int,
    component: int,
    # outputs
    jacobian: wp.array3d(dtype=wp.float32),  # (n_problems, n_residuals, n_coords)
):
    problem_idx = wp.tid()

    residual_idx = start_idx + component

    for j in range(n_coords):
        jacobian[problem_idx, residual_idx, j] = q_grad[problem_idx, j]


@wp.kernel
def _update_rotation_target(
    problem_idx: int,
    new_rotation: wp.vec4,
    # outputs
    target_array: wp.array1d(dtype=wp.vec4),  # (n_problems)
):
    target_array[problem_idx] = new_rotation


@wp.kernel
def _update_rotation_targets(
    new_rotation: wp.array1d(dtype=wp.vec4),  # (n_problems)
    # outputs
    target_array: wp.array1d(dtype=wp.vec4),  # (n_problems)
):
    problem_idx = wp.tid()
    target_array[problem_idx] = new_rotation[problem_idx]


@wp.kernel
def _rot_jac_analytic(
    coord_to_joint: wp.array1d(dtype=wp.int32),  # (n_coords)
    affects_coord: wp.array1d(dtype=wp.uint8),  # (n_coords)
    joint_qd_start: wp.array1d(dtype=wp.int32),  # (n_joints + 1)
    joint_q_start: wp.array1d(dtype=wp.int32),  # (n_joints + 1)
    joint_S_s: wp.array2d(dtype=wp.spatial_vector),  # (n_problems, n_dofs)
    start_idx: int,
    n_coords: int,
    weight: float,
    # outputs
    jacobian: wp.array3d(dtype=wp.float32),  # (n_problems, n_residuals, n_coords)
):
    global_coord = wp.tid()
    problem_idx = global_coord // n_coords
    coord_idx = global_coord % n_coords

    if affects_coord[coord_idx] == 0:
        return

    joint_idx = coord_to_joint[coord_idx]
    if joint_idx < 0:
        return

    joint_coord_start = joint_q_start[joint_idx]

    # General omega-based Jacobian (supports revolute/prismatic/fixed only)
    dof_start = joint_qd_start[joint_idx]

    local_coord = coord_idx - joint_coord_start
    dof = dof_start + local_coord

    S = joint_S_s[problem_idx, dof]
    omega = wp.vec3(S[0], S[1], S[2])

    col = coord_idx
    jacobian[problem_idx, start_idx + 0, col] = weight * omega[0]
    jacobian[problem_idx, start_idx + 1, col] = weight * omega[1]
    jacobian[problem_idx, start_idx + 2, col] = weight * omega[2]


class RotationObjective(IKObjective):
    """
    End-effector rotational target for one link.

    Parameters
    ----------
    link_index : int
        Body index whose frame defines the end-effector.
    link_offset_rotation : wp.quat
        Rotation offset from the body frame (local coordinates).
    target_rotations : wp.array(dtype=wp.vec4)
        One target quaternion per problem (stored as vec4).
    n_problems : int
        Number of parallel IK problems.
    total_residuals : int
        Global residual vector length (for autodiff bookkeeping).
    residual_offset : int
        Starting index of this objective inside the residual vector.
    weight : float, default 1.0
        Scalar weight multiplying both residual and Jacobian rows.
    """

    def __init__(
        self,
        link_index,
        link_offset_rotation,
        target_rotations,
        n_problems,
        total_residuals,
        residual_offset,
        weight=1.0,
    ):
        self.link_index = link_index
        self.link_offset_rotation = link_offset_rotation
        self.target_rotations = target_rotations
        self.n_problems = n_problems
        self.weight = weight
        self.total_residuals = total_residuals
        self.residual_offset = residual_offset

        self.body_to_joint = None
        self.coord_to_joint = None
        self.affects_coord = None

    def init_buffers(self, model, jacobian_mode):
        """Precompute lookup tables for analytic jacobian computation."""
        if jacobian_mode == JacobianMode.ANALYTIC:
            links_per_problem = model.body_count
            n_coords = model.joint_coord_count

            joint_child_np = model.joint_child.numpy()
            body_to_joint_np = np.full(links_per_problem, -1, np.int32)

            for j in range(model.joint_count):
                child = joint_child_np[j]
                if child != -1:
                    body_to_joint_np[child] = j

            self.body_to_joint = wp.array(body_to_joint_np, dtype=wp.int32, device=self.device)

            coord_to_joint_np = np.empty(n_coords, dtype=np.int32)
            joint_q_start_np = model.joint_q_start.numpy()

            for j in range(len(joint_q_start_np) - 1):
                start, end = joint_q_start_np[j : j + 2]
                coord_to_joint_np[start:end] = j

            self.coord_to_joint = wp.array(coord_to_joint_np, dtype=wp.int32, device=self.device)

            ancestors = np.zeros(len(joint_q_start_np) - 1, dtype=bool)
            joint_parent_np = model.joint_parent.numpy()
            body = self.link_index
            while body != -1:
                j = body_to_joint_np[body]
                if j != -1:
                    ancestors[j] = True
                body = joint_parent_np[j] if j != -1 else -1

            affects_coord_np = ancestors[coord_to_joint_np]
            self.affects_coord = wp.array(affects_coord_np.astype(np.uint8), device=self.device)
        elif jacobian_mode == JacobianMode.AUTODIFF:
            self.e_arrays = []
            for component in range(3):
                e = np.zeros((self.n_problems, self.total_residuals), dtype=np.float32)
                for prob_idx in range(self.n_problems):
                    e[prob_idx, self.residual_offset + component] = 1.0
                self.e_arrays.append(wp.array(e.flatten(), dtype=wp.float32, device=self.device))

    def supports_analytic(self):
        return True

    def set_target_rotation(self, problem_idx, new_rotation):
        wp.launch(
            _update_rotation_target,
            dim=1,
            inputs=[problem_idx, new_rotation],
            outputs=[self.target_rotations],
            device=self.device,
        )

    def set_target_rotations(self, new_rotations):
        wp.launch(
            _update_rotation_targets,
            dim=self.n_problems,
            inputs=[new_rotations],
            outputs=[self.target_rotations],
            device=self.device,
        )

    def residual_dim(self):
        return 3

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx):
        wp.launch(
            _rot_residuals,
            dim=self.n_problems,
            inputs=[
                body_q,
                self.target_rotations,
                self.link_index,
                self.link_offset_rotation,
                start_idx,
                self.weight,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, joint_q):
        for component in range(3):
            tape.backward(grads={tape.outputs[0]: self.e_arrays[component].flatten()})

            q_grad = tape.gradients[joint_q]

            n_coords = model.joint_coord_count

            wp.launch(
                _rot_jac_fill,
                dim=self.n_problems,
                inputs=[
                    q_grad,
                    n_coords,
                    start_idx,
                    component,
                ],
                outputs=[
                    jacobian,
                ],
                device=self.device,
            )

            tape.zero()

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        n_coords = model.joint_coord_count

        wp.launch(
            _rot_jac_analytic,
            dim=self.n_problems * n_coords,
            inputs=[
                self.coord_to_joint,
                self.affects_coord,
                model.joint_qd_start,
                model.joint_q_start,
                joint_S_s,
                start_idx,
                n_coords,
                self.weight,
            ],
            outputs=[
                jacobian,
            ],
            device=self.device,
        )
