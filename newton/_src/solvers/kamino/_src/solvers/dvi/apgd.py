# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""APGD unilateral phases using Kamino's existing dense or sparse operator."""

import warp as wp

from . import apgd_kernels as kernels


class UnilateralAPGD:
    """Solve frozen cone QPs inside a De Saxce fixed-point iteration.

    Each correction uses ``v = A x + b`` at the current impulse ``x`` and
    holds ``s = (0, 0, mu * norm(v_t))`` fixed throughout an APGD solve.
    Inner steps evaluate ``A y + b + s`` at the extrapolated impulse ``y``.
    A fresh nonlinear residual decides whether another correction is needed.

    The bilateral block is fixed during an alternating phase. In Schur mode
    each product includes its eliminated response, using the already factored
    bilateral operator. No dense unilateral matrix or contact adjacency is
    constructed for the sparse path.
    """

    def __init__(self, owner):
        """Allocate all iteration and response storage before graph capture.

        Args:
            owner: DVI solver supplying per-world configuration, operator
                storage, impulses, and terminal status.
        """
        self.owner = owner
        self.device = owner.device
        if self.device.is_cuda and not wp.is_conditional_graph_supported():
            raise RuntimeError(
                "CUDA APGD requires conditional graphs: a Warp build and CUDA driver supporting CUDA 12.4+."
            )
        size = owner.size
        self.rows = (size.num_worlds, max(1, size.max_of_max_total_cts))
        self.bilateral_rows = (size.num_worlds, max(1, size.max_of_num_bilateral_joint_cts))
        configs = [c.apgd for c in owner.config]
        entries = []
        for config in configs:
            entry = kernels.APGDConfig()
            entry.max_iterations = config.max_iterations
            entry.max_backtracks = config.max_backtracks
            entry.max_nonlinear_corrections = config.max_nonlinear_corrections
            entry.tolerance = config.tolerance
            entry.relaxation = config.relaxation
            entries.append(entry)
        self.config = wp.array(entries, dtype=kernels.APGDConfig, device=self.device)
        self.state = wp.zeros(size.num_worlds, dtype=kernels.APGDState, device=self.device)
        self.phase = wp.zeros(size.num_worlds, dtype=wp.bool, device=self.device)
        self.active = wp.zeros_like(self.phase)
        self.inner = wp.zeros_like(self.phase)
        self.searching = wp.zeros_like(self.phase)
        self.estimating = wp.zeros_like(self.phase)
        self.operator_scale = wp.ones(size.num_worlds, device=self.device)
        self.correction_condition = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.inner_condition = wp.zeros_like(self.correction_condition)
        self.search_condition = wp.zeros_like(self.correction_condition)
        self.scale_condition = wp.zeros_like(self.correction_condition)
        self.x = wp.zeros(max(1, size.sum_of_max_total_cts), device=self.device)
        self.probe = wp.zeros_like(self.x)
        self.y = wp.zeros_like(self.x)
        self.candidate = wp.zeros_like(self.x)
        self.previous = wp.zeros_like(self.x)
        self.product = wp.zeros_like(self.x)
        self.product_y = wp.zeros_like(self.x)
        self.product_candidate = wp.zeros_like(self.x)
        self.full_product = wp.zeros_like(self.x)
        self.bias = wp.zeros_like(self.x)
        self.shift = wp.zeros_like(self.x)
        self.zero = wp.zeros_like(self.x)
        self.response = wp.zeros_like(self.x)
        self.sparse_product = wp.zeros_like(self.x)
        bilateral = getattr(owner.data, "bilateral_operator", None)
        bilateral_size = bilateral.info.total_vec_size if bilateral is not None else 1
        self.response_rhs = wp.zeros(max(1, bilateral_size), device=self.device)
        self.response_solution = wp.zeros_like(self.response_rhs)
        self.response_dim = wp.zeros(size.num_worlds, dtype=wp.int32, device=self.device)
        self.problem = None

    def _launch(self, kernel, args, *, rows=False):
        """Launch a row operation or a deterministic per-world reduction."""
        wp.launch(kernel, dim=self.rows if rows else self.owner.size.num_worlds, inputs=args, device=self.device)

    def full_matvec(self, x, y, mask):
        """Apply the full Delassus operator to the selected worlds.

        Args:
            x: Input vector in the full constraint layout.
            y: Output product; entries in unselected worlds are preserved.
            mask: Per-world flag selecting operator products.
        """
        problem = self.problem
        data = problem.data
        if problem.sparse:
            problem.delassus.matvec(x, self.sparse_product, mask)
            self._launch(kernels.copy_active, [data.dim, data.vio, mask, self.sparse_product, y], rows=True)
        else:
            self._launch(kernels.dense_matvec, [data.dim, data.mio, data.vio, data.D, mask, x, y], rows=True)

    def matvec(self, x, y, mask):
        """Apply the unilateral operator, including the optional Schur response.

        Unilateral entries of the output contain ``D_uu x`` or
        ``(D_uu - D_ub D_bb^-1 D_bu) x``. The Schur path reuses the owner's
        factored bilateral block without assembling a reduced matrix.

        Args:
            x: Input vector with zero bilateral entries.
            y: Output product; only unilateral entries are used by APGD.
            mask: Per-world flag selecting operator products.
        """
        self.full_matvec(x, y, mask)
        owner = self.owner
        if not owner._use_schur_complement or owner._bilateral_solver is None:
            return
        data = self.problem.data
        operator = owner.data.bilateral_operator
        scale = owner.data.state.bilateral_preconditioner
        wp.launch(
            kernels.build_response_rhs,
            dim=self.bilateral_rows,
            inputs=[data.vio, data.njc, operator.info.vio, scale, y, mask, self.response_rhs, self.response_dim],
            device=self.device,
        )
        full_dim = operator.info.dim
        operator.info.dim = self.response_dim
        try:
            owner._bilateral_solver.solve(b=self.response_rhs, x=self.response_solution)
        finally:
            operator.info.dim = full_dim
        self._launch(
            kernels.assemble_response,
            [data.dim, data.vio, data.njc, operator.info.vio, scale, self.response_solution, x, self.response],
            rows=True,
        )
        self.full_matvec(self.response, y, mask)

    def solve(self, problem, *, block_iteration=-1):
        """Update unilateral impulses through bounded frozen-correction solves.

        Each correction restarts acceleration, solves its QP to the inner
        tolerance or iteration limit, and checks the updated nonlinear map.
        Operator scale is estimated on the device at the first phase and
        reused through alternating blocks. Later QPs can reduce a learned
        denominator toward that estimate when restarting acceleration.
        The default correction budget of one may leave a nonzero residual.
        Impulses and diagnostics are written to the owner's existing arrays.

        Args:
            problem: Dual problem supplying the operator and constraint data.
            block_iteration: Current bilateral/unilateral alternation index.
                A negative value bypasses per-world alternation limits.
        """
        self.problem = problem
        data = problem.data
        owner = self.owner
        status = owner.data.status
        solution = owner.data.solution.lambdas
        layout = [data.dim, data.vio, data.njc]
        projection = [*layout, data.nbc, data.nl, data.bcio, data.cio, data.mu, data.bound_lower, data.bound_upper]
        self.correction_condition.zero_()
        self._launch(
            kernels.initialize_phase,
            [
                data.dim,
                data.njc,
                owner.data.config,
                block_iteration,
                self.operator_scale,
                self.estimating,
                self.phase,
                self.active,
                self.state,
                self.correction_condition,
            ],
        )
        if block_iteration <= 0:

            def estimate_scale(retry=False):
                """Measure the effective operator, including its factored Schur response."""
                self._launch(kernels.seed_scale_probe, [*layout, self.estimating, retry, self.probe], rows=True)
                self.matvec(self.probe, self.product, self.estimating)
                self.scale_condition.zero_()
                self._launch(
                    kernels.finish_scale_probe,
                    [
                        *layout,
                        self.probe,
                        self.product,
                        retry,
                        self.operator_scale,
                        self.state,
                        self.estimating,
                        self.scale_condition,
                    ],
                )

            estimate_scale()
            wp.capture_if(self.scale_condition, on_true=estimate_scale, retry=True)
        # Construct q from the actual iterate before projecting a potentially
        # infeasible warmstart. For Schur, the caller first refreshes B.
        self.x.zero_()
        self.y.zero_()
        self.candidate.zero_()
        self._launch(kernels.copy_unilateral, [*layout, solution, self.x], rows=True)
        self.full_matvec(solution, self.full_product, self.phase)
        self.matvec(self.x, self.product, self.phase)
        self._launch(
            kernels.build_phase_bias,
            [*layout, self.full_product, data.v_f, self.product, self.bias],
            rows=True,
        )
        self._launch(
            kernels.projected_step,
            [*projection, self.phase, self.state, self.x, self.zero, self.zero, self.zero, self.x],
            rows=True,
        )

        def search_body():
            """Project a trial and check the fixed quadratic majorizer."""
            self._launch(
                kernels.projected_step,
                [
                    *projection,
                    self.searching,
                    self.state,
                    self.y,
                    self.product_y,
                    self.bias,
                    self.shift,
                    self.candidate,
                ],
                rows=True,
            )
            self.matvec(self.candidate, self.product_candidate, self.searching)
            self.search_condition.zero_()
            self._launch(
                kernels.check_descent,
                [
                    *layout,
                    self.config,
                    self.y,
                    self.product_y,
                    self.candidate,
                    self.product_candidate,
                    self.state,
                    self.searching,
                    self.inner,
                    self.active,
                    self.search_condition,
                    status,
                ],
            )

        def inner_body():
            """Advance one accelerated projected step, with bounded backtracking."""
            self.matvec(self.y, self.product_y, self.inner)
            self.search_condition.zero_()
            self._launch(kernels.begin_iteration, [self.inner, self.state, self.searching, self.search_condition])
            wp.capture_while(self.search_condition, while_body=search_body)
            self.inner_condition.zero_()
            self._launch(
                kernels.accept_iteration,
                [
                    *projection,
                    self.config,
                    self.product_candidate,
                    self.bias,
                    self.shift,
                    self.candidate,
                    self.x,
                    self.y,
                    self.state,
                    self.inner,
                    self.inner_condition,
                    status,
                ],
            )

        def correction_body():
            """Freeze the correction, solve its QP, and check the nonlinear contact law."""
            wp.copy(self.previous, self.x)
            self.matvec(self.x, self.product, self.active)
            self.inner_condition.zero_()
            self._launch(
                kernels.begin_correction,
                [self.active, self.operator_scale, self.state, self.inner, self.inner_condition],
            )
            self._launch(
                kernels.freeze_correction,
                [
                    *layout,
                    data.ccgo,
                    data.cio,
                    data.mu,
                    self.active,
                    self.product,
                    self.bias,
                    self.x,
                    self.shift,
                    self.y,
                ],
                rows=True,
            )
            wp.capture_while(self.inner_condition, while_body=inner_body)
            self._launch(
                kernels.relax_correction, [*layout, self.active, self.config, self.previous, self.x], rows=True
            )
            self.matvec(self.x, self.product, self.active)
            self.correction_condition.zero_()
            self._launch(
                kernels.finish_correction,
                [
                    *projection,
                    self.config,
                    self.x,
                    self.product,
                    self.bias,
                    self.shift,
                    self.state,
                    self.active,
                    self.correction_condition,
                    status,
                ],
            )

        wp.capture_while(self.correction_condition, while_body=correction_body)
        self._launch(kernels.scatter_solution, [*layout, self.phase, self.x, solution], rows=True)
        self._launch(kernels.finish_phase, [self.phase, self.state, status])
