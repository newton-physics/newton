# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""APGD unilateral phases using Kamino's existing dense or sparse operator.

**Velocity and contact correction.** During a unilateral phase, let ``x`` be
the constraint impulse, ``A`` the unilateral Delassus operator (or its Schur
complement), and ``b`` the fixed velocity bias. Every velocity evaluation uses
the same relation ``v(x) = A x + b``. Contact rows are ordered ``[t0, t1, n]``;
the De Saxce shift is ``s(v) = (0, 0, mu * norm(v_t))`` for each contact and
zero for bounds and limits. Let ``K`` denote the product of the bound
intervals, limit half-lines, and Coulomb cones.

**Nested algorithm.** Each nonlinear correction freezes ``s`` and solves
the convex QP ``min_{x in K} 0.5 * x.T A x + (b + s).T x`` using APGD:

.. code-block:: text

   L_seed = estimate_scale(A)             # Once per DVI solve, on the device
   L = L_seed                            # Retain L across alternating phases
   x = project_K(initial_impulse)
   for each nonlinear correction (at most max_nonlinear_corrections):
       if this is a later frozen QP in the same DVI solve:
           L = max(L_seed, 0.9 * L)
       x_outer = x
       s = correction(A x_outer + b)
       y = x; t = 1                      # Restart acceleration
       for each APGD iteration (at most max_iterations):
           g = A y + b + s
           backtrack for at most max_backtracks trials:
               z = project_K(y - g / L); d = z - y
               abort search on a non-finite trial or denominator
               accept if d.T A d <= L * norm(d)**2 + roundoff
               otherwise double L and retry, keeping y and s fixed
           if the search fails: return x with a failure flag
           t_next = (1 + sqrt(1 + 4 * t**2)) / 2
           beta = (t - 1) / t_next
           if dot(y - z, z - x) > 0: t_next = 1; beta = 0
           y = z + beta * (z - x); x = z; t = t_next
           if norm_inf(x - project_K(x - (A x + b + s))) <= tolerance:
               break
       x = x_outer + relaxation * (x - x_outer)
       s_fresh = correction(A x + b)
       if norm_inf(x - project_K(x - (A x + b + s_fresh))) <= tolerance:
           break
   return x and the fresh nonlinear residual

APGD updates the velocity ``A y + b`` at every extrapolated iterate ``y``;
only the correction remains fixed during the inner solve. A new outer
iteration refreshes that correction from the updated impulse. This makes
the correction consistent with the resulting contact velocity while keeping
one convex objective throughout each inner solve. It does not guarantee
convergence of the outer fixed point for every frictional contact problem.

The inner residual is ``norm_inf(x - project_K(x - (A x + b + s)))``.
The nonlinear residual uses the same expression with ``s`` recomputed from
``A x + b``. Both use a unit projection step. Inner convergence alone does
not establish nonlinear Coulomb convergence.

**Step size.** At the first unilateral phase of each DVI solve, device
kernels initialize ``L = norm(A d) / norm(d)`` using a normalized constant
probe over unilateral rows. The product uses the effective operator, including
the factored bilateral response in Schur mode. A null or non-finite probe
triggers one centered-ramp probe for the affected worlds. If neither gives a
positive finite estimate, ``L`` starts at one and backtracking validates the
trial steps. The estimate is directional, not a certified spectral bound.

The operator estimate is recomputed from current device data on each graph
replay and reused across alternating blocks within a solve. Backtracking
can only increase ``L`` inside a frozen QP. At the next QP, where acceleration
restarts, ``L`` is reduced to ``max(initial_estimate, 0.9 * L)``. This lets
an oversized learned denominator recover without changing the momentum
recurrence inside the QP. No decrease is applied between inner APGD iterations.
With one correction and one unilateral phase, ``L`` never decreases during
the solve. Additional corrections or alternating phases enable the reduction.

For backtracking, ``d = candidate - y``. The numerical allowance is
``1e-6 * max(abs(d.T A d), L * norm(d)**2) + 1e-20``; it is not a solver
convergence tolerance. A rejected finite trial doubles ``L`` and retries.

**Correction accuracy.** The default performs one frozen-correction
approximation. Workloads requiring tighter Coulomb accuracy can set
``max_nonlinear_corrections`` to a larger value, such as 20, before constructing
the solver. More inner APGD iterations
cannot remove error caused by a stale correction. For example, with
``A = I``, ``b = (10, 0, -1)``, ``mu = 0.5``, and zero initial impulse, one
correction gives ``(-0.4, 0, 0.8)``. Repeated corrections approach the Coulomb
solution ``(-0.5, 0, 1)``. Accuracy tests therefore select their correction
budget explicitly and retain the same physical assertions.

APGD uses the full Coulomb cone and the existing stabilized free velocity;
it does not apply PGS's heuristic reduction of the friction load for
penetration recovery.

**Device execution.** Each loop maintains a per-world active mask and an
integer condition counting worlds that need another iteration. The condition
is cleared and recomputed on every pass. A Warp conditional loop exits when
that count reaches zero;
worlds that finish sooner remain masked while other worlds continue. Nonempty
worlds perform at least one trial: residual checks occur after updates, not
before the first step. Projection differences use double precision and direct
gradient expressions in the interior to avoid false convergence from
cancellation at large impulses; operator products and impulses remain float32.

CUDA APGD requires a Warp build and CUDA driver supporting CUDA 12.4+
conditional graphs; unsupported runtimes raise an error during solver
allocation. Each loop body is captured once, so graph size does not grow with
the iteration budgets. All estimates, reductions, and stopping decisions run
on the GPU during graph replay using preallocated arrays. Uncaptured CUDA
execution reads loop conditions back to the host. CPU execution also supports
early termination. There is no fixed-loop fallback.
"""

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
