.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Kamino
======

:class:`~newton.solvers.SolverKamino` simulates constrained rigid multi-body
systems in maximal coordinates. It is designed for mechanical assemblies with
kinematic loops, under- or overactuation, joint limits, hard frictional
contacts, and restitutive impacts.

Unlike the other maximal-coordinate solvers, Kamino focuses on constrained
rigid mechanical assemblies rather than particle or deformable simulation.
Kamino is currently in BETA 1, and Newton users are discouraged from depending
on it. Evaluate it only when kinematic loops and hard contact constraints are
primary requirements and an experimental solver is acceptable.

.. experimental::

   :class:`~newton.solvers.SolverKamino` is experimental. Its public API,
   behavior, feature support, performance, and implementation may change
   without prior notice.

See the :class:`~newton.solvers.SolverKamino` API reference for construction
and configuration details. Runnable workflows are available in the
`Kamino examples <https://github.com/newton-physics/newton/tree/main/newton/examples/kamino>`_.

Body acceleration
-----------------

Kamino populates :attr:`~newton.State.body_qdd` when the extended state
attribute is requested. Values are center-of-mass spatial accelerations in the
world frame, with linear acceleration followed by angular acceleration. For a
step of duration ``dt``, Kamino reports the discrete step average
``(body_qd_out - body_qd_in) / dt``. Consequently, a contact impact reports its
velocity impulse divided by ``dt`` rather than a continuous midpoint
acceleration. Resetting a state clears the requested acceleration in each reset
world.

Choosing a dynamics solver
--------------------------

Kamino provides two forward-dynamics backends:

* ``"padmm"`` (default): proximal ADMM, dense Jacobians/dynamics, and the Euler
  integrator. It is the slower, more robust option because it solves equality
  and inequality constraints together.
* ``"dvi"`` (opt-in): projected dual iterations, sparse Jacobians, dense dynamics
  with the RCM-reordered blocked LLT solver, and the Euler integrator. It is
  generally faster, but approximates the coupled problem by alternating between
  a direct solve for equality constraints and projected iterations for
  inequality constraints. As a rule of thumb, DVI solves inequality constraints
  less accurately than PADMM, particularly as the number of active inequalities
  grows. Dual preconditioning is not supported.

Select the backend when constructing the configuration so dependent defaults
initialize consistently:

.. code-block:: python

   config = newton.solvers.SolverKamino.Config(dynamics_solver="dvi")
   solver = newton.solvers.SolverKamino(model, config=config)

DVI is best suited to performance-sensitive rigid mechanisms with relatively
few active contacts; PADMM remains the safer and more broadly validated choice.
Set ``sparse_jacobian=False`` for fully dense DVI, or set
``sparse_dynamics=True`` to use sparse dynamics with the Conjugate Residual
solver.

For large bilateral systems, opt into RCM-reordered factorization explicitly:

.. code-block:: python

   config.dvi.bilateral_solver_type = "LLTBRCM"
   config.dvi.bilateral_solver_kwargs = {
       "block_size": 32,
       "reuse_permutation": True,
       "parallel_factorization": True,
   }

The cached permutation remains mathematically valid when matrix values or
sparsity change and is recomputed automatically if the active dimension
changes. Keep the default ``"LLTB"`` solver for small systems.

DVI APGD unilateral subsolver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set ``config.dvi.unilateral_solver = "apgd"`` to solve bounded joint rows,
joint limits, and contact cones with accelerated projected gradient steps.
The default remains ``"pgs"``. APGD supports dense and sparse operators and
``config.dvi.use_schur_complement``. It retains the existing bilateral/
unilateral split and rigid-constraint model.

.. code-block:: python

   config = newton.solvers.SolverKamino.Config(dynamics_solver="dvi")
   config.dvi.unilateral_solver = "apgd"
   config.dvi.apgd.max_iterations = 64
   config.dvi.apgd.max_nonlinear_corrections = 1
   config.dvi.apgd.tolerance = 1.0e-5
   solver = newton.solvers.SolverKamino(model, config=config)

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

   x = project_K(initial_impulse)
   for each nonlinear correction:
       x_outer = x
       s = correction(A x_outer + b)
       y = x; restart acceleration
       for each APGD iteration:
           g = A y + b + s
           z = project_K(y - g / L)       # Backtrack L; keep y and s fixed
           y = z + beta * (z - x); x = z # Update momentum, with restart
           stop inner loop if the frozen-QP residual meets tolerance
       x = x_outer + relaxation * (x - x_outer)
       stop outer loop if the fresh nonlinear residual meets tolerance

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

**Stopping conditions.** APGD currently uses an infinity-norm natural map;
there is no configurable residual type. The inner and nonlinear loops share
``apgd.tolerance`` because they measure the same map with different corrections.
Backtracking has a separate acceptance condition:

.. list-table:: Per-world termination
   :header-rows: 1
   :widths: 20 45 35

   * - Loop
     - Successful exit
     - Budget or failure exit
   * - Backtracking
     - Finite trial satisfying ``d.T A d <= L * norm(d)**2``, with roundoff allowance.
     - After ``max_backtracks`` rejected trials, or a non-finite curvature or step-size calculation.
   * - Inner APGD
     - Frozen-correction residual ``<= apgd.tolerance`` after an accepted step.
     - After ``max_iterations`` accepted steps, or a failed line search.
   * - Nonlinear correction
     - Fresh nonlinear residual ``<= apgd.tolerance`` after the relaxed update.
     - After ``max_nonlinear_corrections`` solves, or a failed line search.

For backtracking, ``d = candidate - y``. The numerical allowance is
``1e-6 * max(abs(d.T A d), L * norm(d)**2) + 1e-20``; it is not a solver
convergence tolerance. A rejected finite trial doubles ``L`` and retries.

If the inner iteration budget is exhausted, the outer loop uses the partial
solution, applies relaxation, and checks the nonlinear residual. It can then
start another correction solve if needed and budget remains. Exhausting the
nonlinear budget returns the last accepted, possibly unconverged impulses;
it does not raise an exception or imply convergence. A failed line search
retains the last accepted impulses and stops all APGD loops for that world.

Each loop maintains a per-world active mask and an integer condition counting
worlds that need another iteration. The condition is cleared and recomputed
on every pass. A Warp conditional loop exits when that count reaches zero;
worlds that finish sooner remain masked while other worlds continue. Without
conditional graph support, fixed loops use the same masks and stopping tests,
but still launch the remaining masked work. Nonempty worlds perform at least
one trial: the residual checks occur after updates, not before the first step.

**Budgets and accuracy.** All budgets are upper limits; supported conditional
device loops stop early when their residual meets ``apgd.tolerance``.

.. list-table:: APGD controls
   :header-rows: 1
   :widths: 35 10 55

   * - Control
     - Default
     - Meaning
   * - ``max_iterations``
     - 64
     - Accepted APGD steps per frozen-correction QP.
   * - ``max_backtracks``
     - 24
     - Trial steps per APGD iteration, including the initial trial.
   * - ``max_nonlinear_corrections``
     - 1
     - Frozen-correction QPs per unilateral phase.
   * - ``tolerance``
     - ``1e-5``
     - Absolute infinity-norm tolerance for both natural maps.
   * - ``relaxation``
     - 1.0
     - Damping of the impulse update after each QP, in ``(0, 1]``.
   * - ``use_graph_conditionals``
     - ``True``
     - Use device early exit when supported; otherwise use masked fixed loops.

The default performs one frozen-correction approximation. Workloads requiring
tighter Coulomb accuracy can set ``max_nonlinear_corrections`` to a larger
value, such as 20, before constructing the solver. More inner APGD iterations
cannot remove error caused by a stale correction. For example, with
``A = I``, ``b = (10, 0, -1)``, ``mu = 0.5``, and zero initial impulse, one
correction gives ``(-0.4, 0, 0.8)``. Repeated corrections approach the Coulomb
solution ``(-0.5, 0, 1)``. Accuracy tests therefore select their correction
budget explicitly and retain the same physical assertions.

These controls are runtime Python settings and do not add USD material
attributes. The nonlinear correction loop is separate from DVI's bilateral/
unilateral alternation.
``max_alternating_iterations`` and ``bilateral_solve_interval`` continue to
control the existing alternating path; Schur mode eliminates the bilateral
rows during the unilateral solve and recovers their impulses afterward.

**Status and limitations.** The APGD status reports accepted inner
``iterations``, ``apgd_corrections``,
``apgd_backtracks``, and the last phase's ``apgd_residual``. Budget exhaustion
can leave a nonzero residual. An exhausted or non-finite line search sets
``apgd_line_search_failed`` and cannot report convergence. The existing
terminal full-system status also checks joint and contact conditions after
bilateral recovery. APGD does not apply PGS's heuristic reduction of the
friction load for penetration recovery; it uses the full Coulomb cone and
the existing stabilized free velocity.

The final ``converged`` flag uses DVI's feasibility, bilateral, and
complementarity checks with ``config.dvi.tolerance``. This tolerance is
independent of ``config.dvi.apgd.tolerance``. Inspect ``apgd_residual`` when
requiring the APGD natural-map threshold as well; inner or nonlinear budget
exhaustion alone does not determine the full-system flag.

Inspecting terminal status
--------------------------

After each step, :attr:`~newton.solvers.SolverKamino.status` provides one
device-resident terminal status record per world. PADMM and DVI both provide
``converged``, ``iterations``, ``r_p``, ``r_d``, and ``r_c`` fields. Their
residual definitions are backend-specific:

* **PADMM:** Let ``x`` and ``y`` be the current preconditioned impulse iterates,
  ``x_prev`` and ``y_prev`` their previous values, ``P`` the diagonal dual
  preconditioner, and ``eta`` and ``rho`` the proximal and penalty parameters.
  ``r_p = ||P (x - y)||_inf`` is the primal consensus residual
  [N·s or N·m·s].
  ``r_d = ||P^-1 (eta (x - x_prev) + rho (y - y_prev))||_inf`` is the ADMM dual
  residual [m/s or rad/s]. ``r_c`` is the maximum absolute impulse-velocity
  inner product over inequality blocks [J]. The ``P`` factors
  convert the first two residuals from solver scaling to physical constraint
  units.
* **DVI:** With physical impulse ``lambda`` and augmented constraint velocity
  ``v``, ``r_p`` is the maximum infinity-norm projection distance of unilateral
  impulses from the nonnegative limit cone or Coulomb contact cone
  [N·s or N·m·s]. ``r_d`` is the maximum of the corresponding velocity distance
  from the dual cone and the bilateral velocity violation [m/s or rad/s].
  ``r_c = max |lambda_k dot v_k|`` is the maximum inequality complementarity
  violation [J].

These are absolute maxima: neither backend divides them by a reference norm,
constraint count, or tolerance. Additional fields are not portable between
backends.

.. code-block:: python

   status = solver.status
   assert status.device == model.device
   assert status.shape == (model.world_count,)

   # Host inspection is explicit and synchronizes the device-to-host copy.
   status_host = status.numpy()
   unconverged_worlds = (~status_host["converged"].astype(bool)).nonzero()[0]

Terminal status is always maintained. ``collect_solver_info=True`` enables
additional solver diagnostics and adds runtime and memory overhead; it is not
required to access ``status``.

Actuation and forward kinematics
--------------------------------

Kamino dynamics routes actuation independently for each joint DoF. A DoF can
use explicit effort, unbounded implicit PD, or effort-limited implicit PD;
passive armature, damping, and Coulomb friction are likewise configured per
DoF. Implicit-PD target modes require a non-zero applicable gain: velocity
mode requires derivative gain, while position-based modes require proportional
or derivative gain. Coulomb friction supports all non-free joint types, while
joint dynamics and implicit PD currently support revolute, prismatic, and
gimbal joint types only.

The forward-kinematics solver still partitions each joint as entirely passive
or entirely actuated. Different non-passive target modes are allowed within a
joint, but mixing passive and actuated DoFs within one joint is not yet
supported. The ``fk_actuation_flag`` model attribute provides an explicit
joint-level override for this FK partition.
