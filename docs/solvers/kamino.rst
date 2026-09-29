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

.. experimental::

   The opt-in DVI APGD mode and its ``config.dvi.apgd`` configuration may
   change without prior notice. Nonlinear convergence and performance should
   be evaluated on the intended workload before selecting this mode.

Set ``config.dvi.unilateral_solver = "apgd"`` to solve bounded joint rows,
joint limits, and contact cones with accelerated projected gradient steps.
The default remains ``"pgs"``. APGD supports dense and sparse operators and
``config.dvi.use_schur_complement``. It retains the existing bilateral/
unilateral split and rigid-constraint model.

.. code-block:: python

   config = newton.solvers.SolverKamino.Config(dynamics_solver="dvi")
   config.dvi.unilateral_solver = "apgd"
   config.dvi.apgd.max_iterations = 64
   config.dvi.apgd.max_corrections = 20
   config.dvi.apgd.tolerance = 1.0e-5
   solver = newton.solvers.SolverKamino(model, config=config)

APGD solves a sequence of convex cone quadratic programs. Between programs it
recomputes the De Saxce normal-velocity correction ``mu * norm(v_t)``; that
correction remains fixed during each inner solve and its backtracking search.
The stopping condition evaluates the nonlinear Coulomb natural map with the
updated velocity. This avoids treating convergence of an associated cone QP
as convergence of Coulomb friction.

``apgd.max_iterations`` bounds inner accelerated steps,
``apgd.max_backtracks`` bounds each line search, and
``apgd.max_corrections`` bounds nonlinear correction iterations per unilateral
phase. ``apgd.relaxation`` can damp the nonlinear update. These controls are
runtime Python settings and do not add USD material attributes.
``max_alternating_iterations`` and ``bilateral_solve_interval`` continue to
control the existing alternating path; Schur mode eliminates the bilateral
rows during the unilateral solve and recovers their impulses afterward.

The APGD status reports accepted inner ``iterations``, ``apgd_corrections``,
``apgd_backtracks``, and the last phase's ``apgd_residual``. Budget exhaustion
can leave a nonzero residual. An exhausted or non-finite line search sets
``apgd_line_search_failed`` and cannot report convergence. The existing
terminal full-system status also checks joint and contact conditions after
bilateral recovery. APGD does not apply PGS's heuristic reduction of the
friction load for penetration recovery; it uses the full Coulomb cone and
the existing stabilized free velocity.

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
