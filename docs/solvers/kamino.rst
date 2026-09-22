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

Kamino provides three forward-dynamics backends:

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
* ``"lox"`` (opt-in): a primal splitting method with sparse Jacobians and
  dense per-island dynamics. It performs one frozen-linearization solve at the
  configuration supplied by the selected Kamino integrator. Unlike PADMM and
  DVI, it supports singular-inertia frames. Rod joints are not supported.

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

Experimental LOX contact extensions
-----------------------------------

.. experimental::

   The LOX configuration parameters ``contact_compliance``,
   ``contact_compliance_fraction``, ``contact_restitution``, and
   ``contact_spatial_friction`` are prototype features. Their behavior and
   interfaces may change without following the normal deprecation policy.

Enable these features through ``config.lox`` after selecting
``dynamics_solver="lox"``. They are disabled by default.

* ``contact_compliance`` specifies scalar normal compliance in m/N. It adds
  ``contact_compliance / dt**2`` to the normal contact operator. A zero value
  retains hard contact. ``contact_compliance_fraction`` controls recovery of
  existing penetration: one gives backward Euler recovery, and values in
  ``(0, 1)`` request partial recovery.
* ``contact_restitution=True`` reevaluates the speculative impact branch in
  each local contact update using its current reaction-free normal velocity.
  The gap and pre-impact velocity remain frozen during the timestep. This
  mode cannot be combined with ``contact_recoverable_response``.
* ``contact_spatial_friction=True`` uses the shape ``mu_torsional`` and
  ``mu_rolling`` coefficients, both in metres. Each contact uses the arithmetic
  mean of the two shape coefficients. Sliding, torsion, and rolling share one
  elliptic friction budget. Zero angular coefficients retain the existing 3D
  contact solve.

The spatial prototype warm-starts angular reactions by matching geometry pairs
and nearby contact points in body coordinates. It stores world torques and
converts them to impulses using the new timestep and contact frame, then
projects all friction components into the shared budget. Missing contacts and
reset worlds discard their cached angular reactions. Translational contact
warm starts retain their existing behavior.

For these extended contact modes, ``status.r_contact`` reports the canonical
contact-law residual. Optional solution metrics retain the equations-of-motion
and configuration residuals. The hard-contact dual metrics ``r_v_plus``,
``r_ncp_primal``, ``r_ncp_dual``, ``r_ncp_compl``, ``r_vi_natmap``, ``f_ncp``,
and ``f_ccp`` are unavailable and return NaN, with argmax indices set to -1.

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
* **LOX:** ``converged`` and ``iterations`` report LOX's native splitting
  termination state. With ``compute_solution_metrics=True``, ``r_p``, ``r_d``,
  and ``r_c`` are the NCP primal, dual, and complementarity residuals evaluated
  from the final constraint reactions and velocity. Without solution metrics,
  these three fields are NaN; they are also NaN for the experimental extended
  contact modes. ``r_contact`` is the maximum metric-scaled contact natural-map
  residual [sqrt(J)], available without enabling solution metrics; it is NaN
  before a solve and for failed worlds. LOX additionally reports ``accepted``,
  ``failed``, and ``iteration_limit``.

These are absolute maxima: no backend divides them by a reference norm,
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
