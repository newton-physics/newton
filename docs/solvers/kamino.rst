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

Choosing a dynamics solver
--------------------------

Kamino provides two forward-dynamics backends:

* ``"padmm"`` (default): proximal ADMM, dense Jacobians/dynamics, and the Euler
  integrator. It is the slower, more robust option because it solves equality
  and inequality constraints together.
* ``"dvi"`` (opt-in): projected dual iterations, sparse Jacobians, dense dynamics
  with the RCM-reordered blocked LLT solver, and the Euler integrator. Its
  coupling sweeps use the explicit ``L -> B -> C`` schedule: projected bounded
  joint and joint-limit rows, a direct bilateral-joint solve, then a projected
  contact solve. The contact phase supports De Saxce PGS and an opt-in
  associated-contact APGD solver. It is generally faster than PADMM, but can
  solve large active inequality sets less accurately. Dual preconditioning is
  supported as an opt-in setting through
  ``config.dynamics.preconditioning=True``.

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

DVI physical compliance
-----------------------

The DVI backend provides independent physical-compliance controls through
``config.constraints`` for three constraint families:

* ``joint_compliance`` applies to kinematic bilateral-joint rows.
* ``joint_limit_compliance`` applies to active unilateral joint-limit rows.
* ``contact_compliance`` applies isotropically to the two tangential rows and
  normal row of each active contact.

All three values default to zero, which preserves rigid-constraint behavior.
Each nonzero compliance must be paired with its corresponding stabilization
time: ``joint_stabilization_time``, ``joint_limit_stabilization_time``, or
``contact_stabilization_time``. For family ``f``, timestep ``dt``, physical
compliance ``c_f``, and stabilization time ``tau_f``, Kamino forms

.. math::

   E_f = \frac{c_f}{dt\,(dt + \tau_f)}

and adds this diagonal to that family's DVI operator. Compliance is inverse
stiffness. Its units follow the constrained coordinate: [m/N] for a
translational row, [rad/(N·m)] for a rotational row, and [m/N] for contact.

Joint compliance deliberately excludes dynamic actuator rows. Joint-limit
compliance likewise excludes the bounded joint-friction and actuator-effort
rows (the internal ``nbc`` family). Those rows already have their own dynamic
or bound semantics and are not modeled as compliant positional constraints.

``E`` and its represented counterpart ``E_hat = P E P = P^2 E`` are internal
solver storage, where ``P`` is the diagonal dual preconditioner. Users set only
the physical per-family compliance and stabilization-time fields. The
constitutive term ``E lambda`` participates in the effective constraint
velocity and solver residuals; the exported physical post-constraint velocity
continues to represent ``N lambda + v_f`` without that constitutive term, where
``N`` is the physical Delassus operator and ``v_f`` is the free constraint
velocity.

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
