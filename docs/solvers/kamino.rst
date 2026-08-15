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
solver. With
``collect_solver_info=True``, DVI stores terminal residual status that should
not be interpreted as PADMM ADMM residuals.

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

Joint friction
--------------

Kamino applies :attr:`~newton.Model.joint_friction` as a per-DoF Coulomb
effort magnitude [N or N·m]. For generalized joint velocity ``dq``, the
friction effort is

.. math::

   \tau_f = -f \frac{dq}{\max\left(|dq|, v_t\right)},

where ``f`` is the configured joint friction and ``v_t`` is
:attr:`~newton.solvers.SolverKamino.Config.joint_friction_velocity_threshold`
[m/s or rad/s]. The effort opposes motion, is bounded by ``f``, and decreases
linearly to exactly zero within the threshold.

This continuous regularization is suitable for deterministic sliding friction
but does not model true static stiction at zero velocity. Keep viscous effects
in :attr:`~newton.Model.joint_damping`; Kamino treats damping and Coulomb
friction as separate properties.

Joint friction must be finite and non-negative when constructing
:class:`~newton.solvers.SolverKamino`. The solver aliases the Newton friction
array so runtime edits remain graph-capturable without requiring solver-side
host synchronization. If an aliased runtime value is negative or non-finite,
Kamino ignores that DoF's friction for the step instead of allowing invalid
effort into PADMM or DVI.
