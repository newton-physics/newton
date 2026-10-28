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
* ``"lox"`` (opt-in, experimental): a primal splitting method. Like DVI, it
  alternates between solving for equality constraints (and smooth dynamics in
  general) and solving for inequality constraints; however, in LOX the split is
  performed at the velocity level, and ADMM is used to reach a consensus.

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

DVI unilateral subsolvers
^^^^^^^^^^^^^^^^^^^^^^^^^

DVI offers two subsolvers for bounded joint rows, limits, and contacts:

* **PGS** (default) uses projected Gauss-Seidel updates with less work per
  iteration, making it a simple choice for modest contact problems.
* **APGD** offers a faster convergence rate for each frozen-correction
  quadratic problem, at the cost of objective-gradient evaluations and an
  inner backtracking loop.
  CUDA execution requires conditional-graph support.

Both support dense and sparse operators and bilateral Schur elimination.
See the :class:`~newton.solvers.SolverKamino` configuration reference for
subsolver selection and iteration controls.

Constraint recovery and material compliance
-------------------------------------------

PADMM and DVI share the constraint settings in ``config.constraints``. Each
constraint group has a global recovery time and speed cap. Bilateral joints
remain rigid; contacts and joint limits can opt into an implicit spring-damper
law using model stiffness and damping.

.. code-block:: python

   config = newton.solvers.SolverKamino.Config(dynamics_solver="dvi")
   config.constraints.joint_recovery_time = 0.01
   config.constraints.joint_recovery_speed = 0.05

   config.constraints.contact_compliance = True
   config.constraints.contact_recovery_time = None  # Derive c / k per contact.
   config.constraints.contact_recovery_speed = 0.5

   config.constraints.limit_compliance = True
   config.constraints.limit_recovery_time = None  # Derive c / k per joint DoF.
   config.constraints.limit_recovery_speed = 0.5
   config.constraints.limit_recovery_speed_angular = 1.0
   solver = newton.solvers.SolverKamino(model, config=config)

Recovery times are in seconds. ``joint_recovery_speed`` supplies one numerical
cap for all bilateral rows: m/s for linear rows and rad/s for angular rows.
Contact and linear-limit caps are in m/s; ``limit_recovery_speed_angular`` is
in rad/s. Caps apply independently to each constraint row. Their default is
infinity (unlimited); zero disables positional recovery while retaining the
velocity constraint and any physical compliance. They do not cap restitution,
motor targets, or body velocities. These settings are configured before solver
construction.

For step size :math:`h`, positive stiffness :math:`k`, damping :math:`c`, and
signed constraint violation :math:`\phi`, backward Euler gives

.. math::

   \tau = c/k, \qquad b = \frac{\phi}{h+\tau}, \qquad
   E = \frac{1}{k h(h+\tau)}.

The solver adds :math:`E` to the physical constraint operator and adds the
capped bias :math:`b` to the constraint velocity. A global
``contact_recovery_time`` or ``limit_recovery_time`` overrides :math:`\tau`
and therefore sets the effective damping to :math:`k\tau` in **both** terms.
With no override, the solver uses the model damping. Once the recovery cap is
active, the law differs from the uncapped linear spring-damper. Numerical
regularization and ADMM proximal penalties remain separate from :math:`E`.

Contact stiffness/damping come from positive
:attr:`~newton.Contacts.rigid_contact_stiffness` and the corresponding
:attr:`~newton.Contacts.rigid_contact_damping` when supplied. Otherwise, they
are the arithmetic means of :attr:`~newton.Model.shape_material_ke` and
:attr:`~newton.Model.shape_material_kd` of the two shapes. This fallback also
works with Kamino's internal collision pipeline. Only the contact normal
receives compliance; tangential rows retain the Coulomb friction law.
The normal spring impulse contributes fully to the friction load, including in
DVI's PGS solver. Its hard-contact stabilization exclusion does not apply to
physical spring forces.
Compliant contacts obtain bounce from their spring-damper and ignore the
material restitution coefficient.

Limits use :attr:`~newton.Model.joint_limit_ke` and
:attr:`~newton.Model.joint_limit_kd` of their joint DoF. Limit detection is
unchanged. Non-positive or infinite stiffness retains the hard law for either family;
this interface does not represent a pure damper with zero stiffness. The
per-DOF drive gains and passive damping are independent of structural joint
recovery and are not reused for it.

The speed cap and recovery time apply to penetration or violated limits.
Speculative contacts retain the positive-gap closing allowance
:math:`\max(\phi,0)/h`; the cap does not slow approach before contact. Contact
dead zones and the existing hard-contact restitution rules remain in effect.

Compatibility and USD authoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, compliance is disabled, recovery times are ``None``, and caps are
unlimited. Existing configurations retain their dimensionless ``alpha``
(joints), ``beta`` (limits), and ``gamma`` (contacts) recovery fractions. Hard
rows without a recovery-time override continue to use :math:`s\phi/h` for
their corresponding legacy fraction :math:`s`. An explicit recovery time
takes precedence. At a given timestep, the equivalent time is
:math:`\tau=h(1/s-1)` for :math:`s>0`; this conversion changes with timestep.
For example, a fraction of 0.01 at 240 Hz corresponds to 0.4125 s.
For a zero fraction, set the corresponding recovery time and recovery-speed
cap to zero; for limits, set both the linear and angular caps to zero.
This conversion preserves rigid-row recovery. Enabling finite stiffness can
introduce elastic deflection or bounce, so material gains must suit the intended
motion rather than reproduce rigid-contact trajectories exactly.

Call :meth:`~newton.solvers.SolverKamino.register_custom_attributes` on the
builder before importing USD. The following custom PhysicsScene attributes
use the prefix ``newton:kamino:constraints:``:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Group
     - Attribute suffixes
   * - Joints
     - ``jointRecoveryTime``, ``jointRecoverySpeed``
   * - Limits
     - ``limitRecoveryTime``, ``limitRecoverySpeed``, ``limitRecoverySpeedAngular``, ``limitCompliance``
   * - Contacts
     - ``contactRecoveryTime``, ``contactRecoverySpeed``, ``contactCompliance``

Times and speeds use the same SI units as Python; compliance switches are
booleans. An unauthored time, or the storage sentinel -1, means ``None``.
Explicit Python configuration overrides imported values. Invalid values are
rejected when resolving the configuration. Existing Newton material and
joint-limit USD properties supply the per-constraint gains.

The shared recovery/compliance settings currently support PADMM and DVI.
LOX rejects non-default values for these new settings and retains its existing
``config.lox.contact_compliance`` option and recovery controls.

LOX prescribed bodies and shared settings
-----------------------------------------

LOX advances bodies flagged :attr:`~newton.BodyFlags.KINEMATIC`, and massless
bodies whose only joints are free joints to the world, with their velocity in
``state_in.body_qd``. To animate such a body, write the velocity that reaches its
target pose within the step. PADMM and DVI keep the pose of these bodies and only
read their velocity.

Bodies welded to a kinematic body by fixed joints of its articulation move
rigidly with it, like the bodies of a MuJoCo tree without joints: LOX sets their
pose and velocity from the kinematic body at the beginning and at the end of each
step. Massless bodies on other joints stay dynamic and move with the bodies they
are attached to. With ``config.lox.eliminate_fixed_world_islands`` enabled, LOX
also keeps the bodies welded to the world at their pose.

LOX simulates bodies flagged :attr:`~newton.BodyFlags.PROXY` as dynamic bodies
with their mass and inertia.

LOX keeps speculative contacts, whose shapes are still separated, and lets each
close its gap within the step. It ignores
``config.dynamics.cull_speculative_contacts``.

LOX bounds joint drift with ``config.lox.joint_max_correction``,
``config.lox.position_tolerance`` and ``config.lox.rotation_tolerance``, so it
ignores ``config.constraints.alpha``. ``config.collect_solver_info`` collects the
PADMM and DVI iteration logs; LOX reports its iterations and residuals through
:attr:`~newton.solvers.SolverKamino.status`.

LOX solves its own primal problem, so ``sparse_dynamics``,
``config.dynamics.linear_solver_type``, ``config.dynamics.linear_solver_kwargs``
and ``config.dynamics.preconditioning`` only affect the dual problem of the
optional solution metrics. LOX always warm-starts from its containers;
``config.lox.contact_warmstart_method`` selects how it matches contacts.

LOX unilateral projection
-------------------------

LOX projects contacts, joint limits, and joint friction with mass-split
sweeps. ``config.lox.gauss_seidel_max_colors`` selects the sweep: one gives a
fully parallel Jacobi sweep, and larger values process up to that many
approximate constraint colors sequentially. ``config.lox.projection_acceleration``
optionally accelerates the resulting map:

* ``"none"`` (default) runs plain sweeps.
* ``"apgd"`` applies restarted Nesterov extrapolation and requires
  ``gauss_seidel_max_colors=1``.
* ``"anderson"`` applies one safeguarded Anderson direction
  after every sweep, for either the Jacobi or the colored map. Set
  ``projection_anderson_recycle=False`` to discard the direction between
  splitting iterations.

LOX contact extensions
----------------------

Enable these features through ``config.lox`` after selecting
``dynamics_solver="lox"``. They are disabled by default.

* ``contact_compliance=True`` turns contacts with a positive per-contact
  stiffness ``k`` [N/m] (:attr:`newton.Contacts.rigid_contact_stiffness`) into
  implicit spring-dampers, with the damping ``d`` [N s/m] of
  :attr:`newton.Contacts.rigid_contact_damping`. They add
  ``1 / (dt (k dt + d))`` to the normal contact operator and recover the fraction
  ``k dt / (k dt + d)`` of existing penetration per step. Their bounce comes from
  the spring-damper, so they ignore restitution. Contacts without a positive
  stiffness, and contacts detected by Kamino's own collision pipelines, remain
  hard. :class:`newton.CollisionPipeline` provides per-contact stiffness only for
  hydroelastic contacts.
* ``contact_implicit_restitution=True`` decides the restitution of speculative
  impacts inside each local contact update, from its current reaction-free normal
  velocity, which better preserves energy in speculative impacts. Restitution
  applies either way: by default, the bounce target is fixed at the beginning of
  the step.
* ``contact_spatial_friction=True`` accounts for the shape ``mu_torsional`` and
  ``mu_rolling`` coefficients, both in metres. Sliding, torsion, and rolling
  share one elliptic friction budget. Contacts with zero angular coefficients
  retain the classical 3D contact solve. Kamino's primitive collision pipeline
  provides no angular coefficients.

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
  termination state. With body twists ``v`` of the smooth solve and ``p`` of
  the unilateral projection, and the splitting body weight ``W``,
  ``r_p = ||W (v - p)||_inf`` is the consensus residual [N·s or N·m·s] and
  ``r_d = ||v - v_prev||_inf`` is the last change of the smooth solution
  [m/s or rad/s]. ``r_c`` is the squared maximum Delassus-scaled natural-map
  residual of the joint-friction, limit, and contact rows under their contact
  law [J], which measures feasibility and complementarity together. These
  residuals do not require solution metrics; they are NaN before a solve and
  for failed worlds. LOX additionally reports ``failed``; a world that is
  neither converged nor failed reached the iteration limit and uses its last
  projected iterate. A failed world stays failed until it is reset, for example
  with ``solver.reset(..., world_mask=...)``. Until then, LOX skips its solve,
  and its bodies keep their input velocity, without gravity, joint or contact
  forces, so check ``failed`` to detect and reset these worlds.

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
