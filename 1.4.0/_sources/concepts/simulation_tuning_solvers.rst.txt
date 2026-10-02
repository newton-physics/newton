.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

.. _Tuning Solver Reference:

Solver Tuning Reference
=======================

This page is the solver reference for :ref:`Simulation Tuning`: shared tuning
concepts followed by the supported knobs per Newton solver, and
order-of-magnitude sanity checks. Start from the :ref:`Simulation Tuning`
landing page for the diagnostic workflow.

.. important::

   The knob tables below are last-known, not a contract. Each solver's own
   constructor or ``Config`` is the source of truth for which options exist;
   confirm an option against it before using it. See :ref:`Simulation Tuning` for
   the full "verify specifics against code" guidance.

Shared Concepts, Not Shared Options
-----------------------------------

Newton solvers do not expose one shared set of tuning parameters. Treat these
as shared concepts, then map them to the solver-specific API below. Before
using a tuning option, confirm that the active solver supports it through its
constructor, configuration object, or documented model attributes. If an option
comes from another Newton solver or an external simulator and is not exposed by
the active Newton solver, do not use it.

The examples in this repository most often tune these shared scene surfaces
before reaching for solver-specific keyword arguments:

- Time stepping: ``frame_dt``, ``sim_substeps``, and the derived ``sim_dt``.
- Contact materials and collision setup: ``default_shape_cfg.ke``,
  ``default_shape_cfg.kd``, ``default_shape_cfg.kf``,
  ``default_shape_cfg.mu``, ``ShapeConfig`` material arguments,
  ``soft_contact_ke``, ``soft_contact_kd``, ``soft_contact_mu``, contact
  ``margin`` and ``gap``, and collision pipeline options such as
  ``broad_phase``, ``soft_contact_margin``, ``reduce_contacts``, and
  ``contact_matching``.
- Drives and articulated models: ``joint_target_ke``, ``joint_target_kd``,
  ``joint_armature``, and ``joint_effort_limit`` where the active solver
  supports them.

Treat those as knobs to inspect first, not as settings that every solver
consumes in the same way.

``dt`` and substeps
   Every call to :meth:`newton.solvers.SolverBase.step` receives a timestep
   ``dt``. Examples commonly use ``frame_dt / substeps`` and call ``step`` in a
   substep loop. Smaller ``dt`` improves stability for stiff contacts, stiff
   drives, high velocities, and low mass or inertia. The cost is roughly
   proportional to the number of substeps.

Model scale, mass, and inertia
   Use meter-kilogram-second units. Large mass and inertia ratios make
   constraints harder to solve. Very small inertias combined with large drive
   stiffness can create large accelerations and unstable velocities.

Contact geometry
   Contact tuning cannot compensate for the wrong collision shape. Primitive
   contact is fastest. SDF and hydroelastic contacts are more expensive but can
   produce better force distribution for manipulation and non-convex geometry.
   See :ref:`Collisions`.

Contact material
   Shape material arrays such as :attr:`~Model.shape_material_mu`,
   :attr:`~Model.shape_material_mu_torsional`,
   :attr:`~Model.shape_material_mu_rolling`,
   :attr:`~Model.shape_material_ke`, and :attr:`~Model.shape_material_kd`
   affect different solvers differently. Author defaults through
   :attr:`~ModelBuilder.default_shape_cfg` before adding shapes, or edit the
   model arrays after finalization.

Drive and limit gains
   Joint drives use :attr:`~Model.joint_target_ke` and
   :attr:`~Model.joint_target_kd`. Joint limits use
   :attr:`~Model.joint_limit_ke` and :attr:`~Model.joint_limit_kd` where
   supported. Check :ref:`Joint feature support` (the table in
   ``docs/solvers/index.rst``) before assuming a solver uses a joint feature.

Effort limits, target rates, armature, and friction
   Target rate limits are usually implemented in control code.
   Check :ref:`Joint feature support` for effort limits, velocity limits,
   armature, and friction rather than inferring support from the model arrays.
   Armature adds joint-space inertia and can lower the effective natural
   frequency of a drive, but it changes the physical model and should be
   justified by actuator or gearbox data when possible.

Collision frequency
   Calling ``model.collide`` or ``pipeline.collide`` every substep is most
   robust. Calling it less frequently can improve performance for expensive
   collision models, but stale contacts can increase penetration or weaken
   grasping.

Solver-Specific Knobs
---------------------

Only tune parameters that exist on the solver you are using. Do not copy option
names from another solver or an external simulator. The notes about example
usage are based on tracked files under ``newton/examples``; they show where the
repository examples spend tuning effort, not a shared solver API.

.. list-table::
   :header-rows: 1
   :widths: 22 44 34

   * - Solver
     - Primary Newton knobs
     - Notes
   * - :class:`~newton.solvers.SolverMuJoCo`
     - ``iterations``, ``ls_iterations``, ``solver``, ``integrator``, ``cone``,
       ``jacobian``, ``impratio``, ``tolerance``, ``ls_tolerance``,
       ``ccd_iterations``, ``ccd_tolerance``, ``sdf_iterations``,
       ``sdf_initpoints``, ``nconmax``, ``njmax``, ``use_mujoco_contacts``,
       ``enable_multiccd``, ``disable_contacts``, ``use_mujoco_cpu``.
     - Uses MuJoCo or MuJoCo Warp semantics. Newton's constructor supports the
       listed options, not every option from external MuJoCo documentation.
       Examples most often tune ``iterations``, ``ls_iterations``, ``solver``,
       ``integrator``, ``cone``, ``impratio``, ``nconmax``, ``njmax``, and
       ``use_mujoco_contacts``.
   * - :class:`~newton.solvers.SolverXPBD`
     - ``iterations``, ``soft_body_relaxation``, ``soft_contact_relaxation``,
       ``joint_linear_relaxation``, ``joint_angular_relaxation``,
       ``joint_linear_compliance``, ``joint_angular_compliance``,
       ``rigid_contact_relaxation``, ``rigid_contact_con_weighting``,
       ``angular_damping``, ``enable_restitution``.
     - More iterations reduce residual positional constraint error. Relaxation
       and compliance change convergence and apparent stiffness. XPBD does not
       use armature, joint friction, effort limits, or velocity limits.
       Examples mostly tune ``iterations`` and ``rigid_contact_relaxation``.
   * - :class:`~newton.solvers.SolverVBD`
     - ``iterations``, ``friction_epsilon``, ``rigid_avbd_alpha``,
       ``rigid_avbd_joint_alpha``, ``rigid_avbd_contact_alpha``,
       ``rigid_avbd_beta``, ``rigid_avbd_linear_beta``,
       ``rigid_avbd_angular_beta``, ``rigid_avbd_gamma``,
       ``rigid_contact_hard``, ``rigid_contact_history``,
       ``rigid_contact_k_start``, ``rigid_contact_stick_motion_eps``,
       ``rigid_contact_stick_freeze_translation_eps``,
       ``rigid_contact_stick_freeze_angular_eps``,
       ``rigid_body_contact_buffer_size``,
       ``rigid_body_particle_contact_buffer_size``,
       ``rigid_joint_linear_ke``, ``rigid_joint_angular_ke``,
       ``rigid_joint_linear_k_start``, ``rigid_joint_angular_k_start``,
       ``rigid_joint_linear_kd``, ``rigid_joint_angular_kd``,
       ``integrate_with_external_rigid_solver``,
       ``particle_enable_self_contact``, ``particle_self_contact_radius``,
       ``particle_self_contact_margin``,
       ``particle_conservative_bound_relaxation``,
       ``particle_vertex_contact_buffer_size``,
       ``particle_edge_contact_buffer_size``,
       ``particle_collision_detection_interval``,
       ``particle_edge_parallel_epsilon``, ``particle_enable_tile_solve``,
       ``particle_topological_contact_filter_threshold``,
       ``particle_rest_shape_contact_exclusion_radius``.
     - Contact history requires matched contacts, for example
       ``CollisionPipeline(contact_matching="latest")``. When recording VBD
       steps in a CUDA graph, construct :class:`~newton.CollisionPipeline`
       before :class:`~newton.solvers.SolverVBD` so contact history is
       pre-allocated, or run one uncaptured solver step before capture. Buffer
       sizes that are too small can drop contacts; sizes that are too large cost
       memory and performance. Examples commonly tune ``iterations``, particle
       self-contact radius and margin, particle contact buffers and filters,
       ``particle_collision_detection_interval``, ``particle_enable_tile_solve``,
       ``rigid_body_contact_buffer_size``,
       ``rigid_body_particle_contact_buffer_size``, ``rigid_contact_hard``,
       ``rigid_contact_history``, and ``rigid_avbd_contact_alpha``.
   * - :class:`~newton.solvers.SolverFeatherstone`
     - ``angular_damping``, ``friction_smoothing``,
       ``update_mass_matrix_interval``, ``use_tile_gemm``, ``fuse_cholesky``.
     - This semi-implicit articulated solver is tuned mainly through ``dt``,
       contact materials, drive gains, joint limits, and armature. Large stiffness
       generally requires smaller ``dt``. Examples use
       ``update_mass_matrix_interval`` when articulation dynamics are coupled
       with cloth or soft-body work.
   * - :class:`~newton.solvers.SolverSemiImplicit`
     - ``angular_damping``, ``friction_smoothing``, ``joint_attach_ke``,
       ``joint_attach_kd``, ``enable_tri_contact``.
     - There is no iteration parameter. For instability, reduce ``dt``, reduce
       stiffness, add damping, or use a solver with an implicit or iterative
       constraint formulation. Examples mostly use constructor defaults; one
       tracked example disables triangle contact with ``enable_tri_contact``.
   * - :class:`~newton.solvers.SolverStyle3D`
     - ``iterations``, ``linear_iterations``, ``drag_spring_stiff``,
       ``enable_mouse_dragging``.
     - Cloth material stiffness is usually authored when calling
       :mod:`newton.solvers.style3d` helper functions. Examples tune
       ``iterations`` on the solver and keep interaction-specific drag knobs
       separate from material stiffness.
   * - :class:`~newton.solvers.SolverKamino`
     - :class:`~newton.solvers.SolverKamino.Config`, including
       ``sparse_jacobian``, ``sparse_dynamics``, ``use_collision_detector``,
       ``use_fk_solver``, ``collision_detector``, ``constraints``,
       ``dynamics``, ``padmm``, ``fk``, ``rotation_correction``, and
       ``integrator``.
     - Experimental. Keep tuning guidance conservative and prefer a dedicated
       Kamino guide for detailed PADMM or closed-loop mechanism tuning.
       Examples construct and pass a ``Config`` object, often with
       ``Config.from_model``.

Drive Gain Sanity Checks
------------------------

Very stiff drives can destabilize a simulation even when the solver is correct.
Use hardware specifications when available. If not, start with conservative
gains and increase gradually.

For a rough drive check, define these variables before choosing gains:

.. list-table::
   :header-rows: 1
   :widths: 18 50 32

   * - Symbol
     - Meaning
     - Typical units
   * - :math:`k`
     - Drive stiffness.
     - :math:`N/m` for prismatic drives;
       :math:`N\cdot m/rad` for revolute drives.
   * - :math:`d`
     - Drive damping.
     - :math:`N\cdot s/m` for prismatic drives;
       :math:`N\cdot m\cdot s/rad` for revolute drives.
   * - :math:`m_{\mathrm{eff}}`
     - Effective mass moved by a prismatic drive.
     - :math:`kg`
   * - :math:`I_{\mathrm{eff}}`
     - Relevant effective inertia about the revolute joint axis.
     - :math:`kg\cdot m^2`
   * - :math:`\omega_n`
     - Approximate natural angular frequency.
     - :math:`rad/s`
   * - :math:`\zeta`
     - Approximate damping ratio.
     - dimensionless

For a prismatic drive, estimate:

.. math::

   \omega_n \approx \sqrt{k / m_{\mathrm{eff}}}

For a revolute drive, estimate:

.. math::

   \omega_n \approx \sqrt{k / I_{\mathrm{eff}}}

The damping ratio can be estimated as:

.. math::

   \zeta \approx d / (2 \sqrt{k m_{\mathrm{eff}}})

for a prismatic drive, or:

.. math::

   \zeta \approx d / (2 \sqrt{k I_{\mathrm{eff}}})

for a revolute drive. Treat these as order-of-magnitude checks, not exact
system identification. There is no universal safe threshold for the
dimensionless product :math:`\omega_n \Delta t`; rather than tune to a fixed
number, run a bounded ``dt`` (or stiffness) sweep and watch for instability. If
:math:`\omega_n \Delta t` grows, reduce ``dt``, reduce stiffness, add physically
justified armature, or switch to a solver/formulation that better matches the
problem.

For grippers and manipulation, set actuator limits before increasing gains:

- Use the real actuator's maximum force or torque when known.
- Clamp target velocity or rate-limit target changes when targets jump far
  from the current joint state.
- Test several object sizes and contact configurations in an isolated grasp
  scene before tuning the full task.
- Prefer better contact geometry over extreme stiffness when the object twists,
  rocks, or slips due to sparse contacts.

.. _contact-stiffness-sanity-checks:

Contact Stiffness Sanity Checks
-------------------------------

A similar force-space spring-damper check is useful for contact stiffness when
the active solver interprets :attr:`~Model.shape_material_ke` and
:attr:`~Model.shape_material_kd` as contact stiffness and damping. This check is
most direct for penalty-style contacts. Solver-specific mappings, such as
MuJoCo contact parameters and VBD AVBD ramping or contact history, can change
the practical meaning of stiffness and damping. Define:

.. list-table::
   :header-rows: 1
   :widths: 18 50 32

   * - Symbol
     - Meaning
     - Newton source
   * - :math:`k_{\mathrm{n}}`
     - Normal contact stiffness :math:`N/m`.
     - Commonly :attr:`~Model.shape_material_ke` or
       :attr:`~ModelBuilder.ShapeConfig.ke`.
   * - :math:`d_{\mathrm{n}}`
     - Normal contact damping :math:`N\cdot s/m`.
     - Commonly :attr:`~Model.shape_material_kd` or
       :attr:`~ModelBuilder.ShapeConfig.kd`.
   * - :math:`m_{\mathrm{eff}}`
     - Effective scalar mass seen along the contact normal :math:`kg`. Prefer a
       solver- or Jacobian-derived value. For two free bodies with purely
       translational normal motion, use their reduced mass.
     - Derived from the contact Jacobian and inverse mass matrix, or approximated
       from the two free-body masses.
   * - :math:`\omega_n`
     - Approximate normal contact natural angular frequency :math:`rad/s`.
     - Derived from :math:`k_{\mathrm{n}}` and :math:`m_{\mathrm{eff}}`.
   * - :math:`\zeta_{\mathrm{n}}`
     - Approximate normal contact damping ratio.
     - Derived from :math:`k_{\mathrm{n}}`, :math:`d_{\mathrm{n}}`, and
       :math:`m_{\mathrm{eff}}`.

Then estimate:

.. math::

   m_{\mathrm{eff}} = \left(\frac{1}{m_1} + \frac{1}{m_2}\right)^{-1}

for the two-free-body translational case. The lighter body mass is an upper
bound only for this approximation: it overestimates :math:`m_{\mathrm{eff}}`
and underestimates :math:`\omega_n`, which can make a timestep/stiffness
combination appear safer than it is.

For a scalar contact row with Jacobian :math:`J` and generalized mass matrix
:math:`M`, the principled quantity is conceptually
:math:`m_{\mathrm{eff}} = (J M^{-1} J^T)^{-1}` when that inverse exists.
Rotational inertia, contact-point offset, articulation, and other active
constraints change this value; the lighter body or subtree mass is not a general
bound for those systems. If a Jacobian-derived value is unavailable, label the
estimate as coarse and verify it with a bounded timestep/stiffness sweep and an
explicit safety margin rather than a universal numeric factor.

.. math::

   \omega_n \approx \sqrt{k_{\mathrm{n}} / m_{\mathrm{eff}}}

.. math::

   \zeta_{\mathrm{n}} \approx d_{\mathrm{n}} / (2 \sqrt{k_{\mathrm{n}} m_{\mathrm{eff}}})

Use this to catch obviously over-stiff contact settings before running large
scenes. There is no universal safe threshold for :math:`\omega_n \Delta t`; use
a bounded sweep rather than a fixed cutoff. If it grows large enough to threaten
stability, lower :math:`k_{\mathrm{n}}`, increase damping within reason, reduce
``dt``, or choose a solver and contact representation that can handle the desired
stiffness. For
torsional or rolling contact effects, the same idea applies with effective
inertia instead of effective mass, but the exact parameter mapping is
solver-dependent.

Contact Tuning
--------------

Use this sequence for contact-heavy scenes:

1. Verify geometry and contact normals with visualization.
2. Check that contact margins and gaps are appropriate for the scene scale.
3. Choose the simplest contact representation that reproduces the needed
   behavior. Prefer primitives first, then a convex hull or convex decomposition.
   When task-relevant non-convex geometry remains, attach a precomputed SDF to
   the triangle mesh and choose a resolution that preserves the necessary
   features. Treat live BVH triangle-mesh collision without an SDF as a fallback,
   not the default production path: its cost scales with mesh complexity,
   triangle winding matters, and it does not support hydroelastic contact.
   Benchmark task behavior and cost, and remove collision geometry from parts
   that never make relevant contact. See :ref:`Mesh Collisions`.
4. Set friction coefficients to realistic values before raising stiffness.
5. Increase contact stiffness only while ``dt`` and solver convergence can
   support it.
6. Add damping to reduce bounce or oscillation.
7. Increase solver-specific convergence work if available.
8. Refresh contacts more frequently if fast motion or manipulation depends on
   current contact points.

Common mistakes:

- Raising ``shape_material_ke`` while keeping a large ``dt``.
- Using a detailed visual mesh as a collision mesh when a primitive, convex
  decomposition, SDF, or hydroelastic representation would be more stable.
- Tuning friction before confirming contacts are generated at the expected
  locations.
- Ignoring contact buffer overflow warnings or symptoms.
- Disabling self-collision globally instead of identifying the specific
  unintended colliding pair.

Performance Tradeoffs
---------------------

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Change
     - Usually improves
     - Usually costs
   * - Reduce ``dt`` or increase substeps
     - Stability, contact accuracy, drive stability
     - Runtime
   * - Increase solver iterations
     - Constraint convergence where supported
     - Runtime
   * - Tighten solver tolerance
     - Accuracy where supported
     - Runtime, sometimes little benefit after convergence plateaus
   * - Increase contact stiffness
     - Penetration resistance
     - Stability margin, may require smaller ``dt``
   * - Increase contact damping
     - Bounce and oscillation control
     - Responsiveness, possible overdamping
   * - Use SDF or hydroelastic contacts
     - Complex contact geometry and manipulation fidelity
     - Collision cost and setup cost
   * - Refresh contacts less often
     - Collision performance
     - Contact accuracy for fast or changing contact configurations
   * - Add armature
     - Drive stability and lower effective natural frequency
     - Physical model fidelity if not actuator-justified
   * - Clamp or rate-limit commands
     - Stability and physical realism
     - Tracking performance if clamps or rates are too restrictive
