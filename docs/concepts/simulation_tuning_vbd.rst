.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

.. _Tuning VBD:

VBD Tuning
==========

This page explains how :class:`~newton.solvers.SolverVBD` reacts to its solver
and model parameters, what typically goes wrong and how to fix it, and which
parameter combinations are known to work. See
:ref:`Simulation Tuning` for the diagnostic workflow, :ref:`Tuning Solver
Reference` for the full knob list, and :doc:`VBD </solvers/vbd>` for setup and
API explanations.

.. important::

   The specific values, mode names, and formulas on this page reflect the code
   at a point in time and can drift. Treat them as starting points and verify
   any you rely on against the cited source (for example
   :class:`~newton.solvers.SolverVBD` and its kernels). See
   :ref:`Simulation Tuning` for the full guidance.

.. note::

   This page is under active development. Each section below states its
   intended scope and links to existing references until the full content
   lands.

Mental Model
------------

*In development.* This section will explain, briefly and without paper-level
math, how the solver converges and why it fails:

- how VBD updates one block at a time — a cloth or soft-body vertex, or a
  rigid body — from its local system while its neighbors stay fixed within
  the update, and why that makes heavily deformed or stiffly coupled
  configurations the hard case;
- how joint and contact constraints are enforced inside the same VBD loop
  through an augmented-Lagrangian (ALM) formulation
  (``rigid_compliant_alm=True``, recommended; the legacy path is deprecated),
  and what roles ``rigid_avbd_alpha``, ``rigid_avbd_beta``, and
  ``rigid_avbd_gamma`` play;
- how the penetration-free Divide and Truncate (DAT) scheme bounds and
  truncates per-vertex motion against contacts, and what that implies for
  self-contact parameters.

For the underlying method, see the VBD paper cited in
:class:`~newton.solvers.SolverVBD`.

Parameter Effects
-----------------

VBD-Specific Parameters
~~~~~~~~~~~~~~~~~~~~~~~

*In development.* Effects and interactions of the constructor parameters, by
group: common (``iterations``, ``friction_epsilon``), particle (the
self-contact family, ``particle_collision_detection_interval``,
``particle_enable_tile_solve``, contact buffer sizes), and rigid constraints
(``rigid_compliant_alm`` mode selection,
``rigid_avbd_alpha``/``beta``/``gamma``, penalty seeds and ceilings,
``rigid_contact_hard``, ``rigid_contact_history``).

Until then, the authoritative parameter list with defaults is
:class:`~newton.solvers.SolverVBD`; the supported-knob summary lives in
:ref:`Tuning Solver Reference`.

Model Parameters
~~~~~~~~~~~~~~~~

*In development.* How the solver interprets material and scene parameters:
cloth membrane and bending stiffness and damping (``tri_ke``, ``tri_ka``,
``tri_kd``, ``edge_ke``, ``edge_kd``), soft-body volumetric material, contact
materials (``soft_contact_*`` and shape ``mu``/``ke``/``kd``), joint and rod
stiffness and damping, particle mass and radius, and the
``dt``–substeps–``iterations`` trade-off.

Contact Handling
~~~~~~~~~~~~~~~~

*In development.* How the contact stack responds to tuning: the self-contact
detection family (radius, margin, detection interval, buffer sizes, and the
topological and rest-shape filters), the penetration-free Divide and Truncate
(DAT) truncation and ``particle_conservative_bound_relaxation``, and the
interplay of contact stiffness, damping, and friction. Coverage of Offset
Geometric Contact (OGC) will be added once it is available in Newton.

Best Practices
--------------

What Can Go Wrong and How to Fix It
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*In development.* One entry per failure mode — symptom, cause, fix — topped
by a quick-lookup symptom table. Planned entries include cloth oscillation
that more iterations do not fix, outlier-vertex explosions, self-contact
tunneling, contact chattering and penetration, grasp slip versus finger
penetration, and joint drift.

Until then, use the generic Symptom Table in :ref:`Simulation Tuning`.

Empirical Parameter Combinations That Work
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*In development.* Known-good value tables per scenario family — cloth, soft
body, rods (cables), rigid contacts, mixed scenes — each linking to a
representative
runnable example from the `examples directory
<https://github.com/newton-physics/newton/tree/main/newton/examples>`_.

Rules of Thumb
~~~~~~~~~~~~~~

*In development.* Distilled starting values and tuning orderings, with the
reasoning that makes them portable across scenes.
