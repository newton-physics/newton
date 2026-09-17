.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

VBD
===

:class:`~newton.solvers.SolverVBD` is a unified implicit solver based on
Vertex Block Descent (VBD): cloth, soft bodies, rigid bodies (joints,
contacts, rods), and their interactions are all solved in one VBD loop, with
joint and contact constraints enforced through an augmented-Lagrangian (ALM)
extension of the method.

.. experimental::

   :class:`~newton.solvers.SolverVBD` is experimental. Its public API,
   behavior, feature support, performance, and implementation may change
   without prior notice.

This page is the VBD backend guide: when to choose the solver, how to set it
up correctly, and where to find runnable examples. For symptom-driven
diagnosis and parameter tuning, see :ref:`Tuning VBD` and the
:ref:`Simulation Tuning` landing page.

.. note::

   This guide is under active development. Each section below states its
   intended scope and links to existing references until the full content
   lands.

When to Choose VBD
------------------

*In development.* This section will cover:

- what VBD is built for: cloth, soft volumes, and rods; rigid bodies that
  interact with them; guaranteed penetration-free particle contact; and mixed
  scenes handled by a single solver;
- when another solver is the better fit: articulated robots in generalized
  coordinates (:class:`~newton.solvers.SolverMuJoCo`,
  :class:`~newton.solvers.SolverFeatherstone`), rigid mechanisms with
  kinematic loops (:class:`~newton.solvers.SolverKamino`), and workflows that
  require differentiability;
- current limitations, stated plainly: experimental API, the supported joint
  subset (see :ref:`Joint feature support`), and unsupported joint features
  such as armature, joint friction, and effort limits.

Until then, start from the :ref:`Supported Features` matrix and the
:ref:`Joint feature support` tables in the :doc:`solver overview
</solvers/index>`. The underlying methods are described in the references
cited by :class:`~newton.solvers.SolverVBD`:

- Anka He Chen, Ziheng Liu, Yin Yang, and Cem Yuksel. 2024. Vertex Block
  Descent. *ACM Trans. Graph.* 43, 4. https://doi.org/10.1145/3658179
- Chris Giles, Elie Diaz, and Cem Yuksel. 2025. Augmented Vertex Block
  Descent. *ACM Trans. Graph.* 44, 4. https://doi.org/10.1145/3731195
  (the deprecated legacy rigid constraint path)

API Explanations
----------------

Construction and Stepping
~~~~~~~~~~~~~~~~~~~~~~~~~

*In development.* This section will cover:

- the canonical build-and-step loop as an annotated code example: build the
  model, call :meth:`~newton.ModelBuilder.color`, finalize, construct
  :class:`~newton.CollisionPipeline` and :class:`~newton.Contacts`, construct
  :class:`~newton.solvers.SolverVBD`, then collide/step/swap per substep;
- why coloring is required: VBD updates independent blocks in parallel, and
  the color groups define which vertices and bodies may update together;
- the ``CollisionPipeline`` → ``Contacts`` → ``SolverVBD`` construction-order
  rule and what breaks when it is violated;
- ``dt`` and substeps conventions used by the examples;
- CUDA graph capture: which state must be pre-allocated before capture, the
  stream-ordered memory pool caveat, and the run-one-uncaptured-step
  workaround;
- opt-in determinism (the ``deterministic`` constructor argument).

Until then, see the :class:`~newton.solvers.SolverVBD` API reference, which
documents the constructor and includes a minimal working simulation loop.

Model Inputs
~~~~~~~~~~~~

*In development.* This section will explain what the solver consumes from
:class:`~newton.ModelBuilder`, covering only VBD-specific semantics and
linking to the builder API for the rest:

- cloth: :meth:`~newton.ModelBuilder.add_cloth_mesh` /
  :meth:`~newton.ModelBuilder.add_cloth_grid`, membrane materials
  (``tri_ke``, ``tri_ka``, ``tri_kd``) and bending (``edge_ke``,
  ``edge_kd``), particle mass and radius;
- soft bodies: :meth:`~newton.ModelBuilder.add_soft_mesh` /
  :meth:`~newton.ModelBuilder.add_soft_grid` and the volumetric material
  parameters;
- rods: :meth:`~newton.ModelBuilder.add_rod` /
  :meth:`~newton.ModelBuilder.add_rod_graph`, stretch/shear/bend/twist
  stiffness and damping through joint gains, and Dahl hysteresis attributes;
- rigid bodies and joints: the supported joint types (including ROD), drive
  and limit support, ``joint_enabled``, and the rule that ``body_q`` must
  match ``joint_q`` at solver creation;
- constraint modes: ``rigid_compliant_alm=True`` (recommended) versus the
  deprecated legacy path, hard/soft joint slots,
  :meth:`~newton.solvers.SolverVBD.set_joint_constraint_mode`, and the
  ``JointSlot`` names;
- contact materials: shape ``mu``/``ke``/``kd`` and the ``soft_contact_*``
  model parameters.

Coupling Modes
~~~~~~~~~~~~~~

*In development.* This section will cover:

- interaction inside the solver: deformables and rigid bodies added to the
  same model interact with no extra coupling setup beyond
  :meth:`~newton.ModelBuilder.color`;
- one-way coupling with an external rigid-body solver
  (``integrate_with_external_rigid_solver=True``): the ``state_in`` /
  ``state_out`` semantics, when to prefer it (for example, robot driven by
  :class:`~newton.solvers.SolverMuJoCo` with VBD cloth), and its limits;
- pointers to the multiphysics examples that demonstrate each mode.

Until then, see :doc:`Coupled Solvers </concepts/coupling>`.

Contact Handling
~~~~~~~~~~~~~~~~

*In development.* This section will cover:

- the contact stack: which contacts come from
  :class:`~newton.CollisionPipeline` and which self-contacts the solver
  detects and schedules itself;
- particle self-contact: vertex–triangle and edge–edge detection, buffers,
  and the topological, rest-shape, and external filtering options;
- the penetration-free Divide and Truncate (DAT) scheme: what it guarantees,
  how conservative bounds truncate per-vertex motion, and its cost knobs;
- friction (including ``friction_epsilon`` smoothing) and the rigid contact
  modes (hard contacts, contact history warm-starting);
- Offset Geometric Contact (OGC), to be documented once it is available in
  Newton.

The contact methods are described in:

- Anka H. Chen, Jerry Hsu, Youssef Ayman, and Miles Macklin. 2026. Divide
  and Truncate: A Penetration and Inversion Free Framework for Coupled
  Multi-physics System. In *ACM SIGGRAPH 2026 Conference Papers*.
  https://doi.org/10.1145/3799902.3811143
- Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, and Cem
  Yuksel. 2025. Offset Geometric Contact. *ACM Trans. Graph.* 44, 4.
  https://doi.org/10.1145/3731205

Key Constructor Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

*In development.* One compact orientation table over the three constructor
parameter groups — common, ``particle_*``, and ``rigid_*`` — with a one-line
role per parameter, cross-linked to :class:`~newton.solvers.SolverVBD` for
defaults and to :ref:`Tuning VBD` for effects and recommended values.
Detailed tuning guidance intentionally lives on the tuning page, not here.

Examples
--------

*In development.* This section will present a curated set with a one-line
description each — planned: cloth (``cloth_hanging``, ``cloth_franka``),
soft body (``softbody_hanging``, ``softbody_franka``), rods
(``cable_twist``), rigid contacts (``vbd_rigid_rigid_contact``), mixed
scenes (``vbd_gripper_soft_grid``, ``softbody_dropping_to_cloth``), and
external-solver coupling (``mujoco_vbd_coupled_solver``) — plus how to run
them locally.

Runnable examples using :class:`~newton.solvers.SolverVBD` on GitHub:

- `Cloth examples <https://github.com/newton-physics/newton/tree/main/newton/examples/cloth>`_
- `Soft-body examples <https://github.com/newton-physics/newton/tree/main/newton/examples/softbody>`_
- `Cable examples <https://github.com/newton-physics/newton/tree/main/newton/examples/cable>`_
- `VBD-specific examples <https://github.com/newton-physics/newton/tree/main/newton/examples/vbd>`_
  (rigid and soft contacts, grippers, stiff materials)
- `Multiphysics examples <https://github.com/newton-physics/newton/tree/main/newton/examples/multiphysics>`_
  (coupling with MuJoCo, XPBD, and MPM)
