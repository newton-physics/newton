.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

VBD
===

:class:`~newton.solvers.SolverVBD` is a unified implicit solver: particles
(cloth and soft bodies) are integrated with Vertex Block Descent (VBD), rigid
bodies (joints, contacts, cables) with Augmented Vertex Block Descent (AVBD),
and both systems can be coupled in a single solver instance.

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

*In development.* This section will position VBD against the other Newton
solvers for cloth, soft-body, rigid, and mixed scenes, and state its current
limitations.

Until then, start from the :ref:`Supported Features` matrix and the
:ref:`Joint feature support` tables in the :doc:`solver overview
</solvers/index>`. The underlying methods are described in the references
cited by :class:`~newton.solvers.SolverVBD`:

- Anka He Chen, Ziheng Liu, Yin Yang, and Cem Yuksel. 2024. Vertex Block
  Descent. *ACM Trans. Graph.* 43, 4. https://doi.org/10.1145/3658179
- Chris Giles, Elie Diaz, and Cem Yuksel. 2025. Augmented Vertex Block
  Descent. *ACM Trans. Graph.* 44, 4. https://doi.org/10.1145/3731195

API Explanations
----------------

Construction and Stepping
~~~~~~~~~~~~~~~~~~~~~~~~~

*In development.* This section will cover the canonical setup and step loop:
coloring via :meth:`~newton.ModelBuilder.color` (required for particles and
for rigid bodies integrated by VBD), the recommended
``CollisionPipeline`` → ``Contacts`` → ``SolverVBD`` construction order, and
the CUDA graph capture constraints around contact state pre-allocation.

Until then, see the :class:`~newton.solvers.SolverVBD` API reference, which
documents the constructor and includes a minimal working simulation loop.

Model Inputs
~~~~~~~~~~~~

*In development.* This section will explain what the solver consumes from
:class:`~newton.ModelBuilder` for each system: triangle meshes for cloth,
tetrahedral meshes for soft bodies, rigid bodies with the supported joint
types (including cables), and the hard/soft joint constraint modes and how to
switch them.

Coupling Modes
~~~~~~~~~~~~~~

*In development.* This section will cover two-way particle–rigid coupling
inside the solver and one-way coupling with an external rigid-body solver
(``integrate_with_external_rigid_solver=True``).

Until then, see :doc:`Coupled Solvers </concepts/coupling>`.

Contact Handling
~~~~~~~~~~~~~~~~

*In development.* This section will explain the contact stack: particle
self-contact detection (vertex–triangle and edge–edge), body–particle
contacts, friction, and the penetration-free Divide and Truncate (DAT)
scheme that truncates particle motion against contact bounds. Coverage of
Offset Geometric Contact (OGC) will be added once it is available in
Newton.

The contact methods are described in:

- Anka He Chen, Jerry Hsu, Youssef Ayman, and Miles Macklin. 2026. Divide
  and Truncate: A Penetration and Inversion Free Framework for Coupled
  Multi-physics Systems. arXiv:2604.15513. https://arxiv.org/abs/2604.15513
- Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, and Cem
  Yuksel. 2025. Offset Geometric Contact. *ACM Trans. Graph.* 44, 4.
  https://doi.org/10.1145/3731205

Key Constructor Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

*In development.* A lean orientation over the three constructor parameter
groups — common, ``particle_*``, and ``rigid_*`` — cross-linked to
:class:`~newton.solvers.SolverVBD` for defaults and to :ref:`Tuning VBD` for
effects and recommended values.

Examples
--------

*In development.* This section will walk through a small, representative set
of examples per system rather than the full list.

Runnable examples using :class:`~newton.solvers.SolverVBD` on GitHub:

- `Cloth examples <https://github.com/newton-physics/newton/tree/main/newton/examples/cloth>`_
- `Soft-body examples <https://github.com/newton-physics/newton/tree/main/newton/examples/softbody>`_
- `Cable examples <https://github.com/newton-physics/newton/tree/main/newton/examples/cable>`_
- `VBD-specific examples <https://github.com/newton-physics/newton/tree/main/newton/examples/vbd>`_
  (rigid and soft contacts, grippers, stiff materials)
- `Multiphysics examples <https://github.com/newton-physics/newton/tree/main/newton/examples/multiphysics>`_
  (coupling with MuJoCo, XPBD, and MPM)
