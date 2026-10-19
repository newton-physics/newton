.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

VBD
===

:class:`~newton.solvers.SolverVBD` is a unified implicit solver based on
Vertex Block Descent (VBD): cloth, soft bodies, rigid bodies (joints,
contacts, rods), and their interactions are all solved in one VBD loop, with
joint and contact constraints enforced through an augmented-Lagrangian (ALM)
extension of the method. The compliant ALM constraint path is enabled
by default (``rigid_compliant_alm=True``). Pass ``rigid_compliant_alm=False``
explicitly to retain the deprecated legacy path during its migration period.

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
  (the deprecated legacy rigid constraint path)

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
types (including rods), the default compliant ALM constraint mode
(``rigid_compliant_alm=True``) versus the deprecated legacy path, and the deprecated
legacy-path controls — ``model.vbd.joint_is_hard`` and
:meth:`~newton.solvers.SolverVBD.set_joint_constraint_mode` (deprecated since
1.6; no solver-mode effect under compliant ALM).

Slender Rods
~~~~~~~~~~~~

VBD updates one body at a time, so a rod segment cannot rotate without
stretching its anchors against neighbours that are frozen during its update.
When a rod's stretch stiffness far exceeds its bending stiffness (a ratio
``EA * l**2 / (4 * EI)`` above ~25 for segment length ``l``, typical of threads,
cables and tapes), bending converges very slowly. Pass
``rigid_rod_direct_solve=True`` to solve each rod (a chain, tree or loop of
bodies joined by :attr:`~newton.JointType.ROD` joints) as one block per
iteration instead. See :class:`~newton.solvers.SolverVBD` for the supported
options.

The option allocates its arrays once, so a step captured in a CUDA graph stays
valid after :meth:`~newton.solvers.SolverVBD.notify_model_changed` changes rod
membership (joint enablement, body flags, inverse masses). Call the
notification outside capture, and create a new solver if joints become ROD
joints.

.. experimental::

The block solve costs more per iteration than the coloured per-body update,
and each rod is solved serially by one thread. Measured with
``asv/benchmarks/simulation/bench_rod_direct_solve.py`` (512 rod segments in
total, 60 steps of 10 iterations, seconds):

.. list-table::
   :header-rows: 1

   * - Device
     - 4 rods x 128 segments, coloured / direct
     - 128 rods x 4 segments, coloured / direct
   * - CPU
     - 0.86 / 2.43
     - 0.82 / 1.87
   * - NVIDIA L4
     - 0.26 / 8.69
     - 0.26 / 0.44

These are costs per iteration, not time to a given accuracy: on stiff, slender
rods the direct solve settles in a few iterations where per-body updates need
hundreds. Leave the option off for rods that per-body VBD already resolves
(stretch/bend ratios of order 10 or less), for scenes dominated by a few long
rods on a GPU, and with the unsupported options listed in the reference.

Coupling Modes
~~~~~~~~~~~~~~

*In development.* This section will cover how deformables and rigid bodies
interact inside the solver (no special coupling setup is needed) and one-way
coupling with an external rigid-body solver
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

- Anka H. Chen, Jerry Hsu, Youssef Ayman, and Miles Macklin. 2026. Divide
  and Truncate: A Penetration and Inversion Free Framework for Coupled
  Multi-physics System. In *ACM SIGGRAPH 2026 Conference Papers*.
  https://doi.org/10.1145/3799902.3811143
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
