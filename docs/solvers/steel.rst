.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

Steel
=====

Steel is Skild AI's rigid-body solver for articulated robots and rigid bodies.
It is integrated into Newton as :class:`~newton.solvers.SolverSteel`. The
solver works in generalized coordinates, uses semi-implicit integration with
implicit joint drives, and solves contacts, joint limits and the other
constraint rows with a matrix-free projected Gauss-Seidel sweep that runs on
CPU and CUDA.

.. experimental::

   :class:`~newton.solvers.SolverSteel`'s public API and behavior may change
   without prior notice. Its matrix-free solve runs on CPU and CUDA; contact
   torsion, contact compliance, joint friction, the non-default
   ``friction_mode`` values and sleeping with ``solve_mode="colored"`` require
   CUDA, and ``solve_mode="colored"`` runs the serial propagation sweep on CPU.

Within the Newton documentation, the :class:`~newton.solvers.SolverSteel` API
reference describes construction, supported features and the constructor
options, the Steel rows of the :ref:`Tuning Solver Reference` cover its tuning
knobs, and :doc:`/concepts/determinism` covers its run-to-run determinism.

External resources
------------------

Skild maintains the Steel documentation on its own site, and this page only
points to it, so the two do not drift.

- `Steel documentation <https://steel.skild.ai/>`_: the documentation site,
  with concepts, guides, reference pages and examples.
- `Quick start <https://steel.skild.ai/start/quick-start/>`_: a first Steel
  simulation, then many worlds under a CUDA graph.
- `Tuning <https://steel.skild.ai/guides/tuning/>`_: how to get a scene right
  and check a setting, when to add substeps or iterations, and recipes for
  common scenes.
- `Porting from PhysX <https://steel.skild.ai/guides/porting-from-physx/>`_:
  the guide for moving a PhysX or Isaac Lab scene to Steel. It covers
  attribute mapping, solver option equivalents, what differs, and a migration
  procedure.
- `The step pipeline <https://steel.skild.ai/concepts/step-pipeline/>`_: the
  two halves of a Steel step, the per-articulation dynamics and the per-world
  constraint solve.
- `Constructor options <https://steel.skild.ai/reference/options/>`_: every
  keyword of the :class:`~newton.solvers.SolverSteel` constructor, generated
  from the solver's source, with guidance on when to change it.
- `Examples <https://steel.skild.ai/examples/>`_: the example scenes, grouped
  by topic, each with the command that runs it.
- `skild-ai/steel on GitHub <https://github.com/skild-ai/steel>`_: the
  repository behind the documentation site, with runnable example scenes
  and step-by-step agent workflows for tuning, debugging, validating, and
  porting from PhysX.
- `Launch write-up <https://www.skild.ai/blogs/steel>`_: Skild's announcement
  of Steel.
