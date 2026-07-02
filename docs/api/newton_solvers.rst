.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.solvers
==============

Solvers integrate the dynamics of a :class:`~newton.Model` through the common
:class:`~newton.solvers.SolverBase` interface. Newton provides backends for
rigid articulated systems, maximal-coordinate constraints, particles, and
deformable simulation.

See ``docs/solvers/index.rst`` for solver-selection guidance and the feature,
contact-material, joint-support, and differentiability comparisons. Installed
wheel users can read the rendered guide at
https://newton-physics.github.io/newton/latest/solvers/index.html or through
the :doc:`Solvers guide </solvers/index>` in this documentation.

.. py:module:: newton.solvers
.. currentmodule:: newton.solvers

.. toctree::
   :hidden:

   newton_solvers_style3d

.. rubric:: Submodules

- :doc:`newton.solvers.style3d <newton_solvers_style3d>`

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   SolverBase
   SolverFeatherstone
   SolverImplicitMPM
   SolverKamino
   SolverMuJoCo
   SolverNotifyFlags
   SolverSemiImplicit
   SolverStyle3D
   SolverVBD
   SolverXPBD
