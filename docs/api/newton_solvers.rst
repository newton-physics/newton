.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.solvers
==============

Solvers integrate the dynamics of a :class:`~newton.Model` through the common
:class:`~newton.solvers.SolverBase` interface. Newton provides backends for
rigid articulated systems, maximal-coordinate constraints, particles, and
deformable simulation.

For solver-selection guidance and the feature, contact-material, joint-support,
and differentiability comparisons, see the :doc:`Solvers guide </solvers/index>`.
Installed-wheel users can use the stable hosted guide at
https://newton-physics.github.io/newton/stable/solvers/index.html.

.. py:module:: newton.solvers
.. currentmodule:: newton.solvers

.. toctree::
   :hidden:

   newton_solvers_experimental
   newton_solvers_style3d

.. rubric:: Submodules

- :doc:`newton.solvers.experimental <newton_solvers_experimental>`
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
   SolverSemiImplicit
   SolverStyle3D
   SolverVBD
   SolverXPBD
