.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.solvers.style3d
======================

Style3D solver module.

This module provides helper functions for setting up Style3D cloth assets.
Use :class:`~newton.solvers.SolverStyle3D` as the canonical public solver class.

Use ``dynamics=SolverStyle3D.Dynamics.ALL`` to couple cloth and affine rigid
bodies through particle-shape contacts. Both sides contribute forces and
diagonal blocks to the shared system; the contact coupling is applied during
PCG with a matrix-free product. Soft contact stiffness, damping, and viscous
friction use ``model.soft_contact_ke``, ``model.soft_contact_kd``, and
``model.soft_contact_kf``. The Coulomb coefficient is the geometric mean of
``model.soft_contact_mu`` and the shape's ``mu``. Edge and triangle soft
contacts are not supported.

``SolverStyle3D.Config.affine_rigidity_ratio`` controls the dimensionless
SO(3) penalty strength relative to the mean inertia of the affine directions.
The actual stiffness is ``affine_rigidity_ratio * trace(S) / (3 * dt**2)``,
where ``S`` is the COM mass second moment, equivalently
``affine_rigidity_ratio * trace(body_inertia) / (6 * dt**2)``.
The default ratio is 1e2; zero disables the penalty.

The ``style3d_balls_dropped_into_bag`` example suspends an open plastic bag by two handles and
drops five rigid balls into it at 60 FPS with two substeps::

   python -m newton.examples style3d_balls_dropped_into_bag

.. note::

   This page documents helper functions exposed through the ``newton.solvers.style3d`` attribute.
   Because ``newton.solvers`` is a module rather than a package, use
   ``from newton.solvers import style3d`` instead of ``import newton.solvers.style3d``.

.. py:module:: newton.solvers.style3d
.. currentmodule:: newton.solvers.style3d

.. rubric:: Functions

.. autofunction:: add_cloth_grid

.. autofunction:: add_cloth_mesh

