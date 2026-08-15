.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.selection
================

Structured selection views for model entities.

Use :class:`ArticulationView` for tree-specific roots and reduced-coordinate
operators. Use :class:`JointView` or :class:`BodyView` for topology-independent
attribute access and masks, including closed-loop mechanisms without articulations.

.. py:module:: newton.selection
.. currentmodule:: newton.selection

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ArticulationView
   BodyView
   JointView
