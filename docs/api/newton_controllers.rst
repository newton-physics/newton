.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.controllers
==================

.. py:module:: newton.controllers
.. currentmodule:: newton.controllers

GPU-accelerated, vectorized control laws.

This module provides standalone controllers that compute signals
for the robot to track. Each controller is a concrete
subclass of :class:`ControllerBase`.

.. experimental::

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ControllerBase
   ControllerJointImpedance
   ControllerJointImpedanceModelFree
