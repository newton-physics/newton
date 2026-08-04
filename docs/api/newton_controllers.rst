.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.controllers
==================

GPU-accelerated, vectorized control laws.

This module provides standalone controllers which compute output torques, positions
or velocities for the robot to track. Each controller is a concrete
subclass of :class:`ControllerBase` and operates on flat 1D arrays.

.. experimental::

.. py:module:: newton.controllers
.. currentmodule:: newton.controllers

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ControllerBase
   ControllerJointImpedance
   ControllerJointImpedanceModelFree
