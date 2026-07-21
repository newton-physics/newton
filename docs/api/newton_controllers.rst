.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.controllers
==================

GPU-accelerated, vectorized control laws for Newton physics simulations.

This module provides standalone controllers that compute joint torques or
other actuation signals from simulation state. Each controller is a concrete
subclass of :class:`Controller` and operates on flat 1D arrays matching the
layout of :class:`~newton.State` fields, making them composable with any
Newton solver.

.. experimental::

    The controllers API may change without prior notice. Feedback is welcome —
    please file issues or discussion threads.

.. py:module:: newton.controllers
.. currentmodule:: newton.controllers

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   Controller
   ControllerJointImpedance
   ControllerJointImpedanceModelFree
