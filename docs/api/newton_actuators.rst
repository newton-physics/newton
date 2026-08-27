.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.actuators
================

GPU-accelerated actuator models for physics simulations.

This module provides a modular library of actuator components — drives,
clamping, and delay — that compute joint effort from simulation state and
control targets. Components are composed into an :class:`Actuator` instance
and registered with :meth:`~newton.ModelBuilder.add_actuator` during model
construction.

.. experimental::

    The actuator API may change without prior notice. Feedback is welcome —
    please file issues or discussion threads.

.. py:module:: newton.actuators
.. currentmodule:: newton.actuators

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   Actuator
   ActuatorParsed
   ClampingBase
   ClampingDCMotor
   ClampingMaxEffort
   ClampingPositionBased
   ComponentKind
   Delay
   DriveBase
   DriveNeuralLSTM
   DriveNeuralMLP
   DrivePD
   DrivePID
   ResponseOracle
   SchemaNames

.. rubric:: Functions

.. autosummary::
   :toctree: _generated
   :signatures: long

   parse_actuator_prim
   register_actuator_component

.. rubric:: Deprecated

.. list-table::
   :header-rows: 1

   * - Name
     - Guidance
   * - ``Clamping``
     - Deprecated in 1.6; use ClampingBase instead.
   * - ``Controller``
     - Deprecated in 1.6; use DriveBase instead.
   * - ``ControllerNeuralLSTM``
     - Deprecated in 1.6; use DriveNeuralLSTM instead.
   * - ``ControllerNeuralMLP``
     - Deprecated in 1.6; use DriveNeuralMLP instead.
   * - ``ControllerPD``
     - Deprecated in 1.6; use DrivePD instead.
   * - ``ControllerPID``
     - Deprecated in 1.6; use DrivePID instead.
