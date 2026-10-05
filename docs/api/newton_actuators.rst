.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.actuators
================

GPU-accelerated actuator models for physics simulations.

This module provides a modular library of actuator components — drives,
clamping, and input processors such as delay — that compute joint effort from simulation state and
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
   Battery
   ClampingBase
   ClampingDCMotor
   ClampingMaxEffort
   ClampingPositionBased
   ComponentKind
   DriveBAM
   DriveBase
   DriveNeuralLSTM
   DriveNeuralMLP
   DrivePD
   DrivePID
   InputProcessorBacklash
   InputProcessorBase
   InputProcessorDelay
   InputProcessorRandomDelay
   JointSpaceResponse
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
   * - ``Delay``
     - Deprecated in 1.7; use InputProcessorDelay instead.
