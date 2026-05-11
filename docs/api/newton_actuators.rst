.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.actuators
================

GPU-accelerated actuator models for physics simulations.

This module provides a modular library of actuator components — controllers,
clamping, and delay — that compute joint effort from simulation state and
control targets. Components are composed into an :class:`Actuator` instance
and registered with :meth:`~newton.ModelBuilder.add_actuator` during model
construction.

.. py:module:: newton.actuators
.. currentmodule:: newton.actuators

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   Actuator
   ActuatorParsed
   Clamping
   ClampingDCMotor
   ClampingMaxEffort
   ClampingPositionBased
   ComponentKind
   Controller
   ControllerNeuralLSTM
   ControllerNeuralMLP
   ControllerPD
   ControllerPID
   Delay

.. rubric:: Functions

.. autosummary::
   :toctree: _generated
   :signatures: long

   parse_actuator_prim
   register_actuator_component
