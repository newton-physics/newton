.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.tuning
=============

Public API for tuning simulation parameters against recorded measurements.

.. experimental::

The initial workflow supports cables. This module defines its portable contracts:
:class:`CableEvidenceBundle` holds normalized measurements, and
:class:`CableCalibrationResult` records a fit. Recorded drive trajectories are
supplied through a :class:`CableDataSource`.

Raw recording readers and segmentation remain outside Newton. Stored evidence
and trajectories suffice to describe a fit without the original application.

.. py:module:: newton.tuning
.. currentmodule:: newton.tuning

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   CableCalibrationResult
   CableDataSource
   CableEvidenceBundle
   CableRecording
   MaterializedTrajectorySource

.. rubric:: Constants

.. list-table::
   :header-rows: 1

   * - Name
     - Value
   * - ``SCHEMA_VERSION``
     - ``1``
   * - ``STATUS_FAILED``
     - ``failed``
   * - ``STATUS_INTERRUPTED``
     - ``interrupted``
   * - ``STATUS_OK``
     - ``ok``
   * - ``SUPPORTED_SCHEMA_VERSIONS``
     - ``(1,)``
