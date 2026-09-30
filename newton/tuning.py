# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Public API for tuning simulation parameters against recorded measurements.

.. experimental::

The initial workflow supports cables. This module defines its portable contracts:
:class:`CableEvidenceBundle` holds normalized measurements, and
:class:`CableCalibrationResult` records a fit. Recorded drive trajectories are
supplied through a :class:`CableDataSource`.

Raw recording readers and segmentation remain outside Newton. Stored evidence
and trajectories suffice to describe a fit without the original application.
"""

from ._src.tuning.data_source import CableDataSource
from ._src.tuning.evidence import (
    CableEvidenceBundle,
    CableRecording,
)
from ._src.tuning.result import (
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_OK,
    CableCalibrationResult,
)
from ._src.tuning.schema import SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS
from ._src.tuning.trajectory import MaterializedTrajectorySource

__all__ = [
    "SCHEMA_VERSION",
    "STATUS_FAILED",
    "STATUS_INTERRUPTED",
    "STATUS_OK",
    "SUPPORTED_SCHEMA_VERSIONS",
    "CableCalibrationResult",
    "CableDataSource",
    "CableEvidenceBundle",
    "CableRecording",
    "MaterializedTrajectorySource",
]
