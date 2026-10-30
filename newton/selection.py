# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Select actuators, articulations, and deformable objects to read or update arrays.

See :ref:`selection-api` for an example and the supported array and mask shapes.
"""

from ._src.actuators.view import ActuatorView
from ._src.utils.selection import (
    ArticulationView,
    DeformableCurveView,
    DeformableSurfaceView,
    DeformableVolumeView,
)

__all__ = [
    "ActuatorView",
    "ArticulationView",
    "DeformableCurveView",
    "DeformableSurfaceView",
    "DeformableVolumeView",
]
