# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Select articulations and deformable objects by label to read or update state.

See :ref:`selection-api` for an example and the supported array and mask shapes.
"""

from ._src.utils.selection import (
    ArticulationView,
    DeformableCurveView,
    DeformableSurfaceView,
    DeformableVolumeView,
)

__all__ = [
    "ArticulationView",
    "DeformableCurveView",
    "DeformableSurfaceView",
    "DeformableVolumeView",
]
