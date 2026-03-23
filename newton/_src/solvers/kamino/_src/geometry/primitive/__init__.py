# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides a collision detection pipeline (i.e. backend) optimized for primitive shapes.

This pipeline is provided by:

- :class:`CollisionPipelinePrimitive`:
    A collision detection pipeline optimized for primitive shapes.
    This pipeline uses an `EXPLICIT` broad-phase operating on pre-computed
    geometry pairs and a narrow-phase based on the primitive colliders of Newton.

- :class:`BoundingVolumeType`:
    An enumeration defining the different types of bounding volumes
    supported by the primitive broad-phase collision detection back-end.
"""

from .broadphase import BoundingVolumeType
from .pipeline import CollisionPipelinePrimitive

###
# Module interface
###

__all__ = [
    "BoundingVolumeType",
    "CollisionPipelinePrimitive",
]
