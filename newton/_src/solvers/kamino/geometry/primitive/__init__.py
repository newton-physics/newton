# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
