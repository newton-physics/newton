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
The geometry module of Kamino, providing data types and
pipelines (i.e. backends) for Collision Detection (CD).

This module provides a front-end defined by:

- :class:`ContactMode`:
    An enumeration defining the different modes a contact can be in, such as `OPEN`,
    `STICKING`, and `SLIDING`, and defines utilities for computing contacts modes.

- :class:`ContactsData`:
    A simple dataclass defining the data layout and contents of discrete contacts.

- :class:`Contacts`:
    A data interface class for allocating and managing contacts data. This
    serves as the container with which collision detection pipelines operate,
    storing all generated contacts.

- :class:`CollisionDetector`:
    A high-level interface for wrapping collision detection pipelines (i.e. backends).
    This class provides a unified interface for performing collision detection
    using different pipelines, and is responsible for determining the necessary
    allocations of contacts data based on the contents of the simulation.

- :class:`CollisionDetectorSettings`:
    A simple dataclass defining and bundling the set of configurations supported
    by the :class:`CollisionDetector`: class and its subordinate CD pipelines.

- :class:`CollisionPipelineType`:
    An enumeration defining the different collision detection pipelines
    (i.e. backends) supported by Kamino.

- :class:`BroadPhaseMode`:
    A forward declaration of the broad-phase collision detection modes of Newton.

- :class:`BoundingVolumeType`:
    An enumeration defining the different types of bounding volumes
    supported by Kamino's broad-phase collision detection back-ends.

- :class:`CollisionPipelineUnifiedKamino`:
    A collision detection pipeline wrapping and specializing a unified CD pipeline of Newton.

- :class:`CollisionPipelinePrimitive`:
    A collision detection pipeline optimized for primitive shapes.
    This pipeline uses an `EXPLICIT` broad-phase operating on pre-computed
    geometry pairs and a narrow-phase based on the primitive colliders of Newton.
"""

from .contacts import ContactMode, Contacts, ContactsData
from .detector import (
    CollisionDetector,
    CollisionDetectorSettings,
    CollisionPipelineType,
)
from .primitive import BoundingVolumeType, CollisionPipelinePrimitive
from .unified import BroadPhaseMode, CollisionPipelineUnifiedKamino

###
# Module interface
###

__all__ = [
    "BoundingVolumeType",
    "BroadPhaseMode",
    "CollisionDetector",
    "CollisionDetectorSettings",
    "CollisionPipelinePrimitive",
    "CollisionPipelineType",
    "CollisionPipelineUnifiedKamino",
    "ContactMode",
    "Contacts",
    "ContactsData",
]
