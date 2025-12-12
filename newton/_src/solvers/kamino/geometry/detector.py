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
Provides a unified interface for performing Collision Detection in Kamino.

Usage example:

    # Create a model builder
    builder = ModelBuilder()
    # ... add bodies and collision geometries to the builder ...

    # Create a collision detector with desired settings
    settings = CollisionDetectorSettings(
        pipeline=CollisionPipelineType.PRIMITIVE,
        broadphase=BroadPhaseMode.EXPLICIT,
        bvtype=BoundingVolumeType.AABB,
    )

    # Create the collision detector
    detector = CollisionDetector(builder=builder, settings=settings, device="cuda:0")
"""

from dataclasses import dataclass
from enum import IntEnum

import warp as wp
from warp.context import Devicelike

from ..core.builder import ModelBuilder
from ..core.model import Model, ModelData
from ..core.types import override
from ..geometry.contacts import DEFAULT_GEOM_PAIR_CONTACT_MARGIN, DEFAULT_GEOM_PAIR_MAX_CONTACTS, Contacts
from ..geometry.primitive import BoundingVolumeType, CollisionPipelinePrimitive
from ..geometry.unified import BroadPhaseMode, CollisionPipelineUnifiedKamino
from ..utils import logger as msg

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


class CollisionPipelineType(IntEnum):
    """Defines the collision detection pipelines supported in Kamino."""

    PRIMITIVE = 0
    """
    Use the "fast" collision detection pipeline specialized for geometric
    primitives using an "explicit" broad-phase on pre-computed collision
    shape pairs and a narrow-phase using Newton's primitive colliders.
    """

    UNIFIED = 1
    """
    Use Newton's unified collision-detection pipeline using a configurable
    broad-phase that supports `NXN`, `SAP`, or `EXPLICIT` modes, and a
    unified GJK/MPR-based narrow-phase. This pipeline is more general and
    supports arbitrary collision geometries, including meshes and SDFs.
    """

    @override
    def __str__(self):
        """Returns a string representation of the collision detector mode."""
        return f"CollisionDetectorMode.{self.name} ({self.value})"

    @override
    def __repr__(self):
        """Returns a string representation of the collision detector mode."""
        return self.__str__()


###
# Interfaces
###


@dataclass
class CollisionDetectorSettings:
    """Defines the settings for configuring a CollisionDetector."""

    pipeline: str | CollisionPipelineType = CollisionPipelineType.PRIMITIVE
    """
    The type of collision-detection pipeline to use, either `PRIMITIVE` or `UNIFIED`.\n
    Defaults to `PRIMITIVE`.
    """

    broadphase: str | BroadPhaseMode = BroadPhaseMode.EXPLICIT
    """
    The broad-phase collision-detection to use (`NXN`, `SAP`, or `EXPLICIT`).\n
    Defaults to `EXPLICIT`.
    """

    bvtype: str | BoundingVolumeType = BoundingVolumeType.AABB
    """
    The type of bounding volume to use in the broad-phase.\n
    Defaults to `AABB`.
    """

    max_contacts_per_world: int | None = None
    """
    The per-world maximum contacts allocation override.\n
    If specified, it will override the per-world maximum number of contacts
    computed according to the candidate geom-pairs represented in the model.\n
    Defaults to `None`, allowing contact allocations to occur according to the model.
    """

    max_contacts_per_pair: int = DEFAULT_GEOM_PAIR_MAX_CONTACTS
    """
    The maximum number of contacts to generate per candidate geom-pair.\n
    Used to compute the total maximum contacts allocated for the model,
    in conjunction with the total number of candidate geom-pairs.\n
    Defaults to `DEFAULT_GEOM_PAIR_MAX_CONTACTS` (`10`).
    """

    max_triangle_pairs: int = 1_000_000
    """
    The maximum number of triangle-primitive shape pairs to consider in the narrow-phase.\n
    Used only when the model contains triangle meshes or heightfields.\n
    Defaults to `1_000_000`.
    """

    default_contact_margin: float = DEFAULT_GEOM_PAIR_CONTACT_MARGIN
    """
    The default per-geom contact margin used in the narrow-phase.\n
    Used when a collision geometry does not specify a contact margin.\n
    Defaults to `1e-5`.
    """

    def __post_init__(self):
        """Post-initialization processing to convert string enums to their respective types."""
        if isinstance(self.pipeline, str):
            self.pipeline = CollisionPipelineType[self.pipeline.upper()]
        if isinstance(self.broadphase, str):
            self.broadphase = BroadPhaseMode[self.broadphase.upper()]
        if isinstance(self.bvtype, str):
            self.bvtype = BoundingVolumeType[self.bvtype.upper()]


class CollisionDetector:
    """
    Provides a Collision Detection (CD) front-end for Kamino.

    This class is responsible for performing collision detection as well
    as managing the collision containers and their memory allocations.

    Supports two collision pipeline types:

    - `PRIMITIVE`: A fast collision pipeline with specialized for geometric
    primitives using an "explicit" broad-phase on pre-computed collision
    shape pairs and a narrow-phase using Newton's primitive colliders.

    - `UNIFIED`: Newton's unified collision-detection pipeline using a configurable
    broad-phase that supports `NXN`, `SAP`, or `EXPLICIT` modes, and a unified
    GJK/MPR-based narrow-phase. This pipeline is more general and supports arbitrary
    collision geometries, including meshes and SDFs.
    """

    def __init__(
        self,
        model: Model | None = None,
        builder: ModelBuilder | None = None,
        settings: CollisionDetectorSettings | None = None,
        device: Devicelike = None,
    ):
        """
        Initialize the CollisionDetector.

        Args:
            builder(ModelBuilder):
                ModelBuilder instance containing the host-side model definition.
            device(Devicelike):
                The target Warp device for allocation and execution.\n
                If `None`, uses the default device selected by Warp on the given platform.

        """
        # Cache the target device
        self._device: Devicelike = device

        # Cache the collision detector settings
        self._settings: CollisionDetectorSettings | None = settings

        # Declare the contacts container
        self._contacts: Contacts | None = None

        # Declare the collision detection pipelines
        self._unified_pipeline: CollisionPipelineUnifiedKamino | None = None
        self._primitive_pipeline: CollisionPipelinePrimitive | None = None

        # Declare and initialize the caches of contacts allocation sizes
        self._model_max_contacts: int = 0
        self._world_max_contacts: list[int] = [0]

        # Finalize the collision detector if a builder is provided
        if builder is not None and model is not None:
            self.finalize(builder=builder, model=model, settings=settings, device=device)

    ###
    # Properties
    ###

    @property
    def device(self) -> Devicelike:
        """Returns the device on which the CollisionDetector data is allocated and executes."""
        return self._device

    @property
    def settings(self) -> CollisionDetectorSettings:
        """Returns the settings used to configure the CollisionDetector."""
        return self._settings

    @property
    def model_max_contacts(self) -> int:
        """Returns the total maximum number of contacts allocated for the model across all worlds."""
        return self._model_max_contacts

    @property
    def world_max_contacts(self) -> list[int]:
        """Returns the maximum number of contacts allocated for each world."""
        return self._world_max_contacts

    @property
    def contacts(self) -> Contacts:
        """Returns the Contacts container managed by the CollisionDetector."""
        return self._contacts

    ###
    # Operations
    ###

    def finalize(
        self,
        model: Model,
        builder: ModelBuilder,
        settings: CollisionDetectorSettings | None = None,
        device: Devicelike = None,
    ):
        """
        Allocates CollisionDetector data on the target device.

        Args:
            builder(ModelBuilder):
                ModelBuilder instance containing the host-side model definition.
            settings(CollisionDetectorSettings):
                Settings to configure the CollisionDetector.\n
                If `None`, uses default settings.
            device(Devicelike):
                The target Warp device for allocation and execution.\n
                If `None`, uses the default device selected by Warp on the given platform.
        """
        # Check that the builder is valid
        if builder is None:
            raise ValueError("Cannot finalize CollisionDetector: builder is None")
        if not isinstance(builder, ModelBuilder):
            raise TypeError(f"Cannot finalize CollisionDetector: expected ModelBuilder, got {type(builder)}")

        # Check that the model is valid
        if model is None:
            raise ValueError("Cannot finalize CollisionDetector: model is None")
        if not isinstance(model, Model):
            raise TypeError(f"Cannot finalize CollisionDetector: expected Model, got {type(model)}")

        # Override the settings if specified
        if settings is not None:
            if not isinstance(settings, CollisionDetectorSettings):
                raise TypeError(
                    f"Cannot finalize CollisionDetector: expected CollisionDetectorSettings, got {type(settings)}"
                )
            self._settings = settings
        # Otherwise use default settings
        else:
            self._settings = CollisionDetectorSettings()

        # Override the device if specified explicitly
        if device is not None:
            self._device = device
        # Otherwise, use the device of the model
        else:
            self._device = model.device

        # Compute the maximum number of contacts required for the model and each world
        # NOTE: This is a conservative estimate based on the maximum per-world geom-pairs
        _, world_max_contacts = builder.compute_required_contact_capacity(
            max_contacts_per_pair=self._settings.max_contacts_per_pair,
            max_contacts_per_world=self._settings.max_contacts_per_world,
        )
        self._model_max_contacts = sum(world_max_contacts)
        self._world_max_contacts = world_max_contacts
        msg.debug("CollisionDetector: Will allocate for `model_max_contacts`: %s", self._model_max_contacts)
        msg.debug("CollisionDetector: Will allocate for `world_max_contacts`: %s", self._world_max_contacts)

        # Create the contacts interface which will allocate all contacts data arrays
        # NOTE: If internal allocations happen, then they will contain
        # the contacts generated by the collision detection pipelines
        self._contacts = Contacts(capacity=self._world_max_contacts, device=self._device)

        # Proceed with allocations only if the model admits contacts, which
        # occurs when collision geometries defined in the builder and model
        if self._model_max_contacts > 0:
            # Initialize the configured collision detection pipeline
            match self._settings.pipeline:
                case CollisionPipelineType.PRIMITIVE:
                    self._primitive_pipeline = CollisionPipelinePrimitive(
                        device=self._device,
                        builder=builder,
                        bvtype=self._settings.bvtype,
                        default_margin=self._settings.default_contact_margin,
                    )
                case CollisionPipelineType.UNIFIED:
                    self._unified_pipeline = CollisionPipelineUnifiedKamino(
                        device=self._device,
                        model=model,
                        builder=builder,
                        broadphase=self._settings.broadphase,
                        # TODO: Add support for bvtype in unified pipeline
                        default_margin=self._settings.default_contact_margin,
                        max_triangle_pairs=self._settings.max_triangle_pairs,
                        max_contacts_per_pair=self._settings.max_contacts_per_pair,
                    )
                case _:
                    raise ValueError(f"Unsupported CollisionPipelineType: {self._settings.pipeline}")

    def collide(self, model: Model, data: ModelData):
        """
        Executes collision detection given a model and its associated data.

        This operation will use the `PRIMITIVE` or `UNIFIED` pipeline depending on
        the configuration set during the initialization of the CollisionDetector.

        Args:
            model (Model): The Model instance containing the collision geometries
            data (ModelData): The ModelData instance containing the state of the geometries
        """
        # Skip this operation if no contacts data has been allocated
        if self._contacts is None or self._model_max_contacts <= 0:
            return

        # Ensure that a collision detection pipeline has been created
        if self._primitive_pipeline is None and self._unified_pipeline is None:
            raise RuntimeError("Cannot perform collision detection: a collision pipeline has not been created")

        # Ensure that the model and data are valid
        if model is None:
            raise ValueError("Cannot perform collision detection: model is None")
        if not isinstance(model, Model):
            raise TypeError(f"Cannot perform collision detection: expected Model, got {type(model)}")
        if data is None:
            raise ValueError("Cannot perform collision detection: data is None")
        if not isinstance(data, ModelData):
            raise TypeError(f"Cannot perform collision detection: expected ModelData, got {type(data)}")

        # Execute the configured collision detection pipeline
        match self._settings.pipeline:
            case CollisionPipelineType.PRIMITIVE:
                self._primitive_pipeline.collide(model, data, self._contacts)
            case CollisionPipelineType.UNIFIED:
                self._unified_pipeline.collide(model, data, self._contacts)
            case _:
                raise ValueError(f"Unsupported CollisionPipelineType: {self._settings.pipeline}")
