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
A collision detection pipeline optimized for primitive shapes.

This pipeline uses an `explicit` broad-phase operating on pre-computed
geometry pairs and a narrow-phase based on the primitive colliders of Newton.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import warp as wp

from ...core.data import DataKamino
from ...core.model import ModelKamino
from ...core.shapes import ShapeType
from ...core.state import StateKamino
from ...core.types import float32, int32, vec6f
from ..contacts import DEFAULT_GEOM_PAIR_CONTACT_GAP, ContactsKamino
from .broadphase import (
    PRIMITIVE_BROADPHASE_SUPPORTED_SHAPES,
    BoundingVolumesData,
    BoundingVolumeType,
    CollisionCandidatesData,
    CollisionCandidatesModel,
    primitive_broadphase_explicit,
)
from .narrowphase import PRIMITIVE_NARROWPHASE_SUPPORTED_SHAPE_PAIRS, primitive_narrowphase

###
# Interfaces
###


class CollisionPipelinePrimitive:
    """
    A collision detection pipeline optimized for primitive shapes.

    This pipeline uses an `explicit` broad-phase operating on pre-computed
    geometry pairs and a narrow-phase based on the primitive colliders of Newton.
    """

    def __init__(
        self,
        model: ModelKamino | None = None,
        bvtype: Literal["aabb", "bs"] = "aabb",
        default_gap: float = DEFAULT_GEOM_PAIR_CONTACT_GAP,
        device: wp.DeviceLike = None,
    ):
        """
        Initialize an instance of Kamino's optimized primitive collision detection pipeline.

        Args:
            model (`ModelKamino`, optional):
                The model container holding the time-invariant data of the system being simulated.\n
                If provided, the detector will be finalized using the provided model and settings.\n
                If `None`, the detector will be created empty without allocating data, and
                can be finalized later by providing a model to the `finalize` method.\n
            bvtype (`Literal["aabb", "bs"]`, optional):
                Type of bounding volume to use in broad-phase.
            default_gap (`float`, optional):
                Default detection gap [m] applied as a floor to per-geometry gaps.
            device (`wp.DeviceLike`, optional):
                The target Warp device for allocation and execution.\n
                If `None`, the `model.device` will be used if a model is provided, otherwise
                it will default to the device preferred by Warp on the given platform.
        """
        # Cache the model reference, target device and settings
        self._model: ModelKamino | None = model
        self._default_gap: float = default_gap
        self._device: wp.DeviceLike = device

        # Convert the bounding volume type from string to enum if necessary
        self._bvtype: BoundingVolumeType = BoundingVolumeType.from_string(bvtype)

        # Declare the internal data containers
        self._cmodel: CollisionCandidatesModel | None = None
        self._cdata: CollisionCandidatesData | None = None
        self._bvdata: BoundingVolumesData | None = None

        # If a builder is provided, proceed to finalize all data allocations
        if model is not None:
            self.finalize(model, bvtype, device)

    ###
    # Properties
    ###

    @property
    def device(self) -> wp.DeviceLike:
        """Returns the Warp device the pipeline operates on."""
        return self._device

    ###
    # Operations
    ###

    def finalize(
        self,
        model: ModelKamino,
        bvtype: Literal["aabb", "bs"] | None = None,
        device: wp.DeviceLike = None,
    ):
        """
        Finalizes the collision detection pipeline by allocating all necessary data structures.

        Args:
            model (`ModelKamino`, optional):
                The model container holding the time-invariant data of the system being simulated.\n
                If provided, the detector will be finalized using the provided model and settings.\n
                If `None`, the detector will be created empty without allocating data, and
                can be finalized later by providing a model to the `finalize` method.\n
            bvtype (`Literal["aabb", "bs"]`, optional):
                Type of bounding volume to use in broad-phase.
            device (`wp.DeviceLike`, optional):
                The target Warp device for allocation and execution.\n
                If `None`, the `model.device` will be used if a model is provided, otherwise
                it will default to the device preferred by Warp on the given platform.
        """
        # Override the model if specified
        if model is not None:
            self._model = model
        if self._model is None:
            raise ValueError("Model must be provided to finalize the CollisionPipelinePrimitive.")
        elif not isinstance(self._model, ModelKamino):
            raise TypeError("CollisionPipelinePrimitive only supports models of type ModelKamino.")

        # Override the device if specified
        if device is not None:
            self._device = device

        # Override the device if specified
        if bvtype is not None:
            self._bvtype = BoundingVolumeType.from_string(bvtype)

        # Retrieve the number of world
        num_worlds = self._model.size.num_worlds
        num_geoms = self._model.geoms.num_geoms

        # Ensure that all shape types are supported by the primitive
        # broad-phase and narrow-phase back-ends before proceeding
        world_num_geom_pairs, geom_pair_wid = self._assert_shapes_supported(self._model)

        # Allocate the collision model data
        with wp.ScopedDevice(self._device):
            # Allocate the bounding volumes data
            self._bvdata = BoundingVolumesData()
            match self._bvtype:
                case BoundingVolumeType.AABB:
                    self._bvdata.aabb = wp.zeros(shape=(num_geoms,), dtype=vec6f)
                case BoundingVolumeType.BS:
                    self._bvdata.radius = wp.zeros(shape=(num_geoms,), dtype=float32)
                case _:
                    raise ValueError(f"Unsupported BoundingVolumeType: {self._bvtype}")

            # Allocate the time-invariant collision candidates model
            self._cmodel = CollisionCandidatesModel(
                num_model_geom_pairs=self._model.geoms.num_collidable_pairs,
                num_world_geom_pairs=world_num_geom_pairs,
                model_num_pairs=wp.array([self._model.geoms.num_collidable_pairs], dtype=int32),
                world_num_pairs=wp.array(world_num_geom_pairs, dtype=int32),
                wid=wp.array(geom_pair_wid, dtype=int32),
                geom_pair=self._model.geoms.collidable_pairs,
            )

            # Allocate the time-varying collision candidates data
            self._cdata = CollisionCandidatesData(
                num_model_geom_pairs=self._model.geoms.num_collidable_pairs,
                model_num_collisions=wp.zeros(shape=(1,), dtype=int32),
                world_num_collisions=wp.zeros(shape=(num_worlds,), dtype=int32),
                wid=wp.zeros(shape=(self._model.geoms.num_collidable_pairs,), dtype=int32),
                geom_pair=wp.zeros_like(self._model.geoms.collidable_pairs),
            )

    def collide(self, data: DataKamino, state: StateKamino, contacts: ContactsKamino):
        """
        Runs the unified collision detection pipeline to generate discrete contacts.

        Args:
            data (DataKamino):
                The data container holding internal time-varying state of the solver.
            state (StateKamino):
                The state container holding the time-varying state of the simulation.
            contacts (ContactsKamino):
                Output contacts container (will be cleared and populated)
        """
        # Ensure that the pipeline has been finalized
        # before proceeding with actual operations
        self._assert_finalized()

        # Clear all active collision candidates and contacts
        self._cdata.clear()
        contacts.clear()

        # Perform the broad-phase collision detection to generate candidate pairs
        primitive_broadphase_explicit(
            body_poses=state.q_i,
            geoms_model=self._model.geoms,
            geoms_data=data.geoms,
            bv_type=self._bvtype,
            bv_data=self._bvdata,
            candidates_model=self._cmodel,
            candidates_data=self._cdata,
            default_gap=self._default_gap,
        )

        # Perform the narrow-phase collision detection to generate active contacts
        primitive_narrowphase(self._model, data, self._cdata, contacts, default_gap=self._default_gap)

    ###
    # Internals
    ###

    def _assert_finalized(self):
        """
        Asserts that the collision detection pipeline has been finalized.

        Raises:
            RuntimeError: If the pipeline has not been finalized.
        """
        if self._cmodel is None or self._cdata is None or self._bvdata is None:
            raise RuntimeError(
                "CollisionPipelinePrimitive has not been finalized. "
                "Please call `finalize(builder, device)` before using the pipeline."
            )

    @staticmethod
    def _assert_shapes_supported(model: ModelKamino, skip_checks: bool = False) -> tuple[list[int], np.ndarray]:
        """
        Checks whether all collidable geometries in the provided
        model are supported by the primitive narrow-phase collider.

        Args:
            model (ModelKamino):
                The model container holding the time-invariant parameters of the simulation.

        Raises:
            ValueError: If any unsupported shape type is found.
        """
        # Iterate over each candidate geometry pair
        geom_type = model.geoms.type.numpy()
        geom_wid = model.geoms.wid.numpy()
        geom_pairs = model.geoms.collidable_pairs.numpy()
        world_num_geom_pairs: list[int] = [0] * model.size.num_worlds
        geom_pair_wid: np.ndarray = np.zeros(shape=(geom_pairs.shape[0],), dtype=np.int32)
        for gid_12 in range(geom_pairs.shape[0]):
            # Retrieve the shape types and world indices of the geometry pair
            gid_1 = geom_pairs[gid_12, 0]
            gid_2 = geom_pairs[gid_12, 1]
            shape_1 = ShapeType(geom_type[gid_1])
            shape_2 = ShapeType(geom_type[gid_2])
            candidate_pair = (min((shape_1, shape_2)), max((shape_1, shape_2)))

            # First check if both shapes are supported by the primitive broad-phase
            if not skip_checks and shape_1 not in PRIMITIVE_BROADPHASE_SUPPORTED_SHAPES:
                raise ValueError(
                    f"Builder contains shape '{shape_1}' which is currently not supported by the primitive broad-phase."
                    "\nPlease consider using the `UNIFIED` collision pipeline, or using alternative shape types."
                )
            if not skip_checks and shape_2 not in PRIMITIVE_BROADPHASE_SUPPORTED_SHAPES:
                raise ValueError(
                    f"Builder contains shape '{shape_2}' which is currently not supported by the primitive broad-phase."
                    "\nPlease consider using the `UNIFIED` collision pipeline, or using alternative shape types."
                )

            # Then check if the shape-pair combination is supported by the primitive narrow-phase
            if not skip_checks and candidate_pair not in PRIMITIVE_NARROWPHASE_SUPPORTED_SHAPE_PAIRS:
                raise ValueError(
                    f"Builder contains shape-pair `{candidate_pair}` with pair index `{gid_12}`, "
                    "but it is currently not supported by the primitive narrow-phase."
                    "\nPlease consider using the `UNIFIED` collision pipeline, or using alternative shape types."
                )

            # Store the world index for this geometry pair
            geom_pair_12_wid = geom_wid[gid_1]
            geom_pair_wid[gid_12] = geom_pair_12_wid
            world_num_geom_pairs[geom_pair_12_wid] += 1

        # Return the per-world geometry pair counts and the per-geom-pair world indices
        return world_num_geom_pairs, geom_pair_wid
