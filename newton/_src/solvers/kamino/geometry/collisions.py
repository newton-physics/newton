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
KAMINO: Collision containers use for broad-phase detection
"""

from __future__ import annotations

import numpy as np
import warp as wp
from warp.context import Devicelike

from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.core.joints import JointDoFType
from newton._src.solvers.kamino.core.types import int32, vec2i
from newton._src.solvers.kamino.utils import logger as msg

###
# Module interface
###

__all__ = [
    "Collisions",
    "CollisionsData",
    "CollisionsModel",
    "make_collision_pairs",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###


def make_collision_pairs(builder: ModelBuilder, allow_neighbors: bool = False):
    """
    Construct the collision pair candidates for the given ModelBuilder instance.

    Filtering steps:
        1. filter out self-collisions
        2. filter out same-body collisions
        3. filter out collision between different worlds
        4. filter out collisions according to the collision groupings
        5. filter out neighbor collisions for fixed joints
        6. (optional) filter out neighbor collisions for joints w/ DoF

    Args:
        builder (ModelBuilder): The model builder instance containing the worlds and geometries.
        allow_neighbors (bool, optional): If True, allows neighbor collisions for joints with DoF.
    """
    # Retrieve the number of worlds
    nw = builder.num_worlds

    # Extract the per-world info from the builder
    ncg = [builder._worlds[i].num_collision_geoms for i in range(nw)]

    # Initialize the lists to store the collision candidate pairs and their properties of each world
    world_nxn_num_geom_pairs = []
    model_nxn_geom_pair = []
    model_nxn_pairid = []
    model_nxn_wid = []

    # Iterate over each world and construct the collision geometry pairs info
    ncg_offset = 0
    for wid in range(nw):
        # Initialize the lists to store the collision candidate pairs and their properties
        world_geom_pair = []
        world_pairid = []
        world_wid = []

        # Iterate over each gid pair and filtering out pairs not viable for collision detection
        for gid1_, gid2_ in zip(
            *np.triu_indices(ncg[wid], k=1), strict=False
        ):  # k=1 skip diagonal to exclude self-collisions
            # Convert the per-world local gids to model gid integers
            gid1 = int(gid1_) + ncg_offset
            gid2 = int(gid2_) + ncg_offset

            # Get references to the geometries
            geom1, geom2 = builder.collision_geoms[gid1], builder.collision_geoms[gid2]

            # Get body indices of each geom
            bid1, bid2 = geom1.bid, geom2.bid

            # Get world indices of each geom
            wid1, wid2 = geom1.wid, geom2.wid

            # 2. Check for same-body collision
            is_self_collision = bid1 == bid2

            # 3. Check for different-world collision
            in_same_world = wid1 == wid2

            # 4. Check for collision according to the collision groupings
            are_collidable = ((geom1.group & geom2.collides) != 0) and ((geom2.group & geom1.collides) != 0)
            msg.debug(f"geom1.group: {geom1.group}, geom1.collides: {geom1.collides}")
            msg.debug(f"geom2.group: {geom2.group}, geom2.collides: {geom2.collides}")
            msg.debug(
                f"collision pair ({gid1}, {gid2}): self-collision={is_self_collision}, same-world={in_same_world}, collidable={are_collidable}"
            )

            # 5. Check for neighbor collision for fixed and DoF joints
            are_fixed_neighbors = False
            are_dof_neighbors = False
            for joint in builder.joints:
                if (joint.bid_B == bid1 and joint.bid_F == bid2) or (joint.bid_B == bid2 and joint.bid_F == bid1):
                    if joint.dof_type == JointDoFType.FIXED:
                        are_fixed_neighbors = True
                    elif joint.bid_B < 0:
                        pass
                    else:
                        are_dof_neighbors = True
                    break

            # Assign pairid based on filtering results
            if (not is_self_collision) and (in_same_world) and (are_collidable) and (not are_fixed_neighbors):
                pairid = -1
            else:
                continue  # Skip this pair if it does not pass the filtering

            # Apply final check for DoF neighbor collisions
            if (not allow_neighbors) and are_dof_neighbors:
                continue  # Skip this pair if it does not pass the filtering

            # Append the geometry pair and pairid to the lists
            world_geom_pair.append((gid1, gid2))
            world_pairid.append(pairid)
            world_wid.append(wid)
            msg.debug(f"Adding collision pair: (gid1, gid2): {(gid1, gid2)}")

        # Append the world collision pairs to the model lists
        world_nxn_num_geom_pairs.append(len(world_geom_pair))
        model_nxn_geom_pair.extend(world_geom_pair)
        model_nxn_pairid.extend(world_pairid)
        model_nxn_wid.extend(world_wid)

        # Update the geometry index offset for the next world
        ncg_offset += ncg[wid]

    # Return the model total collision pair candidates and their properties
    return world_nxn_num_geom_pairs, model_nxn_geom_pair, model_nxn_pairid, model_nxn_wid


###
# Containers
###


class CollisionsModel:
    """
    A container to describe the collision model (TODO add more details).

    ncp_w: the number of per-world possible collision pairs, according to the collision groupings

    Details:
        - nw is the number of worlds
        - ng_w is the number of geometries in each world
        - only ng_w*(ng_w-1)/2 pairs are possible in each world because geom self-collisions are excluded.
        - thus the number of collision pairs per-world are ncp_w <= ng_w*(ng_w-1)
        - nxn_pairid indexes the collision pairs in the model (over all worlds)
        - nxn_geom_pair indexes the geometry pairs in the model (over all worlds)
        - nxn_pairid, nxn_geom_pair are constructed from the symmetric matrix of collision pairs
          where the indices of non-zero entries in the upper/lower triangular are represented as
          nxn_geom_pair, and indexed by  nxn_pairid
        - allows running BFS (NxN) collision detection on the pairs in parallel
    """

    def __init__(self):
        self.num_model_geom_pairs: int = 0
        """(host-side) Total number of collision pairs in the model across all worlds."""

        self.num_world_geom_pairs: list[int] = [0]
        """(host-side) Number of collision pairs per world."""

        self.model_num_pairs: wp.array(dtype=int32) | None = None
        """Total number of collisions pairs in the model. Shape of ``(1,)`` and type :class:`int32`."""

        self.world_num_pairs: wp.array(dtype=int32) | None = None
        """The number of collisions pairs per world. Shape of ``(num_worlds,)`` and type :class:`int32`."""

        self.wid: wp.array(dtype=int32) | None = None
        """World index of each collision pair. Shape of ``(sum(ncp_w),)`` and type :class:`int32`."""

        self.pairid: wp.array(dtype=int32) | None = None
        """Index of each the collision pair. Shape of ``(sum(ncp_w),)`` and type :class:`int32`."""

        self.geom_pair: wp.array(dtype=vec2i) | None = None
        """Geometry indices of each collision pair. Shape of ``(sum(ncp_w),)`` and type :class:`vec2i`."""


class CollisionsData:
    def __init__(self):
        self.model_num_collisions: wp.array(dtype=int32) | None = None
        """Number of collisions detected across all worlds in the model. Shape of ``(1,)`` and type :class:`int32`."""

        self.world_num_collisions: wp.array(dtype=int32) | None = None
        """Number of collisions detected per world. Shape of ``(num_worlds,)`` and type :class:`int32`."""

        self.wid: wp.array(dtype=int32) | None = None
        """World index of each active collision pair. Shape of ``(sum(ncp_w),)`` and type :class:`int32`."""

        self.geom_pair: wp.array(dtype=vec2i) | None = None
        """Geometry indices of each active collision pair. Shape of ``(sum(ncp_w),)`` and type :class:`vec2i`."""


###
# Interfaces
###


class Collisions:
    """
    A container to hold and manage collision detection data.
    """

    def __init__(self, builder: ModelBuilder | None = None, device: Devicelike = None):
        # The device on which to allocate the collision data
        self.device = device

        # Collision data containers
        self.cmodel: CollisionsModel = CollisionsModel()
        self.cdata: CollisionsData = CollisionsData()

        # Perofrm memory allocation if max_contacts is specified
        if builder is not None:
            self.allocate(builder, self.device)

    def allocate(self, builder: ModelBuilder, device: Devicelike = None):
        # Override the device if specified
        if device is None:
            self.device = device

        # Retrieve the number of world
        num_worlds = builder.num_worlds

        # Construct collision pairs
        world_nxn_num_geom_pairs, model_nxn_geom_pair, model_nxn_pairid, model_nxn_wid = make_collision_pairs(builder)

        # Allocate the collision model data
        with wp.ScopedDevice(self.device):
            # Set the host-side number of collision pairs allocations
            self.cmodel.num_model_geom_pairs = len(model_nxn_geom_pair)
            self.cmodel.num_world_geom_pairs = world_nxn_num_geom_pairs

            # Allocate the collision model data
            self.cmodel.model_num_pairs = wp.array([self.cmodel.num_model_geom_pairs], dtype=int32)
            self.cmodel.world_num_pairs = wp.array(self.cmodel.num_world_geom_pairs, dtype=int32)
            self.cmodel.wid = wp.array(model_nxn_wid, dtype=int32)
            self.cmodel.pairid = wp.array(model_nxn_pairid, dtype=int32)
            self.cmodel.geom_pair = wp.array(model_nxn_geom_pair, dtype=vec2i)

            # Allocate the collision state data
            self.cdata.model_num_collisions = wp.zeros(shape=(1,), dtype=int32)
            self.cdata.world_num_collisions = wp.zeros(shape=(num_worlds,), dtype=int32)
            self.cdata.wid = wp.zeros(shape=(self.cmodel.num_model_geom_pairs,), dtype=int32)
            self.cdata.geom_pair = wp.zeros(shape=(self.cmodel.num_model_geom_pairs,), dtype=vec2i)

    def clear(self):
        """
        Clears the active collision count.
        """
        self.cdata.model_num_collisions.zero_()
        self.cdata.world_num_collisions.zero_()

    def zero(self):
        """
        Clears the active collision count.
        """
        self.cdata.model_num_collisions.zero_()
        self.cdata.world_num_collisions.zero_()
        self.cdata.wid.zero_()
        self.cdata.geom_pair.zero_()
