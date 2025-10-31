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
KAMINO: Collision Detection: Brute-force (NxN) broad-phase
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.kamino.core.bv import has_aabb_overlap
from newton._src.solvers.kamino.core.geometry import CollisionGeometriesData, CollisionGeometriesModel
from newton._src.solvers.kamino.core.types import int32, mat83f, vec2i
from newton._src.solvers.kamino.geometry.collisions import CollisionsData, CollisionsModel

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###


@wp.func
def add_active_pair(
    # Collision model input:
    wid_in: int32,
    gid1_in: int32,
    gid2_in: int32,
    sid1_in: int32,
    sid2_in: int32,
    pairid_in: int32,
    model_num_pairs_in: int32,
    world_num_pairs_in: int32,
    # Collision state out:
    model_num_collisions_out: wp.array(dtype=int32),
    world_num_collisions_out: wp.array(dtype=int32),
    collision_wid_out: wp.array(dtype=int32),
    collision_geom_pair_out: wp.array(dtype=vec2i),
):
    model_pairid_out = wp.atomic_add(model_num_collisions_out, 0, 1)
    world_pairid_out = wp.atomic_add(world_num_collisions_out, wid_in, 1)

    # TODO: Check if this is necessary
    if model_pairid_out >= model_num_pairs_in or world_pairid_out >= world_num_pairs_in:
        return

    # Correct the pair id order in order to invoke the correct near-phase function
    if sid1_in > sid2_in:
        pair_out = wp.vec2i(gid2_in, gid1_in)
    else:
        pair_out = wp.vec2i(gid1_in, gid2_in)

    # Store the active collision output data
    collision_wid_out[model_pairid_out] = wid_in
    collision_geom_pair_out[model_pairid_out] = pair_out


###
# Kernels
###


@wp.kernel
def _nxn_broadphase(
    # Inputs:
    geom_sid_in: wp.array(dtype=int32),
    geom_aabb_in: wp.array(dtype=mat83f),
    col_model_num_pairs_in: wp.array(dtype=int32),
    col_world_num_pairs_in: wp.array(dtype=int32),
    col_wid_in: wp.array(dtype=int32),
    col_pairid_in: wp.array(dtype=int32),
    col_geom_pair_in: wp.array(dtype=vec2i),
    # Outputs:
    col_model_num_collisions_out: wp.array(dtype=int32),
    col_world_num_collisions_out: wp.array(dtype=int32),
    col_wid_out: wp.array(dtype=int32),
    col_geom_pair_out: wp.array(dtype=vec2i),
):
    # Retrieve the thread id
    tid = wp.tid()

    # Get the world id
    wid = col_wid_in[tid]

    # Get the geometry ids
    pairid = col_pairid_in[tid]
    geom_pair = col_geom_pair_in[tid]
    gid1 = geom_pair[0]
    gid2 = geom_pair[1]

    # Get the shape ids
    sid1 = geom_sid_in[gid1]
    sid2 = geom_sid_in[gid2]

    # Check for BV overlap and if yes then add to active collision pairs
    if has_aabb_overlap(geom_aabb_in[gid1], geom_aabb_in[gid2]):
        add_active_pair(
            wid,
            gid1,
            gid2,
            sid1,
            sid2,
            pairid,
            col_model_num_pairs_in[0],
            col_world_num_pairs_in[wid],
            col_model_num_collisions_out,
            col_world_num_collisions_out,
            col_wid_out,
            col_geom_pair_out,
        )


###
# Kernel Launcher
###


def nxn_broadphase(
    gmodel: CollisionGeometriesModel, gstate: CollisionGeometriesData, cmodel: CollisionsModel, cdata: CollisionsData
):
    # we need to figure out how to keep the overhead of this small - not launching anything
    # for pair types without collisions, as well as updating the launch dimensions.
    wp.launch(
        _nxn_broadphase,
        dim=cmodel.num_model_geom_pairs,
        inputs=[
            # Inputs:
            gmodel.sid,
            gstate.aabb,
            cmodel.model_num_pairs,
            cmodel.world_num_pairs,
            cmodel.wid,
            cmodel.pairid,
            cmodel.geom_pair,
            # Outputs:
            cdata.model_num_collisions,
            cdata.world_num_collisions,
            cdata.wid,
            cdata.geom_pair,
        ],
    )
