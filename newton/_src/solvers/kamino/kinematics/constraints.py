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
KAMINO: Kinematics: Constraints
"""

from __future__ import annotations

import warp as wp
from warp.context import Devicelike

from ..core.model import Model, ModelData
from ..core.types import int32
from ..geometry.contacts import Contacts
from ..kinematics.limits import Limits

###
# Module interface
###

__all__ = [
    "make_unilateral_constraints_info",
    "max_constraints_per_world",
    "update_constraints_info",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###


def max_constraints_per_world(
    model: Model,
    limits: Limits | None,
    contacts: Contacts | None,
) -> list[int]:
    """
    Returns the maximum number of constraints for each world in the model.

    Args:
        model (Model): The model for which to compute the maximum constraints.
        limits (Limits, optional): The container holding the allocated joint-limit data.
        contacts (Contacts, optional): The container holding the allocated contacts data.

    Returns:
        List[int]: A list of the maximum constraints for each world in the model.
    """
    nw = model.info.num_worlds
    njc = [model.worlds[i].num_joint_cts for i in range(nw)]
    maxnl = limits.num_world_max_limits if limits else [0] * nw
    maxnc = contacts.num_world_max_contacts if contacts else [0] * nw
    maxncts = [njc[i] + maxnl[i] + 3 * maxnc[i] for i in range(nw)]
    return maxncts


def make_unilateral_constraints_info(
    model: Model,
    data: ModelData,
    limits: Limits | None = None,
    contacts: Contacts | None = None,
    device: Devicelike = None,
):
    """
    Constructs constraints entries in the ModelInfo member of a model.

    Args:
        model (Model): The model container holding time-invariant data.
        data (ModelData): The solver container holding time-varying data.
        limits (Limits, optional): The limits container holding the joint-limit data.
        contacts (Contacts, optional): The contacts container holding the contact data.
    """

    # Ensure the model is valid
    if not isinstance(model, Model):
        raise TypeError("`model` must be an instance of `Model`")

    # Ensure the data is valid
    if not isinstance(data, ModelData):
        raise TypeError("`data` must be an instance of `ModelData`")

    # Device is not specified, use the model's device
    if device is None:
        device = model.device

    # Retrieve the number of worlds in the model
    num_worlds = model.info.num_worlds

    # Declare the lists of per-world maximum limits and contacts
    # NOTE: These will either be captured by reference from the limits and contacts
    # containers or initialized to zero if no limits or contacts are provided.
    world_maxnl: list[int] = []
    world_maxnc: list[int] = []

    # If a limits container is provided, ensure it is valid
    # and then assign the entity counters to the model info.
    if limits is not None:
        if not isinstance(limits, Limits):
            raise TypeError("`limits` must be an instance of `Limits`")
        world_maxnl = limits.num_world_max_limits
        model.size.sum_of_max_limits = limits.num_model_max_limits
        model.size.max_of_max_limits = max(limits.num_world_max_limits)
        model.info.max_limits = limits.world_max_limits
        data.info.num_limits = limits.world_num_limits
    else:
        world_maxnl = [0] * num_worlds
        model.size.sum_of_max_limits = 0
        model.size.max_of_max_limits = 0
        with wp.ScopedDevice(device):
            model.info.max_limits = wp.zeros(shape=(num_worlds,), dtype=int32)
            data.info.num_limits = wp.zeros(shape=(num_worlds,), dtype=int32)

    # If a contacts container is provided, ensure it is valid
    # and then assign the entity counters to the model info.
    if contacts is not None:
        if not isinstance(contacts, Contacts):
            raise TypeError("`contacts` must be an instance of `Contacts`")
        world_maxnc = contacts.num_world_max_contacts
        model.size.sum_of_max_contacts = contacts.num_model_max_contacts
        model.size.max_of_max_contacts = max(contacts.num_world_max_contacts)
        model.info.max_contacts = contacts.world_max_contacts
        data.info.num_contacts = contacts.world_num_contacts
    else:
        world_maxnc = [0] * num_worlds
        model.size.sum_of_max_contacts = 0
        model.size.max_of_max_contacts = 0
        with wp.ScopedDevice(device):
            model.info.max_contacts = wp.zeros(shape=(num_worlds,), dtype=int32)
            data.info.num_contacts = wp.zeros(shape=(num_worlds,), dtype=int32)

    # Compute the maximum number of unilateral entities (limits and contacts) per world
    world_max_unilaterals: list[int] = [nl + nc for nl, nc in zip(world_maxnl, world_maxnc, strict=False)]
    model.size.sum_of_max_unilaterals = sum(world_max_unilaterals)
    model.size.max_of_max_unilaterals = max(world_max_unilaterals)

    # Compute the maximum number of constraints per world: limits, contacts, and total
    world_maxnlc: list[int] = list(world_maxnl)
    world_maxncc: list[int] = [3 * maxnc for maxnc in world_maxnc]
    world_njc = [world.num_joint_cts for world in model.worlds]
    world_maxncts = [
        maxnl + maxnc + njc for maxnl, maxnc, njc in zip(world_maxnlc, world_maxncc, world_njc, strict=False)
    ]
    model.size.sum_of_max_total_cts = sum(world_maxncts)
    model.size.max_of_max_total_cts = max(world_maxncts)

    # Compute the entity index offsets for limits, contacts and unilaterals
    # NOTE: unilaterals is simply the concatenation of limits and contacts
    world_lio = [0] + [sum(world_maxnl[:i]) for i in range(1, num_worlds + 1)]
    world_cio = [0] + [sum(world_maxnc[:i]) for i in range(1, num_worlds + 1)]
    world_uio = [0] + [sum(world_maxnl[:i]) + sum(world_maxnc[:i]) for i in range(1, num_worlds + 1)]

    # Compute the entity index offsets for limits, contacts and unilaterals
    # NOTE: unilaterals is simply the concatenation of limits and contacts
    world_lcio = [0] + [sum(world_maxnlc[:i]) for i in range(1, num_worlds + 1)]
    world_ccio = [0] + [sum(world_maxncc[:i]) for i in range(1, num_worlds + 1)]
    world_ucio = [0] + [sum(world_maxnlc[:i]) + sum(world_maxncc[:i]) for i in range(1, num_worlds + 1)]
    world_ctsio = [0] + [
        sum(world_njc[:i]) + sum(world_maxnlc[:i]) + sum(world_maxncc[:i]) for i in range(1, num_worlds + 1)
    ]

    # Allocate all constraint info arrays on the target device
    with wp.ScopedDevice(device):
        # Allocate the total constraint arrays
        model.info.max_total_cts = wp.array(world_maxncts, dtype=int32)
        model.info.total_cts_offset = wp.array(world_ctsio[:num_worlds], dtype=int32)

        # Allocate the limit constraint arrays
        model.info.max_limit_cts = wp.array(world_maxnlc, dtype=int32)
        model.info.limits_offset = wp.array(world_lio[:num_worlds], dtype=int32)
        model.info.limit_cts_offset = wp.array(world_lcio[:num_worlds], dtype=int32)

        # Allocate the contact constraint arrays
        model.info.max_contact_cts = wp.array(world_maxncc, dtype=int32)
        model.info.contacts_offset = wp.array(world_cio[:num_worlds], dtype=int32)
        model.info.unilaterals_offset = wp.array(world_uio[:num_worlds], dtype=int32)

        # Allocate the unilateral constraint arrays
        model.info.contact_cts_offset = wp.array(world_ccio[:num_worlds], dtype=int32)
        model.info.unilateral_cts_offset = wp.array(world_ucio[:num_worlds], dtype=int32)

        # Initialize the active constraint counters to zero
        data.info.num_total_cts = wp.zeros(shape=(num_worlds,), dtype=int32)
        data.info.num_limit_cts = wp.zeros(shape=(num_worlds,), dtype=int32)
        data.info.limit_cts_group_offset = wp.zeros(shape=(num_worlds,), dtype=int32)
        data.info.num_contact_cts = wp.zeros(shape=(num_worlds,), dtype=int32)
        data.info.contact_cts_group_offset = wp.zeros(shape=(num_worlds,), dtype=int32)


###
# Kernels
###


@wp.kernel
def _update_constraints_info(
    # Inputs:
    model_info_num_joint_cts: wp.array(dtype=int32),
    model_info_num_limits: wp.array(dtype=int32),
    model_info_num_contacts: wp.array(dtype=int32),
    # Outputs:
    data_info_num_total_cts: wp.array(dtype=int32),
    data_info_num_limit_cts: wp.array(dtype=int32),
    data_info_num_contact_cts: wp.array(dtype=int32),
    data_info_limit_cts_group_offset: wp.array(dtype=int32),
    data_info_contact_cts_group_offset: wp.array(dtype=int32),
):
    # Retrieve the thread index as the world index
    wid = wp.tid()

    # Retrieve the number of joint constraints for this world
    njc = model_info_num_joint_cts[wid]

    # Retrieve the number of unilaterals for this world
    nl = model_info_num_limits[wid]
    nc = model_info_num_contacts[wid]

    # Set the number of active constraints for each group and the total
    nlc = nl  # NOTE: Each limit currently introduces only a single constraint
    ncc = 3 * nc
    ncts = njc + nlc + ncc

    # Set the constraint group offsets, i.e. the starting index
    # of each group within the block allocated for each world
    lcgo = njc
    ccgo = njc + nlc

    # Store the state info for this world
    data_info_num_total_cts[wid] = ncts
    data_info_num_limit_cts[wid] = nlc
    data_info_num_contact_cts[wid] = ncc
    data_info_limit_cts_group_offset[wid] = lcgo
    data_info_contact_cts_group_offset[wid] = ccgo


###
# Launchers
###


def update_constraints_info(
    model: Model,
    data: ModelData,
):
    """
    Updates the active constraints info for the given model and current data.
    """
    wp.launch(
        _update_constraints_info,
        dim=model.info.num_worlds,
        inputs=[
            # Inputs:
            model.info.num_joint_cts,
            data.info.num_limits,
            data.info.num_contacts,
            # Outputs:
            data.info.num_total_cts,
            data.info.num_limit_cts,
            data.info.num_contact_cts,
            data.info.limit_cts_group_offset,
            data.info.contact_cts_group_offset,
        ],
    )
