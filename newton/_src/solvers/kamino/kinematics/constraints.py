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
Provides mechanisms to define and manage constraints and their associated input/output data.
"""

import warp as wp
from warp.context import Devicelike

from ..core.model import Model, ModelData
from ..core.types import float32, int32, vec3f
from ..geometry.contacts import ContactMode, Contacts
from ..kinematics.limits import Limits

###
# Module interface
###

__all__ = [
    "make_unilateral_constraints_info",
    "max_constraints_per_world",
    "unpack_constraint_solutions",
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
    # Ensure the model container is valid
    if model is None:
        raise ValueError("`model` is required but got `None`.")
    else:
        if not isinstance(model, Model):
            raise TypeError(f"`model` is required to be of type `Model` but got {type(model)}.")

    # Ensure the limits container is valid
    if limits is not None:
        if not isinstance(limits, Limits):
            raise TypeError(f"`limits` is required to be of type `Limits` but got {type(limits)}.")

    # Ensure the contacts container is valid
    if contacts is not None:
        if not isinstance(contacts, Contacts):
            raise TypeError(f"`contacts` is required to be of type `Contacts` but got {type(contacts)}.")

    # Compute the maximum number of constraints per world
    nw = model.info.num_worlds
    njc = [model.worlds[i].num_joint_cts for i in range(nw)]
    maxnl = limits.world_max_limits_host if limits and limits.model_max_limits_host > 0 else [0] * nw
    maxnc = contacts.world_max_contacts_host if contacts and contacts.model_max_contacts_host > 0 else [0] * nw
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
        device (Devicelike, optional): The device on which to allocate the constraint info arrays.\n
            If None, the model's device will be used.
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
    num_worlds = model.size.num_worlds

    # Declare the lists of per-world maximum limits and contacts
    # NOTE: These will either be captured by reference from the limits and contacts
    # containers or initialized to zero if no limits or contacts are provided.
    world_maxnl: list[int] = []
    world_maxnc: list[int] = []

    ###
    #  Helper functions
    ###

    def _assign_model_limits_info():
        nonlocal world_maxnl
        world_maxnl = limits.world_max_limits_host
        model.size.sum_of_max_limits = limits.model_max_limits_host
        model.size.max_of_max_limits = max(limits.world_max_limits_host)
        model.info.max_limits = limits.world_max_limits
        data.info.num_limits = limits.world_active_limits

    def _make_empty_model_limits_info():
        nonlocal world_maxnl
        world_maxnl = [0] * num_worlds
        model.size.sum_of_max_limits = 0
        model.size.max_of_max_limits = 0
        with wp.ScopedDevice(device):
            model.info.max_limits = wp.zeros(shape=(num_worlds,), dtype=int32)
            data.info.num_limits = wp.zeros(shape=(num_worlds,), dtype=int32)

    def _assign_model_contacts_info():
        nonlocal world_maxnc
        world_maxnc = contacts.world_max_contacts_host
        model.size.sum_of_max_contacts = contacts.model_max_contacts_host
        model.size.max_of_max_contacts = max(contacts.world_max_contacts_host)
        model.info.max_contacts = contacts.world_max_contacts
        data.info.num_contacts = contacts.world_active_contacts

    def _make_empty_model_contacts_info():
        nonlocal world_maxnc
        world_maxnc = [0] * num_worlds
        model.size.sum_of_max_contacts = 0
        model.size.max_of_max_contacts = 0
        with wp.ScopedDevice(device):
            model.info.max_contacts = wp.zeros(shape=(num_worlds,), dtype=int32)
            data.info.num_contacts = wp.zeros(shape=(num_worlds,), dtype=int32)

    # If a limits container is provided, ensure it is valid
    # and then assign the entity counters to the model info.
    if limits is not None:
        if not isinstance(limits, Limits):
            raise TypeError("`limits` must be an instance of `Limits`")
        if limits.data is not None and limits.model_max_limits_host > 0:
            _assign_model_limits_info()
        else:
            _make_empty_model_limits_info()
    else:
        _make_empty_model_limits_info()

    # If a contacts container is provided, ensure it is valid
    # and then assign the entity counters to the model info.
    if contacts is not None:
        if not isinstance(contacts, Contacts):
            raise TypeError("`contacts` must be an instance of `Contacts`")
        if contacts.data is not None and contacts.model_max_contacts_host > 0:
            _assign_model_contacts_info()
    else:
        _make_empty_model_contacts_info()

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
    data_info_num_limits: wp.array(dtype=int32),
    data_info_num_contacts: wp.array(dtype=int32),
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
    nl = data_info_num_limits[wid]
    nc = data_info_num_contacts[wid]

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


@wp.kernel
def _unpack_joint_constraint_solutions(
    # Inputs:
    model_info_total_cts_offset: wp.array(dtype=int32),
    model_info_joint_cts_offset: wp.array(dtype=int32),
    model_time_inv_dt: wp.array(dtype=float32),
    joint_wid: wp.array(dtype=int32),
    joint_num_cts: wp.array(dtype=int32),
    joint_cts_offset: wp.array(dtype=int32),
    lambdas: wp.array(dtype=float32),
    # Outputs:
    joint_lambda_j: wp.array(dtype=float32),
):
    # Retrieve the thread index as the joint index
    jid = wp.tid()

    # Retrieve the joint-specific model info
    wid = joint_wid[jid]
    num_cts = joint_num_cts[jid]
    cts_offset = joint_cts_offset[jid]

    # Retrieve the world-specific info
    inv_dt = model_time_inv_dt[wid]
    world_total_cts_offset = model_info_total_cts_offset[wid]
    world_joints_cts_offset = model_info_joint_cts_offset[wid]

    # Compute block offsets of the joint's constraints within
    # the joint-only constraints and total constraints arrays
    vio_j = world_total_cts_offset + cts_offset
    jio_j = world_joints_cts_offset + cts_offset

    # Compute and store the joint-constraint reaction forces
    for j in range(num_cts):
        joint_lambda_j[jio_j + j] = inv_dt * lambdas[vio_j + j]


@wp.kernel
def _unpack_limit_constraint_solutions(
    # Inputs:
    model_time_inv_dt: wp.array(dtype=float32),
    model_info_total_cts_offset: wp.array(dtype=int32),
    data_info_limit_cts_group_offset: wp.array(dtype=int32),
    limit_model_num_limits: wp.array(dtype=int32),
    limit_wid: wp.array(dtype=int32),
    limit_lid: wp.array(dtype=int32),
    lambdas: wp.array(dtype=float32),
    v_plus: wp.array(dtype=float32),
    # Outputs:
    limit_reaction: wp.array(dtype=float32),
    limit_velocity: wp.array(dtype=float32),
):
    # Retrieve the thread index as the contact index
    lid = wp.tid()

    # Retrieve the number of limits active in the model
    model_nl = limit_model_num_limits[0]

    # Skip if lid is greater than the number of limits active in the model
    if lid >= model_nl:
        return

    # Retrieve the world index and the world-relative limit index for this limit
    wid = limit_wid[lid]
    lid_l = limit_lid[lid]

    # Retrieve the world-specific info
    inv_dt = model_time_inv_dt[wid]
    total_cts_offset = model_info_total_cts_offset[wid]
    limit_cts_offset = data_info_limit_cts_group_offset[wid]

    # Compute the global constraint index for this limit
    limit_cts_idx = total_cts_offset + limit_cts_offset + lid_l

    # Load the limit reaction and velocity from the global constraint arrays
    lambda_l = lambdas[limit_cts_idx]
    v_plus_l = v_plus[limit_cts_idx]

    # Scale the contact reaction by the time step to convert from lagrange impulse to force
    lambda_l = inv_dt * lambda_l

    # Store the computed limit state
    limit_reaction[lid] = lambda_l
    limit_velocity[lid] = v_plus_l


@wp.kernel
def _unpack_contact_constraint_solutions(
    # Inputs:
    model_time_inv_dt: wp.array(dtype=float32),
    model_info_total_cts_offset: wp.array(dtype=int32),
    data_info_contact_cts_group_offset: wp.array(dtype=int32),
    contact_model_num_contacts: wp.array(dtype=int32),
    contact_wid: wp.array(dtype=int32),
    contact_cid: wp.array(dtype=int32),
    lambdas: wp.array(dtype=float32),
    v_plus: wp.array(dtype=float32),
    # Outputs:
    contact_mode: wp.array(dtype=int32),
    contact_reaction: wp.array(dtype=vec3f),
    contact_velocity: wp.array(dtype=vec3f),
):
    # Retrieve the thread index as the contact index
    cid = wp.tid()

    # Retrieve the number of contacts active in the model
    model_nc = contact_model_num_contacts[0]

    # Skip if cid is greater than the number of contacts active in the model
    if cid >= model_nc:
        return

    # Retrieve the world index and the world-relative contact index for this contact
    wid = contact_wid[cid]
    cid_k = contact_cid[cid]

    # Retrieve the world-specific info
    inv_dt = model_time_inv_dt[wid]
    total_cts_offset = model_info_total_cts_offset[wid]
    contact_cts_offset = data_info_contact_cts_group_offset[wid]

    # Compute block offsets of the contact constraints within
    # the contact-only constraints and total constraints arrays
    contact_cts_start = total_cts_offset + contact_cts_offset + 3 * cid_k

    # Load the contact reaction and velocity from the global constraint arrays
    lambda_k = vec3f(0.0)
    v_plus_k = vec3f(0.0)
    for k in range(3):
        lambda_k[k] = lambdas[contact_cts_start + k]
        v_plus_k[k] = v_plus[contact_cts_start + k]

    # Scale the contact reaction by the time step to convert from lagrange impulse to force
    lambda_k = inv_dt * lambda_k

    # Compute the discrete contact mode based on the reaction magnitude and velocity
    mode_k = wp.static(ContactMode.make_compute_mode_func())(v_plus_k)

    # Store the computed contact state
    contact_mode[cid] = mode_k
    contact_reaction[cid] = lambda_k
    contact_velocity[cid] = v_plus_k


###
# Launchers
###


def update_constraints_info(
    model: Model,
    data: ModelData,
):
    """
    Updates the active constraints info for the given model and current data.

    Args:
        model (Model): The model container holding time-invariant data.
        data (ModelData): The solver container holding time-varying data.
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


def unpack_constraint_solutions(
    lambdas: wp.array,
    v_plus: wp.array,
    model: Model,
    data: ModelData,
    limits: Limits | None = None,
    contacts: Contacts | None = None,
):
    """
    Unpacks the constraint reactions and velocities into respective data containers.

    Args:
        lambdas (wp.array): The array of constraint reactions (i.e. lagrange multipliers).
        v_plus (wp.array): The array of post-event constraint velocities.
        data (ModelData): The solver container holding time-varying data.
        limits (Limits, optional): The limits container holding the joint-limit data.\n
            If None, limits will be skipped.
        contacts (Contacts, optional): The contacts container holding the contact data.\n
            If None, contacts will be skipped.
    """
    # Unpack joint constraint multipliers if the model has joints
    if model.size.sum_of_num_joints > 0:
        wp.launch(
            kernel=_unpack_joint_constraint_solutions,
            dim=model.size.sum_of_num_joints,
            inputs=[
                # Inputs:
                model.info.total_cts_offset,
                model.info.joint_cts_offset,
                model.time.inv_dt,
                model.joints.wid,
                model.joints.num_cts,
                model.joints.cts_offset,
                lambdas,
                # Outputs:
                data.joints.lambda_j,
            ],
        )

    # Unpack limit constraint multipliers if a limits container is provided
    if limits is not None:
        wp.launch(
            kernel=_unpack_limit_constraint_solutions,
            dim=limits.model_max_limits_host,
            inputs=[
                # Inputs:
                model.time.inv_dt,
                model.info.total_cts_offset,
                data.info.limit_cts_group_offset,
                limits.model_active_limits,
                limits.wid,
                limits.lid,
                lambdas,
                v_plus,
                # Outputs:
                limits.reaction,
                limits.velocity,
            ],
        )

    # Unpack contact constraint multipliers if a contacts container is provided
    if contacts is not None:
        wp.launch(
            kernel=_unpack_contact_constraint_solutions,
            dim=contacts.model_max_contacts_host,
            inputs=[
                # Inputs:
                model.time.inv_dt,
                model.info.total_cts_offset,
                data.info.contact_cts_group_offset,
                contacts.model_active_contacts,
                contacts.wid,
                contacts.cid,
                lambdas,
                v_plus,
                # Outputs:
                contacts.mode,
                contacts.reaction,
                contacts.velocity,
            ],
        )
