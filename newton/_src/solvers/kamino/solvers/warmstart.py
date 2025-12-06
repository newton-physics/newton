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
Provides mechanisms to warm-start constraints.

This module provides warm-starting mechanisms for unilateral limit and contact constraints,
in the form of the `WarmstarterLimits` and `WarmstarterContacts` classes, respectively.
These classes utilize cached constraint data from previous simulation steps to initialize
the current step's constraints, improving solver convergence.

The warm-starting process involves matching current constraints to cached ones using unique keys
(e.g., joint-DoF index pairs for limits and geom-pair keys for contacts) and, for contacts,
also considering contact point positions to ensure accurate matching.

For contacts, if a direct match based on position is not found, optional fallback mechanisms using
the net force/wrench on the associated body CoMs are employed to estimate the warm-started reaction.

See the :class:`WarmstarterLimits` and :class:`WarmstarterContacts` classes for detailed usage.
"""

from enum import IntEnum

import warp as wp
from warp.context import Devicelike

from ..core.math import contact_wrench_matrix_from_points
from ..core.model import Model, ModelData
from ..core.types import float32, int32, override, quatf, transformf, uint64, vec2f, vec2i, vec3f, vec6f
from ..geometry.contacts import Contacts, ContactsData
from ..geometry.keying import KeySorter, binary_search_find_range_start
from ..kinematics.limits import Limits, LimitsData
from ..solvers.padmm.math import project_to_coulomb_cone

###
# Module interface
###

__all__ = [
    "WarmstarterContacts",
    "WarmstarterLimits",
    "warmstart_contacts_by_matched_geom_pair_key_and_position",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


@wp.kernel
def _warmstart_limits_by_matched_jid_dof_key(
    # Inputs - Previous:
    sorted_limit_keys_old: wp.array(dtype=uint64),
    sorted_to_unsorted_map_old: wp.array(dtype=int32),
    num_active_limits_old: wp.array(dtype=int32),
    limit_velocity_old: wp.array(dtype=float32),
    limit_reaction_old: wp.array(dtype=float32),
    # Inputs - Next:
    num_active_limits_new: wp.array(dtype=int32),
    limit_key_new: wp.array(dtype=uint64),
    # Outputs:
    limit_reaction_new: wp.array(dtype=float32),
    limit_velocity_new: wp.array(dtype=float32),
):
    """
    Match current limits to previous timestep limits using joint-DoF index pair keys.

    For each current limit, finds a matching limit from the
    previous step using a binary-search on sorted keys (O(log n))
    """
    # Retrieve the limit index as the thread index
    lid = wp.tid()

    # Perform early exit if out of active bounds
    if lid >= num_active_limits_new[0]:
        return

    # Retrieve number of active old limits and the target key to search for
    num_active_old = num_active_limits_old[0]
    target_key = limit_key_new[lid]

    # Perform binary search to find the start index of the
    # target key - i.e. assuming a joint-DoF index pair key
    start = binary_search_find_range_start(0, num_active_old, target_key, sorted_limit_keys_old)

    # If key not found, then mark as a new limit and skip further processing
    # NOTE: This means that a new limit has become active
    if start == -1:
        limit_reaction_new[lid] = 0.0
        limit_velocity_new[lid] = 0.0
    else:
        # Retrieve the old limit index from the sorted->unsorted map
        lid_old = sorted_to_unsorted_map_old[start]
        reaction_old = limit_reaction_old[lid_old]
        velocity_old = limit_velocity_old[lid_old]

        # Retrieve the matched limit reaction and velocity from
        # the old limit and store them to the new limit
        limit_reaction_new[lid] = wp.max(reaction_old, 0.0)
        limit_velocity_new[lid] = wp.max(velocity_old, 0.0)


@wp.kernel
def _warmstart_contacts_by_matched_geom_pair_key_and_position(
    # Inputs - Common:
    tolerance: float32,
    time_dt: wp.array(dtype=float32),
    body_q_i: wp.array(dtype=transformf),
    body_u_i: wp.array(dtype=vec6f),
    # Inputs - Previous:
    sorted_contact_keys_old: wp.array(dtype=uint64),
    sorted_to_unsorted_map_old: wp.array(dtype=int32),
    num_active_contacts_old: wp.array(dtype=int32),
    contact_position_B_old: wp.array(dtype=vec3f),
    contact_frame_old: wp.array(dtype=quatf),
    contact_reaction_old: wp.array(dtype=vec3f),
    contact_velocity_old: wp.array(dtype=vec3f),
    # Inputs - Next:
    num_active_contacts_new: wp.array(dtype=int32),
    contact_key_new: wp.array(dtype=uint64),
    contact_wid_new: wp.array(dtype=int32),
    contact_bid_AB_new: wp.array(dtype=vec2i),
    contact_position_B_new: wp.array(dtype=vec3f),
    contact_frame_new: wp.array(dtype=quatf),
    # Outputs:
    contact_reaction_new: wp.array(dtype=vec3f),
    contact_velocity_new: wp.array(dtype=vec3f),
):
    """
    Match current contacts to previous timestep contacts using geom-pair keys and relative distance.

    For each current contact, finds matching contact from previous step by:
    1. Binary search on sorted keys (O(log n))
    2. Linear scan through matching keys to find matching contact point positions (O(m))
    """
    # Retrieve the contact index as the thread index
    cid = wp.tid()

    # Perform early exit if out of active bounds
    if cid >= num_active_contacts_new[0]:
        return

    # Retrieve number of active old contacts and the target key to search for
    num_active_old = num_active_contacts_old[0]
    target_key = contact_key_new[cid]

    # Initialize the target reaction and velocity to zero
    # to account for the case where no matching contact is found
    target_reaction = vec3f(0.0)
    target_velocity = vec3f(0.0)

    # Perform binary search to find the start index of the target key - i.e. assuming a geom-pair key
    start = binary_search_find_range_start(0, num_active_old, target_key, sorted_contact_keys_old)

    # If key not found, then mark as a new contact and skip further processing
    # NOTE: This means that a new geom-pair collision has occurred
    if start == -1:
        contact_reaction_new[cid] = target_reaction
        contact_velocity_new[cid] = target_velocity
        return

    # Retrieve the new contact position on the corresponding geom B
    # NOTE: We only need to match based on one contact point position
    # on body/geom B in order to handle the general case of static bodies
    # as geom B is by definition always the non-static body in a contact pair
    r_B_target = contact_position_B_new[cid]
    R_k_target = wp.quat_to_matrix(contact_frame_new[cid])

    # Retrieve the timestep delta time for the contact's associated body
    dt = time_dt[contact_wid_new[cid]]

    # Retrieve the body indices and states for the contact's associated bodies
    bid_AB = contact_bid_AB_new[cid]
    r_B = wp.transform_get_translation(body_q_i[bid_AB[1]])
    u_B = body_u_i[bid_AB[1]]

    # Iterate through all old contacts with the same key and check if contacts match
    # based on distance of contact points after accounting for associated body motion
    # NOTE: For the comparison, new_idx -> cid, old_idx -> sorted_to_unsorted_map_old[start + k]
    k = int32(0)
    old_key = sorted_contact_keys_old[start]
    while target_key == old_key:
        # Retrieve the old contact index from the sorted->unsorted map
        cid_old = sorted_to_unsorted_map_old[start + k]
        r_k_B_old = contact_position_B_old[cid_old]
        W_Bk_T = wp.transpose(contact_wrench_matrix_from_points(r_k_B_old, r_B))
        r_B_candidate = r_k_B_old + dt * (W_Bk_T @ u_B)

        # Compute and check the distance to the target contact positions
        dr_B = wp.length(r_B_candidate - r_B_target)
        if dr_B < tolerance:
            # When a match is found, retrieve the contact reaction and velocity
            # from the old contact and transform them to the new contact frame
            q_k_old = contact_frame_old[cid_old]
            lambda_k_old = contact_reaction_old[cid_old]
            v_k_old = contact_velocity_old[cid_old]
            R_k_old = wp.quat_to_matrix(q_k_old)
            R_k_old_to_new = wp.transpose(R_k_target) @ R_k_old
            target_reaction = R_k_old_to_new @ lambda_k_old
            target_velocity = R_k_old_to_new @ v_k_old
            break

        # Update the current old-key to check in the next iteration
        k += 1
        old_key = sorted_contact_keys_old[start + k]

    # Store the new contact reaction and velocity
    # NOTE: These will remain zero if no matching contact is found
    contact_reaction_new[cid] = target_reaction
    contact_velocity_new[cid] = target_velocity


@wp.kernel
def _warmstart_contacts_from_geom_pair_net_force(
    # Inputs - Common:
    scaling: float32,
    body_q_i: wp.array(dtype=transformf),
    body_u_i: wp.array(dtype=vec6f),
    # Inputs - Previous:
    sorted_contact_keys_old: wp.array(dtype=uint64),
    sorted_to_unsorted_map_old: wp.array(dtype=int32),
    num_active_contacts_old: wp.array(dtype=int32),
    contact_frame_old: wp.array(dtype=quatf),
    contact_reaction_old: wp.array(dtype=vec3f),
    # Inputs - Next:
    num_active_contacts_new: wp.array(dtype=int32),
    contact_key_new: wp.array(dtype=uint64),
    contact_bid_AB_new: wp.array(dtype=vec2i),
    contact_position_A_new: wp.array(dtype=vec3f),
    contact_position_B_new: wp.array(dtype=vec3f),
    contact_frame_new: wp.array(dtype=quatf),
    contact_material_new: wp.array(dtype=vec2f),
    # Outputs:
    contact_reaction_new: wp.array(dtype=vec3f),
    contact_velocity_new: wp.array(dtype=vec3f),
):
    """
    Match current contacts to previous timestep contacts using geom-pair keys and relative distance.

    For each current contact, finds matching contact from previous step by:
    1. Binary search on sorted keys (O(log n))
    2. Linear scan through matching keys to find matching contact point positions (O(m))
    """
    # Retrieve the contact index as the thread index
    cid = wp.tid()

    # Perform early exit if out of active bounds
    if cid >= num_active_contacts_new[0]:
        return

    # Retrieve number of active old contacts and the target key to search for
    num_active_old = num_active_contacts_old[0]
    target_key = contact_key_new[cid]

    # Initialize the target reaction and velocity to zero
    # to account for the case where no matching contact is found
    target_reaction = vec3f(0.0)
    target_velocity = vec3f(0.0)

    # Perform binary search to find the start index of the target key - i.e. assuming a geom-pair key
    start = binary_search_find_range_start(0, num_active_old, target_key, sorted_contact_keys_old)

    # If key not found, then mark as a new contact and skip further processing
    # NOTE: This means that a new geom-pair collision has occurred
    if start == -1:
        contact_reaction_new[cid] = target_reaction
        contact_velocity_new[cid] = target_velocity
        return

    # Retrieve the friction coefficient for the contact's associated material
    target_material = contact_material_new[cid]
    target_mu = target_material[0]

    # Retrieve the A/B positions on the corresponding geom-pair contact
    r_A_target = contact_position_A_new[cid]
    r_B_target = contact_position_B_new[cid]
    R_k_target = wp.quat_to_matrix(contact_frame_new[cid])

    # Retrieve the body indices and states for the contact's associated bodies
    # NOTE: If body A is static, then its velocity contribution is zero by definition
    bid_AB = contact_bid_AB_new[cid]
    v_Ak = vec3f(0.0)
    if bid_AB[0] >= 0:
        u_A = body_u_i[bid_AB[0]]
        r_A = wp.transform_get_translation(body_q_i[bid_AB[0]])
        W_Ak_T = wp.transpose(contact_wrench_matrix_from_points(r_A_target, r_A))
        v_Ak = W_Ak_T @ u_A
    u_B = body_u_i[bid_AB[1]]
    r_B = wp.transform_get_translation(body_q_i[bid_AB[1]])
    W_Bk_T = wp.transpose(contact_wrench_matrix_from_points(r_B_target, r_B))
    v_Bk = W_Bk_T @ u_B

    # Compute the new contact velocity based on the measured body motion
    # project to the contact frame and set normal component to zero
    # NOTE: We only need to consider tangential velocity for warm-starting
    # as the normal velocity should always be non-negative in the local
    # contact frame, and positive if the solver computes an opening contact
    # thus, for warm-starting we only need to consider the tangential velocity
    target_velocity = scaling * wp.transpose(R_k_target) @ (v_Bk - v_Ak)
    target_velocity.z = wp.max(target_velocity.z, 0.0)

    # Iterate through all old contacts with the same key and accumulate net body-com wrench
    # NOTE: We only need body B since it is the body on which the contact reaction acts positively
    geom_pair_force_body_B = vec3f(0.0)
    k = int32(0)
    old_key = sorted_contact_keys_old[start]
    while target_key == old_key:
        # Retrieve the old contact index from the sorted->unsorted map
        cid_old = sorted_to_unsorted_map_old[start + k]

        # Load old contact data for the old contact
        q_k_old = contact_frame_old[cid_old]
        lambda_k_old = contact_reaction_old[cid_old]

        # Accumulate the old contact's contribution to the geom-pair net force on body B
        geom_pair_force_body_B += wp.quat_to_matrix(q_k_old) @ lambda_k_old

        # Update the current old-key to check in the next iteration
        k += 1
        old_key = sorted_contact_keys_old[start + k]

    # TODO: We need to cache this value per geom-pair
    # TODO: Replace this with a new cache instead of recomputing every time --- IGNORE ---
    num_contacts_gid_AB_new = int32(0)
    for i in range(num_active_contacts_new[0]):
        if contact_key_new[i] == target_key:
            num_contacts_gid_AB_new += 1

    # Average the net body-com force over the number of contacts for this geom-pair
    contact_force_uniform_new = (1.0 / float32(num_contacts_gid_AB_new)) * geom_pair_force_body_B

    # Project to the new contact frame and local
    # friction cone to obtain the contact reaction
    target_reaction = wp.transpose(R_k_target) @ contact_force_uniform_new
    target_reaction = scaling * project_to_coulomb_cone(target_reaction, target_mu)

    # Store the new contact reaction and velocity
    # NOTE: These will remain zero if no matching contact is found
    contact_reaction_new[cid] = target_reaction
    contact_velocity_new[cid] = target_velocity


@wp.kernel
def _warmstart_contacts_by_matched_geom_pair_key_and_position_with_net_force_backup(
    # Inputs - Common:
    tolerance: float32,
    scaling: float32,
    time_dt: wp.array(dtype=float32),
    body_q_i: wp.array(dtype=transformf),
    body_u_i: wp.array(dtype=vec6f),
    # Inputs - Previous:
    sorted_contact_keys_old: wp.array(dtype=uint64),
    sorted_to_unsorted_map_old: wp.array(dtype=int32),
    num_active_contacts_old: wp.array(dtype=int32),
    contact_position_B_old: wp.array(dtype=vec3f),
    contact_frame_old: wp.array(dtype=quatf),
    contact_reaction_old: wp.array(dtype=vec3f),
    contact_velocity_old: wp.array(dtype=vec3f),
    # Inputs - Next:
    num_active_contacts_new: wp.array(dtype=int32),
    contact_key_new: wp.array(dtype=uint64),
    contact_wid_new: wp.array(dtype=int32),
    contact_bid_AB_new: wp.array(dtype=vec2i),
    contact_position_A_new: wp.array(dtype=vec3f),
    contact_position_B_new: wp.array(dtype=vec3f),
    contact_frame_new: wp.array(dtype=quatf),
    contact_material_new: wp.array(dtype=vec2f),
    # Outputs:
    contact_reaction_new: wp.array(dtype=vec3f),
    contact_velocity_new: wp.array(dtype=vec3f),
):
    """
    Match current contacts to previous timestep contacts using geom-pair keys and relative distance.

    For each current contact, finds matching contact from previous step by:
    1. Binary search on sorted keys (O(log n))
    2. Linear scan through matching keys to find matching contact point positions (O(m))
    """
    # Retrieve the contact index as the thread index
    cid = wp.tid()

    # Perform early exit if out of active bounds
    if cid >= num_active_contacts_new[0]:
        return

    # Retrieve number of active old contacts and the target key to search for
    num_active_old = num_active_contacts_old[0]
    target_key = contact_key_new[cid]

    # Initialize the target reaction and velocity to zero
    # to account for the case where no matching contact is found
    target_reaction = vec3f(0.0)
    target_velocity = vec3f(0.0)

    # Perform binary search to find the start index of the target key - i.e. assuming a geom-pair key
    start = binary_search_find_range_start(0, num_active_old, target_key, sorted_contact_keys_old)

    # If key not found, then mark as a new contact and skip further processing
    # NOTE: This means that a new geom-pair collision has occurred
    if start == -1:
        contact_reaction_new[cid] = target_reaction
        contact_velocity_new[cid] = target_velocity
        return

    # Retrieve the new contact position on the corresponding geom B
    # NOTE: We only need to match based on one contact point position
    # on body/geom B in order to handle the general case of static bodies
    # as geom B is by definition always the non-static body in a contact pair
    r_Bk_target = contact_position_B_new[cid]
    R_k_target = wp.quat_to_matrix(contact_frame_new[cid])

    # Retrieve the timestep delta time for the contact's associated body
    dt = time_dt[contact_wid_new[cid]]

    # Retrieve the body indices and states for the contact's associated bodies
    bid_AB = contact_bid_AB_new[cid]
    r_B = wp.transform_get_translation(body_q_i[bid_AB[1]])
    u_B = body_u_i[bid_AB[1]]

    # Iterate through all old contacts with the same key and check if contacts match
    # based on distance of contact points after accounting for associated body motion
    # NOTE: For the comparison, new_idx -> cid, old_idx -> sorted_to_unsorted_map_old[start + k]
    k = int32(0)
    found_match = int32(0)
    old_key = sorted_contact_keys_old[start]
    while target_key == old_key:
        # Retrieve the old contact index from the sorted->unsorted map
        cid_old = sorted_to_unsorted_map_old[start + k]
        r_k_B_old = contact_position_B_old[cid_old]
        W_Bk_T = wp.transpose(contact_wrench_matrix_from_points(r_k_B_old, r_B))
        r_B_candidate = r_k_B_old + dt * (W_Bk_T @ u_B)

        # Compute and check the distance to the target contact positions
        dr_B = wp.length(r_B_candidate - r_Bk_target)
        if dr_B < tolerance:
            # When a match is found, retrieve the contact reaction and velocity
            # from the old contact and transform them to the new contact frame
            q_k_old = contact_frame_old[cid_old]
            lambda_k_old = contact_reaction_old[cid_old]
            v_k_old = contact_velocity_old[cid_old]
            R_k_old = wp.quat_to_matrix(q_k_old)
            R_k_old_to_new = wp.transpose(R_k_target) @ R_k_old
            target_reaction = R_k_old_to_new @ lambda_k_old
            target_velocity = R_k_old_to_new @ v_k_old
            found_match = int32(1)
            break

        # Update the current old-key to check in the next iteration
        k += 1
        old_key = sorted_contact_keys_old[start + k]

    # If no matching contact found by position, fallback to net wrench approach
    if found_match == 0:
        # Retrieve the friction coefficient for the contact's associated material
        target_material = contact_material_new[cid]
        target_mu = target_material[0]

        # Retrieve the body indices and states for the contact's associated bodies
        # NOTE: If body A is static, then its velocity contribution is zero by definition
        v_Ak = vec3f(0.0)
        if bid_AB[0] >= 0:
            u_A = body_u_i[bid_AB[0]]
            r_A = wp.transform_get_translation(body_q_i[bid_AB[0]])
            r_Ak_target = contact_position_A_new[cid]
            W_Ak_T = wp.transpose(contact_wrench_matrix_from_points(r_Ak_target, r_A))
            v_Ak = W_Ak_T @ u_A
        u_B = body_u_i[bid_AB[1]]
        r_B = wp.transform_get_translation(body_q_i[bid_AB[1]])
        W_Bk_T = wp.transpose(contact_wrench_matrix_from_points(r_Bk_target, r_B))
        v_Bk = W_Bk_T @ u_B

        # Compute the new contact velocity based on the measured body motion
        # project to the contact frame and set normal component to zero
        # NOTE: We only need to consider tangential velocity for warm-starting
        # as the normal velocity should always be non-negative in the local
        # contact frame, and positive if the solver computes an opening contact
        # thus, for warm-starting we only need to consider the tangential velocity
        target_velocity = scaling * wp.transpose(R_k_target) @ (v_Bk - v_Ak)
        target_velocity.z = wp.max(target_velocity.z, 0.0)

        # Iterate through all old contacts with the same key and accumulate net body-com wrench
        # NOTE: We only need body B since it is the body on which the contact reaction acts positively
        geom_pair_force_body_B = vec3f(0.0)
        k = int32(0)
        old_key = sorted_contact_keys_old[start]
        while target_key == old_key:
            # Retrieve the old contact index from the sorted->unsorted map
            cid_old = sorted_to_unsorted_map_old[start + k]

            # Load old contact data for the old contact
            q_k_old = contact_frame_old[cid_old]
            lambda_k_old = contact_reaction_old[cid_old]

            # Accumulate the old contact's contribution to the geom-pair net force on body B
            geom_pair_force_body_B += wp.quat_to_matrix(q_k_old) @ lambda_k_old

            # Update the current old-key to check in the next iteration
            k += 1
            old_key = sorted_contact_keys_old[start + k]

        # TODO: We need to cache this value per geom-pair
        # TODO: Replace this with a new cache instead of recomputing every time --- IGNORE ---
        num_contacts_gid_AB_new = int32(0)
        for i in range(num_active_contacts_new[0]):
            if contact_key_new[i] == target_key:
                num_contacts_gid_AB_new += 1

        # Average the net body-com force over the number of contacts for this geom-pair
        contact_force_uniform_new = (1.0 / float32(num_contacts_gid_AB_new)) * geom_pair_force_body_B

        # Project to the new contact frame and local
        # friction cone to obtain the contact reaction
        target_reaction = wp.transpose(R_k_target) @ contact_force_uniform_new
        target_reaction = scaling * project_to_coulomb_cone(target_reaction, target_mu)

    # Store the new contact reaction and velocity
    # NOTE: These will remain zero if no matching contact is found
    contact_reaction_new[cid] = target_reaction
    contact_velocity_new[cid] = target_velocity


###
# Launchers
###


def warmstart_limits_by_matched_jid_dof_key(
    sorter: KeySorter,
    cache: LimitsData,
    limits: LimitsData,
):
    """
    Warm-starts limits by matching joint-DoF index-pair keys.

    Args:
        sorter (KeySorter): The key sorter used to sort cached limit keys.
        cache (LimitsData): The cached limits data from the previous simulation step.
        limits (LimitsData): The current limits data to be warm-started.
    """
    # First sort the keys of cached limits to facilitate binary search
    sorter.sort(num_active_keys=cache.model_num_limits, keys=cache.key)

    # Launch kernel to warmstart limits by matching jid keys
    wp.launch(
        kernel=_warmstart_limits_by_matched_jid_dof_key,
        dim=limits.num_model_max_limits,
        inputs=[
            # Inputs - Previous:
            sorter.sorted_keys,
            sorter.sorted_to_unsorted_map,
            cache.model_num_limits,
            cache.velocity,
            cache.reaction,
            # Inputs - Next:
            limits.model_num_limits,
            limits.key,
            # Outputs:
            limits.reaction,
            limits.velocity,
        ],
        device=sorter.device,
    )


def warmstart_contacts_by_matched_geom_pair_key_and_position(
    model: Model,
    data: ModelData,
    sorter: KeySorter,
    cache: ContactsData,
    contacts: ContactsData,
    tolerance: float32 | None = None,
):
    """
    Warm-starts contacts by matching geom-pair keys and contact point positions.

    Args:
        model (Model): The model containing simulation parameters.
        data (ModelData): The model data containing body states.
        sorter (KeySorter): The key sorter used to sort cached contact keys.
        cache (ContactsData): The cached contacts data from the previous simulation step.
        contacts (ContactsData): The current contacts data to be warm-started.
    """
    # Define tolerance for matching contact points based on distance after accounting for body motion
    if tolerance is None:
        tolerance = float32(1e-5)

    # First sort the keys of cached contacts to facilitate binary search
    sorter.sort(num_active_keys=cache.model_num_contacts, keys=cache.key)

    # Launch kernel to warmstart contacts by matching geom-pair keys and contact point positions
    wp.launch(
        kernel=_warmstart_contacts_by_matched_geom_pair_key_and_position,
        dim=contacts.num_model_max_contacts,
        inputs=[
            # Inputs - Common:
            tolerance,
            model.time.dt,
            data.bodies.q_i,
            data.bodies.u_i,
            # Inputs - Previous:
            sorter.sorted_keys,
            sorter.sorted_to_unsorted_map,
            cache.model_num_contacts,
            cache.position_B,
            cache.frame,
            cache.reaction,
            cache.velocity,
            # Inputs - Next:
            contacts.model_num_contacts,
            contacts.key,
            contacts.wid,
            contacts.bid_AB,
            contacts.position_B,
            contacts.frame,
            # Outputs:
            contacts.reaction,
            contacts.velocity,
        ],
        device=model.device,
    )


def warmstart_contacts_from_geom_pair_net_force(
    data: ModelData,
    sorter: KeySorter,
    cache: ContactsData,
    contacts: ContactsData,
    scaling: float32 | None = None,
):
    """
    Warm-starts contacts by matching geom-pair keys and contact point positions.

    Args:
        model (Model): The model containing simulation parameters.
        data (ModelData): The model data containing body states.
        sorter (KeySorter): The key sorter used to sort cached contact keys.
        cache (ContactsData): The cached contacts data from the previous simulation step.
        contacts (ContactsData): The current contacts data to be warm-started.
    """
    # Define scaling for warm-started reactions and velocities
    if scaling is None:
        scaling = float32(1.0)

    # First sort the keys of cached contacts to facilitate binary search
    sorter.sort(num_active_keys=cache.model_num_contacts, keys=cache.key)

    # Launch kernel to warmstart contacts by matching geom-pair keys and contact point positions
    wp.launch(
        kernel=_warmstart_contacts_from_geom_pair_net_force,
        dim=contacts.num_model_max_contacts,
        inputs=[
            # Inputs - Common:
            scaling,
            data.bodies.q_i,
            data.bodies.u_i,
            # Inputs - Previous:
            sorter.sorted_keys,
            sorter.sorted_to_unsorted_map,
            cache.model_num_contacts,
            cache.frame,
            cache.reaction,
            # Inputs - Next:
            contacts.model_num_contacts,
            contacts.key,
            contacts.bid_AB,
            contacts.position_A,
            contacts.position_B,
            contacts.frame,
            contacts.material,
            # Outputs:
            contacts.reaction,
            contacts.velocity,
        ],
        device=sorter.device,
    )


def warmstart_contacts_by_matched_geom_pair_key_and_position_with_net_force_backup(
    model: Model,
    data: ModelData,
    sorter: KeySorter,
    cache: ContactsData,
    contacts: ContactsData,
    tolerance: float32 | None = None,
    scaling: float32 | None = None,
):
    """
    Warm-starts contacts by matching geom-pair keys and contact point positions.

    Args:
        model (Model): The model containing simulation parameters.
        data (ModelData): The model data containing body states.
        sorter (KeySorter): The key sorter used to sort cached contact keys.
        cache (ContactsData): The cached contacts data from the previous simulation step.
        contacts (ContactsData): The current contacts data to be warm-started.
    """
    # Define tolerance for matching contact points based on distance after accounting for body motion
    if tolerance is None:
        tolerance = float32(1e-5)
    # Define scaling for warm-started reactions and velocities
    if scaling is None:
        scaling = float32(1.0)

    # First sort the keys of cached contacts to facilitate binary search
    sorter.sort(num_active_keys=cache.model_num_contacts, keys=cache.key)

    # Launch kernel to warmstart contacts by matching geom-pair keys and contact point positions
    wp.launch(
        kernel=_warmstart_contacts_by_matched_geom_pair_key_and_position_with_net_force_backup,
        dim=contacts.num_model_max_contacts,
        inputs=[
            # Inputs - Common:
            tolerance,
            scaling,
            model.time.dt,
            data.bodies.q_i,
            data.bodies.u_i,
            # Inputs - Previous:
            sorter.sorted_keys,
            sorter.sorted_to_unsorted_map,
            cache.model_num_contacts,
            cache.position_B,
            cache.frame,
            cache.reaction,
            cache.velocity,
            # Inputs - Next:
            contacts.model_num_contacts,
            contacts.key,
            contacts.wid,
            contacts.bid_AB,
            contacts.position_A,
            contacts.position_B,
            contacts.frame,
            contacts.material,
            # Outputs:
            contacts.reaction,
            contacts.velocity,
        ],
        device=model.device,
    )


###
# Interfaces
###


class WarmstarterLimits:
    """
    Provides a unified mechanism for warm-starting unilateral limit constraints.
    """

    def __init__(self, limits: Limits):
        """
        Initializes the limits warmstarter using the allocations of the provided limits container.

        Args:
            limits (Limits): The limits container whose allocations are used to initialize the warmstarter.
        """
        # Store the device of the provided contacts container
        self._device: Devicelike = limits.device

        # Declare the internal limits cache
        self._cache: LimitsData | None = None

        # Check if the limits container has allocations and skip cache allocations if not
        if limits.num_model_max_limits <= 0:
            return

        # Allocate contact data cache based on the those of the provided contacts container
        with wp.ScopedDevice(self._device):
            self._cache = LimitsData(
                num_model_max_limits=limits.num_model_max_limits,
                num_world_max_limits=limits.num_world_max_limits,
                model_num_limits=wp.zeros_like(limits.model_num_limits),
                key=wp.zeros_like(limits.key),
                velocity=wp.zeros_like(limits.velocity),
                reaction=wp.zeros_like(limits.reaction),
            )

        # Create a key sorter that can handle the maximum number of contacts
        self._sorter = KeySorter(max_num_keys=limits.num_model_max_limits, device=self._device)

    @property
    def device(self) -> Devicelike:
        """
        Returns the device on which the warmstarter operates.
        """
        return self._device

    @property
    def cache(self) -> LimitsData | None:
        """
        Returns the internal limits cache data.
        """
        return self._cache

    def warmstart(self, limits: Limits):
        """
        Warm-starts the provided contacts container using the internal cache.

        The current implementation matches contacts based on geom-pair keys and contact point positions.

        Args:
            model (Model): The model containing simulation parameters.
            data (ModelData): The model data containing body states.
            limits (Limits): The limits container to warm-start.
        """
        # Early exit if no cache is allocated
        if self._cache is None:
            return

        # Otherwise, perform warm-starting using matched jid-dof keys
        warmstart_limits_by_matched_jid_dof_key(
            sorter=self._sorter,
            cache=self._cache,
            limits=limits.data,
        )

    def update(self, limits: Limits):
        """
        Updates the warmstarter's internal cache with the provided limits data.

        Args:
            limits (Limits): The limits container from which to update the cache.
        """
        # Early exit if no cache is allocated
        if self._cache is None:
            return

        # Otherwise, copy over the limits data to the internal cache
        wp.copy(self._cache.model_num_limits, limits.model_num_limits)
        wp.copy(self._cache.key, limits.key)
        wp.copy(self._cache.velocity, limits.velocity)
        wp.copy(self._cache.reaction, limits.reaction)


class WarmstarterContacts:
    """
    Provides a unified mechanism for warm-starting unilateral contact constraints.

    This class supports multiple warm-starting strategies, selectable via the `Method` enum:
    - `KEY_AND_POSITION`:
        Warm-starts contacts by matching geom-pair keys and contact-point positions.
    - `GEOM_PAIR_NET_FORCE`:
        Warm-starts contacts using the net body-CoM contact force per geom-pair.
    - `GEOM_PAIR_NET_WRENCH`:
        Warm-starts contacts using the net body-CoM contact wrench per geom-pair.
    - `KEY_AND_POSITION_WITH_NET_FORCE_BACKUP`:
        Warm-starts contacts by matching geom-pair keys and contact-point positions,
    - with a backup strategy using the net body-CoM contact force per geom-pair.
    - `KEY_AND_POSITION_WITH_NET_WRENCH_BACKUP`:
        Warm-starts contacts by matching geom-pair keys and contact-point positions,
        with a backup strategy using the net body-CoM contact wrench per geom-pair.

    Geom-pair keys are unique identifiers for pairs of geometries involved in contacts,
    allowing for efficient matching of contacts across simulation steps. This class leverages
    the :class:`KeySorter` utility to facilitate rapid searching and matching of contacts
    based on these keys using Warp's Radix Sort operations.
    """

    class Method(IntEnum):
        """Defines the different warm-starting modes available for contacts."""

        KEY_AND_POSITION = 0
        """Warm-starts contacts by matching geom-pair keys and contact-point positions."""

        GEOM_PAIR_NET_FORCE = 1
        """Warm-starts contacts using the net body-CoM contact force per geom-pair."""

        GEOM_PAIR_NET_WRENCH = 2
        """Warm-starts contacts using the net body-CoM contact wrench per geom-pair."""

        KEY_AND_POSITION_WITH_NET_FORCE_BACKUP = 3
        """
        Warm-starts contacts by matching geom-pair keys and contact-point positions,
        with a backup strategy using the net body-CoM contact force per geom-pair.
        """

        KEY_AND_POSITION_WITH_NET_WRENCH_BACKUP = 4
        """
        Warm-starts contacts by matching geom-pair keys and contact-point positions,
        with a backup strategy using the net body-CoM contact wrench per geom-pair.
        """

        @override
        def __str__(self):
            """Returns a string representation of the WarmstarterContacts.Method."""
            return f"WarmstarterContacts.Method.{self.name} ({self.value})"

        @override
        def __repr__(self):
            """Returns a string representation of the WarmstarterContacts.Method."""
            return self.__str__()

    def __init__(
        self,
        contacts: Contacts,
        method: Method = Method.KEY_AND_POSITION,
        tolerance: float = 1e-5,
        scaling: float = 1.0,
    ):
        """
        Initializes the contacts warmstarter using the allocations of the provided contacts container.

        Args:
            contacts (Contacts): The contacts container whose allocations are used to initialize the warmstarter.
            method (WarmstarterContacts.Method): The warm-starting method to use.
            tolerance (float): The tolerance used for matching contact point positions.\n
                Must be a floating-point value specified in meters, and within the range `[0, +inf)`.\n
                Setting this to `0.0` requires exact position matches, effectively disabling position-based matching.
            scaling (float): The scaling factor applied to warm-started reactions and velocities.\n
                Must be a floating-point value specified in the range `[0, 1.0)`.\n
                Setting this to `0.0` effectively disables warm-starting.
        """
        # Store the specified warm-starting configurations
        self._method: WarmstarterContacts.Method = method
        self._tolerance: float32 = float32(tolerance)
        self._scaling: float32 = float32(scaling)

        # Set the device to use as that of the provided contacts container
        self._device: Devicelike = contacts.device

        # Declare the internal contacts cache
        self._cache: ContactsData | None = None

        # Check if the contacts container has allocations and skip cache allocations if not
        if contacts.num_model_max_contacts <= 0:
            return

        # Allocate contact data cache based on the those of the provided contacts container
        with wp.ScopedDevice(self._device):
            self._cache = ContactsData(
                num_model_max_contacts=contacts.num_model_max_contacts,
                num_world_max_contacts=contacts.num_world_max_contacts,
                model_num_contacts=wp.zeros_like(contacts.model_num_contacts),
                bid_AB=wp.full_like(contacts.bid_AB, value=vec2i(-1, -1)),
                position_A=wp.zeros_like(contacts.position_A),
                position_B=wp.zeros_like(contacts.position_B),
                frame=wp.zeros_like(contacts.frame),
                key=wp.zeros_like(contacts.key),
                velocity=wp.zeros_like(contacts.velocity),
                reaction=wp.zeros_like(contacts.reaction),
            )

        # Create a key sorter that can handle the maximum number of contacts
        self._sorter = KeySorter(max_num_keys=contacts.num_model_max_contacts, device=self._device)

    @property
    def device(self) -> Devicelike:
        """
        Returns the device on which the warmstarter operates.
        """
        return self._device

    @property
    def cache(self) -> ContactsData | None:
        """
        Returns the internal contacts cache data.
        """
        return self._cache

    def warmstart(self, model: Model, data: ModelData, contacts: Contacts):
        """
        Warm-starts the provided contacts container using the internal cache.

        The current implementation matches contacts based on geom-pair keys and contact point positions.

        Args:
            model (Model): The model containing simulation parameters.
            data (ModelData): The model data containing body states.
            contacts (Contacts): The contacts container to warm-start.
        """
        # Early exit if no cache is allocated
        if self._cache is None:
            return

        # Otherwise, perform warm-starting using the selected method
        match self._method:
            case WarmstarterContacts.Method.KEY_AND_POSITION:
                warmstart_contacts_by_matched_geom_pair_key_and_position(
                    model=model,
                    data=data,
                    sorter=self._sorter,
                    cache=self._cache,
                    contacts=contacts.data,
                    tolerance=self._tolerance,
                )

            case WarmstarterContacts.Method.GEOM_PAIR_NET_FORCE:
                warmstart_contacts_from_geom_pair_net_force(
                    data=data,
                    sorter=self._sorter,
                    cache=self._cache,
                    contacts=contacts.data,
                    scaling=self._scaling,
                )

            case WarmstarterContacts.Method.GEOM_PAIR_NET_WRENCH:
                raise NotImplementedError("WarmstarterContacts.Method.GEOM_PAIR_NET_WRENCH is not yet implemented.")

            case WarmstarterContacts.Method.KEY_AND_POSITION_WITH_NET_FORCE_BACKUP:
                warmstart_contacts_by_matched_geom_pair_key_and_position_with_net_force_backup(
                    model=model,
                    data=data,
                    sorter=self._sorter,
                    cache=self._cache,
                    contacts=contacts.data,
                    tolerance=self._tolerance,
                    scaling=self._scaling,
                )

            case WarmstarterContacts.Method.KEY_AND_POSITION_WITH_NET_WRENCH_BACKUP:
                raise NotImplementedError(
                    "WarmstarterContacts.Method.KEY_AND_POSITION_WITH_NET_WRENCH_BACKUP is not yet implemented."
                )

            case _:
                raise ValueError(
                    f"Unknown WarmstarterContacts.Method: {int(self._method)}"
                    "Supported methods are:"
                    "  - KEY_AND_POSITION (0),"
                    "  - GEOM_PAIR_NET_FORCE (1),"
                    "  - GEOM_PAIR_NET_WRENCH (2),"
                    "  - KEY_AND_POSITION_WITH_NET_FORCE_BACKUP (3),"
                    "  - KEY_AND_POSITION_WITH_NET_WRENCH_BACKUP (4)."
                )

    def update(self, contacts: Contacts):
        """
        Updates the warmstarter's internal cache with the provided contacts data.

        Args:
            contacts (Contacts): The contacts container from which to update the cache.
        """
        # Early exit if no cache is allocated
        if self._cache is None:
            return

        # Otherwise, copy over the contacts data to the internal cache
        wp.copy(self._cache.model_num_contacts, contacts.model_num_contacts)
        wp.copy(self._cache.bid_AB, contacts.bid_AB)
        wp.copy(self._cache.position_A, contacts.position_A)
        wp.copy(self._cache.position_B, contacts.position_B)
        wp.copy(self._cache.frame, contacts.frame)
        wp.copy(self._cache.key, contacts.key)
        wp.copy(self._cache.velocity, contacts.velocity)
        wp.copy(self._cache.reaction, contacts.reaction)
