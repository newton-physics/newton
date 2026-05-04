# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Contact aggregation for RL applications.

This module provides functionality to aggregate per-contact data from Kamino's
ContactsKaminoData into per-body and per-geom summaries suitable for RL observations.
The aggregation is performed on GPU using efficient atomic operations.
"""

from dataclasses import dataclass

import warp as wp

from ..core.model import ModelKamino
from ..core.types import int32, quatf, vec2i, vec3f
from .contacts import ContactMode, ContactsKamino

###
# Module interface
###

__all__ = [
    "ContactAggregation",
    "ContactAggregationData",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


@wp.kernel
def _aggregate_contact_force_per_body(
    # Inputs:
    model_info_bodies_start: wp.array[int32],  # Per-world bodies start index
    model_active_contacts: wp.array[int32],  # contacts over all worlds
    contact_wid: wp.array[int32],  # world index per contact
    contact_bid_AB: wp.array[vec2i],  # body pair per contact (global body indices)
    contact_reaction: wp.array[vec3f],  # force in local contact frame
    contact_frame: wp.array[quatf],  # contact frame (rotation quaternion)
    contact_mode: wp.array[int32],  # contact mode
    # Outputs:
    body_net_force: wp.array3d[wp.float32],  # [num_worlds, max_bodies, 3]
    body_contact_flag: wp.array2d[int32],  # [num_worlds, max_bodies]
):
    """
    Aggregate contact force and flags per body across all contacts.

    Each thread processes one contact. Forces are transformed from local
    contact frame to world frame, then atomically accumulated to both
    bodies in the contact pair. Contact flags are set for both bodies.

    Args:
        model_info_bodies_start: Array of start indices for bodies in each world
        model_active_contacts: Number of active contacts over all worlds
        wid: World index for each contact
        bid_AB: Body index pair (A, B) for each contact
        reaction: 3D contact force in local contact frame [normal, tangent1, tangent2]
        frame: Contact frame as rotation quaternion w.r.t world
        mode: Contact mode (INACTIVE, OPENING, STICKING, SLIDING)
        body_net_force: Output array for net force per body (world frame)
        body_contact_flag: Output array for contact flag per body
    """
    # Retrieve the contact index for this thread
    contact_idx = wp.tid()

    # Early exit if this thread is beyond active contacts
    if contact_idx >= model_active_contacts[0]:
        return

    # Skip inactive contacts
    if contact_mode[contact_idx] == ContactMode.INACTIVE:
        return

    # Get contact-specific data
    world_idx = contact_wid[contact_idx]
    bid_AB = contact_bid_AB[contact_idx]
    global_body_A = bid_AB[0]  # Global body index
    global_body_B = bid_AB[1]  # Global body index

    # Retrieve the start index for bodies in this world to convert global body IDs to per-world indices
    bodies_start = model_info_bodies_start[world_idx]

    # Transform force from local contact frame to world frame
    force_local = contact_reaction[contact_idx]
    contact_quat = contact_frame[contact_idx]
    force_world = wp.quat_rotate(contact_quat, force_local)

    # Accumulate force to both bodies (equal and opposite)
    # Skip static bodies (bid < 0, e.g., ground plane)
    # Convert global body indices to per-world body indices for array indexing
    # Need to add each component separately for atomic operations on 3D arrays
    if global_body_A >= 0:
        body_A_in_world = global_body_A - bodies_start  # Convert to per-world index
        for i in range(3):
            wp.atomic_add(body_net_force, world_idx, body_A_in_world, i, -force_world[i])
        wp.atomic_max(body_contact_flag, world_idx, body_A_in_world, int32(1))

    if global_body_B >= 0:
        body_B_in_world = global_body_B - bodies_start  # Convert to per-world index
        for i in range(3):
            wp.atomic_add(body_net_force, world_idx, body_B_in_world, i, force_world[i])
        wp.atomic_max(body_contact_flag, world_idx, body_B_in_world, int32(1))


@wp.kernel
def _aggregate_static_contact_flag_per_body(
    # Inputs:
    model_info_bodies_start: wp.array[int32],  # Per-world bodies start index
    model_active_contacts: wp.array[int32],  # contacts over all worlds
    contact_wid: wp.array[int32],  # world index per contact
    contact_bid_AB: wp.array[vec2i],  # body pair per contact (global body indices)
    contact_mode: wp.array[int32],  # contact mode
    # Outputs:
    static_contact_flag: wp.array2d[int32],  # [num_worlds, max_bodies]
):
    """
    Identify which bodies are in contact with static geometries.

    Each thread processes one contact. If either geometry in the contact
    pair is marked as static, the corresponding non-static body's static
    contact flag is set.

    Args:
        model_active_contacts: Number of active contacts over all worlds
        contact_wid: World index for each contact
        contact_bid_AB: Body index pair (A, B) for each contact
        contact_gid_AB: Geometry index pair (A, B) for each contact
        contact_mode: Contact mode (INACTIVE, OPENING, STICKING, SLIDING)
        static_contact_flag: Output array for static contact flag per body
    """
    # Retrieve the contact index for this thread
    contact_idx = wp.tid()

    # Early exit if this thread is beyond active contacts
    if contact_idx >= model_active_contacts[0]:
        return

    # Skip inactive contacts
    if contact_mode[contact_idx] == ContactMode.INACTIVE:
        return

    # Retrieve contact-specific data
    world_idx = contact_wid[contact_idx]
    bid_AB = contact_bid_AB[contact_idx]
    global_body_A = bid_AB[0]  # Global body index
    global_body_B = bid_AB[1]  # Global body index

    # Retrieve the start index for bodies in this world to convert global body IDs to per-world indices
    bodies_start = model_info_bodies_start[world_idx]

    # Set static contact flag for non-static body
    # Convert global body indices to per-world body indices for array indexing
    # Skip static bodies (bid < 0, e.g., static plane)
    if global_body_B < 0 and global_body_A >= 0:
        # Body A is in contact with static (geom B)
        body_A_in_world = global_body_A - bodies_start
        wp.atomic_max(static_contact_flag, world_idx, body_A_in_world, int32(1))
    if global_body_A < 0 and global_body_B >= 0:
        # Body B is in contact with static (geom A)
        body_B_in_world = global_body_B - bodies_start
        wp.atomic_max(static_contact_flag, world_idx, body_B_in_world, int32(1))


@wp.kernel
def _aggregate_contact_force_per_body_geom(
    # Inputs:
    model_info_geoms_start: wp.array[int32],  # Offset to convert global geom ID to per-world index
    model_active_contacts: wp.array[int32],  # contacts over all worlds
    contact_wid: wp.array[int32],  # world index per contact
    contact_gid_AB: wp.array[vec2i],  # geometry pair per contact
    contact_bid_AB: wp.array[vec2i],  # geometry pair per contact
    contact_reaction: wp.array[vec3f],  # force in local contact frame
    contact_frame: wp.array[quatf],  # contact frame (rotation quaternion)
    contact_mode: wp.array[int32],  # contact mode
    # Outputs:
    geom_net_force: wp.array3d[wp.float32],  # [num_worlds, max_geoms, 3]
    geom_contact_flag: wp.array2d[int32],  # [num_worlds, max_geoms]
):
    """
    Aggregate contact force and flags per geometry across all contacts.

    Similar to _aggregate_contact_force_per_body, but aggregates to geometry
    level instead of body level. Useful for detailed contact analysis in RL.

    Args:
        model_info_geoms_start: Start index of per-world geoms
        world_active_contacts: Number of active contacts per world
        contact_wid: World index for each contact
        contact_gid_AB: Geometry index pair (A, B) for each contact
        contact_reaction: 3D contact force in local contact frame [normal, tangent1, tangent2]
        contact_frame: Contact frame as rotation quaternion w.r.t world
        contact_mode: Contact mode (INACTIVE, OPENING, STICKING, SLIDING)
        geom_net_force: Output array for net force per geometry (world frame)
        geom_contact_flag: Output array for contact flag per geometry
    """
    # Retrieve the contact index for this thread
    contact_idx = wp.tid()

    # Early exit if this thread is beyond active contacts
    if contact_idx >= model_active_contacts[0]:
        return

    # Skip inactive contacts
    if contact_mode[contact_idx] == ContactMode.INACTIVE:
        return

    # Get contact-specific data
    world_idx = contact_wid[contact_idx]
    gid_AB = contact_gid_AB[contact_idx]
    bid_AB = contact_bid_AB[contact_idx]
    global_geom_A = gid_AB[0]  # Global geom index
    global_geom_B = gid_AB[1]  # Global geom index
    global_body_A = bid_AB[0]  # Global body index
    global_body_B = bid_AB[1]  # Global body index

    # Compute in-world geom indices
    world_geom_start = model_info_geoms_start[world_idx]

    # Transform force from local contact frame to world frame
    force_local = contact_reaction[contact_idx]
    contact_quat = contact_frame[contact_idx]
    force_world = wp.quat_rotate(contact_quat, force_local)

    # Accumulate force to both geometries (equal and opposite)
    # Need to add each component separately for atomic operations on 3D arrays
    if global_body_A >= 0:
        world_geom_A = global_geom_A - world_geom_start  # Convert to per-world index
        for i in range(3):
            wp.atomic_add(geom_net_force, world_idx, world_geom_A, i, force_world[i])
        wp.atomic_max(geom_contact_flag, world_idx, world_geom_A, int32(1))
    if global_body_B >= 0:
        world_geom_B = global_geom_B - world_geom_start  # Convert to per-world index
        for i in range(3):
            wp.atomic_add(geom_net_force, world_idx, world_geom_B, i, force_world[i])
        wp.atomic_max(geom_contact_flag, world_idx, world_geom_B, int32(1))


@wp.kernel
def _aggregate_body_pair_contact_flag_per_world(
    # Input: Kamino ContactsData
    wid: wp.array[int32],  # world index per contact
    bid_AB: wp.array[vec2i],  # body pair per contact (global body indices)
    mode: wp.array[int32],  # contact mode
    world_active_contacts: wp.array[int32],  # contacts per world
    # Model data for global to per-world body ID conversion
    model_body_bid: wp.array[int32],  # Per-world body ID for each global body
    num_worlds: int,
    # Target body pair (per-world body indices)
    target_body_a: int,
    target_body_b: int,
    # Output
    body_pair_contact_flag: wp.array[int32],  # [num_worlds]
):
    """
    Detect contact between a specific pair of bodies across all worlds.

    Each thread processes one contact. If the contact involves the target
    body pair (in either order), the per-world flag is set.

    Args:
        wid: World index for each contact
        bid_AB: Body index pair (A, B) for each contact
        mode: Contact mode (INACTIVE, OPENING, STICKING, SLIDING)
        world_active_contacts: Number of active contacts per world
        model_body_bid: Mapping from global body index to per-world body index
        num_worlds: Total number of worlds
        target_body_a: Per-world body index of the first body in the target pair
        target_body_b: Per-world body index of the second body in the target pair
        body_pair_contact_flag: Output flag per world (1 if pair is in contact)
    """
    contact_idx = wp.tid()

    # Calculate total active contacts across all worlds
    total_contacts = int32(0)
    for w in range(num_worlds):
        total_contacts += world_active_contacts[w]

    # Early exit if this thread is beyond active contacts
    if contact_idx >= total_contacts:
        return

    # Skip inactive contacts
    if mode[contact_idx] == ContactMode.INACTIVE:
        return

    # Get contact data
    world_idx = wid[contact_idx]
    body_pair = bid_AB[contact_idx]
    global_body_A = body_pair[0]
    global_body_B = body_pair[1]

    # Skip static bodies (bid < 0)
    if global_body_A < 0 or global_body_B < 0:
        return

    # Convert global body indices to per-world body indices
    body_A_in_world = model_body_bid[global_body_A]
    body_B_in_world = model_body_bid[global_body_B]

    # Check if this contact matches the target pair (in either order)
    if (body_A_in_world == target_body_a and body_B_in_world == target_body_b) or (
        body_A_in_world == target_body_b and body_B_in_world == target_body_a
    ):
        wp.atomic_max(body_pair_contact_flag, world_idx, int32(1))


###
# Types
###


@dataclass
class ContactAggregationData:
    """
    Pre-allocated arrays for aggregating contact data per world and body.
    Designed for efficient GPU computation and zero-copy PyTorch access.
    """

    # === Per-Body Aggregated Data (for RL interface) ===

    body_net_contact_force: wp.array | None = None
    """
    Net contact force per body (world frame).\n
    Shape `(num_worlds, max_bodies_per_world)` and dtype=:class:`vec3f`.
    """

    body_contact_flag: wp.array | None = None
    """
    Binary contact flag per body (any contact).
    Shape `(num_worlds, max_bodies_per_world)` and dtype=:class:`int32` (0 or 1).
    """

    body_static_contact_flag: wp.array | None = None
    """
    Static contact flag per body (contact with static geoms).\n
    Shape `(num_worlds, max_bodies_per_world)` and dtype=:class:`int32` (0 or 1).
    """

    # === Per-Geom Detailed Data (for advanced RL) ===

    geom_net_contact_force: wp.array | None = None
    """
    Net contact force per geometry (world frame).\n
    Shape `(num_worlds, max_geoms_per_world, 3)` and dtype=:class:`float32`.
    """

    geom_contact_flag: wp.array | None = None
    """
    Contact flags per geometry.
    Shape `(num_worlds, max_geoms_per_world)` and dtype=:class:`int32` (0 or 1).
    """

    # === Contact Position/Normal Data (optional, for visualization) ===

    body_contact_position: wp.array | None = None
    """
    Average contact position per body (world frame).\n
    Shape `(num_worlds, max_bodies_per_world, 3)` and dtype=:class:`float32`.
    """

    body_contact_normal: wp.array | None = None
    """
    Average contact normal per body (world frame).\n
    Shape `(num_worlds, max_bodies_per_world, 3)` and dtype=:class:`float32`.
    """

    body_num_contacts: wp.array | None = None
    """
    Number of contacts per body.\n
    Shape `(num_worlds, max_bodies_per_world)` and dtype=:class:`int32`.
    """

    # === Body-Pair Contact Detection ===

    body_pair_contact_flag: wp.array | None = None
    """
    Per-world flag indicating contact between a specific body pair.\n
    Shape `(num_worlds,)` and dtype=:class:`int32` (0 or 1).
    """


###
# Interfaces
###


class ContactAggregation:
    """
    High-level interface for aggregating Kamino contact data for RL.

    This class efficiently aggregates per-contact data from Kamino's ContactsKaminoData
    into per-body and per-geom summaries suitable for RL observations. All computation
    is performed on GPU using atomic operations for efficiency.

    Usage:
        aggregation = ContactAggregation(model, contacts, static_geom_ids=[0])
        aggregation.compute()  # Call after simulator.step()

        # Access via PyTorch tensors (zero-copy)
        net_force = wp.to_torch(aggregation.body_net_force)
        contact_flag = wp.to_torch(aggregation.body_contact_flag)
    """

    def __init__(
        self,
        model: ModelKamino | None = None,
        contacts: ContactsKamino | None = None,
        enable_positions_normals: bool = False,
    ):
        """Initialize contact aggregation.

        Args:
            model (ModelKamino | None):
                The model container describing the system to be simulated.
                If None, call ``finalize()`` later.
            contacts (ContactsKamino | None):
                The contacts container with per-contact data.
                If None, call ``finalize()`` later.
            enable_positions_normals:
                Whether to compute average contact positions and normals per body.
        """
        # Declare the device cache
        self._device: wp.DeviceLike = None

        # Forward declarations
        self._model: ModelKamino | None = None
        self._contacts: ContactsKamino | None = None
        self._data: ContactAggregationData | None = None
        self._enable_positions_normals: bool = enable_positions_normals

        # Body-pair filter (set via set_body_pair_filter)
        self._body_pair_target_a: int = -1
        self._body_pair_target_b: int = -1

        # Proceed with memory allocations if model and contacts are provided
        if model is not None and contacts is not None:
            self.finalize(model=model, contacts=contacts, enable_positions_normals=enable_positions_normals)

    ###
    # Properties
    ###

    @property
    def body_net_force(self) -> wp.array:
        """Net force per body [num_worlds, max_bodies, 3]"""
        return self._data.body_net_contact_force

    @property
    def body_contact_flag(self) -> wp.array:
        """Contact flags per body [num_worlds, max_bodies]"""
        return self._data.body_contact_flag

    @property
    def body_static_contact_flag(self) -> wp.array:
        """Static contact flag per body [num_worlds, max_bodies]"""
        return self._data.body_static_contact_flag

    @property
    def geom_net_force(self) -> wp.array:
        """Net force per geom [num_worlds, max_geoms, 3]"""
        return self._data.geom_net_contact_force

    @property
    def geom_contact_flag(self) -> wp.array:
        """Contact flags per geom [num_worlds, max_geoms]"""
        return self._data.geom_contact_flag

    @property
    def body_pair_contact_flag(self) -> wp.array:
        """Per-world body-pair contact flag [num_worlds]."""
        return self._data.body_pair_contact_flag

    ###
    # Operations
    ###

    def finalize(
        self,
        model: ModelKamino,
        contacts: ContactsKamino,
        enable_positions_normals: bool = False,
    ) -> None:
        """Finalizes memory allocations for the contact aggregation data.

        Args:
            model (ModelKamino): The model container describing the system to be simulated.
            contacts (ContactsKamino): The contacts container with per-contact data.
        """
        # Use the model's device
        self._device = model.device

        # Override the positions/normals flag if different from current setting
        if enable_positions_normals != self._enable_positions_normals:
            self._enable_positions_normals = enable_positions_normals

        # Cache references to source model and contacts containers
        self._model = model
        self._contacts = contacts

        # Create locals for better readability
        num_worlds = model.size.num_worlds
        max_bodies = model.size.max_of_num_bodies
        max_geoms = model.size.max_of_num_geoms
        extended = self._enable_positions_normals

        # Allocate arrays for aggregated data based on model dimensions on the target device
        with wp.ScopedDevice(self._device):
            self._data = ContactAggregationData(
                body_net_contact_force=wp.zeros((num_worlds, max_bodies, 3), dtype=wp.float32),
                body_contact_flag=wp.zeros((num_worlds, max_bodies), dtype=wp.int32),
                body_static_contact_flag=wp.zeros((num_worlds, max_bodies), dtype=wp.int32),
                body_contact_position=wp.zeros((num_worlds, max_bodies, 3), dtype=wp.float32) if extended else None,
                body_contact_normal=wp.zeros((num_worlds, max_bodies, 3), dtype=wp.float32) if extended else None,
                body_num_contacts=wp.zeros((num_worlds, max_bodies), dtype=wp.int32) if extended else None,
                geom_net_contact_force=wp.zeros((num_worlds, max_geoms, 3), dtype=wp.float32),
                geom_contact_flag=wp.zeros((num_worlds, max_geoms), dtype=wp.int32),
            )

    def compute(self, skip_if_no_contacts: bool = False):
        """
        Compute aggregated contact data from current ContactsKaminoData.

        This method should be called after simulator.step() to update contact
        force and flags. It launches GPU kernels to efficiently aggregate
        per-contact data into per-body and per-geom summaries.

        Args:
            skip_if_no_contacts:
                If True, check for zero contacts and return early.\n
                Set to False when using CUDA graphs to avoid GPU-to-CPU copies.
        """

        # Zero out previous results
        self._data.body_net_contact_force.zero_()
        self._data.body_contact_flag.zero_()
        self._data.body_static_contact_flag.zero_()
        self._data.geom_net_contact_force.zero_()
        self._data.geom_contact_flag.zero_()

        if self._enable_positions_normals:
            self._data.body_contact_position.zero_()
            self._data.body_contact_normal.zero_()
            self._data.body_num_contacts.zero_()

        # Get contact data
        contacts_data = self._contacts.data

        # Optionally check if there are any active contacts
        # TODO @agon-serifi: Please check, but I think this might cause CPU-to-GPU transfer
        # command during graph capture, which can be problematic. We might want to require
        # the caller to check this before calling compute() when using graphs.
        # TODO: Might be better to just let the kernels early-exit since they already do this
        if skip_if_no_contacts:
            total_active = contacts_data.model_active_contacts.numpy()[0]
            if total_active == 0:
                return  # No contacts, nothing to aggregate

        # Launch aggregation kernel for per-body force
        wp.launch(
            _aggregate_contact_force_per_body,
            dim=contacts_data.model_max_contacts_host,
            inputs=[
                self._model.info.bodies_offset,
                contacts_data.model_active_contacts,
                contacts_data.wid,
                contacts_data.bid_AB,
                contacts_data.reaction,
                contacts_data.frame,
                contacts_data.mode,
            ],
            outputs=[
                self._data.body_net_contact_force,
                self._data.body_contact_flag,
            ],
            device=self._device,
        )

        # Launch aggregation kernel for static contact flag
        wp.launch(
            _aggregate_static_contact_flag_per_body,
            dim=contacts_data.model_max_contacts_host,
            inputs=[
                self._model.info.bodies_offset,
                contacts_data.model_active_contacts,
                contacts_data.wid,
                contacts_data.bid_AB,
                contacts_data.mode,
            ],
            outputs=[
                self._data.body_static_contact_flag,
            ],
            device=self._device,
        )

        # Launch aggregation kernel for per body-geom force
        # NOTE: body-geom, in this case, refers to geoms belonging to dynamic bodies, meaning that static geoms are excluded
        wp.launch(
            _aggregate_contact_force_per_body_geom,
            dim=contacts_data.model_max_contacts_host,
            inputs=[
                self._model.info.geoms_offset,
                contacts_data.model_active_contacts,
                contacts_data.wid,
                contacts_data.gid_AB,
                contacts_data.bid_AB,
                contacts_data.reaction,
                contacts_data.frame,
                contacts_data.mode,
            ],
            outputs=[
                self._data.geom_net_contact_force,
                self._data.geom_contact_flag,
            ],
            device=self._device,
        )

    # ------------------------------------------------------------------
    # Body-pair contact detection
    # ------------------------------------------------------------------

    def set_body_pair_filter(self, body_a_idx: int, body_b_idx: int) -> None:
        """Configure detection of contacts between a specific body pair.

        After calling this, use :meth:`compute_body_pair_contact` to detect
        whether the specified bodies are in contact in each world.

        Args:
            body_a_idx: Per-world body index of the first body.
            body_b_idx: Per-world body index of the second body.
        """
        self._body_pair_target_a = body_a_idx
        self._body_pair_target_b = body_b_idx

        # Allocate output array if not yet allocated
        num_worlds = self._model.size.num_worlds
        self._data.body_pair_contact_flag = wp.zeros(num_worlds, dtype=wp.int32, device=self._device)

    def compute_body_pair_contact(self) -> None:
        """Detect contact between the configured body pair.

        Must be called after :meth:`set_body_pair_filter`. This method is
        separate from :meth:`compute` so it can be called outside of CUDA
        graph capture when the body pair is configured after graph creation.

        Raises:
            RuntimeError: If no body pair filter has been configured.
        """
        if self._body_pair_target_a < 0 or self._body_pair_target_b < 0:
            return

        self._data.body_pair_contact_flag.zero_()

        contacts_data = self._contacts.data
        num_worlds = self._model.size.num_worlds

        wp.launch(
            _aggregate_body_pair_contact_flag_per_world,
            dim=contacts_data.model_max_contacts_host,
            inputs=[
                contacts_data.wid,
                contacts_data.bid_AB,
                contacts_data.mode,
                contacts_data.world_active_contacts,
                self._model.bodies.bid,
                num_worlds,
                self._body_pair_target_a,
                self._body_pair_target_b,
            ],
            outputs=[
                self._data.body_pair_contact_flag,
            ],
            device=self._device,
        )
