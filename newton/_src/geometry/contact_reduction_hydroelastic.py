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

"""Hydroelastic contact reduction using hashtable-based tracking.

This module provides hydroelastic-specific contact reduction functionality,
building on the core ``GlobalContactReducer`` from ``contact_reduction_global.py``.

**Sign Convention:**

Uses standard SDF sign convention: negative depth = penetrating, positive = separated.
This matches the global contact reducer and other contact systems.

**Hydroelastic Contact Features:**

- Aggregate stiffness calculation: ``c_stiffness = k_eff * |agg_force| / total_depth``
- Normal matching: rotates reduced normals to align with aggregate force direction
- Anchor contact: synthetic contact at center of pressure for moment balance

**Usage:**

Use ``HydroelasticContactReduction`` for the high-level API, or call the individual
kernels for more control over the pipeline.

See Also:
    :class:`GlobalContactReducer` in ``contact_reduction_global.py`` for the
    core contact reduction system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import warp as wp

from newton._src.geometry.hashtable import hashtable_find_or_insert

from .contact_data import ContactData
from .contact_reduction import (
    NUM_NORMAL_BINS,
    NUM_SPATIAL_DIRECTIONS,
    NUM_VOXEL_DEPTH_SLOTS,
    compute_voxel_index,
    get_slot,
    get_spatial_direction_2d,
    project_point_to_plane,
)
from .contact_reduction_global import (
    BETA_THRESHOLD,
    VALUES_PER_KEY,
    GlobalContactReducer,
    GlobalContactReducerData,
    decode_oct,
    export_contact_to_buffer,
    is_contact_already_exported,
    make_contact_key,
    make_contact_value,
    reduction_update_slot,
    unpack_contact_id,
)

# =============================================================================
# Constants for hydroelastic export
# =============================================================================

EPS_LARGE = 1e-8
EPS_SMALL = 1e-20


@wp.func
def _effective_stiffness(k_a: wp.float32, k_b: wp.float32) -> wp.float32:
    denom = k_a + k_b
    if denom <= 0.0:
        return 0.0
    return (k_a * k_b) / denom


# =============================================================================
# Hydroelastic contact buffer function
# =============================================================================


@wp.func
def export_hydroelastic_contact_to_buffer(
    shape_a: int,
    shape_b: int,
    position: wp.vec3,
    normal: wp.vec3,
    depth: float,
    area: float,
    k_eff: float,
    reducer_data: GlobalContactReducerData,
) -> int:
    """Store a hydroelastic contact in the buffer with area and stiffness.

    Extends :func:`export_contact_to_buffer` by storing additional hydroelastic
    data (area and effective stiffness).

    Args:
        shape_a: First shape index
        shape_b: Second shape index
        position: Contact position in world space
        normal: Contact normal
        depth: Penetration depth (negative = penetrating, standard convention)
        area: Contact surface area
        k_eff: Effective stiffness coefficient k_a*k_b/(k_a+k_b)
        reducer_data: GlobalContactReducerData with all arrays

    Returns:
        Contact ID if successfully stored, -1 if buffer full
    """
    # Use base function to store common contact data
    contact_id = export_contact_to_buffer(shape_a, shape_b, position, normal, depth, reducer_data)

    if contact_id >= 0:
        # Store hydroelastic-specific data (k_eff is stored per entry, not per contact)
        reducer_data.contact_area[contact_id] = area

    return contact_id


# =============================================================================
# Hydroelastic reduction kernels
# =============================================================================


def get_reduce_hydroelastic_contacts_kernel(skip_aggregates: bool = False):
    """Create a hydroelastic contact reduction kernel.

    Args:
        skip_aggregates: If True, skip aggregate accumulation (agg_force,
            weighted_pos_sum, weight_sum).  Use this when the generate kernel
            already accumulated aggregates for all penetrating faces.

    Returns:
        A Warp kernel that registers buffered contacts in the hashtable.
    """

    @wp.kernel(enable_backward=False)
    def reduce_hydroelastic_contacts_kernel(
        reducer_data: GlobalContactReducerData,
        shape_material_k_hydro: wp.array(dtype=wp.float32),
        shape_transform: wp.array(dtype=wp.transform),
        shape_collision_aabb_lower: wp.array(dtype=wp.vec3),
        shape_collision_aabb_upper: wp.array(dtype=wp.vec3),
        shape_voxel_resolution: wp.array(dtype=wp.vec3i),
        total_num_threads: int,
    ):
        """Register hydroelastic contacts in the hashtable for reduction.

        Populates all hashtable slots (spatial extremes, max-depth, voxel) with
        real contact_ids from the buffer.  Optionally accumulates aggregates.
        """
        tid = wp.tid()

        num_contacts = reducer_data.contact_count[0]
        if num_contacts == 0:
            return
        num_contacts = wp.min(num_contacts, reducer_data.capacity)

        for i in range(tid, num_contacts, total_num_threads):
            pd = reducer_data.position_depth[i]
            normal = decode_oct(reducer_data.normal[i])
            pair = reducer_data.shape_pairs[i]
            area = reducer_data.contact_area[i]

            position = wp.vec3(pd[0], pd[1], pd[2])
            depth = pd[3]
            shape_a = pair[0]
            shape_b = pair[1]

            aabb_lower = shape_collision_aabb_lower[shape_b]
            aabb_upper = shape_collision_aabb_upper[shape_b]

            ht_capacity = reducer_data.ht_capacity

            # === Part 1: Normal-binned reduction ===
            bin_id = get_slot(normal)
            pos_2d = project_point_to_plane(bin_id, position)
            key = make_contact_key(shape_a, shape_b, bin_id)

            entry_idx = hashtable_find_or_insert(key, reducer_data.ht_keys, reducer_data.ht_active_slots)
            if entry_idx >= 0:
                # k_eff is constant for a shape pair, so redundant writes are safe.
                reducer_data.entry_k_eff[entry_idx] = _effective_stiffness(
                    shape_material_k_hydro[shape_a], shape_material_k_hydro[shape_b]
                )
                use_beta = depth < wp.static(BETA_THRESHOLD) * wp.length(aabb_upper - aabb_lower)
                if use_beta:
                    for dir_i in range(wp.static(NUM_SPATIAL_DIRECTIONS)):
                        dir_2d = get_spatial_direction_2d(dir_i)
                        score = wp.dot(pos_2d, dir_2d)
                        value = make_contact_value(score, i)
                        reduction_update_slot(entry_idx, dir_i, value, reducer_data.ht_values, ht_capacity)

                max_depth_value = make_contact_value(-depth, i)
                reduction_update_slot(
                    entry_idx,
                    wp.static(NUM_SPATIAL_DIRECTIONS),
                    max_depth_value,
                    reducer_data.ht_values,
                    ht_capacity,
                )

                if wp.static(not skip_aggregates):
                    if depth < 0.0:
                        force_weight = area * (-depth)
                        wp.atomic_add(reducer_data.agg_force, entry_idx, force_weight * normal)
                        wp.atomic_add(reducer_data.weighted_pos_sum, entry_idx, force_weight * position)
                        wp.atomic_add(reducer_data.weight_sum, entry_idx, force_weight)
            else:
                wp.atomic_add(reducer_data.ht_insert_failures, 0, 1)

            # === Part 2: Voxel-based reduction ===
            voxel_res = shape_voxel_resolution[shape_b]
            voxel_idx = compute_voxel_index(position, aabb_lower, aabb_upper, voxel_res)
            voxel_idx = wp.clamp(voxel_idx, 0, wp.static(NUM_VOXEL_DEPTH_SLOTS - 1))

            voxels_per_group = wp.static(NUM_SPATIAL_DIRECTIONS + 1)
            voxel_group = voxel_idx // voxels_per_group
            voxel_local_slot = voxel_idx % voxels_per_group

            voxel_bin_id = NUM_NORMAL_BINS + voxel_group
            voxel_key = make_contact_key(shape_a, shape_b, voxel_bin_id)

            voxel_entry_idx = hashtable_find_or_insert(voxel_key, reducer_data.ht_keys, reducer_data.ht_active_slots)
            if voxel_entry_idx >= 0:
                reducer_data.entry_k_eff[voxel_entry_idx] = _effective_stiffness(
                    shape_material_k_hydro[shape_a], shape_material_k_hydro[shape_b]
                )
                voxel_value = make_contact_value(-depth, i)
                reduction_update_slot(
                    voxel_entry_idx,
                    voxel_local_slot,
                    voxel_value,
                    reducer_data.ht_values,
                    ht_capacity,
                )
            else:
                wp.atomic_add(reducer_data.ht_insert_failures, 0, 1)

    return reduce_hydroelastic_contacts_kernel


# =============================================================================
# Hydroelastic export kernel factory
# =============================================================================


def create_export_hydroelastic_reduced_contacts_kernel(
    writer_func: Any,
    margin_contact_area: float,
    normal_matching: bool = True,
    anchor_contact: bool = False,
):
    """Create a kernel that exports reduced hydroelastic contacts using a custom writer function.

    Computes contact stiffness using the aggregate stiffness formula:
        c_stiffness = k_eff * |agg_force| / total_depth

    where:
    - agg_force = sum(area * |depth| * normal) for ALL contacts in the entry
    - total_depth = sum(|depth|) for SELECTED contacts (computed during export)

    This ensures the total contact force matches the aggregate force from all original contacts.

    Args:
        writer_func: A warp function with signature (ContactData, writer_data, int) -> None
        margin_contact_area: Contact area to use for non-penetrating contacts at the margin
        normal_matching: If True, rotate contact normals so their weighted sum aligns with aggregate force
        anchor_contact: If True, add an anchor contact at the center of pressure for each entry

    Returns:
        A warp kernel that can be launched to export reduced hydroelastic contacts.
    """
    # Define vector types for tracking exported contact data
    exported_ids_vec = wp.types.vector(length=VALUES_PER_KEY, dtype=wp.int32)
    exported_depths_vec = wp.types.vector(length=VALUES_PER_KEY, dtype=wp.float32)
    # Cache decoded normals (vec3 per slot) to avoid double decode_oct
    # Stored as 3 separate float vectors (Warp doesn't support vector-of-vec3)
    exported_nx_vec = wp.types.vector(length=VALUES_PER_KEY, dtype=wp.float32)
    exported_ny_vec = wp.types.vector(length=VALUES_PER_KEY, dtype=wp.float32)
    exported_nz_vec = wp.types.vector(length=VALUES_PER_KEY, dtype=wp.float32)

    @wp.kernel(enable_backward=False)
    def export_hydroelastic_reduced_contacts_kernel(
        # Hashtable arrays
        ht_keys: wp.array(dtype=wp.uint64),
        ht_values: wp.array(dtype=wp.uint64),
        ht_active_slots: wp.array(dtype=wp.int32),
        # Aggregate data per entry (from generate kernel)
        agg_force: wp.array(dtype=wp.vec3),
        weighted_pos_sum: wp.array(dtype=wp.vec3),
        weight_sum: wp.array(dtype=wp.float32),
        # Contact buffer arrays
        position_depth: wp.array(dtype=wp.vec4),
        normal: wp.array(dtype=wp.vec2),  # Octahedral-encoded
        shape_pairs: wp.array(dtype=wp.vec2i),
        contact_area: wp.array(dtype=wp.float32),
        entry_k_eff: wp.array(dtype=wp.float32),
        # Shape data for margin
        shape_contact_margin: wp.array(dtype=float),
        shape_transform: wp.array(dtype=wp.transform),
        # Writer data (custom struct)
        writer_data: Any,
        # Grid stride parameters
        total_num_threads: int,
    ):
        """Export reduced hydroelastic contacts to the writer with aggregate stiffness.

        Features:
        - Aggregate stiffness: c_stiffness = k_eff * |agg_force| / total_depth
        - Normal matching: rotates normals so weighted sum aligns with agg_force direction
        - Anchor contact: adds synthetic contact at center of pressure
        """
        tid = wp.tid()

        # Get number of active entries (stored at index = ht_capacity)
        ht_capacity = ht_keys.shape[0]
        num_active = ht_active_slots[ht_capacity]

        # Early exit if no active entries
        if num_active == 0:
            return

        # Grid stride loop over active entries
        for i in range(tid, num_active, total_num_threads):
            # Get the hashtable entry index
            entry_idx = ht_active_slots[i]

            # === First pass: collect unique contacts and compute aggregates ===
            exported_ids = exported_ids_vec()
            exported_depths = exported_depths_vec()
            # Cache decoded normals to avoid double decode_oct in second pass
            cached_nx = exported_nx_vec()
            cached_ny = exported_ny_vec()
            cached_nz = exported_nz_vec()
            num_exported = int(0)
            total_depth = float(0.0)  # Sum of |depth| for penetrating contacts
            max_pen_depth = float(0.0)  # Maximum penetration magnitude (positive value)
            k_eff_first = float(0.0)
            shape_a_first = int(0)
            shape_b_first = int(0)

            # For normal matching: sum of (|depth| * normal) for selected penetrating contacts
            selected_normal_sum = wp.vec3(0.0, 0.0, 0.0)

            # Read all value slots for this entry (slot-major layout)
            for slot in range(wp.static(VALUES_PER_KEY)):
                value = ht_values[slot * ht_capacity + entry_idx]

                # Skip empty slots (value = 0)
                if value == wp.uint64(0):
                    continue

                # Extract contact ID from low 32 bits
                contact_id = unpack_contact_id(value)

                # Skip if already exported
                if is_contact_already_exported(contact_id, exported_ids, num_exported):
                    continue

                # Unpack contact data (decode oct-normal once, cache for second pass)
                pd = position_depth[contact_id]
                contact_normal = decode_oct(normal[contact_id])
                depth = pd[3]

                # Record this contact, its depth, and cached normal
                exported_ids[num_exported] = contact_id
                exported_depths[num_exported] = depth
                cached_nx[num_exported] = contact_normal[0]
                cached_ny[num_exported] = contact_normal[1]
                cached_nz[num_exported] = contact_normal[2]
                num_exported = num_exported + 1

                # Sum penetrating depths for stiffness calculation (depth < 0 = penetrating)
                if depth < 0.0:
                    pen_magnitude = -depth  # Convert to positive magnitude
                    total_depth = total_depth + pen_magnitude
                    max_pen_depth = wp.max(max_pen_depth, pen_magnitude)
                    # Accumulate for normal matching
                    if wp.static(normal_matching):
                        selected_normal_sum = selected_normal_sum + pen_magnitude * contact_normal

                # Store first contact's shape pair (same for all contacts in the entry)
                if k_eff_first == 0.0:
                    k_eff_first = entry_k_eff[entry_idx]
                    pair = shape_pairs[contact_id]
                    shape_a_first = pair[0]
                    shape_b_first = pair[1]

            # Skip entries with no contacts
            if num_exported == 0:
                continue

            # === Compute stiffness and optional features based on entry type ===
            # Normal bin entries (bin_id 0-19): have aggregate force, use aggregate stiffness
            # Voxel bin entries (bin_id 20+): no aggregate force, use per-contact stiffness
            agg_force_vec = agg_force[entry_idx]
            agg_force_mag = wp.length(agg_force_vec)
            use_aggregate_stiffness = agg_force_mag > wp.static(EPS_LARGE)

            # Compute anchor position (center of pressure) for normal bin entries
            anchor_pos = wp.vec3(0.0, 0.0, 0.0)
            add_anchor = int(0)
            entry_weight_sum = weight_sum[entry_idx]
            if wp.static(anchor_contact) and use_aggregate_stiffness and max_pen_depth > 1e-6:
                if entry_weight_sum > wp.static(EPS_SMALL):
                    anchor_pos = weighted_pos_sum[entry_idx] / entry_weight_sum
                    add_anchor = 1

            # Compute total_depth including anchor contribution
            anchor_depth = max_pen_depth  # Anchor uses max penetration depth (positive magnitude)
            total_depth_with_anchor = total_depth + wp.float32(add_anchor) * anchor_depth

            # Compute shared stiffness for normal bin entries
            # c_stiffness = k_eff * |agg_force| / total_depth (matches original hydroelastic system)
            shared_stiffness = float(0.0)
            if use_aggregate_stiffness:
                if total_depth_with_anchor > 0.0:
                    shared_stiffness = k_eff_first * agg_force_mag / (total_depth_with_anchor + wp.static(EPS_LARGE))
                else:
                    # Fallback for non-penetrating contacts
                    shared_stiffness = wp.static(margin_contact_area) * k_eff_first

            # Compute normal matching rotation quaternion
            rotation_q = wp.quat_identity()
            if wp.static(normal_matching) and use_aggregate_stiffness:
                selected_mag = wp.length(selected_normal_sum)
                if selected_mag > wp.static(EPS_LARGE) and agg_force_mag > wp.static(EPS_LARGE):
                    selected_dir = selected_normal_sum / selected_mag
                    agg_dir = agg_force_vec / agg_force_mag

                    cross = wp.cross(selected_dir, agg_dir)
                    cross_mag = wp.length(cross)
                    dot_val = wp.dot(selected_dir, agg_dir)

                    if cross_mag > wp.static(EPS_LARGE):
                        # Normal case: compute rotation around cross product axis
                        axis = cross / cross_mag
                        angle = wp.acos(wp.clamp(dot_val, -1.0, 1.0))
                        rotation_q = wp.quat_from_axis_angle(axis, angle)
                    elif dot_val < 0.0:
                        # Vectors are anti-parallel: rotate 180 degrees around a perpendicular axis
                        perp = wp.vec3(1.0, 0.0, 0.0)
                        if wp.abs(wp.dot(selected_dir, perp)) > 0.9:
                            perp = wp.vec3(0.0, 1.0, 0.0)
                        axis = wp.normalize(wp.cross(selected_dir, perp))
                        rotation_q = wp.quat_from_axis_angle(axis, 3.14159265359)

            # Get transform and margin (same for all contacts in the entry)
            transform_b = shape_transform[shape_b_first]
            margin_a = shape_contact_margin[shape_a_first]
            margin_b = shape_contact_margin[shape_b_first]
            margin = margin_a + margin_b

            # === Second pass: export contacts ===
            for idx in range(num_exported):
                contact_id = exported_ids[idx]
                depth = exported_depths[idx]

                # Read position from buffer; use cached decoded normal (no re-decode)
                pd = position_depth[contact_id]
                position = wp.vec3(pd[0], pd[1], pd[2])
                contact_normal = wp.vec3(cached_nx[idx], cached_ny[idx], cached_nz[idx])

                # Get shape pair
                pair = shape_pairs[contact_id]
                shape_a = pair[0]
                shape_b = pair[1]

                # Apply normal matching rotation for penetrating contacts (depth < 0)
                final_normal = contact_normal
                if wp.static(normal_matching) and use_aggregate_stiffness and depth < 0.0:
                    final_normal = wp.normalize(wp.quat_rotate(rotation_q, contact_normal))

                # Compute stiffness based on entry type
                if use_aggregate_stiffness:
                    # Normal bin: shared stiffness from aggregate force
                    c_stiffness = shared_stiffness
                else:
                    # Voxel bin: per-contact stiffness (area * k_eff)
                    area = contact_area[contact_id]
                    if depth < 0.0:  # Penetrating
                        c_stiffness = area * k_eff_first
                    else:
                        c_stiffness = wp.static(margin_contact_area) * k_eff_first

                # Transform contact to world space
                normal_world = wp.transform_vector(transform_b, final_normal)
                pos_world = wp.transform_point(transform_b, position)

                # Create ContactData struct
                # contact_distance = 2 * depth (depth is already negative for penetrating)
                # This gives negative contact_distance for penetrating contacts
                contact_data = ContactData()
                contact_data.contact_point_center = pos_world
                contact_data.contact_normal_a_to_b = normal_world
                contact_data.contact_distance = 2.0 * depth  # depth is negative = penetrating
                contact_data.radius_eff_a = 0.0
                contact_data.radius_eff_b = 0.0
                contact_data.thickness_a = 0.0
                contact_data.thickness_b = 0.0
                contact_data.shape_a = shape_a
                contact_data.shape_b = shape_b
                contact_data.margin = margin
                contact_data.contact_stiffness = c_stiffness
                contact_data.contact_friction_scale = wp.float32(1.0)

                # Call the writer function
                writer_func(contact_data, writer_data, -1)

            # === Export anchor contact if enabled ===
            if add_anchor == 1:
                # Anchor normal is aligned with aggregate force direction
                anchor_normal = wp.normalize(agg_force_vec)
                anchor_normal_world = wp.transform_vector(transform_b, anchor_normal)
                anchor_pos_world = wp.transform_point(transform_b, anchor_pos)

                # Create ContactData for anchor
                # anchor_depth is positive magnitude, so negate for standard convention
                contact_data = ContactData()
                contact_data.contact_point_center = anchor_pos_world
                contact_data.contact_normal_a_to_b = anchor_normal_world
                contact_data.contact_distance = -2.0 * anchor_depth  # anchor_depth is positive magnitude
                contact_data.radius_eff_a = 0.0
                contact_data.radius_eff_b = 0.0
                contact_data.thickness_a = 0.0
                contact_data.thickness_b = 0.0
                contact_data.shape_a = shape_a_first
                contact_data.shape_b = shape_b_first
                contact_data.margin = margin
                contact_data.contact_stiffness = shared_stiffness
                contact_data.contact_friction_scale = wp.float32(1.0)

                # Call the writer function for anchor
                writer_func(contact_data, writer_data, -1)

    return export_hydroelastic_reduced_contacts_kernel


# =============================================================================
# Hydroelastic Contact Reduction API
# =============================================================================


@dataclass
class HydroelasticReductionConfig:
    """Configuration for hydroelastic contact reduction.

    Attributes:
        normal_matching: If True, rotate reduced contact normals so their weighted
            sum aligns with the aggregate force direction.
        anchor_contact: If True, add an anchor contact at the center of pressure
            for each normal bin. The anchor contact helps preserve moment balance.
        margin_contact_area: Contact area used for non-penetrating contacts at the margin.
    """

    normal_matching: bool = True
    anchor_contact: bool = False
    margin_contact_area: float = 1e-2


class HydroelasticContactReduction:
    """High-level API for hydroelastic contact reduction.

    This class encapsulates the hydroelastic contact reduction pipeline, providing
    a clean interface that hides the low-level kernel launch details. It manages:

    1. A ``GlobalContactReducer`` for contact storage and hashtable tracking
    2. The reduction kernels for hashtable registration
    3. The export kernel for writing reduced contacts

    **Usage Pattern:**

    The typical usage in a contact generation pipeline is:

    1. Call ``clear()`` at the start of each frame
    2. Write contacts to the buffer using ``export_hydroelastic_contact_to_buffer``
       in your contact generation kernel (use ``get_data_struct()`` to get the data)
    3. Call ``reduce()`` to register contacts in the hashtable
    4. Call ``export()`` to write reduced contacts using the writer function

    Example:

        .. code-block:: python

            # Initialize once
            config = HydroelasticReductionConfig(normal_matching=True)
            reduction = HydroelasticContactReduction(
                capacity=100000,
                device="cuda:0",
                writer_func=my_writer_func,
                config=config,
            )

            # Each frame
            reduction.clear()

            # Launch your contact generation kernel that uses:
            # export_hydroelastic_contact_to_buffer(..., reduction.get_data_struct())

            reduction.reduce(shape_transform, shape_sdf_data, grid_size)
            reduction.export(shape_contact_margin, shape_transform, writer_data, grid_size)

    Attributes:
        reducer: The underlying ``GlobalContactReducer`` instance.
        config: The ``HydroelasticReductionConfig`` for this instance.
        contact_count: Array containing the number of contacts in the buffer.

    See Also:
        :func:`export_hydroelastic_contact_to_buffer`: Warp function for writing
            contacts to the buffer from custom kernels.
        :class:`GlobalContactReducerData`: Struct for passing reducer data to kernels.
    """

    def __init__(
        self,
        capacity: int,
        device: str | None = None,
        writer_func: Any = None,
        config: HydroelasticReductionConfig | None = None,
    ):
        """Initialize the hydroelastic contact reduction system.

        Args:
            capacity: Maximum number of contacts to store in the buffer.
            device: Warp device (e.g., "cuda:0", "cpu"). If None, uses default device.
            writer_func: Warp function for writing decoded contacts. Must have signature
                ``(ContactData, writer_data, int) -> None``.
            config: Configuration options. If None, uses default ``HydroelasticReductionConfig``.
        """
        if config is None:
            config = HydroelasticReductionConfig()
        self.config = config
        self.device = device

        # Create the underlying reducer with hydroelastic data storage enabled
        self.reducer = GlobalContactReducer(
            capacity=capacity,
            device=device,
            store_hydroelastic_data=True,
        )

        # Create reduction kernel variants
        self._reduce_kernel = get_reduce_hydroelastic_contacts_kernel(skip_aggregates=False)
        self._reduce_kernel_skip_agg = get_reduce_hydroelastic_contacts_kernel(skip_aggregates=True)

        # Create the export kernel with the configured options
        self._export_kernel = create_export_hydroelastic_reduced_contacts_kernel(
            writer_func=writer_func,
            margin_contact_area=config.margin_contact_area,
            normal_matching=config.normal_matching,
            anchor_contact=config.anchor_contact,
        )

    @property
    def contact_count(self) -> wp.array:
        """Array containing the current number of contacts in the buffer."""
        return self.reducer.contact_count

    @property
    def capacity(self) -> int:
        """Maximum number of contacts that can be stored."""
        return self.reducer.capacity

    def get_data_struct(self) -> GlobalContactReducerData:
        """Get the data struct for passing to Warp kernels.

        Returns:
            A ``GlobalContactReducerData`` struct containing all arrays needed
            for contact storage and reduction.
        """
        return self.reducer.get_data_struct()

    def clear(self):
        """Clear all contacts and reset for a new frame.

        This efficiently clears only the active hashtable entries and resets
        the contact counter. Call this at the start of each simulation step.
        """
        self.reducer.clear_active()

    def reduce(
        self,
        shape_material_k_hydro: wp.array,
        shape_transform: wp.array,
        shape_collision_aabb_lower: wp.array,
        shape_collision_aabb_upper: wp.array,
        shape_voxel_resolution: wp.array,
        grid_size: int,
        skip_aggregates: bool = False,
    ):
        """Register buffered contacts in the hashtable for reduction.

        This launches the reduction kernel that processes all contacts in the
        buffer and registers them in the hashtable based on spatial extremes,
        max-depth per normal bin, and voxel-based slots.

        Args:
            shape_material_k_hydro: Per-shape hydroelastic material stiffness (dtype: float).
            shape_transform: Per-shape world transforms (dtype: wp.transform).
            shape_collision_aabb_lower: Per-shape local AABB lower bounds (dtype: wp.vec3).
            shape_collision_aabb_upper: Per-shape local AABB upper bounds (dtype: wp.vec3).
            shape_voxel_resolution: Per-shape voxel grid resolution (dtype: wp.vec3i).
            grid_size: Number of threads for the kernel launch.
            skip_aggregates: If True, skip aggregate accumulation (use when the
                generate kernel already accumulated aggregates for all faces).
        """
        kernel = self._reduce_kernel_skip_agg if skip_aggregates else self._reduce_kernel
        reducer_data = self.reducer.get_data_struct()
        wp.launch(
            kernel=kernel,
            dim=[grid_size],
            inputs=[
                reducer_data,
                shape_material_k_hydro,
                shape_transform,
                shape_collision_aabb_lower,
                shape_collision_aabb_upper,
                shape_voxel_resolution,
                grid_size,
            ],
            device=self.device,
        )

    def export(
        self,
        shape_contact_margin: wp.array,
        shape_transform: wp.array,
        writer_data: Any,
        grid_size: int,
    ):
        """Export reduced contacts using the writer function.

        This exports the winning contacts from the hashtable, computing
        aggregate stiffness and applying optional normal matching.

        Args:
            shape_contact_margin: Per-shape contact margin (dtype: float).
            shape_transform: Per-shape world transforms (dtype: wp.transform).
            writer_data: Data struct for the writer function.
            grid_size: Number of threads for the kernel launch.
        """
        wp.launch(
            kernel=self._export_kernel,
            dim=[grid_size],
            inputs=[
                self.reducer.hashtable.keys,
                self.reducer.ht_values,
                self.reducer.hashtable.active_slots,
                self.reducer.agg_force,
                self.reducer.weighted_pos_sum,
                self.reducer.weight_sum,
                self.reducer.position_depth,
                self.reducer.normal,
                self.reducer.shape_pairs,
                self.reducer.contact_area,
                self.reducer.entry_k_eff,
                shape_contact_margin,
                shape_transform,
                writer_data,
                grid_size,
            ],
            device=self.device,
        )

    def reduce_and_export(
        self,
        shape_material_k_hydro: wp.array,
        shape_transform: wp.array,
        shape_collision_aabb_lower: wp.array,
        shape_collision_aabb_upper: wp.array,
        shape_voxel_resolution: wp.array,
        shape_contact_margin: wp.array,
        writer_data: Any,
        grid_size: int,
    ):
        """Convenience method to reduce and export in one call.

        Combines ``reduce()`` and ``export()`` into a single method call.

        Args:
            shape_material_k_hydro: Per-shape hydroelastic material stiffness (dtype: float).
            shape_transform: Per-shape world transforms (dtype: wp.transform).
            shape_collision_aabb_lower: Per-shape local AABB lower bounds (dtype: wp.vec3).
            shape_collision_aabb_upper: Per-shape local AABB upper bounds (dtype: wp.vec3).
            shape_voxel_resolution: Per-shape voxel grid resolution (dtype: wp.vec3i).
            shape_contact_margin: Per-shape contact margin (dtype: float).
            writer_data: Data struct for the writer function.
            grid_size: Number of threads for the kernel launch.
        """
        self.reduce(
            shape_material_k_hydro,
            shape_transform,
            shape_collision_aabb_lower,
            shape_collision_aabb_upper,
            shape_voxel_resolution,
            grid_size,
        )
        self.export(shape_contact_margin, shape_transform, writer_data, grid_size)
