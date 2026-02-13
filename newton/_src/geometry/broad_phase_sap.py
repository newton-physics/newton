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

"""Sweep and Prune (SAP) broad phase collision detection.

Provides O(N log N) broad phase by projecting AABBs onto an axis and using
sorted interval overlap tests. More efficient than NxN for larger scenes.

See Also:
    :class:`BroadPhaseAllPairs` in ``broad_phase_nxn.py`` for simpler O(N²) approach.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import warp as wp

from .broad_phase_common import (
    binary_search,
    check_aabb_overlap,
    is_pair_excluded,
    precompute_world_map,
    test_world_and_group_pair,
    write_pair,
)

wp.set_module_options({"enable_backward": False})


class SAPSortType(IntEnum):
    """Sort algorithm to use for SAP broad phase."""

    SEGMENTED = 0  # Use wp.utils.segmented_sort_pairs (default)
    TILE = 1  # Use wp.tile_sort with shared memory (faster for certain sizes)


@wp.func
def _sap_project_aabb(
    elementid: int,
    direction: wp.vec3,  # Must be normalized
    shape_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
    shape_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
    shape_contact_margin: wp.array(
        dtype=float, ndim=1
    ),  # Optional per-shape contact margins (can be empty if AABBs pre-expanded)
) -> wp.vec2:
    lower = shape_bounding_box_lower[elementid]
    upper = shape_bounding_box_upper[elementid]

    # Check if margins are provided (empty array means AABBs are pre-expanded)
    margin = 0.0
    if shape_contact_margin.shape[0] > 0:
        margin = shape_contact_margin[elementid]

    half_size = 0.5 * (upper - lower)
    half_size = wp.vec3(half_size[0] + margin, half_size[1] + margin, half_size[2] + margin)
    radius = wp.dot(direction, half_size)
    center = wp.dot(direction, 0.5 * (lower + upper))
    return wp.vec2(center - radius, center + radius)


@wp.func
def binary_search_segment(
    arr: wp.array(dtype=float, ndim=1),
    base_idx: int,
    value: float,
    start: int,
    end: int,
) -> int:
    """Binary search in a segment of a 1D array.

    Args:
        arr: The array to search in
        base_idx: Base index offset for this segment
        value: Value to search for
        start: Start index (relative to base_idx)
        end: End index (relative to base_idx)

    Returns:
        Index (relative to base_idx) where value should be inserted
    """
    low = int(start)
    high = int(end)

    while low < high:
        mid = (low + high) // 2
        if arr[base_idx + mid] < value:
            low = mid + 1
        else:
            high = mid

    return low


def _create_tile_sort_kernel(tile_size: int):
    """Create a tile-based sort kernel for a specific tile size.

    This uses Warp's tile operations for efficient shared-memory sorting.
    Note: tile_size should match max_geoms_per_world and can be any value.

    Args:
        tile_size: Size of each tile (should match max_geoms_per_world)

    Returns:
        A Warp kernel that performs segmented tile-based sorting
    """

    @wp.kernel
    def tile_sort_kernel(
        sap_projection_lower: wp.array(dtype=float, ndim=1),
        sap_sort_index: wp.array(dtype=int, ndim=1),
        max_geoms_per_world: int,
    ):
        """Tile-based segmented sort kernel.

        Each thread block processes one world's data using shared memory.
        Loads tile_size elements (equal to max_geoms_per_world).
        Padding values (1e30) will sort to the end automatically.
        """
        world_id = wp.tid()

        # Calculate base index for this world
        base_idx = world_id * max_geoms_per_world

        # Load data into tiles (shared memory)
        # tile_size is a closure variable, treated as compile-time constant by Warp
        keys = wp.tile_load(sap_projection_lower, shape=(tile_size,), offset=(base_idx,), storage="shared")
        values = wp.tile_load(sap_sort_index, shape=(tile_size,), offset=(base_idx,), storage="shared")

        # Perform in-place sorting on shared memory
        wp.tile_sort(keys, values)

        # Store sorted data back to global memory
        wp.tile_store(sap_projection_lower, keys, offset=(base_idx,))
        wp.tile_store(sap_sort_index, values, offset=(base_idx,))

    return tile_sort_kernel


@wp.kernel
def _sap_project_kernel(
    direction: wp.vec3,  # Must be normalized
    shape_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
    shape_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
    shape_contact_margin: wp.array(
        dtype=float, ndim=1
    ),  # Optional per-shape contact margins (can be empty if AABBs pre-expanded)
    world_index_map: wp.array(dtype=int, ndim=1),
    world_slice_ends: wp.array(dtype=int, ndim=1),
    max_shapes_per_world: int,
    # Outputs (1D arrays with manual indexing)
    sap_projection_lower_out: wp.array(dtype=float, ndim=1),
    sap_projection_upper_out: wp.array(dtype=float, ndim=1),
    sap_sort_index_out: wp.array(dtype=int, ndim=1),
):
    world_id, local_shape_id = wp.tid()

    # Calculate 1D index: world_id * max_shapes_per_world + local_shape_id
    idx = world_id * max_shapes_per_world + local_shape_id

    # Get slice boundaries for this world
    world_slice_start = 0
    if world_id > 0:
        world_slice_start = world_slice_ends[world_id - 1]
    world_slice_end = world_slice_ends[world_id]
    num_shapes_in_world = world_slice_end - world_slice_start

    # Check if this thread is within valid range
    if local_shape_id >= num_shapes_in_world:
        # Pad with invalid values
        sap_projection_lower_out[idx] = 1e30
        sap_projection_upper_out[idx] = 1e30
        sap_sort_index_out[idx] = -1
        return

    # Map to actual geometry index
    shape_id = world_index_map[world_slice_start + local_shape_id]

    # Project AABB onto direction
    range = _sap_project_aabb(
        shape_id, direction, shape_bounding_box_lower, shape_bounding_box_upper, shape_contact_margin
    )

    sap_projection_lower_out[idx] = range[0]
    sap_projection_upper_out[idx] = range[1]
    sap_sort_index_out[idx] = local_shape_id


@wp.kernel
def _sap_range_kernel(
    world_slice_ends: wp.array(dtype=int, ndim=1),
    max_shapes_per_world: int,
    sap_projection_lower_in: wp.array(dtype=float, ndim=1),
    sap_projection_upper_in: wp.array(dtype=float, ndim=1),
    sap_sort_index_in: wp.array(dtype=int, ndim=1),
    sap_range_out: wp.array(dtype=int, ndim=1),
):
    world_id, local_shape_id = wp.tid()

    # Calculate 1D index
    idx = world_id * max_shapes_per_world + local_shape_id

    # Get number of geometries in this world
    world_slice_start = 0
    if world_id > 0:
        world_slice_start = world_slice_ends[world_id - 1]
    world_slice_end = world_slice_ends[world_id]
    num_shapes_in_world = world_slice_end - world_slice_start

    if local_shape_id >= num_shapes_in_world:
        sap_range_out[idx] = 0
        return

    # Current bounding shape (after sort, this is the original local geometry index)
    # Note: sap_sort_index_in[idx] contains the original local geometry index of the
    # geometry that's now at position local_shape_id in the sorted array
    sort_idx = sap_sort_index_in[idx]

    # Invalid shape (padding)
    if sort_idx < 0:
        sap_range_out[idx] = 0
        return

    # Get upper bound for this shape
    # sort_idx is the original local geometry index, so we use it to index into
    # sap_projection_upper_in (which is NOT sorted, only sap_projection_lower_in is sorted)
    upper_idx = world_id * max_shapes_per_world + sort_idx
    upper = sap_projection_upper_in[upper_idx]

    # Binary search for the limit in this world's segment
    # We need to search in the range [local_shape_id + 1, num_shapes_in_world)
    world_base_idx = world_id * max_shapes_per_world
    limit = binary_search_segment(
        sap_projection_lower_in, world_base_idx, upper, local_shape_id + 1, num_shapes_in_world
    )
    limit = wp.min(num_shapes_in_world, limit)

    # Range of shapes for the sweep and prune process
    sap_range_out[idx] = limit - local_shape_id - 1


@wp.func
def _process_single_sap_pair(
    pair: wp.vec2i,
    shape_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
    shape_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
    shape_contact_margin: wp.array(
        dtype=float, ndim=1
    ),  # Optional per-shape contact margins (can be empty if AABBs pre-expanded)
    candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
    num_candidate_pair: wp.array(dtype=int, ndim=1),  # Size one array
    max_candidate_pair: int,
    filter_pairs: wp.array(dtype=wp.vec2i, ndim=1),  # Sorted excluded pairs (empty if none)
    num_filter_pairs: int,
):
    shape1 = pair[0]
    shape2 = pair[1]

    # Skip explicitly excluded pairs (e.g. shape_collision_filter_pairs)
    if num_filter_pairs > 0 and is_pair_excluded(pair, filter_pairs, num_filter_pairs):
        return

    # Check if margins are provided (empty array means AABBs are pre-expanded)
    margin1 = 0.0
    margin2 = 0.0
    if shape_contact_margin.shape[0] > 0:
        margin1 = shape_contact_margin[shape1]
        margin2 = shape_contact_margin[shape2]

    if check_aabb_overlap(
        shape_bounding_box_lower[shape1],
        shape_bounding_box_upper[shape1],
        margin1,
        shape_bounding_box_lower[shape2],
        shape_bounding_box_upper[shape2],
        margin2,
    ):
        write_pair(
            pair,
            candidate_pair,
            num_candidate_pair,
            max_candidate_pair,
        )


@wp.kernel
def _sap_broadphase_kernel(
    # Input arrays
    shape_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
    shape_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
    shape_contact_margin: wp.array(
        dtype=float, ndim=1
    ),  # Optional per-shape contact margins (can be empty if AABBs pre-expanded)
    collision_group: wp.array(dtype=int, ndim=1),
    shape_world: wp.array(dtype=int, ndim=1),  # World indices
    world_index_map: wp.array(dtype=int, ndim=1),
    world_slice_ends: wp.array(dtype=int, ndim=1),
    sap_sort_index_in: wp.array(dtype=int, ndim=1),  # 1D array with manual indexing
    sap_cumulative_sum_in: wp.array(dtype=int, ndim=1),  # Flattened [num_worlds * max_shapes]
    num_worlds: int,
    max_shapes_per_world: int,
    nsweep_in: int,
    num_regular_worlds: int,  # Number of regular world segments (excluding dedicated -1 segment)
    filter_pairs: wp.array(dtype=wp.vec2i, ndim=1),  # Sorted excluded pairs (empty if none)
    num_filter_pairs: int,
    # Output arrays
    candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
    num_candidate_pair: wp.array(dtype=int, ndim=1),  # Size one array
    max_candidate_pair: int,
):
    tid = wp.tid()

    total_work_packages = sap_cumulative_sum_in[num_worlds * max_shapes_per_world - 1]

    workid = tid
    while workid < total_work_packages:
        # Binary search to find which (world, local_shape) this work package belongs to
        flat_id = binary_search(sap_cumulative_sum_in, workid, 0, num_worlds * max_shapes_per_world)

        # Calculate j from flat_id and workid
        j = flat_id + workid + 1
        if flat_id > 0:
            j -= sap_cumulative_sum_in[flat_id - 1]

        # Convert flat_id to world and local indices
        world_id = flat_id // max_shapes_per_world
        i = flat_id % max_shapes_per_world
        j = j % max_shapes_per_world

        # Get slice boundaries for this world
        world_slice_start = 0
        if world_id > 0:
            world_slice_start = world_slice_ends[world_id - 1]
        world_slice_end = world_slice_ends[world_id]
        num_shapes_in_world = world_slice_end - world_slice_start

        # Check validity: ensure indices are within bounds
        if i >= num_shapes_in_world or j >= num_shapes_in_world:
            workid += nsweep_in
            continue

        # Skip self-pairs (i == j) and invalid pairs (i > j) - pairs must have distinct geometries with i < j
        if i >= j:
            workid += nsweep_in
            continue

        # Get sorted local indices using manual indexing
        idx_i = world_id * max_shapes_per_world + i
        idx_j = world_id * max_shapes_per_world + j
        local_shape1 = sap_sort_index_in[idx_i]
        local_shape2 = sap_sort_index_in[idx_j]

        # Check for invalid indices (padding)
        if local_shape1 < 0 or local_shape2 < 0:
            workid += nsweep_in
            continue

        # Map to actual geometry indices
        shape1_tmp = world_index_map[world_slice_start + local_shape1]
        shape2_tmp = world_index_map[world_slice_start + local_shape2]

        # Skip if mapped to the same geometry (shouldn't happen, but defensive check)
        if shape1_tmp == shape2_tmp:
            workid += nsweep_in
            continue

        # Ensure canonical ordering
        shape1 = wp.min(shape1_tmp, shape2_tmp)
        shape2 = wp.max(shape1_tmp, shape2_tmp)

        # Get collision and world groups
        col_group1 = collision_group[shape1]
        col_group2 = collision_group[shape2]
        world1 = shape_world[shape1]
        world2 = shape_world[shape2]

        # Skip pairs where both geometries are global (world -1), unless we're in the dedicated -1 segment
        # The dedicated -1 segment is the last segment (world_id >= num_regular_worlds)
        is_dedicated_minus_one_segment = world_id >= num_regular_worlds
        if world1 == -1 and world2 == -1 and not is_dedicated_minus_one_segment:
            workid += nsweep_in
            continue

        # Check both world and collision groups
        if test_world_and_group_pair(world1, world2, col_group1, col_group2):
            _process_single_sap_pair(
                wp.vec2i(shape1, shape2),
                shape_bounding_box_lower,
                shape_bounding_box_upper,
                shape_contact_margin,
                candidate_pair,
                num_candidate_pair,
                max_candidate_pair,
                filter_pairs,
                num_filter_pairs,
            )

        workid += nsweep_in


class BroadPhaseSAP:
    """Sweep and Prune (SAP) broad phase collision detection.

    This class implements the sweep and prune algorithm for broad phase collision detection.
    It efficiently finds potentially colliding pairs of objects by sorting their bounding box
    projections along a fixed axis and checking for overlaps.
    """

    def __init__(
        self,
        shape_shape_world,
        shape_flags=None,
        sweep_thread_count_multiplier: int = 5,
        sort_type: SAPSortType = SAPSortType.SEGMENTED,
        tile_block_dim: int | None = None,
        device=None,
    ):
        """Initialize arrays for sweep and prune broad phase collision detection.

        Args:
            shape_shape_world: Array of world indices for each shape (numpy or warp array).
                Represents which world each shape belongs to for world-aware collision detection.
            shape_flags: Optional array of shape flags (numpy or warp array). If provided,
                only shapes with the COLLIDE_SHAPES flag will be included in collision checks.
                This efficiently filters out visual-only shapes.
            sweep_thread_count_multiplier: Multiplier for number of threads used in sweep phase
            sort_type: Type of sorting algorithm to use (SEGMENTED or TILE)
            tile_block_dim: Block dimension for tile-based sorting (optional, auto-calculated if None).
                If None, will be set to next power of 2 >= max_shapes_per_world, capped at 512.
                Minimum value is 32 (required by wp.tile_sort). If provided, will be clamped to [32, 1024].
            device: Device to store the precomputed arrays on. If None, uses CPU for numpy
                arrays or the device of the input warp array.
        """
        self.sweep_thread_count_multiplier = sweep_thread_count_multiplier
        self.sort_type = sort_type
        self.tile_block_dim_override = tile_block_dim  # Store user override if provided

        # Convert to numpy if it's a warp array
        if isinstance(shape_shape_world, wp.array):
            shape_shape_world_np = shape_shape_world.numpy()
            if device is None:
                device = shape_shape_world.device
        else:
            shape_shape_world_np = shape_shape_world
            if device is None:
                device = "cpu"

        # Convert shape_flags to numpy if provided
        shape_flags_np = None
        if shape_flags is not None:
            if isinstance(shape_flags, wp.array):
                shape_flags_np = shape_flags.numpy()
            else:
                shape_flags_np = shape_flags

        # Precompute the world map (filters out non-colliding shapes if flags provided)
        index_map_np, slice_ends_np = precompute_world_map(shape_shape_world_np, shape_flags_np)

        # Calculate number of regular worlds (excluding dedicated -1 segment at end)
        # Must be derived from filtered slices since precompute_world_map applies flags
        # slice_ends_np has length (num_filtered_worlds + 1), where +1 is the dedicated -1 segment
        num_regular_worlds = max(0, len(slice_ends_np) - 1)

        # Store as warp arrays
        self.world_index_map = wp.array(index_map_np, dtype=wp.int32, device=device)
        self.world_slice_ends = wp.array(slice_ends_np, dtype=wp.int32, device=device)

        # Calculate world information
        self.num_worlds = len(slice_ends_np)
        self.num_regular_worlds = int(num_regular_worlds)
        self.max_shapes_per_world = 0
        start_idx = 0
        for end_idx in slice_ends_np:
            num_shapes = end_idx - start_idx
            self.max_shapes_per_world = max(self.max_shapes_per_world, num_shapes)
            start_idx = end_idx

        # Create tile sort kernel if using tile-based sorting
        self.tile_sort_kernel = None
        if self.sort_type == SAPSortType.TILE:
            # Calculate block_dim: next power of 2 >= max_shapes_per_world, capped at 512
            if self.tile_block_dim_override is not None:
                self.tile_block_dim = max(32, min(self.tile_block_dim_override, 1024))
            else:
                block_dim = 1
                while block_dim < self.max_shapes_per_world:
                    block_dim *= 2
                self.tile_block_dim = max(32, min(block_dim, 512))

            # tile_size should match max_shapes_per_world (actual data size)
            # tile_block_dim is for thread block configuration and can be larger
            self.tile_size = int(self.max_shapes_per_world)

            self.tile_sort_kernel = _create_tile_sort_kernel(self.tile_size)

        # Allocate 1D arrays for per-world SAP data
        # Note: projection_lower and sort_index need 2x space for segmented sort scratch memory
        total_elements = int(self.num_worlds * self.max_shapes_per_world)
        self.sap_projection_lower = wp.zeros(2 * total_elements, dtype=wp.float32, device=device)
        self.sap_projection_upper = wp.zeros(total_elements, dtype=wp.float32, device=device)
        self.sap_sort_index = wp.zeros(2 * total_elements, dtype=wp.int32, device=device)
        self.sap_range = wp.zeros(total_elements, dtype=wp.int32, device=device)
        self.sap_cumulative_sum = wp.zeros(total_elements, dtype=wp.int32, device=device)

        # Segment indices for segmented sort (needed for graph capture)
        # [0, max_shapes_per_world, 2*max_shapes_per_world, ..., num_worlds*max_shapes_per_world]
        segment_indices_np = np.array(
            [i * self.max_shapes_per_world for i in range(self.num_worlds + 1)], dtype=np.int32
        )
        self.segment_indices = wp.array(segment_indices_np, dtype=wp.int32, device=device)

    def launch(
        self,
        shape_lower: wp.array(dtype=wp.vec3, ndim=1),  # Lower bounds of shape bounding boxes
        shape_upper: wp.array(dtype=wp.vec3, ndim=1),  # Upper bounds of shape bounding boxes
        shape_contact_margin: wp.array(dtype=float, ndim=1) | None,  # Optional per-shape contact margins
        shape_collision_group: wp.array(dtype=int, ndim=1),  # Collision group ID per box
        shape_shape_world: wp.array(dtype=int, ndim=1),  # World index per box
        shape_count: int,  # Number of active bounding boxes
        # Outputs
        candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),  # Array to store overlapping shape pairs
        num_candidate_pair: wp.array(dtype=int, ndim=1),
        device=None,  # Device to launch on
        filter_pairs: wp.array(dtype=wp.vec2i, ndim=1) | None = None,  # Sorted excluded pairs
        num_filter_pairs: int | None = None,
    ):
        """Launch the sweep and prune broad phase collision detection with per-world segmented sort.

        This method performs collision detection between geometries using a sweep and prune algorithm along a fixed axis.
        It processes each world independently using segmented sort, which is more efficient than global sorting
        when geometries are organized into separate worlds.

        Args:
            shape_lower: Array of lower bounds for each shape's AABB
            shape_upper: Array of upper bounds for each shape's AABB
            shape_contact_margin: Optional array of per-shape contact margins. If None or empty array,
                assumes AABBs are pre-expanded (margins = 0). If provided, margins are added during overlap checks.
            shape_collision_group: Array of collision group IDs for each shape. Positive values indicate
                groups that only collide with themselves (and with negative groups). Negative values indicate
                groups that collide with everything except their negative counterpart. Zero indicates no collisions.
            shape_shape_world: Array of world indices for each shape. Index -1 indicates global entities
                that collide with all worlds. Indices 0, 1, 2, ... indicate world-specific entities.
            shape_count: Number of active bounding boxes to check (not used in world-based approach)
            candidate_pair: Output array to store overlapping shape pairs
            num_candidate_pair: Output array to store number of overlapping pairs found
            device: Device to launch on. If None, uses the device of the input arrays.

        The method will populate candidate_pair with the indices of shape pairs whose AABBs overlap
        (with optional margin expansion), whose collision groups allow interaction, and whose worlds are
        compatible (same world or at least one is global). Pairs in filter_pairs (if provided) are excluded.
        The number of pairs found will be written to num_candidate_pair[0].
        """
        # TODO: Choose an optimal direction
        # random fixed direction
        direction = wp.vec3(0.5935, 0.7790, 0.1235)
        direction = wp.normalize(direction)

        max_candidate_pair = candidate_pair.shape[0]
        num_candidate_pair.zero_()

        if device is None:
            device = shape_lower.device

        # If no margins provided, pass empty array (kernel will use 0.0 margins)
        if shape_contact_margin is None:
            shape_contact_margin = wp.empty(0, dtype=wp.float32, device=device)

        # Exclusion filter: empty array and 0 when not provided or empty
        if filter_pairs is None or filter_pairs.shape[0] == 0:
            filter_pairs_arr = wp.empty(0, dtype=wp.vec2i, device=device)
            n_filter = 0
        else:
            filter_pairs_arr = filter_pairs
            n_filter = num_filter_pairs if num_filter_pairs is not None else filter_pairs.shape[0]

        # Project AABBs onto the sweep axis for each world
        wp.launch(
            kernel=_sap_project_kernel,
            dim=(self.num_worlds, self.max_shapes_per_world),
            inputs=[
                direction,
                shape_lower,
                shape_upper,
                shape_contact_margin,
                self.world_index_map,
                self.world_slice_ends,
                self.max_shapes_per_world,
                self.sap_projection_lower,
                self.sap_projection_upper,
                self.sap_sort_index,
            ],
            device=device,
        )

        # Perform segmented sort - each world is sorted independently
        # Two strategies: tile-based (faster for certain sizes) or segmented (more flexible)
        if self.sort_type == SAPSortType.TILE and self.tile_sort_kernel is not None:
            # Use tile-based sort with shared memory
            wp.launch_tiled(
                kernel=self.tile_sort_kernel,
                dim=self.num_worlds,
                inputs=[
                    self.sap_projection_lower,
                    self.sap_sort_index,
                    self.max_shapes_per_world,
                ],
                block_dim=self.tile_block_dim,
                device=device,
            )
        else:
            # Use segmented sort (default)
            # The count is the number of actual elements to sort (not including scratch space)
            wp.utils.segmented_sort_pairs(
                keys=self.sap_projection_lower,
                values=self.sap_sort_index,
                count=self.num_worlds * self.max_shapes_per_world,
                segment_start_indices=self.segment_indices,
            )

        # Compute range of overlapping geometries for each geometry in each world
        wp.launch(
            kernel=_sap_range_kernel,
            dim=(self.num_worlds, self.max_shapes_per_world),
            inputs=[
                self.world_slice_ends,
                self.max_shapes_per_world,
                self.sap_projection_lower,
                self.sap_projection_upper,
                self.sap_sort_index,
                self.sap_range,
            ],
            device=device,
        )

        # Compute cumulative sum of ranges
        wp.utils.array_scan(self.sap_range, self.sap_cumulative_sum, True)

        # Estimate number of sweep threads
        total_elements = self.num_worlds * self.max_shapes_per_world
        nsweep_in = int(self.sweep_thread_count_multiplier * total_elements)

        # Perform the sweep and generate candidate pairs
        wp.launch(
            kernel=_sap_broadphase_kernel,
            dim=nsweep_in,
            inputs=[
                shape_lower,
                shape_upper,
                shape_contact_margin,
                shape_collision_group,
                shape_shape_world,
                self.world_index_map,
                self.world_slice_ends,
                self.sap_sort_index,
                self.sap_cumulative_sum,
                self.num_worlds,
                self.max_shapes_per_world,
                nsweep_in,
                self.num_regular_worlds,
                filter_pairs_arr,
                n_filter,
            ],
            outputs=[
                candidate_pair,
                num_candidate_pair,
                max_candidate_pair,
            ],
            device=device,
        )
