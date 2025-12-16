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

from __future__ import annotations

import numpy as np
import warp as wp

from .broad_phase_common import check_aabb_overlap, precompute_world_map, test_world_and_group_pair, write_pair


@wp.kernel
def _nxn_broadphase_precomputed_pairs(
    # Input arrays
    shape_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
    shape_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
    shape_contact_margin: wp.array(
        dtype=float, ndim=1
    ),  # Optional per-shape contact margins (can be empty if AABBs pre-expanded)
    nxn_shape_pair: wp.array(dtype=wp.vec2i, ndim=1),
    # Output arrays
    candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
    num_candidate_pair: wp.array(dtype=int, ndim=1),  # Size one array
    max_candidate_pair: int,
):
    elementid = wp.tid()

    pair = nxn_shape_pair[elementid]
    shape1 = pair[0]
    shape2 = pair[1]

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


@wp.func
def _get_lower_triangular_indices(index: int, matrix_size: int) -> tuple[int, int]:
    total = (matrix_size * (matrix_size - 1)) >> 1
    if index >= total:
        # In Warp, we can't throw, so return an invalid pair
        return -1, -1

    low = int(0)
    high = matrix_size - 1
    while low < high:
        mid = (low + high) >> 1
        count = (mid * (2 * matrix_size - mid - 1)) >> 1
        if count <= index:
            low = mid + 1
        else:
            high = mid
    r = low - 1
    f = (r * (2 * matrix_size - r - 1)) >> 1
    c = (index - f) + r + 1
    return r, c


@wp.func
def _find_world_and_local_id(
    tid: int,
    world_cumsum_lower_tri: wp.array(dtype=int, ndim=1),
):
    """Binary search to find world ID and local ID from thread ID.

    Args:
        tid: Global thread ID
        world_cumsum_lower_tri: Cumulative sum of lower triangular elements per world

    Returns:
        tuple: (world_id, local_id) - World ID and local index within that world
    """
    num_worlds = world_cumsum_lower_tri.shape[0]

    # Find world_id using binary search
    # Declare as dynamic variables for loop mutation
    low = int(0)
    high = int(num_worlds - 1)
    world_id = int(0)

    while low <= high:
        mid = (low + high) >> 1
        if tid < world_cumsum_lower_tri[mid]:
            high = mid - 1
            world_id = mid
        else:
            low = mid + 1

    # Calculate local index within this world
    local_id = tid
    if world_id > 0:
        local_id = tid - world_cumsum_lower_tri[world_id - 1]

    return world_id, local_id


@wp.kernel
def _nxn_broadphase_kernel(
    # Input arrays
    shape_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
    shape_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
    shape_contact_margin: wp.array(
        dtype=float, ndim=1
    ),  # Optional per-shape contact margins (can be empty if AABBs pre-expanded)
    collision_group: wp.array(dtype=int, ndim=1),  # per-shape
    shape_world: wp.array(dtype=int, ndim=1),  # per-shape world indices
    world_cumsum_lower_tri: wp.array(dtype=int, ndim=1),  # Cumulative sum of lower tri elements per world
    world_slice_ends: wp.array(dtype=int, ndim=1),  # End indices of each world slice
    world_index_map: wp.array(dtype=int, ndim=1),  # Index map into source geometry
    num_regular_worlds: int,  # Number of regular world segments (excluding dedicated -1 segment)
    # Output arrays
    candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
    num_candidate_pair: wp.array(dtype=int, ndim=1),  # Size one array
    max_candidate_pair: int,
):
    tid = wp.tid()

    # Find which world this thread belongs to and the local index within that world
    world_id, local_id = _find_world_and_local_id(tid, world_cumsum_lower_tri)

    # Get the slice boundaries for this world in the index map
    world_slice_start = 0
    if world_id > 0:
        world_slice_start = world_slice_ends[world_id - 1]
    world_slice_end = world_slice_ends[world_id]

    # Number of geometries in this world
    num_shapes_in_world = world_slice_end - world_slice_start

    # Convert local_id to pair indices within the world
    local_shape1, local_shape2 = _get_lower_triangular_indices(local_id, num_shapes_in_world)

    # Map to actual geometry indices using the world_index_map
    shape1_tmp = world_index_map[world_slice_start + local_shape1]
    shape2_tmp = world_index_map[world_slice_start + local_shape2]

    # Ensure canonical ordering (smaller index first)
    # After mapping, the indices might not preserve local_shape1 < local_shape2 ordering
    shape1 = wp.min(shape1_tmp, shape2_tmp)
    shape2 = wp.max(shape1_tmp, shape2_tmp)

    # Get world and collision groups
    world1 = shape_world[shape1]
    world2 = shape_world[shape2]
    collision_group1 = collision_group[shape1]
    collision_group2 = collision_group[shape2]

    # Skip pairs where both geometries are global (world -1), unless we're in the dedicated -1 segment
    # The dedicated -1 segment is the last segment (world_id >= num_regular_worlds)
    is_dedicated_minus_one_segment = world_id >= num_regular_worlds
    if world1 == -1 and world2 == -1 and not is_dedicated_minus_one_segment:
        return

    # Check both world and collision groups
    if not test_world_and_group_pair(world1, world2, collision_group1, collision_group2):
        return

    # Check if margins are provided (empty array means AABBs are pre-expanded)
    margin1 = 0.0
    margin2 = 0.0
    if shape_contact_margin.shape[0] > 0:
        margin1 = shape_contact_margin[shape1]
        margin2 = shape_contact_margin[shape2]

    # Check AABB overlap
    if check_aabb_overlap(
        shape_bounding_box_lower[shape1],
        shape_bounding_box_upper[shape1],
        margin1,
        shape_bounding_box_lower[shape2],
        shape_bounding_box_upper[shape2],
        margin2,
    ):
        write_pair(
            wp.vec2i(shape1, shape2),
            candidate_pair,
            num_candidate_pair,
            max_candidate_pair,
        )


class BroadPhaseAllPairs:
    """A broad phase collision detection class that performs N x N collision checks between all geometry pairs.

    This class performs collision detection between all possible pairs of geometries by checking for
    axis-aligned bounding box (AABB) overlaps. It uses a lower triangular matrix approach to avoid
    checking each pair twice.

    The collision checks take into account per-geometry cutoff distances and collision groups. Two geometries
    will only be considered as a candidate pair if:
    1. Their AABBs overlap when expanded by their cutoff distances
    2. Their collision groups allow interaction (determined by test_group_pair())

    The class outputs an array of candidate collision pairs that need more detailed narrow phase collision
    checking.
    """

    def __init__(self, shape_world, shape_flags=None, device=None):
        """Initialize the broad phase with world ID information.

        Args:
            shape_world: Array of world IDs (numpy or warp array).
                Positive/zero values represent distinct worlds, negative values represent
                shared entities that belong to all worlds.
            shape_flags: Optional array of shape flags (numpy or warp array). If provided,
                only shapes with the COLLIDE_SHAPES flag will be included in collision checks.
                This efficiently filters out visual-only shapes.
            device: Device to store the precomputed arrays on. If None, uses CPU for numpy
                arrays or the device of the input warp array.
        """
        # Convert to numpy if it's a warp array
        if isinstance(shape_world, wp.array):
            shape_world_np = shape_world.numpy()
            if device is None:
                device = shape_world.device
        else:
            shape_world_np = shape_world
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
        index_map_np, slice_ends_np = precompute_world_map(shape_world_np, shape_flags_np)

        # Calculate number of regular worlds (excluding dedicated -1 segment at end)
        # Must be derived from filtered slices since precompute_world_map applies flags
        # slice_ends_np has length (num_filtered_worlds + 1), where +1 is the dedicated -1 segment
        num_regular_worlds = max(0, len(slice_ends_np) - 1)

        # Calculate cumulative sum of lower triangular elements per world
        # For each world, compute n*(n-1)/2 where n is the number of geometries in that world
        num_worlds = len(slice_ends_np)
        world_cumsum_lower_tri_np = np.zeros(num_worlds, dtype=np.int32)

        start_idx = 0
        cumsum = 0
        for world_idx in range(num_worlds):
            end_idx = slice_ends_np[world_idx]
            # Number of geometries in this world (including shared geometries)
            num_geoms_in_world = end_idx - start_idx
            # Number of lower triangular elements for this world
            num_lower_tri = (num_geoms_in_world * (num_geoms_in_world - 1)) // 2
            cumsum += num_lower_tri
            world_cumsum_lower_tri_np[world_idx] = cumsum
            start_idx = end_idx

        # Store as warp arrays
        self.world_index_map = wp.array(index_map_np, dtype=wp.int32, device=device)
        self.world_slice_ends = wp.array(slice_ends_np, dtype=wp.int32, device=device)
        self.world_cumsum_lower_tri = wp.array(world_cumsum_lower_tri_np, dtype=wp.int32, device=device)

        # Store total number of kernel threads needed (last element of cumsum)
        self.num_kernel_threads = int(world_cumsum_lower_tri_np[-1]) if num_worlds > 0 else 0

        # Store number of regular worlds (for distinguishing dedicated -1 segment)
        self.num_regular_worlds = int(num_regular_worlds)

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
    ):
        """Launch the N x N broad phase collision detection.

        This method performs collision detection between all possible pairs of geometries by checking for
        axis-aligned bounding box (AABB) overlaps. It uses a lower triangular matrix approach to avoid
        checking each pair twice.

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
            shape_count: Number of active bounding boxes to check
            candidate_pair: Output array to store overlapping shape pairs
            num_candidate_pair: Output array to store number of overlapping pairs found
            device: Device to launch on. If None, uses the device of the input arrays.

        The method will populate candidate_pair with the indices of shape pairs (i,j) where i < j whose AABBs overlap
        (with optional margin expansion), whose collision groups allow interaction, and whose world indices are
        compatible (same world or at least one is global). The number of pairs found will be written to num_candidate_pair[0].
        """
        max_candidate_pair = candidate_pair.shape[0]

        num_candidate_pair.zero_()

        if device is None:
            device = shape_lower.device

        # If no margins provided, pass empty array (kernel will use 0.0 margins)
        if shape_contact_margin is None:
            shape_contact_margin = wp.empty(0, dtype=wp.float32, device=device)

        # Launch with the precomputed number of kernel threads
        wp.launch(
            _nxn_broadphase_kernel,
            dim=self.num_kernel_threads,
            inputs=[
                shape_lower,
                shape_upper,
                shape_contact_margin,
                shape_collision_group,
                shape_shape_world,
                self.world_cumsum_lower_tri,
                self.world_slice_ends,
                self.world_index_map,
                self.num_regular_worlds,
            ],
            outputs=[candidate_pair, num_candidate_pair, max_candidate_pair],
            device=device,
        )


class BroadPhaseExplicit:
    """A broad phase collision detection class that only checks explicitly provided geometry pairs.

    This class performs collision detection only between geometry pairs that are explicitly specified,
    rather than checking all possible pairs. This can be more efficient when the set of potential
    collision pairs is known ahead of time.

    The class checks for axis-aligned bounding box (AABB) overlaps between the specified geometry pairs,
    taking into account per-geometry cutoff distances.
    """

    def __init__(self):
        pass

    def launch(
        self,
        shape_lower: wp.array(dtype=wp.vec3, ndim=1),  # Lower bounds of shape bounding boxes
        shape_upper: wp.array(dtype=wp.vec3, ndim=1),  # Upper bounds of shape bounding boxes
        shape_contact_margin: wp.array(dtype=float, ndim=1) | None,  # Optional per-shape contact margins
        shape_pairs: wp.array(dtype=wp.vec2i, ndim=1),  # Precomputed pairs to check
        shape_pair_count: int,
        # Outputs
        candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),  # Array to store overlapping shape pairs
        num_candidate_pair: wp.array(dtype=int, ndim=1),
        device=None,  # Device to launch on
    ):
        """Launch the explicit pairs broad phase collision detection.

        This method checks for AABB overlaps only between explicitly specified shape pairs,
        rather than checking all possible pairs. It populates the candidate_pair array with
        indices of shape pairs whose AABBs overlap.

        Args:
            shape_lower: Array of lower bounds for shape bounding boxes
            shape_upper: Array of upper bounds for shape bounding boxes
            shape_contact_margin: Optional array of per-shape contact margins. If None or empty array,
                assumes AABBs are pre-expanded (margins = 0). If provided, margins are added during overlap checks.
            shape_pairs: Array of precomputed shape pairs to check
            shape_pair_count: Number of shape pairs to check
            candidate_pair: Output array to store overlapping shape pairs
            num_candidate_pair: Output array to store number of overlapping pairs found
            device: Device to launch on. If None, uses the device of the input arrays.

        The method will populate candidate_pair with the indices of shape pairs whose AABBs overlap
        (with optional margin expansion), but only checking the explicitly provided pairs.
        """

        max_candidate_pair = candidate_pair.shape[0]

        num_candidate_pair.zero_()

        if device is None:
            device = shape_lower.device

        # If no margins provided, pass empty array (kernel will use 0.0 margins)
        if shape_contact_margin is None:
            shape_contact_margin = wp.empty(0, dtype=wp.float32, device=device)

        wp.launch(
            kernel=_nxn_broadphase_precomputed_pairs,
            dim=shape_pair_count,
            inputs=[
                shape_lower,
                shape_upper,
                shape_contact_margin,
                shape_pairs,
                candidate_pair,
                num_candidate_pair,
                max_candidate_pair,
            ],
            device=device,
        )
