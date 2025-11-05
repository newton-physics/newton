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

import numpy as np
import warp as wp

from .broad_phase_common import check_aabb_overlap, precompute_world_map, test_world_and_group_pair, write_pair


@wp.kernel
def _nxn_broadphase_precomputed_pairs(
    # Input arrays
    geom_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
    geom_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
    geom_cutoff: wp.array(dtype=float, ndim=1),  # per-geom (take the max)
    nxn_geom_pair: wp.array(dtype=wp.vec2i, ndim=1),
    # Output arrays
    candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
    num_candidate_pair: wp.array(dtype=int, ndim=1),  # Size one array
    max_candidate_pair: int,
):
    elementid = wp.tid()

    pair = nxn_geom_pair[elementid]
    geom1 = pair[0]
    geom2 = pair[1]

    if check_aabb_overlap(
        geom_bounding_box_lower[geom1],
        geom_bounding_box_upper[geom1],
        geom_cutoff[geom1],
        geom_bounding_box_lower[geom2],
        geom_bounding_box_upper[geom2],
        geom_cutoff[geom2],
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
    geom_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
    geom_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
    geom_cutoff: wp.array(dtype=float, ndim=1),  # per-geom (take the max)
    collision_group: wp.array(dtype=int, ndim=1),  # per-geom
    shape_world: wp.array(dtype=int, ndim=1),  # per-geom world indices
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
    num_geoms_in_world = world_slice_end - world_slice_start

    # Convert local_id to pair indices within the world
    local_geom1, local_geom2 = _get_lower_triangular_indices(local_id, num_geoms_in_world)

    # Map to actual geometry indices using the world_index_map
    geom1_tmp = world_index_map[world_slice_start + local_geom1]
    geom2_tmp = world_index_map[world_slice_start + local_geom2]

    # Ensure canonical ordering (smaller index first)
    # After mapping, the indices might not preserve local_geom1 < local_geom2 ordering
    geom1 = wp.min(geom1_tmp, geom2_tmp)
    geom2 = wp.max(geom1_tmp, geom2_tmp)

    # Get world and collision groups
    world1 = shape_world[geom1]
    world2 = shape_world[geom2]
    collision_group1 = collision_group[geom1]
    collision_group2 = collision_group[geom2]

    # Skip pairs where both geometries are global (world -1), unless we're in the dedicated -1 segment
    # The dedicated -1 segment is the last segment (world_id >= num_regular_worlds)
    is_dedicated_minus_one_segment = world_id >= num_regular_worlds
    if world1 == -1 and world2 == -1 and not is_dedicated_minus_one_segment:
        return

    # Check both world and collision groups
    if not test_world_and_group_pair(world1, world2, collision_group1, collision_group2):
        return

    # Check AABB overlap
    if check_aabb_overlap(
        geom_bounding_box_lower[geom1],
        geom_bounding_box_upper[geom1],
        geom_cutoff[geom1],
        geom_bounding_box_lower[geom2],
        geom_bounding_box_upper[geom2],
        geom_cutoff[geom2],
    ):
        write_pair(
            wp.vec2i(geom1, geom2),
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

    def __init__(self, geom_world, geom_flags=None, device=None):
        """Initialize the broad phase with world ID information.

        Args:
            geom_world: Array of world IDs (numpy or warp array).
                Positive/zero values represent distinct worlds, negative values represent
                shared entities that belong to all worlds.
            geom_flags: Optional array of shape flags (numpy or warp array). If provided,
                only shapes with the COLLIDE_SHAPES flag will be included in collision checks.
                This efficiently filters out visual-only shapes.
            device: Device to store the precomputed arrays on. If None, uses CPU for numpy
                arrays or the device of the input warp array.
        """
        # Convert to numpy if it's a warp array
        if isinstance(geom_world, wp.array):
            geom_world_np = geom_world.numpy()
            if device is None:
                device = geom_world.device
        else:
            geom_world_np = geom_world
            if device is None:
                device = "cpu"

        # Convert geom_flags to numpy if provided
        geom_flags_np = None
        if geom_flags is not None:
            if isinstance(geom_flags, wp.array):
                geom_flags_np = geom_flags.numpy()
            else:
                geom_flags_np = geom_flags

        # Precompute the world map (filters out non-colliding shapes if flags provided)
        index_map_np, slice_ends_np = precompute_world_map(geom_world_np, geom_flags_np)

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
        geom_lower: wp.array(dtype=wp.vec3, ndim=1),  # Lower bounds of geometry bounding boxes
        geom_upper: wp.array(dtype=wp.vec3, ndim=1),  # Upper bounds of geometry bounding boxes
        geom_cutoffs: wp.array(dtype=float, ndim=1),  # Cutoff distance per geometry box
        geom_collision_group: wp.array(dtype=int, ndim=1),  # Collision group ID per box
        geom_shape_world: wp.array(dtype=int, ndim=1),  # World index per box
        geom_count: int,  # Number of active bounding boxes
        # Outputs
        candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),  # Array to store overlapping geometry pairs
        num_candidate_pair: wp.array(dtype=int, ndim=1),
        device=None,  # Device to launch on
    ):
        """Launch the N x N broad phase collision detection.

        This method performs collision detection between all possible pairs of geometries by checking for
        axis-aligned bounding box (AABB) overlaps. It uses a lower triangular matrix approach to avoid
        checking each pair twice.

        Args:
            geom_lower: Array of lower bounds for each geometry's AABB
            geom_upper: Array of upper bounds for each geometry's AABB
            geom_cutoffs: Array of cutoff distances for each geometry
            geom_collision_group: Array of collision group IDs for each geometry. Positive values indicate
                groups that only collide with themselves (and with negative groups). Negative values indicate
                groups that collide with everything except their negative counterpart. Zero indicates no collisions.
            geom_shape_world: Array of world indices for each geometry. Index -1 indicates global entities
                that collide with all worlds. Indices 0, 1, 2, ... indicate world-specific entities.
            geom_count: Number of active bounding boxes to check
            candidate_pair: Output array to store overlapping geometry pairs
            num_candidate_pair: Output array to store number of overlapping pairs found
            device: Device to launch on. If None, uses the device of the input arrays.

        The method will populate candidate_pair with the indices of geometry pairs (i,j) where i < j whose AABBs overlap
        when expanded by their cutoff distances, whose collision groups allow interaction, and whose world indices
        are compatible (same world or at least one is global). The number of pairs found will be written to
        num_candidate_pair[0].
        """
        max_candidate_pair = candidate_pair.shape[0]

        num_candidate_pair.zero_()

        if device is None:
            device = geom_lower.device

        # Launch with the precomputed number of kernel threads
        wp.launch(
            _nxn_broadphase_kernel,
            dim=self.num_kernel_threads,
            inputs=[
                geom_lower,
                geom_upper,
                geom_cutoffs,
                geom_collision_group,
                geom_shape_world,
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
        geom_lower: wp.array(dtype=wp.vec3, ndim=1),  # Lower bounds of geometry bounding boxes
        geom_upper: wp.array(dtype=wp.vec3, ndim=1),  # Upper bounds of geometry bounding boxes
        geom_cutoffs: wp.array(dtype=float, ndim=1),  # Cutoff distance per geometry box
        geom_pairs: wp.array(dtype=wp.vec2i, ndim=1),  # Precomputed pairs to check
        geom_pair_count: int,
        # Outputs
        candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),  # Array to store overlapping geometry pairs
        num_candidate_pair: wp.array(dtype=int, ndim=1),
        device=None,  # Device to launch on
    ):
        """Launch the explicit pairs broad phase collision detection.

        This method checks for AABB overlaps only between explicitly specified geometry pairs,
        rather than checking all possible pairs. It populates the candidate_pair array with
        indices of geometry pairs whose AABBs overlap when expanded by their cutoff distances.

        Args:
            geom_lower: Array of lower bounds for geometry bounding boxes
            geom_upper: Array of upper bounds for geometry bounding boxes
            geom_cutoffs: Array of cutoff distances per geometry box
            geom_pairs: Array of precomputed geometry pairs to check
            geom_pair_count: Number of geometry pairs to check
            candidate_pair: Output array to store overlapping geometry pairs
            num_candidate_pair: Output array to store number of overlapping pairs found
            device: Device to launch on. If None, uses the device of the input arrays.

        The method will populate candidate_pair with the indices of geometry pairs whose AABBs overlap
        when expanded by their cutoff distances, but only checking the explicitly provided pairs.
        """

        max_candidate_pair = candidate_pair.shape[0]

        num_candidate_pair.zero_()

        if device is None:
            device = geom_lower.device

        wp.launch(
            kernel=_nxn_broadphase_precomputed_pairs,
            dim=geom_pair_count,
            inputs=[
                geom_lower,
                geom_upper,
                geom_cutoffs,
                geom_pairs,
                candidate_pair,
                num_candidate_pair,
                max_candidate_pair,
            ],
            device=device,
        )
