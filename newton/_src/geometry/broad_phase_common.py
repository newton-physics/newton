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

"""Common utilities for broad phase collision detection.

Provides shared functions for AABB overlap tests, world/group filtering,
and pair output used by both NxN and SAP broad phase implementations.
"""

from typing import Any

import numpy as np
import warp as wp

from .flags import ShapeFlags


@wp.func
def check_aabb_overlap(
    box1_lower: wp.vec3,
    box1_upper: wp.vec3,
    box1_cutoff: float,
    box2_lower: wp.vec3,
    box2_upper: wp.vec3,
    box2_cutoff: float,
) -> bool:
    cutoff_combined = max(box1_cutoff, box2_cutoff)
    return (
        box1_lower[0] <= box2_upper[0] + cutoff_combined
        and box1_upper[0] >= box2_lower[0] - cutoff_combined
        and box1_lower[1] <= box2_upper[1] + cutoff_combined
        and box1_upper[1] >= box2_lower[1] - cutoff_combined
        and box1_lower[2] <= box2_upper[2] + cutoff_combined
        and box1_upper[2] >= box2_lower[2] - cutoff_combined
    )


@wp.func
def binary_search(values: wp.array(dtype=Any), value: Any, lower: int, upper: int) -> int:
    while lower < upper:
        mid = (lower + upper) >> 1
        if values[mid] > value:
            upper = mid
        else:
            lower = mid + 1

    return upper


@wp.func
def _vec2i_less(p: wp.vec2i, q: wp.vec2i) -> bool:
    """Lexicographic less-than for vec2i.

    Args:
        p: First vector to compare.
        q: Second vector to compare.

    Returns:
        True if p < q lexicographically, i.e. p[0] < q[0] or (p[0] == q[0] and p[1] < q[1]).
    """
    if p[0] < q[0]:
        return True
    if p[0] > q[0]:
        return False
    return p[1] < q[1]


@wp.func
def _vec2i_equal(p: wp.vec2i, q: wp.vec2i) -> bool:
    """Element-wise equality for vec2i.

    Args:
        p: First vector to compare.
        q: Second vector to compare.

    Returns:
        True if p[0] == q[0] and p[1] == q[1].
    """
    return p[0] == q[0] and p[1] == q[1]


@wp.func
def is_pair_excluded(
    pair: wp.vec2i,
    filter_pairs: wp.array(dtype=wp.vec2i, ndim=1),
    num_filter_pairs: int,
) -> bool:
    """Check whether a shape pair is in the sorted exclusion list via binary search.

    Args:
        pair: Canonical shape pair (min, max) to look up.
        filter_pairs: Lexicographically sorted array of excluded shape pairs.
            Each entry must be canonical (min, max).
        num_filter_pairs: Number of valid entries in ``filter_pairs``.

    Returns:
        True if ``pair`` is found in ``filter_pairs``, False otherwise.
        Returns False immediately when ``num_filter_pairs`` is 0.
    """
    if num_filter_pairs <= 0:
        return False
    low = int(0)
    high = num_filter_pairs - 1
    while low <= high:
        mid = (low + high) >> 1
        m = filter_pairs[mid]
        if _vec2i_equal(pair, m):
            return True
        if _vec2i_less(pair, m):
            high = mid - 1
        else:
            low = mid + 1
    return False


@wp.func
def write_pair(
    pair: wp.vec2i,
    candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
    candidate_pair_count: wp.array(dtype=int, ndim=1),  # Size one array
    max_candidate_pair: int,
):
    pairid = wp.atomic_add(candidate_pair_count, 0, 1)

    if pairid >= max_candidate_pair:
        return

    candidate_pair[pairid] = pair


# Collision filtering
@wp.func
def test_group_pair(group_a: int, group_b: int) -> bool:
    """Test if two collision groups should interact.

    Args:
        group_a: First collision group ID. Positive values indicate groups that only collide with themselves (and with negative groups).
                Negative values indicate groups that collide with everything except their negative counterpart.
                Zero indicates no collisions.
        group_b: Second collision group ID. Same meaning as group_a.

    Returns:
        bool: True if the groups should collide, False if they should not.
    """
    if group_a == 0 or group_b == 0:
        return False
    if group_a > 0:
        return group_a == group_b or group_b < 0
    if group_a < 0:
        return group_a != group_b


@wp.func
def test_world_and_group_pair(world_a: int, world_b: int, collision_group_a: int, collision_group_b: int) -> bool:
    """Test if two entities should collide based on world indices and collision groups.

    World indices define which simulation world an entity belongs to:
    - Index -1: Global entities that collide with all worlds
    - Indices 0, 1, 2, ...: World-specific entities

    Collision rules:
    1. Entities from different worlds (except -1) do not collide
    2. Global entities (index -1) collide with all worlds
    3. Within the same world, collision groups determine interactions

    Args:
        world_a: World index of first entity
        world_b: World index of second entity
        collision_group_a: Collision group of first entity
        collision_group_b: Collision group of second entity

    Returns:
        bool: True if the entities should collide, False otherwise
    """
    # Check world indices first
    if world_a != -1 and world_b != -1 and world_a != world_b:
        return False

    # If same world or at least one is global (-1), check collision groups
    return test_group_pair(collision_group_a, collision_group_b)


def precompute_world_map(shape_world: np.ndarray, shape_flags: np.ndarray | None = None):
    """Precompute an index map that groups shapes by world ID with shared shapes.

    This method creates an index mapping where shapes belonging to the same world
    (non-negative world ID) are grouped together, and shared shapes
    (world ID -1) are appended to each world's slice.

    A dedicated segment at the end contains only world -1 objects for handling
    -1 vs -1 collisions without duplication.

    Optionally filters out shapes that should not participate in collision detection
    based on their flags (e.g., visual-only shapes without COLLIDE_SHAPES flag).

    Args:
        shape_world: Array of world IDs. Must contain only:
            - World ID -1: Global/shared entities that collide with all worlds
            - World IDs >= 0: World-specific entities (0, 1, 2, ...)
            World IDs < -1 are not supported and will raise ValueError.
        shape_flags: Optional array of shape flags. If provided, only shapes with the
            COLLIDE_SHAPES flag (bit 1) set will be included in the output map. This allows
            efficient filtering of visual-only shapes that shouldn't participate in collision.

    Raises:
        ValueError: If shape_flags is provided and lengths don't match shape_world, or if
            any world IDs are < -1.

    Returns:
        tuple: (index_map, slice_ends)
            - index_map: 1D array of indices into shape_world, arranged such that:
                * Each regular world's indices are followed by all world -1 (shared) indices
                * A final segment contains only world -1 (shared) indices
                Only includes shapes that pass the collision flag filter.
            - slice_ends: 1D array containing the end index (exclusive) of each world's slice
                in the index_map (including the dedicated -1 segment at the end)
    """
    # Ensure shape_world is a numpy array (might be a list from builder)
    if not isinstance(shape_world, np.ndarray):
        shape_world = np.array(shape_world)

    # Filter out non-colliding shapes if flags are provided
    if shape_flags is not None:
        # Ensure shape_flags is also a numpy array
        if not isinstance(shape_flags, np.ndarray):
            shape_flags = np.array(shape_flags)
        if shape_flags.shape[0] != shape_world.shape[0]:
            raise ValueError("shape_flags and shape_world must have the same length")
        colliding_mask = (shape_flags & ShapeFlags.COLLIDE_SHAPES) != 0
    else:
        colliding_mask = np.ones(len(shape_world), dtype=bool)

    # Apply collision filter to get valid indices
    valid_indices = np.where(colliding_mask)[0]

    # Work with filtered world IDs
    filtered_world_ids = shape_world[valid_indices]

    # Validate world IDs: only -1, 0, 1, 2, ... are allowed
    invalid_worlds = shape_world[(shape_world < -1)]
    if len(invalid_worlds) > 0:
        unique_invalid = np.unique(invalid_worlds)
        raise ValueError(
            f"Invalid world IDs detected: {unique_invalid.tolist()}. "
            f"Only world ID -1 (global/shared) and non-negative IDs (0, 1, 2, ...) are supported."
        )

    # Count world -1 (global entities) in filtered set -> num_shared
    # Only world -1 is treated as shared; kernels special-case -1 for deduplication
    negative_mask = filtered_world_ids == -1
    num_shared = np.sum(negative_mask)

    # Get indices of world -1 (shared) entries in the valid set
    shared_local_indices = np.where(negative_mask)[0]
    # Map back to original shape indices
    shared_indices = valid_indices[shared_local_indices]

    # Count how many distinct positive (or zero) world IDs are in filtered set -> world_count
    # Get unique positive/zero world IDs
    positive_mask = filtered_world_ids >= 0
    positive_world_ids = filtered_world_ids[positive_mask]
    unique_worlds = np.unique(positive_world_ids)
    world_count = len(unique_worlds)

    # Calculate total size of result
    # Each world gets its own indices + all shared indices
    # Plus one additional segment at the end with only shared indices
    num_positive = np.sum(positive_mask)
    total_size = num_positive + (num_shared * world_count) + num_shared

    # Allocate output arrays (world_count + 1 to include dedicated -1 segment)
    index_map = np.empty(total_size, dtype=np.int32)
    slice_ends = np.empty(world_count + 1, dtype=np.int32)

    # Build the index map
    current_pos = 0
    for world_idx, world_id in enumerate(unique_worlds):
        # Get indices for this world in the filtered set
        world_local_indices = np.where(filtered_world_ids == world_id)[0]
        # Map back to original shape indices
        world_indices = valid_indices[world_local_indices]
        num_world_shapes = len(world_indices)

        # Copy world-specific indices (using original shape indices)
        index_map[current_pos : current_pos + num_world_shapes] = world_indices
        current_pos += num_world_shapes

        # Append shared (negative) indices (using original shape indices)
        index_map[current_pos : current_pos + num_shared] = shared_indices
        current_pos += num_shared

        # Store the end position of this slice
        slice_ends[world_idx] = current_pos

    # Add dedicated segment at the end with only world -1 objects
    index_map[current_pos : current_pos + num_shared] = shared_indices
    current_pos += num_shared
    slice_ends[world_count] = current_pos

    return index_map, slice_ends
