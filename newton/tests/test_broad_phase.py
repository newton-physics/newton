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

import unittest
from math import sqrt

import numpy as np
import warp as wp

from newton._src.geometry.broad_phase_sap import SAPSortType
from newton._src.geometry.flags import ShapeFlags
from newton.geometry import BroadPhaseAllPairs, BroadPhaseExplicit, BroadPhaseSAP

# NOTE: The test_group_pair and test_world_and_group_pair functions below are copied
# from newton._src.geometry.broad_phase_common because they need to be available as
# host-side Python functions for testing/verification. The original functions are
# decorated with @wp.func for use in GPU kernels and can be called from host code,
# but the overhead is huge. This duplication allows us to verify that the GPU collision
# filtering logic matches the expected behavior efficiently without the overhead of
# calling @wp.func decorated functions from the host.


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
    return False


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


def check_aabb_overlap_host(
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


def find_overlapping_pairs_np(
    box_lower: np.ndarray,
    box_upper: np.ndarray,
    cutoff: np.ndarray,
    collision_group: np.ndarray,
    shape_world: np.ndarray | None = None,
    shape_flags: np.ndarray | None = None,
):
    """
    Brute-force n^2 algorithm to find all overlapping bounding box pairs.
    Each box is axis-aligned, defined by min (lower) and max (upper) corners.
    Returns a list of (i, j) pairs with i < j, where boxes i and j overlap.

    Args:
        shape_flags: Optional array of shape flags. If provided, only geometries with
            COLLIDE_SHAPES flag set will participate in collision detection.
    """
    n = box_lower.shape[0]
    pairs = []
    for i in range(n):
        # Skip if shape_flags is provided and this geometry doesn't have COLLIDE_SHAPES flag
        if shape_flags is not None:
            if (shape_flags[i] & ShapeFlags.COLLIDE_SHAPES) == 0:
                continue

        for j in range(i + 1, n):
            # Skip if shape_flags is provided and this geometry doesn't have COLLIDE_SHAPES flag
            if shape_flags is not None:
                if (shape_flags[j] & ShapeFlags.COLLIDE_SHAPES) == 0:
                    continue

            # Check world and collision group compatibility
            if shape_world is not None:
                world_i = int(shape_world[i])
                world_j = int(shape_world[j])
                group_i = int(collision_group[i])
                group_j = int(collision_group[j])
                # Use the combined test function
                if not test_world_and_group_pair(world_i, world_j, group_i, group_j):
                    continue
            else:
                # No world information, just check collision groups
                if not test_group_pair(int(collision_group[i]), int(collision_group[j])):
                    continue

            # Check for overlap in all three axes
            cutoff_combined = max(cutoff[i], cutoff[j])
            if (
                box_lower[i, 0] <= box_upper[j, 0] + cutoff_combined
                and box_upper[i, 0] >= box_lower[j, 0] - cutoff_combined
                and box_lower[i, 1] <= box_upper[j, 1] + cutoff_combined
                and box_upper[i, 1] >= box_lower[j, 1] - cutoff_combined
                and box_lower[i, 2] <= box_upper[j, 2] + cutoff_combined
                and box_upper[i, 2] >= box_lower[j, 2] - cutoff_combined
            ):
                pairs.append((i, j))
    return pairs


class TestBroadPhase(unittest.TestCase):
    def test_nxn_broadphase(self):
        verbose = False

        # Create random bounding boxes in min-max format
        ngeom = 30

        # Generate random centers and sizes using the new Generator API
        rng = np.random.Generator(np.random.PCG64(42))

        centers = rng.random((ngeom, 3)) * 3.0
        sizes = rng.random((ngeom, 3)) * 2.0  # box half-extent up to 1.0 in each direction
        geom_bounding_box_lower = centers - sizes
        geom_bounding_box_upper = centers + sizes

        np_geom_cutoff = np.zeros(ngeom, dtype=np.float32)
        num_groups = 5  # The zero group does not need to be counted
        np_collision_group = rng.integers(1, num_groups + 1, size=ngeom, dtype=np.int32)

        # Overwrite n random elements with -1
        minus_one_count = int(sqrt(ngeom))  # Number of elements to overwrite with -1
        random_indices = rng.choice(ngeom, size=minus_one_count, replace=False)
        np_collision_group[random_indices] = -1

        pairs_np = find_overlapping_pairs_np(
            geom_bounding_box_lower, geom_bounding_box_upper, np_geom_cutoff, np_collision_group
        )

        if verbose:
            print("Numpy contact pairs:")
            for i, pair in enumerate(pairs_np):
                body_a, body_b = pair
                group_a = np_collision_group[body_a]
                group_b = np_collision_group[body_b]
                print(f"  Pair {i}: bodies ({body_a}, {body_b}) with collision groups ({group_a}, {group_b})")

        # The number of elements in the lower triangular part of an n x n matrix (excluding the diagonal)
        # is given by n * (n - 1) // 2
        num_lower_tri_elements = ngeom * (ngeom - 1) // 2

        geom_lower = wp.array(geom_bounding_box_lower, dtype=wp.vec3)
        geom_upper = wp.array(geom_bounding_box_upper, dtype=wp.vec3)
        geom_cutoff = wp.array(np_geom_cutoff)
        collision_group = wp.array(np_collision_group)
        candidate_pair_count = wp.array(
            [
                0,
            ],
            dtype=wp.int32,
        )
        max_candidate_pair = num_lower_tri_elements
        candidate_pair = wp.array(np.zeros((max_candidate_pair, 2), dtype=wp.int32), dtype=wp.vec2i)

        # Create shape world array with all shapes in global world (-1)
        shape_world = wp.array(np.full(ngeom, 0, dtype=np.int32), dtype=wp.int32)

        # Initialize BroadPhaseAllPairs with shape_world (which represents world grouping)
        nxn_broadphase = BroadPhaseAllPairs(shape_world)

        nxn_broadphase.launch(
            geom_lower,
            geom_upper,
            geom_cutoff,
            collision_group,
            shape_world,
            ngeom,
            candidate_pair,
            candidate_pair_count,
        )

        wp.synchronize()

        pairs_wp = candidate_pair.numpy()
        candidate_pair_count = candidate_pair_count.numpy()[0]

        if verbose:
            print("Warp contact pairs:")
            for i in range(candidate_pair_count):
                pair = pairs_wp[i]
                body_a, body_b = pair[0], pair[1]
                group_a = np_collision_group[body_a]
                group_b = np_collision_group[body_b]
                print(f"  Pair {i}: bodies ({body_a}, {body_b}) with collision groups ({group_a}, {group_b})")

            print("Checking if bounding boxes actually overlap:")
            for i in range(candidate_pair_count):
                pair = pairs_wp[i]
                body_a, body_b = pair[0], pair[1]

                # Get bounding boxes for both bodies
                box_a_lower = geom_bounding_box_lower[body_a]
                box_a_upper = geom_bounding_box_upper[body_a]
                box_b_lower = geom_bounding_box_lower[body_b]
                box_b_upper = geom_bounding_box_upper[body_b]

                # Get cutoffs for both bodies
                cutoff_a = np_geom_cutoff[body_a]
                cutoff_b = np_geom_cutoff[body_b]

                # Check overlap using the function
                overlap = check_aabb_overlap_host(
                    wp.vec3(box_a_lower[0], box_a_lower[1], box_a_lower[2]),
                    wp.vec3(box_a_upper[0], box_a_upper[1], box_a_upper[2]),
                    cutoff_a,
                    wp.vec3(box_b_lower[0], box_b_lower[1], box_b_lower[2]),
                    wp.vec3(box_b_upper[0], box_b_upper[1], box_b_upper[2]),
                    cutoff_b,
                )

                print(f"  Pair {i}: bodies ({body_a}, {body_b}) - overlap: {overlap}")

        if len(pairs_np) != candidate_pair_count:
            print(f"len(pairs_np)={len(pairs_np)}, candidate_pair_count={candidate_pair_count}")
            assert len(pairs_np) == candidate_pair_count

        # Ensure every element in pairs_wp is also present in pairs_np
        pairs_np_set = {tuple(pair) for pair in pairs_np}
        for pair in pairs_wp[:candidate_pair_count]:
            assert tuple(pair) in pairs_np_set, f"Pair {tuple(pair)} from Warp not found in numpy pairs"

        if verbose:
            print(len(pairs_np))

    def test_nxn_broadphase_multiple_worlds(self):
        """Test NxN broad phase with objects in different worlds and mixed collision groups."""
        verbose = False

        # Create random bounding boxes in min-max format
        ngeom = 50
        world_count = 4  # We'll distribute objects across 4 different worlds

        # Generate random centers and sizes using the new Generator API
        rng = np.random.Generator(np.random.PCG64(123))

        centers = rng.random((ngeom, 3)) * 5.0
        sizes = rng.random((ngeom, 3)) * 1.5  # box half-extent up to 1.5 in each direction
        geom_bounding_box_lower = centers - sizes
        geom_bounding_box_upper = centers + sizes

        np_geom_cutoff = np.zeros(ngeom, dtype=np.float32)

        # Randomly assign collision groups (including some negative ones for shared entities)
        num_groups = 5
        np_collision_group = rng.integers(1, num_groups + 1, size=ngeom, dtype=np.int32)

        # Make some entities shared (negative collision group)
        num_shared = int(sqrt(ngeom))
        shared_indices = rng.choice(ngeom, size=num_shared, replace=False)
        np_collision_group[shared_indices] = -1

        # Randomly distribute objects across worlds
        # Some objects in specific worlds (0, 1, 2, 3), some global (-1)
        np_shape_world = rng.integers(0, world_count, size=ngeom, dtype=np.int32)

        # Make some entities global (world -1) - they should collide with all worlds
        num_global = max(3, ngeom // 10)
        global_indices = rng.choice(ngeom, size=num_global, replace=False)
        np_shape_world[global_indices] = -1

        if verbose:
            print("\nTest setup:")
            print(f"  Total geometries: {ngeom}")
            print(f"  Number of worlds: {world_count}")
            print(f"  Global entities (world=-1): {num_global}")
            print(f"  Shared entities (group=-1): {num_shared}")
            print("\nWorld distribution:")
            for world_id in range(-1, world_count):
                count = np.sum(np_shape_world == world_id)
                print(f"  World {world_id}: {count} objects")
            print("\nCollision group distribution:")
            unique_groups = np.unique(np_collision_group)
            for group in unique_groups:
                count = np.sum(np_collision_group == group)
                print(f"  Group {group}: {count} objects")

        # Compute expected pairs using numpy
        pairs_np = find_overlapping_pairs_np(
            geom_bounding_box_lower, geom_bounding_box_upper, np_geom_cutoff, np_collision_group, np_shape_world
        )

        if verbose:
            print(f"\nExpected number of pairs: {len(pairs_np)}")
            if len(pairs_np) <= 20:
                print("Numpy contact pairs:")
                for i, pair in enumerate(pairs_np):
                    body_a, body_b = pair
                    world_a, world_b = np_shape_world[body_a], np_shape_world[body_b]
                    group_a, group_b = np_collision_group[body_a], np_collision_group[body_b]
                    print(
                        f"  Pair {i}: bodies ({body_a}, {body_b}) "
                        f"worlds ({world_a}, {world_b}) groups ({group_a}, {group_b})"
                    )

        # Setup Warp arrays
        num_lower_tri_elements = ngeom * (ngeom - 1) // 2
        max_candidate_pair = num_lower_tri_elements

        geom_lower = wp.array(geom_bounding_box_lower, dtype=wp.vec3)
        geom_upper = wp.array(geom_bounding_box_upper, dtype=wp.vec3)
        geom_cutoff = wp.array(np_geom_cutoff)
        collision_group = wp.array(np_collision_group)
        shape_world = wp.array(np_shape_world, dtype=wp.int32)
        candidate_pair_count = wp.array([0], dtype=wp.int32)
        candidate_pair = wp.array(np.zeros((max_candidate_pair, 2), dtype=wp.int32), dtype=wp.vec2i)

        # Initialize BroadPhaseAllPairs with shape_world for precomputation
        nxn_broadphase = BroadPhaseAllPairs(shape_world)

        if verbose:
            print("\nPrecomputed world map info:")
            print(f"  Number of kernel threads: {nxn_broadphase.num_kernel_threads}")
            print(f"  World slice ends: {nxn_broadphase.world_slice_ends.numpy()}")
            print(f"  World cumsum lower tri: {nxn_broadphase.world_cumsum_lower_tri.numpy()}")

        # Launch broad phase
        nxn_broadphase.launch(
            geom_lower,
            geom_upper,
            geom_cutoff,
            collision_group,
            shape_world,
            ngeom,
            candidate_pair,
            candidate_pair_count,
        )

        wp.synchronize()

        # Get results
        pairs_wp = candidate_pair.numpy()
        num_candidate_pair_result = candidate_pair_count.numpy()[0]

        if verbose:
            print(f"\nWarp found {num_candidate_pair_result} pairs")
            if num_candidate_pair_result <= 20:
                print("Warp contact pairs:")
                for i in range(num_candidate_pair_result):
                    pair = pairs_wp[i]
                    body_a, body_b = pair[0], pair[1]
                    world_a, world_b = np_shape_world[body_a], np_shape_world[body_b]
                    group_a, group_b = np_collision_group[body_a], np_collision_group[body_b]
                    print(
                        f"  Pair {i}: bodies ({body_a}, {body_b}) "
                        f"worlds ({world_a}, {world_b}) groups ({group_a}, {group_b})"
                    )

        # Verify results
        if len(pairs_np) != num_candidate_pair_result:
            print(f"\nMismatch: Expected {len(pairs_np)} pairs, got {num_candidate_pair_result}")

            # Show missing or extra pairs for debugging
            pairs_np_set = {tuple(pair) for pair in pairs_np}
            pairs_wp_set = {tuple(pairs_wp[i]) for i in range(num_candidate_pair_result)}

            missing = pairs_np_set - pairs_wp_set
            extra = pairs_wp_set - pairs_np_set

            if missing:
                print(f"Missing pairs ({len(missing)}):")
                for pair in list(missing)[:10]:
                    body_a, body_b = pair
                    world_a, world_b = np_shape_world[body_a], np_shape_world[body_b]
                    group_a, group_b = np_collision_group[body_a], np_collision_group[body_b]
                    print(f"  {pair}: worlds ({world_a}, {world_b}) groups ({group_a}, {group_b})")

            if extra:
                print(f"Extra pairs ({len(extra)}):")
                for pair in list(extra)[:10]:
                    body_a, body_b = pair
                    world_a, world_b = np_shape_world[body_a], np_shape_world[body_b]
                    group_a, group_b = np_collision_group[body_a], np_collision_group[body_b]
                    print(f"  {pair}: worlds ({world_a}, {world_b}) groups ({group_a}, {group_b})")

            assert len(pairs_np) == num_candidate_pair_result

        # Ensure every element in pairs_wp is also present in pairs_np
        pairs_np_set = {tuple(pair) for pair in pairs_np}
        for pair in pairs_wp[:num_candidate_pair_result]:
            assert tuple(pair) in pairs_np_set, f"Pair {tuple(pair)} from Warp not found in numpy pairs"

        if verbose:
            print(f"\nTest passed! All {len(pairs_np)} pairs matched.")

    def test_nxn_broadphase_with_shape_flags(self):
        """Test NxN broad phase with ShapeFlags filtering.

        This test verifies that:
        - Shapes without COLLIDE_SHAPES flag are correctly filtered out
        - Filtering works correctly with multiple worlds
        - num_regular_worlds is correctly computed after filtering (tests bug fix)
        - Edge case: filtering out all positive-world geometries but keeping -1 (tests critical bug)
        """
        verbose = False

        # Create random bounding boxes in min-max format
        ngeom = 50
        world_count = 4  # We'll distribute objects across 4 different worlds

        # Generate random centers and sizes using the new Generator API
        rng = np.random.Generator(np.random.PCG64(456))

        centers = rng.random((ngeom, 3)) * 5.0
        sizes = rng.random((ngeom, 3)) * 1.5  # box half-extent up to 1.5 in each direction
        geom_bounding_box_lower = centers - sizes
        geom_bounding_box_upper = centers + sizes

        np_geom_cutoff = np.zeros(ngeom, dtype=np.float32)

        # Randomly assign collision groups
        num_groups = 5
        np_collision_group = rng.integers(1, num_groups + 1, size=ngeom, dtype=np.int32)

        # Make some entities shared (negative collision group)
        num_shared = int(sqrt(ngeom))
        shared_indices = rng.choice(ngeom, size=num_shared, replace=False)
        np_collision_group[shared_indices] = -1

        # Randomly distribute objects across worlds
        # Some objects in specific worlds (0, 1, 2, 3), some global (-1)
        np_shape_world = rng.integers(0, world_count, size=ngeom, dtype=np.int32)

        # Make some entities global (world -1) - they should collide with all worlds
        num_global = max(3, ngeom // 10)
        global_indices = rng.choice(ngeom, size=num_global, replace=False)
        np_shape_world[global_indices] = -1

        # Create shape flags: some geometries will be visual-only (no COLLIDE_SHAPES flag)
        # Critical test case: filter out all positive-world geometries, keep only -1
        np_shape_flags = np.zeros(ngeom, dtype=np.int32)

        # Assign flags: ~70% will have COLLIDE_SHAPES flag, 30% will be visual-only
        colliding_indices = rng.choice(ngeom, size=int(0.7 * ngeom), replace=False)
        np_shape_flags[colliding_indices] = ShapeFlags.COLLIDE_SHAPES

        # Also set VISIBLE flag on some for completeness
        visible_indices = rng.choice(ngeom, size=int(0.8 * ngeom), replace=False)
        np_shape_flags[visible_indices] |= ShapeFlags.VISIBLE

        # CRITICAL TEST CASE: Filter out all geometries from world 0 (but keep world -1)
        # This tests the bug where num_regular_worlds was computed incorrectly after filtering
        world_0_mask = np_shape_world == 0
        np_shape_flags[world_0_mask] = 0  # Remove COLLIDE_SHAPES from all world 0 geometries

        # Count how many colliding geometries remain after filtering
        colliding_mask = (np_shape_flags & ShapeFlags.COLLIDE_SHAPES) != 0
        num_colliding = np.sum(colliding_mask)

        if verbose:
            print("\nTest setup with ShapeFlags:")
            print(f"  Total geometries: {ngeom}")
            print(f"  Geometries with COLLIDE_SHAPES flag: {num_colliding}")
            print(f"  Geometries filtered out: {ngeom - num_colliding}")
            print(f"  Number of worlds: {world_count}")
            print(f"  Global entities (world=-1): {num_global}")
            print(f"  World 0 geometries filtered: {np.sum(world_0_mask)}")
            print("\nWorld distribution (after filtering):")
            for world_id in range(-1, world_count):
                count_total = np.sum(np_shape_world == world_id)
                count_colliding = np.sum((np_shape_world == world_id) & colliding_mask)
                print(f"  World {world_id}: {count_total} total, {count_colliding} colliding")

        # Compute expected pairs using numpy (with shape flags filtering)
        pairs_np = find_overlapping_pairs_np(
            geom_bounding_box_lower,
            geom_bounding_box_upper,
            np_geom_cutoff,
            np_collision_group,
            np_shape_world,
            np_shape_flags,
        )

        if verbose:
            print(f"\nExpected number of pairs (after flag filtering): {len(pairs_np)}")
            if len(pairs_np) <= 20:
                print("Numpy contact pairs:")
                for i, pair in enumerate(pairs_np):
                    body_a, body_b = pair
                    world_a, world_b = np_shape_world[body_a], np_shape_world[body_b]
                    group_a, group_b = np_collision_group[body_a], np_collision_group[body_b]
                    flag_a, flag_b = np_shape_flags[body_a], np_shape_flags[body_b]
                    print(
                        f"  Pair {i}: bodies ({body_a}, {body_b}) "
                        f"worlds ({world_a}, {world_b}) groups ({group_a}, {group_b}) "
                        f"flags ({flag_a}, {flag_b})"
                    )

        # Setup Warp arrays
        num_lower_tri_elements = ngeom * (ngeom - 1) // 2
        max_candidate_pair = num_lower_tri_elements

        geom_lower = wp.array(geom_bounding_box_lower, dtype=wp.vec3)
        geom_upper = wp.array(geom_bounding_box_upper, dtype=wp.vec3)
        geom_cutoff = wp.array(np_geom_cutoff)
        collision_group = wp.array(np_collision_group)
        shape_world = wp.array(np_shape_world, dtype=wp.int32)
        shape_flags = wp.array(np_shape_flags, dtype=wp.int32)
        candidate_pair_count = wp.array([0], dtype=wp.int32)
        candidate_pair = wp.array(np.zeros((max_candidate_pair, 2), dtype=wp.int32), dtype=wp.vec2i)

        # Initialize BroadPhaseAllPairs with shape_world AND shape_flags
        nxn_broadphase = BroadPhaseAllPairs(shape_world, shape_flags=shape_flags)

        if verbose:
            print("\nPrecomputed world map info (with flags):")
            print(f"  Number of kernel threads: {nxn_broadphase.num_kernel_threads}")
            print(f"  Number of regular worlds: {nxn_broadphase.num_regular_worlds}")
            print(f"  World slice ends: {nxn_broadphase.world_slice_ends.numpy()}")
            print(f"  World cumsum lower tri: {nxn_broadphase.world_cumsum_lower_tri.numpy()}")

        # Verify num_regular_worlds is correct after filtering
        # It should be the number of worlds that have at least one colliding geometry
        colliding_worlds = np.unique(np_shape_world[colliding_mask])
        colliding_worlds = colliding_worlds[colliding_worlds >= 0]  # Exclude -1
        expected_num_regular_worlds = len(colliding_worlds)

        self.assertEqual(
            nxn_broadphase.num_regular_worlds,
            expected_num_regular_worlds,
            f"num_regular_worlds mismatch: expected {expected_num_regular_worlds} "
            f"(after filtering), got {nxn_broadphase.num_regular_worlds}",
        )

        # Launch broad phase
        nxn_broadphase.launch(
            geom_lower,
            geom_upper,
            geom_cutoff,
            collision_group,
            shape_world,
            ngeom,
            candidate_pair,
            candidate_pair_count,
        )

        wp.synchronize()

        # Get results
        pairs_wp = candidate_pair.numpy()
        num_candidate_pair_result = candidate_pair_count.numpy()[0]

        if verbose:
            print(f"\nWarp found {num_candidate_pair_result} pairs")
            if num_candidate_pair_result <= 20:
                print("Warp contact pairs:")
                for i in range(num_candidate_pair_result):
                    pair = pairs_wp[i]
                    body_a, body_b = pair[0], pair[1]
                    world_a, world_b = np_shape_world[body_a], np_shape_world[body_b]
                    group_a, group_b = np_collision_group[body_a], np_collision_group[body_b]
                    flag_a, flag_b = np_shape_flags[body_a], np_shape_flags[body_b]
                    print(
                        f"  Pair {i}: bodies ({body_a}, {body_b}) "
                        f"worlds ({world_a}, {world_b}) groups ({group_a}, {group_b}) "
                        f"flags ({flag_a}, {flag_b})"
                    )

        # Verify results
        if len(pairs_np) != num_candidate_pair_result:
            print(f"\nMismatch: Expected {len(pairs_np)} pairs, got {num_candidate_pair_result}")

            # Show missing or extra pairs for debugging
            pairs_np_set = {tuple(pair) for pair in pairs_np}
            pairs_wp_set = {tuple(pairs_wp[i]) for i in range(num_candidate_pair_result)}

            missing = pairs_np_set - pairs_wp_set
            extra = pairs_wp_set - pairs_np_set

            if missing:
                print(f"Missing pairs ({len(missing)}):")
                for pair in list(missing)[:10]:
                    body_a, body_b = pair
                    world_a, world_b = np_shape_world[body_a], np_shape_world[body_b]
                    group_a, group_b = np_collision_group[body_a], np_collision_group[body_b]
                    flag_a, flag_b = np_shape_flags[body_a], np_shape_flags[body_b]
                    print(
                        f"  {pair}: worlds ({world_a}, {world_b}) groups ({group_a}, {group_b}) "
                        f"flags ({flag_a}, {flag_b})"
                    )

            if extra:
                print(f"Extra pairs ({len(extra)}):")
                for pair in list(extra)[:10]:
                    body_a, body_b = pair
                    world_a, world_b = np_shape_world[body_a], np_shape_world[body_b]
                    group_a, group_b = np_collision_group[body_a], np_collision_group[body_b]
                    flag_a, flag_b = np_shape_flags[body_a], np_shape_flags[body_b]
                    print(
                        f"  {pair}: worlds ({world_a}, {world_b}) groups ({group_a}, {group_b}) "
                        f"flags ({flag_a}, {flag_b})"
                    )

            assert len(pairs_np) == num_candidate_pair_result

        # Ensure every element in pairs_wp is also present in pairs_np
        pairs_np_set = {tuple(pair) for pair in pairs_np}
        for pair in pairs_wp[:num_candidate_pair_result]:
            assert tuple(pair) in pairs_np_set, f"Pair {tuple(pair)} from Warp not found in numpy pairs"

        # Verify that no pairs contain filtered-out geometries
        for pair in pairs_wp[:num_candidate_pair_result]:
            body_a, body_b = pair[0], pair[1]
            flag_a = np_shape_flags[body_a]
            flag_b = np_shape_flags[body_b]
            assert (flag_a & ShapeFlags.COLLIDE_SHAPES) != 0, (
                f"Pair contains filtered geometry {body_a} with flag {flag_a}"
            )
            assert (flag_b & ShapeFlags.COLLIDE_SHAPES) != 0, (
                f"Pair contains filtered geometry {body_b} with flag {flag_b}"
            )

        if verbose:
            print(f"\nTest passed! All {len(pairs_np)} pairs matched, no filtered geometries included.")

    def test_explicit_pairs_broadphase(self):
        verbose = False

        # Create random bounding boxes in min-max format
        ngeom = 30

        # Generate random centers and sizes using the new Generator API
        rng = np.random.Generator(np.random.PCG64(42))

        centers = rng.random((ngeom, 3)) * 3.0
        sizes = rng.random((ngeom, 3)) * 2.0  # box half-extent up to 1.0 in each direction
        geom_bounding_box_lower = centers - sizes
        geom_bounding_box_upper = centers + sizes

        np_geom_cutoff = np.zeros(ngeom, dtype=np.float32)

        # Create explicit pairs to check - we'll take a subset of all possible pairs
        # For example, check pairs (0,1), (1,2), (2,3), etc.
        num_pairs_to_check = ngeom - 1
        explicit_pairs = np.array([(i, i + 1) for i in range(num_pairs_to_check)], dtype=np.int32)

        # Get ground truth overlaps for these explicit pairs
        pairs_np = []
        for pair in explicit_pairs:
            body_a, body_b = pair[0], pair[1]

            # Get bounding boxes for both bodies
            box_a_lower = geom_bounding_box_lower[body_a]
            box_a_upper = geom_bounding_box_upper[body_a]
            box_b_lower = geom_bounding_box_lower[body_b]
            box_b_upper = geom_bounding_box_upper[body_b]

            # Get cutoffs for both bodies
            cutoff_a = np_geom_cutoff[body_a]
            cutoff_b = np_geom_cutoff[body_b]

            # Check overlap using the function
            if check_aabb_overlap_host(
                wp.vec3(box_a_lower[0], box_a_lower[1], box_a_lower[2]),
                wp.vec3(box_a_upper[0], box_a_upper[1], box_a_upper[2]),
                cutoff_a,
                wp.vec3(box_b_lower[0], box_b_lower[1], box_b_lower[2]),
                wp.vec3(box_b_upper[0], box_b_upper[1], box_b_upper[2]),
                cutoff_b,
            ):
                pairs_np.append(tuple(pair))

        if verbose:
            print("Numpy contact pairs:")
            for i, pair in enumerate(pairs_np):
                print(f"  Pair {i}: bodies {pair}")

        # Convert data to Warp arrays
        geom_lower = wp.array(geom_bounding_box_lower, dtype=wp.vec3)
        geom_upper = wp.array(geom_bounding_box_upper, dtype=wp.vec3)
        geom_cutoff = wp.array(np_geom_cutoff)
        explicit_pairs_wp = wp.array(explicit_pairs, dtype=wp.vec2i)
        candidate_pair_count = wp.array(
            [
                0,
            ],
            dtype=wp.int32,
        )
        max_candidate_pair = num_pairs_to_check
        candidate_pair = wp.array(np.zeros((max_candidate_pair, 2), dtype=np.int32), dtype=wp.vec2i)

        explicit_broadphase = BroadPhaseExplicit()

        explicit_broadphase.launch(
            geom_lower,
            geom_upper,
            geom_cutoff,
            explicit_pairs_wp,
            num_pairs_to_check,
            candidate_pair,
            candidate_pair_count,
        )

        wp.synchronize()

        pairs_wp = candidate_pair.numpy()
        candidate_pair_count = candidate_pair_count.numpy()[0]

        if verbose:
            print("Warp contact pairs:")
            for i in range(candidate_pair_count):
                pair = pairs_wp[i]
                print(f"  Pair {i}: bodies ({pair[0]}, {pair[1]})")

            print("Checking if bounding boxes actually overlap:")
            for i in range(candidate_pair_count):
                pair = pairs_wp[i]
                body_a, body_b = pair[0], pair[1]

                # Get bounding boxes for both bodies
                box_a_lower = geom_bounding_box_lower[body_a]
                box_a_upper = geom_bounding_box_upper[body_a]
                box_b_lower = geom_bounding_box_lower[body_b]
                box_b_upper = geom_bounding_box_upper[body_b]

                # Get cutoffs for both bodies
                cutoff_a = np_geom_cutoff[body_a]
                cutoff_b = np_geom_cutoff[body_b]

                # Check overlap using the function
                overlap = check_aabb_overlap_host(
                    wp.vec3(box_a_lower[0], box_a_lower[1], box_a_lower[2]),
                    wp.vec3(box_a_upper[0], box_a_upper[1], box_a_upper[2]),
                    cutoff_a,
                    wp.vec3(box_b_lower[0], box_b_lower[1], box_b_lower[2]),
                    wp.vec3(box_b_upper[0], box_b_upper[1], box_b_upper[2]),
                    cutoff_b,
                )

                print(f"  Pair {i}: bodies ({body_a}, {body_b}) - overlap: {overlap}")

        if len(pairs_np) != candidate_pair_count:
            print(f"len(pairs_np)={len(pairs_np)}, candidate_pair_count={candidate_pair_count}")
            assert len(pairs_np) == candidate_pair_count

        # Ensure every element in pairs_wp is also present in pairs_np
        pairs_np_set = {tuple(pair) for pair in pairs_np}
        for pair in pairs_wp[:candidate_pair_count]:
            assert tuple(pair) in pairs_np_set, f"Pair {tuple(pair)} from Warp not found in numpy pairs"

        if verbose:
            print(len(pairs_np))

    def _test_sap_broadphase_impl(self, sort_type):
        verbose = False

        # Create random bounding boxes in min-max format
        ngeom = 30

        # Generate random centers and sizes using the new Generator API
        rng = np.random.Generator(np.random.PCG64(42))

        centers = rng.random((ngeom, 3)) * 3.0
        sizes = rng.random((ngeom, 3)) * 2.0  # box half-extent up to 1.0 in each direction
        geom_bounding_box_lower = centers - sizes
        geom_bounding_box_upper = centers + sizes

        np_geom_cutoff = np.zeros(ngeom, dtype=np.float32)
        num_groups = 5  # The zero group does not need to be counted
        np_collision_group = rng.integers(1, num_groups + 1, size=ngeom, dtype=np.int32)

        # Overwrite n random elements with -1
        minus_one_count = int(sqrt(ngeom))  # Number of elements to overwrite with -1
        random_indices = rng.choice(ngeom, size=minus_one_count, replace=False)
        np_collision_group[random_indices] = -1

        pairs_np = find_overlapping_pairs_np(
            geom_bounding_box_lower, geom_bounding_box_upper, np_geom_cutoff, np_collision_group
        )

        if verbose:
            print("Numpy contact pairs:")
            for i, pair in enumerate(pairs_np):
                body_a, body_b = pair
                group_a = np_collision_group[body_a]
                group_b = np_collision_group[body_b]
                print(f"  Pair {i}: bodies ({body_a}, {body_b}) with collision groups ({group_a}, {group_b})")

        # The number of elements in the lower triangular part of an n x n matrix (excluding the diagonal)
        # is given by n * (n - 1) // 2
        num_lower_tri_elements = ngeom * (ngeom - 1) // 2

        geom_lower = wp.array(geom_bounding_box_lower, dtype=wp.vec3)
        geom_upper = wp.array(geom_bounding_box_upper, dtype=wp.vec3)
        geom_cutoff = wp.array(np_geom_cutoff)
        collision_group = wp.array(np_collision_group)
        candidate_pair_count = wp.array(
            [
                0,
            ],
            dtype=wp.int32,
        )
        max_candidate_pair = num_lower_tri_elements
        candidate_pair = wp.array(np.zeros((max_candidate_pair, 2), dtype=wp.int32), dtype=wp.vec2i)

        # Create shape world array with all shapes in world 0
        shape_world = wp.array(np.full(ngeom, 0, dtype=np.int32), dtype=wp.int32)

        # Initialize BroadPhaseSAP with shape_world for precomputation
        sap_broadphase = BroadPhaseSAP(shape_world, sort_type=sort_type)

        sap_broadphase.launch(
            geom_lower,
            geom_upper,
            geom_cutoff,
            collision_group,
            shape_world,
            ngeom,
            candidate_pair,
            candidate_pair_count,
        )

        wp.synchronize()

        pairs_wp = candidate_pair.numpy()
        candidate_pair_count = candidate_pair_count.numpy()[0]

        if verbose:
            print("Warp contact pairs:")
            for i in range(candidate_pair_count):
                pair = pairs_wp[i]
                body_a, body_b = pair[0], pair[1]
                group_a = np_collision_group[body_a]
                group_b = np_collision_group[body_b]
                print(f"  Pair {i}: bodies ({body_a}, {body_b}) with collision groups ({group_a}, {group_b})")

            print("Checking if bounding boxes actually overlap:")
            for i in range(candidate_pair_count):
                pair = pairs_wp[i]
                body_a, body_b = pair[0], pair[1]

                # Get bounding boxes for both bodies
                box_a_lower = geom_bounding_box_lower[body_a]
                box_a_upper = geom_bounding_box_upper[body_a]
                box_b_lower = geom_bounding_box_lower[body_b]
                box_b_upper = geom_bounding_box_upper[body_b]

                # Get cutoffs for both bodies
                cutoff_a = np_geom_cutoff[body_a]
                cutoff_b = np_geom_cutoff[body_b]

                # Check overlap using the function
                overlap = check_aabb_overlap_host(
                    wp.vec3(box_a_lower[0], box_a_lower[1], box_a_lower[2]),
                    wp.vec3(box_a_upper[0], box_a_upper[1], box_a_upper[2]),
                    cutoff_a,
                    wp.vec3(box_b_lower[0], box_b_lower[1], box_b_lower[2]),
                    wp.vec3(box_b_upper[0], box_b_upper[1], box_b_upper[2]),
                    cutoff_b,
                )

                print(f"  Pair {i}: bodies ({body_a}, {body_b}) - overlap: {overlap}")

        if len(pairs_np) != candidate_pair_count:
            print(f"len(pairs_np)={len(pairs_np)}, candidate_pair_count={candidate_pair_count}")
            # print("pairs_np:", pairs_np)
            # print("pairs_wp[:candidate_pair_count]:", pairs_wp[:candidate_pair_count])
            assert len(pairs_np) == candidate_pair_count

        # Ensure every element in pairs_wp is also present in pairs_np
        pairs_np_set = {tuple(pair) for pair in pairs_np}
        for pair in pairs_wp[:candidate_pair_count]:
            assert tuple(pair) in pairs_np_set, f"Pair {tuple(pair)} from Warp not found in numpy pairs"

        if verbose:
            print(len(pairs_np))

    def test_sap_broadphase_segmented(self):
        """Test SAP broad phase with segmented sort."""
        self._test_sap_broadphase_impl(SAPSortType.SEGMENTED)

    def test_sap_broadphase_tile(self):
        """Test SAP broad phase with tile sort."""
        self._test_sap_broadphase_impl(SAPSortType.TILE)

    def _test_sap_broadphase_multiple_worlds_impl(self, sort_type):
        """Test SAP broad phase with objects in different worlds and mixed collision groups."""
        verbose = False

        # Create a scenario with multiple worlds
        ngeom = 40
        rng = np.random.Generator(np.random.PCG64(123))

        # Generate random centers and sizes
        centers = rng.random((ngeom, 3)) * 5.0
        sizes = rng.random((ngeom, 3)) * 1.5
        geom_bounding_box_lower = centers - sizes
        geom_bounding_box_upper = centers + sizes

        np_geom_cutoff = np.zeros(ngeom, dtype=np.float32)

        # Create a mix of world IDs: some in world 0, some in world 1, some in world 2, some shared (-1)
        np_shape_world = np.zeros(ngeom, dtype=np.int32)

        # Distribute geometries across worlds
        world_0_count = ngeom // 3
        world_1_count = ngeom // 3

        np_shape_world[:world_0_count] = 0
        np_shape_world[world_0_count : world_0_count + world_1_count] = 1
        np_shape_world[world_0_count + world_1_count :] = -1  # Shared entities

        # Shuffle to make it more realistic
        rng.shuffle(np_shape_world)

        # Create collision groups (positive values for filtering within worlds)
        num_groups = 4
        np_collision_group = rng.integers(1, num_groups + 1, size=ngeom, dtype=np.int32)

        # Make some collision groups negative (shared across all worlds)
        minus_one_count = int(sqrt(ngeom))
        random_indices = rng.choice(ngeom, size=minus_one_count, replace=False)
        np_collision_group[random_indices] = -1

        if verbose:
            print("\nGeometry world assignments:")
            for i in range(ngeom):
                print(f"  Geom {i}: world={np_shape_world[i]}, collision_group={np_collision_group[i]}")

        # Find expected pairs using numpy
        pairs_np = find_overlapping_pairs_np(
            geom_bounding_box_lower, geom_bounding_box_upper, np_geom_cutoff, np_collision_group, np_shape_world
        )

        if verbose:
            print(f"\nNumpy found {len(pairs_np)} pairs:")
            for i, pair in enumerate(pairs_np):
                body_a, body_b = pair
                world_a = np_shape_world[body_a]
                world_b = np_shape_world[body_b]
                group_a = np_collision_group[body_a]
                group_b = np_collision_group[body_b]
                print(
                    f"  Pair {i}: bodies ({body_a}, {body_b}) worlds ({world_a}, {world_b}) groups ({group_a}, {group_b})"
                )

        # Setup Warp arrays
        num_lower_tri_elements = ngeom * (ngeom - 1) // 2
        geom_lower = wp.array(geom_bounding_box_lower, dtype=wp.vec3)
        geom_upper = wp.array(geom_bounding_box_upper, dtype=wp.vec3)
        geom_cutoff = wp.array(np_geom_cutoff)
        collision_group = wp.array(np_collision_group)
        shape_world = wp.array(np_shape_world, dtype=wp.int32)
        candidate_pair_count = wp.array([0], dtype=wp.int32)
        max_candidate_pair = num_lower_tri_elements
        candidate_pair = wp.array(np.zeros((max_candidate_pair, 2), dtype=wp.int32), dtype=wp.vec2i)

        # Initialize and launch SAP broad phase
        sap_broadphase = BroadPhaseSAP(shape_world, sort_type=sort_type)
        sap_broadphase.launch(
            geom_lower,
            geom_upper,
            geom_cutoff,
            collision_group,
            shape_world,
            ngeom,
            candidate_pair,
            candidate_pair_count,
        )

        wp.synchronize()

        pairs_wp = candidate_pair.numpy()
        num_candidate_pair_val = candidate_pair_count.numpy()[0]

        if verbose:
            print(f"\nWarp found {num_candidate_pair_val} pairs:")
            for i in range(num_candidate_pair_val):
                pair = pairs_wp[i]
                body_a, body_b = pair[0], pair[1]
                world_a = np_shape_world[body_a]
                world_b = np_shape_world[body_b]
                group_a = np_collision_group[body_a]
                group_b = np_collision_group[body_b]
                print(
                    f"  Pair {i}: bodies ({body_a}, {body_b}) worlds ({world_a}, {world_b}) groups ({group_a}, {group_b})"
                )

        # Verify results
        if len(pairs_np) != num_candidate_pair_val:
            print(f"\nMismatch: numpy found {len(pairs_np)} pairs, Warp found {num_candidate_pair_val} pairs")

            # Show missing pairs
            pairs_np_set = {tuple(pair) for pair in pairs_np}
            pairs_wp_set = {tuple(pairs_wp[i]) for i in range(num_candidate_pair_val)}

            missing_in_warp = pairs_np_set - pairs_wp_set
            extra_in_warp = pairs_wp_set - pairs_np_set

            if missing_in_warp:
                print(f"\nPairs in numpy but not in Warp ({len(missing_in_warp)}):")
                for pair in list(missing_in_warp)[:10]:  # Show first 10
                    a, b = pair
                    print(
                        f"  ({a}, {b}): worlds ({np_shape_world[a]}, {np_shape_world[b]}) groups ({np_collision_group[a]}, {np_collision_group[b]})"
                    )

            if extra_in_warp:
                print(f"\nPairs in Warp but not in numpy ({len(extra_in_warp)}):")
                for pair in list(extra_in_warp)[:10]:  # Show first 10
                    a, b = pair
                    print(
                        f"  ({a}, {b}): worlds ({np_shape_world[a]}, {np_shape_world[b]}) groups ({np_collision_group[a]}, {np_collision_group[b]})"
                    )

            assert len(pairs_np) == num_candidate_pair_val

        # Ensure every element in pairs_wp is also present in pairs_np
        pairs_np_set = {tuple(pair) for pair in pairs_np}
        for pair in pairs_wp[:num_candidate_pair_val]:
            pair_tuple = tuple(pair)
            assert pair_tuple in pairs_np_set, f"Pair {pair_tuple} from Warp not found in numpy pairs"

        if verbose:
            print(f"\nTest passed! Found {len(pairs_np)} valid collision pairs across multiple worlds.")

    def test_sap_broadphase_multiple_worlds_segmented(self):
        """Test SAP broad phase with multiple worlds using segmented sort."""
        self._test_sap_broadphase_multiple_worlds_impl(SAPSortType.SEGMENTED)

    def test_sap_broadphase_multiple_worlds_tile(self):
        """Test SAP broad phase with multiple worlds using tile sort."""
        self._test_sap_broadphase_multiple_worlds_impl(SAPSortType.TILE)

    def _test_sap_broadphase_with_shape_flags_impl(self, sort_type):
        """Test SAP broad phase with ShapeFlags filtering.

        This test verifies that:
        - Shapes without COLLIDE_SHAPES flag are correctly filtered out
        - Filtering works correctly with multiple worlds
        - num_regular_worlds is correctly computed after filtering (tests bug fix)
        """
        verbose = False

        # Create random bounding boxes in min-max format
        ngeom = 40
        world_count = 4

        # Generate random centers and sizes
        rng = np.random.Generator(np.random.PCG64(789))

        centers = rng.random((ngeom, 3)) * 5.0
        sizes = rng.random((ngeom, 3)) * 1.5
        geom_bounding_box_lower = centers - sizes
        geom_bounding_box_upper = centers + sizes

        np_geom_cutoff = np.zeros(ngeom, dtype=np.float32)

        # Randomly assign collision groups
        num_groups = 5
        np_collision_group = rng.integers(1, num_groups + 1, size=ngeom, dtype=np.int32)

        # Make some entities shared (negative collision group)
        num_shared = int(sqrt(ngeom))
        shared_indices = rng.choice(ngeom, size=num_shared, replace=False)
        np_collision_group[shared_indices] = -1

        # Randomly distribute objects across worlds
        np_shape_world = rng.integers(0, world_count, size=ngeom, dtype=np.int32)

        # Make some entities global (world -1)
        num_global = max(3, ngeom // 10)
        global_indices = rng.choice(ngeom, size=num_global, replace=False)
        np_shape_world[global_indices] = -1

        # Create shape flags: some geometries will be visual-only
        np_shape_flags = np.zeros(ngeom, dtype=np.int32)

        # Assign flags: ~60% will have COLLIDE_SHAPES flag
        colliding_indices = rng.choice(ngeom, size=int(0.6 * ngeom), replace=False)
        np_shape_flags[colliding_indices] = ShapeFlags.COLLIDE_SHAPES

        # CRITICAL TEST CASE: Filter out all geometries from world 1
        world_1_mask = np_shape_world == 1
        np_shape_flags[world_1_mask] = 0  # Remove COLLIDE_SHAPES from all world 1 geometries

        # Count how many colliding geometries remain after filtering
        colliding_mask = (np_shape_flags & ShapeFlags.COLLIDE_SHAPES) != 0
        num_colliding = np.sum(colliding_mask)

        if verbose:
            print("\nSAP test setup with ShapeFlags:")
            print(f"  Total geometries: {ngeom}")
            print(f"  Geometries with COLLIDE_SHAPES flag: {num_colliding}")
            print(f"  Geometries filtered out: {ngeom - num_colliding}")

        # Compute expected pairs using numpy (with shape flags filtering)
        pairs_np = find_overlapping_pairs_np(
            geom_bounding_box_lower,
            geom_bounding_box_upper,
            np_geom_cutoff,
            np_collision_group,
            np_shape_world,
            np_shape_flags,
        )

        # Setup Warp arrays
        num_lower_tri_elements = ngeom * (ngeom - 1) // 2
        geom_lower = wp.array(geom_bounding_box_lower, dtype=wp.vec3)
        geom_upper = wp.array(geom_bounding_box_upper, dtype=wp.vec3)
        geom_cutoff = wp.array(np_geom_cutoff)
        collision_group = wp.array(np_collision_group)
        shape_world = wp.array(np_shape_world, dtype=wp.int32)
        shape_flags = wp.array(np_shape_flags, dtype=wp.int32)
        candidate_pair_count = wp.array([0], dtype=wp.int32)
        candidate_pair = wp.array(np.zeros((num_lower_tri_elements, 2), dtype=wp.int32), dtype=wp.vec2i)

        # Initialize SAP broad phase with shape_flags
        sap_broadphase = BroadPhaseSAP(shape_world, shape_flags=shape_flags, sort_type=sort_type)

        # Verify num_regular_worlds is correct after filtering
        colliding_worlds = np.unique(np_shape_world[colliding_mask])
        colliding_worlds = colliding_worlds[colliding_worlds >= 0]  # Exclude -1
        expected_num_regular_worlds = len(colliding_worlds)

        self.assertEqual(
            sap_broadphase.num_regular_worlds,
            expected_num_regular_worlds,
            f"num_regular_worlds mismatch: expected {expected_num_regular_worlds} "
            f"(after filtering), got {sap_broadphase.num_regular_worlds}",
        )

        # Launch SAP broad phase
        sap_broadphase.launch(
            geom_lower,
            geom_upper,
            geom_cutoff,
            collision_group,
            shape_world,
            ngeom,
            candidate_pair,
            candidate_pair_count,
        )

        wp.synchronize()

        # Get results
        pairs_wp = candidate_pair.numpy()
        num_candidate_pair_result = candidate_pair_count.numpy()[0]

        # Verify results
        if len(pairs_np) != num_candidate_pair_result:
            pairs_np_set = {tuple(pair) for pair in pairs_np}
            pairs_wp_set = {tuple(pairs_wp[i]) for i in range(num_candidate_pair_result)}

            missing = pairs_np_set - pairs_wp_set
            extra = pairs_wp_set - pairs_np_set

            if missing:
                print(f"\nMissing pairs ({len(missing)}):")
                for pair in list(missing)[:10]:
                    body_a, body_b = pair
                    print(f"  {pair}: worlds ({np_shape_world[body_a]}, {np_shape_world[body_b]})")

            if extra:
                print(f"\nExtra pairs ({len(extra)}):")
                for pair in list(extra)[:10]:
                    body_a, body_b = pair
                    print(f"  {pair}: worlds ({np_shape_world[body_a]}, {np_shape_world[body_b]})")

            assert len(pairs_np) == num_candidate_pair_result

        # Ensure every element in pairs_wp is also present in pairs_np
        pairs_np_set = {tuple(pair) for pair in pairs_np}
        for pair in pairs_wp[:num_candidate_pair_result]:
            assert tuple(pair) in pairs_np_set, f"Pair {tuple(pair)} from Warp not found in numpy pairs"

        # Verify that no pairs contain filtered-out geometries
        for pair in pairs_wp[:num_candidate_pair_result]:
            body_a, body_b = pair[0], pair[1]
            flag_a = np_shape_flags[body_a]
            flag_b = np_shape_flags[body_b]
            assert (flag_a & ShapeFlags.COLLIDE_SHAPES) != 0, (
                f"Pair contains filtered geometry {body_a} with flag {flag_a}"
            )
            assert (flag_b & ShapeFlags.COLLIDE_SHAPES) != 0, (
                f"Pair contains filtered geometry {body_b} with flag {flag_b}"
            )

    def test_sap_broadphase_with_shape_flags_segmented(self):
        """Test SAP broad phase with ShapeFlags using segmented sort."""
        self._test_sap_broadphase_with_shape_flags_impl(SAPSortType.SEGMENTED)

    def test_sap_broadphase_with_shape_flags_tile(self):
        """Test SAP broad phase with ShapeFlags using tile sort."""
        self._test_sap_broadphase_with_shape_flags_impl(SAPSortType.TILE)

    def test_nxn_edge_cases(self):
        """Test NxN broad phase with tricky edge cases to verify GPU code correctness.

        This test includes:
        - Boundary conditions (AABBs exactly touching)
        - Various cutoff distances that create/prevent overlaps
        - Complex world/group interactions
        - Duplicate pair prevention
        - Mixed global (-1) and world-specific entities
        - Large number of geometries to stress-test GPU code
        """
        verbose = False

        # Create a carefully crafted scenario with edge cases
        # Use larger number to really stress test the GPU code
        base_cases = 24  # Base edge cases
        num_clusters = 10  # Number of overlapping clusters
        cluster_size = 8  # Geometries per cluster
        num_isolated = 20  # Isolated geometries
        ngeom = base_cases + (num_clusters * cluster_size) + num_isolated  # Total: 124 geometries

        # Case 1: Two boxes exactly touching (boundary case) - should overlap with cutoff=0
        box1_lower = np.array([0.0, 0.0, 0.0])
        box1_upper = np.array([1.0, 1.0, 1.0])
        box2_lower = np.array([1.0, 0.0, 0.0])  # Exactly touching box1 on x-axis
        box2_upper = np.array([2.0, 1.0, 1.0])

        # Case 2: Two boxes very close but not touching - should overlap with cutoff > 0
        box3_lower = np.array([2.1, 0.0, 0.0])
        box3_upper = np.array([3.0, 1.0, 1.0])
        box4_lower = np.array([3.2, 0.0, 0.0])  # Gap of 0.2
        box4_upper = np.array([4.0, 1.0, 1.0])

        # Case 3: Three boxes in a row, all overlapping
        box5_lower = np.array([5.0, 0.0, 0.0])
        box5_upper = np.array([6.5, 1.0, 1.0])
        box6_lower = np.array([6.0, 0.0, 0.0])
        box6_upper = np.array([7.5, 1.0, 1.0])
        box7_lower = np.array([7.0, 0.0, 0.0])
        box7_upper = np.array([8.5, 1.0, 1.0])

        # Case 4: Multiple boxes in same location (stress test for duplicates)
        # All these boxes should collide with each other
        box8_lower = np.array([10.0, 0.0, 0.0])
        box8_upper = np.array([11.0, 1.0, 1.0])
        box9_lower = np.array([10.1, 0.1, 0.1])
        box9_upper = np.array([10.9, 0.9, 0.9])
        box10_lower = np.array([10.2, 0.2, 0.2])
        box10_upper = np.array([10.8, 0.8, 0.8])
        box11_lower = np.array([10.3, 0.3, 0.3])
        box11_upper = np.array([10.7, 0.7, 0.7])

        # Case 5: Global entities (-1 world) that should collide with multiple worlds
        box12_lower = np.array([15.0, 0.0, 0.0])
        box12_upper = np.array([16.0, 1.0, 1.0])
        box13_lower = np.array([15.2, 0.2, 0.2])
        box13_upper = np.array([15.8, 0.8, 0.8])

        # Case 6: Collision group filtering edge cases
        # Same location but different groups (some should collide, some shouldn't)
        box14_lower = np.array([20.0, 0.0, 0.0])
        box14_upper = np.array([21.0, 1.0, 1.0])
        box15_lower = np.array([20.1, 0.1, 0.1])
        box15_upper = np.array([20.9, 0.9, 0.9])
        box16_lower = np.array([20.2, 0.2, 0.2])
        box16_upper = np.array([20.8, 0.8, 0.8])
        box17_lower = np.array([20.3, 0.3, 0.3])
        box17_upper = np.array([20.7, 0.7, 0.7])

        # Case 7: Different worlds (should NOT collide even if overlapping)
        box18_lower = np.array([25.0, 0.0, 0.0])
        box18_upper = np.array([26.0, 1.0, 1.0])
        box19_lower = np.array([25.2, 0.2, 0.2])
        box19_upper = np.array([25.8, 0.8, 0.8])

        # Case 8: Isolated boxes (no collisions)
        box20_lower = np.array([30.0, 0.0, 0.0])
        box20_upper = np.array([31.0, 1.0, 1.0])
        box21_lower = np.array([35.0, 0.0, 0.0])
        box21_upper = np.array([36.0, 1.0, 1.0])
        box22_lower = np.array([40.0, 0.0, 0.0])
        box22_upper = np.array([41.0, 1.0, 1.0])

        # Case 9: Zero collision group (should never collide)
        box23_lower = np.array([45.0, 0.0, 0.0])
        box23_upper = np.array([46.0, 1.0, 1.0])
        box24_lower = np.array([45.2, 0.2, 0.2])
        box24_upper = np.array([45.8, 0.8, 0.8])

        # Build the base cases list
        base_lowers = [
            box1_lower,
            box2_lower,
            box3_lower,
            box4_lower,
            box5_lower,
            box6_lower,
            box7_lower,
            box8_lower,
            box9_lower,
            box10_lower,
            box11_lower,
            box12_lower,
            box13_lower,
            box14_lower,
            box15_lower,
            box16_lower,
            box17_lower,
            box18_lower,
            box19_lower,
            box20_lower,
            box21_lower,
            box22_lower,
            box23_lower,
            box24_lower,
        ]
        base_uppers = [
            box1_upper,
            box2_upper,
            box3_upper,
            box4_upper,
            box5_upper,
            box6_upper,
            box7_upper,
            box8_upper,
            box9_upper,
            box10_upper,
            box11_upper,
            box12_upper,
            box13_upper,
            box14_upper,
            box15_upper,
            box16_upper,
            box17_upper,
            box18_upper,
            box19_upper,
            box20_upper,
            box21_upper,
            box22_upper,
            box23_upper,
            box24_upper,
        ]

        # Case 10+: Add overlapping clusters (stress test for many-to-many collisions)
        # Each cluster has multiple overlapping geometries in the same location
        rng = np.random.Generator(np.random.PCG64(999))
        cluster_lowers = []
        cluster_uppers = []
        cluster_cutoffs = []
        cluster_groups = []
        cluster_worlds = []

        for cluster_id in range(num_clusters):
            # Random center for this cluster
            cluster_center = np.array([50.0 + cluster_id * 10.0, rng.random() * 5.0, rng.random() * 5.0])
            # Randomly assign world (mix of specific worlds and global)
            if cluster_id < 3:
                cluster_world = -1  # First 3 clusters are global
            else:
                cluster_world = cluster_id % 4  # Distribute across 4 worlds

            # Random collision group for this cluster
            cluster_group = (cluster_id % 3) + 1  # Groups 1, 2, 3

            for _ in range(cluster_size):
                # Create slightly offset overlapping boxes
                offset = rng.random(3) * 0.5
                lower = cluster_center - 0.5 + offset
                upper = cluster_center + 0.5 + offset
                cluster_lowers.append(lower)
                cluster_uppers.append(upper)
                cluster_cutoffs.append(0.0)
                cluster_groups.append(cluster_group)
                cluster_worlds.append(cluster_world)

        # Case 11+: Add isolated geometries (should have no collisions)
        isolated_lowers = []
        isolated_uppers = []
        isolated_cutoffs = []
        isolated_groups = []
        isolated_worlds = []

        for i in range(num_isolated):
            # Place far apart
            center = np.array([200.0 + i * 5.0, 0.0, 0.0])
            isolated_lowers.append(center - 0.3)
            isolated_uppers.append(center + 0.3)
            isolated_cutoffs.append(0.0)
            isolated_groups.append((i % 5) + 1)  # Varied groups
            isolated_worlds.append(i % 3)  # Distribute across worlds

        # Combine all geometries
        all_lowers = base_lowers + cluster_lowers + isolated_lowers
        all_uppers = base_uppers + cluster_uppers + isolated_uppers

        geom_bounding_box_lower = np.array(all_lowers)
        geom_bounding_box_upper = np.array(all_uppers)

        # Combine cutoffs
        base_cutoffs = [
            0.0,
            0.0,  # box1, box2: exactly touching
            0.0,
            0.15,  # box3, box4: gap of 0.2, cutoff 0.15 makes them overlap
            0.0,
            0.0,
            0.0,  # box5-7: overlapping chain
            0.0,
            0.0,
            0.0,
            0.0,  # box8-11: nested boxes
            0.0,
            0.0,  # box12-13: global entities
            0.0,
            0.0,
            0.0,
            0.0,  # box14-17: group filtering
            0.0,
            0.0,  # box18-19: different worlds
            0.0,
            0.0,
            0.0,  # box20-22: isolated
            0.0,
            0.0,  # box23-24: zero group
        ]
        np_geom_cutoff = np.array(base_cutoffs + cluster_cutoffs + isolated_cutoffs, dtype=np.float32)

        # Combine collision groups
        base_groups = [
            1,
            1,  # box1-2: same group, should collide
            2,
            2,  # box3-4: same group, should collide (with cutoff)
            1,
            1,
            1,  # box5-7: same group, chain collision
            1,
            1,
            1,
            1,  # box8-11: same group, all collide
            -1,
            -1,  # box12-13: negative group, collide with each other
            1,
            2,
            -1,
            -2,  # box14-17: mixed groups (1 w/ -1, 2 w/ -1, not 1 w/ 2, not -1 w/ -2)
            1,
            1,  # box18-19: same group but different worlds
            1,
            2,
            3,  # box20-22: different groups, no collision
            0,
            0,  # box23-24: zero group, never collide
        ]
        np_collision_group = np.array(base_groups + cluster_groups + isolated_groups, dtype=np.int32)

        # Combine worlds
        base_worlds = [
            0,
            0,  # box1-2: world 0
            0,
            0,  # box3-4: world 0
            0,
            0,
            0,  # box5-7: world 0
            0,
            0,
            0,
            0,  # box8-11: world 0
            -1,
            -1,  # box12-13: global, collide with all worlds
            0,
            0,
            0,
            0,  # box14-17: world 0
            0,
            1,  # box18-19: different worlds (should NOT collide)
            0,
            0,
            0,  # box20-22: world 0 (but different groups)
            0,
            0,  # box23-24: world 0 (but zero group)
        ]
        np_shape_world = np.array(base_worlds + cluster_worlds + isolated_worlds, dtype=np.int32)

        if verbose:
            print("\n=== NxN Edge Case Test Setup ===")
            print(f"Total geometries: {ngeom}")
            for i in range(ngeom):
                print(
                    f"  Geom {i}: world={np_shape_world[i]}, group={np_collision_group[i]}, "
                    f"cutoff={np_geom_cutoff[i]:.2f}"
                )

        # Compute expected pairs using numpy
        pairs_np = find_overlapping_pairs_np(
            geom_bounding_box_lower, geom_bounding_box_upper, np_geom_cutoff, np_collision_group, np_shape_world
        )

        if verbose:
            print(f"\nExpected {len(pairs_np)} pairs from numpy verification")
            for pair in pairs_np:
                a, b = pair
                print(
                    f"  Pair ({a}, {b}): worlds ({np_shape_world[a]}, {np_shape_world[b]}) "
                    f"groups ({np_collision_group[a]}, {np_collision_group[b]})"
                )

        # Setup Warp arrays
        num_lower_tri_elements = ngeom * (ngeom - 1) // 2
        geom_lower = wp.array(geom_bounding_box_lower, dtype=wp.vec3)
        geom_upper = wp.array(geom_bounding_box_upper, dtype=wp.vec3)
        geom_cutoff = wp.array(np_geom_cutoff)
        collision_group = wp.array(np_collision_group)
        shape_world = wp.array(np_shape_world, dtype=wp.int32)
        candidate_pair_count = wp.array([0], dtype=wp.int32)
        candidate_pair = wp.array(np.zeros((num_lower_tri_elements, 2), dtype=wp.int32), dtype=wp.vec2i)

        # Initialize and launch NxN broad phase
        nxn_broadphase = BroadPhaseAllPairs(shape_world)
        nxn_broadphase.launch(
            geom_lower,
            geom_upper,
            geom_cutoff,
            collision_group,
            shape_world,
            ngeom,
            candidate_pair,
            candidate_pair_count,
        )

        wp.synchronize()

        # Get results
        pairs_wp = candidate_pair.numpy()
        num_candidate_pair_result = candidate_pair_count.numpy()[0]

        if verbose:
            print(f"\nWarp found {num_candidate_pair_result} pairs")
            for i in range(num_candidate_pair_result):
                pair = pairs_wp[i]
                a, b = pair[0], pair[1]
                print(
                    f"  Pair ({a}, {b}): worlds ({np_shape_world[a]}, {np_shape_world[b]}) "
                    f"groups ({np_collision_group[a]}, {np_collision_group[b]})"
                )

        # Verify: check for duplicate pairs
        pairs_wp_set = {tuple(pairs_wp[i]) for i in range(num_candidate_pair_result)}
        self.assertEqual(
            len(pairs_wp_set), num_candidate_pair_result, "Duplicate pairs detected in NxN broad phase results"
        )

        # Verify: check count matches
        if len(pairs_np) != num_candidate_pair_result:
            pairs_np_set = {tuple(pair) for pair in pairs_np}
            missing = pairs_np_set - pairs_wp_set
            extra = pairs_wp_set - pairs_np_set

            if missing:
                print(f"\nMissing pairs ({len(missing)}):")
                for pair in list(missing):
                    a, b = pair
                    print(
                        f"  ({a}, {b}): worlds ({np_shape_world[a]}, {np_shape_world[b]}) "
                        f"groups ({np_collision_group[a]}, {np_collision_group[b]})"
                    )

            if extra:
                print(f"\nExtra pairs ({len(extra)}):")
                for pair in list(extra):
                    a, b = pair
                    print(
                        f"  ({a}, {b}): worlds ({np_shape_world[a]}, {np_shape_world[b]}) "
                        f"groups ({np_collision_group[a]}, {np_collision_group[b]})"
                    )

        self.assertEqual(
            len(pairs_np), num_candidate_pair_result, f"Expected {len(pairs_np)} pairs, got {num_candidate_pair_result}"
        )

        # Verify: all Warp pairs are in numpy pairs
        pairs_np_set = {tuple(pair) for pair in pairs_np}
        for pair in pairs_wp[:num_candidate_pair_result]:
            pair_tuple = tuple(pair)
            self.assertIn(pair_tuple, pairs_np_set, f"Pair {pair_tuple} from Warp not found in numpy pairs")

        if verbose:
            print(f"\n✓ Test passed! All {len(pairs_np)} pairs matched, no duplicates.")

    def _test_sap_edge_cases_impl(self, sort_type):
        """Test SAP broad phase with tricky edge cases to verify GPU code correctness.

        This test includes:
        - Boundary conditions (AABBs exactly touching)
        - Various cutoff distances that create/prevent overlaps
        - Complex world/group interactions
        - Duplicate pair prevention (especially for shared geometries)
        - Mixed global (-1) and world-specific entities
        - Large number of geometries to stress-test GPU sorting and sweep
        """
        verbose = False

        # Create a carefully crafted scenario with edge cases
        # Use larger number to really stress test the SAP algorithm
        base_cases = 26  # Base edge cases
        num_clusters = 12  # Number of overlapping clusters (SAP stress test)
        cluster_size = 10  # Geometries per cluster (larger for SAP)
        num_isolated = 25  # Isolated geometries
        ngeom = base_cases + (num_clusters * cluster_size) + num_isolated  # Total: 171 geometries

        # Case 1: Two boxes exactly touching along sweep axis (x-axis) - should overlap
        box1_lower = np.array([0.0, 0.0, 0.0])
        box1_upper = np.array([1.0, 1.0, 1.0])
        box2_lower = np.array([1.0, 0.0, 0.0])  # Exactly touching box1 on x-axis
        box2_upper = np.array([2.0, 1.0, 1.0])

        # Case 2: Boxes with gap, but cutoff makes them overlap
        box3_lower = np.array([2.15, 0.0, 0.0])
        box3_upper = np.array([3.0, 1.0, 1.0])
        box4_lower = np.array([3.25, 0.0, 0.0])  # Gap of 0.25
        box4_upper = np.array([4.0, 1.0, 1.0])

        # Case 3: Overlapping chain (stress test for SAP sorting)
        box5_lower = np.array([5.0, 0.0, 0.0])
        box5_upper = np.array([6.5, 1.0, 1.0])
        box6_lower = np.array([6.0, 0.0, 0.0])
        box6_upper = np.array([7.5, 1.0, 1.0])
        box7_lower = np.array([7.0, 0.0, 0.0])
        box7_upper = np.array([8.5, 1.0, 1.0])
        box8_lower = np.array([8.0, 0.0, 0.0])
        box8_upper = np.array([9.5, 1.0, 1.0])

        # Case 4: Multiple boxes in same location (duplicate prevention test)
        box9_lower = np.array([10.0, 0.0, 0.0])
        box9_upper = np.array([11.0, 1.0, 1.0])
        box10_lower = np.array([10.1, 0.1, 0.1])
        box10_upper = np.array([10.9, 0.9, 0.9])
        box11_lower = np.array([10.2, 0.2, 0.2])
        box11_upper = np.array([10.8, 0.8, 0.8])

        # Case 5: Global entities that should appear in multiple worlds
        # Critical: these should only generate ONE pair total, not one per world
        box12_lower = np.array([15.0, 0.0, 0.0])
        box12_upper = np.array([16.0, 1.0, 1.0])
        box13_lower = np.array([15.2, 0.2, 0.2])
        box13_upper = np.array([15.8, 0.8, 0.8])

        # Case 6: Mixed global and world-specific entities
        box14_lower = np.array([18.0, 0.0, 0.0])
        box14_upper = np.array([19.0, 1.0, 1.0])
        box15_lower = np.array([18.2, 0.2, 0.2])
        box15_upper = np.array([18.8, 0.8, 0.8])
        box16_lower = np.array([18.4, 0.4, 0.4])
        box16_upper = np.array([18.6, 0.6, 0.6])

        # Case 7: Collision group edge cases at same location
        box17_lower = np.array([22.0, 0.0, 0.0])
        box17_upper = np.array([23.0, 1.0, 1.0])
        box18_lower = np.array([22.1, 0.1, 0.1])
        box18_upper = np.array([22.9, 0.9, 0.9])
        box19_lower = np.array([22.2, 0.2, 0.2])
        box19_upper = np.array([22.8, 0.8, 0.8])
        box20_lower = np.array([22.3, 0.3, 0.3])
        box20_upper = np.array([22.7, 0.7, 0.7])

        # Case 8: Different worlds, overlapping (should NOT collide)
        box21_lower = np.array([26.0, 0.0, 0.0])
        box21_upper = np.array([27.0, 1.0, 1.0])
        box22_lower = np.array([26.2, 0.2, 0.2])
        box22_upper = np.array([26.8, 0.8, 0.8])

        # Case 9: Reverse order in space (tests SAP sorting correctness)
        box23_lower = np.array([30.0, 0.0, 0.0])
        box23_upper = np.array([31.0, 1.0, 1.0])
        box24_lower = np.array([29.0, 0.0, 0.0])  # Lower x than box23
        box24_upper = np.array([30.5, 1.0, 1.0])

        # Case 10: Zero collision group (never collides)
        box25_lower = np.array([33.0, 0.0, 0.0])
        box25_upper = np.array([34.0, 1.0, 1.0])
        box26_lower = np.array([33.2, 0.2, 0.2])
        box26_upper = np.array([33.8, 0.8, 0.8])

        # Build the base cases list
        base_lowers = [
            box1_lower,
            box2_lower,
            box3_lower,
            box4_lower,
            box5_lower,
            box6_lower,
            box7_lower,
            box8_lower,
            box9_lower,
            box10_lower,
            box11_lower,
            box12_lower,
            box13_lower,
            box14_lower,
            box15_lower,
            box16_lower,
            box17_lower,
            box18_lower,
            box19_lower,
            box20_lower,
            box21_lower,
            box22_lower,
            box23_lower,
            box24_lower,
            box25_lower,
            box26_lower,
        ]
        base_uppers = [
            box1_upper,
            box2_upper,
            box3_upper,
            box4_upper,
            box5_upper,
            box6_upper,
            box7_upper,
            box8_upper,
            box9_upper,
            box10_upper,
            box11_upper,
            box12_upper,
            box13_upper,
            box14_upper,
            box15_upper,
            box16_upper,
            box17_upper,
            box18_upper,
            box19_upper,
            box20_upper,
            box21_upper,
            box22_upper,
            box23_upper,
            box24_upper,
            box25_upper,
            box26_upper,
        ]

        # Case 11+: Add overlapping clusters along the sweep axis (SAP stress test)
        # These will all have similar projections on the sweep axis, stressing the sorting
        rng = np.random.Generator(np.random.PCG64(888))
        cluster_lowers = []
        cluster_uppers = []
        cluster_cutoffs = []
        cluster_groups = []
        cluster_worlds = []

        for cluster_id in range(num_clusters):
            # Random center for this cluster, but align them along x-axis for SAP stress
            # This creates a challenging scenario where many boxes overlap in the sweep direction
            x_base = 100.0 + cluster_id * 8.0
            cluster_center = np.array([x_base, rng.random() * 10.0, rng.random() * 10.0])

            # Randomly assign world (mix of specific worlds and global)
            if cluster_id < 4:
                cluster_world = -1  # First 4 clusters are global (tests duplicate prevention)
            else:
                cluster_world = cluster_id % 5  # Distribute across 5 worlds

            # Random collision group for this cluster
            cluster_group = (cluster_id % 4) + 1  # Groups 1, 2, 3, 4

            for i in range(cluster_size):
                # Create overlapping boxes with variation along sweep axis
                offset = rng.random(3) * 0.6
                # Extend along x-axis to create overlaps in sweep direction
                lower = cluster_center - 0.6 + offset
                upper = cluster_center + 0.6 + offset
                cluster_lowers.append(lower)
                cluster_uppers.append(upper)
                cluster_cutoffs.append(0.0 if i % 3 != 0 else 0.1)  # Mix of cutoffs
                cluster_groups.append(cluster_group if i % 5 != 0 else -1)  # Some shared
                cluster_worlds.append(cluster_world)

        # Case 12+: Add isolated geometries along x-axis (tests SAP correctly skips far objects)
        isolated_lowers = []
        isolated_uppers = []
        isolated_cutoffs = []
        isolated_groups = []
        isolated_worlds = []

        for i in range(num_isolated):
            # Place far apart along x-axis
            center = np.array([300.0 + i * 6.0, rng.random() * 5.0, rng.random() * 5.0])
            isolated_lowers.append(center - 0.25)
            isolated_uppers.append(center + 0.25)
            isolated_cutoffs.append(0.0)
            isolated_groups.append((i % 6) + 1)  # Varied groups
            isolated_worlds.append(i % 4)  # Distribute across worlds

        # Combine all geometries
        all_lowers = base_lowers + cluster_lowers + isolated_lowers
        all_uppers = base_uppers + cluster_uppers + isolated_uppers

        geom_bounding_box_lower = np.array(all_lowers)
        geom_bounding_box_upper = np.array(all_uppers)

        # Combine cutoffs
        base_cutoffs = [
            0.0,
            0.0,  # box1-2: exactly touching
            0.0,
            0.15,  # box3-4: gap of 0.25, cutoff 0.15 makes them overlap (combined 0.3)
            0.0,
            0.0,
            0.0,
            0.0,  # box5-8: overlapping chain
            0.0,
            0.0,
            0.0,  # box9-11: nested boxes
            0.0,
            0.0,  # box12-13: global entities (duplicate prevention critical)
            0.0,
            0.0,
            0.0,  # box14-16: mixed global/world
            0.0,
            0.0,
            0.0,
            0.0,  # box17-20: group filtering
            0.0,
            0.0,  # box21-22: different worlds
            0.0,
            0.0,  # box23-24: reverse order
            0.0,
            0.0,  # box25-26: zero group
        ]
        np_geom_cutoff = np.array(base_cutoffs + cluster_cutoffs + isolated_cutoffs, dtype=np.float32)

        # Combine collision groups
        base_groups = [
            1,
            1,  # box1-2: same group
            2,
            2,  # box3-4: same group
            1,
            1,
            1,
            1,  # box5-8: same group, chain
            1,
            1,
            1,  # box9-11: same group, nested
            -1,
            -2,  # box12-13: both negative (SHOULD collide, different negative values)
            -1,
            1,
            2,  # box14-16: global collides with both groups
            1,
            2,
            -1,
            -2,  # box17-20: 1 w/ -1, 2 w/ -1, not 1 w/ 2, not -1 w/ -2
            1,
            1,  # box21-22: same group but different worlds
            1,
            1,  # box23-24: reverse order
            0,
            0,  # box25-26: zero group
        ]
        np_collision_group = np.array(base_groups + cluster_groups + isolated_groups, dtype=np.int32)

        # Combine worlds
        base_worlds = [
            0,
            0,  # box1-2
            0,
            0,  # box3-4
            0,
            0,
            0,
            0,  # box5-8
            0,
            0,
            0,  # box9-11
            -1,
            -1,  # box12-13: BOTH global (critical for duplicate prevention)
            -1,
            1,
            2,  # box14-16: global with world-specific
            0,
            0,
            0,
            0,  # box17-20
            0,
            1,  # box21-22: different worlds
            0,
            0,  # box23-24
            0,
            0,  # box25-26
        ]
        np_shape_world = np.array(base_worlds + cluster_worlds + isolated_worlds, dtype=np.int32)

        if verbose:
            print("\n=== SAP Edge Case Test Setup ===")
            print(f"Total geometries: {ngeom}")
            for i in range(ngeom):
                print(
                    f"  Geom {i}: world={np_shape_world[i]}, group={np_collision_group[i]}, "
                    f"cutoff={np_geom_cutoff[i]:.2f}"
                )

        # Compute expected pairs using numpy
        pairs_np = find_overlapping_pairs_np(
            geom_bounding_box_lower, geom_bounding_box_upper, np_geom_cutoff, np_collision_group, np_shape_world
        )

        if verbose:
            print(f"\nExpected {len(pairs_np)} pairs from numpy verification")
            for pair in pairs_np:
                a, b = pair
                print(
                    f"  Pair ({a}, {b}): worlds ({np_shape_world[a]}, {np_shape_world[b]}) "
                    f"groups ({np_collision_group[a]}, {np_collision_group[b]})"
                )

        # Setup Warp arrays
        num_lower_tri_elements = ngeom * (ngeom - 1) // 2
        geom_lower = wp.array(geom_bounding_box_lower, dtype=wp.vec3)
        geom_upper = wp.array(geom_bounding_box_upper, dtype=wp.vec3)
        geom_cutoff = wp.array(np_geom_cutoff)
        collision_group = wp.array(np_collision_group)
        shape_world = wp.array(np_shape_world, dtype=wp.int32)
        candidate_pair_count = wp.array([0], dtype=wp.int32)
        candidate_pair = wp.array(np.zeros((num_lower_tri_elements, 2), dtype=wp.int32), dtype=wp.vec2i)

        # Initialize and launch SAP broad phase
        sap_broadphase = BroadPhaseSAP(shape_world, sort_type=sort_type)
        sap_broadphase.launch(
            geom_lower,
            geom_upper,
            geom_cutoff,
            collision_group,
            shape_world,
            ngeom,
            candidate_pair,
            candidate_pair_count,
        )

        wp.synchronize()

        # Get results
        pairs_wp = candidate_pair.numpy()
        num_candidate_pair_result = candidate_pair_count.numpy()[0]

        if verbose:
            print(f"\nWarp found {num_candidate_pair_result} pairs")
            for i in range(num_candidate_pair_result):
                pair = pairs_wp[i]
                a, b = pair[0], pair[1]
                print(
                    f"  Pair ({a}, {b}): worlds ({np_shape_world[a]}, {np_shape_world[b]}) "
                    f"groups ({np_collision_group[a]}, {np_collision_group[b]})"
                )

        # Verify: check for duplicate pairs
        pairs_wp_set = {tuple(pairs_wp[i]) for i in range(num_candidate_pair_result)}
        self.assertEqual(
            len(pairs_wp_set), num_candidate_pair_result, "Duplicate pairs detected in SAP broad phase results"
        )

        # Verify: check count matches
        if len(pairs_np) != num_candidate_pair_result:
            pairs_np_set = {tuple(pair) for pair in pairs_np}
            missing = pairs_np_set - pairs_wp_set
            extra = pairs_wp_set - pairs_np_set

            if missing:
                print(f"\nMissing pairs ({len(missing)}):")
                for pair in list(missing):
                    a, b = pair
                    print(
                        f"  ({a}, {b}): worlds ({np_shape_world[a]}, {np_shape_world[b]}) "
                        f"groups ({np_collision_group[a]}, {np_collision_group[b]})"
                    )

            if extra:
                print(f"\nExtra pairs ({len(extra)}):")
                for pair in list(extra):
                    a, b = pair
                    print(
                        f"  ({a}, {b}): worlds ({np_shape_world[a]}, {np_shape_world[b]}) "
                        f"groups ({np_collision_group[a]}, {np_collision_group[b]})"
                    )

        self.assertEqual(
            len(pairs_np), num_candidate_pair_result, f"Expected {len(pairs_np)} pairs, got {num_candidate_pair_result}"
        )

        # Verify: all Warp pairs are in numpy pairs
        pairs_np_set = {tuple(pair) for pair in pairs_np}
        for pair in pairs_wp[:num_candidate_pair_result]:
            pair_tuple = tuple(pair)
            self.assertIn(pair_tuple, pairs_np_set, f"Pair {pair_tuple} from Warp not found in numpy pairs")

        if verbose:
            print(f"\n✓ Test passed! All {len(pairs_np)} pairs matched, no duplicates.")

    def test_sap_edge_cases_segmented(self):
        """Test SAP edge cases with segmented sort."""
        self._test_sap_edge_cases_impl(SAPSortType.SEGMENTED)

    def test_sap_edge_cases_tile(self):
        """Test SAP edge cases with tile sort."""
        self._test_sap_edge_cases_impl(SAPSortType.TILE)

    def test_per_shape_contact_margin_broad_phase(self):
        """
        Test that all broad phase modes correctly handle per-shape contact margins
        by applying them during AABB overlap checks (not pre-expanded).

        Setup two spheres (A and B) at different separations from a ground plane:
        - Sphere A: small margin, should NOT be detected by broad phase when far
        - Sphere B: large margin, SHOULD be detected by broad phase when at same distance

        This tests that the broad phase kernels correctly expand AABBs by the provided
        margins during overlap testing, not requiring pre-expanded AABBs.
        """
        # Create UNEXPANDED AABBs for: ground plane + 2 spheres
        # The margins will be passed separately to test that broad phase applies them correctly

        # Ground plane AABB (infinite in XY, at z=0) WITHOUT margin
        ground_aabb_lower = wp.vec3(-1000.0, -1000.0, 0.0)
        ground_aabb_upper = wp.vec3(1000.0, 1000.0, 0.0)

        # Sphere A (radius=0.2, center at z=0.24) WITHOUT margin
        # AABB: z range = [0.24-0.2, 0.24+0.2] = [0.04, 0.44]
        sphere_a_aabb_lower = wp.vec3(-0.2, -0.2, 0.04)
        sphere_a_aabb_upper = wp.vec3(0.2, 0.2, 0.44)

        # Sphere B (radius=0.2, center at z=0.24) WITHOUT margin
        # AABB: z range = [0.04, 0.44]
        sphere_b_aabb_lower = wp.vec3(10.0 - 0.2, -0.2, 0.04)
        sphere_b_aabb_upper = wp.vec3(10.0 + 0.2, 0.2, 0.44)

        aabb_lower = wp.array([ground_aabb_lower, sphere_a_aabb_lower, sphere_b_aabb_lower], dtype=wp.vec3)
        aabb_upper = wp.array([ground_aabb_upper, sphere_a_aabb_upper, sphere_b_aabb_upper], dtype=wp.vec3)

        # Pass per-shape margins to broad phase - it will apply them during overlap checks
        # ground=0.01, sphereA=0.02, sphereB=0.06
        # With margins applied:
        # - Ground AABB becomes [-0.01, 0.01] in z
        # - Sphere A AABB becomes [0.04-0.02, 0.44+0.02] = [0.02, 0.46] - does NOT overlap ground
        # - Sphere B AABB becomes [0.04-0.06, 0.44+0.06] = [-0.02, 0.50] - DOES overlap ground
        shape_contact_margin = wp.array([0.01, 0.02, 0.06], dtype=wp.float32)

        # Use collision group 1 for all shapes (group -1 collides with everything, group 0 means no collision)
        collision_group = wp.array([1, 1, 1], dtype=wp.int32)
        shape_world = wp.array([0, 0, 0], dtype=wp.int32)

        # Test NXN broad phase
        nxn_bp = BroadPhaseAllPairs(shape_world)
        pairs_nxn = wp.zeros(100, dtype=wp.vec2i)
        pair_count_nxn = wp.zeros(1, dtype=wp.int32)

        nxn_bp.launch(
            aabb_lower,
            aabb_upper,
            shape_contact_margin,
            collision_group,
            shape_world,
            3,
            pairs_nxn,
            pair_count_nxn,
        )
        wp.synchronize()

        pairs_np = pairs_nxn.numpy()
        count_nxn = pair_count_nxn.numpy()[0]

        # Check that sphere B-ground pair is detected, but sphere A-ground is not
        has_sphere_b_ground = any((p[0] == 0 and p[1] == 2) or (p[0] == 2 and p[1] == 0) for p in pairs_np[:count_nxn])
        has_sphere_a_ground = any((p[0] == 0 and p[1] == 1) or (p[0] == 1 and p[1] == 0) for p in pairs_np[:count_nxn])

        self.assertTrue(has_sphere_b_ground, "NXN: Sphere B (large margin) should overlap ground")
        self.assertFalse(has_sphere_a_ground, "NXN: Sphere A (small margin) should NOT overlap ground")

        # Test SAP broad phase
        sap_bp = BroadPhaseSAP(shape_world)
        pairs_sap = wp.zeros(100, dtype=wp.vec2i)
        pair_count_sap = wp.zeros(1, dtype=wp.int32)

        sap_bp.launch(
            aabb_lower,
            aabb_upper,
            shape_contact_margin,
            collision_group,
            shape_world,
            3,
            pairs_sap,
            pair_count_sap,
        )
        wp.synchronize()

        pairs_np = pairs_sap.numpy()
        count_sap = pair_count_sap.numpy()[0]

        has_sphere_b_ground = any((p[0] == 0 and p[1] == 2) or (p[0] == 2 and p[1] == 0) for p in pairs_np[:count_sap])
        has_sphere_a_ground = any((p[0] == 0 and p[1] == 1) or (p[0] == 1 and p[1] == 0) for p in pairs_np[:count_sap])

        self.assertTrue(has_sphere_b_ground, "SAP: Sphere B (large margin) should overlap ground")
        self.assertFalse(has_sphere_a_ground, "SAP: Sphere A (small margin) should NOT overlap ground")


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
