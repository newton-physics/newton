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

from newton._src.geometry.broad_phase_common import test_group_pair
from newton.geometry import BroadPhaseAllPairs, BroadPhaseExplicit, BroadPhaseSAP


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
    box_lower: np.ndarray, box_upper: np.ndarray, cutoff: np.ndarray, collision_group: np.ndarray
):
    """
    Brute-force n^2 algorithm to find all overlapping bounding box pairs.
    Each box is axis-aligned, defined by min (lower) and max (upper) corners.
    Returns a list of (i, j) pairs with i < j, where boxes i and j overlap.
    """
    n = box_lower.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            # Check for overlap in all three axes
            cutoff_combined = max(cutoff[i], cutoff[j])
            if not test_group_pair(int(collision_group[i]), int(collision_group[j])):
                continue

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
        num_candidate_pair = wp.array(
            [
                0,
            ],
            dtype=wp.int32,
        )
        max_candidate_pair = num_lower_tri_elements
        candidate_pair = wp.array(np.zeros((max_candidate_pair, 2), dtype=wp.int32), dtype=wp.vec2i)

        nxn_broadphase = BroadPhaseAllPairs()

        # Create shape world array with all shapes in global world (-1)
        shape_world = wp.array(np.full(ngeom, -1, dtype=np.int32), dtype=wp.int32)

        nxn_broadphase.launch(
            geom_lower,
            geom_upper,
            geom_cutoff,
            collision_group,
            shape_world,
            ngeom,
            candidate_pair,
            num_candidate_pair,
        )

        wp.synchronize()

        pairs_wp = candidate_pair.numpy()
        num_candidate_pair = num_candidate_pair.numpy()[0]

        if verbose:
            print("Warp contact pairs:")
            for i in range(num_candidate_pair):
                pair = pairs_wp[i]
                body_a, body_b = pair[0], pair[1]
                group_a = np_collision_group[body_a]
                group_b = np_collision_group[body_b]
                print(f"  Pair {i}: bodies ({body_a}, {body_b}) with collision groups ({group_a}, {group_b})")

            print("Checking if bounding boxes actually overlap:")
            for i in range(num_candidate_pair):
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

        if len(pairs_np) != num_candidate_pair:
            print(f"len(pairs_np)={len(pairs_np)}, num_candidate_pair={num_candidate_pair}")
            assert len(pairs_np) == num_candidate_pair

        # Ensure every element in pairs_wp is also present in pairs_np
        pairs_np_set = {tuple(pair) for pair in pairs_np}
        for pair in pairs_wp[:num_candidate_pair]:
            assert tuple(pair) in pairs_np_set, f"Pair {tuple(pair)} from Warp not found in numpy pairs"

        if verbose:
            print(len(pairs_np))

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
        num_candidate_pair = wp.array(
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
            num_candidate_pair,
        )

        wp.synchronize()

        pairs_wp = candidate_pair.numpy()
        num_candidate_pair = num_candidate_pair.numpy()[0]

        if verbose:
            print("Warp contact pairs:")
            for i in range(num_candidate_pair):
                pair = pairs_wp[i]
                print(f"  Pair {i}: bodies ({pair[0]}, {pair[1]})")

            print("Checking if bounding boxes actually overlap:")
            for i in range(num_candidate_pair):
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

        if len(pairs_np) != num_candidate_pair:
            print(f"len(pairs_np)={len(pairs_np)}, num_candidate_pair={num_candidate_pair}")
            assert len(pairs_np) == num_candidate_pair

        # Ensure every element in pairs_wp is also present in pairs_np
        pairs_np_set = {tuple(pair) for pair in pairs_np}
        for pair in pairs_wp[:num_candidate_pair]:
            assert tuple(pair) in pairs_np_set, f"Pair {tuple(pair)} from Warp not found in numpy pairs"

        if verbose:
            print(len(pairs_np))

    def test_sap_broadphase(self):
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

        upper_bound = ngeom + minus_one_count * num_groups

        # print(np_collision_group)

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
        num_candidate_pair = wp.array(
            [
                0,
            ],
            dtype=wp.int32,
        )
        max_candidate_pair = num_lower_tri_elements
        candidate_pair = wp.array(np.zeros((max_candidate_pair, 2), dtype=wp.int32), dtype=wp.vec2i)

        sap_broadphase = BroadPhaseSAP(
            max_broad_phase_elements=upper_bound,
            max_num_distinct_positive_groups=num_groups,
            max_num_negative_group_members=minus_one_count,
        )

        # Create shape world array with all shapes in global world (-1)
        shape_world = wp.array(np.full(ngeom, -1, dtype=np.int32), dtype=wp.int32)

        sap_broadphase.launch(
            geom_lower,
            geom_upper,
            geom_cutoff,
            collision_group,
            shape_world,
            ngeom,
            candidate_pair,
            num_candidate_pair,
        )

        wp.synchronize()

        pairs_wp = candidate_pair.numpy()
        num_candidate_pair = num_candidate_pair.numpy()[0]

        if verbose:
            print("Warp contact pairs:")
            for i in range(num_candidate_pair):
                pair = pairs_wp[i]
                body_a, body_b = pair[0], pair[1]
                group_a = np_collision_group[body_a]
                group_b = np_collision_group[body_b]
                print(f"  Pair {i}: bodies ({body_a}, {body_b}) with collision groups ({group_a}, {group_b})")

            print("Checking if bounding boxes actually overlap:")
            for i in range(num_candidate_pair):
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

        if len(pairs_np) != num_candidate_pair:
            print(f"len(pairs_np)={len(pairs_np)}, num_candidate_pair={num_candidate_pair}")
            # print("pairs_np:", pairs_np)
            # print("pairs_wp[:num_candidate_pair]:", pairs_wp[:num_candidate_pair])
            assert len(pairs_np) == num_candidate_pair

        # Ensure every element in pairs_wp is also present in pairs_np
        pairs_np_set = {tuple(pair) for pair in pairs_np}
        for pair in pairs_wp[:num_candidate_pair]:
            assert tuple(pair) in pairs_np_set, f"Pair {tuple(pair)} from Warp not found in numpy pairs"

        if verbose:
            print(len(pairs_np))


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
