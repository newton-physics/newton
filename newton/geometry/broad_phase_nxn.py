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

import warp as wp

from .broad_phase_blocks import check_aabb_overlap, proceed_broad_phase, write_pair


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
def _get_lower_triangular_indices(index: int, matrix_size: int) -> wp.vec2i:
    total = (matrix_size * (matrix_size - 1)) >> 1
    if index >= total:
        # In Warp, we can't throw, so return an invalid pair
        return wp.vec2i(-1, -1)

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
    return wp.vec2i(r, c)


@wp.kernel
def _nxn_broadphase_kernel(
    # Input arrays
    geom_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
    geom_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
    num_boxes: int,
    geom_cutoff: wp.array(dtype=float, ndim=1),  # per-geom (take the max)
    collision_group: wp.array(dtype=int, ndim=1),  # per-geom
    # Output arrays
    candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
    num_candidate_pair: wp.array(dtype=int, ndim=1),  # Size one array
    max_candidate_pair: int,
):
    elementid = wp.tid()

    pair = _get_lower_triangular_indices(elementid, num_boxes)

    geom1 = pair[0]
    geom2 = pair[1]

    if collision_group.shape[0] > 0 and not proceed_broad_phase(collision_group[geom1], collision_group[geom2]):
        return

    # wp.printf("geom1=%d, geom2=%d\n", geom1, geom2)

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


class NxNBroadPhase:
    """A broad phase collision detection class that performs N x N collision checks between all geometry pairs.

    This class performs collision detection between all possible pairs of geometries by checking for
    axis-aligned bounding box (AABB) overlaps. It uses a lower triangular matrix approach to avoid
    checking each pair twice.

    The collision checks take into account per-geometry cutoff distances and collision groups. Two geometries
    will only be considered as a candidate pair if:
    1. Their AABBs overlap when expanded by their cutoff distances
    2. Their collision groups allow interaction (determined by proceed_broad_phase())

    The class outputs an array of candidate collision pairs that need more detailed narrow phase collision
    checking.
    """

    def __init__(self):
        pass

    def launch(
        self,
        geom_bounding_box_lower_wp: wp.array(dtype=wp.vec3, ndim=1),  # Lower bounds of geometry bounding boxes
        geom_bounding_box_upper_wp: wp.array(dtype=wp.vec3, ndim=1),  # Upper bounds of geometry bounding boxes
        num_active_boxes: int,  # Number of active bounding boxes
        geom_cutoff_per_box: wp.array(dtype=float, ndim=1),  # Cutoff distance per geometry box
        collision_group_per_box: wp.array(dtype=int, ndim=1),  # Collision group ID per box
        # Outputs
        candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),  # Array to store overlapping geometry pairs
        num_candidate_pair: wp.array(dtype=int, ndim=1),
    ):
        """Launch the N x N broad phase collision detection.

        This method performs collision detection between all possible pairs of geometries by checking for
        axis-aligned bounding box (AABB) overlaps. It uses a lower triangular matrix approach to avoid
        checking each pair twice.

        Args:
            geom_bounding_box_lower_wp: Array of lower bounds for each geometry's AABB
            geom_bounding_box_upper_wp: Array of upper bounds for each geometry's AABB
            num_active_boxes: Number of active bounding boxes to check
            geom_cutoff_per_box: Array of cutoff distances for each geometry
            collision_group_per_box: Array of collision group IDs for each geometry
            candidate_pair: Output array to store overlapping geometry pairs
            num_candidate_pair: Output array to store number of overlapping pairs found

        The method will populate candidate_pair with the indices of geometry pairs whose AABBs overlap
        when expanded by their cutoff distances and whose collision groups allow interaction.
        """
        # The number of elements in the lower triangular part of an n x n matrix (excluding the diagonal)
        # is given by n * (n - 1) // 2
        num_lower_tri_elements = num_active_boxes * (num_active_boxes - 1) // 2

        max_candidate_pair = candidate_pair.shape[0]

        num_candidate_pair.zero_()

        wp.launch(
            _nxn_broadphase_kernel,
            dim=num_lower_tri_elements,
            inputs=[
                geom_bounding_box_lower_wp,
                geom_bounding_box_upper_wp,
                num_active_boxes,
                geom_cutoff_per_box,
                collision_group_per_box,
            ],
            outputs=[candidate_pair, num_candidate_pair, max_candidate_pair],
        )


class ExplicitPairsBroadPhase:
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
        geom_bounding_box_lower_wp: wp.array(dtype=wp.vec3, ndim=1),  # Lower bounds of geometry bounding boxes
        geom_bounding_box_upper_wp: wp.array(dtype=wp.vec3, ndim=1),  # Upper bounds of geometry bounding boxes
        geom_cutoff_per_box: wp.array(dtype=float, ndim=1),  # Cutoff distance per geometry box
        explicit_cancidate_geom_pairs: wp.array(dtype=wp.vec2i, ndim=1),  # Precomputed pairs to check
        num_paris_to_check: int,
        # Outputs
        candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),  # Array to store overlapping geometry pairs
        num_candidate_pair: wp.array(dtype=int, ndim=1),
    ):
        """Launch the explicit pairs broad phase collision detection.

        This method checks for AABB overlaps only between explicitly specified geometry pairs,
        rather than checking all possible pairs. It populates the candidate_pair array with
        indices of geometry pairs whose AABBs overlap when expanded by their cutoff distances.

        Args:
            geom_bounding_box_lower_wp: Array of lower bounds for geometry bounding boxes
            geom_bounding_box_upper_wp: Array of upper bounds for geometry bounding boxes
            geom_cutoff_per_box: Array of cutoff distances per geometry box
            explicit_cancidate_geom_pairs: Array of precomputed geometry pairs to check
            num_paris_to_check: Number of pairs to check
            candidate_pair: Output array to store overlapping geometry pairs
            num_candidate_pair: Output array to store number of overlapping pairs found

        The method will populate candidate_pair with the indices of geometry pairs whose AABBs overlap
        when expanded by their cutoff distances, but only checking the explicitly provided pairs.
        """

        max_candidate_pair = candidate_pair.shape[0]

        num_candidate_pair.zero_()

        wp.launch(
            kernel=_nxn_broadphase_precomputed_pairs,
            dim=num_paris_to_check,
            inputs=[
                geom_bounding_box_lower_wp,
                geom_bounding_box_upper_wp,
                geom_cutoff_per_box,
                explicit_cancidate_geom_pairs,
                candidate_pair,
                num_candidate_pair,
                max_candidate_pair,
            ],
        )
