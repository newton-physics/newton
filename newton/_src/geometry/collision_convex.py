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
High-level collision detection functions for convex shapes.

This module provides the main entry points for collision detection between convex shapes,
combining GJK, MPR, and multi-contact manifold generation into easy-to-use functions.

Two main collision modes are provided:
1. Single contact: Returns one contact point with signed distance and normal
2. Multi-contact: Returns up to 5 contact points for stable physics simulation

The implementation uses a hybrid approach:
- GJK for fast separation tests (when shapes don't overlap)
- MPR for accurate signed distance and contact points (when shapes overlap)
- Perturbed support mapping + polygon clipping for multi-contact manifolds

All functions are created via factory pattern to bind a specific support mapping function,
allowing the same collision pipeline to work with any convex shape type.
"""

from typing import Any

import warp as wp

from .mpr import create_solve_mpr
from .multicontact import create_build_manifold
from .simplex_solver import create_solve_closest_distance

_mat43f = wp.types.matrix((4, 3), wp.float32)
_mat53f = wp.types.matrix((5, 3), wp.float32)
_vec5 = wp.types.vector(5, wp.float32)
_vec5i = wp.types.vector(5, wp.int32)


def create_solve_convex_multi_contact(support_func: Any):
    """
    Factory function to create a multi-contact collision solver for convex shapes.

    This function creates a collision detector that generates up to 5 contact points
    for stable physics simulation. It combines GJK, MPR, and manifold generation:
    1. MPR for initial collision detection and signed distance (fast for overlapping shapes)
    2. GJK as fallback for separated shapes
    3. Multi-contact manifold generation for stable contact resolution

    Args:
        support_func: Support mapping function for shapes that takes
                     (geometry, direction, data_provider) and returns (point, feature_id)

    Returns:
        solve_convex_multi_contact function that computes up to 5 contact points.
    """

    @wp.func
    def solve_convex_multi_contact(
        geom_a: Any,
        geom_b: Any,
        orientation_a: wp.quat,
        orientation_b: wp.quat,
        position_a: wp.vec3,
        position_b: wp.vec3,
        sum_of_contact_offsets: float,
        data_provider: Any,
        contact_threshold: float = 0.0,
        skip_multi_contact: bool = False,
    ) -> tuple[
        int,
        wp.vec3,
        _vec5,
        _mat53f,
        _vec5i,
    ]:
        """
        Compute up to 5 contact points between two convex shapes.

        This function generates a multi-contact manifold for stable contact resolution:
        1. Runs MPR first (fast for overlapping shapes, which is the common case)
        2. Falls back to GJK if MPR detects no collision
        3. Generates multi-contact manifold via perturbed support mapping + polygon clipping

        Args:
            geom_a: Shape A geometry data
            geom_b: Shape B geometry data
            orientation_a: Orientation quaternion of shape A
            orientation_b: Orientation quaternion of shape B
            position_a: World position of shape A
            position_b: World position of shape B
            sum_of_contact_offsets: Sum of contact offsets for both shapes
            data_provider: Support mapping data provider
            contact_threshold: Signed distance threshold; skip manifold if signed_distance > threshold (default: 0.0)
            skip_multi_contact: If True, return only single contact point (default: False)
        Returns:
            Tuple of:
                count (int): Number of valid contact points (0-5)
                normal (wp.vec3): Contact normal from A to B (same for all contacts)
                signed_distances (_vec5): Signed distances for each contact (negative when overlapping)
                points (_mat53f): Contact points in world space (midpoint between shapes)
                features (_vec5i): Feature IDs for contact tracking
        """
        # Enlarge a little bit to avoid contact flickering when the signed distance is close to 0
        enlarge = 1e-4
        # Try MPR first (optimized for overlapping shapes, which is the common case)
        collision, signed_distance, point, normal, feature_a_id, feature_b_id = wp.static(
            create_solve_mpr(support_func)
        )(
            geom_a,
            geom_b,
            orientation_a,
            orientation_b,
            position_a,
            position_b,
            sum_of_contact_offsets + enlarge,
            data_provider,
        )
        signed_distance += enlarge

        if not collision:
            # MPR reported no collision, fall back to GJK for separated shapes
            collision, signed_distance, point, normal, feature_a_id, feature_b_id = wp.static(
                create_solve_closest_distance(support_func)
            )(
                geom_a,
                geom_b,
                orientation_a,
                orientation_b,
                position_a,
                position_b,
                sum_of_contact_offsets,
                data_provider,
            )

        # Skip multi-contact manifold generation if requested or signed distance exceeds threshold
        if skip_multi_contact or signed_distance > contact_threshold:
            count = 1
            signed_distances = _vec5(signed_distance, 0.0, 0.0, 0.0, 0.0)
            points = _mat53f()
            points[0] = point
            features = _vec5i(0, 0, 0, 0, 0)
            return count, normal, signed_distances, points, features

        # Generate multi-contact manifold using perturbed support mapping and polygon clipping
        count, signed_distances, points, features = wp.static(create_build_manifold(support_func))(
            geom_a,
            geom_b,
            orientation_a,
            orientation_b,
            position_a,
            position_b,
            point - normal * (signed_distance * 0.5),  # Anchor point on shape A
            point + normal * (signed_distance * 0.5),  # Anchor point on shape B
            normal,
            feature_a_id,
            feature_b_id,
            data_provider,
        )

        return count, normal, signed_distances, points, features

    return solve_convex_multi_contact
