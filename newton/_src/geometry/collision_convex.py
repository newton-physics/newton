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

from .contact_data import ContactData
from .mpr import create_solve_mpr
from .multicontact import create_build_manifold
from .simplex_solver import create_solve_closest_distance

_mat43f = wp.types.matrix((4, 3), wp.float32)
_mat53f = wp.types.matrix((5, 3), wp.float32)
_vec5u = wp.types.vector(5, wp.uint32)

# Single-contact types (saves registers)
_mat13f = wp.types.matrix((1, 3), wp.float32)
_vec1 = wp.types.vector(1, wp.float32)
_vec1u = wp.types.vector(1, wp.uint32)


def create_solve_convex_multi_contact(support_func: Any, writer_func: Any, post_process_contact: Any):
    """
    Factory function to create a multi-contact collision solver for convex shapes.

    This function creates a collision detector that generates up to 5 contact points
    for stable physics simulation. It combines GJK, MPR, and manifold generation:
    1. MPR for initial collision detection and signed distance (fast for overlapping shapes)
    2. GJK as fallback for separated shapes
    3. Multi-contact manifold generation for stable contact resolution

    Args:
        support_func: Support mapping function for shapes that takes
                     (geometry, direction, data_provider) and returns a support point
        writer_func: Function to write contact data (signature: (ContactData, writer_data) -> None)
        post_process_contact: Function to post-process contact data

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
        contact_threshold: float,
        skip_multi_contact: bool,
        writer_data: Any,
        contact_template: ContactData,
    ) -> int:
        """
        Compute up to 5 contact points between two convex shapes and write them directly.

        This function generates a multi-contact manifold for stable contact resolution:
        1. Runs MPR first (fast for overlapping shapes, which is the common case)
        2. Falls back to GJK if MPR detects no collision
        3. Generates multi-contact manifold via perturbed support mapping + polygon clipping
        4. Post-processes and writes each contact

        Args:
            geom_a: Shape A geometry data
            geom_b: Shape B geometry data
            orientation_a: Orientation quaternion of shape A
            orientation_b: Orientation quaternion of shape B
            position_a: World position of shape A
            position_b: World position of shape B
            sum_of_contact_offsets: Sum of contact offsets for both shapes
            data_provider: Support mapping data provider
            contact_threshold: Signed distance threshold; skip manifold if signed_distance > threshold
            skip_multi_contact: If True, write only single contact point
            writer_data: Data structure for contact writer
            contact_template: Pre-packed ContactData with static fields

        Returns:
            Number of valid contact points written (0-5)
        """
        # Enlarge a little bit to avoid contact flickering when the signed distance is close to 0
        enlarge = 1e-4
        # Try MPR first (optimized for overlapping shapes, which is the common case)
        collision, signed_distance, point, normal = wp.static(create_solve_mpr(support_func))(
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
            collision, signed_distance, point, normal = wp.static(create_solve_closest_distance(support_func))(
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
            # Write single contact directly using template
            contact_data = contact_template
            contact_data.contact_point_center = point
            contact_data.contact_normal_a_to_b = normal
            contact_data.contact_distance = signed_distance

            contact_data = post_process_contact(
                contact_data, geom_a, position_a, orientation_a, geom_b, position_b, orientation_b
            )
            writer_func(contact_data, writer_data, -1)

            return 1

        # Generate multi-contact manifold using perturbed support mapping and polygon clipping
        count = wp.static(create_build_manifold(support_func, writer_func, post_process_contact))(
            geom_a,
            geom_b,
            orientation_a,
            orientation_b,
            position_a,
            position_b,
            point - normal * (signed_distance * 0.5),  # Anchor point on shape A
            point + normal * (signed_distance * 0.5),  # Anchor point on shape B
            normal,
            data_provider,
            writer_data,
            contact_template,
        )

        return count

    return solve_convex_multi_contact


def create_solve_convex_single_contact(support_func: Any, writer_func: Any, post_process_contact: Any):
    """
    Factory function to create a single-contact collision solver for convex shapes.

    This function creates a collision detector that generates 1 contact point.
    It combines GJK and MPR but skips manifold generation:
    1. MPR for initial collision detection and signed distance (fast for overlapping shapes)
    2. GJK as fallback for separated shapes

    Args:
        support_func: Support mapping function for shapes that takes
                     (geometry, direction, data_provider) and returns a support point
        writer_func: Function to write contact data (signature: (ContactData, writer_data) -> None)
        post_process_contact: Function to post-process contact data

    Returns:
        solve_convex_single_contact function that computes a single contact point.
    """

    @wp.func
    def solve_convex_single_contact(
        geom_a: Any,
        geom_b: Any,
        orientation_a: wp.quat,
        orientation_b: wp.quat,
        position_a: wp.vec3,
        position_b: wp.vec3,
        sum_of_contact_offsets: float,
        data_provider: Any,
        contact_threshold: float,
        writer_data: Any,
        contact_template: ContactData,
    ) -> int:
        """
        Compute a single contact point between two convex shapes and write it directly.

        This function skips multi-contact manifold generation for faster performance:
        1. Runs MPR first (fast for overlapping shapes, which is the common case)
        2. Falls back to GJK if MPR detects no collision
        3. Post-processes and writes the contact

        Args:
            geom_a: Shape A geometry data
            geom_b: Shape B geometry data
            orientation_a: Orientation quaternion of shape A
            orientation_b: Orientation quaternion of shape B
            position_a: World position of shape A
            position_b: World position of shape B
            sum_of_contact_offsets: Sum of contact offsets for both shapes
            data_provider: Support mapping data provider
            contact_threshold: Signed distance threshold; skip manifold if signed_distance > threshold
            writer_data: Data structure for contact writer
            contact_template: Pre-packed ContactData with static fields

        Returns:
            Number of valid contact points written (0 or 1)
        """
        # Enlarge a little bit to avoid contact flickering when the signed distance is close to 0
        enlarge = 1e-4
        # Try MPR first (optimized for overlapping shapes, which is the common case)
        collision, signed_distance, point, normal = wp.static(create_solve_mpr(support_func))(
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
            collision, signed_distance, point, normal = wp.static(create_solve_closest_distance(support_func))(
                geom_a,
                geom_b,
                orientation_a,
                orientation_b,
                position_a,
                position_b,
                sum_of_contact_offsets,
                data_provider,
            )

        # Write single contact
        contact_data = contact_template
        contact_data.contact_point_center = point
        contact_data.contact_normal_a_to_b = normal
        contact_data.contact_distance = signed_distance

        contact_data = post_process_contact(
            contact_data, geom_a, position_a, orientation_a, geom_b, position_b, orientation_b
        )
        writer_func(contact_data, writer_data, -1)

        return 1

    return solve_convex_single_contact
