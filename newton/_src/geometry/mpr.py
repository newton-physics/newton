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

# This code is based on the MPR implementation from Jitter Physics 2
# Original: https://github.com/notgiven688/jitterphysics2
# Copyright (c) Thorben Linneweber (MIT License)
# The code has been translated from C# to Python and modified for use in Newton.
#
# Jitter Physics 2's MPR implementation is itself based on XenoCollide.
# The XenoCollide license (zlib) is preserved in the function docstrings below
# as required by the zlib license terms.

"""
Minkowski Portal Refinement (MPR) collision detection algorithm.

This module implements the MPR algorithm for detecting collisions between convex shapes
and computing penetration depth and contact information. MPR is an alternative to the
GJK+EPA approach that can be more efficient for penetrating contacts.

The algorithm works by:
1. Constructing an initial portal (triangle) in Minkowski space that contains the origin
2. Iteratively refining the portal by moving it closer to the origin
3. Computing penetration depth and contact points once the origin is enclosed

Key features:
- Works directly with penetrating contacts (no need for EPA as a separate step)
- More numerically stable than EPA for deep penetrations
- Returns collision normal, penetration depth, and witness points
- Supports feature ID tracking for contact persistence

The implementation uses support mapping to query shape geometry, making it applicable
to any convex shape that provides a support function.
"""

from typing import Any

import warp as wp


@wp.struct
class Vert:
    """Vertex structure for MPR algorithm containing points on both shapes."""

    A: wp.vec3  # Point on shape A
    B: wp.vec3  # Point on shape B


@wp.func
def vert_v(vert: Vert) -> wp.vec3:
    """Get the Minkowski difference vector V = A - B."""
    return vert.A - vert.B


def create_support_map_function(support_func: Any):
    """
    Factory function to create support mapping functions for MPR algorithm.

    This function creates specialized support mapping functions that work in Minkowski
    space (A - B) and handle coordinate transformations between local and world space.

    Args:
        support_func: Support mapping function for individual shapes that takes
                     (geometry, direction, data_provider) and returns (point, feature_id)

    Returns:
        Tuple of three functions:
        - support_map_b: Support mapping for shape B with world space transformation
        - minkowski_support: Support mapping for Minkowski difference A - B
        - geometric_center: Computes geometric center of Minkowski difference
    """
    # Support mapping functions (these replace the MinkowskiDiff struct methods)
    @wp.func
    def support_map_b(
        geom_b: Any,
        direction: wp.vec3,
        orientation_b: wp.quat,
        position_b: wp.vec3,
        data_provider: Any,
    ) -> tuple[wp.vec3, int]:
        """
        Support mapping for shape B with transformation.

        Args:
            geom_b: Shape B geometry data
            direction: Support direction in world space
            orientation_b: Orientation of shape B
            position_b: Position of shape B
            data_provider: Support mapping data provider

        Returns:
            Tuple of (support point in world space, feature ID)
        """
        # Transform direction to local space of shape B
        tmp = wp.quat_rotate_inv(orientation_b, direction)

        # Get support point in local space
        result, feature_id = support_func(geom_b, tmp, data_provider)

        # Transform result to world space
        result = wp.quat_rotate(orientation_b, result)
        result = result + position_b

        return result, feature_id

    @wp.func
    def minkowski_support(
        geom_a: Any,
        geom_b: Any,
        direction: wp.vec3,
        orientation_b: wp.quat,
        position_b: wp.vec3,
        extend: float,
        data_provider: Any,
    ) -> tuple[Vert, int, int]:
        """
        Compute support point on Minkowski difference A - B.

        Args:
            geom_a: Shape A geometry data
            geom_b: Shape B geometry data
            direction: Support direction
            orientation_b: Orientation of shape B
            position_b: Position of shape B
            extend: Contact offset extension
            data_provider: Support mapping data provider

        Returns:
            Tuple of (Vert containing support points, feature ID A, feature ID B)
        """
        v = Vert()

        # Support point on A in positive direction
        tmp_result_a = support_func(geom_a, direction, data_provider)
        v.A = tmp_result_a[0]
        feature_a_id = tmp_result_a[1]

        # Support point on B in negative direction
        tmp_direction = -direction
        tmp_result_b = support_map_b(geom_b, tmp_direction, orientation_b, position_b, data_provider)
        v.B = tmp_result_b[0]
        feature_b_id = tmp_result_b[1]

        # Apply contact offset extension
        d = wp.normalize(direction) * extend * 0.5
        v.A = v.A + d
        v.B = v.B - d

        return v, feature_a_id, feature_b_id

    @wp.func
    def geometric_center(
        geom_a: Any,
        geom_b: Any,
        orientation_b: wp.quat,
        position_b: wp.vec3,
        data_provider: Any,
    ) -> Vert:
        """
        Compute geometric center of Minkowski difference.

        Args:
            geom_a: Shape A geometry data
            geom_b: Shape B geometry data
            orientation_b: Orientation of shape B
            position_b: Position of shape B
            data_provider: Support mapping data provider

        Returns:
            Vert containing geometric centers of both shapes
        """
        center = Vert()

        # Get geometric center of shape A
        center.A = wp.vec3(0.0)  # center_func(geom_a, data_provider)

        # Get geometric center of shape B and transform to world space
        center.B = wp.vec3(0.0)  # center_func(geom_b, data_provider)
        center.B = wp.quat_rotate(orientation_b, center.B)
        center.B = position_b + center.B

        return center

    return support_map_b, minkowski_support, geometric_center


def create_solve_mpr(support_func: Any):
    """
    Factory function to create MPR solver with specific support and center functions.

    Args:
        support_func: Support mapping function for shapes

    Returns:
        MPR solver function
    """

    support_map_b, minkowski_support, geometric_center = create_support_map_function(support_func)

    @wp.func
    def solve_mpr_core(
        geom_a: Any,
        geom_b: Any,
        orientation_b: wp.quat,
        position_b: wp.vec3,
        extend: float,
        data_provider: Any,
        MAX_ITER: int = 30,
        NUMERIC_EPSILON: float = 1e-16,
    ) -> tuple[bool, wp.vec3, wp.vec3, wp.vec3, float, int, int]:
        """
        Core MPR algorithm implementation.

            XenoCollide is available under the zlib license:

            XenoCollide Collision Detection and Physics Library
            Copyright (c) 2007-2014 Gary Snethen http://xenocollide.com

            This software is provided 'as-is', without any express or implied warranty.
            In no event will the authors be held liable for any damages arising
            from the use of this software.
            Permission is granted to anyone to use this software for any purpose,
            including commercial applications, and to alter it and redistribute it freely,
            subject to the following restrictions:

            1. The origin of this software must not be misrepresented; you must
            not claim that you wrote the original software. If you use this
            software in a product, an acknowledgment in the product documentation
            would be appreciated but is not required.
            2. Altered source versions must be plainly marked as such, and must
            not be misrepresented as being the original software.
            3. This notice may not be removed or altered from any source distribution.

            The XenoCollide implementation below is altered and not identical to the
            original. The license is kept untouched.
        """
        COLLIDE_EPSILON = NUMERIC_EPSILON

        # Initialize variables
        penetration = float(0.0)
        feature_a_id = int(0)
        feature_b_id = int(0)
        point_a = wp.vec3(0.0, 0.0, 0.0)
        point_b = wp.vec3(0.0, 0.0, 0.0)
        normal = wp.vec3(0.0, 0.0, 0.0)

        # Get geometric center
        v0 = geometric_center(geom_a, geom_b, orientation_b, position_b, data_provider)

        normal = vert_v(v0)
        if (
            wp.abs(normal[0]) < NUMERIC_EPSILON
            and wp.abs(normal[1]) < NUMERIC_EPSILON
            and wp.abs(normal[2]) < NUMERIC_EPSILON
        ):
            # Any direction is fine - add small perturbation
            v0.A = v0.A + wp.vec3(1e-05, 0.0, 0.0)

        normal = -vert_v(v0)

        # First support point
        v1, feature_a_id, feature_b_id = minkowski_support(
            geom_a, geom_b, normal, orientation_b, position_b, extend, data_provider
        )

        point_a = v1.A
        point_b = v1.B

        if wp.dot(vert_v(v1), normal) <= 0.0:
            return False, point_a, point_b, normal, penetration, feature_a_id, feature_b_id

        normal = wp.cross(vert_v(v1), vert_v(v0))

        if wp.length_sq(normal) < NUMERIC_EPSILON * NUMERIC_EPSILON:
            normal = vert_v(v1) - vert_v(v0)
            normal = wp.normalize(normal)

            temp1 = v1.A - v1.B
            penetration = wp.dot(temp1, normal)

            return True, point_a, point_b, normal, penetration, feature_a_id, feature_b_id

        # Second support point
        v2, feature_a_id, feature_b_id = minkowski_support(
            geom_a, geom_b, normal, orientation_b, position_b, extend, data_provider
        )

        if wp.dot(vert_v(v2), normal) <= 0.0:
            return False, point_a, point_b, normal, penetration, feature_a_id, feature_b_id

        # Determine whether origin is on + or - side of plane
        temp1 = vert_v(v1) - vert_v(v0)
        temp2 = vert_v(v2) - vert_v(v0)
        normal = wp.cross(temp1, temp2)

        dist = wp.dot(normal, vert_v(v0))

        # If the origin is on the - side of the plane, reverse the direction
        if dist > 0.0:
            # Swap v1 and v2
            tmp_a = v1.A
            tmp_b = v1.B
            v1.A = v2.A
            v1.B = v2.B
            v2.A = tmp_a
            v2.B = tmp_b
            normal = -normal

        phase1 = int(0)
        phase2 = int(0)
        hit = bool(False)

        # Phase One: Identify a portal
        v3 = Vert()
        while True:
            if phase1 > MAX_ITER:
                return False, point_a, point_b, normal, penetration, feature_a_id, feature_b_id

            phase1 += 1

            v3, feature_a_id, feature_b_id = minkowski_support(
                geom_a, geom_b, normal, orientation_b, position_b, extend, data_provider
            )

            if wp.dot(vert_v(v3), normal) <= 0.0:
                return False, point_a, point_b, normal, penetration, feature_a_id, feature_b_id

            # If origin is outside (v1.V(),v0.V(),v3.V()), then eliminate v2.V() and loop
            temp1 = wp.cross(vert_v(v1), vert_v(v3))
            if wp.dot(temp1, vert_v(v0)) < 0.0:
                v2 = v3
                temp1 = vert_v(v1) - vert_v(v0)
                temp2 = vert_v(v3) - vert_v(v0)
                normal = wp.cross(temp1, temp2)
                continue

            # If origin is outside (v3.V(),v0.V(),v2.V()), then eliminate v1.V() and loop
            temp1 = wp.cross(vert_v(v3), vert_v(v2))
            if wp.dot(temp1, vert_v(v0)) < 0.0:
                v1 = v3
                temp1 = vert_v(v3) - vert_v(v0)
                temp2 = vert_v(v2) - vert_v(v0)
                normal = wp.cross(temp1, temp2)
                continue

            break

        # Phase Two: Refine the portal
        v4 = Vert()
        while True:
            phase2 += 1

            # Compute normal of the wedge face
            temp1 = vert_v(v2) - vert_v(v1)
            temp2 = vert_v(v3) - vert_v(v1)
            normal = wp.cross(temp1, temp2)

            normal_sq = wp.length_sq(normal)

            # Can this happen??? Can it be handled more cleanly?
            if normal_sq < NUMERIC_EPSILON * NUMERIC_EPSILON:
                return False, point_a, point_b, normal, penetration, feature_a_id, feature_b_id

            if not hit:
                # Compute distance from origin to wedge face
                d = wp.dot(normal, vert_v(v1))
                # If the origin is inside the wedge, we have a hit
                hit = d >= 0.0

            v4, feature_a_id, feature_b_id = minkowski_support(
                geom_a, geom_b, normal, orientation_b, position_b, extend, data_provider
            )

            temp3 = vert_v(v4) - vert_v(v3)
            delta = wp.dot(temp3, normal)
            penetration = wp.dot(vert_v(v4), normal)

            # If the origin is on the surface of the wedge, return a hit
            if (
                delta * delta <= COLLIDE_EPSILON * COLLIDE_EPSILON * normal_sq
                or penetration <= 0.0
                or phase2 > MAX_ITER
            ):
                if hit:
                    inv_normal = 1.0 / wp.sqrt(normal_sq)
                    penetration *= inv_normal
                    normal = normal * inv_normal

                    # Barycentric interpolation to get witness points
                    temp3 = wp.cross(vert_v(v1), temp1)
                    gamma = wp.dot(temp3, normal) * inv_normal
                    temp3 = wp.cross(temp2, vert_v(v1))
                    beta = wp.dot(temp3, normal) * inv_normal
                    alpha = 1.0 - gamma - beta

                    point_a = alpha * v1.A + beta * v2.A + gamma * v3.A
                    point_b = alpha * v1.B + beta * v2.B + gamma * v3.B

                return hit, point_a, point_b, normal, penetration, feature_a_id, feature_b_id

            # Determine what region of the wedge the origin is in
            temp1 = wp.cross(vert_v(v4), vert_v(v0))
            dot = wp.dot(temp1, vert_v(v1))

            if dot >= 0.0:
                # Origin is outside of (v4.V(),v0.V(),v1.V())
                dot = wp.dot(temp1, vert_v(v2))
                if dot >= 0.0:
                    v1 = v4
                else:
                    v3 = v4
            else:
                # Origin is outside of (v4.V(),v0.V(),v2.V())
                dot = wp.dot(temp1, vert_v(v3))
                if dot >= 0.0:
                    v2 = v4
                else:
                    v1 = v4

    @wp.func
    def solve_mpr(
        geom_a: Any,
        geom_b: Any,
        orientation_a: wp.quat,
        orientation_b: wp.quat,
        position_a: wp.vec3,
        position_b: wp.vec3,
        sum_of_contact_offsets: float,
        data_provider: Any,
        MAX_ITER: int = 30,
        NUMERIC_EPSILON: float = 1e-16,
    ) -> tuple[bool, float, wp.vec3, wp.vec3, int, int]:
        """
        Solve MPR (Minkowski Portal Refinement) for collision detection.

        Args:
            geom_a: Shape A geometry data
            geom_b: Shape B geometry data
            orientation_a: Orientation of shape A
            orientation_b: Orientation of shape B
            position_a: Position of shape A
            position_b: Position of shape B
            sum_of_contact_offsets: Sum of contact offsets for both shapes
            data_provider: Support mapping data provider
            MAX_ITER: Maximum number of iterations for MPR algorithm
            NUMERIC_EPSILON: Small number for numerical comparisons

        Returns:
            Tuple of:
                collision detected (bool): True if shapes are colliding
                penetration (float): Penetration depth (negative indicates separation)
                contact point center (wp.vec3): Midpoint between witness points in world space
                normal (wp.vec3): Contact normal from A to B in world space
                feature A ID (int): Feature ID for shape A at contact point
                feature B ID (int): Feature ID for shape B at contact point
        """
        # Transform shape B to local space of shape A
        relative_orientation_b = wp.quat_inverse(orientation_a) * orientation_b
        relative_position_b = wp.quat_rotate_inv(orientation_a, position_b - position_a)

        # Call the core MPR algorithm
        result = solve_mpr_core(
            geom_a,
            geom_b,
            relative_orientation_b,
            relative_position_b,
            sum_of_contact_offsets,
            data_provider,
            MAX_ITER,
            NUMERIC_EPSILON,
        )

        collision, point_a, point_b, normal, penetration, feature_a_id, feature_b_id = result

        point = 0.5 * (point_a + point_b)

        # Transform results back to world space
        point = wp.quat_rotate(orientation_a, point) + position_a
        normal = wp.quat_rotate(orientation_a, normal)

        # Convert to Newton penetration convention
        penetration = -penetration

        return collision, penetration, point, normal, feature_a_id, feature_b_id

    return solve_mpr
