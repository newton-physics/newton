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

# This code is based on the GJK/simplex solver implementation from Jitter Physics 2
# Original: https://github.com/notgiven688/jitterphysics2
# Copyright (c) Thorben Linneweber (MIT License)
# The code has been translated from C# to Python and modified for use in Newton.

"""
Gilbert-Johnson-Keerthi (GJK) algorithm with simplex solver for collision detection.

This module implements the GJK distance algorithm, which computes the minimum distance
between two convex shapes. GJK operates on the Minkowski difference of the shapes and
iteratively builds a simplex (1-4 vertices) that either contains the origin (indicating
collision) or gets progressively closer to it (for distance computation).

The algorithm works by:
1. Building a simplex in Minkowski space using support mapping
2. Finding the point on the simplex closest to the origin
3. Computing a new search direction toward the origin
4. Iterating until convergence or collision detection

Key features:
- Distance computation between separated shapes
- Collision detection when shapes overlap (returns signed_distance = 0)
- Feature ID tracking for contact persistence
- Barycentric coordinate computation for witness points
- Numerically stable simplex reduction using Johnson's distance subalgorithm

The implementation uses support mapping to query shape geometry, making it applicable
to any convex shape that provides a support function.
"""

from typing import Any

import warp as wp

from .mpr import Vert, create_support_map_function, vert_v

EPSILON = 1e-8


@wp.struct
class Barycentric:
    """Barycentric coordinates for simplex vertices."""

    lambda0: float
    lambda1: float
    lambda2: float
    lambda3: float


@wp.struct
class SimplexSolverAB:
    """Simplex solver for GJK algorithm."""

    v0: Vert
    v1: Vert
    v2: Vert
    v3: Vert
    barycentric: Barycentric
    usage_mask: wp.uint32


def create_solve_closest_distance(support_func: Any):
    """
    Factory function to create GJK distance solver with specific support and center functions.

    Args:
        support_func: Support mapping function for shapes

    Returns:
        GJK distance solver function
    """

    _support_map_b, minkowski_support, geometric_center = create_support_map_function(support_func)

    @wp.func
    def barycentric_get(bc: Barycentric, i: int) -> float:
        """Get barycentric coordinate by index."""
        if i == 0:
            return bc.lambda0
        elif i == 1:
            return bc.lambda1
        elif i == 2:
            return bc.lambda2
        else:
            return bc.lambda3

    @wp.func
    def barycentric_set(bc: Barycentric, i: int, value: float) -> Barycentric:
        """Set barycentric coordinate by index."""
        result = bc
        if i == 0:
            result.lambda0 = value
        elif i == 1:
            result.lambda1 = value
        elif i == 2:
            result.lambda2 = value
        else:
            result.lambda3 = value
        return result

    @wp.func
    def simplex_get_vertex(solver: SimplexSolverAB, i: int) -> Vert:
        """Get vertex by index."""
        if i == 0:
            return solver.v0
        elif i == 1:
            return solver.v1
        elif i == 2:
            return solver.v2
        else:
            return solver.v3

    @wp.func
    def simplex_set_vertex(solver: SimplexSolverAB, i: int, vertex: Vert) -> SimplexSolverAB:
        """Set vertex by index."""
        result = solver
        if i == 0:
            result.v0 = vertex
        elif i == 1:
            result.v1 = vertex
        elif i == 2:
            result.v2 = vertex
        else:
            result.v3 = vertex
        return result

    @wp.func
    def closest_segment(
        solver: SimplexSolverAB,
        i0: int,
        i1: int,
    ) -> tuple[wp.vec3, Barycentric, wp.uint32]:
        """Find closest point on line segment."""

        a = vert_v(simplex_get_vertex(solver, i0))
        b = vert_v(simplex_get_vertex(solver, i1))

        v = b - a
        vsq = wp.length_sq(v)

        degenerate = vsq < EPSILON

        # Guard division by zero in degenerate cases
        denom = vsq
        if degenerate:
            denom = EPSILON
        t = -wp.dot(a, v) / denom
        lambda0 = 1.0 - t
        lambda1 = t

        mask = (wp.uint32(1) << wp.uint32(i0)) | (wp.uint32(1) << wp.uint32(i1))

        bc = Barycentric()

        if lambda0 < 0.0 or degenerate:
            mask = wp.uint32(1) << wp.uint32(i1)
            lambda0 = 0.0
            lambda1 = 1.0
        elif lambda1 < 0.0:
            mask = wp.uint32(1) << wp.uint32(i0)
            lambda0 = 1.0
            lambda1 = 0.0

        bc = barycentric_set(bc, i0, lambda0)
        bc = barycentric_set(bc, i1, lambda1)

        return lambda0 * a + lambda1 * b, bc, mask

    @wp.func
    def closest_triangle(
        solver: SimplexSolverAB,
        i0: int,
        i1: int,
        i2: int,
    ) -> tuple[wp.vec3, Barycentric, wp.uint32]:
        """Find closest point on triangle."""

        a = vert_v(simplex_get_vertex(solver, i0))
        b = vert_v(simplex_get_vertex(solver, i1))
        c = vert_v(simplex_get_vertex(solver, i2))

        u = a - b
        v = a - c

        normal = wp.cross(u, v)

        t = wp.length_sq(normal)
        degenerate = t < EPSILON
        # Guard division by zero in degenerate cases
        denom = t
        if degenerate:
            denom = EPSILON
        it = 1.0 / denom

        c1 = wp.cross(u, a)
        c2 = wp.cross(a, v)

        lambda2 = wp.dot(c1, normal) * it
        lambda1 = wp.dot(c2, normal) * it
        lambda0 = 1.0 - lambda2 - lambda1

        best_distance = 1e30  # Large value
        closest_pt = wp.vec3(0.0, 0.0, 0.0)
        bc = Barycentric()
        mask = wp.uint32(0)

        # Check if we need to fall back to edges
        if lambda0 < 0.0 or degenerate:
            closest, bc_tmp, m = closest_segment(solver, i1, i2)
            dist = wp.length_sq(closest)
            if dist < best_distance:
                bc = bc_tmp
                mask = m
                best_distance = dist
                closest_pt = closest

        if lambda1 < 0.0 or degenerate:
            closest, bc_tmp, m = closest_segment(solver, i0, i2)
            dist = wp.length_sq(closest)
            if dist < best_distance:
                bc = bc_tmp
                mask = m
                best_distance = dist
                closest_pt = closest

        if lambda2 < 0.0 or degenerate:
            closest, bc_tmp, m = closest_segment(solver, i0, i1)
            dist = wp.length_sq(closest)
            if dist < best_distance:
                bc = bc_tmp
                mask = m
                closest_pt = closest

        if mask != wp.uint32(0):
            return closest_pt, bc, mask

        bc = barycentric_set(bc, i0, lambda0)
        bc = barycentric_set(bc, i1, lambda1)
        bc = barycentric_set(bc, i2, lambda2)

        mask = (wp.uint32(1) << wp.uint32(i0)) | (wp.uint32(1) << wp.uint32(i1)) | (wp.uint32(1) << wp.uint32(i2))
        return lambda0 * a + lambda1 * b + lambda2 * c, bc, mask

    @wp.func
    def determinant(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3) -> float:
        """Compute determinant for tetrahedron volume."""
        return wp.dot(b - a, wp.cross(c - a, d - a))

    @wp.func
    def closest_tetrahedron(
        solver: SimplexSolverAB,
    ) -> tuple[wp.vec3, Barycentric, wp.uint32]:
        """Find closest point on tetrahedron."""

        v0 = vert_v(solver.v0)
        v1 = vert_v(solver.v1)
        v2 = vert_v(solver.v2)
        v3 = vert_v(solver.v3)

        det_t = determinant(v0, v1, v2, v3)
        degenerate = wp.abs(det_t) < EPSILON
        # Guard division by zero in degenerate cases
        denom = det_t
        if degenerate:
            denom = EPSILON
        inverse_det_t = 1.0 / denom

        zero = wp.vec3(0.0, 0.0, 0.0)
        lambda0 = determinant(zero, v1, v2, v3) * inverse_det_t
        lambda1 = determinant(v0, zero, v2, v3) * inverse_det_t
        lambda2 = determinant(v0, v1, zero, v3) * inverse_det_t
        lambda3 = 1.0 - lambda0 - lambda1 - lambda2

        best_distance = 1e30  # Large value
        closest_pt = wp.vec3(0.0, 0.0, 0.0)
        bc = Barycentric()
        mask = wp.uint32(0)

        # Check faces
        if lambda0 < 0.0 or degenerate:
            closest, bc_tmp, m = closest_triangle(solver, 1, 2, 3)
            dist = wp.length_sq(closest)
            if dist < best_distance:
                bc = bc_tmp
                mask = m
                best_distance = dist
                closest_pt = closest

        if lambda1 < 0.0 or degenerate:
            closest, bc_tmp, m = closest_triangle(solver, 0, 2, 3)
            dist = wp.length_sq(closest)
            if dist < best_distance:
                bc = bc_tmp
                mask = m
                best_distance = dist
                closest_pt = closest

        if lambda2 < 0.0 or degenerate:
            closest, bc_tmp, m = closest_triangle(solver, 0, 1, 3)
            dist = wp.length_sq(closest)
            if dist < best_distance:
                bc = bc_tmp
                mask = m
                best_distance = dist
                closest_pt = closest

        if lambda3 < 0.0 or degenerate:
            closest, bc_tmp, m = closest_triangle(solver, 0, 1, 2)
            dist = wp.length_sq(closest)
            if dist < best_distance:
                bc = bc_tmp
                mask = m
                closest_pt = closest

        if mask != wp.uint32(0):
            return closest_pt, bc, mask

        bc.lambda0 = lambda0
        bc.lambda1 = lambda1
        bc.lambda2 = lambda2
        bc.lambda3 = lambda3

        mask = wp.uint32(15)  # 0b1111
        return zero, bc, mask

    @wp.func
    def simplex_reset(solver: SimplexSolverAB) -> SimplexSolverAB:
        """Reset simplex solver."""
        result = solver
        result.usage_mask = wp.uint32(0)
        return result

    @wp.func
    def simplex_get_closest(solver: SimplexSolverAB) -> tuple[wp.vec3, wp.vec3]:
        """Get closest points on both shapes."""
        point_a = wp.vec3(0.0, 0.0, 0.0)
        point_b = wp.vec3(0.0, 0.0, 0.0)

        for i in range(4):
            if (solver.usage_mask & (wp.uint32(1) << wp.uint32(i))) == wp.uint32(0):
                continue

            vertex = simplex_get_vertex(solver, i)
            bc_val = barycentric_get(solver.barycentric, i)
            point_a = point_a + bc_val * vertex.A
            point_b = point_b + bc_val * vertex.B

        return point_a, point_b

    @wp.func
    def simplex_add_vertex(
        solver: SimplexSolverAB,
        vertex: Vert,
    ) -> tuple[bool, wp.vec3, SimplexSolverAB]:
        """Add vertex to simplex and compute closest point."""
        result_solver = solver

        # Count used vertices and find free slot
        use_count = 0
        free_slot = 0
        indices = wp.vec4i(0)  # wp.static(wp.zeros(4, dtype=int))

        for i in range(4):
            if (solver.usage_mask & (wp.uint32(1) << wp.uint32(i))) != wp.uint32(0):
                indices[use_count] = i
                use_count += 1
            else:
                free_slot = i

        indices[use_count] = free_slot
        use_count += 1
        result_solver = simplex_set_vertex(result_solver, free_slot, vertex)

        closest = wp.vec3(0.0, 0.0, 0.0)

        if use_count == 1:
            i0 = indices[0]
            closest = vert_v(simplex_get_vertex(result_solver, i0))
            result_solver.usage_mask = wp.uint32(1) << wp.uint32(i0)
            result_solver.barycentric = barycentric_set(result_solver.barycentric, i0, 1.0)
            return True, closest, result_solver
        elif use_count == 2:
            i0 = indices[0]
            i1 = indices[1]
            closest, bc, mask = closest_segment(result_solver, i0, i1)
            result_solver.barycentric = bc
            result_solver.usage_mask = mask
            return True, closest, result_solver
        elif use_count == 3:
            i0 = indices[0]
            i1 = indices[1]
            i2 = indices[2]
            closest, bc, mask = closest_triangle(result_solver, i0, i1, i2)
            result_solver.barycentric = bc
            result_solver.usage_mask = mask
            return True, closest, result_solver
        elif use_count == 4:
            closest, bc, mask = closest_tetrahedron(result_solver)
            result_solver.barycentric = bc
            result_solver.usage_mask = mask
            # If mask == 15 (0b1111), origin is inside tetrahedron (overlap)
            # Return False to indicate overlap detection
            inside_tetrahedron = mask == wp.uint32(15)
            return not inside_tetrahedron, closest, result_solver

        return False, closest, result_solver

    @wp.func
    def solve_closest_distance_core(
        geom_a: Any,
        geom_b: Any,
        orientation_b: wp.quat,
        position_b: wp.vec3,
        extend: float,
        data_provider: Any,
        MAX_ITER: int = 30,
        COLLIDE_EPSILON: float = 1e-4,
    ) -> tuple[bool, wp.vec3, wp.vec3, wp.vec3, float, int, int]:
        """
        Core GJK distance algorithm implementation.

        This function computes the minimum distance between two convex shapes using the
        GJK algorithm. It builds a simplex iteratively using support mapping and finds
        the point on the simplex closest to the origin in Minkowski space.

        Assumes that shape A is located at the origin (position zero) and not rotated.
        Shape B is transformed relative to shape A using the provided orientation and position.

        Args:
            geom_a: Shape A geometry data (in local frame at origin)
            geom_b: Shape B geometry data
            orientation_b: Orientation of shape B relative to shape A
            position_b: Position of shape B relative to shape A
            extend: Contact offset extension (sum of contact offsets)
            data_provider: Support mapping data provider
            MAX_ITER: Maximum number of GJK iterations (default: 30)
            COLLIDE_EPSILON: Convergence threshold for distance computation (default: 1e-4)

        Returns:
            Tuple of:
                separated (bool): True if shapes are separated, False if overlapping
                point_a (wp.vec3): Witness point on shape A (in A's local frame)
                point_b (wp.vec3): Witness point on shape B (in A's local frame)
                normal (wp.vec3): Contact normal from A to B (in A's local frame)
                distance (float): Minimum distance between shapes (0 if overlapping)
                feature_a_id (int): Feature ID for shape A at witness point
                feature_b_id (int): Feature ID for shape B at witness point
        """
        # Initialize variables
        distance = float(0.0)
        feature_a_id = int(0)
        feature_b_id = int(0)
        point_a = wp.vec3(0.0, 0.0, 0.0)
        point_b = wp.vec3(0.0, 0.0, 0.0)
        normal = wp.vec3(0.0, 0.0, 0.0)

        # Initialize simplex solver
        simplex_solver = SimplexSolverAB()
        simplex_solver = simplex_reset(simplex_solver)

        iter_count = int(MAX_ITER)

        # Get geometric center
        center = geometric_center(geom_a, geom_b, orientation_b, position_b, data_provider)

        v = vert_v(center)
        dist_sq = wp.length_sq(v)

        last_search_dir = wp.vec3(1.0, 0.0, 0.0)

        while iter_count > 0:
            iter_count -= 1

            if dist_sq < COLLIDE_EPSILON * COLLIDE_EPSILON:
                # Shapes are overlapping
                distance = 0.0
                normal = wp.vec3(0.0, 0.0, 0.0)
                point_a, point_b = simplex_get_closest(simplex_solver)
                return False, point_a, point_b, normal, distance, feature_a_id, feature_b_id

            # Determine search direction with fallback for near-zero cases
            used_fallback = bool(False)
            search_dir = -v
            if dist_sq < 1.0e-12:
                # Near-zero direction: use fallback to avoid numerical issues
                search_dir = wp.vec3(1.0, 0.0, 0.0)
                used_fallback = bool(True)
            # Track last search direction for robust normal fallback
            last_search_dir = search_dir

            # Get support point in search direction
            w, feature_a_id, feature_b_id = minkowski_support(
                geom_a, geom_b, search_dir, orientation_b, position_b, extend, data_provider
            )

            # Check for convergence using Frank-Wolfe duality gap
            # Skip check when using fallback direction to avoid premature exit
            w_v = vert_v(w)
            if not used_fallback:
                delta_dist = wp.dot(v, v - w_v)
                if delta_dist < COLLIDE_EPSILON * wp.sqrt(dist_sq):
                    break

            # Check for duplicate vertex (numerical stalling)
            is_duplicate = bool(False)
            for i in range(4):
                if (simplex_solver.usage_mask & (wp.uint32(1) << wp.uint32(i))) != wp.uint32(0):
                    existing = simplex_get_vertex(simplex_solver, i)
                    if wp.length_sq(vert_v(existing) - w_v) < COLLIDE_EPSILON * COLLIDE_EPSILON:
                        is_duplicate = bool(True)
                        break
            if is_duplicate:
                break

            success, new_v, simplex_solver = simplex_add_vertex(simplex_solver, w)
            if not success:
                # Shapes are overlapping
                distance = 0.0
                normal = wp.vec3(0.0, 0.0, 0.0)
                point_a, point_b = simplex_get_closest(simplex_solver)
                return False, point_a, point_b, normal, distance, feature_a_id, feature_b_id

            v = new_v
            dist_sq = wp.length_sq(v)

        distance = wp.sqrt(dist_sq)
        # Compute closest points first
        point_a, point_b = simplex_get_closest(simplex_solver)

        # Prefer A->B vector if reliable; otherwise fall back to -v or last search dir
        delta = point_b - point_a
        delta_len_sq = wp.length_sq(delta)
        if delta_len_sq > EPSILON * EPSILON:
            normal = delta * (1.0 / wp.sqrt(delta_len_sq))
        elif distance > COLLIDE_EPSILON:
            # Separated but delta is tiny: use -v
            normal = v * (-1.0 / distance)
        else:
            # Overlap/near-contact: use last_search_dir, then stable axis
            nsq = wp.length_sq(last_search_dir)
            if nsq > 0.0:
                normal = last_search_dir * (1.0 / wp.sqrt(nsq))
            else:
                normal = wp.vec3(1.0, 0.0, 0.0)

        return True, point_a, point_b, normal, distance, feature_a_id, feature_b_id

    @wp.func
    def solve_closest_distance(
        geom_a: Any,
        geom_b: Any,
        orientation_a: wp.quat,
        orientation_b: wp.quat,
        position_a: wp.vec3,
        position_b: wp.vec3,
        sum_of_contact_offsets: float,
        data_provider: Any,
        MAX_ITER: int = 30,
        COLLIDE_EPSILON: float = 1e-4,
    ) -> tuple[bool, float, wp.vec3, wp.vec3, int, int]:
        """
        Solve GJK distance computation between two shapes.

        Args:
            geom_a: Shape A geometry data
            geom_b: Shape B geometry data
            orientation_a: Orientation of shape A
            orientation_b: Orientation of shape B
            position_a: Position of shape A
            position_b: Position of shape B
            sum_of_contact_offsets: Sum of contact offsets for both shapes
            data_provider: Support mapping data provider
            MAX_ITER: Maximum number of iterations for GJK algorithm
            COLLIDE_EPSILON: Small number for numerical comparisons
        Returns:
            Tuple of (collision, distance, contact point center, normal, feature A ID, feature B ID)
        """
        # Transform into reference frame of body A
        relative_orientation_b = wp.quat_inverse(orientation_a) * orientation_b
        relative_position_b = wp.quat_rotate_inv(orientation_a, position_b - position_a)

        # Perform distance test
        result = solve_closest_distance_core(
            geom_a,
            geom_b,
            relative_orientation_b,
            relative_position_b,
            sum_of_contact_offsets,
            data_provider,
            MAX_ITER,
            COLLIDE_EPSILON,
        )

        separated, point_a, point_b, normal, distance, feature_a_id, feature_b_id = result

        point = 0.5 * (point_a + point_b)

        # Transform results back to world space
        point = wp.quat_rotate(orientation_a, point) + position_a
        normal = wp.quat_rotate(orientation_a, normal)

        # Align semantics with MPR: return collision flag
        collision = not separated

        return collision, distance, point, normal, feature_a_id, feature_b_id

    return solve_closest_distance
