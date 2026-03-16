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

Fused MPR + GJK approach with shared support functions and frame transform:
1. MPR with small inflate — exact normals for overlapping and near-touching shapes.
   Exits early for separated shapes (just a few support queries).
2. Only if MPR finds no overlap: GJK for accurate speculative contacts.

Both algorithms share support mapping code and the relative-frame coordinate
transform, reducing compiled code size and register pressure.
"""

from typing import Any

import warp as wp

from .contact_data import ContactData
from .mpr import create_solve_mpr, create_support_map_function
from .multicontact import create_build_manifold
from .simplex_solver import create_solve_closest_distance


def create_solve_convex_multi_contact(support_func: Any, writer_func: Any, post_process_contact: Any):
    """Factory: fused MPR+GJK multi-contact solver with shared support code."""

    # Create support functions ONCE — shared between MPR and GJK.
    support_funcs = create_support_map_function(support_func)
    solve_mpr = create_solve_mpr(support_func, _support_funcs=support_funcs)
    solve_gjk = create_solve_closest_distance(support_func, _support_funcs=support_funcs)

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
        # Shared relative-frame transform (computed once for both algorithms).
        relative_orientation_b = wp.quat_inverse(orientation_a) * orientation_b
        relative_position_b = wp.quat_rotate_inv(orientation_a, position_b - position_a)

        # Enlarge a little bit to avoid contact flickering when the signed distance is close to 0.
        # This ensures MPR consistently detects resting contacts, preventing alternation between
        # MPR and GJK across frames for near-touching shapes.
        enlarge = 1e-4
        # MPR with small inflate for overlapping shapes.
        # Exits early (few support queries) when shapes are separated.
        collision, point_a, point_b, normal, penetration = wp.static(solve_mpr.core)(
            geom_a,
            geom_b,
            relative_orientation_b,
            relative_position_b,
            sum_of_contact_offsets + enlarge,
            data_provider,
        )

        if collision:
            signed_distance = -penetration + enlarge
        else:
            # GJK fallback for separated shapes -- proven accurate normals/distances.
            _separated, point_a, point_b, normal, signed_distance = wp.static(solve_gjk.core)(
                geom_a,
                geom_b,
                relative_orientation_b,
                relative_position_b,
                sum_of_contact_offsets,
                data_provider,
            )

        if skip_multi_contact or signed_distance > contact_threshold:
            # Transform to world space only for the single-contact early-out.
            point = 0.5 * (point_a + point_b)
            point = wp.quat_rotate(orientation_a, point) + position_a
            normal_ws = wp.quat_rotate(orientation_a, normal)

            contact_data = contact_template
            contact_data.contact_point_center = point
            contact_data.contact_normal_a_to_b = normal_ws
            contact_data.contact_distance = signed_distance
            contact_data = post_process_contact(
                contact_data, geom_a, position_a, orientation_a, geom_b, position_b, orientation_b
            )
            writer_func(contact_data, writer_data, -1)
            return 1

        # Generate multi-contact manifold -- pass A-local-frame data directly
        # to avoid redundant world-space round-trip.
        count = wp.static(
            create_build_manifold(support_func, writer_func, post_process_contact, _support_funcs=support_funcs)
        )(
            geom_a,
            geom_b,
            orientation_a,
            position_a,
            relative_orientation_b,
            relative_position_b,
            point_a,
            point_b,
            normal,
            data_provider,
            writer_data,
            contact_template,
        )

        return count

    return solve_convex_multi_contact


def create_solve_convex_single_contact(support_func: Any, writer_func: Any, post_process_contact: Any):
    """Factory: fused MPR+GJK single-contact solver with shared support code."""

    # Create support functions ONCE — shared between MPR and GJK.
    support_funcs = create_support_map_function(support_func)
    solve_mpr = create_solve_mpr(support_func, _support_funcs=support_funcs)
    solve_gjk = create_solve_closest_distance(support_func, _support_funcs=support_funcs)

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
        # Shared relative-frame transform (computed once for both algorithms).
        relative_orientation_b = wp.quat_inverse(orientation_a) * orientation_b
        relative_position_b = wp.quat_rotate_inv(orientation_a, position_b - position_a)

        # Enlarge a little bit to avoid contact flickering when the signed distance is close to 0.
        # This ensures MPR consistently detects resting contacts, preventing alternation between
        # MPR and GJK across frames for near-touching shapes.
        enlarge = 1e-4
        # MPR with small inflate for overlapping shapes.
        collision, point_a, point_b, normal, penetration = wp.static(solve_mpr.core)(
            geom_a,
            geom_b,
            relative_orientation_b,
            relative_position_b,
            sum_of_contact_offsets + enlarge,
            data_provider,
        )

        if collision:
            signed_distance = -penetration + enlarge
        else:
            # GJK fallback for separated shapes.
            _separated, point_a, point_b, normal, signed_distance = wp.static(solve_gjk.core)(
                geom_a,
                geom_b,
                relative_orientation_b,
                relative_position_b,
                sum_of_contact_offsets,
                data_provider,
            )

        # Transform results back to world space (once).
        point = 0.5 * (point_a + point_b)
        point = wp.quat_rotate(orientation_a, point) + position_a
        normal = wp.quat_rotate(orientation_a, normal)

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
