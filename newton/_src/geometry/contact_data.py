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
Contact data structures for collision detection.

This module defines the core contact data structures used throughout the collision detection system.
"""

import warp as wp


@wp.struct
class ContactData:
    """
    Internal contact representation for collision detection.

    This struct stores contact information between two colliding shapes before conversion
    to solver-specific formats. It serves as an intermediate representation passed between
    collision detection algorithms and contact writer functions.

    Attributes:
        contact_point_center: Center point of the contact region in world space
        contact_normal_a_to_b: Unit normal vector pointing from shape A to shape B
        contact_distance: Signed distance between shapes (negative indicates penetration)
        radius_eff_a: Effective radius of shape A (for rounded shapes like spheres/capsules)
        radius_eff_b: Effective radius of shape B (for rounded shapes like spheres/capsules)
        thickness_a: Collision thickness offset for shape A
        thickness_b: Collision thickness offset for shape B
        shape_a: Index of the first shape in the collision pair
        shape_b: Index of the second shape in the collision pair
        margin: Contact detection margin/threshold
        feature: Shape-specific feature identifier (e.g., vertex, edge, or face ID)
        feature_pair_key: Unique key for contact pair matching across timesteps
    """

    contact_point_center: wp.vec3
    contact_normal_a_to_b: wp.vec3
    contact_distance: float
    radius_eff_a: float
    radius_eff_b: float
    thickness_a: float
    thickness_b: float
    shape_a: int
    shape_b: int
    margin: float
    feature: wp.uint32
    feature_pair_key: wp.uint64
