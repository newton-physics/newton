# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Contact data structures for collision detection.

This module defines the core contact data structures used throughout the collision detection system.
"""

import warp as wp

# Bit flag and mask used to encode heightfield shape indices in collision pair buffers.
SHAPE_PAIR_HFIELD_BIT = wp.int32(1 << 30)
SHAPE_PAIR_INDEX_MASK = wp.int32((1 << 30) - 1)


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
        margin_a: Collision surface margin offset for shape A
        margin_b: Collision surface margin offset for shape B
        shape_a: Index of the first shape in the collision pair
        shape_b: Index of the second shape in the collision pair
        gap_sum: Pairwise summed contact gap threshold that determines if a contact should be written
        contact_stiffness: Contact stiffness. 0.0 means no stiffness was set.
        contact_damping: Contact damping scale. 0.0 means no damping was set.
        contact_friction_scale: Friction scaling factor. 0.0 means no friction was set.
        sort_sub_key: Sub-key for deterministic contact sorting (encodes edge/triangle/vertex index).
    """

    contact_point_center: wp.vec3
    contact_normal_a_to_b: wp.vec3
    contact_distance: float
    radius_eff_a: float
    radius_eff_b: float
    margin_a: float
    margin_b: float
    shape_a: int
    shape_b: int
    gap_sum: float
    contact_stiffness: float
    contact_damping: float
    contact_friction_scale: float
    sort_sub_key: int


@wp.func
def make_contact_sort_key(shape_a: int, shape_b: int, sort_sub_key: int) -> wp.int64:
    """Build a 64-bit sort key for deterministic contact ordering.

    Layout (bit 63 kept zero so int64 order matches uint64 order)::

        [62:43] shape_a      (20 bits, max 1,048,575 shapes)
        [42:23] shape_b      (20 bits, max 1,048,575 shapes)
        [22:0]  sort_sub_key (23 bits, max 8,388,607)

    Values exceeding these bit widths are silently masked.  The effective
    limits depend on upstream bit consumption in each contact path:

    - Mesh-triangle contacts: ``(tri_idx << 1) | 1`` — 22 effective bits
      for ``tri_idx`` (~4M triangles).  When expanded by the multi-contact
      path (``<< 3 | i``), this drops to 19 effective bits (~524K triangles).
    - SDF contacts: ``(edge_idx << 2) | (mode << 1)`` — 21 effective bits
      for ``edge_idx`` (~2M edges).  After multi-contact expansion
      (``<< 3``), 18 effective bits (~262K edges).
    """
    return (
        ((wp.int64(shape_a) & wp.int64(0xFFFFF)) << wp.int64(43))
        | ((wp.int64(shape_b) & wp.int64(0xFFFFF)) << wp.int64(23))
        | (wp.int64(sort_sub_key) & wp.int64(0x7FFFFF))
    )


@wp.func
def contact_passes_gap_check(
    contact_data: ContactData,
) -> bool:
    """
    Check if a contact passes the gap threshold check and should be written.

    Args:
        contact_data: ContactData struct containing contact information

    Returns:
        True if the contact distance is within the contact gap threshold, False otherwise
    """
    total_separation_needed = (
        contact_data.radius_eff_a + contact_data.radius_eff_b + contact_data.margin_a + contact_data.margin_b
    )

    # Distance calculation matching box_plane_collision
    contact_normal_a_to_b = wp.normalize(contact_data.contact_normal_a_to_b)

    a_contact_world = contact_data.contact_point_center - contact_normal_a_to_b * (
        0.5 * contact_data.contact_distance + contact_data.radius_eff_a
    )
    b_contact_world = contact_data.contact_point_center + contact_normal_a_to_b * (
        0.5 * contact_data.contact_distance + contact_data.radius_eff_b
    )

    diff = b_contact_world - a_contact_world
    distance = wp.dot(diff, contact_normal_a_to_b)
    d = distance - total_separation_needed

    return d <= contact_data.gap_sum
