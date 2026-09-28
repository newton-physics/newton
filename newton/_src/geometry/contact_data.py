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

CONTACT_SORT_SUB_KEY_BITS = 23
# Primitive contacts use sub-keys 0..3 and generic convex manifolds use 0..4.
CONTACT_SORT_CONVEX_SUB_KEY_BITS = 3
CONTACT_SORT_MAX_SHAPE_INDEX_BITS = 20


def contact_sort_shape_index_bits(shape_count: int) -> int:
    """Return the bounded bit width needed for a contact shape index."""
    required_bits = max(max(int(shape_count) - 1, 0).bit_length(), 1)
    return min(required_bits, CONTACT_SORT_MAX_SHAPE_INDEX_BITS)


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
def make_contact_sort_key_with_bits(
    shape_a: int,
    shape_b: int,
    sort_sub_key: int,
    shape_index_bits: int,
    sub_key_bit_count: int,
) -> wp.int64:
    """Build a lexicographic contact key using explicit field widths.

    Values exceeding either field width are masked. Callers must therefore
    select widths that cover every sub-key emitted by their contact paths.
    """
    shape_bits = wp.int64(shape_index_bits)
    sub_key_bits = wp.int64(sub_key_bit_count)
    shape_mask = (wp.int64(1) << shape_bits) - wp.int64(1)
    sub_key_mask = (wp.int64(1) << sub_key_bits) - wp.int64(1)
    return (
        ((wp.int64(shape_a) & shape_mask) << (sub_key_bits + shape_bits))
        | ((wp.int64(shape_b) & shape_mask) << sub_key_bits)
        | (wp.int64(sort_sub_key) & sub_key_mask)
    )


@wp.func
def make_contact_sort_key(shape_a: int, shape_b: int, sort_sub_key: int, shape_index_bits: int) -> wp.int64:
    """Build a 64-bit sort key for deterministic contact ordering.

    The shape fields use the bit width required by the model, while the
    23-bit contact sub-key remains unchanged. Bit 63 stays zero so int64 order
    matches uint64 order::

        [22 + 2b:23 + b] shape_a      (b bits)
        [22 + b:23]      shape_b      (b bits)
        [22:0]           sort_sub_key (23 bits, max 8,388,607)

    ``b`` is at most 20, retaining the existing limit of 1,048,575 shape
    indices. Values exceeding these bit widths are silently masked. The effective
    limits depend on upstream bit consumption in each contact path:

    - Mesh-triangle contacts: ``(tri_idx << 1) | 1`` — 22 effective bits
      for ``tri_idx`` (~4M triangles).  When expanded by the multi-contact
      path (``<< 3 | i``), this drops to 19 effective bits (~524K triangles).
    - SDF contacts: ``(edge_idx << 2) | (mode << 1)`` — 21 effective bits
      for ``edge_idx`` (~2M edges).  After multi-contact expansion
      (``<< 3``), 18 effective bits (~262K edges).
    - Hydroelastic contacts: ``((voxel_idx * 5 + face_idx) << 1) | source``,
      with bit 0 distinguishing normal-bin and voxel-bin winners and bit 22
      reserved for reduction anchors — 21 effective fingerprint bits (~419K
      iso voxels).
    """
    return make_contact_sort_key_with_bits(
        shape_a,
        shape_b,
        sort_sub_key,
        shape_index_bits,
        CONTACT_SORT_SUB_KEY_BITS,
    )


@wp.func
def compute_contact_approach_speed(
    shape_a: int,
    shape_b: int,
    point_a: wp.vec3,
    point_b: wp.vec3,
    normal_a_to_b: wp.vec3,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
) -> float:
    """Return the closing speed of two shape points along a contact normal."""
    origin_a = wp.transform_get_translation(shape_transform[shape_a])
    origin_b = wp.transform_get_translation(shape_transform[shape_b])
    velocity_a = shape_linear_velocity[shape_a] + wp.cross(shape_angular_velocity[shape_a], point_a - origin_a)
    velocity_b = shape_linear_velocity[shape_b] + wp.cross(shape_angular_velocity[shape_b], point_b - origin_b)
    return wp.max(-wp.dot(velocity_b - velocity_a, normal_a_to_b), 0.0)


@wp.func
def compute_contact_predictive_score(
    shape_a: int,
    shape_b: int,
    point_a: wp.vec3,
    point_b: wp.vec3,
    normal_a_to_b: wp.vec3,
    physical_separation: float,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    collision_update_dt: float,
    max_speculative_extension: float,
) -> float:
    """Return predicted penetration after one collision-update horizon [m]."""
    approach_speed = compute_contact_approach_speed(
        shape_a,
        shape_b,
        point_a,
        point_b,
        normal_a_to_b,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
    )
    extension = wp.min(approach_speed * collision_update_dt, max_speculative_extension)
    return extension - physical_separation


@wp.func
def _contact_passes_gap_check_precomputed(
    contact_data: ContactData,
    contact_normal_a_to_b: wp.vec3,
    total_separation_needed: float,
) -> bool:
    """Check a contact gap using manifold-invariant precomputed values."""
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

    return _contact_passes_gap_check_precomputed(
        contact_data,
        contact_normal_a_to_b,
        total_separation_needed,
    )


@wp.func
def prepare_speculative_contact(contact_data: ContactData) -> tuple[wp.vec3, wp.vec3, wp.vec3, float]:
    """Return the normalized contact geometry used for speculative admission and storage."""
    normal = wp.normalize(contact_data.contact_normal_a_to_b)
    point_a = contact_data.contact_point_center - normal * (
        0.5 * contact_data.contact_distance + contact_data.radius_eff_a
    )
    point_b = contact_data.contact_point_center + normal * (
        0.5 * contact_data.contact_distance + contact_data.radius_eff_b
    )
    total_separation_needed = (
        contact_data.radius_eff_a + contact_data.radius_eff_b + contact_data.margin_a + contact_data.margin_b
    )
    separation = wp.dot(point_b - point_a, normal) - total_separation_needed
    return normal, point_a, point_b, separation


@wp.func
def contact_passes_speculative_gap_check(
    contact_data: ContactData,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    collision_update_dt: float,
    max_speculative_extension: float,
) -> bool:
    """Return whether a contact is present now or predicted before the next collision pass."""
    normal, point_a, point_b, physical_separation = prepare_speculative_contact(contact_data)
    if physical_separation <= contact_data.gap_sum:
        return True
    return (
        compute_contact_predictive_score(
            contact_data.shape_a,
            contact_data.shape_b,
            point_a,
            point_b,
            normal,
            physical_separation,
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            collision_update_dt,
            max_speculative_extension,
        )
        >= 0.0
    )
