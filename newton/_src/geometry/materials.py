# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse lookup for unordered contact material pairs."""

import warp as wp


def material_pair_key(material_a: int, material_b: int) -> int:
    """Pack two validated material IDs in ascending order."""
    return (min(material_a, material_b) << 32) | max(material_a, material_b)


@wp.func
def find_material_pair(keys: wp.array[wp.uint64], material_a: int, material_b: int) -> int:
    """Return the row of a registered pair, or -1 for a missing pair."""
    count = keys.shape[0]
    if count == 0 or material_a < 0 or material_b < 0:
        return -1

    key = (wp.uint64(wp.min(material_a, material_b)) << wp.uint64(32)) | wp.uint64(wp.max(material_a, material_b))
    lower = int(0)
    upper = count
    while lower < upper:
        mid = lower + (upper - lower) // 2
        candidate = keys[mid]
        if candidate == key:
            return mid
        if candidate < key:
            lower = mid + 1
        else:
            upper = mid
    return -1


@wp.struct
class MaterialPairProperties:
    """Experimental rigid contact coefficients for an unordered material pair.

    Each field defaults to the solver's usual contact behavior when stored
    as a negative value. Zero is an explicit coefficient.
    """

    mu: wp.float32
    """Coulomb friction coefficient [dimensionless]."""
    mu_static: wp.float32
    """Static friction coefficient [dimensionless]."""
    mu_torsional: wp.float32
    """Torsional friction coefficient [m]."""
    mu_rolling: wp.float32
    """Rolling friction coefficient [m]."""
    restitution: wp.float32
    """Restitution coefficient [dimensionless]."""


@wp.struct
class MaterialPairTable:
    keys: wp.array[wp.uint64]
    values: wp.array2d[MaterialPairProperties]


@wp.func
def material_pair_value(value: float, fallback: float) -> float:
    return wp.where(value >= 0.0, value, fallback)


@wp.func
def resolve_material_pair(
    pairs: MaterialPairTable, shape_material_id: wp.array[wp.int32], shape_a: int, shape_b: int, world: int
) -> MaterialPairProperties:
    if pairs.keys.shape[0] > 0 and shape_a >= 0 and shape_b >= 0:
        column = find_material_pair(pairs.keys, shape_material_id[shape_a], shape_material_id[shape_b])
        if column >= 0:
            row = 0
            if pairs.values.shape[0] > 1:
                row = wp.max(world, 0)
            return pairs.values[row, column]
    return MaterialPairProperties(-1.0, -1.0, -1.0, -1.0, -1.0)


@wp.kernel(enable_backward=False)
def resolve_contact_material_pairs(
    pairs: MaterialPairTable,
    shape_material_id: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_pairs: wp.array[MaterialPairProperties],
):
    """Resolve pair properties once for a batch of contacts."""
    tid = wp.tid()
    if tid < contact_count[0]:
        shape_a = contact_shape0[tid]
        shape_b = contact_shape1[tid]
        world = -1
        if pairs.values.shape[0] > 1 and shape_a >= 0 and shape_b >= 0:
            world = wp.max(shape_world[shape_a], shape_world[shape_b])
        contact_pairs[tid] = resolve_material_pair(pairs, shape_material_id, shape_a, shape_b, world)
