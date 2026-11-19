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
class MaterialPairFriction:
    keys: wp.array[wp.uint64]
    mu: wp.array[wp.float32]


@wp.func
def resolve_material_pair_friction(
    pairs: MaterialPairFriction, shape_material_id: wp.array[wp.int32], shape_a: int, shape_b: int, fallback: float
) -> float:
    """Return the explicit friction coefficient when both shapes match a pair."""
    if pairs.keys.shape[0] > 0 and shape_a >= 0 and shape_b >= 0:
        row = find_material_pair(pairs.keys, shape_material_id[shape_a], shape_material_id[shape_b])
        if row >= 0:
            return pairs.mu[row]
    return fallback


@wp.kernel(enable_backward=False)
def resolve_contact_material_pair_friction(
    pairs: MaterialPairFriction,
    shape_material_id: wp.array[wp.int32],
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_pair_mu: wp.array[float],
):
    """Resolve fixed pair coefficients once for a batch of contacts."""
    tid = wp.tid()
    if tid < contact_count[0]:
        contact_pair_mu[tid] = resolve_material_pair_friction(
            pairs, shape_material_id, contact_shape0[tid], contact_shape1[tid], -1.0
        )
