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
def _sample_material_pair_coefficient(
    value: float, lower: float, upper: float, seed: int, world: int, gaussian: bool
) -> float:
    if lower < 0.0:
        return value
    if lower == upper:
        return lower
    state = wp.rand_init(seed, world)
    if gaussian:
        normal = wp.randn(state)
        while wp.abs(normal) > 3.0:
            normal = wp.randn(state)
        fraction = 0.5 + normal / 6.0
    else:
        fraction = wp.randf(state)
    return wp.clamp((1.0 - fraction) * lower + fraction * upper, lower, upper)


@wp.kernel(module="unique", enable_backward=False)
def _randomize_material_pair_values(
    columns: wp.array[int],
    bounds_min: wp.array[MaterialPairProperties],
    bounds_max: wp.array[MaterialPairProperties],
    seeds: wp.array2d[int],
    gaussian: bool,
    values: wp.array2d[MaterialPairProperties],
):
    world, pair = wp.tid()
    column = columns[pair]
    props = values[world, column]
    lower = bounds_min[pair]
    upper = bounds_max[pair]
    props.mu = _sample_material_pair_coefficient(props.mu, lower.mu, upper.mu, seeds[pair, 0], world, gaussian)
    props.mu_static = _sample_material_pair_coefficient(
        props.mu_static, lower.mu_static, upper.mu_static, seeds[pair, 1], world, gaussian
    )
    props.mu_torsional = _sample_material_pair_coefficient(
        props.mu_torsional, lower.mu_torsional, upper.mu_torsional, seeds[pair, 2], world, gaussian
    )
    props.mu_rolling = _sample_material_pair_coefficient(
        props.mu_rolling, lower.mu_rolling, upper.mu_rolling, seeds[pair, 3], world, gaussian
    )
    props.restitution = _sample_material_pair_coefficient(
        props.restitution, lower.restitution, upper.restitution, seeds[pair, 4], world, gaussian
    )
    values[world, column] = props


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
