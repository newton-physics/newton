# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Provides utilities to efficiently compare discrete information across worlds."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import warp as wp

from ..core.types import int32, uint64

###
# Module interface
###

__all__ = ["DiscreteSignature", "compute_equivalence_classes"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


@dataclass
class DiscreteSignature:
    """
    Class representing a per-world discrete signature, such as the number of bodies per world, the
    joint types per world etc.

    Can be a single integer per world, or more generally a per-world segment in an integer data array.
    Two worlds are equivalent w.r.t. this signature iff their segments match (in size and contents).

    To be able to compare values of the form per-world constant + within-world index (e.g., follower body
    id per joint), a "delta" value to subtract per world can be specified (e.g., first body index per world).
    """

    num_worlds: int
    """Number of worlds"""

    data: wp.array
    """Flat data array, with type int32, concatenating a signature block for each world"""

    world_offset: wp.array | None = None
    """
    Per-world offset of the signature block into `data`, with shape (num_worlds) and type int32.
    Optional if data has exactly one entry per world.
    """

    world_size: wp.array | None = None
    """
    Per-world size of the signature block into `data`, with shape (num_worlds) and type int32.
    Optional if data has exactly one entry per world.
    """

    world_delta: wp.array | None = None
    """
    Optional per-world delta to subtract from signature values before comparison, with shape (num_worlds)
    and type int32.
    If provided, two signature coefficients are considered equivalent if coeff1 - delta1 = coeff2 - delta2.
    """

    ignore_negative: bool = False
    """
    If True, all negative values (before subtracting delta, if applicable) will be treated as equivalent
    (e.g. to allow handling data with sentinel -1 values).
    """

    def __post_init__(self):
        assert self.num_worlds > 0

        if self.world_offset is None:
            assert self.data.size == self.num_worlds, (
                "world_offset is optional only if data has exactly one entry per world"
            )
            self.world_offset = wp.array(range(self.num_worlds), device=self.data.device, dtype=wp.int32)
        else:
            assert self.world_offset.size == self.num_worlds or self.world_offset.size == self.num_worlds + 1, (
                "invalid size for world_offset"
            )

        if self.world_size is None:
            assert self.data.size == self.num_worlds, (
                "world_size is optional only if data has exactly one entry per world"
            )
            self.world_size = wp.ones(shape=self.num_worlds, device=self.data.device, dtype=wp.int32)
        else:
            assert self.world_size.size == self.num_worlds, "invalid size for world_size"

        if self.world_delta is None:
            self.world_delta = wp.zeros(shape=self.num_worlds, device=self.data.device, dtype=wp.int32)
        else:
            assert self.world_delta.size == self.num_worlds or self.world_delta.size == self.num_worlds + 1, (
                "invalid size for world_delta"
            )


###
# Kernels
###


@wp.func
def comparison_value(value: int32, delta: int32, ignore_negative: wp.bool):
    if ignore_negative and value < 0:
        return -1
    return value - delta


@wp.kernel
def equivalence_mask_kernel(
    data: wp.array[int32],
    world_offset: wp.array[int32],
    world_size: wp.array[int32],
    world_delta: wp.array[int32],
    ignore_negative: wp.bool,
    world_ref_index: wp.array[int32],
    world_equivalence_mask: wp.array[wp.bool],
):
    """Updates a mask indicating whether the signature of each world matches the one of the reference world"""
    wid = wp.tid()  # Retrieve world index

    # Early return if we already know that the world is not equivalent to the reference
    if not world_equivalence_mask[wid]:
        return

    # Get index of reference world to compare against
    ref_id = world_ref_index[wid]
    if ref_id < 0:
        return

    # Read offsets, deltas and size (we assume here the signature size to match with the reference)
    offset_w = world_offset[wid]
    offset_ref = world_offset[ref_id]
    delta_w = world_delta[wid]
    delta_ref = world_delta[ref_id]
    size = world_size[wid]
    assert world_size[ref_id] == size

    # Compare data segments for world and reference
    for i in range(size):
        value_w = comparison_value(data[offset_w + i], delta_w, ignore_negative)
        value_ref = comparison_value(data[offset_ref + i], delta_ref, ignore_negative)
        if value_w != value_ref:
            world_equivalence_mask[wid] = False
            return


@wp.kernel
def hash_kernel(
    data: wp.array[int32],
    world_offset: wp.array[int32],
    world_size: wp.array[int32],
    world_delta: wp.array[int32],
    ignore_negative: wp.bool,
    world_class_index: wp.array[int32],
    world_hash: wp.array[uint64],
):
    """Updates a per-world hash based on the signature, for worlds that don't have a class yet"""
    wid = wp.tid()  # Retrieve world index

    # Early return if a class id is already assigned to this world
    class_id = world_class_index[wid]
    if class_id >= 0:
        return

    # Read offset size and delta
    offset = world_offset[wid]
    size = world_size[wid]
    delta = world_delta[wid]

    # Combine hash of data segments for world with current hash
    h = world_hash[wid]
    prime = wp.uint64(1099511628211)
    for i in range(size):
        val = wp.uint64(comparison_value(data[offset + i], delta, ignore_negative))
        h = h ^ val
        h = h * prime
    world_hash[wid] = h


###
# Functions
###


def compute_equivalence_classes(signatures: Iterable[DiscreteSignature]) -> list[list[int]]:
    """
    For a collection of discrete per-world signatures, splits the worlds into equivalence classes
    within which all signatures are identical.

    Returns:
        Equivalence classes, as a list of lists of world indices (each sorted in ascending order).
    """
    # Convert input to list
    signatures = list(signatures)
    if len(signatures) == 0:
        raise ValueError("`signatures` cannot be empty.")

    # Helper listing ids per class, from a class id per world
    def class_ids_to_classes(class_ids, num_classes):
        classes = [[] for _ in range(num_classes)]
        for i in range(len(class_ids)):
            classes[class_ids[i]].append(i)
        return classes

    # Dimensions
    num_worlds = signatures[0].num_worlds
    assert all(sig.num_worlds == num_worlds for sig in signatures)
    device = signatures[0].data.device

    # Initialize class labels
    class_ids = [-1 for _ in range(num_worlds)]
    next_class = 0

    # Group all worlds by sizes
    sizes_np = [sig.world_size.numpy() for sig in signatures]
    num_signatures = len(sizes_np)
    world_groups = {}
    world_ref = [-1 for _ in range(num_worlds)]  # Reference per group (first world in each group)
    for i in range(num_worlds):
        world_sizes = tuple(sizes_np[sig_id][i] for sig_id in range(num_signatures))
        try:
            indices, ref = world_groups[world_sizes]
            indices.append(i)
            world_ref[i] = ref
        except KeyError:
            world_groups[world_sizes] = [i], i
    world_ref_wp = wp.array(world_ref, dtype=wp.int32, device=device)

    # Run greedy exact comparison within each size group, against the group reference
    # (leading to early exit for homogenous worlds)
    eq_mask_wp = wp.ones(shape=num_worlds, dtype=wp.bool, device=device)
    for sig in signatures:
        wp.launch(
            equivalence_mask_kernel,
            dim=num_worlds,
            inputs=[
                sig.data,
                sig.world_offset,
                sig.world_size,
                sig.world_delta,
                sig.ignore_negative,
                world_ref_wp,
                eq_mask_wp,
            ],
            device=device,
        )

    # Assign class label to all worlds that match the reference in their group
    eq_mask_np = eq_mask_wp.numpy()
    leftover_groups = []
    for indices, _ in world_groups.values():
        non_matching = []
        for index in indices:
            if eq_mask_np[index]:
                class_ids[index] = next_class
            else:
                non_matching.append(index)
        next_class += 1
        if len(non_matching) > 0:
            leftover_groups.append(non_matching)
    if len(leftover_groups) == 0:
        return class_ids_to_classes(class_ids, next_class)

    # Hash signatures of remaining worlds
    class_ids_wp = wp.array(class_ids, dtype=wp.int32, device=device)
    hashes_wp = wp.full(shape=num_worlds, value=1469598103934665603, dtype=wp.uint64, device=device)
    for sig in signatures:
        wp.launch(
            hash_kernel,
            dim=num_worlds,
            inputs=[
                sig.data,
                sig.world_offset,
                sig.world_size,
                sig.world_delta,
                sig.ignore_negative,
                class_ids_wp,
                hashes_wp,
            ],
            device=device,
        )

    # Group remaining worlds by hash
    hashes_np = hashes_wp.numpy()
    world_groups = {}
    world_ref = [-1 for _ in range(num_worlds)]
    for group_id, group in enumerate(leftover_groups):
        for i in group:
            try:
                indices, ref = world_groups[(group_id, hashes_np[i])]
                indices.append(i)
                world_ref[i] = ref
            except KeyError:
                world_groups[(group_id, hashes_np[i])] = [i], i

    while len(world_groups) > 0:
        # Run greedy exact comparison within each hash group
        world_ref_wp.assign(world_ref)
        eq_mask_wp.fill_(True)
        for sig in signatures:
            wp.launch(
                equivalence_mask_kernel,
                dim=num_worlds,
                inputs=[
                    sig.data,
                    sig.world_offset,
                    sig.world_size,
                    sig.world_delta,
                    sig.ignore_negative,
                    world_ref_wp,
                    eq_mask_wp,
                ],
                device=device,
            )

        # Assign class label to all worlds that match the reference in their group
        eq_mask_np = eq_mask_wp.numpy()
        new_groups = {}
        world_ref = [-1 for _ in range(num_worlds)]
        for old_indices, old_ref in world_groups.values():
            for index in old_indices:
                if eq_mask_np[index]:
                    class_ids[index] = next_class
                else:
                    try:
                        indices, ref = new_groups[old_ref]
                        indices.append(index)
                        world_ref[index] = ref
                    except KeyError:
                        new_groups[old_ref] = [index], index
            next_class += 1
        world_groups = new_groups

    return class_ids_to_classes(class_ids, next_class)
