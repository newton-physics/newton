# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic per-body summation of rigid-contact results without atomics.

Warp's deterministic mode records every floating-point atomic update, sorts the
records, and reduces them in a fixed order. Under CUDA graph replay it sorts the
whole worst-case record buffer, so contact kernels launched over the contact
buffer pay for every slot on every launch. Instead, a contact kernel can store
its result for each contact side in its own slot, and a gather kernel sums
each body's slots in ascending contact order. The grouping is computed once
per step with a stable integer sort.

Contact side ``2 * contact + side`` refers to side 0 (shape 0, body A) or
side 1 (shape 1, body B) of a contact.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from ..sim import Contacts, Model


@wp.func
def contact_side_body(shape: int, shape_body: wp.array[int]):
    """Return the body of a contact side's shape, or -1 for static shapes and missing sides."""
    if shape < 0:
        return -1
    return shape_body[shape]


@wp.kernel
def _build_contact_side_keys(
    contact_count: wp.array[int],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    shape_body: wp.array[int],
    body_count: int,
    # outputs
    keys: wp.array[wp.int32],
    sides: wp.array[wp.int32],
):
    side = wp.tid()
    contact = side // 2
    body = -1
    if contact < contact_count[0]:
        shape = contact_shape0[contact]
        if side % 2 == 1:
            shape = contact_shape1[contact]
        body = contact_side_body(shape, shape_body)
    if body < 0:
        body = body_count  # sorts after every body and is skipped
    keys[side] = body
    sides[side] = side


@wp.kernel
def _find_body_ranges(
    side_count: int,
    keys: wp.array[wp.int32],
    body_count: int,
    # outputs
    body_start: wp.array[wp.int32],
    body_end: wp.array[wp.int32],
):
    i = wp.tid()
    body = keys[i]
    if body >= body_count:
        return
    if i == 0 or keys[i - 1] != body:
        body_start[body] = i
    if i == side_count - 1 or keys[i + 1] != body:
        body_end[body] = i + 1


@dataclass
class ContactSidesByBody:
    """Rigid-contact sides grouped by the body they act on.

    The sides of body ``b`` are ``sides[body_start[b]:body_end[b]]`` in ascending
    order; the range is empty for bodies without contacts.
    """

    body_start: wp.array[wp.int32]
    body_end: wp.array[wp.int32]
    sides: wp.array[wp.int32]


def sort_contact_sides_by_body(model: Model, contacts: Contacts) -> ContactSidesByBody:
    """Group the rigid-contact sides in ``contacts`` by body.

    Graph-capturable; the result is valid until the contacts change.
    """
    side_count = 2 * contacts.rigid_contact_max
    device = model.device
    # radix_sort_pairs uses the second half of both arrays as scratch.
    keys = wp.empty(2 * side_count, dtype=wp.int32, device=device)
    sides = wp.empty(2 * side_count, dtype=wp.int32, device=device)
    body_start = wp.zeros(model.body_count, dtype=wp.int32, device=device)
    body_end = wp.zeros(model.body_count, dtype=wp.int32, device=device)
    wp.launch(
        _build_contact_side_keys,
        dim=side_count,
        inputs=[
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            model.shape_body,
            model.body_count,
        ],
        outputs=[keys, sides],
        device=device,
    )
    # The sort is stable, so each body's sides stay in contact order.
    wp.utils.radix_sort_pairs(keys, sides, side_count, end_bit=model.body_count.bit_length())
    wp.launch(
        _find_body_ranges,
        dim=side_count,
        inputs=[side_count, keys, model.body_count],
        outputs=[body_start, body_end],
        device=device,
    )
    return ContactSidesByBody(body_start, body_end, sides)


# Its own module, outside Warp's deterministic lowering: the forward pass has no atomics, and in the
# backward pass every gradient element has a single writer (each contact side belongs to one body),
# while recorded lowering cannot differentiate the read-modify-write of the body output.
@wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.NOT_GUARANTEED})
def gather_contact_side_wrenches(
    body_start: wp.array[wp.int32],
    body_end: wp.array[wp.int32],
    sides: wp.array[wp.int32],
    side_wrench: wp.array[wp.spatial_vector],
    # outputs
    body_wrench: wp.array[wp.spatial_vector],
):
    """Add each body's contact-side wrenches to ``body_wrench`` in ascending contact order."""
    body = wp.tid()
    start = body_start[body]
    end = body_end[body]
    if start == end:
        return
    total = wp.spatial_vector()
    for i in range(start, end):
        total += side_wrench[sides[i]]
    # Each thread owns its body, so a plain read-modify-write keeps earlier contributions.
    body_wrench[body] = body_wrench[body] + total
