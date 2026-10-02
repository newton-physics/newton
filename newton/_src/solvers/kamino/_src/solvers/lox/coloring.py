# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Approximate graph coloring of the LOX unilateral rows.

Rows are colored so that few rows of the same color share a body. Each row
starts from a hashed color; a few repair passes then move rows to colors that
strictly decrease ``sum(occupancy^2)``, using per-body locks to serialize the
moves that touch each body. The result is an ordering of the rows of each
family by color, plus the per-body and per-world occupancy of each color.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from .types import ColoredRowsData, ColoringData, LOXProblemData

###
# Module interface
###

__all__ = ["RowColoring", "compact_colored_rows"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###


_REPAIR_PASSES = 3


_SERIAL_PREFIX_LIMIT = 64
"""Largest color count whose prefix sum runs serially; larger counts use a scan."""


_NO_COLOR = -1


_LOCK_FREE = 0x7FFFFFFF


###
# Functions
###


@wp.func
def _mix_color_key(value: wp.uint32) -> wp.uint32:
    value = (value ^ (value >> wp.uint32(16))) * wp.static(wp.uint32(0x7FEB352D))
    value = (value ^ (value >> wp.uint32(15))) * wp.static(wp.uint32(0x846CA68B))
    return value ^ (value >> wp.uint32(16))


@wp.func
def _initial_color(wid: int, local: int, bid_a: int, bid_b: int, family: int, color_count: int) -> int:
    key = wp.uint32(wid + 1) * wp.static(wp.uint32(0x9E3779B9))
    key = key ^ (wp.uint32(local + 1) * wp.static(wp.uint32(0x85EBCA6B)))
    key = key ^ (wp.uint32(bid_a + 2) * wp.static(wp.uint32(0xC2B2AE35)))
    key = key ^ (wp.uint32(bid_b + 2) * wp.static(wp.uint32(0x27D4EB2F)))
    key = key ^ (wp.uint32(family) * wp.static(wp.uint32(0x165667B1)))
    return int(_mix_color_key(key) % wp.uint32(color_count))


@wp.func
def _occupancy(occupancy: wp.array2d[wp.int32], bid: int, color: int) -> int:
    if bid >= 0:
        return occupancy[bid, color]
    return 0


@wp.func
def _choose_color(bid_a: int, bid_b: int, current: int, color_count: int, occupancy: wp.array2d[wp.int32]) -> int:
    """Return a color that strictly decreases ``sum(occupancy^2)``, or ``_NO_COLOR``."""
    two_bodies = bid_b >= 0 and bid_b != bid_a
    current_sum = _occupancy(occupancy, bid_a, current)
    body_count = int(0)
    if bid_a >= 0:
        body_count += 1
    if two_bodies:
        current_sum += _occupancy(occupancy, bid_b, current)
        body_count += 1
    best = current
    best_sum = current_sum
    best_max = wp.max(_occupancy(occupancy, bid_a, current), _occupancy(occupancy, bid_b, current))
    for color in range(color_count):
        candidate_sum = _occupancy(occupancy, bid_a, color)
        if two_bodies:
            candidate_sum += _occupancy(occupancy, bid_b, color)
        candidate_max = wp.max(_occupancy(occupancy, bid_a, color), _occupancy(occupancy, bid_b, color))
        if candidate_sum < best_sum or (candidate_sum == best_sum and candidate_max < best_max):
            best = color
            best_sum = candidate_sum
            best_max = candidate_max
    # Moving one incidence adds one to the destination and removes one from
    # the source. This is exactly the strict-improvement condition for sum(m^2).
    if best != current and best_sum + body_count < current_sum:
        return best
    return _NO_COLOR


###
# Kernels
###


@wp.kernel
def _assign_initial_colors(
    # Inputs:
    family: int,
    color_count: int,
    row_world: wp.array[wp.int32],
    row_local: wp.array[wp.int32],
    world_row_count: wp.array[wp.int32],
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    # Outputs:
    row_color: wp.array[wp.int32],
):
    row = wp.tid()
    wid = row_world[row]
    if row_local[row] >= world_row_count[wid]:
        row_color[row] = _NO_COLOR
        return
    row_color[row] = _initial_color(wid, row_local[row], body_a[row], body_b[row], family, color_count)


@wp.kernel
def _count_occupancy(
    # Inputs:
    row_world: wp.array[wp.int32],
    row_local: wp.array[wp.int32],
    world_row_count: wp.array[wp.int32],
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    row_color: wp.array[wp.int32],
    # Outputs:
    occupancy: wp.array2d[wp.int32],
    world_color_count: wp.array2d[wp.int32],
):
    row = wp.tid()
    wid = row_world[row]
    if row_local[row] >= world_row_count[wid]:
        return
    color = row_color[row]
    if world_color_count:
        wp.atomic_add(world_color_count, wid, color, 1)
    bid_a = body_a[row]
    bid_b = body_b[row]
    if bid_a >= 0:
        wp.atomic_add(occupancy, bid_a, color, 1)
    if bid_b >= 0 and bid_b != bid_a:
        wp.atomic_add(occupancy, bid_b, color, 1)


@wp.kernel
def _propose_repairs(
    # Inputs:
    color_count: int,
    key_offset: int,
    row_world: wp.array[wp.int32],
    row_local: wp.array[wp.int32],
    world_row_count: wp.array[wp.int32],
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    row_color: wp.array[wp.int32],
    occupancy: wp.array2d[wp.int32],
    # Outputs:
    proposal: wp.array[wp.int32],
    body_lock: wp.array[wp.int32],
):
    """Propose a better color and claim both bodies with the smallest row key."""
    row = wp.tid()
    wid = row_world[row]
    if row_local[row] >= world_row_count[wid]:
        proposal[row] = _NO_COLOR
        return
    bid_a = body_a[row]
    bid_b = body_b[row]
    candidate = _choose_color(bid_a, bid_b, row_color[row], color_count, occupancy)
    proposal[row] = candidate
    if candidate < 0:
        return
    key = key_offset + row
    if bid_a >= 0:
        wp.atomic_min(body_lock, bid_a, key)
    if bid_b >= 0 and bid_b != bid_a:
        wp.atomic_min(body_lock, bid_b, key)


@wp.kernel
def _commit_repairs(
    # Inputs:
    key_offset: int,
    row_world: wp.array[wp.int32],
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    proposal: wp.array[wp.int32],
    body_lock: wp.array[wp.int32],
    # Outputs:
    row_color: wp.array[wp.int32],
    occupancy: wp.array2d[wp.int32],
    world_color_count: wp.array2d[wp.int32],
):
    """Move rows that own the locks of all their bodies to their proposed color."""
    row = wp.tid()
    candidate = proposal[row]
    if candidate < 0:
        return
    key = key_offset + row
    bid_a = body_a[row]
    bid_b = body_b[row]
    two_bodies = bid_b >= 0 and bid_b != bid_a
    if (bid_a >= 0 and body_lock[bid_a] != key) or (two_bodies and body_lock[bid_b] != key):
        return
    current = row_color[row]
    wid = row_world[row]
    if world_color_count:
        wp.atomic_add(world_color_count, wid, current, -1)
        wp.atomic_add(world_color_count, wid, candidate, 1)
    if bid_a >= 0:
        wp.atomic_add(occupancy, bid_a, current, -1)
        wp.atomic_add(occupancy, bid_a, candidate, 1)
    if two_bodies:
        wp.atomic_add(occupancy, bid_b, current, -1)
        wp.atomic_add(occupancy, bid_b, candidate, 1)
    row_color[row] = candidate


@wp.kernel
def _count_colors(
    # Inputs:
    row_color: wp.array[wp.int32],
    color_count: int,
    # Outputs:
    counts: wp.array[wp.int32],
):
    row = wp.tid()
    color = row_color[row]
    if color >= 0 and color < color_count:
        wp.atomic_add(counts, color, 1)


@wp.kernel
def _prefix_color_counts(
    # Inputs:
    color_count: int,
    counts: wp.array[wp.int32],
    # Outputs:
    offsets: wp.array[wp.int32],
    cursors: wp.array[wp.int32],
):
    offset = int(0)
    for color in range(color_count):
        offsets[color] = offset
        cursors[color] = offset
        offset += counts[color]


@wp.kernel
def _scatter_world_color_order(
    # Inputs:
    family_offset: int,
    row_world: wp.array[wp.int32],
    row_local: wp.array[wp.int32],
    world_row_count: wp.array[wp.int32],
    row_color: wp.array[wp.int32],
    world_box_offset: wp.array[wp.int32],
    world_contact_offset: wp.array[wp.int32],
    world_color_count: wp.array2d[wp.int32],
    # Outputs:
    world_color_cursor: wp.array2d[wp.int32],
    world_order: wp.array[wp.int32],
):
    """Sort the active rows of each world by color, numbering the rows of a family from ``family_offset``."""
    row = wp.tid()
    wid = row_world[row]
    if row_local[row] >= world_row_count[wid]:
        return
    color = row_color[row]
    start = world_box_offset[wid] + world_contact_offset[wid]
    for previous in range(color):
        start += world_color_count[wid, previous]
    world_order[start + wp.atomic_add(world_color_cursor, wid, color, 1)] = family_offset + row


@wp.kernel
def _scatter_color_order(
    # Inputs:
    row_color: wp.array[wp.int32],
    color_count: int,
    # Outputs:
    cursors: wp.array[wp.int32],
    order: wp.array[wp.int32],
):
    row = wp.tid()
    color = row_color[row]
    if color >= 0 and color < color_count:
        order[wp.atomic_add(cursors, color, 1)] = row


###
# Launchers
###


def compact_colored_rows(rows: ColoredRowsData, color_count: int, device: wp.DeviceLike) -> None:
    """Sort the active rows of one family by color.

    Args:
        rows: Colors of the row family, whose counts, offsets, and order are updated.
        color_count: Number of colors.
        device: Device of the arrays.
    """
    rows.color_count.zero_()
    if rows.capacity == 0:
        return
    wp.launch(
        _count_colors,
        dim=rows.capacity,
        inputs=[rows.color, color_count],
        outputs=[rows.color_count],
        device=device,
    )
    if color_count <= _SERIAL_PREFIX_LIMIT:
        wp.launch(
            _prefix_color_counts,
            dim=1,
            inputs=[color_count, rows.color_count],
            outputs=[rows.color_offset, rows.cursor],
            device=device,
        )
    else:
        wp.utils.array_scan(rows.color_count, rows.color_offset, inclusive=False)
        wp.copy(rows.cursor, rows.color_offset)
    wp.launch(
        _scatter_color_order,
        dim=rows.capacity,
        inputs=[rows.color, color_count],
        outputs=[rows.cursor, rows.order],
        device=device,
    )


###
# Interfaces
###


class RowColoring:
    """Fixed-capacity coloring of the box rows and contacts of a LOX problem.

    Args:
        problem_data: Rows and body data of the LOX problem.
        data: Coloring arrays, updated by :meth:`build`.
    """

    def __init__(self, problem_data: LOXProblemData, data: ColoringData):
        self.problem_data = problem_data
        self.device = problem_data.device
        self._data = data
        self.color_count = data.color_count

    ###
    # Public API
    ###

    def build(self) -> None:
        """Color the active rows and sort each family by color."""
        self._data.occupancy.zero_()
        if self._data.world_color_count is not None:
            self._data.world_color_count.zero_()
        for family, salt, world, local, count, bid_a, bid_b in self._families():
            if family.capacity == 0:
                continue
            wp.launch(
                _assign_initial_colors,
                dim=family.capacity,
                inputs=[salt, self.color_count, world, local, count, bid_a, bid_b],
                outputs=[family.color],
                device=self.device,
            )
            wp.launch(
                _count_occupancy,
                dim=family.capacity,
                inputs=[world, local, count, bid_a, bid_b, family.color],
                outputs=[self._data.occupancy, self._data.world_color_count],
                device=self.device,
            )
        for _repair in range(_REPAIR_PASSES):
            self._data.body_lock.fill_(_LOCK_FREE)
            key_offset = 0
            for family, _salt, world, local, count, bid_a, bid_b in self._families():
                if family.capacity > 0:
                    wp.launch(
                        _propose_repairs,
                        dim=family.capacity,
                        inputs=[
                            self.color_count,
                            key_offset,
                            world,
                            local,
                            count,
                            bid_a,
                            bid_b,
                            family.color,
                            self._data.occupancy,
                        ],
                        outputs=[family.proposal, self._data.body_lock],
                        device=self.device,
                    )
                key_offset += family.capacity
            key_offset = 0
            for family, _salt, world, _local, _count, bid_a, bid_b in self._families():
                if family.capacity > 0:
                    wp.launch(
                        _commit_repairs,
                        dim=family.capacity,
                        inputs=[key_offset, world, bid_a, bid_b, family.proposal, self._data.body_lock],
                        outputs=[family.color, self._data.occupancy, self._data.world_color_count],
                        device=self.device,
                    )
                key_offset += family.capacity
        compact_colored_rows(self._data.box, self.color_count, self.device)
        compact_colored_rows(self._data.contact, self.color_count, self.device)
        if self._data.world_order is not None:
            self._build_world_order()

    ###
    # Internals
    ###

    def _build_world_order(self) -> None:
        """Sort the active rows of each world by color, box rows first and then contacts by family offset."""
        p = self.problem_data
        self._data.world_color_cursor.zero_()
        family_offset = 0
        for family, _salt, world, local, count, _bid_a, _bid_b in self._families():
            if family.capacity > 0:
                wp.launch(
                    _scatter_world_color_order,
                    dim=family.capacity,
                    inputs=[
                        family_offset,
                        world,
                        local,
                        count,
                        family.color,
                        p.box_rows.world_offset,
                        p.contact_rows.world_offset,
                        self._data.world_color_count,
                    ],
                    outputs=[self._data.world_color_cursor, self._data.world_order],
                    device=self.device,
                )
            family_offset += family.capacity

    def _families(self):
        p = self.problem_data
        return (
            (
                self._data.box,
                0,
                p.box_rows.world,
                p.box_rows.local,
                p.box_rows.world_count,
                p.box_rows.body_a,
                p.box_rows.body_b,
            ),
            (
                self._data.contact,
                1,
                p.contact_rows.world,
                p.contact_rows.local,
                p.contact_rows.world_count,
                p.contact_rows.body_a,
                p.contact_rows.body_b,
            ),
        )
