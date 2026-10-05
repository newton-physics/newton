# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from fnmatch import fnmatch
from types import NoneType
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp
from warp.types import is_array

from ..sim import (
    Control,
    JointType,
    Model,
    State,
    eval_fk,
    eval_inverse_dynamics_force,
    eval_inverse_dynamics_passive,
    eval_jacobian,
    eval_mass_matrix,
)

if TYPE_CHECKING:
    from ..actuators.actuator import Actuator

AttributeFrequency = Model.AttributeFrequency


@wp.kernel
def set_model_articulation_mask_kernel(
    world_arti_mask: wp.array2d[bool],  # (world, arti) mask in ArticulationView
    view_to_model_map: wp.array2d[int],  # map (world, arti) indices to Model articulation id
    model_articulation_mask: wp.array[bool],  # output: mask of Model articulation indices
):
    """
    Set Model articulation mask from a 2D (world, arti) mask in an ArticulationView.
    """
    world, arti = wp.tid()
    if world_arti_mask[world, arti]:
        model_articulation_mask[view_to_model_map[world, arti]] = True


@wp.kernel
def set_model_articulation_mask_per_world_kernel(
    world_mask: wp.array[bool],  # world mask in ArticulationView
    view_to_model_map: wp.array2d[int],  # map (world, arti) indices to Model articulation id
    model_articulation_mask: wp.array[bool],  # output: mask of Model articulation indices
):
    """
    Set Model articulation mask from a 1D world mask in an ArticulationView.
    """
    world, arti = wp.tid()
    if world_mask[world]:
        model_articulation_mask[view_to_model_map[world, arti]] = True


# @wp.kernel
# def set_articulation_attribute_1d_kernel(
#     view_mask: wp.array2d[bool],  # (world, arti) mask in ArticulationView
#     values: Any,  # 1d array or indexedarray
#     attrib: Any,  # 1d array or indexedarray
# ):
#     i = wp.tid()
#     if view_mask[i]:
#         attrib[i] = values[i]


# @wp.kernel
# def set_articulation_attribute_2d_kernel(
#     view_mask: wp.array2d[bool],  # (world, arti) mask in ArticulationView
#     values: Any,  # 2d array or indexedarray
#     attrib: Any,  # 2d array or indexedarray
# ):
#     i, j = wp.tid()
#     if view_mask[i, j]:
#         attrib[i, j] = values[i, j]


@wp.kernel
def set_articulation_attribute_3d_kernel(
    view_mask: wp.array2d[bool],  # (world, arti) mask in ArticulationView
    values: Any,  # 3d array or indexedarray
    attrib: Any,  # 3d array or indexedarray
):
    i, j, k = wp.tid()
    if view_mask[i, j]:
        attrib[i, j, k] = values[i, j, k]


@wp.kernel
def set_articulation_attribute_4d_kernel(
    view_mask: wp.array2d[bool],  # (world, arti) mask in ArticulationView
    values: Any,  # 4d array or indexedarray
    attrib: Any,  # 4d array or indexedarray
):
    i, j, k, l = wp.tid()
    if view_mask[i, j]:
        attrib[i, j, k, l] = values[i, j, k, l]


# @wp.kernel
# def set_articulation_attribute_1d_per_world_kernel(
#     view_mask: wp.array[bool],  # world mask in ArticulationView
#     values: Any,  # 1d array or indexedarray
#     attrib: Any,  # 1d array or indexedarray
# ):
#     i = wp.tid()
#     if view_mask[i]:
#         attrib[i] = values[i]


# @wp.kernel
# def set_articulation_attribute_2d_per_world_kernel(
#     view_mask: wp.array[bool],  # world mask in ArticulationView
#     values: Any,  # 2d array or indexedarray
#     attrib: Any,  # 2d array or indexedarray
# ):
#     i, j = wp.tid()
#     if view_mask[i]:
#         attrib[i, j] = values[i, j]


@wp.kernel
def set_articulation_attribute_3d_per_world_kernel(
    view_mask: wp.array[bool],  # world mask in ArticulationView
    values: Any,  # 3d array or indexedarray
    attrib: Any,  # 3d array or indexedarray
):
    i, j, k = wp.tid()
    if view_mask[i]:
        attrib[i, j, k] = values[i, j, k]


@wp.kernel
def set_articulation_attribute_4d_per_world_kernel(
    view_mask: wp.array[bool],  # world mask in ArticulationView
    values: Any,  # 4d array or indexedarray
    attrib: Any,  # 4d array or indexedarray
):
    i, j, k, l = wp.tid()
    if view_mask[i]:
        attrib[i, j, k, l] = values[i, j, k, l]


# explicit overloads to avoid module reloading
for dtype in [float, int, wp.transform, wp.spatial_vector]:
    for src_array_type in [wp.array, wp.indexedarray]:
        for dst_array_type in [wp.array, wp.indexedarray]:
            # wp.overload(
            #     set_articulation_attribute_1d_kernel,
            #     {"values": src_array_type(dtype=dtype, ndim=1), "attrib": dst_array_type(dtype=dtype, ndim=1)},
            # )
            # wp.overload(
            #     set_articulation_attribute_2d_kernel,
            #     {"values": src_array_type(dtype=dtype, ndim=2), "attrib": dst_array_type(dtype=dtype, ndim=2)},
            # )
            wp.overload(
                set_articulation_attribute_3d_kernel,
                {"values": src_array_type(dtype=dtype, ndim=3), "attrib": dst_array_type(dtype=dtype, ndim=3)},
            )
            wp.overload(
                set_articulation_attribute_4d_kernel,
                {"values": src_array_type(dtype=dtype, ndim=4), "attrib": dst_array_type(dtype=dtype, ndim=4)},
            )
            wp.overload(
                set_articulation_attribute_3d_per_world_kernel,
                {"values": src_array_type(dtype=dtype, ndim=3), "attrib": dst_array_type(dtype=dtype, ndim=3)},
            )
            wp.overload(
                set_articulation_attribute_4d_per_world_kernel,
                {"values": src_array_type(dtype=dtype, ndim=4), "attrib": dst_array_type(dtype=dtype, ndim=4)},
            )


# ========================================================================================
# Differentiable gather kernels for indexed -> contiguous copy


@wp.kernel
def _gather_indexed_3d_kernel(
    src: Any,  # 3d wp.array (pre-indexed, has .grad)
    indices: wp.array[int],  # index mapping for dimension 2
    dst: Any,  # 3d wp.array (contiguous staging buffer, has .grad)
):
    i, j, k = wp.tid()
    dst[i, j, k] = src[i, j, indices[k]]


@wp.kernel
def _gather_indexed_4d_kernel(
    src: Any,  # 4d wp.array
    indices: wp.array[int],
    dst: Any,  # 4d wp.array
):
    i, j, k, l = wp.tid()
    dst[i, j, k, l] = src[i, j, indices[k], l]


for _dtype in [float, wp.transform, wp.spatial_vector]:
    wp.overload(
        _gather_indexed_3d_kernel,
        {"src": wp.array3d[_dtype], "dst": wp.array3d[_dtype]},
    )
    wp.overload(
        _gather_indexed_4d_kernel,
        {"src": wp.array4d[_dtype], "dst": wp.array4d[_dtype]},
    )


# ========================================================================================
# Actuator scatter/gather kernels


@wp.kernel
def build_actuator_dof_mapping_slice_kernel(
    actuator_input_indices: wp.array[wp.uint32],
    actuators_per_world: int,
    base_offset: int,
    slice_start: int,
    slice_stop: int,
    stride_within_worlds: int,
    count_per_world: int,
    dofs_per_arti: int,
    dofs_per_world: int,
    num_worlds: int,
    mapping: wp.array[int],
):
    """Build DOF-to-actuator mapping for slice-based view selection.

    Iterates over first world's actuators only, replicates pattern to all worlds.
    For each actuator, checks all articulations in the view to find matching DOF ranges.
    """
    local_idx = wp.tid()  # 0 to actuators_per_world-1

    # Get global DOF from first world's actuator entry
    global_dof = int(actuator_input_indices[local_idx])

    for arti_idx in range(count_per_world):
        arti_global_start = base_offset + arti_idx * stride_within_worlds + slice_start
        arti_global_stop = base_offset + arti_idx * stride_within_worlds + slice_stop
        if global_dof >= arti_global_start and global_dof < arti_global_stop:
            view_local_pos = arti_idx * dofs_per_arti + (global_dof - arti_global_start)

            # Replicate to all worlds
            for world_idx in range(num_worlds):
                view_pos = world_idx * dofs_per_world + view_local_pos
                actuator_idx = world_idx * actuators_per_world + local_idx
                mapping[view_pos] = actuator_idx
            break


@wp.kernel
def build_actuator_dof_mapping_indices_kernel(
    actuator_input_indices: wp.array[wp.uint32],
    view_dof_indices: wp.array[int],
    base_offset: int,
    stride_within_worlds: int,
    count_per_world: int,
    actuators_per_world: int,
    dofs_per_arti: int,
    dofs_per_world: int,
    num_worlds: int,
    mapping: wp.array[int],
):
    """Build DOF-to-actuator mapping for index-array-based view selection.

    Iterates over first world's actuators only, replicates pattern to all worlds.
    For each actuator, checks all articulations in the view to find matching DOF indices.
    """
    local_idx = wp.tid()  # 0 to actuators_per_world-1

    global_dof = int(actuator_input_indices[local_idx])

    for arti_idx in range(count_per_world):
        arti_base = base_offset + arti_idx * stride_within_worlds
        for i in range(dofs_per_arti):
            # view_dof_indices[i] is local within the articulation, add arti_base to get global
            if arti_base + view_dof_indices[i] == global_dof:
                view_local_pos = arti_idx * dofs_per_arti + i

                # Replicate to all worlds
                for world_idx in range(num_worlds):
                    view_pos = world_idx * dofs_per_world + view_local_pos
                    actuator_idx = world_idx * actuators_per_world + local_idx
                    mapping[view_pos] = actuator_idx
                break


@wp.kernel
def _gather_1d_kernel(
    src: Any,
    indices: wp.array[int],
    dst: Any,
):
    """Gather ``dst[tid] = src[indices[tid]]``. Index -1 means skip (leave dst unchanged)."""
    tid = wp.tid()
    idx = indices[tid]
    if idx >= 0:
        dst[tid] = src[idx]


@wp.kernel
def _scatter_masked_2d_kernel(
    values: Any,
    mapping: wp.array[int],
    mask: wp.array[bool],
    cols: int,
    dst: Any,
):
    """Scatter ``dst[mapping[row * cols + col]] = values[row, col]`` where ``mask[row]`` is true.

    Mapping entries of -1 are skipped.
    """
    row, col = wp.tid()
    if mask[row]:
        dst_idx = mapping[row * cols + col]
        if dst_idx >= 0:
            dst[dst_idx] = values[row, col]


# NOTE: Python slice objects are not hashable in Python < 3.12, so we use this instead.
class Slice:
    def __init__(self, start=None, stop=None):
        self.start = start
        self.stop = stop

    def __hash__(self):
        return hash((self.start, self.stop))

    def __eq__(self, other):
        return isinstance(other, Slice) and self.start == other.start and self.stop == other.stop

    def __str__(self):
        return f"({self.start}, {self.stop})"

    def get(self):
        return slice(self.start, self.stop)


class FrequencyLayout:
    def __init__(
        self,
        offset: int,
        stride_between_worlds: int,
        stride_within_worlds: int,
        value_count: int,
        indices: list[int],
        device,
    ):
        self.offset = offset  # number of values to skip at the beginning of attribute array
        self.stride_between_worlds = stride_between_worlds
        self.stride_within_worlds = stride_within_worlds
        self.value_count = value_count
        self.slice = None
        self.indices = None
        if len(indices) == 0:
            self.slice = slice(0, 0)
        elif is_contiguous_slice(indices):
            self.slice = slice(indices[0], indices[-1] + 1)
        else:
            self.indices = wp.array(indices, dtype=int, device=device)

    @property
    def is_contiguous(self):
        return self.slice is not None

    @property
    def selected_value_count(self):
        if self.slice is not None:
            return self.slice.stop - self.slice.start
        else:
            return len(self.indices)

    def is_packed(self, world_count: int, count_per_world: int) -> bool:
        """Return whether selected rows form one contiguous range across all view axes."""
        if not self.is_contiguous:
            return False
        count = self.selected_value_count
        if count == 0:
            return True
        if count_per_world > 1 and self.stride_within_worlds != count:
            return False
        return world_count <= 1 or self.stride_between_worlds == count_per_world * count

    def __str__(self):
        indices = self.indices if self.indices is not None else self.slice
        return f"FrequencyLayout(\n    offset: {self.offset}\n    stride_between_worlds: {self.stride_between_worlds}\n    stride_within_worlds: {self.stride_within_worlds}\n    indices: {indices}\n)"


def get_name_from_label(label: str):
    """Return the leaf component of a hierarchical label.

    Args:
        label: Slash-delimited label string (e.g. ``"robot/link1"``).

    Returns:
        The final path component of the label.
    """
    return label.rsplit("/", maxsplit=1)[-1]


def find_matching_ids(
    pattern: str | list[str] | re.Pattern[str] | list[int],
    labels: list[str],
    world_ids,
    world_count: int,
) -> tuple[list[list[int]], list[int]]:
    matching_ids = match_labels(labels, pattern)

    if isinstance(pattern, list) and pattern and isinstance(pattern[0], int):
        # ArticulationView derives its layouts from model order. String patterns already produce this order.
        for idx in range(1, len(matching_ids)):
            if matching_ids[idx] <= matching_ids[idx - 1]:
                raise ValueError("Articulation indices must be unique and in ascending order")
        if matching_ids[0] < 0 or matching_ids[-1] >= len(labels):
            raise ValueError(f"Articulation indices must be in range [0, {len(labels)})")

    grouped_ids = [[] for _ in range(world_count)]  # ids grouped by world (exclude world -1)
    global_ids = []  # ids in world -1
    for idx in matching_ids:
        world = world_ids[idx]
        if world == -1:
            global_ids.append(idx)
        elif world >= 0 and world < world_count:
            grouped_ids[world].append(idx)
        else:
            raise ValueError(f"World index out of range: {world}")
    return grouped_ids, global_ids


def match_labels(labels: list[str], pattern: str | list[str] | re.Pattern[str] | list[int]) -> list[int]:
    """Find indices of elements in ``labels`` that match ``pattern``.

    See :ref:`label-matching` for the pattern syntax accepted across Newton APIs.

    Args:
        labels: List of label strings to match against.
        pattern: Glob string, list of glob strings, compiled string regular expression,
            or list of integer indices. Regular expressions use full matching. Integer
            indices are returned as-is.

    Returns:
        Unique list of matching indices, or ``pattern`` itself for ``list[int]``.

    Raises:
        TypeError: If the selector type is unsupported or list elements are not all
            strings or all integers.
    """
    if isinstance(pattern, str):
        return [idx for idx, label in enumerate(labels) if fnmatch(label, pattern)]

    if isinstance(pattern, re.Pattern):
        return [idx for idx, label in enumerate(labels) if pattern.fullmatch(label) is not None]

    if not isinstance(pattern, list):
        raise TypeError(
            "Expected a glob string, list of glob strings, compiled string pattern, "
            f"or list of int indices, got: {type(pattern)}"
        )

    if len(pattern) == 0:
        return pattern

    validation_failure = False

    if isinstance(pattern[0], int):
        # fast path for list[int]
        for item in pattern:
            if not isinstance(item, int):
                validation_failure = True
                break
        if not validation_failure:
            return pattern
    elif all(isinstance(item, str) for item in pattern):
        return [idx for idx, label in enumerate(labels) if any(fnmatch(label, item) for item in pattern)]

    types = {type(item).__name__ for item in pattern}
    raise TypeError(f"Expected a list of str patterns or a list of int indices, got: {', '.join(sorted(types))}")


def all_equal(values):
    return all(x == values[0] for x in values)


def _uniform_value(values):
    """Return the common value of a NumPy array as a Python scalar, or ``None`` if values differ."""
    return values[0].item() if np.all(values == values[0]) else None


def _ragged_arange(starts, counts):
    """Return the group and value of each entry in the concatenated ranges ``[start, start + count)``."""
    groups = np.repeat(np.arange(len(counts)), counts)
    return groups, np.repeat(starts - np.cumsum(counts) + counts, counts) + np.arange(len(groups))


def _select_positions(positions, selected):
    """Map template positions to their index in ``selected``, or -1 if not selected."""
    if selected is None:
        return positions
    lookup = np.full(max(int(positions.max(initial=-1)), max(selected, default=-1)) + 1, -1)
    lookup[selected] = np.arange(len(selected))
    return lookup[positions]


def _check_layout(slots, offsets, owners, origins, world_count, count_per_world, device):
    """Check one frequency of every selected articulation against the first one.

    Each articulation (slot) must select the same number of values, with the same offsets from its
    origin and the same owning joint or link, and origins must be uniformly strided within and
    between worlds.

    Returns:
        The layout, or ``None`` with the reason it is unavailable, and whether selected counts match.
    """
    slot_count = len(origins)
    counts = np.bincount(slots, minlength=slot_count)
    count = int(counts[0])
    failed = counts != count
    uniform_count = not failed.any()
    reasons = [] if uniform_count else ["selected count differs between articulations"]
    if uniform_count:
        failed = np.zeros(slot_count, dtype=bool)
        for values in (offsets.reshape(slot_count, count), owners.reshape(slot_count, count)):
            mismatch = values != values[0]
            if mismatch.any():
                failed |= mismatch.any(axis=1)
        if failed.any():
            reasons.append("selected values or their owners differ between articulations")
    grid = origins.reshape(world_count, count_per_world)
    between = int(grid[1, 0] - grid[0, 0]) if world_count > 1 else 0
    within = int(grid[0, 1] - grid[0, 0]) if count_per_world > 1 else 0
    if count:
        misplaced = grid != grid[0, 0] + between * np.arange(world_count)[:, None] + within * np.arange(count_per_world)
        if misplaced.any():
            reasons.append("start indices are not uniformly strided")
            failed |= misplaced.ravel()
    if reasons:
        world, articulation = divmod(int(np.argmax(failed)), count_per_world)
        return None, f"{'; '.join(reasons)} (first at world {world}, articulation {articulation})", uniform_count
    offsets = offsets[:count].tolist()
    extent = offsets[-1] + 1 if offsets else 0
    # Size-1 axes are never stepped, but Warp only reports packed strides as contiguous.
    if count_per_world == 1:
        within = extent
    if world_count == 1:
        between = within * count_per_world
    return FrequencyLayout(int(grid[0, 0]), between, within, extent, offsets, device), None, True


def _validate_layouts(
    model,
    articulation_ids,
    articulation_start,
    articulation_end,
    joint_articulation,
    joint_child,
    joint_q_start,
    joint_qd_start,
    shape_body,
    selected_joints,
    selected_links,
    include_loop_closing_joints,
):
    """Check each frequency layout of the selected articulations independently.

    ``articulation_ids`` has shape ``(world_count, count_per_world)``. ``selected_joints`` and
    ``selected_links`` are template positions, or ``None`` to select all
    positions of every articulation. Links are the children of the joints an articulation owns.

    Returns:
        The :func:`_check_layout` result for each built-in frequency and for the root joint,
        coordinates, and DOFs (keys ``"root_joint"``, ``"root_coord"``, and ``"root_dof"``).
    """
    world_count, count_per_world = articulation_ids.shape
    ids = articulation_ids.ravel()
    slot_count = len(ids)
    begins = articulation_start[ids]
    ends = articulation_start[ids + 1] if include_loop_closing_joints else articulation_end[ids]
    zeros = np.zeros(slot_count, dtype=int)

    def check(slots, offsets, owners, origins):
        return _check_layout(slots, offsets, owners, origins, world_count, count_per_world, model.device)

    def selected_origins(slots, rows, fallback):
        """Return each slot's first selected row, retaining its fallback when no rows are selected."""
        origins = np.asarray(fallback).copy()
        if len(rows):
            sentinel = np.iinfo(origins.dtype).max
            candidates = np.full(slot_count, sentinel, dtype=origins.dtype)
            np.minimum.at(candidates, slots, rows)
            present = candidates != sentinel
            origins[present] = candidates[present]
        return origins

    slots, positions = _ragged_arange(zeros, ends - begins)
    owners = _select_positions(positions, selected_joints)
    selected = owners >= 0
    slots, positions, owners = slots[selected], positions[selected], owners[selected]
    results = {AttributeFrequency.JOINT: check(slots, positions, owners, begins)}
    joints = begins[slots] + positions
    for frequency, starts in (
        (AttributeFrequency.JOINT_DOF, joint_qd_start),
        (AttributeFrequency.JOINT_COORD, joint_q_start),
    ):
        entries, rows = _ragged_arange(starts[joints], starts[joints + 1] - starts[joints])
        value_slots = slots[entries]
        origins = selected_origins(value_slots, rows, starts[begins])
        results[frequency] = check(value_slots, rows - origins[value_slots], owners[entries], origins)

    root_slots = np.arange(slot_count)
    root_origins = selected_origins(root_slots, begins, begins)
    results["root_joint"] = check(root_slots, begins - root_origins, zeros, root_origins)
    for key, starts in (("root_coord", joint_q_start), ("root_dof", joint_qd_start)):
        fallback = starts[begins]
        entries, rows = _ragged_arange(fallback, starts[begins + 1] - fallback)
        origins = selected_origins(entries, rows, fallback)
        results[key] = check(entries, rows - origins[entries], np.zeros_like(entries), origins)

    slot_of = np.full(len(articulation_start), -1)
    slot_of[ids] = np.arange(slot_count)
    owned = np.flatnonzero((joint_articulation >= 0) & (joint_child >= 0))
    owned = owned[slot_of[joint_articulation[owned]] >= 0]
    keys = np.sort(slot_of[joint_articulation[owned]] * model.body_count + joint_child[owned])
    link_slots, links = np.divmod(keys[np.diff(keys, prepend=-1) != 0], model.body_count)
    counts = np.bincount(link_slots, minlength=slot_count)
    link_origins = np.full(slot_count, model.body_count)
    np.minimum.at(link_origins, link_slots, links)
    owners = _select_positions(np.arange(len(links)) - (np.cumsum(counts) - counts)[link_slots], selected_links)
    selected = owners >= 0
    selected_link_slots = link_slots[selected]
    selected_link_rows = links[selected]
    selected_link_origins = selected_origins(selected_link_slots, selected_link_rows, link_origins)
    results[AttributeFrequency.BODY] = check(
        selected_link_slots,
        selected_link_rows - selected_link_origins[selected_link_slots],
        owners[selected],
        selected_link_origins,
    )

    # shapes belong to the articulation that owns their link and are ordered by ID within it
    link_slot = np.full(model.body_count, -1)
    link_slot[links] = link_slots
    link_owner = np.full(model.body_count, -1)
    link_owner[links[selected]] = owners[selected]
    shapes = np.flatnonzero(shape_body >= 0)
    shapes = shapes[link_slot[shape_body[shapes]] >= 0]
    shapes = shapes[np.argsort(link_slot[shape_body[shapes]], kind="stable")]
    shape_slots = link_slot[shape_body[shapes]]
    shape_origins = np.full(slot_count, model.shape_count)
    np.minimum.at(shape_origins, shape_slots, shapes)
    owners = link_owner[shape_body[shapes]]
    selected = owners >= 0
    selected_shape_slots = shape_slots[selected]
    selected_shapes = shapes[selected]
    selected_shape_origins = selected_origins(selected_shape_slots, selected_shapes, shape_origins)
    results[AttributeFrequency.SHAPE] = check(
        selected_shape_slots,
        selected_shapes - selected_shape_origins[selected_shape_slots],
        owners[selected],
        selected_shape_origins,
    )
    return results


def get_world_offset(world_ids):
    for i in range(len(world_ids)):
        if world_ids[i] > -1:
            return i
    return None


def is_contiguous_slice(indices):
    n = len(indices)
    if n > 1:
        for i in range(1, n):
            if indices[i] != indices[i - 1] + 1:
                return False
    return True


class ArticulationView:
    """
    ArticulationView provides a flexible interface for selecting and manipulating
    subsets of articulations and their joints, links, and shapes within a Model.
    It supports pattern-based selection, inclusion/exclusion filters, and convenient
    attribute access and modification for simulation and control. By default,
    construction fails when any selected data cannot share one batched layout. Set
    ``allow_partial_layouts=True`` to keep using the data that does share a layout.

    With ``allow_partial_layouts=True``, public metadata is ``None`` when it cannot
    describe every selected articulation consistently:

    - A frequency count and its corresponding names and labels are ``None`` when
      selected counts differ.
    - ``joint_dof_counts``, ``joint_coord_counts``, and ``link_shapes`` are ``None``
      when one of the layouts needed for that relationship is unavailable.
    - A ``*_contiguous`` flag is ``None`` when its frequency layout is unavailable;
      otherwise it reports whether the selected rows form one contiguous range across
      all selected articulations and worlds.
    - ``root_joint_type``, ``is_fixed_base``, and ``is_floating_base`` are ``None``
      when that root metadata differs. Non-``None`` root metadata does not guarantee
      root access when the root coordinate or DOF layout differs.
    - A value in ``custom_frequency_counts`` and ``custom_frequency_labels`` is
      ``None`` when custom row counts differ. The corresponding tendon compatibility
      aliases follow those values.

    This is useful in RL and batched simulation workflows where a single policy or
    control routine operates on many parallel environments with consistent tensor shapes.

    Custom frequencies that declare articulation ownership through
    :class:`~newton.ModelBuilder.CustomFrequency` are exposed through the same
    :meth:`get_attribute` and :meth:`set_attribute` interface as built-in frequencies.

    Methods that select articulations with a mask support per-world Boolean masks
    with shape ``(world_count,)`` and per-articulation Boolean masks with shape
    ``(world_count, count_per_world)``. Per-world masks select all articulations
    in each selected world. :meth:`set_actuator_parameter` accepts only the
    per-world layout. Masks provided as Warp arrays must be on the view's device.

    Example:

    .. code-block:: python

        import re

        import newton

        view = newton.selection.ArticulationView(model, pattern="robot*")
        q = view.get_dof_positions(state)
        q_np = q.numpy()
        q_np[..., 0] = 0.0
        view.set_dof_positions(state, q_np)

        regex_view = newton.selection.ArticulationView(
            model,
            pattern=re.compile(r"/World/envs/env_[0-9]+/Robot_(A|B|C)"),
            include_links=re.compile(r"(LF|RF)_FOOT"),
        )

    The ``pattern``, ``include_joints``, ``exclude_joints``, ``include_links``,
    and ``exclude_links`` parameters accept label patterns or integer indices — see
    :ref:`label-matching`. ``pattern`` is matched against full articulation labels.
    Joint and link filters are matched against the final path component of each label.
    Their matches select template positions in every articulation, so a different
    joint or link order can select different labels in later articulations.

    Args:
        model: The model containing the articulations.
        pattern: Glob pattern, list of glob patterns, compiled regular-expression pattern,
            or list of absolute articulation indices. Regular expressions use full matching.
            Indices must be unique and in ascending order.
        include_joints: Glob pattern, list of glob patterns, compiled regular-expression
            pattern, or list of joint indices to include. Integer indices must be in
            ascending order.
        exclude_joints: Glob pattern, list of glob patterns, compiled regular-expression
            pattern, or list of joint indices to exclude.
        include_links: Glob pattern, list of glob patterns, compiled regular-expression
            pattern, or list of link indices to include. Integer indices must be in
            ascending order.
        exclude_links: Glob pattern, list of glob patterns, compiled regular-expression
            pattern, or list of link indices to exclude.
        include_joint_types: List of joint types to include.
        exclude_joint_types: List of joint types to exclude.
        include_loop_closing_joints: If True, include converted loop-closing joints.
        allow_partial_layouts: If True, construct the view when only some selected
            data has a common layout. Access to data without a common layout raises
            :class:`AttributeError`.
        verbose: If True, prints selection summary.
    """

    def __init__(
        self,
        model: Model,
        pattern: str | list[str] | re.Pattern[str] | list[int],
        *,
        include_joints: str | list[str] | re.Pattern[str] | list[int] | None = None,
        exclude_joints: str | list[str] | re.Pattern[str] | list[int] | None = None,
        include_links: str | list[str] | re.Pattern[str] | list[int] | None = None,
        exclude_links: str | list[str] | re.Pattern[str] | list[int] | None = None,
        include_joint_types: list[int] | None = None,
        exclude_joint_types: list[int] | None = None,
        include_loop_closing_joints: bool = False,
        allow_partial_layouts: bool = False,
        verbose: bool | None = None,
    ):
        self.model = model
        self.device = model.device
        self._attribute_array_cache = {}
        self._actuator_dof_mapping_cache = {}

        if verbose is None:
            verbose = wp.config.log_level <= wp.LOG_DEBUG

        for parameter_name, indices in (("include_joints", include_joints), ("include_links", include_links)):
            if (
                isinstance(indices, list)
                and all(isinstance(index, int) for index in indices)
                and any(indices[i] < indices[i - 1] for i in range(1, len(indices)))
            ):
                raise ValueError(f"ArticulationView({parameter_name}=...) indices must be in ascending order")

        # FIXME: avoid/reduce this readback?
        model_articulation_start = model.articulation_start.numpy()
        model_articulation_end = model.articulation_end.numpy()
        model_articulation_world = model.articulation_world.numpy()
        model_joint_type = model.joint_type.numpy()
        model_joint_child = model.joint_child.numpy()
        model_joint_q_start = model.joint_q_start.numpy()
        model_joint_qd_start = model.joint_qd_start.numpy()
        model_joint_articulation = model.joint_articulation.numpy()
        model_shape_body = model.shape_body.numpy()

        # get articulation ids grouped by world
        articulation_ids, global_articulation_ids = find_matching_ids(
            pattern, model.articulation_label, model_articulation_world, model.world_count
        )

        # determine articulation counts per world
        world_count = model.world_count
        articulation_count = 0
        counts_per_world = [0] * world_count
        for world_id in range(world_count):
            count = len(articulation_ids[world_id])
            counts_per_world[world_id] += count
            articulation_count += count

        # can't mix global and per-world articulations in the same view
        if articulation_count > 0 and global_articulation_ids:
            raise ValueError(
                f"Articulation pattern '{pattern}' matches global and per-world articulations, which is currently not supported"
            )

        # handle scenes with only global articulations
        if articulation_count == 0 and global_articulation_ids:
            world_count = 1
            articulation_count = len(global_articulation_ids)
            counts_per_world = [articulation_count]
            articulation_ids = [global_articulation_ids]

        if articulation_count == 0:
            raise KeyError(f"No articulations matching pattern '{pattern}'")

        if not all_equal(counts_per_world):
            raise ValueError("Varying articulation counts per world are not supported")

        count_per_world = counts_per_world[0]

        # use the first articulation as a "template"
        arti_0 = articulation_ids[0][0]

        arti_joint_ids = []
        arti_joint_names = []
        arti_joint_types = []
        arti_link_ids = []
        arti_link_names = []
        arti_link_labels = []
        arti_joint_labels = []
        arti_shape_ids = []
        arti_shape_names = []
        arti_shape_labels = []

        # gather joint info
        arti_joint_begin = int(model_articulation_start[arti_0])
        if include_loop_closing_joints:
            arti_joint_end = int(model_articulation_start[arti_0 + 1])
        else:
            arti_joint_end = int(model_articulation_end[arti_0])
        arti_joint_count = arti_joint_end - arti_joint_begin
        arti_joint_dof_begin = int(model_joint_qd_start[arti_joint_begin])
        arti_joint_coord_begin = int(model_joint_q_start[arti_joint_begin])
        for joint_id in range(arti_joint_begin, arti_joint_end):
            # joint_id = arti_joint_begin + idx
            arti_joint_ids.append(joint_id)
            arti_joint_labels.append(model.joint_label[joint_id])
            arti_joint_names.append(get_name_from_label(model.joint_label[joint_id]))
            arti_joint_types.append(model_joint_type[joint_id])
            if model_joint_articulation[joint_id] == arti_0:
                arti_link_ids.append(int(model_joint_child[joint_id]))

        # use link order as they appear in the model
        arti_link_ids = sorted(set(arti_link_ids))
        arti_link_count = len(arti_link_ids)
        for link_id in arti_link_ids:
            arti_link_labels.append(model.body_label[link_id])
            arti_link_names.append(get_name_from_label(model.body_label[link_id]))
            arti_shape_ids.extend(model.body_shapes[link_id])

        # use shape order as they appear in the model
        arti_shape_ids = sorted(arti_shape_ids)
        for shape_id in arti_shape_ids:
            arti_shape_labels.append(model.shape_label[shape_id])
            arti_shape_names.append(get_name_from_label(model.shape_label[shape_id]))

        joint_dof_offset = arti_joint_dof_begin
        joint_coord_offset = arti_joint_coord_begin

        # create joint inclusion set
        if include_joints is None and include_joint_types is None:
            joint_include_indices = set(range(arti_joint_count))
        else:
            joint_include_indices = set()
            if include_joints is not None:
                matching_joint_indices = match_labels(arti_joint_names, include_joints)
                for index in matching_joint_indices:
                    if index < 0 or index >= arti_joint_count:
                        raise ValueError(
                            f"include_joints indices must be in range [0, {arti_joint_count}), got {index}"
                        )
                joint_include_indices.update(matching_joint_indices)
            if include_joint_types is not None:
                for idx in range(arti_joint_count):
                    if arti_joint_types[idx] in include_joint_types:
                        joint_include_indices.add(idx)

        # create joint exclusion set
        joint_exclude_indices = set()
        if exclude_joints is not None:
            joint_exclude_indices.update(
                idx for idx in match_labels(arti_joint_names, exclude_joints) if 0 <= idx < arti_joint_count
            )
        if exclude_joint_types is not None:
            for idx in range(arti_joint_count):
                if arti_joint_types[idx] in exclude_joint_types:
                    joint_exclude_indices.add(idx)

        # create link inclusion set
        if include_links is None:
            link_include_indices = set(range(arti_link_count))
        else:
            matching_link_indices = match_labels(arti_link_names, include_links)
            for index in matching_link_indices:
                if index < 0 or index >= arti_link_count:
                    raise ValueError(f"include_links indices must be in range [0, {arti_link_count}), got {index}")
            link_include_indices = set(matching_link_indices)

        # create link exclusion set
        link_exclude_indices = set()
        if exclude_links is not None:
            link_exclude_indices.update(
                idx for idx in match_labels(arti_link_names, exclude_links) if 0 <= idx < arti_link_count
            )

        # compute selected indices
        selected_joint_indices = sorted(joint_include_indices - joint_exclude_indices)
        selected_link_indices = sorted(link_include_indices - link_exclude_indices)
        # without filters, every joint and link of each articulation is selected, not only template positions
        all_joints = include_joints is None and include_joint_types is None and not joint_exclude_indices
        all_links = include_links is None and not link_exclude_indices

        self.joint_names: list[str] | None = []
        self.joint_labels: list[str] | None = []
        self.joint_dof_names: list[str] | None = []
        self.joint_dof_counts: list[int] | None = []
        self.joint_coord_names: list[str] | None = []
        self.joint_coord_counts: list[int] | None = []
        self.link_names: list[str] | None = []
        self.link_labels: list[str] | None = []
        self.link_shapes: list[list[int]] | None = []
        self.shape_names: list[str] | None = []
        self.shape_labels: list[str] | None = []

        # populate info for selected joints and dofs
        selected_joint_dof_indices = []
        selected_joint_coord_indices = []
        for joint_idx in selected_joint_indices:
            joint_id = arti_joint_ids[joint_idx]
            joint_name = arti_joint_names[joint_idx]
            self.joint_names.append(joint_name)
            self.joint_labels.append(arti_joint_labels[joint_idx])
            # joint dofs
            dof_begin = int(model_joint_qd_start[joint_id])
            dof_end = int(model_joint_qd_start[joint_id + 1])
            dof_count = dof_end - dof_begin
            self.joint_dof_counts.append(dof_count)
            if dof_count == 1:
                self.joint_dof_names.append(joint_name)
                selected_joint_dof_indices.append(dof_begin - joint_dof_offset)
            elif dof_count > 1:
                for dof in range(dof_count):
                    self.joint_dof_names.append(f"{joint_name}:{dof}")
                    selected_joint_dof_indices.append(dof_begin + dof - joint_dof_offset)
            # joint coords
            coord_begin = int(model_joint_q_start[joint_id])
            coord_end = int(model_joint_q_start[joint_id + 1])
            coord_count = coord_end - coord_begin
            self.joint_coord_counts.append(coord_count)
            if coord_count == 1:
                self.joint_coord_names.append(joint_name)
                selected_joint_coord_indices.append(coord_begin - joint_coord_offset)
            elif coord_count > 1:
                for coord in range(coord_count):
                    self.joint_coord_names.append(f"{joint_name}:{coord}")
                    selected_joint_coord_indices.append(coord_begin + coord - joint_coord_offset)

        # populate info for selected links and shapes
        selected_shape_indices = []
        shape_link_idx = {}  # map arti_shape_idx to local link index in the view
        for link_idx, arti_link_idx in enumerate(selected_link_indices):
            body_id = arti_link_ids[arti_link_idx]
            self.link_names.append(arti_link_names[arti_link_idx])
            self.link_labels.append(arti_link_labels[arti_link_idx])
            shape_ids = model.body_shapes[body_id]
            for shape_id in shape_ids:
                arti_shape_idx = arti_shape_ids.index(shape_id)
                selected_shape_indices.append(arti_shape_idx)
                shape_link_idx[arti_shape_idx] = link_idx
            self.link_shapes.append([])

        selected_shape_indices = sorted(selected_shape_indices)
        for shape_idx, arti_shape_idx in enumerate(selected_shape_indices):
            self.shape_names.append(arti_shape_names[arti_shape_idx])
            self.shape_labels.append(arti_shape_labels[arti_shape_idx])
            link_idx = shape_link_idx[arti_shape_idx]
            self.link_shapes[link_idx].append(shape_idx)

        # selection counts
        self.count = articulation_count
        self.world_count = world_count
        self.count_per_world = count_per_world
        self.joint_count: int | None = len(selected_joint_indices)
        self.joint_dof_count: int | None = len(selected_joint_dof_indices)
        self.joint_coord_count: int | None = len(selected_joint_coord_indices)
        self.link_count: int | None = len(selected_link_indices)
        self.shape_count: int | None = len(selected_shape_indices)

        # TODO: document the layout conventions and requirements
        #
        # |ooXXXoXXXoXXXooo|ooXXXoXXXoXXXooo|ooXXXoXXXoXXXooo|ooXXXoXXXoXXXooo|
        # |  ^   ^   ^     |  ^   ^   ^     |  ^   ^   ^     |  ^   ^   ^     |
        #
        articulation_id_grid = np.asarray(articulation_ids)
        validations = _validate_layouts(
            model,
            articulation_id_grid,
            model_articulation_start,
            model_articulation_end,
            model_joint_articulation,
            model_joint_child,
            model_joint_q_start,
            model_joint_qd_start,
            model_shape_body,
            None if all_joints else selected_joint_indices,
            None if all_links else selected_link_indices,
            include_loop_closing_joints,
        )
        frequencies = (
            AttributeFrequency.JOINT,
            AttributeFrequency.JOINT_DOF,
            AttributeFrequency.JOINT_COORD,
            AttributeFrequency.BODY,
            AttributeFrequency.SHAPE,
        )
        self.frequency_layouts = {f: validations[f][0] for f in frequencies if validations[f][0] is not None}
        self._unavailable_reasons = {f: validations[f][1] for f in frequencies if validations[f][1] is not None}

        root_ids = model_articulation_start[articulation_id_grid.ravel()]
        root_types = model_joint_type[root_ids]
        root_is_fixed = model_joint_qd_start[root_ids + 1] == model_joint_qd_start[root_ids]
        root_is_floating = np.isin(root_types, (JointType.FREE, JointType.DISTANCE))
        # fixed base means that all linear and angular degrees of freedom are locked at the root
        self.is_fixed_base: bool | None = _uniform_value(root_is_fixed)
        # floating base means that all linear and angular degrees of freedom are unlocked at the root
        # (though there might be constraints like distance)
        self.is_floating_base: bool | None = _uniform_value(root_is_floating)
        self.root_joint_type: int | None = _uniform_value(root_types)
        # root transforms and velocities, addressed through the root joint's own layout
        root_keys = ("root_coord", "root_dof") if self.is_floating_base else ("root_joint",)
        self._root_layouts = [validations[key][0] for key in root_keys]
        if self.is_floating_base is None:
            self._root_unavailable_reason = "selected articulations use different root behavior"
        else:
            self._root_unavailable_reason = "; ".join(validations[k][1] for k in root_keys if validations[k][1]) or None

        if not allow_partial_layouts:
            for frequency, reason in self._unavailable_reasons.items():
                raise ValueError(f"Articulation {frequency.name} layout is unavailable: {reason}")
            if self._root_unavailable_reason or self.root_joint_type is None or self.is_fixed_base is None:
                reason = self._root_unavailable_reason or "root metadata differs"
                raise ValueError(f"Articulation root layout is unavailable: {reason}")

        # counts and names are unknown where selected counts differ, relationships where either layout is unavailable
        for frequency, names in (
            (AttributeFrequency.JOINT, ("joint_count", "joint_names", "joint_labels")),
            (AttributeFrequency.JOINT_DOF, ("joint_dof_count", "joint_dof_names")),
            (AttributeFrequency.JOINT_COORD, ("joint_coord_count", "joint_coord_names")),
            (AttributeFrequency.BODY, ("link_count", "link_names", "link_labels")),
            (AttributeFrequency.SHAPE, ("shape_count", "shape_names", "shape_labels")),
        ):
            if not validations[frequency][2]:
                for name in names:
                    setattr(self, name, None)
        for name, related in (
            ("joint_dof_counts", (AttributeFrequency.JOINT, AttributeFrequency.JOINT_DOF)),
            ("joint_coord_counts", (AttributeFrequency.JOINT, AttributeFrequency.JOINT_COORD)),
            ("link_shapes", (AttributeFrequency.BODY, AttributeFrequency.SHAPE)),
        ):
            if not all(frequency in self.frequency_layouts for frequency in related):
                setattr(self, name, None)

        # Build layouts for every custom frequency that declares per-row
        # articulation ownership on the model.
        self.custom_frequency_counts: dict[str, int | None] = {}
        self.custom_frequency_labels: dict[str, list[str] | None] = {}
        for frequency, owner_array in model.custom_frequency_articulation.items():
            owners = owner_array.numpy()
            rows_by_articulation: dict[int, list[int]] = {}
            for row, owner in enumerate(owners):
                if owner >= 0:
                    rows_by_articulation.setdefault(int(owner), []).append(row)

            articulation_rows = [
                [rows_by_articulation.get(articulation_id, []) for articulation_id in world_articulations]
                for world_articulations in articulation_ids
            ]
            row_counts = [[len(rows) for rows in world_rows] for world_rows in articulation_rows]
            flat_row_counts = [count for world_counts in row_counts for count in world_counts]
            if not all_equal(flat_row_counts):
                reason = f"Articulations have different row counts for custom frequency '{frequency}': {row_counts}"
                if not allow_partial_layouts:
                    raise ValueError(reason)
                self._unavailable_reasons[frequency] = reason
                self.custom_frequency_counts[frequency] = None
                self.custom_frequency_labels[frequency] = None
                continue

            value_count = flat_row_counts[0]
            self.custom_frequency_counts[frequency] = value_count
            self.custom_frequency_labels[frequency] = []
            if value_count == 0:
                continue

            template_rows = articulation_rows[0][0]
            offset = template_rows[0]
            selected_indices = [row - offset for row in template_rows]
            # The addressable extent includes gaps between selected rows.
            value_extent = template_rows[-1] - offset + 1
            starts = [[rows[0] for rows in world_rows] for world_rows in articulation_rows]
            reason = None

            if count_per_world > 1:
                inner_strides = [
                    starts[world][articulation] - starts[world][articulation - 1]
                    for world in range(world_count)
                    for articulation in range(1, count_per_world)
                ]
                if not all_equal(inner_strides):
                    reason = f"Non-uniform strides within worlds for custom frequency '{frequency}' are not supported"
                inner_stride = inner_strides[0]
            else:
                inner_stride = value_extent

            if world_count > 1:
                outer_strides = [starts[world][0] - starts[world - 1][0] for world in range(1, world_count)]
                if not all_equal(outer_strides):
                    reason = f"Non-uniform strides between worlds for custom frequency '{frequency}' are not supported"
                outer_stride = outer_strides[0]
            else:
                outer_stride = inner_stride * count_per_world

            for world in range(world_count):
                for articulation in range(count_per_world):
                    relative_rows = [
                        row - starts[world][articulation] for row in articulation_rows[world][articulation]
                    ]
                    if relative_rows != selected_indices:
                        reason = f"Custom frequency '{frequency}' has inconsistent row ordering between articulations"

            if reason is None:
                self.frequency_layouts[frequency] = FrequencyLayout(
                    offset,
                    outer_stride,
                    inner_stride,
                    value_extent,
                    selected_indices,
                    self.device,
                )
            elif allow_partial_layouts:
                self._unavailable_reasons[frequency] = reason
            else:
                raise ValueError(reason)

            label_key = model.custom_frequency_label_attributes.get(frequency)
            if label_key is not None:
                labels = model
                for component in label_key.split(":"):
                    labels = getattr(labels, component)
                self.custom_frequency_labels[frequency] = [get_name_from_label(labels[row]) for row in template_rows]

        # Compatibility aliases backed by the generic custom-frequency metadata.
        self.tendon_count: int | None = self.custom_frequency_counts.get("mujoco:tendon", 0)
        self.tendon_names: list[str] | None = self.custom_frequency_labels.get("mujoco:tendon", [])

        def is_contiguous(frequency):
            layout = self.frequency_layouts.get(frequency)
            return layout.is_packed(self.world_count, self.count_per_world) if layout is not None else None

        self.joints_contiguous: bool | None = is_contiguous(AttributeFrequency.JOINT)
        self.joint_dofs_contiguous: bool | None = is_contiguous(AttributeFrequency.JOINT_DOF)
        self.joint_coords_contiguous: bool | None = is_contiguous(AttributeFrequency.JOINT_COORD)
        self.links_contiguous: bool | None = is_contiguous(AttributeFrequency.BODY)
        self.shapes_contiguous: bool | None = is_contiguous(AttributeFrequency.SHAPE)

        # articulation ids grouped by world
        self.articulation_ids = wp.array(articulation_id_grid, dtype=int, device=self.device)

        # default mask includes all articulations in all worlds
        self.full_mask = wp.full(world_count, True, dtype=bool, device=self.device)

        # create articulation mask
        self.articulation_mask = wp.zeros(model.articulation_count, dtype=bool, device=self.device)
        wp.launch(
            set_model_articulation_mask_per_world_kernel,
            dim=self.articulation_ids.shape,
            inputs=[self.full_mask, self.articulation_ids, self.articulation_mask],
            device=self.device,
        )

        if verbose:
            print(f"Articulation '{pattern}': {self.count}")
            print(f"  Link count:     {self.link_count} ({'' if self.links_contiguous else 'non-'}contiguous)")
            print(f"  Shape count:    {self.shape_count} ({'' if self.shapes_contiguous else 'non-'}contiguous)")
            print(f"  Joint count:    {self.joint_count} ({'' if self.joints_contiguous else 'non-'}contiguous)")
            print(
                f"  DOF count:      {self.joint_dof_count} ({'' if self.joint_dofs_contiguous else 'non-'}contiguous)"
            )
            print(f"  Fixed base?     {self.is_fixed_base}")
            print(f"  Floating base?  {self.is_floating_base}")
            print("Link names:")
            print(f"  {self.link_names}")
            print("Joint names:")
            print(f"  {self.joint_names}")
            print("Joint DOF names:")
            print(f"  {self.joint_dof_names}")
            print("Shapes:")
            for link_idx in range(len(self.link_shapes or [])):
                shape_names = [self.shape_names[shape_idx] for shape_idx in self.link_shapes[link_idx]]
                print(f"  Link '{self.link_names[link_idx]}': {shape_names}")

    @property
    def body_names(self):
        """Alias for `link_names`."""
        return self.link_names

    @property
    def body_shapes(self):
        """Alias for `link_shapes`."""
        return self.link_shapes

    @property
    def body_labels(self):
        """Alias for `link_labels`."""
        return self.link_labels

    # ========================================================================================
    # Generic attribute API

    def _get_attribute_array(
        self, name: str, source: Model | State | Control, _slice: Slice | int | None = None, layout=None
    ):
        key = (name, source, _slice, layout)
        if key not in self._attribute_array_cache:
            self._attribute_array_cache[key] = self._create_attribute_array(name, source, _slice, layout)
        return self._attribute_array_cache[key]

    def _create_attribute_array(
        self, name: str, source: Model | State | Control, _slice: Slice | int | None, layout=None
    ):
        # get the attribute (handle namespaced attributes like "mujoco.tendon_stiffness")
        # Note: the user-facing API uses dots (e.g., "mujoco.tendon_stiffness")
        # but internally attributes are stored with colons (e.g., "mujoco:tendon_stiffness")
        if "." in name:
            parts = name.split(".")
            attrib = source
            for part in parts:
                attrib = getattr(attrib, part)
            # Convert dot notation to colon notation for frequency lookup
            frequency_name = ":".join(parts)
        else:
            attrib = getattr(source, name)
            frequency_name = name
        assert isinstance(attrib, wp.array)

        # get frequency info
        frequency = self.model.get_attribute_frequency(frequency_name)

        if layout is None and frequency in self._unavailable_reasons:
            raise AttributeError(f"Attribute '{name}' is unavailable: {self._unavailable_reasons[frequency]}")
        if layout is None and isinstance(frequency, str):
            layout = self.frequency_layouts.get(frequency)
            if layout is None:
                if frequency in self.model.custom_frequency_articulation:
                    raise AttributeError(
                        f"Attribute '{name}' has frequency '{frequency}' but no rows were found "
                        "in the selected articulations"
                    )
                raise AttributeError(
                    f"Attribute '{name}' has custom frequency '{frequency}', which does not declare "
                    "articulation ownership"
                )
        elif layout is None:
            layout = self.frequency_layouts.get(frequency)
            if layout is None:
                raise AttributeError(
                    f"Unable to determine the layout of frequency '{frequency.name}' for attribute '{name}'"
                )

        value_stride = attrib.strides[0]
        is_indexed = layout.indices is not None

        # handle custom slice
        if isinstance(_slice, Slice):
            _slice = _slice.get()
        elif not isinstance(_slice, (NoneType, int, slice)):
            raise ValueError(f"Invalid slice type: expected slice or int, got {type(_slice)}")

        if _slice is None:
            value_slice = layout.indices if is_indexed else layout.slice
            value_count = layout.value_count
        else:
            value_slice = _slice
            value_count = 1 if isinstance(_slice, int) else _slice.stop - _slice.start

        # trailing dimensions for multidimensional attributes
        trailing_shape = attrib.shape[1:]
        trailing_strides = attrib.strides[1:]
        trailing_slices = [slice(s) for s in trailing_shape]

        shape = (self.world_count, self.count_per_world, value_count, *trailing_shape)
        strides = (
            layout.stride_between_worlds * value_stride,
            layout.stride_within_worlds * value_stride,
            value_stride,
            *trailing_strides,
        )
        slices = (slice(self.world_count), slice(self.count_per_world), value_slice, *trailing_slices)

        # early out for empty selections and empty source arrays (e.g. articulations with only fixed joints)
        if value_count == 0 or attrib.ptr is None:
            result = wp.empty(shape, dtype=attrib.dtype, device=attrib.device)
            result.ptr = None
            return result

        # construct reshaped attribute array, preserving grad connectivity
        source_array = attrib
        source_grad = attrib.grad if attrib.requires_grad else None
        grad_view = None
        if source_grad is not None:
            grad_stride = source_grad.strides[0]
            grad_view = wp.array(
                ptr=int(source_grad.ptr) + layout.offset * grad_stride,
                dtype=source_grad.dtype,
                shape=shape,
                strides=(
                    layout.stride_between_worlds * grad_stride,
                    layout.stride_within_worlds * grad_stride,
                    grad_stride,
                    *source_grad.strides[1:],
                ),
                device=source_grad.device,
                copy=False,
            )
            grad_view._ref = source_grad

        attrib = wp.array(
            ptr=int(attrib.ptr) + layout.offset * value_stride,
            dtype=attrib.dtype,
            shape=shape,
            strides=strides,
            device=attrib.device,
            copy=False,
            grad=grad_view,
        )
        attrib._ref = source_array

        # apply selection (slices or indices)
        pre_indexed = attrib
        attrib = attrib[slices]

        if is_indexed:
            attrib._staging_array = wp.empty_like(attrib)
            if grad_view is not None:
                attrib._staging_array.requires_grad = True
                attrib._gather_src = pre_indexed
                attrib._gather_indices = layout.indices
        else:
            # fixup for empty slices - FIXME: this should be handled by Warp, above
            if attrib.size == 0:
                attrib.ptr = None

        return attrib

    def _get_attribute_values(
        self, name: str, source: Model | State | Control, _slice: slice | None = None, layout=None
    ):
        attrib = self._get_attribute_array(name, source, _slice=_slice, layout=layout)
        if hasattr(attrib, "_staging_array"):
            if hasattr(attrib, "_gather_src"):
                kernel = _gather_indexed_4d_kernel if attrib.ndim == 4 else _gather_indexed_3d_kernel
                wp.launch(
                    kernel,
                    dim=attrib._staging_array.shape,
                    inputs=[attrib._gather_src, attrib._gather_indices],
                    outputs=[attrib._staging_array],
                )
                src_grad = attrib._gather_src.grad
                dst_grad = attrib._staging_array.grad
                if src_grad is not None and dst_grad is not None:
                    grad_slices = tuple(attrib._gather_indices if d == 2 else slice(None) for d in range(src_grad.ndim))
                    wp.copy(dst_grad, src_grad[grad_slices])
            else:
                wp.copy(attrib._staging_array, attrib)
            return attrib._staging_array
        return attrib

    def _set_attribute_values(
        self, name: str, target: Model | State | Control, values, mask=None, _slice: slice | None = None, layout=None
    ):
        attrib = self._get_attribute_array(name, target, _slice=_slice, layout=layout)

        if not is_array(values) or values.dtype != attrib.dtype:
            values = wp.array(values, dtype=attrib.dtype, shape=attrib.shape, device=self.device, copy=False)
        assert values.shape == attrib.shape
        assert values.dtype == attrib.dtype

        # early out for in-place modifications
        if isinstance(attrib, wp.array) and isinstance(values, wp.array):
            if values.ptr == attrib.ptr:
                return
        if isinstance(attrib, wp.indexedarray) and isinstance(values, wp.indexedarray):
            if values.data.ptr == attrib.data.ptr:
                return

        # get mask
        if mask is None:
            mask = self.full_mask
        else:
            mask = self._resolve_mask(mask)

        # launch appropriate kernel based on attribute dimensionality
        # TODO: cache concrete overload per attribute?
        if mask.ndim == 1:
            if attrib.ndim == 3:
                wp.launch(
                    set_articulation_attribute_3d_per_world_kernel,
                    dim=attrib.shape,
                    inputs=[mask, values, attrib],
                    device=self.device,
                )
            elif attrib.ndim == 4:
                wp.launch(
                    set_articulation_attribute_4d_per_world_kernel,
                    dim=attrib.shape,
                    inputs=[mask, values, attrib],
                    device=self.device,
                )
            else:
                raise NotImplementedError(f"Unsupported attribute with ndim={attrib.ndim}")
        else:  # mask.ndim == 2
            if attrib.ndim == 3:
                wp.launch(
                    set_articulation_attribute_3d_kernel,
                    dim=attrib.shape,
                    inputs=[mask, values, attrib],
                    device=self.device,
                )
            elif attrib.ndim == 4:
                wp.launch(
                    set_articulation_attribute_4d_kernel,
                    dim=attrib.shape,
                    inputs=[mask, values, attrib],
                    device=self.device,
                )
            else:
                raise NotImplementedError(f"Unsupported attribute with ndim={attrib.ndim}")

    def get_attribute(self, name: str, source: Model | State | Control):
        """
        Get an attribute from the source (Model, State, or Control).

        Args:
            name: The name of the attribute to get.
            source: The source from which to get the attribute.

        Returns:
            array: The attribute values (dtype matches the attribute).
        """
        return self._get_attribute_values(name, source)

    def set_attribute(
        self,
        name: str,
        target: Model | State | Control,
        values: wp.array[Any],
        mask: wp.array[bool] | wp.array2d[bool] | None = None,
    ) -> None:
        """
        Set an attribute in the target (Model, State, or Control).

        Args:
            name: The name of the attribute to set.
            target: The target where to set the attribute.
            values: The values to set for the attribute.
            mask: Mask of articulations in this ArticulationView (all by default).

        .. note::
            When setting attributes on the Model, it may be necessary to inform the solver about
            such changes by calling :meth:`newton.solvers.SolverBase.notify_model_changed` after finished
            setting Model attributes.
        """
        self._set_attribute_values(name, target, values, mask=mask)

    # ========================================================================================
    # Convenience wrappers to align with legacy tensor API

    def _root_layout(self, index: int) -> FrequencyLayout:
        if self._root_unavailable_reason is not None:
            raise AttributeError(f"Root access is unavailable: {self._root_unavailable_reason}")
        return self._root_layouts[index]

    def get_root_transforms(self, source: Model | State):
        """
        Get the root transforms of the articulations.

        Args:
            source: Where to get the root transforms (Model or State).

        Returns:
            array: The root transforms (dtype=wp.transform).
        """
        layout = self._root_layout(0)
        if self.is_floating_base:
            attrib = self._get_attribute_values("joint_q", source, _slice=Slice(0, 7), layout=layout)
        else:
            attrib = self._get_attribute_values("joint_X_p", self.model, _slice=0, layout=layout)

        if attrib.dtype is wp.transform:
            return attrib
        else:
            return wp.array(attrib, dtype=wp.transform, device=self.device, copy=False)

    def set_root_transforms(
        self,
        target: Model | State,
        values: wp.array[wp.transform],
        mask: wp.array[bool] | wp.array2d[bool] | None = None,
    ) -> None:
        """
        Set the root transforms of the articulations.
        Call :meth:`eval_fk` to apply changes to all articulation links.

        Args:
            target: Where to set the root transforms (Model or State).
            values: The root transforms to set (dtype=wp.transform).
            mask: Mask of articulations in this ArticulationView (all by default).
        """
        layout = self._root_layout(0)
        if self.is_floating_base:
            self._set_attribute_values("joint_q", target, values, mask=mask, _slice=Slice(0, 7), layout=layout)
        else:
            if is_array(values):
                # add the value axis; the strided array returned by get_root_transforms() must be copied first
                values = (values if values.is_contiguous else wp.clone(values)).reshape((*values.shape, 1))
            self._set_attribute_values("joint_X_p", self.model, values, mask=mask, _slice=Slice(0, 1), layout=layout)

    def get_root_velocities(self, source: Model | State):
        """
        Get the root velocities of the articulations.

        Args:
            source: Where to get the root velocities (Model or State).

        Returns:
            array: The root velocities (dtype=wp.spatial_vector).
        """
        layout = self._root_layout(-1)
        if self.is_floating_base:
            attrib = self._get_attribute_values("joint_qd", source, _slice=Slice(0, 6), layout=layout)
        else:
            # FIXME? Non-floating articulations have no root velocities.
            return None

        if attrib.dtype is wp.spatial_vector:
            return attrib
        else:
            return wp.array(attrib, dtype=wp.spatial_vector, device=self.device, copy=False)

    def set_root_velocities(
        self,
        target: Model | State,
        values: wp.array[wp.spatial_vector],
        mask: wp.array[bool] | wp.array2d[bool] | None = None,
    ) -> None:
        """
        Set the root velocities of the articulations.

        Args:
            target: Where to set the root velocities (Model or State).
            values: The root velocities to set (dtype=wp.spatial_vector).
            mask: Mask of articulations in this ArticulationView (all by default).
        """
        layout = self._root_layout(-1)
        if self.is_floating_base:
            self._set_attribute_values("joint_qd", target, values, mask=mask, _slice=Slice(0, 6), layout=layout)
        else:
            return  # no-op

    def get_link_transforms(self, source: Model | State):
        """
        Get the world-space transforms of all links in the selected articulations.

        Args:
            source: The source from which to retrieve the link transforms.

        Returns:
            array: The link transforms (dtype=wp.transform).
        """
        return self._get_attribute_values("body_q", source)

    def get_link_velocities(self, source: Model | State):
        """
        Get the world-space spatial velocities of all links in the selected articulations.

        The returned ``body_qd`` values follow Newton's public convention:
        ``(v_com_world, omega_world)``.

        Args:
            source: The source from which to retrieve the link velocities.

        Returns:
            array: The link velocities (dtype=wp.spatial_vector).
        """
        return self._get_attribute_values("body_qd", source)

    def get_dof_positions(self, source: Model | State):
        """
        Get the joint coordinate positions (DoF positions) for the selected articulations.

        Args:
            source: The source from which to retrieve the DoF positions.

        Returns:
            array: The joint coordinate positions (dtype=float).
        """
        return self._get_attribute_values("joint_q", source)

    def set_dof_positions(
        self,
        target: Model | State,
        values: wp.array[float],
        mask: wp.array[bool] | wp.array2d[bool] | None = None,
    ) -> None:
        """
        Set the joint coordinate positions (DoF positions) for the selected articulations.

        Args:
            target: The target where to set the DoF positions.
            values: The values to set (dtype=float).
            mask: Mask of articulations in this ArticulationView (all by default).
        """
        self._set_attribute_values("joint_q", target, values, mask=mask)

    def get_dof_velocities(self, source: Model | State):
        """
        Get the joint coordinate velocities (DoF velocities) for the selected articulations.

        Args:
            source: The source from which to retrieve the DoF velocities.

        Returns:
            array: The joint coordinate velocities (dtype=float).
        """
        return self._get_attribute_values("joint_qd", source)

    def set_dof_velocities(
        self,
        target: Model | State,
        values: wp.array[float],
        mask: wp.array[bool] | wp.array2d[bool] | None = None,
    ) -> None:
        """
        Set the joint coordinate velocities (DoF velocities) for the selected articulations.

        Args:
            target: The target where to set the DoF velocities.
            values: The values to set (dtype=float).
            mask: Mask of articulations in this ArticulationView (all by default).
        """
        self._set_attribute_values("joint_qd", target, values, mask=mask)

    def get_dof_forces(self, source: Control):
        """
        Get the joint forces (DoF forces) for the selected articulations.

        Args:
            source: The source from which to retrieve the DoF forces.

        Returns:
            array: The joint forces (dtype=float).
        """
        return self._get_attribute_values("joint_f", source)

    def set_dof_forces(
        self,
        target: Control,
        values: wp.array[float],
        mask: wp.array[bool] | wp.array2d[bool] | None = None,
    ) -> None:
        """
        Set the joint forces (DoF forces) for the selected articulations.

        Args:
            target: The target where to set the DoF forces.
            values: The values to set (dtype=float).
            mask: Mask of articulations in this ArticulationView (all by default).
        """
        self._set_attribute_values("joint_f", target, values, mask=mask)

    # ========================================================================================
    # Utilities

    def _resolve_world_mask(self, mask):
        if mask is None:
            return self.full_mask
        if isinstance(mask, wp.array):
            if mask.dtype is not wp.bool:
                raise ValueError(f"Expected Boolean mask, got dtype {mask.dtype}")
            if mask.shape != (self.world_count,):
                raise ValueError(f"Expected mask shape ({self.world_count},), got {mask.shape}")
            if mask.device != self.device:
                raise ValueError(f"Expected mask on device {self.device}, got {mask.device}")
            return mask

        try:
            return wp.array(mask, dtype=bool, shape=(self.world_count,), device=self.device, copy=False)
        except Exception as error:
            raise ValueError(f"Expected Boolean mask with shape ({self.world_count},)") from error

    def _resolve_mask(self, mask):
        # accept 1D and 2D Boolean masks
        if isinstance(mask, wp.array):
            expected_shapes = {
                (self.world_count,),
                (self.world_count, self.count_per_world),
            }
            if mask.dtype is not wp.bool:
                raise ValueError(f"Expected Boolean mask, got dtype {mask.dtype}")
            if mask.shape not in expected_shapes:
                raise ValueError(
                    f"Expected Boolean mask with shape "
                    f"({self.world_count}, {self.count_per_world}) or ({self.world_count},), got {mask.shape}"
                )
            if mask.device != self.device:
                raise ValueError(f"Expected mask on device {self.device}, got {mask.device}")
            return mask
        else:
            # try interpreting as a 1D world mask
            try:
                return wp.array(mask, dtype=bool, shape=self.world_count, device=self.device, copy=False)
            except Exception:
                pass
            # try interpreting as a 2D (world, arti) mask
            try:
                return wp.array(
                    mask, dtype=bool, shape=(self.world_count, self.count_per_world), device=self.device, copy=False
                )
            except Exception:
                pass

        # no match
        raise ValueError(
            f"Expected Boolean mask with shape ({self.world_count}, {self.count_per_world}) or ({self.world_count},)"
        )

    def get_model_articulation_mask(self, mask: wp.array[bool] | wp.array2d[bool] | None = None) -> wp.array[bool]:
        """
        Get Model articulation mask from a mask in this ArticulationView.

        Args:
            mask: Mask of articulations in this ArticulationView (all by default).
        """
        if mask is None:
            return self.articulation_mask
        else:
            mask = self._resolve_mask(mask)
            articulation_mask = wp.zeros(self.model.articulation_count, dtype=bool, device=self.device)
            if mask.ndim == 1:
                wp.launch(
                    set_model_articulation_mask_per_world_kernel,
                    dim=self.articulation_ids.shape,
                    inputs=[mask, self.articulation_ids, articulation_mask],
                    device=self.device,
                )
            else:
                wp.launch(
                    set_model_articulation_mask_kernel,
                    dim=self.articulation_ids.shape,
                    inputs=[mask, self.articulation_ids, articulation_mask],
                    device=self.device,
                )
            return articulation_mask

    def eval_fk(
        self,
        target: Model | State,
        mask: wp.array[bool] | wp.array2d[bool] | None = None,
    ) -> None:
        """
        Evaluates forward kinematics given the joint coordinates and updates the body information.

        The written ``target.body_qd`` values follow Newton's public body-twist
        convention ``(v_com_world, omega_world)``.

        Args:
            target: The target where to evaluate forward kinematics (Model or State).
            mask: Mask of articulations in this ArticulationView (all by default).
        """
        # translate view mask to Model articulation mask
        articulation_mask = self.get_model_articulation_mask(mask=mask)
        eval_fk(self.model, target.joint_q, target.joint_qd, target, mask=articulation_mask)

    def eval_jacobian(self, state: State, J=None, joint_S_s=None, mask=None):
        """Evaluate spatial Jacobian for articulations in this view.

        Computes the spatial Jacobian J that maps joint velocities to spatial
        velocities of each link in world frame, matching ``state.body_qd`` under
        Newton's public COM/world body-twist convention.

        Args:
            state: The state containing body transforms (body_q).
            J: Optional output array for the Jacobian, shape (articulation_count, max_links*6, max_dofs).
               If None, allocates internally.
            joint_S_s: Optional pre-allocated temp array for motion subspaces.
            mask: Optional mask of articulations in this ArticulationView (all by default).

        Returns:
            The Jacobian array J, or None if the model has no articulations.
        """
        articulation_mask = self.get_model_articulation_mask(mask=mask)
        return eval_jacobian(self.model, state, J, joint_S_s=joint_S_s, mask=articulation_mask)

    def eval_mass_matrix(self, state: State, H=None, J=None, body_I_s=None, joint_S_s=None, mask=None):
        """Evaluate generalized mass matrix for articulations in this view.

        Computes the generalized mass matrix H = J^T * M * J, where J is the spatial
        Jacobian and M is the block-diagonal spatial mass matrix. The resulting
        matrix is consistent with kinetic energy computed from COM-referenced
        body twists.

        Args:
            state: The state containing body transforms (body_q).
            H: Optional output array for mass matrix, shape (articulation_count, max_dofs, max_dofs).
               If None, allocates internally.
            J: Optional pre-computed Jacobian. If None, computes internally.
            body_I_s: Optional pre-allocated temp array for spatial inertias.
            joint_S_s: Optional pre-allocated temp array for motion subspaces.
            mask: Optional mask of articulations in this ArticulationView (all by default).

        Returns:
            The mass matrix array H, or None if the model has no articulations.
        """
        articulation_mask = self.get_model_articulation_mask(mask=mask)
        return eval_mass_matrix(
            self.model, state, H, J=J, body_I_s=body_I_s, joint_S_s=joint_S_s, mask=articulation_mask
        )

    def eval_inverse_dynamics_passive(
        self,
        state: State,
        *,
        mass_matrix: wp.array3d[wp.float32] | None = None,
        gravity_force: wp.array[wp.float32] | None = None,
        coriolis_force: wp.array[wp.float32] | None = None,
        mask: wp.array[bool] | wp.array2d[bool] | None = None,
    ) -> None:
        """Compute passive inverse-dynamics quantities for this view.

        Forwards to :func:`~newton.eval_inverse_dynamics_passive` with an
        articulation mask derived from this view and the optional view-local
        ``mask``. Each non-``None`` output is computed; entries belonging to
        articulations outside the selection are zero.

        .. experimental::

        Args:
            state: The state containing the current generalized
                coordinates and velocities. ``state.body_q`` must
                already reflect ``state.joint_q``.
            mass_matrix: Optional output, shape
                ``(model.articulation_count,
                model.max_dofs_per_articulation,
                model.max_dofs_per_articulation)``, dtype float. Entry units
                depend on the row and column DOF types: [kg] for two
                translational DOFs, [kg·m] for mixed translational/rotational
                DOFs, and [kg·m²] for two rotational DOFs.
            gravity_force: Optional gravity-force output [N or N·m, depending
                on joint type], shape ``(model.joint_dof_count,)``, dtype float.
            coriolis_force: Optional Coriolis + centrifugal-force output [N or
                N·m, depending on joint type], shape
                ``(model.joint_dof_count,)``, dtype float.
            mask: Optional mask of articulations in this
                ArticulationView (all by default). Either 1-D
                ``[world_count]`` selecting whole worlds or 2-D
                ``[world_count, count_per_world]`` selecting individual
                articulations per world.
        """
        articulation_mask = self.get_model_articulation_mask(mask=mask)
        eval_inverse_dynamics_passive(
            self.model,
            state,
            mass_matrix=mass_matrix,
            gravity_force=gravity_force,
            coriolis_force=coriolis_force,
            mask=articulation_mask,
        )

    def eval_inverse_dynamics_force(
        self,
        state: State,
        *,
        mass_matrix: wp.array3d[wp.float32],
        joint_qdd: wp.array[wp.float32],
        coriolis_force: wp.array[wp.float32],
        gravity_force: wp.array[wp.float32],
        joint_f: wp.array[wp.float32],
        mask: wp.array[bool] | wp.array2d[bool] | None = None,
    ) -> None:
        """Compute inverse-dynamics joint forces for articulations in this view.

        Entries outside this view or the optional sub-selection are zeroed.

        .. experimental::

        Args:
            state: State providing body transforms consistent with the
                supplied mass matrix and bias forces.
            mass_matrix: Joint-space mass matrix, shape
                ``(model.articulation_count,
                model.max_dofs_per_articulation,
                model.max_dofs_per_articulation)``, dtype float. Entry units
                depend on the row and column DOF types: [kg] for two
                translational DOFs, [kg·m] for mixed translational/rotational
                DOFs, and [kg·m²] for two rotational DOFs.
            joint_qdd: Generalized joint accelerations [m/s² or rad/s²,
                depending on joint type], shape
                ``(model.joint_dof_count,)``, dtype float.
            coriolis_force: Coriolis + centrifugal force [N or N·m, depending
                on joint type], shape ``(model.joint_dof_count,)``, dtype float.
            gravity_force: Gravity force [N or N·m, depending on joint type],
                shape ``(model.joint_dof_count,)``, dtype float.
            joint_f: Output generalized joint force :math:`\tau` [N or N·m,
                depending on joint type], shape ``(model.joint_dof_count,)``,
                dtype float. Uses the same layout and convention as
                :attr:`~newton.Control.joint_f`.
            mask: Optional mask of articulations in this ArticulationView.
                Either 1-D ``[world_count]`` or 2-D
                ``[world_count, count_per_world]``.
        """
        articulation_mask = self.get_model_articulation_mask(mask=mask)
        eval_inverse_dynamics_force(
            self.model,
            state,
            mass_matrix=mass_matrix,
            joint_qdd=joint_qdd,
            coriolis_force=coriolis_force,
            gravity_force=gravity_force,
            joint_f=joint_f,
            mask=articulation_mask,
        )

    # ========================================================================================
    # Actuator parameter access

    def _get_actuator_dof_mapping(self, actuator: Actuator):
        if actuator not in self._actuator_dof_mapping_cache:
            self._actuator_dof_mapping_cache[actuator] = self._create_actuator_dof_mapping(actuator)
        return self._actuator_dof_mapping_cache[actuator]

    def _create_actuator_dof_mapping(self, actuator: Actuator):
        """
        Build mapping from view DOF positions to actuator parameter indices.

        Note:
            Assumes SISO actuators (one DOF per actuator).

        Returns array of shape (world_count * dofs_per_world,) where each element is:
        - actuator parameter index if that DOF is actuated
        - -1 if that DOF is not actuated by this actuator
        """
        num_actuators = actuator.indices.shape[0]
        actuators_per_world = num_actuators // self.world_count

        dof_layout = self.frequency_layouts.get(AttributeFrequency.JOINT_DOF)
        if dof_layout is None:
            reason = self._unavailable_reasons[AttributeFrequency.JOINT_DOF]
            raise AttributeError(f"Actuator parameter access is unavailable: {reason}")
        dofs_per_arti = dof_layout.selected_value_count
        dofs_per_world = dofs_per_arti * self.count_per_world

        if dofs_per_world == 0:
            return wp.empty(0, dtype=int, device=self.device)

        mapping = wp.full(self.world_count * dofs_per_world, -1, dtype=int, device=self.device)

        if dof_layout.is_contiguous:
            wp.launch(
                build_actuator_dof_mapping_slice_kernel,
                dim=actuators_per_world,
                inputs=[
                    actuator.indices,
                    actuators_per_world,
                    dof_layout.offset,
                    dof_layout.slice.start,
                    dof_layout.slice.stop,
                    dof_layout.stride_within_worlds,
                    self.count_per_world,
                    dofs_per_arti,
                    dofs_per_world,
                    self.world_count,
                ],
                outputs=[mapping],
                device=self.device,
            )
        else:
            wp.launch(
                build_actuator_dof_mapping_indices_kernel,
                dim=actuators_per_world,
                inputs=[
                    actuator.indices,
                    dof_layout.indices,
                    dof_layout.offset,
                    dof_layout.stride_within_worlds,
                    self.count_per_world,
                    actuators_per_world,
                    dofs_per_arti,
                    dofs_per_world,
                    self.world_count,
                ],
                outputs=[mapping],
                device=self.device,
            )

        return mapping

    def get_actuator_parameter(self, actuator: Actuator, component: Any, name: str):
        """Read an actuator-component parameter for every DOF in this view.

        The returned array covers all DOFs selected by the view (one column
        per DOF, one row per world).  DOFs that are not driven by
        *actuator* are left at zero; driven DOFs contain the
        corresponding value gathered from ``component.<name>``.

        Args:
            actuator: Actuator instance whose DOF indices determine which
                view DOFs are considered actuated.
            component: The component that owns the parameter — a
                :class:`~newton.actuators.DriveBase`,
                :class:`~newton.actuators.ClampingBase`, or
                :class:`~newton.actuators.Delay` instance.
            name: Attribute name on *component* (e.g. ``"kp"``, ``"max_effort"``,
                ``"delay_steps"``).

        Returns:
            Parameter values shaped ``(world_count, dofs_per_world)`` where
            ``dofs_per_world`` is the total number of DOFs in the view (not
            just the actuated subset).
        """
        mapping = self._get_actuator_dof_mapping(actuator)
        if len(mapping) == 0:
            return wp.empty((self.world_count, 0), dtype=float, device=self.device)

        src = getattr(component, name)
        dofs_per_world = len(mapping) // self.world_count

        dst = wp.zeros(len(mapping), dtype=src.dtype, device=self.device)
        wp.launch(
            _gather_1d_kernel,
            dim=len(mapping),
            inputs=[src, mapping],
            outputs=[dst],
            device=self.device,
        )
        return dst.reshape((self.world_count, dofs_per_world))

    def set_actuator_parameter(
        self,
        actuator: Actuator,
        component: Any,
        name: str,
        values: wp.array,
        mask=None,
    ) -> None:
        """Write an actuator-component parameter for every DOF in this view.

        *values* must cover all DOFs in the view (one column per DOF, one row
        per world).  Only entries whose DOFs are actually driven by *actuator*
        are written back to ``component.<name>``; the rest are ignored.

        Args:
            actuator: Actuator instance whose DOF indices determine which
                view DOFs are considered actuated.
            component: The component that owns the parameter — a
                :class:`~newton.actuators.DriveBase`,
                :class:`~newton.actuators.ClampingBase`, or
                :class:`~newton.actuators.Delay` instance.
            name: Attribute name on *component* (e.g. ``"kp"``, ``"max_effort"``,
                ``"delay_steps"``).
            values: New parameter values shaped ``(world_count, dofs_per_world)``
                where ``dofs_per_world`` is the total number of DOFs in the view.
            mask: Per-world mask ``(world_count,)``. Only masked worlds are updated.
        """
        mask = self._resolve_world_mask(mask)
        mapping = self._get_actuator_dof_mapping(actuator)
        if len(mapping) == 0:
            return

        dst = getattr(component, name)
        dofs_per_world = len(mapping) // self.world_count
        expected_shape = (self.world_count, dofs_per_world, *dst.shape[1:])

        if not is_array(values):
            values = wp.array(values, dtype=dst.dtype, shape=expected_shape, device=self.device, copy=False)

        if values.shape[:2] != expected_shape[:2]:
            raise ValueError(f"Expected values shape {expected_shape}, got {values.shape}")

        wp.launch(
            _scatter_masked_2d_kernel,
            dim=(self.world_count, dofs_per_world),
            inputs=[values, mapping, mask, dofs_per_world],
            outputs=[dst],
            device=self.device,
        )
