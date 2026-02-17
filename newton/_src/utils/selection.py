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

import functools
from fnmatch import fnmatch
from typing import Any

import warp as wp
from warp.types import is_array

from ..sim import Control, JointType, Model, State, eval_fk, eval_jacobian, eval_mass_matrix

AttributeFrequency = Model.AttributeFrequency


@wp.kernel
def set_model_articulation_mask_kernel(
    world_arti_mask: wp.array2d(dtype=bool),  # (world, arti) mask in ArticulationView
    view_to_model_map: wp.array2d(dtype=int),  # map (world, arti) indices to Model articulation id
    model_articulation_mask: wp.array(dtype=bool),  # output: mask of Model articulation indices
):
    """
    Set Model articulation mask from a 2D (world, arti) mask in an ArticulationView.
    """
    world, arti = wp.tid()
    if world_arti_mask[world, arti]:
        model_articulation_mask[view_to_model_map[world, arti]] = True


@wp.kernel
def set_model_articulation_mask_per_world_kernel(
    world_mask: wp.array(dtype=bool),  # world mask in ArticulationView
    view_to_model_map: wp.array2d(dtype=int),  # map (world, arti) indices to Model articulation id
    model_articulation_mask: wp.array(dtype=bool),  # output: mask of Model articulation indices
):
    """
    Set Model articulation mask from a 1D world mask in an ArticulationView.
    """
    world, arti = wp.tid()
    if world_mask[world]:
        model_articulation_mask[view_to_model_map[world, arti]] = True


# @wp.kernel
# def set_articulation_attribute_1d_kernel(
#     view_mask: wp.array2d(dtype=bool),  # (world, arti) mask in ArticulationView
#     values: Any,  # 1d array or indexedarray
#     attrib: Any,  # 1d array or indexedarray
# ):
#     i = wp.tid()
#     if view_mask[i]:
#         attrib[i] = values[i]


# @wp.kernel
# def set_articulation_attribute_2d_kernel(
#     view_mask: wp.array2d(dtype=bool),  # (world, arti) mask in ArticulationView
#     values: Any,  # 2d array or indexedarray
#     attrib: Any,  # 2d array or indexedarray
# ):
#     i, j = wp.tid()
#     if view_mask[i, j]:
#         attrib[i, j] = values[i, j]


@wp.kernel
def set_articulation_attribute_3d_kernel(
    view_mask: wp.array2d(dtype=bool),  # (world, arti) mask in ArticulationView
    values: Any,  # 3d array or indexedarray
    attrib: Any,  # 3d array or indexedarray
):
    i, j, k = wp.tid()
    if view_mask[i, j]:
        attrib[i, j, k] = values[i, j, k]


@wp.kernel
def set_articulation_attribute_4d_kernel(
    view_mask: wp.array2d(dtype=bool),  # (world, arti) mask in ArticulationView
    values: Any,  # 4d array or indexedarray
    attrib: Any,  # 4d array or indexedarray
):
    i, j, k, l = wp.tid()
    if view_mask[i, j]:
        attrib[i, j, k, l] = values[i, j, k, l]


# @wp.kernel
# def set_articulation_attribute_1d_per_world_kernel(
#     view_mask: wp.array(dtype=bool),  # world mask in ArticulationView
#     values: Any,  # 1d array or indexedarray
#     attrib: Any,  # 1d array or indexedarray
# ):
#     i = wp.tid()
#     if view_mask[i]:
#         attrib[i] = values[i]


# @wp.kernel
# def set_articulation_attribute_2d_per_world_kernel(
#     view_mask: wp.array(dtype=bool),  # world mask in ArticulationView
#     values: Any,  # 2d array or indexedarray
#     attrib: Any,  # 2d array or indexedarray
# ):
#     i, j = wp.tid()
#     if view_mask[i]:
#         attrib[i, j] = values[i, j]


@wp.kernel
def set_articulation_attribute_3d_per_world_kernel(
    view_mask: wp.array(dtype=bool),  # world mask in ArticulationView
    values: Any,  # 3d array or indexedarray
    attrib: Any,  # 3d array or indexedarray
):
    i, j, k = wp.tid()
    if view_mask[i]:
        attrib[i, j, k] = values[i, j, k]


@wp.kernel
def set_articulation_attribute_4d_per_world_kernel(
    view_mask: wp.array(dtype=bool),  # world mask in ArticulationView
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

    def __str__(self):
        indices = self.indices if self.indices is not None else self.slice
        return f"FrequencyLayout(\n    offset: {self.offset}\n    stride_between_worlds: {self.stride_between_worlds}\n    stride_within_worlds: {self.stride_within_worlds}\n    indices: {indices}\n)"


def get_name_from_key(key: str):
    return key.split("/")[-1]


def find_matching_ids(pattern: str, keys: list[str], world_ids, world_count: int):
    grouped_ids = [[] for _ in range(world_count)]  # ids grouped by world (exclude world -1)
    global_ids = []  # ids in world -1
    for id, key in enumerate(keys):
        if fnmatch(key, pattern):
            world = world_ids[id]
            if world == -1:
                global_ids.append(id)
            elif world >= 0 and world < world_count:
                grouped_ids[world].append(id)
            else:
                raise ValueError(f"World index out of range: {world}")
    return grouped_ids, global_ids


def all_equal(values):
    return all(x == values[0] for x in values)


def list_of_lists(n):
    return [[] for _ in range(n)]


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
    attribute access and modification for simulation and control.

    This is useful in RL and batched simulation workflows where a single policy or
    control routine operates on many parallel environments with consistent tensor shapes.

    Example:
        >>> import newton
        >>> view = newton.selection.ArticulationView(model, pattern="robot*")
        >>> q = view.get_dof_positions(state)
        >>> q_np = q.numpy()
        >>> q_np[..., 0] = 0.0
        >>> view.set_dof_positions(state, q_np)

    Args:
        model (Model): The model containing the articulations.
        pattern (str): Pattern to match articulation keys.
        include_joints (list[str | int] | None): List of joint names, patterns, or indices to include.
        exclude_joints (list[str | int] | None): List of joint names, patterns, or indices to exclude.
        include_links (list[str | int] | None): List of link names, patterns, or indices to include.
        exclude_links (list[str | int] | None): List of link names, patterns, or indices to exclude.
        include_joint_types (list[int] | None): List of joint types to include.
        exclude_joint_types (list[int] | None): List of joint types to exclude.
        verbose (bool | None): If True, prints selection summary.
    """

    def __init__(
        self,
        model: Model,
        pattern: str,
        include_joints: list[str | int] | None = None,
        exclude_joints: list[str | int] | None = None,
        include_links: list[str | int] | None = None,
        exclude_links: list[str | int] | None = None,
        include_joint_types: list[int] | None = None,
        exclude_joint_types: list[int] | None = None,
        verbose: bool | None = None,
    ):
        self.model = model
        self.device = model.device

        if verbose is None:
            verbose = wp.config.verbose

        # FIXME: avoid/reduce this readback?
        model_articulation_start = model.articulation_start.numpy()
        model_articulation_world = model.articulation_world.numpy()
        model_joint_type = model.joint_type.numpy()
        model_joint_child = model.joint_child.numpy()
        model_joint_q_start = model.joint_q_start.numpy()
        model_joint_qd_start = model.joint_qd_start.numpy()

        # get articulation ids grouped by world
        articulation_ids, global_articulation_ids = find_matching_ids(
            pattern, model.articulation_key, model_articulation_world, model.world_count
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
        arti_shape_ids = []
        arti_shape_names = []

        # gather joint info
        arti_joint_begin = int(model_articulation_start[arti_0])
        arti_joint_end = int(model_articulation_start[arti_0 + 1])
        arti_joint_count = arti_joint_end - arti_joint_begin
        arti_joint_dof_begin = int(model_joint_qd_start[arti_joint_begin])
        arti_joint_dof_end = int(model_joint_qd_start[arti_joint_end])
        arti_joint_dof_count = arti_joint_dof_end - arti_joint_dof_begin
        arti_joint_coord_begin = int(model_joint_q_start[arti_joint_begin])
        arti_joint_coord_end = int(model_joint_q_start[arti_joint_end])
        arti_joint_coord_count = arti_joint_coord_end - arti_joint_coord_begin
        for joint_id in range(arti_joint_begin, arti_joint_end):
            # joint_id = arti_joint_begin + idx
            arti_joint_ids.append(joint_id)
            arti_joint_names.append(get_name_from_key(model.joint_key[joint_id]))
            arti_joint_types.append(model_joint_type[joint_id])
            link_id = int(model_joint_child[joint_id])
            arti_link_ids.append(link_id)

        # use link order as they appear in the model
        arti_link_ids = sorted(arti_link_ids)
        arti_link_count = len(arti_link_ids)
        for link_id in arti_link_ids:
            arti_link_names.append(get_name_from_key(model.body_key[link_id]))
            arti_shape_ids.extend(model.body_shapes[link_id])

        # use shape order as they appear in the model
        arti_shape_ids = sorted(arti_shape_ids)
        arti_shape_count = len(arti_shape_ids)
        for shape_id in arti_shape_ids:
            arti_shape_names.append(get_name_from_key(model.shape_key[shape_id]))

        # compute counts and offsets of joints, links, etc.
        joint_starts = list_of_lists(world_count)
        joint_counts = list_of_lists(world_count)
        joint_dof_starts = list_of_lists(world_count)
        joint_dof_counts = list_of_lists(world_count)
        joint_coord_starts = list_of_lists(world_count)
        joint_coord_counts = list_of_lists(world_count)
        root_joint_types = list_of_lists(world_count)
        link_starts = list_of_lists(world_count)
        shape_starts = list_of_lists(world_count)
        shape_counts = list_of_lists(world_count)
        for world_id in range(world_count):
            for arti_id in articulation_ids[world_id]:
                # joints
                joint_start = int(model_articulation_start[arti_id])
                joint_end = int(model_articulation_start[arti_id + 1])
                joint_starts[world_id].append(joint_start)
                joint_counts[world_id].append(joint_end - joint_start)
                # joint dofs
                joint_dof_start = int(model_joint_qd_start[joint_start])
                joint_dof_end = int(model_joint_qd_start[joint_end])
                joint_dof_starts[world_id].append(joint_dof_start)
                joint_dof_counts[world_id].append(joint_dof_end - joint_dof_start)
                # joint coords
                joint_coord_start = int(model_joint_q_start[joint_start])
                joint_coord_end = int(model_joint_q_start[joint_end])
                joint_coord_starts[world_id].append(joint_coord_start)
                joint_coord_counts[world_id].append(joint_coord_end - joint_coord_start)
                # root joint types
                root_joint_types[world_id].append(int(model_joint_type[joint_start]))
                # links and shapes
                link_ids = []
                shape_ids = []
                for j in range(joint_start, joint_end):
                    link_id = int(model_joint_child[j])
                    link_ids.append(link_id)
                    link_shapes = model.body_shapes.get(link_id, [])
                    shape_ids.extend(link_shapes)
                link_starts[world_id].append(min(link_ids))
                num_shapes = len(shape_ids)
                if num_shapes > 0:
                    shape_starts[world_id].append(min(shape_ids))
                else:
                    shape_starts[world_id].append(-1)
                shape_counts[world_id].append(num_shapes)

        # make sure counts are the same for all articulations
        # NOTE: we currently assume that link count is the same as joint count
        if not (
            all_equal(joint_counts)
            and all_equal(joint_dof_counts)
            and all_equal(joint_coord_counts)
            and all_equal(root_joint_types)
            and all_equal(shape_counts)
        ):
            raise ValueError("Articulations are not identical")

        self.root_joint_type = root_joint_types[0][0]
        # fixed base means that all linear and angular degrees of freedom are locked at the root
        self.is_fixed_base = self.root_joint_type == JointType.FIXED
        # floating base means that all linear and angular degrees of freedom are unlocked at the root
        # (though there might be constraints like distance)
        self.is_floating_base = self.root_joint_type in (JointType.FREE, JointType.DISTANCE)

        joint_offset = joint_starts[0][0]
        joint_dof_offset = joint_dof_starts[0][0]
        joint_coord_offset = joint_coord_starts[0][0]
        link_offset = link_starts[0][0]
        if arti_shape_count > 0:
            shape_offset = shape_starts[0][0]
        else:
            shape_offset = 0

        # compute "outer" strides (strides between worlds)
        if world_count > 1:
            outer_joint_strides = []
            outer_joint_dof_strides = []
            outer_joint_coord_strides = []
            outer_link_strides = []
            outer_shape_strides = []
            for world_id in range(1, world_count):
                outer_joint_strides.append(joint_starts[world_id][0] - joint_starts[world_id - 1][0])
                outer_joint_dof_strides.append(joint_dof_starts[world_id][0] - joint_dof_starts[world_id - 1][0])
                outer_joint_coord_strides.append(joint_coord_starts[world_id][0] - joint_coord_starts[world_id - 1][0])
                outer_link_strides.append(link_starts[world_id][0] - link_starts[world_id - 1][0])
                outer_shape_strides.append(shape_starts[world_id][0] - shape_starts[world_id - 1][0])

            # make sure outer strides are uniform
            if not (
                all_equal(outer_joint_strides)
                and all_equal(outer_joint_dof_strides)
                and all_equal(outer_joint_coord_strides)
                and all_equal(outer_link_strides)
                and all_equal(outer_shape_strides)
            ):
                raise ValueError("Non-uniform strides between worlds are not supported")

            outer_joint_stride = outer_joint_strides[0]
            outer_joint_dof_stride = outer_joint_dof_strides[0]
            outer_joint_coord_stride = outer_joint_coord_strides[0]
            outer_link_stride = outer_link_strides[0]
            outer_shape_stride = outer_shape_strides[0]
        else:
            outer_joint_stride = arti_joint_count
            outer_joint_dof_stride = arti_joint_dof_count
            outer_joint_coord_stride = arti_joint_coord_count
            outer_link_stride = arti_link_count
            outer_shape_stride = arti_shape_count

        # compute "inner" strides (strides within worlds)
        if count_per_world > 1:
            inner_joint_strides = list_of_lists(world_count)
            inner_joint_dof_strides = list_of_lists(world_count)
            inner_joint_coord_strides = list_of_lists(world_count)
            inner_link_strides = list_of_lists(world_count)
            inner_shape_strides = list_of_lists(world_count)
            for world_id in range(world_count):
                for i in range(1, count_per_world):
                    inner_joint_strides[world_id].append(joint_starts[world_id][i] - joint_starts[world_id][i - 1])
                    inner_joint_dof_strides[world_id].append(
                        joint_dof_starts[world_id][i] - joint_dof_starts[world_id][i - 1]
                    )
                    inner_joint_coord_strides[world_id].append(
                        joint_coord_starts[world_id][i] - joint_coord_starts[world_id][i - 1]
                    )
                    inner_link_strides[world_id].append(link_starts[world_id][i] - link_starts[world_id][i - 1])
                    inner_shape_strides[world_id].append(shape_starts[world_id][i] - shape_starts[world_id][i - 1])

            # make sure inner strides are uniform
            if not (
                all_equal(inner_joint_strides)
                and all_equal(inner_joint_dof_strides)
                and all_equal(inner_joint_coord_strides)
                and all_equal(inner_link_strides)
                and all_equal(inner_shape_strides)
            ):
                raise ValueError("Non-uniform strides within worlds are not supported")

            inner_joint_stride = inner_joint_strides[0][0]
            inner_joint_dof_stride = inner_joint_dof_strides[0][0]
            inner_joint_coord_stride = inner_joint_coord_strides[0][0]
            inner_link_stride = inner_link_strides[0][0]
            inner_shape_stride = inner_shape_strides[0][0]
        else:
            inner_joint_stride = arti_joint_count
            inner_joint_dof_stride = arti_joint_dof_count
            inner_joint_coord_stride = arti_joint_coord_count
            inner_link_stride = arti_link_count
            inner_shape_stride = arti_shape_count

        # create joint inclusion set
        if include_joints is None and include_joint_types is None:
            joint_include_indices = set(range(arti_joint_count))
        else:
            joint_include_indices = set()
            if include_joints is not None:
                for id in include_joints:
                    if isinstance(id, str):
                        for idx, name in enumerate(arti_joint_names):
                            if fnmatch(name, id):
                                joint_include_indices.add(idx)
                    elif isinstance(id, int):
                        if id >= 0 and id < arti_joint_count:
                            joint_include_indices.add(id)
                    else:
                        raise TypeError(f"Joint ids must be strings or integers, got {id} of type {type(id)}")
            if include_joint_types is not None:
                for idx in range(arti_joint_count):
                    if arti_joint_types[idx] in include_joint_types:
                        joint_include_indices.add(idx)

        # create joint exclusion set
        joint_exclude_indices = set()
        if exclude_joints is not None:
            for id in exclude_joints:
                if isinstance(id, str):
                    for idx, name in enumerate(arti_joint_names):
                        if fnmatch(name, id):
                            joint_exclude_indices.add(idx)
                elif isinstance(id, int):
                    if id >= 0 and id < arti_joint_count:
                        joint_exclude_indices.add(id)
                else:
                    raise TypeError(f"Joint ids must be strings or integers, got {id} of type {type(id)}")
        if exclude_joint_types is not None:
            for idx in range(arti_joint_count):
                if arti_joint_types[idx] in exclude_joint_types:
                    joint_exclude_indices.add(idx)

        # create link inclusion set
        if include_links is None:
            link_include_indices = set(range(arti_link_count))
        else:
            link_include_indices = set()
            if include_links is not None:
                for id in include_links:
                    if isinstance(id, str):
                        for idx, name in enumerate(arti_link_names):
                            if fnmatch(name, id):
                                link_include_indices.add(idx)
                    elif isinstance(id, int):
                        if id >= 0 and id < arti_link_count:
                            link_include_indices.add(id)
                    else:
                        raise TypeError(f"Link ids must be strings or integers, got {id} of type {type(id)}")

        # create link exclusion set
        link_exclude_indices = set()
        if exclude_links is not None:
            for id in exclude_links:
                if isinstance(id, str):
                    for idx, name in enumerate(arti_link_names):
                        if fnmatch(name, id):
                            link_exclude_indices.add(idx)
                elif isinstance(id, int):
                    if id >= 0 and id < arti_link_count:
                        link_exclude_indices.add(id)
                else:
                    raise TypeError(f"Link ids must be strings or integers, got {id} of type {type(id)}")

        # compute selected indices
        selected_joint_indices = sorted(joint_include_indices - joint_exclude_indices)
        selected_link_indices = sorted(link_include_indices - link_exclude_indices)

        self.joint_names = []
        self.joint_dof_names = []
        self.joint_dof_counts = []
        self.joint_coord_names = []
        self.joint_coord_counts = []
        self.link_names = []
        self.link_shapes = []
        self.shape_names = []

        # populate info for selected joints and dofs
        selected_joint_dof_indices = []
        selected_joint_coord_indices = []
        for joint_idx in selected_joint_indices:
            joint_id = arti_joint_ids[joint_idx]
            joint_name = arti_joint_names[joint_idx]
            self.joint_names.append(joint_name)
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
            shape_ids = model.body_shapes[body_id]
            for shape_id in shape_ids:
                arti_shape_idx = arti_shape_ids.index(shape_id)
                selected_shape_indices.append(arti_shape_idx)
                shape_link_idx[arti_shape_idx] = link_idx
            self.link_shapes.append([])

        selected_shape_indices = sorted(selected_shape_indices)
        for shape_idx, arti_shape_idx in enumerate(selected_shape_indices):
            self.shape_names.append(arti_shape_names[arti_shape_idx])
            link_idx = shape_link_idx[arti_shape_idx]
            self.link_shapes[link_idx].append(shape_idx)

        # selection counts
        self.count = articulation_count
        self.world_count = world_count
        self.count_per_world = count_per_world
        self.joint_count = len(selected_joint_indices)
        self.joint_dof_count = len(selected_joint_dof_indices)
        self.joint_coord_count = len(selected_joint_coord_indices)
        self.link_count = len(selected_link_indices)
        self.shape_count = len(selected_shape_indices)

        # TODO: document the layout conventions and requirements
        #
        # |ooXXXoXXXoXXXooo|ooXXXoXXXoXXXooo|ooXXXoXXXoXXXooo|ooXXXoXXXoXXXooo|
        # |  ^   ^   ^     |  ^   ^   ^     |  ^   ^   ^     |  ^   ^   ^     |
        #
        self.frequency_layouts = {
            AttributeFrequency.JOINT: FrequencyLayout(
                joint_offset,
                outer_joint_stride,
                inner_joint_stride,
                arti_joint_count,
                selected_joint_indices,
                self.device,
            ),
            AttributeFrequency.JOINT_DOF: FrequencyLayout(
                joint_dof_offset,
                outer_joint_dof_stride,
                inner_joint_dof_stride,
                arti_joint_dof_count,
                selected_joint_dof_indices,
                self.device,
            ),
            AttributeFrequency.JOINT_COORD: FrequencyLayout(
                joint_coord_offset,
                outer_joint_coord_stride,
                inner_joint_coord_stride,
                arti_joint_coord_count,
                selected_joint_coord_indices,
                self.device,
            ),
            AttributeFrequency.BODY: FrequencyLayout(
                link_offset, outer_link_stride, inner_link_stride, arti_link_count, selected_link_indices, self.device
            ),
            AttributeFrequency.SHAPE: FrequencyLayout(
                shape_offset,
                outer_shape_stride,
                inner_shape_stride,
                arti_shape_count,
                selected_shape_indices,
                self.device,
            ),
        }

        # ========================================================================================
        # Tendon discovery (for MuJoCo fixed tendons)
        # Tendons are associated with articulations by checking which articulation owns all their joints

        self.tendon_count = 0
        self.tendon_names = []

        # Check if model has MuJoCo tendon attributes
        if hasattr(model, "mujoco") and hasattr(model.mujoco, "tendon_joint"):
            mujoco_attrs = model.mujoco
            tendon_world_arr = mujoco_attrs.tendon_world.numpy()
            tendon_joint_adr_arr = mujoco_attrs.tendon_joint_adr.numpy()
            tendon_joint_num_arr = mujoco_attrs.tendon_joint_num.numpy()
            tendon_joint_arr = mujoco_attrs.tendon_joint.numpy()
            total_tendon_count = len(tendon_world_arr)

            if total_tendon_count > 0:
                # Build a mapping from joint index to articulation index
                joint_to_articulation = {}
                for arti_idx in range(len(model_articulation_start) - 1):
                    joint_begin = int(model_articulation_start[arti_idx])
                    joint_end = int(model_articulation_start[arti_idx + 1])
                    for j in range(joint_begin, joint_end):
                        joint_to_articulation[j] = arti_idx

                # For each articulation, find its tendons
                # A tendon belongs to an articulation if ALL its joints belong to that articulation
                tendon_to_articulation = {}
                for tendon_idx in range(total_tendon_count):
                    joint_adr = int(tendon_joint_adr_arr[tendon_idx])
                    joint_num = int(tendon_joint_num_arr[tendon_idx])

                    if joint_num == 0:
                        continue  # Skip empty tendons

                    articulations_in_tendon = set()
                    for j in range(joint_adr, joint_adr + joint_num):
                        joint_id = int(tendon_joint_arr[j])
                        if joint_id in joint_to_articulation:
                            articulations_in_tendon.add(joint_to_articulation[joint_id])

                    if len(articulations_in_tendon) > 1:
                        raise ValueError(
                            f"Tendon {tendon_idx} spans multiple articulations {articulations_in_tendon}, "
                            f"which is not supported by ArticulationView"
                        )

                    if len(articulations_in_tendon) == 1:
                        tendon_to_articulation[tendon_idx] = articulations_in_tendon.pop()

                # Group tendons by (world, articulation) and filter for selected articulations
                # Build a set of selected articulation IDs for fast lookup
                selected_arti_set = set()
                for world_artis in articulation_ids:
                    for arti_id in world_artis:
                        selected_arti_set.add(arti_id)

                # Find tendons belonging to the template articulation (first selected articulation)
                template_arti_id = articulation_ids[0][0]
                arti_tendon_ids = []  # Tendon indices belonging to the template articulation
                for tendon_idx, arti_id in tendon_to_articulation.items():
                    if arti_id == template_arti_id:
                        arti_tendon_ids.append(tendon_idx)

                arti_tendon_ids = sorted(arti_tendon_ids)
                arti_tendon_count = len(arti_tendon_ids)

                if arti_tendon_count > 0:
                    # Compute tendon layout similar to joints
                    # Group tendons by world and articulation to compute strides
                    tendon_starts = list_of_lists(world_count)
                    tendon_counts = list_of_lists(world_count)

                    for world_id in range(world_count):
                        for arti_id in articulation_ids[world_id]:
                            arti_tendons = [t for t, a in tendon_to_articulation.items() if a == arti_id]
                            arti_tendons = sorted(arti_tendons)
                            if len(arti_tendons) > 0:
                                tendon_starts[world_id].append(min(arti_tendons))
                            else:
                                tendon_starts[world_id].append(-1)
                            tendon_counts[world_id].append(len(arti_tendons))

                    # Validate uniform tendon counts
                    if not all_equal(tendon_counts):
                        raise ValueError("Articulations have different tendon counts, which is not supported")

                    tendon_offset = arti_tendon_ids[0] if arti_tendon_ids else 0

                    # Compute outer stride (between worlds)
                    if world_count > 1:
                        outer_tendon_strides = []
                        for world_id in range(1, world_count):
                            if tendon_starts[world_id][0] >= 0 and tendon_starts[world_id - 1][0] >= 0:
                                outer_tendon_strides.append(tendon_starts[world_id][0] - tendon_starts[world_id - 1][0])
                        if outer_tendon_strides and not all_equal(outer_tendon_strides):
                            raise ValueError("Non-uniform tendon strides between worlds are not supported")
                        outer_tendon_stride = outer_tendon_strides[0] if outer_tendon_strides else arti_tendon_count
                    else:
                        outer_tendon_stride = arti_tendon_count

                    # Compute inner stride (within worlds)
                    if count_per_world > 1:
                        inner_tendon_strides = list_of_lists(world_count)
                        for world_id in range(world_count):
                            for i in range(1, count_per_world):
                                if tendon_starts[world_id][i] >= 0 and tendon_starts[world_id][i - 1] >= 0:
                                    inner_tendon_strides[world_id].append(
                                        tendon_starts[world_id][i] - tendon_starts[world_id][i - 1]
                                    )
                        # Flatten and check uniformity
                        flat_inner = [s for lst in inner_tendon_strides for s in lst]
                        if flat_inner and not all_equal(flat_inner):
                            raise ValueError("Non-uniform tendon strides within worlds are not supported")
                        inner_tendon_stride = flat_inner[0] if flat_inner else arti_tendon_count
                    else:
                        inner_tendon_stride = arti_tendon_count

                    # Validate that tendon indices are contiguous
                    # Non-contiguous tendons (e.g., interleaved with other articulations) are not supported
                    expected_contiguous = list(range(tendon_offset, tendon_offset + arti_tendon_count))
                    if arti_tendon_ids != expected_contiguous:
                        raise ValueError(
                            f"Tendons for articulation are not contiguous (indices {arti_tendon_ids}, "
                            f"expected {expected_contiguous}). Non-contiguous tendons are not supported "
                            f"by ArticulationView."
                        )

                    # Tendons are contiguous, use range-based indexing
                    selected_tendon_indices = list(range(arti_tendon_count))

                    # Store with the full namespaced frequency key (mujoco:tendon)
                    self.frequency_layouts["mujoco:tendon"] = FrequencyLayout(
                        tendon_offset,
                        outer_tendon_stride,
                        inner_tendon_stride,
                        arti_tendon_count,
                        selected_tendon_indices,
                        self.device,
                    )

                    self.tendon_count = arti_tendon_count

                    # Populate tendon_names from model.mujoco.tendon_key if available
                    if hasattr(mujoco_attrs, "tendon_key"):
                        for tendon_idx in arti_tendon_ids:
                            if tendon_idx < len(mujoco_attrs.tendon_key):
                                self.tendon_names.append(mujoco_attrs.tendon_key[tendon_idx])
                            else:
                                self.tendon_names.append(f"tendon_{tendon_idx}")

        self.joints_contiguous = self.frequency_layouts[AttributeFrequency.JOINT].is_contiguous
        self.joint_dofs_contiguous = self.frequency_layouts[AttributeFrequency.JOINT_DOF].is_contiguous
        self.joint_coords_contiguous = self.frequency_layouts[AttributeFrequency.JOINT_COORD].is_contiguous
        self.links_contiguous = self.frequency_layouts[AttributeFrequency.BODY].is_contiguous
        self.shapes_contiguous = self.frequency_layouts[AttributeFrequency.SHAPE].is_contiguous

        # articulation ids grouped by world
        self.articulation_ids = wp.array(articulation_ids, dtype=int, device=self.device)

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
            for link_idx in range(self.link_count):
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

    # ========================================================================================
    # Generic attribute API

    @functools.lru_cache(maxsize=None)  # noqa
    def _get_attribute_array(self, name: str, source: Model | State | Control, _slice: Slice | int | None = None):
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

        # Handle custom frequencies (string frequencies)
        if isinstance(frequency, str):
            # Check if this is a supported custom frequency
            # Tendon frequency can be "tendon" or "mujoco:tendon" (with namespace prefix)
            if frequency == "tendon" or frequency.endswith(":tendon"):
                # Normalize to the stored key format "mujoco:tendon"
                normalized_frequency = "mujoco:tendon"
                layout = self.frequency_layouts.get(normalized_frequency)
                if layout is None:
                    raise AttributeError(
                        f"Attribute '{name}' has frequency '{frequency}' but no tendons were found "
                        f"in the selected articulations"
                    )
            else:
                raise AttributeError(
                    f"Attribute '{name}' has custom frequency '{frequency}' which is not "
                    f"supported by ArticulationView. Custom frequencies are for custom entity types "
                    f"that are not part of articulations."
                )
        else:
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
        elif isinstance(_slice, int):
            _slice = slice(_slice, _slice + 1)

        if _slice is None:
            value_count = layout.value_count
            if is_indexed:
                value_slice = layout.indices
            else:
                value_slice = layout.slice
        else:
            value_count = _slice.stop - _slice.start
            if is_indexed:
                value_slice = layout.indices[_slice]
            else:
                value_slice = _slice

        shape = (self.world_count, self.count_per_world, value_count)
        strides = (
            layout.stride_between_worlds * value_stride,
            layout.stride_within_worlds * value_stride,
            value_stride,
        )
        slices = (slice(self.world_count), slice(self.count_per_world), value_slice)

        # trailing dimensions for multidimensional attributes
        trailing_shape = attrib.shape[1:]
        trailing_strides = attrib.strides[1:]
        trailing_slices = [slice(s) for s in trailing_shape]

        shape = (*shape, *trailing_shape)
        strides = (*strides, *trailing_strides)
        slices = (*slices, *trailing_slices)

        # construct reshaped attribute array
        attrib = wp.array(
            ptr=int(attrib.ptr) + layout.offset * value_stride,
            dtype=attrib.dtype,
            shape=shape,
            strides=strides,
            device=attrib.device,
            copy=False,
        )

        # apply selection (slices or indices)
        attrib = attrib[slices]

        if is_indexed:
            # create a contiguous staging array
            attrib._staging_array = wp.empty_like(attrib)
        else:
            # fixup for empty slices - FIXME: this should be handled by Warp, above
            if attrib.size == 0:
                attrib.ptr = None

        return attrib

    def _get_attribute_values(self, name: str, source: Model | State | Control, _slice: slice | None = None):
        attrib = self._get_attribute_array(name, source, _slice=_slice)
        if hasattr(attrib, "_staging_array"):
            wp.copy(attrib._staging_array, attrib)
            return attrib._staging_array
        else:
            return attrib

    def _set_attribute_values(
        self, name: str, target: Model | State | Control, values, mask=None, _slice: slice | None = None
    ):
        attrib = self._get_attribute_array(name, target, _slice=_slice)

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
            name (str): The name of the attribute to get.
            source (Model | State | Control): The source from which to get the attribute.

        Returns:
            array: The attribute values (dtype matches the attribute).
        """
        return self._get_attribute_values(name, source)

    def set_attribute(self, name: str, target: Model | State | Control, values, mask=None):
        """
        Set an attribute in the target (Model, State, or Control).

        Args:
            name (str): The name of the attribute to set.
            target (Model | State | Control): The target where to set the attribute.
            values (array): The values to set for the attribute.
            mask (array): Mask of articulations in this ArticulationView (all by default).

        .. note::
            When setting attributes on the Model, it may be necessary to inform the solver about
            such changes by calling :meth:`newton.solvers.SolverBase.notify_model_changed` after finished
            setting Model attributes.
        """
        self._set_attribute_values(name, target, values, mask=mask)

    # ========================================================================================
    # Convenience wrappers to align with legacy tensor API

    def get_root_transforms(self, source: Model | State):
        """
        Get the root transforms of the articulations.

        Args:
            source (Model | State): Where to get the root transforms (Model or State).

        Returns:
            array: The root transforms (dtype=wp.transform).
        """
        if self.is_floating_base:
            attrib = self._get_attribute_values("joint_q", source, _slice=Slice(0, 7))
        else:
            attrib = self._get_attribute_values("joint_X_p", self.model, _slice=0)

        if attrib.dtype is wp.transform:
            return attrib
        else:
            return wp.array(attrib, dtype=wp.transform, device=self.device, copy=False)

    def set_root_transforms(self, target: Model | State, values: wp.array, mask=None):
        """
        Set the root transforms of the articulations.
        Call :meth:`eval_fk` to apply changes to all articulation links.

        Args:
            target (Model | State): Where to set the root transforms (Model or State).
            values (array): The root transforms to set (dtype=wp.transform).
            mask (array): Mask of articulations in this ArticulationView (all by default).
        """
        if self.is_floating_base:
            self._set_attribute_values("joint_q", target, values, mask=mask, _slice=Slice(0, 7))
        else:
            self._set_attribute_values("joint_X_p", self.model, values, mask=mask, _slice=0)

    def get_root_velocities(self, source: Model | State):
        """
        Get the root velocities of the articulations.

        Args:
            source (Model | State): Where to get the root velocities (Model or State).

        Returns:
            array: The root velocities (dtype=wp.spatial_vector).
        """
        if self.is_floating_base:
            attrib = self._get_attribute_values("joint_qd", source, _slice=Slice(0, 6))
        else:
            # FIXME? Non-floating articulations have no root velocities.
            return None

        if attrib.dtype is wp.spatial_vector:
            return attrib
        else:
            return wp.array(attrib, dtype=wp.spatial_vector, device=self.device, copy=False)

    def set_root_velocities(self, target: Model | State, values: wp.array, mask=None):
        """
        Set the root velocities of the articulations.

        Args:
            target (Model | State): Where to set the root velocities (Model or State).
            values (array): The root velocities to set (dtype=wp.spatial_vector).
            mask (array): Mask of articulations in this ArticulationView (all by default).
        """
        if self.is_floating_base:
            self._set_attribute_values("joint_qd", target, values, mask=mask, _slice=Slice(0, 6))
        else:
            return  # no-op

    def get_link_transforms(self, source: "Model | State"):
        """
        Get the world-space transforms of all links in the selected articulations.

        Args:
            source (Model | State): The source from which to retrieve the link transforms.

        Returns:
            array: The link transforms (dtype=wp.transform).
        """
        return self._get_attribute_values("body_q", source)

    def get_link_velocities(self, source: "Model | State"):
        """
        Get the world-space spatial velocities of all links in the selected articulations.

        Args:
            source (Model | State): The source from which to retrieve the link velocities.

        Returns:
            array: The link velocities (dtype=wp.spatial_vector).
        """
        return self._get_attribute_values("body_qd", source)

    def get_dof_positions(self, source: "Model | State"):
        """
        Get the joint coordinate positions (DoF positions) for the selected articulations.

        Args:
            source (Model | State): The source from which to retrieve the DoF positions.

        Returns:
            array: The joint coordinate positions (dtype=float).
        """
        return self._get_attribute_values("joint_q", source)

    def set_dof_positions(self, target: "Model | State", values, mask=None):
        """
        Set the joint coordinate positions (DoF positions) for the selected articulations.

        Args:
            target (Model | State): The target where to set the DoF positions.
            values (array): The values to set (dtype=float).
            mask (array, optional): Mask of articulations in this ArticulationView (all by default).
        """
        self._set_attribute_values("joint_q", target, values, mask=mask)

    def get_dof_velocities(self, source: "Model | State"):
        """
        Get the joint coordinate velocities (DoF velocities) for the selected articulations.

        Args:
            source (Model | State): The source from which to retrieve the DoF velocities.

        Returns:
            array: The joint coordinate velocities (dtype=float).
        """
        return self._get_attribute_values("joint_qd", source)

    def set_dof_velocities(self, target: "Model | State", values, mask=None):
        """
        Set the joint coordinate velocities (DoF velocities) for the selected articulations.

        Args:
            target (Model | State): The target where to set the DoF velocities.
            values (array): The values to set (dtype=float).
            mask (array, optional): Mask of articulations in this ArticulationView (all by default).
        """
        self._set_attribute_values("joint_qd", target, values, mask=mask)

    def get_dof_forces(self, source: "Control"):
        """
        Get the joint forces (DoF forces) for the selected articulations.

        Args:
            source (Control): The source from which to retrieve the DoF forces.

        Returns:
            array: The joint forces (dtype=float).
        """
        return self._get_attribute_values("joint_f", source)

    def set_dof_forces(self, target: "Control", values, mask=None):
        """
        Set the joint forces (DoF forces) for the selected articulations.

        Args:
            target (Control): The target where to set the DoF forces.
            values (array): The values to set (dtype=float).
            mask (array, optional): Mask of articulations in this ArticulationView (all by default).
        """
        self._set_attribute_values("joint_f", target, values, mask=mask)

    # ========================================================================================
    # Utilities

    def _resolve_mask(self, mask):
        # accept 1D and 2D Boolean masks
        if isinstance(mask, wp.array):
            if mask.dtype is wp.bool and mask.ndim < 3:
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

    def get_model_articulation_mask(self, mask=None):
        """
        Get Model articulation mask from a mask in this ArticulationView.

        Args:
            mask (array): Mask of articulations in this ArticulationView (all by default).
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

    def eval_fk(self, target: Model | State, mask=None):
        """
        Evaluates forward kinematics given the joint coordinates and updates the body information.

        Args:
            target (Model | State): The target where to evaluate forward kinematics (Model or State).
            mask (array): Mask of articulations in this ArticulationView (all by default).
        """
        # translate view mask to Model articulation mask
        articulation_mask = self.get_model_articulation_mask(mask=mask)
        eval_fk(self.model, target.joint_q, target.joint_qd, target, mask=articulation_mask)

    def eval_jacobian(self, state: State, J=None, joint_S_s=None, mask=None):
        """Evaluate spatial Jacobian for articulations in this view.

        Computes the spatial Jacobian J that maps joint velocities to spatial
        velocities of each link in world frame.

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
        Jacobian and M is the block-diagonal spatial mass matrix.

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
