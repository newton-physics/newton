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

from ..sim import Control, JointType, Model, State, eval_fk

AttributeFrequency = Model.AttributeFrequency


@wp.kernel
def set_model_articulation_mask_kernel(
    view_mask: wp.array(dtype=bool),  # mask in ArticulationView
    view_to_model_map: wp.array(dtype=int),  # maps index in ArticulationView to articulation index in Model
    articulation_mask: wp.array(dtype=bool),  # output: mask of Model articulation indices
):
    """
    Set Model articulation mask from a view mask in an ArticulationView.
    """
    tid = wp.tid()
    if view_mask[tid]:
        articulation_mask[view_to_model_map[tid]] = True


@wp.kernel
def set_articulation_attribute_1d_kernel(
    view_mask: wp.array(dtype=bool),  # mask in ArticulationView
    values: Any,  # 1d array or indexedarray
    attrib: Any,  # 1d array or indexedarray
):
    i = wp.tid()
    if view_mask[i]:
        attrib[i] = values[i]


@wp.kernel
def set_articulation_attribute_2d_kernel(
    view_mask: wp.array(dtype=bool),  # mask in ArticulationView
    values: Any,  # 2d array or indexedarray
    attrib: Any,  # 2d array or indexedarray
):
    i, j = wp.tid()
    if view_mask[i]:
        attrib[i, j] = values[i, j]


@wp.kernel
def set_articulation_attribute_3d_kernel(
    view_mask: wp.array(dtype=bool),  # mask in ArticulationView
    values: Any,  # 3d array or indexedarray
    attrib: Any,  # 3d array or indexedarray
):
    i, j, k = wp.tid()
    if view_mask[i]:
        attrib[i, j, k] = values[i, j, k]


@wp.kernel
def set_articulation_attribute_4d_kernel(
    view_mask: wp.array(dtype=bool),  # mask in ArticulationView
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
            wp.overload(
                set_articulation_attribute_1d_kernel,
                {"values": src_array_type(dtype=dtype, ndim=1), "attrib": dst_array_type(dtype=dtype, ndim=1)},
            )
            wp.overload(
                set_articulation_attribute_2d_kernel,
                {"values": src_array_type(dtype=dtype, ndim=2), "attrib": dst_array_type(dtype=dtype, ndim=2)},
            )
            wp.overload(
                set_articulation_attribute_3d_kernel,
                {"values": src_array_type(dtype=dtype, ndim=3), "attrib": dst_array_type(dtype=dtype, ndim=3)},
            )
            wp.overload(
                set_articulation_attribute_4d_kernel,
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


class ArticulationView:
    """
    ArticulationView provides a flexible interface for selecting and manipulating
    subsets of articulations and their joints, links, and shapes within a Model.
    It supports pattern-based selection, inclusion/exclusion filters, and convenient
    attribute access and modification for simulation and control.

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

        articulation_ids = []
        for id, key in enumerate(model.articulation_key):
            if fnmatch(key, pattern):
                articulation_ids.append(id)

        articulation_count = len(articulation_ids)
        if articulation_count == 0:
            raise KeyError("No matching articulations")

        # FIXME: avoid/reduce this readback?
        model_articulation_start = model.articulation_start.numpy()
        model_joint_type = model.joint_type.numpy()
        model_joint_child = model.joint_child.numpy()
        model_joint_q_start = model.joint_q_start.numpy()
        model_joint_qd_start = model.joint_qd_start.numpy()
        model_shape_body = model.shape_body.numpy()

        # FIXME:
        # - this assumes homogeneous worlds with one selected articulation per world
        # - we're going to have problems if there are any bodies or joints in the "global" world

        arti_0 = articulation_ids[0]

        arti_joint_begin = model_articulation_start[arti_0]
        arti_joint_end = model_articulation_start[arti_0 + 1]  # FIXME: is this always correct?
        arti_joint_count = arti_joint_end - arti_joint_begin
        arti_link_count = arti_joint_count

        arti_joint_ids = []
        arti_joint_names = []
        arti_joint_types = []
        arti_link_ids = []
        arti_link_names = []

        def get_name_from_key(key):
            return key.split("/")[-1]

        for idx in range(arti_joint_count):
            joint_id = arti_joint_begin + idx
            arti_joint_ids.append(int(joint_id))
            arti_joint_names.append(get_name_from_key(model.joint_key[joint_id]))
            arti_joint_types.append(int(model_joint_type[joint_id]))
            link_id = model_joint_child[joint_id]
            arti_link_ids.append(int(link_id))
            arti_link_names.append(get_name_from_key(model.body_key[link_id]))

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

        selected_joint_ids = []
        selected_joint_dof_ids = []
        selected_joint_coord_ids = []
        selected_link_ids = []
        selected_shape_ids = []

        self.joint_names = []
        self.joint_dof_names = []
        self.joint_dof_counts = []
        self.joint_coord_names = []
        self.joint_coord_counts = []
        self.body_names = []
        self.shape_names = []
        self.body_shapes = []

        # populate info for selected joints and dofs
        for idx in selected_joint_indices:
            # joint
            joint_id = arti_joint_ids[idx]
            selected_joint_ids.append(joint_id)
            joint_name = get_name_from_key(model.joint_key[joint_id])
            self.joint_names.append(joint_name)
            # joint dofs
            dof_begin = model_joint_qd_start[joint_id]
            dof_end = model_joint_qd_start[joint_id + 1]
            dof_count = dof_end - dof_begin
            if dof_count == 1:
                self.joint_dof_names.append(joint_name)
                selected_joint_dof_ids.append(int(dof_begin))
            elif dof_count > 1:
                for dof in range(dof_count):
                    self.joint_dof_names.append(f"{joint_name}:{dof}")
                    selected_joint_dof_ids.append(int(dof_begin + dof))
            # joint coords
            coord_begin = model_joint_q_start[joint_id]
            coord_end = model_joint_q_start[joint_id + 1]
            coord_count = coord_end - coord_begin
            if coord_count == 1:
                self.joint_coord_names.append(joint_name)
                selected_joint_coord_ids.append(int(coord_begin))
            elif coord_count > 1:
                for coord in range(coord_count):
                    self.joint_coord_names.append(f"{joint_name}:{coord}")
                    selected_joint_coord_ids.append(int(coord_begin + coord))

        # HACK: skip any leading and trailing static shapes
        worlds_shape_start = 0
        worlds_shape_end = model.shape_count
        for i in range(model.shape_count):
            if model_shape_body[i] > -1 and model_shape_body[-i - 1] > -1:
                break
            if model_shape_body[i] == -1:
                worlds_shape_start += 1
            if model_shape_body[-i - 1] == -1:
                worlds_shape_end -= 1
        self._worlds_shape_start = worlds_shape_start
        self._worlds_shape_end = worlds_shape_end
        self._worlds_shape_count = worlds_shape_end - worlds_shape_start

        # populate info for selected links and shapes
        for idx in selected_link_indices:
            body_id = arti_link_ids[idx]
            selected_link_ids.append(body_id)
            self.body_names.append(get_name_from_key(model.body_key[body_id]))

            shape_ids = model.body_shapes[body_id]
            shape_index_list = []
            for shape_id in shape_ids:
                shape_index = len(selected_shape_ids)
                shape_index_list.append(shape_index)
                selected_shape_ids.append(shape_id)
                self.shape_names.append(get_name_from_key(model.shape_key[shape_id]))
            self.body_shapes.append(shape_index_list)

        # selected counts
        self.count = articulation_count
        self.joint_count = len(selected_joint_ids)
        self.joint_dof_count = len(selected_joint_dof_ids)
        self.joint_coord_count = len(selected_joint_coord_ids)
        self.link_count = len(selected_link_ids)
        self.shape_count = len(selected_shape_ids)

        # support custom slicing and indexing
        self._arti_joint_begin = int(arti_joint_begin)
        self._arti_joint_end = int(arti_joint_end)
        self._arti_joint_dof_begin = int(model_joint_qd_start[arti_joint_begin])
        self._arti_joint_dof_end = int(model_joint_qd_start[arti_joint_end])
        self._arti_joint_coord_begin = int(model_joint_q_start[arti_joint_begin])
        self._arti_joint_coord_end = int(model_joint_q_start[arti_joint_end])

        root_joint_type = arti_joint_types[0]
        # fixed base means that all linear and angular degrees of freedom are locked at the root
        self.is_fixed_base = root_joint_type == JointType.FIXED
        # floating base means that all linear and angular degrees of freedom are unlocked at the root
        # (though there might be constraints like distance)
        self.is_floating_base = root_joint_type in (JointType.FREE, JointType.DISTANCE)

        def is_contiguous_slice(indices):
            n = len(indices)
            if n > 1:
                for i in range(1, n):
                    if indices[i] != indices[i - 1] + 1:
                        return False
            return True

        self.joints_contiguous = is_contiguous_slice(selected_joint_ids)
        self.joint_dofs_contiguous = is_contiguous_slice(selected_joint_dof_ids)
        self.joint_coords_contiguous = is_contiguous_slice(selected_joint_coord_ids)
        self.links_contiguous = is_contiguous_slice(selected_link_ids)
        self.shapes_contiguous = is_contiguous_slice(selected_shape_ids)

        # contiguous slices or indices by attribute frequency
        #
        # FIXME: guard against empty selections
        #
        self._frequency_slices = {}
        self._frequency_indices = {}

        if len(selected_joint_ids) == 0:
            self._frequency_slices[AttributeFrequency.JOINT] = slice(0, 0)
        else:
            if self.joints_contiguous:
                self._frequency_slices[AttributeFrequency.JOINT] = slice(
                    selected_joint_ids[0], selected_joint_ids[-1] + 1
                )
            else:
                self._frequency_indices[AttributeFrequency.JOINT] = wp.array(
                    selected_joint_ids, dtype=int, device=self.device
                )

        if len(selected_joint_dof_ids) == 0:
            self._frequency_slices[AttributeFrequency.JOINT_DOF] = slice(0, 0)
        else:
            if self.joint_dofs_contiguous:
                self._frequency_slices[AttributeFrequency.JOINT_DOF] = slice(
                    selected_joint_dof_ids[0], selected_joint_dof_ids[-1] + 1
                )
            else:
                self._frequency_indices[AttributeFrequency.JOINT_DOF] = wp.array(
                    selected_joint_dof_ids, dtype=int, device=self.device
                )

        if len(selected_joint_coord_ids) == 0:
            self._frequency_slices[AttributeFrequency.JOINT_COORD] = slice(0, 0)
        else:
            if self.joint_coords_contiguous:
                self._frequency_slices[AttributeFrequency.JOINT_COORD] = slice(
                    selected_joint_coord_ids[0], selected_joint_coord_ids[-1] + 1
                )
            else:
                self._frequency_indices[AttributeFrequency.JOINT_COORD] = wp.array(
                    selected_joint_coord_ids, dtype=int, device=self.device
                )

        if len(selected_link_ids) == 0:
            self._frequency_slices[AttributeFrequency.BODY] = slice(0, 0)
        else:
            if self.links_contiguous:
                self._frequency_slices[AttributeFrequency.BODY] = slice(selected_link_ids[0], selected_link_ids[-1] + 1)
            else:
                self._frequency_indices[AttributeFrequency.BODY] = wp.array(
                    selected_link_ids, dtype=int, device=self.device
                )

        if len(selected_shape_ids) == 0:
            self._frequency_slices[AttributeFrequency.SHAPE] = slice(0, 0)
        else:
            if self.shapes_contiguous:
                # HACK: we need to skip leading static shapes
                self._frequency_slices[AttributeFrequency.SHAPE] = slice(
                    selected_shape_ids[0] - worlds_shape_start, selected_shape_ids[-1] + 1 - worlds_shape_start
                )
            else:
                self._frequency_indices[AttributeFrequency.SHAPE] = wp.array(
                    selected_shape_ids, dtype=int, device=self.device
                )

        self.articulation_indices = wp.array(articulation_ids, dtype=int, device=self.device)

        # TODO: zero-stride mask would use less memory
        self.full_mask = wp.full(articulation_count, True, dtype=bool, device=self.device)

        # create articulation mask
        self.articulation_mask = wp.zeros(model.articulation_count, dtype=bool, device=self.device)
        wp.launch(
            set_model_articulation_mask_kernel,
            dim=articulation_count,
            inputs=[self.full_mask, self.articulation_indices, self.articulation_mask],
            device=self.device,
        )

        if verbose:
            print(f"Articulation '{pattern}': {self.count}")
            print(f"  Link count:     {self.link_count} ({'strided' if self.links_contiguous else 'indexed'})")
            print(f"  Shape count:    {self.shape_count} ({'strided' if self.shapes_contiguous else 'indexed'})")
            print(f"  Joint count:    {self.joint_count} ({'strided' if self.joints_contiguous else 'indexed'})")
            print(
                f"  DOF count:      {self.joint_dof_count} ({'strided' if self.joint_dofs_contiguous else 'indexed'})"
            )
            print(f"  Fixed base?     {self.is_fixed_base}")
            print(f"  Floating base?  {self.is_floating_base}")
            print("Link names:")
            print(f"  {self.body_names}")
            print("Joint names:")
            print(f"  {self.joint_names}")
            print("Joint DOF names:")
            print(f"  {self.joint_dof_names}")

            print("Shapes:")
            for body_idx in range(self.link_count):
                body_shape_names = [self.shape_names[shape_idx] for shape_idx in self.body_shapes[body_idx]]
                print(f"  Link '{self.body_names[body_idx]}': {body_shape_names}")

    # ========================================================================================
    # Generic attribute API

    @functools.lru_cache(maxsize=None)  # noqa
    def _get_attribute_array(self, name: str, source: Model | State | Control, _slice: Slice | int | None = None):
        # support structured attributes (e.g., "shape_material_mu")
        name_components = name.split(".")
        name = name_components[0]

        # get the attribute
        attrib = getattr(source, name)

        # handle structures
        if wp.types.is_struct(attrib):
            if len(name_components) < 2:
                raise AttributeError(f"Attribute '{name}' is a structure, use '{name}.attrib' to get an attribute")
            attrib = getattr(attrib, name_components[1])

        assert isinstance(attrib, wp.array)

        frequency = self.model.get_attribute_frequency(name)

        # Custom frequencies (custom entity types) are not supported in ArticulationView
        # They represent solver-specific data that is not per-articulation
        if isinstance(frequency, str):
            raise AttributeError(
                f"Attribute '{name}' has custom frequency '{frequency}' which is not "
                f"supported by ArticulationView. Custom frequencies are for custom entity types "
                f"that are not part of articulations."
            )

        # HACK: trim leading and trailing static shapes
        if frequency == AttributeFrequency.SHAPE:
            attrib = attrib[self._worlds_shape_start : self._worlds_shape_end]

        # reshape with batch dim at front
        assert attrib.shape[0] % self.count == 0
        batched_shape = (self.count, attrib.shape[0] // self.count, *attrib.shape[1:])
        attrib = attrib.reshape(batched_shape)

        if _slice is None:
            _slice = self._frequency_slices.get(frequency)
        elif isinstance(_slice, Slice):
            _slice = _slice.get()
        elif isinstance(_slice, int):
            return attrib[:, _slice]
        else:
            raise TypeError(f"Invalid slice type: expected Slice or int, got {type(_slice)}")

        if _slice is not None:
            # create strided array
            if _slice.start == _slice.stop:
                #! workaround for empty slice until this is fixed: https://github.com/NVIDIA/warp/issues/958
                attrib = wp.array(
                    ptr=attrib.ptr,
                    dtype=attrib.dtype,
                    shape=(attrib.shape[0], 0, *attrib.shape[2:]),
                    strides=attrib.strides,
                    device=attrib.device,
                    pinned=attrib.pinned,
                    copy=False,
                )
            else:
                attrib = attrib[:, _slice]
        else:
            # create indexed array + contiguous staging array
            _indices = self._frequency_indices.get(frequency)
            if _indices is None:
                raise AttributeError(f"Unable to determine the frequency of attribute '{name}'")
            attrib = wp.indexedarray(attrib, [None, _indices])
            attrib._staging_array = wp.empty_like(attrib)

        return attrib

    def _get_attribute_values(self, name: str, source: Model | State | Control, _slice: slice | None = None):
        attrib = self._get_attribute_array(name, source, _slice=_slice)
        if hasattr(attrib, "_staging_array"):
            wp.copy(attrib._staging_array, attrib)
            return attrib._staging_array
        else:
            return attrib

    # def _set_attribute_values(self, attrib, values, mask=None):
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
            if not isinstance(mask, wp.array):
                mask = wp.array(mask, dtype=bool, shape=(self.count,), device=self.device, copy=False)
            assert mask.shape == (self.count,)

        # launch appropriate kernel based on attribute dimensionality
        # TODO: cache concrete overload per attribute?
        if attrib.ndim == 1:
            wp.launch(
                set_articulation_attribute_1d_kernel,
                dim=attrib.shape,
                inputs=[mask, values, attrib],
                device=self.device,
            )
        elif attrib.ndim == 2:
            wp.launch(
                set_articulation_attribute_2d_kernel,
                dim=attrib.shape,
                inputs=[mask, values, attrib],
                device=self.device,
            )
        elif attrib.ndim == 3:
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
            attrib_slice = Slice(self._arti_joint_coord_begin, self._arti_joint_coord_begin + 7)
            attrib = self._get_attribute_values("joint_q", source, _slice=attrib_slice)
        else:
            attrib_slice = self._arti_joint_begin
            attrib = self._get_attribute_values("joint_X_p", self.model, _slice=attrib_slice)

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
            attrib_slice = Slice(self._arti_joint_coord_begin, self._arti_joint_coord_begin + 7)
            self._set_attribute_values("joint_q", target, values, mask=mask, _slice=attrib_slice)
        else:
            attrib_slice = self._arti_joint_begin
            self._set_attribute_values("joint_X_p", self.model, values, mask=mask, _slice=attrib_slice)

    def get_root_velocities(self, source: Model | State):
        """
        Get the root velocities of the articulations.

        Args:
            source (Model | State): Where to get the root velocities (Model or State).

        Returns:
            array: The root velocities (dtype=wp.spatial_vector).
        """
        if self.is_floating_base:
            attrib_slice = Slice(self._arti_joint_dof_begin, self._arti_joint_dof_begin + 6)
            attrib = self._get_attribute_values("joint_qd", source, _slice=attrib_slice)
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
            attrib_slice = Slice(self._arti_joint_dof_begin, self._arti_joint_dof_begin + 6)
            self._set_attribute_values("joint_qd", target, values, mask=mask, _slice=attrib_slice)
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

    def get_model_articulation_mask(self, mask=None):
        """
        Get Model articulation mask from a mask in this ArticulationView.

        Args:
            mask (array): Mask of articulations in this ArticulationView (all by default).
        """
        if mask is None:
            return self.articulation_mask
        else:
            if not isinstance(mask, wp.array):
                mask = wp.array(mask, dtype=bool, device=self.device, copy=False)
            assert mask.shape == (self.count,)
            articulation_mask = wp.zeros(self.model.articulation_count, dtype=bool, device=self.device)
            wp.launch(
                set_model_articulation_mask_kernel,
                dim=mask.size,
                inputs=[mask, self.articulation_indices, articulation_mask],
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
