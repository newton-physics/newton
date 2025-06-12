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

import newton.sim
from newton import Control, Model, State
from newton.sim import JOINT_DISTANCE, JOINT_FIXED, JOINT_FREE, get_joint_dof_count


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
def set_articulation_attribute_2d_kernel(
    view_mask: wp.array(dtype=bool),  # mask in ArticulationView
    values: wp.array2d(dtype=Any),
    attrib: wp.array2d(dtype=Any),
):
    i, j = wp.tid()
    if view_mask[i]:
        attrib[i, j] = values[i, j]


@wp.kernel
def set_articulation_attribute_3d_kernel(
    view_mask: wp.array(dtype=bool),  # mask in ArticulationView
    values: wp.array3d(dtype=Any),
    attrib: wp.array3d(dtype=Any),
):
    i, j, k = wp.tid()
    if view_mask[i]:
        attrib[i, j, k] = values[i, j, k]


@wp.kernel
def set_articulation_attribute_4d_kernel(
    view_mask: wp.array(dtype=bool),  # mask in ArticulationView
    values: wp.array4d(dtype=Any),
    attrib: wp.array4d(dtype=Any),
):
    i, j, k, l = wp.tid()
    if view_mask[i]:
        attrib[i, j, k, l] = values[i, j, k, l]


# explicit overloads to avoid module reloading
wp.overload(
    set_articulation_attribute_2d_kernel, {"values": wp.array2d(dtype=float), "attrib": wp.array2d(dtype=float)}
)
wp.overload(set_articulation_attribute_2d_kernel, {"values": wp.array2d(dtype=int), "attrib": wp.array2d(dtype=int)})
wp.overload(
    set_articulation_attribute_2d_kernel,
    {"values": wp.array2d(dtype=wp.transform), "attrib": wp.array2d(dtype=wp.transform)},
)
wp.overload(
    set_articulation_attribute_2d_kernel,
    {"values": wp.array2d(dtype=wp.spatial_vector), "attrib": wp.array2d(dtype=wp.spatial_vector)},
)

wp.overload(
    set_articulation_attribute_3d_kernel, {"values": wp.array3d(dtype=float), "attrib": wp.array3d(dtype=float)}
)
wp.overload(set_articulation_attribute_3d_kernel, {"values": wp.array3d(dtype=int), "attrib": wp.array3d(dtype=int)})

wp.overload(
    set_articulation_attribute_4d_kernel, {"values": wp.array4d(dtype=float), "attrib": wp.array4d(dtype=float)}
)
wp.overload(set_articulation_attribute_4d_kernel, {"values": wp.array4d(dtype=int), "attrib": wp.array4d(dtype=int)})


class ArticulationView:
    def __init__(self, model: Model, pattern: str, verbose: bool | None = None):
        self.model = model
        self.device = model.device

        if verbose is None:
            verbose = wp.config.verbose

        articulation_ids = []
        for id, key in enumerate(model.articulation_key):
            if fnmatch(key, pattern):
                articulation_ids.append(id)

        count = len(articulation_ids)
        if count == 0:
            raise KeyError("No matching articulations")

        # FIXME: avoid this readback?
        articulation_start = model.articulation_start.numpy()
        joint_type = model.joint_type.numpy()
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_axis_start = model.joint_axis_start.numpy()
        joint_axis_dim = model.joint_axis_dim.numpy()
        joint_q_start = model.joint_q_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()

        # FIXME:
        # - this assumes homogeneous envs with one selected articulation per env
        # - we're going to have problems if there are any bodies or joints in the "global" env

        arti_0 = articulation_ids[0]

        joint_begin = articulation_start[arti_0]
        joint_end = articulation_start[arti_0 + 1]  # FIXME: is this always correct?
        joint_last = joint_end - 1

        links = {}
        for joint_id in range(joint_begin, joint_end):
            if joint_parent[joint_id] != -1:
                links[int(joint_parent[joint_id])] = None
            if joint_child[joint_id] != -1:
                links[int(joint_child[joint_id])] = None

        links = sorted(links.keys())

        # print stuff for debugging
        if verbose:
            print(f"num_joints: {joint_end - joint_begin}")
            for joint_id in range(joint_begin, joint_end):
                joint_name = model.joint_key[joint_id]
                print(f"  joint {joint_name}:")
                print(f"    bodies: {joint_parent[joint_id]} -> {joint_child[joint_id]}")
                print(f"    axis_start: {joint_axis_start[joint_id]}")
                print(f"    axis_dim: {joint_axis_dim[joint_id]}")
            num_links = len(links)
            print(f"num_links: {num_links}, {links}")
            for body_id in links:
                print(f"  {model.body_key[body_id]}")

        joint_coord_begin = joint_q_start[joint_begin]
        joint_coord_end = joint_q_start[joint_end]
        joint_dof_begin = joint_qd_start[joint_begin]
        joint_dof_end = joint_qd_start[joint_end]
        joint_axis_begin = joint_axis_start[joint_begin]
        joint_axis_end = joint_axis_start[joint_last] + joint_axis_dim[joint_last][0] + joint_axis_dim[joint_last][1]
        body_begin = links[0]
        body_end = links[-1] + 1

        self.articulation_indices = wp.array(articulation_ids, dtype=int, device=self.device)

        # TODO: zero-stride mask would use less memory
        self.full_mask = wp.full(count, True, dtype=bool, device=self.device)

        # create articulation mask
        self.articulation_mask = wp.zeros(model.articulation_count, dtype=bool, device=self.device)
        wp.launch(
            set_model_articulation_mask_kernel,
            dim=count,
            inputs=[self.full_mask, self.articulation_indices, self.articulation_mask],
            device=self.device,
        )

        # set some counting properties
        self.count = count
        self.link_count = len(links)
        self.joint_count = joint_end - joint_begin
        self.joint_coord_count = joint_coord_end - joint_coord_begin
        self.joint_dof_count = joint_dof_end - joint_dof_begin
        self.joint_axis_count = joint_axis_end - joint_axis_begin

        # root joint properties
        self.root_type = int(joint_type[joint_begin])
        self.root_axis_count = int(joint_axis_dim[joint_begin][0] + joint_axis_dim[joint_begin][1])
        self.root_dof_count, self.root_coord_count = get_joint_dof_count(self.root_type, self.root_axis_count)

        self.is_fixed_base = self.root_type == JOINT_FIXED
        # floating base means that all linear and angular degrees of freedom are unlocked
        # (though there might be constraints like distance)
        self.is_floating_base = self.root_type == JOINT_FREE or self.root_type == JOINT_DISTANCE

        self.joint_names = []
        self.joint_coord_names = []
        self.joint_dof_names = []
        self.joint_axis_names = []
        self.body_names = []

        def get_name_from_key(key):
            return key.split("/")[-1]

        for joint_id in range(joint_begin, joint_end):
            joint_name = get_name_from_key(model.joint_key[joint_id])
            self.joint_names.append(joint_name)
            num_coords = joint_q_start[joint_id + 1] - joint_q_start[joint_id]
            if num_coords == 1:
                self.joint_coord_names.append(joint_name)
            elif num_coords > 1:
                for coord in range(num_coords):
                    self.joint_coord_names.append(f"{joint_name}:{coord}")
            num_dofs = joint_qd_start[joint_id + 1] - joint_qd_start[joint_id]
            if num_dofs == 1:
                self.joint_dof_names.append(joint_name)
            elif num_dofs > 1:
                for dof in range(num_dofs):
                    self.joint_dof_names.append(f"{joint_name}:{dof}")
            num_axes = joint_axis_dim[joint_id][0] + joint_axis_dim[joint_id][1]
            if num_axes == 1:
                self.joint_axis_names.append(joint_name)
            elif num_axes > 1:
                for axis in range(num_axes):
                    self.joint_axis_names.append(f"{joint_name}:{axis}")

        for body_id in range(body_begin, body_end):
            self.body_names.append(get_name_from_key(model.body_key[body_id]))

        if verbose:
            print("Link names:")
            print(self.body_names)
            print("Joint names:")
            print(self.joint_names)
            print("Joint axis names:")
            print(self.joint_axis_names)
            # print("Joint coord names:")
            # print(self.joint_coord_names)
            # print("Joint dof names:")
            # print(self.joint_dof_names)

        # slices by attribute frequency
        self._slices = {
            "joint": slice(int(joint_begin), int(joint_end)),
            "joint_coord": slice(int(joint_coord_begin), int(joint_coord_end)),
            "joint_dof": slice(int(joint_dof_begin), int(joint_dof_end)),
            "joint_axis": slice(int(joint_axis_begin), int(joint_axis_end)),
            "body": slice(int(body_begin), int(body_end)),
        }

    # ========================================================================================
    # Generic attribute API

    @functools.lru_cache(maxsize=None)  # noqa
    def _get_cached_attribute(self, name: str, source: Model | State | Control):
        # get the attribute array
        attrib = getattr(source, name)
        assert isinstance(attrib, wp.array)

        # reshape with batch dim at front
        assert attrib.shape[0] % self.count == 0
        batched_shape = (self.count, attrib.shape[0] // self.count, *attrib.shape[1:])

        # get attribute slice
        frequency = self.model.get_attribute_frequency(name)
        attrib_slice = self._slices[frequency]

        # create strided array
        attrib = attrib.reshape(batched_shape)
        attrib = attrib[:, attrib_slice]

        return attrib

    def _set_attribute_values(self, attrib, values, mask=None):
        if not is_array(values):
            values = wp.array(values, dtype=attrib.dtype, shape=attrib.shape, device=self.device, copy=False)
        assert values.shape == attrib.shape
        assert values.dtype == attrib.dtype

        # early out for in-place modifications
        if values.ptr == attrib.ptr:
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
        if attrib.ndim == 2:
            wp.launch(set_articulation_attribute_2d_kernel, dim=attrib.shape, inputs=[mask, values, attrib])
        elif attrib.ndim == 3:
            wp.launch(set_articulation_attribute_3d_kernel, dim=attrib.shape, inputs=[mask, values, attrib])
        elif attrib.ndim == 4:
            wp.launch(set_articulation_attribute_4d_kernel, dim=attrib.shape, inputs=[mask, values, attrib])
        else:
            raise NotImplementedError(f"Unsupported attribute with ndim={attrib.ndim}")

    def get_attribute(self, name: str, source: Model | State | Control):
        return self._get_cached_attribute(name, source)

    def set_attribute(self, name: str, target: Model | State | Control, values, mask=None):
        attrib = self._get_cached_attribute(name, target)
        self._set_attribute_values(attrib, values, mask=mask)

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
            attrib = self._get_cached_attribute("joint_q", source)[:, :7]
        else:
            attrib = self._get_cached_attribute("joint_X_p", self.model)[:, 0]

        if attrib.dtype is wp.transform:
            return attrib
        else:
            return wp.array(attrib, dtype=wp.transform, device=self.device, copy=False)

    def set_root_transforms(self, target: Model | State, values: wp.array, mask=None):
        """
        Set the root transforms of the articulations.
        Call `eval_fk()` to apply changes to all articulation links.

        Args:
            target (Model | State): Where to set the root transforms (Model or State).
            values (array): The root transforms to set (dtype=wp.transform).
            mask (array): Mask of articulations in this ArticulationView (all by default).
        """
        if self.is_floating_base:
            attrib = self._get_cached_attribute("joint_q", target)[:, :7]
        else:
            attrib = self._get_cached_attribute("joint_X_p", self.model)[:, 0]

        self._set_attribute_values(attrib, values, mask=mask)

    def get_root_velocities(self, source: Model | State):
        """
        Get the root velocities of the articulations.

        Args:
            source (Model | State): Where to get the root velocities (Model or State).

        Returns:
            array: The root velocities (dtype=wp.spatial_vector).
        """
        if self.is_floating_base:
            attrib = self._get_cached_attribute("joint_qd", source)[:, :6]
        else:
            # Non-floating articulations have no root velocities.
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
            attrib = self._get_cached_attribute("joint_qd", target)[:, :6]
        else:
            return  # no-op

        self._set_attribute_values(attrib, values, mask=mask)

    def get_root_armatures(self, source: Model | State):
        """
        Get the root joint armatures of the articulations.

        Args:
            source (Model | State): Where to get the armatures (Model or State).

        Returns:
            array: The root armatures (dtype=wp.float32).
        """
        if self.is_floating_base:
            return self._get_cached_attribute("joint_armature", source)[:, :6]
        else:
            # Fixed-base articulations have no root armature.
            # For articulations that are neither fixed nor floating, root joint armatures
            # can be accessed using `get_dof_armatures()`
            return None

    def set_root_armatures(self, target: Model | State, values: wp.array, mask=None):
        """
        Set the root joint armatures of the articulations.

        Args:
            target (Model | State): Where to set the root armatures (Model or State).
            values (array): The root armatures to set (dtype=wp.float32).
            mask (array): Mask of articulations in this ArticulationView (all by default).
        """
        if self.is_floating_base:
            attrib = self._get_cached_attribute("joint_armature", target)[:, :6]
        else:
            return  # no-op

        self._set_attribute_values(attrib, values, mask=mask)

    def get_link_transforms(self, source: Model | State):
        return self._get_cached_attribute("body_q", source)

    def get_link_velocities(self, source: Model | State):
        return self._get_cached_attribute("body_qd", source)

    def get_dof_positions(self, source: Model | State):
        if self.is_floating_base:
            return self._get_cached_attribute("joint_q", source)[:, 7:]
        else:
            return self._get_cached_attribute("joint_q", source)

    def set_dof_positions(self, target: Model | State, values, mask=None):
        if self.is_floating_base:
            attrib = self._get_cached_attribute("joint_q", target)[:, 7:]
        else:
            attrib = self._get_cached_attribute("joint_q", target)

        self._set_attribute_values(attrib, values, mask=mask)

    def get_dof_velocities(self, source: Model | State):
        if self.is_floating_base:
            return self._get_cached_attribute("joint_qd", source)[:, 6:]
        else:
            return self._get_cached_attribute("joint_qd", source)

    def set_dof_velocities(self, target: Model | State, values, mask=None):
        if self.is_floating_base:
            attrib = self._get_cached_attribute("joint_qd", target)[:, 6:]
        else:
            attrib = self._get_cached_attribute("joint_qd", target)

        self._set_attribute_values(attrib, values, mask=mask)

    def get_dof_forces(self, source: Control):
        if self.is_floating_base:
            return self._get_cached_attribute("joint_f", source)[:, 6:]
        else:
            return self._get_cached_attribute("joint_f", source)

    def set_dof_forces(self, target: Control, values, mask=None):
        if self.is_floating_base:
            attrib = self._get_cached_attribute("joint_f", target)[:, 6:]
        else:
            attrib = self._get_cached_attribute("joint_f", target)

        self._set_attribute_values(attrib, values, mask=mask)

    def get_dof_armatures(self, source: Model | State):
        if self.is_floating_base:
            return self._get_cached_attribute("joint_armature", source)[:, 6:]
        else:
            return self._get_cached_attribute("joint_armature", source)

    def set_dof_armatures(self, target: Model | State, values, mask=None):
        if self.is_floating_base:
            attrib = self._get_cached_attribute("joint_armature", target)[:, 6:]
        else:
            attrib = self._get_cached_attribute("joint_armature", target)

        self._set_attribute_values(attrib, values, mask=mask)

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
            )
            return articulation_mask

    def eval_fk(self, target: Model | State, mask=None):
        """
        Evaluates forward kinematics given the joint coordinates and updates the body information.

        Args:
            mask (array): Mask of articulations in this ArticulationView (all by default).
        """
        # translate view mask to Model articulation mask
        articulation_mask = self.get_model_articulation_mask(mask=mask)
        newton.sim.eval_fk(self.model, target.joint_q, target.joint_qd, target, mask=articulation_mask)
