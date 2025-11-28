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

"""
KAMINO: Discrete Contact Containers & Operations
"""

from __future__ import annotations

import warp as wp
from warp.context import Devicelike

from newton._src.solvers.kamino.core.types import int32, mat33f, mat63f, vec2f, vec4f

###
# Module interface
###

__all__ = ["Contacts", "ContactsData"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###

MAX_WORLD_CONTACTS_DEFAULT = int(32)
"""The default maximum number of contacts per world."""

W_I = wp.constant(mat63f([1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]))
"""The identity wrench matrix, used to initialize the contact wrench matrices."""


###
# Containers
###


class ContactsData:
    """
    An SoA-based container to hold time-varying contact data of a set of contact elements.

    This container is intended as the final output of collision detectors and as input to solvers.
    """

    def __init__(self):
        self.num_model_max_contacts: int = 0
        """
        The maximum number of contacts allocated across all worlds.\n
        This is cached on the host-side for managing data allocations and setting thread sizes in kernels.
        """

        self.num_world_max_contacts: list[int] = [0]
        """
        The maximum number of contacts allocated per world.\n
        This is cached on the host-side for managing data allocations and setting thread sizes in kernels.
        """

        self.model_max_contacts: wp.array(dtype=int32) | None = None
        """
        The number of active contacts per model.\n
        Shape of ``(1,)`` and type :class:`int32`.
        """

        self.model_num_contacts: wp.array(dtype=int32) | None = None
        """
        The number of active contacts per model.\n
        Shape of ``(1,)`` and type :class:`int32`.
        """

        self.world_max_contacts: wp.array(dtype=int32) | None = None
        """The maximum number of contacts per world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.world_num_contacts: wp.array(dtype=int32) | None = None
        """
        The number of active contacts per world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.wid: wp.array(dtype=int32) | None = None
        """
        The world index of each contact.\n
        Shape of ``(num_model_max_contacts,)`` and type :class:`int32`.
        """

        self.cid: wp.array(dtype=int32) | None = None
        """
        The contact index of each contact w.r.t its world.\n
        Shape of ``(num_model_max_contacts,)`` and type :class:`int32`.
        """

        self.body_A: wp.array(dtype=vec4f) | None = None
        """
        The position of each contact on corresponding body A and the body index (xyz: position, w: bid).\n
        Shape of ``(num_model_max_contacts,)`` and type :class:`vec4f`.
        """

        self.body_B: wp.array(dtype=vec4f) | None = None
        """
        The position of each contact on corresponding body B and the body index (xyz: position, w: bid).\n
        Shape of ``(num_model_max_contacts,)`` and type :class:`vec4f`.
        """

        self.gapfunc: wp.array(dtype=vec4f) | None = None
        """
        The gap-function signed-distance of each contact (xyz: normal, w: penetration).\n
        Shape of ``(num_model_max_contacts,)`` and type :class:`vec4f`.
        """

        self.frame: wp.array(dtype=mat33f) | None = None
        """
        The contact frame of each contact (3x3 rotation matrix).\n
        Shape of ``(num_model_max_contacts,)`` and type :class:`mat33f`.
        """

        self.material: wp.array(dtype=vec2f) | None = None
        """
        The material properties of each contact (0: friction, 1: restitution).\n
        Shape of ``(num_model_max_contacts,)`` and type :class:`vec2f`.
        """


###
# Interfaces
###


class Contacts:
    """
    A container to hold and manage time-varying contacts.
    """

    def __init__(
        self,
        capacity: int | list[int] | None = None,
        default_max_contacts: int | None = None,
        device: Devicelike = None,
    ):
        # The device on which to allocate the contacts data
        self.device = device

        # Set the default maximum number of contacts per world
        self._default_max_world_contacts: int = MAX_WORLD_CONTACTS_DEFAULT
        if default_max_contacts is not None:
            self._default_max_world_contacts = default_max_contacts

        # Contacts data container
        self._data: ContactsData = ContactsData()

        # Perofrm memory allocation if max_contacts is specified
        if capacity is not None:
            self.allocate(capacity=capacity, device=device)

    @property
    def default_max_world_contacts(self) -> int:
        """
        The default maximum number of contacts per world.
        This value is used when the capacity at allocation-time is unspecified or equals 0.
        """
        return self._default_max_world_contacts

    @default_max_world_contacts.setter
    def default_max_world_contacts(self, max_contacts: int):
        """
        Sets the default maximum number of contacts per world.

        Args:
            max_contacts (int): The maximum number of contacts per world.
        """
        if max_contacts < 0:
            raise ValueError("Contacts: max_contacts must be a non-negative integer")
        self._default_max_world_contacts = max_contacts

    @property
    def data(self) -> ContactsData:
        """
        Returns the managed contacts data container.
        """
        return self._data

    @property
    def num_model_max_contacts(self) -> int:
        """
        The maximum number of contacts allocated across all worlds.
        """
        return self._data.num_model_max_contacts

    @property
    def num_world_max_contacts(self) -> list[int]:
        """
        The maximum number of contacts allocated per world.
        """
        return self._data.num_world_max_contacts

    @property
    def model_max_contacts(self) -> wp.array:
        """
        The total number of maximum contacts for the model. Shape of ``(1,)`` and type :class:`int32`.
        """
        return self._data.model_max_contacts

    @property
    def model_num_contacts(self) -> wp.array:
        """
        The total number of active contacts for the model. Shape of ``(1,)`` and type :class:`int32`.
        """
        return self._data.model_num_contacts

    @property
    def world_max_contacts(self) -> wp.array:
        """
        The total number of maximum contacts for the model. Shape of ``(1,)`` and type :class:`int32`.
        """
        return self._data.world_max_contacts

    @property
    def world_num_contacts(self) -> wp.array:
        """
        The total number of active contacts for the model. Shape of ``(1,)`` and type :class:`int32`.
        """
        return self._data.world_num_contacts

    @property
    def wid(self) -> wp.array:
        """
        The world index of each contact. Shape of ``(nc,)`` and type :class:`int32`.
        """
        return self._data.wid

    @property
    def cid(self) -> wp.array:
        """
        The contact index of each contact w.r.t its world. Shape of ``(nc,)`` and type :class:`int32`.
        """
        return self._data.cid

    @property
    def body_A(self) -> wp.array:
        """
        The position of the contact on body A and the body index (xyz: position, w: bid). Shape of ``(nc,)`` and type :class:`vec4f`.
        """
        return self._data.body_A

    @property
    def body_B(self) -> wp.array:
        """
        The position of the contact on body B and the body index (xyz: position, w: bid). Shape of ``(nc,)`` and type :class:`vec4f`.
        """
        return self._data.body_B

    @property
    def gapfunc(self) -> wp.array:
        """
        The contact gap-function signed-distance (xyz: normal, w: penetration). Shape of ``(nc,)`` and type :class:`vec4f`.
        """
        return self._data.gapfunc

    @property
    def frame(self) -> wp.array:
        """
        The contact frame (3x3 rotation matrix). Shape of ``(nc,)`` and type :class:`mat33f`.
        """
        return self._data.frame

    @property
    def material(self) -> wp.array:
        """
        The contact material properties (0: friction, 1: restitution). Shape of ``(nc,)`` and type :class:`vec2f`.
        """
        return self._data.material

    def allocate(self, capacity: int | list[int], device: Devicelike = None):
        # The memory allocation requires the total number of contacts (over multiple worlds)
        # as well as the contacts capacities for each world. Corresponding sizes are defaulted to 0 (empty).
        model_max_contacts = 0
        world_max_contacts = [0]

        # If the capacity is a list, this means we are allocating for multiple worlds
        if isinstance(capacity, list):
            if len(capacity) == 0:
                raise ValueError("Contacts: capacity cannot be an empty list")
            for i in range(len(capacity)):
                if capacity[i] < 0:
                    raise ValueError(f"Contacts: capacity[{i}] must be a non-negative integer")
                if capacity[i] == 0:
                    capacity[i] = self._default_max_world_contacts
            model_max_contacts = sum(capacity)
            world_max_contacts = capacity

        # If the capacity is a single integer, this means we are allocating for a single world
        elif isinstance(capacity, int):
            if capacity < 0:
                raise ValueError("Contacts: capacity must be a non-negative integer")
            if capacity == 0:
                capacity = self._default_max_world_contacts
            model_max_contacts = capacity
            world_max_contacts = [capacity]

        else:
            raise TypeError("Contacts: capacity must be an integer or a list of integers")

        # Override the device if specified
        if device is not None:
            self.device = device

        # Allocate the contacts data on the specified device
        with wp.ScopedDevice(self.device):
            self._data.num_model_max_contacts = model_max_contacts
            self._data.num_world_max_contacts = world_max_contacts
            self._data.model_max_contacts = wp.array([model_max_contacts], dtype=int32)
            self._data.model_num_contacts = wp.zeros(shape=1, dtype=int32)
            self._data.world_max_contacts = wp.array(world_max_contacts, dtype=int32)
            self._data.world_num_contacts = wp.zeros(shape=len(world_max_contacts), dtype=int32)
            self._data.wid = wp.zeros(shape=self.num_model_max_contacts, dtype=int32)
            self._data.cid = wp.zeros(shape=self.num_model_max_contacts, dtype=int32)
            self._data.body_A = wp.zeros(shape=self.num_model_max_contacts, dtype=vec4f)
            self._data.body_B = wp.zeros(shape=self.num_model_max_contacts, dtype=vec4f)
            self._data.gapfunc = wp.zeros(shape=self.num_model_max_contacts, dtype=vec4f)
            self._data.frame = wp.zeros(shape=self.num_model_max_contacts, dtype=mat33f)
            self._data.material = wp.zeros(shape=self.num_model_max_contacts, dtype=vec2f)

    def clear(self):
        """
        Clears the active contacts count.
        """
        self._data.model_num_contacts.zero_()
        self._data.world_num_contacts.zero_()

    def zero(self):
        """
        Resets the contact data to zero.
        """
        self._data.model_num_contacts.zero_()
        self._data.world_num_contacts.zero_()
        self._data.wid.zero_()
        self._data.cid.zero_()
        self._data.body_A.zero_()
        self._data.body_B.zero_()
        self._data.gapfunc.zero_()
        self._data.frame.zero_()
        self._data.material.zero_()
