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
Defines the representation of discrete contacts in Kamino.

This module provides a set of data types and operations that define
the data layout and conventions used to represent discrete contacts
within the Kamino solver. It includes:

- The :class:`ContactsData` dataclass defining the structure of contact data.

- The :class:`ContactMode` enumeration defining the discrete contact modes
and a member function that generates Warp functions to compute the contact
mode based on local contact velocities.

- Utility functions for constructing contact-local coordinate frames
supporting both a Z-up and X-up convention.

- The :class:`Contacts` container which provides a high-level interface to
  manage contact data, including allocations, access, and common operations,
  and fundamentally serves as the primary output of collision detectors
  as well as a cache of contact data to warm-start physics solvers.
"""

from dataclasses import dataclass, field
from enum import IntEnum

import warp as wp
from warp.context import Devicelike

from ..core.math import COS_PI_6, UNIT_X, UNIT_Y
from ..core.types import float32, int32, mat33f, quatf, vec2f, vec2i, vec3f, vec4f
from ..utils import logger as msg

###
# Module interface
###

__all__ = [
    "DEFAULT_GEOM_PAIR_MAX_CONTACTS",
    "DEFAULT_WORLD_MAX_CONTACTS",
    "ContactMode",
    "Contacts",
    "ContactsData",
    "make_contact_frame_xnorm",
    "make_contact_frame_znorm",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###

DEFAULT_WORLD_MAX_CONTACTS: int = 128
"""
The global default for maximum number of contacts per world.\n
Used when allocating contact data without a specified capacity.\n
Set to `128`.
"""

DEFAULT_GEOM_PAIR_MAX_CONTACTS: int = 8
"""
The global default for maximum number of contacts per geom-pair.\n
Used when allocating contact data without a specified capacity.\n
Ignored for mesh-based collisions.\n
Set to `8` (with box-box collisions being a prototypical case).
"""

DEFAULT_GEOM_PAIR_CONTACT_MARGIN: float = 1e-5
"""
The global default for the per-geom-pair contact margin.\n
Each effective contact margin for a geom-pair is computed as:
``max(DEFAULT_GEOM_PAIR_CONTACT_MARGIN, max(geom1_margin, geom2_margin))``\n
Set to `1e-5`.
"""


###
# Types
###


class ContactMode(IntEnum):
    """An enumeration of discrete-contact modes."""

    ###
    # Contact Modes
    ###

    INACTIVE = -1
    """Indicates that contact is inactive (i.e. separated)."""

    OPENING = 0
    """Indicates that contact was previously closed (i.e. STICKING or SLIDING) and is now opening."""

    STICKING = 1
    """Indicates that contact is persisting (i.e. closed) without relative tangential motion."""

    SLIDING = 2
    """Indicates that contact is persisting (i.e. closed) with relative tangential motion."""

    ###
    # Utility Constants
    ###

    DEFAULT_VN_MIN = 1e-3
    """The minimum normal velocity threshold for determining contact open or closed modes."""

    DEFAULT_VT_MIN = 1e-3
    """The minimum tangential velocity threshold for determining contact stick or slip modes."""

    ###
    # Utility Functions
    ###

    @staticmethod
    def make_compute_mode_func(vn_tol: float = DEFAULT_VN_MIN, vt_tol: float = DEFAULT_VT_MIN):
        # Ensure tolerances are non-negative
        if vn_tol < 0.0:
            raise ValueError("ContactMode: vn_tol must be non-negative")
        if vt_tol < 0.0:
            raise ValueError("ContactMode: vt_tol must be non-negative")

        # Generate the compute mode function based on the specified tolerances
        @wp.func
        def _compute_mode(v: vec3f) -> int32:
            """
            Computes the discrete contact mode based on the contact velocity.

            Args:
                v (vec3f): The contact velocity expressed in the local contact frame.

            Returns:
                int32: The discrete contact mode as an integer value.
            """
            # Decompose the velocity into the normal and tangential components
            v_N = v.z
            v_T_norm = wp.sqrt(v.x * v.x + v.y * v.y)

            # Determine the contact mode
            mode = int32(ContactMode.OPENING)
            if v_N <= float32(vn_tol):
                if v_T_norm <= float32(vt_tol):
                    mode = ContactMode.STICKING
                else:
                    mode = ContactMode.SLIDING

            # Return the resulting contact mode integer
            return mode

        # Return the generated compute mode function
        return _compute_mode


@dataclass
class ContactsData:
    """
    An SoA-based container to hold time-varying contact data of a set of contact elements.

    This container is intended as the final output of collision detectors and as input to solvers.
    """

    @staticmethod
    def _default_num_world_max_contacts() -> list[int]:
        return [0]

    model_max_contacts_host: int = 0
    """
    Host-side cache of the maximum number of contacts allocated across all worlds.\n
    Intended for managing data allocations and setting thread sizes in kernels.
    """

    world_max_contacts_host: list[int] = field(default_factory=_default_num_world_max_contacts)
    """
    Host-side cache of the maximum number of contacts allocated per world.\n
    Intended for managing data allocations and setting thread sizes in kernels.
    """

    model_max_contacts: wp.array | None = None
    """
    The number of contacts pre-allocated across all worlds in the model.\n
    Shape of ``(1,)`` and type :class:`int32`.
    """

    model_active_contacts: wp.array | None = None
    """
    The number of active contacts detected across all worlds in the model.\n
    Shape of ``(1,)`` and type :class:`int32`.
    """

    world_max_contacts: wp.array | None = None
    """
    The maximum number of contacts pre-allocated for each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    world_active_contacts: wp.array | None = None
    """
    The number of active contacts detected in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    wid: wp.array | None = None
    """
    The world index of each active contact.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`int32`.
    """

    cid: wp.array | None = None
    """
    The contact index of each active contact w.r.t its world.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`int32`.
    """

    gid_AB: wp.array | None = None
    """
    The geometry indices of the geometry-pair AB associated with each active contact.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`vec2i`.
    """

    bid_AB: wp.array | None = None
    """
    The body indices of the body-pair AB associated with each active contact.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`vec2i`.
    """

    position_A: wp.array | None = None
    """
    The position of each active contact on the associated body-A in world coordinates.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`vec3f`.
    """

    position_B: wp.array | None = None
    """
    The position of each active contact on the associated body-B in world coordinates.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`vec3f`.
    """

    gapfunc: wp.array | None = None
    """
    The gap-function (i.e. signed-distance) of each active contact with format `(xyz: normal, w: penetration)`.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`vec4f`.
    """

    frame: wp.array | None = None
    """
    The coordinate frame of each active contact as a rotation quaternion w.r.t the world.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`quatf`.
    """

    material: wp.array | None = None
    """
    The material properties of each active contact with format `(0: friction, 1: restitution)`.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`vec2f`.
    """

    key: wp.array | None = None
    """
    Integer key uniquely identifying each active contact.\n
    The per-contact key assignment is implementation-dependent, but is typically
    computed from the A/B geom-pair index as well as additional information such as:
    - the triangle index
    - shape-specific topological data
    - contact index w.r.t the geom-pair\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`uint64`.
    """

    reaction: wp.array | None = None
    """
    The 3D contact reaction (force/impulse) expressed in the respective local contact frame.\n
    This is to be set by solvers at each step, and also facilitates contact visualization and warm-starting.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`vec3f`.
    """

    velocity: wp.array | None = None
    """
    The 3D contact velocity expressed in the respective local contact frame.\n
    This is to be set by solvers at each step, and also facilitates contact visualization and warm-starting.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`vec3f`.
    """

    mode: wp.array | None = None
    """
    The discrete contact mode expressed as an integer value.\n
    The possible values correspond to those of the :class:`ContactMode`.\n
    This is to be set by solvers at each step, and also facilitates contact visualization and warm-starting.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`int32`.
    """

    def clear(self):
        """
        Clears the count of active contacts.
        """
        self.model_active_contacts.zero_()
        self.world_active_contacts.zero_()

    def reset(self):
        """
        Clears the count of active contacts and resets contact data
        to sentinel values, indicating an empty set of contacts.
        """
        self.clear()
        self.wid.fill_(-1)
        self.cid.fill_(-1)
        self.gid_AB.fill_(vec2i(-1, -1))
        self.bid_AB.fill_(vec2i(-1, -1))
        self.mode.fill_(ContactMode.INACTIVE)
        self.reaction.zero_()
        self.velocity.zero_()


###
# Functions
###


@wp.func
def make_contact_frame_znorm(n: vec3f) -> mat33f:
    n = wp.normalize(n)
    if wp.abs(wp.dot(n, UNIT_X)) < COS_PI_6:
        e = UNIT_X
    else:
        e = UNIT_Y
    o = wp.normalize(wp.cross(n, e))
    t = wp.normalize(wp.cross(o, n))
    return mat33f(t.x, o.x, n.x, t.y, o.y, n.y, t.z, o.z, n.z)


@wp.func
def make_contact_frame_xnorm(n: vec3f) -> mat33f:
    n = wp.normalize(n)
    if wp.abs(wp.dot(n, UNIT_X)) < COS_PI_6:
        e = UNIT_X
    else:
        e = UNIT_Y
    o = wp.normalize(wp.cross(n, e))
    t = wp.normalize(wp.cross(o, n))
    return mat33f(n.x, t.x, o.x, n.y, t.y, o.y, n.z, t.z, o.z)


###
# Interfaces
###


class Contacts:
    """
    Provides a high-level interface to manage contact data,
    including allocations, access, and common operations.

    This container provides the primary output of collision detectors
    as well as a cache of contact data to warm-start physics solvers.
    """

    def __init__(
        self,
        capacity: int | list[int] | None = None,
        default_max_contacts: int | None = None,
        device: Devicelike = None,
    ):
        # Declare and initialize the default maximum number of contacts per world
        self._default_max_world_contacts: int = DEFAULT_WORLD_MAX_CONTACTS
        if default_max_contacts is not None:
            self._default_max_world_contacts = default_max_contacts

        # Cache the target device for all memory allocations
        self._device: Devicelike = None

        # Declare the contacts data container and initialize it to empty
        self._data: ContactsData = ContactsData()

        # If a capacity is specified, finalize the contacts data allocation
        if capacity is not None:
            self.finalize(capacity=capacity, device=device)

    ###
    # Properties
    ###

    @property
    def default_max_world_contacts(self) -> int:
        """
        Returns the default maximum number of contacts per world.\n
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
            raise ValueError("max_contacts must be a non-negative integer")
        self._default_max_world_contacts = max_contacts

    @property
    def device(self) -> Devicelike:
        """
        Returns the device on which the contacts data is allocated.
        """
        return self._device

    @property
    def data(self) -> ContactsData:
        """
        Returns the managed contacts data container.
        """
        self._assert_has_data()
        return self._data

    @property
    def model_max_contacts_host(self) -> int:
        """
        Returns the host-side cache of the maximum number of contacts allocated across all worlds.\n
        Intended for managing data allocations and setting thread sizes in kernels.
        """
        self._assert_has_data()
        return self._data.model_max_contacts_host

    @property
    def world_max_contacts_host(self) -> list[int]:
        """
        Returns the host-side cache of the maximum number of contacts allocated per world.\n
        Intended for managing data allocations and setting thread sizes in kernels.
        """
        self._assert_has_data()
        return self._data.world_max_contacts_host

    @property
    def model_max_contacts(self) -> wp.array:
        """
        Returns the number of active contacts per model.\n
        Shape of ``(1,)`` and type :class:`int32`.
        """
        self._assert_has_data()
        return self._data.model_max_contacts

    @property
    def model_active_contacts(self) -> wp.array:
        """
        Returns the number of active contacts detected across all worlds in the model.\n
        Shape of ``(1,)`` and type :class:`int32`.
        """
        self._assert_has_data()
        return self._data.model_active_contacts

    @property
    def world_max_contacts(self) -> wp.array:
        """
        Returns the maximum number of contacts pre-allocated for each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """
        self._assert_has_data()
        return self._data.world_max_contacts

    @property
    def world_active_contacts(self) -> wp.array:
        """
        Returns the number of active contacts detected in each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """
        self._assert_has_data()
        return self._data.world_active_contacts

    @property
    def wid(self) -> wp.array:
        """
        Returns the world index of each active contact.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`int32`.
        """
        self._assert_has_data()
        return self._data.wid

    @property
    def cid(self) -> wp.array:
        """
        Returns the contact index of each active contact w.r.t its world.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`int32`.
        """
        self._assert_has_data()
        return self._data.cid

    @property
    def gid_AB(self) -> wp.array:
        """
        Returns the geometry indices of the geometry-pair AB associated with each active contact.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`vec2i`.
        """
        self._assert_has_data()
        return self._data.gid_AB

    @property
    def bid_AB(self) -> wp.array:
        """
        Returns the body indices of the body-pair AB associated with each active contact.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`vec2i`.
        """
        self._assert_has_data()
        return self._data.bid_AB

    @property
    def position_A(self) -> wp.array:
        """
        Returns the position of each active contact on the associated body-A in world coordinates.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`vec3f`.
        """
        self._assert_has_data()
        return self._data.position_A

    @property
    def position_B(self) -> wp.array:
        """
        Returns the position of each active contact on the associated body-B in world coordinates.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`vec3f`.
        """
        self._assert_has_data()
        return self._data.position_B

    @property
    def gapfunc(self) -> wp.array:
        """
        Returns the gap-function (i.e. signed-distance) of each
        active contact with format `(xyz: normal, w: penetration)`.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`vec4f`.
        """
        self._assert_has_data()
        return self._data.gapfunc

    @property
    def frame(self) -> wp.array:
        """
        Returns the coordinate frame of each active contact as a rotation quaternion w.r.t the world.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`quatf`.
        """
        self._assert_has_data()
        return self._data.frame

    @property
    def material(self) -> wp.array:
        """
        Returns the material properties of each active contact with format `(0: friction, 1: restitution)`.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`vec2f`.
        """
        self._assert_has_data()
        return self._data.material

    @property
    def key(self) -> wp.array:
        """
        Returns the integer key uniquely identifying each active contact.\n
        The per-contact key assignment is implementation-dependent, but is typically
        computed from the A/B geom-pair index as well as additional information such as:
        - the triangle index
        - shape-specific topological data
        - contact index w.r.t the geom-pair\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`uint64`.
        """
        self._assert_has_data()
        return self._data.key

    @property
    def reaction(self) -> wp.array:
        """
        Returns the 3D contact reaction (force/impulse) expressed in the respective local contact frame.\n
        This is to be set by solvers at each step, and also facilitates contact visualization and warm-starting.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`vec3f`.
        """
        self._assert_has_data()
        return self._data.reaction

    @property
    def velocity(self) -> wp.array:
        """
        Returns the 3D contact velocity expressed in the respective local contact frame.\n
        This is to be set by solvers at each step, and also facilitates contact visualization and warm-starting.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`vec3f`.
        """
        self._assert_has_data()
        return self._data.velocity

    @property
    def mode(self) -> wp.array:
        """
        Returns the discrete contact mode expressed as an integer value.\n
        The possible values correspond to those of the :class:`ContactMode`.\n
        This is to be set by solvers at each step, and also facilitates contact visualization and warm-starting.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`int32`.
        """
        self._assert_has_data()
        return self._data.mode

    ###
    # Operations
    ###

    def finalize(self, capacity: int | list[int], device: Devicelike = None):
        """
        Finalizes the contacts data allocations based on the specified capacity.

        Args:
            capacity (int | list[int]):
                The maximum number of contacts to allocate.\n
                If an integer is provided, it specifies the capacity for a single world.\n
                If a list of integers is provided, it specifies the capacity for each world.
            device (Devicelike, optional):
                The device on which to allocate the contacts data.
        """
        # The memory allocation requires the total number of contacts (over multiple worlds)
        # as well as the contacts capacities for each world. Corresponding sizes are defaulted to 0 (empty).
        model_max_contacts = 0
        world_max_contacts = [0]

        # If the capacity is a list, this means we are allocating for multiple worlds
        if isinstance(capacity, list):
            if len(capacity) == 0:
                raise ValueError("`capacity` must be an non-empty list")
            for i in range(len(capacity)):
                if capacity[i] < 0:
                    raise ValueError(f"`capacity[{i}]` must be a non-negative integer")
                if capacity[i] == 0:
                    capacity[i] = self._default_max_world_contacts
            model_max_contacts = sum(capacity)
            world_max_contacts = capacity

        # If the capacity is a single integer, this means we are allocating for a single world
        elif isinstance(capacity, int):
            if capacity < 0:
                raise ValueError("`capacity` must be a non-negative integer")
            if capacity == 0:
                capacity = self._default_max_world_contacts
            model_max_contacts = capacity
            world_max_contacts = [capacity]

        else:
            raise TypeError("`capacity` must be an integer or a list of integers")

        # Skip allocation if there are no contacts to allocate
        if model_max_contacts == 0:
            msg.debug("Contacts: Skipping contact data allocations since total requested capacity was `0`.")
            return

        # Override the device if specified
        if device is not None:
            self._device = device

        # Allocate the contacts data on the specified device
        with wp.ScopedDevice(self._device):
            self._data = ContactsData(
                model_max_contacts_host=model_max_contacts,
                world_max_contacts_host=world_max_contacts,
                model_max_contacts=wp.array([model_max_contacts], dtype=int32),
                model_active_contacts=wp.zeros(shape=1, dtype=int32),
                world_max_contacts=wp.array(world_max_contacts, dtype=int32),
                world_active_contacts=wp.zeros(shape=len(world_max_contacts), dtype=int32),
                wid=wp.full(value=-1, shape=(model_max_contacts,), dtype=int32),
                cid=wp.full(value=-1, shape=(model_max_contacts,), dtype=int32),
                gid_AB=wp.full(value=vec2i(-1, -1), shape=(model_max_contacts,), dtype=vec2i),
                bid_AB=wp.full(value=vec2i(-1, -1), shape=(model_max_contacts,), dtype=vec2i),
                position_A=wp.zeros(shape=(model_max_contacts,), dtype=vec3f),
                position_B=wp.zeros(shape=(model_max_contacts,), dtype=vec3f),
                gapfunc=wp.zeros(shape=(model_max_contacts,), dtype=vec4f),
                frame=wp.zeros(shape=(model_max_contacts,), dtype=quatf),
                material=wp.zeros(shape=(model_max_contacts,), dtype=vec2f),
                key=wp.zeros(shape=(model_max_contacts,), dtype=wp.uint64),
                reaction=wp.zeros(shape=(model_max_contacts,), dtype=vec3f),
                velocity=wp.zeros(shape=(model_max_contacts,), dtype=vec3f),
                mode=wp.full(value=ContactMode.INACTIVE, shape=(model_max_contacts,), dtype=int32),
            )

    def clear(self):
        """
        Clears the count of active contacts.
        """
        self._assert_has_data()
        if self._data.model_max_contacts_host > 0:
            self._data.clear()

    def reset(self):
        """
        Clears the count of active contacts and resets data to sentinel values.
        """
        self._assert_has_data()
        if self._data.model_max_contacts_host > 0:
            self._data.reset()

    ###
    # Internals
    ###

    def _assert_has_data(self):
        if self._data.model_max_contacts_host == 0:
            raise RuntimeError("ContactsData has not been allocated. Call `finalize()` before accessing data.")
