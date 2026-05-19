# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Defines the representation of discrete contacts in Kamino.

This module provides a set of data types and operations that define
the data layout and conventions used to represent discrete contacts
within the Kamino solver. It includes:

- The :class:`ContactsKaminoData` dataclass defining the structure of contact data.

- The :class:`ContactMode` enumeration defining the discrete contact modes
and a member function that generates Warp functions to compute the contact
mode based on local contact velocities.

- Utility functions for constructing contact-local coordinate frames
supporting both a Z-up and X-up convention.

- The :class:`ContactsKamino` container which provides a high-level interface to
  manage contact data, including allocations, access, and common operations,
  and fundamentally serves as the primary output of collision detectors
  as well as a cache of contact data to warm-start physics solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import warp as wp

from .....sim.contacts import Contacts
from .....sim.model import Model
from .....sim.state import State
from ..core.math import COS_PI_6, UNIT_X, UNIT_Y
from ..core.model import ModelKamino
from ..utils import logger as msg
from .keying import build_pair_key2

###
# Module interface
###

__all__ = [
    "DEFAULT_GEOM_PAIR_CONTACT_GAP",
    "DEFAULT_GEOM_PAIR_MAX_CONTACTS",
    "DEFAULT_TRIANGLE_MAX_PAIRS",
    "DEFAULT_WORLD_MAX_CONTACTS",
    "ContactMode",
    "ContactsKamino",
    "ContactsKaminoData",
    "convert_contacts_kamino_to_newton",
    "convert_contacts_newton_to_kamino",
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

DEFAULT_MODEL_MAX_CONTACTS: int = 1000
"""
The global default for maximum number of contacts per model.\n
Used when allocating contact data without a specified capacity.\n
Set to `1000`.
"""

DEFAULT_WORLD_MAX_CONTACTS: int = 128
"""
The global default for maximum number of contacts per world.\n
Used when allocating contact data without a specified capacity.\n
Set to `128`.
"""

DEFAULT_GEOM_PAIR_MAX_CONTACTS: int = 12
"""
The global default for maximum number of contacts per geom-pair.\n
Used when allocating contact data without a specified capacity.\n
Ignored for mesh-based collisions.\n
Set to `12` (with box-box collisions being a prototypical case).
"""

DEFAULT_TRIANGLE_MAX_PAIRS: int = 1_000_000
"""
The global default for maximum number of triangle pairs to consider in the narrow-phase.\n
Used only when the model contains triangle meshes or heightfields.\n
Defaults to `1_000_000`.
"""

DEFAULT_GEOM_PAIR_CONTACT_GAP: float = 1e-5
"""
The global default for the per-geometry detection gap [m].\n
Applied as a floor to each per-geometry gap value during pipeline
initialization so that every geometry has at least this detection
threshold.\n
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
        def _compute_mode(v: wp.vec3f) -> wp.int32:
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
            mode = wp.int32(ContactMode.OPENING)
            if v_N <= wp.float32(vn_tol):
                if v_T_norm <= wp.float32(vt_tol):
                    mode = ContactMode.STICKING
                else:
                    mode = ContactMode.SLIDING

            # Return the resulting contact mode integer
            return mode

        # Return the generated compute mode function
        return _compute_mode


@dataclass
class ContactsKaminoData:
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

    model_max_contacts: wp.array[wp.int32] | None = None
    """
    The number of contacts pre-allocated across all worlds in the model.\n
    Shape of ``(1,)`` and type :class:`int32`.
    """

    model_active_contacts: wp.array[wp.int32] | None = None
    """
    The number of active contacts detected across all worlds in the model.\n
    Shape of ``(1,)`` and type :class:`int32`.
    """

    world_max_contacts: wp.array[wp.int32] | None = None
    """
    The maximum number of contacts pre-allocated for each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    world_active_contacts: wp.array[wp.int32] | None = None
    """
    The number of active contacts detected in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    wid: wp.array[wp.int32] | None = None
    """
    The world index of each active contact.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`int32`.
    """

    cid: wp.array[wp.int32] | None = None
    """
    The contact index of each active contact w.r.t its world.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`int32`.
    """

    gid_AB: wp.array[wp.vec2i] | None = None
    """
    The geometry indices of the geometry-pair AB associated with each active contact.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`vec2i`.
    """

    bid_AB: wp.array[wp.vec2i] | None = None
    """
    The body indices of the body-pair AB associated with each active contact.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`vec2i`.
    """

    position_A: wp.array[wp.vec3f] | None = None
    """
    The position of each active contact on the associated body-A in world coordinates.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`vec3f`.
    """

    position_B: wp.array[wp.vec3f] | None = None
    """
    The position of each active contact on the associated body-B in world coordinates.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`vec3f`.
    """

    gapfunc: wp.array[wp.vec4f] | None = None
    """
    Gap-function of each active contact, format ``(xyz: normal, w: signed_distance)``.\n
    The ``w`` component stores the signed distance between margin-shifted surfaces:
    negative means penetration past the resting separation, positive means separation
    within the detection gap.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`vec4f`.
    """

    frame: wp.array[wp.quatf] | None = None
    """
    The coordinate frame of each active contact as a rotation quaternion w.r.t the world.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`quatf`.
    """

    material: wp.array[wp.vec2f] | None = None
    """
    The material properties of each active contact with format `(0: friction, 1: restitution)`.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`vec2f`.
    """

    margins: wp.array[wp.vec2f] | None = None
    """
    The shape-pair margins of each active contact.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`vec2f`.
    """

    key: wp.array[wp.uint64] | None = None
    """
    Integer key uniquely identifying each active contact.\n
    The per-contact key assignment is implementation-dependent, but is typically
    computed from the A/B geom-pair index as well as additional information such as:
    - the triangle index
    - shape-specific topological data
    - contact index w.r.t the geom-pair\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`uint64`.
    """

    reaction: wp.array[wp.vec3f] | None = None
    """
    The 3D contact reaction (force/impulse) expressed in the respective local contact frame.\n
    This is to be set by solvers at each step, and also facilitates contact visualization and warm-starting.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`vec3f`.
    """

    velocity: wp.array[wp.vec3f] | None = None
    """
    The 3D contact velocity expressed in the respective local contact frame.\n
    This is to be set by solvers at each step, and also facilitates contact visualization and warm-starting.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`vec3f`.
    """

    mode: wp.array[wp.int32] | None = None
    """
    The discrete contact mode expressed as an integer value.\n
    The possible values correspond to those of the :class:`ContactMode`.\n
    This is to be set by solvers at each step, and also facilitates contact visualization and warm-starting.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`int32`.
    """

    remap: wp.array[wp.int32] | None = None
    """
    Per-contact mapping back to the source contact index when converted from Newton :class:`Contacts`.\n
    Shape of ``(model_max_contacts_host,)`` and type :class:`int32`.

    Populated by :func:`convert_contacts_newton_to_kamino` so that each Kamino
    contact knows which original Newton contact it was generated from; entries
    default to ``-1`` for unmapped/inactive contacts.

    Consumed by :func:`convert_contacts_kamino_to_newton` along the
    existing-contacts path (``clear_output=False``) to write each contact's
    converted wrench back into the matching Newton slot. Only allocated when
    :class:`ContactsKamino` is constructed with ``remappable=True``.
    """

    def clear(self):
        """
        Clears the count of active contacts.
        """
        self.model_active_contacts.zero_()
        self.world_active_contacts.zero_()
        if self.remap is not None:
            self.remap.fill_(-1)

    def reset(self):
        """
        Clears the count of active contacts and resets contact data
        to sentinel values, indicating an empty set of contacts.
        """
        self.clear()
        self.wid.fill_(-1)
        self.cid.fill_(-1)
        self.gid_AB.fill_(wp.vec2i(-1, -1))
        self.bid_AB.fill_(wp.vec2i(-1, -1))
        self.mode.fill_(ContactMode.INACTIVE)
        self.reaction.zero_()
        self.velocity.zero_()


###
# Functions
###


@wp.func
def make_contact_frame_znorm(n: wp.vec3f) -> wp.mat33f:
    n = wp.normalize(n)
    if wp.abs(wp.dot(n, UNIT_X)) < COS_PI_6:
        e = UNIT_X
    else:
        e = UNIT_Y
    o = wp.normalize(wp.cross(n, e))
    t = wp.normalize(wp.cross(o, n))
    return wp.mat33f(t.x, o.x, n.x, t.y, o.y, n.y, t.z, o.z, n.z)


@wp.func
def make_contact_frame_xnorm(n: wp.vec3f) -> wp.mat33f:
    n = wp.normalize(n)
    if wp.abs(wp.dot(n, UNIT_X)) < COS_PI_6:
        e = UNIT_X
    else:
        e = UNIT_Y
    o = wp.normalize(wp.cross(n, e))
    t = wp.normalize(wp.cross(o, n))
    return wp.mat33f(n.x, t.x, o.x, n.y, t.y, o.y, n.z, t.z, o.z)


###
# Interfaces
###


class ContactsKamino:
    """
    Provides a high-level interface to manage contact data,
    including allocations, access, and common operations.

    This container provides the primary output of collision detectors
    as well as a cache of contact data to warm-start physics solvers.
    """

    def __init__(
        self,
        model: ModelKamino | None = None,
        capacity: int | list[int] | None = None,
        default_max_contacts: int | None = None,
        device: wp.DeviceLike = None,
        remappable: bool = False,
    ):
        """
        Initializes a new ContactsKamino container.

        Args:
            model:
                The model container holding the time-invariant data of the system being simulated.\n
                If provided, the contacts will be finalized using the contact allocation meta-data of the model.\n
                If `None`, the contacts will be created empty without allocating data, and
                can be finalized later by providing a model to the `finalize` method.\n
            capacity:
                The maximum number of contacts to allocate.\n
                If not provided, the contact allocation meta-data of the model will be used.\n
                If an integer is provided, it specifies the capacity for a single world.\n
                If a list of integers is provided, it specifies the capacity for each world.
            default_max_contacts:
                The default maximum number of contacts per world.\n
                If `None`, uses the default value of 1000.
            device:
                The device on which to allocate the contacts data.
        """
        # Declare and initialize the default maximum number of contacts per world
        self._default_max_world_contacts: int = DEFAULT_WORLD_MAX_CONTACTS
        if default_max_contacts is not None:
            self._default_max_world_contacts = default_max_contacts

        # Cache the target device for all memory allocations
        self._device: wp.DeviceLike = None

        # Declare the contacts data container and initialize it to empty
        self._data: ContactsKaminoData = ContactsKaminoData()

        # If a capacity is specified, finalize the contacts data allocation
        if model is not None or capacity is not None:
            self.finalize(model=model, capacity=capacity, device=device, remappable=remappable)

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
    def device(self) -> wp.DeviceLike:
        """
        Returns the device on which the contacts data is allocated.
        """
        return self._device

    @property
    def data(self) -> ContactsKaminoData:
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
    def model_max_contacts(self) -> wp.array[wp.int32]:
        """
        Returns the number of active contacts per model.\n
        Shape of ``(1,)`` and type :class:`int32`.
        """
        self._assert_has_data()
        return self._data.model_max_contacts

    @property
    def model_active_contacts(self) -> wp.array[wp.int32]:
        """
        Returns the number of active contacts detected across all worlds in the model.\n
        Shape of ``(1,)`` and type :class:`int32`.
        """
        self._assert_has_data()
        return self._data.model_active_contacts

    @property
    def world_max_contacts(self) -> wp.array[wp.int32]:
        """
        Returns the maximum number of contacts pre-allocated for each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """
        self._assert_has_data()
        return self._data.world_max_contacts

    @property
    def world_active_contacts(self) -> wp.array[wp.int32]:
        """
        Returns the number of active contacts detected in each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """
        self._assert_has_data()
        return self._data.world_active_contacts

    @property
    def wid(self) -> wp.array[wp.int32]:
        """
        Returns the world index of each active contact.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`int32`.
        """
        self._assert_has_data()
        return self._data.wid

    @property
    def cid(self) -> wp.array[wp.int32]:
        """
        Returns the contact index of each active contact w.r.t its world.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`int32`.
        """
        self._assert_has_data()
        return self._data.cid

    @property
    def gid_AB(self) -> wp.array[wp.vec2i]:
        """
        Returns the geometry indices of the geometry-pair AB associated with each active contact.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`vec2i`.
        """
        self._assert_has_data()
        return self._data.gid_AB

    @property
    def bid_AB(self) -> wp.array[wp.vec2i]:
        """
        Returns the body indices of the body-pair AB associated with each active contact.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`vec2i`.
        """
        self._assert_has_data()
        return self._data.bid_AB

    @property
    def position_A(self) -> wp.array[wp.vec3f]:
        """
        Returns the position of each active contact on the associated body-A in world coordinates.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`vec3f`.
        """
        self._assert_has_data()
        return self._data.position_A

    @property
    def position_B(self) -> wp.array[wp.vec3f]:
        """
        Returns the position of each active contact on the associated body-B in world coordinates.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`vec3f`.
        """
        self._assert_has_data()
        return self._data.position_B

    @property
    def gapfunc(self) -> wp.array[wp.vec4f]:
        """
        Returns the gap-function of each active contact, packed as``(xyz: normal, w: distance)``.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`vec4f`.

        The ``w`` component stores the signed ``distance`` between margin-shifted surfaces:
        - ``w < 0`` means penetration past the resting separation defined by the margin
        - ``w > 0`` means separation within the detection ``distance = gap + margin``
        """
        self._assert_has_data()
        return self._data.gapfunc

    @property
    def frame(self) -> wp.array[wp.quatf]:
        """
        Returns the coordinate frame of each active contact as a rotation quaternion w.r.t the world.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`quatf`.
        """
        self._assert_has_data()
        return self._data.frame

    @property
    def material(self) -> wp.array[wp.vec2f]:
        """
        Returns the material properties of each active contact with format `(0: friction, 1: restitution)`.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`vec2f`.
        """
        self._assert_has_data()
        return self._data.material

    @property
    def margins(self) -> wp.array[wp.float32]:
        """
        Returns the effective shape-pair margins of each active contact.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`float32`.
        """
        self._assert_has_data()
        return self._data.margins

    @property
    def key(self) -> wp.array[wp.uint64]:
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
    def reaction(self) -> wp.array[wp.vec3f]:
        """
        Returns the 3D contact reaction (force/impulse) expressed in the respective local contact frame.\n
        This is to be set by solvers at each step, and also facilitates contact visualization and warm-starting.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`vec3f`.
        """
        self._assert_has_data()
        return self._data.reaction

    @property
    def velocity(self) -> wp.array[wp.vec3f]:
        """
        Returns the 3D contact velocity expressed in the respective local contact frame.\n
        This is to be set by solvers at each step, and also facilitates contact visualization and warm-starting.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`vec3f`.
        """
        self._assert_has_data()
        return self._data.velocity

    @property
    def mode(self) -> wp.array[wp.int32]:
        """
        Returns the discrete contact mode expressed as an integer value.\n
        The possible values correspond to those of the :class:`ContactMode`.\n
        This is to be set by solvers at each step, and also facilitates contact visualization and warm-starting.\n
        Shape of ``(model_max_contacts_host,)`` and type :class:`int32`.
        """
        self._assert_has_data()
        return self._data.mode

    @property
    def remap(self) -> wp.array[wp.int32] | None:
        """
        Returns the remapped contact index of each active contact.
        Shape of ``(model_max_contacts_host,)`` and type :class:`int32`.
        """
        self._assert_has_data()
        return self._data.remap

    ###
    # Operations
    ###

    def finalize(
        self,
        model: ModelKamino | None = None,
        capacity: int | list[int] | None = None,
        device: wp.DeviceLike = None,
        remappable: bool = False,
    ):
        """
        Finalizes the contacts data allocations based on the specified model or capacity.

        Args:
            model:
                The model container holding the time-invariant data of the system being simulated.\n
                If provided, the contacts will be finalized using the contact allocation meta-data of the model.\n
                If `None`, the contacts will be created empty without allocating data, and
                can be finalized later by providing a model to the `finalize` method.\n
            capacity:
                The maximum number of contacts to allocate.\n
                If not provided, the contact allocation meta-data of the model will be used.\n
                If an integer is provided, it specifies the capacity for a single world.\n
                If a list of integers is provided, it specifies the capacity for each world.
            device:
                The device on which to allocate the contacts data.
        """
        # Raise errors if both model and capacity are provided or both are None
        if model is not None and capacity is not None:
            raise ValueError("Expected either 'model' or 'capacity' argument to be provided, but not both.")
        if model is None and capacity is None:
            raise ValueError("Expected either 'model' or 'capacity' argument to be provided, but got neither")

        # If a model is provided, extract the required contacts capacity from that
        if model is not None:
            model_max_contacts: int = 0
            world_max_contacts: list[int] = [0 for _ in range(model.size.num_worlds)]
            if model.geoms.model_minimum_contacts > 0:
                model_max_contacts = model.geoms.model_minimum_contacts
                world_max_contacts = model.geoms.world_minimum_contacts
            else:
                num_worlds = model.size.num_worlds
                world_max_contacts = [model_max_contacts // num_worlds] * num_worlds
            capacity = world_max_contacts
            self._device = model.device
        else:
            self._device = device

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
            msg.debug("ContactsKamino: Skipping contact data allocations since total requested capacity was `0`.")
            return

        # Allocate the contacts data on the specified device
        with wp.ScopedDevice(self._device):
            self._data = ContactsKaminoData(
                model_max_contacts_host=model_max_contacts,
                world_max_contacts_host=world_max_contacts,
                model_max_contacts=wp.array([model_max_contacts], dtype=wp.int32),
                model_active_contacts=wp.zeros(shape=1, dtype=wp.int32),
                world_max_contacts=wp.array(world_max_contacts, dtype=wp.int32),
                world_active_contacts=wp.zeros(shape=len(world_max_contacts), dtype=wp.int32),
                wid=wp.full(value=-1, shape=(model_max_contacts,), dtype=wp.int32),
                cid=wp.full(value=-1, shape=(model_max_contacts,), dtype=wp.int32),
                gid_AB=wp.full(value=wp.vec2i(-1, -1), shape=(model_max_contacts,), dtype=wp.vec2i),
                bid_AB=wp.full(value=wp.vec2i(-1, -1), shape=(model_max_contacts,), dtype=wp.vec2i),
                position_A=wp.zeros(shape=(model_max_contacts,), dtype=wp.vec3f),
                position_B=wp.zeros(shape=(model_max_contacts,), dtype=wp.vec3f),
                gapfunc=wp.zeros(shape=(model_max_contacts,), dtype=wp.vec4f),
                frame=wp.zeros(shape=(model_max_contacts,), dtype=wp.quatf),
                material=wp.zeros(shape=(model_max_contacts,), dtype=wp.vec2f),
                margins=wp.zeros(shape=(model_max_contacts,), dtype=wp.vec2f),
                key=wp.zeros(shape=(model_max_contacts,), dtype=wp.uint64),
                reaction=wp.zeros(shape=(model_max_contacts,), dtype=wp.vec3f),
                velocity=wp.zeros(shape=(model_max_contacts,), dtype=wp.vec3f),
                mode=wp.full(value=ContactMode.INACTIVE, shape=(model_max_contacts,), dtype=wp.int32),
                remap=wp.full(value=-1, shape=(model_max_contacts,), dtype=wp.int32) if remappable else None,
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
            raise RuntimeError("ContactsKaminoData has not been allocated. Call `finalize()` before accessing data.")


###
# Conversions - Kernels
###


@wp.kernel
def _convert_contacts_newton_to_kamino(
    # Inputs:
    num_worlds: wp.int32,
    max_converted_contacts: wp.int32,
    newton_count: wp.array[wp.int32],
    newton_shape0: wp.array[wp.int32],
    newton_shape1: wp.array[wp.int32],
    newton_point0: wp.array[wp.vec3f],
    newton_point1: wp.array[wp.vec3f],
    newton_normal: wp.array[wp.vec3f],
    newton_margin0: wp.array[wp.float32],
    newton_margin1: wp.array[wp.float32],
    newton_force: wp.array[wp.spatial_vectorf],
    shape_body: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    shape_mu: wp.array[wp.float32],
    shape_restitution: wp.array[wp.float32],
    body_q: wp.array[wp.transformf],
    # Outputs:
    kamino_model_active: wp.array[wp.int32],
    kamino_world_active: wp.array[wp.int32],
    kamino_wid: wp.array[wp.int32],
    kamino_cid: wp.array[wp.int32],
    kamino_gid_AB: wp.array[wp.vec2i],
    kamino_bid_AB: wp.array[wp.vec2i],
    kamino_position_A: wp.array[wp.vec3f],
    kamino_position_B: wp.array[wp.vec3f],
    kamino_gapfunc: wp.array[wp.vec4f],
    kamino_frame: wp.array[wp.quatf],
    kamino_material: wp.array[wp.vec2f],
    kamino_margins: wp.array[wp.vec2f],
    kamino_key: wp.array[wp.uint64],
    kamino_reaction: wp.array[wp.vec3f],
    kamino_remap: wp.array[wp.int32],
):
    """
    Convert Newton :class:`Contacts` to Kamino's :class:`ContactsKamino` format.

    Reads body-local contact points from Newton, transforms them to world space,
    and populates the Kamino contact arrays under the A/B convention that
    Kamino's solver core expects: ``bid_B >= 0``, normal points A -> B. When
    Newton's ``shape1`` is world-static (``bid_1 < 0``), shape1 becomes Kamino A
    and shape0 becomes Kamino B (the A<->B swap); otherwise A=shape0, B=shape1.

    Newton's ``rigid_contact_normal`` points from shape0 toward shape1 (A -> B in
    the no-swap case, B -> A in the swap case, which is negated to restore the
    Kamino A->B convention).

    Optionally also converts Newton's :attr:`Contacts.force` (the wrench on body0
    by body1 at the CoM of body0, in world) into Kamino's ``reaction`` (the linear
    force on body B by body A in the local contact frame). The linear part is
    invariant to reference-point shifts, so this is a pure rotation into the
    contact frame, with a sign flip in the no-swap case to convert "force on A"
    into "force on B".
    """
    # Retrieve the contact index for this thread
    cid = wp.tid()

    # Retrieve the total number of active contacts to convert
    num_active = wp.min(newton_count[0], max_converted_contacts)

    # Skip conversion if this contact index exceeds the number
    # of contacts to convert or the maximum output capacity.
    if cid >= num_active:
        return

    # Retrieve the shape and body indices for this contact
    sid_0 = newton_shape0[cid]
    sid_1 = newton_shape1[cid]
    bid_0 = shape_body[sid_0]
    bid_1 = shape_body[sid_1]
    wid_0 = shape_world[sid_0]
    wid_1 = shape_world[sid_1]

    # Determine the world index.  Global shapes (shape_world == -1) can
    # collide with shapes from any world, so fall back to the other shape.
    wid = wid_0
    if wid_0 < 0:
        wid = wid_1
    if wid < 0 or wid >= num_worlds:
        return

    # Body-local → world-space
    X_0 = wp.transform_identity()
    if bid_0 >= 0:
        X_0 = body_q[bid_0]
    X_1 = wp.transform_identity()
    if bid_1 >= 0:
        X_1 = body_q[bid_1]
    r_0 = wp.transform_point(X_0, newton_point0[cid])
    r_1 = wp.transform_point(X_1, newton_point1[cid])

    # Newton normal points from shape0 → shape1 (A → B).
    # Kamino convention: normal points A → B, with bid_B >= 0.
    normal = newton_normal[cid]

    # Reconstruct Newton signed contact distance d from exported fields:
    # d = dot((p1 - p0), n_a_to_b) - (offset0 + offset1),
    # with n_newton = n_a_to_b and offset* stored in newton_margin*.
    margin_0 = newton_margin0[cid]
    margin_1 = newton_margin1[cid]
    margin = margin_0 + margin_1
    distance = wp.dot(r_1 - r_0, normal) - margin

    # Ensure static body is always Kamino A, dynamic body is Kamino B
    if bid_1 < 0:
        # shape1 is world-static → make it Kamino A, shape0 becomes Kamino B.
        # Kamino A→B = shape1→shape0, opposite of Newton's shape0→shape1, so negate.
        gid_A = sid_1
        gid_B = sid_0
        bid_A = bid_1
        bid_B = bid_0
        r_A = r_1
        r_B = r_0
        margin_A = margin_1
        margin_B = margin_0
        normal = -normal
    else:
        # Both dynamic or shape0 is static → keep A=shape0, B=shape1.
        # Newton normal already points A→B, matching Kamino convention.
        gid_A = sid_0
        gid_B = sid_1
        bid_A = bid_0
        bid_B = bid_1
        r_A = r_0
        r_B = r_1
        margin_A = margin_0
        margin_B = margin_1

    # Skip conversion if the contact distance is positive
    if distance > 0.0:
        return

    # Retrieve the material properties for this contact
    # TODO: Integrate use of material manager to retrieve material properties
    mu = 0.5 * (shape_mu[sid_0] + shape_mu[sid_1])
    epsilon = 0.5 * (shape_restitution[sid_0] + shape_restitution[sid_1])

    # Store the contact data in the Kamino format
    gapfunc = wp.vec4f(normal[0], normal[1], normal[2], distance)
    q_frame = wp.quat_from_matrix(make_contact_frame_znorm(normal))

    # Increment the number of active contacts in the model and world
    mcid = wp.atomic_add(kamino_model_active, 0, 1)
    wcid = wp.atomic_add(kamino_world_active, wid, 1)

    # Store the contact data in the Kamino format if the contact is valid
    if mcid < max_converted_contacts:
        kamino_wid[mcid] = wid
        kamino_cid[mcid] = wcid
        kamino_gid_AB[mcid] = wp.vec2i(gid_A, gid_B)
        kamino_bid_AB[mcid] = wp.vec2i(bid_A, bid_B)
        kamino_position_A[mcid] = r_A
        kamino_position_B[mcid] = r_B
        kamino_gapfunc[mcid] = gapfunc
        kamino_frame[mcid] = q_frame
        kamino_material[mcid] = wp.vec2f(mu, epsilon)
        kamino_margins[mcid] = wp.vec2f(margin_A, margin_B)
        kamino_key[mcid] = build_pair_key2(wp.uint32(gid_A), wp.uint32(gid_B))

        # Store the contact source index in the remap array if provided
        if kamino_remap:
            kamino_remap[mcid] = cid

        # Optional contact wrench from Newton convention.
        # Newton stores `force[cid]` as the wrench on body0 by body1 at the CoM
        # of body0 in world coordinates. Kamino's `reaction` is the linear
        # force on body B by body A in the local contact frame. The linear
        # part is invariant under reference-point shifts, so we only need to
        # rotate to the local frame and choose the sign based on the swap:
        #   - no-swap (bid_1 >= 0): Newton body0 = Kamino A, sign = -1
        #   - swap   (bid_1 <  0): Newton body0 = Kamino B, sign = +1
        if newton_force:
            f_world = wp.spatial_top(newton_force[cid])
            f_local = wp.quat_rotate(wp.quat_inverse(q_frame), f_world)
            if bid_1 < 0:
                kamino_reaction[mcid] = f_local
            else:
                kamino_reaction[mcid] = -f_local

    # If the contact is invalid, decrement the number of active contacts in the model and world
    else:
        wp.atomic_sub(kamino_model_active, 0, 1)
        wp.atomic_sub(kamino_world_active, wid, 1)


@wp.kernel
def _convert_active_contacts_kamino_to_newton(
    # Inputs:
    max_converted_contacts: wp.int32,
    model_active_contacts: wp.array[wp.int32],
    kamino_gid_AB: wp.array[wp.vec2i],
    kamino_position_A: wp.array[wp.vec3f],
    kamino_position_B: wp.array[wp.vec3f],
    kamino_gapfunc: wp.array[wp.vec4f],
    kamino_frame: wp.array[wp.quatf],
    kamino_reaction: wp.array[wp.vec3f],
    kamino_margins: wp.array[wp.vec2f],
    shape_body: wp.array[wp.int32],
    body_com: wp.array[wp.vec3f],
    body_q: wp.array[wp.transformf],
    # Outputs:
    newton_count: wp.array[wp.int32],
    newton_shape0: wp.array[wp.int32],
    newton_shape1: wp.array[wp.int32],
    newton_margin0: wp.array[wp.float32],
    newton_margin1: wp.array[wp.float32],
    newton_point0: wp.array[wp.vec3f],
    newton_point1: wp.array[wp.vec3f],
    newton_normal: wp.array[wp.vec3f],
    newton_force: wp.array[wp.spatial_vectorf],
):
    """
    Converts Kamino's active contacts into a freshly-cleared Newton ``Contacts``.

    This version assumes that the output Newton contacts have been cleared and
    repopulates them with currently active Kamino contacts (which may have
    passed additional solver-side filtering). Newton's ``shape0`` is written as
    ``gid_AB[0]`` (Kamino A) and ``shape1`` as ``gid_AB[1]`` (Kamino B), so
    Newton's ``force[cid_out]`` is the wrench on body A by body B at the CoM of
    body A in world coordinates: it is the negation of ``quat_rotate(frame,
    reaction)`` (which is Kamino's force on B by A in world).
    """
    # Retrieve the contact index for this thread
    cid = wp.tid()

    # Determine the total number of contacts to convert, which is the
    # smaller of the number of active contacts and the output capacity.
    num_active = wp.min(model_active_contacts[0], max_converted_contacts)

    # Skip conversion if this contact index exceeds the
    # number of active contacts or the output capacity
    if cid >= num_active:
        if newton_force:
            newton_force[cid] = wp.spatial_vectorf()
        return

    # Retrieve contact-specific data
    gid_01 = kamino_gid_AB[cid]
    r_0 = kamino_position_A[cid]
    r_1 = kamino_position_B[cid]
    gapfunc = kamino_gapfunc[cid]
    margins_01 = kamino_margins[cid]

    # Retrieve the geometry indices for this contact and use
    # them to look up the corresponding shapes and bodies.
    shape_0 = gid_01[0]
    shape_1 = gid_01[1]
    body_0 = shape_body[shape_0]
    body_1 = shape_body[shape_1]
    margin_0 = margins_01[0]
    margin_1 = margins_01[1]

    # Transform the world-space contact positions
    # back to body-local coordinates for Newton.
    X_inv_0 = wp.transform_identity()
    if body_0 >= 0:
        X_inv_0 = wp.transform_inverse(body_q[body_0])
    X_inv_1 = wp.transform_identity()
    if body_1 >= 0:
        X_inv_1 = wp.transform_inverse(body_q[body_1])

    # Increment the number of active contacts in the Newton format
    cid_out = wp.atomic_add(newton_count, 0, 1)

    # Store the converted contact data in the Newton format
    newton_shape0[cid_out] = shape_0
    newton_shape1[cid_out] = shape_1
    newton_normal[cid_out] = wp.vec3f(gapfunc[0], gapfunc[1], gapfunc[2])
    newton_point0[cid_out] = wp.transform_point(X_inv_0, r_0)
    newton_point1[cid_out] = wp.transform_point(X_inv_1, r_1)
    newton_margin0[cid_out] = margin_0
    newton_margin1[cid_out] = margin_1

    # Optional contact wrench in Newton's convention: wrench on body0 by body1
    # at the CoM of body0 in world. The active path writes body0 = Kamino A,
    # so Newton's force on body0 is the negation of Kamino's "force on B by A".
    if newton_force:
        frame = kamino_frame[cid]
        reaction = kamino_reaction[cid]
        f_0_world = -wp.quat_rotate(frame, reaction)

        # Torque is the moment of the linear force about body0's CoM. There is
        # no intrinsic contact torque, only the moment arm from body0's CoM to
        # the contact point on body0 (which is ``r_0 = position_A`` here).
        if body_0 >= 0:
            r_0_com = wp.transform_point(body_q[body_0], body_com[body_0])
            dr_0_com = r_0 - r_0_com
        else:
            dr_0_com = wp.vec3f(0.0)
        tau_0_world = wp.cross(dr_0_com, f_0_world)

        # Store the converted contact data in the Newton format
        newton_force[cid_out] = wp.spatial_vector(f_0_world, tau_0_world)


@wp.kernel
def _convert_existing_contacts_kamino_to_newton(
    # Inputs:
    max_converted_contacts: wp.int32,
    model_active_contacts: wp.array[wp.int32],
    kamino_gid_AB: wp.array[wp.vec2i],
    kamino_frame: wp.array[wp.quatf],
    kamino_reaction: wp.array[wp.vec3f],
    kamino_remap: wp.array[wp.int32],
    newton_shape0: wp.array[wp.int32],
    newton_point0: wp.array[wp.vec3f],
    shape_body: wp.array[wp.int32],
    body_com: wp.array[wp.vec3f],
    body_q: wp.array[wp.transformf],
    # Outputs:
    newton_force: wp.array[wp.spatial_vectorf],
):
    """
    Converts Kamino's contact forces back into an existing Newton ``Contacts``.

    This version assumes that geometric contact data has already been populated
    on the Newton side and only updates the per-contact wrench. The mapping
    from Kamino contact indices back to the original Newton contact indices is
    provided via ``kamino_remap``.

    Newton stores ``force[cid]`` as the wrench on Newton's body0 by Newton's
    body1, expressed at the CoM of body0 in world coordinates. Kamino's
    ``reaction`` is the linear force on body B by body A in the local contact
    frame. Whether the swap A<->B occurred at N->K time is recovered by
    comparing the preserved ``newton_shape0[cid_out]`` against ``gid_AB[0]``:

    - no-swap (``newton_shape0[cid_out] == gid_AB[0]``): Newton body0 = Kamino A,
      so ``f_world_on_body0 = -quat_rotate(frame, reaction)``.
    - swap   (``newton_shape0[cid_out] != gid_AB[0]``): Newton body0 = Kamino B,
      so ``f_world_on_body0 = +quat_rotate(frame, reaction)``.

    The torque is the moment of the linear force about Newton's body0 CoM,
    using the body-local ``newton_point0[cid_out]`` transformed to world space.
    """
    # Ensure that the remap array is provided for this kernel
    assert kamino_remap, "`kamino_remap` is required for existing contacts to be remapped"
    assert newton_force, "`newton_force` is required for existing contacts to be remapped"

    # Retrieve the contact index for this thread
    cid = wp.tid()

    # Determine the total number of contacts to convert, which is the
    # smaller of the number of active contacts and the output capacity.
    num_active = wp.min(model_active_contacts[0], max_converted_contacts)

    # Retrieve the contact index for the Newton format and ensure that
    # it has a valid mapping to the target Newton contacts container.
    cid_out = kamino_remap[cid]

    # Skip conversion if this contact index exceeds the number of active
    # contacts or it has no mapping to the target Newton contacts container.
    if cid >= num_active or cid_out < 0:
        newton_force[cid] = wp.spatial_vectorf()
        return

    # Retrieve contact-specific data
    gid_01 = kamino_gid_AB[cid]
    frame = kamino_frame[cid]
    reaction = kamino_reaction[cid]

    # Recover Newton's preserved body0 (which may correspond to either Kamino A
    # or Kamino B if a swap occurred during N->K conversion).
    shape_0_n = newton_shape0[cid_out]
    body_0_n = shape_body[shape_0_n]
    swap = shape_0_n != gid_01[0]

    # Express the contact reaction in world coordinates as the linear force
    # on Newton's body0 by Newton's body1.
    f_0_world = wp.quat_rotate(frame, reaction)
    if not swap:
        # Newton body0 = Kamino A, so the force on body0 is the reaction on A
        # by B, which is the negation of Kamino's stored "force on B by A".
        f_0_world = -f_0_world

    # Torque about Newton's body0 CoM is the moment arm from CoM to the contact
    # point on body0 crossed with the linear force. Use Newton's preserved
    # body-local ``point0`` to obtain the contact point on body0 in world space.
    if body_0_n >= 0:
        X_0 = body_q[body_0_n]
        r_pt_world = wp.transform_point(X_0, newton_point0[cid_out])
        r_com_world = wp.transform_point(X_0, body_com[body_0_n])
        dr = r_pt_world - r_com_world
    else:
        dr = wp.vec3f(0.0, 0.0, 0.0)
    tau_0_world = wp.cross(dr, f_0_world)

    # Store the converted contact data in the Newton format
    newton_force[cid_out] = wp.spatial_vector(f_0_world, tau_0_world)


###
# Conversions - Launchers
###


def convert_contacts_newton_to_kamino(
    model: Model,
    state: State,
    contacts_in: Contacts,
    contacts_out: ContactsKamino,
    convert_forces: bool = False,
):
    """
    Converts Newton's :class:`Contacts` to Kamino's :class:`ContactsKamino` format.

    Kamino's conventions for contact data are:
    - If a contact pair has one static body, that body becomes Kamino A
      (``bid_A == -1``) and the dynamic body becomes Kamino B (``bid_B >= 0``).
      Otherwise A=shape0, B=shape1 (no swap).
    - ``normal`` points from body A to body B.
    - ``reaction`` is the linear force on body B by body A, expressed at the
      contact point on body B in the local contact frame.

    This operation transforms Newton's body-local contact points to world
    coordinates, applies Kamino's A/B conventions and populates the
    :class:`ContactsKamino` fields.

    If ``convert_forces`` is true and Newton's :attr:`Contacts.force` is set,
    each Newton wrench (on body0 by body1 at the CoM of body0, in world) is
    rotated into the local contact frame and stored as
    :attr:`ContactsKaminoData.reaction`, with a sign chosen by whether the
    A<->B swap occurred. Only the linear part of Newton's wrench is used,
    since the linear force is invariant under reference-point shifts.

    Args:
        model:
            The input :class:`Model` object providing shape and body information
            used to interpret Newton's contact data and populate Kamino's contact data.
        state:
            The input :class:`State` object providing ``body_q`` and ``body_com``
            used to transform contact points from body-local to world coordinates.
        contacts_in:
            The input :class:`Contacts` object containing contact information to be converted.
        contacts_out:
            The output :class:`ContactsKamino` object to populate with the converted contact data.
        convert_forces:
            If ``True``, also convert ``contacts_in.force`` into``contacts_out.reaction``.\n
            If ``False`` or ``contacts_in.force`` is missing, ``contacts_out.reaction`` is left untouched.
    """
    # Skip conversion if there are no contacts to convert or no capacity to store them.
    if contacts_out.model_max_contacts_host == 0 or contacts_in.rigid_contact_max == 0:
        return

    # Ensure that the model, state, contacts_in and
    # contacts_out are all on the same device.
    if (
        contacts_out.device != model.device
        or contacts_out.device != state.body_q.device
        or contacts_out.device != contacts_in.device
    ):
        raise ValueError(
            "All inputs must be on the same device: "
            f"model.device={model.device}, "
            f"state.device={state.body_q.device}, "
            f"contacts_in.device={contacts_in.device}, "
            f"contacts_out.device={contacts_out.device}"
        )

    # Issue warning to the user if the number of contacts to
    # convert exceeds the capacity of the output contacts.
    if contacts_in.rigid_contact_max > contacts_out.model_max_contacts_host:
        msg.warning(
            "Newton `rigid_contact_max` (%d) exceeds Kamino `model_max_contacts_host` (%d); contacts will be truncated.",
            contacts_in.rigid_contact_max,
            contacts_out.model_max_contacts_host,
        )

    # Skip conversion of contact forces if not requested
    contacts_in_force = contacts_in.force if convert_forces else None

    # Set the maximum number of contacts to convert to the smallest of the
    # number of contacts detected and the maximum capacity of the output contacts.
    max_converted_contacts = min(contacts_in.rigid_contact_max, contacts_out.model_max_contacts_host)

    # Clear the output contacts to reset the active contact
    # counts and reset contact data to sentinel values.
    contacts_out.clear()

    # Launch the conversion kernel to convert Newton contacts to Kamino's format
    # NOTE: To reduce overhead, the total thread count is set to the smallest of
    # the number of contacts detected and the maximum capacity of the output contacts.
    wp.launch(
        kernel=_convert_contacts_newton_to_kamino,
        dim=max_converted_contacts,
        inputs=[
            wp.int32(model.world_count),
            wp.int32(max_converted_contacts),
            contacts_in.rigid_contact_count,
            contacts_in.rigid_contact_shape0,
            contacts_in.rigid_contact_shape1,
            contacts_in.rigid_contact_point0,
            contacts_in.rigid_contact_point1,
            contacts_in.rigid_contact_normal,
            contacts_in.rigid_contact_margin0,
            contacts_in.rigid_contact_margin1,
            contacts_in_force,
            model.shape_body,
            model.shape_world,
            model.shape_material_mu,
            model.shape_material_restitution,
            state.body_q,
        ],
        outputs=[
            contacts_out.model_active_contacts,
            contacts_out.world_active_contacts,
            contacts_out.wid,
            contacts_out.cid,
            contacts_out.gid_AB,
            contacts_out.bid_AB,
            contacts_out.position_A,
            contacts_out.position_B,
            contacts_out.gapfunc,
            contacts_out.frame,
            contacts_out.material,
            contacts_out.margins,
            contacts_out.key,
            contacts_out.reaction,
            contacts_out.remap,
        ],
        device=model.device,
    )


def convert_contacts_kamino_to_newton(
    model: Model,
    state: State,
    contacts_in: ContactsKamino,
    contacts_out: Contacts,
    clear_output: bool = False,
    convert_forces: bool = False,
) -> None:
    """
    Converts Kamino :class:`ContactsKamino` to Newton's :class:`Contacts` format.

    Newton's conventions for contact data are:

    - ``rigid_contact_normal[cid]`` points from ``shape0`` to ``shape1``.
    - ``force[cid]`` is the wrench applied on Newton's ``body0`` by Newton's
      ``body1``, expressed at the CoM of ``body0`` in world coordinates.

    This function operates in one of two modes selected by ``clear_output``:

    - ``clear_output=True``: the active-contacts path. The output contacts are
      cleared and repopulated from scratch with the currently active Kamino
      contacts; Newton's ``shape0/shape1`` are written to match Kamino's A/B
      ordering. If ``convert_forces`` is true, ``contacts_out.force`` is
      populated as well; otherwise it is left untouched.
    - ``clear_output=False``: the existing-contacts path. The geometric data
      already on ``contacts_out`` is preserved and only the per-contact
      reaction is converted into ``contacts_out.force``. This path requires
      ``contacts_in.remap`` (allocated by setting ``remappable=True`` on the
      :class:`ContactsKamino`) and ``contacts_out.force``, so it is only valid
      with ``convert_forces=True``.

    Args:
        model:
            The input :class:`Model` object providing shape and body information
            used to interpret Kamino's contact data and populate Newton's contact data.
        state:
            The input :class:`State` object providing ``body_q`` and ``body_com``
            used to transform contact points between world and body-local coordinates.
        contacts_in:
            The input :class:`ContactsKamino` object containing contact information to be converted.
        contacts_out:
            The output :class:`Contacts` object to populate with the converted contact data.
        clear_output:
            If ``True``, overwrite ``contacts_out`` from scratch with the active
            contacts in ``contacts_in``. If ``False``, only the per-contact
            reaction is converted and written into ``contacts_out.force`` using
            the preserved ``shape0``/``point0`` and the ``contacts_in.remap``.
        convert_forces:
            If ``True``, converts ``contacts_in.reaction`` into ``contacts_out.force``
            using Newton's wrench convention. Required when ``clear_output=False``;
            with ``clear_output=False`` and ``convert_forces=False`` the call is a no-op.
    """
    # Skip conversion if there are no contacts to convert or no capacity to store them.
    if contacts_in.model_max_contacts_host == 0 or contacts_out.rigid_contact_max == 0:
        return

    # Ensure that the model, state, contacts_in and
    # contacts_out are all on the same device.
    if (
        contacts_out.device != model.device
        or contacts_out.device != state.body_q.device
        or contacts_out.device != contacts_in.device
    ):
        raise ValueError(
            "All inputs must be on the same device: "
            f"model.device={model.device}, "
            f"state.device={state.body_q.device}, "
            f"contacts_in.device={contacts_in.device}, "
            f"contacts_out.device={contacts_out.device}"
        )

    # Issue warning to the user if the number of contacts to
    # convert exceeds the capacity of the output contacts.
    if contacts_in.model_max_contacts_host > contacts_out.rigid_contact_max:
        msg.warning(
            "Kamino `model_max_contacts_host` (%d) exceeds Newton `rigid_contact_max` (%d); contacts will be truncated.",
            contacts_in.model_max_contacts_host,
            contacts_out.rigid_contact_max,
        )

    # Skip conversion of contact forces if not requested
    contacts_out_force = contacts_out.force if convert_forces else None

    # Set the maximum number of contacts to convert to the smallest of the
    # number of contacts detected and the maximum capacity of the output contacts.
    # NOTE: To reduce overhead, the total thread count is set to the smallest of the
    # number of contacts detected and the maximum capacity of the output contacts.
    max_converted_contacts = min(contacts_in.model_max_contacts_host, contacts_out.rigid_contact_max)

    # Launch the conversion operations to convert Kamino contacts to Newton's format
    # depending on whether the output contacts are to be re-populated with only active
    # contacts or fill in solver-specific contact attributes for existing contacts.
    if clear_output:
        # Clear the output contacts before conversion
        # NOTE: This is necessary when we want to ensure that the
        # output contacts are populated only with active contacts.
        contacts_out.clear()

        # Launch the kernel to re-populate the output
        # from scratch and only with active contacts.
        wp.launch(
            kernel=_convert_active_contacts_kamino_to_newton,
            dim=max_converted_contacts,
            inputs=[
                wp.int32(max_converted_contacts),
                contacts_in.model_active_contacts,
                contacts_in.gid_AB,
                contacts_in.position_A,
                contacts_in.position_B,
                contacts_in.gapfunc,
                contacts_in.frame,
                contacts_in.reaction,
                contacts_in.margins,
                model.shape_body,
                model.body_com,
                state.body_q,
            ],
            outputs=[
                contacts_out.rigid_contact_count,
                contacts_out.rigid_contact_shape0,
                contacts_out.rigid_contact_shape1,
                contacts_out.rigid_contact_margin0,
                contacts_out.rigid_contact_margin1,
                contacts_out.rigid_contact_point0,
                contacts_out.rigid_contact_point1,
                contacts_out.rigid_contact_normal,
                contacts_out_force,
            ],
            device=model.device,
        )
    else:
        # When ``clear_output=False`` we only fill in solver-specific contact
        # attributes (currently the per-contact wrench) on top of pre-populated
        # geometric contact data. If forces are not being converted, there is
        # nothing to do, so this path becomes a no-op.
        if contacts_out_force is None:
            return

        # Conversion of forces on existing contacts requires the
        # ``ContactsKamino.remap`` array so we can recover Newton's original
        # per-contact body0/point0/shape0; without it we cannot determine the
        # destination index or whether an A<->B swap occurred at N->K time.
        if contacts_in.remap is None:
            raise ValueError(
                "`ContactsKamino.remap` is required when `clear_output=False`; "
                "construct `ContactsKamino` with `remappable=True`."
            )

        # Launch the kernel to fill in solver-specific
        # contact attributes for already populated contacts.
        wp.launch(
            kernel=_convert_existing_contacts_kamino_to_newton,
            dim=max_converted_contacts,
            inputs=[
                wp.int32(max_converted_contacts),
                contacts_in.model_active_contacts,
                contacts_in.gid_AB,
                contacts_in.frame,
                contacts_in.reaction,
                contacts_in.remap,
                contacts_out.rigid_contact_shape0,
                contacts_out.rigid_contact_point0,
                model.shape_body,
                model.body_com,
                state.body_q,
            ],
            outputs=[contacts_out_force],
            device=model.device,
        )
