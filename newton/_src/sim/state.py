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

from __future__ import annotations

import warp as wp


class State:
    """
    Represents the time-varying state of a :class:`Model` in a simulation.

    The State object holds all dynamic quantities that change over time during simulation,
    such as particle and rigid body positions, velocities, and forces, as well as joint coordinates.

    State objects are typically created via :meth:`newton.Model.state()` and are used to
    store and update the simulation's current configuration and derived data.
    """

    EXTENDED_STATE_ATTRIBUTES: frozenset[str] = frozenset(
        (
            "body_qdd",
            "body_parent_f",
        )
    )
    """
    Names of optional extended state attributes that are not allocated by default.

    These can be requested via :meth:`newton.ModelBuilder.request_state_attributes` or
    :meth:`newton.Model.request_state_attributes` before calling :meth:`newton.Model.state`.

    See :ref:`extended_state_attributes` for details and usage.
    """

    @classmethod
    def validate_extended_state_attributes(cls, attributes: tuple[str, ...]) -> None:
        """Validate names passed to request_state_attributes().

        Only extended state attributes listed in :attr:`EXTENDED_STATE_ATTRIBUTES` are accepted.
        """
        if not attributes:
            return

        invalid = sorted(set(attributes).difference(cls.EXTENDED_STATE_ATTRIBUTES))
        if invalid:
            allowed = ", ".join(sorted(cls.EXTENDED_STATE_ATTRIBUTES))
            bad = ", ".join(invalid)
            raise ValueError(f"Unknown extended State attribute(s): {bad}. Allowed extended attributes: {allowed}.")

    def __init__(self) -> None:
        """
        Initialize an empty State object.
        To ensure that the attributes are properly allocated create the State object via :meth:`newton.Model.state` instead.
        """

        self.particle_q: wp.array | None = None
        """3D positions of particles, shape (particle_count,), dtype :class:`vec3`."""

        self.particle_qd: wp.array | None = None
        """3D velocities of particles, shape (particle_count,), dtype :class:`vec3`."""

        self.particle_f: wp.array | None = None
        """3D forces on particles, shape (particle_count,), dtype :class:`vec3`."""

        self.body_q: wp.array | None = None
        """Rigid body transforms (7-DOF), shape (body_count,), dtype :class:`transform`."""

        self.body_qd: wp.array | None = None
        """Rigid body velocities (spatial), shape (body_count,), dtype :class:`spatial_vector`.
        First three entries: linear velocity relative to the body's center of mass in world frame; last three: angular velocity in world frame.
        See :ref:`Twist conventions in Newton <Twist conventions>` for more information."""

        self.body_q_prev: wp.array | None = None
        """Previous rigid body transforms for finite-difference velocity computation."""

        self.body_qdd: wp.array | None = None
        """Rigid body accelerations (spatial), shape (body_count,), dtype :class:`spatial_vector`.
        First three entries: linear acceleration relative to the body's center of mass in world frame; last three: angular acceleration in world frame.

        This is an extended state attribute; see :ref:`extended_state_attributes` for more information.
        """

        self.body_f: wp.array | None = None
        """Rigid body forces (spatial), shape (body_count,), dtype :class:`spatial_vector`.
        First three entries: linear force; last three: torque.

        .. note::
            :attr:`body_f` represents external wrenches in world frame, measured at the body's center of mass (COM).
            The linear force component is applied at the COM, and the torque is about the COM.
        """

        self.body_parent_f: wp.array | None = None
        """Parent interaction forces, shape (body_count,), dtype :class:`spatial_vector`.
        First three entries: linear force; last three: torque.

        This is an extended state attribute; see :ref:`extended_state_attributes` for more information.
        """

        self.joint_q: wp.array | None = None
        """Generalized joint position coordinates, shape (joint_coord_count,), dtype float."""

        self.joint_qd: wp.array | None = None
        """Generalized joint velocity coordinates, shape (joint_dof_count,), dtype float."""

    def clear_forces(self) -> None:
        """
        Clear all force arrays (for particles and bodies) in the state object.

        Sets all entries of :attr:`particle_f` and :attr:`body_f` to zero, if present.
        """
        with wp.ScopedTimer("clear_forces", False):
            if self.particle_count:
                self.particle_f.zero_()

            if self.body_count:
                self.body_f.zero_()

    def assign(self, other: State) -> None:
        """
        Copies the array attributes of another State object into this one.

        This can be useful for swapping states in a simulation when using CUDA graphs.
        If the number of substeps is odd, the last state needs to be explicitly copied for the graph to be captured correctly:

        .. code-block:: python

            # Assume we are capturing the following simulation loop in a CUDA graph
            for i in range(sim_substeps):
                state_0.clear_forces()

                solver.step(state_0, state_1, control, contacts, sim_dt)

                # Swap states - handle CUDA graph case specially
                if sim_substeps % 2 == 1 and i == sim_substeps - 1:
                    # Swap states by copying the state arrays for graph capture
                    state_0.assign(state_1)
                else:
                    # We can just swap the state references
                    state_0, state_1 = state_1, state_0

        Args:
            other: The source State object to copy from.

        Raises:
            ValueError: If the states have mismatched attributes (one has an array allocated where the other is None).
        """
        attributes = set(self.__dict__).union(other.__dict__)

        for attr in attributes:
            val_self = getattr(self, attr, None)
            val_other = getattr(other, attr, None)

            if val_self is None and val_other is None:
                continue

            array_self = isinstance(val_self, wp.array)
            array_other = isinstance(val_other, wp.array)

            if not array_self and not array_other:
                continue

            if val_self is None or not array_self:
                raise ValueError(f"State is missing array for '{attr}' which is present in the other state.")

            if val_other is None or not array_other:
                raise ValueError(f"Other state is missing array for '{attr}' which is present in this state.")

            val_self.assign(val_other)

    @property
    def requires_grad(self) -> bool:
        """Indicates whether the state arrays have gradient computation enabled."""
        if self.particle_q:
            return self.particle_q.requires_grad
        if self.body_q:
            return self.body_q.requires_grad
        return False

    @property
    def body_count(self) -> int:
        """The number of bodies represented in the state."""
        if self.body_q is None:
            return 0
        return len(self.body_q)

    @property
    def particle_count(self) -> int:
        """The number of particles represented in the state."""
        if self.particle_q is None:
            return 0
        return len(self.particle_q)

    @property
    def joint_coord_count(self) -> int:
        """The number of generalized joint position coordinates represented in the state."""
        if self.joint_q is None:
            return 0
        return len(self.joint_q)

    @property
    def joint_dof_count(self) -> int:
        """The number of generalized joint velocity coordinates represented in the state."""
        if self.joint_qd is None:
            return 0
        return len(self.joint_qd)
