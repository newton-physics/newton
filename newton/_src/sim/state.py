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

    def __init__(self) -> None:
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
        First three entries: linear velocity; last three: angular velocity."""

        self.body_q_prev: wp.array | None = None
        """Previous rigid body transforms for finite-difference velocity computation."""

        self.body_qdd: wp.array | None = None
        """Rigid body accelerations (spatial), shape (body_count,), dtype :class:`spatial_vector`.
        First three entries: linear acceleration; last three: angular acceleration. Extended state attribute.
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
        First three entries: linear force; last three: torque. Extended state attribute.
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

        Args:
            other: The source State object to copy from.

        Raises:
            ValueError: If the states have mismatched attributes (one has an array where the other is None).
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
