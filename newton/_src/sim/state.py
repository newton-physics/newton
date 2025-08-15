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
    """Time-varying state data for a :class:`Model`.

    Time-varying state data includes particle positions, velocities, rigid body states, and
    anything that is output from the solver as derived data, e.g.: forces.

    The exact attributes depend on the contents of the model. State objects should
    generally be created using the :func:`newton.Model.state()` function.
    """

    def __init__(self) -> None:
        self.particle_q: wp.array | None = None
        """Array of 3D particle positions with shape ``(particle_count,)`` and type :class:`vec3`."""

        self.particle_qd: wp.array | None = None
        """Array of 3D particle velocities with shape ``(particle_count,)`` and type :class:`vec3`."""

        self.particle_f: wp.array | None = None
        """Array of 3D particle forces with shape ``(particle_count,)`` and type :class:`vec3`."""

        self.body_q: wp.array | None = None
        """Array of body coordinates (7-dof transforms) in maximal coordinates with shape ``(body_count,)`` and type :class:`transform`."""

        self.body_qd: wp.array | None = None
        """Array of body velocities in maximal coordinates (first three entries represent angular velocity,
        last three entries represent linear velocity) with shape ``(body_count,)`` and type :class:`spatial_vector`.
        """

        self.body_f: wp.array | None = None
        """Array of body forces in maximal coordinates (first three entries represent torque, last three
        entries represent linear force) with shape ``(body_count,)`` and type :class:`spatial_vector`.

        .. note::
            :attr:`body_f` represents external wrenches in world frame and denotes wrenches measured w.r.t.
            to the body's center of mass for all solvers except :class:`~newton.solvers.SolverFeatherstone`, which
            assumes the wrenches are measured w.r.t. world origin.
        """

        self.joint_q: wp.array | None = None
        """Array of generalized joint coordinates with shape ``(joint_coord_count,)`` and type ``float``."""

        self.joint_qd: wp.array | None = None
        """Array of generalized joint velocities with shape ``(joint_dof_count,)`` and type ``float``."""

    def clear_forces(self) -> None:
        """Clear all forces (for particles and bodies) in the state object."""
        with wp.ScopedTimer("clear_forces", False):
            if self.particle_count:
                self.particle_f.zero_()

            if self.body_count:
                self.body_f.zero_()

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
