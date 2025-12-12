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

"""Defines the State container for Kamino."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp


@dataclass
class State:
    """
    Represents the time-varying state of a :class:`Model` in a simulation.

    The State object holds all dynamic quantities that change over time during simulation,
    such as rigid body poses, twists, and wrenches, as well as joint coordinates, velocities,
    and constraint forces.

    State objects are typically created via :meth:`kamino.Model.state()` and are used to
    store and update the simulation's current configuration and derived data.

    For constrained rigid multi-body system, the state is defined formally using either:
    1. maximal-coordinates, as the absolute poses and twists of all bodies expressed in world coordinates, or
    2. minimal-coordinates, as the joint coordinates and velocities along with the
       pose and twist of a base body when it is a so-called "floating-base" system.

    In Kamino, we formally adopt the maximal-coordinate formulation in order to compute the physics of the
    system, but we are also interested in the state of the joints for the purposes of control and analysis.

    Thus, this container incorporates the data of both representations, and in addition also includes the per-body
    total (i.e. net) wrenches expressed in world coordinates, as well as the joint constraint forces. Thus, it
    provides a complete description of the dynamic state of the constrained rigid multi-body system.

    We adopt the following notational conventions for the state attributes:
    - Generalized coordinates, whether maximal or minimal, are universally denoted by ``q``
    - Generalized velocities for bodies are denoted by ``u`` since they are twists
    - Generalized velocities for joints are denoted by ``dq`` since they are time-derivatives of ``q``
    - Wrenches (forces + torques) are denoted by ``w``
    - Constraint forces are denoted by ``lambda`` since they are effectively Lagrange multipliers
    - Subscripts ``_i`` denote body-indexed quantities, e.g. :attr:`q_i`, :attr:`u_i`, :attr:`w_i`.
    - Subscripts ``_j`` denote joint-indexed quantities, e.g. :attr:`q_j`, :attr:`dq_j`, :attr:`lambda_j`.
    """

    q_i: wp.array | None = None
    """
    Array of absolute body CoM poses expressed in world coordinates.\n
    Each element is a 7D transform consisting of a 3D position + 4D unit quaternion.\n
    Shape is ``(num_bodies,)`` and dtype is :class:`transformf`.
    """

    u_i: wp.array | None = None
    """
    Array of absolute body CoM twists expressed in world coordinates.\n
    Each element is a 6D vector comprising a 3D linear + 3D angular components.\n
    Shape is ``(num_bodies,)`` and dtype is :class:`vec6f`.
    """

    w_i: wp.array | None = None
    """
    Array of body CoM wrenches expressed in world coordinates.\n
    Each element is a 6D vector comprising a 3D linear + 3D angular components.\n
    Shape is ``(num_bodies,)`` and dtype is :class:`vec6f`.
    """

    q_j: wp.array | None = None
    """
    Array of generalized joint coordinates.\n
    Shape is ``(sum(c_j),)`` and dtype is :class:`float32`,\n
    where ``c_j`` is the number of coordinates encoding the configuration of each joint ``j``.
    """

    dq_j: wp.array | None = None
    """
    Array of generalized joint velocities.\n
    Shape is ``(sum(d_j),)`` and dtype is :class:`float32`,\n
    where ``d_j`` is the number of DoFs defining the velocity of each joint ``j``.
    """

    lambda_j: wp.array | None = None
    """
    Array of generalized joint constraint forces.\n
    Shape is ``(sum(m_j),)`` and dtype is :class:`float32`,\n
    where ``m_j`` is the number of constraints enforced by each joint ``j``.
    """

    def copy_to(self, other: State) -> None:
        """
        Copy the State data to another State object.

        Args:
            other: The target State object to copy data into.
        """
        other.copy_from(self)

    def copy_from(self, other: State) -> None:
        """
        Copy the State data from another State object.

        Args:
            other: The source State object to copy data from.
        """
        if self.q_i is None or other.q_i is None:
            raise ValueError("Error copying from/to uninitialized State")

        wp.copy(self.q_i, other.q_i)
        wp.copy(self.u_i, other.u_i)
        wp.copy(self.w_i, other.w_i)
        wp.copy(self.q_j, other.q_j)
        wp.copy(self.dq_j, other.dq_j)
        wp.copy(self.lambda_j, other.lambda_j)
