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
from warp import DeviceLike as Devicelike


class Contacts:
    """
    Stores contact information for rigid and soft body collisions, to be consumed by a solver.

    This class manages buffers for contact data such as positions, normals, thicknesses, and shape indices
    for both rigid-rigid and soft-rigid contacts. The buffers are allocated on the specified device and can
    optionally require gradients for differentiable simulation.

    .. note::
        This class is a temporary solution and its interface may change in the future.
    """

    EXTENDED_ATTRIBUTES: frozenset[str] = frozenset(("force",))
    """
    Names of optional extended contact attributes that are not allocated by default.

    These can be requested via :meth:`newton.ModelBuilder.request_contact_attributes` or
    :meth:`newton.Model.request_contact_attributes` before calling :meth:`newton.Model.contacts` or
    :meth:`newton.CollisionPipeline.contacts`.

    See :ref:`extended_contact_attributes` for details and usage.
    """

    @classmethod
    def validate_extended_attributes(cls, attributes: tuple[str, ...]) -> None:
        """Validate names passed to request_contact_attributes().

        Only extended contact attributes listed in :attr:`EXTENDED_ATTRIBUTES` are accepted.

        Args:
            attributes: Tuple of attribute names to validate.

        Raises:
            ValueError: If any attribute name is not in :attr:`EXTENDED_ATTRIBUTES`.
        """
        if not attributes:
            return

        invalid = sorted(set(attributes).difference(cls.EXTENDED_ATTRIBUTES))
        if invalid:
            allowed = ", ".join(sorted(cls.EXTENDED_ATTRIBUTES))
            bad = ", ".join(invalid)
            raise ValueError(f"Unknown extended contact attribute(s): {bad}. Allowed: {allowed}.")

    def __init__(
        self,
        rigid_contact_max: int,
        soft_contact_max: int,
        requires_grad: bool = False,
        device: Devicelike = None,
        per_contact_shape_properties: bool = False,
        clear_buffers: bool = False,
        requested_attributes: set[str] | None = None,
    ):
        """
        Initialize Contacts storage.

        Args:
            rigid_contact_max: Maximum number of rigid contacts
            soft_contact_max: Maximum number of soft contacts
            requires_grad: Whether **soft** contact arrays require gradients for differentiable
                simulation.  Rigid contact arrays are always allocated without gradients because
                the narrow phase kernels do not support backward passes.  Soft contact arrays
                (body_pos, body_vel, normal) are allocated with ``requires_grad`` so that
                gradient-based optimisation can flow through particle-shape contacts.
            device: Device to allocate buffers on
            per_contact_shape_properties: Enable per-contact stiffness/damping/friction arrays
            clear_buffers: If True, clear() will zero all contact buffers (slower but conservative).
                If False (default), clear() only resets counts, relying on collision detection
                to overwrite active contacts. This is much faster (86-90% fewer kernel launches)
                and safe since solvers only read up to contact_count.
            requested_attributes: Set of extended contact attribute names to allocate.
                See :attr:`EXTENDED_ATTRIBUTES` for available options.
        """
        self.per_contact_shape_properties = per_contact_shape_properties
        self.clear_buffers = clear_buffers
        with wp.ScopedDevice(device):
            # Consolidated counter array to minimize kernel launches for zeroing
            # Layout: [rigid_contact_count, soft_contact_count]
            self._counter_array = wp.zeros(2, dtype=wp.int32)
            # Create sliced views for individual counters (no additional allocation)
            self.rigid_contact_count = self._counter_array[0:1]

            # rigid contacts — never requires_grad (narrow phase has enable_backward=False)
            self.rigid_contact_point_id = wp.zeros(rigid_contact_max, dtype=wp.int32)
            self.rigid_contact_shape0 = wp.full(rigid_contact_max, -1, dtype=wp.int32)
            self.rigid_contact_shape1 = wp.full(rigid_contact_max, -1, dtype=wp.int32)
            self.rigid_contact_point0 = wp.zeros(rigid_contact_max, dtype=wp.vec3)
            self.rigid_contact_point1 = wp.zeros(rigid_contact_max, dtype=wp.vec3)
            self.rigid_contact_offset0 = wp.zeros(rigid_contact_max, dtype=wp.vec3)
            self.rigid_contact_offset1 = wp.zeros(rigid_contact_max, dtype=wp.vec3)
            self.rigid_contact_normal = wp.zeros(rigid_contact_max, dtype=wp.vec3)
            self.rigid_contact_thickness0 = wp.zeros(rigid_contact_max, dtype=wp.float32)
            self.rigid_contact_thickness1 = wp.zeros(rigid_contact_max, dtype=wp.float32)
            self.rigid_contact_tids = wp.full(rigid_contact_max, -1, dtype=wp.int32)
            # to be filled by the solver (currently unused)
            self.rigid_contact_force = wp.zeros(rigid_contact_max, dtype=wp.vec3)

            # contact stiffness/damping/friction (only allocated if per_contact_shape_properties is enabled)
            if self.per_contact_shape_properties:
                self.rigid_contact_stiffness = wp.zeros(rigid_contact_max, dtype=wp.float32)
                self.rigid_contact_damping = wp.zeros(rigid_contact_max, dtype=wp.float32)
                self.rigid_contact_friction = wp.zeros(rigid_contact_max, dtype=wp.float32)
            else:
                self.rigid_contact_stiffness = None
                self.rigid_contact_damping = None
                self.rigid_contact_friction = None

            # soft contacts — requires_grad flows through here for differentiable simulation
            self.soft_contact_count = self._counter_array[1:2]
            self.soft_contact_particle = wp.full(soft_contact_max, -1, dtype=int)
            self.soft_contact_shape = wp.full(soft_contact_max, -1, dtype=int)
            self.soft_contact_body_pos = wp.zeros(soft_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.soft_contact_body_vel = wp.zeros(soft_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.soft_contact_normal = wp.zeros(soft_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.soft_contact_tids = wp.full(soft_contact_max, -1, dtype=int)

            # Extended contact attributes (optional, allocated on demand)
            self.force: wp.array | None = None
            """Contact forces (spatial), shape (rigid_contact_max + soft_contact_max,), dtype :class:`spatial_vector`.
            Force and torque exerted on body0 by body1, referenced to the center of mass (COM) of body0, and in world frame, where body0 and body1 are the bodies of shape0 and shape1.
            First three entries: linear force; last three entries: torque (moment).
            When both rigid and soft contacts are present, soft contact forces follow rigid contact forces.

            This is an extended contact attribute; see :ref:`extended_contact_attributes` for more information.
            """
            if requested_attributes and "force" in requested_attributes:
                total_contacts = rigid_contact_max + soft_contact_max
                self.force = wp.zeros(total_contacts, dtype=wp.spatial_vector, requires_grad=requires_grad)

        self.requires_grad = requires_grad

        self.rigid_contact_max = rigid_contact_max
        self.soft_contact_max = soft_contact_max

    def clear(self):
        """
        Clear contact data, resetting counts and optionally clearing all buffers.

        By default (clear_buffers=False), only resets contact counts. This is highly optimized,
        requiring just 1 kernel launch. Collision detection overwrites all data up to the new
        contact_count, and solvers only read up to count, so clearing stale data is unnecessary.

        If clear_buffers=True (conservative mode), performs full buffer clearing with sentinel
        values and zeros. This requires 7-10 kernel launches but may be useful for debugging.
        """
        # Clear all counters with a single kernel launch (consolidated counter array)
        self._counter_array.zero_()

        if self.clear_buffers:
            # Conservative path: clear all buffers (7-10 kernel launches)
            # This is slower but may be useful for debugging or special cases
            self.rigid_contact_shape0.fill_(-1)
            self.rigid_contact_shape1.fill_(-1)
            self.rigid_contact_tids.fill_(-1)
            self.rigid_contact_force.zero_()

            if self.force is not None:
                self.force.zero_()

            if self.per_contact_shape_properties:
                self.rigid_contact_stiffness.zero_()
                self.rigid_contact_damping.zero_()
                self.rigid_contact_friction.zero_()

            self.soft_contact_particle.fill_(-1)
            self.soft_contact_shape.fill_(-1)
            self.soft_contact_tids.fill_(-1)
        # else: Optimized path (default) - only counter clear needed
        #   Collision detection overwrites all active contacts [0, contact_count)
        #   Solvers only read [0, contact_count), so stale data is never accessed

    @property
    def device(self):
        """
        Returns the device on which the contact buffers are allocated.
        """
        return self.rigid_contact_count.device
