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
from warp.context import Devicelike


class Contacts:
    """Provides contact information to be consumed by a solver.
    Stores the contact distance, position, frame, and geometry for each contact point.

    .. note::
        This class definition is only a temporary solution and will change significantly in the future.
    """

    def __init__(
        self,
        rigid_contact_max: int,
        soft_contact_max: int,
        requires_grad: bool = False,
        device: Devicelike = None,
    ):
        with wp.ScopedDevice(device):
            # rigid contacts
            self.rigid_contact_count = wp.zeros(1, dtype=wp.int32)
            self.rigid_contact_point_id = wp.zeros(rigid_contact_max, dtype=wp.int32)
            self.rigid_contact_shape0 = wp.full(rigid_contact_max, -1, dtype=wp.int32)
            self.rigid_contact_shape1 = wp.full(rigid_contact_max, -1, dtype=wp.int32)
            self.rigid_contact_point0 = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.rigid_contact_point1 = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.rigid_contact_offset0 = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.rigid_contact_offset1 = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.rigid_contact_normal = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.rigid_contact_thickness0 = wp.zeros(rigid_contact_max, dtype=wp.float32, requires_grad=requires_grad)
            self.rigid_contact_thickness1 = wp.zeros(rigid_contact_max, dtype=wp.float32, requires_grad=requires_grad)
            self.rigid_contact_tids = wp.full(rigid_contact_max, -1, dtype=wp.int32)
            # to be filled by the solver (currently unused)
            self.rigid_contact_force = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=requires_grad)

            # soft contacts
            self.soft_contact_count = wp.zeros(1, dtype=wp.int32)
            self.soft_contact_particle = wp.full(soft_contact_max, -1, dtype=int)
            self.soft_contact_shape = wp.full(soft_contact_max, -1, dtype=int)
            self.soft_contact_body_pos = wp.zeros(soft_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.soft_contact_body_vel = wp.zeros(soft_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.soft_contact_normal = wp.zeros(soft_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.soft_contact_tids = wp.full(soft_contact_max, -1, dtype=int)

        self.requires_grad = requires_grad

        self.rigid_contact_max = rigid_contact_max
        self.soft_contact_max = soft_contact_max

    def clear(self):
        """Clear all contacts."""
        self.rigid_contact_count.zero_()
        self.rigid_contact_shape0.fill_(-1)
        self.rigid_contact_shape1.fill_(-1)
        self.rigid_contact_tids.fill_(-1)
        self.rigid_contact_force.zero_()

        self.soft_contact_count.zero_()
        self.soft_contact_particle.fill_(-1)
        self.soft_contact_shape.fill_(-1)
        self.soft_contact_tids.fill_(-1)

    @property
    def device(self):
        return self.rigid_contact_count.device
