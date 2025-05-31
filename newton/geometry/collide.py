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
Collision handling functions and kernels.
"""

from __future__ import annotations

import warp as wp

from .contacts import Contacts
from .kernels import broadphase_collision_pairs, create_soft_contacts, handle_contact_pairs


class CollisionPipeline:
    def __init__(
        self,
        shape_count: int,
        shape_pairs_filtered: wp.array(dtype=int, ndim=2),
        rigid_max_contacts_per_pair: int,
        rigid_contact_margin: float,
        soft_contact_max: int,
        soft_contact_margin: float,
        edge_sdf_iter: int,
        iterate_mesh_vertices: bool,
        requires_grad: bool,
    ):
        # will be allocated during collide
        self.contacts = None

        self.shape_count = shape_count
        self.shape_pairs_filtered = shape_pairs_filtered
        self.shape_pairs_max = len(self.shape_pairs_filtered)

        self.rigid_contact_margin = rigid_contact_margin
        self.rigid_contact_max = self.shape_pairs_max * rigid_max_contacts_per_pair

        # used during broadphase collision handling
        self.rigid_pair_shape0 = wp.empty(self.rigid_contact_max, dtype=wp.int32)
        self.rigid_pair_shape1 = wp.empty(self.rigid_contact_max, dtype=wp.int32)
        self.rigid_pair_point_limit = None  # wp.empty(self.shape_count ** 2, dtype=wp.int32)
        self.rigid_pair_point_count = None  # wp.empty(self.shape_count ** 2, dtype=wp.int32)
        self.rigid_pair_point_id = wp.empty(self.rigid_contact_max, dtype=wp.int32)

        self.soft_contact_margin = soft_contact_margin
        self.soft_contact_max = soft_contact_max

        self.iterate_mesh_vertices = iterate_mesh_vertices
        self.requires_grad = requires_grad
        self.edge_sdf_iter = edge_sdf_iter

    def collide(
        self,
        shape_type: wp.array(dtype=wp.int32),
        shape_is_solid: wp.array(dtype=bool),
        shape_thickness: wp.array(dtype=float),
        shape_source: wp.array(dtype=wp.uint64),
        shape_scale: wp.array(dtype=wp.vec3),
        shape_filter: wp.array(dtype=int),
        shape_radius: wp.array(dtype=float),
        shape_body: wp.array(dtype=float),
        shape_transform: wp.array(dtype=wp.transform),
        body_q: wp.array(dtype=wp.transform),
        particle_q: wp.array(dtype=wp.vec3),
        particle_radius: wp.array(dtype=float),
        particle_flags: wp.array(dtype=wp.uint32),
    ) -> Contacts:
        # allocate new contact memory for contacts if we need gradients
        if self.contacts is None or self.requires_grad:
            self.contacts = Contacts(
                self.rigid_contact_max,
                self.soft_contact_max,
                requires_grad=self.requires_grad,
                device=shape_type.device,
            )

        # output contacts buffer
        contacts = self.contacts

        shape_count = self.shape_count
        particle_count = len(particle_q) if particle_q else 0

        with wp.ScopedTimer("collide", False):
            # generate soft contacts for particles and shapes
            if particle_q and shape_count > 0:
                # clear old count
                contacts.soft_contact_count.zero_()
                wp.launch(
                    kernel=create_soft_contacts,
                    dim=particle_count * shape_count,
                    inputs=[
                        particle_q,
                        particle_radius,
                        particle_flags,
                        body_q,
                        shape_transform,
                        shape_body,
                        shape_type,
                        shape_scale,
                        shape_source,
                        self.soft_contact_margin,
                        self.soft_contact_max,
                        shape_count,
                    ],
                    outputs=[
                        contacts.soft_contact_count,
                        contacts.soft_contact_particle,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_body_vel,
                        contacts.soft_contact_normal,
                        contacts.soft_contact_tids,
                    ],
                    device=contacts.device,
                )

            # generate rigid contacts for shapes
            if self.shape_pairs_filtered is not None:
                # clear old count
                contacts.rigid_contact_count.zero_()
                self.rigid_pair_shape0.fill_(-1)
                self.rigid_pair_shape1.fill_(-1)

                wp.launch(
                    kernel=broadphase_collision_pairs,
                    dim=len(self.shape_pairs_filtered),
                    inputs=[
                        body_q,
                        shape_transform,
                        shape_body,
                        shape_type,
                        shape_scale,
                        shape_source,
                        self.shape_pairs_filtered,
                        shape_radius,
                        shape_count,
                        self.rigid_contact_max,
                        self.rigid_contact_margin,
                        self.rigid_contact_max,
                        self.iterate_mesh_vertices,
                    ],
                    outputs=[
                        contacts.rigid_contact_count,
                        self.rigid_pair_shape0,
                        self.rigid_pair_shape1,
                        self.rigid_pair_point_id,
                        self.rigid_pair_point_limit,
                    ],
                    record_tape=False,
                    device=contacts.device,
                )

                contacts.clear()
                if self.rigid_pair_point_count is not None:
                    self.rigid_pair_point_count.zero_()

                wp.launch(
                    kernel=handle_contact_pairs,
                    dim=self.rigid_contact_max,
                    inputs=[
                        body_q,
                        shape_transform,
                        shape_body,
                        shape_type,
                        shape_scale,
                        shape_source,
                        shape_thickness,
                        shape_count,
                        self.rigid_contact_margin,
                        self.rigid_pair_shape0,
                        self.rigid_pair_shape1,
                        self.rigid_pair_point_id,
                        self.rigid_pair_point_limit,
                        self.edge_sdf_iter,
                    ],
                    outputs=[
                        contacts.rigid_contact_count,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,
                        contacts.rigid_contact_offset0,
                        contacts.rigid_contact_offset1,
                        contacts.rigid_contact_normal,
                        contacts.rigid_contact_thickness,
                        self.rigid_pair_point_count,
                        contacts.rigid_contact_tids,
                    ],
                    device=contacts.device,
                )

            return contacts
