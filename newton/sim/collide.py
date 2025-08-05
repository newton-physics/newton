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

from ..core.types import Devicelike
from ..geometry.kernels import (
    broadphase_collision_pairs,
    count_contact_points,
    create_soft_contacts,
    handle_contact_pairs,
)
from .contacts import Contacts
from .model import Model
from .state import State


def count_rigid_contact_points(model: Model):
    """
    Counts the maximum number of rigid contact points that need to be allocated.

    :returns:
        - count (int): Potential number of rigid contact points
    """

    # calculate the potential number of shape pair contact points
    contact_count = wp.zeros(1, dtype=wp.int32, device=model.device)
    wp.launch(
        kernel=count_contact_points,
        dim=model.shape_contact_pair_count,
        inputs=[
            model.shape_contact_pairs,
            model.shape_type,
            model.shape_scale,
            model.shape_source_ptr,
        ],
        outputs=[contact_count],
        device=model.device,
        record_tape=False,
    )
    counts = contact_count.numpy()
    return int(counts[0])


class CollisionPipeline:
    def __init__(
        self,
        shape_count: int,
        particle_count: int,
        shape_pairs_filtered: wp.array(dtype=wp.vec2i),
        rigid_contact_max: int | None = None,
        rigid_contact_max_per_pair: int = 10,
        rigid_contact_margin: float = 0.01,
        soft_contact_max: int | None = None,
        soft_contact_margin: float = 0.01,
        edge_sdf_iter: int = 10,
        iterate_mesh_vertices: bool = True,
        requires_grad: bool = False,
        device: Devicelike = None,
    ):
        # will be allocated during collide
        self.contacts = None

        self.shape_count = shape_count
        self.shape_pairs_filtered = shape_pairs_filtered
        self.shape_pairs_max = len(self.shape_pairs_filtered)

        self.rigid_contact_margin = rigid_contact_margin
        if rigid_contact_max is not None:
            self.rigid_contact_max = rigid_contact_max
        else:
            self.rigid_contact_max = self.shape_pairs_max * rigid_contact_max_per_pair

        # used during broadphase collision handling
        with wp.ScopedDevice(device):
            self.rigid_pair_shape0 = wp.empty(self.rigid_contact_max, dtype=wp.int32)
            self.rigid_pair_shape1 = wp.empty(self.rigid_contact_max, dtype=wp.int32)
            self.rigid_pair_point_limit = None  # wp.empty(self.shape_count ** 2, dtype=wp.int32)
            self.rigid_pair_point_count = None  # wp.empty(self.shape_count ** 2, dtype=wp.int32)
            self.rigid_pair_point_id = wp.empty(self.rigid_contact_max, dtype=wp.int32)

        if soft_contact_max is None:
            soft_contact_max = shape_count * particle_count
        self.soft_contact_margin = soft_contact_margin
        self.soft_contact_max = soft_contact_max

        self.iterate_mesh_vertices = iterate_mesh_vertices
        self.requires_grad = requires_grad
        self.edge_sdf_iter = edge_sdf_iter

    @classmethod
    def from_model(
        cls,
        model: Model,
        rigid_contact_max_per_pair: int | None = None,
        rigid_contact_margin: float = 0.01,
        soft_contact_max: int | None = None,
        soft_contact_margin: float = 0.01,
        edge_sdf_iter: int = 10,
        iterate_mesh_vertices: bool = True,
        requires_grad: bool | None = None,
    ) -> CollisionPipeline:
        rigid_contact_max = None
        if rigid_contact_max_per_pair is None:
            # count the number of contacts
            rigid_contact_max = model.rigid_contact_max
            rigid_contact_max_per_pair = 0
        if requires_grad is None:
            requires_grad = model.requires_grad
        return CollisionPipeline(
            model.shape_count,
            model.particle_count,
            model.shape_contact_pairs,
            rigid_contact_max,
            rigid_contact_max_per_pair,
            rigid_contact_margin,
            soft_contact_max,
            soft_contact_margin,
            edge_sdf_iter,
            iterate_mesh_vertices,
            requires_grad,
            model.device,
        )

    def collide(self, model: Model, state: State) -> Contacts:
        # allocate new contact memory for contacts if we need gradients
        if self.contacts is None or self.requires_grad:
            self.contacts = Contacts(
                self.rigid_contact_max,
                self.soft_contact_max,
                requires_grad=self.requires_grad,
                device=model.device,
            )
        else:
            self.contacts.clear()

        # output contacts buffer
        contacts = self.contacts

        shape_count = self.shape_count
        particle_count = len(state.particle_q) if state.particle_q else 0

        # generate soft contacts for particles and shapes
        if state.particle_q and shape_count > 0:
            wp.launch(
                kernel=create_soft_contacts,
                dim=particle_count * shape_count,
                inputs=[
                    state.particle_q,
                    model.particle_radius,
                    model.particle_flags,
                    state.body_q,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_type,
                    model.shape_scale,
                    model.shape_source_ptr,
                    self.soft_contact_margin,
                    self.soft_contact_max,
                    shape_count,
                    model.shape_flags,
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
            self.rigid_pair_shape0.fill_(-1)
            self.rigid_pair_shape1.fill_(-1)

            wp.launch(
                kernel=broadphase_collision_pairs,
                dim=len(self.shape_pairs_filtered),
                inputs=[
                    state.body_q,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_type,
                    model.shape_scale,
                    model.shape_source_ptr,
                    self.shape_pairs_filtered,
                    model.shape_collision_radius,
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

            # clear old count
            contacts.rigid_contact_count.zero_()
            if self.rigid_pair_point_count is not None:
                self.rigid_pair_point_count.zero_()

            wp.launch(
                kernel=handle_contact_pairs,
                dim=self.rigid_contact_max,
                inputs=[
                    state.body_q,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_type,
                    model.shape_scale,
                    model.shape_source_ptr,
                    model.shape_thickness,
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
                    contacts.rigid_contact_thickness0,
                    contacts.rigid_contact_thickness1,
                    self.rigid_pair_point_count,
                    contacts.rigid_contact_tids,
                ],
                device=contacts.device,
            )

        return contacts

    @property
    def device(self):
        return self.rigid_pair_shape0.device


__all__ = [
    "CollisionPipeline",
    "count_rigid_contact_points",
]
