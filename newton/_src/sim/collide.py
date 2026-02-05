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
    create_soft_contacts,
    generate_handle_contact_pairs_kernel,
)
from .contacts import Contacts
from .model import Model
from .state import State


class CollisionPipeline:
    """
    CollisionPipeline manages collision detection and contact generation for a simulation.

    This class is responsible for allocating and managing buffers for collision detection,
    generating rigid and soft contacts between shapes and particles, and providing an interface
    for running the collision pipeline on a given simulation state.
    """

    def __init__(
        self,
        shape_count: int,
        particle_count: int,
        shape_pairs_filtered: wp.array(dtype=wp.vec2i),
        rigid_contact_max: int | None = None,
        rigid_contact_max_per_pair: int | None = None,
        soft_contact_max: int | None = None,
        soft_contact_margin: float = 0.01,
        edge_sdf_iter: int = 10,
        requires_grad: bool = False,
        device: Devicelike = None,
    ):
        """
        Initialize the CollisionPipeline.

        Args:
            shape_count (int): Number of shapes in the simulation.
            particle_count (int): Number of particles in the simulation.
            shape_pairs_filtered (wp.array): Array of filtered shape pairs to consider for collision.
            rigid_contact_max (int | None, optional): Maximum number of rigid contacts to allocate.
                If None, computed as shape_pairs_max * rigid_contact_max_per_pair.
            rigid_contact_max_per_pair (int | None, optional): Maximum number of contact points per shape pair.
                If None or <= 0, no limit is applied.
            soft_contact_max (int | None, optional): Maximum number of soft contacts to allocate.
                If None, computed as shape_count * particle_count.
            soft_contact_margin (float, optional): Margin for soft contact generation. Defaults to 0.01.
            edge_sdf_iter (int, optional): Number of iterations for edge SDF collision. Defaults to 10.
            requires_grad (bool, optional): Whether to enable gradient computation. Defaults to False.
            device (Devicelike, optional): The device on which to allocate arrays and perform computation.

        Note:
            Contact margins for rigid contacts are now controlled per-shape via ``model.shape_contact_margin``.
        """

        self.shape_count = shape_count
        self.shape_pairs_filtered = shape_pairs_filtered
        self.shape_pairs_max = len(self.shape_pairs_filtered)

        if rigid_contact_max_per_pair is None or rigid_contact_max_per_pair <= 0:
            rigid_contact_max_per_pair = 0
        self.rigid_contact_max_per_pair = rigid_contact_max_per_pair
        if rigid_contact_max is not None or rigid_contact_max_per_pair == 0:
            self.rigid_contact_max = rigid_contact_max
        else:
            self.rigid_contact_max = self.shape_pairs_max * rigid_contact_max_per_pair

        # Allocate buffers for broadphase collision handling
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

        self.requires_grad = requires_grad
        self.edge_sdf_iter = edge_sdf_iter

        self.handle_contact_pairs_kernel = generate_handle_contact_pairs_kernel(requires_grad)

    @classmethod
    def from_model(
        cls,
        model: Model,
        rigid_contact_max_per_pair: int | None = None,
        soft_contact_max: int | None = None,
        soft_contact_margin: float = 0.01,
        edge_sdf_iter: int = 10,
        requires_grad: bool | None = None,
    ) -> CollisionPipeline:
        """
        Create a CollisionPipeline instance from a Model.

        Args:
            model (Model): The simulation model.
            rigid_contact_max_per_pair (int | None, optional): Maximum number of contact points per shape pair.
                If None, uses :attr:`newton.Model.rigid_contact_max` and sets per-pair to 0 (which indicates no limit).
            soft_contact_max (int | None, optional): Maximum number of soft contacts to allocate.
            soft_contact_margin (float, optional): Margin for soft contact generation. Defaults to 0.01.
            edge_sdf_iter (int, optional): Number of iterations for edge SDF collision. Defaults to 10.
            requires_grad (bool | None, optional): Whether to enable gradient computation. If None, uses model.requires_grad.

        Returns:
            CollisionPipeline: The constructed collision pipeline.

        Note:
            Contact margins for rigid contacts are read from ``model.shape_contact_margin`` array.
        """
        rigid_contact_max = None
        if rigid_contact_max_per_pair is None:
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
            soft_contact_max,
            soft_contact_margin,
            edge_sdf_iter,
            requires_grad,
            model.device,
        )

    def contacts(self, model: Model) -> Contacts:
        """
        Allocate and return a new :class:`Contacts` object for the given model.

        Args:
            model (Model): The simulation model.

        Returns:
            Contacts: A newly allocated contacts buffer sized for this pipeline.
        """
        return Contacts(
            self.rigid_contact_max,
            self.soft_contact_max,
            requires_grad=self.requires_grad,
            device=model.device,
        )

    def collide(self, model: Model, state: State, contacts: Contacts):
        """
        Run collision detection and populate the contacts buffer.

        Args:
            model (Model): The simulation model.
            state (State): The current simulation state.
            contacts (Contacts): The contacts buffer to populate (will be cleared first).
        """
        contacts.clear()

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
                    model.particle_world,
                    state.body_q,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_type,
                    model.shape_scale,
                    model.shape_source_ptr,
                    model.shape_world,
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
                    model.shape_contact_margin,
                    self.rigid_contact_max_per_pair,
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
                kernel=self.handle_contact_pairs_kernel,
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
                    model.shape_contact_margin,
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
        """
        Returns the device on which the collision pipeline's buffers are allocated.

        Returns:
            The device associated with the pipeline's buffers.
        """
        return self.rigid_pair_shape0.device


__all__ = [
    "CollisionPipeline",
]
