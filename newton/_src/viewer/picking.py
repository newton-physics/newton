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

import numpy as np
import warp as wp

import newton

from ..geometry import raycast
from .kernels import apply_picking_force_kernel, compute_pick_state_kernel, update_pick_target_kernel


class Picking:
    """
    Picking system.

    Allows to pick a body in the viewer by right clicking on it and dragging the mouse.
    This can be used to move objects around in the viewer, a typical use case is to check for solver resilience or
    see how well a RL policy is coping with disturbances.
    """

    def __init__(self, model: newton.Model, pick_stiffness: float = 500.0, pick_damping: float = 50.0) -> None:
        """
        Initializes the picking system.

        Args:
            model (newton.Model): The model to pick from.
            pick_stiffness (float): The stiffness that will be used to compute the force applied to the picked body.
            pick_damping (float): The damping that will be used to compute the force applied to the picked body.
        """
        self.model = model
        self.pick_stiffness = pick_stiffness
        self.pick_damping = pick_damping

        self.min_dist = None
        self.min_index = None
        self.min_body_index = None
        self.lock = None
        self._contact_points0 = None
        self._contact_points1 = None
        self._debug = False

        # picking state
        if model and model.device.is_cuda:
            self.pick_body = wp.array([-1], dtype=int, pinned=True)
        else:
            self.pick_body = wp.array([-1], dtype=int, device="cpu")
        # pick_state array format (stored in a warp array for graph capture support):
        # [0:3] - pick point in local space (vec3)
        # [3:6] - pick target point in world space (vec3)
        # [6] - pick spring stiffness
        # [7] - pick spring damping
        # [8:11] - original mouse cursor target in world space (vec3)
        # [11:14] - current world space picked point on geometry (vec3)
        pick_state_np = np.zeros(14, dtype=np.float32)
        if model:
            pick_state_np[6] = pick_stiffness
            pick_state_np[7] = pick_damping
        self.pick_state = wp.array(pick_state_np, dtype=float, device=model.device if model else "cpu")

        self.pick_dist = 0.0
        self.picking_active = False

        self._default_on_mouse_drag = None

    def _apply_picking_force(self, state: newton.State) -> None:
        """
        Applies a force to the picked body.

        Args:
            state (newton.State): The simulation state.
        """
        if self.model is None:
            return

        # Launch kernel always because of graph capture
        wp.launch(
            kernel=apply_picking_force_kernel,
            dim=1,
            inputs=[
                state.body_q,
                state.body_qd,
                state.body_f,
                self.pick_body,
                self.pick_state,
                self.model.body_com,
                self.model.body_mass,
            ],
            device=self.model.device,
        )

    def is_picking(self) -> bool:
        """Checks if picking is active.

        Returns:
            bool: True if picking is active, False otherwise.
        """
        return self.picking_active

    def release(self) -> None:
        """Releases the picking."""
        self.pick_body.fill_(-1)
        self.picking_active = False

    def update(self, ray_start: wp.vec3f, ray_dir: wp.vec3f) -> None:
        """
        Updates the picking target.

        This function is used to track the force that needs to be applied to the picked body as the mouse is dragged.

        Args:
            ray_start (wp.vec3f): The start point of the ray.
            ray_dir (wp.vec3f): The direction of the ray.
        """
        if not self.is_picking():
            return

        wp.launch(
            kernel=update_pick_target_kernel,
            dim=1,
            inputs=[
                ray_start,
                ray_dir,
                self.pick_state,
            ],
            device=self.model.device,
        )

    def pick(self, state: newton.State, ray_start: wp.vec3f, ray_dir: wp.vec3f) -> None:
        """
        Picks the selected geometry and computes the initial state of the picking. I.e. the force that
        will be applied to the picked body.

        Args:
            state (newton.State): The simulation state.
            ray_start (wp.vec3f): The start point of the ray.
            ray_dir (wp.vec3f): The direction of the ray.
        """

        if self.model is None:
            return

        p, d = ray_start, ray_dir

        num_geoms = self.model.shape_count
        if num_geoms == 0:
            return

        if self.min_dist is None:
            self.min_dist = wp.array([1.0e10], dtype=float, device=self.model.device)
            self.min_index = wp.array([-1], dtype=int, device=self.model.device)
            self.min_body_index = wp.array([-1], dtype=int, device=self.model.device)
            self.lock = wp.array([0], dtype=wp.int32, device=self.model.device)
        else:
            self.min_dist.fill_(1.0e10)
            self.min_index.fill_(-1)
            self.min_body_index.fill_(-1)
            self.lock.zero_()

        wp.launch(
            kernel=raycast.raycast_kernel,
            dim=num_geoms,
            inputs=[
                state.body_q,
                self.model.shape_body,
                self.model.shape_transform,
                self.model.shape_type,
                self.model.shape_scale,
                self.model.shape_source_ptr,
                p,
                d,
                self.lock,
            ],
            outputs=[self.min_dist, self.min_index, self.min_body_index],
            device=self.model.device,
        )
        wp.synchronize()

        dist = self.min_dist.numpy()[0]
        index = self.min_index.numpy()[0]
        body_index = self.min_body_index.numpy()[0]

        if dist < 1.0e10 and body_index >= 0:
            self.pick_dist = dist

            # Ensures that the ray direction and start point are vec3f objects
            d = wp.vec3f(d[0], d[1], d[2])
            p = wp.vec3f(p[0], p[1], p[2])
            # world space hit point
            hit_point_world = p + d * float(dist)

            wp.launch(
                kernel=compute_pick_state_kernel,
                dim=1,
                inputs=[state.body_q, body_index, hit_point_world],
                outputs=[self.pick_body, self.pick_state],
                device=self.model.device,
            )
            wp.synchronize()

        self.picking_active = self.pick_body.numpy()[0] >= 0

        if self._debug:
            if dist < 1.0e10:
                print("#" * 80)
                print(f"Hit geom {index} of body {body_index} at distance {dist}")
                print("#" * 80)
