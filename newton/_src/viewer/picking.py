# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

import newton

from ..geometry import raycast
from .kernels import PickingState, apply_picking_force_kernel, compute_pick_state_kernel, update_pick_target_kernel


class Picking:
    """
    Picking system.

    Allows to pick a body in the viewer by right clicking on it and dragging the mouse.
    This can be used to move objects around in the viewer, a typical use case is to check for solver resilience or
    see how well a RL policy is coping with disturbances.
    """

    def __init__(
        self,
        model: newton.Model,
        pick_stiffness: float = 50.0,
        pick_damping: float = 5.0,
        pick_max_acceleration: float = 5.0,
        world_offsets: wp.array | None = None,
    ) -> None:
        """
        Initializes the picking system.

        Args:
            model: The model to pick from.
            pick_stiffness: The stiffness that will be used to compute the force applied to the picked body.
            pick_damping: The damping that will be used to compute the force applied to the picked body.
            pick_max_acceleration: Maximum picking acceleration in multiples of g [9.81 m/s^2].
                Clamps the picking force to prevent runaway divergence on light objects
                near stiff contacts.
            world_offsets: Optional warp array of world offsets (dtype=wp.vec3) for multi-world picking support.
        """
        self.model = model
        self.pick_stiffness = pick_stiffness
        self.pick_damping = pick_damping
        self.world_offsets = world_offsets
        self.visible_worlds_mask: wp.array | None = None

        self.min_dist = None
        self.min_index = None
        self.min_body_index = None
        self.lock = None
        self._contact_points0 = None
        self._contact_points1 = None
        self._debug = False

        # picking state
        if model and model.device.is_cuda:
            self.pick_body = wp.array([-1], dtype=int, pinned=True, device=model.device)
        else:
            self.pick_body = wp.array([-1], dtype=int, device="cpu")

        pick_state_np = np.empty(1, dtype=PickingState.numpy_dtype())
        pick_state_np[0]["pick_stiffness"] = pick_stiffness
        pick_state_np[0]["pick_damping"] = pick_damping
        pick_state_np[0]["pick_max_acceleration"] = pick_max_acceleration
        self.pick_state = wp.array(pick_state_np, dtype=PickingState, device=model.device if model else "cpu", ndim=1)

        self.pick_dist = 0.0
        self.picking_active = False

        self._default_on_mouse_drag = None

        # Pre-compute effective mass per body for picking force clamping.
        # For articulated bodies, use the total articulation mass so that
        # picking a light link (e.g. fingertip) still allows enough force
        # to move the whole chain. Free bodies use their own mass.
        self._pick_effective_mass = self._compute_effective_mass(model)

    def _apply_picking_force(self, state: newton.State) -> None:
        """
        Applies a force to the picked body.

        Args:
            state: The simulation state.
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
                self.model.body_flags,
                self.model.body_com,
                self.model.body_mass,
                self._pick_effective_mass,
            ],
            device=self.model.device,
        )

    @staticmethod
    def _compute_effective_mass(model: newton.Model) -> wp.array:
        """Compute per-body effective mass for picking force clamping.

        For bodies in an articulation, returns the total mass of that
        articulation so that picking a light link still allows enough
        force to move the whole chain.  Free bodies get their own mass.
        """
        if model is None:
            return wp.zeros(1, dtype=float)

        body_mass_np = model.body_mass.numpy()
        effective = body_mass_np.copy()

        if model.joint_count > 0:
            joint_child_np = model.joint_child.numpy()
            joint_art_np = model.joint_articulation.numpy()

            # Map each body to its articulation index (-1 if free)
            body_art = np.full(model.body_count, -1, dtype=np.int32)
            for j in range(model.joint_count):
                child = joint_child_np[j]
                if child >= 0:
                    body_art[child] = joint_art_np[j]

            # Sum mass per articulation
            art_mass = {}
            for b in range(model.body_count):
                a = body_art[b]
                if a >= 0:
                    art_mass[a] = art_mass.get(a, 0.0) + body_mass_np[b]

            # Assign total articulation mass to each body in that articulation
            for b in range(model.body_count):
                a = body_art[b]
                if a >= 0:
                    effective[b] = art_mass[a]

        return wp.array(effective, dtype=float, device=model.device)

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
            ray_start: The start point of the ray.
            ray_dir: The direction of the ray.
        """
        if not self.is_picking():
            return

        # Get the world offset for the picked body
        world_offset = wp.vec3(0.0, 0.0, 0.0)
        if self.world_offsets is not None and self.world_offsets.shape[0] > 0:
            # Get the picked body index
            picked_body_idx = self.pick_body.numpy()[0]
            if picked_body_idx >= 0 and self.model.body_world is not None:
                # Find which world this body belongs to
                body_world_idx = self.model.body_world.numpy()[picked_body_idx]
                if body_world_idx >= 0 and body_world_idx < self.world_offsets.shape[0]:
                    offset_np = self.world_offsets.numpy()[body_world_idx]
                    world_offset = wp.vec3(float(offset_np[0]), float(offset_np[1]), float(offset_np[2]))

        wp.launch(
            kernel=update_pick_target_kernel,
            dim=1,
            inputs=[
                ray_start,
                ray_dir,
                world_offset,
                self.pick_state,
            ],
            device=self.model.device,
        )

    def pick(self, state: newton.State, ray_start: wp.vec3f, ray_dir: wp.vec3f) -> None:
        """
        Picks the selected geometry and computes the initial state of the picking. I.e. the force that
        will be applied to the picked body.

        Args:
            state: The simulation state.
            ray_start: The start point of the ray.
            ray_dir: The direction of the ray.
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

        # Get world offsets if available
        shape_world = (
            self.model.shape_world
            if self.model.shape_world is not None
            else wp.array([], dtype=int, device=self.model.device)
        )
        if self.world_offsets is not None:
            world_offsets = self.world_offsets
        else:
            world_offsets = wp.array([], dtype=wp.vec3, device=self.model.device)

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
            outputs=[
                self.min_dist,
                self.min_index,
                self.min_body_index,
                shape_world,
                world_offsets,
                self.visible_worlds_mask,
            ],
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
            # world space hit point (in offset coordinate system from raycast)
            hit_point_world = p + d * float(dist)

            # Convert hit point from offset space to physics space
            # The raycast was done with world offsets applied, so we need to remove them
            if world_offsets.shape[0] > 0 and shape_world.shape[0] > 0 and index >= 0:
                world_idx_np = shape_world.numpy()[index] if hasattr(shape_world, "numpy") else shape_world[index]
                if world_idx_np >= 0 and world_idx_np < world_offsets.shape[0]:
                    offset_np = world_offsets.numpy()[world_idx_np]
                    hit_point_world = wp.vec3f(
                        hit_point_world[0] - offset_np[0],
                        hit_point_world[1] - offset_np[1],
                        hit_point_world[2] - offset_np[2],
                    )

            wp.launch(
                kernel=compute_pick_state_kernel,
                dim=1,
                inputs=[state.body_q, self.model.body_flags, body_index, hit_point_world],
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
