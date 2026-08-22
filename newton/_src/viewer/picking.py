# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

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
        pick_min_inertia_fraction: float = 0.2,
        world_offsets: wp.array[wp.vec3] | None = None,
    ) -> None:
        """
        Initializes the picking system. All parameters are fixed at construction.

        Args:
            model: The model to pick from.
            pick_stiffness: Stiffness of the picking spring, in acceleration per unit
                displacement [1/s^2]. The commanded acceleration is mapped to a force
                through the operational-space inertia at the pick point, so gains are
                mass-independent and picking a light distal link is as effective as
                picking a heavy one.
            pick_damping: Damping of the picking spring, in acceleration per unit
                velocity of the pick point [1/s].
            pick_max_acceleration: Maximum picking acceleration in multiples of g [9.81 m/s^2].
                Clamps both linear and equivalent rotational acceleration to prevent
                runaway divergence on light or low-inertia objects.
            pick_min_inertia_fraction: Lower bound on the operational-space inertia at the
                pick point, as a fraction of the articulation's total mass.
            world_offsets: Optional warp array of world offsets (dtype=wp.vec3) for multi-world picking support.

        Raises:
            ValueError: If ``pick_max_acceleration`` is negative or non-finite, or if
                ``pick_min_inertia_fraction`` is outside [0, 1].
        """
        pick_max_acceleration = float(pick_max_acceleration)
        if not math.isfinite(pick_max_acceleration) or pick_max_acceleration < 0.0:
            raise ValueError("Picking maximum acceleration must be finite and nonnegative.")

        pick_min_inertia_fraction = float(pick_min_inertia_fraction)
        if not 0.0 <= pick_min_inertia_fraction <= 1.0:
            raise ValueError("Picking minimum inertia fraction must be in [0, 1].")
        self.pick_min_inertia_fraction = pick_min_inertia_fraction

        self.model = model
        self.world_offsets = world_offsets
        self.visible_worlds_mask: wp.array[int] | None = None

        self.min_dist = None
        self.min_index = None
        self.min_body_index = None
        self.lock = None

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

        self._pick_effective_mass = self._compute_effective_mass(model)

        # Operational-space inertia at the pick point, refreshed on every pick.
        # Identity-scaled by the body mass is the correct value for a free body.
        self._pick_os_inertia = wp.zeros(1, dtype=wp.mat33, device=model.device if model else "cpu")

    def _apply_picking_force(self, state: newton.State) -> None:
        """
        Applies a force to the picked body.

        Args:
            state: The simulation state.
        """
        if (
            self.model is None
            or self.model.body_count == 0
            or state.body_q is None
            or state.body_qd is None
            or state.body_f is None
        ):
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
                self.model.body_inv_inertia,
                self._pick_effective_mass,
                self._pick_os_inertia,
                self.model.gravity,
                self.model.body_world,
            ],
            device=self.model.device,
        )

    def _update_os_inertia(self, state: newton.State, body: int, point_world: wp.vec3f) -> None:
        """Refresh the operational-space inertia used to turn commanded acceleration into force.

        Computes :math:`\\Lambda = (J H^{-1} J^T)^{-1}` at the pick point, where ``J`` maps
        articulation velocities to the velocity of that point and ``H`` is the joint-space mass
        matrix. This is the inertia actually felt at the grab point, so picking a light distal
        link is not weaker than picking the base. Eigenvalues are clamped to the articulation
        total from above, which avoids over-commanding and regularizes singular poses, and from
        below to the larger of the body's own mass and the share of the articulation implied by
        ``pick_min_inertia_fraction``. Where the articulation Jacobian is unavailable, as for
        :attr:`~newton.JointType.CABLE` joints, the articulation total stands in, since the
        whole chain is a closer estimate of what resists than the single link. A body outside
        an articulation falls back to :math:`m I`, exact for a free body.

        Held fixed for the duration of the pick.

        Args:
            state: The simulation state supplying the current pose.
            body: Index of the picked body.
            point_world: The pick point in world space.
        """
        model = self.model
        mass = float(model.body_mass.numpy()[body])
        upper = max(float(self._pick_effective_mass.numpy()[body]), mass)
        lower = min(max(mass, self.pick_min_inertia_fraction * upper), upper)
        lam = np.eye(3) * upper

        joints = np.nonzero(model.joint_child.numpy() == body)[0] if model.articulation_count else []
        if len(joints) and mass > 0.0:
            joint = int(joints[0])
            art = int(model.joint_articulation.numpy()[joint])
            first = int(model.articulation_start.numpy()[art])
            qd_start = model.joint_qd_start.numpy()
            row = 6 * (joint - first)
            ndof = int(qd_start[int(model.articulation_end.numpy()[art])]) - int(qd_start[first])

            mask = wp.array(np.arange(model.articulation_count) == art, dtype=bool, device=model.device)
            jacobian = newton.eval_jacobian(model, state, mask=mask)
            h = newton.eval_mass_matrix(model, state, J=jacobian, mask=mask).numpy()[art][:ndof, :ndof]
            jac = jacobian.numpy()[art][:, :ndof]

            # Newton stores twists as (linear, angular); shift the linear rows from the COM to
            # the pick point via v_p = v_com + omega x r.
            X_wb = wp.transform(*state.body_q.numpy()[body])
            r = np.array(point_world) - np.array(wp.transform_point(X_wb, wp.vec3(*model.body_com.numpy()[body])))
            jac_point = jac[row : row + 3] + np.cross(jac[row + 3 : row + 6].T, r).T

            try:
                eigenvalues, eigenvectors = np.linalg.eigh(jac_point @ np.linalg.solve(h, jac_point.T))
                inertia = 1.0 / np.clip(eigenvalues, 1.0 / upper, 1.0 / lower)
                lam = eigenvectors @ np.diag(inertia) @ eigenvectors.T
            except np.linalg.LinAlgError:
                pass

        self._pick_os_inertia.assign(np.ascontiguousarray(lam, dtype=np.float32).reshape(1, 3, 3))

    @staticmethod
    def _compute_effective_mass(model: newton.Model) -> wp.array[float]:
        """Compute the per-body mass bound used to clamp the picking force.

        For bodies in an articulation, returns the total mass of that
        articulation so that picking a light link still allows enough
        force to move the whole chain.  Free bodies get their own mass.

        This bounds the force only; the commanded acceleration is mapped to a force
        by :meth:`_update_os_inertia`.
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

            picked = int(self.pick_body.numpy()[0])
            if picked >= 0:
                self._update_os_inertia(state, picked, hit_point_world)

        self.picking_active = self.pick_body.numpy()[0] >= 0
