# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton

from ..geometry import raycast
from ..sim.articulation import eval_articulation_jacobian
from .kernels import PickingState, apply_picking_force_kernel, compute_pick_state_kernel, update_pick_target_kernel


def _rotation_matrix(X: np.ndarray) -> np.ndarray:
    """Return the rotation of a transform stored as ``(px, py, pz, qx, qy, qz, qw)``."""
    return np.array(wp.quat_to_matrix(wp.quat(*X[3:])), dtype=np.float64).reshape(3, 3)


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
        pick_stiffness: float = 500.0,
        pick_damping: float = 45.0,
        pick_max_acceleration: float = 5.0,
        world_offsets: wp.array[wp.vec3] | None = None,
    ) -> None:
        r"""
        Initializes the picking system. All parameters are fixed at construction.

        Args:
            model: The model to pick from.
            pick_stiffness: Stiffness of the picking spring, in acceleration per unit
                displacement [1/s^2]. The commanded acceleration is mapped to a force
                through the operational-space inertia at the pick point, so gains are
                mass-independent and light distal links stay responsive.
            pick_damping: Damping of the picking spring, in acceleration per unit
                velocity of the pick point [1/s]. The default is close to critical
                damping, :math:`2\sqrt{k}`, for the default stiffness.
            pick_max_acceleration: Picking limit in multiples of g [9.81 m/s^2]. Caps the force at
                this acceleration times the mass of the picked articulation, so a light link can
                still move its chain, and the torque at the equivalent rotational acceleration of
                the picked body alone, to prevent runaway divergence on low-inertia links.
            world_offsets: Optional warp array of world offsets (dtype=wp.vec3) for multi-world picking support.

        Raises:
            ValueError: If ``pick_max_acceleration`` is negative or non-finite.
        """
        pick_max_acceleration = float(pick_max_acceleration)
        if not math.isfinite(pick_max_acceleration) or pick_max_acceleration < 0.0:
            raise ValueError("Picking maximum acceleration must be finite and nonnegative.")

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

        # Operational-space inertia at the pick point in the picked body's frame, set on every pick.
        self._pick_os_inertia = wp.zeros(1, dtype=wp.mat33, device=model.device if model else "cpu")
        # Single-articulation Jacobian scratch, allocated on the first articulated pick.
        self._jacobian = None
        self._joint_S_s = None
        if model is not None:
            self._init_topology(model)
        else:
            self._pick_effective_mass = wp.zeros(1, dtype=float)

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
            ],
            device=self.model.device,
        )

    def _update_os_inertia(self, state: newton.State, body: int, point_world: wp.vec3f) -> None:
        """Set the operational-space inertia used to turn commanded acceleration into force.

        Computes :math:`\\Lambda = (J H^{-1} J^T)^{-1}` at the pick point, the inertia felt at the
        grab point, so light distal links stay responsive. Eigenvalues are capped at the
        articulation total, which bounds directions the joints block. Articulations with
        :attr:`~newton.JointType.ROD` joints use the articulation total instead: each rod joint has
        six DOFs, so the Jacobian would resolve only one segment's mass and leave cables too weak
        to lift. Stored in the body frame, so it turns with the picked link.

        Args:
            state: The simulation state supplying the current pose.
            body: Index of the picked body.
            point_world: The pick point in world space.
        """
        mass = float(self._body_mass[body])
        upper = float(self._effective_mass[body])
        # A massless link, e.g. a URDF frame link with geometry, is moved through its subtree.
        lam = np.eye(3) * (mass if mass > 0.0 else upper)

        art = int(self._body_art[body])
        if art >= 0 and upper > 0.0:
            if self._art_has_rod[art]:
                lam = np.eye(3) * upper
            else:
                inv_lam = self._inverse_os_inertia(state, art, body, np.array(point_world, dtype=np.float64))
                if inv_lam is not None:
                    eigenvalues, eigenvectors = np.linalg.eigh(inv_lam)
                    lam = eigenvectors @ np.diag(1.0 / np.maximum(eigenvalues, 1.0 / upper)) @ eigenvectors.T

        rotation = _rotation_matrix(state.body_q[body : body + 1].numpy()[0])
        lam_body = rotation.T @ lam @ rotation
        self._pick_os_inertia.assign(lam_body.astype(np.float32).reshape(1, 3, 3))

    def _inverse_os_inertia(
        self, state: newton.State, art: int, body: int, point_world: np.ndarray
    ) -> np.ndarray | None:
        """Return :math:`J H^{-1} J^T` at the pick point, evaluating only the picked articulation."""
        model = self.model
        if self._jacobian is None:
            self._jacobian = wp.empty(
                (1, 6 * model.max_joints_per_articulation, model.max_dofs_per_articulation),
                dtype=float,
                device=model.device,
            )
            self._joint_S_s = wp.empty(model.joint_dof_count, dtype=wp.spatial_vector, device=model.device)

        self._jacobian.zero_()
        wp.launch(
            kernel=eval_articulation_jacobian,
            dim=1,
            inputs=[
                model.articulation_start[art : art + 1],
                model.articulation_end[art : art + 1],
                1,
                None,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_ancestor,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_X_p,
                model.joint_axis,
                state.joint_q,
                model.joint_dof_dim,
                state.body_q,
                model.body_com,
            ],
            outputs=[self._jacobian, self._joint_S_s],
            device=model.device,
        )

        first, end = int(self._art_start[art]), int(self._art_end[art])
        children = model.joint_child[first:end].numpy()
        dof_start, dof_end = (int(i) for i in model.joint_qd_start[first : end + 1].numpy()[[0, -1]])
        ndof = dof_end - dof_start
        jac = self._jacobian.numpy()[0, : 6 * len(children), :ndof].astype(np.float64).reshape(-1, 6, ndof)

        lo, hi = int(children.min()), int(children.max()) + 1
        body_q = state.body_q[lo:hi].numpy()
        body_inertia = model.body_inertia[lo:hi].numpy()

        # H = sum_i J_i^T M_i J_i over the links' COM twists, plus armature.
        h = np.diag(model.joint_armature[dof_start:dof_end].numpy().astype(np.float64))
        for i, child in enumerate(children):
            rotation = _rotation_matrix(body_q[child - lo])
            lin, ang = jac[i, :3], jac[i, 3:]
            h += self._body_mass[child] * lin.T @ lin + ang.T @ rotation @ body_inertia[child - lo] @ rotation.T @ ang

        # Newton twists are (linear, angular) at the COM; shift to the pick point via v_p = v_com + w x r.
        X_wb = wp.transform(*body_q[body - lo])
        com = wp.transform_point(X_wb, wp.vec3(*model.body_com[body : body + 1].numpy()[0]))
        r = point_world - np.array(com, dtype=np.float64)
        row = int(self._body_joint[body]) - first
        jac_point = jac[row, :3] + np.cross(jac[row, 3:].T, r).T

        try:
            return jac_point @ np.linalg.solve(h, jac_point.T)
        except np.linalg.LinAlgError:
            return None

    def _init_topology(self, model: newton.Model) -> None:
        """Cache per-body articulation data on the host so a pick copies only the picked articulation."""
        body_mass = model.body_mass.numpy().astype(np.float64)
        body_joint = np.full(model.body_count, -1, dtype=np.int64)
        body_art = np.full(model.body_count, -1, dtype=np.int64)
        art_has_rod = np.zeros(model.articulation_count, dtype=bool)
        if model.joint_count:
            joint_child = model.joint_child.numpy()
            joint_art = model.joint_articulation.numpy()
            # Loop closures (articulation -1) must not detach a body from its tree joint's articulation.
            tree = np.nonzero((joint_art >= 0) & (joint_child >= 0))[0]
            body_joint[joint_child[tree]] = tree
            body_art[joint_child[tree]] = joint_art[tree]
            art_has_rod[joint_art[tree[model.joint_type.numpy()[tree] == int(newton.JointType.ROD)]]] = True

        # The force cap uses the articulation total, so picking a light link can still move the chain.
        in_art = body_art >= 0
        art_mass = np.bincount(body_art[in_art], weights=body_mass[in_art], minlength=model.articulation_count)
        effective_mass = body_mass.copy()
        effective_mass[in_art] = art_mass[body_art[in_art]]

        self._body_mass = body_mass
        self._effective_mass = effective_mass
        self._body_joint = body_joint
        self._body_art = body_art
        self._art_has_rod = art_has_rod
        self._art_start = model.articulation_start.numpy() if model.articulation_count else None
        self._art_end = model.articulation_end.numpy() if model.articulation_count else None
        self._pick_effective_mass = wp.array(effective_mass, dtype=float, device=model.device)

    def _load_kernels(self) -> None:
        """Compile every Warp module picking launches, so the first pick does not stall."""
        if self.model is None:
            return
        device = self.model.device
        wp.load_module(module=raycast, device=device)
        wp.load_module(module="newton._src.viewer.kernels", device=device)
        # Picking an articulated body evaluates its Jacobian.
        if self.model.articulation_count:
            wp.load_module(module="newton._src.sim.articulation", device=device)

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
