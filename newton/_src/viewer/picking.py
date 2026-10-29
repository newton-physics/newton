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


def _validate_max_acceleration(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("Picking maximum acceleration must be finite and nonnegative.")
    return value


def _rotation_matrix(X: np.ndarray) -> np.ndarray:
    """Return the rotation of a transform stored as ``(px, py, pz, qx, qy, qz, qw)``."""
    return np.array(wp.quat_to_matrix(wp.quat(*X[3:])), dtype=np.float64).reshape(3, 3)


class Picking:
    """Interactive picking controller that drags a body with a spring toward a ray.

    A ray cast picks the closest body and anchors a spring at the hit point. As the
    ray moves, the spring target follows it at the target's current depth, and
    :meth:`apply_force` adds the resulting force and torque to the picked body.
    This is useful to move objects around interactively, for example to check
    solver resilience or how well a policy copes with disturbances.

    :class:`~newton.viewer.ViewerGL`, :class:`~newton.viewer.ViewerRTX`, and
    :class:`~newton.viewer.ViewerViser` create one in ``set_model()``, expose it as
    ``viewer.picking``, and drive it from the mouse. Use this class directly to
    add picking to a custom application:

    .. code-block:: python

        picking = newton.viewer.Picking(model)

        # on mouse press, with a world-space ray from the camera
        picking.pick(state, ray_start, ray_dir)
        # on mouse drag
        picking.update(ray_start, ray_dir)
        # on mouse release
        picking.release()

        # every simulation step, after clearing the state's forces
        picking.apply_force(state)
        solver.step(state, state_next, control, contacts, dt)

    To drive the target from a 3D input such as a gizmo, assign :attr:`pick_target`
    instead of calling :meth:`update`. Read :attr:`pick_point` and :attr:`pick_target`
    to draw the spring. Kinematic bodies cannot be picked.
    """

    def __init__(
        self,
        model: newton.Model,
        *,
        pick_stiffness: float = 500.0,
        pick_damping: float = 45.0,
        pick_max_acceleration: float = 5.0,
        world_offsets: wp.array[wp.vec3] | None = None,
        visible_worlds_mask: wp.array[int] | None = None,
    ) -> None:
        r"""
        Initializes the picking system.

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
            world_offsets: Per-world visual offsets [m], shape [world_count]. Rays are
                given in the offset (rendered) space, and the offsets are removed before
                the spring is applied in simulation space. ``None`` means no offsets.
            visible_worlds_mask: Per-world mask, shape [world_count]; rays only hit shapes in
                worlds whose entry is nonzero. ``None`` makes every world pickable.

        Raises:
            ValueError: If ``pick_max_acceleration`` is negative or non-finite.
        """
        pick_max_acceleration = _validate_max_acceleration(pick_max_acceleration)

        self.model = model
        """The model whose bodies are picked."""
        self.world_offsets = world_offsets
        """Per-world visual offsets [m], shape [world_count], or ``None``; see the constructor."""
        self.visible_worlds_mask = visible_worlds_mask
        """Per-world pick mask, shape [world_count], or ``None``; see the constructor."""

        self._min_dist = None
        self._min_index = None
        self._min_body_index = None
        self._lock = None

        # Device-side sentinel read by apply_force(), so captured graphs see pick changes.
        self._pick_body = wp.array([-1], dtype=int, device=model.device)

        pick_state_np = np.empty(1, dtype=PickingState.numpy_dtype())
        pick_state_np[0]["pick_stiffness"] = pick_stiffness
        pick_state_np[0]["pick_damping"] = pick_damping
        pick_state_np[0]["pick_max_acceleration"] = pick_max_acceleration
        self._pick_state = wp.array(pick_state_np, dtype=PickingState, device=model.device, ndim=1)

        self._picking_active = False

        # Operational-space inertia at the pick point in the picked body's frame, set on every pick.
        self._pick_os_inertia = wp.zeros(1, dtype=wp.mat33, device=model.device)
        # Single-articulation Jacobian scratch, allocated on the first articulated pick.
        self._jacobian = None
        self._joint_S_s = None
        self._init_topology(model)

    @property
    def pick_stiffness(self) -> float:
        """Stiffness of the picking spring [1/s^2]; see the constructor."""
        return float(self._pick_state.numpy()[0]["pick_stiffness"])

    @pick_stiffness.setter
    def pick_stiffness(self, value: float) -> None:
        self._set_pick_state_field("pick_stiffness", float(value))

    @property
    def pick_damping(self) -> float:
        """Damping of the picking spring [1/s]; see the constructor."""
        return float(self._pick_state.numpy()[0]["pick_damping"])

    @pick_damping.setter
    def pick_damping(self, value: float) -> None:
        self._set_pick_state_field("pick_damping", float(value))

    @property
    def pick_max_acceleration(self) -> float:
        """Picking limit in multiples of g; see the constructor.

        Raises:
            ValueError: If set to a negative or non-finite value.
        """
        return float(self._pick_state.numpy()[0]["pick_max_acceleration"])

    @pick_max_acceleration.setter
    def pick_max_acceleration(self, value: float) -> None:
        self._set_pick_state_field("pick_max_acceleration", _validate_max_acceleration(value))

    @property
    def pick_body(self) -> int:
        """Index of the picked body, or -1 when nothing is picked."""
        return int(self._pick_body.numpy()[0])

    @property
    def pick_point(self) -> wp.vec3 | None:
        """Attachment point on the picked body as of the last :meth:`apply_force`, in simulation space [m].

        ``None`` when nothing is picked.
        """
        if not self.is_picking():
            return None
        return wp.vec3(*self._pick_state.numpy()[0]["picked_point_world"])

    @property
    def pick_target(self) -> wp.vec3 | None:
        """Point the spring pulls toward, in simulation space [m], or ``None`` when nothing is picked.

        Assign it to drive the target from a 3D input instead of :meth:`update`; the assignment is
        ignored when nothing is picked.
        """
        if not self.is_picking():
            return None
        return wp.vec3(*self._pick_state.numpy()[0]["picking_target_world"])

    @pick_target.setter
    def pick_target(self, target: wp.vec3) -> None:
        if self.is_picking():
            self._set_pick_state_field("picking_target_world", target)

    def _set_pick_state_field(self, name: str, value) -> None:
        # Writes in place, so CUDA graphs that captured apply_force() see the change.
        pick_state = self._pick_state.numpy()
        pick_state[0][name] = value
        self._pick_state.assign(pick_state)

    def apply_force(self, state: newton.State) -> None:
        """Add the picking spring's force and torque to the picked body.

        The wrench is accumulated into :attr:`State.body_f <newton.State.body_f>`,
        so call this after the state's forces are cleared and before the solver
        step. Does nothing when no body is picked. The kernel is launched even
        then, so this call can be captured in a CUDA graph.

        Args:
            state: The simulation state to read the picked body's pose and
                velocity from and add the force to.
        """
        if self.model.body_count == 0 or state.body_q is None or state.body_qd is None or state.body_f is None:
            return

        # Launch kernel always because of graph capture
        wp.launch(
            kernel=apply_picking_force_kernel,
            dim=1,
            inputs=[
                state.body_q,
                state.body_qd,
                state.body_f,
                self._pick_body,
                self._pick_state,
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
        device = self.model.device
        wp.load_module(module=raycast, device=device)
        wp.load_module(module="newton._src.viewer.kernels", device=device)
        # Picking an articulated body evaluates its Jacobian.
        if self.model.articulation_count:
            wp.load_module(module="newton._src.sim.articulation", device=device)

    def is_picking(self) -> bool:
        """Return whether a body is currently picked.

        Returns:
            ``True`` if a body is picked, ``False`` otherwise.
        """
        return self._picking_active

    def release(self) -> None:
        """Release the picked body so :meth:`apply_force` stops applying a force."""
        self._pick_body.fill_(-1)
        self._picking_active = False

    def update(self, ray_start: wp.vec3f, ray_dir: wp.vec3f) -> None:
        """Move the spring target onto a new ray, keeping its distance from the ray start.

        Call this as the cursor is dragged. Does nothing when no body is picked.

        Args:
            ray_start: Ray origin in the offset (rendered) space [m].
            ray_dir: Unit ray direction in the offset (rendered) space.
        """
        if not self.is_picking():
            return

        # Get the world offset for the picked body
        world_offset = wp.vec3(0.0, 0.0, 0.0)
        if self.world_offsets is not None and self.world_offsets.shape[0] > 0:
            # Get the picked body index
            picked_body_idx = self._pick_body.numpy()[0]
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
                self._pick_state,
            ],
            device=self.model.device,
        )

    def pick(self, state: newton.State, ray_start: wp.vec3f, ray_dir: wp.vec3f) -> None:
        """Pick the body whose shape the ray hits first and anchor the spring at the hit point.

        Any previous pick is released first. Shapes in worlds hidden by
        :attr:`visible_worlds_mask` are ignored. If the ray hits nothing, a static shape, or a
        kinematic body, nothing is picked and :meth:`is_picking` returns ``False``. This call
        synchronizes with the device.

        Args:
            state: The simulation state supplying body poses.
            ray_start: Ray origin in the offset (rendered) space [m].
            ray_dir: Unit ray direction in the offset (rendered) space.
        """
        self.release()

        p, d = ray_start, ray_dir

        num_geoms = self.model.shape_count
        if num_geoms == 0:
            return

        if self._min_dist is None:
            self._min_dist = wp.array([1.0e10], dtype=float, device=self.model.device)
            self._min_index = wp.array([-1], dtype=int, device=self.model.device)
            self._min_body_index = wp.array([-1], dtype=int, device=self.model.device)
            self._lock = wp.array([0], dtype=wp.int32, device=self.model.device)
        else:
            self._min_dist.fill_(1.0e10)
            self._min_index.fill_(-1)
            self._min_body_index.fill_(-1)
            self._lock.zero_()

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
                self._lock,
            ],
            outputs=[
                self._min_dist,
                self._min_index,
                self._min_body_index,
                shape_world,
                world_offsets,
                self.visible_worlds_mask,
            ],
            device=self.model.device,
        )
        wp.synchronize()

        dist = self._min_dist.numpy()[0]
        index = self._min_index.numpy()[0]
        body_index = self._min_body_index.numpy()[0]

        if dist < 1.0e10 and body_index >= 0:
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
                outputs=[self._pick_body, self._pick_state],
                device=self.model.device,
            )

            picked = int(self._pick_body.numpy()[0])
            if picked >= 0:
                self._update_os_inertia(state, picked, hit_point_world)

        self._picking_active = self._pick_body.numpy()[0] >= 0
