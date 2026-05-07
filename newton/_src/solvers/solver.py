# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from ..geometry import ParticleFlags
from ..sim import BodyFlags, Contacts, Control, Model, ModelBuilder, State
from ._fixed_joint_merger import (
    FixedJointMergeInfo,
    _propagate_merged_body_poses,
    _propagate_merged_body_velocities,
    _scatter_body_forces_to_survivors,
    _update_effective_inv_mass_inertia_merged,
    compute_fixed_joint_merge,
)


@wp.kernel
def integrate_particles(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    f: wp.array[wp.vec3],
    w: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    dt: float,
    v_max: float,
    x_new: wp.array[wp.vec3],
    v_new: wp.array[wp.vec3],
):
    tid = wp.tid()
    x0 = x[tid]

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        x_new[tid] = x0
        return

    v0 = v[tid]
    f0 = f[tid]

    inv_mass = w[tid]
    world_idx = particle_world[tid]
    world_g = gravity[wp.max(world_idx, 0)]

    # simple semi-implicit Euler. v1 = v0 + a dt, x1 = x0 + v1 dt
    v1 = v0 + (f0 * inv_mass + world_g * wp.step(-inv_mass)) * dt
    # enforce velocity limit to prevent instability
    v1_mag = wp.length(v1)
    if v1_mag > v_max:
        v1 *= v_max / v1_mag
    x1 = x0 + v1 * dt

    x_new[tid] = x1
    v_new[tid] = v1


@wp.func
def integrate_rigid_body(
    q: wp.transform,
    qd: wp.spatial_vector,
    f: wp.spatial_vector,
    com: wp.vec3,
    inertia: wp.mat33,
    inv_mass: float,
    inv_inertia: wp.mat33,
    gravity: wp.vec3,
    angular_damping: float,
    dt: float,
):
    # unpack transform
    x0 = wp.transform_get_translation(q)
    r0 = wp.transform_get_rotation(q)

    # unpack spatial twist
    w0 = wp.spatial_bottom(qd)
    v0 = wp.spatial_top(qd)

    # unpack spatial wrench
    t0 = wp.spatial_bottom(f)
    f0 = wp.spatial_top(f)

    x_com = x0 + wp.quat_rotate(r0, com)

    # linear part
    v1 = v0 + (f0 * inv_mass + gravity * wp.nonzero(inv_mass)) * dt
    x1 = x_com + v1 * dt

    # angular part (compute in body frame)
    wb = wp.quat_rotate_inv(r0, w0)
    tb = wp.quat_rotate_inv(r0, t0) - wp.cross(wb, inertia * wb)  # coriolis forces

    w1 = wp.quat_rotate(r0, wb + inv_inertia * tb * dt)
    r1 = wp.normalize(r0 + wp.quat(w1, 0.0) * r0 * 0.5 * dt)

    # angular damping
    w1 *= 1.0 - angular_damping * dt

    q_new = wp.transform(x1 - wp.quat_rotate(r1, com), r1)
    qd_new = wp.spatial_vector(v1, w1)

    return q_new, qd_new


# semi-implicit Euler integration
@wp.kernel
def integrate_bodies(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    m: wp.array[float],
    I: wp.array[wp.mat33],
    inv_m: wp.array[float],
    inv_I: wp.array[wp.mat33],
    body_flags: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    angular_damping: float,
    dt: float,
    # outputs
    body_q_new: wp.array[wp.transform],
    body_qd_new: wp.array[wp.spatial_vector],
):
    tid = wp.tid()

    if (body_flags[tid] & BodyFlags.KINEMATIC) != 0:
        # Kinematic bodies are user-prescribed and pass through unchanged.
        # NOTE: SemiImplicit does not zero inv_mass/inv_inertia for kinematic
        # bodies in the contact solver, so contact responses may be weaker
        # than XPBD or MuJoCo/Featherstone which treat them as infinite-mass.
        body_q_new[tid] = body_q[tid]
        body_qd_new[tid] = body_qd[tid]
        return

    # positions
    q = body_q[tid]
    qd = body_qd[tid]
    f = body_f[tid]

    # masses
    inv_mass = inv_m[tid]  # 1 / mass

    inertia = I[tid]
    inv_inertia = inv_I[tid]  # inverse of 3x3 inertia matrix

    com = body_com[tid]
    world_idx = body_world[tid]
    world_g = gravity[wp.max(world_idx, 0)]

    q_new, qd_new = integrate_rigid_body(
        q,
        qd,
        f,
        com,
        inertia,
        inv_mass,
        inv_inertia,
        world_g,
        angular_damping,
        dt,
    )

    body_q_new[tid] = q_new
    body_qd_new[tid] = qd_new


@wp.kernel
def _update_effective_inv_mass_inertia(
    body_flags: wp.array[wp.int32],
    model_inv_mass: wp.array[float],
    model_inv_inertia: wp.array[wp.mat33],
    eff_inv_mass: wp.array[float],
    eff_inv_inertia: wp.array[wp.mat33],
):
    tid = wp.tid()
    if (body_flags[tid] & BodyFlags.KINEMATIC) != 0:
        eff_inv_mass[tid] = 0.0
        eff_inv_inertia[tid] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        eff_inv_mass[tid] = model_inv_mass[tid]
        eff_inv_inertia[tid] = model_inv_inertia[tid]


class SolverBase:
    """Generic base class for solvers.

    The implementation provides helper kernels to integrate rigid bodies and
    particles. Concrete solver back-ends should derive from this class and
    override :py:meth:`step` as well as :py:meth:`notify_model_changed` where
    necessary.
    """

    def __init__(self, model: Model):
        self.model = model

    @property
    def device(self) -> wp.Device:
        """
        Get the device used by the solver.

        Returns:
            wp.Device: The device used by the solver.
        """
        return self.model.device

    def _init_kinematic_state(self):
        """Allocate and populate effective inverse mass/inertia arrays."""
        model = self.model
        self.body_inv_mass_effective = wp.empty_like(model.body_inv_mass)
        self.body_inv_inertia_effective = wp.empty_like(model.body_inv_inertia)
        if model.body_count:
            self._refresh_kinematic_state()

    def _refresh_kinematic_state(self):
        """Update effective arrays from model, zeroing kinematic and merged-child bodies."""
        model = self.model
        if not model.body_count:
            return
        merge_info: FixedJointMergeInfo | None = getattr(self, "_merge_info", None)
        if merge_info is not None:
            wp.launch(
                kernel=_update_effective_inv_mass_inertia_merged,
                dim=model.body_count,
                inputs=[
                    model.body_flags,
                    merge_info.survivor_indices_gpu,
                    merge_info.merged_body_inv_mass_gpu,
                    merge_info.merged_body_inv_inertia_gpu,
                    self.body_inv_mass_effective,
                    self.body_inv_inertia_effective,
                ],
                device=model.device,
            )
        else:
            wp.launch(
                kernel=_update_effective_inv_mass_inertia,
                dim=model.body_count,
                inputs=[
                    model.body_flags,
                    model.body_inv_mass,
                    model.body_inv_inertia,
                    self.body_inv_mass_effective,
                    self.body_inv_inertia_effective,
                ],
                device=model.device,
            )

    def _init_fixed_joint_merge(self, merge_info: FixedJointMergeInfo | None) -> None:
        """Install merge metadata and seed effective arrays."""
        model = self.model
        if merge_info is None or not merge_info.has_merges:
            self._merge_info = None
            self.joint_enabled_effective = model.joint_enabled
            return
        self._merge_info = merge_info
        self.joint_enabled_effective = merge_info.joint_enabled_effective_gpu
        if model.body_count and hasattr(self, "body_inv_mass_effective"):
            self._refresh_kinematic_state()

    def _recompute_merge_info(self) -> None:
        """Re-run the merge analysis after model inputs change."""
        # Honor the constructor opt-out: don't resurrect merging if the user
        # built the solver with collapse_fixed_joints=False.
        if not getattr(self, "_collapse_fixed_joints", True):
            self._refresh_kinematic_state()
            return
        joints_to_keep = getattr(self, "_joints_to_keep", None)
        new_info = compute_fixed_joint_merge(self.model, joints_to_keep=joints_to_keep)
        if new_info is None:
            self._merge_info = None
            self.joint_enabled_effective = self.model.joint_enabled
        else:
            self._merge_info = new_info
            self.joint_enabled_effective = new_info.joint_enabled_effective_gpu
        self._refresh_kinematic_state()

    def _scatter_merged_body_forces(self, state_in: State, body_f: wp.array | None) -> None:
        """Move external body_f written to merged-child slots onto their survivors."""
        merge_info: FixedJointMergeInfo | None = getattr(self, "_merge_info", None)
        if merge_info is None or body_f is None:
            return
        model = self.model
        wp.launch(
            kernel=_scatter_body_forces_to_survivors,
            dim=model.body_count,
            inputs=[
                merge_info.survivor_indices_gpu,
                state_in.body_q,
                merge_info.merged_body_com_gpu,
                merge_info.relative_xforms_gpu,
                body_f,
            ],
            device=model.device,
        )

    def _propagate_merged_body_poses_and_velocities(self, state_out: State) -> None:
        """Scatter survivor pose and velocity into each merged child's slots."""
        merge_info: FixedJointMergeInfo | None = getattr(self, "_merge_info", None)
        if merge_info is None:
            return
        model = self.model
        wp.launch(
            kernel=_propagate_merged_body_poses,
            dim=model.body_count,
            inputs=[
                merge_info.survivor_indices_gpu,
                merge_info.relative_xforms_gpu,
                state_out.body_q,
            ],
            device=model.device,
        )
        wp.launch(
            kernel=_propagate_merged_body_velocities,
            dim=model.body_count,
            inputs=[
                merge_info.survivor_indices_gpu,
                state_out.body_q,
                merge_info.relative_xforms_gpu,
                model.body_com,
                merge_info.merged_body_com_gpu,
                state_out.body_qd,
            ],
            device=model.device,
        )

    def integrate_bodies(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        dt: float,
        angular_damping: float = 0.0,
        body_com: wp.array | None = None,
        body_mass: wp.array | None = None,
        body_inertia: wp.array | None = None,
        body_inv_mass: wp.array | None = None,
        body_inv_inertia: wp.array | None = None,
    ) -> None:
        """Integrate the rigid bodies of the model.

        Args:
            model: The model to integrate.
            state_in: The input state.
            state_out: The output state.
            dt: The time step (typically in seconds).
            angular_damping: Angular damping factor. Defaults to 0.0.
            body_com: Override for ``model.body_com``. Used by fixed-joint collapsing.
            body_mass: Override for ``model.body_mass``.
            body_inertia: Override for ``model.body_inertia``.
            body_inv_mass: Override for ``model.body_inv_mass``.
            body_inv_inertia: Override for ``model.body_inv_inertia``.
        """
        if model.body_count:
            wp.launch(
                kernel=integrate_bodies,
                dim=model.body_count,
                inputs=[
                    state_in.body_q,
                    state_in.body_qd,
                    state_in.body_f,
                    body_com if body_com is not None else model.body_com,
                    body_mass if body_mass is not None else model.body_mass,
                    body_inertia if body_inertia is not None else model.body_inertia,
                    body_inv_mass if body_inv_mass is not None else model.body_inv_mass,
                    body_inv_inertia if body_inv_inertia is not None else model.body_inv_inertia,
                    model.body_flags,
                    model.body_world,
                    model.gravity,
                    angular_damping,
                    dt,
                ],
                outputs=[state_out.body_q, state_out.body_qd],
                device=model.device,
            )

    def integrate_particles(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        dt: float,
    ) -> None:
        """
        Integrate the particles of the model.

        Args:
            model (Model): The model to integrate.
            state_in (State): The input state.
            state_out (State): The output state.
            dt (float): The time step (typically in seconds).
        """
        if model.particle_count:
            wp.launch(
                kernel=integrate_particles,
                dim=model.particle_count,
                inputs=[
                    state_in.particle_q,
                    state_in.particle_qd,
                    state_in.particle_f,
                    model.particle_inv_mass,
                    model.particle_flags,
                    model.particle_world,
                    model.gravity,
                    dt,
                    model.particle_max_velocity,
                ],
                outputs=[state_out.particle_q, state_out.particle_qd],
                device=model.device,
            )

    def step(
        self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float
    ) -> None:
        """
        Simulate the model for a given time step using the given control input.

        Args:
            state_in: The input state.
            state_out: The output state.
            control: The control input.
                Defaults to `None` which means the control values from the
                :class:`Model` are used.
            contacts: The contact information.
            dt: The time step (typically in seconds).
        """
        raise NotImplementedError()

    def notify_model_changed(self, flags: int) -> None:
        """Notify the solver that parts of the :class:`~newton.Model` were modified.

        The *flags* argument is a bit-mask composed of the
        :class:`~newton.solvers.SolverNotifyFlags` enums defined in :mod:`newton.solvers`.
        Each flag represents a category of model data that may have been
        updated after the solver was created.  Passing the appropriate
        combination of flags enables a solver implementation to refresh its
        internal buffers without having to recreate the whole solver object.
        Valid flags are:

        ==============================================  =============================================================
        Constant                                        Description
        ==============================================  =============================================================
        ``SolverNotifyFlags.JOINT_PROPERTIES``            Joint transforms or coordinates have changed.
        ``SolverNotifyFlags.JOINT_DOF_PROPERTIES``        Joint axis limits, targets, modes, DOF state, or force buffers have changed.
        ``SolverNotifyFlags.BODY_PROPERTIES``             Rigid-body pose or velocity buffers have changed.
        ``SolverNotifyFlags.BODY_INERTIAL_PROPERTIES``    Rigid-body mass or inertia tensors have changed.
        ``SolverNotifyFlags.SHAPE_PROPERTIES``            Shape transforms or geometry have changed.
        ``SolverNotifyFlags.MODEL_PROPERTIES``            Model global properties (e.g., gravity) have changed.
        ==============================================  =============================================================

        Args:
            flags (int): Bit-mask of model-update flags indicating which model
                properties changed.

        """
        pass

    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        """
        Update a Contacts object with forces from the solver state. Where the solver state contains
        other contact data, convert that data to the Contacts format.

        Args:
            contacts: The object to update from the solver state.
            state: Optional simulation state, used by some solvers.
        """
        raise NotImplementedError()

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """
        Register custom attributes for the solver.

        Args:
            builder (ModelBuilder): The model builder to register the custom attributes to.
        """
        pass
