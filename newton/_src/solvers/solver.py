# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, IntEnum, IntFlag

import warp as wp

from ..geometry import ParticleFlags
from ..sim import BodyFlags, Contacts, Control, Model, ModelBuilder, State


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

    class CouplingEndpointKind(IntEnum):
        """Kinds of model endpoints addressed by coupling hooks."""

        BODY = 0
        PARTICLE = 1

    class CouplingInputStateFlags(IntFlag):
        """Input state updates reported by a coupled wrapper."""

        BODY_Q = 1 << 0
        BODY_QD = 1 << 1
        PARTICLE_Q = 1 << 2
        PARTICLE_QD = 1 << 3
        JOINT_Q = 1 << 4
        JOINT_QD = 1 << 5
        BODY_F = 1 << 6
        PARTICLE_F = 1 << 7
        JOINT_F = 1 << 8
        ITERATION_RESTART = 1 << 9
        """The coupler restarted the same top-level step for another iteration."""

        BODY = BODY_Q | BODY_QD
        PARTICLE = PARTICLE_Q | PARTICLE_QD
        JOINT = JOINT_Q | JOINT_QD
        FORCE = BODY_F | PARTICLE_F | JOINT_F
        ALL = BODY | PARTICLE | JOINT | FORCE

    class CouplingHooks(IntFlag):
        """Coupling dispatch points with generic fallbacks."""

        BODY_PROXY_REWIND_VELOCITY = 1 << 0
        PARTICLE_PROXY_REWIND_VELOCITY = 1 << 1
        BODY_PROXY_HARVEST = 1 << 2
        PARTICLE_PROXY_HARVEST = 1 << 3
        EFFECTIVE_MASS_DIAGONAL = 1 << 4
        EFFECTIVE_MASS_BLOCK = 1 << 5
        NOTIFY_INPUT_STATE_UPDATE = 1 << 6
        PROXY_CONTACT_PREPARE = 1 << 7

    class CouplingCapability(Enum):
        """Solver support level for a coupling hook."""

        DEFAULT = "default"
        CUSTOM = "custom"
        UNSUPPORTED = "unsupported"

    CouplingCapabilities = dict[CouplingHooks, CouplingCapability]

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
        """Update effective arrays from model, zeroing kinematic bodies."""
        model = self.model
        if model.body_count:
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

    def integrate_bodies(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        dt: float,
        angular_damping: float = 0.0,
    ) -> None:
        """
        Integrate the rigid bodies of the model.

        Args:
            model (Model): The model to integrate.
            state_in (State): The input state.
            state_out (State): The output state.
            dt (float): The time step (typically in seconds).
            angular_damping (float, optional): The angular damping factor.
                Defaults to 0.0.
        """
        if model.body_count:
            wp.launch(
                kernel=integrate_bodies,
                dim=model.body_count,
                inputs=[
                    state_in.body_q,
                    state_in.body_qd,
                    state_in.body_f,
                    model.body_com,
                    model.body_mass,
                    model.body_inertia,
                    model.body_inv_mass,
                    model.body_inv_inertia,
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

    def coupling_capabilities(self) -> CouplingCapabilities:
        """Return solver-specific coupling hook support.

        Coupled wrappers use this mapping to decide whether a hook should use
        the generic fallback, call a solver-specific implementation, or reject
        the requested coupling mode. Omitted hooks are treated as
        :attr:`CouplingCapability.DEFAULT`, which means the shared model/state
        fallback is adequate.

        The default fallbacks are:

        * :attr:`CouplingHooks.BODY_PROXY_REWIND_VELOCITY`: rewind proxy
          body velocity by removing previously applied lagged feedback.
        * :attr:`CouplingHooks.PARTICLE_PROXY_REWIND_VELOCITY`: rewind proxy
          particle velocity by removing previously applied lagged feedback.
        * :attr:`CouplingHooks.BODY_PROXY_HARVEST`: harvest momentum change
          from proxy body velocity.
        * :attr:`CouplingHooks.PARTICLE_PROXY_HARVEST`: harvest momentum
          change from proxy particle velocity.
        * :attr:`CouplingHooks.EFFECTIVE_MASS_DIAGONAL`: use model
          mass/inertia.
        * :attr:`CouplingHooks.EFFECTIVE_MASS_BLOCK`: use model mass and full
          body inertia, or the diagonal/model fallback when the caller can
          accept it; otherwise the caller may reject the coupling mode.
        * :attr:`CouplingHooks.NOTIFY_INPUT_STATE_UPDATE`: no-op after public
          input state arrays, including force buffers, are updated; custom
          solvers can sync private buffers or rewrite the public state to match
          solver history.
        * :attr:`CouplingHooks.PROXY_CONTACT_PREPARE`: use supplied contacts
          with generic proxy contact filtering.

        Coupled wrappers mutate :class:`~newton.solvers.ModelView` directly for
        model-view overrides such as proxy mass and inertia, and notify solvers
        through :meth:`notify_model_changed` when those overrides are applied
        after construction. Solvers that need private buffers, private history,
        or custom state mutation for one of the operations below return
        :attr:`CouplingCapability.CUSTOM` for the corresponding
        :class:`CouplingHooks` member. A solver may return
        :attr:`CouplingCapability.UNSUPPORTED` when neither the generic
        fallback nor a custom implementation is valid for that hook.

        Returns:
            Mapping from coupling hook to solver support level.
        """
        return {}

    def coupling_capability(self, hook: CouplingHooks) -> CouplingCapability:
        """Return this solver's support level for one coupling hook.

        Args:
            hook: Coupling dispatch point queried by a coupled wrapper.

        Returns:
            Hook support level. Missing entries default to
            :attr:`CouplingCapability.DEFAULT`.
        """
        return self.coupling_capabilities().get(hook, SolverBase.CouplingCapability.DEFAULT)

    def coupling_eval_effective_mass(
        self,
        endpoint_kind: wp.array[int],
        endpoint_index: wp.array[int],
        endpoint_local_pos: wp.array[wp.vec3],
        out: wp.array[float],
    ) -> None:
        """Evaluate solver-private effective mass for coupling endpoints.

        ADMM penalties and proxy virtual masses can be scaled by endpoint
        inertia, and some solvers know a better effective mass than the shared
        :class:`~newton.Model` exposes, for example an articulated rigid-body
        solver with internal composite inertia. Those solvers should advertise
        a custom
        :attr:`CouplingHooks.EFFECTIVE_MASS_DIAGONAL` capability and
        write one mass value per requested endpoint into ``out``.

        Endpoint data is passed in structure-of-arrays form. All endpoint
        arrays have the same length as ``out``. ``endpoint_kind`` contains
        :class:`CouplingEndpointKind` values, ``endpoint_index`` contains
        model-view body or particle ids, and ``endpoint_local_pos`` stores the
        body-frame point for body endpoints [m]. Particle endpoint local
        positions are zero.

        Generic model-based fallbacks live in
        :class:`~newton.solvers.SolverCoupled`; coupling modes can use those
        fallbacks without calling this method. Override this only when the
        result depends on solver-private data that cannot be reconstructed from
        the model view.

        Args:
            endpoint_kind: Endpoint kind values.
            endpoint_index: Endpoint body or particle indices.
            endpoint_local_pos: Endpoint local positions [m].
            out: Output mass buffer indexed like the endpoint arrays [kg].
        """
        raise NotImplementedError

    def coupling_eval_effective_mass_block(
        self,
        endpoint_kind: wp.array[int],
        endpoint_index: wp.array[int],
        endpoint_local_pos: wp.array[wp.vec3],
        out_mass: wp.array[float],
        out_inertia: "wp.array[wp.mat33] | None" = None,
    ) -> None:
        """Evaluate solver-private block effective mass for coupling endpoints.

        Body proxy virtual inertia can use a full local-frame inertia tensor
        rather than a scalar mass with a model-derived tensor shape. Solvers
        with private articulated or condensed inertia should advertise a custom
        :attr:`CouplingHooks.EFFECTIVE_MASS_BLOCK` capability and write one
        scalar mass per endpoint to ``out_mass``. For body endpoints they should
        also write the corresponding local body inertia tensors [kg*m^2] to
        ``out_inertia``.

        Endpoint data is passed in structure-of-arrays form. All endpoint
        arrays have the same length as ``out_mass``. ``endpoint_kind`` contains
        :class:`CouplingEndpointKind` values, ``endpoint_index`` contains
        model-view body or particle ids, and ``endpoint_local_pos`` stores the
        body-frame point for body endpoints [m]. Particle endpoint local
        positions are zero.

        Generic model-based fallbacks live in
        :class:`~newton.solvers.SolverCoupled`; coupling modes can use those
        fallbacks without calling this method.

        Args:
            endpoint_kind: Endpoint kind values.
            endpoint_index: Endpoint body or particle indices.
            endpoint_local_pos: Endpoint local positions [m].
            out_mass: Output mass buffer indexed like the endpoint arrays [kg].
            out_inertia: Optional output body inertia tensor buffer indexed
                like the endpoint arrays [kg*m^2]. Entries for particle
                endpoints are ignored.
        """
        raise NotImplementedError

    def coupling_notify_input_state_update(
        self,
        state: State,
        flags: CouplingInputStateFlags | int,
        dt: float = 0.0,
    ) -> None:
        """Notify a solver that the coupler updated input state arrays.

        Custom solvers that mark
        :attr:`CouplingHooks.NOTIFY_INPUT_STATE_UPDATE` as
        :attr:`CouplingCapability.CUSTOM` should implement this hook. Coupled
        wrappers call it after copying or mutating input arrays on ``state`` for
        the current sub-solve. The default fallback is no extra action because
        the public state already contains the updated values. Custom
        implementations can synchronize solver-private buffers or rewrite the
        public state to represent the update in solver-native form.

        Args:
            state: Solver input state after the update.
            flags: Bitmask of :class:`CouplingInputStateFlags` entries
                indicating which arrays were updated. The optional
                :attr:`CouplingInputStateFlags.ITERATION_RESTART` bit marks a
                repeated coupling iteration over the same top-level step.
            dt: Coupling time step [s] for updates that represent a
                time-discretized target.
        """
        raise NotImplementedError

    def coupling_harvest_proxy_wrenches(
        self,
        body_local_to_proxy_global: wp.array[int],
        out_body_f: wp.array[wp.spatial_vector],
        *,
        state: State | None = None,
        state_out: State | None = None,
        contacts: Contacts | None = None,
        dt: float = 0.0,
    ) -> None:
        """Harvest solver-private proxy feedback wrenches.

        This hook belongs to proxy coupling and is called by
        :class:`~newton.solvers.SolverProxyCoupled` after the destination
        solver has stepped the proxy bodies. It exists for solvers whose
        coupling feedback is not represented by the proxy body's velocity
        change alone. Examples include solvers that accumulate contact forces
        in private buffers or, for MPM, return collider impulses at grid
        locations.

        ``body_local_to_proxy_global`` is a dense map indexed by local body id
        in this solver's model view. Proxy bodies map to the global proxy body
        id in the parent model; non-proxy entries are ``-1`` and skipped.
        ``state``, ``state_out``, and solver-private buffers use local ids,
        while ``out_body_f`` is indexed by global proxy body id. The generic
        proxy coupler maps those proxy-indexed values back onto source ids when
        applying feedback to the driving solver on the next proxy pass.
        Solvers that leave :attr:`CouplingHooks.BODY_PROXY_HARVEST` at the
        default capability do not need this hook; ``SolverProxyCoupled`` falls
        back to estimating the wrench from proxy momentum change with the
        destination model view's mass and inertia.

        Args:
            body_local_to_proxy_global: Dense map from local body id to global
                proxy body id; non-proxy entries are ``-1``.
            out_body_f: Spatial wrench buffer indexed by global proxy body id.
            state: Prepared destination input state used for the proxy solve.
            state_out: Destination output state after the proxy solve.
            contacts: Contact data used by the proxy solve.
            dt: Coupling time step [s].
        """
        raise NotImplementedError

    def coupling_rewind_proxy_body_velocity(
        self,
        body_local_to_proxy_global: wp.array[int],
        state: State,
        coupling_forces: wp.array[wp.spatial_vector],
        dt: float,
    ) -> None:
        """Rewind destination proxy body velocities before a proxy solve.

        This hook belongs to proxy coupling and is called by
        :class:`~newton.solvers.SolverProxyCoupled` after it synchronizes
        source body poses and velocities into destination proxy bodies, but
        before it updates destination contacts or steps the destination solver.
        It exists for solver-specific velocity rewinds that require private
        solver knowledge.

        The common rewind removes the feedback wrench, public body force input,
        and gravity that the synchronized source velocity already includes
        when running lagged proxy coupling, so the destination solver does not
        see the same acceleration twice. If a solver leaves
        :attr:`CouplingHooks.BODY_PROXY_REWIND_VELOCITY` at the
        default capability, ``SolverProxyCoupled`` applies the generic
        lagged-mode velocity rewind directly from the destination model view.
        Solvers that need to synchronize private buffers or convert a proxy
        pose update into a velocity target should use
        :attr:`CouplingHooks.NOTIFY_INPUT_STATE_UPDATE`.

        Args:
            body_local_to_proxy_global: Dense map from local body id to global
                proxy body id; non-proxy entries are ``-1``.
            state: Destination state after source-to-proxy synchronization.
            coupling_forces: Spatial forces previously applied to source
                bodies, indexed by global proxy body id.
            dt: Coupling time step [s].
        """
        pass

    def coupling_harvest_proxy_particle_forces(
        self,
        particle_local_to_proxy_global: wp.array[int],
        out_particle_f: wp.array[wp.vec3],
        *,
        state: State | None = None,
        state_out: State | None = None,
        contacts: Contacts | None = None,
        dt: float = 0.0,
    ) -> None:
        """Harvest solver-private proxy feedback forces for particles.

        This hook belongs to proxy coupling and is called by
        :class:`~newton.solvers.SolverProxyCoupled` after the destination
        solver has stepped proxy particles. It is the particle-side
        counterpart of :meth:`coupling_harvest_proxy_wrenches` for solvers
        whose contact or constraint feedback is accumulated in private buffers
        rather than being represented solely by the proxy particle velocity
        change.

        ``particle_local_to_proxy_global`` is a dense map indexed by local
        particle id in this solver's model view. Proxy particles map to the
        global proxy particle id in the parent model; non-proxy entries are
        ``-1`` and skipped. ``state``, ``state_out``, and solver-private
        buffers use local ids, while ``out_particle_f`` is indexed by global
        proxy particle id. The generic proxy coupler maps those proxy-indexed
        values back onto source ids when applying feedback to the driving
        solver on the next proxy pass. Solvers that leave
        :attr:`CouplingHooks.PARTICLE_PROXY_HARVEST` at the default capability
        do not need this hook; ``SolverProxyCoupled`` falls back to estimating
        the force from proxy momentum change with the destination model view's
        particle mass.

        Args:
            particle_local_to_proxy_global: Dense map from local particle id
                to global proxy particle id; non-proxy entries are ``-1``.
            out_particle_f: Force buffer indexed by global proxy particle id.
            state: Prepared destination input state used for the proxy solve.
            state_out: Destination output state after the proxy solve.
            contacts: Contact data used by the proxy solve.
            dt: Coupling time step [s].
        """
        raise NotImplementedError

    def coupling_rewind_proxy_particle_velocity(
        self,
        particle_local_to_proxy_global: wp.array[int],
        state: State,
        coupling_forces: wp.array[wp.vec3],
        dt: float,
    ) -> None:
        """Rewind destination proxy particle velocities before a proxy solve.

        This hook belongs to proxy coupling and is called by
        :class:`~newton.solvers.SolverProxyCoupled` after it synchronizes
        source particle positions and velocities into destination proxy
        particles, but before stepping the destination solver. It mirrors
        :meth:`coupling_rewind_proxy_body_velocity` for
        particle-owned solver state.

        The common rewind removes the feedback force, public particle force
        input, and gravity that the synchronized source velocity already
        includes when running lagged proxy coupling, so the destination solver
        does not see the same acceleration twice. Solvers may also update
        private proxy-particle state here. If a solver leaves
        :attr:`CouplingHooks.PARTICLE_PROXY_REWIND_VELOCITY` at the
        default capability, ``SolverProxyCoupled`` applies the generic
        lagged-mode velocity rewind directly from the destination model view.
        Solvers that only need to synchronize the resulting velocity target
        into private buffers should use
        :attr:`CouplingHooks.NOTIFY_INPUT_STATE_UPDATE`.

        Args:
            particle_local_to_proxy_global: Dense map from local particle id
                to global proxy particle id; non-proxy entries are ``-1``.
            state: Destination state after source-to-proxy synchronization.
            coupling_forces: Forces previously applied to source particles,
                indexed by global proxy particle id.
            dt: Coupling time step [s].
        """
        pass

    def coupling_prepare_proxy_contacts(
        self,
        state: State,
        contacts: Contacts | None,
        *,
        contacts_freshly_detected: bool = False,
    ) -> Contacts | None:
        """Return the contacts to use for a destination proxy solve.

        This hook belongs to proxy coupling and is called by
        :class:`~newton.solvers.SolverProxyCoupled` after proxy velocity
        rewind and before the destination solver step. Its main
        responsibility is to make the contact set valid for the destination
        solver's ownership boundary: proxy bodies should only collide with
        objects whose response is owned by that destination solve and whose
        reaction can be harvested back to the source body. Contacts between
        proxies and non-owned dynamic bodies, and other proxy contacts that
        cannot produce meaningful source feedback, must be filtered out.

        Custom solvers that mark
        :attr:`CouplingHooks.PROXY_CONTACT_PREPARE` as
        :attr:`CouplingCapability.CUSTOM` should implement this hook. Solvers
        that use externally supplied contacts can leave that hook at the
        default capability; ``SolverProxyCoupled`` will use ``contacts`` and
        apply its generic proxy contact filter.

        Custom solvers that need a proxy-local collision pipeline should pass a
        pipeline factory through :class:`SolverProxyCoupled.Proxy`; this hook is
        then called after ``SolverProxyCoupled`` has refreshed those contacts.
        ``contacts_freshly_detected`` is true only on passes where the proxy
        collision pipeline actually ran, which lets solvers update contact
        history or filtering work only on refreshed contact buffers. When a
        custom implementation returns the original externally supplied contacts
        object, ``SolverProxyCoupled`` still applies its generic proxy contact
        filter.

        Args:
            state: Destination state after proxy synchronization.
            contacts: Contacts to use for the destination proxy solve.
            contacts_freshly_detected: Whether ``contacts`` was just refreshed
                by a proxy-owned collision pipeline.

        Returns:
            Contacts object to pass to this destination solver.
        """
        del state, contacts_freshly_detected
        return contacts

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """
        Register custom attributes for the solver.

        Args:
            builder (ModelBuilder): The model builder to register the custom attributes to.
        """
        pass
