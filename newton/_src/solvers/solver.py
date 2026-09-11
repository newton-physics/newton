# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum, IntEnum
from typing import Any, ClassVar

import warp as wp

from ..core.reset import normalize_reset_world_mask
from ..geometry import ParticleFlags
from ..sim import BodyFlags, CollisionPipeline, Contacts, Control, Model, ModelBuilder, ModelFlags, State, StateFlags


class SolverObservableFlags(Enum):
    """Standard observable quantities that may be requested from a solver.

    Requests are composed as a :class:`set` rather than a bit mask so that
    solver-specific enums can add entries without coordinating integer bits
    with Newton or other solver implementations.

    .. experimental::

        The solver observable API may change while additional solvers and observable
        categories are migrated to it.
    """

    BODY_QDD = "body_qdd"
    """Rigid-body spatial accelerations."""

    BODY_PARENT_F = "body_parent_f"
    """Incoming parent-joint wrenches on rigid bodies."""

    CONTACT_F = "contact_f"
    """Spatial contact forces aligned with a :class:`~newton.Contacts` container."""


class SolverObservables:
    """Arrays populated by a solver in addition to the simulation state.

    Instances are allocated by :meth:`SolverBase.observables` and may be reused
    across steps. Solver implementations can derive from this class to add
    solver-specific arrays while retaining the standard Newton observables.

    .. experimental::

        The solver observable API may change while additional solvers and observable
        categories are migrated to it.
    """

    def __init__(self, flags: Iterable[Enum] = ()) -> None:
        """Initialize an unallocated observable container.

        Args:
            flags: Observable flags represented by this container.
        """
        self.flags: frozenset[Enum] = frozenset(flags)
        """Observable flags allocated in this container."""

        self.body_qdd: wp.array[wp.spatial_vector] | None = None
        """Rigid-body accelerations [m/s², rad/s²], shape ``(body_count,)``."""

        self.body_parent_f: wp.array[wp.spatial_vector] | None = None
        """Incoming parent-joint wrenches [N, N·m], shape ``(body_count,)``."""

        self.contact_f: wp.array[wp.spatial_vector] | None = None
        """Contact forces [N, N·m], shape ``(rigid_contact_max + soft_contact_max,)``."""

        self._solver: SolverBase | None = None
        self._contacts: Contacts | None = None
        self._contact_capacity: tuple[int, int] | None = None

    def __contains__(self, flag: Enum) -> bool:
        """Return whether an observable flag was allocated."""
        return flag in self.flags

    @property
    def contacts(self) -> Contacts | None:
        """Contact storage bound on the first solver step, or ``None`` before it."""
        return self._contacts


def _set_module_options_if_changed(options: dict[str, Any], module: Any) -> bool:
    current_options = wp.get_module_options(module=module)
    if any(current_options.get(name) != value for name, value in options.items()):
        wp.set_module_options(options, module=module)
        return True
    return False


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
    world_g = gravity[world_idx]

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
    world_g = gravity[world_idx]

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

    class CollisionSlot(IntEnum):
        """Collision-detection categories scheduled by a solver."""

        RIGID = 0
        """Rigid-rigid and particle-shape collision detection."""
        SOFT_SELF_CONTACT = 1
        """Triangle-mesh soft self-contact detection."""

    class CollisionFrequencyType(IntEnum):
        """When, inside a :meth:`step`, a solver-owned collision pipeline runs detection.

        The frequency number in ``collision_frequency`` applies only to
        :attr:`ITERATIONS`; the other members ignore it. Skipping detection
        across steps carries no hidden solver state — set a slot to
        :attr:`NONE` between steps via :meth:`set_collision_frequency`.
        """

        NONE = 0
        """Never detect; the user may run detection externally into :attr:`contacts`."""
        PRE_INIT = 1
        """Once per step, before solver initialization."""
        PRE_POST_INIT = 2
        """Before and after solver initialization (one detection each)."""
        ITERATIONS = 3
        """Before initialization, then immediately before iterations k, 2k, and so on."""
        AUTO = 4
        """Solver-specific default."""

    supports_collision_pipeline: bool = False
    """Whether this solver can own a :class:`~newton.CollisionPipeline` and drive detection itself.

    Currently only :class:`~newton.solvers.SolverVBD` opts in; passing
    ``collision_pipeline`` to any other solver raises ``ValueError`` (drive
    detection externally instead).
    """

    _module_options_revision = 0
    OBSERVABLES_TYPE: ClassVar[type[SolverObservables]] = SolverObservables
    """Container type returned by :meth:`observables`."""

    SUPPORTED_OBSERVABLE_FLAGS: ClassVar[frozenset[Enum]] = frozenset()
    """Observable flags accepted by :meth:`observables`."""

    CONTACT_OBSERVABLE_FLAGS: ClassVar[frozenset[Enum]] = frozenset({SolverObservableFlags.CONTACT_F})
    """Flags requiring contact capacities and storage binding.

    Custom solvers extend this set for their own contact-indexed arrays. This
    declares a sizing dependency, not support; also add each custom flag to
    :attr:`SUPPORTED_OBSERVABLE_FLAGS`.
    """

    def __init__(
        self,
        model: Model,
        *,
        collision_pipeline: CollisionPipeline | None = None,
        collision_frequency: Mapping[CollisionSlot, int] | None = None,
        collision_frequency_type: Mapping[CollisionSlot, CollisionFrequencyType] | None = None,
    ):
        """Initialize common solver state and optional collision scheduling.

        Args:
            model: Simulation model integrated by the solver.
            collision_pipeline: Collision pipeline owned and driven by the
                solver. The pipeline must use ``model``, and the concrete
                solver must set :attr:`supports_collision_pipeline`.
            collision_frequency: Per-slot iteration frequencies. Values must
                be at least one and are used only for slots scheduled with
                :attr:`CollisionFrequencyType.ITERATIONS`. Unspecified slots
                retain their defaults.
            collision_frequency_type: Per-slot detection points. Unspecified
                slots retain their defaults.
        """
        self.model = model
        self._module_options: dict[Any, dict[str, Any]] = {}
        self._applied_module_options_revision = -1

        if collision_pipeline is not None and not self.supports_collision_pipeline:
            raise ValueError(
                f"{type(self).__name__} cannot own a collision pipeline; "
                "drive detection externally via model.collide()."
            )
        if collision_pipeline is not None and collision_pipeline.model is not model:
            raise ValueError("collision_pipeline and solver must use the same model")
        self.collision_pipeline = collision_pipeline
        """The solver-owned collision pipeline, or ``None`` when detection is driven externally."""
        if collision_pipeline is not None:
            self._pipeline_contacts = collision_pipeline.contacts()
        elif not hasattr(self, "_pipeline_contacts"):
            # Preserve contact storage assigned by existing SolverBase subclasses
            # before calling super().__init__().
            self._pipeline_contacts = None

        self._collision_frequency = dict.fromkeys(SolverBase.CollisionSlot, 1)
        self._collision_frequency_type = dict.fromkeys(SolverBase.CollisionSlot, SolverBase.CollisionFrequencyType.AUTO)
        self.set_collision_frequency(
            collision_frequency=collision_frequency,
            collision_frequency_type=collision_frequency_type,
        )

    @property
    def contacts(self) -> Contacts | None:
        """The solver-owned contacts buffer, or ``None`` when no pipeline is owned.

        Unlike :meth:`Model.contacts`, this property does not allocate; it
        returns the buffer created from the owned pipeline at construction.
        With a slot set to ``CollisionFrequencyType.NONE`` the user may fill
        this buffer externally, e.g. ``pipeline.collide(state, solver.contacts)``.
        """
        return self._pipeline_contacts

    @contacts.setter
    def contacts(self, value: Contacts | None) -> None:
        """Set contact storage for compatibility with existing solver subclasses."""
        self._pipeline_contacts = value

    @property
    def collision_frequency(self) -> dict[CollisionSlot, int]:
        """Per-slot detection frequency numbers as a read-only copy."""
        return dict(self._collision_frequency)

    @property
    def collision_frequency_type(self) -> dict[CollisionSlot, CollisionFrequencyType]:
        """Per-slot :class:`CollisionFrequencyType` values as a read-only copy."""
        return dict(self._collision_frequency_type)

    def set_collision_frequency(
        self,
        *,
        collision_frequency: Mapping[CollisionSlot, int] | None = None,
        collision_frequency_type: Mapping[CollisionSlot, CollisionFrequencyType] | None = None,
    ) -> None:
        """Change the detection schedule; takes effect at the next :meth:`step`.

        The solver keeps no hidden cross-step scheduling state, so detecting
        every N steps is expressed by toggling a slot between
        ``CollisionFrequencyType.NONE`` and an active type from the calling
        loop. ``None`` keeps the corresponding current setting. Recapture an
        existing CUDA graph after changing the schedule.

        Args:
            collision_frequency: Frequency numbers keyed by
                :class:`CollisionSlot`; used only by ``ITERATIONS`` slots
                (before iterations k, 2k, and so on) and must be at least one.
            collision_frequency_type: Detection points keyed by
                :class:`CollisionSlot`.
        """
        Slot = SolverBase.CollisionSlot
        Frequency = SolverBase.CollisionFrequencyType
        freq = dict(self._collision_frequency)
        if collision_frequency is not None:
            for slot_key, frequency_value in collision_frequency.items():
                slot = Slot(slot_key)
                frequency = int(frequency_value)
                if frequency < 1:
                    raise ValueError(f"collision_frequency[{slot.name}] must be >= 1, got {frequency}")
                freq[slot] = frequency

        ftype = dict(self._collision_frequency_type)
        if collision_frequency_type is not None:
            for slot, value in collision_frequency_type.items():
                ftype[Slot(slot)] = Frequency(value)
            if self.collision_pipeline is None and ftype[Slot.RIGID] not in (
                Frequency.NONE,
                Frequency.AUTO,
            ):
                raise ValueError(
                    "an active rigid collision_frequency_type requires a solver-owned pipeline; "
                    "pass collision_pipeline=... at construction or drive model.collide() externally."
                )
            if ftype[Slot.RIGID] == Frequency.ITERATIONS and self.collision_pipeline.contact_matching == "disabled":
                raise ValueError(
                    "rigid ITERATIONS collision scheduling requires contact matching so in-flight "
                    "contact state can be carried across re-detection; construct collision_pipeline "
                    "with contact_matching='latest' or 'sticky'."
                )

        self._collision_frequency = freq
        self._collision_frequency_type = ftype

    def _default_collision_frequency_type(self, slot: CollisionSlot) -> CollisionFrequencyType:
        """Resolve ``AUTO`` for a slot; overridable per solver."""
        if slot == SolverBase.CollisionSlot.RIGID and self.collision_pipeline is not None:
            return SolverBase.CollisionFrequencyType.PRE_INIT
        return SolverBase.CollisionFrequencyType.NONE

    def _resolved_collision_frequency_type(self, slot: CollisionSlot) -> CollisionFrequencyType:
        """The slot's effective type with ``AUTO`` resolved."""
        ftype = self._collision_frequency_type[slot]
        if ftype == SolverBase.CollisionFrequencyType.AUTO:
            return self._default_collision_frequency_type(slot)
        return ftype

    def _resolve_step_contacts(self, contacts: Contacts | None) -> Contacts | None:
        """Return the contacts buffer for this step; owning solvers call this first.

        With an owned pipeline the ``contacts`` argument must be ``None`` and
        the owned buffer is used (exactly one source of contact data).
        """
        if self.collision_pipeline is not None:
            if contacts is not None:
                raise ValueError(
                    "step(contacts=...) must be None when the solver owns a collision "
                    "pipeline; the solver detects into its own buffer (solver.contacts)."
                )
            return self._pipeline_contacts
        return contacts

    def _run_rigid_collision(self, state: State, dt: float | None = None) -> None:
        """Run the owned pipeline into the owned contacts buffer."""
        self.collision_pipeline.collide(state, self._pipeline_contacts, dt=dt)

    @property
    def supported_observable_flags(self) -> frozenset[Enum]:
        """Observable flags supported by this solver instance."""
        return self.SUPPORTED_OBSERVABLE_FLAGS

    def observables(
        self,
        flags: Iterable[Enum],
        *,
        requires_grad: bool | None = None,
    ) -> SolverObservables:
        """Allocate reusable arrays for requested solver observables.

        A container is owned by the solver that allocates it and can be passed
        to that solver's :meth:`step` method on every time step. Derived
        solvers add custom flags to :attr:`SUPPORTED_OBSERVABLE_FLAGS`, derive a
        container from :class:`SolverObservables`, and override
        :meth:`_allocate_observables` for their custom arrays.

        Args:
            flags: Set or other iterable of standard and solver-specific observable
                enum members.
            requires_grad: Whether allocated arrays require gradients. If
                ``None``, use the model's setting.

        Returns:
            A solver-owned observable container with requested arrays allocated.

        Raises:
            TypeError: If a request is not a plain enum member or the
                configured observable type does not derive from
                :class:`SolverObservables`.
            ValueError: If this solver does not support a requested observable.
            RuntimeError: If contact-indexed observables are requested before
                constructing :class:`~newton.CollisionPipeline` for the model.

        All requested arrays are allocated before this method returns; ``None``
        always means unrequested. Contact arrays use the model's resolved rigid
        and soft capacities, not the live contact count. Allocate before graph
        capture and pass matching :class:`~newton.Contacts` to :meth:`step`.

        .. experimental::

            The solver observable API may change while additional solvers and
            observable categories are migrated to it.
        """
        requested = frozenset(flags)
        invalid = [flag for flag in requested if not isinstance(flag, Enum) or isinstance(flag, (int, str))]
        if invalid:
            values = ", ".join(repr(flag) for flag in invalid)
            raise TypeError(
                "Solver observable flags must be plain enum.Enum members, not strings, integers, IntEnum members, "
                f"or string-mixin enum members; got: {values}."
            )

        unsupported = requested.difference(self.supported_observable_flags)
        if unsupported:
            names = ", ".join(self._format_observable_flag(flag) for flag in unsupported)
            raise ValueError(f"{type(self).__name__} does not support solver observable(s): {names}.")

        if not issubclass(self.OBSERVABLES_TYPE, SolverObservables):
            raise TypeError("OBSERVABLES_TYPE must derive from SolverObservables.")
        observables = self.OBSERVABLES_TYPE(requested)
        observables._solver = self
        if requires_grad is None:
            requires_grad = self.model.requires_grad
        if requested.intersection(self.CONTACT_OBSERVABLE_FLAGS):
            observables._contact_capacity = self.model._get_contact_capacity()
        self._allocate_observables(observables, requires_grad=requires_grad)
        if observables._contact_capacity is not None:
            self.model._solver_observable_contact_capacity = observables._contact_capacity
        return observables

    @staticmethod
    def _format_observable_flag(flag: Enum) -> str:
        """Format an observable flag for diagnostics."""
        return f"{type(flag).__name__}.{flag.name}"

    def _allocate_observables(self, observables: SolverObservables, *, requires_grad: bool) -> None:
        """Allocate standard arrays requested in an observable container."""
        if SolverObservableFlags.BODY_QDD in observables:
            observables.body_qdd = wp.zeros(
                self.model.body_count,
                dtype=wp.spatial_vector,
                device=self.model.device,
                requires_grad=requires_grad,
            )
        if SolverObservableFlags.BODY_PARENT_F in observables:
            observables.body_parent_f = wp.zeros(
                self.model.body_count,
                dtype=wp.spatial_vector,
                device=self.model.device,
                requires_grad=requires_grad,
            )

        if SolverObservableFlags.CONTACT_F in observables:
            rigid_max, soft_max = observables._contact_capacity
            observables.contact_f = wp.zeros(
                rigid_max + soft_max,
                dtype=wp.spatial_vector,
                device=self.model.device,
                requires_grad=requires_grad,
            )

    def _validate_observables(self, observables: SolverObservables | None, contacts: Contacts | None = None) -> None:
        """Validate that an observable container belongs to this solver."""
        if observables is None:
            return
        if not isinstance(observables, self.OBSERVABLES_TYPE):
            raise TypeError(f"'observables' must be an instance of {self.OBSERVABLES_TYPE.__name__}.")
        if observables._solver is not self:
            raise ValueError("Solver observables must be passed to the solver instance that allocated them.")
        if observables._contact_capacity is not None:
            if contacts is None:
                raise ValueError("Pass Contacts to solver.step() when using contact-indexed solver observables.")
            if contacts.device != self.model.device:
                raise ValueError("Solver observables and Contacts must be on the solver device.")
            if (contacts.rigid_contact_max, contacts.soft_contact_max) != observables._contact_capacity:
                raise ValueError(f"Contacts capacities must match solver observables: {observables._contact_capacity}.")
            if observables._contacts is not None and observables._contacts is not contacts:
                raise ValueError(
                    "Contact solver observables must be used with the Contacts instance bound on the first step."
                )
            observables._contacts = contacts

    def _set_module_options(self, options: dict[str, Any], module: Any) -> None:
        self._module_options[module] = dict(options)
        if _set_module_options_if_changed(options, module):
            SolverBase._module_options_revision += 1
        self._applied_module_options_revision = SolverBase._module_options_revision

    def _apply_module_options(self) -> None:
        if self._applied_module_options_revision == SolverBase._module_options_revision:
            return

        changed = False
        for module, options in self._module_options.items():
            changed |= _set_module_options_if_changed(options, module)
        if changed:
            SolverBase._module_options_revision += 1
        self._applied_module_options_revision = SolverBase._module_options_revision

    def _normalize_reset_world_mask(self, world_mask: wp.array[wp.bool] | None) -> wp.array[wp.bool] | None:
        """Validate a reset mask and return the canonical shape."""
        return normalize_reset_world_mask(
            world_mask,
            world_count=int(self.model.world_count),
            device=self.model.device,
            allow_legacy=True,
        )

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
            model: The model to integrate.
            state_in: The input state.
            state_out: The output state.
            dt: The time step (typically in seconds).
            angular_damping: The angular damping factor.
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
            model: The model to integrate.
            state_in: The input state.
            state_out: The output state.
            dt: The time step (typically in seconds).
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

    def reset(
        self,
        state: State,
        world_mask: wp.array[wp.bool] | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        """Reset the solver internal state data.

        Modifies the given *state* in place.  Derived solvers override this
        to reset solver-specific internal buffers or custom state attributes
        when environments are reset (e.g. during RL training).

        The default implementation is a no-op so solvers that do not require
        special reset logic need not override this method.

        Args:
            state: The simulation state to reset (modified in place).
            world_mask: Optional boolean mask of shape ``(world_count + 1,)``
                specifying which worlds to reset. Entries before the last select
                local worlds by index, and the final entry selects global entities
                whose world is ``-1``. If ``None``, all local and global entities
                are reset.

                .. deprecated:: 1.5
                    Passing a mask with shape ``(world_count,)`` is deprecated.
                    Use shape ``(world_count + 1,)`` with a final ``False`` entry
                    to select local worlds only.
            flags: Optional :class:`~newton.StateFlags` or ``int`` bitmask controlling
                which state attributes need to be reset.  If ``None``, all
                state attributes are reset.
        """
        self._normalize_reset_world_mask(world_mask)

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
        *,
        observables: SolverObservables | None = None,
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
            observables: Optional solver observable arrays allocated by :meth:`observables`.
        """
        raise NotImplementedError()

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Notify the solver that parts of the :class:`~newton.Model` were modified.

        The *flags* argument is a bit-mask composed of the
        :class:`~newton.ModelFlags` enums or custom ``int`` bits.
        Each flag represents a category of model data that may have been
        updated after the solver was created.  Passing the appropriate
        combination of flags enables a solver implementation to refresh its
        internal buffers without having to recreate the whole solver object.
        Valid flags are:

        * ``ModelFlags.JOINT_PROPERTIES``: Joint transforms or coordinates
          have changed.
        * ``ModelFlags.JOINT_DOF_PROPERTIES``: Joint axis limits, targets,
          modes, DOF state, or force buffers have changed.
        * ``ModelFlags.BODY_PROPERTIES``: Rigid-body pose or velocity buffers
          have changed.
        * ``ModelFlags.BODY_INERTIAL_PROPERTIES``: Rigid-body mass or inertia
          tensors have changed.
        * ``ModelFlags.SHAPE_PROPERTIES``: Shape transforms or geometry have
          changed.
        * ``ModelFlags.MODEL_PROPERTIES``: Model global properties (e.g.,
          gravity) have changed.
        * ``ModelFlags.CONSTRAINT_PROPERTIES``: Constraint definitions,
          coefficients, or enable flags have changed.
        * ``ModelFlags.TENDON_PROPERTIES``: Tendon stiffness or related tendon
          properties have changed.
        * ``ModelFlags.ACTUATOR_PROPERTIES``: Actuator gains, biases, limits,
          or force properties have changed.

        Args:
            flags: Bit-mask of :class:`~newton.ModelFlags` or custom ``int``
                bits indicating which model properties changed.

        """
        pass

    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        """Update a legacy Contacts object with forces from the solver state.

        .. deprecated:: 1.6

            Request :attr:`SolverObservableFlags.CONTACT_F` using :meth:`observables`
            and pass the resulting container to :meth:`step` instead.

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
            builder: The model builder to register the custom attributes to.
        """
        pass
