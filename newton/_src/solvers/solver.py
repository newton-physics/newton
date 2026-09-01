# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Any, ClassVar

import warp as wp

from ..core.reset import normalize_reset_world_mask
from ..geometry import ParticleFlags
from ..sim import BodyFlags, Contacts, Control, Model, ModelBuilder, ModelFlags, State, StateFlags


class SolverOutputFlags(Enum):
    """Standard output quantities that may be requested from a solver.

    Requests are composed as a :class:`set` rather than a bit mask so that
    solver-specific enums can add entries without coordinating integer bits
    with Newton or other solver implementations.

    .. experimental::

        The solver output API may change while additional solvers and output
        categories are migrated to it.
    """

    BODY_QDD = "body_qdd"
    """Rigid-body spatial accelerations."""

    BODY_PARENT_F = "body_parent_f"
    """Incoming parent-joint wrenches on rigid bodies."""

    CONTACT_F = "contact_f"
    """Spatial contact forces aligned with a :class:`~newton.Contacts` container."""


class SolverOutputs:
    """Arrays populated by a solver in addition to the simulation state.

    Instances are allocated by :meth:`SolverBase.outputs` and may be reused
    across steps. Solver implementations can derive from this class to add
    solver-specific arrays while retaining the standard Newton outputs.

    .. experimental::

        The solver output API may change while additional solvers and output
        categories are migrated to it.
    """

    def __init__(self, flags: Iterable[Enum] = ()) -> None:
        """Initialize an unallocated output container.

        Args:
            flags: Output flags represented by this container.
        """
        self.flags: frozenset[Enum] = frozenset(flags)
        """Output flags allocated in this container."""

        self.body_qdd: wp.array[wp.spatial_vector] | None = None
        """Rigid-body accelerations [m/s², rad/s²], shape ``(body_count,)``."""

        self.body_parent_f: wp.array[wp.spatial_vector] | None = None
        """Incoming parent-joint wrenches [N, N·m], shape ``(body_count,)``."""

        self.contact_f: wp.array[wp.spatial_vector] | None = None
        """Contact forces [N, N·m], shape ``(rigid_contact_max + soft_contact_max,)``."""

        self._solver: SolverBase | None = None
        self._contacts: Contacts | None = None

    def __contains__(self, flag: Enum) -> bool:
        """Return whether an output flag was allocated."""
        return flag in self.flags

    @property
    def contacts(self) -> Contacts | None:
        """Contact storage associated with contact-indexed outputs, if any."""
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

    _module_options_revision = 0
    OUTPUTS_TYPE: ClassVar[type[SolverOutputs]] = SolverOutputs
    """Container type returned by :meth:`outputs`."""

    SUPPORTED_OUTPUT_FLAGS: ClassVar[frozenset[Enum]] = frozenset()
    """Output flags accepted by :meth:`outputs`."""

    def __init__(self, model: Model):
        self.model = model
        self._module_options: dict[Any, dict[str, Any]] = {}
        self._applied_module_options_revision = -1

    @property
    def supported_output_flags(self) -> frozenset[Enum]:
        """Output flags supported by this solver instance."""
        return self.SUPPORTED_OUTPUT_FLAGS

    def outputs(
        self,
        flags: Iterable[Enum],
        *,
        contacts: Contacts | None = None,
        requires_grad: bool | None = None,
    ) -> SolverOutputs:
        """Allocate reusable arrays for requested solver outputs.

        A container is owned by the solver that allocates it and can be passed
        to that solver's :meth:`step` method on every time step. Derived
        solvers add custom flags to :attr:`SUPPORTED_OUTPUT_FLAGS`, derive a
        container from :class:`SolverOutputs`, and override
        :meth:`_allocate_outputs` for their custom arrays.

        Args:
            flags: Set or other iterable of standard and solver-specific output
                enum members.
            contacts: Contact storage whose capacities determine the shape of
                contact-indexed outputs. Required when requesting
                :attr:`SolverOutputFlags.CONTACT_F`.
            requires_grad: Whether allocated arrays require gradients. If
                ``None``, use the model's setting.

        Returns:
            A solver-owned output container with requested arrays allocated.

        Raises:
            TypeError: If a request is not a plain enum member or the
                configured output type does not derive from
                :class:`SolverOutputs`.
            ValueError: If this solver does not support a requested output.

        .. experimental::

            The solver output API may change while additional solvers and
            output categories are migrated to it.
        """
        requested = frozenset(flags)
        invalid = [flag for flag in requested if not isinstance(flag, Enum) or isinstance(flag, (int, str))]
        if invalid:
            values = ", ".join(repr(flag) for flag in invalid)
            raise TypeError(
                "Solver output flags must be plain enum.Enum members, not strings, integers, IntEnum members, "
                f"or string-mixin enum members; got: {values}."
            )

        unsupported = requested.difference(self.supported_output_flags)
        if unsupported:
            names = ", ".join(self._format_output_flag(flag) for flag in unsupported)
            raise ValueError(f"{type(self).__name__} does not support solver output(s): {names}.")

        if not issubclass(self.OUTPUTS_TYPE, SolverOutputs):
            raise TypeError("OUTPUTS_TYPE must derive from SolverOutputs.")
        result = self.OUTPUTS_TYPE(requested)
        result._solver = self
        result._contacts = contacts
        if requires_grad is None:
            requires_grad = self.model.requires_grad
        if SolverOutputFlags.CONTACT_F in requested:
            if contacts is None:
                raise ValueError("'contacts' is required when requesting SolverOutputFlags.CONTACT_F.")
            if contacts.device != self.model.device:
                raise ValueError("Solver outputs and contacts must be allocated on the solver device.")
        self._allocate_outputs(result, requires_grad=requires_grad)
        return result

    @staticmethod
    def _format_output_flag(flag: Enum) -> str:
        """Format an output flag for diagnostics."""
        return f"{type(flag).__name__}.{flag.name}"

    def _allocate_outputs(self, outputs: SolverOutputs, *, requires_grad: bool) -> None:
        """Allocate standard arrays requested in an output container."""
        if SolverOutputFlags.BODY_QDD in outputs:
            outputs.body_qdd = wp.zeros(
                self.model.body_count,
                dtype=wp.spatial_vector,
                device=self.model.device,
                requires_grad=requires_grad,
            )
        if SolverOutputFlags.BODY_PARENT_F in outputs:
            outputs.body_parent_f = wp.zeros(
                self.model.body_count,
                dtype=wp.spatial_vector,
                device=self.model.device,
                requires_grad=requires_grad,
            )

        if SolverOutputFlags.CONTACT_F in outputs:
            contacts = outputs._contacts
            if contacts is None:
                raise ValueError("Contact storage is missing for SolverOutputFlags.CONTACT_F.")
            outputs.contact_f = wp.zeros(
                contacts.rigid_contact_max + contacts.soft_contact_max,
                dtype=wp.spatial_vector,
                device=self.model.device,
                requires_grad=requires_grad,
            )

    def _validate_outputs(self, outputs: SolverOutputs | None, contacts: Contacts | None = None) -> None:
        """Validate that an output container belongs to this solver."""
        if outputs is None:
            return
        if not isinstance(outputs, self.OUTPUTS_TYPE):
            raise TypeError(f"'outputs' must be an instance of {self.OUTPUTS_TYPE.__name__}.")
        if outputs._solver is not self:
            raise ValueError("Solver outputs must be passed to the solver instance that allocated them.")
        if outputs.contact_f is not None and contacts is not None and outputs._contacts is not contacts:
            raise ValueError("Contact solver outputs must be used with the Contacts instance that sized them.")

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
        outputs: SolverOutputs | None = None,
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
            outputs: Optional solver output arrays allocated by :meth:`outputs`.
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

            Request :attr:`SolverOutputFlags.CONTACT_F` using :meth:`outputs`
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
