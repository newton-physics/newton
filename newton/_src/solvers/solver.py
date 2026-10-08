# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from copy import copy
from dataclasses import Field, dataclass, fields
from dataclasses import field as dataclass_field
from enum import IntEnum
from types import MappingProxyType
from typing import Any, ClassVar

import warp as wp

from ..core.reset import normalize_reset_world_mask
from ..geometry import ParticleFlags
from ..sim import BodyFlags, CollisionPipeline, Contacts, Control, Model, ModelBuilder, ModelFlags, State, StateFlags


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


@dataclass(frozen=True)
class _ObservableField:
    """Immutable allocation metadata stored on a dataclass field."""

    kind: str | None
    dtype: type
    frequency: Model.AttributeFrequency | str


def _normalize_observable_kinds(kinds: Iterable[str]) -> frozenset[str]:
    """Validate a collection of names before allocation or subset selection."""
    if isinstance(kinds, str):
        raise TypeError("Pass observable kinds as a collection of strings, for example {'body_qdd'}.")
    requested = frozenset(kinds)
    invalid = [kind for kind in requested if not isinstance(kind, str) or not kind]
    if invalid:
        values = ", ".join(repr(kind) for kind in invalid)
        raise TypeError(f"Solver observable kinds must be nonempty strings; got: {values}.")
    return requested


class SolverBase:
    """Generic base class for solvers.

    The implementation provides helper kernels to integrate rigid bodies and
    particles. Concrete solver back-ends should derive from this class and
    override :py:meth:`step` as well as :py:meth:`notify_model_changed` where
    necessary.
    """

    class ObservableKind:
        """Inheritable names for standard and solver-specific observable kinds.

        Access kinds through ``solver.ObservableKind`` and compose requests as
        sets. This is an ordinary namespace class of string constants: solvers
        may subclass it and declare new entries, such as
        ``BODY_TEMPERATURE = "body_temperature"``. Inherited names remain available.
        Literal strings and these constants are interchangeable in requests.
        Kind strings must be unique within a solver's observable declarations.

        Subclasses are checked when defined. Every public attribute must be a
        nonempty string. Changing an inherited name's value or assigning one
        string to different names raises an error, including conflicts between
        multiple parents. Redeclaring the same name with the same value is allowed.
        Attributes beginning with an underscore are excluded from these checks.

        The available names do not imply solver support. Query
        :attr:`SolverBase.supported_observables` for the effective capabilities
        of a configured solver. Use that set for iteration or requesting all
        supported observables; the namespace itself is not iterable.

        See :ref:`solver_observables` for request and extension examples.

        .. experimental::

            The solver observable API may change without prior notice.
        """

        BODY_QDD = "body_qdd"
        """Rigid-body spatial accelerations."""

        BODY_PARENT_F = "body_parent_f"
        """Incoming parent-joint wrenches on rigid bodies."""

        CONTACT_F = "contact_f"
        """Spatial contact forces aligned with a :class:`~newton.Contacts` container."""

        def __init_subclass__(cls, **kwargs: Any) -> None:
            """Reject invalid or conflicting kind declarations when a subclass is defined."""
            super().__init_subclass__(**kwargs)
            kinds_by_name: dict[str, str] = {}
            names_by_kind: dict[str, str] = {}
            # Inspect every declaration so MRO shadowing cannot conceal parent conflicts.
            for base in reversed(cls.__mro__):
                for name, kind in vars(base).items():
                    if name.startswith("_"):
                        continue
                    if not isinstance(kind, str) or not kind:
                        raise TypeError(f"{cls.__qualname__}.{name} must be a nonempty string; got {kind!r}.")
                    if name in kinds_by_name and kinds_by_name[name] != kind:
                        raise ValueError(
                            f"Conflicting observable kind {name} in {cls.__qualname__}: "
                            f"{kinds_by_name[name]!r} and {kind!r}."
                        )
                    if kind in names_by_kind and names_by_kind[kind] != name:
                        raise ValueError(
                            f"Duplicate observable kind {kind!r} for {names_by_kind[kind]} "
                            f"and {name} in {cls.__qualname__}."
                        )
                    kinds_by_name[name] = kind
                    names_by_kind[kind] = name

    @dataclass(eq=False)
    class Observables:
        """Arrays populated by a solver in addition to the simulation state.

        Instances are allocated by :meth:`SolverBase.observables` and may be reused
        across steps. Solver implementations can derive from this class to add
        solver-specific arrays while retaining the standard Newton observables.
        Decorate derived containers with ``@dataclass(eq=False)`` and declare
        arrays with :meth:`field`. Identity equality keeps containers usable as
        sources in :class:`~newton.selection.ArticulationView` caches.

        See :ref:`solver_observables` for allocation, selection, and custom field examples.

        .. experimental::

            The solver observable API may change while additional solvers and observable
            categories are migrated to it.
        """

        @staticmethod
        def field(*, dtype: type, frequency: Model.AttributeFrequency | str, kind: str | None = None) -> Any:
            """Declare an optional observable array on a dataclass container.

            See :ref:`solver_observables` for custom observable declarations.

            Args:
                dtype: Warp element type, such as ``wp.float32`` or ``wp.vec3``.
                frequency: Row domain, not update cadence. Custom string
                    frequencies use model count and articulation-ownership metadata.
                kind: Nonempty string identifying this observable. If omitted or
                    ``None``, use the Python field name.

            Returns:
                A dataclass field defaulting to ``None``. No array is allocated
                until :meth:`SolverBase.observables` requests it.

            Raises:
                TypeError: If an explicit kind is not a nonempty string or the frequency
                    is not an attribute frequency or nonempty string.

            .. experimental::
            """
            if kind is not None and (not isinstance(kind, str) or not kind):
                raise TypeError("Solver observable kinds must be nonempty strings.")
            if not isinstance(frequency, (Model.AttributeFrequency, str)) or frequency == "":
                raise TypeError(f"Invalid observable frequency for {kind!r}: {frequency!r}.")
            return dataclass_field(
                default=None,
                init=False,
                repr=False,
                metadata={"solver_observable": _ObservableField(kind, dtype, frequency)},
            )

        # These helpers return dataclass fields, not shared mutable defaults.
        body_qdd: wp.array[wp.spatial_vector] | None = field(  # noqa: RUF009
            dtype=wp.spatial_vector, frequency=Model.AttributeFrequency.BODY
        )
        """Rigid-body accelerations [m/s², rad/s²], shape ``(body_count,)``."""

        body_parent_f: wp.array[wp.spatial_vector] | None = field(  # noqa: RUF009
            dtype=wp.spatial_vector, frequency=Model.AttributeFrequency.BODY
        )
        """Incoming parent-joint wrenches [N, N·m], shape ``(body_count,)``."""

        contact_f: wp.array[wp.spatial_vector] | None = field(  # noqa: RUF009
            dtype=wp.spatial_vector, frequency=Model.AttributeFrequency.CONTACT
        )
        """Contact forces [N, N·m], shape ``(rigid_contact_max + soft_contact_max,)``."""

        _kinds: frozenset[str] = dataclass_field(default_factory=frozenset, init=False, repr=False)
        _solver: SolverBase | None = dataclass_field(default=None, init=False, repr=False)
        _contacts: Contacts | None = dataclass_field(default=None, init=False, repr=False)
        _contact_capacity: tuple[int, int] | None = dataclass_field(default=None, init=False, repr=False)
        _source: SolverBase.Observables | None = dataclass_field(default=None, init=False, repr=False)

        @classmethod
        def _observable_fields(cls) -> Mapping[str, tuple[str, _ObservableField]]:
            """Compile inherited declarations once per concrete container type."""
            cached = cls.__dict__.get("_observable_fields_cache")
            if cached is not None:
                return cached
            if cls.__eq__ is not object.__eq__ or cls.__hash__ is not object.__hash__:
                raise TypeError("Solver observable containers must use @dataclass(eq=False) for identity equality.")
            for base in cls.__mro__:
                if any(
                    isinstance(value, Field) and "solver_observable" in value.metadata
                    for value in base.__dict__.values()
                ):
                    raise TypeError(
                        f"Decorate {base.__name__} with @dataclass(eq=False) to register observable fields."
                    )
            declarations = {}
            for declared in fields(cls):
                spec = declared.metadata.get("solver_observable")
                if spec is None:
                    continue
                if (
                    declared.name.startswith("_")
                    or callable(getattr(SolverBase.Observables, declared.name, None))
                    or isinstance(SolverBase.Observables.__dict__.get(declared.name), property)
                ):
                    raise ValueError(f"Observable field '{declared.name}' conflicts with the container API.")
                kind = declared.name if spec.kind is None else spec.kind
                if kind in declarations:
                    other_name, _ = declarations[kind]
                    raise ValueError(f"Duplicate observable kind {kind!r} for '{other_name}' and '{declared.name}'.")
                declarations[kind] = (declared.name, spec)
            cls._observable_fields_cache = MappingProxyType(declarations)
            return cls._observable_fields_cache

        @property
        def kinds(self) -> frozenset[str]:
            """Observable kinds requested by this container or selected subset."""
            return self._kinds

        def is_requested(self, kind: str) -> bool:
            """Return whether this container requests an observable on the current call.

            This checks the request, not whether its values are fresh. It also
            returns True for requested zero-length arrays and during allocation.

            Args:
                kind: A standard or solver-specific observable name.
            """
            return kind in self._kinds

        def select(self, kinds: Iterable[str]) -> SolverBase.Observables:
            """Return a reusable subset sharing this container's allocated arrays.

            The result has the same concrete type, solver owner, and selected
            array objects, including gradients. Fields omitted from the selection
            are None in the result; this container is not modified. Stepping a
            subset therefore leaves the source's omitted arrays unchanged.

            Create selections before graph capture and reuse them across substeps.
            Only Python containers are created; no array allocation or copying is
            performed. Selections share the source's contact-storage binding.

            Derived containers with nested observable containers should override
            this method, call super(), and select their children in the result.
            Other solver-specific metadata is shallow-copied.

            See :ref:`solver_observables` for selection examples.

            Args:
                kinds: Subset of :attr:`kinds` to request. An empty set requests
                    no observables. Selecting a selection can only narrow it.

            Returns:
                A same-type container referencing the selected arrays.

            Raises:
                TypeError: If kinds is not a collection of nonempty strings.
                ValueError: If this container was not allocated by a solver or a
                    kind is not requested by this container.
            """
            if self._solver is None:
                raise ValueError("Solver observables must be allocated by a solver before selecting fields.")
            requested = _normalize_observable_kinds(kinds)
            missing = requested.difference(self.kinds)
            if missing:
                raise ValueError(f"Cannot select observable kinds not requested by this container: {missing}.")
            selected = copy(self)
            selected._kinds = requested
            selected._source = self._source if self._source is not None else self
            declarations = self._observable_fields()
            for kind in self.kinds.difference(requested):
                name, _ = declarations[kind]
                setattr(selected, name, None)
            if not selected._has_contact_observables():
                selected._contact_capacity = None
            return selected

        def _has_contact_observables(self) -> bool:
            """Validate row frequencies and identify contact-indexed requests."""
            declarations = self._observable_fields()
            frequencies = {declarations[kind][1].frequency for kind in self.kinds}
            return bool(
                frequencies.intersection(
                    (
                        Model.AttributeFrequency.CONTACT,
                        Model.AttributeFrequency.CONTACT_RIGID,
                        Model.AttributeFrequency.CONTACT_SOFT,
                    )
                )
            )

        def get_attribute_frequency(self, name: str) -> Model.AttributeFrequency | str:
            """Return an array's row domain, including inherited declarations.

            Args:
                name: Observable array field name.

            Raises:
                KeyError: If no frequency is declared for the field.
                TypeError: If the container does not use an identity-based dataclass.
            """
            for field_name, spec in self._observable_fields().values():
                if field_name == name:
                    return spec.frequency
            raise KeyError(f"No observable frequency declared for '{name}'.")

        @property
        def model(self) -> Model | None:
            """Model whose indexing this container uses, or ``None`` before allocation."""
            return None if self._solver is None else self._solver.model

        @property
        def contacts(self) -> Contacts | None:
            """Shared contact storage, or None before binding or without contact requests."""
            if self._contact_capacity is None:
                return None
            source = self._source if self._source is not None else self
            return source._contacts

        def bind_contacts(self, contacts: Contacts) -> None:
            """Validate and bind contact storage on its first use by a solver or consumer.

            Newly allocated arrays contain zeros and can be read before the first
            solver step. Selections share the binding with their source; subsequent
            uses must retain the same storage. No device allocation is performed.

            See :ref:`solver_observables` for contact allocation and storage requirements.

            Args:
                contacts: Contact geometry whose rows correspond to these arrays.

            Raises:
                ValueError: If contact observables were not allocated, or the device,
                    capacities, or an existing storage binding do not match.

            .. experimental::
            """
            if self._contact_capacity is None or self.model is None:
                raise ValueError("Allocate contact-indexed SolverBase.Observables before binding Contacts.")
            if contacts.device != self.model.device:
                raise ValueError("Solver observables and Contacts must be on the solver device.")
            if (contacts.rigid_contact_max, contacts.soft_contact_max) != self._contact_capacity:
                raise ValueError(f"Contacts capacities must match solver observables: {self._contact_capacity}.")
            if self.contacts is not None and self.contacts is not contacts:
                raise ValueError("Contact solver observables must use the Contacts instance bound on first use.")
            source = self._source if self._source is not None else self
            source._contacts = contacts

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

    SUPPORTED_OBSERVABLES: ClassVar[frozenset[str]] = frozenset()
    """Names of observables supported by this solver class.

    Entries are string identifiers, such as :attr:`ObservableKind.BODY_QDD`.
    Query :attr:`supported_observables` for the capabilities of an instance.
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
    def supported_observables(self) -> frozenset[str]:
        """Names accepted by :meth:`observables` on this instance.

        Defaults to :attr:`SUPPORTED_OBSERVABLES`. Solvers may narrow the
        class declaration for backend limitations, or derive capabilities from
        child solvers. Applications should query this property.

        See :ref:`solver_observables` for available observables and backend limitations.
        """
        return self.SUPPORTED_OBSERVABLES

    def observables(
        self,
        *,
        kinds: Iterable[str] | None = None,
        requires_grad: bool | None = None,
    ) -> SolverBase.Observables:
        """Allocate reusable arrays for requested solver observables.

        See :ref:`solver_observables` for allocation and usage examples.

        A container is owned by the solver that allocates it and can be passed
        to that solver's :meth:`step` method on every time step. Derived
        solvers add custom kinds to :attr:`SUPPORTED_OBSERVABLES` and
        declare fields on a dataclass derived from :class:`SolverBase.Observables`.
        The default allocator handles both standard and custom fields.
        Override this factory and delegate to ``super()`` when additional
        solver-specific initialization is needed.

        All requested arrays are allocated before this method returns; unrequested
        fields remain ``None``. Contact arrays use the model's resolved rigid
        and soft capacities, not the live contact count. Allocate before graph
        capture and pass matching :class:`~newton.Contacts` to :meth:`step`.

        Args:
            kinds: Set or other iterable of observable names. Entries from
                :class:`ObservableKind` and literal strings are interchangeable.
                If omitted or ``None``, allocate all :attr:`supported_observables`
                for this instance. An empty collection allocates no arrays.
            requires_grad: Whether allocated arrays require gradients. If
                ``None``, use the model's setting.

        Returns:
            A solver-owned observable container with requested arrays allocated.

        Raises:
            TypeError: If a request is not a collection of nonempty strings or the
                configured observable type does not derive from
                :class:`SolverBase.Observables`.
            ValueError: If this solver does not support a requested observable
                or its container does not declare a unique field for the kind.
            RuntimeError: If contact-indexed observables are requested before
                constructing :class:`~newton.CollisionPipeline` for the model.

        .. experimental::

            The solver observable API may change while additional solvers and
            observable categories are migrated to it.
        """
        with self._create_observables(kinds, requires_grad=requires_grad) as observables:
            return observables

    @contextmanager
    def _create_observables(
        self,
        kinds: Iterable[str] | None,
        *,
        requires_grad: bool | None = None,
    ) -> Iterator[SolverBase.Observables]:
        """Share built-in factory allocation, freezing capacities only after setup succeeds."""
        supported = self.supported_observables
        requested = _normalize_observable_kinds(supported if kinds is None else kinds)
        unsupported = requested.difference(supported)
        if unsupported:
            names = ", ".join(repr(kind) for kind in sorted(unsupported))
            raise ValueError(f"{type(self).__name__} does not support solver observable(s): {names}.")

        if not issubclass(self.Observables, SolverBase.Observables):
            raise TypeError("Observables must derive from SolverBase.Observables.")
        declarations = self.Observables._observable_fields()
        missing = requested.difference(declarations)
        if missing:
            names = ", ".join(repr(kind) for kind in sorted(missing))
            raise ValueError(f"No observable field declared for {names}.")
        observables = self.Observables()
        observables._kinds = requested
        observables._solver = self
        if requires_grad is None:
            requires_grad = self.model.requires_grad
        if observables._has_contact_observables():
            observables._contact_capacity = self.model._get_contact_capacity()
        # Declaration order makes allocation deterministic even for unordered requests.
        for kind, (name, spec) in declarations.items():
            if kind not in requested:
                continue
            setattr(
                observables,
                name,
                wp.zeros(
                    self.model._attribute_frequency_count(spec.frequency),
                    dtype=spec.dtype,
                    device=self.model.device,
                    requires_grad=requires_grad,
                ),
            )
        yield observables
        if observables._contact_capacity is not None:
            self.model._solver_observable_contact_capacity = observables._contact_capacity

    def validate_observables(
        self, observables: SolverBase.Observables | None, contacts: Contacts | None = None
    ) -> None:
        """Validate ownership and contact storage before a custom solver step.

        Call this at the start of :meth:`step`, before launching work or writing
        observable arrays. Passing ``None`` is a no-op.

        See :ref:`solver_observables` for custom solver integration.

        Args:
            observables: Optional container allocated by this solver.
            contacts: Required for contact-indexed observables.

        Raises:
            TypeError: If the container has an incompatible type.
            ValueError: If the owner or contact storage is incompatible.

        .. experimental::
        """
        if observables is None:
            return
        if not isinstance(observables, self.Observables):
            raise TypeError(f"'observables' must be an instance of {self.Observables.__name__}.")
        if observables._solver is not self:
            raise ValueError("Solver observables must be passed to the solver instance that allocated them.")
        if observables._contact_capacity is not None:
            if contacts is None:
                raise ValueError("Pass Contacts to solver.step() when using contact-indexed solver observables.")
            observables.bind_contacts(contacts)

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
        observables: SolverBase.Observables | None = None,
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
        Solver implementations must handle each relevant narrow DOF flag:
        checking ``JOINT_DOF_PROPERTIES`` alone does not detect them. The broad
        flag covers force, inertial, and reference properties; friction/damping
        belongs to the force category. Solvers may ignore categories they do
        not cache or support, but should preserve unrelated pending edits
        except for documented backend dependencies. In MuJoCo, notifications
        that recompute constants also republish the current tendon limit
        modes, gains, raw solref values, and ranges, even without
        ``TENDON_PROPERTIES``. Force-only joint notifications leave these
        tendon parameters unchanged.
        Valid flags are:

        * ``ModelFlags.JOINT_PROPERTIES``: Joint transforms, axes, or coordinates
          have changed.
        * ``ModelFlags.JOINT_DOF_PROPERTIES``: Full joint DOF update, including
          force, armature, and reference-pose properties (legacy behavior).
        * ``ModelFlags.JOINT_DOF_FORCE_PROPERTIES``: Joint friction, damping,
          target gains/modes, effort limits, passive stiffness, or limit
          coefficients/bounds have changed.
        * ``ModelFlags.JOINT_DOF_INERTIAL_PROPERTIES``: Joint armature has changed.
        * ``ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES``: Joint reference poses
          or spring references have changed.
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

        .. deprecated:: 1.7

            Request :attr:`ObservableKind.CONTACT_F` using :meth:`observables`
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
