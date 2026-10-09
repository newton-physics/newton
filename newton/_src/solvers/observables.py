# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared solver observable flags, field declarations, and containers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import copy
from dataclasses import Field, dataclass, fields
from dataclasses import field as dataclass_field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import warp as wp

from ..sim import Contacts, Model

if TYPE_CHECKING:
    from .solver import SolverBase


class SolverObservableFlags(Enum):
    """Standard observable quantities that may be requested from a solver.

    Requests are composed as a :class:`set` rather than a bit mask so that
    solver-specific enums can add entries without coordinating integer bits
    with Newton or other solver implementations.

    Containers associate flags with arrays using :meth:`SolverObservables.field`.
    Flag values are identifiers, not array names or allocation instructions.

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


@dataclass(frozen=True)
class _ObservableField:
    """Immutable allocation metadata stored on a dataclass field."""

    flag: Enum
    dtype: type
    frequency: Model.AttributeFrequency | str


@dataclass(eq=False)
class SolverObservables:
    """Arrays populated by a solver in addition to the simulation state.

    Instances are allocated by :meth:`SolverBase.observables` and may be reused
    across steps. Solver implementations can derive from this class to add
    solver-specific arrays while retaining the standard Newton observables.
    Decorate derived containers with ``@dataclass(eq=False)`` and declare
    arrays with :meth:`field`. Identity equality keeps containers usable as
    sources in :class:`~newton.selection.ArticulationView` caches.

    .. experimental::

        The solver observable API may change while additional solvers and observable
        categories are migrated to it.
    """

    @staticmethod
    def field(*, flag: Enum, dtype: type, frequency: Model.AttributeFrequency | str) -> Any:
        """Declare an optional observable array on a dataclass container.

        Args:
            flag: Plain enum member identifying this observable. Its value
                need not match the Python field name.
            dtype: Warp element type, such as ``wp.float32`` or ``wp.vec3``.
            frequency: Row domain, not update cadence. Custom string
                frequencies use model count and articulation-ownership metadata.

        Returns:
            A dataclass field defaulting to ``None``. No array is allocated
            until :meth:`SolverBase.observables` requests it.

        Raises:
            TypeError: If the flag is not a plain enum member or the frequency
                is not an attribute frequency or nonempty string.

        .. experimental::
        """
        if not isinstance(flag, Enum) or isinstance(flag, (int, str)):
            raise TypeError("Solver observable flags must be plain enum.Enum members.")
        if not isinstance(frequency, (Model.AttributeFrequency, str)) or frequency == "":
            raise TypeError(f"Invalid observable frequency for {flag!r}: {frequency!r}.")
        return dataclass_field(
            default=None,
            init=False,
            repr=False,
            metadata={"solver_observable": _ObservableField(flag, dtype, frequency)},
        )

    # These helpers return dataclass fields, not shared mutable defaults.
    body_qdd: wp.array[wp.spatial_vector] | None = field(  # noqa: RUF009
        flag=SolverObservableFlags.BODY_QDD, dtype=wp.spatial_vector, frequency=Model.AttributeFrequency.BODY
    )
    """Rigid-body accelerations [m/s², rad/s²], shape ``(body_count,)``."""

    body_parent_f: wp.array[wp.spatial_vector] | None = field(  # noqa: RUF009
        flag=SolverObservableFlags.BODY_PARENT_F, dtype=wp.spatial_vector, frequency=Model.AttributeFrequency.BODY
    )
    """Incoming parent-joint wrenches [N, N·m], shape ``(body_count,)``."""

    contact_f: wp.array[wp.spatial_vector] | None = field(  # noqa: RUF009
        flag=SolverObservableFlags.CONTACT_F, dtype=wp.spatial_vector, frequency=Model.AttributeFrequency.CONTACT
    )
    """Contact forces [N, N·m], shape ``(rigid_contact_max + soft_contact_max,)``."""

    _flags: frozenset[Enum] = dataclass_field(default_factory=frozenset, init=False, repr=False)
    _solver: SolverBase | None = dataclass_field(default=None, init=False, repr=False)
    _contacts: Contacts | None = dataclass_field(default=None, init=False, repr=False)
    _contact_capacity: tuple[int, int] | None = dataclass_field(default=None, init=False, repr=False)
    _source: SolverObservables | None = dataclass_field(default=None, init=False, repr=False)

    @classmethod
    def _observable_fields(cls) -> Mapping[Enum, tuple[str, _ObservableField]]:
        """Compile inherited declarations once per concrete container type."""
        cached = cls.__dict__.get("_observable_fields_cache")
        if cached is not None:
            return cached
        if cls.__eq__ is not object.__eq__ or cls.__hash__ is not object.__hash__:
            raise TypeError("Solver observable containers must use @dataclass(eq=False) for identity equality.")
        for base in cls.__mro__:
            if any(
                isinstance(value, Field) and "solver_observable" in value.metadata for value in base.__dict__.values()
            ):
                raise TypeError(f"Decorate {base.__name__} with @dataclass(eq=False) to register observable fields.")
        declarations = {}
        for declared in fields(cls):
            spec = declared.metadata.get("solver_observable")
            if spec is None:
                continue
            if (
                declared.name.startswith("_")
                or callable(getattr(SolverObservables, declared.name, None))
                or isinstance(SolverObservables.__dict__.get(declared.name), property)
            ):
                raise ValueError(f"Observable field '{declared.name}' conflicts with the container API.")
            if spec.flag in declarations:
                other_name, _ = declarations[spec.flag]
                raise ValueError(f"Duplicate observable flag {spec.flag!r} for '{other_name}' and '{declared.name}'.")
            declarations[spec.flag] = (declared.name, spec)
        cls._observable_fields_cache = MappingProxyType(declarations)
        return cls._observable_fields_cache

    @property
    def flags(self) -> frozenset[Enum]:
        """Observable flags requested by this container or selected subset."""
        return self._flags

    def is_requested(self, flag: Enum) -> bool:
        """Return whether this container requests an observable on the current call.

        This checks the request, not whether its values are fresh. It also
        returns True for requested zero-length arrays and during allocation.

        Args:
            flag: A standard or solver-specific observable enum member.
        """
        return flag in self._flags

    def select(self, flags: Iterable[Enum]) -> SolverObservables:
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

        Args:
            flags: Subset of :attr:`flags` to request. An empty set requests
                no observables. Selecting a selection can only narrow it.

        Returns:
            A same-type container referencing the selected arrays.

        Raises:
            ValueError: If this container was not allocated by a solver or a
                flag is not requested by this container.
        """
        if self._solver is None:
            raise ValueError("Solver observables must be allocated by a solver before selecting fields.")
        requested = frozenset(flags)
        missing = requested.difference(self.flags)
        if missing:
            raise ValueError(f"Cannot select observable flags not requested by this container: {missing}.")
        selected = copy(self)
        selected._flags = requested
        selected._source = self._source if self._source is not None else self
        declarations = self._observable_fields()
        for flag in self.flags.difference(requested):
            name, _ = declarations[flag]
            setattr(selected, name, None)
        if not selected._has_contact_observables():
            selected._contact_capacity = None
        return selected

    def _has_contact_observables(self) -> bool:
        """Validate row frequencies and identify contact-indexed requests."""
        declarations = self._observable_fields()
        frequencies = {declarations[flag][1].frequency for flag in self.flags}
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

        Args:
            contacts: Contact geometry whose rows correspond to these arrays.

        Raises:
            ValueError: If contact observables were not allocated, or the device,
                capacities, or an existing storage binding do not match.

        .. experimental::
        """
        if self._contact_capacity is None or self.model is None:
            raise ValueError("Allocate contact-indexed SolverObservables before binding Contacts.")
        if contacts.device != self.model.device:
            raise ValueError("Solver observables and Contacts must be on the solver device.")
        if (contacts.rigid_contact_max, contacts.soft_contact_max) != self._contact_capacity:
            raise ValueError(f"Contacts capacities must match solver observables: {self._contact_capacity}.")
        if self.contacts is not None and self.contacts is not contacts:
            raise ValueError("Contact solver observables must use the Contacts instance bound on first use.")
        source = self._source if self._source is not None else self
        source._contacts = contacts
