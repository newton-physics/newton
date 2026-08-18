# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
USD schema resolver infrastructure.

This module defines the base resolver types used to map authored USD schema
attributes onto Newton builder attributes. Public users should import resolver
types from :mod:`newton.usd`.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from enum import IntEnum
from numbers import Real
from typing import TYPE_CHECKING, Any, ClassVar, Literal, ParamSpec, TypeVar, cast, get_type_hints

from . import utils as usd

if TYPE_CHECKING:
    from pxr import Usd

    from ..sim.builder import ModelBuilder


_MISSING_FALLBACK = object()
_NO_COMPARISON = object()
_NO_OVERRIDE = object()
_SAME_AS_DEFAULT = object()
_UNREGISTERED_SCHEMA = object()
_UNREADABLE_FALLBACK = object()

_P = ParamSpec("_P")
_R = TypeVar("_R")
_PolicySelection = Literal["active", "legacy", "composed"]


@dataclass(frozen=True)
class _ImporterDefault:
    """Carry an importer default that may be None."""

    value: Any

    def __repr__(self) -> str:
        return f"<omitted; importer default={self.value!r}>"


def _default_when_omitted(value: Any) -> Any:
    """Mark a public importer argument's value as its omission default."""
    return _ImporterDefault(value)


def _track_omitted_import_defaults(**defaults: Any) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Preserve public defaults while marking omitted keyword-only arguments."""

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(function)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            for name, default in defaults.items():
                if name not in kwargs:
                    kwargs[name] = _ImporterDefault(default)
            return function(*args, **kwargs)

        # The wrapper lives in this module, so resolve forward references while
        # the original function's globals are still available.
        wrapped.__annotations__ = get_type_hints(function, include_extras=True)
        return cast(Callable[_P, _R], wrapped)

    return decorate


def _interpret_import_argument(value: Any) -> tuple[Any, _ImporterDefault]:
    """Preserve an argument's future override and legacy default roles."""
    if isinstance(value, _ImporterDefault):
        return _NO_OVERRIDE, value
    return value, _ImporterDefault(value)


def _resolve_import_option(value: Any, authored_value: Any, *, use_explicit_overrides: bool) -> Any:
    """Resolve importer metadata while preserving legacy argument precedence."""
    override, default = _interpret_import_argument(value)
    if use_explicit_overrides and override is not _NO_OVERRIDE:
        return override
    if authored_value is not None:
        return authored_value
    return default.value


def _importer_default(default: Any) -> tuple[bool, Any]:
    """Return whether an importer default exists and its unwrapped value."""
    if isinstance(default, _ImporterDefault):
        return True, default.value
    return default is not None, default


@dataclass(frozen=True)
class _ResolverValue:
    value: Any
    authored: bool

    @property
    def usable(self) -> bool:
        return self.value is not None


class _SchemaFallbackError(Exception):
    """Base class for expected composed-fallback audit failures."""

    def __init__(self, message: str, label: str):
        super().__init__(message)
        self.label = label


class _PXRValueGetterError(_SchemaFallbackError, TypeError):
    """A PXR-only resolver getter cannot consume source-neutral values."""


class PrimType(IntEnum):
    """Enumeration of USD prim types that can be resolved by schema resolvers."""

    SCENE = 0
    """PhysicsScene prim type."""
    JOINT = 1
    """Joint prim type."""
    SHAPE = 2
    """Shape prim type."""
    BODY = 3
    """Body prim type."""
    MATERIAL = 4
    """Material prim type."""
    ACTUATOR = 5
    """Actuator prim type."""
    ARTICULATION = 6
    """Articulation root prim type."""


class _ValueSource(IntEnum):
    """Internal source categories for source-neutral resolution results."""

    UNRESOLVED = 0
    AUTHORED = 1
    REGISTERED_FALLBACK = 2
    IMPORTER_DEFAULT = 3
    COMPATIBILITY_DEFAULT = 4


_VALUE_SOURCE_LABELS = {
    _ValueSource.UNRESOLVED: "unresolved",
    _ValueSource.AUTHORED: "authored value",
    _ValueSource.REGISTERED_FALLBACK: "registered fallback",
    _ValueSource.IMPORTER_DEFAULT: "importer default",
    _ValueSource.COMPATIBILITY_DEFAULT: "compatibility default",
}


class SchemaResolver:
    """Base class mapping USD schema attributes to Newton attributes."""

    @dataclass
    class SchemaAttribute:
        """
        Specifies a USD attribute and its transformation function.

        Args:
            name: The name of the USD attribute (or primary attribute when using a getter).
            default: Legacy compatibility fallback used after importer defaults.
            usd_value_transformer: Optional function to transform the raw value into the format expected by Newton.
            usd_value_getter: Optional function (prim) -> value used instead of reading a single attribute (e.g. to compute gap from contactOffset - restOffset).
            attribute_names: When set, names used for collect_prim_attrs; otherwise [name] is used.
            fallback_is_unset: Optional predicate that returns ``True`` when a registered schema fallback expresses
                no opinion and resolution should continue.
            angular_unit: Unit used when an angular DOF consumes this value.
        """

        name: str
        default: Any | None = None
        usd_value_transformer: Callable[[Any], Any] | None = None
        usd_value_getter: Callable[[Usd.Prim], Any] | None = None
        attribute_names: Sequence[str] = ()
        fallback_is_unset: Callable[[Any], bool] | None = None
        angular_unit: Literal["degrees", "radians"] = "degrees"
        _reader_value_getter: Callable[[Callable[[str], Any | None]], Any] | None = field(
            default=None,
            init=False,
            repr=False,
            compare=False,
        )

    # mapping is a dictionary for known variables in Newton. Its purpose is to map USD attributes to existing Newton data.
    # PrimType -> Newton variable -> Attribute
    mapping: ClassVar[dict[PrimType, dict[str, SchemaAttribute]]]

    # Name of the schema resolver
    name: ClassVar[str]

    # Applied or typed schema that owns each mapping entry.
    _schema_names: ClassVar[dict[PrimType, str | dict[str, str]]] = {}
    _use_legacy_unowned_defaults: ClassVar[bool] = True

    # extra_attr_namespaces is a list of additional USD attribute namespaces in which the schema attributes may be authored.
    extra_attr_namespaces: ClassVar[list[str]] = []

    # deformable_attr_namespaces lists vendor namespaces that carry the deformable
    # material/geometry attributes (parsed as a fallback to the canonical physics:
    # schema). Kept separate from extra_attr_namespaces so generic rigid-body
    # namespaces are never read as deformable attributes.
    deformable_attr_namespaces: ClassVar[list[str]] = []

    def __init__(self) -> None:
        # Precompute the full set of USD attribute names referenced by this resolver's mapping.
        names: set[str] = set()
        try:
            mapping_items = self.mapping.items()
        except AttributeError:
            mapping_items = []
        for _prim_type, var_map in mapping_items:
            try:
                var_items = var_map.items()
            except AttributeError:
                continue
            for _var, spec in var_items:
                if spec.attribute_names:
                    names.update(spec.attribute_names)
                else:
                    names.add(spec.name)
        self._solver_attributes: list[str] = list(names)

    def get_value(self, prim: Usd.Prim, prim_type: PrimType, key: str) -> Any | None:
        """Get an authored value for a resolver key.

        Args:
            prim: USD prim to query.
            prim_type: Prim type category.
            key: Logical Newton attribute key within the prim category.

        Returns:
            Resolved authored value, or ``None`` when not found.
        """
        if prim is None:
            return None
        return self._get_value_with_reader(
            lambda name: usd.get_attribute(prim, name),
            prim_type,
            key,
            legacy_prim=prim,
        )

    def _get_value_state(self, prim: Usd.Prim, prim_type: PrimType, key: str) -> _ResolverValue:
        if type(self).get_value is not SchemaResolver.get_value:
            value = self.get_value(prim, prim_type, key)
            return _ResolverValue(value, value is not None)
        if prim is None:
            return _ResolverValue(None, False)

        spec = self.mapping.get(prim_type, {}).get(key)
        if spec is None:
            value = self.get_value(prim, prim_type, key)
            return _ResolverValue(value, value is not None)

        names = spec.attribute_names or (spec.name,)
        states: dict[str, bool] = {}

        def read_attribute(name: str) -> Any | None:
            attribute = prim.GetAttribute(name)
            authored = bool(attribute and attribute.HasAuthoredValue())
            states[name] = authored
            return attribute.Get() if authored else None

        value = self._get_value_with_reader(read_attribute, prim_type, key, legacy_prim=prim)
        if value is not None:
            return _ResolverValue(value, True)

        for name in names:
            if name not in states:
                read_attribute(name)

        if spec._reader_value_getter is not None:
            authored = all(states.values())
        else:
            authored = states.get(spec.name, False)
        return _ResolverValue(None, authored)

    def _schema_name(self, prim_type: PrimType, key: str) -> str | None:
        schema_names = self._schema_names.get(prim_type)
        if isinstance(schema_names, str):
            return schema_names if key in self.mapping.get(prim_type, {}) else None
        return schema_names.get(key) if schema_names is not None else None

    def _schema_is_applied(self, prim: Usd.Prim, prim_type: PrimType, key: str) -> bool:
        schema_name = self._schema_name(prim_type, key)
        if schema_name is None or prim is None:
            return False
        return str(prim.GetTypeName()) == schema_name or usd.has_applied_api_schema(prim, schema_name)

    def _get_value_with_reader(
        self,
        read_attribute: Callable[[str], Any | None],
        prim_type: PrimType,
        key: str,
        *,
        legacy_prim: Usd.Prim | None = None,
    ) -> Any | None:
        spec = self.mapping.get(prim_type, {}).get(key)
        if spec is not None:
            if spec._reader_value_getter is not None:
                v = spec._reader_value_getter(read_attribute)
            elif spec.usd_value_getter is not None:
                if legacy_prim is None:
                    schema_name = self._schema_name(prim_type, key)
                    names = ", ".join(spec.attribute_names or (spec.name,))
                    raise _PXRValueGetterError(
                        f"Schema resolver '{self.name}' key '{prim_type.name.lower()}:{key}' uses a "
                        "PXR-only usd_value_getter and cannot resolve schema fallbacks.",
                        f"{schema_name} ({names})",
                    )
                v = spec.usd_value_getter(legacy_prim)
            else:
                v = read_attribute(spec.name)
            if v is not None:
                return spec.usd_value_transformer(v) if spec.usd_value_transformer is not None else v
        return None

    def _get_fallback_with_reader(
        self,
        read_attribute: Callable[[str], Any],
        prim_type: PrimType,
        key: str,
    ) -> Any:
        spec = self.mapping.get(prim_type, {}).get(key)
        if spec is None:
            return _MISSING_FALLBACK
        names = spec.attribute_names or (spec.name,)
        values: dict[str, Any] = {}
        for name in names:
            value = read_attribute(name)
            if value is _MISSING_FALLBACK:
                return _MISSING_FALLBACK
            values[name] = value
        return self._get_value_with_reader(values.get, prim_type, key)

    def collect_prim_attrs(self, prim: Usd.Prim) -> dict[str, Any]:
        """Collect all resolver-relevant attributes for a prim.

        Args:
            prim: USD prim to inspect.

        Returns:
            Dictionary mapping authored USD attribute names to values.
        """
        if prim is None:
            return {}

        # Collect attributes by known prefixes
        # USD expects namespace tokens without ':' (e.g., 'newton', 'mjc', 'physxArticulation')
        main_prefix = self.name
        all_prefixes = [main_prefix]
        if self.extra_attr_namespaces:
            all_prefixes.extend(self.extra_attr_namespaces)
        prefixed_attrs: dict[str, Any] = _collect_attrs_by_namespace(prim, all_prefixes)

        # Collect explicit attribute names defined in the resolver mapping (precomputed)
        prim_solver_attrs = _collect_attrs_by_name(prim, self._solver_attributes) if self._solver_attributes else {}

        # Merge and return (explicit names take precedence)
        merged: dict[str, Any] = {}
        merged.update(prefixed_attrs)
        merged.update(prim_solver_attrs)
        return merged

    def validate_custom_attributes(self, builder: ModelBuilder) -> None:
        """
        Validate that solver-specific custom attributes are registered on the builder.

        Override in subclasses to check that required custom attributes have been
        registered before parsing. Called by parse_usd() before processing entities.

        Args:
            builder: The ModelBuilder to validate custom attributes on.
        """
        del builder


# Backward-compatible alias; prefer SchemaResolver.SchemaAttribute.
SchemaAttribute = SchemaResolver.SchemaAttribute


def _reader_schema_attribute(
    *args: Any,
    _reader_value_getter: Callable[[Callable[[str], Any | None]], Any],
    **kwargs: Any,
) -> SchemaAttribute:
    def _legacy_value_getter(prim: Usd.Prim) -> Any:
        return _reader_value_getter(lambda name: usd.get_attribute(prim, name))

    attribute = SchemaAttribute(*args, usd_value_getter=_legacy_value_getter, **kwargs)
    attribute._reader_value_getter = _reader_value_getter
    return attribute


def _collect_attrs_by_name(prim: Usd.Prim, names: Sequence[str]) -> dict[str, Any]:
    """Collect attributes authored on the prim that have direct mappings in the resolver mapping"""
    out: dict[str, Any] = {}
    for n in names:
        v = usd.get_attribute(prim, n)
        if v is not None:
            out[n] = v
    return out


def _collect_attrs_by_namespace(prim: Usd.Prim, namespaces: Sequence[str]) -> dict[str, Any]:
    """Collect authored attributes using USD namespace queries."""
    out: dict[str, Any] = {}
    if prim is None:
        return out
    for ns in namespaces:
        out.update(usd.get_attributes_in_namespace(prim, ns))
    return out


def _registered_attribute_fallbacks(prim_definition: Any) -> dict[str, Any]:
    if prim_definition is None:
        return {}
    fallbacks = {}
    for name in prim_definition.GetPropertyNames():
        value = prim_definition.GetAttributeFallbackValue(name)
        if value is not None:
            fallbacks[name] = value
    return fallbacks


class _SchemaResolutionPolicy:
    """Resolve one ordered resolver set through source-neutral callbacks."""

    def __init__(self, resolvers: Sequence[SchemaResolver]):
        self._resolvers = tuple(resolvers)

    def _resolve_value(
        self,
        read_value: Callable[[SchemaResolver, str], Any],
        schema_is_applied: Callable[[SchemaResolver, str], bool],
        read_fallback: Callable[[SchemaResolver, str], Any],
        prim_type: PrimType,
        key: str,
        *,
        default: Any = None,
        authored_aliases: Sequence[str] = (),
    ) -> _ResolvedValue:
        compatibility_fallbacks: set[int] = set()
        for resolver in self._resolvers:
            spec = resolver.mapping.get(prim_type, {}).get(key)
            for authored_key in (key, *authored_aliases):
                if authored_key not in resolver.mapping.get(prim_type, {}):
                    continue
                value = read_value(resolver, authored_key)
                if not isinstance(value, _ResolverValue):
                    value = _ResolverValue(value, value is not None)
                if value.authored and value.usable:
                    return _ResolvedValue(
                        value.value,
                        resolver,
                        _ValueSource.AUTHORED,
                        mapping_key=authored_key,
                    )

            if spec is None:
                continue

            if schema_is_applied(resolver, key):
                fallback = read_fallback(resolver, key)
                if fallback is _UNREGISTERED_SCHEMA:
                    compatibility_fallbacks.add(id(resolver))
                elif fallback is _UNREADABLE_FALLBACK:
                    continue
                elif fallback is not _MISSING_FALLBACK and fallback is not None:
                    if spec.fallback_is_unset is None or not spec.fallback_is_unset(fallback):
                        return _ResolvedValue(
                            fallback,
                            resolver,
                            _ValueSource.REGISTERED_FALLBACK,
                            mapping_key=key,
                        )

        has_importer_default, importer_default = _importer_default(default)
        if has_importer_default:
            return _ResolvedValue(importer_default, None, _ValueSource.IMPORTER_DEFAULT)

        for resolver in self._resolvers:
            spec = resolver.mapping.get(prim_type, {}).get(key)
            if (
                spec is None
                or not resolver._use_legacy_unowned_defaults
                or (resolver._schema_name(prim_type, key) is not None and id(resolver) not in compatibility_fallbacks)
                or spec.default is None
            ):
                continue
            value = spec.default
            if spec.usd_value_transformer is not None:
                value = spec.usd_value_transformer(value)
            if value is None:
                continue
            return _ResolvedValue(
                value,
                None,
                _ValueSource.COMPATIBILITY_DEFAULT,
                compatibility_resolver=resolver,
                mapping_key=key,
            )

        return _ResolvedValue(None, None, _ValueSource.UNRESOLVED)


@dataclass(frozen=True)
class _ResolvedValue:
    """Carry a resolved value and its consumer-visible source."""

    value: Any
    resolver: SchemaResolver | None
    source: _ValueSource
    comparison: Any = field(default=_NO_COMPARISON, repr=False, compare=False)
    compatibility_resolver: SchemaResolver | None = field(default=None, repr=False, compare=False)
    mapping_key: str | None = field(default=None, repr=False, compare=False)

    @property
    def authored(self) -> bool:
        return self.source == _ValueSource.AUTHORED


class SchemaResolverManager:
    """
    Manager for resolving multiple USD schemas in a priority order.
    """

    class _AuditProvenance(IntEnum):
        """Describe provenance exposed by a resolver operation."""

        NONE = 0
        RESOLVER = 1
        SOURCE = 2

    @dataclass(frozen=True)
    class _PolicyValues:
        """Hold active and comparison results for one property."""

        active: _ResolvedValue
        legacy: _ResolvedValue | None
        composed: _ResolvedValue | None

    @dataclass(frozen=True)
    class _PolicyChangeCandidate:
        """Describe one input to an assembled property comparison."""

        key: str
        policies: SchemaResolverManager._PolicyValues
        legacy_comparison: Any = field(default=_NO_COMPARISON, repr=False)
        composed_comparison: Any = field(default=_NO_COMPARISON, repr=False)
        compare_source: bool = False

    @dataclass(frozen=True)
    class _MigrationEndpoint:
        """Describe one side of a migration transition."""

        source: _ValueSource
        owner: str | None = None
        attribute_names: tuple[str, ...] = ()

        def format(self) -> str:
            source = _VALUE_SOURCE_LABELS[self.source]
            if self.owner is None:
                return source
            names = ", ".join(self.attribute_names)
            return f"{self.owner} ({names}; {source})"

    @dataclass(frozen=True)
    class _MigrationTransition:
        """Describe a property transition between resolution policies."""

        key: str
        attribute_names: tuple[str, ...]
        legacy: SchemaResolverManager._MigrationEndpoint
        composed: SchemaResolverManager._MigrationEndpoint

        def format(self) -> str:
            if (
                self.legacy.owner is not None
                and self.legacy.owner == self.composed.owner
                and self.legacy.attribute_names == self.composed.attribute_names
            ):
                names = ", ".join(self.legacy.attribute_names)
                transition = (
                    f"{_VALUE_SOURCE_LABELS[self.legacy.source]} -> {_VALUE_SOURCE_LABELS[self.composed.source]}"
                )
                return f"{self.legacy.owner} ({names}; {transition})"

            if self.legacy.owner is None and self.composed.owner is None:
                names = ", ".join(self.attribute_names)
                property_label = f"{self.key} ({names})" if names else self.key
                transition = (
                    f"{_VALUE_SOURCE_LABELS[self.legacy.source]} -> {_VALUE_SOURCE_LABELS[self.composed.source]}"
                )
                return f"{property_label}: {transition}"

            return f"{self.key}: {self.legacy.format()} -> {self.composed.format()}"

    @dataclass(frozen=True)
    class _InterpretedPolicyValue:
        """Keep one raw resolver result and its interpreted property value."""

        resolved: _ResolvedValue
        value: Any

        @property
        def raw_value(self) -> Any:
            return self.resolved.value

        @property
        def resolver(self) -> SchemaResolver | None:
            return self.resolved.resolver

        @property
        def source(self) -> _ValueSource:
            return self.resolved.source

    @dataclass(frozen=True)
    class _InterpretedPolicyValues:
        """Keep interpreted active and migration results for one property."""

        key: str
        raw: SchemaResolverManager._PolicyValues
        active: SchemaResolverManager._InterpretedPolicyValue
        legacy: SchemaResolverManager._InterpretedPolicyValue | None
        composed: SchemaResolverManager._InterpretedPolicyValue | None

        def select(self, policy: _PolicySelection) -> SchemaResolverManager._InterpretedPolicyValue | None:
            if policy == "active":
                return self.active
            if policy == "legacy":
                return self.legacy
            return self.composed

        def contribution(
            self,
            *,
            key: str | None = None,
            legacy_comparison: Any = _NO_COMPARISON,
            composed_comparison: Any = _NO_COMPARISON,
            compare_source: bool = False,
        ) -> SchemaResolverManager._PolicyChangeCandidate:
            """Describe this property's contribution to an assembled change."""
            if legacy_comparison is _NO_COMPARISON and self.legacy is not None:
                legacy_comparison = self.legacy.value
            if composed_comparison is _NO_COMPARISON and self.composed is not None:
                composed_comparison = self.composed.value
            return SchemaResolverManager._PolicyChangeCandidate(
                self.key if key is None else key,
                self.raw,
                legacy_comparison,
                composed_comparison,
                compare_source,
            )

    def __init__(
        self,
        resolvers: Sequence[SchemaResolver],
        *,
        use_applied_schema_fallbacks: bool = False,
    ):
        """
        Initialize resolver manager with resolver instances in priority order.

        Args:
            resolvers: List of instantiated resolvers in priority order.
            use_applied_schema_fallbacks: Use the owning applied schema's fallback
                before importer defaults. Only registered schema definitions supply
                these fallbacks; unregistered resolver defaults remain after importer
                defaults. Defaults to False.
        """
        self.resolvers = list(resolvers)
        self._use_applied_schema_fallbacks = use_applied_schema_fallbacks
        self._resolution = _SchemaResolutionPolicy(self.resolvers)
        self._registered_schema_fallbacks: dict[tuple[str, str], dict[str, Any] | None] = {}
        self._legacy_fallback_properties: dict[SchemaResolverManager._MigrationTransition, set[str]] = {}
        self._legacy_fallback_failures: dict[str, set[str]] = {}

        # Dictionary to accumulate schema attributes as prims are encountered
        # Pre-initialize maps for each configured resolver
        self._schema_attrs: dict[str, dict[str, dict[str, Any]]] = {r.name: {} for r in self.resolvers}

    def _collect_on_first_use(self, resolver: SchemaResolver, prim: Usd.Prim) -> None:
        """Collect and store attributes for this resolver/prim on first use."""
        if prim is None:
            return
        prim_path = str(prim.GetPath())
        if prim_path in self._schema_attrs[resolver.name]:
            return
        self._schema_attrs[resolver.name][prim_path] = resolver.collect_prim_attrs(prim)

    def get_value(
        self,
        prim: Usd.Prim,
        prim_type: PrimType,
        key: str,
        default: Any = None,
        verbose: bool = False,
        *,
        override: Any = _NO_OVERRIDE,
        legacy_default: Any = _SAME_AS_DEFAULT,
        comparison_key: Callable[[Any, SchemaResolver | None], Any] | None = None,
    ) -> Any:
        """
        Resolve a value using the configured resolver policy.

        Args:
            prim: USD prim to query (for scene prim_type, this should be scene_prim)
            prim_type: Prim type (PrimType enum)
            key: Attribute key within the prim type
            default: Default value if not found
            override: Explicit importer value that takes precedence over USD
                resolution when composed fallback resolution is enabled. Under
                legacy resolution, its paired importer default retains the
                existing precedence and suppresses migration auditing.
            legacy_default: Default used only under legacy resolution. Omit to
                use ``default`` for both policies.
            comparison_key: Convert a raw value and resolver into its interpreted
                property form for the compatibility audit.

        Returns:
            Resolved value according to the precedence above.
        """
        if override is not _NO_OVERRIDE and self._uses_composed_fallbacks:
            return override

        resolved = self._get_value_with_policy(
            prim,
            prim_type,
            key,
            default,
            legacy_default=legacy_default,
            audit_provenance=self._AuditProvenance.NONE,
            audit_fallbacks=override is _NO_OVERRIDE,
            comparison_key=comparison_key,
        )
        active_default = self._active_default(default, legacy_default)
        self._report_missing(
            prim,
            prim_type,
            key,
            resolved.value,
            active_default,
            verbose and override is _NO_OVERRIDE,
        )
        return resolved.value

    def get_value_with_resolver(
        self,
        prim: Usd.Prim,
        prim_type: PrimType,
        key: str,
        default: Any = None,
        verbose: bool = False,
        *,
        override: Any = _NO_OVERRIDE,
        legacy_default: Any = _SAME_AS_DEFAULT,
        comparison_key: Callable[[Any, SchemaResolver | None], Any] | None = None,
    ) -> tuple[Any, SchemaResolver | None]:
        """Resolve a value and return the resolver that supplied it."""
        if override is not _NO_OVERRIDE and self._uses_composed_fallbacks:
            return override, None

        resolved = self._get_value_with_policy(
            prim,
            prim_type,
            key,
            default,
            legacy_default=legacy_default,
            audit_provenance=self._AuditProvenance.RESOLVER,
            audit_fallbacks=override is _NO_OVERRIDE,
            comparison_key=comparison_key,
        )
        active_default = self._active_default(default, legacy_default)
        self._report_missing(
            prim,
            prim_type,
            key,
            resolved.value,
            active_default,
            verbose and override is _NO_OVERRIDE,
        )
        return resolved.value, resolved.resolver

    def _get_interpreted_value(
        self,
        prim: Usd.Prim,
        prim_type: PrimType,
        key: str,
        *,
        interpreter: Callable[[_ResolvedValue], Any],
        default: Any = None,
        verbose: bool = False,
        override: Any = _NO_OVERRIDE,
        legacy_default: Any = _SAME_AS_DEFAULT,
    ) -> SchemaResolverManager._InterpretedPolicyValue:
        """Resolve raw and interpreted forms of one property value."""
        if override is not _NO_OVERRIDE and self._uses_composed_fallbacks:
            resolved = _ResolvedValue(override, None, _ValueSource.IMPORTER_DEFAULT)
            return self._InterpretedPolicyValue(resolved, interpreter(resolved))

        resolved = self._get_value_with_policy(
            prim,
            prim_type,
            key,
            default,
            legacy_default=legacy_default,
            audit_provenance=self._AuditProvenance.NONE,
            audit_fallbacks=override is _NO_OVERRIDE,
            comparison_key=None,
            result_interpreter=interpreter,
        )
        active_default = self._active_default(default, legacy_default)
        self._report_missing(
            prim,
            prim_type,
            key,
            resolved.value,
            active_default,
            verbose and override is _NO_OVERRIDE,
        )
        return self._InterpretedPolicyValue(resolved, resolved.comparison)

    @property
    def _uses_composed_fallbacks(self) -> bool:
        return self._use_applied_schema_fallbacks

    def _active_default(self, default: Any, legacy_default: Any) -> Any:
        if not self._uses_composed_fallbacks and legacy_default is not _SAME_AS_DEFAULT:
            return legacy_default
        return default

    @staticmethod
    def _cached_value_reader(
        prim: Usd.Prim,
        prim_type: PrimType,
    ) -> Callable[[SchemaResolver, str], _ResolverValue]:
        value_cache: dict[tuple[int, str], _ResolverValue] = {}

        def read_value(resolver: SchemaResolver, key: str) -> _ResolverValue:
            cache_key = (id(resolver), key)
            if cache_key not in value_cache:
                value_cache[cache_key] = resolver._get_value_state(prim, prim_type, key)
            return value_cache[cache_key]

        return read_value

    def _get_value_with_policy(
        self,
        prim: Usd.Prim,
        prim_type: PrimType,
        key: str,
        default: Any,
        *,
        legacy_default: Any,
        audit_provenance: SchemaResolverManager._AuditProvenance,
        audit_fallbacks: bool,
        comparison_key: Callable[[Any, SchemaResolver | None], Any] | None,
        result_interpreter: Callable[[_ResolvedValue], Any] | None = None,
    ) -> _ResolvedValue:
        read_value = self._cached_value_reader(prim, prim_type)

        if self._uses_composed_fallbacks:
            resolved = self._resolve_value(prim, prim_type, key, default=default, read_value=read_value)
            if resolved.resolver is not None:
                self._collect_on_first_use(resolved.resolver, prim)
        else:
            resolved = self._get_legacy_value(
                prim,
                prim_type,
                key,
                self._active_default(default, legacy_default),
                read_value=read_value,
            )
        if result_interpreter is not None:
            resolved = _ResolvedValue(
                resolved.value,
                resolved.resolver,
                resolved.source,
                result_interpreter(resolved),
                resolved.compatibility_resolver,
                resolved.mapping_key,
            )
        elif comparison_key is not None:
            resolved = _ResolvedValue(
                resolved.value,
                resolved.resolver,
                resolved.source,
                comparison_key(resolved.value, resolved.resolver),
                resolved.compatibility_resolver,
                resolved.mapping_key,
            )
        if not self._uses_composed_fallbacks and audit_fallbacks:
            self._record_legacy_fallback(
                prim,
                prim_type,
                key,
                default,
                resolved,
                audit_provenance=audit_provenance,
                comparison_key=comparison_key,
                result_interpreter=result_interpreter,
                read_value=read_value,
            )
        return resolved

    def _resolve_specialized_value(
        self,
        prim: Usd.Prim,
        prim_type: PrimType,
        key: str,
        default: Any,
        resolve_legacy: Callable[
            [Sequence[SchemaResolver], Callable[[SchemaResolver, str], _ResolverValue]],
            _ResolvedValue,
        ],
        *,
        result_interpreter: Callable[[_ResolvedValue], Any] | None = None,
    ) -> _ResolvedValue:
        """Resolve a specialized consumer value under the shared policy and audit."""
        read_value = self._cached_value_reader(prim, prim_type)

        if self._uses_composed_fallbacks:
            resolved = self._resolve_value(prim, prim_type, key, default=default, read_value=read_value)
        else:
            resolved = resolve_legacy(tuple(self.resolvers), read_value)
            self._record_legacy_fallback(
                prim,
                prim_type,
                key,
                default,
                resolved,
                audit_provenance=self._AuditProvenance.SOURCE,
                result_interpreter=result_interpreter,
                read_value=read_value,
            )

        if resolved.resolver is not None:
            self._collect_on_first_use(resolved.resolver, prim)
        return resolved

    def _resolve_policy_values(
        self,
        prim: Usd.Prim,
        prim_type: PrimType,
        key: str,
        default: Any,
        *,
        legacy_default: Any = _SAME_AS_DEFAULT,
        resolve_legacy: Callable[
            [Sequence[SchemaResolver], Callable[[SchemaResolver, str], _ResolverValue]],
            _ResolvedValue,
        ]
        | None = None,
        read_value: Callable[[SchemaResolver, str], _ResolverValue] | None = None,
        authored_aliases: Sequence[str] = (),
    ) -> SchemaResolverManager._PolicyValues:
        """Resolve active and migration-comparison values without auditing."""
        if read_value is None:
            read_value = self._cached_value_reader(prim, prim_type)

        if self._uses_composed_fallbacks:
            composed = self._resolve_value(
                prim,
                prim_type,
                key,
                default=default,
                read_value=read_value,
                authored_aliases=authored_aliases,
            )
            if composed.resolver is not None:
                self._collect_on_first_use(composed.resolver, prim)
            return self._PolicyValues(composed, None, composed)

        if resolve_legacy is None:
            legacy = self._get_legacy_value(
                prim,
                prim_type,
                key,
                self._active_default(default, legacy_default),
                read_value=read_value,
            )
        else:
            legacy = resolve_legacy(tuple(self.resolvers), read_value)
            if legacy.resolver is not None:
                self._collect_on_first_use(legacy.resolver, prim)

        try:
            composed = self._resolve_value(
                prim,
                prim_type,
                key,
                default=default,
                read_value=read_value,
                authored_aliases=authored_aliases,
            )
        except _SchemaFallbackError as error:
            self._record_fallback_location(self._legacy_fallback_failures, error.label, prim)
            composed = None
        return self._PolicyValues(legacy, legacy, composed)

    def _resolve_interpreted_policies(
        self,
        prim: Usd.Prim,
        prim_type: PrimType,
        key: str,
        default: Any = None,
        *,
        interpreter: Callable[[_ResolvedValue], Any] | None = None,
        legacy_default: Any = _SAME_AS_DEFAULT,
        verbose: bool = False,
        resolve_legacy: Callable[
            [Sequence[SchemaResolver], Callable[[SchemaResolver, str], _ResolverValue]],
            _ResolvedValue,
        ]
        | None = None,
        read_value: Callable[[SchemaResolver, str], _ResolverValue] | None = None,
        authored_aliases: Sequence[str] = (),
    ) -> _InterpretedPolicyValues:
        """Resolve, interpret, and report one property's policy values."""
        policies = self._resolve_policy_values(
            prim,
            prim_type,
            key,
            default,
            legacy_default=legacy_default,
            resolve_legacy=resolve_legacy,
            read_value=read_value,
            authored_aliases=authored_aliases,
        )

        def interpret(resolved: _ResolvedValue | None) -> SchemaResolverManager._InterpretedPolicyValue | None:
            if resolved is None:
                return None
            value = resolved.value if interpreter is None else interpreter(resolved)
            return self._InterpretedPolicyValue(resolved, value)

        active = interpret(policies.active)
        assert active is not None
        self._report_missing(
            prim,
            prim_type,
            key,
            active.value,
            self._active_default(default, legacy_default),
            verbose,
        )
        return self._InterpretedPolicyValues(
            key,
            policies,
            active,
            interpret(policies.legacy),
            interpret(policies.composed),
        )

    def _get_legacy_value(
        self,
        prim: Usd.Prim,
        prim_type: PrimType,
        key: str,
        default: Any,
        *,
        read_value: Callable[[SchemaResolver, str], _ResolverValue],
    ) -> _ResolvedValue:
        for resolver in self.resolvers:
            value = read_value(resolver, key).value
            if value is None:
                continue
            self._collect_on_first_use(resolver, prim)
            return _ResolvedValue(value, resolver, _ValueSource.AUTHORED, mapping_key=key)

        has_importer_default, importer_default = _importer_default(default)
        if has_importer_default:
            return _ResolvedValue(importer_default, None, _ValueSource.IMPORTER_DEFAULT)

        for resolver in self.resolvers:
            spec = resolver.mapping.get(prim_type, {}).get(key)
            if spec is None or spec.default is None:
                continue
            value = spec.default
            if spec.usd_value_transformer is not None:
                value = spec.usd_value_transformer(value)
            return _ResolvedValue(
                value,
                None,
                _ValueSource.COMPATIBILITY_DEFAULT,
                compatibility_resolver=resolver,
                mapping_key=key,
            )

        return _ResolvedValue(None, None, _ValueSource.UNRESOLVED)

    @staticmethod
    def _values_equal(left: Any, right: Any) -> bool:
        if left is right:
            return True
        if isinstance(left, Real) and isinstance(right, Real):
            if math.isnan(float(left)) and math.isnan(float(right)):
                return True
            return math.isclose(float(left), float(right), rel_tol=1.0e-7, abs_tol=1.0e-12)
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            if not isinstance(left, Mapping) or not isinstance(right, Mapping) or left.keys() != right.keys():
                return False
            return all(SchemaResolverManager._values_equal(left[key], right[key]) for key in left)
        if isinstance(left, Set) or isinstance(right, Set):
            return isinstance(left, Set) and isinstance(right, Set) and left == right
        if (
            isinstance(left, Sequence)
            and isinstance(right, Sequence)
            and not isinstance(left, (str, bytes))
            and not isinstance(right, (str, bytes))
        ):
            return len(left) == len(right) and all(
                SchemaResolverManager._values_equal(a, b) for a, b in zip(left, right, strict=True)
            )
        try:
            equal = left == right
            if isinstance(equal, bool):
                return equal
            if hasattr(equal, "all"):
                return bool(equal.all())
            return all(equal)
        except (TypeError, ValueError, OverflowError):
            return False

    def _record_legacy_fallback(
        self,
        prim: Usd.Prim,
        prim_type: PrimType,
        key: str,
        default: Any,
        legacy: _ResolvedValue,
        *,
        audit_provenance: SchemaResolverManager._AuditProvenance,
        comparison_key: Callable[[Any, SchemaResolver | None], Any] | None = None,
        result_interpreter: Callable[[_ResolvedValue], Any] | None = None,
        read_value: Callable[[SchemaResolver, str], _ResolverValue] | None = None,
    ) -> None:
        """Record properties whose legacy and composed resolution diverge."""
        if self._uses_composed_fallbacks:
            return

        try:
            resolved = self._resolve_value(prim, prim_type, key, default=default, read_value=read_value)
        except _SchemaFallbackError as error:
            self._record_fallback_location(self._legacy_fallback_failures, error.label, prim)
            return
        if resolved.authored and audit_provenance != self._AuditProvenance.SOURCE:
            return
        if legacy.comparison is not _NO_COMPARISON:
            legacy_comparison = legacy.comparison
            if result_interpreter is not None:
                composed_comparison = result_interpreter(resolved)
            else:
                assert comparison_key is not None
                composed_comparison = comparison_key(resolved.value, resolved.resolver)
        elif result_interpreter is not None:
            legacy_comparison = result_interpreter(legacy)
            composed_comparison = result_interpreter(resolved)
        elif comparison_key is None:
            legacy_comparison = legacy.value
            composed_comparison = resolved.value
        else:
            legacy_comparison = comparison_key(legacy.value, legacy.resolver)
            composed_comparison = comparison_key(resolved.value, resolved.resolver)
        values_differ = not self._values_equal(legacy_comparison, composed_comparison)
        provenance_differ = (
            comparison_key is None
            and result_interpreter is None
            and (
                (audit_provenance == self._AuditProvenance.RESOLVER and legacy.resolver is not resolved.resolver)
                or (
                    audit_provenance == self._AuditProvenance.SOURCE
                    and (legacy.source != resolved.source or legacy.resolver is not resolved.resolver)
                )
            )
        )
        if not values_differ and not provenance_differ:
            return

        self._record_resolution_transition(
            prim,
            prim_type,
            key,
            legacy,
            resolved,
        )

    def _audit_assembled_property(
        self,
        prim: Usd.Prim,
        prim_type: PrimType,
        legacy_comparison: Any,
        composed_comparison: Any,
        candidates: Sequence[SchemaResolverManager._PolicyChangeCandidate],
    ) -> None:
        """Audit inputs that contribute to an assembled property change."""
        if self._uses_composed_fallbacks or self._values_equal(legacy_comparison, composed_comparison):
            return

        for candidate in candidates:
            policies = candidate.policies
            legacy = policies.legacy
            composed = policies.composed
            if legacy is None or composed is None:
                continue
            if candidate.legacy_comparison is _NO_COMPARISON:
                legacy_value = legacy.value
                composed_value = composed.value
            else:
                legacy_value = candidate.legacy_comparison
                composed_value = candidate.composed_comparison
            values_differ = not self._values_equal(legacy_value, composed_value)
            sources_differ = legacy.source != composed.source or legacy.resolver is not composed.resolver
            if not values_differ and not (candidate.compare_source and sources_differ):
                continue
            self._record_resolution_transition(
                prim,
                prim_type,
                candidate.key,
                legacy,
                composed,
            )

    def _record_resolution_transition(
        self,
        prim: Usd.Prim,
        prim_type: PrimType,
        key: str,
        legacy: _ResolvedValue,
        resolved: _ResolvedValue,
    ) -> None:
        """Record one property transition between resolution policies."""

        def endpoint(value: _ResolvedValue) -> SchemaResolverManager._MigrationEndpoint:
            resolver = value.resolver or value.compatibility_resolver
            if resolver is None:
                return self._MigrationEndpoint(value.source)
            mapping_key = value.mapping_key or key
            spec = resolver.mapping.get(prim_type, {}).get(mapping_key)
            if spec is None:
                return self._MigrationEndpoint(value.source, resolver.name)
            owner = (
                resolver.name
                if value.source == _ValueSource.COMPATIBILITY_DEFAULT
                else resolver._schema_name(prim_type, mapping_key) or resolver.name
            )
            names = tuple(spec.attribute_names or (spec.name,))
            return self._MigrationEndpoint(value.source, owner, names)

        legacy_endpoint = endpoint(legacy)
        resolved_endpoint = endpoint(resolved)
        representative = (
            legacy.resolver or legacy.compatibility_resolver or resolved.resolver or resolved.compatibility_resolver
        )
        if representative is None:
            representative = next(
                (resolver for resolver in self.resolvers if key in resolver.mapping.get(prim_type, {})),
                None,
            )
        if representative is None:
            return

        attribute_names = legacy_endpoint.attribute_names or resolved_endpoint.attribute_names
        if not attribute_names:
            mapping = representative.mapping.get(prim_type, {})
            representative_key = legacy.mapping_key or resolved.mapping_key or key
            spec = mapping.get(representative_key) or mapping.get(key)
            if spec is not None:
                attribute_names = tuple(spec.attribute_names or (spec.name,))
        transition = self._MigrationTransition(
            key,
            attribute_names,
            legacy_endpoint,
            resolved_endpoint,
        )
        self._record_fallback_location(
            self._legacy_fallback_properties,
            transition,
            prim,
        )

    @staticmethod
    def _prim_path(prim: Usd.Prim) -> str:
        try:
            return str(prim.GetPath()) if prim is not None else "<None>"
        except (AttributeError, RuntimeError):
            return "<invalid>"

    @classmethod
    def _record_fallback_location(cls, entries: dict[Any, set[str]], label: Any, prim: Usd.Prim) -> None:
        entries.setdefault(label, set()).add(cls._prim_path(prim))

    @staticmethod
    def _format_fallback_locations(entries: dict[Any, set[str]], max_paths: int = 3) -> str:
        details = []
        formatted_entries = (
            (label if isinstance(label, str) else label.format(), paths) for label, paths in entries.items()
        )
        for label, all_paths in sorted(formatted_entries):
            paths = sorted(all_paths)
            shown_paths = ", ".join(paths[:max_paths])
            omitted = len(paths) - max_paths
            if omitted > 0:
                shown_paths = f"{shown_paths}, and {omitted} more"
            details.append(f"{label} on {shown_paths}")
        return "; ".join(details)

    def _fallback_migration_warning(self) -> str | None:
        """Build one actionable warning for audited precedence changes."""
        if self._uses_composed_fallbacks or not self._legacy_fallback_properties:
            return None

        properties = self._format_fallback_locations(self._legacy_fallback_properties)
        details = [f"future USD property precedence changes resolution for {properties}"]
        if self._legacy_fallback_failures:
            failures = self._format_fallback_locations(self._legacy_fallback_failures)
            details.append(f"schema fallbacks could not be audited for {failures}")
        return (
            "This import used deprecated legacy USD property precedence; "
            f"{' and '.join(details)}. In a future release, registered schema fallbacks will be considered "
            "before importer defaults. Compatibility defaults will remain available only for resolvers with an "
            "applicable unregistered schema or without declared schema ownership; "
            "pass use_applied_schema_fallbacks=True to adopt that behavior now. To preserve current results, "
            "author values on a higher-priority schema, reorder schema_resolvers, or use a supported explicit "
            "importer override."
        )

    @staticmethod
    def _report_missing(
        prim: Usd.Prim,
        prim_type: PrimType,
        key: str,
        value: Any,
        default: Any,
        verbose: bool,
    ) -> None:
        has_importer_default, _ = _importer_default(default)
        if value is not None or has_importer_default or not verbose:
            return
        prim_path = SchemaResolverManager._prim_path(prim)
        print(
            f"Error: Cannot resolve value for '{prim_type.name.lower()}:{key}' on prim '{prim_path}'; "
            "no authored value, explicit default, or applicable resolver fallback."
        )

    def _resolve_value(
        self,
        prim: Usd.Prim,
        prim_type: PrimType,
        key: str,
        *,
        default: Any = None,
        read_value: Callable[[SchemaResolver, str], _ResolverValue] | None = None,
        authored_aliases: Sequence[str] = (),
    ) -> _ResolvedValue:
        """Resolve a value while retaining source provenance."""

        def read_from_prim(resolver: SchemaResolver, key: str) -> _ResolverValue:
            return resolver._get_value_state(prim, prim_type, key)

        return self._resolution._resolve_value(
            read_from_prim if read_value is None else read_value,
            lambda resolver, key: resolver._schema_is_applied(prim, prim_type, key),
            lambda resolver, key: self._schema_fallback(resolver, prim, prim_type, key),
            prim_type,
            key,
            default=default,
            authored_aliases=authored_aliases,
        )

    def _schema_fallback(
        self,
        resolver: SchemaResolver,
        prim: Usd.Prim,
        prim_type: PrimType,
        key: str,
    ) -> Any:
        schema_name = resolver._schema_name(prim_type, key)
        if schema_name is None or prim is None:
            return _MISSING_FALLBACK

        from pxr import Usd

        prim_type_name = str(prim.GetTypeName())
        cache_key = (prim_type_name, schema_name)
        if cache_key not in self._registered_schema_fallbacks:
            registry = Usd.SchemaRegistry()
            if prim_type_name == schema_name:
                prim_definition = registry.FindConcretePrimDefinition(schema_name)
            else:
                schema_type_name, _ = registry.GetTypeNameAndInstance(schema_name)
                schema_definition = registry.FindAppliedAPIPrimDefinition(schema_type_name)
                prim_definition = (
                    registry.BuildComposedPrimDefinition(prim_type_name, [schema_name])
                    if schema_definition is not None
                    else None
                )
            self._registered_schema_fallbacks[cache_key] = (
                _registered_attribute_fallbacks(prim_definition) if prim_definition is not None else None
            )

        fallbacks = self._registered_schema_fallbacks[cache_key]
        if fallbacks is None:
            return _UNREGISTERED_SCHEMA
        try:
            authored_values = {}
            use_authored_values = False
            spec = resolver.mapping.get(prim_type, {}).get(key)
            if spec is not None and spec._reader_value_getter is not None:
                names = spec.attribute_names or (spec.name,)
                for name in names:
                    attribute = prim.GetAttribute(name)
                    if attribute and attribute.HasAuthoredValue():
                        value = attribute.Get()
                        if value is not None:
                            authored_values[name] = value

                # A compound property follows USD composition per constituent.
                # Fully authored but unusable values still fall through to the
                # registered fallback as one candidate.
                use_authored_values = 0 < len(authored_values) < len(names)

            def read_fallback(name: str) -> Any:
                if use_authored_values and name in authored_values:
                    return authored_values[name]
                return fallbacks.get(name, _MISSING_FALLBACK)

            return resolver._get_fallback_with_reader(
                read_fallback,
                prim_type,
                key,
            )
        except _PXRValueGetterError:
            return _UNREADABLE_FALLBACK

    def deformable_compat_namespaces(self) -> list[str]:
        """Deformable vendor attribute namespaces declared by the active resolvers.

        Returns the union of every resolver's ``deformable_attr_namespaces``, in
        resolver priority order. Used to accept deformable material/geometry
        attributes authored under vendor namespaces (e.g. ``omniphysics:``,
        ``physxDeformableBody:``) as a fallback to the canonical ``physics:``
        schema. This is deliberately separate from the generic
        ``extra_attr_namespaces`` so unrelated namespaces (``physxScene``,
        ``drive``, ``state``, ...) are never read as deformable schema attributes.
        Empty by default, so a default import reads only the canonical schema.
        """
        seen: set[str] = set()
        namespaces: list[str] = []
        for r in self.resolvers:
            for ns in r.deformable_attr_namespaces:
                if ns not in seen:
                    seen.add(ns)
                    namespaces.append(ns)
        return namespaces

    def read_deformable_attr(self, prim: Usd.Prim, name: str) -> Any:
        """Read a deformable physics attribute: canonical ``physics:`` first, then the
        resolver-declared vendor namespaces. The first authored value, or ``None``."""
        return usd._read_physics_attr(prim, name, self.deformable_compat_namespaces())

    def collect_prim_attrs(self, prim: Usd.Prim) -> None:
        """
        Collect and accumulate schema attributes for a single prim.

        Args:
            prim: USD prim to collect attributes from
        """
        if prim is None:
            return

        prim_path = str(prim.GetPath())

        for resolver in self.resolvers:
            # only collect if we haven't seen this prim for this resolver
            if prim_path not in self._schema_attrs[resolver.name]:
                self._schema_attrs[resolver.name][prim_path] = resolver.collect_prim_attrs(prim)

    @property
    def schema_attrs(self) -> dict[str, dict[str, dict[str, Any]]]:
        """
        Get the accumulated attributes.

        Returns:
            Dictionary with structure: schema_name -> prim_path -> {attr_name: attr_value}
            e.g., {"mjc": {"/World/Cube": {"mjc:option:timestep": 0.01}}}
        """
        return self._schema_attrs
