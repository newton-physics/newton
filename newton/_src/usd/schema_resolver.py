# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
USD schema resolver infrastructure.

This module defines the base resolver types used to map authored USD schema
attributes onto Newton builder attributes. Public users should import resolver
types from :mod:`newton.usd`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from numbers import Real
from typing import TYPE_CHECKING, Any, ClassVar

from . import utils as usd

if TYPE_CHECKING:
    from pxr import Usd

    from ..sim.builder import ModelBuilder


_MISSING_FALLBACK = object()
_NO_OVERRIDE = object()
_UNREGISTERED_SCHEMA = object()


@dataclass(frozen=True)
class _ImporterDefault:
    """Carry an importer default that may be None."""

    value: Any

    def __repr__(self) -> str:
        return f"<omitted; importer default={self.value!r}>"


def _default_when_omitted(value: Any) -> Any:
    """Mark a public importer argument's value as its omission default."""
    return _ImporterDefault(value)


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
    blocked: bool = False


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
        """

        name: str
        default: Any | None = None
        usd_value_transformer: Callable[[Any], Any] | None = None
        usd_value_getter: Callable[[Usd.Prim], Any] | None = None
        attribute_names: Sequence[str] = ()
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
        states: dict[str, tuple[bool, bool]] = {}

        def read_attribute(name: str) -> Any | None:
            attribute = prim.GetAttribute(name)
            authored = bool(attribute and attribute.HasAuthoredValue())
            blocked = bool(attribute and not authored and attribute.GetResolveInfo().ValueIsBlocked())
            states[name] = (authored, blocked)
            return attribute.Get() if authored else None

        value = self._get_value_with_reader(read_attribute, prim_type, key, legacy_prim=prim)
        if value is not None:
            return _ResolverValue(value, True)

        for name in names:
            if name not in states:
                read_attribute(name)

        if any(blocked for _, blocked in states.values()):
            return _ResolverValue(None, False, True)
        if spec._reader_value_getter is not None:
            authored = all(authored for authored, _ in states.values())
        else:
            authored = states.get(spec.name, (False, False))[0]
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


class _SchemaResolution:
    """Applied-schema resolution policy for one ordered resolver set."""

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
    ) -> _ResolvedValue:
        compatibility_fallbacks: set[int] = set()
        for resolver in self._resolvers:
            spec = resolver.mapping.get(prim_type, {}).get(key)
            if spec is None:
                continue
            value = read_value(resolver, key)
            if not isinstance(value, _ResolverValue):
                value = _ResolverValue(value, value is not None)
            if value.authored:
                return _ResolvedValue(value.value, resolver, True)

            if not value.blocked and schema_is_applied(resolver, key):
                fallback = read_fallback(resolver, key)
                if fallback is _UNREGISTERED_SCHEMA:
                    compatibility_fallbacks.add(id(resolver))
                elif fallback is not _MISSING_FALLBACK:
                    return _ResolvedValue(fallback, resolver, False)

        has_importer_default, importer_default = _importer_default(default)
        if has_importer_default:
            return _ResolvedValue(importer_default, None, False)

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
            return _ResolvedValue(value, None, False)

        return _ResolvedValue(None, None, False)


@dataclass(frozen=True)
class _ResolvedValue:
    value: Any
    resolver: SchemaResolver | None
    authored: bool


class SchemaResolverManager:
    """
    Manager for resolving multiple USD schemas in a priority order.
    """

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
        self._resolution = _SchemaResolution(self.resolvers)
        self._registered_schema_fallbacks: dict[tuple[str, str], dict[str, Any] | None] = {}
        self._legacy_fallback_properties: dict[str, set[str]] = {}
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
        comparison_key: Callable[[Any, SchemaResolver | None], Any] | None = None,
        legacy_value_transformer: Callable[[Any, SchemaResolver | None], Any] | None = None,
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
            comparison_key: Convert a raw value and resolver into its
                consumer-observable form for the compatibility audit.
            legacy_value_transformer: Convert values returned by the legacy
                policy and authored values returned by the composed policy
                without changing registered schema fallbacks.

        Returns:
            Resolved value according to the precedence above.
        """
        if override is not _NO_OVERRIDE and self._uses_composed_fallbacks:
            return override

        value, _ = self._get_value_with_policy(
            prim,
            prim_type,
            key,
            default,
            compare_resolver=False,
            audit_fallbacks=override is _NO_OVERRIDE,
            comparison_key=comparison_key,
            legacy_value_transformer=legacy_value_transformer,
        )
        self._report_missing(prim, prim_type, key, value, verbose and override is _NO_OVERRIDE)
        return value

    def get_value_with_resolver(
        self,
        prim: Usd.Prim,
        prim_type: PrimType,
        key: str,
        default: Any = None,
        verbose: bool = False,
        *,
        override: Any = _NO_OVERRIDE,
        comparison_key: Callable[[Any, SchemaResolver | None], Any] | None = None,
        legacy_value_transformer: Callable[[Any, SchemaResolver | None], Any] | None = None,
    ) -> tuple[Any, SchemaResolver | None]:
        """Resolve a value and return the resolver that supplied it."""
        if override is not _NO_OVERRIDE and self._uses_composed_fallbacks:
            return override, None

        value, resolver = self._get_value_with_policy(
            prim,
            prim_type,
            key,
            default,
            compare_resolver=True,
            audit_fallbacks=override is _NO_OVERRIDE,
            comparison_key=comparison_key,
            legacy_value_transformer=legacy_value_transformer,
        )
        self._report_missing(prim, prim_type, key, value, verbose and override is _NO_OVERRIDE)
        return value, resolver

    @property
    def _uses_composed_fallbacks(self) -> bool:
        return self._use_applied_schema_fallbacks

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
        compare_resolver: bool,
        audit_fallbacks: bool,
        comparison_key: Callable[[Any, SchemaResolver | None], Any] | None,
        legacy_value_transformer: Callable[[Any, SchemaResolver | None], Any] | None,
    ) -> tuple[Any, SchemaResolver | None]:
        read_value = self._cached_value_reader(prim, prim_type)

        if self._uses_composed_fallbacks:
            try:
                resolved = self._resolve_value(prim, prim_type, key, default=default, read_value=read_value)
            except _PXRValueGetterError:
                value, resolver = self._get_legacy_value(prim, prim_type, key, default, read_value=read_value)
                if legacy_value_transformer is not None:
                    value = legacy_value_transformer(value, resolver)
                return value, resolver
            value = resolved.value
            if resolved.authored and legacy_value_transformer is not None:
                value = legacy_value_transformer(value, resolved.resolver)
            if resolved.resolver is not None:
                self._collect_on_first_use(resolved.resolver, prim)
            return value, resolved.resolver

        value, resolver = self._get_legacy_value(prim, prim_type, key, default, read_value=read_value)
        if legacy_value_transformer is not None:
            value = legacy_value_transformer(value, resolver)
        if audit_fallbacks:
            self._record_legacy_fallback(
                prim,
                prim_type,
                key,
                default,
                value,
                resolver,
                compare_resolver=compare_resolver,
                comparison_key=comparison_key,
                read_value=read_value,
            )
        return value, resolver

    def _resolve_specialized_value(
        self,
        prim: Usd.Prim,
        prim_type: PrimType,
        key: str,
        default: Any,
        resolve_legacy: Callable[
            [Sequence[SchemaResolver], Callable[[SchemaResolver, str], _ResolverValue]],
            tuple[Any, SchemaResolver | None, bool],
        ],
    ) -> _ResolvedValue:
        """Resolve a specialized consumer value under the shared policy and audit."""
        read_value = self._cached_value_reader(prim, prim_type)

        if self._uses_composed_fallbacks:
            resolved = self._resolve_value(prim, prim_type, key, default=default, read_value=read_value)
        else:
            value, resolver, authored = resolve_legacy(tuple(self.resolvers), read_value)
            resolved = _ResolvedValue(value, resolver, authored)
            self._record_legacy_fallback(
                prim,
                prim_type,
                key,
                default,
                resolved.value,
                resolved.resolver,
                compare_resolver=False,
                read_value=read_value,
            )

        if resolved.resolver is not None:
            self._collect_on_first_use(resolved.resolver, prim)
        return resolved

    def _get_legacy_value(
        self,
        prim: Usd.Prim,
        prim_type: PrimType,
        key: str,
        default: Any,
        *,
        read_value: Callable[[SchemaResolver, str], _ResolverValue],
    ) -> tuple[Any, SchemaResolver | None]:
        for resolver in self.resolvers:
            value = read_value(resolver, key).value
            if value is None:
                continue
            self._collect_on_first_use(resolver, prim)
            return value, resolver

        has_importer_default, importer_default = _importer_default(default)
        if has_importer_default:
            return importer_default, None

        for resolver in self.resolvers:
            spec = resolver.mapping.get(prim_type, {}).get(key)
            if spec is None or spec.default is None:
                continue
            value = spec.default
            if spec.usd_value_transformer is not None:
                value = spec.usd_value_transformer(value)
            return value, None

        return None, None

    @staticmethod
    def _values_equal(left: Any, right: Any) -> bool:
        if left is right:
            return True
        if isinstance(left, Real) and isinstance(right, Real):
            return math.isclose(float(left), float(right), rel_tol=1.0e-7, abs_tol=1.0e-12)
        try:
            equal = left == right
            if isinstance(equal, bool):
                if equal:
                    return True
                if isinstance(left, (str, bytes)) or isinstance(right, (str, bytes)):
                    return False
                return len(left) == len(right) and all(
                    SchemaResolverManager._values_equal(a, b) for a, b in zip(left, right, strict=True)
                )
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
        legacy_value: Any,
        legacy_resolver: SchemaResolver | None,
        *,
        compare_resolver: bool,
        comparison_key: Callable[[Any, SchemaResolver | None], Any] | None = None,
        read_value: Callable[[SchemaResolver, str], _ResolverValue] | None = None,
    ) -> None:
        """Record properties whose legacy and composed resolution diverge."""
        if self._uses_composed_fallbacks:
            return
        if legacy_resolver is not None:
            for resolver in self.resolvers:
                if resolver is legacy_resolver:
                    return
                if resolver._schema_is_applied(prim, prim_type, key):
                    break
            else:
                return
        elif not any(resolver._schema_is_applied(prim, prim_type, key) for resolver in self.resolvers):
            return

        try:
            resolved = self._resolve_value(prim, prim_type, key, default=default, read_value=read_value)
        except _SchemaFallbackError as error:
            self._record_fallback_location(self._legacy_fallback_failures, error.label, prim)
            return
        if resolved.resolver is None or resolved.authored:
            return
        if comparison_key is None:
            legacy_comparison = legacy_value
            composed_comparison = resolved.value
            resolvers_differ = compare_resolver and legacy_resolver is not resolved.resolver
        else:
            legacy_comparison = comparison_key(legacy_value, legacy_resolver)
            composed_comparison = comparison_key(resolved.value, resolved.resolver)
            resolvers_differ = False
        values_differ = not self._values_equal(legacy_comparison, composed_comparison)
        if not values_differ and not resolvers_differ:
            return

        spec = resolved.resolver.mapping[prim_type][key]
        schema_name = resolved.resolver._schema_name(prim_type, key)
        names = ", ".join(spec.attribute_names or (spec.name,))
        self._record_fallback_location(self._legacy_fallback_properties, f"{schema_name} ({names})", prim)

    @staticmethod
    def _prim_path(prim: Usd.Prim) -> str:
        try:
            return str(prim.GetPath()) if prim is not None else "<None>"
        except (AttributeError, RuntimeError):
            return "<invalid>"

    @classmethod
    def _record_fallback_location(cls, entries: dict[str, set[str]], label: str, prim: Usd.Prim) -> None:
        entries.setdefault(label, set()).add(cls._prim_path(prim))

    @staticmethod
    def _format_fallback_locations(entries: dict[str, set[str]], max_paths: int = 3) -> str:
        details = []
        for label, all_paths in sorted(entries.items()):
            paths = sorted(all_paths)
            shown_paths = ", ".join(paths[:max_paths])
            omitted = len(paths) - max_paths
            if omitted > 0:
                shown_paths = f"{shown_paths}, and {omitted} more"
            details.append(f"{label} on {shown_paths}")
        return "; ".join(details)

    def _fallback_migration_warning(self) -> str | None:
        """Build one actionable warning for audited registered-schema changes."""
        details = []
        if self._legacy_fallback_properties:
            properties = self._format_fallback_locations(self._legacy_fallback_properties)
            details.append(f"schema fallbacks will take precedence for {properties}")
        if self._legacy_fallback_failures:
            failures = self._format_fallback_locations(self._legacy_fallback_failures)
            details.append(f"schema fallbacks could not be audited for {failures}")
        if not details:
            return None
        return (
            "This import retained legacy values for applied but unauthored USD schema properties; "
            f"{' and '.join(details)}. In a future release, applied-schema fallbacks will take precedence; "
            "pass use_applied_schema_fallbacks=True to adopt that behavior now, or author the intended values "
            "explicitly to preserve them."
        )

    @staticmethod
    def _report_missing(prim: Usd.Prim, prim_type: PrimType, key: str, value: Any, verbose: bool) -> None:
        if value is not None or not verbose:
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
        value = resolver._get_fallback_with_reader(
            lambda name: fallbacks.get(name, _MISSING_FALLBACK),
            prim_type,
            key,
        )
        return value

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
