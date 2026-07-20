# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
USD schema resolver infrastructure.

This module defines the base resolver types used to map authored USD schema
attributes onto Newton builder attributes. Public users should import resolver
types from :mod:`newton.usd`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, ClassVar

from . import utils as usd
from ._schema_fallbacks import _schema_fallbacks

if TYPE_CHECKING:
    from pxr import Usd

    from ..sim.builder import ModelBuilder


_MISSING_FALLBACK = object()


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
            default: Legacy fallback used by the existing resolver path.
            usd_value_transformer: Optional function to transform the raw value into the format expected by Newton.
            usd_value_getter: Optional function (prim) -> value used instead of reading a single attribute (e.g. to compute gap from contactOffset - restOffset).
            attribute_names: When set, names used for collect_prim_attrs; otherwise [name] is used.
        """

        name: str
        default: Any | None = None
        usd_value_transformer: Callable[[Any], Any] | None = None
        usd_value_getter: Callable[[Usd.Prim], Any] | None = None
        attribute_names: Sequence[str] = ()

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
            reader_value_getter = getattr(spec, "_reader_value_getter", None)
            if reader_value_getter is not None:
                v = reader_value_getter(read_attribute)
            elif spec.usd_value_getter is not None:
                if legacy_prim is None:
                    raise TypeError(
                        f"Schema resolver '{self.name}' key '{prim_type.name.lower()}:{key}' uses a "
                        "PXR-only usd_value_getter and cannot resolve schema fallbacks."
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


class _SchemaResolution:
    """Applied-schema resolution policy for one ordered resolver set."""

    def __init__(
        self,
        resolvers: Sequence[SchemaResolver],
        schema_fallbacks: Mapping[str, Mapping[str, Any]] | None = None,
    ):
        self._resolvers = tuple(resolvers)
        self._schema_fallbacks = _schema_fallbacks(schema_fallbacks)

    def _mapping_fallback(self, resolver: SchemaResolver, prim_type: PrimType, key: str) -> Any:
        schema_name = resolver._schema_name(prim_type, key)
        values = self._schema_fallbacks.get(schema_name, {}) if schema_name is not None else {}
        return resolver._get_fallback_with_reader(
            lambda name: values.get(name, _MISSING_FALLBACK),
            prim_type,
            key,
        )

    def _resolve_value(
        self,
        read_value: Callable[[SchemaResolver, str], Any | None],
        schema_is_applied: Callable[[SchemaResolver, str], bool],
        read_fallback: Callable[[SchemaResolver, str], Any],
        prim_type: PrimType,
        key: str,
        *,
        default: Any = None,
    ) -> _ResolvedValue:
        for resolver in self._resolvers:
            spec = resolver.mapping.get(prim_type, {}).get(key)
            if spec is None:
                continue
            value = read_value(resolver, key)
            if value is not None:
                return _ResolvedValue(value, resolver, True)

            if schema_is_applied(resolver, key):
                value = read_fallback(resolver, key)
                if value is _MISSING_FALLBACK:
                    schema_name = resolver._schema_name(prim_type, key)
                    names = ", ".join(spec.attribute_names or (spec.name,))
                    raise RuntimeError(
                        f"Cannot resolve USD fallback for applied schema '{schema_name}' property '{names}'. "
                        "Register the schema plugin or supply schema_fallbacks."
                    )
                return _ResolvedValue(value, resolver, False)

        if default is not None:
            return _ResolvedValue(default, None, False)

        for resolver in self._resolvers:
            spec = resolver.mapping.get(prim_type, {}).get(key)
            if (
                spec is None
                or not resolver._use_legacy_unowned_defaults
                or resolver._schema_name(prim_type, key) is not None
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


def create_schema_resolution(
    resolvers: Sequence[SchemaResolver],
    *,
    schema_fallbacks: Mapping[str, Mapping[str, Any]] | None = None,
):
    """Create an applied-schema resolution policy for :meth:`ModelBuilder.add_usd`.

    Args:
        resolvers: Resolver instances in priority order.
        schema_fallbacks: Overrides or additions to the built-in USD fallback
            catalog, keyed first by schema name and then by attribute name.

    Returns:
        An opaque policy to pass to ``ModelBuilder.add_usd(schema_resolution=...)``.
    """
    return _SchemaResolution(resolvers, schema_fallbacks)


class SchemaResolverManager:
    """
    Manager for resolving multiple USD schemas in a priority order.
    """

    def __init__(
        self,
        resolvers: Sequence[SchemaResolver] | None = None,
        resolution: _SchemaResolution | None = None,
    ):
        """
        Initialize resolver manager with resolver instances in priority order.

        Args:
            resolvers: List of instantiated resolvers in priority order.
            resolution: Optional applied-schema resolution policy.
        """
        if resolution is not None:
            if resolvers is not None:
                raise ValueError("resolvers and resolution are mutually exclusive")
            self.resolvers = list(resolution._resolvers)
        elif resolvers is not None:
            self.resolvers = list(resolvers)
        else:
            raise ValueError("resolvers or resolution is required")
        self._resolution = resolution
        self._pxr_schema_fallbacks: dict[tuple[str, str], dict[str, Any]] = {}

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
        self, prim: Usd.Prim, prim_type: PrimType, key: str, default: Any = None, verbose: bool = False
    ) -> Any:
        """
        Resolve a value using the configured resolver policy.

        Args:
            prim: USD prim to query (for scene prim_type, this should be scene_prim)
            prim_type: Prim type (PrimType enum)
            key: Attribute key within the prim type
            default: Default value if not found

        Returns:
            Resolved value according to the precedence above.
        """
        value, _ = self.get_value_with_resolver(prim, prim_type, key, default, verbose)
        return value

    def get_value_with_resolver(
        self, prim: Usd.Prim, prim_type: PrimType, key: str, default: Any = None, verbose: bool = False
    ) -> tuple[Any, SchemaResolver | None]:
        """Resolve a value and return the resolver that supplied it."""
        if self._resolution is None:
            for resolver in self.resolvers:
                value = resolver.get_value(prim, prim_type, key)
                if value is None:
                    continue
                self._collect_on_first_use(resolver, prim)
                return value, resolver

            if default is not None:
                return default, None

            for resolver in self.resolvers:
                spec = resolver.mapping.get(prim_type, {}).get(key)
                if spec is None or spec.default is None:
                    continue
                value = spec.default
                if spec.usd_value_transformer is not None:
                    value = spec.usd_value_transformer(value)
                return value, None
        else:
            resolved = self._resolve_value(prim, prim_type, key, default=default)
            if resolved.resolver is not None:
                self._collect_on_first_use(resolved.resolver, prim)
                return resolved.value, resolved.resolver
            if resolved.value is not None:
                return resolved.value, None

        # Nothing found
        try:
            prim_path = str(prim.GetPath()) if prim is not None else "<None>"
        except (AttributeError, RuntimeError):
            prim_path = "<invalid>"
        if verbose:
            error_message = (
                f"Error: Cannot resolve value for '{prim_type.name.lower()}:{key}' on prim '{prim_path}'; "
                + "no authored value, explicit default, or applicable resolver fallback."
            )
            print(error_message)
        return None, None

    def _resolve_value(
        self,
        prim: Usd.Prim,
        prim_type: PrimType,
        key: str,
        *,
        default: Any = None,
    ) -> _ResolvedValue:
        """Resolve a value while retaining source provenance."""
        if self._resolution is None:
            raise RuntimeError("composed schema resolution is not enabled")
        return self._resolution._resolve_value(
            lambda resolver, key: resolver.get_value(prim, prim_type, key),
            lambda resolver, key: resolver._schema_is_applied(prim, prim_type, key),
            lambda resolver, key: self._pxr_fallback(resolver, prim, prim_type, key),
            prim_type,
            key,
            default=default,
        )

    def _pxr_fallback(
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
        if cache_key not in self._pxr_schema_fallbacks:
            registry = Usd.SchemaRegistry()
            if prim_type_name == schema_name:
                prim_definition = registry.FindConcretePrimDefinition(schema_name)
            else:
                prim_definition = registry.BuildComposedPrimDefinition(prim_type_name, [schema_name])
            self._pxr_schema_fallbacks[cache_key] = (
                {name: prim_definition.GetAttributeFallbackValue(name) for name in prim_definition.GetPropertyNames()}
                if prim_definition is not None
                else {}
            )

        fallbacks = self._pxr_schema_fallbacks[cache_key]
        value = resolver._get_fallback_with_reader(
            lambda name: fallbacks.get(name, _MISSING_FALLBACK),
            prim_type,
            key,
        )
        if value is not _MISSING_FALLBACK:
            return value
        return self._resolution._mapping_fallback(resolver, prim_type, key)

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
