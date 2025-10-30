# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, ClassVar

from ..sim.model import CustomAttribute, ModelAttributeFrequency
from . import utils as usd

if TYPE_CHECKING:
    from pxr import Usd


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


@dataclass
class SchemaAttribute:
    """
    Specifies a USD attribute and its transformation function.

    Args:
        usd_name (str): The name of the USD attribute.
        default (Any | None): Default USD-authored value from schema, if any.
        transformer (Callable[[Any], Any] | None): A function to transform the raw USD attribute value
            into the format expected by Newton. Takes the USD value as input and returns the transformed
            value. For example, converting PhysX timeStepsPerSecond to Newton's timestep by computing 1/Hz.
    """

    usd_name: str
    default: Any | None = None
    transformer: Callable[[Any], Any] | None = None


class SchemaResolver:
    # mapping is a dictionary for known variables in Newton. Its purpose is to map USD attributes to existing Newton data.
    # PrimType -> Newton variable -> Attribute
    mapping: ClassVar[dict[PrimType, dict[str, SchemaAttribute]]]

    # Name of the schema resolver
    name: ClassVar[str]

    # extra_attr_namespaces is a list of namespaces for extra attributes that are not in the mapping.
    extra_attr_namespaces: ClassVar[list[str]]

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
                names.add(spec.usd_name)
        self._solver_attributes: list[str] = list(names)

    def get_value(self, prim, prim_type: PrimType, key: str) -> tuple[Any, str] | None:
        """
        Get attribute value for a given prim type and key.

        Args:
            prim: USD prim to query
            prim_type: Prim type (PrimType enum)
            key: Attribute key within the prim type

        Returns:
            Tuple of (value, usd_attr_name) if found, None otherwise
        """
        if prim is None:
            return None
        spec = self.mapping.get(prim_type, {}).get(key)
        if spec is not None:
            v = usd.get_attribute(prim, spec.usd_name)
            if v is not None:
                return (spec.transformer(v) if spec.transformer is not None else v), spec.usd_name
        return None

    def collect_prim_attrs(self, prim) -> dict[str, Any]:
        """
        Collect attributes pertaining to this schema for a single prim.
        Returns dictionary of attributes for this prim.
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


def _collect_attrs_by_name(prim, names: list[str]) -> dict[str, Any]:
    """Collect attributes authored on the prim that have direct mappings in the resolver mapping"""
    out: dict[str, Any] = {}
    for n in names:
        v = usd.get_attribute(prim, n)
        if v is not None:
            out[n] = v
    return out


def _collect_attrs_by_namespace(prim, namespaces: list[str]) -> dict[str, Any]:
    """Collect authored attributes using USD namespace queries."""

    out: dict[str, Any] = {}
    if prim is None or Usd is None:
        return out

    for ns in namespaces:
        for prop in prim.GetAuthoredPropertiesInNamespace(ns):
            if isinstance(prop, Usd.Attribute) and prop.IsValid() and prop.HasAuthoredValue():
                out[prop.GetName()] = prop.Get()

    return out


class SchemaResolverManager:
    def __init__(
        self,
        resolvers: list[SchemaResolver],
        existing_custom_attributes: dict[str, CustomAttribute],
        collect_schema_attrs: bool = True,
    ):
        """
        Initialize resolver manager with resolver instances in priority order.

        Args:
            resolvers: List of instantiated resolvers in priority order.
            existing_custom_attributes: Dictionary of existing custom attributes, e.g. from :attr:`ModelBuilder.custom_attributes`.
            collect_schema_attrs: Whether to collect schema-specific attributes.
        """
        # Use provided resolver instances directly
        self.resolvers = list(resolvers)
        self._collect_schema_attrs = bool(collect_schema_attrs)

        # Dictionary to accumulate schema attributes as prims are encountered
        # Pre-initialize maps for each configured resolver
        self.schema_attrs: dict[str, dict[str, dict[str, Any]]] = {r.name: {} for r in self.resolvers}

        # accumulator for declared custom attributes (declaration-first pattern)
        # Key format: "namespace:attr_name" or "attr_name" for default namespace
        self._custom_attributes: dict[str, CustomAttribute] = existing_custom_attributes

        # Flag to track if declarations have been parsed from PhysicsScene
        self._declarations_parsed = False

    def _collect_on_first_use(self, resolver: SchemaResolver, prim) -> None:
        """Collect and store attributes for this resolver/prim on first use."""
        if prim is None:
            return
        if not self._collect_schema_attrs:
            return
        prim_path = str(prim.GetPath())
        if prim_path in self.schema_attrs[resolver.name]:
            return
        attrs = resolver.collect_prim_attrs(prim)
        if attrs:
            self.schema_attrs[resolver.name][prim_path] = attrs

    def get_custom_attributes_for_prim(self, prim, expected_frequency: ModelAttributeFrequency) -> dict[str, Any]:
        """
        Extract custom attributes from a prim that match the expected frequency.
        Providing the frequency is an additional check to ensure the attributes are used with the correct prim type.

        This method reads custom attributes from a USD prim, validates they are
        declared and match the expected frequency, then returns them as a
        dictionary.

        Args:
            prim: USD prim to extract attributes from
            expected_frequency: Expected frequency for the prim type

        Returns:
            Dictionary of custom attribute values keyed by full name
            (namespace:attr_name or attr_name)
        """
        if prim is None:
            return {}

        result = {}

        ####

        def resolve_custom_attributes(prim: Usd.Prim, custom_attributes: list[CustomAttribute]) -> dict[str, Any]:
            custom_attrs = {}
            for attr in custom_attributes:
                usd_attr_name = attr.usd_attribute_name
                usd_attr = usd.get_attribute(prim, usd_attr_name)
                if usd_attr is not None and usd_attr.HasAuthoredValue():
                    custom_attrs[attr.key] = usd_attr.Get()
            return custom_attrs

        ###

        for attr in prim.GetAttributes():
            attr_name = attr.GetName()
            parsed = self._parse_custom_attr_name(attr_name)
            if not parsed:
                continue

            namespace, local_name = parsed
            full_key = f"{namespace}:{local_name}" if namespace else local_name

            # Check if this attribute was declared
            custom_attr = self._custom_attributes.get(full_key)
            if custom_attr is None:
                # Attribute not declared - skip it since we only collect attributes that are declared
                continue

            # Verify frequency matches (use frequency from declaration)
            if custom_attr.frequency != expected_frequency:
                continue

            # Get the value and convert it
            value = attr.Get()
            converted_value = self._usd_to_wp(value)

            result[full_key] = converted_value

        return result

    def get_value(self, prim, prim_type: PrimType, key: str, default: Any = None) -> Any:
        """
        Resolve value using schema priority, with layered fallbacks:

        1) First authored value found in resolver order (highest priority first)
        2) If none authored, use the provided 'default' argument if not None
        3) If still None, use the first non-None mapping default from resolvers in priority order

        Args:
            prim: USD prim to query (for scene prim_type, this should be scene_prim)
            prim_type: Prim type (PrimType enum)
            key: Attribute key within the prim type
            default: Default value if not found

        Returns:
            Resolved value according to the precedence above.
        """
        # 1) Authored value by schema priority
        for r in self.resolvers:
            got = r.get_value(prim, prim_type, key)
            if got is not None:
                val, _usd_attr = got
                if val is not None:
                    if self._collect_schema_attrs:
                        self._collect_on_first_use(r, prim)
                    return val

        # 2) Caller-provided default, if any
        if default is not None:
            return default

        # 3) Resolver mapping defaults in priority order
        for resolver in self.resolvers:
            spec = resolver.mapping.get(prim_type, {}).get(key) if hasattr(resolver, "mapping") else None
            if spec is not None:
                d = getattr(spec, "default", None)
                if d is not None:
                    transformer = getattr(spec, "transformer", None)
                    return transformer(d) if transformer is not None else d

        # Nothing found
        try:
            prim_path = str(prim.GetPath()) if prim is not None else "<None>"
        except (AttributeError, RuntimeError):
            prim_path = "<invalid>"
        print(
            f"Error: Cannot resolve value for '{prim_type.name.lower()}:{key}' on prim '{prim_path}'; "
            f"no authored value, no explicit default, and no solver mapping default."
        )
        return None

    def collect_prim_attrs(self, prim) -> None:
        """
        Collect and accumulate schema attributes for a single prim.

        Args:
            prim: USD prim to collect attributes from
        """
        if prim is None:
            return

        prim_path = str(prim.GetPath())

        if not self._collect_schema_attrs:
            return
        for resolver in self.resolvers:
            # only collect if we haven't seen this prim for this resolver
            if prim_path not in self.schema_attrs[resolver.name]:
                attrs = resolver.collect_prim_attrs(prim)
                if attrs:
                    self.schema_attrs[resolver.name][prim_path] = attrs

    def get_schema_attrs(self) -> dict[str, dict[str, dict[str, Any]]]:
        """
        Get the accumulated attributes.

        Returns:
            Dictionary with structure: solver_name -> prim_path -> {attr_name: attr_value}
            e.g., {"mjc": {"/World/Cube": {"mjc:option:timestep": 0.01}}}
        """
        return self.schema_attrs.copy()

    def get_custom_attribute_declarations(
        self,
    ) -> dict[str, CustomAttribute]:
        """
        Get custom attribute declarations parsed from PhysicsScene.

        Returns:
            Dictionary keyed by full attribute name with CustomAttribute values.
        """
        return self._custom_attributes.copy()
