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
        name (str): The name of the USD attribute.
        default (Any | None): Default USD-authored value from schema, if any.
        usd_value_transformer (Callable[[Any], Any] | None): A function to transform the raw USD attribute value
            into the format expected by Newton. Takes the USD value as input and returns the transformed
            value. For example, converting PhysX timeStepsPerSecond to Newton's timestep by computing 1/Hz.
    """

    name: str
    default: Any | None = None
    usd_value_transformer: Callable[[Any], Any] | None = None


class SchemaResolver:
    """
    Defines a mapping from USD schema attributes to Newton attributes.
    """

    # mapping is a dictionary for known variables in Newton. Its purpose is to map USD attributes to existing Newton data.
    # PrimType -> Newton variable -> Attribute
    mapping: ClassVar[dict[PrimType, dict[str, SchemaAttribute]]]

    # Name of the schema resolver
    name: ClassVar[str]

    # extra_attr_namespaces is a list of additional USD attribute namespaces in which the schema attributes may be authored.
    extra_attr_namespaces: ClassVar[list[str]] = []

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
                names.add(spec.name)
        self._solver_attributes: list[str] = list(names)

    def get_value(self, prim: Usd.Prim, prim_type: PrimType, key: str) -> Any | None:
        """
        Get attribute value for a given prim type and key.

        Args:
            prim: USD prim to query
            prim_type: Prim type (PrimType enum)
            key: Attribute key within the prim type

        Returns:
            Value if found, None otherwise
        """
        if prim is None:
            return None
        spec = self.mapping.get(prim_type, {}).get(key)
        if spec is not None:
            v = usd.get_attribute(prim, spec.name)
            if v is not None:
                return spec.usd_value_transformer(v) if spec.usd_value_transformer is not None else v
        return None

    def collect_prim_attrs(self, prim: Usd.Prim) -> dict[str, Any]:
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


def _collect_attrs_by_name(prim: Usd.Prim, names: list[str]) -> dict[str, Any]:
    """Collect attributes authored on the prim that have direct mappings in the resolver mapping"""
    out: dict[str, Any] = {}
    for n in names:
        v = usd.get_attribute(prim, n)
        if v is not None:
            out[n] = v
    return out


def _collect_attrs_by_namespace(prim: Usd.Prim, namespaces: list[str]) -> dict[str, Any]:
    """Collect authored attributes using USD namespace queries."""
    out: dict[str, Any] = {}
    if prim is None:
        return out
    for ns in namespaces:
        out.update(usd.get_attributes_in_namespace(prim, ns))
    return out


class SchemaResolverManager:
    """
    Manager for resolving multiple USD schemas in a priority order.
    """

    def __init__(self, resolvers: list[SchemaResolver]):
        """
        Initialize resolver manager with resolver instances in priority order.

        Args:
            resolvers: List of instantiated resolvers in priority order.
        """
        # Use provided resolver instances directly
        self.resolvers = list(resolvers)

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

    def get_value(self, prim: Usd.Prim, prim_type: PrimType, key: str, default: Any = None) -> Any:
        """
        Resolve value using schema priority, with layered fallbacks:

        1) First authored value found in resolver order (highest priority first)
        2) If none authored, use the provided 'default' argument if not None
        3) If no default provided, use the first non-None mapping default from resolvers in priority order
        4) If no mapping default found, return None

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
            val = r.get_value(prim, prim_type, key)
            if val is None:
                continue
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
                    transformer = getattr(spec, "usd_value_transformer", None)
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
