# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Utilities for working with the Universal Scene Description (USD) format.

This module provides both low-level USD utility helpers and public schema
resolver types used by :meth:`newton.ModelBuilder.add_usd`.
"""

# ==================================================================================
# USD utility functions
# ==================================================================================
from ._src.usd.utils import (
    find_tetmesh_prims,
    get_attribute,
    get_attributes_in_namespace,
    get_custom_attribute_declarations,
    get_custom_attribute_values,
    get_float,
    get_gprim_axis,
    get_mesh,
    get_quat,
    get_scale,
    get_tetmesh,
    get_transform,
    has_applied_api_schema,
    has_attribute,
    type_to_warp,
    value_to_warp,
)

__all__ = [
    "find_tetmesh_prims",
    "get_attribute",
    "get_attributes_in_namespace",
    "get_custom_attribute_declarations",
    "get_custom_attribute_values",
    "get_float",
    "get_gprim_axis",
    "get_mesh",
    "get_quat",
    "get_scale",
    "get_tetmesh",
    "get_transform",
    "has_applied_api_schema",
    "has_attribute",
    "type_to_warp",
    "value_to_warp",
]


# ==================================================================================
# USD schema resolution
# ==================================================================================

from ._src.usd.schema_resolver import (
    PrimType,
    SchemaResolver,
)
from ._src.usd.schemas import (
    SchemaResolverMjc,
    SchemaResolverNewton,
    SchemaResolverPhysx,
)

__all__ += [
    "PrimType",
    "SchemaResolver",
    "SchemaResolverMjc",
    "SchemaResolverNewton",
    "SchemaResolverPhysx",
]
