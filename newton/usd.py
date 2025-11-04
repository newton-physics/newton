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

"""
Utilities for working with the Universal Scene Description (USD) format.
"""

# ==================================================================================
# USD utility functions
# ==================================================================================
from ._src.usd.utils import (
    convert_warp_type,
    convert_warp_value,
    get_attribute,
    get_attributes_in_namespace,
    get_axis,
    get_custom_attribute_declarations,
    get_custom_attribute_values,
    get_float,
    get_quat,
    get_scale,
    get_transform,
    has_attribute,
)

__all__ = [
    "convert_warp_type",
    "convert_warp_type",
    "convert_warp_value",
    "get_attribute",
    "get_attributes_in_namespace",
    "get_axis",
    "get_custom_attribute_declarations",
    "get_custom_attribute_values",
    "get_float",
    "get_quat",
    "get_scale",
    "get_transform",
    "has_attribute",
]


# ==================================================================================
# USD schema resolution
# ==================================================================================

from ._src.usd.schema_resolver import (
    PrimType,
    SchemaResolver,
    SchemaResolverManager,
)
from ._src.usd.schemas import (
    SchemaResolverMjc,
    SchemaResolverNewton,
    SchemaResolverPhysx,
)

__all__ += [
    "PrimType",
    "SchemaResolver",
    "SchemaResolverManager",
    "SchemaResolverMjc",
    "SchemaResolverNewton",
    "SchemaResolverPhysx",
]
