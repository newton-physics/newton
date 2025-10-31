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

from collections.abc import Sequence
from typing import Any, Literal

import warp as wp

from ..sim.model import CustomAttribute


def parse_warp_value_from_string(value: str, warp_dtype: Any) -> Any:
    """
    Parse a Warp value from a string. This is useful for parsing values from XML files.
    For example, "1.0 2.0 3.0" will be parsed as wp.vec3(1.0, 2.0, 3.0).

    Raises:
        ValueError: If the dtype is invalid.

    Args:
        value: The string value to parse.
        warp_dtype: The Warp dtype to parse the value as.

    Returns:
        The parsed Warp value.
    """

    def get_vector(scalar_type: Any):
        return [scalar_type(x) for x in value.split()]

    if wp.types.type_is_quaternion(warp_dtype):
        return warp_dtype(*get_vector(float))
    if wp.types.type_is_int(warp_dtype):
        return warp_dtype(int(value))
    if wp.types.type_is_float(warp_dtype):
        return warp_dtype(float(value))
    if warp_dtype is wp.bool or warp_dtype is bool:
        return warp_dtype(bool(value))
    if wp.types.type_is_vector(warp_dtype) or wp.types.type_is_matrix(warp_dtype):
        scalar_type = warp_dtype._wp_scalar_type_
        if wp.types.type_is_int(scalar_type):
            return warp_dtype(*get_vector(int))
        if wp.types.type_is_float(scalar_type):
            return warp_dtype(*get_vector(float))
        if scalar_type is wp.bool or scalar_type is bool:
            return warp_dtype(*get_vector(bool))
        raise ValueError(f"Unable to parse vector/matrix value: {value} as {warp_dtype}.")
    raise ValueError(f"Invalid dtype: {warp_dtype}. Must be a valid Warp dtype.")


def parse_custom_attributes(
    dictlike: dict[str, str],
    custom_attributes: Sequence[CustomAttribute],
    parsing_mode: Literal["usd", "mjcf", "urdf"] = "usd",
) -> dict[str, Any]:
    """
    Parse custom attributes from a dictionary.

    Args:
        dictlike: The dictionary (or XML element) to parse the custom attributes from. This object behaves like a string-valued dictionary that implements the ``get`` method and returns the value for the given key.
        custom_attributes: The custom attributes to parse. This is a sequence of :class:`CustomAttribute` objects.
        parsing_mode: The parsing mode to use. This can be "usd", "mjcf", or "urdf". It determines which attribute name and value transformer to use.

    Returns:
        A dictionary of the parsed custom attributes. The keys are the custom attribute keys :attr:`CustomAttribute.key` and the values are the parsed values.
    """
    out = {}
    for attr in custom_attributes:
        transformer = None
        name = None
        if parsing_mode == "mjcf":
            name = attr.mjcf_attribute_name
            transformer = attr.mjcf_value_transformer
        elif parsing_mode == "urdf":
            name = attr.urdf_attribute_name
            transformer = attr.urdf_value_transformer
        elif parsing_mode == "usd":
            name = attr.usd_attribute_name
            transformer = attr.usd_value_transformer
        if transformer is None:

            def transform(x: str, dtype: Any = attr.dtype) -> Any:
                return parse_warp_value_from_string(x, dtype)

            transformer = transform

        if name is None:
            name = attr.name
        dict_value = dictlike.get(name)
        if dict_value is not None:
            out[attr.key] = transformer(dict_value)
    return out
