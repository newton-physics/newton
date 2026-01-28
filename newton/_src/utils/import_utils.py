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
from typing import TYPE_CHECKING, Any, Literal

import warp as wp

from ..sim.joints import ActuatorMode

if TYPE_CHECKING:
    from ..sim.builder import ModelBuilder


def parse_warp_value_from_string(value: str, warp_dtype: Any, default: Any = None) -> Any:
    """
    Parse a Warp value from a string. This is useful for parsing values from XML files.
    For example, "1.0 2.0 3.0" will be parsed as wp.vec3(1.0, 2.0, 3.0).

    If fewer values are provided than expected for vector/matrix types, the remaining
    values will be filled from the default value if provided.

    Raises:
        ValueError: If the dtype is invalid.

    Args:
        value: The string value to parse.
        warp_dtype: The Warp dtype to parse the value as.
        default: Optional default value to use for padding incomplete vectors/matrices.

    Returns:
        The parsed Warp value.
    """

    def get_vector(scalar_type: Any):
        return [scalar_type(x) for x in value.split()]

    def get_bool(tok: str) -> bool:
        # just casting string to bool is not enough, we need to actually evaluate the
        # falsey values
        s = tok.strip().lower()
        if s in {"1", "true", "t", "yes", "y"}:
            return True
        if s in {"0", "false", "f", "no", "n"}:
            return False
        # fall back to numeric interpretation if provided
        try:
            return bool(int(float(s)))
        except Exception as e:
            raise ValueError(f"Unable to parse boolean value: {tok}") from e

    if wp.types.type_is_quaternion(warp_dtype):
        parsed_values = get_vector(float)
        # Pad with default values if necessary
        expected_length = 4  # Quaternions always have 4 components
        if len(parsed_values) < expected_length and default is not None:
            if hasattr(default, "__len__"):
                default_values = [default[i] for i in range(len(default))]
            else:
                default_values = [default] * expected_length
            parsed_values.extend(default_values[len(parsed_values) : expected_length])
        return warp_dtype(*parsed_values)
    if wp.types.type_is_int(warp_dtype):
        return warp_dtype(int(value))
    if wp.types.type_is_float(warp_dtype):
        return warp_dtype(float(value))
    if warp_dtype is wp.bool or warp_dtype is bool:
        return warp_dtype(get_bool(value))
    if wp.types.type_is_vector(warp_dtype) or wp.types.type_is_matrix(warp_dtype):
        scalar_type = warp_dtype._wp_scalar_type_
        parsed_values = None
        if wp.types.type_is_int(scalar_type):
            parsed_values = get_vector(int)
        elif wp.types.type_is_float(scalar_type):
            parsed_values = get_vector(float)
        elif scalar_type is wp.bool or scalar_type is bool:
            parsed_values = get_vector(bool)
        else:
            raise ValueError(f"Unable to parse vector/matrix value: {value} as {warp_dtype}.")

        # Pad with default values if necessary
        expected_length = warp_dtype._length_
        if len(parsed_values) < expected_length and default is not None:
            # Extract default values and pad
            if hasattr(default, "__len__"):
                default_values = [default[i] for i in range(len(default))]
            else:
                default_values = [default] * expected_length
            parsed_values.extend(default_values[len(parsed_values) : expected_length])

        return warp_dtype(*parsed_values)
    raise ValueError(f"Invalid dtype: {warp_dtype}. Must be a valid Warp dtype.")


def parse_custom_attributes(
    dictlike: dict[str, str],
    custom_attributes: Sequence[ModelBuilder.CustomAttribute],
    parsing_mode: Literal["usd", "mjcf", "urdf"],
) -> dict[str, Any]:
    """
    Parse custom attributes from a dictionary.

    Args:
        dictlike: The dictionary (or XML element) to parse the custom attributes from. This object behaves like a string-valued dictionary that implements the ``get`` method and returns the value for the given key.
        custom_attributes: The custom attributes to parse. This is a sequence of :class:`ModelBuilder.CustomAttribute` objects.
        parsing_mode: The parsing mode to use. This can be "usd", "mjcf", or "urdf". It determines which attribute name and value transformer to use.

    Returns:
        A dictionary of the parsed custom attributes. The keys are the custom attribute keys :attr:`ModelBuilder.CustomAttribute.key`
        and the values are the parsed values. Only attributes that were explicitly specified in the source are included
        in the output dict. Unspecified attributes are not included, allowing defaults to be filled in during model finalization.
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

            def transform(x: str, dtype: Any = attr.dtype, default: Any = attr.default) -> Any:
                return parse_warp_value_from_string(x, dtype, default)

            transformer = transform

        if name is None:
            name = attr.name
        dict_value = dictlike.get(name)
        if dict_value is not None:
            out[attr.key] = transformer(dict_value)
    return out


def sanitize_xml_content(source: str) -> str:
    # Strip leading whitespace and byte-order marks
    xml_content = source.strip()
    # Remove BOM if present
    if xml_content.startswith("\ufeff"):
        xml_content = xml_content[1:]
    # Remove leading XML comments
    while xml_content.strip().startswith("<!--"):
        end_comment = xml_content.find("-->")
        if end_comment != -1:
            xml_content = xml_content[end_comment + 3 :].strip()
        else:
            break
    xml_content = xml_content.strip()
    return xml_content


def sanitize_name(name: str) -> str:
    """Sanitize a name for use as a key in the ModelBuilder.

    Replaces characters that are invalid in USD paths (e.g., "-") with underscores.

    Args:
        name: The name to sanitize.

    Returns:
        The sanitized name.
    """
    return name.replace("-", "_")


def is_xml_content(source: str) -> bool:
    """Check if a string appears to be XML content rather than a file path.

    Uses the presence of XML angle brackets which are required for any XML
    content and practically never appear in file paths.

    Args:
        source: String to check

    Returns:
        True if the string appears to be XML content, False if it looks like a file path
    """
    return any(char in source for char in "<>")


def infer_actuator_mode(
    target_ke: float,
    target_kd: float,
    force_position_velocity: bool = False,
    has_drive: bool = False,
) -> ActuatorMode:
    """Infer actuator mode from position and velocity gains.

    Args:
        target_ke: Position gain (stiffness).
        target_kd: Velocity gain (damping).
        force_position_velocity: If True and both gains are non-zero,
            forces POSITION_VELOCITY mode instead of just POSITION.
        has_drive: If True, a drive/actuator is applied to the joint.
            When True but both gains are 0, returns EFFORT mode.
            When False, returns NONE regardless of gains.

    Returns:
        The inferred ActuatorMode based on which gains are non-zero:
        - NONE: No drive applied
        - EFFORT: Drive applied but both gains are 0 (direct torque control)
        - POSITION: Only position gain is non-zero
        - VELOCITY: Only velocity gain is non-zero
        - POSITION_VELOCITY: Both gains non-zero (or forced)
    """
    if not has_drive:
        return ActuatorMode.NONE

    if force_position_velocity and (target_ke != 0.0 and target_kd != 0.0):
        return ActuatorMode.POSITION_VELOCITY
    elif target_ke != 0.0:
        return ActuatorMode.POSITION
    elif target_kd != 0.0:
        return ActuatorMode.VELOCITY
    else:
        return ActuatorMode.EFFORT
