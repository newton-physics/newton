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

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, overload

import numpy as np
import warp as wp

from ..core.types import Axis, AxisType, nparray
from ..sim.model import ModelAttributeAssignment, ModelAttributeFrequency

if TYPE_CHECKING:
    from ..sim.builder import ModelBuilder

try:
    from pxr import Gf, Usd, UsdGeom
except ImportError:
    Usd = None
    Gf = None
    UsdGeom = None


def get_attribute(prim: Usd.Prim, name: str, default: Any | None = None) -> Any | None:
    """
    Get an attribute value from a USD prim, returning a default if not found.

    Args:
        prim: The USD prim to query.
        name: The name of the attribute to retrieve.
        default: The default value to return if the attribute is not found or invalid.

    Returns:
        The attribute value if it exists and is valid, otherwise the default value.
    """
    attr = prim.GetAttribute(name)
    if not attr or not attr.HasAuthoredValue():
        return default
    return attr.Get()


def get_attributes_in_namespace(prim: Usd.Prim, namespace: str) -> dict[str, Any]:
    """
    Get all attributes in a namespace from a USD prim.

    Args:
        prim: The USD prim to query.
        namespace: The namespace to query.

    Returns:
        A dictionary of attributes in the namespace mapping from attribute name to value.
    """
    out: dict[str, Any] = {}
    for attr in prim.GetAuthoredPropertiesInNamespace(namespace):
        if attr.IsValid() and attr.HasAuthoredValue():
            out[attr.GetName()] = attr.Get()
    return out


def has_attribute(prim: Usd.Prim, name: str) -> bool:
    """
    Check if a USD prim has a valid and authored attribute.

    Args:
        prim: The USD prim to query.
        name: The name of the attribute to check.

    Returns:
        True if the attribute exists, is valid, and has an authored value, False otherwise.
    """
    attr = prim.GetAttribute(name)
    return attr and attr.HasAuthoredValue()


@overload
def get_float(prim: Usd.Prim, name: str, default: float) -> float: ...


@overload
def get_float(prim: Usd.Prim, name: str, default: None = None) -> float | None: ...


def get_float(prim: Usd.Prim, name: str, default: float | None = None) -> float | None:
    """
    Get a float attribute value from a USD prim, validating that it's finite.

    Args:
        prim: The USD prim to query.
        name: The name of the float attribute to retrieve.
        default: The default value to return if the attribute is not found or is not finite.

    Returns:
        The float attribute value if it exists and is finite, otherwise the default value.
    """
    attr = prim.GetAttribute(name)
    if not attr or not attr.HasAuthoredValue():
        return default
    val = attr.Get()
    if np.isfinite(val):
        return val
    return default


def get_float_with_fallback(prims: Iterable[Usd.Prim], name: str, default: float = 0.0) -> float:
    """
    Get a float attribute value from the first prim in a list that has it defined.

    Args:
        prims: An iterable of USD prims to query in order.
        name: The name of the float attribute to retrieve.
        default: The default value to return if no prim has the attribute.

    Returns:
        The float attribute value from the first prim that has a finite value,
        otherwise the default value.
    """
    ret = default
    for prim in prims:
        if not prim:
            continue
        attr = prim.GetAttribute(name)
        if not attr or not attr.HasAuthoredValue():
            continue
        val = attr.Get()
        if np.isfinite(val):
            ret = val
            break
    return ret


def from_gfquat(gfquat: Gf.Quat) -> wp.quat:
    """
    Convert a USD Gf.Quat to a normalized Warp quaternion.

    Args:
        gfquat: A USD Gf.Quat quaternion.

    Returns:
        A normalized Warp quaternion.
    """
    return wp.normalize(wp.quat(*gfquat.imaginary, gfquat.real))


@overload
def get_quat(prim: Usd.Prim, name: str, default: wp.quat) -> wp.quat: ...


@overload
def get_quat(prim: Usd.Prim, name: str, default: None = None) -> wp.quat | None: ...


def get_quat(prim: Usd.Prim, name: str, default: wp.quat | None = None) -> wp.quat | None:
    """
    Get a quaternion attribute value from a USD prim, validating that it's finite and non-zero.

    Args:
        prim: The USD prim to query.
        name: The name of the quaternion attribute to retrieve.
        default: The default value to return if the attribute is not found or invalid.

    Returns:
        The quaternion attribute value as a Warp quaternion if it exists and is valid,
        otherwise the default value.
    """
    attr = prim.GetAttribute(name)
    if not attr or not attr.HasAuthoredValue():
        return default
    val = attr.Get()
    quat = from_gfquat(val)
    l = wp.length(quat)
    if np.isfinite(l) and l > 0.0:
        return quat
    return default


@overload
def get_vector(prim: Usd.Prim, name: str, default: nparray) -> nparray: ...


@overload
def get_vector(prim: Usd.Prim, name: str, default: None = None) -> nparray | None: ...


def get_vector(prim: Usd.Prim, name: str, default: nparray | None = None) -> nparray | None:
    """
    Get a vector attribute value from a USD prim, validating that all components are finite.

    Args:
        prim: The USD prim to query.
        name: The name of the vector attribute to retrieve.
        default: The default value to return if the attribute is not found or has non-finite values.

    Returns:
        The vector attribute value as a numpy array with dtype float32 if it exists and
        all components are finite, otherwise the default value.
    """
    attr = prim.GetAttribute(name)
    if not attr or not attr.HasAuthoredValue():
        return default
    val = attr.Get()
    if np.isfinite(val).all():
        return np.array(val, dtype=np.float32)
    return default


def get_scale(prim: Usd.Prim) -> wp.vec3:
    """
    Extract the scale component from a USD prim's local transformation.

    Args:
        prim: The USD prim to query for scale information.

    Returns:
        The scale as a Warp vec3.
    """
    # first get local transform matrix
    local_mat = np.array(UsdGeom.Xform(prim).GetLocalTransformation(), dtype=np.float32)
    # then get scale from the matrix
    scale = np.sqrt(np.sum(local_mat[:3, :3] ** 2, axis=0))
    return wp.vec3(*scale)


def get_gprim_axis(prim: Usd.Prim, name: str = "axis", default: AxisType = "Z") -> Axis:
    """
    Get an axis attribute from a USD prim and convert it to an :class:`~newton.Axis` enum.

    Args:
        prim: The USD prim to query.
        name: The name of the axis attribute to retrieve.
        default: The default axis string to use if the attribute is not found.

    Returns:
        An :class:`~newton.Axis` enum value converted from the attribute string.
    """
    axis_str = get_attribute(prim, name, default)
    return Axis.from_string(axis_str)


def get_transform(prim: Usd.Prim, local: bool = True, invert_rotation: bool = True) -> wp.transform:
    """
    Extract the transform (position and rotation) from a USD Xform prim.

    Args:
        prim: The USD prim to query.
        local: If True, get the local transformation; if False, get the world transformation.
        invert_rotation: If True, transpose the rotation matrix before converting to quaternion.

    Returns:
        A Warp transform containing the position and rotation extracted from the prim.
    """
    xform = UsdGeom.Xform(prim)
    if local:
        mat = np.array(xform.GetLocalTransformation(), dtype=np.float32)
    else:
        mat = np.array(xform.GetWorldTransformation(), dtype=np.float32)
    if invert_rotation:
        rot = wp.quat_from_matrix(wp.mat33(mat[:3, :3].T.flatten()))
    else:
        rot = wp.quat_from_matrix(wp.mat33(mat[:3, :3].flatten()))
    pos = mat[3, :3]
    return wp.transform(pos, rot)


def convert_warp_value(v: Any, warp_dtype: Any | None = None) -> Any:
    """
    Convert a USD value (such as Gf.Quat, Gf.Vec3, or float) to a Warp value.
    If a dtype is given, the value will be converted to that dtype.
    Otherwise, the value will be converted to the most appropriate Warp dtype.

    Args:
        v: The value to convert.
        warp_dtype: The Warp dtype to convert to. If None, the value will be converted to the most appropriate Warp dtype.

    Returns:
        The converted value.
    """
    if warp_dtype is wp.quat or (hasattr(v, "real") and hasattr(v, "imaginary")):
        return from_gfquat(v)
    if warp_dtype is not None:
        # assume the type is a vector, matrix, or scalar
        if hasattr(v, "__len__"):
            return warp_dtype(*v)
        else:
            return warp_dtype(v)
    # without a given Warp dtype, we attempt to infer the dtype from the value
    if hasattr(v, "__len__"):
        if len(v) == 2:
            return wp.vec2(*v)
        if len(v) == 3:
            return wp.vec3(*v)
        if len(v) == 4:
            return wp.vec4(*v)
    # the value is a scalar or we weren't able to resolve the dtype
    return v


def convert_warp_type(v: Any) -> Any:
    """
    Determine the Warp type, e.g. wp.quat, wp.vec3, or wp.float32, from a USD value.

    Args:
        v: The USD value from which to infer the Warp type.

    Returns:
        The Warp type.
    """
    try:
        # Check for quat first (before generic length checks)
        if hasattr(v, "real") and hasattr(v, "imaginary"):
            return wp.quat
        # Vector3-like
        if hasattr(v, "__len__") and len(v) == 3:
            return wp.vec3
        # Vector2-like
        if hasattr(v, "__len__") and len(v) == 2:
            return wp.vec2
        # Vector4-like (but not quat)
        if hasattr(v, "__len__") and len(v) == 4:
            return wp.vec4
    except (TypeError, AttributeError):
        # fallthrough to scalar checks
        pass
    if isinstance(v, bool):
        return wp.bool
    if isinstance(v, int):
        return wp.int32
    # default to float32 for scalars
    return wp.float32


def get_custom_attribute_declarations(prim: Usd.Prim) -> dict[str, ModelBuilder.CustomAttribute]:
    """
    Get custom attribute declarations from a USD prim, typically from a ``PhysicsScene`` prim.

    Supports metadata format with assignment and frequency specified as ``customData``:

    .. code-block:: usda

        custom float newton:namespace:attr_name = 150.0 (
            customData = {
                string assignment = "control"
                string frequency = "joint_dof"
            }
        )

    Args:
        prim: USD ``PhysicsScene`` prim to parse declarations from.

    Returns:
        A dictionary of custom attribute declarations mapping from attribute name to :class:`ModelBuilder.CustomAttribute` object.
    """
    from ..sim.builder import ModelBuilder  # noqa: PLC0415

    def parse_custom_attr_name(name: str) -> tuple[str | None, str] | None:
        """
        Parse custom attribute names in the format 'newton:namespace:attr_name' or 'newton:attr_name'.

        Returns:
            Tuple of (namespace, attr_name) where namespace can be None for default namespace,
            or None if the name doesn't match the expected format.
        """

        parts = name.split(":")
        if len(parts) < 2 or parts[0] != "newton":
            return None

        if len(parts) == 2:
            # newton:attr_name (default namespace)
            return None, parts[1]
        elif len(parts) == 3:
            # newton:namespace:attr_name
            return parts[1], parts[2]
        else:
            # Invalid format
            return None

    out: dict[str, ModelBuilder.CustomAttribute] = {}
    for attr in prim.GetAttributes():
        attr_name = attr.GetName()
        parsed = parse_custom_attr_name(attr_name)
        if not parsed:
            continue

        namespace, local_name = parsed
        default_value = attr.Get()

        # Try to read customData for assignment and frequency
        assignment_meta = attr.GetCustomDataByKey("assignment")
        frequency_meta = attr.GetCustomDataByKey("frequency")

        if assignment_meta and frequency_meta:
            # Metadata format
            try:
                assignment_val = ModelAttributeAssignment[assignment_meta.upper()]
                frequency_val = ModelAttributeFrequency[frequency_meta.upper()]
            except KeyError:
                print(
                    f"Warning: Custom attribute '{attr_name}' has invalid assignment or frequency in customData. Skipping."
                )
                continue
        else:
            # No metadata found - skip with warning
            print(
                f"Warning: Custom attribute '{attr_name}' is missing required customData (assignment and frequency). Skipping."
            )
            continue

        # Infer dtype from default value
        converted_value = convert_warp_value(default_value)
        dtype = convert_warp_type(default_value)

        # Create custom attribute specification
        # Note: name should be the local name, namespace is stored separately
        custom_attr = ModelBuilder.CustomAttribute(
            assignment=assignment_val,
            frequency=frequency_val,
            name=local_name,
            dtype=dtype,
            default=converted_value,
            namespace=namespace,
        )

        out[custom_attr.key] = custom_attr

    return out


def get_custom_attribute_values(
    prim: Usd.Prim, custom_attributes: Sequence[ModelBuilder.CustomAttribute]
) -> dict[str, Any]:
    """
    Get custom attribute values from a USD prim and a set of known custom attributes.
    Returns a dictionary mapping from :attr:`ModelBuilder.CustomAttribute.key` to the converted Warp value.
    The conversion is performed by :meth:`ModelBuilder.CustomAttribute.usd_value_transformer`.

    Args:
        prim: The USD prim to query.
        custom_attributes: The custom attributes to get values for.

    Returns:
        A dictionary of found custom attribute values mapping from attribute name to value.
    """
    out: dict[str, Any] = {}
    for attr in custom_attributes:
        usd_attr_name = attr.usd_attribute_name
        usd_attr = prim.GetAttribute(usd_attr_name)
        if usd_attr is not None and usd_attr.HasAuthoredValue():
            if attr.usd_value_transformer is not None:
                out[attr.key] = attr.usd_value_transformer(usd_attr.Get())
            else:
                out[attr.key] = convert_warp_value(usd_attr.Get(), attr.dtype)
    return out
