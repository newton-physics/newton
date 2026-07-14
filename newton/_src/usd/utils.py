# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
import os
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import warp as wp

from ..core.types import Axis, AxisType
from ..geometry import Gaussian, Mesh
from ..sim.model import Model
from ..utils.color import color_linear_to_srgb
from ..utils.import_usd_deformable_utils import _validate_mass_array, _warn_geometry_authored_material_attrs
from ..utils.texture import linear_texture_to_srgb, load_texture

logger = logging.getLogger("newton")

AttributeAssignment = Model.AttributeAssignment
AttributeFrequency = Model.AttributeFrequency

if TYPE_CHECKING:
    from ..geometry.types import TetMesh
    from ..sim.builder import ModelBuilder

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
except ImportError:
    Usd = None
    Gf = None
    UsdGeom = None
    Sdf = None
    UsdShade = None


@overload
def get_attribute(prim: Usd.Prim, name: str, default: None = None) -> Any | None: ...


@overload
def get_attribute(prim: Usd.Prim, name: str, default: Any) -> Any: ...


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
    for prop in prim.GetAuthoredPropertiesInNamespace(namespace):
        if not prop.IsValid():
            continue
        if hasattr(prop, "GetTargets"):
            continue
        if hasattr(prop, "HasAuthoredValue") and prop.HasAuthoredValue():
            out[prop.GetName()] = prop.Get()
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


def _get_raw_api_schemas(prim: Usd.Prim) -> list[str]:
    """Return API schema tokens from raw ``apiSchemas`` list-op metadata."""
    listop = prim.GetMetadata("apiSchemas")
    if listop is None:
        return []
    return (
        list(getattr(listop, "prependedItems", []))
        + list(getattr(listop, "appendedItems", []))
        + list(getattr(listop, "explicitItems", []))
    )


def get_applied_api_schemas(prim: Usd.Prim) -> list[str]:
    """Return the API schema tokens applied to *prim*.

    Falls back to raw ``apiSchemas`` list-op metadata when the schema plugin
    is not loaded and :meth:`pxr.Usd.Prim.GetAppliedSchemas` returns nothing.

    Args:
        prim: Prim to query.

    Returns:
        Applied API schema tokens (e.g. ``["NewtonPDControlAPI"]``).
    """
    schemas = list(prim.GetAppliedSchemas())
    return schemas if schemas else _get_raw_api_schemas(prim)


def has_applied_api_schema(prim: Usd.Prim, schema_name: str) -> bool:
    """
    Check if a USD prim has an applied API schema, even if the schema is not
    registered with USD's schema registry.

    For registered schemas (e.g. ``UsdPhysics.RigidBodyAPI``), ``prim.HasAPI()``
    is sufficient. However, non-core schemas that may be in draft state or not
    yet registered (e.g. MuJoCo-specific schemas like ``MjcSiteAPI``) will not
    be found by ``HasAPI()``. This helper falls back to inspecting the raw
    ``apiSchemas`` metadata on the prim.

    Args:
        prim: The USD prim to query.
        schema_name: The API schema name to check for (e.g. ``"MjcSiteAPI"``).

    Returns:
        True if the schema is applied to the prim, False otherwise.
    """
    return prim.HasAPI(schema_name) or schema_name in _get_raw_api_schemas(prim)


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
    quat = value_to_warp(val)
    l = wp.length(quat)
    if np.isfinite(l) and l > 0.0:
        return quat
    return default


@overload
def get_vector(prim: Usd.Prim, name: str, default: np.ndarray) -> np.ndarray: ...


@overload
def get_vector(prim: Usd.Prim, name: str, default: None = None) -> np.ndarray | None: ...


def get_vector(prim: Usd.Prim, name: str, default: np.ndarray | None = None) -> np.ndarray | None:
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


def _get_xform_matrix(
    prim: Usd.Prim,
    local: bool = True,
    xform_cache: UsdGeom.XformCache | None = None,
) -> np.ndarray:
    """
    Get the transformation matrix for a USD prim.

    Args:
        prim: The USD prim to query.
        local: If True, get the local transformation; if False, get the world transformation.
        xform_cache: Optional USD XformCache to reuse when computing world transforms (only used if ``local`` is False).

    Returns:
        The transformation matrix as a numpy array (float32).
    """
    xform = UsdGeom.Xformable(prim)
    if local:
        mat = xform.GetLocalTransformation()
        # USD may return (matrix, resetXformStack)
        if isinstance(mat, tuple):
            mat = mat[0]
    else:
        if xform_cache is None:
            time = Usd.TimeCode.Default()
            mat = xform.ComputeLocalToWorldTransform(time)
        else:
            mat = xform_cache.GetLocalToWorldTransform(prim)
    return np.array(mat, dtype=np.float32)


def get_scale(prim: Usd.Prim, local: bool = True, xform_cache: UsdGeom.XformCache | None = None) -> wp.vec3:
    """
    Extract the scale component from a USD prim's transformation.

    Args:
        prim: The USD prim to query for scale information.
        local: If True, get the local scale; if False, get the world scale.
        xform_cache: Optional USD XformCache to reuse when computing world transforms (only used if ``local`` is False).

    Returns:
        The scale as a Warp vec3.
    """
    mat = get_transform_matrix(prim, local=local, xform_cache=xform_cache)
    _pos, _rot, scale = wp.transform_decompose(mat)
    scale = np.array(scale, dtype=np.float32)

    authored_scale = _get_authored_scale(prim, local=local)
    if authored_scale is not None:
        sign = np.sign(authored_scale)
        sign[sign == 0.0] = 1.0
        scale = np.abs(scale) * sign

    return wp.vec3(*scale)


def _get_authored_scale(prim: Usd.Prim, local: bool = True) -> np.ndarray | None:
    """Return the product of authored scale ops, preserving negative signs."""
    if UsdGeom is None:
        return None

    prims = [prim]
    if not local:
        prims = []
        current = prim
        while current and current.IsValid() and not current.IsPseudoRoot():
            prims.append(current)
            current = current.GetParent()
        prims.reverse()

    scale = np.ones(3, dtype=np.float32)
    found = False
    for p in prims:
        xformable = UsdGeom.Xformable(p)
        if not xformable:
            continue
        if not local and xformable.GetResetXformStack():
            scale = np.ones(3, dtype=np.float32)
            found = False
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() != UsdGeom.XformOp.TypeScale:
                continue
            value = op.Get()
            if value is None:
                continue
            op_scale = np.array(value, dtype=np.float32)
            if op.IsInverseOp():
                with np.errstate(divide="ignore", invalid="ignore"):
                    op_scale = np.divide(1.0, op_scale, out=np.ones_like(op_scale), where=op_scale != 0.0)
            scale *= op_scale
            found = True

    return scale if found else None


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


def get_transform_matrix(prim: Usd.Prim, local: bool = True, xform_cache: UsdGeom.XformCache | None = None) -> wp.mat44:
    """
    Extract the full transformation matrix from a USD Xform prim.

    Args:
        prim: The USD prim to query.
        local: If True, get the local transformation; if False, get the world transformation.
        xform_cache: Optional USD XformCache to reuse when computing world transforms (only used if ``local`` is False).

    Returns:
        A Warp 4x4 transform matrix. This representation composes left-to-right with `@`, matching
        `wp.transform_decompose` expectations.
    """
    mat = _get_xform_matrix(prim, local=local, xform_cache=xform_cache)
    return wp.mat44(mat.T)


def get_transform(prim: Usd.Prim, local: bool = True, xform_cache: UsdGeom.XformCache | None = None) -> wp.transform:
    """
    Extract the transform (position and rotation) from a USD Xform prim.

    Args:
        prim: The USD prim to query.
        local: If True, get the local transformation; if False, get the world transformation.
        xform_cache: Optional USD XformCache to reuse when computing world transforms (only used if ``local`` is False).

    Returns:
        A Warp transform containing the position and rotation extracted from the prim.
    """
    mat = _get_xform_matrix(prim, local=local, xform_cache=xform_cache)
    xform_pos, xform_rot, _scale = wp.transform_decompose(wp.mat44(mat.T))
    return wp.transform(xform_pos, xform_rot)


def value_to_warp(v: Any, warp_dtype: Any | None = None) -> Any:
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
        return wp.normalize(wp.quat(*v.imaginary, v.real))
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


def type_to_warp(v: Any) -> Any:
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

    def is_schema_attribute(prim, attr_name: str) -> bool:
        """Check if attribute is defined by a registered schema."""
        # Check the prim's type schema
        prim_def = Usd.SchemaRegistry().FindConcretePrimDefinition(prim.GetTypeName())
        if prim_def and attr_name in prim_def.GetPropertyNames():
            return True

        # Check all applied API schemas
        for schema_name in prim.GetAppliedSchemas():
            api_def = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(schema_name)
            if api_def and attr_name in api_def.GetPropertyNames():
                return True

        # TODO: handle multi-apply schemas once newton-usd-schemas has support for them

        return False

    def parse_custom_attr_name(name: str) -> tuple[str | None, str | None]:
        """
        Parse custom attribute names in the format 'newton:namespace:attr_name' or 'newton:attr_name'.

        Returns:
            Tuple of (namespace, attr_name) where namespace can be None for default namespace,
            and attr_name can be None if the name is invalid.
        """

        parts = name.split(":")
        if len(parts) == 2:
            # newton:attr_name (default namespace)
            return None, parts[1]
        elif len(parts) == 3:
            # newton:namespace:attr_name
            return parts[1], parts[2]
        else:
            # Invalid format
            return None, None

    out: dict[str, ModelBuilder.CustomAttribute] = {}
    for attr in prim.GetAuthoredPropertiesInNamespace("newton"):
        if is_schema_attribute(prim, attr.GetName()):
            continue
        attr_name = attr.GetName()
        namespace, local_name = parse_custom_attr_name(attr_name)
        if not local_name:
            continue

        default_value = attr.Get()

        # Try to read customData for assignment and frequency
        assignment_meta = attr.GetCustomDataByKey("assignment")
        frequency_meta = attr.GetCustomDataByKey("frequency")

        if assignment_meta and frequency_meta:
            # Metadata format
            try:
                assignment_val = AttributeAssignment[assignment_meta.upper()]
                frequency_val = AttributeFrequency[frequency_meta.upper()]
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
        converted_value = value_to_warp(default_value)
        dtype = type_to_warp(default_value)

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
    prim: Usd.Prim,
    custom_attributes: Sequence[ModelBuilder.CustomAttribute],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Get custom attribute values from a USD prim and a set of known custom attributes.
    Returns a dictionary mapping from :attr:`~newton.ModelBuilder.CustomAttribute.key` to the converted Warp value.
    The conversion is performed by the ``CustomAttribute.usd_value_transformer`` callable.

    The context dictionary passed to the transformer function always contains:
    - ``"prim"``: The USD prim to query.
    - ``"attr"``: The :class:`~newton.ModelBuilder.CustomAttribute` object to get the value for.
    It may additionally include caller-provided keys from the ``context`` argument.

    Args:
        prim: The USD prim to query.
        custom_attributes: The :class:`~newton.ModelBuilder.CustomAttribute` objects to get values for.
        context: Optional extra context keys to forward to transformers.

    Returns:
        A dictionary of found custom attribute values mapping from attribute name to value.
    """
    out: dict[str, Any] = {}
    for attr in custom_attributes:
        transformer_context: dict[str, Any] = {}
        if context:
            transformer_context.update(context)
        # Keep builtin keys authoritative even if caller passes same names.
        transformer_context["prim"] = prim
        transformer_context["attr"] = attr
        usd_attr_name = attr.usd_attribute_name
        if usd_attr_name == "*":
            # Just apply the transformer to all prims of this frequency
            if attr.usd_value_transformer is not None:
                value = attr.usd_value_transformer(None, transformer_context)
                if value is None:
                    # Treat None as "undefined" to allow defaults to be applied later.
                    continue
                out[attr.key] = value
            continue
        usd_attr = prim.GetAttribute(usd_attr_name)
        if usd_attr is not None and usd_attr.HasAuthoredValue():
            if attr.usd_value_transformer is not None:
                value = attr.usd_value_transformer(usd_attr.Get(), transformer_context)
                if value is None:
                    # Treat None as "undefined" to allow defaults to be applied later.
                    continue
                out[attr.key] = value
            else:
                out[attr.key] = value_to_warp(usd_attr.Get(), attr.dtype)
    return out


def _get_tetmesh_custom_attribute_values(
    prim: Usd.Prim,
    custom_attributes: Sequence[ModelBuilder.CustomAttribute],
) -> dict[str, np.ndarray]:
    """Read builder-declared TetMesh arrays without inferring their frequency."""
    out: dict[str, np.ndarray] = {}
    for spec in custom_attributes:
        usd_name = spec.usd_attribute_name
        if not usd_name or usd_name == "*":
            continue
        usd_attr = prim.GetAttribute(usd_name)
        if not usd_attr or not usd_attr.HasAuthoredValue():
            continue
        primvar = UsdGeom.Primvar(usd_attr)
        value = primvar.ComputeFlattened() if primvar else usd_attr.Get()
        if value is None:
            continue
        if spec.usd_value_transformer is not None:
            value = spec.usd_value_transformer(value, {"prim": prim, "attr": spec})
            if value is None:
                continue
        out[spec.key] = np.asarray(value)
    return out


def _newell_normal(P: np.ndarray) -> np.ndarray:
    """Newell's method for polygon normal (not normalized)."""
    x = y = z = 0.0
    n = len(P)
    for i in range(n):
        p0 = P[i]
        p1 = P[(i + 1) % n]
        x += (p0[1] - p1[1]) * (p0[2] + p1[2])
        y += (p0[2] - p1[2]) * (p0[0] + p1[0])
        z += (p0[0] - p1[0]) * (p0[1] + p1[1])
    return np.array([x, y, z], dtype=np.float64)


def _orthonormal_basis_from_normal(n: np.ndarray):
    """Given a unit normal n, return orthonormal (tangent u, bitangent v, normal n)."""
    # Pick the largest non-collinear axis for stability
    if abs(n[2]) < 0.9:
        a = np.array([0.0, 0.0, 1.0])
    else:
        a = np.array([1.0, 0.0, 0.0])
    u = np.cross(a, n)
    nu = np.linalg.norm(u)
    if nu < 1e-20:
        # fallback (degenerate normal); pick arbitrary
        u = np.array([1.0, 0.0, 0.0])
    else:
        u /= nu
    v = np.cross(n, u)
    return u, v, n


def corner_angles(face_pos: np.ndarray) -> np.ndarray:
    """
    Compute interior corner angles (radians) for a single polygon face.

    Args:
        face_pos: (N, 3) float array
            Vertex positions of the face in winding order (CW or CCW).

    Returns:
        angles: (N,) float array
            Interior angle at each vertex in [0, pi] (radians). For degenerate
            corners/edges, the angle is set to 0.
    """
    P = np.asarray(face_pos, dtype=np.float64)
    N = len(P)
    if N < 3:
        return np.zeros((N,), dtype=np.float64)

    # Face plane via Newell
    n = _newell_normal(P)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-20:
        # Degenerate polygon (nearly collinear); fallback: use 3D formula via atan2 on cross/dot
        # after constructing tangents from edges. But simplest is to return zeros.
        return np.zeros((N,), dtype=np.float64)
    n /= n_norm

    # Local 2D frame on the plane
    u, v, _ = _orthonormal_basis_from_normal(n)

    # Project to 2D (u,v)
    # (subtract centroid for numerical stability)
    c = P.mean(axis=0)
    Q = P - c
    x = Q @ u  # (N,)
    y = Q @ v  # (N,)

    # Roll arrays to get prev/next for each vertex
    x_prev = np.roll(x, 1)
    y_prev = np.roll(y, 1)
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)

    # Edge vectors at each corner (pointing into the corner from prev/next to current)
    # a: current->prev, b: current->next (sign doesn't matter for angle magnitude)
    ax = x_prev - x
    ay = y_prev - y
    bx = x_next - x
    by = y_next - y

    # Normalize edge vectors to improve numerical stability on very different scales
    a_len = np.hypot(ax, ay)
    b_len = np.hypot(bx, by)
    valid = (a_len > 1e-30) & (b_len > 1e-30)
    ax[valid] /= a_len[valid]
    ay[valid] /= a_len[valid]
    bx[valid] /= b_len[valid]
    by[valid] /= b_len[valid]

    # Angle via atan2(||a x b||, a·b) in 2D; ||a x b|| = |ax*by - ay*bx|
    cross = ax * by - ay * bx
    dot = ax * bx + ay * by
    # Clamp dot to [-1,1] only where needed; atan2 handles it well, but clamp helps with noise
    dot = np.clip(dot, -1.0, 1.0)

    angles = np.zeros((N,), dtype=np.float64)
    angles[valid] = np.arctan2(np.abs(cross[valid]), dot[valid])  # [0, pi]

    return angles


def fan_triangulate_faces(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Perform fan triangulation on polygonal faces.

    Args:
        counts: Array of vertex counts per face
        indices: Flattened array of vertex indices

    Returns:
        Array of shape (num_triangles, 3) containing triangle indices (dtype=np.int32)
    """
    counts = np.asarray(counts, dtype=np.int32)
    indices = np.asarray(indices, dtype=np.int32)

    num_tris = int(np.sum(counts - 2))

    if num_tris == 0:
        return np.zeros((0, 3), dtype=np.int32)

    # Vectorized approach: build all triangle indices at once
    # For each face with n vertices, we create (n-2) triangles
    # Each triangle uses: [base, base+i+1, base+i+2] for i in range(n-2)

    # Array to track which face each triangle belongs to
    tri_face_ids = np.repeat(np.arange(len(counts), dtype=np.int32), counts - 2)

    # Array for triangle index within each face (0 to n-3)
    tri_local_ids = np.concatenate([np.arange(n - 2, dtype=np.int32) for n in counts])

    # Base index for each face
    face_bases = np.concatenate([[0], np.cumsum(counts[:-1], dtype=np.int32)])

    out = np.empty((num_tris, 3), dtype=np.int32)
    out[:, 0] = indices[face_bases[tri_face_ids]]  # First vertex (anchor)
    out[:, 1] = indices[face_bases[tri_face_ids] + tri_local_ids + 1]  # Second vertex
    out[:, 2] = indices[face_bases[tri_face_ids] + tri_local_ids + 2]  # Third vertex

    return out


def _expand_indexed_primvar(
    values: np.ndarray,
    indices: np.ndarray | None,
    primvar_name: str,
    prim_path: str,
) -> np.ndarray:
    """
    Expand primvar values using indices if provided.

    USD primvars can be stored in an indexed form where a compact set of unique
    values is stored along with an index array that maps each face corner (or vertex)
    to the appropriate value. This function expands such indexed primvars to their
    full form.

    Args:
        values: The primvar values array.
        indices: Optional index array for expansion.
        primvar_name: Name of the primvar (for error messages).
        prim_path: Path to the prim (for error messages).

    Returns:
        The expanded values array (same as input if no indices provided).

    Raises:
        ValueError: If indices are out of range.
    """
    if indices is None or len(indices) == 0:
        return values

    indices = np.asarray(indices, dtype=np.int64)

    # Validate indices are within range
    if indices.max() >= len(values):
        raise ValueError(
            f"{primvar_name} primvar index out of range: max index {indices.max()} >= "
            f"number of values {len(values)} for mesh {prim_path}"
        )
    if indices.min() < 0:
        raise ValueError(f"Negative {primvar_name} primvar index found: {indices.min()} for mesh {prim_path}")

    return values[indices]


def _triangulate_face_varying_indices(counts: Sequence[int], flip_winding: bool) -> np.ndarray:
    """Return flattened corner indices for fan-triangulated face-varying data."""
    counts_i32 = np.asarray(counts, dtype=np.int32)
    num_tris = int(np.sum(counts_i32 - 2))
    if num_tris <= 0:
        return np.zeros((0,), dtype=np.int32)

    tri_face_ids = np.repeat(np.arange(len(counts_i32), dtype=np.int32), counts_i32 - 2)
    tri_local_ids = np.concatenate([np.arange(n - 2, dtype=np.int32) for n in counts_i32])
    face_bases = np.concatenate([[0], np.cumsum(counts_i32[:-1], dtype=np.int32)])

    corner_faces = np.empty((num_tris, 3), dtype=np.int32)
    corner_faces[:, 0] = face_bases[tri_face_ids]
    corner_faces[:, 1] = face_bases[tri_face_ids] + tri_local_ids + 1
    corner_faces[:, 2] = face_bases[tri_face_ids] + tri_local_ids + 2
    if flip_winding:
        corner_faces = corner_faces[:, ::-1]
    return corner_faces.reshape(-1)


def _is_usd_url(source: str) -> bool:
    """Return True when ``source`` is an HTTPS USD asset URL."""
    return source.startswith("https://")


def _open_usd_stage(source: str | os.PathLike[str]):
    """Open a USD stage from a local path or resolved URL."""
    if Usd is None:
        raise ImportError("Failed to import pxr. Please install USD (e.g. via `pip install usd-core`).")

    source_path = os.fspath(source)
    if source_path.startswith("http://"):
        raise ValueError("HTTP USD URLs are not supported; use HTTPS or download the asset explicitly.")
    if _is_usd_url(source_path):
        from ..utils.import_usd import resolve_usd_from_url  # noqa: PLC0415

        source_path = resolve_usd_from_url(source_path)

    stage = Usd.Stage.Open(source_path, Usd.Stage.LoadAll)
    if stage is None:
        raise FileNotFoundError(f"Unable to open USD stage: {source_path}")
    return stage


def _get_root_prim(stage, root_path: str | None):
    """Return the merge root prim for a stage-backed mesh load."""
    if root_path is None or root_path == "/":
        return stage.GetPseudoRoot()

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise ValueError(f"USD root path '{root_path}' does not exist.")
    return root


def _iter_mesh_prims(root) -> list[Any]:
    """Return all mesh prims under ``root`` in traversal order."""
    return [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdGeom.Mesh)]


def _matrix_to_numpy(matrix) -> np.ndarray:
    """Convert a USD matrix value into a NumPy row-vector matrix."""
    return np.array(matrix, dtype=np.float64)


def _relative_transform_matrix(prim, root, xform_cache) -> np.ndarray:
    """Return ``prim`` transform relative to the selected merge ``root``."""
    prim_world = _matrix_to_numpy(xform_cache.GetLocalToWorldTransform(prim))
    if root.IsPseudoRoot():
        return prim_world

    root_world = _matrix_to_numpy(xform_cache.GetLocalToWorldTransform(root))
    return prim_world @ np.linalg.inv(root_world)


def _transform_mesh_data(mesh: Mesh, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Apply a row-vector USD transform to mesh vertices, winding, and normals."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices_h = np.concatenate((vertices, np.ones((len(vertices), 1), dtype=np.float64)), axis=1)
    transformed_vertices = (vertices_h @ matrix)[:, :3].astype(np.float32)

    indices = np.asarray(mesh.indices, dtype=np.int32).copy()
    linear = matrix[:3, :3]
    if np.linalg.det(linear) < 0.0:
        indices = indices.reshape(-1, 3)
        indices[:, [1, 2]] = indices[:, [2, 1]]
        indices = indices.reshape(-1)

    normals = None
    if mesh.normals is not None:
        try:
            normal_matrix = np.linalg.inv(linear).T
        except np.linalg.LinAlgError:
            normal_matrix = None
        if normal_matrix is not None:
            normals = np.asarray(mesh.normals, dtype=np.float64) @ normal_matrix
            lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            lengths[lengths < 1e-20] = 1.0
            normals = (normals / lengths).astype(np.float32)

    return transformed_vertices, indices, normals


def _get_mesh_from_source(
    source,
    *,
    root_path: str | None,
    load_normals: bool,
    load_uvs: bool,
    maxhullvert: int | None,
    face_varying_normal_conversion: Literal["vertex_averaging", "angle_weighted", "vertex_splitting"],
    vertex_splitting_angle_threshold_deg: float,
    preserve_facevarying_uvs: bool,
    compute_inertia: bool,
    apply_stage_units: bool,
) -> Mesh:
    """Load and merge mesh prims from a USD stage, path, URL, or prim subtree."""
    if Usd is None or UsdGeom is None:
        raise ImportError("Failed to import pxr. Please install USD (e.g. via `pip install usd-core`).")
    if preserve_facevarying_uvs:
        raise ValueError(
            "preserve_facevarying_uvs is not supported for merged USD sources; "
            "load a single mesh prim with return_uv_indices=True or use preserve_facevarying_uvs=False."
        )

    if isinstance(source, str | os.PathLike):
        stage = _open_usd_stage(source)
        root = _get_root_prim(stage, root_path)
    elif isinstance(source, Usd.Stage):
        stage = source
        root = _get_root_prim(stage, root_path)
    elif isinstance(source, Usd.Prim):
        stage = source.GetStage()
        root = source if root_path is None else _get_root_prim(stage, root_path)
    else:
        raise TypeError(
            f"get_mesh() expected a USD prim, USD stage, filesystem path, or URL; received {type(source).__name__}."
        )

    mesh_prims = _iter_mesh_prims(root)
    if not mesh_prims:
        raise ValueError(f"No UsdGeom.Mesh prims found under '{root.GetPath()}'.")

    linear_unit = 1.0
    if apply_stage_units and UsdGeom.StageHasAuthoredMetersPerUnit(stage):
        linear_unit = float(UsdGeom.GetStageMetersPerUnit(stage))

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    vertices_parts = []
    indices_parts = []
    normals_parts = []
    uvs_parts = []
    vertex_offset = 0
    all_have_normals = True
    all_have_uvs = True
    source_meshes = []

    for prim in mesh_prims:
        source_mesh = get_mesh(
            prim,
            load_normals=load_normals,
            load_uvs=load_uvs,
            maxhullvert=maxhullvert,
            face_varying_normal_conversion=face_varying_normal_conversion,
            vertex_splitting_angle_threshold_deg=vertex_splitting_angle_threshold_deg,
            preserve_facevarying_uvs=preserve_facevarying_uvs,
            compute_inertia=False,
        )
        source_meshes.append(source_mesh)
        matrix = _relative_transform_matrix(prim, root, xform_cache)
        vertices, indices, normals = _transform_mesh_data(source_mesh, matrix)

        vertices_parts.append(vertices)
        indices_parts.append(indices + vertex_offset)
        vertex_offset += len(vertices)

        if normals is None:
            all_have_normals = False
        else:
            normals_parts.append(normals)

        if source_mesh.uvs is None:
            all_have_uvs = False
        else:
            uvs_parts.append(np.asarray(source_mesh.uvs, dtype=np.float32))

    vertices = np.concatenate(vertices_parts, axis=0)
    if linear_unit != 1.0:
        vertices *= linear_unit
    indices = np.concatenate(indices_parts, axis=0)
    normals = np.concatenate(normals_parts, axis=0) if all_have_normals else None
    uvs = np.concatenate(uvs_parts, axis=0) if all_have_uvs else None

    material_source = source_meshes[0] if len(source_meshes) == 1 else None
    return Mesh(
        vertices,
        indices,
        normals=normals,
        uvs=uvs,
        maxhullvert=maxhullvert,
        compute_inertia=compute_inertia,
        color=material_source.color if material_source is not None else None,
        texture=material_source.texture if material_source is not None else None,
        metallic=material_source.metallic if material_source is not None else None,
        roughness=material_source.roughness if material_source is not None else None,
    )


@overload
def get_mesh(
    source: Usd.Prim | Usd.Stage | str | os.PathLike[str],
    load_normals: bool = False,
    load_uvs: bool = False,
    maxhullvert: int | None = None,
    face_varying_normal_conversion: Literal[
        "vertex_averaging", "angle_weighted", "vertex_splitting"
    ] = "vertex_splitting",
    vertex_splitting_angle_threshold_deg: float = 25.0,
    preserve_facevarying_uvs: bool = False,
    return_uv_indices: Literal[False] = False,
    root_path: str | None = None,
    compute_inertia: bool = True,
    apply_stage_units: bool = True,
) -> Mesh: ...


@overload
def get_mesh(
    source: Usd.Prim,
    load_normals: bool = False,
    load_uvs: bool = False,
    maxhullvert: int | None = None,
    face_varying_normal_conversion: Literal[
        "vertex_averaging", "angle_weighted", "vertex_splitting"
    ] = "vertex_splitting",
    vertex_splitting_angle_threshold_deg: float = 25.0,
    preserve_facevarying_uvs: bool = False,
    return_uv_indices: Literal[True] = True,
    root_path: None = None,
    compute_inertia: bool = True,
    apply_stage_units: bool = True,
) -> tuple[Mesh, np.ndarray | None]: ...


@overload
def get_mesh(
    source: None = None,
    load_normals: bool = False,
    load_uvs: bool = False,
    maxhullvert: int | None = None,
    face_varying_normal_conversion: Literal[
        "vertex_averaging", "angle_weighted", "vertex_splitting"
    ] = "vertex_splitting",
    vertex_splitting_angle_threshold_deg: float = 25.0,
    preserve_facevarying_uvs: bool = False,
    return_uv_indices: Literal[False] = False,
    root_path: str | None = None,
    compute_inertia: bool = True,
    apply_stage_units: bool = True,
    *,
    prim: Usd.Prim,
) -> Mesh: ...


@overload
def get_mesh(
    source: None = None,
    load_normals: bool = False,
    load_uvs: bool = False,
    maxhullvert: int | None = None,
    face_varying_normal_conversion: Literal[
        "vertex_averaging", "angle_weighted", "vertex_splitting"
    ] = "vertex_splitting",
    vertex_splitting_angle_threshold_deg: float = 25.0,
    preserve_facevarying_uvs: bool = False,
    return_uv_indices: Literal[True] = True,
    root_path: None = None,
    compute_inertia: bool = True,
    apply_stage_units: bool = True,
    *,
    prim: Usd.Prim,
) -> tuple[Mesh, np.ndarray | None]: ...


def get_mesh(
    source: Usd.Prim | Usd.Stage | str | os.PathLike[str] | None = None,
    load_normals: bool = False,
    load_uvs: bool = False,
    maxhullvert: int | None = None,
    face_varying_normal_conversion: Literal[
        "vertex_averaging", "angle_weighted", "vertex_splitting"
    ] = "vertex_splitting",
    vertex_splitting_angle_threshold_deg: float = 25.0,
    preserve_facevarying_uvs: bool = False,
    return_uv_indices: bool = False,
    root_path: str | None = None,
    compute_inertia: bool = True,
    apply_stage_units: bool = True,
    *,
    prim: Usd.Prim | None = None,
) -> Mesh | tuple[Mesh, np.ndarray | None]:
    """
    Load a triangle mesh from a USD mesh prim, stage, file path, or URL.

    When ``source`` is a mesh prim, the mesh is loaded in the prim's local
    coordinates. When ``source`` is a stage, path, URL, or non-mesh prim, all
    ``UsdGeom.Mesh`` prims under ``root_path`` are merged into one
    :class:`newton.Mesh` with authored transforms applied relative to that root.

    Example:

        .. testcode::

            from pxr import Usd
            import newton.examples
            import newton.usd

            usd_stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
            demo_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/bunny"), load_normals=True)

            builder = newton.ModelBuilder()
            body_mesh = builder.add_body()
            builder.add_shape_mesh(body_mesh, mesh=demo_mesh)

            assert len(demo_mesh.vertices) == 6102
            assert len(demo_mesh.indices) == 36600
            assert len(demo_mesh.normals) == 6102

    Args:
        source: USD mesh prim, stage, file path, or URL to load the mesh from.
        prim: Legacy keyword alias for ``source`` when loading a USD prim.
        load_normals: Whether to load the normals.
        load_uvs: Whether to load the UVs.
        maxhullvert: The maximum number of vertices for the convex hull approximation.
        face_varying_normal_conversion:
            This argument specifies how to convert "faceVarying" normals
            (normals defined per-corner rather than per-vertex) into per-vertex normals for the mesh.
            If ``load_normals`` is False, this argument is ignored.
            The options are summarized below:

            .. list-table::
                :widths: 20 80
                :header-rows: 1

                * - Method
                  - Description
                * - ``"vertex_averaging"``
                  - For each vertex, averages all the normals of the corners that share that vertex. This produces smooth shading except at explicit vertex splits. This method is the most efficient.
                * - ``"angle_weighted"``
                  - For each vertex, computes a weighted average of the normals of the corners it belongs to, using the corner angle as a weight (i.e., larger face angles contribute more), for more visually-accurate smoothing at sharp edges.
                * - ``"vertex_splitting"``
                  - Splits a vertex into multiple vertices if the difference between the corner normals exceeds a threshold angle (see ``vertex_splitting_angle_threshold_deg``). This preserves sharp features by assigning separate (duplicated) vertices to corners with widely different normals.

        vertex_splitting_angle_threshold_deg: The threshold angle in degrees for splitting vertices based on the face normals in case of faceVarying normals and ``face_varying_normal_conversion`` is "vertex_splitting". Corners whose normals differ by more than ``vertex_splitting_angle_threshold_deg`` will be split
            into different vertex clusters. Lower = more splits (sharper), higher = fewer splits (smoother).
        preserve_facevarying_uvs: If True, keep faceVarying UVs in their
            original corner layout and avoid UV-driven vertex splitting. The
            returned mesh keeps its original topology. This is useful when the
            caller needs the original UV indexing (e.g., panel-space cloth).
        return_uv_indices: If True, return a tuple ``(mesh, uv_indices)``
            where ``uv_indices`` is a flattened triangle index buffer for the
            UVs when available. For faceVarying UVs and
            ``preserve_facevarying_uvs=True``, these indices reference the
            face-varying UV array. Only supported for a single
            ``UsdGeom.Mesh`` prim when ``root_path`` is None.
        root_path: USD prim path to use as the merge root for stage, file path,
            URL, or non-mesh prim sources. Defaults to the stage pseudo-root for
            stages and paths, or the provided prim for non-mesh prim sources.
        compute_inertia: If True, compute mass properties for the returned
            :class:`newton.Mesh`.
        apply_stage_units: If True, convert merged stage, file path, URL, or
            non-mesh prim sources from authored USD distance units to meters.
            Single mesh prim sources keep their authored coordinates for
            backward compatibility unless ``root_path`` is provided.

    Returns:
        newton.Mesh: The loaded mesh, or ``(mesh, uv_indices)`` if
        ``return_uv_indices`` is True.
    """
    if prim is not None:
        if source is not None:
            raise TypeError("get_mesh() received both 'source' and legacy 'prim'; pass only one.")
        source = prim
    elif source is None:
        raise TypeError("get_mesh() missing required argument: 'source'.")

    if maxhullvert is None:
        maxhullvert = Mesh.MAX_HULL_VERTICES

    should_load_source = isinstance(source, str | os.PathLike) or (
        Usd is not None
        and (
            isinstance(source, Usd.Stage)
            or (isinstance(source, Usd.Prim) and (root_path is not None or not source.IsA(UsdGeom.Mesh)))
        )
    )
    if should_load_source:
        if return_uv_indices:
            raise ValueError("return_uv_indices is only supported when loading a single UsdGeom.Mesh prim.")
        return _get_mesh_from_source(
            source,
            root_path=root_path,
            load_normals=load_normals,
            load_uvs=load_uvs,
            maxhullvert=maxhullvert,
            face_varying_normal_conversion=face_varying_normal_conversion,
            vertex_splitting_angle_threshold_deg=vertex_splitting_angle_threshold_deg,
            preserve_facevarying_uvs=preserve_facevarying_uvs,
            compute_inertia=compute_inertia,
            apply_stage_units=apply_stage_units,
        )

    if Usd is not None and isinstance(source, Usd.Prim):
        prim = source
    else:
        raise TypeError(
            f"get_mesh() expected a USD prim, USD stage, filesystem path, or URL; received {type(source).__name__}."
        )

    mesh = UsdGeom.Mesh(prim)

    points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    counts = mesh.GetFaceVertexCountsAttr().Get()

    uvs = None
    uvs_interpolation = None
    # Tracks whether we already duplicated vertices (and per-vertex UVs) during
    # faceVarying normal conversion, so we don't split again in the UV pass.
    did_split_vertices = False
    if load_uvs:
        uv_primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        if uv_primvar:
            uvs = uv_primvar.Get()
            if uvs is not None:
                uvs = np.array(uvs)
                # Get interpolation from primvar
                uvs_interpolation = uv_primvar.GetInterpolation()
                # Check if this primvar is indexed and expand if so
                if uv_primvar.IsIndexed():
                    uv_indices = uv_primvar.GetIndices()
                    uvs = _expand_indexed_primvar(uvs, uv_indices, "UV", str(prim.GetPath()))

    normals = None
    normals_interpolation = None
    normal_indices = None
    if load_normals:
        # First, try to load normals from primvars:normals (takes precedence)
        normals_primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar("normals")
        if normals_primvar:
            normals = normals_primvar.Get()
            if normals is not None:
                # Use primvar interpolation
                normals_interpolation = normals_primvar.GetInterpolation()
                # Check for primvar indices
                if normals_primvar.IsIndexed():
                    normal_indices = normals_primvar.GetIndices()
                # Fall back to direct attribute access for backwards compatibility
                if normal_indices is None:
                    normals_index_attr = prim.GetAttribute("primvars:normals:indices")
                    if normals_index_attr and normals_index_attr.HasAuthoredValue():
                        normal_indices = normals_index_attr.Get()

        # Fall back to mesh.GetNormalsAttr() only if primvar is not present or has no data
        if normals is None:
            normals_attr = mesh.GetNormalsAttr()
            if normals_attr:
                normals = normals_attr.Get()
                if normals is not None:
                    # Use mesh normals interpolation (only relevant for non-primvar normals)
                    normals_interpolation = mesh.GetNormalsInterpolation()

    if normals is not None:
        normals = np.array(normals, dtype=np.float64)
        if normals_interpolation == UsdGeom.Tokens.faceVarying:
            prim_path = str(prim.GetPath())
            if normal_indices is not None and len(normal_indices) > 0:
                normals_fv = _expand_indexed_primvar(normals, normal_indices, "Normal", prim_path)
            else:
                # If faceVarying, values length must match number of corners
                if len(normals) != len(indices):
                    raise ValueError(
                        f"Length of normals ({len(normals)}) does not match length of indices ({len(indices)}) for mesh {prim_path}"
                    )
                normals_fv = normals  # (C,3)

            V = len(points)
            accum = np.zeros((V, 3), dtype=np.float64)
            if face_varying_normal_conversion == "vertex_splitting":
                C = len(indices)
                Nfv = np.asarray(normals_fv, dtype=np.float64)
                if indices.shape[0] != Nfv.shape[0]:
                    raise ValueError(
                        f"Length of indices ({indices.shape[0]}) does not match length of faceVarying normals ({Nfv.shape[0]}) for mesh {prim.GetPath()}"
                    )

                # Normalize corner normals (direction only)
                nlen = np.linalg.norm(Nfv, axis=1, keepdims=True)
                nlen = np.clip(nlen, 1e-30, None)
                Ndir = Nfv / nlen

                cos_thresh = np.cos(np.deg2rad(vertex_splitting_angle_threshold_deg))

                # For each original vertex v, we'll keep a list of clusters:
                # each cluster stores (sum_dir, count, new_vid)
                clusters_per_v = [[] for _ in range(V)]

                new_points = []
                new_norm_sums = []  # accumulate directions per new vertex id
                new_indices = np.empty_like(indices)
                new_uvs = [] if uvs is not None else None

                # Helper to create a new vertex clone from original v
                def _new_vertex_from(v, n_dir, corner_idx):
                    new_vid = len(new_points)
                    new_points.append(points[v])
                    new_norm_sums.append(n_dir.copy())
                    clusters_per_v[v].append([n_dir.copy(), 1, new_vid])
                    if new_uvs is not None:
                        # Use corner UV if faceVarying, otherwise use vertex UV
                        if uvs_interpolation == UsdGeom.Tokens.faceVarying:
                            new_uvs.append(uvs[corner_idx])
                        else:
                            new_uvs.append(uvs[v])
                    return new_vid

                # Assign each corner to a cluster (new vertex) based on angular proximity
                for c in range(C):
                    v = int(indices[c])
                    n_dir = Ndir[c]

                    clusters = clusters_per_v[v]
                    assigned = False
                    # try to match an existing cluster
                    for cl in clusters:
                        sum_dir, cnt, new_vid = cl
                        # compare with current mean direction (sum_dir normalized)
                        mean_dir = sum_dir / max(np.linalg.norm(sum_dir), 1e-30)
                        if float(np.dot(mean_dir, n_dir)) >= cos_thresh:
                            # assign to this cluster
                            cl[0] = sum_dir + n_dir
                            cl[1] = cnt + 1
                            new_norm_sums[new_vid] += n_dir
                            new_indices[c] = new_vid
                            assigned = True
                            break

                    if not assigned:
                        new_vid = _new_vertex_from(v, n_dir, c)
                        new_indices[c] = new_vid

                new_points = np.asarray(new_points, dtype=np.float64)

                # Produce per-vertex normalized normals for the new vertices
                new_norm_sums = np.asarray(new_norm_sums, dtype=np.float64)
                nn = np.linalg.norm(new_norm_sums, axis=1, keepdims=True)
                nn = np.clip(nn, 1e-30, None)
                new_vertex_normals = (new_norm_sums / nn).astype(np.float32)

                points = new_points
                indices = new_indices
                normals = new_vertex_normals
                uvs = new_uvs
                # Vertex splitting creates a new per-vertex layout (and UVs
                # if available). Skip the later faceVarying UV split to avoid
                # dropping/duplicating UVs.
                did_split_vertices = True
            elif face_varying_normal_conversion == "vertex_averaging":
                # basic averaging
                for c, v in enumerate(indices):
                    accum[v] += normals_fv[c]
                # normalize
                lengths = np.linalg.norm(accum, axis=1, keepdims=True)
                lengths[lengths < 1e-20] = 1.0
                # vertex normals
                normals = (accum / lengths).astype(np.float32)
            elif face_varying_normal_conversion == "angle_weighted":
                # area- or corner-angle weighting
                offset = 0
                for nverts in counts:
                    face_idx = indices[offset : offset + nverts]
                    face_pos = points[face_idx]  # (n,3)
                    # compute per-corner angles at each vertex in the face (omitted here for brevity)
                    weights = corner_angles(face_pos)  # (n,)
                    for i in range(nverts):
                        v = face_idx[i]
                        accum[v] += normals_fv[offset + i] * weights[i]
                    offset += nverts

                vertex_normals = accum / np.clip(np.linalg.norm(accum, axis=1, keepdims=True), 1e-20, None)
                normals = vertex_normals.astype(np.float32)
            else:
                raise ValueError(f"Invalid face_varying_normal_conversion: {face_varying_normal_conversion}")

    faces = fan_triangulate_faces(counts, indices)

    flip_winding = False
    orientation_attr = mesh.GetOrientationAttr()
    if orientation_attr:
        handedness = orientation_attr.Get()
        if handedness and handedness.lower() == "lefthanded":
            flip_winding = True
    if flip_winding:
        faces = faces[:, ::-1]

    uv_indices = None
    if uvs is not None:
        uvs = np.array(uvs, dtype=np.float32)
        # If vertices were already split for faceVarying normals, UVs (if any)
        # were converted to per-vertex. Avoid a second split here.
        if uvs_interpolation == UsdGeom.Tokens.faceVarying and not did_split_vertices:
            if len(uvs) != len(indices):
                logger.info(
                    "Mesh %s: UV primvar length (%d) does not match indices length (%d); dropping UVs.",
                    prim.GetPath(),
                    len(uvs),
                    len(indices),
                )
                uvs = None
            else:
                corner_flat = _triangulate_face_varying_indices(counts, flip_winding)
                if not preserve_facevarying_uvs:
                    points_original = points
                    points = points_original[indices[corner_flat]]
                    if normals is not None:
                        if len(normals) == len(points_original):
                            normals = normals[indices[corner_flat]]
                        elif len(normals) == len(corner_flat):
                            normals = normals[corner_flat]
                        else:
                            warnings.warn(
                                f"Normals length ({len(normals)}) does not match vertices after UV splitting for mesh {prim.GetPath()}; "
                                "dropping normals.",
                                stacklevel=2,
                            )
                            normals = None
                    uvs = uvs[corner_flat]
                    faces = np.arange(len(corner_flat), dtype=np.int32).reshape(-1, 3)
                elif return_uv_indices:
                    uv_indices = corner_flat

    if return_uv_indices and uvs is not None and uv_indices is None:
        uv_indices = faces.reshape(-1)

    material_props = resolve_material_properties_for_prim(prim)

    mesh_out = Mesh(
        points,
        faces.flatten(),
        normals=normals,
        uvs=uvs,
        maxhullvert=maxhullvert,
        compute_inertia=compute_inertia,
        color=material_props.get("color"),
        texture=material_props.get("texture"),
        metallic=material_props.get("metallic"),
        roughness=material_props.get("roughness"),
    )
    if return_uv_indices:
        return mesh_out, uv_indices
    return mesh_out


# Schema-defined TetMesh attribute names excluded from custom attribute parsing.
_TETMESH_SCHEMA_ATTRS = frozenset(
    {
        "points",
        "tetVertexIndices",
        "surfaceFaceVertexIndices",
        "extent",
        "orientation",
        "purpose",
        "visibility",
        "xformOpOrder",
        "proxyPrim",
        # Standard UsdGeom.PointBased attributes (velocities, accelerations, normals): importing them
        # is deferred to a follow-up, so skip them here rather than capturing them as custom data.
        "velocities",
        "accelerations",
        "normals",
    }
)


# Vendor attribute namespaces that get_tetmesh() read by default before the canonical
# ``physics:`` deformable schema existed. Pass as ``compat_namespaces`` to read them
# off a bound material.
DEFORMABLE_LEGACY_NAMESPACES: tuple[str, ...] = ("omniphysics", "physxDeformableBody")


def _material_authors_legacy_deformable_attrs(prim: Usd.Prim) -> bool:
    """Return whether ``prim``'s bound physics material authors deformable moduli only under
    the legacy vendor namespaces (see :data:`DEFORMABLE_LEGACY_NAMESPACES`).

    True means a canonical-only read would silently drop the authored stiffness/density:
    the material lacks ``PhysicsVolumeDeformableMaterialAPI`` but carries vendor-namespaced
    ``youngsModulus`` / ``poissonsRatio`` / ``density``. Callers use this to keep a
    deprecation window for such assets.
    """
    material_prim = _find_physics_material_prim(prim)
    if material_prim is None or has_applied_api_schema(material_prim, "PhysicsVolumeDeformableMaterialAPI"):
        return False
    return any(
        material_prim.GetAttribute(f"{namespace}:{name}").HasAuthoredValue()
        for namespace in DEFORMABLE_LEGACY_NAMESPACES
        for name in ("youngsModulus", "poissonsRatio", "density")
    )


def _material_authors_unscoped_canonical_attrs(prim: Usd.Prim) -> bool:
    """Return whether ``prim``'s bound physics material authors canonical ``physics:``
    deformable moduli without ``PhysicsVolumeDeformableMaterialAPI``.

    The deprecated ``get_tetmesh()`` default reads these off any bound material, while
    canonical-only behavior (``compat_namespaces=()``) scopes moduli to materials with the
    applied API and would silently drop them -- the second case where the legacy default
    is load-bearing and deserves the deprecation warning.
    """
    material_prim = _find_physics_material_prim(prim)
    if material_prim is None or has_applied_api_schema(material_prim, "PhysicsVolumeDeformableMaterialAPI"):
        return False
    return any(
        material_prim.GetAttribute(f"physics:{name}").HasAuthoredValue()
        for name in ("youngsModulus", "poissonsRatio", "density")
    )


def get_tetmesh(
    prim: Usd.Prim,
    *,
    compat_namespaces: Sequence[str] | None = None,
    _load_custom_attributes: bool = True,
) -> TetMesh:
    """Load a tetrahedral mesh from a USD prim with the ``UsdGeom.TetMesh`` schema.

    Reads vertex positions from the ``points`` attribute and tetrahedral
    connectivity from ``tetVertexIndices``. If a physics material is bound
    to the prim (via ``material:binding:physics``) and contains
    ``youngsModulus``, ``poissonsRatio``, or ``density`` attributes (canonical
    ``physics:`` namespace, with ``compat_namespaces`` as a fallback),
    those values are read and converted to Lame parameters (``k_mu``,
    ``k_lambda``) and density on the returned TetMesh. Material properties
    are set to ``None`` if not present.

    Custom primvars use their resolved interpolation to determine attribute
    frequency. Other custom arrays use length-based inference; arrays whose
    frequency is ambiguous or cannot be inferred emit a warning and are
    omitted without preventing the TetMesh from loading.

    Material-attribute namespaces (deprecated default): with ``compat_namespaces=None``
    (the default) the legacy vendor namespaces (``omniphysics:`` / ``physxDeformableBody:``)
    are read off any bound material, matching the pre-canonical behavior. That default is
    deprecated and emits a ``DeprecationWarning`` when it is load-bearing: the bound material
    authors vendor-namespaced deformable attributes, or canonical ``physics:`` attributes
    without ``PhysicsVolumeDeformableMaterialAPI`` (API-applied canonical or render-only
    materials do not warn); a future
    release will default to canonical ``physics:``-only. Pass ``compat_namespaces=()`` to adopt
    the canonical-only behavior now -- moduli are then read only from a material that applies
    ``PhysicsVolumeDeformableMaterialAPI`` -- or pass an explicit list (e.g.
    ``newton.usd.DEFORMABLE_LEGACY_NAMESPACES``) to keep reading vendor namespaces without the
    warning.

    Example:

        .. code-block:: python

            from pxr import Usd
            import newton
            import newton.usd

            usd_stage = Usd.Stage.Open("tetmesh.usda")
            tetmesh = newton.usd.get_tetmesh(usd_stage.GetPrimAtPath("/MyTetMesh"))

            # tetmesh.vertices  -- np.ndarray, shape (N, 3)
            # tetmesh.tet_indices -- np.ndarray, flattened (4 per tet)

    Args:
        prim: The USD prim to load the tetrahedral mesh from.
        compat_namespaces: Vendor attribute namespaces accepted as a fallback to the canonical
            ``physics:`` material attributes, lifting the ``PhysicsVolumeDeformableMaterialAPI``
            gate. ``None`` (the default) selects the deprecated legacy namespaces; pass ``()`` for
            canonical-only.

    Returns:
        TetMesh: A :class:`newton.TetMesh` with vertex positions and tet connectivity.
    """
    from ..geometry.types import TetMesh  # noqa: PLC0415

    tet_mesh = UsdGeom.TetMesh(prim)

    points_attr = tet_mesh.GetPointsAttr().Get()
    if points_attr is None:
        raise ValueError(f"TetMesh prim '{prim.GetPath()}' has no points attribute.")

    tet_indices_attr = tet_mesh.GetTetVertexIndicesAttr().Get()
    if tet_indices_attr is None:
        raise ValueError(f"TetMesh prim '{prim.GetPath()}' has no tetVertexIndices attribute.")

    vertices = np.array(points_attr, dtype=np.float32)
    tet_indices = np.array(tet_indices_attr, dtype=np.int32).flatten()

    # Flip winding order for left-handed meshes (e.g. Houdini exports)
    handedness = tet_mesh.GetOrientationAttr().Get()
    if handedness and handedness.lower() == "lefthanded" and tet_indices.size % 4 == 0:
        tet_indices = tet_indices.reshape(-1, 4)
        tet_indices[:, [1, 2]] = tet_indices[:, [2, 1]]
        tet_indices = tet_indices.reshape(-1)

    # Try to read physics material properties if bound
    k_mu = None
    k_lambda = None
    density = None

    # Volume material moduli (youngsModulus/poissonsRatio/...) belong on the bound material, not the
    # geometry; warn if authored on the TetMesh prim itself so the misplacement is visible to direct
    # get_tetmesh() callers too (add_usd's deformable pass warns separately).
    _warn_geometry_authored_material_attrs(
        prim, str(prim.GetPath()), "PhysicsVolumeDeformableMaterialAPI", _read_physics_attr
    )

    material_prim = _find_physics_material_prim(prim)
    if compat_namespaces is None:
        # Deprecated legacy default: read vendor namespaces off any bound material, and
        # canonical moduli off materials without the deformable material API. Warn only when
        # that default is load-bearing -- vendor attrs authored, or canonical attrs on an
        # API-less material -- so materials whose reads the default change does not alter
        # (API-applied canonical, render-only) never warn.
        if _material_authors_legacy_deformable_attrs(prim) or _material_authors_unscoped_canonical_attrs(prim):
            warnings.warn(
                "get_tetmesh(): the default reads deformable material attributes off any bound "
                "material (canonical physics: and legacy omniphysics: / physxDeformableBody: "
                "namespaces); this is deprecated, and a future release will read canonical "
                "physics: attributes only off a material applying "
                "PhysicsVolumeDeformableMaterialAPI. Pass compat_namespaces=() to adopt the "
                "canonical-only behavior now, or compat_namespaces="
                "newton.usd.DEFORMABLE_LEGACY_NAMESPACES to keep the current behavior explicitly.",
                DeprecationWarning,
                stacklevel=2,
            )
        compat_namespaces = DEFORMABLE_LEGACY_NAMESPACES
    # Canonical behavior (compat_namespaces=()) scopes the moduli to the volume deformable material
    # API, so they are not read off an unrelated physics material. A non-empty compat_namespaces reads
    # the listed vendor namespaces off any bound material.
    read_material = material_prim is not None and (
        bool(compat_namespaces) or has_applied_api_schema(material_prim, "PhysicsVolumeDeformableMaterialAPI")
    )
    if read_material:
        youngs = _read_physics_attr(material_prim, "youngsModulus", compat_namespaces)
        poissons = _read_physics_attr(material_prim, "poissonsRatio", compat_namespaces)
        density_val = _read_physics_attr(material_prim, "density", compat_namespaces)

        # The proposal declares youngsModulus with a fallback of -inf, meaning "simulator
        # default": treat it like an unauthored modulus rather than an invalid value.
        if youngs is not None and float(youngs) == float("-inf"):
            youngs = None
        if youngs is not None:
            E = float(youngs)
            # The proposal declares physics:poissonsRatio with a fallback of 0.3. The schema
            # is not registered with USD, so the fallback cannot be injected by composition
            # and must be applied here; otherwise an authored Young's modulus would be
            # silently discarded whenever the ratio is left at its default.
            nu = 0.3 if poissons is None else float(poissons)
            if not (math.isfinite(E) and E >= 0.0 and math.isfinite(nu)):
                warnings.warn(
                    f"{material_prim.GetPath()}: invalid volume material (youngsModulus={E}, "
                    f"poissonsRatio={nu}); ignoring the authored elastic moduli.",
                    stacklevel=2,
                )
            else:
                # Clamp Poisson's ratio to the open interval (-1, 0.5) to avoid
                # division by zero in the Lame parameter conversion.
                nu = max(-0.999, min(nu, 0.499))
                k_mu = E / (2.0 * (1.0 + nu))
                k_lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        if density_val is not None:
            authored_density = float(density_val)
            # The proposal declares density with a range of (0, inf) and a fallback of 0
            # meaning "ignored": zero falls through to the caller's density precedence
            # (body override, then the builder default) without being an invalid value.
            if math.isfinite(authored_density) and authored_density > 0.0:
                density = authored_density
            elif authored_density != 0.0:
                warnings.warn(
                    f"{material_prim.GetPath()}: invalid volume material density "
                    f"{authored_density}; expected a finite positive value, ignoring it.",
                    stacklevel=2,
                )

    if density is None:
        # The base UsdPhysicsMaterialAPI (which the family APIs extend) supplies
        # density too; a plain rigid-style physics material is a valid source.
        density = _get_physics_material_density(material_prim)

    if not _load_custom_attributes:
        return TetMesh(
            vertices=vertices,
            tet_indices=tet_indices,
            k_mu=k_mu,
            k_lambda=k_lambda,
            density=density,
        )

    # Read custom primvars and attributes (per-vertex, per-tet, etc.)
    # Primvar interpolation is used to determine the attribute frequency
    # when available, falling back to length-based inference in TetMesh.__init__.
    from ..sim.model import Model as _Model  # noqa: PLC0415

    # USD interpolation → Newton frequency for TetMesh prims.
    # "uniform" means one value per geometric element (cell); for a TetMesh
    # the cells are tetrahedra, so it maps to TETRAHEDRON.
    _INTERP_TO_FREQ = {
        "vertex": _Model.AttributeFrequency.PARTICLE,
        "varying": _Model.AttributeFrequency.PARTICLE,
        "uniform": _Model.AttributeFrequency.TETRAHEDRON,
        "constant": _Model.AttributeFrequency.ONCE,
    }

    custom_attributes: dict[str, np.ndarray | tuple[np.ndarray, _Model.AttributeFrequency]] = {}

    primvars_api = UsdGeom.PrimvarsAPI(prim)
    for primvar in primvars_api.GetPrimvarsWithValues():
        name = primvar.GetPrimvarName()
        if name in ("st", "normals"):
            continue  # skip well-known primvars handled elsewhere
        val = primvar.ComputeFlattened()
        if val is not None:
            arr = np.array(val)
            interp = primvar.GetInterpolation()
            freq = _INTERP_TO_FREQ.get(interp)
            if freq is not None:
                custom_attributes[str(name)] = (arr, freq)
            else:
                # Unknown interpolation — let TetMesh infer from length
                custom_attributes[str(name)] = arr

    # Also read non-schema custom attributes (not primvars, not relationships)
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if name in _TETMESH_SCHEMA_ATTRS:
            continue
        if name.startswith("primvars:") or name.startswith("xformOp:"):
            continue
        # Deformable physics-schema attributes are handled by the deformable importer
        # where supported (e.g. physics:masses) or intentionally skipped (e.g.
        # physics:restShapePoints, whose rest-shape import is not yet supported and is
        # warned about by the importer); none are carried as generic mesh data here.
        if name.startswith(("physics:", "omniphysics:", "physxDeformableBody:")):
            continue
        if not attr.HasAuthoredValue():
            continue
        val = attr.Get()
        if val is not None:
            try:
                arr = np.array(val)
                if arr.ndim >= 1:
                    custom_attributes[name] = arr
            except (TypeError, ValueError):
                pass  # skip non-array attributes

    result = TetMesh(
        vertices=vertices,
        tet_indices=tet_indices,
        k_mu=k_mu,
        k_lambda=k_lambda,
        density=density,
    )
    tri_count = len(result.surface_tri_indices) // 3
    for name, value in custom_attributes.items():
        if name in result._RESERVED_ATTR_KEYS:
            warnings.warn(
                f"{prim.GetPath()}: custom attribute '{name}' uses a reserved TetMesh name; skipping the attribute.",
                stacklevel=2,
            )
            continue
        if isinstance(value, tuple):
            arr, frequency = value
        else:
            arr = value
            try:
                frequency = result._infer_frequency(arr, result.vertex_count, result.tet_count, tri_count, name)
            except ValueError as exc:
                warnings.warn(f"{prim.GetPath()}: {exc}; skipping the attribute.", stacklevel=2)
                continue
        result.custom_attributes[name] = (np.asarray(arr), frequency)
    return result


def _find_physics_material_prim(prim: Usd.Prim):
    """Resolve the ``physics``-purpose bound material prim (or ``None``).

    Via :meth:`UsdShade.MaterialBindingAPI.ComputeBoundMaterial`, honoring inherited,
    collection-based, and strength-resolved (``bindMaterialAs``) bindings.
    """
    material, _rel = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
    mat_prim = material.GetPrim()
    return mat_prim if mat_prim and mat_prim.IsValid() else None


def _read_physics_attr(prim: Usd.Prim, name: str, compat_namespaces: Sequence[str] = ()):
    """Read a deformable physics attribute, canonical ``physics:`` namespace first.

    The AOUSD deformable proposal authors under ``physics:``; this is parsed as
    written. ``compat_namespaces`` lists vendor namespaces (e.g. ``omniphysics``,
    ``physxDeformableBody``) accepted as a fallback, sourced from the active
    schema resolvers (see :meth:`SchemaResolverManager.deformable_compat_namespaces`),
    so a default import reads only the canonical schema.
    """
    for prefix in ("physics", *compat_namespaces):
        attr = prim.GetAttribute(f"{prefix}:{name}")
        if attr and attr.HasAuthoredValue():
            return attr.Get()
    return None


def _read_deformable_material(
    prim: Usd.Prim, read_attr: Callable[[Usd.Prim, str], Any], api_schema: str, attr_names: Sequence[str]
) -> dict[str, float] | None:
    """Read a per-family deformable material's authored, in-range parameters bound to a prim.

    Shared by the per-family readers (:func:`_get_curve_deformable_material`,
    :func:`_get_surface_deformable_material`): resolves the physics material bound via
    ``material:binding:physics`` and reads ``attr_names`` through ``read_attr`` (the resolver's
    single-source namespace read, see :meth:`SchemaResolverManager.read_deformable_attr`) when the
    bound material declares ``api_schema``.

    Returns a dict of the authored, finite values among ``attr_names``, or ``None`` if the bound
    material does not declare ``api_schema``. Stiffness fields keep an authored zero (the proposal's
    range is ``[0, inf)``); ``thickness`` and ``density`` must be positive. The schema's ``-inf``
    "simulator default" sentinel (and any out-of-range value) is dropped so the caller falls back to
    its defaults.
    """
    material_prim = _find_physics_material_prim(prim)
    if material_prim is None or not has_applied_api_schema(material_prim, api_schema):
        return None
    out: dict[str, float] = {}
    for name in attr_names:
        val = read_attr(material_prim, name)
        if val is None:
            continue
        val = float(val)
        if not math.isfinite(val):
            continue  # drops the -inf "simulator default" sentinel
        # Stiffness accepts [0, inf), so an authored zero is preserved. Thickness and
        # density must be strictly positive.
        if name in ("thickness", "density"):
            if val > 0.0:
                out[name] = val
            elif name == "thickness" or val < 0.0:
                # A finite non-positive thickness (or negative density) is malformed, not the
                # unauthored sentinel (-inf); say it is dropped so users can tell it apart
                # from an unauthored value. An authored density of exactly 0 stays silent:
                # that is the proposal's "ignored" sentinel.
                warnings.warn(
                    f"{material_prim.GetPath()}: invalid physics:{name} {val:g} (expected > 0); "
                    f"treating it as unauthored.",
                    stacklevel=2,
                )
        elif val >= 0.0:
            out[name] = val
    return out


def _get_curve_deformable_material(
    prim: Usd.Prim, read_attr: Callable[[Usd.Prim, str], Any]
) -> dict[str, float] | None:
    """Read curve-deformable (cable) ``PhysicsCurvesDeformableMaterialAPI`` parameters bound to a prim.

    Returns a dict of authored, finite values among ``thickness``, ``stretchStiffness``,
    ``shearStiffness``, ``bendStiffness``, ``twistStiffness`` and ``density``; or ``None`` if the
    bound material does not declare ``PhysicsCurvesDeformableMaterialAPI``. See
    :func:`_read_deformable_material` for the value-validation rules.
    """
    return _read_deformable_material(
        prim,
        read_attr,
        "PhysicsCurvesDeformableMaterialAPI",
        ("thickness", "stretchStiffness", "shearStiffness", "bendStiffness", "twistStiffness", "density"),
    )


def _get_surface_deformable_material(
    prim: Usd.Prim, read_attr: Callable[[Usd.Prim, str], Any]
) -> dict[str, float] | None:
    """Read surface-deformable (cloth) ``PhysicsSurfaceDeformableMaterialAPI`` parameters bound to a prim.

    Returns a dict of authored, finite values among ``thickness``, ``stretchStiffness``,
    ``shearStiffness``, ``bendStiffness`` and ``density``; or ``None`` if the bound material does not
    declare ``PhysicsSurfaceDeformableMaterialAPI``. See :func:`_read_deformable_material` for the
    value-validation rules.
    """
    return _read_deformable_material(
        prim,
        read_attr,
        "PhysicsSurfaceDeformableMaterialAPI",
        ("thickness", "stretchStiffness", "shearStiffness", "bendStiffness", "density"),
    )


def _get_physics_material_density(material_prim) -> float | None:
    """Read a bound material's base ``UsdPhysicsMaterialAPI`` density.

    The proposal reuses the rigid ``UsdPhysicsMaterialAPI`` for deformables, so a
    material applying only the base API still supplies density (the family
    material APIs extend it). Accepts a finite value greater than zero; zero is
    the proposal's ignored fallback; other values warn and are ignored.
    """
    from pxr import UsdPhysics

    if material_prim is None or not (
        material_prim.HasAPI(UsdPhysics.MaterialAPI) or has_applied_api_schema(material_prim, "PhysicsMaterialAPI")
    ):
        return None
    attr = material_prim.GetAttribute("physics:density")
    value = attr.Get() if attr else None
    if value is None:
        return None
    density = float(value)
    if math.isfinite(density) and density > 0.0:
        return density
    if density != 0.0:
        warnings.warn(
            f"{material_prim.GetPath()}: invalid physics material density {density}; "
            f"expected a finite positive value, ignoring it.",
            stacklevel=2,
        )
    return None


def _deformable_body_ancestor(prim: Usd.Prim) -> Usd.Prim | None:
    """Find the deformable body whose subtree contains ``prim`` (ownership, not governance).

    Unlike :func:`_find_deformable_body_prim`, which resolves the *governing* body of a
    simulation geometry (the prim itself or its direct parent), this walks every ancestor
    to answer subtree containment: the proposal makes all PointBased prims in a body's
    subtree part of that body. The walk stops at a rigid body or articulation root, whose
    subtree is native content the deformable must not claim.
    """
    from pxr import UsdPhysics

    p = prim
    while p and p.IsValid():
        if has_applied_api_schema(p, "PhysicsDeformableBodyAPI"):
            return p
        if p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.ArticulationRootAPI):
            return None
        p = p.GetParent()
    return None


def _find_deformable_body_prim(prim: Usd.Prim) -> Usd.Prim | None:
    """Find the ``PhysicsDeformableBodyAPI`` prim governing a simulation geometry.

    The deformable proposal allows the body API on the simulation geometry itself or
    on the prim whose direct child is the simulation geometry; nothing deeper governs
    it. A body API found only on a distant ancestor warns (so its intended overrides
    are not silently dropped) and is not used.
    """
    from pxr import UsdPhysics

    if has_applied_api_schema(prim, "PhysicsDeformableBodyAPI"):
        return prim
    parent = prim.GetParent()
    if parent and parent.IsValid():
        if has_applied_api_schema(parent, "PhysicsDeformableBodyAPI"):
            return parent
        # Advisory walk: warn when a body API sits deeper, so its intended overrides
        # are not dropped silently. Stop at a rigid or articulation boundary: that
        # content is native and a body API above it does not relate to this prim.
        p = parent
        while p and p.IsValid():
            if p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.ArticulationRootAPI):
                break
            if has_applied_api_schema(p, "PhysicsDeformableBodyAPI"):
                warnings.warn(
                    f"{prim.GetPath()}: PhysicsDeformableBodyAPI on ancestor {p.GetPath()} does not "
                    f"govern this simulation geometry (the deformable proposal allows the body API "
                    f"only on the geometry itself or its direct parent); ignoring it.",
                    stacklevel=2,
                )
                break
            p = p.GetParent()
    return None


def _get_deformable_body_overrides(
    prim: Usd.Prim, read_attr: Callable[[Usd.Prim, str], Any]
) -> tuple[float | None, float | None]:
    """Read ``PhysicsDeformableBodyAPI`` ``mass`` / ``density`` overrides.

    Both default to 0 in the schema, meaning "ignore for mass distribution"; only
    positive authored values are returned. ``density`` here overrides the bound
    material's density (see the precedence in :meth:`ModelBuilder.add_usd`).

    Returns:
        ``(mass, density)`` with each entry ``None`` when unset / non-positive.
    """
    body_prim = _find_deformable_body_prim(prim)
    if body_prim is None:
        return None, None
    mass = read_attr(body_prim, "mass")
    density = read_attr(body_prim, "density")
    # Require a finite positive value; drop unset, non-positive, or inf/nan overrides.
    mass = float(mass) if mass is not None and math.isfinite(float(mass)) and float(mass) > 0.0 else None
    density = float(density) if density is not None and math.isfinite(float(density)) and float(density) > 0.0 else None
    return mass, density


def _get_deformable_point_masses(prim: Usd.Prim, read_attr: Callable[[Usd.Prim, str], Any]) -> list[float] | None:
    """Read the simulation API's per-point ``physics:masses`` array.

    Per-point masses take precedence over body and material mass/density (proposal
    "Simulation Geometry and Rest Shape"). Returns ``None`` when unauthored/empty.
    """
    val = read_attr(prim, "masses")
    if val is None:
        return None
    return _validate_mass_array(val, str(prim.GetPath()))


def find_tetmesh_prims(stage: Usd.Stage) -> list[Usd.Prim]:
    """Find all prims with the ``UsdGeom.TetMesh`` schema in a USD stage.

    Example:

        .. code-block:: python

            from pxr import Usd
            import newton.usd

            stage = Usd.Stage.Open("scene.usda")
            prims = newton.usd.find_tetmesh_prims(stage)
            tetmeshes = [newton.usd.get_tetmesh(p) for p in prims]

    Args:
        stage: The USD stage to search.

    Returns:
        list[Usd.Prim]: All prims in the stage that have the TetMesh schema.
    """
    return [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.TetMesh)]


def _resolve_asset_path(
    asset: Sdf.AssetPath | str | os.PathLike[str] | None,
    prim: Usd.Prim,
    asset_attr: Any | None = None,
) -> str | None:
    """Resolve a USD asset reference to a usable path or URL.

    Args:
        asset: The asset value or asset path authored on a shader input.
        prim: The prim providing the stage context for relative paths.
        asset_attr: Optional USD attribute providing authored layer resolution.

    Returns:
        Absolute path or URL to the asset, or ``None`` when missing.
    """
    if asset is None:
        return None

    if asset_attr is not None:
        try:
            resolved_attr_path = asset_attr.GetResolvedPath()
        except Exception:
            resolved_attr_path = None
        if resolved_attr_path:
            return resolved_attr_path

    if isinstance(asset, Sdf.AssetPath):
        if asset.resolvedPath:
            return asset.resolvedPath
        asset_path = asset.path
    elif isinstance(asset, os.PathLike):
        asset_path = os.fspath(asset)
    elif isinstance(asset, str):
        asset_path = asset
    else:
        # Ignore non-path inputs (e.g. numeric shader parameters).
        return None

    if not asset_path:
        return None
    if asset_path.startswith(("http://", "https://", "file:")):
        return asset_path
    if os.path.isabs(asset_path):
        return asset_path

    source_layer = None
    if asset_attr is not None:
        try:
            resolve_info = asset_attr.GetResolveInfo()
        except Exception:
            resolve_info = None
        if resolve_info is not None:
            for getter_name in ("GetSourceLayer", "GetLayer"):
                getter = getattr(resolve_info, getter_name, None)
                if getter is None:
                    continue
                try:
                    source_layer = getter()
                except Exception:
                    source_layer = None
                if source_layer is not None:
                    break
        if source_layer is None:
            try:
                spec = asset_attr.GetSpec()
            except Exception:
                spec = None
            if spec is not None:
                source_layer = getattr(spec, "layer", None)

    root_layer = prim.GetStage().GetRootLayer()
    base_layer = source_layer or root_layer
    if base_layer is not None:
        try:
            resolved = Sdf.ComputeAssetPathRelativeToLayer(base_layer, asset_path)
        except Exception:
            resolved = None
        if resolved:
            return resolved
        base_dir = os.path.dirname(base_layer.realPath or base_layer.identifier or "")
        if base_dir:
            return os.path.abspath(os.path.join(base_dir, asset_path))

    return asset_path


def _get_attribute_color_space(attr: Usd.Attribute | None) -> str:
    """Return directly authored or inherited USD color-space metadata."""
    color_space = None
    if attr is not None:
        try:
            color_space = attr.GetColorSpace()
        except Exception:
            color_space = None
        if not color_space and Usd is not None:
            try:
                color_space = Usd.ColorSpaceAPI.ComputeColorSpaceName(attr, None)
            except Exception:
                color_space = None
    return str(color_space or "")


def _get_texture_source_color_space(shader: UsdShade.Shader | None, file_attr: Usd.Attribute | None = None) -> str:
    source_color_space = None
    if shader is not None:
        inp = shader.GetInput("sourceColorSpace")
        if inp:
            try:
                source_color_space = inp.Get()
            except Exception:
                source_color_space = None
            if source_color_space is None:
                try:
                    attrs = UsdShade.Utils.GetValueProducingAttributes(inp)
                except Exception:
                    attrs = ()
                if attrs:
                    try:
                        source_color_space = attrs[0].Get()
                    except Exception:
                        source_color_space = None

    if isinstance(source_color_space, str) and source_color_space.strip().lower() == "auto":
        source_color_space = None

    if source_color_space is None and file_attr is not None:
        source_color_space = _get_attribute_color_space(file_attr)

    return str(source_color_space or "")


def _texture_source_is_linear(color_space: str) -> bool:
    token = color_space.strip().lower()
    return token in ("identity", "raw", "data", "linear", "lin_rec709_scene") or token.startswith("lin_")


def _color_to_display_space(
    color: tuple[float, float, float],
    attr: Usd.Attribute | None,
) -> tuple[float, float, float]:
    """Normalize a USD scalar color to Newton's display/sRGB model color space."""
    color_space = _get_attribute_color_space(attr)
    if not color_space or _texture_source_is_linear(color_space):
        return color_linear_to_srgb(color)
    return color


def _resolve_color_texture_asset(
    asset: Any,
    prim: Usd.Prim,
    asset_attr: Usd.Attribute | None,
    shader: UsdShade.Shader | None = None,
) -> str | np.ndarray | None:
    texture = _resolve_asset_path(asset, prim, asset_attr)
    if not texture:
        return None

    if not _texture_source_is_linear(_get_texture_source_color_space(shader, asset_attr)):
        return texture

    pixels = load_texture(texture)
    if pixels is None:
        return texture
    return linear_texture_to_srgb(pixels)


def _find_texture_in_shader(shader: UsdShade.Shader | None, prim: Usd.Prim) -> str | np.ndarray | None:
    """Search a shader network for a connected texture asset.

    Args:
        shader: The shader node to inspect.
        prim: The prim providing stage context for asset resolution.

    Returns:
        Resolved texture asset path, converted display texture array, or
        ``None`` if not found.
    """
    if shader is None:
        return None
    shader_id = shader.GetIdAttr().Get()
    if shader_id == "UsdUVTexture":
        file_input = shader.GetInput("file")
        if file_input:
            attrs = UsdShade.Utils.GetValueProducingAttributes(file_input)
            if attrs:
                asset = attrs[0].Get()
                return _resolve_color_texture_asset(asset, prim, attrs[0], shader)
            asset = file_input.Get()
            if asset:
                return _resolve_color_texture_asset(asset, prim, file_input.GetAttr(), shader)
        return None
    if shader_id == "UsdPreviewSurface":
        for input_name in ("diffuseColor", "baseColor"):
            shader_input = shader.GetInput(input_name)
            if shader_input:
                source = shader_input.GetConnectedSource()
                if source:
                    source_shader = UsdShade.Shader(source[0].GetPrim())
                    texture = _find_texture_in_shader(source_shader, prim)
                    if texture is not None:
                        return texture
    return None


def _get_input_value_and_attr(
    shader: UsdShade.Shader | None,
    names: tuple[str, ...],
) -> tuple[Any | None, Usd.Attribute | None]:
    """Fetch the effective input value and producing attribute from a shader."""
    if shader is None:
        return None, None
    try:
        if not shader.GetPrim().IsValid():
            return None, None
    except Exception:
        return None, None

    for name in names:
        inp = shader.GetInput(name)
        if inp is None:
            continue
        try:
            attrs = UsdShade.Utils.GetValueProducingAttributes(inp)
        except Exception:
            continue
        if attrs:
            value = attrs[0].Get()
            if value is not None:
                return value, attrs[0]
    return None, None


def _get_input_value(shader: UsdShade.Shader | None, names: tuple[str, ...]) -> Any | None:
    """Fetch the effective input value from a shader, following connections."""
    return _get_input_value_and_attr(shader, names)[0]


def _empty_material_properties() -> dict[str, Any]:
    """Return an empty material properties dictionary."""
    return {"color": None, "metallic": None, "roughness": None, "texture": None}


def _coerce_color(value: Any) -> tuple[float, float, float] | None:
    """Coerce a value to an RGB color tuple, or None if not possible."""
    if value is None:
        return None
    color_np = np.array(value, dtype=np.float32).reshape(-1)
    if color_np.size >= 3:
        return (float(color_np[0]), float(color_np[1]), float(color_np[2]))
    return None


def _coerce_float(value: Any) -> float | None:
    """Coerce a value to a float, or None if not possible."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_preview_surface_properties(shader: UsdShade.Shader | None, prim: Usd.Prim) -> dict[str, Any]:
    """Extract material properties from a UsdPreviewSurface shader.

    Args:
        shader: The UsdPreviewSurface shader node to inspect.
        prim: The prim providing stage context for asset resolution.

    Returns:
        Dictionary with ``color``, ``metallic``, ``roughness``, and ``texture``.
    """
    properties = _empty_material_properties()
    if shader is None:
        return properties
    shader_id = shader.GetIdAttr().Get()
    if shader_id != "UsdPreviewSurface":
        return properties

    color_input = shader.GetInput("baseColor") or shader.GetInput("diffuseColor")
    if color_input:
        source = color_input.GetConnectedSource()
        if source:
            source_shader = UsdShade.Shader(source[0].GetPrim())
            properties["texture"] = _find_texture_in_shader(source_shader, prim)
            if properties["texture"] is None:
                color_value, color_attr = _get_input_value_and_attr(
                    source_shader,
                    (
                        "diffuseColor",
                        "baseColor",
                        "diffuse_color",
                        "base_color",
                        "diffuse_color_constant",
                        "displayColor",
                    ),
                )
                coerced_color = _coerce_color(color_value)
                if coerced_color is not None:
                    properties["color"] = _color_to_display_space(coerced_color, color_attr)
        else:
            coerced_color = _coerce_color(color_input.Get())
            if coerced_color is not None:
                properties["color"] = _color_to_display_space(coerced_color, color_input.GetAttr())

    metallic_input = shader.GetInput("metallic")
    if metallic_input:
        try:
            has_metallic_source = metallic_input.HasConnectedSource()
        except Exception:
            has_metallic_source = False
        if has_metallic_source:
            source = metallic_input.GetConnectedSource()
            source_shader = UsdShade.Shader(source[0].GetPrim()) if source else None
            metallic_value = _get_input_value(source_shader, ("metallic", "metallic_constant"))
            properties["metallic"] = _coerce_float(metallic_value)
            if properties["metallic"] is None:
                warnings.warn(
                    "Metallic texture inputs are not yet supported; using scalar fallback.",
                    stacklevel=2,
                )
        else:
            properties["metallic"] = _coerce_float(metallic_input.Get())

    roughness_input = shader.GetInput("roughness")
    if roughness_input:
        try:
            has_roughness_source = roughness_input.HasConnectedSource()
        except Exception:
            has_roughness_source = False
        if has_roughness_source:
            source = roughness_input.GetConnectedSource()
            source_shader = UsdShade.Shader(source[0].GetPrim()) if source else None
            roughness_value = _get_input_value(
                source_shader,
                ("roughness", "roughness_constant", "reflection_roughness_constant"),
            )
            properties["roughness"] = _coerce_float(roughness_value)
            if properties["roughness"] is None:
                warnings.warn(
                    "Roughness texture inputs are not yet supported; using scalar fallback.",
                    stacklevel=2,
                )
        else:
            properties["roughness"] = _coerce_float(roughness_input.Get())

    return properties


def _extract_shader_properties(shader: UsdShade.Shader | None, prim: Usd.Prim) -> dict[str, Any]:
    """Extract common material properties from a shader node.

    This routine starts with UsdPreviewSurface parsing and then falls back to
    common input names used by other shader implementations.

    Args:
        shader: The shader node to inspect.
        prim: The prim providing stage context for asset resolution.

    Returns:
        Dictionary with ``color``, ``metallic``, ``roughness``, and ``texture``.
    """
    properties = _extract_preview_surface_properties(shader, prim)
    if shader is None:
        return properties
    try:
        if not shader.GetPrim().IsValid():
            return properties
    except Exception:
        return properties

    if properties["color"] is None:
        color_value, color_attr = _get_input_value_and_attr(
            shader,
            (
                "diffuse_color_constant",
                "diffuse_color",
                "diffuseColor",
                "base_color",
                "baseColor",
                "displayColor",
            ),
        )
        color = _coerce_color(color_value)
        if color is not None:
            properties["color"] = _color_to_display_space(color, color_attr)
    if properties["metallic"] is None:
        metallic_value = _get_input_value(shader, ("metallic_constant", "metallic"))
        properties["metallic"] = _coerce_float(metallic_value)
    if properties["roughness"] is None:
        roughness_value = _get_input_value(shader, ("reflection_roughness_constant", "roughness_constant", "roughness"))
        properties["roughness"] = _coerce_float(roughness_value)

    if properties["texture"] is None:
        for inp in shader.GetInputs():
            name = inp.GetBaseName()
            if inp.HasConnectedSource():
                source = inp.GetConnectedSource()
                source_shader = UsdShade.Shader(source[0].GetPrim())
                texture = _find_texture_in_shader(source_shader, prim)
                if texture is not None:
                    properties["texture"] = texture
                    break
            elif "file" in name or "texture" in name:
                asset = inp.Get()
                if asset:
                    properties["texture"] = _resolve_color_texture_asset(asset, prim, inp.GetAttr())
                    break

    return properties


def _extract_material_input_properties(material: UsdShade.Material | None, prim: Usd.Prim) -> dict[str, Any]:
    """Extract material properties from inputs on a UsdShade.Material prim.

    This supports assets that author texture references directly on the Material,
    without creating a shader network.
    """
    properties = _empty_material_properties()
    if material is None:
        return properties

    for inp in material.GetInputs():
        name = inp.GetBaseName()
        name_lower = name.lower()
        try:
            if inp.HasConnectedSource():
                continue
        except Exception:
            continue
        value = inp.Get()
        if value is None:
            continue

        if properties["texture"] is None and ("texture" in name_lower or "file" in name_lower):
            texture = _resolve_color_texture_asset(value, prim, inp.GetAttr())
            if texture is not None:
                properties["texture"] = texture
                continue

        if properties["color"] is None and name_lower in (
            "diffusecolor",
            "basecolor",
            "diffuse_color",
            "base_color",
            "displaycolor",
        ):
            color = _coerce_color(value)
            if color is not None:
                properties["color"] = _color_to_display_space(color, inp.GetAttr())
                continue

        if properties["metallic"] is None and name_lower in ("metallic", "metallic_constant"):
            metallic = _coerce_float(value)
            if metallic is not None:
                properties["metallic"] = metallic
                continue

        if properties["roughness"] is None and name_lower in (
            "roughness",
            "roughness_constant",
            "reflection_roughness_constant",
        ):
            roughness = _coerce_float(value)
            if roughness is not None:
                properties["roughness"] = roughness

    return properties


def _get_bound_material(target_prim: Usd.Prim) -> UsdShade.Material | None:
    """Get the material bound to a prim.

    Resolution is UsdShade's canonical ``ComputeBoundMaterial``: ancestor bindings, binding
    strength, collection-based bindings, and purpose all follow the USD spec instead of a
    partial reimplementation. Prims that author ``material:binding`` without applying
    ``MaterialBindingAPI`` are invalid USD; resolution still tolerates them but emits one
    ``TfWarn`` per query (there is no Python-side suppression API, and instanced prims cannot
    be normalized in-session). Import caches material resolution per prim, so the warning
    volume is bounded by the number of non-conformant prims — fix such assets at source with
    ``usdchecker`` or ``usd-validation-nvidia``.
    """
    if not target_prim or not target_prim.IsValid():
        return None
    bound_material, _ = UsdShade.MaterialBindingAPI(target_prim).ComputeBoundMaterial()
    return bound_material if bound_material else None


def _resolve_prim_material_properties(target_prim: Usd.Prim) -> dict[str, Any] | None:
    """Resolve material properties from a prim's bound material.

    Returns None if no material is bound or no properties could be extracted.
    """
    material = _get_bound_material(target_prim)
    if not material:
        return None

    surface_output = material.GetSurfaceOutput()
    if not surface_output:
        surface_output = material.GetOutput("surface")
    if not surface_output:
        surface_output = material.GetOutput("mdl:surface")

    source_shader = None
    if surface_output:
        source = surface_output.GetConnectedSource()
        if source:
            source_shader = UsdShade.Shader(source[0].GetPrim())

    if source_shader is None:
        # Fallback: scan material children for a shader node (MDL-style materials).
        for child in material.GetPrim().GetChildren():
            if child.IsA(UsdShade.Shader):
                source_shader = UsdShade.Shader(child)
                break

    if source_shader is None:
        material_props = _extract_material_input_properties(material, target_prim)
        if any(value is not None for value in material_props.values()):
            return material_props
        return None

    # Always call _extract_shader_properties even if shader_id is None because
    # it has fallback logic for common shader input names.
    properties = _extract_shader_properties(source_shader, target_prim)
    material_props = _extract_material_input_properties(material, target_prim)
    for key in ("texture", "color", "metallic", "roughness"):
        if properties.get(key) is None and material_props.get(key) is not None:
            properties[key] = material_props[key]
    if properties["color"] is None and properties["texture"] is None:
        display_color = UsdGeom.PrimvarsAPI(target_prim).GetPrimvar("displayColor")
        if display_color:
            color = _coerce_color(display_color.Get())
            if color is not None:
                properties["color"] = _color_to_display_space(color, display_color.GetAttr())

    return properties


def resolve_material_properties_for_prim(prim: Usd.Prim) -> dict[str, Any]:
    """Resolve surface material properties bound to a prim.

    Args:
        prim: The prim whose bound material should be inspected.

    Returns:
        Dictionary with ``color``, ``metallic``, ``roughness``, and ``texture``.
    """
    if not prim or not prim.IsValid():
        return _empty_material_properties()

    properties = _resolve_prim_material_properties(prim)
    if properties is not None:
        return properties

    proto_prim = None
    try:
        if prim.IsInstanceProxy():
            proto_prim = prim.GetPrimInPrototype()
        elif prim.IsInstance():
            proto_prim = prim.GetPrototype()
    except Exception:
        proto_prim = None
    if proto_prim and proto_prim.IsValid():
        properties = _resolve_prim_material_properties(proto_prim)
        if properties is not None:
            return properties

    if UsdGeom is not None:
        try:
            is_mesh = prim.IsA(UsdGeom.Mesh)
        except Exception:
            is_mesh = False
        if is_mesh:
            fallback_props = None
            for child in prim.GetChildren():
                try:
                    is_subset = child.IsA(UsdGeom.Subset)
                except Exception:
                    is_subset = False
                if not is_subset:
                    continue
                subset_props = _resolve_prim_material_properties(child)
                if subset_props is None:
                    continue
                if subset_props.get("texture") is not None or subset_props.get("color") is not None:
                    return subset_props
                if fallback_props is None:
                    fallback_props = subset_props
            if fallback_props is not None:
                return fallback_props

    return _empty_material_properties()


def get_gaussian(prim: Usd.Prim, min_response: float = 0.1) -> Gaussian:
    """Load Gaussian splat data from a USD prim.

    Reads positions from attributes: `positions`, `orientations`, `scales`, `opacities` and `radiance:sphericalHarmonicsCoefficients`.

    Args:
        prim: A USD prim containing Gaussian splat data.
        min_response: Min response (default = 0.1).

    Returns:
        A new :class:`Gaussian` instance.
    """

    def _get_float_array_attr(name):
        attr = prim.GetAttribute(name)
        if attr and attr.HasValue():
            return np.array(attr.Get(), dtype=np.float32)

        attr = prim.GetAttribute(f"{name}h")
        if attr and attr.HasValue():
            return np.array(attr.Get(), dtype=np.float32)

        return None

    positions = _get_float_array_attr("positions")
    if positions is None:
        raise ValueError("USD Gaussian prim is missing required 'positions' attribute")

    sorting_mode = Gaussian.SortingMode.RAY_HIT_DISTANCE
    if usd_sorting_mode := get_attribute(prim, "sortingModeHint"):
        if usd_sorting_mode == "zDepth":
            sorting_mode = Gaussian.SortingMode.Z_DEPTH
        elif usd_sorting_mode == "cameraDistance":
            sorting_mode = Gaussian.SortingMode.CAMERA_DISTANCE
        elif usd_sorting_mode == "rayHitDistance":
            sorting_mode = Gaussian.SortingMode.RAY_HIT_DISTANCE
        else:
            raise ValueError(f"Unsupported gaussian sorting mode: {usd_sorting_mode}")

    return Gaussian(
        positions=positions,
        rotations=_get_float_array_attr("orientations"),
        scales=_get_float_array_attr("scales"),
        opacities=_get_float_array_attr("opacities"),
        sh_coeffs=_get_float_array_attr("radiance:sphericalHarmonicsCoefficients"),
        sh_degree=get_attribute(prim, "radiance:sphericalHarmonicsDegree"),
        min_response=min_response,
        sorting_mode=sorting_mode,
    )
