# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
import os
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, overload

import numpy as np
import warp as wp

from ..core.types import Axis, AxisType
from ..exceptions import NewtonDeprecationWarning, NewtonWarning
from ..geometry import Gaussian
from ..sim.model import Model
from ..utils.color import color_linear_to_srgb
from ..utils.import_usd_deformable_utils import (
    _AOUSD_DEFAULT_POISSONS_RATIO,
    _AOUSD_DEFAULT_YOUNGS_MODULUS,
    _is_usd_float_representable,
    _warn_geometry_authored_material_attrs,
)
from ..utils.texture import linear_texture_to_srgb, load_texture
from ._mesh_loading import (  # noqa: F401
    _get_mesh_from_source,
    corner_angles,
    fan_triangulate_faces,
    get_mesh,
)

logger = logging.getLogger(__name__)

AttributeAssignment = Model.AttributeAssignment
AttributeFrequency = Model.AttributeFrequency

if TYPE_CHECKING:
    from ..geometry.types import TetMesh
    from ..sim.builder import ModelBuilder

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    Usd = None
    Gf = None
    UsdGeom = None
    UsdPhysics = None
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
    mat = _get_xform_matrix(prim, local=local, xform_cache=xform_cache)
    # USD stores basis vectors in rows. Only their lengths are needed here;
    # constructing a Warp matrix and extracting its quaternion is unnecessary.
    # Match transform_decompose's host-side scalar arithmetic before sqrt.
    basis = mat[:3, :3].astype(np.float64)
    scale = np.sqrt(np.sum(basis * basis, axis=1).astype(np.float32))

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
                logger.warning(
                    "Custom attribute '%s' has invalid assignment or frequency in customData. Skipping.", attr_name
                )
                continue
        else:
            # No metadata found - skip with warning
            logger.warning(
                "Custom attribute '%s' is missing required customData (assignment and frequency). Skipping.", attr_name
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
        "displayOpacity",
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


def _should_load_tetmesh_material_for_import(prim: Usd.Prim) -> bool:
    """Keep legacy TetMesh material loading when the proposal pass does not own it."""
    return (
        not has_applied_api_schema(prim, "PhysicsVolumeDeformableSimAPI")
        or _material_authors_legacy_deformable_attrs(prim)
        or _material_authors_unscoped_canonical_attrs(prim)
    )


def _resolve_deformable_poissons_ratio(value: float, path: str, *, attr_namespace: str = "physics") -> float | None:
    """Validate a deformable Poisson ratio and apply Newton's compatibility approximation."""
    incompressible_approximation = 0.499
    if -1.0 < value < 0.5:
        return value
    if 0.5 <= value < 1.0:
        if value == 0.5:
            message = (
                "is incompressible and gives an infinite Lamé parameter; "
                f"approximating it as {incompressible_approximation:g} for Newton"
            )
        else:
            message = (
                "is outside the proposal range (-1, 0.5]; "
                f"approximating it as {incompressible_approximation:g} for Newton compatibility"
            )
        warnings.warn(
            f"{path}: {attr_namespace}:poissonsRatio={value:g} {message}.",
            NewtonWarning,
            stacklevel=2,
        )
        return incompressible_approximation

    warnings.warn(
        f"{path}: invalid {attr_namespace}:poissonsRatio {value:g} "
        "(expected a finite value with -1 < value <= 0.5); treating it as unauthored.",
        NewtonWarning,
        stacklevel=2,
    )
    return None


def _deformable_lame_parameters(youngs: float, poissons: float) -> tuple[float, float]:
    """Convert isotropic deformable properties to Lamé parameters."""
    k_mu = youngs / (2.0 * (1.0 + poissons))
    k_lambda = youngs * poissons / ((1.0 + poissons) * (1.0 - 2.0 * poissons))
    return k_mu, k_lambda


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
    those values are read and converted to Lamé parameters (``k_mu``,
    ``k_lambda``) and density on the returned TetMesh, expressed in the stage's
    configured units. Their SI-equivalent units are [Pa] for the Lamé parameters
    and [kg/m^3] for density. A material applying
    ``PhysicsVolumeDeformableMaterialAPI`` receives the proposal's elasticity
    fallbacks; API-less compatibility materials leave missing properties unset.

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
        A :class:`newton.TetMesh` with vertex positions and tet connectivity.
    """
    return _get_tetmesh(
        prim,
        compat_namespaces=compat_namespaces,
        load_custom_attributes=_load_custom_attributes,
        load_material=True,
    )


def _get_tetmesh(
    prim: Usd.Prim,
    *,
    compat_namespaces: Sequence[str] | None,
    load_custom_attributes: bool,
    load_material: bool,
) -> TetMesh:
    """Load a TetMesh with importer-specific data controls."""
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
    density_val = None
    raw_density = None

    # Volume material moduli (youngsModulus/poissonsRatio/...) belong on the bound material, not the
    # geometry; warn if authored on the TetMesh prim itself so the misplacement is visible to direct
    # get_tetmesh() callers too (add_usd's deformable pass warns separately).
    if load_material:
        _warn_geometry_authored_material_attrs(
            prim, str(prim.GetPath()), "PhysicsVolumeDeformableMaterialAPI", _read_physics_attr
        )

    material_prim = _find_physics_material_prim(prim) if load_material else None
    if compat_namespaces is None:
        # Deprecated legacy default: read vendor namespaces off any bound material, and
        # canonical moduli off materials without the deformable material API. Warn only when
        # that default is load-bearing -- vendor attrs authored, or canonical attrs on an
        # API-less material -- so materials whose reads the default change does not alter
        # (API-applied canonical, render-only) never warn.
        if load_material and (
            _material_authors_legacy_deformable_attrs(prim) or _material_authors_unscoped_canonical_attrs(prim)
        ):
            warnings.warn(
                "get_tetmesh(): the default reads deformable material attributes off any bound "
                "material (canonical physics: and legacy omniphysics: / physxDeformableBody: "
                "namespaces); this is deprecated, and a future release will read canonical "
                "physics: attributes only off a material applying "
                "PhysicsVolumeDeformableMaterialAPI. Pass compat_namespaces=() to adopt the "
                "canonical-only behavior now, or compat_namespaces="
                "newton.usd.DEFORMABLE_LEGACY_NAMESPACES to keep the current behavior explicitly.",
                NewtonDeprecationWarning,
                stacklevel=2,
            )
        compat_namespaces = DEFORMABLE_LEGACY_NAMESPACES
    # Canonical behavior (compat_namespaces=()) scopes the moduli to the volume deformable material
    # API, so they are not read off an unrelated physics material. A non-empty compat_namespaces reads
    # the listed vendor namespaces off any bound material.
    is_current_volume_material = material_prim is not None and has_applied_api_schema(
        material_prim, "PhysicsVolumeDeformableMaterialAPI"
    )
    read_material = material_prim is not None and (bool(compat_namespaces) or is_current_volume_material)
    if read_material:
        linear_unit = 1.0
        if UsdGeom.StageHasAuthoredMetersPerUnit(prim.GetStage()):
            linear_unit = float(UsdGeom.GetStageMetersPerUnit(prim.GetStage()))
        youngs = _coerce_deformable_float(
            _read_physics_attr(material_prim, "youngsModulus", compat_namespaces), material_prim, "youngsModulus"
        )
        poissons = _coerce_deformable_float(
            _read_physics_attr(material_prim, "poissonsRatio", compat_namespaces), material_prim, "poissonsRatio"
        )
        raw_density = _read_physics_attr(material_prim, "density", compat_namespaces)
        density_val = _coerce_deformable_float(raw_density, material_prim, "density")

        # The unregistered proposal schema cannot inject its sentinel fallback, so apply
        # the documented physical value only when the current material API owns the field.
        if youngs is not None:
            authored_youngs = youngs
            if authored_youngs == -math.inf:
                youngs = None
            elif math.isfinite(authored_youngs) and not _is_usd_float_representable(authored_youngs):
                warnings.warn(
                    f"{material_prim.GetPath()}: invalid physics:youngsModulus {authored_youngs:g} "
                    f"(outside the finite USD float range); treating it as unauthored.",
                    NewtonWarning,
                    stacklevel=2,
                )
                youngs = None
        if youngs is None and is_current_volume_material:
            youngs = _AOUSD_DEFAULT_YOUNGS_MODULUS * linear_unit
        if poissons is not None:
            poissons = _resolve_deformable_poissons_ratio(poissons, str(material_prim.GetPath()))
        if youngs is not None:
            E = youngs
            nu = _AOUSD_DEFAULT_POISSONS_RATIO if poissons is None else poissons
            if not (math.isfinite(E) and E >= 0.0):
                warnings.warn(
                    f"{material_prim.GetPath()}: invalid physics:youngsModulus {E:g} "
                    f"(expected a finite value or the -inf sentinel); treating it as unauthored.",
                    NewtonWarning,
                    stacklevel=2,
                )
                E = None
                if is_current_volume_material:
                    E = _AOUSD_DEFAULT_YOUNGS_MODULUS * linear_unit
            if E is not None:
                k_mu, k_lambda = _deformable_lame_parameters(E, nu)

        if density_val is not None:
            authored_density = density_val
            # The proposal declares density with a range of (0, inf) and a fallback of 0
            # meaning "ignored": zero falls through to the caller's density precedence
            # (body override, then the builder default) without being an invalid value.
            if _is_usd_float_representable(authored_density) and authored_density > 0.0:
                density = authored_density
            elif authored_density != 0.0:
                warnings.warn(
                    f"{material_prim.GetPath()}: invalid volume material density "
                    f"{authored_density}; expected a positive value in the finite USD float range, ignoring it.",
                    NewtonWarning,
                    stacklevel=2,
                )

    if density is None and raw_density is None:
        # The base UsdPhysicsMaterialAPI (which the family APIs extend) supplies
        # density too; a plain rigid-style physics material is a valid source.
        density = _get_physics_material_density(material_prim)

    if not load_custom_attributes:
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
        if name in ("st", "normals", "displayColor", "displayOpacity"):
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
        # Deformable physics-schema attributes (e.g. physics:masses, physics:restShapePoints)
        # are handled by the deformable importer; none are carried as generic mesh data here.
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
                NewtonWarning,
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
                warnings.warn(f"{prim.GetPath()}: {exc}; skipping the attribute.", NewtonWarning, stacklevel=2)
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


def _coerce_deformable_float(
    value: Any,
    prim: Usd.Prim,
    name: str,
    *,
    attr_namespace: str = "physics",
    warn_on_failure: bool = True,
) -> float | None:
    """Return a numeric deformable scalar, optionally warning when conversion fails."""
    if value is None:
        return None
    if isinstance(value, (bool, str, bytes)):
        result = None
    else:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            result = None
    if result is None and warn_on_failure:
        warnings.warn(
            f"{prim.GetPath()}: invalid {attr_namespace}:{name} {value!r} (expected a numeric scalar); "
            "treating it as unauthored.",
            NewtonWarning,
            stacklevel=2,
        )
    return result


def _resolve_attachment_gains(
    prim: Usd.Prim,
    stiffness_value: Any,
    damping_value: Any,
) -> tuple[float | None, float | None]:
    """Resolve attachment gains without emitting generic scalar warnings.

    The cable graph prepass must reject malformed gains silently so the attachment remains
    available to the post-pass. That post-pass preserves the authored values in metadata and emits
    one attachment-specific warning, so warning here would duplicate diagnostics. ``None`` retains
    the proposal's hard-stiffness and zero-damping fallbacks.
    """
    stiffness = (
        math.inf
        if stiffness_value is None
        else _coerce_deformable_float(stiffness_value, prim, "stiffness", warn_on_failure=False)
    )
    damping = (
        0.0
        if damping_value is None
        else _coerce_deformable_float(damping_value, prim, "damping", warn_on_failure=False)
    )
    return stiffness, damping


def _read_deformable_material(
    prim: Usd.Prim,
    read_attr: Callable[[Usd.Prim, str], Any],
    api_schema: str,
    attr_names: Sequence[str],
    *,
    attr_namespace: str = "physics",
    material_prim: Usd.Prim | None = None,
) -> dict[str, float] | None:
    """Read a per-family deformable material's authored, in-range parameters bound to a prim.

    Shared by the per-family readers (:func:`_get_curve_deformable_material`,
    :func:`_get_surface_deformable_material`): resolves the physics material bound via
    ``material:binding:physics`` and reads ``attr_names`` through ``read_attr`` (the resolver's
    single-source namespace read, see :meth:`SchemaResolverManager.read_deformable_attr`) when the
    bound material declares ``api_schema``.

    Returns a dict of the authored, resolved values among ``attr_names``, or ``None`` if the bound
    material does not declare ``api_schema``; an applied API with no valid authored values returns
    an empty dict. ``attr_namespace`` identifies the schema namespace in diagnostics; ``read_attr``
    remains responsible for the actual namespace resolution. ``material_prim`` may supply an
    already-resolved binding when one caller reads multiple APIs from the same material. Stiffness,
    damping, and Young's modulus accept zero; thickness must be positive; and density must be
    positive to be returned, while zero is its ignored sentinel. Poisson's ratios in the proposal
    range ``(-1, 0.5)`` are preserved; values in ``[0.5, 1)`` use Newton's warned ``0.499``
    compatibility approximation. The ``-inf`` simulator-default sentinel used by stiffness,
    damping, Young's modulus, and thickness is silently dropped. Other out-of-range or non-finite
    values are dropped with a warning.

    Args:
        prim: Prim whose bound physics material is resolved.
        read_attr: Callable that reads an attribute from the bound material.
        api_schema: Applied material API required on the bound material.
        attr_names: Attribute names to read and validate.
        attr_namespace: Namespace used when reporting invalid attributes.
        material_prim: Previously resolved bound material, if available.
    """
    if material_prim is None:
        material_prim = _find_physics_material_prim(prim)
    if material_prim is None or not has_applied_api_schema(material_prim, api_schema):
        return None
    out: dict[str, float] = {}
    for name in attr_names:
        val = read_attr(material_prim, name)
        val = _coerce_deformable_float(val, material_prim, name, attr_namespace=attr_namespace)
        if val is None:
            continue
        if name == "poissonsRatio":
            val = _resolve_deformable_poissons_ratio(
                val,
                str(material_prim.GetPath()),
                attr_namespace=attr_namespace,
            )
            if val is not None:
                out[name] = val
            continue
        has_negative_infinity_sentinel = name not in ("density", "poissonsRatio")
        if val == -math.inf and has_negative_infinity_sentinel:
            continue  # schema "simulator default" sentinel
        if not math.isfinite(val):
            expected = "a finite value or the -inf sentinel" if has_negative_infinity_sentinel else "a finite value"
            warnings.warn(
                f"{material_prim.GetPath()}: invalid {attr_namespace}:{name} {val:g} (expected {expected}); "
                f"treating it as unauthored.",
                NewtonWarning,
                stacklevel=2,
            )
            continue
        if not _is_usd_float_representable(val):
            warnings.warn(
                f"{material_prim.GetPath()}: invalid {attr_namespace}:{name} {val:g} "
                f"(outside the finite USD float range); treating it as unauthored.",
                NewtonWarning,
                stacklevel=2,
            )
            continue
        # Stiffness, damping, and Young's modulus accept [0, inf), so an authored zero is preserved.
        # Thickness and density must be strictly positive.
        if name in ("thickness", "surfaceThickness", "curvesThickness", "density"):
            if val > 0.0:
                out[name] = val
            elif name == "density" and val == 0.0:
                # The AOUSD deformables proposal defines zero density as an ignored sentinel.
                continue
            else:
                # A finite non-positive thickness (or negative density) is malformed, not the
                # unauthored sentinel (-inf); say it is dropped so users can tell it apart
                # from an unauthored value.
                warnings.warn(
                    f"{material_prim.GetPath()}: invalid {attr_namespace}:{name} {val:g} (expected > 0); "
                    f"treating it as unauthored.",
                    NewtonWarning,
                    stacklevel=2,
                )
        elif val >= 0.0:
            out[name] = val
        else:
            warnings.warn(
                f"{material_prim.GetPath()}: invalid {attr_namespace}:{name} {val:g} "
                f"(expected >= 0); treating it as unauthored.",
                NewtonWarning,
                stacklevel=2,
            )
    return out


_NEWTON_CURVE_DAMPING_ATTRS = (
    "curvesStretchDamping",
    "curvesShearDamping",
    "curvesBendDamping",
    "curvesTwistDamping",
)


def _get_curve_deformable_material(
    prim: Usd.Prim, read_attr: Callable[[Usd.Prim, str], Any]
) -> dict[str, float] | None:
    """Read curve-deformable (cable) ``PhysicsCurvesDeformableMaterialAPI`` parameters bound to a prim.

    Returns a dict of authored, in-range values from the current AOUSD curve material proposal,
    removed ``curvesThickness`` and the earlier unprefixed material attributes during their
    deprecation windows, and the four per-mode ``newton:curves*Damping`` values when
    ``NewtonCurvesDeformableMaterialAPI`` is also applied. Returns ``None`` if the bound material
    does not declare ``PhysicsCurvesDeformableMaterialAPI``. See
    :func:`_read_deformable_material` for value-validation rules.

    Args:
        prim: Curve prim whose bound physics material is read.
        read_attr: Callable that reads an AOUSD attribute from the bound material.
    """
    material_prim = _find_physics_material_prim(prim)
    if material_prim is None:
        return None

    material = _read_deformable_material(
        prim,
        read_attr,
        "PhysicsCurvesDeformableMaterialAPI",
        (
            "curvesThickness",
            "youngsModulus",
            "poissonsRatio",
            "curvesStretchStiffness",
            "curvesShearStiffness",
            "curvesBendStiffness",
            "curvesTwistStiffness",
            "density",
            # Compatibility with the proposal revision imported by Newton 1.4.
            "thickness",
            "stretchStiffness",
            "shearStiffness",
            "bendStiffness",
            "twistStiffness",
        ),
        material_prim=material_prim,
    )
    if material is None:
        return None

    newton_damping = _read_deformable_material(
        prim,
        lambda mat_prim, name: get_attribute(mat_prim, f"newton:{name}"),
        "NewtonCurvesDeformableMaterialAPI",
        _NEWTON_CURVE_DAMPING_ATTRS,
        attr_namespace="newton",
        material_prim=material_prim,
    )
    if newton_damping is not None:
        material.update(newton_damping)
    return material


def _get_volume_deformable_material(
    prim: Usd.Prim, read_attr: Callable[[Usd.Prim, str], Any]
) -> dict[str, float] | None:
    """Read volume-deformable material parameters bound to a prim."""
    return _read_deformable_material(
        prim,
        read_attr,
        "PhysicsVolumeDeformableMaterialAPI",
        ("youngsModulus", "poissonsRatio", "density"),
    )


def _get_surface_deformable_material(
    prim: Usd.Prim, read_attr: Callable[[Usd.Prim, str], Any]
) -> dict[str, float] | None:
    """Read surface-deformable (cloth) ``PhysicsSurfaceDeformableMaterialAPI`` parameters bound to a prim.

    Returns authored, in-range elasticity and structural stiffness values, plus removed
    ``surfaceThickness`` and earlier unprefixed attributes during their deprecation windows.
    Returns ``None`` if the bound material does not declare
    ``PhysicsSurfaceDeformableMaterialAPI``. See :func:`_read_deformable_material` for validation.
    """
    return _read_deformable_material(
        prim,
        read_attr,
        "PhysicsSurfaceDeformableMaterialAPI",
        (
            "surfaceThickness",
            "youngsModulus",
            "poissonsRatio",
            "surfaceStretchStiffness",
            "surfaceShearStiffness",
            "surfaceBendStiffness",
            "density",
            # Compatibility with the proposal revision imported by Newton 1.4.
            "thickness",
            "stretchStiffness",
            "shearStiffness",
            "bendStiffness",
        ),
    )


def load_physics_from_range(stage, root_paths, exclude_paths=()):
    """Parse a stage's native physics descriptors, across OpenUSD versions.

    OpenUSD 26.08 renamed ``UsdPhysics.LoadUsdPhysicsFromRange`` to
    ``UsdPhysics.UsdPhysicsLoadStageFromPrimRange`` and deprecated the old name, so calling
    it emits a ``DeprecationWarning``. Both spellings take the same arguments and return the
    same descriptor dict; prefer the new name where it exists.

    Args:
        stage: The USD stage to parse.
        root_paths: Roots of the subtrees to parse.
        exclude_paths: Prim paths whose subtrees should be excluded from the parse.

    Returns:
        The parser's object-type to descriptor mapping.
    """
    from pxr import UsdPhysics

    load = getattr(UsdPhysics, "UsdPhysicsLoadStageFromPrimRange", None)
    if load is None:
        load = UsdPhysics.LoadUsdPhysicsFromRange
    return load(stage, list(root_paths), excludePaths=list(exclude_paths))


def _get_physics_material_density(material_prim) -> float | None:
    """Read a bound material's base ``UsdPhysicsMaterialAPI`` density.

    The proposal reuses the rigid ``UsdPhysicsMaterialAPI`` for deformables, so a
    material applying only the base API still supplies density (the family
    material APIs extend it). Accepts a positive value in the finite USD float
    range; zero is the proposal's ignored fallback; other values warn and are
    ignored.
    """
    from pxr import UsdPhysics

    if material_prim is None or not (
        material_prim.HasAPI(UsdPhysics.MaterialAPI) or has_applied_api_schema(material_prim, "PhysicsMaterialAPI")
    ):
        return None
    attr = material_prim.GetAttribute("physics:density")
    value = attr.Get() if attr else None
    density = _coerce_deformable_float(value, material_prim, "density")
    if density is None:
        return None
    if _is_usd_float_representable(density) and density > 0.0:
        return density
    if density != 0.0:
        warnings.warn(
            f"{material_prim.GetPath()}: invalid physics material density {density}; "
            f"expected a positive value in the finite USD float range, ignoring it.",
            NewtonWarning,
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
                    NewtonWarning,
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
        ``(mass, density)`` with each entry ``None`` when unset, zero, or invalid.
    """
    body_prim = _find_deformable_body_prim(prim)
    if body_prim is None:
        return None, None

    def read_override(name: str) -> float | None:
        raw_value = read_attr(body_prim, name)
        value = _coerce_deformable_float(raw_value, body_prim, name)
        if value is None:
            return None
        if value == 0.0:
            return None
        if math.isfinite(value) and value > 0.0:
            return value
        warnings.warn(
            f"{body_prim.GetPath()}: invalid physics:{name} {value:g} "
            "(expected a finite positive value or the zero sentinel); treating it as unauthored.",
            NewtonWarning,
            stacklevel=2,
        )
        return None

    return read_override("mass"), read_override("density")


def _get_physics_scenes_from_results(stage: Usd.Stage, physics_results: dict[Any, Any]) -> list[UsdPhysics.Scene]:
    """Get physics scenes from parsed OpenUSD physics results."""
    scene_results = physics_results.get(UsdPhysics.ObjectType.Scene)
    if scene_results is None:
        return []

    scene_paths, _ = scene_results
    return [UsdPhysics.Scene.Get(stage, path) for path in scene_paths]


def get_physics_scenes(
    stage: Usd.Stage,
    root_path: str = "/",
    exclude_paths: Sequence[str] | None = None,
) -> list[UsdPhysics.Scene]:
    """Get physics scenes from a USD stage.

    The search uses OpenUSD's physics parser, including its instance-proxy
    traversal and subtree-pruning behavior.

    Args:
        stage: The USD stage to search.
        root_path: The root of the subtree to search.
        exclude_paths: Prim paths whose subtrees should be excluded from the search.

    Returns:
        Physics scenes in parser order.
    """
    physics_results = load_physics_from_range(stage, [root_path], exclude_paths or ())
    return _get_physics_scenes_from_results(stage, physics_results)


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
    return {
        "color": None,
        "opacity": None,
        "metallic": None,
        "roughness": None,
        "texture": None,
        "texture_transform": None,
    }


def _coerce_color(value: Any) -> tuple[float, float, float] | None:
    """Coerce a value to a clamped RGB color tuple, or None if not possible."""
    if value is None:
        return None
    # A per-vertex or per-face primvar holds one entry per element and only the leading one
    # is used, so take it before the numpy conversion rather than flattening the whole array.
    if hasattr(value, "__len__") and len(value) > 0 and hasattr(value[0], "__len__"):
        value = value[0]
    color_np = np.array(value, dtype=np.float32).reshape(-1)
    if color_np.size < 3:
        return None
    rgb = color_np[:3]
    color = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    if not np.all(np.isfinite(rgb)):
        warnings.warn(f"Ignoring non-finite imported color {color!r}.", NewtonWarning, stacklevel=2)
        return None
    # Builder triangle colors reject values outside [0, 1]; clamp like opacity so display-only data cannot abort an import.
    rgb = np.clip(rgb, 0.0, 1.0)
    clamped_color = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    if clamped_color != color:
        warnings.warn(
            f"Clamping imported color {color!r} to {clamped_color!r}.",
            NewtonWarning,
            stacklevel=2,
        )
    return clamped_color


def _coerce_float(value: Any) -> float | None:
    """Coerce a value to a float, or None if not possible."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_opacity(value: Any, *, warn_on_multiple: bool = False) -> float | None:
    """Coerce an opacity value to a clamped scalar, or None if not possible."""
    if value is None:
        return None
    try:
        value_np = np.array(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        value_np = np.asarray([_coerce_float(value)], dtype=np.float32)
    if value_np.size == 0:
        return None
    if value_np.size > 1 and warn_on_multiple:
        warnings.warn(
            f"Opacity data contains {value_np.size} values; using the first value as the shape opacity.",
            NewtonWarning,
            stacklevel=2,
        )
    opacity = float(value_np[0])
    if not np.isfinite(opacity):
        warnings.warn(f"Ignoring non-finite imported opacity {opacity!r}.", NewtonWarning, stacklevel=2)
        return None
    clamped_opacity = float(np.clip(opacity, 0.0, 1.0))
    if clamped_opacity != opacity:
        warnings.warn(
            f"Clamping imported opacity {opacity!r} to {clamped_opacity!r}.",
            NewtonWarning,
            stacklevel=2,
        )
    return clamped_opacity


def _coerce_color_opacity(value: Any) -> float | None:
    """Extract alpha only from a single four-component color value."""
    if value is None:
        return None
    try:
        value_np = np.array(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if value_np.size != 4:
        return None
    return _coerce_opacity(value_np[3])


def _extract_display_primvar_properties(prim: Usd.Prim) -> dict[str, Any]:
    """Extract displayColor/displayOpacity primvars from a prim."""
    properties = _empty_material_properties()
    if UsdGeom is None or not prim or not prim.IsValid():
        return properties
    primvars_api = UsdGeom.PrimvarsAPI(prim)

    display_color = primvars_api.FindPrimvarWithInheritance("displayColor")
    if display_color:
        color_value = display_color.Get()
        color = _coerce_color(color_value)
        if color is not None:
            properties["color"] = _color_to_display_space(color, display_color.GetAttr())

    display_opacity = primvars_api.FindPrimvarWithInheritance("displayOpacity")
    if display_opacity:
        opacity = _coerce_opacity(display_opacity.Get(), warn_on_multiple=True)
        if opacity is not None:
            properties["opacity"] = opacity

    if properties["opacity"] is None:
        attr = prim.GetAttribute("displayOpacity")
        if attr:
            properties["opacity"] = _coerce_opacity(attr.Get(), warn_on_multiple=True)

    return properties


def _coerce_vec2(value: Any) -> tuple[float, float] | None:
    """Coerce a value to a finite two-component tuple, or ``None``."""
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if result.size != 2 or not np.all(np.isfinite(result)):
        return None
    return (float(result[0]), float(result[1]))


def _extract_usd_transform2d(
    texture_shader: UsdShade.Shader,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Resolve the standard ``UsdTransform2d`` directly feeding a ``UsdUVTexture``."""
    st_input = texture_shader.GetInput("st")
    source = st_input.GetConnectedSource() if st_input else None
    if not source:
        return None
    transform = UsdShade.Shader(source[0].GetPrim())
    if transform.GetIdAttr().Get() != "UsdTransform2d":
        return None

    scale = _coerce_vec2(_get_input_value(transform, ("scale",))) or (1.0, 1.0)
    translation = _coerce_vec2(_get_input_value(transform, ("translation",))) or (0.0, 0.0)
    rotation = _coerce_float(_get_input_value(transform, ("rotation",)))
    rotation = rotation if rotation is not None and math.isfinite(rotation) else 0.0
    cosine, sine = math.cos(math.radians(rotation)), math.sin(math.radians(rotation))

    # UsdTransform2d is scale, then counter-clockwise rotation, then translation.
    return (
        (cosine * scale[0], -sine * scale[1], translation[0]),
        (sine * scale[0], cosine * scale[1], translation[1]),
    )


def _extract_preview_surface_properties(shader: UsdShade.Shader | None, prim: Usd.Prim) -> dict[str, Any]:
    """Extract material properties from a UsdPreviewSurface shader.

    Args:
        shader: The UsdPreviewSurface shader node to inspect.
        prim: The prim providing stage context for asset resolution.

    Returns:
        Dictionary with scalar surface properties (including opacity), texture
        data, and a standard texture-coordinate transform.
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
            if properties["texture"] is not None:
                properties["texture_transform"] = _extract_usd_transform2d(source_shader)
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
                if properties["opacity"] is None:
                    properties["opacity"] = _coerce_color_opacity(color_value)
        else:
            color_value = color_input.Get()
            coerced_color = _coerce_color(color_value)
            if coerced_color is not None:
                properties["color"] = _color_to_display_space(coerced_color, color_input.GetAttr())
            if properties["opacity"] is None:
                properties["opacity"] = _coerce_color_opacity(color_value)

    opacity_input = shader.GetInput("opacity") or shader.GetInput("alpha")
    if opacity_input:
        try:
            has_opacity_source = opacity_input.HasConnectedSource()
        except Exception:
            has_opacity_source = False
        if has_opacity_source:
            source = opacity_input.GetConnectedSource()
            source_shader = UsdShade.Shader(source[0].GetPrim()) if source else None
            opacity_value = _get_input_value(source_shader, ("opacity", "alpha", "opacity_constant", "alpha_constant"))
            properties["opacity"] = _coerce_opacity(opacity_value)
        else:
            properties["opacity"] = _coerce_opacity(opacity_input.Get())

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
                    NewtonWarning,
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
                    NewtonWarning,
                    stacklevel=2,
                )
        else:
            properties["roughness"] = _coerce_float(roughness_input.Get())

    return properties


def _output_channel_count(type_name: Sdf.ValueTypeName) -> int:
    """Return the component count of a shader output value type (float -> 1, float3 -> 3)."""
    default = type_name.defaultValue
    if hasattr(default, "__len__"):
        return len(default)
    return 1


# Base-color input name fragments used to identify the diffuse/albedo parameter
# feeding a surface shader. A texture connection's shape alone is ambiguous
# (a normal map is a 3-channel ``rgb`` connection like a diffuse map), so the
# input name is the disambiguator for both connected and direct-asset textures.
_COLOR_TEXTURE_INPUT_NAMES = ("diffuse", "albedo", "basecolor", "base_color", "displaycolor")


def _is_color_texture_input_name(base_name: str) -> bool:
    """Return whether an input name denotes a base-color/albedo texture parameter."""
    name = base_name.lower()
    return any(fragment in name for fragment in _COLOR_TEXTURE_INPUT_NAMES)


def _color_texture_from_input(surface_input: UsdShade.Input, prim: Usd.Prim) -> str | np.ndarray | None:
    """Return the base-color texture feeding a surface shader input, if any.

    The input must be a base-color/albedo parameter by name: a normal map is
    conventionally wired as ``UsdUVTexture.outputs:rgb`` too, so its 3-channel
    connection is indistinguishable from a diffuse map by shape alone. Two
    shapes are then supported:

    - A connected ``UsdUVTexture``: accepted only when the connected output is a
      multi-channel color output (``rgb`` / ``rgba``). Single-channel outputs
      (``r``/``g``/``b``/``a``) are scalar data maps (roughness, metallic, ...)
      and are ignored.
    - A direct asset value (e.g. an MDL ``diffuse_texture`` parameter): there is
      no texture node to inspect, so the base-color input is identified by name.
    """
    if not _is_color_texture_input_name(surface_input.GetBaseName()):
        return None
    try:
        producing = UsdShade.Utils.GetValueProducingAttributes(surface_input)
    except Exception:
        producing = ()
    for attr in producing:
        source_shader = UsdShade.Shader(attr.GetPrim())
        try:
            if source_shader.GetIdAttr().Get() != "UsdUVTexture":
                continue
        except Exception:
            continue
        if _output_channel_count(attr.GetTypeName()) < 3:
            continue
        texture = _find_texture_in_shader(source_shader, prim)
        if texture is not None:
            return texture

    try:
        connected = surface_input.HasConnectedSource()
    except Exception:
        connected = False
    if not connected:
        asset = surface_input.Get()
        if asset:
            return _resolve_color_texture_asset(asset, prim, surface_input.GetAttr())
    return None


def _extract_shader_properties(shader: UsdShade.Shader | None, prim: Usd.Prim) -> dict[str, Any]:
    """Extract common material properties from a shader node.

    This routine starts with UsdPreviewSurface parsing and then falls back to
    common input names used by other shader implementations.

    Args:
        shader: The shader node to inspect.
        prim: The prim providing stage context for asset resolution.

    Returns:
        Dictionary with scalar surface properties (including opacity), texture
        data, and a standard texture-coordinate transform.
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
        if properties["opacity"] is None:
            properties["opacity"] = _coerce_color_opacity(color_value)
    if properties["opacity"] is None:
        opacity_value = _get_input_value(
            shader,
            ("opacity", "alpha", "displayOpacity", "opacity_constant", "alpha_constant"),
        )
        properties["opacity"] = _coerce_opacity(opacity_value)
    if properties["metallic"] is None:
        metallic_value = _get_input_value(shader, ("metallic_constant", "metallic"))
        properties["metallic"] = _coerce_float(metallic_value)
    if properties["roughness"] is None:
        roughness_value = _get_input_value(shader, ("reflection_roughness_constant", "roughness_constant", "roughness"))
        properties["roughness"] = _coerce_float(roughness_value)

    if properties["texture"] is None:
        for inp in shader.GetInputs():
            texture = _color_texture_from_input(inp, prim)
            if texture is not None:
                properties["texture"] = texture
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
                if properties["opacity"] is None:
                    properties["opacity"] = _coerce_color_opacity(value)
                continue

        if properties["opacity"] is None and name_lower in (
            "opacity",
            "alpha",
            "displayopacity",
            "opacity_constant",
            "alpha_constant",
        ):
            opacity = _coerce_opacity(value)
            if opacity is not None:
                properties["opacity"] = opacity
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
    for key in ("texture", "color", "opacity", "metallic", "roughness"):
        if properties.get(key) is None and material_props.get(key) is not None:
            properties[key] = material_props[key]
    display_props = _extract_display_primvar_properties(target_prim)
    if properties["color"] is None and properties["texture"] is None and display_props.get("color") is not None:
        properties["color"] = display_props["color"]
    if properties["opacity"] is None and display_props.get("opacity") is not None:
        properties["opacity"] = display_props["opacity"]

    return properties


def resolve_material_properties_for_prim(prim: Usd.Prim) -> dict[str, Any]:
    """Resolve surface material properties bound to a prim.

    Args:
        prim: The prim whose bound material should be inspected.

    Returns:
        Dictionary with scalar surface properties (including opacity), texture
        data, and a standard texture-coordinate transform.
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
                if (
                    subset_props.get("texture") is not None
                    or subset_props.get("color") is not None
                    or subset_props.get("opacity") is not None
                ):
                    return subset_props
                if fallback_props is None:
                    fallback_props = subset_props
            if fallback_props is not None:
                return fallback_props

    display_props = _extract_display_primvar_properties(prim)
    if display_props.get("color") is not None or display_props.get("opacity") is not None:
        return display_props

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
