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

import os
import warnings
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import warp as wp

from ..core.types import Axis, AxisType, nparray
from ..geometry import MESH_MAXHULLVERT, Mesh
from ..sim.model import Model

AttributeAssignment = Model.AttributeAssignment
AttributeFrequency = Model.AttributeFrequency

if TYPE_CHECKING:
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
                out[attr.key] = value_to_warp(usd_attr.Get(), attr.dtype)
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


def fan_triangulate_faces(counts: nparray, indices: nparray) -> nparray:
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


@overload
def get_mesh(
    prim: Usd.Prim,
    load_normals: bool = False,
    load_uvs: bool = False,
    maxhullvert: int = MESH_MAXHULLVERT,
    face_varying_normal_conversion: Literal[
        "vertex_averaging", "angle_weighted", "vertex_splitting"
    ] = "vertex_splitting",
    vertex_splitting_angle_threshold_deg: float = 25.0,
    preserve_facevarying_uvs: bool = False,
    return_uv_indices: Literal[False] = False,
) -> Mesh: ...


@overload
def get_mesh(
    prim: Usd.Prim,
    load_normals: bool = False,
    load_uvs: bool = False,
    maxhullvert: int = MESH_MAXHULLVERT,
    face_varying_normal_conversion: Literal[
        "vertex_averaging", "angle_weighted", "vertex_splitting"
    ] = "vertex_splitting",
    vertex_splitting_angle_threshold_deg: float = 25.0,
    preserve_facevarying_uvs: bool = False,
    return_uv_indices: Literal[True] = True,
) -> tuple[Mesh, np.ndarray | None]: ...


def get_mesh(
    prim: Usd.Prim,
    load_normals: bool = False,
    load_uvs: bool = False,
    maxhullvert: int = MESH_MAXHULLVERT,
    face_varying_normal_conversion: Literal[
        "vertex_averaging", "angle_weighted", "vertex_splitting"
    ] = "vertex_splitting",
    vertex_splitting_angle_threshold_deg: float = 25.0,
    preserve_facevarying_uvs: bool = False,
    return_uv_indices: bool = False,
) -> Mesh | tuple[Mesh, np.ndarray | None]:
    """
    Load a triangle mesh from a USD prim that has the ``UsdGeom.Mesh`` schema.

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
        prim (Usd.Prim): The USD prim to load the mesh from.
        load_normals (bool): Whether to load the normals.
        load_uvs (bool): Whether to load the UVs.
        maxhullvert (int): The maximum number of vertices for the convex hull approximation.
        face_varying_normal_conversion (Literal["vertex_averaging", "angle_weighted", "vertex_splitting"]):
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

        vertex_splitting_angle_threshold_deg (float): The threshold angle in degrees for splitting vertices based on the face normals in case of faceVarying normals and ``face_varying_normal_conversion`` is "vertex_splitting". Corners whose normals differ by more than angle_deg will be split
            into different vertex clusters. Lower = more splits (sharper), higher = fewer splits (smoother).
        preserve_facevarying_uvs (bool): If True, keep faceVarying UVs in their
            original corner layout and avoid UV-driven vertex splitting. The
            returned mesh keeps its original topology. This is useful when the
            caller needs the original UV indexing (e.g., panel-space cloth).
        return_uv_indices (bool): If True, return a tuple ``(mesh, uv_indices)``
            where ``uv_indices`` is a flattened triangle index buffer for the
            UVs when available. For faceVarying UVs and
            ``preserve_facevarying_uvs=True``, these indices reference the
            face-varying UV array.

    Returns:
        newton.Mesh: The loaded mesh, or ``(mesh, uv_indices)`` if
        ``return_uv_indices`` is True.
    """

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
                warnings.warn(
                    f"UV primvar length ({len(uvs)}) does not match indices length ({len(indices)}) for mesh {prim.GetPath()}; "
                    "dropping UVs.",
                    stacklevel=2,
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

    mesh_out = Mesh(points, faces.flatten(), normals=normals, uvs=uvs, maxhullvert=maxhullvert)
    if return_uv_indices:
        return mesh_out, uv_indices
    return mesh_out


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


def _find_texture_in_shader(shader: UsdShade.Shader | None, prim: Usd.Prim) -> str | None:
    """Search a shader network for a connected texture asset.

    Args:
        shader: The shader node to inspect.
        prim: The prim providing stage context for asset resolution.

    Returns:
        Resolved texture asset path, or ``None`` if not found.
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
                return _resolve_asset_path(asset, prim, attrs[0])
        return None
    if shader_id == "UsdPreviewSurface":
        for input_name in ("diffuseColor", "baseColor"):
            shader_input = shader.GetInput(input_name)
            if shader_input:
                source = shader_input.GetConnectedSource()
                if source:
                    source_shader = UsdShade.Shader(source[0].GetPrim())
                    texture = _find_texture_in_shader(source_shader, prim)
                    if texture:
                        return texture
    return None


def _get_input_value(shader: UsdShade.Shader | None, names: tuple[str, ...]) -> Any | None:
    """Fetch the effective input value from a shader, following connections."""
    if shader is None:
        return None
    try:
        if not shader.GetPrim().IsValid():
            return None
    except Exception:
        return None

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
                return value
    return None


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
                color_value = _get_input_value(
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
                properties["color"] = _coerce_color(color_value)
        else:
            properties["color"] = _coerce_color(color_input.Get())

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
        color_value = _get_input_value(
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
        properties["color"] = _coerce_color(color_value)
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
                if texture:
                    properties["texture"] = texture
                    break
            elif "file" in name or "texture" in name:
                asset = inp.Get()
                if asset:
                    properties["texture"] = _resolve_asset_path(asset, prim, inp.GetAttr())
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
            texture = _resolve_asset_path(value, prim, inp.GetAttr())
            if texture:
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
                properties["color"] = color
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
    """Get the material bound to a prim."""
    if not target_prim or not target_prim.IsValid():
        return None
    if target_prim.HasAPI(UsdShade.MaterialBindingAPI):
        binding_api = UsdShade.MaterialBindingAPI(target_prim)
        bound_material, _ = binding_api.ComputeBoundMaterial()
        return bound_material

    # Some assets author material:binding relationships without applying MaterialBindingAPI.
    rels = [rel for rel in target_prim.GetRelationships() if rel.GetName().startswith("material:binding")]
    if not rels:
        return None
    rels.sort(
        key=lambda rel: 0
        if rel.GetName() == "material:binding"
        else 1
        if rel.GetName() == "material:binding:preview"
        else 2
    )
    for rel in rels:
        targets = rel.GetTargets()
        if targets:
            mat_prim = target_prim.GetStage().GetPrimAtPath(targets[0])
            if mat_prim and mat_prim.IsValid():
                return UsdShade.Material(mat_prim)
    return None


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

    # Always call _extract_shader_properties even if shader_id is None (e.g., for MDL shaders like OmniPBR)
    # because _extract_shader_properties has fallback logic for common input names
    properties = _extract_shader_properties(source_shader, target_prim)
    material_props = _extract_material_input_properties(material, target_prim)
    for key in ("texture", "color", "metallic", "roughness"):
        if properties.get(key) is None and material_props.get(key) is not None:
            properties[key] = material_props[key]
    if properties["color"] is None and properties["texture"] is None:
        display_color = UsdGeom.PrimvarsAPI(target_prim).GetPrimvar("displayColor")
        if display_color:
            properties["color"] = _coerce_color(display_color.Get())

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
