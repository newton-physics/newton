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
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import warp as wp

from ..core.types import Axis, AxisType, nparray
from ..geometry import MESH_MAXHULLVERT, Mesh
from ..sim.model import ModelAttributeAssignment, ModelAttributeFrequency

if TYPE_CHECKING:
    from ..sim.builder import ModelBuilder

try:
    from pxr import Gf, Usd, UsdGeom
except ImportError:
    Usd = None
    Gf = None
    UsdGeom = None


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


def get_transform_matrix(prim: Usd.Prim, local: bool = True) -> wp.mat44:
    """
    Extract the full transformation matrix from a USD Xform prim.

    Args:
        prim: The USD prim to query.
        local: If True, get the local transformation; if False, get the world transformation.

    Returns:
        A Warp 4x4 transform matrix.
    """
    xform = UsdGeom.Xformable(prim)

    if local:
        mat = np.array(xform.GetLocalTransformation(), dtype=np.float32)
    else:
        time = Usd.TimeCode.Default()
        mat = np.array(xform.ComputeLocalToWorldTransform(time), dtype=np.float32)
    return wp.mat44(mat.T)


def get_transform(prim: Usd.Prim, local: bool = True) -> wp.transform:
    """
    Extract the transform (position and rotation) from a USD Xform prim.

    Args:
        prim: The USD prim to query.
        local: If True, get the local transformation; if False, get the world transformation.

    Returns:
        A Warp transform containing the position and rotation extracted from the prim.
    """
    xform = UsdGeom.Xform(prim)
    if local:
        mat = np.array(xform.GetLocalTransformation(), dtype=np.float32)
    else:
        time = Usd.TimeCode.Default()
        mat = np.array(xform.ComputeLocalToWorldTransform(time), dtype=np.float32)
    rot = wp.quat_from_matrix(wp.mat33(mat[:3, :3].T.flatten()))
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


def get_mesh(
    prim: Usd.Prim,
    load_normals: bool = False,
    load_uvs: bool = False,
    maxhullvert: int = MESH_MAXHULLVERT,
    face_varying_normal_conversion: Literal[
        "vertex_averaging", "angle_weighted", "vertex_splitting"
    ] = "vertex_averaging",
    vertex_splitting_angle_threshold_deg: float = 25.0,
) -> Mesh:
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

    Returns:
        newton.Mesh: The loaded mesh.
    """

    mesh = UsdGeom.Mesh(prim)

    points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    counts = mesh.GetFaceVertexCountsAttr().Get()

    uvs = None
    if load_uvs:
        uv_primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        if uv_primvar:
            uvs = uv_primvar.Get()

    normals = None
    if load_normals:
        normals_attr = mesh.GetNormalsAttr()
        if normals_attr:
            normals = normals_attr.Get()

    if normals is not None:
        normals = np.array(normals, dtype=np.float64)
        if mesh.GetNormalsInterpolation() == "faceVarying":
            # compute vertex normals
            # try to read primvars:normals:indices (the primvar indexer)
            normals_index_attr = prim.GetAttribute("primvars:normals:indices")
            if normals_index_attr:
                normal_indices = np.array(normals_index_attr.Get(), dtype=np.int64)
                normals_fv = normals[normal_indices]  # (C,3) expanded
            else:
                # If faceVarying, values length must match number of corners
                if len(normals) != len(indices):
                    raise ValueError(
                        f"Length of normals ({len(normals)}) does not match length of indices ({len(indices)}) for mesh {prim.GetPath()}"
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
                def _new_vertex_from(v, n_dir):
                    new_vid = len(new_points)
                    new_points.append(points[v])
                    new_norm_sums.append(n_dir.copy())
                    clusters_per_v[v].append([n_dir.copy(), 1, new_vid])
                    if new_uvs is not None:
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
                        new_vid = _new_vertex_from(v, n_dir)
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

    if uvs is not None:
        uvs = np.array(uvs, dtype=np.float32)

    return Mesh(points, faces.flatten(), normals=normals, uvs=uvs, maxhullvert=maxhullvert)
