# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read USD triangle meshes, including topology, shading, and merged sources."""

from __future__ import annotations

import logging
import math
import os
import warnings
from collections.abc import Sequence
from typing import Any, Literal, overload

import numpy as np

from ..geometry import Mesh
from ..utils.deprecation import deprecate_nonkeyword_arguments

logger = logging.getLogger("newton")

try:
    from pxr import Sdf, Usd, UsdGeom, UsdShade
except ImportError:
    Sdf = None
    Usd = None
    UsdGeom = None
    UsdShade = None


def _create_mesh(source=None, *, prim=None, **kwargs) -> Mesh:
    """Load a Mesh through the class factory's argument and return conventions."""
    if prim is not None:
        if source is not None:
            raise TypeError("Mesh.create_from_usd() received both 'source' and legacy 'prim'; pass only one.")
        source = prim

    result = get_mesh(source, **kwargs)
    if isinstance(result, tuple):
        return result[0]
    return result


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
    load_visual_materials: bool,
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
            load_visual_materials=load_visual_materials,
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
        texture_transform=material_source.texture_transform
        if material_source is not None
        else ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )


@overload
def get_mesh(
    source: Usd.Prim | Usd.Stage | str | os.PathLike[str],
    *,
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
    load_visual_materials: bool = True,
) -> Mesh: ...


@overload
def get_mesh(
    source: Usd.Prim,
    *,
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
    load_visual_materials: bool = True,
) -> tuple[Mesh, np.ndarray | None]: ...


@overload
def get_mesh(
    source: None = None,
    *,
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
    prim: Usd.Prim,
    load_visual_materials: bool = True,
) -> Mesh: ...


@overload
def get_mesh(
    source: None = None,
    *,
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
    prim: Usd.Prim,
    load_visual_materials: bool = True,
) -> tuple[Mesh, np.ndarray | None]: ...


@deprecate_nonkeyword_arguments
def get_mesh(
    source: Usd.Prim | Usd.Stage | str | os.PathLike[str] | None = None,
    *,
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
    prim: Usd.Prim | None = None,
    load_visual_materials: bool = True,
) -> Mesh | tuple[Mesh, np.ndarray | None]:
    """
    Load a triangle mesh from a USD mesh prim, stage, file path, or URL.

    When ``source`` is a mesh prim, the mesh is loaded in the prim's local
    coordinates. When ``source`` is a stage, path, URL, or non-mesh prim, all
    ``UsdGeom.Mesh`` prims under ``root_path`` are merged into one
    :class:`newton.Mesh` with authored transforms applied relative to that root.

    With ``load_normals=True``, shading is resolved to per-vertex normals on
    the triangulated mesh. Missing normals are faceted for ``none`` and
    ``bilinear`` subdivision schemes and smooth otherwise. Faceted shading
    duplicates triangle vertices. Source subdivision control surfaces are
    not evaluated or retained; viewers receive final triangles and normals.
    Use ``load_normals=False`` for geometry-only loading without normal-driven
    vertex splitting.

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
        load_normals: Whether to load authored normals or generate missing
            normals and convert them to the per-vertex representation used by
            :class:`Mesh`. This may split vertices to represent sharp shading.
        load_uvs: Whether to load the UVs.
        maxhullvert: The maximum number of vertices for the convex hull approximation.
        face_varying_normal_conversion:
            This argument specifies how to convert authored "uniform" or "faceVarying" normals
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

        vertex_splitting_angle_threshold_deg: The threshold angle in degrees for splitting vertices based on authored uniform or faceVarying normals when ``face_varying_normal_conversion`` is "vertex_splitting". Corners whose normals differ by more than ``vertex_splitting_angle_threshold_deg`` will be split
            into different vertex clusters. Lower = more splits (sharper), higher = fewer splits (smoother).
        preserve_facevarying_uvs: If True, keep faceVarying UVs in their
            original corner layout and avoid UV-driven vertex splitting. The
            returned mesh can still split vertices for normals when
            ``load_normals=True``. This is useful when the
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
        load_visual_materials: If True, resolve the mesh's visual material and
            populate :attr:`newton.Mesh.color`, :attr:`newton.Mesh.texture`,
            :attr:`newton.Mesh.metallic`, and :attr:`newton.Mesh.roughness`.
            Resolution also covers materials bound through an instance
            prototype or a ``UsdGeom.Subset`` child, and the ``displayColor``
            primvar fallback. If False, those attributes keep their
            :class:`newton.Mesh` defaults; set it to False when only mesh
            geometry is needed. If ``load_uvs`` is True, the material's shader
            network may still be inspected to select the UV primvar used by
            the texture.

    Returns:
        newton.Mesh: The loaded mesh, or ``(mesh, uv_indices)`` if
        ``return_uv_indices`` is True.
    """
    from .utils import resolve_material_properties_for_prim  # noqa: PLC0415

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
            load_visual_materials=load_visual_materials,
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
    source_points = points
    source_indices = indices

    uvs = None
    uvs_interpolation = None
    # Tracks whether we already duplicated vertices (and per-vertex UVs) during
    # faceVarying normal conversion, so we don't split again in the UV pass.
    did_split_vertices = False
    if load_uvs:
        uv_primvar = _find_uv_primvar(prim)
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
        prim_path = str(prim.GetPath())
        normals = np.array(normals, dtype=np.float64)
        if normal_indices is not None and len(normal_indices) > 0:
            normals = _expand_indexed_primvar(normals, normal_indices, "Normal", prim_path)

        if normals_interpolation == UsdGeom.Tokens.constant:
            if len(normals) != 1:
                raise ValueError(f"Length of constant normals ({len(normals)}) must be 1 for mesh {prim_path}")
            normals = np.repeat(normals, len(points), axis=0)
        elif normals_interpolation == UsdGeom.Tokens.uniform:
            if len(normals) != len(counts):
                raise ValueError(
                    f"Length of uniform normals ({len(normals)}) does not match number of faces ({len(counts)}) "
                    f"for mesh {prim_path}"
                )
            normals = np.repeat(normals, np.asarray(counts, dtype=np.int32), axis=0)
            normals_interpolation = UsdGeom.Tokens.faceVarying
        elif normals_interpolation in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying):
            if len(normals) != len(points):
                raise ValueError(
                    f"Length of {normals_interpolation} normals ({len(normals)}) does not match number of points "
                    f"({len(points)}) for mesh {prim_path}"
                )
        elif normals_interpolation != UsdGeom.Tokens.faceVarying:
            raise ValueError(f"Unsupported normals interpolation '{normals_interpolation}' for mesh {prim_path}")

        if normals_interpolation == UsdGeom.Tokens.faceVarying:
            # Face-varying values must match the number of mesh corners.
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

                # faceVarying UVs carry one value per corner; if the count does
                # not match, they can't be indexed per-corner, so drop them
                # (matching the non-splitting UV path below).
                uvs_facevarying = uvs is not None and uvs_interpolation == UsdGeom.Tokens.faceVarying
                if uvs_facevarying and len(uvs) != C:
                    logger.info(
                        "Mesh %s: UV primvar length (%d) does not match corner count (%d); dropping UVs.",
                        prim.GetPath(),
                        len(uvs),
                        C,
                    )
                    uvs = None
                    uvs_facevarying = False

                # Corners that share a smooth normal but carry different faceVarying UVs
                # lie on a texture seam and must not be merged, or the seam's UVs would
                # collapse onto one value.
                if uvs is None:
                    corner_uvs = None
                else:
                    corner_uvs = np.asarray(uvs).reshape(len(uvs), -1)
                    if not uvs_facevarying:
                        corner_uvs = corner_uvs[indices]

                points, indices, normals, uvs = _split_corners_into_vertices(
                    points, indices, Ndir, corner_uvs, vertex_splitting_angle_threshold_deg
                )
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

    flip_winding = False
    orientation_attr = mesh.GetOrientationAttr()
    if orientation_attr:
        handedness = orientation_attr.Get()
        if handedness and handedness.lower() == "lefthanded":
            flip_winding = True
    corner_flat = _triangulate_face_varying_indices(counts, flip_winding)
    faces = indices[corner_flat].reshape(-1, 3)

    generate_flat_normals = False
    if load_normals and normals is None:
        generate_flat_normals = mesh.GetSubdivisionSchemeAttr().Get() in (UsdGeom.Tokens.none, UsdGeom.Tokens.bilinear)
        if not generate_flat_normals:
            from ..utils.mesh import compute_vertex_normals  # noqa: PLC0415

            # Compute before UV expansion so texture seams do not become shading seams.
            normals = compute_vertex_normals(points, faces)

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

    if generate_flat_normals:
        triangle_points = points[faces]
        face_normals = np.cross(
            triangle_points[:, 1] - triangle_points[:, 0], triangle_points[:, 2] - triangle_points[:, 0]
        )
        face_normals /= np.clip(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-20, None)
        normals = np.repeat(face_normals, 3, axis=0)
        vertex_indices = faces.reshape(-1)
        points = points[vertex_indices]
        if (
            uvs is not None
            and uv_indices is None
            and not (preserve_facevarying_uvs and uvs_interpolation == UsdGeom.Tokens.faceVarying)
        ):
            uvs = uvs[vertex_indices]
        faces = np.arange(len(vertex_indices), dtype=np.int32).reshape(-1, 3)

    if return_uv_indices and uvs is not None and uv_indices is None:
        uv_indices = faces.reshape(-1)

    if normals is not None and len(normals) != len(points):
        raise ValueError(
            f"Canonicalized normals length ({len(normals)}) does not match vertex count ({len(points)}) "
            f"for mesh {prim.GetPath()}"
        )

    material_props = resolve_material_properties_for_prim(prim) if load_visual_materials else {}

    visual_topology = points is not source_points
    mesh_out = Mesh(
        points,
        faces.flatten(),
        normals=normals,
        uvs=uvs,
        maxhullvert=maxhullvert,
        compute_inertia=compute_inertia and not visual_topology,
        color=material_props.get("color"),
        opacity=material_props.get("opacity"),
        texture=material_props.get("texture"),
        metallic=material_props.get("metallic"),
        roughness=material_props.get("roughness"),
        texture_transform=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        if material_props.get("texture_transform") is None
        else material_props["texture_transform"],
    )
    if compute_inertia and visual_topology:
        from ..geometry.inertia import compute_inertia_mesh  # noqa: PLC0415

        mesh_out.mass, mesh_out.com, mesh_out.inertia, _ = compute_inertia_mesh(
            1.0,
            np.asarray(source_points, dtype=np.float32),
            source_indices[corner_flat],
            is_solid=mesh_out.is_solid,
        )
        mesh_out.has_inertia = True
    if return_uv_indices:
        return mesh_out, uv_indices
    return mesh_out


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


def _fan_triangulation_corner_indices(counts: Sequence[int]) -> np.ndarray:
    """Return polygon-corner indices for fan-triangulated faces."""
    counts = np.asarray(counts, dtype=np.int32)
    triangle_counts = counts - 2
    num_tris = int(np.sum(triangle_counts))
    if num_tris <= 0:
        return np.zeros((0, 3), dtype=np.int32)

    tri_face_ids = np.repeat(np.arange(len(counts), dtype=np.int32), triangle_counts)
    tri_group_starts = np.cumsum(triangle_counts, dtype=np.int32) - triangle_counts
    tri_local_ids = np.arange(num_tris, dtype=np.int32) - np.repeat(tri_group_starts, triangle_counts)
    face_bases = np.concatenate([[0], np.cumsum(counts[:-1], dtype=np.int32)])

    corners = np.empty((num_tris, 3), dtype=np.int32)
    corners[:, 0] = face_bases[tri_face_ids]
    corners[:, 1] = face_bases[tri_face_ids] + tri_local_ids + 1
    corners[:, 2] = face_bases[tri_face_ids] + tri_local_ids + 2
    return corners


def fan_triangulate_faces(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Perform fan triangulation on polygonal faces.

    Args:
        counts: Array of vertex counts per face
        indices: Flattened array of vertex indices

    Returns:
        Array of shape (num_triangles, 3) containing triangle indices (dtype=np.int32)
    """
    indices = np.asarray(indices, dtype=np.int32)
    return indices[_fan_triangulation_corner_indices(counts)]


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


def _split_corners_into_vertices(
    points: np.ndarray,
    indices: np.ndarray,
    corner_dirs: np.ndarray,
    corner_uvs: np.ndarray | None,
    angle_threshold_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Duplicate vertices whose faceVarying corners disagree in normal direction or UV.

    The corners of a vertex are clustered greedily in corner order: a corner joins the
    first cluster whose mean direction is within ``angle_threshold_deg`` and whose UV
    matches, otherwise it starts a new cluster. Every cluster becomes one output vertex,
    numbered by the corner that created it.

    Most vertices resolve to a single cluster, which is decided for all of them at once
    with array operations; only the vertices that fail that test run the sequential
    clustering. A vertex whose corner directions all lie within half the threshold angle
    of their mean lie within the full threshold of each other, so the sequential pass
    would place every one of them in the first cluster.

    A corner whose angle to the cluster mean is exactly ``angle_threshold_deg`` may fall
    on either side of the comparison: the dot product is rounded differently depending on
    the arithmetic used, so which cluster such a corner lands in is not defined beyond
    "one of the clusters it is within the threshold of".

    Args:
        points: Source vertex positions, shape [vertex_count, 3].
        indices: Face-corner vertex indices, shape [corner_count].
        corner_dirs: Unit corner normals, shape [corner_count, 3].
        corner_uvs: Per-corner UVs, shape [corner_count, channels], or ``None``.
        angle_threshold_deg: Maximum angle between a corner normal and its cluster mean [deg].

    Returns:
        Split positions, the remapped corner indices, the per-vertex normals, and the
        per-vertex UVs (``None`` when ``corner_uvs`` is ``None``).
    """
    corner_count = len(indices)
    if corner_count == 0:
        empty_uvs = None if corner_uvs is None else corner_uvs[:0]
        return points[:0], indices.copy(), np.zeros((0, 3), dtype=np.float32), empty_uvs

    cos_thresh = math.cos(math.radians(angle_threshold_deg))
    cos_half_thresh = math.cos(math.radians(angle_threshold_deg) * 0.5)

    # Group the corners by vertex; the stable sort keeps corner order inside each group.
    order = np.argsort(indices, kind="stable")
    grouped_vertices = indices[order]
    grouped_dirs = corner_dirs[order]
    grouped_uvs = None if corner_uvs is None else corner_uvs[order]

    starts = np.flatnonzero(np.concatenate(([True], grouped_vertices[1:] != grouped_vertices[:-1])))
    group_sizes = np.diff(np.append(starts, corner_count))
    group_vertices = grouped_vertices[starts]
    group_of_corner = np.repeat(np.arange(len(starts)), group_sizes)

    # The normalized sum of a group's corner directions is the cluster mean it would end
    # up with if every corner joined the same cluster.
    dir_sums = np.add.reduceat(grouped_dirs, starts, axis=0)
    dir_means = dir_sums / np.clip(np.linalg.norm(dir_sums, axis=1, keepdims=True), 1e-30, None)
    dots = np.einsum("ij,ij->i", grouped_dirs, dir_means[group_of_corner])
    single_cluster = np.minimum.reduceat(dots, starts) >= cos_half_thresh
    if grouped_uvs is not None:
        same_uv = np.all(grouped_uvs == grouped_uvs[starts][group_of_corner], axis=1)
        single_cluster &= np.logical_and.reduceat(same_uv, starts)

    # Provisional ids: the single-cluster groups first, then the sequential pass.
    simple_groups = np.flatnonzero(single_cluster)
    provisional_of_group = np.full(len(starts), -1, dtype=np.int64)
    provisional_of_group[simple_groups] = np.arange(len(simple_groups))
    provisional_of_corner = provisional_of_group[group_of_corner]

    next_id = len(simple_groups)
    split_creation: list[int] = []
    split_vertices: list[int] = []
    split_dir_sums: list[tuple[float, float, float]] = []
    split_uvs: list[tuple[float, ...]] = []
    for group in np.flatnonzero(~single_cluster):
        begin = int(starts[group])
        end = begin + int(group_sizes[group])
        source_vertex = int(group_vertices[group])
        clusters: list[list] = []
        for corner in range(begin, end):
            dir_x, dir_y, dir_z = (float(value) for value in grouped_dirs[corner])
            corner_uv = None if grouped_uvs is None else tuple(grouped_uvs[corner].tolist())
            for cluster in clusters:
                sum_x, sum_y, sum_z = cluster[0], cluster[1], cluster[2]
                # Scalar arithmetic rounds this dot product differently from the
                # equivalent NumPy expression (which uses FMA and a scaled norm), so a
                # corner sitting exactly on the threshold can cluster either way.
                scale = max(math.sqrt(sum_x * sum_x + sum_y * sum_y + sum_z * sum_z), 1e-30)
                if (sum_x * dir_x + sum_y * dir_y + sum_z * dir_z) / scale < cos_thresh:
                    continue
                if corner_uv is not None and cluster[3] != corner_uv:
                    continue
                cluster[0] = sum_x + dir_x
                cluster[1] = sum_y + dir_y
                cluster[2] = sum_z + dir_z
                provisional_of_corner[corner] = cluster[4]
                break
            else:
                clusters.append([dir_x, dir_y, dir_z, corner_uv, next_id])
                provisional_of_corner[corner] = next_id
                split_creation.append(int(order[corner]))
                split_vertices.append(source_vertex)
                if corner_uv is not None:
                    split_uvs.append(corner_uv)
                next_id += 1
        # Cluster order matches the id order assigned above.
        split_dir_sums.extend((cluster[0], cluster[1], cluster[2]) for cluster in clusters)

    creation_corners = np.concatenate([order[starts[simple_groups]], np.asarray(split_creation, dtype=np.int64)])
    source_vertices = np.concatenate([group_vertices[simple_groups], np.asarray(split_vertices, dtype=np.int64)])
    new_dir_sums = np.concatenate(
        [dir_sums[simple_groups], np.asarray(split_dir_sums, dtype=np.float64).reshape(-1, 3)]
    )

    # Number the output vertices by the corner that created them, matching the order a
    # purely sequential pass over the corners would produce.
    final_of_provisional = np.empty(len(creation_corners), dtype=np.int64)
    final_of_provisional[np.argsort(creation_corners, kind="stable")] = np.arange(len(creation_corners))
    provisional_in_final_order = np.argsort(final_of_provisional, kind="stable")

    new_indices = np.empty(corner_count, dtype=indices.dtype)
    new_indices[order] = final_of_provisional[provisional_of_corner]
    new_points = points[source_vertices[provisional_in_final_order]]
    ordered_sums = new_dir_sums[provisional_in_final_order]
    lengths = np.clip(np.linalg.norm(ordered_sums, axis=1, keepdims=True), 1e-30, None)
    new_normals = (ordered_sums / lengths).astype(np.float32)

    new_uvs = None
    if grouped_uvs is not None:
        split_uv_rows = np.asarray(split_uvs, dtype=grouped_uvs.dtype).reshape(-1, grouped_uvs.shape[1])
        new_uvs = np.concatenate([grouped_uvs[starts[simple_groups]], split_uv_rows])[provisional_in_final_order]

    return new_points, new_indices, new_normals, new_uvs


def _triangulate_face_varying_indices(counts: Sequence[int], flip_winding: bool) -> np.ndarray:
    """Return flattened corner indices for fan-triangulated face-varying data."""
    corner_faces = _fan_triangulation_corner_indices(counts)
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
        from ._asset_download import resolve_usd_from_url  # noqa: PLC0415

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


def _material_surface_shader(material: UsdShade.Material | None) -> UsdShade.Shader | None:
    """Return the surface shader driving a material (UsdPreviewSurface or MDL)."""
    if not material:
        return None
    surface_output = material.GetSurfaceOutput() or material.GetOutput("surface") or material.GetOutput("mdl:surface")
    if surface_output:
        source = surface_output.GetConnectedSource()
        if source:
            return UsdShade.Shader(source[0].GetPrim())
    for child in material.GetPrim().GetChildren():
        if child.IsA(UsdShade.Shader):
            return UsdShade.Shader(child)
    return None


def _uvtexture_reader_varname(texture_shader: UsdShade.Shader) -> str | None:
    """Return the primvar read by a ``UsdUVTexture``, through an optional ``UsdTransform2d``."""
    st_input = texture_shader.GetInput("st")
    source = st_input.GetConnectedSource() if st_input else None
    if not source:
        return None
    reader = UsdShade.Shader(source[0].GetPrim())
    if reader.GetIdAttr().Get() == "UsdTransform2d":
        transform_input = reader.GetInput("in")
        source = transform_input.GetConnectedSource() if transform_input else None
        if not source:
            return None
        reader = UsdShade.Shader(source[0].GetPrim())
    varname = reader.GetInput("varname")
    if not varname:
        return None
    value = varname.Get()
    if value is None:
        try:
            attrs = UsdShade.Utils.GetValueProducingAttributes(varname)
        except Exception:
            attrs = ()
        value = attrs[0].Get() if attrs else None
    return str(value) if value else None


def _uv_primvar_name_from_shader(shader: UsdShade.Shader | None) -> str | None:
    """Resolve the texcoord primvar name a shader's base-color texture reads, or ``None``.

    - ``UsdPreviewSurface``: follow the base-color ``UsdUVTexture``'s ``st`` input to its
      ``UsdPrimvarReader_float2`` and read ``inputs:varname``.
    - ``OmniPBR`` and other MDL shaders: ``st_<inputs:uv_space_index>``.
    """
    if shader is None:
        return None
    try:
        shader_id = shader.GetIdAttr().Get()
    except Exception:
        shader_id = None
    if shader_id == "UsdPreviewSurface":
        for color_name in ("baseColor", "diffuseColor"):
            color_input = shader.GetInput(color_name)
            source = color_input.GetConnectedSource() if color_input else None
            if not source:
                continue
            texture = UsdShade.Shader(source[0].GetPrim())
            if texture.GetIdAttr().Get() == "UsdUVTexture":
                name = _uvtexture_reader_varname(texture)
                if name:
                    return name
    uv_index_input = shader.GetInput("uv_space_index")
    if uv_index_input:
        value = uv_index_input.Get()
        if value is not None:
            return f"st_{int(value)}"
    return None


def _resolve_material_uv_primvar_name(prim: Usd.Prim) -> str | None:
    """Resolve the texcoord primvar name from the material(s) bound to a mesh or its subsets."""
    from .utils import _get_bound_material  # noqa: PLC0415

    candidate_prims = [prim]
    try:
        subsets = UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets()
        candidate_prims += [subset.GetPrim() for subset in subsets]
    except Exception:
        pass
    for candidate in candidate_prims:
        name = _uv_primvar_name_from_shader(_material_surface_shader(_get_bound_material(candidate)))
        if name:
            return name
    return None


def _find_uv_primvar(prim: Usd.Prim):
    """Return a mesh's texture-coordinate primvar, or ``None``.

    The primvar name is resolved from the bound material's shader network — the
    ``UsdPreviewSurface`` texture reader's ``varname`` or an MDL/OmniPBR
    ``uv_space_index`` — because a mesh may carry several UV sets and only the
    material identifies the correct one. Falls back to the conventional ``st``
    primvar, then to the first ``float2``/``texCoord2f`` primvar named ``st*``/``uv*``.
    """
    api = UsdGeom.PrimvarsAPI(prim)

    resolved = _resolve_material_uv_primvar_name(prim)
    if resolved:
        primvar = api.GetPrimvar(resolved)
        if primvar and primvar.HasValue():
            return primvar

    primvar = api.GetPrimvar("st")
    if primvar and primvar.HasValue():
        return primvar

    fallback = None
    for candidate in api.GetPrimvarsWithValues():
        if candidate.GetTypeName() not in (Sdf.ValueTypeNames.TexCoord2fArray, Sdf.ValueTypeNames.Float2Array):
            continue
        name = candidate.GetPrimvarName().lower()
        if name.startswith("st"):
            return candidate
        if (name.startswith("uv") or name == "map1") and fallback is None:
            fallback = candidate
    return fallback
