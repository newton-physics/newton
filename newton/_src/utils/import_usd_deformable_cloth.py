# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""USD surface-deformable (cloth) import pass.

Imports ``PhysicsSurfaceDeformableSimAPI`` polygon ``UsdGeom.Mesh`` prims as cloth, mapping the
surface material onto the isotropic membrane. Driven by :func:`.import_usd.parse_usd` via a
:class:`.import_usd_deformable_utils._DeformableImportContext`.
"""

from __future__ import annotations

import logging
import math
import warnings

import numpy as np
import warp as wp

from ..exceptions import NewtonDeprecationWarning, NewtonWarning
from .import_usd_deformable_utils import (
    _AOUSD_DEFAULT_POISSONS_RATIO,
    _AOUSD_DEFAULT_THICKNESS,
    _AOUSD_DEFAULT_YOUNGS_MODULUS,
    _apply_particle_masses,
    _bake_world_points,
    _deformable_body_skip_reason,
    _deformable_collision_enabled,
    _DeformableImportContext,
    _is_ignored_path,
    _read_deformable_element_array,
    _resolve_deformable_density,
    _rest_index_array,
    _rest_topology_matches,
    _skip_for_deformable_body_owner,
    _transform_rest_points,
    _warn_collision_approximated,
    _warn_collision_not_disableable,
    _warn_dropped_velocities,
    _warn_geometry_authored_material_attrs,
    _warn_subset_material_bindings,
    _world_matrix_reflects,
)
from .mesh import MeshAdjacency

logger = logging.getLogger(__name__)

# Attributes introduced after the family-prefix rename; density is shared and intentionally omitted.
_POST_RENAME_SURFACE_MATERIAL_ATTRS = (
    "surfaceThickness",
    "youngsModulus",
    "poissonsRatio",
    "surfaceStretchStiffness",
    "surfaceShearStiffness",
    "surfaceBendStiffness",
)
_LEGACY_SURFACE_MATERIAL_ATTRS = ("thickness", "stretchStiffness", "shearStiffness", "bendStiffness")


def _surface_thickness_samples(
    authored,
    tri_faces: np.ndarray,
    tri_source_faces: np.ndarray,
    point_count: int,
    fallback: float,
) -> tuple[list[float], list[float]]:
    """Resolve authored surface thicknesses at faces and points."""
    if authored is None:
        face_values = [fallback] * len(tri_faces)
        return face_values, [fallback] * point_count
    if authored.element_type == "constant":
        face_values = [authored.values[0]] * len(tri_faces)
    elif authored.element_type == "face":
        face_values = [authored.values[int(source)] for source in tri_source_faces]
        point_sources: list[set[int]] = [set() for _ in range(point_count)]
        for face, source in zip(tri_faces, tri_source_faces, strict=True):
            for point in face:
                point_sources[int(point)].add(int(source))
        point_values = [
            sum(authored.values[source] for source in sources) / len(sources) if sources else fallback
            for sources in point_sources
        ]
        return face_values, point_values
    else:
        face_values = [sum(authored.values[int(point)] for point in face) / 3.0 for face in tri_faces]
        return face_values, list(authored.values)

    point_sums = [0.0] * point_count
    point_counts = [0] * point_count
    for face, value in zip(tri_faces, face_values, strict=True):
        for point in face:
            point_sums[int(point)] += value
            point_counts[int(point)] += 1
    point_values = [
        point_sums[point] / point_counts[point] if point_counts[point] else fallback for point in range(point_count)
    ]
    return face_values, point_values


def _has_legacy_surface_material(material: dict[str, float]) -> bool:
    """Whether attributes from the earlier surface-material revision are authored."""
    return any(name in material for name in _LEGACY_SURFACE_MATERIAL_ATTRS)


def _is_legacy_only_surface_material(material: dict[str, float]) -> bool:
    """Whether only attributes from the earlier surface-material revision are authored."""
    return _has_legacy_surface_material(material) and not any(
        name in material for name in _POST_RENAME_SURFACE_MATERIAL_ATTRS
    )


def _warn_legacy_surface_material(path: str, material: dict[str, float] | None) -> None:
    """Warn when an earlier surface-material attribute is authored."""
    if material is None:
        return
    if "surfaceThickness" in material:
        warnings.warn(
            f"{path}: physics:surfaceThickness has moved off the material and is deprecated; "
            "author physics:thicknesses on the simulation geometry with "
            "physics:thicknesses:elementType instead.",
            NewtonDeprecationWarning,
            stacklevel=2,
        )
    if _has_legacy_surface_material(material):
        warnings.warn(
            f"{path}: unprefixed surface material attributes follow an earlier AOUSD proposal revision "
            f"and are deprecated; move physics:thickness to simulation-geometry physics:thicknesses "
            f"with an element type, and convert the old moduli to structural "
            f"physics:surface*Stiffness values.",
            NewtonDeprecationWarning,
            stacklevel=2,
        )


def _resolve_surface_structural_stiffnesses(
    material: dict[str, float] | None,
    thickness: float | np.ndarray | None,
    linear_unit: float,
) -> tuple[float | np.ndarray | None, float | np.ndarray | None, float | np.ndarray | None] | None:
    """Resolve scalar or vector surface stretch, shear, and bend structural stiffnesses."""
    if material is None or thickness is None:
        return None

    def constant(value: float) -> float | np.ndarray:
        if isinstance(thickness, np.ndarray):
            return np.full_like(thickness, value)
        return value

    if _is_legacy_only_surface_material(material):
        stretch = material.get("stretchStiffness")
        shear = material.get("shearStiffness")
        bend = material.get("bendStiffness")
        return (
            None if stretch is None else stretch * thickness,
            None if shear is None else shear * thickness,
            None if bend is None else bend * thickness**3,
        )

    youngs = material.get("youngsModulus", _AOUSD_DEFAULT_YOUNGS_MODULUS * linear_unit)
    poissons = material.get("poissonsRatio", _AOUSD_DEFAULT_POISSONS_RATIO)
    shear_modulus = youngs / (2.0 * (1.0 + poissons))

    def resolve(current_name: str, legacy_name: str, derived: float | np.ndarray) -> float | np.ndarray:
        if current_name in material:
            return constant(material[current_name])
        if legacy_name in material:
            legacy = material[legacy_name]
            return legacy * (thickness**3 if current_name == "surfaceBendStiffness" else thickness)
        return derived

    plane_stress = 1.0 - poissons**2
    return (
        resolve("surfaceStretchStiffness", "stretchStiffness", youngs * thickness / plane_stress),
        resolve("surfaceShearStiffness", "shearStiffness", shear_modulus * thickness),
        resolve(
            "surfaceBendStiffness",
            "bendStiffness",
            youngs * thickness**3 / (12.0 * plane_stress),
        ),
    )


def _cloth_rest_data(
    points: np.ndarray, triangles: np.ndarray, *, with_poses: bool = True
) -> tuple[np.ndarray | None, np.ndarray]:
    """Compute areas, and unless ``with_poses`` is false the inverse rest bases, for ordered cloth elements.

    Raises ``ValueError`` naming the offending triangles for non-finite or degenerate input. The
    1e-12 margins keep the released cloth preflight and leave headroom over
    :meth:`ModelBuilder.add_triangles`, which drops only non-positive areas computed with a
    slightly different float64 formula.
    """

    def reject(mask: np.ndarray, what: str) -> None:
        bad = np.flatnonzero(mask)
        if len(bad):
            raise ValueError(f"{len(bad)} {what}(s), first at {bad[:5].tolist()}")

    if not np.all(np.isfinite(points)):
        raise ValueError("non-finite point coordinates")
    p = points[triangles[:, 0]]
    qp = points[triangles[:, 1]] - p
    rp = points[triangles[:, 2]] - p
    normals = np.cross(qp, rp)
    normal_lengths = np.linalg.norm(normals, axis=1)
    edge_lengths = np.linalg.norm(np.stack((qp, rp, rp - qp), axis=1), axis=2)
    reject(~np.isfinite(normal_lengths) | ~np.isfinite(edge_lengths).all(axis=1), "non-finite triangle")
    reject((normal_lengths <= 1.0e-12) | (edge_lengths <= 1.0e-12).any(axis=1), "degenerate triangle")
    normals /= normal_lengths[:, None]
    e1 = qp / edge_lengths[:, 0:1]
    e2 = np.cross(normals, e1)
    rest_basis = np.stack((e1, e2), axis=1) @ np.stack((qp, rp), axis=2)
    areas = np.linalg.det(rest_basis) * 0.5
    reject(~np.isfinite(areas), "non-finite triangle")
    reject(areas <= 1.0e-12, "degenerate triangle")
    return (np.linalg.inv(rest_basis) if with_poses else None), areas


def _normalized_rows(vectors: np.ndarray) -> np.ndarray:
    """Normalize rows, leaving zero rows unchanged like :meth:`ModelBuilder.add_edges`."""
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return vectors / lengths


def _rest_corners(
    rest_points: np.ndarray,
    rest_triangles: np.ndarray,
    sim_triangles: np.ndarray,
    tris: np.ndarray,
    verts: np.ndarray,
) -> np.ndarray:
    """Rest positions of simulation vertices ``verts`` read from the rest copies of ``tris``.

    Disjoint rest topology stores one rest point per triangle corner, so a simulation vertex is
    resolved through the corner slot it occupies in each adjacent simulation triangle.
    """
    slots = np.argmax(sim_triangles[tris] == verts[:, None], axis=1)
    return rest_points[rest_triangles[tris, slots]]


def _cloth_rest_default_angles(
    adjacency: MeshAdjacency, rest_points: np.ndarray, rest_triangles: np.ndarray, default_mode: str
) -> np.ndarray:
    """Resolve ``restBendAnglesDefault`` in the simulation mesh's deterministic edge order."""
    edges = adjacency.edge_indices
    edge_tris = adjacency.edge_tri_indices
    rest_angles = np.zeros(len(edges), dtype=np.float64)
    interior = np.flatnonzero(edge_tris[:, 1] != -1)
    if default_mode != "restShape" or len(interior) == 0:
        return rest_angles
    # Use the vertex roles and sign convention of ModelBuilder.add_edges, evaluated on each
    # triangle's own rest copy, so the dihedral matches the builder's live computation and does
    # not depend on the authored face winding.
    sim_triangles = adjacency.indices
    tri0, tri1 = edge_tris[interior, 0], edge_tris[interior, 1]
    o0, o1, v0, v1 = (edges[interior, column] for column in range(4))
    x1 = _rest_corners(rest_points, rest_triangles, sim_triangles, tri0, o0)
    x3 = _rest_corners(rest_points, rest_triangles, sim_triangles, tri0, v0)
    x4 = _rest_corners(rest_points, rest_triangles, sim_triangles, tri0, v1)
    x2 = _rest_corners(rest_points, rest_triangles, sim_triangles, tri1, o1)
    x3_other = _rest_corners(rest_points, rest_triangles, sim_triangles, tri1, v0)
    x4_other = _rest_corners(rest_points, rest_triangles, sim_triangles, tri1, v1)
    n1 = _normalized_rows(np.cross(x3 - x1, x4 - x1))
    n2 = _normalized_rows(np.cross(x4_other - x2, x3_other - x2))
    edge_dir = _normalized_rows(x4 - x3)
    cos_theta = np.clip(np.einsum("ij,ij->i", n1, n2), -1.0, 1.0)
    sin_theta = np.einsum("ij,ij->i", np.cross(n1, n2), edge_dir)
    rest_angles[interior] = np.arctan2(sin_theta, cos_theta)
    return rest_angles


def _cloth_rest_bend_overrides(
    rest_angles: np.ndarray,
    adjacency: MeshAdjacency,
    tri_source_faces: np.ndarray,
    authored_pairs,
    authored_angles,
    path: str,
) -> np.ndarray:
    """Apply authored ``restAdjTriPairs`` / ``restBendAngles`` entries to a copy of ``rest_angles``."""
    pairs = _rest_index_array(authored_pairs if authored_pairs is not None else [], 2, "restAdjTriPairs")
    angles = np.asarray(authored_angles if authored_angles is not None else [], dtype=np.float64).reshape(-1)
    if len(pairs) != len(angles):
        raise ValueError("restAdjTriPairs and restBendAngles must have matching lengths")
    # The proposal defines adjacency over the simulation mesh's faces, so a pair of fan-triangulated
    # polygon faces addresses the one bending edge between their triangles; the fan diagonals inside
    # a face are not authorable.
    edge_tris = adjacency.edge_tri_indices
    interior = np.flatnonzero(edge_tris[:, 1] != -1)
    faces = tri_source_faces[edge_tris[interior]]
    across = faces[:, 0] != faces[:, 1]
    face_count = int(tri_source_faces.max()) + 1
    # Group the bending edges by the sorted face pair they connect; two faces sharing several
    # edges all receive the pair's angle.
    pair_codes = np.min(faces[across], axis=1) * face_count + np.max(faces[across], axis=1)
    order = np.argsort(pair_codes, kind="stable")
    pair_codes = pair_codes[order]
    pair_edges = interior[across][order]
    unique_codes, starts, counts = np.unique(pair_codes, return_index=True, return_counts=True)
    rest_angles = rest_angles.copy()
    seen = set()
    for entry, (pair, angle) in enumerate(zip(pairs, angles, strict=True)):
        key = tuple(sorted((int(pair[0]), int(pair[1]))))
        slot = len(unique_codes)
        if 0 <= key[0] and key[1] < face_count:
            code = key[0] * face_count + key[1]
            slot = int(np.searchsorted(unique_codes, code))
            if slot == len(unique_codes) or unique_codes[slot] != code:
                slot = len(unique_codes)
        if slot == len(unique_codes):
            logger.warning(
                "%s: restAdjTriPairs entry %s contains non-adjacent face pair %s; skipping entry.",
                path,
                entry,
                key,
            )
            continue
        if not math.isfinite(float(angle)):
            logger.warning(
                "%s: restBendAngles entry %s for pair %s contains non-finite value %s; skipping entry.",
                path,
                entry,
                key,
                angle,
            )
            continue
        if key in seen:
            logger.warning(
                "%s: duplicate restAdjTriPairs entry %s for pair %s; using the last valid angle.",
                path,
                entry,
                key,
            )
        seen.add(key)
        # The AOUSD proposal's Rest Shape Attributes section authors bend angles in degrees.
        rest_angles[pair_edges[starts[slot] : starts[slot] + counts[slot]]] = math.radians(float(angle))
    return rest_angles


def _cloth_rest_edge_lengths(
    adjacency: MeshAdjacency, rest_points: np.ndarray, rest_triangles: np.ndarray, path: str
) -> np.ndarray:
    """Compute bending-edge lengths from the rest copies of each simulation edge."""
    sim_triangles = adjacency.indices
    edges = adjacency.edge_indices
    edge_tris = adjacency.edge_tri_indices
    v0, v1 = edges[:, 2], edges[:, 3]
    tri0 = edge_tris[:, 0]
    lengths = np.linalg.norm(
        _rest_corners(rest_points, rest_triangles, sim_triangles, tri0, v1)
        - _rest_corners(rest_points, rest_triangles, sim_triangles, tri0, v0),
        axis=1,
    )
    interior = np.flatnonzero(edge_tris[:, 1] != -1)
    if len(interior) == 0:
        return lengths
    tri1 = edge_tris[interior, 1]
    other_lengths = np.linalg.norm(
        _rest_corners(rest_points, rest_triangles, sim_triangles, tri1, v1[interior])
        - _rest_corners(rest_points, rest_triangles, sim_triangles, tri1, v0[interior]),
        axis=1,
    )
    # restShapePoints are float32, so two rest copies of one edge placed at different panel
    # positions round differently; tolerate that quantization before calling them inconsistent.
    quantization = 4.0 * np.finfo(np.float32).eps * float(np.max(np.abs(rest_points), initial=0.0))
    mismatched = np.flatnonzero(~np.isclose(lengths[interior], other_lengths, rtol=1.0e-5, atol=quantization))
    if len(mismatched):
        first = mismatched[0]
        edge = interior[first]
        logger.warning(
            "%s: disjoint rest topology assigns inconsistent lengths to %s simulation "
            "edge(s), e.g. [%.9g, %.9g] on edge (%s, %s); using the last rest length for bending.",
            path,
            len(mismatched),
            lengths[edge],
            other_lengths[first],
            v0[edge],
            v1[edge],
        )
    lengths[interior] = other_lengths
    return lengths


def _deformable_import_cloth(ctx: _DeformableImportContext) -> None:
    """Import surface deformables (``PhysicsSurfaceDeformableSimAPI`` polygon ``Mesh`` -> cloth).

    n-gon faces are fan-triangulated, so the source need not be pre-triangulated. The surface
    material is mapped onto the isotropic membrane and results land in ``path_cloth_map`` / attrs.
    """
    from pxr import UsdGeom

    from ..usd import utils as usd  # noqa: PLC0415
    from ..usd.schema_resolver import PrimType  # noqa: PLC0415

    builder = ctx.builder
    root_prim = ctx.root_prim
    ignore_paths = ctx.ignore_paths
    incoming_world_xform = ctx.incoming_world_xform
    verbose = ctx.verbose
    deformable_read = ctx.deformable_read
    get_prim_world_mat = ctx.get_prim_world_mat
    resolver = ctx.resolver
    path_cloth_map = ctx.path_cloth_map
    path_cloth_attrs = ctx.path_cloth_attrs

    if not (root_prim and root_prim.IsValid()):
        return
    for prim in ctx.prims.cloth:
        path = str(prim.GetPath())
        if _is_ignored_path(path, ignore_paths):
            continue
        skip_reason = _deformable_body_skip_reason(prim, deformable_read)
        if skip_reason is not None:
            warnings.warn(f"{path}: {skip_reason}; skipping cloth import.", NewtonWarning, stacklevel=2)
            continue
        if _skip_for_deformable_body_owner(ctx, prim, path):
            continue

        mesh = UsdGeom.Mesh(prim)
        mesh_points = mesh.GetPointsAttr().Get()
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not mesh_points or not face_counts or not face_indices:
            logger.warning("%s: cloth mesh missing points / topology; skipping.", path)
            continue
        if any(int(c) < 3 for c in face_counts):
            logger.warning("%s: cloth mesh has a face with fewer than 3 vertices; skipping.", path)
            continue
        # Validate before builder mutation so malformed topology cannot crash the import
        # or leave a partially appended cloth behind.
        if sum(int(c) for c in face_counts) != len(face_indices):
            logger.warning(
                "%s: cloth mesh faceVertexCounts sum %s != faceVertexIndices length %s; skipping.",
                path,
                sum(int(c) for c in face_counts),
                len(face_indices),
            )
            continue
        if any(i < 0 or i >= len(mesh_points) for i in face_indices):
            logger.warning(
                "%s: cloth mesh has a face vertex index outside the %s-point array; skipping.", path, len(mesh_points)
            )
            continue
        # Reuse the shared mesh handling from the rigid path: fan-triangulate faces
        # (n-gons such as quads; exact for convex faces, preserving vertex indices so
        # each mesh point stays one particle) and flip winding for left-handed
        # orientation. Subdivision scheme is not consulted -- the polygon cage is simulated.
        world_mat = get_prim_world_mat(prim, None, incoming_world_xform)
        authored_tri_faces = usd.fan_triangulate_faces(np.asarray(face_counts), np.asarray(face_indices))
        tri_source_faces = np.repeat(np.arange(len(face_counts)), np.asarray(face_counts, dtype=np.int64) - 2)
        # A left-handed mesh and a reflective world transform (negative determinant) each reverse
        # triangle winding, so flip on their XOR to keep consistent outward orientation.
        flip_winding = (mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded) != _world_matrix_reflects(
            world_mat
        )
        tri_faces = authored_tri_faces[:, ::-1] if flip_winding else authored_tri_faces
        tri_vertex_indices = tri_faces.reshape(-1).tolist()
        authored_rest_bend_angles_default = deformable_read(prim, "restBendAnglesDefault")
        rest_bend_angles_default = str(authored_rest_bend_angles_default or "flat")
        if rest_bend_angles_default not in ("flat", "restShape"):
            logger.warning(
                "%s: invalid physics:restBendAnglesDefault '%s' (expected 'flat' or 'restShape'); using 'flat'.",
                path,
                rest_bend_angles_default,
            )
            rest_bend_angles_default = "flat"
        _warn_dropped_velocities(prim, path)
        _warn_geometry_authored_material_attrs(prim, path, "PhysicsSurfaceDeformableMaterialAPI", deformable_read)
        _warn_subset_material_bindings(prim, path)

        # add_cloth_mesh creates one particle per mesh vertex and takes only a uniform scale, so bake
        # the full world affine (incl. non-uniform scale, shear, reflection) into the vertices and
        # pass an identity placement -- wp.transform_decompose would drop reflection parity.
        cloth_vertices = _bake_world_points(mesh_points, world_mat)
        cloth_vertices_np = np.asarray(cloth_vertices, dtype=np.float64)

        # Validate the live geometry like a rest shape before any builder mutation; its areas are
        # the mass fallback when no authored rest shape applies.
        try:
            _, live_areas = _cloth_rest_data(cloth_vertices_np, tri_faces, with_poses=False)
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.warning("%s: invalid cloth simulation shape (%s); skipping the prim.", path, exc)
            continue

        rest_points_value = deformable_read(prim, "restShapePoints")
        rest_indices_value = deformable_read(prim, "restTriVertexIndices")
        if rest_points_value is not None and np.size(rest_points_value) == 0:
            rest_points_value = None
        rest_points = cloth_vertices_np
        rest_triangles = tri_faces
        rest_poses, rest_areas = None, live_areas
        rest_shape_applied = False
        if rest_points_value is not None:
            try:
                rest_points = _transform_rest_points(rest_points_value, world_mat)
                if rest_indices_value is None or len(rest_indices_value) == 0:
                    if len(rest_points) != len(mesh_points):
                        raise ValueError("restShapePoints must match points when restTriVertexIndices is omitted")
                    rest_triangles = tri_faces
                else:
                    rest_triangles = _rest_index_array(rest_indices_value, 3, "restTriVertexIndices")
                    if len(rest_triangles) != len(tri_faces):
                        raise ValueError("restTriVertexIndices must contain one triangle per simulation triangle")
                    if np.any(rest_triangles < 0) or np.any(rest_triangles >= len(rest_points)):
                        raise ValueError("restTriVertexIndices contains an out-of-range point index")
                    if flip_winding:
                        rest_triangles = rest_triangles[:, ::-1]
                rest_poses, rest_areas = _cloth_rest_data(rest_points, rest_triangles)
                rest_shape_applied = True
            except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
                logger.warning(
                    "%s: invalid cloth rest shape (%s); using the simulation shape.",
                    path,
                    exc,
                )
                rest_points = cloth_vertices_np
                rest_triangles = tri_faces
                rest_poses, rest_areas = None, live_areas
        elif rest_indices_value is not None and np.size(rest_indices_value):
            if not _rest_topology_matches(rest_indices_value, authored_tri_faces, 3):
                logger.warning(
                    "%s: restTriVertexIndices is authored without restShapePoints and does not match the "
                    "simulation topology; using the simulation shape.",
                    path,
                )

        authored_pairs = deformable_read(prim, "restAdjTriPairs")
        authored_angles = deformable_read(prim, "restBendAngles")
        if authored_pairs is not None and np.size(authored_pairs) == 0:
            authored_pairs = None
        if authored_angles is not None and np.size(authored_angles) == 0:
            authored_angles = None

        surface_material = usd._get_surface_deformable_material(prim, deformable_read)
        cloth_mat = surface_material or {}
        _warn_legacy_surface_material(path, surface_material)
        authored_thicknesses = _read_deformable_element_array(
            prim,
            "thicknesses",
            {"constant": 1, "face": len(face_counts), "point": len(mesh_points)},
            deformable_read,
        )
        # Removed material thickness and Newton's shell-thickness extension remain fallback
        # sources when geometry thickness is unauthored.
        thickness = cloth_mat.get("surfaceThickness", cloth_mat.get("thickness"))
        if thickness is None and resolver.get_value(prim, PrimType.SHAPE, "mass_model", default="solid") == "shell":
            shell_thickness_val = resolver.get_value(prim, PrimType.SHAPE, "shell_thickness")
            if shell_thickness_val is not None and math.isfinite(float(shell_thickness_val)):
                if float(shell_thickness_val) > 0.0:
                    thickness = float(shell_thickness_val)
        # Resolve the volumetric density before the thickness fallback: a density authored on
        # the deformable body or a base physics material carries no thickness by construction
        # (only the surface material can author one), yet still needs the areal conversion.
        vol_density = _resolve_deformable_density(
            prim,
            cloth_mat.get("density"),
            deformable_read,
            ctx.linear_unit,
            read_base_material=surface_material is None,
        )
        if thickness is None:
            thickness = _AOUSD_DEFAULT_THICKNESS / ctx.linear_unit
            if authored_thicknesses is None:
                previous_thickness = 2.0 * _AOUSD_DEFAULT_THICKNESS / ctx.linear_unit
                warnings.warn(
                    f"{path}: no valid surface thickness is available; using the AOUSD proposal's "
                    f"1 mm physical fallback ({thickness:g} stage units) instead of Newton's previous "
                    f"2 mm default. To preserve the previous behavior, author physics:thicknesses = "
                    f"[{previous_thickness:g}] with physics:thicknesses:elementType = 'constant'.",
                    NewtonWarning,
                    stacklevel=2,
                )

        face_thicknesses, point_thicknesses = _surface_thickness_samples(
            authored_thicknesses,
            tri_faces,
            tri_source_faces,
            len(mesh_points),
            thickness,
        )
        face_thickness_array = np.asarray(face_thicknesses, dtype=np.float64)
        point_thickness_array = np.asarray(point_thicknesses, dtype=np.float64)

        # Newton's isotropic membrane cannot apply stretch and shear independently, so
        # stretch drives its in-plane mode and shear remains metadata. Keep the area mode
        # at zero: None would inject an unauthored builder default. Missing current modes
        # derive from E, nu, and h; deprecated moduli retain their former conversion.
        structural_stiffnesses = _resolve_surface_structural_stiffnesses(
            surface_material, face_thicknesses[0], ctx.linear_unit
        )
        if structural_stiffnesses is None:
            tri_ke = None
            edge_ke = None
        else:
            tri_ke, _surface_shear_ke, edge_ke = structural_stiffnesses
        tri_ka = 0.0  # No independently representable area mode; None would inject a builder default.
        shear_name = None
        if "surfaceShearStiffness" in cloth_mat:
            shear_name = "surfaceShearStiffness"
        elif "shearStiffness" in cloth_mat:
            shear_name = "shearStiffness"
        if shear_name is not None:
            warnings.warn(
                f"{path}: {shear_name} is not applied -- Newton's isotropic cloth membrane makes "
                f"stretch and shear share one modulus. An anisotropic membrane (e.g. SolverStyle3D's "
                f"tri_aniso_ke) can honor it; the value is preserved in path_cloth_attrs.",
                NewtonWarning,
                stacklevel=2,
            )
        resolved_cloth_density = vol_density
        # Masses are assigned from per-face volumes after the triangles are built.
        density = 0.0
        particle_radius = 0.5 * point_thicknesses[0]

        # Newton has no per-particle collision toggle, so authored no-collision intent
        # cannot be honored for particle deformables; see the collision-gating docs.
        collision_enabled, approximated_from = _deformable_collision_enabled(prim, ctx.ignore_paths)
        _warn_collision_approximated(path, approximated_from)
        if not collision_enabled:
            _warn_collision_not_disableable(path)

        p0, t0, e0 = builder.particle_count, builder.tri_count, builder.edge_count
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=cloth_vertices,
            indices=tri_vertex_indices,
            density=density,
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            edge_ke=edge_ke,
            particle_radius=particle_radius,
            label=path,
        )
        if builder.tri_count - t0 != len(tri_faces):
            raise RuntimeError(
                f"{path}: ModelBuilder kept {builder.tri_count - t0} of {len(tri_faces)} preflighted triangles; "
                "the cloth preflight and ModelBuilder.add_triangles disagree."
            )
        if rest_shape_applied:
            for local_tri, (pose, area) in enumerate(zip(rest_poses.tolist(), rest_areas.tolist(), strict=True)):
                builder.tri_poses[t0 + local_tri] = pose
                builder.tri_areas[t0 + local_tri] = area
        builder.particle_radius[p0 : builder.particle_count] = (0.5 * point_thickness_array).tolist()

        element_volumes = rest_areas * face_thickness_array
        density_element_masses = resolved_cloth_density * element_volumes
        density_point_masses = np.bincount(
            tri_faces.reshape(-1),
            weights=np.repeat(density_element_masses / 3.0, 3),
            minlength=len(mesh_points),
        )
        builder.particle_mass[p0 : builder.particle_count] = density_point_masses.tolist()

        uniform_face_thickness = bool(np.all(face_thickness_array == face_thickness_array[0]))
        if not uniform_face_thickness and surface_material is not None:
            face_structural_stiffnesses = _resolve_surface_structural_stiffnesses(
                surface_material, face_thickness_array, ctx.linear_unit
            )
            face_stretches = (
                np.full_like(face_thickness_array, builder.default_tri_ke)
                if face_structural_stiffnesses is None or face_structural_stiffnesses[0] is None
                else face_structural_stiffnesses[0]
            )
            for tri_offset, stretch in enumerate(face_stretches):
                material = builder.tri_materials[t0 + tri_offset]
                builder.tri_materials[t0 + tri_offset] = (
                    float(stretch),
                    material[1],
                    material[2],
                    material[3],
                    material[4],
                )

        point_thickness_authored = authored_thicknesses is not None and authored_thicknesses.element_type == "point"
        uniform_edge_thickness = (
            bool(np.all(point_thickness_array == point_thickness_array[0]))
            if point_thickness_authored
            else uniform_face_thickness
        )
        if not uniform_edge_thickness and surface_material is not None:
            edge_face_indices: dict[tuple[int, int], list[int]] = {}
            for tri_offset, face in enumerate(tri_faces):
                for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                    key = tuple(sorted((int(edge[0]), int(edge[1]))))
                    edge_face_indices.setdefault(key, []).append(tri_offset)
            edge_thickness_array = np.empty(builder.edge_count - e0, dtype=np.float64)
            for edge_index, edge_offset in enumerate(range(e0, builder.edge_count)):
                _opposite_a, _opposite_b, edge_a, edge_b = builder.edge_indices[edge_offset]
                key = tuple(sorted((int(edge_a) - p0, int(edge_b) - p0)))
                adjacent_faces = edge_face_indices[key]
                if point_thickness_authored:
                    edge_thickness = 0.5 * (point_thicknesses[key[0]] + point_thicknesses[key[1]])
                else:
                    edge_thickness = sum(face_thicknesses[index] for index in adjacent_faces) / len(adjacent_faces)
                edge_thickness_array[edge_index] = edge_thickness
            edge_structural_stiffnesses = _resolve_surface_structural_stiffnesses(
                surface_material, edge_thickness_array, ctx.linear_unit
            )
            edge_bends = (
                np.full_like(edge_thickness_array, builder.default_edge_ke)
                if edge_structural_stiffnesses is None or edge_structural_stiffnesses[2] is None
                else edge_structural_stiffnesses[2]
            )
            for edge_offset, bend in zip(range(e0, builder.edge_count), edge_bends, strict=True):
                properties = builder.edge_bending_properties[edge_offset]
                builder.edge_bending_properties[edge_offset] = (float(bend), properties[1])
        # Resolve the bending rest state on the builder's own edge numbering: its rows are the
        # importer's local triangles offset by p0, so remapping them avoids a second edge derivation.
        # Without authored rest geometry or bend overrides, a restShape default keeps the dihedrals
        # the builder already derived from the live pose.
        bend_overrides_authored = authored_pairs is not None or authored_angles is not None
        rest_angles = None
        if rest_shape_applied or bend_overrides_authored:
            local_edges = np.asarray(builder.edge_indices[e0 : builder.edge_count], dtype=np.int64)
            adjacency = MeshAdjacency(tri_faces, edge_indices=np.where(local_edges >= 0, local_edges - p0, -1))
            if rest_shape_applied:
                rest_edge_lengths = _cloth_rest_edge_lengths(adjacency, rest_points, rest_triangles, path)
                for local_edge, length in enumerate(rest_edge_lengths.tolist()):
                    builder.edge_rest_length[e0 + local_edge] = length
            rest_angles = _cloth_rest_default_angles(adjacency, rest_points, rest_triangles, rest_bend_angles_default)
            if bend_overrides_authored:
                try:
                    rest_angles = _cloth_rest_bend_overrides(
                        rest_angles, adjacency, tri_source_faces, authored_pairs, authored_angles, path
                    )
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "%s: invalid authored rest bend angles (%s); using '%s' defaults.",
                        path,
                        exc,
                        rest_bend_angles_default,
                    )
        elif rest_bend_angles_default == "flat":
            rest_angles = np.zeros(builder.edge_count - e0)
        if rest_bend_angles_default == "flat" and rest_points_value is None and not bend_overrides_authored:
            has_nonplanar_rest_angle = any(
                builder.edge_indices[edge_offset][0] != -1
                and builder.edge_indices[edge_offset][1] != -1
                and abs(float(builder.edge_rest_angle[edge_offset])) > 1.0e-6
                for edge_offset in range(e0, builder.edge_count)
            )
            if authored_rest_bend_angles_default is None and has_nonplanar_rest_angle:
                warnings.warn(
                    f"{path}: unauthored physics:restBendAnglesDefault uses the proposal's 'flat' "
                    "fallback and replaces non-planar imported dihedral rest angles; author "
                    "physics:restBendAnglesDefault = 'restShape' to preserve them.",
                    NewtonWarning,
                    stacklevel=2,
                )
        if rest_angles is not None:
            for local_edge, angle in enumerate(rest_angles.tolist()):
                builder.edge_rest_angle[e0 + local_edge] = angle

        authored_masses = _apply_particle_masses(
            builder,
            prim,
            p0,
            builder.particle_count,
            deformable_read,
            element_name="face",
            element_indices=tri_faces,
            element_volumes=element_volumes,
            element_count=len(face_counts),
            element_sources=tri_source_faces,
        )
        path_cloth_map[path] = {
            "particle": (p0, builder.particle_count),
            "tri": (t0, builder.tri_count),
            "edge": (e0, builder.edge_count),
        }
        path_cloth_attrs[path] = {
            "material": dict(cloth_mat),
            "resolved_density": resolved_cloth_density,
            "rest_bend_angles_default": rest_bend_angles_default,
        }
        if authored_thicknesses is not None or authored_masses is not None:
            path_cloth_attrs[path]["simulation"] = {}
            if authored_thicknesses is not None:
                path_cloth_attrs[path]["simulation"]["thicknesses"] = {
                    "values": list(authored_thicknesses.values),
                    "element_type": authored_thicknesses.element_type,
                }
            if authored_masses is not None:
                path_cloth_attrs[path]["simulation"]["masses"] = {
                    "values": list(authored_masses.values),
                    "element_type": authored_masses.element_type,
                    "legacy_implicit_type": authored_masses.legacy_implicit_type,
                }
        if verbose:
            logger.info("Added cloth %s with %s particles.", path, builder.particle_count - p0)
