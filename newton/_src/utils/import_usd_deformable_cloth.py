# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""USD surface-deformable (cloth) import pass.

Imports ``PhysicsSurfaceDeformableSimAPI`` polygon ``UsdGeom.Mesh`` prims as cloth, mapping the
surface material onto the isotropic membrane. Driven by :func:`.import_usd.parse_usd` via a
:class:`.import_usd_deformable_utils._DeformableImportContext`.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import warp as wp

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
    _skip_for_deformable_body_owner,
    _warn_collision_approximated,
    _warn_collision_not_disableable,
    _warn_dropped_velocities,
    _warn_geometry_authored_material_attrs,
    _warn_subset_material_bindings,
    _warn_unsupported_rest_fields,
    _world_matrix_reflects,
)

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
            DeprecationWarning,
            stacklevel=2,
        )
    if _has_legacy_surface_material(material):
        warnings.warn(
            f"{path}: unprefixed surface material attributes follow an earlier AOUSD proposal revision "
            f"and are deprecated; move physics:thickness to simulation-geometry physics:thicknesses "
            f"with an element type, and convert the old moduli to structural "
            f"physics:surface*Stiffness values.",
            DeprecationWarning,
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
            warnings.warn(f"{path}: {skip_reason}; skipping cloth import.", stacklevel=2)
            continue
        if _skip_for_deformable_body_owner(ctx, prim, path):
            continue

        mesh = UsdGeom.Mesh(prim)
        mesh_points = mesh.GetPointsAttr().Get()
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not mesh_points or not face_counts or not face_indices:
            warnings.warn(f"{path}: cloth mesh missing points / topology; skipping.", stacklevel=2)
            continue
        if any(int(c) < 3 for c in face_counts):
            warnings.warn(f"{path}: cloth mesh has a face with fewer than 3 vertices; skipping.", stacklevel=2)
            continue
        # Validate the flattened topology before any builder mutation (matching the cable
        # pass's warn-and-skip policy), so malformed authoring cannot crash the import or
        # leave a partially-appended cloth behind.
        if sum(int(c) for c in face_counts) != len(face_indices):
            warnings.warn(
                f"{path}: cloth mesh faceVertexCounts sum {sum(int(c) for c in face_counts)} != "
                f"faceVertexIndices length {len(face_indices)}; skipping.",
                stacklevel=2,
            )
            continue
        if any(i < 0 or i >= len(mesh_points) for i in face_indices):
            warnings.warn(
                f"{path}: cloth mesh has a face vertex index outside the {len(mesh_points)}-point array; skipping.",
                stacklevel=2,
            )
            continue
        # Reuse the shared mesh handling from the rigid path: fan-triangulate faces
        # (n-gons such as quads; exact for convex faces, preserving vertex indices so
        # each mesh point stays one particle) and flip winding for left-handed
        # orientation. Subdivision scheme is not consulted -- the polygon cage is simulated.
        world_mat = get_prim_world_mat(prim, None, incoming_world_xform)
        tri_faces = usd.fan_triangulate_faces(np.asarray(face_counts), np.asarray(face_indices))
        tri_source_faces = np.repeat(np.arange(len(face_counts)), np.asarray(face_counts, dtype=np.int64) - 2)
        # A left-handed mesh and a reflective world transform (negative determinant) each reverse
        # triangle winding, so flip on their XOR to keep consistent outward orientation.
        if (mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded) != _world_matrix_reflects(world_mat):
            tri_faces = tri_faces[:, ::-1]
        tri_vertex_indices = tri_faces.reshape(-1).tolist()
        authored_rest_bend_angles_default = deformable_read(prim, "restBendAnglesDefault")
        rest_bend_angles_default = str(authored_rest_bend_angles_default or "flat")
        if rest_bend_angles_default not in ("flat", "restShape"):
            warnings.warn(
                f"{path}: invalid physics:restBendAnglesDefault '{rest_bend_angles_default}' "
                "(expected 'flat' or 'restShape'); using 'flat'.",
                stacklevel=2,
            )
            rest_bend_angles_default = "flat"
        _warn_unsupported_rest_fields(
            prim,
            path,
            ("restShapePoints", "restTriVertexIndices", "restBendAngles", "restAdjTriPairs"),
            deformable_read,
        )
        _warn_dropped_velocities(prim, path)
        _warn_geometry_authored_material_attrs(prim, path, "PhysicsSurfaceDeformableMaterialAPI", deformable_read)
        _warn_subset_material_bindings(prim, path)

        # add_cloth_mesh creates one particle per mesh vertex and takes only a uniform scale, so bake
        # the full world affine (incl. non-uniform scale, shear, reflection) into the vertices and
        # pass an identity placement -- wp.transform_decompose would drop reflection parity.
        cloth_vertices = _bake_world_points(mesh_points, world_mat)

        # A zero-area triangle cannot form an FEM element; add_cloth_mesh would drop it and
        # leave a partial import (particles without their triangle). Contain it like other
        # malformed topology: warn and skip the prim before any builder mutation.
        vert_np = np.array([[v[0], v[1], v[2]] for v in cloth_vertices], dtype=np.float64)
        nonfinite_points = int(np.count_nonzero(~np.isfinite(vert_np).all(axis=1)))
        if nonfinite_points:
            warnings.warn(
                f"{path}: cloth mesh has {nonfinite_points} point(s) with non-finite coordinates; skipping.",
                stacklevel=2,
            )
            continue
        edge1 = vert_np[tri_faces[:, 1]] - vert_np[tri_faces[:, 0]]
        edge2 = vert_np[tri_faces[:, 2]] - vert_np[tri_faces[:, 0]]
        tri_areas = 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)
        nonfinite_areas = int(np.count_nonzero(~np.isfinite(tri_areas)))
        if nonfinite_areas:
            warnings.warn(
                f"{path}: cloth mesh has {nonfinite_areas} triangle(s) with non-finite area; skipping.",
                stacklevel=2,
            )
            continue
        degenerate = int(np.count_nonzero(tri_areas < 1.0e-12))
        if degenerate:
            warnings.warn(
                f"{path}: cloth mesh has {degenerate} zero-area (degenerate) triangle(s); skipping.",
                stacklevel=2,
            )
            continue

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
        builder.particle_radius[p0 : builder.particle_count] = (0.5 * point_thickness_array).tolist()

        element_volumes = tri_areas * face_thickness_array
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
        if rest_bend_angles_default == "flat":
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
                    stacklevel=2,
                )
            for edge_offset in range(e0, builder.edge_count):
                builder.edge_rest_angle[edge_offset] = 0.0

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
        builder._record_cloth_group(
            path,
            (p0, builder.particle_count),
            (t0, builder.tri_count),
            (e0, builder.edge_count),
        )
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
            print(f"Added cloth {path} with {builder.particle_count - p0} particles.")
