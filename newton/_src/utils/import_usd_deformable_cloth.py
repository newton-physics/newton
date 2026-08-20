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
    _DEFAULT_CLOTH_THICKNESS,
    _apply_particle_masses,
    _bake_world_points,
    _deformable_body_skip_reason,
    _deformable_collision_enabled,
    _DeformableImportContext,
    _is_ignored_path,
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

_CURRENT_SURFACE_MATERIAL_ATTRS = (
    "surfaceThickness",
    "youngsModulus",
    "poissonsRatio",
    "surfaceStretchStiffness",
    "surfaceShearStiffness",
    "surfaceBendStiffness",
)
_LEGACY_SURFACE_MATERIAL_ATTRS = ("thickness", "stretchStiffness", "shearStiffness", "bendStiffness")


def _has_legacy_surface_material(material: dict[str, float]) -> bool:
    """Whether attributes from the earlier surface-material revision are authored."""
    return any(name in material for name in _LEGACY_SURFACE_MATERIAL_ATTRS)


def _is_legacy_only_surface_material(material: dict[str, float]) -> bool:
    """Whether only attributes from the earlier surface-material revision are authored."""
    return _has_legacy_surface_material(material) and not any(
        name in material for name in _CURRENT_SURFACE_MATERIAL_ATTRS
    )


def _warn_legacy_surface_material(path: str, material: dict[str, float] | None) -> None:
    """Warn when an earlier surface-material attribute is authored."""
    if material is not None and _has_legacy_surface_material(material):
        warnings.warn(
            f"{path}: unprefixed surface material attributes follow an earlier AOUSD proposal revision "
            f"and are deprecated; migrate physics:thickness to physics:surfaceThickness and convert "
            f"the old moduli to structural physics:surface*Stiffness values.",
            DeprecationWarning,
            stacklevel=2,
        )


def _resolve_surface_structural_stiffnesses(
    material: dict[str, float] | None, thickness: float | None, linear_unit: float
) -> tuple[float | None, float | None, float | None] | None:
    """Resolve surface stretch, shear, and bend structural stiffnesses."""
    if material is None or thickness is None:
        return None

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

    def resolve(current_name: str, legacy_name: str, derived: float) -> float:
        if current_name in material:
            return material[current_name]
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
        # A left-handed mesh and a reflective world transform (negative determinant) each reverse
        # triangle winding, so flip on their XOR to keep consistent outward orientation.
        if (mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded) != _world_matrix_reflects(world_mat):
            tri_faces = tri_faces[:, ::-1]
        tri_vertex_indices = tri_faces.reshape(-1).tolist()
        _warn_unsupported_rest_fields(
            prim,
            path,
            ("restShapePoints", "restBendAngles", "restAdjTriPairs", "restBendAnglesDefault"),
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
        edge1 = vert_np[tri_faces[:, 1]] - vert_np[tri_faces[:, 0]]
        edge2 = vert_np[tri_faces[:, 2]] - vert_np[tri_faces[:, 0]]
        tri_areas = 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)
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
        # Surface thickness: prefer the material's authored value; otherwise fall back to a
        # shell mass model's thickness (NewtonMassAPI massModel="shell" / shellThickness,
        # resolved across Newton / MuJoCo like the rigid shape path above).
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
        if thickness is None and surface_material is not None and not _is_legacy_only_surface_material(cloth_mat):
            thickness = _AOUSD_DEFAULT_THICKNESS / ctx.linear_unit
        if thickness is None:
            # Preserve Newton's released behavior for assets that have no current surface
            # material contract. Current materials use the proposal's 1 mm fallback above.
            thickness = _DEFAULT_CLOTH_THICKNESS / ctx.linear_unit
            warnings.warn(
                f"{path}: no current surface material thickness is resolvable; preserving "
                f"the compatibility default thickness of {thickness:g} stage units "
                f"(~{_DEFAULT_CLOTH_THICKNESS:g} m) for the mass, stiffness, and collision-radius "
                f"conversions. Author physics:surfaceThickness on the surface material (or a "
                f"shell mass model) to override.",
                stacklevel=2,
            )

        # Newton's isotropic membrane cannot apply stretch and shear independently, so
        # stretch drives its in-plane mode and shear remains metadata. Keep the area mode
        # at zero: None would inject an unauthored builder default. Missing current modes
        # derive from E, nu, and h; deprecated moduli retain their former conversion.
        structural_stiffnesses = _resolve_surface_structural_stiffnesses(surface_material, thickness, ctx.linear_unit)
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
        # Newton cloth density is areal; convert the volumetric density (resolved above) with
        # the surface thickness (required for surface mass per the proposal).
        resolved_cloth_density = vol_density
        # The areal value is builder-specific; keep it local to add_cloth_mesh.
        density = resolved_cloth_density * thickness if thickness is not None else resolved_cloth_density
        # Collision radius from the shell's physical half-thickness rather than the generic default.
        particle_radius = 0.5 * thickness if thickness is not None else None

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
        _apply_particle_masses(builder, prim, p0, builder.particle_count, deformable_read)
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
        }
        if verbose:
            print(f"Added cloth {path} with {builder.particle_count - p0} particles.")
