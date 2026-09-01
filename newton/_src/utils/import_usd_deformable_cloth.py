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
    _DEFAULT_CLOTH_THICKNESS,
    _apply_particle_masses,
    _bake_world_points,
    _deformable_body_skip_reason,
    _deformable_collision_enabled,
    _DeformableImportContext,
    _is_ignored_path,
    _mass_weight_density,
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

        cloth_mat = usd._get_surface_deformable_material(prim, deformable_read) or {}
        # Surface thickness: prefer the material's authored value; otherwise fall back to a
        # shell mass model's thickness (NewtonMassAPI massModel="shell" / shellThickness,
        # resolved across Newton / MuJoCo like the rigid shape path above).
        thickness = cloth_mat.get("thickness")
        if thickness is None and resolver.get_value(prim, PrimType.SHAPE, "mass_model", default="solid") == "shell":
            shell_thickness_val = resolver.get_value(prim, PrimType.SHAPE, "shell_thickness")
            if shell_thickness_val is not None and math.isfinite(float(shell_thickness_val)):
                if float(shell_thickness_val) > 0.0:
                    thickness = float(shell_thickness_val)
        # Resolve the volumetric density before the thickness fallback: a density authored on
        # the deformable body or a base physics material carries no thickness by construction
        # (only the surface material can author one), yet still needs the areal conversion.
        vol_density = _resolve_deformable_density(prim, cloth_mat.get("density"), deformable_read)
        if thickness is None and (
            vol_density is not None
            or any(key in cloth_mat for key in ("stretchStiffness", "bendStiffness", "shearStiffness"))
        ):
            # The proposal authors volumetric quantities and its unauthored-thickness
            # sentinel delegates to a simulator default; assume a fabric-like shell so the
            # values get a physical surface conversion.
            thickness = _DEFAULT_CLOTH_THICKNESS / ctx.linear_unit
            warnings.warn(
                f"{path}: volumetric physics values are authored but no thickness is "
                f"resolvable; assuming the default thickness of {thickness:g} stage units "
                f"(~{_DEFAULT_CLOTH_THICKNESS:g} m) for the mass, stiffness, and collision-radius "
                f"conversions. Author physics:thickness on the surface material (or a shell mass "
                f"model) to override.",
                stacklevel=2,
            )

        # Map the surface material onto Newton's isotropic triangle membrane (used by
        # SolverVBD and SolverSemiImplicit), whose triangle has three parameters:
        #   tri_ke  = mu     -> in-plane elastic stiffness  <- stretchStiffness
        #   edge_ke          -> dihedral bending stiffness  <- bendStiffness
        #   tri_ka  = lambda -> area preservation (Poisson) <- (no proposal attribute)
        # An isotropic membrane cannot separate stretch from shear: both live in mu. We
        # therefore drive mu from stretchStiffness and drop shearStiffness (an anisotropic
        # membrane such as SolverStyle3D's tri_aniso_ke is needed to honor it). tri_ka encodes
        # the Poisson ratio nu = tri_ka / (tri_ka + 2*tri_ke); the surface material authors no
        # Poisson term, so tri_ka is set to 0 (nu = 0). Passing None instead would inject the
        # builder's default area stiffness, giving even a zero-stiffness material an unauthored
        # area-preservation response.
        #
        # The proposal authors moduli in force/area; Newton integrates the triangle energy over
        # area, so a modulus is scaled by the shell thickness when one is authored (membrane
        # stiffness ~ E*h, bending ~ E*h^3) -- the surface analog of the cable path's E*A/L.
        #
        # Either way the raw, as-authored moduli (including the dropped shearStiffness) survive
        # in path_cloth_attrs, so another solver can rebuild from them.
        tri_ke = cloth_mat.get("stretchStiffness")
        edge_ke = cloth_mat.get("bendStiffness")
        if thickness is not None:
            tri_ke = tri_ke * thickness if tri_ke is not None else None
            edge_ke = edge_ke * thickness**3 if edge_ke is not None else None
        tri_ka = 0.0  # No proposal Poisson term; None would inject the builder's area default.
        if "shearStiffness" in cloth_mat:
            warnings.warn(
                f"{path}: shearStiffness is not applied -- Newton's isotropic cloth membrane makes "
                f"stretch and shear share one modulus. An anisotropic membrane (e.g. SolverStyle3D's "
                f"tri_aniso_ke) can honor it; the value is preserved in path_cloth_attrs.",
                stacklevel=2,
            )
        # Newton cloth density is areal; convert the volumetric density (resolved above) with
        # the surface thickness (required for surface mass per the proposal).
        resolved_cloth_density = vol_density if vol_density is not None else builder.default_shape_cfg.density
        # The areal value is builder-specific; keep it local to add_cloth_mesh. The weight
        # density stays neutral-positive when a body mass must be distributed over it.
        weight_density = _mass_weight_density(prim, resolved_cloth_density, deformable_read)
        density = weight_density * thickness if thickness is not None else weight_density
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
