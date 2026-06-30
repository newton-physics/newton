# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""USD volume-deformable import pass.

Imports ``UsdGeom.TetMesh`` prims as soft bodies via :meth:`ModelBuilder.add_soft_mesh`. Driven by
:func:`.import_usd.parse_usd` via a
:class:`.import_usd_deformable_utils._DeformableImportContext`.
"""

from __future__ import annotations

import copy

import numpy as np
import warp as wp

from .import_usd_deformable_utils import (
    _apply_particle_masses,
    _DeformableImportContext,
    _is_ignored_path,
    _resolve_deformable_density,
    _warn_geometry_authored_material_attrs,
    _warn_unsupported_rest_fields,
    _world_matrix_reflects,
)


def _deformable_import_volume(ctx: _DeformableImportContext) -> None:
    """Import volume deformables (``UsdGeom.TetMesh`` -> soft body via ``add_soft_mesh``).

    ``PhysicsVolumeDeformableSimAPI`` (or a ``PhysicsDeformableBodyAPI``) opts into the proposal
    mass precedence; a bare TetMesh stays legacy. Results land in ``path_soft_map`` / attrs.
    """
    from pxr import Usd, UsdGeom

    from ..usd import utils as usd  # noqa: PLC0415

    builder = ctx.builder
    root_prim = ctx.root_prim
    ignore_paths = ctx.ignore_paths
    incoming_world_xform = ctx.incoming_world_xform
    verbose = ctx.verbose
    deformable_read = ctx.deformable_read
    get_prim_world_mat = ctx.get_prim_world_mat
    get_tetmesh_cached = ctx.get_tetmesh_cached
    resolver = ctx.resolver
    collect_schema_attrs = ctx.collect_schema_attrs
    path_soft_map = ctx.path_soft_map
    path_soft_attrs = ctx.path_soft_attrs

    if not (root_prim and root_prim.IsValid()):
        return
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.TetMesh):
            continue

        path = str(prim.GetPath())
        # Per-instance proxies are imported via TraverseInstanceProxies above; USD does
        # not surface prototype masters under a scene-root traversal, so no prototype
        # filter is needed (a non-rendered template is authored as a class/inactive prim).
        if _is_ignored_path(path, ignore_paths):
            continue

        is_volume_deformable = (
            usd.has_applied_api_schema(prim, "PhysicsVolumeDeformableSimAPI")
            or usd._find_deformable_body_prim(prim) is not None
        )
        if is_volume_deformable:
            _warn_unsupported_rest_fields(prim, path, ("restShapePoints",), deformable_read)
            _warn_geometry_authored_material_attrs(prim, path, "PhysicsVolumeDeformableMaterialAPI", deformable_read)

        if collect_schema_attrs:
            resolver.collect_prim_attrs(prim)

        tetmesh = get_tetmesh_cached(prim)
        tetmesh_for_builder = tetmesh
        if tetmesh.custom_attributes:
            filtered_custom_attributes = {
                k: v for k, v in tetmesh.custom_attributes.items() if k in builder.custom_attributes
            }
            if len(filtered_custom_attributes) != len(tetmesh.custom_attributes):
                # Preserve the cached TetMesh while keeping add_usd's
                # current behavior of dropping unregistered import attrs.
                tetmesh_for_builder = copy.copy(tetmesh)
                tetmesh_for_builder.custom_attributes = filtered_custom_attributes

        soft_mesh_mat = get_prim_world_mat(prim, None, incoming_world_xform)
        # Bake the full world affine into the tet vertices and pass an identity placement, so a
        # reflective or sheared transform is applied exactly. wp.transform_decompose drops the
        # reflection parity, which would mirror the soft body back to a non-reflected pose.
        world_vertices = np.array(
            [
                wp.transform_point(soft_mesh_mat, wp.vec3(float(v[0]), float(v[1]), float(v[2])))
                for v in tetmesh_for_builder.vertices
            ],
            dtype=np.float32,
        )
        add_soft_mesh_kwargs = {
            "pos": wp.vec3(0.0, 0.0, 0.0),
            "rot": wp.quat_identity(),
            "scale": 1.0,
            "vel": wp.vec3(0.0, 0.0, 0.0),
            "mesh": tetmesh_for_builder,
            "vertices": world_vertices,
            "label": path,
        }
        if _world_matrix_reflects(soft_mesh_mat):
            # A reflection flips each tet's orientation (negative rest volume); swap two vertices per
            # tet to restore a positive orientation while keeping the same reflected shape. tet_indices
            # is read-only on TetMesh, so override via the explicit indices argument (it wins over mesh).
            flipped = np.asarray(tetmesh_for_builder.tet_indices).reshape(-1, 4).copy()
            flipped[:, [1, 2]] = flipped[:, [2, 1]]
            add_soft_mesh_kwargs["indices"] = flipped.reshape(-1)
        # Body density overrides the TetMesh's material density.
        if is_volume_deformable:
            resolved_density = _resolve_deformable_density(prim, tetmesh_for_builder.density, deformable_read)
            if resolved_density is not None:
                add_soft_mesh_kwargs["density"] = resolved_density

        soft_p0, soft_t0 = builder.particle_count, builder.tet_count
        builder.add_soft_mesh(**add_soft_mesh_kwargs)
        if is_volume_deformable:
            _apply_particle_masses(builder, prim, soft_p0, builder.particle_count, deformable_read)
        path_soft_map[path] = {
            "particle": (soft_p0, builder.particle_count),
            "tet": (soft_t0, builder.tet_count),
        }
        path_soft_attrs[path] = {
            # The density actually used: the resolved override if present, else the
            # TetMesh's own material density.
            "resolved_density": add_soft_mesh_kwargs.get("density", tetmesh_for_builder.density),
        }

        if verbose:
            print(f"Added soft mesh {path} with {tetmesh.vertex_count} vertices and {tetmesh.tet_count} tetrahedra.")
