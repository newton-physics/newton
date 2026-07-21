# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""USD volume-deformable import pass.

Imports ``UsdGeom.TetMesh`` prims as soft bodies via :meth:`ModelBuilder.add_soft_mesh`. Driven by
:func:`.import_usd.parse_usd` via a
:class:`.import_usd_deformable_utils._DeformableImportContext`.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import warp as wp

from ..sim.model import Model
from .import_usd_deformable_utils import (
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


def _deformable_import_volume(ctx: _DeformableImportContext) -> None:
    """Import volume deformables (``UsdGeom.TetMesh`` -> soft body via ``add_soft_mesh``).

    Only a TetMesh with ``PhysicsVolumeDeformableSimAPI`` takes the proposal mass precedence; other
    TetMeshes (graphics / collision, or bare) still import as legacy soft bodies. Results land in
    ``path_soft_map`` / attrs.
    """
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
    for prim in ctx.prims.tetmeshes:
        path = str(prim.GetPath())
        if _is_ignored_path(path, ignore_paths):
            continue
        is_volume_deformable = usd.has_applied_api_schema(prim, "PhysicsVolumeDeformableSimAPI")
        # A deformable-body subtree simulates only its simulation mesh: any other TetMesh in it
        # is graphics/collision geometry and is skipped, else it would add mass beyond the body's
        # authored total. Ownership stops at rigid/articulation boundaries, so a bare TetMesh
        # still imports as a legacy soft body. This runs before the body-flag checks: a graphics
        # TetMesh never becomes a body.
        if not is_volume_deformable:
            owner_body = usd._deformable_body_ancestor(prim)
            if owner_body is not None:
                warnings.warn(
                    f"{path}: non-simulation TetMesh under deformable body {owner_body.GetPath()}; "
                    f"treated as graphics/collision geometry (not simulated).",
                    stacklevel=2,
                )
                continue

        skip_reason = _deformable_body_skip_reason(prim, deformable_read)
        if skip_reason is not None:
            warnings.warn(f"{path}: {skip_reason}; skipping soft-body import.", stacklevel=2)
            continue
        # One simulation geometry per deformable body across ALL families (the scout picks the
        # first candidate in traversal order), so a body-level mass is applied exactly once.
        if is_volume_deformable and _skip_for_deformable_body_owner(ctx, prim, path):
            continue
        if is_volume_deformable:
            _warn_unsupported_rest_fields(prim, path, ("restShapePoints",), deformable_read)
            _warn_dropped_velocities(prim, path)
            _warn_geometry_authored_material_attrs(prim, path, "PhysicsVolumeDeformableMaterialAPI", deformable_read)
            _warn_subset_material_bindings(prim, path)

        if collect_schema_attrs:
            resolver.collect_prim_attrs(prim)

        try:
            tetmesh = get_tetmesh_cached(prim)
        except ValueError as exc:
            # Malformed authored topology (e.g. out-of-range tet indices) must not abort the
            # whole import; skip the prim like other broken deformable geometry.
            warnings.warn(f"{path}: invalid TetMesh; skipping soft-body import ({exc}).", stacklevel=2)
            continue
        supported_frequencies = {
            Model.AttributeFrequency.PARTICLE,
            Model.AttributeFrequency.TETRAHEDRON,
            Model.AttributeFrequency.TRIANGLE,
        }
        registered_attributes = [
            attr
            for attr in builder.custom_attributes.values()
            if attr.frequency in supported_frequencies and attr.usd_attribute_name != "*"
        ]
        registered_values = usd._get_tetmesh_custom_attribute_values(prim, registered_attributes)
        once_attributes = [
            attr
            for attr in builder.custom_attributes.values()
            if attr.frequency == Model.AttributeFrequency.ONCE and attr.usd_attribute_name != "*"
        ]
        once_values = usd._get_tetmesh_custom_attribute_values(prim, once_attributes)
        for attr in once_attributes:
            if attr.key in once_values:
                warnings.warn(
                    f"{path}: registered TetMesh attribute '{attr.usd_attribute_name}' has ONCE frequency, "
                    "which cannot be attached per soft body and is not imported.",
                    stacklevel=2,
                )
        frequency_counts = {
            Model.AttributeFrequency.PARTICLE: (tetmesh.vertex_count, "particle"),
            Model.AttributeFrequency.TETRAHEDRON: (tetmesh.tet_count, "tetrahedron"),
            Model.AttributeFrequency.TRIANGLE: (len(tetmesh.surface_tri_indices) // 3, "triangle"),
        }
        imported_attributes = {}
        for attr in registered_attributes:
            if attr.key not in registered_values:
                continue
            arr = np.asarray(registered_values[attr.key])
            actual_count = arr.shape[0] if arr.ndim >= 1 else 1
            expected_count, frequency_name = frequency_counts[attr.frequency]
            if arr.ndim == 0 or actual_count != expected_count:
                warnings.warn(
                    f"{path}: registered TetMesh attribute '{attr.usd_attribute_name}' has array length "
                    f"{actual_count}, which does not match {frequency_name} count {expected_count}; "
                    "skipping the attribute.",
                    stacklevel=2,
                )
                continue
            imported_attributes[attr.key] = (arr, attr.frequency)
        tetmesh_for_builder = tetmesh
        if tetmesh.custom_attributes or imported_attributes:
            # The builder declaration supplies both the USD name and authoritative frequency.
            # Preserve the cached TetMesh because another import pass may reuse it.
            tetmesh_for_builder = copy.copy(tetmesh)
            tetmesh_for_builder.custom_attributes = imported_attributes

        soft_mesh_mat = get_prim_world_mat(prim, None, incoming_world_xform)
        # Bake the full world affine into the tet vertices and pass an identity placement, so a
        # reflective or sheared transform is applied exactly. wp.transform_decompose drops the
        # reflection parity, which would mirror the soft body back to a non-reflected pose.
        world_vertices = np.array(_bake_world_points(tetmesh_for_builder.vertices, soft_mesh_mat), dtype=np.float32)
        if is_volume_deformable:
            # Newton has no per-particle collision toggle, so authored no-collision
            # intent cannot be honored; the legacy bare-TetMesh path is unchanged.
            collision_enabled, approximated_from = _deformable_collision_enabled(prim, ctx.ignore_paths)
            _warn_collision_approximated(path, approximated_from)
            if not collision_enabled:
                _warn_collision_not_disableable(path)

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
        neutral_weight = False
        if is_volume_deformable:
            resolved_density = _resolve_deformable_density(prim, tetmesh_for_builder.density, deformable_read)
            if resolved_density is not None:
                add_soft_mesh_kwargs["density"] = resolved_density
            else:
                # Mirror add_soft_mesh's own fallback to see the density that would build the
                # particle masses: a non-positive one leaves nothing for the body-mass rescale
                # in _apply_particle_masses to distribute. The neutral weight keeps the masses
                # volume-proportional, which the rescale turns into the proposal's
                # density-independent m_p = m_tot * V_p / V_tot (as the cable/cloth paths do).
                fallback_density = tetmesh_for_builder.density
                if fallback_density is None:
                    fallback_density = builder.default_tet_density
                weight_density = _mass_weight_density(prim, fallback_density, deformable_read)
                if weight_density != fallback_density:
                    add_soft_mesh_kwargs["density"] = weight_density
                    neutral_weight = True

        soft_p0, soft_t0 = builder.particle_count, builder.tet_count
        builder.add_soft_mesh(**add_soft_mesh_kwargs)
        if is_volume_deformable:
            _apply_particle_masses(builder, prim, soft_p0, builder.particle_count, deformable_read)
        path_soft_map[path] = {
            "particle": (soft_p0, builder.particle_count),
            "tet": (soft_t0, builder.tet_count),
        }
        builder._record_soft_group(
            path,
            (soft_p0, builder.particle_count),
            (soft_t0, builder.tet_count),
        )
        # The density actually used, mirroring add_soft_mesh's own resolution order:
        # explicit override, else the TetMesh's material density, else the builder default.
        # A neutral weight only distributes a body-mass total and is not a physical
        # density, so the metadata reports the unmodified resolution instead.
        effective_density = add_soft_mesh_kwargs.get("density", tetmesh_for_builder.density)
        if neutral_weight:
            effective_density = tetmesh_for_builder.density
        if effective_density is None:
            effective_density = builder.default_tet_density
        path_soft_attrs[path] = {
            "resolved_density": effective_density,
        }

        if verbose:
            print(f"Added soft mesh {path} with {tetmesh.vertex_count} vertices and {tetmesh.tet_count} tetrahedra.")
