# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Import proposal graphics geometry under deformable bodies as render meshes.

Per the AOUSD deformable proposal, graphics geometries are not tagged: they are
the ``UsdGeomPointBased`` prims under a ``PhysicsDeformableBodyAPI`` prim that
carry neither a simulation API nor a collision API. This pass walks each
imported deformable body's subtree (only bodies discovered by the scout, so a
stage without deformables pays nothing), reads each untagged ``UsdGeom.Mesh``,
extracts its proposal bind pose (``PhysicsDeformablePoseAPI`` with a
``bindPose`` purpose, falling back to the mesh's ``points``), and embeds it in
the body's simulation geometry via
:meth:`~newton.ModelBuilder.add_deformable_render_mesh`:

- volume bodies embed into the owning tet range;
- surface bodies embed into the owning triangle range;
- cable bodies bind to the curve's imported segment bodies.

Only ``UsdGeom.Mesh`` graphics prims are imported (a renderable triangle
surface); other point-based graphics geometry is left untouched.
"""

from __future__ import annotations

import warnings

import numpy as np

from .import_usd_deformable_utils import _DeformableImportContext


def _transform_points_np(world_mat, points: np.ndarray) -> np.ndarray:
    """Apply a warp 4x4 world matrix to an [N, 3] point array (full affine)."""
    m = np.asarray(world_mat, dtype=np.float64).reshape(4, 4)
    return points @ m[:3, :3].T + m[:3, 3]


def _sim_bind_positions(ctx: _DeformableImportContext, sim_path: str, particle_range) -> np.ndarray | None:
    """Particle positions overridden with the simulation geometry's bind pose.

    Embedding must pair the graphics bind pose with the simulation bind pose.
    When the simulation prim authors no ``bindPose`` its imported points are the
    bind configuration and no override is needed (returns ``None``).
    """
    from ..usd import utils as usd  # noqa: PLC0415

    sim_prim = ctx.stage.GetPrimAtPath(sim_path)
    if not sim_prim or not sim_prim.IsValid():
        return None
    bind = usd._get_deformable_bind_pose(sim_prim, strict=True)
    if bind is None:
        return None
    p0, p1 = particle_range
    if len(bind) != p1 - p0:
        warnings.warn(
            f"{sim_path}: bind pose has {len(bind)} points but the imported geometry has "
            f"{p1 - p0} particles; using the imported points as the bind pose.",
            stacklevel=2,
        )
        return None
    world_mat = ctx.get_prim_world_mat(sim_prim, None, ctx.incoming_world_xform)
    positions = np.array(ctx.builder.particle_q, dtype=np.float64)
    positions[p0:p1] = _transform_points_np(world_mat, bind)
    return positions


def _deformable_import_render(ctx: _DeformableImportContext) -> None:
    """Import graphics meshes for every imported deformable body and embed them."""
    from pxr import UsdGeom

    from ..usd import utils as usd  # noqa: PLC0415

    builder = ctx.builder
    for body_path, sim_path in ctx.prims.body_owner.items():
        if sim_path in ctx.path_soft_map:
            family = "soft"
        elif sim_path in ctx.path_cloth_map:
            family = "cloth"
        elif sim_path in ctx.path_cable_map or any(key.startswith(f"{sim_path}_curve") for key in ctx.path_cable_map):
            family = "cable"
        else:
            continue  # the governing simulation geometry did not import
        body_prim = ctx.stage.GetPrimAtPath(body_path)
        if not body_prim or not body_prim.IsValid():
            continue

        for prim in ctx.prims.visual_meshes.get(body_path, ()):
            if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
            path = str(prim.GetPath())
            mesh = ctx.get_mesh_cached(prim, load_uvs=True)
            if mesh is None or len(mesh.vertices) == 0 or len(mesh.indices) == 0:
                warnings.warn(f"{path}: graphics mesh has no geometry; skipping render mesh.", stacklevel=2)
                continue

            # Proposal bind pose (or the mesh points), in the common world frame the
            # simulation geometry was imported in.
            try:
                bind = usd._get_deformable_bind_pose(prim, strict=True)
            except ValueError as exc:
                warnings.warn(f"{path}: invalid render bind pose; skipping ({exc})", stacklevel=2)
                continue
            points = bind if bind is not None else np.asarray(mesh.vertices, dtype=np.float64)
            if len(points) != len(mesh.vertices):
                warnings.warn(
                    f"{path}: bind pose has {len(points)} points but the mesh has "
                    f"{len(mesh.vertices)} vertices; skipping render mesh.",
                    stacklevel=2,
                )
                continue
            world_mat = ctx.get_prim_world_mat(prim, None, ctx.incoming_world_xform)
            world_verts = _transform_points_np(world_mat, points).astype(np.float32)
            uvs = mesh._uvs if mesh._uvs is not None and len(mesh._uvs) == len(world_verts) else None
            texture = getattr(mesh, "texture", None)
            indices = np.asarray(mesh.indices, dtype=np.int32)

            try:
                if family == "soft":
                    ranges = ctx.path_soft_map[sim_path]
                    tet_range = ranges["tet"]
                    positions = _sim_bind_positions(ctx, sim_path, ranges["particle"])
                    parent, weights = builder._embed_render_vertices_in_tets(
                        world_verts, tet_range, positions=positions
                    )
                    index = builder.add_deformable_render_mesh(
                        world_verts,
                        indices,
                        kind="tet",
                        tet_range=tet_range,
                        parent=parent,
                        weights=weights,
                        uvs=uvs,
                        texture=texture,
                        label=path,
                    )
                elif family == "cloth":
                    ranges = ctx.path_cloth_map[sim_path]
                    tri_range = ranges["tri"]
                    positions = _sim_bind_positions(ctx, sim_path, ranges["particle"])
                    parent, weights = builder._embed_render_vertices_in_triangles(
                        world_verts, tri_range, positions=positions
                    )
                    index = builder.add_deformable_render_mesh(
                        world_verts,
                        indices,
                        kind="triangle",
                        tri_range=tri_range,
                        parent=parent,
                        weights=weights,
                        uvs=uvs,
                        texture=texture,
                        label=path,
                    )
                else:
                    # A multi-curve prim records per-curve entries; a welded graph curve
                    # keeps its exact (possibly non-contiguous) body list.
                    bodies: list[int] = []
                    for key, (curve_bodies, _joints) in ctx.path_cable_map.items():
                        if key == sim_path or key.startswith(f"{sim_path}_curve"):
                            bodies.extend(int(b) for b in curve_bodies)
                    index = builder.add_deformable_render_mesh(
                        world_verts,
                        indices,
                        kind="body",
                        bodies=bodies,
                        uvs=uvs,
                        texture=texture,
                        label=path,
                    )
            except ValueError as exc:
                warnings.warn(f"{path}: could not embed render mesh; skipping ({exc})", stacklevel=2)
                continue

            spec = builder._deformable_render_meshes[index]
            spec["body_path"] = body_path
            spec["sim_path"] = sim_path
            spec["graphics_path"] = path
            if ctx.verbose:
                print(f"  Embedded render mesh {path} ({len(world_verts)} verts) in {family} {sim_path}.")
