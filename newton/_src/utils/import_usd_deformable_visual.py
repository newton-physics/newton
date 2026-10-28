# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Import proposal graphics geometry under deformable bodies.

Per the AOUSD deformable proposal, graphics geometries are not tagged: they are
the ``UsdGeomPointBased`` prims under a ``PhysicsDeformableBodyAPI`` prim that
carry neither a simulation API nor a collision API. This pass walks each
imported deformable body's subtree (only bodies discovered by the scout, so a
stage without deformables pays nothing), reads each untagged ``UsdGeom.Mesh``,
extracts its proposal bind pose (``PhysicsDeformablePoseAPI`` with a
``bindPose`` purpose, falling back to the mesh's ``points``), and embeds it in
the body's simulation geometry via
:meth:`~newton.ModelBuilder.add_deformable_visual_mesh`:

- volume bodies embed into the owning tet range;
- surface bodies embed into the owning triangle range;
- cable bodies bind to the curve's imported segment bodies.

"""

from __future__ import annotations

import logging

import numpy as np
import warp as wp

from .import_usd_deformable_utils import _DeformableImportContext, _world_matrix_reflects, _world_points

logger = logging.getLogger(__name__)


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
    bind = usd.get_deformable_bind_pose(sim_prim)
    p0, p1 = particle_range
    if len(bind) != p1 - p0:
        raise ValueError(
            f"{sim_path}: bind pose has {len(bind)} points but the imported geometry has {p1 - p0} particles"
        )
    if np.array_equal(bind, sim_prim.GetAttribute("points").Get()):
        return None
    world_mat = ctx.get_prim_world_mat(sim_prim, None, ctx.incoming_world_xform)
    positions = np.array(ctx.builder.particle_q, dtype=np.float64)
    positions[p0:p1] = _world_points(bind, world_mat, translate=True)
    return positions


def _record_cable_visual_bind_poses(ctx, rod, bodies, point_sources):
    """Pair imported segment bodies with their bind-pose frames without changing physics.

    Each body's imported frame is transported from its initial tangent to its bind
    tangent, so graph cables, cables rooted on rigid bodies, and rotated bind poses
    keep the roll the cable was built with. Failures are recorded per body and
    reported by the visual pass instead of stopping the simulation import.
    """
    if not ctx.load_visual_shapes:
        return
    from ..sim.builder import ModelBuilder  # noqa: PLC0415
    from ..usd import utils as usd  # noqa: PLC0415

    try:
        authored = {}
        for path, _point in (source for sources in point_sources for source in sources):
            if path in authored:
                continue
            prim = ctx.stage.GetPrimAtPath(path)
            bind = usd.get_deformable_bind_pose(prim)
            # Keep the imported rod frames when the visual reference is the initial geometry.
            if np.array_equal(bind, prim.GetAttribute("points").Get()):
                bind = None
            else:
                matrix = ctx.get_prim_world_mat(prim, None, ctx.incoming_world_xform)
                bind = _world_points(bind, matrix, translate=True)
            authored[path] = bind
        if all(points is None for points in authored.values()):
            return
        initial = np.asarray(rod.points, dtype=np.float64).reshape(-1, 3)
        points = initial.copy()
        # Welds accept gaps up to a tenth of the radius, so authored bind poses may differ by that much.
        tolerance = 0.1 * float(rod.radius) if rod.radius else 1.0e-6
        for index, sources in enumerate(point_sources):
            candidates = [authored[path][point] for path, point in sources if authored[path] is not None]
            if candidates:
                if not np.allclose(candidates, candidates[0], atol=tolerance, rtol=0.0):
                    raise ValueError("welded cable points have inconsistent bind poses")
                points[index] = candidates[0]
        body_q = np.asarray(ctx.builder.body_q, dtype=np.float64).reshape(-1, 7)
        for body, (start, end) in zip(bodies, rod.edges, strict=True):
            initial_direction = initial[end] - initial[start]
            bind_direction = points[end] - points[start]
            # The body origin sits somewhere along its segment (the middle for rods); keep that spot.
            along = float(
                np.dot(body_q[body, :3] - initial[start], initial_direction)
                / max(float(np.dot(initial_direction, initial_direction)), 1.0e-30)
            )
            frame = ModelBuilder._transport_rod_frame(
                "add_usd",
                "bindPose",
                wp.quat(*body_q[body, 3:7]),
                wp.vec3(*initial_direction),
                wp.vec3(*bind_direction),
            )
            origin = points[start] + along * bind_direction
            ctx.cable_visual_bind_poses[body] = wp.transform(wp.vec3(*origin), frame)
    except ValueError as exc:
        # Invalid graphics references must not prevent the simulation from importing;
        # the visual pass reports them when it reaches the affected bodies.
        for body in bodies:
            ctx.cable_visual_bind_errors[body] = str(exc)


def _deformable_import_visual(ctx: _DeformableImportContext) -> None:
    """Import graphics payloads for every imported deformable body and embed them."""
    from ..usd import utils as usd  # noqa: PLC0415

    builder = ctx.builder
    # The simulation bind pose is read once per body, however many skins it carries.
    sim_positions: dict[str, np.ndarray | None] = {}
    for body_path, sim_path in ctx.prims.body_owner.items():
        if sim_path in ctx.path_soft_map:
            family = "soft"
        elif sim_path in ctx.path_cloth_map:
            family = "cloth"
        elif sim_path in ctx.path_cable_map:
            family = "cable"
        else:
            continue  # the governing simulation geometry did not import
        body_prim = ctx.stage.GetPrimAtPath(body_path)
        if not body_prim or not body_prim.IsValid():
            continue

        for prim in ctx.prims.visual_meshes.get(body_path, ()):
            # Viewport semantics, as for rigid visual shapes: visible, and purpose default or proxy.
            if not ctx.is_viewport_drawn(prim):
                continue
            path = str(prim.GetPath())
            # Proposal bind pose (or the mesh points), in the common world frame the
            # simulation geometry was imported in.
            try:
                mesh, _, source_point_indices = usd.get_mesh(prim, load_uvs=True, return_source_point_indices=True)
                points = usd.get_deformable_bind_pose(prim)[source_point_indices]
            except (ValueError, IndexError) as exc:
                # get_mesh indexes authored topology before checking it, hence IndexError.
                logger.warning("%s: invalid visual mesh, material, or bind pose; skipping (%s)", path, exc)
                continue
            if len(mesh.vertices) == 0 or len(mesh.indices) == 0:
                logger.warning("%s: graphics mesh has no geometry; skipping visual mesh.", path)
                continue
            world_mat = ctx.get_prim_world_mat(prim, None, ctx.incoming_world_xform)
            world_verts = _world_points(points, world_mat, translate=True).astype(np.float32)
            indices = np.asarray(mesh.indices, dtype=np.int32)
            if _world_matrix_reflects(world_mat):
                # A reflecting transform flips the winding, as the cloth and volume passes handle it.
                indices = indices.reshape(-1, 3)[:, ::-1].reshape(-1)
            uvs = mesh.uvs if mesh.uvs is not None and len(mesh.uvs) == len(world_verts) else None
            texture = mesh.texture
            if texture is not None and uvs is None:
                logger.warning("%s: visual texture requires UVs; importing the visual mesh without its texture.", path)
                texture = None
            if texture is not None:
                # Skinned meshes store final UVs, not a separate material transform.
                transform = np.asarray(mesh.texture_transform, dtype=uvs.dtype)
                uvs = uvs @ transform[:, :2].T + transform[:, 2]

            try:
                if family == "soft":
                    ranges = ctx.path_soft_map[sim_path]
                    if sim_path not in sim_positions:
                        sim_positions[sim_path] = _sim_bind_positions(ctx, sim_path, ranges["particle"])
                    parent, weights = builder._embed_visual_vertices_in_tets(
                        world_verts, ranges["tet"], positions=sim_positions[sim_path]
                    )
                    index = builder.add_deformable_visual_mesh(
                        world_verts,
                        indices,
                        kind="tetrahedron",
                        tet_range=ranges["tet"],
                        parent=parent,
                        weights=weights,
                        uvs=uvs,
                        texture=texture,
                        label=path,
                    )
                elif family == "cloth":
                    ranges = ctx.path_cloth_map[sim_path]
                    if sim_path not in sim_positions:
                        sim_positions[sim_path] = _sim_bind_positions(ctx, sim_path, ranges["particle"])
                    parent, weights = builder._embed_visual_vertices_in_triangles(
                        world_verts, ranges["tri"], positions=sim_positions[sim_path]
                    )
                    index = builder.add_deformable_visual_mesh(
                        world_verts,
                        indices,
                        kind="triangle",
                        tri_range=ranges["tri"],
                        parent=parent,
                        weights=weights,
                        uvs=uvs,
                        texture=texture,
                        label=path,
                    )
                else:
                    bodies = [int(b) for b in ctx.path_cable_map[sim_path][0]]
                    failed = [
                        ctx.cable_visual_bind_errors[body] for body in bodies if body in ctx.cable_visual_bind_errors
                    ]
                    if failed:
                        raise ValueError(failed[0])
                    reference_binding = None
                    if any(body in ctx.cable_visual_bind_poses for body in bodies):
                        poses = [ctx.cable_visual_bind_poses.get(body, builder.body_q[body]) for body in bodies]
                        reference_binding = builder._bind_visual_vertices_to_bodies(
                            world_verts, bodies, nearest=True, poses=np.asarray(poses)
                        )
                    index = builder.add_deformable_visual_mesh(
                        world_verts,
                        indices,
                        kind="body",
                        bodies=bodies,
                        uvs=uvs,
                        texture=texture,
                        label=path,
                    )
                    if reference_binding is not None:
                        spec = builder._deformable_visual_meshes[index]
                        spec["parent"], spec["local_offsets"] = reference_binding
            except ValueError as exc:
                logger.warning("%s: could not embed visual mesh; skipping (%s)", path, exc)
                continue

            spec = builder._deformable_visual_meshes[index]
            spec["body_path"] = body_path
            spec["sim_path"] = sim_path
            spec["graphics_path"] = path
            if ctx.verbose:
                logger.info("  Embedded visual mesh %s (%d verts) in %s %s.", path, len(world_verts), family, sim_path)
