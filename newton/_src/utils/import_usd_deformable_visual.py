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

Experimental ``ParticleField3DGaussianSplat`` graphics become Gaussian visuals
when they belong to a volume deformable.
"""

from __future__ import annotations

import logging

import numpy as np
import warp as wp

from .import_usd_deformable_utils import _DeformableImportContext, _world_matrix_reflects, _world_points

logger = logging.getLogger(__name__)


def _sim_bind_positions(ctx: _DeformableImportContext, sim_path: str, particle_range) -> np.ndarray:
    """Return only the owning simulation geometry's bind-pose positions.

    Embedding must pair the graphics bind pose with the simulation bind pose.
    When the simulation prim authors no ``bindPose`` its imported points are the
    bind configuration.
    """
    from ..usd import utils as usd  # noqa: PLC0415

    sim_prim = ctx.stage.GetPrimAtPath(sim_path)
    bind = usd.get_deformable_bind_pose(sim_prim)
    p0, p1 = particle_range
    if len(bind) != p1 - p0:
        raise ValueError(
            f"{sim_path}: bind pose has {len(bind)} points but the imported geometry has {p1 - p0} particles"
        )
    if np.array_equal(bind, sim_prim.GetAttribute("points").Get()):
        return np.asarray(ctx.builder.particle_q[p0:p1], dtype=np.float64)
    world_mat = ctx.get_prim_world_mat(sim_prim, None, ctx.incoming_world_xform)
    return _world_points(bind, world_mat, translate=True)


def _record_cable_visual_bind_poses(ctx, rod, bodies, point_sources):
    """Pair imported segment bodies with their bind-pose frames without changing physics.

    Each body's imported frame is transported from its initial tangent to its bind
    tangent, so graph cables, cables rooted on rigid bodies, and rotated bind poses
    keep the roll the cable was built with. Failures are recorded per body and
    reported by the visual pass instead of stopping the simulation import.
    """
    if not ctx.load_visual_shapes or not any(
        path in ctx.cable_visual_sim_paths for sources in point_sources for path, _ in sources
    ):
        return
    from ..sim.builder import ModelBuilder  # noqa: PLC0415
    from ..usd import utils as usd  # noqa: PLC0415

    try:
        authored = {}
        for path, _point in (source for sources in point_sources for source in sources):
            if path in authored:
                continue
            if path not in ctx.cable_visual_bind_points:
                prim = ctx.stage.GetPrimAtPath(path)
                bind = usd.get_deformable_bind_pose(prim)
                # Several curves can share a prim; read and transform its points only once.
                if np.array_equal(bind, prim.GetAttribute("points").Get()):
                    bind = None
                else:
                    matrix = ctx.get_prim_world_mat(prim, None, ctx.incoming_world_xform)
                    bind = _world_points(bind, matrix, translate=True)
                ctx.cable_visual_bind_points[path] = bind
            authored[path] = ctx.cable_visual_bind_points[path]
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
        for body, (start, end) in zip(bodies, rod.edges, strict=True):
            body_q = np.asarray(ctx.builder.body_q[body], dtype=np.float64)
            initial_direction = initial[end] - initial[start]
            bind_direction = points[end] - points[start]
            # The body origin sits somewhere along its segment (the middle for rods); keep that spot.
            along = float(
                np.dot(body_q[:3] - initial[start], initial_direction)
                / max(float(np.dot(initial_direction, initial_direction)), 1.0e-30)
            )
            frame = ModelBuilder._transport_rod_frame(
                "add_usd",
                "bindPose",
                wp.quat(*body_q[3:7]),
                ModelBuilder._rod_segment_direction(initial[start], initial[end])[0],
                ModelBuilder._rod_segment_direction(points[start], points[end])[0],
            )
            origin = points[start] + along * bind_direction
            ctx.cable_visual_bind_poses[body] = wp.transform(wp.vec3(*origin), frame)
    except ValueError as exc:
        # Invalid graphics references must not prevent the simulation from importing;
        # the visual pass reports them when it reaches the affected bodies.
        for body in bodies:
            ctx.cable_visual_bind_errors[body] = str(exc)


def _read_gaussian_embedding(prim, tet_range: tuple[int, int], count: int):
    """Read experimental tet-index and barycentric-weight attributes."""
    parent_attr = prim.GetAttribute("newton:deformableSkin:tetIndices")
    weights_attr = prim.GetAttribute("newton:deformableSkin:influenceWeights")
    has_parent = bool(parent_attr and parent_attr.HasValue())
    has_weights = bool(weights_attr and weights_attr.HasValue())
    if has_parent != has_weights:
        raise ValueError("tetIndices and influenceWeights must be authored together")
    if not has_parent:
        return None, None

    parent = np.asarray(parent_attr.Get(), dtype=np.int32).reshape(-1)
    weights = np.asarray(weights_attr.Get(), dtype=np.float32).reshape(-1, 4)
    if len(parent) != count or len(weights) != count:
        raise ValueError("tetIndices and influenceWeights must have one row per Gaussian")
    # USD indices address the simulation TetMesh locally; Newton's builder uses
    # absolute indices so independently imported bodies can share one builder.
    parent = parent + int(tet_range[0])
    return parent, weights


def _gaussian_in_world(ctx: _DeformableImportContext, prim, gaussian):
    """Bake the prim placement into Gaussian centers, orientations, and scales."""
    world_mat = ctx.get_prim_world_mat(prim, None, ctx.incoming_world_xform)
    positions = _world_points(gaussian.positions, world_mat, translate=True).astype(np.float32)
    linear = np.asarray(world_mat, dtype=np.float64).reshape(4, 4)[:3, :3]
    rotations = np.empty_like(gaussian.rotations)
    scales = np.empty_like(gaussian.scales)
    for i, (rotation, scale) in enumerate(zip(gaussian.rotations, gaussian.scales, strict=True)):
        rest_rotation = np.asarray(wp.quat_to_matrix(wp.quat(*rotation)), dtype=np.float64).reshape(3, 3)
        axes = linear @ rest_rotation @ np.diag(scale)
        world_rotation, world_scale, _ = np.linalg.svd(axes)
        if np.linalg.det(world_rotation) < 0.0:
            world_rotation[:, -1] *= -1.0
        rotations[i] = np.asarray(wp.quat_from_matrix(wp.mat33f(world_rotation.astype(np.float32))), dtype=np.float32)
        scales[i] = world_scale.astype(np.float32)
    return type(gaussian)(
        positions=positions,
        rotations=rotations,
        scales=scales,
        opacities=gaussian.opacities,
        sh_coeffs=gaussian.sh_coeffs,
        sh_degree=gaussian.sh_degree,
        sh_color_space=gaussian.sh_color_space,
        min_response=gaussian.min_response,
        sorting_mode=gaussian.sorting_mode,
    )


def _deformable_import_visual(ctx: _DeformableImportContext) -> None:
    """Import graphics payloads for every imported deformable body and embed them."""
    from pxr import UsdGeom, UsdShade

    from ..geometry.flags import ShapeFlags  # noqa: PLC0415
    from ..usd import utils as usd  # noqa: PLC0415

    builder = ctx.builder
    # The simulation bind pose is read once per body, however many skins it carries.
    sim_positions: dict[str, np.ndarray] = {}
    skinned_bodies: set[int] = set()
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
            if any(
                subset.GetIndicesAttr().Get()
                and any(
                    rel.GetName().startswith("material:binding") and rel.GetTargets()
                    for rel in subset.GetPrim().GetRelationships()
                )
                for subset in UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets()
            ):
                logger.warning(
                    "%s: graphics material subsets are not supported; using only the mesh-level material.", path
                )
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
                        world_verts,
                        ranges["tet"],
                        positions=sim_positions[sim_path],
                        particle_start=ranges["particle"][0],
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
                        world_verts,
                        ranges["tri"],
                        positions=sim_positions[sim_path],
                        particle_start=ranges["particle"][0],
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
                    poses = None
                    if any(body in ctx.cable_visual_bind_poses for body in bodies):
                        poses = [ctx.cable_visual_bind_poses.get(body, builder.body_q[body]) for body in bodies]
                    index = builder._add_deformable_visual_mesh(
                        world_verts,
                        indices,
                        kind="body",
                        bodies=bodies,
                        body_poses=poses,
                        uvs=uvs,
                        texture=texture,
                        label=path,
                    )
                    # Unused visual vertices do not replace a segment's capsule.
                    parents = builder._deformable_visual_meshes[index]["parent"]
                    skinned_bodies.update(parents[vertex] for vertex in indices)
            except ValueError as exc:
                logger.warning("%s: could not embed visual mesh; skipping (%s)", path, exc)
                continue

            ctx.path_deformable_visual_map[path] = index
            builder._set_deformable_visual_source_paths(
                index, body_path=body_path, sim_path=sim_path, graphics_path=path
            )
            if ctx.verbose:
                logger.info("  Embedded visual mesh %s (%d verts) in %s %s.", path, len(world_verts), family, sim_path)

        for prim in ctx.prims.visual_gaussians.get(body_path, ()):
            path = str(prim.GetPath())
            if family != "soft":
                logger.warning("%s: Gaussian graphics currently require a volume deformable; skipping.", path)
                continue
            imageable = UsdGeom.Imageable(prim)
            if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
            try:
                gaussian = _gaussian_in_world(ctx, prim, usd.get_gaussian(prim))
                ranges = ctx.path_soft_map[sim_path]
                tet_range = ranges["tet"]
                if sim_path not in sim_positions:
                    sim_positions[sim_path] = _sim_bind_positions(ctx, sim_path, ranges["particle"])
                positions = sim_positions[sim_path]
                parent, weights = _read_gaussian_embedding(prim, tet_range, gaussian.count)
                if parent is None:
                    parent, weights = builder._embed_visual_vertices_in_tets(
                        gaussian.positions,
                        tet_range,
                        positions=positions,
                        operation="add_deformable_visual_gaussian",
                        particle_start=ranges["particle"][0],
                    )
                if np.any(parent < tet_range[0]) or np.any(parent >= tet_range[1]):
                    raise ValueError("tetIndices references a tetrahedron outside the owning volume")
                parent_vertices = np.asarray([builder.tet_indices[i] for i in parent])
                corners = positions[parent_vertices - ranges["particle"][0]]
                basis = np.transpose(corners[:, 1:] - corners[:, :1], (0, 2, 1))
                if np.any(np.abs(np.linalg.det(basis)) <= 1.0e-20):
                    raise ValueError("degenerate_parent: Gaussian bind tetrahedron is degenerate")
                bind_poses = np.linalg.inv(basis).astype(np.float32)
                index = builder.add_deformable_visual_gaussian(
                    gaussian,
                    kind="tetrahedron",
                    tet_range=tet_range,
                    parent=parent,
                    weights=weights,
                    label=path,
                )
            except ValueError as exc:
                logger.warning("%s: could not embed Gaussian graphics; skipping (%s)", path, exc)
                continue

            spec = builder._deformable_visual_gaussians[index]
            spec["_bind_poses"] = bind_poses
            spec["_bind_offsets"] = (gaussian.positions - np.einsum("ni,nij->nj", spec["weights"], corners)).astype(
                np.float32
            )
            spec["body_path"] = body_path
            spec["sim_path"] = sim_path
            spec["graphics_path"] = path
            ctx.path_shape_map[path] = spec["shape"]
            if ctx.verbose:
                logger.info("  Embedded Gaussian graphics %s (%d splats) in volume %s.", path, gaussian.count, sim_path)

    if ctx.hide_collision_shapes:
        for shape, body in enumerate(builder.shape_body):
            if body in skinned_bodies:
                builder.shape_flags[shape] &= ~ShapeFlags.VISIBLE
