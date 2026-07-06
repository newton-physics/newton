# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render-mesh embedding data for deformable bodies.

A render mesh is a high-resolution, textured display surface that is embedded in
a coarse simulation deformable (cloth, volumetric soft body, or a cable's rigid
segment chain) and skinned from the simulation state each frame. The simulation
continues to run on the coarse mesh; the render mesh is visualization and sensor
geometry only and never participates in the solve or in collision.

See :meth:`newton.ModelBuilder.add_deformable_render_mesh`.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import warp as wp


class DeformableRenderMesh:
    """A textured render mesh skinned from a deformable's simulation state.

    Instances are model output: they are created by
    :meth:`newton.ModelBuilder.finalize` from bindings registered with
    :meth:`newton.ModelBuilder.add_deformable_render_mesh` (or by the USD
    importer) and stored in :attr:`newton.Model.deformable_render_meshes`. The
    bind-pose vertices and topology are immutable asset data; the per-vertex
    embedding (:attr:`parent` plus :attr:`weights` or :attr:`local_offsets`)
    references simulation elements so consumers can evaluate the current
    surface from the state, and so future per-element simulation fields can be
    projected onto the visual vertices.

    Attributes are device :class:`warp.array` objects unless noted otherwise.
    """

    class Kind(IntEnum):
        """How a render mesh is embedded into its driving deformable."""

        PARTICLE = 0
        """Each render vertex is bound to one simulation particle (shared or
        1:1-remapped topology). The deformed position is the particle position
        directly. This is not a general high-resolution surface embedding; use
        :attr:`TRIANGLE` for independently discretized surface meshes."""

        TRIANGLE = 1
        """Each render vertex is embedded in a simulation triangle of a surface
        deformable via three barycentric weights. The deformed position is the
        weighted sum of the triangle's particle positions (no normal offset)."""

        TET = 2
        """Each render vertex is embedded in a tetrahedron of a volumetric soft
        body via four barycentric weights. The deformed position is the
        weighted sum of the tet's particle positions."""

        BODY = 3
        """Each render vertex is rigidly bound to one rigid body (e.g. a cable
        or rod capsule segment) by a body-local offset. The deformed position
        is that offset transformed by the body's current pose. Single-segment
        binding: seams can show at segment boundaries."""

    def __init__(
        self,
        kind: Kind,
        rest_vertices: wp.array[wp.vec3],
        indices: wp.array[wp.int32],
        parent: wp.array[wp.int32],
        weights: wp.array[wp.vec4] | wp.array[wp.vec3] | None = None,
        local_offsets: wp.array[wp.vec3] | None = None,
        uvs: wp.array[wp.vec2] | None = None,
        texture: np.ndarray | str | None = None,
        world: int = -1,
        label: str = "",
        index: int = -1,
        body_path: str | None = None,
        sim_path: str | None = None,
        graphics_path: str | None = None,
    ):
        self.kind = DeformableRenderMesh.Kind(kind)
        """Embedding kind (see :class:`DeformableRenderMesh.Kind`)."""
        self.rest_vertices = rest_vertices
        """Bind-pose render vertices [m], shape [vertex_count, 3]."""
        self.indices = indices
        """Flattened triangle indices into the render vertices, shape [tri_count*3]."""
        self.parent = parent
        """Per-render-vertex driver index, shape [vertex_count]. A particle index
        for :attr:`Kind.PARTICLE`, a triangle index into
        :attr:`newton.Model.tri_indices` for :attr:`Kind.TRIANGLE`, a tetrahedron
        index into :attr:`newton.Model.tet_indices` for :attr:`Kind.TET`, and a
        body index into ``State.body_q`` for :attr:`Kind.BODY`."""
        self.weights = weights
        """Barycentric weights: shape [vertex_count, 4] (vec4) for
        :attr:`Kind.TET`, shape [vertex_count, 3] (vec3) for
        :attr:`Kind.TRIANGLE`; ``None`` for other kinds."""
        self.local_offsets = local_offsets
        """Body-local bind offsets for :attr:`Kind.BODY`,
        shape [vertex_count, 3]; ``None`` for other kinds."""
        self.uvs = uvs
        """Per-render-vertex texture coordinates, shape [vertex_count, 2], or ``None``."""
        self.texture = texture
        """Albedo texture as an image array (H, W, C) or a path, or ``None``."""
        self.world = world
        """World index this render mesh belongs to (-1 for global)."""
        self.label = label
        """Display label. Not unique; use :attr:`index` for a stable identity."""
        self.index = index
        """Invariant index of this mesh in :attr:`newton.Model.deformable_render_meshes`."""
        self.body_path = body_path
        """USD path of the owning ``PhysicsDeformableBodyAPI`` prim, or ``None``
        when created programmatically."""
        self.sim_path = sim_path
        """USD path of the owning simulation geometry prim, or ``None`` when
        created programmatically."""
        self.graphics_path = graphics_path
        """USD path of the source graphics geometry prim, or ``None`` when
        created programmatically."""

    @property
    def vertex_count(self) -> int:
        return 0 if self.rest_vertices is None else len(self.rest_vertices)


@wp.kernel
def _skin_render_mesh_particle(
    particle_q: wp.array[wp.vec3],
    parent: wp.array[wp.int32],
    out_points: wp.array[wp.vec3],
):
    i = wp.tid()
    out_points[i] = particle_q[parent[i]]


@wp.kernel
def _skin_render_mesh_triangle(
    particle_q: wp.array[wp.vec3],
    tri_indices: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    weights: wp.array[wp.vec3],
    out_points: wp.array[wp.vec3],
):
    i = wp.tid()
    t = parent[i]
    w = weights[i]
    out_points[i] = (
        w[0] * particle_q[tri_indices[3 * t + 0]]
        + w[1] * particle_q[tri_indices[3 * t + 1]]
        + w[2] * particle_q[tri_indices[3 * t + 2]]
    )


@wp.kernel
def _skin_render_mesh_tet(
    particle_q: wp.array[wp.vec3],
    tet_indices: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    weights: wp.array[wp.vec4],
    out_points: wp.array[wp.vec3],
):
    i = wp.tid()
    t = parent[i]
    w = weights[i]
    out_points[i] = (
        w[0] * particle_q[tet_indices[4 * t + 0]]
        + w[1] * particle_q[tet_indices[4 * t + 1]]
        + w[2] * particle_q[tet_indices[4 * t + 2]]
        + w[3] * particle_q[tet_indices[4 * t + 3]]
    )


@wp.kernel
def _skin_render_mesh_body(
    body_q: wp.array[wp.transform],
    parent: wp.array[wp.int32],
    local_offsets: wp.array[wp.vec3],
    out_points: wp.array[wp.vec3],
):
    i = wp.tid()
    out_points[i] = wp.transform_point(body_q[parent[i]], local_offsets[i])


@wp.kernel
def _accumulate_face_normals(
    points: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
):
    # Face normals are weighted by triangle area (the un-normalized cross
    # product), yielding area-weighted vertex normals after accumulation.
    f = wp.tid()
    i0 = indices[3 * f + 0]
    i1 = indices[3 * f + 1]
    i2 = indices[3 * f + 2]
    n = wp.cross(points[i1] - points[i0], points[i2] - points[i0])
    wp.atomic_add(normals, i0, n)
    wp.atomic_add(normals, i1, n)
    wp.atomic_add(normals, i2, n)


@wp.kernel
def _normalize_normals(normals: wp.array[wp.vec3]):
    i = wp.tid()
    n = normals[i]
    length = wp.length(n)
    if length > 1.0e-12:
        normals[i] = n / length


def skin_render_mesh(
    mesh: DeformableRenderMesh,
    state,
    model,
    out_points: wp.array[wp.vec3],
    device=None,
) -> None:
    """Evaluate a render mesh's current vertex positions from the simulation state.

    Writes the skinned positions in the simulation frame into ``out_points``
    (shape [vertex_count, 3]); world offsets and layer transforms are the
    consumer's responsibility so viewers, sensors, and external integrations can
    apply their own placement. Runs entirely on ``device`` (the simulation
    device by default).
    """
    device = device if device is not None else out_points.device
    kind = DeformableRenderMesh.Kind
    if mesh.kind == kind.TET:
        wp.launch(
            _skin_render_mesh_tet,
            dim=len(out_points),
            inputs=[state.particle_q, model.tet_indices.flatten(), mesh.parent, mesh.weights],
            outputs=[out_points],
            device=device,
        )
    elif mesh.kind == kind.TRIANGLE:
        wp.launch(
            _skin_render_mesh_triangle,
            dim=len(out_points),
            inputs=[state.particle_q, model.tri_indices.flatten(), mesh.parent, mesh.weights],
            outputs=[out_points],
            device=device,
        )
    elif mesh.kind == kind.BODY:
        wp.launch(
            _skin_render_mesh_body,
            dim=len(out_points),
            inputs=[state.body_q, mesh.parent, mesh.local_offsets],
            outputs=[out_points],
            device=device,
        )
    else:
        wp.launch(
            _skin_render_mesh_particle,
            dim=len(out_points),
            inputs=[state.particle_q, mesh.parent],
            outputs=[out_points],
            device=device,
        )


def compute_render_mesh_normals(
    points: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    out_normals: wp.array[wp.vec3],
    device=None,
) -> None:
    """Recompute area-weighted vertex normals from current positions and topology."""
    device = device if device is not None else out_normals.device
    out_normals.zero_()
    wp.launch(
        _accumulate_face_normals,
        dim=len(indices) // 3,
        inputs=[points, indices, out_normals],
        device=device,
    )
    wp.launch(_normalize_normals, dim=len(out_normals), inputs=[out_normals], device=device)
