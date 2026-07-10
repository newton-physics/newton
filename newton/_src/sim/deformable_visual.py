# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Visual-mesh embedding data for deformable bodies.

A visual mesh is a high-resolution, textured display surface that is embedded in
a coarse simulation deformable (cloth, volumetric soft body, or a cable's rigid
segment chain) and skinned from the simulation state each frame. The simulation
continues to run on the coarse mesh; the visual mesh is visualization and sensor
geometry only and never participates in the solve or in collision.

See :meth:`newton.ModelBuilder.add_deformable_visual_mesh`.
"""

from __future__ import annotations

import operator
from enum import IntEnum
from typing import TYPE_CHECKING, SupportsIndex

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from .model import Model
    from .state import State


class DeformableVisualBinding:
    """Payload-neutral binding from visual points to simulation drivers.

    The binding is independent of the payload being skinned. Today the payload
    is a triangle mesh; the same binding data can later drive other visual
    payloads, such as Gaussian splats, without changing how importer code
    selects simulation drivers.
    """

    def __init__(
        self,
        kind,
        parent: wp.array[wp.int32],
        weights: wp.array[wp.vec4] | wp.array[wp.vec3] | None = None,
        local_offsets: wp.array[wp.vec3] | None = None,
    ):
        self.kind = kind
        """Embedding kind."""
        self.parent = parent
        """Per-visual-point driver index, shape [point_count]."""
        self.weights = weights
        """Barycentric weights for triangle or tet bindings, or ``None``."""
        self.local_offsets = local_offsets
        """Body-local offsets for rigid body bindings, or ``None``."""


class DeformableVisualMesh:
    """A textured visual mesh skinned from a deformable's simulation state.

    Instances are model output: they are created by
    :meth:`newton.ModelBuilder.finalize` from bindings registered with
    :meth:`newton.ModelBuilder.add_deformable_visual_mesh` (or by the USD
    importer) and stored in :attr:`newton.Model.deformable_visual_meshes`. The
    bind-pose vertices and topology are immutable asset data; the per-vertex
    embedding is stored in :attr:`binding` and also exposed through the
    compatibility attributes :attr:`parent`, :attr:`weights`, and
    :attr:`local_offsets`. It references simulation elements so consumers can
    evaluate the current surface from the state, and so future per-element
    simulation fields can be projected onto the visual vertices.

    Attributes are device :class:`warp.array` objects unless noted otherwise.
    """

    class Kind(IntEnum):
        """How a visual mesh is embedded into its driving deformable."""

        PARTICLE = 0
        """Each visual vertex is bound to one simulation particle (shared or
        1:1-remapped topology). The deformed position is the particle position
        directly. This is not a general high-resolution surface embedding; use
        :attr:`TRIANGLE` for independently discretized surface meshes."""

        TRIANGLE = 1
        """Each visual vertex is embedded in a simulation triangle of a surface
        deformable via three barycentric weights. The deformed position is the
        weighted sum of the triangle's particle positions (no normal offset)."""

        TET = 2
        """Each visual vertex is embedded in a tetrahedron of a volumetric soft
        body via four barycentric weights. The deformed position is the
        weighted sum of the tet's particle positions."""

        BODY = 3
        """Each visual vertex is rigidly bound to one rigid body (e.g. a cable
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
        self.kind = DeformableVisualMesh.Kind(kind)
        """Embedding kind (see :class:`DeformableVisualMesh.Kind`)."""
        self.rest_vertices = rest_vertices
        """Bind-pose visual vertices [m], shape [vertex_count, 3]."""
        self.indices = indices
        """Flattened triangle indices into the visual vertices, shape [tri_count*3]."""
        self.binding = DeformableVisualBinding(
            kind=self.kind,
            parent=parent,
            weights=weights,
            local_offsets=local_offsets,
        )
        """Binding from visual vertices to simulation drivers."""
        self.parent = parent
        """Per-visual-vertex driver index, shape [vertex_count]. A particle index
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
        """Per-visual-vertex texture coordinates, shape [vertex_count, 2], or ``None``."""
        self.texture = texture
        """Albedo texture as an image array (H, W, C) or a path, or ``None``."""
        self.world = world
        """World index this visual mesh belongs to (-1 for global)."""
        self.label = label
        """Display label. Not unique; use :attr:`index` for a stable identity."""
        self.index = index
        """Invariant index of this mesh in :attr:`newton.Model.deformable_visual_meshes`."""
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


class DeformableVisuals:
    """Current skinned points and normals for a model's deformable visuals.

    Allocate this result with :meth:`newton.Model.deformable_visuals` and
    populate it with :meth:`newton.Model.update_deformable_visuals`. One result
    can be reused as simulation states are swapped. Allocate separate results
    when multiple states must remain available simultaneously.

    .. experimental::

        This API and its synchronization contract may change without a formal
        deprecation cycle while deformable visual support is experimental.
    """

    def __init__(self, model: Model):
        self._model = model
        self.device = model.device
        """Device containing :attr:`points` and :attr:`normals`."""

        ranges: list[tuple[int, int]] = []
        vertex_start = 0
        for mesh in model.deformable_visual_meshes:
            vertex_end = vertex_start + mesh.vertex_count
            ranges.append((vertex_start, vertex_end))
            vertex_start = vertex_end

        self.mesh_ranges = tuple(ranges)
        """Stable ``[start, end)`` vertex range for each visual mesh."""
        self.vertex_count = vertex_start
        """Total number of current visual vertices."""
        self.points = wp.empty(vertex_start, dtype=wp.vec3, device=self.device)
        """Current skinned visual points [m], shape [vertex_count, 3]."""
        self.normals = wp.zeros(vertex_start, dtype=wp.vec3, device=self.device)
        """Current visual unit normals, shape [vertex_count, 3]."""

        self._state: State | None = None
        self._completion_event = wp.Event(self.device) if self.device.is_cuda else None

    @property
    def model(self) -> Model:
        """Model whose visual mesh layout this result uses."""
        return self._model

    @property
    def state(self) -> State | None:
        """State used by the most recent update, or ``None`` before the first update."""
        return self._state

    @property
    def completion_event(self) -> wp.Event | None:
        """Event recorded after the most recent device update, or ``None`` on CPU."""
        return self._completion_event

    def _mesh_index(self, mesh: DeformableVisualMesh | SupportsIndex) -> int:
        if isinstance(mesh, DeformableVisualMesh):
            index = mesh.index
            if index < 0 or index >= len(self.mesh_ranges) or self._model.deformable_visual_meshes[index] is not mesh:
                raise ValueError("The deformable visual mesh does not belong to this DeformableVisuals model.")
            return index

        try:
            index = operator.index(mesh)
        except TypeError as exc:
            raise TypeError("mesh must be a DeformableVisualMesh or an integer mesh index") from exc
        if index < 0 or index >= len(self.mesh_ranges):
            raise IndexError(f"Deformable visual mesh index {index} is out of range")
        return index

    def _require_updated(self, state: State | None = None) -> None:
        if self._state is None:
            raise RuntimeError(
                "DeformableVisuals has not been updated; call model.update_deformable_visuals(state, visuals) first."
            )
        if state is not None and self._state is not state:
            raise ValueError("DeformableVisuals was last updated from another state.")

    def _validate_model(self, model: Model) -> None:
        if self._model is not model:
            raise ValueError("DeformableVisuals was created for another model.")

    def _mark_updated(self, state: State) -> None:
        self._state = state
        if self._completion_event is not None:
            stream = wp.get_stream(self.device)
            stream.record_event(self._completion_event, external=stream.is_capturing)

    def wait(self, stream: wp.Stream | None = None) -> None:
        """Make a device stream wait for the most recent visual update.

        Args:
            stream: Consumer stream. Uses the current stream on :attr:`device`
                when omitted. This method is a no-op on CPU.
        """
        self._require_updated()
        if self._completion_event is not None:
            if stream is None:
                stream = wp.get_stream(self.device)
            stream.wait_event(self._completion_event, external=stream.is_capturing)

    def get_points(self, mesh: DeformableVisualMesh | SupportsIndex) -> wp.array[wp.vec3]:
        """Return the current point view for one deformable visual mesh.

        Args:
            mesh: Mesh object or invariant index in
                :attr:`newton.Model.deformable_visual_meshes`.

        Returns:
            Zero-copy view of current points [m], shape [mesh.vertex_count, 3].
        """
        self._require_updated()
        start, end = self.mesh_ranges[self._mesh_index(mesh)]
        return self.points[start:end]

    def get_normals(self, mesh: DeformableVisualMesh | SupportsIndex) -> wp.array[wp.vec3]:
        """Return the current unit-normal view for one deformable visual mesh.

        Args:
            mesh: Mesh object or invariant index in
                :attr:`newton.Model.deformable_visual_meshes`.

        Returns:
            Zero-copy view of unit normals, shape [mesh.vertex_count, 3].
        """
        self._require_updated()
        start, end = self.mesh_ranges[self._mesh_index(mesh)]
        return self.normals[start:end]


@wp.kernel
def _skin_deformable_visual_mesh_particle(
    particle_q: wp.array[wp.vec3],
    parent: wp.array[wp.int32],
    out_offset: int,
    out_points: wp.array[wp.vec3],
):
    i = wp.tid()
    out_points[out_offset + i] = particle_q[parent[i]]


@wp.kernel
def _skin_deformable_visual_mesh_triangle(
    particle_q: wp.array[wp.vec3],
    tri_indices: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    weights: wp.array[wp.vec3],
    out_offset: int,
    out_points: wp.array[wp.vec3],
):
    i = wp.tid()
    t = parent[i]
    w = weights[i]
    out_points[out_offset + i] = (
        w[0] * particle_q[tri_indices[3 * t + 0]]
        + w[1] * particle_q[tri_indices[3 * t + 1]]
        + w[2] * particle_q[tri_indices[3 * t + 2]]
    )


@wp.kernel
def _skin_deformable_visual_mesh_tet(
    particle_q: wp.array[wp.vec3],
    tet_indices: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    weights: wp.array[wp.vec4],
    out_offset: int,
    out_points: wp.array[wp.vec3],
):
    i = wp.tid()
    t = parent[i]
    w = weights[i]
    out_points[out_offset + i] = (
        w[0] * particle_q[tet_indices[4 * t + 0]]
        + w[1] * particle_q[tet_indices[4 * t + 1]]
        + w[2] * particle_q[tet_indices[4 * t + 2]]
        + w[3] * particle_q[tet_indices[4 * t + 3]]
    )


@wp.kernel
def _skin_deformable_visual_mesh_body(
    body_q: wp.array[wp.transform],
    parent: wp.array[wp.int32],
    local_offsets: wp.array[wp.vec3],
    out_offset: int,
    out_points: wp.array[wp.vec3],
):
    i = wp.tid()
    out_points[out_offset + i] = wp.transform_point(body_q[parent[i]], local_offsets[i])


@wp.kernel
def _accumulate_face_normals(
    points: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    point_offset: int,
    normal_offset: int,
    normals: wp.array[wp.vec3],
):
    # Face normals are weighted by triangle area (the un-normalized cross
    # product), yielding area-weighted vertex normals after accumulation.
    f = wp.tid()
    i0 = indices[3 * f + 0]
    i1 = indices[3 * f + 1]
    i2 = indices[3 * f + 2]
    n = wp.cross(
        points[point_offset + i1] - points[point_offset + i0],
        points[point_offset + i2] - points[point_offset + i0],
    )
    wp.atomic_add(normals, normal_offset + i0, n)
    wp.atomic_add(normals, normal_offset + i1, n)
    wp.atomic_add(normals, normal_offset + i2, n)


@wp.kernel
def _normalize_normals(normals: wp.array[wp.vec3], normal_offset: int):
    i = wp.tid()
    n = normals[normal_offset + i]
    length = wp.length(n)
    if length > 1.0e-12:
        normals[normal_offset + i] = n / length


def skin_deformable_visual_mesh(
    mesh: DeformableVisualMesh,
    state,
    model,
    out_points: wp.array[wp.vec3],
    device=None,
    out_offset: int = 0,
) -> None:
    """Evaluate a visual mesh's current vertex positions from the simulation state.

    Writes the skinned positions in the simulation frame into ``out_points``
    (shape [vertex_count, 3]); world offsets and layer transforms are the
    consumer's responsibility so viewers, sensors, and external integrations can
    apply their own placement. Runs entirely on ``device`` (the simulation
    device by default).
    """
    device = device if device is not None else out_points.device
    kind = DeformableVisualMesh.Kind
    if mesh.kind == kind.TET:
        wp.launch(
            _skin_deformable_visual_mesh_tet,
            dim=mesh.vertex_count,
            inputs=[state.particle_q, model.tet_indices.flatten(), mesh.parent, mesh.weights, out_offset],
            outputs=[out_points],
            device=device,
        )
    elif mesh.kind == kind.TRIANGLE:
        wp.launch(
            _skin_deformable_visual_mesh_triangle,
            dim=mesh.vertex_count,
            inputs=[state.particle_q, model.tri_indices.flatten(), mesh.parent, mesh.weights, out_offset],
            outputs=[out_points],
            device=device,
        )
    elif mesh.kind == kind.BODY:
        wp.launch(
            _skin_deformable_visual_mesh_body,
            dim=mesh.vertex_count,
            inputs=[state.body_q, mesh.parent, mesh.local_offsets, out_offset],
            outputs=[out_points],
            device=device,
        )
    else:
        wp.launch(
            _skin_deformable_visual_mesh_particle,
            dim=mesh.vertex_count,
            inputs=[state.particle_q, mesh.parent, out_offset],
            outputs=[out_points],
            device=device,
        )


def compute_deformable_visual_mesh_normals(
    points: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    out_normals: wp.array[wp.vec3],
    device=None,
    point_offset: int = 0,
    normal_offset: int = 0,
    vertex_count: int | None = None,
    clear: bool = True,
) -> None:
    """Recompute area-weighted vertex normals from current positions and topology."""
    device = device if device is not None else out_normals.device
    if vertex_count is None:
        vertex_count = len(out_normals) - normal_offset
    if clear:
        if normal_offset != 0 or vertex_count != len(out_normals):
            raise ValueError("Partial normal updates require clear=False and a pre-cleared output array")
        out_normals.zero_()
    wp.launch(
        _accumulate_face_normals,
        dim=len(indices) // 3,
        inputs=[points, indices, point_offset, normal_offset, out_normals],
        device=device,
    )
    wp.launch(_normalize_normals, dim=vertex_count, inputs=[out_normals, normal_offset], device=device)
