# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Visual payloads embedded in deformable bodies.

Visual samples can be embedded in a coarse simulation deformable and evaluated
from simulation state each frame. The simulation continues to run on the coarse
representation; these visual meshes are used for visualization only and
never participate in the solve or in collision. Triangle meshes are supported.

See :meth:`newton.ModelBuilder.add_deformable_visual_mesh`.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from enum import IntEnum
from typing import TYPE_CHECKING, Any, SupportsIndex

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from ..sim.model import Model
    from ..sim.state import State


class DeformableVisualMesh:
    """A visual mesh skinned from a deformable's simulation state.

    Instances are created by :meth:`newton.ModelBuilder.finalize` from meshes
    registered with :meth:`newton.ModelBuilder.add_deformable_visual_mesh` and
    stored in :attr:`newton.Model.deformable_visual_meshes`. Treat a mesh as fixed
    asset data. Every attribute except :attr:`label` and :attr:`texture` is a
    read-only property. Do not modify the returned arrays in place: consumers
    may read them only once when they are created, and meshes
    with the same rest geometry, such as replicated copies, share one device copy
    of :attr:`rest_vertices`, :attr:`indices` and :attr:`uvs`. The :attr:`parent`,
    :attr:`weights`, and :attr:`local_offsets` arrays describe how simulation
    elements move the visual vertices.

    Attributes are device :class:`warp.array` objects unless noted otherwise.

    .. experimental::

        This API may change without a formal deprecation cycle.
    """

    class Kind(IntEnum):
        """How visual samples are bound to simulation elements."""

        PARTICLE = 0
        """Each visual sample is bound to one simulation particle. This is a
        shared or one-to-one map, not a general high-resolution surface
        embedding; use :attr:`TRIANGLE` for an independently discretized
        surface."""

        TRIANGLE = 1
        """Each visual sample is embedded in a simulation triangle with three
        barycentric weights. Its current position is the weighted sum of the
        triangle's particle positions; normal offsets are not retained."""

        TETRAHEDRON = 2
        """Each visual sample is embedded in a simulation tetrahedron with four
        barycentric weights. Its current position is the weighted sum of the
        tetrahedron's particle positions."""

        BODY = 3
        """Each visual sample is rigidly bound to one body by a body-local
        offset. A visual surface bound to separate bodies can show seams at
        body boundaries."""

    def __init__(
        self,
        kind: Kind,
        rest_vertices: wp.array[wp.vec3],
        indices: wp.array[wp.int32],
        parent: wp.array[wp.int32],
        *,
        weights: wp.array[wp.vec4] | wp.array[wp.vec3] | None = None,
        local_offsets: wp.array[wp.vec3] | None = None,
        uvs: wp.array[wp.vec2] | None = None,
        texture: np.ndarray | str | None = None,
        world: int = -1,
        label: str = "",
        index: int = -1,
    ) -> None:
        """Store a visual mesh and its simulation binding.

        Args:
            kind: Simulation element kind driving the visual vertices.
            rest_vertices: Bind-pose visual positions [m], shape [vertex_count, 3].
            indices: Flattened triangle vertex indices.
            parent: Simulation driver index for each visual vertex.
            weights: Triangle or tetrahedron barycentric weights, when needed.
            local_offsets: Body-local offsets [m] for body bindings.
            uvs: Per-vertex texture coordinates, or ``None``.
            texture: Albedo image array or image path, or ``None``.
            world: Owning model world, or ``-1`` for global geometry.
            label: Display label.
            index: Index in :attr:`newton.Model.deformable_visual_meshes`.
        """
        self._kind = DeformableVisualMesh.Kind(kind)
        self._rest_vertices = rest_vertices
        self._indices = indices
        self._parent = parent
        self._weights = weights
        self._local_offsets = local_offsets
        self._uvs = uvs
        self.texture = texture
        """Albedo texture as an image array (H, W, C) or a path, or ``None``. Viewers
        may cache this asset, so replace the value instead of editing its image
        contents in place; the USD-based viewers keep the texture they first
        exported for a mesh."""
        self._world = world
        self._index = index
        self.label = label
        """Display label. Not unique; use :attr:`index` for a stable identity."""

    @property
    def kind(self) -> Kind:
        """Embedding kind (see :class:`DeformableVisualMesh.Kind`). Read-only."""
        return self._kind

    @property
    def rest_vertices(self) -> wp.array[wp.vec3]:
        """Authored bind-pose visual vertices [m], shape [vertex_count, 3]. Read-only.
        Copies retain this frame; evaluated points follow the copied simulation."""
        return self._rest_vertices

    @property
    def indices(self) -> wp.array[wp.int32]:
        """Flattened triangle indices into the visual vertices, shape [tri_count*3]. Read-only."""
        return self._indices

    @property
    def parent(self) -> wp.array[wp.int32]:
        """Per-visual-vertex driver index, shape [vertex_count]. Read-only. A particle
        index for :attr:`Kind.PARTICLE`, a triangle index into
        :attr:`newton.Model.tri_indices` for :attr:`Kind.TRIANGLE`, a tetrahedron
        index into :attr:`newton.Model.tet_indices` for :attr:`Kind.TETRAHEDRON`, and a
        body index into ``State.body_q`` for :attr:`Kind.BODY`. A body parent
        of ``-1`` fixes the vertex in world space after fixed-joint collapse."""
        return self._parent

    @property
    def weights(self) -> wp.array[wp.vec4] | wp.array[wp.vec3] | None:
        """Barycentric weights, shape [vertex_count, 4] (vec4) for
        :attr:`Kind.TETRAHEDRON`, shape [vertex_count, 3] (vec3) for
        :attr:`Kind.TRIANGLE`; ``None`` for other kinds. Read-only."""
        return self._weights

    @property
    def local_offsets(self) -> wp.array[wp.vec3] | None:
        """Body-local bind offsets [m] for :attr:`Kind.BODY`,
        shape [vertex_count, 3]; ``None`` for other kinds. Read-only."""
        return self._local_offsets

    @property
    def uvs(self) -> wp.array[wp.vec2] | None:
        """Per-visual-vertex texture coordinates, shape [vertex_count, 2], or ``None``. Read-only."""
        return self._uvs

    @property
    def world(self) -> int:
        """World index this visual mesh belongs to (-1 for global). Read-only."""
        return self._world

    @property
    def index(self) -> int:
        """Invariant index of this mesh in :attr:`newton.Model.deformable_visual_meshes`. Read-only."""
        return self._index

    @property
    def vertex_count(self) -> int:
        """Number of visual mesh vertices."""
        return len(self._rest_vertices)


class DeformableVisuals:
    """Current evaluated geometry for a model's deformable visuals.

    Allocate this result with :meth:`newton.Model.deformable_visuals` and
    populate it with :meth:`newton.Model.update_deformable_visuals`. One result
    can be reused across states and ordered graph replays. Update or replay
    before consumption; this output does not track its source State or freshness.
    Finish consumption before overwriting the buffers. Allocate separate results
    for simultaneous snapshots or concurrent updates.
    These output buffers do not support differentiation through skinning.

    .. experimental::

        This API and its synchronization contract may change without a formal
        deprecation cycle while deformable visual support is experimental.
    """

    def __init__(self, model: Model) -> None:
        """Allocate fixed visual output buffers for a model.

        Args:
            model: Model whose visual meshes define the output layout.
        """
        self._model = model
        self.device = model.device
        """Device containing :attr:`points` and :attr:`normals`."""

        # Keep the batches this output is sized for, so updates skin exactly these meshes. A model
        # restored from a recording builds them here.
        self._skin_batches = model._get_deformable_visual_skin_batches()
        self.mesh_ranges = self._skin_batches.mesh_ranges
        """Stable ``[start, end)`` vertex range for each visual mesh."""
        self.vertex_count = self._skin_batches.vertex_count
        """Total number of current visual vertices."""
        self.points = wp.empty(self.vertex_count, dtype=wp.vec3, device=self.device)
        """Current skinned visual points [m], shape [vertex_count, 3]."""
        self.normals = wp.zeros(self.vertex_count, dtype=wp.vec3, device=self.device)
        """Current visual unit normals, shape [vertex_count, 3]."""

        self._updated = False
        self._completion_event = wp.Event(self.device) if self.device.is_cuda else None

    @property
    def model(self) -> Model:
        """Model whose visual mesh layout this result uses."""
        return self._model

    def _mesh_index(self, mesh: DeformableVisualMesh | SupportsIndex) -> int:
        if isinstance(mesh, DeformableVisualMesh):
            index = mesh.index
            # Compare with the meshes these buffers were sized for, not the model's current tuple.
            if index < 0 or index >= len(self.mesh_ranges) or self._skin_batches.meshes[index] is not mesh:
                raise ValueError("The deformable visual mesh does not belong to this DeformableVisuals model.")
            return index

        try:
            index = operator.index(mesh)
        except TypeError as exc:
            raise TypeError("mesh must be a DeformableVisualMesh or an integer mesh index") from exc
        if index < 0 or index >= len(self.mesh_ranges):
            raise IndexError(f"Deformable visual mesh index {index} is out of range")
        return index

    def _require_updated(self) -> None:
        if not self._updated:
            raise RuntimeError(
                "DeformableVisuals has not been updated; call model.update_deformable_visuals(state, visuals) first."
            )

    def _validate_model(self, model: Model) -> None:
        if self._model is not model:
            raise ValueError("DeformableVisuals was created for another model.")

    def _mark_updated(self) -> None:
        # This records a submitted update, not graph execution or buffer freshness.
        self._updated = True
        if self._completion_event is not None:
            stream = wp.get_stream(self.device)
            stream.record_event(self._completion_event, external=stream.is_capturing)

    def wait(self, stream: wp.Stream | None = None) -> None:
        """Make a device stream wait for the most recent visual update.

        Launch any captured update before calling this method. The wait does not
        prevent another update from overwriting buffers still in use by a consumer.

        Args:
            stream: Consumer stream. Uses the current stream on :attr:`device`
                when omitted. This method is a no-op on CPU.

        Raises:
            RuntimeError: If the output has not been updated.
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

        Raises:
            RuntimeError: If the output has not been updated.
            TypeError: If ``mesh`` is not a visual mesh or integer index.
            IndexError: If the index is out of range.
            ValueError: If the mesh belongs to another model.
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

        Raises:
            RuntimeError: If the output has not been updated.
            TypeError: If ``mesh`` is not a visual mesh or integer index.
            IndexError: If the index is out of range.
            ValueError: If the mesh belongs to another model.
        """
        self._require_updated()
        start, end = self.mesh_ranges[self._mesh_index(mesh)]
        return self.normals[start:end]


@wp.kernel
def _skin_deformable_visual_batch_particle(
    particle_q: wp.array[wp.vec3],
    parent: wp.array[wp.int32],
    dest: wp.array[wp.int32],
    out_offset: int,
    out_points: wp.array[wp.vec3],
):
    i = wp.tid()
    out_points[out_offset + dest[i]] = particle_q[parent[i]]


@wp.kernel
def _skin_deformable_visual_batch_triangle(
    particle_q: wp.array[wp.vec3],
    tri_indices: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    weights: wp.array[wp.vec3],
    dest: wp.array[wp.int32],
    out_offset: int,
    out_points: wp.array[wp.vec3],
):
    i = wp.tid()
    t = parent[i]
    w = weights[i]
    out_points[out_offset + dest[i]] = (
        w[0] * particle_q[tri_indices[3 * t + 0]]
        + w[1] * particle_q[tri_indices[3 * t + 1]]
        + w[2] * particle_q[tri_indices[3 * t + 2]]
    )


@wp.kernel
def _skin_deformable_visual_batch_tet(
    particle_q: wp.array[wp.vec3],
    tet_indices: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    weights: wp.array[wp.vec4],
    dest: wp.array[wp.int32],
    out_offset: int,
    out_points: wp.array[wp.vec3],
):
    i = wp.tid()
    t = parent[i]
    w = weights[i]
    out_points[out_offset + dest[i]] = (
        w[0] * particle_q[tet_indices[4 * t + 0]]
        + w[1] * particle_q[tet_indices[4 * t + 1]]
        + w[2] * particle_q[tet_indices[4 * t + 2]]
        + w[3] * particle_q[tet_indices[4 * t + 3]]
    )


@wp.kernel
def _skin_deformable_visual_batch_body(
    body_q: wp.array[wp.transform],
    parent: wp.array[wp.int32],
    local_offsets: wp.array[wp.vec3],
    dest: wp.array[wp.int32],
    out_offset: int,
    out_points: wp.array[wp.vec3],
):
    i = wp.tid()
    point = local_offsets[i]
    if parent[i] >= 0:
        point = wp.transform_point(body_q[parent[i]], point)
    out_points[out_offset + dest[i]] = point


class _DeformableVisualSkinBatches:
    """Skin every visual mesh of a model into one vertex buffer with one launch per binding kind,
    and shade all of them with two launches.

    The bindings of all meshes of one :class:`DeformableVisualMesh.Kind` live in one array per
    driver, together with each vertex's destination index, so the launch count is independent
    of the number of meshes, which replication multiplies by the world count.
    :meth:`newton.ModelBuilder.finalize` builds the batches from its host data and gives each
    mesh views into them (see :meth:`mesh_bindings`), so a model holds one copy of its binding
    data. A model restored from a recording builds them from its meshes on first use; those
    meshes keep their own arrays, which other models loaded from the same recording may share.
    :attr:`meshes` is the tuple the batches were built for; consumers size their buffers from it.
    """

    def __init__(
        self,
        kinds: Sequence[DeformableVisualMesh.Kind],
        parents: Sequence[np.ndarray],
        drivers: Sequence[np.ndarray | None],
        indices: Sequence[np.ndarray],
        device: wp.Device,
    ) -> None:
        """Concatenate per-mesh host bindings per kind and upload each kind once.

        Args:
            kinds: Binding kind of each mesh, in model order.
            parents: Driver index of every vertex, per mesh.
            drivers: Barycentric weights (triangle, tetrahedron) or body-local offsets (body)
                per mesh; ``None`` for particle bindings.
            indices: Flattened triangle indices of each mesh, into its own vertices.
            device: Device that holds the concatenated arrays and runs the launches.
        """
        kind = DeformableVisualMesh.Kind
        self.device = device
        self.needs_particles = any(k != kind.BODY for k in kinds)
        """Whether skinning reads ``State.particle_q``."""
        self.needs_bodies = any(k == kind.BODY for k in kinds)
        """Whether skinning reads ``State.body_q``."""
        self.meshes: tuple[DeformableVisualMesh, ...] = ()
        """The mesh tuple these batches skin, in model order, once the meshes exist."""
        groups: dict[DeformableVisualMesh.Kind, list[int]] = {k: [] for k in kind}
        starts: list[int] = []
        start = 0
        for index, (k, parent) in enumerate(zip(kinds, parents, strict=True)):
            groups[kind(k)].append(index)
            starts.append(start)
            start += len(parent)
        self.vertex_count = start
        """Number of destination vertices the batches write, the sum over all meshes."""
        self.mesh_ranges = tuple((s, s + len(parent)) for s, parent in zip(starts, parents, strict=True))
        """Destination ``[start, end)`` vertex range of each mesh, in model order; consumers size and
        index their buffers from it so that it agrees with the destination indices by construction."""
        faces = [np.asarray(idx, dtype=np.int32).reshape(-1) + s for idx, s in zip(indices, starts, strict=True)]
        self.face_indices = wp.array(
            np.concatenate(faces) if faces else np.empty(0, dtype=np.int32), dtype=wp.int32, device=device
        )
        """Triangle indices of every mesh into the destination vertices, so one normal pass covers all meshes."""
        self._batches: list[
            tuple[DeformableVisualMesh.Kind, wp.array[wp.int32], wp.array[Any] | None, wp.array[wp.int32]]
        ] = []
        # Position of each mesh inside its kind's arrays: (batch, offset, length).
        self._mesh_slices: list[tuple[int, int, int]] = [(0, 0, 0)] * len(kinds)
        for k, members in groups.items():
            if not members:
                continue
            offset = 0
            for index in members:
                self._mesh_slices[index] = (len(self._batches), offset, len(parents[index]))
                offset += len(parents[index])
            parent = np.concatenate([np.asarray(parents[i], dtype=np.int32) for i in members])
            dest = np.concatenate([np.arange(starts[i], starts[i] + len(parents[i]), dtype=np.int32) for i in members])
            driver = None
            if k != kind.PARTICLE:
                values = np.concatenate([np.asarray(drivers[i], dtype=np.float32) for i in members])
                driver = wp.array(values, dtype=wp.vec4 if k == kind.TETRAHEDRON else wp.vec3, device=device)
            self._batches.append(
                (
                    k,
                    wp.array(parent, dtype=wp.int32, device=device),
                    driver,
                    wp.array(dest, dtype=wp.int32, device=device),
                )
            )

    @classmethod
    def from_meshes(cls, meshes: Sequence[DeformableVisualMesh], device: wp.Device) -> _DeformableVisualSkinBatches:
        """Build the batches from existing meshes, reading each binding array back once."""
        kind = DeformableVisualMesh.Kind
        drivers: list[np.ndarray | None] = []
        for mesh in meshes:
            if mesh.kind == kind.BODY:
                drivers.append(mesh.local_offsets.numpy())
            elif mesh.kind == kind.PARTICLE:
                drivers.append(None)
            else:
                drivers.append(mesh.weights.numpy())
        batches = cls(
            [mesh.kind for mesh in meshes],
            [mesh.parent.numpy() for mesh in meshes],
            drivers,
            [mesh.indices.numpy() for mesh in meshes],
            device,
        )
        batches.meshes = tuple(meshes)
        return batches

    def mesh_bindings(self, index: int) -> tuple[wp.array[wp.int32], wp.array[Any] | None]:
        """Return views of mesh ``index``'s ``parent`` and driver arrays into its kind's arrays."""
        batch, offset, length = self._mesh_slices[index]
        _, parent, driver, _ = self._batches[batch]
        return parent[offset : offset + length], None if driver is None else driver[offset : offset + length]

    def skin(self, state: State, model: Model, out_points: wp.array[wp.vec3], out_offset: int = 0) -> None:
        """Write the current skinned positions [m] of every mesh into ``out_points``.

        Args:
            state: Simulation state providing current particle positions and body poses.
            model: Model providing triangle and tetrahedron topology.
            out_points: Destination positions [m].
            out_offset: Added to every destination index, for buffers that hold other
                vertices before the visual meshes.

        Raises:
            ValueError: If ``out_points`` cannot hold every destination vertex.
        """
        if out_offset + self.vertex_count > out_points.shape[0]:
            raise ValueError(
                f"out_points holds {out_points.shape[0]} vertices but the visual meshes need "
                f"{out_offset + self.vertex_count}"
            )
        kind = DeformableVisualMesh.Kind
        for k, parent, drivers, dest in self._batches:
            if k == kind.TETRAHEDRON:
                inputs = [state.particle_q, model.tet_indices.flatten(), parent, drivers, dest, out_offset]
                kernel = _skin_deformable_visual_batch_tet
            elif k == kind.TRIANGLE:
                inputs = [state.particle_q, model.tri_indices.flatten(), parent, drivers, dest, out_offset]
                kernel = _skin_deformable_visual_batch_triangle
            elif k == kind.BODY:
                inputs = [state.body_q, parent, drivers, dest, out_offset]
                kernel = _skin_deformable_visual_batch_body
            else:
                inputs = [state.particle_q, parent, dest, out_offset]
                kernel = _skin_deformable_visual_batch_particle
            wp.launch(kernel, dim=dest.shape[0], inputs=inputs, outputs=[out_points], device=self.device)

    def compute_normals(self, points: wp.array[wp.vec3], out_normals: wp.array[wp.vec3]) -> None:
        """Write unit area-weighted vertex normals of every mesh into ``out_normals``.

        Args:
            points: Skinned positions [m] of every destination vertex, as written by :meth:`skin`
                with ``out_offset=0``.
            out_normals: Destination normals, one per destination vertex; cleared first.
        """
        from ..utils.mesh import compute_vertex_normals  # noqa: PLC0415

        compute_vertex_normals(points, self.face_indices, out_normals, device=self.device)
