# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Visual payloads embedded in deformable bodies.

Visual samples can be embedded in a coarse simulation deformable and evaluated
from simulation state each frame. The simulation continues to run on the coarse
representation; these payloads are visualization and sensor geometry only and
never participate in the solve or in collision. Triangle meshes are currently
the supported visual payload.

See :meth:`newton.ModelBuilder.add_deformable_visual_mesh`.
"""

from __future__ import annotations

import operator
from enum import IntEnum
from typing import TYPE_CHECKING, SupportsIndex

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from ..geometry import Gaussian
    from .model import Model
    from .state import State


class DeformableVisualBinding:
    """Payload-neutral binding from visual points to simulation drivers.

    The binding is independent of the payload being skinned. Today the payload
    is a triangle mesh; the same binding data can later drive other visual
    payloads, such as Gaussian splats, without changing how importer code
    selects simulation drivers.
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

        TET = 2
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
        parent: wp.array[wp.int32],
        weights: wp.array[wp.vec4] | wp.array[wp.vec3] | None = None,
        local_offsets: wp.array[wp.vec3] | None = None,
    ) -> None:
        """Store a visual-to-simulation binding.

        Args:
            kind: Simulation element kind driving the visual points.
            parent: Driver index for each visual point.
            weights: Barycentric weights for triangle or tetrahedron bindings.
            local_offsets: Body-local binding offsets [m].
        """
        self.kind = kind
        """Embedding kind."""
        self.parent = parent
        """Per-visual-point driver index, shape [point_count]."""
        self.weights = weights
        """Barycentric weights for triangle or tet bindings, or ``None``."""
        self.local_offsets = local_offsets
        """Body-local offsets [m] for rigid body bindings, or ``None``."""


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

    Kind = DeformableVisualBinding.Kind
    """Compatibility alias for :class:`DeformableVisualBinding.Kind`."""

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
    ) -> None:
        self.kind = DeformableVisualBinding.Kind(kind)
        """Embedding kind (see :class:`DeformableVisualBinding.Kind`)."""
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
        """Body-local bind offsets [m] for :attr:`Kind.BODY`,
        shape [vertex_count, 3]; ``None`` for other kinds."""
        self.uvs = uvs
        """Per-visual-vertex texture coordinates, shape [vertex_count, 2], or ``None``."""
        self.texture = texture
        """Albedo texture as an image array (H, W, C) or a path, or ``None``."""
        self.world = world
        """World index this visual mesh belongs to (-1 for global)."""
        self.label = label
        """Display label. Imported path labels follow requested namespace rebasing.
        Not unique; use :attr:`index` for a stable identity."""
        self.index = index
        """Invariant index of this mesh in :attr:`newton.Model.deformable_visual_meshes`."""
        self.body_path = body_path
        """USD path of the owning ``PhysicsDeformableBodyAPI`` prim, or ``None``
        when created programmatically. Replicated imported meshes use the
        destination path when namespace rebasing is requested."""
        self.sim_path = sim_path
        """USD path of the owning simulation geometry prim, or ``None`` when
        created programmatically. Replicated imported meshes use the
        destination path when namespace rebasing is requested."""
        self.graphics_path = graphics_path
        """USD path of the graphics geometry prim, or ``None`` when
        created programmatically. Replicated imported meshes use the
        destination path when namespace rebasing is requested."""

    @property
    def vertex_count(self) -> int:
        return 0 if self.rest_vertices is None else len(self.rest_vertices)


class DeformableVisualGaussian:
    """A Gaussian field whose samples are bound to a deformable body.

    Instances are immutable model output created by
    :meth:`newton.ModelBuilder.add_deformable_visual_gaussian`. Current sample
    transforms and scales will be stored separately in :class:`DeformableVisuals`.

    .. experimental::

        This API may change without a formal deprecation cycle while deformable
        visual support is experimental.
    """

    def __init__(
        self,
        gaussian: Gaussian,
        binding: DeformableVisualBinding,
        rest_rotations: wp.array[wp.quat],
        rest_scales: wp.array[wp.vec3],
        world: int = -1,
        label: str = "",
        index: int = -1,
        body_path: str | None = None,
        sim_path: str | None = None,
        graphics_path: str | None = None,
    ) -> None:
        """Store a Gaussian visual and its simulation binding.

        Args:
            gaussian: Immutable rest Gaussian appearance.
            binding: Binding from Gaussian centers to simulation drivers.
            rest_rotations: Rest Gaussian orientations, shape [count].
            rest_scales: Rest Gaussian axis scales [m], shape [count, 3].
            world: Owning model world, or ``-1`` for global visuals.
            label: Display label.
            index: Stable index in :attr:`newton.Model.deformable_visual_gaussians`.
            body_path: Owning deformable body USD path, when imported.
            sim_path: Driving simulation geometry USD path, when imported.
            graphics_path: Gaussian graphics USD path, when imported.
        """
        self.gaussian = gaussian
        """Immutable rest Gaussian appearance."""
        self.binding = binding
        """Binding from Gaussian centers to simulation drivers."""
        self.kind = binding.kind
        """Embedding kind (see :class:`DeformableVisualBinding.Kind`)."""
        self.parent = binding.parent
        """Per-Gaussian simulation driver indices, shape [count]."""
        self.weights = binding.weights
        """Barycentric weights, or ``None`` for non-barycentric bindings."""
        self.local_offsets = binding.local_offsets
        """Body-local offsets [m], or ``None`` for non-body bindings."""
        self.rest_rotations = rest_rotations
        """Rest Gaussian orientations, shape [count]."""
        self.rest_scales = rest_scales
        """Rest Gaussian axis scales [m], shape [count, 3]."""
        self.world = world
        """Owning model world, or ``-1`` for a global visual."""
        self.label = label
        """Display label."""
        self.index = index
        """Stable index in :attr:`newton.Model.deformable_visual_gaussians`."""
        self.body_path = body_path
        """Owning deformable body USD path, or ``None``."""
        self.sim_path = sim_path
        """Driving simulation geometry USD path, or ``None``."""
        self.graphics_path = graphics_path
        """Gaussian graphics USD path, or ``None``."""

    @property
    def count(self) -> int:
        """Number of Gaussian samples."""
        return self.gaussian.count


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

    def __init__(self, model: Model) -> None:
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

        ranges = []
        gaussian_start = 0
        for gaussian in model.deformable_visual_gaussians:
            gaussian_end = gaussian_start + gaussian.count
            ranges.append((gaussian_start, gaussian_end))
            gaussian_start = gaussian_end
        self.gaussian_ranges = tuple(ranges)
        """Stable ``[start, end)`` sample range for each Gaussian visual."""
        self.gaussian_count = gaussian_start
        """Total number of current Gaussian samples."""
        self.gaussian_transforms = wp.empty(gaussian_start, dtype=wp.transform, device=self.device)
        """Current Gaussian centers [m] and orientations, shape [gaussian_count]."""
        self.gaussian_scales = wp.empty(gaussian_start, dtype=wp.vec3, device=self.device)
        """Current positive Gaussian axis scales [m], shape [gaussian_count, 3]."""

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

    def _gaussian_index(self, gaussian: DeformableVisualGaussian | SupportsIndex) -> int:
        if isinstance(gaussian, DeformableVisualGaussian):
            index = gaussian.index
            if (
                index < 0
                or index >= len(self.gaussian_ranges)
                or self._model.deformable_visual_gaussians[index] is not gaussian
            ):
                raise ValueError("The deformable Gaussian visual does not belong to this DeformableVisuals model.")
            return index
        try:
            index = operator.index(gaussian)
        except TypeError as exc:
            raise TypeError("gaussian must be a DeformableVisualGaussian or an integer index") from exc
        if index < 0 or index >= len(self.gaussian_ranges):
            raise IndexError(f"Deformable Gaussian visual index {index} is out of range")
        return index

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

    def get_gaussian_transforms(self, gaussian: DeformableVisualGaussian | SupportsIndex) -> wp.array[wp.transform]:
        """Return the current transform view for one Gaussian visual.

        Args:
            gaussian: Gaussian visual or stable model index.

        Returns:
            Zero-copy current center/orientation transforms, shape [count].
        """
        self._require_updated()
        start, end = self.gaussian_ranges[self._gaussian_index(gaussian)]
        return self.gaussian_transforms[start:end]

    def get_gaussian_scales(self, gaussian: DeformableVisualGaussian | SupportsIndex) -> wp.array[wp.vec3]:
        """Return the current scale view for one Gaussian visual.

        Args:
            gaussian: Gaussian visual or stable model index.

        Returns:
            Zero-copy positive axis scales [m], shape [count, 3].
        """
        self._require_updated()
        start, end = self.gaussian_ranges[self._gaussian_index(gaussian)]
        return self.gaussian_scales[start:end]


@wp.kernel(enable_backward=False)
def _skin_deformable_visual_gaussian_tet(
    particle_q: wp.array[wp.vec3],
    tet_indices: wp.array2d[wp.int32],
    tet_poses: wp.array[wp.mat33],
    parent: wp.array[wp.int32],
    weights: wp.array[wp.vec4],
    rest_rotations: wp.array[wp.quat],
    rest_scales: wp.array[wp.vec3],
    out_offset: int,
    out_transforms: wp.array[wp.transform],
    out_scales: wp.array[wp.vec3],
):
    i = wp.tid()
    tet = parent[i]
    i0 = tet_indices[tet, 0]
    i1 = tet_indices[tet, 1]
    i2 = tet_indices[tet, 2]
    i3 = tet_indices[tet, 3]
    q0 = particle_q[i0]
    q1 = particle_q[i1]
    q2 = particle_q[i2]
    q3 = particle_q[i3]
    w = weights[i]
    center = w[0] * q0 + w[1] * q1 + w[2] * q2 + w[3] * q3

    current_basis = wp.matrix_from_cols(q1 - q0, q2 - q0, q3 - q0)
    deformation_gradient = current_basis * tet_poses[tet]
    rest_scale = rest_scales[i]
    scale_matrix = wp.mat33(rest_scale[0], 0.0, 0.0, 0.0, rest_scale[1], 0.0, 0.0, 0.0, rest_scale[2])
    axes = deformation_gradient * wp.quat_to_matrix(rest_rotations[i]) * scale_matrix
    rotation, singular_values, _ = wp.svd3(axes)
    if wp.determinant(rotation) < 0.0:
        rotation = wp.matrix_from_cols(
            wp.vec3(rotation[0, 0], rotation[1, 0], rotation[2, 0]),
            wp.vec3(rotation[0, 1], rotation[1, 1], rotation[2, 1]),
            -wp.vec3(rotation[0, 2], rotation[1, 2], rotation[2, 2]),
        )
    minimum_scale = 1.0e-6
    out_transforms[out_offset + i] = wp.transform(center, wp.quat_from_matrix(rotation))
    out_scales[out_offset + i] = wp.vec3(
        wp.max(wp.abs(singular_values[0]), minimum_scale),
        wp.max(wp.abs(singular_values[1]), minimum_scale),
        wp.max(wp.abs(singular_values[2]), minimum_scale),
    )


def skin_deformable_visual_gaussian(
    visual: DeformableVisualGaussian,
    state: State,
    model: Model,
    out_transforms: wp.array[wp.transform],
    out_scales: wp.array[wp.vec3],
    out_offset: int = 0,
) -> None:
    """Evaluate current transforms and scales for one Gaussian visual."""
    if visual.kind != DeformableVisualBinding.Kind.TET:
        raise ValueError(f"Unsupported deformable Gaussian binding kind {visual.kind}")
    wp.launch(
        _skin_deformable_visual_gaussian_tet,
        dim=visual.count,
        inputs=[
            state.particle_q,
            model.tet_indices,
            model.tet_poses,
            visual.parent,
            visual.weights,
            visual.rest_rotations,
            visual.rest_scales,
            out_offset,
            out_transforms,
            out_scales,
        ],
        device=model.device,
    )


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
    state: State,
    model: Model,
    out_points: wp.array[wp.vec3],
    device: wp.DeviceLike | None = None,
    out_offset: int = 0,
) -> None:
    """Evaluate a visual mesh's current vertex positions from the simulation state.

    Writes the skinned positions in the simulation frame into ``out_points``
    (shape [vertex_count, 3]); world offsets and layer transforms are the
    consumer's responsibility so viewers, sensors, and external integrations can
    apply their own placement. Runs entirely on ``device`` (the simulation
    device by default).

    Args:
        mesh: Visual mesh and its simulation binding.
        state: Simulation state providing current driver positions [m] and poses.
        model: Model providing triangle and tetrahedron topology.
        out_points: Destination positions [m].
        device: Device on which to evaluate the binding.
        out_offset: First destination vertex to write.
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
    device: wp.DeviceLike | None = None,
    point_offset: int = 0,
    normal_offset: int = 0,
    vertex_count: int | None = None,
    clear: bool = True,
) -> None:
    """Recompute area-weighted vertex normals from current positions and topology.

    Args:
        points: Current visual vertex positions [m].
        indices: Flattened triangle vertex indices.
        out_normals: Destination unit normals.
        device: Device on which to compute the normals.
        point_offset: Offset applied to each triangle vertex index.
        normal_offset: First destination normal to write.
        vertex_count: Number of destination normals to normalize.
        clear: Whether to clear the complete destination before accumulating.
    """
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
