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

from enum import IntEnum

import numpy as np
import warp as wp


class DeformableVisualMesh:
    """A visual mesh skinned from a deformable's simulation state.

    Instances are created by :meth:`newton.ModelBuilder.finalize` from meshes
    registered with :meth:`newton.ModelBuilder.add_deformable_visual_mesh` and
    stored in :attr:`newton.Model.deformable_visual_meshes`. Bind-pose vertices
    and topology are immutable asset data. The :attr:`parent`, :attr:`weights`,
    and :attr:`local_offsets` arrays describe how simulation elements move the
    visual vertices.

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
        self.kind = DeformableVisualMesh.Kind(kind)
        """Embedding kind (see :class:`DeformableVisualMesh.Kind`)."""
        self.rest_vertices = rest_vertices
        """Authored bind-pose visual vertices [m], shape [vertex_count, 3].
        Copies retain this frame; evaluated points follow the copied simulation."""
        self.indices = indices
        """Flattened triangle indices into the visual vertices, shape [tri_count*3]."""
        self.parent = parent
        """Per-visual-vertex driver index, shape [vertex_count]. A particle index
        for :attr:`Kind.PARTICLE`, a triangle index into
        :attr:`newton.Model.tri_indices` for :attr:`Kind.TRIANGLE`, a tetrahedron
        index into :attr:`newton.Model.tet_indices` for :attr:`Kind.TET`, and a
        body index into ``State.body_q`` for :attr:`Kind.BODY`. A body parent
        of ``-1`` fixes the vertex in world space after fixed-joint collapse."""
        self.weights = weights
        """Barycentric weights, shape [vertex_count, 4] (vec4) for
        :attr:`Kind.TET`, shape [vertex_count, 3] (vec3) for
        :attr:`Kind.TRIANGLE`; ``None`` for other kinds."""
        self.local_offsets = local_offsets
        """Body-local bind offsets [m] for :attr:`Kind.BODY`,
        shape [vertex_count, 3]; ``None`` for other kinds."""
        self.uvs = uvs
        """Per-visual-vertex texture coordinates, shape [vertex_count, 2], or ``None``."""
        self.texture = texture
        """Albedo texture as an image array (H, W, C) or a path, or ``None``.
        Viewers may cache this asset; do not edit its image contents in place."""
        self.world = world
        """World index this visual mesh belongs to (-1 for global)."""
        self.label = label
        """Display label. Not unique; use :attr:`index` for a stable identity."""
        self.index = index
        """Invariant index of this mesh in :attr:`newton.Model.deformable_visual_meshes`."""

    @property
    def vertex_count(self) -> int:
        """Number of visual mesh vertices."""
        return 0 if self.rest_vertices is None else len(self.rest_vertices)
