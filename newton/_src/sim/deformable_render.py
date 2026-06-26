# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render-mesh embedding data for deformable bodies.

A render mesh is a high-resolution, textured display surface that is embedded in
a coarse simulation deformable (cloth or volumetric soft body) and skinned from
the simulation state each frame. The simulation continues to run on the coarse
mesh; the render mesh is visualization-only and never participates in the solve.

See :meth:`newton.ModelBuilder.add_deformable_render_mesh`.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import warp as wp


class DeformableRenderKind(IntEnum):
    """How a render mesh is embedded into its driving deformable."""

    CLOTH_SHARED = 0
    """Each render vertex is bound to a single simulation particle (surface
    deformable). The deformed position is the particle position directly; used
    when the render mesh shares the simulation topology or maps 1:1 onto it."""

    TET_EMBED = 1
    """Each render vertex is embedded in a tetrahedron of a volumetric soft body
    via barycentric weights. The deformed position is the weighted sum of the
    tet's four particle positions."""


class DeformableRenderMesh:
    """A textured render mesh skinned from a deformable's simulation state.

    The bind-pose (rest) vertices and topology are immutable asset data. The
    per-vertex embedding (``parent`` and, for :attr:`DeformableRenderKind.TET_EMBED`,
    ``weights``) is computed once at :meth:`newton.ModelBuilder.finalize` time and
    used by the viewer to skin the mesh each frame.

    Attributes are device :class:`warp.array` objects unless noted otherwise.
    """

    def __init__(
        self,
        kind: DeformableRenderKind,
        rest_vertices: wp.array[wp.vec3],
        indices: wp.array[wp.int32],
        parent: wp.array[wp.int32],
        weights: wp.array[wp.vec4] | None = None,
        uvs: wp.array[wp.vec2] | None = None,
        normals_rest: wp.array[wp.vec3] | None = None,
        texture: np.ndarray | str | None = None,
        world: int = -1,
        label: str = "",
    ):
        self.kind = DeformableRenderKind(kind)
        """Embedding kind (see :class:`DeformableRenderKind`)."""
        self.rest_vertices = rest_vertices
        """Bind-pose render vertices [m], shape [vertex_count, 3]."""
        self.indices = indices
        """Flattened triangle indices into the render vertices, shape [tri_count*3]."""
        self.parent = parent
        """Per-render-vertex driver index, shape [vertex_count]. For
        :attr:`DeformableRenderKind.CLOTH_SHARED` this is a particle index; for
        :attr:`DeformableRenderKind.TET_EMBED` this is a tetrahedron index into
        :attr:`newton.Model.tet_indices`."""
        self.weights = weights
        """Barycentric weights for :attr:`DeformableRenderKind.TET_EMBED`,
        shape [vertex_count, 4]; ``None`` for cloth meshes."""
        self.uvs = uvs
        """Per-render-vertex texture coordinates, shape [vertex_count, 2], or ``None``."""
        self.normals_rest = normals_rest
        """Bind-pose per-vertex normals, shape [vertex_count, 3], or ``None``."""
        self.texture = texture
        """Albedo texture as an image array (H, W, C) or a path, or ``None``."""
        self.world = world
        """World index this render mesh belongs to (-1 for global)."""
        self.label = label
        """Display label, used to build a stable viewer object name."""

    @property
    def vertex_count(self) -> int:
        return 0 if self.rest_vertices is None else len(self.rest_vertices)
