# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
from collections.abc import Sequence

import numpy as np
import warp as wp

from ..core.types import Devicelike, Vec2, Vec3, nparray, override


class GeoType(enum.IntEnum):
    """
    Enumeration of geometric shape types supported in Newton.

    Each member represents a different primitive or mesh-based geometry
    that can be used for collision, rendering, or simulation.
    """

    PLANE = 0
    """Plane."""

    HFIELD = 1
    """Height field (terrain)."""

    SPHERE = 2
    """Sphere."""

    CAPSULE = 3
    """Capsule (cylinder with hemispherical ends)."""

    ELLIPSOID = 4
    """Ellipsoid."""

    CYLINDER = 5
    """Cylinder."""

    BOX = 6
    """Axis-aligned box."""

    MESH = 7
    """Triangle mesh."""

    SDF = 8
    """Signed distance field."""

    CONE = 9
    """Cone."""

    CONVEX_MESH = 10
    """Convex hull."""

    NONE = 11
    """No geometry (placeholder)."""


# Default maximum vertices for convex hull approximation
MESH_MAXHULLVERT = 64


class SDF:
    """
    Represents a signed distance field (SDF) for simulation.

    An SDF is a volumetric representation of a shape, where each point in the volume
    stores the signed distance to the closest surface. This class encapsulates the
    SDF volume and its physical properties for use in simulation.
    """

    def __init__(self, volume: wp.Volume | None = None, I=None, mass=1.0, com=None):
        """
        Initialize an SDF object.

        Args:
            volume (wp.Volume | None): The Warp volume object representing the SDF.
            I (Mat33, optional): 3x3 inertia matrix. Defaults to identity.
            mass (float, optional): Total mass. Defaults to 1.0.
            com (Vec3, optional): Center of mass. Defaults to zero vector.
        """
        self.volume = volume
        self.I = I if I is not None else wp.mat33(np.eye(3))
        self.mass = mass
        self.com = com if com is not None else wp.vec3()

        # Need to specify these for now
        self.has_inertia = True
        self.is_solid = True

    def finalize(self) -> wp.uint64:
        """
        Returns the ID of the underlying SDF volume.

        Returns:
            wp.uint64: The unique identifier of the SDF volume.
        """
        return self.volume.id

    @override
    def __hash__(self) -> int:
        return hash(self.volume.id)


class Mesh:
    """
    Represents a triangle mesh for collision and simulation.

    This class encapsulates a triangle mesh, including its geometry, physical properties,
    and utility methods for simulation. Meshes are typically used for collision detection,
    visualization, and inertia computation in physics simulation.

    Example:
        Load a mesh from an OBJ file using OpenMesh and create a Newton Mesh:

        .. code-block:: python

            import numpy as np
            import newton
            import openmesh

            m = openmesh.read_trimesh("mesh.obj")
            mesh_points = np.array(m.points())
            mesh_indices = np.array(m.face_vertex_indices(), dtype=np.int32).flatten()
            mesh = newton.Mesh(mesh_points, mesh_indices)
    """

    def __init__(
        self,
        vertices: Sequence[Vec3] | nparray,
        indices: Sequence[int] | nparray,
        normals: Sequence[Vec3] | nparray | None = None,
        uvs: Sequence[Vec2] | nparray | None = None,
        compute_inertia: bool = True,
        is_solid: bool = True,
        maxhullvert: int = MESH_MAXHULLVERT,
        color: Vec3 | None = None,
    ):
        """
        Construct a Mesh object from a triangle mesh.

        The mesh's center of mass and inertia tensor are automatically calculated
        using a density of 1.0 if ``compute_inertia`` is True. This computation is only valid
        if the mesh is closed (two-manifold).

        Args:
            vertices (Sequence[Vec3] | nparray): List or array of mesh vertices, shape (N, 3).
            indices (Sequence[int] | nparray): Flattened list or array of triangle indices (3 per triangle).
            normals (Sequence[Vec3] | nparray | None, optional): Optional per-vertex normals, shape (N, 3).
            uvs (Sequence[Vec2] | nparray | None, optional): Optional per-vertex UVs, shape (N, 2).
            compute_inertia (bool, optional): If True, compute mass, inertia tensor, and center of mass (default: True).
            is_solid (bool, optional): If True, mesh is assumed solid for inertia computation (default: True).
            maxhullvert (int, optional): Max vertices for convex hull approximation (default: 64).
            color (Vec3 | None, optional): Optional per-mesh base color (values in [0, 1]).
        """
        from .inertia import compute_mesh_inertia  # noqa: PLC0415

        self._vertices = np.array(vertices, dtype=np.float32).reshape(-1, 3)
        self._indices = np.array(indices, dtype=np.int32).flatten()
        self._normals = np.array(normals, dtype=np.float32).reshape(-1, 3) if normals is not None else None
        self._uvs = np.array(uvs, dtype=np.float32).reshape(-1, 2) if uvs is not None else None
        self._color = color
        self.is_solid = is_solid
        self.has_inertia = compute_inertia
        self.mesh = None
        self.maxhullvert = maxhullvert
        self._cached_hash = None

        if compute_inertia:
            self.mass, self.com, self.I, _ = compute_mesh_inertia(1.0, vertices, indices, is_solid=is_solid)
        else:
            self.I = wp.mat33(np.eye(3))
            self.mass = 1.0
            self.com = wp.vec3()

    def copy(
        self,
        vertices: Sequence[Vec3] | nparray | None = None,
        indices: Sequence[int] | nparray | None = None,
        recompute_inertia: bool = False,
    ):
        """
        Create a copy of this mesh, optionally with new vertices or indices.

        Args:
            vertices (Sequence[Vec3] | nparray | None, optional): New vertices to use (default: current vertices).
            indices (Sequence[int] | nparray | None, optional): New indices to use (default: current indices).
            recompute_inertia (bool, optional): If True, recompute inertia properties (default: False).

        Returns:
            Mesh: A new Mesh object with the specified properties.
        """
        if vertices is None:
            vertices = self.vertices.copy()
        if indices is None:
            indices = self.indices.copy()
        m = Mesh(
            vertices,
            indices,
            compute_inertia=recompute_inertia,
            is_solid=self.is_solid,
            maxhullvert=self.maxhullvert,
            normals=self.normals.copy() if self.normals is not None else None,
            uvs=self.uvs.copy() if self.uvs is not None else None,
        )
        if not recompute_inertia:
            m.I = self.I
            m.mass = self.mass
            m.com = self.com
            m.has_inertia = self.has_inertia
        return m

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        self._vertices = np.array(value, dtype=np.float32).reshape(-1, 3)
        self._cached_hash = None

    @property
    def indices(self):
        return self._indices

    @indices.setter
    def indices(self, value):
        self._indices = np.array(value, dtype=np.int32).flatten()
        self._cached_hash = None

    @property
    def normals(self):
        return self._normals

    @property
    def uvs(self):
        return self._uvs

    # construct simulation ready buffers from points
    def finalize(self, device: Devicelike = None, requires_grad: bool = False) -> wp.uint64:
        """
        Construct a simulation-ready Warp Mesh object from the mesh data and return its ID.

        Args:
            device (Devicelike, optional): Device on which to allocate mesh buffers.
            requires_grad (bool, optional): If True, mesh points and velocities are allocated with gradient tracking.

        Returns:
            wp.uint64: The ID of the simulation-ready Warp Mesh.
        """
        with wp.ScopedDevice(device):
            pos = wp.array(self.vertices, requires_grad=requires_grad, dtype=wp.vec3)
            vel = wp.zeros_like(pos)
            indices = wp.array(self.indices, dtype=wp.int32)

            self.mesh = wp.Mesh(points=pos, velocities=vel, indices=indices, bvh_constructor="sah")
            return self.mesh.id

    def compute_convex_hull(self, replace: bool = False) -> "Mesh":
        """
        Compute and return the convex hull of this mesh.

        Args:
            replace (bool, optional): If True, replace this mesh's vertices/indices with the convex hull (in-place).
                                      If False, return a new Mesh for the convex hull.

        Returns:
            Mesh: The convex hull mesh (either new or self, depending on `replace`).
        """
        from .utils import remesh_convex_hull  # noqa: PLC0415

        hull_vertices, hull_faces = remesh_convex_hull(self.vertices, maxhullvert=self.maxhullvert)
        if replace:
            self.vertices = hull_vertices
            self.indices = hull_faces
            return self
        else:
            # create a new mesh for the convex hull
            hull_mesh = Mesh(hull_vertices, hull_faces, compute_inertia=False)
            hull_mesh.maxhullvert = self.maxhullvert  # preserve maxhullvert setting
            hull_mesh.is_solid = self.is_solid
            hull_mesh.has_inertia = self.has_inertia
            hull_mesh.mass = self.mass
            hull_mesh.com = self.com
            hull_mesh.I = self.I
            return hull_mesh

    @override
    def __hash__(self) -> int:
        """
        Compute a hash of the mesh data for use in caching.

        The hash considers the mesh vertices, indices, and whether the mesh is solid.
        Uses a cached hash if available, otherwise computes and caches the hash.

        Returns:
            int: The hash value for the mesh.
        """
        if self._cached_hash is None:
            self._cached_hash = hash(
                (tuple(np.array(self.vertices).flatten()), tuple(np.array(self.indices).flatten()), self.is_solid)
            )
        return self._cached_hash
