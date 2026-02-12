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
import os
from collections.abc import Sequence

import numpy as np
import warp as wp

from ..core.types import Devicelike, Mat33, Vec2, Vec3, nparray, override
from ..utils.texture import compute_texture_hash


def _normalize_texture_input(texture: str | os.PathLike[str] | nparray | None) -> str | nparray | None:
    """Normalize texture input for lazy storage.

    String paths and PathLike objects are stored as strings (no decoding).
    Arrays are normalized to contiguous arrays.
    Decoding of paths is deferred until the viewer requests the image data.
    """
    if texture is None:
        return None
    if isinstance(texture, os.PathLike):
        return os.fspath(texture)
    if isinstance(texture, str):
        return texture
    # Array input: make it contiguous
    return np.ascontiguousarray(np.asarray(texture))


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


class SDF:
    """
    Represents a signed distance field (SDF) for simulation.

    An SDF is a volumetric representation of a shape, where each point in the volume
    stores the signed distance to the closest surface. This class encapsulates the
    SDF volume and its physical properties for use in simulation.
    """

    def __init__(
        self,
        volume: wp.Volume | None = None,
        inertia: Mat33 | None = None,
        mass: float = 1.0,
        com: Vec3 | None = None,
    ):
        """
        Initialize an SDF object.

        Args:
            volume: The Warp volume object representing the SDF.
            inertia: 3x3 inertia matrix. Defaults to identity.
            mass: Total mass. Defaults to 1.0.
            com: Center of mass. Defaults to zero vector.
        """
        self.volume = volume
        self.I = inertia if inertia is not None else wp.mat33(np.eye(3))
        self.mass = mass
        self.com = com if com is not None else wp.vec3()

        # Need to specify these for now
        self.has_inertia = True
        self.is_solid = True

    def finalize(self) -> wp.uint64:
        """
        Returns the ID of the underlying SDF volume.

        Returns:
            The unique identifier of the SDF volume.
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

    MAX_HULL_VERTICES = 64
    _color: Vec3 | None = None

    def __init__(
        self,
        vertices: Sequence[Vec3] | nparray,
        indices: Sequence[int] | nparray,
        normals: Sequence[Vec3] | nparray | None = None,
        uvs: Sequence[Vec2] | nparray | None = None,
        compute_inertia: bool = True,
        is_solid: bool = True,
        maxhullvert: int | None = None,
        color: Vec3 | None = None,
        roughness: float | None = None,
        metallic: float | None = None,
        texture: str | nparray | None = None,
    ):
        """
        Construct a Mesh object from a triangle mesh.

        The mesh's center of mass and inertia tensor are automatically calculated
        using a density of 1.0 if ``compute_inertia`` is True. This computation is only valid
        if the mesh is closed (two-manifold).

        Args:
            vertices: List or array of mesh vertices, shape (N, 3).
            indices: Flattened list or array of triangle indices (3 per triangle).
            normals: Optional per-vertex normals, shape (N, 3).
            uvs: Optional per-vertex UVs, shape (N, 2).
            compute_inertia: If True, compute mass, inertia tensor, and center of mass (default: True).
            is_solid: If True, mesh is assumed solid for inertia computation (default: True).
            maxhullvert: Max vertices for convex hull approximation (default: :attr:`~newton.Mesh.MAX_HULL_VERTICES`).
            color: Optional per-mesh base color (values in [0, 1]).
            roughness: Optional mesh roughness in [0, 1].
            metallic: Optional mesh metallic in [0, 1].
            texture: Optional texture path/URL or image data (H, W, C).
        """
        from .inertia import compute_mesh_inertia  # noqa: PLC0415

        self._vertices = np.array(vertices, dtype=np.float32).reshape(-1, 3)
        self._indices = np.array(indices, dtype=np.int32).flatten()
        self._normals = np.array(normals, dtype=np.float32).reshape(-1, 3) if normals is not None else None
        self._uvs = np.array(uvs, dtype=np.float32).reshape(-1, 2) if uvs is not None else None
        self.color = color
        # Store texture lazily: strings/paths are kept as-is, arrays are normalized
        self._texture = _normalize_texture_input(texture)
        self._roughness = roughness
        self._metallic = metallic
        self.is_solid = is_solid
        self.has_inertia = compute_inertia
        self.mesh = None
        if maxhullvert is None:
            maxhullvert = Mesh.MAX_HULL_VERTICES
        self.maxhullvert = maxhullvert
        self._cached_hash = None
        self._texture_hash = None

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
            vertices: New vertices to use (default: current vertices).
            indices: New indices to use (default: current indices).
            recompute_inertia: If True, recompute inertia properties (default: False).

        Returns:
            A new Mesh object with the specified properties.
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
            color=self.color,
            texture=self._texture
            if isinstance(self._texture, str)
            else (self._texture.copy() if self._texture is not None else None),
            roughness=self._roughness,
            metallic=self._metallic,
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

    @property
    def color(self) -> Vec3 | None:
        return self._color

    @color.setter
    def color(self, value: Vec3 | None):
        self._color = value

    @property
    def texture(self) -> str | nparray | None:
        return self._texture

    @texture.setter
    def texture(self, value: str | nparray | None):
        # Store texture lazily: strings/paths are kept as-is, arrays are normalized
        self._texture = _normalize_texture_input(value)
        self._texture_hash = None
        self._cached_hash = None

    def _compute_texture_hash(self) -> int:
        if self._texture_hash is None:
            self._texture_hash = compute_texture_hash(self._texture)
        return self._texture_hash

    @property
    def roughness(self) -> float | None:
        return self._roughness

    @roughness.setter
    def roughness(self, value: float | None):
        self._roughness = value
        self._cached_hash = None

    @property
    def metallic(self) -> float | None:
        return self._metallic

    @metallic.setter
    def metallic(self, value: float | None):
        self._metallic = value
        self._cached_hash = None

    # construct simulation ready buffers from points
    def finalize(self, device: Devicelike = None, requires_grad: bool = False) -> wp.uint64:
        """
        Construct a simulation-ready Warp Mesh object from the mesh data and return its ID.

        Args:
            device: Device on which to allocate mesh buffers.
            requires_grad: If True, mesh points and velocities are allocated with gradient tracking.

        Returns:
            The ID of the simulation-ready Warp Mesh.
        """
        with wp.ScopedDevice(device):
            pos = wp.array(self.vertices, requires_grad=requires_grad, dtype=wp.vec3)
            vel = wp.zeros_like(pos)
            indices = wp.array(self.indices, dtype=wp.int32)

            self.mesh = wp.Mesh(points=pos, velocities=vel, indices=indices)
            return self.mesh.id

    def compute_convex_hull(self, replace: bool = False) -> "Mesh":
        """
        Compute and return the convex hull of this mesh.

        Args:
            replace: If True, replace this mesh's vertices/indices with the convex hull (in-place).
                If False, return a new Mesh for the convex hull.

        Returns:
            The convex hull mesh (either new or self, depending on `replace`).
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
            The hash value for the mesh.
        """
        if self._cached_hash is None:
            self._cached_hash = hash(
                (
                    tuple(np.array(self.vertices).flatten()),
                    tuple(np.array(self.indices).flatten()),
                    self.is_solid,
                    self._compute_texture_hash(),
                    self._roughness,
                    self._metallic,
                )
            )
        return self._cached_hash


class Heightfield:
    """
    Represents a heightfield (2D elevation grid) for terrain and large static surfaces.

    Heightfields are efficient representations of terrain using a 2D grid of elevation values.
    They are always static (zero mass, zero inertia) and more memory-efficient than equivalent
    triangle meshes.

    The elevation data is always normalized to [0, 1] internally. World-space heights are
    computed as: ``z = min_z + data[r, c] * (max_z - min_z)``.

    Example:
        Create a heightfield from raw elevation data (auto-normalizes):

        .. code-block:: python

            import numpy as np
            import newton

            nrow, ncol = 10, 10
            elevation = np.random.rand(nrow, ncol).astype(np.float32) * 5.0  # 0-5 meters

            hfield = newton.Heightfield(
                data=elevation,
                nrow=nrow,
                ncol=ncol,
                hx=5.0,  # half-extent X (field spans [-5, +5] meters)
                hy=5.0,  # half-extent Y
            )
            # min_z and max_z are auto-derived from the data (0.0 and 5.0)

        Create with explicit height range:

        .. code-block:: python

            hfield = newton.Heightfield(
                data=normalized_data,  # any values, will be normalized
                nrow=nrow,
                ncol=ncol,
                hx=5.0,
                hy=5.0,
                min_z=-1.0,
                max_z=3.0,
            )
    """

    def __init__(
        self,
        data: Sequence[Sequence[float]] | nparray,
        nrow: int,
        ncol: int,
        hx: float = 1.0,
        hy: float = 1.0,
        min_z: float | None = None,
        max_z: float | None = None,
    ):
        """
        Construct a Heightfield object from a 2D elevation grid.

        The input data is normalized to [0, 1]. If ``min_z`` and ``max_z`` are not provided,
        they are derived from the data's minimum and maximum values.

        Args:
            data: 2D array of elevation values, shape (nrow, ncol). Any numeric values are
                accepted and will be normalized to [0, 1] internally.
            nrow: Number of rows in the heightfield grid.
            ncol: Number of columns in the heightfield grid.
            hx: Half-extent in X direction (meters). The heightfield spans [-hx, +hx].
            hy: Half-extent in Y direction (meters). The heightfield spans [-hy, +hy].
            min_z: World-space Z value corresponding to data minimum. Must be provided
                together with ``max_z``, or both omitted to auto-derive from data.
            max_z: World-space Z value corresponding to data maximum. Must be provided
                together with ``min_z``, or both omitted to auto-derive from data.
        """
        if (min_z is None) != (max_z is None):
            raise ValueError("min_z and max_z must both be provided or both omitted")

        raw = np.array(data, dtype=np.float32).reshape(nrow, ncol)
        d_min, d_max = float(raw.min()), float(raw.max())

        # Normalize data to [0, 1]
        if d_max > d_min:
            self._data = (raw - d_min) / (d_max - d_min)
        else:
            self._data = np.zeros_like(raw)

        self.nrow = nrow
        self.ncol = ncol
        self.hx = hx
        self.hy = hy
        self.min_z = d_min if min_z is None else float(min_z)
        self.max_z = d_max if max_z is None else float(max_z)

        self.is_solid = True
        self.has_inertia = False
        self.warp_array = None  # Will be set by finalize()
        self._cached_hash = None

        # Heightfields are always static
        self.I = wp.mat33()
        self.mass = 0.0
        self.com = wp.vec3()

    @property
    def data(self):
        """Get the normalized [0, 1] elevation data as a 2D numpy array."""
        return self._data

    @data.setter
    def data(self, value):
        """Set the elevation data from a 2D array. Data is normalized to [0, 1]."""
        raw = np.array(value, dtype=np.float32).reshape(self.nrow, self.ncol)
        d_min, d_max = float(raw.min()), float(raw.max())
        if d_max > d_min:
            self._data = (raw - d_min) / (d_max - d_min)
        else:
            self._data = np.zeros_like(raw)
        self.min_z = d_min
        self.max_z = d_max
        self._cached_hash = None

    def finalize(self, device: Devicelike = None, requires_grad: bool = False) -> wp.uint64:
        """
        Construct a simulation-ready Warp array from the heightfield data and return its ID.

        Args:
            device: Device on which to allocate heightfield buffers.
            requires_grad: If True, data is allocated with gradient tracking.

        Returns:
            The ID (pointer) of the simulation-ready Warp array.
        """
        with wp.ScopedDevice(device):
            self.warp_array = wp.array(self._data.flatten(), requires_grad=requires_grad, dtype=wp.float32)
            return self.warp_array.ptr

    @override
    def __hash__(self) -> int:
        """
        Compute a hash of the heightfield data for use in caching.

        Returns:
            The hash value for the heightfield.
        """
        if self._cached_hash is None:
            self._cached_hash = hash(
                (
                    tuple(self._data.flatten()),
                    self.nrow,
                    self.ncol,
                    self.hx,
                    self.hy,
                    self.min_z,
                    self.max_z,
                )
            )
        return self._cached_hash
