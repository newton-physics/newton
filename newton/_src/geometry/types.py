# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import enum
import os
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..core.types import Axis, Devicelike, Vec2, Vec3, override
from ..utils.texture import compute_texture_hash

if TYPE_CHECKING:
    from ..sim.model import Model
    from .sdf_utils import SDF


def _normalize_texture_input(texture: str | os.PathLike[str] | np.ndarray | None) -> str | np.ndarray | None:
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

    NONE = 0
    """No geometry (placeholder)."""

    PLANE = 1
    """Plane."""

    HFIELD = 2
    """Height field (terrain)."""

    SPHERE = 3
    """Sphere."""

    CAPSULE = 4
    """Capsule (cylinder with hemispherical ends)."""

    ELLIPSOID = 5
    """Ellipsoid."""

    CYLINDER = 6
    """Cylinder."""

    BOX = 7
    """Axis-aligned box."""

    MESH = 8
    """Triangle mesh."""

    CONE = 9
    """Cone."""

    CONVEX_MESH = 10
    """Convex hull."""

    GAUSSIAN = 11
    """Gaussian splat."""


class Mesh:
    """
    Represents a triangle mesh for collision and simulation.

    This class encapsulates a triangle mesh, including its geometry, physical properties,
    and utility methods for simulation. Meshes are typically used for collision detection,
    visualization, and inertia computation in physics simulation.

    Attributes:
        mass [kg]: Mesh mass in local coordinates, computed with density 1.0 when
            ``compute_inertia`` is ``True``.
        com [m]: Mesh center of mass in local coordinates.
        inertia [kg*m^2]: Mesh inertia tensor about :attr:`com` in local coordinates.

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
    """Default maximum vertex count for convex hull approximation."""

    def __init__(
        self,
        vertices: Sequence[Vec3] | np.ndarray,
        indices: Sequence[int] | np.ndarray,
        normals: Sequence[Vec3] | np.ndarray | None = None,
        uvs: Sequence[Vec2] | np.ndarray | None = None,
        compute_inertia: bool = True,
        is_solid: bool = True,
        maxhullvert: int | None = None,
        color: Vec3 | None = None,
        roughness: float | None = None,
        metallic: float | None = None,
        texture: str | np.ndarray | None = None,
        *,
        sdf: "SDF | None" = None,
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
            sdf: Optional prebuilt SDF object owned by this mesh.
        """
        from .inertia import compute_inertia_mesh  # noqa: PLC0415

        self._vertices = np.array(vertices, dtype=np.float32).reshape(-1, 3)
        self._indices = np.array(indices, dtype=np.int32).flatten()
        self._normals = np.array(normals, dtype=np.float32).reshape(-1, 3) if normals is not None else None
        self._uvs = np.array(uvs, dtype=np.float32).reshape(-1, 2) if uvs is not None else None
        self._color: Vec3 | None = None
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
        self.sdf = sdf

        if compute_inertia:
            self.mass, self.com, self.inertia, _ = compute_inertia_mesh(1.0, vertices, indices, is_solid=is_solid)
        else:
            self.inertia = wp.mat33(np.eye(3))
            self.mass = 1.0
            self.com = wp.vec3()

    @staticmethod
    def create_sphere(
        radius: float = 1.0,
        *,
        num_latitudes: int = 32,
        num_longitudes: int = 32,
        reverse_winding: bool = False,
        compute_normals: bool = True,
        compute_uvs: bool = True,
        compute_inertia: bool = True,
    ) -> "Mesh":
        """Create a UV sphere mesh.

        Args:
            radius [m]: Sphere radius.
            num_latitudes: Number of latitude subdivisions.
            num_longitudes: Number of longitude subdivisions.
            reverse_winding: If ``True``, reverse triangle winding order.
            compute_normals: If ``True``, generate per-vertex normals.
            compute_uvs: If ``True``, generate per-vertex UV coordinates.
            compute_inertia: If ``True``, compute mesh mass properties.

        Returns:
            A sphere mesh.
        """
        from ..utils.mesh import create_mesh_sphere  # noqa: PLC0415

        positions, indices, normals, uvs = create_mesh_sphere(
            radius,
            num_latitudes=num_latitudes,
            num_longitudes=num_longitudes,
            reverse_winding=reverse_winding,
            compute_normals=compute_normals,
            compute_uvs=compute_uvs,
        )
        return Mesh(
            vertices=positions,
            indices=indices,
            normals=normals,
            uvs=uvs,
            compute_inertia=compute_inertia,
        )

    @staticmethod
    def create_ellipsoid(
        rx: float = 1.0,
        ry: float = 1.0,
        rz: float = 1.0,
        *,
        num_latitudes: int = 32,
        num_longitudes: int = 32,
        reverse_winding: bool = False,
        compute_normals: bool = True,
        compute_uvs: bool = True,
        compute_inertia: bool = True,
    ) -> "Mesh":
        """Create a UV ellipsoid mesh.

        Args:
            rx [m]: Semi-axis length along X.
            ry [m]: Semi-axis length along Y.
            rz [m]: Semi-axis length along Z.
            num_latitudes: Number of latitude subdivisions.
            num_longitudes: Number of longitude subdivisions.
            reverse_winding: If ``True``, reverse triangle winding order.
            compute_normals: If ``True``, generate per-vertex normals.
            compute_uvs: If ``True``, generate per-vertex UV coordinates.
            compute_inertia: If ``True``, compute mesh mass properties.

        Returns:
            An ellipsoid mesh.
        """
        from ..utils.mesh import create_mesh_ellipsoid  # noqa: PLC0415

        positions, indices, normals, uvs = create_mesh_ellipsoid(
            rx,
            ry,
            rz,
            num_latitudes=num_latitudes,
            num_longitudes=num_longitudes,
            reverse_winding=reverse_winding,
            compute_normals=compute_normals,
            compute_uvs=compute_uvs,
        )
        return Mesh(
            vertices=positions,
            indices=indices,
            normals=normals,
            uvs=uvs,
            compute_inertia=compute_inertia,
        )

    @staticmethod
    def create_capsule(
        radius: float,
        half_height: float,
        *,
        up_axis: Axis = Axis.Y,
        segments: int = 32,
        compute_normals: bool = True,
        compute_uvs: bool = True,
        compute_inertia: bool = True,
    ) -> "Mesh":
        """Create a capsule mesh.

        Args:
            radius [m]: Radius of the capsule hemispheres and cylindrical body.
            half_height [m]: Half-height of the cylindrical section.
            up_axis: Long axis as a ``newton.Axis`` value.
            segments: Tessellation resolution for both caps and body.
            compute_normals: If ``True``, generate per-vertex normals.
            compute_uvs: If ``True``, generate per-vertex UV coordinates.
            compute_inertia: If ``True``, compute mesh mass properties.

        Returns:
            A capsule mesh.
        """
        from ..utils.mesh import create_mesh_capsule  # noqa: PLC0415

        positions, indices, normals, uvs = create_mesh_capsule(
            radius,
            half_height,
            up_axis=int(up_axis),
            segments=segments,
            compute_normals=compute_normals,
            compute_uvs=compute_uvs,
        )
        return Mesh(
            vertices=positions,
            indices=indices,
            normals=normals,
            uvs=uvs,
            compute_inertia=compute_inertia,
        )

    @staticmethod
    def create_cylinder(
        radius: float,
        half_height: float,
        *,
        up_axis: Axis = Axis.Y,
        segments: int = 32,
        top_radius: float | None = None,
        compute_normals: bool = True,
        compute_uvs: bool = True,
        compute_inertia: bool = True,
    ) -> "Mesh":
        """Create a cylinder or truncated cone mesh.

        Args:
            radius [m]: Bottom radius.
            half_height [m]: Half-height along the cylinder axis.
            up_axis: Long axis as a ``newton.Axis`` value.
            segments: Circumferential tessellation resolution.
            top_radius [m]: Optional top radius. If ``None``, equals ``radius``.
            compute_normals: If ``True``, generate per-vertex normals.
            compute_uvs: If ``True``, generate per-vertex UV coordinates.
            compute_inertia: If ``True``, compute mesh mass properties.

        Returns:
            A cylinder or truncated-cone mesh.
        """
        from ..utils.mesh import create_mesh_cylinder  # noqa: PLC0415

        positions, indices, normals, uvs = create_mesh_cylinder(
            radius,
            half_height,
            up_axis=int(up_axis),
            segments=segments,
            top_radius=top_radius,
            compute_normals=compute_normals,
            compute_uvs=compute_uvs,
        )
        return Mesh(
            vertices=positions,
            indices=indices,
            normals=normals,
            uvs=uvs,
            compute_inertia=compute_inertia,
        )

    @staticmethod
    def create_cone(
        radius: float,
        half_height: float,
        *,
        up_axis: Axis = Axis.Y,
        segments: int = 32,
        compute_normals: bool = True,
        compute_uvs: bool = True,
        compute_inertia: bool = True,
    ) -> "Mesh":
        """Create a cone mesh.

        Args:
            radius [m]: Base radius.
            half_height [m]: Half-height from center to apex/base.
            up_axis: Long axis as a ``newton.Axis`` value.
            segments: Circumferential tessellation resolution.
            compute_normals: If ``True``, generate per-vertex normals.
            compute_uvs: If ``True``, generate per-vertex UV coordinates.
            compute_inertia: If ``True``, compute mesh mass properties.

        Returns:
            A cone mesh.
        """
        from ..utils.mesh import create_mesh_cone  # noqa: PLC0415

        positions, indices, normals, uvs = create_mesh_cone(
            radius,
            half_height,
            up_axis=int(up_axis),
            segments=segments,
            compute_normals=compute_normals,
            compute_uvs=compute_uvs,
        )
        return Mesh(
            vertices=positions,
            indices=indices,
            normals=normals,
            uvs=uvs,
            compute_inertia=compute_inertia,
        )

    @staticmethod
    def create_arrow(
        base_radius: float,
        base_height: float,
        *,
        cap_radius: float | None = None,
        cap_height: float | None = None,
        up_axis: Axis = Axis.Y,
        segments: int = 32,
        compute_normals: bool = True,
        compute_uvs: bool = True,
        compute_inertia: bool = True,
    ) -> "Mesh":
        """Create an arrow mesh (cylinder shaft + cone head).

        Args:
            base_radius [m]: Shaft radius.
            base_height [m]: Shaft full height (not half-height).
            cap_radius [m]: Optional arrowhead base radius. If ``None``, uses
                ``base_radius * 1.8``.
            cap_height [m]: Optional arrowhead full height (not half-height).
                If ``None``, uses ``base_height * 0.18``.
            up_axis: Long axis as a ``newton.Axis`` value.
            segments: Circumferential tessellation resolution.
            compute_normals: If ``True``, generate per-vertex normals.
            compute_uvs: If ``True``, generate per-vertex UV coordinates.
            compute_inertia: If ``True``, compute mesh mass properties.

        Returns:
            An arrow mesh.
        """
        from ..utils.mesh import create_mesh_arrow  # noqa: PLC0415

        positions, indices, normals, uvs = create_mesh_arrow(
            base_radius,
            base_height,
            cap_radius=cap_radius,
            cap_height=cap_height,
            up_axis=int(up_axis),
            segments=segments,
            compute_normals=compute_normals,
            compute_uvs=compute_uvs,
        )
        return Mesh(
            vertices=positions,
            indices=indices,
            normals=normals,
            uvs=uvs,
            compute_inertia=compute_inertia,
        )

    @staticmethod
    def create_box(
        hx: float,
        hy: float | None = None,
        hz: float | None = None,
        *,
        duplicate_vertices: bool = True,
        compute_normals: bool = True,
        compute_uvs: bool = True,
        compute_inertia: bool = True,
    ) -> "Mesh":
        """Create a box mesh from half-extents.

        Args:
            hx [m]: Half-extent along X.
            hy [m]: Half-extent along Y. If ``None``, uses ``hx``.
            hz [m]: Half-extent along Z. If ``None``, uses ``hx``.
            duplicate_vertices: If ``True``, duplicate vertices per face.
            compute_normals: If ``True``, generate per-vertex normals.
            compute_uvs: If ``True``, generate per-vertex UV coordinates.
            compute_inertia: If ``True``, compute mesh mass properties.

        Returns:
            A box mesh.
        """
        from ..utils.mesh import create_mesh_box  # noqa: PLC0415

        if hy is None:
            hy = hx
        if hz is None:
            hz = hx

        positions, indices, normals, uvs = create_mesh_box(
            float(hx),
            float(hy),
            float(hz),
            duplicate_vertices=duplicate_vertices,
            compute_normals=compute_normals,
            compute_uvs=compute_uvs,
        )
        return Mesh(
            vertices=positions,
            indices=indices,
            normals=normals,
            uvs=uvs,
            compute_inertia=compute_inertia,
        )

    @staticmethod
    def create_plane(
        width: float,
        length: float,
        *,
        compute_normals: bool = True,
        compute_uvs: bool = True,
        compute_inertia: bool = True,
    ) -> "Mesh":
        """Create a rectangular plane mesh.

        The plane lies in the XY plane and faces +Z (normals point along +Z).

        Args:
            width [m]: Plane width along X.
            length [m]: Plane length along Y.
            compute_normals: If ``True``, generate per-vertex normals.
            compute_uvs: If ``True``, generate per-vertex UV coordinates.
            compute_inertia: If ``True``, compute mesh mass properties.

        Returns:
            A plane mesh.
        """
        from ..utils.mesh import create_mesh_plane  # noqa: PLC0415

        positions, indices, normals, uvs = create_mesh_plane(
            width,
            length,
            compute_normals=compute_normals,
            compute_uvs=compute_uvs,
        )
        return Mesh(
            vertices=positions,
            indices=indices,
            normals=normals,
            uvs=uvs,
            compute_inertia=compute_inertia,
        )

    @staticmethod
    def create_terrain(
        grid_size: tuple[int, int] = (4, 4),
        block_size: tuple[float, float] = (5.0, 5.0),
        terrain_types: list[str] | str | object | None = None,
        terrain_params: dict | None = None,
        seed: int | None = None,
        *,
        compute_inertia: bool = True,
    ) -> "Mesh":
        """Create a procedural terrain mesh from terrain blocks.

        Args:
            grid_size: Terrain grid size as ``(rows, cols)``.
            block_size [m]: Terrain block dimensions as ``(width, length)``.
            terrain_types: Terrain type name(s) or callable generator(s).
            terrain_params: Optional per-terrain parameter dictionary.
            seed: Optional random seed for deterministic terrain generation.
            compute_inertia: If ``True``, compute mesh mass properties.

        Returns:
            A terrain mesh.
        """
        from .terrain_generator import create_mesh_terrain  # noqa: PLC0415

        vertices, indices = create_mesh_terrain(
            grid_size=grid_size,
            block_size=block_size,
            terrain_types=terrain_types,
            terrain_params=terrain_params,
            seed=seed,
        )
        return Mesh(vertices, indices, compute_inertia=compute_inertia)

    @staticmethod
    def create_heightfield(
        heightfield: np.ndarray,
        extent_x: float,
        extent_y: float,
        center_x: float = 0.0,
        center_y: float = 0.0,
        ground_z: float = 0.0,
        *,
        compute_inertia: bool = True,
    ) -> "Mesh":
        """Create a watertight mesh from a 2D heightfield.

        Args:
            heightfield: Height samples as a 2D array using ij-indexing where
                ``heightfield[i, j]`` maps to ``(x_i, y_j)`` (i = X, j = Y).
            extent_x [m]: Total extent along X.
            extent_y [m]: Total extent along Y.
            center_x [m]: Heightfield center position along X.
            center_y [m]: Heightfield center position along Y.
            ground_z [m]: Bottom surface Z value for watertight side walls.
            compute_inertia: If ``True``, compute mesh mass properties.

        Returns:
            A heightfield mesh.
        """
        from .terrain_generator import create_mesh_heightfield  # noqa: PLC0415

        vertices, indices = create_mesh_heightfield(
            heightfield=heightfield,
            extent_x=extent_x,
            extent_y=extent_y,
            center_x=center_x,
            center_y=center_y,
            ground_z=ground_z,
        )
        return Mesh(vertices, indices, compute_inertia=compute_inertia)

    def copy(
        self,
        vertices: Sequence[Vec3] | np.ndarray | None = None,
        indices: Sequence[int] | np.ndarray | None = None,
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
            m.inertia = self.inertia
            m.mass = self.mass
            m.com = self.com
            m.has_inertia = self.has_inertia
        m.sdf = self.sdf
        return m

    def build_sdf(
        self,
        *,
        device: Devicelike | None = None,
        narrow_band_range: tuple[float, float] | None = None,
        target_voxel_size: float | None = None,
        max_resolution: int | None = None,
        margin: float | None = None,
        shape_margin: float = 0.0,
        scale: tuple[float, float, float] | None = None,
        texture_format: str = "uint16",
    ) -> "SDF":
        """Build and attach an SDF for this mesh.

        Args:
            device: CUDA device for SDF allocation. When ``None``, uses the
                current :class:`wp.ScopedDevice` or the Warp default device.
            narrow_band_range: Signed narrow-band distance range [m] as
                ``(inner, outer)``. Uses ``(-0.1, 0.1)`` when not provided.
            target_voxel_size: Target sparse-grid voxel size [m]. If provided,
                takes precedence over ``max_resolution``.
            max_resolution: Maximum sparse-grid dimension [voxel] along the longest
                AABB axis, used when ``target_voxel_size`` is not provided. Must be
                divisible by 8.
            margin: Extra AABB padding [m] added before discretization. Uses
                ``0.05`` when not provided.
            shape_margin: Shape margin offset [m] to subtract from SDF values.
                When non-zero, the SDF surface is effectively shrunk inward by
                this amount. Useful for modeling compliant layers in hydroelastic
                collision. Defaults to ``0.0``.
            scale: Scale factors ``(sx, sy, sz)`` to bake into the SDF. When
                provided, the mesh vertices are scaled before SDF generation
                and ``scale_baked`` is set to ``True`` in the resulting SDF.
                Required for hydroelastic collision with non-unit shape scale.
                Defaults to ``None`` (no scale baking, scale applied at runtime).
            texture_format: Subgrid texture storage format for the SDF.
                ``"uint16"`` (default) stores subgrid voxels in 16-bit
                normalized textures (half the memory of float32).
                ``"float32"`` stores full-precision values. ``"uint8"`` uses
                8-bit textures for minimum memory at lower precision.

        Returns:
            The attached :class:`SDF` instance.

        Raises:
            RuntimeError: If this mesh already has an SDF attached.
        """
        if self.sdf is not None:
            raise RuntimeError("Mesh already has an SDF. Call clear_sdf() before rebuilding.")

        _valid_tex_fmts = ("float32", "uint16", "uint8")
        if texture_format not in _valid_tex_fmts:
            raise ValueError(f"Unknown texture_format {texture_format!r}. Expected one of {list(_valid_tex_fmts)}.")

        from .sdf_utils import SDF  # noqa: PLC0415

        self.sdf = SDF.create_from_mesh(
            self,
            device=device,
            narrow_band_range=narrow_band_range if narrow_band_range is not None else (-0.1, 0.1),
            target_voxel_size=target_voxel_size,
            max_resolution=max_resolution,
            margin=margin if margin is not None else 0.05,
            shape_margin=shape_margin,
            scale=scale,
            texture_format=texture_format,
        )
        return self.sdf

    def clear_sdf(self) -> None:
        """Detach and release the currently attached SDF.

        Returns:
            ``None``.
        """
        self.sdf = None

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
    def texture(self) -> str | np.ndarray | None:
        return self._texture

    @texture.setter
    def texture(self, value: str | np.ndarray | None):
        # Store texture lazily: strings/paths are kept as-is, arrays are normalized
        self._texture = _normalize_texture_input(value)
        self._texture_hash = None
        self._cached_hash = None

    @property
    def texture_hash(self) -> int:
        """Content-based hash of the assigned texture.

        Returns a stable integer hash derived from the texture data.
        The value is lazily computed and cached until :attr:`texture`
        is reassigned.
        """
        return self._compute_texture_hash()

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
            hull_mesh.inertia = self.inertia
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

    # ---- Factory methods ---------------------------------------------------

    @staticmethod
    def create_from_usd(prim, **kwargs) -> "Mesh":
        """Load a Mesh from a USD prim with the ``UsdGeom.Mesh`` schema.

        This is a convenience wrapper around :func:`newton.usd.get_mesh`.
        See that function for full documentation.

        Args:
            prim: The USD prim to load the mesh from.
            **kwargs: Additional arguments passed to :func:`newton.usd.get_mesh`
                (e.g. ``load_normals``, ``load_uvs``).

        Returns:
            Mesh: A new Mesh instance.
        """
        from ..usd.utils import get_mesh  # noqa: PLC0415

        result = get_mesh(prim, **kwargs)
        if isinstance(result, tuple):
            return result[0]
        return result

    @staticmethod
    def create_from_file(filename: str, method: str | None = None, **kwargs) -> "Mesh":
        """Load a Mesh from a 3D model file.

        Supports common surface mesh formats including OBJ, PLY, STL, and
        other formats supported by trimesh, meshio, openmesh, or pcu.

        Args:
            filename: Path to the mesh file.
            method: Loading backend to use (``"trimesh"``, ``"meshio"``,
                ``"pcu"``, ``"openmesh"``). If ``None``, each backend is
                tried in order until one succeeds.
            **kwargs: Additional arguments passed to the :class:`Mesh`
                constructor (e.g. ``compute_inertia``, ``is_solid``).

        Returns:
            Mesh: A new Mesh instance.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        from .utils import load_mesh  # noqa: PLC0415

        mesh_points, mesh_indices = load_mesh(filename, method=method)
        return Mesh(vertices=mesh_points, indices=mesh_indices, **kwargs)


class TetMesh:
    """Represents a tetrahedral mesh for volumetric deformable simulation.

    Stores vertex positions (surface + interior nodes), tetrahedral element
    connectivity, and an optional surface triangle mesh. If no surface mesh
    is provided, it is automatically computed from the open (unshared) faces
    of the tetrahedra.

    Optionally carries per-element material arrays and a density value loaded
    from file. These are used as defaults by builder methods and can be
    overridden at instantiation time.

    Example:
        Create a TetMesh from raw arrays:

        .. code-block:: python

            import numpy as np
            import newton

            vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
            tet_indices = np.array([0, 1, 2, 3], dtype=np.int32)
            tet_mesh = newton.TetMesh(vertices, tet_indices)
    """

    _RESERVED_ATTR_KEYS = frozenset({"vertices", "tet_indices", "k_mu", "k_lambda", "k_damp", "density"})

    def __init__(
        self,
        vertices: Sequence[Vec3] | np.ndarray,
        tet_indices: Sequence[int] | np.ndarray,
        k_mu: np.ndarray | float | None = None,
        k_lambda: np.ndarray | float | None = None,
        k_damp: np.ndarray | float | None = None,
        density: float | None = None,
        custom_attributes: (
            "dict[str, np.ndarray] | dict[str, tuple[np.ndarray, Model.AttributeFrequency]] | None"
        ) = None,
    ):
        """Construct a TetMesh from vertex positions and tet connectivity.

        Args:
            vertices: Vertex positions [m], shape (N, 3).
            tet_indices: Tetrahedral element indices, flattened (4 per tet).
            k_mu: First elastic Lame parameter [Pa]. Scalar (uniform) or
                per-element array of shape (tet_count,).
            k_lambda: Second elastic Lame parameter [Pa]. Scalar (uniform) or
                per-element array of shape (tet_count,).
            k_damp: Rayleigh damping coefficient [-] (dimensionless). Scalar
                (uniform) or per-element array of shape (tet_count,).
            density: Uniform density [kg/m^3] for mass computation.
            custom_attributes: Dictionary of named custom arrays with their
                :class:`~newton.Model.AttributeFrequency`. Each value can be
                either a bare array (frequency auto-inferred from length) or a
                ``(array, frequency)`` tuple.
        """
        self._vertices = np.array(vertices, dtype=np.float32).reshape(-1, 3)
        self._tet_indices = np.array(tet_indices, dtype=np.int32).flatten()
        if len(self._tet_indices) % 4 != 0:
            raise ValueError(f"tet_indices length must be a multiple of 4, got {len(self._tet_indices)}.")

        vertex_count = len(self._vertices)
        if len(self._tet_indices) > 0:
            idx_min = int(self._tet_indices.min())
            idx_max = int(self._tet_indices.max())
            if idx_min < 0:
                raise ValueError(f"tet_indices contains negative index {idx_min}.")
            if idx_max >= vertex_count:
                raise ValueError(f"tet_indices contains index {idx_max} which exceeds vertex count {vertex_count}.")

        tet_count = len(self._tet_indices) // 4

        self._k_mu = self._broadcast_material(k_mu, tet_count, "k_mu")
        self._k_lambda = self._broadcast_material(k_lambda, tet_count, "k_lambda")
        self._k_damp = self._broadcast_material(k_damp, tet_count, "k_damp")
        self._density = density
        # Compute surface triangles from boundary faces (before custom attrs so tri_count is available)
        self._surface_tri_indices = self._compute_surface_triangles()
        tri_count = len(self._surface_tri_indices) // 3

        self.custom_attributes: dict[str, tuple[np.ndarray, int]] = {}
        for k, v in (custom_attributes or {}).items():
            if k in self._RESERVED_ATTR_KEYS:
                raise ValueError(
                    f"Custom attribute name '{k}' is reserved. Reserved names: {sorted(self._RESERVED_ATTR_KEYS)}"
                )
            if isinstance(v, tuple):
                arr, freq = v
                self.custom_attributes[k] = (np.asarray(arr), freq)
            else:
                arr = np.asarray(v)
                freq = self._infer_frequency(arr, vertex_count, tet_count, tri_count, k)
                self.custom_attributes[k] = (arr, freq)

        self._cached_hash: int | None = None

    @staticmethod
    def _broadcast_material(value: np.ndarray | float | None, tet_count: int, name: str) -> np.ndarray | None:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            return np.full(tet_count, arr.item(), dtype=np.float32)
        arr = arr.flatten()
        if len(arr) == 1:
            return np.full(tet_count, arr[0], dtype=np.float32)
        if len(arr) != tet_count:
            raise ValueError(f"{name} array length ({len(arr)}) does not match tet count ({tet_count}).")
        return arr

    @staticmethod
    def _infer_frequency(
        arr: np.ndarray, vertex_count: int, tet_count: int, tri_count: int, name: str
    ) -> "Model.AttributeFrequency":
        """Infer :class:`~newton.Model.AttributeFrequency` from array length.

        Args:
            arr: The attribute array.
            vertex_count: Number of vertices in the mesh.
            tet_count: Number of tetrahedra in the mesh.
            tri_count: Number of surface triangles in the mesh.
            name: Attribute name (for error messages).

        Returns:
            The inferred frequency.

        Raises:
            ValueError: If the array length is ambiguous (matches multiple
                counts) or matches none of the known counts.
        """
        from ..sim.model import Model  # noqa: PLC0415

        first_dim = arr.shape[0] if arr.ndim >= 1 else 1
        counts = {"vertex_count": vertex_count, "tet_count": tet_count, "tri_count": tri_count}
        matches = [label for label, c in counts.items() if first_dim == c and c > 0]
        if len(matches) > 1:
            raise ValueError(
                f"Cannot infer frequency for custom attribute '{name}': array length {first_dim} matches "
                f"{', '.join(matches)}. Pass an explicit (array, frequency) tuple instead."
            )
        if first_dim == vertex_count and vertex_count > 0:
            return Model.AttributeFrequency.PARTICLE
        if first_dim == tet_count and tet_count > 0:
            return Model.AttributeFrequency.TETRAHEDRON
        if first_dim == tri_count and tri_count > 0:
            return Model.AttributeFrequency.TRIANGLE
        raise ValueError(
            f"Cannot infer frequency for custom attribute '{name}': array length {first_dim} matches none of "
            f"vertex_count ({vertex_count}), tet_count ({tet_count}), tri_count ({tri_count}). "
            f"Pass an explicit (array, frequency) tuple instead."
        )

    @staticmethod
    def compute_surface_triangles(tet_indices: np.ndarray) -> np.ndarray:
        """Extract boundary triangles from tetrahedral element indices.

        Finds faces that belong to exactly one tetrahedron (boundary faces)
        using a vectorized approach.

        Args:
            tet_indices: Flattened tetrahedral element indices (4 per tet).

        Returns:
            Flattened boundary triangle indices, 3 per triangle, int32.
        """
        tet_indices = np.asarray(tet_indices, dtype=np.int32).flatten()
        tets = tet_indices.reshape(-1, 4)
        n = len(tets)
        if n == 0:
            return np.array([], dtype=np.int32)

        # Each tet contributes 4 faces with specific winding order:
        #   face 0: (v0, v2, v1)
        #   face 1: (v1, v2, v3)
        #   face 2: (v0, v1, v3)
        #   face 3: (v0, v3, v2)
        # fmt: off
        face_idx = np.array([
            [0, 2, 1],
            [1, 2, 3],
            [0, 1, 3],
            [0, 3, 2],
        ])
        # fmt: on

        # Build all faces: shape (4*n, 3) with original winding
        all_faces = tets[:, face_idx].reshape(-1, 3)

        # Sort vertex indices per face to create canonical keys
        sorted_faces = np.sort(all_faces, axis=1)

        # Find unique sorted faces and their counts
        _, inverse, counts = np.unique(sorted_faces, axis=0, return_inverse=True, return_counts=True)

        # Boundary faces appear exactly once
        boundary_mask = counts[inverse] == 1

        return all_faces[boundary_mask].astype(np.int32).flatten()

    def _compute_surface_triangles(self) -> np.ndarray:
        return TetMesh.compute_surface_triangles(self._tet_indices)

    # ---- Properties --------------------------------------------------------

    @property
    def vertices(self) -> np.ndarray:
        """Vertex positions [m], shape (N, 3), float32."""
        return self._vertices

    @property
    def tet_indices(self) -> np.ndarray:
        """Tetrahedral element indices, flattened, 4 per tet."""
        return self._tet_indices

    @property
    def tet_count(self) -> int:
        """Number of tetrahedral elements."""
        return len(self._tet_indices) // 4

    @property
    def vertex_count(self) -> int:
        """Number of vertices."""
        return len(self._vertices)

    @property
    def surface_tri_indices(self) -> np.ndarray:
        """Surface triangle indices (open faces), flattened, 3 per tri.

        Automatically computed from tet connectivity at construction time
        by extracting boundary faces (faces belonging to exactly one tet).
        """
        return self._surface_tri_indices

    @property
    def k_mu(self) -> np.ndarray | None:
        """Per-element first Lame parameter [Pa], shape (tet_count,) or None."""
        return self._k_mu

    @property
    def k_lambda(self) -> np.ndarray | None:
        """Per-element second Lame parameter [Pa], shape (tet_count,) or None."""
        return self._k_lambda

    @property
    def k_damp(self) -> np.ndarray | None:
        """Per-element Rayleigh damping coefficient [-], shape (tet_count,) or None."""
        return self._k_damp

    @property
    def density(self) -> float | None:
        """Uniform density [kg/m^3] or None."""
        return self._density

    # ---- Factory methods ---------------------------------------------------

    @staticmethod
    def create_from_usd(prim) -> "TetMesh":
        """Load a tetrahedral mesh from a USD prim with the ``UsdGeom.TetMesh`` schema.

        Reads vertex positions from the ``points`` attribute and tetrahedral
        connectivity from ``tetVertexIndices``. If a physics material is bound
        to the prim (via ``material:binding:physics``) and contains
        ``youngsModulus``, ``poissonsRatio``, or ``density`` attributes
        (under the ``omniphysics:`` or ``physxDeformableBody:`` namespaces),
        those values are read and converted to Lame parameters (``k_mu``,
        ``k_lambda``) and density on the returned TetMesh. Material properties
        are set to ``None`` if not present.

        Example:

            .. code-block:: python

                from pxr import Usd
                import newton
                import newton.usd

                usd_stage = Usd.Stage.Open("tetmesh.usda")
                tetmesh = newton.usd.get_tetmesh(usd_stage.GetPrimAtPath("/MyTetMesh"))

                # tetmesh.vertices  -- np.ndarray, shape (N, 3)
                # tetmesh.tet_indices -- np.ndarray, flattened (4 per tet)

        Args:
            prim: The USD prim to load the tetrahedral mesh from.

        Returns:
            TetMesh: A :class:`newton.TetMesh` with vertex positions and tet connectivity.
        """
        from ..usd.utils import get_tetmesh  # noqa: PLC0415

        return get_tetmesh(prim)

    @staticmethod
    def create_from_file(filename: str) -> "TetMesh":
        """Load a TetMesh from a volumetric mesh file.

        Supports ``.vtk``, ``.msh``, ``.vtu``, and other formats with
        tetrahedral cells via meshio. Also supports ``.npz`` files saved
        by :meth:`TetMesh.save` (numpy only, no extra dependencies).

        Args:
            filename: Path to the volumetric mesh file.

        Returns:
            TetMesh: A new TetMesh instance.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        ext = os.path.splitext(filename)[1].lower()

        if ext == ".npz":
            data = np.load(filename)
            kwargs = {}
            for key in ("k_mu", "k_lambda", "k_damp"):
                if key in data:
                    kwargs[key] = data[key]
            if "density" in data:
                kwargs["density"] = float(data["density"])
            known_keys = {
                "vertices",
                "tet_indices",
                "k_mu",
                "k_lambda",
                "k_damp",
                "density",
                "__custom_names__",
                "__custom_freqs__",
            }
            freq_map: dict[str, int] = {}
            if "__custom_names__" in data and "__custom_freqs__" in data:
                from ..sim.model import Model as _Model  # noqa: PLC0415

                names = data["__custom_names__"]
                freqs = data["__custom_freqs__"]
                for n, f in zip(names, freqs, strict=True):
                    freq_map[str(n)] = int(f)
            custom: dict[str, np.ndarray | tuple] = {}
            for k in data.files:
                if k not in known_keys:
                    arr = np.asarray(data[k])
                    if k in freq_map:
                        from ..sim.model import Model as _Model  # noqa: PLC0415

                        custom[k] = (arr, _Model.AttributeFrequency(freq_map[k]))
                    else:
                        custom[k] = arr
            if custom:
                kwargs["custom_attributes"] = custom
            return TetMesh(
                vertices=data["vertices"],
                tet_indices=data["tet_indices"],
                **kwargs,
            )

        import meshio

        m = meshio.read(filename)

        # Find tetrahedral cells
        tet_indices = None
        tet_cell_idx = None
        for i, cell_block in enumerate(m.cells):
            if cell_block.type == "tetra":
                tet_indices = np.array(cell_block.data, dtype=np.int32).flatten()
                tet_cell_idx = i
                break

        if tet_indices is None:
            raise ValueError(f"No tetrahedral cells found in '{filename}'.")

        vertices = np.array(m.points, dtype=np.float32)

        # Read material arrays from cell data
        kwargs: dict = {}
        material_keys = {"k_mu", "k_lambda", "k_damp", "density"}
        if m.cell_data and tet_cell_idx is not None:
            for key in material_keys:
                if key in m.cell_data:
                    arr = np.asarray(m.cell_data[key][tet_cell_idx], dtype=np.float32)
                    if key == "density":
                        if arr.size > 1 and not np.allclose(arr, arr[0]):
                            raise ValueError(
                                f"Non-uniform per-element density found in '{filename}'. "
                                f"TetMesh only supports a single uniform density value."
                            )
                        kwargs["density"] = float(arr[0])
                    else:
                        kwargs[key] = arr

        # Read custom attributes from cell data and point data
        from ..sim.model import Model as _Model  # noqa: PLC0415

        custom: dict[str, tuple[np.ndarray, _Model.AttributeFrequency]] = {}
        if m.cell_data and tet_cell_idx is not None:
            for key, arrays in m.cell_data.items():
                if key not in material_keys:
                    custom[key] = (np.asarray(arrays[tet_cell_idx]), _Model.AttributeFrequency.TETRAHEDRON)
        if m.point_data:
            for key, arr in m.point_data.items():
                custom[key] = (np.asarray(arr), _Model.AttributeFrequency.PARTICLE)
        if custom:
            kwargs["custom_attributes"] = custom

        return TetMesh(vertices=vertices, tet_indices=tet_indices, **kwargs)

    def save(self, filename: str):
        """Save the TetMesh to a file.

        For ``.npz``, saves all arrays via :func:`numpy.savez` (no extra
        dependencies). For other formats (``.vtk``, ``.msh``, ``.vtu``,
        etc.), uses meshio.

        Args:
            filename: Path to write the file to.
        """
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".npz":
            save_dict = {
                "vertices": self._vertices,
                "tet_indices": self._tet_indices,
            }
            if self._k_mu is not None:
                save_dict["k_mu"] = self._k_mu
            if self._k_lambda is not None:
                save_dict["k_lambda"] = self._k_lambda
            if self._k_damp is not None:
                save_dict["k_damp"] = self._k_damp
            if self._density is not None:
                save_dict["density"] = np.array(self._density)
            custom_names = []
            custom_freqs = []
            for k, (arr, freq) in self.custom_attributes.items():
                save_dict[k] = arr
                custom_names.append(k)
                custom_freqs.append(int(freq))
            if custom_names:
                save_dict["__custom_names__"] = np.array(custom_names)
                save_dict["__custom_freqs__"] = np.array(custom_freqs, dtype=np.int32)
            np.savez(filename, **save_dict)
            return

        import meshio

        cells = [("tetra", self._tet_indices.reshape(-1, 4))]
        cell_data: dict[str, list[np.ndarray]] = {}
        point_data: dict[str, np.ndarray] = {}

        # Save material arrays as cell data
        for name, arr in [("k_mu", self._k_mu), ("k_lambda", self._k_lambda), ("k_damp", self._k_damp)]:
            if arr is not None:
                cell_data[name] = [arr]
        if self._density is not None:
            cell_data["density"] = [np.full(self.tet_count, self._density, dtype=np.float32)]

        # Save custom attributes as point or cell data based on frequency
        from ..sim.model import Model as _Model  # noqa: PLC0415

        for name, (arr, freq) in self.custom_attributes.items():
            if freq == _Model.AttributeFrequency.TETRAHEDRON:
                cell_data[name] = [arr]
            elif freq == _Model.AttributeFrequency.PARTICLE:
                point_data[name] = arr
            else:
                warnings.warn(
                    f"Custom attribute '{name}' with frequency {freq} cannot be saved to meshio format "
                    f"(only PARTICLE and TETRAHEDRON are supported). Skipping.",
                    stacklevel=2,
                )

        mesh = meshio.Mesh(
            points=self._vertices,
            cells=cells,
            cell_data=cell_data if cell_data else {},
            point_data=point_data if point_data else {},
        )
        mesh.write(filename)

    def __eq__(self, other):
        if not isinstance(other, TetMesh):
            return NotImplemented
        return np.array_equal(self._vertices, other._vertices) and np.array_equal(self._tet_indices, other._tet_indices)

    def __hash__(self):
        if self._cached_hash is None:
            self._cached_hash = hash((self._vertices.tobytes(), self._tet_indices.tobytes()))
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
        data: Sequence[Sequence[float]] | np.ndarray,
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
            hx: Half-extent in X direction. The heightfield spans [-hx, +hx].
            hy: Half-extent in Y direction. The heightfield spans [-hy, +hy].
            min_z: World-space Z value corresponding to data minimum. Must be provided
                together with ``max_z``, or both omitted to auto-derive from data.
            max_z: World-space Z value corresponding to data maximum. Must be provided
                together with ``min_z``, or both omitted to auto-derive from data.
        """
        if nrow < 2 or ncol < 2:
            raise ValueError(f"Heightfield requires nrow >= 2 and ncol >= 2, got nrow={nrow}, ncol={ncol}")
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
        self.inertia = wp.mat33()
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


class Gaussian:
    """Represents a Gaussian splat asset for rendering and rigid body attachment.

    A Gaussian splat is a collection of oriented, scaled 3D Gaussians with
    appearance data (color via spherical harmonics or flat RGB). Gaussian
    objects can be attached to rigid bodies as a shape type (``GeoType.GAUSSIAN``)
    for rendering, with collision handled by an optional proxy geometry.

    Example:
        Load a Gaussian splat from a ``.ply`` file and inspect it:

        .. code-block:: python

            import newton

            gaussian = newton.Gaussian.create_from_ply("object.ply")
            print(gaussian.count, gaussian.sh_degree)
    """

    @wp.struct
    class Data:
        num_points: wp.int32
        transforms: wp.array(dtype=wp.transformf)
        scales: wp.array(dtype=wp.vec3f)
        opacities: wp.array(dtype=wp.float32)
        sh_coeffs: wp.array(dtype=wp.float32, ndim=2)
        bvh_id: wp.uint64
        min_response: wp.float32

    def __init__(
        self,
        positions: np.ndarray,
        rotations: np.ndarray | None = None,
        scales: np.ndarray | None = None,
        opacities: np.ndarray | None = None,
        sh_coeffs: np.ndarray | None = None,
        sh_degree: int | None = None,
        min_response: float = 0.1,
    ):
        """Construct a Gaussian splat asset from arrays.

        Args:
            positions: Gaussian centers in local space [m], shape ``(N, 3)``, float.
            rotations: Quaternion orientations ``(x, y, z, w)``, shape ``(N, 4)``, float.
                If ``None``, defaults to identity quaternions.
            scales: Per-axis scales (linear), shape ``(N, 3)``, float.
                If ``None``, defaults to ones.
            opacities: Opacity values ``[0, 1]``, shape ``(N,)``, float.
                If ``None``, defaults to ones (fully opaque).
            sh_coeffs: Spherical harmonic coefficients, shape ``(N, C)``, float.
                The number of coefficients *C* determines the SH degree
                (``C = 3`` -> degree 0, ``C = 12`` -> degree 1, etc.).
            sh_degree: Spherical harmonic degree.
            min_response: Minimum response required for alpha testing.
        """

        self._positions = np.ascontiguousarray(np.asarray(positions, dtype=np.float32).reshape(-1, 3))
        n = self._positions.shape[0]

        if rotations is not None:
            self._rotations = np.ascontiguousarray(np.asarray(rotations, dtype=np.float32).reshape(n, 4))
        else:
            self._rotations = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1))

        if scales is not None:
            self._scales = np.ascontiguousarray(np.asarray(scales, dtype=np.float32).reshape(n, 3))
        else:
            self._scales = np.ones((n, 3), dtype=np.float32)

        if opacities is not None:
            self._opacities = np.ascontiguousarray(np.asarray(opacities, dtype=np.float32).reshape(n))
        else:
            self._opacities = np.ones(n, dtype=np.float32)

        if sh_coeffs is not None:
            self._sh_coeffs = np.ascontiguousarray(np.asarray(sh_coeffs, dtype=np.float32).reshape(n, -1))
        else:
            self._sh_coeffs = np.ones((n, 3), dtype=np.float32)

        if sh_degree is not None:
            self._sh_degree = sh_degree
        else:
            self._sh_degree = self._find_sh_degree()

        if not np.isfinite(min_response) or not (0.0 < min_response < 1.0):
            raise ValueError("min_response must be finite and in (0, 1)")
        self._min_response = float(min_response)

        self._cached_hash = None
        self._positions.setflags(write=False)
        self._rotations.setflags(write=False)
        self._scales.setflags(write=False)
        self._opacities.setflags(write=False)
        self._sh_coeffs.setflags(write=False)

        # GPU arrays populated by finalize()
        self.warp_bvh: wp.Bvh = None
        self.warp_data: Gaussian.Data = None

        # Inertia: Gaussians are render-only so they contribute no mass
        self.has_inertia = False
        self.mass = 0.0
        self.com = wp.vec3()
        self.I = wp.mat33()
        self.is_solid = False

    # ---- Properties ----------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of Gaussians in this asset."""
        return self._positions.shape[0]

    @property
    def positions(self) -> np.ndarray:
        """Gaussian centers in local space [m], shape ``(N, 3)``, float."""
        return self._positions

    @property
    def rotations(self) -> np.ndarray:
        """Quaternion orientations ``(x, y, z, w)``, shape ``(N, 4)``, float."""
        return self._rotations

    @property
    def scales(self) -> np.ndarray:
        """Per-axis linear scales, shape ``(N, 3)``, float."""
        return self._scales

    @property
    def opacities(self) -> np.ndarray:
        """Opacity values ``[0, 1]``, shape ``(N,)``, float."""
        return self._opacities

    @property
    def sh_coeffs(self) -> np.ndarray | None:
        """Spherical harmonic coefficients, shape ``(N, C)``, float."""
        return self._sh_coeffs

    @property
    def sh_degree(self) -> int:
        """Spherical harmonics degree (0-3), int"""
        return self._sh_degree

    @property
    def min_response(self) -> float:
        """Min response, float."""
        return self._min_response

    def _find_sh_degree(self) -> int:
        """Spherical harmonics degree (0-3), inferred from *sh_coeffs* shape."""
        c = self._sh_coeffs.shape[1]
        # SH bands: degree 0 -> 1*3=3, degree 1 -> 4*3=12,
        #           degree 2 -> 9*3=27, degree 3 -> 16*3=48
        for deg, num in ((3, 48), (2, 27), (1, 12), (0, 3)):
            if c >= num:
                return deg
        return 0

    # ---- Finalize (GPU upload) -----------------------------------------------

    def finalize(self, device: Devicelike = None) -> Data:
        """Upload Gaussian data to the GPU as Warp arrays.

        Args:
            device: Device on which to allocate buffers.

        Returns:
            Gaussian.Data struct containing the Warp arrays.
        """

        from ..sensors.warp_raytrace.gaussians import compute_gaussian_bvh_bounds  # noqa: PLC0415

        with wp.ScopedDevice(device):
            self.warp_data = Gaussian.Data()
            self.warp_data.transforms = wp.array(
                np.append(self._positions, self._rotations, axis=1), dtype=wp.transformf
            )
            self.warp_data.scales = wp.array(self._scales, dtype=wp.vec3f)
            self.warp_data.opacities = wp.array(self._opacities, dtype=wp.float32)
            self.warp_data.sh_coeffs = wp.array(self._sh_coeffs, dtype=wp.float32)
            self.warp_data.min_response = self.min_response
            self.warp_data.num_points = self.warp_data.transforms.shape[0]

            lowers = wp.zeros(self.count, dtype=wp.vec3f)
            uppers = wp.zeros(self.count, dtype=wp.vec3f)

            wp.launch(
                kernel=compute_gaussian_bvh_bounds,
                dim=self.count,
                inputs=[self.warp_data, lowers, uppers],
            )

            self.warp_bvh = wp.Bvh(lowers, uppers)
            self.warp_data.bvh_id = self.warp_bvh.id
        return self.warp_data

    # ---- Factory methods -----------------------------------------------------

    @staticmethod
    def create_from_ply(filename: str, min_response: float = 0.1) -> "Gaussian":
        """Load Gaussian splat data from a ``.ply`` file (standard 3DGS format).

        Reads positions (``x/y/z``), rotations (``rot_0..3``), scales
        (``scale_0..2``, stored as log-scale), opacities (logit-space),
        and SH coefficients (``f_dc_*``, ``f_rest_*``). Converts log-scale
        and logit-opacity to linear values.

        Args:
            filename: Path to a ``.ply`` file in standard 3DGS format.
            min_response: Min response (default = 0.1).

        Returns:
            A new :class:`Gaussian` instance.
        """
        import open3d as o3d

        pcd = o3d.t.io.read_point_cloud(filename)
        point_attrs = {name: np.asarray(tensor.numpy()) for name, tensor in pcd.point.items()}

        positions = point_attrs.get("positions")
        if positions is None:
            raise ValueError("PLY Gaussian point cloud is missing required 'positions' attribute")
        positions = np.ascontiguousarray(np.asarray(positions, dtype=np.float32).reshape(-1, 3))

        def _get_point_attr(name: str, width: int | None = None) -> np.ndarray | None:
            values = point_attrs.get(name)
            if values is None:
                return None

            values = np.asarray(values, dtype=np.float32)
            if width is None:
                return np.ascontiguousarray(values.reshape(-1))
            return np.ascontiguousarray(values.reshape(-1, width))

        def _require_point_attr(name: str, message: str) -> np.ndarray:
            values = _get_point_attr(name)
            if values is None:
                raise ValueError(message)
            return values

        # Rotations (quaternion w,x,y,z)
        if "rot_0" in point_attrs:
            missing_rotation = "PLY Gaussian point cloud is missing one or more rotation attributes"
            rot_0 = _require_point_attr("rot_0", missing_rotation)
            rot_1 = _require_point_attr("rot_1", missing_rotation)
            rot_2 = _require_point_attr("rot_2", missing_rotation)
            rot_3 = _require_point_attr("rot_3", missing_rotation)

            rotations = np.stack([rot_1, rot_2, rot_3, rot_0], axis=1).astype(np.float32)
            rotations /= np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1e-12)
        else:
            rotations = None

        # Scales (stored as log-scale in standard 3DGS)
        if "scale_0" in point_attrs:
            missing_scale = "PLY Gaussian point cloud is missing one or more scale attributes"
            scale_0 = _require_point_attr("scale_0", missing_scale)
            scale_1 = _require_point_attr("scale_1", missing_scale)
            scale_2 = _require_point_attr("scale_2", missing_scale)

            log_scales = np.stack([scale_0, scale_1, scale_2], axis=1).astype(np.float32)
            scales = np.exp(log_scales)
        else:
            scales = None

        # Opacities (stored in logit-space in standard 3DGS)
        if "opacity" in point_attrs:
            logit_opacities = _get_point_attr("opacity")
            opacities = 1.0 / (1.0 + np.exp(-logit_opacities))
        else:
            opacities = None

        # Spherical harmonic coefficients
        sh_dc_names = [f"f_dc_{i}" for i in range(3)]
        has_sh_dc = all(name in point_attrs for name in sh_dc_names)

        sh_coeffs = None
        if has_sh_dc:
            sh_dc = np.stack(
                [
                    _require_point_attr(name, "PLY Gaussian point cloud is missing SH DC attributes")
                    for name in sh_dc_names
                ],
                axis=1,
            ).astype(np.float32)

            rest_names = []
            i = 0
            while f"f_rest_{i}" in point_attrs:
                rest_names.append(f"f_rest_{i}")
                i += 1

            if rest_names:
                sh_rest = np.stack(
                    [
                        _require_point_attr(name, "PLY Gaussian point cloud is missing SH rest attributes")
                        for name in rest_names
                    ],
                    axis=1,
                ).astype(np.float32)
                sh_coeffs = np.concatenate([sh_dc, sh_rest], axis=1)
            else:
                sh_coeffs = sh_dc

        return Gaussian(
            positions=positions,
            rotations=rotations,
            scales=scales,
            opacities=opacities,
            sh_coeffs=sh_coeffs,
            min_response=min_response,
        )

    @staticmethod
    def create_from_usd(prim, min_response: float = 0.1) -> "Gaussian":
        """Load Gaussian splat data from a USD prim.

        Reads positions from attributes: `positions`, `orientations`, `scales`, `opacities` and `radiance:sphericalHarmonicsCoefficients`.

        Args:
            prim: A USD prim containing Gaussian splat data.
            min_response: Min response (default = 0.1).

        Returns:
            A new :class:`Gaussian` instance.
        """

        from ..usd.utils import get_gaussian  # noqa: PLC0415

        return get_gaussian(prim, min_response=min_response)

    # ---- Utility -------------------------------------------------------------

    def compute_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute axis-aligned bounding box of Gaussian centers.

        Returns:
            Tuple of ``(lower, upper)`` as ``(3,)`` arrays [m].
        """
        lower = self._positions.min(axis=0)
        upper = self._positions.max(axis=0)
        return lower, upper

    def compute_proxy_mesh(self, method: str = "convex_hull") -> "Mesh":
        """Generate a proxy collision :class:`Mesh` from Gaussian positions.

        Args:
            method: ``"convex_hull"`` (default) or ``"alphashape"`` or ``"points"``.

        Returns:
            A :class:`Mesh` for use as collision proxy.
        """

        if method == "convex_hull":
            from .utils import remesh_convex_hull  # noqa: PLC0415

            hull_verts, hull_faces = remesh_convex_hull(self._positions)
            return Mesh(hull_verts, hull_faces, compute_inertia=True)
        elif method == "alphashape":
            from .utils import remesh_alphashape  # noqa: PLC0415

            hull_verts, hull_faces = remesh_alphashape(self._positions)
            return Mesh(hull_verts, hull_faces, compute_inertia=True)
        elif method == "points":
            return self.compute_points_mesh()

        raise ValueError(
            f"Unsupported proxy mesh method: {method!r}. Supported: 'convex_hull', 'alphashape', 'points'."
        )

    def compute_points_mesh(self) -> "Mesh":
        from ..utils.mesh import create_mesh_box  # noqa: PLC0415

        mesh_points, mesh_indices, _normals, _uvs = create_mesh_box(
            1.0,
            1.0,
            1.0,
            duplicate_vertices=False,
            compute_normals=False,
            compute_uvs=False,
        )

        points = (
            (self.positions[: self.count][:, None] + self.scales[: self.count][:, None] * mesh_points)
            .flatten()
            .reshape(-1, 3)
        )
        offsets = mesh_points.shape[0] * np.arange(self.count)
        indices = (offsets[:, None] + mesh_indices).flatten()
        return Mesh(vertices=points, indices=indices)

    @override
    def __hash__(self) -> int:
        if self._cached_hash is None:
            self._cached_hash = hash(
                (
                    self._positions.data.tobytes(),
                    self._rotations.data.tobytes(),
                    self._scales.data.tobytes(),
                    self._opacities.data.tobytes(),
                    self._sh_coeffs.data.tobytes(),
                    float(self._min_response),
                )
            )
        return self._cached_hash

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Gaussian):
            return NotImplemented
        return (
            np.array_equal(self._positions, other._positions)
            and np.array_equal(self._rotations, other._rotations)
            and np.array_equal(self._scales, other._scales)
            and np.array_equal(self._opacities, other._opacities)
            and np.array_equal(self._sh_coeffs, other._sh_coeffs)
            and self._min_response == other._min_response
        )
