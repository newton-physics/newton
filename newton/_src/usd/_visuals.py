# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read USD visual properties using mesh data shared with physics.

Imported lazily by the USD importer so PXR remains an optional dependency.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ..core.types import Axis
from ..geometry import Mesh
from . import utils as usd

if TYPE_CHECKING:
    from pxr import Usd

logger = logging.getLogger(__name__)


def _geometry_path(prim: Usd.Prim) -> str:
    """Return the path of the prim that authors ``prim``'s geometry.

    Instance proxies resolve to their prim in the shared USD prototype.
    """
    if prim.IsInstanceProxy():
        prim = prim.GetPrimInPrototype()
    return str(prim.GetPath())


def _freeze(value: Any) -> Any:
    """Return a hashable stand-in for one resolved material property value.

    Arrays, such as decoded textures, are keyed by identity: hashing their contents
    would copy whole images per prim. The import's material cache keeps them alive.
    """
    if isinstance(value, np.ndarray):
        return ("ndarray", id(value))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    try:
        hash(value)
    except TypeError:
        return ("id", id(value))
    return value


def _material_key(material_props: dict[str, Any]) -> tuple:
    """Return a hashable key identifying a resolved material."""
    return _freeze(material_props)


class _UsdVisuals:
    """Keep material and mesh reads cached for one import."""

    def __init__(self, stage: Usd.Stage):
        self.stage = stage
        self.material_props_cache: dict[str, dict[str, Any]] = {}
        self.mesh_cache: dict[tuple[str, bool, bool], Mesh] = {}
        self.material_mesh_cache: dict[tuple[str | None, ...], Mesh] = {}
        self.uv_primvar_name_cache: dict[str, str | None] = {}
        # Subset index (-1 for the uncovered-face fallback) and render mesh per instanced mesh.
        self.subset_mesh_cache: dict[tuple, list[tuple[int, Mesh]]] = {}

    def get_material_props_cached(self, prim: Usd.Prim) -> dict[str, Any]:
        """Get material properties with caching to avoid repeated traversal."""
        prim_path = str(prim.GetPath())
        if prim_path not in self.material_props_cache:
            self.material_props_cache[prim_path] = usd.resolve_material_properties_for_prim(prim)
        return self.material_props_cache[prim_path]

    def _get_geometry_key(self, prim: Usd.Prim, load_uvs: bool) -> tuple[str, str | None]:
        """Return the prototype geometry path and, when loading UVs, the material-selected UV set.

        A mesh may carry several UV sets, and the material bound to each instance can
        select a different one, so instances share UV-loaded geometry only per UV set.
        """
        if not load_uvs:
            return _geometry_path(prim), None
        prim_path = str(prim.GetPath())
        if prim_path not in self.uv_primvar_name_cache:
            from ._mesh_loading import _resolve_material_uv_primvar_name  # noqa: PLC0415

            self.uv_primvar_name_cache[prim_path] = _resolve_material_uv_primvar_name(prim)
        return _geometry_path(prim), self.uv_primvar_name_cache[prim_path]

    def get_mesh_cached(self, prim: Usd.Prim, *, load_uvs: bool = False, load_normals: bool = False) -> Mesh:
        """Load and cache mesh data to avoid repeated expensive USD mesh extraction.

        Instance proxies of one USD prototype share the cache entry, since their
        geometry is authored once in the prototype.
        """
        key = (*self._get_geometry_key(prim, load_uvs), load_uvs, load_normals)
        if key in self.mesh_cache:
            return self.mesh_cache[key]

        # Normal/UV expansion can change topology, so cache each representation separately.
        mesh = usd.get_mesh(
            prim,
            load_uvs=load_uvs,
            load_normals=load_normals,
            load_visual_materials=False,
        )
        self.mesh_cache[key] = mesh
        return mesh

    def apply_visual_material(self, mesh: Mesh, material_props: dict[str, Any]) -> None:
        """Apply one resolved USD visual material to its owning mesh."""
        texture = material_props.get("texture")
        if texture is not None:
            mesh.texture = texture
        if mesh.texture is not None:
            # Textures provide albedo; do not tint them with the shape palette.
            mesh.color = (1.0, 1.0, 1.0)
        elif material_props.get("color") is not None:
            mesh.color = material_props["color"]

        for key in ("opacity", "roughness", "metallic", "texture_transform"):
            value = material_props.get(key)
            if value is not None:
                setattr(mesh, key, value)

    def get_mesh_with_material(
        self,
        prim: Usd.Prim,
        *,
        load_uvs: bool = False,
        load_normals: bool = False,
        maxhullvert: int | None = None,
    ) -> Mesh:
        """Return a cached mesh carrying the prim's resolved visual material.

        The geometry cache entry stays unmodified. Prims sharing geometry, an identical
        resolved material, and hull-vertex limit, such as instances of one USD prototype,
        share the returned mesh, so callers must not mutate it per prim.

        Args:
            prim: Mesh prim to load.
            load_uvs: Whether to load the material-selected UV set.
            load_normals: Whether to load normals.
            maxhullvert: Hull-vertex limit to store on the mesh, or ``None`` for the default.
        """
        material_props = self.get_material_props_cached(prim)
        geometry_key = self._get_geometry_key(prim, load_uvs)
        key = (*geometry_key, load_uvs, load_normals, maxhullvert, _material_key(material_props))
        mesh = self.material_mesh_cache.get(key)
        if mesh is None:
            mesh = self.get_mesh_cached(prim, load_uvs=load_uvs, load_normals=load_normals).copy()
            self.apply_visual_material(mesh, material_props)
            if maxhullvert is not None:
                mesh.maxhullvert = maxhullvert
            self.material_mesh_cache[key] = mesh
        return mesh

    def get_mesh_with_visual_material(self, prim: Usd.Prim, *, path_name: str, maxhullvert: int | None = None) -> Mesh:
        """Load a renderable mesh without changing physics mass properties."""
        texture = self.get_material_props_cached(prim).get("texture")
        mesh = self.get_mesh_with_material(
            prim, load_uvs=texture is not None, load_normals=True, maxhullvert=maxhullvert
        )
        if mesh.texture is not None and mesh.uvs is None:
            logger.debug("Mesh %s has a texture but no UV coordinates; texture sampling is disabled.", path_name)
        return mesh

    def get_face_material_subsets(self, prim: Usd.Prim) -> list[Usd.Prim]:
        """Return face-based material subsets authored directly under a mesh prim."""
        from pxr import UsdGeom

        subsets = []
        for child in prim.GetChildren():
            try:
                is_subset = child.IsA(UsdGeom.Subset)
            except Exception:
                is_subset = False
            if not is_subset:
                continue

            subset = UsdGeom.Subset(child)
            element_type = subset.GetElementTypeAttr().Get()
            if element_type != UsdGeom.Tokens.face:
                continue
            family_name = subset.GetFamilyNameAttr().Get()
            if family_name and family_name != "materialBind":
                continue
            indices = subset.GetIndicesAttr().Get()
            if not indices:
                continue
            subsets.append(child)
        return subsets

    def get_subset_uvs(self, prim: Usd.Prim, used_vertices: np.ndarray, expected_count: int) -> np.ndarray | None:
        """Return UVs for a material subset when a matching primvar is authored."""
        from pxr import UsdGeom

        max_used_vertex = int(np.max(used_vertices, initial=-1))
        full_mesh_uvs = None
        for primvar in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
            name = primvar.GetBaseName()
            if not name.startswith("st"):
                continue
            values = primvar.Get()
            if values is None:
                continue
            uvs = np.asarray(values, dtype=np.float32)
            if primvar.IsIndexed():
                indices = primvar.GetIndices()
                if indices is None:
                    continue
                indices = np.asarray(indices, dtype=np.int32)
                if len(indices) == expected_count:
                    uvs = uvs[indices]
                    if len(uvs) == expected_count:
                        return uvs
                    continue
                if len(indices) > max_used_vertex:
                    uvs = uvs[indices]
                else:
                    continue
            if len(uvs) == expected_count:
                return uvs
            if full_mesh_uvs is None and len(uvs) > max_used_vertex:
                full_mesh_uvs = uvs[used_vertices]
        return full_mesh_uvs

    def make_visual_submesh(
        self,
        mesh: Mesh,
        triangle_indices: np.ndarray,
        material_props: dict[str, Any],
        *,
        prim: Usd.Prim,
        path_name: str,
    ) -> Mesh | None:
        """Create a render-only mesh slice for the selected triangle rows."""
        if len(triangle_indices) == 0:
            return None

        triangles = mesh.indices.reshape(-1, 3)[triangle_indices]
        used_vertices = np.unique(triangles)
        vertex_remap = np.full(len(mesh.vertices), -1, dtype=np.int32)
        vertex_remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int32)

        normals = None
        if mesh.normals is not None and len(mesh.normals) == len(mesh.vertices):
            normals = mesh.normals[used_vertices]

        uvs = None
        if mesh.uvs is not None and len(mesh.uvs) == len(mesh.vertices):
            uvs = mesh.uvs[used_vertices]
        elif material_props.get("texture") is not None:
            uvs = self.get_subset_uvs(prim, used_vertices, len(used_vertices))

        submesh = Mesh(
            mesh.vertices[used_vertices],
            vertex_remap[triangles].reshape(-1),
            normals=normals,
            uvs=uvs,
            compute_inertia=False,
            is_solid=mesh.is_solid,
            maxhullvert=mesh.maxhullvert,
        )

        self.apply_visual_material(submesh, material_props)
        if submesh.texture is not None and submesh.uvs is None:
            logger.debug(
                "Mesh material subset %s has a texture but no UV coordinates; texture sampling is disabled.",
                path_name,
            )
        return submesh

    def get_visual_material_subset_meshes(self, prim: Usd.Prim) -> list[tuple[str, Mesh]]:
        """Load one render mesh per USD material subset when subsets are authored."""
        from pxr import UsdGeom

        subsets = self.get_face_material_subsets(prim)
        if not subsets:
            return []

        mesh_schema = UsdGeom.Mesh(prim)
        face_counts = mesh_schema.GetFaceVertexCountsAttr().Get()
        if face_counts is None:
            return []
        face_counts = np.asarray(face_counts, dtype=np.int32)
        if len(face_counts) == 0 or np.any(face_counts < 3):
            return []

        subset_props = [(str(subset.GetPath()), usd.resolve_material_properties_for_prim(subset)) for subset in subsets]
        # Instance proxies of one prototype mesh share subset topology, so instances resolving
        # the same materials share their render meshes. Materials can still differ per instance.
        cache_key = None
        if prim.IsInstanceProxy():
            cache_key = (
                *self._get_geometry_key(prim, load_uvs=True),
                tuple(_material_key(material_props) for _, material_props in subset_props),
                _material_key(self.get_material_props_cached(prim)),
            )
            cached = self.subset_mesh_cache.get(cache_key)
            if cached is not None:
                prim_path = str(prim.GetPath())
                return [(subset_props[index][0] if index >= 0 else prim_path, mesh) for index, mesh in cached]
        # Load UVs (and matching authored normals) so each submesh slices real
        # per-corner texture coordinates instead of recovering per-vertex UVs,
        # which scrambles faceVarying UV sets. UV loading unwelds vertices while
        # preserving triangle order, so the per-face subset selection still aligns.
        mesh = self.get_mesh_cached(prim, load_uvs=True, load_normals=True)
        triangle_face_indices = np.repeat(np.arange(len(face_counts), dtype=np.int32), face_counts - 2)
        covered_faces = np.zeros(len(face_counts), dtype=bool)

        submeshes = []
        submesh_indices = []
        for subset_index, (subset_path, material_props) in enumerate(subset_props):
            # Split on authored binding structure, not on whether the bound material's properties
            # resolve: a subset that binds a material Newton does not recognize still becomes its
            # own (unshaded) submesh, so import topology never depends on material vocabulary.
            # The gate is "a binding authored on the subset itself" — direct or collection-based,
            # with or without MaterialBindingAPI applied. ComputeBoundMaterial is deliberately not
            # used here: every subset inherits the parent mesh's binding through it, so full
            # resolution would split unbound subsets, and an ancestor rebind with
            # strongerThanDescendants would make topology depend on rebinding again. Subsets with
            # no authored binding fall through to the uncovered-faces fallback below, which
            # applies the parent mesh material.
            subset = UsdGeom.Subset(self.stage.GetPrimAtPath(subset_path))
            has_authored_binding = any(
                rel.GetName().startswith("material:binding") and rel.GetTargets()
                for rel in subset.GetPrim().GetRelationships()
            )
            if not has_authored_binding:
                continue
            subset_indices = np.asarray(subset.GetIndicesAttr().Get(), dtype=np.int32)
            valid = (subset_indices >= 0) & (subset_indices < len(face_counts))
            if not np.all(valid):
                logger.debug(
                    "Mesh material subset %s: face indices outside the mesh face range; "
                    "out-of-range indices will be ignored.",
                    subset_path,
                )
                subset_indices = subset_indices[valid]
            if len(subset_indices) == 0:
                continue

            face_mask = np.zeros(len(face_counts), dtype=bool)
            face_mask[subset_indices] = True
            triangle_indices = np.nonzero(face_mask[triangle_face_indices])[0]
            submesh = self.make_visual_submesh(mesh, triangle_indices, material_props, prim=prim, path_name=subset_path)
            if submesh is None:
                continue
            covered_faces[subset_indices] = True
            submeshes.append((subset_path, submesh))
            submesh_indices.append(subset_index)

        if not submeshes:
            if cache_key is not None:
                self.subset_mesh_cache[cache_key] = []
            return []

        uncovered_faces = np.nonzero(~covered_faces)[0]
        if len(uncovered_faces) > 0:
            face_mask = np.zeros(len(face_counts), dtype=bool)
            face_mask[uncovered_faces] = True
            triangle_indices = np.nonzero(face_mask[triangle_face_indices])[0]
            fallback_mesh = self.make_visual_submesh(
                mesh,
                triangle_indices,
                self.get_material_props_cached(prim),
                prim=prim,
                path_name=str(prim.GetPath()),
            )
            if fallback_mesh is not None:
                submeshes.insert(0, (str(prim.GetPath()), fallback_mesh))
                submesh_indices.insert(0, -1)

        if cache_key is not None:
            self.subset_mesh_cache[cache_key] = [
                (index, mesh) for index, (_, mesh) in zip(submesh_indices, submeshes, strict=True)
            ]
        return submeshes

    def get_axial_visual_dimensions(
        self, prim: Usd.Prim, scale: wp.vec3, axis: Axis, default_radius: float, default_height: float
    ) -> tuple[float, float]:
        """Return scaled (radius, half_height); radius uses the largest perpendicular scale to match UsdPhysics."""
        radius = usd.get_float(prim, "radius", default_radius)
        half_height = usd.get_float(prim, "height", default_height) / 2
        axis_index = int(axis)
        radius_scale = max(scale[index] for index in range(3) if index != axis_index)
        return radius * radius_scale, half_height * scale[axis_index]

    def get_planar_visual_dimensions(self, prim: Usd.Prim, scale: wp.vec3, axis: Axis) -> tuple[float, float]:
        """Return scaled (width, length); UsdGeomPlane aligns width to Z for X-axis planes and length to Z for Y-axis planes."""
        width_scale = scale[2] if axis == Axis.X else scale[0]
        length_scale = scale[2] if axis == Axis.Y else scale[1]
        width = usd.get_float(prim, "width", 0.0) * width_scale
        length = usd.get_float(prim, "length", 0.0) * length_scale
        return width, length

    def has_visual_material_properties(self, material_props: dict[str, Any]) -> bool:
        # Require PBR-like material cues to avoid promoting generic displayColor-only colliders.
        return any(material_props.get(key) is not None for key in ("texture", "roughness", "metallic"))

    def is_effectively_visible(self, prim: Usd.Prim) -> bool:
        """Return whether ``prim`` is effectively visible in USD.

        A prim is effectively visible only when it is a :class:`UsdGeom.Imageable`
        whose inherited visibility is not ``invisible``. Non-imageable prims are
        not renderable in USD, so they are treated as not effectively visible.
        """
        from pxr import UsdGeom

        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            return False
        return imageable.ComputeVisibility() != UsdGeom.Tokens.invisible

    def is_viewport_drawn(self, prim: Usd.Prim) -> bool:
        """Return whether a prim is drawn under viewport semantics.

        USD viewports draw the ``default`` and ``proxy`` purposes and hide ``guide`` and
        ``render``; the allowlist also keeps any future purpose hidden until explicitly
        handled. This is what decides whether a collider is drawn: ``guide`` is the
        conventional purpose for authored collision geometry (e.g. the MuJoCo USD
        exporter), and such a prim is not viewport geometry. ``force_show_colliders``
        is the explicit override for inspecting it anyway.
        """
        from pxr import UsdGeom

        if not self.is_effectively_visible(prim):
            return False
        return UsdGeom.Imageable(prim).ComputePurpose() in (UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy)
