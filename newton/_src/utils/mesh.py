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

import os
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast, overload

import numpy as np
import warp as wp

from ..geometry.types import Mesh


@wp.kernel
def accumulate_vertex_normals(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    # output
    normals: wp.array(dtype=wp.vec3),
):
    """Accumulate per-face normals into per-vertex normals (not normalized)."""
    face = wp.tid()
    i0 = indices[face * 3]
    i1 = indices[face * 3 + 1]
    i2 = indices[face * 3 + 2]
    v0 = points[i0]
    v1 = points[i1]
    v2 = points[i2]
    normal = wp.cross(v1 - v0, v2 - v0)
    wp.atomic_add(normals, i0, normal)
    wp.atomic_add(normals, i1, normal)
    wp.atomic_add(normals, i2, normal)


@wp.kernel
def normalize_vertex_normals(normals: wp.array(dtype=wp.vec3)):
    """Normalize per-vertex normals in-place."""
    tid = wp.tid()
    normals[tid] = wp.normalize(normals[tid])


@overload
def compute_vertex_normals(
    points: wp.array,
    indices: wp.array | np.ndarray,
    normals: wp.array | None = None,
    *,
    device: wp.DeviceLike = None,
    normalize: bool = True,
) -> wp.array: ...


@overload
def compute_vertex_normals(
    points: np.ndarray,
    indices: np.ndarray,
    normals: np.ndarray | None = None,
    *,
    device: wp.DeviceLike = None,
    normalize: bool = True,
) -> np.ndarray: ...


def compute_vertex_normals(
    points: wp.array | np.ndarray,
    indices: wp.array | np.ndarray,
    normals: wp.array | np.ndarray | None = None,
    *,
    device: wp.DeviceLike = None,
    normalize: bool = True,
) -> wp.array | np.ndarray:
    """Compute per-vertex normals from triangle indices.

    Supports Warp and NumPy arrays. NumPy inputs run on the CPU via Warp and return
    NumPy output.

    Args:
        points: Vertex positions (wp.vec3 array or Nx3 NumPy array).
        indices: Triangle indices (flattened or Nx3). Warp arrays are expected to be flattened.
        normals: Optional output array to reuse (Warp or NumPy to match ``points``).
        device: Warp device to run on. NumPy inputs default to CPU.
        normalize: Whether to normalize the accumulated normals.

    Returns:
        Per-vertex normals as a Warp array or NumPy array matching the input type.
    """
    if isinstance(points, wp.array):
        if normals is not None and not isinstance(normals, wp.array):
            raise TypeError("normals must be a Warp array when points is a Warp array.")
        device_obj = points.device if device is None else wp.get_device(device)
        indices_wp = indices
        if isinstance(indices, np.ndarray):
            indices_np = np.asarray(indices, dtype=np.int32)
            if indices_np.ndim == 2:
                indices_np = indices_np.reshape(-1)
            elif indices_np.ndim != 1:
                raise ValueError("indices must be flat or (N, 3) for NumPy inputs.")
            indices_wp = wp.array(indices_np, dtype=wp.int32, device=device_obj)
        indices_wp = cast(wp.array, indices_wp)
        if normals is None:
            normals_wp = wp.zeros_like(points)
        else:
            normals_wp = cast(wp.array, normals)
            normals_wp.zero_()
        if len(indices_wp) == 0 or len(points) == 0:
            return normals_wp
        indices_i32 = indices_wp if indices_wp.dtype == wp.int32 else indices_wp.view(dtype=wp.int32)
        wp.launch(
            accumulate_vertex_normals,
            dim=len(indices_i32) // 3,
            inputs=[points, indices_i32],
            outputs=[normals_wp],
            device=device_obj,
        )
        if normalize:
            wp.launch(normalize_vertex_normals, dim=len(normals_wp), inputs=[normals_wp], device=device_obj)
        return normals_wp

    points_np = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    indices_np = np.asarray(indices, dtype=np.int32)
    if indices_np.ndim == 2:
        indices_np = indices_np.reshape(-1)
    elif indices_np.ndim != 1:
        raise ValueError("indices must be flat or (N, 3) for NumPy inputs.")

    normals_np = None
    if normals is not None:
        normals_np = np.asarray(normals, dtype=np.float32).reshape(points_np.shape)
    device_obj = wp.get_device("cpu") if device is None else wp.get_device(device)
    points_wp = wp.array(points_np, dtype=wp.vec3, device=device_obj)
    indices_wp = wp.array(indices_np, dtype=wp.int32, device=device_obj)
    if normals_np is None:
        normals_wp = wp.zeros_like(points_wp)
    else:
        normals_wp = wp.array(normals_np, dtype=wp.vec3, device=device_obj)
        normals_wp.zero_()
    if len(points_wp) == 0 or len(indices_wp) == 0:
        if normals_np is None:
            return np.zeros_like(points_np, dtype=np.float32)
        normals_np[...] = 0.0
        return normals_np
    wp.launch(
        accumulate_vertex_normals,
        dim=len(indices_wp) // 3,
        inputs=[points_wp, indices_wp],
        outputs=[normals_wp],
        device=device_obj,
    )
    if normalize:
        wp.launch(normalize_vertex_normals, dim=len(normals_wp), inputs=[normals_wp], device=device_obj)
    normals_out = normals_wp.numpy()
    if normals_np is not None:
        normals_np[...] = normals_out
        return normals_np
    return normals_out


def smooth_vertex_normals_by_position(
    mesh_vertices: np.ndarray, mesh_faces: np.ndarray, eps: float = 1.0e-6
) -> np.ndarray:
    """Smooth vertex normals by averaging normals of vertices with shared positions."""
    normals = compute_vertex_normals(mesh_vertices, mesh_faces)
    if len(mesh_vertices) == 0:
        return normals
    keys = np.round(mesh_vertices / eps).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    accum = np.zeros((len(unique_keys), 3), dtype=np.float32)
    np.add.at(accum, inverse, normals)
    lengths = np.linalg.norm(accum, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1.0e-8)
    accum = accum / lengths
    return accum[inverse]


# Default number of segments for mesh generation
default_num_segments = 32


class MeshAdjacency:
    """Builds and stores edge adjacency information for a triangle mesh.

    This class processes triangle indices to create a mapping from edges to
    their adjacent triangles. Each edge stores references to both adjacent
    triangles (if they exist) along with the opposite vertices.

    Attributes:
        edges: Dictionary mapping edge keys (min_vertex, max_vertex) to MeshAdjacency.Edge objects.
        indices: The original triangle indices used to build the adjacency.
    """

    @dataclass
    class Edge:
        """Represents an edge in a triangle mesh with adjacency information.

        Stores the two vertices of the edge, the opposite vertices from each
        adjacent triangle, and the indices of those triangles. The winding order
        is consistent: the first triangle is reconstructed as {v0, v1, o0}, and
        the second triangle as {v1, v0, o1}.

        For boundary edges (edges with only one adjacent triangle), o1 and f1
        are set to -1.
        """

        v0: int
        """Index of the first vertex of the edge."""
        v1: int
        """Index of the second vertex of the edge."""
        o0: int
        """Index of the vertex opposite to the edge in the first adjacent triangle."""
        o1: int
        """Index of the vertex opposite to the edge in the second adjacent triangle, or -1 if boundary."""
        f0: int
        """Index of the first adjacent triangle."""
        f1: int
        """Index of the second adjacent triangle, or -1 if boundary edge."""

    def __init__(self, indices: Sequence[Sequence[int]] | np.ndarray):
        """Build edge adjacency from triangle indices.

        Args:
            indices: Array-like of triangle indices, where each element is a
                sequence of 3 vertex indices defining a triangle.
        """
        self.edges: dict[tuple[int, int], MeshAdjacency.Edge] = {}
        self.indices = indices

        for index, tri in enumerate(indices):
            self.add_edge(tri[0], tri[1], tri[2], index)
            self.add_edge(tri[1], tri[2], tri[0], index)
            self.add_edge(tri[2], tri[0], tri[1], index)

    def add_edge(self, i0: int, i1: int, o: int, f: int):
        """Add or update an edge in the adjacency structure.

        If the edge already exists, updates it with the second adjacent triangle.
        If the edge would have more than two adjacent triangles, prints a warning
        (non-manifold edge).

        Args:
            i0: Index of the first vertex of the edge.
            i1: Index of the second vertex of the edge.
            o: Index of the opposite vertex in the triangle.
            f: Index of the triangle containing this edge.
        """
        key = (min(i0, i1), max(i0, i1))
        edge = None

        if key in self.edges:
            edge = self.edges[key]

            if edge.f1 != -1:
                warnings.warn("Detected non-manifold edge", stacklevel=2)
                return
            else:
                # update other side of the edge
                edge.o1 = o
                edge.f1 = f
        else:
            # create new edge with opposite yet to be filled
            edge = MeshAdjacency.Edge(i0, i1, o, -1, f, -1)

        self.edges[key] = edge


def create_mesh_sphere(
    radius: float = 1.0,
    *,
    num_latitudes: int = default_num_segments,
    num_longitudes: int = default_num_segments,
    reverse_winding: bool = False,
    compute_normals: bool = True,
    compute_uvs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Create sphere geometry data with optional normals and UVs."""
    positions = []
    normals = [] if compute_normals else None
    uvs = [] if compute_uvs else None
    indices = []

    for i in range(num_latitudes + 1):
        theta = i * np.pi / num_latitudes
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        for j in range(num_longitudes + 1):
            phi = j * 2 * np.pi / num_longitudes
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            x = cos_phi * sin_theta
            y = cos_theta
            z = sin_phi * sin_theta
            positions.append([x * radius, y * radius, z * radius])
            if compute_normals:
                normals.append([x, y, z])
            if compute_uvs:
                u = float(j) / num_longitudes
                v = float(i) / num_latitudes
                uvs.append([u, v])

    for i in range(num_latitudes):
        for j in range(num_longitudes):
            first = i * (num_longitudes + 1) + j
            second = first + num_longitudes + 1
            if reverse_winding:
                indices.extend([first, second, first + 1, second, second + 1, first + 1])
            else:
                indices.extend([first, first + 1, second, second, first + 1, second + 1])

    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
        None if normals is None else np.asarray(normals, dtype=np.float32),
        None if uvs is None else np.asarray(uvs, dtype=np.float32),
    )


def create_mesh_ellipsoid(
    rx: float = 1.0,
    ry: float = 1.0,
    rz: float = 1.0,
    *,
    num_latitudes: int = default_num_segments,
    num_longitudes: int = default_num_segments,
    reverse_winding: bool = False,
    compute_normals: bool = True,
    compute_uvs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Create ellipsoid geometry data with optional normals and UVs."""
    positions = []
    normals = [] if compute_normals else None
    uvs = [] if compute_uvs else None
    indices = []

    for i in range(num_latitudes + 1):
        theta = i * np.pi / num_latitudes
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        for j in range(num_longitudes + 1):
            phi = j * 2 * np.pi / num_longitudes
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            ux = cos_phi * sin_theta
            uy = cos_theta
            uz = sin_phi * sin_theta
            px = ux * rx
            py = uy * ry
            pz = uz * rz
            positions.append([px, py, pz])

            if compute_normals:
                nx = ux / rx
                ny = uy / ry
                nz = uz / rz
                n_len = np.sqrt(nx * nx + ny * ny + nz * nz)
                if n_len > 1e-10:
                    nx /= n_len
                    ny /= n_len
                    nz /= n_len
                normals.append([nx, ny, nz])
            if compute_uvs:
                u = float(j) / num_longitudes
                v = float(i) / num_latitudes
                uvs.append([u, v])

    for i in range(num_latitudes):
        for j in range(num_longitudes):
            first = i * (num_longitudes + 1) + j
            second = first + num_longitudes + 1
            if reverse_winding:
                indices.extend([first, second, first + 1, second, second + 1, first + 1])
            else:
                indices.extend([first, first + 1, second, second, first + 1, second + 1])

    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
        None if normals is None else np.asarray(normals, dtype=np.float32),
        None if uvs is None else np.asarray(uvs, dtype=np.float32),
    )


def _normalize_color(color) -> tuple[float, float, float] | None:
    if color is None:
        return None
    color = np.asarray(color, dtype=np.float32).flatten()
    if color.size >= 3:
        if np.max(color) > 1.0:
            color = color / 255.0
        return (float(color[0]), float(color[1]), float(color[2]))
    return None


def _extract_trimesh_texture(visual_or_material, base_dir: str) -> np.ndarray | str | None:
    """Extract texture from a trimesh visual or a single material object."""
    material = getattr(visual_or_material, "material", visual_or_material)
    if material is None:
        return None

    image = getattr(material, "image", None)
    image_path = getattr(material, "image_path", None)

    if image is None:
        base_color_texture = getattr(material, "baseColorTexture", None)
        if base_color_texture is not None:
            image = getattr(base_color_texture, "image", None)
            image_path = image_path or getattr(base_color_texture, "image_path", None)

    if image is not None:
        try:
            return np.array(image)
        except Exception:
            pass

    if image_path:
        if not os.path.isabs(image_path):
            image_path = os.path.abspath(os.path.join(base_dir, image_path))
        return image_path

    return None


def _extract_trimesh_material_params(
    material,
) -> tuple[float | None, float | None, tuple[float, float, float] | None]:
    if material is None:
        return None, None, None

    base_color = None
    metallic = None
    roughness = None

    color_candidates = [
        getattr(material, "baseColorFactor", None),
        getattr(material, "diffuse", None),
        getattr(material, "diffuseColor", None),
    ]
    for candidate in color_candidates:
        if candidate is not None:
            base_color = _normalize_color(candidate)
            break

    for attr_name in ("metallicFactor", "metallic"):
        value = getattr(material, attr_name, None)
        if value is not None:
            metallic = float(value)
            break

    for attr_name in ("roughnessFactor", "roughness"):
        value = getattr(material, attr_name, None)
        if value is not None:
            roughness = float(value)
            break

    if roughness is None:
        for attr_name in ("glossiness", "shininess"):
            value = getattr(material, attr_name, None)
            if value is not None:
                gloss = float(value)
                if attr_name == "shininess":
                    gloss = min(max(gloss / 1000.0, 0.0), 1.0)
                roughness = 1.0 - min(max(gloss, 0.0), 1.0)
                break

    return roughness, metallic, base_color


def load_meshes_from_file(
    filename: str,
    *,
    scale: np.ndarray | list[float] | tuple[float, ...] = (1.0, 1.0, 1.0),
    maxhullvert: int,
    override_color: np.ndarray | list[float] | tuple[float, float, float] | None = None,
    override_texture: np.ndarray | str | None = None,
) -> list[Mesh]:
    """Load meshes from a file using trimesh and capture texture data if present.

    Args:
        filename: Path to the mesh file.
        scale: Per-axis scale to apply to vertices.
        maxhullvert: Maximum vertices for convex hull approximation.
        override_color: Optional base color override (RGB).
        override_texture: Optional texture path/URL or image override.

    Returns:
        List of Mesh objects.
    """
    import trimesh

    filename = os.fspath(filename)
    scale = np.asarray(scale, dtype=np.float32)
    base_dir = os.path.dirname(filename)

    def _parse_dae_material_colors(
        path: str,
    ) -> tuple[list[str], dict[str, dict[str, float | tuple[float, float, float] | None]]]:
        try:
            tree = ET.parse(path)
            root = tree.getroot()
        except Exception:
            return [], {}

        def strip(tag: str) -> str:
            return tag.split("}", 1)[-1] if "}" in tag else tag

        # Map effect id -> material properties
        effect_props: dict[str, dict[str, float | tuple[float, float, float] | None]] = {}
        for effect in root.iter():
            if strip(effect.tag) != "effect":
                continue
            effect_id = effect.attrib.get("id")
            if not effect_id:
                continue
            diffuse_color = None
            specular_color = None
            specular_intensity = None
            shininess = None
            for shader_tag in ("phong", "lambert", "blinn"):
                shader = None
                for elem in effect.iter():
                    if strip(elem.tag) == shader_tag:
                        shader = elem
                        break
                if shader is None:
                    continue
                for node in shader.iter():
                    tag = strip(node.tag)
                    if tag == "diffuse":
                        for col in node.iter():
                            if strip(col.tag) == "color" and col.text:
                                values = [float(x) for x in col.text.strip().split()]
                                if len(values) >= 3:
                                    # DAE diffuse colors are commonly authored in linear space.
                                    # Convert to sRGB for the viewer shader (which converts to linear).
                                    diffuse = np.clip(values[:3], 0.0, 1.0)
                                    srgb = np.power(diffuse, 1.0 / 2.2)
                                    diffuse_color = (float(srgb[0]), float(srgb[1]), float(srgb[2]))
                                    break
                        continue
                    if tag == "specular":
                        for col in node.iter():
                            if strip(col.tag) == "color" and col.text:
                                values = [float(x) for x in col.text.strip().split()]
                                if len(values) >= 3:
                                    specular_color = (values[0], values[1], values[2])
                                    break
                        continue
                    if tag == "reflectivity":
                        for val in node.iter():
                            if strip(val.tag) == "float" and val.text:
                                try:
                                    specular_intensity = float(val.text.strip())
                                except ValueError:
                                    specular_intensity = None
                                break
                        continue
                    if tag == "shininess":
                        for val in node.iter():
                            if strip(val.tag) == "float" and val.text:
                                try:
                                    shininess = float(val.text.strip())
                                except ValueError:
                                    shininess = None
                                break
                        continue
                if diffuse_color is not None:
                    break
            metallic = None
            if specular_color is not None:
                metallic = float(np.clip(np.max(specular_color), 0.0, 1.0))
            elif specular_intensity is not None:
                metallic = float(np.clip(specular_intensity, 0.0, 1.0))
            roughness = None
            if shininess is not None:
                if shininess > 1.0:
                    shininess = min(shininess / 128.0, 1.0)
                roughness = float(np.clip(1.0 - shininess, 0.0, 1.0))
            if diffuse_color is not None:
                effect_props[effect_id] = {
                    "color": diffuse_color,
                    "metallic": metallic,
                    "roughness": roughness,
                }

        # Map material id/name -> material properties
        material_colors: dict[str, dict[str, float | tuple[float, float, float] | None]] = {}
        for material in root.iter():
            if strip(material.tag) != "material":
                continue
            mat_id = material.attrib.get("id") or material.attrib.get("name")
            effect_url = None
            for inst in material.iter():
                if strip(inst.tag) == "instance_effect":
                    effect_url = inst.attrib.get("url")
                    break
            if mat_id and effect_url and effect_url.startswith("#"):
                effect_id = effect_url[1:]
                if effect_id in effect_props:
                    material_colors[mat_id] = effect_props[effect_id]

        # Collect triangle material assignments in order
        face_materials: list[str] = []
        for triangles in root.iter():
            if strip(triangles.tag) != "triangles":
                continue
            mat = triangles.attrib.get("material")
            count = triangles.attrib.get("count")
            if not mat or count is None:
                continue
            try:
                tri_count = int(count)
            except ValueError:
                continue
            face_materials.extend([mat] * tri_count)

        return face_materials, material_colors

    dae_face_materials: list[str] = []
    dae_material_colors: dict[str, dict[str, float | tuple[float, float, float] | None]] = {}
    if filename.lower().endswith(".dae"):
        dae_face_materials, dae_material_colors = _parse_dae_material_colors(filename)

    tri = trimesh.load(filename, force="mesh")
    tri_meshes = tri.geometry.values() if hasattr(tri, "geometry") else [tri]

    meshes = []
    for tri_mesh in tri_meshes:
        vertices = np.array(tri_mesh.vertices, dtype=np.float32) * scale
        faces = np.array(tri_mesh.faces, dtype=np.int32)
        normals = np.array(tri_mesh.vertex_normals, dtype=np.float32) if tri_mesh.vertex_normals is not None else None
        if normals is None or not np.isfinite(normals).all() or np.allclose(normals, 0.0):
            normals = compute_vertex_normals(vertices, faces)

        uvs = None
        if hasattr(tri_mesh, "visual") and getattr(tri_mesh.visual, "uv", None) is not None:
            uvs = np.array(tri_mesh.visual.uv, dtype=np.float32)

        color = _normalize_color(override_color) if override_color is not None else None
        texture = override_texture

        def add_mesh_from_faces(
            face_indices,
            *,
            mat_color=None,
            mat_roughness=None,
            mat_metallic=None,
            mesh_vertices=None,
            mesh_normals=None,
            mesh_uvs=None,
            mesh_texture=None,
        ):
            used = np.unique(face_indices.flatten())
            remap = {int(old): i for i, old in enumerate(used)}
            remapped_faces = np.vectorize(remap.get)(face_indices).astype(np.int32)

            sub_vertices = mesh_vertices[used]
            sub_normals = mesh_normals[used] if mesh_normals is not None else None
            force_smooth = False
            if mat_metallic is not None and mat_metallic > 0.0:
                force_smooth = True
            if mat_roughness is not None and mat_roughness < 0.6:
                force_smooth = True
            if sub_normals is None or force_smooth:
                sub_normals = smooth_vertex_normals_by_position(sub_vertices, remapped_faces)
            sub_uvs = mesh_uvs[used] if mesh_uvs is not None else None

            meshes.append(
                Mesh(
                    sub_vertices,
                    remapped_faces.flatten(),
                    normals=sub_normals,
                    uvs=sub_uvs,
                    maxhullvert=maxhullvert,
                    color=mat_color,
                    texture=mesh_texture,
                    roughness=mat_roughness,
                    metallic=mat_metallic,
                )
            )

        # If a uniform override is provided, skip per-material splitting.
        if color is not None or texture is not None:
            add_mesh_from_faces(
                faces,
                mat_color=color,
                mesh_vertices=vertices,
                mesh_normals=normals,
                mesh_uvs=uvs,
                mesh_texture=texture,
            )
            continue

        # Handle per-face materials if available (e.g. DAE with multiple materials)
        face_materials = getattr(tri_mesh.visual, "face_materials", None) if hasattr(tri_mesh, "visual") else None
        materials = getattr(tri_mesh.visual, "materials", None) if hasattr(tri_mesh, "visual") else None
        if face_materials is not None and materials is not None:
            face_materials = np.array(face_materials, dtype=np.int32).flatten()
            for mat_index in np.unique(face_materials):
                mat_faces = faces[face_materials == mat_index]
                material = materials[int(mat_index)] if int(mat_index) < len(materials) else None
                roughness, metallic, base_color = _extract_trimesh_material_params(material)
                mat_color = base_color
                mat_texture = _extract_trimesh_texture(material, base_dir)
                if mat_color is None and hasattr(tri_mesh.visual, "main_color"):
                    mat_color = _normalize_color(tri_mesh.visual.main_color)
                add_mesh_from_faces(
                    mat_faces,
                    mat_color=mat_color,
                    mat_roughness=roughness,
                    mat_metallic=metallic,
                    mesh_vertices=vertices,
                    mesh_normals=normals,
                    mesh_uvs=uvs,
                    mesh_texture=mat_texture,
                )
            continue

        # DAE fallback: use material groups from the source file if trimesh didn't expose them
        if dae_face_materials and len(dae_face_materials) == len(faces):
            face_materials = np.array(dae_face_materials, dtype=object)
            for mat_name in np.unique(face_materials):
                mat_faces = faces[face_materials == mat_name]
                mat_props = dae_material_colors.get(str(mat_name), {})
                mat_color = mat_props.get("color")
                mat_roughness = mat_props.get("roughness")
                mat_metallic = mat_props.get("metallic")
                add_mesh_from_faces(
                    mat_faces,
                    mat_color=mat_color,
                    mat_roughness=mat_roughness,
                    mat_metallic=mat_metallic,
                    mesh_vertices=vertices,
                    mesh_normals=normals,
                    mesh_uvs=uvs,
                    mesh_texture=texture,
                )
            continue

        # Handle per-face color visuals (common for DAE via ColorVisuals)
        face_colors = getattr(tri_mesh.visual, "face_colors", None) if hasattr(tri_mesh, "visual") else None
        if face_colors is not None:
            face_colors = np.array(face_colors, dtype=np.float32)
            if face_colors.shape[0] == faces.shape[0]:
                # Normalize to 0..1 rgb
                if np.max(face_colors) > 1.0:
                    face_colors = face_colors / 255.0
                rgb = face_colors[:, :3]
                # quantize to avoid tiny float differences
                rgb = np.round(rgb, 4)
                unique_colors, inverse = np.unique(rgb, axis=0, return_inverse=True)
                for color_idx, mat_color in enumerate(unique_colors):
                    mat_faces = faces[inverse == color_idx]
                    add_mesh_from_faces(
                        mat_faces,
                        mat_color=(float(mat_color[0]), float(mat_color[1]), float(mat_color[2])),
                        mesh_vertices=vertices,
                        mesh_normals=normals,
                        mesh_uvs=uvs,
                        mesh_texture=texture,
                    )
                continue

        # Handle per-vertex colors by computing face colors
        vertex_colors = getattr(tri_mesh.visual, "vertex_colors", None) if hasattr(tri_mesh, "visual") else None
        if vertex_colors is not None:
            vertex_colors = np.array(vertex_colors, dtype=np.float32)
            if np.max(vertex_colors) > 1.0:
                vertex_colors = vertex_colors / 255.0
            rgb = vertex_colors[:, :3]
            face_rgb = rgb[faces].mean(axis=1)
            face_rgb = np.round(face_rgb, 4)
            unique_colors, inverse = np.unique(face_rgb, axis=0, return_inverse=True)
            for color_idx, mat_color in enumerate(unique_colors):
                mat_faces = faces[inverse == color_idx]
                add_mesh_from_faces(
                    mat_faces,
                    mat_color=(float(mat_color[0]), float(mat_color[1]), float(mat_color[2])),
                    mesh_vertices=vertices,
                    mesh_normals=normals,
                    mesh_uvs=uvs,
                    mesh_texture=texture,
                )
            continue

        # Single-material mesh fallback
        roughness = None
        metallic = None
        if color is None and hasattr(tri_mesh, "visual") and hasattr(tri_mesh.visual, "main_color"):
            color = _normalize_color(tri_mesh.visual.main_color)

        if hasattr(tri_mesh, "visual") and texture is None:
            texture = _extract_trimesh_texture(tri_mesh.visual, base_dir)
            material = getattr(tri_mesh.visual, "material", None)
            roughness, metallic, base_color = _extract_trimesh_material_params(material)
            if color is None and base_color is not None:
                color = base_color

        meshes.append(
            Mesh(
                vertices,
                faces.flatten(),
                normals=normals,
                uvs=uvs,
                maxhullvert=maxhullvert,
                color=color,
                texture=texture,
                roughness=roughness,
                metallic=metallic,
            )
        )

    return meshes


def create_mesh_capsule(
    radius: float,
    half_height: float,
    *,
    up_axis: int = 1,
    segments: int = default_num_segments,
    compute_normals: bool = True,
    compute_uvs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Create capsule geometry data with optional normals and UVs."""
    positions = []
    normals = [] if compute_normals else None
    uvs = [] if compute_uvs else None
    indices = []

    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be between 0 and 2")

    x_dir, y_dir, z_dir = ((1, 2, 0), (0, 1, 2), (2, 0, 1))[up_axis]
    up_vector = np.zeros(3, dtype=np.float32)
    up_vector[up_axis] = half_height

    for i in range(segments + 1):
        theta = i * np.pi / segments
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        for j in range(segments + 1):
            phi = j * 2 * np.pi / segments
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            z = cos_phi * sin_theta
            y = cos_theta
            x = sin_phi * sin_theta

            xyz = np.array((x, y, z), dtype=np.float32)
            normal = xyz[[x_dir, y_dir, z_dir]]
            pos = normal * radius
            if normal[up_axis] >= 0.0:
                pos += up_vector
            else:
                pos -= up_vector

            positions.append(pos.tolist())
            if compute_normals:
                normals.append(normal.tolist())
            if compute_uvs:
                u = cos_theta * 0.5 + 0.5
                v = cos_phi * sin_theta * 0.5 + 0.5
                uvs.append([u, v])

    nv = len(positions)
    for i in range(segments):
        for j in range(segments):
            first = (i * (segments + 1) + j) % nv
            second = (first + segments + 1) % nv
            indices.extend([first, second, (first + 1) % nv, second, (second + 1) % nv, (first + 1) % nv])

    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
        None if normals is None else np.asarray(normals, dtype=np.float32),
        None if uvs is None else np.asarray(uvs, dtype=np.float32),
    )


def create_mesh_cone(
    radius: float,
    half_height: float,
    *,
    up_axis: int = 1,
    segments: int = default_num_segments,
    compute_normals: bool = True,
    compute_uvs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Create cone geometry data with optional normals and UVs."""
    return create_mesh_cylinder(
        radius,
        half_height,
        up_axis=up_axis,
        segments=segments,
        top_radius=0.0,
        compute_normals=compute_normals,
        compute_uvs=compute_uvs,
    )


def create_mesh_cylinder(
    radius: float,
    half_height: float,
    *,
    up_axis: int = 1,
    segments: int = default_num_segments,
    top_radius: float | None = None,
    compute_normals: bool = True,
    compute_uvs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Create cylinder/truncated cone geometry data with optional normals/UVs."""
    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be between 0 and 2")

    x_dir, y_dir, z_dir = ((1, 2, 0), (0, 1, 2), (2, 0, 1))[up_axis]
    if top_radius is None:
        top_radius = radius

    indices = []
    positions = []
    normals = [] if compute_normals else None
    uvs = [] if compute_uvs else None

    def add_vertex(position: np.ndarray, normal: np.ndarray | None, uv: tuple[float, float] | None) -> int:
        idx = len(positions)
        positions.append(position.tolist())
        if compute_normals:
            assert normals is not None
            normals.append([0.0, 0.0, 0.0] if normal is None else normal.tolist())
        if compute_uvs:
            assert uvs is not None
            uvs.append([0.0, 0.0] if uv is None else [uv[0], uv[1]])
        return idx

    side_slope = -np.arctan2(top_radius - radius, 2 * half_height)

    # Side vertices first (contiguous layout for robust indexing).
    side_bottom_indices = []
    for i in range(segments):
        theta = 2 * np.pi * i / segments
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        position = np.array([radius * cos_theta, -half_height, radius * sin_theta], dtype=np.float32)
        position = position[[x_dir, y_dir, z_dir]]

        side_normal = None
        if compute_normals:
            side_normal = np.array([cos_theta, side_slope, sin_theta], dtype=np.float32)
            side_normal = side_normal / np.linalg.norm(side_normal)
            side_normal = side_normal[[x_dir, y_dir, z_dir]]

        side_uv = (i / max(segments - 1, 1), 0.0) if compute_uvs else None
        side_bottom_indices.append(add_vertex(position, side_normal, side_uv))

    side_top_indices = []
    side_apex_index: int | None = None
    if top_radius > 0.0:
        for i in range(segments):
            theta = 2 * np.pi * i / segments
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            position = np.array([top_radius * cos_theta, half_height, top_radius * sin_theta], dtype=np.float32)
            position = position[[x_dir, y_dir, z_dir]]

            side_normal = None
            if compute_normals:
                side_normal = np.array([cos_theta, side_slope, sin_theta], dtype=np.float32)
                side_normal = side_normal / np.linalg.norm(side_normal)
                side_normal = side_normal[[x_dir, y_dir, z_dir]]

            side_uv = (i / max(segments - 1, 1), 1.0) if compute_uvs else None
            side_top_indices.append(add_vertex(position, side_normal, side_uv))
    else:
        apex_position = np.array([0.0, half_height, 0.0], dtype=np.float32)[[x_dir, y_dir, z_dir]]
        apex_normal = None
        if compute_normals:
            apex_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)[[x_dir, y_dir, z_dir]]
        side_apex_index = add_vertex(apex_position, apex_normal, (0.5, 1.0) if compute_uvs else None)

    # Cap vertices after side vertices (also contiguous per cap).
    cap_center_bottom_idx: int | None = None
    cap_center_top_idx: int | None = None

    if radius > 0.0:
        cap_center_bottom_pos = np.array([0.0, -half_height, 0.0], dtype=np.float32)[[x_dir, y_dir, z_dir]]
        cap_center_bottom_n = (
            np.array([0.0, -1.0, 0.0], dtype=np.float32)[[x_dir, y_dir, z_dir]] if compute_normals else None
        )
        cap_center_bottom_idx = add_vertex(
            cap_center_bottom_pos, cap_center_bottom_n, (0.5, 0.5) if compute_uvs else None
        )

    if top_radius > 0.0:
        cap_center_top_pos = np.array([0.0, half_height, 0.0], dtype=np.float32)[[x_dir, y_dir, z_dir]]
        cap_center_top_n = (
            np.array([0.0, 1.0, 0.0], dtype=np.float32)[[x_dir, y_dir, z_dir]] if compute_normals else None
        )
        cap_center_top_idx = add_vertex(cap_center_top_pos, cap_center_top_n, (0.5, 0.5) if compute_uvs else None)

    cap_ring_bottom_indices = []
    if radius > 0.0:
        for i in range(segments):
            theta = 2 * np.pi * i / segments
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            position = np.array([radius * cos_theta, -half_height, radius * sin_theta], dtype=np.float32)
            position = position[[x_dir, y_dir, z_dir]]
            cap_normal = (
                np.array([0.0, -1.0, 0.0], dtype=np.float32)[[x_dir, y_dir, z_dir]] if compute_normals else None
            )
            cap_uv = (cos_theta * 0.5 + 0.5, sin_theta * 0.5 + 0.5) if compute_uvs else None
            cap_ring_bottom_indices.append(add_vertex(position, cap_normal, cap_uv))

    cap_ring_top_indices = []
    if top_radius > 0.0:
        for i in range(segments):
            theta = 2 * np.pi * i / segments
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            position = np.array([top_radius * cos_theta, half_height, top_radius * sin_theta], dtype=np.float32)
            position = position[[x_dir, y_dir, z_dir]]
            cap_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)[[x_dir, y_dir, z_dir]] if compute_normals else None
            cap_uv = (cos_theta * 0.5 + 0.5, sin_theta * 0.5 + 0.5) if compute_uvs else None
            cap_ring_top_indices.append(add_vertex(position, cap_normal, cap_uv))

    # Bottom cap
    if cap_center_bottom_idx is not None and cap_ring_bottom_indices:
        for i in range(segments):
            i0 = cap_ring_bottom_indices[i]
            i1 = cap_ring_bottom_indices[(i + 1) % segments]
            indices.extend([i1, i0, cap_center_bottom_idx])

    # Top cap
    if cap_center_top_idx is not None and cap_ring_top_indices:
        for i in range(segments):
            i0 = cap_ring_top_indices[i]
            i1 = cap_ring_top_indices[(i + 1) % segments]
            indices.extend([cap_center_top_idx, i0, i1])

    # Side faces
    for i in range(segments):
        bottom_i = side_bottom_indices[i]
        bottom_next = side_bottom_indices[(i + 1) % segments]

        if top_radius > 0.0:
            top_i = side_top_indices[i]
            top_next = side_top_indices[(i + 1) % segments]
            indices.extend([top_i, top_next, bottom_i, top_next, bottom_next, bottom_i])
        else:
            assert side_apex_index is not None
            indices.extend([side_apex_index, bottom_next, bottom_i])

    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
        None if normals is None else np.asarray(normals, dtype=np.float32),
        None if uvs is None else np.asarray(uvs, dtype=np.float32),
    )


def create_mesh_arrow(
    base_radius: float,
    base_height: float,
    *,
    cap_radius: float | None = None,
    cap_height: float | None = None,
    up_axis: int = 1,
    segments: int = default_num_segments,
    compute_normals: bool = True,
    compute_uvs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Create arrow geometry data with optional normals and UVs."""
    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be between 0 and 2")
    if cap_radius is None:
        cap_radius = base_radius * 1.8
    if cap_height is None:
        cap_height = base_height * 0.18

    up_vector = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up_vector[up_axis] = 1.0

    base_positions, base_indices, base_normals, base_uvs = create_mesh_cylinder(
        base_radius,
        base_height / 2,
        up_axis=up_axis,
        segments=segments,
        compute_normals=compute_normals,
        compute_uvs=compute_uvs,
    )
    cap_positions, cap_indices, cap_normals, cap_uvs = create_mesh_cone(
        cap_radius,
        cap_height / 2,
        up_axis=up_axis,
        segments=segments,
        compute_normals=compute_normals,
        compute_uvs=compute_uvs,
    )

    base_positions = base_positions.copy()
    cap_positions = cap_positions.copy()
    base_positions[:, :3] += base_height / 2 * up_vector
    cap_positions[:, :3] += (base_height + cap_height / 2 - 1e-3 * base_height) * up_vector

    positions = np.vstack((base_positions, cap_positions))
    indices = np.hstack((base_indices, cap_indices + len(base_positions)))
    normals = None
    uvs = None
    if compute_normals:
        normals = np.vstack((base_normals, cap_normals))
    if compute_uvs:
        uvs = np.vstack((base_uvs, cap_uvs))
    return positions.astype(np.float32), indices.astype(np.uint32), normals, uvs


def create_mesh_box(
    hx: float,
    hy: float,
    hz: float,
    *,
    duplicate_vertices: bool = True,
    compute_normals: bool = True,
    compute_uvs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Create box geometry data with optional duplicated vertices, normals, and UVs."""
    if duplicate_vertices:
        # fmt: off
        positions = np.array(
            [
                [-hx, -hy, -hz], [-hx, -hy,  hz], [-hx,  hy,  hz], [-hx,  hy, -hz],
                [ hx, -hy, -hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [ hx,  hy, -hz],
                [-hx, -hy, -hz], [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx, -hy, -hz],
                [-hx,  hy, -hz], [-hx,  hy,  hz], [ hx,  hy,  hz], [ hx,  hy, -hz],
                [-hx, -hy, -hz], [-hx,  hy, -hz], [ hx,  hy, -hz], [ hx, -hy, -hz],
                [-hx, -hy,  hz], [-hx,  hy,  hz], [ hx,  hy,  hz], [ hx, -hy,  hz],
            ],
            dtype=np.float32,
        )
        indices = np.array(
            [
                 0,  1,  2,  0,  2,  3,   4,  6,  5,  4,  7,  6,
                 8, 10,  9,  8, 11, 10,  12, 13, 14, 12, 14, 15,
                16, 17, 18, 16, 18, 19,  20, 22, 21, 20, 23, 22,
            ],
            dtype=np.uint32,
        )
        # fmt: on
        normals = None
        uvs = None
        if compute_normals:
            normals = np.array(
                [
                    [-1, 0, 0],
                    [-1, 0, 0],
                    [-1, 0, 0],
                    [-1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 0, -1],
                    [0, 0, -1],
                    [0, 0, -1],
                    [0, 0, -1],
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
        if compute_uvs:
            face_uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
            uvs = np.vstack([face_uv] * 6).astype(np.float32)
        return positions, indices, normals, uvs

    positions = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    # fmt: off
    indices = np.array(
        [
            0, 2, 1, 0, 3, 2,  4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,  2, 3, 7, 2, 7, 6,
            0, 4, 7, 0, 7, 3,  1, 2, 6, 1, 6, 5,
        ],
        dtype=np.uint32,
    )
    # fmt: on
    normals = None
    uvs = None
    if compute_normals:
        normals = compute_vertex_normals(positions, indices).astype(np.float32)
    if compute_uvs:
        uvs = np.zeros((len(positions), 2), dtype=np.float32)
    return positions, indices, normals, uvs


def create_mesh_plane(
    width: float,
    length: float,
    *,
    compute_normals: bool = True,
    compute_uvs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Create plane geometry data with optional normals and UVs."""
    half_width = width / 2
    half_length = length / 2
    positions = np.array(
        [
            [-half_width, -half_length, 0.0],
            [half_width, -half_length, 0.0],
            [half_width, half_length, 0.0],
            [-half_width, half_length, 0.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    normals = None
    uvs = None
    if compute_normals:
        normals = np.array([[0.0, 0.0, 1.0]] * 4, dtype=np.float32)
    if compute_uvs:
        uvs = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    return positions, indices, normals, uvs


@wp.kernel
def solidify_mesh_kernel(
    indices: wp.array(dtype=int, ndim=2),
    vertices: wp.array(dtype=wp.vec3, ndim=1),
    thickness: wp.array(dtype=float, ndim=1),
    # outputs
    out_vertices: wp.array(dtype=wp.vec3, ndim=1),
    out_indices: wp.array(dtype=int, ndim=2),
):
    """Extrude each triangle into a triangular prism (wedge) for solidification.

    For each input triangle, creates 6 vertices (3 on each side of the surface)
    and 8 output triangles forming a closed wedge. The extrusion is along the
    face normal, with per-vertex thickness values.

    Launch with dim=num_triangles.

    Args:
        indices: Triangle indices of shape (num_triangles, 3).
        vertices: Vertex positions of shape (num_vertices,).
        thickness: Per-vertex thickness values of shape (num_vertices,).
        out_vertices: Output vertices of shape (num_vertices * 2,). Each input
            vertex produces two output vertices (offset ± thickness along normal).
        out_indices: Output triangle indices of shape (num_triangles * 8, 3).
    """
    tid = wp.tid()
    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]

    vi = vertices[i]
    vj = vertices[j]
    vk = vertices[k]

    normal = wp.normalize(wp.cross(vj - vi, vk - vi))
    ti = normal * thickness[i]
    tj = normal * thickness[j]
    tk = normal * thickness[k]

    # wedge vertices
    vi0 = vi + ti
    vi1 = vi - ti
    vj0 = vj + tj
    vj1 = vj - tj
    vk0 = vk + tk
    vk1 = vk - tk

    i0 = i * 2
    i1 = i * 2 + 1
    j0 = j * 2
    j1 = j * 2 + 1
    k0 = k * 2
    k1 = k * 2 + 1

    out_vertices[i0] = vi0
    out_vertices[i1] = vi1
    out_vertices[j0] = vj0
    out_vertices[j1] = vj1
    out_vertices[k0] = vk0
    out_vertices[k1] = vk1

    oid = tid * 8
    out_indices[oid + 0, 0] = i0
    out_indices[oid + 0, 1] = j0
    out_indices[oid + 0, 2] = k0
    out_indices[oid + 1, 0] = j0
    out_indices[oid + 1, 1] = k1
    out_indices[oid + 1, 2] = k0
    out_indices[oid + 2, 0] = j0
    out_indices[oid + 2, 1] = j1
    out_indices[oid + 2, 2] = k1
    out_indices[oid + 3, 0] = j0
    out_indices[oid + 3, 1] = i1
    out_indices[oid + 3, 2] = j1
    out_indices[oid + 4, 0] = j0
    out_indices[oid + 4, 1] = i0
    out_indices[oid + 4, 2] = i1
    out_indices[oid + 5, 0] = j1
    out_indices[oid + 5, 1] = i1
    out_indices[oid + 5, 2] = k1
    out_indices[oid + 6, 0] = i1
    out_indices[oid + 6, 1] = i0
    out_indices[oid + 6, 2] = k0
    out_indices[oid + 7, 0] = i1
    out_indices[oid + 7, 1] = k0
    out_indices[oid + 7, 2] = k1


def solidify_mesh(
    faces: np.ndarray,
    vertices: np.ndarray,
    thickness: float | list | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a surface mesh into a solid mesh by extruding along face normals.

    Takes a triangle mesh representing a surface and creates a closed solid
    mesh by extruding each triangle into a triangular prism (wedge). Each input
    triangle produces 8 output triangles forming the top, bottom, and sides
    of the prism.

    Args:
        faces: Triangle indices of shape (N, 3), where N is the number of
            triangles.
        vertices: Vertex positions of shape (M, 3), where M is the number of
            vertices.
        thickness: Extrusion distance from the surface. Can be a single float
            (uniform thickness), a list, or an array of shape (M,) for
            per-vertex thickness.

    Returns:
        A tuple containing:
            - faces: Output triangle indices of shape (N * 8, 3).
            - vertices: Output vertex positions of shape (M * 2, 3).
    """
    faces = np.array(faces).reshape(-1, 3)
    out_faces = wp.zeros((len(faces) * 8, 3), dtype=wp.int32)
    out_vertices = wp.zeros(len(vertices) * 2, dtype=wp.vec3)
    if not isinstance(thickness, np.ndarray) and not isinstance(thickness, list):
        thickness = [thickness] * len(vertices)
    wp.launch(
        solidify_mesh_kernel,
        dim=len(faces),
        inputs=[wp.array(faces, dtype=int), wp.array(vertices, dtype=wp.vec3), wp.array(thickness, dtype=float)],
        outputs=[out_vertices, out_faces],
    )
    faces = out_faces.numpy()
    vertices = out_vertices.numpy()
    return faces, vertices
