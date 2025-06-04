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

import numpy as np
import warp as wp

from .download_assets import clear_git_cache, download_asset
from .import_mjcf import parse_mjcf
from .import_urdf import parse_urdf
from .import_usd import parse_usd
from .render import SimRenderer, SimRendererOpenGL, SimRendererUsd
from .topology import topological_sort


@wp.func
def transform_inertia(t: wp.transform, I: wp.spatial_matrix):
    """
    Computes adj_t^-T*I*adj_t^-1 (tensor change of coordinates).
    (Frank & Park, section 8.2.3, pg 290)
    """

    t_inv = wp.transform_inverse(t)

    q = wp.transform_get_rotation(t_inv)
    p = wp.transform_get_translation(t_inv)

    r1 = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    r2 = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))
    r3 = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))

    R = wp.matrix_from_cols(r1, r2, r3)
    S = wp.mul(wp.skew(p), R)

    T = wp.spatial_adjoint(R, S)

    return wp.mul(wp.mul(wp.transpose(T), I), T)


@wp.func
def boltzmann(a: float, b: float, alpha: float):
    e1 = wp.exp(alpha * a)
    e2 = wp.exp(alpha * b)
    return (a * e1 + b * e2) / (e1 + e2)


@wp.func
def smooth_max(a: float, b: float, eps: float):
    d = a - b
    return 0.5 * (a + b + wp.sqrt(d * d + eps))


@wp.func
def smooth_min(a: float, b: float, eps: float):
    d = a - b
    return 0.5 * (a + b - wp.sqrt(d * d + eps))


@wp.func
def leaky_max(a: float, b: float):
    return smooth_max(a, b, 1e-5)


@wp.func
def leaky_min(a: float, b: float):
    return smooth_min(a, b, 1e-5)


@wp.func
def vec_min(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.min(a[0], b[0]), wp.min(a[1], b[1]), wp.min(a[2], b[2]))


@wp.func
def vec_max(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.max(a[0], b[0]), wp.max(a[1], b[1]), wp.max(a[2], b[2]))


@wp.func
def vec_leaky_min(a: wp.vec3, b: wp.vec3):
    return wp.vec3(leaky_min(a[0], b[0]), leaky_min(a[1], b[1]), leaky_min(a[2], b[2]))


@wp.func
def vec_leaky_max(a: wp.vec3, b: wp.vec3):
    return wp.vec3(leaky_max(a[0], b[0]), leaky_max(a[1], b[1]), leaky_max(a[2], b[2]))


@wp.func
def vec_abs(a: wp.vec3):
    return wp.vec3(wp.abs(a[0]), wp.abs(a[1]), wp.abs(a[2]))


def load_mesh(filename: str, method: str | None = None):
    """
    Loads a 3D triangular surface mesh from a file.

    Args:
        filename (str): The path to the 3D model file (obj, and other formats supported by the different methods) to load.
        method (str): The method to use for loading the mesh (default None). Can be either `"trimesh"`, `"meshio"`, `"pcu"`, or `"openmesh"`. If None, every method is tried and the first successful mesh import where the number of vertices is greater than 0 is returned.

    Returns:
        Tuple of (mesh_points, mesh_indices), where mesh_points is a Nx3 numpy array of vertex positions (float32),
        and mesh_indices is a Mx3 numpy array of vertex indices (int32) for the triangular faces.
    """
    import os

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    def load_mesh_with_method(method):
        if method == "meshio":
            import meshio

            m = meshio.read(filename)
            mesh_points = np.array(m.points)
            mesh_indices = np.array(m.cells[0].data, dtype=np.int32)
            mesh_uvs = []  # todo
        elif method == "openmesh":
            import openmesh

            m = openmesh.read_trimesh(filename)
            mesh_points = np.array(m.points())
            mesh_indices = np.array(m.face_vertex_indices(), dtype=np.int32)
            mesh_uvs = []  # todo
        elif method == "pcu":
            import point_cloud_utils as pcu

            mesh_points, mesh_indices = pcu.load_mesh_vf(filename)
            mesh_indices = mesh_indices.flatten()
            mesh_uvs = []  # todo
        else:
            import trimesh

            m = trimesh.load(filename)
            if hasattr(m, "geometry"):
                # multiple meshes are contained in a scene; combine to one mesh
                mesh_points = []
                mesh_indices = []
                mesh_uvs = []
                index_offset = 0
                for geom in m.geometry.values():
                    vertices = np.array(geom.vertices, dtype=np.float32)
                    faces = np.array(geom.faces.flatten(), dtype=np.int32)
                    uvs = np.array(geom.uv, dtype=np.float32)
                    mesh_points.append(vertices)
                    mesh_indices.append(faces + index_offset)
                    mesh_uvs.append(uvs)
                    index_offset += len(vertices)
                mesh_points = np.concatenate(mesh_points, axis=0)
                mesh_indices = np.concatenate(mesh_indices)
                mesh_uvs = np.concatenate(mesh_uvs, axis=0)
            else:
                # a single mesh
                mesh_points = np.array(m.vertices, dtype=np.float32)
                mesh_indices = np.array(m.faces.flatten(), dtype=np.int32)
                mesh_uvs = np.array(m.visual.uv, dtype=np.float32)
        return mesh_points, mesh_indices, mesh_uvs

    if method is None:
        methods = ["trimesh", "meshio", "pcu", "openmesh"]
        for method in methods:
            try:
                mesh = load_mesh_with_method(method)
                if mesh is not None and len(mesh[0]) > 0:
                    return mesh
            except Exception:
                pass
        raise ValueError(f"Failed to load mesh using any of the methods: {methods}")
    else:
        mesh = load_mesh_with_method(method)
        if mesh is None or len(mesh[0]) == 0:
            raise ValueError(f"Failed to load mesh using method {method}")
        return mesh


def visualize_meshes(
    meshes: list[tuple[list, list]], num_cols=0, num_rows=0, titles=None, scale_axes=True, show_plot=True
):
    # render meshes in a grid with matplotlib
    import matplotlib.pyplot as plt

    if titles is None:
        titles = []

    num_cols = min(num_cols, len(meshes))
    num_rows = min(num_rows, len(meshes))
    if num_cols and not num_rows:
        num_rows = int(np.ceil(len(meshes) / num_cols))
    elif num_rows and not num_cols:
        num_cols = int(np.ceil(len(meshes) / num_rows))
    else:
        num_cols = len(meshes)
        num_rows = 1

    vertices = [np.array(v).reshape((-1, 3)) for v, _ in meshes]
    faces = [np.array(f, dtype=np.int32).reshape((-1, 3)) for _, f in meshes]
    if scale_axes:
        ranges = np.array([v.max(axis=0) - v.min(axis=0) for v in vertices])
        max_range = ranges.max()
        mid_points = np.array([v.max(axis=0) + v.min(axis=0) for v in vertices]) * 0.5

    fig = plt.figure(figsize=(12, 6))
    for i, (vertices, faces) in enumerate(meshes):
        ax = fig.add_subplot(num_rows, num_cols, i + 1, projection="3d")
        if i < len(titles):
            ax.set_title(titles[i])
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, edgecolor="k")
        if scale_axes:
            mid = mid_points[i]
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    if show_plot:
        plt.show()
    return fig


__all__ = [
    "SimRenderer",
    "SimRendererOpenGL",
    "SimRendererUsd",
    "boltzmann",
    "clear_git_cache",
    "download_asset",
    "leaky_max",
    "leaky_min",
    "parse_mjcf",
    "parse_urdf",
    "parse_usd",
    "smooth_max",
    "smooth_min",
    "topological_sort",
    "vec_abs",
    "vec_leaky_max",
    "vec_leaky_min",
    "vec_max",
    "vec_min",
]
