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

import contextlib
import os
import warnings
from collections import defaultdict
from typing import Literal

import numpy as np
import warp as wp

from ..core.types import Vec3, nparray
from .inertia import compute_mesh_inertia
from .types import (
    SDF,
    GeoType,
    Heightfield,
    Mesh,
)


# Warp kernel for inertia-based OBB computation
@wp.kernel(enable_backward=False)
def compute_obb_candidates(
    vertices: wp.array(dtype=wp.vec3),
    base_quat: wp.quat,
    volumes: wp.array2d(dtype=float),
    transforms: wp.array2d(dtype=wp.transform),
    extents: wp.array2d(dtype=wp.vec3),
):
    """Compute OBB candidates for different rotations around principal axes."""
    angle_idx, axis_idx = wp.tid()
    num_angles_per_axis = volumes.shape[0]

    # Compute rotation angle around one of the principal axes (X=0, Y=1, Z=2)
    angle = float(angle_idx) * (2.0 * wp.pi) / float(num_angles_per_axis)

    # Select the standard basis vector for the current axis
    local_axis = wp.vec3(0.0, 0.0, 0.0)
    local_axis[axis_idx] = 1.0

    # Create incremental rotation around principal axis
    incremental_quat = wp.quat_from_axis_angle(local_axis, angle)

    # Compose rotations: first rotate into principal frame, then apply incremental rotation
    quat = base_quat * incremental_quat

    # Initialize bounds
    min_bounds = wp.vec3(1e10, 1e10, 1e10)
    max_bounds = wp.vec3(-1e10, -1e10, -1e10)

    # Compute bounds for all vertices
    num_verts = vertices.shape[0]
    for i in range(num_verts):
        rotated = wp.quat_rotate(quat, vertices[i])
        min_bounds = wp.min(min_bounds, rotated)
        max_bounds = wp.max(max_bounds, rotated)

    # Compute extents and volume
    box_extents = (max_bounds - min_bounds) * 0.5
    volume = box_extents[0] * box_extents[1] * box_extents[2]

    # Compute center in rotated space and transform back
    center = (max_bounds + min_bounds) * 0.5
    world_center = wp.quat_rotate_inv(quat, center)

    # Store results
    volumes[angle_idx, axis_idx] = volume
    extents[angle_idx, axis_idx] = box_extents
    transforms[angle_idx, axis_idx] = wp.transform(world_center, wp.quat_inverse(quat))


def compute_shape_radius(geo_type: int, scale: Vec3, src: Mesh | SDF | Heightfield | None) -> float:
    """
    Calculates the radius of a sphere that encloses the shape, used for broadphase collision detection.
    """
    if geo_type == GeoType.SPHERE:
        return scale[0]
    elif geo_type == GeoType.BOX:
        return np.linalg.norm(scale)
    elif geo_type == GeoType.CAPSULE or geo_type == GeoType.CYLINDER or geo_type == GeoType.CONE:
        return scale[0] + scale[1]
    elif geo_type == GeoType.ELLIPSOID:
        # Bounding sphere radius is the largest semi-axis
        return max(scale[0], scale[1], scale[2])
    elif geo_type == GeoType.MESH or geo_type == GeoType.CONVEX_MESH:
        vmax = np.max(np.abs(src.vertices), axis=0) * np.max(scale)
        return np.linalg.norm(vmax)
    elif geo_type == GeoType.PLANE:
        if scale[0] > 0.0 and scale[1] > 0.0:
            # finite plane
            return np.linalg.norm(scale)
        else:
            return 1.0e6
    elif geo_type == GeoType.HFIELD:
        # Heightfield bounding sphere from half-extents
        if src is not None:
            half_x = src.hx * scale[0]
            half_y = src.hy * scale[1]
            # Vertical range: from min_z to max_z, centered at midpoint
            half_z = (src.max_z - src.min_z) / 2.0 * scale[2]
            return np.sqrt(half_x**2 + half_y**2 + half_z**2)
        else:
            return np.linalg.norm(scale)
    else:
        return 10.0


def compute_aabb(vertices: nparray) -> tuple[Vec3, Vec3]:
    """Compute the axis-aligned bounding box of a set of vertices."""
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    return min_coords, max_coords


def compute_pca_obb(vertices: nparray) -> tuple[wp.transform, wp.vec3]:
    """Compute the oriented bounding box of a set of vertices.

    Args:
        vertices: A numpy array of shape (N, 3) containing the vertex positions.

    Returns:
        A tuple containing:
        - transform: The transform of the oriented bounding box
        - extents: The half-extents of the box along its principal axes
    """
    if len(vertices) == 0:
        return wp.transform_identity(), wp.vec3(0.0, 0.0, 0.0)
    if len(vertices) == 1:
        return wp.transform(wp.vec3(vertices[0]), wp.quat_identity()), wp.vec3(0.0, 0.0, 0.0)

    # Center the vertices
    center = np.mean(vertices, axis=0)
    centered_vertices = vertices - center

    # Compute covariance matrix with handling for degenerate cases
    if len(vertices) < 3:
        # For 2 points, create a line-aligned OBB
        direction = centered_vertices[1] if len(vertices) > 1 else np.array([1, 0, 0])
        direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 1e-6 else np.array([1, 0, 0])
        # Create orthogonal basis
        if abs(direction[0]) < 0.9:
            perpendicular = np.cross(direction, [1, 0, 0])
        else:
            perpendicular = np.cross(direction, [0, 1, 0])
        perpendicular = perpendicular / np.linalg.norm(perpendicular)
        third = np.cross(direction, perpendicular)
        eigenvectors = np.column_stack([direction, perpendicular, third])
    else:
        cov_matrix = np.cov(centered_vertices.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # Sort by eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Ensure right-handed coordinate system
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] *= -1

    # Project vertices onto principal axes
    projected = centered_vertices @ eigenvectors

    # Compute extents
    min_coords = np.min(projected, axis=0)
    max_coords = np.max(projected, axis=0)
    extents = (max_coords - min_coords) / 2.0

    # Calculate the center in the projected coordinate system
    center_offset = (max_coords + min_coords) / 2.0
    # Transform the center offset back to the original coordinate system
    center = center + center_offset @ eigenvectors.T

    # Convert rotation matrix to quaternion
    # The rotation matrix should transform from the original coordinate system to the principal axes
    # eigenvectors is the rotation matrix from original to principal axes
    rotation_matrix = eigenvectors

    # Convert to quaternion using Warp's quat_from_matrix function
    # First convert numpy array to Warp matrix
    orientation = wp.quat_from_matrix(wp.mat33(rotation_matrix))

    return wp.transform(wp.vec3(center), orientation), wp.vec3(extents)


def compute_inertia_obb(
    vertices: nparray,
    num_angle_steps: int = 360,
) -> tuple[wp.transform, wp.vec3]:
    """
    Compute oriented bounding box using inertia-based principal axes.

    This method provides more stable results than PCA for symmetric objects:
    1. Computes convex hull of the input vertices
    2. Computes inertia tensor of the hull and extracts principal axes
    3. Uses Warp kernels to test rotations around each principal axis
    4. Returns the OBB with minimum volume

    Args:
        vertices: Array of shape (N, 3) containing the vertex positions
        num_angle_steps: Number of angle steps to test per axis (default: 360)

    Returns:
        Tuple of (transform, extents)
    """
    if len(vertices) == 0:
        return wp.transform_identity(), wp.vec3(0.0, 0.0, 0.0)

    if len(vertices) == 1:
        return wp.transform(wp.vec3(vertices[0]), wp.quat_identity()), wp.vec3(0.0, 0.0, 0.0)

    # Step 1: Compute convex hull
    hull_vertices, hull_faces = remesh_convex_hull(vertices, maxhullvert=0)  # 0 = no limit
    hull_indices = hull_faces.flatten()

    # Step 2: Compute mesh inertia
    _mass, com, inertia_tensor, _volume = compute_mesh_inertia(
        density=1.0,  # Unit density
        vertices=hull_vertices.tolist(),
        indices=hull_indices.tolist(),
        is_solid=True,
    )

    # Adjust vertices to be centered at COM
    center = np.array(com)
    centered_vertices = hull_vertices - center

    # Convert inertia tensor to numpy array for diagonalization
    inertia = np.array(inertia_tensor).reshape(3, 3)

    # Get principal axes by diagonalizing inertia tensor
    eigenvalues, eigenvectors = np.linalg.eigh(inertia)

    # Sort by eigenvalues in ascending order (largest inertia = smallest dimension)
    # This helps with consistent ordering
    sorted_indices = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, sorted_indices]

    # Ensure no reflection in the transformation
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1

    principal_axes = eigenvectors

    # Convert principal axes rotation matrix to quaternion
    # The principal_axes matrix transforms from world to principal frame
    base_quat = wp.quat_from_matrix(wp.mat33(principal_axes.T.flatten()))

    # Step 3: Warp kernel search
    # Allocate 2D arrays: (num_angle_steps, 3 axes)
    vertices_wp = wp.array(centered_vertices, dtype=wp.vec3)
    volumes = wp.zeros((num_angle_steps, 3), dtype=float)
    transforms = wp.zeros((num_angle_steps, 3), dtype=wp.transform)
    extents = wp.zeros((num_angle_steps, 3), dtype=wp.vec3)

    # Launch kernel with 2D dimensions
    wp.launch(
        compute_obb_candidates,
        dim=(num_angle_steps, 3),
        inputs=[vertices_wp, base_quat, volumes, transforms, extents],
    )

    # Find minimum volume
    volumes_host = volumes.numpy()
    best_idx = np.unravel_index(np.argmin(volumes_host), volumes_host.shape)

    # Get results
    best_transform = transforms.numpy()[best_idx]
    best_extents = extents.numpy()[best_idx]

    # Adjust transform to account for original center
    best_transform[0:3] += center

    return wp.transform(*best_transform), wp.vec3(*best_extents)


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
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    def load_mesh_with_method(method):
        if method == "meshio":
            import meshio  # noqa: PLC0415

            m = meshio.read(filename)
            mesh_points = np.array(m.points)
            mesh_indices = np.array(m.cells[0].data, dtype=np.int32)
        elif method == "openmesh":
            import openmesh  # noqa: PLC0415

            m = openmesh.read_trimesh(filename)
            mesh_points = np.array(m.points())
            mesh_indices = np.array(m.face_vertex_indices(), dtype=np.int32)
        elif method == "pcu":
            import point_cloud_utils as pcu  # noqa: PLC0415

            mesh_points, mesh_indices = pcu.load_mesh_vf(filename)
            mesh_indices = mesh_indices.flatten()
        else:
            import trimesh  # noqa: PLC0415

            m = trimesh.load(filename)
            if hasattr(m, "geometry"):
                # multiple meshes are contained in a scene; combine to one mesh
                mesh_points = []
                mesh_indices = []
                index_offset = 0
                for geom in m.geometry.values():
                    vertices = np.array(geom.vertices, dtype=np.float32)
                    faces = np.array(geom.faces.flatten(), dtype=np.int32)
                    mesh_points.append(vertices)
                    mesh_indices.append(faces + index_offset)
                    index_offset += len(vertices)
                mesh_points = np.concatenate(mesh_points, axis=0)
                mesh_indices = np.concatenate(mesh_indices)
            else:
                # a single mesh
                mesh_points = np.array(m.vertices, dtype=np.float32)
                mesh_indices = np.array(m.faces.flatten(), dtype=np.int32)
        return mesh_points, mesh_indices

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
    """Render meshes in a grid with matplotlib."""

    import matplotlib.pyplot as plt  # noqa: PLC0415

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


@contextlib.contextmanager
def silence_stdio():
    """
    Redirect *both* Python-level and C-level stdout/stderr to os.devnull
    for the duration of the with-block.
    """
    devnull = open(os.devnull, "w")
    # Duplicate the real fds so we can restore them later
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)

    try:
        # Point fds 1 and 2 at /dev/null
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)

        # Also patch the Python objects that wrap those fds
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
    finally:
        # Restore original fds
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        devnull.close()


def remesh_ftetwild(vertices, faces, optimize=False, edge_length_fac=0.05, verbose=False):
    """Remesh a 3D triangular surface mesh using "Fast Tetrahedral Meshing in the Wild" (fTetWild).

    This is useful for improving the quality of the mesh, and for ensuring that the mesh is
    watertight. This function first tetrahedralizes the mesh, then extracts the surface mesh.
    The resulting mesh is guaranteed to be watertight and may have a different topology than the
    input mesh.

    Uses pytetwild, a Python wrapper for fTetWild, to perform the remeshing.
    See https://github.com/pyvista/pytetwild.

    Args:
        vertices: A numpy array of shape (N, 3) containing the vertex positions.
        faces: A numpy array of shape (M, 3) containing the vertex indices of the faces.
        optimize: Whether to optimize the mesh quality during remeshing.
        edge_length_fac: The target edge length of the tetrahedral element as a fraction of the bounding box diagonal.

    Returns:
        A tuple (vertices, faces) containing the remeshed mesh. Returns the original vertices and faces
        if the remeshing fails.
    """

    from pytetwild import tetrahedralize  # noqa: PLC0415

    def tet_fn(v, f):
        return tetrahedralize(v, f, optimize=optimize, edge_length_fac=edge_length_fac)

    if verbose:
        tet_vertices, tet_indices = tet_fn(vertices, faces)
    else:
        # Suppress stdout and stderr during tetrahedralize
        with silence_stdio():
            tet_vertices, tet_indices = tet_fn(vertices, faces)

    def face_indices(tet):
        face1 = (tet[0], tet[2], tet[1])
        face2 = (tet[1], tet[2], tet[3])
        face3 = (tet[0], tet[1], tet[3])
        face4 = (tet[0], tet[3], tet[2])
        return (
            (face1, tuple(sorted(face1))),
            (face2, tuple(sorted(face2))),
            (face3, tuple(sorted(face3))),
            (face4, tuple(sorted(face4))),
        )

    # determine surface faces
    elements_per_face = defaultdict(set)
    unique_faces = {}
    for e, tet in enumerate(tet_indices):
        for face, key in face_indices(tet):
            elements_per_face[key].add(e)
            unique_faces[key] = face
    surface_faces = [face for key, face in unique_faces.items() if len(elements_per_face[key]) == 1]

    new_vertices = np.array(tet_vertices)
    new_faces = np.array(surface_faces, dtype=np.int32)

    if len(new_vertices) == 0 or len(new_faces) == 0:
        warnings.warn(
            "Remeshing failed, the optimized mesh has no vertices or faces; return previous mesh.", stacklevel=2
        )
        return vertices, faces

    return new_vertices, new_faces


def remesh_alphashape(vertices, alpha: float = 3.0):
    """Remesh a 3D triangular surface mesh using the alpha shape algorithm.

    Args:
        vertices: A numpy array of shape (N, 3) containing the vertex positions.
        faces: A numpy array of shape (M, 3) containing the vertex indices of the faces (not needed).
        alpha: The alpha shape parameter.

    Returns:
        A tuple (vertices, faces) containing the remeshed mesh.
    """
    import alphashape  # noqa: PLC0415

    with silence_stdio():
        alpha_shape = alphashape.alphashape(vertices, alpha)
    return np.array(alpha_shape.vertices), np.array(alpha_shape.faces, dtype=np.int32)


def remesh_quadratic(vertices, faces, target_reduction=0.5, target_count=None, **kwargs):
    """Remesh a 3D triangular surface mesh using fast quadratic mesh simplification.

    https://github.com/pyvista/fast-simplification

    Args:
        vertices: A numpy array of shape (N, 3) containing the vertex positions.
        faces: A numpy array of shape (M, 3) containing the vertex indices of the faces.
        target_reduction: The target reduction factor for the number of faces (0.0 to 1.0).
        **kwargs: Additional keyword arguments for the remeshing algorithm.

    Returns:
        A tuple (vertices, faces) containing the remeshed mesh.
    """
    from fast_simplification import simplify  # noqa: PLC0415

    return simplify(vertices, faces, target_reduction=target_reduction, target_count=target_count, **kwargs)


def remesh_convex_hull(vertices, maxhullvert: int = 0):
    """Compute the convex hull of a set of 3D points and return the vertices and faces of the convex hull mesh.

    Uses ``scipy.spatial.ConvexHull`` to compute the convex hull.

    Args:
        vertices: A numpy array of shape (N, 3) containing the vertex positions.
        maxhullvert: The maximum number of vertices for the convex hull. If 0, no limit is applied.

    Returns:
        A tuple (verts, faces) where:
        - verts: A numpy array of shape (M, 3) containing the vertex positions of the convex hull.
        - faces: A numpy array of shape (K, 3) containing the vertex indices of the triangular faces of the convex hull.
    """

    from scipy.spatial import ConvexHull  # noqa: PLC0415

    qhull_options = "Qt"
    if maxhullvert > 0:
        # qhull "TA" actually means "number of vertices added after the initial simplex"
        # from mujoco's user_mesh.cc
        qhull_options += f" TA{maxhullvert - 4}"
    hull = ConvexHull(vertices, qhull_options=qhull_options)
    verts = hull.points.copy().astype(np.float32)
    faces = hull.simplices.astype(np.int32)

    # fix winding order of faces
    centre = verts.mean(0)
    for i, tri in enumerate(faces):
        a, b, c = verts[tri]
        normal = np.cross(b - a, c - a)
        if np.dot(normal, a - centre) < 0:
            faces[i] = tri[[0, 2, 1]]

    # trim vertices to only those that are used in the faces
    unique_verts = np.unique(faces.flatten())
    verts = verts[unique_verts]
    # update face indices to use the new vertex indices
    mapping = {v: i for i, v in enumerate(unique_verts)}
    faces = np.array([mapping[v] for v in faces.flatten()], dtype=np.int32).reshape(faces.shape)

    return verts, faces


RemeshingMethod = Literal["ftetwild", "alphashape", "quadratic", "convex_hull", "poisson"]


def remesh(
    vertices, faces, method: RemeshingMethod = "quadratic", visualize=False, **remeshing_kwargs
) -> tuple[nparray, nparray]:
    """
    Remeshes a 3D triangular surface mesh using the specified method.

    Args:
        vertices: A numpy array of shape (N, 3) containing the vertex positions.
        faces: A numpy array of shape (M, 3) containing the vertex indices of the faces.
        method: The remeshing method to use. One of "ftetwild", "quadratic", "convex_hull",
            "alphashape", or "poisson".
        visualize: Whether to render the input and output meshes using matplotlib.
        **remeshing_kwargs: Additional keyword arguments passed to the remeshing function.

    Returns:
        A tuple (vertices, faces) containing the remeshed mesh.
    """
    if method == "ftetwild":
        new_vertices, new_faces = remesh_ftetwild(vertices, faces, **remeshing_kwargs)
    elif method == "alphashape":
        new_vertices, new_faces = remesh_alphashape(vertices, **remeshing_kwargs)
    elif method == "quadratic":
        new_vertices, new_faces = remesh_quadratic(vertices, faces, **remeshing_kwargs)
    elif method == "convex_hull":
        new_vertices, new_faces = remesh_convex_hull(vertices, **remeshing_kwargs)
    elif method == "poisson":
        from newton._src.geometry.remesh import remesh_poisson  # noqa: PLC0415

        new_vertices, new_faces = remesh_poisson(vertices, faces, **remeshing_kwargs)
    else:
        raise ValueError(f"Unknown remeshing method: {method}")

    if visualize:
        # side-by-side visualization of the input and output meshes
        visualize_meshes(
            [(vertices, faces), (new_vertices, new_faces)],
            titles=[
                f"Original ({len(vertices)} verts, {len(faces)} faces)",
                f"Remeshed ({len(new_vertices)} verts, {len(new_faces)} faces)",
            ],
        )
    return new_vertices, new_faces


def remesh_mesh(
    mesh: Mesh,
    method: RemeshingMethod = "quadratic",
    recompute_inertia: bool = False,
    inplace: bool = False,
    **remeshing_kwargs,
) -> Mesh:
    """
    Remeshes a Mesh object using the specified remeshing method.

    Args:
        mesh (Mesh): The mesh to be remeshed.
        method (RemeshingMethod, optional): The remeshing method to use.
            One of "ftetwild", "quadratic", "convex_hull", "alphashape", or "poisson".
            Defaults to "quadratic".
        recompute_inertia (bool, optional): If True, recompute the mass, center of mass,
            and inertia tensor of the mesh after remeshing. Defaults to False.
        inplace (bool, optional): If True, modify the mesh in place. If False,
            return a new mesh instance with the remeshed geometry. Defaults to False.
        **remeshing_kwargs: Additional keyword arguments passed to the remeshing function.

    Returns:
        Mesh: The remeshed mesh. If `inplace` is True, returns the modified input mesh.
    """
    if method == "convex_hull":
        remeshing_kwargs["maxhullvert"] = mesh.maxhullvert
    vertices, indices = remesh(mesh.vertices, mesh.indices.reshape(-1, 3), method=method, **remeshing_kwargs)
    if inplace:
        mesh.vertices = vertices
        mesh.indices = indices.flatten()
        if recompute_inertia:
            mesh.mass, mesh.com, mesh.I, _ = compute_mesh_inertia(1.0, vertices, indices, is_solid=mesh.is_solid)
    else:
        return mesh.copy(vertices=vertices, indices=indices, recompute_inertia=recompute_inertia)
    return mesh


def create_box_mesh(half_extents: Vec3, duplicate_vertices: bool = False) -> tuple[nparray, nparray]:
    """Create a box mesh with the given half extents.

    Args:
        half_extents: Half extents of the box along each axis (x, y, z).
        duplicate_vertices: If True, duplicate vertices for each face to enable proper flat shading.
            Each face will have its own 4 vertices (24 total), allowing distinct normals per face.
            If False (default), use 8 shared corner vertices for a more compact representation.

    Returns:
        A tuple of (vertices, indices) numpy arrays.
    """
    x_extent, y_extent, z_extent = half_extents

    if duplicate_vertices:
        # 24 vertices (4 per face) for proper flat shading
        vertices = np.array(
            [
                # Bottom face (z = -z_extent) - vertices 0-3
                [-x_extent, -y_extent, -z_extent],
                [x_extent, -y_extent, -z_extent],
                [x_extent, y_extent, -z_extent],
                [-x_extent, y_extent, -z_extent],
                # Top face (z = z_extent) - vertices 4-7
                [-x_extent, -y_extent, z_extent],
                [x_extent, -y_extent, z_extent],
                [x_extent, y_extent, z_extent],
                [-x_extent, y_extent, z_extent],
                # Front face (y = -y_extent) - vertices 8-11
                [-x_extent, -y_extent, -z_extent],
                [x_extent, -y_extent, -z_extent],
                [x_extent, -y_extent, z_extent],
                [-x_extent, -y_extent, z_extent],
                # Back face (y = y_extent) - vertices 12-15
                [x_extent, y_extent, -z_extent],
                [-x_extent, y_extent, -z_extent],
                [-x_extent, y_extent, z_extent],
                [x_extent, y_extent, z_extent],
                # Left face (x = -x_extent) - vertices 16-19
                [-x_extent, -y_extent, -z_extent],
                [-x_extent, -y_extent, z_extent],
                [-x_extent, y_extent, z_extent],
                [-x_extent, y_extent, -z_extent],
                # Right face (x = x_extent) - vertices 20-23
                [x_extent, -y_extent, -z_extent],
                [x_extent, y_extent, -z_extent],
                [x_extent, y_extent, z_extent],
                [x_extent, -y_extent, z_extent],
            ],
            dtype=np.float32,
        )
        indices = np.array(
            [
                # Bottom face
                0,
                2,
                1,
                0,
                3,
                2,
                # Top face
                4,
                5,
                6,
                4,
                6,
                7,
                # Front face
                8,
                9,
                10,
                8,
                10,
                11,
                # Back face
                12,
                13,
                14,
                12,
                14,
                15,
                # Left face
                16,
                17,
                18,
                16,
                18,
                19,
                # Right face
                20,
                21,
                22,
                20,
                22,
                23,
            ],
            dtype=np.int32,
        )
    else:
        # 8 shared corner vertices (original compact representation)
        vertices = np.array(
            [
                [-x_extent, -y_extent, -z_extent],
                [x_extent, -y_extent, -z_extent],
                [x_extent, y_extent, -z_extent],
                [-x_extent, y_extent, -z_extent],
                [-x_extent, -y_extent, z_extent],
                [x_extent, -y_extent, z_extent],
                [x_extent, y_extent, z_extent],
                [-x_extent, y_extent, z_extent],
            ],
            dtype=np.float32,
        )
        indices = np.array(
            [
                # Bottom face (z = -z_extent)
                0,
                2,
                1,
                0,
                3,
                2,
                # Top face (z = z_extent)
                4,
                5,
                6,
                4,
                6,
                7,
                # Front face (y = -y_extent)
                0,
                1,
                5,
                0,
                5,
                4,
                # Back face (y = y_extent)
                2,
                3,
                7,
                2,
                7,
                6,
                # Left face (x = -x_extent)
                0,
                4,
                7,
                0,
                7,
                3,
                # Right face (x = x_extent)
                1,
                2,
                6,
                1,
                6,
                5,
            ],
            dtype=np.int32,
        )
    return vertices, indices


def transform_points(points: nparray, transform: wp.transform, scale: Vec3 | None = None) -> nparray:
    if scale is not None:
        points = points * np.array(scale, dtype=np.float32)
    return points @ np.array(wp.quat_to_matrix(transform.q)).reshape(3, 3) + transform.p


@wp.kernel(enable_backward=False)
def get_total_kernel(
    counts: wp.array(dtype=int),
    prefix_sums: wp.array(dtype=int),
    num_elements: wp.array(dtype=int),
    max_elements: int,
    total: wp.array(dtype=int),
):
    """
    Get the total of an array of counts and prefix sums.
    """
    if num_elements[0] <= 0 or max_elements <= 0:
        total[0] = 0
        return

    # Clip to array bounds to avoid out-of-bounds access
    n = wp.min(num_elements[0], max_elements)
    final_idx = n - 1
    total[0] = prefix_sums[final_idx] + counts[final_idx]


def scan_with_total(
    counts: wp.array(dtype=int),
    prefix_sums: wp.array(dtype=int),
    num_elements: wp.array(dtype=int),
    total: wp.array(dtype=int),
):
    """
    Computes an exclusive prefix sum and total of a counts array.

    Args:
        counts: Input array of per-element counts.
        prefix_sums: Output array for exclusive prefix sums (same size as counts).
        num_elements: Single-element array containing the number of valid elements in counts.
        total: Single-element output array that will contain the sum of all counts.
    """
    wp.utils.array_scan(counts, prefix_sums, inclusive=False)
    wp.launch(get_total_kernel, dim=[1], inputs=[counts, prefix_sums, num_elements, counts.shape[0], total])


__all__ = ["compute_shape_radius", "load_mesh", "visualize_meshes"]
