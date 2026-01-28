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

"""Rod meshing using Hermite interpolation and parallel transport frames.

This module generates smooth tube meshes from rod centerline positions,
based on the Unity RodMesher.cs reference implementation.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .kernels import update_rod_mesh_gpu


def _basis_from_vector(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create an orthonormal basis from a direction vector.

    Args:
        w: Normalized direction vector (3,).

    Returns:
        Tuple of (u, v) orthonormal vectors perpendicular to w.
    """
    w = w / (np.linalg.norm(w) + 1e-10)

    if abs(w[0]) > abs(w[1]):
        inv_len = 1.0 / np.sqrt(w[0] * w[0] + w[2] * w[2] + 1e-10)
        u = np.array([-w[2] * inv_len, 0.0, w[0] * inv_len], dtype=np.float32)
    else:
        inv_len = 1.0 / np.sqrt(w[1] * w[1] + w[2] * w[2] + 1e-10)
        u = np.array([0.0, w[2] * inv_len, -w[1] * inv_len], dtype=np.float32)

    v = np.cross(w, u)
    return u, v


def _rotation_matrix(angle: float, axis: np.ndarray) -> np.ndarray:
    """Generate a 4x4 rotation matrix around an axis.

    Based on PBRT rotation matrix formula.

    Args:
        angle: Rotation angle in radians.
        axis: Rotation axis (will be normalized).

    Returns:
        4x4 rotation matrix.
    """
    a = axis / (np.linalg.norm(axis) + 1e-10)
    s = np.sin(angle)
    c = np.cos(angle)

    m = np.eye(4, dtype=np.float32)

    m[0, 0] = a[0] * a[0] + (1.0 - a[0] * a[0]) * c
    m[1, 0] = a[0] * a[1] * (1.0 - c) + a[2] * s
    m[2, 0] = a[0] * a[2] * (1.0 - c) - a[1] * s

    m[0, 1] = a[0] * a[1] * (1.0 - c) - a[2] * s
    m[1, 1] = a[1] * a[1] + (1.0 - a[1] * a[1]) * c
    m[2, 1] = a[1] * a[2] * (1.0 - c) + a[0] * s

    m[0, 2] = a[0] * a[2] * (1.0 - c) + a[1] * s
    m[1, 2] = a[1] * a[2] * (1.0 - c) - a[0] * s
    m[2, 2] = a[2] * a[2] + (1.0 - a[2] * a[2]) * c

    return m


def _hermite_interpolate(p1: np.ndarray, p2: np.ndarray, m1: np.ndarray, m2: np.ndarray, t: float) -> np.ndarray:
    """Hermite interpolation between two points with tangents.

    Args:
        p1: Start point.
        p2: End point.
        m1: Start tangent.
        m2: End tangent.
        t: Interpolation parameter [0, 1].

    Returns:
        Interpolated position.
    """
    t2 = t * t
    t3 = t2 * t

    w1 = 1.0 - 3.0 * t2 + 2.0 * t3
    w2 = t2 * (3.0 - 2.0 * t)
    w3 = t3 - 2.0 * t2 + t
    w4 = t2 * (t - 1.0)

    return p1 * w1 + p2 * w2 + m1 * w3 + m2 * w4


def _hermite_tangent(p1: np.ndarray, p2: np.ndarray, m1: np.ndarray, m2: np.ndarray, t: float) -> np.ndarray:
    """Hermite tangent (first derivative) at interpolation parameter t.

    Args:
        p1: Start point.
        p2: End point.
        m1: Start tangent.
        m2: End tangent.
        t: Interpolation parameter [0, 1].

    Returns:
        Tangent direction (not normalized).
    """
    t2 = t * t

    w1 = 6.0 * t2 - 6.0 * t
    w2 = -6.0 * t2 + 6.0 * t
    w3 = 3.0 * t2 - 4.0 * t + 1.0
    w4 = 3.0 * t2 - 2.0 * t

    return p1 * w1 + p2 * w2 + m1 * w3 + m2 * w4


class RodMesher:
    """Generates tube mesh from rod centerline positions.

    Uses Hermite interpolation for smooth curves and parallel transport
    (Bishop frame) for consistent tube orientation along the rod.

    The mesh topology (triangles, UVs) is static and created once at
    initialization. Only vertex positions and normals are updated
    each frame.

    Attributes:
        num_points: Number of control points (rod particles).
        radius: Base tube radius.
        resolution: Number of vertices around the tube circumference.
        smoothing: Number of interpolation subdivisions between control points.
        num_rings: Total number of vertex rings in the mesh.
        num_vertices: Total number of vertices in the mesh.
        num_triangles: Total number of triangles in the mesh.
        vertices: Vertex positions (num_vertices, 3).
        normals: Vertex normals (num_vertices, 3).
        uvs: UV coordinates (num_vertices, 2).
        indices: Triangle indices (num_triangles * 3,).
    """

    def __init__(
        self,
        num_points: int,
        radius: float = 0.02,
        resolution: int = 8,
        smoothing: int = 3,
        texture_u: float = 1.0,
        texture_v: float = 1.0,
        radii: np.ndarray | None = None,
        device: wp.Device | None = None,
    ):
        """Initialize the rod mesher.

        Args:
            num_points: Number of control points (rod particles).
            radius: Base tube radius.
            resolution: Number of vertices around the tube circumference.
            smoothing: Number of interpolation subdivisions between control points.
            texture_u: UV scale in U direction.
            texture_v: UV scale in V direction.
            radii: Per-vertex radii (optional, for varying radius along rod).
            device: Warp device for GPU arrays.
        """
        self.num_points = num_points
        self.radius = radius
        self.resolution = resolution
        self.smoothing = smoothing
        self.texture_u = texture_u
        self.texture_v = texture_v
        self.device = device or wp.get_device()

        # Compute mesh dimensions
        # Number of rings = (num_points - 1) * smoothing + 1
        # (each segment has `smoothing` subdivisions, but we include endpoints)
        self.num_rings = (num_points - 1) * smoothing + 1 if num_points >= 2 else 0
        self.num_vertices = self.num_rings * resolution if self.num_rings > 0 else 0
        self.num_triangles = (self.num_rings - 1) * resolution * 2 if self.num_rings > 1 else 0

        # Initialize per-vertex radii
        if radii is not None:
            self._radii = np.array(radii, dtype=np.float32)
        else:
            self._radii = np.full(self.num_rings, radius, dtype=np.float32)

        # Allocate arrays
        self.vertices = np.zeros((self.num_vertices, 3), dtype=np.float32)
        self.normals = np.zeros((self.num_vertices, 3), dtype=np.float32)
        self.uvs = np.zeros((self.num_vertices, 2), dtype=np.float32)
        self.indices = np.zeros(self.num_triangles * 3, dtype=np.uint32)

        # Create static topology
        self._create_topology()

        # Warp arrays (lazily allocated)
        self._vertices_wp = None
        self._normals_wp = None
        self._uvs_wp = None
        self._indices_wp = None
        self._positions_wp = None
        self._radii_wp = None
        self._frame_wp = None

    def _create_topology(self) -> None:
        """Create static mesh topology (triangles and UVs).

        Triangle connectivity connects each ring of vertices to the previous
        ring with two triangles per quad (for each pair of adjacent vertices
        in the ring).
        """
        if self.num_rings < 2:
            return

        # UV coordinates
        uv_step_h = self.texture_u / float(self.resolution)
        uv_step_v = self.texture_v / float(self.num_rings - 1)

        for ring_idx in range(self.num_rings):
            for vert_idx in range(self.resolution):
                v_id = ring_idx * self.resolution + vert_idx
                self.uvs[v_id, 0] = vert_idx * uv_step_h
                self.uvs[v_id, 1] = ring_idx * uv_step_v

        # Triangle indices
        tri_idx = 0
        for ring_idx in range(1, self.num_rings):
            start_index = ring_idx * self.resolution

            for vert_idx in range(self.resolution):
                cur_index = start_index + vert_idx
                next_index = start_index + (vert_idx + 1) % self.resolution

                # First triangle
                self.indices[tri_idx * 3 + 0] = cur_index
                self.indices[tri_idx * 3 + 1] = cur_index - self.resolution
                self.indices[tri_idx * 3 + 2] = next_index - self.resolution
                tri_idx += 1

                # Second triangle
                self.indices[tri_idx * 3 + 0] = next_index - self.resolution
                self.indices[tri_idx * 3 + 1] = next_index
                self.indices[tri_idx * 3 + 2] = cur_index
                tri_idx += 1

    def set_radius(self, radius: float) -> None:
        """Set uniform radius for the entire rod.

        Args:
            radius: Uniform radius value.
        """
        self.radius = radius
        self._radii = np.full(self.num_rings, radius, dtype=np.float32)
        # Invalidate GPU array
        self._radii_wp = None

    def set_radii(self, radii: np.ndarray) -> None:
        """Set per-ring radii for varying radius along the rod.

        Args:
            radii: Array of radii, one per ring (num_rings,).
        """
        if len(radii) != self.num_rings:
            # Interpolate to match number of rings
            t = np.linspace(0, 1, self.num_rings)
            t_src = np.linspace(0, 1, len(radii))
            self._radii = np.interp(t, t_src, radii).astype(np.float32)
        else:
            self._radii = np.array(radii, dtype=np.float32)

        # Invalidate GPU array
        self._radii_wp = None

    def update_numpy(
        self,
        positions: np.ndarray,
        radii: np.ndarray | None = None,
    ) -> None:
        """Update mesh vertices and normals using NumPy (CPU).

        Args:
            positions: Control point positions (num_points, 3).
            radii: Optional per-point radii (num_points,). If None, uses
                   the radii set at initialization.
        """
        if self.num_points < 2:
            return

        # Update radii if provided
        if radii is not None:
            # Interpolate per-point radii to per-ring radii
            ring_radii = np.zeros(self.num_rings, dtype=np.float32)
            ring_idx = 0
            for i in range(self.num_points - 1):
                r1 = radii[i]
                r2 = radii[i + 1]
                segments = self.smoothing if i < self.num_points - 2 else self.smoothing + 1
                for s in range(segments):
                    t = s / float(self.smoothing)
                    ring_radii[ring_idx] = r1 * (1.0 - t) + r2 * t
                    ring_idx += 1
            self._radii = ring_radii

        # Initialize frame from first segment direction
        w = positions[1] - positions[0]
        w = w / (np.linalg.norm(w) + 1e-10)
        u, v = _basis_from_vector(w)

        frame = np.eye(4, dtype=np.float32)
        frame[0:3, 0] = u
        frame[0:3, 1] = v
        frame[0:3, 2] = w

        # Process each segment
        v_id = 0
        ring_idx = 0

        for i in range(self.num_points - 1):
            # Hermite control points
            a = max(i - 1, 0)
            b = i
            c = min(i + 1, self.num_points - 1)
            d = min(i + 2, self.num_points - 1)

            p1 = positions[b]
            p2 = positions[c]
            m1 = 0.5 * (positions[c] - positions[a])
            m2 = 0.5 * (positions[d] - positions[b])

            # Handle last segment correctly (include endpoint)
            segments = self.smoothing if i < self.num_points - 2 else self.smoothing + 1

            for s in range(segments):
                t = s / float(self.smoothing)

                # Interpolate position and tangent
                pos = _hermite_interpolate(p1, p2, m1, m2, t)
                tangent = _hermite_tangent(p1, p2, m1, m2, t)
                tangent = tangent / (np.linalg.norm(tangent) + 1e-10)

                # Get current frame direction
                cur_dir = frame[0:3, 2]

                # Compute rotation to align frame with new tangent
                dot_val = np.clip(np.dot(cur_dir, tangent), -1.0, 1.0)
                angle = np.arccos(dot_val)

                if abs(angle) > 0.001:
                    axis = np.cross(cur_dir, tangent)
                    axis_len = np.linalg.norm(axis)
                    if axis_len > 1e-10:
                        axis = axis / axis_len
                        rot = _rotation_matrix(angle, axis)
                        frame = rot @ frame

                # Get radius for this ring
                r = self._radii[ring_idx] if ring_idx < len(self._radii) else self.radius

                # Generate ring vertices
                for cc in range(self.resolution):
                    angle2 = 2.0 * np.pi * cc / float(self.resolution)

                    # Local position on unit circle in frame's XY plane
                    local = np.array([np.cos(angle2), np.sin(angle2), 0.0, 0.0], dtype=np.float32)

                    # Transform to world space
                    world_dir = frame @ local

                    self.vertices[v_id, 0:3] = world_dir[0:3] * r + pos
                    self.normals[v_id, 0:3] = world_dir[0:3]
                    v_id += 1

                ring_idx += 1

    def _ensure_warp_arrays(self) -> None:
        """Allocate Warp GPU arrays if needed."""
        if self._vertices_wp is None:
            self._vertices_wp = wp.zeros(self.num_vertices, dtype=wp.vec3, device=self.device)
        if self._normals_wp is None:
            self._normals_wp = wp.zeros(self.num_vertices, dtype=wp.vec3, device=self.device)
        if self._uvs_wp is None:
            self._uvs_wp = wp.array(self.uvs, dtype=wp.vec2, device=self.device)
        if self._indices_wp is None:
            self._indices_wp = wp.array(self.indices, dtype=wp.uint32, device=self.device)
        if self._radii_wp is None:
            self._radii_wp = wp.array(self._radii, dtype=wp.float32, device=self.device)

    def get_warp_arrays(self) -> tuple[wp.array, wp.array, wp.array, wp.array]:
        """Get Warp GPU arrays for rendering.

        Returns:
            Tuple of (vertices_wp, indices_wp, normals_wp, uvs_wp).
        """
        self._ensure_warp_arrays()

        # Copy current CPU data to GPU
        self._vertices_wp.assign(wp.array(self.vertices, dtype=wp.vec3, device=self.device))
        self._normals_wp.assign(wp.array(self.normals, dtype=wp.vec3, device=self.device))

        return self._vertices_wp, self._indices_wp, self._normals_wp, self._uvs_wp

    def update_warp(
        self,
        positions_wp: wp.array,
        radii_wp: wp.array | None = None,
    ) -> None:
        """Update mesh vertices and normals using Warp GPU kernels.

        This method updates the mesh directly on the GPU without
        host-device transfers.

        Args:
            positions_wp: Control point positions (num_points,) wp.vec3 array.
            radii_wp: Optional per-point radii (num_points,) wp.float32 array.
        """
        self._ensure_warp_arrays()

        if radii_wp is not None:
            self._radii_wp = radii_wp

        # Allocate frame storage if needed (9 floats for 3x3 rotation matrix)
        if self._frame_wp is None:
            self._frame_wp = wp.zeros(9, dtype=wp.float32, device=self.device)

        update_rod_mesh_gpu(
            positions_wp,
            self._radii_wp,
            self._vertices_wp,
            self._normals_wp,
            self._frame_wp,
            self.num_points,
            self.resolution,
            self.smoothing,
            self.radius,
            self.device,
        )


__all__ = ["RodMesher"]
