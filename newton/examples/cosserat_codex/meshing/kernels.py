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

"""Warp GPU kernels for rod mesh generation.

These kernels update mesh vertices and normals from rod centerline
positions using Hermite interpolation and parallel transport frames.
"""

from __future__ import annotations

import warp as wp


@wp.func
def basis_from_vector(w: wp.vec3) -> wp.mat33:
    """Create an orthonormal basis (rotation matrix) from a direction vector.

    Args:
        w: Normalized direction vector.

    Returns:
        3x3 rotation matrix with w as the third column.
    """
    w_norm = wp.normalize(w)

    # Choose perpendicular vector based on which component is smaller
    if wp.abs(w_norm[0]) > wp.abs(w_norm[1]):
        inv_len = 1.0 / wp.sqrt(w_norm[0] * w_norm[0] + w_norm[2] * w_norm[2] + 1e-10)
        u = wp.vec3(-w_norm[2] * inv_len, 0.0, w_norm[0] * inv_len)
    else:
        inv_len = 1.0 / wp.sqrt(w_norm[1] * w_norm[1] + w_norm[2] * w_norm[2] + 1e-10)
        u = wp.vec3(0.0, w_norm[2] * inv_len, -w_norm[1] * inv_len)

    v = wp.cross(w_norm, u)

    # Build rotation matrix (columns are u, v, w)
    return wp.mat33(u[0], v[0], w_norm[0], u[1], v[1], w_norm[1], u[2], v[2], w_norm[2])


@wp.func
def rotation_matrix_axis_angle(angle: float, axis: wp.vec3) -> wp.mat33:
    """Generate a rotation matrix from axis-angle representation.

    Args:
        angle: Rotation angle in radians.
        axis: Rotation axis (will be normalized).

    Returns:
        3x3 rotation matrix.
    """
    a = wp.normalize(axis)
    s = wp.sin(angle)
    c = wp.cos(angle)

    # Rodrigues' rotation formula
    m00 = a[0] * a[0] + (1.0 - a[0] * a[0]) * c
    m10 = a[0] * a[1] * (1.0 - c) + a[2] * s
    m20 = a[0] * a[2] * (1.0 - c) - a[1] * s

    m01 = a[0] * a[1] * (1.0 - c) - a[2] * s
    m11 = a[1] * a[1] + (1.0 - a[1] * a[1]) * c
    m21 = a[1] * a[2] * (1.0 - c) + a[0] * s

    m02 = a[0] * a[2] * (1.0 - c) + a[1] * s
    m12 = a[1] * a[2] * (1.0 - c) - a[0] * s
    m22 = a[2] * a[2] + (1.0 - a[2] * a[2]) * c

    return wp.mat33(m00, m01, m02, m10, m11, m12, m20, m21, m22)


@wp.func
def hermite_interpolate(p1: wp.vec3, p2: wp.vec3, m1: wp.vec3, m2: wp.vec3, t: float) -> wp.vec3:
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


@wp.func
def hermite_tangent(p1: wp.vec3, p2: wp.vec3, m1: wp.vec3, m2: wp.vec3, t: float) -> wp.vec3:
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


@wp.func
def get_frame_column(frame: wp.array(dtype=wp.float32), col: int) -> wp.vec3:
    """Extract a column from the frame stored as flat array."""
    return wp.vec3(frame[col * 3 + 0], frame[col * 3 + 1], frame[col * 3 + 2])


@wp.func
def set_frame_column(frame: wp.array(dtype=wp.float32), col: int, v: wp.vec3):
    """Set a column in the frame stored as flat array."""
    frame[col * 3 + 0] = v[0]
    frame[col * 3 + 1] = v[1]
    frame[col * 3 + 2] = v[2]


@wp.kernel
def _update_rod_mesh_kernel(
    positions: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.float32),
    vertices_out: wp.array(dtype=wp.vec3),
    normals_out: wp.array(dtype=wp.vec3),
    frame: wp.array(dtype=wp.float32),  # 9 floats for 3x3 matrix (column-major)
    num_points: int,
    resolution: int,
    smoothing: int,
    default_radius: float,
):
    """Update rod mesh vertices and normals.

    This kernel must be launched with dim=1 since frame transport is sequential.
    For multiple rods, launch multiple kernels or use batched approach.
    """
    tid = wp.tid()

    if tid != 0:
        return

    if num_points < 2:
        return

    # Initialize frame from first segment direction
    w = wp.normalize(positions[1] - positions[0])
    frame_mat = basis_from_vector(w)

    # Store frame (column-major: col0 = u, col1 = v, col2 = w)
    set_frame_column(frame, 0, wp.vec3(frame_mat[0, 0], frame_mat[1, 0], frame_mat[2, 0]))
    set_frame_column(frame, 1, wp.vec3(frame_mat[0, 1], frame_mat[1, 1], frame_mat[2, 1]))
    set_frame_column(frame, 2, wp.vec3(frame_mat[0, 2], frame_mat[1, 2], frame_mat[2, 2]))

    # Declare dynamic variables for loop indices
    v_id = int(0)
    ring_idx = int(0)
    num_rings = (num_points - 1) * smoothing + 1

    for i in range(num_points - 1):
        # Hermite control point indices
        a = wp.max(i - 1, 0)
        b = i
        c = wp.min(i + 1, num_points - 1)
        d = wp.min(i + 2, num_points - 1)

        p1 = positions[b]
        p2 = positions[c]
        m1 = 0.5 * (positions[c] - positions[a])
        m2 = 0.5 * (positions[d] - positions[b])

        # Handle last segment correctly (include endpoint)
        segments = smoothing
        if i >= num_points - 2:
            segments = smoothing + 1

        for s in range(segments):
            t = float(s) / float(smoothing)

            # Interpolate position and tangent
            pos = hermite_interpolate(p1, p2, m1, m2, t)
            tangent = wp.normalize(hermite_tangent(p1, p2, m1, m2, t))

            # Get current frame direction (column 2)
            cur_dir = get_frame_column(frame, 2)

            # Compute rotation to align frame with new tangent
            dot_val = wp.clamp(wp.dot(cur_dir, tangent), -1.0, 1.0)
            angle = wp.acos(dot_val)

            if wp.abs(angle) > 0.001:
                axis = wp.cross(cur_dir, tangent)
                axis_len = wp.length(axis)
                if axis_len > 1e-10:
                    axis = axis / axis_len
                    rot = rotation_matrix_axis_angle(angle, axis)

                    # Apply rotation to frame columns
                    u_frame = get_frame_column(frame, 0)
                    v_frame = get_frame_column(frame, 1)
                    w_frame = get_frame_column(frame, 2)

                    u_new = rot * u_frame
                    v_new = rot * v_frame
                    w_new = rot * w_frame

                    set_frame_column(frame, 0, u_new)
                    set_frame_column(frame, 1, v_new)
                    set_frame_column(frame, 2, w_new)

            # Get radius for this ring
            r = default_radius
            if ring_idx < num_rings:
                r = radii[ring_idx]

            # Generate ring vertices
            for cc in range(resolution):
                angle2 = 2.0 * 3.14159265359 * float(cc) / float(resolution)

                # Local position on unit circle in frame's XY plane
                cos_a = wp.cos(angle2)
                sin_a = wp.sin(angle2)

                # Get frame axes
                u_axis = get_frame_column(frame, 0)
                v_axis = get_frame_column(frame, 1)

                # World direction
                world_dir = u_axis * cos_a + v_axis * sin_a

                vertices_out[v_id] = world_dir * r + pos
                normals_out[v_id] = world_dir
                v_id = v_id + 1

            ring_idx = ring_idx + 1


@wp.kernel
def _interpolate_radii_kernel(
    point_radii: wp.array(dtype=wp.float32),
    ring_radii: wp.array(dtype=wp.float32),
    num_points: int,
    smoothing: int,
):
    """Interpolate per-point radii to per-ring radii."""
    ring_idx = wp.tid()

    num_rings = (num_points - 1) * smoothing + 1
    if ring_idx >= num_rings:
        return

    # Find which segment this ring belongs to
    segment_idx = ring_idx // smoothing
    local_idx = ring_idx % smoothing

    # Handle edge cases
    if segment_idx >= num_points - 1:
        segment_idx = num_points - 2
        local_idx = smoothing

    t = float(local_idx) / float(smoothing)
    r1 = point_radii[segment_idx]
    r2 = point_radii[segment_idx + 1]

    ring_radii[ring_idx] = r1 * (1.0 - t) + r2 * t


def update_rod_mesh_gpu(
    positions_wp: wp.array,
    radii_wp: wp.array,
    vertices_wp: wp.array,
    normals_wp: wp.array,
    frame_wp: wp.array,
    num_points: int,
    resolution: int,
    smoothing: int,
    default_radius: float,
    device: wp.Device,
) -> None:
    """Update rod mesh vertices and normals on GPU.

    Args:
        positions_wp: Control point positions (num_points,) wp.vec3 array.
        radii_wp: Per-ring radii (num_rings,) wp.float32 array.
        vertices_wp: Output vertex positions (num_vertices,) wp.vec3 array.
        normals_wp: Output vertex normals (num_vertices,) wp.vec3 array.
        frame_wp: Frame storage (9,) wp.float32 array.
        num_points: Number of control points.
        resolution: Number of vertices around circumference.
        smoothing: Number of subdivisions between control points.
        default_radius: Default tube radius.
        device: Warp device.
    """
    wp.launch(
        _update_rod_mesh_kernel,
        dim=1,  # Sequential due to frame transport dependency
        inputs=[
            positions_wp,
            radii_wp,
            vertices_wp,
            normals_wp,
            frame_wp,
            num_points,
            resolution,
            smoothing,
            default_radius,
        ],
        device=device,
    )


__all__ = ["update_rod_mesh_gpu"]
