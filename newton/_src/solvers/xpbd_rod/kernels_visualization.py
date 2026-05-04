# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Visualization kernels for elastic rod material frames."""

import warp as wp


@wp.kernel
def compute_director_lines_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    edge_q: wp.array(dtype=wp.quat),
    num_edges: int,
    axis_length: float,
    # outputs: 3 axes per edge (d1=red, d2=green, d3=blue)
    line_starts: wp.array(dtype=wp.vec3),
    line_ends: wp.array(dtype=wp.vec3),
    line_colors: wp.array(dtype=wp.vec3),
):
    """Compute line segments for visualizing material frames.

    Each edge has 3 axes (d1, d2, d3) drawn from its midpoint.

    Args:
        particle_q: Particle positions.
        edge_q: Edge quaternions.
        num_edges: Number of edges.
        axis_length: Length of director axes to draw.
        line_starts: Output line start positions (num_edges * 3).
        line_ends: Output line end positions (num_edges * 3).
        line_colors: Output line colors (num_edges * 3).
    """
    tid = wp.tid()
    edge_idx = tid // 3
    axis_idx = tid % 3  # 0=d1(red), 1=d2(green), 2=d3(blue)

    if edge_idx >= num_edges:
        return

    # Compute edge midpoint
    p0 = particle_q[edge_idx]
    p1 = particle_q[edge_idx + 1]
    midpoint = (p0 + p1) * 0.5

    q = edge_q[edge_idx]

    # Compute the director based on axis_idx
    if axis_idx == 0:
        # d1 = q * e1 * conj(q) where e1 = (1,0,0)
        x, y, z, w = q[0], q[1], q[2], q[3]
        d1_x = w * w + x * x - y * y - z * z
        d1_y = 2.0 * (x * y + w * z)
        d1_z = 2.0 * (x * z - w * y)
        director = wp.vec3(d1_x, d1_y, d1_z)
        color = wp.vec3(1.0, 0.0, 0.0)  # Red
    elif axis_idx == 1:
        # d2 = q * e2 * conj(q) where e2 = (0,1,0)
        x, y, z, w = q[0], q[1], q[2], q[3]
        d2_x = 2.0 * (x * y - w * z)
        d2_y = w * w - x * x + y * y - z * z
        d2_z = 2.0 * (y * z + w * x)
        director = wp.vec3(d2_x, d2_y, d2_z)
        color = wp.vec3(0.0, 1.0, 0.0)  # Green
    else:
        # d3 = q * e3 * conj(q) where e3 = (0,0,1)
        x, y, z, w = q[0], q[1], q[2], q[3]
        d3_x = 2.0 * (x * z + w * y)
        d3_y = 2.0 * (y * z - w * x)
        d3_z = w * w - x * x - y * y + z * z
        director = wp.vec3(d3_x, d3_y, d3_z)
        color = wp.vec3(0.0, 0.0, 1.0)  # Blue

    line_starts[tid] = midpoint
    line_ends[tid] = midpoint + director * axis_length
    line_colors[tid] = color
