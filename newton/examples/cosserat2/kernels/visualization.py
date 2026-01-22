# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Visualization kernels for Cosserat rod simulations.

Contains director visualization and rest shape update kernels.
"""

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


@wp.kernel
def update_rest_darboux_kernel(
    rest_bend_d1: float,
    rest_bend_d2: float,
    rest_twist: float,
    num_bend: int,
    # output
    rest_darboux: wp.array(dtype=wp.quat),
):
    """Update rest Darboux vectors to define the rod's rest shape.

    The Darboux vector represents the relative rotation between adjacent frames.
    For small angles, we use: q approx (sin(theta/2)*axis, cos(theta/2)) approx (theta/2*axis, 1) for small theta

    Args:
        rest_bend_d1: Bending rate around d1 axis (curvature in d2-d3 plane) [rad/segment].
        rest_bend_d2: Bending rate around d2 axis (curvature in d1-d3 plane) [rad/segment].
        rest_twist: Twist rate around d3 axis [rad/segment].
        num_bend: Number of bend constraints.
        rest_darboux: Output rest Darboux vectors as quaternions.
    """
    tid = wp.tid()
    if tid >= num_bend:
        return

    # Build quaternion from axis-angle: half-angles for quaternion representation
    half_bend_d1 = rest_bend_d1 * 0.5
    half_bend_d2 = rest_bend_d2 * 0.5
    half_twist = rest_twist * 0.5

    # Use exact formula for robustness
    angle_sq = half_bend_d1 * half_bend_d1 + half_bend_d2 * half_bend_d2 + half_twist * half_twist
    angle = wp.sqrt(angle_sq)

    if angle < 1.0e-8:
        # Near-identity: use limit formula
        rest_darboux[tid] = wp.quat(0.0, 0.0, 0.0, 1.0)
    else:
        # Quaternion from rotation vector (bend_d1, bend_d2, twist)
        # The rotation vector in the material frame corresponds to:
        # x -> d1 axis (bending)
        # y -> d2 axis (bending)
        # z -> d3 axis (twist)
        s = wp.sin(angle) / angle
        c = wp.cos(angle)
        rest_darboux[tid] = wp.quat(s * half_bend_d1, s * half_bend_d2, s * half_twist, c)


@wp.kernel
def update_tip_rest_darboux_kernel(
    tip_rest_bend_d1: float,
    tip_start_idx: int,
    num_bend: int,
    # output
    rest_darboux: wp.array(dtype=wp.quat),
):
    """Update rest Darboux vectors for only the tip (last N particles) of the rod.

    Only modifies bend constraints from tip_start_idx to num_bend-1.
    Only affects bending around the d1 axis.

    Args:
        tip_rest_bend_d1: Bending rate around d1 axis for tip [rad/segment].
        tip_start_idx: Starting index of tip bend constraints.
        num_bend: Total number of bend constraints.
        rest_darboux: Output rest Darboux vectors as quaternions.
    """
    tid = wp.tid()
    constraint_idx = tip_start_idx + tid
    if constraint_idx >= num_bend:
        return

    # Build quaternion from axis-angle for bend around d1 only
    half_bend_d1 = tip_rest_bend_d1 * 0.5
    angle = wp.abs(half_bend_d1)

    if angle < 1.0e-8:
        # Near-identity: use limit formula
        rest_darboux[constraint_idx] = wp.quat(0.0, 0.0, 0.0, 1.0)
    else:
        # Quaternion from rotation vector (bend_d1, 0, 0)
        s = wp.sin(angle) / angle
        c = wp.cos(angle)
        rest_darboux[constraint_idx] = wp.quat(s * half_bend_d1, 0.0, 0.0, c)
