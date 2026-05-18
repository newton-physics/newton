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

"""Warp kernels for constraint computation and Jacobian assembly.

This module contains kernels for computing stretch and bend constraints,
their Jacobians, and building the RHS for the linear system.
"""

from __future__ import annotations

import warp as wp

from .kernels_math import (
    _warp_jacobian_index,
    _warp_quat_conjugate,
    _warp_quat_mul,
    _warp_quat_rotate_vector,
)


@wp.kernel
def _warp_prepare_compliance(
    rest_lengths: wp.array(dtype=wp.float32),
    bend_stiffness: wp.array(dtype=wp.vec3),
    young_modulus: float,
    torsion_modulus: float,
    dt: float,
    compliance: wp.array(dtype=wp.float32),
):
    """Compute compliance values for XPBD constraints.

    Computes compliance = 1/(stiffness * dt^2) for each constraint DOF.

    Args:
        rest_lengths: Rest length for each edge.
        bend_stiffness: Bend/twist stiffness coefficients per edge.
        young_modulus: Young's modulus for stretch.
        torsion_modulus: Torsion modulus for twist.
        dt: Time step size.
        compliance: Output compliance values (6 per edge).
    """
    i = wp.tid()
    L = rest_lengths[i]
    dt2 = dt * dt
    eps = 1.0e-10

    k_bend1_ref = young_modulus * bend_stiffness[i].x
    k_bend2_ref = young_modulus * bend_stiffness[i].y
    k_twist_ref = torsion_modulus * bend_stiffness[i].z

    k_bend1_eff = k_bend1_ref * L
    k_bend2_eff = k_bend2_ref * L
    k_twist_eff = k_twist_ref * L

    stretch_compliance = 1.0e-10

    base = i * 6
    compliance[base + 0] = stretch_compliance
    compliance[base + 1] = stretch_compliance
    compliance[base + 2] = stretch_compliance
    compliance[base + 3] = 1.0 / (k_bend1_eff * dt2 + eps)
    compliance[base + 4] = 1.0 / (k_bend2_eff * dt2 + eps)
    compliance[base + 5] = 1.0 / (k_twist_eff * dt2 + eps)


@wp.kernel
def _warp_update_constraints_direct(
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quat),
    rest_lengths: wp.array(dtype=wp.float32),
    rest_darboux: wp.array(dtype=wp.vec3),
    constraint_values: wp.array(dtype=wp.float32),
):
    """Compute constraint values for direct solve.

    Computes stretch constraint (3 DOF) and Darboux constraint (3 DOF) per edge.

    Args:
        positions: Particle positions.
        orientations: Particle orientations.
        rest_lengths: Rest length for each edge.
        rest_darboux: Rest Darboux vector for each edge.
        constraint_values: Output constraint values (6 per edge).
    """
    i = wp.tid()
    p0 = positions[i]
    p1 = positions[i + 1]
    q0 = orientations[i]
    q1 = orientations[i + 1]

    half_L = 0.5 * rest_lengths[i]
    r0_world = _warp_quat_rotate_vector(q0, wp.vec3(0.0, 0.0, half_L))
    r1_world = _warp_quat_rotate_vector(q1, wp.vec3(0.0, 0.0, -half_L))

    c0 = p0 + r0_world
    c1 = p1 + r1_world
    stretch_error = c0 - c1

    q_rel = _warp_quat_mul(_warp_quat_conjugate(q0), q1)
    omega = wp.vec3(q_rel.x, q_rel.y, q_rel.z)
    darboux_error = omega - rest_darboux[i]

    base = i * 6
    constraint_values[base + 0] = stretch_error.x
    constraint_values[base + 1] = stretch_error.y
    constraint_values[base + 2] = stretch_error.z
    constraint_values[base + 3] = darboux_error.x
    constraint_values[base + 4] = darboux_error.y
    constraint_values[base + 5] = darboux_error.z


@wp.kernel
def _warp_compute_jacobians_direct(
    orientations: wp.array(dtype=wp.quat),
    rest_lengths: wp.array(dtype=wp.float32),
    jacobian_pos: wp.array(dtype=wp.float32),
    jacobian_rot: wp.array(dtype=wp.float32),
):
    """Compute Jacobians for direct solve.

    Computes position and rotation Jacobians for stretch and Darboux constraints.

    Args:
        orientations: Particle orientations.
        rest_lengths: Rest length for each edge.
        jacobian_pos: Output position Jacobian (36 floats per edge).
        jacobian_rot: Output rotation Jacobian (36 floats per edge).
    """
    i = wp.tid()
    q0 = orientations[i]
    q1 = orientations[i + 1]

    half_L = 0.5 * rest_lengths[i]
    r0 = _warp_quat_rotate_vector(q0, wp.vec3(0.0, 0.0, half_L))
    r1 = _warp_quat_rotate_vector(q1, wp.vec3(0.0, 0.0, -half_L))

    # Identity for J_pos
    for d in range(3):
        jacobian_pos[_warp_jacobian_index(i, d, d)] = 1.0
        jacobian_pos[_warp_jacobian_index(i, d, d + 3)] = -1.0

    # -skew(r0) and skew(r1) for stretch rotation Jacobians
    jacobian_rot[_warp_jacobian_index(i, 0, 0)] = 0.0
    jacobian_rot[_warp_jacobian_index(i, 0, 1)] = r0.z
    jacobian_rot[_warp_jacobian_index(i, 0, 2)] = -r0.y

    jacobian_rot[_warp_jacobian_index(i, 1, 0)] = -r0.z
    jacobian_rot[_warp_jacobian_index(i, 1, 1)] = 0.0
    jacobian_rot[_warp_jacobian_index(i, 1, 2)] = r0.x

    jacobian_rot[_warp_jacobian_index(i, 2, 0)] = r0.y
    jacobian_rot[_warp_jacobian_index(i, 2, 1)] = -r0.x
    jacobian_rot[_warp_jacobian_index(i, 2, 2)] = 0.0

    jacobian_rot[_warp_jacobian_index(i, 0, 3)] = 0.0
    jacobian_rot[_warp_jacobian_index(i, 0, 4)] = -r1.z
    jacobian_rot[_warp_jacobian_index(i, 0, 5)] = r1.y

    jacobian_rot[_warp_jacobian_index(i, 1, 3)] = r1.z
    jacobian_rot[_warp_jacobian_index(i, 1, 4)] = 0.0
    jacobian_rot[_warp_jacobian_index(i, 1, 5)] = -r1.x

    jacobian_rot[_warp_jacobian_index(i, 2, 3)] = -r1.y
    jacobian_rot[_warp_jacobian_index(i, 2, 4)] = r1.x
    jacobian_rot[_warp_jacobian_index(i, 2, 5)] = 0.0

    x0 = q0.x
    y0 = q0.y
    z0 = q0.z
    w0 = q0.w
    x1 = q1.x
    y1 = q1.y
    z1 = q1.z
    w1 = q1.w

    j0_r0 = wp.vec4(-w1, -z1, y1, x1)
    j0_r1 = wp.vec4(z1, -w1, -x1, y1)
    j0_r2 = wp.vec4(-y1, x1, -w1, z1)

    g0_c0 = wp.vec4(0.5 * w0, -0.5 * z0, 0.5 * y0, -0.5 * x0)
    g0_c1 = wp.vec4(0.5 * z0, 0.5 * w0, -0.5 * x0, -0.5 * y0)
    g0_c2 = wp.vec4(-0.5 * y0, 0.5 * x0, 0.5 * w0, -0.5 * z0)

    j1_r0 = wp.vec4(w0, z0, -y0, -x0)
    j1_r1 = wp.vec4(-z0, w0, x0, -y0)
    j1_r2 = wp.vec4(y0, -x0, w0, -z0)

    g1_c0 = wp.vec4(0.5 * w1, -0.5 * z1, 0.5 * y1, -0.5 * x1)
    g1_c1 = wp.vec4(0.5 * z1, 0.5 * w1, -0.5 * x1, -0.5 * y1)
    g1_c2 = wp.vec4(-0.5 * y1, 0.5 * x1, 0.5 * w1, -0.5 * z1)

    jacobian_rot[_warp_jacobian_index(i, 3, 0)] = wp.dot(j0_r0, g0_c0)
    jacobian_rot[_warp_jacobian_index(i, 3, 1)] = wp.dot(j0_r0, g0_c1)
    jacobian_rot[_warp_jacobian_index(i, 3, 2)] = wp.dot(j0_r0, g0_c2)

    jacobian_rot[_warp_jacobian_index(i, 4, 0)] = wp.dot(j0_r1, g0_c0)
    jacobian_rot[_warp_jacobian_index(i, 4, 1)] = wp.dot(j0_r1, g0_c1)
    jacobian_rot[_warp_jacobian_index(i, 4, 2)] = wp.dot(j0_r1, g0_c2)

    jacobian_rot[_warp_jacobian_index(i, 5, 0)] = wp.dot(j0_r2, g0_c0)
    jacobian_rot[_warp_jacobian_index(i, 5, 1)] = wp.dot(j0_r2, g0_c1)
    jacobian_rot[_warp_jacobian_index(i, 5, 2)] = wp.dot(j0_r2, g0_c2)

    jacobian_rot[_warp_jacobian_index(i, 3, 3)] = wp.dot(j1_r0, g1_c0)
    jacobian_rot[_warp_jacobian_index(i, 3, 4)] = wp.dot(j1_r0, g1_c1)
    jacobian_rot[_warp_jacobian_index(i, 3, 5)] = wp.dot(j1_r0, g1_c2)

    jacobian_rot[_warp_jacobian_index(i, 4, 3)] = wp.dot(j1_r1, g1_c0)
    jacobian_rot[_warp_jacobian_index(i, 4, 4)] = wp.dot(j1_r1, g1_c1)
    jacobian_rot[_warp_jacobian_index(i, 4, 5)] = wp.dot(j1_r1, g1_c2)

    jacobian_rot[_warp_jacobian_index(i, 5, 3)] = wp.dot(j1_r2, g1_c0)
    jacobian_rot[_warp_jacobian_index(i, 5, 4)] = wp.dot(j1_r2, g1_c1)
    jacobian_rot[_warp_jacobian_index(i, 5, 5)] = wp.dot(j1_r2, g1_c2)


@wp.kernel
def _warp_build_rhs(
    constraint_values: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    lambda_sum: wp.array(dtype=wp.float32),
    n_dofs: int,
    rhs: wp.array(dtype=wp.float32),
):
    """Build the RHS vector for the linear system.

    Computes: rhs = -constraint - compliance * lambda_sum

    Args:
        constraint_values: Current constraint values.
        compliance: Compliance values.
        lambda_sum: Accumulated Lagrange multipliers.
        n_dofs: Number of DOFs in the system.
        rhs: Output RHS vector.
    """
    i = wp.tid()
    if i < n_dofs:
        rhs[i] = -constraint_values[i] - compliance[i] * lambda_sum[i]
    else:
        rhs[i] = 0.0


@wp.kernel
def _warp_prepare_compliance_batched(
    rest_lengths: wp.array(dtype=wp.float32),
    bend_stiffness: wp.array(dtype=wp.vec3),
    edge_rod_id: wp.array(dtype=wp.int32),
    young_modulus: wp.array(dtype=wp.float32),
    torsion_modulus: wp.array(dtype=wp.float32),
    dt: float,
    compliance: wp.array(dtype=wp.float32),
):
    """Compute compliance values for all rods in a single launch.

    This batched version processes all edges across all rods. Each edge
    looks up its rod ID to get the appropriate material properties.

    Args:
        rest_lengths: Concatenated rest lengths for all rods.
        bend_stiffness: Concatenated bend/twist stiffness per edge.
        edge_rod_id: Rod index for each edge.
        young_modulus: Per-rod Young's modulus [n_rods].
        torsion_modulus: Per-rod torsion modulus [n_rods].
        dt: Time step size.
        compliance: Output compliance values (6 per edge).
    """
    i = wp.tid()
    L = rest_lengths[i]
    dt2 = dt * dt
    eps = 1.0e-10

    rod_id = edge_rod_id[i]
    E = young_modulus[rod_id]
    G = torsion_modulus[rod_id]

    k_bend1_ref = E * bend_stiffness[i].x
    k_bend2_ref = E * bend_stiffness[i].y
    k_twist_ref = G * bend_stiffness[i].z

    k_bend1_eff = k_bend1_ref * L
    k_bend2_eff = k_bend2_ref * L
    k_twist_eff = k_twist_ref * L

    stretch_compliance = 1.0e-10

    base = i * 6
    compliance[base + 0] = stretch_compliance
    compliance[base + 1] = stretch_compliance
    compliance[base + 2] = stretch_compliance
    compliance[base + 3] = 1.0 / (k_bend1_eff * dt2 + eps)
    compliance[base + 4] = 1.0 / (k_bend2_eff * dt2 + eps)
    compliance[base + 5] = 1.0 / (k_twist_eff * dt2 + eps)


@wp.kernel
def _warp_update_constraints_batched_v2(
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quat),
    rest_lengths: wp.array(dtype=wp.float32),
    rest_darboux: wp.array(dtype=wp.vec3),
    rod_offsets: wp.array(dtype=wp.int32),
    edge_offsets: wp.array(dtype=wp.int32),
    edge_rod_id: wp.array(dtype=wp.int32),
    constraint_values: wp.array(dtype=wp.float32),
):
    """Compute constraint values for all rods in a single launch.

    This batched version processes all edges across all rods.

    Args:
        positions: Concatenated particle positions.
        orientations: Concatenated particle orientations.
        rest_lengths: Concatenated rest lengths.
        rest_darboux: Concatenated rest Darboux vectors.
        rod_offsets: Cumulative point offsets [n_rods + 1].
        edge_offsets: Cumulative edge offsets [n_rods + 1].
        edge_rod_id: Rod index for each edge.
        constraint_values: Output constraint values (6 per edge).
    """
    global_edge = wp.tid()
    rod_id = edge_rod_id[global_edge]

    # Local edge index within this rod
    local_edge = global_edge - edge_offsets[rod_id]

    # Global particle indices
    p0_idx = rod_offsets[rod_id] + local_edge
    p1_idx = p0_idx + 1

    p0 = positions[p0_idx]
    p1 = positions[p1_idx]
    q0 = orientations[p0_idx]
    q1 = orientations[p1_idx]

    half_L = 0.5 * rest_lengths[global_edge]
    r0_world = _warp_quat_rotate_vector(q0, wp.vec3(0.0, 0.0, half_L))
    r1_world = _warp_quat_rotate_vector(q1, wp.vec3(0.0, 0.0, -half_L))

    c0 = p0 + r0_world
    c1 = p1 + r1_world
    stretch_error = c0 - c1

    q_rel = _warp_quat_mul(_warp_quat_conjugate(q0), q1)
    omega = wp.vec3(q_rel.x, q_rel.y, q_rel.z)
    darboux_error = omega - rest_darboux[global_edge]

    base = global_edge * 6
    constraint_values[base + 0] = stretch_error.x
    constraint_values[base + 1] = stretch_error.y
    constraint_values[base + 2] = stretch_error.z
    constraint_values[base + 3] = darboux_error.x
    constraint_values[base + 4] = darboux_error.y
    constraint_values[base + 5] = darboux_error.z


@wp.kernel
def _warp_compute_jacobians_batched(
    orientations: wp.array(dtype=wp.quat),
    rest_lengths: wp.array(dtype=wp.float32),
    rod_offsets: wp.array(dtype=wp.int32),
    edge_offsets: wp.array(dtype=wp.int32),
    edge_rod_id: wp.array(dtype=wp.int32),
    jacobian_pos: wp.array(dtype=wp.float32),
    jacobian_rot: wp.array(dtype=wp.float32),
):
    """Compute Jacobians for all rods in a single launch.

    Args:
        orientations: Concatenated particle orientations.
        rest_lengths: Concatenated rest lengths.
        rod_offsets: Cumulative point offsets [n_rods + 1].
        edge_offsets: Cumulative edge offsets [n_rods + 1].
        edge_rod_id: Rod index for each edge.
        jacobian_pos: Output position Jacobian (36 floats per edge).
        jacobian_rot: Output rotation Jacobian (36 floats per edge).
    """
    global_edge = wp.tid()
    rod_id = edge_rod_id[global_edge]

    # Local edge index within this rod
    local_edge = global_edge - edge_offsets[rod_id]

    # Global particle indices
    p0_idx = rod_offsets[rod_id] + local_edge
    p1_idx = p0_idx + 1

    q0 = orientations[p0_idx]
    q1 = orientations[p1_idx]

    half_L = 0.5 * rest_lengths[global_edge]
    r0 = _warp_quat_rotate_vector(q0, wp.vec3(0.0, 0.0, half_L))
    r1 = _warp_quat_rotate_vector(q1, wp.vec3(0.0, 0.0, -half_L))

    i = global_edge

    # Identity for J_pos
    for d in range(3):
        jacobian_pos[_warp_jacobian_index(i, d, d)] = 1.0
        jacobian_pos[_warp_jacobian_index(i, d, d + 3)] = -1.0

    # -skew(r0) and skew(r1) for stretch rotation Jacobians
    jacobian_rot[_warp_jacobian_index(i, 0, 0)] = 0.0
    jacobian_rot[_warp_jacobian_index(i, 0, 1)] = r0.z
    jacobian_rot[_warp_jacobian_index(i, 0, 2)] = -r0.y

    jacobian_rot[_warp_jacobian_index(i, 1, 0)] = -r0.z
    jacobian_rot[_warp_jacobian_index(i, 1, 1)] = 0.0
    jacobian_rot[_warp_jacobian_index(i, 1, 2)] = r0.x

    jacobian_rot[_warp_jacobian_index(i, 2, 0)] = r0.y
    jacobian_rot[_warp_jacobian_index(i, 2, 1)] = -r0.x
    jacobian_rot[_warp_jacobian_index(i, 2, 2)] = 0.0

    jacobian_rot[_warp_jacobian_index(i, 0, 3)] = 0.0
    jacobian_rot[_warp_jacobian_index(i, 0, 4)] = -r1.z
    jacobian_rot[_warp_jacobian_index(i, 0, 5)] = r1.y

    jacobian_rot[_warp_jacobian_index(i, 1, 3)] = r1.z
    jacobian_rot[_warp_jacobian_index(i, 1, 4)] = 0.0
    jacobian_rot[_warp_jacobian_index(i, 1, 5)] = -r1.x

    jacobian_rot[_warp_jacobian_index(i, 2, 3)] = -r1.y
    jacobian_rot[_warp_jacobian_index(i, 2, 4)] = r1.x
    jacobian_rot[_warp_jacobian_index(i, 2, 5)] = 0.0

    x0 = q0.x
    y0 = q0.y
    z0 = q0.z
    w0 = q0.w
    x1 = q1.x
    y1 = q1.y
    z1 = q1.z
    w1 = q1.w

    j0_r0 = wp.vec4(-w1, -z1, y1, x1)
    j0_r1 = wp.vec4(z1, -w1, -x1, y1)
    j0_r2 = wp.vec4(-y1, x1, -w1, z1)

    g0_c0 = wp.vec4(0.5 * w0, -0.5 * z0, 0.5 * y0, -0.5 * x0)
    g0_c1 = wp.vec4(0.5 * z0, 0.5 * w0, -0.5 * x0, -0.5 * y0)
    g0_c2 = wp.vec4(-0.5 * y0, 0.5 * x0, 0.5 * w0, -0.5 * z0)

    j1_r0 = wp.vec4(w0, z0, -y0, -x0)
    j1_r1 = wp.vec4(-z0, w0, x0, -y0)
    j1_r2 = wp.vec4(y0, -x0, w0, -z0)

    g1_c0 = wp.vec4(0.5 * w1, -0.5 * z1, 0.5 * y1, -0.5 * x1)
    g1_c1 = wp.vec4(0.5 * z1, 0.5 * w1, -0.5 * x1, -0.5 * y1)
    g1_c2 = wp.vec4(-0.5 * y1, 0.5 * x1, 0.5 * w1, -0.5 * z1)

    jacobian_rot[_warp_jacobian_index(i, 3, 0)] = wp.dot(j0_r0, g0_c0)
    jacobian_rot[_warp_jacobian_index(i, 3, 1)] = wp.dot(j0_r0, g0_c1)
    jacobian_rot[_warp_jacobian_index(i, 3, 2)] = wp.dot(j0_r0, g0_c2)

    jacobian_rot[_warp_jacobian_index(i, 4, 0)] = wp.dot(j0_r1, g0_c0)
    jacobian_rot[_warp_jacobian_index(i, 4, 1)] = wp.dot(j0_r1, g0_c1)
    jacobian_rot[_warp_jacobian_index(i, 4, 2)] = wp.dot(j0_r1, g0_c2)

    jacobian_rot[_warp_jacobian_index(i, 5, 0)] = wp.dot(j0_r2, g0_c0)
    jacobian_rot[_warp_jacobian_index(i, 5, 1)] = wp.dot(j0_r2, g0_c1)
    jacobian_rot[_warp_jacobian_index(i, 5, 2)] = wp.dot(j0_r2, g0_c2)

    jacobian_rot[_warp_jacobian_index(i, 3, 3)] = wp.dot(j1_r0, g1_c0)
    jacobian_rot[_warp_jacobian_index(i, 3, 4)] = wp.dot(j1_r0, g1_c1)
    jacobian_rot[_warp_jacobian_index(i, 3, 5)] = wp.dot(j1_r0, g1_c2)

    jacobian_rot[_warp_jacobian_index(i, 4, 3)] = wp.dot(j1_r1, g1_c0)
    jacobian_rot[_warp_jacobian_index(i, 4, 4)] = wp.dot(j1_r1, g1_c1)
    jacobian_rot[_warp_jacobian_index(i, 4, 5)] = wp.dot(j1_r1, g1_c2)

    jacobian_rot[_warp_jacobian_index(i, 5, 3)] = wp.dot(j1_r2, g1_c0)
    jacobian_rot[_warp_jacobian_index(i, 5, 4)] = wp.dot(j1_r2, g1_c1)
    jacobian_rot[_warp_jacobian_index(i, 5, 5)] = wp.dot(j1_r2, g1_c2)


@wp.kernel
def _warp_build_rhs_stretch(
    constraint_values: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    lambda_sum: wp.array(dtype=wp.float32),
    n_edges: int,
    rhs: wp.array(dtype=wp.float32),
):
    """Build the RHS vector for stretch constraints only.

    Computes: rhs[3*edge + i] = -constraint[6*edge + i] - compliance[6*edge + i] * lambda_sum[6*edge + i]
    for i in 0, 1, 2 (stretch components).

    Args:
        constraint_values: Current constraint values (6 per edge).
        compliance: Compliance values (6 per edge).
        lambda_sum: Accumulated Lagrange multipliers (6 per edge).
        n_edges: Number of edges.
        rhs: Output RHS vector (3 floats per edge for stretch only).
    """
    edge = wp.tid()
    if edge >= n_edges:
        return

    for i in range(3):
        idx = edge * 6 + i  # Stretch = first 3 components
        rhs[edge * 3 + i] = -constraint_values[idx] - compliance[idx] * lambda_sum[idx]


@wp.kernel
def _warp_build_rhs_darboux(
    constraint_values: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    lambda_sum: wp.array(dtype=wp.float32),
    n_edges: int,
    rhs: wp.array(dtype=wp.float32),
):
    """Build the RHS vector for darboux constraints only.

    Computes: rhs[3*edge + i] = -constraint[6*edge + i + 3] - compliance[6*edge + i + 3] * lambda_sum[6*edge + i + 3]
    for i in 0, 1, 2 (darboux components).

    Args:
        constraint_values: Current constraint values (6 per edge).
        compliance: Compliance values (6 per edge).
        lambda_sum: Accumulated Lagrange multipliers (6 per edge).
        n_edges: Number of edges.
        rhs: Output RHS vector (3 floats per edge for darboux only).
    """
    edge = wp.tid()
    if edge >= n_edges:
        return

    for i in range(3):
        idx = edge * 6 + i + 3  # Darboux = last 3 components
        rhs[edge * 3 + i] = -constraint_values[idx] - compliance[idx] * lambda_sum[idx]


__all__ = [
    "_warp_build_rhs",
    "_warp_build_rhs_darboux",
    "_warp_build_rhs_stretch",
    "_warp_compute_jacobians_batched",
    "_warp_compute_jacobians_direct",
    "_warp_prepare_compliance",
    "_warp_prepare_compliance_batched",
    "_warp_update_constraints_batched_v2",
    "_warp_update_constraints_direct",
]
