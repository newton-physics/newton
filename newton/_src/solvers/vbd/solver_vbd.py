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

from __future__ import annotations

import warnings

import numpy as np
import warp as wp
from warp.types import float32, matrix

from ...core.types import override
from ...geometry import ParticleFlags
from ...geometry.kernels import triangle_closest_point
from ...sim import Contacts, Control, JointType, Model, State
from ..solver import SolverBase, integrate_rigid_body
from .tri_mesh_collision import (
    TriMeshCollisionDetector,
    TriMeshCollisionInfo,
)

# TODO: Grab changes from Warp that has fixed the backward pass
wp.set_module_options({"enable_backward": False})

VBD_DEBUG_PRINTING_OPTIONS = {
    # "elasticity_force_hessian",
    # "contact_force_hessian",
    # "contact_force_hessian_vt",
    # "contact_force_hessian_ee",
    # "overall_force_hessian",
    # "inertia_force_hessian",
    # "connectivity",
    # "contact_info",
}

NUM_THREADS_PER_COLLISION_PRIMITIVE = 4
TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE = 16

# Rotational dynamics precision control
# Set to True for high-accuracy scenarios with large rotations (slower but more accurate)
# Set to False for standard simulations (faster with small-angle approximations)
USE_EXACT_ROTATIONAL_DYNAMICS = wp.constant(False)
# Minimum stretch stiffness for cable constraints
STRETCH_STIFFNESS_MIN = wp.constant(1000.0)


class mat32(matrix(shape=(3, 2), dtype=float32)):
    pass


@wp.struct
class ForceElementAdjacencyInfo:
    r"""
    - vertex_adjacent_[element]: the flatten adjacency information. Its size is \sum_{i\inV} 2*N_i, where N_i is the
    number of vertex i's adjacent [element]. For each adjacent element it stores 2 information:
        - the id of the adjacent element
        - the order of the vertex in the element, which is essential to compute the force and hessian for the vertex
    - vertex_adjacent_[element]_offsets: stores where each vertex information starts in the  flatten adjacency array.
    Its size is |V|+1 such that the number of vertex i's adjacent [element] can be computed as
    vertex_adjacent_[element]_offsets[i+1]-vertex_adjacent_[element]_offsets[i].
    """

    v_adj_faces: wp.array(dtype=int)
    v_adj_faces_offsets: wp.array(dtype=int)

    v_adj_edges: wp.array(dtype=int)
    v_adj_edges_offsets: wp.array(dtype=int)

    v_adj_springs: wp.array(dtype=int)
    v_adj_springs_offsets: wp.array(dtype=int)

    # Rigid body joint adjacency
    body_adj_joints: wp.array(dtype=int)
    body_adj_joints_offsets: wp.array(dtype=int)

    def to(self, device):
        if device == self.v_adj_faces.device:
            return self
        else:
            adjacency_gpu = ForceElementAdjacencyInfo()
            adjacency_gpu.v_adj_faces = self.v_adj_faces.to(device)
            adjacency_gpu.v_adj_faces_offsets = self.v_adj_faces_offsets.to(device)

            adjacency_gpu.v_adj_edges = self.v_adj_edges.to(device)
            adjacency_gpu.v_adj_edges_offsets = self.v_adj_edges_offsets.to(device)

            adjacency_gpu.v_adj_springs = self.v_adj_springs.to(device)
            adjacency_gpu.v_adj_springs_offsets = self.v_adj_springs_offsets.to(device)

            adjacency_gpu.body_adj_joints = self.body_adj_joints.to(device)
            adjacency_gpu.body_adj_joints_offsets = self.body_adj_joints_offsets.to(device)

            return adjacency_gpu


@wp.func
def get_vertex_num_adjacent_edges(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32):
    return (adjacency.v_adj_edges_offsets[vertex + 1] - adjacency.v_adj_edges_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_edge_id_order(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32, edge: wp.int32):
    offset = adjacency.v_adj_edges_offsets[vertex]
    return adjacency.v_adj_edges[offset + edge * 2], adjacency.v_adj_edges[offset + edge * 2 + 1]


@wp.func
def get_vertex_num_adjacent_faces(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32):
    return (adjacency.v_adj_faces_offsets[vertex + 1] - adjacency.v_adj_faces_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_face_id_order(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32, face: wp.int32):
    offset = adjacency.v_adj_faces_offsets[vertex]
    return adjacency.v_adj_faces[offset + face * 2], adjacency.v_adj_faces[offset + face * 2 + 1]


@wp.func
def get_vertex_num_adjacent_springs(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32):
    return adjacency.v_adj_springs_offsets[vertex + 1] - adjacency.v_adj_springs_offsets[vertex]


@wp.func
def get_vertex_adjacent_spring_id(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32, spring: wp.int32):
    offset = adjacency.v_adj_springs_offsets[vertex]
    return adjacency.v_adj_springs[offset + spring]


@wp.func
def get_body_num_adjacent_joints(adjacency: ForceElementAdjacencyInfo, body: wp.int32):
    return adjacency.body_adj_joints_offsets[body + 1] - adjacency.body_adj_joints_offsets[body]


@wp.func
def get_body_adjacent_joint_id(adjacency: ForceElementAdjacencyInfo, body: wp.int32, joint: wp.int32):
    offset = adjacency.body_adj_joints_offsets[body]
    return adjacency.body_adj_joints[offset + joint]


@wp.kernel
def _test_compute_force_element_adjacency(
    adjacency: ForceElementAdjacencyInfo,
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    face_indices: wp.array(dtype=wp.int32, ndim=2),
):
    wp.printf("num vertices: %d\n", adjacency.v_adj_edges_offsets.shape[0] - 1)
    for vertex in range(adjacency.v_adj_edges_offsets.shape[0] - 1):
        num_adj_edges = get_vertex_num_adjacent_edges(adjacency, vertex)
        for i_bd in range(num_adj_edges):
            bd_id, v_order = get_vertex_adjacent_edge_id_order(adjacency, vertex, i_bd)

            if edge_indices[bd_id, v_order] != vertex:
                print("Error!!!")
                wp.printf("vertex: %d | num_adj_edges: %d\n", vertex, num_adj_edges)
                wp.printf("--iBd: %d | ", i_bd)
                wp.printf("edge id: %d | v_order: %d\n", bd_id, v_order)

        num_adj_faces = get_vertex_num_adjacent_faces(adjacency, vertex)

        for i_face in range(num_adj_faces):
            face, v_order = get_vertex_adjacent_face_id_order(
                adjacency,
                vertex,
                i_face,
            )

            if face_indices[face, v_order] != vertex:
                print("Error!!!")
                wp.printf("vertex: %d | num_adj_faces: %d\n", vertex, num_adj_faces)
                wp.printf("--i_face: %d | face id: %d | v_order: %d\n", i_face, face, v_order)
                wp.printf(
                    "--face: %d %d %d\n",
                    face_indices[face, 0],
                    face_indices[face, 1],
                    face_indices[face, 2],
                )


@wp.func
def build_orthonormal_basis(n: wp.vec3):
    """
    Builds an orthonormal basis given a normal vector `n`. Return the two axes that is perpendicular to `n`.

    :param n: A 3D vector (list or array-like) representing the normal vector
    """
    b1 = wp.vec3()
    b2 = wp.vec3()
    if n[2] < 0.0:
        a = 1.0 / (1.0 - n[2])
        b = n[0] * n[1] * a
        b1[0] = 1.0 - n[0] * n[0] * a
        b1[1] = -b
        b1[2] = n[0]

        b2[0] = b
        b2[1] = n[1] * n[1] * a - 1.0
        b2[2] = -n[1]
    else:
        a = 1.0 / (1.0 + n[2])
        b = -n[0] * n[1] * a
        b1[0] = 1.0 - n[0] * n[0] * a
        b1[1] = b
        b1[2] = -n[0]

        b2[0] = b
        b2[1] = 1.0 - n[1] * n[1] * a
        b2[2] = -n[1]

    return b1, b2


@wp.func
def evaluate_stvk_force_hessian(
    face: int,
    v_order: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_pose: wp.mat22,
    area: float,
    mu: float,
    lmbd: float,
    damping: float,
    dt: float,
):
    # StVK energy density: psi = mu * ||G||_F^2 + 0.5 * lambda * (trace(G))^2

    # Deformation gradient F = [f0, f1] (3x2 matrix as two 3D column vectors)
    v0 = tri_indices[face, 0]
    v1 = tri_indices[face, 1]
    v2 = tri_indices[face, 2]

    x0 = pos[v0]
    x01 = pos[v1] - x0
    x02 = pos[v2] - x0

    # Cache tri_pose elements
    DmInv00 = tri_pose[0, 0]
    DmInv01 = tri_pose[0, 1]
    DmInv10 = tri_pose[1, 0]
    DmInv11 = tri_pose[1, 1]

    # Compute F columns directly: F = [x01, x02] * tri_pose = [f0, f1]
    f0 = x01 * DmInv00 + x02 * DmInv10
    f1 = x01 * DmInv01 + x02 * DmInv11

    # Green strain tensor: G = 0.5(F^T F - I) = [[G00, G01], [G01, G11]] (symmetric 2x2)
    f0_dot_f0 = wp.dot(f0, f0)
    f1_dot_f1 = wp.dot(f1, f1)
    f0_dot_f1 = wp.dot(f0, f1)

    G00 = 0.5 * (f0_dot_f0 - 1.0)
    G11 = 0.5 * (f1_dot_f1 - 1.0)
    G01 = 0.5 * f0_dot_f1

    # Frobenius norm squared of Green strain: ||G||_F^2 = G00^2 + G11^2 + 2 * G01^2
    G_frobenius_sq = G00 * G00 + G11 * G11 + 2.0 * G01 * G01
    if G_frobenius_sq < 1.0e-20:
        return wp.vec3(0.0), wp.mat33(0.0)

    trace_G = G00 + G11

    # First Piola-Kirchhoff stress tensor (StVK model)
    # PK1 = 2*mu*F*G + lambda*trace(G)*F = [PK1_col0, PK1_col1] (3x2)
    lambda_trace_G = lmbd * trace_G
    two_mu = 2.0 * mu

    PK1_col0 = f0 * (two_mu * G00 + lambda_trace_G) + f1 * (two_mu * G01)
    PK1_col1 = f0 * (two_mu * G01) + f1 * (two_mu * G11 + lambda_trace_G)

    # Vertex selection using masks to avoid branching
    mask0 = float(v_order == 0)
    mask1 = float(v_order == 1)
    mask2 = float(v_order == 2)

    # Deformation gradient derivatives w.r.t. current vertex position
    df0_dx = DmInv00 * (mask1 - mask0) + DmInv10 * (mask2 - mask0)
    df1_dx = DmInv01 * (mask1 - mask0) + DmInv11 * (mask2 - mask0)

    # Force via chain rule: force = -(dpsi/dF) : (dF/dx)
    dpsi_dx = PK1_col0 * df0_dx + PK1_col1 * df1_dx
    force = -dpsi_dx

    # Hessian computation using Cauchy-Green invariants
    df0_dx_sq = df0_dx * df0_dx
    df1_dx_sq = df1_dx * df1_dx
    df0_df1_cross = df0_dx * df1_dx

    Ic = f0_dot_f0 + f1_dot_f1
    two_dpsi_dIc = -mu + (0.5 * Ic - 1.0) * lmbd
    I33 = wp.identity(n=3, dtype=float)

    f0_outer_f0 = wp.outer(f0, f0)
    f1_outer_f1 = wp.outer(f1, f1)
    f0_outer_f1 = wp.outer(f0, f1)
    f1_outer_f0 = wp.outer(f1, f0)

    H_IIc00_scaled = mu * (f0_dot_f0 * I33 + 2.0 * f0_outer_f0 + f1_outer_f1)
    H_IIc11_scaled = mu * (f1_dot_f1 * I33 + 2.0 * f1_outer_f1 + f0_outer_f0)
    H_IIc01_scaled = mu * (f0_dot_f1 * I33 + f1_outer_f0)

    # d2(psi)/dF^2 components
    d2E_dF2_00 = lmbd * f0_outer_f0 + two_dpsi_dIc * I33 + H_IIc00_scaled
    d2E_dF2_01 = lmbd * f0_outer_f1 + H_IIc01_scaled
    d2E_dF2_11 = lmbd * f1_outer_f1 + two_dpsi_dIc * I33 + H_IIc11_scaled

    # Chain rule: H = (dF/dx)^T * (d2(psi)/dF^2) * (dF/dx)
    hessian = df0_dx_sq * d2E_dF2_00 + df1_dx_sq * d2E_dF2_11 + df0_df1_cross * (d2E_dF2_01 + wp.transpose(d2E_dF2_01))

    if damping > 0.0:
        inv_dt = 1.0 / dt

        # Previous deformation gradient for velocity
        x0_prev = pos_prev[v0]
        x01_prev = pos_prev[v1] - x0_prev
        x02_prev = pos_prev[v2] - x0_prev

        vel_x01 = (x01 - x01_prev) * inv_dt
        vel_x02 = (x02 - x02_prev) * inv_dt

        df0_dt = vel_x01 * DmInv00 + vel_x02 * DmInv10
        df1_dt = vel_x01 * DmInv01 + vel_x02 * DmInv11

        # First constraint: Cmu = ||G||_F (Frobenius norm of Green strain)
        Cmu = wp.sqrt(G_frobenius_sq)

        G00_normalized = G00 / Cmu
        G01_normalized = G01 / Cmu
        G11_normalized = G11 / Cmu

        # Time derivative of Green strain: dG/dt = 0.5 * (F^T * dF/dt + (dF/dt)^T * F)
        dG_dt_00 = wp.dot(f0, df0_dt)  # dG00/dt
        dG_dt_11 = wp.dot(f1, df1_dt)  # dG11/dt
        dG_dt_01 = 0.5 * (wp.dot(f0, df1_dt) + wp.dot(f1, df0_dt))  # dG01/dt

        # Time derivative of first constraint: dCmu/dt = (1/||G||_F) * (G : dG/dt)
        dCmu_dt = G00_normalized * dG_dt_00 + G11_normalized * dG_dt_11 + 2.0 * G01_normalized * dG_dt_01

        # Gradient of first constraint w.r.t. deformation gradient: dCmu/dF = (G/||G||_F) * F
        dCmu_dF_col0 = G00_normalized * f0 + G01_normalized * f1  # dCmu/df0
        dCmu_dF_col1 = G01_normalized * f0 + G11_normalized * f1  # dCmu/df1

        # Gradient of constraint w.r.t. vertex position: dCmu/dx = (dCmu/dF) : (dF/dx)
        dCmu_dx = df0_dx * dCmu_dF_col0 + df1_dx * dCmu_dF_col1

        # Damping force from first constraint: -mu * damping * (dCmu/dt) * (dCmu/dx)
        kd_mu = mu * damping
        force += -kd_mu * dCmu_dt * dCmu_dx

        # Damping Hessian: mu * damping * (1/dt) * (dCmu/dx) x (dCmu/dx)
        hessian += kd_mu * inv_dt * wp.outer(dCmu_dx, dCmu_dx)

        # Second constraint: Clmbd = trace(G) = G00 + G11 (trace of Green strain)
        # Time derivative of second constraint: dClmbd/dt = trace(dG/dt)
        dClmbd_dt = dG_dt_00 + dG_dt_11

        # Gradient of second constraint w.r.t. deformation gradient: dClmbd/dF = F
        dClmbd_dF_col0 = f0  # dClmbd/df0
        dClmbd_dF_col1 = f1  # dClmbd/df1

        # Gradient of Clmbd w.r.t. vertex position: dClmbd/dx = (dClmbd/dF) : (dF/dx)
        dClmbd_dx = df0_dx * dClmbd_dF_col0 + df1_dx * dClmbd_dF_col1

        # Damping force from second constraint: -lambda * damping * (dClmbd/dt) * (dClmbd/dx)
        kd_lmbd = lmbd * damping
        force += -kd_lmbd * dClmbd_dt * dClmbd_dx

        # Damping Hessian from second constraint: lambda * damping * (1/dt) * (dClmbd/dx) x (dClmbd/dx)
        hessian += kd_lmbd * inv_dt * wp.outer(dClmbd_dx, dClmbd_dx)

    # Apply area scaling
    force *= area
    hessian *= area

    return force, hessian


@wp.func
def compute_normalized_vector_derivative(
    unnormalized_vec_length: float, normalized_vec: wp.vec3, unnormalized_vec_derivative: wp.mat33
) -> wp.mat33:
    projection_matrix = wp.identity(n=3, dtype=float) - wp.outer(normalized_vec, normalized_vec)

    # d(normalized_vec)/dx = (1/|unnormalized_vec|) * (I - normalized_vec * normalized_vec^T) * d(unnormalized_vec)/dx
    return (1.0 / unnormalized_vec_length) * projection_matrix * unnormalized_vec_derivative


@wp.func
def compute_angle_derivative(
    n1_hat: wp.vec3,
    n2_hat: wp.vec3,
    e_hat: wp.vec3,
    dn1hat_dx: wp.mat33,
    dn2hat_dx: wp.mat33,
    sin_theta: float,
    cos_theta: float,
    skew_n1: wp.mat33,
    skew_n2: wp.mat33,
) -> wp.vec3:
    dsin_dx = wp.transpose(skew_n1 * dn2hat_dx - skew_n2 * dn1hat_dx) * e_hat
    dcos_dx = wp.transpose(dn1hat_dx) * n2_hat + wp.transpose(dn2hat_dx) * n1_hat

    # dtheta/dx = dsin/dx * cos - dcos/dx * sin
    return dsin_dx * cos_theta - dcos_dx * sin_theta


@wp.func
def evaluate_dihedral_angle_based_bending_force_hessian(
    bending_index: int,
    v_order: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angle: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    stiffness: float,
    damping: float,
    dt: float,
):
    # Skip invalid edges (boundary edges with missing opposite vertices)
    if edge_indices[bending_index, 0] == -1 or edge_indices[bending_index, 1] == -1:
        return wp.vec3(0.0), wp.mat33(0.0)

    eps = 1.0e-6

    vi0 = edge_indices[bending_index, 0]
    vi1 = edge_indices[bending_index, 1]
    vi2 = edge_indices[bending_index, 2]
    vi3 = edge_indices[bending_index, 3]

    x0 = pos[vi0]  # opposite 0
    x1 = pos[vi1]  # opposite 1
    x2 = pos[vi2]  # edge start
    x3 = pos[vi3]  # edge end

    # Compute edge vectors
    x02 = x2 - x0
    x03 = x3 - x0
    x13 = x3 - x1
    x12 = x2 - x1
    e = x3 - x2

    # Compute normals
    n1 = wp.cross(x02, x03)
    n2 = wp.cross(x13, x12)

    n1_norm = wp.length(n1)
    n2_norm = wp.length(n2)
    e_norm = wp.length(e)

    # Early exit for degenerate cases
    if n1_norm < eps or n2_norm < eps or e_norm < eps:
        return wp.vec3(0.0), wp.mat33(0.0)

    n1_hat = n1 / n1_norm
    n2_hat = n2 / n2_norm
    e_hat = e / e_norm

    sin_theta = wp.dot(wp.cross(n1_hat, n2_hat), e_hat)
    cos_theta = wp.dot(n1_hat, n2_hat)
    theta = wp.atan2(sin_theta, cos_theta)

    k = stiffness * edge_rest_length[bending_index]
    dE_dtheta = k * (theta - edge_rest_angle[bending_index])

    # Pre-compute skew matrices (shared across all angle derivative computations)
    skew_e = wp.skew(e)
    skew_x03 = wp.skew(x03)
    skew_x02 = wp.skew(x02)
    skew_x13 = wp.skew(x13)
    skew_x12 = wp.skew(x12)
    skew_n1 = wp.skew(n1_hat)
    skew_n2 = wp.skew(n2_hat)

    # Compute the derivatives of unit normals with respect to each vertex; required for computing angle derivatives
    dn1hat_dx0 = compute_normalized_vector_derivative(n1_norm, n1_hat, skew_e)
    dn2hat_dx0 = wp.mat33(0.0)

    dn1hat_dx1 = wp.mat33(0.0)
    dn2hat_dx1 = compute_normalized_vector_derivative(n2_norm, n2_hat, -skew_e)

    dn1hat_dx2 = compute_normalized_vector_derivative(n1_norm, n1_hat, -skew_x03)
    dn2hat_dx2 = compute_normalized_vector_derivative(n2_norm, n2_hat, skew_x13)

    dn1hat_dx3 = compute_normalized_vector_derivative(n1_norm, n1_hat, skew_x02)
    dn2hat_dx3 = compute_normalized_vector_derivative(n2_norm, n2_hat, -skew_x12)

    # Compute all angle derivatives (required for damping)
    dtheta_dx0 = compute_angle_derivative(
        n1_hat, n2_hat, e_hat, dn1hat_dx0, dn2hat_dx0, sin_theta, cos_theta, skew_n1, skew_n2
    )
    dtheta_dx1 = compute_angle_derivative(
        n1_hat, n2_hat, e_hat, dn1hat_dx1, dn2hat_dx1, sin_theta, cos_theta, skew_n1, skew_n2
    )
    dtheta_dx2 = compute_angle_derivative(
        n1_hat, n2_hat, e_hat, dn1hat_dx2, dn2hat_dx2, sin_theta, cos_theta, skew_n1, skew_n2
    )
    dtheta_dx3 = compute_angle_derivative(
        n1_hat, n2_hat, e_hat, dn1hat_dx3, dn2hat_dx3, sin_theta, cos_theta, skew_n1, skew_n2
    )

    # Use float masks for branch-free selection
    mask0 = float(v_order == 0)
    mask1 = float(v_order == 1)
    mask2 = float(v_order == 2)
    mask3 = float(v_order == 3)

    # Select the derivative for the current vertex without branching
    dtheta_dx = dtheta_dx0 * mask0 + dtheta_dx1 * mask1 + dtheta_dx2 * mask2 + dtheta_dx3 * mask3

    # Compute elastic force and hessian
    bending_force = -dE_dtheta * dtheta_dx
    bending_hessian = k * wp.outer(dtheta_dx, dtheta_dx)

    if damping > 0.0:
        inv_dt = 1.0 / dt
        x_prev0 = pos_prev[vi0]
        x_prev1 = pos_prev[vi1]
        x_prev2 = pos_prev[vi2]
        x_prev3 = pos_prev[vi3]

        # Compute displacement vectors
        dx0 = x0 - x_prev0
        dx1 = x1 - x_prev1
        dx2 = x2 - x_prev2
        dx3 = x3 - x_prev3

        # Compute angular velocity using all derivatives
        dtheta_dt = (
            wp.dot(dtheta_dx0, dx0) + wp.dot(dtheta_dx1, dx1) + wp.dot(dtheta_dx2, dx2) + wp.dot(dtheta_dx3, dx3)
        ) * inv_dt

        damping_coeff = damping * k  # damping coefficients following the VBD convention
        damping_force = -damping_coeff * dtheta_dt * dtheta_dx
        damping_hessian = damping_coeff * inv_dt * wp.outer(dtheta_dx, dtheta_dx)

        bending_force = bending_force + damping_force
        bending_hessian = bending_hessian + damping_hessian

    return bending_force, bending_hessian


@wp.func
def evaluate_rigid_contact_from_collision(
    body_a_index: int,
    body_b_index: int,
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    contact_point_a_local: wp.vec3,  # Local contact point on body A
    contact_point_b_local: wp.vec3,  # Local contact point on body B
    contact_normal: wp.vec3,  # Contact normal (A to B)
    penetration_depth: float,  # Penetration depth (> 0 when penetrating)
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    dt: float,
):
    """
    Compute contact forces and VBD Hessian blocks for body-body collision.

    Uses linear penalty model with damping for repulsion and regularized Coulomb
    friction.

    Args:
        body_a_index: Body A index (-1 for static/kinematic body)
        body_b_index: Body B index (-1 for static/kinematic body)
        body_q: Current body transforms (world space)
        body_q_prev: Previous body transforms (world space)
        body_com: Body center-of-mass offsets (local body coordinates)
        contact_point_a_local: Contact point on body A (local body coordinates)
        contact_point_b_local: Contact point on body B (local body coordinates)
        contact_normal: Unit contact normal from collision detection (A to B, world coordinates; expected normalized)
        penetration_depth: Penetration depth from collision detection (> 0 when penetrating)
        soft_contact_ke: Contact normal stiffness
        soft_contact_kd: Contact damping coefficient
        friction_mu: Coulomb friction coefficient
        friction_epsilon: Friction regularization parameter
        dt: Time step

    Returns:
        Tuple of (force_a, torque_a, h_ll_a, h_al_a, h_aa_a,
                  force_b, torque_b, h_ll_b, h_al_b, h_aa_b):
        Per-body forces, torques, and VBD Hessian blocks:
        - h_ll: Linear-linear coupling
        - h_al: Angular-linear coupling
        - h_aa: Angular-angular coupling
    """

    # Reusable zero constants for entire function
    zero_vec = wp.vec3(0.0)
    zero_mat = wp.mat33(0.0)

    # Early exit: no penetration or zero stiffness
    if penetration_depth <= 0.0 or soft_contact_ke <= 0.0:
        return (zero_vec, zero_vec, zero_mat, zero_mat, zero_mat, zero_vec, zero_vec, zero_mat, zero_mat, zero_mat)

    # Handle static bodies (index < 0) with identity transforms
    X_wa = wp.transform_identity() if body_a_index < 0 else body_q[body_a_index]
    X_wa_prev = wp.transform_identity() if body_a_index < 0 else body_q_prev[body_a_index]
    body_a_com_local = wp.vec3(0.0) if body_a_index < 0 else body_com[body_a_index]

    X_wb = wp.transform_identity() if body_b_index < 0 else body_q[body_b_index]
    X_wb_prev = wp.transform_identity() if body_b_index < 0 else body_q_prev[body_b_index]
    body_b_com_local = wp.vec3(0.0) if body_b_index < 0 else body_com[body_b_index]

    # Centers of mass in world coordinates
    x_com_a_now = wp.transform_point(X_wa, body_a_com_local)
    x_com_b_now = wp.transform_point(X_wb, body_b_com_local)

    # Contact points in world coordinates
    x_c_a_now = wp.transform_point(X_wa, contact_point_a_local)
    x_c_b_now = wp.transform_point(X_wb, contact_point_b_local)
    x_c_a_prev = wp.transform_point(X_wa_prev, contact_point_a_local)
    x_c_b_prev = wp.transform_point(X_wb_prev, contact_point_b_local)

    # Contact motion for damping and friction (finite difference velocity estimation)
    dx_a = x_c_a_now - x_c_a_prev  # Motion of contact point on A over timestep dt
    dx_b = x_c_b_now - x_c_b_prev  # Motion of contact point on B over timestep dt
    dx_rel = dx_b - dx_a  # Relative contact motion (B relative to A)

    # Contact geometry - assume contact_normal is already unit length from collision detection
    n = contact_normal  # Unit normal (A to B)

    # Normal repulsion force using linear penalty method: F = ke * depth * n
    f_total = n * (soft_contact_ke * penetration_depth)
    K_total = soft_contact_ke * wp.outer(n, n)  # Stiffness matrix: K = ke * outer(n, n)

    # Apply damping when contacts are compressing
    if soft_contact_kd > 0.0 and wp.dot(n, dx_rel) < 0.0:
        # Contact is being compressed, apply damping
        damping_hessian = (soft_contact_kd / dt) * K_total
        f_total = f_total - (damping_hessian * dx_rel)
        K_total = K_total + damping_hessian

    # Friction forces
    if friction_mu > 0.0:
        # Relative contact velocity for friction
        v_rel = dx_rel / dt

        # Tangential slip velocity
        v_n = n * wp.dot(n, v_rel)
        v_t = v_rel - v_n
        u = v_t * dt  # Tangential slip over timestep
        eps_u = friction_epsilon * dt

        # Normal load (use updated normal force including damping)
        N = wp.max(0.0, wp.dot(f_total, n))

        # Tangent frame and 2D projection
        e0, e1 = build_orthonormal_basis(n)
        T = mat32(e0[0], e1[0], e0[1], e1[1], e0[2], e1[2])
        u2 = wp.transpose(T) * u

        # Apply friction using shared helper
        f_friction, K_friction = compute_friction(friction_mu, N, T, u2, eps_u)
        f_total = f_total + f_friction
        K_total = K_total + K_friction

    # Split total contact force to both bodies (Newton's 3rd law)
    force_a = -f_total  # Force on A (opposite to normal, pushes A away from B)
    force_b = f_total  # Force on B (along normal, pushes B away from A)

    # Torque arms and resulting torques
    r_a = x_c_a_now - x_com_a_now  # Moment arm from A's COM to contact point
    r_b = x_c_b_now - x_com_b_now  # Moment arm from B's COM to contact point

    # Small lever arm guards: prevent numerical issues when contact is near COM
    r_a_mag_sq = wp.dot(r_a, r_a)
    r_b_mag_sq = wp.dot(r_b, r_b)
    lever_arm_threshold = 1e-10  # Threshold for tiny lever arms

    # VBD Hessian blocks for body A using contact Jacobian approach
    # For contact force at point r_a from COM, the generalized force Jacobian is:
    # J_a = [-[r_a]x, I] where [r_a]x is the skew-symmetric matrix of r_a
    # VBD Hessian: H_a = J_a^T * K_total * J_a
    if r_a_mag_sq > lever_arm_threshold:
        # Normal case: compute torque and angular Hessian blocks
        torque_a = wp.cross(r_a, force_a)
        r_a_skew = wp.mat33(0.0, -r_a[2], r_a[1], r_a[2], 0.0, -r_a[0], -r_a[1], r_a[0], 0.0)
        r_a_skew_T_K = wp.transpose(r_a_skew) * K_total  # Common operation
        h_aa_a = r_a_skew_T_K * r_a_skew  # Angular-angular
        h_al_a = -r_a_skew_T_K  # Angular-linear
    else:
        # Tiny lever arm: zero out angular components (contact near COM)
        torque_a = zero_vec
        h_aa_a = zero_mat
        h_al_a = zero_mat

    h_ll_a = K_total  # Linear-linear (always computed)

    # VBD Hessian blocks for body B (same structure as body A)
    if r_b_mag_sq > lever_arm_threshold:
        # Normal case: compute torque and angular Hessian blocks
        torque_b = wp.cross(r_b, force_b)
        r_b_skew = wp.mat33(0.0, -r_b[2], r_b[1], r_b[2], 0.0, -r_b[0], -r_b[1], r_b[0], 0.0)
        r_b_skew_T_K = wp.transpose(r_b_skew) * K_total  # Common operation
        h_aa_b = r_b_skew_T_K * r_b_skew  # Angular-angular
        h_al_b = -r_b_skew_T_K  # Angular-linear
    else:
        # Tiny lever arm: zero out angular components (contact near COM)
        torque_b = zero_vec
        h_aa_b = zero_mat
        h_al_b = zero_mat

    h_ll_b = K_total  # Linear-linear (always computed)

    return (force_a, torque_a, h_ll_a, h_al_a, h_aa_a, force_b, torque_b, h_ll_b, h_al_b, h_aa_b)


@wp.func
def evaluate_body_particle_contact(
    particle_index: int,
    particle_pos: wp.vec3,
    particle_prev_pos: wp.vec3,
    contact_index: int,
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    particle_radius: wp.array(dtype=float),
    shape_material_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    dt: float,
):
    shape_index = contact_shape[contact_index]
    body_index = shape_body[shape_index]

    X_wb = wp.transform_identity()
    X_com = wp.vec3()
    if body_index >= 0:
        X_wb = body_q[body_index]
        X_com = body_com[body_index]

    # body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[contact_index])

    n = contact_normal[contact_index]

    penetration_depth = -(wp.dot(n, particle_pos - bx) - particle_radius[particle_index])
    if penetration_depth > 0:
        body_contact_force_norm = penetration_depth * soft_contact_ke
        body_contact_force = n * body_contact_force_norm
        body_contact_hessian = soft_contact_ke * wp.outer(n, n)

        mu = shape_material_mu[shape_index]

        dx = particle_pos - particle_prev_pos

        if wp.dot(n, dx) < 0:
            damping_hessian = (soft_contact_kd / dt) * body_contact_hessian
            body_contact_hessian = body_contact_hessian + damping_hessian
            body_contact_force = body_contact_force - damping_hessian * dx

        # body velocity
        if body_q_prev:
            # if body_q_prev is available, compute velocity using finite difference method
            # this is more accurate for simulating static friction
            X_wb_prev = wp.transform_identity()
            if body_index >= 0:
                X_wb_prev = body_q_prev[body_index]
            bx_prev = wp.transform_point(X_wb_prev, contact_body_pos[contact_index])
            bv = (bx - bx_prev) / dt + wp.transform_vector(X_wb, contact_body_vel[contact_index])

        else:
            # otherwise use the instantaneous velocity
            r = bx - wp.transform_point(X_wb, X_com)
            body_v_s = wp.spatial_vector()
            if body_index >= 0:
                body_v_s = body_qd[body_index]

            body_w = wp.spatial_bottom(body_v_s)
            body_v = wp.spatial_top(body_v_s)

            # compute the body velocity at the particle position
            bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, contact_body_vel[contact_index])

        relative_translation = dx - bv * dt

        # friction
        e0, e1 = build_orthonormal_basis(n)

        T = mat32(e0[0], e1[0], e0[1], e1[1], e0[2], e1[2])

        u = wp.transpose(T) * relative_translation
        eps_u = friction_epsilon * dt

        friction_force, friction_hessian = compute_friction(mu, body_contact_force_norm, T, u, eps_u)
        body_contact_force = body_contact_force + friction_force
        body_contact_hessian = body_contact_hessian + friction_hessian
    else:
        body_contact_force = wp.vec3(0.0, 0.0, 0.0)
        body_contact_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return body_contact_force, body_contact_hessian


@wp.func
def evaluate_self_contact_force_norm(dis: float, collision_radius: float, k: float):
    # Adjust distance and calculate penetration depth

    penetration_depth = collision_radius - dis

    # Initialize outputs
    dEdD = wp.float32(0.0)
    d2E_dDdD = wp.float32(0.0)

    # C2 continuity calculation
    tau = collision_radius * 0.5
    if tau > dis > 1e-5:
        k2 = 0.5 * tau * tau * k
        dEdD = -k2 / dis
        d2E_dDdD = k2 / (dis * dis)
    else:
        dEdD = -k * penetration_depth
        d2E_dDdD = k

    return dEdD, d2E_dDdD


@wp.func
def damp_collision(
    displacement: wp.vec3,
    collision_normal: wp.vec3,
    collision_hessian: wp.mat33,
    collision_damping: float,
    dt: float,
):
    if wp.dot(displacement, collision_normal) > 0:
        damping_hessian = (collision_damping / dt) * collision_hessian
        damping_force = damping_hessian * displacement
        return damping_force, damping_hessian
    else:
        return wp.vec3(0.0), wp.mat33(0.0)


@wp.func
def evaluate_edge_edge_contact(
    v: int,
    v_order: int,
    e1: int,
    e2: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    dt: float,
    edge_edge_parallel_epsilon: float,
):
    r"""
    Returns the edge-edge contact force and hessian, including the friction force.
    Args:
        v:
        v_order: \in {0, 1, 2, 3}, 0, 1 is vertex 0, 1 of e1, 2,3 is vertex 0, 1 of e2
        e0
        e1
        pos
        pos_prev,
        edge_indices
        collision_radius
        collision_stiffness
        dt
        edge_edge_parallel_epsilon: threshold to determine whether 2 edges are parallel
    """
    e1_v1 = edge_indices[e1, 2]
    e1_v2 = edge_indices[e1, 3]

    e1_v1_pos = pos[e1_v1]
    e1_v2_pos = pos[e1_v2]

    e2_v1 = edge_indices[e2, 2]
    e2_v2 = edge_indices[e2, 3]

    e2_v1_pos = pos[e2_v1]
    e2_v2_pos = pos[e2_v2]

    st = wp.closest_point_edge_edge(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos, edge_edge_parallel_epsilon)
    s = st[0]
    t = st[1]
    e1_vec = e1_v2_pos - e1_v1_pos
    e2_vec = e2_v2_pos - e2_v1_pos
    c1 = e1_v1_pos + e1_vec * s
    c2 = e2_v1_pos + e2_vec * t

    # c1, c2, s, t = closest_point_edge_edge_2(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos)

    diff = c1 - c2
    dis = st[2]
    collision_normal = diff / dis

    if dis < collision_radius:
        bs = wp.vec4(1.0 - s, s, -1.0 + t, -t)
        v_bary = bs[v_order]

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

        collision_force = -dEdD * v_bary * collision_normal
        collision_hessian = d2E_dDdD * v_bary * v_bary * wp.outer(collision_normal, collision_normal)

        # friction
        c1_prev = pos_prev[e1_v1] + (pos_prev[e1_v2] - pos_prev[e1_v1]) * s
        c2_prev = pos_prev[e2_v1] + (pos_prev[e2_v2] - pos_prev[e2_v1]) * t

        dx = (c1 - c1_prev) - (c2 - c2_prev)
        axis_1, axis_2 = build_orthonormal_basis(collision_normal)

        T = mat32(
            axis_1[0],
            axis_2[0],
            axis_1[1],
            axis_2[1],
            axis_1[2],
            axis_2[2],
        )

        u = wp.transpose(T) * dx
        eps_U = friction_epsilon * dt

        # fmt: off
        if wp.static("contact_force_hessian_ee" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "    collision force:\n    %f %f %f,\n    collision hessian:\n    %f %f %f,\n    %f %f %f,\n    %f %f %f\n",
                collision_force[0], collision_force[1], collision_force[2], collision_hessian[0, 0], collision_hessian[0, 1], collision_hessian[0, 2], collision_hessian[1, 0], collision_hessian[1, 1], collision_hessian[1, 2], collision_hessian[2, 0], collision_hessian[2, 1], collision_hessian[2, 2],
            )
        # fmt: on

        friction_force, friction_hessian = compute_friction(friction_coefficient, -dEdD, T, u, eps_U)
        friction_force = friction_force * v_bary
        friction_hessian = friction_hessian * v_bary * v_bary

        # # fmt: off
        # if wp.static("contact_force_hessian_ee" in VBD_DEBUG_PRINTING_OPTIONS):
        #     wp.printf(
        #         "    friction force:\n    %f %f %f,\n    friction hessian:\n    %f %f %f,\n    %f %f %f,\n    %f %f %f\n",
        #         friction_force[0], friction_force[1], friction_force[2], friction_hessian[0, 0], friction_hessian[0, 1], friction_hessian[0, 2], friction_hessian[1, 0], friction_hessian[1, 1], friction_hessian[1, 2], friction_hessian[2, 0], friction_hessian[2, 1], friction_hessian[2, 2],
        #     )
        # # fmt: on

        if v_order == 0:
            displacement = pos_prev[e1_v1] - e1_v1_pos
        elif v_order == 1:
            displacement = pos_prev[e1_v2] - e1_v2_pos
        elif v_order == 2:
            displacement = pos_prev[e2_v1] - e2_v1_pos
        else:
            displacement = pos_prev[e2_v2] - e2_v2_pos

        collision_normal_sign = wp.vec4(1.0, 1.0, -1.0, -1.0)
        if wp.dot(displacement, collision_normal * collision_normal_sign[v_order]) > 0:
            damping_hessian = (collision_damping / dt) * collision_hessian
            collision_hessian = collision_hessian + damping_hessian
            collision_force = collision_force + damping_hessian * displacement

        collision_force = collision_force + friction_force
        collision_hessian = collision_hessian + friction_hessian
    else:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return collision_force, collision_hessian


@wp.func
def evaluate_edge_edge_contact_2_vertices(
    e1: int,
    e2: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    dt: float,
    edge_edge_parallel_epsilon: float,
):
    r"""
    Returns the edge-edge contact force and hessian, including the friction force.
    Args:
        v:
        v_order: \in {0, 1, 2, 3}, 0, 1 is vertex 0, 1 of e1, 2,3 is vertex 0, 1 of e2
        e0
        e1
        pos
        edge_indices
        collision_radius
        collision_stiffness
        dt
    """
    e1_v1 = edge_indices[e1, 2]
    e1_v2 = edge_indices[e1, 3]

    e1_v1_pos = pos[e1_v1]
    e1_v2_pos = pos[e1_v2]

    e2_v1 = edge_indices[e2, 2]
    e2_v2 = edge_indices[e2, 3]

    e2_v1_pos = pos[e2_v1]
    e2_v2_pos = pos[e2_v2]

    st = wp.closest_point_edge_edge(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos, edge_edge_parallel_epsilon)
    s = st[0]
    t = st[1]
    e1_vec = e1_v2_pos - e1_v1_pos
    e2_vec = e2_v2_pos - e2_v1_pos
    c1 = e1_v1_pos + e1_vec * s
    c2 = e2_v1_pos + e2_vec * t

    # c1, c2, s, t = closest_point_edge_edge_2(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos)

    diff = c1 - c2
    dis = st[2]
    collision_normal = diff / dis

    if 0.0 < dis < collision_radius:
        bs = wp.vec4(1.0 - s, s, -1.0 + t, -t)

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

        collision_force = -dEdD * collision_normal
        collision_hessian = d2E_dDdD * wp.outer(collision_normal, collision_normal)

        # friction
        c1_prev = pos_prev[e1_v1] + (pos_prev[e1_v2] - pos_prev[e1_v1]) * s
        c2_prev = pos_prev[e2_v1] + (pos_prev[e2_v2] - pos_prev[e2_v1]) * t

        dx = (c1 - c1_prev) - (c2 - c2_prev)
        axis_1, axis_2 = build_orthonormal_basis(collision_normal)

        T = mat32(
            axis_1[0],
            axis_2[0],
            axis_1[1],
            axis_2[1],
            axis_1[2],
            axis_2[2],
        )

        u = wp.transpose(T) * dx
        eps_U = friction_epsilon * dt

        # fmt: off
        if wp.static("contact_force_hessian_ee" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "    collision force:\n    %f %f %f,\n    collision hessian:\n    %f %f %f,\n    %f %f %f,\n    %f %f %f\n",
                collision_force[0], collision_force[1], collision_force[2], collision_hessian[0, 0], collision_hessian[0, 1], collision_hessian[0, 2], collision_hessian[1, 0], collision_hessian[1, 1], collision_hessian[1, 2], collision_hessian[2, 0], collision_hessian[2, 1], collision_hessian[2, 2],
            )
        # fmt: on

        friction_force, friction_hessian = compute_friction(friction_coefficient, -dEdD, T, u, eps_U)

        # # fmt: off
        # if wp.static("contact_force_hessian_ee" in VBD_DEBUG_PRINTING_OPTIONS):
        #     wp.printf(
        #         "    friction force:\n    %f %f %f,\n    friction hessian:\n    %f %f %f,\n    %f %f %f,\n    %f %f %f\n",
        #         friction_force[0], friction_force[1], friction_force[2], friction_hessian[0, 0], friction_hessian[0, 1], friction_hessian[0, 2], friction_hessian[1, 0], friction_hessian[1, 1], friction_hessian[1, 2], friction_hessian[2, 0], friction_hessian[2, 1], friction_hessian[2, 2],
        #     )
        # # fmt: on

        displacement_0 = pos_prev[e1_v1] - e1_v1_pos
        displacement_1 = pos_prev[e1_v2] - e1_v2_pos

        collision_force_0 = collision_force * bs[0]
        collision_force_1 = collision_force * bs[1]

        collision_hessian_0 = collision_hessian * bs[0] * bs[0]
        collision_hessian_1 = collision_hessian * bs[1] * bs[1]

        collision_normal_sign = wp.vec4(1.0, 1.0, -1.0, -1.0)
        damping_force, damping_hessian = damp_collision(
            displacement_0,
            collision_normal * collision_normal_sign[0],
            collision_hessian_0,
            collision_damping,
            dt,
        )

        collision_force_0 += damping_force + bs[0] * friction_force
        collision_hessian_0 += damping_hessian + bs[0] * bs[0] * friction_hessian

        damping_force, damping_hessian = damp_collision(
            displacement_1,
            collision_normal * collision_normal_sign[1],
            collision_hessian_1,
            collision_damping,
            dt,
        )
        collision_force_1 += damping_force + bs[1] * friction_force
        collision_hessian_1 += damping_hessian + bs[1] * bs[1] * friction_hessian

        return True, collision_force_0, collision_force_1, collision_hessian_0, collision_hessian_1
    else:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return False, collision_force, collision_force, collision_hessian, collision_hessian


@wp.func
def evaluate_vertex_triangle_collision_force_hessian(
    v: int,
    v_order: int,
    tri: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    dt: float,
):
    a = pos[tri_indices[tri, 0]]
    b = pos[tri_indices[tri, 1]]
    c = pos[tri_indices[tri, 2]]

    p = pos[v]

    closest_p, bary, feature_type = triangle_closest_point(a, b, c, p)

    diff = p - closest_p
    dis = wp.length(diff)
    collision_normal = diff / dis

    if dis < collision_radius:
        bs = wp.vec4(-bary[0], -bary[1], -bary[2], 1.0)
        v_bary = bs[v_order]

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

        collision_force = -dEdD * v_bary * collision_normal
        collision_hessian = d2E_dDdD * v_bary * v_bary * wp.outer(collision_normal, collision_normal)

        # friction force
        dx_v = p - pos_prev[v]

        closest_p_prev = (
            bary[0] * pos_prev[tri_indices[tri, 0]]
            + bary[1] * pos_prev[tri_indices[tri, 1]]
            + bary[2] * pos_prev[tri_indices[tri, 2]]
        )

        dx = dx_v - (closest_p - closest_p_prev)

        e0, e1 = build_orthonormal_basis(collision_normal)

        T = mat32(e0[0], e1[0], e0[1], e1[1], e0[2], e1[2])

        u = wp.transpose(T) * dx
        eps_U = friction_epsilon * dt

        friction_force, friction_hessian = compute_friction(friction_coefficient, -dEdD, T, u, eps_U)

        # fmt: off
        if wp.static("contact_force_hessian_vt" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "v: %d dEdD: %f\nnormal force: %f %f %f\nfriction force: %f %f %f\n",
                v,
                dEdD,
                collision_force[0], collision_force[1], collision_force[2], friction_force[0], friction_force[1], friction_force[2],
            )
        # fmt: on

        if v_order == 0:
            displacement = pos_prev[tri_indices[tri, 0]] - a
        elif v_order == 1:
            displacement = pos_prev[tri_indices[tri, 1]] - b
        elif v_order == 2:
            displacement = pos_prev[tri_indices[tri, 2]] - c
        else:
            displacement = pos_prev[v] - p

        collision_normal_sign = wp.vec4(-1.0, -1.0, -1.0, 1.0)
        if wp.dot(displacement, collision_normal * collision_normal_sign[v_order]) > 0:
            damping_hessian = (collision_damping / dt) * collision_hessian
            collision_hessian = collision_hessian + damping_hessian
            collision_force = collision_force + damping_hessian * displacement

        collision_force = collision_force + v_bary * friction_force
        collision_hessian = collision_hessian + v_bary * v_bary * friction_hessian
    else:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return collision_force, collision_hessian


@wp.func
def evaluate_vertex_triangle_collision_force_hessian_4_vertices(
    v: int,
    tri: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    dt: float,
):
    a = pos[tri_indices[tri, 0]]
    b = pos[tri_indices[tri, 1]]
    c = pos[tri_indices[tri, 2]]

    p = pos[v]

    closest_p, bary, feature_type = triangle_closest_point(a, b, c, p)

    diff = p - closest_p
    dis = wp.length(diff)
    collision_normal = diff / dis

    if 0.0 < dis < collision_radius:
        bs = wp.vec4(-bary[0], -bary[1], -bary[2], 1.0)

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

        collision_force = -dEdD * collision_normal
        collision_hessian = d2E_dDdD * wp.outer(collision_normal, collision_normal)

        # friction force
        dx_v = p - pos_prev[v]

        closest_p_prev = (
            bary[0] * pos_prev[tri_indices[tri, 0]]
            + bary[1] * pos_prev[tri_indices[tri, 1]]
            + bary[2] * pos_prev[tri_indices[tri, 2]]
        )

        dx = dx_v - (closest_p - closest_p_prev)

        e0, e1 = build_orthonormal_basis(collision_normal)

        T = mat32(e0[0], e1[0], e0[1], e1[1], e0[2], e1[2])

        u = wp.transpose(T) * dx
        eps_U = friction_epsilon * dt

        friction_force, friction_hessian = compute_friction(friction_coefficient, -dEdD, T, u, eps_U)

        # fmt: off
        if wp.static("contact_force_hessian_vt" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "v: %d dEdD: %f\nnormal force: %f %f %f\nfriction force: %f %f %f\n",
                v,
                dEdD,
                collision_force[0], collision_force[1], collision_force[2], friction_force[0], friction_force[1],
                friction_force[2],
            )
        # fmt: on

        displacement_0 = pos_prev[tri_indices[tri, 0]] - a
        displacement_1 = pos_prev[tri_indices[tri, 1]] - b
        displacement_2 = pos_prev[tri_indices[tri, 2]] - c
        displacement_3 = pos_prev[v] - p

        collision_force_0 = collision_force * bs[0]
        collision_force_1 = collision_force * bs[1]
        collision_force_2 = collision_force * bs[2]
        collision_force_3 = collision_force * bs[3]

        collision_hessian_0 = collision_hessian * bs[0] * bs[0]
        collision_hessian_1 = collision_hessian * bs[1] * bs[1]
        collision_hessian_2 = collision_hessian * bs[2] * bs[2]
        collision_hessian_3 = collision_hessian * bs[3] * bs[3]

        collision_normal_sign = wp.vec4(-1.0, -1.0, -1.0, 1.0)
        damping_force, damping_hessian = damp_collision(
            displacement_0,
            collision_normal * collision_normal_sign[0],
            collision_hessian_0,
            collision_damping,
            dt,
        )

        collision_force_0 += damping_force + bs[0] * friction_force
        collision_hessian_0 += damping_hessian + bs[0] * bs[0] * friction_hessian

        damping_force, damping_hessian = damp_collision(
            displacement_1,
            collision_normal * collision_normal_sign[1],
            collision_hessian_1,
            collision_damping,
            dt,
        )
        collision_force_1 += damping_force + bs[1] * friction_force
        collision_hessian_1 += damping_hessian + bs[1] * bs[1] * friction_hessian

        damping_force, damping_hessian = damp_collision(
            displacement_2,
            collision_normal * collision_normal_sign[2],
            collision_hessian_2,
            collision_damping,
            dt,
        )
        collision_force_2 += damping_force + bs[2] * friction_force
        collision_hessian_2 += damping_hessian + bs[2] * bs[2] * friction_hessian

        damping_force, damping_hessian = damp_collision(
            displacement_3,
            collision_normal * collision_normal_sign[3],
            collision_hessian_3,
            collision_damping,
            dt,
        )
        collision_force_3 += damping_force + bs[3] * friction_force
        collision_hessian_3 += damping_hessian + bs[3] * bs[3] * friction_hessian
        return (
            True,
            collision_force_0,
            collision_force_1,
            collision_force_2,
            collision_force_3,
            collision_hessian_0,
            collision_hessian_1,
            collision_hessian_2,
            collision_hessian_3,
        )
    else:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return (
            False,
            collision_force,
            collision_force,
            collision_force,
            collision_force,
            collision_hessian,
            collision_hessian,
            collision_hessian,
            collision_hessian,
        )


@wp.func
def compute_friction(mu: float, normal_contact_force: float, T: mat32, u: wp.vec2, eps_u: float):
    """
    Returns the 1D friction force and hessian.
    Args:
        mu: Friction coefficient.
        normal_contact_force: normal contact force.
        T: Transformation matrix (3x2 matrix).
        u: 2D displacement vector.
    """
    # Friction
    u_norm = wp.length(u)

    if u_norm > 0.0:
        # IPC friction
        if u_norm > eps_u:
            # constant stage
            f1_SF_over_x = 1.0 / u_norm
        else:
            # smooth transition
            f1_SF_over_x = (-u_norm / eps_u + 2.0) / eps_u

        force = -mu * normal_contact_force * T * (f1_SF_over_x * u)

        # Different from IPC, we treat the contact normal as constant
        # this significantly improves the stability
        hessian = mu * normal_contact_force * T * (f1_SF_over_x * wp.identity(2, float)) * wp.transpose(T)
    else:
        force = wp.vec3(0.0, 0.0, 0.0)
        hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return force, hessian


@wp.kernel
def forward_step(
    dt: float,
    gravity: wp.vec3,
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    external_force: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    inertia: wp.array(dtype=wp.vec3),
):
    particle = wp.tid()

    pos_prev[particle] = pos[particle]
    if not particle_flags[particle] & ParticleFlags.ACTIVE:
        inertia[particle] = pos_prev[particle]
        return
    vel_new = vel[particle] + (gravity + external_force[particle] * inv_mass[particle]) * dt
    pos[particle] = pos[particle] + vel_new * dt
    inertia[particle] = pos[particle]


@wp.kernel
def forward_step_penetration_free(
    dt: float,
    gravity: wp.vec3,
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    external_force: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
):
    particle_index = wp.tid()

    pos_prev[particle_index] = pos[particle_index]
    if not particle_flags[particle_index] & ParticleFlags.ACTIVE:
        inertia[particle_index] = pos_prev[particle_index]
        return
    vel_new = vel[particle_index] + (gravity + external_force[particle_index] * inv_mass[particle_index]) * dt
    pos_inertia = pos[particle_index] + vel_new * dt
    inertia[particle_index] = pos_inertia

    pos[particle_index] = apply_conservative_bound_truncation(
        particle_index, pos_inertia, pos_prev_collision_detection, particle_conservative_bounds
    )


@wp.kernel
def forward_step_rigid_bodies(
    dt: float,
    gravity: wp.vec3,
    body_q_prev: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_inertia_q: wp.array(dtype=wp.transform),
):
    """Forward integration step for rigid bodies in VBD solver.

    This kernel integrates rigid body motion using semi-implicit Euler integration.
    Only the inertial transforms are stored following VBD pattern (velocities computed later).
    No damping is applied, consistent with particle forward_step.

    Args:
        dt: Time step
        gravity: Gravity vector
        body_q_prev: Previous body transforms (output)
        body_q: Current body transforms
        body_qd: Current body velocities
        body_f: External forces on bodies
        body_com: Center of mass offsets
        body_inertia: Inertia tensors
        body_inv_mass: Inverse masses
        body_inv_inertia: Inverse inertia tensors
        body_inertia_q: Inertial body transforms (output)
    """
    body_index = wp.tid()

    # Store previous transform
    body_q_prev[body_index] = body_q[body_index]

    # Skip kinematic bodies (zero inverse mass)
    if body_inv_mass[body_index] == 0.0:
        body_inertia_q[body_index] = body_q[body_index]
        return

    # Integrate rigid body motion using base solver function
    q_new, _ = integrate_rigid_body(
        body_q[body_index],
        body_qd[body_index],
        body_f[body_index],
        body_com[body_index],
        body_inertia[body_index],
        body_inv_mass[body_index],
        body_inv_inertia[body_index],
        gravity,
        0.0,
        dt,
    )

    body_q[body_index] = q_new  # Update current transform
    body_inertia_q[body_index] = q_new  # Set inertial target


@wp.func
def sym33(a: wp.mat33) -> wp.mat33:
    # Force exact symmetry (guards tiny numeric asymmetries)
    return wp.mat33(
        0.5 * (a[0, 0] + a[0, 0]),
        0.5 * (a[0, 1] + a[1, 0]),
        0.5 * (a[0, 2] + a[2, 0]),
        0.5 * (a[1, 0] + a[0, 1]),
        0.5 * (a[1, 1] + a[1, 1]),
        0.5 * (a[1, 2] + a[2, 1]),
        0.5 * (a[2, 0] + a[0, 2]),
        0.5 * (a[2, 1] + a[1, 2]),
        0.5 * (a[2, 2] + a[2, 2]),
    )


@wp.func
def chol33_lower(A: wp.mat33) -> wp.mat33:
    # Cholesky factorization A = L * L^T, where L is lower-triangular (SPD assumed)
    a00 = A[0, 0]
    a10 = A[1, 0]
    a11 = A[1, 1]
    a20 = A[2, 0]
    a21 = A[2, 1]
    a22 = A[2, 2]

    L00 = wp.sqrt(a00)
    L10 = a10 / L00
    L20 = a20 / L00

    L11 = wp.sqrt(a11 - L10 * L10)
    L21 = (a21 - L20 * L10) / L11

    L22 = wp.sqrt(a22 - L20 * L20 - L21 * L21)

    # Return L (lower-triangular in a mat33)
    return wp.mat33(L00, 0.0, 0.0, L10, L11, 0.0, L20, L21, L22)


@wp.func
def solve_chol33(L: wp.mat33, b: wp.vec3) -> wp.vec3:
    # Solve (L L^T) x = b, using forward/back substitution

    # Forward: L y = b
    y0 = b[0] / L[0, 0]
    y1 = (b[1] - L[1, 0] * y0) / L[1, 1]
    y2 = (b[2] - L[2, 0] * y0 - L[2, 1] * y1) / L[2, 2]

    # Back: L^T x = y
    x2 = y2 / L[2, 2]
    x1 = (y1 - L[2, 1] * x2) / L[1, 1]
    x0 = (y0 - L[1, 0] * x1 - L[2, 0] * x2) / L[0, 0]

    return wp.vec3(x0, x1, x2)


@wp.kernel
def solve_rigid_body(
    dt: float,
    body_ids_in_color: wp.array(dtype=wp.int32),
    body_q_prev: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    body_q_rest: wp.array(dtype=wp.transform),
    body_mass: wp.array(dtype=float),
    body_inv_mass: wp.array(dtype=float),
    body_inertia: wp.array(dtype=wp.mat33),
    body_inertia_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    adjacency: ForceElementAdjacencyInfo,
    # joint data
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_target: wp.array(dtype=float),
    external_forces: wp.array(dtype=wp.vec3),
    external_torques: wp.array(dtype=wp.vec3),
    external_hessian_ll: wp.array(dtype=wp.mat33),  # Linear-linear from collision
    external_hessian_al: wp.array(dtype=wp.mat33),  # Angular-linear from collision
    external_hessian_aa: wp.array(dtype=wp.mat33),  # Angular-angular from collision
    # output
    body_q_new: wp.array(dtype=wp.transform),
):
    """VBD solve step for rigid bodies.
    Assembles joint and collision contributions to update body poses for the current color group.

    Args:
        dt: Time step
        body_ids_in_color: Array of body indices in current color group
        body_q_prev: Previous body transforms
        body_q: Current body transforms
        body_mass: Body masses
        body_inv_mass: Inverse body masses (for kinematic body detection)
        body_inertia: Body inertia tensors
        body_inertia_q: Inertial target transforms
        body_com: Center of mass offsets
        adjacency: Force element adjacency information
        [joint parameters]: Joint configuration and state data
        external_forces: External forces from collisions
        external_torques: External torques from collisions
        external_hessian_ll: Linear-linear collision Hessian
        external_hessian_al: Angular-linear collision Hessian
        external_hessian_aa: Angular-angular collision Hessian
        body_q_new: Output updated body transforms
    """
    tid = wp.tid()
    body_index = body_ids_in_color[tid]

    # Skip kinematic bodies (zero inverse mass)
    if body_inv_mass[body_index] == 0.0:
        body_q_new[body_index] = body_q[body_index]
        return

    # Inertia force and hessian
    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # Inertial transforms
    q_inertial = body_inertia_q[body_index]
    q_current = body_q[body_index]

    # Current and target pose
    pos_current = wp.transform_get_translation(q_current)
    rot_current = wp.transform_get_rotation(q_current)
    pos_star = wp.transform_get_translation(q_inertial)
    rot_star = wp.transform_get_rotation(q_inertial)
    body_com_local = body_com[body_index]

    m = body_mass[body_index]

    # Linear inertial force / Hessian
    # Apply inertial forces to center of mass
    com_current = pos_current + wp.quat_rotate(rot_current, body_com_local)
    com_star = pos_star + wp.quat_rotate(rot_star, body_com_local)

    inertial_coeff = m * dt_sqr_reciprocal
    f_lin = (com_star - com_current) * inertial_coeff

    # Angular inertial force / Hessian
    dq = wp.mul(wp.quat_inverse(rot_current), rot_star)

    # Enforce shortest arc
    if dq[3] < 0.0:
        dq = wp.quat(-dq[0], -dq[1], -dq[2], -dq[3])

    # Rotational dynamics
    if not USE_EXACT_ROTATIONAL_DYNAMICS:
        v = wp.vec3(dq[0], dq[1], dq[2])
        theta_body = 2.0 * v
    else:
        v = wp.vec3(dq[0], dq[1], dq[2])
        angle = wp.length(v)
        w_scalar = dq[3]
        if angle > 1.0e-12:
            # dq.w >= 0 after shortest-arc enforcement
            theta_magnitude = 2.0 * wp.atan2(angle, w_scalar)
            theta_body = theta_magnitude * (v / angle)
        else:
            theta_body = 2.0 * v

    I_body = body_inertia[body_index]
    tau_body = I_body * (theta_body * dt_sqr_reciprocal)
    tau_world = wp.quat_rotate(rot_current, tau_body)

    # Build world angular Hessian matrix
    R = wp.quat_to_matrix(rot_current)
    ex = wp.vec3(R[0, 0], R[1, 0], R[2, 0])
    ey = wp.vec3(R[0, 1], R[1, 1], R[2, 1])
    ez = wp.vec3(R[0, 2], R[1, 2], R[2, 2])

    # For capsule/cylinder: ex and ey axes have same inertia (Ia), ez axis has different inertia (Ib)
    Ia = I_body[0, 0]  # Inertia about X,Y axes (perpendicular to capsule)
    Ib = I_body[2, 2]  # Inertia about Z axis (along capsule length)
    angular_hessian = dt_sqr_reciprocal * (Ia * (wp.outer(ex, ex) + wp.outer(ey, ey)) + Ib * wp.outer(ez, ez))

    # Initialize accumulators with inertial terms and pre-add external collision contributions; joints are added next.
    f_torque = external_torques[body_index] + tau_world
    f_force = external_forces[body_index] + f_lin
    h_aa = external_hessian_aa[body_index] + angular_hessian
    h_al = external_hessian_al[body_index]
    I3 = wp.identity(3, float)
    h_ll = external_hessian_ll[body_index] + inertial_coeff * I3

    # Joint forces and Hessians
    num_adj_joints = get_body_num_adjacent_joints(adjacency, body_index)
    for joint_counter in range(num_adj_joints):
        joint_idx = get_body_adjacent_joint_id(adjacency, body_index, joint_counter)
        joint_force, joint_torque, joint_H_ll, joint_H_al, joint_H_aa = evaluate_joint_force_hessian(
            body_index,
            joint_idx,
            body_q,
            body_q_rest,
            body_com,
            joint_type,
            joint_parent,
            joint_child,
            joint_X_p,
            joint_X_c,
            joint_qd_start,
            joint_target_ke,
        )

        # Accumulate joint contributions
        f_force = f_force + joint_force
        f_torque = f_torque + joint_torque
        h_ll = h_ll + joint_H_ll
        h_al = h_al + joint_H_al
        h_aa = h_aa + joint_H_aa

    # VBD solve

    # Regularization for numerical stability (trace-scaled)
    trM = (h_ll[0, 0] + h_ll[1, 1] + h_ll[2, 2]) / 3.0
    trA = (h_aa[0, 0] + h_aa[1, 1] + h_aa[2, 2]) / 3.0
    epsM = 1.0e-6 * (trM + 1.0)
    epsA = 1.0e-6 * (trA + 1.0)

    M_reg = h_ll + epsM * I3
    A_reg = h_aa + epsA * I3

    # Cholesky factorization of M_reg (SPD)
    Lm = chol33_lower(M_reg)

    # MinvF := (M_reg)^{-1} F  via chol solve
    MinvF = solve_chol33(Lm, f_force)

    # Minv * C^T (three solves with the same Lm)
    # rows of C are columns of C^T
    C_r0 = wp.vec3(h_al[0, 0], h_al[0, 1], h_al[0, 2])
    C_r1 = wp.vec3(h_al[1, 0], h_al[1, 1], h_al[1, 2])
    C_r2 = wp.vec3(h_al[2, 0], h_al[2, 1], h_al[2, 2])

    X0 = solve_chol33(Lm, C_r0)  # column 0 of MinvCt
    X1 = solve_chol33(Lm, C_r1)  # column 1
    X2 = solve_chol33(Lm, C_r2)  # column 2

    MinvCt = wp.mat33(X0[0], X1[0], X2[0], X0[1], X1[1], X2[1], X0[2], X1[2], X2[2])

    # Schur complement: S = A_reg - C * Minv * C^T
    S = A_reg - (h_al * MinvCt)

    # Regularization
    trS = (S[0, 0] + S[1, 1] + S[2, 2]) / 3.0
    epsS = 1.0e-9 * (trS + 1.0)
    S = S + epsS * I3

    # Cholesky of S
    Ls = chol33_lower(S)

    rhs_w = f_torque - (h_al * MinvF)

    # Solve S * w_world = rhs_w
    w_world = solve_chol33(Ls, rhs_w)

    # x_inc = Minv * (F - C^T * w_world)
    Ct_w = wp.vec3(
        h_al[0, 0] * w_world[0] + h_al[1, 0] * w_world[1] + h_al[2, 0] * w_world[2],
        h_al[0, 1] * w_world[0] + h_al[1, 1] * w_world[1] + h_al[2, 1] * w_world[2],
        h_al[0, 2] * w_world[0] + h_al[1, 2] * w_world[1] + h_al[2, 2] * w_world[2],
    )
    x_inc = solve_chol33(Lm, f_force - Ct_w)

    ang_mag = wp.length(w_world)

    if USE_EXACT_ROTATIONAL_DYNAMICS:
        if ang_mag > 1.0e-12:
            dq_world = wp.quat_from_axis_angle(w_world / ang_mag, ang_mag)
            rot_new = wp.mul(dq_world, rot_current)
        else:
            rot_new = rot_current
    else:
        half_w = w_world * 0.5
        dq_world = wp.quat(half_w[0], half_w[1], half_w[2], 1.0)
        dq_world = wp.normalize(dq_world)
        rot_new = wp.mul(dq_world, rot_current)

    rot_new = wp.normalize(rot_new)

    com_new = com_current + x_inc
    pos_new = com_new - wp.quat_rotate(rot_new, body_com[body_index])

    body_q_new[body_index] = wp.transform(pos_new, rot_new)


@wp.kernel
def compute_particle_conservative_bound(
    # inputs
    conservative_bound_relaxation: float,
    collision_query_radius: float,
    adjacency: ForceElementAdjacencyInfo,
    collision_info: TriMeshCollisionInfo,
    # outputs
    particle_conservative_bounds: wp.array(dtype=float),
):
    particle_index = wp.tid()
    min_dist = wp.min(collision_query_radius, collision_info.vertex_colliding_triangles_min_dist[particle_index])

    # bound from neighbor triangles
    for i_adj_tri in range(
        get_vertex_num_adjacent_faces(
            adjacency,
            particle_index,
        )
    ):
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(
            adjacency,
            particle_index,
            i_adj_tri,
        )
        min_dist = wp.min(min_dist, collision_info.triangle_colliding_vertices_min_dist[tri_index])

    # bound from neighbor edges
    for i_adj_edge in range(
        get_vertex_num_adjacent_edges(
            adjacency,
            particle_index,
        )
    ):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(
            adjacency,
            particle_index,
            i_adj_edge,
        )
        # vertex is on the edge; otherwise it only effects the bending energy
        if vertex_order_on_edge == 2 or vertex_order_on_edge == 3:
            # collisions of neighbor edges
            min_dist = wp.min(min_dist, collision_info.edge_colliding_edges_min_dist[nei_edge_index])

    particle_conservative_bounds[particle_index] = conservative_bound_relaxation * min_dist


@wp.kernel
def validate_conservative_bound(
    pos: wp.array(dtype=wp.vec3),
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
):
    v_index = wp.tid()

    displacement = wp.length(pos[v_index] - pos_prev_collision_detection[v_index])

    if displacement > particle_conservative_bounds[v_index] * 1.01 and displacement > 1e-5:
        # wp.expect_eq(displacement <= particle_conservative_bounds[v_index] * 1.01, True)
        wp.printf(
            "Vertex %d has moved by %f exceeded the limit of %f\n",
            v_index,
            displacement,
            particle_conservative_bounds[v_index],
        )


@wp.func
def apply_conservative_bound_truncation(
    v_index: wp.int32,
    pos_new: wp.vec3,
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
):
    particle_pos_prev_collision_detection = pos_prev_collision_detection[v_index]
    accumulated_displacement = pos_new - particle_pos_prev_collision_detection
    conservative_bound = particle_conservative_bounds[v_index]

    accumulated_displacement_norm = wp.length(accumulated_displacement)
    if accumulated_displacement_norm > conservative_bound and conservative_bound > 1e-5:
        accumulated_displacement_norm_truncated = conservative_bound
        accumulated_displacement = accumulated_displacement * (
            accumulated_displacement_norm_truncated / accumulated_displacement_norm
        )

        return particle_pos_prev_collision_detection + accumulated_displacement
    else:
        return pos_new


@wp.kernel
def solve_trimesh_no_self_contact_tile(
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=float, ndim=2),
    tri_areas: wp.array(dtype=float),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angles: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    edge_bending_properties: wp.array(dtype=float, ndim=2),
    adjacency: ForceElementAdjacencyInfo,
    # contact info
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
    # output
    pos_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    block_idx = tid // TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
    thread_idx = tid % TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
    particle_index = particle_ids_in_color[block_idx]

    if not particle_flags[particle_index] & ParticleFlags.ACTIVE:
        if thread_idx == 0:
            pos_new[particle_index] = pos[particle_index]
        return

    particle_pos = pos[particle_index]

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # # inertia force and hessian
    # f = mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
    # h = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)

    f = wp.vec3(0.0)
    h = wp.mat33(0.0)

    num_adj_faces = get_vertex_num_adjacent_faces(adjacency, particle_index)

    batch_counter = wp.int32(0)

    # loop through all the adjacent triangles using whole block
    while batch_counter + thread_idx < num_adj_faces:
        adj_tri_counter = thread_idx + batch_counter
        batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
        # elastic force and hessian
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, adj_tri_counter)

        f_tri, h_tri = evaluate_stvk_force_hessian(
            tri_index,
            vertex_order,
            pos,
            pos_prev,
            tri_indices,
            tri_poses[tri_index],
            tri_areas[tri_index],
            tri_materials[tri_index, 0],
            tri_materials[tri_index, 1],
            tri_materials[tri_index, 2],
            dt,
        )
        # compute damping

        f += f_tri
        h += h_tri

        # fmt: off
        if wp.static("elasticity_force_hessian" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "particle: %d, i_adj_tri: %d, particle_order: %d, \nforce:\n %f %f %f, \nhessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
                particle_index,
                thread_idx,
                vertex_order,
                f[0], f[1], f[2], h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2],
            )
            # fmt: on

    #
    batch_counter = wp.int32(0)
    num_adj_edges = get_vertex_num_adjacent_edges(adjacency, particle_index)
    while batch_counter + thread_idx < num_adj_edges:
        adj_edge_counter = batch_counter + thread_idx
        batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(
            adjacency, particle_index, adj_edge_counter
        )
        if edge_bending_properties[nei_edge_index, 0] != 0.0:
            f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
                nei_edge_index,
                vertex_order_on_edge,
                pos,
                pos_prev,
                edge_indices,
                edge_rest_angles,
                edge_rest_length,
                edge_bending_properties[nei_edge_index, 0],
                edge_bending_properties[nei_edge_index, 1],
                dt,
            )

            f += f_edge
            h += h_edge

    f_tile = wp.tile(f, preserve_type=True)
    h_tile = wp.tile(h, preserve_type=True)

    f_total = wp.tile_reduce(wp.add, f_tile)[0]
    h_total = wp.tile_reduce(wp.add, h_tile)[0]

    if thread_idx == 0:
        h_total = (
            h_total
            + mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)
            + particle_hessians[particle_index]
        )
        if abs(wp.determinant(h_total)) > 1e-5:
            h_inv = wp.inverse(h_total)
            f_total = (
                f_total
                + mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
                + particle_forces[particle_index]
            )

            pos_new[particle_index] = particle_pos + h_inv * f_total


@wp.kernel
def solve_trimesh_no_self_contact(
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=float, ndim=2),
    tri_areas: wp.array(dtype=float),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angles: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    edge_bending_properties: wp.array(dtype=float, ndim=2),
    adjacency: ForceElementAdjacencyInfo,
    # contact info
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
    # output
    pos_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    particle_index = particle_ids_in_color[tid]

    if not particle_flags[particle_index] & ParticleFlags.ACTIVE:
        pos_new[particle_index] = pos[particle_index]
        return

    particle_pos = pos[particle_index]

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # inertia force and hessian
    f = mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
    h = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)

    # elastic force and hessian
    for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
        tri_id, particle_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)

        # fmt: off
        if wp.static("connectivity" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "particle: %d | num_adj_faces: %d | ",
                particle_index,
                get_vertex_num_adjacent_faces(particle_index, adjacency),
            )
            wp.printf("i_face: %d | face id: %d | v_order: %d | ", i_adj_tri, tri_id, particle_order)
            wp.printf(
                "face: %d %d %d\n",
                tri_indices[tri_id, 0],
                tri_indices[tri_id, 1],
                tri_indices[tri_id, 2],
            )
        # fmt: on

        f_tri, h_tri = evaluate_stvk_force_hessian(
            tri_id,
            particle_order,
            pos,
            pos_prev,
            tri_indices,
            tri_poses[tri_id],
            tri_areas[tri_id],
            tri_materials[tri_id, 0],
            tri_materials[tri_id, 1],
            tri_materials[tri_id, 2],
            dt,
        )

        f = f + f_tri
        h = h + h_tri

        # fmt: off
        if wp.static("elasticity_force_hessian" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "particle: %d, i_adj_tri: %d, particle_order: %d, \nforce:\n %f %f %f, \nhessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
                particle_index,
                i_adj_tri,
                particle_order,
                f[0], f[1], f[2], h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2],
            )
        # fmt: on

    for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_index, i_adj_edge)
        if edge_bending_properties[nei_edge_index, 0] != 0.0:
            f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
                nei_edge_index,
                vertex_order_on_edge,
                pos,
                pos_prev,
                edge_indices,
                edge_rest_angles,
                edge_rest_length,
                edge_bending_properties[nei_edge_index, 0],
                edge_bending_properties[nei_edge_index, 1],
                dt,
            )

            f += f_edge
            h += h_edge

    h += particle_hessians[particle_index]
    f += particle_forces[particle_index]

    if abs(wp.determinant(h)) > 1e-5:
        hInv = wp.inverse(h)
        pos_new[particle_index] = particle_pos + hInv * f


@wp.kernel
def copy_particle_positions_back(
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    pos_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    particle = particle_ids_in_color[tid]

    pos[particle] = pos_new[particle]


@wp.kernel
def copy_rigid_body_transforms_back(
    body_ids_in_color: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_q_new: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    body_index = body_ids_in_color[tid]

    body_q[body_index] = body_q_new[body_index]


@wp.kernel
def update_particle_velocity(
    dt: float, pos_prev: wp.array(dtype=wp.vec3), pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3)
):
    particle = wp.tid()
    vel[particle] = (pos[particle] - pos_prev[particle]) / dt


@wp.kernel
def update_body_velocity(
    dt: float,
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    body_index = wp.tid()

    pose = body_q[body_index]
    pose_prev = body_q_prev[body_index]

    x = wp.transform_get_translation(pose)
    x_prev = wp.transform_get_translation(pose_prev)

    q = wp.transform_get_rotation(pose)
    q_prev = wp.transform_get_rotation(pose_prev)

    # Compute COM positions at current and previous time
    x_com = x + wp.quat_rotate(q, body_com[body_index])
    x_com_prev = x_prev + wp.quat_rotate(q_prev, body_com[body_index])

    # Linear velocity of COM: v = (pos - pos_prev) / dt
    v = (x_com - x_com_prev) / dt

    # Angular velocity from quaternion difference
    dq = q * wp.quat_inverse(q_prev)
    dq = wp.normalize(dq)

    # Enforce shortest arc (ensure scalar part non-negative) for both paths
    if dq[3] < 0.0:
        dq = wp.quat(-dq[0], -dq[1], -dq[2], -dq[3])

    if USE_EXACT_ROTATIONAL_DYNAMICS:
        # Exact axis-angle extraction using atan2 for robust angle computation
        v_part = wp.vec3(dq[0], dq[1], dq[2])
        w_scalar = dq[3]
        v_norm = wp.length(v_part)
        if v_norm > 1.0e-12:
            theta = 2.0 * wp.atan2(v_norm, w_scalar)
            omega = (theta / dt) * (v_part / v_norm)
        else:
            # Small angle fallback: omega approx 2/dt * v_part
            omega = (2.0 / dt) * v_part
    else:
        # Small-angle approximation: omega approx 2/dt * dq.xyz
        omega = (2.0 / dt) * wp.vec3(dq[0], dq[1], dq[2])

    body_qd[body_index] = wp.spatial_vector(v, omega)


@wp.func
def evaluate_cable_stretch_force_hessian(
    X_wp: wp.transform,
    X_wc: wp.transform,
    com_p: wp.vec3,
    com_c: wp.vec3,
    pose_p: wp.transform,
    pose_c: wp.transform,
    is_parent: bool,
    ke: float,
):
    """
    Compute stretch/shear force and Hessian for cable joints.

    Energy: U = 0.5 * ke * ||e||^2, with e = x_c - x_p.

    Args:
        X_wp, X_wc: Pre-computed joint frames in world coordinates (optimization)
        com_p, com_c: Center-of-mass positions for parent and child (in body coords)
        pose_p, pose_c: Body poses (needed for COM world transforms)
        is_parent: True if computing for parent body, False for child
        ke: Stretch stiffness

    Returns:
        tuple: (force, torque, H_ll, H_al, H_aa) - VBD force and Hessian components
            force: Linear force component [3x1]
            torque: Angular force component [3x1]
            H_ll: Linear-linear Hessian block [3x3]
            H_al: Angular-linear coupling Hessian [3x3]
            H_aa: Angular-angular Hessian block [3x3]
    """
    if ke <= 0.0:
        return wp.vec3(), wp.vec3(), wp.mat33(), wp.mat33(), wp.mat33()

    # Attachment points in world coordinates
    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)

    # Residual e = x_c - x_p (child attachment - parent attachment)
    e = x_c - x_p

    # Force (-grad U) per body
    # dU/dx_c = ke * e,  dU/dx_p = -ke * e  =>  f_parent = +ke * e,  f_child = -ke * e
    if is_parent:
        # Parent moment arm (from COM to attachment)
        com_p_w = wp.transform_point(pose_p, com_p)
        r = x_p - com_p_w
        f_lin = ke * e
    else:
        # Child moment arm (from COM to attachment)
        com_c_w = wp.transform_point(pose_c, com_c)
        r = x_c - com_c_w
        f_lin = -ke * e

    torque = wp.cross(r, f_lin)
    force = f_lin

    # Hessian components (Gauss-Newton: H = ke * J^T J) per body
    # Absolute per-body contact-point Jacobian used for Hessian:
    #   J_abs = ( -[r]_x , +I )   # maps body twist [omega; v] to point motion
    # Per-body block: H_i = J_abs^T * (ke * I) * J_abs
    rx = wp.skew(r)

    I3 = wp.identity(3, float)

    # Linear-Linear block
    H_ll = ke * I3

    # Angular-Linear coupling
    H_al = ke * rx

    # Angular-Angular block: [r]_x^T [r]_x = ||r||^2 I - r r^T
    r2 = wp.dot(r, r)
    rrT = wp.outer(r, r)
    H_aa = ke * (r2 * I3 - rrT)

    return force, torque, H_ll, H_al, H_aa


@wp.func
def compute_right_jacobian_inverse(kappa: wp.vec3) -> wp.mat33:
    """
    Compute the right Jacobian inverse of SO(3) for rotation vector kappa.

    The right Jacobian relates rotation vector variations to angular velocity
    in the tangent space of SO(3). For rotational dynamics with energy
    E = 0.5 * kappa^T * K * kappa, the force is tau = Jr_inv^T * (K * kappa).

    Mathematical formulation:
    - Small angles: Jr_inv = I + 0.5*[kappa]_x + (1/12)*[kappa]_x^2 + O(theta^4)
    - General case: Jr_inv = I + 0.5*[kappa]_x + b*[kappa]_x^2
      where b = (1/theta^2) - (1+cos(theta))/(2*theta*sin(theta))
      and theta = ||kappa|| is the rotation angle

    Args:
        kappa: Rotation vector (axis * angle) [3x1]

    Returns:
        3x3 right Jacobian inverse matrix
    """
    theta = wp.length(kappa)
    I3 = wp.identity(3, float)

    if theta < 1.0e-7:
        # Small angle series expansion for numerical stability
        kappa_skew = wp.skew(kappa)
        kappa_skew2 = kappa_skew * kappa_skew
        # Taylor series: I + (1/2)*[kappa]_x + (1/12)*[kappa]_x^2 + O(theta^4)
        return I3 + 0.5 * kappa_skew + (1.0 / 12.0) * kappa_skew2

    # Full formula for general rotations
    kappa_skew = wp.skew(kappa)
    kappa_skew2 = kappa_skew * kappa_skew
    sin_theta = wp.sin(theta)
    cos_theta = wp.cos(theta)

    # Coefficients for Jr_inv = I + a*[kappa]_x + b*[kappa]_x^2
    a = 0.5  # Coefficient for linear term
    b = (1.0 / (theta * theta)) - (1.0 + cos_theta) / (2.0 * theta * sin_theta)

    return I3 + a * kappa_skew + b * kappa_skew2


@wp.func
def evaluate_cable_bend_force_hessian(
    q_wp: wp.quat,
    q_wc: wp.quat,
    q_wp_rest: wp.quat,
    q_wc_rest: wp.quat,
    is_parent: bool,
    ke: float,
):
    """
    Compute bending force and Hessian for cable/rod rotational constraints.

    Cable joints enforce rotational constraints between joint frames on two bodies.
    This implements a pure rotational constraint with isotropic angular stiffness,
    resisting both twist (rotation about cable axis) and bend (rotation about
    perpendicular axes) with equal stiffness.

    Energy model: U = 0.5 * ke * ||kappa||^2 where kappa is the rotation vector
    Forces: tau = -grad(U) with respect to body orientations

    Args:
        q_wp: Current parent joint frame rotation in world coordinates
        q_wc: Current child joint frame rotation in world coordinates
        q_wp_rest: Rest parent joint frame rotation in world coordinates
        q_wc_rest: Rest child joint frame rotation in world coordinates
        is_parent: True for parent body computation, False for child body
        ke: Angular stiffness coefficient [torque/angle]

    Returns:
        tuple: (torque, H_aa) - Force and Hessian blocks for VBD
            torque: Angular force component [3x1]
            H_aa: Angular-angular Hessian block [3x3]

    Note:
        Pure rotational constraint - only returns angular components.
        Linear components are always zero and not included for efficiency.
    """
    # Initialize only non-zero components
    torque = wp.vec3(0.0)
    H_aa = wp.mat33(0.0)

    if ke <= 0.0:
        return torque, H_aa

    # Compare current state against actual rest configuration
    # Current relative rotation between joint frames
    r_current = wp.mul(wp.quat_inverse(q_wp), q_wc)
    # Rest relative rotation between joint frames (captured at initialization)
    r_rest = wp.mul(wp.quat_inverse(q_wp_rest), q_wc_rest)
    # Constraint violation = change from rest relative rotation
    relative_quat = wp.mul(r_current, wp.quat_inverse(r_rest))

    # Ensure shortest arc representation for numerical stability
    if relative_quat[3] < 0.0:
        relative_quat = wp.quat(-relative_quat[0], -relative_quat[1], -relative_quat[2], -relative_quat[3])
    relative_quat = wp.normalize(relative_quat)

    # Convert quaternion to rotation vector (axis-angle), robust around zero
    # For quaternion q = [qx, qy, qz, qw], rotation vector kappa = axis * angle
    quat_vec = wp.vec3(relative_quat[0], relative_quat[1], relative_quat[2])
    v_norm = wp.length(quat_vec)
    w_scalar = relative_quat[3]
    # Threshold for small-angle handling
    if v_norm < 1.0e-9:
        # Small-angle: kappa ~= 2 * v (since v ~= axis * theta/2)
        kappa_local = quat_vec * 2.0
    else:
        theta = 2.0 * wp.atan2(v_norm, w_scalar)
        if theta < 1.0e-9:
            kappa_local = quat_vec * 2.0
        else:
            axis_local = quat_vec / v_norm
            kappa_local = axis_local * theta

    # Compute right Jacobian inverse for accurate rotational dynamics
    if USE_EXACT_ROTATIONAL_DYNAMICS:
        Jr_inv = compute_right_jacobian_inverse(kappa_local)
    else:
        Jr_inv = wp.identity(3, float)  # Small-angle approximation

    # Compute torque in local coordinates using VBD formulation
    # Energy: U = 0.5 * kappa^T * K * kappa with K = ke * I (isotropic stiffness)
    # Force: tau = Jr_inv^T * grad_kappa(U) = Jr_inv^T * (K * kappa) = Jr_inv^T * (ke * kappa)
    gradient_local = kappa_local * ke  # grad_kappa(U) = K * kappa = ke * kappa
    tau_local = wp.transpose(Jr_inv) * gradient_local  # tau = Jr_inv^T * grad_kappa(U)

    # Transform torque from parent joint frame to world coordinates
    R_wp = wp.quat_to_matrix(q_wp)  # Parent joint frame rotation matrix
    tau_world = R_wp * tau_local

    # Apply sign based on parent/child relationship
    if not is_parent:
        tau_world = -tau_world

    torque = tau_world

    # Local SPD angular Hessian: H_local = ke * (Jr_inv^T * Jr_inv)
    Jr_inv_local = Jr_inv
    H_local = ke * (wp.transpose(Jr_inv_local) * Jr_inv_local)

    # World Hessian: H_aa = R_wp * H_local * R_wp^T
    H_aa = (R_wp * H_local) * wp.transpose(R_wp)

    return torque, H_aa


@wp.func
def evaluate_joint_force_hessian(
    body_index: int,
    joint_index: int,
    body_q: wp.array(dtype=wp.transform),
    body_q_rest: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_qd_start: wp.array(dtype=int),
    joint_target_ke: wp.array(dtype=float),
):
    """
    Evaluate force and Hessian contributions from a joint constraint on a specific body.


    Args:
        body_index: Index of the body to compute forces for
        joint_index: Index of the joint constraint
        body_q: Current body poses [transforms]
        body_com: Body centers of mass in local coordinates [vec3]
        joint_type: Joint type identifiers [int]
        joint_parent: Parent body indices for each joint [int]
        joint_child: Child body indices for each joint [int]
        joint_X_p: Parent joint frame transforms in local coordinates [transforms]
        joint_X_c: Child joint frame transforms in local coordinates [transforms]
        joint_qd_start: Starting index for joint DOFs [int]
        joint_target_ke: Joint stiffness parameters [float]

    Returns:
        tuple: (force, torque, H_ll, H_al, H_aa) - VBD force and Hessian components
            force: Linear force component [3x1]
            torque: Angular force component [3x1]
            H_ll: Linear-linear Hessian block [3x3]
            H_al: Angular-linear coupling Hessian [3x3]
            H_aa: Angular-angular Hessian block [3x3]
    """
    # Initialize force and Hessian components to zero (for early exits and accumulation)
    total_force = wp.vec3(0.0)
    total_torque = wp.vec3(0.0)
    total_H_ll = wp.mat33(0.0)
    total_H_al = wp.mat33(0.0)
    total_H_aa = wp.mat33(0.0)

    # Currently, only cable joints are supported
    if joint_type[joint_index] != JointType.CABLE:
        return total_force, total_torque, total_H_ll, total_H_al, total_H_aa

    # Get parent and child body indices for this joint
    parent_index = joint_parent[joint_index]
    child_index = joint_child[joint_index]

    # Only compute forces for bodies directly connected to this joint
    is_parent_body = body_index == parent_index
    if body_index != parent_index and body_index != child_index:
        return total_force, total_torque, total_H_ll, total_H_al, total_H_aa

    # Skip joints without valid parent (joint requires parent-child hierarchy)
    if parent_index < 0:
        return total_force, total_torque, total_H_ll, total_H_al, total_H_aa

    # Extract joint frame transforms in body-local coordinates
    X_pj = joint_X_p[joint_index]  # Parent joint frame in local coordinates
    X_cj = joint_X_c[joint_index]  # Child joint frame in local coordinates

    # Get current body poses and centers of mass
    parent_pose = body_q[parent_index]
    child_pose = body_q[child_index]
    parent_com = body_com[parent_index]
    child_com = body_com[child_index]

    # Transform joint frames to world coordinates
    X_wp = parent_pose * X_pj  # Parent joint frame in world
    X_wc = child_pose * X_cj  # Child joint frame in world

    # Compute stiffness parameters for cable constraints
    dof_start_index = joint_qd_start[joint_index]
    bend_stiffness = joint_target_ke[dof_start_index] if dof_start_index < joint_target_ke.shape[0] else 1000.0
    stretch_stiffness = wp.max(
        STRETCH_STIFFNESS_MIN, 100.0 * bend_stiffness
    )  # Stretch stiffness >> bend for cable rigidity

    # Compute stretch constraint forces and Hessians (position-based)
    stretch_force, stretch_torque, stretch_H_ll, stretch_H_al, stretch_H_aa = evaluate_cable_stretch_force_hessian(
        X_wp, X_wc, parent_com, child_com, parent_pose, child_pose, is_parent_body, stretch_stiffness
    )

    # Compute bend constraint forces and Hessians (rotation-based, only angular components)
    bend_torque = wp.vec3(0.0)
    bend_H_aa = wp.mat33(0.0)
    if bend_stiffness > 0.0:
        # Rest poses and rotations only needed if bend is active
        parent_pose_rest = body_q_rest[parent_index]
        child_pose_rest = body_q_rest[child_index]
        X_wp_rest = parent_pose_rest * X_pj  # Parent joint frame in rest world
        X_wc_rest = child_pose_rest * X_cj  # Child joint frame in rest world

        # Extract rotation quaternions for bend constraint computation
        q_wp = wp.transform_get_rotation(X_wp)  # current
        q_wc = wp.transform_get_rotation(X_wc)
        q_wp_rest = wp.transform_get_rotation(X_wp_rest)  # rest
        q_wc_rest = wp.transform_get_rotation(X_wc_rest)

        bend_torque, bend_H_aa = evaluate_cable_bend_force_hessian(
            q_wp, q_wc, q_wp_rest, q_wc_rest, is_parent_body, bend_stiffness
        )

    # Combine constraint contributions
    total_torque = stretch_torque + bend_torque  # Both constraints contribute torque
    total_force = stretch_force  # Only stretch contributes linear force
    total_H_aa = stretch_H_aa + bend_H_aa  # Both constraints contribute angular stiffness
    total_H_al = stretch_H_al  # Only stretch contributes angular-linear coupling
    total_H_ll = stretch_H_ll  # Only stretch contributes linear stiffness

    return total_force, total_torque, total_H_ll, total_H_al, total_H_aa


@wp.kernel
def convert_body_particle_contact_data_kernel(
    # inputs
    body_particle_contact_buffer_pre_alloc: int,
    soft_contact_particle: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_max: int,
    # outputs
    body_particle_contact_buffer: wp.array(dtype=int),
    body_particle_contact_count: wp.array(dtype=int),
):
    contact_index = wp.tid()
    count = min(contact_max, contact_count[0])
    if contact_index >= count:
        return

    particle_index = soft_contact_particle[contact_index]
    offset = particle_index * body_particle_contact_buffer_pre_alloc

    contact_counter = wp.atomic_add(body_particle_contact_count, particle_index, 1)
    if contact_counter < body_particle_contact_buffer_pre_alloc:
        body_particle_contact_buffer[offset + contact_counter] = contact_index


@wp.kernel
def accumulate_contact_force_and_hessian(
    # inputs
    dt: float,
    current_color: int,
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    particle_colors: wp.array(dtype=int),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    # self contact
    collision_info_array: wp.array(dtype=TriMeshCollisionInfo),
    collision_radius: float,
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    edge_edge_parallel_epsilon: float,
    # body-particle contact
    particle_radius: wp.array(dtype=float),
    soft_contact_particle: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_max: int,
    shape_material_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    # outputs: particle force and hessian
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
):
    t_id = wp.tid()
    collision_info = collision_info_array[0]

    primitive_id = t_id // NUM_THREADS_PER_COLLISION_PRIMITIVE
    t_id_current_primitive = t_id % NUM_THREADS_PER_COLLISION_PRIMITIVE

    # process edge-edge collisions
    if primitive_id < collision_info.edge_colliding_edges_buffer_sizes.shape[0]:
        e1_idx = primitive_id

        collision_buffer_counter = t_id_current_primitive
        collision_buffer_offset = collision_info.edge_colliding_edges_offsets[primitive_id]
        while collision_buffer_counter < collision_info.edge_colliding_edges_buffer_sizes[primitive_id]:
            e2_idx = collision_info.edge_colliding_edges[2 * (collision_buffer_offset + collision_buffer_counter) + 1]

            if e1_idx != -1 and e2_idx != -1:
                e1_v1 = edge_indices[e1_idx, 2]
                e1_v2 = edge_indices[e1_idx, 3]

                c_e1_v1 = particle_colors[e1_v1]
                c_e1_v2 = particle_colors[e1_v2]
                if c_e1_v1 == current_color or c_e1_v2 == current_color:
                    has_contact, collision_force_0, collision_force_1, collision_hessian_0, collision_hessian_1 = (
                        evaluate_edge_edge_contact_2_vertices(
                            e1_idx,
                            e2_idx,
                            pos,
                            pos_prev,
                            edge_indices,
                            collision_radius,
                            soft_contact_ke,
                            soft_contact_kd,
                            friction_mu,
                            friction_epsilon,
                            dt,
                            edge_edge_parallel_epsilon,
                        )
                    )

                    if has_contact:
                        # here we only handle the e1 side, because e2 will also detection this contact and add force and hessian on its own
                        if c_e1_v1 == current_color:
                            wp.atomic_add(particle_forces, e1_v1, collision_force_0)
                            wp.atomic_add(particle_hessians, e1_v1, collision_hessian_0)
                        if c_e1_v2 == current_color:
                            wp.atomic_add(particle_forces, e1_v2, collision_force_1)
                            wp.atomic_add(particle_hessians, e1_v2, collision_hessian_1)
            collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE

    # process vertex-triangle collisions
    if primitive_id < collision_info.vertex_colliding_triangles_buffer_sizes.shape[0]:
        particle_idx = primitive_id
        collision_buffer_counter = t_id_current_primitive
        collision_buffer_offset = collision_info.vertex_colliding_triangles_offsets[primitive_id]
        while collision_buffer_counter < collision_info.vertex_colliding_triangles_buffer_sizes[primitive_id]:
            tri_idx = collision_info.vertex_colliding_triangles[
                (collision_buffer_offset + collision_buffer_counter) * 2 + 1
            ]

            if particle_idx != -1 and tri_idx != -1:
                tri_a = tri_indices[tri_idx, 0]
                tri_b = tri_indices[tri_idx, 1]
                tri_c = tri_indices[tri_idx, 2]

                c_v = particle_colors[particle_idx]
                c_tri_a = particle_colors[tri_a]
                c_tri_b = particle_colors[tri_b]
                c_tri_c = particle_colors[tri_c]

                if (
                    c_v == current_color
                    or c_tri_a == current_color
                    or c_tri_b == current_color
                    or c_tri_c == current_color
                ):
                    (
                        has_contact,
                        collision_force_0,
                        collision_force_1,
                        collision_force_2,
                        collision_force_3,
                        collision_hessian_0,
                        collision_hessian_1,
                        collision_hessian_2,
                        collision_hessian_3,
                    ) = evaluate_vertex_triangle_collision_force_hessian_4_vertices(
                        particle_idx,
                        tri_idx,
                        pos,
                        pos_prev,
                        tri_indices,
                        collision_radius,
                        soft_contact_ke,
                        soft_contact_kd,
                        friction_mu,
                        friction_epsilon,
                        dt,
                    )

                    if has_contact:
                        # particle
                        if c_v == current_color:
                            wp.atomic_add(particle_forces, particle_idx, collision_force_3)
                            wp.atomic_add(particle_hessians, particle_idx, collision_hessian_3)

                        # tri_a
                        if c_tri_a == current_color:
                            wp.atomic_add(particle_forces, tri_a, collision_force_0)
                            wp.atomic_add(particle_hessians, tri_a, collision_hessian_0)

                        # tri_b
                        if c_tri_b == current_color:
                            wp.atomic_add(particle_forces, tri_b, collision_force_1)
                            wp.atomic_add(particle_hessians, tri_b, collision_hessian_1)

                        # tri_c
                        if c_tri_c == current_color:
                            wp.atomic_add(particle_forces, tri_c, collision_force_2)
                            wp.atomic_add(particle_hessians, tri_c, collision_hessian_2)
            collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE

    particle_body_contact_count = min(contact_max, contact_count[0])

    if t_id < particle_body_contact_count:
        particle_idx = soft_contact_particle[t_id]

        if particle_colors[particle_idx] == current_color:
            body_contact_force, body_contact_hessian = evaluate_body_particle_contact(
                particle_idx,
                pos[particle_idx],
                pos_prev[particle_idx],
                t_id,
                soft_contact_ke,
                soft_contact_kd,
                friction_mu,
                friction_epsilon,
                particle_radius,
                shape_material_mu,
                shape_body,
                body_q,
                body_q_prev,
                body_qd,
                body_com,
                contact_shape,
                contact_body_pos,
                contact_body_vel,
                contact_normal,
                dt,
            )
            wp.atomic_add(particle_forces, particle_idx, body_contact_force)
            wp.atomic_add(particle_hessians, particle_idx, body_contact_hessian)


@wp.func
def evaluate_spring_force_and_hessian(
    particle_idx: int,
    spring_idx: int,
    dt: float,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    spring_indices: wp.array(dtype=int),
    spring_rest_length: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
):
    v0 = spring_indices[spring_idx * 2]
    v1 = spring_indices[spring_idx * 2 + 1]

    diff = pos[v0] - pos[v1]
    l = wp.length(diff)
    l0 = spring_rest_length[spring_idx]

    force_sign = 1.0 if particle_idx == v0 else -1.0

    spring_force = force_sign * spring_stiffness[spring_idx] * (l0 - l) / l * diff
    spring_hessian = spring_stiffness[spring_idx] * (
        wp.identity(3, float) - (l0 / l) * (wp.identity(3, float) - wp.outer(diff, diff) / (l * l))
    )

    # compute damping
    h_d = spring_hessian * (spring_damping[spring_idx] / dt)

    f_d = h_d * (pos_prev[particle_idx] - pos[particle_idx])

    spring_force = spring_force + f_d
    spring_hessian = spring_hessian + h_d

    return spring_force, spring_hessian


@wp.kernel
def accumulate_spring_force_and_hessian(
    # inputs
    dt: float,
    current_color: int,
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    particle_ids_in_color: wp.array(dtype=int),
    adjacency: ForceElementAdjacencyInfo,
    # spring constraints
    spring_indices: wp.array(dtype=int),
    spring_rest_length: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
    # outputs: particle force and hessian
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
):
    t_id = wp.tid()

    particle_index = particle_ids_in_color[t_id]

    num_adj_springs = get_vertex_num_adjacent_springs(adjacency, particle_index)
    for spring_counter in range(num_adj_springs):
        spring_index = get_vertex_adjacent_spring_id(adjacency, particle_index, spring_counter)
        spring_force, spring_hessian = evaluate_spring_force_and_hessian(
            particle_index,
            spring_index,
            dt,
            pos,
            pos_prev,
            spring_indices,
            spring_rest_length,
            spring_stiffness,
            spring_damping,
        )

        particle_forces[particle_index] = particle_forces[particle_index] + spring_force
        particle_hessians[particle_index] = particle_hessians[particle_index] + spring_hessian


@wp.kernel
def accumulate_contact_force_and_hessian_no_self_contact(
    # inputs
    dt: float,
    current_color: int,
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    particle_colors: wp.array(dtype=int),
    # body-particle contact
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    particle_radius: wp.array(dtype=float),
    soft_contact_particle: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_max: int,
    shape_material_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    # outputs: particle force and hessian
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
):
    t_id = wp.tid()

    particle_body_contact_count = min(contact_max, contact_count[0])

    if t_id < particle_body_contact_count:
        particle_idx = soft_contact_particle[t_id]

        if particle_colors[particle_idx] == current_color:
            body_contact_force, body_contact_hessian = evaluate_body_particle_contact(
                particle_idx,
                pos[particle_idx],
                pos_prev[particle_idx],
                t_id,
                soft_contact_ke,
                soft_contact_kd,
                friction_mu,
                friction_epsilon,
                particle_radius,
                shape_material_mu,
                shape_body,
                body_q,
                body_q_prev,
                body_qd,
                body_com,
                contact_shape,
                contact_body_pos,
                contact_body_vel,
                contact_normal,
                dt,
            )
            wp.atomic_add(particle_forces, particle_idx, body_contact_force)
            wp.atomic_add(particle_hessians, particle_idx, body_contact_hessian)


@wp.kernel
def solve_trimesh_with_self_contact_penetration_free(
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=float, ndim=2),
    tri_areas: wp.array(dtype=float),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angles: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    edge_bending_properties: wp.array(dtype=float, ndim=2),
    adjacency: ForceElementAdjacencyInfo,
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
    # output
    pos_new: wp.array(dtype=wp.vec3),
):
    t_id = wp.tid()

    particle_index = particle_ids_in_color[t_id]
    particle_pos = pos[particle_index]

    if not particle_flags[particle_index] & ParticleFlags.ACTIVE:
        pos_new[particle_index] = particle_pos
        return

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # inertia force and hessian
    f = mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
    h = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)

    # fmt: off
    if wp.static("inertia_force_hessian" in VBD_DEBUG_PRINTING_OPTIONS):
        wp.printf(
            "particle: %d after accumulate inertia\nforce:\n %f %f %f, \nhessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
            particle_index,
            f[0], f[1], f[2], h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2],
        )

    # elastic force and hessian
    for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)

        # fmt: off
        if wp.static("connectivity" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "particle: %d | num_adj_faces: %d | ",
                particle_index,
                get_vertex_num_adjacent_faces(particle_index, adjacency),
            )
            wp.printf("i_face: %d | face id: %d | v_order: %d | ", i_adj_tri, tri_index, vertex_order)
            wp.printf(
                "face: %d %d %d\n",
                tri_indices[tri_index, 0],
                tri_indices[tri_index, 1],
                tri_indices[tri_index, 2],
            )
        # fmt: on

        f_tri, h_tri = evaluate_stvk_force_hessian(
            tri_index,
            vertex_order,
            pos,
            pos_prev,
            tri_indices,
            tri_poses[tri_index],
            tri_areas[tri_index],
            tri_materials[tri_index, 0],
            tri_materials[tri_index, 1],
            tri_materials[tri_index, 2],
            dt,
        )

        f = f + f_tri
        h = h + h_tri


    for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_index, i_adj_edge)
        # vertex is on the edge; otherwise it only effects the bending energy n
        if edge_bending_properties[nei_edge_index, 0] != 0.0:
            f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
                nei_edge_index, vertex_order_on_edge, pos, pos_prev, edge_indices, edge_rest_angles, edge_rest_length,
                edge_bending_properties[nei_edge_index, 0], edge_bending_properties[nei_edge_index, 1], dt
            )

            f = f + f_edge
            h = h + h_edge

    # fmt: off
    if wp.static("overall_force_hessian" in VBD_DEBUG_PRINTING_OPTIONS):
        wp.printf(
            "vertex: %d final\noverall force:\n %f %f %f, \noverall hessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
            particle_index,
            f[0], f[1], f[2], h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2],
        )

    # # fmt: on
    h = h + particle_hessians[particle_index]
    f = f + particle_forces[particle_index]

    if abs(wp.determinant(h)) > 1e-5:
        h_inv = wp.inverse(h)
        particle_pos_new = pos[particle_index] + h_inv * f

        pos_new[particle_index] = apply_conservative_bound_truncation(
            particle_index, particle_pos_new, pos_prev_collision_detection, particle_conservative_bounds
        )


@wp.kernel
def solve_trimesh_with_self_contact_penetration_free_tile(
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=float, ndim=2),
    tri_areas: wp.array(dtype=float),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angles: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    edge_bending_properties: wp.array(dtype=float, ndim=2),
    adjacency: ForceElementAdjacencyInfo,
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
    # output
    pos_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    block_idx = tid // TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
    thread_idx = tid % TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
    particle_index = particle_ids_in_color[block_idx]

    if not particle_flags[particle_index] & ParticleFlags.ACTIVE:
        if thread_idx == 0:
            pos_new[particle_index] = pos[particle_index]
        return

    particle_pos = pos[particle_index]

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # elastic force and hessian
    num_adj_faces = get_vertex_num_adjacent_faces(adjacency, particle_index)

    f = wp.vec3(0.0)
    h = wp.mat33(0.0)

    batch_counter = wp.int32(0)

    # loop through all the adjacent triangles using whole block
    while batch_counter + thread_idx < num_adj_faces:
        adj_tri_counter = thread_idx + batch_counter
        batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
        # elastic force and hessian
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, adj_tri_counter)

        # fmt: off
        if wp.static("connectivity" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "particle: %d | num_adj_faces: %d | ",
                particle_index,
                get_vertex_num_adjacent_faces(particle_index, adjacency),
            )
            wp.printf("i_face: %d | face id: %d | v_order: %d | ", adj_tri_counter, tri_index, vertex_order)
            wp.printf(
                "face: %d %d %d\n",
                tri_indices[tri_index, 0],
                tri_indices[tri_index, 1],
                tri_indices[tri_index, 2],
            )
        # fmt: on

        f_tri, h_tri = evaluate_stvk_force_hessian(
            tri_index,
            vertex_order,
            pos,
            pos_prev,
            tri_indices,
            tri_poses[tri_index],
            tri_areas[tri_index],
            tri_materials[tri_index, 0],
            tri_materials[tri_index, 1],
            tri_materials[tri_index, 2],
            dt,
        )

        f += f_tri
        h += h_tri

    batch_counter = wp.int32(0)
    num_adj_edges = get_vertex_num_adjacent_edges(adjacency, particle_index)
    while batch_counter + thread_idx < num_adj_edges:
        adj_edge_counter = batch_counter + thread_idx
        batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(
            adjacency, particle_index, adj_edge_counter
        )
        if edge_bending_properties[nei_edge_index, 0] != 0.0:
            f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
                nei_edge_index,
                vertex_order_on_edge,
                pos,
                pos_prev,
                edge_indices,
                edge_rest_angles,
                edge_rest_length,
                edge_bending_properties[nei_edge_index, 0],
                edge_bending_properties[nei_edge_index, 1],
                dt,
            )

            f += f_edge
            h += h_edge

    f_tile = wp.tile(f, preserve_type=True)
    h_tile = wp.tile(h, preserve_type=True)

    f_total = wp.tile_reduce(wp.add, f_tile)[0]
    h_total = wp.tile_reduce(wp.add, h_tile)[0]

    if thread_idx == 0:
        h_total = (
            h_total
            + mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)
            + particle_hessians[particle_index]
        )
        if abs(wp.determinant(h_total)) > 1e-5:
            h_inv = wp.inverse(h_total)
            f_total = (
                f_total
                + mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
                + particle_forces[particle_index]
            )
            particle_pos_new = particle_pos + h_inv * f_total

            pos_new[particle_index] = apply_conservative_bound_truncation(
                particle_index, particle_pos_new, pos_prev_collision_detection, particle_conservative_bounds
            )


class SolverVBD(SolverBase):
    """An implicit solver using Vertex Block Descent (VBD) for cloth simulation.

    References:
        - Anka He Chen, Ziheng Liu, Yin Yang, and Cem Yuksel. 2024. Vertex Block Descent. ACM Trans. Graph. 43, 4, Article 116 (July 2024), 16 pages.
          https://doi.org/10.1145/3658179

    Note:
        `SolverVBD` requires coloring information for both particles and rigid bodies through
        :attr:`newton.Model.particle_color_groups` and :attr:`newton.Model.body_color_groups`.
        You may call :meth:`newton.ModelBuilder.color` to color entities or use :meth:`newton.ModelBuilder.set_coloring`
        to provide your own coloring.

    Example
    -------

    .. code-block:: python

        # color particles
        builder.color()
        # or you can use your custom coloring
        builder.set_coloring(user_provided_particle_coloring)

        model = builder.finalize()

        solver = newton.solvers.SolverVBD(model)

        # simulation loop
        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in
    """

    def __init__(
        self,
        model: Model,
        iterations: int = 10,
        handle_self_contact: bool = False,
        self_contact_radius: float = 0.2,
        self_contact_margin: float = 0.2,
        integrate_with_external_rigid_solver: bool = False,
        penetration_free_conservative_bound_relaxation: float = 0.42,
        friction_epsilon: float = 1e-2,
        vertex_collision_buffer_pre_alloc: int = 32,
        edge_collision_buffer_pre_alloc: int = 64,
        collision_detection_interval: int = 0,
        edge_edge_parallel_epsilon: float = 1e-5,
        use_tile_solve: bool = True,
    ):
        """
        Args:
            model: The `Model` object used to initialize the integrator. Must be identical to the `Model` object passed
                to the `step` function.
            iterations: Number of VBD iterations per step.
            handle_self_contact: whether to self-contact.
            self_contact_radius: The radius used for self-contact detection. This is the distance at which vertex-triangle
                pairs and edge-edge pairs will start to interact with each other.
            self_contact_margin: The margin used for self-contact detection. This is the distance at which vertex-triangle
                pairs and edge-edge will be considered in contact generation. It should be larger than `self_contact_radius`
                to avoid missing contacts.
            integrate_with_external_rigid_solver: an indicator of coupled rigid body - cloth simulation.  When set to
                `True`, the solver assumes the rigid body solve is handled  externally.
            penetration_free_conservative_bound_relaxation: Relaxation factor for conservative penetration-free projection.
            friction_epsilon: Threshold to smooth small relative velocities in friction computation.
            vertex_collision_buffer_pre_alloc: Preallocation size for each vertex's vertex-triangle collision buffer.
            edge_collision_buffer_pre_alloc: Preallocation size for edge's edge-edge collision buffer.
            edge_edge_parallel_epsilon: Threshold to detect near-parallel edges in edge-edge collision handling.
            collision_detection_interval: Controls how frequently collision detection is applied during the simulation.
                If set to a value < 0, collision detection is only performed once before the initialization step.
                If set to 0, collision detection is applied twice: once before and once immediately after initialization.
                If set to a value `k` >= 1, collision detection is applied before every `k` VBD iterations.
            use_tile_solve: whether to accelerate the solver using tile API
        Note:
            - The `integrate_with_external_rigid_solver` argument is an indicator of one-way coupling between rigid body
              and soft body solvers. If set to True, the rigid states should be integrated externally, with `state_in`
              passed to `step` function representing the previous rigid state and `state_out` representing the current one. Frictional forces are
              computed accordingly.
            - vertex_collision_buffer_pre_alloc` and `edge_collision_buffer_pre_alloc` are fixed and will not be
              dynamically resized during runtime.
              Setting them too small may result in undetected collisions.
              Setting them excessively large may increase memory usage and degrade performance.

        """
        super().__init__(model)
        self.iterations = iterations
        self.integrate_with_external_rigid_solver = integrate_with_external_rigid_solver
        self.collision_detection_interval = collision_detection_interval

        # add new attributes for VBD solve
        self.particle_q_prev = wp.zeros_like(model.particle_q, device=self.device)
        self.inertia = wp.zeros_like(model.particle_q, device=self.device)

        # Rigid body storage for forward stepping
        self.body_q_prev = wp.zeros_like(model.body_q, device=self.device)
        self.body_inertia_q = wp.zeros_like(model.body_q, device=self.device)

        self.adjacency = self.compute_force_element_adjacency(model).to(self.device)

        self.body_particle_contact_count = wp.zeros((model.particle_count,), dtype=wp.int32, device=self.device)

        self.handle_self_contact = handle_self_contact
        self.self_contact_radius = self_contact_radius
        self.self_contact_margin = self_contact_margin

        if model.device.is_cpu and use_tile_solve:
            warnings.warn("Tiled solve requires model.device='cuda'. Tiled solve is disabled.", stacklevel=2)

        self.use_tile_solve = use_tile_solve and model.device.is_cuda

        soft_contact_max = model.shape_count * model.particle_count
        rigid_contact_max = model.shape_count * model.shape_count  # Rigid body vs rigid body contacts

        if handle_self_contact:
            if self_contact_margin < self_contact_radius:
                raise ValueError(
                    "self_contact_margin is smaller than self_contact_radius, this will result in missing contacts and cause instability.\n"
                    "It is advisable to make self_contact_margin 1.5-2 times larger than self_contact_radius."
                )

            self.conservative_bound_relaxation = penetration_free_conservative_bound_relaxation
            self.pos_prev_collision_detection = wp.zeros_like(model.particle_q, device=self.device)
            self.particle_conservative_bounds = wp.full((model.particle_count,), dtype=float, device=self.device)

            self.trimesh_collision_detector = TriMeshCollisionDetector(
                self.model,
                vertex_collision_buffer_pre_alloc=vertex_collision_buffer_pre_alloc,
                edge_collision_buffer_pre_alloc=edge_collision_buffer_pre_alloc,
                edge_edge_parallel_epsilon=edge_edge_parallel_epsilon,
            )

            self.trimesh_collision_info = wp.array(
                [self.trimesh_collision_detector.collision_info], dtype=TriMeshCollisionInfo, device=self.device
            )

            self.soft_contact_launch_size = max(
                self.model.particle_count * NUM_THREADS_PER_COLLISION_PRIMITIVE,
                self.model.edge_count * NUM_THREADS_PER_COLLISION_PRIMITIVE,
                soft_contact_max,
            )
        else:
            self.soft_contact_launch_size = soft_contact_max

        self.rigid_contact_launch_size = rigid_contact_max

        # spaces for particle force and hessian
        self.particle_forces = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=self.device)
        self.particle_hessians = wp.zeros(self.model.particle_count, dtype=wp.mat33, device=self.device)

        # Store torques and forces separately for better performance
        self.body_torques = wp.zeros(self.model.body_count, dtype=wp.vec3, device=self.device)
        self.body_forces = wp.zeros(self.model.body_count, dtype=wp.vec3, device=self.device)

        # Collision Hessian blocks
        self.body_hessian_aa = wp.zeros(self.model.body_count, dtype=wp.mat33, device=self.device)  # Angular-angular
        self.body_hessian_al = wp.zeros(self.model.body_count, dtype=wp.mat33, device=self.device)  # Angular-linear
        self.body_hessian_ll = wp.zeros(self.model.body_count, dtype=wp.mat33, device=self.device)  # Linear-linear

        self.friction_epsilon = friction_epsilon

        # Check that we have coloring information for the entities we're simulating
        has_particles = self.model.particle_count > 0
        has_bodies = self.model.body_count > 0
        has_particle_coloring = len(self.model.particle_color_groups) > 0
        has_body_coloring = len(self.model.body_color_groups) > 0

        if has_particles and not has_particle_coloring:
            raise ValueError(
                "model.particle_color_groups is empty but particles are present! When using the SolverVBD you must call ModelBuilder.color() "
                "or ModelBuilder.set_coloring() before calling ModelBuilder.finalize()."
            )

        if has_bodies and not has_body_coloring:
            raise ValueError(
                "model.body_color_groups is empty but rigid bodies are present! When using the SolverVBD you must call ModelBuilder.color() "
                "or ModelBuilder.set_coloring() before calling ModelBuilder.finalize()."
            )

        if not has_particles and not has_bodies:
            raise ValueError(
                "Model has no particles or rigid bodies! VBD solver requires at least one type of entity to simulate."
            )

        # tests
        # wp.launch(kernel=_test_compute_force_element_adjacency,
        #           inputs=[self.adjacency, model.edge_indices, model.tri_indices],
        #           dim=1, device=self.device)

    def compute_force_element_adjacency(self, model):
        adjacency = ForceElementAdjacencyInfo()
        edges_array = model.edge_indices.to("cpu")
        spring_array = model.spring_indices.to("cpu")
        face_indices = model.tri_indices.to("cpu")

        with wp.ScopedDevice("cpu"):
            if edges_array.size:
                # build vertex-edge adjacency data
                num_vertex_adjacent_edges = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                wp.launch(
                    kernel=self.count_num_adjacent_edges,
                    inputs=[edges_array, num_vertex_adjacent_edges],
                    dim=1,
                )

                num_vertex_adjacent_edges = num_vertex_adjacent_edges.numpy()
                vertex_adjacent_edges_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
                vertex_adjacent_edges_offsets[1:] = np.cumsum(2 * num_vertex_adjacent_edges)[:]
                vertex_adjacent_edges_offsets[0] = 0
                adjacency.v_adj_edges_offsets = wp.array(vertex_adjacent_edges_offsets, dtype=wp.int32)

                # temporal variables to record how much adjacent edges has been filled to each vertex
                vertex_adjacent_edges_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                edge_adjacency_array_size = 2 * num_vertex_adjacent_edges.sum()
                # vertex order: o0: 0, o1: 1, v0: 2, v1: 3,
                adjacency.v_adj_edges = wp.empty(shape=(edge_adjacency_array_size,), dtype=wp.int32)

                wp.launch(
                    kernel=self.fill_adjacent_edges,
                    inputs=[
                        edges_array,
                        adjacency.v_adj_edges_offsets,
                        vertex_adjacent_edges_fill_count,
                        adjacency.v_adj_edges,
                    ],
                    dim=1,
                )
            else:
                adjacency.v_adj_edges_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                adjacency.v_adj_edges = wp.empty(shape=(0,), dtype=wp.int32)

            if face_indices.size:
                # compute adjacent triangles
                # count number of adjacent faces for each vertex
                num_vertex_adjacent_faces = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)
                wp.launch(kernel=self.count_num_adjacent_faces, inputs=[face_indices, num_vertex_adjacent_faces], dim=1)

                # preallocate memory based on counting results
                num_vertex_adjacent_faces = num_vertex_adjacent_faces.numpy()
                vertex_adjacent_faces_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
                vertex_adjacent_faces_offsets[1:] = np.cumsum(2 * num_vertex_adjacent_faces)[:]
                vertex_adjacent_faces_offsets[0] = 0
                adjacency.v_adj_faces_offsets = wp.array(vertex_adjacent_faces_offsets, dtype=wp.int32)

                vertex_adjacent_faces_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                face_adjacency_array_size = 2 * num_vertex_adjacent_faces.sum()
                # (face, vertex_order) * num_adj_faces * num_particles
                # vertex order: v0: 0, v1: 1, o0: 2, v2: 3
                adjacency.v_adj_faces = wp.empty(shape=(face_adjacency_array_size,), dtype=wp.int32)

                wp.launch(
                    kernel=self.fill_adjacent_faces,
                    inputs=[
                        face_indices,
                        adjacency.v_adj_faces_offsets,
                        vertex_adjacent_faces_fill_count,
                        adjacency.v_adj_faces,
                    ],
                    dim=1,
                )
            else:
                adjacency.v_adj_faces_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                adjacency.v_adj_faces = wp.empty(shape=(0,), dtype=wp.int32)

            if spring_array.size:
                # build vertex-springs adjacency data
                num_vertex_adjacent_spring = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                wp.launch(
                    kernel=self.count_num_adjacent_springs,
                    inputs=[spring_array, num_vertex_adjacent_spring],
                    dim=1,
                )

                num_vertex_adjacent_spring = num_vertex_adjacent_spring.numpy()
                vertex_adjacent_springs_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
                vertex_adjacent_springs_offsets[1:] = np.cumsum(num_vertex_adjacent_spring)[:]
                vertex_adjacent_springs_offsets[0] = 0
                adjacency.v_adj_springs_offsets = wp.array(vertex_adjacent_springs_offsets, dtype=wp.int32)

                # temporal variables to record how much adjacent springs has been filled to each vertex
                vertex_adjacent_springs_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)
                adjacency.v_adj_springs = wp.empty(shape=(num_vertex_adjacent_spring.sum(),), dtype=wp.int32)

                wp.launch(
                    kernel=self.fill_adjacent_springs,
                    inputs=[
                        spring_array,
                        adjacency.v_adj_springs_offsets,
                        vertex_adjacent_springs_fill_count,
                        adjacency.v_adj_springs,
                    ],
                    dim=1,
                )

            else:
                adjacency.v_adj_springs_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                adjacency.v_adj_springs = wp.empty(shape=(0,), dtype=wp.int32)

            # Build body-joint adjacency data
            if model.joint_count > 0:
                joint_parent_cpu = model.joint_parent.to("cpu")
                joint_child_cpu = model.joint_child.to("cpu")

                # Count joints connected to each body
                num_body_adjacent_joints = wp.zeros(shape=(model.body_count,), dtype=wp.int32)

                wp.launch(
                    kernel=self.count_num_adjacent_joints,
                    inputs=[joint_parent_cpu, joint_child_cpu, num_body_adjacent_joints],
                    dim=1,
                )

                # Create offsets array (cumulative sum)
                num_body_adjacent_joints = num_body_adjacent_joints.numpy()
                body_adjacent_joints_offsets = np.empty(shape=(model.body_count + 1,), dtype=wp.int32)
                body_adjacent_joints_offsets[1:] = np.cumsum(num_body_adjacent_joints)[:]
                body_adjacent_joints_offsets[0] = 0
                adjacency.body_adj_joints_offsets = wp.array(body_adjacent_joints_offsets, dtype=wp.int32)

                # Fill joint adjacency array
                body_adjacent_joints_fill_count = wp.zeros(shape=(model.body_count,), dtype=wp.int32)
                adjacency.body_adj_joints = wp.empty(shape=(num_body_adjacent_joints.sum(),), dtype=wp.int32)

                wp.launch(
                    kernel=self.fill_adjacent_joints,
                    inputs=[
                        joint_parent_cpu,
                        joint_child_cpu,
                        adjacency.body_adj_joints_offsets,
                        body_adjacent_joints_fill_count,
                        adjacency.body_adj_joints,
                    ],
                    dim=1,
                )
            else:
                adjacency.body_adj_joints_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                adjacency.body_adj_joints = wp.empty(shape=(0,), dtype=wp.int32)

        return adjacency

    @override
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):
        if self.handle_self_contact:
            self.simulate_one_step_with_collisions_penetration_free(state_in, state_out, control, contacts, dt)
        else:
            self.simulate_one_step_no_self_contact(state_in, state_out, control, contacts, dt)

    def simulate_one_step_no_self_contact(
        self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float
    ):
        model = self.model

        # Forward step for particles
        if model.particle_count > 0:
            wp.launch(
                kernel=forward_step,
                inputs=[
                    dt,
                    model.gravity,
                    self.particle_q_prev,
                    state_in.particle_q,
                    state_in.particle_qd,
                    self.model.particle_inv_mass,
                    state_in.particle_f,
                    self.model.particle_flags,
                    self.inertia,
                ],
                dim=self.model.particle_count,
                device=self.device,
            )

        # Forward step for rigid bodies
        if model.body_count > 0:
            wp.launch(
                kernel=forward_step_rigid_bodies,
                inputs=[
                    dt,
                    model.gravity,
                    self.body_q_prev,
                    state_in.body_q,
                    state_in.body_qd,
                    state_in.body_f,
                    model.body_com,
                    model.body_inertia,
                    model.body_inv_mass,
                    model.body_inv_inertia,
                    self.body_inertia_q,
                ],
                dim=self.model.body_count,
                device=self.device,
            )

        for _iter in range(self.iterations):
            self.particle_forces.zero_()
            self.particle_hessians.zero_()

            self.body_torques.zero_()
            self.body_forces.zero_()
            self.body_hessian_aa.zero_()
            self.body_hessian_al.zero_()
            self.body_hessian_ll.zero_()

            for color in range(len(self.model.particle_color_groups)):
                wp.launch(
                    kernel=accumulate_contact_force_and_hessian_no_self_contact,
                    dim=self.soft_contact_launch_size,
                    inputs=[
                        dt,
                        color,
                        self.particle_q_prev,
                        state_in.particle_q,
                        self.model.particle_colors,
                        # body-particle contact
                        self.model.soft_contact_ke,
                        self.model.soft_contact_kd,
                        self.model.soft_contact_mu,
                        self.friction_epsilon,
                        self.model.particle_radius,
                        contacts.soft_contact_particle,
                        contacts.soft_contact_count,
                        contacts.soft_contact_max,
                        self.model.shape_material_mu,
                        self.model.shape_body,
                        state_out.body_q if self.integrate_with_external_rigid_solver else state_in.body_q,
                        state_in.body_q if self.integrate_with_external_rigid_solver else None,
                        self.model.body_qd,
                        self.model.body_com,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_body_vel,
                        contacts.soft_contact_normal,
                    ],
                    outputs=[self.particle_forces, self.particle_hessians],
                    device=self.device,
                )

                if model.spring_count:
                    wp.launch(
                        kernel=accumulate_spring_force_and_hessian,
                        inputs=[
                            dt,
                            color,
                            self.particle_q_prev,
                            state_in.particle_q,
                            self.model.particle_color_groups[color],
                            self.adjacency,
                            self.model.spring_indices,
                            self.model.spring_rest_length,
                            self.model.spring_stiffness,
                            self.model.spring_damping,
                        ],
                        outputs=[self.particle_forces, self.particle_hessians],
                        dim=self.model.particle_color_groups[color].size,
                        device=self.device,
                    )

                if self.use_tile_solve:
                    wp.launch(
                        kernel=solve_trimesh_no_self_contact_tile,
                        inputs=[
                            dt,
                            self.model.particle_color_groups[color],
                            self.particle_q_prev,
                            state_in.particle_q,
                            state_in.particle_qd,
                            self.model.particle_mass,
                            self.inertia,
                            self.model.particle_flags,
                            self.model.tri_indices,
                            self.model.tri_poses,
                            self.model.tri_materials,
                            self.model.tri_areas,
                            self.model.edge_indices,
                            self.model.edge_rest_angle,
                            self.model.edge_rest_length,
                            self.model.edge_bending_properties,
                            self.adjacency,
                            self.particle_forces,
                            self.particle_hessians,
                        ],
                        outputs=[
                            state_out.particle_q,
                        ],
                        dim=self.model.particle_color_groups[color].size * TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
                        block_dim=TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
                        device=self.device,
                    )
                else:
                    wp.launch(
                        kernel=solve_trimesh_no_self_contact,
                        inputs=[
                            dt,
                            self.model.particle_color_groups[color],
                            self.particle_q_prev,
                            state_in.particle_q,
                            state_in.particle_qd,
                            self.model.particle_mass,
                            self.inertia,
                            self.model.particle_flags,
                            self.model.tri_indices,
                            self.model.tri_poses,
                            self.model.tri_materials,
                            self.model.tri_areas,
                            self.model.edge_indices,
                            self.model.edge_rest_angle,
                            self.model.edge_rest_length,
                            self.model.edge_bending_properties,
                            self.adjacency,
                            self.particle_forces,
                            self.particle_hessians,
                        ],
                        outputs=[
                            state_out.particle_q,
                        ],
                        dim=self.model.particle_color_groups[color].size,
                        device=self.device,
                    )

                wp.launch(
                    kernel=copy_particle_positions_back,
                    inputs=[self.model.particle_color_groups[color], state_in.particle_q],
                    outputs=[state_out.particle_q],
                    dim=self.model.particle_color_groups[color].size,
                    device=self.device,
                )

            for color in range(len(self.model.body_color_groups)):
                wp.launch(
                    kernel=accumulate_rigid_contact_force_and_hessian,
                    dim=self.rigid_contact_launch_size,
                    inputs=[
                        dt,
                        color,
                        self.model.body_colors,
                        self.body_q_prev,
                        state_in.body_q,
                        self.model.body_com,
                        self.model.body_inv_mass,
                        self.model.soft_contact_ke,
                        self.model.soft_contact_kd,
                        self.model.soft_contact_mu,
                        self.friction_epsilon,
                        self.model.shape_material_mu,
                        contacts.rigid_contact_count,
                        contacts.rigid_contact_max,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,
                        contacts.rigid_contact_normal,
                        contacts.rigid_contact_thickness0,
                        contacts.rigid_contact_thickness1,
                        model.shape_body,
                    ],
                    outputs=[
                        self.body_forces,
                        self.body_torques,
                        self.body_hessian_ll,
                        self.body_hessian_al,
                        self.body_hessian_aa,
                    ],
                    device=self.device,
                )

                wp.launch(
                    kernel=solve_rigid_body,
                    inputs=[
                        dt,
                        self.model.body_color_groups[color],
                        self.body_q_prev,
                        state_in.body_q,
                        self.model.body_q,
                        self.model.body_mass,
                        self.model.body_inv_mass,
                        self.model.body_inertia,
                        self.body_inertia_q,
                        self.model.body_com,
                        self.adjacency,
                        self.model.joint_type,
                        self.model.joint_parent,
                        self.model.joint_child,
                        self.model.joint_X_p,
                        self.model.joint_X_c,
                        self.model.joint_qd_start,
                        self.model.joint_dof_dim,
                        self.model.joint_axis,
                        self.model.joint_target_ke,
                        self.model.joint_target_kd,
                        state_in.joint_q,
                        state_in.joint_qd,
                        self.model.joint_target,
                        self.body_forces,
                        self.body_torques,
                        self.body_hessian_ll,
                        self.body_hessian_al,
                        self.body_hessian_aa,
                    ],
                    outputs=[
                        state_out.body_q,
                    ],
                    dim=self.model.body_color_groups[color].size,
                    device=self.device,
                )

                wp.launch(
                    kernel=copy_rigid_body_transforms_back,
                    inputs=[self.model.body_color_groups[color], state_in.body_q],
                    outputs=[state_out.body_q],
                    dim=self.model.body_color_groups[color].size,
                    device=self.device,
                )

        if model.particle_count > 0:
            wp.launch(
                kernel=update_particle_velocity,
                inputs=[dt, self.particle_q_prev, state_out.particle_q],
                outputs=[state_out.particle_qd],
                dim=self.model.particle_count,
                device=self.device,
            )

        if model.body_count > 0:
            wp.launch(
                kernel=update_body_velocity,
                inputs=[dt, state_out.body_q, self.body_q_prev, self.model.body_com],
                outputs=[state_out.body_qd],
                dim=model.body_count,
                device=self.device,
            )

    def simulate_one_step_with_collisions_penetration_free(
        self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float
    ):
        # collision detection before initialization to compute conservative bounds for initialization
        self.collision_detection_penetration_free(state_in, dt)

        model = self.model

        # Forward step for particles
        if model.particle_count > 0:
            wp.launch(
                kernel=forward_step_penetration_free,
                inputs=[
                    dt,
                    model.gravity,
                    self.particle_q_prev,
                    state_in.particle_q,
                    state_in.particle_qd,
                    self.model.particle_inv_mass,
                    state_in.particle_f,
                    self.model.particle_flags,
                    self.pos_prev_collision_detection,
                    self.particle_conservative_bounds,
                    self.inertia,
                ],
                dim=self.model.particle_count,
                device=self.device,
            )

        # Forward step for rigid bodies
        if model.body_count > 0:
            wp.launch(
                kernel=forward_step_rigid_bodies,
                inputs=[
                    dt,
                    model.gravity,
                    self.body_q_prev,
                    state_in.body_q,
                    state_in.body_qd,
                    state_in.body_f,
                    model.body_com,
                    model.body_inertia,
                    model.body_inv_mass,
                    model.body_inv_inertia,
                    self.body_inertia_q,
                ],
                dim=self.model.body_count,
                device=self.device,
            )

        for _iter in range(self.iterations):
            # after initialization, we need new collision detection to update the bounds
            if (self.collision_detection_interval == 0 and _iter == 0) or (
                self.collision_detection_interval >= 1 and _iter % self.collision_detection_interval == 0
            ):
                self.collision_detection_penetration_free(state_in, dt)

            self.particle_forces.zero_()
            self.particle_hessians.zero_()

            self.body_torques.zero_()
            self.body_forces.zero_()
            self.body_hessian_aa.zero_()
            self.body_hessian_al.zero_()
            self.body_hessian_ll.zero_()

            for color in range(len(self.model.particle_color_groups)):
                if contacts is not None:
                    wp.launch(
                        kernel=accumulate_contact_force_and_hessian,
                        dim=self.soft_contact_launch_size,
                        inputs=[
                            dt,
                            color,
                            self.particle_q_prev,
                            state_in.particle_q,
                            self.model.particle_colors,
                            self.model.tri_indices,
                            self.model.edge_indices,
                            # self-contact
                            self.trimesh_collision_info,
                            self.self_contact_radius,
                            self.model.soft_contact_ke,
                            self.model.soft_contact_kd,
                            self.model.soft_contact_mu,
                            self.friction_epsilon,
                            self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                            # body-particle contact
                            self.model.particle_radius,
                            contacts.soft_contact_particle,
                            contacts.soft_contact_count,
                            contacts.soft_contact_max,
                            self.model.shape_material_mu,
                            self.model.shape_body,
                            state_out.body_q if self.integrate_with_external_rigid_solver else state_in.body_q,
                            state_in.body_q if self.integrate_with_external_rigid_solver else None,
                            self.model.body_qd,
                            self.model.body_com,
                            contacts.soft_contact_shape,
                            contacts.soft_contact_body_pos,
                            contacts.soft_contact_body_vel,
                            contacts.soft_contact_normal,
                        ],
                        outputs=[self.particle_forces, self.particle_hessians],
                        device=self.device,
                        max_blocks=self.model.device.sm_count,
                    )

                if model.spring_count:
                    wp.launch(
                        kernel=accumulate_spring_force_and_hessian,
                        inputs=[
                            dt,
                            color,
                            self.particle_q_prev,
                            state_in.particle_q,
                            self.model.particle_color_groups[color],
                            self.adjacency,
                            self.model.spring_indices,
                            self.model.spring_rest_length,
                            self.model.spring_stiffness,
                            self.model.spring_damping,
                        ],
                        outputs=[self.particle_forces, self.particle_hessians],
                        dim=self.model.particle_color_groups[color].size,
                        device=self.device,
                    )

                if self.use_tile_solve:
                    wp.launch(
                        kernel=solve_trimesh_with_self_contact_penetration_free_tile,
                        dim=self.model.particle_color_groups[color].size * TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
                        block_dim=TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
                        inputs=[
                            dt,
                            self.model.particle_color_groups[color],
                            self.particle_q_prev,
                            state_in.particle_q,
                            state_in.particle_qd,
                            self.model.particle_mass,
                            self.inertia,
                            self.model.particle_flags,
                            self.model.tri_indices,
                            self.model.tri_poses,
                            self.model.tri_materials,
                            self.model.tri_areas,
                            self.model.edge_indices,
                            self.model.edge_rest_angle,
                            self.model.edge_rest_length,
                            self.model.edge_bending_properties,
                            self.adjacency,
                            self.particle_forces,
                            self.particle_hessians,
                            self.pos_prev_collision_detection,
                            self.particle_conservative_bounds,
                        ],
                        outputs=[
                            state_out.particle_q,
                        ],
                        device=self.device,
                    )
                else:
                    wp.launch(
                        kernel=solve_trimesh_with_self_contact_penetration_free,
                        dim=self.model.particle_color_groups[color].size,
                        inputs=[
                            dt,
                            self.model.particle_color_groups[color],
                            self.particle_q_prev,
                            state_in.particle_q,
                            state_in.particle_qd,
                            self.model.particle_mass,
                            self.inertia,
                            self.model.particle_flags,
                            self.model.tri_indices,
                            self.model.tri_poses,
                            self.model.tri_materials,
                            self.model.tri_areas,
                            self.model.edge_indices,
                            self.model.edge_rest_angle,
                            self.model.edge_rest_length,
                            self.model.edge_bending_properties,
                            self.adjacency,
                            self.particle_forces,
                            self.particle_hessians,
                            self.pos_prev_collision_detection,
                            self.particle_conservative_bounds,
                        ],
                        outputs=[
                            state_out.particle_q,
                        ],
                        device=self.device,
                    )

                wp.launch(
                    kernel=copy_particle_positions_back,
                    inputs=[self.model.particle_color_groups[color], state_in.particle_q],
                    outputs=[state_out.particle_q],
                    dim=self.model.particle_color_groups[color].size,
                    device=self.device,
                )

            for color in range(len(self.model.body_color_groups)):
                wp.launch(
                    kernel=accumulate_rigid_contact_force_and_hessian,
                    dim=self.rigid_contact_launch_size,
                    inputs=[
                        dt,
                        color,
                        self.model.body_colors,
                        self.body_q_prev,
                        state_in.body_q,
                        self.model.body_com,
                        self.model.body_inv_mass,
                        self.model.soft_contact_ke,
                        self.model.soft_contact_kd,
                        self.model.soft_contact_mu,
                        self.friction_epsilon,
                        self.model.shape_material_mu,
                        contacts.rigid_contact_count,
                        contacts.rigid_contact_max,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,
                        contacts.rigid_contact_normal,
                        contacts.rigid_contact_thickness0,
                        contacts.rigid_contact_thickness1,
                        model.shape_body,
                    ],
                    outputs=[
                        self.body_forces,
                        self.body_torques,
                        self.body_hessian_ll,
                        self.body_hessian_al,
                        self.body_hessian_aa,
                    ],
                    device=self.device,
                )

                wp.launch(
                    kernel=solve_rigid_body,
                    inputs=[
                        dt,
                        self.model.body_color_groups[color],
                        self.body_q_prev,
                        state_in.body_q,
                        self.model.body_q,
                        self.model.body_mass,
                        self.model.body_inv_mass,
                        self.model.body_inertia,
                        self.body_inertia_q,
                        self.model.body_com,
                        self.adjacency,
                        self.model.joint_type,
                        self.model.joint_parent,
                        self.model.joint_child,
                        self.model.joint_X_p,
                        self.model.joint_X_c,
                        self.model.joint_qd_start,
                        self.model.joint_dof_dim,
                        self.model.joint_axis,
                        self.model.joint_target_ke,
                        self.model.joint_target_kd,
                        state_in.joint_q,
                        state_in.joint_qd,
                        self.model.joint_target,
                        self.body_forces,
                        self.body_torques,
                        self.body_hessian_ll,
                        self.body_hessian_al,
                        self.body_hessian_aa,
                    ],
                    outputs=[
                        state_out.body_q,
                    ],
                    dim=self.model.body_color_groups[color].size,
                    device=self.device,
                )

                wp.launch(
                    kernel=copy_rigid_body_transforms_back,
                    inputs=[self.model.body_color_groups[color], state_in.body_q],
                    outputs=[state_out.body_q],
                    dim=self.model.body_color_groups[color].size,
                    device=self.device,
                )

        if model.particle_count > 0:
            wp.launch(
                kernel=update_particle_velocity,
                inputs=[dt, self.particle_q_prev, state_out.particle_q],
                outputs=[state_out.particle_qd],
                dim=self.model.particle_count,
                device=self.device,
            )

        if model.body_count > 0:
            wp.launch(
                kernel=update_body_velocity,
                inputs=[dt, state_out.body_q, self.body_q_prev, self.model.body_com],
                outputs=[state_out.body_qd],
                dim=model.body_count,
                device=self.device,
            )

    def collision_detection_penetration_free(self, current_state: State, dt: float):
        self.trimesh_collision_detector.refit(current_state.particle_q)
        self.trimesh_collision_detector.vertex_triangle_collision_detection(self.self_contact_margin)
        self.trimesh_collision_detector.edge_edge_collision_detection(self.self_contact_margin)

        self.pos_prev_collision_detection.assign(current_state.particle_q)
        wp.launch(
            kernel=compute_particle_conservative_bound,
            inputs=[
                self.conservative_bound_relaxation,
                self.self_contact_margin,
                self.adjacency,
                self.trimesh_collision_detector.collision_info,
            ],
            outputs=[
                self.particle_conservative_bounds,
            ],
            dim=self.model.particle_count,
            device=self.device,
        )

    def rebuild_bvh(self, state: State):
        """This function will rebuild the BVHs used for detecting self-contacts using the input `state`.

        When the simulated object deforms significantly, simply refitting the BVH can lead to deterioration of the BVH's
        quality. In these cases, rebuilding the entire tree is necessary to achieve better querying efficiency.

        Args:
            state (newton.State):  The state whose particle positions (:attr:`State.particle_q`) will be used for rebuilding the BVHs.
        """
        if self.handle_self_contact:
            self.trimesh_collision_detector.rebuild(state.particle_q)

    @wp.kernel
    def count_num_adjacent_edges(
        edges_array: wp.array(dtype=wp.int32, ndim=2), num_vertex_adjacent_edges: wp.array(dtype=wp.int32)
    ):
        for edge_id in range(edges_array.shape[0]):
            o0 = edges_array[edge_id, 0]
            o1 = edges_array[edge_id, 1]

            v0 = edges_array[edge_id, 2]
            v1 = edges_array[edge_id, 3]

            num_vertex_adjacent_edges[v0] = num_vertex_adjacent_edges[v0] + 1
            num_vertex_adjacent_edges[v1] = num_vertex_adjacent_edges[v1] + 1

            if o0 != -1:
                num_vertex_adjacent_edges[o0] = num_vertex_adjacent_edges[o0] + 1
            if o1 != -1:
                num_vertex_adjacent_edges[o1] = num_vertex_adjacent_edges[o1] + 1

    @wp.kernel
    def fill_adjacent_edges(
        edges_array: wp.array(dtype=wp.int32, ndim=2),
        vertex_adjacent_edges_offsets: wp.array(dtype=wp.int32),
        vertex_adjacent_edges_fill_count: wp.array(dtype=wp.int32),
        vertex_adjacent_edges: wp.array(dtype=wp.int32),
    ):
        for edge_id in range(edges_array.shape[0]):
            v0 = edges_array[edge_id, 2]
            v1 = edges_array[edge_id, 3]

            fill_count_v0 = vertex_adjacent_edges_fill_count[v0]
            buffer_offset_v0 = vertex_adjacent_edges_offsets[v0]
            vertex_adjacent_edges[buffer_offset_v0 + fill_count_v0 * 2] = edge_id
            vertex_adjacent_edges[buffer_offset_v0 + fill_count_v0 * 2 + 1] = 2
            vertex_adjacent_edges_fill_count[v0] = fill_count_v0 + 1

            fill_count_v1 = vertex_adjacent_edges_fill_count[v1]
            buffer_offset_v1 = vertex_adjacent_edges_offsets[v1]
            vertex_adjacent_edges[buffer_offset_v1 + fill_count_v1 * 2] = edge_id
            vertex_adjacent_edges[buffer_offset_v1 + fill_count_v1 * 2 + 1] = 3
            vertex_adjacent_edges_fill_count[v1] = fill_count_v1 + 1

            o0 = edges_array[edge_id, 0]
            if o0 != -1:
                fill_count_o0 = vertex_adjacent_edges_fill_count[o0]
                buffer_offset_o0 = vertex_adjacent_edges_offsets[o0]
                vertex_adjacent_edges[buffer_offset_o0 + fill_count_o0 * 2] = edge_id
                vertex_adjacent_edges[buffer_offset_o0 + fill_count_o0 * 2 + 1] = 0
                vertex_adjacent_edges_fill_count[o0] = fill_count_o0 + 1

            o1 = edges_array[edge_id, 1]
            if o1 != -1:
                fill_count_o1 = vertex_adjacent_edges_fill_count[o1]
                buffer_offset_o1 = vertex_adjacent_edges_offsets[o1]
                vertex_adjacent_edges[buffer_offset_o1 + fill_count_o1 * 2] = edge_id
                vertex_adjacent_edges[buffer_offset_o1 + fill_count_o1 * 2 + 1] = 1
                vertex_adjacent_edges_fill_count[o1] = fill_count_o1 + 1

    @wp.kernel
    def count_num_adjacent_faces(
        face_indices: wp.array(dtype=wp.int32, ndim=2), num_vertex_adjacent_faces: wp.array(dtype=wp.int32)
    ):
        for face in range(face_indices.shape[0]):
            v0 = face_indices[face, 0]
            v1 = face_indices[face, 1]
            v2 = face_indices[face, 2]

            num_vertex_adjacent_faces[v0] = num_vertex_adjacent_faces[v0] + 1
            num_vertex_adjacent_faces[v1] = num_vertex_adjacent_faces[v1] + 1
            num_vertex_adjacent_faces[v2] = num_vertex_adjacent_faces[v2] + 1

    @wp.kernel
    def fill_adjacent_faces(
        face_indices: wp.array(dtype=wp.int32, ndim=2),
        vertex_adjacent_faces_offsets: wp.array(dtype=wp.int32),
        vertex_adjacent_faces_fill_count: wp.array(dtype=wp.int32),
        vertex_adjacent_faces: wp.array(dtype=wp.int32),
    ):
        for face in range(face_indices.shape[0]):
            v0 = face_indices[face, 0]
            v1 = face_indices[face, 1]
            v2 = face_indices[face, 2]

            fill_count_v0 = vertex_adjacent_faces_fill_count[v0]
            buffer_offset_v0 = vertex_adjacent_faces_offsets[v0]
            vertex_adjacent_faces[buffer_offset_v0 + fill_count_v0 * 2] = face
            vertex_adjacent_faces[buffer_offset_v0 + fill_count_v0 * 2 + 1] = 0
            vertex_adjacent_faces_fill_count[v0] = fill_count_v0 + 1

            fill_count_v1 = vertex_adjacent_faces_fill_count[v1]
            buffer_offset_v1 = vertex_adjacent_faces_offsets[v1]
            vertex_adjacent_faces[buffer_offset_v1 + fill_count_v1 * 2] = face
            vertex_adjacent_faces[buffer_offset_v1 + fill_count_v1 * 2 + 1] = 1
            vertex_adjacent_faces_fill_count[v1] = fill_count_v1 + 1

            fill_count_v2 = vertex_adjacent_faces_fill_count[v2]
            buffer_offset_v2 = vertex_adjacent_faces_offsets[v2]
            vertex_adjacent_faces[buffer_offset_v2 + fill_count_v2 * 2] = face
            vertex_adjacent_faces[buffer_offset_v2 + fill_count_v2 * 2 + 1] = 2
            vertex_adjacent_faces_fill_count[v2] = fill_count_v2 + 1

    @wp.kernel
    def count_num_adjacent_springs(
        springs_array: wp.array(dtype=wp.int32), num_vertex_adjacent_springs: wp.array(dtype=wp.int32)
    ):
        num_springs = springs_array.shape[0] / 2
        for spring_id in range(num_springs):
            v0 = springs_array[spring_id * 2]
            v1 = springs_array[spring_id * 2 + 1]

            num_vertex_adjacent_springs[v0] = num_vertex_adjacent_springs[v0] + 1
            num_vertex_adjacent_springs[v1] = num_vertex_adjacent_springs[v1] + 1

    @wp.kernel
    def fill_adjacent_springs(
        springs_array: wp.array(dtype=wp.int32),
        vertex_adjacent_springs_offsets: wp.array(dtype=wp.int32),
        vertex_adjacent_springs_fill_count: wp.array(dtype=wp.int32),
        vertex_adjacent_springs: wp.array(dtype=wp.int32),
    ):
        num_springs = springs_array.shape[0] / 2
        for spring_id in range(num_springs):
            v0 = springs_array[spring_id * 2]
            v1 = springs_array[spring_id * 2 + 1]

            fill_count_v0 = vertex_adjacent_springs_fill_count[v0]
            buffer_offset_v0 = vertex_adjacent_springs_offsets[v0]
            vertex_adjacent_springs[buffer_offset_v0 + fill_count_v0] = spring_id
            vertex_adjacent_springs_fill_count[v0] = fill_count_v0 + 1

            fill_count_v1 = vertex_adjacent_springs_fill_count[v1]
            buffer_offset_v1 = vertex_adjacent_springs_offsets[v1]
            vertex_adjacent_springs[buffer_offset_v1 + fill_count_v1] = spring_id
            vertex_adjacent_springs_fill_count[v1] = fill_count_v1 + 1

    @wp.kernel
    def count_num_adjacent_joints(
        joint_parent: wp.array(dtype=wp.int32),
        joint_child: wp.array(dtype=wp.int32),
        num_body_adjacent_joints: wp.array(dtype=wp.int32),
    ):
        joint_count = joint_parent.shape[0]
        for joint_id in range(joint_count):
            parent_id = joint_parent[joint_id]
            child_id = joint_child[joint_id]

            # Skip world joints (parent/child == -1)
            if parent_id >= 0:
                num_body_adjacent_joints[parent_id] = num_body_adjacent_joints[parent_id] + 1
            if child_id >= 0:
                num_body_adjacent_joints[child_id] = num_body_adjacent_joints[child_id] + 1

    @wp.kernel
    def fill_adjacent_joints(
        joint_parent: wp.array(dtype=wp.int32),
        joint_child: wp.array(dtype=wp.int32),
        body_adjacent_joints_offsets: wp.array(dtype=wp.int32),
        body_adjacent_joints_fill_count: wp.array(dtype=wp.int32),
        body_adjacent_joints: wp.array(dtype=wp.int32),
    ):
        joint_count = joint_parent.shape[0]
        for joint_id in range(joint_count):
            parent_id = joint_parent[joint_id]
            child_id = joint_child[joint_id]

            # Add joint to parent body's adjacency list
            if parent_id >= 0:
                fill_count_parent = body_adjacent_joints_fill_count[parent_id]
                buffer_offset_parent = body_adjacent_joints_offsets[parent_id]
                body_adjacent_joints[buffer_offset_parent + fill_count_parent] = joint_id
                body_adjacent_joints_fill_count[parent_id] = fill_count_parent + 1

            # Add joint to child body's adjacency list
            if child_id >= 0:
                fill_count_child = body_adjacent_joints_fill_count[child_id]
                buffer_offset_child = body_adjacent_joints_offsets[child_id]
                body_adjacent_joints[buffer_offset_child + fill_count_child] = joint_id
                body_adjacent_joints_fill_count[child_id] = fill_count_child + 1


@wp.func
def get_both_bodies_from_contact_with_thickness(
    t_id: int,
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_thickness0: wp.array(dtype=float),
    rigid_contact_thickness1: wp.array(dtype=float),
    shape_body: wp.array(dtype=wp.int32),
):
    """Extract both bodies from a rigid contact pair."""
    shape_id_0 = rigid_contact_shape0[t_id]
    shape_id_1 = rigid_contact_shape1[t_id]

    body_id_0 = shape_body[shape_id_0] if shape_id_0 >= 0 else -1
    body_id_1 = shape_body[shape_id_1] if shape_id_1 >= 0 else -1

    return (
        body_id_0,
        shape_id_0,
        rigid_contact_point0[t_id],
        rigid_contact_thickness0[t_id],
        body_id_1,
        shape_id_1,
        rigid_contact_point1[t_id],
        rigid_contact_thickness1[t_id],
    )


@wp.kernel
def accumulate_rigid_contact_force_and_hessian(
    dt: float,
    current_color: int,
    body_colors: wp.array(dtype=int),
    body_q_prev: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_inv_mass: wp.array(dtype=float),
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    shape_material_mu: wp.array(dtype=float),
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_max: int,
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_thickness0: wp.array(dtype=float),
    rigid_contact_thickness1: wp.array(dtype=float),
    shape_body: wp.array(dtype=wp.int32),
    body_forces: wp.array(dtype=wp.vec3),
    body_torques: wp.array(dtype=wp.vec3),
    body_hessian_ll: wp.array(dtype=wp.mat33),
    body_hessian_al: wp.array(dtype=wp.mat33),
    body_hessian_aa: wp.array(dtype=wp.mat33),
):
    """
    Rigid body collision accumulation kernel.

    Processes rigid body-body contacts, computing contact forces and Hessians
    using penalty method with friction. Only applies forces to dynamic bodies
    in the current color group for parallel processing.
    """
    t_id = wp.tid()

    rigid_body_contact_count = min(rigid_contact_max, rigid_contact_count[0])

    if t_id < rigid_body_contact_count:
        (
            body_id_0,
            shape_id_0,
            contact_point_0,
            collision_thickness_0,
            body_id_1,
            shape_id_1,
            contact_point_1,
            collision_thickness_1,
        ) = get_both_bodies_from_contact_with_thickness(
            t_id,
            rigid_contact_shape0,
            rigid_contact_shape1,
            rigid_contact_point0,
            rigid_contact_point1,
            rigid_contact_thickness0,
            rigid_contact_thickness1,
            shape_body,
        )

        # Determine which bodies are in the current color set
        apply_to_body_0 = body_id_0 >= 0 and body_colors[body_id_0] == current_color and body_inv_mass[body_id_0] > 0.0
        apply_to_body_1 = body_id_1 >= 0 and body_colors[body_id_1] == current_color and body_inv_mass[body_id_1] > 0.0

        if apply_to_body_0 or apply_to_body_1:
            contact_normal = -rigid_contact_normal[t_id]

            # The contact points are on the surfaces of the collision shapes.
            contact_point_0_world = (
                wp.transform_point(body_q[body_id_0], contact_point_0) if body_id_0 >= 0 else contact_point_0
            )
            contact_point_1_world = (
                wp.transform_point(body_q[body_id_1], contact_point_1) if body_id_1 >= 0 else contact_point_1
            )

            # Penetration is the geometric overlap, calculated consistently for all types.
            thickness = collision_thickness_0 + collision_thickness_1
            dist = wp.dot(contact_normal, contact_point_0_world - contact_point_1_world)
            actual_penetration = wp.max(0.0, thickness - dist)

            # Process contact forces only if there is penetration
            if actual_penetration > 1.0e-9:
                # Use average material properties for contact
                contact_mu = (
                    0.5 * (shape_material_mu[shape_id_0] + shape_material_mu[shape_id_1])
                    if body_id_0 >= 0 and body_id_1 >= 0
                    else shape_material_mu[shape_id_1]
                    if body_id_0 < 0
                    else shape_material_mu[shape_id_0]
                )

                (force_a, torque_a, h_ll_a, h_al_a, h_aa_a, force_b, torque_b, h_ll_b, h_al_b, h_aa_b) = (
                    evaluate_rigid_contact_from_collision(
                        body_id_0,
                        body_id_1,
                        body_q,
                        body_q_prev,
                        body_com,
                        contact_point_0,
                        contact_point_1,
                        contact_normal,
                        actual_penetration,
                        soft_contact_ke,
                        soft_contact_kd,
                        contact_mu,
                        friction_epsilon,
                        dt,
                    )
                )

                # Apply forces only to bodies in current color
                if apply_to_body_0:
                    wp.atomic_add(body_forces, body_id_0, force_a)
                    wp.atomic_add(body_torques, body_id_0, torque_a)
                    wp.atomic_add(body_hessian_ll, body_id_0, h_ll_a)
                    wp.atomic_add(body_hessian_al, body_id_0, h_al_a)
                    wp.atomic_add(body_hessian_aa, body_id_0, h_aa_a)

                if apply_to_body_1:
                    wp.atomic_add(body_forces, body_id_1, force_b)
                    wp.atomic_add(body_torques, body_id_1, torque_b)
                    wp.atomic_add(body_hessian_ll, body_id_1, h_ll_b)
                    wp.atomic_add(body_hessian_al, body_id_1, h_al_b)
                    wp.atomic_add(body_hessian_aa, body_id_1, h_aa_b)
