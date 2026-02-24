# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Particle/soft-body VBD helper routines.

This module is intended to host the particle/soft-body specific parts of the
VBD solver (cloth, springs, triangles, tets, particle contacts, etc.).

The high-level :class:`SolverVBD` interface should remain in
``solver_vbd.py`` and call into functions defined here.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from newton._src.math import orthonormal_basis
from newton._src.solvers.vbd.rigid_vbd_kernels import evaluate_body_particle_contact

from ...geometry import ParticleFlags
from ...geometry.kernels import triangle_closest_point
from .tri_mesh_collision import (
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
TILE_SIZE_SELF_CONTACT_SOLVE = 8


class mat32(wp.types.matrix(shape=(3, 2), dtype=wp.float32)):
    pass


class mat99(wp.types.matrix(shape=(9, 9), dtype=wp.float32)):
    pass


class mat93(wp.types.matrix(shape=(9, 3), dtype=wp.float32)):
    pass


class mat43(wp.types.matrix(shape=(4, 3), dtype=wp.float32)):
    pass


class vec9(wp.types.vector(length=9, dtype=wp.float32)):
    pass


@wp.struct
class ParticleForceElementAdjacencyInfo:
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

    v_adj_tets: wp.array(dtype=int)
    v_adj_tets_offsets: wp.array(dtype=int)

    def to(self, device):
        if device == self.v_adj_faces.device:
            return self
        else:
            adjacency_gpu = ParticleForceElementAdjacencyInfo()
            adjacency_gpu.v_adj_faces = self.v_adj_faces.to(device)
            adjacency_gpu.v_adj_faces_offsets = self.v_adj_faces_offsets.to(device)

            adjacency_gpu.v_adj_edges = self.v_adj_edges.to(device)
            adjacency_gpu.v_adj_edges_offsets = self.v_adj_edges_offsets.to(device)

            adjacency_gpu.v_adj_springs = self.v_adj_springs.to(device)
            adjacency_gpu.v_adj_springs_offsets = self.v_adj_springs_offsets.to(device)

            adjacency_gpu.v_adj_tets = self.v_adj_tets.to(device)
            adjacency_gpu.v_adj_tets_offsets = self.v_adj_tets_offsets.to(device)

            return adjacency_gpu


@wp.func
def get_vertex_num_adjacent_edges(adjacency: ParticleForceElementAdjacencyInfo, vertex: wp.int32):
    return (adjacency.v_adj_edges_offsets[vertex + 1] - adjacency.v_adj_edges_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_edge_id_order(adjacency: ParticleForceElementAdjacencyInfo, vertex: wp.int32, edge: wp.int32):
    offset = adjacency.v_adj_edges_offsets[vertex]
    return adjacency.v_adj_edges[offset + edge * 2], adjacency.v_adj_edges[offset + edge * 2 + 1]


@wp.func
def get_vertex_num_adjacent_faces(adjacency: ParticleForceElementAdjacencyInfo, vertex: wp.int32):
    return (adjacency.v_adj_faces_offsets[vertex + 1] - adjacency.v_adj_faces_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_face_id_order(adjacency: ParticleForceElementAdjacencyInfo, vertex: wp.int32, face: wp.int32):
    offset = adjacency.v_adj_faces_offsets[vertex]
    return adjacency.v_adj_faces[offset + face * 2], adjacency.v_adj_faces[offset + face * 2 + 1]


@wp.func
def get_vertex_num_adjacent_springs(adjacency: ParticleForceElementAdjacencyInfo, vertex: wp.int32):
    return adjacency.v_adj_springs_offsets[vertex + 1] - adjacency.v_adj_springs_offsets[vertex]


@wp.func
def get_vertex_adjacent_spring_id(adjacency: ParticleForceElementAdjacencyInfo, vertex: wp.int32, spring: wp.int32):
    offset = adjacency.v_adj_springs_offsets[vertex]
    return adjacency.v_adj_springs[offset + spring]


@wp.func
def get_vertex_num_adjacent_tets(adjacency: ParticleForceElementAdjacencyInfo, vertex: wp.int32):
    return (adjacency.v_adj_tets_offsets[vertex + 1] - adjacency.v_adj_tets_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_tet_id_order(adjacency: ParticleForceElementAdjacencyInfo, vertex: wp.int32, tet: wp.int32):
    offset = adjacency.v_adj_tets_offsets[vertex]
    return adjacency.v_adj_tets[offset + tet * 2], adjacency.v_adj_tets[offset + tet * 2 + 1]


@wp.func
def assemble_tet_vertex_force_and_hessian(
    dE_dF: vec9,
    H: mat99,
    m1: float,
    m2: float,
    m3: float,
):
    f = wp.vec3(
        -(dE_dF[0] * m1 + dE_dF[3] * m2 + dE_dF[6] * m3),
        -(dE_dF[1] * m1 + dE_dF[4] * m2 + dE_dF[7] * m3),
        -(dE_dF[2] * m1 + dE_dF[5] * m2 + dE_dF[8] * m3),
    )
    h = wp.mat33()

    h[0, 0] += (
        m1 * (H[0, 0] * m1 + H[3, 0] * m2 + H[6, 0] * m3)
        + m2 * (H[0, 3] * m1 + H[3, 3] * m2 + H[6, 3] * m3)
        + m3 * (H[0, 6] * m1 + H[3, 6] * m2 + H[6, 6] * m3)
    )

    h[1, 0] += (
        m1 * (H[1, 0] * m1 + H[4, 0] * m2 + H[7, 0] * m3)
        + m2 * (H[1, 3] * m1 + H[4, 3] * m2 + H[7, 3] * m3)
        + m3 * (H[1, 6] * m1 + H[4, 6] * m2 + H[7, 6] * m3)
    )

    h[2, 0] += (
        m1 * (H[2, 0] * m1 + H[5, 0] * m2 + H[8, 0] * m3)
        + m2 * (H[2, 3] * m1 + H[5, 3] * m2 + H[8, 3] * m3)
        + m3 * (H[2, 6] * m1 + H[5, 6] * m2 + H[8, 6] * m3)
    )

    h[0, 1] += (
        m1 * (H[0, 1] * m1 + H[3, 1] * m2 + H[6, 1] * m3)
        + m2 * (H[0, 4] * m1 + H[3, 4] * m2 + H[6, 4] * m3)
        + m3 * (H[0, 7] * m1 + H[3, 7] * m2 + H[6, 7] * m3)
    )

    h[1, 1] += (
        m1 * (H[1, 1] * m1 + H[4, 1] * m2 + H[7, 1] * m3)
        + m2 * (H[1, 4] * m1 + H[4, 4] * m2 + H[7, 4] * m3)
        + m3 * (H[1, 7] * m1 + H[4, 7] * m2 + H[7, 7] * m3)
    )

    h[2, 1] += (
        m1 * (H[2, 1] * m1 + H[5, 1] * m2 + H[8, 1] * m3)
        + m2 * (H[2, 4] * m1 + H[5, 4] * m2 + H[8, 4] * m3)
        + m3 * (H[2, 7] * m1 + H[5, 7] * m2 + H[8, 7] * m3)
    )

    h[0, 2] += (
        m1 * (H[0, 2] * m1 + H[3, 2] * m2 + H[6, 2] * m3)
        + m2 * (H[0, 5] * m1 + H[3, 5] * m2 + H[6, 5] * m3)
        + m3 * (H[0, 8] * m1 + H[3, 8] * m2 + H[6, 8] * m3)
    )

    h[1, 2] += (
        m1 * (H[1, 2] * m1 + H[4, 2] * m2 + H[7, 2] * m3)
        + m2 * (H[1, 5] * m1 + H[4, 5] * m2 + H[7, 5] * m3)
        + m3 * (H[1, 8] * m1 + H[4, 8] * m2 + H[7, 8] * m3)
    )

    h[2, 2] += (
        m1 * (H[2, 2] * m1 + H[5, 2] * m2 + H[8, 2] * m3)
        + m2 * (H[2, 5] * m1 + H[5, 5] * m2 + H[8, 5] * m3)
        + m3 * (H[2, 8] * m1 + H[5, 8] * m2 + H[8, 8] * m3)
    )

    return f, h


@wp.func
def damp_force_and_hessian(
    particle_pos_prev: wp.vec3,
    particle_pos: wp.vec3,
    force: wp.vec3,
    hessian: wp.mat33,
    damping: float,
    dt: float,
):
    displacement = particle_pos_prev - particle_pos
    h_d = hessian * (damping / dt)
    f_d = h_d * displacement

    return force + f_d, hessian + h_d


# @wp.func
# def evaluate_volumetric_neo_hookean_force_and_hessian(
#     tet_id: int,
#     v_order: int,
#     pos_prev: wp.array(dtype=wp.vec3),
#     pos: wp.array(dtype=wp.vec3),
#     tet_indices: wp.array(dtype=wp.int32, ndim=2),
#     Dm_inv: wp.mat33,
#     mu: float,
#     lmbd: float,
#     damping: float,
#     dt: float,
# ) -> tuple[wp.vec3, wp.mat33]:

#     # ============ Get Vertices ============
#     v0 = pos[tet_indices[tet_id, 0]]
#     v1 = pos[tet_indices[tet_id, 1]]
#     v2 = pos[tet_indices[tet_id, 2]]
#     v3 = pos[tet_indices[tet_id, 3]]

#     # ============ Compute rest volume from Dm_inv ============
#     rest_volume = 1.0 / (wp.determinant(Dm_inv) * 6.0)

#     # ============ Deformation Gradient ============
#     Ds = wp.mat33(v1 - v0, v2 - v0, v3 - v0)
#     F = Ds * Dm_inv

#     # ============ Flatten F to vec9 ============
#     f = vec9(
#         F[0,0], F[1,0], F[2,0],
#         F[0,1], F[1,1], F[2,1],
#         F[0,2], F[1,2], F[2,2],
#     )

#     # ============ Useful Quantities ============
#     J = wp.determinant(F)
#     alpha = 1.0 + mu / lmbd
#     F_inv = wp.inverse(F)
#     cof = J * wp.transpose(F_inv)

#     cof_vec = vec9(
#         cof[0,0], cof[1,0], cof[2,0],
#         cof[0,1], cof[1,1], cof[2,1],
#         cof[0,2], cof[1,2], cof[2,2],
#     )

#     # ============ Stress ============
#     P_vec = rest_volume * (mu * f + lmbd * (J - alpha) * cof_vec)

#     # ============ Hessian ============
#     H = (mu * wp.identity(n=9, dtype=float)
#          + lmbd * wp.outer(cof_vec, cof_vec)
#          + compute_cofactor_derivative(F, lmbd * (J - alpha)))
#     H = rest_volume * H

#     # ============ G_i ============
#     G_i = compute_G_matrix(Dm_inv, v_order)

#     # ============ Force & Hessian ============
#     force = -wp.transpose(G_i) * P_vec
#     hessian = wp.transpose(G_i) * H * G_i

#     # ============ Damping ============
#     if damping > 0.0:
#         inv_dt = 1.0 / dt

#         v0_prev = pos_prev[tet_indices[tet_id, 0]]
#         v1_prev = pos_prev[tet_indices[tet_id, 1]]
#         v2_prev = pos_prev[tet_indices[tet_id, 2]]
#         v3_prev = pos_prev[tet_indices[tet_id, 3]]

#         Ds_dot = wp.mat33(
#             (v1 - v1_prev) - (v0 - v0_prev),
#             (v2 - v2_prev) - (v0 - v0_prev),
#             (v3 - v3_prev) - (v0 - v0_prev),
#         ) * inv_dt
#         F_dot = Ds_dot * Dm_inv

#         f_dot = vec9(
#             F_dot[0,0], F_dot[1,0], F_dot[2,0],
#             F_dot[0,1], F_dot[1,1], F_dot[2,1],
#             F_dot[0,2], F_dot[1,2], F_dot[2,2],
#         )

#         P_damp = damping * (H * f_dot)

#         force = force - wp.transpose(G_i) * P_damp
#         hessian = hessian + (damping * inv_dt) * wp.transpose(G_i) * H * G_i

#     return force, hessian


@wp.func
def evaluate_volumetric_neo_hookean_force_and_hessian(
    tet_id: int,
    v_order: int,
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    Dm_inv: wp.mat33,
    mu: float,
    lmbd: float,
    damping: float,
    dt: float,
) -> tuple[wp.vec3, wp.mat33]:
    # ============ Get Vertices ============
    v0 = pos[tet_indices[tet_id, 0]]
    v1 = pos[tet_indices[tet_id, 1]]
    v2 = pos[tet_indices[tet_id, 2]]
    v3 = pos[tet_indices[tet_id, 3]]

    # ============ Compute rest volume from Dm_inv ============
    rest_volume = 1.0 / (wp.determinant(Dm_inv) * 6.0)

    # ============ Deformation Gradient ============
    Ds = wp.matrix_from_cols(v1 - v0, v2 - v0, v3 - v0)
    F = Ds * Dm_inv

    # ============ Flatten F to vec9 ============
    f = vec9(
        F[0, 0],
        F[1, 0],
        F[2, 0],
        F[0, 1],
        F[1, 1],
        F[2, 1],
        F[0, 2],
        F[1, 2],
        F[2, 2],
    )

    # ============ Useful Quantities ============
    J = wp.determinant(F)
    # Guard against division by zero in lambda (Lamé's first parameter)
    # For numerical stability, ensure lmbd has a reasonable minimum magnitude
    lmbd_safe = wp.sign(lmbd) * wp.max(wp.abs(lmbd), 1e-6)
    alpha = 1.0 + mu / lmbd_safe
    # Compute cofactor (adjugate) matrix directly for numerical stability when J ≈ 0
    cof = compute_cofactor(F)

    cof_vec = vec9(
        cof[0, 0],
        cof[1, 0],
        cof[2, 0],
        cof[0, 1],
        cof[1, 1],
        cof[2, 1],
        cof[0, 2],
        cof[1, 2],
        cof[2, 2],
    )

    # ============ Stress ============
    P_vec = rest_volume * (mu * f + lmbd * (J - alpha) * cof_vec)

    # ============ Hessian ============
    H = (
        mu * wp.identity(n=9, dtype=float)
        + lmbd * wp.outer(cof_vec, cof_vec)
        + compute_cofactor_derivative(F, lmbd * (J - alpha))
    )
    H = rest_volume * H

    # ============ Assemble Pointwise Force ============
    if v_order == 0:
        m = wp.vec3(
            -(Dm_inv[0, 0] + Dm_inv[1, 0] + Dm_inv[2, 0]),
            -(Dm_inv[0, 1] + Dm_inv[1, 1] + Dm_inv[2, 1]),
            -(Dm_inv[0, 2] + Dm_inv[1, 2] + Dm_inv[2, 2]),
        )
    elif v_order == 1:
        m = wp.vec3(Dm_inv[0, 0], Dm_inv[0, 1], Dm_inv[0, 2])
    elif v_order == 2:
        m = wp.vec3(Dm_inv[1, 0], Dm_inv[1, 1], Dm_inv[1, 2])
    else:
        m = wp.vec3(Dm_inv[2, 0], Dm_inv[2, 1], Dm_inv[2, 2])

    force, hessian = assemble_tet_vertex_force_and_hessian(P_vec, H, m[0], m[1], m[2])

    # ============ Damping ============
    if damping > 0.0:
        inv_dt = 1.0 / dt

        v0_prev = pos_prev[tet_indices[tet_id, 0]]
        v1_prev = pos_prev[tet_indices[tet_id, 1]]
        v2_prev = pos_prev[tet_indices[tet_id, 2]]
        v3_prev = pos_prev[tet_indices[tet_id, 3]]

        Ds_dot = (
            wp.matrix_from_cols(
                (v1 - v1_prev) - (v0 - v0_prev),
                (v2 - v2_prev) - (v0 - v0_prev),
                (v3 - v3_prev) - (v0 - v0_prev),
            )
            * inv_dt
        )
        F_dot = Ds_dot * Dm_inv

        f_dot = vec9(
            F_dot[0, 0],
            F_dot[1, 0],
            F_dot[2, 0],
            F_dot[0, 1],
            F_dot[1, 1],
            F_dot[2, 1],
            F_dot[0, 2],
            F_dot[1, 2],
            F_dot[2, 2],
        )

        P_damp = damping * (H * f_dot)

        f_damp = wp.vec3(
            -(P_damp[0] * m[0] + P_damp[3] * m[1] + P_damp[6] * m[2]),
            -(P_damp[1] * m[0] + P_damp[4] * m[1] + P_damp[7] * m[2]),
            -(P_damp[2] * m[0] + P_damp[5] * m[1] + P_damp[8] * m[2]),
        )
        force = force + f_damp
        hessian = hessian * (1.0 + damping * inv_dt)

    return force, hessian


# ============ Helper Functions ============


@wp.func
def compute_G_matrix(Dm_inv: wp.mat33, v_order: int) -> mat93:
    """G_i = ∂vec(F)/∂x_i"""

    if v_order == 0:
        m = wp.vec3(
            -(Dm_inv[0, 0] + Dm_inv[1, 0] + Dm_inv[2, 0]),
            -(Dm_inv[0, 1] + Dm_inv[1, 1] + Dm_inv[2, 1]),
            -(Dm_inv[0, 2] + Dm_inv[1, 2] + Dm_inv[2, 2]),
        )
    elif v_order == 1:
        m = wp.vec3(Dm_inv[0, 0], Dm_inv[0, 1], Dm_inv[0, 2])
    elif v_order == 2:
        m = wp.vec3(Dm_inv[1, 0], Dm_inv[1, 1], Dm_inv[1, 2])
    else:
        m = wp.vec3(Dm_inv[2, 0], Dm_inv[2, 1], Dm_inv[2, 2])

    # G = [m[0]*I₃, m[1]*I₃, m[2]*I₃]ᵀ (stacked vertically)
    return mat93(
        m[0],
        0.0,
        0.0,
        0.0,
        m[0],
        0.0,
        0.0,
        0.0,
        m[0],
        m[1],
        0.0,
        0.0,
        0.0,
        m[1],
        0.0,
        0.0,
        0.0,
        m[1],
        m[2],
        0.0,
        0.0,
        0.0,
        m[2],
        0.0,
        0.0,
        0.0,
        m[2],
    )


@wp.func
def compute_cofactor(F: wp.mat33) -> wp.mat33:
    """Compute the cofactor (adjugate) matrix directly without using inverse.

    This is numerically stable even when det(F) ≈ 0, unlike J * transpose(inverse(F)).
    """
    F11, F21, F31 = F[0, 0], F[1, 0], F[2, 0]
    F12, F22, F32 = F[0, 1], F[1, 1], F[2, 1]
    F13, F23, F33 = F[0, 2], F[1, 2], F[2, 2]

    return wp.mat33(
        F22 * F33 - F23 * F32,
        F23 * F31 - F21 * F33,
        F21 * F32 - F22 * F31,
        F13 * F32 - F12 * F33,
        F11 * F33 - F13 * F31,
        F12 * F31 - F11 * F32,
        F12 * F23 - F13 * F22,
        F13 * F21 - F11 * F23,
        F11 * F22 - F12 * F21,
    )


@wp.func
def compute_cofactor_derivative(F: wp.mat33, scale: float) -> mat99:
    """scale * ∂cof(F)/∂F"""

    F11, F21, F31 = F[0, 0], F[1, 0], F[2, 0]
    F12, F22, F32 = F[0, 1], F[1, 1], F[2, 1]
    F13, F23, F33 = F[0, 2], F[1, 2], F[2, 2]

    return mat99(
        0.0,
        0.0,
        0.0,
        0.0,
        scale * F33,
        -scale * F23,
        0.0,
        -scale * F32,
        scale * F22,
        0.0,
        0.0,
        0.0,
        -scale * F33,
        0.0,
        scale * F13,
        scale * F32,
        0.0,
        -scale * F12,
        0.0,
        0.0,
        0.0,
        scale * F23,
        -scale * F13,
        0.0,
        -scale * F22,
        scale * F12,
        0.0,
        0.0,
        -scale * F33,
        scale * F23,
        0.0,
        0.0,
        0.0,
        0.0,
        scale * F31,
        -scale * F21,
        scale * F33,
        0.0,
        -scale * F13,
        0.0,
        0.0,
        0.0,
        -scale * F31,
        0.0,
        scale * F11,
        -scale * F23,
        scale * F13,
        0.0,
        0.0,
        0.0,
        0.0,
        scale * F21,
        -scale * F11,
        0.0,
        0.0,
        scale * F32,
        -scale * F22,
        0.0,
        -scale * F31,
        scale * F21,
        0.0,
        0.0,
        0.0,
        -scale * F32,
        0.0,
        scale * F12,
        scale * F31,
        0.0,
        -scale * F11,
        0.0,
        0.0,
        0.0,
        scale * F22,
        -scale * F12,
        0.0,
        -scale * F21,
        scale * F11,
        0.0,
        0.0,
        0.0,
        0.0,
    )


@wp.kernel
def _count_num_adjacent_edges(
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
def _fill_adjacent_edges(
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
def _count_num_adjacent_faces(
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
def _fill_adjacent_faces(
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
def _count_num_adjacent_springs(
    springs_array: wp.array(dtype=wp.int32), num_vertex_adjacent_springs: wp.array(dtype=wp.int32)
):
    num_springs = springs_array.shape[0] / 2
    for spring_id in range(num_springs):
        v0 = springs_array[spring_id * 2]
        v1 = springs_array[spring_id * 2 + 1]

        num_vertex_adjacent_springs[v0] = num_vertex_adjacent_springs[v0] + 1
        num_vertex_adjacent_springs[v1] = num_vertex_adjacent_springs[v1] + 1


@wp.kernel
def _fill_adjacent_springs(
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
def _count_num_adjacent_tets(
    tet_indices: wp.array(dtype=wp.int32, ndim=2), num_vertex_adjacent_tets: wp.array(dtype=wp.int32)
):
    for tet in range(tet_indices.shape[0]):
        v0 = tet_indices[tet, 0]
        v1 = tet_indices[tet, 1]
        v2 = tet_indices[tet, 2]
        v3 = tet_indices[tet, 3]

        num_vertex_adjacent_tets[v0] = num_vertex_adjacent_tets[v0] + 1
        num_vertex_adjacent_tets[v1] = num_vertex_adjacent_tets[v1] + 1
        num_vertex_adjacent_tets[v2] = num_vertex_adjacent_tets[v2] + 1
        num_vertex_adjacent_tets[v3] = num_vertex_adjacent_tets[v3] + 1


@wp.kernel
def _fill_adjacent_tets(
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    vertex_adjacent_tets_offsets: wp.array(dtype=wp.int32),
    vertex_adjacent_tets_fill_count: wp.array(dtype=wp.int32),
    vertex_adjacent_tets: wp.array(dtype=wp.int32),
):
    for tet in range(tet_indices.shape[0]):
        v0 = tet_indices[tet, 0]
        v1 = tet_indices[tet, 1]
        v2 = tet_indices[tet, 2]
        v3 = tet_indices[tet, 3]

        fill_count_v0 = vertex_adjacent_tets_fill_count[v0]
        buffer_offset_v0 = vertex_adjacent_tets_offsets[v0]
        vertex_adjacent_tets[buffer_offset_v0 + fill_count_v0 * 2] = tet
        vertex_adjacent_tets[buffer_offset_v0 + fill_count_v0 * 2 + 1] = 0
        vertex_adjacent_tets_fill_count[v0] = fill_count_v0 + 1

        fill_count_v1 = vertex_adjacent_tets_fill_count[v1]
        buffer_offset_v1 = vertex_adjacent_tets_offsets[v1]
        vertex_adjacent_tets[buffer_offset_v1 + fill_count_v1 * 2] = tet
        vertex_adjacent_tets[buffer_offset_v1 + fill_count_v1 * 2 + 1] = 1
        vertex_adjacent_tets_fill_count[v1] = fill_count_v1 + 1

        fill_count_v2 = vertex_adjacent_tets_fill_count[v2]
        buffer_offset_v2 = vertex_adjacent_tets_offsets[v2]
        vertex_adjacent_tets[buffer_offset_v2 + fill_count_v2 * 2] = tet
        vertex_adjacent_tets[buffer_offset_v2 + fill_count_v2 * 2 + 1] = 2
        vertex_adjacent_tets_fill_count[v2] = fill_count_v2 + 1

        fill_count_v3 = vertex_adjacent_tets_fill_count[v3]
        buffer_offset_v3 = vertex_adjacent_tets_offsets[v3]
        vertex_adjacent_tets[buffer_offset_v3 + fill_count_v3 * 2] = tet
        vertex_adjacent_tets[buffer_offset_v3 + fill_count_v3 * 2 + 1] = 3
        vertex_adjacent_tets_fill_count[v3] = fill_count_v3 + 1


@wp.kernel
def _test_compute_force_element_adjacency(
    adjacency: ParticleForceElementAdjacencyInfo,
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
def evaluate_stvk_force_hessian(
    face: int,
    v_order: int,
    pos: wp.array(dtype=wp.vec3),
    pos_anchor: wp.array(dtype=wp.vec3),
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
        x0_prev = pos_anchor[v0]
        x01_prev = pos_anchor[v1] - x0_prev
        x02_prev = pos_anchor[v2] - x0_prev

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
    pos_anchor: wp.array(dtype=wp.vec3),
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
        x_prev0 = pos_anchor[vi0]
        x_prev1 = pos_anchor[vi1]
        x_prev2 = pos_anchor[vi2]
        x_prev3 = pos_anchor[vi3]

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
    pos_anchor: wp.array(dtype=wp.vec3),
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
        pos_anchor,
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
        c1_prev = pos_anchor[e1_v1] + (pos_anchor[e1_v2] - pos_anchor[e1_v1]) * s
        c2_prev = pos_anchor[e2_v1] + (pos_anchor[e2_v2] - pos_anchor[e2_v1]) * t

        dx = (c1 - c1_prev) - (c2 - c2_prev)
        axis_1, axis_2 = orthonormal_basis(collision_normal)

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
            displacement = pos_anchor[e1_v1] - e1_v1_pos
        elif v_order == 1:
            displacement = pos_anchor[e1_v2] - e1_v2_pos
        elif v_order == 2:
            displacement = pos_anchor[e2_v1] - e2_v1_pos
        else:
            displacement = pos_anchor[e2_v2] - e2_v2_pos

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
    pos_anchor: wp.array(dtype=wp.vec3),
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
        c1_prev = pos_anchor[e1_v1] + (pos_anchor[e1_v2] - pos_anchor[e1_v1]) * s
        c2_prev = pos_anchor[e2_v1] + (pos_anchor[e2_v2] - pos_anchor[e2_v1]) * t

        dx = (c1 - c1_prev) - (c2 - c2_prev)
        axis_1, axis_2 = orthonormal_basis(collision_normal)

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

        displacement_0 = pos_anchor[e1_v1] - e1_v1_pos
        displacement_1 = pos_anchor[e1_v2] - e1_v2_pos

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
    pos_anchor: wp.array(dtype=wp.vec3),
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

    closest_p, bary, _feature_type = triangle_closest_point(a, b, c, p)

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
        dx_v = p - pos_anchor[v]

        closest_p_prev = (
            bary[0] * pos_anchor[tri_indices[tri, 0]]
            + bary[1] * pos_anchor[tri_indices[tri, 1]]
            + bary[2] * pos_anchor[tri_indices[tri, 2]]
        )

        dx = dx_v - (closest_p - closest_p_prev)

        e0, e1 = orthonormal_basis(collision_normal)

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
            displacement = pos_anchor[tri_indices[tri, 0]] - a
        elif v_order == 1:
            displacement = pos_anchor[tri_indices[tri, 1]] - b
        elif v_order == 2:
            displacement = pos_anchor[tri_indices[tri, 2]] - c
        else:
            displacement = pos_anchor[v] - p

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
    pos_anchor: wp.array(dtype=wp.vec3),
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

    closest_p, bary, _feature_type = triangle_closest_point(a, b, c, p)

    diff = p - closest_p
    dis = wp.length(diff)
    collision_normal = diff / dis

    if 0.0 < dis < collision_radius:
        bs = wp.vec4(-bary[0], -bary[1], -bary[2], 1.0)

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

        collision_force = -dEdD * collision_normal
        collision_hessian = d2E_dDdD * wp.outer(collision_normal, collision_normal)

        # friction force
        dx_v = p - pos_anchor[v]

        closest_p_prev = (
            bary[0] * pos_anchor[tri_indices[tri, 0]]
            + bary[1] * pos_anchor[tri_indices[tri, 1]]
            + bary[2] * pos_anchor[tri_indices[tri, 2]]
        )

        dx = dx_v - (closest_p - closest_p_prev)

        e0, e1 = orthonormal_basis(collision_normal)

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

        displacement_0 = pos_anchor[tri_indices[tri, 0]] - a
        displacement_1 = pos_anchor[tri_indices[tri, 1]] - b
        displacement_2 = pos_anchor[tri_indices[tri, 2]] - c
        displacement_3 = pos_anchor[v] - p

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
    gravity: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    external_force: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    inertia_out: wp.array(dtype=wp.vec3),
    displacements_out: wp.array(dtype=wp.vec3),
):
    particle = wp.tid()

    pos_prev[particle] = pos[particle]
    if not particle_flags[particle] & ParticleFlags.ACTIVE or inv_mass[particle] == 0:
        inertia_out[particle] = pos_prev[particle]
        if displacements_out:
            displacements_out[particle] = wp.vec3(0.0, 0.0, 0.0)
        return
    vel_new = vel[particle] + (gravity[0] + external_force[particle] * inv_mass[particle]) * dt
    inertia = pos[particle] + vel_new * dt
    inertia_out[particle] = inertia
    if displacements_out:
        displacements_out[particle] = vel_new * dt


@wp.kernel
def compute_particle_conservative_bound(
    # inputs
    conservative_bound_relaxation: float,
    collision_query_radius: float,
    adjacency: ParticleForceElementAdjacencyInfo,
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
        tri_index, _vertex_order = get_vertex_adjacent_face_id_order(
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
def update_velocity(
    dt: float, pos_prev: wp.array(dtype=wp.vec3), pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3)
):
    particle = wp.tid()
    vel[particle] = (pos[particle] - pos_prev[particle]) / dt


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
def accumulate_self_contact_force_and_hessian(
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


def _csr_row(vals: np.ndarray, offs: np.ndarray, i: int) -> np.ndarray:
    """Extract CSR row `i` from the flattened adjacency arrays."""
    return vals[offs[i] : offs[i + 1]]


def set_to_csr(
    list_of_sets: list[set[int]], dtype: np.dtype = np.int32, sort: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of integer sets into CSR (Compressed Sparse Row) structure.
    Args:
        list_of_sets: Iterable where each entry is a set of ints.
        dtype: Output dtype for the flattened arrays.
        sort: Whether to sort each row when writing into ``flat``.
    Returns:
        A tuple ``(flat, offsets)`` representing the CSR values and offsets.
    """
    offsets = np.zeros(len(list_of_sets) + 1, dtype=dtype)
    sizes = np.fromiter((len(s) for s in list_of_sets), count=len(list_of_sets), dtype=dtype)
    np.cumsum(sizes, out=offsets[1:])
    flat = np.empty(offsets[-1], dtype=dtype)
    idx = 0
    for s in list_of_sets:
        if sort:
            arr = np.fromiter(sorted(s), count=len(s), dtype=dtype)
        else:
            arr = np.fromiter(s, count=len(s), dtype=dtype)

        flat[idx : idx + len(arr)] = arr
        idx += len(arr)
    return flat, offsets


def one_ring_vertices(
    v: int, edge_indices: np.ndarray, v_adj_edges: np.ndarray, v_adj_edges_offsets: np.ndarray
) -> np.ndarray:
    """
    Find immediate neighboring vertices that share an edge with vertex `v`.
    Args:
        v: Vertex index whose neighborhood is queried.
        edge_indices: Array of shape [num_edges, 4] storing edge endpoint indices.
        v_adj_edges: Flattened CSR adjacency array listing edge ids and local order.
        v_adj_edges_offsets: CSR offsets indexing into `v_adj_edges`.
    Returns:
        Sorted array of neighboring vertex indices, excluding `v`.
    """
    e_u = edge_indices[:, 2]
    e_v = edge_indices[:, 3]
    # preserve only the adjacent edge information, remove the order information
    inc_edges = _csr_row(v_adj_edges, v_adj_edges_offsets, v)[::2]
    inc_edges_order = _csr_row(v_adj_edges, v_adj_edges_offsets, v)[1::2]
    if inc_edges.size == 0:
        return np.empty(0)
    us = e_u[inc_edges[np.where(inc_edges_order >= 2)]]
    vs = e_v[inc_edges[np.where(inc_edges_order >= 2)]]

    assert (np.logical_or(us == v, vs == v)).all()
    nbrs = np.unique(np.concatenate([us, vs]))
    return nbrs[nbrs != v]


def leq_n_ring_vertices(
    v: int, edge_indices: np.ndarray, n: int, v_adj_edges: np.ndarray, v_adj_edges_offsets: np.ndarray
) -> np.ndarray:
    """
    Find all vertices within n-ring distance of vertex v using BFS.
    Args:
        v: Starting vertex index
        edge_indices: Edge connectivity array
        n: Maximum ring distance
        v_adj_edges: CSR values for vertex-edge adjacency
        v_adj_edges_offsets: CSR offsets for vertex-edge adjacency
    Returns:
        Array of all vertices within n-ring distance, including v itself
    """
    visited = {v}
    frontier = {v}
    for _ in range(n):
        next_frontier = set()
        for u in frontier:
            for w in one_ring_vertices(u, edge_indices, v_adj_edges, v_adj_edges_offsets):  # iterable of neighbors of u
                if w not in visited:
                    visited.add(w)
                    next_frontier.add(w)
        if not next_frontier:
            break
        frontier = next_frontier
    return np.fromiter(visited, dtype=int)


def build_vertex_n_ring_tris_collision_filter(
    n: int,
    num_vertices: int,
    edge_indices: np.ndarray,
    v_adj_edges: np.ndarray,
    v_adj_edges_offsets: np.ndarray,
    v_adj_faces: np.ndarray,
    v_adj_faces_offsets: np.ndarray,
):
    """
    For each vertex v, return ONLY triangles adjacent to v's one ring neighbor vertices.
    Excludes triangles incident to v itself (dist 0).
    Returns:
      v_two_flat, v_two_offs: CSR of strict-2-ring triangle ids per vertex
    """

    if n <= 1:
        return None, None

    v_nei_tri_sets = [set() for _ in range(num_vertices)]

    for v in range(num_vertices):
        # distance-1 vertices

        if n == 2:
            ring_n_minus_1 = one_ring_vertices(v, edge_indices, v_adj_edges, v_adj_edges_offsets)
        else:
            ring_n_minus_1 = leq_n_ring_vertices(v, edge_indices, n - 1, v_adj_edges, v_adj_edges_offsets)

        ring_1_tri_set = set(_csr_row(v_adj_faces, v_adj_faces_offsets, v)[::2])

        nei_tri_set = v_nei_tri_sets[v]
        for w in ring_n_minus_1:
            if w != v:
                # preserve only the adjacent edge information, remove the order information
                nei_tri_set.update(_csr_row(v_adj_faces, v_adj_faces_offsets, w)[::2])

        nei_tri_set.difference_update(ring_1_tri_set)

    return v_nei_tri_sets


def build_edge_n_ring_edge_collision_filter(
    n: int,
    edge_indices: np.ndarray,
    v_adj_edges: np.ndarray,
    v_adj_edges_offsets: np.ndarray,
):
    """
    For each vertex v, return ONLY triangles adjacent to v's one ring neighbor vertices.
    Excludes triangles incident to v itself (dist 0).
    Returns:
      v_two_flat, v_two_offs: CSR of strict-2-ring triangle ids per vertex
    """

    if n <= 1:
        return None, None

    edge_nei_edge_sets = [set() for _ in range(edge_indices.shape[0])]

    for e_idx in range(edge_indices.shape[0]):
        # distance-1 vertices
        v1 = edge_indices[e_idx, 2]
        v2 = edge_indices[e_idx, 3]

        if n == 2:
            ring_n_minus_1_v1 = one_ring_vertices(v1, edge_indices, v_adj_edges, v_adj_edges_offsets)
            ring_n_minus_1_v2 = one_ring_vertices(v2, edge_indices, v_adj_edges, v_adj_edges_offsets)
        else:
            ring_n_minus_1_v1 = leq_n_ring_vertices(v1, edge_indices, n - 1, v_adj_edges, v_adj_edges_offsets)
            ring_n_minus_1_v2 = leq_n_ring_vertices(v2, edge_indices, n - 1, v_adj_edges, v_adj_edges_offsets)

        all_neighbors = set(ring_n_minus_1_v1)
        all_neighbors.update(ring_n_minus_1_v2)

        ring_1_edge_set = set(_csr_row(v_adj_edges, v_adj_edges_offsets, v1)[::2])
        ring_2_edge_set = set(_csr_row(v_adj_edges, v_adj_edges_offsets, v2)[::2])

        nei_edge_set = edge_nei_edge_sets[e_idx]
        for w in all_neighbors:
            if w != v1 and w != v2:
                # preserve only the adjacent edge information, remove the order information
                # nei_tri_set.update(_csr_row(v_adj_faces, v_adj_faces_offsets, w)[::2])
                adj_edges = _csr_row(v_adj_edges, v_adj_edges_offsets, w)[::2]
                adj_edges_order = _csr_row(v_adj_edges, v_adj_edges_offsets, w)[1::2]
                adj_collision_edges = adj_edges[np.where(adj_edges_order >= 2)]
                nei_edge_set.update(adj_collision_edges)

        nei_edge_set.difference_update(ring_1_edge_set)
        nei_edge_set.difference_update(ring_2_edge_set)

    return edge_nei_edge_sets


@wp.func
def evaluate_spring_force_and_hessian(
    particle_idx: int,
    spring_idx: int,
    dt: float,
    pos: wp.array(dtype=wp.vec3),
    pos_anchor: wp.array(dtype=wp.vec3),
    spring_indices: wp.array(dtype=int),
    spring_rest_length: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
):
    v0 = spring_indices[spring_idx * 2]
    v1 = spring_indices[spring_idx * 2 + 1]

    diff = pos[v0] - pos[v1]
    spring_length = wp.length(diff)
    # Clamp to epsilon to avoid division by zero for coincident vertices
    spring_length = wp.max(spring_length, 1e-8)
    l0 = spring_rest_length[spring_idx]

    force_sign = 1.0 if particle_idx == v0 else -1.0

    spring_force = force_sign * spring_stiffness[spring_idx] * (l0 - spring_length) / spring_length * diff
    spring_hessian = spring_stiffness[spring_idx] * (
        wp.identity(3, float)
        - (l0 / spring_length) * (wp.identity(3, float) - wp.outer(diff, diff) / (spring_length * spring_length))
    )

    # compute damping
    h_d = spring_hessian * (spring_damping[spring_idx] / dt)

    f_d = h_d * (pos_anchor[particle_idx] - pos[particle_idx])

    spring_force = spring_force + f_d
    spring_hessian = spring_hessian + h_d

    return spring_force, spring_hessian


@wp.func
def evaluate_spring_force_and_hessian_both_vertices(
    spring_idx: int,
    dt: float,
    pos: wp.array(dtype=wp.vec3),
    pos_anchor: wp.array(dtype=wp.vec3),
    spring_indices: wp.array(dtype=int),
    spring_rest_length: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
):
    """Evaluate spring force and hessian for both vertices of a spring.

    Returns forces and hessians for v0 and v1 respectively.
    """
    v0 = spring_indices[spring_idx * 2]
    v1 = spring_indices[spring_idx * 2 + 1]

    diff = pos[v0] - pos[v1]
    spring_length = wp.length(diff)
    # Clamp to epsilon to avoid division by zero for coincident vertices
    spring_length = wp.max(spring_length, 1e-8)
    l0 = spring_rest_length[spring_idx]

    # Base spring force for v0 (v1 gets the opposite)
    base_force = spring_stiffness[spring_idx] * (l0 - spring_length) / spring_length * diff

    # Hessian is the same for both vertices (symmetric)
    spring_hessian = spring_stiffness[spring_idx] * (
        wp.identity(3, float)
        - (l0 / spring_length) * (wp.identity(3, float) - wp.outer(diff, diff) / (spring_length * spring_length))
    )

    # Compute damping hessian contribution
    h_d = spring_hessian * (spring_damping[spring_idx] / dt)

    # Damping force for each vertex
    f_d_v0 = h_d * (pos_anchor[v0] - pos[v0])
    f_d_v1 = h_d * (pos_anchor[v1] - pos[v1])

    # Total force and hessian for each vertex
    force_v0 = base_force + f_d_v0
    force_v1 = -base_force + f_d_v1  # Opposite direction for v1
    hessian_total = spring_hessian + h_d

    return v0, v1, force_v0, force_v1, hessian_total


@wp.kernel
def accumulate_spring_force_and_hessian(
    # inputs
    dt: float,
    current_color: int,
    pos_anchor: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    particle_colors: wp.array(dtype=int),
    num_springs: int,
    # spring constraints
    spring_indices: wp.array(dtype=int),
    spring_rest_length: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
    # outputs: particle force and hessian
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
):
    """Accumulate spring forces and hessians, parallelized by springs.

    Each thread handles one spring and uses atomic operations to add
    forces and hessians to vertices with the current color.
    """
    spring_idx = wp.tid()

    if spring_idx < num_springs:
        v0 = spring_indices[spring_idx * 2]
        v1 = spring_indices[spring_idx * 2 + 1]

        c_v0 = particle_colors[v0]
        c_v1 = particle_colors[v1]

        # Only evaluate if at least one vertex has the current color
        if c_v0 == current_color or c_v1 == current_color:
            _, _, force_v0, force_v1, hessian = evaluate_spring_force_and_hessian_both_vertices(
                spring_idx,
                dt,
                pos,
                pos_anchor,
                spring_indices,
                spring_rest_length,
                spring_stiffness,
                spring_damping,
            )

            # Only add to vertices with the current color
            if c_v0 == current_color:
                wp.atomic_add(particle_forces, v0, force_v0)
                wp.atomic_add(particle_hessians, v0, hessian)
            if c_v1 == current_color:
                wp.atomic_add(particle_forces, v1, force_v1)
                wp.atomic_add(particle_hessians, v1, hessian)


@wp.kernel
def accumulate_contact_force_and_hessian_no_self_contact(
    # inputs
    dt: float,
    current_color: int,
    pos_anchor: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    particle_colors: wp.array(dtype=int),
    # body-particle contact
    friction_epsilon: float,
    particle_radius: wp.array(dtype=float),
    body_particle_contact_particle: wp.array(dtype=int),
    body_particle_contact_count: wp.array(dtype=int),
    body_particle_contact_max: int,
    # per-contact soft AVBD parameters for body-particle contacts (shared with rigid side)
    body_particle_contact_penalty_k: wp.array(dtype=float),
    body_particle_contact_material_kd: wp.array(dtype=float),
    body_particle_contact_material_mu: wp.array(dtype=float),
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

    particle_body_contact_count = min(body_particle_contact_max, body_particle_contact_count[0])

    if t_id < particle_body_contact_count:
        particle_idx = body_particle_contact_particle[t_id]

        if particle_colors[particle_idx] == current_color:
            # Read per-contact AVBD penalty and material properties shared with the rigid side
            contact_ke = body_particle_contact_penalty_k[t_id]
            contact_kd = body_particle_contact_material_kd[t_id]
            contact_mu = body_particle_contact_material_mu[t_id]

            body_contact_force, body_contact_hessian = evaluate_body_particle_contact(
                particle_idx,
                pos[particle_idx],
                pos_anchor[particle_idx],
                t_id,
                contact_ke,
                contact_kd,
                contact_mu,
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


# =============================================================================
# Planar DAT (Divide and Truncate) kernels
# =============================================================================


@wp.func
def segment_plane_intersects(
    v: wp.vec3,
    delta_v: wp.vec3,
    n: wp.vec3,
    d: wp.vec3,
    eps_parallel: float,  # e.g., 1e-8
    eps_intersect_near: float,  # e.g., 1e-8
    eps_intersect_far: float,  # e.g., 1e-8
    coplanar_counts: bool,  # True if you want a coplanar segment to count as "hit"
) -> bool:
    # Plane eq: n·(p - d) = 0
    # Segment: p(t) = v + t * delta_v,  t in [0, 1]
    nv = wp.dot(n, delta_v)
    num = -wp.dot(n, v - d)

    # Parallel (or nearly): either coplanar or no hit
    if wp.abs(nv) < eps_parallel:
        return coplanar_counts and (wp.abs(num) < eps_parallel)

    t = num / nv
    # consider tiny tolerance at ends
    return (t >= eps_intersect_near) and (t <= 1.0 + eps_intersect_far)


@wp.func
def create_vertex_triangle_division_plane_closest_pt(
    v: wp.vec3,
    delta_v: wp.vec3,
    t1: wp.vec3,
    delta_t1: wp.vec3,
    t2: wp.vec3,
    delta_t2: wp.vec3,
    t3: wp.vec3,
    delta_t3: wp.vec3,
):
    """
    n points to the vertex side
    """
    closest_p, _bary, _feature_type = triangle_closest_point(t1, t2, t3, v)

    n_hat = v - closest_p

    if wp.length(n_hat) < 1e-12:
        return wp.vector(False, False, False, False, length=4, dtype=wp.bool), wp.vec3(0.0), v

    n = wp.normalize(n_hat)

    delta_v_n = wp.max(-wp.dot(n, delta_v), 0.0)
    delta_t_n = wp.max(
        wp.vec4(
            wp.dot(n, delta_t1),
            wp.dot(n, delta_t2),
            wp.dot(n, delta_t3),
            0.0,
        )
    )

    if delta_t_n + delta_v_n == 0.0:
        d = closest_p + 0.5 * n_hat
    else:
        lmbd = delta_t_n / (delta_t_n + delta_v_n)
        lmbd = wp.clamp(lmbd, 0.05, 0.95)
        d = closest_p + lmbd * n_hat

    if delta_v_n == 0.0:
        is_dummy_for_v = True
    else:
        is_dummy_for_v = not segment_plane_intersects(v, delta_v, n, d, 1e-6, -1e-8, 1e-8, False)

    if delta_t_n == 0.0:
        is_dummy_for_t_1 = True
        is_dummy_for_t_2 = True
        is_dummy_for_t_3 = True
    else:
        is_dummy_for_t_1 = not segment_plane_intersects(t1, delta_t1, n, d, 1e-6, -1e-8, 1e-8, False)
        is_dummy_for_t_2 = not segment_plane_intersects(t2, delta_t2, n, d, 1e-6, -1e-8, 1e-8, False)
        is_dummy_for_t_3 = not segment_plane_intersects(t3, delta_t3, n, d, 1e-6, -1e-8, 1e-8, False)

    return (
        wp.vector(is_dummy_for_v, is_dummy_for_t_1, is_dummy_for_t_2, is_dummy_for_t_3, length=4, dtype=wp.bool),
        n,
        d,
    )


@wp.func
def robust_edge_pair_normal(
    e0_v0_pos: wp.vec3,
    e0_v1_pos: wp.vec3,
    e1_v0_pos: wp.vec3,
    e1_v1_pos: wp.vec3,
    eps: float = 1.0e-6,
) -> wp.vec3:
    # Edge directions
    dir0 = e0_v1_pos - e0_v0_pos
    dir1 = e1_v1_pos - e1_v0_pos

    len0 = wp.length(dir0)
    len1 = wp.length(dir1)

    if len0 > eps:
        dir0 = dir0 / len0
    else:
        dir0 = wp.vec3(0.0, 0.0, 0.0)

    if len1 > eps:
        dir1 = dir1 / len1
    else:
        dir1 = wp.vec3(0.0, 0.0, 0.0)

    # Primary: cross of two valid directions
    n = wp.cross(dir0, dir1)
    len_n = wp.length(n)
    if len_n > eps:
        return n / len_n

    # Parallel or degenerate: pick best non-zero direction
    reference = dir0
    if wp.length(reference) <= eps:
        reference = dir1

    if wp.length(reference) <= eps:
        # Both edges collapsed: fall back to canonical axis
        return wp.vec3(1.0, 0.0, 0.0)

    # Try bridge vector between midpoints
    bridge = 0.5 * ((e1_v0_pos + e1_v1_pos) - (e0_v0_pos + e0_v1_pos))
    bridge_len = wp.length(bridge)
    if bridge_len > eps:
        n = wp.cross(reference, bridge / bridge_len)
        len_n = wp.length(n)
        if len_n > eps:
            return n / len_n

    # Use an axis guaranteed (numerically) to be non-parallel
    fallback_axis = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(wp.dot(reference, fallback_axis)) > 0.9:
        fallback_axis = wp.vec3(0.0, 1.0, 0.0)

    n = wp.cross(reference, fallback_axis)
    len_n = wp.length(n)
    if len_n > eps:
        return n / len_n

    # Final guard: use the remaining canonical axis
    fallback_axis = wp.vec3(0.0, 0.0, 1.0)
    n = wp.cross(reference, fallback_axis)
    len_n = wp.length(n)
    if len_n > eps:
        return n / len_n

    return wp.vec3(1.0, 0.0, 0.0)


@wp.func
def create_edge_edge_division_plane_closest_pt(
    e0_v0_pos: wp.vec3,
    delta_e0_v0: wp.vec3,
    e0_v1_pos: wp.vec3,
    delta_e0_v1: wp.vec3,
    e1_v0_pos: wp.vec3,
    delta_e1_v0: wp.vec3,
    e1_v1_pos: wp.vec3,
    delta_e1_v1: wp.vec3,
):
    st = wp.closest_point_edge_edge(e0_v0_pos, e0_v1_pos, e1_v0_pos, e1_v1_pos, 1e-6)
    s = st[0]
    t = st[1]
    c1 = e0_v0_pos + (e0_v1_pos - e0_v0_pos) * s
    c2 = e1_v0_pos + (e1_v1_pos - e1_v0_pos) * t

    n_hat = c1 - c2

    if wp.length(n_hat) < 1e-12:
        return (
            wp.vector(False, False, False, False, length=4, dtype=wp.bool),
            robust_edge_pair_normal(e0_v0_pos, e0_v1_pos, e1_v0_pos, e1_v1_pos),
            c1 * 0.5 + c2 * 0.5,
        )

    n = wp.normalize(n_hat)

    delta_e0 = wp.max(
        wp.vec3(
            -wp.dot(n, delta_e0_v0),
            -wp.dot(n, delta_e0_v1),
            0.0,
        )
    )
    delta_e1 = wp.max(
        wp.vec3(
            wp.dot(n, delta_e1_v0),
            wp.dot(n, delta_e1_v1),
            0.0,
        )
    )

    if delta_e0 + delta_e1 == 0.0:
        d = c2 + 0.5 * n_hat
    else:
        lmbd = delta_e1 / (delta_e1 + delta_e0)

        lmbd = wp.clamp(lmbd, 0.05, 0.95)
        d = c2 + lmbd * n_hat

    if delta_e0 == 0.0:
        is_dummy_for_e0_v0 = True
        is_dummy_for_e0_v1 = True
    else:
        is_dummy_for_e0_v0 = not segment_plane_intersects(e0_v0_pos, delta_e0_v0, n, d, 1e-6, -1e-8, 1e-6, False)
        is_dummy_for_e0_v1 = not segment_plane_intersects(e0_v1_pos, delta_e0_v1, n, d, 1e-6, -1e-8, 1e-6, False)

    if delta_e1 == 0.0:
        is_dummy_for_e1_v0 = True
        is_dummy_for_e1_v1 = True
    else:
        is_dummy_for_e1_v0 = not segment_plane_intersects(e1_v0_pos, delta_e1_v0, n, d, 1e-6, -1e-8, 1e-6, False)
        is_dummy_for_e1_v1 = not segment_plane_intersects(e1_v1_pos, delta_e1_v1, n, d, 1e-6, -1e-8, 1e-6, False)

    return (
        wp.vector(
            is_dummy_for_e0_v0, is_dummy_for_e0_v1, is_dummy_for_e1_v0, is_dummy_for_e1_v1, length=4, dtype=wp.bool
        ),
        n,
        d,
    )


@wp.func
def planar_truncation(
    v: wp.vec3, delta_v: wp.vec3, n: wp.vec3, d: wp.vec3, eps: float, gamma_r: float, gamma_min: float = 1e-3
):
    nv = wp.dot(n, delta_v)
    num = wp.dot(n, d - v)

    # Parallel (or nearly): do not truncate
    if wp.abs(nv) < eps:
        return delta_v

    t = num / nv

    t = wp.max(wp.min(t * gamma_r, t - gamma_min), 0.0)
    if t >= 1:
        return delta_v
    else:
        return t * delta_v


@wp.func
def planar_truncation_t(
    v: wp.vec3, delta_v: wp.vec3, n: wp.vec3, d: wp.vec3, eps: float, gamma_r: float, gamma_min: float = 1e-3
):
    denom = wp.dot(n, delta_v)

    # Parallel (or nearly parallel) → no intersection
    if wp.abs(denom) < eps:
        return 1.0

    # Solve: dot(n, v + t*delta_v - d) = 0
    t = wp.dot(n, d - v) / denom

    if t < 0:
        return 1.0

    t = wp.clamp(wp.min(t * gamma_r, t - gamma_min), 0.0, 1.0)
    return t


@wp.kernel
def apply_planar_truncation_parallel_by_collision(
    # inputs
    pos: wp.array(dtype=wp.vec3),
    displacement_in: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_info_array: wp.array(dtype=TriMeshCollisionInfo),
    parallel_eps: float,
    gamma: float,
    truncation_t_out: wp.array(dtype=float),
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

                e1_v1_pos = pos[e1_v1]
                e1_v2_pos = pos[e1_v2]

                delta_e1_v1 = displacement_in[e1_v1]
                delta_e1_v2 = displacement_in[e1_v2]

                e2_v1 = edge_indices[e2_idx, 2]
                e2_v2 = edge_indices[e2_idx, 3]

                e2_v1_pos = pos[e2_v1]
                e2_v2_pos = pos[e2_v2]

                delta_e2_v1 = displacement_in[e2_v1]
                delta_e2_v2 = displacement_in[e2_v2]

                # n points to the edge 1 side
                is_dummy, n, d = create_edge_edge_division_plane_closest_pt(
                    e1_v1_pos,
                    delta_e1_v1,
                    e1_v2_pos,
                    delta_e1_v2,
                    e2_v1_pos,
                    delta_e2_v1,
                    e2_v2_pos,
                    delta_e2_v2,
                )

                # For each, check the corresponding is_dummy entry in the vec4 is_dummy
                if not is_dummy[0]:
                    t = planar_truncation_t(e1_v1_pos, delta_e1_v1, n, d, parallel_eps, gamma)
                    wp.atomic_min(truncation_t_out, e1_v1, t)
                if not is_dummy[1]:
                    t = planar_truncation_t(e1_v2_pos, delta_e1_v2, n, d, parallel_eps, gamma)
                    wp.atomic_min(truncation_t_out, e1_v2, t)
                if not is_dummy[2]:
                    t = planar_truncation_t(e2_v1_pos, delta_e2_v1, n, d, parallel_eps, gamma)
                    wp.atomic_min(truncation_t_out, e2_v1, t)
                if not is_dummy[3]:
                    t = planar_truncation_t(e2_v2_pos, delta_e2_v2, n, d, parallel_eps, gamma)
                    wp.atomic_min(truncation_t_out, e2_v2, t)

                # planar truncation for 2 sides
            collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE

    # process vertex-triangle collisions
    if primitive_id < collision_info.vertex_colliding_triangles_buffer_sizes.shape[0]:
        particle_idx = primitive_id

        colliding_particle_pos = pos[particle_idx]
        colliding_particle_displacement = displacement_in[particle_idx]

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

                t1 = pos[tri_a]
                t2 = pos[tri_b]
                t3 = pos[tri_c]
                delta_t1 = displacement_in[tri_a]
                delta_t2 = displacement_in[tri_b]
                delta_t3 = displacement_in[tri_c]

                is_dummy, n, d = create_vertex_triangle_division_plane_closest_pt(
                    colliding_particle_pos,
                    colliding_particle_displacement,
                    t1,
                    delta_t1,
                    t2,
                    delta_t2,
                    t3,
                    delta_t3,
                )

                # planar truncation for 2 sides
                if not is_dummy[0]:
                    t = planar_truncation_t(
                        colliding_particle_pos, colliding_particle_displacement, n, d, parallel_eps, gamma
                    )
                    wp.atomic_min(truncation_t_out, particle_idx, t)
                if not is_dummy[1]:
                    t = planar_truncation_t(t1, delta_t1, n, d, parallel_eps, gamma)
                    wp.atomic_min(truncation_t_out, tri_a, t)
                if not is_dummy[2]:
                    t = planar_truncation_t(t2, delta_t2, n, d, parallel_eps, gamma)
                    wp.atomic_min(truncation_t_out, tri_b, t)
                if not is_dummy[3]:
                    t = planar_truncation_t(t3, delta_t3, n, d, parallel_eps, gamma)
                    wp.atomic_min(truncation_t_out, tri_c, t)

            collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE

    # Don't forget to do the final truncation based on the maximum displacement allowance!


@wp.kernel
def apply_truncation_ts(
    pos: wp.array(dtype=wp.vec3),
    displacement_in: wp.array(dtype=wp.vec3),
    truncation_ts: wp.array(dtype=float),
    max_displacement: float,
    displacement_out: wp.array(dtype=wp.vec3),
    pos_out: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    t = truncation_ts[i]
    particle_displacement = displacement_in[i] * t

    # Nuts-saving truncation: clamp displacement magnitude to max_displacement
    len_displacement = wp.length(particle_displacement)
    if len_displacement > max_displacement:
        particle_displacement = particle_displacement * max_displacement / len_displacement

    displacement_out[i] = particle_displacement
    if pos_out:
        pos_out[i] = pos[i] + particle_displacement


@wp.kernel
def accumulate_particle_body_contact_force_and_hessian(
    # inputs
    dt: float,
    current_color: int,
    pos_anchor: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    particle_colors: wp.array(dtype=int),
    # body-particle contact
    friction_epsilon: float,
    particle_radius: wp.array(dtype=float),
    body_particle_contact_particle: wp.array(dtype=int),
    body_particle_contact_count: wp.array(dtype=int),
    body_particle_contact_max: int,
    # per-contact soft AVBD parameters for body-particle contacts (shared with rigid side)
    body_particle_contact_penalty_k: wp.array(dtype=float),
    body_particle_contact_material_kd: wp.array(dtype=float),
    body_particle_contact_material_mu: wp.array(dtype=float),
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

    particle_body_contact_count = min(body_particle_contact_max, body_particle_contact_count[0])

    if t_id < particle_body_contact_count:
        particle_idx = body_particle_contact_particle[t_id]

        if particle_colors[particle_idx] == current_color:
            # Read per-contact AVBD penalty and material properties shared with the rigid side
            contact_ke = body_particle_contact_penalty_k[t_id]
            contact_kd = body_particle_contact_material_kd[t_id]
            contact_mu = body_particle_contact_material_mu[t_id]

            body_contact_force, body_contact_hessian = evaluate_body_particle_contact(
                particle_idx,
                pos[particle_idx],
                pos_anchor[particle_idx],
                t_id,
                contact_ke,
                contact_kd,
                contact_mu,
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
def solve_elasticity_tile(
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
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
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_poses: wp.array(dtype=wp.mat33),
    tet_materials: wp.array(dtype=float, ndim=2),
    particle_adjacency: ParticleForceElementAdjacencyInfo,
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
    # output
    particle_displacements: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    block_idx = tid // TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
    thread_idx = tid % TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
    particle_index = particle_ids_in_color[block_idx]

    if not particle_flags[particle_index] & ParticleFlags.ACTIVE or mass[particle_index] == 0:
        if thread_idx == 0:
            particle_displacements[particle_index] = wp.vec3(0.0)
        return

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # elastic force and hessian
    num_adj_faces = get_vertex_num_adjacent_faces(particle_adjacency, particle_index)

    f = wp.vec3(0.0)
    h = wp.mat33(0.0)

    batch_counter = wp.int32(0)

    if tri_indices:
        # loop through all the adjacent triangles using whole block
        while batch_counter + thread_idx < num_adj_faces:
            adj_tri_counter = thread_idx + batch_counter
            batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
            # elastic force and hessian
            tri_index, vertex_order = get_vertex_adjacent_face_id_order(
                particle_adjacency, particle_index, adj_tri_counter
            )

            # fmt: off
            if wp.static("connectivity" in VBD_DEBUG_PRINTING_OPTIONS):
                wp.printf(
                    "particle: %d | num_adj_faces: %d | ",
                    particle_index,
                    get_vertex_num_adjacent_faces(particle_adjacency, particle_index),
                )
                wp.printf("i_face: %d | face id: %d | v_order: %d | ", adj_tri_counter, tri_index, vertex_order)
                wp.printf(
                    "face: %d %d %d\n",
                    tri_indices[tri_index, 0],
                    tri_indices[tri_index, 1],
                    tri_indices[tri_index, 2],
                )
            # fmt: on

            if tri_materials[tri_index, 0] > 0.0 or tri_materials[tri_index, 1] > 0.0:
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

    if edge_indices:
        batch_counter = wp.int32(0)
        num_adj_edges = get_vertex_num_adjacent_edges(particle_adjacency, particle_index)
        while batch_counter + thread_idx < num_adj_edges:
            adj_edge_counter = batch_counter + thread_idx
            batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
            nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(
                particle_adjacency, particle_index, adj_edge_counter
            )
            if edge_bending_properties[nei_edge_index, 0] > 0.0:
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

    if tet_indices:
        # solve tet elasticity
        batch_counter = wp.int32(0)
        num_adj_tets = get_vertex_num_adjacent_tets(particle_adjacency, particle_index)
        while batch_counter + thread_idx < num_adj_tets:
            adj_tet_counter = batch_counter + thread_idx
            batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
            nei_tet_index, vertex_order_on_tet = get_vertex_adjacent_tet_id_order(
                particle_adjacency, particle_index, adj_tet_counter
            )
            if tet_materials[nei_tet_index, 0] > 0.0 or tet_materials[nei_tet_index, 1] > 0.0:
                f_tet, h_tet = evaluate_volumetric_neo_hookean_force_and_hessian(
                    nei_tet_index,
                    vertex_order_on_tet,
                    pos_prev,
                    pos,
                    tet_indices,
                    tet_poses[nei_tet_index],
                    tet_materials[nei_tet_index, 0],
                    tet_materials[nei_tet_index, 1],
                    tet_materials[nei_tet_index, 2],
                    dt,
                )

                f += f_tet
                h += h_tet

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
        if abs(wp.determinant(h_total)) > 1e-8:
            h_inv = wp.inverse(h_total)
            f_total = (
                f_total
                + mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
                + particle_forces[particle_index]
            )
            particle_displacements[particle_index] = particle_displacements[particle_index] + h_inv * f_total


@wp.kernel
def solve_elasticity(
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
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
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_poses: wp.array(dtype=wp.mat33),
    tet_materials: wp.array(dtype=float, ndim=2),
    particle_adjacency: ParticleForceElementAdjacencyInfo,
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
    # output
    particle_displacements: wp.array(dtype=wp.vec3),
):
    t_id = wp.tid()

    particle_index = particle_ids_in_color[t_id]

    if not particle_flags[particle_index] & ParticleFlags.ACTIVE or mass[particle_index] == 0:
        particle_displacements[particle_index] = wp.vec3(0.0)
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

    if tri_indices:
        # elastic force and hessian
        for i_adj_tri in range(get_vertex_num_adjacent_faces(particle_adjacency, particle_index)):
            tri_index, vertex_order = get_vertex_adjacent_face_id_order(particle_adjacency, particle_index, i_adj_tri)

            # fmt: off
            if wp.static("connectivity" in VBD_DEBUG_PRINTING_OPTIONS):
                wp.printf(
                    "particle: %d | num_adj_faces: %d | ",
                    particle_index,
                    get_vertex_num_adjacent_faces(particle_adjacency, particle_index),
                )
                wp.printf("i_face: %d | face id: %d | v_order: %d | ", i_adj_tri, tri_index, vertex_order)
                wp.printf(
                    "face: %d %d %d\n",
                    tri_indices[tri_index, 0],
                    tri_indices[tri_index, 1],
                    tri_indices[tri_index, 2],
                )
            # fmt: on

            if tri_materials[tri_index, 0] > 0.0 or tri_materials[tri_index, 1] > 0.0:
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

    if edge_indices:
        for i_adj_edge in range(get_vertex_num_adjacent_edges(particle_adjacency, particle_index)):
            nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(particle_adjacency, particle_index, i_adj_edge)
            # vertex is on the edge; otherwise it only effects the bending energy n
            if edge_bending_properties[nei_edge_index, 0] > 0.0:
                f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
                    nei_edge_index, vertex_order_on_edge, pos, pos_prev, edge_indices, edge_rest_angles, edge_rest_length,
                    edge_bending_properties[nei_edge_index, 0], edge_bending_properties[nei_edge_index, 1], dt
                )

                f = f + f_edge
                h = h + h_edge

    if tet_indices:
        # solve tet elasticity
        num_adj_tets = get_vertex_num_adjacent_tets(particle_adjacency, particle_index)
        for adj_tet_counter in range(num_adj_tets):
            nei_tet_index, vertex_order_on_tet = get_vertex_adjacent_tet_id_order(
                particle_adjacency, particle_index, adj_tet_counter
            )
            if tet_materials[nei_tet_index, 0] > 0.0 or tet_materials[nei_tet_index, 1] > 0.0:
                f_tet, h_tet = evaluate_volumetric_neo_hookean_force_and_hessian(
                    nei_tet_index,
                    vertex_order_on_tet,
                    pos_prev,
                    pos,
                    tet_indices,
                    tet_poses[nei_tet_index],
                    tet_materials[nei_tet_index, 0],
                    tet_materials[nei_tet_index, 1],
                    tet_materials[nei_tet_index, 2],
                    dt,
                )

                f += f_tet
                h += h_tet

    # fmt: off
    if wp.static("overall_force_hessian" in VBD_DEBUG_PRINTING_OPTIONS):
        wp.printf(
            "vertex: %d final\noverall force:\n %f %f %f, \noverall hessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
            particle_index,
            f[0], f[1], f[2], h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2],
        )

    # fmt: on
    h = h + particle_hessians[particle_index]
    f = f + particle_forces[particle_index]

    if abs(wp.determinant(h)) > 1e-8:
        h_inv = wp.inverse(h)
        particle_displacements[particle_index] = particle_displacements[particle_index] + h_inv * f


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
