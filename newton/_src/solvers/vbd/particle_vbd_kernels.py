# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Particle/soft-body VBD helper routines.

This module is intended to host the particle/soft-body specific parts of the
VBD solver (cloth, springs, triangles, tets, particle contacts, etc.).

The high-level :class:`SolverVBD` interface should remain in
``solver_vbd.py`` and call into functions defined here.
"""

from __future__ import annotations

import warp as wp

from newton._src.math import orthonormal_basis
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    _eval_body_particle_contact,
    _eval_soft_ef_contact,
    evaluate_body_particle_contact,
)

from ...geometry import ParticleFlags
from ...geometry.kernels import triangle_closest_point
from ...utils.mesh import (
    MeshAdjacencyData,
    get_vertex_adjacent_edge_id_order,
    get_vertex_adjacent_face_id_order,
    get_vertex_adjacent_tet_id_order,
    get_vertex_num_adjacent_edges,
    get_vertex_num_adjacent_faces,
    get_vertex_num_adjacent_tets,
)
from .tri_mesh_collision import TriMeshCollisionInfo

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
def evaluate_volumetric_neo_hookean_force_and_hessian(
    tet_id: int,
    v_order: int,
    pos_prev: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    tet_indices: wp.array2d[wp.int32],
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
    # Convert Lamé parameters to stable Neo-Hookean parameters per Smith et al.
    # 2018, §3.4 (eq. 13): the symbols (mu, lambda) appearing in the NH energy
    # are not directly the Lamé parameters; matching the small-strain limit
    # gives mu_NH = mu_Lamé, lambda_NH = lambda_Lamé + mu_Lamé.
    mu_nh = mu
    lmbd_nh = lmbd + mu
    # Guard against division by zero in lambda_NH
    lmbd_safe = wp.sign(lmbd_nh) * wp.max(wp.abs(lmbd_nh), 1e-6)
    alpha = 1.0 + mu_nh / lmbd_safe
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
    s = lmbd_nh * (J - alpha)
    P_vec = rest_volume * (mu_nh * f + s * cof_vec)

    # ============ Hessian ============
    # The full elastic Hessian also has an s * d^2 J / dF^2 term, but its
    # contribution to VBD's per-vertex 3x3 block is identically zero:
    # the Levi-Civita tensor in d^2 J / dF^2 contracts against (m^a x m^a),
    # which vanishes. Drop it; the remaining two terms are SPD by inspection.
    H = mu_nh * wp.identity(n=9, dtype=float) + lmbd_nh * wp.outer(cof_vec, cof_vec)
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

        Ds_prev = wp.matrix_from_cols(v1_prev - v0_prev, v2_prev - v0_prev, v3_prev - v0_prev)
        F_prev = Ds_prev * Dm_inv

        f0 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
        f1 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
        f2 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])
        f0_prev = wp.vec3(F_prev[0, 0], F_prev[1, 0], F_prev[2, 0])
        f1_prev = wp.vec3(F_prev[0, 1], F_prev[1, 1], F_prev[2, 1])
        f2_prev = wp.vec3(F_prev[0, 2], F_prev[1, 2], F_prev[2, 2])

        c00_rate = (wp.dot(f0, f0) - wp.dot(f0_prev, f0_prev)) * inv_dt
        c01_rate = (wp.dot(f0, f1) - wp.dot(f0_prev, f1_prev)) * inv_dt
        c02_rate = (wp.dot(f0, f2) - wp.dot(f0_prev, f2_prev)) * inv_dt
        c11_rate = (wp.dot(f1, f1) - wp.dot(f1_prev, f1_prev)) * inv_dt
        c12_rate = (wp.dot(f1, f2) - wp.dot(f1_prev, f2_prev)) * inv_dt
        c22_rate = (wp.dot(f2, f2) - wp.dot(f2_prev, f2_prev)) * inv_dt

        dc00_dx = 2.0 * m[0] * f0
        dc01_dx = m[0] * f1 + m[1] * f0
        dc02_dx = m[0] * f2 + m[2] * f0
        dc11_dx = 2.0 * m[1] * f1
        dc12_dx = m[1] * f2 + m[2] * f1
        dc22_dx = 2.0 * m[2] * f2

        f_damp = (
            c00_rate * dc00_dx
            + 2.0 * c01_rate * dc01_dx
            + 2.0 * c02_rate * dc02_dx
            + c11_rate * dc11_dx
            + 2.0 * c12_rate * dc12_dx
            + c22_rate * dc22_dx
        )
        force = force - rest_volume * damping * f_damp
        hessian = hessian + rest_volume * damping * inv_dt * (
            wp.outer(dc00_dx, dc00_dx)
            + 2.0 * wp.outer(dc01_dx, dc01_dx)
            + 2.0 * wp.outer(dc02_dx, dc02_dx)
            + wp.outer(dc11_dx, dc11_dx)
            + 2.0 * wp.outer(dc12_dx, dc12_dx)
            + wp.outer(dc22_dx, dc22_dx)
        )

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
def _test_compute_force_element_adjacency(
    adjacency: MeshAdjacencyData,
    edge_indices: wp.array2d[wp.int32],
    face_indices: wp.array2d[wp.int32],
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
def evaluate_neo_hookean_membrane_force_hessian(
    face: int,
    v_order: int,
    pos: wp.array[wp.vec3],
    pos_anchor: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    tri_pose: wp.mat22,
    area: float,
    mu: float,
    lmbd: float,
    damping: float,
    dt: float,
):
    # Stable Neo-Hookean energy for 2D membranes (Smith et al. 2018 adapted to shells):
    #   psi = (mu/2)(I_c - 2) + (lambda/2)(J_s - alpha)^2
    # where I_c = tr(F^T F), J_s = sqrt(det(F^T F)) = area ratio,
    # alpha = 1 + mu/lambda (ensures zero stress at the rest configuration).

    v0 = tri_indices[face, 0]
    v1 = tri_indices[face, 1]
    v2 = tri_indices[face, 2]

    x0 = pos[v0]
    x01 = pos[v1] - x0
    x02 = pos[v2] - x0

    DmInv00 = tri_pose[0, 0]
    DmInv01 = tri_pose[0, 1]
    DmInv10 = tri_pose[1, 0]
    DmInv11 = tri_pose[1, 1]

    # Deformation gradient F = [f0, f1] (3x2 as two column vectors)
    f0 = x01 * DmInv00 + x02 * DmInv10
    f1 = x01 * DmInv01 + x02 * DmInv11

    # Cauchy-Green invariants
    f0_dot_f0 = wp.dot(f0, f0)
    f1_dot_f1 = wp.dot(f1, f1)
    f0_dot_f1 = wp.dot(f0, f1)

    # J_s = area ratio = sqrt(det(F^T F))
    J_s_sq = f0_dot_f0 * f1_dot_f1 - f0_dot_f1 * f0_dot_f1
    J_s_sq = wp.max(J_s_sq, 1.0e-20)
    J_s = wp.sqrt(J_s_sq)
    inv_J_s = 1.0 / J_s

    # Convert Lamé parameters to stable Neo-Hookean parameters per Smith et al.
    # 2018, §3.4 (eq. 13): mu_NH = mu_Lamé, lambda_NH = lambda_Lamé + mu_Lamé.
    mu_nh = mu
    lmbd_nh = lmbd + mu
    lmbd_safe = wp.sign(lmbd_nh) * wp.max(wp.abs(lmbd_nh), 1.0e-6)
    alpha = 1.0 + mu_nh / lmbd_safe

    # 2D "cofactor" vectors: g_i = dJ_s/df_i
    g0 = inv_J_s * (f1_dot_f1 * f0 - f0_dot_f1 * f1)
    g1 = inv_J_s * (f0_dot_f0 * f1 - f0_dot_f1 * f0)

    # First Piola-Kirchhoff stress: P = mu*F + lambda*(J_s - alpha)*[g0, g1]
    s = lmbd_nh * (J_s - alpha)
    P_col0 = mu_nh * f0 + s * g0
    P_col1 = mu_nh * f1 + s * g1

    # Vertex selection masks
    mask0 = float(v_order == 0)
    mask1 = float(v_order == 1)
    mask2 = float(v_order == 2)

    df0_dx = DmInv00 * (mask1 - mask0) + DmInv10 * (mask2 - mask0)
    df1_dx = DmInv01 * (mask1 - mask0) + DmInv11 * (mask2 - mask0)

    # Force: -(dψ/dF):(dF/dx)
    dpsi_dx = P_col0 * df0_dx + P_col1 * df1_dx
    force = -dpsi_dx

    # --- Hessian (per-vertex 3x3, SPD-projected) ---
    # max(0, s) is the tight PSD clamp for the membrane per-vertex block.
    # The volumetric-tet cancellation does not apply here (F is 3x2, J_s
    # non-polynomial), so the cofactor-derivative term must be kept.
    s_clamp = wp.max(0.0, s)
    r = s_clamp * inv_J_s
    c1 = lmbd_nh - r

    df0_dx_sq = df0_dx * df0_dx
    df1_dx_sq = df1_dx * df1_dx

    # Projected gradient of J_s w.r.t. vertex position
    dJ_dx = g0 * df0_dx + g1 * df1_dx
    # Cross-column vector for cofactor-derivative contraction
    w = f1 * df0_dx - f0 * df1_dx

    I_coeff = mu_nh * (df0_dx_sq + df1_dx_sq) + r * (
        df0_dx_sq * f1_dot_f1 + df1_dx_sq * f0_dot_f0 - 2.0 * df0_dx * df1_dx * f0_dot_f1
    )

    I33 = wp.identity(n=3, dtype=float)
    hessian = I_coeff * I33 + c1 * wp.outer(dJ_dx, dJ_dx) - r * wp.outer(w, w)

    # Objective damping based on the metric C = F^T F, so rigid rotations do not damp.
    if damping > 0.0:
        inv_dt = 1.0 / dt

        x0_prev = pos_anchor[v0]
        x01_prev = pos_anchor[v1] - x0_prev
        x02_prev = pos_anchor[v2] - x0_prev

        f0_prev = x01_prev * DmInv00 + x02_prev * DmInv10
        f1_prev = x01_prev * DmInv01 + x02_prev * DmInv11

        c00_rate = (f0_dot_f0 - wp.dot(f0_prev, f0_prev)) * inv_dt
        c01_rate = (f0_dot_f1 - wp.dot(f0_prev, f1_prev)) * inv_dt
        c11_rate = (f1_dot_f1 - wp.dot(f1_prev, f1_prev)) * inv_dt

        dc00_dx = 2.0 * df0_dx * f0
        dc01_dx = df0_dx * f1 + df1_dx * f0
        dc11_dx = 2.0 * df1_dx * f1

        f_damp = c00_rate * dc00_dx + 2.0 * c01_rate * dc01_dx + c11_rate * dc11_dx
        force += -damping * f_damp

        hessian += (
            damping
            * inv_dt
            * (wp.outer(dc00_dx, dc00_dx) + 2.0 * wp.outer(dc01_dx, dc01_dx) + wp.outer(dc11_dx, dc11_dx))
        )

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
    pos: wp.array[wp.vec3],
    pos_anchor: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    edge_rest_angle: wp.array[float],
    edge_rest_length: wp.array[float],
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

        x02_prev = x_prev2 - x_prev0
        x03_prev = x_prev3 - x_prev0
        x13_prev = x_prev3 - x_prev1
        x12_prev = x_prev2 - x_prev1
        e_prev = x_prev3 - x_prev2

        n1_prev_raw = wp.cross(x02_prev, x03_prev)
        n2_prev_raw = wp.cross(x13_prev, x12_prev)
        n1_prev_norm = wp.length(n1_prev_raw)
        n2_prev_norm = wp.length(n2_prev_raw)
        e_prev_norm = wp.length(e_prev)
        if n1_prev_norm < eps or n2_prev_norm < eps or e_prev_norm < eps:
            return bending_force, bending_hessian

        n1_prev = n1_prev_raw / n1_prev_norm
        n2_prev = n2_prev_raw / n2_prev_norm
        e_hat_prev = e_prev / e_prev_norm

        sin_theta_prev = wp.dot(wp.cross(n1_prev, n2_prev), e_hat_prev)
        cos_theta_prev = wp.dot(n1_prev, n2_prev)
        theta_prev = wp.atan2(sin_theta_prev, cos_theta_prev)

        dtheta = theta - theta_prev
        if dtheta > 3.141592653589793:
            dtheta = dtheta - 6.283185307179586
        elif dtheta < -3.141592653589793:
            dtheta = dtheta + 6.283185307179586

        dtheta_dt = dtheta * inv_dt

        rest_len = edge_rest_length[bending_index]
        damping_force = -damping * rest_len * dtheta_dt * dtheta_dx
        damping_hessian = damping * rest_len * inv_dt * wp.outer(dtheta_dx, dtheta_dx)

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
    d_min = 1.0e-5
    if tau > dis > d_min:
        # Log-barrier region: E ∝ -ln(dis)
        k2 = tau * tau * k
        dEdD = -k2 / dis
        d2E_dDdD = k2 / (dis * dis)
    elif dis <= d_min:
        # Quadratic extension below d_min (Taylor of the log-barrier at d_min)
        # preserving C2 continuity: constant Hessian, linear gradient
        k2 = tau * tau * k
        d_min_sq = d_min * d_min
        dEdD = k2 * (dis - 2.0 * d_min) / d_min_sq
        d2E_dDdD = k2 / d_min_sq
    else:
        dEdD = -k * penetration_depth
        d2E_dDdD = k

    return dEdD, d2E_dDdD


@wp.func
def damp_collision(
    gap_rate: float,
    b_i: float,
    collision_normal: wp.vec3,
    collision_damping: float,
    dt: float,
):
    """Damp collision with the contact gap rate, not absolute vertex motion."""
    if gap_rate < -1.0e-6:
        n_outer = wp.outer(collision_normal, collision_normal)
        damping_force = -collision_damping * gap_rate * b_i * collision_normal
        damping_hessian = (collision_damping / dt) * b_i * b_i * n_outer
        return damping_force, damping_hessian
    else:
        return wp.vec3(0.0), wp.mat33(0.0)


@wp.func
def evaluate_edge_edge_contact(
    v: int,
    v_order: int,
    e1: int,
    e2: int,
    pos: wp.array[wp.vec3],
    pos_anchor: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
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

        gap_rate = wp.dot(collision_normal, dx) / dt

        damping_force, damping_hessian = damp_collision(gap_rate, v_bary, collision_normal, collision_damping, dt)
        collision_force = collision_force + damping_force
        collision_hessian = collision_hessian + damping_hessian

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
    pos: wp.array[wp.vec3],
    pos_anchor: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
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

        gap_rate = wp.dot(collision_normal, dx) / dt

        collision_force_0 = collision_force * bs[0]
        collision_force_1 = collision_force * bs[1]

        collision_hessian_0 = collision_hessian * bs[0] * bs[0]
        collision_hessian_1 = collision_hessian * bs[1] * bs[1]

        damping_force, damping_hessian = damp_collision(gap_rate, bs[0], collision_normal, collision_damping, dt)

        collision_force_0 += damping_force + bs[0] * friction_force
        collision_hessian_0 += damping_hessian + bs[0] * bs[0] * friction_hessian

        damping_force, damping_hessian = damp_collision(gap_rate, bs[1], collision_normal, collision_damping, dt)
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
    pos: wp.array[wp.vec3],
    pos_anchor: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
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

        gap_rate = wp.dot(collision_normal, dx) / dt

        damping_force, damping_hessian = damp_collision(gap_rate, v_bary, collision_normal, collision_damping, dt)
        collision_force = collision_force + damping_force
        collision_hessian = collision_hessian + damping_hessian

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
    pos: wp.array[wp.vec3],
    pos_anchor: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
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

        gap_rate = wp.dot(collision_normal, dx) / dt

        collision_force_0 = collision_force * bs[0]
        collision_force_1 = collision_force * bs[1]
        collision_force_2 = collision_force * bs[2]
        collision_force_3 = collision_force * bs[3]

        collision_hessian_0 = collision_hessian * bs[0] * bs[0]
        collision_hessian_1 = collision_hessian * bs[1] * bs[1]
        collision_hessian_2 = collision_hessian * bs[2] * bs[2]
        collision_hessian_3 = collision_hessian * bs[3] * bs[3]

        damping_force, damping_hessian = damp_collision(gap_rate, bs[0], collision_normal, collision_damping, dt)

        collision_force_0 += damping_force + bs[0] * friction_force
        collision_hessian_0 += damping_hessian + bs[0] * bs[0] * friction_hessian

        damping_force, damping_hessian = damp_collision(gap_rate, bs[1], collision_normal, collision_damping, dt)
        collision_force_1 += damping_force + bs[1] * friction_force
        collision_hessian_1 += damping_hessian + bs[1] * bs[1] * friction_hessian

        damping_force, damping_hessian = damp_collision(gap_rate, bs[2], collision_normal, collision_damping, dt)
        collision_force_2 += damping_force + bs[2] * friction_force
        collision_hessian_2 += damping_hessian + bs[2] * bs[2] * friction_hessian

        damping_force, damping_hessian = damp_collision(gap_rate, bs[3], collision_normal, collision_damping, dt)
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
    gravity: wp.array[wp.vec3],
    pos_prev: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    external_force: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    inertia_out: wp.array[wp.vec3],
    displacements_out: wp.array[wp.vec3],
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
    adjacency: MeshAdjacencyData,
    collision_info: TriMeshCollisionInfo,
    # outputs
    particle_conservative_bounds: wp.array[float],
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
    pos: wp.array[wp.vec3],
    pos_prev_collision_detection: wp.array[wp.vec3],
    particle_conservative_bounds: wp.array[float],
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
    pos_prev_collision_detection: wp.array[wp.vec3],
    particle_conservative_bounds: wp.array[float],
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
def update_velocity(dt: float, pos_prev: wp.array[wp.vec3], pos: wp.array[wp.vec3], vel: wp.array[wp.vec3]):
    particle = wp.tid()
    vel[particle] = (pos[particle] - pos_prev[particle]) / dt


@wp.kernel
def convert_body_particle_contact_data_kernel(
    # inputs
    body_particle_contact_buffer_pre_alloc: int,
    soft_contact_particle: wp.array[int],
    contact_count: wp.array[int],
    contact_max: int,
    # outputs
    body_particle_contact_buffer: wp.array[int],
    body_particle_contact_count: wp.array[int],
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
    pos_prev: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    particle_colors: wp.array[int],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    # self contact
    collision_info_array: wp.array[TriMeshCollisionInfo],
    collision_radius: float,
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    edge_edge_parallel_epsilon: float,
    # outputs: particle force and hessian
    particle_forces: wp.array[wp.vec3],
    particle_hessians: wp.array[wp.mat33],
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


@wp.func
def evaluate_spring_force_and_hessian(
    particle_idx: int,
    spring_idx: int,
    dt: float,
    pos: wp.array[wp.vec3],
    pos_anchor: wp.array[wp.vec3],
    spring_indices: wp.array[int],
    spring_rest_length: wp.array[float],
    spring_stiffness: wp.array[float],
    spring_damping: wp.array[float],
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
    structural = wp.identity(3, float) - (l0 / spring_length) * (
        wp.identity(3, float) - wp.outer(diff, diff) / (spring_length * spring_length)
    )
    spring_hessian = spring_stiffness[spring_idx] * structural

    spring_direction = diff / spring_length
    diff_anchor = pos_anchor[v0] - pos_anchor[v1]
    spring_length_anchor = wp.max(wp.length(diff_anchor), 1e-8)
    length_change = spring_length - spring_length_anchor
    h_d = wp.outer(spring_direction, spring_direction) * (spring_damping[spring_idx] / dt)
    f_d = -force_sign * spring_direction * (spring_damping[spring_idx] / dt * length_change)

    spring_force = spring_force + f_d
    spring_hessian = spring_hessian + h_d

    return spring_force, spring_hessian


@wp.func
def evaluate_spring_force_and_hessian_both_vertices(
    spring_idx: int,
    dt: float,
    pos: wp.array[wp.vec3],
    pos_anchor: wp.array[wp.vec3],
    spring_indices: wp.array[int],
    spring_rest_length: wp.array[float],
    spring_stiffness: wp.array[float],
    spring_damping: wp.array[float],
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

    structural = wp.identity(3, float) - (l0 / spring_length) * (
        wp.identity(3, float) - wp.outer(diff, diff) / (spring_length * spring_length)
    )
    spring_hessian = spring_stiffness[spring_idx] * structural

    spring_direction = diff / spring_length
    diff_anchor = pos_anchor[v0] - pos_anchor[v1]
    spring_length_anchor = wp.max(wp.length(diff_anchor), 1e-8)
    length_change = spring_length - spring_length_anchor
    h_d = wp.outer(spring_direction, spring_direction) * (spring_damping[spring_idx] / dt)
    f_d_v0 = -spring_direction * (spring_damping[spring_idx] / dt * length_change)
    f_d_v1 = spring_direction * (spring_damping[spring_idx] / dt * length_change)

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
    pos_anchor: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    particle_colors: wp.array[int],
    num_springs: int,
    # spring constraints
    spring_indices: wp.array[int],
    spring_rest_length: wp.array[float],
    spring_stiffness: wp.array[float],
    spring_damping: wp.array[float],
    # outputs: particle force and hessian
    particle_forces: wp.array[wp.vec3],
    particle_hessians: wp.array[wp.mat33],
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
    pos_anchor: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    particle_colors: wp.array[int],
    # body-particle contact
    friction_epsilon: float,
    particle_radius: wp.array[float],
    body_particle_contact_particle: wp.array[int],
    body_particle_contact_count: wp.array[int],
    body_particle_contact_max: int,
    # per-contact soft AVBD parameters for body-particle contacts (shared with rigid side)
    body_particle_contact_penalty_k: wp.array[float],
    body_particle_contact_material_ke: wp.array[float],
    body_particle_contact_material_kd: wp.array[float],
    body_particle_contact_material_mu: wp.array[float],
    shape_material_mu: wp.array[float],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    # outputs: particle force and hessian
    particle_forces: wp.array[wp.vec3],
    particle_hessians: wp.array[wp.mat33],
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

            body_contact_force, body_contact_hessian = _eval_body_particle_contact(
                particle_idx,
                pos[particle_idx],
                pos_anchor[particle_idx],
                t_id,
                contact_ke,
                contact_kd,
                contact_mu,
                friction_epsilon,
                particle_radius,
                shape_body,
                body_q,
                body_q_prev,
                body_qd,
                body_com,
                contact_shape,
                contact_body_pos,
                contact_body_vel,
                contact_normal,
                shape_margin,
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
    pos: wp.array[wp.vec3],
    displacement_in: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    collision_info_array: wp.array[TriMeshCollisionInfo],
    parallel_eps: float,
    gamma: float,
    truncation_t_out: wp.array[float],
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
    pos: wp.array[wp.vec3],
    displacement_in: wp.array[wp.vec3],
    truncation_ts: wp.array[float],
    max_displacement: float,
    displacement_out: wp.array[wp.vec3],
    pos_out: wp.array[wp.vec3],
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
    pos_anchor: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    particle_colors: wp.array[int],
    # body-particle contact
    friction_epsilon: float,
    particle_radius: wp.array[float],
    body_particle_contact_indices: wp.array[wp.vec3i],
    body_particle_contact_count: wp.array[int],
    body_particle_contact_max: int,
    # per-contact soft AVBD parameters for body-particle contacts (shared with rigid side)
    body_particle_contact_penalty_k: wp.array[float],
    body_particle_contact_material_ke: wp.array[float],
    body_particle_contact_material_kd: wp.array[float],
    body_particle_contact_material_mu: wp.array[float],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    # Barycentric weights on each record's soft particles; (1, 0, 0) for a particle contact.
    contact_barycentric: wp.array[wp.vec3],
    # outputs: particle force and hessian
    particle_forces: wp.array[wp.vec3],
    particle_hessians: wp.array[wp.mat33],
):
    t_id = wp.tid()

    # One unified soft-contact stream. body_particle_contact_count[0] is the total soft-contact count;
    # each record self-describes via its -1-padded corner ids: (p, -1, -1) is a particle contact,
    # (v0, v1, -1) an edge, (v0, v1, v2) a face. A contact energy E(x) at x = sum_i bary[i]*pos[c_i]
    # contributes bary[i]*force to corner i and bary[i]^2*hessian to its block. VBD solves one color
    # per launch, so only scatter to this record's corners of the active color.
    count = min(body_particle_contact_max, body_particle_contact_count[0])
    if t_id >= count:
        return

    corners = body_particle_contact_indices[t_id]
    # Per-contact AVBD penalty + material properties shared with the rigid side.
    contact_ke = body_particle_contact_penalty_k[t_id]
    contact_kd = body_particle_contact_material_kd[t_id]
    contact_mu = body_particle_contact_material_mu[t_id]

    if corners[1] < 0:
        # Particle contact (p, -1, -1): single-vertex path, unchanged from the pre-unification code.
        particle_idx = corners[0]
        if particle_colors[particle_idx] == current_color:
            body_contact_force, body_contact_hessian = _eval_body_particle_contact(
                particle_idx,
                pos[particle_idx],
                pos_anchor[particle_idx],
                t_id,
                contact_ke,
                contact_kd,
                contact_mu,
                friction_epsilon,
                particle_radius,
                shape_body,
                body_q,
                body_q_prev,
                body_qd,
                body_com,
                contact_shape,
                contact_body_pos,
                contact_body_vel,
                contact_normal,
                shape_margin,
                dt,
            )
            wp.atomic_add(particle_forces, particle_idx, body_contact_force)
            wp.atomic_add(particle_hessians, particle_idx, body_contact_hessian)
    else:
        # Edge/face contact: barycentric point over the record's 2-3 soft particles.
        bary = contact_barycentric[t_id]
        ef_force, ef_hessian, _cp_world = _eval_soft_ef_contact(
            t_id,
            corners,
            bary,
            pos,
            pos_anchor,
            particle_radius,
            contact_ke,
            contact_kd,
            contact_mu,
            friction_epsilon,
            shape_body,
            body_q,
            body_q_prev,
            body_qd,
            body_com,
            contact_shape,
            contact_body_pos,
            contact_body_vel,
            contact_normal,
            shape_margin,
            dt,
        )
        for i in range(3):
            ci = corners[i]
            if ci >= 0:
                w = bary[i]
                if particle_colors[ci] == current_color:
                    wp.atomic_add(particle_forces, ci, w * ef_force)
                    wp.atomic_add(particle_hessians, ci, (w * w) * ef_hessian)


@wp.kernel
def solve_elasticity_tile(
    dt: float,
    particle_ids_in_color: wp.array[wp.int32],
    pos_prev: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    mass: wp.array[float],
    inertia: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    tri_poses: wp.array[wp.mat22],
    tri_materials: wp.array2d[float],
    tri_areas: wp.array[float],
    edge_indices: wp.array2d[wp.int32],
    edge_rest_angles: wp.array[float],
    edge_rest_length: wp.array[float],
    edge_bending_properties: wp.array2d[float],
    tet_indices: wp.array2d[wp.int32],
    tet_poses: wp.array[wp.mat33],
    tet_materials: wp.array2d[float],
    particle_adjacency: MeshAdjacencyData,
    particle_forces: wp.array[wp.vec3],
    particle_hessians: wp.array[wp.mat33],
    # output
    particle_displacements: wp.array[wp.vec3],
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
    f = wp.vec3(0.0)
    h = wp.mat33(0.0)

    batch_counter = wp.int32(0)

    if tri_indices.shape[0] > 0:
        num_adj_faces = get_vertex_num_adjacent_faces(particle_adjacency, particle_index)
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
                f_tri, h_tri = evaluate_neo_hookean_membrane_force_hessian(
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

    if edge_indices.shape[0] > 0:
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

    if tet_indices.shape[0] > 0:
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
    particle_ids_in_color: wp.array[wp.int32],
    pos_prev: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    mass: wp.array[float],
    inertia: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    tri_poses: wp.array[wp.mat22],
    tri_materials: wp.array2d[float],
    tri_areas: wp.array[float],
    edge_indices: wp.array2d[wp.int32],
    edge_rest_angles: wp.array[float],
    edge_rest_length: wp.array[float],
    edge_bending_properties: wp.array2d[float],
    tet_indices: wp.array2d[wp.int32],
    tet_poses: wp.array[wp.mat33],
    tet_materials: wp.array2d[float],
    particle_adjacency: MeshAdjacencyData,
    particle_forces: wp.array[wp.vec3],
    particle_hessians: wp.array[wp.mat33],
    # output
    particle_displacements: wp.array[wp.vec3],
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

    if tri_indices.shape[0] > 0:
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
                f_tri, h_tri = evaluate_neo_hookean_membrane_force_hessian(
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

    if edge_indices.shape[0] > 0:
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

    if tet_indices.shape[0] > 0:
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
    pos_prev: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    particle_colors: wp.array[int],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    # self contact
    collision_info_array: wp.array[TriMeshCollisionInfo],
    collision_radius: float,
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    edge_edge_parallel_epsilon: float,
    # body-particle contact
    particle_radius: wp.array[float],
    soft_contact_particle: wp.array[int],
    contact_count: wp.array[int],
    contact_max: int,
    shape_material_mu: wp.array[float],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    # outputs: particle force and hessian
    particle_forces: wp.array[wp.vec3],
    particle_hessians: wp.array[wp.mat33],
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
                shape_margin,
                dt,
            )
            wp.atomic_add(particle_forces, particle_idx, body_contact_force)
            wp.atomic_add(particle_hessians, particle_idx, body_contact_hessian)
