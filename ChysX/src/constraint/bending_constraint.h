// SPDX-License-Identifier: Apache-2.0
//
// chysx::constraint::BendingConstraint
//
// Four-particle ("N=4") dihedral bending element following Baraff &
// Witkin's 1998 cloth model and the Bridson-style discrete bending
// energy used by cuda-cloth's `KernelComputeDihedralForcesAndHessianFast`.
//
// Each instance is a *dihedral*: a pair of triangles sharing an edge,
// described by four particles
//
//     v_0, v_1   : the two endpoints of the shared edge
//     v_2        : the third vertex of the "left" triangle
//                  (the face that contains the directed edge v_0 -> v_1)
//     v_3        : the third vertex of the "right" triangle
//                  (the face that contains v_1 -> v_0)
//
// Geometry
// --------
//     e10 = v1 - v0         (shared edge)
//     e20 = v2 - v0
//     e30 = v3 - v0
//     n1  = normalize(e20 x e10)             (face-1 inward normal)
//     n2  = normalize(e10 x e30)             (face-2 inward normal)
//     theta = atan2( (n1 x n2) . e10/|e10|,  n1 . n2 )
//
// Energy
// ------
//     E = (k_bending / 2) * (theta - theta_rest)^2
//
// Gradient (Bridson)
// ------------------
// Define the four dihedral shape vectors a_i = d theta / d x_i:
//
//     omega_1 = (e10 . e20) / |e10|^2
//     omega_2 = (e10 . e30) / |e10|^2
//     h_1     = |e20 - omega_1 e10|        (height of v_2 over edge)
//     h_2     = |e30 - omega_2 e10|        (height of v_3 over edge)
//
//     t1 = ( omega_1 - 1, -omega_1, 1, 0 ) / h_1
//     t2 = ( omega_2 - 1, -omega_2, 0, 1 ) / h_2
//
//     a_i = t1[i] * n1 + t2[i] * n2          (i = 0, 1, 2, 3)
//
// Then  grad E w.r.t. x_i = k_bending * (theta - theta_rest) * a_i.
//
// Hessian (Gauss-Newton / PSD-projected)
// --------------------------------------
//     H_{i,j} = k_bending * (a_i ⊗ a_j)
//
// This drops the second-derivative term (theta - theta_rest) * d^2 theta
// / dx_i dx_j; the remaining outer-product part is positive semi-
// definite by construction, which is the same Gauss-Newton trick BW98
// uses for stretch + shear and the only branch cuda-cloth keeps in its
// fast kernel.
//
// Storage
// -------
// `ConstraintN<4>::indices_` holds the per-dihedral (v0, v1, v2, v3)
// tuple as a `Vec4i`.  `rest_angles_` is a parallel float buffer (one
// rest theta per dihedral).  All instances share a single stiffness `k`.

#pragma once

#include <cstdint>

#include "../math/matrix.cuh"
#include "../math/vec.cuh"
#include "../memory/cuda_array.h"
#include "../memory/device_span.h"
#include "constraint_n.h"

namespace chysx {
namespace constraint {

class BendingConstraint : public ConstraintN<4> {
public:
    BendingConstraint() = default;
    ~BendingConstraint() override = default;

    // Upload `n` dihedrals.  `host_dihedrals[c]` is the (v0, v1, v2, v3)
    // tuple of dihedral c.  Replaces any previously installed instances.
    // Rest angles are computed from the *current* device-side positions
    // (i.e. the configuration is treated as the rest pose).
    void set_dihedrals_from_positions(
        const math::Vec4i* host_dihedrals,
        int n,
        DeviceSpan<math::Vec3f> positions,
        std::uintptr_t cuda_stream = 0);

    // Stiffness shared by every dihedral [N·m / rad^2-ish].  cuda-cloth
    // uses this as a single global `k_bending` constant; we follow.
    void set_stiffness(float k) noexcept { stiffness_ = k; }
    float stiffness() const noexcept { return stiffness_; }

    // ---- buffer access ----------------------------------------------

    CudaArray<float>& rest_angles() noexcept { return rest_angles_; }
    const CudaArray<float>& rest_angles() const noexcept { return rest_angles_; }

    // ---- Constraint overrides ---------------------------------------

    float compute_energy(
        DeviceSpan<math::Vec3f> positions,
        std::uintptr_t cuda_stream = 0) const override;

    void accumulate_gradient(
        DeviceSpan<math::Vec3f> positions,
        DeviceSpan<math::Vec3f> out_grad,
        std::uintptr_t cuda_stream = 0) const override;

    // Each dihedral contributes 16 = 4x4 local 3x3 blocks.  Four of
    // them are diagonal (i == j) and land on `A.diag`; the other
    // twelve are off-diagonal and land on `A.values` at the slots
    // computed by `bind_hessian_layout`.
    void accumulate_hessian(
        DeviceSpan<math::Vec3f> positions,
        sparse::BlockCSR3& A,
        std::uintptr_t cuda_stream = 0) const override;

private:
    CudaArray<float> rest_angles_;
    float stiffness_ = 1.0e-3f;

    mutable CudaArray<float> energy_buffer_;
};

}  // namespace constraint
}  // namespace chysx
