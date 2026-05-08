// SPDX-License-Identifier: Apache-2.0
//
// chysx::constraint::TriangleStretchConstraint
//
// Three-particle ("N=3") membrane stretch element following Baraff &
// Witkin's 1998 cloth model.  For each triangle (i_0, i_1, i_2) we
// store
//
//   * `Dm_inv_[t]`   : inverse of the rest-shape edge matrix in the
//                      triangle's local 2-D UV frame (Mat2f).
//   * `areas_[t]`    : the rest-configuration triangle area used as
//                      a per-element stiffness weight.
//
// Energy
// ------
//   F   = D_s D_m^{-1}                       (3x2 deformation gradient)
//   w_u = F[:,0], w_v = F[:,1]
//   c_u = |w_u| - 1                          (in-plane stretch in U)
//   c_v = |w_v| - 1                          (in-plane stretch in V)
//   E_t = (k * area / 2) * (c_u^2 + c_v^2)
//
// Gradient
// --------
//   ∂c_u/∂x_a = dw_u[a] * (w_u / |w_u|)       (a ∈ {0,1,2})
//   ∂c_v/∂x_a = dw_v[a] * (w_v / |w_v|)
//   g_a       = k * area * (c_u dc_u/dx_a + c_v dc_v/dx_a)
//
// Hessian (PSD-filtered, Baraff-Witkin)
// -------------------------------------
//   H_{a,b}   = k * area * ( dcudx_a ⊗ dcudx_b + dcvdx_a ⊗ dcvdx_b )
//   if c_u > 0 (stretched along u):
//       H_{a,b} += k * area * c_u * |w_u|^{-1} * dw_u[a] * dw_u[b] *
//                  ( I - ŵ_u ŵ_u^T )
//   if c_v > 0:  same with w_v.
//
// The compressed-side branches are skipped, leaving a positive
// semi-definite block per triangle — same trick the SpringConstraint
// uses for its rest-length term.
//
// Each triangle contributes 9 = 3x3 block triplets to the global
// Hessian: every (a, b) pair of its three vertices.

#pragma once

#include <cstdint>

#include "../math/matrix.cuh"
#include "../math/vec.cuh"
#include "../memory/cuda_array.h"
#include "../memory/device_span.h"
#include "constraint_n.h"

namespace chysx {
namespace constraint {

class TriangleStretchConstraint : public ConstraintN<3> {
public:
    TriangleStretchConstraint() = default;
    ~TriangleStretchConstraint() override = default;

    // Upload `n` triangles and compute their reference shapes from
    // `positions` (the current device-side particle positions, treated
    // as the rest configuration).  Replaces any previously installed
    // triangles.
    //
    // For each triangle (a, b, c) the reference frame is built by
    // flattening the 3-D vertices into a 2-D UV system:
    //
    //     UV(a) = (0, 0)
    //     UV(b) = (|x_b - x_a|, 0)
    //     UV(c) = ( (x_c - x_a) . (x_b - x_a)/|x_b - x_a| ,
    //               sqrt(|x_c - x_a|^2 - that^2) )
    //
    // and `Dm` is the resulting 2x2 edge matrix [UV(b)-UV(a) | UV(c)-UV(a)].
    //
    // `material_rotation_rad` rotates the (u, v) material axes used by
    // the constraint by that angle before storing `Dm_inv`.  At the
    // default (theta = 0) the constraint pins the lengths of the U
    // and V edges, i.e. classical Baraff-Witkin stretch.  Setting
    // theta = pi/4 reuses the very same energy / gradient / Hessian
    // kernels but along the diagonal directions, which is exactly the
    // BW98 shear constraint as implemented in cuda-cloth's
    // `KernelComputeStretchShearForceAndHessianFast` (the kernel is
    // identical except for the `Dm_inv * R^-1` substitution applied
    // here once at setup).  Two instances of this class — one with
    // theta=0 and one with theta=pi/4 — therefore cover both
    // in-plane membrane modes.
    void set_triangles_from_positions(
        const math::Vec3i* host_triangles,
        int n,
        DeviceSpan<math::Vec3f> positions,
        std::uintptr_t cuda_stream = 0,
        float material_rotation_rad = 0.0f);

    // Stiffness shared by every triangle [N/m^2].  Per-element weight
    // `area * stiffness` is applied inside the kernels.
    void set_stiffness(float k) noexcept { stiffness_ = k; }
    float stiffness() const noexcept { return stiffness_; }

    // ---- buffer access ----------------------------------------------

    CudaArray<math::Mat2f>& Dm_inv() noexcept { return Dm_inv_; }
    const CudaArray<math::Mat2f>& Dm_inv() const noexcept { return Dm_inv_; }
    CudaArray<float>& areas() noexcept { return areas_; }
    const CudaArray<float>& areas() const noexcept { return areas_; }

    // ---- Constraint overrides ---------------------------------------

    float compute_energy(
        DeviceSpan<math::Vec3f> positions,
        std::uintptr_t cuda_stream = 0) const override;

    void accumulate_gradient(
        DeviceSpan<math::Vec3f> positions,
        DeviceSpan<math::Vec3f> out_grad,
        std::uintptr_t cuda_stream = 0) const override;

    // Each triangle contributes 9 = 3x3 local Hessian blocks.  Three
    // of them are diagonal (a == b) and land on `A.diag`; the other
    // six are off-diagonal and land on `A.values` at slots computed
    // by `bind_hessian_layout`.
    void accumulate_hessian(
        DeviceSpan<math::Vec3f> positions,
        sparse::BlockCSR3& A,
        std::uintptr_t cuda_stream = 0) const override;

private:
    CudaArray<math::Mat2f> Dm_inv_;
    CudaArray<float>       areas_;
    float                  stiffness_ = 1.0e3f;

    mutable CudaArray<float> energy_buffer_;
};

}  // namespace constraint
}  // namespace chysx
