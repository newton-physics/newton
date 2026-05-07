// SPDX-License-Identifier: Apache-2.0
//
// chysx::constraint::SpringConstraint
//
// Two-particle ("N=2") Hookean stretch spring with a Baraff–Witkin
// PSD-projected Hessian.  For each spring instance e = (a, b) with
// rest length L:
//
//     d   = x_a - x_b
//     l   = |d|
//     d̂   = d / l
//     E_e = (k / 2) * (l - L)^2
//
//     ∂E/∂x_a = +k (l - L) d̂          ∂E/∂x_b = -k (l - L) d̂
//
//     H_blk = (k - a) I + a (d̂ ⊗ d̂)        if l ≥ L  (stretched)
//           = a (d̂ ⊗ d̂)                    if l <  L  (compressed)
//     where a = k L / l
//
//     ∂²E/∂x_a² = ∂²E/∂x_b² = +H_blk
//     ∂²E/∂x_a∂x_b = ∂²E/∂x_b∂x_a = -H_blk
//
// The compressed-branch projection drops the `(k - a) I` term that
// becomes negative-definite when a > k, leaving a rank-1 PSD block.
// This is the standard Baraff–Witkin filtering (1998) and the same
// trick used by cuda-cloth's SpringMass.
//
// Storage
// -------
// `ConstraintN<2>::indices_` holds the per-spring vertex pair (Vec2i).
// `rest_lengths_` is a parallel float buffer (one rest length per
// spring).  All springs share a single stiffness `k` for now; if you
// need per-spring stiffness later, swap `stiffness_` for a CudaArray.

#pragma once

#include <cstdint>

#include "../math/matrix.cuh"
#include "../math/vec.cuh"
#include "../memory/cuda_array.h"
#include "../memory/device_span.h"
#include "constraint_n.h"

namespace chysx {
namespace constraint {

class SpringConstraint : public ConstraintN<2> {
public:
    SpringConstraint() = default;
    ~SpringConstraint() override = default;

    // Upload `n` springs.  `host_edges[c]` is the (i, j) particle
    // pair of spring c; `host_rest_lengths[c]` is its rest length L.
    // Both arrays must have length n.
    //
    // Replaces any previously installed springs.
    void set_springs(const math::Vec2i* host_edges,
                     const float* host_rest_lengths,
                     int n);

    // Convenience: install springs whose rest length is taken from
    // the *current* device-side positions (i.e. the configuration is
    // already at rest).  Useful when `edges` were pulled from a
    // TriangleMesh.
    void set_springs_from_positions(
        const math::Vec2i* host_edges,
        int n,
        DeviceSpan<math::Vec3f> positions,
        std::uintptr_t cuda_stream = 0);

    // Stiffness shared by every spring [N/m].
    void set_stiffness(float k) noexcept { stiffness_ = k; }
    float stiffness() const noexcept { return stiffness_; }

    // ---- buffer access ----------------------------------------------

    CudaArray<float>& rest_lengths() noexcept { return rest_lengths_; }
    const CudaArray<float>& rest_lengths() const noexcept { return rest_lengths_; }

    // ---- Constraint overrides ---------------------------------------

    float compute_energy(
        DeviceSpan<math::Vec3f> positions,
        std::uintptr_t cuda_stream = 0) const override;

    void accumulate_gradient(
        DeviceSpan<math::Vec3f> positions,
        DeviceSpan<math::Vec3f> out_grad,
        std::uintptr_t cuda_stream = 0) const override;

    // Each spring contributes 4 = 2x2 local blocks:
    //   (a,a,+H) (a,b,-H) (b,a,-H) (b,b,+H).
    // The (a, a) and (b, b) blocks land on `A.diag`; the (a, b) and
    // (b, a) blocks land on `A.values` at the slots computed by
    // `bind_hessian_layout`.
    void accumulate_hessian(
        DeviceSpan<math::Vec3f> positions,
        sparse::BlockCSR3& A,
        std::uintptr_t cuda_stream = 0) const override;

private:
    CudaArray<float> rest_lengths_;
    float stiffness_ = 1.0e3f;

    mutable CudaArray<float> energy_buffer_;  // 1-element scratch
};

}  // namespace constraint
}  // namespace chysx
