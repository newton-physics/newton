// SPDX-License-Identifier: Apache-2.0
//
// chysx::constraint::PinConstraint
//
// One-particle ("N=1") penalty constraint that pulls particle i_c
// toward a fixed world-space target t_c with stiffness k:
//
//     E_c     = (k / 2) * | x_{i_c} - t_c |^2
//     g_c(i_c) = k * (x_{i_c} - t_c)
//     H_c(i_c, i_c) = k * I_3
//
// "Hard" Dirichlet-style pinning is recovered by setting k very large
// (1e6 .. 1e8): the diagonal block dominates that particle's row of
// the linear system, so any CG correction along its axes is driven
// to zero.
//
// Storage
// -------
// `ConstraintN<1>` already owns one int per instance (the particle
// index i_c) in `indices_`.  PinConstraint adds a matching
// `targets_` array of `Vec3f`.

#pragma once

#include <cstdint>

#include "../math/matrix.cuh"
#include "../math/vec.cuh"
#include "../memory/cuda_array.h"
#include "../memory/device_span.h"
#include "constraint_n.h"

namespace chysx {
namespace constraint {

class PinConstraint : public ConstraintN<1> {
public:
    PinConstraint() = default;
    ~PinConstraint() override = default;

    // Upload `n` pin instances.  `host_indices[c]` is the global
    // particle index of pin c; `host_targets[c]` is its target world-
    // space position.  Both arrays must have length `n`.
    //
    // Replaces any previously installed pins.
    void set_pins(const int* host_indices,
                  const math::Vec3f* host_targets,
                  int n);

    // Stiffness shared by every pin.  For hard Dirichlet-style
    // constraints set this large (e.g. 1e6 .. 1e8); for an elastic
    // tether use whatever your material model needs.
    void set_stiffness(float k) noexcept { stiffness_ = k; }
    float stiffness() const noexcept { return stiffness_; }

    // ---- target buffer access ----------------------------------------

    CudaArray<math::Vec3f>& targets() noexcept { return targets_; }
    const CudaArray<math::Vec3f>& targets() const noexcept { return targets_; }

    // ---- Constraint overrides ----------------------------------------

    float compute_energy(
        DeviceSpan<math::Vec3f> positions,
        std::uintptr_t cuda_stream = 0) const override;

    void accumulate_gradient(
        DeviceSpan<math::Vec3f> positions,
        DeviceSpan<math::Vec3f> out_grad,
        std::uintptr_t cuda_stream = 0) const override;

    // Each instance contributes exactly one diagonal 3x3 block
    // (i_c, i_c, k * I_3); `bind_hessian_layout` therefore produces
    // a slot table of N = size() entries, all encoded as -i_c - 1.
    void accumulate_hessian(
        DeviceSpan<math::Vec3f> positions,
        sparse::BlockCSR3& A,
        std::uintptr_t cuda_stream = 0) const override;

private:
    CudaArray<math::Vec3f> targets_;
    float stiffness_ = 1.0e6f;

    // Single-float scratch buffer for the device-side energy
    // reduction.  mutable so `compute_energy` can keep its const
    // signature while still owning a reusable accumulator.
    mutable CudaArray<float> energy_buffer_;
};

}  // namespace constraint
}  // namespace chysx
