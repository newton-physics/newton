// SPDX-License-Identifier: Apache-2.0
//
// chysx::constraint::SelfCollisionConstraint
//
// Penalty self-contact constraint that consumes the output of a
// `chysx::collision::SelfCollisionDetector`.  Contributions split into
// two halves:
//
//   * Gradient -> RHS scatter
//
//       grad E_i  =  -k * w_i * depth * n
//
//     accumulated into the global gradient buffer the cloth simulator
//     hands to `assemble_rhs_kernel`.  This is the force-on-vertex
//     analogue cuda-cloth's `KernelComputeCollisionHessianAndForce_4`
//     adds to its `f` buffer (with sign flipped, since chysx stores
//     `+grad E` while cuda-cloth stores `+force = -grad E`).
//
//   * Hessian -> COO sidecar (NOT into BlockCSR3)
//
//       H_ii      =  k * w_i^2 * (n n^T)         (diagonal-only,
//                                                  matches the _4 path)
//
//     Returned as a `chysx::collision::ContactSpMVOp` POD that the
//     PCG solver consumes during its iteration: every `A * x` becomes
//     `A * x + C * x` where C reads the contact pairs/weights without
//     ever touching `BlockCSR3`'s topology.  The static elastic CSR
//     therefore stays valid frame-to-frame even though the contact
//     set changes every detection pass.

#pragma once

#include <cstdint>

#include "../collision/contact_spmv.h"
#include "../collision/self_collision.h"
#include "../math/vec.cuh"
#include "../memory/device_span.h"

namespace chysx {
namespace constraint {

class SelfCollisionConstraint {
public:
    SelfCollisionConstraint() = default;

    SelfCollisionConstraint(const SelfCollisionConstraint&) = delete;
    SelfCollisionConstraint& operator=(const SelfCollisionConstraint&) = delete;
    SelfCollisionConstraint(SelfCollisionConstraint&&) noexcept = default;
    SelfCollisionConstraint& operator=(SelfCollisionConstraint&&) noexcept =
        default;

    void set_stiffness(float k) noexcept { stiffness_ = k; }
    float stiffness() const noexcept { return stiffness_; }

    // Build a POD operator describing this constraint's Hessian
    // contribution as `(pairs, weights, count_ptr, max_contacts, k)`.
    // Returned by value; the underlying device buffers belong to
    // `detector` and must outlive any consumer of the operator.
    //
    // When `stiffness_ == 0` the returned op is `active() == false`
    // and the PCG can skip it.
    collision::ContactSpMVOp make_spmv_op(
        const collision::SelfCollisionDetector& detector) const noexcept;

    // Scatter `+ grad E_collision` into `out_grad` (one Vec3f per
    // global particle).  Reads the dynamic contact list from
    // `detector` (the device-side counter governs the in-kernel loop
    // bound, so this is correct with no host->device sync).
    void accumulate_gradient(
        const collision::SelfCollisionDetector& detector,
        DeviceSpan<math::Vec3f> out_grad,
        std::uintptr_t cuda_stream = 0) const;

private:
    float stiffness_ = 0.0f;
};

}  // namespace constraint
}  // namespace chysx
