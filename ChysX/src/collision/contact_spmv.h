// SPDX-License-Identifier: Apache-2.0
//
// chysx::collision::ContactSpMVOp
//
// POD describing a *dynamic* additive contribution to the linear
// system the PCG solver iterates against.  The static elastic
// stiffness + inertia matrix lives in a `chysx::sparse::BlockCSR3`
// whose topology is fixed across the simulation; collisions are
// applied as a separate COO-style operator so contact churn between
// frames never invalidates that topology.
//
// Algebraically, the PCG sees the augmented system
//
//     ( A_csr  +  C_op )  x  =  b
//
// where `A_csr` is the BlockCSR3 (elastic + inertia + pin) and
// `C_op` is the FULL contact penalty Hessian
//
//     C_op[i][j]  =  sum_{c : pair c contains both i and j}
//                       k * w_{c,i} * w_{c,j} * (n_c n_c^T).
//
// To match cuda-cloth's split between
// `KernelComputeCollisionHessianAndForce_4` (DcdUtils.cu, diagonal +
// force baked per-step) and `CollisionSpmv_4` (SolverUtils.cu, off-
// diagonal-only sidecar called inside the PCG SpMV) the operator is
// applied in two pieces:
//
//   1. Once per step, BEFORE PCG starts, `bake_contact_diag` adds the
//      contact diagonal blocks
//
//          C_diag[i] += k * w_{c,i}^2 * (n_c n_c^T)
//
//      directly into the BlockCSR3's `diag` array, so the block-Jacobi
//      preconditioner the PCG iterates with becomes contact-aware
//      (otherwise stiff penalty contacts converge unacceptably slowly
//      and leave residual penetration).
//
//   2. Inside every PCG SpMV, `apply_contact_spmv` adds the OFF-
//      diagonal cross-particle blocks (i != j) into `y`.  The diagonal
//      part is intentionally skipped here -- it already lives in
//      `H_.diag` and is consumed by the regular CSR SpMV.
//
// Layout matches cuda-cloth's `KernelComputeCollisionHessianAndForce_4`
// / `CollisionSpmv_4`:
//
//   pairs[c]     = (i0, i1, i2, i3)              four particle indices
//   weights[c]   = (w0, w1, w2, w3, n.x, n.y, n.z, depth)
//   count_dev    : single-int counter the detector populated this pass

#pragma once

#include <cstdint>

#include "../math/matrix.cuh"
#include "../math/vec.cuh"
#include "self_collision.h"  // for ContactWeights

namespace chysx {
namespace collision {

struct ContactSpMVOp {
    const math::Vec4i*       pairs        = nullptr;
    const ContactWeights*    weights      = nullptr;
    const int*               count_dev    = nullptr;  // single-int counter
    int                      max_contacts = 0;
    float                    stiffness    = 0.0f;

    bool active() const noexcept {
        return max_contacts > 0 && stiffness > 0.0f && pairs != nullptr;
    }
};

// Bake `alpha * C_diag` into `diag_blocks[]` once per step.
//
// For every active contact (idx < *count_dev, clamped at max_contacts)
// adds `alpha * stiffness * w_i^2 * (n n^T)` to `diag_blocks[pair_i]`
// for each of the four particles in the contact.
//
//   diag_blocks : pointer to a `chysx::sparse::BlockCSR3::diag` array
//                 (length n_particles, row-major Mat3f blocks).
//   alpha       : extra multiplier (1.0f for normal use; -1.0f could
//                 in principle subtract previously-baked terms).
//
// Atomic-safe with the elastic Hessian scatter kernels because they
// use the same per-element `atomicAdd<float>` primitive on
// `Mat3f::data[k]`.
void bake_contact_diag(math::Mat3f* diag_blocks,
                       int n_particles,
                       const ContactSpMVOp& op,
                       float alpha,
                       std::uintptr_t cuda_stream);

// y += alpha * C_off * x      (off-diagonal-only contact Hessian)
//
// For every active contact and every (i, j) with i != j:
//
//     y[pair_i] += alpha * stiffness * w_i * w_j * (n n^T) * x[pair_j]
//
// Diagonal terms are intentionally skipped -- the caller is expected
// to have baked them into the BlockCSR3's `diag` (see
// `bake_contact_diag`) so they're already covered by the regular CSR
// SpMV.  Mirrors cuda-cloth's `CollisionSpmv_4` (SolverUtils.cu) which
// also has `if (i == j) continue;` inside the inner loop.
//
// `cuda_stream` is a `cudaStream_t` cast to `uintptr_t`.  The kernel
// is launched on that stream so it captures cleanly into a
// `cudaGraph_t` recorded around the surrounding PCG iteration.
void apply_contact_spmv(const ContactSpMVOp& op,
                        const math::Vec3f* x,
                        math::Vec3f* y,
                        int n_particles,
                        float alpha,
                        std::uintptr_t cuda_stream);

}  // namespace collision
}  // namespace chysx
