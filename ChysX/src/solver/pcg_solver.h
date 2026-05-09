// SPDX-License-Identifier: Apache-2.0
//
// chysx::solver::PCGSolver
//
// Preconditioned Conjugate Gradient solver for the linear system
//
//     A x = b
//
// where `A` is a `chysx::sparse::BlockCSR3` (block-CSR matrix with
// 3x3 single-precision non-zeros) and `x`, `b` are arrays of
// `math::Vec3f` (one Vec3 per particle / block row).
//
// Preconditioner
// --------------
// The solver uses a block-Jacobi preconditioner: M = block_diag(A) and
// M^-1 is the matrix that contains the inverse 3x3 of every diagonal
// block of A.  Because `BlockCSR3` already stores the diagonal as a
// dedicated `A.diag` array (separate from the off-diagonal CSR), no
// extraction pass is needed — the solver just inverts each `A.diag[i]`
// at the start of every solve.
//
// Algorithm
// ---------
// Standard right-preconditioned CG.  The initial guess `x_0` is taken
// from the contents of the caller's `x` buffer on entry, so callers
// can warm-start by leaving the previous solve's solution in place.
// (Pass an explicitly zeroed `x` for the cold-start variant.)
//
//     r_0 = b - A x_0
//     z_0 = M^-1 r_0
//     p_0 = z_0
//     rho_0 = <r_0, z_0>
//     for k = 0, 1, 2, ...
//         q_k     = A p_k
//         alpha_k = rho_k / <p_k, q_k>
//         x_{k+1} = x_k + alpha_k p_k
//         r_{k+1} = r_k - alpha_k q_k
//         z_{k+1} = M^-1 r_{k+1}
//         rho_{k+1} = <r_{k+1}, z_{k+1}>
//         beta_k  = rho_{k+1} / rho_k
//         p_{k+1} = z_{k+1} + beta_k p_k
//
// All scalar coefficients (alpha, beta, rho) live on the device; the
// solver never copies them back to the host inside the iteration so a
// full solve runs without host-device synchronisation.
//
// Stream usage
// ------------
// Every kernel the solve issues is dispatched onto the
// caller-supplied `cuda_stream`.  No internal stream is created and
// no synchronisation is performed against any other stream.  Callers
// who want CUDA Graph capture should wrap `solve()` in their own
// `wp.ScopedCapture` / `cudaStreamBeginCapture` block — every PCG
// kernel will be recorded as a node in that outer graph automatically.

#pragma once

#include <cstdint>

#include "../collision/contact_spmv.h"
#include "../math/matrix.cuh"
#include "../math/vec.cuh"
#include "../memory/cuda_array.h"
#include "../memory/device_span.h"
#include "../sparse/block_csr.h"

// Forward-declare the CUDA Runtime stream type so this header does
// not need <cuda_runtime.h>; the .cu file casts std::uintptr_t to
// `cudaStream_t` (which is `CUstream_st*`) at call sites.
struct CUstream_st;

namespace chysx {
namespace solver {

struct PCGParams {
    int max_iterations = 100;
};

class PCGSolver {
public:
    PCGSolver() = default;

    PCGSolver(const PCGSolver&) = delete;
    PCGSolver& operator=(const PCGSolver&) = delete;

    PCGSolver(PCGSolver&& other) noexcept = default;
    PCGSolver& operator=(PCGSolver&& other) noexcept = default;

    ~PCGSolver() = default;

    // Allocate (or reuse) workspace for `num_block_rows` particles.
    // Idempotent: if the size already matches no allocation happens.
    void initialize(int num_block_rows);

    // Solve A x = b in place.
    //
    //   A : block-CSR matrix.  `A.diag` (per-particle 3x3 diagonal
    //       block) is read every solve and inverted on the fly to
    //       produce the block-Jacobi preconditioner.
    //   b : right-hand side, length = A.num_block_rows().
    //   x : solution buffer, length = A.num_block_rows().  Used as
    //       the initial guess on entry and overwritten with the
    //       solution on exit.  Pass a zeroed buffer for cold-start
    //       behaviour, or leave the previous solve's result in place
    //       to warm-start.
    //
    // `contact` (optional) attaches a dynamic COO-style additive
    // operator to the system: every `A * x` evaluation inside the
    // iteration becomes `(A + C) * x`, where C reads the contact
    // pairs/weights from `*contact`.  The static CSR topology of A
    // is therefore never modified by collision -- contact churn
    // between frames is absorbed entirely by the COO sidecar.  Pass
    // `nullptr` (default) to recover the plain `A x = b` solver.
    //
    // Returns the number of iterations actually performed.
    int solve(const sparse::BlockCSR3& A,
              DeviceSpan<math::Vec3f> b,
              DeviceSpan<math::Vec3f> x,
              const PCGParams& params = PCGParams{},
              std::uintptr_t cuda_stream = 0,
              const collision::ContactSpMVOp* contact = nullptr);

    // Last solve's preconditioner-weighted residual <r, z> from the
    // final iteration, copied to host on demand.  Useful for cheap
    // convergence checks outside the main loop.
    float last_residual();

private:
    int num_block_rows_ = 0;

    CudaArray<math::Vec3f> r_;
    CudaArray<math::Vec3f> p_;
    CudaArray<math::Vec3f> z_;
    CudaArray<math::Vec3f> Ap_;

    CudaArray<math::Mat3f> M_inv_;        // block-Jacobi preconditioner

    // Three scalar reductions live in a single 4-element buffer so we
    // can read them out with one cudaMemcpy when needed:
    //   coeff_[0] = <r, z>      (rho_k)
    //   coeff_[1] = <p, A p>    (sigma_k)
    //   coeff_[2] = <r, z>_new  (rho_{k+1})
    //   coeff_[3] = scratch
    CudaArray<float> coeff_;
};

}  // namespace solver
}  // namespace chysx
