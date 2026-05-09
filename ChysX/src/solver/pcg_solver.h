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
// CUDA Graph mode
// ---------------
// At cloth scale the iteration body is dominated by per-kernel launch
// overhead (50 iterations × 8 small kernels = ~400 launches per solve;
// on Windows / WDDM that's roughly 1.5 ms of pure dispatch on top of
// the actual compute).  When `graph_enabled()` is true (default), the
// solver captures the entire `solve()` into a single `cudaGraphExec_t`
// the first time it sees a given (matrix layout, b, x, max_iter)
// configuration and replays it on every subsequent call with one
// `cudaGraphLaunch`.  The captured graph stays valid as long as the
// underlying device pointers and dimensions are unchanged; whenever
// any of them shifts (e.g. topology rebuild resizes A.values, or
// particle count changes) the solver transparently re-captures.
//
// Capture is done on an internally-owned non-default stream because
// stream capture is not allowed on stream 0; events are used to
// rendezvous with whatever stream the caller passes to `solve()`.

#pragma once

#include <cstdint>

#include "../collision/contact_spmv.h"
#include "../math/matrix.cuh"
#include "../math/vec.cuh"
#include "../memory/cuda_array.h"
#include "../memory/device_span.h"
#include "../sparse/block_csr.h"

// Forward-declare CUDA Runtime types so the header doesn't need
// <cuda_runtime.h>.  These are all opaque pointer typedefs in the
// real header, so storing them as `void*` then casting in the .cu
// keeps the public interface CUDA-toolkit-free.
struct CUstream_st;
struct CUevent_st;
struct CUgraphExec_st;

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

    // Move-only.  Owned CUDA resources (the capture stream, two
    // events, and the executable graph) transfer to the destination;
    // the source becomes empty.
    PCGSolver(PCGSolver&& other) noexcept;
    PCGSolver& operator=(PCGSolver&& other) noexcept;

    ~PCGSolver();

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

    // Toggle CUDA Graph capture/replay.  Default is ON.  Disabling
    // also drops any cached graph; subsequent solves run kernel-by-
    // kernel directly on the caller's stream.
    void set_graph_enabled(bool enabled);
    bool graph_enabled() const noexcept { return graph_enabled_; }

    // Drop the cached graph executable (if any).  The next solve()
    // re-captures from scratch.  Useful when the caller knows the
    // device pointers it passes will change but doesn't want to
    // toggle the graph flag.
    void invalidate_graph();

private:
    // Submit the entire solve (prologue + max_iter loop) to `stream`.
    // Used both as the direct (non-graph) execution path and as the
    // capture body for graph mode.
    void submit_solve_ops_(struct CUstream_st* stream,
                           const sparse::BlockCSR3& A,
                           DeviceSpan<math::Vec3f> b,
                           DeviceSpan<math::Vec3f> x,
                           const PCGParams& params,
                           const collision::ContactSpMVOp* contact);

    // Capture `submit_solve_ops_` into a fresh `graph_exec_`.  The
    // previous executable (if any) is destroyed first.  Caller is
    // responsible for ensuring no work is in flight on
    // `graph_stream_` at the time of the call.
    void capture_graph_(const sparse::BlockCSR3& A,
                        DeviceSpan<math::Vec3f> b,
                        DeviceSpan<math::Vec3f> x,
                        const PCGParams& params,
                        const collision::ContactSpMVOp* contact);

    // Lazy-create graph stream + events.  Called the first time graph
    // mode is actually used.
    void ensure_graph_resources_();

    // Cache key — the graph is only valid while every one of these
    // matches the captured configuration.  Contact-side fields have
    // to be part of the key because their values (pointers, ints,
    // floats) are baked into the captured kernel launches; mutating
    // them after capture would NOT be picked up.
    struct GraphKey {
        int n            = 0;
        int max_iter     = 0;
        const void* A_diag        = nullptr;
        const void* A_values      = nullptr;
        const void* A_row_offsets = nullptr;
        const void* A_col_indices = nullptr;
        const void* b             = nullptr;
        void*       x             = nullptr;
        // Contact (COO sidecar) — ignored when no contact op is
        // attached to the solve.
        const void* C_pairs      = nullptr;
        const void* C_weights    = nullptr;
        const void* C_count      = nullptr;
        int         C_max        = 0;
        float       C_stiffness  = 0.0f;
        bool operator==(const GraphKey& o) const noexcept {
            return n == o.n && max_iter == o.max_iter &&
                   A_diag == o.A_diag && A_values == o.A_values &&
                   A_row_offsets == o.A_row_offsets &&
                   A_col_indices == o.A_col_indices &&
                   b == o.b && x == o.x &&
                   C_pairs == o.C_pairs && C_weights == o.C_weights &&
                   C_count == o.C_count && C_max == o.C_max &&
                   C_stiffness == o.C_stiffness;
        }
        bool operator!=(const GraphKey& o) const noexcept { return !(*this == o); }
    };

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

    // ---- CUDA Graph state ------------------------------------------
    bool graph_enabled_ = true;

    // Owned resources, lazily created on first graph-mode solve.
    // Stored as CUDA Runtime opaque types to keep this header free of
    // <cuda_runtime.h>.
    struct CUstream_st*     graph_stream_ = nullptr;
    struct CUevent_st*      pre_event_    = nullptr;
    struct CUevent_st*      post_event_   = nullptr;
    struct CUgraphExec_st*  graph_exec_   = nullptr;

    GraphKey cached_key_{};
};

}  // namespace solver
}  // namespace chysx
